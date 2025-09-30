import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.distributions import Categorical
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import config


@dataclass
class QwenTransition:
    frame: np.ndarray
    metadata: Dict[str, str]
    action_index: int
    reward: float
    done: bool
    log_prob: float


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    use_mps = os.environ.get("QWEN_USE_MPS", "0") == "1"
    if use_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class AgentQwenGRPO:
    """GRPO-capable policy head backed by a local Qwen model."""

    def __init__(self, model_path: Optional[str] = None, *, buffer_size: int = 2048, batch_size: int = 16):
        self.model_path = model_path or config.QWEN_MODEL_PATH
        self.device = _resolve_device()
        self.batch_size = batch_size
        self.replay_buffer: deque[QwenTransition] = deque(maxlen=buffer_size)

        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.adv_norm_eps = 1e-6
        self.max_grad_norm = 1.0

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=load_dtype,
        )

        # Make sure the tokenizer knows about our discrete action tokens.
        missing_tokens = [
            token for token in config.QWEN_ACTION_TOKENS if token not in self.tokenizer.get_vocab()
        ]
        if missing_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.model.train()

        self.action_tokens = list(config.QWEN_ACTION_TOKENS)
        self.action_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(t) for t in self.action_tokens],
            dtype=torch.long,
        ).to(self.device)
        self.env_actions = [0, 1]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        self.last_action_index = 0
        self.pending_transition: Optional[QwenTransition] = None
        self.epsilon = 0.0  # retained for compatibility with existing training loops
        self.death_replay_buffer: deque[QwenTransition] = deque(maxlen=config.NSTEP)

    def act(self, state: torch.Tensor, info: Optional[Any] = None) -> int:
        frame = self._state_to_frame(state)
        metadata = self._build_metadata(info)
        prompt_messages = self._messages_from_frame(frame, metadata)
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(prompt_messages)
        encoded = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        encoded = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in encoded.items()
        }

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits[:, -1, self.action_token_ids]

        dist = Categorical(logits=logits)
        action_index = dist.sample()
        log_prob = torch.log_softmax(logits, dim=-1)[0, action_index]

        self.pending_transition = QwenTransition(
            frame=frame,
            metadata=metadata.copy(),
            action_index=action_index.item(),
            reward=0.0,
            done=False,
            log_prob=log_prob.item(),
        )
        self.last_action_index = action_index.item()
        return self.env_actions[action_index.item()]

    def remember(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        *extra: Any,
    ) -> None:
        # Accept optional metadata like (is_ship, done) or just (done,)
        done = False
        if extra:
            done = bool(extra[-1])

        if self.pending_transition is None:
            return

        transition = self.pending_transition
        transition.reward = float(reward)
        transition.done = done
        self.replay_buffer.append(transition)
        self.pending_transition = None

    def train(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        if len(self.death_replay_buffer) > 0:
            sample_count = min(len(self.death_replay_buffer), max(1, self.batch_size // 4))
            batch.extend(random.sample(self.death_replay_buffer, sample_count))
            random.shuffle(batch)
        messages_batch = [self._messages_from_frame(sample.frame, sample.metadata) for sample in batch]
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        action_indices = torch.tensor(
            [sample.action_index for sample in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor([sample.reward for sample in batch], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(
            [sample.log_prob for sample in batch], dtype=torch.float32, device=self.device
        )

        encoded = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        encoded = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in encoded.items()
        }

        outputs = self.model(**encoded)
        logits = outputs.logits

        seq_ends = encoded["attention_mask"].sum(dim=1) - 1
        logits_last = logits[torch.arange(logits.size(0), device=self.device), seq_ends]
        action_logits = logits_last[:, self.action_token_ids]

        dist = Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(action_indices)
        ratios = torch.exp(new_log_probs - old_log_probs)

        advantages = rewards - rewards.mean()
        std = advantages.std(unbiased=False)
        if std > self.adv_norm_eps:
            advantages = advantages / (std + self.adv_norm_eps)

        clipped_ratios = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surrogate_1 = ratios * advantages
        surrogate_2 = clipped_ratios * advantages
        policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
        entropy_loss = -self.entropy_coef * dist.entropy().mean()

        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:  # Compatibility shim
        pass

    def on_death(self) -> None:
        if len(self.replay_buffer) < 5:
            return

        death_reward = self.replay_buffer[-1].reward
        for i in range(-5, 0):
            transition = self.replay_buffer[i]
            updated = QwenTransition(
                frame=transition.frame,
                metadata=transition.metadata,
                action_index=transition.action_index,
                reward=death_reward,
                done=transition.done,
                log_prob=transition.log_prob,
            )
            self.replay_buffer[i] = updated
            self.death_replay_buffer.append(updated)

    def _state_to_frame(self, state: torch.Tensor) -> np.ndarray:
        state_cpu = state.detach().to(torch.float32).cpu()
        latest_frame = state_cpu[0, -1]
        if latest_frame.shape[0] == 1:
            array = latest_frame.squeeze(0).numpy()
        else:
            array = latest_frame.mean(dim=0).numpy()
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
        return array

    def _build_metadata(self, info: Optional[Any]) -> Dict[str, str]:
        info_bits: Dict[str, str] = {}
        if isinstance(info, dict):
            info_bits.update({k: str(v) for k, v in info.items()})
        elif isinstance(info, bool):
            info_bits["ship_mode"] = "true" if info else "false"
        elif info is not None:
            info_bits["context"] = str(info)

        info_bits["prev_action"] = self.action_tokens[self.last_action_index]
        return info_bits

    def _messages_from_frame(self, frame: np.ndarray, metadata: Dict[str, str]) -> list[Dict[str, Any]]:
        image = self._frame_to_image(frame)
        metadata_text = ", ".join(f"{k}={v}" for k, v in sorted(metadata.items())) or "none"

        instruction = (
            "Observe the frame and decide the next move. "
            f"Metadata: {metadata_text}. "
            f"Reply with only {self.action_tokens[0]} to stay or {self.action_tokens[1]} to jump."
        )

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

    def _frame_to_image(self, frame: np.ndarray) -> Image.Image:
        image = Image.fromarray(frame, mode="L").convert("RGB")
        return image
