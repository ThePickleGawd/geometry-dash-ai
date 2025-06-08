import torch
from model import ExpertsFromDeeperDQNModelv2

# === Paths ===
ship_path = "../checkpoints/deeperdqnv2/ship_model.pt"
cube_path = "../checkpoints/deeperdqnv2/cube_model.pt"
output_path = "../checkpoints/deeperdqnv2/expert_merged.pt"

# === Load expert model with two full backbones
model = ExpertsFromDeeperDQNModelv2.load_experts(
    ship_path,
    cube_path,
    in_channels=1,      # adjust if RGB
    stack_size=4,
    num_actions=2
)

# === Save compatible training checkpoint format
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": None,
    "episode": 0,
    "time_alive": {},
    "total_reward": {},
    "epsilon": {0: 1.0},
}

torch.save(checkpoint, output_path)
print(f"✅ Saved merged expert model to {output_path}")
