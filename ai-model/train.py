import os
import time
import subprocess
import queue
import torch
from threading import Thread
from collections import deque
from torchvision.transforms import v2
from tqdm import tqdm

from geometry_dash_gym.envs import GeometryDashEnv
from tcp import gdclient
from model import DQNModel
from agent import Agent
import config

# frame_queue gets piped all frame data all the time. frame_buffer is built from frame_queue when we need to see the state
frame_queue = queue.Queue(maxsize=config.FRAME_STACK_SIZE * 2)
frame_buffer = deque(maxlen=config.FRAME_STACK_SIZE)

os.makedirs("checkpoints", exist_ok=True)

def start_geometry_dash():
    subprocess.Popen(["geode", "run"])
    print("Waiting 5s for Geometry Dash…")
    time.sleep(5)

def listen_for_frame_buffer():
    try:
        while True:
            frame = gdclient.receive_frame()
            if frame is None:
                print("Connection lost")
                break

            tensor = torch.from_numpy(frame).permute(2, 0, 1) # HWC → CHW
            tensor = tensor.flip(-2) # vertical flip

            # drop if full, then enqueue
            if frame_queue.full():
                _ = frame_queue.get_nowait()
            frame_queue.put(tensor)

    except Exception as e:
        print("Frame listener error:", e)
    finally:
        gdclient.close()

def build_state(transform):
    """Block until we have FRAME_STACK_SIZE frames, then stack them."""
    while len(frame_buffer) < config.FRAME_STACK_SIZE:
        try:
            f = frame_queue.get(timeout=1.0)
        except queue.Empty:
            raise RuntimeError("Timed out waiting for initial frames")
        frame_buffer.append(f)

    # for new states, just pull one more frame:
    try:
        new = frame_queue.get(timeout=1.0)
        frame_buffer.append(new)
    except queue.Empty:
        raise RuntimeError("Timed out waiting for next frame")

    processed = [transform(f).unsqueeze(0) for f in frame_buffer]
    # -> [T, C, H, W], then add batch dim -> [1, T, C, H, W]
    return torch.cat(processed, dim=0).unsqueeze(0)

def train(num_episodes=1000, max_steps=1000, resume=False):
    env   = GeometryDashEnv()
    model = DQNModel()
    agent = Agent(model)
    start_ep = 0

    # resume
    if resume and os.path.exists("checkpoints/latest.pt"):
        cp = torch.load("checkpoints/latest.pt")
        agent.model.load_state_dict(cp["model_state"])
        agent.optimizer.load_state_dict(cp["optimizer_state"])
        start_ep = cp["episode"] + 1
        print(f"Resumed at episode {start_ep}")

    transform = v2.Compose([
        v2.Resize((1270, 720)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # TODO: Fix waiting hack
    print("Start the level! You have 3 seconds before training starts")
    time.sleep(3)

    for ep in range(start_ep, num_episodes):
        env.reset()
        total_r = 0

        # initial fill of buffer
        state = build_state(transform)

        for step in tqdm(range(max_steps), desc=f"Ep{ep+1}"):
            action = agent.act(state)
            _, reward, done, _ = env.step(action)
            total_r += reward

            next_state = build_state(transform)
            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            if done:
                break

        print(f"Ep {ep+1} → reward {total_r:.1f}")

        if (ep + 1) % config.SAVE_EPOCH == 0:
            torch.save({
                "episode": ep,
                "model_state": agent.model.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
            }, "checkpoints/latest.pt")
            print(f"Saved checkpoint @ ep {ep+1}")

    env.close()

if __name__ == "__main__":
    # start_geometry_dash()
    gdclient.connect()
    Thread(target=listen_for_frame_buffer, daemon=True).start()
    train()
