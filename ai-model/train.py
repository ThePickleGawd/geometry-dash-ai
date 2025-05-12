import numpy as np
import gymnasium as gym
import subprocess
import time
import torch
from torchvision.transforms import v2
import cv2
from threading import Thread
import queue
from tqdm import tqdm
import os

from geometry_dash_gym.envs import GeometryDashEnv
from tcp import gdclient
from model import DQNModel
from agent import Agent
import config

# Save last few frames
frame_queue = queue.Queue(maxsize=config.FRAME_STACK_SIZE)
os.makedirs("checkpoints", exist_ok=True)

def start_geometry_dash():
    # Open geometry dash
    subprocess.Popen(["geode", "run"])
    print("Waiting 5 seconds for Geometry Dash to load...")
    time.sleep(5)

def listen_for_frame_buffer():
    while True:
        frame = gdclient.receive_frame()
        if frame is None:
            print("Connection lost")
            break

        # Convert to tensor and flip vertically
        tensor = torch.from_numpy(frame).permute(2, 0, 1)  # HWC → CHW
        tensor = tensor.flip(-2)  # vertical flip

        if frame_queue.full():
            frame_queue.get_nowait() # Remove oldest frame
        frame_queue.put(tensor)
        print("YO FRAME!")

    gdclient.close()

# Thread safe; will clear queue after fetching
def get_stacked_frames(frame_queue, transform, stack_size):
    """
    Waits until `stack_size` frames are in the queue, then grabs the latest ones,
    applies `transform`, and returns a tensor of shape [1, stack_size, C, H, W].
    """
    while frame_queue.qsize() < stack_size:
        time.sleep(0.01)

    # Drain the queue safely
    frames = []
    while not frame_queue.empty():
        try:
            frames.append(frame_queue.get_nowait())
        except queue.Empty:
            break

    # Use the last `stack_size` frames
    frames = frames[-stack_size:]

    # Transform and stack
    processed = [transform(f).unsqueeze(0) for f in frames]  # [1, C, H, W] each
    stacked = torch.cat(processed, dim=0).unsqueeze(0)       # [1, T, C, H, W]
    return stacked


def train(num_episodes=1000, max_steps_per_episode=1000, resume=False):
    os.makedirs("checkpoints", exist_ok=True)

    # Model and Environment
    env = GeometryDashEnv()
    model = DQNModel()
    agent = Agent(model)

    start_episode = 0

    # Resume logic
    if resume:
        checkpoint_path = "checkpoints/latest.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            agent.model.load_state_dict(checkpoint["model_state"])
            agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_episode = checkpoint["episode"] + 1
            print(f"Resumed from episode {start_episode}")
        else:
            print("No checkpoint found. Starting from scratch.")

    # Image transform
    transform = v2.Compose([
        v2.Resize((1270, 720)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    for episode in range(start_episode, num_episodes):
        if episode > start_episode:
            obs = env.reset()
            
        total_reward = 0

        for step in tqdm(range(max_steps_per_episode)):
            # Await get state and action
            state_t0 = get_stacked_frames(frame_queue, transform, config.FRAME_STACK_SIZE)
            action = agent.act(state=state_t0)

            # Step
            _, reward, done, info = env.step(action)
            total_reward += reward

            # Get newly observed state
            state_t1 = get_stacked_frames(frame_queue, transform, config.FRAME_STACK_SIZE) 

            # Remember and train
            agent.remember(state_t0, action, reward, state_t1, done)
            agent.train()

            if done:
                break

        print(f"Episode {episode+1} finished with reward {total_reward}")

        # Save checkpoint
        if (episode + 1) % config.SAVE_EPOCH == 0:
            torch.save({
                "episode": episode,
                "model_state": agent.model.state_dict(),
                "optimizer_state": agent.optimizer.state_dict()
            }, "checkpoints/latest.pt")
            print(f"Checkpoint saved at episode {episode+1}")

    env.close()

if __name__ == "__main__":
    start_geometry_dash()
    gdclient.connect()
    thread = Thread(target=listen_for_frame_buffer, daemon=True)
    thread.start()
    train()