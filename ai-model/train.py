import numpy as np
import gymnasium as gym
import subprocess
import time
import torch
from torchvision.transforms import v2
import cv2
from threading import Thread
import queue

from geometry_dash_gym.envs import GeometryDashEnv
from tcp.server import GDServer
from model import DQNModel
from agent import Agent

# Save last 5 frames
frame_queue = queue.Queue(maxsize=5)  

def start_geometry_dash():
    # Open geometry dash
    subprocess.Popen(["geode", "run"])
    print("Waiting 5 seconds for Geometry Dash to load...")
    time.sleep(5)

def listen_for_frame_buffer():
    server = GDServer()
    server.start()

    while True:
        frame = server.receive_frame()
        if frame is None:
            print("Connection lost")
            break

        # Convert to tensor and flip vertically
        tensor = torch.from_numpy(frame).permute(2, 0, 1)  # HWC → CHW
        tensor = tensor.flip(-2)  # vertical flip

        if frame_queue.full():
            frame_queue.get_nowait() # Remove oldest frame
        frame_queue.put(tensor)

    server.close()

def train(num_episodes=1000, max_steps_per_episode=1000):
    # Model and Environment
    env = GeometryDashEnv()
    model = DQNModel()
    agent = Agent(model)

    # Image transform
    transform = v2.Compose([
        v2.Resize((1270, 720)), # HD Resolution
        v2.ToDtype(torch.float32, scale=True),
    ])

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            # Get game state (just the screen frame)
            frame = frame_queue.get()
            frame = transform(frame).unsqueeze(0)  # (1, C, H, W)

            # Get action
            action = agent.act(state=frame)

            # Do action
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Add to replay buffer
            # agent.remember(frame, action, reward, )
            
            # Train
            agent.train()

            if done:
                break

        print(f"Episode {episode+1} finished with reward {total_reward}")

    env.close()

if __name__ == "__main__":
    start_geometry_dash()
    thread = Thread(target=listen_for_frame_buffer, daemon=True)
    thread.start()
    train()
