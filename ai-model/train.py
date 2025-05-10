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
from tcp import gdclient
from model import DQNModel
from agent import Agent

# Save last 5 frames
frame_queue = queue.Queue(maxsize=5)  

def start_geometry_dash():
    # Open geometry dash
    subprocess.Popen(["geode", "run"])
    print("Waiting 4 seconds for Geometry Dash to load...")
    time.sleep(4)

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

    gdclient.close()

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
            while frame_queue.empty():
                time.sleep(0.01) # TODO: This is a super hacky way, so someone can clean this up

            # Get game state (just the screen frame)
            frame = frame_queue.get()
            frame = transform(frame).unsqueeze(0)  # (1, C, H, W)

            # Get action
            action = agent.act(state=frame)

            # Do action
            print("going to step")
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print("done step")

            # Get new frame buffer
            with frame_queue.mutex:
                frame_queue.queue.clear()
            while frame_queue.empty():
                time.sleep(0.01) # TODO: This is a super hacky way, so someone can clean this up

            print("frame queue is fresh")
            frame_t1 = frame_queue.get()
            frame = transform(frame).unsqueeze(0)

            # Add to replay buffer
            agent.remember(frame, action, reward, frame_t1, False) # TODO: If done, then set done to True

            # Train
            agent.train()

            if done:
                break

        print(f"Episode {episode+1} finished with reward {total_reward}")

    env.close()

if __name__ == "__main__":
    start_geometry_dash()
    gdclient.connect()
    thread = Thread(target=listen_for_frame_buffer, daemon=True)
    thread.start()
    train()
