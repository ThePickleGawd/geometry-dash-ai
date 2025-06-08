import os
import time
import subprocess
import queue
import torch
from threading import Thread
from collections import deque
from torchvision.transforms import v2
from tqdm import tqdm
import cv2
import random

from gym import GeometryDashEnv
from tcp import gdclient
from model import ExpertsModel, ExpertsFromDeeperDQNModelv2
from agent import AgentExperts
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

            tensor = torch.from_numpy(frame).unsqueeze(0)
            tensor = tensor.flip(-2) # vertical flip

            if frame_queue.full():
                _ = frame_queue.get_nowait()
            frame_queue.put(tensor)
            
    except Exception as e:
        print("Frame listener error:", e)
    finally:
        gdclient.close()

def build_state(transform):
    while len(frame_buffer) < config.FRAME_STACK_SIZE:
        try:
            f = frame_queue.get(timeout=1.0)
        except queue.Empty:
            raise RuntimeError("[build_state] Timed out waiting for initial frames")
        frame_buffer.append(f)

    try:
        new = frame_queue.get(timeout=1.0)
        frame_buffer.append(new)
    except queue.Empty:
        raise RuntimeError("[build_state] Timed out waiting for next frame")

    processed = [transform(f).unsqueeze(0) for f in frame_buffer]
    stacked = torch.cat(processed, dim=0).unsqueeze(0)
    return stacked

def train(num_episodes=50000, max_steps=5000):
    env = GeometryDashEnv()
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = ExpertsFromDeeperDQNModelv2(is_train=True).to(device)
    agent = AgentExperts(model)

    if os.path.exists("checkpoints/latest.pt"):
        cp = torch.load("checkpoints/latest.pt", weights_only=False)
        agent.model.load_state_dict(cp["model_state"])
        agent.target_model.load_state_dict(cp["model_state"])
        agent.epsilon = 0
    else:
        assert "Needs checkpoint"

    transform = v2.Compose([
        v2.Resize((128, 128)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    print("Start the level! You have 5 seconds before training starts")
    time.sleep(5)

    total_steps = 0

    for ep in range(0, num_episodes):
        r = random.random()
        if r < config.SHIP_SPAWN_PERCENTAGE:
            while True:
                pct = random.uniform(1, 100)
                if (30.01 < pct < 46.78) or (86 < pct < 90):
                    break
        elif r < config.SHIP_SPAWN_PERCENTAGE + config.START_SPAWN_PERCENTAGE:
            pct = config.SET_SPAWN
        else:
            while True:
                pct = random.uniform(1, 86)
                if pct < 30 or (pct > 46.79 and pct <= 86):
                    break

        rand = True
        env.reset(pct if rand else 0)
        start_time = time.time()
        total_r = 0
        state = build_state(transform)
        info = { "percent": pct }
        is_ship = False
        episode_ship_accs = []

        pbar = tqdm(range(max_steps), desc=f"Ep{ep+1}")
        for step in pbar:
            if info['percent'] < 86 and not (30 < info['percent'] < 46.79):
                is_ship = False
            else:
                is_ship = True

            action = agent.act(state, is_ship)
            _, reward, done, info = env.step(action, start_percent=pct)
            total_r += reward
            pbar.set_postfix(r=round(total_r, 2), lvl_percent=round(info["percent"], 1))

            if done:
                print(f"Died at step {step}.")
                agent.on_death()
                break

    env.close()

if __name__ == "__main__":
    start_geometry_dash()
    gdclient.connect()
    Thread(target=listen_for_frame_buffer, daemon=True).start()
    train()
