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

from gym import GeometryDashEnv
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

            tensor = torch.from_numpy(frame).unsqueeze(0)
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

def play(num_episodes=50000, max_steps=10000, resume=True):
    env   = GeometryDashEnv()
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = DQNModel().to(device)
    agent = Agent(model)
    agent.epsilon = 0
    start_ep = 0
    best_percent = 0    
    # resume
    if resume and os.path.exists("checkpoints/latest.pt"):
        cp = torch.load("checkpoints/latest.pt")
        agent.model.load_state_dict(cp["model_state"])
        agent.target_model.load_state_dict(cp["model_state"])
        agent.optimizer.load_state_dict(cp["optimizer_state"])
        start_ep = cp["episode"] + 1
        time_alive_per_ep = cp.get("time_alive", {})
        print(f"Resumed at episode {start_ep}")

    transform = v2.Compose([
        v2.Resize((128, 128)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # TODO: Fix waiting hack
    print("Start the level! You have 5 seconds before training starts")
    time.sleep(5)

    total_steps = 0

    for ep in range(start_ep, num_episodes):
        env.reset()

        total_r = 0

        # Init state
        state = build_state(transform)
        
        pbar = tqdm(range(max_steps), desc=f"Ep{ep+1}")
        for step in pbar:
            # Get predicted action
            action = agent.act(state)

            img = state[0, -1]  # take the most recent frame (C, H, W)

            show_img = False
            if show_img:
                img_np = img.permute(1, 2, 0).cpu().numpy()  # CHW → HWC
                img_np = (img_np * 255).astype('uint8')      # scale to [0, 255] if float32

                cv2.imshow("test", img_np)
                cv2.waitKey(1)

            # Simulate
            _, reward, done, info = env.step(action)
            total_r += reward
            pbar.set_postfix(r=round(total_r, 2), lvl_percent=round(info["percent"], 1))

            # Get resutling state and train
            next_state = build_state(transform)
            state = next_state
            if info['percent']>best_percent:
                best_percent = info['percent']
            
            if done:
                print(f"Died at step {step}.")
                break

        print(f"Ep {ep+1} → reward {total_r:.1f}")


if __name__ == "__main__":
    start_geometry_dash()
    gdclient.connect()
    Thread(target=listen_for_frame_buffer, daemon=True).start()
    play()
