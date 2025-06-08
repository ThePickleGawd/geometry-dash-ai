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
from model import ExpertsModel, ExpertsModelTransformer
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

def train(num_episodes=50000, max_steps=5000, resume=True):
    env = GeometryDashEnv()
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = ExpertsModelTransformer(is_train=True).to(device)
    agent = AgentExperts(model)

    start_ep = 0
    best_percent = 0    
    time_alive_per_ep = {}
    epsilon_per_ep = {}
    reward_per_ep = {}
    ship_acc_per_ep = {}

    if resume and os.path.exists("checkpoints/latest.pt"):
        cp = torch.load("checkpoints/latest.pt")
        agent.model.load_state_dict(cp["model_state"])
        agent.target_model.load_state_dict(cp["model_state"])
        agent.optimizer.load_state_dict(cp["optimizer_state"])
        start_ep = cp["episode"] + 1
        time_alive_per_ep = cp.get("time_alive", {})
        reward_per_ep = cp.get("total_reward", {})
        epsilon_per_ep = cp.get("epsilon", {})
        ship_acc_per_ep = cp.get("ship_acc", {})
        agent.epsilon = epsilon_per_ep[len(epsilon_per_ep)-1]
        print(f"Resumed at episode {start_ep}")

    transform = v2.Compose([
        v2.Resize((128, 128)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    print("Start the level! You have 5 seconds before training starts")
    time.sleep(5)

    total_steps = 0

    for ep in range(start_ep, num_episodes):
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

        env.reset(pct)
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

            next_state = build_state(transform)
            agent.remember(state, action, reward, next_state, is_ship, done)
            ship_pred_acc = agent.train()
            episode_ship_accs.append(ship_pred_acc)

            state = next_state
            if info['percent'] > best_percent:
                best_percent = info['percent']

            total_steps += 1
            if total_steps % config.STEPS_BEFORE_TARGET_UPDATE == 0:
                agent.update_target_network()

            if done:
                print(f"Died at step {step}.")
                agent.on_death()
                break

        end_time = time.time()
        time_alive = end_time - start_time
        time_alive_per_ep[ep] = time_alive
        epsilon_per_ep[ep] = agent.epsilon
        reward_per_ep[ep] = total_r
        valid_accs = [a for a in episode_ship_accs if a is not None]
        ship_acc_per_ep[ep] = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0


        print(f"Ep {ep+1} → reward {total_r:.1f}")

        save_dict = {
            "episode": ep,
            "model_state": agent.model.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "time_alive": time_alive_per_ep,
            "total_reward": reward_per_ep,
            "epsilon": epsilon_per_ep,
            "ship_acc": ship_acc_per_ep,
        }

        if (ep + 1) % config.SAVE_EPOCH == 0:
            torch.save(save_dict, f"checkpoints/{ep+1}.pt")
            print(f"Saved checkpoint @ ep {ep+1}")

        torch.save(save_dict, f"checkpoints/latest.pt")

    env.close()
    print("\nBEST PERCENTAGE:", best_percent)

if __name__ == "__main__":
    start_geometry_dash()
    gdclient.connect()
    Thread(target=listen_for_frame_buffer, daemon=True).start()
    train(resume=False)
