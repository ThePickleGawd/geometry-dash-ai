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
from model import DQNModel, DUEL_DQNModel, DeeperDQNModel, DeeperDQNModelv2, NoisyDeeperDQNModelv2, smallDQN, ActionDeeperDQNModelv2
from agent import Agent, AgentACTION
import config
import random
import math

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

def train(num_episodes=50000, max_steps=5000, resume=False):
    env   = GeometryDashEnv()
    device = "cuda" if torch.cuda.is_available() else "mps"
    # model = NoisyDeeperDQNModelv2().to(device)
    model = DeeperDQNModelv2().to(device)
    # model = DUEL_DQNModel().to(device)
    # model = smallDQN().to(device)
    agent = Agent(model)

    start_ep = 0 
    time_alive_per_ep = {}
    epsilon_per_ep = {}
    reward_per_ep = {}
    best_time = 0
    if config.PREVIOUS_ACTION:
        previous_action = torch.unsqueeze(torch.zeros(2),0)

    # resume
    if resume and os.path.exists("checkpoints/latest.pt"):
        cp = torch.load("checkpoints/latest.pt",weights_only=False)
        agent.model.load_state_dict(cp["model_state"])
        agent.target_model.load_state_dict(cp["model_state"])
        agent.optimizer.load_state_dict(cp["optimizer_state"])
        start_ep = cp["episode"] + 1
        time_alive_per_ep = cp.get("time_alive", {})
        reward_per_ep = cp.get("total_reward", {})
        epsilon_per_ep = cp.get("epsilon", {})
        agent.epsilon = epsilon_per_ep[start_ep-1]
        # agent.epsilon = config.EPSILON_START * (config.EPSILON_DECAY ** (start_ep * 20)) # TODO: A hack to step epsilon (assume about 20 steps per episode)
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
        #cube 1,29.76 47,85.9
        #ship 30,46.79 86,98
        best_percent = 0   
        if random.random() < config.RANDOM_SPAWN_PERCENTAGE:
            pct = random.randint(1,90)
            #Cube only Random Spawn Below
            # pct = random.randint(1, 80-(47-29))
            # if pct>29:
            #     pct = pct+(47-29)
            #SHIP
            # pct = random.randint(30, 90-(86-46))
            # if pct>46:
            #     pct = pct+(86-46)
        else:
            pct = config.SET_SPAWN
        env.reset(0)

        start_time = time.time()
        total_r = 0

        # Init state
        state = build_state(transform)
        
        previous_steps = total_steps
        pbar = tqdm(range(max_steps), desc=f"Ep{ep+1}")
        
        stupid_death = False
        #for NEWSTATE
        # env.percentCount.append([False] * 101)
        # previous_percent = 0
        # memory = []
        for step in pbar:
            #for NEWSTATE
            # totalvisits= 0
            # for i in range(len(env.percentCount)):
            #     totalvisits += env.percentCount[i][int(previous_percent)]
            
            # Get predicted action
            if config.PREVIOUS_ACTION:
                action = agent.act(state,previous_action)
                if action==1: previous_action = torch.unsqueeze(torch.Tensor([0,1]),0)
                else:         previous_action = torch.unsqueeze(torch.Tensor([1,0]),0)
            else:
                action = agent.act(state)


            img = state[0, -1]  # take the most recent frame (C, H, W)

            show_img = False
            if show_img:
                img_np = img.permute(1, 2, 0).cpu().numpy()  # CHW → HWC
                img_np = (img_np * 255).astype('uint8')      # scale to [0, 255] if float32

                cv2.imshow("test", img_np)
                cv2.waitKey(1)

            # Simulate
            _, reward, done, info = env.step(action,start_percent=pct)
            total_r += reward
            pbar.set_postfix(r=round(total_r, 2), lvl_percent=round(info["percent"], 1))

            # Get resutling state and train
            next_state = build_state(transform)
            if step==0 and done == True:
                stupid_death = True
                break
            #CHANGED to clip reward
            if config.PREVIOUS_ACTION:
                agent.remember(state, action, min(max(reward,-1),1), next_state, done,previous_action)
            else: agent.remember(state, action, min(max(reward,-1),1), next_state, done)
            #for NEWSTATE
            # memory.append([state, action, reward, next_state, done])
            
            # agent.train() #if u want uncomment this, comment out line 184 for the batched

            state = next_state
            if info['percent']>best_percent:
                best_percent = info['percent']
            
            #for NEWSTATE
            # previous_percent = info['percent']
            
            # Update target network every STEPS_BEFORE_TARGET_UPDATE steps
            total_steps += 1
            if total_steps % config.STEPS_BEFORE_TARGET_UPDATE == 0:
                agent.update_target_network()
            
            if done:
                print(f"Died at step {step}.")
                # agent.save_death_replay()
                break
        if stupid_death:
            print('stupid death')
            continue
        end_time = time.time()
        time_alive = end_time - start_time
        time_alive_per_ep[ep] = time_alive  # save for this ep
        if best_time < time_alive: best_time = time_alive
        epsilon_per_ep[ep] = agent.epsilon
        reward_per_ep[ep] = total_r

        print(f"Ep {ep+1} → reward {total_r:.1f}")

        # newstate reward
        # reward += (1/2 + int(best_percent)/100)*(config.NEW_STATE_REWARD / max(1,math.sqrt(totalvisits)))

        #adding delayed new state reward
        # for step in memory:
        #     step[2]+=reward
        #     agent.remember(*step)

        #USE IF DOING NOISYNET
        # agent.model.reset_noise()
        # agent.target_model.reset_noise()

        if (ep + 1) % config.SAVE_EPOCH == 0 or (round(best_time,1) == round(time_alive,1) and best_time > 15):
            torch.save({
                "episode": ep,
                "model_state": agent.model.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
                "time_alive": time_alive_per_ep,
                "total_reward": reward_per_ep,
                "epsilon":epsilon_per_ep,
            }, f"checkpoints/{ep+1}_{round(time_alive,1)}.pt")
            # torch.save(agent.replay_buffer,f"checkpoints/replay_buffer_ep{ep}.pt")
            print(f"Saved checkpoint @ ep {ep+1}")

        torch.save({
            "episode": ep,
            "model_state": agent.model.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "time_alive": time_alive_per_ep,
            "total_reward": reward_per_ep,
            "epsilon":epsilon_per_ep,
        }, f"checkpoints/latest.pt")
        
        #batched training at the end of att
        for _ in range(total_steps-previous_steps):
            agent.train()

    env.close()
    print("\nBEST PERCENTAGE:", best_percent)

if __name__ == "__main__":
    start_geometry_dash()
    gdclient.connect()
    Thread(target=listen_for_frame_buffer, daemon=True).start()
    train(resume=False)
