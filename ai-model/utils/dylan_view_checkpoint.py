import torch
import matplotlib.pyplot as plt
import numpy as np

# Load checkpoint
cp = torch.load("../checkpoints/experts-6-7/latest.pt", map_location="cpu")
time_alive_per_ep = cp.get("time_alive", {})
epsilon_per_ep = cp.get("epsilon", {})
reward_per_ep = cp.get("total_reward", {})
ship_acc_per_ep = cp.get("ship_acc", {})

# Convert tensor values to floats (in case they weren't saved as pure Python)
def safe_float(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().item()
    return float(v)

# Sort episodes
episodes = sorted(time_alive_per_ep.keys())

def extract(metric_dict):
    return [safe_float(metric_dict.get(ep, 0.0)) for ep in episodes]

times_alive = extract(time_alive_per_ep)
epsilons     = extract(epsilon_per_ep)
rewards      = extract(reward_per_ep)
ship_accs    = extract(ship_acc_per_ep)

# Smoothing
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

smoothing = 50
def smooth(data):
    return moving_average(data, smoothing), episodes[smoothing - 1:]

smoothed_times, smoothed_episodes = smooth(times_alive)
smoothed_rewards, _ = smooth(rewards)
smoothed_ship_acc, _ = smooth(ship_accs)

# Time Alive Plot
plt.figure(figsize=(10, 4))
plt.plot(episodes, times_alive, label="Original", alpha=0.3)
plt.plot(smoothed_episodes, smoothed_times, label="Smoothed", color='red')
plt.title("Time Alive per Episode")
plt.xlabel("Episode")
plt.ylabel("Time Alive (s)")
plt.legend()
plt.grid(True)

# Ship Accuracy Plot
plt.figure(figsize=(10, 4))
plt.plot(episodes, ship_accs, label="Original", alpha=0.3)
plt.plot(smoothed_episodes, smoothed_ship_acc, label="Smoothed", color='green')
plt.title("Ship Mode Classification Accuracy per Episode")
plt.xlabel("Episode")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Reward Plot
plt.figure(figsize=(10, 4))
plt.plot(episodes, rewards, label="Original", alpha=0.3)
plt.plot(smoothed_episodes, smoothed_rewards, label="Smoothed", color='blue')
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

# Epsilon Plot
plt.figure(figsize=(10, 4))
plt.plot(episodes, epsilons, label="Epsilon", color='purple')
plt.title("Epsilon per Episode")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
