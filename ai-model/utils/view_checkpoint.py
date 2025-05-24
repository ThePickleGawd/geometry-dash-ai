import torch
import matplotlib.pyplot as plt
import numpy as np

# Load checkpoint
cp = torch.load("../checkpoints/latest.pt")
time_alive_per_ep = cp.get("time_alive", {})
epsilon_per_ep = cp.get("epsilon", {})
reward_per_ep = cp.get("total_reward", {})

# Sort episodes for consistent x-axis
episodes = sorted(time_alive_per_ep.keys())
times_alive = [time_alive_per_ep[ep] for ep in episodes]
epsilons = [epsilon_per_ep[ep] for ep in episodes]
rewards = [reward_per_ep[ep] for ep in episodes]

# Apply smoothing
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

smoothing = 50  # Change this to control the smoothing
smoothed_times = moving_average(times_alive, smoothing)
smoothed_episodes = episodes[smoothing - 1:]  # Adjust x-axis to match length

smoothed_rewards = moving_average(rewards, smoothing)

#Time Alive Plot
plt.figure(figsize=(10, 5))
plt.plot(episodes, times_alive, label="Original", alpha=0.3, linewidth=1)
plt.plot(smoothed_episodes, smoothed_times, label=f"Smoothed (window={smoothing})", color='red', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Time Alive (seconds)")
plt.title("Agent Survival Time per Episode")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#Reward plot
plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards, label="Original", alpha=0.3, linewidth=1)
plt.plot(smoothed_episodes, smoothed_rewards, label=f"Smoothed (window={smoothing})", color='red', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Total Reward per Episode")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#Epsilon Plot
plt.figure(figsize=(10, 5))
plt.plot(episodes, epsilons, label="Epsilon", linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Randomness Factor")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
