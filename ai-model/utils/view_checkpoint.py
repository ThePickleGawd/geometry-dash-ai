import torch
import matplotlib.pyplot as plt

# Load checkpoint
cp = torch.load("../checkpoints/5503.pt")
time_alive_per_ep = cp.get("time_alive", {})

# Sort episodes for consistent x-axis
episodes = sorted(time_alive_per_ep.keys())
times_alive = [time_alive_per_ep[ep] for ep in episodes]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(episodes, times_alive, label="Time Alive (s)", marker='o', markersize=3, linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Time Alive (seconds)")
plt.title("Agent Survival Time per Episode")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
