import numpy as np
from geometry_dash_gym.envs import GeometryDashEnv
import gymnasium as gym
import subprocess
import time

def start_geometry_dash():
    # Open geometry dash
    subprocess.Popen(["geode", "run"])
    print("Waiting 5 seconds for Geometry Dash to load...")
    time.sleep(5)  # Wait for the game to load

def train(num_episodes=1000, max_steps_per_episode=1000):
    env = GeometryDashEnv()

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            # Random action: 0 (nothing) or 1 (hold)
            action = np.random.choice([0, 1])

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                break

        print(f"Episode {episode+1} finished with reward {total_reward}")

    env.close()

if __name__ == "__main__":
    start_geometry_dash()
    train()