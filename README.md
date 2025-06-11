# Reinforcement Learning for Geometry Dash

A DQN framework for playing Geometry Dash. Our project contains two parts:

1. A suite of DQN agents, models, and a custom gymnasium environment. `/ai-model`
2. A C++ mod to control the game. `/gd-rl-mod`

[Paper](/docs/paper.pdf) | [Demo](https://youtu.be/PKDMGPf-PEA)

![demo](docs/demo.gif)

## Setup

### AI Model

```bash
# Install uv
pip install uv

# Install
cd ai-model
uv sync

# Download or train model checkpoints to /checkpoints
```

### C++ Mod

1. Setup Geode. More details: https://docs.geode-sdk.org/getting-started/geode-cli#macos

```bash
# Install Geode (tested on v4.4.0)
brew install geode-sdk/geode/geode-cli

# Setup profile
geode config setup

# Install SDK
geode sdk install
geode sdk install-binaries
```

2. Build the mod

```bash
geode build
```
