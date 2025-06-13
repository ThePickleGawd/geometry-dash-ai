# Reinforcement Learning for Geometry Dash

A DQN framework for playing Geometry Dash. Our project contains two parts:

1. A suite of DQN agents, models, and a custom gymnasium environment. `/ai-model`
2. A C++ mod to control the game. `/gd-rl-mod`

[Paper](/docs/paper.pdf) | [Demo](https://youtu.be/PKDMGPf-PEA)

## Demo

https://youtu.be/PKDMGPf-PEA

Standard DQN Agent:

![demo](docs/demo.gif)

Mixture-of-Experts DQN Agent:

https://github.com/user-attachments/assets/d316da5f-1fc3-4a10-9835-31f90c44d017

## Setup

Since the mod only works on Mac, this is the only supported platform. See a side note about Windows below.

### AI Model

```bash
# Install uv
pip install uv

# Install
cd ai-model
uv sync

# Train (change resume=False or True near end of script)
uv run train.py

# Play
# Ensure model checkpoints is in /checkpoints/latest.pt
uv run play.py
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

### Windows

The socket and OpenGL libraries are different for Mac vs. Windows, and we've only got it to work for Mac. Some progress is made in the `windows` branch, here are some clues.

- Platform specific imports: [platform.hpp](https://github.com/ThePickleGawd/geometry-dash-ai/blob/windows/gd-rl-mod/src/utils/platform.hpp)
- Error about socket library redefinition if we include "Windows.h", otherwise it's library not found [server.cpp](https://github.com/ThePickleGawd/geometry-dash-ai/blob/windows/gd-rl-mod/src/utils/server.cpp)
