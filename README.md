# Reinforcement Learning for Geometry Dash

A Deep Q Network to learn Geometry Dash and a C++ mod to control the game. Our final project for UCSB CS 190A: Machine Learning.

![demo](docs/demo.gif)

## Setup

TODO: Setup ai model and geode mod (mac only)

## Data Pipeline

- Python commands C++ mod to step, restart, etc via tcp `localhost:22222`
- C++ sends screen buffer to Python via tcp `localhost:22223`
