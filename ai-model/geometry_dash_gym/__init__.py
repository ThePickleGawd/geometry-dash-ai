from gymnasium.envs.registration import register

# TODO: Someone fix this pls!!
register(
    id="geometry_dash/GridWorld-v0",
    entry_point="geometry_dash.envs:GridWorldEnv",
)
