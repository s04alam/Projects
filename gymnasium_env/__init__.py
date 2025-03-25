from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="gymnasium_env.envs.grid_world:GridWorldEnv",
)

