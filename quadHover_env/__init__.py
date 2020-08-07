from gym.envs.registration import register

register(
    id='quadcopterHover-v0',
    entry_point='quadHover_env.envs.quad_env4:QuadRotorEnv',
)
