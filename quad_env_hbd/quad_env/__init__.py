from gym.envs.registration import register

register(
    id='quadcopter-v3',
    entry_point='quad_env.envs.quad_env3:QuadRotorEnv',
)
