from gym.envs.registration import register

register(
    id='quadcopter-v4',
    entry_point='quad_env.envs.quad_env4:QuadRotorEnv',
)
