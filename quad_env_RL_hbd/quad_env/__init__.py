from gym.envs.registration import register

register(
    id='quadcopter-v5',
    entry_point='quad_env.envs.quad_envF:QuadRotorEnv',
)
