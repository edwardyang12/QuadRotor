from gym.envs.registration import register

register(
    id='quadcopter-v3',
    entry_point='gym_quadcopter_3.envs:QuadRotorEnv',
)
