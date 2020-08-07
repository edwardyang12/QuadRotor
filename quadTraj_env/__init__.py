from gym.envs.registration import register

register(
    id='quadcopterTrajectory-v0',
    entry_point='quadTraj_env.envs.quad_env4:QuadRotorEnv',
)
