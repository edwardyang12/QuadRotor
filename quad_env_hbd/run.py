import gym
import quad_env
from quad_env.envs.controller import run

env = gym.make('quadcopter-v3')

for i in range(1000):
    desired_state = env.trajectory(1.2)
    F, M = run(env, desired_state)
    env.step(F, M)
    env.render()
