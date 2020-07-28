import gym
import quad_env
from quad_env.envs.controller import run

env = gym.make('quadcopter-v3')

while True:
    done = False
    i = 0
    while not done:
        env.trajectory(1.2)
        desired_state = env.des_state
        F, M = run(env, desired_state)
        state, reward, done = env.step([F, M])
        if i%env.control_iterations==0:
            env.render()
        i += 1
    env.reset()
