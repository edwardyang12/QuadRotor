import gym
import quad_env
from quad_env.envs.controller import run

env = gym.make('quadcopter-v5')

while True:
    done = False
    i = 0
    while not done:
        desired_state = env.des_state
        F, M = run(env, desired_state)
        state, reward, done = env.step(F)
        if i%env.control_iterations==0:
            env.render()
        i += 1
    env.reset()
