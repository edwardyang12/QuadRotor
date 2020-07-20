import gym
import quad_env

env = gym.make('quadcopter-v3')

for i in range(1000):
    desired_state = env.trajectory(1.2)
    F, M = controller.run(quadcopter, desired_state)
    env.step(F, M)
    env.render()
