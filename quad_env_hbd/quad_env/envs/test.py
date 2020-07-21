from quad_env3 import QuadRotorEnv
import controller

env = QuadRotorEnv()

for i in range(1000):
    desired_state = env.trajectory(1.2)
    F, M = controller.run(env, desired_state)
    env.step(F, M)
    env.render()
