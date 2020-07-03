import gym
from gym import spaces
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Simulation parameters
g = 9.81
T = 5

# dimensions of quadrotor
m = 0.2
x = 0.44
y = 0.44
z = 0.12

class QuadRotorEnv(gym.Env):

    def __init__(self, target_pos):

        # position of quadrotor
        self.x_pos = 15
        self.y_pos = 15
        self.z_pos = 5

        # velocity of quadrotor
        self.x_vel = 0
        self.y_vel = 0
        self.z_vel = 0

        # acceleration of quadrotor
        self.x_acc = 0
        self.y_acc = 0
        self.z_acc = 0

        # orientation of quadrotor
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        # spin of quadrotor
        self.roll_vel = 0
        self.pitch_vel = 0
        self.yaw_vel = 0

        self.target_pos = target_pos

        # time
        self.dt = 0.1

        # total run time
        self.t = 0

        self.max_acc = 1

        self.action_space = spaces.Box(
            # acceleration in x,y,z
            low = np.array([-self.max_acc,-self.max_acc,-self.max_acc]),
            high = np.array([self.max_acc, self.max_acc, self.max_acc]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            # position, velocity, acceleration, rotation_velocity
            low = np.array([[0,0,0],[-20,-20,-20],[-1,-1,-1],[-10,-10,-10]]),
            high = np.array([[100,100,100],[20,20,20],[1,1,1],[10,10,10]]),
            dtype=np.float32
        )

    def step(self, action):

        #action: accelarations of each coordinates.
        acc = action
        done = False
        self.x_acc = acc[0]
        self.y_acc = acc[1]
        self.z_acc = acc[2]

        self.x_acc = np.clip(self.x_acc, -self.max_acc, self.max_acc)
        self.y_acc = np.clip(self.y_acc, -self.max_acc, self.max_acc)
        self.z_acc = np.clip(self.z_acc, -self.max_acc, self.max_acc)

        self.x_vel += self.x_acc * self.dt
        self.y_vel += self.y_acc * self.dt
        self.z_vel += (self.z_acc - g) * self.dt
        self.x_pos += self.x_vel * self.dt
        self.y_pos += self.y_vel * self.dt
        self.z_pos += self.z_vel * self.dt

        xy_acc = np.sqrt(self.x_acc**2 + self.y_acc**2)
        yz_acc = np.sqrt(self.y_acc**2 + self.z_acc**2)
        xz_acc = np.sqrt(self.x_acc**2 + self.z_acc**2)

        # adjusting yaw by assuming centripetal motion
        xy_angleVel = np.sqrt(xy_acc / np.sqrt(x**2 + y**2))
        if(self.x_acc * self.y_acc <0):
            xy_angleVel = -xy_angleVel
        self.yaw_vel += xy_angleVel

        # adjusting roll by assuming centripetal motion
        yz_angleVel = np.sqrt(yz_acc / np.sqrt(z**2 + y**2))
        if(self.z_acc * self.y_acc <0):
            yz_angleVel = -yz_angleVel
        self.roll_vel += yz_angleVel

        # adjusting pitch by assuming centripetal motion
        xz_angleVel = np.sqrt(xz_acc / np.sqrt(z**2 + x**2))
        if(self.z_acc * self.x_acc <0):
            xz_angleVel = -xz_angleVel
        self.pitch_vel += xz_angleVel

        self.yaw += self.yaw_vel * self.dt
        self.roll += self.roll_vel * self.dt
        self.pitch += self.pitch_vel * self.dt

        self.t += self.dt

        # restrict space
        if(self.x_pos > 100 or self.x_pos < 0 ):
            done = True
        if(self.y_pos > 100 or self.y_pos < 0 ):
            done = True
        if(self.z_pos > 100 or self.z_pos < 0 ):
            done = True

        if(self.x_vel > 20 or self.x_vel < -20):
            done = True
        if(self.y_vel > 20 or self.y_vel < -20):
            done = True
        if(self.z_vel > 20 or self.z_vel < -20):
            done = True

        if(self.roll_vel > 10 or self.roll_vel < -10):
            done = True
        if(self.pitch_vel > 10 or self.pitch_vel < -10):
            done = True
        if(self.yaw_vel > 10 or self.yaw_vel < -10):
            done = True

        return self._get_obs(), self._get_reward(), done

    def _get_obs(self):
        #pos_array = [self.target_pos[0] - self.x_pos, self.target_pos[1] - self.y_pos, self.target_pos[2] - self.z_pos]
        pos_array = [self.x_pos, self.y_pos, self.z_pos]
        vel_array = [self.x_vel, self.y_vel, self.z_vel]
        acc_array = [self.x_acc, self.y_acc, self.z_acc]
        rotVel_array = [self.roll_vel, self.pitch_vel, self.yaw_vel]
        return [pos_array, vel_array, acc_array, rotVel_array]

    def _get_reward(self):
        pos_r = 4e-3 * np.linalg.norm([self.target_pos[0] - self.x_pos, self.target_pos[1] - self.y_pos, self.target_pos[2] - self.z_pos])
        vel_r = 5e-4 * np.linalg.norm([self.x_vel, self.y_vel, self.z_vel])
        acc_r = 2e-4 * np.linalg.norm([self.x_acc, self.y_acc, self.z_acc])
        rotvel_r = 3e-4 * np.linalg.norm([self.roll_vel, self.pitch_vel, self.yaw_vel])
        return -(pos_r + vel_r + acc_r + rotvel_r)

    def reset(self):
        # position of quadrotor
        self.x_pos = 15
        self.y_pos = 15
        self.z_pos = 5

        # velocity of quadrotor
        self.x_vel = 0
        self.y_vel = 0
        self.z_vel = 0

        # acceleration of quadrotor
        self.x_acc = 0
        self.y_acc = 0
        self.z_acc = 0

        # orientation of quadrotor
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        # spin of quadrotor
        self.roll_vel = 0
        self.pitch_vel = 0
        self.yaw_vel = 0

        # time
        self.dt = 0.1

        # total run time
        self.t = 0
        return self._get_obs()

    def render(self, mode='human'):
        pos_array = [self.target_pos[0] - self.x_pos, self.target_pos[1] - self.y_pos, self.target_pos[2] - self.z_pos]
        print(f'Distance to Target: {pos_array}')

if __name__ == '__main__':
    drone = QuadRotorEnv([10,20,30])
    for i in range(20):
        print(drone.step([-1,2,3])[0])
