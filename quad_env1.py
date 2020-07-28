import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
g = 9.81
T = 5

# dimensions of quadrotor
m = 0.2
x = 0.44
y = 0.44
z = 0.12

# Moment of inertia (to be calculated later based on dimensions)
Ixx = 0.1
Iyy = 0.1
Izz = 0.05

class QuadRotorEnv(gym.Env):

    def __init__(self, target_pos):

        # position of quadrotor (x,y,z)
        self.pos = [50,50,50]

        # velocity of quadrotor (x,y,z)
        self.vel = [0,0,0]

        # orientation of quadrotor (roll, pitch, yaw)
        self.orientation = [0,0,0]

        # spin of quadrotor (roll, pitch, yaw)
        self.angle_vel = [0,0,0]

        self.target_pos = target_pos

        # 1 CCW, 2 CCW, 3 CW, 4 CW
        self.thrust = [0,0,0,0]

        # time
        self.dt = 0.1

        # total run time
        self.t = 0

        self.max_thrust = 1

        self.action_space = spaces.Box(
            # thrust in x,y,z
            low = np.array([-self.max_thrust,-self.max_thrust,-self.max_thrust, -self.max_thrust]),
            high = np.array([self.max_thrust,self.max_thrust,self.max_thrust, self.max_thrust]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            # position, velocity, rotation_velocity, orientation
            low = np.array([[0,0,0],[-20,-20,-20],[-10,-10,-10], [-5, -5, -5]]),
            high = np.array([[100,100,100],[20,20,20],[10,10,10], [5, 5, 5]]),
            dtype=np.float32
        )

    def _torque_(self, F,d1,d2):
        return F * np.sqrt(d1**2 + d2**2)

    def step(self, action):

        #action: thrust per propeller
        self.thrust = action
        done = False

        # moment calculation
        L = self.thrust[0] * y/2 - self.thrust[1] * y/2 - self.thrust[2] * y/2 + self.thrust[3] * y/2
        M = -self.thrust[0] * x/2 + self.thrust[1] * x/2 - self.thrust[2] * x/2 + self.thrust[3] * x/2
        N = -self._torque_(self.thrust[0],x/2,y/z) - self._torque_(self.thrust[1],x/2,y/z) + self._torque_(self.thrust[2],x/2,y/z) + self._torque_(self.thrust[3],x/2,y/z)
        Fz = self.thrust[0] +  self.thrust[1] + self.thrust[2]  + self.thrust[3]

        ub = self.vel[0]
        vb = self.vel[1]
        wb = self.vel[2]
        p = self.angle_vel[0]
        q = self.angle_vel[1]
        r = self.angle_vel[2]
        phi = self.orientation[0]
        theta = self.orientation[1]
        psi = self.orientation[2]
        xE = self.pos[0]
        yE = self.pos[1]
        hE = self.pos[2]

        cphi = np.cos(phi);   sphi = np.sin(phi)
        cthe = np.cos(theta); sthe = np.sin(theta)
        cpsi = np.cos(psi);   spsi = np.sin(psi)

        # new velocities
        vxdot = (-g * sthe + r * vb - q * wb) * self.dt + self.vel[0]
        vydot = (g * sphi*cthe - r * ub + p * wb) * self.dt + self.vel[1]
        vzdot = (1/m * (-Fz) + g*cphi*cthe + q * ub - p * vb) * self.dt + self.vel[2]

        # new angular velocities
        avxdot = (1/Ixx * (L + (Iyy - Izz) * q * r)) * self.dt + self.angle_vel[0]
        avydot = (1/Iyy * (M + (Izz - Ixx) * p * r)) * self.dt + self.angle_vel[1]
        avzdot = (1/Izz * (N + (Ixx - Iyy) * p * q)) * self.dt + self.angle_vel[2]

        # new angle
        axdot = (p + (q*sphi + r*cphi) * sthe / cthe) * self.dt + self.orientation[0]
        aydot = (q * cphi - r * sphi) * self.dt + self.orientation[1]
        azdot = ((q * sphi + r * cphi) / cthe) * self.dt + self.orientation[2]

        # position
        pxdot = (cthe*cpsi*ub + (-cphi*spsi + sphi*sthe*cpsi) * vb + (sphi*spsi+cphi*sthe*cpsi) * wb) * self.dt + self.pos[0]
        pydot = (cthe*spsi * ub + (cphi*cpsi+sphi*sthe*spsi) * vb + (-sphi*cpsi+cphi*sthe*spsi) * wb) * self.dt + self.pos[1]
        pzdot = (-1*(-sthe * ub + sphi*cthe * vb + cphi*cthe * wb)) * self.dt + self.pos[2]

        self.vel = [vxdot, vydot, vzdot]
        self.angle_vel = [avxdot, avydot, avzdot]
        # self.vel = np.clip([vxdot, vydot, vzdot], -20, 20)
        # self.angle_vel = np.clip([avxdot, avydot, avzdot], -20, 20)
        self.orientation = [axdot, aydot, azdot]
        self.pos = [pxdot, pydot, pzdot]

        self.t += self.dt

        # restrict space
        if(self.pos[0] > 100 or self.pos[0] < 0 ):
            done = True
        if(self.pos[1] > 100 or self.pos[1] < 0 ):
            done = True
        if(self.pos[2] > 100 or self.pos[2] < 0 ):
            done = True

        return self._get_obs(), self._get_reward(), done

    def _get_obs(self):
        return [self.pos, self.vel, self.angle_vel, self.orientation]

    def _get_reward(self):
        pos_r = -4e-1 * np.linalg.norm([self.target_pos[0] - self.pos[0], self.target_pos[1] - self.pos[1], self.target_pos[2] - self.pos[2]])
        #vel_r = 5e-4 * np.linalg.norm(self.vel)
        rotvel_r = -3e-4 * np.linalg.norm(self.angle_vel)
        return (pos_r + rotvel_r)

    def reset(self):

        self.pos = [50,50,50]
        self.vel = [0,0,0]
        self.orientation = [0,0,0]
        self.angle_vel = [0,0,0]
        self.angle_acc = [0,0,0]
        self.thrust = [0,0,0,0]
        self.t = 0

        return self._get_obs()

    def render(self, mode='human'):
        pos_array = [self.target_pos[0] - self.pos[0], self.target_pos[1] - self.pos[1], self.target_pos[2] - self.pos[2]]
        print(f'Distance to Target: {pos_array}')

if __name__ == '__main__':
    drone = QuadRotorEnv([10,20,30])
    for i in range(5):
        print(drone.step([-1,-1,1,1])[0])
