# referenced from https://github.com/bonn0062/quadcopter2/blob/master/task.py

# for hovering
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from quadHover_env.envs.TrajectoryGenerator import earth_to_body_frame, body_to_earth_frame
from quadHover_env.envs.quadPlot import set_limit, plot_waypoints
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

# simulation parameters
gravity = -9.81 # gravity
rho = 1.2 # density of air
mass = 0.958  # 300 g
width, length, height = .51, .51, .235

C_d = 0.3 # unknown

# propeller parameters
l_to_rotor = 0.4
propeller_size = 0.1

# moments of inertia
I_x = 1 / 12. * mass * (height**2 + width**2)
I_y = 1 / 12. * mass * (height**2 + length**2)  # 0.0112 was a measured value
I_z = 1 / 12. * mass * (width**2 + length**2)

env_bounds = 50.0  # 300 m / 300 m / 300 m

class QuadRotorEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):

        self.final_pos = [0.,0.,40.]
        self.pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

        self.T = 5.

        self.viewer = None

        self.action_repeat = 1

        self.dt = 1/15.

        self.dims = np.array([width, length, height])  # x, y, z dimensions of quadcopter
        self.areas = np.array([length * height, width * height, width * length])

        self.moments_of_inertia = np.array([I_x, I_y, I_z])  # moments of inertia

        self.rotor_speeds = [0,0,0,0]
        self.max_rotor_speed = 900.
        self.action_space = spaces.Box(
            # rotor speeds of each propeller
            # low = np.array([-self.max_rotor_speed,-self.max_rotor_speed,-self.max_rotor_speed, -self.max_rotor_speed]),
            low = np.array([1.,1.,1.,1.]),
            high = np.array([self.max_rotor_speed, self.max_rotor_speed, self.max_rotor_speed, self.max_rotor_speed]),
            dtype=np.float32
        )

        self.lower_bounds = np.array([-env_bounds / 2, -env_bounds / 2, 0])
        self.upper_bounds = np.array([env_bounds / 2, env_bounds / 2, env_bounds])

        # orientation, angular_vel, distance, velocity
        high = np.concatenate([np.array([7., 7.,7., 30., 30., 30., env_bounds/2-self.final_pos[0], env_bounds/2-self.final_pos[1], env_bounds-self.final_pos[2], 30.,30.,30.,40.,40.,40.],
                        dtype=np.float32)] * self.action_repeat)
        low = np.concatenate([np.array([-7., -7.,-7., -30., -30., -30., -env_bounds/2-self.final_pos[0], -env_bounds/2-self.final_pos[1], 0-self.final_pos[2],-30.,-30.,-30.,-40.,-40.,-40.],
                        dtype=np.float32)] *self.action_repeat)
        self.observation_space = spaces.Box(
            low = low,
            high = high,
            dtype=np.float32
        )

        self.reset()

    def reset(self):

        self.close()

##        z = random.uniform(1, env_bounds)
##        self.final_pos = np.array([0,0,z])

        # position + angular position (x, y, z, roll, pitch, yaw)
        # x = random.uniform(-env_bounds/2, env_bounds/2)
        # y = random.uniform(-env_bounds/2, env_bounds/2)
        # z = random.uniform(1, env_bounds)
        # roll = random.uniform(-7, 7)
        # pitch = random.uniform(-7, 7)
        # yaw = random.uniform(-7, 7)
        # self.pose = np.array([x,y,z,roll,pitch,yaw])
        self.pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

        self.v = np.array([0.0, 0.0, 0.0])
        self.angular_v = np.array([0.0, 0.0, 0.0])
        self.linear_accel = np.array([0.0, 0.0, 0.0])
        self.angular_accels = np.array([0.0, 0.0, 0.0])
        self.prop_wind_speed = np.array([0., 0., 0., 0.])
        self.done = False

        self.rotor_speeds = [0,0,0,0]

        self.runtime = 0

        self.xList = []
        self.yList = []
        self.zList = []

        # return np.concatenate([self.pose[:3]] * self.action_repeat )
        distance = [(self.pose[0] - self.final_pos[0]),(self.pose[1] - self.final_pos[1]),(self.pose[2]- self.final_pos[2])]
        return np.concatenate([np.concatenate((self.pose[3:],self.angular_v,distance,self.v,self.linear_accel), axis=0)] * self.action_repeat)

    def setupGraph(self):
        self.viewer = plt.figure()
        ax = self.viewer.add_axes([0, 0, 1, 1], projection='3d')
        ax.plot([], [], [], '-', c='cyan')[0]
        ax.plot([], [], [], '-', c='red')[0]
        ax.plot([], [], [], '-', c='blue', marker='o', markevery=2)[0]
        ax.plot([], [], [], '.', c='red', markersize=4)[0]
        ax.plot([], [], [], '.', c='blue', markersize=2)[0]
        set_limit((-env_bounds/2,env_bounds/2), (-env_bounds/2,env_bounds/2), (0,env_bounds))
        plot_waypoints(np.array([self.pose[:3],self.final_pos]))
        ax = plt.gca()
        self.lines = ax.get_lines()

    def render(self, mode='human', close=False):
        if not close:
            if self.viewer is None:
                self.setupGraph()
            x,y,z = self.pose[:3]
            self.xList.append(x)
            self.yList.append(y)
            self.zList.append(z)
            self.lines[-1].set_data(self.xList, self.yList)
            self.lines[-1].set_3d_properties(self.zList)
            self.viewer.canvas.draw()
            plt.pause(0.01)

    def close(self):
        if self.viewer:
            plt.close()
            self.viewer=None

    def _find_body_velocity(self):
        body_velocity = np.matmul(earth_to_body_frame(*list(self.pose[3:])), self.v)
        return body_velocity

    def _get_linear_drag(self):
        linear_drag = 0.5 * rho * self._find_body_velocity()**2 * self.areas * C_d
        return linear_drag

    def _get_linear_forces(self, thrusts):

        gravity_force = mass * gravity * np.array([0, 0, 1])
        thrust_body_force = np.array([0, 0, sum(thrusts)])
        drag_body_force = -self._get_linear_drag()
        body_forces = thrust_body_force + drag_body_force

        linear_forces = np.matmul(body_to_earth_frame(*list(self.pose[3:])), body_forces)
        linear_forces += gravity_force
        return linear_forces

    def _get_moments(self, thrusts):
        thrust_moment = np.array([(thrusts[3] - thrusts[2]) * l_to_rotor,
                            (thrusts[1] - thrusts[0]) * l_to_rotor,
                            0])# (thrusts[2] + thrusts[3] - thrusts[0] - thrusts[1]) * self.T_q])  # Moment from thrust

        drag_moment =  C_d * 0.5 * rho * self.angular_v * np.absolute(self.angular_v) * self.areas * self.dims * self.dims
        moments = thrust_moment - drag_moment # + motor_inertia_moment
        return moments

    def _calc_prop_wind_speed(self):
        body_velocity = self._find_body_velocity()
        phi_dot, theta_dot = self.angular_v[0], self.angular_v[1]
        s_0 = np.array([0., 0., theta_dot * l_to_rotor])
        s_1 = -s_0
        s_2 = np.array([0., 0., phi_dot * l_to_rotor])
        s_3 = -s_2
        speeds = [s_0, s_1, s_2, s_3]
        for num in range(4):
            perpendicular_speed = speeds[num] + body_velocity
            self.prop_wind_speed[num] = perpendicular_speed[2]

    def _get_propeller_thrust(self, rotor_speeds):
        '''calculates net thrust (thrust - drag) based on velocity
        of propeller and incoming power'''
        self.rotor_speeds = rotor_speeds
        thrusts = []
        for prop_number in range(4):
            V = self.prop_wind_speed[prop_number]
            D = propeller_size
            n = rotor_speeds[prop_number]
            J = V / (n * D)
            #print(V, J, self.pose)
            # From http://m-selig.ae.illinois.edu/pubs/BrandtSelig-2011-AIAA-2011-1255-LRN-Propellers.pdf
            C_T = max(.12 - .07*max(0, J)-.1*max(0, J)**2, 0)
            thrusts.append(C_T * rho * n**2 * D**4)
        return thrusts

    def _get_reward_hover(self):
        reward = 0


        xrewardpos = -np.abs(self.pose[0]- np.array(self.final_pos[0]))/25. + 1
        yrewardpos = -np.abs(self.pose[1]- np.array(self.final_pos[1]))/25. + 1
        zrewardpos = -np.abs(self.pose[2]- np.array(self.final_pos[2]))/40. + 1

        rewardpos = xrewardpos*0.3 + yrewardpos*0.3 + zrewardpos * 0.4

        rewardacc = -np.linalg.norm(self.linear_accel/40.)/np.sqrt(3) + 1
        rewardvel = -np.linalg.norm(self.v/30.)/np.sqrt(3)+1
        rewardangular =  -np.linalg.norm(self.angular_v/30.)/np.sqrt(3)+ 1

        reward = rewardpos*0.8 + rewardacc*0.05 + rewardvel*0.1 + rewardangular*0.05

        return reward

    def _next_timestep(self, rotor_speeds):
        self._calc_prop_wind_speed()
        thrusts = self._get_propeller_thrust(rotor_speeds)
        self.linear_accel = self._get_linear_forces(thrusts) / mass

        position = self.pose[:3] + self.v * self.dt + 0.5 * self.linear_accel * self.dt**2
        self.v += self.linear_accel * self.dt

        moments = self._get_moments(thrusts)

        self.angular_accels = moments / self.moments_of_inertia
        angles = self.pose[3:] + self.angular_v * self.dt + 0.5 * self.angular_accels * self.dt**2
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        self.angular_v = self.angular_v + self.angular_accels * self.dt

        new_positions = []
        for ii in range(3):
            if position[ii] <= self.lower_bounds[ii]:
                new_positions.append(self.lower_bounds[ii])
                self.done = True
            elif position[ii] > self.upper_bounds[ii]:
                new_positions.append(self.upper_bounds[ii])
                self.done = True
            else:
                new_positions.append(position[ii])

        self.pose = np.array(new_positions + list(angles))

        self.runtime += self.dt

        if self.runtime > self.T:
            self.done = True

        return self.done

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # print(rotor_speeds)
            self._next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self._get_reward_hover()/self.action_repeat
            # pose_all.append(self.pose[:3])
            distance = [(self.pose[0] - self.final_pos[0]),(self.pose[1] - self.final_pos[1]),(self.pose[2]- self.final_pos[2])]
            pose_all.append(np.concatenate((self.pose[3:], self.angular_v, distance, self.v,self.linear_accel), axis=0))
        next_state = np.concatenate(pose_all)
        return next_state, reward, self.done, {}

if __name__ == '__main__':
    drone = QuadRotorEnv()
    for i in range(56): #gain momentum
        print(drone.step([900.,900.,900.,900.]))
        if drone.done:
            break
        drone.render()
    print("==============================") # ~88% of path is done and then its good
    for i in range(75): # let glide
        print(drone.step([0.01,0.01,0.01,0.01]))
        if drone.done:
            break
        drone.render()
    print("++++++++++++++++++++++++++++++")   
    for i in range(75): # hover
        print(drone.step([430.,430.,430.,430.]))
        if drone.done:
            break
        drone.render()
    print(drone.pose)
