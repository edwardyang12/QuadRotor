# referenced from https://github.com/bonn0062/quadcopter2/blob/master/task.py

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from quad_env.envs.TrajectoryGenerator import earth_to_body_frame, body_to_earth_frame
from quad_env.envs.quadPlot import set_limit, plot_waypoints
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

        self.divides = 1

        self.T = 5.
        
        self.action_repeat = 1

        self.dt = 1/50.0

        self.dims = np.array([width, length, height])  # x, y, z dimensions of quadcopter
        self.areas = np.array([length * height, width * height, width * length])

        self.moments_of_inertia = np.array([I_x, I_y, I_z])  # moments of inertia

        self.rotor_speeds = [0,0,0,0]
        self.max_rotor_speed = 900.
        self.action_space = spaces.Box(
            # rotor speeds of each propeller
            low = np.array([-self.max_rotor_speed,-self.max_rotor_speed,-self.max_rotor_speed, -self.max_rotor_speed]),
            high = np.array([self.max_rotor_speed, self.max_rotor_speed, self.max_rotor_speed, self.max_rotor_speed]),
            dtype=np.float32
        )

        self.lower_bounds = np.array([-env_bounds / 2, -env_bounds / 2, 0])
        self.upper_bounds = np.array([env_bounds / 2, env_bounds / 2, env_bounds])
        # pos (x,y,z), distance, vel
        high = np.concatenate([np.array([7., 7.,7., 30., 30., 30., 45., 5., 10., 100.,100.,100.],
                        dtype=np.float32)] * self.action_repeat)
        low = np.concatenate([np.array([-7., -7.,-7., -30., -30., -30., -5., -45., -40.,-100.,-100.,-100.],
                        dtype=np.float32)] *self.action_repeat)
        self.observation_space = spaces.Box(
            low = low,
            high = high,
            dtype=np.float32
        )

        self.reset()

    def setupGraph(self):
        self.fig = plt.figure()
        ax = self.fig.add_axes([0, 0, 1, 1], projection='3d')
        ax.plot([], [], [], '-', c='cyan')[0]
        ax.plot([], [], [], '-', c='red')[0]
        ax.plot([], [], [], '-', c='blue', marker='o', markevery=2)[0]
        ax.plot([], [], [], '.', c='red', markersize=4)[0]
        ax.plot([], [], [], '.', c='blue', markersize=2)[0]
        set_limit((-env_bounds/2,env_bounds/2), (-env_bounds/2,env_bounds/2), (0,env_bounds))
        plot_waypoints(np.array(self.traj_path))
        ax = plt.gca()
        self.lines = ax.get_lines()

    def render(self, mode='human', close=False):
        if not close:
            x,y,z = self.pose[:3]
            self.xList.append(x)
            self.yList.append(y)
            self.zList.append(z)
            self.lines[-1].set_data(self.xList, self.yList)
            self.lines[-1].set_3d_properties(self.zList)
            self.fig.canvas.draw()
            plt.pause(0.01)

    def get_trajectory(self):
        self.path_index = 1
        self.traj_path = []

        T = int(self.divides)
        dx = (self.final_pos[0] - self.pose[0])/self.divides
        dy = (self.final_pos[1] - self.pose[1])/self.divides
        dz = (self.final_pos[2] - self.pose[2])/self.divides
        tempx = self.pose[0]
        tempy = self.pose[1]
        tempz = self.pose[2]
        for i in range(T+1):
            self.traj_path.append([tempx,tempy,tempz])
            tempx += dx
            tempy += dy
            tempz += dz


    def reset(self):
##        self.final_pos = [0.,0.,40.]
        self.final_pos = [-20.,20.,40.]
##        x = random.uniform(-env_bounds / 2, env_bounds / 2)
##        y = random.uniform(-env_bounds / 2, env_bounds / 2)
##        z = random.uniform(20, env_bounds)
##        self.final_pos = np.array([x,y,z])
        plt.close()
        # position + angular position (x, y, z, roll, pitch, yaw)
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

        self.get_trajectory()
        self.setupGraph()

        # return np.concatenate([self.pose[:3]] * self.action_repeat )
        distance = [(self.pose[0] - self.traj_path[self.path_index][0]),(self.pose[1] - self.traj_path[self.path_index][1]),(self.pose[2]- self.traj_path[self.path_index][2])]
        return np.concatenate([np.concatenate((self.pose[3:],self.angular_v,distance,self._find_body_velocity()), axis=0)] * self.action_repeat)

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
            J = V / n * D
            #print(V, J, self.pose)
            # From http://m-selig.ae.illinois.edu/pubs/BrandtSelig-2011-AIAA-2011-1255-LRN-Propellers.pdf
            C_T = max(.12 - .07*max(0, J)-.1*max(0, J)**2, 0)
            thrusts.append(C_T * rho * n**2 * D**4)
        return thrusts

    def _get_reward_hover(self):
        reward = 0
       
        xrewardpos = np.e**(-1.5*np.abs(self.pose[0]- np.array(self.traj_path[self.path_index][0]))/20)
        yrewardpos = np.e**(-1.5*np.abs(self.pose[1]- np.array(self.traj_path[self.path_index][1]))/20)
        zrewardpos = np.e**(-1.5*np.abs(self.pose[2]- np.array(self.traj_path[self.path_index][2]))/40)
        rewardpos = xrewardpos*0.3 + yrewardpos*0.3 + zrewardpos * 0.4
        rewardangle = -np.linalg.norm(self.pose[3:]/7)/np.sqrt(3) + 1
        rewardvel = -np.linalg.norm(self._find_body_velocity()/50)/np.sqrt(3)+1
        rewardangular =  -np.linalg.norm(self.angular_v/30)/np.sqrt(3)+ 1
        reward = rewardpos*0.8 + rewardangle*0.1 + rewardvel*0.05 + rewardangular*0.05
        return reward
            
    def _get_reward_target(self):
        """Uses current pose of sim to return reward."""
        reward = 0.
##        reward = np.tanh(1 - 1.2/np.sqrt(3)*(np.linalg.norm((self.pose[:3] - np.array(self.traj_path[self.path_index]))/env_bounds)))
##        reward = (1 + self.path_index)* np.tanh(1 - 0.0005*(abs((self.pose[:3] - np.array(self.traj_path[self.path_index])))).sum())
##        reward = -(np.linalg.norm(self.pose[:3] - np.array(self.traj_path[self.path_index])))/env_bounds/np.sqrt(3)
##        for index, x in enumerate(self.passed):
##            if x<self.path_index:
##                # x
##                reward += 2**x/(np.linalg.norm((np.array(self.traj_path[self.path_index]) - np.array(self.reachedpos[index]))/env_bounds)/np.sqrt(3))
##            else:
##                reward += 2**x/(np.linalg.norm((self.pose[:3] - np.array(self.traj_path[self.path_index]))/env_bounds)/np.sqrt(3))
        
        if(self._reached()):
            reward = 1.

##        elif (self.pose[2]- np.array(self.traj_path[self.path_index][2]))>0:
##            xrewardpos = np.e**(-3.5*np.abs(self.pose[0]- np.array(self.traj_path[self.path_index][0]))/env_bounds)
##            yrewardpos = np.e**(-3.5*np.abs(self.pose[1]- np.array(self.traj_path[self.path_index][1]))/env_bounds)
##            rewardpos = xrewardpos*0.5 + yrewardpos*0.5
##            #rewardvel = -(np.linalg.norm(self._find_body_velocity()))/np.sqrt(3)/50 + 1
##            reward = rewardpos#* 0.1 + rewardvel* 0.9
            
        else:
##            xrewardpos = -np.e**(3.5*np.abs(self.pose[0]- np.array(self.traj_path[self.path_index][0]))/env_bounds-3.5)+1
##            yrewardpos = -np.e**(3.5*np.abs(self.pose[1]- np.array(self.traj_path[self.path_index][1]))/env_bounds-3.5)+1
            zrewardpos = np.e**(1.5*-np.abs(self.pose[2]- np.array(self.traj_path[self.path_index][2]))/40)
            xrewardpos = np.e**(-1.5*np.abs(self.pose[0]- np.array(self.traj_path[self.path_index][0]))/45)
            yrewardpos = np.e**(-1.5*np.abs(self.pose[1]- np.array(self.traj_path[self.path_index][1]))/45)
##            zrewardpos = -np.e**(3.5*np.abs(self.pose[2]- np.array(self.traj_path[self.path_index][2]))/40-3.5) + 1

##            xrewardpos = -np.abs(self.pose[0]- np.array(self.traj_path[self.path_index][0]))/env_bounds + 1
##            yrewardpos = -np.abs(self.pose[1]- np.array(self.traj_path[self.path_index][1]))/env_bounds + 1
##            zrewardpos = -np.abs(self.pose[2]- np.array(self.traj_path[self.path_index][2]))/env_bounds + 1
            rewardpos = xrewardpos*0.4 + yrewardpos*0.4 + zrewardpos * 0.2
            # rewardpos = -(np.linalg.norm(self.pose[:3] - np.array(self.traj_path[self.path_index])))/env_bounds/np.sqrt(3) + 1
            # rewardangle = -(np.linalg.norm(self.pose[3:]))/np.sqrt(3)/7 + 1
            # rewardvel = -(np.linalg.norm(self._find_body_velocity()))/np.sqrt(3)/50 + 1
            reward = rewardpos #* 0.9 + rewardvel* 0.1
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

        self._nearest_traj()
        self.runtime += self.dt

        if self.runtime > self.T:
            self.done = True

        # comment out if doing hover
        if self.path_index == int(self.divides) and self._reached():
            print("Reached end of path")
            self.done = True
       
        return self.done

    # computes nearest point further along than initial point
    def _nearest_traj(self):
        closest = self.path_index
        sum = np.linalg.norm(self.pose[:3] - np.array(self.traj_path[closest]))/np.sqrt(3)
        for index, coord in enumerate(self.traj_path):
            compare = np.linalg.norm(self.pose[:3] - np.array(coord))/np.sqrt(3)
            if((compare - sum)<2.5 and index > self.path_index):
                closest = index
                sum = compare
        if(np.linalg.norm(self.pose[:3] - np.array(self.traj_path[closest]))/np.sqrt(3) > 2.5):
            self.get_trajectory()
        else:
            self.path_index = closest

    # computes whether drone has reached target point in trajectory
    def _reached(self):
        sum = np.linalg.norm(self.pose[:3] - np.array(self.traj_path[self.path_index]))/np.sqrt(3)
        if(sum<2.5): # "1" is euclidian distance away ARBITRARY NUMBER
            return True
        else:
            return False


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            self._next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self._get_reward_target()/self.action_repeat
            # pose_all.append(self.pose[:3])
            distance = [(self.pose[0] - self.traj_path[self.path_index][0]),(self.pose[1] - self.traj_path[self.path_index][1]),(self.pose[2]- self.traj_path[self.path_index][2])]
            pose_all.append(np.concatenate((self.pose[3:], self.angular_v, distance, self._find_body_velocity()), axis=0))
        next_state = np.concatenate(pose_all)
        return next_state, reward, self.done, {}

if __name__ == '__main__':
    drone = QuadRotorEnv()
    for i in range(1000):
        print(drone.step([-900,-900,900,900]))
        if drone.done:
            break
        drone.render()
    print(drone.pose)
