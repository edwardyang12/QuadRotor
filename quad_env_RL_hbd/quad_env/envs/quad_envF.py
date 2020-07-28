# from https://github.com/hbd730/quadcopter-simulation/blob/master/model/quadcopter.py

import numpy as np
from math import sin, cos
from numpy import linalg as LA
import gym
from gym import spaces
from quad_env.envs.utils import RPYToRot, RotToQuat, RotToRPY
from quad_env.envs.quaternion import Quaternion
import scipy.integrate as integrate
from quad_env.envs.params import *
from quad_env.envs.Trajectory import get_helix_waypoints, get_MST_coefficients, DesiredState, get_poly_cc
from quad_env.envs.quadPlot import set_limit, plot_waypoints
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

k_d_x = 30
k_p_x = 3
k_d_y = 30
k_p_y = 3
k_p_z = 1000
k_d_z = 200
k_p_phi = 160
k_d_phi = 3
k_p_theta = 160
k_d_theta = 3
k_p_psi = 80
k_d_psi = 5

class QuadRotorEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.v = 1.2
        self.T = 2
        self.animation_frequency = 50
        self.control_frequency = 200 # Hz for attitude control loop
        self.control_iterations = self.control_frequency / self.animation_frequency
        self.dt = 1.0 / self.control_frequency #0.1

        self.action_space = spaces.Box(
            low = np.array([-1000]),
            high = np.array([1000]),
            dtype=np.float32
        )
        # pos (x,y,z), vel, attitude, omega
        high = np.array([0.5,0.5,8,2,2,5,1.5,1.5,1.5,1.5,100,100,100],
                        dtype=np.float32)
        low = np.array([-0.5,-0.5,-0.5,-2,-2,-5,-1.5,-1.5,-1.5,-1.5,-100,-100,-100],
                        dtype=np.float32)
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
        set_limit((-0.5,0.5), (-0.5,0.5), (-0.5,8))
        plot_waypoints(self.waypoints)
        ax = plt.gca()
        self.lines = ax.get_lines()

    def reset(self):
        plt.close()
        """ pos = [x,y,z] attitude = [rool,pitch,yaw]
            """
        pos = (0.5,0.0,0.0)
        attitude = (0.0,0.0,0.0)
        self.state = np.zeros(13)
        roll, pitch, yaw = attitude
        rot    = RPYToRot(roll, pitch, yaw)
        quat   = RotToQuat(rot)
        self.state[0] = pos[0]
        self.state[1] = pos[1]
        self.state[2] = pos[2]
        self.state[6] = quat[0]
        self.state[7] = quat[1]
        self.state[8] = quat[2]
        self.state[9] = quat[3]
        self.xList = []
        self.yList = []
        self.zList = []
        self.time = 0
        self.waypoints = get_helix_waypoints(0, 9)
        self.coeff_x, self.coeff_y, self.coeff_z = get_MST_coefficients(self.waypoints)
        self.done = False
        self.yaw = 0.0
        self.current_heading = np.zeros(2)
        self.setupGraph()
        self.trajectory(self.v)
        return self.state

    def world_frame(self):
        """ position returns a 3x6 matrix
            where row is [x, y, z] column is m1 m2 m3 m4 origin h
            """
        origin = self.state[0:3]
        quat = Quaternion(self.state[6:10])
        rot = quat.as_rotation_matrix()
        wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
        quadBodyFrame = body_frame.T
        quadWorldFrame = wHb.dot(quadBodyFrame)
        world_frame = quadWorldFrame[0:3]
        return world_frame

    def position(self):
        return self.state[0:3]

    def velocity(self):
        return self.state[3:6]

    def attitude(self):
        rot = Quaternion(self.state[6:10]).as_rotation_matrix()
        return RotToRPY(rot)

    def omega(self):
        return self.state[10:13]

    def state_dot(self, state, t, F, M):
        x, y, z, xdot, ydot, zdot, qw, qx, qy, qz, p, q, r = state
        quat = np.array([qw,qx,qy,qz])

        bRw = Quaternion(quat).as_rotation_matrix() # world to body rotation matrix
        wRb = bRw.T # orthogonal matrix inverse = transpose
        # acceleration - Newton's second law of motion
        accel = 1.0 / mass * (wRb.dot(np.array([[0, 0, F]]).T)
                    - np.array([[0, 0, mass * g]]).T)
        # angular velocity - using quternion
        # http://www.euclideanspace.com/physics/kinematics/angularvelocity/
        K_quat = 2.0; # this enforces the magnitude 1 constraint for the quaternion
        quaterror = 1.0 - (qw**2 + qx**2 + qy**2 + qz**2)
        qdot = (-1.0/2) * np.array([[0, -p, -q, -r],
                                    [p,  0, -r,  q],
                                    [q,  r,  0, -p],
                                    [r, -q,  p,  0]]).dot(quat) + K_quat * quaterror * quat;

        # angular acceleration - Euler's equation of motion
        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        omega = np.array([p,q,r])
        pqrdot = invI.dot( M.flatten() - np.cross(omega, I.dot(omega)) )
        state_dot = np.zeros(13)
        state_dot[0]  = xdot
        state_dot[1]  = ydot
        state_dot[2]  = zdot
        state_dot[3]  = accel[0]
        state_dot[4]  = accel[1]
        state_dot[5]  = accel[2]
        state_dot[6]  = qdot[0]
        state_dot[7]  = qdot[1]
        state_dot[8]  = qdot[2]
        state_dot[9]  = qdot[3]
        state_dot[10] = pqrdot[0]
        state_dot[11] = pqrdot[1]
        state_dot[12] = pqrdot[2]

        return state_dot

    def _get_reward(self):
        reward = np.tanh(1 - 0.0005*(abs(self.state[:3] - np.array(self.des_state.pos))).sum())
        return reward

    def step(self, F):
        F= [F]
        self.trajectory(self.v)
        x, y, z = self.position()
        x_dot, y_dot, z_dot = self.velocity()
        phi, theta, psi = self.attitude()
        p, q, r = self.omega()

        des_x, des_y, des_z = self.des_state.pos
        des_x_dot, des_y_dot, des_z_dot = self.des_state.vel
        des_x_ddot, des_y_ddot, des_z_ddot = self.des_state.acc
        des_psi = self.des_state.yaw
        des_psi_dot = self.des_state.yawdot

        #print("pos {}".format(des_state.pos))
        #print("vel {}".format(des_state.vel))
        #print("acc {}".format(des_state.acc))
        #print("yaw {}".format(des_state.yaw))
        #print("yawdot {}".format(des_state.yawdot))
        # Commanded accelerations
        commanded_r_ddot_x = des_x_ddot + k_d_x * (des_x_dot - x_dot) + k_p_x * (des_x - x)
        commanded_r_ddot_y = des_y_ddot + k_d_y * (des_y_dot - y_dot) + k_p_y * (des_y - y)

        # Moment
        p_des = 0
        q_des = 0
        r_des = des_psi_dot
        des_phi = 1 / g * (commanded_r_ddot_x * sin(des_psi) - commanded_r_ddot_y * cos(des_psi))
        des_theta = 1 / g * (commanded_r_ddot_x * cos(des_psi) + commanded_r_ddot_y * sin(des_psi))

        self.M = np.array([[k_p_phi * (des_phi - phi) + k_d_phi * (p_des - p),
                       k_p_theta * (des_theta - theta) + k_d_theta * (q_des - q),
                       k_p_psi * (des_psi - psi) + k_d_psi * (r_des - r)]]).T


        # limit thrust and Moment
        L = arm_length
        prop_thrusts = invA.dot(np.r_[np.array([[F]]), self.M])
        prop_thrusts_clamped = np.maximum(np.minimum(prop_thrusts, maxF/4), minF/4)
        F = np.sum(prop_thrusts_clamped)
        self.M = A[1:].dot(prop_thrusts_clamped)
        self.state = integrate.odeint(self.state_dot, self.state, [0,self.dt], args = (F,self.M))[1]

        self.time += self.dt

        if(self.time>self.T):
            self.done = True

        return self.state, self._get_reward(), self.done, {}

    def trajectory(self, v):
        """ The function takes known number of waypoints and time, then generates a
        minimum snap trajectory which goes through each waypoint. The output is
        the desired state associated with the next waypont for the time t.
        waypoints is [N,3] matrix, waypoints = [[x0,y0,z0]...[xn,yn,zn]].
        v is velocity in m/s
        """

        t = self.time
        waypoints = self.waypoints
        coeff_x = self.coeff_x
        coeff_y = self.coeff_y
        coeff_z = self.coeff_z
        yaw = self.yaw
        current_heading = self.current_heading

        yawdot = 0.0
        pos = np.zeros(3)
        acc = np.zeros(3)
        vel = np.zeros(3)

        # distance vector array, represents each segment's distance
        distance = waypoints[0:-1] - waypoints[1:]
        # T is now each segment's travel time
        T = (1.0 / v) * np.sqrt(distance[:,0]**2 + distance[:,1]**2 + distance[:,2]**2)
        # accumulated time
        S = np.zeros(len(T) + 1)
        S[1:] = np.cumsum(T)

        # find which segment current t belongs to
        t_index = np.where(t >= S)[0][-1]

        # prepare the next desired state
        if t == 0:
            pos = waypoints[0]
            t0 = get_poly_cc(8, 1, 0)
            self.current_heading = np.array([coeff_x[0:8].dot(t0), coeff_y[0:8].dot(t0)]) * (1.0 / T[0])
        # stay hover at the last waypoint position
        elif t > S[-1]:
            pos = waypoints[-1]
        else:
            # scaled time
            scale = (t - S[t_index]) / T[t_index]
            start = 8 * t_index
            end = 8 * (t_index + 1)

            t0 = get_poly_cc(8, 0, scale)
            pos = np.array([coeff_x[start:end].dot(t0), coeff_y[start:end].dot(t0), coeff_z[start:end].dot(t0)])

            t1 = get_poly_cc(8, 1, scale)
            # chain rule applied
            vel = np.array([coeff_x[start:end].dot(t1), coeff_y[start:end].dot(t1), coeff_z[start:end].dot(t1)]) * (1.0 / T[t_index])

            t2 = get_poly_cc(8, 2, scale)
            # chain rule applied
            acc = np.array([coeff_x[start:end].dot(t2), coeff_y[start:end].dot(t2), coeff_z[start:end].dot(t2)]) * (1.0 / T[t_index]**2)

            # calculate desired yaw and yaw rate
            next_heading = np.array([vel[0], vel[1]])
            # angle between current vector with the next heading vector
            delta_psi = np.arccos(np.dot(current_heading, next_heading) / (LA.norm(current_heading)*LA.norm(next_heading)))
            # cross product allow us to determine rotating direction
            norm_v = np.cross(current_heading,next_heading)

            if norm_v > 0:
                self.yaw += delta_psi
            else:
                self.yaw -= delta_psi

            # dirty hack, quadcopter's yaw range represented by quaternion is [-pi, pi]
            if self.yaw > np.pi:
                self.yaw = self.yaw - 2*np.pi

            # print next_heading, current_heading, "yaw", yaw*180/np.pi, 'pos', pos
            self.current_heading = next_heading
            yawdot = delta_psi / 0.005 # dt is control period
        self.des_state =  DesiredState(pos, vel, acc, self.yaw, yawdot)

    def render(self, mode='human', close=False):
        if not close:
            frame = self.world_frame()
            lines_data = [frame[:,[0,2]], frame[:,[1,3]], frame[:,[4,5]]]
            for line, line_data in zip(self.lines[:3], lines_data):
                x, y, z = line_data
                line.set_data(x, y)
                line.set_3d_properties(z)
            x,y,z = frame[:,4]
            self.xList.append(x)
            self.yList.append(y)
            self.zList.append(z)
            self.lines[-1].set_data(np.array(self.xList), np.array(self.yList))
            self.lines[-1].set_3d_properties(np.array(self.zList))
            self.fig.canvas.draw()
            plt.pause(0.0001)

if __name__ == '__main__':
    F = 5
    M = np.array([[2 * (1) + 3 * (2),
                   4 * (2) + 1 * (2),
                   3 * (2) + 4 * (8)]]).T

    drone = QuadRotorEnv()
    print(drone.state)
    drone.step(F,M)
    print(drone.state)
