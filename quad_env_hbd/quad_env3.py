# from https://github.com/hbd730/quadcopter-simulation/blob/master/model/quadcopter.py

import numpy as np
import gym
from utils import RPYToRot, RotToQuat, RotToRPY
from quaternion import Quaternion
import scipy.integrate as integrate
import params

class QuadRotorEnv(gym.Env):
    def __init__(self, pos, attitude):
        """ pos = [x,y,z] attitude = [rool,pitch,yaw]
            """
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
        self.dt = 0.005
        self.T = 5

    def reset(self):
        self.state = np.zeros(13)

    def world_frame(self):
        """ position returns a 3x6 matrix
            where row is [x, y, z] column is m1 m2 m3 m4 origin h
            """
        origin = self.state[0:3]
        quat = Quaternion(self.state[6:10])
        rot = quat.as_rotation_matrix()
        wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
        quadBodyFrame = params.body_frame.T
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
        accel = 1.0 / params.mass * (wRb.dot(np.array([[0, 0, F]]).T)
                    - np.array([[0, 0, params.mass * params.g]]).T)
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
        pqrdot = params.invI.dot( M.flatten() - np.cross(omega, params.I.dot(omega)) )
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
        return 10

    def step(self, F, M):
        # limit thrust and Moment
        L = params.arm_length
        r = params.r
        prop_thrusts = params.invA.dot(np.r_[np.array([[F]]), M])
        prop_thrusts_clamped = np.maximum(np.minimum(prop_thrusts, params.maxF/4), params.minF/4)
        F = np.sum(prop_thrusts_clamped)
        M = params.A[1:].dot(prop_thrusts_clamped)
        self.state = integrate.odeint(self.state_dot, self.state, [0,self.dt], args = (F, M))[1]

        done = True

        return self.state, self._get_reward(), done

if __name__ == '__main__':
    pose = [9,9,9]
    attitude = [0,0,0]
    F = 5
    M = np.array([[2 * (1) + 3 * (2),
                   4 * (2) + 1 * (2),
                   3 * (2) + 4 * (8)]]).T

    drone = QuadRotorEnv(pose,attitude)
    print(drone.state)
    drone.step(F,M)
    print(drone.state)
