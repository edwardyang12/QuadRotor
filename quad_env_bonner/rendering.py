# from https://github.com/ngc92/quadgym/blob/master/gym_quadrotor/envs/rendering.py

#from gym_quadrotor.dynamics import coordinates
from TrajectoryGenerator import earth_to_body_frame, body_to_earth_frame
import numpy as np


class Renderer:
    def __init__(self):
        self.viewer = None
        self.center = None

        self.scroll_speed = 0.1
        self.objects = []

    def draw_line_2d(self, start, end):
        self.viewer.draw_line(start, end)

    def draw_line_3d(self, start, end):
        self.draw_line_2d((start[0], start[2]), (end[0], end[2]))

    def draw_circle(self, position, radius, color):  # pragma: no cover
        from gym.envs.classic_control import rendering
        copter = rendering.make_circle(radius)
        copter.set_color(*color)
        if len(position) == 3:
            position = (position[0], position[2])
        copter.add_attr(rendering.Transform(translation=position))
        self.viewer.add_onetime(copter)

    def add_object(self, new):
        self.objects.append(new)

    def set_center(self, new_center):
        # new_center is None => We are resetting.
        if new_center is None:
            self.center = None
            return

        # self.center is None => First step, jump to target
        if self.center is None:
            self.center = new_center

        # otherwise do soft update.
        self.center = (1.0 - self.scroll_speed) * self.center + self.scroll_speed * new_center
        if self.viewer is not None:
            self.viewer.set_bounds(-7 + self.center, 7 + self.center, -1, 13)

    def setup(self):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)

    def render(self, mode='human', close=False):
        if close:
            self.close()
            return

        if self.viewer is None:
            self.setup()

        for draw_ob in self.objects:  # type RenderedObject
            draw_ob.draw(self)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class RenderedObject:
    def draw(self, renderer: Renderer):
        raise NotImplementedError()


class Ground(RenderedObject):  # pragma: no cover
    def __init__(self, step_size=2):
        self._step_size = step_size

    def draw(self, renderer):
        """ Draws the ground indicator.
        """
        center = renderer.center
        renderer.draw_line_2d((-10 + center, 0.0), (10 + center, 0.0))
        pos = round(center / self._step_size) * self._step_size

        for i in range(-8, 10, self._step_size):
            renderer.draw_line_2d((pos + i, 0.0), (pos + i - 2, -2.0))


class QuadCopter(RenderedObject):  # pragma: no cover
    def __init__(self, source):
        self.source = source
        self._show_thrust = True

    def draw(self, renderer):
        status = self.source
        setup = self.source.max_rotor_speed

        # transformed main axis
        trafo = status.pose[3:]  # type: coordinates.Euler

        # draw current orientation
        rotated = np.dot(body_to_earth_frame(trafo[0], trafo[1], trafo[2]),[0, 0, 0.5])
        #rotated = coordinates.body_to_world(trafo, [0, 0, 0.5])
        renderer.draw_line_3d(status.pose[:3], status.pose[:3] + rotated)

        self.draw_propeller(renderer, trafo, status.pose[:3], [1, 0, 0], status.rotor_speeds[0] / setup)
        self.draw_propeller(renderer, trafo, status.pose[:3], [0, 1, 0], status.rotor_speeds[1] / setup)
        self.draw_propeller(renderer, trafo, status.pose[:3], [-1, 0, 0], status.rotor_speeds[2] / setup)
        self.draw_propeller(renderer, trafo, status.pose[:3], [0, -1, 0], status.rotor_speeds[3] / setup)

    @staticmethod
    def draw_propeller(renderer, euler, position, propeller_position, rotor_speed):
        structure_line = np.dot(body_to_earth_frame(euler[0], euler[1], euler[2]), propeller_position)
        #structure_line = coordinates.body_to_world(euler, propeller_position)
        renderer.draw_line_3d(position, position + structure_line)
        renderer.draw_circle(position + structure_line, 0.1, (0, 0, 0))
        thrust_line = np.dot(body_to_earth_frame(euler[0], euler[1], euler[2]), [0, 0, -0.5*rotor_speed**2])
        #thrust_line = coordinates.body_to_world(euler, [0, 0, -0.5*rotor_speed**2])
        renderer.draw_line_3d(position + structure_line, position + structure_line + thrust_line)

if __name__ == '__main__':

    renderer = Renderer()
    renderer.close()
    # renderer.setup()
    # renderer.draw_line_2d(1, 2)
    # renderer.draw_line_3d((1, 2, 3), (4, 5, 6))
    # renderer.close()
