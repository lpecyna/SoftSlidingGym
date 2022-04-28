import abc
import numpy as np
from gym.spaces import Box
# from softgym.utils.misc import rotation_2d_around_center, extend_along_center
import pyflex
# import scipy.spatial


class ActionToolBase(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self, state):
        """ Reset """

    @abc.abstractmethod
    def step(self, action):
        """ Step funciton to change the action space states. Does not call pyflex.step() """


class Picker(ActionToolBase):
    def __init__(self, num_picker=1, picker_size=(0.0125, 0.0075), init_pos=(0., -0.1, 0.), picker_threshold=0.004, particle_radius=0.008,
                 picker_low=(-0.4, 0., -0.4), picker_high=(0.4, 0.5, 0.4), init_particle_pos=None, spring_coef=1.2, **kwargs):
        """

        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        super(Picker).__init__()
        self.grasp = 0
        self.picker_size = picker_size
        self.picker_threshold = picker_threshold
        self.num_picker = num_picker
        self.picked_particles = [None] * self.num_picker
        self.picker_low, self.picker_high = np.array(list(picker_low)), np.array(list(picker_high))
        self.init_pos = init_pos
        self.particle_radius = particle_radius
        self.init_particle_pos = init_particle_pos
        self.spring_coef = spring_coef  # Prevent picker to drag two particles too far away

        # For velocity scaling:
        space_low = np.array([-0.025, -0.025, 0] * self.num_picker) * 0.1  # [dx,dz]
        space_high = np.array([0.025, 0.025, 10] * self.num_picker) * 0.1
        # Original from SoftGym:
        # space_low = np.array([-0.1, -0.1, -0.1, 0] * self.num_picker) * 0.1  # [dx, dy, dz, [0, 1]]
        # space_high = np.array([0.1, 0.1, 0.1, 10] * self.num_picker) * 0.1
        self.action_space = Box(space_low, space_high, dtype=np.float32)

    def update_picker_boundary(self, picker_low, picker_high):
        self.picker_low, self.picker_high = np.array(picker_low).copy(), np.array(picker_high).copy()

    def visualize_picker_boundary(self):
        halfEdge = np.array(self.picker_high - self.picker_low) / 2.
        center = np.array(self.picker_high + self.picker_low) / 2.
        quat = np.array([1., 0., 0., 0.])
        pyflex.add_box(halfEdge, center, quat)

    def _apply_picker_boundary(self, picker_pos):
        clipped_picker_pos = picker_pos.copy()
        for i in range(3):
            #clipped_picker_pos[i] = np.clip(picker_pos[i], self.picker_low[i] + self.picker_size[0], self.picker_high[i] - self.picker_size[1])
            clipped_picker_pos[i] = np.clip(picker_pos[i], self.picker_low[i], self.picker_high[i])
        return clipped_picker_pos

    def _get_centered_picker_pos(self, center):
        #r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.
        r = np.sqrt(self.num_picker - 1) * (self.picker_size[0] + self.picker_size[1]) * 2.
        pos = []
        for i in range(self.num_picker):
            x = center[0] + np.sin(2 * np.pi * i / self.num_picker) * r
            y = center[1]
            z = center[2] + np.cos(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, center):
        for i in (0, 2):
            self.picker_low[i] = -0.3#+= offset
            self.picker_high[i] = 0.3#+= offset
        init_picker_poses = [center, center]
        self.quat = [0, 0.70710677, 0, -0.70710677]
        for picker_pos in init_picker_poses:
            pyflex.add_capsule(self.picker_size, picker_pos, self.quat)

        pos = pyflex.get_shape_states()  # Need to call this to update the shape collision
        pyflex.set_shape_states(pos)

        self.picked_particles = [None] * self.num_picker
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_picker_pos = self._get_centered_picker_pos(center)

        curr_pos = pyflex.get_positions().reshape(-1, 4)

        if curr_pos[0, 1] > self.picker_size[1]:
            centered_picker_pos = curr_pos[0, 0:3] + [[-0.008, -0.019, 0], [-0.008, 0.0145, 0]]

        if curr_pos[0, 1] > self.picker_size[1]:
            new = [0, 0, 0, -1]
        else:
            new = self.quat
        for (i, centered_picker_pos) in enumerate(centered_picker_pos):
            shape_state[i] = np.hstack([centered_picker_pos, centered_picker_pos, [0, 0.70710677, 0, -0.70710677], [0, 0.70710677, 0, -0.70710677]])
            self.quat=new
        pyflex.set_shape_states(shape_state)
        self.particle_inv_mass = pyflex.get_positions().reshape(-1, 4)[:, 3]

    @staticmethod
    def _get_pos():
        """ Get the current pos of the pickers and the particles, along with the inverse mass of each particle """
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        return picker_pos[:, :3], particle_pos

    @staticmethod
    def _set_pos(picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)

    @staticmethod
    def set_picker_pos(picker_pos):
        """ Caution! Should only be called during the reset of the environment. Used only for cloth drop environment. """
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = picker_pos
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)

    def step(self, action, config):
        """ action = [translation, pick/unpick] * num_pickers.
        1. Determine whether to pick/unpick the particle and which one, for each picker
        2. Update picker pos
        3. Update picked particle pos
        """
        action = np.reshape(action, [-1, 3])  # if two pickers then [[x' y' z' h][x' y' z, h]]

        pick_flag = action[:, 2] > 0.5  # checking if grip (h) is higher than 0.5

        picker_pos, particle_pos = self._get_pos()
        new_picker_pos, new_particle_pos = picker_pos.copy(), particle_pos.copy()

        # Un-pick the particles
        for i in range(self.num_picker):
            if self.picked_particles[i] is not None:  # when it is not holding and there are some particles
                new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[self.picked_particles[i]]  # Revert the mass
                self.picked_particles[i] = None
        # Pick new particles and update the mass and the positions
        action_real = np.zeros([np.shape(action)[0], 3])
        action_real[:, 0] = action[:, 0]
        action_real[:, 1] = 0
        action_real[:, 2] = action[:, 1]
        self.grasp = action[0, 2]
        for i in range(self.num_picker):

            new_picker_pos[i, :] = self._apply_picker_boundary(picker_pos[i, :] + action_real[i, :3])
            new_picker_pos[i+1, :] = self._apply_picker_boundary(picker_pos[i+1, :] + action_real[i, :3])###########
            if pick_flag[i]:
                #print("YES Pick")
                if self.picked_particles[i] is None:
                    idx = self.get_hold_idx(picker_pos[i].reshape((-1, 3)), particle_pos[:, :3].reshape((-1, 3)))
                    #distance to axes:
                    dist_ax = abs(particle_pos[idx, 0]-picker_pos[i].reshape((-1, 3))[0, 0])

                    if idx.shape[0] > 0:
                        pick_id, pick_height = None, None
                        for j in range(idx.shape[0]):
                            if idx[j] not in self.picked_particles and idx[j] != 0 and (pick_id is None or dist_ax[j] < min_dist_ax):# should be one that is closest to the centre?
                                pick_id = idx[j]
                                min_dist_ax = dist_ax[j]
                        if pick_id is not None:
                            self.picked_particles[i] = int(pick_id)


                if self.picked_particles[i] is not None:
                    # TODO1 The position of the particle needs to be updated such that it is close to the picker particle
                    new_particle_pos[self.picked_particles[i], :3] = particle_pos[self.picked_particles[i], :3] + new_picker_pos[i, :] - picker_pos[i,
                                                                                                                                         :]
                    new_particle_pos[self.picked_particles[i], 3] = max(4.5 - 5 * max(0.5, action[0, 2]), 0)


        # check for e.g., rope, the picker is not dragging the particles too far away that violates the actual physicals constraints.
        """
        if self.picked_particles[0] is not None:
            i = self.picked_particles[0]
            violated = False
            if i > 0:
                now_distance = np.linalg.norm(new_particle_pos[i, :3] -
                                          new_particle_pos[i-1, :3])

                if now_distance > 2.5*0.5*config['radius']:
                    violated = True
            if i < config['segment']:
                now_distance = np.linalg.norm(new_particle_pos[i, :3] -
                                              new_particle_pos[i + 1, :3])

                if now_distance > 2.5*0.5*config['radius']:
                    violated = True
            if violated:
                new_picker_pos = picker_pos.copy()
                new_particle_pos[i, :3] = particle_pos[i, :3].copy()

        if self.init_particle_pos is not None:
            picked_particle_idices = []
            active_picker_indices = []
            for i in range(self.num_picker):
                if self.picked_particles[i] is not None:
                    picked_particle_idices.append(self.picked_particles[i])
                    active_picker_indices.append(i)

            l = len(picked_particle_idices)
            for i in range(l):
                for j in range(i + 1, l):
                    init_distance = np.linalg.norm(self.init_particle_pos[picked_particle_idices[i], :3] -
                                                   self.init_particle_pos[picked_particle_idices[j], :3])
                    now_distance = np.linalg.norm(new_particle_pos[picked_particle_idices[i], :3] -
                                                  new_particle_pos[picked_particle_idices[j], :3])
                    if now_distance >= init_distance * self.spring_coef:  # if dragged too long, make the action has no effect; revert it
                        new_picker_pos[active_picker_indices[i], :] = picker_pos[active_picker_indices[i], :].copy()
                        new_picker_pos[active_picker_indices[j], :] = picker_pos[active_picker_indices[j], :].copy()
                        new_particle_pos[picked_particle_idices[i], :3] = particle_pos[picked_particle_idices[i], :3].copy()
                        new_particle_pos[picked_particle_idices[j], :3] = particle_pos[picked_particle_idices[j], :3].copy()
        
        """
        self._set_pos(new_picker_pos, new_particle_pos)

    def get_hold_idx(self, picker_pos, particle_pos):
        p0 = picker_pos
        p1 = p0 + np.array([0, 0, self.picker_size[1]])  # if I change the orientation the added element need to be oriented
        p2 = p0 + np.array([0, 0, -self.picker_size[1]])  # if I change the orientation the added element need to be oriented
        p3 = particle_pos
        dists = np.linalg.norm(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1), axis=1)
        # Checking if the point projection is on the segment
        l2 = np.sum((p1 - p2) ** 2)
        t = np.sum((p3 - p1) * (p2 - p1), axis=1) / l2
        max_dist = self.picker_threshold + self.picker_size[0] + self.particle_radius
        mask1 = (t < np.ones(np.shape(t))) * (t > np.zeros(np.shape(t))) * (dists < max_dist* np.ones(np.shape(t)))
        mask2 = particle_pos[:, 1] > picker_pos[0][1] + self.picker_size[0]
        mask = np.logical_and(mask1, mask2)
        sel_points = np.array(range(np.shape(p3)[0]))[mask]
        return sel_points