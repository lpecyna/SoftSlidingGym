import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from copy import deepcopy
from softgym.envs.cloth_sliding_env import ClothSlidingEnv


class ClothFollowEnv(ClothSlidingEnv):
    def __init__(self, cached_states_path='cloth_follow_init_states.pkl', **kwargs):
        self.start_height = 0.8
        #kwargs['cached_states_path'] = 'cloth_follow_drop_init_states.pkl'
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        self.indx = 0

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [64, 32],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0, 0.4, 0.67]),#np.array([1.07199, 0.94942, 1.15691]),
                                   'angle': np.array([0, -20 / 180 * np.pi, 0]),#np.array([0.633549, -0.397932, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0
        }
        return config

    def generate_env_variation(self, num_variations=1, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 500  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.1  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']
            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])

            pickpoints = self._get_key_point_idx()[:1]  # Pick two corners of the cloth and wait until stablize

            config['target_pos'] = self._get_flat_pos()
            self._set_to_vertical(x_low=np.random.random() * 0.2 - 0.1, height_low=np.random.random() * 0.1 + 0.1)

            # Get height of the cloth without the gravity. With gravity, it will be longer
            p1, _, p2, _ = self._get_key_point_idx()
            # cloth_height = np.linalg.norm(curr_pos[p1] - curr_pos[p2])

            curr_pos = pyflex.get_positions().reshape(cloth_dimx, cloth_dimy, 4)
            #curr_pos = pyflex.get_positions().reshape(-1, 4)
            original_inv_mass = curr_pos[pickpoints, 3]
            #curr_pos[0] += np.random.random() * 0.001  # Add small jittering
            #curr_pos[0, :3] = [-0.25, 0.1, 0]  #Holding one corner
            #curr_pos[pickpoints, 3] = 0  # Set mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            sq_size = 7 #  square between two grippers
            x = np.array([i * self.cloth_particle_radius for i in range(sq_size)])-0.25
            y = np.array([-i * self.cloth_particle_radius for i in range(sq_size)])
            xx, yy = np.meshgrid(x, y)
            curr_pos[0:sq_size, 0:sq_size, 0] = xx
            curr_pos[0:sq_size, 0:sq_size, 1] = 0.1
            curr_pos[0:sq_size, 0:sq_size, 2] = yy
            inv_mass_org = curr_pos[0:sq_size, 0:sq_size, 3].copy()
            curr_pos[0:sq_size, 0:sq_size, 3] = 0
            curr_pos = curr_pos.reshape(-1,4)

            pickpoint_pos = curr_pos[pickpoints, :3]
            pyflex.set_positions(curr_pos.flatten())

            #picker_radius = self.action_tool.picker_radius
            #self.action_tool.update_picker_boundary([-0.3, 0.05, -0.5], [0.5, 2, 0.5])
            #self.action_tool.update_picker_boundary([-0.3, 0, -0.5], [0.5, 2, 0.5])
            #self.action_tool.set_picker_pos(picker_pos=pickpoint_pos + np.array([0., picker_radius, 0.]))


            # Pick up the cloth and wait to stablize
            for j in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold) and j > 300:
                    break
                curr_pos[pickpoints, :3] = pickpoint_pos
                pyflex.set_positions(curr_pos)

            self.action_tool.reset([0., 0.4, 0.])
            self.action_tool.update_picker_boundary([-0.3, 0, -0.5], [0.5, 2, 0.5])

            # Release from flat square
            curr_pos = pyflex.get_positions().reshape(cloth_dimx, cloth_dimy, 4)
            curr_pos[0:sq_size, 0:sq_size, 3] = inv_mass_org
            curr_pos[0, 0, 3] = 0
            pyflex.set_positions(curr_pos)
            for j in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold) and j > 300:
                    break
                curr_pos[pickpoints, :3] = pickpoint_pos
                pyflex.set_positions(curr_pos)


            # picker_radius = self.action_tool.picker_radius

            #curr_pos = pyflex.get_positions().reshape((-1, 4))
            #curr_pos[pickpoints, 3] = original_inv_mass
            #pyflex.set_positions(curr_pos.flatten())

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))

            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def _reset(self):
        """ Right now only use one initial state"""
        self.indx = 0
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_key_point_idx()[:2], :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)  # middle point is not really useful
            #picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0, -0.5], [0.5, 2, 0.5])
            #self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))

        """
        curr_pos = pyflex.get_positions().reshape(40, 40, 4)
        curr_pos[0:7, 0:7, 3] = curr_pos[8, 8, 3]
        curr_pos[0,0,3] = 0
        curr_pos = curr_pos.reshape(-1, 4)
        pyflex.set_positions(curr_pos.flatten())
        pyflex.step()
        """

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        return self._get_obs()

    def _set_to_vertical(self, x_low, height_low):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        vertical_pos = self._get_vertical_pos(x_low, height_low)
        curr_pos[:, :3] = vertical_pos
        max_height = np.max(curr_pos[:, 1])
        if max_height < 0.5:
            curr_pos[:, 1] += 0.5 - max_height
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _get_vertical_pos(self, x_low, height_low):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        x = np.array(list(reversed(x)))
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        # x = x - np.mean(x)
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = x_low
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = xx.flatten() - np.min(xx) + height_low
        return curr_pos

    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_b_init = self.init_pos[self.fold_group_b]
        group_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        fixation_dist = np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        performance = -group_dist - 1.2 * fixation_dist
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        info = {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (0. - performance_init),
            'neg_group_dist': -group_dist,
            'neg_fixation_dist': -fixation_dist
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def _step(self, action):
        config = self.get_current_config()
        self.action_tool.step(action, config)
        if self.action_mode in ['sawyer', 'franka']:
            print(self.action_tool.next_action)
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """
        The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
        particle in group a and the crresponding particle in group b
        :param pos: nx4 matrix (x, y, z, inv_mass)
        """
        """ Reward is the distance between the endpoints of the rope"""
        radius = 0.00625
        if np.size(self.sel_points) == 0:
            r = - 0.5
        else:
            config = self.get_current_config()
            cloth_dimx, cloth_dimy = config['ClothSize']
            self.indx = np.mean(self.sel_points)
            #One:
            #r = self.indx*self.rope_length/conf['segment']

            #Second version:
            dist = self.indx * radius / cloth_dimy  # self.indx * self.rope_length / cloth_dimy
            r = (dist - self.prev_dist)*10
            #r = r - 0.1
            self.prev_dist = dist
            if self.indx > cloth_dimy-20:
                r = r + (self.indx+20-cloth_dimy)/20#*(indx+10-conf['segment'])/10
        self.prev_reward = r
        return r
