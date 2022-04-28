import numpy as np
import pickle
import os.path as osp
import pyflex
from softgym.envs.rope_sliding_env import RopeNewEnv
from copy import deepcopy
from softgym.utils.pyflex_utils import random_pick_and_place2, center_object, center_first

class RopeFollowEnv(RopeNewEnv):
    def __init__(self, cached_states_path='rope_follow_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.prev_distance_diff = None
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        self.farthest = -1000
        self.stay_last10run = 0
        self.stay_last10 = 0
        self.indx = 0

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        max_wait_step = 1000  #500 Maximum number of steps waiting for the rope to stablize
        stable_vel_threshold = 0.001  #0.005 Rope stable when all particles' vel are smaller than this
        if config is None:
            config = self.get_default_config()
        default_config = config            
        for i in range(num_variations):
            config = deepcopy(default_config)
            config['segment'] = self.get_random_rope_seg_num()
            self.set_scene(config)

            self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            config['camera_params'] = deepcopy(self.camera_params)
            self.action_tool.reset([0., -1., 0.])
            curr_pos = pyflex.get_positions().reshape(-1, 4)
            #num_particles = curr_pos.shape[0]

            curr_pos[0, :3] += [0, 0.1, 0] #Holding one end higher
            curr_pos[0, 3] = 0

            pyflex.set_positions(curr_pos.flatten())
            center_first()#shifting appropriatly so that there is space to move right

            random_pick_and_place2(pick_num=4, pick_scale=0.0025)

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def get_random_rope_seg_num(self):
        return np.random.randint(30, 61)

    def _reset(self):
        self.farthest = -1000
        self.stay_last10run = 0
        self.stay_last10 = 0
        self.indx = 0

        config = self.current_config
        self.rope_length = config['segment'] * config['radius'] * 0.5

        # set reward range
        self.reward_max = self.rope_length#0
        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)

        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx = curr_pos[0][0]+0.25
            cy = curr_pos[0][0]
            self.action_tool.reset([cx, 0.1, cy])

        # set reward range
        self.reward_max = self.rope_length
        self.reward_min = 0
        self.reward_range = self.reward_max - self.reward_min

        self.prev_reward = 0
        self.prev_dist = 0
        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            config = self.get_current_config()
            self.action_tool.step(action, config)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def _get_endpoint_distance(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        p1, p2 = pos[0, :3], pos[-1, :3]
        return np.linalg.norm(p1 - p2).squeeze()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ Reward is the distance between the endpoints of the rope"""
        if np.size(self.sel_points) == 0:
            r = - 0.5
        else:
            self.indx = np.mean(self.sel_points)
            conf = self.get_current_config()
            #One:
            #r = self.indx*self.rope_length/conf['segment']

            #Second version:
            dist = self.indx * self.rope_length / conf['segment']
            r = (dist - self.prev_dist)*10
            #r = r - 0.1
            self.prev_dist = dist
            if self.indx > conf['segment']-20:
                r = r + (self.indx+20-conf['segment'])/20#*(indx+10-conf['segment'])/10
        self.prev_reward = r
        return r

    def _get_info(self):
        conf = self.get_current_config()
        reward = self.prev_reward
        self.farthest = max(self.farthest, -conf['segment'] + self.indx)
        if conf['segment'] - self.indx < 10:
            self.stay_last10run = self.stay_last10run + 1
        else:
            self.stay_last10run = 0
        self.stay_last10 = max(self.stay_last10run, self.stay_last10)

        end_type = [0, 0, 0, 0]
        if conf['segment'] - self.indx < 5:  # it got above 5
            if np.size(self.sel_points) == 0:
                # option 2 - get to the end but fall
                end_type[1] = 1
            else:
                # option 1 - get to the end and stay
                end_type[0] = 1
        else:
            if np.size(self.sel_points) != 0:
                # option 3
                end_type[2] = 1
            #else (option 4) - fall on the way


        return {
            'step_reward': reward,
            'stay_last10': self.stay_last10,
            'farthest': self.farthest,
            'internal_step': self.time_step,
            'get_end_stay': end_type[0],
            'get_end_fall': end_type[1],
            'stop_before': end_type[2]
        }
