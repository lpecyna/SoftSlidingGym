import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_variant_env import FlexEnv
from softgym.action_space.action_cloth_space_sliding import Picker  #, PickerPickPlace, PickerQPG
from softgym.action_space.robot_env import RobotBase
from copy import deepcopy


class ClothSlidingEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=1, render_mode='particle', picker_size=(0.0125, 0.0075), picker_radius=0.05, particle_radius=0.00625, **kwargs):
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.cloth_particle_radius = 0.004*2#particle_radius
        self.particle_radius = self.cloth_particle_radius
        super().__init__(**kwargs)

        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['picker', 'pickerpickplace', 'sawyer', 'franka', 'picker_qpg']
        self.observation_mode = observation_mode

        if action_mode == 'picker':
            #self.action_tool = Picker(num_picker, picker_radius=picker_radius, particle_radius=particle_radius,
                                      #picker_low=(-0.4, 0., -0.4), picker_high=(1.0, 0.5, 0.4))
            self.action_tool = Picker(num_picker, picker_size, picker_threshold=particle_radius / 2,
                                      # picker_size=picker_size
                                      particle_radius=particle_radius, picker_low=(-0.35, 0., -0.35),
                                      picker_high=(0.35, 0.3, 0.35))
            self.action_space = self.action_tool.action_space
            self.picker_radius = picker_radius
        elif action_mode == 'pickerpickplace':
            self.action_tool = PickerPickPlace(num_picker=num_picker, particle_radius=particle_radius, env=self,
                                               picker_low=(-0.5, 0., -0.5), picker_high=(0.5, 0.3, 0.5))
            self.action_space = self.action_tool.action_space
            assert self.action_repeat == 1
        elif action_mode in ['sawyer', 'franka']:
            self.action_tool = RobotBase(action_mode)
            self.action_space = self.action_tool.action_space
        elif action_mode == 'picker_qpg':
            cam_pos, cam_angle = self.get_camera_params()
            self.action_tool = PickerQPG((self.camera_height, self.camera_height), cam_pos, cam_angle,
                                         num_picker=num_picker, particle_radius=particle_radius, env=self,
                                         picker_low=(-0.3, 0., -0.3), picker_high=(0.3, 0.3, 0.3)
                                         )
            self.action_space = self.action_tool.action_space
        if observation_mode in ['key_point', 'point_cloud']:
            if observation_mode == 'key_point':
                obs_dim = 4+2#len(self._get_key_point_idx()) * 3
            else:
                max_particles = 120 * 120
                obs_dim = max_particles * 3
                self.particle_obs_dim = obs_dim
            if action_mode.startswith('picker'):
                obs_dim += num_picker * 4#3
            else:
                raise NotImplementedError
            self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

    def _sample_cloth_size(self):
        return 40, 40#np.random.randint(60, 120), np.random.randint(60, 120)

    def _get_flat_pos(self):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        x = x - np.mean(x)
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def get_camera_params(self):
        config = self.get_current_config()
        camera_name = config['camera_name']
        cam_pos = config['camera_params'][camera_name]['pos']
        cam_angle = config['camera_params'][camera_name]['angle']
        return cam_pos, cam_angle

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        particle_radius = self.cloth_particle_radius
        if self.action_mode in ['sawyer', 'franka']:
            cam_pos, cam_angle = np.array([0.0, 1.62576, 1.04091]), np.array([0.0, -0.844739, 0])
        else:
            cam_pos, cam_angle = np.array([0, 0.4, 0.67]), np.array([0, -20 / 180 * np.pi, 0])#np.array([-0.0, 0.82, 0.82]), np.array([0, -45 / 180. * np.pi, 0.])
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [int(0.6 / particle_radius), int(0.368 / particle_radius)],
            'ClothStiff': [1,1.6,1.2],#[0.8, 1, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0
        }

        return config

    def _get_obs(self):
        #print(self.observation_mode)
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_height, self.camera_width)
        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
            pos = keypoint_pos

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])

        #new:
        config = self.get_current_config()
        cloth_dimx, cloth_dimy = config['ClothSize']
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3]
        particle_pos = np.array(pyflex.get_positions()).reshape(cloth_dimx, cloth_dimy, 4)
        p0 = picker_pos[0].reshape((-1, 3))
        p3 = particle_pos[0, :, :3].reshape((-1, 3))
        #p3_1 = particle_pos[:, 0, :3].reshape((-1, 3))

        idx = self.action_tool.get_hold_idx(p0, p3)
        sel_2D_points = p3[idx][:, [0, 2]]
        self.sel_points = idx

        if np.shape(sel_2D_points)[0] > 1:
            coef = np.polyfit(sel_2D_points[:, 0], sel_2D_points[:, 1], 1)
            yl = coef[0] * p0[0, 0] + coef[1]
            Y = yl - p0[0, 2]  # + 0.002*np.random.normal(0, 1-self.action_tool.grasp) #it was 0.01
            th = -np.arctan(coef[0])  # + 0.5*np.random.normal(0, 1-self.action_tool.grasp)
            self.done = False
        else:
            Y = 0
            th = 0
            self.done = True

        # Vision:
        # finding particle that is at the edge of the griper os its position is gripper position plus radius of gripper
        # and particle in x (0) direction. Nominal distance
        if self.done == True:
            full_vision = np.array([0, 0, 0, 0])
        else:
            P_tact_side1 = p0[0][0]+self.action_tool.picker_size[0] + 0.25*self.particle_radius
            conf = self.get_current_config()
            Edge1_idx = idx[-1]
            p_right_idx = Edge1_idx
            p_right2D = p3[p_right_idx][[0, 2]]
            for j in range(10):
                curr_p = p3[idx[-1]+j, 0]
                if curr_p-P_tact_side1 > 0 or idx[-1]+j+1 >= cloth_dimy:# TO CHECK conf['segment']: # till the particle is on the side of the capsule
                    Edge1_idx = idx[-1]+j
                    break
            p_edge1 = p3[Edge1_idx][[0, 2]]
            if Edge1_idx+1 >= cloth_dimy:# TO CHECK #conf['segment']:
                is_out = 0
                out_pos1 = 0
            else:
                is_out = 1
                out_pos1 = p3[Edge1_idx+1][2] - p0[0, 2]
            for j in range(10):
                p_right_idx = Edge1_idx+j
                p_right2D = p3[p_right_idx][[0, 2]]
                if p_right2D[0] - p_edge1[0] < -0.25*self.particle_radius or p_right_idx+1 >= cloth_dimy:# TO CHECK#conf['segment']: # if particles not going right (or not "almost" straight down)
                    break
            if p_right2D[0] <= p_edge1[0]:
                ang1_conf = 0
                ang1 = 0
            else:
                ang1_conf = min(1, (p_right2D[0] - p_edge1[0])/(3*self.particle_radius))
                ang_2D_points = p3[Edge1_idx:p_right_idx][:, [0, 2]]
                coef_out1 = np.polyfit(ang_2D_points[:, 0], ang_2D_points[:, 1], 1)
                ang1 = -np.arctan(coef_out1[0])
            full_vision = np.array([is_out, out_pos1, ang1_conf, ang1])


        # Vision before!!!!!!!!!!!!!!!:
        do_left_v = False
        if do_left_v == True:
            if self.done == True:
                full_vision_l = np.array([0, 0, 0, 0])
            else:
                P_tact_side1_l = p0[0][0] - self.action_tool.picker_size[0] - 0.25 * self.particle_radius
                #should be half - that is position of particle at the edge but I assumeed less (so that not whole particle has to be out)
                conf = self.get_current_config()
                Edge1_idx_l = idx[0]  #Eddge - most right hold particle
                p_left_idx = Edge1_idx_l
                p_left2D = p3[p_left_idx][[0, 2]] # 2d (x,y) projection of most right hold point
                for j in range(10):
                    curr_p = p3[idx[0] - j, 0] # going to the right (10 steps) - getting x coordinate
                    # curr_p-P_edge1 # loop till this negative
                    if curr_p - P_tact_side1_l < 0 or idx[0] - j - 1 < 0:  # till the particle is on the side of the capsule (or final particle), if further than start - finding real edge
                        Edge1_idx_l = idx[0] - j
                        break
                p_edge1_l = p3[Edge1_idx_l][[0, 2]]

                if Edge1_idx_l - 1 < 0:
                    is_out_l = 0
                    out_pos1_l = 0
                else:
                    is_out_l = 1
                    out_pos1_l = p3[Edge1_idx_l - 1][2] - p0[0, 2] #first outside particle - its y position
                # print("EDGE: ", p_edge1)
                for j in range(10):
                    #finding 2d position of right particles if it is going (or almost going right) and if it is not the end of rope
                    p_left_idx = Edge1_idx_l - j
                    p_left2D = p3[p_left_idx][[0, 2]]
                    if p_edge1_l[0] - p_left2D[0] < -0.25 * self.particle_radius or p_left_idx - 1 < 0:
                        # if particles not going right (or not "almost" straight down)
                        break
                if p_left2D[0] >= p_edge1_l[0]:
                    ang1_conf_l = 0
                    ang1_l = 0
                else:
                    ang1_conf_l = min(1, (p_edge1_l[0] - p_left2D[0]) / (3 * self.particle_radius))
                    ang_2D_points_l = p3[p_left_idx:Edge1_idx_l][:, [0, 2]]
                    coef_out1_l = np.polyfit(ang_2D_points_l[:, 0], ang_2D_points_l[:, 1], 1)
                    ang1_l = -np.arctan(coef_out1_l[0])

                full_vision_l = np.array([is_out_l, out_pos1_l, ang1_conf_l, ang1_l])

        #print(np.concatenate((np.array([Y, th]), picker_pos[0], [self.action_tool.grasp], np.array(full_vision))))
        #print(full_vision)

        #print(np.array([Y, th*180/3.14]))
        pos = np.concatenate((np.array([Y, th]), picker_pos[0], [self.action_tool.grasp], np.array(full_vision)))
        return pos

    # Cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1

    def _get_key_point_idx(self):
        """ The keypoints are defined as the four corner points of the cloth """
        dimx, dimy = self.current_config['ClothSize']
        idx_p1 = 0
        idx_p2 = dimx * (dimy - 1)
        idx_p3 = dimx - 1
        idx_p4 = dimx * dimy - 1
        return np.array([idx_p1, idx_p2, idx_p3, idx_p4])

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3
        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 0 if 'env_idx' not in config else config['env_idx']
        mass = config['mass'] if 'mass' in config else 0.5
        mass = 320
        scene_params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], render_mode,
                                 *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], mass,
                                 config['flip_mesh']])
        if self.version == 2:
            robot_params = [1.] if self.action_mode in ['sawyer', 'franka'] else []
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, scene_params, 0)

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)