import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_variant_env import FlexEnv
from softgym.action_space.action_space_sliding import Picker  # modified - using sliding action space
from softgym.action_space.robot_env import RobotBase
from copy import deepcopy


class RopeNewEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, horizon=75, render_mode='particle', picker_size=[0.02, 0.01, 0.04], **kwargs):
        self.render_mode = render_mode
        self.particle_radius = 0.004*2  # rope is half of that radius (see scale)
        super().__init__(**kwargs)

        assert observation_mode in ['point_cloud', 'cam_rgb', 'key_point']
        assert action_mode in ['picker', 'sawyer', 'franka']
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.num_picker = num_picker

        if action_mode == 'picker':
            self.action_tool = Picker(num_picker, picker_radius=0.02, picker_threshold=self.particle_radius/2,  # picker_size=picker_size
            particle_radius=self.particle_radius, picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))
            self.action_space = self.action_tool.action_space
        elif action_mode in ['sawyer', 'franka']:
            self.action_tool = RobotBase(action_mode)

        if observation_mode in ['key_point', 'point_cloud']:
            if observation_mode == 'key_point':
                obs_dim = 4+2  # 2+4  # This has to be modified when changing the inputs
            else:
                max_particles = 41
                obs_dim = max_particles * 3
                self.particle_obs_dim = obs_dim
            if action_mode in ['picker']:
                obs_dim += num_picker * 4  # 4 # This has to be modified when changing the inputs
            else:
                raise NotImplementedError
            self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

        self.horizon = horizon
        # print("init of rope new env done!")

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'init_pos': [0., 0., 0.],
            'stretchstiffness': np.random.uniform(0.8, 1.4), #0.9, #0.0005,#1.5,#0.9,
            'bendingstiffness': np.random.uniform(0.8, 2.4), #0.8, #0.8
            'radius': self.particle_radius,
            'segment': 30,#40,
            'mass': 0.5,
            'scale': 0.5,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0, 0.4, 0.67]),#[0, 1.4, 0]),# #[0, 0.85, 0]),
                                   'angle': np.array([0, -20 / 180 * np.pi, 0]),#[0, -90 / 180 * np.pi, 0]), #0 * np.pi, -90 / 180. * np.pi, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'friction': np.random.uniform(0.04, 0.3) #0.2
        }
        return config

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_height, self.camera_width)
        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
        elif self.observation_mode == 'key_point':
            #particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            #keypoint_pos = particle_pos[self.key_point_indices, :3]
            #pos = keypoint_pos.flatten()

            # more_info = np.array([self.rope_length, self._get_endpoint_distance()])
            # pos = np.hstack([more_info, pos])
            # print("in _get_obs, pos is: ", pos)

            # calculate new distance:
            picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3]
            particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
            p0 = picker_pos[0].reshape((-1, 3))
            p3 = particle_pos[:, :3].reshape((-1, 3))

            #OLD---------------------------------------------
            """
            p1 = p0 + np.array(
                [0, 0, self.action_tool.picker_size[1]])  # if I change the orientation the added element need to be oriented
            p2 = p0 + np.array(
                [0, 0, -self.action_tool.picker_size[1]])  # if I change the orientation the added element need to be oriented
            dists = np.linalg.norm(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1), axis=1)
            #Checking if the point projectsion is on the segment
            l2 = np.sum((p1 - p2) ** 2)
            t = np.sum((p3 - p1) * (p2 - p1), axis=1) / l2
            mask = (t < np.ones(np.shape(t))) * (t > np.zeros(np.shape(t))) * (dists < 0.031 * np.ones(np.shape(t)))
            #only this points that close and thair projectsions are inside the gripper
            # print(mask)
            sel_2D_points = p3[mask][:, [0, 2]]
            self.sel_points = np.array(range(np.shape(p3)[0]))[mask]
            """
            #-----------------------------------------------------------

            #NEW--------------------------------------------------------
            idx = self.action_tool.get_hold_idx(p0, p3) # NEW
            sel_2D_points = p3[idx][:, [0, 2]]
            self.sel_points = idx
            #-----------------------------------------------------------

            #print(mask)
            #print(self.sel_points)
            #print(sel_2D_points)
            if np.shape(sel_2D_points)[0] > 1:
                coef = np.polyfit(sel_2D_points[:, 0], sel_2D_points[:, 1], 1)
                yl = coef[0] * p0[0, 0] + coef[1]
                Y = yl - p0[0, 2] #+ 0.002*np.random.normal(0, 1-self.action_tool.grasp) #it was 0.01
                th = -np.arctan(coef[0]) #+ 0.5*np.random.normal(0, 1-self.action_tool.grasp)
                self.done = False
                #print("Position y: ", Y, ", Angle theta: ", th*180/np.pi)
            else:
                Y = 0
                th = 0
                self.done = True
                #print("DONE!....")
            #pos = np.concatenate((np.array([Y, th]), picker_pos[0], [self.action_tool.grasp]))

            # finding particle that is at the edge of the griper os its position is gripper position plus radius of gripper
            # and particle in x (0) direction. Nominal dystance

            #Vision:
            if self.done == True:
                full_vision = np.array([0, 0, 0, 0])
            else:
                P_tact_side1 = p0[0][0]+self.action_tool.picker_size[0] + 0.25*self.particle_radius
                conf = self.get_current_config()
                Edge1_idx = idx[-1]# + 1
                p_right_idx = Edge1_idx
                p_right2D = p3[p_right_idx][[0, 2]]
                for j in range(10):
                    curr_p = p3[idx[-1]+j, 0]
                    if curr_p-P_tact_side1 > 0 or idx[-1]+j+1 >= conf['segment']: # till the particle is on the side of the capsule
                        Edge1_idx = idx[-1]+j
                        break
                p_edge1 = p3[Edge1_idx][[0, 2]]
                if Edge1_idx+1 >= conf['segment']:
                    is_out = 0
                    out_pos1 = 0
                else:
                    is_out = 1
                    out_pos1 = p3[Edge1_idx+1][2] - p0[0, 2]
                for j in range(10):
                    p_right_idx = Edge1_idx+j
                    p_right2D = p3[p_right_idx][[0, 2]]
                    if p_right2D[0] - p_edge1[0] < -0.25*self.particle_radius or p_right_idx+1 >= conf['segment']: # if particles not going right (or not "almost" straight down)
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
                #print("Tactile Y: ", Y, "th: ", th)
                #print("full_vision: ", full_vision)


            # Vision before!!!!!!!!!!!!!!!:
            do_left_v = True
            """
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
                        # print("J: ", j, " and idx+j: ", idx[-1]+j, " of ",conf['segment'])
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
            """
            #pos = full_vision_l
            #pos = full_vision
            pos = np.concatenate((np.array([Y, th]), picker_pos[0], [self.action_tool.grasp], np.array(full_vision)))

        return pos

    def _get_key_point_idx(self, num=None):
        indices = [0]
        interval = (num - 2) // 8
        for i in range(1, 9):
            indices.append(i * interval)
        indices.append(num - 1)

        return indices

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        else:
            render_mode = 2

        camera_params = config['camera_params'][config['camera_name']]
        params = np.array(
            [*config['init_pos'], config['stretchstiffness'], config['bendingstiffness'], config['radius'], config['segment'], config['mass'],
                config['scale'], 
                *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], render_mode, config['friction']]
            )

        env_idx = 2

        if self.version == 2:
            robot_params = [1.] if self.action_mode in ['sawyer', 'franka'] else []
            self.params = (params, robot_params)
            pyflex.set_scene(env_idx, params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, params, 0)

        num_particles = pyflex.get_n_particles()
        # print("with {} segments, the number of particles are {}".format(config['segment'], num_particles))
        # exit()
        self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)

    def _get_info(self):
        return {}

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)



if __name__ == '__main__':
    env = RopeNewEnv(observation_mode='key_point',
                  action_mode='picker',
                  num_picker=2,
                  render=True,
                  headless=False,
                  horizon=75,
                  action_repeat=8,
                  num_variations=10,
                  use_cached_states=False,
                  save_cached_states=False,
                  deterministic=False)
    env.reset(config=env.get_default_config())
    for i in range(1000):
        print(i)
        print("right before pyflex step")
        pyflex.step()
        print("right after pyflex step")
        print("right before pyflex render")
        pyflex.render()
        print("right after pyflex render")
