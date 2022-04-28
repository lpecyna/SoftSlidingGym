#  from softgym.envs.pour_water import PourWaterPosControlEnv
#  from softgym.envs.pour_water_amount import PourWaterAmountPosControlEnv
#  from softgym.envs.pass_water import PassWater1DEnv
from softgym.envs.rope_flatten import RopeFlattenEnv
from softgym.envs.rope_configuration import RopeConfigurationEnv
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.envs.cloth_drop import ClothDropEnv
from softgym.envs.cloth_fold_crumpled import ClothFoldCrumpledEnv
from softgym.envs.cloth_fold_drop import ClothFoldDropEnv
from softgym.envs.rope_follow import RopeFollowEnv
from softgym.envs.cloth_follow import ClothFollowEnv

from collections import OrderedDict

env_arg_dict = {
    'PourWater': {'observation_mode': 'cam_rgb',
                  'action_mode': 'rotation_bottom',
                  'render_mode': 'fluid',
                  'deterministic': False,
                  'render': True,
                  'action_repeat': 8,
                  'headless': True,
                  'num_variations': 1000,
                  'horizon': 100,
                  'use_cached_states': True,
                  'camera_name': 'default_camera'},
    'PourWaterAmount': {'observation_mode': 'cam_rgb',
                        'action_mode': 'rotation_bottom',
                        'render_mode': 'fluid',
                        'action_repeat': 8,
                        'deterministic': False,
                        'render': True,
                        'headless': True,
                        'num_variations': 1000,
                        'use_cached_states': True,
                        'horizon': 100,
                        'camera_name': 'default_camera'},
    'RopeFlatten': {
        'observation_mode': 'cam_rgb',
        'action_mode': 'picker',
        'num_picker': 2,
        'render': True,
        'headless': True,
        'horizon': 75,
        'action_repeat': 8,
        'render_mode': 'cloth',
        'num_variations': 1000,
        'use_cached_states': True,
        'deterministic': False
    },
    'RopeConfiguration': {'observation_mode': 'cam_rgb',
                          'action_mode': 'picker',
                          'num_picker': 2,
                          'render': True,
                          'headless': True,
                          'horizon': 100,  # this task is harder than just straigtening rope, therefore has larger horizon.
                          'action_repeat': 8,
                          'render_mode': 'cloth',
                          'num_variations': 1000,
                          'use_cached_states': True,
                          'deterministic': False},
    'ClothFlatten': {'observation_mode': 'cam_rgb',
                     'action_mode': 'picker',
                     'num_picker': 2,
                     'render': True,
                     'headless': True,
                     'horizon': 100,
                     'action_repeat': 8,
                     'render_mode': 'cloth',
                     'num_variations': 1000,
                     'use_cached_states': True,
                     'deterministic': False},
    'ClothFlattenPPP': {'observation_mode': 'cam_rgb',
                        'action_mode': 'pickerpickplace',
                        'num_picker': 2,
                        'render': True,
                        'headless': True,
                        'horizon': 20,
                        'action_repeat': 1,
                        'render_mode': 'cloth',
                        'num_variations': 1000,
                        'use_cached_states': True,
                        'deterministic': False},
    'ClothFoldPPP': {'observation_mode': 'cam_rgb',
                     'action_mode': 'pickerpickplace',
                     'num_picker': 2,
                     'render': True,
                     'headless': True,
                     'horizon': 20,
                     'action_repeat': 1,
                     'render_mode': 'cloth',
                     'num_variations': 1000,
                     'use_cached_states': True,
                     'deterministic': False},
    'ClothFold': {'observation_mode': 'cam_rgb',
                  'action_mode': 'picker',
                  'num_picker': 2,
                  'render': True,
                  'headless': True,
                  'horizon': 100,
                  'action_repeat': 8,
                  'render_mode': 'cloth',
                  'num_variations': 1000,
                  'use_cached_states': True,
                  'deterministic': False},
    'ClothFoldCrumpled': {'observation_mode': 'cam_rgb',
                          'action_mode': 'picker',
                          'num_picker': 2,
                          'render': True,
                          'headless': True,
                          'horizon': 100,
                          'action_repeat': 8,
                          'render_mode': 'cloth',
                          'num_variations': 1000,
                          'use_cached_states': True,
                          'deterministic': False},
    'ClothFoldDrop': {'observation_mode': 'cam_rgb',
                      'action_mode': 'picker',
                      'num_picker': 2,
                      'render': True,
                      'headless': True,
                      'horizon': 100,
                      'action_repeat': 8,
                      'render_mode': 'cloth',
                      'num_variations': 1000,
                      'use_cached_states': True,
                      'deterministic': False},
    'ClothDrop': dict(observation_mode='cam_rgb',
                      action_mode='picker',
                      num_picker=2,
                      render=True,
                      headless=True,
                      horizon=30,
                      action_repeat=16,
                      render_mode='cloth',
                      num_variations=1000,
                      use_cached_states=True,
                      deterministic=False),
    'PassWater': dict(observation_mode='cam_rgb',
                      action_mode='direct',
                      render=True,
                      headless=True,
                      horizon=75,
                      action_repeat=8,
                      render_mode='fluid',
                      deterministic=False,
                      num_variations=1000),
    'PassWaterGoal': {
        "observation_mode": 'point_cloud',  # will be later wrapped by ImageEnv
        "horizon": 75,
        "action_mode": 'direct',
        "deterministic": False,
        "render_mode": 'fluid',
        "render": True,
        "headless": True,
        "action_repeat": 8,
        "num_variations": 1000,
    },
    "PourWaterGoal": {
        'observation_mode': 'point_cloud',
        'action_mode': 'direct',
        'render_mode': 'fluid',
        'deterministic': False,
        'render': True,
        'headless': True,
        'num_variations': 1000,
        'horizon': 100,
        'camera_name': 'default_camera'
    },
    "ClothManipulate": dict(
        observation_mode='point_cloud',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=True,
        horizon=100,
        action_repeat=8,
        render_mode='cloth',
        num_variations=1000,
        deterministic=False
    ),
    'RopeFollow': {
            'observation_mode': 'key_point',
            'action_mode': 'picker',
            'num_picker': 1,
            'render': True,
            'headless': True,
            'horizon': 150,# 100
            'action_repeat': 8,
            'render_mode': 'cloth',
            'num_variations': 1000,
            'use_cached_states': True,
            'deterministic': False
    },
    'ClothFollow': {'observation_mode': 'cam_rgb',
                          'action_mode': 'picker',
                          'num_picker': 1,
                          'render': True,
                          'headless': True,
                          'horizon': 100,
                          'action_repeat': 8,
                          'render_mode': 'cloth',
                          'num_variations': 1000,
                          'use_cached_states': True,
                          'deterministic': False},
}

SOFTGYM_ENVS = OrderedDict({
    #  'PourWater': PourWaterPosControlEnv,
    #  'PourWaterAmount': PourWaterAmountPosControlEnv,
    #  'PassWater': PassWater1DEnv,
    'ClothFlatten': ClothFlattenEnv,
    'ClothFold': ClothFoldEnv,
    'ClothDrop': ClothDropEnv,
    'ClothFoldDrop': ClothFoldDropEnv,
    'ClothFlattenPPP': ClothFlattenEnv,
    'ClothFoldPPP': ClothFoldEnv,
    'ClothFoldCrumpled': ClothFoldCrumpledEnv,
    'RopeFlatten': RopeFlattenEnv,
    'RopeConfiguration': RopeConfigurationEnv,
    'RopeFollow': RopeFollowEnv,
    'ClothFollow': ClothFollowEnv,
})
