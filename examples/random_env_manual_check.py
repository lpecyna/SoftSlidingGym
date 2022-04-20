import os.path as osp
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize


parser = argparse.ArgumentParser(description='Process some integers.')
# ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
parser.add_argument('--env_name', type=str, default='RopeFollow')
parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
parser.add_argument('--img_size', type=int, default=2*256, help='Size of the recorded videos')

args = parser.parse_args()

env_kwargs = env_arg_dict[args.env_name]

# Generate and save the initial states for running this environment for the first time
env_kwargs['use_cached_states'] = False
env_kwargs['save_cached_states'] = False
env_kwargs['num_variations'] = args.num_variations
env_kwargs['render'] = True
env_kwargs['headless'] = args.headless

if not env_kwargs['use_cached_states']:
    print('Waiting to generate environment variations. May take 1 minute for each variation...')
env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
env.reset()
frames = [env.get_image(args.img_size, args.img_size)]
#action = np.zeros(3)
#action[0] = 0.1
#for i in range(5):
    # _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
    #obs, reward, done, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
    # print("Reward external:")
    # print(reward)


