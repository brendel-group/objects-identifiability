import numpy as np
import random
from spriteworld import environment as spriteworld_environment
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import tasks
import argparse
from spriteworld import renderers as spriteworld_renderers

def random_sprites_config(num_objects):
    """
    Computes config for spriteworld renderer
    Adapted from code given by Loic Matthey: https://gist.github.com/Azhag/f249b51b4cf3bfe568584f3a827708c1

    Args:
        num_objects: maximum number of objects in image

    Returns:
        config for spriteworld renderer
    """
    random.seed(0)
    factors = distribs.Product([
      distribs.Continuous('x', .1, .8),
      distribs.Continuous('y', .1, .8),
      distribs.Discrete('shape', ['square', 'triangle', 'circle']),
      distribs.Continuous('scale', .1, .15),
      distribs.Continuous('angle', 0, 0),
      distribs.Continuous('c0', 0., 1.),
      distribs.Continuous('c1', 3., 3.),
      distribs.Continuous('c2', 1., 1.),
    ])
    num_sprites = lambda: np.random.randint(2, num_objects+1)
    sprite_gen = sprite_generators.generate_sprites(factors, num_sprites=num_sprites)

    renderers = {
      'image':
          spriteworld_renderers.PILRenderer(
              image_size=(64, 64),
              anti_aliasing=5,
              color_to_rgb=spriteworld_renderers.color_maps.hsv_to_rgb,
          ),
      'attributes':
          spriteworld_renderers.SpriteFactors(
              factors=('x', 'y', 'shape', 'angle', 'scale', 'c0', 'c1', 'c2')),
    }

    config = {
      'task': tasks.NoReward(),
      'action_space': None,
      'renderers': renderers,
      'init_sprites': sprite_gen,
      'max_episode_length': 1,
    }
    return config


def collect_frames(config, max_objects, num_frames, shape_dict):
    """
    Instantiate config as environment and get single images from it.
    Adapted from code given by Loic Matthey: https://gist.github.com/Azhag/f249b51b4cf3bfe568584f3a827708c1

    Args:
        config: sprites config given by def random_sprites_config
        max_objects: maximum number of objects in image
        num_frames: number of samples in dataset
        shape_dict: dictionary specifying possible values for shapes in image

    Returns:
        arrays containing dataset of images, corresponding latents, and number of objects in each image
    """
    env = spriteworld_environment.Environment(**config)
    images = []
    num_obj = []
    Z = np.zeros((num_frames, max_objects, 5))
    for i in range(num_frames):
        print(i)
        ts = env.reset()
        for j in range(len(env._sprites)):
            Z[i, j, 0] = env._sprites[j].x
            Z[i, j, 1] = env._sprites[j].y
            Z[i, j, 2] = env._sprites[j].scale
            Z[i, j, 3] = env._sprites[j].c0
            Z[i, j, 4] = shape_dict[env._sprites[j].shape]

        obs = ts.observation['image']
        obs.insert(0, obs[len(obs)-1])
        del obs[-1]
        num_obj.append(len(obs) - 1)
        images.append(obs[0])
        
    return images, Z, num_obj

def gen_sprites(max_objects, num_obs):
    """
    Function to generate sprites dataset. Saves dataset of observations, latents, and number of objects in each image
    as a numpy array.
    Adapted from code given by Loic Matthey: https://gist.github.com/Azhag/f249b51b4cf3bfe568584f3a827708c1

    Args:
        max_objects: maximum number of objects in image
        num_obs: number of samples in dataset
    """
    data_name = str(max_objects) + "_obj_sprites"
    shape_dict = {"triangle": 1, "circle": 2, "square": 3}

    X, Z, num_obj = collect_frames(random_sprites_config(max_objects), max_objects, num_obs, shape_dict)
    np.savez_compressed("data/datasets/"+data_name, np.array(X), Z, num_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_objects", help="Maximum number of objects in each image", type=int, default="4")
    parser.add_argument("--nobs", help="Number of samples in dataset", type=int, default="100000")
    args = parser.parse_args()
    gen_sprites(max_objects=args.max_objects, num_obs=args.nobs)