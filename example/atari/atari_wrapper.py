import importlib
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env import ShmemVectorEnv


def grayscale_v0(env):
    from supersuit.lambda_wrappers import observation_lambda_v0
    def change_obs(obs, obs_space):
        import cv2
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = obs.reshape(obs_space.shape)
        return obs

    def change_space(space):
        from supersuit.utils.basic_transforms import convert_box
        return convert_box(lambda obs: change_obs(obs, space), space)
        
    return observation_lambda_v0(env, change_obs, change_space)

def get_env(env_name, clip_reward=False, render_mode="rgb_array"):
    env_cls = getattr(importlib.import_module("pettingzoo.atari"), env_name)
    env = env_cls.env(render_mode=render_mode)
    import supersuit

    env = supersuit.max_observation_v0(env, 2)
    env = supersuit.frame_skip_v0(env, 4)
    env = grayscale_v0(env)  # TODO: check bug in grayscale
    env = supersuit.resize_v1(env, 84, 84)
    env = supersuit.frame_stack_v1(env, 4)
    if clip_reward:
        env = supersuit.clip_reward_v0(env)
        
    return PettingZooEnv(env)

def make_atari_env(env_name, seed, training_num, test_num=1):
    # ======== environment setup =========
    train_envs = ShmemVectorEnv([
        lambda: 
        get_env(env_name, True) for _ in range(training_num)
    ])
    test_envs = ShmemVectorEnv([
        lambda: 
        get_env(env_name, False) for _ in range(test_num)
    ])

    # seeding
    train_envs.seed(seed)
    test_envs.seed(seed)

    return train_envs, test_envs
    