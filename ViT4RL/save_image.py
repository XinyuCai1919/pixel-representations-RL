import copy
import math
import os
import numpy as np

import dmc2gym
import hydra
import torch
import utils
from logger import Logger

torch.backends.cudnn.benchmark = True

os.environ['MUJOCO_GL'] = 'osmesa'

def make_env(resource_files):
    """Helper function to create dm_control environment"""
    env = dmc2gym.make(
        domain_name='walker',
        task_name='walk',
        resource_files=resource_files,
        img_source='video',
        total_frames=100,
        seed=23,
        visualize_reward=False,
        from_pixels=True,
        height=64,
        width=64,
        frame_skip=4,
    )

    env = utils.FrameStack(env, k=3)

    env.seed(23)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self):
        utils.set_seed_everywhere(23)

    def run(self):
        self.episode_rewards = []
        for i in range(1, 11):
            indoors_path = f"/home/xycai/pixel-representations-RL/distractors/indoors/video{i}.mp4"
            indoors_env = make_env(indoors_path)
            obs = indoors_env.reset()
            np.save(f"./indoors_64_{i}.npy", obs)

        self.episode_rewards = []
        for i in range(1, 11):
            outdoors_path = f"/home/xycai/pixel-representations-RL/distractors/outdoors/video{i}.mp4"
            outdoors_env = make_env(outdoors_path)
            obs = outdoors_env.reset()
            np.save(f"./outdoors_64_{i}.npy", obs)

def main():
    workspace = Workspace()
    workspace.run()


if __name__ == '__main__':
    main()
