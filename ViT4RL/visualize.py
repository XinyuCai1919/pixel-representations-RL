import argparse
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

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

def make_env(cfg, resource_files):
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
    def __init__(self, cfg):
        self.work_dir = cfg.model_dir
        self.eval_step = cfg.eval_step
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg, cfg.resource_files)

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)


    def run(self):
        self.agent.actor.load_state_dict(torch.load(self.work_dir + f"actor_{self.eval_step}-th_model.pth"))
        print("load pretrained model done")

        self.episode_rewards = []
        for i in range(1, 11):
            indoors_path = f"/home/xycai/pixel-representations-RL/ViT4RL/indoors_{i}.npy"


        self.episode_rewards = []
        for i in range(1, 11):
            outdoors_path = f"/home/xycai/pixel-representations-RL/ViT4RL/outdoors_{i}.npy"


@hydra.main(config_path='config_distract.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
