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


def make_env(cfg, resource_files):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        resource_files=resource_files,
        img_source=cfg.img_source,
        total_frames=1000,
        seed=cfg.seed,
        visualize_reward=False,
        from_pixels=True,
        height=cfg.image_size,
        width=cfg.image_size,
        frame_skip=cfg.action_repeat,
    )

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = cfg.work_dir
        self.eval_step = cfg.eval_step
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

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

        self.episode_rewards = []

    def evaluate(self, env):
        for episode in range(5):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = env.step(action)
                episode_reward += reward

            self.episode_rewards.append(episode_reward)

    def run(self):
        self.agent.actor.load_state_dict(torch.load(self.work_dir + f"/actor_{self.eval_step}-th_model.pth"))
        print("load pretrained model done")

        self.episode_rewards = []
        for i in range(1, 11):
            indoors_path = f"/home/xycai/pixel-representations-RL/distractors/indoors/video{i}.mp4"
            indoors_env = make_env(self.cfg, indoors_path)
            self.evaluate(indoors_env)
            indoors_result = np.array(self.episode_rewards)
            print(f"indoors_mean_{indoors_result.mean()}_std_{indoors_result.std()}")
            np.save(f"indoors_mean_{indoors_result.mean()}_std_{indoors_result.std()}", indoors_result)

        self.episode_rewards = []
        for i in range(1, 11):
            outdoors_path = f"/home/xycai/pixel-representations-RL/distractors/outdoors/video{i}.mp4"
            outdoors_env = make_env(self.cfg, outdoors_path)
            self.evaluate(outdoors_env)
            outdoors_result = np.array(self.episode_rewards)
            print(f"outdoors_mean_{outdoors_result.mean()}_std_{outdoors_result.std()}")
            np.save(f"outdoors_mean_{outdoors_result.mean()}_std_{outdoors_result.std()}", outdoors_result)


@hydra.main(config_path='config_distract.yaml', strict=True)
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
