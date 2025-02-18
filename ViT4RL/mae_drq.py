import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
# from vit import Timm_Encoder_toy as Timm_Encoder
from models_mae import mae_vit_small_patch6_h3d6_dec192d3b, mae_vit_small_patch12_h3d6_dec192d3b
import utils
import hydra
import kornia.augmentation as aug

# timm_encoder = Timm_Encoder(obs_shape=[9, 84, 84], feature_dim=50)
cutout = aug.RandomErasing(scale=(0.05, 0.05), ratio=(1.0, 1.0), p=1)
# mae = mae_vit_small_patch6_h3d6_dec192d3b(img_size=84, in_chans=9, latent_feature_dim=50)
mae = mae_vit_small_patch12_h3d6_dec192d3b(img_size=84, in_chans=9, latent_feature_dim=50)

class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.encoder = mae
        self.encoder_env = encoder_cfg
        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(50, hidden_dim,
                               2 * action_shape[0], hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        loss, obs, pred, mask = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth):
        super().__init__()

        self.encoder = mae
        self.encoder_env = encoder_cfg
        self.Q1 = utils.mlp(50 + action_shape[0],
                            hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(50 + action_shape[0],
                            hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        loss, obs, pred, mask = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2, loss

    def log(self, logger, step):
        # self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class DRQAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""

    def __init__(self, obs_shape, action_shape, action_range, device,
                 encoder_cfg, critic_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.lr = lr

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        # self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr)

        self.log_alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_optimizer(self, scale=0.01):
        self.critic_optimizer = torch.optim.AdamW([{'params': self.critic.encoder.parameters(), 'lr': self.lr * scale},
                                                   {'params': self.critic.Q1.parameters(), 'lr': self.lr},
                                                   {'params': self.critic.Q2.parameters(), 'lr': self.lr}])

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, logger, step):
        # imgs = obs.detach()
        # aug_imgs = obs_aug.detach()
        # rec = self.critic_target.encoder.forward_rec(imgs)
        # rec = self.critic_target.encoder.byol_project(rec).detach()
        # aug_rec = self.critic.encoder.forward_rec(aug_imgs)
        # project_rec = self.critic.encoder.byol_project(aug_rec)
        # predict_rec = self.critic.encoder.byol_predict(project_rec)
        # rec_loss = F.mse_loss(predict_rec, rec)

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2, _ = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)
            target_Q1, target_Q2, _ = self.critic_target(next_obs_aug,
                                                      next_action_aug)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2, rec_loss = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug, _ = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q) # + rec_loss

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2, _ = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def save_model(self, path, step):
        torch.save(self.critic.state_dict(), path + "/critic_{}-th_model.pth".format(step))
        torch.save(self.actor.state_dict(), path + "/actor_{}-th_model.pth".format(step))
        torch.save(self.critic_target.state_dict(), path + "/critic_target_{}-th_model.pth".format(step))


