import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
import copy
import math

from encoder import make_encoder

import utils
import data_augs as rad
from multistep_dynamics import MultiStepDynamicsModel
import multistep_utils as mutils


LOG_FREQ = 10000

        
def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CustomDataParallel(DataParallel):
    def __init__(self, *args, **kwargs):
        super(CustomDataParallel, self).__init__(*args, **kwargs)

    def __getitem__(self, name):
        return getattr(self.module, name)

    @property
    def device(self):
        return self.device_ids[0] if self.device_ids is not None else None


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, action_shape, hidden_dim,
        encoder_feature_dim, encoder, log_std_min, log_std_max
    ):
        super().__init__()

        self.encoder = encoder

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False, mask_ratio=0.75
    ):
        obs = self.encoder(obs, mask_ratio=mask_ratio, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, action_shape, hidden_dim,
        encoder_feature_dim, encoder
    ):
        super().__init__()

        self.encoder = encoder

        self.Q1 = QFunction(
            encoder_feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            encoder_feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False, noencode=False):
        # detach_encoder allows to stop gradient propogation to encoder
        if noencode:
            pass
        else:
            obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class BaseSacAgent(object):
    """Learning Representations of Pixel Observations with SAC + Self-Supervised Techniques.."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        horizon,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_lr=1e-3,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        latent_dim=128,
        data_augs = '',
        use_metric_loss=False
    ):
        
        self.horizon = horizon
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.latent_dim = latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.data_augs = data_augs
        self.use_metric_loss = use_metric_loss

        self.augs_funcs = {}

        aug_to_func = {
                'crop':rad.random_crop,
                'grayscale':rad.random_grayscale,
                'cutout':rad.random_cutout,
                'cutout_color':rad.random_cutout_color,
                'flip':rad.random_flip,
                'rotate':rad.random_rotation,
                'rand_conv':rad.random_convolution,
                'color_jitter':rad.random_color_jitter,
                'translate':rad.random_translate,
                'no_aug':rad.no_aug,
            }

        for aug_name in self.data_augs.split('-'):
            assert aug_name in aug_to_func, 'invalid data aug string'
            self.augs_funcs[aug_name] = aug_to_func[aug_name]

        encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.actor = Actor(
            action_shape, hidden_dim, encoder_feature_dim, encoder,
            actor_log_std_min, actor_log_std_max,
        ).to(device)

        self.critic = Critic(
            action_shape, hidden_dim, encoder_feature_dim, encoder,
        ).to(device)

        self.critic_target = Critic(
            action_shape, hidden_dim, encoder_feature_dim, encoder,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self._device_ids = None
        self._output_device = None
        self._dim = None
        self._be_data_parallel = False

    def to_data_parallel(self, device_ids=None, output_device=None, axis=0):
        self._device_ids = device_ids
        self._output_device = output_device
        self._dim = axis
        self.recover_data_parallel()

    def to_normal(self):
        if self._be_data_parallel and self._device_ids is not None:
            self._be_data_parallel = False
            for module_name in dir(self):
                item = getattr(self, module_name)
                if isinstance(item, CustomDataParallel):
                    setattr(self, module_name, item.module)

    def recover_data_parallel(self):
        if self._device_ids is None:
            return
        self._be_data_parallel = True
        for module_name in dir(self):
            item = getattr(self, module_name)
            if isinstance(item, nn.Module):
                setattr(self, module_name, CustomDataParallel(module=item, device_ids=self._device_ids,
                                                              output_device=self._output_device, dim=self._dim))

    def is_data_parallel(self):
        return self._be_data_parallel

    # @property
    # def device(self):
    #     if self._device_ids is None or not self._be_data_parallel:
    #         return next(self.parameters()).device
    #     else:
    #         return self._device_ids[0]

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, mask_ratio=0.75):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            if self.is_data_parallel():
                mu, _, _, _ = self.actor.module(
                    obs, compute_pi=False, compute_log_pi=False, mask_ratio=mask_ratio
                )
            else:
                mu, _, _, _ = self.actor(
                    obs, compute_pi=False, compute_log_pi=False, mask_ratio=mask_ratio
                )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs, mask_ratio=0.75):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            if self.is_data_parallel():
                mu, pi, _, _ = self.actor.module(obs, compute_log_pi=False, mask_ratio=mask_ratio)
            else:
                mu, pi, _, _ = self.actor(obs, compute_log_pi=False, mask_ratio=mask_ratio)

            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            if self.is_data_parallel():
                _, policy_action, log_pi, _ = self.actor.module(next_obs)
                target_Q1, target_Q2 = self.critic_target.module(next_obs, policy_action)
            else:
                _, policy_action, log_pi, _ = self.actor(next_obs)
                target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        if self.is_data_parallel():
            current_Q1, current_Q2 = self.critic.module(
                obs, action, detach_encoder=self.detach_encoder)
        else:
            current_Q1, current_Q2 = self.critic(
                obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.is_data_parallel():
            self.critic.module.log(L, step)
        else:
            self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        if self.is_data_parallel():
            _, pi, log_pi, log_std = self.actor.module(obs, detach_encoder=True)
            actor_Q1, actor_Q2 = self.critic.module(obs, pi, detach_encoder=True)
        else:
            _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
            actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.is_data_parallel():
            self.actor.module.log(L, step)
        else:
            self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def save(self, model_dir, step):
        if self.is_data_parallel():
            torch.save(
                self.actor.module.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
            )
            torch.save(
                self.critic.module.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
            )
            torch.save(
                self.critic.module.encoder.state_dict(), '%s/critic_encoder_%s.pt' % (model_dir, step)
            )
        else:
            torch.save(
                self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
            )
            torch.save(
                self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
            )
            torch.save(
                self.critic.encoder.state_dict(), '%s/critic_encoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )