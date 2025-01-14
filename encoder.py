import torch
import torch.nn as nn
import models_mae

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


### Works with Finism SAC
# OUT_DIM = {2: 39, 4: 35, 6: 31}
# OUT_DIM_84 = {2: 39, 4: 35, 6: 31}
# OUT_DIM_100 = {2: 39, 4: 43, 6: 31}

### Works with RAD SAC
OUT_DIM = {2: 39, 4: 35, 6: 31}
OUT_DIM_84 = {2: 29, 4: 35, 6: 21}
OUT_DIM_100 = {4: 43}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        # try 2 5x5s with strides 2x2. with samep adding, it should reduce 84 to 21, so with valid, it should be even smaller than 21.
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        if obs_shape[-1] == 100:
            assert num_layers in OUT_DIM_100
            out_dim = OUT_DIM_100[num_layers]
        elif obs_shape[-1] == 84:
            out_dim = OUT_DIM_84[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        if obs.max() > 1.:
            obs = obs / 255.

        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)

        return h

    def forward(self, obs, detach=False, mask_ratio=0.0):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder, 'vit': models_mae.mae_vit_base_patch12_dec64d4b}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    # assert encoder_type in _AVAILABLE_ENCODERS
    if 'vit' in encoder_type:
        if encoder_type == 'mae_vit_base_patch12_dec64d4b':
            return models_mae.mae_vit_base_patch12_dec64d4b(img_size=obs_shape[-1], in_chans=obs_shape[0],
                                                            latent_feature_dim=feature_dim)
        if encoder_type == 'mae_vit_large_patch12_h4d4_dec512d4b':
            return models_mae.mae_vit_large_patch12_h4d4_dec512d4b(img_size=obs_shape[-1], in_chans=obs_shape[0],
                                                                   latent_feature_dim=feature_dim)
        if encoder_type == 'mae_vit_large_patch12_h12d4_dec512d4b':
            return models_mae.mae_vit_large_patch12_h12d4_dec512d4b(img_size=obs_shape[-1], in_chans=obs_shape[0],
                                                                    latent_feature_dim=feature_dim)
        if encoder_type == 'mae_vit_small_patch6_h3d6_dec192d3b':
            return models_mae.mae_vit_small_patch6_h3d6_dec192d3b(img_size=obs_shape[-1], in_chans=obs_shape[0],
                                                                    latent_feature_dim=feature_dim)
        if encoder_type == 'mae_vit_small_patch12_h3d6_dec192d3b':
            return models_mae.mae_vit_small_patch12_h3d6_dec192d3b(img_size=obs_shape[-1], in_chans=obs_shape[0],
                                                                    latent_feature_dim=feature_dim)
    else:
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, num_layers, num_filters, output_logits
        )
