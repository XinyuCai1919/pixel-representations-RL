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


class CNNEncoder(nn.Module):
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

        self.byol_project = nn.Sequential(
            nn.Linear(50, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, 96),
            nn.BatchNorm1d(96),
        )
        self.byol_predict = nn.Sequential(
            nn.Linear(96, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, 96),
        )

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

    def forward(self, obs, detach=False):
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

    def forward_0(self, obs, detach=False):
        return self.forward(obs, detach)

    def forward_rec(self, img_sequence):
        return self.forward(img_sequence)