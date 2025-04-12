import math
from dataclasses import dataclass
from typing import Union, List
from timm.models.vision_transformer import Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pscan import pscan

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison.

- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


@dataclass
class MambaConfig:
    d_model: int  #  D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  #  N in paper/comments
    expand_factor: int = 2  #  E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  #  "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True  #  use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig, window=7):
        super().__init__()

        self.config = config
        self.window = window
        #  projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        self.d_inner = config.d_inner
        self.conv1d_0 = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                  kernel_size=config.d_conv, bias=config.conv_bias,
                                  groups=config.d_inner,
                                  padding=config.d_conv - 1)
        self.conv1d_1 = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                  kernel_size=config.d_conv, bias=config.conv_bias,
                                  groups=config.d_inner,
                                  padding=config.d_conv - 1)
        self.conv1d_2 = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                  kernel_size=1, bias=config.conv_bias,
                                  groups=config.d_inner,
                                  padding=0)
        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.x_proj_1 = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.x_proj_2 = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        #  x : (B, L, D)
        output = 0
        # y : (B, L, D)
        _, L, D = x.shape
        h, w = int(math.sqrt(L)), int(math.sqrt(L))
        # x = x.transpose(1, 2).reshape(-1, 576, 14, 14)
        # x = x.flatten(2).transpose(1, 2)
        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  #  (B, L, ED), (B, L, ED)
        _, _, ED = x.shape
        x, class_token = x[:, 0:L - 1, :], x[:, L - 1, :]
        x_ = x.transpose(1, 2).reshape(-1, ED, h, w)
        # window
        x1, x2, x3, x4 = x_[:, :ED, :self.window, :self.window], x_[:, :ED, :self.window, self.window:], \
                         x_[:, :ED, self.window:, :self.window], \
                         x_[:, :ED, self.window:, self.window:]
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x3 = x3.flatten(2).transpose(1, 2)
        x4 = x4.flatten(2).transpose(1, 2)

        all_win = [x1, x2, x3, x4]
        # 横竖扫描
        x_1 = x_.flatten(2).transpose(1, 2)
        x_2 = x_.transpose(2, 3).flatten(2).transpose(1, 2)
        class_token = class_token.reshape(class_token.shape[0], 1, class_token.shape[1])
        x_1 = torch.cat([x_1, class_token], dim=1)
        x_2 = torch.cat([x_2, class_token], dim=1)
        two_x = [x_1, x_2]
        z = F.silu(z)
        #  x branch
        # global
        for i in range(len(two_x)):
            x = two_x[i]
            x = x.transpose(1, 2)  #  (B, ED, L)
            x = eval('self.conv1d_' + str(i))(x)[:, :, :L]  #  depthwise convolution over time, with a short filter
            x = x.transpose(1, 2)  #  (B, L, ED)
            x = F.silu(x)
            y = self.ssm(x, count=i)

            if i == 1:
                y, class_token = y[:, 0:L - 1, :], y[:, L - 1:, :]
                y = y.transpose(1, 2).reshape(-1, ED, h, w)
                y = y.transpose(2, 3).flatten(2).transpose(1, 2)
                y = torch.cat([class_token, y], dim=1)
            else:
                y, class_token = y[:, 0:L - 1, :], y[:, L - 1:, :]
                y = torch.cat([class_token, y], dim=1)
            #  z branch

            output += y * z
        # window-base
        win = []
        for u in all_win:
            u = u.transpose(1, 2)
            u = self.conv1d_2(u)
            u = u.transpose(1, 2)  #  (B, L, ED)
            u = F.silu(u)
            u = self.ssm(u, count=2)
            u = u.transpose(1, 2).reshape(-1, self.config.d_model, self.window, self.window)
            win.append(u)
        temp1 = torch.cat((win[0], win[1]), dim=3)
        temp2 = torch.cat((win[2], win[3]), dim=3)
        temp = torch.cat((temp1, temp2), dim=2).reshape(temp1.shape[0], temp1.shape[1], -1).transpose(1, 2)
        y = torch.cat((temp, class_token), dim=1)
        output += y * z
        output = self.out_proj(output)  #  (B, L, D)
        return output

    def ssm(self, x, count):
        #  x : (B, L, ED)

        #  y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        #  TODO remove .float()
        if count == 0:
            deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)
        elif count == 1:
            deltaBC = self.x_proj_1(x)
        else:
            deltaBC = self.x_proj_2(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        """Does selective scan algorithm. See:
                   - Section 2 State Space Models in the Mamba paper [1]
                   - Algorithm 2 in Section 3.2 in the Mamba paper [1]
                   - run_SSM(A, B, C, u) in The Annotated S4 [2]

               This is the classic discrete state space formula:
                   x(t + 1) = Ax(t) + Bu(t)
                   y(t)     = Cx(t) + Du(t)
               except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

               Args:
                   u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
                   delta: shape (b, l, d_in)
                   A: shape (d_in, n)
                   B: shape (b, l, n)
                   C: shape (b, l, n)
                   D: shape (d_in,)

               Returns:
                   output: shape (b, l, d_in)

               Official Implementation:
                   selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
                   Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

               """
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        # y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  #  (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    #  -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs : (B, ED, d_conv-1)

        #  y : (B, D)
        #  cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  #  (B, ED), (B, ED)

        #  x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  #  (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  #  (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        #  x : (B, ED)
        #  h : (B, ED, N)

        #  y : (B, ED)
        #  h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  #  (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)

        h = deltaA * h + BX  #  (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        #  todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)


#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class MambaFusion(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        #  projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        #  projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        #  projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, tensors: List):
        #  x : (B, L, D)
        # y : (B, L, D)
        # batch_size, seq_len, feature_dim = tensors[0].shape
        x = torch.cat([tensor[:, i::4, :] for i in range(4) for tensor in tensors], dim=1)
        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  #  (B, L, ED), (B, L, ED)

        #  x branch
        x = x.transpose(1, 2)  #  (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  #  (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)
        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, L, D)

        # 获取交叉拼接后的张量的形状
        batch_size, total_length, features = x.shape
        tensor_length = total_length // 4
        restored_tensors = []

        for i in range(4):
            temp = output[:, i::4, :]
            temp = temp[:, :tensor_length, :]
            temp += tensors[i]
            restored_tensors.append(temp)

        return restored_tensors

    def ssm(self, x):
        #  x : (B, L, ED)

        #  y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        # delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  #  (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)

        delta = delta.transpose(1, 2)
        delta = F.softplus(delta + self.dt_proj.bias)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y


class SelectiveModule(nn.Module):
    def __init__(self, channels, hidden_channels, drop=0.,
                 tau=1., ratio_per_sample=True,
                 version=1, inference=False):
        super(SelectiveModule, self).__init__()
        self.inference = inference
        self.norm = nn.LayerNorm(channels)
        self.ratio_per_sample = ratio_per_sample
        self.tau = tau
        self.version = version  # version 0 includes CLS token when selecting, version 1 excludes CLS while selecting
        assert version in (0, 1)
        if self.version == 0 or self.version == 1:
            self.mlp = Mlp(channels, hidden_channels, 2, act_layer=nn.GELU, drop=drop)
        self.gate2 = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        B, L, C = x.shape
        if self.version == 1:
            x = x[:, 1:, :]
        x = self.norm(x)
        x = self.mlp(x)  # shape (B,L,1) or (B,L,2)
        scale = self.gate2(x)  # shape (B,L,1) or (B,L,2)
        if not self.inference:
            selector = F.gumbel_softmax(scale, tau=self.tau, hard=True)[:, :, 0:1]  # shape (B, L, 1)
        else:
            selector = torch.argmin(scale, dim=-1, keepdim=True)
        diff_selector = selector
        if self.version == 1:
            selector = torch.cat((torch.ones(B, 1, 1, device=selector.device), selector), dim=1).bool().squeeze(2)
        else:  # self.version = 0
            selector = selector.bool().squeeze(2)

        return selector, diff_selector


def _ratio_loss(ratio_per_sample, selector, ratio=1.):
    if ratio_per_sample is True:
        n_tokens = selector.shape[1]
        return ((selector.sum(dim=1) / n_tokens - ratio) ** 2).mean()
    else:
        return (selector.sum() / (selector.shape[0] * selector.shape[1]) - ratio) ** 2

# model = SelectiveModule(channels=576, hidden_channels=576 * 4,inference=True)
# # # model=MambaChannelFusion(MambaConfig(d_model=197,n_layers=1))
# tensor = torch.randn(1, 196 * 4, 576)
# x, y = model(tensor)
#
# tensor =tensor  * x.unsqueeze(-1)
#
# print(tensor)
# # tensor2 = torch.randn(1, 197, 576)
# # tensor3 = torch.randn(1, 197, 576)
# # tensor4 = torch.randn(1, 197, 576)
# # tensors = [tensor1, tensor2, tensor3, tensor4]
# y=model(tensor)
# print(y[0].shape)
