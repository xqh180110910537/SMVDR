###--------- This code refers to https://github.com/pengzhiliang/Conformer ------###
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.non_local_simple_version import NONLocalBlock2D
from timm.models.layers import DropPath, trunc_normal_
from models.mamba import MambaFusion, MambaConfig, MambaBlock, SelectiveModule
from einops import rearrange
from models.moe import GatedMoE, entropy_regularization_loss
from models.feature_visualization import draw_feature_map


def CosineSimilarity(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


class mamba_Attention(nn.Module):
    def __init__(self, dim, proj_drop=0., expand_factor=1):
        super().__init__()
        config = MambaConfig(d_model=dim, n_layers=1, pscan=True, expand_factor=expand_factor)
        self.mamba = MambaBlock(config)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        #
        B, N, C = x.shape
        x = self.mamba(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_TopK(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)
        self.init_n = 14 * 14

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.keep_rate < 1:
            left_tokens = int(self.keep_rate * self.init_n)
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, index, idx

        return x, None, None


class Block_TopK(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_TopK(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                   keep_rate=keep_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, N, C = x.shape

        tmp, index, idx = self.attn(self.norm1(x))
        x = x + self.drop_path(tmp)

        if index is not None:
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]
            x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1
        if index is not None:
            return x, n_tokens, idx
        return x, n_tokens, None


class SSA(nn.Module):
    """
    Scalable Self-Attention, which scale spatial and channel dimension to obtain a better trade-off.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        sr_ratio (float): spatial reduction ratio, varied with stages.
        c_ratio (float): channel ratio, varied with stages.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, c_ratio=1.25, ):
        super(SSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.sr_ratio = sr_ratio
        self.c_new = int(dim * c_ratio)  # scaled channel dimension
        print(f'@ dim: {dim}, dim_new: {self.c_new}, c_ratio: {c_ratio}, sr_ratio: {sr_ratio}\n')

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        if self.sr_ratio > 1:
            self.q = nn.Linear(dim, self.c_new, bias=qkv_bias)
            self.reduction = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=int(sr_ratio), stride=int(sr_ratio), groups=dim),
                nn.Conv2d(dim, self.c_new, kernel_size=1, stride=1))
            self.norm_act = nn.Sequential(
                nn.LayerNorm(self.c_new),
                nn.GELU())
            self.k = nn.Linear(self.c_new, self.c_new, bias=qkv_bias)
            self.v = nn.Linear(self.c_new, dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x=x[:,:x.shape[1]-1,:]
        B, N, C = x.shape
        H, W = int(pow(N, 0.5)), int(pow(N, 0.5))
        if self.sr_ratio > 1:
            # reduction
            _x = x.permute(0, 2, 1).reshape(B, C, H, W)
            _x = self.reduction(_x).reshape(B, self.c_new, -1).permute(0, 2, 1)  # shape=(B, N', C')
            _x = self.norm_act(_x)
            # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(N, N', C)
            q = self.q(x).reshape(B, N, self.num_heads, int(self.c_new / self.num_heads)).permute(0, 2, 1, 3)
            k = self.k(_x).reshape(B, -1, self.num_heads, int(self.c_new / self.num_heads)).permute(0, 2, 1, 3)
            v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
        else:
            q = self.q(x).reshape(B, N, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, int(C / self.num_heads)).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 c_ratio=1.25,
                 sr_ratio=2, is_Mamba=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # #NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        if is_Mamba:
            self.attn = mamba_Attention(dim=dim)
        else:
            self.attn = SSA(dim=dim,
                            num_heads=num_heads,
                            qkv_bias=qkv_bias,
                            attn_drop=attn_drop,
                            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransMambaBlock(nn.Module):

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1, sr_ratio=1, is_Mamba=False):
        super(TransMambaBlock, self).__init__()
        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, sr_ratio=sr_ratio, is_Mamba=is_Mamba)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x_t):
        x_t = self.trans_block(x_t)
        x_tout = x_t
        return x_tout


class SMVDR(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_views=4):
        print('new-model! ')
        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0
        self.num_view = 4
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        # self.trans_cls_head = Late_Fusion(num_views, [embed_dim, 256, 128], num_classes)
        self.trans_cls_head = GatedMoE(embed_dim, 256, num_classes, num_experts=4, top_k=2, tau=1)
        # self.trans_cls_head = RCML(num_views, [embed_dim, 256, 128], num_classes)
        self.trans_cls_head_dan = \
            nn.Sequential(
                nn.Linear(embed_dim * num_views, 512),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(512, num_classes),
            )
        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        # self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, sr_ratio=1,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             is_Mamba=True
                             )

        # 2~4 stage
        init_stage = 2
        fin_stage = 6
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            TransMambaBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, is_Mamba=True
                            )
                            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = 6  # 6
        fin_stage = 9  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            TransMambaBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block,
                            )
                            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)

        for i in range(0, self.num_view):
            s = 2
            in_channel = stage_2_channel
            res_conv = True
            last_fusion = True
            self.add_module('mv_models_' + str(i),
                            TransMambaBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1], sr_ratio=1,
                                num_med_block=num_med_block, last_fusion=last_fusion
                            )
                            )

        self.fin_stage = fin_stage
        self.jointLayer = jointLayer(stage_3_channel)
        trunc_normal_(self.cls_token, std=.02)
        self.fusion = MambaFusion(MambaConfig(d_model=embed_dim, n_layers=1))
        self.key_select = SelectiveModule(channels=embed_dim * num_views, hidden_channels=embed_dim * num_views,
                                          inference=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def joint(self, x):
        x = rearrange(x, '(b v) c h w -> b v c h w', v=4)
        b, v, c, h, w = x.shape
        arr = x
        arr = arr.view(b, c, 2 * h, 2 * w)
        arr[:, :, :h, :w] = x[:, 0, :, :, :]
        arr[:, :, :h, w:2 * w] = x[:, 1, :, :, :]
        arr[:, :, h:2 * h, :w] = x[:, 2, :, :, :]
        arr[:, :, h:2 * h, w:2 * w] = x[:, 3, :, :, :]
        return arr

    def _add(self, x):
        #
        x = rearrange(x, '(b v) c e -> b c (v e)', v=4)
        return x

    def forward(self, x):
        B = x.shape[0]
        b = B // self.num_view
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))  # [N, 64, 56, 56]
        x_t = self.trans_patch_conv(x_base)  # [B*4, 576, 14, 14]

        x_t = x_t.flatten(2).transpose(1, 2)  # [B*4, 196, 576]

        x_t = torch.cat([cls_tokens, x_t], dim=1)

        x_t = self.trans_1(x_t)

        for i in range(2, self.fin_stage):
            # x_t = x_t.transpose(1, 2).reshape(-1, 576, 14, 14)
            x_t = eval('self.conv_trans_' + str(i))(x_t)
            # u = rearrange(x_t, '(b v) c e -> b v c e', v=4)
            # u = u[0:1, 0, 1:, :].reshape(1, -1, 14, 14)
            # draw_feature_map(u,name=f'{i}')
        x_t = rearrange(x_t, '(b v) c e -> b v c e', v=4)
        # zan=torch.stack(mv_x_t, 1)

        mv_x_t = []
        for i in range(0, self.num_view):
            sv_x_t = eval('self.mv_models_' + str(i))(x_t[:, i])
            mv_x_t.append(sv_x_t)

        mv_x_t = rearrange(torch.stack(mv_x_t, 1), 'b v c e -> b c (v e)')
        # selector
        selector, diff_se = self.key_select(mv_x_t)
        mv_x_y = mv_x_t * selector.unsqueeze(-1) + mv_x_t
        mv_x_y = rearrange(mv_x_y, 'b c (v e) -> b v c e', v=4)
        # mambafusion
        mv_x_1 = self.fusion([mv_x_y[:, i, :, :] for i in range(self.num_view)])
        x_t = torch.stack(mv_x_1, 1)
        # mvmoe
        output, expert_probabilities = self.trans_cls_head([x_t[:, i, 0] for i in range(self.num_view)])

        return output, expert_probabilities, selector


class jointLayer(nn.Module):
    def __init__(self, in_channels=1536):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding="same", bias=True)
        self.NONLocalBlock2D = NONLocalBlock2D(in_channels=in_channels, sub_sample=True)

    def joint(self, x):
        x = rearrange(x, '(b v) c h w -> b v c h w', v=4)
        b, v, c, h, w = x.shape
        arr = x.clone()
        arr = arr.view(b, c, 2 * h, 2 * w)
        arr[:, :, :h, :w] = x[:, 0, :, :, :]
        arr[:, :, :h, w:2 * w] = x[:, 1, :, :, :]
        arr[:, :, h:2 * h, :w] = x[:, 2, :, :, :]
        arr[:, :, h:2 * h, w:2 * w] = x[:, 3, :, :, :]
        return arr

    def forward(self, x):
        x = self.joint(x)
        x = self.NONLocalBlock2D(x)

        return x
