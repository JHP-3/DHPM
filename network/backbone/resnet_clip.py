import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import Tuple, Union

import os
import warnings
import urllib
import hashlib
from tqdm import tqdm
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from itertools import combinations
import random

_tokenizer = _Tokenizer()

model_urls = {
    'RN50': 'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64,
                 replace_stride_with_dilation=[False, False, True]):
        super().__init__()

        self.input_resolution = input_resolution
        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.dilation = 1
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])  # True for OS=16

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks - 1):
            layers.append(Bottleneck(self._inplanes, planes, dilation=previous_dilation))
        layers.append(Bottleneck(self._inplanes, planes, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x, trunc1=False, trunc2=False, trunc3=False, trunc4=False, get1=False, get2=False, get3=False,
                get4=False):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        if not trunc1 and not trunc2 and not trunc3 and not trunc4:
            x = x.type(self.conv1.weight.dtype)
            x = stem(x)
            x = self.layer1(x)
            if get1:
                return x
        if not trunc2 and not trunc3 and not trunc4:
            x = self.layer2(x)
            if get2:
                return x
        if not trunc3 and not trunc4:
            x = self.layer3(x)
            if get3:
                return x
        if not trunc4:
            x = self.layer4(x)
            if get4:
                return x
        x = self.attnpool(x)

        return x


class CLIP_encoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 replace_stride_with_dilation: list,
                 ):
        super().__init__()

        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width,
            replace_stride_with_dilation=replace_stride_with_dilation
        )

        self.initialize_parameters()

    def initialize_parameters(self):

        for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, input, trunc1, trunc2, trunc3, trunc4, get1, get2, get3, get4):
        return self.visual(input.type(self.dtype), trunc1, trunc2, trunc3, trunc4, get1, get2, get3, get4)


class TextEncoder(nn.Module):
    def __init__(self, clip_model, n_cls):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_cls = n_cls

    def forward(self, prompts_list, tokenized_prompts_lists):

        for idx in range(len(prompts_list)):
            prompts = prompts_list[idx]
            tokenized_prompts = tokenized_prompts_lists[idx]

            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

            if idx == 0:
                final_feat = x.unsqueeze(0)
            else:
                final_feat = torch.concat([final_feat, x.unsqueeze(0)], dim=0)

        return final_feat


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = False
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        n_att = 4
        self.n_att = n_att

        # shape, color, material, function
        att1_text = "snow"
        att2_text = "snow-covered roads"
        att3_text = "winter"
        att4_text = "piloting a vehicle in snowy terrain"
        att5_text = "lighting effects"
        att_list = [att1_text, att2_text, att3_text, att4_text, att5_text]
        self.all_combinations = []
        for r in range(1, len(att_list) + 1):
            if r > 2:
                combos = combinations(att_list, r)
                self.all_combinations.extend(combos)
        random.shuffle(self.all_combinations)
        all_comb_len = len(self.all_combinations)
        self.att_weight = torch.tensor(torch.rand(all_comb_len, dtype=dtype).cuda(), requires_grad=True)

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        # coop没有用到初始化
        else:
            # random initialization
            # if cfg.TRAINER.COOP.CSC:  # usually false
            # print("Initializing class-specific contexts")
            # ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            # else:
            # print("Initializing a generic context")
            # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(prompts)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "middle"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim) [1, 512]
                    ctx,  # (n_cls, n_ctx, dim) [16, 512]
                    suffix,  # (n_cls, *, dim) [17:77, 512]
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                print(f'name len is {name_len}')
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "attribute":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                print(f'name len is {name_len}')
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        class_i,  # (1, name_len, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompt_list = self.prompt_learner.tokenized_prompt_list
        self.image_encoder = clip_model.visual
        self.n_cls = self.prompt_learner.n_cls
        self.text_encoder = TextEncoder(clip_model, self.n_cls)
        self.logit_scale = clip_model.logit_scale
        self.att_weight = self.prompt_learner.att_weight
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        att_weight = F.softmax(self.att_weight.float(), dim=0)

        prompts_list = self.prompt_learner()

        tokenized_prompts_list = self.tokenized_prompt_list
        text_features = self.text_encoder(prompts_list, tokenized_prompts_list)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 32, 512
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 15,50,512

        logit_scale = self.logit_scale.exp()
        logits = 0
        for idx in range(text_features.size(0)):
            logits += att_weight[idx] * logit_scale * image_features @ (text_features[idx, :, :]).t()

        return logits


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


to_remove = ["positional_embedding", "text_projection", "logit_scale", "input_resolution", "context_length",
             "vocab_size", "transformer.resblocks.0.attn.in_proj_weight", "transformer.resblocks.0.attn.in_proj_bias",
             "transformer.resblocks.0.attn.out_proj.weight", "transformer.resblocks.0.attn.out_proj.bias",
             "transformer.resblocks.0.ln_1.weight", "transformer.resblocks.0.ln_1.bias",
             "transformer.resblocks.0.mlp.c_fc.weight", "transformer.resblocks.0.mlp.c_fc.bias",
             "transformer.resblocks.0.mlp.c_proj.weight", "transformer.resblocks.0.mlp.c_proj.bias",
             "transformer.resblocks.0.ln_2.weight", "transformer.resblocks.0.ln_2.bias",
             "transformer.resblocks.1.attn.in_proj_weight", "transformer.resblocks.1.attn.in_proj_bias",
             "transformer.resblocks.1.attn.out_proj.weight", "transformer.resblocks.1.attn.out_proj.bias",
             "transformer.resblocks.1.ln_1.weight", "transformer.resblocks.1.ln_1.bias",
             "transformer.resblocks.1.mlp.c_fc.weight", "transformer.resblocks.1.mlp.c_fc.bias",
             "transformer.resblocks.1.mlp.c_proj.weight", "transformer.resblocks.1.mlp.c_proj.bias",
             "transformer.resblocks.1.ln_2.weight", "transformer.resblocks.1.ln_2.bias",
             "transformer.resblocks.2.attn.in_proj_weight", "transformer.resblocks.2.attn.in_proj_bias",
             "transformer.resblocks.2.attn.out_proj.weight", "transformer.resblocks.2.attn.out_proj.bias",
             "transformer.resblocks.2.ln_1.weight", "transformer.resblocks.2.ln_1.bias",
             "transformer.resblocks.2.mlp.c_fc.weight", "transformer.resblocks.2.mlp.c_fc.bias",
             "transformer.resblocks.2.mlp.c_proj.weight", "transformer.resblocks.2.mlp.c_proj.bias",
             "transformer.resblocks.2.ln_2.weight", "transformer.resblocks.2.ln_2.bias",
             "transformer.resblocks.3.attn.in_proj_weight", "transformer.resblocks.3.attn.in_proj_bias",
             "transformer.resblocks.3.attn.out_proj.weight", "transformer.resblocks.3.attn.out_proj.bias",
             "transformer.resblocks.3.ln_1.weight", "transformer.resblocks.3.ln_1.bias",
             "transformer.resblocks.3.mlp.c_fc.weight", "transformer.resblocks.3.mlp.c_fc.bias",
             "transformer.resblocks.3.mlp.c_proj.weight", "transformer.resblocks.3.mlp.c_proj.bias",
             "transformer.resblocks.3.ln_2.weight", "transformer.resblocks.3.ln_2.bias",
             "transformer.resblocks.4.attn.in_proj_weight", "transformer.resblocks.4.attn.in_proj_bias",
             "transformer.resblocks.4.attn.out_proj.weight", "transformer.resblocks.4.attn.out_proj.bias",
             "transformer.resblocks.4.ln_1.weight", "transformer.resblocks.4.ln_1.bias",
             "transformer.resblocks.4.mlp.c_fc.weight", "transformer.resblocks.4.mlp.c_fc.bias",
             "transformer.resblocks.4.mlp.c_proj.weight", "transformer.resblocks.4.mlp.c_proj.bias",
             "transformer.resblocks.4.ln_2.weight", "transformer.resblocks.4.ln_2.bias",
             "transformer.resblocks.5.attn.in_proj_weight", "transformer.resblocks.5.attn.in_proj_bias",
             "transformer.resblocks.5.attn.out_proj.weight", "transformer.resblocks.5.attn.out_proj.bias",
             "transformer.resblocks.5.ln_1.weight", "transformer.resblocks.5.ln_1.bias",
             "transformer.resblocks.5.mlp.c_fc.weight", "transformer.resblocks.5.mlp.c_fc.bias",
             "transformer.resblocks.5.mlp.c_proj.weight", "transformer.resblocks.5.mlp.c_proj.bias",
             "transformer.resblocks.5.ln_2.weight", "transformer.resblocks.5.ln_2.bias",
             "transformer.resblocks.6.attn.in_proj_weight", "transformer.resblocks.6.attn.in_proj_bias",
             "transformer.resblocks.6.attn.out_proj.weight", "transformer.resblocks.6.attn.out_proj.bias",
             "transformer.resblocks.6.ln_1.weight", "transformer.resblocks.6.ln_1.bias",
             "transformer.resblocks.6.mlp.c_fc.weight", "transformer.resblocks.6.mlp.c_fc.bias",
             "transformer.resblocks.6.mlp.c_proj.weight", "transformer.resblocks.6.mlp.c_proj.bias",
             "transformer.resblocks.6.ln_2.weight", "transformer.resblocks.6.ln_2.bias",
             "transformer.resblocks.7.attn.in_proj_weight", "transformer.resblocks.7.attn.in_proj_bias",
             "transformer.resblocks.7.attn.out_proj.weight", "transformer.resblocks.7.attn.out_proj.bias",
             "transformer.resblocks.7.ln_1.weight", "transformer.resblocks.7.ln_1.bias",
             "transformer.resblocks.7.mlp.c_fc.weight", "transformer.resblocks.7.mlp.c_fc.bias",
             "transformer.resblocks.7.mlp.c_proj.weight", "transformer.resblocks.7.mlp.c_proj.bias",
             "transformer.resblocks.7.ln_2.weight", "transformer.resblocks.7.ln_2.bias",
             "transformer.resblocks.8.attn.in_proj_weight", "transformer.resblocks.8.attn.in_proj_bias",
             "transformer.resblocks.8.attn.out_proj.weight", "transformer.resblocks.8.attn.out_proj.bias",
             "transformer.resblocks.8.ln_1.weight", "transformer.resblocks.8.ln_1.bias",
             "transformer.resblocks.8.mlp.c_fc.weight", "transformer.resblocks.8.mlp.c_fc.bias",
             "transformer.resblocks.8.mlp.c_proj.weight", "transformer.resblocks.8.mlp.c_proj.bias",
             "transformer.resblocks.8.ln_2.weight", "transformer.resblocks.8.ln_2.bias",
             "transformer.resblocks.9.attn.in_proj_weight", "transformer.resblocks.9.attn.in_proj_bias",
             "transformer.resblocks.9.attn.out_proj.weight", "transformer.resblocks.9.attn.out_proj.bias",
             "transformer.resblocks.9.ln_1.weight", "transformer.resblocks.9.ln_1.bias",
             "transformer.resblocks.9.mlp.c_fc.weight", "transformer.resblocks.9.mlp.c_fc.bias",
             "transformer.resblocks.9.mlp.c_proj.weight", "transformer.resblocks.9.mlp.c_proj.bias",
             "transformer.resblocks.9.ln_2.weight", "transformer.resblocks.9.ln_2.bias",
             "transformer.resblocks.10.attn.in_proj_weight", "transformer.resblocks.10.attn.in_proj_bias",
             "transformer.resblocks.10.attn.out_proj.weight", "transformer.resblocks.10.attn.out_proj.bias",
             "transformer.resblocks.10.ln_1.weight", "transformer.resblocks.10.ln_1.bias",
             "transformer.resblocks.10.mlp.c_fc.weight", "transformer.resblocks.10.mlp.c_fc.bias",
             "transformer.resblocks.10.mlp.c_proj.weight", "transformer.resblocks.10.mlp.c_proj.bias",
             "transformer.resblocks.10.ln_2.weight", "transformer.resblocks.10.ln_2.bias",
             "transformer.resblocks.11.attn.in_proj_weight", "transformer.resblocks.11.attn.in_proj_bias",
             "transformer.resblocks.11.attn.out_proj.weight", "transformer.resblocks.11.attn.out_proj.bias",
             "transformer.resblocks.11.ln_1.weight", "transformer.resblocks.11.ln_1.bias",
             "transformer.resblocks.11.mlp.c_fc.weight", "transformer.resblocks.11.mlp.c_fc.bias",
             "transformer.resblocks.11.mlp.c_proj.weight", "transformer.resblocks.11.mlp.c_proj.bias",
             "transformer.resblocks.11.ln_2.weight", "transformer.resblocks.11.ln_2.bias", "token_embedding.weight",
             "ln_final.weight", "ln_final.bias"]


#
def build_model(state_dict: dict, replace_stride_with_dilation=[False, False, True]):
    embed_dim = state_dict["text_projection"].shape[1]
    for k in to_remove:
        state_dict.pop(k, None)

    counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
    vision_layers = tuple(counts)
    vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    image_resolution = output_width * 32

    model = CLIP_encoder(embed_dim, image_resolution, vision_layers, vision_width, replace_stride_with_dilation)

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target