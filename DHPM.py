################
# script to augment features with CLIP + SCR (intra/inter-class losses)
import pickle
import os
import clip
import torch
import network
import torch.nn as nn
import torch.nn.functional as F
from utils.stats import calc_mean_std
import argparse
from main import get_dataset
from torch.utils import data
import numpy as np
import random
import yaml
from torch.utils.tensorboard import SummaryWriter
from resnet_clip import PromptLearner


# -------- Text->Vision adapter (stable) --------
class CouplingFunction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, language_features):   # (B, D_text) -> (B, C)
        x = self.fc(language_features)
        return torch.tanh(x) * 0.1


def compose_text_with_templates(text: str, templates: list, attributes: list) -> list:
    texts = []
    for attr in attributes:
        phrase = f"{text}, {attr}"
        for template in templates:
            texts.append(template.format(phrase))
    return texts


imagenet_templates = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.',
    'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
    'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.',
    'a photo of a hard to see {}.', 'a bright photo of a {}.',
    'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.',
    'a photo of the cool {}.', 'a close-up photo of a {}.',
    'a black and white photo of the {}.', 'a painting of the {}.',
    'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.',
    'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.',
    'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.',
    'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.',
    'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.',
    'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.',
    'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.',
    'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.',
    'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.',
    'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.',
    'a photo of the nice {}.', 'a photo of the small {}.',
    'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.',
    'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
    'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.',
    'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
]


# night
attributes = []
#snow
#attributes = []
#rain
#attributes = []
#game
#attributes = []
#gta5_cs
#attributes = []


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--data_root", type=str, default='./datasets/data', help="path to dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="path for learnt parameters saving")
    parser.add_argument("--dataset", type=str, default='cityscapes', choices=['cityscapes', 'gta5', 'ACDC'], help='Name of dataset')
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=8, help='batch size (default: 8)')

    available_models = sorted(
        name for name in network.modeling.__dict__
        if name.islower() and not (name.startswith("__") or name.startswith('_'))
        and callable(network.modeling.__dict__[name])
    )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip', choices=available_models, help='model name')
    parser.add_argument("--BB", type=str, default='RN50', help="backbone name")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--total_it", type=int, default=100, help="total number of optimization iterations")
    parser.add_argument("--resize_feat", action='store_true', default=False, help="resize the features map to the dimension corresponding to CLIP")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--domain_desc", type=str, default="driving in rain", help="description of the target domain")

    # loss weights
    parser.add_argument("--lambda_sub_agg", type=float, default=0.25, help="weight for intra-class (sub-agg) loss")
    parser.add_argument("--lambda_div",     type=float, default=0.25, help="weight for inter-class loss")
    parser.add_argument("--lambda_hier",    type=float, default=1, help="weight: parent-child consistency")
    parser.add_argument("--lambda_subdiv",  type=float, default=1, help="weight: separation among sub-prototypes of same parent")

    # margins / thresholds
    parser.add_argument("--tau", type=float, default=0.3, help="cosine margin for inter-class separation")
    parser.add_argument("--mu_margin", type=float, default=0.15, help="cos margin among sub-prototypes of same parent")

    # memory config
    parser.add_argument("--mb_momentum", type=float, default=0.1, help="EMA momentum for prototype updates")
    parser.add_argument("--num_classes", type=int, default=19, help="number of semantic classes")

    # ===== 软约束 + 自适应预算：新增/修改 =====
    parser.add_argument("--kmax_sub", type=int, default=4,
                        help="SOFT cap per class (can exceed, but spawn becomes harder)")

    parser.add_argument("--max_total_sub", type=int, default=300,
                        help="GLOBAL budget for total number of sub-prototypes (across all classes)")

    parser.add_argument("--soft_overflow_delta", type=float, default=0.05,
                        help="extra strictness per extra sub-prototype beyond soft cap")

    parser.add_argument("--soft_overflow_patience_mul", type=float, default=1.5,
                        help="patience multiplier when exceeding soft cap")

    # ===== 淘汰策略：新增 hybrid =====
    parser.add_argument("--evict_by", type=str, default="hybrid",
                        choices=["count", "count_lastused", "hybrid"],
                        help="eviction strategy when exceeding global budget")

    # ===== hybrid eviction weights =====
    parser.add_argument("--evict_beta", type=float, default=0.2,
                        help="weight for absolute-count term in hybrid eviction (auxiliary)")
    parser.add_argument("--evict_alpha", type=float, default=0.05,
                        help="weight for stale term in hybrid eviction (tie-break)")
    parser.add_argument("--evict_stale_norm", type=float, default=2000.0,
                        help="normalize stale by this value in hybrid score")
    parser.add_argument("--evict_min_ratio", type=float, default=0.01,
                        help="if ratio < this, give an extra deletion bonus (hybrid)")
    parser.add_argument("--evict_min_count", type=int, default=5,
                        help="if count < this, give an extra deletion bonus (hybrid)")

    parser.add_argument("--birth_patience", type=int, default=5, help="consecutive violations needed to spawn a new sub-prototype")
    parser.add_argument("--merge_thresh", type=float, default=0.85, help="cos sim threshold to merge sub-prototypes")
    parser.add_argument("--prune_min_count", type=int, default=10, help="min assignments to keep a sub-prototype")
    parser.add_argument("--min_pixels_spawn", type=int, default=512, help="min pixel count to allow spawning a sub-prototype")

    parser.add_argument("--hard_neg_k", type=int, default=5, help="top-K hard negatives for inter-class separation")

    parser.add_argument("--save_proto_every", type=int, default=0,
                        help="save proto_memory.pt every N batches (0 means only save at end)")

    return parser


class PIN(nn.Module):
    def __init__(self, shape, content_feat, coupling_function):
        super(PIN, self).__init__()
        self.shape = shape

        self.content_feat = content_feat.clone().detach()    # (B, C, H, W)
        self.content_mean, self.content_std = calc_mean_std(self.content_feat)
        self.size = self.content_feat.size()                 # (B, C, H, W)

        self.content_feat_norm = (self.content_feat - self.content_mean.expand(self.size)) / \
                                 (self.content_std.expand(self.size) + 1e-6)

        self.style_mean = nn.Parameter(self.content_mean.clone().detach(), requires_grad=True)
        self.style_std  = nn.Parameter(self.content_std.clone().detach(),  requires_grad=True)

        self.relu = nn.ReLU(inplace=False)

        self.coupling_function = coupling_function

    def forward(self, language_features):
        # language_features: (B, D_text)
        visual_bias = self.coupling_function(language_features).view(
            self.size[0], self.size[1], 1, 1
        )  # (B, C, 1, 1)

        style_std_pos = torch.clamp(self.style_std, min=0)

        target_feat = self.content_feat_norm * style_std_pos.expand(self.size) + \
                      self.style_mean.expand(self.size)

        target_feat = target_feat + visual_bias
        target_feat = self.relu(target_feat)
        return target_feat


class HierarchicalPrototypeMemory:
    def __init__(self, num_classes, feat_dim, device, momentum=0.1,
                 kmax_sub=4, birth_patience=3, merge_thresh=0.85, prune_min_count=5,
                 max_total_sub=300, soft_overflow_delta=0.05, soft_overflow_patience_mul=1.5,
                 evict_by="hybrid",
                 evict_beta=0.2, evict_alpha=0.05, evict_stale_norm=2000.0,
                 evict_min_ratio=0.01, evict_min_count=5,
                 birth_hist_len=1000, birth_min_samples=20,
                 birth_mad_scale=2.0, default_birth_cos=0.7, birth_min_cos=0.3):

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.momentum = momentum

        self.parent_valid = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.parent_proto = torch.zeros(num_classes, feat_dim, device=device)

        self.sub_protos = [[] for _ in range(num_classes)]
        self.sub_counts = [[] for _ in range(num_classes)]
        self.birth_hits = [0 for _ in range(num_classes)]

        self.k_soft_sub = kmax_sub
        self.max_total_sub = max_total_sub
        self.soft_overflow_delta = soft_overflow_delta
        self.soft_overflow_patience_mul = soft_overflow_patience_mul

        self.evict_by = evict_by
        self.evict_beta = evict_beta
        self.evict_alpha = evict_alpha
        self.evict_stale_norm = evict_stale_norm
        self.evict_min_ratio = evict_min_ratio
        self.evict_min_count = evict_min_count

        self.step = 0
        self.sub_last_used = [[] for _ in range(num_classes)]

        self.birth_patience = birth_patience
        self.merge_thresh = merge_thresh
        self.prune_min_count = prune_min_count

        self.birth_hist_len = birth_hist_len
        self.birth_min_samples = birth_min_samples
        self.birth_mad_scale = birth_mad_scale
        self.default_birth_cos = default_birth_cos
        self.birth_min_cos = birth_min_cos
        self.cos_history = [[] for _ in range(num_classes)]

    @torch.no_grad()
    def _cos(self, a, b):
        return torch.dot(a, b) / ((a.norm() + 1e-6) * (b.norm() + 1e-6))

    def total_subs(self):
        return sum(len(lst) for lst in self.sub_protos)

    @torch.no_grad()
    def _evict_one(self):
        all_counts = []
        for c in range(self.num_classes):
            all_counts += [float(x) for x in self.sub_counts[c]]
        global_med = float(np.median(all_counts)) if len(all_counts) > 0 else 1.0
        global_med = max(global_med, 1.0)

        def select_candidate(keep_one_per_class: bool):
            best_local = None  # (score, c, j)
            for c in range(self.num_classes):
                n = len(self.sub_protos[c])
                if keep_one_per_class and n <= 1:
                    continue

                total_c = float(sum(self.sub_counts[c])) + 1e-6
                for j in range(n):
                    cnt = float(self.sub_counts[c][j])
                    ratio = cnt / total_c
                    last = self.sub_last_used[c][j] if j < len(self.sub_last_used[c]) else 0
                    stale = float(self.step - last)

                    if self.evict_by == "count":
                        score = cnt
                    elif self.evict_by == "count_lastused":
                        score = cnt - 0.001 * stale
                    else:
                        # hybrid
                        count_term = cnt / (global_med + 1e-6)  # ~O(1)
                        stale_term = stale / (self.evict_stale_norm + 1e-6)

                        score = ratio + self.evict_beta * count_term - self.evict_alpha * stale_term

                        if ratio < self.evict_min_ratio:
                            score -= 0.05
                        if cnt < self.evict_min_count:
                            score -= 0.05

                    if (best_local is None) or (score < best_local[0]):
                        best_local = (score, c, j)
            return best_local

        best = select_candidate(keep_one_per_class=True)
        if best is None:
            best = select_candidate(keep_one_per_class=False)
        if best is None:
            return False

        _, c, j = best
        del self.sub_protos[c][j]
        del self.sub_counts[c][j]
        if j < len(self.sub_last_used[c]):
            del self.sub_last_used[c][j]
        return True

    @torch.no_grad()
    def ensure_budget(self, force_one: bool = False):
        if self.max_total_sub is None:
            return

        target = self.max_total_sub - (1 if force_one else 0)
        while self.total_subs() > target:
            ok = self._evict_one()
            if not ok:
                break

    @torch.no_grad()
    def _update_cos_history(self, cls_id, cos_val):
        cos_val = float(cos_val)
        if not np.isfinite(cos_val):
            return
        hist = self.cos_history[cls_id]
        hist.append(cos_val)
        if len(hist) > self.birth_hist_len:
            hist.pop(0)

    @torch.no_grad()
    def _get_birth_threshold(self, cls_id):
        hist = self.cos_history[cls_id]
        if len(hist) < self.birth_min_samples:
            return self.default_birth_cos
        arr = np.asarray(hist, dtype=np.float32)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median)) + 1e-6
        thr = median - self.birth_mad_scale * mad
        thr = max(self.birth_min_cos, min(float(thr), 0.99))
        return thr

    @torch.no_grad()
    def assign_sub(self, cls_id, feat):
        subs = self.sub_protos[cls_id]
        if len(subs) == 0:
            return -1, float("-inf")
        sims = [self._cos(feat, s) for s in subs]
        sims_t = torch.tensor(sims, device=feat.device, dtype=feat.dtype)
        idx = int(sims_t.argmax().item())
        return idx, sims[idx]

    @torch.no_grad()
    def maybe_birth(self, cls_id, feat, cos_to_near):
        feat = feat / (feat.norm(p=2) + 1e-6)
        feat_clean = feat.detach().clone()

        subs = self.sub_protos[cls_id]

        # 第一个子原型：直接建
        if len(subs) == 0:
            self.sub_protos[cls_id].append(feat_clean)
            self.sub_counts[cls_id].append(1)
            self.sub_last_used[cls_id].append(self.step)
            self.birth_hits[cls_id] = 0

            if not self.parent_valid[cls_id]:
                self.parent_proto[cls_id] = feat_clean.clone()
                self.parent_valid[cls_id] = True

            self.ensure_budget(force_one=False)
            return True

        self._update_cos_history(cls_id, cos_to_near)
        thr = self._get_birth_threshold(cls_id)

        overflow = max(0, len(subs) - self.k_soft_sub + 1)
        thr_eff = thr - overflow * self.soft_overflow_delta
        thr_eff = max(self.birth_min_cos, float(thr_eff))
        pat_eff = int(np.ceil(self.birth_patience * (self.soft_overflow_patience_mul ** overflow)))

        if (self.max_total_sub is not None) and (self.total_subs() >= self.max_total_sub):
            self.ensure_budget(force_one=True)

        can_spawn = (self.max_total_sub is None) or (self.total_subs() < self.max_total_sub)

        if can_spawn and (cos_to_near < thr_eff):
            self.birth_hits[cls_id] += 1
            if self.birth_hits[cls_id] >= pat_eff:
                self.sub_protos[cls_id].append(feat_clean)
                self.sub_counts[cls_id].append(1)
                self.sub_last_used[cls_id].append(self.step)
                self.birth_hits[cls_id] = 0

                self.ensure_budget(force_one=False)
                return True
        else:
            self.birth_hits[cls_id] = 0

        return False

    @torch.no_grad()
    def update_sub(self, cls_id, sub_idx, feat):
        feat = feat.detach()
        proto = self.sub_protos[cls_id][sub_idx]
        proto = (1 - self.momentum) * proto + self.momentum * feat
        proto = proto / (proto.norm(p=2) + 1e-6)
        self.sub_protos[cls_id][sub_idx] = proto
        self.sub_counts[cls_id][sub_idx] += 1

        while len(self.sub_last_used[cls_id]) < len(self.sub_protos[cls_id]):
            self.sub_last_used[cls_id].append(0)
        self.sub_last_used[cls_id][sub_idx] = self.step

    @torch.no_grad()
    def update_parent(self, cls_id):
        if len(self.sub_protos[cls_id]) == 0:
            return
        mean_child = torch.stack(self.sub_protos[cls_id], dim=0).mean(dim=0)
        if not self.parent_valid[cls_id]:
            self.parent_proto[cls_id] = mean_child
            self.parent_valid[cls_id] = True
        else:
            p = self.parent_proto[cls_id]
            p = (1 - self.momentum) * p + self.momentum * mean_child
            p = p / (p.norm(p=2) + 1e-6)
            self.parent_proto[cls_id] = p

    @torch.no_grad()
    def merge_and_prune(self, cls_id):
        # ---- merge ----
        changed = True
        while changed and len(self.sub_protos[cls_id]) >= 2:
            changed = False
            n = len(self.sub_protos[cls_id])
            for i in range(n):
                for j in range(i + 1, n):
                    sim = self._cos(self.sub_protos[cls_id][i], self.sub_protos[cls_id][j])
                    if sim > self.merge_thresh:
                        ci, cj = self.sub_counts[cls_id][i], self.sub_counts[cls_id][j]
                        merged = (self.sub_protos[cls_id][i] * ci + self.sub_protos[cls_id][j] * cj) / (ci + cj + 1e-6)
                        merged = merged / (merged.norm(p=2) + 1e-6)

                        self.sub_protos[cls_id][i] = merged
                        self.sub_counts[cls_id][i] = ci + cj

                        # last_used：合并取 max
                        while len(self.sub_last_used[cls_id]) < len(self.sub_protos[cls_id]):
                            self.sub_last_used[cls_id].append(0)
                        li = self.sub_last_used[cls_id][i] if i < len(self.sub_last_used[cls_id]) else 0
                        lj = self.sub_last_used[cls_id][j] if j < len(self.sub_last_used[cls_id]) else 0
                        self.sub_last_used[cls_id][i] = max(li, lj)

                        del self.sub_protos[cls_id][j]
                        del self.sub_counts[cls_id][j]
                        if j < len(self.sub_last_used[cls_id]):
                            del self.sub_last_used[cls_id][j]

                        changed = True
                        break
                if changed:
                    break

        if len(self.sub_counts[cls_id]) > 0:
            keep_idx = [k for k, cnt in enumerate(self.sub_counts[cls_id]) if cnt >= self.prune_min_count]

            if len(keep_idx) == 0:
                best_k = int(np.argmax(np.asarray(self.sub_counts[cls_id], dtype=np.float32)))
                keep_idx = [best_k]

            self.sub_protos[cls_id] = [self.sub_protos[cls_id][k] for k in keep_idx]
            self.sub_counts[cls_id] = [self.sub_counts[cls_id][k] for k in keep_idx]
            if len(self.sub_last_used[cls_id]) > 0:
                self.sub_last_used[cls_id] = [self.sub_last_used[cls_id][k] for k in keep_idx]

        self.ensure_budget(force_one=False)


def masked_class_means_single(feat_map_1, labels_1, num_classes):
    """
    feat_map_1: (1,C,H,W)
    labels_1 : (1,H,W)
    return: dict{cls: (C,)}, dict{cls: pixel_count_float}
    """
    _, C, H, W = feat_map_1.shape
    class_means = {}
    class_counts = {}
    for cls_id in range(num_classes):
        mask = (labels_1 == cls_id)
        if not mask.any():
            continue
        mask_f = mask.view(1, 1, H, W).float()
        feat_sum = (feat_map_1 * mask_f).sum(dim=(0, 2, 3))
        denom = mask_f.sum(dim=(0, 2, 3)) + 1e-6
        mean_feat = feat_sum / denom
        class_means[int(cls_id)] = mean_feat
        class_counts[int(cls_id)] = float(denom.item())
    return class_means, class_counts


def dump_memory(memory: HierarchicalPrototypeMemory, save_path: str):
    mem = {
        "num_classes": memory.num_classes,
        "feat_dim": memory.feat_dim,
        "parent_valid": memory.parent_valid.detach().cpu(),
        "parent_proto": memory.parent_proto.detach().cpu(),
        "sub_protos": [[t.detach().cpu() for t in lst] for lst in memory.sub_protos],
        "sub_counts": memory.sub_counts,
        "sub_last_used": memory.sub_last_used,
        "step": memory.step,
        "k_soft_sub": memory.k_soft_sub,
        "max_total_sub": memory.max_total_sub,
        "evict_by": memory.evict_by,
        "evict_beta": memory.evict_beta,
        "evict_alpha": memory.evict_alpha,
        "evict_stale_norm": memory.evict_stale_norm,
        "evict_min_ratio": memory.evict_min_ratio,
        "evict_min_count": memory.evict_min_count,
    }
    torch.save(mem, save_path)
    print(f"[INFO] proto memory saved to: {save_path}")


def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    torch.manual_seed(opts.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts.dataset, opts.data_root, opts.crop_size, data_aug=False)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0, drop_last=False
    )
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes, BB=opts.BB,
        replace_stride_with_dilation=[False, False, False]
    )
    model = model.to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    writer = SummaryWriter()
    os.makedirs(opts.save_dir, exist_ok=True)

    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56, 56))
    else:
        t1 = lambda x: x

    if len(attributes) == 0:
        target = [tmpl.format(opts.domain_desc) for tmpl in imagenet_templates]
    else:
        target = compose_text_with_templates(opts.domain_desc, imagenet_templates, attributes)

    with torch.no_grad():
        tokens = clip.tokenize(target).to(device)
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_target = text_feats.mean(dim=0, keepdim=True).float()  # (1,D)

    text_target = text_target.repeat(opts.batch_size, 1)    # (B, D_text)

    classnames = ['driving at night']
    with open('/media/cs4007/50e71e09-dfc5-4acc-9e94-8ae14f169300/home/cs4007/pjh/PODA-master3/cfg/rn50_ep50_ctxv1.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    prompt_learner = PromptLearner(cfg, classnames, clip_model).to(device)

    with torch.no_grad():
        _ = prompt_learner()

    memory = HierarchicalPrototypeMemory(
        num_classes=opts.num_classes,
        feat_dim=256,
        device=device,
        momentum=opts.mb_momentum,
        kmax_sub=opts.kmax_sub,  # soft cap
        birth_patience=opts.birth_patience,
        merge_thresh=opts.merge_thresh,
        prune_min_count=opts.prune_min_count,
        max_total_sub=opts.max_total_sub,
        soft_overflow_delta=opts.soft_overflow_delta,
        soft_overflow_patience_mul=opts.soft_overflow_patience_mul,
        evict_by=opts.evict_by,

        evict_beta=opts.evict_beta,
        evict_alpha=opts.evict_alpha,
        evict_stale_norm=opts.evict_stale_norm,
        evict_min_ratio=opts.evict_min_ratio,
        evict_min_count=opts.evict_min_count,
    )


    for i, (img_id, tar_id, images, labels) in enumerate(train_loader):
        print("batch:", i)

        images = images.to(device)
        labels = labels.to(device)

        # backbone feature
        f1 = model.backbone(
            images, trunc1=False, trunc2=False, trunc3=False, trunc4=False,
            get1=True, get2=False, get3=False, get4=False
        )  # (B, 256, H, W)


        coupling = CouplingFunction(input_dim=text_target.shape[1], output_dim=256).to(device)
        model_pin_1 = PIN([f1.shape[0], 256, 1, 1], f1, coupling).to(device)

        base_params = [p for n, p in model_pin_1.named_parameters()
                       if not n.startswith('coupling_function.')]
        optimizer_pin_1 = torch.optim.SGD(
            params=[
                {'params': base_params, 'lr': 1.0},
                {'params': model_pin_1.coupling_function.parameters(), 'lr': 0.1},
            ],
            lr=1, momentum=0.9, weight_decay=opts.weight_decay
        )


        if i == len(train_loader) - 1 and f1.shape[0] < opts.batch_size:
            text_target_b = text_target[:f1.shape[0]]
        else:
            text_target_b = text_target

        with torch.no_grad():
            labels_resized = F.interpolate(
                labels.unsqueeze(1).float(),
                size=f1.shape[-2:], mode='nearest'
            ).squeeze(1).long()

        cur_itrs = 0
        while cur_itrs < opts.total_it:
            cur_itrs += 1
            memory.step += 1

            optimizer_pin_1.zero_grad()

            f1_hal = model_pin_1(text_target_b)
            f1_hal_trans = t1(f1_hal)

            target_features_from_f1 = model.backbone(
                f1_hal_trans, trunc1=True, trunc2=False, trunc3=False, trunc4=False,
                get1=False, get2=False, get3=False, get4=False
            )
            target_features_from_f1 = target_features_from_f1 / (
                target_features_from_f1.norm(dim=-1, keepdim=True) + 1e-6
            )
            loss_CLIP1 = (1 - torch.cosine_similarity(text_target_b, target_features_from_f1, dim=1)).mean()
            writer.add_scalar(f"loss_CLIP_f1/b{i}", loss_CLIP1.item(), cur_itrs)


            f1_norm = F.normalize(f1_hal, p=2, dim=1)


            class_mean_dict_batch = {}
            class_count_dict_batch = {}
            B, C, H, W = f1_norm.shape
            for cls_id in range(opts.num_classes):
                mask = (labels_resized == cls_id)
                if not mask.any():
                    continue
                mask_f = mask.view(B, 1, H, W).float()
                feat_sum = (f1_norm * mask_f).sum(dim=(0, 2, 3))
                denom = mask_f.sum(dim=(0, 2, 3)) + 1e-6
                mean_feat = feat_sum / denom
                class_mean_dict_batch[int(cls_id)] = mean_feat
                class_count_dict_batch[int(cls_id)] = float(denom.item())

            L_sub_agg = f1_norm.new_tensor(0.0)
            L_div     = f1_norm.new_tensor(0.0)
            L_hier    = f1_norm.new_tensor(0.0)
            L_sub_div = f1_norm.new_tensor(0.0)

            present_classes = list(class_mean_dict_batch.keys())
            total_pixels = (sum(class_count_dict_batch[c] for c in present_classes) + 1e-6) if present_classes else 1.0

            for c in present_classes:
                w_c = class_count_dict_batch[c] / total_pixels
                f_c = F.normalize(class_mean_dict_batch[c], p=2, dim=0)

                sub_idx, cos_near = memory.assign_sub(c, f_c)

                if class_count_dict_batch[c] >= opts.min_pixels_spawn:
                    spawned = memory.maybe_birth(c, f_c, cos_near)
                else:
                    spawned = False

                if spawned:
                    sub_idx, cos_near = memory.assign_sub(c, f_c)

                if sub_idx == -1:
                    if memory.parent_valid[c]:
                        parent_proto = memory.parent_proto[c].detach().clone()
                        L_sub_agg = L_sub_agg + w_c * (f_c - parent_proto).pow(2).mean()
                    continue

                sub_proto = memory.sub_protos[c][sub_idx].detach().clone()
                L_sub_agg = L_sub_agg + w_c * (f_c - sub_proto).pow(2).mean()

                if memory.parent_valid[c]:
                    parent_proto = memory.parent_proto[c].detach().clone()
                    L_hier = L_hier + w_c * (sub_proto - parent_proto).pow(2).mean()


                if len(memory.sub_protos[c]) > 1:
                    sub_terms = []
                    for kk in range(len(memory.sub_protos[c])):
                        if kk == sub_idx:
                            continue
                        other_sub = memory.sub_protos[c][kk].detach().clone()
                        cos_sk = torch.dot(sub_proto, other_sub) / (
                            (sub_proto.norm() + 1e-6) * (other_sub.norm() + 1e-6)
                        )
                        sub_terms.append(F.relu(opts.mu_margin - cos_sk))
                    if sub_terms:
                        L_sub_div = L_sub_div + w_c * torch.stack(sub_terms).mean()


                neg_sims = []
                for k in range(opts.num_classes):
                    if k == c or not memory.parent_valid[k]:
                        continue
                    other_p = memory.parent_proto[k].detach().clone()
                    cos_ck = torch.dot(f_c, other_p) / (
                        (f_c.norm() + 1e-6) * (other_p.norm() + 1e-6)
                    )
                    neg_sims.append(cos_ck)

                if neg_sims:
                    sims = torch.stack(neg_sims)
                    K = min(getattr(opts, "hard_neg_k", 5), sims.numel())
                    topk_vals, _ = torch.topk(sims, K)
                    div_terms = F.relu(opts.tau - topk_vals)
                    L_div = L_div + w_c * div_terms.mean()

            warm = min(1.0, cur_itrs / 30.0)
            total_loss = (
                loss_CLIP1
                + opts.lambda_sub_agg * L_sub_agg
                + warm * (opts.lambda_div * L_div + opts.lambda_hier * L_hier + opts.lambda_subdiv * L_sub_div)
            )

            writer.add_scalar(f"loss_sub_agg/b{i}", L_sub_agg.item(), cur_itrs)
            writer.add_scalar(f"loss_div/b{i}",     L_div.item(),     cur_itrs)
            writer.add_scalar(f"loss_hier/b{i}",    L_hier.item(),    cur_itrs)
            writer.add_scalar(f"loss_sub_div/b{i}", L_sub_div.item(), cur_itrs)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_pin_1.parameters(), max_norm=1.0)
            optimizer_pin_1.step()

            with torch.no_grad():
                for c in present_classes:
                    f_c_upd = F.normalize(class_mean_dict_batch[c], p=2, dim=0).detach()
                    if class_count_dict_batch[c] < opts.min_pixels_spawn:
                        sub_idx2, _ = memory.assign_sub(c, f_c_upd)
                        if sub_idx2 != -1:
                            memory.update_sub(c, sub_idx2, f_c_upd)
                    else:
                        sub_idx2, _ = memory.assign_sub(c, f_c_upd)
                        if sub_idx2 == -1:
                            memory.maybe_birth(c, f_c_upd, cos_to_near=-1.0)
                        else:
                            memory.update_sub(c, sub_idx2, f_c_upd)

                    memory.update_parent(c)

                if cur_itrs % 20 == 0:
                    for c in present_classes:
                        memory.merge_and_prune(c)

                memory.ensure_budget(force_one=False)

        learnt_mu_f1 = None
        learnt_std_f1 = None
        for name, param in model_pin_1.named_parameters():
            if param.requires_grad and name == 'style_mean':
                learnt_mu_f1 = param.detach().clone()
            elif param.requires_grad and name == 'style_std':
                learnt_std_f1 = param.detach().clone()

        with torch.no_grad():
            per_img_means = []
            per_img_counts = []
            for b in range(f1_norm.size(0)):
                means_b, counts_b = masked_class_means_single(
                    f1_norm[b:b+1], labels_resized[b:b+1], opts.num_classes
                )
                per_img_means.append(means_b)
                per_img_counts.append(counts_b)

        for k in range(learnt_mu_f1.shape[0]):
            mu_k  = learnt_mu_f1[k].detach().cpu()
            std_k = learnt_std_f1[k].detach().cpu()

            means_k = per_img_means[k]
            counts_k = per_img_counts[k]

            stats = {
                'mu_f1': mu_k,
                'std_f1': std_k,
                'present_classes': list(means_k.keys()),
                'class_means': {int(c): means_k[c].detach().cpu() for c in means_k},
                'class_counts': {int(c): float(counts_k[c]) for c in counts_k},
            }

            out_name = os.path.join(opts.save_dir, os.path.basename(img_id[k]) + '.pkl')
            with open(out_name, 'wb') as f:
                pickle.dump(stats, f)

        print("style_mean:", learnt_mu_f1.shape)
        print("style_std :", learnt_std_f1.shape)

        if opts.save_proto_every and ((i + 1) % opts.save_proto_every == 0):
            dump_memory(memory, os.path.join(opts.save_dir, "proto_memory.pt"))

    dump_memory(memory, os.path.join(opts.save_dir, "proto_memory.pt"))


if __name__ == "__main__":
    main()
