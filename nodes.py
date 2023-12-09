from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as nnf
import einops
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.attention import optimized_attention, optimized_attention_masked
import comfy.ops

T = torch.Tensor


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = True
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    shared_score_scale: float = 1.0
    shared_score_shift: float = 0.0
    only_self_level: float = 0.0


def expand_first(
    feat: T,
    scale=1.0,
) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


def sdpa(q: T, k: T, v: T, mask=None, heads: int = 8) -> T:
    if mask:
        return optimized_attention_masked(q, k, v, heads, mask)
    else:
        return optimized_attention(q, k, v, heads)


class SharedAttentionProcessor:
    def __init__(
        self,
        style_aligned_args: StyleAlignedArgs,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        dtype=None,
        device=None,
        operations=comfy.ops,
    ):
        super().__init__()
        self.args = style_aligned_args
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        assert context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(
            query_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_k = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_v = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )

        self.to_out = nn.Sequential(
            operations.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout),
        )

    def shifted_scaled_dot_product_attention(self, query: T, key: T, value: T) -> T:
        logits = torch.einsum("bhqd,bhkd->bhqk", query, key)
        logits[:, :, :, query.shape[2] :] += self.args.shared_score_shift
        probs = logits.softmax(-1)
        return torch.einsum("bhqk,bhkd->bhqd", probs, value)

    def forward(self, x, context=None, value=None, mask=None):
        query = self.to_q(x)
        context = default(context, x)
        key = self.to_k(context)
        v = self.to_v(x)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        if self.args.adain_queries:
            query = adain(query)
        if self.args.adain_keys:
            key = adain(key)
        if self.args.adain_values:
            v = adain(v)
        if self.args.share_attention:
            key = concat_first(key, -2, scale=self.args.shared_score_scale)
            v = concat_first(v, -2)
            if self.args.shared_score_shift != 0:
                x = self.shifted_scaled_dot_product_attention(
                    query,
                    key,
                    v,
                )
            else:
                x = sdpa(query, key, v, mask, self.heads)
        else:
            x = sdpa(query, key, v, mask, self.heads)

        return x

def register_shared_norm(
    model: ModelPatcher,
    share_group_norm: bool = True,
    share_layer_norm: bool = True,
):
    def register_norm_forward(
        norm_layer: nn.GroupNorm | nn.LayerNorm,
    ) -> nn.GroupNorm | nn.LayerNorm:
        if not hasattr(norm_layer, "orig_forward"):
            setattr(norm_layer, "orig_forward", norm_layer.forward)
        orig_forward = norm_layer.orig_forward

        def forward_(hidden_states: T) -> T:
            n = hidden_states.shape[-2]
            hidden_states = concat_first(hidden_states, dim=-2)
            hidden_states = orig_forward(hidden_states)
            return hidden_states[..., :n, :]

        norm_layer.forward = forward_  # type: ignore
        return norm_layer

    def get_norm_layers(
        layer, norm_layers_: dict[str, list[nn.GroupNorm | nn.LayerNorm]]
    ):
        if isinstance(layer, nn.LayerNorm) and share_layer_norm:
            norm_layers_["layer"].append(layer)
        if isinstance(layer, nn.GroupNorm) and share_group_norm:
            norm_layers_["group"].append(layer)
        else:
            for layer in layer.children():
                get_norm_layers(layer, norm_layers_)

    norm_layers = {"group": [], "layer": []}
    get_norm_layers(model, norm_layers)
    return [register_norm_forward(layer) for layer in norm_layers["group"]] + [
        register_norm_forward(layer) for layer in norm_layers["layer"]
    ]


def _get_switch_vec(total_num_layers, level):
    if level == 0:
        return torch.zeros(total_num_layers, dtype=torch.bool)
    if level == 1:
        return torch.ones(total_num_layers, dtype=torch.bool)
    to_flip = level > 0.5
    if to_flip:
        level = 1 - level
    num_switch = int(level * total_num_layers)
    vec = torch.arange(total_num_layers)
    vec = vec % (total_num_layers // num_switch)
    vec = vec == 0
    if to_flip:
        vec = ~vec
    return vec


def init_attention_processors(model, style_aligned_args: StyleAlignedArgs):
    attn_procs = {}
    number_of_self, number_of_cross = 0, 0
    num_self_layers = len(
        [name for name in model.keys() if "attn1" in name]
    )
    if style_aligned_args is None:
        only_self_vec = _get_switch_vec(num_self_layers, 1)
    else:
        only_self_vec = _get_switch_vec(
            num_self_layers, style_aligned_args.only_self_level
        )
    for i, name in enumerate(model.keys()):
        is_self_attention = "attn1" in name
        if is_self_attention:
            number_of_self += 1
            if only_self_vec[i // 2]:
                attn_procs[name] = SharedAttentionProcessor(style_aligned_args)


class StyleAlignedPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "style_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "custom_node_experiments"

    def __init__(self, model: ModelPatcher) -> None:
        self.args = StyleAlignedArgs()
        self.norm_layers = register_shared_norm(
            model, self.args.share_group_norm, self.args.share_layer_norm
        )

    def patch(self, model):
        m = model.clone()
        return (m,)


NODE_CLASS_MAPPINGS = {
    "StyleAlignedPatch": StyleAlignedPatch,
}
