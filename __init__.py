from dataclasses import dataclass
import torch
import torch.nn as nn
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.attention import optimized_attention, optimized_attention_masked
import comfy.ops
from typing import Optional, Union
import comfy.sample
import latent_preview
import comfy.utils

T = torch.Tensor


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


class StyleAlignedArgs:
    def __init__(self, share_attn: str) -> None:
        self.adain_keys = "k" in share_attn
        self.adain_values = "v" in share_attn
        self.adain_queries = "q" in share_attn

    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = True


def expand_first(
    feat: T,
    scale=1.0,
) -> T:
    """
    Expand the first element so it has the same shape as the rest of the batch.
    """
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:
    """
    concat the the feature and the style feature expanded above
    """
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> "tuple[T, T]":
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
    def __init__(self, args: StyleAlignedArgs, scale: float):
        self.args = args
        self.scale = scale

    def __call__(self, q, k, v, extra_options):
        if self.args.adain_queries:
            q = adain(q)
        if self.args.adain_keys:
            k = adain(k)
        if self.args.adain_values:
            v = adain(v)
        if self.args.share_attention:
            k = concat_first(k, -2, scale=self.scale)
            v = concat_first(v, -2)

        return q, k, v


def get_norm_layers(
    layer: nn.Module,
    norm_layers_: "dict[str, list[Union[nn.GroupNorm, nn.LayerNorm]]]",
    share_layer_norm: bool,
    share_group_norm: bool,
):
    if isinstance(layer, nn.LayerNorm) and share_layer_norm:
        norm_layers_["layer"].append(layer)
    if isinstance(layer, nn.GroupNorm) and share_group_norm:
        norm_layers_["group"].append(layer)
    else:
        for child_layer in layer.children():
            get_norm_layers(
                child_layer, norm_layers_, share_layer_norm, share_group_norm
            )


def register_norm_forward(
    norm_layer: Union[nn.GroupNorm, nn.LayerNorm],
) -> Union[nn.GroupNorm, nn.LayerNorm]:
    if not hasattr(norm_layer, "orig_forward"):
        setattr(norm_layer, "orig_forward", norm_layer.forward)
    orig_forward = norm_layer.orig_forward

    def forward_(hidden_states: T) -> T:
        n = hidden_states.shape[-2]
        hidden_states = concat_first(hidden_states, dim=-2)
        hidden_states = orig_forward(hidden_states)  # type: ignore
        return hidden_states[..., :n, :]

    norm_layer.forward = forward_  # type: ignore
    return norm_layer


def register_shared_norm(
    model: ModelPatcher,
    share_group_norm: bool = True,
    share_layer_norm: bool = True,
):
    norm_layers = {"group": [], "layer": []}
    get_norm_layers(model.model, norm_layers, share_layer_norm, share_group_norm)
    print(
        f"Patching {len(norm_layers['group'])} group norms, {len(norm_layers['layer'])} layer norms."
    )
    return [register_norm_forward(layer) for layer in norm_layers["group"]] + [
        register_norm_forward(layer) for layer in norm_layers["layer"]
    ]


SHARE_NORM_OPTIONS = ["both", "group", "layer", "disabled"]
SHARE_ATTN_OPTIONS = ["q+k", "q+k+v", "disabled"]


class StyleAlignedReferenceSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "share_norm": (SHARE_NORM_OPTIONS,),
                "share_attn": (SHARE_ATTN_OPTIONS,),
                "scale": ("FLOAT", {"default": 1, "min": 0, "max": 2.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "ref_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "patch"
    CATEGORY = "style_aligned"

    def patch(
        self,
        model: ModelPatcher,
        share_norm: str,
        share_attn: str,
        scale: float,
        batch_size: int,
        noise_seed: int,
        cfg: float,
        positive: T,
        negative: T,
        sampler: T,
        sigmas: T,
        ref_latent: "dict[str, T]",
    ) -> "tuple[dict, dict]":
        m = model.clone()
        args = StyleAlignedArgs(share_attn)

        # Concat batch with style latent
        style_latent_tensor = ref_latent["samples"]
        height, width = style_latent_tensor.shape[-2:]
        latent_t = torch.zeros(
            [batch_size, 4, height, width], device=ref_latent["samples"].device
        )
        latent = {"samples": latent_t}
        noise = comfy.sample.prepare_noise(latent_t, noise_seed)

        latent_t = torch.cat((style_latent_tensor, latent_t), dim=0)
        ref_noise = torch.zeros_like(noise[0]).unsqueeze(0)
        noise = torch.cat((ref_noise, noise), dim=0)

        x0_output = {}
        callback = latent_preview.prepare_callback(m, sigmas.shape[-1] - 1, x0_output)

        # Register shared norms
        share_group_norm = share_norm in ["group", "both"]
        share_layer_norm = share_norm in ["layer", "both"]
        register_shared_norm(m, share_group_norm, share_layer_norm)

        # Patch cross attn
        m.set_model_attn1_patch(SharedAttentionProcessor(args, scale))

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample_custom(
            m,
            noise,
            cfg,
            sampler,
            sigmas,
            positive,
            negative,
            latent_t,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
        )

        # remove reference image
        samples = samples[1:]

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            x0 = x0_output["x0"][1:]
            out_denoised["samples"] = m.model.process_latent_out(x0.cpu())
        else:
            out_denoised = out
        return (out, out_denoised)


class StyleAlignedBatchAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "share_norm": (SHARE_NORM_OPTIONS,),
                "share_attn": (SHARE_ATTN_OPTIONS,),
                "scale": ("FLOAT", {"default": 1, "min": 0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "style_aligned"

    def patch(
        self,
        model: ModelPatcher,
        share_norm: str,
        share_attn: str,
        scale: float,
    ):
        m = model.clone()
        share_group_norm = share_norm in ["group", "both"]
        share_layer_norm = share_norm in ["layer", "both"]
        register_shared_norm(model, share_group_norm, share_layer_norm)
        args = StyleAlignedArgs(share_attn)
        m.set_model_attn1_patch(SharedAttentionProcessor(args, scale))
        return (m,)


NODE_CLASS_MAPPINGS = {
    "StyleAlignedReferenceSampler": StyleAlignedReferenceSampler,
    "StyleAlignedBatchAlign": StyleAlignedBatchAlign,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleAlignedReferenceSampler": "StyleAligned Reference Sampler",
    "StyleAlignedBatchAlign": "StyleAligned Batch Align",
}
