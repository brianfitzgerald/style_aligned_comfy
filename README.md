# StyleAligned for ComfyUI

Implementation of the [StyleAligned](https://style-aligned-gen.github.io/) technique for ComfyUI.

This implementation is split into two different nodes, and does not require any additional models or dependencies.

#### StyleAligned Reference Sampler

This node replaces the KSampler, and lets you reference an existing latent as a style reference. In order to retrieve the latent, you will need to perform DDIM inversion; an example workflow for this is provided [here](resources/style_aligned_inversion.json).

![](resources/reference_sampler.png)

_Above, a reference image, and a batch of images generated using the prompt 'a robot' and the reference image shown as style input._

##### Parameters

- `model`: The base model to patch.
- `share_attn`: Which components of self-attention are normalized. Defaults to `q+k`. Set to `q+k+v` for more extreme sharing, at the cost of quality in some cases.
- `share_norm`: Whether to share normalization across the batch. Defaults to `both`. Set to `group` or `layer` to only share group or layer normalization, respectively.
- `scale`: The scale at which to apply the style-alignment effect. Defaults to `1`.

#### StyleAligned Batch Align

Instead of referencing a single latent, this node aligns the style of the entire batch with the first image generated in the batch, effectively causing all images in the batch to be generated with the same style.

![](resources/batch_align.jpg)

_A batch of generations with the same parameters and the Batch Align node applied (left) and disabled (right)._

##### Parameters

- `model`: The base model to patch.
- `share_attn`: Which components of self-attention are normalized. Defaults to `q+k`. Set to `q+k+v` for more extreme sharing, at the cost of quality in some cases.
- `share_norm`: Whether to share normalization across the batch. Defaults to `both`. Set to `group` or `layer` to only share group or layer normalization, respectively.
- `scale`: The scale at which to apply the style-alignment effect. Defaults to `1`.
- `batch_size`, `noise_seed`, `control_after_generate`, `cfg`: Identical to the standard `KSampler` parameters.

### Installation

Simply download or git clone this repository in `ComfyUI/custom_nodes/`. Example workflows are included in `resources/`.