# StyleAligned for ComfyUI

Implementation of the [StyleAligned](https://style-aligned-gen.github.io/) paper for ComfyUI.

This node allows you to apply a consistent style to all images in a batch; by default it will use the first image in the batch as the style reference, forcing all other images to be consistent with it.

![](resources/header.jpg)

_A batch of generations with the same parameters, with the node applied (left) and without (right)._

In the next few days I plan on implementing the second feature of the paper - the ability to use another image as the style reference.

### Installation

Simply download or git clone this repository in `ComfyUI/custom_nodes/`.

### Usage

Use the example workflow from [here](resources/example_workflow.json).

Or, simply add the `StyleAlignedPatch` node after `LoadCheckpoint`.

### Parameters

- `model`: Required, the base model to patch.
- `style_image`: (**not implemented yet!**) Optional, path to the latent to use as a style inference. If left blank, the first latent in the batch will be used, effectively making the output of the batch consistent.
- `share_norm`: Whether to share normalization across the batch. Defaults to `both`. Set to `group` or `layer` to only share group or layer normalization, respectively.
- `scale`: The scale at which to apply the style-alignment effect. Defaults to `1`.