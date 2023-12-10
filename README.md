# StyleAligned for ComfyUI

Implementation of the [StyleAligned](https://style-aligned-gen.github.io/) paper for ComfyUI.

### Installation

Simply run `git clone git@github.com:brianfitzgerald/style_aligned_comfy.git` in your `custom_nodes` directory.

### Usage

Use the example workflow from [here](resources/example_workflow.json)
Or, simply add the `StyleAlignedPatch` node after `LoadCheckpoint`.

### Parameters

- `model`: Required, the base model to patch.
- `style_image`: Optional, path to the latent to use as a style inference. If left blank, the first latent in the batch will be used, effectively making the output of the batch consistent.
- `share_norm`: Whether to share normalization across the batch. Defaults to `both`. Set to `group` or `layer` to only share group or layer normalization, respectively.
- `scale`: The scale at which to apply the style-alignment effect. Defaults to `1`.