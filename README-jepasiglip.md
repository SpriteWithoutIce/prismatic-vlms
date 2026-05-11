# JEPA+SigLIP Integration Notes

This repo now includes a fused `JEPASigLIP` vision backbone and a matching
`prism-jepasiglip+0_5b` training config.

## Goal

Reuse the existing Prismatic training stack while replacing the original
single-tower V-JEPA setup with a two-tower vision encoder:

- `V-JEPA 2.1 ViT-L 384px`
- `SigLIP SO400M 384px`
- `Qwen2.5-0.5B`

The target behavior matches the existing `DinoSigLIPViTBackbone` pattern:
extract patch features from each tower independently, align them to a common
grid, then concatenate the features along the channel dimension before sending
them into the projector.

## Design

The main implementation lives in
[prismatic/models/backbones/vision/jepasiglip_vit.py](/home/handoff/Desktop/prismatic-vlms/prismatic/models/backbones/vision/jepasiglip_vit.py:1).

Key choices:

- The fused backbone wraps the existing `VJEPA21ViTBackbone` instead of
  reimplementing V-JEPA checkpoint loading.
- SigLIP keeps the repo's default loading path through `timm`.
- Both towers use `384px` inputs and `resize-naive`.
- V-JEPA 384 and SigLIP 384 do not emit the same patch count because their patch
  sizes differ. The implementation resamples the SigLIP patch grid onto the
  JEPA grid before fusion.
- Fusion is still patch-level `torch.cat(..., dim=2)` after grid alignment.

## What Changed

- Added `JEPASigLIPViTBackbone`
- Registered `jepasiglip-vit-l-so-384px` in the vision backbone factory
- Added the model config `prism-jepasiglip+0_5b`
- Updated `scripts/run.sh` to train the new model

## Training Command

The default launcher is now [scripts/run.sh](/home/handoff/Desktop/prismatic-vlms/scripts/run.sh:1).

Important runtime inputs:

- `--model.vision_checkpoint_path` still points to the local V-JEPA checkpoint
- `--model.siglip_local_path` can point to a local SigLIP checkpoint directory or file
- `--stage finetune` keeps the existing one-stage Prismatic/Qwen workflow

## Constraints

- This first version assumes `384px` for both towers.
- It also assumes `resize-naive`, because the current V-JEPA wrapper does not
  support the same full resize policy surface as the TIMM-only fused backbones.
- The fused token grid follows the V-JEPA patch layout, with SigLIP resized to
  match it at runtime.
- No change was made to the training loop, dataset code, or projector logic.
  The integration is isolated to the vision backbone registry plus model config.
