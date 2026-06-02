#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPUS="${GPUS:-8}"
DATA_ROOT="${DATA_ROOT:-data}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-runs}"
WANDB_PROJECT="${WANDB_PROJECT:-prismatic-vjepa}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
HF_TOKEN="${HF_TOKEN:-.hf_token}"

torchrun --standalone --nnodes 1 --nproc-per-node "${GPUS}" scripts/pretrain.py \
    --model.type "prism-qwen25-vjepa21-vitl-384px+0_5b+fusedfan-projector" \
    --dataset.type "llava-lvis4v-lrv" \
    --dataset.dataset_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT_DIR}" \
    --stage "finetune" \
    --hf_token "${HF_TOKEN}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}"
