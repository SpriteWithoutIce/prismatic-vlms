torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
    --model.type prism-qwen25-vjepa21-vitl-384px+0_5b \
    --dataset.type llava-v15 \
    --dataset.dataset_root_dir /home/jwhe/linyihan/datasets/llava-v1.5-instruct \
    --model.vision_checkpoint_path /home/jwhe/linyihan/CKPT/vjepa2_1_vitl_384.pt \
    --model.llm_local_path /home/jwhe/linyihan/CKPT/Qwen2.5-0.5B \
    --stage finetune \
    --run_root_dir /path/to/runs \
    --trackers '["jsonl","wandb"]' \
    --wandb_project V-JEPA_pretrain \