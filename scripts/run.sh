torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
    --model.type prism-jepasiglip+0_5b \
    --dataset.type llava-v15 \
    --dataset.dataset_root_dir /ssd/linyihan/datasets \
    --model.vision_checkpoint_path /ssd/linyihan/ckpt/vjepa2_1_vitl_dist_vitG_384.pt \
    --model.llm_local_path /ssd/linyihan/ckpt/Qwen2.5-0.5B \
    --stage finetune \
    --model.finetune_per_device_batch_size 4 \
    --run_root_dir ./runs \
    --trackers '["jsonl"]' \
    # --wandb_project JEPA_SigLIP_pretrain \
    # --wandb_entity 22373442
