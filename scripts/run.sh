# export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
    --model.type prism-qwen25-vjepa21-vitg-384px+0_5b \
    --dataset.type llava-v15 \
    --dataset.dataset_root_dir /ssd/linyihan/datasets \
    --model.vision_checkpoint_path /ssd/linyihan/ckpt/vjepa2_1_vitg_dist_vitG_384.pt \
    --model.llm_local_path /ssd/linyihan/ckpt/Qwen2.5-0.5B \
    --stage finetune \
    --model.finetune_per_device_batch_size 4 \
    --model.finetune_global_batch_size 32 \
    --run_root_dir ./runs \
    --trackers '["jsonl", "wandb"]' \
    --wandb_project VJEPA_Qwen25_LLaVA_pretrain \
    --wandb_entity 22373442
