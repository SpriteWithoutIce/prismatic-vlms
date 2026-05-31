# export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
    --model.type prism-qwen3-vjepa21-vitl-384px+1_7b \
    --dataset.type llava-v15 \
    --dataset.dataset_root_dir /ssd/linyihan/datasets \
    --model.vision_checkpoint_path /ssd/linyihan/ckpt/vjepa2_1_vitl_dist_vitG_384.pt \
    --model.llm_local_path /ssd/linyihan/ckpt/Qwen3-1.7B-Base \
    --stage finetune \
    --model.finetune_per_device_batch_size 2 \
    --model.finetune_global_batch_size 32 \
    --run_root_dir ./runs \
    --trackers '["jsonl", "wandb"]' \
    --wandb_project VJEPA_Qwen3_LLaVA_pretrain \
    --wandb_entity 22373442
