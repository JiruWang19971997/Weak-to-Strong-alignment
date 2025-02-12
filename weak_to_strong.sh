

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate  launch --multi_gpu  weak_to_strong.py --train_data_path "/root/train_small" --output_path "/root/autodl-tmp/weak_to_strong"   --student_layers 3 7 11
# --deepspeed ./deepspeed_configs/deepspeed_3.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch weak_to_strong.py --deepspeed ./deepspeed_configs/deepspeed_1.json
