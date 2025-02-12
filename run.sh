

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch llama_small.py --extract_layers 3 7 11 --train_data_path "/root/train_1" --output_path "/root/autodl-tmp/weak_superviser"

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch llama_small.py --extract_layers 3 7 --train_data_path "/root/autodl-tmp/train_1" --output_path "/root/autodl-tmp/weak_superviser"

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch llama_small.py --extract_layers 3 7 11 --train_data_path "/root/autodl-tmp/train_1" --output_path "/root/autodl-tmp/weak_superviser"

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch llama_small.py --extract_layers 3 7 11 15 --train_data_path "/root/autodl-tmp/train_1" --output_path "/root/autodl-tmp/weak_superviser"


# --deepspeed ./deepspeed_configs/deepspeed_3.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch weak_to_strong.py --deepspeed ./deepspeed_configs/deepspeed_1.json
