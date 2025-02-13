

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch llama_small.py --extract_layers 3 7 11 15 --train_data_path "/root/train" --output_path "/root/autodl-tmp/"


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch llama_small.py --extract_layers 2 3 6 7 10 11 14 15 --train_data_path "/root/train" --output_path "/root/autodl-tmp/"
