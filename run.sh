
export HF_DATASETS_CACHE="/root/autodl-tmp/.cache/dataset"

export HF_HOME="/root/autodl-tmp/.cache"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ./bradley-terry-rm/llama3.py --model_name "meta-llama/Llama-3.2-3B-Instruct"  --max_length 2048 --train_set_path hendrydong/preference_700K  --role "teacher"
# --deepspeed ./deepspeed_configs/deepspeed_1.json

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ./bradley-terry-rm/llama3.py --model_name "meta-llama/Llama-3.2-1B-Instruct"  --max_length 2048 --train_set_path hendrydong/preference_700K  --role "student"
