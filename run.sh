
export HF_DATASETS_CACHE="/root/autodl-tmp/.cache/dataset"

export HF_HOME="/root/autodl-tmp/.cache"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"

# torchrun --nproc_per_node=4 ./bradley-terry-rm/gemma_2b_BT.py --model_name google/gemma-2b-it --max_length 4096 --train_set_path hendrydong/preference_700K
deepspeed --num_gpus=4 ./bradley-terry-rm/gemma_2b_BT.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --max_length 2048 --train_set_path hendrydong/preference_700K