import argparse
import json
import os
import random

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

processed_data_dir = "/root/dataset"


def preprocess_and_save(
    train_path, seed=42, tokenizer_name="meta-llama/Llama-3.2-1B-Instruct"
):
    """
    Preprocess dataset and save it as JSONL files.

    Args:
        train_path (str): Path to the dataset or Hugging Face dataset ID.
        seed (int, optional): Random seed for dataset splitting (default: 42).
        tokenizer_name (str, optional): Tokenizer model name (default: "meta-llama/Llama-3.2-1B-Instruct").
        cache_dir (str, optional): Directory for dataset caching (default: "/root/autodl-tmp/.cache/bakk").
    """

    # **1. Load tokenizer**
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = 2048

    # **2. Define tokenization function**
    def tokenize(sample):
        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")

        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)

        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]

        return sample

    if processed_data_dir and os.path.exists(processed_data_dir):
        # 如果处理后的数据集已存在，则直接加载
        dataset_dict = load_from_disk(processed_data_dir)
        train_dataset = dataset_dict["train_dataset"]
        train_dataset_2 = dataset_dict["train_dataset_2"]
        train_dataset_small = dataset_dict["train_small"]
        eval_dataset = dataset_dict["eval_dataset"]
    else:
        # 否则，加载原始数据集并进行处理
        ds = load_dataset(train_path, split="train")
        ds = ds.map(tokenize, num_proc=32)

        split_ds = ds.train_test_split(test_size=0.5, seed=seed)
        train_dataset = split_ds["train"]
        train_dataset_2 = split_ds["test"]

        eval_dataset = ds.select(range(1000))
        sample_size = len(train_dataset_2)

        if processed_data_dir:
            # 将处理后的数据集保存到指定目录
            dataset_dict = DatasetDict(
                {
                    "train": train_dataset,
                    "train_2": train_dataset_2,
                    "train_small": train_dataset_2.select(sample_size // 4),
                    "eval": eval_dataset,
                }
            )
            dataset_dict.save_to_disk(processed_data_dir)

    return


def show():
    """
    加载保存的数据集并显示前几个样本。

    参数:
        processed_data_dir (str): 保存的数据集目录路径。
        num_samples (int, optional): 要显示的样本数量，默认为 5。
    """
    # 加载保存的数据集
    dataset_dict = load_from_disk(processed_data_dir)

    # 获取训练集
    train_dataset = dataset_dict["train"]

    # 打印前几个样本
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        print(sample)
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset and save tokenized data"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to dataset or Hugging Face dataset ID",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Tokenizer model name",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for dataset splitting"
    )
    parser.add_argument(
        "--show", type=str, help="Path to JSONL file for dataset visualization"
    )

    args = parser.parse_args()

    if args.show:
        show()
    else:
        preprocess_and_save(
            train_path=args.train_path,
            tokenizer_name=args.tokenizer_name,
            seed=args.seed,
        )
