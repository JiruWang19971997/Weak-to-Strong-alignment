import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Union
from transformers.utils import PaddingStrategy
import argparse
from dataclasses import dataclass
from tqdm import tqdm
from accelerate import Accelerator

# 设置Accelerator
accelerator = Accelerator()

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Training a model on a dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    return parser.parse_args()

# 加载弱教师模型
def load_model(model_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    return model

# 加载分词器
def load_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = 2048
    return tokenizer

# 加载数据集
def load_dataset(data_path: str):
    return load_from_disk(data_path)

# 数据预处理器
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch

# 计算指标函数
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred[0][::2]  # 取出正样本得分
    neg_predictions_scores = eval_pred[0][1::2]  # 取出负样本得分
    acc = (pos_predictions_scores > neg_predictions_scores).float().mean()
    return acc

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()

    # 加载模型和数据
    llama_model = load_model(args.model_path)
    tokenizer = load_tokenizer(args.tokenizer_path)
    train_dataset = load_dataset(args.data_path)

    batch_size = 4  # 批次大小
    collator = RewardDataCollatorWithPadding(tokenizer=tokenizer)

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # 使用Accelerator进行设备管理
    llama_model = accelerator.prepare(llama_model)  # 准备模型，自动分配到可用的设备
    train_dataloader = accelerator.prepare(train_dataloader)  # 准备数据加载器

    # 评估模式
    llama_model.eval()

    acc = []
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Processing Batches"):  # 添加进度条
            inputs = {key: value for key, value in batch.items() if key in ["input_ids", "attention_mask"]}
            outputs = llama_model(**inputs)
            acc.append(compute_metrics(outputs))

    # 计算平均准确率
    average_acc = torch.mean(torch.tensor(acc))  # 将 acc 转换为 tensor 计算均值
    print("Average Accuracy:", average_acc.item())  # .item() 取出 Python 数值

# 运行主函数
if __name__ == "__main__":
    main()
