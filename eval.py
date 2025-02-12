import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Union
from transformers.utils import PaddingStrategy

# Load weak teacher model
model_path = "/root/autodl-tmp/weak_superviser_0.32B/last_checkpoint"
weak_teacher_model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Load dataset
train_dataset = load_from_disk("/root/autodl-tmp/train_small")
from dataclasses import dataclass, field

# 创建数据加载器
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
    
batch_size = 1  # 你可以调整批次大小
collator = RewardDataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collator  # 修正参数名称
)

# 评估模式
weak_teacher_model.eval()

def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[:, 0]  # 取出正样本得分
    neg_predictions_scores = eval_pred.predictions[:, 1]  # 取出负样本得分
    result["accuracy"] = np.mean(pos_predictions_scores > neg_predictions_scores)
    return result

# 预测
all_predictions = []





with torch.no_grad():
    for batch in train_dataloader:
        inputs = {key: value for key, value in batch.items() if key in ["input_ids", "attention_mask"]}
        outputs = weak_teacher_model(**inputs)
        predictions = outputs.logits.cpu().to(torch.float32).numpy()
        all_predictions.append(predictions)

# 处理预测结果
all_predictions = np.concatenate(all_predictions, axis=0)

# 计算评估指标
metrics = compute_metrics(eval_pred={"predictions": all_predictions})

print("Evaluation Metrics:", metrics)
