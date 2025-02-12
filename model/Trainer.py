########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Mistral model which should be handled carefully because it does not have an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_from_disk

# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    LlamaConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

class MoeTrainer(Trainer):
    def __init__(self, moe_model):
        super().__init__(*args, **kwargs)
        self.moe_model = moe_model
    #     self.moe_model = moe_model_list
    #     self.gating_model = gating_model
    #     self.temperature = temperature  # 蒸馏温度
    #     self.alpha = alpha  # 控制蒸馏损失和原始损失的权重
    #     self.mse = nn.MSELoss()

    def compute_loss(self, inputs, return_outputs=False):
        import pdb;pdb.set_trace()
        
        student_logits, DIS_LOSS = self.moe_model()(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0].to(torch.bfloat16)
        
        
        # 计算教师模型的预测（不计算梯度）
        # with torch.no_grad():
        #     teacher_logits = self.weak_teacher_model(
        #         input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        #     )[0]
        # 计算 KL 散度损失
        loss = self.mse(student_logits, teacher_logits)
        return loss