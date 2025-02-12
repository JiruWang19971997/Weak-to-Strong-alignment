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
from model.model_builder import *
from model.Trainer import *

accelerator = Accelerator()

local_rank = accelerator.state.process_index


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    model_path: Optional[str] = field(
        default="/root/autodl-tmp/Llama-3.2-1B-Instruct",
        metadata={"help": "prepossed model weight"},
    )

    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    # for 8 GPU, the global batch size is 512
    gradient_accumulation_steps: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=2e-6)
    weight_decay: Optional[float] = field(default=0.001)
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    data_path: Optional[str] = field(
        default="/root/autodl-tmp/dataset",
        metadata={"help": "The dir of the whole dataset prepossed"},
    )
    output_path: Optional[str] = field(
        default="/root/autodl-tmp/small_models",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )
    role: Optional[str] = field(
        default="teacher",
        metadata={"help": "Eval the model every x steps"},
    )
    


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

tokenizer_name = script_args.model_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length

model_path = script_args.model_path
config = LlamaConfig.from_pretrained(model_path, num_labels=1, torch_dtype=torch.bfloat16,use_flash_attention_2=True,)
config.num_hidden_layers //= 4
teacher_model = AutoModelForSequenceClassification.from_config(config)
print(teacher_model)
teacher_model.config.use_cache = not script_args.gradient_checkpointing
teacher_model.config.pad_token_id = tokenizer.pad_token_id
teacher_model.resize_token_embeddings(len(tokenizer))

student_model = AutoModelForSequenceClassification.from_config(config)


output_name = script_args.output_path

# # Get the dataset
train_dataset = load_from_disk("/root/eval")
eval_dataset = load_from_disk("/root/eval")
print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))
assert train_dataset is not None, f"Dataset {script_args.role} not found in dataset_dict!"

# # Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to="wandb",
    disable_tqdm=not accelerator.is_main_process,
)

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




class FeatureConnector(nn.Module):
    def __init__(self, t_dim, s_dim):
        super(FeatureConnector, self).__init__()
        self.proj = nn.Linear(s_dim, t_dim)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, x):
        return self.proj(x)

class MoETransformerDistiller(nn.Module):
    def __init__(self, teacher_models, student_model, distill_layers=[3, 6, 9, 12]):
        super(MoETransformerDistiller, self).__init__()
        self.student = student_model
        self.distill_layers = distill_layers
        
        t_hidden_dim = teacher_models[0].config.hidden_size
        s_hidden_dim = self.student.config.hidden_size
        
        self.connectors = nn.ModuleList([FeatureConnector(t_hidden_dim, s_hidden_dim) for _ in distill_layers])
        self.gating_network = nn.Linear(s_hidden_dim, len(teacher_models))  # 门控网络
        
        for i, teacher in enumerate(teacher_models):
            self.register_buffer(f'teacher_{i}', torch.tensor(0))  # 避免 Trainer 误优化教师
        
        self.teachers = teacher_models  # 存储教师模型
        for teacher in self.teachers:
            teacher.eval()  # 冻结教师模型

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            t_outputs = [teacher(input_ids, attention_mask=attention_mask, output_hidden_states=True) for teacher in self.teachers]
        
        s_outputs = self.student(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        s_hidden_states = s_outputs.hidden_states  # 学生的所有层输出
        loss_distill = 0
        
        for idx, layer in enumerate(self.distill_layers):
            teacher_features = torch.stack([t.hidden_states[layer] for t in t_outputs], dim=0)  # 组合所有教师特征
            gating_scores = F.softmax(self.gating_network(s_hidden_states[layer].mean(dim=1)), dim=-1)  # 计算权重
            weighted_teacher_feature = torch.einsum("tbsd,tb->bsd", teacher_features, gating_scores)  # 加权求和
            
            s_proj = self.connectors[idx](s_hidden_states[layer])  # 变换学生特征
            loss_distill += F.mse_loss(s_proj, weighted_teacher_feature.detach()) / len(self.distill_layers)
        
        loss_ce = F.cross_entropy(s_outputs.logits.view(-1, s_outputs.logits.size(-1)), labels.view(-1)) if labels is not None else 0
        loss = loss_distill + loss_ce
        
        return {'loss': loss, 'logits': s_outputs.logits, 'loss_distill': loss_distill}

# Trainer 适配器
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

# 训练参数
training_args = TrainingArguments(
    output_dir="./distilled_model",
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=1000,
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"
)

# 初始化模型
distiller_model = MoETransformerDistiller(teacher_models, student_model)

# Trainer 训练
trainer = DistillationTrainer(
    model=distiller_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()








# # We need to define a special data collator that batches the data in our j vs k format.
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







# Train the model, woohoo.
trainer = MoeTrainer(
    moe_model=moe_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length
    ),
)


# trainer.train()


# print("Saving last checkpoint of the model")
# # model.save_pretrained(output_name + "/last_checkpoint")
# trainer.save_model(output_name + "/last_checkpoint")
# tokenizer.save_pretrained(output_name + "/last_checkpoint")
