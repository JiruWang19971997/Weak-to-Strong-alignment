import torch
import torch.nn as nn
import deepspeed
import json
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, HfArgumentParser
from tqdm import tqdm
import os
import torch.distributed as dist
import logging
from utils.data import RewardDataCollatorWithPadding
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# **初始化 DeepSpeed**
deepspeed.init_distributed()
local_rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)

# **日志配置**
log_file = f"deepspeed_rank_{local_rank}.log"
logging.basicConfig(filename=log_file, filemode='a', format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("DeepSpeed")

# **训练参数**
@dataclass
class ScriptArguments:
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(default="decapoda-research/llama-7b-hf")
    bf16: Optional[bool] = field(default=True)
    num_train_epochs: Optional[int] = field(default=1)
    train_set_path: Optional[str] = field(default="hendrydong/preference_700K")
    eval_set_path: Optional[str] = field(default="hendrydong/preference_700K")
    output_path: Optional[str] = field(default="./bt_models/llama")
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: Optional[str] = field(default="cosine")
    max_length: Optional[int] = field(default=4096)
    save_steps: Optional[int] = field(default=1000)

parser = HfArgumentParser(ScriptArguments)
script_args, unknown = parser.parse_args_into_dataclasses(return_remaining_strings=True)

# **加载 tokenizer**
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length

# **加载数据**
def build_dataset(tokenizer, train_path, eval_path):
    def tokenize(sample):
        sample['positive'] = tokenizer.apply_chat_template(sample['chosen'], tokenize=False).replace(tokenizer.bos_token, "")
        sample['negative'] = tokenizer.apply_chat_template(sample['rejected'], tokenize=False).replace(tokenizer.bos_token, "")

        sample["input_ids_j"] = tokenizer(sample['positive'], truncation=True)["input_ids"]
        sample["attention_mask_j"] = tokenizer(sample['positive'], truncation=True)["attention_mask"]
        sample["input_ids_k"] = tokenizer(sample['negative'], truncation=True)["input_ids"]
        sample["attention_mask_k"] = tokenizer(sample['negative'], truncation=True)["attention_mask"]
        return sample

    ds = load_dataset(train_path, split="train").shuffle(seed=42).map(tokenize, num_proc=8)
    eval_ds = ds.select(range(500))
    return ds, eval_ds

train_dataset, eval_dataset = build_dataset(tokenizer, script_args.train_set_path, script_args.eval_set_path)

# **加载 DeepSpeed 配置**
with open("/root/RLHF-Reward-Modeling/bradley-terry-rm/deepspeed_config.json", "r") as f:
    ds_config = json.load(f)

# **创建 DataLoader**
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
train_dataloader = DataLoader(train_dataset, batch_size=ds_config["train_micro_batch_size_per_gpu"], sampler=train_sampler, collate_fn=RewardDataCollatorWithPadding(tokenizer))

eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=local_rank)
eval_dataloader = DataLoader(eval_dataset, batch_size=script_args.per_device_eval_batch_size, sampler=eval_sampler)

# **加载模型**
quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # 8-bit 量化
model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name,
    num_labels=1,
    # quantization_config=quantization_config,
    device_map="auto"
)
model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

# **DeepSpeed 初始化**
# 仅传递优化器参数，不传递 model
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    # model_parameters=model.parameters(),
    dist_init_required=True  # 确保分布式初始化
)

# **损失函数**
def compute_loss(model, inputs):
    rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
    jidx = torch.arange(0, rewards.size(0), 2)
    kidx = jidx + 1
    return -nn.functional.logsigmoid(rewards[jidx] - rewards[kidx]).mean()

# **评估**
def evaluate(model, eval_dataloader):
    model.eval()
    pos_scores, neg_scores = [], []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            reward_j = model(input_ids=batch["input_ids_j"], attention_mask=batch["attention_mask_j"])[0]
            reward_k = model(input_ids=batch["input_ids_k"], attention_mask=batch["attention_mask_k"])[0]
            pos_scores.extend(reward_j.cpu().numpy())
            neg_scores.extend(reward_k.cpu().numpy())

    accuracy = np.mean(np.array(pos_scores) > np.array(neg_scores))
    return {"accuracy": accuracy}

# **训练循环**
best_eval_accuracy = 0
for epoch in range(script_args.num_train_epochs):
    model_engine.train()  # 用 model_engine，而不是 model
    for step, batch in enumerate(tqdm(train_dataloader)):
        loss = compute_loss(model, batch)
        model_engine.backward(loss)  # 用 model_engine 进行反向传播
        model_engine.step()  

        if (step + 1) % script_args.save_steps == 0:
            eval_results = evaluate(model, eval_dataloader)
            print(f"[Evaluation] Step {step+1}, Accuracy: {eval_results['accuracy']:.4f}")
            if eval_results["accuracy"] > best_eval_accuracy and local_rank == 0:
                best_eval_accuracy = eval_results["accuracy"]
                model.save_checkpoint(script_args.output_path)
                print(f"Saved Best Model at {script_args.output_path}")

    if local_rank == 0:
        model.save_checkpoint(script_args.output_path)
