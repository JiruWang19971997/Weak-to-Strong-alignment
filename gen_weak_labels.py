########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Mistral model which should be handled carefully because it does not have an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_from_disk, Dataset
from transformers import EarlyStoppingCallback
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

from model.model_builder import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 注意：PORT需要是开放的
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    
    
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

class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feature = self.dataset[idx]
        return feature  # 这里直接返回原始 feature

    
 
def compute(rank, world_size):
    
    
    model_path = "/root/0.51B"
    output_path = "soft_weak"
    setup(rank, world_size)
    tokenizer_name = model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = 2048
    llama_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    ).to(rank)
    ddp_model = DDP(llama_model, device_ids=[rank])



    ddp_model = ddp_model.eval()

    output_name = f"{output_path}_{str(rank)}"
    dataset = load_from_disk("/root/train_2")

    data_collator = RewardDataCollatorWithPadding(tokenizer=tokenizer)
    reward_dataset = RewardDataset(dataset)
    sampler = torch.utils.data.DistributedSampler(reward_dataset, num_replicas=world_size, rank=rank)


    dataloader = DataLoader(reward_dataset, batch_size=8, collate_fn=data_collator, sampler=sampler)

    new_data = {
            'input_ids_j': [],
            'input_ids_k': [],
            'attention_mask_j': [],
            'attention_mask_k': [],
            'logits_j': [],
            'logits_k': [],
        }
    all_predictions = []
    c = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference Progress", dynamic_ncols=True):  # 添加进度条
            # Send batch to device
            input_ids = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            

            # Forward pass
            outputs = ddp_model(input_ids=input_ids, attention_mask=attention_mask)

            # Get the logits (you can adjust depending on your task)
            logits = outputs[0]
            
            
            bsz = input_ids.size(0)
            jidx = torch.arange(0, bsz, 2)
            kidx = jidx + 1
            
            input_ids_j = input_ids[jidx]
            input_ids_k = input_ids[kidx]
            attention_mask_j = attention_mask[jidx]
            attention_mask_k = attention_mask[kidx]

            # 使用类似的方式提取logits
            logits_j = logits[jidx]
            logits_k = logits[kidx]
            new_data['input_ids_j'].extend(input_ids_j.cpu().numpy())
            new_data['input_ids_k'].extend(input_ids_k.cpu().numpy())
            new_data['attention_mask_j'].extend(attention_mask_j.cpu().numpy())
            new_data['attention_mask_k'].extend(attention_mask_k.cpu().numpy())
            new_data['logits_j'].extend(logits_j.to(torch.float32).cpu().numpy())
            new_data['logits_k'].extend(logits_k.to(torch.float32).cpu().numpy())

    new_dataset = Dataset.from_dict(new_data)

    # Optionally save the new dataset
    new_dataset.save_to_disk(f"/root/autodl-tmp/{output_name}_{str(rank)}")

    # Print or process new dataset
    print(new_dataset)        

            
            
            
if __name__ == "__main__":
    world_size = 4
    torch.multiprocessing.spawn(compute, args=(world_size,), nprocs=world_size, join=True)

