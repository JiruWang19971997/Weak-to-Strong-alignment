########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Mistral model which should be handled carefully because it does not have an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os
# import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
import torch.distributed as dist
from accelerate import Accelerator
from transformers.utils import PaddingStrategy
accelerator = Accelerator()

local_rank = accelerator.state.process_index

os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/.cache/dataset"
os.environ["HF_HOME"] = "/root/autodl-tmp/.cache"

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    # for 8 GPU, the global batch size is 512
    gradient_accumulation_steps: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=2e-6)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    teacher_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
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
    train_set_path: Optional[str] = field(
        default="hendrydong/preference_700K",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default="hendrydong/preference_700K",
        metadata={"help": "The dir of the subset of the eval data to use"},
    )
    output_path: Optional[str] = field(
        default="/root/autodl-tmp/llama3",
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


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the value-head model and tokenizer.
tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct" # all llama model use the same tokenizer, and cached dataset
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length



# Get the dataset
train_path = script_args.train_set_path
eval_path = script_args.eval_set_path
output_name = script_args.output_path


# def build_dataset(tokenizer, train_path, eval_path, seed=42):

#     def tokenize(sample):
#         sample['positive'] = tokenizer.apply_chat_template(
#             sample['chosen'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
#         sample['negative'] = tokenizer.apply_chat_template(
#             sample['rejected'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
#         tokenized_pos = tokenizer(sample['positive'], truncation=True)
#         tokenized_neg = tokenizer(sample['negative'], truncation=True)
#         sample["input_ids_j"] = tokenized_pos["input_ids"]
#         sample["attention_mask_j"] = tokenized_pos["attention_mask"]
#         sample["input_ids_k"] = tokenized_neg["input_ids"]
#         sample["attention_mask_k"] = tokenized_neg["attention_mask"]
#         return sample
    
#     ds = load_dataset(train_path, split="train", cache_dir=os.environ["HF_DATASETS_CACHE"])
#     ds = ds.map(tokenize, num_proc=4)
    
#     split_ds = ds.train_test_split(test_size=0.5, seed=seed)  
#     train_dataset = split_ds["train"]
#     train_dataset_2 = split_ds["test"]  
        

#     eval_dataset = None
#     #eval_dataset = load_dataset(eval_path, split="train").shuffle(seed=42).select(range(500))
#     eval_dataset = ds.select(range(1000))
#     return train_dataset, train_dataset_2,  eval_dataset


# train_dataset, train_dataset_2, eval_dataset = build_dataset(tokenizer, train_path, eval_path)
# print("Training set: ", len(train_dataset_2), " Eval set: ", len(eval_dataset))

# Define the trainer
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
    report_to='wandb',
    disable_tqdm=not accelerator.is_main_process
)



weak_teacher_model = AutoModelForSequenceClassification.from_pretrained(
    script_args.teacher_name, num_labels=1, torch_dtype=torch.bfloat16, use_flash_attention_2=True,
)


model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16, use_flash_attention_2=True,
)

model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# We need to define a special data collator that batches the data in our j vs k format.
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


# Define the trainer
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result['accuracy'] = np.sum(
        pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    return result

class DistillationTrainer(Trainer):
    def __init__(self, weak_teacher_model, *args, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.weak_teacher_model = weak_teacher_model
        self.temperature = temperature  # 蒸馏温度
        self.alpha = alpha  # 控制蒸馏损失和原始损失的权重
        self.weak_teacher_model.eval()  # 设为评估模式，避免更新权重

    def compute_loss(self, model, inputs, return_outputs=False):
        # 计算学生模型的预测
        student_logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]

        # 计算教师模型的预测（不计算梯度）
        with torch.no_grad():
            teacher_logits = self.weak_teacher_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )[0]

        # 计算原始奖励模型损失
        bsz = student_logits.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = student_logits[jidx]
        rewards_k = student_logits[kidx]
        original_loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()

        # 计算 KL 散度损失
        distillation_loss = nn.KLDivLoss(reduction="batchmean")(
            torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1),
            torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1),
        )

        # 总损失 = 原始损失 + 蒸馏损失（加权组合）
        loss = self.alpha * original_loss + (1 - self.alpha) * distillation_loss

        if return_outputs:
            return loss, {"original_loss": original_loss, "distillation_loss": distillation_loss}
        return loss


# Train the model, woohoo.
trainer = DistillationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_2,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length),
)


trainer.train()


print("Saving last checkpoint of the model")
#model.save_pretrained(output_name + "/last_checkpoint")
trainer.save_model(output_name + "/last_checkpoint")
tokenizer.save_pretrained(output_name + "/last_checkpoint")
