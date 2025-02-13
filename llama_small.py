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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

from model.model_builder import *

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
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.01)
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
    train_data_path: Optional[str] = field(
        default="/root/dataset",
        metadata={"help": "The dir of the whole dataset prepossed"},
    )
    output_path: Optional[str] = field(
        default="/root/autodl-tmp",
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
    extract_layers: Optional[List[int]] = field(
        default_factory=lambda: [0],
        metadata={"help": "scale of small models of llama"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
model_path = script_args.model_path

tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length
llama_model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
)
llama_model.config.use_cache = False
llama_model.config.pad_token_id = tokenizer.pad_token_id
llama_model.resize_token_embeddings(len(tokenizer))


for i in reversed(range(len(llama_model.model.layers))):
    if i not in script_args.extract_layers:
        del llama_model.model.layers[i]


llama_model.config.num_hidden_layers = len(script_args.extract_layers)

model = llama_model
total_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Model size: {total_params:.2f}B parameters")

output_name = f"{script_args.output_path}{total_params:.2f}B"
print(f"Save path: {output_name}")


# Get the dataset
dataset = load_from_disk(script_args.train_data_path)
split_dataset = dataset.train_test_split(test_size=1000)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(
    f"Training set name: {script_args.train_data_path}",
    len(train_dataset),
    " Eval set: ",
    len(eval_dataset),
)

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
    save_steps=1000,
    save_total_limit=1,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=100,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to="wandb",
    disable_tqdm=not accelerator.is_main_process,
    resume_from_checkpoint=True,  # 允许从 checkpoint 继续训练
)


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
    result["accuracy"] = np.sum(pos_predictions_scores > neg_predictions_scores) / len(
        pos_predictions_scores
    )
    return result


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class EvaluationCallback(TrainerCallback):
    def __init__(self, trainer, eval_steps=50, save_dir="best_model"):
        self.trainer = trainer
        self.eval_steps = eval_steps
        self.best_accuracy = 0.0
        self.save_dir = save_dir

    def on_step_end(self, args, state, control, **kwargs):

        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            metrics = self.trainer.evaluate()
            accuracy = metrics.get("eval_accuracy", 0.0)
            print(f"🔥 Step {state.global_step}: Accuracy = {accuracy:.4f}")

            # 如果当前 accuracy 是最高的，则保存模型
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print(f"✅ New best accuracy! Saving model to {output_name}")
                self.trainer.save_model(output_name)


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length
    ),
)
trainer.add_callback(EvaluationCallback(trainer, eval_steps=50, save_dir="best_model"))


last_checkpoint = None
if os.path.isdir(output_name):
    checkpoints = [d for d in os.listdir(output_name) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(
            output_name, sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        )
        print(f"Resuming training from checkpoint: {last_checkpoint}")


trainer.train(resume_from_checkpoint=last_checkpoint)


print("Saving last checkpoint of the model")
# model.save_pretrained(output_name + "/last_checkpoint")
trainer.save_model(output_name + "/last_checkpoint")
tokenizer.save_pretrained(output_name + "/last_checkpoint")
