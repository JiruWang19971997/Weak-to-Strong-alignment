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
    TrainingArguments,
    TrainerCallback
)
from transformers.utils import PaddingStrategy
from model.model_builder import *
accelerator = Accelerator()
device = accelerator.device

local_rank = accelerator.state.process_index
from model.model_builder import *

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
    student_model_path: Optional[str] = field(
        default="/root/autodl-tmp/Llama-3.2-1B-Instruct",
        metadata={"help": "prepossed model weight"},
    )
    teacher_model_path: Optional[str] = field(
        default="/root/autodl-tmp/0.51B",
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
    train_data_path: Optional[str] = field(
        default="/root/train_small",
        metadata={"help": "The dir of the whole dataset prepossed"},
    )
    output_path: Optional[str] = field(
        default="/root/autodl-tmp/weak_to_strong_",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="adamw_hf",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_every_steps: Optional[int] = field(
        default=200,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=200,
        metadata={"help": "Eval the model every x steps"},)
    student_train_layer: Optional[int] = field(
        default=2,
        metadata={"help": "trainable student model layer"},
    )
    
    
    
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# load tokenizer(teacher and student model share the same trained tokenizer)
tokenizer_name = script_args.teacher_model_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
# load strong student model 

model_path = script_args.student_model_path
strong_student_model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    torch_dtype=torch.bfloat16,
)

# # Set use_cache and pad_token_id based on gradient checkpointing and tokenizer
strong_student_model.config.use_cache = not script_args.gradient_checkpointing
strong_student_model.config.pad_token_id = tokenizer.pad_token_id
strong_student_model.resize_token_embeddings(len(tokenizer))


# Prune layers based on student_layers
# for i in reversed(range(len(strong_student_model.model.layers))):
#     if i not in script_args.student_layers:
#         del strong_student_model.model.layers[i]

# # Adjust the number of hidden layers to match the pruned model
# strong_student_model.config.num_hidden_layers = len(script_args.student_layers)

# Move model to the appropriate device (i.e., accelerator device)

# load weak teacher model 
model_path = script_args.teacher_model_path
weak_teacher_model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    torch_dtype=torch.bfloat16,
)
distill_model = DistillationModel(student_model = strong_student_model, teacher_model = weak_teacher_model, student_train_layer = script_args.student_train_layer)
model = accelerator.prepare(distill_model)
output_name = script_args.output_path + "_" + str(script_args.student_train_layer)

# load the dataset/
dataset = load_from_disk(script_args.train_data_path)
split_dataset = dataset.train_test_split(test_size=1000)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))

# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.save_every_steps,
    save_strategy="steps",  
    save_steps=script_args.save_every_steps,
    save_total_limit=1,   
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
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
    student_output, _  = eval_pred.predictions
    pos_predictions_scores = student_output[::2]
    neg_predictions_scores = student_output[1::2] 
    # We assume that the first sample is preferred by default in groundtruth
    result["accuracy"] = np.sum(pos_predictions_scores > neg_predictions_scores) / len(
        pos_predictions_scores
    )
    return result


class DistillationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = nn.MSELoss()
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_logits, teacher_logits = model(
            inputs=inputs["input_ids"], attention=inputs["attention_mask"]
        )
        assert student_logits[0].shape == teacher_logits[0].shape
        loss = self.mse_loss(student_logits[0], teacher_logits[0])
        if return_outputs:
            return loss, {"student_logits": student_logits[0], "teacher_logits": teacher_logits[0]}
        return loss


class EvaluationCallback(TrainerCallback):
    def __init__(self, trainer, eval_steps=50, save_dir="best_model"):
        self.trainer = trainer
        self.eval_steps = eval_steps
        self.best_accuracy = 0.0
        self.save_dir = save_dir

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            metrics = self.trainer.evaluate()
            accuracy = metrics.get("eval_accuracy", 0.0)
            print(f"🔥 Step {state.global_step}: Accuracy = {accuracy:.4f}")

            # 如果当前 accuracy 是最高的，则保存模型
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print(f"✅ New best accuracy! Saving student model to {self.save_dir}")
                
                # 保存学生模型（假设 student_model 是 distill_model 中的学生模型部分）
                student_model = self.trainer.model.student_model  # 获取学生模型
                student_model.save_pretrained(self.save_dir)  # 保存学生模型

# Train the model, woohoo.
trainer = DistillationTrainer(
    model=distill_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length
    ),
)

# 添加 EvaluationCallback
trainer.add_callback(EvaluationCallback(trainer, eval_steps=script_args.eval_every_steps, save_dir=output_name))



last_checkpoint = None
if os.path.isdir(output_name):
    checkpoints = [d for d in os.listdir(output_name) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(output_name, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()


print("Saving last checkpoint of the model")
# model.save_pretrained(output_name + "/last_checkpoint")
trainer.save_model(output_name + "/last_checkpoint")
tokenizer.save_pretrained(output_name + "/last_checkpoint")
