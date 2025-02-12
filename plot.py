########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Mistral model which should be handled carefully because it does not have an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from sklearn.manifold import TSNE

# import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer_name = "/root/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
model_path = "/root/Llama-3.2-1B-Instruct"


tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.truncation_side = "left"
tokenizer.model_max_length = 2048

# original_model
llama_model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    # use_flash_attention_2=True,
)
llama_model.config.use_cache = False
llama_model.config.pad_token_id = tokenizer.pad_token_id
llama_model.resize_token_embeddings(len(tokenizer))

extract_layers = [3,7,11]
for i in reversed(range(len([3,7,11]))):
    if i not in extract_layers:
        del llama_model.model.layers[i]


llama_model.config.num_hidden_layers = len(extract_layers)

# Groud_truth model
gt_model = AutoModelForSequenceClassification.from_pretrained(
    "/root/Weak-to-Strong-alignment/ground_truth_0.45B",
    num_labels=1,
    torch_dtype=torch.bfloat16,
    # use_flash_attention_2=True,
)

# inference
llama_model.to(device)
gt_model.to(device)

dataset = load_from_disk("/root/train_small").select(range(100))
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
train_dataloader = DataLoader(
    dataset, 
    batch_size=10,  # 设置适当的 batch size
    shuffle=False,  # 数据是否随机打乱
    num_workers=0,  # 设置为适当的线程数 (通常设置为 CPU 核心数或稍少)
    pin_memory=True,  # 如果使用 GPU，可以启用此选项
    collate_fn=RewardDataCollatorWithPadding(tokenizer)
)
import torch.nn.functional as F
from tqdm import tqdm  # 导入 tqdm 库

num_batches = 0
# layerwise_mse_sums = [0.0] * 4  # 存储所有 batch 的 MSE 累加

# # input_text = "Your input text here."
# # inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
# for idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Processing Batches"):
#     inputs = data["input_ids"].to(device)
#     attention = data["attention_mask"].to(device)


#     # Run inference on llama_model
#     llama_model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():  # Disable gradient computation for inference
#         llama_output = llama_model(inputs, attention, output_hidden_states=True)  # Feed the tokenized input
#     # Run inference on gt_model
#     gt_model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         gt_output = gt_model(inputs, attention, output_hidden_states=True)
#     llama_hidden_states = llama_output.hidden_states  # Tuple: 每层 hidden state
#     gt_hidden_states = gt_output.hidden_states  # Tuple: 每层 hidden state
#     for i, (llama_layer, gt_layer) in enumerate(zip(llama_hidden_states, gt_hidden_states)):
#         mse_loss = F.mse_loss(llama_layer, gt_layer)  # 计算 MSE Loss
#         layerwise_mse_sums[i] += mse_loss.item()  # 累加到总 Loss 中
#         if (idx+1) %100 == 0:
#             tmp_losses = [total_loss / (idx*8) for total_loss in layerwise_mse_sums]

#             print(tmp_losses)
# num_batches += 1 
# avg_mse_losses = [total_loss / (num_batches*8) for total_loss in layerwise_mse_sums]
# print(avg_mse_losses)
llama_hidden_states_sum = [None for i in range(4)]
gt_hidden_states_sum = [None for i in range(4)]
# Opt
for idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Processing Batches"):
    inputs = data["input_ids"].to(device)
    attention = data["attention_mask"].to(device)

    # 运行 llama_model
    llama_model.eval()
    with torch.no_grad():
        llama_output = llama_model(inputs, attention, output_hidden_states=True)
    
    # 运行 gt_model
    gt_model.eval()
    with torch.no_grad():
        gt_output = gt_model(inputs, attention, output_hidden_states=True)
    
    llama_hidden_states = llama_output.hidden_states  # tuple of hidden states
    gt_hidden_states = gt_output.hidden_states  # tuple of hidden states

    # 遍历每一层的输出并将每个样本的输出保存起来
    for i, (llama_layer, gt_layer) in enumerate(zip(llama_hidden_states, gt_hidden_states)):
        llama_layer = llama_layer.to(torch.float32)  # 转换为 Float32
        gt_layer = gt_layer.to(torch.float32)  # 转换为 Float32
        
        # flatten the output for each sample in the batch and append to the list
        # 将每个层的输出展平后添加到列表
        
        # import pdb;pdb.set_trace()
        
        # llama_flattened = llama_layer.cpu().numpy().reshape(llama_layer.shape[0], -1)
        # gt_flattened = gt_layer.cpu().numpy().reshape(gt_layer.shape[0], -1)

        # 初始化累加器（在第一次迭代时）
        if llama_hidden_states_sum[i] is None:
            llama_hidden_states_sum[i] = llama_layer
            gt_hidden_states_sum[i] = gt_layer
        else:
            llama_hidden_states_sum[i]+= llama_layer
            gt_hidden_states_sum[i] += gt_layer
    break

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# 假设 llama_hidden_states_sum 和 gt_hidden_states_sum 是长度为 4 的列表，
# 每个元素是模型每一层的输出，形状为 (20, 1282, 2048)
# 这表示模型的 4 个隐藏层的输出

# 将每一层的输出 reshape 为 (20 * 1282, 2048)
def reshape_for_tsne(output):
    return output.view(-1, output.shape[-1])  # 变为 (20*1282, 2048)

# 选择第一个模型和第二个模型的每一层输出
model1_layers = llama_hidden_states_sum  # 假设是 4 层，每层的 shape 是 (20, 1282, 2048)
model2_layers = gt_hidden_states_sum    # 假设是 4 层，每层的 shape 是 (20, 1282, 2048)

# 确保 t-SNE 结果保存目录存在
os.makedirs('tsne_results', exist_ok=True)

# 对每一层分别进行 t-SNE 降维并绘制
for layer_idx in range(4):  # 假设模型有 4 层
    # 获取第 layer_idx 层的输出
    model1_output = model1_layers[layer_idx]
    model2_output = model2_layers[layer_idx]

    # 将每层的输出 reshape 为 (20 * 1282, 2048)
    model1_reshaped = reshape_for_tsne(model1_output)
    model2_reshaped = reshape_for_tsne(model2_output)

    # 合并两个模型的输出
    combined_data = torch.cat((model1_reshaped, model2_reshaped), dim=0).cpu().numpy()  # 转为 numpy 数组

    # t-SNE 降维
    tsne_filename = f'tsne_results/layer_{layer_idx + 1}_tsne_result.npy'

    # 如果已有结果，直接加载
    if os.path.exists(tsne_filename):
        print(f"Loading precomputed t-SNE results for Layer {layer_idx + 1}...")
        tsne_result = np.load(tsne_filename)
    else:
        print(f"Start fitting t-SNE for Layer {layer_idx + 1}...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(combined_data)

        # 保存 t-SNE 结果以备下次加载
        np.save(tsne_filename, tsne_result)

    # 标记每个数据点属于哪个模型
    labels = ['Model 1'] * model1_reshaped.shape[0] + ['Model 2'] * model2_reshaped.shape[0]

    # 为不同层和模型使用不同的颜色
    color_map = {'Model 1': 'blue', 'Model 2': 'red'}

    # 创建一个新的图并绘制每一层的 t-SNE 结果
    plt.figure(figsize=(8, 6))

    # 绘制 t-SNE 结果
    for i, label in enumerate(np.unique(labels)):
        # 获取属于当前标签的数据点
        mask = np.array(labels) == label
        layer_tsne_result = tsne_result[mask]

        # 绘制对应标签的数据点，点小一些
        plt.scatter(layer_tsne_result[:, 0], layer_tsne_result[:, 1], 
                    color=color_map[label], label=f'{label} - Layer {layer_idx + 1}', alpha=0.01, s=1)

    # 添加标题和标签
    plt.title(f't-SNE Visualization of Model Outputs - Layer {layer_idx + 1}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='upper right')

    # 保存当前层的 t-SNE 图
    plt.savefig(f'tsne_results/layer_{layer_idx + 1}_tsne.png')
    plt.close()  # 关闭当前图，避免重叠

print("t-SNE visualization and saving complete!")