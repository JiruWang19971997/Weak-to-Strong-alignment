import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, LlamaModel, PreTrainedModel

class CustomLlamaModel(nn.Module):
    def __init__(self, original_model, selected_layers):
        super().__init__()
        self.embeddings = original_model.model.embed_tokens  # 共享原始模型的embedding层
        self.layers = nn.ModuleList([original_model.model.layers[i] for i in selected_layers])
        self.norm = original_model.model.norm  # 共享原始模型的归一化层
        self.output_head = original_model.score  # 共享分类头（如果需要）

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
        hidden_states = self.norm(hidden_states)
        logits = self.output_head(hidden_states[:, 0, :])  # 取 [CLS] token 进行分类
        return logits
    def gradient_checkpointing_enable(self, **kwargs):  
        print("Warning: gradient_checkpointing_enable() is not implemented") 

class FeatureConnector(nn.Module):
    def __init__(self, t_dim, s_dim):
        super(FeatureConnector, self).__init__()
        self.proj = nn.Linear(s_dim, t_dim)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, x):
        return self.proj(x)

class MoETransformerDistiller(PreTrainedModel):
    def __init__(self, config, teacher_models, student_model, distill_layers=[3, 6, 9, 12]):
        super().__init__()
        self.teachers = [t.eval() for t in teacher_models]  # 冻结所有教师模型
        self.student = student_model
        self.distill_layers = distill_layers
        
        t_hidden_dim = self.teachers[0].config.hidden_size
        s_hidden_dim = self.student.config.hidden_size
        
        self.connectors = nn.ModuleList([FeatureConnector(t_hidden_dim, s_hidden_dim) for _ in distill_layers])
        self.gating_network = nn.Linear(s_hidden_dim, len(self.teachers))  # 门控网络
        
    def forward(self, input_ids, attention_mask):
        import pdb;pdb.set_trace()
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
        
        return s_outputs.logits, loss_distill
    
    
class DistillationModel(nn.Module):
    _keys_to_ignore_on_save = [] 
    def __init__(self, student_model, teacher_model, temperature=2.0, alpha=0.5):
        super(DistillationModel, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, inputs, attention, labels=None, return_loss=False):
        student_outputs = self.student_model(
            input_ids=inputs, attention_mask=attention
        )  
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                    input_ids=inputs, attention_mask=attention
                )

        return student_outputs, teacher_outputs
    def gradient_checkpointing_enable(self, **kwargs):  
        print("Warning: gradient_checkpointing_enable() is not implemented") 
