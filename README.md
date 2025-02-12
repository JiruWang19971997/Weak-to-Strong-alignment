# Weak-to-Strong-alignment
This repo tried to recompleted the experiment of RM from [openai weak-to-strong](https://arxiv.org/pdf/2312.09390) paper, and some futher experiments of MOE distillation to achieve better performance.

## 1. Reproduction of  REWARD MODELING
In this part, I reproduce the  PGR score mentioned in https://arxiv.org/pdf/2312.09390 on several generative model architectures.   
#### experiments
Dataset: mentioned in Training Data part.     
I only use 1/8 training dataset to do experiments on small-scale models, and they are split into train_1 and train_2, which can be [downloaded](https://www.alipan.com/t/2fO30HxrwOQ7HkHTvdgQ).   
```bash
train_teacher_model.sh //train 4 teacher models on train_1, which extract 4,6,8,10 layers from llama-3.2
train_student_model.sh //train 8 student models on train_2, which extract 4,6,8,10,12,14,15 layers from llama-3.2
weak_to_strong.sh // weak-to-strong generalization
python eval.py --model xxxx // calculate accuracy on train_2, for weak teacher, strong student ground truth, weak-to-strong student
```
all trained model weights could be [downloaded]()   

results:  
Small scale model:LLAMA-3.2:               Small scale model:Gemma-2-it:   
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/f7ddc15a-a29d-47ca-9bd7-87dcf8cdbb8b" width="30%" />
  <img src="https://github.com/user-attachments/assets/74f5be4e-3e9f-4368-8241-aeabedd2dacf" width="30%" />
</div>


- [ ] meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.1-8B-Instruct
- [ ] EleutherAI/gpt-neo-125m, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B
- [ ] GPT2-M, GPT2-LARGE, GPT-XL

The experiments on small-scale models are roughly consistent with results from [openai weak-to-strong](https://arxiv.org/pdf/2312.09390) paper.   
Form the experiments:
### ***Why do Stronger students not always perform better under the same weak teacher models?***
#### Bias
The size of the teacher model plays a crucial role, as a larger teacher model tends to achieve higher accuracy on train_set2 and provides a stronger supervisory signal. However, a better-trained teacher model does not always guarantee improved accuracy for the student model. In this weak-to-strong distillation process, no ground truth is considered—training solely with MSE loss between the two models may lead to the student inheriting biases from weak supervision labels, which is more likely appear in small scale teacher model. 
#### Inconsistency between reward model and language model
 If the teacher model is too small compared to the student, it may fail to unlock the student’s potential using MSE loss alone. Unlike NLP tasks, where token representations are learned during pretraining, reward model training requires transforming representations to maintain relative distances between positive and negative pairs (similar to contrastive learning)—a challenge that MSE loss alone cannot effectively address.

## Bias： MOE & contrastive learning
***Contribution 1: Self-Correction by contrastive learning***

***Contribution 2: multiple experts to determine the positive and negative sample for contrastive learning***

![截屏2025-02-12 19 58 23](https://github.com/user-attachments/assets/72173ba2-ee94-4ffd-99a5-2ab3ca5a41f9)





## Inconsistency： Feature Alignment
### Experiments: Are all hidden layers changed during the preference alignment task? 
I conducted experiments on pre-trained llama3.2 and well-fine-tuned llama3.2 on Reward Model tasks, by plotting their TSNE on lower dimensions.
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/11dc820c-6c14-44fa-af21-43f088310a09" width="20%" />
  <img src="https://github.com/user-attachments/assets/872a8b5c-7823-4e14-b4de-85d21d615f4b" width="20%" />
  <img src="https://github.com/user-attachments/assets/50eeb054-f583-4323-ac0e-596df9f6ae97" width="20%" />
  <img src="https://github.com/user-attachments/assets/61499806-7376-4f93-96ea-36a0c1f07a6d" width="20%" />
</div>   
From the figure, we can see that during the fine-tuning process on the reward model task, the deeper layers undergo more transformation compared to the shallow layers.


#### ***Contribution 1: Retaining the full weak teacher architecture and only fine-tuning the deep layer:***
The experiments show that keeping the full architecture of the weak teacher model is crucial. In the weak-to-strong framework, even when the teacher model is not fully trained, it is still possible to achieve a significant improvement in the student model's performance by only updating the parameters of the deep layers in the student model with minimal computational cost. This approach allows for a recovery of higher student capabilities.   

Llama3.2B   

![截屏2025-02-12 19 58 23](https://github.com/user-attachments/assets/35e1ff86-8c8a-4fc7-a4a0-15143940c353)

- [ ] Llama8B
- [ ] gemma


## Training Data
### Dataset Description
The traditional preference dataset format like [sardinelab/MT-pref](https://huggingface.co/datasets/sardinelab/MT-pref) or [trl-lib/Capybara-Preferences](https://huggingface.co/datasets/trl-lib/Capybara-Preferences) is:
```python
prompt responce_a responce_b score_a score_b
```

In order to fit generative language models, the dataset is reformatted followed the approach from https://github.com/RLHFlow/RLHF-Reward-Modeling.git :

```python
prompt+responce_a prompt+responce_b score_a score_b
```

In this repo, two datasets are used: ltraFeedback-preference-standard & hendrydong/preference_700K.
### Dataset preprocessing
In order to seed the outcome, we pre-prepare the training data, which prevents data from being repeatedly downloaded or encountering errors during multi-process training, while also saving training time and disk memory.
For prepossing hendrydong/preference_700 dataset, roughly need 15G memory to store the prepossed data.
```python
cd data
python preprocess.py --train_path "hendrydong/preference_700K" --output_dir "./processed_data" --tokenizer_name "meta-llama/Llama-3.2-1B-Instruct"
// show the data
python preprocess.py --train_path "hendrydong/preference_700K" --output_dir "./processed_data" --tokenizer_name "meta-llama/Llama-3.2-1B-Instruct" --show True
```

## Model preparing
Three series of large language models(LLM) were selected to illustrate the modify of MOE distilation of weak-to-strong approach.
```python
EleutherAI/gpt-neo-125m, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B
meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.1-8B-Instruct
ministral/Ministral-3b-instruct, ministral/Ministral-4b-instruct, mistralai/Mistral-7B-Instruct-v0.1
```
To download model:
```python
cd model
python download.py
```

- [ ] Dataset: RLHFlow/UltraFeedback-preference-standard
- [ ] Dataset: hendrydong/preference_700K

