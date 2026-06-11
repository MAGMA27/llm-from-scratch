# llm-from-scratch
本项目基于[stanford CS336](https://cs336.stanford.edu/)构建。

## pre training
从基础的pytroch功能出发，实现BPE、tokenizer、MHA等模块，以及training loop，DDP training loop，generating等功能，最终组装并使用[Tiny Stories](https://huggingface.co/datasets/roneneldan/TinyStories)数据集，训练了一个总参数量约为22.7M，非embedding参数约17.6M的Transformer Language Model。

用wandb监控训练参数，训练5w步，过程过程稳定，loss下降到2.8左右，perplexity在16.4，从生成结果以及其他人对该数据集的loss看，模型仍属于欠拟合阶段。
![Alt text](https://cdn4.winhlb.com/2026/05/22/6a100196b1444.png)

根据完成项目中的思考，额外梳理了一份[Transformer语言模型架构和细节梳理](Transformer语言模型架构和细节梳理.md)笔记，如有错误欢迎指出。

## fine tuning
基于Qwen 2.5 Math 1.5B模型开展SFT与GRPO微调实验，目标是提高模型在数学数据集上的表现。

### SFT
训练参数设置如下：
```python
config = {
        "n_sft_steps": 200,
        "learning_rate": 1e-5,
        "train_batch_size": 256,
        "gradient_accumulation_steps": 128,
    }
```
使用data中的sft数据集训练200步，根据梯度累积方法拆分总256的batch size，microbatch为2。step_loss如下图。

![Alt text](https://cdn4.winhlb.com/2026/05/26/6a1505afa2f44.png)

用5000条valid数据集评估SFT效果，结果如下表所示，准确率获得巨大提升。

| 模型     | 格式准确率 (%) | 答案准确率 (%) | 完全准确率 (%) |
| -------- | -------------- | -------------- | -------------- |
| baseline | 17.32          | 2.86           | 2.86           |
| sft      | 74.88          | 36.04          | 36.04          |

## GRPO
训练参数设置如下：
```python
config = {
        "n_grpo_steps": 200,
        "learning_rate": 1e-5,
        "advantage_eps": 1e-6,
        "rollout_batch_size": 256,
        "group_size": 8,
        "sampling_temperature": 1.0,
        "sampling_min_tokens": 4, # As in Expiter, disallow empty string responses
        "sampling_max_tokens": 1024,
        "epochs_per_rollout_batch": 1, # On-policy
        "train_batch_size": 256, # On-policy
        "gradient_accumulation_steps": 128, # microbatch size is 2, will fit on H100
        "gpu_memory_utilization": 0.8,
        "loss_type": "reinforce_with_baseline",
        "use_std_normalization": True,
    }
```

使用data中的sft数据集训练200步，根据梯度累积方法拆分总256的batch size，microbatch为2。step_loss如下图。由于这里的 loss 实际代表每组的 policy gradient 目标函数值，仅表示更新方向，因此显示出振荡情况
![Alt text](https://free.aiai.lat/2026/06/11/6a2a9e9792a31.png)

用5000条valid数据集评估GRPO效果，结果如下表所示，准确率获得的提升不如SFT明显，推测原因如下，跳过SFT直接进行GRPO，模型每轮输出正确格式，获得奖励的解空间太稀疏，导致正向样本太少，因此训练效果不如SFT。没有经过 SFT 的基座模型，既不懂“按照格式回答”，也不懂任务目标。它只能随机采样，从海量可能输出中“蒙”对一个既满足格式、又能拿奖励的答案。在 GRPO 的组内采样里，可能一整组都没有一个正向样本，奖励全是零，梯度信号几乎消失，模型自然难以进步。

| 模型     | 格式准确率 (%) | 答案准确率 (%) | 完全准确率 (%) |
| -------- | -------------- | -------------- | -------------- |
| baseline | 17.32          | 2.86           | 2.86           |
| grpo     | 25.06          | 13.66          | 13.66          |


同样梳理了一份[SFT、GRPO架构和细节梳理](SFT、GRPO架构和细节梳理.md)笔记，如有错误欢迎指出。