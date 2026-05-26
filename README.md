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
使用data中的sft数据集训练200步，根据梯度累积方法拆分总256的batch size，microbatch为2。step_loss如下图，稳定收敛。
![Alt text](https://cdn4.winhlb.com/2026/05/26/6a1505afa2f44.png)

## GRPO
【代码已完成，但计算资源受限，尚未整理完毕】


同样梳理了一份[SFT、GRPO架构和细节梳理](SFT、GRPO架构和细节梳理.md)笔记，如有错误欢迎指出。