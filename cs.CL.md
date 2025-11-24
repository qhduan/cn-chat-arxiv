# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) | 本文提出了一种将大型语言模型的知识蒸馏到更小模型的方法，通过使用反向KLD替换标准KD方法中的前向KLD目标，有效避免了学生模型高估教师分布的低概率区域。 |

# 详细

[^1]: MiniLLM：大型语言模型的知识蒸馏

    MiniLLM: Knowledge Distillation of Large Language Models

    [https://arxiv.org/abs/2306.08543](https://arxiv.org/abs/2306.08543)

    本文提出了一种将大型语言模型的知识蒸馏到更小模型的方法，通过使用反向KLD替换标准KD方法中的前向KLD目标，有效避免了学生模型高估教师分布的低概率区域。

    

    知识蒸馏（KD）是一种减少大型语言模型（LLMs）高计算需求的有前途的技术。然而，先前的KD方法主要应用于白盒分类模型或训练小模型来模仿如ChatGPT之类的黑盒模型API。如何有效地将白盒LLMs的知识蒸馏到小模型中仍未得到充分探讨，随着开源LLMs的蓬勃发展，这变得更为重要。在这项工作中，我们提出一种KD方法，将LLMs蒸馏到更小的语言模型。

    arXiv:2306.08543v2 Announce Type: replace-cross  Abstract: Knowledge Distillation (KD) is a promising technique for reducing the high computational demand of large language models (LLMs). However, previous KD methods are primarily applied to white-box classification models or training small models to imitate black-box model APIs like ChatGPT. How to effectively distill the knowledge of white-box LLMs into small models is still under-explored, which becomes more important with the prosperity of open-source LLMs. In this work, we propose a KD approach that distills LLMs into smaller language models. We first replace the forward Kullback-Leibler divergence (KLD) objective in the standard KD approaches with reverse KLD, which is more suitable for KD on generative language models, to prevent the student model from overestimating the low-probability regions of the teacher distribution. Then, we derive an effective optimization approach to learn this objective. The student models are named Mi
    

