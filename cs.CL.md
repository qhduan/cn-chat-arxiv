# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scaling Laws for Downstream Task Performance of Large Language Models](https://arxiv.org/abs/2402.04177) | 本研究探讨了在转移学习环境中大型语言模型的尺度行为，发现微调数据集的大小和预训练数据与下游数据的分布一致性对下游性能有显著影响。 |

# 详细

[^1]: 大型语言模型的下游任务性能的尺度律

    Scaling Laws for Downstream Task Performance of Large Language Models

    [https://arxiv.org/abs/2402.04177](https://arxiv.org/abs/2402.04177)

    本研究探讨了在转移学习环境中大型语言模型的尺度行为，发现微调数据集的大小和预训练数据与下游数据的分布一致性对下游性能有显著影响。

    

    尺度律提供了重要的见解，可以指导大型语言模型（LLM）的设计。现有研究主要集中在研究预训练（上游）损失的尺度律。然而，在转移学习环境中，LLM先在无监督数据集上进行预训练，然后在下游任务上进行微调，我们通常也关心下游性能。在这项工作中，我们研究了在转移学习环境中的尺度行为，其中LLM被微调用于机器翻译任务。具体而言，我们研究了预训练数据的选择和大小对下游性能（翻译质量）的影响，使用了两个评价指标：下游交叉熵和BLEU分数。我们的实验证明，微调数据集的大小和预训练数据与下游数据的分布一致性显著影响尺度行为。在充分一致性情况下，下游交叉熵和BLEU分数都会逐渐提升。

    Scaling laws provide important insights that can guide the design of large language models (LLMs). Existing work has primarily focused on studying scaling laws for pretraining (upstream) loss. However, in transfer learning settings, in which LLMs are pretrained on an unsupervised dataset and then finetuned on a downstream task, we often also care about the downstream performance. In this work, we study the scaling behavior in a transfer learning setting, where LLMs are finetuned for machine translation tasks. Specifically, we investigate how the choice of the pretraining data and its size affect downstream performance (translation quality) as judged by two metrics: downstream cross-entropy and BLEU score. Our experiments indicate that the size of the finetuning dataset and the distribution alignment between the pretraining and downstream data significantly influence the scaling behavior. With sufficient alignment, both downstream cross-entropy and BLEU score improve monotonically with 
    

