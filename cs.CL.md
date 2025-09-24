# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Is Pre-training Truly Better Than Meta-Learning?.](http://arxiv.org/abs/2306.13841) | 在少样本学习中，当数据集的正式多样性较低时，预训练模型（PT）胜过模型无关元学习（MAML）。当正式多样性较高时，MAML更好。 |

# 详细

[^1]: 预训练真的比元学习更好吗？

    Is Pre-training Truly Better Than Meta-Learning?. (arXiv:2306.13841v1 [cs.LG])

    [http://arxiv.org/abs/2306.13841](http://arxiv.org/abs/2306.13841)

    在少样本学习中，当数据集的正式多样性较低时，预训练模型（PT）胜过模型无关元学习（MAML）。当正式多样性较高时，MAML更好。

    

    在少样本学习的背景下，目前普遍认为固定的预训练模型（PT）加上在评价时微调最后一层，胜过标准的元学习算法。我们通过深入的实证研究和广泛的数据集比较PT和模型无关元学习（MAML）这些说法。与以前的工作不同，我们强调使用相同的体系结构、相同的优化器，以及所有模型都训练到收敛。关键地，我们使用一个更严格的统计工具——效应量（Cohen's d）——来确定使用PT与使用MAML之间的模型差异的实际意义。然后使用一个预先提出的度量——多样性系数——来计算数据集的平均正式多样性。使用这种分析，我们证明了以下事实：1. 当数据集的正式多样性较低时，PT在平均意义上胜过MAML；2. 当正式多样性较高时，MAML胜过PT。

    In the context of few-shot learning, it is currently believed that a fixed pre-trained (PT) model, along with fine-tuning the final layer during evaluation, outperforms standard meta-learning algorithms. We re-evaluate these claims under an in-depth empirical examination of an extensive set of formally diverse datasets and compare PT to Model Agnostic Meta-Learning (MAML). Unlike previous work, we emphasize a fair comparison by using: the same architecture, the same optimizer, and all models trained to convergence. Crucially, we use a more rigorous statistical tool -- the effect size (Cohen's d) -- to determine the practical significance of the difference between a model trained with PT vs. a MAML. We then use a previously proposed metric -- the diversity coefficient -- to compute the average formal diversity of a dataset. Using this analysis, we demonstrate the following: 1. when the formal diversity of a data set is low, PT beats MAML on average and 2. when the formal diversity is hi
    

