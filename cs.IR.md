# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Investigating the Robustness of Sequential Recommender Systems Against Training Data Perturbations: an Empirical Study.](http://arxiv.org/abs/2307.13165) | 本研究通过对多个数据集进行评估发现，顺序推荐系统中删除序列末尾的项目显著降低了性能，而删除序列开头或中间的项目则没有明显影响。这一发现强调了考虑训练数据中扰动项目位置的重要性，并能指导更具鲁棒性的顺序推荐系统的设计。 |
| [^2] | [Embracing Uncertainty: Adaptive Vague Preference Policy Learning for Multi-round Conversational Recommendation.](http://arxiv.org/abs/2306.04487) | 本文提出了一种称为“模糊偏好多轮会话推荐”（VPMCR）的新场景，考虑了用户在 CRS 中的模糊和波动的偏好。该方法采用软估计机制避免过滤过度，并通过自适应模糊偏好策略学习框架获得了实验上的良好推荐效果。 |
| [^3] | [Editing Language Model-based Knowledge Graph Embeddings.](http://arxiv.org/abs/2301.10405) | 本文提出了一种新的任务——编辑基于语言模型的知识图谱嵌入，旨在实现对KG嵌入的数据高效和快速更新。针对这一任务，提出了一个简单而强大的方案——KGEditor，可以更好地更新特定事实而不影响其余部分的性能。 |

# 详细

[^1]: 研究顺序推荐系统对训练数据扰动的鲁棒性：一项经验研究

    Investigating the Robustness of Sequential Recommender Systems Against Training Data Perturbations: an Empirical Study. (arXiv:2307.13165v1 [cs.IR])

    [http://arxiv.org/abs/2307.13165](http://arxiv.org/abs/2307.13165)

    本研究通过对多个数据集进行评估发现，顺序推荐系统中删除序列末尾的项目显著降低了性能，而删除序列开头或中间的项目则没有明显影响。这一发现强调了考虑训练数据中扰动项目位置的重要性，并能指导更具鲁棒性的顺序推荐系统的设计。

    

    顺序推荐系统被广泛用于建模用户随时间变化的行为，然而其在面对训练数据扰动时的鲁棒性是一个关键问题。本文进行了一项经验研究，探究了在时间顺序序列中不同位置上删除项目的效果。我们评估了两种不同的顺序推荐系统模型在多个数据集上的表现，使用归一化折现累积增益（NDCG）指标和排名敏感度列表（Rank Sensitivity List）指标来衡量其性能。我们的结果显示，删除序列末尾的项目显著影响性能，NDCG下降高达60％，而删除序列开头或中间的项目没有显著影响。这些发现凸显了考虑训练数据中扰动项目位置的重要性，并可指导更具鲁棒性的顺序推荐系统的设计。

    Sequential Recommender Systems (SRSs) have been widely used to model user behavior over time, but their robustness in the face of perturbations to training data is a critical issue. In this paper, we conduct an empirical study to investigate the effects of removing items at different positions within a temporally ordered sequence. We evaluate two different SRS models on multiple datasets, measuring their performance using Normalized Discounted Cumulative Gain (NDCG) and Rank Sensitivity List metrics. Our results demonstrate that removing items at the end of the sequence significantly impacts performance, with NDCG decreasing up to 60\%, while removing items from the beginning or middle has no significant effect. These findings highlight the importance of considering the position of the perturbed items in the training data and shall inform the design of more robust SRSs.
    
[^2]: 接受不确定性：自适应模糊偏好策略学习用于多轮会话推荐

    Embracing Uncertainty: Adaptive Vague Preference Policy Learning for Multi-round Conversational Recommendation. (arXiv:2306.04487v1 [cs.IR])

    [http://arxiv.org/abs/2306.04487](http://arxiv.org/abs/2306.04487)

    本文提出了一种称为“模糊偏好多轮会话推荐”（VPMCR）的新场景，考虑了用户在 CRS 中的模糊和波动的偏好。该方法采用软估计机制避免过滤过度，并通过自适应模糊偏好策略学习框架获得了实验上的良好推荐效果。

    

    会话式推荐系统 (CRS) 通过多轮交互，动态引导用户表达偏好，有效地解决信息不对称问题。现有的 CRS 基本上假设用户有明确的偏好。在这种情况下，代理将完全信任用户反馈，并将接受或拒绝信号视为过滤项目和减少候选空间的强指标，这可能导致过滤过度的问题。然而，在现实中，用户的偏好往往是模糊和波动的，存在不确定性，他们在交互过程中的愿望和决策可能会发生变化。为了解决这个问题，我们引入了一个新颖的场景，称为“模糊偏好多轮会话推荐”（VPMCR），它考虑到用户在 CRS 中的模糊和波动的偏好。VPMCR 采用软估计机制为所有候选项目分配非零置信度分数，自然地避免了过滤过度的问题。在 VPMCR 设置中，我们提出了一种自适应模糊偏好策略学习框架，利用强化学习和偏好引导来学习 CRS 代理的最优策略。在两个真实数据集上的实验结果表明，相较于几种最先进的基准方法，我们提出的 VPMCR 方法具有更好的推荐效果。

    Conversational recommendation systems (CRS) effectively address information asymmetry by dynamically eliciting user preferences through multi-turn interactions. Existing CRS widely assumes that users have clear preferences. Under this assumption, the agent will completely trust the user feedback and treat the accepted or rejected signals as strong indicators to filter items and reduce the candidate space, which may lead to the problem of over-filtering. However, in reality, users' preferences are often vague and volatile, with uncertainty about their desires and changing decisions during interactions.  To address this issue, we introduce a novel scenario called Vague Preference Multi-round Conversational Recommendation (VPMCR), which considers users' vague and volatile preferences in CRS.VPMCR employs a soft estimation mechanism to assign a non-zero confidence score for all candidate items to be displayed, naturally avoiding the over-filtering problem. In the VPMCR setting, we introduc
    
[^3]: 基于语言模型的知识图谱嵌入编辑

    Editing Language Model-based Knowledge Graph Embeddings. (arXiv:2301.10405v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.10405](http://arxiv.org/abs/2301.10405)

    本文提出了一种新的任务——编辑基于语言模型的知识图谱嵌入，旨在实现对KG嵌入的数据高效和快速更新。针对这一任务，提出了一个简单而强大的方案——KGEditor，可以更好地更新特定事实而不影响其余部分的性能。

    

    近几十年来，使用语言模型进行知识图谱（KG）嵌入已经取得了实证成功。但是，基于语言模型的KG嵌入通常作为静态工件部署，修改起来具有挑战性，需要重新训练。为了解决这个问题，本文提出了一种新的任务，即编辑基于语言模型的KG嵌入。该任务旨在实现对KG嵌入的数据高效和快速更新，而不影响其余部分的性能。我们构建了四个新数据集：E-FB15k237、A-FB15k237、E-WN18RR 和 A-WN18RR，并评估了几种知识编辑基线，证明了之前的模型处理该任务的能力有限。我们进一步提出了一个简单但强大的基线——KGEditor，它利用超网络的附加参数层来编辑/添加事实。全面的实验结果表明，当更新特定事实而不影响其余部分的性能时，KGEditor 的表现更好。

    Recently decades have witnessed the empirical success of framing Knowledge Graph (KG) embeddings via language models. However, language model-based KG embeddings are usually deployed as static artifacts, which are challenging to modify without re-training after deployment. To address this issue, we propose a new task of editing language model-based KG embeddings in this paper. The proposed task aims to enable data-efficient and fast updates to KG embeddings without damaging the performance of the rest. We build four new datasets: E-FB15k237, A-FB15k237, E-WN18RR, and A-WN18RR, and evaluate several knowledge editing baselines demonstrating the limited ability of previous models to handle the proposed challenging task. We further propose a simple yet strong baseline dubbed KGEditor, which utilizes additional parametric layers of the hyper network to edit/add facts. Comprehensive experimental results demonstrate that KGEditor can perform better when updating specific facts while not affec
    

