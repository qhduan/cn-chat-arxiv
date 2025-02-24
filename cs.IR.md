# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CFaiRLLM: Consumer Fairness Evaluation in Large-Language Model Recommender System](https://arxiv.org/abs/2403.05668) | 这项研究引入了一个全面的评估框架 CFaiRLLM，旨在评估和减轻 RecLLMs 中消费者端的偏见 |
| [^2] | [GraphFM: Graph Factorization Machines for Feature Interaction Modeling](https://arxiv.org/abs/2105.11866) | 提出了一种名为GraphFM的图因子分解机方法，通过图结构自然表示特征，并将FM的交互功能集成到GNN的特征聚合策略中，能够模拟任意阶特征交互。 |
| [^3] | [Embracing Uncertainty: Adaptive Vague Preference Policy Learning for Multi-round Conversational Recommendation.](http://arxiv.org/abs/2306.04487) | 本文提出了一种称为“模糊偏好多轮会话推荐”（VPMCR）的新场景，考虑了用户在 CRS 中的模糊和波动的偏好。该方法采用软估计机制避免过滤过度，并通过自适应模糊偏好策略学习框架获得了实验上的良好推荐效果。 |

# 详细

[^1]: CFaiRLLM：大型语言模型推荐系统中的消费者公平评估

    CFaiRLLM: Consumer Fairness Evaluation in Large-Language Model Recommender System

    [https://arxiv.org/abs/2403.05668](https://arxiv.org/abs/2403.05668)

    这项研究引入了一个全面的评估框架 CFaiRLLM，旨在评估和减轻 RecLLMs 中消费者端的偏见

    

    在推荐系统不断发展的过程中，像ChatGPT这样的大型语言模型的整合标志着引入了基于语言模型的推荐（RecLLM）的新时代。虽然这些进展承诺提供前所未有的个性化和效率，但也引发了对公平性的重要关切，特别是在推荐可能无意中继续或放大与敏感用户属性相关的偏见的情况下。为了解决这些问题，我们的研究引入了一个全面的评估框架CFaiRLLM，旨在评估（从而减轻）RecLLMs中消费者端的偏见。

    arXiv:2403.05668v1 Announce Type: new  Abstract: In the evolving landscape of recommender systems, the integration of Large Language Models (LLMs) such as ChatGPT marks a new era, introducing the concept of Recommendation via LLM (RecLLM). While these advancements promise unprecedented personalization and efficiency, they also bring to the fore critical concerns regarding fairness, particularly in how recommendations might inadvertently perpetuate or amplify biases associated with sensitive user attributes. In order to address these concerns, our study introduces a comprehensive evaluation framework, CFaiRLLM, aimed at evaluating (and thereby mitigating) biases on the consumer side within RecLLMs.   Our research methodically assesses the fairness of RecLLMs by examining how recommendations might vary with the inclusion of sensitive attributes such as gender, age, and their intersections, through both similarity alignment and true preference alignment. By analyzing recommendations gener
    
[^2]: GraphFM：图因子分解机用于特征交互建模

    GraphFM: Graph Factorization Machines for Feature Interaction Modeling

    [https://arxiv.org/abs/2105.11866](https://arxiv.org/abs/2105.11866)

    提出了一种名为GraphFM的图因子分解机方法，通过图结构自然表示特征，并将FM的交互功能集成到GNN的特征聚合策略中，能够模拟任意阶特征交互。

    

    因子分解机（FM）是处理高维稀疏数据时建模成对（二阶）特征交互的一种常见方法。然而，一方面，FM未能捕捉到高阶特征交互，受到组合扩展的影响。另一方面，考虑每对特征之间的交互可能会引入噪声并降低预测准确性。为了解决这些问题，我们提出了一种新方法，称为Graph Factorization Machine（GraphFM），通过将特征自然表示成图结构。具体而言，我们设计了一种机制来选择有益的特征交互，并将其形式化为特征之间的边。然后，所提出的模型将FM的交互功能整合到图神经网络（GNN）的特征聚合策略中，通过堆叠层来模拟图结构特征上的任意阶特征交互。

    arXiv:2105.11866v4 Announce Type: replace-cross  Abstract: Factorization machine (FM) is a prevalent approach to modeling pairwise (second-order) feature interactions when dealing with high-dimensional sparse data. However, on the one hand, FM fails to capture higher-order feature interactions suffering from combinatorial expansion. On the other hand, taking into account interactions between every pair of features may introduce noise and degrade prediction accuracy. To solve the problems, we propose a novel approach, Graph Factorization Machine (GraphFM), by naturally representing features in the graph structure. In particular, we design a mechanism to select the beneficial feature interactions and formulate them as edges between features. Then the proposed model, which integrates the interaction function of FM into the feature aggregation strategy of Graph Neural Network (GNN), can model arbitrary-order feature interactions on the graph-structured features by stacking layers. Experime
    
[^3]: 接受不确定性：自适应模糊偏好策略学习用于多轮会话推荐

    Embracing Uncertainty: Adaptive Vague Preference Policy Learning for Multi-round Conversational Recommendation. (arXiv:2306.04487v1 [cs.IR])

    [http://arxiv.org/abs/2306.04487](http://arxiv.org/abs/2306.04487)

    本文提出了一种称为“模糊偏好多轮会话推荐”（VPMCR）的新场景，考虑了用户在 CRS 中的模糊和波动的偏好。该方法采用软估计机制避免过滤过度，并通过自适应模糊偏好策略学习框架获得了实验上的良好推荐效果。

    

    会话式推荐系统 (CRS) 通过多轮交互，动态引导用户表达偏好，有效地解决信息不对称问题。现有的 CRS 基本上假设用户有明确的偏好。在这种情况下，代理将完全信任用户反馈，并将接受或拒绝信号视为过滤项目和减少候选空间的强指标，这可能导致过滤过度的问题。然而，在现实中，用户的偏好往往是模糊和波动的，存在不确定性，他们在交互过程中的愿望和决策可能会发生变化。为了解决这个问题，我们引入了一个新颖的场景，称为“模糊偏好多轮会话推荐”（VPMCR），它考虑到用户在 CRS 中的模糊和波动的偏好。VPMCR 采用软估计机制为所有候选项目分配非零置信度分数，自然地避免了过滤过度的问题。在 VPMCR 设置中，我们提出了一种自适应模糊偏好策略学习框架，利用强化学习和偏好引导来学习 CRS 代理的最优策略。在两个真实数据集上的实验结果表明，相较于几种最先进的基准方法，我们提出的 VPMCR 方法具有更好的推荐效果。

    Conversational recommendation systems (CRS) effectively address information asymmetry by dynamically eliciting user preferences through multi-turn interactions. Existing CRS widely assumes that users have clear preferences. Under this assumption, the agent will completely trust the user feedback and treat the accepted or rejected signals as strong indicators to filter items and reduce the candidate space, which may lead to the problem of over-filtering. However, in reality, users' preferences are often vague and volatile, with uncertainty about their desires and changing decisions during interactions.  To address this issue, we introduce a novel scenario called Vague Preference Multi-round Conversational Recommendation (VPMCR), which considers users' vague and volatile preferences in CRS.VPMCR employs a soft estimation mechanism to assign a non-zero confidence score for all candidate items to be displayed, naturally avoiding the over-filtering problem. In the VPMCR setting, we introduc
    

