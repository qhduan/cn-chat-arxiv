# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dual-Granularity Medication Recommendation Based on Causal Inference](https://arxiv.org/abs/2403.00880) | 提出了DGMed框架，利用因果推断和创新的特征对齐方法进行双粒度药物推荐 |
| [^2] | [Retention Induced Biases in a Recommendation System with Heterogeneous Users](https://arxiv.org/abs/2402.13959) | 通过研究留存引发的偏见，发现改变推荐算法会导致推荐系统的行为在过渡期间与其新稳态不同，从而破坏了A/B实验作为评估RS改进的可靠性。 |
| [^3] | [Fair Ranking under Disparate Uncertainty](https://arxiv.org/abs/2309.01610) | 提出了一种新的公平排名标准Equal-Opportunity Ranking（EOR），将底层相关性模型的不确定性差异考虑在内，通过组内公平抽奖实现公平排名。 |
| [^4] | [ChatQA: Building GPT-4 Level Conversational QA Models.](http://arxiv.org/abs/2401.10225) | ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。 |

# 详细

[^1]: 基于因果推断的双粒度药物推荐

    Dual-Granularity Medication Recommendation Based on Causal Inference

    [https://arxiv.org/abs/2403.00880](https://arxiv.org/abs/2403.00880)

    提出了DGMed框架，利用因果推断和创新的特征对齐方法进行双粒度药物推荐

    

    随着医疗需求增长和机器学习技术的进步，基于人工智能的诊断和治疗系统备受关注。药物推荐旨在将患者的长期健康记录与医学知识整合，为特定疾病推荐准确和安全的药物组合。然而，大多数现有研究将药物推荐系统仅视为传统推荐系统的变体，忽视了药物和疾病之间的异质性。为解决这一挑战，我们提出了DGMed，一个用于药物推荐的框架。DGMed利用因果推断揭示医学实体之间的联系，并提出了一种创新的特征对齐方法来解决异质性问题。具体而言，该研究首先应用因果推断分析历史记录中药物对特定疾病的量化治疗效果，揭示...

    arXiv:2403.00880v1 Announce Type: cross  Abstract: As medical demands grow and machine learning technology advances, AI-based diagnostic and treatment systems are garnering increasing attention. Medication recommendation aims to integrate patients' long-term health records with medical knowledge, recommending accuracy and safe medication combinations for specific conditions. However, most existing researches treat medication recommendation systems merely as variants of traditional recommendation systems, overlooking the heterogeneity between medications and diseases. To address this challenge, we propose DGMed, a framework for medication recommendation. DGMed utilizes causal inference to uncover the connections among medical entities and presents an innovative feature alignment method to tackle heterogeneity issues. Specifically, this study first applies causal inference to analyze the quantified therapeutic effects of medications on specific diseases from historical records, uncoverin
    
[^2]: 具有异构用户的推荐系统中的留存引发偏见

    Retention Induced Biases in a Recommendation System with Heterogeneous Users

    [https://arxiv.org/abs/2402.13959](https://arxiv.org/abs/2402.13959)

    通过研究留存引发的偏见，发现改变推荐算法会导致推荐系统的行为在过渡期间与其新稳态不同，从而破坏了A/B实验作为评估RS改进的可靠性。

    

    我研究了一个具有用户流入和流失动态的推荐系统（RS）的概念模型。当流入和流失达到平衡时，用户分布达到稳定状态。改变推荐算法会改变稳定状态并产生过渡期。在这个期间，RS的行为与其新稳态不同。特别是，在过渡期内获得的A/B实验指标是RS长期性能的偏见指标。然而，学者和实践者经常在引入新算法后不久进行A/B测试以验证其有效性。然而，这种被广泛认为是评估RS改进的黄金标准的A/B实验范式可能产生错误结论。我还简要讨论了用户保留动态造成的数据偏见。

    arXiv:2402.13959v1 Announce Type: new  Abstract: I examine a conceptual model of a recommendation system (RS) with user inflow and churn dynamics. When inflow and churn balance out, the user distribution reaches a steady state. Changing the recommendation algorithm alters the steady state and creates a transition period. During this period, the RS behaves differently from its new steady state. In particular, A/B experiment metrics obtained in transition periods are biased indicators of the RS's long term performance. Scholars and practitioners, however, often conduct A/B tests shortly after introducing new algorithms to validate their effectiveness. This A/B experiment paradigm, widely regarded as the gold standard for assessing RS improvements, may consequently yield false conclusions. I also briefly discuss the data bias caused by the user retention dynamics.
    
[^3]: 不同不确定性下的公平排名

    Fair Ranking under Disparate Uncertainty

    [https://arxiv.org/abs/2309.01610](https://arxiv.org/abs/2309.01610)

    提出了一种新的公平排名标准Equal-Opportunity Ranking（EOR），将底层相关性模型的不确定性差异考虑在内，通过组内公平抽奖实现公平排名。

    

    排名是一种广泛使用的方法，用于将人类评估者的注意力集中在可管理的选项子集上。它作为人类决策过程的一部分的使用范围从在电子商务网站上展示潜在相关产品到为人工审查优先处理大学申请。虽然排名可以通过将关注集中在最有前途的选项上使人类评估更加高效，但我们认为，如果底层相关性模型的不确定性在不同组别的选项之间存在差异，排名可能会引入不公平。不幸的是，这种不确定性差异似乎普遍存在，常常对少数群体造成损害，因为这些群体的相关性估计可能由于缺乏数据或合适的特征而具有更高的不确定性。为了解决这个公平问题，我们提出了Equal-Opportunity Ranking（EOR）作为排名的新公平标准，并展示它对应于在相关选项之间进行组内公平抽奖

    arXiv:2309.01610v2 Announce Type: replace  Abstract: Ranking is a ubiquitous method for focusing the attention of human evaluators on a manageable subset of options. Its use as part of human decision-making processes ranges from surfacing potentially relevant products on an e-commerce site to prioritizing college applications for human review. While ranking can make human evaluation more effective by focusing attention on the most promising options, we argue that it can introduce unfairness if the uncertainty of the underlying relevance model differs between groups of options. Unfortunately, such disparity in uncertainty appears widespread, often to the detriment of minority groups for which relevance estimates can have higher uncertainty due to a lack of data or appropriate features. To address this fairness issue, we propose Equal-Opportunity Ranking (EOR) as a new fairness criterion for ranking and show that it corresponds to a group-wise fair lottery among the relevant options even
    
[^4]: ChatQA: 构建GPT-4级对话问答模型

    ChatQA: Building GPT-4 Level Conversational QA Models. (arXiv:2401.10225v1 [cs.CL])

    [http://arxiv.org/abs/2401.10225](http://arxiv.org/abs/2401.10225)

    ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。

    

    在这项工作中，我们介绍了ChatQA，一系列具有GPT-4级别准确性的对话问答模型。具体地，我们提出了一个两阶段的指令调整方法，可以显著提高大型语言模型（LLM）在零-shot对话问答中的结果。为了处理对话问答中的检索问题，我们在多轮问答数据集上进行了密集检索器的微调，这样可以提供与使用最先进的查询重写模型相当的结果，同时大大降低部署成本。值得注意的是，我们的ChatQA-70B可以在10个对话问答数据集的平均分上超过GPT-4（54.14 vs. 53.90），而不依赖于OpenAI GPT模型的任何合成数据。

    In this work, we introduce ChatQA, a family of conversational question answering (QA) models, that obtain GPT-4 level accuracies. Specifically, we propose a two-stage instruction tuning method that can significantly improve the zero-shot conversational QA results from large language models (LLMs). To handle retrieval in conversational QA, we fine-tune a dense retriever on a multi-turn QA dataset, which provides comparable results to using the state-of-the-art query rewriting model while largely reducing deployment cost. Notably, our ChatQA-70B can outperform GPT-4 in terms of average score on 10 conversational QA datasets (54.14 vs. 53.90), without relying on any synthetic data from OpenAI GPT models.
    

