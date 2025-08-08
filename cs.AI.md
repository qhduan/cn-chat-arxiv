# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A](https://arxiv.org/abs/2402.13213) | 多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。 |
| [^2] | [Causal Perception.](http://arxiv.org/abs/2401.13408) | 这项研究提出了因果感知的概念，并将其应用于自动决策系统中。感知对决策的公平性有重要影响，因为公平性是与背景相关的，并且其解释取决于评判人是谁。 |
| [^3] | [Unsupervised Graph Deep Learning Reveals Emergent Flood Risk Profile of Urban Areas.](http://arxiv.org/abs/2309.14610) | 本研究基于无监督图深度学习模型提出了集成城市洪水风险评级模型，能够捕捉区域之间的空间依赖关系和洪水危险与城市要素之间的复杂相互作用，揭示了城市地区的突发洪水风险概况 |
| [^4] | [Unified Bayesian Frameworks for Multi-criteria Decision-making Problems.](http://arxiv.org/abs/2208.13390) | 本文引入了贝叶斯框架解决多准则决策问题，在团体决策问题和准则相关性方面提供了统计优雅的解决方案，适应了不同形式的决策者偏好，开发了识别决策者子群的混合模型，并设计了评估准则和备选方案相对重要性的概率排序方案。 |

# 详细

[^1]: 软最大概率（大部分时候）在多项选择问答任务中预测大型语言模型的正确性

    Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A

    [https://arxiv.org/abs/2402.13213](https://arxiv.org/abs/2402.13213)

    多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。

    

    尽管大型语言模型（LLMs）在许多任务上表现出色，但过度自信仍然是一个问题。我们假设在多项选择问答任务中，错误答案将与最大softmax概率（MSPs）较小相关，相比之下正确答案较大。我们在十个开源LLMs和五个数据集上全面评估了这一假设，在表现良好的原始问答任务中发现了对我们假设的强有力证据。对于表现最佳的六个LLMs，从MSP导出的AUROC在59/60个实例中都优于随机机会，p < 10^{-4}。在这六个LLMs中，平均AUROC范围在60%至69%之间。利用这些发现，我们提出了一个带有弃权选项的多项选择问答任务，并展示通过根据初始模型响应的MSP有选择地弃权可以提高性能。我们还用预softmax logits而不是softmax进行了相同的实验。

    arXiv:2402.13213v1 Announce Type: cross  Abstract: Although large language models (LLMs) perform impressively on many tasks, overconfidence remains a problem. We hypothesized that on multiple-choice Q&A tasks, wrong answers would be associated with smaller maximum softmax probabilities (MSPs) compared to correct answers. We comprehensively evaluate this hypothesis on ten open-source LLMs and five datasets, and find strong evidence for our hypothesis among models which perform well on the original Q&A task. For the six LLMs with the best Q&A performance, the AUROC derived from the MSP was better than random chance with p < 10^{-4} in 59/60 instances. Among those six LLMs, the average AUROC ranged from 60% to 69%. Leveraging these findings, we propose a multiple-choice Q&A task with an option to abstain and show that performance can be improved by selectively abstaining based on the MSP of the initial model response. We also run the same experiments with pre-softmax logits instead of sof
    
[^2]: 因果感知

    Causal Perception. (arXiv:2401.13408v1 [cs.AI])

    [http://arxiv.org/abs/2401.13408](http://arxiv.org/abs/2401.13408)

    这项研究提出了因果感知的概念，并将其应用于自动决策系统中。感知对决策的公平性有重要影响，因为公平性是与背景相关的，并且其解释取决于评判人是谁。

    

    当两个个体对相同的信息进行不同解读时，感知会发生。尽管这是一个已知现象，对决策中偏见有影响，但是感知在自动决策系统中仍然被忽视。感知对于ADM系统的公平性或公平使用具有重要影响，因为公平本身是与背景相关的，其解释取决于评判人是谁。本文将感知在因果推理中形式化，以捕捉个体的解释行为。我们还将个体经验形式化为额外的因果知识，个体会使用这些知识。此外，我们定义和讨论了易引发感知的属性，即易引发感知的属性。敏感属性，如性别和种族，就是易引发感知的明确示例。我们根据因果原则定义了两种感知，即不忠实感知和不一致感知。

    Perception occurs when two individuals interpret the same information differently. Despite being a known phenomenon with implications for bias in decision-making, as individuals' experience determines interpretation, perception remains largely overlooked in automated decision-making (ADM) systems. In particular, it can have considerable effects on the fairness or fair usage of an ADM system, as fairness itself is context-specific and its interpretation dependent on who is judging. In this work, we formalize perception under causal reasoning to capture the act of interpretation by an individual. We also formalize individual experience as additional causal knowledge that comes with and is used by an individual. Further, we define and discuss loaded attributes, which are attributes prone to evoke perception. Sensitive attributes, such as gender and race, are clear examples of loaded attributes. We define two kinds of causal perception, unfaithful and inconsistent, based on the causal prop
    
[^3]: 无监督的图深度学习揭示了城市地区突发洪水风险概况

    Unsupervised Graph Deep Learning Reveals Emergent Flood Risk Profile of Urban Areas. (arXiv:2309.14610v1 [cs.LG])

    [http://arxiv.org/abs/2309.14610](http://arxiv.org/abs/2309.14610)

    本研究基于无监督图深度学习模型提出了集成城市洪水风险评级模型，能够捕捉区域之间的空间依赖关系和洪水危险与城市要素之间的复杂相互作用，揭示了城市地区的突发洪水风险概况

    

    城市洪水风险源于与洪水危险、洪水暴露以及社会和物理脆弱性相关的多个要素之间的复杂和非线性相互作用，以及复杂的空间洪水依赖关系。然而，现有的用于表征城市洪水风险的方法主要是基于洪水平原地图，侧重于有限数量的要素，主要是危险和暴露要素，没有考虑要素之间的相互作用或空间区域之间的依赖关系。为了填补这一空白，本研究提出了一种基于新颖的无监督图深度学习模型（称为FloodRisk-Net）的集成城市洪水风险评级模型。FloodRisk-Net能够捕捉区域之间的空间依赖关系以及洪水危险和城市要素之间的复杂和非线性相互作用，从而确定突发洪水风险。利用美国多个都市统计区（MSAs）的数据，该模型将它们的洪水风险特征化为

    Urban flood risk emerges from complex and nonlinear interactions among multiple features related to flood hazard, flood exposure, and social and physical vulnerabilities, along with the complex spatial flood dependence relationships. Existing approaches for characterizing urban flood risk, however, are primarily based on flood plain maps, focusing on a limited number of features, primarily hazard and exposure features, without consideration of feature interactions or the dependence relationships among spatial areas. To address this gap, this study presents an integrated urban flood-risk rating model based on a novel unsupervised graph deep learning model (called FloodRisk-Net). FloodRisk-Net is capable of capturing spatial dependence among areas and complex and nonlinear interactions among flood hazards and urban features for specifying emergent flood risk. Using data from multiple metropolitan statistical areas (MSAs) in the United States, the model characterizes their flood risk into
    
[^4]: 统一贝叶斯框架用于多准则决策问题

    Unified Bayesian Frameworks for Multi-criteria Decision-making Problems. (arXiv:2208.13390v4 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2208.13390](http://arxiv.org/abs/2208.13390)

    本文引入了贝叶斯框架解决多准则决策问题，在团体决策问题和准则相关性方面提供了统计优雅的解决方案，适应了不同形式的决策者偏好，开发了识别决策者子群的混合模型，并设计了评估准则和备选方案相对重要性的概率排序方案。

    

    本文引入了贝叶斯框架来解决多准则决策问题的各个方面，利用概率解释多准则决策方法和挑战。通过利用贝叶斯模型的灵活性，提出的框架为多准则决策问题中的关键挑战，如团体决策问题和准则相关性，提供了统计优雅的解决方案。此外，这些模型可以适应决策者偏好中各种形式的不确定性，包括正态分布、三角分布和区间偏好。为了解决大规模团体多准则决策场景，开发了一个概率混合模型，可以识别出一致的决策者子群。此外，设计了一个概率排序方案，根据决策者偏好评估准则和备选方案的相对重要性。通过在各种数值示例上实验，验证了提出的框架，证明了它们的有效性。

    This paper introduces Bayesian frameworks for tackling various aspects of multi-criteria decision-making (MCDM) problems, leveraging a probabilistic interpretation of MCDM methods and challenges. By harnessing the flexibility of Bayesian models, the proposed frameworks offer statistically elegant solutions to key challenges in MCDM, such as group decision-making problems and criteria correlation. Additionally, these models can accommodate diverse forms of uncertainty in decision makers' (DMs) preferences, including normal and triangular distributions, as well as interval preferences. To address large-scale group MCDM scenarios, a probabilistic mixture model is developed, enabling the identification of homogeneous subgroups of DMs. Furthermore, a probabilistic ranking scheme is devised to assess the relative importance of criteria and alternatives based on DM(s) preferences. Through experimentation on various numerical examples, the proposed frameworks are validated, demonstrating their
    

