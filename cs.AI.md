# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Computing Universal Plans for Partially Observable Multi-Agent Path Finding.](http://arxiv.org/abs/2305.16203) | 本文提出了一种通用计划的解决方案，能够确保多个智能体之间不发生碰撞，并使用 ASP-MAUPF 系统进行实验，对其适用性和环境依赖度进行了观察和分析。 |
| [^2] | [InstructBio: A Large-scale Semi-supervised Learning Paradigm for Biochemical Problems.](http://arxiv.org/abs/2304.03906) | InstructBio是一种针对生物化学问题的大规模半监督学习算法，引入教练模型提供有效的置信度比率来指导目标模型对不同数据点给予明显关注，避免依赖有限的标记数据和不正确的伪注释，提高了分子模型的泛化能力。 |

# 详细

[^1]: 计算多智能体部分可观测路径规划问题的通用计划

    On Computing Universal Plans for Partially Observable Multi-Agent Path Finding. (arXiv:2305.16203v2 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2305.16203](http://arxiv.org/abs/2305.16203)

    本文提出了一种通用计划的解决方案，能够确保多个智能体之间不发生碰撞，并使用 ASP-MAUPF 系统进行实验，对其适用性和环境依赖度进行了观察和分析。

    

    多智能体路径规划问题在现今广泛应用于仓库机器人、物流自动化、交通控制等领域。本文将其看作是通用规划问题，提出了通用计划（又称策略）的解决方案，并实现了一个名为 ASP-MAUPF 的系统来计算它们。该系统能够在任意二维地图和智能体目标配置下，找到一个适用于每个智能体的通用计划，以确保它们之间互不干扰。

    Multi-agent routing problems have drawn significant attention nowadays due to their broad industrial applications in, e.g., warehouse robots, logistics automation, and traffic control. Conventionally, they are modelled as classical planning problems. In this paper, we argue that it is beneficial to formulate them as universal planning problems. We therefore propose universal plans, also known as policies, as the solution concepts, and implement a system called ASP-MAUPF (Answer Set Programming for Multi-Agent Universal Plan Finding) for computing them. Given an arbitrary two-dimensional map and a profile of goals for the agents, the system finds a feasible universal plan for each agent that ensures no collision with others. We use the system to conduct some experiments, and make some observations on the types of goal profiles and environments that will have feasible policies, and how they may depend on agents' sensors. We also demonstrate how users can customize action preferences to c
    
[^2]: InstructBio：一种针对生物化学问题的大规模半监督学习范式。

    InstructBio: A Large-scale Semi-supervised Learning Paradigm for Biochemical Problems. (arXiv:2304.03906v1 [cs.LG])

    [http://arxiv.org/abs/2304.03906](http://arxiv.org/abs/2304.03906)

    InstructBio是一种针对生物化学问题的大规模半监督学习算法，引入教练模型提供有效的置信度比率来指导目标模型对不同数据点给予明显关注，避免依赖有限的标记数据和不正确的伪注释，提高了分子模型的泛化能力。

    

    在科学人工智能领域，面对真实世界问题中的有限标记数据始终是一个重要的挑战。目前的方法是在大型未标记语料库上预训练强力的任务无关模型，但在向下游任务转移知识方面可能存在困难。在本研究中，我们提出了InstructBio，一种半监督学习算法，更好地利用未标记的样例。它引入教练模型来提供伪标签可靠性的置信度比率。这些置信度分数然后指导目标模型对不同的数据点给予明显的关注，避免对标记数据的过度依赖以及不正确的伪注释的负面影响。全面的实验表明，InstructBio显著提高了分子模型的泛化能力，不仅在分子属性预测方面，在活性悬崖估计方面也表现出优越性。

    In the field of artificial intelligence for science, it is consistently an essential challenge to face a limited amount of labeled data for real-world problems. The prevailing approach is to pretrain a powerful task-agnostic model on a large unlabeled corpus but may struggle to transfer knowledge to downstream tasks. In this study, we propose InstructMol, a semi-supervised learning algorithm, to take better advantage of unlabeled examples. It introduces an instructor model to provide the confidence ratios as the measurement of pseudo-labels' reliability. These confidence scores then guide the target model to pay distinct attention to different data points, avoiding the over-reliance on labeled data and the negative influence of incorrect pseudo-annotations. Comprehensive experiments show that InstructBio substantially improves the generalization ability of molecular models, in not only molecular property predictions but also activity cliff estimations, demonstrating the superiority of 
    

