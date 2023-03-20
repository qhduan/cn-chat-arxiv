# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stat-weight: Improving the Estimator of Interleaved Methods Outcomes with Statistical Hypothesis Testing.](http://arxiv.org/abs/2303.10094) | 本文提出了一种基于统计假设检验的交叉方法结果估计器 Stat-weight，针对长尾现实场景中的不平衡查询分布，重点考虑查询重复和排名函数获胜率对估计的影响，提供更准确和稳健的结果。 |
| [^2] | [STIXnet: A Novel and Modular Solution for Extracting All STIX Objects in CTI Reports.](http://arxiv.org/abs/2303.09999) | 提出了一种名为STIXnet的解决方案，可以自动提取CTI报告中所有的STIX实体和关系。 |
| [^3] | [Contrastive Self-supervised Learning in Recommender Systems: A Survey.](http://arxiv.org/abs/2303.09902) | 本综述论文对基于对比自监督学习的推荐方法进行了综合评估和分类，并提供了统一的框架来概述这些方法。对比自监督学习方法由于灵活性和性能优良而吸引了大量关注，并成为了自监督学习推荐方法的主导分支。 |
| [^4] | [High-Dimensional Approximate Nearest Neighbor Search: with Reliable and Efficient Distance Comparison Operations.](http://arxiv.org/abs/2303.09855) | 该论文提出了一种名为ADSampling的随机化算法的方法，通过他们的技术插件增强了现有的AKNN算法，以提高距离比较操作的效率，而几乎没有精度损失。 |

# 详细

[^1]: Stat-weight: 基于统计假设检验的交错方法结果估计改进

    Stat-weight: Improving the Estimator of Interleaved Methods Outcomes with Statistical Hypothesis Testing. (arXiv:2303.10094v1 [cs.IR])

    [http://arxiv.org/abs/2303.10094](http://arxiv.org/abs/2303.10094)

    本文提出了一种基于统计假设检验的交叉方法结果估计器 Stat-weight，针对长尾现实场景中的不平衡查询分布，重点考虑查询重复和排名函数获胜率对估计的影响，提供更准确和稳健的结果。

    

    交错是一种信息检索系统的在线评估方法，用于比较排名函数在解释用户隐式反馈方面的有效性。先前的工作评估了当时最有前途的交错方法在均匀查询分布上的表现。在现实世界中，通常存在一个不平衡的重复查询分布，它遵循着长尾用户搜索需求曲线。随着一个查询被不同的用户（或在不同的会话中）执行的次数增加，收集相关搜索结果的隐式反馈（交互/点击）的概率也越高。本文首先旨在在均匀查询分布上复制团队循环交叉检验准确度评估，然后集中评估该方法如何推广到长尾现实场景。我们提出了 Stat-weight，一种基于统计假设检验的交叉方法结果估计器，重点考虑查询重复和排名函数获胜率对估计的影响，提供更准确和稳健的结果。我们通过广泛的模拟和实际查询场景实验展示了 Stat-weight 相对于先前方法的优越性。

    Interleaving is an online evaluation approach for information retrieval systems that compares the effectiveness of ranking functions in interpreting the users' implicit feedback. Previous work such as Hofmann et al (2011) has evaluated the most promising interleaved methods at the time, on uniform distributions of queries. In the real world, ordinarily, there is an unbalanced distribution of repeated queries that follows a long-tailed users' search demand curve. The more a query is executed, by different users (or in different sessions), the higher the probability of collecting implicit feedback (interactions/clicks) on the related search results. This paper first aims to replicate the Team Draft Interleaving accuracy evaluation on uniform query distributions and then focuses on assessing how this method generalizes to long-tailed real-world scenarios. The reproducibility work raised interesting considerations on how the winning ranking function for each query should impact the overall
    
[^2]: STIXnet: 一种从CTI报告中提取所有STIX对象的新型模块化解决方案

    STIXnet: A Novel and Modular Solution for Extracting All STIX Objects in CTI Reports. (arXiv:2303.09999v1 [cs.IR])

    [http://arxiv.org/abs/2303.09999](http://arxiv.org/abs/2303.09999)

    提出了一种名为STIXnet的解决方案，可以自动提取CTI报告中所有的STIX实体和关系。

    

    从网络威胁情报(CTI)报告中自动提取信息对于风险管理至关重要。本文提出了一种名为STIXnet的解决方案，通过使用自然语言处理（NLP）技术和交互式实体知识库（KB），可以自动提取CTI报告中所有的STIX实体和关系。

    The automatic extraction of information from Cyber Threat Intelligence (CTI) reports is crucial in risk management. The increased frequency of the publications of these reports has led researchers to develop new systems for automatically recovering different types of entities and relations from textual data. Most state-of-the-art models leverage Natural Language Processing (NLP) techniques, which perform greatly in extracting a few types of entities at a time but cannot detect heterogeneous data or their relations. Furthermore, several paradigms, such as STIX, have become de facto standards in the CTI community and dictate a formal categorization of different entities and relations to enable organizations to share data consistently. This paper presents STIXnet, the first solution for the automated extraction of all STIX entities and relationships in CTI reports. Through the use of NLP techniques and an interactive Knowledge Base (KB) of entities, our approach obtains F1 scores comparab
    
[^3]: 推荐系统中对比自监督学习综述

    Contrastive Self-supervised Learning in Recommender Systems: A Survey. (arXiv:2303.09902v1 [cs.IR])

    [http://arxiv.org/abs/2303.09902](http://arxiv.org/abs/2303.09902)

    本综述论文对基于对比自监督学习的推荐方法进行了综合评估和分类，并提供了统一的框架来概述这些方法。对比自监督学习方法由于灵活性和性能优良而吸引了大量关注，并成为了自监督学习推荐方法的主导分支。

    

    近年来，基于深度学习的推荐系统取得了显著的成功。但是，这些方法通常严重依赖于有标签的数据（即用户-物品交互），遭受着数据稀疏和冷启动等问题。自监督学习是一种新兴的范式，它从未标记的数据中提取信息，为解决这些问题提供了见解。具体来说，对比自监督学习由于其灵活性和良好的性能，已经吸引了相当多的关注，并最近成为基于自监督学习的推荐方法中的主导分支。在本调查中，我们提供了当前基于对比自监督学习的推荐方法的最新和全面的评估。首先，我们提出一个统一的框架来概述这些方法。然后，我们根据框架的关键组成部分，包括视图生成策略、对比任务和对比目标，提出了一个分类法。对于每个组成部分，

    Deep learning-based recommender systems have achieved remarkable success in recent years. However, these methods usually heavily rely on labeled data (i.e., user-item interactions), suffering from problems such as data sparsity and cold-start. Self-supervised learning, an emerging paradigm that extracts information from unlabeled data, provides insights into addressing these problems. Specifically, contrastive self-supervised learning, due to its flexibility and promising performance, has attracted considerable interest and recently become a dominant branch in self-supervised learning-based recommendation methods. In this survey, we provide an up-to-date and comprehensive review of current contrastive self-supervised learning-based recommendation methods. Firstly, we propose a unified framework for these methods. We then introduce a taxonomy based on the key components of the framework, including view generation strategy, contrastive task, and contrastive objective. For each component,
    
[^4]: 高维近似最近邻搜索：带有可靠和高效距离比较操作。

    High-Dimensional Approximate Nearest Neighbor Search: with Reliable and Efficient Distance Comparison Operations. (arXiv:2303.09855v1 [cs.DS])

    [http://arxiv.org/abs/2303.09855](http://arxiv.org/abs/2303.09855)

    该论文提出了一种名为ADSampling的随机化算法的方法，通过他们的技术插件增强了现有的AKNN算法，以提高距离比较操作的效率，而几乎没有精度损失。

    

    近似K最近邻（AKNN）搜索是一个基础而且具有挑战性的问题。我们观察到，在高维空间中，几乎所有AKNN算法的时间消耗都被距离比较操作（DCO）所支配。对于每个操作，它会扫描物体的全维度，因此，它的运行时间与维度成线性关系。为了加速该操作，我们提出了一种随机化算法，称为ADSampling，对于大多数DCO，它的运行时间与维度成对数关系，并且高概率成功。此外，基于ADSampling，我们开发了一般性和两个特定算法的技术作为插件来增强现有的AKNN算法。理论和实证研究都证实了：（1）我们的技术几乎不会引入精度损失，（2）它们一致地提高了效率。

    Approximate K nearest neighbor (AKNN) search is a fundamental and challenging problem. We observe that in high-dimensional space, the time consumption of nearly all AKNN algorithms is dominated by that of the distance comparison operations (DCOs). For each operation, it scans full dimensions of an object and thus, runs in linear time wrt the dimensionality. To speed it up, we propose a randomized algorithm named ADSampling which runs in logarithmic time wrt to the dimensionality for the majority of DCOs and succeeds with high probability. In addition, based on ADSampling we develop one general and two algorithm-specific techniques as plugins to enhance existing AKNN algorithms. Both theoretical and empirical studies confirm that: (1) our techniques introduce nearly no accuracy loss and (2) they consistently improve the efficiency.
    

