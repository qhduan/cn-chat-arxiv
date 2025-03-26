# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DistDNAS: Search Efficient Feature Interactions within 2 Hours.](http://arxiv.org/abs/2311.00231) | DistDNAS是一种在推荐系统中高效搜索特征交互的解决方案，通过分布式搜索和选择最佳交互模块，实现了巨大的加速并将搜索时间从2天缩短到2小时。 |
| [^2] | [Multi-Task Knowledge Enhancement for Zero-Shot and Multi-Domain Recommendation in an AI Assistant Application.](http://arxiv.org/abs/2306.06302) | 本文提出了一种利用多领域交互信息和外部知识图来进行新领域预测的方法，并将其应用于一个AI助手应用中，以提高推荐系统的预测准确性。 |

# 详细

[^1]: DistDNAS: 在2小时内高效搜索特征交互

    DistDNAS: Search Efficient Feature Interactions within 2 Hours. (arXiv:2311.00231v1 [cs.IR])

    [http://arxiv.org/abs/2311.00231](http://arxiv.org/abs/2311.00231)

    DistDNAS是一种在推荐系统中高效搜索特征交互的解决方案，通过分布式搜索和选择最佳交互模块，实现了巨大的加速并将搜索时间从2天缩短到2小时。

    

    在推荐系统中，搜索效率和服务效率是构建特征交互和加快模型开发过程的两个主要方面。在大规模基准测试中，由于大量数据上的顺序工作流程，搜索最佳特征交互设计需要付出巨大成本。此外，融合各种来源、顺序和数学运算的交互会引入潜在的冲突和额外的冗余，导致性能和服务成本的次优权衡。本文提出了DistDNAS作为一种简洁的解决方案，可以快速且高效地进行特征交互设计。DistDNAS提出了一个超级网络，将不同顺序和类型的交互模块作为搜索空间进行整合。为了优化搜索效率，DistDNAS在不同的数据日期上分布式搜索并汇总选择最佳的交互模块，实现了超过25倍的加速，将搜索成本从2天减少到2小时。

    Search efficiency and serving efficiency are two major axes in building feature interactions and expediting the model development process in recommender systems. On large-scale benchmarks, searching for the optimal feature interaction design requires extensive cost due to the sequential workflow on the large volume of data. In addition, fusing interactions of various sources, orders, and mathematical operations introduces potential conflicts and additional redundancy toward recommender models, leading to sub-optimal trade-offs in performance and serving cost. In this paper, we present DistDNAS as a neat solution to brew swift and efficient feature interaction design. DistDNAS proposes a supernet to incorporate interaction modules of varying orders and types as a search space. To optimize search efficiency, DistDNAS distributes the search and aggregates the choice of optimal interaction modules on varying data dates, achieving over 25x speed-up and reducing search cost from 2 days to 2 
    
[^2]: 多任务知识增强在AI助手应用中的零样本和多领域推荐中的应用

    Multi-Task Knowledge Enhancement for Zero-Shot and Multi-Domain Recommendation in an AI Assistant Application. (arXiv:2306.06302v1 [cs.IR])

    [http://arxiv.org/abs/2306.06302](http://arxiv.org/abs/2306.06302)

    本文提出了一种利用多领域交互信息和外部知识图来进行新领域预测的方法，并将其应用于一个AI助手应用中，以提高推荐系统的预测准确性。

    

    推荐系统在商业上取得了巨大的成功，但仍然难以将新用户整合进去。由于用户经常在不同领域与内容进行交互，因此可以利用用户在之前的领域中的交互来改善其在新领域中的推荐（多领域推荐）。知识图增强的单一领域推荐（知识图增强）的研究线程独立于此使用外部知识图来提高推荐系统的预测准确性。我们在这项工作中提出将这些方法统一起来：利用其他领域中的交互信息以及外部知识图来进行新领域的推荐。我们将这些想法应用于一个从数百万用户请求的视频、音乐和书籍的数据集中，该数据集用于一个AI助手应用中。

    Recommender systems have found significant commercial success but still struggle with integrating new users. Since users often interact with content in different domains, it is possible to leverage a user's interactions in previous domains to improve that user's recommendations in a new one (multi-domain recommendation). A separate research thread on knowledge graph enhancement uses external knowledge graphs to improve single domain recommendations (knowledge graph enhancement). Both research threads incorporate related information to improve predictions in a new domain. We propose in this work to unify these approaches: Using information from interactions in other domains as well as external knowledge graphs to make predictions in a new domain that would be impossible with either information source alone. We apply these ideas to a dataset derived from millions of users' requests for content across three domains (videos, music, and books) in a live virtual assistant application. We dem
    

