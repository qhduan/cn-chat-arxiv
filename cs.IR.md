# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Algorithmic Collusion or Competition: the Role of Platforms' Recommender Systems.](http://arxiv.org/abs/2309.14548) | 这项研究填补了关于电子商务平台推荐算法在算法勾结研究中被忽视的空白，并发现推荐算法可以决定基于AI的定价算法的竞争或勾结动态。 |
| [^2] | [An Offline Metric for the Debiasedness of Click Models.](http://arxiv.org/abs/2304.09560) | 该论文介绍了一种离线评估点击模型去协变偏移的鲁棒性的方法，并提出了去偏差性这一概念和测量方法，这是恢复无偏一致相关性评分和点击模型对排名分布变化不变性的必要条件。 |

# 详细

[^1]: 算法勾结还是竞争：平台推荐系统的角色

    Algorithmic Collusion or Competition: the Role of Platforms' Recommender Systems. (arXiv:2309.14548v1 [cs.AI])

    [http://arxiv.org/abs/2309.14548](http://arxiv.org/abs/2309.14548)

    这项研究填补了关于电子商务平台推荐算法在算法勾结研究中被忽视的空白，并发现推荐算法可以决定基于AI的定价算法的竞争或勾结动态。

    

    最近的学术研究广泛探讨了基于人工智能(AI)的动态定价算法导致的算法勾结。然而，电子商务平台使用推荐算法来分配不同产品的曝光，而这一重要方面在先前的算法勾结研究中被大部分忽视。我们的研究填补了文献中这一重要的空白，并检验了推荐算法如何决定基于AI的定价算法的竞争或勾结动态。具体而言，我们研究了两种常用的推荐算法：(i)以最大化卖家总利润为目标的推荐系统和(ii)以最大化平台上产品需求为目标的推荐系统。我们构建了一个重复博弈框架，将卖家的定价算法和平台的推荐算法进行了整合。

    Recent academic research has extensively examined algorithmic collusion resulting from the utilization of artificial intelligence (AI)-based dynamic pricing algorithms. Nevertheless, e-commerce platforms employ recommendation algorithms to allocate exposure to various products, and this important aspect has been largely overlooked in previous studies on algorithmic collusion. Our study bridges this important gap in the literature and examines how recommendation algorithms can determine the competitive or collusive dynamics of AI-based pricing algorithms. Specifically, two commonly deployed recommendation algorithms are examined: (i) a recommender system that aims to maximize the sellers' total profit (profit-based recommender system) and (ii) a recommender system that aims to maximize the demand for products sold on the platform (demand-based recommender system). We construct a repeated game framework that incorporates both pricing algorithms adopted by sellers and the platform's recom
    
[^2]: 离线度量点击模型的去偏差性

    An Offline Metric for the Debiasedness of Click Models. (arXiv:2304.09560v1 [cs.IR])

    [http://arxiv.org/abs/2304.09560](http://arxiv.org/abs/2304.09560)

    该论文介绍了一种离线评估点击模型去协变偏移的鲁棒性的方法，并提出了去偏差性这一概念和测量方法，这是恢复无偏一致相关性评分和点击模型对排名分布变化不变性的必要条件。

    

    在学习用户点击时，固有偏见是数据中普遍存在的一个问题，例如位置偏见或信任偏见。点击模型是从用户点击中提取信息的常用方法，例如在Web搜索中提取文档相关性，或者估计点击偏差以用于下游应用，例如反事实的学习排序、广告位置和公平排序。最近的研究表明，社区中的当前评估实践不能保证性能良好的点击模型对于下游任务的泛化能力，其中排名分布与训练分布不同，即在协变偏移下。在这项工作中，我们提出了一个基于条件独立性测试的评估度量，以检测点击模型对协变偏移的缺乏鲁棒性。我们引入了去偏差性的概念和一种测量方法。我们证明，去偏差性是恢复无偏的一致相关性评分以及使点击模型对排名分布变化的不变性的必要条件。

    A well-known problem when learning from user clicks are inherent biases prevalent in the data, such as position or trust bias. Click models are a common method for extracting information from user clicks, such as document relevance in web search, or to estimate click biases for downstream applications such as counterfactual learning-to-rank, ad placement, or fair ranking. Recent work shows that the current evaluation practices in the community fail to guarantee that a well-performing click model generalizes well to downstream tasks in which the ranking distribution differs from the training distribution, i.e., under covariate shift. In this work, we propose an evaluation metric based on conditional independence testing to detect a lack of robustness to covariate shift in click models. We introduce the concept of debiasedness and a metric for measuring it. We prove that debiasedness is a necessary condition for recovering unbiased and consistent relevance scores and for the invariance o
    

