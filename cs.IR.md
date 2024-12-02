# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Model, Analyze, and Comprehend User Interactions and Various Attributes within a Social Media Platform](https://arxiv.org/abs/2403.15937) | 本研究提出了一种新颖的基于图的方法，用于模型化和分析社交媒体平台内用户互动，揭示了活跃用户之间的连接模式、对社区动态的贡献率以及用户活动与受欢迎程度之间的相关性。 |
| [^2] | [Frequency and cardinality recovery from sketched data: a novel approach bridging Bayesian and frequentist views.](http://arxiv.org/abs/2309.15408) | 该论文研究了如何仅使用压缩表示来恢复大规模数据集中符号的频率，并引入了新的估计方法，将贝叶斯和频率论观点结合起来，提供了更好的解决方案。此外，还扩展了该方法以解决基数恢复问题。 |

# 详细

[^1]: 模型、分析和理解社交媒体平台内用户互动和各种属性

    Model, Analyze, and Comprehend User Interactions and Various Attributes within a Social Media Platform

    [https://arxiv.org/abs/2403.15937](https://arxiv.org/abs/2403.15937)

    本研究提出了一种新颖的基于图的方法，用于模型化和分析社交媒体平台内用户互动，揭示了活跃用户之间的连接模式、对社区动态的贡献率以及用户活动与受欢迎程度之间的相关性。

    

    本研究提出了一种基于帖子-评论关系的新颖基于图的方法，用于模型化和分析社交媒体平台内用户互动，构建用户互动图并分析以深入了解社区动态、用户行为和内容偏好。研究发现，社区内56.05%的活跃用户强连接在一起，但只有0.8%的用户对其动态有显著贡献。此外，我们观察到社区活动存在时间变化，某些时期经历了更高的参与度。此外，我们的发现强调了用户活动与受欢迎程度之间的相关性，表明更活跃的用户通常更受欢迎。

    arXiv:2403.15937v1 Announce Type: cross  Abstract: How can we effectively model, analyze, and comprehend user interactions and various attributes within a social media platform based on post-comment relationship? In this study, we propose a novel graph-based approach to model and analyze user interactions within a social media platform based on post-comment relationship. We construct a user interaction graph from social media data and analyze it to gain insights into community dynamics, user behavior, and content preferences. Our investigation reveals that while 56.05% of the active users are strongly connected within the community, only 0.8% of them significantly contribute to its dynamics. Moreover, we observe temporal variations in community activity, with certain periods experiencing heightened engagement. Additionally, our findings highlight a correlation between user activity and popularity showing that more active users are generally more popular. Alongside these, a preference f
    
[^2]: 从压缩数据中恢复频率和基数：一种将贝叶斯和频率论观点连接起来的新方法

    Frequency and cardinality recovery from sketched data: a novel approach bridging Bayesian and frequentist views. (arXiv:2309.15408v1 [stat.ME])

    [http://arxiv.org/abs/2309.15408](http://arxiv.org/abs/2309.15408)

    该论文研究了如何仅使用压缩表示来恢复大规模数据集中符号的频率，并引入了新的估计方法，将贝叶斯和频率论观点结合起来，提供了更好的解决方案。此外，还扩展了该方法以解决基数恢复问题。

    

    我们研究如何仅使用通过随机哈希获得的对数据进行压缩表示或草图来恢复大规模离散数据集中符号的频率。这是一个在计算机科学中的经典问题，有各种算法可用，如计数最小草图。然而，这些算法通常假设数据是固定的，处理随机采样数据时估计过于保守且可能不准确。在本文中，我们将草图数据视为未知分布的随机样本，然后引入改进现有方法的新估计器。我们的方法结合了贝叶斯非参数和经典（频率论）观点，解决了它们独特的限制，提供了一个有原则且实用的解决方案。此外，我们扩展了我们的方法以解决相关但不同的基数恢复问题，该问题涉及估计数据集中不同对象的总数。

    We study how to recover the frequency of a symbol in a large discrete data set, using only a compressed representation, or sketch, of those data obtained via random hashing. This is a classical problem in computer science, with various algorithms available, such as the count-min sketch. However, these algorithms often assume that the data are fixed, leading to overly conservative and potentially inaccurate estimates when dealing with randomly sampled data. In this paper, we consider the sketched data as a random sample from an unknown distribution, and then we introduce novel estimators that improve upon existing approaches. Our method combines Bayesian nonparametric and classical (frequentist) perspectives, addressing their unique limitations to provide a principled and practical solution. Additionally, we extend our method to address the related but distinct problem of cardinality recovery, which consists of estimating the total number of distinct objects in the data set. We validate
    

