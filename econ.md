# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Design of Cluster-Randomized Trials with Cross-Cluster Interference.](http://arxiv.org/abs/2310.18836) | 该论文提出了一种新的集群随机试验设计方法，考虑了交叉集群干扰的问题。通过排除可能受到干扰影响的单元，提出了新的估计器，并证明了这种方法可以大大减少偏差。这项研究还提供了优化估计器收敛速率的集群设计方法。 |

# 详细

[^1]: 设计具有交叉集群干扰的集群随机试验

    Design of Cluster-Randomized Trials with Cross-Cluster Interference. (arXiv:2310.18836v1 [stat.ME])

    [http://arxiv.org/abs/2310.18836](http://arxiv.org/abs/2310.18836)

    该论文提出了一种新的集群随机试验设计方法，考虑了交叉集群干扰的问题。通过排除可能受到干扰影响的单元，提出了新的估计器，并证明了这种方法可以大大减少偏差。这项研究还提供了优化估计器收敛速率的集群设计方法。

    

    集群随机试验经常涉及空间上分布不规律且没有明显分离社区的单元。在这种情况下，由于潜在的交叉集群干扰，集群构建是设计的一个关键方面。现有的文献依赖于部分干扰模型，该模型将集群视为给定，并假设没有交叉集群干扰。我们通过允许干扰与单元之间的地理距离衰减来放宽这个假设。这导致了一个偏差-方差的权衡：构建较少、较大的集群可以减少干扰引起的偏差，但会增加方差。我们提出了一种新的估计器，排除可能受到交叉集群干扰影响的单元，并显示相对于传统的均值差估计器，这大大降低了渐近偏差。然后，我们研究了优化估计器收敛速率的集群设计。我们提供了一个新的设计的正式证明，该设计选择了集群的数量。

    Cluster-randomized trials often involve units that are irregularly distributed in space without well-separated communities. In these settings, cluster construction is a critical aspect of the design due to the potential for cross-cluster interference. The existing literature relies on partial interference models, which take clusters as given and assume no cross-cluster interference. We relax this assumption by allowing interference to decay with geographic distance between units. This induces a bias-variance trade-off: constructing fewer, larger clusters reduces bias due to interference but increases variance. We propose new estimators that exclude units most potentially impacted by cross-cluster interference and show that this substantially reduces asymptotic bias relative to conventional difference-in-means estimators. We then study the design of clusters to optimize the estimators' rates of convergence. We provide formal justification for a new design that chooses the number of clus
    

