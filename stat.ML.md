# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization.](http://arxiv.org/abs/2310.07983) | RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。 |
| [^2] | [A General Framework for Learning under Corruption: Label Noise, Attribute Noise, and Beyond.](http://arxiv.org/abs/2307.08643) | 该研究提出了一个通用框架，在分布层面上对不同类型的数据污染模型进行了形式化分析，并通过分析贝叶斯风险的变化展示了这些污染对标准监督学习的影响。这些发现为进一步研究提供了新的方向和基础。 |

# 详细

[^1]: RandCom：去中心化随机通信跳跃方法用于分布式随机优化

    RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization. (arXiv:2310.07983v1 [cs.LG])

    [http://arxiv.org/abs/2310.07983](http://arxiv.org/abs/2310.07983)

    RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。

    

    具有随机通信跳过的分布式优化方法因其在加速通信复杂性方面具有的优势而受到越来越多的关注。然而，现有的研究主要集中在强凸确定性设置的集中式通信协议上。在本研究中，我们提出了一种名为RandCom的分布式优化方法，它采用了概率性的本地更新。我们分析了RandCom在随机非凸、凸和强凸设置中的性能，并证明了它能够通过通信概率来渐近地减少通信开销。此外，我们证明当节点数量增加时，RandCom能够实现线性加速。在随机强凸设置中，我们进一步证明了RandCom可以通过独立于网络的步长实现线性加速。此外，我们将RandCom应用于联邦学习，并提供了关于实现线性加速的潜力的积极结果。

    Distributed optimization methods with random communication skips are gaining increasing attention due to their proven benefits in accelerating communication complexity. Nevertheless, existing research mainly focuses on centralized communication protocols for strongly convex deterministic settings. In this work, we provide a decentralized optimization method called RandCom, which incorporates probabilistic local updates. We analyze the performance of RandCom in stochastic non-convex, convex, and strongly convex settings and demonstrate its ability to asymptotically reduce communication overhead by the probability of communication. Additionally, we prove that RandCom achieves linear speedup as the number of nodes increases. In stochastic strongly convex settings, we further prove that RandCom can achieve linear speedup with network-independent stepsizes. Moreover, we apply RandCom to federated learning and provide positive results concerning the potential for achieving linear speedup and
    
[^2]: 一个学习受到污染的通用框架：标签噪声、属性噪声等等

    A General Framework for Learning under Corruption: Label Noise, Attribute Noise, and Beyond. (arXiv:2307.08643v1 [cs.LG])

    [http://arxiv.org/abs/2307.08643](http://arxiv.org/abs/2307.08643)

    该研究提出了一个通用框架，在分布层面上对不同类型的数据污染模型进行了形式化分析，并通过分析贝叶斯风险的变化展示了这些污染对标准监督学习的影响。这些发现为进一步研究提供了新的方向和基础。

    

    数据中的污染现象很常见，并且已经在不同的污染模型下进行了广泛研究。尽管如此，对于这些模型之间的关系仍然了解有限，缺乏对污染及其对学习的影响的统一视角。在本研究中，我们通过基于马尔可夫核的一般性和详尽的框架，在分布层面上正式分析了污染模型。我们强调了标签和属性上存在的复杂联合和依赖性污染，这在现有研究中很少触及。此外，我们通过分析贝叶斯风险变化来展示这些污染如何影响标准的监督学习。我们的发现提供了对于“更复杂”污染对学习问题影响的定性洞察，并为未来的定量比较提供了基础。该框架的应用包括污染校正学习，其中包含一个子案例。

    Corruption is frequently observed in collected data and has been extensively studied in machine learning under different corruption models. Despite this, there remains a limited understanding of how these models relate such that a unified view of corruptions and their consequences on learning is still lacking. In this work, we formally analyze corruption models at the distribution level through a general, exhaustive framework based on Markov kernels. We highlight the existence of intricate joint and dependent corruptions on both labels and attributes, which are rarely touched by existing research. Further, we show how these corruptions affect standard supervised learning by analyzing the resulting changes in Bayes Risk. Our findings offer qualitative insights into the consequences of "more complex" corruptions on the learning problem, and provide a foundation for future quantitative comparisons. Applications of the framework include corruption-corrected learning, a subcase of which we 
    

