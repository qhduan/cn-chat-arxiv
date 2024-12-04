# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Supervised Multiple Kernel Learning approaches for multi-omics data integration](https://arxiv.org/abs/2403.18355) | MKL方法提供了一种灵活有效的多组学数据集成方法，可以与复杂的监督式多组学整合方法竞争 |
| [^2] | [Reliable uncertainty with cheaper neural network ensembles: a case study in industrial parts classification](https://arxiv.org/abs/2403.10182) | 研究在工业零部件分类中探讨了利用更便宜的神经网络集成实现可靠的不确定性估计的方法 |
| [^3] | [Expressive Higher-Order Link Prediction through Hypergraph Symmetry Breaking](https://arxiv.org/abs/2402.11339) | 通过引入预处理算法识别展现对称性的正则子超图，从而提高超图在高阶链接预测中的表达能力和区分能力。 |
| [^4] | [Scalable Kernel Logistic Regression with Nystr\"om Approximation: Theoretical Analysis and Application to Discrete Choice Modelling](https://arxiv.org/abs/2402.06763) | 本文介绍了使用Nystr\"om近似方法解决大规模数据集上核逻辑回归的可扩展性问题。研究提供了理论分析并验证了不同的地标选择方法的性能。 |
| [^5] | [Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation](https://arxiv.org/abs/2402.03687) | PARD是一种将扩散模型与自回归方法相结合的置换不变性自回归扩散模型，通过使用图中的部分顺序以块逐块的自回归方式生成图。 |
| [^6] | [Optimal partitioning of directed acyclic graphs with dependent costs between clusters.](http://arxiv.org/abs/2308.03970) | 本论文提出了一种名为DCMAP的算法，用于对具有依赖成本的有向无环图进行最优分区。该算法通过优化基于DAG和集群映射的成本函数来寻找所有最优集群，并在途中返回接近最优解。实验证明在复杂系统的DBN模型中，该算法具有时间效率性。 |

# 详细

[^1]: 监督多核学习方法用于多组学数据集成

    Supervised Multiple Kernel Learning approaches for multi-omics data integration

    [https://arxiv.org/abs/2403.18355](https://arxiv.org/abs/2403.18355)

    MKL方法提供了一种灵活有效的多组学数据集成方法，可以与复杂的监督式多组学整合方法竞争

    

    高通量技术的进展导致越来越多的组学数据集的可用性。多种异质数据源的集成目前是生物学和生物信息学领域的一个问题。多核学习（MKL）已被证明是一种灵活和有效的方法，可以考虑多组学输入的多样性，尽管它在基因组数据挖掘中是一种不常用的工具。我们提供了基于不同核融合策略的新颖MKL方法。为了从输入核的元核中学习，我们将无监督集成算法调整为支持向量机的监督任务。我们还测试了用于核融合和分类的深度学习架构。结果显示，基于MKL的模型可以与更复杂、最先进的监督式多组学整合方法竞争。多核学习为多组学基因组数据中的预测模型提供了一个自然的框架。

    arXiv:2403.18355v1 Announce Type: cross  Abstract: Advances in high-throughput technologies have originated an ever-increasing availability of omics datasets. The integration of multiple heterogeneous data sources is currently an issue for biology and bioinformatics. Multiple kernel learning (MKL) has shown to be a flexible and valid approach to consider the diverse nature of multi-omics inputs, despite being an underused tool in genomic data mining.We provide novel MKL approaches based on different kernel fusion strategies.To learn from the meta-kernel of input kernels, we adaptedunsupervised integration algorithms for supervised tasks with support vector machines.We also tested deep learning architectures for kernel fusion and classification.The results show that MKL-based models can compete with more complex, state-of-the-art, supervised multi-omics integrative approaches. Multiple kernel learning offers a natural framework for predictive models in multi-omics genomic data. Our resu
    
[^2]: 用更便宜的神经网络集成实现可靠的不确定性：工业零部件分类案例研究

    Reliable uncertainty with cheaper neural network ensembles: a case study in industrial parts classification

    [https://arxiv.org/abs/2403.10182](https://arxiv.org/abs/2403.10182)

    研究在工业零部件分类中探讨了利用更便宜的神经网络集成实现可靠的不确定性估计的方法

    

    在运筹学(OR)中，预测模型经常会遇到数据分布与训练数据分布不同的场景。近年来，神经网络(NNs)在图像分类等领域的出色性能使其在OR中备受关注。然而，当面对OOD数据时，NNs往往会做出自信但不正确的预测。不确定性估计为自信的模型提供了一个解决方案，当输出应(不应)被信任时进行通信。因此，在OR领域中，NNs中的可靠不确定性量化至关重要。由多个独立NNs组成的深度集合已经成为一种有前景的方法，不仅提供强大的预测准确性，还能可靠地估计不确定性。然而，它们的部署由于较大的计算需求而具有挑战性。最近的基础研究提出了更高效的NN集成，即sna

    arXiv:2403.10182v1 Announce Type: new  Abstract: In operations research (OR), predictive models often encounter out-of-distribution (OOD) scenarios where the data distribution differs from the training data distribution. In recent years, neural networks (NNs) are gaining traction in OR for their exceptional performance in fields such as image classification. However, NNs tend to make confident yet incorrect predictions when confronted with OOD data. Uncertainty estimation offers a solution to overconfident models, communicating when the output should (not) be trusted. Hence, reliable uncertainty quantification in NNs is crucial in the OR domain. Deep ensembles, composed of multiple independent NNs, have emerged as a promising approach, offering not only strong predictive accuracy but also reliable uncertainty estimation. However, their deployment is challenging due to substantial computational demands. Recent fundamental research has proposed more efficient NN ensembles, namely the sna
    
[^3]: 通过超图对称性打破进行高阶链接预测

    Expressive Higher-Order Link Prediction through Hypergraph Symmetry Breaking

    [https://arxiv.org/abs/2402.11339](https://arxiv.org/abs/2402.11339)

    通过引入预处理算法识别展现对称性的正则子超图，从而提高超图在高阶链接预测中的表达能力和区分能力。

    

    一种超图由一组节点以及称为超边的节点子集合组成。更高阶链接预测是预测一个超图中是否存在缺失的超边的任务。为高阶链接预测学习的超边表示在同构下不失去区分能力时具有完全表达性。许多现有的超图表示学习器受到广义Weisfeiler Lehman-1（GWL-1）算法的表达能力限制，它是Weisfeiler Lehman-1算法的推广。然而，GWL-1的表达能力有限。事实上，具有相同GWL-1值节点的诱导子超图是无法区分的。此外，在超图上进行消息传递可能已经在GPU内存上变得计算昂贵。为了解决这些限制，我们设计了一种可以识别出展现对称性的特定正则子超图的预处理算法。

    arXiv:2402.11339v1 Announce Type: new  Abstract: A hypergraph consists of a set of nodes along with a collection of subsets of the nodes called hyperedges. Higher-order link prediction is the task of predicting the existence of a missing hyperedge in a hypergraph. A hyperedge representation learned for higher order link prediction is fully expressive when it does not lose distinguishing power up to an isomorphism. Many existing hypergraph representation learners, are bounded in expressive power by the Generalized Weisfeiler Lehman-1 (GWL-1) algorithm, a generalization of the Weisfeiler Lehman-1 algorithm. However, GWL-1 has limited expressive power. In fact, induced subhypergraphs with identical GWL-1 valued nodes are indistinguishable. Furthermore, message passing on hypergraphs can already be computationally expensive, especially on GPU memory. To address these limitations, we devise a preprocessing algorithm that can identify certain regular subhypergraphs exhibiting symmetry. Our p
    
[^4]: 使用Nystr\"om近似的可扩展核逻辑回归：理论分析和离散选择建模应用

    Scalable Kernel Logistic Regression with Nystr\"om Approximation: Theoretical Analysis and Application to Discrete Choice Modelling

    [https://arxiv.org/abs/2402.06763](https://arxiv.org/abs/2402.06763)

    本文介绍了使用Nystr\"om近似方法解决大规模数据集上核逻辑回归的可扩展性问题。研究提供了理论分析并验证了不同的地标选择方法的性能。

    

    将基于核的机器学习技术应用于使用大规模数据集的离散选择建模时，经常面临存储需求和模型中涉及的大量参数的挑战。这种复杂性影响了大规模模型的高效训练。本文通过引入Nystr\"om近似方法解决了可扩展性问题，用于大规模数据集上的核逻辑回归。研究首先进行了理论分析，其中：i) 对KLR解的集合进行了描述，ii) 给出了使用Nystr\"om近似的KLR解的上界，并最后描述了专门用于Nystr\"om KLR的优化算法的特化。之后，对Nystr\"om KLR进行了计算验证。测试了四种地标选择方法，包括基本均匀采样、k-means采样策略和基于杠杆得分的两种非均匀方法。这些策略的性能进行了评估。

    The application of kernel-based Machine Learning (ML) techniques to discrete choice modelling using large datasets often faces challenges due to memory requirements and the considerable number of parameters involved in these models. This complexity hampers the efficient training of large-scale models. This paper addresses these problems of scalability by introducing the Nystr\"om approximation for Kernel Logistic Regression (KLR) on large datasets. The study begins by presenting a theoretical analysis in which: i) the set of KLR solutions is characterised, ii) an upper bound to the solution of KLR with Nystr\"om approximation is provided, and finally iii) a specialisation of the optimisation algorithms to Nystr\"om KLR is described. After this, the Nystr\"om KLR is computationally validated. Four landmark selection methods are tested, including basic uniform sampling, a k-means sampling strategy, and two non-uniform methods grounded in leverage scores. The performance of these strategi
    
[^5]: Pard: 具有置换不变性的自回归扩散用于图生成

    Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation

    [https://arxiv.org/abs/2402.03687](https://arxiv.org/abs/2402.03687)

    PARD是一种将扩散模型与自回归方法相结合的置换不变性自回归扩散模型，通过使用图中的部分顺序以块逐块的自回归方式生成图。

    

    尽管自回归模型对于图的顺序敏感，但其简单有效，在图生成领域一直占据主导地位。然而，扩散模型因其置换不变性而越来越受关注。目前的图扩散模型一次性生成图，但需要额外的特征和成千上万步的去噪才能达到最佳性能。我们引入了PARD，一种将扩散模型与自回归方法相结合的置换不变性自回归扩散模型。PARD利用自回归模型的效果和效率，同时保持置换不变性，无需关注图的顺序敏感性。具体来说，我们发现与集合不同，图中的元素并不是完全无序的，节点和边有一个独特的部分顺序。利用这个部分顺序，PARD以块逐块的自回归方式生成图，其中每个块的概率为c。

    Graph generation has been dominated by autoregressive models due to their simplicity and effectiveness, despite their sensitivity to ordering. Yet diffusion models have garnered increasing attention, as they offer comparable performance while being permutation-invariant. Current graph diffusion models generate graphs in a one-shot fashion, but they require extra features and thousands of denoising steps to achieve optimal performance. We introduce PARD, a Permutation-invariant Auto Regressive Diffusion model that integrates diffusion models with autoregressive methods. PARD harnesses the effectiveness and efficiency of the autoregressive model while maintaining permutation invariance without ordering sensitivity. Specifically, we show that contrary to sets, elements in a graph are not entirely unordered and there is a unique partial order for nodes and edges. With this partial order, PARD generates a graph in a block-by-block, autoregressive fashion, where each block's probability is c
    
[^6]: 对具有依赖成本的有向无环图进行最优分区

    Optimal partitioning of directed acyclic graphs with dependent costs between clusters. (arXiv:2308.03970v1 [cs.DS])

    [http://arxiv.org/abs/2308.03970](http://arxiv.org/abs/2308.03970)

    本论文提出了一种名为DCMAP的算法，用于对具有依赖成本的有向无环图进行最优分区。该算法通过优化基于DAG和集群映射的成本函数来寻找所有最优集群，并在途中返回接近最优解。实验证明在复杂系统的DBN模型中，该算法具有时间效率性。

    

    许多统计推断场景，包括贝叶斯网络、马尔可夫过程和隐马尔可夫模型，可以通过将基础的有向无环图（DAG）划分成集群来支持。然而，在统计推断中，最优划分是具有挑战性的，因为要优化的成本取决于集群内的节点以及通过父节点和/或子节点连接的集群之间的映射，我们将其称为依赖集群。我们提出了一种名为DCMAP的新算法，用于具有依赖集群的最优集群映射。在基于DAG和集群映射的任意定义的正成本函数的基础上，我们证明DCMAP收敛于找到所有最优集群，并在途中返回接近最优解。通过实验证明，该算法对使用计算成本函数的一个海草复杂系统的DBN模型具有时间效率性。对于一个25个和50个节点的DBN，搜索空间大小分别为$9.91\times 10^9$和$1.5$

    Many statistical inference contexts, including Bayesian Networks (BNs), Markov processes and Hidden Markov Models (HMMS) could be supported by partitioning (i.e.~mapping) the underlying Directed Acyclic Graph (DAG) into clusters. However, optimal partitioning is challenging, especially in statistical inference as the cost to be optimised is dependent on both nodes within a cluster, and the mapping of clusters connected via parent and/or child nodes, which we call dependent clusters. We propose a novel algorithm called DCMAP for optimal cluster mapping with dependent clusters. Given an arbitrarily defined, positive cost function based on the DAG and cluster mappings, we show that DCMAP converges to find all optimal clusters, and returns near-optimal solutions along the way. Empirically, we find that the algorithm is time-efficient for a DBN model of a seagrass complex system using a computation cost function. For a 25 and 50-node DBN, the search space size was $9.91\times 10^9$ and $1.5
    

