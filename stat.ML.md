# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On a Neural Implementation of Brenier's Polar Factorization](https://arxiv.org/abs/2403.03071) | 提出了Brenier的极分解定理的神经实现，探讨了在机器学习中的应用，并通过神经网络参数化潜在函数$u$，从最新神经最优输运领域的进展中汲取灵感。 |
| [^2] | [Dual-Directed Algorithm Design for Efficient Pure Exploration.](http://arxiv.org/abs/2310.19319) | 该论文研究了在有限备选方案集合中的纯探索问题。通过使用对偶变量，提出了一种新的算法设计原则，能够避免组合结构的复杂性，实现高效纯探索，从而准确回答查询问题。 |
| [^3] | [CLEVRER-Humans: Describing Physical and Causal Events the Human Way.](http://arxiv.org/abs/2310.03635) | CLEVRER-Humans是一个用于因果判断的视频推理数据集，通过人工标注来解决合成事件和合成语言描述的缺乏多样性问题，并通过迭代事件填空和神经语言生成模型提高数据收集效率。 |
| [^4] | [Model-based Clustering using Non-parametric Hidden Markov Models.](http://arxiv.org/abs/2309.12238) | 本文研究了使用非参数隐马尔可夫模型进行基于模型的聚类时的贝叶斯风险，并提出了相应的聚类方法。通过研究分类的贝叶斯风险和聚类的贝叶斯风险之间的关系，确定了聚类任务的难度。同时，在插值分类器和在线设置中的结果也得到了证明。模拟实验验证了这些发现。 |
| [^5] | [Graph topological property recovery with heat and wave dynamics-based features on graphsD.](http://arxiv.org/abs/2309.09924) | 本文提出了一种名为图微分方程网络（GDeNet）的方法，利用热和波动方程动力学特征来恢复图的拓扑属性，能够在各种下游任务中获得优秀的表现，同时在实际应用中也展现了较好的性能。 |
| [^6] | [Clustered Multi-Agent Linear Bandits.](http://arxiv.org/abs/2309.08710) | 本文研究了集群化的多智能体线性赌博机问题，提出了一种新颖的算法，通过智能体之间的协作来加速优化问题。通过理论分析和实证评估，证明了算法在遗憾最小化和聚类质量上的有效性。 |
| [^7] | [Learning under Selective Labels with Heterogeneous Decision-makers: An Instrumental Variable Approach.](http://arxiv.org/abs/2306.07566) | 本文提出了一种处理选择性标记数据的学习问题的方法。通过利用历史决策由一组异质决策者做出的事实，我们建立了一种有原理的工具变量框架，并提出了一种加权学习方法，用于学习预测规则。 |

# 详细

[^1]: 论Brenier的极分解的神经实现

    On a Neural Implementation of Brenier's Polar Factorization

    [https://arxiv.org/abs/2403.03071](https://arxiv.org/abs/2403.03071)

    提出了Brenier的极分解定理的神经实现，探讨了在机器学习中的应用，并通过神经网络参数化潜在函数$u$，从最新神经最优输运领域的进展中汲取灵感。

    

    在1991年，Brenier证明了一个定理，将$QR$分解（分为半正定矩阵$\times$酉矩阵）推广到任意矢量场$F:\mathbb{R}^d\rightarrow \mathbb{R}^d$。这个被称为极分解定理的定理表明，任意场$F$都可以表示为凸函数$u$的梯度与保测度映射$M$的复合，即$F=\nabla u \circ M$。我们提出了这一具有深远理论意义的结果的实际实现，并探讨了在机器学习中可能的应用。该定理与最优输运（OT）理论密切相关，我们借鉴了神经最优输运领域的最新进展，将潜在函数$u$参数化为输入凸神经网络。映射$M$可以通过使用$u^*$，即$u$的凸共轭，逐点计算得到，即$M=\nabla u^* \circ F$，或者作为辅助网络学习得到。因为$M$在基因

    arXiv:2403.03071v1 Announce Type: cross  Abstract: In 1991, Brenier proved a theorem that generalizes the $QR$ decomposition for square matrices -- factored as PSD $\times$ unitary -- to any vector field $F:\mathbb{R}^d\rightarrow \mathbb{R}^d$. The theorem, known as the polar factorization theorem, states that any field $F$ can be recovered as the composition of the gradient of a convex function $u$ with a measure-preserving map $M$, namely $F=\nabla u \circ M$. We propose a practical implementation of this far-reaching theoretical result, and explore possible uses within machine learning. The theorem is closely related to optimal transport (OT) theory, and we borrow from recent advances in the field of neural optimal transport to parameterize the potential $u$ as an input convex neural network. The map $M$ can be either evaluated pointwise using $u^*$, the convex conjugate of $u$, through the identity $M=\nabla u^* \circ F$, or learned as an auxiliary network. Because $M$ is, in gene
    
[^2]: 高效纯探索的双向算法设计

    Dual-Directed Algorithm Design for Efficient Pure Exploration. (arXiv:2310.19319v1 [stat.ML])

    [http://arxiv.org/abs/2310.19319](http://arxiv.org/abs/2310.19319)

    该论文研究了在有限备选方案集合中的纯探索问题。通过使用对偶变量，提出了一种新的算法设计原则，能够避免组合结构的复杂性，实现高效纯探索，从而准确回答查询问题。

    

    我们考虑在有限的备选方案集合中的随机顺序自适应实验的纯探索问题。决策者的目标是通过最小的测量工作以高置信度准确回答与备选方案相关的查询问题。一个典型的查询问题是确定表现最佳的备选方案，这在排名和选择问题以及机器学习文献中称为最佳臂识别问题。我们专注于固定精度的设定，并导出了一个与样本最优分配有强收敛性概念相关的优化条件的充分条件。使用对偶变量，我们刻画了一个分配是否最优的必要和充分条件。对偶变量的使用使我们能够绕过完全依赖于原始变量的最优条件的组合结构。值得注意的是，这些最优条件使得双向算法设计原则的扩展成为可能。

    We consider pure-exploration problems in the context of stochastic sequential adaptive experiments with a finite set of alternative options. The goal of the decision-maker is to accurately answer a query question regarding the alternatives with high confidence with minimal measurement efforts. A typical query question is to identify the alternative with the best performance, leading to ranking and selection problems, or best-arm identification in the machine learning literature. We focus on the fixed-precision setting and derive a sufficient condition for optimality in terms of a notion of strong convergence to the optimal allocation of samples. Using dual variables, we characterize the necessary and sufficient conditions for an allocation to be optimal. The use of dual variables allow us to bypass the combinatorial structure of the optimality conditions that relies solely on primal variables. Remarkably, these optimality conditions enable an extension of top-two algorithm design princ
    
[^3]: CLEVRER-Humans: 用人类的方式描述物理和因果事件

    CLEVRER-Humans: Describing Physical and Causal Events the Human Way. (arXiv:2310.03635v1 [cs.AI])

    [http://arxiv.org/abs/2310.03635](http://arxiv.org/abs/2310.03635)

    CLEVRER-Humans是一个用于因果判断的视频推理数据集，通过人工标注来解决合成事件和合成语言描述的缺乏多样性问题，并通过迭代事件填空和神经语言生成模型提高数据收集效率。

    

    构建能够推理物理事件及其因果关系的机器对于与物理世界进行灵活互动非常重要。然而，现有的大多数物理和因果推理基准都仅基于合成事件和合成自然语言描述的因果关系。这种设计存在两个问题：一是事件类型和自然语言描述缺乏多样性；二是基于手动定义的启发式规则的因果关系与人类判断不一致。为了解决这两个问题，我们提出了CLEVRER-Humans基准，这是一个用人工标注的视频推理数据集，用于对物理事件的因果判断。我们采用了两种技术来提高数据收集效率：首先，一种新颖的迭代事件填空任务，以 eliciting 视频中事件的新表示方式，我们称之为因果事件图 (CEGs)；其次，一种基于神经语言生成模型的数据增强技术。

    Building machines that can reason about physical events and their causal relationships is crucial for flexible interaction with the physical world. However, most existing physical and causal reasoning benchmarks are exclusively based on synthetically generated events and synthetic natural language descriptions of causal relationships. This design brings up two issues. First, there is a lack of diversity in both event types and natural language descriptions; second, causal relationships based on manually-defined heuristics are different from human judgments. To address both shortcomings, we present the CLEVRER-Humans benchmark, a video reasoning dataset for causal judgment of physical events with human labels. We employ two techniques to improve data collection efficiency: first, a novel iterative event cloze task to elicit a new representation of events in videos, which we term Causal Event Graphs (CEGs); second, a data augmentation technique based on neural language generative models.
    
[^4]: 使用非参数隐马尔可夫模型的基于模型的聚类

    Model-based Clustering using Non-parametric Hidden Markov Models. (arXiv:2309.12238v1 [math.ST])

    [http://arxiv.org/abs/2309.12238](http://arxiv.org/abs/2309.12238)

    本文研究了使用非参数隐马尔可夫模型进行基于模型的聚类时的贝叶斯风险，并提出了相应的聚类方法。通过研究分类的贝叶斯风险和聚类的贝叶斯风险之间的关系，确定了聚类任务的难度。同时，在插值分类器和在线设置中的结果也得到了证明。模拟实验验证了这些发现。

    

    非参数隐马尔可夫模型（HMM）由于其依赖结构，可以在不指定群组分布的情况下进行基于模型的聚类。本文研究了在使用HMM进行聚类时的贝叶斯风险，并提出了相应的聚类方法。首先，我们给出了将分类的贝叶斯风险与聚类的贝叶斯风险联系起来的结果，用以确定聚类任务的难度的关键数量。我们还在独立同分布的框架下证明了这一结果，这可能具有独立的兴趣。然后我们研究了插值分类器的过度风险。所有这些结果都被证明在在线设置中仍然有效，在该设置下，观测结果被顺序聚类。模拟实验证明了我们的发现。

    Thanks to their dependency structure, non-parametric Hidden Markov Models (HMMs) are able to handle model-based clustering without specifying group distributions. The aim of this work is to study the Bayes risk of clustering when using HMMs and to propose associated clustering procedures. We first give a result linking the Bayes risk of classification and the Bayes risk of clustering, which we use to identify the key quantity determining the difficulty of the clustering task. We also give a proof of this result in the i.i.d. framework, which might be of independent interest. Then we study the excess risk of the plugin classifier. All these results are shown to remain valid in the online setting where observations are clustered sequentially. Simulations illustrate our findings.
    
[^5]: 基于热和波动动力学特征的图拓扑属性恢复

    Graph topological property recovery with heat and wave dynamics-based features on graphsD. (arXiv:2309.09924v1 [cs.LG])

    [http://arxiv.org/abs/2309.09924](http://arxiv.org/abs/2309.09924)

    本文提出了一种名为图微分方程网络（GDeNet）的方法，利用热和波动方程动力学特征来恢复图的拓扑属性，能够在各种下游任务中获得优秀的表现，同时在实际应用中也展现了较好的性能。

    

    本文提出了一种名为图微分方程网络（GDeNet）的方法，利用图上的PDE解的表达能力，为各种下游任务获得连续的节点和图级表示。我们推导出了热和波动方程动力学与图的谱特性以及连续时间随机游走在图上行为之间的理论结果。我们通过恢复随机图生成参数、Ricci曲率和持久同调等方式实验证明了这些动力学能够捕捉到图形几何和拓扑的显著方面。此外，我们还展示了GDeNet在包括引用图、药物分子和蛋白质在内的真实世界数据集上的优越性能。

    In this paper, we propose Graph Differential Equation Network (GDeNet), an approach that harnesses the expressive power of solutions to PDEs on a graph to obtain continuous node- and graph-level representations for various downstream tasks. We derive theoretical results connecting the dynamics of heat and wave equations to the spectral properties of the graph and to the behavior of continuous-time random walks on graphs. We demonstrate experimentally that these dynamics are able to capture salient aspects of graph geometry and topology by recovering generating parameters of random graphs, Ricci curvature, and persistent homology. Furthermore, we demonstrate the superior performance of GDeNet on real-world datasets including citation graphs, drug-like molecules, and proteins.
    
[^6]: 集群化的多智能体线性赌博机

    Clustered Multi-Agent Linear Bandits. (arXiv:2309.08710v1 [cs.LG])

    [http://arxiv.org/abs/2309.08710](http://arxiv.org/abs/2309.08710)

    本文研究了集群化的多智能体线性赌博机问题，提出了一种新颖的算法，通过智能体之间的协作来加速优化问题。通过理论分析和实证评估，证明了算法在遗憾最小化和聚类质量上的有效性。

    

    本文针对多智能体线性随机赌博问题的一个特定实例，即集群化的多智能体线性赌博机进行了研究。在这个设置中，我们提出了一种新颖的算法，通过智能体之间的有效协作来加速整体优化问题。在这一贡献中，网络控制器负责估计网络的基本集群结构并优化同一组中智能体之间的经验分享。我们对遗憾最小化问题和聚类质量进行了理论分析。通过对合成数据和真实数据进行与最先进算法的实证评估，我们证明了我们方法的有效性：我们的算法显著改善了遗憾最小化，并成功恢复了真实的基本集群划分。

    We address in this paper a particular instance of the multi-agent linear stochastic bandit problem, called clustered multi-agent linear bandits. In this setting, we propose a novel algorithm leveraging an efficient collaboration between the agents in order to accelerate the overall optimization problem. In this contribution, a network controller is responsible for estimating the underlying cluster structure of the network and optimizing the experiences sharing among agents within the same groups. We provide a theoretical analysis for both the regret minimization problem and the clustering quality. Through empirical evaluation against state-of-the-art algorithms on both synthetic and real data, we demonstrate the effectiveness of our approach: our algorithm significantly improves regret minimization while managing to recover the true underlying cluster partitioning.
    
[^7]: 学习选择标签下的异质决策者：一种工具变量方法

    Learning under Selective Labels with Heterogeneous Decision-makers: An Instrumental Variable Approach. (arXiv:2306.07566v1 [stat.ML])

    [http://arxiv.org/abs/2306.07566](http://arxiv.org/abs/2306.07566)

    本文提出了一种处理选择性标记数据的学习问题的方法。通过利用历史决策由一组异质决策者做出的事实，我们建立了一种有原理的工具变量框架，并提出了一种加权学习方法，用于学习预测规则。

    

    我们研究了在选择性标记数据下的学习问题。这种问题在历史决策导致结果仅部分标记时出现。标记数据分布可能与整体人群有显著差异，特别是当历史决策和目标结果可以同时受某些未观察到的因素影响时。因此，仅基于标记数据进行学习可能会导致在整体人群中的严重偏差。我们的论文通过利用许多应用中历史决策由一组异质决策者做出的事实来解决此挑战。具体而言，我们在一个有原理的工具变量框架下分析了这种设置。我们建立了满足观察到的数据时任何给定预测规则的全体风险的点识别条件，并在点识别失败时提供了尖锐的风险界限。我们进一步提出了一种加权学习方法，用于学习预测规则。

    We study the problem of learning with selectively labeled data, which arises when outcomes are only partially labeled due to historical decision-making. The labeled data distribution may substantially differ from the full population, especially when the historical decisions and the target outcome can be simultaneously affected by some unobserved factors. Consequently, learning with only the labeled data may lead to severely biased results when deployed to the full population. Our paper tackles this challenge by exploiting the fact that in many applications the historical decisions were made by a set of heterogeneous decision-makers. In particular, we analyze this setup in a principled instrumental variable (IV) framework. We establish conditions for the full-population risk of any given prediction rule to be point-identified from the observed data and provide sharp risk bounds when the point identification fails. We further propose a weighted learning approach that learns prediction ru
    

