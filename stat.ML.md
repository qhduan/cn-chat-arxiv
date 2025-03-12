# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving generalisation via anchor multivariate analysis](https://arxiv.org/abs/2403.01865) | 引入因果正则化扩展到锚回归（AR）中，提出了与锚框架相匹配的损失函数确保稳健性，各种多元分析算法均在锚框架内，简单正则化增强了OOD设置中的稳健性，验证了锚正则化的多功能性和对因果推断方法论的推进。 |
| [^2] | [Random Geometric Graph Alignment with Graph Neural Networks](https://arxiv.org/abs/2402.07340) | 本文研究了在图对齐问题中，通过图神经网络可以高概率恢复正确的顶点对齐。通过特定的特征稀疏性和噪声水平条件，我们证明了图神经网络的有效性，并与直接匹配方法进行了比较。 |
| [^3] | [The VampPrior Mixture Model](https://arxiv.org/abs/2402.04412) | 本论文提出了VampPrior混合模型（VMM），它是一种新颖的DLVM先验，可用于深度潜变量模型的集成和聚类，通过改善当前聚类先验的不足，并提出了一个清晰区分变分和先验参数的推理过程。使用VMM的变分自动编码器在基准数据集上取得了强大的聚类性能，将VMM与scVI相结合可以显著提高其性能，并自动将细胞分组为具有生物意义的聚类。 |
| [^4] | [Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization](https://arxiv.org/abs/2402.02746) | 标准 Gaussian 过程在高维贝叶斯优化中表现优秀，经验证据显示其在函数估计和协方差建模中克服了高维输入困难，比专门为高维优化设计的方法表现更好。 |
| [^5] | [Hypergraph Structure Inference From Data Under Smoothness Prior.](http://arxiv.org/abs/2308.14172) | 本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。 |
| [^6] | [SketchOGD: Memory-Efficient Continual Learning.](http://arxiv.org/abs/2305.16424) | SketchOGD提出了一种内存高效的解决灾难性遗忘的方法，通过采用在线草图算法，将模型梯度压缩为固定大小的矩阵，从而改进了现有的算法——正交梯度下降（OGD）。 |
| [^7] | [Statistical Guarantees of Group-Invariant GANs.](http://arxiv.org/abs/2305.13517) | 本研究提出了群不变GAN的统计保证，发现当学习群不变分布时，群不变GAN所需样本数会按群体大小的幂比例减少。 |
| [^8] | [Q-malizing flow and infinitesimal density ratio estimation.](http://arxiv.org/abs/2305.11857) | 研究提出了一种可以从一个数据分布P传输到任意访问通过有限样本的Q的流模型。这个模型通过神经ODE模型进行，可以进行无穷小DRE。 |
| [^9] | [Ledoit-Wolf linear shrinkage with unknown mean.](http://arxiv.org/abs/2304.07045) | 本文研究了在未知均值下的大维协方差矩阵估计问题，并提出了一种新的估计器，证明了其二次收敛性，在实验中表现优于其他标准估计器。 |
| [^10] | [Learning Hypergraphs From Signals With Dual Smoothness Prior.](http://arxiv.org/abs/2211.01717) | 本研究提出了一种基于双重平滑先验的超图结构学习框架，可从观察到的信号中学习超图结构以捕获实体间的内在高阶关系。 |

# 详细

[^1]: 通过锚多元分析改善泛化能力

    Improving generalisation via anchor multivariate analysis

    [https://arxiv.org/abs/2403.01865](https://arxiv.org/abs/2403.01865)

    引入因果正则化扩展到锚回归（AR）中，提出了与锚框架相匹配的损失函数确保稳健性，各种多元分析算法均在锚框架内，简单正则化增强了OOD设置中的稳健性，验证了锚正则化的多功能性和对因果推断方法论的推进。

    

    我们在锚回归（AR）中引入因果正则化扩展，以改善超出分布（OOD）的泛化能力。我们提出了与锚框架相匹配的损失函数，以确保对分布转移的稳健性。各种多元分析（MVA）算法，如（正交化）PLS、RRR和MLR，均在锚框架内。我们观察到简单的正则化增强了OOD设置中的稳健性。在合成和真实的气候科学问题中，为所选算法提供了估计器，展示了其一致性和有效性。经验验证突显了锚正则化的多功能性，强调其与MVA方法的兼容性，并强调其在增强可复制性的同时抵御分布转移中的作用。扩展的AR框架推进了因果推断方法论，解决了可靠OOD泛化的需求。

    arXiv:2403.01865v1 Announce Type: cross  Abstract: We introduce a causal regularisation extension to anchor regression (AR) for improved out-of-distribution (OOD) generalisation. We present anchor-compatible losses, aligning with the anchor framework to ensure robustness against distribution shifts. Various multivariate analysis (MVA) algorithms, such as (Orthonormalized) PLS, RRR, and MLR, fall within the anchor framework. We observe that simple regularisation enhances robustness in OOD settings. Estimators for selected algorithms are provided, showcasing consistency and efficacy in synthetic and real-world climate science problems. The empirical validation highlights the versatility of anchor regularisation, emphasizing its compatibility with MVA approaches and its role in enhancing replicability while guarding against distribution shifts. The extended AR framework advances causal inference methodologies, addressing the need for reliable OOD generalisation.
    
[^2]: 用图神经网络对随机几何图进行对齐

    Random Geometric Graph Alignment with Graph Neural Networks

    [https://arxiv.org/abs/2402.07340](https://arxiv.org/abs/2402.07340)

    本文研究了在图对齐问题中，通过图神经网络可以高概率恢复正确的顶点对齐。通过特定的特征稀疏性和噪声水平条件，我们证明了图神经网络的有效性，并与直接匹配方法进行了比较。

    

    我们研究了在顶点特征信息存在的情况下，图神经网络在图对齐问题中的性能。具体而言，给定两个独立扰动的单个随机几何图以及噪声稀疏特征的情况下，任务是恢复两个图的顶点之间的未知一对一映射关系。我们证明在特征向量的稀疏性和噪声水平满足一定条件的情况下，经过精心设计的单层图神经网络可以在很高的概率下通过图结构来恢复正确的顶点对齐。我们还证明了噪声水平的条件上界，仅存在对数因子差距。最后，我们将图神经网络的性能与直接在噪声顶点特征上求解分配问题进行了比较。我们证明了当噪声水平至少为常数时，这种直接匹配会导致恢复不完全，而图神经网络可以容忍n

    We characterize the performance of graph neural networks for graph alignment problems in the presence of vertex feature information. More specifically, given two graphs that are independent perturbations of a single random geometric graph with noisy sparse features, the task is to recover an unknown one-to-one mapping between the vertices of the two graphs. We show under certain conditions on the sparsity and noise level of the feature vectors, a carefully designed one-layer graph neural network can with high probability recover the correct alignment between the vertices with the help of the graph structure. We also prove that our conditions on the noise level are tight up to logarithmic factors. Finally we compare the performance of the graph neural network to directly solving an assignment problem on the noisy vertex features. We demonstrate that when the noise level is at least constant this direct matching fails to have perfect recovery while the graph neural network can tolerate n
    
[^3]: VampPrior混合模型

    The VampPrior Mixture Model

    [https://arxiv.org/abs/2402.04412](https://arxiv.org/abs/2402.04412)

    本论文提出了VampPrior混合模型（VMM），它是一种新颖的DLVM先验，可用于深度潜变量模型的集成和聚类，通过改善当前聚类先验的不足，并提出了一个清晰区分变分和先验参数的推理过程。使用VMM的变分自动编码器在基准数据集上取得了强大的聚类性能，将VMM与scVI相结合可以显著提高其性能，并自动将细胞分组为具有生物意义的聚类。

    

    当前用于深度潜变量模型（DLVMs）的聚类先验需要预先定义聚类的数量，并且容易受到较差的初始化的影响。解决这些问题可以通过同时执行集成和聚类的方式极大地改进基于深度学习的scRNA-seq分析。我们将VampPrior（Tomczak和Welling，2018）调整为Dirichlet过程高斯混合模型，得到VampPrior混合模型（VMM），这是一种新颖的DLVM先验。我们提出了一个推理过程，交替使用变分推理和经验贝叶斯，以清楚地区分变分和先验参数。在基准数据集上使用VMM的变分自动编码器获得了极具竞争力的聚类性能。将VMM与广受欢迎的scRNA-seq集成方法scVI（Lopez等，2018）相结合，显著改善了其性能，并自动将细胞分组为具有生物意义的聚类。

    Current clustering priors for deep latent variable models (DLVMs) require defining the number of clusters a-priori and are susceptible to poor initializations. Addressing these deficiencies could greatly benefit deep learning-based scRNA-seq analysis by performing integration and clustering simultaneously. We adapt the VampPrior (Tomczak & Welling, 2018) into a Dirichlet process Gaussian mixture model, resulting in the VampPrior Mixture Model (VMM), a novel prior for DLVMs. We propose an inference procedure that alternates between variational inference and Empirical Bayes to cleanly distinguish variational and prior parameters. Using the VMM in a Variational Autoencoder attains highly competitive clustering performance on benchmark datasets. Augmenting scVI (Lopez et al., 2018), a popular scRNA-seq integration method, with the VMM significantly improves its performance and automatically arranges cells into biologically meaningful clusters.
    
[^4]: 标准 Gaussian 过程在高维贝叶斯优化中足以应对

    Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization

    [https://arxiv.org/abs/2402.02746](https://arxiv.org/abs/2402.02746)

    标准 Gaussian 过程在高维贝叶斯优化中表现优秀，经验证据显示其在函数估计和协方差建模中克服了高维输入困难，比专门为高维优化设计的方法表现更好。

    

    长期以来，人们普遍认为使用标准 Gaussian 过程（GP）进行贝叶斯优化（BO），即标准 BO，在高维优化问题中效果不佳。这种观念可以部分归因于 Gaussian 过程在协方差建模和函数估计中对高维输入的困难。虽然这些担忧看起来合理，但缺乏支持这种观点的经验证据。本文系统地研究了在各种合成和真实世界基准问题上，使用标准 GP 回归进行高维优化的贝叶斯优化。令人惊讶的是，标准 GP 的表现始终位于最佳范围内，往往比专门为高维优化设计的现有 BO 方法表现更好。与刻板印象相反，我们发现标准 GP 可以作为学习高维目标函数的能力强大的代理。在没有强结构假设的情况下，使用标准 GP 进行 BO 可以获得非常好的性能。

    There has been a long-standing and widespread belief that Bayesian Optimization (BO) with standard Gaussian process (GP), referred to as standard BO, is ineffective in high-dimensional optimization problems. This perception may partly stem from the intuition that GPs struggle with high-dimensional inputs for covariance modeling and function estimation. While these concerns seem reasonable, empirical evidence supporting this belief is lacking. In this paper, we systematically investigated BO with standard GP regression across a variety of synthetic and real-world benchmark problems for high-dimensional optimization. Surprisingly, the performance with standard GP consistently ranks among the best, often outperforming existing BO methods specifically designed for high-dimensional optimization by a large margin. Contrary to the stereotype, we found that standard GP can serve as a capable surrogate for learning high-dimensional target functions. Without strong structural assumptions, BO wit
    
[^5]: 从数据中基于光滑性先验推断超图结构

    Hypergraph Structure Inference From Data Under Smoothness Prior. (arXiv:2308.14172v1 [cs.LG])

    [http://arxiv.org/abs/2308.14172](http://arxiv.org/abs/2308.14172)

    本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。

    

    超图在处理涉及多个实体的高阶关系数据中非常重要。在没有明确超图可用的情况下，希望能够从节点特征中推断出有意义的超图结构，以捕捉数据内在的关系。然而，现有的方法要么采用简单预定义的规则，不能精确捕捉潜在超图结构的分布，要么学习超图结构和节点特征之间的映射，但需要大量标记数据（即预先存在的超图结构）进行训练。这两种方法都局限于实际情景中的应用。为了填补这一空白，我们提出了一种新的光滑性先验，使我们能够设计一种方法，在没有标记数据作为监督的情况下推断出每个潜在超边的概率。所提出的先验表示超边中的节点特征与包含该超边的超边的特征高度相关。

    Hypergraphs are important for processing data with higher-order relationships involving more than two entities. In scenarios where explicit hypergraphs are not readily available, it is desirable to infer a meaningful hypergraph structure from the node features to capture the intrinsic relations within the data. However, existing methods either adopt simple pre-defined rules that fail to precisely capture the distribution of the potential hypergraph structure, or learn a mapping between hypergraph structures and node features but require a large amount of labelled data, i.e., pre-existing hypergraph structures, for training. Both restrict their applications in practical scenarios. To fill this gap, we propose a novel smoothness prior that enables us to design a method to infer the probability for each potential hyperedge without labelled data as supervision. The proposed prior indicates features of nodes in a hyperedge are highly correlated by the features of the hyperedge containing th
    
[^6]: SketchOGD：内存高效的持续学习

    SketchOGD: Memory-Efficient Continual Learning. (arXiv:2305.16424v1 [cs.LG])

    [http://arxiv.org/abs/2305.16424](http://arxiv.org/abs/2305.16424)

    SketchOGD提出了一种内存高效的解决灾难性遗忘的方法，通过采用在线草图算法，将模型梯度压缩为固定大小的矩阵，从而改进了现有的算法——正交梯度下降（OGD）。

    

    当机器学习模型在一系列任务上持续训练时，它们容易忘记先前任务上学习到的知识，这种现象称为灾难性遗忘。现有的解决灾难性遗忘的方法往往涉及存储过去任务的信息，这意味着内存使用是确定实用性的主要因素。本文提出了一种内存高效的解决灾难性遗忘的方法，改进了一种已有的算法——正交梯度下降（OGD）。OGD利用先前模型梯度来找到维持先前数据点性能的权重更新。然而，由于存储先前模型梯度的内存成本随算法运行时间增长而增加，因此OGD不适用于任意长时间跨度的连续学习。针对这个问题，本文提出了SketchOGD。SketchOGD采用在线草图算法，将模型梯度压缩为固定大小的矩阵。

    When machine learning models are trained continually on a sequence of tasks, they are liable to forget what they learned on previous tasks -- a phenomenon known as catastrophic forgetting. Proposed solutions to catastrophic forgetting tend to involve storing information about past tasks, meaning that memory usage is a chief consideration in determining their practicality. This paper proposes a memory-efficient solution to catastrophic forgetting, improving upon an established algorithm known as orthogonal gradient descent (OGD). OGD utilizes prior model gradients to find weight updates that preserve performance on prior datapoints. However, since the memory cost of storing prior model gradients grows with the runtime of the algorithm, OGD is ill-suited to continual learning over arbitrarily long time horizons. To address this problem, this paper proposes SketchOGD. SketchOGD employs an online sketching algorithm to compress model gradients as they are encountered into a matrix of a fix
    
[^7]: Group-Invariant GAN的统计保证

    Statistical Guarantees of Group-Invariant GANs. (arXiv:2305.13517v1 [stat.ML])

    [http://arxiv.org/abs/2305.13517](http://arxiv.org/abs/2305.13517)

    本研究提出了群不变GAN的统计保证，发现当学习群不变分布时，群不变GAN所需样本数会按群体大小的幂比例减少。

    

    Group-Invariant生成对抗网络(GAN)是一种GAN，其中生成器和判别器具有硬性集团对称性。实证研究表明，这些网络能够学习具有显着改进数据效率的集团不变分布。在本研究中，我们旨在通过分析群不变GAN的样本复杂度减少来严格量化这种改进。我们的研究发现，在学习群不变分布时，群不变GAN所需样本数按照群体大小的幂比例减少，这个幂取决于分布支持的本质维度。据我们所知，这项工作是首个为群不变生成模型，特别是GAN提供统计估计的工作，并可以为其他群不变生成模型的研究提供借鉴。

    Group-invariant generative adversarial networks (GANs) are a type of GANs in which the generators and discriminators are hardwired with group symmetries. Empirical studies have shown that these networks are capable of learning group-invariant distributions with significantly improved data efficiency. In this study, we aim to rigorously quantify this improvement by analyzing the reduction in sample complexity for group-invariant GANs. Our findings indicate that when learning group-invariant distributions, the number of samples required for group-invariant GANs decreases proportionally with a power of the group size, and this power depends on the intrinsic dimension of the distribution's support. To our knowledge, this work presents the first statistical estimation for group-invariant generative models, specifically for GANs, and it may shed light on the study of other group-invariant generative models.
    
[^8]: Q-malizing流和无穷小密度比估计

    Q-malizing flow and infinitesimal density ratio estimation. (arXiv:2305.11857v1 [stat.ML])

    [http://arxiv.org/abs/2305.11857](http://arxiv.org/abs/2305.11857)

    研究提出了一种可以从一个数据分布P传输到任意访问通过有限样本的Q的流模型。这个模型通过神经ODE模型进行，可以进行无穷小DRE。

    

    连续的正则化流在生成任务中被广泛使用，其中流网络从数据分布P传输到正态分布。一种能够从P传输到任意Q的流模型，其中P和Q都可通过有限样本访问，将在各种应用兴趣中使用，特别是在最近开发的望远镜密度比估计中（DRE），它需要构建中间密度以在P和Q之间建立桥梁。在这项工作中，我们提出了这样的“Q-malizing流”，通过神经ODE模型进行，该模型通过经验样本的可逆传输从P到Q（反之亦然），并通过最小化传输成本进行正则化。训练好的流模型使我们能够沿与时间参数化的log密度进行无穷小DRE，通过训练附加的连续时间流网络使用分类损失来估计log密度的时间偏导数。通过积分时间得分网络

    Continuous normalizing flows are widely used in generative tasks, where a flow network transports from a data distribution $P$ to a normal distribution. A flow model that can transport from $P$ to an arbitrary $Q$, where both $P$ and $Q$ are accessible via finite samples, would be of various application interests, particularly in the recently developed telescoping density ratio estimation (DRE) which calls for the construction of intermediate densities to bridge between $P$ and $Q$. In this work, we propose such a ``Q-malizing flow'' by a neural-ODE model which is trained to transport invertibly from $P$ to $Q$ (and vice versa) from empirical samples and is regularized by minimizing the transport cost. The trained flow model allows us to perform infinitesimal DRE along the time-parametrized $\log$-density by training an additional continuous-time flow network using classification loss, which estimates the time-partial derivative of the $\log$-density. Integrating the time-score network
    
[^9]: Ledoit-Wolf线性收缩方法在未知均值的情况下的应用(arXiv:2304.07045v1 [math.ST])

    Ledoit-Wolf linear shrinkage with unknown mean. (arXiv:2304.07045v1 [math.ST])

    [http://arxiv.org/abs/2304.07045](http://arxiv.org/abs/2304.07045)

    本文研究了在未知均值下的大维协方差矩阵估计问题，并提出了一种新的估计器，证明了其二次收敛性，在实验中表现优于其他标准估计器。

    

    本研究探讨了在未知均值下的大维协方差矩阵估计问题。当维数和样本数成比例并趋向于无穷大时，经验协方差估计器失效，此时称为Kolmogorov渐进性。当均值已知时，Ledoit和Wolf（2004）提出了一个线性收缩估计器，并证明了在这些演进下的收敛性。据我们所知，当均值未知时，尚未提出正式证明。为了解决这个问题，我们提出了一个新的估计器，并在Ledoit和Wolf的假设下证明了它的二次收敛性。最后，我们通过实验证明它胜过了其他标准估计器。

    This work addresses large dimensional covariance matrix estimation with unknown mean. The empirical covariance estimator fails when dimension and number of samples are proportional and tend to infinity, settings known as Kolmogorov asymptotics. When the mean is known, Ledoit and Wolf (2004) proposed a linear shrinkage estimator and proved its convergence under those asymptotics. To the best of our knowledge, no formal proof has been proposed when the mean is unknown. To address this issue, we propose a new estimator and prove its quadratic convergence under the Ledoit and Wolf assumptions. Finally, we show empirically that it outperforms other standard estimators.
    
[^10]: 基于双重平滑先验学习信号的超图结构

    Learning Hypergraphs From Signals With Dual Smoothness Prior. (arXiv:2211.01717v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.01717](http://arxiv.org/abs/2211.01717)

    本研究提出了一种基于双重平滑先验的超图结构学习框架，可从观察到的信号中学习超图结构以捕获实体间的内在高阶关系。

    

    超图结构学习是从观察到的信号中学习超图结构，以捕捉实体之间内在的高阶关系，当数据集中没有可用的超图拓扑结构时，这变得非常关键。本文提出了一种新的双重平滑先验的超图结构学习框架HGSL，通过把每个超边与具有节点信号平滑性和边连接性的子图对应起来，揭示了观察到的节点信号和超图结构之间的映射。实验结果表明了该方法的有效性。

    Hypergraph structure learning, which aims to learn the hypergraph structures from the observed signals to capture the intrinsic high-order relationships among the entities, becomes crucial when a hypergraph topology is not readily available in the datasets. There are two challenges that lie at the heart of this problem: 1) how to handle the huge search space of potential hyperedges, and 2) how to define meaningful criteria to measure the relationship between the signals observed on nodes and the hypergraph structure. In this paper, for the first challenge, we adopt the assumption that the ideal hypergraph structure can be derived from a learnable graph structure that captures the pairwise relations within signals. Further, we propose a hypergraph structure learning framework HGSL with a novel dual smoothness prior that reveals a mapping between the observed node signals and the hypergraph structure, whereby each hyperedge corresponds to a subgraph with both node signal smoothness and e
    

