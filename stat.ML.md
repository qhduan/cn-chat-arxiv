# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provable Privacy with Non-Private Pre-Processing](https://arxiv.org/abs/2403.13041) | 提出了一个框架，能够评估非私密数据相关预处理算法引起的额外隐私成本，并利用平滑DP和预处理算法的有界敏感性建立整体隐私保证的上限 |
| [^2] | [Error bounds for particle gradient descent, and extensions of the log-Sobolev and Talagrand inequalities](https://arxiv.org/abs/2403.02004) | 证明了粒子梯度下降算法对于一般化的log-Sobolev和Polyak-Lojasiewicz不等式模型的收敛速度，以及推广了Bakry-Emery定理。 |
| [^3] | [VaR\ and CVaR Estimation in a Markov Cost Process: Lower and Upper Bounds.](http://arxiv.org/abs/2310.11389) | 本研究解决了在马尔可夫成本过程中估计无限时间折现成本的VaR和CVaR的问题。首先，我们推导出了估计误差的最小最大下界，并使用有限时间截断方案得出了上界。这是在马尔可夫设置中首次提供任何风险度量估计误差的下界和上界的工作。 |
| [^4] | [Grokking as the Transition from Lazy to Rich Training Dynamics.](http://arxiv.org/abs/2310.06110) | 研究发现洞察现象可能是由神经网络从懒惰训练动态过渡到丰富的特征学习模式的结果，通过跟踪足够的统计量，发现洞察是在网络首先尝试拟合核回归解决方案后，进行后期特征学习找到通用解决方案之后的结果。 |
| [^5] | [Neural Hilbert Ladders: Multi-Layer Neural Networks in Function Space.](http://arxiv.org/abs/2307.01177) | 本文提出了神经希尔伯特阶梯(NHL)的概念，它将多层神经网络描述为一系列的再生核希尔伯特空间，进一步推广了浅层神经网络的理论研究，并探讨了其在函数空间内的性质和应用。通过证明不同层次的NHL与多层NNs之间的对应关系，证明了学习NHL的泛化保证，并提出了NHL的特征动力学模型。最后，在ReLU和二次激活函数下展示了NHLs中的深度分离现象。 |
| [^6] | [Spectral clustering in the Gaussian mixture block model.](http://arxiv.org/abs/2305.00979) | 本文首次研究了从高维高斯混合块模型中抽样的图聚类和嵌入问题。 |
| [^7] | [Performance is not enough: a story of the Rashomon's quartet.](http://arxiv.org/abs/2302.13356) | 本文介绍了Rashomon的四重奏，这是一个合成数据集，其中来自不同类别的四个模型具有几乎相同的预测性能，同时其可视化揭示了极其不同的方法来理解数据中的相关性结构。 |

# 详细

[^1]: 具有非私密预处理的可证明隐私

    Provable Privacy with Non-Private Pre-Processing

    [https://arxiv.org/abs/2403.13041](https://arxiv.org/abs/2403.13041)

    提出了一个框架，能够评估非私密数据相关预处理算法引起的额外隐私成本，并利用平滑DP和预处理算法的有界敏感性建立整体隐私保证的上限

    

    当分析差分私密（DP）机器学习管道时，通常会忽略数据相关的预处理的潜在隐私成本。在这项工作中，我们提出了一个通用框架，用于评估由非私密数据相关预处理算法引起的额外隐私成本。我们的框架通过利用两个新的技术概念建立了整体隐私保证的上限：一种称为平滑DP的DP变体以及预处理算法的有界敏感性。

    arXiv:2403.13041v1 Announce Type: cross  Abstract: When analysing Differentially Private (DP) machine learning pipelines, the potential privacy cost of data-dependent pre-processing is frequently overlooked in privacy accounting. In this work, we propose a general framework to evaluate the additional privacy cost incurred by non-private data-dependent pre-processing algorithms. Our framework establishes upper bounds on the overall privacy guarantees by utilising two new technical notions: a variant of DP termed Smooth DP and the bounded sensitivity of the pre-processing algorithms. In addition to the generic framework, we provide explicit overall privacy guarantees for multiple data-dependent pre-processing algorithms, such as data imputation, quantization, deduplication and PCA, when used in combination with several DP algorithms. Notably, this framework is also simple to implement, allowing direct integration into existing DP pipelines.
    
[^2]: 粒子梯度下降的误差界限，以及log-Sobolev和Talagrand不等式的推广

    Error bounds for particle gradient descent, and extensions of the log-Sobolev and Talagrand inequalities

    [https://arxiv.org/abs/2403.02004](https://arxiv.org/abs/2403.02004)

    证明了粒子梯度下降算法对于一般化的log-Sobolev和Polyak-Lojasiewicz不等式模型的收敛速度，以及推广了Bakry-Emery定理。

    

    我们证明了粒子梯度下降(PGD)~(Kuntz等人，2023)的非渐近误差界限，这是一种最大似然估计的算法，用于离散化自由能梯度流获得的大型潜变量模型。我们首先展示了对于满足一般化log-Sobolev和Polyak-Lojasiewicz不等式（LSI和PLI）的模型，流以指数速度收敛到自由能的极小化集合。我们通过将最优输运文献中众所周知的结果（LSI意味着Talagrand不等式）及其在优化文献中的对应物（PLI意味着所谓的二次增长条件）扩展并应用到我们的新设置，来实现这一点。我们还推广了Bakry-Emery定理，并展示了对于具有强凹对数似然的模型，LSI/PLI的概括成立。

    arXiv:2403.02004v1 Announce Type: new  Abstract: We prove non-asymptotic error bounds for particle gradient descent (PGD)~(Kuntz et al., 2023), a recently introduced algorithm for maximum likelihood estimation of large latent variable models obtained by discretizing a gradient flow of the free energy. We begin by showing that, for models satisfying a condition generalizing both the log-Sobolev and the Polyak--{\L}ojasiewicz inequalities (LSI and P{\L}I, respectively), the flow converges exponentially fast to the set of minimizers of the free energy. We achieve this by extending a result well-known in the optimal transport literature (that the LSI implies the Talagrand inequality) and its counterpart in the optimization literature (that the P{\L}I implies the so-called quadratic growth condition), and applying it to our new setting. We also generalize the Bakry--\'Emery Theorem and show that the LSI/P{\L}I generalization holds for models with strongly concave log-likelihoods. For such m
    
[^3]: 在马尔可夫成本过程中的VaR和CVaR估计：下界和上界

    VaR\ and CVaR Estimation in a Markov Cost Process: Lower and Upper Bounds. (arXiv:2310.11389v1 [cs.LG])

    [http://arxiv.org/abs/2310.11389](http://arxiv.org/abs/2310.11389)

    本研究解决了在马尔可夫成本过程中估计无限时间折现成本的VaR和CVaR的问题。首先，我们推导出了估计误差的最小最大下界，并使用有限时间截断方案得出了上界。这是在马尔可夫设置中首次提供任何风险度量估计误差的下界和上界的工作。

    

    我们解决了在马尔可夫成本过程中估计无限时间折现成本的风险价值（Value-at-Risk，VaR）和条件风险价值（Conditional Value-at-Risk，CVaR）的问题。首先，我们推导出一个最小最大下界，该下界在期望意义和概率意义下都成立，其误差界为$\Omega(1/\sqrt{n})$。然后，利用有限时间截断方案，我们推导出CVaR估计误差的上界，该上界与我们的下界匹配，只有常数因子的差异。最后，我们讨论了我们的估计方案的扩展，涵盖了更通用的满足一定连续性准则的风险度量，例如谱风险度量和基于效用的缺口风险度量。据我们所知，我们的工作是第一个在马尔可夫设置中为任何风险度量提供估计误差的下界和上界的工作。我们指出，我们的下界也可扩展到无限时间折现成本的均值。即使在这种情况下，我们的结果$\Omega(1/\sqrt{n})$也优于现有结果$\Omega(

    We tackle the problem of estimating the Value-at-Risk (VaR) and the Conditional Value-at-Risk (CVaR) of the infinite-horizon discounted cost within a Markov cost process. First, we derive a minimax lower bound of $\Omega(1/\sqrt{n})$ that holds both in an expected and in a probabilistic sense. Then, using a finite-horizon truncation scheme, we derive an upper bound for the error in CVaR estimation, which matches our lower bound up to constant factors. Finally, we discuss an extension of our estimation scheme that covers more general risk measures satisfying a certain continuity criterion, e.g., spectral risk measures, utility-based shortfall risk. To the best of our knowledge, our work is the first to provide lower and upper bounds on the estimation error for any risk measure within Markovian settings. We remark that our lower bounds also extend to the infinite-horizon discounted costs' mean. Even in that case, our result $\Omega(1/\sqrt{n}) $ improves upon the existing result $\Omega(
    
[^4]: 从懒惰到丰富训练动态的洞察力

    Grokking as the Transition from Lazy to Rich Training Dynamics. (arXiv:2310.06110v1 [stat.ML])

    [http://arxiv.org/abs/2310.06110](http://arxiv.org/abs/2310.06110)

    研究发现洞察现象可能是由神经网络从懒惰训练动态过渡到丰富的特征学习模式的结果，通过跟踪足够的统计量，发现洞察是在网络首先尝试拟合核回归解决方案后，进行后期特征学习找到通用解决方案之后的结果。

    

    我们提出了洞察现象，即神经网络的训练损失在测试损失之前大幅下降，可能是由于神经网络从懒惰的训练动态转变为丰富的特征学习模式。为了说明这一机制，我们研究了在没有正则化的情况下，使用Vanilla梯度下降方法在多项式回归问题上进行的两层神经网络的训练，该训练展现了无法用现有理论解释的洞察现象。我们确定了该网络测试损失的足够统计量，并通过训练跟踪这些统计量揭示了洞察现象的发生。我们发现，在这种情况下，网络首先尝试使用初始特征拟合核回归解决方案，接着在训练损失已经很低的情况下进行后期特征学习，从而找到了一个能够泛化的解决方案。我们发现，洞察产生的关键因素是特征学习的速率，这可以通过缩放网络参数来精确控制。

    We propose that the grokking phenomenon, where the train loss of a neural network decreases much earlier than its test loss, can arise due to a neural network transitioning from lazy training dynamics to a rich, feature learning regime. To illustrate this mechanism, we study the simple setting of vanilla gradient descent on a polynomial regression problem with a two layer neural network which exhibits grokking without regularization in a way that cannot be explained by existing theories. We identify sufficient statistics for the test loss of such a network, and tracking these over training reveals that grokking arises in this setting when the network first attempts to fit a kernel regression solution with its initial features, followed by late-time feature learning where a generalizing solution is identified after train loss is already low. We find that the key determinants of grokking are the rate of feature learning -- which can be controlled precisely by parameters that scale the ne
    
[^5]: 神经希尔伯特阶梯：函数空间中的多层神经网络

    Neural Hilbert Ladders: Multi-Layer Neural Networks in Function Space. (arXiv:2307.01177v1 [cs.LG])

    [http://arxiv.org/abs/2307.01177](http://arxiv.org/abs/2307.01177)

    本文提出了神经希尔伯特阶梯(NHL)的概念，它将多层神经网络描述为一系列的再生核希尔伯特空间，进一步推广了浅层神经网络的理论研究，并探讨了其在函数空间内的性质和应用。通过证明不同层次的NHL与多层NNs之间的对应关系，证明了学习NHL的泛化保证，并提出了NHL的特征动力学模型。最后，在ReLU和二次激活函数下展示了NHLs中的深度分离现象。

    

    神经网络(NNs)所探索的函数空间的特征化是深度学习理论的重要方面。本文将具有任意宽度的多层NN视为定义特定层次的再生核希尔伯特空间(RKHS)的神经希尔伯特阶梯(NHL)。这使得我们能够定义一个函数空间和一个复杂度度量，该度量推广了浅层NNs的先前结果，并研究了它们在几个方面的理论特性和影响。首先，我们证明了L层NNs表示的函数与属于L层NHLs的函数之间的对应关系。其次，我们证明了学习具有受控复杂度度量的NHL的泛化保证。第三，对应于在无穷宽均场极限下训练多层NNs，我们导出了NHL的特征动力学，该动力学被描述为多个随机场的演化。第四，在ReLU和二次激活函数下展示了NHLs中的深度分离示例。

    The characterization of the functions spaces explored by neural networks (NNs) is an important aspect of deep learning theory. In this work, we view a multi-layer NN with arbitrary width as defining a particular hierarchy of reproducing kernel Hilbert spaces (RKHSs), named a Neural Hilbert Ladder (NHL). This allows us to define a function space and a complexity measure that generalize prior results for shallow NNs, and we then examine their theoretical properties and implications in several aspects. First, we prove a correspondence between functions expressed by L-layer NNs and those belonging to L-level NHLs. Second, we prove generalization guarantees for learning an NHL with the complexity measure controlled. Third, corresponding to the training of multi-layer NNs in the infinite-width mean-field limit, we derive an evolution of the NHL characterized as the dynamics of multiple random fields. Fourth, we show examples of depth separation in NHLs under ReLU and quadratic activation fun
    
[^6]: 高斯混合块模型中的谱聚类

    Spectral clustering in the Gaussian mixture block model. (arXiv:2305.00979v1 [stat.ML])

    [http://arxiv.org/abs/2305.00979](http://arxiv.org/abs/2305.00979)

    本文首次研究了从高维高斯混合块模型中抽样的图聚类和嵌入问题。

    

    高斯混合块模型是用于模拟现代网络的图分布：对于这样的模型生成一个图，我们将每个顶点 $i$ 与一个从高斯混合中抽样到的潜在特征向量 $u_i \in \mathbb{R}^d$ 相关联，当且仅当特征向量足够相似，即 $\langle u_i,u_j \rangle \ge \tau$ 时，我们才会添加边 $(i,j)$。高斯混合的不同组成部分表示可能具有不同特征分布的不同类型的节点，例如在社交网络中，每个组成部分都表示独特社区的不同属性。这些网络涉及到的自然算法任务有嵌入（恢复潜在的特征向量）和聚类（通过其混合组分将节点分组）。本文开启了对从高维高斯混合块模型抽样的图进行聚类和嵌入研究。

    Gaussian mixture block models are distributions over graphs that strive to model modern networks: to generate a graph from such a model, we associate each vertex $i$ with a latent feature vector $u_i \in \mathbb{R}^d$ sampled from a mixture of Gaussians, and we add edge $(i,j)$ if and only if the feature vectors are sufficiently similar, in that $\langle u_i,u_j \rangle \ge \tau$ for a pre-specified threshold $\tau$. The different components of the Gaussian mixture represent the fact that there may be different types of nodes with different distributions over features -- for example, in a social network each component represents the different attributes of a distinct community. Natural algorithmic tasks associated with these networks are embedding (recovering the latent feature vectors) and clustering (grouping nodes by their mixture component).  In this paper we initiate the study of clustering and embedding graphs sampled from high-dimensional Gaussian mixture block models, where the
    
[^7]: 表现不足以为盈，深究Rashomon的四重奏

    Performance is not enough: a story of the Rashomon's quartet. (arXiv:2302.13356v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.13356](http://arxiv.org/abs/2302.13356)

    本文介绍了Rashomon的四重奏，这是一个合成数据集，其中来自不同类别的四个模型具有几乎相同的预测性能，同时其可视化揭示了极其不同的方法来理解数据中的相关性结构。

    

    预测建模通常被简化为寻找最优模型来优化选定的性能度量。但如果第二优模型能够以完全不同的方式同样描述数据呢？第三个模型呢？最有效的模型会学到完全不同的数据关系吗？受到Anscombe四重奏的启发，本文介绍了Rashomon的四重奏，这是一个合成数据集，其中来自不同类别的四个模型具有几乎相同的预测性能。然而，它们的可视化揭示了极其不同的方法来理解数据中的相关性结构。引入的简单示例旨在进一步促进可视化作为比较预测模型超越性能的必要工具。我们需要开发富有洞察力的技术来解释模型集。

    Predictive modelling is often reduced to finding the best model that optimizes a selected performance measure. But what if the second-best model describes the data equally well but in a completely different way? What about the third? Is it possible that the most effective models learn completely different relationships in the data? Inspired by Anscombe's quartet, this paper introduces Rashomon's quartet, a synthetic dataset for which four models from different classes have practically identical predictive performance. However, their visualization reveals drastically distinct ways of understanding the correlation structure in data. The introduced simple illustrative example aims to further facilitate visualization as a mandatory tool to compare predictive models beyond their performance. We need to develop insightful techniques for the explanatory analysis of model sets.
    

