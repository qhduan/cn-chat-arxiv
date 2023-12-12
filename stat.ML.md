# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Almost Equivariance via Lie Algebra Convolutions.](http://arxiv.org/abs/2310.13164) | 本文研究了几乎等变性的主题，并提供了一个不同于现有定义的几乎等变性定义，并通过利用李群的李代数给出了在模型中编码几乎等变性的实用方法。 |
| [^2] | [Estimating Shape Distances on Neural Representations with Limited Samples.](http://arxiv.org/abs/2310.05742) | 本论文研究了在数据有限情况下，对高维神经表示进行形状距离估计的问题。通过推导出对形状距离标准估计器最坏情况下的收敛上下界，我们揭示了这个问题的挑战性质。为了克服挑战，我们引入了一种新的矩法估计器，并展示了其在高维设置下相对于标准估计器的优越性能。 |
| [^3] | [Multi-Domain Causal Representation Learning via Weak Distributional Invariances.](http://arxiv.org/abs/2310.02854) | 本文提出了一种通过弱分布不变性进行多领域因果表示学习的方法，证明了融入这种不变性的自编码器能够可靠地识别出稳定的变量集合。 |
| [^4] | [The Rashomon Importance Distribution: Getting RID of Unstable, Single Model-based Variable Importance.](http://arxiv.org/abs/2309.13775) | 提出了一种新的变量重要性框架，该框架在数据分布上是稳定的，并可以与现有的模型类和全局变量重要性指标结合使用。 |
| [^5] | [Learning Bayesian Networks with Heterogeneous Agronomic Data Sets via Mixed-Effect Models and Hierarchical Clustering.](http://arxiv.org/abs/2308.06399) | 本研究介绍了一种将混合效应模型和层次聚类应用于贝叶斯网络学习的新方法，在农学研究中广泛应用。通过整合随机效应，该方法可以提高贝叶斯网络的结构学习能力，实现因果关系网络的发现。 |
| [^6] | [Bidirectional Attention as a Mixture of Continuous Word Experts.](http://arxiv.org/abs/2307.04057) | 双向注意力模型具有混合专家权重，类似于连续词袋模型（CBOW）的统计模型，它在大型语言模型中起到了重要作用。 |
| [^7] | [The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit.](http://arxiv.org/abs/2306.17759) | 在无限深度和宽度的比例极限下，我们通过修改Softmax-based注意力模型，研究了Transformer的协方差矩阵。我们发现在初始化时，极限分布可以用随机微分方程来描述。通过修改注意力机制并使用残差连接，我们可以控制网络的稳定性和协方差结构的行为。 |
| [^8] | [Large-Scale Quantum Separability Through a Reproducible Machine Learning Lens.](http://arxiv.org/abs/2306.09444) | 本研究提出了一个机器学习管道用于大规模场景下量子可分性的近似解，通过有效算法近似查找最近的可分离密度矩阵，并将量子可分性视为分类问题，对任何二维混合状态都适用。 |
| [^9] | [Restless Bandits with Average Reward: Breaking the Uniform Global Attractor Assumption.](http://arxiv.org/abs/2306.00196) | 本文提出了一个通用的框架，将任何单臂策略转化为原始的$N$臂问题的策略，解决了依赖于复杂UGAP假设的问题，并实现了具有$O(1/\sqrt{N})$最优性差距的策略。 |
| [^10] | [The Selectively Adaptive Lasso.](http://arxiv.org/abs/2205.10697) | 本文提出了一种新算法——Selectively Adaptive Lasso（SAL），它基于HAL的理论构建，保留了无维度、非参数收敛速率的优点，同时也具有可扩展到大规模高维数据集的能力。这种算法将许多回归系数自动设置为零。 |

# 详细

[^1]: 几乎等变性通过李代数卷积

    Almost Equivariance via Lie Algebra Convolutions. (arXiv:2310.13164v1 [cs.LG])

    [http://arxiv.org/abs/2310.13164](http://arxiv.org/abs/2310.13164)

    本文研究了几乎等变性的主题，并提供了一个不同于现有定义的几乎等变性定义，并通过利用李群的李代数给出了在模型中编码几乎等变性的实用方法。

    

    最近，在机器学习中，模型相对于群作用的等变性已成为一个重要的研究课题。然而，赋予一个架构具体的群等变性对模型所期望看到的数据变换类型施加了强大的先验。严格等变模型强制执行对称性，但真实世界的数据并不总是符合这样的严格等变性，可能是因为数据中的噪声或仅编码了近似或部分对称性的潜在物理定律。在这种情况下，严格等变性的先验实际上可能过于强大，导致模型在真实数据上表现不佳。因此，在这项工作中，我们研究了一个相关的主题，即几乎等变性。我们提供了一个与当前文献中现有定义不同的几乎等变性定义，并通过利用李群的李代数给出了在模型中编码几乎等变性的实用方法。

    Recently, the equivariance of models with respect to a group action has become an important topic of research in machine learning. However, imbuing an architecture with a specific group equivariance imposes a strong prior on the types of data transformations that the model expects to see. While strictly-equivariant models enforce symmetries, real-world data does not always conform to such strict equivariances, be it due to noise in the data or underlying physical laws that encode only approximate or partial symmetries. In such cases, the prior of strict equivariance can actually prove too strong and cause models to underperform on real-world data. Therefore, in this work we study a closely related topic, that of almost equivariance. We provide a definition of almost equivariance that differs from those extant in the current literature and give a practical method for encoding almost equivariance in models by appealing to the Lie algebra of a Lie group. Specifically, we define Lie algebr
    
[^2]: 有限采样下神经表示的形状距离估计

    Estimating Shape Distances on Neural Representations with Limited Samples. (arXiv:2310.05742v1 [stat.ML])

    [http://arxiv.org/abs/2310.05742](http://arxiv.org/abs/2310.05742)

    本论文研究了在数据有限情况下，对高维神经表示进行形状距离估计的问题。通过推导出对形状距离标准估计器最坏情况下的收敛上下界，我们揭示了这个问题的挑战性质。为了克服挑战，我们引入了一种新的矩法估计器，并展示了其在高维设置下相对于标准估计器的优越性能。

    

    在神经科学和深度学习领域，衡量高维网络表示之间的几何相似性一直是一个长期的研究兴趣。尽管已经提出了许多方法，但只有少数工作对它们的统计效率进行了严格分析，或者对数据有限情况下的估计器不确定性进行了量化。在这里，我们推导出了标准形状距离估计器（由Williams et al. (2021)提出）的最坏情况收敛上下界。这些界限揭示了在高维特征空间中这个问题的挑战性质。为了克服这些挑战，我们引入了一种新的矩法估计器，具有可调的偏差-方差权衡。我们展示了这个估计器在模拟和神经数据上相对于标准估计器在高维设置下实现了更好的性能。因此，我们为高维形状分析奠定了严格的统计理论基础。

    Measuring geometric similarity between high-dimensional network representations is a topic of longstanding interest to neuroscience and deep learning. Although many methods have been proposed, only a few works have rigorously analyzed their statistical efficiency or quantified estimator uncertainty in data-limited regimes. Here, we derive upper and lower bounds on the worst-case convergence of standard estimators of shape distance$\unicode{x2014}$a measure of representational dissimilarity proposed by Williams et al. (2021). These bounds reveal the challenging nature of the problem in high-dimensional feature spaces. To overcome these challenges, we introduce a new method-of-moments estimator with a tunable bias-variance tradeoff. We show that this estimator achieves superior performance to standard estimators in simulation and on neural data, particularly in high-dimensional settings. Thus, we lay the foundation for a rigorous statistical theory for high-dimensional shape analysis, an
    
[^3]: 通过弱分布不变性实现多领域因果表示学习

    Multi-Domain Causal Representation Learning via Weak Distributional Invariances. (arXiv:2310.02854v1 [cs.LG])

    [http://arxiv.org/abs/2310.02854](http://arxiv.org/abs/2310.02854)

    本文提出了一种通过弱分布不变性进行多领域因果表示学习的方法，证明了融入这种不变性的自编码器能够可靠地识别出稳定的变量集合。

    

    因果表示学习已成为因果机器学习研究的核心。特别是，多领域数据集为展示因果表示学习相对于标准无监督表示学习的优势提供了自然机会。虽然最近的研究在学习因果表示方面取得了重要进展，但由于过于简化数据的假设，它们往往不能适用于多领域数据集；例如，每个领域都来自不同的单节点完美干预。在本文中，我们放宽了这些假设，并利用以下观察结果：在多领域数据中，往往存在一部分潜变量的某些分布属性（例如支持度、方差）在不同领域之间保持稳定；当每个领域来自多节点不完美干预时，这个属性成立。利用这个观察结果，我们证明了融入这种不变性的自编码器能够可靠地识别出稳定的变量集合。

    Causal representation learning has emerged as the center of action in causal machine learning research. In particular, multi-domain datasets present a natural opportunity for showcasing the advantages of causal representation learning over standard unsupervised representation learning. While recent works have taken crucial steps towards learning causal representations, they often lack applicability to multi-domain datasets due to over-simplifying assumptions about the data; e.g. each domain comes from a different single-node perfect intervention. In this work, we relax these assumptions and capitalize on the following observation: there often exists a subset of latents whose certain distributional properties (e.g., support, variance) remain stable across domains; this property holds when, for example, each domain comes from a multi-node imperfect intervention. Leveraging this observation, we show that autoencoders that incorporate such invariances can provably identify the stable set o
    
[^4]: 论文标题：The Rashomon Importance Distribution: 摆脱不稳定的基于单一模型的变量重要性

    The Rashomon Importance Distribution: Getting RID of Unstable, Single Model-based Variable Importance. (arXiv:2309.13775v1 [cs.LG])

    [http://arxiv.org/abs/2309.13775](http://arxiv.org/abs/2309.13775)

    提出了一种新的变量重要性框架，该框架在数据分布上是稳定的，并可以与现有的模型类和全局变量重要性指标结合使用。

    

    量化变量重要性对于回答遗传学、公共政策和医学等领域的重大问题至关重要。当前的方法通常计算给定数据集上训练的给定模型的变量重要性。然而，对于给定数据集，可能有许多模型同样能解释目标结果;如果不考虑所有可能的解释，不同的研究者可能会得出许多冲突但同样有效的结论。此外，即使考虑了给定数据集的所有可能解释，这些洞察力可能不具有普适性，因为并非所有好的解释在合理的数据扰动下都是稳定的。我们提出了一种新的变量重要性框架，该框架量化了在所有好的模型集合中的变量重要性，并且在数据分布上是稳定的。我们的框架非常灵活，可以与大多数现有的模型类和全局变量重要性指标结合使用。

    Quantifying variable importance is essential for answering high-stakes questions in fields like genetics, public policy, and medicine. Current methods generally calculate variable importance for a given model trained on a given dataset. However, for a given dataset, there may be many models that explain the target outcome equally well; without accounting for all possible explanations, different researchers may arrive at many conflicting yet equally valid conclusions given the same data. Additionally, even when accounting for all possible explanations for a given dataset, these insights may not generalize because not all good explanations are stable across reasonable data perturbations. We propose a new variable importance framework that quantifies the importance of a variable across the set of all good models and is stable across the data distribution. Our framework is extremely flexible and can be integrated with most existing model classes and global variable importance metrics. We d
    
[^5]: 通过混合效应模型和层次聚类学习具有异构农业数据集的贝叶斯网络

    Learning Bayesian Networks with Heterogeneous Agronomic Data Sets via Mixed-Effect Models and Hierarchical Clustering. (arXiv:2308.06399v1 [stat.ML])

    [http://arxiv.org/abs/2308.06399](http://arxiv.org/abs/2308.06399)

    本研究介绍了一种将混合效应模型和层次聚类应用于贝叶斯网络学习的新方法，在农学研究中广泛应用。通过整合随机效应，该方法可以提高贝叶斯网络的结构学习能力，实现因果关系网络的发现。

    

    在涉及多样但相关数据集的研究中，其中协变量与结果之间的关联可能会有所不同，在包括农学研究在内的各个领域都很普遍。在这种情况下，常常使用层次模型，也被称为多层模型，来融合来自不同数据集的信息，并适应它们的不同特点。然而，它们的结构超出了简单的异质性，因为变量通常形成复杂的因果关系网络。贝叶斯网络（BNs）使用有向无环图来模拟这种关系的强大框架。本研究介绍了一种将随机效应整合到BN学习中的新方法。这种方法基于线性混合效应模型，特别适用于处理层次数据。来自真实农学试验的结果表明，采用这种方法可以增强结构学习，从而实现发现

    Research involving diverse but related data sets, where associations between covariates and outcomes may vary, is prevalent in various fields including agronomic studies. In these scenarios, hierarchical models, also known as multilevel models, are frequently employed to assimilate information from different data sets while accommodating their distinct characteristics. However, their structure extend beyond simple heterogeneity, as variables often form complex networks of causal relationships.  Bayesian networks (BNs) provide a powerful framework for modelling such relationships using directed acyclic graphs to illustrate the connections between variables. This study introduces a novel approach that integrates random effects into BN learning. Rooted in linear mixed-effects models, this approach is particularly well-suited for handling hierarchical data. Results from a real-world agronomic trial suggest that employing this approach enhances structural learning, leading to the discovery 
    
[^6]: 双向注意力作为连续词专家的混合物

    Bidirectional Attention as a Mixture of Continuous Word Experts. (arXiv:2307.04057v1 [cs.CL])

    [http://arxiv.org/abs/2307.04057](http://arxiv.org/abs/2307.04057)

    双向注意力模型具有混合专家权重，类似于连续词袋模型（CBOW）的统计模型，它在大型语言模型中起到了重要作用。

    

    双向注意力由位置编码和屏蔽语言模型（MLM）目标组成的自注意力构成，已成为现代大型语言模型（LLMs）的关键组件。尽管它在实践中取得了成功，但很少有研究探讨它的统计基础：双向注意力隐含地拟合了什么统计模型？它与非注意机制的先驱有何不同？本文探讨了这些问题。关键观察是，重新参数化后，拟合单层单头双向注意力等于拟合具有混合专家权重的连续词袋（CBOW）模型。此外，具有多个头和多个层的双向注意力等价于堆叠的MoEs和MoEs的混合。这个统计观点揭示了MoE在双向注意力中的独特用途，这与其在处理异构性方面的实际有效性相一致。

    Bidirectional attention $\unicode{x2013}$ composed of self-attention with positional encodings and the masked language model (MLM) objective $\unicode{x2013}$ has emerged as a key component of modern large language models (LLMs). Despite its empirical success, few studies have examined its statistical underpinnings: What statistical model is bidirectional attention implicitly fitting? What sets it apart from its non-attention predecessors? We explore these questions in this paper. The key observation is that fitting a single-layer single-head bidirectional attention, upon reparameterization, is equivalent to fitting a continuous bag of words (CBOW) model with mixture-of-experts (MoE) weights. Further, bidirectional attention with multiple heads and multiple layers is equivalent to stacked MoEs and a mixture of MoEs, respectively. This statistical viewpoint reveals the distinct use of MoE in bidirectional attention, which aligns with its practical effectiveness in handling heterogeneous
    
[^7]: 受形状改变的Transformer：在无限深度和宽度极限中的注意力模型

    The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit. (arXiv:2306.17759v1 [stat.ML])

    [http://arxiv.org/abs/2306.17759](http://arxiv.org/abs/2306.17759)

    在无限深度和宽度的比例极限下，我们通过修改Softmax-based注意力模型，研究了Transformer的协方差矩阵。我们发现在初始化时，极限分布可以用随机微分方程来描述。通过修改注意力机制并使用残差连接，我们可以控制网络的稳定性和协方差结构的行为。

    

    在深度学习理论中，表示的协方差矩阵用作检查网络可训练性的代理。受Transformer的成功启发，我们研究了在无限深度和宽度的比例极限下，带有跳跃连接的修改Softmax-based注意力模型的协方差矩阵。我们展示了在初始化时，极限分布可以用深度与宽度比率为索引的随机微分方程（SDE）来描述。为了实现良定义的随机极限，Transformer的注意力机制通过将Softmax输出居中在单位矩阵上，并通过宽度相关的温度参数对Softmax logits进行缩放来进行修改。我们通过相应的SDE研究了网络的稳定性，展示了如何通过残差连接优雅地控制漂移和扩散的尺度。稳定SDE的存在意味着协方差结构是良 behaved 的，即使对于非常大的深度和宽度也是如此。

    In deep learning theory, the covariance matrix of the representations serves as a proxy to examine the network's trainability. Motivated by the success of Transformers, we study the covariance matrix of a modified Softmax-based attention model with skip connections in the proportional limit of infinite-depth-and-width. We show that at initialization the limiting distribution can be described by a stochastic differential equation (SDE) indexed by the depth-to-width ratio. To achieve a well-defined stochastic limit, the Transformer's attention mechanism is modified by centering the Softmax output at identity, and scaling the Softmax logits by a width-dependent temperature parameter. We examine the stability of the network through the corresponding SDE, showing how the scale of both the drift and diffusion can be elegantly controlled with the aid of residual connections. The existence of a stable SDE implies that the covariance structure is well-behaved, even for very large depth and widt
    
[^8]: 基于可复制的机器学习方法的大规模量子可分性研究

    Large-Scale Quantum Separability Through a Reproducible Machine Learning Lens. (arXiv:2306.09444v1 [quant-ph])

    [http://arxiv.org/abs/2306.09444](http://arxiv.org/abs/2306.09444)

    本研究提出了一个机器学习管道用于大规模场景下量子可分性的近似解，通过有效算法近似查找最近的可分离密度矩阵，并将量子可分性视为分类问题，对任何二维混合状态都适用。

    

    量子可分性问题是指如何判断一个二分体密度矩阵是纠缠的还是可分的。我们提出了一种机器学习管道，用于在大规模场景下找到此NP-难问题的近似解。我们提供了一种基于Frank-Wolfe的有效算法来近似查找最近的可分离密度矩阵，并推导了一种系统的方法将密度矩阵标记为可分离的或纠缠的，使我们能够将量子可分性视为分类问题。我们的方法适用于任何二维混合状态。对3-和7维度中的量子态进行的数值实验验证了所提出的程序的效率，并证明它可以扩展到上千个密度矩阵，并具有高量子纠缠检测精度。这一进展有助于基准测试量子可分性，并支持更强大的纠缠检测技术的发展。

    The quantum separability problem consists in deciding whether a bipartite density matrix is entangled or separable. In this work, we propose a machine learning pipeline for finding approximate solutions for this NP-hard problem in large-scale scenarios. We provide an efficient Frank-Wolfe-based algorithm to approximately seek the nearest separable density matrix and derive a systematic way for labeling density matrices as separable or entangled, allowing us to treat quantum separability as a classification problem. Our method is applicable to any two-qudit mixed states. Numerical experiments with quantum states of 3- and 7-dimensional qudits validate the efficiency of the proposed procedure, and demonstrate that it scales up to thousands of density matrices with a high quantum entanglement detection accuracy. This takes a step towards benchmarking quantum separability to support the development of more powerful entanglement detection techniques.
    
[^9]: 具有平均奖励的不安定赌徒问题：打破统一全局引子假设

    Restless Bandits with Average Reward: Breaking the Uniform Global Attractor Assumption. (arXiv:2306.00196v1 [cs.LG])

    [http://arxiv.org/abs/2306.00196](http://arxiv.org/abs/2306.00196)

    本文提出了一个通用的框架，将任何单臂策略转化为原始的$N$臂问题的策略，解决了依赖于复杂UGAP假设的问题，并实现了具有$O(1/\sqrt{N})$最优性差距的策略。

    

    我们研究了具有平均奖励标准下的无限时不安定赌徒问题，包括离散时间和连续时间设置。一个基本问题是如何设计计算有效的策略，使得优化差距随着臂的数量$N$的增加而减小。现有的渐近最优性结果都依赖于统一全局引子性质(UGAP)，这是一个复杂且难以验证的假设。在本文中，我们提出了一个通用的、基于模拟的框架，将任何单臂策略转化为原始的$N$臂问题的策略。这是通过在每个臂上模拟单臂策略，并仔细地将真实状态引导向模拟状态来实现的。我们的框架可以实例化，产生一个具有$O(1/\sqrt{N})$的最优解差距的策略。在离散时间设置中，我们的结果在更简单的同步假设下成立，涵盖了一些不满足UGAP的问题实例。更值得注意的是，我们的框架可以处理比现有方法更大的问题类，而不需对问题实例做任何特定的结构假设。

    We study the infinite-horizon restless bandit problem with the average reward criterion, under both discrete-time and continuous-time settings. A fundamental question is how to design computationally efficient policies that achieve a diminishing optimality gap as the number of arms, $N$, grows large. Existing results on asymptotical optimality all rely on the uniform global attractor property (UGAP), a complex and challenging-to-verify assumption. In this paper, we propose a general, simulation-based framework that converts any single-armed policy into a policy for the original $N$-armed problem. This is accomplished by simulating the single-armed policy on each arm and carefully steering the real state towards the simulated state. Our framework can be instantiated to produce a policy with an $O(1/\sqrt{N})$ optimality gap. In the discrete-time setting, our result holds under a simpler synchronization assumption, which covers some problem instances that do not satisfy UGAP. More notabl
    
[^10]: Selectively Adaptive Lasso选适应Lasso

    The Selectively Adaptive Lasso. (arXiv:2205.10697v5 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2205.10697](http://arxiv.org/abs/2205.10697)

    本文提出了一种新算法——Selectively Adaptive Lasso（SAL），它基于HAL的理论构建，保留了无维度、非参数收敛速率的优点，同时也具有可扩展到大规模高维数据集的能力。这种算法将许多回归系数自动设置为零。

    

    机器学习回归方法能够进行无需过多的参数假设的函数估计。虽然它们可以在预测误差方面表现出色，但大多数缺乏类半参数有效估计（例如，TMLE，AIPW）所需的理论收敛速度。高度自适应Lasso（HAL）是唯一经证明能够快速收敛到意义上的大类函数的回归方法，与预测变量的维度无关。不幸的是，HAL无法扩展计算。在本文中，我们在HAL理论的基础上构建选择自适应Lasso（SAL），一种新的算法，保留HAL的无维度、非参数收敛率，但也能扩展到大规模的高维数据集。为了实现这一目标，我们证明了一些与嵌套Donsker类中的经验损失最小化有关的一般理论结果。我们的算法是一种梯度下降形式，具有简单的分组规则，自动将许多回归系数设为零。

    Machine learning regression methods allow estimation of functions without unrealistic parametric assumptions. Although they can perform exceptionally in prediction error, most lack theoretical convergence rates necessary for semi-parametric efficient estimation (e.g. TMLE, AIPW) of parameters like average treatment effects. The Highly Adaptive Lasso (HAL) is the only regression method proven to converge quickly enough for a meaningfully large class of functions, independent of the dimensionality of the predictors. Unfortunately, HAL is not computationally scalable. In this paper we build upon the theory of HAL to construct the Selectively Adaptive Lasso (SAL), a new algorithm which retains HAL's dimension-free, nonparametric convergence rate but which also scales computationally to large high-dimensional datasets. To accomplish this, we prove some general theoretical results pertaining to empirical loss minimization in nested Donsker classes. Our resulting algorithm is a form of gradie
    

