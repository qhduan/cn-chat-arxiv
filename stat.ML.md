# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoDiff: combining Auto-encoder and Diffusion model for tabular data synthesizing.](http://arxiv.org/abs/2310.15479) | 使用自动编码器和扩散模型结合的AutoDiff模型可以有效地生成合成的表格数据，克服了表格数据中的异构特征和特征间相关性的挑战，生成的数据与真实数据在统计上非常相似，并在机器学习任务中表现良好。 |
| [^2] | [Random Forest Dissimilarity for High-Dimension Low Sample Size Classification.](http://arxiv.org/abs/2310.14710) | 针对高维度低样本(HDLSS)分类问题，本论文提出了一种基于随机森林相似度的学习预计算SVM核方法(RFSVM)，通过在40个公共HDLSS分类数据集上的实验证明，该方法在HDLSS问题上优于现有方法并且保持了非常一致的结果。 |
| [^3] | [In search of dispersed memories: Generative diffusion models are associative memory networks.](http://arxiv.org/abs/2309.17290) | 本研究将生成扩散模型解释为基于能量的模型，证明其在训练离散模式时与现代Hopfield网络的能量函数等效。这种等效性使得我们可以将扩散模型的有监督训练解释为在权重结构中编码现代Hopfield网络的关联动力学的突触学习过程。 |
| [^4] | [Approximately Equivariant Graph Networks.](http://arxiv.org/abs/2308.10436) | 本文关注于图神经网络（GNNs）的主动对称性，通过考虑信号在固定图上的学习设置，提出了一种近似的对称性概念，通过图粗化实现。这篇工作提出了一个偏差-方差公式来衡量近似对称性... |
| [^5] | [Optimal partitioning of directed acyclic graphs with dependent costs between clusters.](http://arxiv.org/abs/2308.03970) | 本论文提出了一种名为DCMAP的算法，用于对具有依赖成本的有向无环图进行最优分区。该算法通过优化基于DAG和集群映射的成本函数来寻找所有最优集群，并在途中返回接近最优解。实验证明在复杂系统的DBN模型中，该算法具有时间效率性。 |
| [^6] | [Addressing caveats of neural persistence with deep graph persistence.](http://arxiv.org/abs/2307.10865) | 本文发现网络权重的方差和大权重的空间集中是影响神经持久性的主要因素，并提出了将神经持久性扩展到整个神经网络的深度图持久性测量方法。 |
| [^7] | [Solving Linear Inverse Problems using Higher-Order Annealed Langevin Diffusion.](http://arxiv.org/abs/2305.05014) | 本论文提出了一种高阶 Langevin 动力学的算法，可更高效地采样未知变量后验分布，同时加入淬火过程，能应用于离散未知变量情况，数值实验表明在多个任务中相对竞争方法具有更高性能。 |
| [^8] | [Inferential Moments of Uncertain Multivariable Systems.](http://arxiv.org/abs/2305.01841) | 本文提出了一种新的分析不确定多变量系统行为的方法，使用推断矩描述分布预计如何响应新信息，特别关注推断偏差，以改善情境感知能力。 |
| [^9] | [Understanding and Exploring the Whole Set of Good Sparse Generalized Additive Models.](http://arxiv.org/abs/2303.16047) | 提出一种有效而准确地近似稀疏广义可加模型（GAMs）的Rashomon集的技术，并使用这些近似模型来解决实际应用的挑战。 |
| [^10] | [Causal Inference of General Treatment Effects using Neural Networks with A Diverging Number of Confounders.](http://arxiv.org/abs/2009.07055) | 本文提出了一种利用神经网络进行因果推断的方法，已经通过仿真研究验证了其可行性，可以缓解维数诅咒问题。 |

# 详细

[^1]: AutoDiff:结合自动编码器和扩散模型用于表格数据合成

    AutoDiff: combining Auto-encoder and Diffusion model for tabular data synthesizing. (arXiv:2310.15479v1 [stat.ML])

    [http://arxiv.org/abs/2310.15479](http://arxiv.org/abs/2310.15479)

    使用自动编码器和扩散模型结合的AutoDiff模型可以有效地生成合成的表格数据，克服了表格数据中的异构特征和特征间相关性的挑战，生成的数据与真实数据在统计上非常相似，并在机器学习任务中表现良好。

    

    扩散模型已成为现代机器学习许多子领域中合成数据生成的主要范式，包括计算机视觉、语言模型或语音合成。在本文中，我们利用扩散模型的力量来生成合成的表格数据。表格数据中的异构特征一直是表格数据合成的主要障碍，我们通过使用自动编码器的架构来解决这个问题。与最先进的表格合成器相比，我们模型生成的合成表格在统计上与真实数据非常相似，并在机器学习工具的下游任务中表现良好。我们在15个公开可用的数据集上进行了实验。值得注意的是，我们的模型灵活地捕捉了特征之间的相关性，这是表格数据合成中长期存在的挑战。如若接纳了论文，我们的代码将根据要求提供，并且将公开发布。

    Diffusion model has become a main paradigm for synthetic data generation in many subfields of modern machine learning, including computer vision, language model, or speech synthesis. In this paper, we leverage the power of diffusion model for generating synthetic tabular data. The heterogeneous features in tabular data have been main obstacles in tabular data synthesis, and we tackle this problem by employing the auto-encoder architecture. When compared with the state-of-the-art tabular synthesizers, the resulting synthetic tables from our model show nice statistical fidelities to the real data, and perform well in downstream tasks for machine learning utilities. We conducted the experiments over 15 publicly available datasets. Notably, our model adeptly captures the correlations among features, which has been a long-standing challenge in tabular data synthesis. Our code is available upon request and will be publicly released if paper is accepted.
    
[^2]: 高维度低样本分类的随机森林差异性

    Random Forest Dissimilarity for High-Dimension Low Sample Size Classification. (arXiv:2310.14710v1 [stat.ML])

    [http://arxiv.org/abs/2310.14710](http://arxiv.org/abs/2310.14710)

    针对高维度低样本(HDLSS)分类问题，本论文提出了一种基于随机森林相似度的学习预计算SVM核方法(RFSVM)，通过在40个公共HDLSS分类数据集上的实验证明，该方法在HDLSS问题上优于现有方法并且保持了非常一致的结果。

    

    高维度低样本(HDLSS)问题在机器学习的实际应用中很常见。从医学影像到文本处理，传统的机器学习算法通常无法从这样的数据中学习到最佳的概念。我们在之前的工作中提出了一种基于差异性的多视角分类方法，即随机森林差异性(RFD)，该方法在这类问题上取得了最先进的结果。在本研究中，我们将该方法的核心原则转化为解决HDLSS分类问题的方法，通过使用随机森林相似度作为学习的预计算SVM核(RFSVM)。我们展示了这样的学习相似度度量在这种分类上特别适用和准确。通过对40个公共HDLSS分类数据集进行的实验，配合严格的统计分析，结果显示RFSVM方法在大多数HDLSS问题上优于现有方法，并且同时非常连贯。

    High dimension, low sample size (HDLSS) problems are numerous among real-world applications of machine learning. From medical images to text processing, traditional machine learning algorithms are usually unsuccessful in learning the best possible concept from such data. In a previous work, we proposed a dissimilarity-based approach for multi-view classification, the Random Forest Dissimilarity (RFD), that perfoms state-of-the-art results for such problems. In this work, we transpose the core principle of this approach to solving HDLSS classification problems, by using the RF similarity measure as a learned precomputed SVM kernel (RFSVM). We show that such a learned similarity measure is particularly suited and accurate for this classification context. Experiments conducted on 40 public HDLSS classification datasets, supported by rigorous statistical analyses, show that the RFSVM method outperforms existing methods for the majority of HDLSS problems and remains at the same time very co
    
[^3]: 搜索分散的记忆：生成扩散模型是关联记忆网络

    In search of dispersed memories: Generative diffusion models are associative memory networks. (arXiv:2309.17290v1 [stat.ML])

    [http://arxiv.org/abs/2309.17290](http://arxiv.org/abs/2309.17290)

    本研究将生成扩散模型解释为基于能量的模型，证明其在训练离散模式时与现代Hopfield网络的能量函数等效。这种等效性使得我们可以将扩散模型的有监督训练解释为在权重结构中编码现代Hopfield网络的关联动力学的突触学习过程。

    

    Hopfield网络被广泛用作神经科学中的简化理论模型，用于生物关联记忆。原始的Hopfield网络通过编码二元关联模式来存储记忆，从而产生了一种称为Hebbian学习规则的突触学习机制。现代的Hopfield网络可以通过使用高度非线性的能量函数来实现指数级容量扩展。然而，这些新模型的能量函数不能直接压缩为二元突触耦合，并且也不能直接提供新的突触学习规则。在本研究中，我们展示了生成扩散模型可以被解释为基于能量的模型，并且在训练离散模式时，它们的能量函数与现代的Hopfield网络相等。这种等价性使我们能够将扩散模型的有监督训练解释为在权重结构中编码现代Hopfield网络的关联动力学的突触学习过程。

    Hopfield networks are widely used in neuroscience as simplified theoretical models of biological associative memory. The original Hopfield networks store memories by encoding patterns of binary associations, which result in a synaptic learning mechanism known as Hebbian learning rule. Modern Hopfield networks can achieve exponential capacity scaling by using highly non-linear energy functions. However, the energy function of these newer models cannot be straightforwardly compressed into binary synaptic couplings and it does not directly provide new synaptic learning rules. In this work we show that generative diffusion models can be interpreted as energy-based models and that, when trained on discrete patterns, their energy function is equivalent to that of modern Hopfield networks. This equivalence allows us to interpret the supervised training of diffusion models as a synaptic learning process that encodes the associative dynamics of a modern Hopfield network in the weight structure 
    
[^4]: 近似等变图网络

    Approximately Equivariant Graph Networks. (arXiv:2308.10436v1 [stat.ML])

    [http://arxiv.org/abs/2308.10436](http://arxiv.org/abs/2308.10436)

    本文关注于图神经网络（GNNs）的主动对称性，通过考虑信号在固定图上的学习设置，提出了一种近似的对称性概念，通过图粗化实现。这篇工作提出了一个偏差-方差公式来衡量近似对称性...

    

    图神经网络（GNNs）通常被描述为对图中的节点重新排序具有置换等变性。GNNs的这种对称性常被与欧几里得卷积神经网络（CNNs）的平移等变性比较。然而，这两种对称性本质上是不同的：CNNs的平移等变性对应于作用于图像信号的固定域的对称性（有时称为主动对称性），而在GNNs中，任何置换都作用于图信号和图域（有时描述为被动对称性）。在这项工作中，我们聚焦于GNNs的主动对称性，考虑信号在一个固定图上进行学习的情况。在这种情况下，GNNs的自然对称性是图的自同构。由于现实世界中的图往往是非对称的，我们通过形式化图粗化来放松对称性的概念，提出了一个偏差-方差公式来衡量...

    Graph neural networks (GNNs) are commonly described as being permutation equivariant with respect to node relabeling in the graph. This symmetry of GNNs is often compared to the translation equivariance symmetry of Euclidean convolution neural networks (CNNs). However, these two symmetries are fundamentally different: The translation equivariance of CNNs corresponds to symmetries of the fixed domain acting on the image signal (sometimes known as active symmetries), whereas in GNNs any permutation acts on both the graph signals and the graph domain (sometimes described as passive symmetries). In this work, we focus on the active symmetries of GNNs, by considering a learning setting where signals are supported on a fixed graph. In this case, the natural symmetries of GNNs are the automorphisms of the graph. Since real-world graphs tend to be asymmetric, we relax the notion of symmetries by formalizing approximate symmetries via graph coarsening. We present a bias-variance formula that qu
    
[^5]: 对具有依赖成本的有向无环图进行最优分区

    Optimal partitioning of directed acyclic graphs with dependent costs between clusters. (arXiv:2308.03970v1 [cs.DS])

    [http://arxiv.org/abs/2308.03970](http://arxiv.org/abs/2308.03970)

    本论文提出了一种名为DCMAP的算法，用于对具有依赖成本的有向无环图进行最优分区。该算法通过优化基于DAG和集群映射的成本函数来寻找所有最优集群，并在途中返回接近最优解。实验证明在复杂系统的DBN模型中，该算法具有时间效率性。

    

    许多统计推断场景，包括贝叶斯网络、马尔可夫过程和隐马尔可夫模型，可以通过将基础的有向无环图（DAG）划分成集群来支持。然而，在统计推断中，最优划分是具有挑战性的，因为要优化的成本取决于集群内的节点以及通过父节点和/或子节点连接的集群之间的映射，我们将其称为依赖集群。我们提出了一种名为DCMAP的新算法，用于具有依赖集群的最优集群映射。在基于DAG和集群映射的任意定义的正成本函数的基础上，我们证明DCMAP收敛于找到所有最优集群，并在途中返回接近最优解。通过实验证明，该算法对使用计算成本函数的一个海草复杂系统的DBN模型具有时间效率性。对于一个25个和50个节点的DBN，搜索空间大小分别为$9.91\times 10^9$和$1.5$

    Many statistical inference contexts, including Bayesian Networks (BNs), Markov processes and Hidden Markov Models (HMMS) could be supported by partitioning (i.e.~mapping) the underlying Directed Acyclic Graph (DAG) into clusters. However, optimal partitioning is challenging, especially in statistical inference as the cost to be optimised is dependent on both nodes within a cluster, and the mapping of clusters connected via parent and/or child nodes, which we call dependent clusters. We propose a novel algorithm called DCMAP for optimal cluster mapping with dependent clusters. Given an arbitrarily defined, positive cost function based on the DAG and cluster mappings, we show that DCMAP converges to find all optimal clusters, and returns near-optimal solutions along the way. Empirically, we find that the algorithm is time-efficient for a DBN model of a seagrass complex system using a computation cost function. For a 25 and 50-node DBN, the search space size was $9.91\times 10^9$ and $1.5
    
[^6]: 通过深度图的持久性解决神经持久性的问题

    Addressing caveats of neural persistence with deep graph persistence. (arXiv:2307.10865v1 [cs.LG])

    [http://arxiv.org/abs/2307.10865](http://arxiv.org/abs/2307.10865)

    本文发现网络权重的方差和大权重的空间集中是影响神经持久性的主要因素，并提出了将神经持久性扩展到整个神经网络的深度图持久性测量方法。

    

    神经持久性是一种用于量化神经网络复杂性的重要指标，提出于深度学习中新兴的拓扑数据分析领域。然而，在理论和实证上我们发现，网络权重的方差和大权重的空间集中是影响神经持久性的主要因素。虽然这对于线性分类器有用的信息，但我们发现在深度神经网络的后几层中没有相关的空间结构，使得神经持久性大致等于权重的方差。此外，对于深度神经网络，所提出的层间平均过程没有考虑层间的交互。基于我们的分析，我们提出了对神经持久性基础结构的扩展，从单层改为整个神经网络，这相当于在一个特定矩阵上计算神经持久性。这得到了我们的深度图持久性测量方法。

    Neural Persistence is a prominent measure for quantifying neural network complexity, proposed in the emerging field of topological data analysis in deep learning. In this work, however, we find both theoretically and empirically that the variance of network weights and spatial concentration of large weights are the main factors that impact neural persistence. Whilst this captures useful information for linear classifiers, we find that no relevant spatial structure is present in later layers of deep neural networks, making neural persistence roughly equivalent to the variance of weights. Additionally, the proposed averaging procedure across layers for deep neural networks does not consider interaction between layers. Based on our analysis, we propose an extension of the filtration underlying neural persistence to the whole neural network instead of single layers, which is equivalent to calculating neural persistence on one particular matrix. This yields our deep graph persistence measur
    
[^7]: 使用高阶淬火随机漂移解决线性反问题

    Solving Linear Inverse Problems using Higher-Order Annealed Langevin Diffusion. (arXiv:2305.05014v1 [stat.ML])

    [http://arxiv.org/abs/2305.05014](http://arxiv.org/abs/2305.05014)

    本论文提出了一种高阶 Langevin 动力学的算法，可更高效地采样未知变量后验分布，同时加入淬火过程，能应用于离散未知变量情况，数值实验表明在多个任务中相对竞争方法具有更高性能。

    

    我们提出了一种基于高阶 Langevin 漂移的线性反问题解决方案。更具体地，我们提出了预处理的二阶和三阶 Langevin 动力学，这些动力学明显地从我们感兴趣的未知变量的后验分布中采样，同时比其一阶对应物和两种动力学的非预处理版本更具计算效率。此外，我们证明了两种预处理动力学是良定义的，并且具有与非预处理情况相同的唯一不变分布。我们还加入了一个淬火过程，这具有双重优点，一方面进一步加速了算法的收敛速度，另一方面，允许我们适应未知变量为离散的情况。在两个不同的任务（MIMO 符号检测和通道估计）的数值实验中，展示了我们方法的通用性，并说明了相对于竞争方法（包括基于学习的方法）所实现的高性能。

    We propose a solution for linear inverse problems based on higher-order Langevin diffusion. More precisely, we propose pre-conditioned second-order and third-order Langevin dynamics that provably sample from the posterior distribution of our unknown variables of interest while being computationally more efficient than their first-order counterpart and the non-conditioned versions of both dynamics. Moreover, we prove that both pre-conditioned dynamics are well-defined and have the same unique invariant distributions as the non-conditioned cases. We also incorporate an annealing procedure that has the double benefit of further accelerating the convergence of the algorithm and allowing us to accommodate the case where the unknown variables are discrete. Numerical experiments in two different tasks (MIMO symbol detection and channel estimation) showcase the generality of our method and illustrate the high performance achieved relative to competing approaches (including learning-based ones)
    
[^8]: 不确定多变量系统的推断矩

    Inferential Moments of Uncertain Multivariable Systems. (arXiv:2305.01841v1 [physics.data-an])

    [http://arxiv.org/abs/2305.01841](http://arxiv.org/abs/2305.01841)

    本文提出了一种新的分析不确定多变量系统行为的方法，使用推断矩描述分布预计如何响应新信息，特别关注推断偏差，以改善情境感知能力。

    

    本文提出了一种使用称为“推断矩”的一组量来分析不确定多变量系统行为的新范式。边缘化是一种不确定性量化过程，它通过平均条件概率来量化所关注概率的期望值。推断矩是描述分布预计如何响应新信息的高阶条件概率矩。本文研究了推断偏差，它是期望的概率波动，随着推断更新另一个变量而发生变化。我们以推断矩的形式找到了互信息的幂级数展开式，这意味着推断矩逻辑可能对通常使用信息论工具执行的任务有用。我们在两个应用中探讨了贝叶斯网络的推断偏差，以改善情境感知能力。

    This article offers a new paradigm for analyzing the behavior of uncertain multivariable systems using a set of quantities we call \emph{inferential moments}. Marginalization is an uncertainty quantification process that averages conditional probabilities to quantify the \emph{expected value} of a probability of interest. Inferential moments are higher order conditional probability moments that describe how a distribution is expected to respond to new information. Of particular interest in this article is the \emph{inferential deviation}, which is the expected fluctuation of the probability of one variable in response to an inferential update of another. We find a power series expansion of the Mutual Information in terms of inferential moments, which implies that inferential moment logic may be useful for tasks typically performed with information theoretic tools. We explore this in two applications that analyze the inferential deviations of a Bayesian Network to improve situational aw
    
[^9]: 理解和探索稀疏广义可加模型的整个优秀集合

    Understanding and Exploring the Whole Set of Good Sparse Generalized Additive Models. (arXiv:2303.16047v1 [cs.LG])

    [http://arxiv.org/abs/2303.16047](http://arxiv.org/abs/2303.16047)

    提出一种有效而准确地近似稀疏广义可加模型（GAMs）的Rashomon集的技术，并使用这些近似模型来解决实际应用的挑战。

    

    在实际应用中，机器学习模型与领域专家之间的交互至关重要；然而，通常只生成单个模型的经典机器学习范式不利于此类交互。近似和探索Rashomon集，即所有近乎最优模型的集合，通过提供用户可搜索的空间包含多样性模型的方法，解决了这一实际挑战，领域专家可以从中选择。我们提出了一种有效而准确地近似稀疏广义可加模型（GAMs）的Rashomon集的技术。我们提供了用于近似具有固定支持集的GAMs的Rashomon集的椭球形算法，并使用这些椭球形近似了许多不同支持集的Rashomon集。近似的Rashomon集为解决实际挑战，例如（1）研究模型类的变量重要性；（2）在用户指定约束条件下查找模型，提供了重要的基础。

    In real applications, interaction between machine learning model and domain experts is critical; however, the classical machine learning paradigm that usually produces only a single model does not facilitate such interaction. Approximating and exploring the Rashomon set, i.e., the set of all near-optimal models, addresses this practical challenge by providing the user with a searchable space containing a diverse set of models from which domain experts can choose. We present a technique to efficiently and accurately approximate the Rashomon set of sparse, generalized additive models (GAMs). We present algorithms to approximate the Rashomon set of GAMs with ellipsoids for fixed support sets and use these ellipsoids to approximate Rashomon sets for many different support sets. The approximated Rashomon set serves as a cornerstone to solve practical challenges such as (1) studying the variable importance for the model class; (2) finding models under user-specified constraints (monotonicity
    
[^10]: 利用具有多个混淆因素的神经网络进行一般治疗效果的因果推断

    Causal Inference of General Treatment Effects using Neural Networks with A Diverging Number of Confounders. (arXiv:2009.07055v6 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2009.07055](http://arxiv.org/abs/2009.07055)

    本文提出了一种利用神经网络进行因果推断的方法，已经通过仿真研究验证了其可行性，可以缓解维数诅咒问题。

    

    因果效应的估算是行为科学、社会科学、经济学和生物医学科学的主要目标。本文考虑了一种广义优化框架，使用前馈神经网络（ANNs）有效估计一般治疗效果，即使协变量的数量可以随着样本数量的增加而增加。我们通过神经网络来估算干扰函数，并为ANN近似器建立了一个新的近似误差界限，当干扰函数属于混合Sobolev空间时。我们证明了在这种情况下，ANN可以缓解维数诅咒问题。我们进一步建立了所提出的治疗效果估计量的一致性和渐近正常性，并应用了一个加权自助法进行推断。仿真研究展示了这些方法。

    The estimation of causal effects is a primary goal of behavioral, social, economic and biomedical sciences. Under the unconfoundedness condition, adjustment for confounders requires estimating the nuisance functions relating outcome and/or treatment to confounders. This paper considers a generalized optimization framework for efficient estimation of general treatment effects using feedforward artificial neural networks (ANNs) when the number of covariates is allowed to increase with the sample size. We estimate the nuisance function by ANNs, and develop a new approximation error bound for the ANNs approximators when the nuisance function belongs to a mixed Sobolev space. We show that the ANNs can alleviate the curse of dimensionality under this circumstance. We further establish the consistency and asymptotic normality of the proposed treatment effects estimators, and apply a weighted bootstrap procedure for conducting inference. The proposed methods are illustrated via simulation stud
    

