# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs](https://arxiv.org/abs/2402.07309) | 本文提出了HyperBERT模型，通过在预训练的BERT模型中引入超图感知层，克服了现有方法在节点分类任务上难以捕捉超图结构信息和文本属性的局限性，提高了模型的效果和泛化能力。 |
| [^2] | [On Rademacher Complexity-based Generalization Bounds for Deep Learning](https://arxiv.org/abs/2208.04284) | 该论文研究了基于Rademacher复杂度的方法在对卷积神经网络进行少类别图像分类时生成非空泛化界限。其中的关键技术贡献是发展了针对函数空间和具有一般Lipschitz激活函数的CNNs的新的Talagrand压缩引理。 |
| [^3] | [Estimating heterogeneous treatment effect from survival outcomes via (orthogonal) censoring unbiased learning.](http://arxiv.org/abs/2401.11263) | 该论文开发了一种适用于具有和没有竞争风险的生存结果的截尾无偏变换方法，可以估计异质治疗效应。这种方法可以应用于更多最先进的适用于被截尾结果的HTE学习方法，并提供了限制有限样本过度风险的方法。 |
| [^4] | [Optimal Differentially Private PCA and Estimation for Spiked Covariance Matrices.](http://arxiv.org/abs/2401.03820) | 本文研究了在尖峰协方差模型中的最优差分隐私主成分分析和协方差估计问题，并提出了高效的差分隐私估计器，并证明了它们的最小最大性。 |
| [^5] | [Exponential Quantum Communication Advantage in Distributed Learning.](http://arxiv.org/abs/2310.07136) | 在分布式学习中，我们提出了一个基于量子网络的框架，可以使用指数级较少的通信和相对较小的时间和空间复杂度开销进行推理和训练。这是第一个展示了具有密集经典数据的通用机器学习问题具有指数量子优势的例子。 |
| [^6] | [A Differentially Private Weighted Empirical Risk Minimization Procedure and its Application to Outcome Weighted Learning.](http://arxiv.org/abs/2307.13127) | 本文提出了一种差分隐私加权经验风险最小化算法，可以在使用敏感数据的情况下保护隐私。这是第一个在权重ERM中应用差分隐私的算法，并且在一定的条件下提供了严格的DP保证。 |
| [^7] | [Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality.](http://arxiv.org/abs/2212.09900) | 本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。 |

# 详细

[^1]: HyperBERT:将混合超图感知层与语言模型用于文本属性超图上的节点分类

    HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs

    [https://arxiv.org/abs/2402.07309](https://arxiv.org/abs/2402.07309)

    本文提出了HyperBERT模型，通过在预训练的BERT模型中引入超图感知层，克服了现有方法在节点分类任务上难以捕捉超图结构信息和文本属性的局限性，提高了模型的效果和泛化能力。

    

    超图通过复杂的拓扑结构标记，表达多个实体之间的高阶相互作用，其中超边扮演重要角色。最近，基于超图的深度学习方法在学习文本属性超图上的节点分类问题中引起了越来越多的研究关注。然而，现有方法往往难以同时捕捉超图结构信息的全部内容和节点属性中的丰富语言属性，这在很大程度上影响了它们的效果和泛化能力。为了克服这些挑战，我们探索了如何通过为节点分类任务进一步增强预训练的BERT模型，引入专门的超图感知层。这些层将高阶结构归纳偏差引入语言模型中，从而提高模型利用超图结构中的高阶上下文信息和文本中的语义信息的能力。

    Hypergraphs are marked by complex topology, expressing higher-order interactions among multiple entities with hyperedges. Lately, hypergraph-based deep learning methods to learn informative data representations for the problem of node classification on text-attributed hypergraphs have garnered increasing research attention. However, existing methods struggle to simultaneously capture the full extent of hypergraph structural information and the rich linguistic attributes inherent in the nodes attributes, which largely hampers their effectiveness and generalizability. To overcome these challenges, we explore ways to further augment a pretrained BERT model with specialized hypergraph-aware layers for the task of node classification. Such layers introduce higher-order structural inductive bias into the language model, thus improving the model's capacity to harness both higher-order context information from the hypergraph structure and semantic information present in text. In this paper, we
    
[^2]: 基于Rademacher复杂度的深度学习一般化界限研究

    On Rademacher Complexity-based Generalization Bounds for Deep Learning

    [https://arxiv.org/abs/2208.04284](https://arxiv.org/abs/2208.04284)

    该论文研究了基于Rademacher复杂度的方法在对卷积神经网络进行少类别图像分类时生成非空泛化界限。其中的关键技术贡献是发展了针对函数空间和具有一般Lipschitz激活函数的CNNs的新的Talagrand压缩引理。

    

    我们展示了基于Rademacher复杂度的方法可以生成对卷积神经网络（CNNs）进行分类少量类别图像非空泛化界限。新的Talagrand压缩引理的发展对于高维映射函数空间和具有一般Lipschitz激活函数的CNNs是一个关键技术贡献。我们的结果表明，Rademacher复杂度不依赖于CNNs的网络长度，特别是对于诸如ReLU，Leaky ReLU，Parametric Rectifier Linear Unit，Sigmoid和Tanh等特定类型的激活函数。

    We show that the Rademacher complexity-based approach can generate non-vacuous generalisation bounds on Convolutional Neural Networks (CNNs) for classifying a small number of classes of images. The development of new Talagrand's contraction lemmas for high-dimensional mappings between function spaces and CNNs for general Lipschitz activation functions is a key technical contribution. Our results show that the Rademacher complexity does not depend on the network length for CNNs with some special types of activation functions such as ReLU, Leaky ReLU, Parametric Rectifier Linear Unit, Sigmoid, and Tanh.
    
[^3]: 通过（正交）完全无偏的截尾学习来估计生存结果的异质治疗效应

    Estimating heterogeneous treatment effect from survival outcomes via (orthogonal) censoring unbiased learning. (arXiv:2401.11263v1 [stat.ME])

    [http://arxiv.org/abs/2401.11263](http://arxiv.org/abs/2401.11263)

    该论文开发了一种适用于具有和没有竞争风险的生存结果的截尾无偏变换方法，可以估计异质治疗效应。这种方法可以应用于更多最先进的适用于被截尾结果的HTE学习方法，并提供了限制有限样本过度风险的方法。

    

    从观察数据中估计异质治疗效应（HTE）的方法主要集中在连续或二元结果上，较少关注生存结果，几乎没有关注竞争风险情景。在这项工作中，我们开发了适用于具有和没有竞争风险的生存结果的截尾无偏变换（CUTs）。使用这些CUTs将时间到事件结果转换后，对连续结果的HTE学习方法的直接应用可以产生一致估计的异质累积发生率效应、总效应和可分离直接效应。我们的CUTs可以使用比以前更多的最先进的适用于被截尾结果的HTE学习方法，特别是在竞争风险情景下。我们提供了通用的无模型学习特定oracle不等式来限制有限样本的过度风险。oracle效率结果取决于一个oracle选择器和从所有步骤中估计的干扰函数。

    Methods for estimating heterogeneous treatment effects (HTE) from observational data have largely focused on continuous or binary outcomes, with less attention paid to survival outcomes and almost none to settings with competing risks. In this work, we develop censoring unbiased transformations (CUTs) for survival outcomes both with and without competing risks.After converting time-to-event outcomes using these CUTs, direct application of HTE learners for continuous outcomes yields consistent estimates of heterogeneous cumulative incidence effects, total effects, and separable direct effects. Our CUTs enable application of a much larger set of state of the art HTE learners for censored outcomes than had previously been available, especially in competing risks settings. We provide generic model-free learner-specific oracle inequalities bounding the finite-sample excess risk. The oracle efficiency results depend on the oracle selector and estimated nuisance functions from all steps invol
    
[^4]: 在带有尖峰协方差矩阵中的最优差分隐私主成分分析和估计

    Optimal Differentially Private PCA and Estimation for Spiked Covariance Matrices. (arXiv:2401.03820v1 [math.ST])

    [http://arxiv.org/abs/2401.03820](http://arxiv.org/abs/2401.03820)

    本文研究了在尖峰协方差模型中的最优差分隐私主成分分析和协方差估计问题，并提出了高效的差分隐私估计器，并证明了它们的最小最大性。

    

    在当代统计学中，估计协方差矩阵及其相关的主成分是一个基本问题。尽管已开发出具有良好性质的最优估计程序，但对隐私保护的增加需求给这个经典问题引入了新的复杂性。本文研究了在尖峰协方差模型中的最优差分隐私主成分分析（PCA）和协方差估计。我们精确地刻画了在该模型下特征值和特征向量的敏感性，并建立了估计主成分和协方差矩阵的最小最大收敛率。这些收敛率包括一般的Schatten范数，包括谱范数，Frobenius范数和核范数。我们引入了计算高效的差分隐私估计器，并证明它们的最小最大性，直到对数因子。另外，匹配的minimax最小最大率也得到了证明。

    Estimating a covariance matrix and its associated principal components is a fundamental problem in contemporary statistics. While optimal estimation procedures have been developed with well-understood properties, the increasing demand for privacy preservation introduces new complexities to this classical problem. In this paper, we study optimal differentially private Principal Component Analysis (PCA) and covariance estimation within the spiked covariance model.  We precisely characterize the sensitivity of eigenvalues and eigenvectors under this model and establish the minimax rates of convergence for estimating both the principal components and covariance matrix. These rates hold up to logarithmic factors and encompass general Schatten norms, including spectral norm, Frobenius norm, and nuclear norm as special cases.  We introduce computationally efficient differentially private estimators and prove their minimax optimality, up to logarithmic factors. Additionally, matching minimax l
    
[^5]: 分布式学习中的指数量子通信优势

    Exponential Quantum Communication Advantage in Distributed Learning. (arXiv:2310.07136v1 [quant-ph])

    [http://arxiv.org/abs/2310.07136](http://arxiv.org/abs/2310.07136)

    在分布式学习中，我们提出了一个基于量子网络的框架，可以使用指数级较少的通信和相对较小的时间和空间复杂度开销进行推理和训练。这是第一个展示了具有密集经典数据的通用机器学习问题具有指数量子优势的例子。

    

    使用超过单个设备内存容量的大型机器学习模型进行训练和推理需要设计分布式架构，必须考虑通信限制。我们提出了一种在量子网络上进行分布式计算的框架，其中数据被编码为特殊的量子态。我们证明，在该框架内的某些模型中，使用梯度下降进行推理和训练的通信开销相对于其经典对应模型可以指数级降低，并且相对于标准基于梯度的方法，时间和空间复杂性开销相对较小。据我们所知，这是第一个在具有密集经典数据的通用机器学习问题的情况下，无论数据编码成本如何，都具有指数量子优势的示例。此外，我们还展示了该类模型可以编码输入的高度非线性特征，并且它们的表达能力呈指数增加。

    Training and inference with large machine learning models that far exceed the memory capacity of individual devices necessitates the design of distributed architectures, forcing one to contend with communication constraints. We present a framework for distributed computation over a quantum network in which data is encoded into specialized quantum states. We prove that for certain models within this framework, inference and training using gradient descent can be performed with exponentially less communication compared to their classical analogs, and with relatively modest time and space complexity overheads relative to standard gradient-based methods. To our knowledge, this is the first example of exponential quantum advantage for a generic class of machine learning problems with dense classical data that holds regardless of the data encoding cost. Moreover, we show that models in this class can encode highly nonlinear features of their inputs, and their expressivity increases exponenti
    
[^6]: 一个差分隐私加权经验风险最小化算法及其在结果加权学习中的应用

    A Differentially Private Weighted Empirical Risk Minimization Procedure and its Application to Outcome Weighted Learning. (arXiv:2307.13127v1 [stat.ML])

    [http://arxiv.org/abs/2307.13127](http://arxiv.org/abs/2307.13127)

    本文提出了一种差分隐私加权经验风险最小化算法，可以在使用敏感数据的情况下保护隐私。这是第一个在权重ERM中应用差分隐私的算法，并且在一定的条件下提供了严格的DP保证。

    

    在经验风险最小化(ERM)框架中，使用包含个人信息的数据来构建预测模型是常见的做法。尽管这些模型在预测上可以非常准确，但使用敏感数据得到的结果可能容易受到隐私攻击。差分隐私(DP)是一种有吸引力的框架，可以通过提供数学上可证明的隐私损失界限来解决这些数据隐私问题。先前的工作主要集中在将DP应用于无权重的ERM中。我们考虑到了权重ERM(wERM)的重要推广。在wERM中，可以为每个个体的目标函数贡献分配不同的权重。在这个背景下，我们提出了第一个有差分隐私保障的wERM算法，并在一定的正则条件下提供了严格的理论证明。将现有的DP-ERM程序扩展到wERM为结果加权学习铺平了道路。

    It is commonplace to use data containing personal information to build predictive models in the framework of empirical risk minimization (ERM). While these models can be highly accurate in prediction, results obtained from these models with the use of sensitive data may be susceptible to privacy attacks. Differential privacy (DP) is an appealing framework for addressing such data privacy issues by providing mathematically provable bounds on the privacy loss incurred when releasing information from sensitive data. Previous work has primarily concentrated on applying DP to unweighted ERM. We consider an important generalization to weighted ERM (wERM). In wERM, each individual's contribution to the objective function can be assigned varying weights. In this context, we propose the first differentially private wERM algorithm, backed by a rigorous theoretical proof of its DP guarantees under mild regularity conditions. Extending the existing DP-ERM procedures to wERM paves a path to derivin
    
[^7]: 无交叠策略学习：悲观和广义经验Bernstein不等式

    Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality. (arXiv:2212.09900v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.09900](http://arxiv.org/abs/2212.09900)

    本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。

    

    本文研究了离线策略学习，旨在利用先前收集到的观测（来自于固定的或是适应演变的行为策略）来学习给定类别中的最优个性化决策规则。现有的策略学习方法依赖于一个统一交叠假设，即离线数据集中探索所有个性化特征的所有动作的倾向性下界。换句话说，这些方法的性能取决于离线数据集中最坏的倾向性。由于数据收集过程不受控制，在许多情况下，这种假设可能不太现实，特别是当允许行为策略随时间演变并且倾向性减弱时。为此，本文提出了一种新的算法，它优化策略价值的下限置信区间（LCBs）——而不是点估计。LCBs通过量化增强倒数倾向权重的估计不确定性来构建。

    This paper studies offline policy learning, which aims at utilizing observations collected a priori (from either fixed or adaptively evolving behavior policies) to learn the optimal individualized decision rule in a given class. Existing policy learning methods rely on a uniform overlap assumption, i.e., the propensities of exploring all actions for all individual characteristics are lower bounded in the offline dataset. In other words, the performance of these methods depends on the worst-case propensity in the offline dataset. As one has no control over the data collection process, this assumption can be unrealistic in many situations, especially when the behavior policies are allowed to evolve over time with diminishing propensities.  In this paper, we propose a new algorithm that optimizes lower confidence bounds (LCBs) -- instead of point estimates -- of the policy values. The LCBs are constructed by quantifying the estimation uncertainty of the augmented inverse propensity weight
    

