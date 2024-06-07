# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Last-Iterate Convergence of Shuffling Gradient Methods](https://arxiv.org/abs/2403.07723) | 该论文证明了针对目标函数的洗牌梯度方法最后迭代的收敛速率，弥合了在不同设置中最后迭代的良好性能与现有理论之间的差距。 |
| [^2] | [Variational Learning is Effective for Large Deep Networks](https://arxiv.org/abs/2402.17641) | 变分学习在大型深度网络中展现出非常好的效果，IVON优化器在训练大型网络时几乎能与Adam相媲美甚至胜过它，且预测不确定性更准确，对模型微调、泛化误差预测和数据敏感性估计均有显著改进。 |
| [^3] | [Categorical Deep Learning: An Algebraic Theory of Architectures](https://arxiv.org/abs/2402.15332) | 提出了一种关于深度学习架构的代数理论，应用范畴论构建了一个桥梁，有效地涵盖了神经网络设计的不同风格，同时自然地编码了计算机科学和自动机理论中的许多标准结构。 |
| [^4] | [Imbalanced Data Clustering using Equilibrium K-Means](https://arxiv.org/abs/2402.14490) | Equilibrium K-Means（EKM）是一种新颖且简单的K均值类型算法，通过减少聚类中心在大类簇中心聚集的倾向，显著改善了不平衡数据的聚类结果。 |
| [^5] | [Predictive Uncertainty Quantification via Risk Decompositions for Strictly Proper Scoring Rules](https://arxiv.org/abs/2402.10727) | 通过引入风险分解和适当评分规则，我们提出了一个通用框架来量化预测不确定性的不同来源，并澄清了它们之间的关系。 |
| [^6] | [Rolling Diffusion Models](https://arxiv.org/abs/2402.09470) | 本文介绍了一种滚动扩散模型，用于处理时间数据，通过滑动窗口去噪并根据帧在序列中的时间先后分配不同的噪声量，更好地捕捉到复杂的时间动态。通过实验证明，在视频预测和混沌流体动力学预测任务中，该模型优于传统扩散方法。 |
| [^7] | [An operator learning perspective on parameter-to-observable maps](https://arxiv.org/abs/2402.06031) | 本论文从算子学习的视角研究了参数到可观测映射，提出了适应有限维输入和输出的傅里叶神经映射框架，并发展了通用逼近定理来支持该方法。此外，讨论了学习PtO映射的端到端方法和先学习解算子再计算可观测值的效率问题。 |
| [^8] | [Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997) | 本研究提出了离散流模型（DFMs），通过连续时间马尔可夫链实现离散空间流匹配的离散等效，为将基于流的生成模型应用于多模态连续和离散数据问题提供了解决方案。此方法在蛋白质共设计任务上取得了最先进的性能。 |
| [^9] | [A Random Matrix Approach to Low-Multilinear-Rank Tensor Approximation](https://arxiv.org/abs/2402.03169) | 该研究采用随机矩阵方法，在低多线性秩张量逼近中展示了对种植的低秩信号的估计，并根据大维谱行为和信噪比准确预测了重建性能，并给出了HOOI收敛的充分条件。 |
| [^10] | [Learning from higher-order statistics, efficiently: hypothesis tests, random features, and neural networks](https://arxiv.org/abs/2312.14922) | 神经网络在高维数据中发现统计模式，研究了如何高效地从高阶累积量中提取特征，并探讨了在尖峰累积量模型中的统计和计算限制。 |
| [^11] | [Neural Collapse for Cross-entropy Class-Imbalanced Learning with Unconstrained ReLU Feature Model.](http://arxiv.org/abs/2401.02058) | 本文研究了在交叉熵类别不平衡学习中的神经折叠现象，证明了在存在类别不平衡情况下，神经折叠仍然存在，但类均值的几何特性会发生偏移。 |
| [^12] | [DPZero: Dimension-Independent and Differentially Private Zeroth-Order Optimization.](http://arxiv.org/abs/2310.09639) | 该论文提出了DPZero算法，这是一种与维度无关且具有差分隐私的零阶优化算法，用于解决在细调大型语言模型时面临的内存和隐私挑战。 |
| [^13] | [Model-Free Algorithm with Improved Sample Efficiency for Zero-Sum Markov Games.](http://arxiv.org/abs/2308.08858) | 本文提出了一种模型自由的算法，可以在零和马尔可夫博弈中实现与模型为基础算法相同的样本复杂度，首次证明了模型自由算法可以在时间段依赖性方面达到同样的优化效果。 |
| [^14] | [Guaranteed Optimal Generative Modeling with Maximum Deviation from the Empirical Distribution.](http://arxiv.org/abs/2307.16422) | 本文提供了一种理论方法来训练生成模型，确保其在样本大小趋近无穷时与真实数据生成分布的误差收敛为零，并且远离复制训练数据中示例的任何分布。 |
| [^15] | [Degree Heterogeneity in Higher-Order Networks: Inference in the Hypergraph $\boldsymbol{\beta}$-Model.](http://arxiv.org/abs/2307.02818) | 本文对具有多层的超图β模型进行了研究，推导了最大似然估计的收敛速率和极限分布，并构建了模型参数的置信区间。同时，我们还建立了超图β模型中似然比检验的渐近正态性。 |
| [^16] | [Provably Efficient Bayesian Optimization with Unbiased Gaussian Process Hyperparameter Estimation.](http://arxiv.org/abs/2306.06844) | 针对贝叶斯优化中常见的数据偏差问题，提出了一种新方法，在无需事先知道真实高斯过程超参数的情况下，使用多臂老虎机技术向BO过程中添加随机数据点，采用新的训练损失函数进行超参数估计，以达到次线性收敛到全局最优解的目的。 |
| [^17] | [Catapults in SGD: spikes in the training loss and their impact on generalization through feature learning.](http://arxiv.org/abs/2306.04815) | 本文通过研究SGD训练损失中的尖峰现象，提出了“投石机”优化现象，通过增加与真实预测器的平均梯度外积对齐来促进特征学习，并证明较小的批量大小可提高泛化性能。 |
| [^18] | [Variational Gradient Descent using Local Linear Models.](http://arxiv.org/abs/2305.15577) | 本文提出了使用局部线性模型实现目标和粒子分布KL散度降低的新估计器，可以使用样本进行计算而不需要目标得分函数，具有比SVGD更简单有效的计算方法，对于高维度情况下的模型也有优化，提升估计精度。 |
| [^19] | [Augmented balancing weights as linear regression.](http://arxiv.org/abs/2304.14545) | 本文探索了自动去偏重重配的新颖特征描述，并将其等同于基于内核岭回归的单个欠平滑岭回归，进一步将这种方法推广到特定的结果和重配模型选择上。 |
| [^20] | [Towards Minimax Optimality of Model-based Robust Reinforcement Learning.](http://arxiv.org/abs/2302.05372) | 本文研究了在鲁棒强化学习中，对于仅具有对正常核心的生成模型访问权限时，获得ε-最优策略的样本复杂度。对于sa（s-）矩形不确定集合，已知最佳样本复杂度为ε^2/（H^4 * |S|^2 * |A|）（响应为ε^2/（H^4 * |S|^2 * |A|^2）），对于特定算法和基于总变差（TV）、KL或卡方散度的不确定集合。 |
| [^21] | [Bayesian sequential design of computer experiments for quantile set inversion.](http://arxiv.org/abs/2211.01008) | 本论文提出了一种基于贝叶斯策略的量化集反演方法，通过高斯过程建模和逐步不确定性减少原理，顺序选择评估函数的点，从而有效近似感兴趣的集合。 |
| [^22] | [Event-Triggered Time-Varying Bayesian Optimization.](http://arxiv.org/abs/2208.10790) | 本文提出了一种事件触发算法ET-GP-UCB，用于解决时变贝叶斯优化中的探索和开发的权衡问题。通过基于高斯过程回归的概率均匀误差界，算法能够在未知变化速率的情况下自适应地适应实际的时间变化。数值实验结果表明，ET-GP-UCB在合成和实际数据上优于现有算法。 |

# 详细

[^1]: 关于洗牌梯度方法的最后迭代收敛性

    On the Last-Iterate Convergence of Shuffling Gradient Methods

    [https://arxiv.org/abs/2403.07723](https://arxiv.org/abs/2403.07723)

    该论文证明了针对目标函数的洗牌梯度方法最后迭代的收敛速率，弥合了在不同设置中最后迭代的良好性能与现有理论之间的差距。

    

    洗牌梯度方法，也被称为无替换的随机梯度下降（SGD），在实践中被广泛应用，特别包括三种流行算法：Random Reshuffle（RR）、Shuffle Once（SO）和Incremental Gradient（IG）。与经验成功相比，长期以来对于洗牌梯度方法的理论保证并不充分了解。最近，只为凸函数的平均迭代和强凸问题的最后迭代（以平方距离为度量）建立了收敛速率。然而，当将函数值差作为收敛准则时，现有理论无法解释在不同设置中（例如受约束的优化）最后迭代的良好性能。为了弥合这种实践与理论之间的差距，我们针对目标函数证明了洗牌梯度方法最后迭代的收敛速率。

    arXiv:2403.07723v1 Announce Type: new  Abstract: Shuffling gradient methods, which are also known as stochastic gradient descent (SGD) without replacement, are widely implemented in practice, particularly including three popular algorithms: Random Reshuffle (RR), Shuffle Once (SO), and Incremental Gradient (IG). Compared to the empirical success, the theoretical guarantee of shuffling gradient methods was not well-understanding for a long time. Until recently, the convergence rates had just been established for the average iterate for convex functions and the last iterate for strongly convex problems (using squared distance as the metric). However, when using the function value gap as the convergence criterion, existing theories cannot interpret the good performance of the last iterate in different settings (e.g., constrained optimization). To bridge this gap between practice and theory, we prove last-iterate convergence rates for shuffling gradient methods with respect to the objectiv
    
[^2]: 变分学习对大型深度网络有效

    Variational Learning is Effective for Large Deep Networks

    [https://arxiv.org/abs/2402.17641](https://arxiv.org/abs/2402.17641)

    变分学习在大型深度网络中展现出非常好的效果，IVON优化器在训练大型网络时几乎能与Adam相媲美甚至胜过它，且预测不确定性更准确，对模型微调、泛化误差预测和数据敏感性估计均有显著改进。

    

    我们提供了大量实证证据，反驳了变分学习对大型神经网络无效的普遍看法。我们展示了一种名为Improved Variational Online Newton (IVON)的优化器，在训练大型网络（如GPT-2和ResNets）时始终能够与Adam相匹配或胜过它。IVON的计算成本几乎与Adam相同，但其预测不确定性更好。我们展示了IVON的几种新用例，其中我们改进了大型语言模型的微调和模型合并，在准确预测泛化误差和忠实估计对数据的敏感性方面。我们找到了大量支持变分学习有效性的证据。

    arXiv:2402.17641v1 Announce Type: cross  Abstract: We give extensive empirical evidence against the common belief that variational learning is ineffective for large neural networks. We show that an optimizer called Improved Variational Online Newton (IVON) consistently matches or outperforms Adam for training large networks such as GPT-2 and ResNets from scratch. IVON's computational costs are nearly identical to Adam but its predictive uncertainty is better. We show several new use cases of IVON where we improve fine-tuning and model merging in Large Language Models, accurately predict generalization error, and faithfully estimate sensitivity to data. We find overwhelming evidence in support of effectiveness of variational learning.
    
[^3]: 分类深度学习：一种关于架构的代数理论

    Categorical Deep Learning: An Algebraic Theory of Architectures

    [https://arxiv.org/abs/2402.15332](https://arxiv.org/abs/2402.15332)

    提出了一种关于深度学习架构的代数理论，应用范畴论构建了一个桥梁，有效地涵盖了神经网络设计的不同风格，同时自然地编码了计算机科学和自动机理论中的许多标准结构。

    

    我们提出了一个关于指定和研究深度学习架构的通用框架的立场。我们认为到目前为止关于这一领域的关键尝试缺乏一种一致的桥梁，能够指定模型必须满足的约束并规定它们的实现方式。专注于构建这样一个桥梁，我们建议应用范畴论——准确地说，单子值于参数映射的二范畴的通用代数——作为一种单一理论，优雅地包含了神经网络设计的这两种风格。为了支持我们的观点，我们展示了这一理论如何恢复由几何深度学习导致的约束，以及从神经网络不同领域的多种架构（如RNNs）的实现。我们还展示了这一理论如何自然地编码了计算机科学和自动机理论中的许多标准结构。

    arXiv:2402.15332v1 Announce Type: cross  Abstract: We present our position on the elusive quest for a general-purpose framework for specifying and studying deep learning architectures. Our opinion is that the key attempts made so far lack a coherent bridge between specifying constraints which models must satisfy and specifying their implementations. Focusing on building a such a bridge, we propose to apply category theory -- precisely, the universal algebra of monads valued in a 2-category of parametric maps -- as a single theory elegantly subsuming both of these flavours of neural network design. To defend our position, we show how this theory recovers constraints induced by geometric deep learning, as well as implementations of many architectures drawn from the diverse landscape of neural networks, such as RNNs. We also illustrate how the theory naturally encodes many standard constructs in computer science and automata theory.
    
[^4]: 使用Equilibrium K-Means进行不平衡数据聚类

    Imbalanced Data Clustering using Equilibrium K-Means

    [https://arxiv.org/abs/2402.14490](https://arxiv.org/abs/2402.14490)

    Equilibrium K-Means（EKM）是一种新颖且简单的K均值类型算法，通过减少聚类中心在大类簇中心聚集的倾向，显著改善了不平衡数据的聚类结果。

    

    不平衡数据指的是数据点在不同类别之间分布不均衡，这给传统的硬聚类算法和模糊聚类算法（如硬K均值（HKM，或者Lloyd算法）和模糊K均值（FKM，或者Bezdek算法））带来了挑战。本文介绍了一种新颖且简单的K均值类型算法——Equilibrium K-Means（EKM），它在两个步骤之间交替进行，显著改善了不平衡数据的聚类结果，减少了聚类中心向大类簇中心聚集的倾向。我们还提出了对HKM、FKM和EKM的统一视角，表明它们本质上是具有明确关系的牛顿方法的梯度下降算法。EKM具有与FKM相同的时间和空间复杂度，但对其成员定义提供了更清晰的物理意义。我们在两个合成数据集和十个真实数据集上展示了EKM的性能，并将其与各种聚类算法进行了比较。

    arXiv:2402.14490v1 Announce Type: new  Abstract: Imbalanced data, characterized by an unequal distribution of data points across different clusters, poses a challenge for traditional hard and fuzzy clustering algorithms, such as hard K-means (HKM, or Lloyd's algorithm) and fuzzy K-means (FKM, or Bezdek's algorithm). This paper introduces equilibrium K-means (EKM), a novel and simple K-means-type algorithm that alternates between just two steps, yielding significantly improved clustering results for imbalanced data by reducing the tendency of centroids to crowd together in the center of large clusters. We also present a unifying perspective for HKM, FKM, and EKM, showing they are essentially gradient descent algorithms with an explicit relationship to Newton's method. EKM has the same time and space complexity as FKM but offers a clearer physical meaning for its membership definition. We illustrate the performance of EKM on two synthetic and ten real datasets, comparing it to various cl
    
[^5]: 通过风险分解实现严格适当评分规则的预测不确定性量化

    Predictive Uncertainty Quantification via Risk Decompositions for Strictly Proper Scoring Rules

    [https://arxiv.org/abs/2402.10727](https://arxiv.org/abs/2402.10727)

    通过引入风险分解和适当评分规则，我们提出了一个通用框架来量化预测不确定性的不同来源，并澄清了它们之间的关系。

    

    在各个领域的预测模型应用中，区分预测不确定性的来源至关重要。尽管提出了许多不确定性度量，但并没有严格的定义来解开它们。此外，不同不确定性量化措施之间的关系仍然有些不清晰。在这项工作中，我们引入了一个根植于统计推理的通用框架，不仅允许创建新的不确定性度量，还澄清了它们之间的相互关系。我们的方法利用统计风险来区分aleatoric和epistemic不确定性成分，并利用适当的评分规则对其进行量化。为了使其在实践中易于处理，我们提出了在这一框架中整合贝叶斯推理的想法，并讨论了所提近似的性质。

    arXiv:2402.10727v1 Announce Type: cross  Abstract: Distinguishing sources of predictive uncertainty is of crucial importance in the application of forecasting models across various domains. Despite the presence of a great variety of proposed uncertainty measures, there are no strict definitions to disentangle them. Furthermore, the relationship between different measures of uncertainty quantification remains somewhat unclear. In this work, we introduce a general framework, rooted in statistical reasoning, which not only allows the creation of new uncertainty measures but also clarifies their interrelations. Our approach leverages statistical risk to distinguish aleatoric and epistemic uncertainty components and utilizes proper scoring rules to quantify them. To make it practically tractable, we propose an idea to incorporate Bayesian reasoning into this framework and discuss the properties of the proposed approximation.
    
[^6]: 滚动扩散模型

    Rolling Diffusion Models

    [https://arxiv.org/abs/2402.09470](https://arxiv.org/abs/2402.09470)

    本文介绍了一种滚动扩散模型，用于处理时间数据，通过滑动窗口去噪并根据帧在序列中的时间先后分配不同的噪声量，更好地捕捉到复杂的时间动态。通过实验证明，在视频预测和混沌流体动力学预测任务中，该模型优于传统扩散方法。

    

    最近，扩散模型越来越多地应用于时间数据，如视频、流体力学模拟或气候数据。这些方法通常将后续帧在扩散过程中的噪声量视为相等。本文探讨了滚动扩散：一种使用滑动窗口去噪的新方法。它确保扩散过程逐渐通过时间进行破坏，通过将更多的噪声分配给序列中出现较晚的帧，反映出随着生成过程的展开，对未来的不确定性越来越大。通过实证研究，我们表明当时间动态复杂时，滚动扩散优于标准扩散。特别是在使用Kinetics-600视频数据集进行视频预测任务和混沌流体动力学预测实验中证明了这一结果。

    arXiv:2402.09470v1 Announce Type: new  Abstract: Diffusion models have recently been increasingly applied to temporal data such as video, fluid mechanics simulations, or climate data. These methods generally treat subsequent frames equally regarding the amount of noise in the diffusion process. This paper explores Rolling Diffusion: a new approach that uses a sliding window denoising process. It ensures that the diffusion process progressively corrupts through time by assigning more noise to frames that appear later in a sequence, reflecting greater uncertainty about the future as the generation process unfolds. Empirically, we show that when the temporal dynamics are complex, Rolling Diffusion is superior to standard diffusion. In particular, this result is demonstrated in a video prediction task using the Kinetics-600 video dataset and in a chaotic fluid dynamics forecasting experiment.
    
[^7]: 参数到可观测映射的算子学习视角

    An operator learning perspective on parameter-to-observable maps

    [https://arxiv.org/abs/2402.06031](https://arxiv.org/abs/2402.06031)

    本论文从算子学习的视角研究了参数到可观测映射，提出了适应有限维输入和输出的傅里叶神经映射框架，并发展了通用逼近定理来支持该方法。此外，讨论了学习PtO映射的端到端方法和先学习解算子再计算可观测值的效率问题。

    

    计算高效的参数化物理模型替代品在科学和工程中起着至关重要的作用。算子学习提供了一个数据驱动的替代方案，可以在函数空间中进行映射。然而，通常只有有限维的模型输入参数化或有限维的模型输出可观测数据可用，而不是全场测量数据。本文基于傅里叶神经算子，引入了傅里叶神经映射（Fourier Neural Mappings，FNMs）框架，能够适应这样的有限维输入和输出。本文为该方法发展了通用逼近定理。此外，在许多应用中，底层的参数到可观测（PtO）映射是通过无穷维算子来隐式定义的，例如偏微分方程的解算子。一个自然的问题是，是更有效地学习PtO映射的端到端方法，还是首先学习解算子，然后计算可观测值。

    Computationally efficient surrogates for parametrized physical models play a crucial role in science and engineering. Operator learning provides data-driven surrogates that map between function spaces. However, instead of full-field measurements, often the available data are only finite-dimensional parametrizations of model inputs or finite observables of model outputs. Building off of Fourier Neural Operators, this paper introduces the Fourier Neural Mappings (FNMs) framework that is able to accommodate such finite-dimensional inputs and outputs. The paper develops universal approximation theorems for the method. Moreover, in many applications the underlying parameter-to-observable (PtO) map is defined implicitly through an infinite-dimensional operator, such as the solution operator of a partial differential equation. A natural question is whether it is more data-efficient to learn the PtO map end-to-end or first learn the solution operator and subsequently compute the observable fro
    
[^8]: 在离散状态空间上的生成型流：实现多模态流并应用于蛋白质共设计

    Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design

    [https://arxiv.org/abs/2402.04997](https://arxiv.org/abs/2402.04997)

    本研究提出了离散流模型（DFMs），通过连续时间马尔可夫链实现离散空间流匹配的离散等效，为将基于流的生成模型应用于多模态连续和离散数据问题提供了解决方案。此方法在蛋白质共设计任务上取得了最先进的性能。

    

    将离散和连续数据相结合对于生成模型来说是一项重要的能力。我们提出了离散流模型（DFMs），这是一种新的基于流的离散数据模型，可以实现将基于流的生成模型应用于多模态连续和离散数据问题。我们的关键见解是，离散空间流匹配的离散等效可以通过使用连续时间马尔可夫链来实现。DFMs通过简单的推导包括离散扩散模型作为特定实例，同时允许在现有基于扩散的方法上改进性能。我们利用DFMs方法构建了一个多模态基于流的建模框架。我们将此能力应用于蛋白质共设计的任务，其中我们学习了一个能够同时生成蛋白质结构和序列的模型。我们的方法在共设计性能上达到了最先进水平，同时允许使用同一个多模态模型进行序列或结构的灵活生成。

    Combining discrete and continuous data is an important capability for generative models. We present Discrete Flow Models (DFMs), a new flow-based model of discrete data that provides the missing link in enabling flow-based generative models to be applied to multimodal continuous and discrete data problems. Our key insight is that the discrete equivalent of continuous space flow matching can be realized using Continuous Time Markov Chains. DFMs benefit from a simple derivation that includes discrete diffusion models as a specific instance while allowing improved performance over existing diffusion-based approaches. We utilize our DFMs method to build a multimodal flow-based modeling framework. We apply this capability to the task of protein co-design, wherein we learn a model for jointly generating protein structure and sequence. Our approach achieves state-of-the-art co-design performance while allowing the same multimodal model to be used for flexible generation of the sequence or str
    
[^9]: 低多线性秩张量逼近的随机矩阵方法

    A Random Matrix Approach to Low-Multilinear-Rank Tensor Approximation

    [https://arxiv.org/abs/2402.03169](https://arxiv.org/abs/2402.03169)

    该研究采用随机矩阵方法，在低多线性秩张量逼近中展示了对种植的低秩信号的估计，并根据大维谱行为和信噪比准确预测了重建性能，并给出了HOOI收敛的充分条件。

    

    本研究从计算阈值附近的一般尖峰张量模型，对种植的低秩信号估计进行了全面的认识。依靠大型随机矩阵理论的标准工具，我们表征了数据张量的展开的大维谱行为，并展示了决定主要信号方向可检测性的相关信噪比。这些结果可以准确地预测在非平凡区域的截断多线性奇异值分解(MLSVD)的重建性能。这一点尤其重要，因为它作为更高阶正交迭代(HOOI)方案的初始化，其收敛到最佳低多线性秩逼近完全取决于其初始化。我们给出了HOOI收敛的充分条件，并证明在大维极限下收敛前的迭代次数趋于1。

    This work presents a comprehensive understanding of the estimation of a planted low-rank signal from a general spiked tensor model near the computational threshold. Relying on standard tools from the theory of large random matrices, we characterize the large-dimensional spectral behavior of the unfoldings of the data tensor and exhibit relevant signal-to-noise ratios governing the detectability of the principal directions of the signal. These results allow to accurately predict the reconstruction performance of truncated multilinear SVD (MLSVD) in the non-trivial regime. This is particularly important since it serves as an initialization of the higher-order orthogonal iteration (HOOI) scheme, whose convergence to the best low-multilinear-rank approximation depends entirely on its initialization. We give a sufficient condition for the convergence of HOOI and show that the number of iterations before convergence tends to $1$ in the large-dimensional limit.
    
[^10]: 从高阶统计量中高效学习：假设检验、随机特征和神经网络

    Learning from higher-order statistics, efficiently: hypothesis tests, random features, and neural networks

    [https://arxiv.org/abs/2312.14922](https://arxiv.org/abs/2312.14922)

    神经网络在高维数据中发现统计模式，研究了如何高效地从高阶累积量中提取特征，并探讨了在尖峰累积量模型中的统计和计算限制。

    

    神经网络擅长发现高维数据集中的统计模式。在实践中，度量三个或更多变量间的非高斯相关性的高阶累积量对神经网络的性能特别重要。但神经网络有多有效地从高阶累积量中提取特征？我们在尖峰累积量模型中探讨了这个问题，这里统计学家需要从$d$维输入的阶-$p\ge 4$累积量中恢复出一个特权方向或“尖峰”。我们首先通过分析所需样本数$n$来表征恢复尖峰的基本统计和计算限制，以强烈区分来自尖峰累积量模型和各向同性高斯输入的输入。我们发现，统计上的可区分性需要$n\gtrsim d$个样本，而在多项式时间内区分这两个分布则需要

    arXiv:2312.14922v2 Announce Type: replace-cross  Abstract: Neural networks excel at discovering statistical patterns in high-dimensional data sets. In practice, higher-order cumulants, which quantify the non-Gaussian correlations between three or more variables, are particularly important for the performance of neural networks. But how efficient are neural networks at extracting features from higher-order cumulants? We study this question in the spiked cumulant model, where the statistician needs to recover a privileged direction or "spike" from the order-$p\ge 4$ cumulants of $d$-dimensional inputs. We first characterise the fundamental statistical and computational limits of recovering the spike by analysing the number of samples $n$ required to strongly distinguish between inputs from the spiked cumulant model and isotropic Gaussian inputs. We find that statistical distinguishability requires $n\gtrsim d$ samples, while distinguishing the two distributions in polynomial time require
    
[^11]: 使用无约束ReLU特征模型进行交叉熵类别不平衡学习的神经折叠

    Neural Collapse for Cross-entropy Class-Imbalanced Learning with Unconstrained ReLU Feature Model. (arXiv:2401.02058v1 [cs.LG])

    [http://arxiv.org/abs/2401.02058](http://arxiv.org/abs/2401.02058)

    本文研究了在交叉熵类别不平衡学习中的神经折叠现象，证明了在存在类别不平衡情况下，神经折叠仍然存在，但类均值的几何特性会发生偏移。

    

    目前训练深度神经网络进行分类任务的范式包括最小化经验风险，将训练损失值推向零，即使训练误差已经消失。在训练的最后阶段，观察到最后一层特征会折叠到它们的类均值，并且这些类均值会收敛到一个简单典型等角紧框（ETF）的顶点。这一现象被称为神经折叠（NC）。为了从理论上理解这一现象，最近的研究使用了一个简化的无约束特征模型来证明NC出现在训练问题的全局解中。然而，当训练数据集存在类别不平衡时，一些NC特性将不再成立。例如，当损失收敛时，类均值几何会偏离简单典型等角紧框。在本文中，我们将NC推广到不平衡情况下的交叉熵损失和无约束ReLU特征模型。我们证明，在训练问题的全局解中，当存在类别不平衡时，NC仍然存在，但对于类均值的几何特性会发生偏移。

    The current paradigm of training deep neural networks for classification tasks includes minimizing the empirical risk that pushes the training loss value towards zero, even after the training error has been vanished. In this terminal phase of training, it has been observed that the last-layer features collapse to their class-means and these class-means converge to the vertices of a simplex Equiangular Tight Frame (ETF). This phenomenon is termed as Neural Collapse (NC). To theoretically understand this phenomenon, recent works employ a simplified unconstrained feature model to prove that NC emerges at the global solutions of the training problem. However, when the training dataset is class-imbalanced, some NC properties will no longer be true. For example, the class-means geometry will skew away from the simplex ETF when the loss converges. In this paper, we generalize NC to imbalanced regime for cross-entropy loss under the unconstrained ReLU feature model. We prove that, while the wi
    
[^12]: DPZero：与维度无关且具有差分隐私的零阶优化算法

    DPZero: Dimension-Independent and Differentially Private Zeroth-Order Optimization. (arXiv:2310.09639v1 [cs.LG])

    [http://arxiv.org/abs/2310.09639](http://arxiv.org/abs/2310.09639)

    该论文提出了DPZero算法，这是一种与维度无关且具有差分隐私的零阶优化算法，用于解决在细调大型语言模型时面临的内存和隐私挑战。

    

    在细调预训练的大型语言模型（LLM）以适应特定领域数据的广泛实践中，面临着内存和隐私两个主要挑战。首先，随着LLM的规模不断增长，达到数十亿个参数，基于梯度的反向传播训练方法所需的内存消耗变得难以承受。其次，考虑到LLM倾向于记忆和泄露敏感的训练数据，必须保护细调数据的隐私。为此，我们探索了将零阶方法与差分隐私优化相结合用于LLM的细调的潜力。零阶方法仅依赖前向传递，大大减少了训练过程中的内存消耗。然而，直接将它们与标准的差分隐私机制结合在一起会导致维度相关的复杂性。为了弥合这一差距，我们引入了DPZero，一种具有近乎维度无关率的新型差分隐私零阶算法。我们的理论分析揭示出了

    The widespread practice of fine-tuning pretrained large language models (LLMs) on domain-specific data faces two major challenges in memory and privacy. First, as the size of LLMs continue to grow, encompassing billions of parameters, the memory demands of gradient-based training methods via backpropagation become prohibitively high. Second, given the tendency of LLMs to memorize and disclose sensitive training data, the privacy of fine-tuning data must be respected. To this end, we explore the potential of zeroth-order methods in differentially private optimization for fine-tuning LLMs. Zeroth-order methods, which rely solely on forward passes, substantially reduce memory consumption during training. However, directly combining them with standard differential privacy mechanism poses dimension-dependent complexity. To bridge the gap, we introduce DPZero, a novel differentially private zeroth-order algorithm with nearly dimension-independent rates. Our theoretical analysis reveals that 
    
[^13]: 没有模型的算法在零和马尔可夫博弈中提高了样本效率

    Model-Free Algorithm with Improved Sample Efficiency for Zero-Sum Markov Games. (arXiv:2308.08858v1 [cs.LG])

    [http://arxiv.org/abs/2308.08858](http://arxiv.org/abs/2308.08858)

    本文提出了一种模型自由的算法，可以在零和马尔可夫博弈中实现与模型为基础算法相同的样本复杂度，首次证明了模型自由算法可以在时间段依赖性方面达到同样的优化效果。

    

    最近，两人零和马尔可夫博弈问题在多智能体强化学习的理论研究中引起了越来越多的兴趣。特别是对于有限时间段的马尔可夫决策过程，已经证明了模型为基础的算法可以通过样本复杂度为$O(H^3SAB/\epsilon^2)$找到$\epsilon$-最优的纳什均衡（NE），其中$H$是时间段，$S$是状态数量（$A$和$B$分别表示两个玩家的动作数量）。然而，目前没有一种现有的模型自由算法可以达到这样的优化效果。在这项工作中，我们提出了一种模型自由的阶段性Q学习算法，并展示它实现了与最佳模型为基础算法相同的样本复杂度，因此首次证明了模型自由算法可以在时间段依赖性方面享受与模型为基础算法相同的优化效果。对于$H$的依赖性的主要改进来源于...

    The problem of two-player zero-sum Markov games has recently attracted increasing interests in theoretical studies of multi-agent reinforcement learning (RL). In particular, for finite-horizon episodic Markov decision processes (MDPs), it has been shown that model-based algorithms can find an $\epsilon$-optimal Nash Equilibrium (NE) with the sample complexity of $O(H^3SAB/\epsilon^2)$, which is optimal in the dependence of the horizon $H$ and the number of states $S$ (where $A$ and $B$ denote the number of actions of the two players, respectively). However, none of the existing model-free algorithms can achieve such an optimality. In this work, we propose a model-free stage-based Q-learning algorithm and show that it achieves the same sample complexity as the best model-based algorithm, and hence for the first time demonstrate that model-free algorithms can enjoy the same optimality in the $H$ dependence as model-based algorithms. The main improvement of the dependency on $H$ arises by
    
[^14]: 保证从经验分布中最大偏差的最佳生成建模

    Guaranteed Optimal Generative Modeling with Maximum Deviation from the Empirical Distribution. (arXiv:2307.16422v1 [math.ST])

    [http://arxiv.org/abs/2307.16422](http://arxiv.org/abs/2307.16422)

    本文提供了一种理论方法来训练生成模型，确保其在样本大小趋近无穷时与真实数据生成分布的误差收敛为零，并且远离复制训练数据中示例的任何分布。

    

    生成建模是一种广泛应用于科学和工业领域的机器学习方法。其主要目标是在给定训练数据的情况下，模拟从未知分布中抽取的新示例，同时确保多样性并避免从训练数据中复制示例。本文提出了关于训练生成模型的理论见解，该模型具有两个属性：（i）将真实数据生成分布与训练数据生成分布替换的误差在样本大小趋近无穷时应最佳收敛于零；（ii）训练数据生成分布应远离复制训练数据中示例的任何分布。我们提供了以有限样本风险界为形式的非渐近结果，量化了这些属性，并取决于相关参数，如样本大小、环境空间的维数和潜空间的维数。我们的结果适用于生成模型的各种应用情况。

    Generative modeling is a widely-used machine learning method with various applications in scientific and industrial fields. Its primary objective is to simulate new examples drawn from an unknown distribution given training data while ensuring diversity and avoiding replication of examples from the training data.  This paper presents theoretical insights into training a generative model with two properties: (i) the error of replacing the true data-generating distribution with the trained data-generating distribution should optimally converge to zero as the sample size approaches infinity, and (ii) the trained data-generating distribution should be far enough from any distribution replicating examples in the training data.  We provide non-asymptotic results in the form of finite sample risk bounds that quantify these properties and depend on relevant parameters such as sample size, the dimension of the ambient space, and the dimension of the latent space. Our results are applicable to g
    
[^15]: 高阶网络中的度异质性：超图β模型的推断

    Degree Heterogeneity in Higher-Order Networks: Inference in the Hypergraph $\boldsymbol{\beta}$-Model. (arXiv:2307.02818v1 [math.ST])

    [http://arxiv.org/abs/2307.02818](http://arxiv.org/abs/2307.02818)

    本文对具有多层的超图β模型进行了研究，推导了最大似然估计的收敛速率和极限分布，并构建了模型参数的置信区间。同时，我们还建立了超图β模型中似然比检验的渐近正态性。

    

    随机图中的β模型通常用于表示具有度异质性的网络中的配对交互。超图β模型超越了配对交互，Stasi等人于2014年引入了超图β模型，用于捕捉具有高阶（多向）交互的网络中的度异质性。本文首次对具有多层的超图β模型进行了严格研究，它允许在不同层次中存在不同大小的超边。首先，我们推导了最大似然（ML）估计的收敛速率，并确定了它们的最小极小速率。我们还推导了ML估计的极限分布，并构建了模型参数的渐近有效置信区间。接下来，我们考虑了超图β模型中的拟合优度问题。具体而言，我们在零假设下建立了似然比（LR）检验的渐近正态性。

    The $\boldsymbol{\beta}$-model for random graphs is commonly used for representing pairwise interactions in a network with degree heterogeneity. Going beyond pairwise interactions, Stasi et al. (2014) introduced the hypergraph $\boldsymbol{\beta}$-model for capturing degree heterogeneity in networks with higher-order (multi-way) interactions. In this paper we initiate the rigorous study of the hypergraph $\boldsymbol{\beta}$-model with multiple layers, which allows for hyperedges of different sizes across the layers. To begin with, we derive the rates of convergence of the maximum likelihood (ML) estimate and establish their minimax rate optimality. We also derive the limiting distribution of the ML estimate and construct asymptotically valid confidence intervals for the model parameters. Next, we consider the goodness-of-fit problem in the hypergraph $\boldsymbol{\beta}$-model. Specifically, we establish the asymptotic normality of the likelihood ratio (LR) test under the null hypothe
    
[^16]: 具有无偏高斯过程超参数估计的可证明高效贝叶斯优化

    Provably Efficient Bayesian Optimization with Unbiased Gaussian Process Hyperparameter Estimation. (arXiv:2306.06844v1 [stat.ML])

    [http://arxiv.org/abs/2306.06844](http://arxiv.org/abs/2306.06844)

    针对贝叶斯优化中常见的数据偏差问题，提出了一种新方法，在无需事先知道真实高斯过程超参数的情况下，使用多臂老虎机技术向BO过程中添加随机数据点，采用新的训练损失函数进行超参数估计，以达到次线性收敛到全局最优解的目的。

    

    基于高斯过程的贝叶斯优化是一种有效优化黑盒函数的方法。该方法的实际性能和理论保证，取决于正确估计高斯过程超参数值。但在实践中，由于常用于贝叶斯优化的数据采样策略可能会引起数据偏差，从而导致超参数估计错误。为了解决这个问题，我们提出了一种新的贝叶斯优化方法，即使在事先不知道真实高斯过程超参数并需要从观察数据中进行估计时，该方法也能够次线性收敛到目标函数的全局最优解。我们的方法使用多臂老虎机技术(EXP3)向BO过程中添加随机数据点，并使用新的训练损失函数用于高斯过程超参数估计过程的训练。

    Gaussian process (GP) based Bayesian optimization (BO) is a powerful method for optimizing black-box functions efficiently. The practical performance and theoretical guarantees associated with this approach depend on having the correct GP hyperparameter values, which are usually unknown in advance and need to be estimated from the observed data. However, in practice, these estimations could be incorrect due to biased data sampling strategies commonly used in BO. This can lead to degraded performance and break the sub-linear global convergence guarantee of BO. To address this issue, we propose a new BO method that can sub-linearly converge to the global optimum of the objective function even when the true GP hyperparameters are unknown in advance and need to be estimated from the observed data. Our method uses a multi-armed bandit technique (EXP3) to add random data points to the BO process, and employs a novel training loss function for the GP hyperparameter estimation process that ens
    
[^17]: SGD中的投石机：训练损失中的尖峰及其通过特征学习对泛化的影响

    Catapults in SGD: spikes in the training loss and their impact on generalization through feature learning. (arXiv:2306.04815v1 [cs.LG])

    [http://arxiv.org/abs/2306.04815](http://arxiv.org/abs/2306.04815)

    本文通过研究SGD训练损失中的尖峰现象，提出了“投石机”优化现象，通过增加与真实预测器的平均梯度外积对齐来促进特征学习，并证明较小的批量大小可提高泛化性能。

    

    本文首先解释了神经网络在使用随机梯度下降（SGD）进行训练时为什么经常出现训练损失尖峰的现象。我们提供了证据表明，SGD训练损失中的尖峰是“投石机”，这是一种优化现象，最初在[Lewkowycz等人，2020年]的大学习率GD中观察到。我们通过实验证明这些投石机出现在由正切内核的前几个特征向量所张成的低维子空间中，适用于GD和SGD。其次，我们提出了一个解释，即投石机如何通过增加与真实预测器的平均梯度外积（AGOP）对齐来促进特征学习，从而实现更好的泛化。此外，我们证明，在SGD中，更小的批量大小会导致更多的投石机出现，从而提高AGOP对齐和测试性能。

    In this paper, we first present an explanation regarding the common occurrence of spikes in the training loss when neural networks are trained with stochastic gradient descent (SGD). We provide evidence that the spikes in the training loss of SGD are "catapults", an optimization phenomenon originally observed in GD with large learning rates in [Lewkowycz et al. 2020]. We empirically show that these catapults occur in a low-dimensional subspace spanned by the top eigenvectors of the tangent kernel, for both GD and SGD. Second, we posit an explanation for how catapults lead to better generalization by demonstrating that catapults promote feature learning by increasing alignment with the Average Gradient Outer Product (AGOP) of the true predictor. Furthermore, we demonstrate that a smaller batch size in SGD induces a larger number of catapults, thereby improving AGOP alignment and test performance.
    
[^18]: 采用局部线性模型的变分梯度下降

    Variational Gradient Descent using Local Linear Models. (arXiv:2305.15577v1 [stat.ML])

    [http://arxiv.org/abs/2305.15577](http://arxiv.org/abs/2305.15577)

    本文提出了使用局部线性模型实现目标和粒子分布KL散度降低的新估计器，可以使用样本进行计算而不需要目标得分函数，具有比SVGD更简单有效的计算方法，对于高维度情况下的模型也有优化，提升估计精度。

    

    Stein Variational Gradient Descent (SVGD) 能够沿着轨迹传输粒子，从而减少目标和粒子分布之间的KL散度，但需要目标得分函数来计算更新。我们提出了一种新的SVGD视角，将其视为反向KL梯度流的局部估计器。这种视角启发我们提出了使用局部线性模型来实现相同目的的新估计器。这些提议的估计器可以仅使用目标和粒子分布的样本进行计算，而不需要目标得分函数。我们提议的变分梯度估计器利用了局部线性模型，从而在保持估计偏差与SVGD相当的效果的同时具有计算简便性。此外，我们证明，在温和的假设下，高维梯度流的估计可以转化为一个低维估计问题，从而导致更好的估计精度。我们对提议的方法进行了验证，并对其进行了比较。

    Stein Variational Gradient Descent (SVGD) can transport particles along trajectories that reduce the KL divergence between the target and particle distribution but requires the target score function to compute the update. We introduce a new perspective on SVGD that views it as a local estimator of the reversed KL gradient flow. This perspective inspires us to propose new estimators that use local linear models to achieve the same purpose. The proposed estimators can be computed using only samples from the target and particle distribution without needing the target score function. Our proposed variational gradient estimators utilize local linear models, resulting in computational simplicity while maintaining effectiveness comparable to SVGD in terms of estimation biases. Additionally, we demonstrate that under a mild assumption, the estimation of high-dimensional gradient flow can be translated into a lower-dimensional estimation problem, leading to improved estimation accuracy. We vali
    
[^19]: 自动去偏重重配作为线性回归

    Augmented balancing weights as linear regression. (arXiv:2304.14545v1 [stat.ME])

    [http://arxiv.org/abs/2304.14545](http://arxiv.org/abs/2304.14545)

    本文探索了自动去偏重重配的新颖特征描述，并将其等同于基于内核岭回归的单个欠平滑岭回归，进一步将这种方法推广到特定的结果和重配模型选择上。

    

    我们提供了对于自动去偏重重配(AutoDML)的新颖特征描述。这些估算器将结果建模与重配相结合，直接估计反向倾向积分权重。当结果与权重模型都是某些（可能是无限的）基础中的线性时，我们表明增强的估算器等同于具有将原始结果模型系数和OLS相结合的系数的单个线性模型；在许多设置中，增强估算器合并为仅使用OLS. 然后，我们将这些结果扩展到特定的结果和重配模型选择上。我们首先表明，使用(内核)岭回归作为结果和重配模型的联合估算器等同于单个、欠平滑(内核)岭回归；当考虑到渐近速率时，这一结果也成立。当代替权重模型为套索回归时，我们给出了特殊情况的解析表达式并且演示了…

    We provide a novel characterization of augmented balancing weights, also known as Automatic Debiased Machine Learning (AutoDML). These estimators combine outcome modeling with balancing weights, which estimate inverse propensity score weights directly. When the outcome and weighting models are both linear in some (possibly infinite) basis, we show that the augmented estimator is equivalent to a single linear model with coefficients that combine the original outcome model coefficients and OLS; in many settings, the augmented estimator collapses to OLS alone. We then extend these results to specific choices of outcome and weighting models. We first show that the combined estimator that uses (kernel) ridge regression for both outcome and weighting models is equivalent to a single, undersmoothed (kernel) ridge regression; this also holds when considering asymptotic rates. When the weighting model is instead lasso regression, we give closed-form expressions for special cases and demonstrate
    
[^20]: 迈向模型基础鲁棒强化学习的最小最大优化

    Towards Minimax Optimality of Model-based Robust Reinforcement Learning. (arXiv:2302.05372v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.05372](http://arxiv.org/abs/2302.05372)

    本文研究了在鲁棒强化学习中，对于仅具有对正常核心的生成模型访问权限时，获得ε-最优策略的样本复杂度。对于sa（s-）矩形不确定集合，已知最佳样本复杂度为ε^2/（H^4 * |S|^2 * |A|）（响应为ε^2/（H^4 * |S|^2 * |A|^2）），对于特定算法和基于总变差（TV）、KL或卡方散度的不确定集合。

    

    我们研究了在只有对正常核心的生成模型访问权限时，获得ε-最优策略的采样复杂度。这个问题在非鲁棒情况下已经得到了广泛研究，并且已知任何应用于经验MDP的规划方法，只需要用ε^2/（H^3 * |S| * |A|）个样本来估计，均可提供ε-最优策略，从而最小最大优化。鲁棒情况下的结果更加少见。对于sa（s-）矩形不确定集合，已知最佳样本复杂度为ε^2/（H^4 * |S|^2 * |A|）（响应为ε^2/（H^4 * |S|^2 * |A|^2）），对于特定算法和基于总变差（TV）、KL或卡方散度的不确定集合。在本文中，我们考虑用Lp球定义的不确定集合（回复到TV情况），并且...

    We study the sample complexity of obtaining an $\epsilon$-optimal policy in \emph{Robust} discounted Markov Decision Processes (RMDPs), given only access to a generative model of the nominal kernel. This problem is widely studied in the non-robust case, and it is known that any planning approach applied to an empirical MDP estimated with $\tilde{\mathcal{O}}(\frac{H^3 \mid S \mid\mid A \mid}{\epsilon^2})$ samples provides an $\epsilon$-optimal policy, which is minimax optimal. Results in the robust case are much more scarce. For $sa$(resp $s$-)rectangular uncertainty sets, the best known sample complexity is $\tilde{\mathcal{O}}(\frac{H^4 \mid S \mid^2\mid A \mid}{\epsilon^2})$ (resp. $\tilde{\mathcal{O}}(\frac{H^4 \mid S \mid^2\mid A \mid^2}{\epsilon^2})$), for specific algorithms and when the uncertainty set is based on the total variation (TV), the KL or the Chi-square divergences. In this paper, we consider uncertainty sets defined with an $L_p$-ball (recovering the TV case), and
    
[^21]: 基于贝叶斯序贯设计的计算机实验量化集反演

    Bayesian sequential design of computer experiments for quantile set inversion. (arXiv:2211.01008v2 [stat.ML] CROSS LISTED)

    [http://arxiv.org/abs/2211.01008](http://arxiv.org/abs/2211.01008)

    本论文提出了一种基于贝叶斯策略的量化集反演方法，通过高斯过程建模和逐步不确定性减少原理，顺序选择评估函数的点，从而有效近似感兴趣的集合。

    

    我们考虑一个未知的多元函数，它代表着一个系统，如一个复杂的数值模拟器，同时具有确定性和不确定性的输入。我们的目标是估计确定性输入集，这些输入导致的输出（就不确定性输入的分布而言）属于给定集合的概率小于给定阈值。这个问题被称为量化集反演（QSI），例如在稳健（基于可靠性）优化问题的背景下，当寻找满足约束条件且具有足够大概率的解集时会发生。为了解决QSI问题，我们提出了一种基于高斯过程建模和逐步不确定性减少（SUR）原理的贝叶斯策略，以顺序选择应该评估函数的点，以便高效近似感兴趣的集合。通过几个数值实验，我们展示了所提出的SUR策略的性能和价值

    We consider an unknown multivariate function representing a system-such as a complex numerical simulator-taking both deterministic and uncertain inputs. Our objective is to estimate the set of deterministic inputs leading to outputs whose probability (with respect to the distribution of the uncertain inputs) of belonging to a given set is less than a given threshold. This problem, which we call Quantile Set Inversion (QSI), occurs for instance in the context of robust (reliability-based) optimization problems, when looking for the set of solutions that satisfy the constraints with sufficiently large probability. To solve the QSI problem, we propose a Bayesian strategy based on Gaussian process modeling and the Stepwise Uncertainty Reduction (SUR) principle, to sequentially choose the points at which the function should be evaluated to efficiently approximate the set of interest. We illustrate the performance and interest of the proposed SUR strategy through several numerical experiment
    
[^22]: 事件触发的时变贝叶斯优化

    Event-Triggered Time-Varying Bayesian Optimization. (arXiv:2208.10790v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.10790](http://arxiv.org/abs/2208.10790)

    本文提出了一种事件触发算法ET-GP-UCB，用于解决时变贝叶斯优化中的探索和开发的权衡问题。通过基于高斯过程回归的概率均匀误差界，算法能够在未知变化速率的情况下自适应地适应实际的时间变化。数值实验结果表明，ET-GP-UCB在合成和实际数据上优于现有算法。

    

    我们考虑使用时变贝叶斯优化（TVBO）顺序优化时变目标函数的问题。其中，关键挑战是在时间变化下的勘探与开发的权衡。当前的TVBO方法需要对变化速率有先验知识。然而，在实践中，变化速率通常是未知的。我们提出了一种事件触发算法ET-GP-UCB，它将优化问题视为静态问题，直到在线检测到目标函数的变化并重置数据集。这使得算法能够适应实际的时间变化，而不需要先验知识。事件触发基于高斯过程回归中使用的概率均匀误差界。我们给出了ET-GP-UCB的遗憾界，并通过数值实验表明，它在合成和实际数据上优于现有算法。此外，这些结果表明ET-GP-UCB可广泛应用于不同的设定。

    We consider the problem of sequentially optimizing a time-varying objective function using time-varying Bayesian optimization (TVBO). Here, the key challenge is the exploration-exploitation trade-off under time variations. Current approaches to TVBO require prior knowledge of a constant rate of change. However, in practice, the rate of change is usually unknown. We propose an event-triggered algorithm, ET-GP-UCB, that treats the optimization problem as static until it detects changes in the objective function online and then resets the dataset. This allows the algorithm to adapt to realized temporal changes without the need for prior knowledge. The event-trigger is based on probabilistic uniform error bounds used in Gaussian process regression. We provide regret bounds for ET-GP-UCB and show in numerical experiments that it outperforms state-of-the-art algorithms on synthetic and real-world data. Furthermore, these results demonstrate that ET-GP-UCB is readily applicable to various set
    

