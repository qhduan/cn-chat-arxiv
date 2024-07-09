# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Space-Time Bridge-Diffusion](https://arxiv.org/abs/2402.08847) | 介绍了一种利用时空混合策略生成独立同分布合成样本的方法，并通过线性和非线性随机过程实现最佳转运，进一步细化通过分数匹配技术训练方法 |
| [^2] | [PAC-Bayesian Adversarially Robust Generalization Bounds for Graph Neural Network](https://arxiv.org/abs/2402.04038) | 本文提出了一种基于PAC-Bayesian框架的方法，来研究对抗鲁棒性泛化界限问题，针对两种流行的图神经网络模型进行了分析，结果发现图上扩散矩阵的谱范数、权重的谱范数和扰动因子对模型的鲁棒泛化界限有重要影响。 |
| [^3] | [Nonparametric Estimation via Variance-Reduced Sketching.](http://arxiv.org/abs/2401.11646) | 本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。 |
| [^4] | [Model-based causal feature selection for general response types.](http://arxiv.org/abs/2309.12833) | 本研究基于模型提出了一种通用响应类型的因果特征选择方法，该方法利用不变性假设从异质环境的数据中输出一部分因果特征的子集，适用于一般的加性噪声模型和非参数设置，解决了非参数条件独立性测试低功率的问题。 |
| [^5] | [Prominent Roles of Conditionally Invariant Components in Domain Adaptation: Theory and Algorithms.](http://arxiv.org/abs/2309.10301) | 该论文研究了领域自适应中条件不变组件的作用，提出了一种基于条件不变惩罚的新算法，该算法在目标风险保证方面具有优势。 |
| [^6] | [Graph topological property recovery with heat and wave dynamics-based features on graphsD.](http://arxiv.org/abs/2309.09924) | 本文提出了一种名为图微分方程网络（GDeNet）的方法，利用热和波动方程动力学特征来恢复图的拓扑属性，能够在各种下游任务中获得优秀的表现，同时在实际应用中也展现了较好的性能。 |
| [^7] | [Shadows of quantum machine learning.](http://arxiv.org/abs/2306.00061) | 量子机器学习模型需要使用量子计算机进行评估，但我们提出在训练完后，使用量子计算机生成一个经典阴影模型来计算函数的经典计算近似，避免了对量子计算机的需求。 |
| [^8] | [Deep Stochastic Mechanics.](http://arxiv.org/abs/2305.19685) | 本文提出了一种基于深度学习的方法，用于数值模拟时间演化薛定谔方程，利用马尔可夫扩散采样来适应波函数的潜在低维结构，并提出了新的随机量子力学方程，具有线性的计算复杂度。数值模拟显示出显着的优势。 |

# 详细

[^1]: 时空桥扩散方法

    Space-Time Bridge-Diffusion

    [https://arxiv.org/abs/2402.08847](https://arxiv.org/abs/2402.08847)

    介绍了一种利用时空混合策略生成独立同分布合成样本的方法，并通过线性和非线性随机过程实现最佳转运，进一步细化通过分数匹配技术训练方法

    

    在这项研究中，我们介绍了一种新的方法，用于从由一组地面真实样本（GT样本）隐式定义的高维实值概率分布中生成独立同分布（i.i.d.）的新合成样本。我们的方法的核心是通过时空混合策略在时间和空间维度上进行扩展。我们的方法基于三个相互关联的随机过程，旨在实现从容易处理的初始概率分布到由GT样本表示的目标分布的最佳转运：（a）包含时空混合的线性过程产生高斯条件概率密度，（b）其桥扩散模拟，条件为初始和最终状态向量，以及（c）通过分数匹配技术进行细化的非线性随机过程。我们训练方法的关键在于精调

    arXiv:2402.08847v1 Announce Type: cross Abstract: In this study, we introduce a novel method for generating new synthetic samples that are independent and identically distributed (i.i.d.) from high-dimensional real-valued probability distributions, as defined implicitly by a set of Ground Truth (GT) samples. Central to our method is the integration of space-time mixing strategies that extend across temporal and spatial dimensions. Our methodology is underpinned by three interrelated stochastic processes designed to enable optimal transport from an easily tractable initial probability distribution to the target distribution represented by the GT samples: (a) linear processes incorporating space-time mixing that yield Gaussian conditional probability densities, (b) their bridge-diffusion analogs that are conditioned to the initial and final state vectors, and (c) nonlinear stochastic processes refined through score-matching techniques. The crux of our training regime involves fine-tuning
    
[^2]: PAC-Bayesian Adversarially Robust Generalization Bounds for Graph Neural Network

    PAC-Bayesian Adversarially Robust Generalization Bounds for Graph Neural Network

    [https://arxiv.org/abs/2402.04038](https://arxiv.org/abs/2402.04038)

    本文提出了一种基于PAC-Bayesian框架的方法，来研究对抗鲁棒性泛化界限问题，针对两种流行的图神经网络模型进行了分析，结果发现图上扩散矩阵的谱范数、权重的谱范数和扰动因子对模型的鲁棒泛化界限有重要影响。

    

    图神经网络（GNNs）在各种与图相关的任务中广受欢迎。然而，类似于深度神经网络，GNNs也容易受到对抗攻击。经验研究表明，对抗鲁棒性泛化在建立有效的抵御对抗攻击的防御算法方面起着关键作用。本文通过使用PAC-Bayesian框架，为两种流行的GNNs，即图卷积网络（GCN）和消息传递图神经网络，提供了对抗鲁棒泛化界限。我们的结果揭示了图上扩散矩阵的谱范数、权重的谱范数以及扰动因子对两个模型的鲁棒泛化界限的影响。我们的界限是（Liao等人，2020）中结果的非平凡推广，从标准设置扩展到对抗设置，同时避免了最大节点度的指数依赖。作为推论，我们得出更好的界限...

    Graph neural networks (GNNs) have gained popularity for various graph-related tasks. However, similar to deep neural networks, GNNs are also vulnerable to adversarial attacks. Empirical studies have shown that adversarially robust generalization has a pivotal role in establishing effective defense algorithms against adversarial attacks. In this paper, we contribute by providing adversarially robust generalization bounds for two kinds of popular GNNs, graph convolutional network (GCN) and message passing graph neural network, using the PAC-Bayesian framework. Our result reveals that spectral norm of the diffusion matrix on the graph and spectral norm of the weights as well as the perturbation factor govern the robust generalization bounds of both models. Our bounds are nontrivial generalizations of the results developed in (Liao et al., 2020) from the standard setting to adversarial setting while avoiding exponential dependence of the maximum node degree. As corollaries, we derive bette
    
[^3]: 通过方差降低的草图进行非参数估计

    Nonparametric Estimation via Variance-Reduced Sketching. (arXiv:2401.11646v1 [stat.ML])

    [http://arxiv.org/abs/2401.11646](http://arxiv.org/abs/2401.11646)

    本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。

    

    非参数模型在各个科学和工程领域中备受关注。经典的核方法在低维情况下具有数值稳定性和统计可靠性，但在高维情况下由于维度灾难变得不够适用。在本文中，我们引入了一个名为Variance-Reduced Sketching（VRS）的新框架，专门用于在降低维度灾难的同时在高维度中估计密度函数和非参数回归函数。我们的框架将多变量函数概念化为无限大小的矩阵，并借鉴了数值线性代数文献中的一种新的草图技术来降低估计问题中的方差。我们通过一系列的模拟实验和真实数据应用展示了VRS的鲁棒性能。值得注意的是，在许多密度估计问题中，VRS相较于现有的神经网络估计器和经典的核方法表现出显著的改进。

    Nonparametric models are of great interest in various scientific and engineering disciplines. Classical kernel methods, while numerically robust and statistically sound in low-dimensional settings, become inadequate in higher-dimensional settings due to the curse of dimensionality. In this paper, we introduce a new framework called Variance-Reduced Sketching (VRS), specifically designed to estimate density functions and nonparametric regression functions in higher dimensions with a reduced curse of dimensionality. Our framework conceptualizes multivariable functions as infinite-size matrices, and facilitates a new sketching technique motivated by numerical linear algebra literature to reduce the variance in estimation problems. We demonstrate the robust numerical performance of VRS through a series of simulated experiments and real-world data applications. Notably, VRS shows remarkable improvement over existing neural network estimators and classical kernel methods in numerous density 
    
[^4]: 基于模型的通用响应类型因果特征选择

    Model-based causal feature selection for general response types. (arXiv:2309.12833v1 [stat.ME])

    [http://arxiv.org/abs/2309.12833](http://arxiv.org/abs/2309.12833)

    本研究基于模型提出了一种通用响应类型的因果特征选择方法，该方法利用不变性假设从异质环境的数据中输出一部分因果特征的子集，适用于一般的加性噪声模型和非参数设置，解决了非参数条件独立性测试低功率的问题。

    

    从观测数据中发现因果关系是一项基本而具有挑战性的任务。在某些应用中，仅学习给定响应变量的因果特征可能已经足够，而不是学习整个潜在的因果结构。不变因果预测（ICP）是一种用于因果特征选择的方法，需要来自异质环境的数据。ICP假设从直接原因生成响应的机制在所有环境中都相同，并利用这种不变性输出一部分因果特征的子集。ICP的框架已经扩展到一般的加性噪声模型和非参数设置，使用条件独立性测试。然而，非参数条件独立性测试经常受到低功率（或较差的类型I错误控制）的困扰，并且上述参数模型不适用于响应不是在连续刻度上测量的应用情况，而是反映了分类信息的情况。

    Discovering causal relationships from observational data is a fundamental yet challenging task. In some applications, it may suffice to learn the causal features of a given response variable, instead of learning the entire underlying causal structure. Invariant causal prediction (ICP, Peters et al., 2016) is a method for causal feature selection which requires data from heterogeneous settings. ICP assumes that the mechanism for generating the response from its direct causes is the same in all settings and exploits this invariance to output a subset of the causal features. The framework of ICP has been extended to general additive noise models and to nonparametric settings using conditional independence testing. However, nonparametric conditional independence testing often suffers from low power (or poor type I error control) and the aforementioned parametric models are not suitable for applications in which the response is not measured on a continuous scale, but rather reflects categor
    
[^5]: 领域自适应中条件不变组件的突出作用：理论和算法

    Prominent Roles of Conditionally Invariant Components in Domain Adaptation: Theory and Algorithms. (arXiv:2309.10301v1 [stat.ML])

    [http://arxiv.org/abs/2309.10301](http://arxiv.org/abs/2309.10301)

    该论文研究了领域自适应中条件不变组件的作用，提出了一种基于条件不变惩罚的新算法，该算法在目标风险保证方面具有优势。

    

    领域自适应是一个统计学习问题，当用于训练模型的源数据分布与用于评估模型的目标数据分布不同时出现。虽然许多领域自适应算法已经证明了相当大的实证成功，但是盲目应用这些算法往往会导致在新的数据集上表现更差。为了解决这个问题，重要的是澄清领域自适应算法在具备良好目标性能的假设下。在这项工作中，我们关注在预测中具备条件不变的组件（CICs）的存在假设，这些组件在源数据和目标数据之间保持条件不变。我们证明了CICs，通过条件不变惩罚（CIP）可以估计，具备在领域自适应中提供目标风险保证的三个突出作用。首先，我们提出了一种基于CICs的新算法，即重要性加权的条件不变惩罚（IW-CIP），它在目标风险保证方面超越了简单的方法。

    Domain adaptation (DA) is a statistical learning problem that arises when the distribution of the source data used to train a model differs from that of the target data used to evaluate the model. While many DA algorithms have demonstrated considerable empirical success, blindly applying these algorithms can often lead to worse performance on new datasets. To address this, it is crucial to clarify the assumptions under which a DA algorithm has good target performance. In this work, we focus on the assumption of the presence of conditionally invariant components (CICs), which are relevant for prediction and remain conditionally invariant across the source and target data. We demonstrate that CICs, which can be estimated through conditional invariant penalty (CIP), play three prominent roles in providing target risk guarantees in DA. First, we propose a new algorithm based on CICs, importance-weighted conditional invariant penalty (IW-CIP), which has target risk guarantees beyond simple 
    
[^6]: 基于热和波动动力学特征的图拓扑属性恢复

    Graph topological property recovery with heat and wave dynamics-based features on graphsD. (arXiv:2309.09924v1 [cs.LG])

    [http://arxiv.org/abs/2309.09924](http://arxiv.org/abs/2309.09924)

    本文提出了一种名为图微分方程网络（GDeNet）的方法，利用热和波动方程动力学特征来恢复图的拓扑属性，能够在各种下游任务中获得优秀的表现，同时在实际应用中也展现了较好的性能。

    

    本文提出了一种名为图微分方程网络（GDeNet）的方法，利用图上的PDE解的表达能力，为各种下游任务获得连续的节点和图级表示。我们推导出了热和波动方程动力学与图的谱特性以及连续时间随机游走在图上行为之间的理论结果。我们通过恢复随机图生成参数、Ricci曲率和持久同调等方式实验证明了这些动力学能够捕捉到图形几何和拓扑的显著方面。此外，我们还展示了GDeNet在包括引用图、药物分子和蛋白质在内的真实世界数据集上的优越性能。

    In this paper, we propose Graph Differential Equation Network (GDeNet), an approach that harnesses the expressive power of solutions to PDEs on a graph to obtain continuous node- and graph-level representations for various downstream tasks. We derive theoretical results connecting the dynamics of heat and wave equations to the spectral properties of the graph and to the behavior of continuous-time random walks on graphs. We demonstrate experimentally that these dynamics are able to capture salient aspects of graph geometry and topology by recovering generating parameters of random graphs, Ricci curvature, and persistent homology. Furthermore, we demonstrate the superior performance of GDeNet on real-world datasets including citation graphs, drug-like molecules, and proteins.
    
[^7]: 量子机器学习的阴影

    Shadows of quantum machine learning. (arXiv:2306.00061v1 [quant-ph])

    [http://arxiv.org/abs/2306.00061](http://arxiv.org/abs/2306.00061)

    量子机器学习模型需要使用量子计算机进行评估，但我们提出在训练完后，使用量子计算机生成一个经典阴影模型来计算函数的经典计算近似，避免了对量子计算机的需求。

    

    量子机器学习经常被认为是利用量子计算机解决实际问题的最有前途的应用之一。然而，阻碍其在实践中广泛使用的主要障碍是这些模型即使在训练过程后，仍需要访问量子计算机才能对新数据进行评估。为解决这个问题，我们建议在量子模型的训练阶段之后，量子计算机可以用来生成我们所谓的该模型的“经典阴影”，即已学习函数的经典计算近似。虽然最近的研究已经探讨了这个想法并提出了构建这种影子模型的方法，但它们也提出了一个完全经典模型可能代替的可能性，从而首先回避了量子计算机的需要。本文采用新的方法，基于量子线性模型和经典阴影重构的框架来定义阴影模型。

    Quantum machine learning is often highlighted as one of the most promising uses for a quantum computer to solve practical problems. However, a major obstacle to the widespread use of quantum machine learning models in practice is that these models, even once trained, still require access to a quantum computer in order to be evaluated on new data. To solve this issue, we suggest that following the training phase of a quantum model, a quantum computer could be used to generate what we call a classical shadow of this model, i.e., a classically computable approximation of the learned function. While recent works already explore this idea and suggest approaches to construct such shadow models, they also raise the possibility that a completely classical model could be trained instead, thus circumventing the need for a quantum computer in the first place. In this work, we take a novel approach to define shadow models based on the frameworks of quantum linear models and classical shadow tomogr
    
[^8]: 深度随机力学

    Deep Stochastic Mechanics. (arXiv:2305.19685v1 [cs.LG])

    [http://arxiv.org/abs/2305.19685](http://arxiv.org/abs/2305.19685)

    本文提出了一种基于深度学习的方法，用于数值模拟时间演化薛定谔方程，利用马尔可夫扩散采样来适应波函数的潜在低维结构，并提出了新的随机量子力学方程，具有线性的计算复杂度。数值模拟显示出显着的优势。

    

    本文引入了一种基于深度学习的方法，用于数值模拟时间演化薛定谔方程，受随机力学和生成性扩散模型的启发。与现有方法不同的是，我们的方法允许我们通过从马尔可夫扩散中采样来适应波函数潜在的低维结构，因此可以在更高的维度上降低计算复杂度。此外，我们提出了新的随机量子力学方程，结果具有与维数数量线性的计算复杂度。数值模拟验证了我们的理论发现，并显示出我们的方法与其他用于量子力学的基于深度学习的方法相比具有显着优势。

    This paper introduces a novel deep-learning-based approach for numerical simulation of a time-evolving Schr\"odinger equation inspired by stochastic mechanics and generative diffusion models. Unlike existing approaches, which exhibit computational complexity that scales exponentially in the problem dimension, our method allows us to adapt to the latent low-dimensional structure of the wave function by sampling from the Markovian diffusion. Depending on the latent dimension, our method may have far lower computational complexity in higher dimensions. Moreover, we propose novel equations for stochastic quantum mechanics, resulting in linear computational complexity with respect to the number of dimensions. Numerical simulations verify our theoretical findings and show a significant advantage of our method compared to other deep-learning-based approaches used for quantum mechanics.
    

