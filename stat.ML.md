# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Library of Mirrors: Deep Neural Nets in Low Dimensions are Convex Lasso Models with Reflection Features](https://arxiv.org/abs/2403.01046) | 证明在1-D数据上训练神经网络等价于解决一个具有固定特征字典矩阵的凸Lasso问题，为全局最优网络和解空间提供了洞察。 |
| [^2] | [Learning Best-in-Class Policies for the Predict-then-Optimize Framework](https://arxiv.org/abs/2402.03256) | 我们提出了一种新颖的决策感知替代损失函数家族，用于predict-then-optimize框架，并且通过数值证据证实了其在误设置下的优越性。 |
| [^3] | [When Does Bottom-up Beat Top-down in Hierarchical Community Detection?.](http://arxiv.org/abs/2306.00833) | 本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。 |
| [^4] | [Vector-Valued Variation Spaces and Width Bounds for DNNs: Insights on Weight Decay Regularization.](http://arxiv.org/abs/2305.16534) | 该论文提供了关于通过加权衰减训练的多输出ReLU神经网络的函数类型和相应的解决方案的新见解。 |
| [^5] | [Physics-informed Information Field Theory for Modeling Physical Systems with Uncertainty Quantification.](http://arxiv.org/abs/2301.07609) | 该论文扩展了信息场理论(IFT)到物理信息场理论(PIFT)，将描述场的物理定律的信息编码为函数先验。从这个PIFT得出的后验与任何数值方案无关，并且可以捕捉多种模式。 |
| [^6] | [$\Phi$-DVAE: Physics-Informed Dynamical Variational Autoencoders for Unstructured Data Assimilation.](http://arxiv.org/abs/2209.15609) | 本文提出了一种物理指导的动态变分自编码器 ($\Phi$-DVAE) 用于将非结构化数据同化到物理模型中，解决了传统方法在未知映射情况下无法实现一致模型与数据综合的问题。 |
| [^7] | [Targeted Adaptive Design.](http://arxiv.org/abs/2205.14208) | TAD是一种新的目标自适应设计算法，可以通过高斯过程代理模型在指定公差中确定产生期望设计特征的最佳控制设置，相比其他自适应设计算法，具有更高的精度和效率。 |

# 详细

[^1]: 一个镜子的库：低维深度神经网络是具有反射特征的凸Lasso模型

    A Library of Mirrors: Deep Neural Nets in Low Dimensions are Convex Lasso Models with Reflection Features

    [https://arxiv.org/abs/2403.01046](https://arxiv.org/abs/2403.01046)

    证明在1-D数据上训练神经网络等价于解决一个具有固定特征字典矩阵的凸Lasso问题，为全局最优网络和解空间提供了洞察。

    

    我们证明在1-D数据上训练神经网络等价于解决一个带有固定、明确定义的特征字典矩阵的凸Lasso问题。具体的字典取决于激活函数和深度。我们考虑具有分段线性激活函数的两层网络，深窄的ReLU网络最多有4层，以及具有符号激活和任意深度的矩形和树网络。有趣的是，在ReLU网络中，第四层创建代表训练数据关于自身的反射的特征。Lasso表示法揭示了全局最优网络和解空间的洞察。

    arXiv:2403.01046v1 Announce Type: cross  Abstract: We prove that training neural networks on 1-D data is equivalent to solving a convex Lasso problem with a fixed, explicitly defined dictionary matrix of features. The specific dictionary depends on the activation and depth. We consider 2-layer networks with piecewise linear activations, deep narrow ReLU networks with up to 4 layers, and rectangular and tree networks with sign activation and arbitrary depth. Interestingly in ReLU networks, a fourth layer creates features that represent reflections of training data about themselves. The Lasso representation sheds insight to globally optimal networks and the solution landscape.
    
[^2]: 学习Predict-then-Optimize框架中的最优策略

    Learning Best-in-Class Policies for the Predict-then-Optimize Framework

    [https://arxiv.org/abs/2402.03256](https://arxiv.org/abs/2402.03256)

    我们提出了一种新颖的决策感知替代损失函数家族，用于predict-then-optimize框架，并且通过数值证据证实了其在误设置下的优越性。

    

    我们提出了一种新颖的决策感知替代损失函数家族，称为Perturbation Gradient（PG）损失，用于predict-then-optimize框架。这些损失直接近似了下游决策损失，并可以使用现成的基于梯度的方法进行优化。重要的是，与现有的替代损失不同，我们的PG损失的近似误差随着样本数量的增加而消失。这意味着优化我们的替代损失可以在渐近意义下得到最佳策略，即使在误设置下也是如此。这是第一个在误设置下的这样的结果，我们提供了数值证据证实了当基础模型误设置且噪声不是中心对称时，我们的PG损失在实践中显著优于现有的提案。鉴于在实践中误设置很常见--特别是当我们可能更喜欢一个更简单、更可解释的模型时--PG损失提供了一种新颖的、理论上有依据的、可计算的决策感知方法。

    We propose a novel family of decision-aware surrogate losses, called Perturbation Gradient (PG) losses, for the predict-then-optimize framework. These losses directly approximate the downstream decision loss and can be optimized using off-the-shelf gradient-based methods. Importantly, unlike existing surrogate losses, the approximation error of our PG losses vanishes as the number of samples grows. This implies that optimizing our surrogate loss yields a best-in-class policy asymptotically, even in misspecified settings. This is the first such result in misspecified settings and we provide numerical evidence confirming our PG losses substantively outperform existing proposals when the underlying model is misspecified and the noise is not centrally symmetric. Insofar as misspecification is commonplace in practice -- especially when we might prefer a simpler, more interpretable model -- PG losses offer a novel, theoretically justified, method for computationally tractable decision-aware 
    
[^3]: 自下而上何时击败自上而下进行分层社区检测？

    When Does Bottom-up Beat Top-down in Hierarchical Community Detection?. (arXiv:2306.00833v1 [cs.SI])

    [http://arxiv.org/abs/2306.00833](http://arxiv.org/abs/2306.00833)

    本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。

    

    网络的分层聚类是指查找一组社区的树形结构，其中层次结构的较低级别显示更细粒度的社区结构。解决这一问题的算法有两个主要类别：自上而下的算法和自下而上的算法。本文研究了使用自下而上算法恢复分层随机块模型的树形结构和社区结构的理论保证。我们还确定了这种自下而上算法在层次结构的中间层次上达到了确切恢复信息理论阈值。值得注意的是，这些恢复条件相对于现有的自上而下算法的条件来说，限制更少。

    Hierarchical clustering of networks consists in finding a tree of communities, such that lower levels of the hierarchy reveal finer-grained community structures. There are two main classes of algorithms tackling this problem. Divisive ($\textit{top-down}$) algorithms recursively partition the nodes into two communities, until a stopping rule indicates that no further split is needed. In contrast, agglomerative ($\textit{bottom-up}$) algorithms first identify the smallest community structure and then repeatedly merge the communities using a $\textit{linkage}$ method. In this article, we establish theoretical guarantees for the recovery of the hierarchical tree and community structure of a Hierarchical Stochastic Block Model by a bottom-up algorithm. We also establish that this bottom-up algorithm attains the information-theoretic threshold for exact recovery at intermediate levels of the hierarchy. Notably, these recovery conditions are less restrictive compared to those existing for to
    
[^4]: 向量值变分空间和DNN的宽度界：关于权重衰减正则化的见解。

    Vector-Valued Variation Spaces and Width Bounds for DNNs: Insights on Weight Decay Regularization. (arXiv:2305.16534v1 [stat.ML])

    [http://arxiv.org/abs/2305.16534](http://arxiv.org/abs/2305.16534)

    该论文提供了关于通过加权衰减训练的多输出ReLU神经网络的函数类型和相应的解决方案的新见解。

    

    深度神经网络(DNNs)通过梯度下降最小化损失项和平方权重和相应，对应于训练加权衰减的常见方法。本文提供了有关这种常见学习框架的新见解。我们表征了训练加权衰减以获得多输出(向量值)ReLU神经网络学习的函数类型。这扩展了先前限于单输出(标量值)网络的表征。这种表征需要定义我们称之为向量值变分(VV)空间的新类神经函数空间。我们通过一种新的表征定理证明，神经网络(NNs)是通过VV空间中提出学习问题的最优解。这个新的表征定理表明，这些学习问题的解存在于宽度受训练数据数限制的向量值神经网络中。接下来，通过与多任务lasso问题的新联系，我们导出了

    Deep neural networks (DNNs) trained to minimize a loss term plus the sum of squared weights via gradient descent corresponds to the common approach of training with weight decay. This paper provides new insights into this common learning framework. We characterize the kinds of functions learned by training with weight decay for multi-output (vector-valued) ReLU neural networks. This extends previous characterizations that were limited to single-output (scalar-valued) networks. This characterization requires the definition of a new class of neural function spaces that we call vector-valued variation (VV) spaces. We prove that neural networks (NNs) are optimal solutions to learning problems posed over VV spaces via a novel representer theorem. This new representer theorem shows that solutions to these learning problems exist as vector-valued neural networks with widths bounded in terms of the number of training data. Next, via a novel connection to the multi-task lasso problem, we derive
    
[^5]: 物理学知识作为不确定性量化模型的信息场理论

    Physics-informed Information Field Theory for Modeling Physical Systems with Uncertainty Quantification. (arXiv:2301.07609v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.07609](http://arxiv.org/abs/2301.07609)

    该论文扩展了信息场理论(IFT)到物理信息场理论(PIFT)，将描述场的物理定律的信息编码为函数先验。从这个PIFT得出的后验与任何数值方案无关，并且可以捕捉多种模式。

    

    数据驱动的方法结合物理学知识是建模系统的强有力技术。此类模型的目标是通过将测量结果与已知物理定律相结合，高效地求解基本场。由于许多系统包含未知元素，如缺失参数、嘈杂数据或不完整的物理定律，因此这通常被视为一种不确定性量化问题。处理所有变量的常见技术通常取决于用于近似后验的数值方案，并且希望有一种不依赖于任何离散化的方法。信息场理论（IFT）提供了对不一定是高斯场的场进行统计学的工具。我们通过将描述场的物理定律的信息编码为函数先验来扩展IFT到物理信息场理论（PIFT）。从这个PIFT得出的后验与任何数值方案无关，并且可以捕捉多种模式。

    Data-driven approaches coupled with physical knowledge are powerful techniques to model systems. The goal of such models is to efficiently solve for the underlying field by combining measurements with known physical laws. As many systems contain unknown elements, such as missing parameters, noisy data, or incomplete physical laws, this is widely approached as an uncertainty quantification problem. The common techniques to handle all the variables typically depend on the numerical scheme used to approximate the posterior, and it is desirable to have a method which is independent of any such discretization. Information field theory (IFT) provides the tools necessary to perform statistics over fields that are not necessarily Gaussian. We extend IFT to physics-informed IFT (PIFT) by encoding the functional priors with information about the physical laws which describe the field. The posteriors derived from this PIFT remain independent of any numerical scheme and can capture multiple modes,
    
[^6]: $\Phi$-DVAE: 物理指导的动态变分自编码器用于非结构化数据同化

    $\Phi$-DVAE: Physics-Informed Dynamical Variational Autoencoders for Unstructured Data Assimilation. (arXiv:2209.15609v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.15609](http://arxiv.org/abs/2209.15609)

    本文提出了一种物理指导的动态变分自编码器 ($\Phi$-DVAE) 用于将非结构化数据同化到物理模型中，解决了传统方法在未知映射情况下无法实现一致模型与数据综合的问题。

    

    在数据同化中，将非结构化数据纳入物理模型是一个具有挑战性的问题。传统方法通常关注具有明确定义观测算子的情况，其函数形式通常被假定为已知。这阻止了这些方法在从数据空间到模型空间的映射未知的配置中实现一致的模型与数据综合。为了解决这些问题，在本文中我们开发了一种物理指导的动态变分自编码器($\Phi$-DVAE)，将多样化的数据流嵌入到由微分方程描述的时变物理系统中。我们的方法结合了一个标准的、可能是非线性的潜在状态空间模型滤波器和一个变分自编码器，将非结构化数据同化到潜在的动态系统中。在我们的示例系统中，非结构化数据采用视频数据和速度场测量的形式，但该方法的适用性足够通用，可以允许任意未知的观测算子。

    Incorporating unstructured data into physical models is a challenging problem that is emerging in data assimilation. Traditional approaches focus on well-defined observation operators whose functional forms are typically assumed to be known. This prevents these methods from achieving a consistent model-data synthesis in configurations where the mapping from data-space to model-space is unknown. To address these shortcomings, in this paper we develop a physics-informed dynamical variational autoencoder ($\Phi$-DVAE) to embed diverse data streams into time-evolving physical systems described by differential equations. Our approach combines a standard, possibly nonlinear, filter for the latent state-space model and a VAE, to assimilate the unstructured data into the latent dynamical system. Unstructured data, in our example systems, comes in the form of video data and velocity field measurements, however the methodology is suitably generic to allow for arbitrary unknown observation operat
    
[^7]: 目标自适应设计

    Targeted Adaptive Design. (arXiv:2205.14208v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.14208](http://arxiv.org/abs/2205.14208)

    TAD是一种新的目标自适应设计算法，可以通过高斯过程代理模型在指定公差中确定产生期望设计特征的最佳控制设置，相比其他自适应设计算法，具有更高的精度和效率。

    

    现代先进制造和高级材料设计往往需要在较高维度的过程控制参数空间中搜索最佳结构、性能和性能参数的设置。从前者到后者的映射必须通过嘈杂的实验或昂贵的模拟来确定。我们把这个问题抽象成一个数学框架，其中必须通过昂贵的嘈杂测量来确定从控制空间到设计空间的未知函数，该函数在指定的公差范围内定位产生期望设计特征的最佳控制设置，并量化不确定性。我们描述了目标自适应设计 (TAD)，这是一种有效执行这个采样任务的新算法。TAD 在每个迭代阶段创建一个未知映射的高斯过程代理模型，建议一批新的控制设置进行实验采样，并优化更新的目标设计特征的对数预测似然。

    Modern advanced manufacturing and advanced materials design often require searches of relatively high-dimensional process control parameter spaces for settings that result in optimal structure, property, and performance parameters. The mapping from the former to the latter must be determined from noisy experiments or from expensive simulations. We abstract this problem to a mathematical framework in which an unknown function from a control space to a design space must be ascertained by means of expensive noisy measurements, which locate optimal control settings generating desired design features within specified tolerances, with quantified uncertainty. We describe targeted adaptive design (TAD), a new algorithm that performs this sampling task efficiently. TAD creates a Gaussian process surrogate model of the unknown mapping at each iterative stage, proposing a new batch of control settings to sample experimentally and optimizing the updated log-predictive likelihood of the target desi
    

