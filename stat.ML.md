# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient geometric Markov chain Monte Carlo for nonlinear Bayesian inversion enabled by derivative-informed neural operators](https://arxiv.org/abs/2403.08220) | 运用导数信息的神经算子加速了几何马尔可夫链蒙特卡洛方法，显著加快了解决非线性贝叶斯反问题的过程。 |
| [^2] | [Differentially Private Fair Binary Classifications](https://arxiv.org/abs/2402.15603) | 该论文提出了一种差分隐私与公平性约束下的二元分类算法，通过解耦技术和差分隐私的引入，实现了在保证公平性的同时提升了隐私性能和效用保证。 |
| [^3] | [REMEDI: Corrective Transformations for Improved Neural Entropy Estimation](https://arxiv.org/abs/2402.05718) | REMEDI是一种用于改进神经熵估计的校正转换方法，通过交叉熵最小化和相对熵估计基模型的偏差，提高了估计任务的准确性和效率。 |
| [^4] | [Hypergraph Node Classification With Graph Neural Networks](https://arxiv.org/abs/2402.05569) | 本研究提出了一种简单高效的框架，利用加权子图扩展的图神经网络(WCE-GNN)实现了超图节点分类。实验证明，WCE-GNN具有优秀的预测效果和较低的计算复杂度。 |
| [^5] | [Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High-dimensional Gaussian Mixtures](https://arxiv.org/abs/2402.02697) | 本文通过对深度均衡模型和显式神经网络模型进行理论分析和实验证明，在高维高斯混合数据下，可以通过设计浅显式网络来实现与给定深度均衡模型相同的特征光谱行为。 |
| [^6] | [Rademacher Complexity of Neural ODEs via Chen-Fliess Series.](http://arxiv.org/abs/2401.16655) | 本文通过Chen-Fliess序列展开将连续深度神经ODE模型转化为单层、无限宽度的网络，并利用此框架推导出了将初始条件映射到某个终端时间的ODE模型的Rademacher复杂度的紧凑表达式。 |
| [^7] | [Bayesian Nonparametrics meets Data-Driven Robust Optimization.](http://arxiv.org/abs/2401.15771) | 本文提出了一种将贝叶斯非参数方法与最新的决策理论模型相结合的鲁棒优化准则，通过这种方法，可以在线性回归问题中获得有稳定性和优越性能的结果。 |
| [^8] | [A Model-Agnostic Graph Neural Network for Integrating Local and Global Information.](http://arxiv.org/abs/2309.13459) | MaGNet是一种模型无关的图神经网络框架，能够顺序地整合不同顺序的信息，并通过识别有影响力的紧凑图结构提供有意义且可解释的结果。 |
| [^9] | [On the sample complexity of estimation in logistic regression.](http://arxiv.org/abs/2307.04191) | 本文研究了逻辑回归模型在标准正态协变量下的参数估计样本复杂度，发现样本复杂度曲线在逆温度方面有两个转折点，明确划分了低、中和高温度区域。 |
| [^10] | [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models.](http://arxiv.org/abs/2305.19187) | 本研究提出应对大型语言模型可信度问题的方法，研究黑盒模型中置信度与不确定性量化，并将其应用于选择性自然语言生成。 |
| [^11] | [On the connections between optimization algorithms, Lyapunov functions, and differential equations: theory and insights.](http://arxiv.org/abs/2305.08658) | 本研究在推广线性矩阵不等式框架的基础上，研究了微分方程和优化算法的联系，提出了针对一个两参数Nesterov优化方法家族新的李亚普诺夫函数并表征其收敛速度，在此基础上证明了有显著改进的Nesterov方法的收敛速度，并确定出产生最佳速度的系数选择。 |
| [^12] | [An Analysis of Quantile Temporal-Difference Learning.](http://arxiv.org/abs/2301.04462) | 本文证明了量化时间差分学习（QTD）在一定状态下的收敛概率为1，建立了QTD与非线性微分包含式之间的联系。 |
| [^13] | [Blessings and Curses of Covariate Shifts: Adversarial Learning Dynamics, Directional Convergence, and Equilibria.](http://arxiv.org/abs/2212.02457) | 协变量转移和对抗扰动对统计学习的稳健性提出了挑战。本文在无限维度的情况下研究了对抗协变量转移对外推区域的影响以及其对后续学习的平衡的影响。 |
| [^14] | [Fair Active Learning: Solving the Labeling Problem in Insurance.](http://arxiv.org/abs/2112.09466) | 本文旨在解决保险行业中普遍存在的机器学习模型在数据中发现的偏见和歧视，提出了公平主动学习方法，能够在实现模型预测性能的同时保证数据公平性。 |

# 详细

[^1]: 非线性贝叶斯反问题的高效几何马尔可夫链蒙特卡洛方法：利用导数信息的神经算子

    Efficient geometric Markov chain Monte Carlo for nonlinear Bayesian inversion enabled by derivative-informed neural operators

    [https://arxiv.org/abs/2403.08220](https://arxiv.org/abs/2403.08220)

    运用导数信息的神经算子加速了几何马尔可夫链蒙特卡洛方法，显著加快了解决非线性贝叶斯反问题的过程。

    

    我们提出了一种运算学习方法来加速几何马尔可夫链蒙特卡洛（MCMC）以解决无限维非线性贝叶斯反问题。虽然几何MCMC采用适应后验局部几何的高质量提议，但在参数到可观测（PtO）映射通过昂贵的模型模拟定义时，需要计算对数似然的局部梯度和Hessian信息，造成高成本。我们考虑了一个由PtO映射的神经算子替代驱动的延迟接受几何马尔可夫链蒙特卡洛方法，其中提议被设计为利用对数似然和其梯度和Hessian的快速替代估计。为了实现显著加速，替代品需要准确预测可观测及其参数导数（可观测与参数之间的导数）。通过传统的方法对这样的替代品进行训练

    arXiv:2403.08220v1 Announce Type: cross  Abstract: We propose an operator learning approach to accelerate geometric Markov chain Monte Carlo (MCMC) for solving infinite-dimensional nonlinear Bayesian inverse problems. While geometric MCMC employs high-quality proposals that adapt to posterior local geometry, it requires computing local gradient and Hessian information of the log-likelihood, incurring a high cost when the parameter-to-observable (PtO) map is defined through expensive model simulations. We consider a delayed-acceptance geometric MCMC method driven by a neural operator surrogate of the PtO map, where the proposal is designed to exploit fast surrogate approximations of the log-likelihood and, simultaneously, its gradient and Hessian. To achieve a substantial speedup, the surrogate needs to be accurate in predicting both the observable and its parametric derivative (the derivative of the observable with respect to the parameter). Training such a surrogate via conventional o
    
[^2]: 差分隐私公平二元分类

    Differentially Private Fair Binary Classifications

    [https://arxiv.org/abs/2402.15603](https://arxiv.org/abs/2402.15603)

    该论文提出了一种差分隐私与公平性约束下的二元分类算法，通过解耦技术和差分隐私的引入，实现了在保证公平性的同时提升了隐私性能和效用保证。

    

    在本工作中，我们研究了在差分隐私和公平性约束下的二元分类。我们首先提出了一种基于解耦技术的算法，用于学习一个仅具有公平性保证的分类器。该算法接受针对不同人口群体训练的分类器，并生成一个满足统计平衡的单一分类器。然后，我们改进了该算法以纳入差分隐私。最终算法的性能在隐私、公平性和效用保证方面得到了严格检验。对Adult和信用卡数据集进行的实证评估表明，我们的算法在公平性保证方面优于现有技术，同时保持了相同水平的隐私和效用。

    arXiv:2402.15603v1 Announce Type: new  Abstract: In this work, we investigate binary classification under the constraints of both differential privacy and fairness. We first propose an algorithm based on the decoupling technique for learning a classifier with only fairness guarantee. This algorithm takes in classifiers trained on different demographic groups and generates a single classifier satisfying statistical parity. We then refine this algorithm to incorporate differential privacy. The performance of the final algorithm is rigorously examined in terms of privacy, fairness, and utility guarantees. Empirical evaluations conducted on the Adult and Credit Card datasets illustrate that our algorithm outperforms the state-of-the-art in terms of fairness guarantees, while maintaining the same level of privacy and utility.
    
[^3]: REMEDI: 改进神经熵估计的校正转换

    REMEDI: Corrective Transformations for Improved Neural Entropy Estimation

    [https://arxiv.org/abs/2402.05718](https://arxiv.org/abs/2402.05718)

    REMEDI是一种用于改进神经熵估计的校正转换方法，通过交叉熵最小化和相对熵估计基模型的偏差，提高了估计任务的准确性和效率。

    

    信息论量在机器学习中起着核心作用。数据和模型复杂性的增加使得准确估计这些量的需求增加。然而，随着维度的增加，估计存在重大挑战，现有方法在相对较低的维度中已经困难重重。为了解决这个问题，在这项工作中，我们引入了REMEDI，用于高效准确地估计微分熵，一种基本的信息论量。该方法结合了简单自适应基模型的交叉熵最小化和其相对熵从数据密度中估计的偏差。我们的方法在各种估计任务中得到了改进，包括对合成数据和自然数据的熵估计。此外，我们将重要的理论一致性结果扩展到我们方法所需的更广义的设置中。我们展示了我们的方法如何提高熵估计的准确性和效率。

    Information theoretic quantities play a central role in machine learning. The recent surge in the complexity of data and models has increased the demand for accurate estimation of these quantities. However, as the dimension grows the estimation presents significant challenges, with existing methods struggling already in relatively low dimensions. To address this issue, in this work, we introduce $\texttt{REMEDI}$ for efficient and accurate estimation of differential entropy, a fundamental information theoretic quantity. The approach combines the minimization of the cross-entropy for simple, adaptive base models and the estimation of their deviation, in terms of the relative entropy, from the data density. Our approach demonstrates improvement across a broad spectrum of estimation tasks, encompassing entropy estimation on both synthetic and natural data. Further, we extend important theoretical consistency results to a more generalized setting required by our approach. We illustrate how
    
[^4]: 使用图神经网络进行超图节点分类

    Hypergraph Node Classification With Graph Neural Networks

    [https://arxiv.org/abs/2402.05569](https://arxiv.org/abs/2402.05569)

    本研究提出了一种简单高效的框架，利用加权子图扩展的图神经网络(WCE-GNN)实现了超图节点分类。实验证明，WCE-GNN具有优秀的预测效果和较低的计算复杂度。

    

    超图是用来模拟现实世界数据中的高阶相互作用的关键。图神经网络（GNNs）的成功揭示了神经网络处理具有成对交互的数据的能力。这激发了使用神经网络处理具有高阶相互作用的数据的想法，从而导致了超图神经网络（HyperGNNs）的发展。GNNs和HyperGNNs通常被认为是不同的，因为它们被设计用于处理不同几何拓扑的数据。然而，在本文中，我们在理论上证明，在节点分类的上下文中，大多数HyperGNNs可以使用带有超图的加权子图扩展的GNN来近似。这导致了WCE-GNN，一种简单高效的框架，包括一个GNN和一个加权子图扩展（WCE），用于超图节点分类。对于九个真实世界的超图节点分类数据集的实验表明，WCE-GNN不仅具有优秀的预测效果，而且具有较低的计算复杂度。

    Hypergraphs, with hyperedges connecting more than two nodes, are key for modelling higher-order interactions in real-world data. The success of graph neural networks (GNNs) reveals the capability of neural networks to process data with pairwise interactions. This inspires the usage of neural networks for data with higher-order interactions, thereby leading to the development of hypergraph neural networks (HyperGNNs). GNNs and HyperGNNs are typically considered distinct since they are designed for data on different geometric topologies. However, in this paper, we theoretically demonstrate that, in the context of node classification, most HyperGNNs can be approximated using a GNN with a weighted clique expansion of the hypergraph. This leads to WCE-GNN, a simple and efficient framework comprising a GNN and a weighted clique expansion (WCE), for hypergraph node classification. Experiments on nine real-world hypergraph node classification benchmarks showcase that WCE-GNN demonstrates not o
    
[^5]: 深度均衡模型与高维高斯混合模型中不太深的显式模型几乎等价

    Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High-dimensional Gaussian Mixtures

    [https://arxiv.org/abs/2402.02697](https://arxiv.org/abs/2402.02697)

    本文通过对深度均衡模型和显式神经网络模型进行理论分析和实验证明，在高维高斯混合数据下，可以通过设计浅显式网络来实现与给定深度均衡模型相同的特征光谱行为。

    

    深度均衡模型（DEQs）作为典型的隐式神经网络，在各种任务上取得了显着的成功。然而，我们对隐式DEQ和显式神经网络模型之间的连接和差异缺乏理论上的理解。在本文中，我们借鉴最近在随机矩阵理论方面的进展，对高维高斯混合模型输入数据下，隐式DEQ的共轭核（CK）和神经切向核（NTK）矩阵的特征光谱进行了深入分析。我们在这个设置中证明了这些隐式-CKs和NTKs的光谱行为取决于DEQ激活函数和初始权重方差，但仅通过一组四个非线性方程。作为这一理论结果的直接影响，我们证明可以精心设计一个浅显式网络来产生与给定DEQ相同的CK或NTK。尽管这里是针对高斯混合数据推导的，经验结果表明

    Deep equilibrium models (DEQs), as a typical implicit neural network, have demonstrated remarkable success on various tasks. There is, however, a lack of theoretical understanding of the connections and differences between implicit DEQs and explicit neural network models. In this paper, leveraging recent advances in random matrix theory (RMT), we perform an in-depth analysis on the eigenspectra of the conjugate kernel (CK) and neural tangent kernel (NTK) matrices for implicit DEQs, when the input data are drawn from a high-dimensional Gaussian mixture. We prove, in this setting, that the spectral behavior of these Implicit-CKs and NTKs depend on the DEQ activation function and initial weight variances, but only via a system of four nonlinear equations. As a direct consequence of this theoretical result, we demonstrate that a shallow explicit network can be carefully designed to produce the same CK or NTK as a given DEQ. Despite derived here for Gaussian mixture data, empirical results 
    
[^6]: 通过Chen-Fliess序列，我们展示了如何将连续深度神经ODE模型构建为单层、无限宽度的网络。

    Rademacher Complexity of Neural ODEs via Chen-Fliess Series. (arXiv:2401.16655v1 [stat.ML])

    [http://arxiv.org/abs/2401.16655](http://arxiv.org/abs/2401.16655)

    本文通过Chen-Fliess序列展开将连续深度神经ODE模型转化为单层、无限宽度的网络，并利用此框架推导出了将初始条件映射到某个终端时间的ODE模型的Rademacher复杂度的紧凑表达式。

    

    本文将连续深度神经ODE模型使用Chen-Fliess序列展开为单层、无限宽度的网络。在这个网络中，输出的“权重”来自控制输入的特征序列，它由控制输入在单纯形上的迭代积分构成。而“特征”则基于受控ODE模型中输出函数相对于向量场的迭代李导数。本文的主要结果是，应用这个框架推导出了将初始条件映射到某个终端时间的ODE模型的Rademacher复杂度的紧凑表达式。这一结果利用了单层结构所带来的直接分析性质。最后，我们通过一些具体系统的例子实例化该界，并讨论了可能的后续工作。

    We show how continuous-depth neural ODE models can be framed as single-layer, infinite-width nets using the Chen--Fliess series expansion for nonlinear ODEs. In this net, the output ''weights'' are taken from the signature of the control input -- a tool used to represent infinite-dimensional paths as a sequence of tensors -- which comprises iterated integrals of the control input over a simplex. The ''features'' are taken to be iterated Lie derivatives of the output function with respect to the vector fields in the controlled ODE model. The main result of this work applies this framework to derive compact expressions for the Rademacher complexity of ODE models that map an initial condition to a scalar output at some terminal time. The result leverages the straightforward analysis afforded by single-layer architectures. We conclude with some examples instantiating the bound for some specific systems and discuss potential follow-up work.
    
[^7]: 贝叶斯非参数方法与数据驱动鲁棒优化的结合

    Bayesian Nonparametrics meets Data-Driven Robust Optimization. (arXiv:2401.15771v1 [stat.ML])

    [http://arxiv.org/abs/2401.15771](http://arxiv.org/abs/2401.15771)

    本文提出了一种将贝叶斯非参数方法与最新的决策理论模型相结合的鲁棒优化准则，通过这种方法，可以在线性回归问题中获得有稳定性和优越性能的结果。

    

    训练机器学习和统计模型通常涉及优化数据驱动的风险准则。风险通常是根据经验数据分布计算的，但由于分布不确定性，这可能导致性能不稳定和不好的样本外表现。在分布鲁棒优化的精神下，我们提出了一个新颖的鲁棒准则，将贝叶斯非参数（即狄利克雷过程）理论和最近的平滑模糊规避偏好的决策理论模型的见解相结合。首先，我们强调了与标准正则化经验风险最小化技术的新连接，其中包括岭回归和套索回归。然后，我们从理论上证明了鲁棒优化过程在有限样本和渐近统计保证方面的有利性存在。对于实际实施，我们提出并研究了基于众所周知的狄利克雷过程表示的可行近似准则。

    Training machine learning and statistical models often involves optimizing a data-driven risk criterion. The risk is usually computed with respect to the empirical data distribution, but this may result in poor and unstable out-of-sample performance due to distributional uncertainty. In the spirit of distributionally robust optimization, we propose a novel robust criterion by combining insights from Bayesian nonparametric (i.e., Dirichlet Process) theory and recent decision-theoretic models of smooth ambiguity-averse preferences. First, we highlight novel connections with standard regularized empirical risk minimization techniques, among which Ridge and LASSO regressions. Then, we theoretically demonstrate the existence of favorable finite-sample and asymptotic statistical guarantees on the performance of the robust optimization procedure. For practical implementation, we propose and study tractable approximations of the criterion based on well-known Dirichlet Process representations. 
    
[^8]: 模型无关的图神经网络用于整合局部和全局信息的研究

    A Model-Agnostic Graph Neural Network for Integrating Local and Global Information. (arXiv:2309.13459v1 [stat.ML])

    [http://arxiv.org/abs/2309.13459](http://arxiv.org/abs/2309.13459)

    MaGNet是一种模型无关的图神经网络框架，能够顺序地整合不同顺序的信息，并通过识别有影响力的紧凑图结构提供有意义且可解释的结果。

    

    图神经网络（GNNs）在各种以图为重点的任务中取得了令人满意的性能。尽管取得了成功，但现有的GNN存在两个重要限制：由于黑盒特性，结果缺乏可解释性；无法学习不同顺序的表示。为了解决这些问题，我们提出了一种新的模型无关的图神经网络（MaGNet）框架，能够顺序地整合不同顺序的信息，从高阶邻居中提取知识，并通过识别有影响力的紧凑图结构提供有意义且可解释的结果。特别地，MaGNet由两个组件组成：图拓扑下复杂关系的潜在表示的估计模型和识别有影响力的节点、边和重要节点特征的解释模型。从理论上，我们通过经验Rademacher复杂度建立了MaGNet的泛化误差界，并展示了其强大的能力。

    Graph Neural Networks (GNNs) have achieved promising performance in a variety of graph-focused tasks. Despite their success, existing GNNs suffer from two significant limitations: a lack of interpretability in results due to their black-box nature, and an inability to learn representations of varying orders. To tackle these issues, we propose a novel Model-agnostic Graph Neural Network (MaGNet) framework, which is able to sequentially integrate information of various orders, extract knowledge from high-order neighbors, and provide meaningful and interpretable results by identifying influential compact graph structures. In particular, MaGNet consists of two components: an estimation model for the latent representation of complex relationships under graph topology, and an interpretation model that identifies influential nodes, edges, and important node features. Theoretically, we establish the generalization error bound for MaGNet via empirical Rademacher complexity, and showcase its pow
    
[^9]: 关于逻辑回归中参数估计的样本复杂度研究

    On the sample complexity of estimation in logistic regression. (arXiv:2307.04191v1 [math.ST])

    [http://arxiv.org/abs/2307.04191](http://arxiv.org/abs/2307.04191)

    本文研究了逻辑回归模型在标准正态协变量下的参数估计样本复杂度，发现样本复杂度曲线在逆温度方面有两个转折点，明确划分了低、中和高温度区域。

    

    逻辑回归模型是噪声二元分类问题中最常见的数据生成模型之一。本文研究了在标准正态协变量下，以$\ell_2$误差为限，估计逻辑回归模型参数的样本复杂度，考虑了维度和逆温度的影响。逆温度控制了数据生成过程中的信噪比。虽然逻辑回归的广义界限和渐近性能已经有了深入研究，但关于参数估计的非渐近样本复杂度在之前的分析中没有讨论其与误差和逆温度的依赖关系。我们展示了样本复杂度曲线在逆温度方面具有两个转折点（或临界点），明确划分了低、中和高温度区域。

    The logistic regression model is one of the most popular data generation model in noisy binary classification problems. In this work, we study the sample complexity of estimating the parameters of the logistic regression model up to a given $\ell_2$ error, in terms of the dimension and the inverse temperature, with standard normal covariates. The inverse temperature controls the signal-to-noise ratio of the data generation process. While both generalization bounds and asymptotic performance of the maximum-likelihood estimator for logistic regression are well-studied, the non-asymptotic sample complexity that shows the dependence on error and the inverse temperature for parameter estimation is absent from previous analyses. We show that the sample complexity curve has two change-points (or critical points) in terms of the inverse temperature, clearly separating the low, moderate, and high temperature regimes.
    
[^10]: 生成可信的文本：大型语言模型的不确定性量化

    Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models. (arXiv:2305.19187v1 [cs.CL])

    [http://arxiv.org/abs/2305.19187](http://arxiv.org/abs/2305.19187)

    本研究提出应对大型语言模型可信度问题的方法，研究黑盒模型中置信度与不确定性量化，并将其应用于选择性自然语言生成。

    

    近期，专门用于自然语言生成的大型语言模型(LLMs)在各个领域表现出了很好的能力，但是评估LLMs生成的结果的可信度仍然是一个挑战，关于自然语言生成的不确定性量化的研究也较少。此外，现有的文献通常假定对语言模型的白盒访问，这要么是由于最新的LLMs的封闭源代码的性质，要么是由于计算限制。本文研究了黑盒LLMs的不确定性量化问题。我们首先区分了两种密切相关的概念: 只与输入有关的“不确定性”和还与生成的回复有关的“置信度”。然后我们提出并比较了几个置信度/不确定度指标，将它们应用于“选择性自然语言生成”，其中不可靠的结果可以被忽略或者移交给进一步的分析。

    Large language models (LLMs) specializing in natural language generation (NLG) have recently started exhibiting promising capabilities across a variety of domains. However, gauging the trustworthiness of responses generated by LLMs remains an open challenge, with limited research on uncertainty quantification for NLG. Furthermore, existing literature typically assumes white-box access to language models, which is becoming unrealistic either due to the closed-source nature of the latest LLMs or due to computational constraints. In this work, we investigate uncertainty quantification in NLG for $\textit{black-box}$ LLMs. We first differentiate two closely-related notions: $\textit{uncertainty}$, which depends only on the input, and $\textit{confidence}$, which additionally depends on the generated response. We then propose and compare several confidence/uncertainty metrics, applying them to $\textit{selective NLG}$, where unreliable results could either be ignored or yielded for further 
    
[^11]: 关于优化算法、李亚普诺夫函数和微分方程的联系：理论与洞见研究

    On the connections between optimization algorithms, Lyapunov functions, and differential equations: theory and insights. (arXiv:2305.08658v1 [math.OC])

    [http://arxiv.org/abs/2305.08658](http://arxiv.org/abs/2305.08658)

    本研究在推广线性矩阵不等式框架的基础上，研究了微分方程和优化算法的联系，提出了针对一个两参数Nesterov优化方法家族新的李亚普诺夫函数并表征其收敛速度，在此基础上证明了有显著改进的Nesterov方法的收敛速度，并确定出产生最佳速度的系数选择。

    

    本研究通过推广Fazylab等人在2018年发展的线性矩阵不等式框架，研究了用李亚普诺夫函数研究$m$-强凸和$L$-光滑函数的微分方程和优化算法之间的联系。使用新框架，我们针对一个两参数Nesterov优化方法家族的新型（离散）李亚普诺夫函数进行了解析推导，并表征了其收敛速度。这使得我们能够证明对于标准系数的Nesterov方法的先前证明速度有了明显改进，并且表征了产生最佳速度的系数选择。我们为Polyak ODE获得了新的李亚普诺夫函数，并重新审视了此ODE与Nesterov算法之间的联系。此外，讨论了将Nesterov方法解释为加性Runge-Kutta离散化的新方法，并解释了离散化Polyak方程的结构条件。

    We study connections between differential equations and optimization algorithms for $m$-strongly and $L$-smooth convex functions through the use of Lyapunov functions by generalizing the Linear Matrix Inequality framework developed by Fazylab et al. in 2018. Using the new framework we derive analytically a new (discrete) Lyapunov function for a two-parameter family of Nesterov optimization methods and characterize their convergence rate. This allows us to prove a convergence rate that improves substantially on the previously proven rate of Nesterov's method for the standard choice of coefficients, as well as to characterize the choice of coefficients that yields the optimal rate. We obtain a new Lyapunov function for the Polyak ODE and revisit the connection between this ODE and the Nesterov's algorithms. In addition discuss a new interpretation of Nesterov method as an additive Runge-Kutta discretization and explain the structural conditions that discretizations of the Polyak equation
    
[^12]: 量化时间差分学习的分析

    An Analysis of Quantile Temporal-Difference Learning. (arXiv:2301.04462v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.04462](http://arxiv.org/abs/2301.04462)

    本文证明了量化时间差分学习（QTD）在一定状态下的收敛概率为1，建立了QTD与非线性微分包含式之间的联系。

    

    本文分析了一个分布式强化学习算法：量化时间差分学习（QTD），该算法已成为多个成功的强化学习大规模应用的关键组成部分。尽管在实证方面取得了成功，但到目前为止，QTD的理论认识一直难以捉摸。与可以使用标准随机逼近工具来进行分析的经典TD学习不同，QTD的更新并不近似于收缩算子，高度非线性并且可能具有多个不动点。本文的核心结果是证明在与一类动态规划程序的不动点相应的状态下，QTD的收敛概率为1，从而让QTD在理论上得到了确定性的基础。证明通过随机逼近理论和非光滑分析将QTD与非线性微分包含式建立了联系。

    We analyse quantile temporal-difference learning (QTD), a distributional reinforcement learning algorithm that has proven to be a key component in several successful large-scale applications of reinforcement learning. Despite these empirical successes, a theoretical understanding of QTD has proven elusive until now. Unlike classical TD learning, which can be analysed with standard stochastic approximation tools, QTD updates do not approximate contraction mappings, are highly non-linear, and may have multiple fixed points. The core result of this paper is a proof of convergence to the fixed points of a related family of dynamic programming procedures with probability 1, putting QTD on firm theoretical footing. The proof establishes connections between QTD and non-linear differential inclusions through stochastic approximation theory and non-smooth analysis.
    
[^13]: 协变量转移的祝福和诅咒：对抗学习动态、方向收敛和平衡的影响

    Blessings and Curses of Covariate Shifts: Adversarial Learning Dynamics, Directional Convergence, and Equilibria. (arXiv:2212.02457v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.02457](http://arxiv.org/abs/2212.02457)

    协变量转移和对抗扰动对统计学习的稳健性提出了挑战。本文在无限维度的情况下研究了对抗协变量转移对外推区域的影响以及其对后续学习的平衡的影响。

    

    协变量分布转移和对抗扰动对传统统计学习框架的稳健性提出了挑战：测试协变量分布中的轻微转移能显著影响基于训练分布学习的统计模型性能。当外推发生时，即协变量转移到训练分布稀缺的区域时，模型性能通常会降低，因此，学习模型信息很少。为了稳健性和正则化考虑，建议采用对抗扰动技术，然而，需要对给定学习模型时对抗协变量转移的外推区域进行仔细研究。本文在无限维度的设置中精确刻画了外推区域，在回归和分类方面进行了研究。研究了对抗协变量转移对随后的平衡学习的影响。

    Covariate distribution shifts and adversarial perturbations present robustness challenges to the conventional statistical learning framework: mild shifts in the test covariate distribution can significantly affect the performance of the statistical model learned based on the training distribution. The model performance typically deteriorates when extrapolation happens: namely, covariates shift to a region where the training distribution is scarce, and naturally, the learned model has little information. For robustness and regularization considerations, adversarial perturbation techniques are proposed as a remedy; however, careful study needs to be carried out about what extrapolation region adversarial covariate shift will focus on, given a learned model. This paper precisely characterizes the extrapolation region, examining both regression and classification in an infinite-dimensional setting. We study the implications of adversarial covariate shifts to subsequent learning of the equi
    
[^14]: 公平主动学习：解决保险行业中的标注问题

    Fair Active Learning: Solving the Labeling Problem in Insurance. (arXiv:2112.09466v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2112.09466](http://arxiv.org/abs/2112.09466)

    本文旨在解决保险行业中普遍存在的机器学习模型在数据中发现的偏见和歧视，提出了公平主动学习方法，能够在实现模型预测性能的同时保证数据公平性。

    

    本文针对在保险行业广泛使用机器学习模型所面临的重大障碍，特别关注促进公平性。最初的挑战在于有效利用未标记的保险数据，通过主动学习技术降低标注的工作量，并强调数据相关性。本文探讨了各种主动学习抽样方法，并评估它们对合成和实际保险数据集的影响。该分析强调了实现公正模型推断的困难，因为机器学习模型可能会复制底层数据中存在的偏见和歧视。为了解决这些相互关联的挑战，本文介绍了一种创新的公平主动学习方法。所提出的方法采样信息量充足且公平的实例，在模型预测性能和公平性之间取得了良好的平衡，这一点在保险数据集上的数值实验中得到了证实。

    This paper addresses significant obstacles that arise from the widespread use of machine learning models in the insurance industry, with a specific focus on promoting fairness. The initial challenge lies in effectively leveraging unlabeled data in insurance while reducing the labeling effort and emphasizing data relevance through active learning techniques. The paper explores various active learning sampling methodologies and evaluates their impact on both synthetic and real insurance datasets. This analysis highlights the difficulty of achieving fair model inferences, as machine learning models may replicate biases and discrimination found in the underlying data. To tackle these interconnected challenges, the paper introduces an innovative fair active learning method. The proposed approach samples informative and fair instances, achieving a good balance between model predictive performance and fairness, as confirmed by numerical experiments on insurance datasets.
    

