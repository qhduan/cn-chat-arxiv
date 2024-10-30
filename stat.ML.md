# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Propensity Score Alignment of Unpaired Multimodal Data](https://arxiv.org/abs/2404.01595) | 本文提出了一种解决多模态表示学习中对齐不配对样本挑战的方法，通过估计倾向得分来定义样本之间的距离。 |
| [^2] | [Doubly Robust Inference in Causal Latent Factor Models](https://arxiv.org/abs/2402.11652) | 提出了一种双重稳健的估计量框架，可以在现代数据丰富的环境中估计存在未观察混杂因素下平均处理效应，具有良好的有限样本和渐近性质，并在参数速率下将其误差收敛为零均值高斯分布。 |
| [^3] | [OptEx: Expediting First-Order Optimization with Approximately Parallelized Iterations](https://arxiv.org/abs/2402.11427) | OptEx是第一个通过利用并行计算来减轻一阶优化的迭代瓶颈并增强效率的框架，使用核化梯度估计实现迭代的并行化，提供理论保证。 |
| [^4] | [A comprehensive framework for multi-fidelity surrogate modeling with noisy data: a gray-box perspective.](http://arxiv.org/abs/2401.06447) | 该论文介绍了一个综合的多保真度代理建模框架，能够将黑盒模型和白盒模型的信息结合起来，并能够处理噪声污染的数据，并估计出无噪声的高保真度函数。 |
| [^5] | [Externally Valid Policy Evaluation Combining Trial and Observational Data.](http://arxiv.org/abs/2310.14763) | 这项研究提出了一种结合试验和观察数据的外部有效策略评估方法，利用试验数据对目标人群上的政策结果进行有效推断，并给出了可验证的评估结果。 |
| [^6] | [On the Computational Complexity of Private High-dimensional Model Selection via the Exponential Mechanism.](http://arxiv.org/abs/2310.07852) | 本文研究了在高维稀疏线性回归模型中的差分隐私模型选择问题。我们使用指数机制进行模型选择，并提出了Metropolis-Hastings算法来克服指数搜索空间的计算复杂性。我们的算法在一定边界条件下能够实现强模型恢复性质，并具有多项式混合时间和近似差分隐私性质。 |
| [^7] | [A Neural Tangent Kernel View on Federated Averaging for Deep Linear Neural Network.](http://arxiv.org/abs/2310.05495) | 本论文介绍了一种基于神经切向核视角的联邦平均方法在深度线性神经网络上的应用，并探讨了该方法面临的挑战。 |
| [^8] | [Out-of-Distribution Optimality of Invariant Risk Minimization.](http://arxiv.org/abs/2307.11972) | 本文旨在提供IRM的理论验证，严格证明了解决方案可以最小化区外风险。 |
| [^9] | [Absorbing Phase Transitions in Artificial Deep Neural Networks.](http://arxiv.org/abs/2307.02284) | 本文研究了在适当初始化的有限神经网络中的吸收相变及其普适性，证明了即使在有限网络中仍然存在着从有序状态到混沌状态的过渡，并且不同的网络架构会反映在过渡的普适类上。 |
| [^10] | [Efficient Convex Algorithms for Universal Kernel Learning.](http://arxiv.org/abs/2304.07472) | 本文提出了一种基于SVD-QCQP原始对偶算法的高效内核学习实现，用于学习半分离核，大大降低了计算复杂度，并在几个基准数据集上展示了其准确性和速度。 |
| [^11] | [On the potential benefits of entropic regularization for smoothing Wasserstein estimators.](http://arxiv.org/abs/2210.06934) | 本文研究了熵正则化作为一种平滑方法在Wasserstein估计器中的潜在益处，通过替换最优输运成本的正则化版本来实现。主要发现是熵正则化可以以较低的计算成本达到与未正则化的Wasserstein估计器相当的统计性能。 |

# 详细

[^1]: 多模态数据无配对倾向得分对齐

    Propensity Score Alignment of Unpaired Multimodal Data

    [https://arxiv.org/abs/2404.01595](https://arxiv.org/abs/2404.01595)

    本文提出了一种解决多模态表示学习中对齐不配对样本挑战的方法，通过估计倾向得分来定义样本之间的距离。

    

    多模态表示学习技术通常依赖于配对样本来学习共同的表示，但在生物学等领域，往往难以收集配对样本，因为测量设备通常会破坏样本。本文介绍了一种解决多模态表示学习中对齐不配对样本的方法。我们将因果推断中的潜在结果与多模态观察中的潜在视图进行类比，这使我们能够使用Rubin的框架来估计一个共同的空间，以匹配样本。我们的方法假设我们收集了经过处理实验干扰的样本，并利用此来从每种模态中估计倾向得分，其中包括潜在状态和处理之间的所有共享信息，并可用于定义样本之间的距离。我们尝试了两种利用这一方法的对齐技术。

    arXiv:2404.01595v1 Announce Type: new  Abstract: Multimodal representation learning techniques typically rely on paired samples to learn common representations, but paired samples are challenging to collect in fields such as biology where measurement devices often destroy the samples. This paper presents an approach to address the challenge of aligning unpaired samples across disparate modalities in multimodal representation learning. We draw an analogy between potential outcomes in causal inference and potential views in multimodal observations, which allows us to use Rubin's framework to estimate a common space in which to match samples. Our approach assumes we collect samples that are experimentally perturbed by treatments, and uses this to estimate a propensity score from each modality, which encapsulates all shared information between a latent state and treatment and can be used to define a distance between samples. We experiment with two alignment techniques that leverage this di
    
[^2]: 因果潜在因子模型中的双重稳健推断

    Doubly Robust Inference in Causal Latent Factor Models

    [https://arxiv.org/abs/2402.11652](https://arxiv.org/abs/2402.11652)

    提出了一种双重稳健的估计量框架，可以在现代数据丰富的环境中估计存在未观察混杂因素下平均处理效应，具有良好的有限样本和渐近性质，并在参数速率下将其误差收敛为零均值高斯分布。

    

    本文介绍了一种在现代数据丰富环境中估计存在未观察混杂因素下的平均处理效应的新框架，该环境具有大量单位和结果。所提出的估计量是双重稳健的，结合了结果填补、倒数概率加权以及一种用于矩阵补全的新型交叉配对程序。我们推导了有限样本和渐近保证，并展示了新估计量的误差收敛到参数速率下的零均值高斯分布。模拟结果展示了本文分析的估计量的形式特性的实际相关性。

    arXiv:2402.11652v1 Announce Type: cross  Abstract: This article introduces a new framework for estimating average treatment effects under unobserved confounding in modern data-rich environments featuring large numbers of units and outcomes. The proposed estimator is doubly robust, combining outcome imputation, inverse probability weighting, and a novel cross-fitting procedure for matrix completion. We derive finite-sample and asymptotic guarantees, and show that the error of the new estimator converges to a mean-zero Gaussian distribution at a parametric rate. Simulation results demonstrate the practical relevance of the formal properties of the estimators analyzed in this article.
    
[^3]: OptEx: 利用近似并行化迭代加速一阶优化

    OptEx: Expediting First-Order Optimization with Approximately Parallelized Iterations

    [https://arxiv.org/abs/2402.11427](https://arxiv.org/abs/2402.11427)

    OptEx是第一个通过利用并行计算来减轻一阶优化的迭代瓶颈并增强效率的框架，使用核化梯度估计实现迭代的并行化，提供理论保证。

    

    第一阶优化（FOO）算法在诸如机器学习和信号去噪等众多计算领域中至关重要。然而，将它们应用于神经网络训练等复杂任务往往导致显著的低效，因为需要许多顺序迭代以实现收敛。为此，我们引入了第一阶优化加速近似并行迭代（OptEx），这是第一个通过利用并行计算来减轻其迭代瓶颈而增强FOO效率的框架。OptEx采用核化梯度估计来利用梯度历史进行未来梯度预测，实现了迭代的并行化 -- 这是一种曾经被认为由于FOO中固有的迭代依赖而不切实际的策略。我们为我们的核化梯度估计的可靠性和基于SGD的OptEx的迭代复杂度提供理论保证，并确认了其可靠性。

    arXiv:2402.11427v1 Announce Type: cross  Abstract: First-order optimization (FOO) algorithms are pivotal in numerous computational domains such as machine learning and signal denoising. However, their application to complex tasks like neural network training often entails significant inefficiencies due to the need for many sequential iterations for convergence. In response, we introduce first-order optimization expedited with approximately parallelized iterations (OptEx), the first framework that enhances the efficiency of FOO by leveraging parallel computing to mitigate its iterative bottleneck. OptEx employs kernelized gradient estimation to make use of gradient history for future gradient prediction, enabling parallelization of iterations -- a strategy once considered impractical because of the inherent iterative dependency in FOO. We provide theoretical guarantees for the reliability of our kernelized gradient estimation and the iteration complexity of SGD-based OptEx, confirming t
    
[^4]: 一个综合的多保真度代理建模框架，带有噪声数据：从灰盒的角度来看

    A comprehensive framework for multi-fidelity surrogate modeling with noisy data: a gray-box perspective. (arXiv:2401.06447v1 [stat.ME])

    [http://arxiv.org/abs/2401.06447](http://arxiv.org/abs/2401.06447)

    该论文介绍了一个综合的多保真度代理建模框架，能够将黑盒模型和白盒模型的信息结合起来，并能够处理噪声污染的数据，并估计出无噪声的高保真度函数。

    

    计算机模拟（即白盒模型）在模拟复杂工程系统方面比以往任何时候都更加必不可少。然而，仅凭计算模型往往无法完全捕捉现实的复杂性。当物理实验可行时，增强计算模型提供的不完整信息变得非常重要。灰盒建模涉及到将数据驱动模型（即黑盒模型）和白盒模型（即基于物理的模型）的信息融合的问题。在本文中，我们提出使用多保真度代理模型（MFSMs）来执行这个任务。MFSM将不同计算保真度的模型的信息集成到一个新的代理模型中。我们提出的多保真度代理建模框架能够处理被噪声污染的数据，并能够估计底层无噪声的高保真度函数。我们的方法强调以置信度的形式提供其预测中不确定性的精确估计。

    Computer simulations (a.k.a. white-box models) are more indispensable than ever to model intricate engineering systems. However, computational models alone often fail to fully capture the complexities of reality. When physical experiments are accessible though, it is of interest to enhance the incomplete information offered by computational models. Gray-box modeling is concerned with the problem of merging information from data-driven (a.k.a. black-box) models and white-box (i.e., physics-based) models. In this paper, we propose to perform this task by using multi-fidelity surrogate models (MFSMs). A MFSM integrates information from models with varying computational fidelity into a new surrogate model. The multi-fidelity surrogate modeling framework we propose handles noise-contaminated data and is able to estimate the underlying noise-free high-fidelity function. Our methodology emphasizes on delivering precise estimates of the uncertainty in its predictions in the form of confidence 
    
[^5]: 外部验证策略评估结合试验和观察数据

    Externally Valid Policy Evaluation Combining Trial and Observational Data. (arXiv:2310.14763v1 [stat.ME])

    [http://arxiv.org/abs/2310.14763](http://arxiv.org/abs/2310.14763)

    这项研究提出了一种结合试验和观察数据的外部有效策略评估方法，利用试验数据对目标人群上的政策结果进行有效推断，并给出了可验证的评估结果。

    

    随机试验被广泛认为是评估决策策略影响的金 standard。然而，试验数据来自可能与目标人群不同的人群，这引发了外部效度（也称为泛化能力）的问题。在本文中，我们试图利用试验数据对目标人群上的政策结果进行有效推断。目标人群的额外协变量数据用于模拟试验研究中个体的抽样。我们开发了一种方法，在任何指定的模型未校准范围内产生可验证的基于试验的政策评估。该方法是非参数的，即使样本是有限的，有效性也得到保证。使用模拟和实际数据说明了认证的政策评估结果。

    Randomized trials are widely considered as the gold standard for evaluating the effects of decision policies. Trial data is, however, drawn from a population which may differ from the intended target population and this raises a problem of external validity (aka. generalizability). In this paper we seek to use trial data to draw valid inferences about the outcome of a policy on the target population. Additional covariate data from the target population is used to model the sampling of individuals in the trial study. We develop a method that yields certifiably valid trial-based policy evaluations under any specified range of model miscalibrations. The method is nonparametric and the validity is assured even with finite samples. The certified policy evaluations are illustrated using both simulated and real data.
    
[^6]: 关于通过指数机制进行高维私有模型选择的计算复杂性

    On the Computational Complexity of Private High-dimensional Model Selection via the Exponential Mechanism. (arXiv:2310.07852v1 [stat.ML])

    [http://arxiv.org/abs/2310.07852](http://arxiv.org/abs/2310.07852)

    本文研究了在高维稀疏线性回归模型中的差分隐私模型选择问题。我们使用指数机制进行模型选择，并提出了Metropolis-Hastings算法来克服指数搜索空间的计算复杂性。我们的算法在一定边界条件下能够实现强模型恢复性质，并具有多项式混合时间和近似差分隐私性质。

    

    在差分隐私框架下，我们考虑了高维稀疏线性回归模型中的模型选择问题。具体而言，我们考虑了差分隐私最佳子集选择的问题，并研究了其效用保证。我们采用了广为人知的指数机制来选择最佳模型，并在一定边界条件下，建立了其强模型恢复性质。然而，指数机制的指数搜索空间导致了严重的计算瓶颈。为了克服这个挑战，我们提出了Metropolis-Hastings算法来进行采样步骤，并在问题参数$n$、$p$和$s$中建立了其到稳态分布的多项式混合时间。此外，我们还利用其混合性质建立了Metropolis-Hastings随机行走的最终估计的近似差分隐私性质。最后，我们还进行了一些说明性模拟，印证了我们主要结果的理论发现。

    We consider the problem of model selection in a high-dimensional sparse linear regression model under the differential privacy framework. In particular, we consider the problem of differentially private best subset selection and study its utility guarantee. We adopt the well-known exponential mechanism for selecting the best model, and under a certain margin condition, we establish its strong model recovery property. However, the exponential search space of the exponential mechanism poses a serious computational bottleneck. To overcome this challenge, we propose a Metropolis-Hastings algorithm for the sampling step and establish its polynomial mixing time to its stationary distribution in the problem parameters $n,p$, and $s$. Furthermore, we also establish approximate differential privacy for the final estimates of the Metropolis-Hastings random walk using its mixing property. Finally, we also perform some illustrative simulations that echo the theoretical findings of our main results
    
[^7]: 基于神经切向核的联邦平均在深度线性神经网络上的视角

    A Neural Tangent Kernel View on Federated Averaging for Deep Linear Neural Network. (arXiv:2310.05495v1 [cs.LG])

    [http://arxiv.org/abs/2310.05495](http://arxiv.org/abs/2310.05495)

    本论文介绍了一种基于神经切向核视角的联邦平均方法在深度线性神经网络上的应用，并探讨了该方法面临的挑战。

    

    联邦平均（FedAvg）是一种广泛使用的范式，用于在不共享数据的情况下协同训练来自分布式客户端的模型。如今，由于其卓越性能，神经网络取得了显著的成功，这使得它成为FedAvg中的首选模型。然而，神经网络的优化问题通常是非凸的甚至是非光滑的。此外，FedAvg总是涉及多个客户端和本地更新，导致不准确的更新方向。这些属性给分析FedAvg在训练神经网络中的收敛性带来了困难。最近，神经切向核（NTK）理论已被提出，用于理解解决神经网络非凸问题中的一阶方法的收敛性。深度线性神经网络是理论学科中的经典模型，由于其简单的公式。然而，在训练深度线性神经网络上，对于FedAvg的收敛性目前还没有理论结果。

    Federated averaging (FedAvg) is a widely employed paradigm for collaboratively training models from distributed clients without sharing data. Nowadays, the neural network has achieved remarkable success due to its extraordinary performance, which makes it a preferred choice as the model in FedAvg. However, the optimization problem of the neural network is often non-convex even non-smooth. Furthermore, FedAvg always involves multiple clients and local updates, which results in an inaccurate updating direction. These properties bring difficulties in analyzing the convergence of FedAvg in training neural networks. Recently, neural tangent kernel (NTK) theory has been proposed towards understanding the convergence of first-order methods in tackling the non-convex problem of neural networks. The deep linear neural network is a classical model in theoretical subject due to its simple formulation. Nevertheless, there exists no theoretical result for the convergence of FedAvg in training the d
    
[^8]: 不变风险最小化的区外优化性

    Out-of-Distribution Optimality of Invariant Risk Minimization. (arXiv:2307.11972v1 [stat.ML])

    [http://arxiv.org/abs/2307.11972](http://arxiv.org/abs/2307.11972)

    本文旨在提供IRM的理论验证，严格证明了解决方案可以最小化区外风险。

    

    深度神经网络经常继承训练数据中嵌入的虚假相关性，因此可能无法泛化到具有与提供训练数据的领域不同的未知域。M. Arjovsky等人（2019年）引入了区外（o.o.d.）风险的概念，即所有域中的最大风险，并将由虚假相关性引起的问题规定为最小化区外风险的问题。不变风险最小化（IRM）被认为是最小化区外风险的一种有前途的方法：IRM通过解决一个双层优化问题来估计最小化的区外风险。尽管IRM以实证成功吸引了相当多的关注，但它缺乏一些理论保证。特别是，还没有确立双层优化问题给出最小化区外风险的坚实理论保证。本文旨在提供IRM的理论验证，严格证明了解决方案可以通过在大仿真跟踪数据库中进行实时仿真，其包括对周围环境的直接感知，对潜在路线规划的策略认识，同时考虑到多车辆交互，以实现该问题的全局优化目标。

    Deep Neural Networks often inherit spurious correlations embedded in training data and hence may fail to generalize to unseen domains, which have different distributions from the domain to provide training data. M. Arjovsky et al. (2019) introduced the concept out-of-distribution (o.o.d.) risk, which is the maximum risk among all domains, and formulated the issue caused by spurious correlations as a minimization problem of the o.o.d. risk. Invariant Risk Minimization (IRM) is considered to be a promising approach to minimize the o.o.d. risk: IRM estimates a minimum of the o.o.d. risk by solving a bi-level optimization problem. While IRM has attracted considerable attention with empirical success, it comes with few theoretical guarantees. Especially, a solid theoretical guarantee that the bi-level optimization problem gives the minimum of the o.o.d. risk has not yet been established. Aiming at providing a theoretical justification for IRM, this paper rigorously proves that a solution to
    
[^9]: 人工深度神经网络中的吸收相变

    Absorbing Phase Transitions in Artificial Deep Neural Networks. (arXiv:2307.02284v1 [stat.ML])

    [http://arxiv.org/abs/2307.02284](http://arxiv.org/abs/2307.02284)

    本文研究了在适当初始化的有限神经网络中的吸收相变及其普适性，证明了即使在有限网络中仍然存在着从有序状态到混沌状态的过渡，并且不同的网络架构会反映在过渡的普适类上。

    

    由于著名的平均场理论，对于各种体系的无限宽度神经网络的行为的理论理解已经迅速发展。然而，对于更实际和现实重要性更强的有限网络，缺乏清晰直观的框架来延伸我们的理解。在本文中，我们展示了适当初始化的神经网络的行为可以用吸收相变中的普遍临界现象来理解。具体而言，我们研究了全连接前馈神经网络和卷积神经网络中从有序状态到混沌状态的相变，并强调了体系架构的差异与相变的普适类之间的关系。值得注意的是，我们还成功地应用了有限尺度扩展的方法，这表明了直观的现象学。

    Theoretical understanding of the behavior of infinitely-wide neural networks has been rapidly developed for various architectures due to the celebrated mean-field theory. However, there is a lack of a clear, intuitive framework for extending our understanding to finite networks that are of more practical and realistic importance. In the present contribution, we demonstrate that the behavior of properly initialized neural networks can be understood in terms of universal critical phenomena in absorbing phase transitions. More specifically, we study the order-to-chaos transition in the fully-connected feedforward neural networks and the convolutional ones to show that (i) there is a well-defined transition from the ordered state to the chaotics state even for the finite networks, and (ii) difference in architecture is reflected in that of the universality class of the transition. Remarkably, the finite-size scaling can also be successfully applied, indicating that intuitive phenomenologic
    
[^10]: 通用核学习的高效凸优化算法

    Efficient Convex Algorithms for Universal Kernel Learning. (arXiv:2304.07472v1 [stat.ML])

    [http://arxiv.org/abs/2304.07472](http://arxiv.org/abs/2304.07472)

    本文提出了一种基于SVD-QCQP原始对偶算法的高效内核学习实现，用于学习半分离核，大大降低了计算复杂度，并在几个基准数据集上展示了其准确性和速度。

    

    基于核优化的机器学习算法的准确性和复杂性取决于它们能够优化的核集。理想的核集应该：具有线性参数化（以便于可处理性）；在所有核集中密集（以便于鲁棒性）；是通用的（以便于准确性）。最近，提出了一种框架，使用正定矩阵来参数化一类正半分离核。尽管此类核能够满足所有三个标准，但之前用于优化此类核的算法仅限于分类，并且还依赖于计算复杂的半定规划（SDP）算法。在本文中，我们将学习半分离核的问题作为极小化极大化优化问题，并提出了一种SVD-QCQP原始对偶算法，其与之前基于SDP的方法相比，大大降低了计算复杂度。此外，我们提供了一种高效的内核学习实现，并在几个基准数据集上展示了其准确性和速度。

    The accuracy and complexity of machine learning algorithms based on kernel optimization are determined by the set of kernels over which they are able to optimize. An ideal set of kernels should: admit a linear parameterization (for tractability); be dense in the set of all kernels (for robustness); be universal (for accuracy). Recently, a framework was proposed for using positive matrices to parameterize a class of positive semi-separable kernels. Although this class can be shown to meet all three criteria, previous algorithms for optimization of such kernels were limited to classification and furthermore relied on computationally complex Semidefinite Programming (SDP) algorithms. In this paper, we pose the problem of learning semiseparable kernels as a minimax optimization problem and propose a SVD-QCQP primal-dual algorithm which dramatically reduces the computational complexity as compared with previous SDP-based approaches. Furthermore, we provide an efficient implementation of thi
    
[^11]: 关于使用熵正则化平滑Wasserstein估计器的潜在益处

    On the potential benefits of entropic regularization for smoothing Wasserstein estimators. (arXiv:2210.06934v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.06934](http://arxiv.org/abs/2210.06934)

    本文研究了熵正则化作为一种平滑方法在Wasserstein估计器中的潜在益处，通过替换最优输运成本的正则化版本来实现。主要发现是熵正则化可以以较低的计算成本达到与未正则化的Wasserstein估计器相当的统计性能。

    

    本文专注于研究熵正则化在最优输运中作为Wasserstein估计器的平滑方法，通过统计学中逼近误差和估计误差的经典权衡。Wasserstein估计器被定义为解决变分问题的解，其目标函数涉及概率测度之间的最优输运成本的使用。这样的估计器可以通过用熵惩罚替换最优输运成本的正则化版本来进行正则化，从而对结果估计器产生潜在的平滑效果。在这项工作中，我们探讨了熵正则化对正则化Wasserstein估计器的逼近和估计性质可能带来的益处。我们的主要贡献是讨论熵正则化如何以更低的计算成本达到与未正则化的Wasserstein估计器相当的统计性能。

    This paper is focused on the study of entropic regularization in optimal transport as a smoothing method for Wasserstein estimators, through the prism of the classical tradeoff between approximation and estimation errors in statistics. Wasserstein estimators are defined as solutions of variational problems whose objective function involves the use of an optimal transport cost between probability measures. Such estimators can be regularized by replacing the optimal transport cost by its regularized version using an entropy penalty on the transport plan. The use of such a regularization has a potentially significant smoothing effect on the resulting estimators. In this work, we investigate its potential benefits on the approximation and estimation properties of regularized Wasserstein estimators. Our main contribution is to discuss how entropic regularization may reach, at a lower computational cost, statistical performances that are comparable to those of un-regularized Wasserstein esti
    

