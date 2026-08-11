# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimizing CUDA like a Human: Micro-Profiling Tools as Expert Surrogates for LLM-Based GPU Kernel Optimization](https://arxiv.org/abs/2606.26453) | 提出KernelPro闭环多智能体系统，通过将专家启发式编码为可插拔微性能分析工具，结合多层级分析器反馈和领域自适应蒙特卡洛树搜索，实现GPU内核代码的自动生成与迭代优化。 |
| [^2] | [Unbiased Canonical Set-Valued Oracles Via Lattice Theory](https://arxiv.org/abs/2606.26418) | 本文通过Knaster–Tarski不动点定理，在完备格框架下提出了一种规范的非平凡credal集，解决了自指预言机在无偏和自洽约束下的唯一性问题。 |
| [^3] | [Matrix Completion via Nonsmooth Regularization of Fully Connected Neural Networks](https://arxiv.org/abs/2403.10232) | 通过对全连接神经网络进行非光滑正则化，可以控制过拟合问题，提高矩阵补全性能。 |
| [^4] | [Weak Correlations as the Underlying Principle for Linearization of Gradient-Based Learning Systems.](http://arxiv.org/abs/2401.04013) | 本文研究了梯度下降学习算法在参数动力学中的线性结构，发现这种线性化现象是由于初始值附近假设函数的一阶和高阶导数之间的弱相关性所致。这一发现为深度学习模型的线性化提供了新的认识。 |
| [^5] | [From Fake to Real (FFR): A two-stage training pipeline for mitigating spurious correlations with synthetic data.](http://arxiv.org/abs/2308.04553) | 本文提出了一个两阶段训练流程，通过在一个平衡的合成数据集上进行预训练，然后在真实数据上进行微调，减少了视觉识别模型学习到与数据集偏差相关的错误的问题。 |
| [^6] | [A Differentially Private Weighted Empirical Risk Minimization Procedure and its Application to Outcome Weighted Learning.](http://arxiv.org/abs/2307.13127) | 本文提出了一种差分隐私加权经验风险最小化算法，可以在使用敏感数据的情况下保护隐私。这是第一个在权重ERM中应用差分隐私的算法，并且在一定的条件下提供了严格的DP保证。 |
| [^7] | [Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization.](http://arxiv.org/abs/2307.10053) | 本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。 |
| [^8] | [Causal Falsification of Digital Twins.](http://arxiv.org/abs/2301.07210) | 这篇论文提出了一种数字孪生的因果伪证方法，以可靠并实用的方式在最小限度的假设下提供孪生的信息和评估结果。 |

# 详细

[^1]: 像人类一样优化CUDA：微性能分析工具作为基于LLM的GPU内核优化的专家替代方案

    Optimizing CUDA like a Human: Micro-Profiling Tools as Expert Surrogates for LLM-Based GPU Kernel Optimization

    [https://arxiv.org/abs/2606.26453](https://arxiv.org/abs/2606.26453)

    提出KernelPro闭环多智能体系统，通过将专家启发式编码为可插拔微性能分析工具，结合多层级分析器反馈和领域自适应蒙特卡洛树搜索，实现GPU内核代码的自动生成与迭代优化。

    

    arXiv:2606.26453v1 公告类型：新 摘要：我们提出了KernelPro，一个闭环多智能体系统，通过将大语言模型代码生成与硬件分析器反馈及可插拔的瓶颈检测工具相结合，自动生成、分析并迭代优化GPU内核代码。KernelPro贡献了四个方面：（1）一个语义反馈算子，将专家启发式编码为可插拔的微性能分析工具，将原始硬件指标转化为可操作的自然语言指导；（2）一个两阶段工具调用架构，其中基于屋顶线的瓶颈分类筛选出哪些专门分析工具执行，结合内核级（ncu）、指令级（SASS）和系统级（nsys）分析；（3）一个领域自适应的蒙特卡洛树搜索，具备渐进扩展、非对称分支、对数奖励校准、死胡同剪枝和用于跨迭代学习的搜索记忆；（4）通过自主协作直接生成CuTe源代码。

    arXiv:2606.26453v1 Announce Type: new  Abstract: We present KernelPro, a closed-loop multi-agent system that automatically generates, profiles, and iteratively optimizes GPU kernel code by integrating large language model (LLM) code generation with hardware profiler feedback and pluggable bottleneck detection tools. KernelPro introduces four contributions: (1) a semantic feedback operator that encodes expert heuristics as pluggable micro-profiling tools, transforming raw hardware metrics into actionable natural language guidance; (2) a two-stage tool invocation architecture where roofline-based bottleneck classification filters which specialized analysis tools execute, combining kernel-level (ncu), instruction-level (SASS), and system-level (nsys) profiling; (3) a domain-adapted MCTS with progressive widening, asymmetric branching, log-reward calibration, dead-end pruning, and search memory for cross-iteration learning; and (4) direct CuTe source-level code generation via autonomous co
    
[^2]: 基于格理论的无偏规范集值预言机

    Unbiased Canonical Set-Valued Oracles Via Lattice Theory

    [https://arxiv.org/abs/2606.26418](https://arxiv.org/abs/2606.26418)

    本文通过Knaster–Tarski不动点定理，在完备格框架下提出了一种规范的非平凡credal集，解决了自指预言机在无偏和自洽约束下的唯一性问题。

    

    非智能体“预言机”AI在估计未来事件概率时面临自指问题：一旦其答案被学习并采取行动，就会改变它被要求报告的概率本身。针对科学家AI计划所倡导的一种回应是只询问反事实问题，并假设答案没有影响进行评估。我们观察到，这类答案一旦被学习就会变得无关紧要，恰恰是因为其前提随后变为假。因此，我们探索了一种自指替代方案：预言机报告的不是单一概率，而是一个同时无偏且与学习后果自洽的credal集。朴素的自洽性要求被太多集合满足（包括无用的答案[0,1]），因此问题在于挑选出一个规范的、非平凡的成员。我们通过闭包完备格上的Knaster–Tarski不动点定理实现了这一点。

    arXiv:2606.26418v1 Announce Type: new  Abstract: A non-agentic "oracle" AI that estimates probabilities of future events faces a self-reference problem: once its answer is learned and acted upon, it can change the very probability it was asked to report. One response, advocated for the Scientist AI programme, is to ask only counterfactual questions, evaluated as if the answer had no influence. We observe that such answers tend to become irrelevant the moment they are learned, precisely because their premise is then false. We therefore explore a self-referential alternative in which the oracle reports not a single probability but a credal set that is simultaneously unbiased and self-consistent with the consequences of being learned. The naive self-consistency requirement is satisfied by too many sets (including the useless answer $[0,1]$), so the problem is to single out a canonical, nontrivial member. We do so with the Knaster--Tarski fixed-point theorem on the complete lattice of clos
    
[^3]: 通过对全连接神经网络进行非光滑正则化的矩阵补全

    Matrix Completion via Nonsmooth Regularization of Fully Connected Neural Networks

    [https://arxiv.org/abs/2403.10232](https://arxiv.org/abs/2403.10232)

    通过对全连接神经网络进行非光滑正则化，可以控制过拟合问题，提高矩阵补全性能。

    

    传统的矩阵补全方法通过假设矩阵具有低秩来逼近缺失值，从而导致缺失值的线性逼近。已经表明，使用非线性估计器（如深度神经网络）可以获得更好的性能。深度全连接神经网络（FCNN）是矩阵补全最适合的架构之一，由于其高容量而导致过拟合，进而导致泛化能力低。本文通过在中间表示的 $\ell_{1}$ 范数和权重矩阵的核范数方面对FCNN模型进行正则化来控制过拟合。因此，得到的正则化目标函数变得非光滑和非凸，即现有的基于梯度的方法无法应用于我们的模型。我们提出了一种近端梯度方法的变体，并研究其收敛到临界点。在FCNN的初始时期。

    arXiv:2403.10232v1 Announce Type: cross  Abstract: Conventional matrix completion methods approximate the missing values by assuming the matrix to be low-rank, which leads to a linear approximation of missing values. It has been shown that enhanced performance could be attained by using nonlinear estimators such as deep neural networks. Deep fully connected neural networks (FCNNs), one of the most suitable architectures for matrix completion, suffer from over-fitting due to their high capacity, which leads to low generalizability. In this paper, we control over-fitting by regularizing the FCNN model in terms of the $\ell_{1}$ norm of intermediate representations and nuclear norm of weight matrices. As such, the resulting regularized objective function becomes nonsmooth and nonconvex, i.e., existing gradient-based methods cannot be applied to our model. We propose a variant of the proximal gradient method and investigate its convergence to a critical point. In the initial epochs of FCNN
    
[^4]: 以弱相关性作为梯度下降学习系统线性化的基本原则

    Weak Correlations as the Underlying Principle for Linearization of Gradient-Based Learning Systems. (arXiv:2401.04013v1 [cs.LG])

    [http://arxiv.org/abs/2401.04013](http://arxiv.org/abs/2401.04013)

    本文研究了梯度下降学习算法在参数动力学中的线性结构，发现这种线性化现象是由于初始值附近假设函数的一阶和高阶导数之间的弱相关性所致。这一发现为深度学习模型的线性化提供了新的认识。

    

    深度学习模型，如宽神经网络，可以被概念化为非线性动力学物理系统，其具有多个相互作用的自由度。在无限极限下，这些系统趋向于表现出简化的动力学。本文深入研究了基于梯度下降的学习算法，在其参数动力学中展示出与神经切向核类似的线性结构。我们发现，这种明显的线性化是因为在初始值附近，假设函数的一阶和高阶导数之间的弱相关性。这一洞见表明，这些弱相关性可能是此类系统中观察到的线性化的潜在原因。作为一个例证，我们展示了在宽度很大的神经网络中存在的这种弱相关性结构。利用线性和弱相关性之间的关系，我们推导出线性度偏离的一个界限。

    Deep learning models, such as wide neural networks, can be conceptualized as nonlinear dynamical physical systems characterized by a multitude of interacting degrees of freedom. Such systems in the infinite limit, tend to exhibit simplified dynamics. This paper delves into gradient descent-based learning algorithms, that display a linear structure in their parameter dynamics, reminiscent of the neural tangent kernel. We establish this apparent linearity arises due to weak correlations between the first and higher-order derivatives of the hypothesis function, concerning the parameters, taken around their initial values. This insight suggests that these weak correlations could be the underlying reason for the observed linearization in such systems. As a case in point, we showcase this weak correlations structure within neural networks in the large width limit. Exploiting the relationship between linearity and weak correlations, we derive a bound on deviations from linearity observed duri
    
[^5]: 从假到真（FFR）：一种用于减少与合成数据相关性错误的两阶段训练流程

    From Fake to Real (FFR): A two-stage training pipeline for mitigating spurious correlations with synthetic data. (arXiv:2308.04553v1 [cs.CV])

    [http://arxiv.org/abs/2308.04553](http://arxiv.org/abs/2308.04553)

    本文提出了一个两阶段训练流程，通过在一个平衡的合成数据集上进行预训练，然后在真实数据上进行微调，减少了视觉识别模型学习到与数据集偏差相关的错误的问题。

    

    视觉识别模型容易学习到由于训练集的不平衡导致的相关性错误，其中某些群体（如女性）在某些类别（如程序员）中代表性不足。生成模型通过为少数样本生成合成数据来减少这种偏差，从而平衡训练集。然而，先前使用这些方法的工作忽视了视觉识别模型往往能够学习区分真实图像和合成图像的能力，因此无法消除原始数据集中的偏差。在我们的工作中，我们提出了一种新颖的两阶段流程来减少这个问题，其中1）我们在平衡的合成数据集上进行预训练，然后2）在真实数据上进行微调。使用这个流程，我们避免了在真实数据和合成数据上的训练，从而避免了真实数据和合成数据之间的偏差。此外，在第一步中我们学习到了抵抗偏差的稳健特征，在第二步中减轻了偏差。

    Visual recognition models are prone to learning spurious correlations induced by an imbalanced training set where certain groups (\eg Females) are under-represented in certain classes (\eg Programmers). Generative models offer a promising direction in mitigating this bias by generating synthetic data for the minority samples and thus balancing the training set. However, prior work that uses these approaches overlooks that visual recognition models could often learn to differentiate between real and synthetic images and thus fail to unlearn the bias in the original dataset. In our work, we propose a novel two-stage pipeline to mitigate this issue where 1) we pre-train a model on a balanced synthetic dataset and then 2) fine-tune on the real data. Using this pipeline, we avoid training on both real and synthetic data, thus avoiding the bias between real and synthetic data. Moreover, we learn robust features against the bias in the first step that mitigate the bias in the second step. Mor
    
[^6]: 一个差分隐私加权经验风险最小化算法及其在结果加权学习中的应用

    A Differentially Private Weighted Empirical Risk Minimization Procedure and its Application to Outcome Weighted Learning. (arXiv:2307.13127v1 [stat.ML])

    [http://arxiv.org/abs/2307.13127](http://arxiv.org/abs/2307.13127)

    本文提出了一种差分隐私加权经验风险最小化算法，可以在使用敏感数据的情况下保护隐私。这是第一个在权重ERM中应用差分隐私的算法，并且在一定的条件下提供了严格的DP保证。

    

    在经验风险最小化(ERM)框架中，使用包含个人信息的数据来构建预测模型是常见的做法。尽管这些模型在预测上可以非常准确，但使用敏感数据得到的结果可能容易受到隐私攻击。差分隐私(DP)是一种有吸引力的框架，可以通过提供数学上可证明的隐私损失界限来解决这些数据隐私问题。先前的工作主要集中在将DP应用于无权重的ERM中。我们考虑到了权重ERM(wERM)的重要推广。在wERM中，可以为每个个体的目标函数贡献分配不同的权重。在这个背景下，我们提出了第一个有差分隐私保障的wERM算法，并在一定的正则条件下提供了严格的理论证明。将现有的DP-ERM程序扩展到wERM为结果加权学习铺平了道路。

    It is commonplace to use data containing personal information to build predictive models in the framework of empirical risk minimization (ERM). While these models can be highly accurate in prediction, results obtained from these models with the use of sensitive data may be susceptible to privacy attacks. Differential privacy (DP) is an appealing framework for addressing such data privacy issues by providing mathematically provable bounds on the privacy loss incurred when releasing information from sensitive data. Previous work has primarily concentrated on applying DP to unweighted ERM. We consider an important generalization to weighted ERM (wERM). In wERM, each individual's contribution to the objective function can be assigned varying weights. In this context, we propose the first differentially private wERM algorithm, backed by a rigorous theoretical proof of its DP guarantees under mild regularity conditions. Extending the existing DP-ERM procedures to wERM paves a path to derivin
    
[^7]: 非平滑非凸优化中随机次梯度方法的收敛性保证

    Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization. (arXiv:2307.10053v1 [math.OC])

    [http://arxiv.org/abs/2307.10053](http://arxiv.org/abs/2307.10053)

    本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。

    

    本文研究了随机梯度下降（SGD）方法及其变种在训练由非平滑激活函数构建的神经网络中的收敛性质。我们提出了一种新颖的框架，为更新动量项和变量的步长分配了不同的时间尺度。在一些温和的条件下，我们证明了我们提出的框架在单时间尺度和双时间尺度情况下的全局收敛性。我们还证明了我们提出的框架包含了很多已知的SGD类型方法，包括heavy-ball SGD、SignSGD、Lion、normalized SGD和clipped SGD。此外，当目标函数采用有限和形式时，我们基于我们提出的框架证明了这些SGD类型方法的收敛性质。特别地，在温和的假设下，我们证明了这些SGD类型方法在随机选择的步长和初始点上能够找到目标函数的Clarke稳定点。

    In this paper, we investigate the convergence properties of the stochastic gradient descent (SGD) method and its variants, especially in training neural networks built from nonsmooth activation functions. We develop a novel framework that assigns different timescales to stepsizes for updating the momentum terms and variables, respectively. Under mild conditions, we prove the global convergence of our proposed framework in both single-timescale and two-timescale cases. We show that our proposed framework encompasses a wide range of well-known SGD-type methods, including heavy-ball SGD, SignSGD, Lion, normalized SGD and clipped SGD. Furthermore, when the objective function adopts a finite-sum formulation, we prove the convergence properties for these SGD-type methods based on our proposed framework. In particular, we prove that these SGD-type methods find the Clarke stationary points of the objective function with randomly chosen stepsizes and initial points under mild assumptions. Preli
    
[^8]: 数字孪生的因果伪证

    Causal Falsification of Digital Twins. (arXiv:2301.07210v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2301.07210](http://arxiv.org/abs/2301.07210)

    这篇论文提出了一种数字孪生的因果伪证方法，以可靠并实用的方式在最小限度的假设下提供孪生的信息和评估结果。

    

    数字孪生在很多应用中具有巨大的潜力，但是在安全关键场景下广泛部署它们的精度评估需要严格的程序。通过在因果推理框架内制定这个任务，我们表明，使用现实数据尝试证明孪生的正确性是不可靠的，除非在数据生成过程中进行可能有风险的假设。为了避免这些假设，我们提出了一种评估策略，旨在找到孪生不正确的情况，并提出了用于实现此目标的通用统计过程，可用于各种应用和孪生模型。我们的方法在最小假设下提供了可靠和可操作的孪生信息和评估结果。通过包含脉冲生理学引擎中脓毒症建模的大型案例研究，我们证明了我们方法的有效性。

    Digital twins hold substantial promise in many applications, but rigorous procedures for assessing their accuracy are essential for their widespread deployment in safety-critical settings. By formulating this task within the framework of causal inference, we show that attempts to certify the correctness of a twin using real-world observational data are unsound unless potentially tenuous assumptions are made about the data-generating process. To avoid these assumptions, we propose an assessment strategy that instead aims to find cases where the twin is not correct, and present a general-purpose statistical procedure for doing so that may be used across a wide variety of applications and twin models. Our approach yields reliable and actionable information about the twin under minimal assumptions about the twin and the real-world process of interest. We demonstrate the effectiveness of our methodology via a large-scale case study involving sepsis modelling within the Pulse Physiology Engi
    

