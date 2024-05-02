# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Model Collapse Demystified: The Case of Regression](https://arxiv.org/abs/2402.07712) | 本研究在核回归的简化环境中解析了模型崩溃现象，并发现了模型能够处理虚假数据与性能完全崩溃之间的交叉点。通过提出基于自适应正则化的策略，成功缓解了模型崩溃问题。这些发现通过实验证实。 |
| [^2] | [Hidden yet quantifiable: A lower bound for confounding strength using randomized trials](https://arxiv.org/abs/2312.03871) | 利用随机试验设计了一种统计检验，能够量化未观察到的混淆强度，并估计其下界，有效应用于现实世界中识别混淆。 |
| [^3] | [Cross-Validation Conformal Risk Control.](http://arxiv.org/abs/2401.11974) | 本文提出了一种基于交叉验证的新型合规风险控制方法(CV-CRC)，它扩展了一致性预测的概念，能够控制更广泛的风险函数，并在预测器集合的平均风险上提供了理论保证。 |
| [^4] | [Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint.](http://arxiv.org/abs/2312.11456) | 该论文研究了在KL约束下的反馈强化学习的理论框架，并提出了有效的算法和实践。实证评估表明，该框架在大型语言模型的对齐实验中表现出良好的效果。 |
| [^5] | [Tractable MCMC for Private Learning with Pure and Gaussian Differential Privacy.](http://arxiv.org/abs/2310.14661) | 本文介绍了一种基于纯差分隐私和高斯差分隐私的可计算MCMC私有学习方法，通过引入近似采样扰动算法，结合Metropolis-Hastings算法和局部化步骤，实现了对隐私的保护并获得了较好的收敛性能。 |
| [^6] | [Efficient Algorithms for the CCA Family: Unconstrained Objectives with Unbiased Gradients.](http://arxiv.org/abs/2310.01012) | 本论文提出了一个新颖的无约束目标，通过应用随机梯度下降（SGD）到CCA目标，实现了一系列快速算法，包括随机PLS、随机CCA和深度CCA。这些方法在各种基准测试中表现出比先前最先进方法更快的收敛速度和更高的相关性恢复。 |
| [^7] | [Prediction without Preclusion: Recourse Verification with Reachable Sets.](http://arxiv.org/abs/2308.12820) | 这项研究引入了一种称为后续验证的正式测试程序，用于检测模型分配固定预测的情况。通过开发可靠的机制，可以确定给定模型是否能为决策对象提供后续措施，从而解决了模型分配固定预测可能带来的问题。该研究还展示了如何在真实世界的数据集中确保后续措施和对抗鲁棒性，并探讨了在贷款数据集中实现后续措施的不可行性。 |
| [^8] | [Gaussian random field approximation via Stein's method with applications to wide random neural networks.](http://arxiv.org/abs/2306.16308) | 本研究利用Stein方法推导出Wasserstein距离的上界，通过高斯平滑技术将平滑度量转化为Wasserstein距离。通过特殊化结果，我们获得了广义随机神经网络中对高斯随机场逼近的首个上界。 |
| [^9] | [Multi-Objective Optimization Using the R2 Utility.](http://arxiv.org/abs/2305.11774) | 本文提出将多目标优化问题转化为一组单目标问题进行解决，并介绍了R2效用函数作为适当的目标函数。该效用函数单调且次模，可以使用贪心优化算法计算全局最优解。 |
| [^10] | [Inverse Unscented Kalman Filter.](http://arxiv.org/abs/2304.01698) | 本论文提出了针对非线性系统动态的反向无味卡尔曼滤波器（I-UKF）以及基于再生核希尔伯特空间的UKF（RKHS-UKF），用于学习未知的系统模型并估计状态。 |

# 详细

[^1]: 模型崩溃解密：回归案例研究

    Model Collapse Demystified: The Case of Regression

    [https://arxiv.org/abs/2402.07712](https://arxiv.org/abs/2402.07712)

    本研究在核回归的简化环境中解析了模型崩溃现象，并发现了模型能够处理虚假数据与性能完全崩溃之间的交叉点。通过提出基于自适应正则化的策略，成功缓解了模型崩溃问题。这些发现通过实验证实。

    

    在像ChatGPT这样的大型语言模型的时代，"模型崩溃"现象指的是模型在递归地训练自身上一代又一代生成的数据时，其性能逐渐降低，最终变得完全无用，即模型崩溃。在这项工作中，我们在核回归的简化环境中研究了这一现象，并获得了结果，显示模型能够处理虚假数据与模型性能完全崩溃之间存在明显的交叉点。在多项式衰减的光谱和源条件下，我们获得了修改后的缩放定律，展示了从快速到缓慢速率的新交叉现象。我们还提出了基于自适应正则化的简单策略来缓解模型崩溃。我们的理论结果通过实验证实。

    In the era of large language models like ChatGPT, the phenomenon of "model collapse" refers to the situation whereby as a model is trained recursively on data generated from previous generations of itself over time, its performance degrades until the model eventually becomes completely useless, i.e the model collapses. In this work, we study this phenomenon in the simplified setting of kernel regression and obtain results which show a clear crossover between where the model can cope with fake data, and a regime where the model's performance completely collapses. Under polynomial decaying spectral and source conditions, we obtain modified scaling laws which exhibit new crossover phenomena from fast to slow rates. We also propose a simple strategy based on adaptive regularization to mitigate model collapse. Our theoretical results are validated with experiments.
    
[^2]: 隐蔽而可量化：使用随机试验的混淆强度下界

    Hidden yet quantifiable: A lower bound for confounding strength using randomized trials

    [https://arxiv.org/abs/2312.03871](https://arxiv.org/abs/2312.03871)

    利用随机试验设计了一种统计检验，能够量化未观察到的混淆强度，并估计其下界，有效应用于现实世界中识别混淆。

    

    在快节奏精准医学时代，观察性研究在正确评估临床实践中新疗法方面发挥着重要作用。然而，未观察到的混淆可能严重损害从非随机数据中得出的因果结论。我们提出了一种利用随机试验来量化未观察到的混淆的新策略。首先，我们设计了一种统计检验来检测强度超过给定阈值的未观察到的混淆。然后，我们使用该检验来估计未观察到的混淆强度的渐近有效下界。我们在几个合成和半合成数据集上评估了我们的统计检验的功效和有效性。此外，我们展示了我们的下界如何能够在真实环境中正确识别未观察到的混淆的存在和不存在。

    arXiv:2312.03871v2 Announce Type: replace-cross  Abstract: In the era of fast-paced precision medicine, observational studies play a major role in properly evaluating new treatments in clinical practice. Yet, unobserved confounding can significantly compromise causal conclusions drawn from non-randomized data. We propose a novel strategy that leverages randomized trials to quantify unobserved confounding. First, we design a statistical test to detect unobserved confounding with strength above a given threshold. Then, we use the test to estimate an asymptotically valid lower bound on the unobserved confounding strength. We evaluate the power and validity of our statistical test on several synthetic and semi-synthetic datasets. Further, we show how our lower bound can correctly identify the absence and presence of unobserved confounding in a real-world setting.
    
[^3]: 交叉验证合规风险控制

    Cross-Validation Conformal Risk Control. (arXiv:2401.11974v1 [cs.LG])

    [http://arxiv.org/abs/2401.11974](http://arxiv.org/abs/2401.11974)

    本文提出了一种基于交叉验证的新型合规风险控制方法(CV-CRC)，它扩展了一致性预测的概念，能够控制更广泛的风险函数，并在预测器集合的平均风险上提供了理论保证。

    

    合规风险控制（CRC）是一种最近提出的技术，它应用于传统的点预测器上，以提供校准保证。在CRC中推广一致性预测（CP），通过从点预测器中提取一个预测器集合来控制风险函数（如误覆盖概率或错误负例率），从而确保校准性。原始的CRC需要将可用数据集分为训练和验证数据集。当数据可用性有限时，这可能导致预测器集合效率低下。本文介绍了一种基于交叉验证而不是原始CRC的新型CRC方法。所提出的交叉验证CRC（CV-CRC）将CP的一种版本扩展到CRC，可以控制更广泛的风险函数。CV-CRC被证明在预测器集合的平均风险上具有理论保证。此外，通过数值实验证明CV-CRC在实践中的有效性。

    Conformal risk control (CRC) is a recently proposed technique that applies post-hoc to a conventional point predictor to provide calibration guarantees. Generalizing conformal prediction (CP), with CRC, calibration is ensured for a set predictor that is extracted from the point predictor to control a risk function such as the probability of miscoverage or the false negative rate. The original CRC requires the available data set to be split between training and validation data sets. This can be problematic when data availability is limited, resulting in inefficient set predictors. In this paper, a novel CRC method is introduced that is based on cross-validation, rather than on validation as the original CRC. The proposed cross-validation CRC (CV-CRC) extends a version of the jackknife-minmax from CP to CRC, allowing for the control of a broader range of risk functions. CV-CRC is proved to offer theoretical guarantees on the average risk of the set predictor. Furthermore, numerical exper
    
[^4]: 人类反馈的迭代偏好学习：在KL约束下将理论与实践联系起来的RLHF

    Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint. (arXiv:2312.11456v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.11456](http://arxiv.org/abs/2312.11456)

    该论文研究了在KL约束下的反馈强化学习的理论框架，并提出了有效的算法和实践。实证评估表明，该框架在大型语言模型的对齐实验中表现出良好的效果。

    

    本文研究了生成模型与强化学习从人类反馈中的对齐过程的理论框架。我们考虑了一个标准的数学表达式，即反向KL正则化的上下文多臂赌博机用于RLHF。尽管它被广泛应用于实际应用，但对这个公式的严格理论分析仍然很开放。我们研究了它在离线、在线和混合三种不同场景下的行为，并提出了具有有限样本理论保证的高效算法。朝着实际应用的方向，我们的框架通过对信息理论策略改进预言的稳健近似，自然地产生了几种新颖的RLHF算法。这包括在线场景中的迭代版本的直接偏好优化(DPO)算法，以及离线情景下的多步拒绝抽样策略。我们对大型语言模型的真实对齐实验进行了实证评估。

    This paper studies the theoretical framework of the alignment process of generative models with Reinforcement Learning from Human Feedback (RLHF). We consider a standard mathematical formulation, the reverse-KL regularized contextual bandit for RLHF. Despite its widespread practical application, a rigorous theoretical analysis of this formulation remains open. We investigate its behavior in three distinct settings -- offline, online, and hybrid -- and propose efficient algorithms with finite-sample theoretical guarantees.  Moving towards practical applications, our framework, with a robust approximation of the information-theoretical policy improvement oracle, naturally gives rise to several novel RLHF algorithms. This includes an iterative version of the Direct Preference Optimization (DPO) algorithm for online settings, and a multi-step rejection sampling strategy for offline scenarios. Our empirical evaluations on real-world alignment experiment of large language model demonstrate t
    
[^5]: 基于纯差分隐私和高斯差分隐私的可计算MCMC私有学习方法。

    Tractable MCMC for Private Learning with Pure and Gaussian Differential Privacy. (arXiv:2310.14661v1 [cs.LG])

    [http://arxiv.org/abs/2310.14661](http://arxiv.org/abs/2310.14661)

    本文介绍了一种基于纯差分隐私和高斯差分隐私的可计算MCMC私有学习方法，通过引入近似采样扰动算法，结合Metropolis-Hastings算法和局部化步骤，实现了对隐私的保护并获得了较好的收敛性能。

    

    后验采样即从后验分布中采样的指数机制，提供ε-纯差分隐私（DP）保证，并不受（ε，δ）-近似DP引入的潜在无界隐私泄漏的影响。然而，在实践中，需要应用近似采样方法，如马尔科夫链蒙特卡洛（MCMC），从而重新引入了对隐私保证的δ-近似误差。为了弥合这一差距，我们提出了近似采样扰动（即ASAP）算法，该算法通过与满足纯DP或纯高斯DP（即δ=0）的参考分布有界Wasserstein无穷距离的MCMC样本加噪声。然后利用Metropolis-Hastings算法生成样本并证明算法在W$_\infty$距离上收敛。我们展示了通过将我们的新技术与细致的局部化步骤相结合，我们获得了第一个可计算MCMC私有学习方法。

    Posterior sampling, i.e., exponential mechanism to sample from the posterior distribution, provides $\varepsilon$-pure differential privacy (DP) guarantees and does not suffer from potentially unbounded privacy breach introduced by $(\varepsilon,\delta)$-approximate DP. In practice, however, one needs to apply approximate sampling methods such as Markov chain Monte Carlo (MCMC), thus re-introducing the unappealing $\delta$-approximation error into the privacy guarantees. To bridge this gap, we propose the Approximate SAample Perturbation (abbr. ASAP) algorithm which perturbs an MCMC sample with noise proportional to its Wasserstein-infinity ($W_\infty$) distance from a reference distribution that satisfies pure DP or pure Gaussian DP (i.e., $\delta=0$). We then leverage a Metropolis-Hastings algorithm to generate the sample and prove that the algorithm converges in W$_\infty$ distance. We show that by combining our new techniques with a careful localization step, we obtain the first ne
    
[^6]: CCA家族的高效算法：无约束目标与无偏梯度

    Efficient Algorithms for the CCA Family: Unconstrained Objectives with Unbiased Gradients. (arXiv:2310.01012v1 [cs.LG])

    [http://arxiv.org/abs/2310.01012](http://arxiv.org/abs/2310.01012)

    本论文提出了一个新颖的无约束目标，通过应用随机梯度下降（SGD）到CCA目标，实现了一系列快速算法，包括随机PLS、随机CCA和深度CCA。这些方法在各种基准测试中表现出比先前最先进方法更快的收敛速度和更高的相关性恢复。

    

    典型相关分析（CCA）方法在多视角学习中具有基础性作用。正则化线性CCA方法可以看作是偏最小二乘（PLS）的推广，并与广义特征值问题（GEP）框架统一。然而，这些线性方法的传统算法在大规模数据上计算上是不可行的。深度CCA的扩展显示出很大的潜力，但目前的训练过程缓慢且复杂。我们首先提出了一个描述GEPs的顶级子空间的新颖无约束目标。我们的核心贡献是一系列快速算法，用随机梯度下降（SGD）应用于相应的CCA目标，从而获得随机PLS、随机CCA和深度CCA。这些方法在所有标准CCA和深度CCA基准测试中显示出比先前最先进方法更快的收敛速度和更高的相关性恢复。这样的速度使我们能够首次进行大规模生物数据的PLS分析。

    The Canonical Correlation Analysis (CCA) family of methods is foundational in multi-view learning. Regularised linear CCA methods can be seen to generalise Partial Least Squares (PLS) and unified with a Generalized Eigenvalue Problem (GEP) framework. However, classical algorithms for these linear methods are computationally infeasible for large-scale data. Extensions to Deep CCA show great promise, but current training procedures are slow and complicated. First we propose a novel unconstrained objective that characterizes the top subspace of GEPs. Our core contribution is a family of fast algorithms for stochastic PLS, stochastic CCA, and Deep CCA, simply obtained by applying stochastic gradient descent (SGD) to the corresponding CCA objectives. These methods show far faster convergence and recover higher correlations than the previous state-of-the-art on all standard CCA and Deep CCA benchmarks. This speed allows us to perform a first-of-its-kind PLS analysis of an extremely large bio
    
[^7]: 不排除预测：基于可达集的后续验证方法

    Prediction without Preclusion: Recourse Verification with Reachable Sets. (arXiv:2308.12820v1 [cs.LG])

    [http://arxiv.org/abs/2308.12820](http://arxiv.org/abs/2308.12820)

    这项研究引入了一种称为后续验证的正式测试程序，用于检测模型分配固定预测的情况。通过开发可靠的机制，可以确定给定模型是否能为决策对象提供后续措施，从而解决了模型分配固定预测可能带来的问题。该研究还展示了如何在真实世界的数据集中确保后续措施和对抗鲁棒性，并探讨了在贷款数据集中实现后续措施的不可行性。

    

    机器学习模型常被用于决定谁有资格得到贷款、面试或公共福利。标准技术用于构建这些模型时，会使用关于人的特征，但忽视他们的可操作性。因此，模型可能会分配固定的预测，这意味着被拒绝贷款、面试或福利的消费者可能永久被排除在获得信贷、就业或援助的机会之外。在这项工作中，我们引入了一种正式的测试程序来检测分配固定预测的模型，我们称之为后续验证。我们开发了一套机制可靠地确定给定模型是否能提供对决策对象的后续手段，这些手段由用户指定的可操作性约束确定。我们演示了我们的工具如何在真实世界的数据集中确保后续措施和对抗鲁棒性，并利用它们研究了在真实世界的贷款数据集中实现后续措施的不可行性。我们的结果凸显了模型如何无意中分配固定预测，从而永久禁止使用者获得相关权益。

    Machine learning models are often used to decide who will receive a loan, a job interview, or a public benefit. Standard techniques to build these models use features about people but overlook their actionability. In turn, models can assign predictions that are fixed, meaning that consumers who are denied loans, interviews, or benefits may be permanently locked out from access to credit, employment, or assistance. In this work, we introduce a formal testing procedure to flag models that assign fixed predictions that we call recourse verification. We develop machinery to reliably determine if a given model can provide recourse to its decision subjects from a set of user-specified actionability constraints. We demonstrate how our tools can ensure recourse and adversarial robustness in real-world datasets and use them to study the infeasibility of recourse in real-world lending datasets. Our results highlight how models can inadvertently assign fixed predictions that permanently bar acces
    
[^8]: 通过Stein方法对高斯随机场进行逼近及其在广义随机神经网络中的应用

    Gaussian random field approximation via Stein's method with applications to wide random neural networks. (arXiv:2306.16308v1 [math.PR])

    [http://arxiv.org/abs/2306.16308](http://arxiv.org/abs/2306.16308)

    本研究利用Stein方法推导出Wasserstein距离的上界，通过高斯平滑技术将平滑度量转化为Wasserstein距离。通过特殊化结果，我们获得了广义随机神经网络中对高斯随机场逼近的首个上界。

    

    我们利用Stein方法推导出了基于Wasserstein距离（$W_1$）的上界，该距离是连续随机场与高斯分布之间的距离。我们开发了一种新颖的高斯平滑技术，使我们能够将平滑度量中的上界转化为$W_1$距离。平滑性是基于使用Laplacian算子的幂构建的协方差函数，设计成与Cameron-Martin或Reproducing Kernel Hilbert Space相关联的高斯过程具有易操作的特征。这个特征使我们能够超越之前文献中考虑的一维区间型指标集。通过特化我们的一般结果，我们获得了在任意深度和Lipschitz激活函数的广义随机神经网络中对高斯随机场逼近的首个上界。我们的上界明确地用网络宽度和随机权重的矩来表示。

    We derive upper bounds on the Wasserstein distance ($W_1$), with respect to $\sup$-norm, between any continuous $\mathbb{R}^d$ valued random field indexed by the $n$-sphere and the Gaussian, based on Stein's method. We develop a novel Gaussian smoothing technique that allows us to transfer a bound in a smoother metric to the $W_1$ distance. The smoothing is based on covariance functions constructed using powers of Laplacian operators, designed so that the associated Gaussian process has a tractable Cameron-Martin or Reproducing Kernel Hilbert Space. This feature enables us to move beyond one dimensional interval-based index sets that were previously considered in the literature. Specializing our general result, we obtain the first bounds on the Gaussian random field approximation of wide random neural networks of any depth and Lipschitz activation functions at the random field level. Our bounds are explicitly expressed in terms of the widths of the network and moments of the random wei
    
[^9]: 使用R2效用的多目标优化

    Multi-Objective Optimization Using the R2 Utility. (arXiv:2305.11774v1 [math.OC])

    [http://arxiv.org/abs/2305.11774](http://arxiv.org/abs/2305.11774)

    本文提出将多目标优化问题转化为一组单目标问题进行解决，并介绍了R2效用函数作为适当的目标函数。该效用函数单调且次模，可以使用贪心优化算法计算全局最优解。

    

    多目标优化的目标是确定描述多目标之间最佳权衡的点集合。为了解决这个矢量值优化问题，从业者常常使用标量化函数将多目标问题转化为一组单目标问题。这组标量化问题可以使用传统的单目标优化技术来解决。在这项工作中，我们将这个约定形式化为一个通用的数学框架。我们展示了这种策略如何有效地将原始的多目标优化问题重新转化为定义在集合上的单目标优化问题。针对这个新问题的适当类别的目标函数是R2效用函数，它被定义为标量化优化问题的加权积分。我们证明了这个效用函数是单调的和次模的集合函数，可以通过贪心优化算法有效地计算出全局最优解。

    The goal of multi-objective optimization is to identify a collection of points which describe the best possible trade-offs between the multiple objectives. In order to solve this vector-valued optimization problem, practitioners often appeal to the use of scalarization functions in order to transform the multi-objective problem into a collection of single-objective problems. This set of scalarized problems can then be solved using traditional single-objective optimization techniques. In this work, we formalise this convention into a general mathematical framework. We show how this strategy effectively recasts the original multi-objective optimization problem into a single-objective optimization problem defined over sets. An appropriate class of objective functions for this new problem is the R2 utility function, which is defined as a weighted integral over the scalarized optimization problems. We show that this utility function is a monotone and submodular set function, which can be op
    
[^10]: 反向无味卡尔曼滤波器

    Inverse Unscented Kalman Filter. (arXiv:2304.01698v1 [math.OC])

    [http://arxiv.org/abs/2304.01698](http://arxiv.org/abs/2304.01698)

    本论文提出了针对非线性系统动态的反向无味卡尔曼滤波器（I-UKF）以及基于再生核希尔伯特空间的UKF（RKHS-UKF），用于学习未知的系统模型并估计状态。

    

    设计认知和反对手系统的快速进步促进了反贝叶斯滤波器的发展。在这种情况下，认知“对手”通过随机框架（如卡尔曼滤波器（KF））跟踪其感兴趣的目标。然后，目标或“防御者”使用另一个逆随机滤波器来推断通过对手计算的防御者的前向滤波器估计。对于线性系统，最近已经证明了逆Kalman滤波器（I-KF）在这些反对抗应用中是有效的。本文与之前的工作相反，我们专注于非线性系统动态并制定逆非线性卡尔曼滤波器（I-UKF），以估计防御者的状态并减小线性化误差。然后，我们将这一框架推广到未知系统模型，通过提出基于再生核希尔伯特空间的UKF（RKHS-UKF）来学习系统动态并基于其观测来估计状态。我们的理论分析旨在保证所提出方法的随机稳定性，并进行数值模拟以证实所提出方法的有效性。

    Rapid advances in designing cognitive and counter-adversarial systems have motivated the development of inverse Bayesian filters. In this setting, a cognitive `adversary' tracks its target of interest via a stochastic framework such as a Kalman filter (KF). The target or `defender' then employs another inverse stochastic filter to infer the forward filter estimates of the defender computed by the adversary. For linear systems, inverse Kalman filter (I-KF) has been recently shown to be effective in these counter-adversarial applications. In the paper, contrary to prior works, we focus on non-linear system dynamics and formulate the inverse unscented KF (I-UKF) to estimate the defender's state with reduced linearization errors. We then generalize this framework to an unknown system model by proposing reproducing kernel Hilbert space-based UKF (RKHS-UKF) to learn the system dynamics and estimate the state based on its observations. Our theoretical analyses to guarantee the stochastic stab
    

