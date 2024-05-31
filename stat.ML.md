# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Characterizing Overfitting in Kernel Ridgeless Regression Through the Eigenspectrum](https://rss.arxiv.org/abs/2402.01297) | 我们通过推导核矩阵的特征数界限，增强了核岭回归的测试误差界限。对于多项式谱衰减的核，我们恢复了先前的结果；对于指数谱衰减，我们提出了新的非平凡的界限。我们的研究表明，特征谱衰减多项式的核回归器具有良好的泛化能力，而特征谱指数衰减的核回归器则具有灾难性的过拟合。 |
| [^2] | [On the Last-Iterate Convergence of Shuffling Gradient Methods](https://arxiv.org/abs/2403.07723) | 该论文证明了针对目标函数的洗牌梯度方法最后迭代的收敛速率，弥合了在不同设置中最后迭代的良好性能与现有理论之间的差距。 |
| [^3] | [Near-optimal Per-Action Regret Bounds for Sleeping Bandits](https://arxiv.org/abs/2403.01315) | 该论文提出了针对睡眠臂决策问题的接近最优每次行动遗憾界，通过直接最小化每次行动遗憾，使用Generalized EXP3、EXP3-IX和Tsallis entropy下的FTRL方法，获得了较之现有方法更好的界。 |
| [^4] | [Dataset Fairness: Achievable Fairness on Your Data With Utility Guarantees](https://arxiv.org/abs/2402.17106) | 该论文提出了一种针对数据集特性量身定制的近似公平性-准确性权衡曲线计算方法，能够有效减轻训练多个模型的计算负担并提供了严格的统计保证 |
| [^5] | [Position Paper: Challenges and Opportunities in Topological Deep Learning](https://arxiv.org/abs/2402.08871) | 拓扑深度学习将拓扑特征引入深度学习模型，可作为图表示学习和几何深度学习的补充，给各种机器学习环境提供了自然选择。本文讨论了拓扑深度学习中的开放问题，并提出了未来的研究机会。 |
| [^6] | [Thresholded Oja does Sparse PCA?](https://arxiv.org/abs/2402.07240) | 阈值和重新归一化Oja算法的输出可获得一个接近最优的错误率，与未经阈值处理的Oja向量相比，这大大减小了误差。 |
| [^7] | [Sharp Rates in Dependent Learning Theory: Avoiding Sample Size Deflation for the Square Loss](https://arxiv.org/abs/2402.05928) | 本文研究了依赖学习理论中的尖锐率，主要是为了避免样本大小缩减对方差产生影响。当假设类别的拓扑结构符合某些条件时，经验风险最小化者的性能与类别的复杂性和二阶统计量有关。 |
| [^8] | [Fast and Regret Optimal Best Arm Identification: Fundamental Limits and Low-Complexity Algorithms.](http://arxiv.org/abs/2309.00591) | 本文提出了一种名为 ROBAI 的算法，旨在快速识别并选择最佳臂，并在一系列连续回合中最大化奖励。该算法在预定停止时间和自适应停止时间要求下均实现了渐进最优遗憾，并且在预定停止时间下仅需 $\mathcal{O}(\log T)$ 回合即可选择最佳臂，在自适应停止时间下仅需 $\mathcal{O}(\log^2 T)$ 回合即可选择最佳臂。 |
| [^9] | [Neural Networks for Extreme Quantile Regression with an Application to Forecasting of Flood Risk.](http://arxiv.org/abs/2208.07590) | 本文提出了一种结合神经网络和极值理论的EQRN模型，它能够在存在复杂预测变量相关性的情况下进行外推，并且能够应用于洪水风险预测中，提供一天前回归水平和超出概率的预测。 |

# 详细

[^1]: 通过特征谱表征核岭回归的过拟合

    Characterizing Overfitting in Kernel Ridgeless Regression Through the Eigenspectrum

    [https://rss.arxiv.org/abs/2402.01297](https://rss.arxiv.org/abs/2402.01297)

    我们通过推导核矩阵的特征数界限，增强了核岭回归的测试误差界限。对于多项式谱衰减的核，我们恢复了先前的结果；对于指数谱衰减，我们提出了新的非平凡的界限。我们的研究表明，特征谱衰减多项式的核回归器具有良好的泛化能力，而特征谱指数衰减的核回归器则具有灾难性的过拟合。

    

    我们推导了核矩阵的条件数的新界限，然后利用这些界限增强了在固定输入维度的过参数化区域中核岭回归的现有非渐近测试误差界限。对于具有多项式谱衰减的核，我们恢复了先前工作的界限；对于指数衰减，我们的界限是非平凡和新颖的。我们对过拟合的结论是双重的：(i) 谱衰减多项式的核回归器必须在存在噪声标记的训练数据的情况下得到很好的泛化；这些模型表现出所谓的温和过拟合；(ii) 如果任何核岭回归器的特征谱指数衰减，则其泛化差，即表现出灾难性过拟合。这增加了核岭回归器表现出良性过拟合的可用特征谱衰减次多项式的极端情况的表征。我们的分析结合了新的随机矩阵理论(RMT)。

    We derive new bounds for the condition number of kernel matrices, which we then use to enhance existing non-asymptotic test error bounds for kernel ridgeless regression in the over-parameterized regime for a fixed input dimension. For kernels with polynomial spectral decay, we recover the bound from previous work; for exponential decay, our bound is non-trivial and novel.   Our conclusion on overfitting is two-fold: (i) kernel regressors whose eigenspectrum decays polynomially must generalize well, even in the presence of noisy labeled training data; these models exhibit so-called tempered overfitting; (ii) if the eigenspectrum of any kernel ridge regressor decays exponentially, then it generalizes poorly, i.e., it exhibits catastrophic overfitting. This adds to the available characterization of kernel ridge regressors exhibiting benign overfitting as the extremal case where the eigenspectrum of the kernel decays sub-polynomially. Our analysis combines new random matrix theory (RMT) te
    
[^2]: 关于洗牌梯度方法的最后迭代收敛性

    On the Last-Iterate Convergence of Shuffling Gradient Methods

    [https://arxiv.org/abs/2403.07723](https://arxiv.org/abs/2403.07723)

    该论文证明了针对目标函数的洗牌梯度方法最后迭代的收敛速率，弥合了在不同设置中最后迭代的良好性能与现有理论之间的差距。

    

    洗牌梯度方法，也被称为无替换的随机梯度下降（SGD），在实践中被广泛应用，特别包括三种流行算法：Random Reshuffle（RR）、Shuffle Once（SO）和Incremental Gradient（IG）。与经验成功相比，长期以来对于洗牌梯度方法的理论保证并不充分了解。最近，只为凸函数的平均迭代和强凸问题的最后迭代（以平方距离为度量）建立了收敛速率。然而，当将函数值差作为收敛准则时，现有理论无法解释在不同设置中（例如受约束的优化）最后迭代的良好性能。为了弥合这种实践与理论之间的差距，我们针对目标函数证明了洗牌梯度方法最后迭代的收敛速率。

    arXiv:2403.07723v1 Announce Type: new  Abstract: Shuffling gradient methods, which are also known as stochastic gradient descent (SGD) without replacement, are widely implemented in practice, particularly including three popular algorithms: Random Reshuffle (RR), Shuffle Once (SO), and Incremental Gradient (IG). Compared to the empirical success, the theoretical guarantee of shuffling gradient methods was not well-understanding for a long time. Until recently, the convergence rates had just been established for the average iterate for convex functions and the last iterate for strongly convex problems (using squared distance as the metric). However, when using the function value gap as the convergence criterion, existing theories cannot interpret the good performance of the last iterate in different settings (e.g., constrained optimization). To bridge this gap between practice and theory, we prove last-iterate convergence rates for shuffling gradient methods with respect to the objectiv
    
[^3]: 睡眠臂决策问题中接近最优的每次行动遗憾界

    Near-optimal Per-Action Regret Bounds for Sleeping Bandits

    [https://arxiv.org/abs/2403.01315](https://arxiv.org/abs/2403.01315)

    该论文提出了针对睡眠臂决策问题的接近最优每次行动遗憾界，通过直接最小化每次行动遗憾，使用Generalized EXP3、EXP3-IX和Tsallis entropy下的FTRL方法，获得了较之现有方法更好的界。

    

    我们推导了针对睡眠臂决策问题的接近最优每次行动遗憾界，其中敌手选择每轮可用臂的集合和它们的损失。在每轮至多有 $A$ 个可用臂的 $K$ 个总臂的情况下，已知的最好上界为 $O(K\sqrt{TA\ln{K}})$，通过间接最小化内部睡眠遗憾获得。与极小值 $\Omega(\sqrt{TA})$ 下界相比，这个上界包含额外的乘数因子 $K\ln{K}$。我们通过直接最小化每次行动遗憾，使用EXP3、EXP3-IX和带有Tsallis熵的FTRL的推广版本，从而获得了顺序为 $O(\sqrt{TA\ln{K}})$ 和 $O(\sqrt{T\sqrt{AK}})$ 的接近最优界。我们将结果扩展到了从睡眠专家获得建议的臂决策问题设置，同时推广了EXP4。这为现有的多个自适应和跟踪遗憾界的新证明铺平了道路。

    arXiv:2403.01315v1 Announce Type: new  Abstract: We derive near-optimal per-action regret bounds for sleeping bandits, in which both the sets of available arms and their losses in every round are chosen by an adversary. In a setting with $K$ total arms and at most $A$ available arms in each round over $T$ rounds, the best known upper bound is $O(K\sqrt{TA\ln{K}})$, obtained indirectly via minimizing internal sleeping regrets. Compared to the minimax $\Omega(\sqrt{TA})$ lower bound, this upper bound contains an extra multiplicative factor of $K\ln{K}$. We address this gap by directly minimizing the per-action regret using generalized versions of EXP3, EXP3-IX and FTRL with Tsallis entropy, thereby obtaining near-optimal bounds of order $O(\sqrt{TA\ln{K}})$ and $O(\sqrt{T\sqrt{AK}})$. We extend our results to the setting of bandits with advice from sleeping experts, generalizing EXP4 along the way. This leads to new proofs for a number of existing adaptive and tracking regret bounds for 
    
[^4]: 数据集公平性：在您的数据上实现具有效用保证的公平性

    Dataset Fairness: Achievable Fairness on Your Data With Utility Guarantees

    [https://arxiv.org/abs/2402.17106](https://arxiv.org/abs/2402.17106)

    该论文提出了一种针对数据集特性量身定制的近似公平性-准确性权衡曲线计算方法，能够有效减轻训练多个模型的计算负担并提供了严格的统计保证

    

    在机器学习公平性中，训练能够最小化不同敏感群体之间差异的模型通常会导致准确性下降，这种现象被称为公平性-准确性权衡。这种权衡的严重程度基本取决于数据集的特性，如数据集的不均衡或偏见。因此，在数据集之间使用统一的公平性要求仍然值得怀疑，并且往往会导致效用极低的模型。为了解决这个问题，我们提出了一种针对单个数据集量身定制的近似公平性-准确性权衡曲线的计算效率高的方法，该方法支持严格的统计保证。通过利用You-Only-Train-Once（YOTO）框架，我们的方法减轻了在逼近权衡曲线时需要训练多个模型的计算负担。此外，我们通过在该曲线周围引入置信区间来量化我们近似值的不确定性，

    arXiv:2402.17106v1 Announce Type: cross  Abstract: In machine learning fairness, training models which minimize disparity across different sensitive groups often leads to diminished accuracy, a phenomenon known as the fairness-accuracy trade-off. The severity of this trade-off fundamentally depends on dataset characteristics such as dataset imbalances or biases. Therefore using a uniform fairness requirement across datasets remains questionable and can often lead to models with substantially low utility. To address this, we present a computationally efficient approach to approximate the fairness-accuracy trade-off curve tailored to individual datasets, backed by rigorous statistical guarantees. By utilizing the You-Only-Train-Once (YOTO) framework, our approach mitigates the computational burden of having to train multiple models when approximating the trade-off curve. Moreover, we quantify the uncertainty in our approximation by introducing confidence intervals around this curve, offe
    
[^5]: 位置论文：拓扑深度学习中的挑战与机遇

    Position Paper: Challenges and Opportunities in Topological Deep Learning

    [https://arxiv.org/abs/2402.08871](https://arxiv.org/abs/2402.08871)

    拓扑深度学习将拓扑特征引入深度学习模型，可作为图表示学习和几何深度学习的补充，给各种机器学习环境提供了自然选择。本文讨论了拓扑深度学习中的开放问题，并提出了未来的研究机会。

    

    拓扑深度学习是一个快速发展的领域，它利用拓扑特征来理解和设计深度学习模型。本文认为，通过融入拓扑概念，拓扑深度学习可以补充图表示学习和几何深度学习，并成为各种机器学习环境下的自然选择。为此，本文讨论了拓扑深度学习中的开放问题，涵盖了从实用益处到理论基础的各个方面。针对每个问题，它概述了潜在的解决方案和未来的研究机会。同时，本文也是对科学界的邀请，希望积极参与拓扑深度学习研究，开发这个新兴领域的潜力。

    arXiv:2402.08871v1 Announce Type: new Abstract: Topological deep learning (TDL) is a rapidly evolving field that uses topological features to understand and design deep learning models. This paper posits that TDL may complement graph representation learning and geometric deep learning by incorporating topological concepts, and can thus provide a natural choice for various machine learning settings. To this end, this paper discusses open problems in TDL, ranging from practical benefits to theoretical foundations. For each problem, it outlines potential solutions and future research opportunities. At the same time, this paper serves as an invitation to the scientific community to actively participate in TDL research to unlock the potential of this emerging field.
    
[^6]: 阈值Oja是否适用于稀疏PCA？

    Thresholded Oja does Sparse PCA?

    [https://arxiv.org/abs/2402.07240](https://arxiv.org/abs/2402.07240)

    阈值和重新归一化Oja算法的输出可获得一个接近最优的错误率，与未经阈值处理的Oja向量相比，这大大减小了误差。

    

    我们考虑了当比值$d/n \rightarrow c > 0$时稀疏主成分分析（PCA）的问题。在离线设置下，关于稀疏PCA的最优率已经有很多研究，其中所有数据都可以用于多次传递。相比之下，当人口特征向量是$s$-稀疏时，具有$O(d)$存储和$O(nd)$时间复杂度的流算法通常要求强初始化条件，否则会有次优错误。我们展示了一种简单的算法，对Oja算法的输出（Oja向量）进行阈值和重新归一化，从而获得接近最优的错误率。这非常令人惊讶，因为没有阈值，Oja向量的误差很大。我们的分析集中在限制未归一化的Oja向量的项上，这涉及将一组独立随机矩阵的乘积在随机初始向量上的投影。 这是非平凡且新颖的，因为以前的Oja算法分析没有考虑这一点。

    arXiv:2402.07240v2 Announce Type: cross  Abstract: We consider the problem of Sparse Principal Component Analysis (PCA) when the ratio $d/n \rightarrow c > 0$. There has been a lot of work on optimal rates on sparse PCA in the offline setting, where all the data is available for multiple passes. In contrast, when the population eigenvector is $s$-sparse, streaming algorithms that have $O(d)$ storage and $O(nd)$ time complexity either typically require strong initialization conditions or have a suboptimal error. We show that a simple algorithm that thresholds and renormalizes the output of Oja's algorithm (the Oja vector) obtains a near-optimal error rate. This is very surprising because, without thresholding, the Oja vector has a large error. Our analysis centers around bounding the entries of the unnormalized Oja vector, which involves the projection of a product of independent random matrices on a random initial vector. This is nontrivial and novel since previous analyses of Oja's al
    
[^7]: 依赖学习理论中的尖锐率：避免样本大小缩减的平方损失

    Sharp Rates in Dependent Learning Theory: Avoiding Sample Size Deflation for the Square Loss

    [https://arxiv.org/abs/2402.05928](https://arxiv.org/abs/2402.05928)

    本文研究了依赖学习理论中的尖锐率，主要是为了避免样本大小缩减对方差产生影响。当假设类别的拓扑结构符合某些条件时，经验风险最小化者的性能与类别的复杂性和二阶统计量有关。

    

    本文研究了具有依赖性（β-混合）数据和平方损失的统计学习，在一个假设类别Φ_p的子集F中，其中Φ_p是范数∥f∥_Φ_p≡sup_m≥1 m^{-1/p}∥f∥_L^m，其中p∈[2，∞]。我们的研究动机是在具有依赖性数据的学习中寻找尖锐的噪声交互项或方差代理。在没有任何可实现性假设的情况下，典型的非渐近结果显示出方差代理通过底层协变量过程的混合时间进行了乘积缩减。我们证明，只要在我们的假设类别F上，L^2和Φ_p的拓扑是可比较的，即Φ_p是一个弱亚高斯类别：∥f∥_Φ_p≲∥f∥_L^2^η，其中η∈(0，1]，经验风险最小化者在其主导项中只实现了一种只依赖于类别复杂性和二阶统计量的速率。我们的结果适用于许多依赖性数据模型。

    In this work, we study statistical learning with dependent ($\beta$-mixing) data and square loss in a hypothesis class $\mathscr{F}\subset L_{\Psi_p}$ where $\Psi_p$ is the norm $\|f\|_{\Psi_p} \triangleq \sup_{m\geq 1} m^{-1/p} \|f\|_{L^m} $ for some $p\in [2,\infty]$. Our inquiry is motivated by the search for a sharp noise interaction term, or variance proxy, in learning with dependent data. Absent any realizability assumption, typical non-asymptotic results exhibit variance proxies that are deflated \emph{multiplicatively} by the mixing time of the underlying covariates process. We show that whenever the topologies of $L^2$ and $\Psi_p$ are comparable on our hypothesis class $\mathscr{F}$ -- that is, $\mathscr{F}$ is a weakly sub-Gaussian class: $\|f\|_{\Psi_p} \lesssim \|f\|_{L^2}^\eta$ for some $\eta\in (0,1]$ -- the empirical risk minimizer achieves a rate that only depends on the complexity of the class and second order statistics in its leading term. Our result holds whether t
    
[^8]: 快速和遗憾最小的最佳臂识别：基本限制和低复杂度算法

    Fast and Regret Optimal Best Arm Identification: Fundamental Limits and Low-Complexity Algorithms. (arXiv:2309.00591v1 [cs.LG])

    [http://arxiv.org/abs/2309.00591](http://arxiv.org/abs/2309.00591)

    本文提出了一种名为 ROBAI 的算法，旨在快速识别并选择最佳臂，并在一系列连续回合中最大化奖励。该算法在预定停止时间和自适应停止时间要求下均实现了渐进最优遗憾，并且在预定停止时间下仅需 $\mathcal{O}(\log T)$ 回合即可选择最佳臂，在自适应停止时间下仅需 $\mathcal{O}(\log^2 T)$ 回合即可选择最佳臂。

    

    本文考虑具有双重目标的随机多臂老虎机(MAB)问题：(i) 快速识别并选择最佳臂，以及(ii) 在一系列T个连续回合中最大化奖励。尽管每个目标都已经得到了独立的深入研究，即(i)的最佳臂识别和(ii)的遗憾最小化，但是同时实现这两个目标仍然是一个开放的问题，尽管它在实践中非常重要。本文引入了“遗憾最小化的最佳臂识别”(ROBAI)，旨在实现这两个双重目标。为了解决具有预定停止时间和自适应停止时间要求的ROBAI，我们分别提出了$\mathsf{EOCP}$算法及其变体，不仅在高斯老虎机和一般老虎机中达到了渐进最优遗憾，而且在预定停止时间下，在$\mathcal{O}(\log T)$回合内选择了最佳臂，在自适应停止时间下，选择了最佳臂在$\mathcal{O}(\log^2 T)$回合内。

    This paper considers a stochastic multi-armed bandit (MAB) problem with dual objectives: (i) quick identification and commitment to the optimal arm, and (ii) reward maximization throughout a sequence of $T$ consecutive rounds. Though each objective has been individually well-studied, i.e., best arm identification for (i) and regret minimization for (ii), the simultaneous realization of both objectives remains an open problem, despite its practical importance. This paper introduces \emph{Regret Optimal Best Arm Identification} (ROBAI) which aims to achieve these dual objectives. To solve ROBAI with both pre-determined stopping time and adaptive stopping time requirements, we present the $\mathsf{EOCP}$ algorithm and its variants respectively, which not only achieve asymptotic optimal regret in both Gaussian and general bandits, but also commit to the optimal arm in $\mathcal{O}(\log T)$ rounds with pre-determined stopping time and $\mathcal{O}(\log^2 T)$ rounds with adaptive stopping ti
    
[^9]: 极端分位数回归的神经网络与洪水风险预测应用

    Neural Networks for Extreme Quantile Regression with an Application to Forecasting of Flood Risk. (arXiv:2208.07590v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.07590](http://arxiv.org/abs/2208.07590)

    本文提出了一种结合神经网络和极值理论的EQRN模型，它能够在存在复杂预测变量相关性的情况下进行外推，并且能够应用于洪水风险预测中，提供一天前回归水平和超出概率的预测。

    

    针对极端事件的风险评估需要准确估计超出历史观测范围的高分位数。当风险依赖于观测预测变量的值时，回归技术用于在预测空间中进行插值。我们提出了EQRN模型，它将神经网络和极值理论的工具结合起来，形成一种能够在复杂预测变量相关性存在的情况下进行外推的方法。神经网络可以自然地将数据中的附加结构纳入其中。我们开发了EQRN的循环版本，能够捕捉时间序列中复杂的顺序相关性。我们将这种方法应用于瑞士Aare流域的洪水风险预测。它利用空间和时间上的多个协变量信息，提供一天前回归水平和超出概率的预测。这个输出补充了传统极值分析的静态回归水平，并且预测能够适应分布变化。

    Risk assessment for extreme events requires accurate estimation of high quantiles that go beyond the range of historical observations. When the risk depends on the values of observed predictors, regression techniques are used to interpolate in the predictor space. We propose the EQRN model that combines tools from neural networks and extreme value theory into a method capable of extrapolation in the presence of complex predictor dependence. Neural networks can naturally incorporate additional structure in the data. We develop a recurrent version of EQRN that is able to capture complex sequential dependence in time series. We apply this method to forecasting of flood risk in the Swiss Aare catchment. It exploits information from multiple covariates in space and time to provide one-day-ahead predictions of return levels and exceedances probabilities. This output complements the static return level from a traditional extreme value analysis and the predictions are able to adapt to distribu
    

