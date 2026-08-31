# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning a Size-Weight Frontier for Synthetic-Augmented Inference](https://arxiv.org/abs/2608.28576) | 提出合成增强推断框架，通过从历史任务中学习“规模-权重前沿”，为所有规模-权重配置提供有限样本覆盖保证，在真实数据稀缺时安全利用合成数据并显著收窄置信区间。 |
| [^2] | [Learning between the peaks: sharp asymptotics for kernel ridge regression under power-law anisotropy](https://arxiv.org/abs/2608.28564) | 该论文针对幂律各向异性高斯数据下的核岭回归，推导出核谱与泛化误差的渐近精确表达式，揭示了弱各向异性会使方差峰值随 α 增大而逐渐衰减、且与主方向对齐的目标的偏差在分数样本复杂度处下降并与插值峰值解耦，而强各向异性（α>1）则会改变有效维数。 |
| [^3] | [Generalized Splines and Gaussian Processes](https://arxiv.org/abs/2608.28446) | 本章将有限维高斯线性逆问题中“最小均方误差估计等价于正则化最小二乘拟合”的经典结论推广到无穷维设定，建立了广义样条与核空间上广义高斯过程之间的对应等价关系。 |
| [^4] | [Localizing Global Discrepancies: Marginal Contributions and Contextual Anomaly Detection](https://arxiv.org/abs/2608.28375) | 该论文提出了一个将全局分布差异定位到具体观测值的框架，通过为每个观测分配其在随机统计情境中的边际贡献，统一了重采样诊断、数据估值与事件级异常检测，并由此获得更高效的估计量。 |
| [^5] | [I-FLOP: Fast Learning of Order and Parents from Interventional Data](https://arxiv.org/abs/2608.28245) | I-FLOP将FLOP算法从观测数据扩展到干预数据场景，通过将干预BIC评分适配到基于Cholesky的迭代评分更新机制中，实现了兼具速度优势与理论保证（可恢复正确干预马尔可夫等价类）的快速因果结构学习。 |
| [^6] | [Performative Privacy: When Differential Privacy Maximizes Utility](https://arxiv.org/abs/2608.28198) | 该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。 |
| [^7] | [Conformal Risk-Averse Decision Making with Optimized Certainty Equivalent Risk Control](https://arxiv.org/abs/2608.28179) | 该论文提出了基于优化确定性等价（OCE）度量的风险规避决策框架，证明CVaR下的最优策略可归结为基于预测集的解，从而为保形预测提供了操作性解释，并针对未知分布设计了基于合成似然模型与留出校准数据的数据驱动校准策略，实现对OCE风险的高概率控制。 |
| [^8] | [Generalized Gibbs Ensemble Weighting for Forecast Combination](https://arxiv.org/abs/2608.28116) | 本文提出了广义吉布斯集成加权（GGEW）概率框架，将预测模型视为专家并通过归一化预测损失的吉布斯式指数变换分配集成权重，进一步扩展出数值稳定、多样性感知与在线自适应的一系列方法，以提升预测组合的稳健性与性能。 |
| [^9] | [Fast Weight Attention for Continual Learning](https://arxiv.org/abs/2608.27763) | 该论文在“写后读”自回归语义下将快速权重记忆与状态空间模型的状态转移统一视为在线学习规则，并推导出面向持续学习前缀预测的归一化一阶更新家族（Falcon 系列回归与内积变体）。 |
| [^10] | [The role of parameter Jacobians in the stability of network outputs](https://arxiv.org/abs/2608.27748) | 该论文将网络动力学与NTK线性化动力学统一归结为希尔伯特空间上的算子半群框架，并针对NTK固定核线性化构造给出了半群扰动的显式有限时间范数界等新的先验估计。 |
| [^11] | [On the Computational and Statistical Efficiency of the Empirical Maximum Entropy on the Mean Method](https://arxiv.org/abs/2608.27705) | 本文将经验均值最大熵（MEM）方法的期望收敛速率从 $O(n^{-1/4})$ 提升至参数化的 $O(n^{-1/2})$，并通过将MEM对偶问题重构为期望风险最小化问题，使其融入现代随机优化框架。 |
| [^12] | [Curvature-Aware Radius Shrinkage for Adaptive Nearest Neighbor Classification](https://arxiv.org/abs/2608.27634) | 提出了几何驱动的CARSANN框架，通过基于形状算子的局部平均曲率估计来自适应收缩最近邻邻域半径，使高曲率区域获得更紧凑的邻域、平坦区域保留更宽的空间支撑，从而让最近邻分类适应流形上变化的局部几何。 |
| [^13] | [Robust model-based clustering via mixtures of multivariate pseudo-Voigt distributions](https://arxiv.org/abs/2608.27606) | 该论文提出了多元伪Voigt分布（高斯与柯西分布的加权凸组合）的有限混合模型，通过EM算法进行参数估计，实现了对重尾数据的鲁棒聚类和异常值检测。 |
| [^14] | [Towards a mathematical theory of superposition](https://arxiv.org/abs/2608.27540) | 该论文利用框架理论与压缩感知工具，首次为神经网络中的叠加现象建立了严格的数学恢复理论，在随机和最坏情况支撑集设定下均证明了特征恢复定理，并精确确定了等角紧框架的恢复阈值。 |
| [^15] | [Optimal Transport for Network Comparison: A Review with Machine Learning Applications](https://arxiv.org/abs/2608.27500) | 本文综述了基于最优传输的网络比较方法，系统梳理了Wasserstein、Gromov-Wasserstein和Bures-Wasserstein三种距离，突出传输方案可解释图间差异的节点来源，并利用拉普拉斯谱为Bures-Wasserstein距离推导高效边界，进而在聚类和时间序列网络任务中验证了这些方法。 |
| [^16] | [On efficiency gains via augmenting a tiny sample with a massive auxiliary sample](https://arxiv.org/abs/2608.26610) | 本文揭示了全似然方法能利用大规模辅助样本实现完全效率增益，而逆概率加权方法受限于目标样本量，并提供了相关理论及神经网络训练应用。 |
| [^17] | [A Deep Zero-Inflated Model of North Atlantic Right Whale Presence To Support Blue Economy Management in the U.S. East Coast](https://arxiv.org/abs/2606.14403) | 本文提出了一种深度零膨胀伯努利模型，联合建模物种存在性与检测概率，有效处理被动声学监测数据中的零膨胀和复杂依赖，为濒危物种保护与蓝色经济管理提供新工具。 |
| [^18] | [Negligible in Size, Significant in Effect: On Scale Vectors in Large Language Models](https://arxiv.org/abs/2605.26895) | 缩放向量虽仅占大语言模型参数的极小部分，但并非用于增强表达能力，而是通过自放大预条件效应改善优化过程，对模型预训练效果至关重要。 |
| [^19] | [More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations](https://arxiv.org/abs/2605.26647) | 本文提出令牌自适应的激活混合（MoA）前馈层设计，通过轻量级输入相关门控混合多个激活函数，并从理论上证明了其表达能力严格超越可学习激活（LA）和固定激活FFN。 |
| [^20] | [Online Learning-to-Defer with Varying Experts](https://arxiv.org/abs/2605.12340) | 本文提出了一种将查询动作的老虎机反馈与动态变化专家池相结合的在线多分类学习延迟算法，实现了次线性真实延迟遗憾 $O(T^{2/3})$，并在集中评分条件下提升至 $O(\sqrt T)$。 |
| [^21] | [Budget-Constrained Causal Bandits: Bridging Uplift Modeling and Sequential Decision-Making](https://arxiv.org/abs/2604.26169) | 该论文提出预算约束因果老虎机（BCCB）在线框架，将个体处理效应学习、不确定性探索与预算节奏控制三者统一起来，并基于拉格朗日松弛的 KKT 条件推导出决策规则，从而解决了冷启动场景下数字广告的预算分配问题。 |
| [^22] | [Robust Assortment Optimization from Observational Data](https://arxiv.org/abs/2602.10696) | 提出了一个鲁棒的数据驱动商品组合优化框架，通过建模顾客选择行为中潜在的分布偏移，克服了传统方法因假设偏好稳定和选择模型正确而在现实中导致的泛化差和收益损失问题。 |
| [^23] | [Autotune: fast, accurate, and automatic tuning parameter selection for Lasso](https://arxiv.org/abs/2512.11139) | 该论文提出autotune方法，通过在回归系数与噪声标准差之间交替优化带惩罚的高斯对数似然，实现Lasso调优参数的全自动选择，在低信噪比情形下比现有方法更快且具有更优的泛化性能和模型选择效果。 |
| [^24] | [Prequential posteriors](https://arxiv.org/abs/2511.17721) | 本文提出基于预测序列损失函数的prequential后验方法，解决了深度生成预测模型因似然函数不可解而无法应用标准贝叶斯数据同化的难题，并证明了其在温和条件下的理论一致性保证。 |
| [^25] | [Probabilistic Symbolic Regression for Equation Discovery via Operator-induced and Regularized Symbolic Forests](https://arxiv.org/abs/2509.19710) | 该论文提出一种概率符号回归框架，将数学表达式表示为符号树集成，通过树拓扑上的正则化先验控制表达式复杂度，并利用基于奥卡姆窗口的后验摘要刻画多个合理符号模型的不确定性，为方程发现提供了兼具精度、简洁性与不确定性量化的统一解决方案。 |
| [^26] | [Off the Normal Path: Learning Spatial Density Models of Node Mobility](https://arxiv.org/abs/2411.10997) | 该论文引入Möbius分布混合模型来学习二维地形上移动节点的稳态空间密度，相比混合密度网络和归一化流等现成方法，提供了更可解释、更简洁且性能相当或更优的模型。 |
| [^27] | [On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling](https://arxiv.org/abs/2402.05098) | 本研究探讨了训练扩散模型以从给定分布中采样的问题，并针对随机控制和采样提出了一种新的探索策略，通过基准测试比较了不同推断方法的相对优劣，并对过去的工作提出了质疑。 |
| [^28] | [Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network.](http://arxiv.org/abs/2305.19366) | 本文提出了在单一生成流网络中联合建模贝叶斯网络结构和参数的方法，包括非离散样本空间，提高了贝叶斯网络局部概率模型的灵活性。 |
| [^29] | [Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets.](http://arxiv.org/abs/2305.17010) | 本文提出了一种名为GFlowNets的机器，可以有效地解决组合优化问题，同时在训练方面进行了优化，结果表明其可以高效地找到高质量的解决方案。 |
| [^30] | [Trajectory balance: Improved credit assignment in GFlowNets.](http://arxiv.org/abs/2201.13259) | GFlowNets使用轨迹平衡作为一种更高效的学习目标，解决了先前学习目标中信用传播效率低下的问题，并且在实验中证明了其在收敛性、生成样本多样性以及鲁棒性方面的优势。 |

# 详细

[^1]: 学习合成增强推断中的规模-权重前沿

    Learning a Size-Weight Frontier for Synthetic-Augmented Inference

    [https://arxiv.org/abs/2608.28576](https://arxiv.org/abs/2608.28576)

    提出合成增强推断框架，通过从历史任务中学习“规模-权重前沿”，为所有规模-权重配置提供有限样本覆盖保证，在真实数据稀缺时安全利用合成数据并显著收窄置信区间。

    

    当真实数据稀缺时，合成数据可以改善统计推断，但简单地将合成样本当作真实数据会引入偏差并导致不可靠的推断。我们开发了一个面向相关任务总体的合成增强推断通用框架。该框架通过合成观测的数量及其权重来刻画合成增强。框架的核心是一个规模-权重前沿，它为每个权重指定了最大的合成样本量，使得所有不大于该样本量的配置都能达到目标任务边际覆盖率。我们从历史任务中估计这一前沿，并对估计前沿上或以下的所有规模-权重配置同时建立了有限样本覆盖保证。在使用大语言模型响应来增强舆论调查数据的实验中，我们的方法实现了目标覆盖率，并大幅收窄了置信区间。

    arXiv:2608.28576v1 Announce Type: cross  Abstract: Synthetic data can improve statistical inference when real data are scarce, but naively treating synthetic samples as real data can introduce bias and lead to unreliable inference. We develop a general framework for synthetic-augmented inference across a population of related tasks. It characterizes synthetic augmentation by the number of synthetic observations and their weight. Central to our framework is a size-weight frontier that specifies, for each weight, the largest synthetic sample size for which all smaller sizes attain the target task-marginal coverage. We estimate this frontier from historical tasks, and establish a finite-sample coverage guarantee simultaneously for all size-weight configurations on or below the estimated frontier. In experiments using large language model responses to augment opinion survey data, our procedure achieves target coverage and substantially narrows confidence intervals.
    
[^2]: 峰间学习：幂律各向异性下核岭回归的精确渐近分析

    Learning between the peaks: sharp asymptotics for kernel ridge regression under power-law anisotropy

    [https://arxiv.org/abs/2608.28564](https://arxiv.org/abs/2608.28564)

    该论文针对幂律各向异性高斯数据下的核岭回归，推导出核谱与泛化误差的渐近精确表达式，揭示了弱各向异性会使方差峰值随 α 增大而逐渐衰减、且与主方向对齐的目标的偏差在分数样本复杂度处下降并与插值峰值解耦，而强各向异性（α>1）则会改变有效维数。

    

    我们研究了各向异性高斯数据下的核岭回归问题，其中对于多项式内积核，输入协方差以指数 α≥0 的幂律衰减。我们在多项式高维区域 n=Θ(d^κ) 下推导出了核谱与泛化误差的渐近精确表达式，揭示了各向异性如何重塑学习曲线。对于弱各向异性（0<α<1），问题在本质上仍是高维的，既保留了各向同性情形的某些特征，又在其他方面有所偏离：方差仍然在整数样本复杂度 κ∈ℕ 处出现峰值，但随着 α 的增大，这些峰值会被逐渐衰减；同时，对于与数据主方向高度对齐的目标函数，偏差会在分数样本复杂度处下降，从而使偏差的转变与插值峰值解耦。对于强各向异性（α>1），有效维数……（原文摘要在此处截断）

    arXiv:2608.28564v1 Announce Type: cross  Abstract: We study kernel ridge regression under anisotropic Gaussian data, where the input covariance decays as a power law with exponent $\alpha\geq 0$ for polynomial inner-product kernels. We derive asymptotically sharp expressions for the kernel spectrum and the generalization error in the polynomial high-dimensional regime $n=\Theta(d^\kappa)$, revealing how anisotropy reshapes the learning curves. For weak anisotropy ($0<\alpha<1$), the problem remains effectively high-dimensional and retains some features of the isotropic case, while departing from it in others: the variance still peaks at integer sample complexities $\kappa\in\mathbb{N}$, but these peaks are progressively damped as $\alpha$ grows; meanwhile, for targets strongly aligned with the data's principal directions, the bias drops at fractional sample complexities, decoupling the bias transitions from the interpolation peaks. For strong anisotropy ($\alpha > 1$), the effective di
    
[^3]: 广义样条与高斯过程

    Generalized Splines and Gaussian Processes

    [https://arxiv.org/abs/2608.28446](https://arxiv.org/abs/2608.28446)

    本章将有限维高斯线性逆问题中“最小均方误差估计等价于正则化最小二乘拟合”的经典结论推广到无穷维设定，建立了广义样条与核空间上广义高斯过程之间的对应等价关系。

    

    对于变量服从高斯分布的有限维线性逆问题，众所周知，最小均方误差估计器表现为正则化最小二乘数据拟合的形式。在本章中，我们证明这一等价性可以推广到一个更为广泛的无穷维设定：其中广义样条充当线性回归器的角色，而核空间 $S$ 上的广义高斯过程则对应于高斯随机向量。这一扩展的范畴在性质上类似于从经典函数概念到分布（也称为“广义函数”）的转变。我们的形式化体系涉及一个白化/正则化算子 $L: S\to S'$，其连续延拓诱导出一个本征希尔伯特空间 $H\subset S'$，该空间在我们的刻画中起着核心作用。本阐述在大部分内容上是自包含的，并且具有极高的普适性与威力。它能够恢复……（原文摘要在此处截断）

    arXiv:2608.28446v1 Announce Type: cross  Abstract: For finite-dimensional linear inverse problems where the variables are Gaussian, it is well-known that the minimum-mean-square error estimator takes the form of a regularized least-squares data fit. In this chapter, we show that this equivalence extends to a much broader infinite-dimensional setting where generalized splines take the role of linear regressors and generalized Gaussian processes on a nuclear space $S$ are the counterpart of Gaussian random vectors. The scope of this extension is of the same nature as the switch from the classic notion of function to that of a distribution, also known as a "generalized function." Our formalism involves a whitening/regularization operator $L: S\to S'$ whose continuous extension induces a native Hilbert space $H\subset S'$ that plays a central role in our characterization. The presentation is self-contained for the most part and remarkably general and powerful. It allows for the recovery of
    
[^4]: 定位全局差异：边际贡献与情境异常检测

    Localizing Global Discrepancies: Marginal Contributions and Contextual Anomaly Detection

    [https://arxiv.org/abs/2608.28375](https://arxiv.org/abs/2608.28375)

    该论文提出了一个将全局分布差异定位到具体观测值的框架，通过为每个观测分配其在随机统计情境中的边际贡献，统一了重采样诊断、数据估值与事件级异常检测，并由此获得更高效的估计量。

    

    全局拟合优度与差异统计量能够判定一个样本偏离了参考分布，但无法识别是哪些观测值导致了这种偏离。我们为这一定位问题开发了一个框架，通过为每个观测值分配其在随机统计情境中的条件贡献或边际贡献。这一方法将重采样诊断与数据估值同投影理论以及事件级异常检测联系起来。对于对称统计量，固定大小的替换与中心化条件定位完全等价。对于U-统计量，添加得分恰好等于第一阶Hoeffding/Hájek贡献；对于光滑分布泛函，其在一阶近似下与影响函数相关；而对于已知背景的无偏MMD，它则恰好简化为MMD见证函数。这一视角还带来了更高效的估计量。匹配情境减除法能够消除与观测对象无关的波动……

    arXiv:2608.28375v1 Announce Type: cross  Abstract: Global goodness-of-fit and discrepancy statistics can establish that a sample departs from a reference distribution without identifying which observations drive the departure. We develop a framework for this localization problem by assigning to each observation its conditional or marginal contribution across random statistical contexts. This connects resampling diagnostics and data valuation to projection theory and event-level anomaly detection. For symmetric statistics, fixed-size replacement is exactly equivalent to centered conditional localization. For U-statistics, the addition score equals the first Hoeffding/H\'ajek contribution; for smooth distributional functionals it is related at leading order to the influence function; and for unbiased known-background MMD it reduces exactly to the MMD witness.   This viewpoint also yields more efficient estimators. Matched-context subtraction removes fluctuations unrelated to the observat
    
[^5]: I-FLOP：基于干预数据的序与父节点快速学习

    I-FLOP: Fast Learning of Order and Parents from Interventional Data

    [https://arxiv.org/abs/2608.28245](https://arxiv.org/abs/2608.28245)

    I-FLOP将FLOP算法从观测数据扩展到干预数据场景，通过将干预BIC评分适配到基于Cholesky的迭代评分更新机制中，实现了兼具速度优势与理论保证（可恢复正确干预马尔可夫等价类）的快速因果结构学习。

    

    我们将Wienöbst等人（2026）近期提出的FLOP（序与父节点快速学习）算法从观测数据扩展到干预数据。特别地，我们采用Hauser和Bühlmann（2012）提出的干预BIC评分，并将其适配到基于Cholesky分解的迭代评分更新框架中，而后者正是FLOP算法速度优势的部分来源。我们证明，在样本极限情况下，I-FLOP能够恢复出与数据生成DAG处于同一干预马尔可夫等价类中的DAG。我们在真实和模拟的干预数据上将I-FLOP与现有的因果结构学习算法进行比较，结果表明I-FLOP在性能和运行时间两方面均表现优异。

    arXiv:2608.28245v1 Announce Type: cross  Abstract: We extend the FLOP (fast learning of order and parents) algorithm recently proposed by Wien\"obst et al. (2026) from observational to interventional data. In particular, we use the interventional BIC score of Hauser and B\"uhlmann (2012), adapting it to be used with the iterative Cholesky-based score updates that are partly responsible for FLOP's speed. We show that, in the sample limit, I-FLOP recovers a DAG in the same interventional Markov equivalence class as the data-generating DAG. We compare I-FLOP to existing causal structure learning algorithms on real and simulated interventional data, where it performs favorably in terms of both performance and run time.
    
[^6]: 表演性隐私：差分隐私何时能最大化效用

    Performative Privacy: When Differential Privacy Maximizes Utility

    [https://arxiv.org/abs/2608.28198](https://arxiv.org/abs/2608.28198)

    该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。

    

    保护隐私的学习通常源于这样一种理念：保护用户数据可以维持信任，从而保持用户参与，进而在长期内提升效用。然而，这一论点迄今为止尚未被形式化。与此同时，表演性学习为研究部署行为会影响其后续观测数据的学习系统提供了一个框架。在本工作中，我们将这两种视角结合起来，提出了“表演性隐私”的概念，即数据泄露会降低未来的用户参与度。我们研究了一个简单模型：智能体反复贡献数据用于均值估计，但当其数据被泄露时可能会退出系统。隐私通过差分隐私机制来实现，从而在估计噪声与未来参与度之间形成权衡。通过对该动态过程的理论研究和数值实验，我们证明了在某些条件下，有限的隐私预算在长期内可以优于非隐私估计。

    arXiv:2608.28198v1 Announce Type: new  Abstract: Privacy-preserving learning is often motivated by the idea that protecting users' data can preserve trust and thus participation, improving utility in the long term. However, this claim has not been formalized so far. In parallel, performative learning provides a framework for studying learning systems whose deployment affects the data they later observe. In this work, we bring these two perspectives together and introduce \emph{performative privacy}, where data leakage reduces future participation. We study a simple model where agents repeatedly contribute data for mean estimation but may leave the system when their data is leaked. Privacy is implemented through differentially private mechanisms, creating a trade-off between estimation noise and future participation. We show, through a theoretical study of the dynamics and numerical experiments, that a finite privacy budget can outperform non-private estimation in the long term when the
    
[^7]: 基于优化确定性等价风险控制的保形风险规避决策

    Conformal Risk-Averse Decision Making with Optimized Certainty Equivalent Risk Control

    [https://arxiv.org/abs/2608.28179](https://arxiv.org/abs/2608.28179)

    该论文提出了基于优化确定性等价（OCE）度量的风险规避决策框架，证明CVaR下的最优策略可归结为基于预测集的解，从而为保形预测提供了操作性解释，并针对未知分布设计了基于合成似然模型与留出校准数据的数据驱动校准策略，实现对OCE风险的高概率控制。

    

    我们研究风险规避决策问题，其中智能体在对真实系统状态不确定的情况下选择动作。风险通过优化确定性等价（OCE）度量来衡量，该度量推广了均值-方差风险和条件风险价值（CVaR）等流行准则。我们在分布已知的情况下刻画了最优策略，并证明对于CVaR，该策略可简化为基于预测集的解，这为保形预测类型的预测集提供了一种操作性的解释。对于分布未知的情况，我们基于似然的合成模型和留出的校准数据，开发了一种数据驱动的校准策略，从而实现对OCE风险的高概率控制。该方法在两个无线波束成形场景中进行了评估。

    arXiv:2608.28179v1 Announce Type: cross  Abstract: We study risk-averse decision making, in which an agent selects actions while being uncertain about the true system state. The risk is measured via optimized certainty equivalent (OCE) metrics, which generalize popular criteria such as mean-variance risk and conditional value-at-risk (CVaR). We characterize the optimal policy under known distributions, and show that it reduces to a prediction set-based solution for the CVaR. This provides an operational interpretation of conformal prediction-type prediction sets. For unknown distributions, we develop a data-driven calibration strategy, based on a synthetic model for the likelihood and held-out calibration data, yielding high-probability control of the OCE risk. The approach is evaluated on two wireless beamforming settings.
    
[^8]: 面向预测组合的广义吉布斯集成加权

    Generalized Gibbs Ensemble Weighting for Forecast Combination

    [https://arxiv.org/abs/2608.28116](https://arxiv.org/abs/2608.28116)

    本文提出了广义吉布斯集成加权（GGEW）概率框架，将预测模型视为专家并通过归一化预测损失的吉布斯式指数变换分配集成权重，进一步扩展出数值稳定、多样性感知与在线自适应的一系列方法，以提升预测组合的稳健性与性能。

    

    当有多个预测模型可用时，预测组合是提高预测性能的可靠方法。简单的聚合规则（如均值、中位数、截尾均值、逆损失加权和指数加权）通常是很强的基线方法，但它们的相对性能可能因数据集、预测时间跨度、部署设置以及基础预测器之间的分歧程度而有所不同。我们提出了广义吉布斯集成加权（GGEW），这是一个概率框架，它将预测模型视为专家，并使用归一化预测损失的吉布斯式指数变换来分配集成权重。该框架通过数值稳定化、多样性感知的得分修正以及在线超参数自适应来扩展这一基本加权规则。GGEW产生了一系列相关方法，包括稳定吉布斯加权、方向性吉布斯-NCL和对称吉布斯-NCL。这些变体共享一个核心……

    arXiv:2608.28116v1 Announce Type: new  Abstract: Forecast combination is a reliable way to improve predictive performance when several forecasting models are available. Simple aggregation rules such as the mean, median, trimmed mean, inverse-loss weighting, and exponential weighting are often strong baselines, but their relative performance can vary across datasets, forecast horizons, deployment settings, and levels of disagreement among base forecasters. We develop Generalized Gibbs Ensemble Weighting (GGEW), a probabilistic framework that treats forecasting models as experts and assigns ensemble weights using a Gibbs-style exponential transformation of normalized predictive loss. The framework extends this basic weighting rule through numerical stabilization, diversity-aware score corrections, and online hyperparameter adaptation. GGEW produces a family of related methods, including Stable Gibbs weighting, Directional Gibbs-NCL, and Symmetric Gibbs-NCL. These variants share one core 
    
[^9]: 面向持续学习的快速权重注意力

    Fast Weight Attention for Continual Learning

    [https://arxiv.org/abs/2608.27763](https://arxiv.org/abs/2608.27763)

    该论文在“写后读”自回归语义下将快速权重记忆与状态空间模型的状态转移统一视为在线学习规则，并推导出面向持续学习前缀预测的归一化一阶更新家族（Falcon 系列回归与内积变体）。

    

    循环快速权重记忆与选择性状态空间模型将不断增长的上下文压缩进固定大小的循环状态中，从而使状态转移成为一种在线学习规则。我们在“写后读”自回归语义下研究这一规则。对于本文所考虑的前缀预测目标，在第 $t$ 步揭示的局部快速记忆样本是前缀对齐对 $(\mathbf{x}_t,\mathbf{y}_t)=(\phi(\mathbf{k}_{t-1}),\mathbf{v}_t)$；常见的同一步关联 $(\phi(\mathbf{k}_t),\mathbf{v}_t)$ 虽然仍满足因果性，但优化的是另一种内部目标。我们为平方误差回归和负内积目标推导了归一化的一阶更新规则：回归家族包括 Falcon-1（标量 NLMS 更新）、Falcon-2（其按列扩展）以及 Falcon-3（滑动窗口小批量更新）；Falcon-1A/Falcon-2A/Falcon-3A 则是相应的内积变体。我们提供了循环的、带掩码的……

    arXiv:2608.27763v1 Announce Type: cross  Abstract: Recurrent fast-weight memories and selective state-space models compress an expanding context into a fixed-size recurrent state, making the state transition an online learning rule. We study this rule under read-after-write autoregressive semantics. For the prefix-prediction objective considered here, the local fast-memory example revealed at step $t$ is the prefix-aligned pair $(\mathbf{x}_t,\mathbf{y}_t)=(\phi(\mathbf{k}_{t-1}),\mathbf{v}_t)$. The common same-step association $(\phi(\mathbf{k}_t),\mathbf{v}_t)$ remains causal, but optimizes a different internal objective. We derive normalized first-order updates for squared-error regression and negative inner-product objectives. The regression family comprises Falcon-1 (a scalar NLMS update), Falcon-2 (its per-column extension), and Falcon-3 (a sliding-window mini-batch update); Falcon-1A/Falcon-2A/Falcon-3A are the corresponding inner-product variants. We provide recurrent, masked-p
    
[^10]: 参数雅可比矩阵在网络输出稳定性中的作用

    The role of parameter Jacobians in the stability of network outputs

    [https://arxiv.org/abs/2608.27748](https://arxiv.org/abs/2608.27748)

    该论文将网络动力学与NTK线性化动力学统一归结为希尔伯特空间上的算子半群框架，并针对NTK固定核线性化构造给出了半群扰动的显式有限时间范数界等新的先验估计。

    

    在网络动力学、学习模型和神经正切核（NTK）的框架下，我们证明了相应的线性化动力学自然地引出一个半群表述。更准确地说，在我们对输入/输出模型的分析中，时间动力学通过希尔伯特空间上线性算子的特殊半群来呈现，并伴随一类相关的半群扰动。在此背景下，我们给出了新的且显式的先验扰动界结果：对于NTK设定中出现的固定核线性化构造，我们证明了相应半群扰动的范数界，其形式为显式的有限时间扰动估计。我们进一步给出了在指定任务空间上的改进结果、Cesàro平均（遍历）比较估计，以及用谱分布条件替代下谱边缘假设的版本。我们还将该比较推广到非自治情形。

    arXiv:2608.27748v1 Announce Type: cross  Abstract: In the framework of network dynamics, learning models, and neural tangent kernels (NTK), we show that the corresponding linearized dynamics leads naturally to a semigroup formulation. More precisely, in our analysis of input/output models, the time-dynamics is presented via special semigroups of linear operators on Hilbert spaces, together with an associated class of semigroup perturbations. In this context, we then present new and explicit a priori perturbation-bound results: for the fixed-kernel linearization constructions arising in the NTK setting, we prove norm-bounds on the corresponding semigroup perturbations, in the form of explicit finite-time perturbation estimates. We further present refinements on prescribed task spaces, Ces\`aro-averaged (ergodic) comparisons estimates, and versions in which the lower spectral edge assumption is replaced by a spectral-distribution condition. We also extend the comparison to nonautonomous 
    
[^11]: 关于经验均值最大熵方法的计算与统计效率

    On the Computational and Statistical Efficiency of the Empirical Maximum Entropy on the Mean Method

    [https://arxiv.org/abs/2608.27705](https://arxiv.org/abs/2608.27705)

    本文将经验均值最大熵（MEM）方法的期望收敛速率从 $O(n^{-1/4})$ 提升至参数化的 $O(n^{-1/2})$，并通过将MEM对偶问题重构为期望风险最小化问题，使其融入现代随机优化框架。

    

    均值最大熵（MEM）方法通过将数据保真度与基于熵的正则化相结合，为求解逆问题提供了一个灵活的计算框架。然而在实践中，先验分布通常是未知的，但可以从数据中进行估计，由此产生了经验MEM方法。我们为经验MEM方法建立了期望意义下 $O(n^{-1/2})$ 的参数化收敛速率，改进了King-Roskamp等人（2026）先前建立的 $O(n^{-1/4})$ 保证。我们的证明基于一种新颖的稳定性分析，即在下层概率测度受扰动时对原始与对偶优化问题的稳定性分析，且仅依赖于凸分析与概率论的基础工具。我们进一步证明，MEM对偶问题可以被重新表述为期望风险最小化问题，从而将MEM纳入现代随机优化框架，并使可扩展的随机……（原文在此处截断）

    arXiv:2608.27705v1 Announce Type: cross  Abstract: The Maximum Entropy on the Mean (MEM) method provides a flexible computational framework for solving inverse problems by combining data fidelity with entropy-based regularization. In practice, however, the prior distribution is typically unknown but can be estimated from data, giving rise to the empirical MEM method. We establish a parametric convergence rate of $O(n^{-1/2})$ in expectation for empirical MEM, improving upon the previously established $O(n^{-1/4})$ guarantee by King-Roskamp et al. (2026). Our proof is based on a novel stability analysis of the primal and dual optimization problems under perturbations of the underlying probability measure, relying only on foundational tools from convex analysis and probability. We further show that the MEM dual problem admits a reformulation as an expected risk minimization problem, thereby placing MEM within the modern framework of stochastic optimization and enabling scalable stochasti
    
[^12]: 面向自适应最近邻分类的曲率感知半径收缩方法

    Curvature-Aware Radius Shrinkage for Adaptive Nearest Neighbor Classification

    [https://arxiv.org/abs/2608.27634](https://arxiv.org/abs/2608.27634)

    提出了几何驱动的CARSANN框架，通过基于形状算子的局部平均曲率估计来自适应收缩最近邻邻域半径，使高曲率区域获得更紧凑的邻域、平坦区域保留更宽的空间支撑，从而让最近邻分类适应流形上变化的局部几何。

    

    最近邻分类从根本上依赖于如何定义局部性，然而传统的 k-NN 在整个特征空间中施加了相同的邻域基数。这一假设对于局部几何在底层流形上变化显著的数据而言可能并不适用。我们提出了曲率感知半径收缩自适应最近邻分类，这是一种几何驱动的框架，能够根据局部几何复杂度自适应调整每个邻域的空间支撑范围。CARSANN 首先使用 TwoNN 估计内在维度，并通过主成分分析构建内在表示。随后基于形状算子的公式估计局部平均曲率，并以此控制邻域尺度：高曲率区域受到更强的半径收缩，而近似平坦的区域则保留更宽的空间支撑。与仅修改邻居数量的方法不同……

    arXiv:2608.27634v1 Announce Type: new  Abstract: Nearest neighbor classification relies fundamentally on how locality is defined, yet conventional $k$-NN imposes the same neighborhood cardinality throughout the feature space. This assumption can be inadequate for data whose local geometry varies substantially across the underlying manifold. We introduce Curvature-Aware Radius Shrinkage for Adaptive Nearest Neighbor Classification (CARSANN), a geometry-driven framework that adapts the spatial support of each neighborhood according to local geometric complexity. CARSANN first estimates intrinsic dimensionality using TwoNN and constructs an intrinsic representation through principal component analysis. Local mean curvature is then estimated using a shape-operator-based formulation and controls neighborhood scale: highly curved regions receive stronger radius shrinkage, whereas approximately flat regions retain broader spatial support. Unlike methods that modify only the number of neighbor
    
[^13]: 基于多元伪Voigt分布混合的鲁棒模型聚类

    Robust model-based clustering via mixtures of multivariate pseudo-Voigt distributions

    [https://arxiv.org/abs/2608.27606](https://arxiv.org/abs/2608.27606)

    该论文提出了多元伪Voigt分布（高斯与柯西分布的加权凸组合）的有限混合模型，通过EM算法进行参数估计，实现了对重尾数据的鲁棒聚类和异常值检测。

    

    我们提出了伪Voigt轮廓的多元扩展——即高斯分布与柯西分布的加权凸组合——并将其置于有限混合建模框架中，用于鲁棒的基于模型的聚类和异常值检测。为确保模型的简洁性和簇内的一致性，在高斯成分和柯西成分之间施加了共享的位置和尺度参数。参数估计通过期望最大化（EM）算法进行，并借助潜变量实现高效的基于似然的推断。通过模拟研究和真实数据应用评估了所提出模型的性能。与已有的鲁棒模型（包括污染正态分布混合模型）进行了比较，以展示该模型的聚类精度和异常值检测能力。结果表明，该框架对于具有重尾特征的数据特别有效。

    arXiv:2608.27606v1 Announce Type: cross  Abstract: We propose a multivariate extension of the pseudo-Voigt profile-a weighted convex combination of Gaussian and Cauchy distributions-within a finite mixture modeling framework for robust model-based clustering and outlier detection. To ensure parsimony and coherence within clusters, shared location and scale parameters are imposed between the Gaussian and Cauchy components. Parameter estimation is carried out via an Expectation Maximization algorithm, with latent variables facilitating efficient likelihood-based inference. The performance of the proposed model is evaluated through simulation studies and applications to real-world data. Comparisons with established robust models, including mixtures of contaminated normal distributions, are provided to illustrate the model's clustering accuracy and outlier detection capabilities. The framework is shown to be particularly effective for data characterized by heavy-tailed behavior.
    
[^14]: 迈向叠加的数学理论

    Towards a mathematical theory of superposition

    [https://arxiv.org/abs/2608.27540](https://arxiv.org/abs/2608.27540)

    该论文利用框架理论与压缩感知工具，首次为神经网络中的叠加现象建立了严格的数学恢复理论，在随机和最坏情况支撑集设定下均证明了特征恢复定理，并精确确定了等角紧框架的恢复阈值。

    

    我们利用框架理论和压缩感知的工具，为神经网络中的叠加现象建立了一套数学理论。在我们的模型中，一个由激活特征构成的稀疏二值向量 \(x\) 通过一个过完备字典 \(W\) 进行编码，特征恢复通过应用 \(\operatorname{ReLU}(W^\top W x+b)\)（配合适当的偏置向量 \(b\)）来实现。我们为该模型证明了多个恢复定理。在随机支撑集设定下，我们针对近似紧凑、低相干性的字典建立了高概率支撑恢复的结果，当期望稀疏度达到 \(d/\log n\) 量级时仍能提供保证。在最坏情况支撑集设定下，我们给出了一个锐利且可计算的判据，用以确定哪些稀疏度水平允许支撑恢复。我们将该判据应用于高斯随机矩阵和等角紧框架。对于 \(n>d+1\) 的实等角紧框架，我们以相干性为参数确定了精确的恢复阈值。

    arXiv:2608.27540v1 Announce Type: cross  Abstract: We develop a mathematical theory of superposition in neural networks using tools from frame theory and compressed sensing. In our model, a sparse binary vector \(x\) of active features is encoded through an overcomplete dictionary \(W\), and feature recovery is performed by applying \(\operatorname{ReLU}(W^\top W x+b)\) with an appropriate bias vector \(b\). We prove several recovery theorems for this model. In the random-support setting, we establish high-probability support recovery for nearly tight, low-coherence dictionaries, with guarantees when the expected sparsity is up to order \(d/\log n\). In the worst-case support setting, we give a sharp and computable criterion for which sparsity levels permit support recovery. We apply this criterion to Gaussian random matrices and equiangular tight frames. For real equiangular tight frames with \(n>d+1\), we determine the exact recovery threshold in terms of the coherence. The proof of 
    
[^15]: 用于网络比较的最优传输：综述及其机器学习应用

    Optimal Transport for Network Comparison: A Review with Machine Learning Applications

    [https://arxiv.org/abs/2608.27500](https://arxiv.org/abs/2608.27500)

    本文综述了基于最优传输的网络比较方法，系统梳理了Wasserstein、Gromov-Wasserstein和Bures-Wasserstein三种距离，突出传输方案可解释图间差异的节点来源，并利用拉普拉斯谱为Bures-Wasserstein距离推导高效边界，进而在聚类和时间序列网络任务中验证了这些方法。

    

    运用最优传输进行网络比较是网络科学中一个不断发展的研究领域。与标准的图度量不同，最优传输不仅计算网络间的相异性，还提供一个传输方案来解释一张图如何演变为另一张图。本文综述了如何利用三种主要距离——Wasserstein距离、Gromov-Wasserstein距离和Bures-Wasserstein距离——来比较无向无权图。我们考察了通过节点特征概率分布在一维情形下Wasserstein距离的闭式解，并展示了Wasserstein距离和Gromov-Wasserstein距离的传输方案如何捕捉图扰动后具体哪些节点影响了距离。对于Bures-Wasserstein距离，我们利用拉普拉斯谱推导出上界，从而避免了完整的谱分解。最后，我们使用合成网络数据集评估这些距离在聚类任务中的表现，并应用于真实世界的时间序列网络数据。

    arXiv:2608.27500v1 Announce Type: cross  Abstract: Network comparison using optimal transport is a growing area of research in network science. Unlike standard graph metrics, optimal transport computes both network dissimilarity and a transport plan that explains how one graph morphs into another. In this paper, we review how optimal transport compares undirected, unweighted graphs using three primary distances: the Wasserstein, Gromov-Wasserstein, and Bures-Wasserstein distances. We examine the closed form of the Wasserstein distance in one dimension via node feature probability distributions, and show how the transport plans of the Wasserstein and Gromov-Wasserstein distances capture which specific nodes influence the distance after graph perturbation. For the Bures-Wasserstein distance, we derive bounds using Laplacian spectra to bypass full spectral decompositions. Finally, we evaluate these distances using a synthetic network dataset for clustering and a real-world time series net
    
[^16]: 关于通过大规模辅助样本增强极小目标样本的效率增益研究

    On efficiency gains via augmenting a tiny sample with a massive auxiliary sample

    [https://arxiv.org/abs/2608.26610](https://arxiv.org/abs/2608.26610)

    本文揭示了全似然方法能利用大规模辅助样本实现完全效率增益，而逆概率加权方法受限于目标样本量，并提供了相关理论及神经网络训练应用。

    

    arXiv:2608.26610v1 公告类型：交叉 摘要：在本文中，我们研究了用大规模辅助样本增强极小目标样本的问题。利用Tukey分解，有两种常用方法：逆概率加权（IPW）和全似然（FL）方法。我们表明，IPW方法受限于目标样本量小的问题，而FL方法可以以大规模辅助样本量的速率估计某些模型参数，这一现象我们称之为完全效率增益。我们研究了指数族和指数族混合模型下完全效率增益的理论。我们还研究了非参数程序下IPW方法的效率增益，并展示了它如何达到目标样本量的参数速率。作为附注，我们还讨论了如何利用FL同时训练目标分布和几率模型的神经网络模型。

    arXiv:2608.26610v1 Announce Type: cross  Abstract: In this paper, we study the problem of augmenting a tiny target sample with a massive auxiliary sample. Utilizing Tukey's factorization, there are two popular approaches: the inverse probability weight (IPW) and the full-likelihood (FL) methods. We show that the IPW approach suffers from the limited target sample problem while the FL method may estimate some model parameters at the rate of the massive auxiliary sample size, a phenomenon we call full efficiency gain. We study the theory behind the full efficiency gain for exponential families and mixtures of exponential families. We also study the efficiency gain for the IPW method under a nonparametric procedure and show how it can achieve a parametric rate of the target sample size. As a side note, we also discuss how one may use FL to train neural network models simultaneously for both the target distribution and the odds model.
    
[^17]: 一种深度零膨胀模型用于美国东海岸北大西洋露脊鲸存在性建模以支持蓝色经济管理

    A Deep Zero-Inflated Model of North Atlantic Right Whale Presence To Support Blue Economy Management in the U.S. East Coast

    [https://arxiv.org/abs/2606.14403](https://arxiv.org/abs/2606.14403)

    本文提出了一种深度零膨胀伯努利模型，联合建模物种存在性与检测概率，有效处理被动声学监测数据中的零膨胀和复杂依赖，为濒危物种保护与蓝色经济管理提供新工具。

    

    arXiv:2606.14403v2 公告类型：替换-交叉  摘要：对濒危海洋哺乳动物物种（如北大西洋露脊鲸）的有效建模，对于在日益增长的蓝色经济中平衡海洋保护至关重要。由自主水下航行器收集的被动声学监测数据为局部海洋物种检测和海洋学感知提供了新机会，但也引入了复杂的统计挑战，如零膨胀、不完全检测和复杂依赖结构。为此，我们提出了深度零膨胀伯努利（DeepZIB）模型——一种深度统计方法，该方法联合建模潜在物种存在性和条件检测概率，同时从异构协变量信息中学习复杂的栖息地关系。我们建立了模型结构性质的理论结果，并进行了模拟实验以证明其恢复底层参数和潜在存在场的能力。应用...

    arXiv:2606.14403v2 Announce Type: replace-cross  Abstract: Effective modeling of endangered marine mammal species, such as the North Atlantic Right Whale, is critical for balancing marine conservation with the growing blue economy. Passive acoustic monitoring data collected by autonomous underwater vehicles provide new opportunities for localized marine species detection and oceanographic sensing, but introduce complex statistical challenges such as zero inflation, imperfect detection, and intricate dependence structures. In response, we propose the Deep Zero-Inflated Bernoulli (DeepZIB) model--a deep statistical method which jointly models latent species presence and conditional detection probabilities while learning complex habitat relationships from heterogeneous covariate information. We establish theoretical results on the model's structural properties and conduct simulation experiments to demonstrate its ability to recover underlying parameters and latent presence fields. Applica
    
[^18]: 微不足道的规模，举足轻重的作用：论大语言模型中的缩放向量

    Negligible in Size, Significant in Effect: On Scale Vectors in Large Language Models

    [https://arxiv.org/abs/2605.26895](https://arxiv.org/abs/2605.26895)

    缩放向量虽仅占大语言模型参数的极小部分，但并非用于增强表达能力，而是通过自放大预条件效应改善优化过程，对模型预训练效果至关重要。

    

    现代大语言模型（LLM）中的归一化层由一个确定性的归一化操作和一个可学习的缩放向量组成。尽管归一化操作已被广泛研究，但缩放向量虽然被普遍使用，却仍然缺乏深入理解。在这项工作中，我们从表达能力、优化和架构结构的角度对大语言模型中的缩放向量进行了系统研究。首先，我们通过实验证明，尽管缩放向量仅占模型参数中极小的一部分，但移除它们会显著降低大语言模型的预训练效果。我们的理论进一步表明，在Pre-Norm架构中，缩放向量并不能增加模型的表达能力；相反，它们通过对后续线性映射的自放大预条件效应来改善优化过程。其次，我们研究了权重衰减对缩放向量的作用。通过区分Input-Norm和Output-Norm层，我们从理论上……

    arXiv:2605.26895v2 Announce Type: replace  Abstract: Normalization layers in modern large language models (LLMs) consist of a deterministic normalization operation and a learnable scale vector. While the normalization operation has been extensively studied, the scale vector remains poorly understood despite its ubiquitous use. In this work, we present a systematic study of scale vectors in LLMs from the perspectives of expressivity, optimization, and architectural structure. First, we show empirically that although scale vectors constitute only a negligible fraction of model parameters, removing them substantially degrades LLM pre-training. Our theory further shows that, in Pre-Norm architectures, scale vectors do not increase expressivity; instead, they improve optimization through a self-amplifying preconditioning effect on subsequent linear mappings. Second, we investigate the role of weight decay for scale vectors. By distinguishing Input-Norm and Output-Norm layers, we theoretical
    
[^19]: 更具表达力的前馈层：第一部分：令牌自适应的激活混合

    More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations

    [https://arxiv.org/abs/2605.26647](https://arxiv.org/abs/2605.26647)

    本文提出令牌自适应的激活混合（MoA）前馈层设计，通过轻量级输入相关门控混合多个激活函数，并从理论上证明了其表达能力严格超越可学习激活（LA）和固定激活FFN。

    

    前馈网络（FFN）层在基于Transformer的大语言模型（LLM）中占据了相当大的参数比例和非线性表达能力。尽管激活函数已从ReLU和GELU演进到SwiGLU等门控变体，但大多数FFN设计仍然使用单一固定的激活函数，对所有令牌（token）应用相同的非线性变换。在这项工作中，我们提出了激活混合（Mixture of Activations, MoA），这是一种令牌自适应的FFN设计，它使用轻量级的、依赖于输入的门控机制来混合一个激活函数字典，同时共享相同的线性投影。作为输入无关的对应方案，我们还引入了可学习激活（Learnable Activations, LA），它为ReLU型和SwiGLU型FFN构造激活函数的线性组合。在理论方面，我们在固定激活FFN、LA和MoA之间建立了严格的有限宽度表达能力分离关系：LA严格包含固定激活FFN的表达能力，而MoA又严格包含LA，并进一步……

    arXiv:2605.26647v2 Announce Type: replace  Abstract: Feedforward network (FFN) layers account for a large fraction of parameters and nonlinear expressivity in Transformer-based large language models (LLMs). Despite the evolution from ReLU and GELU to gated variants such as SwiGLU, most FFN designs still use a single fixed activation function, applying the same nonlinear transformation to all tokens. In this work, we propose Mixture of Activations (MoA), a token-adaptive FFN design that mixes a dictionary of activation functions using lightweight input-dependent gates while sharing the same linear projections. As an input-independent counterpart, we also introduce learnable activations (LA), which form linear combinations of activation functions for both ReLU-type and SwiGLU-type FFNs. Theoretically, we establish strict finite-width expressive separations among fixed-activation FFNs, LA, and MoA: LA strictly contains fixed-activation FFNs, while MoA strictly contains LA, with the additi
    
[^20]: 面向动态变化专家的在线学习延迟决策

    Online Learning-to-Defer with Varying Experts

    [https://arxiv.org/abs/2605.12340](https://arxiv.org/abs/2605.12340)

    本文提出了一种将查询动作的老虎机反馈与动态变化专家池相结合的在线多分类学习延迟算法，实现了次线性真实延迟遗憾 $O(T^{2/3})$，并在集中评分条件下提升至 $O(\sqrt T)$。

    

    学习延迟（Learning-to-Defer, L2D）方法将每个查询要么路由给预测模型，要么路由给外部专家。现实世界的部署需要处理流式数据、不断变化的专家可用性、专家可靠性的漂移，以及仅针对所选动作才能观测到的反馈。我们提出了一种在线多分类L2D算法，该算法将查询动作的老虎机反馈与动态变化的专家池相结合。设 $N=n+n_e$，设 $B$ 为线性评分矩阵的 Frobenius 范数的上界，$\rho$ 为增广输入范数的上界。在线性校准以及投影比较器类上代理最小化间隙为零的假设下，我们的方法达到了期望真实延迟遗憾 $O((BN^{3/2}\rho+1)T^{2/3})$，并在集中评分条件下改进为 $O(BN^{3/2}\rho\sqrt T+B^2N^3\rho^2)$。该分析将在线 $\mathcal H$-一致性转移界与投影在线凸优化相结合。在合成数据（摘要在此处被截断）……

    arXiv:2605.12340v5 Announce Type: replace-cross  Abstract: Learning-to-Defer (L2D) methods route each query either to a predictive model or to external experts. Real-world deployments require handling streaming data, changing expert availability, shifting expert reliability, and feedback observed only for the selected action. We introduce an online multiclass L2D algorithm that combines queried-action bandit feedback with a dynamically varying pool of experts. Let $N=n+n_e$, let $B$ bound the Frobenius norm of the linear score matrix, and let $\rho$ bound the augmented input norm. Assuming linear calibration and zero surrogate minimizability gap for the projected comparator class, our method achieves expected true-deferral regret $O((BN^{3/2}\rho+1)T^{2/3})$, improving to $O(BN^{3/2}\rho\sqrt T+B^2N^3\rho^2)$ under a concentrated-score condition. The analysis combines an online $\mathcal H$-consistency transfer bound with projected online convex optimization. Experiments on synthetic a
    
[^21]: 预算约束下的因果老虎机：连接增益建模与序贯决策

    Budget-Constrained Causal Bandits: Bridging Uplift Modeling and Sequential Decision-Making

    [https://arxiv.org/abs/2604.26169](https://arxiv.org/abs/2604.26169)

    该论文提出预算约束因果老虎机（BCCB）在线框架，将个体处理效应学习、不确定性探索与预算节奏控制三者统一起来，并基于拉格朗日松弛的 KKT 条件推导出决策规则，从而解决了冷启动场景下数字广告的预算分配问题。

    

    预算约束下的处理分配是数字广告中的核心挑战。标准做法是在历史数据上训练离线增益模型，然后通过求解约束优化来分配预算，但在几乎没有历史数据的冷启动场景中这种方法会失效。我们提出了预算约束因果老虎机，这是一个在线框架，能够在花费预算的同时学习哪些用户会对广告作出响应。BCCB 统一了三个组件：学习个体层面的处理效应、探索响应不确定的用户、以及随时间推移对预算进行节奏控制。我们将每次用户到达时的决策规则推导为预算化因果分配目标的拉格朗日松弛的 KKT 条件，为算法提供了有原则的理论基础。我们在 Criteo Uplift 数据集上使用 20 个随机种子并进行配对统计检验进行评估。我们的核心发现是在 n = 7,500 处存在一个数据效率交叉点。

    arXiv:2604.26169v2 Announce Type: replace  Abstract: Treatment allocation under budget constraints is a central challenge in digital advertising. The standard approach trains an offline uplift model on historical data, then solves a constrained optimization to allocate budget. This fails in cold-start settings where little historical data exists. We propose Budget-Constrained Causal Bandits (BCCB), an online framework that learns which users respond to ads while simultaneously spending the budget. BCCB unifies three components: learning individual-level treatment effects, exploring users whose response is uncertain, and pacing the budget over time. We derive the per-arrival decision rule as the KKT condition of a Lagrangian relaxation of the budgeted causal-allocation objective, providing a principled foundation for the algorithm. We evaluate on the Criteo Uplift dataset using 20 random seeds with paired statistical tests. Our central finding is a data-efficiency crossover at n = 7,500
    
[^22]: 基于观测数据的鲁棒商品组合优化

    Robust Assortment Optimization from Observational Data

    [https://arxiv.org/abs/2602.10696](https://arxiv.org/abs/2602.10696)

    提出了一个鲁棒的数据驱动商品组合优化框架，通过建模顾客选择行为中潜在的分布偏移，克服了传统方法因假设偏好稳定和选择模型正确而在现实中导致的泛化差和收益损失问题。

    

    商品组合优化是现代零售和推荐系统中的一项根本性挑战，其目标是在复杂的顾客选择行为下，选择能够最大化预期收益的产品子集。尽管数据驱动方法的最新进展已利用历史数据来学习和优化商品组合，但这些方法通常依赖于较强的假设——即顾客偏好的稳定性以及底层选择模型的正确性。然而，在现实场景中，由于偏好漂移和模型误设，这些假设经常失效，导致泛化能力差和收益损失。受此局限性的启发，我们提出了一个鲁棒的数据驱动商品组合优化框架，该框架考虑了顾客选择行为中潜在的分布偏移。我们的方法对相对于生成数据的标称选择模型可能发生的偏好偏移进行建模，并寻求……（摘要在此处截断）

    arXiv:2602.10696v3 Announce Type: replace-cross  Abstract: Assortment optimization is a fundamental challenge in modern retail and recommendation systems, where the goal is to select a subset of products that maximizes expected revenue under complex customer choice behaviors. While recent advances in data-driven methods have leveraged historical data to learn and optimize assortments, these approaches typically rely on strong assumptions -- namely, the stability of customer preferences and the correctness of the underlying choice models. However, such assumptions frequently break in real-world scenarios due to preference shifts and model misspecification, leading to poor generalization and revenue loss. Motivated by this limitation, we propose a robust framework for data-driven assortment optimization that accounts for potential distributional shifts in customer choice behavior. Our approach models potential preference shift from a nominal choice model that generates data and seeks to 
    
[^23]: Autotune：面向Lasso的快速、准确且自动化的调优参数选择方法

    Autotune: fast, accurate, and automatic tuning parameter selection for Lasso

    [https://arxiv.org/abs/2512.11139](https://arxiv.org/abs/2512.11139)

    该论文提出autotune方法，通过在回归系数与噪声标准差之间交替优化带惩罚的高斯对数似然，实现Lasso调优参数的全自动选择，在低信噪比情形下比现有方法更快且具有更优的泛化性能和模型选择效果。

    

    最小绝对收缩与选择算子（Lasso）是一种流行的高维回归方法，目前已被广泛用于估计诸如向量自回归（VAR）等高维时间序列模型。尽管已有大量可选方法，如何高效且准确地选择其调优参数仍然是一个挑战。我们提出了 $\mathsf{autotune}$，这是一种让Lasso自动完成调优的策略，它通过在回归系数和噪声标准差之间交替优化带惩罚的高斯对数似然来实现。通过在回归模型和VAR模型上开展的大量模拟实验，我们表明在低信噪比环境下，$\mathsf{autotune}$ 比现有方法更快，并具有更好的泛化能力和模型选择性能。在此过程中，$\mathsf{autotune}$ 还提供了一种可用于高维统计推断的新的噪声标准差估计量，以及一种新的可视化……

    arXiv:2512.11139v3 Announce Type: replace-cross  Abstract: Least absolute shrinkage and selection operator (Lasso), a popular method for high-dimensional regression, is now used widely for estimating high-dimensional time series models such as the vector autoregression (VAR). Selecting its tuning parameter efficiently and accurately remains a challenge, despite the abundance of available methods for doing so. We propose $\mathsf{autotune}$, a strategy for Lasso to automatically tune itself by optimizing a penalized Gaussian log-likelihood alternately over regression coefficients and noise standard deviation. Using extensive simulation experiments on regression and VAR models, we show that $\mathsf{autotune}$ is faster, and provides better generalization and model selection than established alternatives in low signal-to-noise regimes. In the process, $\mathsf{autotune}$ provides a new estimator of noise standard deviation that can be used for high-dimensional inference, and a new visual
    
[^24]: 预测序后验

    Prequential posteriors

    [https://arxiv.org/abs/2511.17721](https://arxiv.org/abs/2511.17721)

    本文提出基于预测序列损失函数的prequential后验方法，解决了深度生成预测模型因似然函数不可解而无法应用标准贝叶斯数据同化的难题，并证明了其在温和条件下的理论一致性保证。

    

    数据同化是在观测到新数据时更新预测模型的一项基础任务，其应用涵盖天气预报到在线强化学习等领域。深度生成预测模型（DGFMs）在这些领域表现出色，但由于其似然函数难以处理，将数据同化到此类模型中极具挑战性。这一局限性限制了标准贝叶斯数据同化方法在DGFMs中的应用。为了克服这一问题，我们提出了基于预测序列损失函数的prequential后验；该方法天然适用于时间相关数据，而这正是预测任务的核心关注点。由于真实的数据生成过程往往超出了所假设的模型类别，我们采用了一种替代的一致性概念，并证明在温和的条件下，prequential损失最小化器和prequential后验均会集中在……

    arXiv:2511.17721v2 Announce Type: replace-cross  Abstract: Data assimilation is a fundamental task in updating forecasting models upon observing new data, with applications ranging from weather prediction to online reinforcement learning. Deep generative forecasting models (DGFMs) have shown excellent performance in these areas, but assimilating data into such models is challenging due to their intractable likelihood functions. This limitation restricts the use of standard Bayesian data assimilation methodologies for DGFMs. To overcome this, we introduce prequential posteriors, based upon a predictive-sequential (prequential) loss function; an approach naturally suited for temporally dependent data which is the focus of forecasting tasks. Since the true data-generating process often lies outside the assumed model class, we adopt an alternative notion of consistency and prove that, under mild conditions, both the prequential loss minimizer and the prequential posterior concentrate aroun
    
[^25]: 基于算子诱导与正则化符号森林的概率符号回归方程发现方法

    Probabilistic Symbolic Regression for Equation Discovery via Operator-induced and Regularized Symbolic Forests

    [https://arxiv.org/abs/2509.19710](https://arxiv.org/abs/2509.19710)

    该论文提出一种概率符号回归框架，将数学表达式表示为符号树集成，通过树拓扑上的正则化先验控制表达式复杂度，并利用基于奥卡姆窗口的后验摘要刻画多个合理符号模型的不确定性，为方程发现提供了兼具精度、简洁性与不确定性量化的统一解决方案。

    

    符号回归已成为人工智能驱动的科学发现的强大工具，它通过学习可解释的解析表达式，直接从数据中揭示变量间的支配性关系。然而，现有方法往往依赖启发式搜索，在噪声环境下难以平衡预测精度与表达式复杂度，且对符号不确定性的刻画十分有限。能够以统一方式解决这些挑战的概率化方法仍未得到充分探索。我们提出了一种概率符号回归框架，将数学表达式表示为符号树的集成。树拓扑结构上的正则化先验用于控制表达式复杂度，而基于奥卡姆窗口的后验摘要则用于捕捉多个合理符号模型之间的不确定性。鉴于符号回归领域现有的理论研究较为匮乏，我们进一步发展了后验集中性保证。

    arXiv:2509.19710v3 Announce Type: replace-cross  Abstract: Symbolic regression has emerged as a powerful tool for artificial intelligence-driven scientific discovery by learning interpretable analytical expressions that reveal governing relationships directly from data. Existing methods, however, often rely on heuristic search, struggle to balance predictive accuracy with expression complexity in noisy settings, and offer limited characterization of symbolic uncertainty. Probabilistic approaches that address these challenges in a unified manner remain underexplored. We introduce a probabilistic symbolic regression framework that represents mathematical expressions as ensembles of symbolic trees. A regularizing prior over tree topology controls expression complexity, while an Occam's window-based posterior summary captures uncertainty across multiple plausible symbolic models. Given the limited existing theoretical treatment of symbolic regression, we develop posterior concentration gua
    
[^26]: 偏离正态之路：学习节点移动性的空间密度模型

    Off the Normal Path: Learning Spatial Density Models of Node Mobility

    [https://arxiv.org/abs/2411.10997](https://arxiv.org/abs/2411.10997)

    该论文引入Möbius分布混合模型来学习二维地形上移动节点的稳态空间密度，相比混合密度网络和归一化流等现成方法，提供了更可解释、更简洁且性能相当或更优的模型。

    

    我们研究学习空间密度函数模型的问题，该函数表示在二维地形上移动的移动节点的稳态密度。推导此类模型可以辅助网络设计与优化问题，例如在参数扫描过程中加速密度函数的计算。我们探讨了现成的混合密度网络模型以及两种类型的归一化流在描述圆盘上移动节点密度方面的适用性。我们引入了Möbius分布来保持对称的空间关系。我们的结果表明，Möbius分布的混合为所研究的稳态密度分布提供了可解释且简洁的模型，其性能与替代方法相当或更优。

    arXiv:2411.10997v2 Announce Type: replace-cross  Abstract: We consider the problem of learning models of spatial density functions, representing the steady-state density of mobile nodes moving on a two-dimensional terrain. Deriving such models can assist in network design and optimization problems, e.g., by accelerating the computation of the density function during a parameter sweep. We address the question of applicability of off-the-shelf mixture density network models and of, two varieties of, normalizing flows for the description of mobile node density over a disk. We introduce the use of M\"obius distributions to retain symmetric spatial relations. Our results indicate that mixtures of M\"obius distributions provide interpretable, parsimonious models for the studied steady state density distributions, that match or outperform the alternatives.
    
[^27]: 关于分散推断模型的扩散模型：基准测试和改进随机控制和采样

    On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling

    [https://arxiv.org/abs/2402.05098](https://arxiv.org/abs/2402.05098)

    本研究探讨了训练扩散模型以从给定分布中采样的问题，并针对随机控制和采样提出了一种新的探索策略，通过基准测试比较了不同推断方法的相对优劣，并对过去的工作提出了质疑。

    

    我们研究了训练扩散模型以从给定的非标准化密度或能量函数分布中采样的问题。我们对几种扩散结构推断方法进行了基准测试，包括基于模拟的变分方法和离策略方法（连续生成流网络）。我们的结果揭示了现有算法的相对优势，同时对过去的研究提出了一些质疑。我们还提出了一种新颖的离策略方法探索策略，基于目标空间中的局部搜索和回放缓冲区的使用，并证明它可以改善各种目标分布上的样本质量。我们研究的采样方法和基准测试的代码已公开在https://github.com/GFNOrg/gfn-diffusion，作为未来在分散推断模型上工作的基础。

    We study the problem of training diffusion models to sample from a distribution with a given unnormalized density or energy function. We benchmark several diffusion-structured inference methods, including simulation-based variational approaches and off-policy methods (continuous generative flow networks). Our results shed light on the relative advantages of existing algorithms while bringing into question some claims from past work. We also propose a novel exploration strategy for off-policy methods, based on local search in the target space with the use of a replay buffer, and show that it improves the quality of samples on a variety of target distributions. Our code for the sampling methods and benchmarks studied is made public at https://github.com/GFNOrg/gfn-diffusion as a base for future work on diffusion models for amortized inference.
    
[^28]: 单一生成流网络中的图结构与参数的联合贝叶斯推理

    Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network. (arXiv:2305.19366v1 [cs.LG])

    [http://arxiv.org/abs/2305.19366](http://arxiv.org/abs/2305.19366)

    本文提出了在单一生成流网络中联合建模贝叶斯网络结构和参数的方法，包括非离散样本空间，提高了贝叶斯网络局部概率模型的灵活性。

    

    生成流网络是一类对离散和结构化样本空间进行建模的生成模型。先前的研究已将其应用于推断给定观测数据的贝叶斯网络的有向无环图（DAG）的边缘后验分布。本文基于最近的研究进展，在非离散样本空间上将此框架扩展到联合后验分布的建模，不仅包括贝叶斯网络的结构，还考虑了其条件概率分布的参数。

    Generative Flow Networks (GFlowNets), a class of generative models over discrete and structured sample spaces, have been previously applied to the problem of inferring the marginal posterior distribution over the directed acyclic graph (DAG) of a Bayesian Network, given a dataset of observations. Based on recent advances extending this framework to non-discrete sample spaces, we propose in this paper to approximate the joint posterior over not only the structure of a Bayesian Network, but also the parameters of its conditional probability distributions. We use a single GFlowNet whose sampling policy follows a two-phase process: the DAG is first generated sequentially one edge at a time, and then the corresponding parameters are picked once the full structure is known. Since the parameters are included in the posterior distribution, this leaves more flexibility for the local probability models of the Bayesian Network, making our approach applicable even to non-linear models parametrized
    
[^29]: 利用GFlowNets解决图形组合优化问题

    Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets. (arXiv:2305.17010v1 [cs.LG])

    [http://arxiv.org/abs/2305.17010](http://arxiv.org/abs/2305.17010)

    本文提出了一种名为GFlowNets的机器，可以有效地解决组合优化问题，同时在训练方面进行了优化，结果表明其可以高效地找到高质量的解决方案。

    

    组合优化问题通常是NP难题，因此不适用于精确算法，这使它们成为应用机器学习方法的理想领域。这些问题中高度结构化的限制可能会直接阻碍优化或采样解决方案的空间。另一方面，GFlowNets最近被发现是一种强大的机器，可以顺序地从复合非规范化密度中有效地采样，并具有在CO中分摊此类解决方案搜索过程以及生成不同的解决方案候选项的潜力。在本文中，我们设计了适用于不同组合问题的马尔科夫决策过程（MDP），并提出训练有条件的GFlowNets从解空间中采样的策略。还开发了高效的训练技术来受益于远程信用分配。通过对各种使用合成和实际数据的不同CO任务的广泛实验，我们证明了GFlowNet策略可以有效地找到高质量的解。

    Combinatorial optimization (CO) problems are often NP-hard and thus out of reach for exact algorithms, making them a tempting domain to apply machine learning methods. The highly structured constraints in these problems can hinder either optimization or sampling directly in the solution space. On the other hand, GFlowNets have recently emerged as a powerful machinery to efficiently sample from composite unnormalized densities sequentially and have the potential to amortize such solution-searching processes in CO, as well as generate diverse solution candidates. In this paper, we design Markov decision processes (MDPs) for different combinatorial problems and propose to train conditional GFlowNets to sample from the solution space. Efficient training techniques are also developed to benefit long-range credit assignment. Through extensive experiments on a variety of different CO tasks with synthetic and realistic data, we demonstrate that GFlowNet policies can efficiently find high-quali
    
[^30]: 轨迹平衡：改进了GFlowNets中的信用分配

    Trajectory balance: Improved credit assignment in GFlowNets. (arXiv:2201.13259v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.13259](http://arxiv.org/abs/2201.13259)

    GFlowNets使用轨迹平衡作为一种更高效的学习目标，解决了先前学习目标中信用传播效率低下的问题，并且在实验中证明了其在收敛性、生成样本多样性以及鲁棒性方面的优势。

    

    生成流网络（GFlowNets）是一种学习使用动作序列生成组合对象（如图形或字符串）的随机策略的方法，其中许多可能的动作序列可能导致相同的对象。我们发现先前提出的GFlowNets学习目标，即流匹配和详细平衡，类似于时间差分学习，容易在长的动作序列中出现信用传播效率低下的问题。因此，我们提出了一种新的学习目标，即轨迹平衡，作为先前使用目标的更高效的替代方法。我们证明了轨迹平衡目标的任何全局极小值可以定义一个从目标分布精确采样的策略。在四个不同领域的实验中，我们从实证上证明了轨迹平衡目标对于GFlowNet收敛性、生成样本的多样性以及对长动作序列和噪声的鲁棒性的益处。

    Generative flow networks (GFlowNets) are a method for learning a stochastic policy for generating compositional objects, such as graphs or strings, from a given unnormalized density by sequences of actions, where many possible action sequences may lead to the same object. We find previously proposed learning objectives for GFlowNets, flow matching and detailed balance, which are analogous to temporal difference learning, to be prone to inefficient credit propagation across long action sequences. We thus propose a new learning objective for GFlowNets, trajectory balance, as a more efficient alternative to previously used objectives. We prove that any global minimizer of the trajectory balance objective can define a policy that samples exactly from the target distribution. In experiments on four distinct domains, we empirically demonstrate the benefits of the trajectory balance objective for GFlowNet convergence, diversity of generated samples, and robustness to long action sequences and
    

