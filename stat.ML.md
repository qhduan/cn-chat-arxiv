# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Instance-dependent sequential limits as an exact information ledger](https://arxiv.org/abs/2610.42133) | 本文将多臂老虎机、序贯检验、最优臂识别和马尔可夫决策过程的实例相关下界统一为一条“精确信息账本”式的证明链路——从散度分解经数据处理（恰损失一个在自然指数族下消失的条件散度）到检验与优化，并借助单参数鞅恒等式族推广到实例间的任意倾斜。 |
| [^2] | [Gradient Descent on Two ReLU Neurons: Global Landscape and Bifurcation Dynamics](https://arxiv.org/abs/2610.42118) | 该论文证明双神经元ReLU教师-学生模型中的梯度流动力学由“容易”的平分线方向和“困难”的分裂方向共同组织，学生神经元先坍缩至平分线鞍点再逃逸并实现特征专业化，且所有非平凡临界点要么是全局最小值要么是鞍点。 |
| [^3] | [Regret-Optimal Wasserstein-Robust Linear Regression](https://arxiv.org/abs/2610.42109) | 该论文刻画了在Wasserstein扰动下多元最小二乘回归中分布鲁棒优化（DRO）的遗憾性能，证明其在小扰动极限和有界残差有效秩情形下能达到实例最优或极小极大速率，同时也识别出DRO按维度相关因子次优的失效模式。 |
| [^4] | [CausalWAM: Learning the Causes of Actions in World Action Models](https://arxiv.org/abs/2610.42096) | 该论文提出CausalWAM，从理论上证明在稀疏性约束下，可将引发动作的因果潜变量与预测所需的干扰变量解耦，从而确保世界动作模型仅从动作相关的潜在成因生成动作。 |
| [^5] | [Cost-Aware Stopping for Freeze-Thaw Bayesian Optimization: When Does One-Step Lookahead Suffice?](https://arxiv.org/abs/2610.42085) | 该论文为冻结-解冻贝叶斯优化提出了一种成本感知的EI停止规则，证明其期望效用不低于立即停止，并构建两步前瞻价值下界证书来识别过早停止；实验表明在高成本场景下单步前瞻通常已足够，而低成本场景下修正方法更有价值。 |
| [^6] | [Near-Optimal Second-Order Convex Finite-Sum Optimization under Individual Smoothness](https://arxiv.org/abs/2610.42040) | 本文证明了个体光滑性与凸性假设可将二阶凸有限和优化的复杂度下界改进为 $\Omega(n + n^{5/7} \epsilon^{-2/7})$，并通过提出新颖的二阶自界定不等式和泰勒修正惰性牛顿方法，结合最优 Monteiro-Svaiter 加速，实现了与下界仅差对数因子的近最优算法复杂度。 |
| [^7] | [On Statistical-Computational Gaps in Graph Combinatorial Testing Problems](https://arxiv.org/abs/2610.41995) | 本文刻画了图组合假设检验问题中强检测所需信号强度的精确统计阈值与计算阈值，并首次识别出多种植团、超团、粗循环和粗路径等问题上新的统计-计算鸿沟。 |
| [^8] | [WASABI: Wasserstein Sampling via Bridge Interpolants](https://arxiv.org/abs/2610.41958) | 提出WASABI方法，利用Wasserstein梯度流和桥插值训练扩散模型，仅凭能量函数即可从玻尔兹曼分布中采样，且避免了难以计算的对数密度梯度，训练完成后无需再评估能量或其梯度。 |
| [^9] | [Chronos-2: A foundation model for multivariate time series forecasting](https://arxiv.org/abs/2610.41956) | Chronos-2是一个零样本多变量时间序列预测基础模型，通过大规模预训练能够建模时间序列之间及其与异构协变量之间的交互，直接从历史数据中推断关系并生成带有不确定性量化的概率预测。 |
| [^10] | [Statistical Foundations of Reinforcement Learning: A Non-Asymptotic Perspective](https://arxiv.org/abs/2610.41903) | 本文系统阐述了强化学习的非渐近统计理论基础，聚焦于在有限样本情形下理解并提升强化学习算法的数据效率这一核心问题。 |
| [^11] | [Bethe Hessian Recovery through the Kesten-Stigum Regime](https://arxiv.org/abs/2610.41802) | 本文证明对于任意固定期望度 d>1 的随机块模型，Bethe Hessian 的离群特征向量都能实现直至信息论最优的 Kesten-Stigum 阈值的弱恢复，仅用厄米特矩阵即匹配了非回溯谱方法的恢复保证。 |
| [^12] | [PDE-Residual-Informed Kriging](https://arxiv.org/abs/2610.41590) | 本文提出基于PDE残差核的PDE残差信息克里金法（PRIK），将整个定义域上的全局PDE残差信息直接融入高斯过程的协方差结构，在保持模型灵活性与标准GP条件化的同时，无需额外的残差权重参数。 |
| [^13] | [Reliability-Weighted Direct Preference Optimization: Penalized Weights for Robust Preference Learning](https://arxiv.org/abs/2610.41588) | 提出RW-DPO方法，通过为每条偏好比较引入可学习的可靠性权重并结合极小极大凹惩罚，以几乎不增加DPO计算成本的方式自动识别并降权噪声比较，实现稳健的偏好学习。 |
| [^14] | [Conditional Neural SDEs for Financial Time-Series Generation](https://arxiv.org/abs/2610.41405) | 提出一种残差锚定的条件神经随机微分方程架构，通过将目标分解为条件锚点与随机残差，稳定地生成能够响应外部上下文的金融时间序列轨迹。 |
| [^15] | [Coarse-Graining Optimal Transport for Rollout-Free Post-Training of Flow Matching](https://arxiv.org/abs/2610.41269) | 提出 ReCouple 可控框架，通过构造粗粒度最优传输耦合来实现流匹配的无需滚动后训练，并扩展出面向连续源设置的 ReCouple+ 与 ReCouple++，从而在低函数评估次数下改善采样质量。 |
| [^16] | [Instrumental Transfer Learning for Solution Operators](https://arxiv.org/abs/2610.41262) | 提出了一种基于预解恒等式和白化工具变量的三阶段结构化迁移学习方法，能够在目标系统观测稀缺时，利用相关源系统数据无偏地学习微分方程的解算子。 |
| [^17] | [Stochastic Mirror Descent under Dependent Observations](https://arxiv.org/abs/2610.41251) | 该论文突破了传统独立同分布假设，通过引入适用于几何或多项式相关序列的灵活“遗忘”条件，在相关观测下建立了随机镜像下降算法的显式有限样本收敛界，并恢复了经典的 O(1/T) 与 O(1/√T) 收敛速率。 |
| [^18] | [A Unified Framework for Probability Estimation of Data-Dependent Events](https://arxiv.org/abs/2610.41244) | 该论文提出了一个统一框架来估计由样本数据经映射决定的事件概率，该框架同时涵盖缺失质量、自适应数据分析目标与模型测试误差等经典问题，并证明了代入估计器的误差恰为泛化差距、留一估计器对稳定映射均准确。 |
| [^19] | [Adaptive Uncertainty Quantification on Simplicial Complexes with Conformalized Simplicial Neural Networks](https://arxiv.org/abs/2610.41166) | 提出保形化单纯神经网络（CF-SNN），利用保形预测为单纯复形上的多个预测目标同时生成具有覆盖率保证的预测集，为SNNs提供了严格的不确定性量化能力。 |
| [^20] | [MUON-RT Reusing the Teacher's Trajectory for Momentum-Free Muon for LLM Self-Distillation](https://arxiv.org/abs/2610.41104) | 提出Muon-RT，通过重用自蒸馏中教师与学生的位移替代Muon优化器的动量缓冲区，在降低训练内存开销的同时保证了理论收敛性，并提升大语言模型自蒸馏微调的性能。 |
| [^21] | [Where, When, and How Many Bits? Scaling Laws for SGD with Stochastic Rounding](https://arxiv.org/abs/2610.41074) | 本文通过两层线性网络的理论分析，证明了随机舍入下低精度SGD训练预测误差的匹配上下界，并给出了达到最优计算缩放时所需的模型规模、训练时间与精度调度选择规律。 |
| [^22] | [Omniprediction for All Mixable Proper Losses and Beyond](https://arxiv.org/abs/2610.41023) | 本文证明当损失类扩展为无限类时，全能预测的次线性遗憾一般无法实现——即使预测器类仅含两个线性预测器，也不可能对所有有界恰当损失同时达到次线性遗憾，从而为该问题给出否定答案。 |
| [^23] | [Density-Guided Bootstrap Boundaries for Latent World-Model Reinforcement Learning](https://arxiv.org/abs/2610.41022) | 该论文提出密度引导自举边界机制，根据轨迹可靠性自适应调整想象回报的展开深度，从而在潜在世界模型强化学习中平衡模型误差累积与对评论家近似的过度依赖。 |
| [^24] | [Robust-by-Design Distributional Learning from Contaminated Samples](https://arxiv.org/abs/2610.40884) | 该论文提出污染校正的最大均值差异估计量（CC-MMD），通过将带噪声的代理分数与小规模已审计子集的精确标签校正相结合，从受污染的混合样本中稳健地学习潜在的目标分布。 |
| [^25] | [Near-Optimal First-Order Regret for Bandits with Feedback Graphs](https://arxiv.org/abs/2610.40861) | 本文针对固定已知有向反馈图的对抗性赌博机，提出了期望遗憾界为 $\widetilde{O}(\sqrt{\alpha L^\star}+\alpha)$ 的算法，将一阶遗憾的图依赖性从团划分数改进到独立数，从而弥合了最坏情况遗憾与一阶遗憾之间的差距，达到了近最优的一阶遗憾保证。 |
| [^26] | [Certified Behavioral Retention for Continual Adaptation with Held-Out Anchors](https://arxiv.org/abs/2610.40740) | 该论文提出CABA方法，通过在留出锚点上比较新旧策略、筛选最坏锚点前向KL散度不超预算的最大插值系数，在无需动作标签或完整轨迹的情况下为持续适应中的模型更新提供可认证的行为保持保证。 |
| [^27] | [INLA for Adaptive UAV Sampling of Radio Environment Maps in IoT Networks: Prior Width Governs Accuracy and Calibration](https://arxiv.org/abs/2610.40723) | 该论文提出基于INLA更新Matérn GMRF后验的自适应无人机采样方法来构建无线电环境地图，证明后验信息采样显著优于随机与割草机式基线，且增大先验阴影衰落标准差可在4–16 dB范围内单调提升精度与校准。 |
| [^28] | [The Geometry of Multitask Learning: Sharp Rates for Shared Subspace Estimation](https://arxiv.org/abs/2610.40713) | 本文提出一个通用多任务引理，通过任务回归函数及其估计量的几何条件刻画共享子空间估计的充分条件，首次获得线性回归与分类的精确极小极大速率。 |
| [^29] | [CausalGrade: Evidence-Guided Causal Graph Scoring with Jev](https://arxiv.org/abs/2610.40692) | 提出基于概率判断模型Jev的因果图评分方法CausalGrade，通过融合观测数据证据与因果发现基础模型CDFM的边概率对候选因果图进行评分，评分与结构误差呈强负相关，可在真实因果结构未知时评估因果图质量。 |
| [^30] | [Dual-Corrected Stochastic Block Model ($\text{DC}^2\text{SBM}$): A Generalized Structure Detection Method for Attributed Networks](https://arxiv.org/abs/2610.40676) | 提出双重校正随机块模型DC²SBM，通过在连边生成和属性生成通道中分别引入独立的拓扑度因子和属性丰度因子，实现节点活跃度与成员关系推断的解耦，从而有效检测属性网络中的广义结构。 |
| [^31] | [Recertifying decisions via e-closure: uniform improvements of conformal risk control and randomized e-BH](https://arxiv.org/abs/2610.35821) | 提出“重新认证”这一通用范式，借助e-闭包原理利用风险控制程序有效性证书中的可计算松弛，统一改进了共形风险控制和随机化e-BH方法，构造出更少保守的决策。 |
| [^32] | [Sharp Non-Asymptotic Analysis of the Penalized Challenger in $\beta$-EB-TCI for Bernoulli Bandits](https://arxiv.org/abs/2610.01951) | 本文首次对β-EB-TCI算法的惩罚挑战者给出精确非渐近分析，证明经验领导者成为真实最优臂后停止时间达到T_β*(μ)log(1/δ)（直至低阶项），从而为所有唯一最优臂的伯努利实例建立了非渐近高概率界。 |
| [^33] | [On Large-Scale Multiple Testing Over Networks: A Non-Asymptotic Approach](https://arxiv.org/abs/2609.14170) | 该论文发现分布式多重检验中有限样本FDR失控源于赢者诅咒偏差，并提出交叉拟合贪心聚合算法（CFGA），通过数据分割实现了网络上分布式多重检验在有限样本下的严格FDR控制。 |
| [^34] | [Toward Optimal Switching Regret for Multi-Armed Bandits with Oblivious Adversary](https://arxiv.org/abs/2609.13547) | 本文提出单一算法，在非自适应对手下对任意未知的切换次数 $S$ 均能达到最优的 $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ 期望切换遗憾，解决了 Auer 等人提出的开放问题。 |
| [^35] | [Minimax Alternating Regret for the Experts Problem and Online Convex Optimization](https://arxiv.org/abs/2608.25182) | 本文首次解决了专家问题和一般在线凸优化中交替遗憾的极小极大率，证明$d$-专家问题的交替遗憾仅为$\Theta(\log d)$且与时间$T$无关，大幅改进了先前结果。 |
| [^36] | [Geometry of Relaxed Fair Regression: A Unified Framework for Aware and Unaware Settings](https://arxiv.org/abs/2605.28233) | 该论文将感知与无感知设置下的弛豫公平回归统一表述为最优传输问题，刻画了最优预测函数，并揭示Wasserstein惩罚体现全体人群的平滑折中而全变差惩罚强制子集均等这两种根本不同的公平哲学。 |
| [^37] | [Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](https://arxiv.org/abs/2605.25592) | 我们提出了一种计算高效的多项逻辑特模型最优实验设计框架，通过混合整数线性规划和多项式时间替代目标实现近G-最优性，并应用于最佳品类识别。 |
| [^38] | [Non-Gaussian Uncertainty Estimation via Semi-nonparametric Density](https://arxiv.org/abs/2604.09375) | 本文提出了基于半非参数（SNP）密度的非高斯不确定性估计框架，利用平方Hermite多项式保证密度非负性，推导了解析的边际分布与累积分布函数并给出无损凸的系数估计方法，除多项式阶数外无需额外超参数即可直接应用于非线性动力学系统的不确定性传播。 |
| [^39] | [Co-optimization for Adaptive Conformal Prediction](https://arxiv.org/abs/2603.01719) | 提出CoCP方法，通过中心与半径的交替协同优化构建自适应保形预测区间，修正居中偏差，在保证有限样本边际有效性的同时实现更窄的区间、渐近条件覆盖和最优区间长度。 |
| [^40] | [Spectral Condition for $\mu$P under Width-Depth Scaling](https://arxiv.org/abs/2603.00541) | 本工作提出了一个简单统一的谱框架来刻画宽度-深度联合缩放下的μP，揭示了残差块中变换数量从k=1到k≥2的基本转变，从而统一了先前分散的μP理论并确立了CompleteP式参数化的地位。 |
| [^41] | [Cross-Fitted Clipped Covariance Estimation with a Data-Driven Tail-Energy Criterion](https://arxiv.org/abs/2602.14020) | 该论文提出了完全数据驱动的分位数尾部能量替代准则（QTES），通过结合交叉拟合方差证书与截断所去除尾部能量的保留估计来自动选择最优截断水平，无需分布先验参数，并在有限样本下提供相对于最优候选估计器的理论保证。 |
| [^42] | [A Function-Space Stability Boundary for Generalization in Interpolating Learning Systems](https://arxiv.org/abs/2602.03514) | 该论文提出函数空间稳定性证书来刻画算法稳定性何时能解释插值学习系统的泛化，并证明存在稳定性无法解释的低风险插值区域，从而界定了稳定性解释的适用边界。 |
| [^43] | [Variational decomposition autoencoding improves disentanglement of latent representations](https://arxiv.org/abs/2601.06844) | 该论文提出变分分解自编码框架VDA，通过仅在编码器端的DecVAE网络在分布层面进行潜在自编码，将信号分解的结构偏置融入变分表示学习，从而更好地解耦复杂非平稳信号的潜在表示。 |
| [^44] | [Who Gets Watched? Allocating Scarce Escalation Resources on Pediatric Wards with Neural Index Policies](https://arxiv.org/abs/2510.22069) | 该论文提出MILOP方法，将病房患者升级处置决策建模为多动作不宁多臂老虎机问题，并通过决策聚焦学习训练神经网络生成指数策略，从而在资源有限时优化儿科病房中对病情恶化患者的升级资源分配。 |
| [^45] | [Critically-Damped Higher-Order Langevin Dynamics for Generative Modeling](https://arxiv.org/abs/2506.21741) | 该论文提出将临界阻尼技术从CLD和TOLD++推广到任意阶数的高阶朗之万动力学，从而改进基于扩散模型的生成建模方法。 |
| [^46] | [Transport-Free Lower Bounds for Ollivier--Ricci Curvature in Higher-Order Networks: Scope, Tightness, and Computational Applications](https://arxiv.org/abs/2405.13302) | 本文将Bauer–Jost–Liu曲率界推广至超图Ollivier–Ricci曲率，建立了无需最优传输计算的下界，刻画了其适用范围、紧致性与极值几何，并给出了高效的计算应用。 |
| [^47] | [Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets](https://arxiv.org/abs/2003.03685) | 提出了PCMCI+方法，通过优化条件集选择，显著提升了在强自相关非线性时间序列中发现同期与滞后因果关系的召回率和可靠性。 |

# 详细

[^1]: 作为精确信息账本的实例相关序列极限

    Instance-dependent sequential limits as an exact information ledger

    [https://arxiv.org/abs/2610.42133](https://arxiv.org/abs/2610.42133)

    本文将多臂老虎机、序贯检验、最优臂识别和马尔可夫决策过程的实例相关下界统一为一条“精确信息账本”式的证明链路——从散度分解经数据处理（恰损失一个在自然指数族下消失的条件散度）到检验与优化，并借助单参数鞅恒等式族推广到实例间的任意倾斜。

    

    摘要：arXiv:2610.42133v1 公告类型：新论文 摘要：多臂老虎机遗憾、序贯检验、最优臂识别和马尔可夫决策过程的实例相关极限通过一条链路得到证明。该链路从散度分解出发，经过数据处理到达一个统计量，再经过一个将类别保证转化为对历史记录要求的检验步骤，最后在备选实例和采样分配上进行优化。其中只有第一步是恒等式。本工作将这条链路写成一个精确的账本。根据链式法则，数据处理恰好损失一个条件散度，而对于自然指数族中的臂，在每臂计数和奖励总和处该条件散度恰好为零。一个单参数的鞅恒等式族将该分解扩展到两个实例之间的任意倾斜。经典极限通过对带有明确命名缺陷的松弛得到。除渐近步骤的缺陷外，所有缺陷都可以从算法自身的运行中计算出来，而一致量化的缺陷仅当类别保证是一个数值时才能计算。在七个已发……

    arXiv:2610.42133v1 Announce Type: new  Abstract: Instance-dependent limits for bandit regret, sequential tests, best-arm identification and Markov decision processes are proved by one chain. It runs from a divergence decomposition through data processing to a statistic, a testing step that turns a class guarantee into a requirement on the history, and optimization over alternatives and allocations. Only the first step is an identity. This work writes the chain as an exact ledger. By the chain rule, data processing loses exactly a conditional divergence, which vanishes for arms in natural exponential families at the per-arm counts and reward sums. A one-parameter family of martingale identities extends the decomposition to every tilt between two instances. The classical limits follow by relaxations with named defects. All defects but the asymptotic step's can be computed from an algorithm's own runs, the uniform quantification's only when the class guarantee is a number. On seven publis
    
[^2]: 两个ReLU神经元上的梯度下降：全局景观与分岔动力学

    Gradient Descent on Two ReLU Neurons: Global Landscape and Bifurcation Dynamics

    [https://arxiv.org/abs/2610.42118](https://arxiv.org/abs/2610.42118)

    该论文证明双神经元ReLU教师-学生模型中的梯度流动力学由“容易”的平分线方向和“困难”的分裂方向共同组织，学生神经元先坍缩至平分线鞍点再逃逸并实现特征专业化，且所有非平凡临界点要么是全局最小值要么是鞍点。

    

    理解梯度下降如何学习特征仍然是神经网络理论中的一个核心挑战。我们在一个已经展现出丰富多阶段梯度动力学的最小多指标设置中研究这个问题：一个良定义的双神经元ReLU教师-学生模型，在各向同性高斯输入下，从小初始化开始通过总体梯度流进行训练。我们证明动力学由两个方向组织：携带主要信号的“容易”平分线方向，以及控制特征专业化的“困难”分裂方向。为了刻画损失结构和学习行为，我们分析了总体平方损失景观，并证明每个学生权重都非零的临界点要么是全局最小值，要么是鞍点。随后我们追踪梯度流轨迹，表明学生神经元首先坍缩向平分线鞍点，之后才逃逸并专业化以匹配教师神经元。

    arXiv:2610.42118v1 Announce Type: new  Abstract: Understanding how gradient descent learns features remains a central challenge in neural network theory. We study this question in a minimal multi-index setting that already exhibits rich multi-phase gradient dynamics: a well-specified two-neuron ReLU teacher-student model under isotropic Gaussian inputs, trained by population gradient flow from small initialization. We show that the dynamics are organized by two directions: the "easy" bisector direction, which carries the leading signal, and the "hard" splitting direction, which governs specialization. To characterize the loss structure and learning behavior, we analyze the population squared-loss landscape and show that every critical point with both student weights nonzero is either a global minimum or a saddle. We then track the gradient-flow trajectory, showing that student neurons first collapse toward a bisector saddle before escaping and specializing to teacher neurons. Our resul
    
[^3]: 遗憾最优的Wasserstein鲁棒线性回归

    Regret-Optimal Wasserstein-Robust Linear Regression

    [https://arxiv.org/abs/2610.42109](https://arxiv.org/abs/2610.42109)

    该论文刻画了在Wasserstein扰动下多元最小二乘回归中分布鲁棒优化（DRO）的遗憾性能，证明其在小扰动极限和有界残差有效秩情形下能达到实例最优或极小极大速率，同时也识别出DRO按维度相关因子次优的失效模式。

    

    分布鲁棒优化（DRO）在分布不确定性下最小化最坏情况风险，但其决策在超额风险（即遗憾）方面可能是次优的。我们针对多元最小二乘回归研究这一差距，假设观测到的源分布与未知的目标分布之间存在大小为 ε 的对抗性 2-Wasserstein 扰动。我们首先确定了在 ε→0 极限下鲁棒决策的渐近复杂度，该复杂度由一个问题特定的条件数 κ 刻画。在此情形下，包括 DRO 在内的众多简单算法均可达到实例最优的 O(κε²) 速率。对于固定的 ε>0，我们证明当问题具有有界的残差有效秩时，DRO 仍能达到极小极大速率。另一方面，我们识别出 DRO 被证明以依赖于维度的因子次优的显著失效模式。幸运的是，我们证明……（原文摘要在此截断）

    arXiv:2610.42109v1 Announce Type: cross  Abstract: Distributionally robust optimization (DRO) minimizes worst-case risk under distributional uncertainty, but its decisions can be suboptimal in terms of their excess risk, or regret. We study this gap for multivariate least-squares regression, assuming that the observed source law and unknown target law differ by an adversarial 2-Wasserstein perturbation of size $\varepsilon$. We first identify the asymptotic complexity of robust decision-making in the $\varepsilon \to 0$ limit, characterized by a problem-specific condition number $\kappa$. Here, a wide spectrum of simple algorithms, including DRO, achieve the instance-optimal rate of $O(\kappa \varepsilon^2)$. For fixed $\varepsilon > 0$, we prove that DRO still achieves the minimax rate if the problem has bounded residual effective rank. On the other hand, we identify significant failure modes where DRO is provably suboptimal by dimension-dependent factors. Fortunately, we prove that t
    
[^4]: CausalWAM：在世界动作模型中学习动作的因果成因

    CausalWAM: Learning the Causes of Actions in World Action Models

    [https://arxiv.org/abs/2610.42096](https://arxiv.org/abs/2610.42096)

    该论文提出CausalWAM，从理论上证明在稀疏性约束下，可将引发动作的因果潜变量与预测所需的干扰变量解耦，从而确保世界动作模型仅从动作相关的潜在成因生成动作。

    

    世界动作模型从观测中学习潜变量，并利用这些潜变量来预测场景的演化以及生成动作。预测必须针对整个场景，而场景中只有一部分内容会引发动作。通常来说，通过演示数据拟合的世界动作模型并不能保证仅从演示动作的潜在成因（即与动作相关的潜变量）来生成动作，因为模型完全可能以这种方式学习模仿，同时将与动作相关的潜变量与干扰变量纠缠在一起。因此，我们将引发动作的因素与预测所依赖的因素分离开来。我们证明，在稀疏性约束下，用于生成动作的潜变量仅是动作相关潜变量的可逆函数，且真实结构在潜变量置换的意义下可以被识别。这既不需要充分的时间依赖关系（摘要在此处被截断）。

    arXiv:2610.42096v1 Announce Type: cross  Abstract: World action models (WAMs) learn latent variables from observations and use these to both predict the evolution of the scene and to generate actions. The prediction has to be of the whole scene, whereas only a portion of the scene causes actions. WAMs fitted to demonstrations do not, in general, guarantee that they will generate actions solely from the latent causes of the demonstrated actions (the action-relevant latent variables), as the model could just as well learn to imitate in this way whilst entangling the action-relevant latent variables with the nuisance variables. We therefore separate out what causes action from what prediction depends on. We prove that under a sparsity constraint the latent variables used to generate actions are an invertible function of the action-relevant latent variables alone and the true structure is identified up to a permutation of the latent variables. This requires neither sufficient temporal depe
    
[^5]: 面向冻结-解冻贝叶斯优化的成本感知停止：单步前瞻何时足够？

    Cost-Aware Stopping for Freeze-Thaw Bayesian Optimization: When Does One-Step Lookahead Suffice?

    [https://arxiv.org/abs/2610.42085](https://arxiv.org/abs/2610.42085)

    该论文为冻结-解冻贝叶斯优化提出了一种成本感知的EI停止规则，证明其期望效用不低于立即停止，并构建两步前瞻价值下界证书来识别过早停止；实验表明在高成本场景下单步前瞻通常已足够，而低成本场景下修正方法更有价值。

    

    冻结-解冻贝叶斯优化（BO）以可恢复的增量方式评估候选对象，但通常在预算耗尽或收敛性检验时才停止。我们最大化一种不依赖预算的效用函数——即最佳观测得分减去每次增量的成本——并采用单步前瞻：在固定代理模型下推进期望改进（EI）最大的候选对象，一旦没有任何EI超过成本便停止。这一规则称为EI停止，其期望效用从不低于立即停止，但它可能过早停止。我们构建了一个证书——即两步前瞻价值的下界——用以证明此类停止是过早的，且在代理模型下重复这一修正过程从不会降低期望效用。在代理模型下，EI停止在低成本时平均而言是过早的，而在高成本时损失很小。在三个超参数优化基准上，一项预先注册的测试发现，在大多数高成本设置中，EI停止不劣于修正方法，而修正方法在……

    arXiv:2610.42085v1 Announce Type: new  Abstract: Freeze-thaw Bayesian optimization (BO) evaluates candidates in resumable increments but usually stops at a budget or convergence test. We maximize a budget-free utility, the best observed score minus a cost per increment, with a one-step lookahead: advance the candidate with the largest expected improvement (EI) under a fixed surrogate, and stop once no EI exceeds the cost. This rule, EI stopping, never falls below stopping immediately in expected utility, yet it can stop too early. We construct a certificate, a lower bound on the value of two-step lookahead, that proves such a stop premature, and repeating this correction never lowers expected utility under the surrogate. Under the surrogate, EI stops are premature on average at low costs and lose little at high costs. On three hyperparameter optimization benchmarks, a pre-registered test finds EI stopping noninferior to the correction in most high-cost settings, while the correction ga
    
[^6]: 个体光滑性条件下近最优的二阶凸有限和优化

    Near-Optimal Second-Order Convex Finite-Sum Optimization under Individual Smoothness

    [https://arxiv.org/abs/2610.42040](https://arxiv.org/abs/2610.42040)

    本文证明了个体光滑性与凸性假设可将二阶凸有限和优化的复杂度下界改进为 $\Omega(n + n^{5/7} \epsilon^{-2/7})$，并通过提出新颖的二阶自界定不等式和泰勒修正惰性牛顿方法，结合最优 Monteiro-Svaiter 加速，实现了与下界仅差对数因子的近最优算法复杂度。

    

    我们研究了求解具有 n 个分量的光滑凸有限和优化问题的随机二阶预言机复杂度。最近，Tsedenov 与 Kovalev（2026）通过建立几乎匹配的上界与下界，证明了在平均光滑性和凸性条件下的极小极大最优复杂度为 $\tilde{\Theta}(n + n^{6/7} \epsilon^{-2/7})$。在本文中，我们证明了个体光滑性和凸性能够带来可证明的进一步加速，在这一更强假设下，复杂度下界为 $\Omega(n + n^{5/7} \epsilon^{-2/7})$。随后，通过建立一种新颖的二阶自界定不等式，我们设计了一种新的泰勒修正惰性牛顿方法，通过应用最优的 Monteiro-Svaiter 加速技术，可以达到更好的复杂度 $\tilde{\mathcal{O}}(n + n^{5/7} \epsilon^{-2/7})$，该结果与我们的下界仅相差对数因子。

    arXiv:2610.42040v1 Announce Type: cross  Abstract: We study the stochastic second-order oracle complexity of solving smooth convex finite-sum optimization with $n$ components. Very recently, Tsedenov \& Kovalev (2026) showed that the minimax optimal complexity under averaged smoothness and convexity is $\tilde{\Theta}(n + n^{6/7} \epsilon^{-2/7})$ by establishing nearly matching upper and lower bounds. In this paper, we show that individual smoothness and convexity provably enable further acceleration, where the lower complexity bound is $\Omega(n+ n^{5/7} \epsilon^{-2/7})$ under this stronger assumption. Then, by establishing a novel second-order self-bounding inequality, we design a new Taylor-corrected lazy Newton method, which can lead to a better complexity of $\tilde{\mathcal{O}}(n + n^{5/7} \epsilon^{-2/7})$ by applying the optimal Monteiro-Svaiter acceleration, matching our lower bound up to only logarithmic factors.
    
[^7]: 论图组合检验问题中的统计-计算鸿沟

    On Statistical-Computational Gaps in Graph Combinatorial Testing Problems

    [https://arxiv.org/abs/2610.41995](https://arxiv.org/abs/2610.41995)

    本文刻画了图组合假设检验问题中强检测所需信号强度的精确统计阈值与计算阈值，并首次识别出多种植团、超团、粗循环和粗路径等问题上新的统计-计算鸿沟。

    

    我们研究高维图组合假设检验问题，其中信号支撑在一个未知子图的边集上，该子图抽自一个在顶点重标记下保持不变的图类。对于高斯观测数据，我们在多种组合图备择假设下，刻画了强检测所需信号强度的精确统计阈值与计算阈值（在多项式阶上）。我们的分析基于利用两个独立潜在结构之间的重叠来控制完全似然比与低度似然比的二阶矩，并将由此得到的下界与扫描检验、平均检验和逐坐标最大值检验进行比较。在适当的低度计算难度猜想下，我们获得了与高斯子矩阵检测中已知阈值相匹配的种植团阈值，并识别出多种植团、超团、粗循环和粗路径等问题上新的统计-计算鸿沟。

    arXiv:2610.41995v1 Announce Type: cross  Abstract: We study high-dimensional graph combinatorial hypothesis testing problems in which a signal is supported on the edge set of an unknown subgraph drawn from a class invariant under vertex relabeling. For Gaussian observations, we characterize sharp statistical and computational thresholds (at polynomial order) for the signal magnitude required for strong detection across various combinatorial graph alternatives. Our analysis is based on bounding the second moments of the full and low-degree likelihood ratios in terms of the overlap of two independent latent structures and comparing the resulting lower bounds with scan, average, and coordinatewise maximum tests. Under an appropriate low-degree conjecture for computational hardness, we obtain planted-clique thresholds matching those known for Gaussian submatrix detection and identify new statistical-computational gaps for multiple planted cliques and hypercliques, thick cycles and thick pa
    
[^8]: WASABI：基于桥插值的Wasserstein采样

    WASABI: Wasserstein Sampling via Bridge Interpolants

    [https://arxiv.org/abs/2610.41958](https://arxiv.org/abs/2610.41958)

    提出WASABI方法，利用Wasserstein梯度流和桥插值训练扩散模型，仅凭能量函数即可从玻尔兹曼分布中采样，且避免了难以计算的对数密度梯度，训练完成后无需再评估能量或其梯度。

    

    从玻尔兹曼分布中采样是物理学、化学和生物学中的一个基本问题。我们提出了一种通过桥插值进行Wasserstein采样的方法（WASABI），该方法通过训练扩散模型，在仅能访问能量函数的情况下，从目标玻尔兹曼分布中生成样本。WASABI利用Wasserstein梯度流来最小化其扩散模型生成的分布与玻尔兹曼目标分布之间的KL散度。该方法建立在经典采样方法（如朗之万动力学）所基于的原理之上。然而，将Wasserstein梯度流应用于扩散模型具有挑战性，因为更新过程需要模型自身对数密度的梯度，而这通常是难以计算的。我们推导出了一种实用的训练算法，避免了计算这一梯度。一旦训练完成，WASABI无需进一步评估能量函数或其梯度即可生成新样本。在丙氨酸二肽上，W

    arXiv:2610.41958v1 Announce Type: cross  Abstract: Sampling from Boltzmann distributions is a fundamental problem in physics, chemistry, and biology. We introduce WAsserstein SAmpling via Bridge Interpolants (WASABI), a method that trains a diffusion model to generate samples from a target Boltzmann distribution given access only to the energy function. WASABI minimizes the KL divergence between the distribution generated by its diffusion model and the Boltzmann target using Wasserstein gradient flows. This approach builds on a principle underlying classical sampling methods such as Langevin dynamics. However, applying Wasserstein gradient flows to diffusion models is challenging because the updates require the gradient of the model's own log density, which is generally intractable. We derive a practical training algorithm that avoids computing this gradient. Once trained, WASABI can generate new samples without further evaluations of the energy or its gradient. On alanine dipeptide, W
    
[^9]: Chronos-2：一个用于多变量时间序列预测的基础模型

    Chronos-2: A foundation model for multivariate time series forecasting

    [https://arxiv.org/abs/2610.41956](https://arxiv.org/abs/2610.41956)

    Chronos-2是一个零样本多变量时间序列预测基础模型，通过大规模预训练能够建模时间序列之间及其与异构协变量之间的交互，直接从历史数据中推断关系并生成带有不确定性量化的概率预测。

    

    在从流行病学、经济学到能源系统和地球科学等各种科学应用中，带有不确定性量化的准确预测至关重要。然而，大多数预测模型仍需要针对每个特定领域的数据集单独开发和训练。时间序列基础模型（TSFM）旨在消除这一限制，通过大规模预训练使同一模型能够跨领域应用。然而，大多数现有的时间序列基础模型将每条时间序列独立处理，因此无法建模时间序列之间以及时间序列与外部驱动因素之间的相互作用。本文提出了Chronos-2，一个用于零样本预测的基础模型，它能够建模多变量时间序列与异构协变量之间的交互，而这在科学应用中十分常见。该模型直接从历史上下文中推断时间序列之间的关系，并生成概率预测，无需任务……

    arXiv:2610.41956v1 Announce Type: new  Abstract: Accurate forecasting with uncertainty quantification is critical in a variety of scientific applications ranging from epidemiology and economics to energy systems and earth sciences. Yet most forecasting models still need to be developed and trained separately for each domain-specific dataset. Time series foundation models (TSFMs) aim to remove this limitation and enable the application of the same model across domains through large-scale pretraining. However, most existing TSFMs treat each time series independently and therefore cannot model interactions among time series and external drivers. Here, we introduce Chronos-2, a foundation model for zero-shot forecasting that models the interactions between multivariate time series and heterogeneous covariates, which commonly arise in scientific applications. The model infers relationships among time series directly from historical context and generates probabilistic forecasts without task-
    
[^10]: 强化学习的统计基础：非渐近视角

    Statistical Foundations of Reinforcement Learning: A Non-Asymptotic Perspective

    [https://arxiv.org/abs/2610.41903](https://arxiv.org/abs/2610.41903)

    本文系统阐述了强化学习的非渐近统计理论基础，聚焦于在有限样本情形下理解并提升强化学习算法的数据效率这一核心问题。

    

    强化学习（RL）体现了在不确定性条件下进行序贯决策的基础性范式。新兴应用中模型复杂度的快速增长，显著加剧了实现数据高效强化学习的挑战，尤其是在样本匮乏的场景中——这些场景下数据收集可能成本高昂、耗时漫长，甚至涉及高风险。因此，理解并提升强化学习算法的数据效率，已成为一个兼具基础理论意义和实际重要性的核心问题。尽管经过数十年的研究，强化学习的统计基础直到近期仍相当不完善，特别是在与现代实践最为相关的有限样本情形下。过去几年中，借鉴高维统计学、概率论和信息论的思想，推动非渐近统计强化学习理论前沿的研究活动大量涌现……

    arXiv:2610.41903v1 Announce Type: cross  Abstract: Reinforcement learning (RL) embodies a foundational paradigm for sequential decision-making in the face of uncertainty. The rapid growth of model complexity in emerging applications has significantly amplified the challenges of achieving data-efficient RL, particularly in sample-starved scenarios where data collection may be costly, time-consuming, or even high-stakes. Consequently, understanding and improving the data efficiency of RL algorithms has become a central problem of both fundamental theoretical interest and practical importance.   Despite decades of research, the statistical underpinnings of RL remained remarkably underdeveloped until relatively recently, particularly in the finite-sample regimes most relevant to modern practice. Over the past few years, there has been a surge of activity advancing the frontiers of non-asymptotic statistical RL theory, drawing on ideas from high-dimensional statistics and probability, infor
    
[^11]: 跨越 Kesten-Stigum 区域的 Bethe Hessian 恢复

    Bethe Hessian Recovery through the Kesten-Stigum Regime

    [https://arxiv.org/abs/2610.41802](https://arxiv.org/abs/2610.41802)

    本文证明对于任意固定期望度 d>1 的随机块模型，Bethe Hessian 的离群特征向量都能实现直至信息论最优的 Kesten-Stigum 阈值的弱恢复，仅用厄米特矩阵即匹配了非回溯谱方法的恢复保证。

    

    我们证明，对于每个固定的公共期望度 $d>1$，Bethe Hessian 的离群特征向量在随机块模型中能够实现弱恢复，直至 Kesten-Stigum 阈值。这扩展了以往需要 $d$ 足够大的恢复保证。该阈值在对称双社区模型中是信息论意义上最优的，并被推测标志着更广泛对称模型中高效恢复能力的极限。我们的结果在使用厄米特（Hermitian）$n\times n$ 矩阵的同时，达到了与非回溯谱方法相当的恢复保证。为此，我们给出了负离群特征值的递归公式，并证明这些特征空间与相应的种植特征空间之间存在非零的重叠。证明过程通过在极限泊松树上利用种植标签构造近似特征向量（其权重由格林函数给出），并将这些近似结果转移至……

    arXiv:2610.41802v1 Announce Type: cross  Abstract: We show that outlier eigenvectors of the Bethe Hessian achieve weak recovery in the stochastic block model down to the Kesten-Stigum threshold, for every fixed common expected degree $d>1$. This extends previous recovery guarantees, which required $d$ to be sufficiently large. This threshold is information-theoretically optimal for the symmetric two-community model and is conjectured to mark the limit of efficient recovery in broader symmetric models. Our result matches the recovery guarantees of non-backtracking spectral methods while using Hermitian $n\times n$ matrices.   To do this, we identify recursive formulas for the negative outlier eigenvalues and show that their eigenspaces have nonvanishing overlap with the corresponding planted eigenspaces. The proof constructs approximate eigenvectors from the planted labels on the limiting Poisson tree, with weights given by its Green's function, and transfers these approximations to the
    
[^12]: 偏微分方程残差信息克里金法

    PDE-Residual-Informed Kriging

    [https://arxiv.org/abs/2610.41590](https://arxiv.org/abs/2610.41590)

    本文提出基于PDE残差核的PDE残差信息克里金法（PRIK），将整个定义域上的全局PDE残差信息直接融入高斯过程的协方差结构，在保持模型灵活性与标准GP条件化的同时，无需额外的残差权重参数。

    

    物理信息高斯过程（GP）方法为偏微分方程（PDE）求解提供了概率框架。现有方法以不同方式引入PDE信息：逐点条件化方法仅在有限个选定位置处使用控制方程；全局约束GP方法将可容许函数限制为严格满足PDE；残差惩罚方法则通过附加的惩罚权重控制整个定义域上的PDE失配。目前仍然缺失的是一种GP构造，能够将整个定义域上的全局PDE残差信息直接融入其协方差结构，同时保持模型的灵活性和标准的GP条件化方式，且无需引入单独的残差权重参数。为解决这一问题，本研究提出基于PDE残差核的PDE残差信息克里金法（PRIK）。PDE残差核通过在核截面空间上装备由 $L^{（摘要在此处被截断，后续内容不完整）

    arXiv:2610.41590v1 Announce Type: cross  Abstract: Physics-informed Gaussian process (GP) methods provide a probabilistic framework for PDE solving. Existing approaches incorporate PDE information in different ways: pointwise conditioning uses the governing equation only at finitely many selected locations, globally constrained GPs restrict the admissible functions to satisfy the PDE exactly, and residual-penalized methods control the PDE mismatch over the whole domain through an additional penalty weight. What is still missing is a GP formulation that directly incorporates global PDE-residual information of the entire domain into its covariance structure, while retaining model flexibility and standard GP conditioning without a separate residual-weight parameter. To solve this issue, this study proposes PDE-residual-informed Kriging (PRIK) based on a PDE-residual kernel. The PDE-residual kernel is constructed by equipping a kernel-section space with an inner product defined by the $L^{
    
[^13]: 可靠性加权直接偏好优化：用于稳健偏好学习的惩罚化权重

    Reliability-Weighted Direct Preference Optimization: Penalized Weights for Robust Preference Learning

    [https://arxiv.org/abs/2610.41588](https://arxiv.org/abs/2610.41588)

    提出RW-DPO方法，通过为每条偏好比较引入可学习的可靠性权重并结合极小极大凹惩罚，以几乎不增加DPO计算成本的方式自动识别并降权噪声比较，实现稳健的偏好学习。

    

    直接偏好优化将每一条报告的比较都视为正确的，然而偏好数据中却包含反转的、平局的或其他有噪声的比较。RW-DPO 为每一条比较赋予一个未知的可靠性权重 $w_i\in[0,1]$，该权重以线性方式进入似然函数——DPO 对应于 $w\equiv1$，被反转的比较对应于 $w_i=0$——并在 $-\log w_i$ 上的极小极大凹惩罚（minimax-concave penalty）下与策略联合估计这些权重。该惩罚使模型所认同的每一条比较的权重恰好保持在 $w_i=1$，只对与模型相矛盾的比较发挥作用，并通过闭式权重更新以几乎不增加 DPO 计算成本的方式实现。我们证明，在每条比较仅有一次标注的条件下，恢复所有被翻转的比较需要 $\log n$ 量级的间隔（margin）。此外，在线性指标模型中，从指向正确方向的初始拟合出发，我们的估计量能够联合且一致地恢复偏好方向与权重。

    arXiv:2610.41588v1 Announce Type: cross  Abstract: Direct Preference Optimization (DPO) treats every reported comparison as correct, yet preference data contain   reversed, tied or otherwise noisy comparisons. RW-DPO gives each comparison an unknown reliability weight   $w_i\in[0,1]$ that enters the likelihood linearly -- DPO is $w\equiv1$, a reversed comparison $w_i=0$ -- and estimates   the weights jointly with the policy under a minimax-concave penalty on $-\log w_i$. The penalty keeps every   comparison the model agrees with at exactly $w_i=1$ and acts only on those it contradicts, through closed-form weight   updates at negligible cost over DPO. We prove that recovering all flipped comparisons from a single annotation per   comparison requires margins of order $\log n$. Furthermore, in a linear-index model, starting from an initial fit   pointing in the correct direction, our estimator jointly consistently recovers the preference direction and weights,   as well as identifies the 
    
[^14]: 面向金融时间序列生成的条件神经随机微分方程

    Conditional Neural SDEs for Financial Time-Series Generation

    [https://arxiv.org/abs/2610.41405](https://arxiv.org/abs/2610.41405)

    提出一种残差锚定的条件神经随机微分方程架构，通过将目标分解为条件锚点与随机残差，稳定地生成能够响应外部上下文的金融时间序列轨迹。

    

    时间序列路径的条件生成是概率建模中的一个核心问题，其目标是模拟既真实又能响应外部上下文的未来轨迹，在金融和经济学领域有着直接的应用。神经随机微分方程通过将连续时间随机动力学与可学习的漂移和扩散函数相结合，为这一任务提供了天然的框架。然而，在条件设定下直接对目标状态建模可能是不稳定的，尤其是当目标变量与条件变量之间存在强烈的水平依赖、训练数据有限或两者维度不同时。我们提出了一种残差锚定神经SDE架构，将目标分解为一个学习得到的条件锚点和一个随机残差：锚点捕捉依赖于上下文的基线，而神经SDE（随机微分方程）则对该基线附近的残差动态进行建模。

    arXiv:2610.41405v1 Announce Type: cross  Abstract: Conditional generation of time-series paths is a central problem in probabilistic modeling, where the goal is to simulate future trajectories that are both realistic and responsive to external context. It has direct applications in finance and economics. Neural stochastic differential equations provide a natural framework for this task by combining continuous-time stochastic dynamics with learnable drift and diffusion functions. However, directly modeling the target state can be unstable in conditional settings, especially when the target and conditioning variables have strong level dependence, limited training data, or different dimensions.We propose a residualanchor Neural SDE architecture that decomposes the target into a learned conditional anchor and a stochastic residual. The anchor captures the context-dependent baseline, while the Neural SDE (Stochastic Differential Equation) models the residual dynamics around this baseline. T
    
[^15]: 面向流匹配无需滚动后训练的粗粒度最优传输方法

    Coarse-Graining Optimal Transport for Rollout-Free Post-Training of Flow Matching

    [https://arxiv.org/abs/2610.41269](https://arxiv.org/abs/2610.41269)

    提出 ReCouple 可控框架，通过构造粗粒度最优传输耦合来实现流匹配的无需滚动后训练，并扩展出面向连续源设置的 ReCouple+ 与 ReCouple++，从而在低函数评估次数下改善采样质量。

    

    arXiv:2610.41269v1 公告类型： cross 摘要：流匹配在很大程度上依赖于用于配对源样本与目标样本的耦合方式。最优传输（OT）耦合可以减少速度冲突并拉直轨迹；但在实践中仍存在挑战：源数据来自连续分布，可获取的目标样本数据数量有限，且神经网络的容量有限。出于这些原因，让耦合更接近最优传输并不一定能持续改善采样质量，尤其是在低函数评估次数（NFE）的场景下。受此启发，我们提出了 ReCouple，一个可控的框架，通过构建粗粒度的最优传输耦合，以更好地理解基于最优传输的无需滚动后训练。我们还考虑了该基本框架的扩展：ReCouple+ 通过提高源分辨率将该框架扩展到连续源设置，并在适当的极限下收敛到半离散最优传输；ReCouple++ 则进一步增加了空间重（原文摘要在此处截断）

    arXiv:2610.41269v1 Announce Type: cross  Abstract: Flow matching depends critically on the coupling used to pair source and target samples. Optimal transport (OT) couplings can reduce velocity conflict and straighten trajectories; but, in practice, challenges remain: the source comes from a continuous distribution, only a finite number of target samples data are accessible, and the neural network has finite capacity. For these reasons, making the coupling more OT-like may not consistently improve sampling quality, in particular in the low number-of-function-evaluation (NFE) regime. Motivated by this, we introduce ReCouple, a controllable framework that constructs coarse-grained OT couplings, to better understand rollout-free OT-based post-training. We also consider extensions of this basic framework: ReCouple+ extends this toward the continuous-source setting by increasing source resolution, converging to semi-discrete OT in the appropriate limit; and ReCouple++ further adds spatial re
    
[^16]: 面向解算子的工具变量迁移学习方法

    Instrumental Transfer Learning for Solution Operators

    [https://arxiv.org/abs/2610.41262](https://arxiv.org/abs/2610.41262)

    提出了一种基于预解恒等式和白化工具变量的三阶段结构化迁移学习方法，能够在目标系统观测稀缺时，利用相关源系统数据无偏地学习微分方程的解算子。

    

    微分方程解算子的数据驱动学习具有广泛的科学意义，但来自目标系统的观测数据往往稀缺且采集成本高昂。相关的源系统在相似物理条件下可以提供大量观测数据，然而控制方程即使发生微小变化，也可能引起解算子的复杂变化，使得简单的数据池化方法失效。我们提出了一种结构化迁移学习方法，该方法能够处理控制方程中相对于已知候选项库的未知稀疏差异。利用预解恒等式和白化工具变量方法，我们为目标算子构造了无偏的源估计方程。基于这些方程，我们开发了一个三阶段估计器，包括仅使用目标数据的岭回归初步估计、按源系统的稀疏校准，以及旨在消除主要校准误差的投影重拟合。

    arXiv:2610.41262v1 Announce Type: cross  Abstract: Data-driven learning for the solution operator of a differential equation has broad scientific importance, but observations from a target system are often scarce and costly to collect. Related source systems may provide abundant observations under similar physical conditions, whereas even small changes in the governing equation can induce complex changes in the solution operator, making naive data pooling ineffective. We propose a structured transfer learning method that accounts for unknown sparse differences in the governing equations from a known library of candidate terms. Using the resolvent identity and a whitened instrumental variable approach, we construct unbiased source estimating equations for the target operator. Building on these equations, we develop a three-stage estimator that consists of a target-only ridge pilot, source-wise sparse calibration, and a projected refit designed to remove the leading calibration error inh
    
[^17]: 相关观测下的随机镜像下降

    Stochastic Mirror Descent under Dependent Observations

    [https://arxiv.org/abs/2610.41251](https://arxiv.org/abs/2610.41251)

    该论文突破了传统独立同分布假设，通过引入适用于几何或多项式相关序列的灵活“遗忘”条件，在相关观测下建立了随机镜像下降算法的显式有限样本收敛界，并恢复了经典的 O(1/T) 与 O(1/√T) 收敛速率。

    

    我们研究了当带噪一阶预言机由相关观测驱动时，随机镜像下降（SMD）算法的非渐近收敛性。与现有大多数理论保证所依赖的经典独立同分布（i.i.d.）假设不同，我们引入了一个灵活的“遗忘”条件，该条件能够容纳几何相关或多项式相关的序列。在该条件下，我们推导出了平均化SMD迭代点的显式有限样本精度界，在大初始误差和小初始误差两种情形下分别恢复了经典的 O(1/T) 和 O(1/√T) 收敛速率。我们将该通用框架应用于两个具体实例：具有相关回归变量的高维线性回归，以及函数型自回归模型漂移函数的非参数估计。

    arXiv:2610.41251v1 Announce Type: cross  Abstract: We study the non-asymptotic convergence of Stochastic Mirror Descent (SMD) algorithm when the noisy first-order oracle is driven by dependent observations. Departing from the classical i.i.d. assumption underlying most existing guarantees, we introduce a flexible ``forgetting" condition that accommodates geometrically or polynomially dependent sequences. Under this condition we derive explicit finite-sample accuracy bounds for the averaged SMD iterates, recovering the classical $O(1/T)$ and $O(1/\sqrt{T})$ rates in the large- and small-initial-error regimes. We instantiate the general framework on two applications: high-dimensional linear regression with dependent regressors and nonparametric estimation of the drift function of a functional autoregressive model.
    
[^18]: 数据依赖事件概率估计的统一框架

    A Unified Framework for Probability Estimation of Data-Dependent Events

    [https://arxiv.org/abs/2610.41244](https://arxiv.org/abs/2610.41244)

    该论文提出了一个统一框架来估计由样本数据经映射决定的事件概率，该框架同时涵盖缺失质量、自适应数据分析目标与模型测试误差等经典问题，并证明了代入估计器的误差恰为泛化差距、留一估计器对稳定映射均准确。

    

    我们研究估计 $D(\dde(S))$ 的一般性问题，即未知分布 $D$ 赋予某一事件的概率，该事件由来自 $D$ 的 $n$ 个独立同分布样本 $S$ 通过已知映射 $\dde$ 所确定。估计算法知道 $\dde$ 和 $S$，但无法获得来自 $D$ 的任何额外样本。这一表述涵盖了经典的缺失质量（missing mass）问题、自适应数据分析和后选择推断中的数据依赖目标，以及数据训练预测器的测试误差，后者的估计最近由 Gastpar 等人（2024a,b）研究。在该框架内，我们考察了两种经典的估计方法。经验代入（plug-in）估计器的误差恰好是一个泛化差距，因此泛化界限可以直接迁移到它上面。留一（leave-one-out）估计器可恢复出 Good–Turing 估计器，对于每一个在删除单个样本点下保持稳定的映射都是准确的，包括 ne…（摘要在此处截断）

    arXiv:2610.41244v1 Announce Type: new  Abstract: We study the general problem of estimating $D(\dde(S))$, the probability that an unknown distribution $D$ assigns to an event determined by a sample $S$ of $n$ i.i.d.\ draws from $D$ through a known mapping $\dde$. The estimation algorithm knows $\dde$ and $S$ but does not have access to any additional samples from $D$. This formulation captures classical missing mass, the data-dependent targets of adaptive data analysis and post-selection inference, and the test error of a data-trained predictor, whose estimation was recently studied by \new{Gastpar et al.\ (2024a,b)}. Within this framework we examine the two classical approaches to estimation. The error of the empirical plug-in estimator is exactly a generalization gap, so generalization bounds transfer to it directly. The leave-one-out estimator, which recovers the Good--Turing estimator, is accurate for every mapping that is stable under the deletion of one sample point, including ne
    
[^19]: 基于保形化单纯神经网络的单纯复形上的自适应不确定性量化

    Adaptive Uncertainty Quantification on Simplicial Complexes with Conformalized Simplicial Neural Networks

    [https://arxiv.org/abs/2610.41166](https://arxiv.org/abs/2610.41166)

    提出保形化单纯神经网络（CF-SNN），利用保形预测为单纯复形上的多个预测目标同时生成具有覆盖率保证的预测集，为SNNs提供了严格的不确定性量化能力。

    

    单纯神经网络（SNNs）是处理存储在单纯复形（SCs）——一种图的高阶泛化形式——上的高阶关系数据集预测任务的强大工具。然而，SNNs通常缺乏严格的不确定性量化（UQ），这限制了它们在高风险场景中的实用性。我们提出了保形化SNN（CF-SNN），这是一个框架：给定一个具有m个预测目标的单纯复形实体，它能够同时为所有m个目标生成预测集。在温和的假设下，CF-SNN保证对于每个目标，预测集以用户指定的名义覆盖率包含真实标签，并且保证所有m个目标上的实际未覆盖比例以高概率保持在名义水平附近。在广泛的SC预测任务上的大量实验表明，CF-SNN是一个灵活的不确定性量化框架，能够在实现用户指定覆盖率的同时，产生（紧凑的预测集）……

    arXiv:2610.41166v1 Announce Type: cross  Abstract: Simplicial neural networks (SNNs) are powerful tools for prediction tasks on higher-order relational datasets stored on simplicial complexes (SCs), a higher-order generalization of graphs. However, SNNs usually lack rigorous uncertainty quantification (UQ), limiting their usefulness in high-stakes scenarios. We propose conformalized SNN (CF-SNN), a framework that, given a simplicial-complex entity with m prediction targets, generates prediction sets for all m targets simultaneously. Under mild assumptions, CF-SNN guarantees that, for each target, the prediction set contains the true label with a user-specified nominal coverage probability, and the realized miscoverage proportion across all m targets is guaranteed to remain close to the nominal level with high probability. Extensive experiments across a wide range of SC prediction tasks demonstrate that CF-SNN is a flexible UQ framework that achieves user-specified coverage while produc
    
[^20]: MUON-RT：重用教师轨迹实现无动量Muon的大语言模型自蒸馏

    MUON-RT Reusing the Teacher's Trajectory for Momentum-Free Muon for LLM Self-Distillation

    [https://arxiv.org/abs/2610.41104](https://arxiv.org/abs/2610.41104)

    提出Muon-RT，通过重用自蒸馏中教师与学生的位移替代Muon优化器的动量缓冲区，在降低训练内存开销的同时保证了理论收敛性，并提升大语言模型自蒸馏微调的性能。

    

    对于采用指数移动平均（EMA）教师的大语言模型自蒸馏方法而言，学生更新的历史记录本身已被保留。然而，Muon优化器仍需存储一个单独的动量缓冲区，从而增加了持续的训练内存开销。我们提出了重用教师的Muon（Muon-RT），该方法通过重用教师-学生位移来消除隐藏层权重矩阵的动量缓冲区。我们证明，该位移编码了已应用更新的指数加权历史，并在局部模型下平滑了时间上不相关的噪声。Muon-RT去除该位移的径向分量，对其进行谱归一化，并在Newton-Schulz（NS）正交化之后将其与当前梯度混合。在明确的正则性与对齐性假设下，我们建立了期望目标差距界向残差误差水平的线性收缩。通过在Qwen2.5-7B-Instruct上进行自蒸馏微调（SDFT），Muon-RT相比AdamW等基线方法提升了平均准确率（摘要原文在此处截断）。

    arXiv:2610.41104v1 Announce Type: cross  Abstract: Self-distillation methods for large language models with an exponential moving average (EMA) teacher already retain a history of student updates. Muon nevertheless stores a separate momentum buffer, increasing persistent training memory. We propose Muon with Reused Teacher (Muon-RT), which reuses the teacher-student displacement to eliminate momentum buffers for hidden weight matrices. We show that this displacement encodes an exponentially weighted history of applied updates and smooths temporally uncorrelated noise under a local model. Muon-RT removes its radial component, normalizes it spectrally, and mixes it with the current gradient after Newton-Schulz (NS) orthogonalization. Under explicit regularity and alignment assumptions, we establish linear contraction of an expected objective-gap bound toward a residual error level. With Self-Distillation Fine-Tuning (SDFT) on Qwen2.5-7B-Instruct, Muon-RT improves average accuracy over Ad
    
[^21]: 何处、何时、以及多少比特？随机舍入下SGD的缩放定律

    Where, When, and How Many Bits? Scaling Laws for SGD with Stochastic Rounding

    [https://arxiv.org/abs/2610.41074](https://arxiv.org/abs/2610.41074)

    本文通过两层线性网络的理论分析，证明了随机舍入下低精度SGD训练预测误差的匹配上下界，并给出了达到最优计算缩放时所需的模型规模、训练时间与精度调度选择规律。

    

    低精度训练可以减少计算量，但会引入额外的舍入误差。我们从理论上研究了随着模型规模、训练时间和精度的变化，随机舍入（SR）如何影响预测风险。我们考虑一个第一层固定且随机的两层线性网络，在具有幂律谱的全新高斯数据上通过小批量SGD进行训练。在每一步SGD中，前向和反向计算所使用的特征被独立舍入。对于训练前预先选定的通用学习率与精度调度，我们针对最终模型以及均匀参数平均证明了预测误差的匹配上下界。对于固定精度和恒定学习率并结合参数平均的情形，我们推导出最小预测误差如何随计算量下降，并识别出达到这种最优计算缩放所需的模型规模和训练时间。仅对前向特征进行舍入可保持……的计算缩放规律（摘要在此处被截断）。

    arXiv:2610.41074v1 Announce Type: new  Abstract: Low-precision training reduces computation but introduces additional rounding error. We theoretically study how stochastic rounding (SR) affects prediction risk as model size, training time, and precision change. We consider a two-layer linear network with a fixed random first layer, trained by mini-batch SGD on fresh Gaussian data with power-law spectra. At each SGD step, the features used in forward and backward computations are rounded independently. We prove matched upper and lower bounds on prediction error for the final model and for uniform parameter averaging, with general learning-rate and precision schedules chosen before training. For fixed precisions and constant learning rates with parameter averaging, we derive how the smallest prediction error decreases with compute, and we identify model sizes and training times that achieve this optimal-compute scaling. Rounding only the forward features preserves the compute scaling of 
    
[^22]: 面向所有可混合恰当损失及更广泛情形的全能预测

    Omniprediction for All Mixable Proper Losses and Beyond

    [https://arxiv.org/abs/2610.41023](https://arxiv.org/abs/2610.41023)

    本文证明当损失类扩展为无限类时，全能预测的次线性遗憾一般无法实现——即使预测器类仅含两个线性预测器，也不可能对所有有界恰当损失同时达到次线性遗憾，从而为该问题给出否定答案。

    

    全能预测关注的是如何做出在不同损失下都能同时保持准确的预测。早期结果 [CV09] 表明，在在线学习设定下，对于有限预测器类 $\mathcal{F}$ 中的每个预测器，以及由可混合恰当损失构成的有限类 $\mathcal{L}$ 中的每种损失，可以同时实现 $O(\log(|\mathcal{F}||\mathcal{L}|))$ 的遗憾；而最近的进展 [OKK25, FP26] 表明，对于由有界损失构成的有限类 $\mathcal{L}$ 中的每种损失，可以同时实现 $\widetilde{O}(\sqrt{T\log(|\mathcal{F}||\mathcal{L}|)})$ 的遗憾。在本工作中，我们探究当 $\mathcal{L}$ 是无限恰当损失类时，次线性遗憾是否仍然可能实现。我们的第一个结果给出了一般情形下的否定答案：即使 $\mathcal{F}$ 仅包含两个线性预测器，也不可能对所有有界恰当损失同时实现次线性遗憾。随后我们展示……（原文摘要在此处被截断）

    arXiv:2610.41023v1 Announce Type: new  Abstract: Omniprediction concerns the task of making predictions that are simultaneously accurate with respect to different losses. Earlier results by [CV09] show that in an online learning setting, $O(\log(|\mathcal{F}||\mathcal{L}|))$ regret is achievable simultaneously against every predictor in a finite predictor class $\mathcal{F}$ and for every loss in a finite class $\mathcal{L}$ of mixable proper losses, while recent advances [OKK25, FP26] show that $\widetilde{O}(\sqrt{T\log(|\mathcal{F}||\mathcal{L}|)})$ regret is achievable simultaneously for every loss in a finite class $\mathcal{L}$ of bounded losses.   In this work, we ask whether sublinear regret remains possible when $\mathcal{L}$ is an infinite class of proper losses. Our first result gives a negative answer in general: even when $\mathcal{F}$ contains only two linear predictors, it is impossible to achieve sublinear regret simultaneously for all bounded proper losses. We then sho
    
[^23]: 面向潜在世界模型强化学习的密度引导自举边界

    Density-Guided Bootstrap Boundaries for Latent World-Model Reinforcement Learning

    [https://arxiv.org/abs/2610.41022](https://arxiv.org/abs/2610.41022)

    该论文提出密度引导自举边界机制，根据轨迹可靠性自适应调整想象回报的展开深度，从而在潜在世界模型强化学习中平衡模型误差累积与对评论家近似的过度依赖。

    

    潜在世界模型使强化学习智能体能够从想象的轨迹中学习策略，然而确定模型依赖的最优深度仍然是一个根本性挑战。固定的想象视野无论轨迹可靠性如何都采用统一的预测深度，导致目标值暴露于不断累积的模型误差中，或过度依赖评论家的近似。我们提出了密度引导自举边界，这是一种依赖轨迹的机制，利用分数信息引导的相对密度代理来自适应地调整想象回报的深度。该代理沿着想象路径累积由特征位移网络和噪声条件分数估计器构成的分数校正辅助场的估计散度。我们提出了两种门控机制：一种是在离散状态处停止回报展开的硬边界，另一种是将回报递归连续地转向评论家自举的软边界。两种变体都保持了……

    arXiv:2610.41022v1 Announce Type: new  Abstract: Latent world models allow reinforcement learning agents to learn policies from imagined trajectories, yet determining the optimal depth of model reliance remains a fundamental challenge. Fixed imagination horizons apply uniform prediction depths regardless of trajectory reliability, exposing target values to compounding model error or over-relying on critic approximations. We introduce density-guided bootstrap boundaries, a trajectory-dependent mechanism that adapts the depth of imagined returns using a score-informed relative density proxy. The proxy accumulates the estimated divergence of a score-corrected auxiliary field formed by a feature displacement network and a noise-conditioned score estimator along imagined paths. We propose two gating mechanisms: a hard boundary that halts return expansion at a discrete state, and a soft boundary that continuously shifts the return recursion toward critic bootstrapping. Both variants preserve
    
[^24]: 从受污染样本中进行设计上稳健的分布学习

    Robust-by-Design Distributional Learning from Contaminated Samples

    [https://arxiv.org/abs/2610.40884](https://arxiv.org/abs/2610.40884)

    该论文提出污染校正的最大均值差异估计量（CC-MMD），通过将带噪声的代理分数与小规模已审计子集的精确标签校正相结合，从受污染的混合样本中稳健地学习潜在的目标分布。

    

    现代数据分析流程日益依赖于通过检索、抓取、自动过滤和事后整理构建的大型数据集，在这些场景下，观测到的样本往往是一个受污染的混合体，而非目标总体。标准的分布学习目标要么忽略污染并收敛到错误的目标，要么仅保留一个小的可信子集而丢弃大部分数据。我们研究有监督与无监督学习问题：将一个干净的参考样本与来自 $r=(1-\epsilon)q+\epsilon z$ 的混合样本进行比较，其中 $q$ 是潜在的目标分布，$z$ 是未知的污染物。受如下场景启发——所有观测都具有带噪声的成员资格分数，而一个小的已审计子集具有精确标签——我们提出了污染校正的最大均值差异（CC-MMD），这是一种基于设计的估计量，它将代理分数与已审计的残差校正相结合，以恢复本应得到的预言机 MMD（原文摘要在此处截断）。

    arXiv:2610.40884v1 Announce Type: cross  Abstract: Modern data analysis pipelines increasingly rely on large datasets assembled through retrieval, scraping, automated filtering, and post-hoc curation, where the observed sample is often a contaminated mixture rather than the target population. Standard distributional objectives either ignore contamination and converge to the wrong target, or retain only a small trusted subset and discard most of the data. We study supervised and unsupervised learning when a clean reference sample is compared with a pooled sample from $r=(1-\epsilon)q+\epsilon z$, where $q$ is the latent target distribution and $z$ is an unknown contaminant. Motivated by settings with noisy membership scores for all observations and exact labels for a small audited subset, we propose contamination-corrected maximum mean discrepancy (CC-MMD), a design-based estimator that combines proxy scores with audited residual corrections to recover the oracle MMD that would have bee
    
[^25]: 带反馈图的赌博机的近最优一阶遗憾

    Near-Optimal First-Order Regret for Bandits with Feedback Graphs

    [https://arxiv.org/abs/2610.40861](https://arxiv.org/abs/2610.40861)

    本文针对固定已知有向反馈图的对抗性赌博机，提出了期望遗憾界为 $\widetilde{O}(\sqrt{\alpha L^\star}+\alpha)$ 的算法，将一阶遗憾的图依赖性从团划分数改进到独立数，从而弥合了最坏情况遗憾与一阶遗憾之间的差距，达到了近最优的一阶遗憾保证。

    

    我们研究了具有固定且已知有向反馈图的对抗性赌博机问题的一阶遗憾保证。对于强可观测图，[ACBDK15] 建立了极小极大遗憾为 $\widetilde{\Theta}(\sqrt{\alpha T})$ 的结果，其中 $\alpha$ 表示图的独立数，$T$ 表示时间范围。另一方面，现有的最佳一阶上界为 $\widetilde{O}(\sqrt{(\kappa+1) L^\star})$ [LLZ20]，其中 $\kappa$ 表示团划分数，满足 $\kappa+1\geq\alpha$，$L^\star$ 表示最佳臂的累积损失，这在最坏情况保证与一阶保证的图依赖性之间留下了差距。在本文中，我们通过获得期望遗憾界 $\widetilde{O}(\sqrt{\alpha L^\star}+\alpha)$ 来弥合这一差距。我们的第一个结果是针对自感知图（即每个顶点都有自环）这一特殊情况的一种算法。该算法将 EXP3 [ACBFS02] 的更新与一种相结合

    arXiv:2610.40861v1 Announce Type: new  Abstract: We study first-order regret guarantees for adversarial bandits with a fixed, known directed feedback graph. For strongly observable graphs, [ACBDK15] establishes a minimax regret of $\widetilde{\Theta}(\sqrt{\alpha T})$, where $\alpha$ denotes the independence number of the graph and $T$ denotes the horizon. On the other hand, the best existing first-order bound scales as $\widetilde{O}(\sqrt{(\kappa+1) L^\star})$ [LLZ20], where $\kappa$ denotes the clique partition number, satisfying $\kappa+1\geq\alpha$, and $L^\star$ denotes the cumulative loss of the best arm, leaving a gap between the graph dependence in worst-case and first-order guarantees. In this paper, we bridge this gap by obtaining an expected regret bound of $\widetilde{O}(\sqrt{\alpha L^\star}+\alpha)$. Our first result is an algorithm for the special case of self-aware graphs, in which every vertex has a self-loop. The algorithm combines the update of EXP3 [ACBFS02] with a
    
[^26]: 基于留出锚点的持续适应认证行为保持

    Certified Behavioral Retention for Continual Adaptation with Held-Out Anchors

    [https://arxiv.org/abs/2610.40740](https://arxiv.org/abs/2610.40740)

    该论文提出CABA方法，通过在留出锚点上比较新旧策略、筛选最坏锚点前向KL散度不超预算的最大插值系数，在无需动作标签或完整轨迹的情况下为持续适应中的模型更新提供可认证的行为保持保证。

    

    持续适应不仅需要决定如何更新模型，还需要决定是否在不超出针对受保护行为所声明的变更预算的前提下提交该更新。我们提出了认证锚点适应（CABA），这是一种与候选无关的更新后门控机制，它将历史观测数据保留在候选优化之外，在冻结的表示上比较旧策略与候选策略，并选出其最坏锚点前向KL散度仍保持在预算内的最大已测试插值系数。该有限检验既不需要动作标签，也不需要完整轨迹。我们开发了一个以理论为主导的认证框架，将实际测量的锚点保持性与条件性的锚点外保证分离开来。在明确的表示覆盖和Lipschitz假设下，被接受的更新在受保护支持集上满足一致的行为散度界。该分析为仿射softmax头给出了正则性界，概率……（摘要在此处被截断）

    arXiv:2610.40740v1 Announce Type: new  Abstract: Continual adaptation requires deciding not only how to update a model, but whether to commit an update without exceeding a declared budget for changes to protected behavior. We introduce Certified Anchor-Based Adaptation (CABA), a candidate-agnostic post-update gate that reserves historical observations from candidate optimization, compares old and candidate policies on frozen representations, and selects the largest tested interpolation coefficient whose worst-anchor forward KL divergence remains within budget. The finite check requires neither action labels nor complete trajectories. We develop a theory-led certification framework that separates measured anchor retention from conditional off-anchor guarantees. Under explicit representation-coverage and Lipschitz assumptions, an accepted update satisfies a uniform behavioral-divergence bound on protected support. The analysis gives regularity bounds for affine softmax heads, probabilist
    
[^27]: 物联网网络中无线电环境地图自适应无人机采样的INLA方法：先验宽度主导精度与校准

    INLA for Adaptive UAV Sampling of Radio Environment Maps in IoT Networks: Prior Width Governs Accuracy and Calibration

    [https://arxiv.org/abs/2610.40723](https://arxiv.org/abs/2610.40723)

    该论文提出基于INLA更新Matérn GMRF后验的自适应无人机采样方法来构建无线电环境地图，证明后验信息采样显著优于随机与割草机式基线，且增大先验阴影衰落标准差可在4–16 dB范围内单调提升精度与校准。

    

    我们研究了从非高斯物联网观测中构建无线电环境地图（REM）的自适应无人机采样问题。集成嵌套拉普拉斯近似（INLA）用于更新Matérn高斯马尔可夫随机场（GMRF）后验，航点通过熵预筛选和基于可达集的Sherman-Morrison前瞻方法进行选择。我们证明似然诱导的偏正态（Skew-Normal）校正具有六次熵效应，这解释了其对航点选择影响有限的原因。在19个保留的射线追踪窗口中，随机采样和割草机式采样相比后验信息采样分别多产生18.0%和14.0%的中位数残余NMSE，即相对于这些基线分别实现了15.3%和12.3%的降低，而各信息式规则之间的差异仍未解决。将先验阴影衰落标准差从6.0 dB拓宽至10.025 dB可改善NMSE和可信区间覆盖率，在该语料库4–16 dB的扫描范围内呈单调改善且未观察到拐点。

    arXiv:2610.40723v1 Announce Type: cross  Abstract: We study adaptive UAV sampling for radio environment map (REM) construction from non-Gaussian IoT observations. Integrated nested Laplace approximation(INLA) updates a Mat\'ern GMRF posterior, with waypoints selected by entropy pre-screening and Sherman--Morrison lookahead over the reachable set. We show that the likelihood-induced Skew-Normal correction has a sextic entropy effect, explaining its limited influence on waypoint selection. Across 19 retained ray-traced windows,Random and lawnmower sampling cost a median 18.0\% and 14.0\% more residual NMSE than posterior-informed sampling, equivalently reductions of 15.3\% and 12.3\% when referred to those baselines, while differences among informative rules remain unresolved. Widening the prior shadowing standard deviation from 6.0 to 10.025\,dB improves NMSE and credible-interval coverage, with monotone improvement over a 4--16\,dB sweep on this corpus and no observed turning point, al
    
[^28]: 多任务学习的几何学：共享子空间估计的精确速率

    The Geometry of Multitask Learning: Sharp Rates for Shared Subspace Estimation

    [https://arxiv.org/abs/2610.40713](https://arxiv.org/abs/2610.40713)

    本文提出一个通用多任务引理，通过任务回归函数及其估计量的几何条件刻画共享子空间估计的充分条件，首次获得线性回归与分类的精确极小极大速率。

    

    在多任务学习中学习共享子空间是一个重要的统计问题，然而在文献中考虑的许多形式化设置下，其精确速率仍未被充分理解。为填补这一源于现有上界技术局限性的空白，我们提出了一个通用的多任务引理。该引理以任务回归函数及其估计量的几何条件的形式，为一般性多任务设置中的快速子空间估计提供了充分条件。随后我们证明，该多任务引理在所考虑的情形下为线性回归和分类给出了此前未知的精确极小极大速率，并为加性噪声回归给出了在对数因子意义下相匹配的速率。

    arXiv:2610.40713v1 Announce Type: cross  Abstract: Learning a shared subspace in multitask learning is an important statistical problem, yet exact rates remain poorly understood in many of the formal settings considered in the literature. To address this gap, which arises from limitations of existing upper-bound techniques, we propose a general multitask lemma. The lemma provides sufficient conditions for fast subspace estimation in generic multitask settings, in the form of geometric conditions on the task regression functions and their estimators. We subsequently show that the multitask lemma yields sharp minimax rates---previously unknown---for linear regression and classification in the regimes considered, and rates matching up to logarithmic factors for additive-noise regression.
    
[^29]: CausalGrade：基于Jev的证据引导因果图评分方法

    CausalGrade: Evidence-Guided Causal Graph Scoring with Jev

    [https://arxiv.org/abs/2610.40692](https://arxiv.org/abs/2610.40692)

    提出基于概率判断模型Jev的因果图评分方法CausalGrade，通过融合观测数据证据与因果发现基础模型CDFM的边概率对候选因果图进行评分，评分与结构误差呈强负相关，可在真实因果结构未知时评估因果图质量。

    

    在真实因果结构未知的情况下，评估候选因果图的结构质量是应用因果发现结果时的一个重要问题。本文提出了CausalGrade，一种基于概率判断模型Jev的因果图评分方法。该方法将观测数据中提供的关联性、条件依赖性和方向性证据，以及由因果发现基础模型CDFM产生的边概率共同提供给Jev。Jev对直接因果边给出概率性判断，并通过衡量被评估图的邻接结构与这些边级判断之间的一致性，得到图层面的评分。在300个合成数据集上的实验表明，该评分在四种因果发现算法与五种变量数量的全部20种组合中均与结构误差呈现较强的负相关，说明分数越高通常对应着结构误差越小、因果图越准确。

    arXiv:2610.40692v1 Announce Type: cross  Abstract: Evaluating the structural quality of candidate causal graphs when the true causal structure is unknown is an important problem in applying causal discovery results. This paper proposes CausalGrade, a causal graph scoring method based on the probabilistic judgment model Jev. The method jointly provides Jev with evidence of association, conditional dependence, and directionality from observational data and edge probabilities produced by the causal discovery foundation model CDFM. Jev produces probabilistic judgments about direct causal edges, and a graph-level score is obtained by measuring the agreement between the adjacency structure of the graph under evaluation and these edge-level judgments. Across 300 synthetic datasets, the score exhibits a relatively strong negative correlation with structural error in all 20 combinations of four causal discovery algorithms and five variable counts, indicating that higher scores generally corresp
    
[^30]: 双重校正随机块模型（DC²SBM）：一种面向属性网络的广义结构检测方法

    Dual-Corrected Stochastic Block Model ($\text{DC}^2\text{SBM}$): A Generalized Structure Detection Method for Attributed Networks

    [https://arxiv.org/abs/2610.40676](https://arxiv.org/abs/2610.40676)

    提出双重校正随机块模型DC²SBM，通过在连边生成和属性生成通道中分别引入独立的拓扑度因子和属性丰度因子，实现节点活跃度与成员关系推断的解耦，从而有效检测属性网络中的广义结构。

    

    属性网络中的社区发现是复杂网络分析的核心任务之一。随机块模型（SBM）及其变体能够检测广义的网络结构，如同质结构、异质结构和混合结构。然而，现有的生成式方法要么忽略节点的异质性，要么用同一个因子来刻画节点在拓扑维度和属性维度上的活跃度，导致当这两种活跃度相互解耦时难以正确建模节点活跃度。本文提出了双重校正随机块模型（DC²SBM），该模型在连边生成通道和属性生成通道中分别引入独立的校正因子——拓扑度因子和属性丰度因子，从而有效地将节点活跃度与成员关系推断解耦，并进一步引入全局缩放参数以解决……

    arXiv:2610.40676v1 Announce Type: cross  Abstract: Community detection in attributed networks is one of the core tasks of complex network analysis. The stochastic block model (SBM) and its variants are able to detect generalized network structures such as assortative, disassortative and mixed structures. However, existing generative methods either ignore node heterogeneity, or characterize node activity in the topological and attribute dimensions by one and the same factor, making it difficult to correctly model node activity when the two kinds of activity are decoupled. In this paper, we propose the Dual-Corrected Stochastic Block Model ($\text{DC}^2\text{SBM}$), which introduces independent correction factors---a topological degree factor and an attribute abundance factor---into the link-generation and attribute-generation channels, respectively, thereby effectively decoupling node activity from membership inference, and which further introduces a global scaling parameter to resolve 
    
[^31]: 通过e-闭包重新认证决策：共形风险控制与随机化e-BH的一致改进

    Recertifying decisions via e-closure: uniform improvements of conformal risk control and randomized e-BH

    [https://arxiv.org/abs/2610.35821](https://arxiv.org/abs/2610.35821)

    提出“重新认证”这一通用范式，借助e-闭包原理利用风险控制程序有效性证书中的可计算松弛，统一改进了共形风险控制和随机化e-BH方法，构造出更少保守的决策。

    

    统计程序通常在规定水平上控制期望损失，同时留下部分风险预算未被利用。基于风险受控决策的e值表示以及多重检验中的e-闭包原则，我们提出了“重新认证”这一通用范式，通过利用有效性证书中可计算的松弛来改进统计程序。一个有效程序的损失按其风险预算归一化后，构成一个以状态为索引的e值集合；反之，归一化损失受该集合约束的决策将继承其风险保证。重新认证利用分布结构来量化松弛，在保持有效性的前提下扩大认证e值，从而构造出保守性更低的决策。我们开发了两个应用：对于已知损失族的共形风险控制（CRC），重新认证在每个假设增广样本的留出点之间重新分配未使用的风险预算……（原文摘要至此被截断）

    arXiv:2610.35821v1 Announce Type: cross  Abstract: Statistical procedures often control expected loss at a prescribed level while leaving part of their risk budget unused. Building on the e-value representation of risk-controlled decisions and the e-closure principle for multiple testing, we formulate recertification, a general paradigm for improving procedures by exploiting computable slack in their validity certificates. A valid procedure's losses normalized by their risk budgets form a state-indexed e-value collection; conversely, decisions whose normalized losses are bounded by this collection inherit its risk guarantees. Recertification uses distributional structure to quantify slack, enlarge the certifying e-values while preserving validity, and construct less conservative decisions.   We develop two applications. For conformal risk control (CRC) with a known loss family, recertification redistributes unused risk budget among the held-out points of each hypothetical augmented sam
    
[^32]: 伯努利老虎机中β-EB-TCI惩罚挑战者的精确非渐近分析

    Sharp Non-Asymptotic Analysis of the Penalized Challenger in $\beta$-EB-TCI for Bernoulli Bandits

    [https://arxiv.org/abs/2610.01951](https://arxiv.org/abs/2610.01951)

    本文首次对β-EB-TCI算法的惩罚挑战者给出精确非渐近分析，证明经验领导者成为真实最优臂后停止时间达到T_β*(μ)log(1/δ)（直至低阶项），从而为所有唯一最优臂的伯努利实例建立了非渐近高概率界。

    

    Top-two算法是固定置信度最优臂识别中简单而有效的方法，但其精确的非渐近行为仍不被充分理解。我们通过β-EB-TCI——Jourdan等人提出的经验最优top-two规则——来研究伯努利老虎机的这一问题，该规则使用带对数计数惩罚的伯努利传输代价来选择挑战者。我们证明，在经验领导者成为真实最优臂且其采样比例保持在接近β之后，停止时间在低阶集中项范围内为T_β*(μ)log(1/δ)。我们还证明，在此区域内，每个挑战者都会被线性次数地采样。因此，对于不含强制探索的原始算法，剩下的主要难点是控制经验领导者何时永久性地变为正确。这些结果为所有具有唯一最优臂的伯努利实例给出了非渐近高概率界。

    arXiv:2610.01951v1 Announce Type: cross  Abstract: Top-two algorithms are simple and effective for fixed-confidence best-arm identification, but their sharp non-asymptotic behavior is still not well understood. We study this problem for Bernoulli bandits through $\beta$-EB-TCI, the empirical-best top-two rule of Jourdan et al., whose challenger is chosen using a Bernoulli transportation cost with a logarithmic count penalty. We prove that, after the empirical leader has become the true best arm and its sampling fraction stays close to $\beta$, the stopping time is $T_{\beta}^{\star}(\mu)\log(1/\delta)$ up to lower-order concentration terms. We also show that, in this regime, every challenger is sampled linearly often. Thus, for the original algorithm without forced exploration, the main remaining difficulty is to control when the empirical leader becomes permanently correct. These results imply a non-asymptotic high-probability bound for all Bernoulli instances with a unique best arm. 
    
[^33]: 关于网络上大规模多重检验的非渐近方法

    On Large-Scale Multiple Testing Over Networks: A Non-Asymptotic Approach

    [https://arxiv.org/abs/2609.14170](https://arxiv.org/abs/2609.14170)

    该论文发现分布式多重检验中有限样本FDR失控源于赢者诅咒偏差，并提出交叉拟合贪心聚合算法（CFGA），通过数据分割实现了网络上分布式多重检验在有限样本下的严格FDR控制。

    

    分布式多重检验要求网络中的N个站点在严格的通信预算下控制全局错误发现率（FDR）。Pournaderi和Xiang（2024）提出的贪心区间聚合算法渐近地解决了这一问题，但在有限样本下可能违反FDR≤α的约束。我们将这一违反追溯到被选密度统计量中的“赢者诅咒”偏差，在标准带宽ε≍m⁻¹/²下其精确阶为Θ(m⁻¹/⁴√(log m))，其中m是网络中p值的总数。交叉拟合贪心聚合算法（CFGA）通过在每个节点数据的一半上选择嵌套拒绝族，并在另一半上进行评分，从而消除这一偏差，在每节点零假设比例已知的情况下实现有限样本FDR≤α；一个膨胀变体在η=1/m的可忽略松弛下覆盖了插入法（plug-in）设定。BONuS-GA则通过计数型knockoff进行校准，掩盖一组合成的均匀零假设，使得每个……（原文摘要在此处截断）

    arXiv:2609.14170v1 Announce Type: cross  Abstract: Distributed multiple testing asks $N$ sites to control a global false discovery rate (FDR) under a tight communication budget. The greedy interval-aggregation algorithm of Pournaderi and Xiang (2024) solves this asymptotically but can violate $\mathrm{FDR}\le\alpha$ at finite samples. We trace the violation to a winner's-curse bias in the selected density statistics, of exact order $\Theta(m^{-1/4}\sqrt{\log m})$ at the standard bandwidth $\varepsilon\asymp m^{-1/2}$, with $m$ the total number of p-values in the network. Cross-Fit Greedy Aggregation (CFGA) eliminates the curse by selecting the nested rejection family on one half of each node's data and scoring it on the other, achieving finite-sample $\mathrm{FDR}\le\alpha$ when per-node null rates are known; an inflated variant covers the plug-in setting at a vanishing $\eta=1/m$ slack. BONuS-GA instead masks a bag of synthetic uniform nulls calibrated by counting knockoffs, so every 
    
[^34]: 面向具有非自适应对手的多臂老虎机的最优切换遗憾

    Toward Optimal Switching Regret for Multi-Armed Bandits with Oblivious Adversary

    [https://arxiv.org/abs/2609.13547](https://arxiv.org/abs/2609.13547)

    本文提出单一算法，在非自适应对手下对任意未知的切换次数 $S$ 均能达到最优的 $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ 期望切换遗憾，解决了 Auer 等人提出的开放问题。

    

    我们研究对抗性多臂老虎机中的切换遗憾问题，其中学习者与一个至多变化 $S$ 次的动作序列竞争。当 $S$ 已知时，可以获得 $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ 的最优期望遗憾 [Auer et al., 2002]。然而，当 $S$ 未知时，Marinov 和 Zimmert [2021] 证明在自适应对手下这一保证是不可能实现的。在本文中，我们证明单个算法在非自适应对手下对每一个 $S$ 都能达到 $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ 的期望遗憾，从而解决了 Auer 等人 [2019b] 提出的一个开放问题。我们的算法结合了以小学习率初始化的固定分享学习器，以及利用随机化学习率和隐式探索来搜索局部改进的二分区间子程序。重要的是，非均匀先验偏向于跟随主学习器，使得维护众多子程序的代价保持在较小水平。当子程序……

    arXiv:2609.13547v1 Announce Type: new  Abstract: We study switching regret in adversarial multi-armed bandits, where the learner competes with an arm sequence that changes at most $S$ times. When $S$ is known, an optimal expected regret of $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ is obtainable [Auer et al., 2002]. However, when $S$ is unknown, Marinov and Zimmert [2021] show that this guarantee is impossible under an adaptive adversary. In this paper, we show that a single algorithm achieves $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ expected regret for every $S$ against an oblivious adversary, resolving an open problem of Auer et al. [2019b]. Our algorithm combines a fixed-share learner initialized with a small learning rate and dyadic-interval subroutines that search for local improvements using randomized learning rates and implicit exploration. Importantly, a non-uniform prior favors following the main learner, keeping the cost of maintaining many subroutines small. When the subrou
    
[^35]: 专家问题与在线凸优化中的极小极大交替遗憾

    Minimax Alternating Regret for the Experts Problem and Online Convex Optimization

    [https://arxiv.org/abs/2608.25182](https://arxiv.org/abs/2608.25182)

    本文首次解决了专家问题和一般在线凸优化中交替遗憾的极小极大率，证明$d$-专家问题的交替遗憾仅为$\Theta(\log d)$且与时间$T$无关，大幅改进了先前结果。

    

    本文研究在线凸优化（OCO）中的交替遗憾，受两人博弈中交替学习动态成功的启发。尽管先前工作表明在损失函数和可行域的各种假设下可实现$o(\sqrt{T})$的交替遗憾，但即使对于专家问题，极小极大遗憾率仍未解决。本文通过展示专家问题和一般OCO的匹配下界与上界来解决此问题。令人有些惊讶的是，对于$d$-专家问题，我们证明极小极大交替遗憾为$\Theta(\log d)$，与时间范围$T$无关。这显著优于Hait等人[2025]建立的最佳已知$\mathcal{O}(T^{1/3}\log^{2/3} d)$。我们进一步将结果扩展到$d$维紧凸集上的一般OCO，并证明最坏情况下的极小极大交替遗憾为$\Theta\left(d\log \l\right)$（注：原文截断，完整形式为$\Theta(d\log d)$）。

    arXiv:2608.25182v1 Announce Type: cross  Abstract: In this paper, we study alternating regret in online convex optimization (OCO), motivated by the success of alternating learning dynamics in two-player games. Although previous works have shown that $o(\sqrt{T})$ alternating regret is achievable under various assumptions on the loss functions and feasible domains, the minimax regret rate has remained open even for the expert problem. In this paper, we resolve this question by showing matching lower and upper bounds for both the expert problem and general OCO. Somewhat surprisingly, for the $d$-expert problem, we show that the minimax alternating regret is $\Theta(\log d)$, independent of the horizon $T$. This significantly improves upon the best-known $\mathcal{O}(T^{1/3}\log^{2/3} d)$ established by Hait et al. [2025]. We further extend our results to general OCO over a $d$-dimensional compact convex set and prove that the worst-case minimax alternating regret is $\Theta\left(d\log \l
    
[^36]: 弛豫公平回归的几何学：感知与无感知设置的统一框架

    Geometry of Relaxed Fair Regression: A Unified Framework for Aware and Unaware Settings

    [https://arxiv.org/abs/2605.28233](https://arxiv.org/abs/2605.28233)

    该论文将感知与无感知设置下的弛豫公平回归统一表述为最优传输问题，刻画了最优预测函数，并揭示Wasserstein惩罚体现全体人群的平滑折中而全变差惩罚强制子集均等这两种根本不同的公平哲学。

    

    公平性与准确性之间的权衡是公平感知机器学习方法部署中的核心问题。当敏感属性在推理阶段不可用时——即所谓的“无感知”设置——在松弛公平约束下获得准确预测的原则性方法在很大程度上是缺失的。在本工作中，我们通过将在人口均等惩罚下的回归问题表述为一个最优传输问题来填补这一空白。我们的框架统一了感知与无感知两种设置，并在平方Wasserstein-2惩罚与全变差惩罚下，借助最优传输映射刻画了最优预测函数。这些结果揭示了惩罚项的选择反映了根本不同的公平哲学：Wasserstein惩罚带来一种平滑的、面向全体人群的折中，而全变差惩罚则对个体的某个子集强制实现均等。基于这些理论刻画，我们提出……（摘要原文在此处被截断）

    arXiv:2605.28233v2 Announce Type: replace-cross  Abstract: Fairness-accuracy trade-offs are a central concern in the deployment of fairness-aware machine learning methods. When sensitive attributes are unavailable at inference time-the so called unawareness setting, principled methods for obtaining accurate predictions under relaxed fairness constraints are largely missing. In this work, we address this gap by formulating regression under a demographic parity penalty as an optimal transport problem. Our framework, unifies both the aware and unaware settings and characterizes optimal prediction functions via optimal transport maps, under both squared Wasserstein-2 and Total Variation penalties. These results reveal that the choice of penalty reflects fundamentally different fairness philosophies: the Wasserstein penalty induces a smooth, population-wide compromise, while Total Variation enforces parity for a subset of individuals. Building on these theoretical characterizations, we prop
    
[^37]: 多项逻辑特模型的最优设计及其在最佳品类识别中的应用

    Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification

    [https://arxiv.org/abs/2605.25592](https://arxiv.org/abs/2605.25592)

    我们提出了一种计算高效的多项逻辑特模型最优实验设计框架，通过混合整数线性规划和多项式时间替代目标实现近G-最优性，并应用于最佳品类识别。

    

    我们研究了多项逻辑特（MNL）赌博机的最优实验设计问题，其中智能体从大小为N的基础集合中反复选择K个物品的子集，并观察单一选择反馈。与线性或广义线性赌博机不同，MNL赌博机具有组合动作空间，这使得经典的最优设计方法和对所有子集的朴素优化在计算上不可行。我们提出了一种计算高效的多项逻辑特模型最优设计框架，通过两种互补方法实现统计效率和可扩展性：（i）将设计预言机精确或认证近似重构为0-1混合整数线性规划（MILP），并带有求解器认证的提前停止；（ii）一种完全多项式时间的提升设计，用可处理的替代目标替换非线性目标。利用基弗-沃尔福威茨等价定理，我们建立了近G-最优性保证。

    arXiv:2605.25592v2 Announce Type: replace-cross  Abstract: We study optimal experimental design for multinomial logit (MNL) bandits, where an agent repeatedly selects a subset of $K$ items from a ground set of size $N$ and observes single-choice feedback. Unlike linear or generalized linear bandits, MNL bandits have a combinatorial action space, which makes classical optimal design approaches and naive optimization over all subsets computationally intractable. We propose a computationally efficient optimal design framework for MNL models that achieves both statistical efficiency and scalability through two complementary approaches: (i) an exact or certified-approximate reformulation of the design oracle as a $0$-$1$ mixed-integer linear program (MILP) with solver-certified early stopping, and (ii) a fully polynomial-time lifted design that replaces the nonlinear objective with a tractable surrogate. Using the Kiefer-Wolfowitz equivalence theorem, we establish near G-optimality guarante
    
[^38]: 基于半非参数密度的非高斯不确定性估计

    Non-Gaussian Uncertainty Estimation via Semi-nonparametric Density

    [https://arxiv.org/abs/2604.09375](https://arxiv.org/abs/2604.09375)

    本文提出了基于半非参数（SNP）密度的非高斯不确定性估计框架，利用平方Hermite多项式保证密度非负性，推导了解析的边际分布与累积分布函数并给出无损凸的系数估计方法，除多项式阶数外无需额外超参数即可直接应用于非线性动力学系统的不确定性传播。

    

    准确且高效地表示非高斯状态分布对于估计、控制和决策至关重要，尤其是当系统具有高度非线性时。本文开发了一个通过半非参数（SNP）密度（即Gallant-Nychka密度）估计非高斯不确定性的数学框架。SNP密度使用平方概率Hermite多项式来保证概率密度函数的非负性，同时能够刻画非高斯行为。在不确定性传播方面，一旦对传播后的状态样本完成拟合，SNP密度便可直接进行解析的密度和概率评估，且除多项式阶数外无需调节其他超参数。为了将其应用于非线性动力学系统，我们推导了解析的边际分布函数和累积分布函数，并在给定符号分支上提出了系数估计问题的无损凸公式化方法。

    arXiv:2604.09375v2 Announce Type: replace-cross  Abstract: Accurate and efficient representation of non-Gaussian state distributions is critical for estimation, control, and decision-making, especially when the system is highly nonlinear. This paper develops a mathematical framework for estimating non-Gaussian uncertainty via semi-nonparametric (SNP), or Gallant-Nychka, densities. SNP densities use squared probabilists' Hermite polynomials to guarantee nonnegative probability densities while representing non-Gaussian behavior. For uncertainty propagation, once fitted to propagated state samples, SNP densities enable direct analytical density and probability evaluations without needing to tune additional hyperparameters besides the polynomial order. To enable their application to nonlinear dynamical systems, we derive analytic marginal and cumulative distribution functions and develop a lossless convex formulation of the coefficient estimation problem along a given sign branch. The appr
    
[^39]: 自适应保形预测的协同优化

    Co-optimization for Adaptive Conformal Prediction

    [https://arxiv.org/abs/2603.01719](https://arxiv.org/abs/2603.01719)

    提出CoCP方法，通过中心与半径的交替协同优化构建自适应保形预测区间，修正居中偏差，在保证有限样本边际有效性的同时实现更窄的区间、渐近条件覆盖和最优区间长度。

    

    在回归问题中，由于固定且非自适应的区间居中方式，保形预测在异方差和偏斜分布下常常效率低下。我们提出了CoCP，这是一种新方法，它通过中心 $m(x)$ 和半径 $h(x)$ 对预测区间进行参数化，并在通过折叠残差上的平滑分位数回归学习 $h(x)$ 与使用平滑区间损失优化 $m(x)$ 之间交替进行。该方法修正了区间居中偏差，并将区间推向高密度区域。除了通过分裂保形校准获得有限样本边际有效性之外，我们还证明了当基础估计量一致时，CoCP 还能实现渐近条件覆盖和最优区间长度。实验表明，CoCP 产生更窄的预测区间，并提升了经验条件覆盖的可靠性。代码可在 https://github.com/jjoeysu/cocp 获取。

    arXiv:2603.01719v2 Announce Type: replace-cross  Abstract: In regression, conformal prediction often suffers from inefficiency under heteroscedasticity and skewness due to fixed, non-adaptive interval centering. We propose CoCP, a new method that parameterizes prediction intervals by a center $m(x)$ and a radius $h(x)$, and alternates between learning $h(x)$ via smooth quantile regression on folded residuals and refining $m(x)$ using a smooth interval loss. This corrects mis-centering and drives the interval toward high-density regions. Beyond finite-sample marginal validity via split-conformal calibration, we prove that CoCP also attains asymptotic conditional coverage and optimal interval length if the base estimators are consistent. Experiments demonstrate that CoCP yields tighter intervals and achieves improved empirical conditional-coverage reliability. Code is available at https://github.com/jjoeysu/cocp.
    
[^40]: 宽度-深度缩放下μP的谱条件

    Spectral Condition for $\mu$P under Width-Depth Scaling

    [https://arxiv.org/abs/2603.00541](https://arxiv.org/abs/2603.00541)

    本工作提出了一个简单统一的谱框架来刻画宽度-深度联合缩放下的μP，揭示了残差块中变换数量从k=1到k≥2的基本转变，从而统一了先前分散的μP理论并确立了CompleteP式参数化的地位。

    

    生成式基础模型在宽度和深度两个维度上不断扩展，这为稳定特征学习以及跨模型规模的可靠超参数迁移带来了重大挑战。虽然最大更新参数化（μP）已为宽度缩放下的这两个问题提供了有原则的解决方案，但现有针对宽度-深度联合缩放机制的扩展仍然碎片化、依赖于特定架构和优化器，且往往依赖技术上复杂的理论。在本工作中，我们为宽度-深度联合缩放下的μP开发了一个简单而统一的谱框架。对于残差块包含k个变换的深度残差网络，该框架规定了权重范数及其每步更新应如何随宽度和深度进行缩放。它揭示了从k=1到k≥2之间的基本转变，统一了先前各不相同、彼此割裂的μP表述，并确定了CompleteP风格的参数化（……

    arXiv:2603.00541v3 Announce Type: replace  Abstract: Generative foundation models are increasingly scaled in both width and depth, posing significant challenges for stable feature learning and reliable hyperparameter (HP) transfer across model sizes. While maximal update parameterization~($\mu$P) has provided a principled solution to both problems for width scaling, existing extensions to the joint width-depth scaling regime remain fragmented, architecture- and optimizer-specific, and often rely on technically involved theories. In this work, we develop a simple and unified spectral framework for $\mu$P under joint width-depth scaling. For deep residual networks whose residual blocks contain $k$ transformations, the framework specifies how the norms of weights and their per-step updates should scale with width and depth. It reveals a fundamental transition from $k=1$ to $k\geq 2$, {unifying previously disparate $\mu$P formulations and identifying the CompleteP-style parameterization ($
    
[^41]: 基于数据驱动尾部能量准则的交叉拟合截断协方差估计

    Cross-Fitted Clipped Covariance Estimation with a Data-Driven Tail-Energy Criterion

    [https://arxiv.org/abs/2602.14020](https://arxiv.org/abs/2602.14020)

    该论文提出了完全数据驱动的分位数尾部能量替代准则（QTES），通过结合交叉拟合方差证书与截断所去除尾部能量的保留估计来自动选择最优截断水平，无需分布先验参数，并在有限样本下提供相对于最优候选估计器的理论保证。

    

    重尾数据使得协方差估计对截断水平非常敏感：更强的截断会降低方差但会增加偏差。我们研究如何在径向截断协方差族中从数据出发选择这一截断水平。我们提出了分位数尾部能量替代准则（QTES），这是一种完全数据驱动的规则，它将交叉拟合的方差证书与对截断所去除尾部能量的保留样本估计相结合。QTES不需要任何分布先验参数。对于欧几里得截断，算子范数偏差被这一标量尾部能量量所界定。在有限 $L_4$ 矩条件以及QTES块构造的温和可行性条件下，我们证明了正截断网格上的一致有限样本校准界，以及QTES所选择的估计器相对于同一方差加偏差准则下该网格上最优候选者的有限样本保证。实验在……

    arXiv:2602.14020v2 Announce Type: replace-cross  Abstract: Heavy-tailed data make covariance estimation sensitive to the clipping level: stronger clipping reduces variance but increases bias. We study how to choose this clipping level from the data within a radial clipped covariance family. We propose the quantile tail-energy surrogate (QTES), a fully data-driven rule that combines a cross-fitted variance certificate with a held-out estimate of the tail energy removed by clipping. QTES requires no distributional prior parameters. For Euclidean clipping, the operator-norm bias is bounded by this scalar tail-energy quantity. Under a finite $L_4$ moment condition, together with a mild feasibility condition for the QTES block construction, we prove a uniform finite-sample calibration bound over a positive clipping grid and a finite-sample guarantee for the estimator selected by QTES, relative to the best candidate on that grid under the same variance-plus-bias criterion. Experiments on cle
    
[^42]: 插值学习系统中泛化的函数空间稳定性边界

    A Function-Space Stability Boundary for Generalization in Interpolating Learning Systems

    [https://arxiv.org/abs/2602.03514](https://arxiv.org/abs/2602.03514)

    该论文提出函数空间稳定性证书来刻画算法稳定性何时能解释插值学习系统的泛化，并证明存在稳定性无法解释的低风险插值区域，从而界定了稳定性解释的适用边界。

    

    现代学习系统通常能够对训练数据进行插值，同时仍保持良好的泛化能力，然而算法稳定性何时能解释这一现象仍不清楚。我们将训练过程建模为函数空间中的一条轨迹，并测量沿该轨迹对单样本扰动的敏感性。我们提出了一个收缩传播条件，并通过展开所得递归得到一个稳定性证书。较小的证书意味着基于稳定性的泛化，同时我们还证明了存在风险较低的插值情形，其中这种收缩敏感性不可能成立，这表明稳定性并非一种普适的解释。实验证实，证书的增长能够预测不同优化器、步长和数据集扰动下的泛化差异。因此，该框架识别出了稳定性可以解释泛化的情形，以及必须借助其他机制来解释成功的情形。

    arXiv:2602.03514v3 Announce Type: replace  Abstract: Modern learning systems often interpolate training data while still generalizing well, yet it remains unclear when algorithmic stability explains this behavior. We model training as a function-space trajectory and measure sensitivity to single-sample perturbations along this trajectory. We propose a contractive propagation condition and a stability certificate obtained by unrolling the resulting recursion. A small certificate implies stability-based generalization, while we also prove that there exist interpolating regimes with small risk where such contractive sensitivity cannot hold, showing that stability is not a universal explanation. Experiments confirm that certificate growth predicts generalization differences across optimizers, step sizes, and dataset perturbations. The framework therefore identifies regimes where stability explains generalization and where alternative mechanisms must account for success.
    
[^43]: 变分分解自编码提升潜在表示的解耦性

    Variational decomposition autoencoding improves disentanglement of latent representations

    [https://arxiv.org/abs/2601.06844](https://arxiv.org/abs/2601.06844)

    该论文提出变分分解自编码框架VDA，通过仅在编码器端的DecVAE网络在分布层面进行潜在自编码，将信号分解的结构偏置融入变分表示学习，从而更好地解耦复杂非平稳信号的潜在表示。

    

    理解复杂、非平稳、高维时变信号的结构是科学数据分析中的一个核心挑战。在语音和生物医学信号处理等许多领域中，学习解耦且可解释的表示对于揭示潜在的生成机制至关重要。传统的无监督表示学习方法，包括变分自编码器（VAE），往往难以捕捉此类数据中固有的时间与频谱多样性。在此，我们提出变分分解自编码（VDA），这是一个在变分表示学习中引入强烈结构偏置、偏向信号分解的框架。VDA通过变分分解自编码器（DecVAE）实现，即仅在编码器端的神经网络，在分布层面执行潜在自编码，以学习多个与时间（原文此处截断）对齐的潜在子空间。

    arXiv:2601.06844v2 Announce Type: replace  Abstract: Understanding the structure of complex, nonstationary, high-dimensional time-evolving signals is a central challenge in scientific data analysis. In many domains, such as speech and biomedical signal processing, the ability to learn disentangled and interpretable representations is critical for uncovering latent generative mechanisms. Traditional approaches to unsupervised representation learning, including variational autoencoders (VAEs), often struggle to capture the temporal and spectral diversity inherent in such data. Here we introduce variational decomposition autoencoding (VDA), a framework that incorporates a strong structural bias toward signal decomposition into variational representation learning. VDA is instantiated through variational decomposition autoencoders (DecVAEs), i.e., encoder-only neural networks that perform latent autoencoding at the level of distributions to learn multiple latent subspaces aligned with time-
    
[^44]: 谁会被重点关注？基于神经指数策略分配儿科病房中稀缺的升级处置资源

    Who Gets Watched? Allocating Scarce Escalation Resources on Pediatric Wards with Neural Index Policies

    [https://arxiv.org/abs/2510.22069](https://arxiv.org/abs/2510.22069)

    该论文提出MILOP方法，将病房患者升级处置决策建模为多动作不宁多臂老虎机问题，并通过决策聚焦学习训练神经网络生成指数策略，从而在资源有限时优化儿科病房中对病情恶化患者的升级资源分配。

    

    问题定义：医院使用不同的路径来识别和应对病房中病情恶化的患者。在亚特兰大儿童医疗中心（CHOA），这些路径包括“重点关注者”监测、重症监护外联团队评估以及转入重症监护室。然而，当多名患者同时需要关注时，在人员和容量有限的情况下，缺乏统一的流程来决定谁应接受哪种响应。我们研究如何协调这些行动，以支持更早的升级处置，同时避免不必要的监测。方法/结果：我们使用多动作不宁多臂老虎机模型对病房升级决策进行建模，并开发了一种计算求解方法——多动作指数学习与优化策略（MILOP）。MILOP采用决策聚焦学习方法训练神经网络，以生成高质量的指数策略。我们使用CHOA的历史数据对MILOP进行训练和评估，并将其与两种基线方法进行比较。

    arXiv:2510.22069v2 Announce Type: replace  Abstract: Problem definition: Hospitals use separate pathways to identify and respond to deteriorating ward patients. At Children's Healthcare of Atlanta (CHOA), these include Watcher surveillance, Critical Care Outreach Team assessment, and transfer to the intensive care unit. When several patients need attention simultaneously, however, no common process determines who should receive which response under limited staffing and capacity. We study how to coordinate these actions to support earlier escalation without unnecessary monitoring. Methodology/results: We model ward escalation decisions using a multi-action restless multi-armed bandit model and develop Multi-action Index Learning and Optimization Policy (MILOP) as a computational solution approach. MILOP trains a neural network to generate a high-quality index policy using a decision-focused learning approach. We train and evaluate MILOP using historical CHOA data and compare it with two
    
[^45]: 面向生成建模的临界阻尼高阶朗之万动力学

    Critically-Damped Higher-Order Langevin Dynamics for Generative Modeling

    [https://arxiv.org/abs/2506.21741](https://arxiv.org/abs/2506.21741)

    该论文提出将临界阻尼技术从CLD和TOLD++推广到任意阶数的高阶朗之万动力学，从而改进基于扩散模型的生成建模方法。

    

    去噪扩散概率模型（DDPM）代表了一类尚未被充分探索的全新生成式人工智能方法。这类方法使用以随机微分方程表示的朗之万动力学来描述两个过程：将数据转化为噪声的前向过程，以及将噪声转化为生成数据的反向过程。许多此类方法利用辅助变量，将数据表述为“位置”变量，而辅助变量则被称为“速度”、“加速度”等。从这个意义上说，可以对动力学进行“临界阻尼”处理。临界阻尼已被成功应用于临界阻尼朗之万动力学（CLD）和临界阻尼三阶朗之万动力学（TOLD++），但尚未被应用于任意阶数的动力学。所提出的方法对近期最先进的高阶朗之万动力学（HOLD）进行了推广。

    arXiv:2506.21741v3 Announce Type: replace-cross  Abstract: Denoising diffusion probabilistic models (DDPMs) represent an entirely new class of generative AI methods that have yet to be fully explored. They use Langevin dynamics, represented as stochastic differential equations, to describe a process that transforms data into noise, the forward process, and a process that transforms noise into generated data, the reverse process. Many of these methods utilize auxiliary variables that formulate the data as a ``position" variable, and the auxiliary variables are referred to as ``velocity", ``acceleration", etc. In this sense, it is possible to ``critically damp" the dynamics. Critical damping has been successfully introduced in Critically-Damped Langevin Dynamics (CLD) and Critically-Damped Third-Order Langevin Dynamics (TOLD++), but has not yet been applied to dynamics of arbitrary order. The proposed methodology generalizes Higher-Order Langevin Dynamics (HOLD), a recent state-of-the-ar
    
[^46]: 高阶网络中Ollivier–Ricci曲率的无传输下界：适用范围、紧致性与计算应用

    Transport-Free Lower Bounds for Ollivier--Ricci Curvature in Higher-Order Networks: Scope, Tightness, and Computational Applications

    [https://arxiv.org/abs/2405.13302](https://arxiv.org/abs/2405.13302)

    本文将Bauer–Jost–Liu曲率界推广至超图Ollivier–Ricci曲率，建立了无需最优传输计算的下界，刻画了其适用范围、紧致性与极值几何，并给出了高效的计算应用。

    

    我们研究Bauer–Jost–Liu曲率界如何应用于超图Ollivier–Ricci曲率（ORC），以及其精度如何依赖于局部质量和几何结构。我们给出了一个显式的加权图表示，说明图定理如何适用于超图情形；并通过一个传输构造将该界推广到单调成对超边聚合。我们分析了惰性条件下的端点抵消现象，量化了相对于早期混合界的精确改进幅度，并刻画了在固定端点和重叠统计量下的极值几何。进一步的结果量化了最坏情况下的松弛度、稳定性以及孤立超边上的精确性。一种容量受限的短绕行精化方法利用额外的几何信息，并随着查询预算的增加而单调改进。作为计算应用，阈值级联方法能够保留精确的传输决策。数值验证和公开超图实验检验了界的紧致性和对游走的依赖性。

    arXiv:2405.13302v2 Announce Type: replace-cross  Abstract: We study how the Bauer--Jost--Liu curvature bound applies to hypergraph Ollivier--Ricci curvature (ORC) and how its accuracy depends on local mass and geometry. An explicit weighted-graph representation shows how the graph theorem applies. A transport construction transfers the bounds to monotone pairwise hyperedge aggregations. We analyze endpoint cancellation under laziness, quantify the exact improvement over the earlier mixture bound, and characterize the extremal geometry at fixed endpoint and overlap statistics. Further results quantify worst-case slack, stability, and exactness on isolated hyperedges. A capacity-constrained short-detour refinement uses additional geometric information and improves monotonically as the query budget increases. As a computational application, a threshold cascade preserves exact transport decisions. Numerical verification and public-hypergraph experiments examine tightness, walk dependence, 
    
[^47]: 在自相关非线性时间序列数据集中发现同期与滞后因果关系

    Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets

    [https://arxiv.org/abs/2003.03685](https://arxiv.org/abs/2003.03685)

    提出了PCMCI+方法，通过优化条件集选择，显著提升了在强自相关非线性时间序列中发现同期与滞后因果关系的召回率和可靠性。

    

    本文介绍了一种新颖的基于条件独立性（CI）的方法，用于在因果充分性假设下从观测时间序列中进行线性和非线性、滞后和同期的因果发现。现有的基于条件独立性的方法（如PC算法）以及其他框架的常用方法，在面对时间序列中普遍存在的强自相关这一挑战时，存在召回率低和假阳性部分膨胀的问题。新方法PCMCI$^+$扩展了PCMCI [Runge et al., 2019b]，加入了同期因果链接的发现。PCMCI$^+$通过优化条件集的选择来提高条件独立性检验的可靠性，甚至能从自相关中获益。该方法与变量顺序无关，且在理想情形下具有一致性。大量的数值实验表明，与其他方法相比，PCMCI$^+$具有更高的邻接检测能力，尤其是更高的同期方向召回率。

    arXiv:2003.03685v3 Announce Type: replace-cross  Abstract: The paper introduces a novel conditional independence (CI) based method for linear and nonlinear, lagged and contemporaneous causal discovery from observational time series in the causally sufficient case. Existing CI-based methods such as the PC algorithm and also common methods from other frameworks suffer from low recall and partially inflated false positives for strong autocorrelation which is an ubiquitous challenge in time series. The novel method, PCMCI$^+$, extends PCMCI [Runge et al., 2019b] to include discovery of contemporaneous links. PCMCI$^+$ improves the reliability of CI tests by optimizing the choice of conditioning sets and even benefits from autocorrelation. The method is order-independent and consistent in the oracle case. A broad range of numerical experiments demonstrates that PCMCI$^+$ has higher adjacency detection power and especially more contemporaneous orientation recall compared to other methods whi
    

