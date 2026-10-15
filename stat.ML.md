# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Testing Conditional Distribution Equality Using Generative Models](https://arxiv.org/abs/2610.20447) | 本文提出一种基于生成模型的条件分布相等性检验方法，通过结合观测数据与合成样本以及样本分割与学生化技术，在高维条件变量情形下实现了精确的检验水平控制和更强的检验功效。 |
| [^2] | [Tracking a Small Set of Experts in Online Convex Optimization](https://arxiv.org/abs/2610.20416) | 本文证明了在线凸优化中“追踪一小组专家”问题存在本质的计算复杂性分离：信息论上可达的最优遗憾度 $\widetilde{O}(md+k)$ 在超图植入团猜想等复杂性假设下无法被任何多项式时间算法实现，高效算法只能达到 $\widetilde{O}(dk)$ 量级的遗憾度。 |
| [^3] | [When Does Momentum Help in Reinforcement Learning with Verifiable Rewards?](https://arxiv.org/abs/2610.20413) | 本文通过带偏差修正EMA动量的老虎机形式化分析RLVR，理论上证明精确梯度下增大动量系数β可严格加速优化，而采样估计梯度下需增大β以维持有效步长η/(1-β)来克服采样波动。 |
| [^4] | [Posterior Frequencies and the Cost of Not Fitting: A Correction to a Memorization Lower Bound](https://arxiv.org/abs/2610.20402) | 本文通过精确的三点反例证明 Feldman 记忆化下界中的频率系数不成立且无法用普遍正常数因子修复，并基于全样本后验概率给出了一个精确的替代系数。 |
| [^5] | [Fractional is Better: Learnable Derivative Orders in Neural Operator Learning](https://arxiv.org/abs/2610.20363) | 该论文提出在神经算子中引入与偏微分方程结构对齐的可学习导数特征，从理论上证明其可将逼近界从 $O(W^{-(s-m)/d})$ 改进为 $O(W^{-s/d})$，并出人意料地发现最优导数阶数是严格低于偏微分方程阶数 $m$ 的分数阶 $\beta^*$。 |
| [^6] | [Beyond Expert Rankings: Contrast Calibration for Mixed-Precision MoE Quantization](https://arxiv.org/abs/2610.20353) | 提出NatScale框架，通过模型特定的幂变换将基于梯度的专家权重校准到信息论对比度目标，并联合评估门控与上下投影配置的输出误差，从而在存储预算下实现更优的混合精度MoE量化位宽分配。 |
| [^7] | [SpectrumQuant: Spectrum Alignment for LLM Quantization](https://arxiv.org/abs/2610.20303) | 提出SpectrumQuant，一种基于协方差最大化的闭式正交旋转方法，通过频谱对齐构建量化友好的变换，从而提升大语言模型训练后低比特量化的效果。 |
| [^8] | [From Plato to Aristotle: Toward a General Theory of Representation Hypotheses](https://arxiv.org/abs/2610.20286) | 该论文提出格表征假设（LRH），将各种表征对齐准则统一到同一个格结构中，从而解释了柏拉图表征假设的收敛声明与其实际证据之间的差异，并提出亚里士多德表征假设将收敛限定于局部邻域。 |
| [^9] | [MAEM: Mitigating Recency Bias via a Replay-Free, Capacity-Controlled Dynamic Mixture of Experts for Class-Incremental Graph Learning](https://arxiv.org/abs/2610.20238) | 提出MAEM框架，通过为每个新会话训练并冻结专用专家、以无重放且容量受控的动态专家混合方式进行路由，从而在无需任务标识的情况下缓解类增量图学习中的近期偏差与灾难性遗忘。 |
| [^10] | [The minimal width of neural networks and immersions of real projective space](https://arxiv.org/abs/2610.20229) | 本文证明了当实射影空间 $\mathbb{RP}^{n-1}$ 可光滑浸入 $\mathbb{R}^n$ 时，神经网络逼近从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 连续函数的最小宽度至少为 $n+\min\{n,m\}$，从而在 $n=2,3,4,7,8$ 时确定了 ReLU 网络的最小宽度为 $n+m$，揭示了神经网络逼近理论与微分拓扑之间的深刻联系。 |
| [^11] | [Mixed Poisson diffusion models and application to posterior sampling](https://arxiv.org/abs/2610.20218) | 该论文提出了一种由多元计数过程驱动的混合泊松扩散模型（MPDM），为观测数据离散且服从泊松分布的泊松逆问题提供了后验采样的新框架。 |
| [^12] | [Optimizing Sparse Boolean Rule Networks using Mixed Integer Linear Programming](https://arxiv.org/abs/2610.20170) | 该论文提出用混合整数线性规划训练稀疏的布尔规则网络（DNF网络），通过增强的公式大幅减少整数决策变量并消除冗余约束，在保持精确布尔语义的同时实现误分类成本与稀疏性之间的全局最优权衡。 |
| [^13] | [A Large-Dimensional Analysis of Least-Squares Federated Learning via Generalized Total Variation Minimization](https://arxiv.org/abs/2610.20075) | 本文通过随机矩阵理论对基于广义全变差最小化的最小二乘联邦学习进行大维度分析，给出了独立同分布与非独立同分布高斯混合数据下渐近精确性能的精确解析表达式，量化了正则化如何在局部适应与协作之间进行权衡。 |
| [^14] | [Weighted Adjoint Matching: Finetuning Flow Matching Models with Time-Dependent Control Cost](https://arxiv.org/abs/2610.20035) | 提出加权伴随匹配方法，通过在控制代价上引入时间依赖权重，早期保持粗略语义特征的多样性与平衡、后期精细化细节以提升奖励，从而在微调流匹配模型时实现奖励与模式平衡之间更好的权衡。 |
| [^15] | [MCMC Can See Can't See](https://arxiv.org/abs/2610.20020) | 本文证明了任何有限的收敛诊断清单都无法保证MCMC真正收敛，并定量刻画了最坏情况下未被检测到的未收敛性如何随诊断方法增加而衰减、如何最优选择诊断方法，以及未被检测到的偏差如何限制有限诊断所能认证的有效样本量。 |
| [^16] | [Outperforming 1000 Euclidean Layers With Hyperbolic Goal-Conditioned RL](https://arxiv.org/abs/2610.20009) | 通过将对比强化学习中的欧几里得距离替换为双曲空间的测地距离，仅64层深的双曲智能体就能以超过12倍的训练速度和约13倍的参数量优势，超越1024层欧几里得网络的导航性能。 |
| [^17] | [Minimax estimation of the pseudo-spectral gap of continuous Markov chains, with applications to empirical concentration bounds](https://arxiv.org/abs/2610.20007) | 本文提出了紧致连续状态空间马尔可夫链伪谱隙的首个极小极大最优估计器，结合 Bernstein 不等式推导出经验集中不等式，并据此在无需伪谱隙先验知识的情况下，为马尔可夫奖励老虎机问题给出了具有接近独立同分布情形遗憾界的 UCB 算法。 |
| [^18] | [Convergence Rates of Gaussian Process Thompson Sampling for Efficient Global Optimization](https://arxiv.org/abs/2610.19925) | 本文首次在无噪声RKHS设定下为高斯过程汤普森采样建立了最坏情况简单遗憾上界 $O(N^{-\nu/d}\sqrt{\log N})$ 及匹配的下界，证明对数惩罚项是该算法的固有代价。 |
| [^19] | [Maximum likelihood, neural networks and posterior mean for inferring fractional Brownian motion parameters](https://arxiv.org/abs/2610.19888) | 该论文提出将极大似然估计与神经网络（经证明可逼近后验均值）相结合来联合估计分数布朗运动的Hurst指数和广义扩散系数，并表明后验均值结合轮廓似然估计在绝大多数情况下具有最低的均方根误差。 |
| [^20] | [Neural Wasserstein Splines: Representation-Invariant Second-Order Priors for Trajectory Inference](https://arxiv.org/abs/2610.19878) | 提出神经瓦瑟斯坦样条（NWSP），通过瓦瑟斯坦平滑先验和内在瓦瑟斯坦加速度的有限维替代量，实现表示不变的二阶正则化，从而从不成对的快照中学习连续的种群演化曲线。 |
| [^21] | [On the Optimisation Landscape of Overparameterised Weighted MMD](https://arxiv.org/abs/2610.19861) | 该论文首次为加权MMD优化提供了有限粒子的过参数化保证，证明对于一维高斯核，在粒子数量足够多时所有局部极小值都具有任意小的MMD，且所有虚假驻点均不稳定。 |
| [^22] | [Text-Informed Training Data Filtering for Enhanced Virtual Flow Meter Performance](https://arxiv.org/abs/2610.19860) | 本文提出一种利用大语言模型将非结构化文本信息（如工程师评论和操作日志）与表格化传感器数据相结合的数据过滤方法，通过上下文异常检测去除低质量数据，从而提升虚拟流量计的预测性能。 |
| [^23] | [Calibration and Risk-Budgeted Decisions with System One Models](https://arxiv.org/abs/2610.19811) | 该论文提出在LLM工作流的二元决策场景中，通过不确定时采取保守回退动作，并利用Learn Then Test（LTT）的有限样本统计保证来控制假遗漏风险，显著优于朴素阈值选择基线方法。 |
| [^24] | [Efficient contextual partial monitoring with general function approximation](https://arxiv.org/abs/2610.19804) | 本文提出了计算高效的上下文部分监控框架 E2D.PMSide，针对原始 E2D 算法探索规则难以计算的问题设计了两种高效的探索规则，在通用函数逼近下实现了最优遗憾阶。 |
| [^25] | [Asymptotically Exact Negative Guidance of Diffusion Models via Positive-Unlabeled Learning](https://arxiv.org/abs/2610.19802) | 提出一种从显式目标密度采样的扩散模型负引导新方法，利用正-未标记学习在线估计样本属于不良条件的概率，从而实现渐近精确、偏差可通过增加计算量系统降低的负引导。 |
| [^26] | [SNACK: A Sequential Notation Framework for Probabilistic Graph Generation](https://arxiv.org/abs/2610.19781) | SNACK提出了一种将图线性化为字符串序列的表示框架，通过序列与有序邻接矩阵的双射对应和无效logit屏蔽机制，使序列模型能够以100%有效率生成图，并在保持有竞争力样本质量的同时显著提升GFlowNet训练吞吐量。 |
| [^27] | [From Global to Tail Convergence: Dynamics of $\alpha$-Stable Score-Based Generative Models](https://arxiv.org/abs/2610.19773) | 本文对α-稳定基于分数的生成模型建立了非渐近收敛性分析，证明在全局Wasserstein距离下α-稳定动力学与布朗扩散模型表现相当，但两者在尾部收敛行为上存在本质区别。 |
| [^28] | [Annealed L\'{e}vy Diffusion via Gaussian Surrogate Reverse Dynamics](https://arxiv.org/abs/2610.19692) | 提出退火Lévy扩散（ALD），通过随时间退火的稳定性指数将重尾探索与高斯精炼结合，并把稳定噪声表示为高斯尺度混合，从而在标准Gaussian噪声预测训练中仅需增加一个标量尺度输入即可实现。 |
| [^29] | [Function-Space Temporal Self-Distillation for Policy Optimization](https://arxiv.org/abs/2610.19687) | 本文提出单策略的函数空间策略优化框架FSPO，以稀疏的历史下一词元分布作为固定时间目标进行自蒸馏，无需维护持久教师模型即可提供稳定密集的词元级监督，并通过特权分歧校正与可靠性自适应锚定提升策略优化效果。 |
| [^30] | [On Global Joint Lower Bounds in Zeroth-Order Nonconvex Optimization](https://arxiv.org/abs/2610.19652) | 本文在无限制全空间设置下为光滑随机非凸优化建立了标量预言机全局联合下界，回答了零阶方法复杂度关于维度、光滑化参数和精度的依赖是否联合最优这一公开问题。 |
| [^31] | [Risk-Controlled DFT Selection for Machine-Learned Interatomic Potentials](https://arxiv.org/abs/2610.19630) | 本文提出一种基于保形风险控制（CRC）的框架，利用轻量级预测器估计MLIP与DFT之间的不一致分数并校准为选择阈值，在保证MLIP误差可控的前提下自动决定哪些构型需要昂贵的DFT计算，从而突破自动化材料发现的通量瓶颈。 |
| [^32] | [Projected Wasserstein-Fisher-Rao Flows for Hawkes Background-Intensity Estimation](https://arxiv.org/abs/2610.19588) | 本文提出一种基于Wasserstein-Fisher-Rao几何的似然变分方法，通过向统计流形投影的分布框架来估计Hawkes过程的背景强度，并证明了相应估计量的条件一致性定理。 |
| [^33] | [On Inherent Privacy of Posterior Sampling: A Unified R\'enyi-Divergence Framework](https://arxiv.org/abs/2610.19573) | 本文提出统一的Rényi散度框架，证明从贝叶斯后验分布中采样（OPS）天然获得“免费”的差分隐私保证，其隐私损失由单条记录对数似然比的后验指数矩刻画，并在多种尾部条件与代表性贝叶斯模型下给出显式隐私保证。 |
| [^34] | [Noisy-Label Calibration with Multiple Unreliable Labels](https://arxiv.org/abs/2610.19543) | 该论文提出了噪声校正ECE（NC-ECE）估计器、噪声概率估计算法以及事后重校准方法，使得在仅有多个不可靠标注的情况下，也能对二元分类器的校准性能进行无偏评估和可靠重校准。 |
| [^35] | [High-Probability Regret Bounds for Feel-Good Thompson Sampling](https://arxiv.org/abs/2610.19513) | 本文首次为 Feel-Good Thompson 采样在上下文老虎机中建立了高概率遗憾界，在线性设置下达到了与最优算法在维度和时间范围依赖性上相匹配的遗憾保证。 |
| [^36] | [State-Matched Sequence Conformal Prediction](https://arxiv.org/abs/2610.19484) | 提出了状态匹配序列保形预测（SMSCP），通过构建局部线性状态并利用状态相似性选取匹配的历史校准分数参考集，解决了可比状态在非相邻时间重现时基于近期性的传统方法失效的问题。 |
| [^37] | [Large-Step Last-Iterate Convergence of Random-Reshuffling Stochastic Gradient Descent](https://arxiv.org/abs/2610.19444) | 本文为随机重排随机梯度下降（RR-SGD）的末次迭代点建立了大步长收敛保证，通过保留随机比较点的采样偏差，从步长限制中移除了显式的 1/n 因子，使末次迭代的收敛速率在对数因子内匹配平均迭代点的收敛速率。 |
| [^38] | [Multi-Sensitive Distributional Fairness: Joint Wasserstein Audits and Shapley Allocation](https://arxiv.org/abs/2610.19252) | 该论文提出用一维Wasserstein几何对多个敏感属性进行联合公平性审计，将联合差异刻画为全单元条件分位数过程的方差，并借助Shapley值实现与顺序无关的差异归因，从而捕捉单坐标审计会遗漏的交叉性歧视。 |
| [^39] | [Generalizing Identifiability Results in Causal Discovery from Complete Data to Missing Data Settings](https://arxiv.org/abs/2610.19211) | 该论文建立了在包括非随机缺失在内的多种缺失机制下因果图保持可辨识的一般性充分条件，提供了将完整数据因果发现算法的正确性保证扩展到缺失数据场景的框架，并针对加性噪声模型提出了两个渐近正确的算法。 |
| [^40] | [Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization](https://arxiv.org/abs/2610.19198) | 该论文揭示了语义嵌入的几何结构是LLM路由在未见任务上失效的根源，并提出用Qwen2.5-7B-Instruct深层最后token的隐藏状态替代语义嵌入作为表示，从而显著提升路由方法对未见任务的泛化能力。 |
| [^41] | [Structure-aware Bayesian full waveform inversion via diffusion-regularized Stein variational gradient descent](https://arxiv.org/abs/2610.19187) | 该论文提出将学习得到的扩散模型作为隐式正则化机制嵌入基于退火Stein变分梯度下降的贝叶斯全波形反演框架，以刻画复杂地质结构先验并增强反演的不确定性量化能力。 |
| [^42] | [Student Copula Markov Networks](https://arxiv.org/abs/2610.19135) | 本文基于copula密度刻画成对马尔可夫性质，提出学生t-Copula马尔可夫网络及相应的ECM算法，将概率图模型扩展至非高斯数据并实现参数估计与图结构选择。 |
| [^43] | [Neural Supervised Clustering for Learning Individualized Treatment Rules with Many Treatments](https://arxiv.org/abs/2610.19104) | 提出了一种结合共享非线性协变量表示、低维治疗嵌入与成对聚类融合的神经Q学习框架NSCF，能够从每个治疗臂有限的观测数据中学习多治疗方案下的个体化治疗规则，并生成可解释的治疗分组划分。 |
| [^44] | [Conformalized Physics-Informed Weak Regression for Parameter Estimation in Epidemic Models](https://arxiv.org/abs/2610.19058) | 该论文提出物理信息弱回归（PIWR）方法，通过在时间上积分模型而非对数据求导来估计传染病模型参数，并结合由物理模型自身生成校准集的分割保形预测，为所有参数提供具有有限样本覆盖率保证的无分布同步置信区间。 |
| [^45] | [What Information Buys in Active Inference](https://arxiv.org/abs/2610.19046) | 本文将主动推理（AIF）与贝叶斯UCBVI置于相同的MDP状态抽象下进行对比，发现AIF的信息增益衰减过快导致探索不足，进而通过对信息增益进行平方根校准，使校准后的AIF达到了√T阶的后悔界。 |
| [^46] | [Loss Curvature as a Sampling-Free Proxy for Memorization in Diffusion Models](https://arxiv.org/abs/2610.19042) | 该论文提出输入损失曲率（训练损失Hessian矩阵的Frobenius范数）可作为扩散模型记忆化的免采样代理指标，通过在训练中追踪其均值即可预测模型何时开始记忆并复现训练数据。 |
| [^47] | [Discrete Diffusion Models are Minimax Optimal Under Sparsity](https://arxiv.org/abs/2610.19020) | 该论文证明了在稀疏性假设下，基于正则化得分熵目标的离散扩散模型能够达到极小极大最优的统计速率，且所用正则化方法无需预知稀疏参数即可实现自适应。 |
| [^48] | [Limits of Statistical Assurance for Embodied and Agentic AI under Adaptive Adversaries](https://arxiv.org/abs/2610.18948) | 本文证明了对于具身智能和智能体AI，任何有限时长的安全认证在原理上都无法排除那些在认证前伪装正常行为、认证后才激活的隐蔽自适应对手，并精确刻画了此类攻击带来的最坏情形期望损失。 |
| [^49] | [Semi-Amortized Bayesian Inference with Reusable Local Posteriors](https://arxiv.org/abs/2610.18945) | 该论文提出一种半摊销贝叶斯推断方法，将参数分块并为计算昂贵或频繁更新的块训练可复用的神经估计器来近似局部条件后验，再通过马尔可夫链将这些神经采样与常规更新交替进行，从而高效地实现高维复杂模型的贝叶斯推断。 |
| [^50] | [Wong--Zakai Convergence and FNO Learning for Stochastic Volterra Equations](https://arxiv.org/abs/2610.18939) | 本文为随机Volterra积分方程建立了Wong–Zakai收敛定理，并在此基础上证明镜像填充傅里叶神经算子能够学习其解算子，实现对完整解轨迹的准确预测以及向更细时间网格的泛化。 |
| [^51] | [Delay Complexity Dimension: Learning with Delayed and Censored Labels](https://arxiv.org/abs/2610.18935) | 本文提出“延迟复杂度维度”（DCD），将VC维度与观测信道的标签信息率相结合，刻画了标签延迟且被删失情形下PAC学习的样本复杂度，证明任何学习器需要 Ω(DCD/ε²) 个样本，且伪标签ERM可在常数与对数因子内达到该界。 |
| [^52] | [Thompson Sampling for Learning in Zero-Sum Games](https://arxiv.org/abs/2610.18899) | 本文首次证明了在具有唯一纳什均衡的未知双人零和矩阵博弈中，汤普森采样可实现 $O(\sqrt T)$ 的对偶间隙遗憾（纯均衡情形下为 $O(\log T)$），并给出了与之匹配的极小极大下界和实例相关下界。 |
| [^53] | [Bayesian Causal Forests for Zero-Inflated Continuous Outcomes](https://arxiv.org/abs/2610.18889) | 本文提出ZIC-BCF-Smear方法，通过解耦的两部分贝叶斯因果森林（Probit BCF建模零值参与、高斯BCF建模正值强度），结合子群体倾向得分调整与Duan非参数平滑估计，有效解决了半连续结果下异质性处理效应估计的难题。 |
| [^54] | [Inverse Reasoning Models for Efficient Post-Training and Verification from Answer-Only Data](https://arxiv.org/abs/2610.18878) | 该论文提出一种自蒸馏方法，通过训练一个“逆向推理模型”根据答案反向生成推理轨迹，从而仅利用提示-答案对数据即可高效后训练推理大语言模型，在数据效率与计算效率之间取得平衡。 |
| [^55] | [Convergence-Rate-Guided Algorithm Design of Expected Improvement for Continuous Simulation Optimization](https://arxiv.org/abs/2610.18434) | 该论文通过推导简单遗憾的高概率上界，在收敛率分析的基础上系统性地设计了随机克里金期望改进算法的三个关键组件——当前最优解、采样重复次数和终端推荐，解决了未知异方差噪声下连续仿真优化中依赖启发式规则进行算法设计的局限。 |
| [^56] | [Strong Gaussian Approximation of Constant-Learning-Rate SGD: Bias and Trajectory Inference](https://arxiv.org/abs/2610.14567) | 本文为重尾噪声下的常学习率随机梯度下降建立了KMT型强高斯逼近，实现了有限样本下同时性的整个轨迹层面的推断，并蕴含泛函中心极限定理。 |
| [^57] | [Target-Constrained Pooling for Trajectory Extrapolation](https://arxiv.org/abs/2610.13034) | 提出目标约束池化（TCP）方法，利用已完成的相关轨迹从两种互补池化信号中估计轨迹终点并修正自由预测，在五个领域的25,245个目标上将平均RMSE降低35.8%至42.1%。 |
| [^58] | [Steering Diffusion Models to Rare Events with Sequential Monte Carlo](https://arxiv.org/abs/2610.08652) | 本文提出DireSMC，一种序列蒙特卡洛方法，通过引导加权样本群体趋向扩散模型中的稀有事件，不仅能生成稀有事件样本，还能给出其概率的校准估计，并可轻松扩展到各类用户自定义稀有事件。 |
| [^59] | [ProximalFM: Amortized Proximal Causal Inference under Hidden Confounding](https://arxiv.org/abs/2610.08078) | 该论文提出ProximalFM，利用先验数据拟合网络（PFN）以摊销式贝叶斯方法在隐藏混杂下进行近端因果推断，通过先验正则化缓解了非参数近端估计中病态积分方程的数据饥渴、超参数敏感和优化不稳定等问题。 |
| [^60] | [Vine Copula VAR:From Recursive Margins to Joint Forecast Inference](https://arxiv.org/abs/2610.07589) | 本文在稳定的藤Copula VAR框架下，推导出历史递归估计误差与终端预测的联合影响，并提出一个协方差估计量，可为固定单侧事件概率提供渐近有效的重复样本预测区间。 |
| [^61] | [Predictively Oriented Gaussian Process Posteriors](https://arxiv.org/abs/2610.03201) | 提出预测导向高斯过程（PrO-GPs），将预测不确定性作为主要推断目标，在模型错误设定下比标准高斯过程产生校准更好的预测分布。 |
| [^62] | [Amortized ratio-estimation importance sampling and localized simulation-based calibration for intractable likelihoods](https://arxiv.org/abs/2609.39712) | 该论文提出了针对难解似然模型的顺序化后验估计方法与局部化模拟校准技术，利用局部化条件密度代理和摊销化的似然-证据比率估计器，以低计算成本实现准确且经过校准的贝叶斯推断。 |
| [^63] | [The Statistical Benefits of Multiple Responses for Learning from Demonstrations](https://arxiv.org/abs/2609.33291) | 当示范者并非最优时，返回多个候选响应（pass@$k$）能带来更强的统计优势——将最坏情况下对目标精度的依赖从 $1/\varepsilon^2$ 改善为 $1/\varepsilon$，并将对奖励类大小的依赖从 $\log N$ 改善为 $\log N/\log k$。 |
| [^64] | [Relatively Smart II: Tractable or Semi-Supervised Instance-Optimal Learning](https://arxiv.org/abs/2609.10886) | 本文证明ERM乃至任何正规一致学习器在无分布二分类中都是相对智能的，从而解决了此前留下的开放问题，即自然的可处理学习器能否达到与单包含图学习器可比的保证。 |
| [^65] | [Q-Edge: Symmetry-Reduced Quantum Simulation of Structured Extreme Dependence](https://arxiv.org/abs/2609.03706) | Q-Edge提出了一种基于对称性的无损轨道空间表示方法，将30维问题中约160万个角度状态压缩为256个轨道状态，从而实现了结构化极端依赖系统的可扩展量子模拟与数字孪生。 |
| [^66] | [Comparing SGLD and a fixed-noise predictor-corrector adaptation in canonical joint energy-based models on CIFAR-10](https://arxiv.org/abs/2608.05025) | 该论文系统复现了经典联合能量模型（JEM），并首次检验了用固定噪声预测-校正采样器替代SGLD的效果，结果显示其分类性能接近经典方法但生成质量略有差距，同时记录了两种失败模式。 |
| [^67] | [The Zero Pattern of a Design Matrix Drives Multiple Descent in Over-parameterized Regression](https://arxiv.org/abs/2607.24041) | 本文放宽了过参数化线性回归中协变量独立且协方差矩阵非退化的假设，证明协方差退化与变量相关性会引发多重下降现象，并借助二部图的最大匹配和Dulmage-Mendelsohn分解精确定位了风险峰值出现的位置。 |
| [^68] | [Hopformer: Homogeneity-Pursuit Transformer for Time Series Forecasting](https://arxiv.org/abs/2607.22299) | Hopformer提出两阶段时间序列预测框架，先通过稀疏模式聚合（SPA）提取融合协变量的共同低方差趋势作为同质化层，再用LoRA微调的Transformer建模残差中的复杂依赖，在理论上证明了近最优偏差-方差权衡与泛化界，在多个基准上平均提升MASE 6.56%。 |
| [^69] | [Gauge-Invariant, Parameter-Insensitive Regularization for Potential Recovery from Flow on Directed Graphs](https://arxiv.org/abs/2607.13609) | 该论文提出用规范不变的图狄利克雷能量替代岭正则化，从有向图流数据中恢复势能，解决了岭正则化导致的排序反转问题，并实现了跨λ四个数量级的参数不敏感性。 |
| [^70] | [What Does a Discrete Diffusion Model Learn? One Reverse Process in Three Coordinates](https://arxiv.org/abs/2607.05381) | 本文证明离散扩散模型中的去噪器、桥接插件和具体分数三种方法学习的其实是同一对象——加噪过程的精确时间反演，只是处于三种不同坐标系，并给出 Oracle Distance 恒等式（负 ELBO 等于数据熵加路径 KL 散度）及三种坐标间的精确转换。 |
| [^71] | [Self-Organized Conformal Prediction: Reducing Regional Coverage Gaps with Unsupervised Group Discovery](https://arxiv.org/abs/2606.29403) | 提出自组织保形预测（SOCP），利用无需校准标签的无监督自组织映射（SOM）发现输入空间分组并从中提取校准缓冲，在预测器和非一致性分数保持不变的前提下实现精确的条件覆盖有效性，从而减少特征空间异质区域的覆盖差距。 |
| [^72] | [Robust inference for risk heterogeneity under group imbalance](https://arxiv.org/abs/2606.00797) | 提出一种基于Neyman正交性的稳健框架，用于在组别失衡情形下推断两个总体间的风险异质性，有效降低偏差并提升有限样本下的推断稳定性。 |
| [^73] | [Memory by Design: Probabilistic Sequence Layers](https://arxiv.org/abs/2605.31163) | 本文提出了一种设计模型框架，通过贝叶斯滤波和协方差传播统一多种次二次递归序列层，并恢复协方差传播以增强记忆保留和检索。 |
| [^74] | [Symmetry-Compatible Principle for Optimizer Design: Embeddings, LM Heads, SwiGLU MLPs, and MoE Routers](https://arxiv.org/abs/2605.18106) | 该论文提出优化器设计的对称兼容原则——梯度更新规则应在权重块的对称群下保持等变，据此统一了双正交等变更新方法，并为嵌入层、语言模型输出头、SwiGLU MLP和MoE路由器等结构推导出对称兼容的优化器。 |
| [^75] | [A proximal gradient algorithm for composite log-concave sampling](https://arxiv.org/abs/2605.12461) | 提出了一种针对复合对数凹分布的近端梯度采样算法，通过结合 $f$ 的梯度评估与 $g$ 的受限高斯预言机，在 $f+g$ 为 $\alpha$-凸且 $f$ 为 $\beta$-光滑的条件下，以 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$ 次预言机调用实现总变差距离下 $\varepsilon$ 精度的采样，匹配了 $g=0$ 情形下的最优结果。 |
| [^76] | [When Can Digital Personas Reliably Approximate Human Survey Findings?](https://arxiv.org/abs/2605.10659) | 基于LISS面板的研究表明，大语言模型驱动的数字人格能在分布层面较好地近似人类调查回答（尤其在涉及稳定属性和价值观的领域），但在个体预测和恢复受访者多变量结构方面仍存在根本局限。 |
| [^77] | [Sharp Statistical Thresholds for Attributed Gaussian Graph Matching](https://arxiv.org/abs/2603.23305) | 本文针对边权重与节点特征联合相关的属性化高斯图匹配问题，推导出了精确恢复、几乎精确恢复和部分恢复的精确信息论阈值，并发现节点特征信息的引入使得精确恢复阈值从共享阈值中分离，呈现出比标准图匹配更丰富的相变结构。 |
| [^78] | [Maximum-Entropy Exploration with Future State-Action Visitation Measures](https://arxiv.org/abs/2603.18965) | 该论文提出了一种与未来状态-动作特征折扣分布的熵成比例的内在奖励方法，理论上证明其期望总和是从初始状态出发的轨迹所访问分布熵的下界且可通过收缩算子的不动点性质进行离策略估计，从而实现了更优的最大熵探索。 |
| [^79] | [Is Flow Matching Just Trajectory Replay for Sequential Data?](https://arxiv.org/abs/2602.08318) | 本文推导了序列数据上流匹配目标的最优速度场，揭示其本质是一个非参数、记忆增强的连续时间动力系统，其速度场可表示为观测转移的瞬时速度的加权混合，为神经流匹配模型提供可解释视角并提出FreeFM。 |
| [^80] | [Beyond Sustainable Development Goal (SDG) Accelerators: Directed Evidence for Adaptive Sustainability Portfolios](https://arxiv.org/abs/2601.20875) | 本研究基于114个国家2000至2024年的面板数据，结合面板格兰杰因果检验与局部投影方法，识别出84个定向SDG联系（40个协同、44个权衡），证明不存在普适的SDG加速器，并主张依据定向因果证据构建适应性的可持续发展政策组合。 |
| [^81] | [Sig-DEG for Distillation: Making Diffusion Models Faster and Lighter](https://arxiv.org/abs/2508.16939) | 本文提出Sig-DEG，利用部分签名与循环结构逼近随机微分方程的解来蒸馏预训练扩散模型，在保持有竞争力的样本质量的同时显著提升推理速度。 |
| [^82] | [GLL: A Differentiable Graph Learning Layer for Neural Networks](https://arxiv.org/abs/2412.08016) | 该论文通过伴随方法推导出反向传播方程，提出了可微图学习层GLL，实现了图学习算法与神经网络的端到端精确集成，从而在分类任务中利用样本间的关系信息。 |
| [^83] | [Noise-Aware Differentially Private Variational Inference](https://arxiv.org/abs/2410.19371) | 提出了一种基于随机梯度变分推断的噪声感知差分隐私近似贝叶斯推断新方法，突破了以往方法仅适用于简单概率模型的局限，可应用于高维和非共轭模型，并提出了一种更准确的噪声感知后验评估方法。 |
| [^84] | [Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series](https://arxiv.org/abs/2402.05203) | 贝尔曼符合推断（BCI）是一个框架，通过解决一维随机控制问题，利用多步预测来提供校准的时间序列预测区间。BCI在任意分布转换和时间依赖性下实现了长期覆盖，且在波动率预测问题上生成更短的预测区间。 |

# 详细

[^1]: 基于生成模型的条件分布相等性检验

    Testing Conditional Distribution Equality Using Generative Models

    [https://arxiv.org/abs/2610.20447](https://arxiv.org/abs/2610.20447)

    本文提出一种基于生成模型的条件分布相等性检验方法，通过结合观测数据与合成样本以及样本分割与学生化技术，在高维条件变量情形下实现了精确的检验水平控制和更强的检验功效。

    

    条件分布相等性检验是一个核心问题，在因果推断、模型诊断和迁移学习中有着广泛的应用。当条件变量为高维时，现有方法常常会损失检验功效。我们提出了一种新的检验方法，该方法利用生成模型，结合观测数据和合成样本来比较条件分布。我们引入了样本分割和学生化方案来控制估计误差。在原假设下，检验统计量具有渐近正态性，并且我们针对连续邻近备择假设建立了局部功效理论。我们在原假设和固定备择假设下的理论仅要求所学习的条件分布具有一致性，因此可以容纳一大类生成模型。实证结果表明，与代表性的基于核方法的两样本检验相比，该方法在高维设置下实现了精确的检验水平控制和强大的检验功效。

    arXiv:2610.20447v1 Announce Type: cross  Abstract: Conditional distribution equality testing is a core problem with applications in causal inference, model diagnostics, and transfer learning. Existing methods often lose power when the conditioning variable is high-dimensional. We propose a new test that leverages generative models to compare conditional distributions using both observed data and synthetic samples. A sample-splitting and studentization scheme is introduced to control estimation error. Under the null, the test statistic is asymptotically normal, and we establish local power against contiguous alternatives. Our theory under the null and fixed alternative requires only consistency of the learned conditional distribution, allowing for a broad class of generative models. Empirical results demonstrate accurate size control and strong power in high-dimensional settings compared to representative kernel-based conditional and joint two-sample tests.
    
[^2]: 在线凸优化中追踪一小组专家

    Tracking a Small Set of Experts in Online Convex Optimization

    [https://arxiv.org/abs/2610.20416](https://arxiv.org/abs/2610.20416)

    本文证明了在线凸优化中“追踪一小组专家”问题存在本质的计算复杂性分离：信息论上可达的最优遗憾度 $\widetilde{O}(md+k)$ 在超图植入团猜想等复杂性假设下无法被任何多项式时间算法实现，高效算法只能达到 $\widetilde{O}(dk)$ 量级的遗憾度。

    

    对于从专家建议中进行预测这一基本问题，Freund 曾提出疑问：算法能否与一个仅在 $m$ 位专家之间切换 $k$ 次的比较对象相竞争。对于离散的专家集合，Bousquet 和 Warmuth（2020）解决了这一问题，他们给出了一个高效算法，其遗憾度达到了信息论极限。我们在在线凸优化中研究类似的问题，此时动作空间是连续的。通过将动作空间离散化并应用 Bousquet–Warmuth 算法，可以导出一个低效的算法，其遗憾度为 $\widetilde{O}(md+k)$，而目前已知的针对该问题的最佳高效算法遗憾度为 $\widetilde{O}(dk)$。我们证明了这种计算上的低效是该问题所固有的，从而为该问题建立了信息论遗憾界与多项式时间可达遗憾界之间的分离。具体而言，在超图植入团猜想（hypergraphic planted-clique conjecture）之下，我们证明……（原文摘要在此处截断）

    arXiv:2610.20416v1 Announce Type: new  Abstract: For the fundamental problem of prediction from expert advice, Freund asked whether one could compete with a comparator that switches $k$ times among only $m$ experts. For a discrete set of experts, this problem was resolved by Bousquet and Warmuth (2020), who gave an efficient algorithm with optimal regret matching the information-theoretic limit.   We study the analogous problem in online convex optimization, where the action space is continuous. An inefficient algorithm can be derived by discretizing the action space and applying the Bousquet--Warmuth algorithm, giving regret of $\widetilde{O}(md+k)$, while the best known efficient algorithm for this problem has regret $\widetilde{O}(dk)$. We show that this computational inefficiency is inherent, establishing a separation between the information-theoretic and polynomial-time regret achievable for this problem. Specifically, under the hypergraphic planted-clique conjecture, we show that
    
[^3]: 动量何时有助于可验证奖励强化学习？

    When Does Momentum Help in Reinforcement Learning with Verifiable Rewards?

    [https://arxiv.org/abs/2610.20413](https://arxiv.org/abs/2610.20413)

    本文通过带偏差修正EMA动量的老虎机形式化分析RLVR，理论上证明精确梯度下增大动量系数β可严格加速优化，而采样估计梯度下需增大β以维持有效步长η/(1-β)来克服采样波动。

    

    动量是加速凸优化的经典工具，也是现代大语言模型训练的标准组成部分。然而，近期的实证工作对一阶动量在可验证奖励强化学习（RLVR）中的有效性提出了质疑。我们通过带有偏差修正指数移动平均（EMA）动量的RLVR多臂老虎机形式化来研究这一问题，其中EMA系数 $\beta$ 控制过去梯度权重衰减的速度。当策略梯度被精确计算时，我们证明在防止过冲（overshoot）的范围内增大 $\beta$ 能严格加速优化。当梯度从采样响应中估计得到时，正的期望增益可能被采样波动所掩盖。我们证明，随着学习率 $\eta$ 减小，固定 $\beta$ 下的期望增益比这些波动消失得更快，而增大 $\beta$ 以使 $\eta/(1-\beta)$ 保持在……（摘要在此处截断）

    arXiv:2610.20413v1 Announce Type: new  Abstract: Momentum is a classical tool for accelerating convex optimization and a standard component of modern large language model training. Yet recent empirical work has questioned the usefulness of first-moment momentum in reinforcement learning with verifiable rewards (RLVR). We study this question through a bandit formulation of RLVR with bias-corrected exponential moving average (EMA) momentum, where the EMA coefficient $\beta$ controls how quickly the weights on past gradients decay. When policy gradients are evaluated exactly, we prove that increasing $\beta$ within a range that prevents overshoot strictly accelerates optimization. When gradients are estimated from sampled responses, a positive expected gain can be obscured by sampling fluctuations. We show that as the learning rate $\eta$ decreases, the expected gain with fixed $\beta$ vanishes faster than these fluctuations, whereas increasing $\beta$ to keep $\eta/(1-\beta)$ within the 
    
[^4]: 后验频率与不拟合的代价：对记忆化下界的一个修正

    Posterior Frequencies and the Cost of Not Fitting: A Correction to a Memorization Lower Bound

    [https://arxiv.org/abs/2610.20402](https://arxiv.org/abs/2610.20402)

    本文通过精确的三点反例证明 Feldman 记忆化下界中的频率系数不成立且无法用普遍正常数因子修复，并基于全样本后验概率给出了一个精确的替代系数。

    

    arXiv:2610.20402v1 公告类型：cross（交叉发布） 摘要：Feldman 的记忆化下界根据数据点在训练样本中出现的频率来赋予其频率系数。我们利用独立公平的标签和一个置换等变的学习器，构造了一个精确的三点反例，否定了其条件形式与平均形式。通过构造一列采样先验，使所提出的惩罚项与实际超额风险之比呈指数增长，从而排除了任何普遍的正常数因子修复的可能性。这些先验随样本量变化，出问题的频率剖面变得稀有，且两个量都趋近于零。在保留原有模型的前提下，我们用全样本后验概率得到了一个精确的替代结果。在可交换性条件下，这些后验概率取决于出现次数（多重性）以及完整的频率剖面。我们刻画了对无限制学习器有效的、仅依赖多重性的最大系数。原始系数具有 Good 精确频率公式的后验均值形式，并且是一种占用加权平均。

    arXiv:2610.20402v1 Announce Type: cross  Abstract: Feldman's memorization lower bound assigns a frequency coefficient based on how often a point appears in the training sample. We give an exact three-point counterexample to its conditional and averaged forms, using independent fair labels and a permutation-equivariant learner. A sequence of sampling priors makes the ratio of the proposed penalty to actual excess risk grow exponentially, ruling out a universal positive constant-factor repair. The priors vary with sample size, the offending profiles become rare, and both quantities vanish. Retaining the original model, we obtain an exact replacement using full-sample posterior probabilities. Under exchangeability, these depend on multiplicity and the complete frequency profile. We characterize the largest multiplicity-only coefficients valid for unrestricted learners. The original coefficient has the posterior-mean form of Good's exact frequency formula and is an occupancy-weighted avera
    
[^5]: 分数阶更优：神经算子学习中的可学习导数阶数

    Fractional is Better: Learnable Derivative Orders in Neural Operator Learning

    [https://arxiv.org/abs/2610.20363](https://arxiv.org/abs/2610.20363)

    该论文提出在神经算子中引入与偏微分方程结构对齐的可学习导数特征，从理论上证明其可将逼近界从 $O(W^{-(s-m)/d})$ 改进为 $O(W^{-s/d})$，并出人意料地发现最优导数阶数是严格低于偏微分方程阶数 $m$ 的分数阶 $\beta^*$。

    

    神经算子学习函数空间之间的映射，从而能够为偏微分方程提供快速的代理求解方案。尽管这些方法在架构上具有显著的多样性，它们通常共享一种通用的输入表示：原始的坐标-数值对 $(x, u(x))$。我们探究与偏微分方程微分结构对齐的输入能否改进学习效果。通过对温和解进行Picard迭代，我们证明输入的导数会显式地出现在解算子中，这表明提供导数特征应当能够减轻网络隐式微分的负担。我们证明了这一直觉：提供导数特征可将逼近界从 $O(W^{-(s-m)/d})$ 改进为 $O(W^{-s/d})$，其中 $W$ 是参数数量，$s$ 是输入正则性，$m$ 是偏微分方程的阶数，$d$ 是空间维度。然而，我们的核心发现出人意料：最优导数阶数 $\beta^*$ 严格小于偏微分方程的阶数 $m$。这一差距

    arXiv:2610.20363v1 Announce Type: new  Abstract: Neural operators learn mappings between function spaces, enabling fast surrogate solutions to partial differential equations. Despite remarkable architectural diversity, these methods often share a common input representation: raw coordinate-value pairs $(x, u(x))$. We ask whether inputs aligned with PDE differential structure can improve learning. Through Picard iteration on mild solutions, we show that derivatives of the input appear explicitly in the solution operator, suggesting that providing derivative features should reduce the network's implicit differentiation burden. We prove this intuition: providing derivative features improves approximation bounds from $O(W^{-(s-m)/d})$ to $O(W^{-s/d})$, where $W$ is parameter count, $s$ is input regularity, $m$ is the PDE order, and $d$ is spatial dimension. Our central finding, however, is a surprise: the optimal derivative order $\beta^*$ is strictly less than the PDE order $m$. This gap 
    
[^6]: 超越专家排序：面向混合精度MoE量化的对比校准

    Beyond Expert Rankings: Contrast Calibration for Mixed-Precision MoE Quantization

    [https://arxiv.org/abs/2610.20353](https://arxiv.org/abs/2610.20353)

    提出NatScale框架，通过模型特定的幂变换将基于梯度的专家权重校准到信息论对比度目标，并联合评估门控与上下投影配置的输出误差，从而在存储预算下实现更优的混合精度MoE量化位宽分配。

    

    混合专家语言模型在每个token上仅激活少数专家，但所有专家权重都必须被存储。混合精度量化通过为敏感专家分配更高位宽、为鲁棒专家分配更低位宽来降低这种存储成本，但仅凭专家排序并不能决定对其误差的加权强度。为填补这一空白，我们提出了NatScale，一个用于混合精度MoE量化的对比校准分配框架。它使用模型特定的幂变换，在保持排序不变的前提下，将基于梯度的专家权重校准到信息论对比度目标。我们联合评估每个专家的门控、上投影和下投影配置以估计其输出误差。然后，我们将这些误差与校准后的权重，同现有的基于权重谱、方差和精度衰减的先验相结合，在存储预算下选择配置。在三个模型上，NatScale取得了最低的（摘要在此处截断）

    arXiv:2610.20353v1 Announce Type: new  Abstract: Mixture-of-experts (MoE) language models activate only a few experts per token, yet all expert weights must be stored. Mixed-precision quantization reduces this storage cost by assigning higher bit-widths to sensitive experts and lower bit-widths to resilient ones, but expert rankings alone do not determine how strongly to weight their errors. To address this gap, we propose NatScale, a contrast-calibrated allocation framework for mixed-precision MoE quantization. It uses a model-specific power transformation to calibrate gradient-based expert weights to an information-theoretic contrast target while preserving rankings. We jointly evaluate each expert's gate, up and down configurations to estimate its output error. We then combine these errors and calibrated weights with an existing prior based on weight spectra, variance and precision decay to select configurations under a storage budget. Across three models, NatScale achieves the lowe
    
[^7]: SpectrumQuant：面向大语言模型量化的频谱对齐方法

    SpectrumQuant: Spectrum Alignment for LLM Quantization

    [https://arxiv.org/abs/2610.20303](https://arxiv.org/abs/2610.20303)

    提出SpectrumQuant，一种基于协方差最大化的闭式正交旋转方法，通过频谱对齐构建量化友好的变换，从而提升大语言模型训练后低比特量化的效果。

    

    正交变换已成为大语言模型（LLM）低比特量化的有效工具，因为它能够在不改变底层模型函数的情况下重塑激活分布。现有方法通常依赖于固定的、与数据无关的变换（如Hadamard旋转），或者依赖于在校准阶段对可学习旋转的优化。我们从第一性原理出发研究旋转选择问题，并证明最大化协方差可以带来频谱对齐，从而为构建量化友好的正交变换提供了一个有理论依据的准则。基于这一洞察，我们提出了SpectrumQuant，这是一种闭式解的旋转方法，它从校准数据中估计激活协方差，并利用它构建协方差感知的正交变换以用于训练后量化。我们的公式化将量化友好的旋转刻画为对齐分量与混合分量的组合，同时将Hadamard旋转作为特例恢复出来（此句原文被截断）。

    arXiv:2610.20303v1 Announce Type: new  Abstract: Orthogonal transforms have emerged as an effective tool for low-bit quantization of large language models (LLMs), as they reshape activation distributions without changing the underlying model function. Existing approaches typically rely either on fixed data-agnostic transforms, such as Hadamard rotations, or on calibration-time optimization of learned rotations. We study rotation selection from first principles and show that maximizing covariance leads to spectral alignment and provides a principled criterion for constructing quantization-friendly orthogonal transforms. Based on this insight, we propose SpectrumQuant, a closed-form rotation method that estimates activation covariance from calibration data and uses it to construct a covariance-aware orthogonal transform for post-training quantization. Our formulation characterizes quantization-friendly rotations as compositions of alignment and mixing components, while recovering Hadamar
    
[^8]: 从柏拉图到亚里士多德：迈向表征假设的通用理论

    From Plato to Aristotle: Toward a General Theory of Representation Hypotheses

    [https://arxiv.org/abs/2610.20286](https://arxiv.org/abs/2610.20286)

    该论文提出格表征假设（LRH），将各种表征对齐准则统一到同一个格结构中，从而解释了柏拉图表征假设的收敛声明与其实际证据之间的差异，并提出亚里士多德表征假设将收敛限定于局部邻域。

    

    柏拉图表征假设（PRH）认为，能力日益增强的模型会收敛到一种共同的表征。然而，最近的研究通过邻域、生成树和距离等方式对齐表征时呈现出不同的观点。我们证明这些观点之所以存在差异，是因为它们背后的对齐准则位于同一个格的不同位置，并将这一解释形式化为格表征假设（LRH）。在LRH框架下，PRH所声称的收敛对应于格顶部的共识，而其实际证据则来自邻域重叠——这是一种接近格底部的准则。亚里士多德表征假设（ARH）将收敛限制在局部邻域，因为校准保留了邻域重叠随模型规模增长的趋势，却在很大程度上消除了中心化核相关性（CKA）的增长。我们证明这种消除可能源于对层间最大值的校准，因此全局收敛……

    arXiv:2610.20286v1 Announce Type: new  Abstract: The Platonic Representation Hypothesis (PRH) holds that increasingly capable models converge toward a common representation. However, recent studies have shown different views on aligning representations via their neighborhoods, spanning trees, and distances. We show that these views can differ because the alignment criteria behind them sit at different points of one lattice, and we formalize this explanation as the Lattice Representation Hypothesis (LRH). Under the LRH, the convergence claimed by the PRH is agreement at the top of the lattice, whereas its evidence comes from neighbor overlap, a criterion near the bottom. The Aristotelian Representation Hypothesis (ARH) restricts convergence to local neighborhoods because calibration keeps the growth of neighbor overlap with model size but largely removes that of centered kernel alignment. We show that this removal can come from calibrating a maximum over layers, so global convergence re
    
[^9]: MAEM：通过无重放、容量控制的动态专家混合来缓解类增量图学习中的近期偏差

    MAEM: Mitigating Recency Bias via a Replay-Free, Capacity-Controlled Dynamic Mixture of Experts for Class-Incremental Graph Learning

    [https://arxiv.org/abs/2610.20238](https://arxiv.org/abs/2610.20238)

    提出MAEM框架，通过为每个新会话训练并冻结专用专家、以无重放且容量受控的动态专家混合方式进行路由，从而在无需任务标识的情况下缓解类增量图学习中的近期偏差与灾难性遗忘。

    

    图神经网络在图结构数据上表现出色，但当类别按顺序依次到达时会遭受灾难性遗忘。在类增量图学习中，预测结果往往由最近学习的类别主导，从而产生严重的近期偏差。动态扩展方法可以将知识在不同会话间隔离，但许多方法假设的是任务增量设置，在测试时需要任务标识来选择分支或输出头——而这一信息在类增量学习中是无法获得的。可学习的路由器提供了一种替代方案，但在新数据上持续微调路由器可能使路由偏向最近添加的组件。我们提出了MAEM，一个面向类增量图学习的无重放、容量控制的动态专家混合框架。MAEM使用简化图卷积（SGC）构建节点表示，并为每个新会话训练一个专用专家，随后将其冻结以保护已有知识。对于无需任务ID的路由机制……（摘要原文在此处截断）

    arXiv:2610.20238v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) excel on graph-structured data but suffer catastrophic forgetting when classes arrive sequentially. In class-incremental graph learning, predictions are often dominated by recently learned classes, causing severe recency bias. Dynamic expansion can isolate knowledge across sessions, but many methods assume task-incremental settings and require task identities at test time to select branches or heads---information unavailable in class-incremental learning. Learnable routers offer an alternative, yet continual fine-tuning the router on new data can bias routing toward recent components. We propose MAEM, a replay-free, capacity-controlled dynamic mixture-of-experts framework for class-incremental graph learning. MAEM builds representations with Simplified Graph Convolution (SGC) and trains a dedicated expert for each new session, then freezes it to protect prior knowledge. For routing without task IDs or a share
    
[^10]: 神经网络的最小宽度与实射影空间的浸入

    The minimal width of neural networks and immersions of real projective space

    [https://arxiv.org/abs/2610.20229](https://arxiv.org/abs/2610.20229)

    本文证明了当实射影空间 $\mathbb{RP}^{n-1}$ 可光滑浸入 $\mathbb{R}^n$ 时，神经网络逼近从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 连续函数的最小宽度至少为 $n+\min\{n,m\}$，从而在 $n=2,3,4,7,8$ 时确定了 ReLU 网络的最小宽度为 $n+m$，揭示了神经网络逼近理论与微分拓扑之间的深刻联系。

    

    我们研究了在紧集上一致收敛拓扑下，能够逼近所有从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 的连续函数的深度神经网络的最小宽度。我们证明，只要实射影空间 $\mathbb{RP}^{n-1}$ 可以光滑浸入到 $\mathbb{R}^n$ 中，在激活函数连续且单调的假设下，最小宽度至少为 $n+\min\{n,m\}$。由此可得，当 $n$ 等于 2、3、4、7 或 8 且 $1\le m \le n$ 时，ReLU 网络的最小宽度为 $n+m$，这扩展了 Hanin-Sellke 和 Hwang 的结果。这表明神经网络的逼近理论与微分拓扑之间存在着一种有趣的联系。

    arXiv:2610.20229v1 Announce Type: cross  Abstract: We investigate the minimal width of deep neural networks that can approximate all continuous functions from $\mathbb{R}^n$ to $\mathbb{R}^m$ in the topology of uniform convergence on compact subsets. We show that whenever the real projective space $\mathbb{RP}^{n-1}$ admits a smooth immersion into $\mathbb{R}^n$, the minimal width is at least $n+\min\{n,m\}$, assuming that the activation function is continuous and monotone. As a consequence, the minimal width of ReLU networks is $n+m$ whenever $n$ is equal to 2, 3, 4, 7 or 8 and $1\le m \le n$, extending results of Hanin-Sellke and Hwang. This suggests an intriguing connection between the approximation theory of neural networks and differential topology.
    
[^11]: 混合泊松扩散模型及其在后验采样中的应用

    Mixed Poisson diffusion models and application to posterior sampling

    [https://arxiv.org/abs/2610.20218](https://arxiv.org/abs/2610.20218)

    该论文提出了一种由多元计数过程驱动的混合泊松扩散模型（MPDM），为观测数据离散且服从泊松分布的泊松逆问题提供了后验采样的新框架。

    

    扩散模型是学习复杂概率分布的强大框架。虽然广泛使用的连续数据扩散模型依赖于高斯加噪过程，但我们转而开发了基于泊松噪声的扩散模型。更准确地说，我们引入并研究了一个新的数学框架，称为混合泊松扩散模型（Mixed Poisson Diffusion Models, MPDM），其加噪动力学由多元计数过程驱动。我们的方法受泊松逆问题的启发。高斯扩散模型的成功使其在越来越多的逆问题后验采样方法中被用作先验模型。然而，这些方法大多聚焦于高斯逆问题，即观测值通过确定性前向算子生成并被加性高斯噪声所污染。相比之下，观测值离散且服从泊松分布的泊松逆问题在此领域仍相对缺乏探索。

    arXiv:2610.20218v1 Announce Type: cross  Abstract: Diffusion models are a powerful framework for learning complex probability distributions. While widely used diffusion models for continuous data rely on Gaussian noising processes, we instead develop diffusion models based on Poisson noise. More precisely, we introduce and study a new mathematical framework, called Mixed Poisson Diffusion Models (MPDM), with noising dynamics driven by multivariate counting processes.   Our methodology is motivated by Poisson inverse problems. The success of Gaussian diffusion models has led to their use as prior models in a growing number of posterior sampling methods for inverse problems. However, most of these approaches focus on Gaussian inverse problems, in which observations are generated through a deterministic forward operator corrupted by additive Gaussian noise. By contrast, Poisson inverse problems, where observations are discrete and Poisson distributed, remain comparatively underexplored in
    
[^12]: 使用混合整数线性规划优化稀疏布尔规则网络

    Optimizing Sparse Boolean Rule Networks using Mixed Integer Linear Programming

    [https://arxiv.org/abs/2610.20170](https://arxiv.org/abs/2610.20170)

    该论文提出用混合整数线性规划训练稀疏的布尔规则网络（DNF网络），通过增强的公式大幅减少整数决策变量并消除冗余约束，在保持精确布尔语义的同时实现误分类成本与稀疏性之间的全局最优权衡。

    

    我们研究学习准确且认知上简单的二分类器的问题，这类分类器可以直接解释为布尔规则。我们提出了用于训练稀疏析取范式（DNF）网络的混合整数线性规划（MILP）公式，这类网络可被视为由合取隐藏层和析取输出层组成的小型网络。与以往使用MILP进行归纳规则学习的工作不同，我们从固定的网络结构出发，进而为其寻找最优权重。正则化的目标函数明确地在误分类成本与稀疏性之间进行权衡，从而产生全局优化的规则集。我们的主要技术贡献是一种增强的公式，它大幅减少了整数决策变量的数量并消除了冗余约束，同时对所有可行解保持精确的布尔语义。此外，我们开发了一种基于初始化和dropout相结合的方法来打破对称性……

    arXiv:2610.20170v1 Announce Type: new  Abstract: We study the problem of learning accurate and cognitively simple binary classifiers that are directly interpretable as Boolean rules. We propose mixed integer linear programming (MILP) formulations for training sparse disjunctive normal form (DNF) networks, viewed as small networks with a conjunctive hidden layer and a disjunctive output layer. Contrary to previous work that uses MILP for inductive rule learning, we start with a fixed network structure, for which we then find optimal weights. A regularized objective explicitly trades off misclassification costs against sparsity, yielding globally optimized rule sets. Our main technical contribution is an enhanced formulation that substantially reduces the number of integer decision variables and eliminates redundant constraints, while preserving exact Boolean semantics for all feasible solutions. Furthermore, we develop a combined approach based on initialization and dropout to break sym
    
[^13]: 基于广义全变差最小化的最小二乘联邦学习的大维度分析

    A Large-Dimensional Analysis of Least-Squares Federated Learning via Generalized Total Variation Minimization

    [https://arxiv.org/abs/2610.20075](https://arxiv.org/abs/2610.20075)

    本文通过随机矩阵理论对基于广义全变差最小化的最小二乘联邦学习进行大维度分析，给出了独立同分布与非独立同分布高斯混合数据下渐近精确性能的精确解析表达式，量化了正则化如何在局部适应与协作之间进行权衡。

    

    去中心化网络上的联邦学习本质上是在通过协作抑制局部有限样本噪声与协作在异构节点之间引入的偏差之间进行权衡。我们通过对基于最小二乘模型的广义全变差最小化进行大维度分析来研究这种张力。利用随机矩阵理论的工具，我们证明了全局GTVMin解的Lipschitz观测在大维度下是集中的，且对任何底层网络结构都普遍成立。我们提供了在固定网络上独立同分布与非独立同分布高斯混合分布下渐近精确性能的精确解析表达式。对于后者，这些表达式量化了正则化如何在局部适应与协作之间取得平衡。我们的理论结果通过合成数据和真实世界数据的实验得到了验证。

    arXiv:2610.20075v1 Announce Type: new  Abstract: Federated learning over decentralized networks inherently trades off the suppression of local finite-sample noise through collaboration against the bias that collaboration introduces between heterogeneous nodes. We study this tension by performing a large-dimensional analysis of Generalized Total Variation Minimization with least-squares models. Using tools from random matrix theory, we show that Lipschitz observations of the global GTVMin solution concentrate in large dimensions, universally for any underlying network structure. We provide exact analytic expressions for the asymptotically precise performance under both i.i.d. and non-i.i.d. Gaussian mixture distributions on a fixed network. For the latter, these expressions quantify how regularization balances local adaptation against collaboration. Our theoretical results are confirmed by experiments on synthetic and real-world data.
    
[^14]: 加权伴随匹配：使用时间依赖控制代价微调流匹配模型

    Weighted Adjoint Matching: Finetuning Flow Matching Models with Time-Dependent Control Cost

    [https://arxiv.org/abs/2610.20035](https://arxiv.org/abs/2610.20035)

    提出加权伴随匹配方法，通过在控制代价上引入时间依赖权重，早期保持粗略语义特征的多样性与平衡、后期精细化细节以提升奖励，从而在微调流匹配模型时实现奖励与模式平衡之间更好的权衡。

    

    我们提出了加权伴随匹配，这是一种在给定可微奖励的情况下微调预训练扩散模型或流匹配模型的方法。该方法源自伴随匹配，后者将微调重新表述为具有二次控制代价的随机最优控制问题。基于这些模型先生成粗略特征、后生成精细细节的观察，我们在控制代价上引入了时间依赖的权重。在生成过程早期施加高代价可以保持携带大部分语义信息的粗略特征的平衡性与多样性，而在最后阶段施加低代价则允许模型通过细化细节来提高奖励。我们证明了该方法等价于在生成路径上施加时变KL散度正则化条件下最大化奖励。在高斯混合分布和MNIST数据集上的实验表明，与伴随匹配相比，我们的方法在奖励和模式平衡之间实现了更好的权衡。

    arXiv:2610.20035v1 Announce Type: cross  Abstract: We introduce Weighted Adjoint Matching, a method to finetune a pretrained diffusion or flow matching model given a differentiable reward. We derive it from Adjoint Matching, which reformulates finetuning as a stochastic optimal control problem with a quadratic control cost. Building on the observation that these models generate coarse features before fine details, we place a time-dependent weight on the control cost. A high cost early in the generative process preserves balance and diversity in the coarse features, which carry most of the semantics, while a low cost at the end lets the model increase the reward by refining details. We prove that our method is equivalent to maximizing the reward under time-varying Kullback--Leibler regularization on the generative paths. Experiments on Gaussian mixtures and on MNIST show a better tradeoff between reward and mode balance than Adjoint Matching.
    
[^15]: MCMC 能看见的与看不见的

    MCMC Can See Can't See

    [https://arxiv.org/abs/2610.20020](https://arxiv.org/abs/2610.20020)

    本文证明了任何有限的收敛诊断清单都无法保证MCMC真正收敛，并定量刻画了最坏情况下未被检测到的未收敛性如何随诊断方法增加而衰减、如何最优选择诊断方法，以及未被检测到的偏差如何限制有限诊断所能认证的有效样本量。

    

    众所周知，通过“收敛性”诊断并不能保证马尔可夫链已经收敛。我们证明了这一点对于任何有限的诊断清单而言都是一种不可能性。运行更多的链、更长的链以及更聪明的诊断方法，都无法避免在认证收敛时的这种不可能性。在正则性假设下，我们确定了随着加入更多诊断方法，最坏情况下未被检测到的未收敛性如何衰减，以及如何最优地选择诊断方法。民间观点认为，采样器混合最慢的模式是需要监测的“危险”模式；我们证明了随着采样的进行，慢模式能产生近乎最优的诊断。对于给定的源类别，我们还给出了剩余差异的任意时刻有效界限，并展示了未被检测到的偏差如何限制有限诊断清单所能认证的有效样本量。

    arXiv:2610.20020v1 Announce Type: cross  Abstract: It is folklore that passing "convergence" diagnostics does not guarantee that a Markov chain has converged. We prove this as an impossibility for any finite checklist of diagnostics. Running more chains, longer chains, and smarter diagnostics does not avoid this impossibility in certifying convergence. Under regularity assumptions, we determine how the worst-case undetected non-convergence decays as more diagnostics are added, and how to choose diagnostics optimally. Folklore suggests that the sampler's slowest mixing modes are the "dangerous" ones that must be monitored; we prove that the slow modes yield nearly optimal diagnostics as sampling proceeds. Given a source class, we also give an anytime-valid bound on the remaining discrepancy and show how undetected bias caps the effective sample size that a finite checklist can certify.
    
[^16]: 使用双曲目标条件强化学习超越1000层欧几里得网络

    Outperforming 1000 Euclidean Layers With Hyperbolic Goal-Conditioned RL

    [https://arxiv.org/abs/2610.20009](https://arxiv.org/abs/2610.20009)

    通过将对比强化学习中的欧几里得距离替换为双曲空间的测地距离，仅64层深的双曲智能体就能以超过12倍的训练速度和约13倍的参数量优势，超越1024层欧几里得网络的导航性能。

    

    将自监督对比强化学习（CRL）扩展到1000层欧几里得神经网络已经实现了最先进的目标达成性能，但其实际运行时间随深度线性增长。我们发现，改变几何结构可以用浅得多的网络实现更强的性能。这在复杂导航环境中尤为重要，因为墙壁会使路径距离与欧几里得距离产生显著偏差。在U形迷宫中，我们证明了路径越长，其与欧几里得距离的偏差就越大。双曲CRL通过将CRL对比目标中的欧几里得距离替换为双曲空间中的测地距离来解决这种不匹配问题。在困难导航任务上，归纳偏置胜过规模：深度为64的双曲智能体超越了1024层深的欧几里得网络，同时训练速度快12倍以上，参数量少约13倍。我们观察到双曲智能体的行为在质上……

    arXiv:2610.20009v1 Announce Type: new  Abstract: Scaling self-supervised contrastive reinforcement learning (CRL) to 1000-layer Euclidean neural networks has enabled state-of-the-art goal-reaching performance, yet wall-clock time scales linearly with depth. We find that changing the geometry achieves stronger performance with far shallower networks. This matters in complex navigation environments, where walls can make route distance deviate substantially from Euclidean distance. In a U-shaped maze, we show that the longer the route, the more it deviates from Euclidean distance. Hyperbolic CRL addresses this mismatch by replacing Euclidean distances in CRL's contrastive objective with geodesic distances in hyperbolic space. On hard navigation tasks, inductive bias beats scale: depth 64 hyperbolic agents outperform 1024-layer-deep Euclidean networks while training more than $12 \times$ faster and using about $13 \times$ fewer parameters. We observe that hyperbolic agents behave qualitati
    
[^17]: 连续马尔可夫链伪谱隙的极小极大估计及其在经验集中界中的应用

    Minimax estimation of the pseudo-spectral gap of continuous Markov chains, with applications to empirical concentration bounds

    [https://arxiv.org/abs/2610.20007](https://arxiv.org/abs/2610.20007)

    本文提出了紧致连续状态空间马尔可夫链伪谱隙的首个极小极大最优估计器，结合 Bernstein 不等式推导出经验集中不等式，并据此在无需伪谱隙先验知识的情况下，为马尔可夫奖励老虎机问题给出了具有接近独立同分布情形遗憾界的 UCB 算法。

    

    我们研究了基于单条轨迹观测来对马尔可夫链伪谱隙 $\gamma_{ps}$ 进行极小极大估计的问题。通过扩展 Wolfer 和 Kontorovich (2024) 关于有限状态空间链的早期工作，我们针对紧致连续状态空间的情形提出了一个估计器。在转移算子具有 Lipschitz 密度的链类上，我们的估计器在相差对数因子的意义下似乎是极小极大最优的。将该估计器与 Paulin 版本的 Bernstein 不等式相结合，可以为马尔可夫链推导出经验集中不等式。这开启了许多应用，包括置信区间、主动学习和泛化界。特别地，对于具有马尔可夫奖励的静止老虎机问题，我们提出了一个 UCB 算法版本，其遗憾界与独立同分布情形类似，且无需对链的伪谱隙有任何先验知识。

    arXiv:2610.20007v1 Announce Type: cross  Abstract: We study the minimax estimation of the pseudo-spectral gap $\gamma_{ps}$ of a Markov chain from the observation of a single trajectory. Extending earlier works on finite-state space chains by Wolfer and Kontorovich (2024), we provide an estimator in the case of a compact, continuous state-space. Our estimator appears to be minimax-optimal on the class of chains with a transition operator with Lipschitz density, up to logarithmic terms. Our estimator, combined with Paulin's version of Bernstein inequality, allows to derive empirical concentration inequalities for Markov chains. This unlocks many applications, from confidence intervals, to active learning and generalization bounds. In particular, for rested bandits with Markov rewards, we provide a version of UCB with a regret bound similar to the one in the iid case, without assuming any knowledge on the pseudo-spectral gap of the chains.
    
[^18]: 高效全局优化中高斯过程汤普森采样的收敛速率

    Convergence Rates of Gaussian Process Thompson Sampling for Efficient Global Optimization

    [https://arxiv.org/abs/2610.19925](https://arxiv.org/abs/2610.19925)

    本文首次在无噪声RKHS设定下为高斯过程汤普森采样建立了最坏情况简单遗憾上界 $O(N^{-\nu/d}\sqrt{\log N})$ 及匹配的下界，证明对数惩罚项是该算法的固有代价。

    

    高效全局优化旨在在有限的评估预算 $N$ 内找到昂贵的黑盒函数的全局最优解。我们在无噪声观测条件下研究高斯过程汤普森采样（GP-TS）的收敛速率，并假设目标函数固定于再生核希尔伯特空间（RKHS）中。对于维度为 $d$、光滑度参数为 $\nu$ 的Matérn核，我们为足够精细网格上的离散GP-TS及其连续版本建立了最坏情况下期望简单遗憾上界 $O(N^{-\nu/d}\sqrt{\log N})$。对于连续实现，我们开发了一种嵌套网格论证方法来控制整个定义域上的后验样本路径。我们进一步在固定的一维Matérn-5/2核下，为两种实现建立了匹配的下界 $\Omega(N^{-5/2}\sqrt{\log N})$，证明了对数惩罚项是标准GP-TS在此类设定中的固有特性。

    arXiv:2610.19925v1 Announce Type: cross  Abstract: Efficient global optimization aims to identify a global optimum of an expensive black-box function within a limited evaluation budget $N$. We investigate the convergence rates of Gaussian process Thompson sampling (GP-TS) under noiseless observations, assuming a fixed objective function in a reproducing kernel Hilbert space (RKHS). For Mat\'ern kernels with smoothness parameter $\nu$ in dimension $d$, we establish a worst-case expected simple-regret upper bound of $O(N^{-\nu/d}\sqrt{\log N})$ for both discrete GP-TS on sufficiently fine grids and its continuous counterpart. For the continuous implementation, we develop a nested-grid argument to control posterior sample paths over the entire domain. We further establish matching lower bounds of $\Omega(N^{-5/2}\sqrt{\log N})$ for both implementations under a fixed one-dimensional Mat\'ern-$5/2$ kernel, demonstrating that the logarithmic penalty is intrinsic to standard GP-TS in this set
    
[^19]: 极大似然、神经网络与后验均值用于推断分数布朗运动参数

    Maximum likelihood, neural networks and posterior mean for inferring fractional Brownian motion parameters

    [https://arxiv.org/abs/2610.19888](https://arxiv.org/abs/2610.19888)

    该论文提出将极大似然估计与神经网络（经证明可逼近后验均值）相结合来联合估计分数布朗运动的Hurst指数和广义扩散系数，并表明后验均值结合轮廓似然估计在绝大多数情况下具有最低的均方根误差。

    

    分数布朗运动（fBM）是一种异常扩散模型，由Hurst指数 $H \in (0,1)$ 和广义扩散系数 $D_H > 0$ 参数化。我们使用极大似然估计（MLE），从带有泊松计数噪声的密度测量数据以及带有高斯定位误差的单粒子轨迹数据中联合估计 $H$ 和 $D_H$。我们证明，单一时间点的密度测量无法同时推断 $H$ 和 $D_H$，需要多个时间点才能实现，这对推断精度至关重要。对于轨迹数据，我们证明使用平方误差损失训练的神经网络可以逼近 $H$ 的后验均值，而后验均值可以以可忽略的成本被精确计算。结合 $D_H$ 的轮廓似然估计，后验均值在 $H$ 和 $D_H$ 上均具有最低的均方根误差，除了在 $H=0$ 和 $H=1$ 附近，此时MLE能更好地估计 $H$。在来自两种细胞类型的囊泡轨迹数据中，$H$ 和 $D_H$ 形成一个连……

    arXiv:2610.19888v1 Announce Type: cross  Abstract: Fractional Brownian motion (fBM) is a model of anomalous diffusion parameterised by a Hurst exponent $H \in (0,1)$ and a generalised diffusion coefficient $D_H > 0$. We jointly estimate $H$ and $D_H$ using maximum likelihood (MLE) from density measurements with Poisson counting noise and single-particle trajectory with Gaussian localisation error. We show that one density measurement in time cannot infer both $H$ and $D_H$ and that several time points are needed, which matters most for inference precision. For trajectories, we show that a neural network trained with a squared error loss approximates the posterior mean of $H$, which can be computed exactly at negligible cost. Paired with the profile-likelihood estimate of $D_H$, the posterior mean has the lowest root mean squared error in both $H$ and $D_H$, except near $H=0$ and $H=1$ where MLE estimates $H$ better. In vesicle trajectories from two cell types, $H$ and $D_H$ form a cont
    
[^20]: 神经瓦瑟斯坦样条：用于轨迹推断的表示不变二阶先验

    Neural Wasserstein Splines: Representation-Invariant Second-Order Priors for Trajectory Inference

    [https://arxiv.org/abs/2610.19878](https://arxiv.org/abs/2610.19878)

    提出神经瓦瑟斯坦样条（NWSP），通过瓦瑟斯坦平滑先验和内在瓦瑟斯坦加速度的有限维替代量，实现表示不变的二阶正则化，从而从不成对的快照中学习连续的种群演化曲线。

    

    种群动态通常只能以不成对的快照形式被观测到，这使得测量之间的演化过程欠定。二阶正则化可以约束这一演化，但对采样粒子加速度施加惩罚是对联合粒子路径的先验，而同一边际曲线的不同粒子表示可能具有不同的加速度代价。为此，我们提出了神经瓦瑟斯坦样条（Neural Wasserstein Splines, NWSP），这是一种基于样本的方法，在瓦瑟斯坦平滑先验下学习连续的种群曲线。一个神经网络将潜变量映射为时间上三次样条的控制点，因此任意时刻的样本都可以直接由样条基产生，而无需数值积分。我们将生成器的速度投影到有限标量势字典的梯度上，并对这一依赖于测度的投影进行时间求导，从而得到内在瓦瑟斯坦加速度的有限维替代量，用于训练……

    arXiv:2610.19878v1 Announce Type: new  Abstract: Population dynamics are often observed only as unpaired snapshots, which leave the evolution between measurements underdetermined. Second-order regularization can constrain it, but a penalty on the accelerations of sampled particles is a prior on joint particle paths, and different particle representations of the same marginal curve can have different acceleration costs. We therefore introduce Neural Wasserstein Splines (NWSP), a sample-based method that learns a continuous population curve under a Wasserstein smoothness prior. A neural network maps latent variables to the control points of a cubic spline in time, so samples at any time come from the spline basis without numerical integration. We project the generator's velocity onto gradients of a finite dictionary of scalar potentials and differentiate this measure-dependent projection in time, obtaining a finite-dimensional surrogate for the intrinsic Wasserstein acceleration, trained
    
[^21]: 关于过参数化加权最大均值差异（MMD）的优化景观研究

    On the Optimisation Landscape of Overparameterised Weighted MMD

    [https://arxiv.org/abs/2610.19861](https://arxiv.org/abs/2610.19861)

    该论文首次为加权MMD优化提供了有限粒子的过参数化保证，证明对于一维高斯核，在粒子数量足够多时所有局部极小值都具有任意小的MMD，且所有虚假驻点均不稳定。

    

    我们研究了加权原子测度上最大均值差异（MMD）的优化景观。对于一维高斯核，我们证明，对于任何紧支撑的目标分布，足够多的粒子可以确保每个局部极小值都具有任意小的MMD。此外，任何MMD仍然远离零的驻点，在粒子位置的无权重梯度流下都是不稳定的。这些结果既适用于权重被约束在单纯形上的情况，也适用于权重不受约束但每一步都被优化的情况。当目标分布本身是有限支撑时，我们进一步加强了这一结果：超过一个明确的粒子数量阈值后，不存在虚假局部极小值，且每个虚假驻点都是不稳定的。据我们所知，这些是针对加权MMD优化的首类有限粒子过参数化保证。

    arXiv:2610.19861v1 Announce Type: cross  Abstract: We study the optimisation landscape of the maximum mean discrepancy (MMD) over weighted atomic measures. For the one-dimensional Gaussian kernel, we show that, for any compactly supported target distribution, sufficiently many particles ensure that every local minimum has arbitrarily small MMD. Moreover, any stationary point whose MMD remains bounded away from zero is unstable under an unweighted gradient flow over the particle locations. These results hold both when the weights are constrained to the simplex, and when they are unconstrained but optimised at each step. When the target is itself finitely supported, we further strengthen this result: beyond an explicit particle threshold, there are no spurious local minima, and every spurious stationary point is unstable. To our knowledge, these are the first finite-particle overparameterisation guarantees of their kind for weighted MMD optimisation.
    
[^22]: 基于文本信息的训练数据过滤以提升虚拟流量计性能

    Text-Informed Training Data Filtering for Enhanced Virtual Flow Meter Performance

    [https://arxiv.org/abs/2610.19860](https://arxiv.org/abs/2610.19860)

    本文提出一种利用大语言模型将非结构化文本信息（如工程师评论和操作日志）与表格化传感器数据相结合的数据过滤方法，通过上下文异常检测去除低质量数据，从而提升虚拟流量计的预测性能。

    

    在能源、水产养殖和化工等行业中，机理模型和数据驱动模型的有效性常常受到低质量训练数据的限制，这些数据会掩盖潜在的物理规律并降低预测精度。此类质量问题通常表现为数据集中的异常，涵盖传感器故障、工艺扰动以及操作员的错误输入等。虽然传统的基于规则的过滤方法能够去除一部分异常，但它无法利用以非结构化文本形式存储的上下文信息，例如解释数据采集条件的工程师评论和操作日志。我们提出了一种基于文本信息的数据过滤方法，利用大语言模型（LLM）将非结构化运营数据与表格化传感器数据相结合。该方法包含两个步骤：（1）通过对两种数据模态进行推理来检测上下文异常；（2）生成一个二……（摘要原文在此处截断）

    arXiv:2610.19860v1 Announce Type: cross  Abstract: In sectors such as energy, aquaculture and chemical processing, the effectiveness of both mechanistic and data-driven models is frequently limited by poor-quality training data, which obscures underlying physical laws and degrades predictive accuracy. Such quality issues commonly manifest as anomalies in the dataset, ranging from sensor faults to process upsets and incorrect operator inputs. While traditional rule-based filtering can remove a subset of the anomalies, it is unable to leverage contextual information stored as unstructured text, such as engineer comments and operational logs that explain under what conditions the data was collected. We propose a text-informed data filtering method that leverages large language models (LLMs) to integrate unstructured operational data with tabular sensor data. The method consists of two steps: (1) contextual anomaly detection by reasoning over both data modalities, and (2) generation of a b
    
[^23]: 基于系统1模型的校准与风险预算决策

    Calibration and Risk-Budgeted Decisions with System One Models

    [https://arxiv.org/abs/2610.19811](https://arxiv.org/abs/2610.19811)

    该论文提出在LLM工作流的二元决策场景中，通过不确定时采取保守回退动作，并利用Learn Then Test（LTT）的有限样本统计保证来控制假遗漏风险，显著优于朴素阈值选择基线方法。

    

    大语言模型（LLM）工作流与智能体依赖于一系列二元决策，例如允许/拦截输入，或丢弃/保留检索到的文档。当无法进行人工转交处理时，不确定的案例仍必须被赋予一个动作。我们在该场景下研究系统1决策分数：在不确定时默认采取保守的回退动作，并在被分配效用最大化动作的输入子集内控制假遗漏风险。在涵盖内容审核、安全性和重排序的六个数据集上，我们将通用系统1模型与任务基线以及一个本地开源权重微调的系统1模型进行比较。结果表明，朴素的阈值选择基线在大约一半具有明确测试风险的设定中超出目标风险，而利用有限样本统计保证的Learn Then Test（LTT）方法仅超出7.8%。我们的结果说明了为什么仅依赖类似概率的输出接口不足以支撑部署决策，以及如何进行风险预算分配。

    arXiv:2610.19811v1 Announce Type: cross  Abstract: LLM workflows and agents rely on a set of binary decisions like allow/block an input, or discard/retain a retrieved document. When human deferral is unavailable, uncertain cases must still receive an action. We study System One decision scores in this setting by defaulting to a conservative fallback when uncertain, and controlling the false omission risk within the subset of inputs assigned the utility-maximizing action. Across six datasets covering moderation, safety, and reranking, we compare general System One models with task baselines and a local open-weights fine-tuned System One model. We show that naive baselines for threshold selection exceed the target risk in roughly half of settings with defined test risk, compared with 7.8\% for Learn Then Test (LTT), which leverages finite-sample statistical guarantees. Our results show why relying on a probability-like interface is insufficient for deployment decisions and how risk-budge
    
[^24]: 基于通用函数逼近的高效上下文部分监控

    Efficient contextual partial monitoring with general function approximation

    [https://arxiv.org/abs/2610.19804](https://arxiv.org/abs/2610.19804)

    本文提出了计算高效的上下文部分监控框架 E2D.PMSide，针对原始 E2D 算法探索规则难以计算的问题设计了两种高效的探索规则，在通用函数逼近下实现了最优遗憾阶。

    

    我们研究了带有辅助信息的有限随机部分监控（PM）博弈，这是部分反馈序列学习的一种范式。在每一轮 t ∈ [T] 中，玩家和环境都观察到一个上下文，玩家选择一个动作，环境同时选择一个结果。玩家既无法观察到结果，也无法观察到损失；相反，他们只能观察到一个关于结果提供部分信息的反馈符号。我们利用 Foster 等人（2021）提出的估计到决策（E2D）元算法，设计了 E2D.PMSide——一个计算高效的上下文部分监控框架，该框架使用在线回归预言机，基于历史的（上下文、动作、反馈符号）元组高效地估计结果分布。由于原始 E2D 算法所给出的探索规则难以计算，我们提出了两种计算高效的探索规则，实现了最优的遗憾阶。

    arXiv:2610.19804v1 Announce Type: new  Abstract: We study finite stochastic partial monitoring (PM) games with side information, a paradigm of sequential learning with partial feedback. On each round $t \in [T]$, both player and environment observe a context and player chooses an action and environment simultaneously chooses an outcome. The player neither observes the outcome nor the loss; instead, they only observe a feedback symbol which is partially informative about the outcome. We utilize the Estimation-to-Decisions (E2D) meta-algorithm of Foster et al.(2021) to design E2D.PMSide, a computationally efficient framework for contextual partial monitoring, which uses an online regression oracle to efficiently estimate the outcome distributions based on historical (context, action, feedback symbol) tuples. Since the exploration rule given by the original E2D algorithm is intractable to compute, we propose two computationally efficient exploration rules that achieve optimal regret order
    
[^25]: 基于正-未标记学习的扩散模型渐近精确负引导

    Asymptotically Exact Negative Guidance of Diffusion Models via Positive-Unlabeled Learning

    [https://arxiv.org/abs/2610.19802](https://arxiv.org/abs/2610.19802)

    提出一种从显式目标密度采样的扩散模型负引导新方法，利用正-未标记学习在线估计样本属于不良条件的概率，从而实现渐近精确、偏差可通过增加计算量系统降低的负引导。

    

    扩散模型的负引导旨在生成避开某种不期望条件的样本，该条件既可以由参考样本指定，也可以由条件得分与无条件模型共同指定。现有方法在每个样本处注入由局部信息推导出的排斥项，因此所得采样器并不以某个显式密度为目标，其偏差也无法通过增加计算量而被系统地降低。与之相反，我们提出了一种从显式目标密度采样的负引导方法，该方法能够统一处理基于参考样本和基于得分的两种条件指定方式，并以一个参数化目标密度本身的回避强度来抑制不期望条件附近的概率质量。实现这一目标需要在每个扩散时间步估计每个样本属于不期望条件的可能性。我们将这一估计问题构造为一个正-未标记学习问题，该问题支持轻量级的在线训练。

    arXiv:2610.19802v1 Announce Type: new  Abstract: Negative guidance of diffusion models generates samples that avoid an undesired condition, specified either by reference samples or by a conditional score alongside an unconditional model. Existing methods inject repulsive terms derived from local information at each sample, so the resulting sampler does not target an explicit density, and its bias cannot be systematically reduced by increasing computation. We instead present a negative-guidance method that samples from an explicit target density, which uniformly handles both reference-sample and score-based specifications and suppresses probability mass near the undesired condition with an avoidance strength that parametrizes the target density itself. Realizing this target requires estimating, at each diffusion timestep, how likely each sample is to belong to the undesired condition. We cast this estimation as a Positive-Unlabeled learning problem that admits lightweight online trainin
    
[^26]: SNACK：一种用于概率图生成的序列表示框架

    SNACK: A Sequential Notation Framework for Probabilistic Graph Generation

    [https://arxiv.org/abs/2610.19781](https://arxiv.org/abs/2610.19781)

    SNACK提出了一种将图线性化为字符串序列的表示框架，通过序列与有序邻接矩阵的双射对应和无效logit屏蔽机制，使序列模型能够以100%有效率生成图，并在保持有竞争力样本质量的同时显著提升GFlowNet训练吞吐量。

    

    我们提出了SNACK，这是一种用于概率图生成的字符串表示方法，使序列模型能够生成并建模具有离散节点和边特征的图。SNACK通过编码下三角邻接矩阵的非零元素来实现图的线性化，在有效序列与有序邻接矩阵之间建立了双射对应关系。这种设计使得无效logit屏蔽成为可能，从而能够强制执行结构约束（如化学价规则和排序规则），保证生成的序列100%有效。SNACK还通过显式考虑节点排序对称性，实现了基于字符串的原则性GFlowNet训练。在标准图生成任务和分子分布学习基准测试中，SNACK取得了具有竞争力的样本质量，同时相对于基于图的生成方法大幅提升了GFlowNet的训练吞吐量。探测分析进一步表明，在SNACK上训练的模型编码了（原文摘要在此处被截断）

    arXiv:2610.19781v1 Announce Type: cross  Abstract: We introduce SNACK, a string representation for probabilistic graph generation that enables sequence models to generate and model graphs with discrete node and edge features. SNACK linearizes graphs by encoding the nonzero entries of the lower-triangular adjacency matrix, establishing a bijective correspondence between valid sequences and ordered adjacency matrices. This design enables invalid logit masking to enforce structural constraints--such as chemical valence and ordering rules--guaranteeing 100% valid generated sequences. SNACK also enables principled string-based GFlowNet training by explicitly accounting for node-ordering symmetries. Across standard graph-generation tasks and molecular distribution-learning benchmarks, SNACK achieves competitive sample quality, while substantially improving the throughput of GFlowNet training relative to graph-based generation. Probing analyses further show that models trained on SNACK encode
    
[^27]: 从全局收敛到尾部收敛：α-稳定基于分数的生成模型的动力学

    From Global to Tail Convergence: Dynamics of $\alpha$-Stable Score-Based Generative Models

    [https://arxiv.org/abs/2610.19773](https://arxiv.org/abs/2610.19773)

    本文对α-稳定基于分数的生成模型建立了非渐近收敛性分析，证明在全局Wasserstein距离下α-稳定动力学与布朗扩散模型表现相当，但两者在尾部收敛行为上存在本质区别。

    

    由α-稳定Lévy过程驱动的基于分数的生成模型已被提出作为布朗扩散模型的替代方案，经验证据表明其在采样方面有所改进，并能更好地建模重尾数据。然而，这些模型的理论保证仍然有限。我们对保持尺度的α-稳定生成动力学建立了非渐近收敛性分析，涵盖紧支撑目标和幂律目标。在p-Wasserstein距离下，我们量化了早停、初始化、分数逼近和时间离散化的影响，并证明初始化误差呈指数衰减。与布朗动力学的比较表明，仅凭全局Wasserstein收敛并不能从根本上区分这两种模型：在共同的重尾目标类别上，两种模型的初始化误差都随正向时间范围呈指数衰减。而一个鲜明的分离出现在尾部。

    arXiv:2610.19773v1 Announce Type: new  Abstract: Score-based generative models driven by $\alpha$-stable L{\'e}vy processes have been proposed as an alternative to Brownian diffusion models, with empirical evidence suggesting improved sampling and better modeling of heavy-tailed data. Yet theoretical guarantees for these models remain limited. We develop a non-asymptotic convergence analysis for scale-preserving $\alpha$-stable generative dynamics, covering both compactly supported and power-law targets. In p-Wasserstein distance, we quantify the effects of early stopping, initialization, score approximation, and time discretization, and show exponential decay of the initialization error. A comparison with Brownian dynamics reveals that global Wasserstein convergence alone does not fundamentally distinguish the two models: both initialization errors decay exponentially with the forward horizon on the common class of heavy-tailed targets. A sharp separation emerges instead in the tails.
    
[^28]: 基于高斯替代反向动力学的退火Lévy扩散

    Annealed L\'{e}vy Diffusion via Gaussian Surrogate Reverse Dynamics

    [https://arxiv.org/abs/2610.19692](https://arxiv.org/abs/2610.19692)

    提出退火Lévy扩散（ALD），通过随时间退火的稳定性指数将重尾探索与高斯精炼结合，并把稳定噪声表示为高斯尺度混合，从而在标准Gaussian噪声预测训练中仅需增加一个标量尺度输入即可实现。

    

    高斯扩散模型为生成建模提供了一个实用的框架。去噪Lévy概率模型（DLPM）在整个去噪过程中引入尾部指数固定不变的重尾扰动。我们提出了退火Lévy扩散（ALD）。ALD通过随时间变化的稳定性指数 $\alpha(t)$，将重尾探索与高斯精炼相结合。我们的“边缘优先”表述将稳定噪声表示为标量高斯尺度混合。这一表述使得标准的Gaussian噪声预测训练只需增加一个额外的尺度输入即可实现。我们研究了两种采样过程：ALD-renoise估计干净样本并注入新的边缘噪声；ALD-posterior则通过累积的条件方差和高斯桥更新来保持相邻状态之间的依赖关系。两种采样器都复用同一个训练好的预测器，仅使用不同的标量输入。我们在四个二维基准和一个不平衡的九模态混合分布上对ALD进行了评估。

    arXiv:2610.19692v1 Announce Type: new  Abstract: Gaussian diffusion models offer a practical framework for generative modeling. Denoising L\'{e}vy Probabilistic Models (DLPM) introduce heavy-tailed perturbations with a fixed tail index throughout denoising. We introduce Annealed L\'{e}vy Diffusion (ALD). ALD varies the stability index $\alpha(t)$ to combine heavy-tailed exploration with Gaussian refinement. Our marginal-first formulation represents stable noise as a scalar Gaussian scale mixture. This formulation enables standard Gaussian noise-prediction training with one additional scale input. We study two sampling processes. ALD-renoise estimates the clean sample and injects fresh marginal noise. ALD-posterior preserves dependence between successive states through an accumulated conditional variance and a Gaussian bridge update. Both samplers reuse the same trained predictor with different scalar inputs. We evaluate ALD on four 2D benchmarks and an imbalanced nine-mode mixture. We 
    
[^29]: 面向策略优化的函数空间时间自蒸馏

    Function-Space Temporal Self-Distillation for Policy Optimization

    [https://arxiv.org/abs/2610.19687](https://arxiv.org/abs/2610.19687)

    本文提出单策略的函数空间策略优化框架FSPO，以稀疏的历史下一词元分布作为固定时间目标进行自蒸馏，无需维护持久教师模型即可提供稳定密集的词元级监督，并通过特权分歧校正与可靠性自适应锚定提升策略优化效果。

    

    可验证奖励强化学习（RLVR）提升了语言模型的推理能力，但标量结果只能为响应轨迹提供粗略的监督信号。自蒸馏通过将无辅助的策略与基于反馈信息的自教师进行比较，能够提供更密集的词元级指导；然而教师会随学生一同演化，导致监督目标不稳定，因而需要额外的正则化。我们提出了函数空间策略优化（FSPO），这是一个单策略框架，将当前的特权适应与函数空间时间整合相结合。FSPO将稀疏的历史下一词元分布存储为固定的时间目标，在无需维护持久教师模型的情况下实现时间上的分离。该框架进一步引入特权分歧校正以显式抑制局部被高估的动作，并引入可靠性自适应锚定，在自生成证据重复时强化参考监督（摘要原文在此处截断）。

    arXiv:2610.19687v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) improves reasoning in language models, but scalar outcomes provide coarse supervision over response trajectories. Self-distillation offers denser token-level guidance by comparing an unassisted policy with a feedback-informed self-teacher, yet the teacher evolves with the student, creating unstable targets and motivating additional regularization. We introduce Function-Space Policy Optimization (FSPO), a single-policy framework that combines current privileged adaptation with function-space temporal consolidation. FSPO stores sparse historical next-token distributions as fixed temporal targets, providing temporal separation without maintaining a persistent teacher model. It further introduces privileged disagreement correction to explicitly suppress locally overvalued actions, reliability-adaptive grounding that strengthens reference supervision when self-generated evidence repeated
    
[^30]: 关于零阶非凸优化中的全局联合下界

    On Global Joint Lower Bounds in Zeroth-Order Nonconvex Optimization

    [https://arxiv.org/abs/2610.19652](https://arxiv.org/abs/2610.19652)

    本文在无限制全空间设置下为光滑随机非凸优化建立了标量预言机全局联合下界，回答了零阶方法复杂度关于维度、光滑化参数和精度的依赖是否联合最优这一公开问题。

    

    近期的研究工作试图通过梯度估计器将一阶非凸优化的保证转移到零阶方法上，这通常会产生与维度 $d$ 呈线性关系的因子。Kornowski 等人（2024）针对非光滑非凸优化获得了 $\mathcal{O}(d\delta^{-1}\epsilon^{-3})$ 的预言机复杂度，并提出了其关于这三个参数的依赖关系是否联合最优的问题。尽管关于维度的依赖性以及对 $(\delta,\epsilon)$ 的一阶依赖性已被分别证明是不可避免的，但与之匹配的零阶下界一直悬而未决。我们在无限制的全空间设置中回答了这个问题。对于光滑随机非凸优化，在假设种群目标函数光滑且全局下有界、以及样本 Lipschitz 常数满足二阶矩有界的条件下，我们建立了标量预言机下界 $\Omega(dL_0^2\min\{\Delta H,L_0^2\}/\epsilon^4)$。随后通过一个光滑化归，进一步得到了一个下界……（摘要在此处被截断）

    arXiv:2610.19652v1 Announce Type: new  Abstract: Recent work has sought to transfer first-order nonconvex optimization guarantees to zeroth-order methods through gradient estimators, typically incurring a factor linear in $d$. \citet{kornowski2024} obtained an $\mathcal{O}(d\delta^{-1}\epsilon^{-3})$ oracle complexity for nonsmooth nonconvex optimization and asked whether its dependence on all three parameters is jointly optimal. Although the dimension dependence and the first-order dependence on $(\delta,\epsilon)$ are separately known to be unavoidable, a matching zeroth-order lower bound has remained open. We answer this question in the unrestricted full-space setting. For smooth stochastic nonconvex optimization, we establish a scalar-oracle lower bound of $\Omega\!\left(dL_0^2\min\{\Delta H,L_0^2\}/\epsilon^4\right)$ assuming a smooth, globally lower-bounded population objective and a second-moment bound on sample Lipschitz constants. A smooth reduction then yields a lower bound f
    
[^31]: 面向机器学习原子间势的风险控制DFT选择方法

    Risk-Controlled DFT Selection for Machine-Learned Interatomic Potentials

    [https://arxiv.org/abs/2610.19630](https://arxiv.org/abs/2610.19630)

    本文提出一种基于保形风险控制（CRC）的框架，利用轻量级预测器估计MLIP与DFT之间的不一致分数并校准为选择阈值，在保证MLIP误差可控的前提下自动决定哪些构型需要昂贵的DFT计算，从而突破自动化材料发现的通量瓶颈。

    

    自动化材料发现需要评估大量候选构型，而密度泛函理论（DFT）计算是主要的通量瓶颈。预训练的机器学习原子间势（MLIPs）能够以低成本近似DFT，但其误差在不同构型之间可能相差数个数量级，这使得难以决定何时应该信任MLIP的预测、何时应回退到DFT计算。我们将这一决策问题构建为风险控制选择问题，并在保形风险控制（CRC）框架内进行求解。该方法将每个构型从原子序数和笛卡尔坐标映射为固定的几何描述符，轻量级预测器利用该描述符估计能量、平均力和最大力方面的MLIP-DFT不一致分数。随后，CRC将该分数校准为DFT标记的选择阈值，使得未被选中的构型（保留其MLIP标签）上的MLIP误差保持在用户指定的水平以下，从而在保证预测质量的同时大幅减少所需的DFT计算量。

    arXiv:2610.19630v1 Announce Type: cross  Abstract: Automated materials discovery requires evaluating large pools of candidate configurations, with density functional theory (DFT) as a major throughput bottleneck. Pretrained machine-learned interatomic potentials (MLIPs) approximate DFT cheaply, but their errors vary by orders of magnitude across configurations, making it difficult to decide when to trust the MLIP predictions and when to fall back to DFT. We frame this decision as a risk-controlled selection problem and solve it within a conformal risk control (CRC) framework. Each configuration is mapped from atomic numbers and Cartesian coordinates to a fixed geometric descriptor, which a lightweight predictor uses to estimate an MLIP-DFT disagreement score in energy, average force, and maximum force. CRC then calibrates this score into a selection threshold for DFT labeling such that the MLIP error on the unselected configurations, which keep their MLIP labels, stays below a user-cho
    
[^32]: 用于Hawkes过程背景强度估计的投影Wasserstein-Fisher-Rao流

    Projected Wasserstein-Fisher-Rao Flows for Hawkes Background-Intensity Estimation

    [https://arxiv.org/abs/2610.19588](https://arxiv.org/abs/2610.19588)

    本文提出一种基于Wasserstein-Fisher-Rao几何的似然变分方法，通过向统计流形投影的分布框架来估计Hawkes过程的背景强度，并证明了相应估计量的条件一致性定理。

    

    我们考虑在触发机制已知的情况下估计Hawkes过程背景强度的问题，将背景视为一个具有正则密度的未知有限正测度。我们提出了一种基于Wasserstein-Fisher-Rao几何的似然变分方法。在这一设定下，相关的一阶变分天然地是分布取值的，因此超出了经典梯度流表述的范畴。我们为由此产生的动力学建立了一个分布框架，并通过向统计流形的投影赋予其有限维意义。我们将真实背景识别为极限广义梯度的唯一零点，并在适当的稳定性、紧性、可积性和逼近条件下，为在概率趋于1的事件上满足经验投影方程的可测估计量建立了条件一致性定理。

    arXiv:2610.19588v1 Announce Type: cross  Abstract: We consider the problem of estimating the background intensity of a Hawkes process when the triggering mechanism is known, treating the background as an unknown finite positive measure with a regular density. We propose a likelihood-based variational approach based on the Wasserstein-Fisher-Rao geometry. In this setting, the relevant first variations are naturally distribution-valued and therefore fall outside the classical gradient-flow formulation. We develop a distributional framework for the resulting dynamics and give them a finite-dimensional meaning through projections onto statistical manifolds. We identify the true background as the unique zero of a limiting generalized gradient and, under suitable stability, compactness, integrability and approximation conditions, establish a conditional consistency theorem for measurable estimators satisfying the empirical projected equations on events whose probabilities tend to one. Finall
    
[^33]: 论后验采样的固有隐私性：一个统一的Rényi散度框架

    On Inherent Privacy of Posterior Sampling: A Unified R\'enyi-Divergence Framework

    [https://arxiv.org/abs/2610.19573](https://arxiv.org/abs/2610.19573)

    本文提出统一的Rényi散度框架，证明从贝叶斯后验分布中采样（OPS）天然获得“免费”的差分隐私保证，其隐私损失由单条记录对数似然比的后验指数矩刻画，并在多种尾部条件与代表性贝叶斯模型下给出显式隐私保证。

    

    差分隐私（DP）为量化隐私保证提供了一个严格的框架。虽然大多数方法通过注入经过校准的随机噪声来实现差分隐私，但某些过程（如采样）的内在随机性可以“免费”带来差分隐私保证。在这项工作中，我们开发了一个统一的Rényi散度框架，用于刻画从贝叶斯后验分布中采样的固有差分隐私保证。我们证明，单次后验采样（One-Posterior-Sampling, OPS）所导致的隐私损失由单条记录对数似然比的后验指数矩决定，并在一致有界、次高斯或次指数尾部的情形下建立了显式的差分隐私保证。我们将该框架应用于代表性的贝叶斯模型，包括带任意先验的类别似然模型、高斯-高斯共轭对、来自指数族的广义线性模型以及高斯似然的线性回归等。

    arXiv:2610.19573v1 Announce Type: cross  Abstract: Differential Privacy (DP) provides a rigorous framework for quantifying privacy guarantees. While most methods achieve DP by injecting calibrated random noise, the intrinsic randomness of certain procedures such as sampling can yield DP guarantees ``for free''. In this work, we develop a unified R\'enyi divergence framework to characterize the inherent DP guarantees of sampling from a Bayesian posterior distribution. We show that the privacy loss incurred by One-Posterior-Sampling (OPS) is governed by posterior exponential moments of the single-record log-likelihood ratio, and establish explicit DP guarantees under uniformly bounded, sub-Gaussian, or sub-exponential tail regimes. We apply the framework to representative Bayesian models, including categorical likelihoods with arbitrary priors, Gaussian-Gaussian conjugate pair, generalized linear models from the exponential family and linear regression with Gaussian likelihood, both with
    
[^34]: 基于多个不可靠标签的噪声标签校准

    Noisy-Label Calibration with Multiple Unreliable Labels

    [https://arxiv.org/abs/2610.19543](https://arxiv.org/abs/2610.19543)

    该论文提出了噪声校正ECE（NC-ECE）估计器、噪声概率估计算法以及事后重校准方法，使得在仅有多个不可靠标注的情况下，也能对二元分类器的校准性能进行无偏评估和可靠重校准。

    

    在医学标注和众包场景中，多个标注者会为每个样本分配标签，但这些标签可能是错误的。将此类噪声标签视为真实标签会使校准评估产生偏差。为解决这一问题，我们提出了在潜在干净标签分布下，仅利用噪声标注来评估和重新校准固定二元预测器的方法。首先，在给定各标注者噪声概率的条件下，我们提出了噪声校正ECE（NC-ECE），即类条件噪声下期望校准误差（ECE）的分箱估计器，并推导了该估计器偏差的非渐近界。随后，我们在条件独立性和标签方向性假设下，提出了估计未知噪声概率的实用算法。我们还开发了基于损失和基于直方图的事后重校准方法，分别具有干净风险保证和校准保证。在半合成基准上的实验表明，所提出的方法……

    arXiv:2610.19543v1 Announce Type: cross  Abstract: In medical annotation and crowdsourcing, multiple annotators assign labels to each example, but these labels may be incorrect. Treating such noisy labels as ground truth can bias calibration evaluation. To address this problem, we propose methods for evaluating and recalibrating a fixed binary predictor under the underlying clean-label distribution, using only noisy annotations. First, given annotator-specific noise probabilities, we propose Noise Corrected ECE (NC-ECE), a binned estimator of expected calibration error (ECE) under class-conditional noise. We derive non-asymptotic bounds on the estimator's bias. Then, we propose practical algorithms for estimating unknown noise probabilities under conditional independence and label-orientation assumptions. We also develop loss-based and histogram post-hoc recalibration with clean-risk and calibration guarantees, respectively. Experiments on semi-synthetic benchmarks show that the propos
    
[^35]: Feel-Good Thompson 采样的高概率遗憾界

    High-Probability Regret Bounds for Feel-Good Thompson Sampling

    [https://arxiv.org/abs/2610.19513](https://arxiv.org/abs/2610.19513)

    本文首次为 Feel-Good Thompson 采样在上下文老虎机中建立了高概率遗憾界，在线性设置下达到了与最优算法在维度和时间范围依赖性上相匹配的遗憾保证。

    

    Feel-Good Thompson 采样（FGTS）通过在标准 Thompson 采样的后验中加入一个探索项来增强，该探索项倾向于偏向预测最优值较高的奖励函数。这一改进在线性上下文老虎机中实现了除对数因子外极小极大最优的遗憾，但现有关于其两种标准变体的分析仅建立了期望意义下的遗憾保证。在本文中，我们在具有一般可实现函数逼近的上下文老虎机中，首次为累积型（cumulative）和当前步型（current-step）两种 FGTS 推导出了高概率遗憾界。我们的界取决于相应的解耦系数以及真实奖励函数附近的先验概率质量。在 d 维线性设置下，两种变体都以至少 1-δ 的概率满足 Regret(T)=O(d√T log(2T)+√T log(1/δ))，其中 T 为时间范围。该界在维度和时间范围依赖性上与基于……的算法相匹配。

    arXiv:2610.19513v1 Announce Type: cross  Abstract: Feel-good Thompson sampling (FGTS) augments the posterior of vanilla Thompson sampling with an exploration term that favors reward functions with high predicted optimal values. This modification achieves minimax-optimal regret up to logarithmic factors in linear contextual bandits, but existing analyses of its two standard variants establish regret guarantees in expectation. In this paper, we derive the first high-probability regret bounds for both cumulative and current-step FGTS in contextual bandits with general realizable function approximation. Our bounds depend on the corresponding decoupling coefficients and the prior mass near the true reward function. In the $d$-dimensional linear setting, both variants satisfy $\mathrm{Regret}(T)=O\big(d\sqrt T\log(2T)+\sqrt T\log(1/\delta)\big)$ with probability at least $1-\delta$, where $T$ is the time horizon. This bound matches the dimension and horizon dependence of algorithms based on 
    
[^36]: 状态匹配序列保形预测

    State-Matched Sequence Conformal Prediction

    [https://arxiv.org/abs/2610.19484](https://arxiv.org/abs/2610.19484)

    提出了状态匹配序列保形预测（SMSCP），通过构建局部线性状态并利用状态相似性选取匹配的历史校准分数参考集，解决了可比状态在非相邻时间重现时基于近期性的传统方法失效的问题。

    

    序列保形预测需要决定哪些过去的误差应当用于校准当前目标的不确定性。当近期观测仍然具有相关性时，近期性原则和在线误差反馈是有用的；但当可比状态在非相邻时间点重现时，最近的过去可能是一个糟糕的参考，且这种重现模式通常是未知的。我们提出了状态匹配序列保形预测（SMSCP），该方法从目标前信息中构建显式的单侧局部线性状态，利用状态相似性选择过去校准分数的匹配参考集，然后应用普通的未加权保形次序统计量。与现有方法不同，SMSCP利用状态相似性来识别与目标分数分位数相关的校准观测。这促使我们对时间依赖性下邻近状态是否具有可比的目标分数分位数进行理论分析。我们建立了对（摘要在此处被截断）的控制。

    arXiv:2610.19484v1 Announce Type: cross  Abstract: Sequential conformal prediction needs to decide which past errors should calibrate uncertainty for the current target. Recency and online error feedback are useful when recent observations remain relevant, but the recent past can be a poor reference when comparable states recur at nonadjacent times, with the recurrence pattern generally unknown. We propose State-Matched Sequence Conformal Prediction (SMSCP), which constructs an explicit one-sided local-linear state from pre-target information, uses state similarity to select a matched reference set of past calibration scores, and then applies the ordinary unweighted conformal order statistic. In contrast to existing approaches, SMSCP uses state similarity to identify calibration observations relevant to the target score quantile. This motivates a theoretical analysis of whether nearby states have comparable target score quantiles under temporal dependence. We establish control of the r
    
[^37]: 随机重排随机梯度下降的大步长末次迭代收敛性

    Large-Step Last-Iterate Convergence of Random-Reshuffling Stochastic Gradient Descent

    [https://arxiv.org/abs/2610.19444](https://arxiv.org/abs/2610.19444)

    本文为随机重排随机梯度下降（RR-SGD）的末次迭代点建立了大步长收敛保证，通过保留随机比较点的采样偏差，从步长限制中移除了显式的 1/n 因子，使末次迭代的收敛速率在对数因子内匹配平均迭代点的收敛速率。

    

    随机重排随机梯度下降（RR-SGD）在机器学习中广泛应用于有限和优化问题。对于凸的 L-光滑分量函数，近期的工作为平均迭代点建立了收敛性保证，其分量步长可达到 1/L 的量级。然而，末次迭代点是实践中自然的输出，而现有在此设定下的末次迭代收敛保证要求步长与分量数 n 成反比。我们在无约束问题上为标准 RR-SGD 的末次迭代点建立了期望目标函数间隙界，所用常数分量步长满足 0<η≲1/(L(1+log(nK)))，其中 K 为训练轮数。通过在随机比较点保留采样偏差，我们从步长限制中移除了显式的 1/n 因子。结合现有的小步长界，我们的结果在对数因子内匹配了相应的平均迭代点收敛速率。

    arXiv:2610.19444v1 Announce Type: cross  Abstract: Random reshuffling stochastic gradient descent (RR-SGD) is widely used for finite-sum optimization in machine learning. For convex $L$-smooth components, recent work establishes convergence guarantees for an averaged iterate with component stepsizes as large as order $1/L$. However, the final iterate is a natural output in practice, and existing last-iterate guarantees in this setting require stepsizes that scale inversely with the number of components $n$. We establish an expected objective-gap bound for the last iterate of standard RR-SGD on unconstrained problems with a constant component stepsize satisfying $0<\eta\lesssim 1/(L(1 + \log(nK)))$, where $K$ is the number of epochs. By retaining the sampling bias at random comparison points, we remove the explicit $1/n$ factor from the stepsize restriction. Combined with existing small-step bounds, our result matches the corresponding averaged-iterate convergence rate up to logarithmic
    
[^38]: 多敏感属性分布公平性：联合Wasserstein审计与Shapley归因分配

    Multi-Sensitive Distributional Fairness: Joint Wasserstein Audits and Shapley Allocation

    [https://arxiv.org/abs/2610.19252](https://arxiv.org/abs/2610.19252)

    该论文提出用一维Wasserstein几何对多个敏感属性进行联合公平性审计，将联合差异刻画为全单元条件分位数过程的方差，并借助Shapley值实现与顺序无关的差异归因，从而捕捉单坐标审计会遗漏的交叉性歧视。

    

    针对多个敏感属性的公平性审计通常一次只针对单个属性坐标进行。这可能遗漏纯粹的交叉性差异：一个评分可能与每个敏感坐标各自独立，却依赖于它们的联合特征。我们利用一维Wasserstein几何，研究实值评分的分布性人口统计均等。我们区分了两种互补的构造。条件审计和顺序审计通过其零点集刻画联合均等，但它们的正值通常无法对固定的联合Wasserstein差异进行分解。相比之下，全单元条件分位数函数定义了一个取值于希尔伯特空间的随机变量，其方差即为联合差异。该分位数过程的条件期望诱导出一个单调信息博弈：任意属性揭示顺序都能给出同一联合差异的精确鞅分解，而Shapley值则提供了一种跨敏感属性、与顺序无关的归因分配方法。

    arXiv:2610.19252v1 Announce Type: cross  Abstract: Fairness audits with several sensitive attributes are often performed one coordinate at a time. This can miss purely intersectional disparities: a score may be independent of each sensitive coordinate while depending on their joint profile. We study distributional demographic parity for a real-valued score using one-dimensional Wasserstein geometry. We distinguish two complementary constructions. Conditional and sequential audits characterize joint parity through their zero sets, but their positive values do not generally decompose a fixed joint Wasserstein disparity. By contrast, full-cell conditional quantile functions define a Hilbert-valued random variable whose variance is the joint disparity. Conditional expectations of this quantile process induce a monotone information game: every reveal order yields an exact martingale decomposition of the same joint disparity, while Shapley values provide an order-free attribution across sens
    
[^39]: 将因果发现中基于完整数据的可辨识性结果推广至缺失数据场景

    Generalizing Identifiability Results in Causal Discovery from Complete Data to Missing Data Settings

    [https://arxiv.org/abs/2610.19211](https://arxiv.org/abs/2610.19211)

    该论文建立了在包括非随机缺失在内的多种缺失机制下因果图保持可辨识的一般性充分条件，提供了将完整数据因果发现算法的正确性保证扩展到缺失数据场景的框架，并针对加性噪声模型提出了两个渐近正确的算法。

    

    虽然针对完整数据的因果发现方法已经成熟，但能够处理缺失值的此类算法仍然有限。我们建立了一般性的充分条件，在这些条件下，因果图在各种缺失类别（包括非随机缺失，Missing Not At Random）下仍然是可辨识的（即可以从可用数据分布中唯一确定），并提供了一个框架，用于将完整数据因果发现算法的正确性保证扩展到缺失数据场景。我们的结果表明，当满足以下条件时，无未观测混杂因素的因果图的可辨识性得以保持：(1) 各缺失机制之间不存在因果交互作用；(2) 没有变量直接导致其自身的缺失；(3) 缺失机制满足忠实性条件，或遵循广义线性模型。基于这些理论结果，我们提出了两个针对加性噪声模型的渐近正确算法，用于恢复潜在的因果……

    arXiv:2610.19211v1 Announce Type: cross  Abstract: While causal discovery methods have matured for complete data, such algorithms for handling missing values remain limited. We establish general sufficient conditions under which causal graphs remain identifiable (i.e., uniquely determined from the available data distribution) under various missingness categories, including Missing Not At Random, and provide a framework to extend correctness guarantees from causal discovery algorithms for complete data to missing data settings. Our results show that identifiability of causal graphs with no unobserved confounders is preserved when (1) missingness mechanisms do not causally interact, (2) no variable directly causes its own missingness, and (3) either faithfulness holds for missingness mechanisms, or they follow generalized linear models. Guided by these theoretical results, we introduce two asymptotically correct algorithms for Additive Noise Models (ANMs) that recover the underlying caus
    
[^40]: 嵌入几何形态决定大语言模型路由：追踪失效模式并提升泛化能力

    Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization

    [https://arxiv.org/abs/2610.19198](https://arxiv.org/abs/2610.19198)

    该论文揭示了语义嵌入的几何结构是LLM路由在未见任务上失效的根源，并提出用Qwen2.5-7B-Instruct深层最后token的隐藏状态替代语义嵌入作为表示，从而显著提升路由方法对未见任务的泛化能力。

    

    随着可用的大语言模型（LLM）数量不断增长，如何为给定提示选择合适的模型已成为一个关键问题。路由领域的许多研究依赖于将提示语义嵌入到向量空间中，然后对嵌入表示应用轻量级分类器。这些方法在分布内表现稳健，但难以泛化到未见过的任务。通过研究语义嵌入的几何结构，我们追踪到一个明确的失效模式：决定未见提示邻居的特征是LLM选择的不可靠代理。因此，我们用以下映射替换语义嵌入：对提示运行Qwen2.5-7B-Instruct的部分前向传播，并使用其在深层的最后一个token的隐藏状态作为表示。当对这种新表示应用kNN和其他简单分类器时，我们在泛化方面获得了明确的结果：在平均各种成本惩罚下，它们的性能……

    arXiv:2610.19198v1 Announce Type: cross  Abstract: With the growing number of LLMs available, selecting which one to use for a prompt has become a key question. A large part of the routing literature relies on semantically embedding prompts into a vector space and then applying a lightweight classifier to the embedded representations. These methods perform robustly in-distribution but they struggle to generalize to unseen tasks. By studying the geometry of semantic embeddings, we trace a clear failure mode: features that determine the neighbors of unseen prompts are an unreliable proxy for LLM selection. We therefore replace the semantic embedding with the following map: we run a partial forward pass of Qwen2.5-7B-Instruct on the prompt and use the hidden state of its last token at a deep layer as the representation. When applying kNN and other simple classifiers to this new representation, we obtain a clear result in generalization: on average across cost penalties, their performance 
    
[^41]: 基于扩散正则化Stein变分梯度下降的结构感知贝叶斯全波形反演

    Structure-aware Bayesian full waveform inversion via diffusion-regularized Stein variational gradient descent

    [https://arxiv.org/abs/2610.19187](https://arxiv.org/abs/2610.19187)

    该论文提出将学习得到的扩散模型作为隐式正则化机制嵌入基于退火Stein变分梯度下降的贝叶斯全波形反演框架，以刻画复杂地质结构先验并增强反演的不确定性量化能力。

    

    全波形反演（FWI）是高分辨率地下模型重建的强大工具，但它仍具有高度非线性和不适定性，尤其是在初始模型不准确或采集数据含噪且不完整的情况下。退火Stein变分梯度下降（ASVGD）通过用一组速度模型集合来近似后验分布，为FWI提供了一种基于粒子的贝叶斯框架，从而能够在单一确定性解之外实现不确定性量化。然而，传统的ASVGD-FWI通常依赖于简单的手工先验，如高斯或平滑假设，这往往不足以刻画复杂的地质结构。为解决这一局限，我们在ASVGD-FWI框架中引入训练好的扩散模型（DM）作为隐式正则化机制。训练好的扩散模型无需显式计算扩散先验密度，而是作为……（摘要在此处截断）

    arXiv:2610.19187v1 Announce Type: cross  Abstract: Full waveform inversion (FWI) is a powerful tool for high-resolution subsurface model reconstruction, but it remains highly nonlinear and ill-posed, especially when the initial model is inaccurate or the acquired data are noisy and incomplete. Annealed Stein variational gradient descent (ASVGD) provides a particle-based Bayesian framework for FWI by approximating the posterior distribution with an ensemble of velocity models, thereby enabling uncertainty quantification beyond a single deterministic solution. However, conventional ASVGD-FWI commonly relies on simple handcrafted priors, such as Gaussian or smoothing assumptions, which are often insufficient to represent complex geological structures. To address this limitation, we introduce a learned diffusion model (DM) as an implicit regularization mechanism within the ASVGD-FWI framework. Instead of explicitly evaluating the diffusion prior density, the trained diffusion model acts as
    
[^42]: 学生Copula马尔可夫网络

    Student Copula Markov Networks

    [https://arxiv.org/abs/2610.19135](https://arxiv.org/abs/2610.19135)

    本文基于copula密度刻画成对马尔可夫性质，提出学生t-Copula马尔可夫网络及相应的ECM算法，将概率图模型扩展至非高斯数据并实现参数估计与图结构选择。

    

    本文将概率图模型的应用范围扩展到了非高斯数据。该文建立了无向概率图模型中成对马尔可夫性质基于copula密度的刻画方式。基于这一刻画，我们推导出了学生t-Copula马尔可夫网络中成对马尔可夫性质的必要条件。文中提供了一种期望-条件最大化算法（ECM），用于实现带惩罚项的极大似然估计和图结构选择。通过模拟研究对该算法的性能进行了评估，结果表明该算法在估计模型参数和恢复底层图结构方面具有良好效果。

    arXiv:2610.19135v1 Announce Type: cross  Abstract: This paper extends the applicability of probabilistic graphical models to non Gaussian data. It establishes a characterization of the pairwise Markov property in undirected probabilistic graphical models in terms of copula densities. Building on this characterization, we derive a necessary condition for the pairwise Markov property in Student copula Markov networks. An Expectation-Conditional Maximization algorithm (ECM) is provided to facilitate the penalized maximum likelihood estimation and graph selection. The performance of the proposed algorithm is evaluated through simulation studies, which demonstrate its effectiveness in estimating model parameters and recovering the underlying graph structure.
    
[^43]: 用于学习多治疗方案个体化治疗规则的神经监督聚类方法

    Neural Supervised Clustering for Learning Individualized Treatment Rules with Many Treatments

    [https://arxiv.org/abs/2610.19104](https://arxiv.org/abs/2610.19104)

    提出了一种结合共享非线性协变量表示、低维治疗嵌入与成对聚类融合的神经Q学习框架NSCF，能够从每个治疗臂有限的观测数据中学习多治疗方案下的个体化治疗规则，并生成可解释的治疗分组划分。

    

    在存在多种治疗方案的情况下学习个体化治疗规则（ITR），需要从每个治疗臂有限的观测数据中估计治疗特异性效应。具有相似条件响应的治疗方案本可以共享信息，但线性对比模型可能无法捕捉非线性的异质性。我们提出了非线性监督聚类融合方法（NSCF），这是一个神经Q学习框架，包含共享的非线性协变量表示和低维治疗嵌入。通过成对融合，该方法将学习到的对比效应相似的治疗方案归为一组，从而得到一个个体化治疗规则和一个可解释的划分结构。我们设计了一种交替算法，将神经网络训练、嵌入融合、基于验证集的划分选择以及无惩罚的分组重新拟合相结合。在非线性精确共享的模拟实验中，NSCF相较于强大的神经基准方法降低了遗憾值；扰动分析刻画了硬共享机制的适用边界。在黑色素瘤异种移植应用中，NSCF提高了平均策略价值……

    arXiv:2610.19104v1 Announce Type: cross  Abstract: Learning an individualized treatment rule (ITR) with many treatments requires estimating treatment-specific effects from limited observations per arm. Treatments with similar conditional responses can share information, but linear contrast models may miss nonlinear heterogeneity. We propose nonlinear supervised clustering fusion (NSCF), a neural Q-learning framework with a shared nonlinear covariate representation and low-dimensional treatment embeddings. Pairwise fusion groups treatments with similar learned contrasts, yielding an ITR and an interpretable partition. An alternating algorithm combines neural training, embedding fusion, validation-based partition selection, and unpenalized grouped refitting. NSCF lowers regret relative to a strong neural reference in nonlinear exact-sharing simulations; perturbations characterize the limits of hard sharing. In a melanoma xenograft application, NSCF improves average policy value over the 
    
[^44]: 传染病模型参数估计的保形化物理信息弱回归

    Conformalized Physics-Informed Weak Regression for Parameter Estimation in Epidemic Models

    [https://arxiv.org/abs/2610.19058](https://arxiv.org/abs/2610.19058)

    该论文提出物理信息弱回归（PIWR）方法，通过在时间上积分模型而非对数据求导来估计传染病模型参数，并结合由物理模型自身生成校准集的分割保形预测，为所有参数提供具有有限样本覆盖率保证的无分布同步置信区间。

    

    传染病的房室模型在系数上通常是线性的，这些系数是物理参数或它们的乘积或比率。我们利用这一特性，采用闭式最小二乘法进行估计，并为估计结果提供无分布假设的不确定性量化。物理信息回归（PIR）通过将参数拟合到数据的有限差分导数上来实现；我们提出了物理信息弱回归（PIWR），它改为在时间上对模型进行积分，无需对数据求导，可联合估计初始状态，并具有高阶精度。为了量化不确定性，我们将这两种估计器都纳入分割保形预测框架中，其校准群体由基于物理的模型自身生成，从而为所有参数提供具有有限样本覆盖率保证的同步置信区间。一个残差自适应因子（无需真实值即可计算）可将区间缩放至各个数据集（PIWR-RACC）。在十一个疾病信息神经……的基准模型上（原文摘要在此处截断）

    arXiv:2610.19058v1 Announce Type: cross  Abstract: Compartment models of infectious diseases are often linear in coefficients that are physical parameters or products or ratios of them. We exploit this with closed-form least squares and equip the estimates with distribution-free uncertainty. Physics-informed regression (PIR) fits the parameters to finite-difference derivatives of the data; we introduce physics-informed weak regression (PIWR), which integrates the model in time instead, needs no derivatives of the data, estimates the initial state jointly and is high-order accurate. To quantify uncertainty we wrap both estimators in split conformal prediction, with a calibration population generated by the physics-based model itself, giving simultaneous intervals for all parameters with a finite-sample coverage guarantee. A residual-adaptive factor, computed without ground truth, scales the intervals to each data set (PIWR-RACC). On the eleven benchmark models of disease-informed neural
    
[^45]: 主动推理中信息能换来什么

    What Information Buys in Active Inference

    [https://arxiv.org/abs/2610.19046](https://arxiv.org/abs/2610.19046)

    本文将主动推理（AIF）与贝叶斯UCBVI置于相同的MDP状态抽象下进行对比，发现AIF的信息增益衰减过快导致探索不足，进而通过对信息增益进行平方根校准，使校准后的AIF达到了√T阶的后悔界。

    

    信息论式的探索是强化学习（RL）方法学研究与计算认知神经科学的主要关注点，而成熟的强化学习理论则基于置信集来构建后悔保证。这种情况留下了一个悬而未决的问题：在何种条件下，自然界中存在的信息驱动探索比成熟强化学习理论的探索更具优势。我们将这两种范式中最先进的方法——主动推理（AIF）与 Bayes-UCBVI——置于同一起跑线上：二者都通过同一个针对具有 $S$ 个状态的表格型 MDP 的状态抽象来行动，且它们的贪婪形式仅在探索项上有所差别。我们发现 AIF 的信息增益衰减得过快，无法充当置信宽度，这使得依赖它进行探索的贪婪智能体被锁定在无法选择最优动作的状态。通过对信息增益进行平方根校准，我们得到了校准后的 AIF（calibrated AIF），它在精确的情况下达到了 $\sqrt T$ 阶的后悔界。

    arXiv:2610.19046v1 Announce Type: new  Abstract: Information-theoretic exploration is of primary interest in methodological studies of reinforcement learning (RL) and computational cognitive neuroscience, while the established RL theory builds regret guarantees on confidence sets. This situation leaves open the conditions under which the information-driven exploration found in nature is more advantageous than the exploration of the established RL theory. We put the state-of-the-art methods of these two paradigms, Active Inference (AIF) and Bayes-UCBVI, on an equal footing: both act through the same state abstraction of a tabular MDP with $S$ states, and their greedy forms differ only in the exploration term. We find that AIF's information gain decays too fast to serve as a confidence width, locking greedy agents that explore by it out of the optimal action. Calibrating the information gain by a square root yields calibrated AIF, which attains regret of order $\sqrt T$ for an exact abst
    
[^46]: 损失曲率作为扩散模型记忆化的免采样代理指标

    Loss Curvature as a Sampling-Free Proxy for Memorization in Diffusion Models

    [https://arxiv.org/abs/2610.19042](https://arxiv.org/abs/2610.19042)

    该论文提出输入损失曲率（训练损失Hessian矩阵的Frobenius范数）可作为扩散模型记忆化的免采样代理指标，通过在训练中追踪其均值即可预测模型何时开始记忆并复现训练数据。

    

    理解扩散模型为何会记忆其训练数据背后的机制，仍是生成式建模中一个悬而未决的问题。本文从两个视角考察这一问题：(a) 模型层面，即模型在训练过程中何时开始出现记忆化；(b) 样本层面，即单个样本被给定模型记忆的倾向性。我们证明，所谓的输入损失曲率，即扩散模型训练损失Hessian矩阵的Frobenius范数，是量化这两个记忆化视角的有效指标。特别是，我们通过实验证明，在训练过程中持续追踪训练样本与验证样本上平均的输入曲率，可以识别模型何时可能开始复现其训练数据。然而，该指标不足以衡量单个样本被记忆的可能性。为解决……（原文摘要在此处截断）

    arXiv:2610.19042v1 Announce Type: new  Abstract: Understanding the mechanism behind why diffusion models memorize their training data remains an open question in generative modeling. In this paper, we examine this question from two perspectives: (a) the model-level perspective concerning the onset of memorization in a model during training, and (b) the sample-level perspective concerning the proclivity of an individual sample to be memorized by a given model. We show that the so-called input loss curvature, or the Frobenius norm of the Hessian of the diffusion model's training loss, is an effective metric for quantifying both perspectives on memorization. In particular, we empirically demonstrate that input curvatures averaged across training and validation samples can identify when a model is likely to start regurgitating its training data when tracked over training time. However, this metric is not sufficient for measuring an individual sample's likelihood of being memorized. To addr
    
[^47]: 稀疏条件下的离散扩散模型是极小极大最优的

    Discrete Diffusion Models are Minimax Optimal Under Sparsity

    [https://arxiv.org/abs/2610.19020](https://arxiv.org/abs/2610.19020)

    该论文证明了在稀疏性假设下，基于正则化得分熵目标的离散扩散模型能够达到极小极大最优的统计速率，且所用正则化方法无需预知稀疏参数即可实现自适应。

    

    离散扩散模型已成为生成文本及其他类别数据时替代自回归模型的领先方案，但其理论基础的发展仍落后于连续扩散模型。对于连续扩散模型，一条重要的理论路线表明，这些模型能够自适应数据中的低维结构，从而提升统计效率与泛化能力。在本工作中，我们通过研究稀疏性假设下离散扩散模型的统计速率，来探索这一现象在离散情形中的对应。通过分析由正则化得分熵目标导出的非参数估计量，我们在稀疏情形下建立了极小极大最优速率。所提出的正则化方法可以看作是对分类任务中标签平滑技术的一种改进，并且不需要预先知道稀疏性参数。这表明，离散扩散模型……

    arXiv:2610.19020v1 Announce Type: cross  Abstract: Discrete diffusion models have emerged as a leading alternative to autoregressive models for generating text and other categorical data, yet their theoretical foundations remain less developed than those of their continuous counterparts. For continuous diffusion models, a prominent line of theory suggests that these models can adapt to low-dimensional structure in the data, thereby improving statistical efficiency and generalization. In this work, we investigate the discrete analogue of this phenomenon by studying the statistical rates of discrete diffusion models under sparsity assumptions. By analyzing a nonparametric estimator derived from a regularized score-entropy objective, we establish minimax-optimal rates in the sparse regime. The proposed regularization can be viewed as an adaptation of label smoothing techniques introduced for classification, and does not require the knowledge of the sparsity parameter. This shows that disc
    
[^48]: 自适应对手下具身智能与智能体AI统计保障的极限

    Limits of Statistical Assurance for Embodied and Agentic AI under Adaptive Adversaries

    [https://arxiv.org/abs/2610.18948](https://arxiv.org/abs/2610.18948)

    本文证明了对于具身智能和智能体AI，任何有限时长的安全认证在原理上都无法排除那些在认证前伪装正常行为、认证后才激活的隐蔽自适应对手，并精确刻画了此类攻击带来的最坏情形期望损失。

    

    有限的安全评估无法排除那些在认证之前保持未受攻击行为、之后才激活的对手。我们针对具身智能和工具使用型AI刻画了这一局限性。一个在可观测、不可撤销的认证之前完全匹配观测规律、并在之后保留干预能力的对手，在 $n$ 次试验中可达到 $v\mathbb{E}[(n-S_0)_+]$ 的预期认证后事件数，其中 $v$ 是激活事件概率，$S_0$ 是未受攻击的认证时间。该结果覆盖了自适应评估器和私有随机化评估器。在无记忆、事件递增干预且不存在约束性干预预算的情况下，在认证时刻切换对于固定似然比检验而言是精确最优的。当干预在授权部署时间范围内提高事件率时，即使认证被推迟到最后期限，正向的清洁认证要求也会导致线性增长的最坏情况预期超额。

    arXiv:2610.18948v1 Announce Type: cross  Abstract: Finite safety assessments cannot exclude an adversary that preserves unattacked behavior until certification and activates afterward. We characterize this limitation for embodied and tool-use AI. An adversary that matches the complete observation law until observable, irrevocable certification and retains intervention access afterward attains $v\mathbb{E}[(n-S_0)_+]$ expected post-certification events over $n$ trials, where $v$ is the active event probability and $S_0$ the unattacked certification time. The result covers adaptive and privately randomized evaluators. Switching at certification is exactly optimal for a fixed likelihood-ratio test under memoryless, event-increasing interventions without a binding intervention budget. A positive clean-certification requirement forces linear worst-case expected excess when intervention raises event rates over an authorized deployment horizon, even if certification is deferred to the deadlin
    
[^49]: 具有可复用局部后验的半摊销贝叶斯推断

    Semi-Amortized Bayesian Inference with Reusable Local Posteriors

    [https://arxiv.org/abs/2610.18945](https://arxiv.org/abs/2610.18945)

    该论文提出一种半摊销贝叶斯推断方法，将参数分块并为计算昂贵或频繁更新的块训练可复用的神经估计器来近似局部条件后验，再通过马尔可夫链将这些神经采样与常规更新交替进行，从而高效地实现高维复杂模型的贝叶斯推断。

    

    摊销贝叶斯推断利用模拟来执行贝叶斯估计，适用于似然函数计算代价高昂或无法解析获得的情形。然而，在具有复杂依赖关系的高维模型中，学习完整的联合后验分布极具挑战性。我们提出了一种半摊销方法，将参数划分为较小的块，并利用条件独立性来识别推断每个块所需的信息。对于后验更新计算代价高昂或频繁重复的块，我们为其训练神经估计器。每个估计器从相关模型组件的模拟中学习，在给定相关观测值及其所依赖变量取值的情况下，近似该块的完整条件后验分布。随后，马尔可夫链在直接从这些估计器采样与对剩余变量进行常规更新之间交替进行。每次神经更新都使用当前的条件变量取值，并且既不需要似然……

    arXiv:2610.18945v1 Announce Type: cross  Abstract: Amortized Bayesian inference uses simulations to perform Bayesian estimation when likelihoods are expensive or analytically unavailable. However, learning the full joint posterior is challenging in high-dimensional models with complex dependencies. We introduce a semi-amortized method that divides parameters into smaller blocks, and uses conditional independence to identify the information needed to infer each block. We train neural estimators for blocks whose posterior updates are computationally expensive or frequently repeated. Each estimator learns from simulations of the relevant model components to approximate the block's full conditional posterior, given the relevant observations and the values of the variables on which it depends. A Markov chain then alternates direct draws from these estimators with conventional updates of the remaining variables. Each neural update uses the current conditioning values and requires neither lik
    
[^50]: 随机Volterra方程的Wong–Zakai收敛性与傅里叶神经算子学习

    Wong--Zakai Convergence and FNO Learning for Stochastic Volterra Equations

    [https://arxiv.org/abs/2610.18939](https://arxiv.org/abs/2610.18939)

    本文为随机Volterra积分方程建立了Wong–Zakai收敛定理，并在此基础上证明镜像填充傅里叶神经算子能够学习其解算子，实现对完整解轨迹的准确预测以及向更细时间网格的泛化。

    

    本文研究了描述具有记忆效应的随机系统的随机Volterra积分方程（SVIE）的Wong–Zakai收敛性与傅里叶神经算子学习。文中证明了一个Wong–Zakai定理，建立了由布朗运动的分段线性逼近所驱动的逐路径解向相应SVIE解的收敛性。该收敛性为使用镜像填充傅里叶神经算子来学习解算子奠定了基础，该解算子将初始条件和驱动噪声映射到完整的解轨迹。数值实验展示了具有竞争力的预测精度，以及向比训练所用时间网格更细的网格的泛化能力，同时评估了填充方式和计算成本的影响。作为对\citet{LKP:26}的后续研究，本研究通过对卷积记忆特性量身定制的独立收敛性分析与逼近性分析，将其算子学习框架扩展到了SVIE。

    arXiv:2610.18939v1 Announce Type: cross  Abstract: This paper studies Wong--Zakai convergence and Fourier neural operator learning for stochastic Volterra integral equations (SVIEs) describing random systems with memory. A Wong--Zakai theorem establishes convergence of pathwise solutions driven by piecewise linear approximations of Brownian motion to the corresponding SVIE solution. This convergence provides a foundation for using mirror-padded Fourier neural operators to learn the solution operator mapping initial conditions and driving noise to entire solution trajectories. Numerical experiments demonstrate competitive prediction accuracy and generalization to temporal grids finer than those used for training, while assessing the influence of padding and computational cost. As a follow-up to \citet{LKP:26}, this study extends their operator-learning framework to SVIEs through separate convergence and approximation analyses tailored to convolutional memory.
    
[^51]: 延迟复杂度维度：带延迟与删失标签的学习

    Delay Complexity Dimension: Learning with Delayed and Censored Labels

    [https://arxiv.org/abs/2610.18935](https://arxiv.org/abs/2610.18935)

    本文提出“延迟复杂度维度”（DCD），将VC维度与观测信道的标签信息率相结合，刻画了标签延迟且被删失情形下PAC学习的样本复杂度，证明任何学习器需要 Ω(DCD/ε²) 个样本，且伪标签ERM可在常数与对数因子内达到该界。

    

    我们研究二元分类器的PAC学习问题，其中标签在随机延迟后到达，并在观测窗口 $W$ 之外被删失（截断）。这种观测模型需要多少样本？我们提出了一个依赖于分布的答案——延迟复杂度维度。DCD 将VC维度与观测信道的精确标签信息率相结合。任何学习器都需要 $\Omega(\mathrm{DCD}/\varepsilon^2)$ 个样本；在给出该下界的实例族上，一个预言机伪标签ERM（经验风险最小化）在常数和对数因子范围内达到这一速率。在该实例族上，DCD 与经典删失ERM复杂度相差一个常数因子；而在二者分离的情形下，DCD 速率是否可达到仍是开放问题。该理论还给出了一种量化方案，在该方案下延迟信息可降低残余标签方差；提出了一种在其版本空间目标上具有 $(1-1/e)$ 保证的探索策略；以及一个……（原文摘要在此处截断）

    arXiv:2610.18935v1 Announce Type: cross  Abstract: We study PAC learning of binary classifiers when labels arrive after random delays and are censored beyond an observation window $W$. How many samples does this observation model require? We propose a distribution-dependent answer, the Delay Complexity Dimension (DCD). The DCD combines the VC dimension with the exact label-information rate of the observation channel. Every learner needs $\Omega(\mathrm{DCD}/\varepsilon^2)$ samples. An oracle pseudo-label ERM matches this rate on the instance family of the lower bound, up to constants and logarithmic factors. On that family the DCD is within a constant factor of the classical censored-ERM complexity, and whether the DCD rate is achievable where the two separate remains open. The theory also yields a quantization scheme under which delay information lowers the residual label variance, an expedition policy with a $(1-1/e)$ guarantee for its version-space objective, and a phenomenon with n
    
[^52]: 零和博弈中学习的汤普森采样

    Thompson Sampling for Learning in Zero-Sum Games

    [https://arxiv.org/abs/2610.18899](https://arxiv.org/abs/2610.18899)

    本文首次证明了在具有唯一纳什均衡的未知双人零和矩阵博弈中，汤普森采样可实现 $O(\sqrt T)$ 的对偶间隙遗憾（纯均衡情形下为 $O(\log T)$），并给出了与之匹配的极小极大下界和实例相关下界。

    

    汤普森采样（TS）在经典随机老虎机中实现了对数级和渐近最优的遗憾值，但其在多智能体学习中的行为仍鲜为人知。我们研究了汤普森采样在具有高斯老虎机反馈的未知有限双人零和矩阵博弈中的表现。在集中式自博弈下，TS从后验中采样单个收益矩阵，并执行采样博弈的一个纳什均衡。对于每一个具有唯一纳什均衡的固定零和博弈，我们证明了期望累积对偶间隙为 $O(\sqrt T)$。对于纯均衡情形，该速率可提升至 $O(\log T)$。随后我们确定了 $\sqrt T$ 项的精确系数，该系数仅取决于均衡策略和噪声水平。我们还建立了与 $T$ 的平方根依赖关系相匹配的极小极大下界，以及对于一类具有唯一纯均衡的博弈，与 $T$ 的对数依赖关系相匹配的实例相关下界。实验表明……

    arXiv:2610.18899v1 Announce Type: cross  Abstract: Thompson sampling (TS) achieves logarithmic and asymptotically optimal regret in classical stochastic bandits, but its behavior in multi-agent learning remains less understood. We study TS in an unknown finite two-player zero-sum matrix game with Gaussian bandit feedback. Under centralized self-play, TS samples a single payoff matrix from the posterior and plays a Nash equilibrium of the sampled game. For every fixed zero-sum game with a unique Nash equilibrium, we show that the expected cumulative duality gap is $O(\sqrt T)$. The rate improves to $O(\log T)$ for pure equilibria. We then identify the exact coefficient of the $\sqrt T$ term, which depends only on the equilibrium strategies and the noise level. We also establish a minimax lower bound matching the square-root dependence on $T$ and an instance-dependent lower bound matching the logarithmic dependence on $T$ for a class of games with a unique pure equilibrium. Experiments s
    
[^53]: 面向零膨胀连续结果的贝叶斯因果森林

    Bayesian Causal Forests for Zero-Inflated Continuous Outcomes

    [https://arxiv.org/abs/2610.18889](https://arxiv.org/abs/2610.18889)

    本文提出ZIC-BCF-Smear方法，通过解耦的两部分贝叶斯因果森林（Probit BCF建模零值参与、高斯BCF建模正值强度），结合子群体倾向得分调整与Duan非参数平滑估计，有效解决了半连续结果下异质性处理效应估计的难题。

    

    当结果变量为半连续型（即在零处呈现离散点质量，而正值部分呈现偏态的连续分布）时，估计（异质性）处理效应仍然极具挑战性，尽管这种数据类型在现实世界场景中广泛存在。现有方法要么忽略零值点质量，要么施加脆弱的参数化假设，要么面临极高的计算成本和数据增强噪声。本文提出了带有Duan平滑估计的零膨胀连续贝叶斯因果森林，这是一种解耦的两部分非参数估计器：在完整样本上拟合Probit BCF以建模参与（是否取零值），并严格在活跃子群体上拟合高斯BCF以建模正值的强度。子群体倾向得分调整缓解了活跃子群体内部由正则化引起的混杂，而Duan的非参数平滑估计器则将预测重新转换回响应尺度。

    arXiv:2610.18889v1 Announce Type: cross  Abstract: Estimating (heterogeneous) treatment effects is still challenging when the outcome is semicontinuous (i.e., exhibiting both a discrete point mass at zero and a skewed continuous distribution for positive values), despite the strong presence of this data type in real world settings. Existing methods either ignore the zero mass, impose fragile parametric assumptions, or suffer from extreme computational costs and data augmentation noise. This paper proposes Zero-Inflated Continuous Bayesian Causal Forests with Duan's Smearing (ZIC-BCF-Smear), a decoupled two-part nonparametric estimator that fits a Probit BCF on the full sample to model participation and a Gaussian BCF strictly on the active subpopulation to model positive intensity. Subpopulation Propensity Adjustment (SPA) mitigates regularization-induced confounding within the active subpopulation, and Duan's non-parametric smearing estimator re-transforms predictions to the response 
    
[^54]: 基于仅答案数据的高效后训练与验证的逆向推理模型

    Inverse Reasoning Models for Efficient Post-Training and Verification from Answer-Only Data

    [https://arxiv.org/abs/2610.18878](https://arxiv.org/abs/2610.18878)

    该论文提出一种自蒸馏方法，通过训练一个“逆向推理模型”根据答案反向生成推理轨迹，从而仅利用提示-答案对数据即可高效后训练推理大语言模型，在数据效率与计算效率之间取得平衡。

    

    arXiv:2610.18878v1 公告类型：新论文 摘要：对推理型大语言模型进行后训练通常需要在昂贵的监督与低效的计算之间做出权衡：监督微调通常需要推理轨迹，这在数据上是低效的，因为在许多应用场景中，人们只有提示-答案对，而缺乏真实的思维链（CoT）序列。另一方面，强化学习（例如使用可验证奖励的方法）虽然可以在没有CoT数据的情况下运行，但在计算上却低效得多，因为它在每次优化步骤中都需要进行多次自回归生成，而这些生成无法保证产生非零的策略梯度。在本工作中，我们探索了一种折中方案——一种仅使用答案数据的自蒸馏技术，将思维链视为隐变量。这依赖于训练一个配套的“逆向推理大语言模型”，该模型能够根据答案生成原始模型的推理轨迹。该逆向模型只需训练一次，即可在后续传入的各个批次中重复使用……

    arXiv:2610.18878v1 Announce Type: new  Abstract: Post-training reasoning LLMs often requires making a tradeoff between costly supervision and compute inefficiency: supervised finetuning typically requires reasoning traces, which is data inefficient since in many applications one only has prompt-answer tuples and lacks the groundtruth chain-of-thought (CoT) sequence. On the other hand, reinforcement learning (with verifiable rewards, for example) can operate without CoT data but it is much more compute-inefficient since it requires multiple autoregressive completions per optimization step, none of which are guaranteed to give a non-zero policy gradient. In this work we explore a middleground, a self-distillation technique that uses answer-only data, treating CoT as latent variables. This relies on training a partner ``inverse reasoning LLM" that generates the original's reasoning traces conditioned on the answer. The inverse model is trained once and reused across incoming batches of an
    
[^55]: 基于收敛率引导的连续仿真优化期望改进算法设计

    Convergence-Rate-Guided Algorithm Design of Expected Improvement for Continuous Simulation Optimization

    [https://arxiv.org/abs/2610.18434](https://arxiv.org/abs/2610.18434)

    该论文通过推导简单遗憾的高概率上界，在收敛率分析的基础上系统性地设计了随机克里金期望改进算法的三个关键组件——当前最优解、采样重复次数和终端推荐，解决了未知异方差噪声下连续仿真优化中依赖启发式规则进行算法设计的局限。

    

    基于随机克里金替代模型的期望改进方法被广泛应用于连续仿真优化，但仿真输出上未知的异方差噪声使得三个基本设计选择变得至关重要：用于定义改进的当前最优解、每个采样点的重复次数，以及终端推荐。现有方法通常要么假设噪声水平已知，要么通过启发式规则或经验校准分别选择这些组件。我们转而推导简单遗憾的高概率上界，并在一般规则类内系统地选择这三个组件以收紧该上界。我们首先为具有未知和已估计噪声水平的随机克里金预测器建立有限样本置信界，作为三个设计选择的基础。随后我们分析了用于当前最优解和终端推荐的一般后验分位数族。

    arXiv:2610.18434v1 Announce Type: new  Abstract: Expected improvement (EI) based on stochastic kriging (SK) surrogate is widely used in continuous simulation optimization, but unknown heteroscedastic noise on simulation output makes three basic design choices consequential: the incumbent used to define improvement, the replication level at each sampled point, and the terminal recommendation. Existing methods typically either assume the noise levels are known or choose these components separately through heuristic rules or empirical calibration. We instead derive a high-probability upper bound on the simple regret and systematically choose the three components within general rule classes to tighten this bound. We first establish a finite-sample confidence bound for the SK predictor with unknown and estimated noise levels as a foundation for the three design choices. We then analyze general posterior-quantile families for both the incumbent and the terminal recommendation. For the incumb
    
[^56]: 常学习率随机梯度下降的强高斯逼近：偏差与轨迹推断

    Strong Gaussian Approximation of Constant-Learning-Rate SGD: Bias and Trajectory Inference

    [https://arxiv.org/abs/2610.14567](https://arxiv.org/abs/2610.14567)

    本文为重尾噪声下的常学习率随机梯度下降建立了KMT型强高斯逼近，实现了有限样本下同时性的整个轨迹层面的推断，并蕴含泛函中心极限定理。

    

    本文为常学习率随机梯度下降（SGD）建立了强高斯逼近（KMT型逼近）。我们在重尾噪声的情形下开展工作，并假设迭代次数增长与学习率收缩同时进行。我们的工作在同一概率空间上构造了简单的高斯过程，并以概率方式界定了SGD误差与其各自极限之间的最大差异。尽管目前已存在以最优点为中心的常学习率SGD的中心极限定理，但这些定理仅能提供单个迭代点的推断。相比之下，我们的淬火型（quenched）KMT逼近能够实现有限样本下同时性的、整个轨迹层面的推断，并蕴含随机过程的收敛性（即泛函中心极限定理FCLT）。对于以真实最优点为中心的累积误差，当学习率满足 $\gamma_n=\frac{c}{\sqrt n}+o(n^{-1/2})$（其中 $c$ 为常数）时，我们首先得到了KMT型逼近界。

    arXiv:2610.14567v1 Announce Type: new  Abstract: This paper develops strong Gaussian approximations (KMT type) for constant learning rate stochastic gradient descent (SGD). We work with heavy tailed noise and assume that the number of iterates grows and the learning rate shrinks simultaneously. Our work constructs simple Gaussian processes on the same space and bounds, in probability, the maximum difference between the SGD errors and their respective limits. While there exist central limit theorems for constant rate SGD centered at the optimum, these only provide individual iterate inference. By contrast, our quenched KMT approximations allow for finite sample, simultaneous, whole-trajectory-level inference and imply stochastic process convergence (Functional Central Limit Theorems (FCLTs)). For accumulated errors centered at the true optimum, we first obtain a KMT-type approximation bound when the learning rate scales as $\gamma_n=\frac{c}{\sqrt n}+o(n^{-1/2})$ for a constant $c \in \
    
[^57]: 面向轨迹外推的目标约束池化

    Target-Constrained Pooling for Trajectory Extrapolation

    [https://arxiv.org/abs/2610.13034](https://arxiv.org/abs/2610.13034)

    提出目标约束池化（TCP）方法，利用已完成的相关轨迹从两种互补池化信号中估计轨迹终点并修正自由预测，在五个领域的25,245个目标上将平均RMSE降低35.8%至42.1%。

    

    从部分目标轨迹进行轨迹外推是一个欠定问题，但已完成的相关轨迹能提供有关未确定延续部分的信息。我们提出了目标约束池化（TCP），该方法将有界时间范围的终点与延续形态分离，并从两个互补的源信号中估计该终点。均匀池化捕捉最终响应水平，而匹配增量池化在对齐源与目标状态后捕捉剩余变化。TCP选择代表性源，将两个终点估计投影到与目标一致的区域，将其转化为对自由预测的修正，并通过强度校准和退化感知加权来控制其影响。在来自五个领域、涵盖两个预测方向的25,245个目标上，TCP使所有十个仅基于目标的预测器的平均RMSE降低了35.8%至42.1%，并改进了全部100个预测器、数据集与方向的组合。

    arXiv:2610.13034v1 Announce Type: new  Abstract: Trajectory extrapolation from a partial target trajectory is underdetermined, but completed related trajectories provide information about the unresolved continuation. We propose target-constrained pooling (TCP), which separates the finite-horizon endpoint from continuation shape and estimates that endpoint from two complementary source signals. Uniform pooling captures final response levels, while matched-increment pooling captures the remaining change after aligning source and target states. TCP selects representative sources, projects both endpoint estimates onto a target-consistent region, converts them into corrections of the free forecast, and controls their influence through strength calibration and degradation-aware weighting. Across 25,245 targets from five domains and both forecasting directions, TCP reduces mean RMSE by 35.8% to 42.1% for all ten target-only forecasters and improves all 100 forecaster, dataset, and direction c
    
[^58]: 利用序列蒙特卡洛引导扩散模型生成稀有事件

    Steering Diffusion Models to Rare Events with Sequential Monte Carlo

    [https://arxiv.org/abs/2610.08652](https://arxiv.org/abs/2610.08652)

    本文提出DireSMC，一种序列蒙特卡洛方法，通过引导加权样本群体趋向扩散模型中的稀有事件，不仅能生成稀有事件样本，还能给出其概率的校准估计，并可轻松扩展到各类用户自定义稀有事件。

    

    扩散模型正日益被用作天气预报、分子动力学和材料设计等领域中昂贵模拟器的替代品。在这些模型中，计算事件 $E$ 的概率 $p_0[E]$ 十分困难，尤其当所关注的事件是稀有事件时。使用蒙特卡洛方法进行稳定估计在计算上将变得不可行，因为需要随稀有程度增加而不断增长的样本量（$\propto 1/p_0[E]$）来补偿。在本文中，我们提出了稀有事件的扩散重要性采样方法（Diffusion Importance Sampling of Rare Events，简称 DireSMC），这是一种序列蒙特卡洛方案，通过引导一组加权样本趋向稀有事件，不仅能获得样本，还能得到其概率的校准估计。我们通过对事件集合进行解析松弛来构建引导机制，使该方法能够轻松扩展到广泛的用户自定义稀有事件。我们在一个具有解析解的玩具问题以及一个基于分数的模型上验证了我们的方法。

    arXiv:2610.08652v1 Announce Type: cross  Abstract: Diffusion models are increasingly used as surrogates for expensive simulators in weather prediction, molecular dynamics, and materials design. In these models, computing the probability $p_0[E]$ of an event $E$ is difficult, especially when the event of interest is rare. A stable estimate using Monte Carlo becomes computationally intractable, requiring a growing sample size $\propto\!1/p_0[E]$ to compensate for an increasing rarity. In this paper, we present Diffusion Importance Sampling of Rare Events or DireSMC, a sequential Monte Carlo scheme that guides a population of weighted samples towards the rare event, giving access not only to samples but also to a calibrated estimate of its probability. We set up our guidance using an analytical relaxation of the event set, allowing the method to easily extend to a wide range of user-defined rare events. We validate our method on a toy problem with analytical solutions and on a score-based
    
[^59]: ProximalFM：隐藏混杂下的摊销式近端因果推断

    ProximalFM: Amortized Proximal Causal Inference under Hidden Confounding

    [https://arxiv.org/abs/2610.08078](https://arxiv.org/abs/2610.08078)

    该论文提出ProximalFM，利用先验数据拟合网络（PFN）以摊销式贝叶斯方法在隐藏混杂下进行近端因果推断，通过先验正则化缓解了非参数近端估计中病态积分方程的数据饥渴、超参数敏感和优化不稳定等问题。

    

    标准的因果识别方法通常假设不存在未观测的混杂因素，当相关混杂因素未被观测到时这些方法可能会失效。近端因果推断则转而使用代理变量在隐藏混杂存在的情况下识别因果效应。然而，非参数近端估计在实践中可能颇具挑战性：恢复条件平均处理效应（CATE）等因果估计量需要求解一个病态的积分方程，该方程对数据需求量大、对超参数敏感且优化过程不稳定。对此类模型进行贝叶斯推断提供了一种理想的替代方案，可通过先验进行正则化来缓解上述困难。然而，计算后验本身也具有挑战性，因为典型的似然函数中会包含潜变量。借鉴近期表格基础模型在后门、工具变量和前门等设定中取得的成功，我们提出先验数据拟合网络（PFN）在……方面具有独特优势（摘要在此处截断）

    arXiv:2610.08078v1 Announce Type: cross  Abstract: Standard causal identification methods often assume no unmeasured confounding and can fail when relevant confounders are unobserved. Proximal causal inference instead uses proxy variables to identify effects under hidden confounding. However, nonparametric proximal estimation can be challenging in practice: recovering causal estimands such as the conditional average treatment effect (CATE) requires solving an ill-posed integral equation that is data-hungry, hyperparameter-sensitive, and optimization-unstable. Bayesian inference for such models provides a desirable alternative, mitigating these difficulties by regularizing through the prior. However, computing a posterior is itself challenging, as a typical likelihood function will include latent variables. Following the recent success of tabular foundation models in backdoor, instrumental variable, and frontdoor settings, we propose that prior-data fitted networks (PFNs) are uniquely s
    
[^60]: 藤Copula VAR：从递归边际到联合预测推断

    Vine Copula VAR:From Recursive Margins to Joint Forecast Inference

    [https://arxiv.org/abs/2610.07589](https://arxiv.org/abs/2610.07589)

    本文在稳定的藤Copula VAR框架下，推导出历史递归估计误差与终端预测的联合影响，并提出一个协方差估计量，可为固定单侧事件概率提供渐近有效的重复样本预测区间。

    

    联合事件预测通常将基于过去预测误差的依赖性估计与新估计的边际分布相结合。当每个历史误差保留其发布日期时可获得的边际拟合时，推断必须考虑一系列相互重叠的估计误差。我们在一个具有正态创新项边际、且采用固定的、正确设定的Gaussian或正Clayton藤结构的稳定Vine Copula VAR中，推导出这些历史误差与终端预测估计的联合影响。利用截距恒等式和稳定的VAR滤波器，历史修正被简化为调和加权的创新项矩，而终端斜率的不确定性仍然保留。由此得到的协方差估计量，能够在已实现的预测状态下，为固定的单侧事件概率提供渐近有效的重复样本区间。在Gaussian子模型中，相对于在同一……

    arXiv:2610.07589v1 Announce Type: new  Abstract: Joint-event forecasts often combine a dependence estimate based on past forecast errors with newly estimated marginal distributions. When each historical error retains the marginal fit available at its issue date, inference must account for an overlapping sequence of estimation errors. We derive their joint influence with the terminal forecast estimates in a stable Vine Copula VAR with normal innovation margins and a fixed, correctly specified Gaussian or positive Clayton vine. An intercept identity and the stable VAR filter reduce the historical correction to harmonically weighted innovation moments, while terminal slope uncertainty remains. The resulting covariance estimator gives asymptotically valid repeated-sample intervals for fixed one-sided event probabilities at the realized forecast state. In the Gaussian submodel, retaining issued transforms adds a positive semidefinite covariance term relative to refitting margins on the same
    
[^61]: 预测导向的高斯过程后验

    Predictively Oriented Gaussian Process Posteriors

    [https://arxiv.org/abs/2610.03201](https://arxiv.org/abs/2610.03201)

    提出预测导向高斯过程（PrO-GPs），将预测不确定性作为主要推断目标，在模型错误设定下比标准高斯过程产生校准更好的预测分布。

    

    高斯过程（GP）是建模和量化函数关系中不确定性的强大工具。然而，它们要求使用者做出许多设计决策，例如核函数和观测模型的选择。次优的选择可能产生错误设定的模型，无法捕捉潜在的数据生成过程。我们提出了预测导向高斯过程（PrO-GPs），它将预测不确定性作为主要的推断目标，为标准高斯过程提供了一种稳健的替代方案。尽管对非参数模型直接计算PrO后验是不可行的，我们推导了一种简化形式和实用的采样方案以实现高效计算。通过合成数据和真实数据的实验，我们表明与标准高斯过程方法相比，PrO-GPs在模型错误设定的情况下能够产生更好校准的预测分布。

    arXiv:2610.03201v1 Announce Type: cross  Abstract: Gaussian Processes (GPs) are a powerful tool for modelling and quantifying uncertainty in functional relationships. However, they require practitioners to make a number of design decisions, such as the choice of the kernel and the observation model. Suboptimal choices can produce misspecified models that do not capture the underlying data generating process. We introduce Predictively Oriented Gaussian Processes (PrO-GPs), which treat predictive uncertainty as the primary inferential target and provide a robust alternative to standard GPs. Although direct computation of a PrO posterior for nonparametric models is intractable, we derive a reduced formulation and practical sampling scheme for efficient computation. Through synthetic and real data experiments, we show that PrO-GPs produce better calibrated predictive distributions under model misspecification compared to standard GP approaches.
    
[^62]: 针对难解似然的摊销比率估计重要性采样与局部化基于模拟的校准

    Amortized ratio-estimation importance sampling and localized simulation-based calibration for intractable likelihoods

    [https://arxiv.org/abs/2609.39712](https://arxiv.org/abs/2609.39712)

    该论文提出了针对难解似然模型的顺序化后验估计方法与局部化模拟校准技术，利用局部化条件密度代理和摊销化的似然-证据比率估计器，以低计算成本实现准确且经过校准的贝叶斯推断。

    

    我们考虑对具有难解似然但具备可操作前向模拟的模型参数进行基于模拟的贝叶斯推断。在专家混合高斯代理模型的基础上，作为第一个贡献，我们开发了一种用于后验估计的顺序化程序，其中使用逐步局部化的、由数据引导的条件密度近似作为提议分布。随后，通过重要性采样步骤对最终的后验分布代理进行校正，该步骤引入了一个局部摊销化的似然-证据比率估计器。第二个贡献是局部化的基于模拟的校准。局部化SBC在刻意设置得比可用后验更宽的邻域上进行校准探测，且额外计算成本很低。局部化SBC无需为每个模拟的伪观测数据重复运行完整的推断流程，而是拟合局部代理模型和比率估

    arXiv:2609.39712v1 Announce Type: cross  Abstract: We consider simulation-based Bayesian inference (SBI) for the parameters of models with intractable likelihoods but tractable forward simulation. Building on Gaussian mixtures-of-experts surrogates, as a first contribution we develop a sequential procedure for posterior estimation in which progressively localized, data informed, conditional density approximations are used as proposal distributions. A final surrogate of the posterior distribution is then corrected using an importance sampling step that introduces a locally amortized likelihood-to-evidence ratio estimator. A second contribution is localized simulation-based calibration (SBC). Localized SBC probe calibration over neighborhoods that are deliberately broader than the available posterior, at low additional computational cost. Rather than repeatedly running the full inference procedure for each simulated pseudo-observation, localized SBC fits the local surrogate and ratio est
    
[^63]: 多候选响应用于从示范中学习的统计优势

    The Statistical Benefits of Multiple Responses for Learning from Demonstrations

    [https://arxiv.org/abs/2609.33291](https://arxiv.org/abs/2609.33291)

    当示范者并非最优时，返回多个候选响应（pass@$k$）能带来更强的统计优势——将最坏情况下对目标精度的依赖从 $1/\varepsilon^2$ 改善为 $1/\varepsilon$，并将对奖励类大小的依赖从 $\log N$ 改善为 $\log N/\log k$。

    

    许多生成式系统会返回多个候选响应，并根据其中最优的一个进行评估。最近的研究表明，当示范为最优时，pass@$k$ 能够将通过示范学习的样本复杂度降低一个 $k$ 的对数因子。我们探讨了当示范者不被假设为最优时会发生什么。我们发现，在这种设定下，多个响应能带来定性上更强的优势。在一个没有奖励反馈的有限奖励类模型中，从 pass@$1$ 转变为任何 $k\ge2$ 的 pass@$k$，会将对目标精度的最坏情况依赖从 $1/\varepsilon^2$ 改善为 $1/\varepsilon$，且该结论在所有示范者质量下均成立。在标准评估设定下（即一个未知奖励在训练前被固定），增大 $k$ 还提供了一个额外的独特好处：对大小为 $N$ 的奖励类的最优依赖从 $\log N$ 改善为 $\log N/\log k$。我们进一步证明这两个效应……

    arXiv:2609.33291v2 Announce Type: replace-cross  Abstract: Many generative systems return multiple candidate responses and are evaluated according to the best one. Recent work shows that, when demonstrations are optimal, pass@$k$ can reduce the sample complexity of learning from demonstrations by a logarithmic factor in $k$. We ask what happens when the demonstrator is not assumed to be optimal. We find that multiple responses provide a qualitatively stronger benefit in this setting. In a finite reward-class model with no reward feedback, moving from pass@$1$ to any pass@$k$ with $k\ge2$ changes the worst-case dependence on target accuracy from $1/\varepsilon^2$ to $1/\varepsilon$, uniformly over demonstrator quality. Under standard evaluation, where an unknown reward is fixed before training, increasing $k$ provides an additional and distinct benefit: the optimal dependence on a reward class of size $N$ improves from $\log N$ to $\log N/\log k$. We further show that these two effects 
    
[^64]: 相对智能学习 II：可处理或半监督的实例最优学习

    Relatively Smart II: Tractable or Semi-Supervised Instance-Optimal Learning

    [https://arxiv.org/abs/2609.10886](https://arxiv.org/abs/2609.10886)

    本文证明ERM乃至任何正规一致学习器在无分布二分类中都是相对智能的，从而解决了此前留下的开放问题，即自然的可处理学习器能否达到与单包含图学习器可比的保证。

    

    我们继续研究由 Dughmi 和 Pour（2026）提出的“相对智能学习”概念，它要求监督学习器在逐一边际的意义上，与每一个能够从无标注数据中可靠验证的、针对固定分布的误差保证相竞争。他们证明了单包含图学习器是相对智能的，但样本复杂度存在二次方膨胀，且任何相对智能学习器都无法做得更好，由此留下了 ERM 或其他自然、可处理的学习器能否达到可比保证的开放问题。他们还留下另一个开放问题：这种样本复杂度膨胀是否可以仅限于无标注数据。我们的第一个结果表明，ERM——实际上是任何正规一致学习器——在无分布设定下的二分类问题中都是相对智能的。我们证明了，若在 m 个样本下具有较小的可验证误差，则意味着在大小为 O(m²) 的随机样本上的均匀分布中也具有类似的较小误差，从而在该样本上产生大小至多为 2^(m+1) 的覆盖……

    arXiv:2609.10886v1 Announce Type: new  Abstract: We continue the study of relatively smart learning, introduced by Dughmi and Pour (2026), which asks a supervised learner to compete, marginal by marginal, with every distribution-fixed error guarantee soundly certifiable from unlabeled data. They showed that the One-Inclusion Graph (OIG) learner is relatively smart with a quadratic sample-complexity blowup, and that no relatively smart learner can do better, leaving open whether ERM or another natural or tractable learner achieves comparable guarantees. They also left open whether the blowup can be restricted to unlabeled data.   Our firs results shows that ERM---and in fact any proper consistent learner---is relatively smart for binary classification in the distribution-free setting. We show that a small certifiable error with $m$ samples implies a similarly small error on the uniform distribution over a random sample of size $O(m^2)$, yielding a cover of size at most $2^{m+1}$ on that
    
[^65]: Q-Edge：对称性约简的结构化极端依赖量子模拟

    Q-Edge: Symmetry-Reduced Quantum Simulation of Structured Extreme Dependence

    [https://arxiv.org/abs/2609.03706](https://arxiv.org/abs/2609.03706)

    Q-Edge提出了一种基于对称性的无损轨道空间表示方法，将30维问题中约160万个角度状态压缩为256个轨道状态，从而实现了结构化极端依赖系统的可扩展量子模拟与数字孪生。

    

    高维多元极端事件的模拟从根本上受限于依赖关系的组合复杂性，这种限制往往甚至超过极端观测数据稀缺所带来的制约。我们证明，对称性允许一种无损的轨道空间表示，它在保持结构化极端依赖的同时，用一个紧凑的对称类集合取代了指数级庞大的依赖空间。基于这一原理，我们开发了Q-Edge（量子极端依赖引擎），这是一个直接在轨道空间中运行的对称性约简量子框架，能够实现结构化极端系统的可扩展模拟与数字孪生。通过将对称性转移到数据表示中而非量子电路中，Q-Edge使不受约束的量子生成模型能够利用大幅缩减的状态空间。对于一个30维问题，约160万个角度状态坍缩为256个轨道状态，从而显著降低了所需的量子表示规模。

    arXiv:2609.03706v1 Announce Type: cross  Abstract: High-dimensional simulation of multivariate extremes is fundamentally limited by the combinatorial complexity of dependence, often more than by the scarcity of extreme observations. We show that symmetry admits a lossless orbit-space representation that preserves structured extreme dependence while replacing an exponentially large dependence space with a compact set of symmetry classes. Based on this principle, we develop Q-Edge (Quantum Extreme Dependence Engine), a symmetry-reduced quantum framework that operates directly in orbit space, enabling scalable simulation and digital twins of structured extreme systems. By transferring symmetry into the data representation rather than the quantum circuit, Q-Edge allows unconstrained quantum generative models to exploit dramatically reduced state spaces. For a 30-dimensional problem, approximately 1.6 million angular states collapse to 256 orbit states, reducing the required quantum represe
    
[^66]: 在CIFAR-10上比较SGLD与固定噪声预测-校正采样器在经典联合能量模型中的改进应用

    Comparing SGLD and a fixed-noise predictor-corrector adaptation in canonical joint energy-based models on CIFAR-10

    [https://arxiv.org/abs/2608.05025](https://arxiv.org/abs/2608.05025)

    该论文系统复现了经典联合能量模型（JEM），并首次检验了用固定噪声预测-校正采样器替代SGLD的效果，结果显示其分类性能接近经典方法但生成质量略有差距，同时记录了两种失败模式。

    

    联合能量模型将分类与生成统一在单个网络中，并支持分布外（OOD）检测。经典的JEM训练依赖于随机梯度朗之万动力学（SGLD）；而一种有理论依据的替代方法——预测-校正（PC）采样器——此前尚未在经典模型上经过系统的复现检验。我们在两次独立运行中复现了基于WideResNet-28-10（不含归一化层）的经典JEM，并测试了一种固定噪声PC改进方法——用确定性梯度步骤替代存在缺陷的退火噪声预测器——涵盖三种协议：改进采样器在整个训练轨迹（115-132个epoch）中替代SGLD；冷启动生成（FID）；以及精炼式多OOD检测（AUROC）。复现结果达到92.88%的测试准确率和44.46的buffer-FID（经典方法为92.90%和38.40）。我们记录了两种失败模式：灾难性……

    arXiv:2608.05025v3 Announce Type: replace  Abstract: Joint Energy-Based Models (JEM) unify classification and generation within a single network and support out-of-distribution (OOD) detection. Canonical JEM training relies on stochastic gradient Langevin dynamics (SGLD); a theoretically motivated alternative, the Predictor-Corrector (PC) sampler, has not previously undergone a systematic replication test on the canonical model. We reproduce canonical JEM on WideResNet-28-10 without normalisation layers on two independent runs and test a fixed-noise PC adaptation - with the degenerate annealed-noise predictor replaced by a deterministic gradient step - across three protocols: the adapted sampler replacing SGLD throughout the full training trajectories (115-132 epochs); cold-start generation (FID); and refinement-style multi-OOD detection (AUROC). The reconstruction reaches 92.88% test accuracy and buffer-FID 44.46 (canonical: 92.90% and 38.40). We document two failure modes: catastroph
    
[^67]: 设计矩阵的零模式驱动过参数化回归中的多重下降

    The Zero Pattern of a Design Matrix Drives Multiple Descent in Over-parameterized Regression

    [https://arxiv.org/abs/2607.24041](https://arxiv.org/abs/2607.24041)

    本文放宽了过参数化线性回归中协变量独立且协方差矩阵非退化的假设，证明协方差退化与变量相关性会引发多重下降现象，并借助二部图的最大匹配和Dulmage-Mendelsohn分解精确定位了风险峰值出现的位置。

    

    过参数化线性回归在过去十年中得到了广泛研究。然而，大多数现有工作假设协变量是相互独立的，且其协方差矩阵是非退化的。本文放宽了这两个假设，并在岭正则化趋于零的情形下推导出预测风险的确定性等价形式。我们证明协方差矩阵的退化性和变量间的相关性可以导致多重下降现象，并刻画了相应峰值可能出现的位置。我们的证明采用了一种新颖的方差轮廓图表示方法。我们证明了相关二部图的最大匹配和Dulmage--Mendelsohn分解能够识别出方差变得奇异的配置。

    arXiv:2607.24041v2 Announce Type: replace-cross  Abstract: Over-parameterized linear regression has been widely studied over the last decade. However, most existing works assume that the covariates are independent and that their covariance matrices are non-degenerate. In this paper, we relax both assumptions and derive deterministic equivalents for the prediction risk in a vanishing-ridge regime. We show that degeneracy of the covariance matrices and dependence can lead to multiple descent, and characterize where the corresponding peaks can occur. Our proofs use a novel graph representation of the variance profile. We show that maximum matchings and the Dulmage--Mendelsohn decomposition of the associated bipartite graph identify the configurations at which the variance becomes singular.
    
[^68]: Hopformer：面向时间序列预测的同质性追求Transformer

    Hopformer: Homogeneity-Pursuit Transformer for Time Series Forecasting

    [https://arxiv.org/abs/2607.22299](https://arxiv.org/abs/2607.22299)

    Hopformer提出两阶段时间序列预测框架，先通过稀疏模式聚合（SPA）提取融合协变量的共同低方差趋势作为同质化层，再用LoRA微调的Transformer建模残差中的复杂依赖，在理论上证明了近最优偏差-方差权衡与泛化界，在多个基准上平均提升MASE 6.56%。

    

    对包含高维协变量的多条时间序列进行预测面临一个核心挑战：如何在统一共同的时间模式的同时保留有意义的序列特定信息。我们提出了Hopformer（同质性追求Transformer），一个解决该挑战的两阶段框架。在第一阶段，我们执行稀疏模式聚合（SPA）方案，提取一个融合协变量的共同低方差趋势，这充当了一个同质化层。在第二阶段，一个经LoRA微调的Transformer对残差中剩余的复杂依赖关系进行建模。我们的方法具有坚实的理论基础：我们通过oracle不等式证明SPA实现了近乎最优的偏差-方差权衡，并在相依时间序列数据下为第二阶段提供了泛化界。Hopformer创造了新的最先进水平，在合成和真实世界预测基准上平均将MASE提升了6.56%。

    arXiv:2607.22299v2 Announce Type: replace-cross  Abstract: Forecasting multiple time-series with high-dimensional covariates presents a core challenge: unifying common temporal patterns while retaining meaningful series-specific information. We introduce Hopformer (Homogeneity-Pursuit Transformer), a two-stage framework that addresses this challenge. In the first stage, we perform a Sparsity Pattern Aggregation (SPA) scheme extracting a common low-variance trend that incorporates the covariates. This acts as a homogenization layer. In the second stage, a LoRA-fine-tuned Transformer models the remaining complex dependencies in the residual. Our method is theoretically grounded. We prove that SPA achieves a near-optimal bias-variance trade-off via an oracle inequality. We also provide generalization bounds for the second stage under dependent time series data. Hopformer sets a new state of the art, improving MASE by an average of 6.56% across synthetic and real-world forecasting benchmar
    
[^69]: 有向图上基于流恢复势能的规范不变、参数不敏感正则化方法

    Gauge-Invariant, Parameter-Insensitive Regularization for Potential Recovery from Flow on Directed Graphs

    [https://arxiv.org/abs/2607.13609](https://arxiv.org/abs/2607.13609)

    该论文提出用规范不变的图狄利克雷能量替代岭正则化，从有向图流数据中恢复势能，解决了岭正则化导致的排序反转问题，并实现了跨λ四个数量级的参数不敏感性。

    

    从有向图上观测到的流中恢复潜在势能（即带狄利克雷边界的离散泊松问题）是一个不适定问题，而标准的解决方案反而适得其反：岭正则化会向规范无意义的原点收缩，导致恢复的排序发生坍塌和反转（相对于植入的真实值，秩相关从+0.81变为-0.42）。规范不变的图狄利克雷能量消除了这一风险，并带来了参数不敏感性：估计值在λ的四个数量级范围内保持稳定，而岭正则化在所有λ>0的情况下都会反转排序。作者证明了约化求解是SPD（对称正定）的，并在岭正则化使其坍塌的地方精确保存了动态范围，还通过泊松残差仅从流中定位吸收边界。H^1半范数本身是经典的；新颖之处在于规范诊断、由此获得的参数不敏感性，以及表明结果对提取方法稳健的消融实验。在三个公开的点击流数据集上……（摘要被截断）

    arXiv:2607.13609v2 Announce Type: replace  Abstract: Recovering a latent potential from observed flow on a directed graph (a discrete Poisson problem with Dirichlet boundaries) is ill-posed, and the standard fix backfires: ridge regularization shrinks toward a gauge-meaningless origin, collapsing and reversing the recovered ordering ($+0.81\to-0.42$ rank correlation against a planted ground truth). The gauge-invariant graph Dirichlet energy removes the hazard and delivers parameter-insensitivity: the estimate is stable across four orders of magnitude in $\lambda$, whereas ridge inverts the ordering for every $\lambda>0$. We prove the reduced solve is SPD and preserves dynamic range exactly where ridge collapses it, and localize absorbing boundaries from flow alone via a Poisson residual. The $H^1$ seminorm is classical; what is new is the gauge diagnosis, the parameter-insensitivity it buys, and an ablation showing the result is robust to the extraction method. On three public clickstr
    
[^70]: 离散扩散模型学到了什么？三种坐标系下的同一个逆向过程

    What Does a Discrete Diffusion Model Learn? One Reverse Process in Three Coordinates

    [https://arxiv.org/abs/2607.05381](https://arxiv.org/abs/2607.05381)

    本文证明离散扩散模型中的去噪器、桥接插件和具体分数三种方法学习的其实是同一对象——加噪过程的精确时间反演，只是处于三种不同坐标系，并给出 Oracle Distance 恒等式（负 ELBO 等于数据熵加路径 KL 散度）及三种坐标间的精确转换。

    

    离散扩散模型究竟学到了什么？现有文献分别训练去噪器、桥接插件和具体分数，每种方法都在不同的约定下推导出自己的 ELBO 目标。这种碎片化使得所学习的对象不清晰，ELBO 数值也难以解释和比较。本文指出，这三种方法在不同的坐标系中学习的是同一个对象：加噪过程的精确时间反演。我们首先证明了 Oracle Distance 恒等式：对于任意加噪过程，负 ELBO 不仅仅是似然的一个界；它等于数据熵加上从反向加噪过程到所学过程的路径 KL 散度。接下来，针对令牌因子化的噪声，我们证明去噪器、桥接插件和具体分数是该逆向过程的三种坐标，推导出它们之间的精确转换关系，并确定了每种方法所学习的概率分布。特别是，与其标准解释相反……（摘要原文在此处被截断）

    arXiv:2607.05381v2 Announce Type: replace-cross  Abstract: What does a discrete diffusion model actually learn? The literature trains denoisers, bridge plug-ins, and concrete scores, each deriving its own ELBO objective under different conventions. This fragmentation makes the object being learned unclear and ELBO values difficult to interpret and compare. All three methods learn the same object in different coordinates: the exact time-reversal of the noising process. We first prove the Oracle Distance identity: for any noising process, the negative ELBO is not just a bound on the likelihood; it equals the data entropy plus the path KL divergence from the reversed noising process to the learned one. Next, for token-factorizing noise, we show that denoiser, bridge plug-in, and concrete score are three coordinates of this reverse process, derive exact conversions between them, and identify the probability distribution learned by each. In particular, contrary to its standard interpretatio
    
[^71]: 自组织保形预测：通过无监督群体发现减少区域覆盖差距

    Self-Organized Conformal Prediction: Reducing Regional Coverage Gaps with Unsupervised Group Discovery

    [https://arxiv.org/abs/2606.29403](https://arxiv.org/abs/2606.29403)

    提出自组织保形预测（SOCP），利用无需校准标签的无监督自组织映射（SOM）发现输入空间分组并从中提取校准缓冲，在预测器和非一致性分数保持不变的前提下实现精确的条件覆盖有效性，从而减少特征空间异质区域的覆盖差距。

    

    保形预测能够保证边际覆盖率，但汇总的校准分位数可能会掩盖特征空间中异质区域的系统性覆盖不足问题。我们提出自组织保形预测（SOCP），这是一种校准方案，利用无需校准标签训练的无监督自组织映射（SOM）来发现输入空间的分组。在预测时，查询的最佳匹配单元（BMU）从单个单元、固定网格邻域或基于原型的扩展中提取校准缓冲。当固定邻域过于稀疏时，模式3（Regime 3）根据原型距离添加单元，其全局预算在观察到任何校准分数之前，就从训练单元的占用情况和计划校准规模中选定。预测器和非一致性分数保持不变。仅单元检索具有精确的单元条件有效性，且每个固定的单元并集都具有精确的检索集有效性。（摘要在此处截断）

    arXiv:2606.29403v2 Announce Type: replace-cross  Abstract: Conformal prediction guarantees marginal coverage, but a pooled calibration quantile can hide systematic undercoverage across heterogeneous regions of the feature space. We introduce Self-Organized Conformal Prediction (SOCP), a calibration scheme that discovers input-space groups with an unsupervised Self-Organizing Map (SOM) trained without calibration labels. At prediction time, the query's best-matching unit (BMU) draws a calibration buffer from one cell, a fixed grid neighborhood, or a prototype-based enlargement. When fixed neighborhoods are too sparse, Regime 3 adds cells by prototype distance, using a global budget selected from training-cell occupancies and the planned calibration size before any calibration score is observed. The predictor and nonconformity score remain unchanged. Cell-only retrieval has exact cell-conditional validity, and each fixed union of cells has exact retrieved-set validity. Interpreting a nei
    
[^72]: 组别失衡下风险异质性的稳健推断

    Robust inference for risk heterogeneity under group imbalance

    [https://arxiv.org/abs/2606.00797](https://arxiv.org/abs/2606.00797)

    提出一种基于Neyman正交性的稳健框架，用于在组别失衡情形下推断两个总体间的风险异质性，有效降低偏差并提升有限样本下的推断稳定性。

    

    总体层面的异质性在生物医学数据中普遍存在，人口统计学或临床亚组之间的差异可能会显著改变风险模式。例如，在重症监护病房（ICU）研究中，与特定入院诊断相关的死亡风险可能因种族群体而异。现有的风险异质性检测方法往往对基线模型的误设和正则化偏差敏感，而这两种问题在实践中经常出现。本文提出了一种利用Neyman正交性来推断两个总体之间风险异质性的稳健框架，该方法所得到的估计量对冗余参数的估计误差具有局部不敏感性。所提出的估计量具有一致性和渐近正态性，仿真研究表明，在有限样本条件下，与基于标准似然的方法相比，我们的方法显著降低了偏差并提高了推断的稳定性。

    arXiv:2606.00797v2 Announce Type: replace-cross  Abstract: Population-level heterogeneity is ubiquitous in biomedical data, where differences across demographic or clinical subgroups can substantially alter risk patterns. For example, in intensive care unit (ICU) studies, the mortality risk associated with specific admission diagnoses can vary across ethnic groups. Existing approaches for detecting risk heterogeneity are often sensitive to baseline model misspecification and regularization bias, both of which commonly arise in practice. In this paper, we propose a robust framework for inferring risk heterogeneity between two populations using Neyman orthogonality, which yields estimators that are locally insensitive to nuisance parameter estimation error. The proposed estimator is consistent and asymptotically normal, and simulation studies demonstrate that in finite samples our method substantially reduces bias and improves inferential stability compared with standard likelihood-based
    
[^73]: 记忆设计：概率序列层

    Memory by Design: Probabilistic Sequence Layers

    [https://arxiv.org/abs/2605.31163](https://arxiv.org/abs/2605.31163)

    本文提出了一种设计模型框架，通过贝叶斯滤波和协方差传播统一多种次二次递归序列层，并恢复协方差传播以增强记忆保留和检索。

    

    arXiv:2605.31163v3 公告类型：交叉替换 摘要：我们引入了“设计模型框架”：一种从关于记忆的明确假设中推导高效循环序列映射的方法。设计模型通过精确贝叶斯滤波将证据写入记忆；查询相关的读取输出产生预测分布，其均值作为层输出。在我们的线性-高斯实例化中，“贝叶斯层”同时传播均值和协方差：协方差跟踪存储关联中的不确定性，将写入导向不确定方向，随着证据积累而衰减增益，并保留自信记忆。同一框架统一了多种次二次递归：线性注意力、GLA和Mamba-2/SSD在潜在输入设计模型下是精确滤波器，而DeltaNet及相关Delta规则模型是贝叶斯层设计模型的协方差重置简化。恢复协方差传播为检索提供闭式预测。

    arXiv:2605.31163v3 Announce Type: replace-cross  Abstract: We introduce the \emph{design-model framework}: a way to derive efficient recurrent sequence maps from explicit assumptions about memory. A design model writes evidence into memory by exact Bayesian filtering; a query- dependent readout produces a predictive distribution whose mean is the layer output. In our linear-Gaussian instantiation, the \emph{Bayesian Layer} propagates both a mean and a covariance: the covariance tracks uncertainty over stored associations, steering writes toward uncertain directions, attenuating gains as evidence accumulates, and preserving confident memories. The same framework unifies several sub-quadratic recurrences: linear attention, GLA, and Mamba-2/SSD are exact filters under a latent-input design model, whereas DeltaNet and related Delta-rule models are covariance-reset reductions of the Bayesian Layer's design model. Restoring covariance propagation yields closed-form predictions for retrieval 
    
[^74]: 优化器设计的对称兼容原则：嵌入层、语言模型输出头、SwiGLU多层感知机与混合专家路由器

    Symmetry-Compatible Principle for Optimizer Design: Embeddings, LM Heads, SwiGLU MLPs, and MoE Routers

    [https://arxiv.org/abs/2605.18106](https://arxiv.org/abs/2605.18106)

    该论文提出优化器设计的对称兼容原则——梯度更新规则应在权重块的对称群下保持等变，据此统一了双正交等变更新方法，并为嵌入层、语言模型输出头、SwiGLU MLP和MoE路由器等结构推导出对称兼容的优化器。

    

    摘要：深度学习实践中长期存在一个显著的几何差异。尽管现代神经网络架构天然展现出丰富的对称性与等变性，但Adam及其变体等流行的优化器在本质上是逐坐标操作的，因而无法尊重参数空间的等变性结构。我们通过为优化器设计引入一个对称兼容原则来解决这一差异：梯度更新规则应当在作用于相应权重块的对称群下保持等变。遵循这一原则，我们首先为一般矩阵层的双正交等变更新提供了统一视角，这正是随机谱下降、Muon、Scion和极梯度方法所采用的方式。更重要的是，通过从正交群拓展到置换对称与共享平移对称，我们为具有相应对称性的参数块推导出了对称兼容的优化器……

    arXiv:2605.18106v5 Announce Type: replace-cross  Abstract: A striking geometric disparity has long persisted in the practice of deep learning. While modern neural network architectures naturally exhibit rich symmetry and equivariance properties, popular optimizers such as Adam and its variants operate inherently coordinate-wise, rendering them unable to respect the equivariance structures of the parameter space. We address this disparity by introducing a symmetry-compatible principle for optimizer design: the gradient update rule should be equivariant under the symmetry group acting on the corresponding weight block. Following this principle, we first provide a unified perspective on bi-orthogonally equivariant updates for general matrix layers, as employed by stochastic spectral descent, Muon, Scion, and polar gradient methods. More importantly, by moving from orthogonal groups to permutation and shared-shift symmetries, we derive symmetry-compatible optimizers for parameter blocks wh
    
[^75]: 一种用于复合对数凹采样的近端梯度算法

    A proximal gradient algorithm for composite log-concave sampling

    [https://arxiv.org/abs/2605.12461](https://arxiv.org/abs/2605.12461)

    提出了一种针对复合对数凹分布的近端梯度采样算法，通过结合 $f$ 的梯度评估与 $g$ 的受限高斯预言机，在 $f+g$ 为 $\alpha$-凸且 $f$ 为 $\beta$-光滑的条件下，以 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$ 次预言机调用实现总变差距离下 $\varepsilon$ 精度的采样，匹配了 $g=0$ 情形下的最优结果。

    

    我们提出了一种从 $\mathbb{R}^d$ 上的复合对数凹分布中进行采样的算法，即形如 $\pi\propto e^{-f-g}$ 的密度，假设我们能够获得 $\alpha_f$-凸函数 $f$ 的梯度评估，以及 $\alpha_g$-凸函数 $g$（其中 $\alpha_g \ge 0$）的受限高斯预言机（RGO）。后一要求意味着我们可以容易地从密度 $\text{RGO}_{g,h,y}(x) \propto \exp(-g(x) -\frac{1}{2h}||y-x||^2)$ 中采样，这相当于 $g$ 的近端算子在采样上的类比。如果 $f + g$ 是 $\alpha$-凸的且 $f$ 是 $\beta$-光滑的，我们的采样器在总变差距离下达到 $\varepsilon$ 误差，所需的预言机调用次数为 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$，其中 $\kappa := 1 \vee \beta/\alpha$，$0\le \alpha_g \le \beta$ 且 $\alpha := \alpha_f + \alpha_g > 0$，这与之前 $g=0$ 情形下最先进的结果相匹配。我们进一步将结果扩展到以下情形：(1) ...（摘要在此处被截断）

    arXiv:2605.12461v2 Announce Type: replace-cross  Abstract: We propose an algorithm to sample from composite log-concave distributions over $\mathbb{R}^d$, i.e., densities of the form $\pi\propto e^{-f-g}$, assuming access to gradient evaluations of $\alpha_f$-convex $f$ and a restricted Gaussian oracle (RGO) for $\alpha_g$-convex $g$, where $\alpha_g \ge 0$. The latter requirement means that we can easily sample from the density $\text{RGO}_{g,h,y}(x) \propto \exp(-g(x) -\frac{1}{2h}||y-x||^2)$, which is the sampling analogue of the proximal operator for $g$. If $f + g$ is $\alpha$-convex and $f$ is $\beta$-smooth, our sampler achieves $\varepsilon$ error in total variation distance using $\widetilde{\mathcal O}(\kappa \sqrt d \log^4(1/\varepsilon))$ oracle calls where $\kappa := 1 \vee \beta/\alpha$, with $0\le \alpha_g \le \beta$ and $\alpha := \alpha_f + \alpha_g > 0$, which matches prior state-of-the-art results for the case $g=0$. We further extend our results to cases where (1) $
    
[^76]: 数字人格何时能可靠地近似人类调查结果？

    When Can Digital Personas Reliably Approximate Human Survey Findings?

    [https://arxiv.org/abs/2605.10659](https://arxiv.org/abs/2605.10659)

    基于LISS面板的研究表明，大语言模型驱动的数字人格能在分布层面较好地近似人类调查回答（尤其在涉及稳定属性和价值观的领域），但在个体预测和恢复受访者多变量结构方面仍存在根本局限。

    

    由大语言模型驱动的数字人格日益被提议作为人类调查受访者的替代品，但它们何时能够可靠地近似人类调查结果仍不清楚。我们利用LISS面板回答这一问题：从受访者的背景变量和2023年之前的调查历史中构建数字人格，然后将其与同一受访者在截止日期之后保留的答案进行对照测试。在四种人格架构、三个大语言模型和两个预测任务中，我们在问题、受访者、分布、公平性和聚类等多个层面评估其表现。数字人格在人类回答分布的对齐上有所改善，尤其是在与稳定属性和价值观相关的领域，但在个体预测方面仍然有限，且无法恢复受访者的多变量结构。检索增强架构带来了最明显的收益，但其性能更多取决于人类回答的结构而非（模型本身）（摘要原文在此处截断）。

    arXiv:2605.10659v3 Announce Type: replace  Abstract: Digital personas powered by Large Language Models (LLMs) are increasingly proposed as substitutes for human survey respondents, yet it remains unclear when they can reliably approximate human survey findings. We answer this question using the LISS panel, constructing personas from respondents' background variables and pre-2023 survey histories, then testing them against the same respondents' held-out post-cutoff answers. Across four persona architectures, three LLMs, and two prediction tasks, we assess performance at the question, respondent, distributional, equity, and clustering levels. Digital personas improve alignment with human response distributions, especially in domains tied to stable attributes and values, but remain limited for individual prediction and fail to recover multivariate respondent structure. Retrieval-augmented architectures provide the clearest gains, but performance depends more on human response structure th
    
[^77]: 属性化高斯图匹配的尖锐统计阈值

    Sharp Statistical Thresholds for Attributed Gaussian Graph Matching

    [https://arxiv.org/abs/2603.23305](https://arxiv.org/abs/2603.23305)

    本文针对边权重与节点特征联合相关的属性化高斯图匹配问题，推导出了精确恢复、几乎精确恢复和部分恢复的精确信息论阈值，并发现节点特征信息的引入使得精确恢复阈值从共享阈值中分离，呈现出比标准图匹配更丰富的相变结构。

    

    我们研究网络匹配问题，其中边权重和节点特征都按照特定的节点对应关系相关联。虽然先前的工作已经为单独匹配图或特征分别建立了尖锐阈值，但联合设置的情形却很少被探索。在本研究中，我们推导了底层对齐的精确恢复、几乎精确恢复和部分恢复的精确信息论阈值。这些阈值由图和特征的相关强度、节点数量以及特征维度来表示。有趣的是，尽管在标准图匹配场景中观察到“全有或全无”的相变现象，但节点级上下文信息的引入带来了更丰富的结构：精确恢复阈值从几乎精确恢复和部分恢复的共享阈值中分离出来。填补了先前工作留下的空白，我们提供……

    arXiv:2603.23305v2 Announce Type: replace-cross  Abstract: We investigate the problem of matching networks where both edge weights and node features are correlated following a specific node correspondence. While prior work has established sharp thresholds for matching either graphs or features individually, the joint setting has been significantly less explored. In this study, we derive precise information-theoretic thresholds for exact recovery, almost-exact recovery, and partial recovery of the underlying alignment. These are expressed in terms of graph and feature correlation strengths, the number of nodes, and the dimension of the features. Interestingly, whereas an all-or-nothing phase transition is observed in the standard graph-matching scenario, the inclusion of contextual information at the node level introduces a richer structure: the exact-recovery threshold separates from the shared threshold for almost-exact and partial recovery. Closing gaps left by prior works, we provid
    
[^78]: 基于未来状态-动作访问度量的最大熵探索

    Maximum-Entropy Exploration with Future State-Action Visitation Measures

    [https://arxiv.org/abs/2603.18965](https://arxiv.org/abs/2603.18965)

    该论文提出了一种与未来状态-动作特征折扣分布的熵成比例的内在奖励方法，理论上证明其期望总和是从初始状态出发的轨迹所访问分布熵的下界且可通过收缩算子的不动点性质进行离策略估计，从而实现了更优的最大熵探索。

    

    最大熵强化学习激励智能体探索状态和动作以最大化某个分布的熵，通常做法是提供与该熵函数成比例的额外内在奖励。在本文中，我们研究了与未来时间步中所访问的状态-动作特征的折扣分布熵成比例的内在奖励。该方法基于两个理论结果：首先，我们证明了这些内在奖励的期望总和是从初始状态出发的轨迹中所访问状态-动作特征折扣分布的熵的下界，我们将其与另一种最大熵目标联系起来；其次，我们证明了内在奖励定义中使用的分布是某个收缩算子的不动点，因此可以进行离策略估计。实验表明，新目标通常能够带来对特征更好的访问覆盖。

    arXiv:2603.18965v2 Announce Type: replace  Abstract: Maximum entropy reinforcement learning motivates agents to explore states and actions to maximize the entropy of some distribution, typically by providing additional intrinsic rewards proportional to that entropy function. In this paper, we study intrinsic rewards proportional to the entropy of the discounted distribution of state-action features visited during future time steps. This approach is motivated by two results. First, we show that the expected sum of these intrinsic rewards is a lower bound on the entropy of the discounted distribution of state-action features visited in trajectories starting from the initial states, which we relate to an alternative maximum entropy objective. Second, we show that the distribution used in the intrinsic reward definition is the fixed point of a contraction operator and can therefore be estimated off-policy. Experiments highlight that the new objective often leads to improved visitation of f
    
[^79]: 流匹配对于序列数据仅仅是轨迹重放吗？

    Is Flow Matching Just Trajectory Replay for Sequential Data?

    [https://arxiv.org/abs/2602.08318](https://arxiv.org/abs/2602.08318)

    本文推导了序列数据上流匹配目标的最优速度场，揭示其本质是一个非参数、记忆增强的连续时间动力系统，其速度场可表示为观测转移的瞬时速度的加权混合，为神经流匹配模型提供可解释视角并提出FreeFM。

    

    流匹配在科学领域中被越来越多地用于时间序列的生成与预测，而这些数据往往源自底层的动力系统。然而，目前尚不清楚流匹配究竟是学习到了可迁移的动力学结构，还是仅仅执行了一种有效的“轨迹重放”。我们通过推导经验流匹配目标在序列数据上所对应的最优速度场来研究这一问题。对于实践中常用的高斯条件路径，我们证明由此得到的采样器是一个常微分方程（ODE），其动力学描述了一个非参数的、记忆增强的连续时间动力系统。该驱动速度场具有闭式表达式，即由观测到的状态转移所确定的瞬时速度的加权混合。这一刻画为神经流匹配模型提供了一种可解释的视角——将其视为对底层非参数目标的参数化估计——并自然引出了FreeFM：一种...

    arXiv:2602.08318v3 Announce Type: replace-cross  Abstract: Flow matching (FM) is increasingly used in scientific domains for time series generation and forecasting, where data often arise from underlying dynamical systems. However, it is not well-understood whether it learns transferable dynamical structure or simply performs an effective ''trajectory replay''. We study this question by deriving the optimal velocity field targeted by the empirical FM objective on sequential data. For the Gaussian conditional paths commonly used in practice, we show that the resulting sampler is an ODE whose dynamics describe a nonparametric, memory-augmented continuous-time dynamical system. The driving velocity field admits a closed-form expression as a weighted mixture of instantaneous velocities determined by observed transitions. This characterization provides an interpretable view of neural FM models as parametric estimation of an underlying nonparametric target and naturally leads to FreeFM: a cl
    
[^80]: 超越可持续发展目标（SDG）加速器：面向适应性可持续发展组合的定向证据

    Beyond Sustainable Development Goal (SDG) Accelerators: Directed Evidence for Adaptive Sustainability Portfolios

    [https://arxiv.org/abs/2601.20875](https://arxiv.org/abs/2601.20875)

    本研究基于114个国家2000至2024年的面板数据，结合面板格兰杰因果检验与局部投影方法，识别出84个定向SDG联系（40个协同、44个权衡），证明不存在普适的SDG加速器，并主张依据定向因果证据构建适应性的可持续发展政策组合。

    

    政府和发展伙伴常常寻求可持续发展目标（SDG）“加速器”，希望其进展能将收益传播至人类福祉、制度、资源和环境优先事项等各领域。这类主张需要关于方向、时机、稳健性和权衡的证据，而不仅仅是网络连通性。我们利用2000年至2024年间对114个国家进行年度观测的平衡面板数据，绘制了定向SDG相互依赖关系。由于目标序列具有持续性、趋势性和横截面依赖性，我们将针对一阶差分序列的Dumitrescu-Hurlin面板格兰杰非因果检验与覆盖272个定向目标配对的错误发现控制相结合，并采用带Driscoll-Kraay标准误的面板局部投影法分析31个基于理论推导的指标联系。校正后我们发现84个定向联系：40个协同效应和44个权衡关系。这一密集结构不支持存在普适性加速器。驱动-接收排名（原文在此处截断）

    arXiv:2601.20875v3 Announce Type: replace-cross  Abstract: Governments and development partners often seek Sustainable Development Goal (SDG) "accelerators" whose progress can propagate gains across human well-being, institutions, resources, and environmental priorities. Such claims require evidence on direction, timing, robustness, and trade-offs, not only network connectivity. We map directed SDG interdependence using a balanced panel of 114 countries observed annually from 2000 to 2024. Because the goal series are persistent, trending, and cross-sectionally dependent, we combine Dumitrescu-Hurlin panel Granger non-causality tests on first-differenced series with false-discovery control across 272 directed goal pairs, and panel local projections with Driscoll-Kraay standard errors for 31 theory-derived indicator linkages. We find 84 directed linkages after correction: 40 synergies and 44 trade-offs. This dense structure does not support a universal accelerator. Driver-receiver rankin
    
[^81]: Sig-DEG用于蒸馏：让扩散模型更快更轻量

    Sig-DEG for Distillation: Making Diffusion Models Faster and Lighter

    [https://arxiv.org/abs/2508.16939](https://arxiv.org/abs/2508.16939)

    本文提出Sig-DEG，利用部分签名与循环结构逼近随机微分方程的解来蒸馏预训练扩散模型，在保持有竞争力的样本质量的同时显著提升推理速度。

    

    扩散模型实现了最先进的生成性能，但在推理阶段的计算成本依然高昂，通常需要大量的离散化步骤。我们提出了Sig-DEG（基于签名的微分方程生成器），一种用于蒸馏预训练扩散模型的生成器。受高阶随机泰勒展开的启发，Sig-DEG将总结子区间上布朗运动信息的部分签名与循环结构相结合，以逼近随机微分方程的解，从而实现通用性。我们将蒸馏问题表述为监督学习，在粗时间网格上匹配细分辨率的教师轨迹。在推理阶段，Sig-DEG能够实现快速生成，因为部分签名可以从其已知分布中被精确模拟，无需细粒度的布朗路径。在图像和时间序列生成任务上的实验表明，该方法在样本质量具有竞争力的同时，推理速度可大幅提升。

    arXiv:2508.16939v2 Announce Type: replace  Abstract: Diffusion models achieve state-of-the-art generative performance but remain computationally expensive at inference, often requiring many discretization steps. We propose Sig-DEG (Signature-based Differential Equation Generator), a generator for distilling pre-trained diffusion models. Inspired by high-order stochastic Taylor expansions, Sig-DEG combines partial signatures that summarize Brownian motion over subintervals with a recurrent structure to approximate stochastic differential equation solutions to achieve universality. We formulate distillation as supervised learning, matching fine-resolution teacher trajectories on a coarse time grid. At inference, Sig-DEG enables fast generation, as the partial signatures can be simulated exactly from their known distributions without requiring fine-grained Brownian paths. Experiments on image and time-series generation demonstrate competitive sample quality with inference speedups of up t
    
[^82]: GLL：一种面向神经网络的可微图学习层

    GLL: A Differentiable Graph Learning Layer for Neural Networks

    [https://arxiv.org/abs/2412.08016](https://arxiv.org/abs/2412.08016)

    该论文通过伴随方法推导出反向传播方程，提出了可微图学习层GLL，实现了图学习算法与神经网络的端到端精确集成，从而在分类任务中利用样本间的关系信息。

    

    标准的用于分类任务的深度学习架构通过投影头和softmax激活函数生成标签预测。尽管这些方法取得了成功，但它们未能利用样本之间的关系信息来生成标签预测。在近期的研究中，基于图的学习技术（即拉普拉斯学习）已被启发式地与神经网络结合，用于有监督和半监督学习（SSL）任务。然而，先前的工作要么对损失函数关于图学习算法的梯度进行近似，要么将这两个过程解耦处理，未能实现与神经网络的端到端集成。在这项工作中，我们通过伴随方法推导了反向传播方程，从而能够将一类通用的图学习层引入神经网络。所得方法不同于图神经网络，使我们能够精确地集成相似度图构建与...

    arXiv:2412.08016v3 Announce Type: replace  Abstract: Standard deep learning architectures used for classification generate label predictions with a projection head and softmax activation function. Although successful, these methods fail to leverage the relational information between samples for generating label predictions. In recent works, graph-based learning techniques, namely Laplace learning, have been heuristically combined with neural networks for both supervised and semi-supervised learning (SSL) tasks. However, prior works approximate the gradient of the loss function with respect to the graph learning algorithm or decouple the processes; end-to-end integration with neural networks is not achieved. In this work, we derive backpropagation equations, via the adjoint method, for inclusion of a general family of graph learning layers into a neural network. The resulting method, distinct from graph neural networks, allows us to precisely integrate similarity graph construction and 
    
[^83]: 噪声感知的差分隐私变分推断

    Noise-Aware Differentially Private Variational Inference

    [https://arxiv.org/abs/2410.19371](https://arxiv.org/abs/2410.19371)

    提出了一种基于随机梯度变分推断的噪声感知差分隐私近似贝叶斯推断新方法，突破了以往方法仅适用于简单概率模型的局限，可应用于高维和非共轭模型，并提出了一种更准确的噪声感知后验评估方法。

    

    差分隐私（DP）为统计推断提供了强大的隐私保障，但这可能导致下游应用中出现不可靠的结果和偏差。虽然已有若干噪声感知方法被提出，可以将DP扰动整合到推断过程中，但这些方法仅限于特定类型的简单概率模型。在本工作中，我们提出了一种基于随机梯度变分推断的新型噪声感知近似贝叶斯推断方法，该方法还可应用于高维和非共轭模型。此外，我们还提出了一种更准确的噪声感知后验评估方法。实验表明，在现有方法适用的领域内，我们的推断方法具有与现有方法相似的性能。在此领域之外，我们在高维贝叶斯线性回归中获得了准确的覆盖率，并在使用UCI Adult数据集的贝叶斯逻辑回归中获得了良好校准的预测概率。

    arXiv:2410.19371v4 Announce Type: replace-cross  Abstract: Differential privacy (DP) provides robust privacy guarantees for statistical inference, but this can lead to unreliable results and biases in downstream applications. While several noise-aware approaches have been proposed which integrate DP perturbation into the inference, they are limited to specific types of simple probabilistic models. In this work, we propose a novel method for noise-aware approximate Bayesian inference based on stochastic gradient variational inference which can also be applied to high-dimensional and non-conjugate models. We also propose a more accurate evaluation method for noise-aware posteriors. Empirically, our inference method has similar performance to existing methods in the domain where they are applicable. Outside this domain, we obtain accurate coverages on high-dimensional Bayesian linear regression and well-calibrated predictive probabilities on Bayesian logistic regression with the UCI Adult
    
[^84]: 贝尔曼符合推断：时间序列预测中预测区间的校准

    Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series

    [https://arxiv.org/abs/2402.05203](https://arxiv.org/abs/2402.05203)

    贝尔曼符合推断（BCI）是一个框架，通过解决一维随机控制问题，利用多步预测来提供校准的时间序列预测区间。BCI在任意分布转换和时间依赖性下实现了长期覆盖，且在波动率预测问题上生成更短的预测区间。

    

    我们引入了贝尔曼符合推断（BCI），这是一个围绕任何时间序列预测模型的框架，可以提供校准的预测区间。与现有方法不同，BCI能够利用多步预测，并通过在每个时间步骤上解决一维随机控制问题（SCP）来显式优化平均区间长度。特别地，我们使用动态规划算法来找到SCP的最优策略。我们证明了在任意分布转换和时间依赖性下，BCI能够实现长期覆盖，即使多步预测较差。我们在实证中发现，与现有方法相比，BCI避免了无信息区间（长度无限）的生成，并在波动率预测问题上生成了明显更短的预测区间。

    We introduce Bellman Conformal Inference (BCI), a framework that wraps around any time series forecasting models and provides calibrated prediction intervals. Unlike the existing methods, BCI is able to leverage multi-step ahead forecasts and explicitly optimize the average interval lengths by solving a one-dimensional stochastic control problem (SCP) at each time step. In particular, we use the dynamic programming algorithm to find the optimal policy for the SCP. We prove that BCI achieves long-term coverage under arbitrary distribution shifts and temporal dependence, even with poor multi-step ahead forecasts. We find empirically that BCI avoids uninformative intervals that have infinite lengths and generates substantially shorter prediction intervals on volatility forecasting problems when compared with existing methods.
    

