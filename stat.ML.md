# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Likelihood-free inference with nuisance parameters through normalizing flows](https://arxiv.org/abs/2609.10534) | 该论文提出了一种基于神经网络的归一化流分解方法，能在存在讨厌参数的情况下自动发现近枢轴统计量以实现无似然推断，其检验功效优于传统方法且速度显著更快。 |
| [^2] | [A positive resolution of the gap-entropy conjecture](https://arxiv.org/abs/2609.10529) | 本文正面解决了固定置信度最优臂识别中的gap-熵猜想，证明最优期望采样复杂度在常数因子内等于 $H(\log(1/\delta)+\mathrm{Ent}(I))$，并给出了一个与实例无关的匹配算法。 |
| [^3] | [An Exponential Deterministic--Randomized Gap in ERM-Oracle Complexity for Thresholds on an Unknown Order](https://arxiv.org/abs/2609.10196) | 该论文针对未知全序上阈值的直推在线学习，证明了在固定的自然ERM预言机下，任何确定性学习器的错误数与调用次数之和至少为T−ε，而随机学习器仅需O(log T)的期望调用与错误，从而首次给出了该设置下确定性与随机化学习之间指数级的预言机复杂度分离。 |
| [^4] | [A statistical approach to bias in zero-shot learning: the lens of handwriting recognition](https://arxiv.org/abs/2609.10084) | 本文提出一种统计方法，将传统GZSL特征学习器视为黑盒并纠正其在判别数据点训练状态（已见/未见）时的固有偏差，从而实现了超大词汇表上的零样本手写单词识别。 |
| [^5] | [Optimal Value Inference for Reinforcement Learning](https://arxiv.org/abs/2609.09981) | 该论文提出了一种基于Neyman正交性的去偏估计方法，通过softmax近似的自诱导贝尔曼方程构建冗余参数，实现了强化学习中最优价值的有效统计推断，且在视界发散和行为策略随时间变化的情况下依然保持渐近正态性。 |
| [^6] | [Adversarial Training for Tabular Credit Scoring: A Multi-Attack Robustness Evaluation in P2P Lending](https://arxiv.org/abs/2609.09945) | 该论文针对P2P借贷信用评分构建了系统性对抗鲁棒性基准，评估逻辑回归、前馈神经网络和表格Transformer三种模型在FGSM、PGD、椒盐噪声、DeepFool及混合攻击下的防御泛化能力。 |
| [^7] | [FlowCPO: A Unified Divergence View of Preference Alignment for Flow Models](https://arxiv.org/abs/2609.09905) | 提出FlowCPO，一种基于统一散度框架的离线前向KL偏好对齐目标，以有界的对比流匹配损失作为可求解替代，使流模型无需在线采样即可同时利用偏好与非偏好样本完成对齐。 |
| [^8] | [Beyond Conventional Federated Learning via High-Order Regularization](https://arxiv.org/abs/2609.09904) | 提出HiFedProx方法，用尺度匹配的幂型正则化器（p≥2）替代FedProx的二次惩罚，有效压缩联邦学习中客户端参数位移幅度的差异，从而更好地控制异常大的客户端移动。 |
| [^9] | [A Unifying Perspective on Probabilities as Model Predictions](https://arxiv.org/abs/2609.09855) | 本文提出“每个概率都是预测方法的输出”这一统一视角，消解了贝叶斯与频率学派之争，指出包括看似客观概率在内的所有概率都依赖于模型，并给出了在有限事件集合上进行可靠决策所需的有限校准准则。 |
| [^10] | [Muon-C: Operator-Aligned Muon for Convolutional Kernels](https://arxiv.org/abs/2609.09676) | 提出Muon-C，一种算子对齐的Muon优化器，通过将卷积核动量表示为频率域中的通道转移矩阵并独立极化，其优化保证在理论上优于标准展开方法，并在CIFAR-10流匹配任务上取得了更优的生成效果。 |
| [^11] | [Why Learning Rediscovers the Closed-Form Diagonal Regularizer](https://arxiv.org/abs/2609.09656) | 本文提出对角饱和原理，证明当截断噪声各向同性时，贝叶斯最优的Tikhonov正则化是仅由先验决定的闭式幂律，因此学习得到的对角正则化器难以稳健超越闭式解，在声学房间逆问题中闭式解已接近最优。 |
| [^12] | [Distillation of Synthetic Data for Time Series Foundation Models](https://arxiv.org/abs/2609.09586) | 本文提出合成数据蒸馏（SDD）方法，通过将时间序列基础模型的输出与每条轨迹的条件预测分布而非已实现的未来值进行比较来构建预训练损失目标，该方法在理论上可证明降低随机梯度协方差，并在实证中使400万至25亿参数规模的模型验证损失收敛更快。 |
| [^13] | [Learning with Synthetic Data via SGD in High-Dimensional Linear Regression](https://arxiv.org/abs/2609.09572) | 本研究通过高维线性回归中的理论分析发现，混合使用合成数据会导致不可避免的强模型坍塌，而两阶段训练策略（仅在第一阶段使用合成数据）可以避免风险下限，证明模型坍塌并非不可避免。 |
| [^14] | [High-probability guarantees for linear accessibility in feature superposition](https://arxiv.org/abs/2609.09556) | 该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。 |
| [^15] | [Oracle Complexity of Stochastic Fixed-Point Equations with Nonexpansive Maps](https://arxiv.org/abs/2609.09524) | 本文提出一种基于递归锚定技术的算法，为非扩张映射的随机不动点问题建立了 $\tilde O(\sigma^2 \epsilon^{-3} + \epsilon^{-1})$ 的Oracle复杂度上界，并给出了近乎匹配的复杂度下界。 |
| [^16] | [Recovery Theory for Projected Power Iterations in Permutation Synchronization](https://arxiv.org/abs/2609.09502) | 该论文首次为置换同步中的投影幂迭代法建立了严格的恢复理论，证明了在稀疏均匀污染模型下，仅需一次迭代即可将收敛域内（包括数据依赖的）估计的块误差收缩至可忽略水平，并给出了收缩因子和误差下限的精确刻画。 |
| [^17] | [Gaussian Approximation for Multivariate Martingale Sums from Uniformly Ergodic Markov Chains](https://arxiv.org/abs/2609.09480) | 本文为一致遍历马尔可夫链生成的多元鞅和建立了高阶 Wasserstein 距离下的显式高斯逼近界，并首次在平衡增量情形及马尔可夫链加性泛函中获得最优的 $O(n^{-1/2})$ 逼近率。 |
| [^18] | [Mode Coverage in Normalizing Flow Boltzmann Generators via Log-Ratio Variation](https://arxiv.org/abs/2609.09473) | 本文提出基于对数比变化的新损失函数KLXX，通过在前向KL散度中加入目标加权项和混合样本加权项，解决归一化流玻尔兹曼生成器因训练样本偏差或遗漏而导致的模式缺失问题，实现更全面的模式覆盖。 |
| [^19] | [MiNCE: Nonparametric, Strongly Consistent Confidence Envelopes for Band-Limited Functions and their Smoothed Spectra](https://arxiv.org/abs/2609.09436) | 本文提出了最小范数置信包络（MiNCE）框架，首次证明了该框架在无噪声和有噪声观测模型下所得置信带的强一致一致性，并将其扩展到频域，为带限函数及其平滑谱构建了非渐近、同步、强一致一致的置信带。 |
| [^20] | [Tensor-Train Weak SINDy: Identifying High-Dimensional Nonlinear Dynamics](https://arxiv.org/abs/2609.09434) | 本文提出TT-WSINDy方法，通过张量列车格式结合MANDy和WSINDy技术，实现对指数增长的候选函数空间的高效搜索，从而避免维数灾难并完成高维非线性动力学的数据驱动识别。 |
| [^21] | [Tensor Network Moral Graph Recovery of Discrete Probability Distributions](https://arxiv.org/abs/2609.09258) | 该论文提出使用带核范数正则化键修正的全连接张量网络从离散概率分布中恢复因果图的道德图，并证明了在忠实性等条件下零重构误差的最优张量网络的有效图恰好等于道德图。 |
| [^22] | [Accountable and uncertainty-aware evaluation of sensor-based AI under distribution shift: devices, subjects, and nearly three years underground](https://arxiv.org/abs/2609.09257) | 本文提出一种分阶段、可问责且量化不确定性的传感器AI评估协议，通过依次留出设备、受试者和时间（近三年）的四个累积泛化阶段，并以5%分位数决策规则取代均值判断，从而系统性地揭示分布偏移导致的性能退化。 |
| [^23] | [CAST: Canonical Approximate Schur Tree for Approximate Cholesky on Graphs](https://arxiv.org/abs/2609.09255) | 提出CAST方法，用从消去主元顶点所产生的稠密Schur补完全子图中直接采样的加权随机生成树（通过包含概率倒数重加权以保证无偏且与邻居排序无关）替代该完全子图，从而实现图上更高效的近似Cholesky预条件分解。 |
| [^24] | [What Fixed-Rollout pass@k Evaluations Can Identify](https://arxiv.org/abs/2609.09245) | 该论文证明固定 n 次 rollout 的成功计数只能识别任务成功率分布的前 n 个矩，因此 pass@k 仅在 k ≤ n 时可识别，任何超出采样预算的外推在原理上都无法确定。 |
| [^25] | [Critical initialization destabilizes higher input derivatives in wide scalar-input networks](https://arxiv.org/abs/2609.09244) | 该论文揭示了临界初始化虽能使宽网络一阶输入扰动的方差与深度无关，但会使高阶输入导数方差随深度线性增长而失稳，同时证明了分支尺度为 L^{-1/2} 的残差网络可使任意固定有限阶导数的方差一致有界。 |
| [^26] | [A Subsampled Davis-Kahan Bound for Large-Scale Eigenspace Estimation](https://arxiv.org/abs/2609.09211) | 本文提出基于独立伯努利抽样的子采样Davis-Kahan界，揭示了大规模特征子空间估计中计算代价与统计误差随抽样概率变化的权衡关系。 |
| [^27] | [Tail-Likelihood Reinforcement Learning](https://arxiv.org/abs/2609.02987) | 提出TailRL方法，通过最大化策略超过随机选择的奖励阈值的对数概率来直接优化对高奖励结果的覆盖能力，使罕见的高奖励输出在梯度中获得更大权重，从而解决平均奖励优化无法衡量生成式策略产生稀有高奖励输出概率差异的问题。 |
| [^28] | [Recovering Expert Critic-Sourced Network Adjacency between Musical Artists from Acoustic Distributions: A Construct-Validity Approach](https://arxiv.org/abs/2608.27291) | 本文通过构念效度方法，验证了乐评人来源的艺术家网络关联是否基于声学内容而非社会背景，从而为该信号在音乐推荐中的外部有效性提供了证据。 |
| [^29] | [CardioState-JEPA: Delay-Aware Cross-Modal Learning of a Shared Cardiac Representation](https://arxiv.org/abs/2608.12944) | 本文提出CardioState-JEPA，一种利用延迟感知跨模态预测架构联合学习ECG、PPG和PCG共享心脏表征的基础模型，以捕捉跨传感器的共同生理状态。 |
| [^30] | [Online Learning of Scale Parameters in Score-Driven Filters](https://arxiv.org/abs/2608.09218) | 该论文提出将得分驱动滤波器中的尺度参数（增益）视为决策变量进行在线学习，并发现加速递归中的负乘积得分反馈等价于预测损失的随机梯度，从而提供了一种新的变分视角。 |
| [^31] | [Dead Directions: Geometric Singular Learning](https://arxiv.org/abs/2606.05957) | 本文提出“死方向”这一基本概念，无需Hironaka奇点解析即可在原始坐标系中通过Fisher度量的衰减速率恢复KL阶数，从而架起信息几何与奇异学习理论之间的桥梁，将Fisher度量退化与Watanabe的实对数典范阈值不变量联系起来。 |
| [^32] | [Unbiased and Biased Variance-Reduced Forward-Reflected-Backward Splitting Methods for Stochastic Composite Inclusions](https://arxiv.org/abs/2603.15576) | 本文首次提出了一个能够同时处理无偏和有偏估计器的方差缩减框架，并将其应用于前向-反射-后向分裂方法以求解随机复合包含问题，实现了期望残差平方范数的O(1/k)收敛速率。 |
| [^33] | [RL unknotter, hard unknots and unknotting number](https://arxiv.org/abs/2603.07955) | 本文开发了基于强化学习的纽结图简化流水线，智能体通过学习Reidemeister移动策略成功解开“非常困难”的平凡纽结图，并通过自改进的工作簿驱动扩展系统性地改进了素纽结解结数的上界。 |
| [^34] | [Manifold-Aligned Generative Transport](https://arxiv.org/abs/2602.19600) | 提出MAGT方法，通过低维基础分布到数据空间的直接传输实现单次求值生成，在控制支撑集外质量的同时给出流形上的内在密度和极小极大最优的Wasserstein收敛保证。 |
| [^35] | [Theoretical Analysis of Measure Consistency Regularization for Partially Observed Data](https://arxiv.org/abs/2602.01437) | 本文从神经网络距离的角度对测度一致性正则化（MCR）进行了理论分析，证明在理想插值与相容性条件下，MCR能够获得更有利的有限样本估计误差上界。 |
| [^36] | [DFNN: A Deep Fr\'echet Neural Network Framework for Learning Metric-Space-Valued Responses](https://arxiv.org/abs/2510.17072) | 本文提出深度Fréchet神经网络（DFNN）框架，通过最小化Fréchet风险来逼近条件Fréchet均值，实现了从欧几里得预测变量到度量空间值响应的端到端回归预测，并为其建立了通用逼近定理。 |
| [^37] | [New Accelerated Past-Extragradient Methods with Variance Reduction for Generalized Equations](https://arxiv.org/abs/2508.16791) | 该论文提出了一种结合Nesterov加速与方差缩减技术的新型过去额外梯度算法框架，用于求解含非单调算子的广义方程，实现了O(1/k²)的期望收敛速率以及更快的o(1/k²)几乎必然收敛速率。 |
| [^38] | [Multi-fidelity batch Bayesian optimization for bioprocess development across scales](https://arxiv.org/abs/2508.10970) | 本文提出一种多保真度批量贝叶斯优化框架，通过集成定制的高斯过程与混合变量优化，在每次迭代中同时推荐实验条件、工艺尺度和生物催化剂选择，从而加速生物工艺开发并降低实验成本。 |
| [^39] | [Gaussian Processes and Reproducing Kernel Hilbert Spaces: Connections and Equivalences](https://arxiv.org/abs/2506.17366) | 本专著系统揭示了高斯过程与再生核希尔伯特空间在回归、插值等核心任务中的深层等价关系，并提出了基于高斯希尔伯特空间与RKHS等价性的统一理论框架，以促进两个研究领域的交叉融合。 |
| [^40] | [Variance-Reduced Fast Krasnoselkii-Mann Methods for Finite-Sum Root-Finding Problems](https://arxiv.org/abs/2406.02413) | 提出了带有新型无偏方差缩减估计器的单循环快速Krasnoselkii-Mann方法用于求解有限和共强制方程，实现了 $\mathcal{O}(1/k^2)$ 和 $o(1/k^2)$ 的最后迭代收敛速率，并以 $\mathcal{O}(n + n^{2/3}\epsilon^{-1})$ 的oracle复杂度达到 $\epsilon$-解。 |
| [^41] | [A Farewell to the Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized Machine Learning](https://arxiv.org/abs/2109.02355) | 本文综述了过参数化机器学习理论，解释了为何高度过参数化的模型（从线性模型到深度神经网络）能够在完美拟合噪声训练数据的同时仍具有良好的泛化能力，以及双重下降现象如何挑战了传统的偏差-方差权衡教义。 |
| [^42] | [Regularized Estimation and Feature Selection in Mixtures of Generalized Linear Experts](https://arxiv.org/abs/1907.06994) | 该论文提出了一个正则化最大似然框架，通过L1惩罚和近端Newton-EM算法，在广义线性专家混合模型中同时实现参数估计与特征选择，并统一支持高斯、泊松和多项式响应。 |

# 详细

[^1]: 通过归一化流实现含讨厌参数的无似然推断

    Likelihood-free inference with nuisance parameters through normalizing flows

    [https://arxiv.org/abs/2609.10534](https://arxiv.org/abs/2609.10534)

    该论文提出了一种基于神经网络的归一化流分解方法，能在存在讨厌参数的情况下自动发现近枢轴统计量以实现无似然推断，其检验功效优于传统方法且速度显著更快。

    

    我们提出了一种基于神经网络的归一化流的简单分解方法，该方法在存在讨厌参数的情况下，仅基于感兴趣分布的样本生成器，就能自然地发现一个枢轴统计量（或接近枢轴的统计量）。我们证明该统计量在p值与均匀分布之间平均KL散度最小的意义上是近似枢轴的，并论证了当统计量的维度等于参数的维度时，可以期望它具有良好的检验功效。该方法能够融入关于群不变性的先验知识，例如平移和尺度不变性。它几乎可以精确地重新发现单样本t检验，在受限方差比范围内的最坏情况检验水平上优于Welch检验，并在部分双列相关上实现了良好的校准，同时在中小样本上比轮廓似然比技术展现出更高的检验功效（且速度快得多）。

    arXiv:2609.10534v1 Announce Type: cross  Abstract: We present a simple decomposition of a neural-network-based normalizing flow that naturally uncovers a pivotal statistic (or something close) in the presence of nuisance parameters, based only on a sample generator from the distribution of interest. We show that the statistic is near-pivotal in the sense of minimum average KL-divergence of its $p$-values versus uniform and we argue that it can be expected to have good power when the dimension of the statistic equals the dimension of the parameter. It is able to incorporate prior knowledge about group invariances such as translation and scale. It can discover the one-sample $t$-test almost exactly, outperforms the Welch test in terms of worst-case size over a constrained variance-ratio range and achieves good calibration on partial biserial correlations, while showing higher power (and being much faster) on small-to-moderate samples than profile likelihood-ratio techniques.
    
[^2]: gap-熵猜想的正面解决

    A positive resolution of the gap-entropy conjecture

    [https://arxiv.org/abs/2609.10529](https://arxiv.org/abs/2609.10529)

    本文正面解决了固定置信度最优臂识别中的gap-熵猜想，证明最优期望采样复杂度在常数因子内等于 $H(\log(1/\delta)+\mathrm{Ent}(I))$，并给出了一个与实例无关的匹配算法。

    

    我们在固定置信度最优臂识别问题中证明了gap-熵猜想，该问题考虑独立单位方差的高斯臂，其均值位于 $[0,1]$ 区间内，且存在唯一的最优臂。对于每个次优臂 $i$，设 $\Delta_i=\mu_*-\mu_i$ 为其与最优均值的差距，并记 $H=\sum_{i\ne *}\Delta_i^{-2}$。设 $p_r$ 为满足 $2^{-(r+1)}<\Delta_i\le2^{-r}$ 的臂对 $H$ 的贡献比例，并定义 $\mathrm{Ent}(I)=\sum_{r:p_r>0} p_r\log(1/p_r)$。在所有能在每个高斯实例上以至少 $1-\delta$ 的概率识别出最优臂的算法中，给定实例上的最优期望采样次数（对臂标签的所有排列取平均）与 $H(\log(1/\delta)+\mathrm{Ent}(I))$ 相差至多常数倍。此外，存在一个与实例无关的算法，其期望采样次数被该量的常数倍加上 $g^{-2}\log\log(e^e/g)$ 所界定，其中 $g=\min_{i\ne *}$……

    arXiv:2609.10529v1 Announce Type: cross  Abstract: We prove the gap-entropy conjecture for fixed-confidence best-arm identification with independent unit-variance Gaussian arms, means in $[0,1]$, and a unique optimal arm. For each suboptimal arm $i$, let $\Delta_i=\mu_*-\mu_i$ be its gap from the optimal mean, and write $H=\sum_{i\ne *}\Delta_i^{-2}$. Let $p_r$ be the fraction of $H$ contributed by arms with $2^{-(r+1)}<\Delta_i\le2^{-r}$, and let $\mathrm{Ent}(I)=\sum_{r:p_r>0} p_r\log(1/p_r)$. Among all algorithms that identify the optimal arm with probability at least $1-\delta$ on every Gaussian instance, the optimal expected number of samples on a given instance, averaged over all permutations of the arm labels, is within absolute constant factors of $H(\log(1/\delta)+\mathrm{Ent}(I))$. Moreover, there is an algorithm, independent of the instance, whose expected number of samples is bounded by a constant multiple of this quantity plus $g^{-2}\log\log(e^e/g)$, where $g=\min_{i\ne *
    
[^3]: 未知序上阈值问题的ERM-预言机复杂度中确定性与随机化之间的指数级差距

    An Exponential Deterministic--Randomized Gap in ERM-Oracle Complexity for Thresholds on an Unknown Order

    [https://arxiv.org/abs/2609.10196](https://arxiv.org/abs/2609.10196)

    该论文针对未知全序上阈值的直推在线学习，证明了在固定的自然ERM预言机下，任何确定性学习器的错误数与调用次数之和至少为T−ε，而随机学习器仅需O(log T)的期望调用与错误，从而首次给出了该设置下确定性与随机化学习之间指数级的预言机复杂度分离。

    

    Attias、Hanneke和Ramaswami（NeurIPS 2025）提出了这样一个问题：当学习问题类只能通过预言机访问时，随机化是否能够被证明可以减少在线学习所需的预言机调用次数。我们研究了他们特别指出的实例：在T个实例构成的未知全序上的阈值的直推式在线学习，使用一致性类型的ERM预言机，该预言机返回一个与所查询带标签集合一致的完整概念（或报告不可实现性）。我们的主要结果是针对一个固定的自然预言机的分离结论。当预言机为最小前缀规则（或最大前缀规则）时，任何确定性学习器在某个实例上都会犯M次错误并进行Q次调用，满足 M+Q ≥ T−ε（其中 ε∈{0,1}，取决于空前缀是否为一个概念），且该常数是精确的；因此，O(log T) 次错误的代价是 T−ε−O(log T) 次调用，而该论文提出的随机学习器则在相同设置下实现了 O(log T) 的期望调用次数和错误次数。

    arXiv:2609.10196v1 Announce Type: cross  Abstract: Attias, Hanneke and Ramaswami (NeurIPS 2025) asked whether randomization provably reduces the oracle calls needed for online learning when the class is accessible only through an oracle. We study the instance they singled out: transductive online learning of thresholds on an unknown total order of T instances, with a consistency-type ERM oracle that returns a full concept consistent with a queried labeled set (or reports non-realizability). Our main result is a separation for a fixed natural oracle. When the oracle is the minimal-prefix rule (or the maximal-prefix rule), every deterministic learner makes M mistakes and Q calls with $M+Q\ge T-\varepsilon$ on some instance ($\varepsilon\in\{0,1\}$, according to whether the empty prefix is a concept), and the constant is exact; hence $O(\log T)$ mistakes cost $T-\varepsilon-O(\log T)$ calls, whereas that paper's randomized learner achieves $O(\log T)$ expected calls and mistakes under the
    
[^4]: 一种解决零样本学习偏差的统计方法：以手写识别为视角

    A statistical approach to bias in zero-shot learning: the lens of handwriting recognition

    [https://arxiv.org/abs/2609.10084](https://arxiv.org/abs/2609.10084)

    本文提出一种统计方法，将传统GZSL特征学习器视为黑盒并纠正其在判别数据点训练状态（已见/未见）时的固有偏差，从而实现了超大词汇表上的零样本手写单词识别。

    

    广义零样本学习（GZSL）已成为视觉识别系统的重要范式，这些系统必须泛化到训练期间未观察到的类别。传统的GZSL技术受限于其仅能适用于数量相对较少的未见类别，其可扩展性面临挑战，原因在于其众所周知的、倾向于训练期间已见类别的误分类偏差。在这项工作中，我们通过超大词汇表上零样本手写单词识别的视角来研究GZSL范式。我们提出了一种纠正这种偏差的统计方法，该方法将任何经典的GZSL特征学习器视为一个黑盒机制，其在对典型数据点的训练状态（已见 vs. 未见）进行识别时存在固有偏差，我们旨在纠正这种偏差，这类似于分布外推断问题。我们的方法利用了一个简单的两阶段分层架构，结合了经典的...

    arXiv:2609.10084v1 Announce Type: new  Abstract: Generalized zero-shot learning (GZSL) has emerged as an important paradigm for visual recognition systems that must generalize to classes that were not observed during training. Traditional GZSL techniques are limited by their applicability to a relatively small number of such unseen classes, scalability beyond which is challenging due to its well-known misclassification bias towards classes observed during training. In this work, we investigate the GZSL paradigm through the lens of zero-shot handwritten word recognition over extremely large vocabularies. We propose a statistical approach to rectifying this bias, which views any classical GZSL feature learner as a black box mechanism whose intrinsic bias in identifying the training status (seen vs. unseen) of a typical data point we aim to correct, similar to an out of distribution inferential problem. Our method leverages a simple two-stage hierarchical architecture, combining a classic
    
[^5]: 强化学习中的最优价值推断

    Optimal Value Inference for Reinforcement Learning

    [https://arxiv.org/abs/2609.09981](https://arxiv.org/abs/2609.09981)

    该论文提出了一种基于Neyman正交性的去偏估计方法，通过softmax近似的自诱导贝尔曼方程构建冗余参数，实现了强化学习中最优价值的有效统计推断，且在视界发散和行为策略随时间变化的情况下依然保持渐近正态性。

    

    我们研究强化学习中离线推断最优价值的问题。我们将两个新的冗余参数推导为自诱导贝尔曼方程的不动点，其中我们用最大贝尔曼算子的softmax对应形式进行近似。我们通过Neyman正交性提出了一个去偏估计量，并在视界发散的情况下建立了其渐近正态性，即使行为策略随时间变化，只要这些冗余参数达到许多机器学习方法所能实现的统计收敛速率即可。我们为这些冗余参数提供了具体的估计程序，并证明它们能够带来有效的统计推断。合成实验验证了我们推断方法的数值性能，我们还在现实决策问题中实现了该方法，包括自行车重新调配和AI智能体工具使用。

    arXiv:2609.09981v1 Announce Type: new  Abstract: We study offline inference for the optimal value in reinforcement learning. Two new nuisances are derived as fixed points of a self-induced Bellman equation, in which we approximate the maximum Bellman operator by its softmax correspondence. We propose a debiased estimator through the Neyman orthogonality and establish its asymptotic normality under diverging horizons even when the behavior policy changes with time, as long as the nuisances have the statistical rates that can be achieved by many machine learning methods. We provide a concrete estimating procedure for these nuisances and show they can lead to valid inference. Synthetic experiments validate the numerical performance of our inference method, and we implement it in real-life decision-making problems, including bike repositioning and AI agentic tool use.
    
[^6]: 面向表格信用评分的对抗训练：P2P借贷中的多攻击鲁棒性评估

    Adversarial Training for Tabular Credit Scoring: A Multi-Attack Robustness Evaluation in P2P Lending

    [https://arxiv.org/abs/2609.09945](https://arxiv.org/abs/2609.09945)

    该论文针对P2P借贷信用评分构建了系统性对抗鲁棒性基准，评估逻辑回归、前馈神经网络和表格Transformer三种模型在FGSM、PGD、椒盐噪声、DeepFool及混合攻击下的防御泛化能力。

    

    基于机器学习的信用评分在点对点（P2P）借贷中日益成为核心，然而其对抗操纵的韧性——即申请人策略性地篡改自报输入以获取有利的贷款决策——目前仍缺乏充分理解。现有的对抗鲁棒性证据大多来自图像和文本领域，且通常仅评估单一攻击与匹配防御的组合，对于防御方法如何在表格信用数据上跨攻击类型进行泛化几乎没有提供指导。我们通过在一个大型Lending Club数据子集上构建系统性的训练-测试鲁棒性基准来应对这一问题，涵盖三种模型家族（逻辑回归、前馈神经网络以及面向表格数据的Transformer），以及四种仅限于申请人可自行修改特征的攻击方法：快速梯度符号法（FGSM）、投影梯度下降（PGD）、椒盐噪声和DeepFool，此外还引入了一种混合攻击机制。在通过分层交叉验证评估的完整网格中……

    arXiv:2609.09945v1 Announce Type: cross  Abstract: Machine learning-based credit scoring is increasingly central to Peer-to-Peer (P2P) lending, yet its resilience to adversarial manipulation, where applicants strategically alter self-reported inputs to secure favourable decisions, remains poorly understood. Most adversarial-robustness evidence comes from image and text domains and evaluates a single attack against a matching defence, offering little guidance on how defences generalise across attack types in tabular credit data. We address this with a systematic train-test robustness benchmark on a large Lending Club subset, spanning three model families (logistic regression, a feed-forward neural network, and a transformer for tabular data) and four attacks confined to applicant-mutable features: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), Salt-and-Pepper (S&P) noise, and DeepFool, plus a mixed-attack regime. Across a full grid evaluated with stratified cross-va
    
[^7]: FlowCPO：流模型偏好对齐的统一散度视角

    FlowCPO: A Unified Divergence View of Preference Alignment for Flow Models

    [https://arxiv.org/abs/2609.09905](https://arxiv.org/abs/2609.09905)

    提出FlowCPO，一种基于统一散度框架的离线前向KL偏好对齐目标，以有界的对比流匹配损失作为可求解替代，使流模型无需在线采样即可同时利用偏好与非偏好样本完成对齐。

    

    针对流模型和扩散模型的偏好对齐目前涵盖了在线强化学习和离线偏好优化两大类方法，但这两类方法之间的关系仍不清晰。特别是，现有的前向过程对齐方法需要从当前模型获取新鲜样本，而基于固定偏好对的离线方法则主要依赖仅正样本微调或DPO式的似然比替代目标。我们通过一个基于散度的框架来统一组织这些方法，并提出了FlowCPO，这是一种离线的前向KL散度目标函数，无需在线采样即可同时利用偏好样本和非偏好样本。对于线性插值情形，我们在明确的正则性条件下证明，前向KL散度目标可被一个对比流匹配损失所界定，从而在固定数据上得到一个可求解的替代目标。我们进一步证明该损失是非负的，而简化版FlowDPO的带符号回归损失则可以无下界。

    arXiv:2609.09905v1 Announce Type: new  Abstract: Preference alignment for flow and diffusion models now spans online reinforcement learning and offline preference optimization, but the relation between these methods remains unclear. In particular, existing forward-process alignment methods require fresh samples from the current model, while offline methods based on fixed preference pairs rely primarily on positive-only fine-tuning or DPO-style likelihood-ratio surrogates. We organize these approaches through a divergence-based framework and introduce FlowCPO, an offline forward-KL objective that uses both preferred and dispreferred samples without online rollouts. For linear interpolation, we show under explicit regularity conditions that the forward-KL objective is bounded by a contrastive flow matching loss, yielding a tractable surrogate on fixed data. We further show that this loss is nonnegative, whereas the signed regression loss of simplified FlowDPO can be unbounded below. In t
    
[^8]: 通过高阶正则化超越传统联邦学习

    Beyond Conventional Federated Learning via High-Order Regularization

    [https://arxiv.org/abs/2609.09904](https://arxiv.org/abs/2609.09904)

    提出HiFedProx方法，用尺度匹配的幂型正则化器（p≥2）替代FedProx的二次惩罚，有效压缩联邦学习中客户端参数位移幅度的差异，从而更好地控制异常大的客户端移动。

    

    执行多个本地优化步骤的联邦学习客户端，其返回的参数位移幅度可能差异巨大。FedProx的二次正则化随位移线性增长，因此对于普通客户端移动与异常大客户端移动之间的对比，其控制能力有限。我们在此提出HiFedProx，它用一个尺度匹配的幂型正则化器（由指数p≥2标识）取代了二次惩罚。所有幂次在参考位移R处具有相同的正则化梯度幅值，而每个p>2在低于R时给出更弱的响应，在高于R时给出更强的响应。精确的仿射参考计算表明，增大p可以压缩相对位移差异，尽管非常大的幂次会趋近于固定半径行为并增加局部曲率。HiFedProx将这种几何特性与有限预算的随机客户端优化以及同小批量的Armijo回溯法相结合。

    arXiv:2609.09904v1 Announce Type: cross  Abstract: Federated clients that perform several local optimization steps can return parameter displacements with widely different magnitudes. The quadratic regularization of FedProx grows linearly with displacement and therefore offers limited control over the contrast between ordinary and unusually large client movements. We here introduce HiFedProx, which replaces the quadratic penalty with a scale-matched power-type regularizer indexed by $p\geq2$. All powers have the same regularization-gradient magnitude at a reference displacement $R$, while every $p>2$ gives a weaker response below $R$ and a stronger response above it. An exact affine reference calculation shows that increasing $p$ compresses relative displacement disparities, although very large powers approach fixed-radius behavior and increase local curvature. HiFedProx combines this geometry with finite-budget stochastic client optimization and same-minibatch Armijo backtracking. In 
    
[^9]: 概率作为模型预测的统一视角

    A Unifying Perspective on Probabilities as Model Predictions

    [https://arxiv.org/abs/2609.09855](https://arxiv.org/abs/2609.09855)

    本文提出“每个概率都是预测方法的输出”这一统一视角，消解了贝叶斯与频率学派之争，指出包括看似客观概率在内的所有概率都依赖于模型，并给出了在有限事件集合上进行可靠决策所需的有限校准准则。

    

    尽管概率性陈述无处不在，但关于其理解的根本分歧依然存在，贝叶斯学派与频率学派之间的争论便是例证；此外，何时以及为何基于概率采取行动才能真正带来理想结果，这一点仍不明确。在本文中，我们论证每一个概率都是某种“预测方法”的输出，也就是说，它既取决于构建抽象的特定方式，也取决于将这些抽象转化为预测的方式。通过这一视角，我们为 supposedly 不同类型的概率提供了一个统一的解释框架，并表明即使是看似客观的概率也依赖于模型。我们证明，当满足有限校准准则时，人们可以预判给定策略的效用分布，从而为有限事件集合上的成功决策提供依据。基于预测方法、归纳论证和概率演算的概念，我们解释了……的可行性

    arXiv:2609.09855v1 Announce Type: new  Abstract: Although probabilistic statements are ubiquitous, foundational disagreements persist about their understanding, as exemplified by debates between Bayesians and frequentists; moreover, it is unclear when and why acting on them actually leads to desirable outcomes. Here, we argue that every probability is the output of a \emph{prediction method}, that is, it depends on both a particular way of constructing abstractions and a way of transforming them into predictions. Through this, we provide a unifying perspective on supposedly different kinds of probabilities and show that even supposedly objective ones are model-dependent. We demonstrate that when a finite calibration criterion is met, one can anticipate the distribution of utilities for a given policy and inform successful decision-making on finite sets of events. Based on the notion of prediction methods, inductive arguments, and the probability calculus, we explain the feasibility of 
    
[^10]: Muon-C：面向卷积核的算子对齐Muon优化器

    Muon-C: Operator-Aligned Muon for Convolutional Kernels

    [https://arxiv.org/abs/2609.09676](https://arxiv.org/abs/2609.09676)

    提出Muon-C，一种算子对齐的Muon优化器，通过将卷积核动量表示为频率域中的通道转移矩阵并独立极化，其优化保证在理论上优于标准展开方法，并在CIFAR-10流匹配任务上取得了更优的生成效果。

    

    Muon优化器用近似正交的极化方向替代矩阵动量，但其几何结构依赖于矩阵的表示方式。对于卷积而言，标准的展开方法描述的是局部补丁映射，而非卷积算子本身。我们提出了Muon-C，一种算子对齐的优化器，它将卷积核动量表示为逐频率的通道转移矩阵，独立地对这些块进行极化处理，并利用临界傅里叶网格将更新精确地返回到原始有限卷积核的支撑域。我们证明了这种新的几何结构源于块划分与傅里叶坐标的结合。精确极化方向是在临界采样卷积范数下的线性最小化预言机。相对于连续卷积算子范数，其最坏情况保证从不弱于展开方法，且对于$3\times3$卷积核严格更强。在应用更新RMS匹配的CIFAR-10流匹配任务上，Muon-C在40(步)达到了9.87 FID。

    arXiv:2609.09676v1 Announce Type: cross  Abstract: Muon replaces matrix momentum with an approximately orthogonal polar direction, but its geometry depends on the matrix representation. For convolution, standard unfolding describes a local patch map rather than the convolution operator. We introduce Muon-C, an operator-aligned optimizer that represents kernel momentum as frequency-wise channel-transfer matrices, polarizes these blocks independently, and uses a critical Fourier grid to return updates exactly to the original finite kernel support. We show that the new geometry arises from combining the block partition and Fourier coordinates. The exact-polar direction is a linear minimization oracle under the critically sampled convolution norm. Its worst-case guarantee relative to the continuous convolution-operator norm is never weaker than unfolding and is strictly stronger for $3\times3$ kernels. On CIFAR-10 flow matching with matched applied-update RMS, Muon-C reaches 9.87 FID at 40
    
[^11]: 为什么学习会重新发现闭式对角正则化器

    Why Learning Rediscovers the Closed-Form Diagonal Regularizer

    [https://arxiv.org/abs/2609.09656](https://arxiv.org/abs/2609.09656)

    本文提出对角饱和原理，证明当截断噪声各向同性时，贝叶斯最优的Tikhonov正则化是仅由先验决定的闭式幂律，因此学习得到的对角正则化器难以稳健超越闭式解，在声学房间逆问题中闭式解已接近最优。

    

    我们在模态逆问题中确定了一个对角饱和原理：当截断噪声为各向同性时，贝叶斯最优的Tikhonov形状是一个闭式幂律 Gamma_k ∝ lambda_k^|s|，该形状仅由先验决定，与具体的域无关。Berry的随机波猜想使各模态之间的截断噪声去相关，而Weyl的特征值计数定律提供了足够多的模态数，使得即使在经验上违反Berry猜想的情况下，该结论依然成立。二者共同预测了在整个逐模态参数族上的损失景观近似平坦，因此留给对角正则化器稳健超越闭式解的空间非常有限。在FEM模拟的声学房间上，闭式解相对于逐房间的oracle调参在各观测窗口下都接近最优；并且在相同数据上训练的三种对角架构尽管学习到了性质截然不同的谱，其重建误差与闭式解的差距仍在1个百分点以内。该框架还可通过一个已知的扩展方式推广到热扩散问题……

    arXiv:2609.09656v1 Announce Type: new  Abstract: We identify a diagonal saturation principle in modal inverse problems: when truncation noise is isotropic, the Bayes-optimal Tikhonov shape is a closed-form power law Gamma_k proportional to lambda_k^|s| set by the prior alone, independent of the domain. Berry's random-wave conjecture decorrelates the truncation noise across modes, and Weyl's eigenvalue counting law supplies enough modes for the conclusion to survive empirical Berry violations. Together they predict an approximately flat loss landscape across the per-mode family, leaving narrow scope for a diagonal regularizer to robustly beat the closed form. On FEM-simulated acoustic rooms, the closed form is near-optimal relative to per-room oracle tuning across observation windows, and three diagonal architectures trained on the same data match its reconstruction error within 1 pp despite learning qualitatively different spectra. The framework extends to heat diffusion via a known ex
    
[^12]: 时间序列基础模型的合成数据蒸馏

    Distillation of Synthetic Data for Time Series Foundation Models

    [https://arxiv.org/abs/2609.09586](https://arxiv.org/abs/2609.09586)

    本文提出合成数据蒸馏（SDD）方法，通过将时间序列基础模型的输出与每条轨迹的条件预测分布而非已实现的未来值进行比较来构建预训练损失目标，该方法在理论上可证明降低随机梯度协方差，并在实证中使400万至25亿参数规模的模型验证损失收敛更快。

    

    时间序列基础模型（TSFMs）越来越多地在合成生成的时间序列轨迹上进行预训练，其中数据生成过程是已知的。当前的预训练方案基于将TSFM输出与每条轨迹的已实现未来值进行比较的损失目标。我们转而提出将TSFM输出与每条轨迹的条件预测分布进行比较的损失目标，我们将这一过程称为合成数据蒸馏（SDD）。SDD对应于训练目标的Rao-Blackwell化，即它在保持随机梯度期望不变的同时，在Loewner偏序意义下可证明地降低随机梯度的协方差。我们在参数规模从400万到25亿的TSFM模型家族上对SDD进行了实证验证，并观察到在每个模型规模上验证损失都收敛得更快：在高斯过程数据上，SDD达到或超越了现状损失方法的表现。

    arXiv:2609.09586v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) are increasingly pre-trained on synthetically generated time series trajectories, where the data generating process is known. Current pre-training recipes are based on loss objectives which compare TSFM outputs to realized future values of each trajectory. We instead propose loss objectives which compare TSFM outputs to the conditional forecast distribution of each trajectory, a procedure we call synthetic data distillation (SDD). SDD corresponds to a Rao-Blackwellization of the training objective, in that it leaves the expectation of stochastic gradients unchanged while provably reducing the covariance of the stochastic gradient under the Loewner partial ordering. We empirically validate SDD on a TSFM model family of sizes from $4$M to $2.5$B parameters, and observe faster convergence of validation loss at every model size: on Gaussian Process data, SDD attains or improves upon the Status Quo loss w
    
[^13]: 高维线性回归中基于合成数据的SGD学习

    Learning with Synthetic Data via SGD in High-Dimensional Linear Regression

    [https://arxiv.org/abs/2609.09572](https://arxiv.org/abs/2609.09572)

    本研究通过高维线性回归中的理论分析发现，混合使用合成数据会导致不可避免的强模型坍塌，而两阶段训练策略（仅在第一阶段使用合成数据）可以避免风险下限，证明模型坍塌并非不可避免。

    

    合成数据已成为突破有限人类生成数据限制、扩展模型训练规模的一种有前景的方法，但它也可能引发强烈的模型坍塌现象，即任何固定比例的合成数据都会阻止模型性能随数据规模扩大而提升，留下一个不可消失的额外风险下限。本文研究了在具有模型偏移的高维线性回归中，合成数据如何影响单遍SGD的泛化性能。我们为混合训练和两阶段训练建立了有限样本风险界，将标准的偏差和方差与源分布不匹配效应分离开来，具体包括混合训练下的波动和持续漂移，以及两阶段训练下的过滤初始化偏差。这些风险界揭示了鲜明的对比：混合训练会引发强模型坍塌，而两阶段训练通过仅在第一阶段使用合成数据避免了这一风险下限，这表明在简单的数据课程安排下模型坍塌并非不可避免。

    arXiv:2609.09572v1 Announce Type: new  Abstract: Synthetic data has become a promising way to scale model training beyond limited human-generated data but it may also induce strong model collapse (Dohmatob et al., 2024), where any fixed fraction of synthetic data prevents model performance from improving under data scaling, leaving a non-vanishing excess risk floor. In this paper, we study how synthetic data affects the generalization of one-pass SGD in high-dimensional linear regression with model shift. We establish finite-sample risk bounds for mixed and two-stage training, separating standard bias and variance from source-mismatch effects, namely fluctuation and persistent drift under mixing and filtered initialization bias under two-stage. These bounds reveal a sharp contrast: mixed training induces strong model collapse, while two-stage training avoids the floor by using synthetic data only in the first stage, showing that collapse is not inevitable under a simple data curriculum
    
[^14]: 特征叠加中线性能及性的高概率保证

    High-probability guarantees for linear accessibility in feature superposition

    [https://arxiv.org/abs/2609.09556](https://arxiv.org/abs/2609.09556)

    该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。

    

    神经网络可以利用特征叠加来编码比维度数量更多的概念，但特征间的交叉干扰限制了同时激活特征的线性能及性。通过将线性能及性建模为一个压缩感知问题，我们在次高斯噪声下针对固定支撑集推导出高概率界，证明了充分维度以线性方式扩展（d=O_ε(k log m)），而非此前最坏情况下的二次方限制。随后，我们通过高斯尾近似在各系统参数下验证了这些界。这些结果量化了线性表示假设的几何约束，为评估稀疏自编码器、组合泛化和神经网络可解释性提供了一个框架。

    arXiv:2609.09556v1 Announce Type: cross  Abstract: Neural networks can leverage feature superposition to encode more concepts than dimensions, but cross-feature interference constrains the linear accessibility of simultaneously active features. By framing linear accessibility as a compressed sensing problem, we derive high-probability bounds for fixed supports under subgaussian noise, proving the sufficient dimension scales linearly ($d=O_{\varepsilon}(k \log m)$) rather than prior worst-case quadratic limits. We then validate these bounds across system parameters through Gaussian-tail approximations. These results quantify the geometric constraints of the linear representation hypothesis, providing a framework for evaluating sparse autoencoders, compositional generalization, and neural interpretability.
    
[^15]: 非扩张映射的随机不动点方程的Oracle复杂度

    Oracle Complexity of Stochastic Fixed-Point Equations with Nonexpansive Maps

    [https://arxiv.org/abs/2609.09524](https://arxiv.org/abs/2609.09524)

    本文提出一种基于递归锚定技术的算法，为非扩张映射的随机不动点问题建立了 $\tilde O(\sigma^2 \epsilon^{-3} + \epsilon^{-1})$ 的Oracle复杂度上界，并给出了近乎匹配的复杂度下界。

    

    我们研究了在一般范数 $\|\cdot\|$ 和紧凸集上的自映射 $T$ 条件下，计算满足小不动点残差 $\|T(x)-x\| \leq \epsilon$ 的点的Oracle复杂度。我们在如下设置中研究该问题：$T$ 关于同一范数 $\|\cdot\|$ 是非扩张的，并通过方差有界为 $\sigma^2$ 的无偏随机Oracle进行访问。我们提供了一种算法，能够以高概率求解任何具有弱Rademacher型 $q > 1$ 的范数的此类实例。该算法基于递归锚定技术。对于2-型空间，例如 $p \in [2, \infty]$ 的 $\ell_p$ 空间，我们的算法达到了 $\tilde O(\sigma^2 \epsilon^{-3} + \epsilon^{-1})$ 的随机Oracle复杂度。我们进一步证明了高维空间中此类 $\ell_{\infty}$-范数实例的近乎匹配的下界（即在对数多项式因子内匹配）。我们的下界对任何成功概率有界的随机化算法均成立。

    arXiv:2609.09524v1 Announce Type: cross  Abstract: We study the oracle complexity of computing a point with small fixed-point residual $\|T(x)-x\| \leq \epsilon$, for a general norm $\|\cdot\|$ and a self-map $T$ of a compact convex set. We study this problem in the setting where $T$ is nonexpansive with respect to the same norm $\|\cdot\|$ and accessed via an unbiased stochastic oracle with bounded variance $\sigma^2$. We provide an algorithm that solves such instances for any norm with a weak Rademacher type $q > 1$, with high probability. The algorithm is based on a recursive anchoring technique. For type-$2$ spaces, such as $\ell_p$-spaces for $p \in [2, \infty]$, our algorithm attains stochastic oracle complexity $\tilde O(\sigma^2 \epsilon^{-3} + \epsilon^{-1})$. We further prove a near-matching lower bound (i.e., matching up to poly-log factors) for such $\ell_{\infty}$-norm instances in high dimensions. Our lower bound holds against any randomized algorithm that succeeds with c
    
[^16]: 置换同步中投影幂迭代法的恢复理论

    Recovery Theory for Projected Power Iterations in Permutation Synchronization

    [https://arxiv.org/abs/2609.09502](https://arxiv.org/abs/2609.09502)

    该论文首次为置换同步中的投影幂迭代法建立了严格的恢复理论，证明了在稀疏均匀污染模型下，仅需一次迭代即可将收敛域内（包括数据依赖的）估计的块误差收缩至可忽略水平，并给出了收缩因子和误差下限的精确刻画。

    

    我们研究了投影幂方法（PPM），用于在可能稀疏的均匀污染模型下同步 n 个关于 m 个对象的未知置换。每对对象以概率 p 被观测，被观测到的测量以概率 π₀ 未受污染，否则是一个独立的均匀随机置换。在 log m = o(npπ₀²) 的条件下，我们证明了：对于具有固定正多数正确块的独立估计，每个指定块都可以（以高概率）实现精确的一步恢复。当 np ≥ C₀ log n 且 m = o(npπ₀²) 时，我们证明了一个高概率事件可以同时对所有最优对齐误差不超过 0.5-ε 的估计实现块误差收缩。收缩因子为 O(m/(npπ₀²))，误差下限为 O(e^{-cnpπ₀}+e^{-cnpπ₀²}+log n/n)。因此，一次更新即可将该收敛域内的每一个（可能是数据依赖的）估计映射为可忽略的块误差。

    arXiv:2609.09502v1 Announce Type: new  Abstract: We study the projected power method (PPM) for synchronizing \(n\) unknown permutations of \(m\) objects under a possibly sparse uniform corruption model. Each pair is observed with probability \(p\), and an observed measurement is uncorrupted with probability \(\pi_0\) and is otherwise an independent uniform permutation. Under \(\log m=o(np\pi_0^2)\), we prove exact one-step recovery (with high probability) of each prescribed block for an independent estimate with a fixed positive majority of correct blocks. When \(np\ge C_0\log n\) and \(m=o(np\pi_0^2)\), we prove that one high-probability event yields a block-error contraction simultaneously for every estimate whose optimally aligned error is at most \(0.5-\epsilon\). The contraction factor is \(O(m/(np\pi_0^2))\) and the error floor is \(O(e^{-cnp\pi_0}+e^{-cnp\pi_0^2}+{\log n}/{n})\). Consequently, one update maps every possibly data-dependent estimate in this basin to vanishing bloc
    
[^17]: 一致遍历马尔可夫链生成的多元鞅和的高斯逼近

    Gaussian Approximation for Multivariate Martingale Sums from Uniformly Ergodic Markov Chains

    [https://arxiv.org/abs/2609.09480](https://arxiv.org/abs/2609.09480)

    本文为一致遍历马尔可夫链生成的多元鞅和建立了高阶 Wasserstein 距离下的显式高斯逼近界，并首次在平衡增量情形及马尔可夫链加性泛函中获得最优的 $O(n^{-1/2})$ 逼近率。

    

    我们为一致遍历马尔可夫链生成的多元鞅差之和，在高阶 Wasserstein 距离 $W_p$（$p\geq2$）下建立了高斯逼近界。在 $L^{(2+\eta)p}$ 矩条件（$\eta>0$）下，我们给出了显式界 $$ O\left( p^3 \|A\|_4^2 + pd^{1/4}\|A\|_2^{1/2}\|A\|_4^2 \right) $$ 其中 $A\in\mathbb{R}^n$ 汇总了 $n$ 个单个鞅增量的 $L^{(2+\eta)p}$ 规模。在平衡增量情形（即各个增量具有 $n^{-1/2}$ 量级的可比规模）下，该界给出了当 $p$ 与维度 $d$ 固定时首个最优的 $O(n^{-1/2})$ 高斯逼近率。由此，我们还首次获得了一致遍历马尔可夫链多元加性泛函的最优 $O(n^{-1/2})$ $W_p$ 高斯逼近率。我们的分析发展了两种技术，用以处理高阶 Wasserstein 距离与时间依赖性之间的相互作用。

    arXiv:2609.09480v1 Announce Type: cross  Abstract: We develop Gaussian approximation bounds in higher-order Wasserstein distance $W_p$, $p\geq2$, for sums of multivariate martingale differences generated by a uniformly ergodic Markov chain. Under an $L^{(2+\eta)p}$-moment condition with $\eta>0$, we establish the explicit bound $$ O\left( p^3 \|A\|_4^2 + pd^{1/4}\|A\|_2^{1/2}\|A\|_4^2 \right) $$ where $A\in\mathbb{R}^n$ collects the $L^{(2+\eta)p}$-sizes of the $n$ individual martingale increments. In the balanced-increment regime where the individual increments have comparable sizes of order $n^{-1/2}$, it yields the first optimal $O(n^{-1/2})$ Gaussian approximation rate for fixed $p$ and $d$. Consequently, we also obtain the first optimal $O(n^{-1/2})$ $W_p$ Gaussian approximation rate for multivariate additive functionals of uniformly ergodic Markov chains.   Our analysis develops two techniques for addressing the interplay between higher-order Wasserstein distance and temporal dep
    
[^18]: 基于对数比变化的归一化流玻尔兹曼生成器的模式覆盖

    Mode Coverage in Normalizing Flow Boltzmann Generators via Log-Ratio Variation

    [https://arxiv.org/abs/2609.09473](https://arxiv.org/abs/2609.09473)

    本文提出基于对数比变化的新损失函数KLXX，通过在前向KL散度中加入目标加权项和混合样本加权项，解决归一化流玻尔兹曼生成器因训练样本偏差或遗漏而导致的模式缺失问题，实现更全面的模式覆盖。

    

    归一化流玻尔兹曼生成器保留了可处理的推前密度，但使用前向KL散度进行训练依赖于目标样本，而这些样本可能存在偏差或遗漏某些模式。因此，流模型可能在遗漏目标质量的同时，其观测到的重要性权重仍然给出较高的有效样本量。我们引入对数比变化 $\X_\omega$，即在加权测度 $\omega$ 下目标与推前对数密度比的平均成对绝对差，并用它来定义KLXX，一种新的损失函数。两个对数比变化项被添加到前向KL散度中（由两个X表示）：其中一个由目标分布加权以提高精度，另一个由淬火与退火样本和推前样本的混合加权以搜索候选模式。我们推导了KLXX的Fisher-Rao梯度流，证明两个变化项都贡献非正的耗散，并为KLXX建立了固定代理误差界。我们在自适应分阶段玻尔兹曼生成器中应用KLXX，结合重要性采样……

    arXiv:2609.09473v1 Announce Type: new  Abstract: Normalizing flow Boltzmann generators retain a tractable pushforward density, but training with forward KL depends on target samples that may be biased or omit modes. As a result, a flow can miss target mass while its observed importance weights give a high effective sample size. We introduce the log-ratio variation $\X_\omega$, the mean absolute pairwise difference of the target-to-pushforward log-density ratio under a weighting measure $\omega$, and use it to define KLXX, a new loss function. Two log-ratio variations are added to the forward KL (denoted by the two X's): one weighted by the target to improve accuracy, the other by a mixture of quench and temper samples with pushforward samples to search candidate modes. We derive the Fisher--Rao gradient flow of KLXX, where both variations contribute nonpositive dissipation, and a fixed-surrogate error bound for KLXX. We use KLXX in an adaptive-staging Boltzmann generator, with importan
    
[^19]: MiNCE：带限函数及其平滑谱的非参数、强一致置信包络

    MiNCE: Nonparametric, Strongly Consistent Confidence Envelopes for Band-Limited Functions and their Smoothed Spectra

    [https://arxiv.org/abs/2609.09436](https://arxiv.org/abs/2609.09436)

    本文提出了最小范数置信包络（MiNCE）框架，首次证明了该框架在无噪声和有噪声观测模型下所得置信带的强一致一致性，并将其扩展到频域，为带限函数及其平滑谱构建了非渐近、同步、强一致一致的置信带。

    

    最小范数置信包络策略利用再生核希尔伯特空间（RKHS）理论，为带限函数构建非渐近的同步置信区域提供了一种非参数方法。虽然这些包络的有限样本覆盖保证已被建立，但其一致性至今尚未得到分析。在本文中，我们研究了这一构造（此处称为最小范数置信包络（MiNCE）框架），并在测量噪声的温和假设下，建立了所得置信带在无噪声和有噪声观测模型下的强一致一致性。我们进一步将这一构造扩展到频域，为平滑谱推导出非渐近、同步、强一致一致的置信带。在非参数回归和谱估计中的数值实验从经验上证实了我们的理论结果，说明了……

    arXiv:2609.09436v1 Announce Type: cross  Abstract: Minimum-norm confidence envelope strategies offer a nonparametric approach to constructing nonasymptotic, simultaneous confidence regions for band-limited functions, exploiting the theory of Reproducing Kernel Hilbert Spaces (RKHS). While the finite-sample coverage guarantees of these envelopes have been established, their consistency has not been analyzed so far. In this paper, we study this construction, here termed the Minimum-Norm Confidence Envelope (MiNCE) framework, and establish the strong uniform consistency of the resulting bands, both for noise-free and noisy observation models, under mild assumptions on the measurement noises. We further extend this formulation to the frequency domain, deriving nonasymptotic, simultaneous, strongly uniformly consistent confidence bands for the smoothed spectra. Numerical experiments in nonparametric regression and spectral estimation empirically confirm our theoretical results, illustrating
    
[^20]: 张量列车弱形式SINDy：识别高维非线性动力学

    Tensor-Train Weak SINDy: Identifying High-Dimensional Nonlinear Dynamics

    [https://arxiv.org/abs/2609.09434](https://arxiv.org/abs/2609.09434)

    本文提出TT-WSINDy方法，通过张量列车格式结合MANDy和WSINDy技术，实现对指数增长的候选函数空间的高效搜索，从而避免维数灾难并完成高维非线性动力学的数据驱动识别。

    

    近年来，弱形式方法在数据驱动的动力系统发现领域取得了重大进展。然而，在高维设置下，现有技术在计算和内存方面的代价可能十分高昂。在这项工作中，我们提出了TT-WSINDy方法，该方法结合了非线性动力学多维近似（MANDy）和非线性动力学弱稀疏识别（WSINDy）两种方法的技术，并以张量列车（TT）格式实现所需的计算。我们证明了该方法能够在指数增长的候选函数空间中进行搜索——执行弱形式变换、回归和稀疏化——而不会遭受维数灾难的影响。

    arXiv:2609.09434v1 Announce Type: cross  Abstract: In recent years, weak-form methods have made significant advances in data-driven discovery of dynamical systems. However, in high-dimensional settings, current techniques can prove expensive in both computation and memory. In this work, we introduce TT-WSINDy, which combines techniques of the Multidimensional Approximation of Nonlinear Dynamics (MANDy) and Weak Sparse Identification of Nonlinear Dynamics (WSINDy) methods, implementing requisite computations in the tensor-train (TT) format. We demonstrate that this method is able to search an exponentially-growing space of candidate functions -- performing weak-form transformation, regression, and sparsification -- without suffering from the curse of dimensionality.
    
[^21]: 离散概率分布的张量网络道德图恢复

    Tensor Network Moral Graph Recovery of Discrete Probability Distributions

    [https://arxiv.org/abs/2609.09258](https://arxiv.org/abs/2609.09258)

    该论文提出使用带核范数正则化键修正的全连接张量网络从离散概率分布中恢复因果图的道德图，并证明了在忠实性等条件下零重构误差的最优张量网络的有效图恰好等于道德图。

    

    我们提出了一种从离散变量上的概率分布中恢复因果有向无环图（DAG）道德图的方法，该方法使用具有核范数正则化键修正的全连接张量网络（FCTNs）。每个键矩阵被参数化为基线全1矩阵加上低秩修正 $C_{ij} = U_{ij}V_{ij}^\top$，通过对因子施加变分Frobenius范数惩罚来实现的修正项核范数会将不必要的键驱动至零。我们证明，在忠实性、正性以及局部张量架构上无隐式重路由假设的条件下，每个具有零重构误差（$\varepsilon = 0$）的最优FCTN的有效图恰好等于道德图。对于近似情形（$\varepsilon > 0$），我们利用条件互信息的Fannes-Audenaert连续性给出了显式的恢复界，并推导了正则化参数的充分条件。

    arXiv:2609.09258v1 Announce Type: new  Abstract: We present a method for recovering the moral graph of a causal DAG from a probability distribution over discrete variables, using fully connected tensor networks (FCTNs) with nuclear-norm-regularized bond corrections. Each bond matrix is parameterized as a baseline all-ones matrix plus a low-rank correction $C_{ij} = U_{ij}V_{ij}^\top$, and the nuclear norm of the correction implemented via the variational Frobenius norm penalty on the factors drives unnecessary bonds to zero. We prove that under faithfulness, positivity, and a no-implicit-rerouting assumption on the local tensor architecture, \textbf{every} optimal FCTN with zero reconstruction error $\varepsilon = 0$ has effective graph exactly equal to the moral graph. For the approximate regime ($\varepsilon > 0$), we provide explicit recovery bounds using the Fannes-Audenaert continuity of conditional mutual information, and derive a sufficient condition on the regularization parame
    
[^22]: 分布偏移下基于传感器的AI的可问责且不确定性感知评估：设备、受试者与近三年的地下数据

    Accountable and uncertainty-aware evaluation of sensor-based AI under distribution shift: devices, subjects, and nearly three years underground

    [https://arxiv.org/abs/2609.09257](https://arxiv.org/abs/2609.09257)

    本文提出一种分阶段、可问责且量化不确定性的传感器AI评估协议，通过依次留出设备、受试者和时间（近三年）的四个累积泛化阶段，并以5%分位数决策规则取代均值判断，从而系统性地揭示分布偏移导致的性能退化。

    

    基于传感器的AI系统很少在其训练时所处条件下运行：设备、人员和记录时期会发生变化，而每一种变化都会导致性能下降，这是随机训练-测试划分无法揭示的。我们提出了一种分阶段的、可问责的评估协议，将已部署模型的评估视为一种带有明确参考水平和量化不确定性的测量过程。四个累积的泛化阶段分别留出设备、受试者和时间。每个阶段都基于多次重复训练的分位数、并以具有正确类别数量的随机参考作为对照进行评判；一种“超出现有范围率”指标揭示了模型相对于训练时对部署中已不存在的类别产生的无声误导；明确的决策规则将推广部署决策与5%分位数而非均值挂钩。我们在两个真实场景中，使用基于智能手机的循环分类器在无基础设施的地磁定位任务上验证了该协议。

    arXiv:2609.09257v1 Announce Type: cross  Abstract: Sensor-based AI systems are rarely operated under the conditions under which they were trained: devices, personnel and recording epochs change, and each change degrades performance in ways a random train-test split cannot reveal. We propose a staged, accountable evaluation protocol that treats the evaluation of a deployed model as a measurement with declared reference levels and a quantified uncertainty. Four cumulative generalisation stages hold out devices, subjects and time. Each stage is judged on quantiles of repeated trainings against chance references with the correct class count, an out-of-present-scope rate exposes silent misdirection towards classes that are no longer present in deployment relative to training, and an explicit decision rule ties roll-out decisions not to means but to 5% quantiles. We demonstrate the protocol on infrastructure-free geomagnetic localisation with smartphone-based recurrent classifiers in two rea
    
[^23]: CAST：用于图上近似Cholesky分解的规范近似Schur树

    CAST: Canonical Approximate Schur Tree for Approximate Cholesky on Graphs

    [https://arxiv.org/abs/2609.09255](https://arxiv.org/abs/2609.09255)

    提出CAST方法，用从消去主元顶点所产生的稠密Schur补完全子图中直接采样的加权随机生成树（通过包含概率倒数重加权以保证无偏且与邻居排序无关）替代该完全子图，从而实现图上更高效的近似Cholesky预条件分解。

    

    图数据工作负载（如扩散估计、排名、半监督学习和网络优化）通常需要求解多个具有相同系数矩阵的拉普拉斯方程或对称对角占优M矩阵（SDDM）系统。近似Cholesky预条件子逐个消去顶点，并存储得到的稀疏近似分解（即“因子”），其构建成本可在多次求解中摊销。然而，消去一个顶点（即“主元”）会在其d个活跃邻居之间产生一个稠密的Schur补完全子图。我们提出CAST（规范近似Schur树），直接从该完全子图中采样一个加权随机生成树来替代它。每次采样结果都是连通的且恰好包含d-1条边，而通过树包含概率的倒数对每条选中的边进行重新加权，可使更新保持无偏。该分布与主元邻居的排序无关……

    arXiv:2609.09255v1 Announce Type: new  Abstract: Graph-data workloads such as diffusion estimation, ranking, semi-supervised learning, and network optimization often solve many Laplacian or symmetric diagonally dominant M-matrix (SDDM) systems with the same coefficient matrix. Approximate Cholesky preconditioners eliminate vertices one at a time and store the resulting sparse approximate factorization, the \emph{factor}, whose construction cost is amortized across these solves. But eliminating a vertex, the \emph{pivot}, creates a dense Schur-complement clique among its $d$ active neighbors. We introduce CAST (Canonical Approximate Schur Tree), which replaces this clique with a weighted random spanning tree sampled directly from it. Every realization is connected and contains exactly d-1 edges, while reweighting each selected edge by the reciprocal of its tree-inclusion probability makes the update unbiased. The distribution is independent of the ordering of the pivot neighbors, and we
    
[^24]: 固定 Rollout 次数的 pass@k 评估能识别什么

    What Fixed-Rollout pass@k Evaluations Can Identify

    [https://arxiv.org/abs/2609.09245](https://arxiv.org/abs/2609.09245)

    该论文证明固定 n 次 rollout 的成功计数只能识别任务成功率分布的前 n 个矩，因此 pass@k 仅在 k ≤ n 时可识别，任何超出采样预算的外推在原理上都无法确定。

    

    重复采样评估越来越多地将 pass@k 外推到远超每个问题实际采集样本数 n 的范围。我们证明，在合并/随机任务的条件二项模型中，固定 n 次的成功计数仅能识别潜在单任务成功率分布的 n 个自由矩。因此，当 k ≤ n 时，直接 pass@k 是可识别的；但当 k > n 时，一般的外推 pass@k、尾部指数和尾部常数均不可识别，即使在同一 rollout 预算下拥有任意多个可交换任务也是如此。这一结论比“常用估计量在超过 n 时无定义”的观察更强：它刻画了固定深度计数律实验所缺失的信息。我们给出了保持计数律但外推结果互不相容的精确构造，阐述了唯一可延拓的例外情形，并通过 Hausdorff 主表示计算了尖锐的总体可识别区间。在公开的每个问题 10,000 次 rollout 的数据上……（摘要原文在此处截断）

    arXiv:2609.09245v1 Announce Type: new  Abstract: Repeated-sampling evaluations increasingly extrapolate pass@k far beyond the number n of samples collected per problem. We show that, in the pooled/random-task conditional-Binomial model, fixed-n success counts identify only the n free moments of the latent per-task success distribution. Consequently, direct pass@k is identified for k <= n, but generic extrapolated pass@k, tail exponents, and tail constants are not identified for k > n, even with arbitrarily many exchangeable tasks at the same rollout budget. This is stronger than the observation that the usual estimator is undefined beyond n: it characterizes the information missing from the fixed-depth count-law experiment. We give exact count-law-preserving constructions with incompatible extrapolations, state the exceptional unique-extension case, and compute sharp population identified intervals through Hausdorff principal representations. On the public 10,000-rollout-per-problem re
    
[^25]: 临界初始化使宽标量输入网络中的高阶输入导数失稳

    Critical initialization destabilizes higher input derivatives in wide scalar-input networks

    [https://arxiv.org/abs/2609.09244](https://arxiv.org/abs/2609.09244)

    该论文揭示了临界初始化虽能使宽网络一阶输入扰动的方差与深度无关，但会使高阶输入导数方差随深度线性增长而失稳，同时证明了分支尺度为 L^{-1/2} 的残差网络可使任意固定有限阶导数的方差一致有界。

    

    混沌边缘条件能够在宽随机初始化网络中保持一阶输入扰动的稳定，但物理信息损失、分数匹配以及导数正则化都依赖于更高阶的输入导数。对于光滑的标量输入全连接网络，我们利用在每个固定深度下于无限宽度极限中成立的有限导数喷射的联合高斯性，推导出直至三阶的平均场递推关系，这些递推在方差不动点处是精确的，且有限深度修正呈几何级数衰减。在临界点上，一阶导数方差与深度无关，而只要激活函数具有非零曲率，二阶导数方差就会随深度线性增长。由此得到的三阶系统在平均场敏感度上封闭。对于分支尺度为 L^{-1/2} 的残差网络，我们证明了在明确的正则性假设下，每个固定的有限导数阶数都具有一致有界的方差。模拟……（原文此处截断）

    arXiv:2609.09244v1 Announce Type: new  Abstract: The edge-of-chaos condition preserves first-order input perturbations in wide randomly initialized networks, but physics-informed losses, score matching and derivative regularization depend on higher input derivatives. For smooth scalar-input fully connected networks, using a joint Gaussianity of the finite derivative jet that holds in the infinite-width limit at each fixed depth, we derive mean-field recursions through third order that are exact at the variance fixed point, with finite-depth corrections that decay geometrically. At criticality, the first-derivative variance is depth-invariant, whereas the second-derivative variance grows linearly whenever the activation has nonzero curvature. The resulting third-order system closes on mean-field susceptibilities. For residual networks with branch scale L^{-1/2}, we prove that every fixed finite derivative order has uniformly bounded variance under explicit regularity assumptions. Simula
    
[^26]: 大规模特征子空间估计的子采样Davis-Kahan界

    A Subsampled Davis-Kahan Bound for Large-Scale Eigenspace Estimation

    [https://arxiv.org/abs/2609.09211](https://arxiv.org/abs/2609.09211)

    本文提出基于独立伯努利抽样的子采样Davis-Kahan界，揭示了大规模特征子空间估计中计算代价与统计误差随抽样概率变化的权衡关系。

    

    Davis-Kahan定理是谱分析中的基本工具，它为对称矩阵及其扰动的特征子空间之间的距离提供了定量控制。然而，当矩阵维度很大时，计算主特征向量的计算代价非常高昂，这限制了谱方法在现代大规模应用中的实际使用。本文通过提出一种独立的伯努利抽样方案来解决这一问题，并证明了子采样矩阵的主左奇异向量能够忠实地逼近低秩对称矩阵的目标子空间。我们的主要结果是一个子采样Davis-Kahan界，它给出了直接依赖于抽样概率的显式误差界。该界揭示了如下权衡关系：计算代价与抽样概率呈线性关系，而统计误差则与抽样概率的平方根倒数成正比。

    arXiv:2609.09211v1 Announce Type: new  Abstract: The Davis-Kahan theorem is a fundamental tool in spectral analysis, providing quantitative control over the distance between the eigenspaces of a symmetric matrix and its perturbation. However, when the matrix dimension is large, computing leading eigenvectors is computationally expensive, limiting the practical use of spectral methods in modern large-scale applications. This paper addresses this problem by proposing an independent Bernoulli sampling scheme and proves that the leading left singular vectors of the subsampled matrix faithfully approximate the target subspace of a low-rank symmetric matrix. Our main result is a subsampled Davis-Kahan bound that gives an explicit error bound depending directly on the sampling probability. The bound reveals the trade-off: the computational cost scales linearly with the sampling probability, while the statistical error scales as the inverse square root of the sampling probability. Our result t
    
[^27]: 尾部似然强化学习

    Tail-Likelihood Reinforcement Learning

    [https://arxiv.org/abs/2609.02987](https://arxiv.org/abs/2609.02987)

    提出TailRL方法，通过最大化策略超过随机选择的奖励阈值的对数概率来直接优化对高奖励结果的覆盖能力，使罕见的高奖励输出在梯度中获得更大权重，从而解决平均奖励优化无法衡量生成式策略产生稀有高奖励输出概率差异的问题。

    

    强化学习通常优化平均奖励。对于生成式策略而言，平均值可能掩盖一个重要的区别：两个策略可以达到相同的平均奖励，但在产生罕见但高奖励的输出方面的机会却截然不同。随着训练和推理过程中采样数量的增加，这一点变得尤为重要，因为采样的收益取决于策略是否在高奖励结果上保留概率质量。我们提出直接优化这种覆盖能力。我们不仅考虑期望奖励，还考虑其所有的上尾部分：对于每个奖励阈值，策略超过该阈值的可能性有多大？这将连续的奖励转化为一族二元成功事件。我们提出了尾部似然强化学习，该方法最大化超过随机选择的奖励阈值的对数概率。其梯度对罕见的高奖励输出赋予更大的权重，并且可以解释为Best-of-(k)梯度的混合。

    arXiv:2609.02987v1 Announce Type: new  Abstract: Reinforcement learning typically optimizes average reward. For generative policies, the average can hide an important distinction: two policies can achieve the same mean reward while having very different chances of producing a rare but high-reward rollout. This matters as sampling increases during training and inference, since its benefit depends on retaining probability mass on high-reward outcomes. We propose to optimize this coverage directly. Rather than considering only expected reward, we consider all of its upper tails: for each reward threshold, how likely is the policy to exceed it? This turns a continuous reward into a family of binary success events. We introduce Tail-Likelihood Reinforcement Learning (TailRL), which maximizes the log-probability of exceeding a randomly chosen reward threshold. Its gradient gives more weight to rare, high-reward rollouts and can be interpreted as a mixture of Best-of-(k) gradients. TailRL req
    
[^28]: 从声学分布中恢复乐评人来源的音乐艺术家网络关联：一种构念效度方法

    Recovering Expert Critic-Sourced Network Adjacency between Musical Artists from Acoustic Distributions: A Construct-Validity Approach

    [https://arxiv.org/abs/2608.27291](https://arxiv.org/abs/2608.27291)

    本文通过构念效度方法，验证了乐评人来源的艺术家网络关联是否基于声学内容而非社会背景，从而为该信号在音乐推荐中的外部有效性提供了证据。

    

    摘要：arXiv:2608.27291v1 公告类型：交叉 摘要：音乐推荐主要依赖两种信号：用户-物品交互，在冷启动场景下失效；以及内在音乐内容，适用于任何录音。我们认为，第三种很大程度上未被利用的信号既更丰富又更有原则性：批评性邻接，即当专家评论家在长篇散文中明确关联两位艺术家时建立的成对关系。它编码了关于哪些艺术家属于一起的深思熟虑的判断。先前工作确立了其内部效度，表明它能恢复连贯、可解释的社区，并在无用户数据的情况下，在用户满意度模拟中可与协同过滤匹敌。所缺失的是外部验证：这种来自评论家的关系是否根植于音乐本身，而非社会文化背景。我们将其与声学内容进行检验，将问题重新框定为构念效度问题。我们将艺术家表示为80个低级Es特征上的经验分布。

    arXiv:2608.27291v1 Announce Type: cross  Abstract: Music recommendation relies primarily on two signals: user-item interactions, which fail in the cold-start regime, and intrinsic musical content, available for any recording. We argue that a third, largely untapped signal is both richer and more principled: critical adjacency, the pairwise relation established when an expert critic explicitly links two artists in long-form prose. It encodes deliberate judgments about which artists belong together. Prior work established its internal validity, showing it recovers coherent, interpretable communities and can match collaborative filtering in user-satisfaction simulations, with no user data. What has been missing is external validation: whether this critic-sourced relation is grounded in the music itself versus sociological context. We test it against acoustic content, reframing the question as one of construct validity.   Representing artists as empirical distributions over 80 low-level Es
    
[^29]: CardioState-JEPA：延迟感知的跨模态共享心脏表征学习

    CardioState-JEPA: Delay-Aware Cross-Modal Learning of a Shared Cardiac Representation

    [https://arxiv.org/abs/2608.12944](https://arxiv.org/abs/2608.12944)

    本文提出CardioState-JEPA，一种利用延迟感知跨模态预测架构联合学习ECG、PPG和PCG共享心脏表征的基础模型，以捕捉跨传感器的共同生理状态。

    

    摘要：心电图（ECG）、光电容积脉搏波（PPG）和心音图（PCG）提供了同一心动周期的互补视角，然而现有的心脏基础模型仅针对单一传感模态进行训练，未利用跨传感器间的共享生理信息。我们提出了CardioState-JEPA，一种心脏基础模型，旨在跨ECG、PPG和PCG联合学习单一共享表征，其基于生理感知的联合嵌入预测架构构建。该模型将异质波形映射到统一的令牌空间，通过单一共享的Transformer编码器进行处理，并通过预测掩码的潜在心脏状态进行学习，将预训练目标聚焦于共享生理过程而非传感器特定的波形外观。为处理电、机械和血流动力学事件之间的时间偏移，跨模态预测采用了一个学习的延迟对齐器，以在相应的心脏时间点匹配信号。

    arXiv:2608.12944v1 Announce Type: new  Abstract: Electrocardiography (ECG), photoplethysmography (PPG), and phonocardiography (PCG) provide complementary views of the same cardiac cycle, yet existing cardiac foundation models are trained for a single sensing modality, leaving the shared physiology across sensors unexploited. We introduce CardioState-JEPA, a cardiac foundation model to learn a single shared representation jointly across ECG, PPG, and PCG, built on a physiology-aware joint-embedding predictive architecture. The model maps heterogeneous waveforms into a common token space, processes them with a single shared Transformer encoder, and learns by predicting masked latent cardiac states, placing the pretraining target on shared physiology rather than sensor-specific waveform appearance. To handle the temporal offsets between electrical, mechanical, and hemodynamic events, cross-modal prediction uses a learned delay aligner that matches signals at the corresponding cardiac time
    
[^30]: 在线学习得分驱动滤波器中的尺度参数

    Online Learning of Scale Parameters in Score-Driven Filters

    [https://arxiv.org/abs/2608.09218](https://arxiv.org/abs/2608.09218)

    该论文提出将得分驱动滤波器中的尺度参数（增益）视为决策变量进行在线学习，并发现加速递归中的负乘积得分反馈等价于预测损失的随机梯度，从而提供了一种新的变分视角。

    

    得分驱动滤波器通过将缩放的对数似然得分乘以一个控制更新幅度的尺度参数来更新时变参数。我们将这个尺度参数命名为增益，将其视为决策变量，并研究其在线学习。在当前状态、观测值、得分和缩放规则条件下，每个可接受的增益都会产生一个可达到的下一状态和一个一步超前预测密度；标量增益沿直线选择距离，而对角增益则选择坐标方向上的传输速率并可能改变方向。增益选择成为一个具有Kullback-Leibler目标的条件一步预测决策问题。我们的核心观察是，加速得分驱动递归中使用的负乘积得分反馈可以被解读为该预测损失的随机梯度，这提供了一种新的变分视角。因此，自适应增益学习可以被视为一种...

    arXiv:2608.09218v2 Announce Type: replace  Abstract: Score-driven filters update a time-varying parameter by multiplying a scaled log-likelihood score by a scale parameter that controls the magnitude of the update. We name this scale parameter gain, consider it a decision variable, and study its online learning. Conditional on the current state, observation, score, and scaling rule, each admissible gain induces a reachable next state and a one-step-ahead predictive density; a scalar gain selects distance along a line, whereas a diagonal gain selects coordinatewise transmission rates and may change direction. Gain selection becomes a conditional one-step predictive decision problem with a Kullback-Leibler objective. Our central observation is that the negative product-of-scores feedback employed in accelerated score-driven recursions can be read as the stochastic gradient of this predictive loss, offering a new variational perspective. Adaptive gain learning can therefore be viewed as a
    
[^31]: 死方向：几何奇异学习

    Dead Directions: Geometric Singular Learning

    [https://arxiv.org/abs/2606.05957](https://arxiv.org/abs/2606.05957)

    本文提出“死方向”这一基本概念，无需Hironaka奇点解析即可在原始坐标系中通过Fisher度量的衰减速率恢复KL阶数，从而架起信息几何与奇异学习理论之间的桥梁，将Fisher度量退化与Watanabe的实对数典范阈值不变量联系起来。

    

    奇异学习理论与信息几何研究的是同一类空间：前者在解析坐标系中进行，后者则在原始坐标系中依赖于非退化性假设——而过参数化模型恰恰违反了这一假设。本文沿着这两个理论之间桥梁的一个方向（即从Watanabe不变量到Fisher几何），通过一个基本概念——“死方向”——来展开：死方向是指Fisher度量沿其退化的单位向量，等价地，是穿过解析奇异集且KL散度在其上保持高阶零点的方向，其KL阶数由该散度消失的快慢程度决定。我们的核心结果是在原始坐标系中、无需借助Hironaka奇点解析的情况下，将KL阶数恢复为方向性Fisher二次型逼近奇异点时的衰减速率。光滑纤维上的一个选择规则将该速率转换为Watanabe对实对数典范阈值的单方向贡献，并且该恢复…

    arXiv:2606.05957v2 Announce Type: replace-cross  Abstract: Singular learning theory and information geometry study the same spaces: the former in resolved coordinates, the latter in original coordinates under a non-degeneracy assumption that overparameterised models violate. This paper carries one direction of the bridge between them, from Watanabe's invariants to Fisher geometry, through one primitive, the dead direction: a unit vector along which the Fisher metric degenerates, equivalently a direction crossing the analytic singular set along which the KL divergence keeps a zero of high order, its KL order set by how fast that divergence vanishes. Our central result recovers the KL order as the decay rate of the directional Fisher quadratic form approaching the singularity, in original coordinates, without a Hironaka resolution. A selection rule on smooth fibres translates this rate into Watanabe's single-direction contribution to the real log canonical threshold, and the recovery ext
    
[^32]: 面向随机复合包含问题的无偏与有偏方差缩减前向-反射-后向分裂方法

    Unbiased and Biased Variance-Reduced Forward-Reflected-Backward Splitting Methods for Stochastic Composite Inclusions

    [https://arxiv.org/abs/2603.15576](https://arxiv.org/abs/2603.15576)

    本文首次提出了一个能够同时处理无偏和有偏估计器的方差缩减框架，并将其应用于前向-反射-后向分裂方法以求解随机复合包含问题，实现了期望残差平方范数的O(1/k)收敛速率。

    

    本文为前向-反射-后向分裂（FRBS）方法开发了新的方差缩减技术，用于求解一类可能非单调的随机复合包含问题。与诸如小批量采样等无偏估计器不同，开发随机有偏变体面临着根本性的技术挑战，此前从未被应用于包含问题和不动点问题。我们通过设计一个能够同时处理无偏和有偏估计器的新框架来填补这一空白。我们的主要思想是为前向-反射方向构建随机方差缩减估计器，并利用这些估计器执行迭代更新。首先，我们提出了一类无偏方差缩减估计器，并证明递增小批量SGD、loopless-SVRG和SAGA估计器均属于该类。对于这些无偏估计器，我们建立了期望残差平方范数的 $\mathcal{O}(1/k)$ 最优迭代收敛速率……

    arXiv:2603.15576v2 Announce Type: replace-cross  Abstract: This paper develops new variance-reduction techniques for the forward-reflected-backward splitting (FRBS) method to solve a class of possibly nonmonotone stochastic composite inclusions. Unlike unbiased estimators such as mini-batching, developing stochastic biased variants faces a fundamental technical challenge and has not been utilized before for inclusions and fixed-point problems. We fill this gap by designing a new framework that can handle both unbiased and biased estimators. Our main idea is to construct stochastic variance-reduced estimators for the forward-reflected direction and use them to perform iterate updates. First, we propose a class of unbiased variance-reduced estimators and show that increasing mini-batch SGD, loopless-SVRG, and SAGA estimators fall within this class. For these unbiased estimators, we establish a $\mathcal{O}(1/k)$ best-iterate convergence rate for the expected squared residual norm, togeth
    
[^33]: 强化学习解结器、困难平凡纽结与解结数

    RL unknotter, hard unknots and unknotting number

    [https://arxiv.org/abs/2603.07955](https://arxiv.org/abs/2603.07955)

    本文开发了基于强化学习的纽结图简化流水线，智能体通过学习Reidemeister移动策略成功解开“非常困难”的平凡纽结图，并通过自改进的工作簿驱动扩展系统性地改进了素纽结解结数的上界。

    

    我们开发了一个用于简化纽结图的强化学习流水线。经过训练的智能体学习移动建议和价值启发式方法，以执行Reidemeister移动。该流水线适用于任意纽结和链环；我们在“非常困难”的平凡纽结图上对其进行测试，并利用图膨胀方法在 $4_1\#9_{10}$ 上进行测试，其中我们研究了最近确立的令人惊讶的解结数上界3。此外，我们阐述了一种基于工作簿驱动的自改进流水线扩展，该扩展系统地改进了素纽结的解结数上界。

    arXiv:2603.07955v4 Announce Type: replace-cross  Abstract: We develop a reinforcement learning pipeline for simplifying knot diagrams. A trained agent learns move proposals and a value heuristic for navigating Reidemeister moves. The pipeline applies to arbitrary knots and links; we test it on ``very hard'' unknot diagrams and, using diagram inflation, on $4_1\#9_{10}$ where we investigate the recently established and surprising upper bound of three for the unknotting number. In addition, we explain a self-improving workbook-driven extension of the pipeline that systematically improves unknotting number upper bounds on the prime knots.
    
[^34]: 流形对齐生成式传输

    Manifold-Aligned Generative Transport

    [https://arxiv.org/abs/2602.19600](https://arxiv.org/abs/2602.19600)

    提出MAGT方法，通过低维基础分布到数据空间的直接传输实现单次求值生成，在控制支撑集外质量的同时给出流形上的内在密度和极小极大最优的Wasserstein收敛保证。

    

    许多高维数据集集中在嵌入于环境空间中的低维结构附近。针对这类数据的生成模型必须在控制支撑集之外质量的同时保持计算上的可行性。扩散模型在推理时依赖迭代去噪，而标准流模型则要求可逆且保持维度的映射。我们提出了MAGT（流形对齐生成式传输），一种从低维基础分布到数据空间的直接传输方法。其核心目标是在选定的Gaussian平滑水平上比较数据得分与生成器诱导的得分。通过一个后验恒等式，该得分可以用隐变量的条件均值来表示，并通过对有限锚点集合进行自归一化重要性采样来近似。训练完成后，生成仅需对传输映射进行一次求值，且其像还携带相对于流形体积的内在密度。我们建立了极小极大最优的Wasserstein收敛界。

    arXiv:2602.19600v2 Announce Type: replace  Abstract: Many high-dimensional datasets concentrate near a low-dimensional structure embedded in the ambient space. Generative models for such data must control off-support mass while remaining computationally practical. Diffusion models use iterative denoising at inference, whereas standard normalizing flows require invertible, dimension-preserving maps. We propose MAGT (Manifold-Aligned Generative Transport), a direct transport from a low-dimensional base distribution to the data space. Its core objective compares the data and generator-induced scores at a selected Gaussian smoothing level. A posterior identity expresses this score through a latent conditional mean, which is approximated by self-normalized importance sampling over a finite anchor set. After training, generation requires one evaluation of the transport, whose image also carries an intrinsic density with respect to manifold volume. We establish a minimax-optimal Wasserstein c
    
[^35]: 部分观测数据下测度一致性正则化的理论分析

    Theoretical Analysis of Measure Consistency Regularization for Partially Observed Data

    [https://arxiv.org/abs/2602.01437](https://arxiv.org/abs/2602.01437)

    本文从神经网络距离的角度对测度一致性正则化（MCR）进行了理论分析，证明在理想插值与相容性条件下，MCR能够获得更有利的有限样本估计误差上界。

    

    数据损坏、特征缺失或模态缺失的问题持续困扰着现代机器学习领域。为了解决这一问题，一类在插补数据与完整观测数据之间强制执行一致性的正则化方法已成为提升模型泛化能力的一种有前景的途径，尤其是在部分观测的设置中。我们将这类方法称为测度一致性正则化（Measure Consistency Regularization，MCR）。尽管此类方法在图像修复、数据插补和半监督学习等各种应用中取得了实证上的成功，但人们对MCR理论基础的深入理解仍然有限。本文通过神经网络距离的视角，对MCR何时能够产生更有利的有限样本估计误差上界提供了理论见解，从而弥合了这一空白。在理想插值和相容性条件下，我们证明MCR估计（摘要在此处被截断）

    arXiv:2602.01437v2 Announce Type: replace-cross  Abstract: The problem of corrupted data, missing features, or missing modalities continues to plague the modern machine learning landscape. To address this issue, a class of regularization methods that enforce consistency between imputed and fully observed data has emerged as a promising approach for improving model generalization, particularly in partially observed settings. We refer to this class of methods as Measure Consistency Regularization (MCR). Despite its empirical success in various applications, such as image inpainting, data imputation and semi-supervised learning, a fundamental understanding of the theoretical underpinnings of MCR remains limited. This paper bridges this gap by offering theoretical insights into when MCR yields a more favorable finite-sample estimation-error upper bound, viewed through the lens of neural network distance.   Under ideal interpolation and compatibility conditions, we show that the MCR estimat
    
[^36]: DFNN：一种用于学习度量空间值响应的深度Fréchet神经网络框架

    DFNN: A Deep Fr\'echet Neural Network Framework for Learning Metric-Space-Valued Responses

    [https://arxiv.org/abs/2510.17072](https://arxiv.org/abs/2510.17072)

    本文提出深度Fréchet神经网络（DFNN）框架，通过最小化Fréchet风险来逼近条件Fréchet均值，实现了从欧几里得预测变量到度量空间值响应的端到端回归预测，并为其建立了通用逼近定理。

    

    带有非欧几里得响应的回归——例如概率分布、网络、对称正定矩阵和成分数据——在现代应用中变得越来越重要。本文提出了深度Fréchet神经网络（DFNN），这是一个端到端的深度学习框架，用于从欧几里得预测变量预测非欧几里得响应——这些响应被视为度量空间中的随机对象。我们的方法利用深度神经网络（DNN）的表示学习能力，通过最小化Fréchet风险来逼近给定预测变量下响应的条件Fréchet均值，即条件期望在度量空间中的类比。该框架高度灵活，可适应多样的度量方式和高维预测变量。我们为DFNN建立了通用逼近定理，将神经网络逼近理论的最前沿推进到一般度量空间-

    arXiv:2510.17072v2 Announce Type: replace  Abstract: Regression with non-Euclidean responses---e.g., probability distributions, networks, symmetric positive-definite matrices, and compositions---has become increasingly important in modern applications. In this paper, we propose deep Fr\'echet neural networks (DFNNs), an end-to-end deep learning framework for predicting non-Euclidean responses---which are considered as random objects in a metric space---from Euclidean predictors. Our method utilizes the representation-learning power of deep neural networks (DNNs) to the task of approximating conditional Fr\'echet means of the response given the predictors, the metric-space analogue of conditional expectations, by minimizing a Fr\'echet risk. The framework is highly flexible, accommodating diverse metrics and high-dimensional predictors. We establish a universal approximation theorem for DFNNs, advancing the state-of-the-art of neural network approximation theory to general metric-space-
    
[^37]: 针对广义方程的带方差缩减的新型加速过去额外梯度方法

    New Accelerated Past-Extragradient Methods with Variance Reduction for Generalized Equations

    [https://arxiv.org/abs/2508.16791](https://arxiv.org/abs/2508.16791)

    该论文提出了一种结合Nesterov加速与方差缩减技术的新型过去额外梯度算法框架，用于求解含非单调算子的广义方程，实现了O(1/k²)的期望收敛速率以及更快的o(1/k²)几乎必然收敛速率。

    

    我们开发了一种新颖的过去额外梯度类型算法框架，结合了Nesterov加速技术和方差缩减技术，用于求解数据驱动应用中一类涉及可能非单调算子的广义方程。我们的框架涵盖了广泛的随机方差缩减方案，包括小批量采样以及无偏和有偏的控制变量估计器。我们证明了在Lipschitz连续性和一类“共次单调性”假设下，我们的方法在残差平方范数的期望意义下达到了O(1/k²)的收敛速率，相比非加速的对应方法显著提升了1/k的因子。我们还证明了更快的o(1/k²)收敛速率，无论是在期望意义下还是几乎必然意义下均成立。此外，我们证明了我们方法生成的迭代序列几乎必然收敛到底层问题的解。

    arXiv:2508.16791v2 Announce Type: replace-cross  Abstract: We develop a novel past-extragradient-type algorithmic framework, combining both Nesterov's \textit{acceleration} and \textit{variance-reduction} techniques, to solve a class of generalized equations involving possibly \textit{nonmonotone operators} in data-driven applications. Our framework covers a wide class of stochastic variance-reduced schemes, including mini-batching and both unbiased and biased control-variate estimators. We establish that our method achieves $\mathcal{O}(1/k^2)$ convergence rates in expectation for the squared norm of the residual under Lipschitz continuity and a ``co-hypomonotonicity-type'' assumption, significantly improving upon non-accelerated counterparts by a factor of $1/k$. We also prove faster $o(1/k^2)$ convergence rates, both in expectation and almost surely. In addition, we show that the sequence of iterates generated by our method almost surely converges to a solution of the underlying pro
    
[^38]: 跨尺度生物工艺开发的多保真度批量贝叶斯优化

    Multi-fidelity batch Bayesian optimization for bioprocess development across scales

    [https://arxiv.org/abs/2508.10970](https://arxiv.org/abs/2508.10970)

    本文提出一种多保真度批量贝叶斯优化框架，通过集成定制的高斯过程与混合变量优化，在每次迭代中同时推荐实验条件、工艺尺度和生物催化剂选择，从而加速生物工艺开发并降低实验成本。

    

    生物工艺是现代生物技术的核心，能够实现药品、特种化学品、化妆品和食品的可持续生产。然而，开发高性能的工艺仍然成本高昂且复杂，需要从微孔板到中试反应器的迭代式、多尺度实验。传统的实验设计方法往往难以解决工艺放大问题，以及反应条件与生物催化剂选择的联合优化。我们提出了一种多保真度批量贝叶斯优化框架，以加速生物工艺开发并降低实验成本。该方法集成了专为多保真度建模定制的高斯过程和混合变量优化技术。在每次迭代中，该算法不仅能提出下一步的实验条件，还能提出合适的实验尺度以及生物催化剂（即细胞克隆）的选择。为了对性能进行基准测试，我们开发了一个自定义仿真……

    arXiv:2508.10970v2 Announce Type: replace-cross  Abstract: Bioprocesses are central to modern biotechnology, enabling sustainable production of pharmaceuticals, specialty chemicals, cosmetics, and food. However, developing high-performing processes remains costly and complex, requiring iterative, multi-scale experimentation from microtiter plates to pilot reactors. Conventional Design of Experiments (DoE) approaches often struggle to address process scale-up and the joint optimization of reaction conditions and biocatalyst selection.   We present a multi-fidelity batch Bayesian optimization framework to accelerate bioprocess development and reduce experimental costs. The method integrates Gaussian processes tailored for multi-fidelity modeling and mixed-variable optimization. At each iteration, the algorithm proposes not only the next experimental conditions but also the appropriate scale and choice of biocatalyst (i.e., cell clones). To benchmark performance, we developed a custom sim
    
[^39]: 高斯过程与再生核希尔伯特空间：联系与等价性

    Gaussian Processes and Reproducing Kernel Hilbert Spaces: Connections and Equivalences

    [https://arxiv.org/abs/2506.17366](https://arxiv.org/abs/2506.17366)

    本专著系统揭示了高斯过程与再生核希尔伯特空间在回归、插值等核心任务中的深层等价关系，并提出了基于高斯希尔伯特空间与RKHS等价性的统一理论框架，以促进两个研究领域的交叉融合。

    

    本专著研究了使用正定核的两种方法之间的关系：使用高斯过程的概率方法，以及使用再生核希尔伯特空间（RKHS）的非概率方法。这两种方法在机器学习、统计学和数值分析中被广泛研究和应用。我们探讨了回归、插值、数值积分、分布差异和统计依赖性等基本主题，以及高斯过程的样本路径性质之间的联系和等价性。基于高斯希尔伯特空间与RKHS之间的等价性，建立了这些等价性的统一视角。本专著旨在为桥接基于高斯过程和再生核的许多其他方法奠定基础，这些方法目前由两个研究社区并行发展。

    arXiv:2506.17366v2 Announce Type: replace-cross  Abstract: This monograph studies the relations between two approaches using positive definite kernels: probabilistic methods using Gaussian processes, and non-probabilistic methods using reproducing kernel Hilbert spaces (RKHS). They are widely studied and used in machine learning, statistics, and numerical analysis. We study connections and equivalences for fundamental topics such as regression, interpolation, numerical integration, distributional discrepancies, and statistical dependence, as well as sample path properties of Gaussian processes. A unifying perspective for these equivalences is established, based on the equivalence between the Gaussian Hilbert space and the RKHS. The monograph serves as a basis to bridge many other methods based on Gaussian processes and reproducing kernels, which are developed in parallel by the two research communities.
    
[^40]: 用于有限和求根问题的方差缩减快速Krasnoselkii-Mann方法

    Variance-Reduced Fast Krasnoselkii-Mann Methods for Finite-Sum Root-Finding Problems

    [https://arxiv.org/abs/2406.02413](https://arxiv.org/abs/2406.02413)

    提出了带有新型无偏方差缩减估计器的单循环快速Krasnoselkii-Mann方法用于求解有限和共强制方程，实现了 $\mathcal{O}(1/k^2)$ 和 $o(1/k^2)$ 的最后迭代收敛速率，并以 $\mathcal{O}(n + n^{2/3}\epsilon^{-1})$ 的oracle复杂度达到 $\epsilon$-解。

    

    我们提出了一类新的带方差缩减的快速Krasnoselkii-Mann方法，用于求解有限和共强制方程 $Gx = 0$。我们的算法是单循环的，并利用了一族新的无偏方差缩减估计器，该估计器专门为更广泛类别的求根算法而设计。我们的方法在 $\mathbb{E}[\| Gx^k\|^2]$ 度量下实现了 $\mathcal{O}(1/k^2)$ 和 $o(1/k^2)$ 的最后迭代收敛速率，其中 $k$ 是迭代计数器，$\mathbb{E}[\cdot]$ 是总期望。我们还建立了几乎必然的 $o(1/k^2)$ 收敛速率，以及迭代序列 $\{x^k\}$ 几乎必然收敛到 $Gx=0$ 的解。我们将该框架在两种著名的估计器上进行了实例化：SVRG和SAGA。通过适当选择参数，两种变体都能以 $\mathcal{O}(n + n^{2/3}\epsilon^{-1})$ 的oracle复杂度达到 $\epsilon$-解，其中 $n$ 表示有限和中的求和项数量。

    arXiv:2406.02413v4 Announce Type: replace-cross  Abstract: We propose a new class of fast Krasnoselkii--Mann methods with variance reduction to solve a finite-sum co-coercive equation $Gx = 0$. Our algorithm is single-loop and leverages a new family of unbiased variance-reduced estimators specifically designed for a wider class of root-finding algorithms. Our method achieves both $\mathcal{O}(1/k^2)$ and $o(1/k^2)$ last-iterate convergence rates in terms of $\mathbb{E}[\| Gx^k\|^2]$, where $k$ is the iteration counter and $\mathbb{E}[\cdot]$ is the total expectation. We also establish almost sure $o(1/k^2)$ convergence rates and the almost sure convergence of iterates $\{x^k\}$ to a solution of $Gx=0$. We instantiate our framework for two prominent estimators: SVRG and SAGA. By an appropriate choice of parameters, both variants attain an oracle complexity of $\mathcal{O}(n + n^{2/3}\epsilon^{-1})$ to reach an $\epsilon$-solution, where $n$ represents the number of summands in the finit
    
[^41]: 告别偏差-方差权衡？过参数化机器学习理论综述

    A Farewell to the Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized Machine Learning

    [https://arxiv.org/abs/2109.02355](https://arxiv.org/abs/2109.02355)

    本文综述了过参数化机器学习理论，解释了为何高度过参数化的模型（从线性模型到深度神经网络）能够在完美拟合噪声训练数据的同时仍具有良好的泛化能力，以及双重下降现象如何挑战了传统的偏差-方差权衡教义。

    

    过去十年机器学习（ML）的进展，特别是深度学习时代，提出了许多挑战该领域长期教义的科学问题。其中最重要的谜题之一是过参数化模型良好的经验泛化表现。过参数化模型相对于训练数据集的规模而言具有高度复杂性，这使它们能够完美拟合（即插值）即使是有噪声的训练数据。这种对噪声数据的插值传统上被认为与有害的过拟合相关，然而从简单的线性模型到深度神经网络，人们观察到各种插值模型在新的测试数据上泛化得相当好。事实上，双重下降现象的发现揭示了高度过参数化的模型在测试性能上可以超越最佳的欠参数化模型。理解这种过参数化情境下的学习……

    arXiv:2109.02355v2 Announce Type: replace  Abstract: The last decade of progress in machine learning (ML), especially the deep learning era, has raised a number of scientific questions that challenge the longstanding dogma of the field. One of the most important riddles was the good empirical generalization of overparameterized models. Overparameterized models are highly complex with respect to the size of the training dataset, which enables them to perfectly fit (i.e., interpolate) even noisy training data. Such interpolation of noisy data is traditionally associated with detrimental overfitting, and yet a wide range of interpolating models -- from simple linear models to deep neural networks -- have been observed to generalize remarkably well on fresh test data. Indeed, the discovery of the double descent phenomenon has revealed that highly overparameterized models can improve over the best underparameterized model in test performance. Understanding learning in this overparameterized
    
[^42]: 广义线性专家混合模型中的正则化估计与特征选择

    Regularized Estimation and Feature Selection in Mixtures of Generalized Linear Experts

    [https://arxiv.org/abs/1907.06994](https://arxiv.org/abs/1907.06994)

    该论文提出了一个正则化最大似然框架，通过L1惩罚和近端Newton-EM算法，在广义线性专家混合模型中同时实现参数估计与特征选择，并统一支持高斯、泊松和多项式响应。

    

    专家混合模型（MoE）是一类条件混合模型，其混合比例和分量密度均依赖于预测变量，被广泛应用于回归、分类以及基于模型的异构数据聚类。当预测变量数量众多或彼此相关时，通过最大似然法拟合MoE会变得不稳定，有时甚至不可行。我们提出了一个正则化最大似然框架，用于专家属于广义线性模型族的MoE的同时参数估计与特征选择，该框架在单一公式中统一涵盖了高斯、泊松和多项式响应。通过L1惩罚在门控网络和专家网络中同时诱导稀疏性，并采用近端Newton-EM算法来最大化惩罚对数似然，其M步简化为具有闭式坐标上升更新的加权Lasso问题。与现有的惩罚化MoE方法不同……

    arXiv:1907.06994v2 Announce Type: replace-cross  Abstract: Mixtures of experts (MoE) are conditional mixture models in which both the mixing proportions and the component densities depend on the predictors, and are widely used for regression, classification and model-based clustering of heterogeneous data. Fitting MoE by maximum likelihood becomes unstable, and sometimes infeasible, when the predictors are numerous or correlated. We propose a regularized maximum likelihood framework for simultaneous parameter estimation and feature selection in MoE whose experts belong to the generalized linear model family, covering Gaussian, Poisson and multinomial responses within a single formulation. Sparsity is induced in both the gating network and the experts through $\ell_1$ penalties, and the penalized log-likelihood is maximized by a proximal Newton-EM algorithm whose M-step reduces to weighted Lasso problems with closed-form coordinate-ascent updates. Unlike existing penalized MoE procedure
    

