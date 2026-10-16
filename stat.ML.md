# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Explore Once, Evaluate Many: Multiple Policy Evaluation for Stochastic Linear Bandits](https://arxiv.org/abs/2610.22252) | 提出了 PRISMA 算法，通过一次无奖励探索收集数据并结合正则化信息共享与混合分配，能够在随机线性上下文老虎机中以样本复杂度仅随策略数 $K$ 对数增长的代价同时高精度评估 $K$ 个目标策略的价值。 |
| [^2] | [SurvICL: a Foundation Model for in-context Survival Analysis](https://arxiv.org/abs/2610.22242) | SurvICL是一个通过新型比例几率先验生成合成数据并以分位数回归损失预训练的上下文生存分析基础模型，首次超越经典Cox模型，并配套推出了迄今最大的103数据集生存分析基准。 |
| [^3] | [Query-Efficient Rejection Sampling for Parallel Masked Diffusion](https://arxiv.org/abs/2610.22238) | 提出单增量拒绝采样器（SIRS），每次仅需一次查询即可估计并行掩码扩散模型拒绝采样中的对数接受概率，并结合伯努利工厂与Rao-Blackwell化显著降低查询成本并减少截断偏差。 |
| [^4] | [Learning Changes the World and Then What Was Learned has Changed: Training and Deploying Neural Networks in the Control Loop](https://arxiv.org/abs/2610.22084) | 本文提出了一个统一分类体系来刻画离线训练的神经网络部署到闭环控制后因自身动作改变状态分布而引发的三类失效模式，并通过时间尺度分离的在线适应设计策略来缓解该分布偏移问题。 |
| [^5] | [Rethinking Successive Rejects Algorithms](https://arxiv.org/abs/2610.22051) | 本文通过将 SR 算法的阶段长度调度设计表述为归一化误差指数的优化问题，提出了 H1-平衡连续拒绝算法，从而在固定预算最优臂识别中提升了 SR 算法的性能。 |
| [^6] | [Designing Experiments for External Decision Makers](https://arxiv.org/abs/2610.22032) | 提出“使用感知实验设计”框架，通过训练自适应设计策略最小化“使用风险”，以应对数据将被实验者无法控制的外部规则（可能是非贝叶斯或黑盒）使用的情况，并揭示在此类场景下更具信息量的数据反而可能有害。 |
| [^7] | [Why Shallow Neural Networks Learn Sinusoidal Components First: An Orthogonal Polynomial Dynamics Framework](https://arxiv.org/abs/2610.21964) | 该论文提出基于一般傅里叶级数展开的正交多项式动力学框架，通过分析动力学矩阵的特征值衰减与特征向量形状，从理论上解释了浅层神经网络为何优先学习正弦（低频）分量的谱偏置现象，并适用于一般激活函数、弱数据分布假设和任意网络结构。 |
| [^8] | [SelectPFN: Amortized Bayesian Variable Selection for Sparse Nonlinear Regression with Prior-Data Fitted Networks](https://arxiv.org/abs/2610.21950) | SelectPFN是一种在单次前向传播中同时完成稀疏非线性回归贝叶斯变量选择与预测的先验数据拟合网络，可输出每个变量的后验包含概率，并通过敲除列实现精确的有限样本错误发现率控制。 |
| [^9] | [The Tilted Sampling Problem: Optimism, Posterior Sampling, and Logarithmic Regret](https://arxiv.org/abs/2610.21908) | 该论文提出了倾斜采样这一在线学习问题，以温度归一化累积KL散度定义采样遗憾，并构建了一个通用遗憾框架，揭示倾斜采样与经典遗憾最小化具有不同的不确定性结构，借助乐观原理和后验采样可将对数级遗憾推广至从未知奖励诱导的吉布斯分布中采样的场景。 |
| [^10] | [In-Context Learning Before Specialisation: Theory of Learning in a Solvable Topic Model](https://arxiv.org/abs/2610.21901) | 该论文在一个可解的主题模型中建立了学习理论，从根本层面解释了序列模型为何以及何时会倾向于采用上下文学习而非存储训练数据的专门化外推。 |
| [^11] | [Provably Efficient Risk-Sensitive Offline Reinforcement Learning via Optimized Certainty Equivalents](https://arxiv.org/abs/2610.21870) | 该论文针对通用风险度量类“优化确定性等价（OCE）”下的离线风险敏感强化学习提出了动态与静态两种形式化的悲观价值迭代算法，并建立了紧密的次线性次优性理论界限，首次对该离线场景给出了一般性的理论保证。 |
| [^12] | [Robust Covariance Estimation for Entry-wise Contaminated and Partially Observed Data via Expectation Maximization Algorithm](https://arxiv.org/abs/2610.21709) | 本文提出一种在缩减潜空间上运行的随机近似EM（SAEM）算法，通过将坐标级尺度的随机更新与Rao-Blackwell化的高斯条件矩相结合，实现了逐元素污染与缺失数据下Student-$t^*$模型的鲁棒协方差估计，无需对潜在高斯变量显式采样。 |
| [^13] | [Classical methods for modern applications: synthetic tabular data generation using lightweight discrete response continuous latent variable model](https://arxiv.org/abs/2610.21674) | 本文提出一种利用经验边际分布和分位数变换将含连续与离散变量的表格数据映射到连续潜空间的轻量级生成模型，能够高保真地生成合成表格数据，且实现简单、训练稳健、采样高效。 |
| [^14] | [Conditional Mean Flows: Few-Step Densities for Simulation-Based Inference](https://arxiv.org/abs/2610.21657) | 提出条件均值流方法，将少步采样与可解析的密度评估相结合，大幅降低了基于模拟推断中后验采样及MCMC重复似然评估的计算成本。 |
| [^15] | [Market Incentives for AI Safety Investment](https://arxiv.org/abs/2610.21607) | 该研究构建了包含上游大语言模型提供商和竞争性下游企业的AI供应链博弈论模型，发现下游竞争促使下游企业的安全投资超过行业利润最优水平，而上游提供商因无法充分获得安全投资创造的剩余而倾向于投资不足。 |
| [^16] | [Sharp and Stable Label-Free Calibration Certificates from Logical Coherence](https://arxiv.org/abs/2610.21567) | 本文利用极小极大定理从逻辑一致性出发，为查询条件概率预测器建立了锐利且稳定的无标签校准残差下界证书，并证明Lipschitz赌注形式既稳定又可通过在线梯度上升在独立同分布数据流上有效优化。 |
| [^17] | [When Is Muon Optimal? Radial Fragility and Minimax Geometry](https://arxiv.org/abs/2610.21549) | 该论文引入随机半径各向同性曲率模型与极化因子容许集，精确刻画了Muon缩放极化更新在任意径向定律下保持最优的几何条件，证明非标量最优响应仅来自与曲率扭结对齐的径向原子，因而无原子径向定律呈现零容量刚性。 |
| [^18] | [Two Sample Sizes in Test-Time LLM Ensembling: Buffered ERM and Minimax Theory](https://arxiv.org/abs/2610.21363) | 该论文研究了测试时大语言模型加权集成中标注问题数 Q 与每问题生成数 N 这两个样本量的相互作用，证明带校准间隔缓冲的 ERM 可达到 Q^{-1/2}+N^{-β/2} 阶的超额风险，并通过简化的双模型子模型分析验证了该风险率的最优性。 |
| [^19] | [Controlling the False Discovery Rate via Model-Agnostic Importance Measures](https://arxiv.org/abs/2610.21360) | 本文提出了MAS方法，通过样本分割策略和新颖的agent-X变量重要性度量，在无需对协变量做分布假设的情况下实现错误发现率（FDR）控制下的模型无关变量选择，与Model-X knockoffs形成互补。 |
| [^20] | [Dimension-Free PAC-Bayes Generalization for Deep Neural Networks via Dirichlet Processes](https://arxiv.org/abs/2610.21340) | 该论文通过狄利克雷过程将PAC-Bayes分析的先验从高维参数空间转移到数据分布空间，并借助数据处理不等式证明参数空间相对熵被可闭式计算的数据空间DP散度严格界定，从而获得了与网络维度无关的深度神经网络泛化界，解决了过参数化网络泛化界空洞的问题。 |
| [^21] | [Regret Lower Bound for Online Learning of Quantum States with Logarithmic Loss, with Applications to Quantum State Tomography and Confidence Sequences for Matrix Means](https://arxiv.org/abs/2610.21273) | 本文建立了基于对数损失的量子态在线学习的最优遗憾下界，并由此导出样本量最优（至多相差多对数因子）的量子态层析成像算法，以及矩阵均值置信序列的构造算法。 |
| [^22] | [CleanRM: Reward Modeling from Noisy Feedback with a Few Clean Feedback for RLHF](https://arxiv.org/abs/2610.21138) | 提出CleanRM方法，利用少量干净反馈并通过最优传输校准逐样本噪声比例，实现从噪声反馈中对RLHF奖励模型的无偏学习。 |
| [^23] | [A One-Sided McDiarmid-Type Inequality for Heavy Tails via Reverse KL Tensorization](https://arxiv.org/abs/2610.21103) | 本文提出了一种新的反向KL张量化不等式，证明了对几乎必然有上界且仅满足 $p\in(1,2]$ 阶条件矩的独立随机变量函数，均值上方的大偏差呈指数级不可能，即使方差无限也成立，并应用于无界损失的单边一致风险界。 |
| [^24] | [StochLOB: Heavy-Tailed Score Matching for Robust Limit Order Book Trend Prediction](https://arxiv.org/abs/2610.21081) | 该论文提出StochLOB训练框架，用跳跃Lévy噪声和截断α稳定噪声等重尾非高斯扩散过程替代传统的布朗高斯噪声来加噪训练数据，以更好地捕捉限价订单簿订单流的重尾与跳跃特性，从而提升LOB趋势预测的稳健性，同时推理阶段保持确定性预测器。 |
| [^25] | [Score Mismatch and Set-Constrained Sampling in Zero-Shot Conditional Discrete Diffusion](https://arxiv.org/abs/2610.21061) | 本文首次对分数失配的离散扩散模型进行了理论分析，推导出一般的KL散度上界来量化失配造成的额外偏差，并覆盖文本填充、句子补全等部分观测约束场景，为零样本条件生成提供了理论保证。 |
| [^26] | [Efficient Agnostic Learning of Univariate Densities Under Local Differential Privacy](https://arxiv.org/abs/2610.21033) | 本文提出了一个非交互式局部差分隐私学习框架，实现了对可被分段多项式逼近的单变量分布（如高斯混合模型）首个样本高效且计算高效的不可知学习，并为此开发了可独立使用的大域LDP合成数据生成算法。 |
| [^27] | [On the Last-Iterate Convergence of Clipped Gradient Methods](https://arxiv.org/abs/2610.21026) | 本文针对梯度噪声为重尾分布这一更贴近实际的情形，研究并证明了裁剪梯度方法（如Clipped-SGD）在随机（强）凸优化中的末次迭代收敛性，克服了现有SGD末次迭代收敛结果依赖强噪声条件（如指数衰减尾部）的局限。 |
| [^28] | [Acceleration and its Limits in Gaussian Variational Inference](https://arxiv.org/abs/2610.21020) | 该论文证明高斯变分推断在可精确计算高斯期望时能达到经典的 √κ 加速收敛率，但通过构造一族强对数凹目标证明仅有逐点导数时任何算法都需要 κ^(1-o(1)) 次查询，从而说明仅凭曲率界不足以保证加速。 |
| [^29] | [Failure Is the Mother of Success: Recovering Supervision with Grounding Hindsight](https://arxiv.org/abs/2610.21018) | 提出 RSHD 方法，让策略为 GRPO 中的失败尝试撰写纠正性笔记并用其重解问题，仅当重放成功率超过原始组时才将笔记蒸馏回策略，从而从全组失败的难题中恢复监督信号。 |
| [^30] | [Halve-Then-Greedy: Adaptive Sampling for Posterior Success with Near-Optimal Instance Dependence](https://arxiv.org/abs/2610.20997) | 提出“先减半后贪心”（HTG）自适应采样算法，在固定预算下最大化推荐设计表现超过目标的后验证据概率，并证明了实例依赖的指数级失败概率界，其样本复杂度在单正臂情形下达到近最优（仅差一个对数因子）。 |
| [^31] | [Two-Stage Score Estimation for Group-Specific Diffusion Models](https://arxiv.org/abs/2610.20979) | 该论文提出了一个针对群体特定扩散模型的两阶段得分估计统计框架，通过将共享池化得分与目标特定偏移量分离分析，推导出收敛速率，并证明扩散模型可作为密度估计器获得全变差保证。 |
| [^32] | [Ranked Right, Sized Wrong: Ensemble Spread Fails Silently at Streamflow Extremes](https://arxiv.org/abs/2610.20947) | 研究发现深度学习径流预报模型的集合离散度虽能正确排序预测误差，但其量级随流量增大而系统性低估误差（平常流量下差2倍、极端流量下可达18倍），表明该不确定性信号在径流极端事件中会悄然失效。 |
| [^33] | [Sign Betting for Misspecified Regression: Variance Adaptivity and Optimal Approximation](https://arxiv.org/abs/2610.20903) | 本文证明符号投注回归在模型误设下可实现高概率的 $L_1$ 预言机不等式，其误差界自适应于平均条件方差，并证明其对恰当学习所能达到的普适逼近因子 3 是最优的。 |
| [^34] | [Stochastic Gradient Descent: A Nonlinear Time Series Perspective](https://arxiv.org/abs/2610.20894) | 本文在迭代随机函数框架下，通过几何矩收缩（GMC）证明了恒定步长SGD在欧氏距离下的收敛性，克服了固定初始化带来的非平稳性问题，证明了平均SGD的淬火中心极限定理、不变原理和α-稳定收敛，并通过研究关于步长的导数为理解步长的影响提供了新视角。 |
| [^35] | [Flexible Doubly Robust Proximal Causal Inference for Censored Outcomes](https://arxiv.org/abs/2610.20893) | 该论文提出了邻近生存桥梁学习（PSB），一个针对删失结果的灵活双重稳健邻近因果推断框架，通过核估计器（PSB-K）和神经估计器（PSB-N）两种灵活桥梁学习器直接处理删失数据和生存曲线目标，突破了现有方法依赖限制性参数化桥梁模型的局限。 |
| [^36] | [A Mixture-of-Experts Framework for Multi-Stream Quickest Change Detection](https://arxiv.org/abs/2610.20876) | 本文提出了一种用于最快变化检测的专家混合框架，并将其应用于多流变化检测问题，所得到的算法在性能上优于文献中已知的最优算法。 |
| [^37] | [Time-partitioning Gaussian Process Mixture Model](https://arxiv.org/abs/2610.20860) | 提出了一种贝叶斯非参数框架TPGPMM，通过自适应Pitman-Yor先验与特定状态高斯过程的结合，自动识别时间序列中的持续状态与变点，无需马尔可夫假设即可建模非平稳依赖，并能有效区分持续变化与瞬态冲击。 |
| [^38] | [Complete or Sparse: A Tale of Two Identifiabilities](https://arxiv.org/abs/2610.20851) | 本文证明在生成式与判别式两种因果方向下，表示维度充足时最优编码器是潜在变量的线性函数，而结合非负激活与能量效率约束可使每个神经元恰好编码一个潜在变量，为“祖母神经元”现象提供了精确的理论解释。 |
| [^39] | [Selection-Safe Conformal Early Exit: Distribution-Free Coverage under Adaptive Depth](https://arxiv.org/abs/2610.20773) | 该论文提出一种选择安全的保形早退方法，通过前缀最大值构造的终端路径分数和单一保形分位数校准，在任意数据依赖的提前退出策略下均能保证无分布覆盖率，无需 alpha/L 修正，且实验表明其覆盖率显著优于朴素的逐层选择方法。 |
| [^40] | [RoLaQ: Joint Rotation and Range Calibration for 2-Bit KV Caches in Multi-Head Latent Attention](https://arxiv.org/abs/2610.20686) | 提出RoLaQ离线校准方法，通过在多头潜在注意力模型中联合学习共享旋转和相互耦合的量化范围，实现2比特KV缓存量化，在多个模型和基准测试上取得了2比特方法中最高的平均准确率。 |
| [^41] | [Learning Topology-Aware Representations for Mechanics-Relevant TPMS Metamaterials](https://arxiv.org/abs/2610.20684) | 提出将持续同调融入DMAE骨干网络的拓扑感知表示学习框架，通过持续性引导的掩码策略和辅助持续同调解码器显式编码TPMS超材料的全局结构连通性，从而更好地支持力学分析与结构设计。 |
| [^42] | [Diminishing Returns of Data in Sparse Autoencoders: Functional Behavior does not Guarantee Feature Reproducibility](https://arxiv.org/abs/2610.20680) | 该论文发现稀疏自编码器的功能表现（如重构）在远小于常规规模的数据量下即可接近全量训练水平，但特征的一对一可复现性会随数据量增加持续提升，表明功能行为良好并不能保证特征可复现，数据规模和数据选择方式对特征复现至关重要。 |
| [^43] | [Spectral Indistinguishability and Learning-Curve Identifiability in Quadratic Networks](https://arxiv.org/abs/2610.20678) | 该论文构造了谱测度完全一致（谱不可区分）但学习曲线仍可区分的二次网络对，证明谱摘要无法唯一辨识网络，而标量学习曲线可以实现辨识，且其查询复杂度随循环长度呈指数增长。 |
| [^44] | [On the Stability and Convergence of Flow Map Training](https://arxiv.org/abs/2610.20674) | 本文通过分析MeanFlow与拉格朗日损失诱导的函数梯度流，首次从理论上解释了流映射训练中停止梯度算子和自适应加权等技巧的必要性，并证明了训练会收敛到真实流映射。 |
| [^45] | [Graph Additive Models disentangling feature, topology, and interactions](https://arxiv.org/abs/2610.20662) | 提出AMIGA模型，通过挖掘的图模体表示拓扑结构，并将节点贡献分解为特征、结构和交互三个加性通道，从而解决了图神经加性网络中虚假相关性传递和效应纠缠的问题。 |
| [^46] | [Latent Annealed Posterior Sampling with a Variance-Preserving Bridge for Sparsely Observed Random Fields](https://arxiv.org/abs/2610.20659) | 提出LAPS方法，利用流匹配学习潜在先验并在潜在空间中进行退火后验采样，结合方差保持桥控制协方差膨胀，从而高效解决稀疏观测随机场后验推断中空间不确定性刻画与计算成本增长的难题。 |
| [^47] | [A polylogarithmic bound for list replicability](https://arxiv.org/abs/2610.20657) | 本文为任意二元概念类的列表可复现性数 O(log²M log N) 这一首个多对数上界提供了一个简短且自包含的新证明。 |
| [^48] | [Follow the Perturbed Leader for Linear Bandits: $\sqrt{T}$ Regret over General Convex Sets](https://arxiv.org/abs/2610.17818) | 本文提出了首个基于FTPL的对抗性线性赌博机算法，通过一个适应估计奖励的扰动分布，在一般凸动作集上实现了 $\sqrt{T}$ 遗憾，改进了此前 $T^{2/3}$ 的遗憾界，同时比连续指数权重方法具有更好的计算效率和更简单的实现。 |
| [^49] | [Identifiability of a dissipative knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational data](https://arxiv.org/abs/2610.09889) | 该论文将人类学习建模为参数具有机制含义的耗散常微分方程组，证明了在设计激励条件下模型参数可从数据中被精确恢复（双概念情形有闭式解），而在仅依赖观测数据时可辨识性会退化，并提出了数值精确等价且速度大幅提升的半隐式L-稳定批量求解器。 |
| [^50] | [LASER: Latent Space Adjoint Matching for Support-Constrained Entropy-Regularized Offline RL](https://arxiv.org/abs/2610.08989) | LASER通过潜在空间伴随匹配实现熵正则化的潜在空间离线强化学习，既防止了策略坍缩成单一脆弱模式，又避免了时间反向传播，在40个不同数据质量的OGBench任务上取得了优异表现。 |
| [^51] | [Optimal State-Space Order for Spectral Gaps of Sliding-Window Occupation Counts](https://arxiv.org/abs/2609.27836) | 本文证明了滑动窗口占据计数所定义的投影计数核的谱隙与原马尔可夫核谱隙之比的最优常数 $c_m^\star$ 满足 $\Theta(1/m)$ 的阶，给出了普适下界 $1/(1080m)$ 和形如 $(m+\log m+O(1))^{-1}$ 的上界，但精确值仍待确定。 |
| [^52] | [ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing](https://arxiv.org/abs/2608.26083) | ICON分解通过多变量分析，在控制其他概念和结果后精确量化每个概念对模型表示的独特贡献，从而有效识别捷径学习并提高解释的准确性。 |
| [^53] | [Optimal Watermark Localization in Mixed-Source Large Language Model Texts](https://arxiv.org/abs/2608.14906) | 本文提出了混合来源LLM文本中水印定位的渐近最优框架，明确了全局检测、发现和分类的相变边界，并证明发现任务难度高于分类。 |
| [^54] | [Nonconvex Composite Functional Constraints via First-Order Augmented Lagrangian Methods under Local Regularity](https://arxiv.org/abs/2607.08954) | 该论文在仅需可行集附近局部乘子正则性的弱条件下，无需假设乘子有界，通过带对偶正则化的光滑近端线性增广拉格朗日原始-对偶方法，为非凸复合约束优化问题建立了非渐近KKT收敛保证，达到 $O(K^{-1/3})$ 的KKT误差率。 |
| [^55] | [ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum](https://arxiv.org/abs/2607.04535) | ManifoldFlow通过对固定谱Stiefel层进行最小松弛，将权重分解为 W = Q S^{1/2}，在保持基向量位于Stiefel流形上的同时学习有界的正定奇异谱，使特征值裁剪成为直接的奇异值控制机制，并在序列、表格和图像任务中优于固定谱Stiefel层。 |
| [^56] | [Set Membership Estimation for Stochastic Linear Contextual Bandits](https://arxiv.org/abs/2606.20022) | 本文首次将控制与系统辨识领域成熟的集合成员估计（SME）方法引入线性上下文赌博机，提出基于SME的OFU算法并建立了次线性遗憾界。 |
| [^57] | [Optimal score function estimation via derivatives constraints](https://arxiv.org/abs/2606.19084) | 该论文建立了统计理论，证明当数据分布支撑在低维流形上时，在导数约束下的经验分数匹配估计可达到极小化极大最优的收敛速率，且该速率取决于流形的内在维度而非环境维度，从而为扩散生成模型的泛化能力而非过拟合提供了理论保证。 |
| [^58] | [Near-Exponential Convergence Rates for kNN Classification based on Boltzmann Margin](https://arxiv.org/abs/2606.10361) | 本文提出了一种介于Tsybakov边际与Massart边际之间的新条件——玻尔兹曼边际，并据此首次为kNN分类建立了近指数收敛速率。 |
| [^59] | [Detectability without Interference: Efficient Canary Design for Privacy Auditing in One Run](https://arxiv.org/abs/2605.27292) | 本文提出IBIS框架，通过基于影响的贪心初始化与双层优化设计相互干扰最小、检测分数最优的金丝雀样本，使隐私审计仅需一次训练运行即可高效完成。 |
| [^60] | [Supervised Learning Objectives for High-Dimensional Treatment Effects](https://arxiv.org/abs/2605.27281) | 该论文将因果推断重新表述为监督学习问题，证明了因果误差可分解为递增阶数的矩平衡误差，并据此设计出能直接改善高维处理效应（如文本干预）因果估计的学习目标，同时可将效应投影到更粗粒度的属性上。 |
| [^61] | [Plan Before You Trade: Inference-Time Optimization for RL Trading Agents](https://arxiv.org/abs/2605.12653) | 提出了受模型预测控制启发的FinPILOT框架，利用多步价格预测在推理时对任意预训练的强化学习交易策略进行优化，实现“先规划、后交易”的策略自适应调整。 |
| [^62] | [A proximal gradient algorithm for composite log-concave sampling](https://arxiv.org/abs/2605.12461) | 提出了一种针对复合对数凹分布的近端梯度采样算法，通过结合 $f$ 的梯度评估与 $g$ 的受限高斯预言机，在 $f+g$ 为 $\alpha$-凸且 $f$ 为 $\beta$-光滑的条件下，以 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$ 次预言机调用实现总变差距离下 $\varepsilon$ 精度的采样，匹配了 $g=0$ 情形下的最优结果。 |
| [^63] | [Mind the Gap: Structure-Aware Consistency in Preference Learning](https://arxiv.org/abs/2604.27733) | 本文证明了标准替代损失在LLM对齐中无法保证排序一致性，并提出一种结构感知的边际调整方法（SA-DPO），通过利用响应间的语义距离实现稳定且一致的对齐。 |
| [^64] | [A Distribution-to-Distribution Neural Probabilistic Forecasting Framework for Dynamical Systems](https://arxiv.org/abs/2603.25370) | 本文提出了一种分布到分布（D2D）神经概率预测框架，利用核均值嵌入编码输入分布、混合密度网络参数化输出分布，实现了动力系统的递归多步概率预测，避免了求解计算代价高昂的分布演化方程。 |
| [^65] | [A Model Ensemble-Based Post-Processing Framework for Fairness-Aware Prediction](https://arxiv.org/abs/2603.18838) | 本文提出一种基于模型集成的后处理框架，该方法独立于模型内部结构、适用于多种学习任务和公平性定义，能够在几乎不损失预测精度的前提下有效提升模型公平性。 |
| [^66] | [Computational and Statistical Hardness of Calibration Distance](https://arxiv.org/abs/2603.18391) | 该论文提出了在均匀边际和无噪声标签假设下精确计算校准距离的高效算法，证明移除任一假设后问题即变为NP难，并给出了适用于一般情形的PTAS以及估计问题所需的Θ(1/ε³)样本复杂度的紧界。 |
| [^67] | [Diffusion Models in Simulation-Based Inference: A Tutorial Review](https://arxiv.org/abs/2512.20685) | 本综述系统梳理了扩散模型在基于模拟推断（SBI）中的最新进展，涵盖训练、推断与评估的设计选择，并讨论了引导、流匹配、一致性模型等概念以及噪声调度、参数化和采样器对效率与统计精度的影响。 |
| [^68] | [Boltzmann generators for amorphous particle systems](https://arxiv.org/abs/2512.16607) | 本研究针对平衡采样极其困难的非晶材料（玻璃），通过将所需的等变性直接嵌入黎曼结构中，开发了首个专为非晶粒子系统定制的玻尔兹曼生成器。 |
| [^69] | [Multi-task Neural Diffusion Processes](https://arxiv.org/abs/2510.03419) | 提出多任务神经扩散过程（MT-NDPs），通过任务编码器提取任务表示并条件化扩散模型，实现跨任务信息共享的任务条件化概率回归与少样本自适应。 |
| [^70] | [A Greedy PDE Router for Blending Neural Operators and Classical Methods](https://arxiv.org/abs/2509.24814) | 本文提出一种近似贪心PDE路由器，在混合迭代求解过程中每步智能选择神经算子或经典数值求解器以融合两者优势，在多种PDE求解任务中持续缩短求解时间。 |
| [^71] | [Expected Signature Kernels for Rough Paths](https://arxiv.org/abs/2509.07893) | 本文将Lévy粗糙路径的期望签名核刻画为扩展Goursat型方程组解的标量分量，证明了跳跃情形下无限维系统的适定性并给出显式误差控制的有限维截断，从而无需辅助参考样本即可直接计算单样本签名-MMD检验中的参考依赖项。 |
| [^72] | [Optimal Condition for Initialization Variance in Deep Neural Networks: An SGD Dynamics Perspective](https://arxiv.org/abs/2508.12834) | 该论文通过将SGD近似为朗之万动力学并利用Fokker-Planck方程与KL散度，建立了期望损失与初始化分布的联系，从而推导出深度神经网络初始化方差的最优条件。 |
| [^73] | [Formal Bayesian Transfer Learning via the Total Risk Prior](https://arxiv.org/abs/2507.23768) | 该论文提出了一种新的贝叶斯迁移学习方法，其核心创新在于使用以源参数为条件的风险最小化器（而非传统的经验损失最小化器）来构建源数据集与目标数据集所有参数的单一联合先验分布，从而实现完整的贝叶斯不确定性量化以及通过Gibbs采样对源数据集进行选择和模型平均。 |
| [^74] | [PAC-Causal Learnability: From Identification to Finite-Sample Guarantees](https://arxiv.org/abs/2507.18903) | 该论文提出了PACC因果可学习性框架，首次从PAC学习角度为因果估计提供了有限样本的统一精度-置信度保证，并揭示其样本复杂度由因果识别强度、目标复杂度和干扰复杂度三个关键因素决定。 |
| [^75] | [Parallel Layer Normalization for Universal Approximation](https://arxiv.org/abs/2505.13142) | 该论文证明仅含单个层归一化单元的浅层网络无论宽度多大都不具备万能逼近能力，而提出的并行层归一化网络（PLN-Nets）随着并行分组数量的增加即可实现万能逼近，且该机制还可推广到RMSNorm变体、逐位置前馈网络和卷积网络。 |
| [^76] | [STAMP: Symmetric Temporal Averaging Message Passing for Domain Adaptation in Chronologically Split Graphs](https://arxiv.org/abs/2411.10957) | 提出对称时间平均消息传递（STAMP），通过在时间戳间平衡消息强制时间域间的一阶矩不变性，从而解决按时间顺序划分图中的分布偏移问题，提升半监督节点分类性能。 |
| [^77] | [Second Maximum of a Gaussian Random Field and Exact (t-)Spacing test](https://arxiv.org/abs/2406.18397) | 本文提出了黎曼流形上高斯随机场次大值的新概念，借助 Kac-Rice 公式推导出以次大值和 Hessian 矩阵独立部分为条件的全局最大值的显式分布，从而构造出一种可用于检测稀疏备择假设的精确间距检验方法。 |
| [^78] | [Obtaining Explainable Classification Models using Distributionally Robust Optimization.](http://arxiv.org/abs/2311.01994) | 本论文介绍了一种利用分布鲁棒优化获取可解释的分类模型的方法，通过构建稀疏的规则集合来同时解决规则集的稀疏性和预测准确性之间的权衡，从而保证泛化性能并降低计算成本。 |

# 详细

[^1]: 一次探索，多次评估：随机线性老虎机中的多策略评估

    Explore Once, Evaluate Many: Multiple Policy Evaluation for Stochastic Linear Bandits

    [https://arxiv.org/abs/2610.22252](https://arxiv.org/abs/2610.22252)

    提出了 PRISMA 算法，通过一次无奖励探索收集数据并结合正则化信息共享与混合分配，能够在随机线性上下文老虎机中以样本复杂度仅随策略数 $K$ 对数增长的代价同时高精度评估 $K$ 个目标策略的价值。

    

    我们研究如何在具有线性奖励结构的随机上下文老虎机中同时评估 $K$ 个目标策略。在给定已知的 $d$ 维特征映射并可以无奖励方式访问策略查询的条件下，我们提出了 PRISMA（通过正则化信息共享与混合分配进行策略评估）算法，该算法能够以至少 $1-\delta$ 的概率将每个目标策略的价值估计到精度 $\varepsilon$。该算法使用 $\widetilde{\mathcal{O}}\!\left( \frac{(L^2S^2+\beta^2)(1\vee\rho^\star)}{\varepsilon^2} \cdot \log(K) \right)$ 次上下文抽取和奖励观测，其中 $\rho^\star\le d$ 是一个与实例相关的量，用于衡量设计共享采样分布的难度。其中，$L$、$S$ 和 $\beta$ 分别约束特征范数、奖励参数范数和奖励噪声，被省略的因子是关于 $K$、$d$ 和 $1/\delta$ 的对数项。我们的方法结合了无奖励规划、正则化……（摘要在此截断）

    arXiv:2610.22252v1 Announce Type: new  Abstract: We study how to evaluate $K$ target policies simultaneously in stochastic contextual bandits with linear reward structure. Given a known $d$-dimensional feature map and reward-free access to policy queries, we propose PRISMA (Policy evaluation through Regularized Information Sharing and Mixture Allocation), an algorithm that estimates every target policy's value to accuracy $\varepsilon$ with probability at least $1-\delta$. The algorithm uses $\widetilde{\mathcal{O}}\!\left( \frac{(L^2S^2+\beta^2)(1\vee\rho^\star)}{\varepsilon^2} \cdot \log(K) \right)$ context draws and reward observations, where $\rho^\star\le d$ is an instance-dependent measure of the difficulty of designing a shared sampling distribution. Here, $L$, $S$, and $\beta$ bound the feature norm, reward parameter norm, and reward noise, respectively, and the suppressed factors are logarithmic in $K$, $d$, and $1/\delta$. Our method combines reward-free planning, a regulariz
    
[^2]: SurvICL：一个用于上下文生存分析的基础模型

    SurvICL: a Foundation Model for in-context Survival Analysis

    [https://arxiv.org/abs/2610.22242](https://arxiv.org/abs/2610.22242)

    SurvICL是一个通过新型比例几率先验生成合成数据并以分位数回归损失预训练的上下文生存分析基础模型，首次超越经典Cox模型，并配套推出了迄今最大的103数据集生存分析基准。

    

    表格基础模型（TFM）目前在分类和回归任务上已超越梯度提升树。然而在生存分析领域——即从删失数据中预测事件发生时间，且通常基于小规模临床队列——我们发现现有TFM均无法击败拥有半个世纪历史的Cox模型。为解决这一问题，我们提出了SurvICL，一个能够以上下文学习方式预测生存曲线的表格基础模型。SurvICL在纯合成数据集上进行预训练，这些数据由我们提出的一种新型灵活比例几率先验生成，该先验在消融研究中表现优于其他类型的先验。与大多数其他生存分析模型不同，SurvICL采用简单的分位数回归损失进行训练，这之所以可行，是因为其合成预训练数据生成器能够提供未删失的事件时间。此外，我们推出了迄今为止规模最大的生存分析基准，涵盖103个数据集。在从所有设计决策中保留的63个数据集上，SurvICL超越了20种基于树的方法、深度学习方法和基础模型。

    arXiv:2610.22242v1 Announce Type: cross  Abstract: Tabular foundation models (TFMs) now outperform gradient-boosted trees in classification and regression. Yet in survival analysis, which predicts when an event occurs from censored data, often on small clinical cohorts, we find that no existing TFM beats the half-century-old Cox model. We address this issue by introducing SurvICL, a TFM that predicts survival curves in-context. SurvICL is pretrained on purely synthetic datasets generated from a novel flexible proportional-odds prior that outperforms other prior types in our ablation study. Unlike most other survival models, SurvICL is trained with a plain quantile regression loss, which is possible because its synthetic pre-training data generator can provide uncensored event times. Additionally, we introduce the largest survival benchmark to date spanning 103 datasets. On the 63 datasets held out from all design decisions, SurvICL outperforms 20 tree-based, deep, and foundation-model 
    
[^3]: 面向并行掩码扩散的查询高效拒绝采样

    Query-Efficient Rejection Sampling for Parallel Masked Diffusion

    [https://arxiv.org/abs/2610.22238](https://arxiv.org/abs/2610.22238)

    提出单增量拒绝采样器（SIRS），每次仅需一次查询即可估计并行掩码扩散模型拒绝采样中的对数接受概率，并结合伯努利工厂与Rao-Blackwell化显著降低查询成本并减少截断偏差。

    

    掩码扩散模型（MDMs）通过并行解掩码多个token来加速生成，但独立采样这些token忽略了它们之间的相关性，从而使输出产生偏差。Anari等人（2026）利用拒绝采样纠正了这一偏差，将对数接受概率分解为token级项之和，但评估时需要对每个被解掩码的token进行一次分数查询。我们提出了单增量拒绝采样器，它仅以一次查询的代价，从单个随机采样的增量中提供对数接受概率的估计。随后，SIRS通过改编自Chen等人（2026）的伯努利工厂，将泊松数量的截断估计转换为接受/拒绝决策。由于每次查询都会返回块中所有剩余token的条件分布，我们进一步以零额外代价对这些token取平均来对每个增量进行Rao-Blackwell化，从而减少截断误差。我们证明了KL散度……（摘要原文在此处截断）

    arXiv:2610.22238v1 Announce Type: new  Abstract: Masked diffusion models (MDMs) accelerate generation by unmasking multiple tokens in parallel, but sampling these tokens independently ignores their correlations and biases the output. Anari et al. (2026) corrected this bias with rejection sampling, decomposing the log acceptance probability into a sum of token-level terms, but the evaluation requires one score query per unmasked token. We propose the Single-Increment Rejection Sampler (SIRS), which provides an estimate of the log acceptance probability from a single randomly sampled increment at the cost of one query. SIRS then converts a Poisson number of clipped estimates into accept/reject decisions through a Bernoulli factory adapted from Chen et al. (2026). Since each query returns the conditionals of all remaining tokens in the block, we further Rao-Blackwellize each increment by averaging over these tokens at no extra cost, thereby reducing clipping. We show that the KL-divergenc
    
[^4]: 学习改变了世界，而后所学到的也随之改变：在控制回路中训练与部署神经网络

    Learning Changes the World and Then What Was Learned has Changed: Training and Deploying Neural Networks in the Control Loop

    [https://arxiv.org/abs/2610.22084](https://arxiv.org/abs/2610.22084)

    本文提出了一个统一分类体系来刻画离线训练的神经网络部署到闭环控制后因自身动作改变状态分布而引发的三类失效模式，并通过时间尺度分离的在线适应设计策略来缓解该分布偏移问题。

    

    神经网络为控制那些标称建模难以处理的复杂系统提供了关键的适应能力。然而，当离线训练的神经网络被部署到闭环控制系统中时，其动作会从根本上改变其训练时所依据的状态分布。本文引入了一个统一的分类体系，对这些顺序发生的失效模式进行归类：初始的闭环辨识误差、部署引发的动作误差以及持续的跟踪误差。通过离散时间敏感性分析，本工作形式化地阐明了离线学习的参数如何在部署时局部地导致轨迹发散，并因局部损失几何结构而造成性能退化。为缓解这种分布偏移，我们提出了以时间尺度分离的在线适应为核心的可操作设计策略。这些机制通过在非线性Duffing振子上的消融实验得到验证，表明……

    arXiv:2610.22084v1 Announce Type: cross  Abstract: Neural networks offer critical adaptability for controlling complex systems where nominal modeling proves intractable. However, when offline-trained neural networks are deployed within closed-loop control systems, their actions fundamentally alter the state distribution upon which they were trained. This paper introduces a unified taxonomy to categorize these sequential failure modes: the initial Closed-Loop Identification Error, the deployment-induced Action Error, and the continuous Tracking Error. Through discrete-time sensitivity analysis, this work formalizes how offline-learned parameters locally guarantee a trajectory divergence upon deployment, degrading performance due to local loss geometry. To mitigate this distribution shift, we propose actionable design policies centered on time-scale separated online adaptation. These mechanisms are validated through ablation studies on a nonlinear Duffing oscillator, demonstrating that b
    
[^5]: 重新思考连续拒绝算法

    Rethinking Successive Rejects Algorithms

    [https://arxiv.org/abs/2610.22051](https://arxiv.org/abs/2610.22051)

    本文通过将 SR 算法的阶段长度调度设计表述为归一化误差指数的优化问题，提出了 H1-平衡连续拒绝算法，从而在固定预算最优臂识别中提升了 SR 算法的性能。

    

    在最优臂识别问题中，算法通过自适应地选择各个臂来找到平均奖励最大的臂。连续拒绝算法是一种简单的固定预算最优臂识别算法，它在每个阶段淘汰一个候选臂。乍一看，SR 似乎自适应能力有限，因为它遵循预先确定的阶段长度。然而，对于具有共同方差的高斯臂，SR 以流行的复杂度度量 H1(P) 归一化后的最坏情况误差指数在常数因子范围内是最优的，这部分解释了为什么 SR 的性能与最先进的自适应采样算法相比具有竞争力。在本文中，我们通过优化 SR 预先定义的阶段长度调度来扩展其框架。我们将阶段调度设计表述为一个关于归一化误差指数的优化问题。然后，我们引入了一种具有优化调度的 SR 版本，我们称之为 H1-平衡连续拒绝算法。

    arXiv:2610.22051v1 Announce Type: cross  Abstract: In the best-arm identification problem, an algorithm adaptively selects arms to find the arm of the largest mean reward. Successive Rejects (SR) is a simple fixed-budget best-arm identification algorithm that eliminates one candidate arm at each phase. At first glance, SR seems to have limited adaptivity, since it follows predetermined phase lengths. However, for Gaussian arms with common variance, SR's worst-instance error exponent normalized by the popular complexity measure $H_1(P)$ is optimal up to a constant factor, which partly explains why SR's performance is competitive compared with state-of-the-art adaptive sampling algorithms. In this paper, we extend the framework of SR by optimizing its predefined schedule of phase lengths. We formulate phase-schedule design as an optimization problem for the normalized error exponent. Then, we introduce a version of SR with the optimized schedule, which we call $H_1$-balanced Successive R
    
[^6]: 为外部决策者设计实验

    Designing Experiments for External Decision Makers

    [https://arxiv.org/abs/2610.22032](https://arxiv.org/abs/2610.22032)

    提出“使用感知实验设计”框架，通过训练自适应设计策略最小化“使用风险”，以应对数据将被实验者无法控制的外部规则（可能是非贝叶斯或黑盒）使用的情况，并揭示在此类场景下更具信息量的数据反而可能有害。

    

    标准的贝叶斯实验设计（BED）方法假设实验者本人会依据数据采取行动，即基于自身的信念和目标选择最优行动。但在实践中，数据往往由外部各方来使用，其处理流程是实验者无法改变且可能并不了解的，尤其是当数据被公开发布时。为了将BED扩展到此类场景，我们提出了“使用感知实验设计”框架，通过训练自适应设计策略来最小化“使用风险”，即当一条或多条外部规则作用于所收集数据时实验者的期望损失。这些规则可能是非贝叶斯的或黑盒的；当可能有多条规则适用时，我们或对所有规则取平均，或采用最坏情况。我们证明，与标准BED不同，在此类规则下更具信息量的数据可能是有害的，因此使用风险通常无法表示为期望不确定性缩减；我们的框架因此严格推广了标准BED。

    arXiv:2610.22032v1 Announce Type: cross  Abstract: Standard Bayesian experimental design (BED) methods assume the experimenter alone will act on the data, choosing the best action given their own beliefs and goals. In practice, data are often acted on by external parties whose procedures the experimenter cannot change and may not know, particularly when the data are released publicly. To extend BED to such settings, we introduce usage-aware experimental design, a framework in which an adaptive design policy is trained to minimise the usage risk: the experimenter's expected loss when one or more external rules act on the collected data. These rules may be non-Bayesian or black-box, and when several may apply, we either average over them or take the worst case. We show that, unlike in standard BED, more informative data can be harmful under such rules, so the usage risk cannot in general be expressed as an expected uncertainty reduction; our framework thus strictly generalises standard B
    
[^7]: 为什么浅层神经网络首先学习正弦分量：一个正交多项式动力学框架

    Why Shallow Neural Networks Learn Sinusoidal Components First: An Orthogonal Polynomial Dynamics Framework

    [https://arxiv.org/abs/2610.21964](https://arxiv.org/abs/2610.21964)

    该论文提出基于一般傅里叶级数展开的正交多项式动力学框架，通过分析动力学矩阵的特征值衰减与特征向量形状，从理论上解释了浅层神经网络为何优先学习正弦（低频）分量的谱偏置现象，并适用于一般激活函数、弱数据分布假设和任意网络结构。

    

    神经网络通常先学习低频分量、后学习高频分量——这一现象被称为谱偏置或F-原理。许多数值研究已经证实了这一现象，理论研究也尝试对其进行解释。然而，以下几个目标仍未得到充分解决：(1) 渐近地解释谱偏置，包括其起源以及为什么被优先学习的分量通常是正弦的；(2) 将解释推广到一般的激活函数，如sigmoid和ReLU；(3) 对数据分布做较弱的假设；(4) 避免混叠效应，例如有限域傅里叶分析中出现的混叠效应；(5) 发展一个适用于任意网络的理论框架。我们通过分析动力学矩阵H的特征值衰减和特征向量形状，从理论上实现了这些目标——该矩阵在我们将网络输出展开为一般傅里叶级数（GFS）时产生……

    arXiv:2610.21964v1 Announce Type: cross  Abstract: Neural networks often learn low-frequency components before high-frequency ones --- a phenomenon called spectral bias or termed F-principle. Many numerical studies have demonstrated this phenomenon, and theoretical studies have sought to explain it. However, several goals remain insufficiently addressed: (1) explaining spectral bias asymptotically, including its origin and why the preferred components are generally sinusoidal; (2) extending the explanation to general activation functions, such as sigmoid and ReLU; (3) making weak assumptions about the data distribution; (4) avoiding aliasing effects, such as those arising in finite-domain Fourier analysis; and (5) developing a theoretical framework applicable to arbitrary networks. We address these goals theoretically by analyzing the eigenvalue decay and eigenvector shapes of the dynamics matrix H, which arises when we expand the network output as a general Fourier series (GFS) in an 
    
[^8]: SelectPFN：基于先验数据拟合网络的稀疏非线性回归摊销贝叶斯变量选择

    SelectPFN: Amortized Bayesian Variable Selection for Sparse Nonlinear Regression with Prior-Data Fitted Networks

    [https://arxiv.org/abs/2610.21950](https://arxiv.org/abs/2610.21950)

    SelectPFN是一种在单次前向传播中同时完成稀疏非线性回归贝叶斯变量选择与预测的先验数据拟合网络，可输出每个变量的后验包含概率，并通过敲除列实现精确的有限样本错误发现率控制。

    

    基于先验数据拟合网络（PFNs）的表格基础模型可以在单次前向传播中对新数据集进行预测，但它们无法报告其预测依赖于哪些输入。我们提出SelectPFN，这是一种为稀疏非线性回归联合执行贝叶斯变量选择和预测的PFN。在一次前向传播中，它返回每个变量的后验包含概率，且其预测仅从采样的变量子集计算得出，因此被排除的变量可证明无法影响预测；我们证明这正是贝叶斯模型所规定的结果。该网络基于围绕组尖峰-平板稀疏输入神经网络构建的元先验，在合成任务上进行一次性预训练。列置换等变性结合原始列与敲除列的随机交错，产生了具有精确有限样本错误发现率控制的敲除统计量，即使元先验不准确时依然成立。我们为摊销式方法建立了相应的理论。

    arXiv:2610.21950v1 Announce Type: cross  Abstract: Tabular foundation models based on prior-data fitted networks (PFNs) predict on a new data set in a single forward pass, but they cannot report which inputs their predictions depend on. We propose SelectPFN, a PFN that performs Bayesian variable selection and prediction jointly for sparse nonlinear regression. In one forward pass it returns posterior inclusion probabilities for every variable, and its predictions are computed from sampled subsets only, so excluded variables provably cannot affect them; we show this is exactly what the Bayesian model prescribes. The network is pretrained once on synthetic tasks from a meta-prior built around a group spike-and-slab sparse-input neural network. Column-permutation equivariance combined with random interleaving of original and knockoff columns yields knockoff statistics with exact finite-sample false discovery rate control, even when the meta-prior is wrong. We develop theory for amortized 
    
[^9]: 倾斜采样问题：乐观原理、后验采样与对数遗憾

    The Tilted Sampling Problem: Optimism, Posterior Sampling, and Logarithmic Regret

    [https://arxiv.org/abs/2610.21908](https://arxiv.org/abs/2610.21908)

    该论文提出了倾斜采样这一在线学习问题，以温度归一化累积KL散度定义采样遗憾，并构建了一个通用遗憾框架，揭示倾斜采样与经典遗憾最小化具有不同的不确定性结构，借助乐观原理和后验采样可将对数级遗憾推广至从未知奖励诱导的吉布斯分布中采样的场景。

    

    许多现代学习系统的用途并非选择单一最优动作，而是采样多样化的高质量输出，例如熵正则化强化学习、RLHF（基于人类反馈的强化学习）以及奖励引导的生成建模。受这些场景启发，我们研究了倾斜采样：一个在线学习问题，其目标是从由未知奖励函数诱导的吉布斯分布中进行采样。我们将采样遗憾定义为学习者的分布与吉布斯目标分布之间经温度归一化的累积KL散度，它对应于熵正则化奖励最大化中的次优性差距。我们的框架推广了赌博机和强化学习中近期基于采样的以及KL正则化的多种表述。我们的主要贡献是一个通用的遗憾框架，它刻画了倾斜采样与经典遗憾最小化之间不同的不确定性结构。虽然标准遗憾的量级为 $\widetilde{O}(\sqr...（原文摘要在此处截断）

    arXiv:2610.21908v1 Announce Type: new  Abstract: Many modern learning systems are used not to select a single best action, but to sample diverse high-quality outputs. Examples include entropy-regularized reinforcement learning, RLHF, and reward-guided generative modeling. Motivated by these settings, we study tilted sampling: an online learning problem in which the goal is to sample from the Gibbs distribution induced by an unknown reward function. We define sampling regret as the temperature-normalized cumulative KL divergence between the learner's distribution and the Gibbs target, corresponding to the suboptimality gap in entropy-regularized reward maximization. Our framework generalizes recent sampling-based and KL-regularized formulations in bandits and reinforcement learning. Our main contribution is a general regret framework that captures the different uncertainty structure of tilted sampling and classical regret minimization. While standard regret scales as $\widetilde{O}(\sqr
    
[^10]: 专业化之前的上下文学习：可解主题模型中的学习理论

    In-Context Learning Before Specialisation: Theory of Learning in a Solvable Topic Model

    [https://arxiv.org/abs/2610.21901](https://arxiv.org/abs/2610.21901)

    该论文在一个可解的主题模型中建立了学习理论，从根本层面解释了序列模型为何以及何时会倾向于采用上下文学习而非存储训练数据的专门化外推。

    

    序列模型可以通过两种根本不同的方式进行泛化：一是外推训练中已经见过的结构，二是利用其提示来适应全新的任务。模型采取的策略决定了其在训练分布之外的行为方式，从而控制着潜在的能力与风险。近期的工作已在理论上刻画了多种不同的策略，并从实证角度描述了随着预训练数据质量与数量的变化，这些策略之间如何发生转变。然而，目前尚无针对这种竞争关系的学习理论，因此无法从根本上理解为什么一种解会比另一种解更受青睐。我们在一个可解的主题模型中提供了这样的理论——文档从有限的潜在主题混合分布中抽取，以此反映现代预训练数据的异质性。该数据集存在两种截然不同的解：一种是专门化预测器，它存储了有限的主题混合结构；另一种是上下文解，对于从未见过的主题而言是最优的。

    arXiv:2610.21901v1 Announce Type: cross  Abstract: Sequence models can generalise in fundamentally distinct ways: extrapolating structures seen in training, or using their prompt to adapt to novel tasks. The prevailing strategy determines how the model behaves beyond its training distribution, controlling potential capabilities and risks. Recent work has characterised many different strategies theoretically, and empirically described the transition between them as the quality and quantity of pretraining data vary. However, there is no learning theory for this competition, hence no fundamental understanding of \textit{why} one solution might be favoured over another. Here we provide such a theory in a solvable topic model---documents drawn from a finite mixture of latent topics, mirroring the heterogeneity of modern pretraining data. The dataset comes with two distinct solutions: a specialised predictor, that stores the finite topic mixture, and an in-context one, optimal for topics nev
    
[^11]: 通过优化确定性等价实现可证明高效的风险敏感离线强化学习

    Provably Efficient Risk-Sensitive Offline Reinforcement Learning via Optimized Certainty Equivalents

    [https://arxiv.org/abs/2610.21870](https://arxiv.org/abs/2610.21870)

    该论文针对通用风险度量类“优化确定性等价（OCE）”下的离线风险敏感强化学习提出了动态与静态两种形式化的悲观价值迭代算法，并建立了紧密的次线性次优性理论界限，首次对该离线场景给出了一般性的理论保证。

    

    风险敏感强化学习（RL）旨在处理不确定性下的决策问题以及不利结果带来的影响。当与环境进行额外交互的代价高昂时，从固定数据集中学习风险敏感策略就显得尤为重要。现有的风险敏感强化学习研究主要集中在线上场景或特定的风险度量上，导致对其离线场景的一般性理论理解仍不完善。在这项工作中，我们利用优化确定性等价（OCE）——一类通用的风险度量——来研究高效的离线风险敏感强化学习。我们考虑了两种形式化方式：动态OCE强化学习，它在每一步递归地应用OCE；以及静态OCE强化学习，它对累积奖励仅应用一次OCE。在表格型设置下，我们为这两种形式化方法都开发了悲观价值迭代算法，并建立了紧密的次线性次优性界限，明确刻画了由效用函数所编码的风险敏感性如何影响……

    arXiv:2610.21870v1 Announce Type: cross  Abstract: Risk-sensitive reinforcement learning (RL) addresses decision-making under uncertainty and adverse outcomes. Learning risk-sensitive policies from fixed datasets is important when additional interaction with the environment is costly. Existing work on risk-sensitive RL focuses on online settings or specific risk measures, leaving a general theoretical understanding of its offline setting underdeveloped. In this work, we investigate efficient offline risk-sensitive RL using the optimized certainty equivalent (OCE), a general class of risk measures. We consider two formulations: dynamic-OCE RL, which applies OCE recursively at each step, and static-OCE RL, which applies a single OCE to the cumulative rewards. Under the tabular setting, we develop pessimistic value-iteration algorithms for both formulations and establish tight sublinear suboptimality bounds, explicitly capturing how risk sensitivity, encoded by utility functions, affects 
    
[^12]: 基于期望最大化算法的逐元素污染与部分观测数据的鲁棒协方差估计

    Robust Covariance Estimation for Entry-wise Contaminated and Partially Observed Data via Expectation Maximization Algorithm

    [https://arxiv.org/abs/2610.21709](https://arxiv.org/abs/2610.21709)

    本文提出一种在缩减潜空间上运行的随机近似EM（SAEM）算法，通过将坐标级尺度的随机更新与Rao-Blackwell化的高斯条件矩相结合，实现了逐元素污染与缺失数据下Student-$t^*$模型的鲁棒协方差估计，无需对潜在高斯变量显式采样。

    

    arXiv:2610.21709v1 公告类型：cross 摘要：在各种信号处理应用中，逐元素污染和缺失数据下的鲁棒协方差估计是一个具有挑战性的问题。在Student-$t$模型下，观测级重加权会在样本间产生一致的衰减，即使大多数坐标仍包含有用信息，也可能导致累积的信息损失。另一种Student-$t^*$模型可以实现坐标级的鲁棒性，但其E步不存在闭式解。我们通过一个在缩减潜空间上运行的随机近似EM（SAEM）算法，将Student-$t^*$模型的推断扩展到不完整观测情形：将坐标级尺度的随机更新与Rao-Blackwell化的高斯条件矩相结合，在避免对潜在高斯变量进行显式采样的同时，保持了基于似然的框架。在合成数据和真实传感器数据上的实验表明，相比高斯、Student-$t$、插补方法，该方法具有更低的Fisher-Rao误差（摘要在此处被截断）。

    arXiv:2610.21709v1 Announce Type: cross  Abstract: Robust covariance estimation under entry-wise contamination and missing data is challenging in various signal processing applications. Under the Student-$t$ model, observation-level reweighting induces a uniform attenuation across samples, which can lead to cumulative information loss even when most coordinates remain informative. The alternative Student-$t^*$ model enables coordinate-wise robustness, but its E-step is not available in closed form. We extend inference for the Student-$t^*$ model to incomplete observations through a stochastic approximation EM (SAEM) algorithm operating on a reduced latent space: stochastic updates of the coordinate-wise scales are combined with Rao-Blackwellized Gaussian conditional moments, which avoids explicit sampling of latent Gaussian variables while preserving a likelihood-based framework. Experiments on synthetic and real sensor data show a lower Fisher-Rao error than Gaussian, Student-$t$, imp
    
[^13]: 经典方法的现代应用：使用轻量级离散响应连续潜变量模型生成合成表格数据

    Classical methods for modern applications: synthetic tabular data generation using lightweight discrete response continuous latent variable model

    [https://arxiv.org/abs/2610.21674](https://arxiv.org/abs/2610.21674)

    本文提出一种利用经验边际分布和分位数变换将含连续与离散变量的表格数据映射到连续潜空间的轻量级生成模型，能够高保真地生成合成表格数据，且实现简单、训练稳健、采样高效。

    

    在本文中，我们提出了一种轻量级算法，用于对同时包含连续变量和离散变量的表格数据进行建模。该方法使用经验边际分布和选定的分位数函数将数据变换到连续潜空间中，在该空间中可以高效地估计和应用所选的分布，例如多元高斯分布。虽然这一方法与经典成熟的机器学习方法密切相关，但我们提出了一种新颖的改进，使其能够很好地融入当代的计算生态系统。与深度架构不同，该方法实现简单、依赖项少、训练与估计稳健，且采样高效。该变换能够保留复杂的边际分布和相关性，提供高保真度的模型，尤其适用于单峰分布的数据。所提方法的关键特性之一是其连续性

    arXiv:2610.21674v1 Announce Type: new  Abstract: In this paper, we present a lightweight algorithm for modelling tabular data containing both continuous and discrete variables. The method uses empirical marginal distributions and a selected quantile function to transform the data into a continuous latent space, where a distribution of choice, e.g. a multivariate Gaussian, can be efficiently estimated and applied. While this approach is closely related to classical well-established machine learning methods, we present a novel adaptation that fits well within the contemporary computational ecosystem. Unlike in the case of deep architectures, it results in a simple implementation with few dependencies, robust training/estimation, as well as efficient sampling. The transformation preserves complex marginal distributions and correlations, providing high-fidelity models, especially for data with unimodal distributions. One of the key characteristics of the proposed approach is its continuous
    
[^14]: 条件均值流：面向基于模拟推断的少步数密度估计

    Conditional Mean Flows: Few-Step Densities for Simulation-Based Inference

    [https://arxiv.org/abs/2610.21657](https://arxiv.org/abs/2610.21657)

    提出条件均值流方法，将少步采样与可解析的密度评估相结合，大幅降低了基于模拟推断中后验采样及MCMC重复似然评估的计算成本。

    

    流匹配已成为基于模拟推断（SBI）中一类强大的生成模型族。然而，其采样和密度评估在推断阶段需要数值积分，这使得神经后验估计中的后验采样代价高昂。更重要的是，它会使重复的似然评估变得极其昂贵，对马尔可夫链蒙特卡洛（MCMC）采样而言不切实际，因为MCMC可能需要数千次重复的密度评估。在此，我们开发了用于后验和似然近似的条件均值流，将少步采样与可处理的密度评估相结合。我们证明均值流目标能够识别条件流映射，且独立掩码机制可直接估计边际似然。我们在多个任务上对后验估计和似然估计进行了评估，结果表明条件均值流在显著降低计算成本的同时，保留了基于流的密度估计器的灵活性。

    arXiv:2610.21657v1 Announce Type: cross  Abstract: Flow matching has emerged as a powerful generative family for simulation-based inference (SBI). However, sampling and density evaluation require numerical integration at inference time. This makes posterior sampling in neural posterior estimation costly. More importantly, it can make repeated likelihood evaluation prohibitively expensive and impractical for Markov chain Monte Carlo (MCMC) sampling, which may require thousands of repeated density evaluations. Here, we develop conditional mean flows for posterior and likelihood approximation, combining few-step sampling with tractable density evaluation. We show that the mean flow objective identifies conditional flow maps and that independent masking enables direct estimation of marginal likelihoods. We evaluate both posterior and likelihood estimation across several tasks and show that conditional mean flows retain the flexibility of flow-based density estimators while significantly re
    
[^15]: AI安全投资的市场激励

    Market Incentives for AI Safety Investment

    [https://arxiv.org/abs/2610.21607](https://arxiv.org/abs/2610.21607)

    该研究构建了包含上游大语言模型提供商和竞争性下游企业的AI供应链博弈论模型，发现下游竞争促使下游企业的安全投资超过行业利润最优水平，而上游提供商因无法充分获得安全投资创造的剩余而倾向于投资不足。

    

    尽管现代人工智能系统具有重要的实用价值，但它们也带来了重大的社会风险，包括毒性、恶意使用和虚假信息。这些风险的缓解不仅取决于技术可行性，还取决于开发和部署这些系统的企业的激励机制。我们展示了利润最大化如何塑造一个AI供应链中的安全投资，该供应链由一个上游大语言模型提供商和若干相互竞争的下游企业组成。在我们的博弈论模型中，我们推导出市场驱动的安全投资水平，并将其与最大化市场福利和行业利润的投资水平进行比较。我们发现，下游竞争通常会促使下游企业在安全方面的投资超过行业利润最优水平。相比之下，上游提供商的投资可能低于这些基准，因为其安全投资所带来的剩余未能得到充分补偿。最后，我们分析了不同的监管干预措施如何影响……

    arXiv:2610.21607v1 Announce Type: new  Abstract: Despite their practical significance, modern AI systems pose significant societal risks, including toxicity, malicious use, and misinformation. Their mitigation depends not only on technical feasibility but also on the incentives of firms that develop and deploy these systems. We show how profit maximization shapes safety investment in an AI supply chain with an upstream LLM provider and several competing downstream firms. In our game-theoretic model, we derive the market-driven level of safety investment and compare it with the levels that maximize market welfare and industry profit. We find that downstream competition generally induces downstream firms to invest more in safety than the industry-profit-optimal level. By contrast, the upstream provider may underinvest relative to these benchmarks because it is not fully compensated for the surplus from safety investments. Finally, we analyze how different regulatory interventions affect 
    
[^16]: 基于逻辑一致性的锐利且稳定的无标签校准证书

    Sharp and Stable Label-Free Calibration Certificates from Logical Coherence

    [https://arxiv.org/abs/2610.21567](https://arxiv.org/abs/2610.21567)

    本文利用极小极大定理从逻辑一致性出发，为查询条件概率预测器建立了锐利且稳定的无标签校准残差下界证书，并证明Lipschitz赌注形式既稳定又可通过在线梯度上升在独立同分布数据流上有效优化。

    

    以查询为条件的概率预测器每次回答一个逻辑相关的问题，而用于检验其校准性的标签常常缺失。针对对赌注者可使用的赌注信息的每一种限制，一个极小极大定理在答案具有有限支撑分布的条件下，确定了相应校准残差的锐利无标签下界。在不同事件的捆绑上，其均值、边缘与联合阶段可以严格分离，并且在不对其他问题作假设的前提下，一个捆绑的下界不能归因于某个单一问题。锐利的边缘下界关于答案是不连续的，且在Wasserstein距离下连续的任何无标签下界都无法保持在其常数倍以内。Lipschitz赌注给出一个稳定的下界，它对光滑的校准残差是锐利的，并且对每个正的Lipschitz常数，它为正当且仅当锐利边缘下界为正。在独立同分布的数据流上，在固定节点处进行投影在线梯度上升可以（原文摘要在此处截断）

    arXiv:2610.21567v1 Announce Type: new  Abstract: Query-conditioned probabilistic predictors answer logically related questions one at a time, and labels to check their calibration are often missing. For each restriction on what a bettor's stakes may use, a minimax theorem identifies, for finitely supported laws of the answers, the sharp label-free lower bound on the corresponding calibration residual. On bundles of different events its mean, marginal and joint stages can separate strictly, and a bundle's bound cannot be attributed to one question without assumptions on the others. The sharp marginal bound is discontinuous in the answers, and no label-free lower bound continuous in the Wasserstein distance stays within a constant factor of it. Lipschitz stakes give a stable bound, sharp for a smooth calibration residual and, for every positive Lipschitz constant, positive exactly when the sharp marginal bound is. On i.i.d. streams, projected online gradient ascent at fixed nodes gives f
    
[^17]: Muon何时最优？径向脆弱性与极小极大几何

    When Is Muon Optimal? Radial Fragility and Minimax Geometry

    [https://arxiv.org/abs/2610.21549](https://arxiv.org/abs/2610.21549)

    该论文引入随机半径各向同性曲率模型与极化因子容许集，精确刻画了Muon缩放极化更新在任意径向定律下保持最优的几何条件，证明非标量最优响应仅来自与曲率扭结对齐的径向原子，因而无原子径向定律呈现零容量刚性。

    

    Muon优化器通过用近似极化因子替换矩阵梯度，在神经网络训练中展现出强大的实证性能。各向同性曲率模型为这一更新提供了基于曲率的解释，其核心问题是：梯度正交化何时能最小化单步局部损失代理函数。我们研究了当ICM中的固定输入半径被一般径向定律取代时，这一解释的稳定性。我们引入了随机半径各向同性曲率模型（RR-ICM）和极化因子容许集，后者将精确的缩放极化最优性编码为梯度正极化因子上的一个几何条件。我们针对任意径向定律推导出了该容许集的精确刻画，并证明所有非标量的容许响应都源自与曲率扭结对齐的径向原子。因此，每一个无原子径向定律都表现出零容量刚性：精确的缩放极化最优性仅可能在……

    arXiv:2610.21549v1 Announce Type: cross  Abstract: Muon replaces a matrix gradient by an approximate polar factor and has shown strong empirical performance in neural-network training. The Isotropic Curvature Model (ICM) provides a curvature-based explanation for this update by asking when gradient orthogonalization minimizes a one-step local-loss surrogate. We investigate the stability of this explanation when the fixed input radius of the ICM is replaced by a general radial law. We introduce the Random-Radius Isotropic Curvature Model (RR-ICM) and the Polar-Factor Admissibility Set, which encodes exact scaled-polar optimality as a geometric condition on the positive polar factor of the gradient. We derive an exact characterization of this set for arbitrary radial laws and show that all nonscalar admissible responses arise from radial atoms aligned with curvature kinks. Consequently, every atomless radial law exhibits zero-capacity rigidity: exact scaled-polar optimality is possible o
    
[^18]: 测试时大语言模型集成中的两个样本量：缓冲经验风险最小化与极小极大理论

    Two Sample Sizes in Test-Time LLM Ensembling: Buffered ERM and Minimax Theory

    [https://arxiv.org/abs/2610.21363](https://arxiv.org/abs/2610.21363)

    该论文研究了测试时大语言模型加权集成中标注问题数 Q 与每问题生成数 N 这两个样本量的相互作用，证明带校准间隔缓冲的 ERM 可达到 Q^{-1/2}+N^{-β/2} 阶的超额风险，并通过简化的双模型子模型分析验证了该风险率的最优性。

    

    测试时扩展可以通过为一个问题生成多个答案并对其进行聚合来提升大语言模型的性能。加权集成进一步拓展了这一思想，即从多个模型中抽取答案，其中权重决定了每个模型被使用的频率。从训练数据集中学习这些权重涉及两个样本量：Q 个带标注的问题，以及每个模型对每个问题进行的 N 次生成。我们通过干净风险来研究这两个样本量之间的相互作用，干净风险即该集成在无限次生成下的投票规则的总体误差。在指数为 β 的间隔条件下，我们证明带有校准间隔缓冲的经验风险最小化（ERM）方法，在模型数量和候选答案数量固定的情况下，可以达到量级为 Q^{-1/2} + N^{-β/2} 的超额风险（仅相差对数因子）。为了探究这种对 Q 和 N 的依赖关系是否最优，我们分析了一个简化的子模型，该模型包含两个模型且每个问题只有两个可能的答案。我们……

    arXiv:2610.21363v1 Announce Type: cross  Abstract: Test-time scaling can improve the performance of large language models by generating multiple answers to a question and aggregating them. Weighted ensembles extend this idea by drawing answers from several models, with the weights determining how often each model is used. Learning these weights from a training dataset involves two sample sizes: $Q$ labeled questions and $N$ generations per model and question. We study their interplay through the clean risk, the population error of the ensemble's infinite-generation voting rule. Under a margin condition with exponent $\beta$, we show that empirical risk minimization (ERM) with a calibrated margin buffer achieves excess risk of order $Q^{-1/2}+N^{-\beta/2}$, up to logarithmic factors, for fixed numbers of models and candidate answers. To investigate the optimality of this dependence on $Q$ and $N$, we analyze a simplified submodel with two models and two possible answers per question. We
    
[^19]: 通过模型无关的重要性度量控制错误发现率

    Controlling the False Discovery Rate via Model-Agnostic Importance Measures

    [https://arxiv.org/abs/2610.21360](https://arxiv.org/abs/2610.21360)

    本文提出了MAS方法，通过样本分割策略和新颖的agent-X变量重要性度量，在无需对协变量做分布假设的情况下实现错误发现率（FDR）控制下的模型无关变量选择，与Model-X knockoffs形成互补。

    

    在控制错误发现率（FDR）的同时选择协变量对于可解释的科学发现至关重要，尤其是在使用复杂机器学习模型的场景下。传统的变量选择方法通常依赖于特定的模型假设，而Model-X knockoffs则提供了一种通用方法，利用协变量分布的知识来构建变量选择算法。本研究提出了一种与Model-X knockoffs互补的方法，称为基于agent-X重要性的模型无关选择（MAS），该方法利用样本分割策略和新颖的变量重要性度量，从而绕过对协变量分布的假设。所提出的模型无关agent-X重要性被定义为在总体水平上评估原始分布与加入agent变量后的分布之间oracle预测能力的差异，并且可以通过将agent变量构造为某种形式的变量来高效估计（摘要在此处被截断）。

    arXiv:2610.21360v1 Announce Type: cross  Abstract: Selecting covariates while controlling the false discovery rate (FDR) is critical for interpretable scientific discovery, especially with complex machine learning models. Traditional variable selection methods often rely on specific model assumptions, while Model-X knockoffs provide a general way that leverages knowledge of the covariate distribution to produce a variable selection algorithm. This study puts forward a complementary method to Model-X knockoffs, named Model-Agnostic Selection with agent-X importance (MAS), which utilizes the sample-splitting strategy and a novel variable importance measure to bypass distributional assumptions on covariates. The proposed model-agnostic agent-X importance is defined to evaluate the difference of oracle predictiveness on both the original distribution and the distribution with agent variable in population level, and can be efficiently estimated by constructing the agent variable as some con
    
[^20]: 基于狄利克雷过程的深度神经网络维度无关PAC-Bayes泛化

    Dimension-Free PAC-Bayes Generalization for Deep Neural Networks via Dirichlet Processes

    [https://arxiv.org/abs/2610.21340](https://arxiv.org/abs/2610.21340)

    该论文通过狄利克雷过程将PAC-Bayes分析的先验从高维参数空间转移到数据分布空间，并借助数据处理不等式证明参数空间相对熵被可闭式计算的数据空间DP散度严格界定，从而获得了与网络维度无关的深度神经网络泛化界，解决了过参数化网络泛化界空洞的问题。

    

    arXiv:2610.21340v1 公告类型：cross 摘要：经典的PAC-Bayesian泛化理论将先验分布置于欧氏参数空间 $\Theta \subseteq \mathbb{R}^D$ 上，导致相对熵随网络维度 $D$ 增长。对于现代过参数化神经网络（其中 $D \gg n$），这种基于参数计数的依赖性产生了空洞的泛化界（超过1,000%）。在本工作中，我们通过狄利克雷过程 $\mathcal{DP}(\alpha, F_0)$ 将先验设定从内部参数转移到数据分布的非参数空间 $\mathcal{M}_1(\mathcal{Z})$，并通过学习算法下的前推测度诱导出参数分布，从而突破了这一障碍。根据数据处理不等式，参数空间的相对熵被数据空间的DP散度严格上界约束，我们利用Digamma函数以精确闭式形式对该散度进行计算。所得到的PAC-Bayesian证书对网络参数严格不变（摘要在此处截断）。

    arXiv:2610.21340v1 Announce Type: cross  Abstract: Classical PAC-Bayesian generalization theory places prior distributions over Euclidean parameter space $\Theta \subseteq \mathbb{R}^D$, causing relative entropy to scale with network dimension $D$. For modern overparameterized neural networks where $D \gg n$, this parameter-counting dependency yields vacuous generalization bounds ($> 1{,}000\%$). In this work, we resolve this barrier by shifting prior specification from internal parameters to the non-parametric space of data distributions $\mathcal{M}_1(\mathcal{Z})$ via a \textbf{Dirichlet Process (DP)}, $\mathcal{DP}(\alpha, F_0)$, inducing parameter distributions through push-forward measures under the learning algorithm. By the Data Processing Inequality, parameter-space relative entropy is strictly bounded by the data-space DP divergence, which we evaluate in exact closed form using Digamma functions. The resulting PAC-Bayesian certificate is strictly invariant to network paramete
    
[^21]: 基于对数损失的量子态在线学习的遗憾下界，及其在量子态层析成像与矩阵均值置信序列中的应用

    Regret Lower Bound for Online Learning of Quantum States with Logarithmic Loss, with Applications to Quantum State Tomography and Confidence Sequences for Matrix Means

    [https://arxiv.org/abs/2610.21273](https://arxiv.org/abs/2610.21273)

    本文建立了基于对数损失的量子态在线学习的最优遗憾下界，并由此导出样本量最优（至多相差多对数因子）的量子态层析成像算法，以及矩阵均值置信序列的构造算法。

    

    我们研究基于对数损失的量子态在线学习问题，该问题是在线投资组合选择的量子推广，并应用于量子态层析成像，即从测量结果中估计未知量子态的任务。我们首先建立了一个遗憾下界，表明指数权重算法和Tseng等人（2025）提出的算法的遗憾率在维度上具有最优的多项式依赖性。我们的分析还带来了两个进一步的结果。首先，我们的下界证明导出了一种量子态层析成像算法，其样本复杂度在样本量方面达到最优（仅相差多对数因子）。其次，我们开发了一种用于构造矩阵值随机变量均值的置信序列的算法。数值实验表明，后一种算法与现有方法相比具有竞争力。

    arXiv:2610.21273v1 Announce Type: new  Abstract: We study the problem of online learning of quantum states with logarithmic loss, which is a quantum generalization of online portfolio selection and has applications to quantum state tomography, the task of estimating an unknown quantum state from measurement outcomes. We first establish a regret lower bound, showing that the regret rates of Exponential Weights and the algorithm of Tseng et al. (2025) have optimal polynomial dependence on the dimension. Our analysis leads to two further results. First, our lower-bound proof leads to a quantum state tomography algorithm that is sample optimal up to polylogarithmic factors in the sample size. Second, we develop an algorithm for constructing confidence sequences for the mean of a matrix-valued random variable. A numerical experiment shows that the latter algorithm is competitive with existing methods.
    
[^22]: CleanRM：利用少量干净反馈从噪声反馈中进行RLHF奖励建模

    CleanRM: Reward Modeling from Noisy Feedback with a Few Clean Feedback for RLHF

    [https://arxiv.org/abs/2610.21138](https://arxiv.org/abs/2610.21138)

    提出CleanRM方法，利用少量干净反馈并通过最优传输校准逐样本噪声比例，实现从噪声反馈中对RLHF奖励模型的无偏学习。

    

    反馈噪声在偏好数据中普遍存在，其来源包括主观的人类判断、标注错误以及不完善的LLM裁判。在此类反馈上训练的标准奖励建模会产生对干净风险的有偏估计。尽管经典的去噪方法可以直接应用于噪声反馈学习，但它们通常假设样本间具有统一的噪声比例。然而，这一假设对反馈数据并不成立，因为反馈数据中的噪声本质上是逐样本的。为应对这一挑战，我们提出了CleanRM，它利用少量干净反馈通过最优传输来校准逐样本的噪声比例。具体而言，CleanRM将噪声反馈还原出的偏好联合分布与干净反馈的偏好联合分布对齐，然后使用噪声校正的代理损失来训练奖励模型。在理论上，我们证明了所得目标相对于干净风险是无偏的。大量实验表明，CleanRM

    arXiv:2610.21138v1 Announce Type: new  Abstract: Feedback noise is pervasive in preference data, arising from subjective human judgments, annotation mistakes, and imperfect LLM judges. Standard reward modeling trained on such feedback yields a biased estimate of the clean risk. Although classical denoising methods are directly applicable to noisy feedback learning, they generally assume a uniform noise ratio across samples. However, this assumption fails for feedback data, where noise is intrinsically sample-wise. To address this challenge, we propose CleanRM, which uses a few clean feedback to calibrate sample-wise noise ratios via optimal transport. Specifically, CleanRM aligns the joint distribution of preferences reverted from noisy feedback with that of clean feedback and then trains the reward model using a noise-corrected surrogate loss. Theoretically, we prove that the resulting objective is unbiased with respect to the clean risk. Extensive experiments demonstrate that CleanRM
    
[^23]: 基于反向KL张量化的重尾单边McDiarmid型不等式

    A One-Sided McDiarmid-Type Inequality for Heavy Tails via Reverse KL Tensorization

    [https://arxiv.org/abs/2610.21103](https://arxiv.org/abs/2610.21103)

    本文提出了一种新的反向KL张量化不等式，证明了对几乎必然有上界且仅满足 $p\in(1,2]$ 阶条件矩的独立随机变量函数，均值上方的大偏差呈指数级不可能，即使方差无限也成立，并应用于无界损失的单边一致风险界。

    

    我们重新审视了McDiarmid不等式，它是统计学习理论和算法分析中的一个核心工具。我们证明，对于几乎必然有上界且满足阶数为 $p\in(1,2]$ 的条件矩假设的独立随机变量的函数，偏离均值上方的大偏差呈指数级不可能。当 $p<2$ 时，即使函数具有无限方差，该结果依然适用。我们的证明依赖于Kullback–Leibler散度的一个新的张量化不等式，称为反向KL张量化，这一结果可能具有独立的研究价值。应用包括样本均值和中位数均值估计量的集中性，以及无界损失的单边一致风险界。

    arXiv:2610.21103v1 Announce Type: cross  Abstract: We revisit McDiarmid's inequality, a central tool in statistical learning theory and algorithm analysis. We show that large deviations above the mean are exponentially unlikely for functions of independent random variables that are almost surely bounded above and satisfy a conditional moment assumption of order $p\in(1,2]$. When $p<2$, the result applies even when the function has infinite variance. Our proof relies on a new tensorization inequality for the Kullback--Leibler divergence, termed reverse KL tensorization, which may be of independent interest. Applications include concentration of the sample mean and median-of-means estimator, and one-sided uniform risk bounds for unbounded losses.
    
[^24]: StochLOB：用于稳健限价订单簿趋势预测的重尾分数匹配

    StochLOB: Heavy-Tailed Score Matching for Robust Limit Order Book Trend Prediction

    [https://arxiv.org/abs/2610.21081](https://arxiv.org/abs/2610.21081)

    该论文提出StochLOB训练框架，用跳跃Lévy噪声和截断α稳定噪声等重尾非高斯扩散过程替代传统的布朗高斯噪声来加噪训练数据，以更好地捕捉限价订单簿订单流的重尾与跳跃特性，从而提升LOB趋势预测的稳健性，同时推理阶段保持确定性预测器。

    

    基于分数的扩散方法通常使用布朗高斯噪声对数据进行扰动，这对应于轻尾的前向随机微分方程（SDE）。对于市场时间序列而言，这种SDE可能是一种糟糕的抽象：限价订单簿（LOB）的订单流具有重尾性、突发性，且偶尔呈现跳跃状。我们研究了STOCHLOB，这是一个仅用于训练阶段的分数匹配框架，它改变了前向扩散的损坏规律，同时保持部署的预测器为确定性模型。该加噪过程不再仅使用布朗高斯损坏，而是采用两种受非高斯SDE启发的替代方案：通过复合泊松方差混合实现的跳跃Lévy噪声，以及通过正稳定高斯尺度混合实现的截断α稳定噪声。这些损坏机制在训练中注入了有限跳跃和重尾残差，而无需在推理阶段进行反向时间扩散采样。所得的分数目标具有一个有用的解释：其径向因子h（摘要在此处截断）

    arXiv:2610.21081v1 Announce Type: new  Abstract: Score-based diffusion methods usually perturb data with Brownian Gaussian noise, corresponding to a light-tailed forward stochastic differential equation (SDE). For market time series this SDE can be a poor abstraction: limit order book (LOB) order flow is heavy-tailed, bursty, and occasionally jump-like. We study STOCHLOB, a training-only score-matching framework that changes the forward diffusion corruption law while leaving the deployed predictor deterministic. Instead of Brownian-only Gaussian corruption, the noising process uses two non-Gaussian SDE-inspired alternatives: jump-L\'evy noise through a compound-Poisson variance mixture, and clipped $\alpha$-stable noise through a positive-stable Gaussian scale mixture. These corruptions inject finite jumps and heavy-tailed residuals into training without requiring reverse-time diffusion sampling at inference. The resulting score target has a useful interpretation: its radial factor $h_
    
[^25]: 零样本条件离散扩散中的分数失配与集合约束采样

    Score Mismatch and Set-Constrained Sampling in Zero-Shot Conditional Discrete Diffusion

    [https://arxiv.org/abs/2610.21061](https://arxiv.org/abs/2610.21061)

    本文首次对分数失配的离散扩散模型进行了理论分析，推导出一般的KL散度上界来量化失配造成的额外偏差，并覆盖文本填充、句子补全等部分观测约束场景，为零样本条件生成提供了理论保证。

    

    离散扩散模型已成为在众多应用中建模复杂离散分布的强大框架。尽管理论收敛保证依赖于目标分布与训练分布之间的一致性，但这种情况在实践中很少成立，这使得分数失配成为离散扩散采样中的核心挑战。这一问题在零样本条件生成中尤为突出，即在推理阶段直接将在无条件分布上训练的模型重用于从条件目标分布中采样。本文对分数失配的离散扩散模型提供了首次理论分析，并推导出一般的KL散度上界，该上界清晰地刻画了由这种失配引起的额外偏差。我们的分析涵盖了部分观测约束，包括文本填充、句子补全以及更一般的逐坐标集合约束。

    arXiv:2610.21061v1 Announce Type: cross  Abstract: Discrete diffusion models have become a powerful framework for modeling complex discrete distributions across a range of applications. While theoretical convergence guarantees depend on consistency between the target and training distributions, such a case rarely holds in practice, making score mismatch a central challenge in discrete diffusion sampling. This is particularly relevant to zero-shot conditional generation, where a model trained on the unconditional distribution is directly reused at inference time to sample from a conditional target distribution. In this paper, we provide the first theoretical analysis for score-mismatched discrete diffusion models and derive a general KL divergence upper bound that clearly identifies the additional bias caused by such a mismatch. Our analysis covers partial-observation constraints, including text infilling, sentence completion, and more general coordinatewise set constraints. Specificall
    
[^26]: 局部差分隐私下单变量密度的高效不可知学习

    Efficient Agnostic Learning of Univariate Densities Under Local Differential Privacy

    [https://arxiv.org/abs/2610.21033](https://arxiv.org/abs/2610.21033)

    本文提出了一个非交互式局部差分隐私学习框架，实现了对可被分段多项式逼近的单变量分布（如高斯混合模型）首个样本高效且计算高效的不可知学习，并为此开发了可独立使用的大域LDP合成数据生成算法。

    

    我们提出了一个用于非交互式局部差分隐私（LDP）学习的通用框架，适用于可由分段多项式逼近的结构化单变量分布。这为若干分布类提供了首个样本高效的LDP学习器，包括对数凹分布的混合分布，例如高斯混合模型（GMMs）。这些学习器还提供不可知保证，并且在计算上是高效的。我们还证明了在LDP下学习GMM的一个下界，改进了Bun等人[BKSW19]的标准纯DP装箱下界。为了构建我们的通用学习框架，我们开发了一种针对大域上合成数据生成的高效非交互式LDP算法，该算法可能具有独立的研究价值。随后，我们将其与Acharya等人[ADLS17]的非私有分段多项式学习器的确定性改编版本结合使用。

    arXiv:2610.21033v1 Announce Type: cross  Abstract: We propose a general framework for non-interactive locally differentially private (LDP) learning of structured univariate distributions that can be approximated by piece-wise polynomials. This gives the first sample-efficient LDP learners for several distribution classes, including mixtures of log-concave distributions, such as Gaussian mixture models (GMMs). These learners also provide agnostic guarantees and are computationally efficient. We also prove a lower bound for learning GMMs under LDP, improving on the standard pure-DP packing lower bounds of Bun et al. [BKSW19]. To obtain our general learning framework, we develop an efficient non-interactive LDP algorithm for synthetic data generation over large domains, which may be of independent interest. We then use it together with a deterministic adaptation of the non-private piecewise polynomial learner of Acharya et al. [ADLS17].
    
[^27]: 论裁剪梯度方法的末次迭代收敛性

    On the Last-Iterate Convergence of Clipped Gradient Methods

    [https://arxiv.org/abs/2610.21026](https://arxiv.org/abs/2610.21026)

    本文针对梯度噪声为重尾分布这一更贴近实际的情形，研究并证明了裁剪梯度方法（如Clipped-SGD）在随机（强）凸优化中的末次迭代收敛性，克服了现有SGD末次迭代收敛结果依赖强噪声条件（如指数衰减尾部）的局限。

    

    在随机（强）凸优化中，为一阶方法建立末次迭代收敛保证近来受到越来越多的关注，因为这不仅加深了对这些算法的理论理解，而且与实践更为贴近。值得注意的是，人们已经证明，最著名、简单且流行的随机梯度下降（SGD）方法能够可证明地实现末次迭代收敛。然而，现有结果的适用性可能有限，因为它们通常依赖于对梯度噪声的较强条件（例如指数衰减的尾部），而非更现实的重尾噪声。不幸的是，面对重尾噪声时，已知SGD会表现出不良行为甚至无法收敛。为应对重尾噪声的挑战，人们提出了Clipped-SGD，这是一种将……（原文摘要在此处截断）

    arXiv:2610.21026v1 Announce Type: cross  Abstract: In stochastic (strongly) convex optimization, establishing last-iterate convergence guarantees for first-order methods has recently garnered increasing attention, as this not only deepens the theoretical understanding of these algorithms but also aligns more closely with practice. Notably, people have demonstrated that one of the most famous, simple, and popular methods, Stochastic Gradient Descent ($\mathtt{SGD}$), provably achieves last-iterate convergence. However, the existing results may have limited applicability, since they typically rely on strong conditions on the gradient noise (e.g., an exponentially decaying tail) rather than the more realistic heavy-tailed noise. Unfortunately, when faced with heavy-tailed noise, $\mathtt{SGD}$ is known to exhibit undesirable behavior or even fail to converge. To address the challenge of heavy-tailed noise, people have proposed $\mathtt{Clipped}\text{-}\mathtt{SGD}$, an algorithm that comb
    
[^28]: 高斯变分推断中的加速及其极限

    Acceleration and its Limits in Gaussian Variational Inference

    [https://arxiv.org/abs/2610.21020](https://arxiv.org/abs/2610.21020)

    该论文证明高斯变分推断在可精确计算高斯期望时能达到经典的 √κ 加速收敛率，但通过构造一族强对数凹目标证明仅有逐点导数时任何算法都需要 κ^(1-o(1)) 次查询，从而说明仅凭曲率界不足以保证加速。

    

    我们研究了经典的基于动量的加速方法——对于强凸光滑目标，它能将梯度下降对条件数的依赖从 κ 改进到 √κ——能否推广到针对强对数凹、对数光滑目标分布的高斯变分推断（GVI）。在可以精确计算高斯期望的情况下，GVI 的凸复合形式能够达到经典的 √κ 加速收敛率。然而，当仅有逐点导数可用时，我们构造了一族强对数凹的目标分布，证明任何随机化算法都需要 κ^(1-o(1)) 次查询才能达到固定的目标精度，即使初始目标差距与 κ 无关地有界也是如此。因此，仅凭曲率界无法保证一致加速的总查询复杂度。我们随后探讨何种附加结构可以允许加速，并给出了正例：在 Hessian-Lipschitz 假设下……

    arXiv:2610.21020v1 Announce Type: cross  Abstract: We study whether the classical momentum-based acceleration that improves the condition-number dependence of gradient descent from $\kappa$ to $\sqrt{\kappa}$ for strongly convex, smooth objectives extends to Gaussian variational inference (GVI) for strongly log-concave, log-smooth target distributions. With exact Gaussian expectations, a convex composite formulation of GVI yields the classical accelerated $\sqrt{\kappa}$ convergence rate. However, when only pointwise derivatives are available, we construct a family of strongly log-concave targets for which any randomized algorithm requires $\kappa^{1-o(1)}$ queries to achieve a fixed objective accuracy, even when the initial objective gap is bounded independently of $\kappa$. Thus curvature bounds alone cannot guarantee uniform accelerated total-query complexity. We then ask which additional structure permits acceleration and give positive examples: under a Hessian-Lipschitz assumption
    
[^29]: 失败乃成功之母：以接地式后见之明恢复监督信号

    Failure Is the Mother of Success: Recovering Supervision with Grounding Hindsight

    [https://arxiv.org/abs/2610.21018](https://arxiv.org/abs/2610.21018)

    提出 RSHD 方法，让策略为 GRPO 中的失败尝试撰写纠正性笔记并用其重解问题，仅当重放成功率超过原始组时才将笔记蒸馏回策略，从而从全组失败的难题中恢复监督信号。

    

    当一个问题上的所有 rollout 全部失败时，GRPO 既得不到成功范例，也无法获得非零的组相对优势，因此它恰恰在最需要学习的问题上学不到任何东西。然而，失败的推理过程仍然能够提示下一步应该尝试什么。我们提出了重放筛选式后见之明蒸馏（Replay-Selected Hindsight Distillation, RSHD），它在 GRPO 训练过程中将失败尝试转化为一种额外的、经过检验的监督来源。策略在不查看参考答案或标准解答的情况下，为每一次失败尝试撰写一条纠正性笔记，然后带着这条笔记重新解题。只有当笔记的重放成功率高于原始组、并且在筛除那些直接给出答案的笔记之后，该笔记才会被采纳；被采纳的重放随后会被蒸馏进纯提示词策略中。由于采纳与否由全新的成功表现决定，而非由失败的 rollout 本身决定，该流程能够从所有尝试均告失败的组中恢复出监督信号。

    arXiv:2610.21018v1 Announce Type: new  Abstract: When every rollout for a problem fails, GRPO receives neither a successful exemplar nor a nonzero group-relative advantage, so it learns nothing from exactly the problems it most needs to learn. The failed reasoning can nevertheless suggest what to try next. We present Replay-Selected Hindsight Distillation (RSHD), which turns failed attempts into an additional, tested source of supervision during GRPO training. The policy writes a corrective note about each failed attempt without seeing a reference answer or solution, then re-solves the problem with the note in context. A note is admitted only if its replays succeed more often than the original group did, after screening for notes that state the answer, and an admitted replay is distilled into the plain-prompt policy. Because admission is decided by fresh successes rather than by the failed rollout itself, the procedure can recover supervision from groups in which every attempt failed, 
    
[^30]: 先减半后贪心：具有近最优实例依赖性的后验成功自适应采样

    Halve-Then-Greedy: Adaptive Sampling for Posterior Success with Near-Optimal Instance Dependence

    [https://arxiv.org/abs/2610.20997](https://arxiv.org/abs/2610.20997)

    提出“先减半后贪心”（HTG）自适应采样算法，在固定预算下最大化推荐设计表现超过目标的后验证据概率，并证明了实例依赖的指数级失败概率界，其样本复杂度在单正臂情形下达到近最优（仅差一个对数因子）。

    

    我们研究固定预算下的自适应实验问题，旨在最大化获得充分后验证据的概率，即推荐设计的均值表现超过目标值。与最优臂识别不同，该目标无需找到最优设计。对于具有高斯先验的高斯老虎机问题，我们提出了“先减半后贪心”（HTG）算法，该算法结合了大规模初始均衡分配、逐次减半和后验贪心采样。我们证明了后验失败概率的实例依赖、有限预算指数界，该界同时刻画了与不利臂的分离程度以及多个有利臂的存在。在单正臂问题上，我们建立了渐近样本复杂度下界，HTG 在相差一个对数因子的范围内与之匹配。我们进一步分析了均匀采样和逐次减半方法，揭示了在某些情形下它们分别需要比 HTG 多出线性因子和对数因子数量的样本。

    arXiv:2610.20997v1 Announce Type: cross  Abstract: We study fixed-budget adaptive experimentation aimed at maximizing the probability of obtaining sufficient posterior evidence that a recommended design's mean performance exceeds a target. Unlike best-arm identification, this objective does not require finding the best design. For Gaussian bandits with a Gaussian prior, we propose \emph{Halve-Then-Greedy} (HTG), which combines a large initial balanced allocation, successive halving, and posterior-greedy sampling. We prove instance-dependent, finite-budget exponential bounds on posterior-failure probability that capture both separation from unfavorable arms and the presence of multiple favorable arms. On one-positive-arm families, we establish an asymptotic sample-complexity lower bound that HTG matches up to a logarithmic factor. We further analyze uniform sampling and sequential halving, revealing regimes where they require more samples than HTG by linear and logarithmic factors, resp
    
[^31]: 面向群体特定扩散模型的两阶段得分估计

    Two-Stage Score Estimation for Group-Specific Diffusion Models

    [https://arxiv.org/abs/2610.20979](https://arxiv.org/abs/2610.20979)

    该论文提出了一个针对群体特定扩散模型的两阶段得分估计统计框架，通过将共享池化得分与目标特定偏移量分离分析，推导出收敛速率，并证明扩散模型可作为密度估计器获得全变差保证。

    

    跨相关分布的生成建模提出了一个问题：共享结构如何在观测有限的情况下支持对目标分布的准确生成？我们开发了一个用于扩散迁移的统计框架，分析共享的池化得分与目标特定的偏移量。在池化密度满足Besov正则性以及偏移量满足稳定分层复合模型的条件下，我们推导出能够分离学习这两个组件成本的收敛速率，并识别出何时更简单的偏移量可以产生比单独拟合更快的风险界速率。在适当的边界和正则性假设下，我们证明随着扩散时间趋于零，偏移量保持有界且具有联合正则性。这些结果使得下时间截断能够以多项式方式趋于消失，并为扩散模型作为密度估计器提供了全变差保证。合成模拟和图像基准测试提供了概念验证。

    arXiv:2610.20979v1 Announce Type: cross  Abstract: Generative modeling across related distributions raises a question: how can shared structure support accurate generation for a target distribution with limited observations? We develop a statistical framework for diffusion transfer that analyzes a shared pooled score together with a target-specific offset. Under Besov regularity of the pooled density and a stable hierarchical composition model for the offset, we derive convergence rates that separate the costs of learning these components and identify when a simpler offset yields a faster risk-bound rate than separate fitting. Under suitable boundary and regularity assumptions, we prove that the offset remains bounded and jointly regular as diffusion time approaches zero. These results allow the lower time cutoff to vanish polynomially and provide total-variation guarantees for diffusion models as density estimators. Synthetic simulations and image benchmarks provide a proof of concept
    
[^32]: 排序正确，量级失准：集合离散度在径流极端事件中悄然失效

    Ranked Right, Sized Wrong: Ensemble Spread Fails Silently at Streamflow Extremes

    [https://arxiv.org/abs/2610.20947](https://arxiv.org/abs/2610.20947)

    研究发现深度学习径流预报模型的集合离散度虽能正确排序预测误差，但其量级随流量增大而系统性低估误差（平常流量下差2倍、极端流量下可达18倍），表明该不确定性信号在径流极端事件中会悄然失效。

    

    深度学习模型，尤其是长短期记忆网络，正越来越多地用于业务化径流预报，其集合离散度可提供预测不确定性的信号。随着气候变化导致水文气候极端事件频率增加，这些模型将越来越多地在历史记录中罕见的流量条件下运行。因此，我们评估了集合离散度在流量增大时是否仍具有信息量，从其相对于误差的量级以及对误差进行排序的能力两方面进行考察。结果显示：在平常流量下，预测误差约为校正后离散度的2倍，在最高流量下高达4倍，且该偏差随流量增大而加剧（对数-对数斜率1.30，95%置信区间：[1.20, 1.41]）。欠离散现象在超出训练记录的观测极端流量中最为严重，偏差可达18倍。这一现象普遍存在，且无法用条件均值偏差来解释。Area Under t（摘要原文在此处截断，后文缺失）

    arXiv:2610.20947v1 Announce Type: cross  Abstract: Deep learning models, particularly Long Short-Term Memory networks, are increasingly used for operational streamflow forecasting, with ensemble spread providing a signal of predictive uncertainty. As climate change increases the frequency of hydroclimatic extremes, these models will increasingly operate under flow conditions that are rare in their historical records. We therefore evaluate whether ensemble spread remains informative under increasing flows, examining both its magnitude relative to error and its ability to rank errors. Prediction error is approximately $2\times$ the corrected spread at ordinary flows and up to $4\times$ at the highest flows, with the discrepancy increasing with flow (log-log slope $1.30$, 95% CI: $[1.20,1.41]$). Under-dispersion is strongest in observed extreme flows beyond the training record, reaching an $18\times$ discrepancy. This is pervasive and not explainable by conditional mean bias. Area Under t
    
[^33]: 面向误设回归的符号投注方法：方差自适应与最优逼近

    Sign Betting for Misspecified Regression: Variance Adaptivity and Optimal Approximation

    [https://arxiv.org/abs/2610.20903](https://arxiv.org/abs/2610.20903)

    本文证明符号投注回归在模型误设下可实现高概率的 $L_1$ 预言机不等式，其误差界自适应于平均条件方差，并证明其对恰当学习所能达到的普适逼近因子 3 是最优的。

    

    我们研究模型误设下的有界回归问题，通过函数 $f$ 与条件均值 $\eta(x)=\mathbb{E}[Y\mid X=x]$ 之间的 $L_1(P_X)$ 距离来评估函数，而非采用标准的 $L_2(P_X)$ 距离。这一 $L_1$ 度量对多臂老虎机和强化学习等下游决策问题而言是自然的。我们分析了 Baraud 与 Maillard（2025）提出的极小极大估计器，并将其称为“符号投注回归”。我们的主要结果表明，符号投注回归在模型误设情形下能够实现高概率的 $L_1$ 预言机不等式。在可实现性条件下，我们的界能够自适应于平均条件方差，并恢复了 Li 等人（2026）的方差自适应速率。我们进一步刻画了恰当学习中最优的普适逼近因子，证明符号投注方法可以达到因子 3，而没有任何恰当学习器能够在无分布假设、高概率的 $L_1$ 预言机不等式下满足更小的普适逼近因子（原文摘要在此处截断）。

    arXiv:2610.20903v1 Announce Type: new  Abstract: We study bounded regression under model misspecification, evaluating a function $f$ by its $L_1(P_X)$ distance to the conditional mean $\eta(x)=\mathbb{E}[Y\mid X=x]$, rather than the standard $L_2(P_X)$. This $L_1$ metric is natural for downstream decision problems such as bandits and reinforcement learning. We analyze the min--max estimator of Baraud & Maillard (2025), which we call \emph{sign-betting regression}. Our main results show that sign-betting regression achieves high-probability $L_1$ oracle inequalities under misspecification. Under realizability, our bound adapts to the average conditional variance and recovers the variance-adaptive rate of Li et al. (2026). We further characterize the optimal universal approximation factor for proper learning by showing that sign betting achieves factor 3, while no proper learner can satisfy a distribution-free, high-probability $L_1$ oracle inequality with a universal approximation facto
    
[^34]: 随机梯度下降：非线性时间序列的视角

    Stochastic Gradient Descent: A Nonlinear Time Series Perspective

    [https://arxiv.org/abs/2610.20894](https://arxiv.org/abs/2610.20894)

    本文在迭代随机函数框架下，通过几何矩收缩（GMC）证明了恒定步长SGD在欧氏距离下的收敛性，克服了固定初始化带来的非平稳性问题，证明了平均SGD的淬火中心极限定理、不变原理和α-稳定收敛，并通过研究关于步长的导数为理解步长的影响提供了新视角。

    

    本文在迭代随机函数的框架下研究了具有恒定步长的随机梯度下降（SGD）的统计行为。与以往建立SGD在概率测度（例如Wasserstein距离）意义下收敛性的研究相比，我们的方法通过证明SGD的几何矩收缩（GMC）性质，直接给出了欧氏距离下的收敛性。这种新的收敛性能够解决由固定初始点导致的SGD非平稳性问题，并能对SGD进行更精细的渐近分析。具体而言，我们证明了平均SGD（ASGD）在任意初始化条件下的淬火中心极限定理、不变原理以及α-稳定收敛性。此外，我们通过研究SGD关于步长的导数，为理解步长对SGD的影响提供了一个新颖的视角。我们证明了导数过程平稳解的存在性，这使得……

    arXiv:2610.20894v1 Announce Type: cross  Abstract: This paper investigates the statistical behavior of Stochastic Gradient Descent (SGD) with constant step size under the framework of iterated random functions. Compared to previous studies establishing the convergence of SGD in probability measure, e.g., Wasserstein distance, our approach directly provides the convergence in Euclidean distance by showing the Geometric Moment Contraction (GMC) of SGD. This new convergence can address the non-stationarity of SGD due to fixed initial points and can provide a more refined asymptotic analysis of SGD. Specifically, we prove quenched central limit theorem, invariance principle and $\alpha$-stable convergence for averaged SGD (ASGD) regardless of initialization. Furthermore, we provide a novel perspective to understand the impact of step sizes in SGD by studying its derivative with respect to the step size. The existence of stationary solutions for the derivative processes is shown, enabling t
    
[^35]: 针对删失结果的灵活双重稳健邻近因果推断

    Flexible Doubly Robust Proximal Causal Inference for Censored Outcomes

    [https://arxiv.org/abs/2610.20893](https://arxiv.org/abs/2610.20893)

    该论文提出了邻近生存桥梁学习（PSB），一个针对删失结果的灵活双重稳健邻近因果推断框架，通过核估计器（PSB-K）和神经估计器（PSB-N）两种灵活桥梁学习器直接处理删失数据和生存曲线目标，突破了现有方法依赖限制性参数化桥梁模型的局限。

    

    在存在未测量混杂因素的情况下，估计右删失生存时间结果的因果效应具有挑战性。邻近因果推断通过代理变量来解决未测量混杂问题，但现有的邻近生存方法主要依赖于限制性较强的参数化桥梁模型。我们提出了邻近生存桥梁学习，这是一个针对删失结果的灵活双重稳健邻近因果推断框架。基于现有的治疗特异性生存曲线的邻近识别方法，该框架给出了结果桥梁（OB）、治疗桥梁（TB）和双重稳健（DR）估计器，并提供了两种灵活的桥梁学习器：PSB-K，一种具有低秩时间分解的核估计器；以及PSB-N，一种通过投影最大矩约束损失训练的神经估计器。与现有的针对标量结果的灵活邻近学习器不同，PSB能够直接处理删失数据和生存曲线目标。在标准结构方程模型基准上，……

    arXiv:2610.20893v1 Announce Type: cross  Abstract: Estimating causal effects on right-censored time-to-event outcomes is challenging under unmeasured confounding. Proximal causal inference addresses unmeasured confounding through proxy variables, but existing proximal survival methods rely largely on restrictive parametric bridge models. We introduce Proximal Survival Bridge learning (PSB), a flexible doubly robust proximal framework for censored outcomes. Building on existing proximal identification of treatment-specific survival curves, the framework yields outcome-bridge (OB), treatment-bridge (TB), and doubly robust (DR) estimators, with two flexible bridge learners: PSB-K, a kernel estimator with a low-rank time decomposition, and PSB-N, a neural estimator trained by a projected maximum-moment-restriction loss. Unlike existing flexible proximal learners for scalar outcomes, PSB directly handles censoring and survival-curve targets. On a standard SEM benchmark, a proximal semi-synt
    
[^36]: 面向多流最快变化检测的专家混合框架

    A Mixture-of-Experts Framework for Multi-Stream Quickest Change Detection

    [https://arxiv.org/abs/2610.20876](https://arxiv.org/abs/2610.20876)

    本文提出了一种用于最快变化检测的专家混合框架，并将其应用于多流变化检测问题，所得到的算法在性能上优于文献中已知的最优算法。

    

    提出了一种用于最快变化检测（QCD）的专家混合框架，其中包含多个专家，每个专家生成一系列用于变化检测的统计量。当这些统计量的加权和超过阈值时，即宣告检测到一次变化。随后，将该框架应用于多流最快变化检测问题，得到了一系列性能强大的算法，其表现优于文献中已知的最优算法。在多流最快变化检测问题中，决策者观察多个观测流，在变化点处，未知子集的流的分布会发生改变。基于广义似然比的最优算法计算代价高昂，而文献中在计算和性能上效率最高的算法之一是基于Shiryaev-Roberts统计量的秩加权和（RWS）方案。论文中证明了基于专家混合的多流QCD算法……

    arXiv:2610.20876v1 Announce Type: cross  Abstract: A Mixture-of-Experts (MoE) framework is proposed for quickest change detection (QCD) where there are multiple experts, and each expert produces a sequence of statistics for change detection. A change is declared when a weighted sum of these statistics crosses a threshold. This framework is then applied to the problem of multi-stream QCD to obtain a family of powerful algorithms that outperform the best known algorithms in the literature. In the multi-stream QCD problem, a decision-maker observes multiple streams of observations. At the change point, the distribution of an unknown subset of the streams changes. The optimal algorithm based on a generalized likelihood ratio is computationally expensive. One of the most computationally and performance-efficient algorithms in the literature is the rank-weighted sum (RWS) scheme based on Shiryaev-Roberts statistics. It is shown in the paper that the MoE-based QCD algorithm for multi-stream d
    
[^37]: 时间划分高斯过程混合模型

    Time-partitioning Gaussian Process Mixture Model

    [https://arxiv.org/abs/2610.20860](https://arxiv.org/abs/2610.20860)

    提出了一种贝叶斯非参数框架TPGPMM，通过自适应Pitman-Yor先验与特定状态高斯过程的结合，自动识别时间序列中的持续状态与变点，无需马尔可夫假设即可建模非平稳依赖，并能有效区分持续变化与瞬态冲击。

    

    我们提出了时间划分高斯过程混合模型（TPGPMM），这是一个用于识别时间序列中持续状态并建模非平稳依赖关系的贝叶斯非参数框架。该模型将连续分段上的自适应Pitman-Yor先验与特定状态的高斯过程相结合，而不施加状态内的马尔可夫依赖。通过学习Pitman-Yor折扣参数，模型能够适应同质和异质的状态长度。一种带有精确Gibbs边界更新的分裂-合并Metropolis-within-Gibbs采样器可联合学习变点和模型超参数，量化分段及状态内动态的不确定性。使用RBF核和固定的保守观测噪声水平，TPGPMM倾向于稀疏分段，能够区分持续变化与瞬态冲击。仿真实验证明了该模型对状态结构的恢复能力。该方法已应用于CBOE波动率…

    arXiv:2610.20860v1 Announce Type: cross  Abstract: We propose the Time-partitioning Gaussian Process Mixture Model (TPGPMM), a Bayesian nonparametric framework for identifying persistent regimes and modelling nonstationary dependence in time series. The model combines an adaptive Pitman--Yor prior on contiguous segmentations with regime-specific Gaussian processes, without imposing Markovian within-regime dependence. Learning the Pitman--Yor discount parameter allows adaptation to homogeneous and heterogeneous regime lengths.   A split--merge Metropolis-within-Gibbs sampler with exact Gibbs boundary updates jointly learns change points and model hyperparameters, quantifying uncertainty over segmentations and within-regime dynamics. With an RBF kernel and a fixed, conservative observation-noise level, TPGPMM favours sparse segmentations that distinguish persistent shifts from transient shocks. Simulations demonstrate recovery of the regime structure. Applications to the CBOE Volatility 
    
[^38]: 完整还是稀疏：两种可辨识性的故事

    Complete or Sparse: A Tale of Two Identifiabilities

    [https://arxiv.org/abs/2610.20851](https://arxiv.org/abs/2610.20851)

    本文证明在生成式与判别式两种因果方向下，表示维度充足时最优编码器是潜在变量的线性函数，而结合非负激活与能量效率约束可使每个神经元恰好编码一个潜在变量，为“祖母神经元”现象提供了精确的理论解释。

    

    理解神经网络何时学习到可解释的表示，是机制可解释性研究的核心问题。我们通过可辨识性来研究这一问题：监督学习何时能够恢复数据背后真实的潜在变量？我们引入了两种以因果方向相区分的数据生成过程：一种是由标签导致潜在变量（生成式），另一种是由潜在变量导致标签（判别式）。当表示的维度不少于潜在空间的维度时，我们证明最优编码器是潜在变量的线性函数；相反，当维度不足时，可辨识性通常会失效，编码器变为非线性。随后我们进一步证明，额外的约束能够恢复单个潜在变量，而不仅仅是它们的线性混合。在生成式设定中，非负激活与能量效率迫使每个神经元恰好编码一个潜在变量，从而为“选择性祖母神经元”现象提供了精确的理论解释。

    arXiv:2610.20851v1 Announce Type: cross  Abstract: Understanding when neural networks learn interpretable representations is central to mechanistic interpretability. We study this through identifiability: when can supervised learning recover the true latent variables underlying the data? We introduce two data generating processes distinguished by their causal direction where either labels cause latents (generative) or latents cause labels (discriminative). When the representation has at least as many dimensions as the latent space, we prove that the optimal encoders are linear in the latents. Conversely, when dimensions are insufficient, identifiability generically fails and encoders become nonlinear. We then show that additional constraints yield recovery of individual latent variables, not just linear mixtures. In the generative setting, non-negative activations and energy efficiency force each neuron to encode exactly one latent, providing a precise account of selective grandmother 
    
[^39]: 选择安全的保形早退：自适应深度下的无分布覆盖保证

    Selection-Safe Conformal Early Exit: Distribution-Free Coverage under Adaptive Depth

    [https://arxiv.org/abs/2610.20773](https://arxiv.org/abs/2610.20773)

    该论文提出一种选择安全的保形早退方法，通过前缀最大值构造的终端路径分数和单一保形分位数校准，在任意数据依赖的提前退出策略下均能保证无分布覆盖率，无需 alpha/L 修正，且实验表明其覆盖率显著优于朴素的逐层选择方法。

    

    在每个早退点分别校准保形预测集，一旦选择了依赖于数据的退出点，便无法保持覆盖率。我们从各层前缀最大值构造一个终端路径分数，校准单一的分隔保形分位数，并返回全部包含有效终端集的嵌套前缀集。该保证无需 alpha/L 修正即具有选择安全性，并且同时适用于所有在校准之后仍保持终端集包含关系的无标签停止策略。我们还通过早退分数超出终端分数部分的经验尾部，界定了路径最大值带来的效率损失。在八个编码器任务-主干组合的实验中，路径级选择方法达到 0.901 的宏覆盖率，而朴素的逐层选择仅为 0.814，同时平均集合大小为 1.41，延迟比例为 0.71。在 CIFAR-100 上使用具有 24 个退出点的 ViT-L/16（n_cal=500）时，预算 b=4 下的首次满足条件停止策略达到覆盖率 0.902、集合大小 2.76 和延迟比例 0.57，且同样的（摘要在此处被截断）

    arXiv:2610.20773v1 Announce Type: new  Abstract: Calibrating a conformal prediction set at each early exit does not preserve coverage once a data-dependent exit is selected. We construct one terminal path score from prefix maxima, calibrate a single split-conformal quantile, and return nested prefix sets that all contain the valid terminal set. The guarantee is selection-safe without an alpha/L correction and holds simultaneously for every post-calibration label-free stopping policy that preserves terminal-set containment. We also bound the efficiency cost of the path maximum by the empirical tail of the early-over-terminal score excess. Across eight encoder task-backbone cells, pathwise selection attains macro coverage 0.901 versus 0.814 for naive layerwise selection, with average set size 1.41 and latency fraction 0.71. On CIFAR-100 with a 24-exit ViT-L/16 (n_cal=500), first-eligible stopping at budget b=4 attains coverage 0.902, set size 2.76, and latency fraction 0.57, and the same
    
[^40]: RoLaQ：面向多头潜在注意力中2比特KV缓存的旋转与量化范围联合校准

    RoLaQ: Joint Rotation and Range Calibration for 2-Bit KV Caches in Multi-Head Latent Attention

    [https://arxiv.org/abs/2610.20686](https://arxiv.org/abs/2610.20686)

    提出RoLaQ离线校准方法，通过在多头潜在注意力模型中联合学习共享旋转和相互耦合的量化范围，实现2比特KV缓存量化，在多个模型和基准测试上取得了2比特方法中最高的平均准确率。

    

    多头潜在注意力（MLA）将键值缓存的内容压缩为每个token共享的潜在表示，使得低比特存储对长上下文推理极具吸引力。在2比特量化下，该表示中的误差会同时干扰注意力匹配和数值读取，其影响取决于误差方向和重建尺度。我们提出了RoLaQ（旋转校准的潜在量化），这是一种用于联合旋转与量化范围校准的离线方法。RoLaQ在由冻结的键投影和值投影所定义的子空间内细化共享旋转，并联合学习量化范围，其阈值与重建增益相互耦合，以控制共享信号的响应。在模型权重冻结的条件下，通过预测匹配和注意力输出监督来引导校准过程。在三个MLA模型和六个基准测试上，RoLaQ无论是否使用token保护，均在所比较的2比特方法中取得最高的平均准确率。

    arXiv:2610.20686v1 Announce Type: new  Abstract: Multi-head latent attention (MLA) compresses the content of a key-value cache into a shared latent per token, making low-bit storage attractive for long-context inference. At two bits, errors in this representation perturb both attention matching and value readout, with effects that depend on error direction and reconstruction scale. We introduce RoLaQ (Rotation-calibrated Latent Quantization), an offline method for joint rotation and range calibration. RoLaQ refines a shared rotation within subspaces defined by the frozen key and value projections, and jointly learns quantization ranges whose thresholds and reconstruction gains are coupled to control the shared signal's response. Prediction matching and attention-output supervision guide calibration with model weights frozen. Across three MLA models and six benchmarks, RoLaQ achieves the highest average accuracy among the compared 2-bit methods both without and with token protection. On
    
[^41]: 面向力学相关TPMS超材料的拓扑感知表示学习

    Learning Topology-Aware Representations for Mechanics-Relevant TPMS Metamaterials

    [https://arxiv.org/abs/2610.20684](https://arxiv.org/abs/2610.20684)

    提出将持续同调融入DMAE骨干网络的拓扑感知表示学习框架，通过持续性引导的掩码策略和辅助持续同调解码器显式编码TPMS超材料的全局结构连通性，从而更好地支持力学分析与结构设计。

    

    学习力学三重周期极小曲面（TPMS）超材料的信息性表示对于力学分析和结构设计十分重要，然而现有方法很少利用与力学响应相关的全局结构连通性信息来显式监督所学习到的表示。我们提出了一种拓扑感知的表示学习框架，将持续同调融入DMAE骨干网络，以显式编码全局结构连通性。一种由持续性引导的掩码策略优先关注与持续拓扑特征相关的图像块，同时一个辅助的持续同调解码器促使潜在表示保留关于连通分量、隧道和空洞等拓扑特征数量的信息。我们在两个互补的TPMS数据集上，通过家族分类、力学性能分类、无监督等任务评估了所学习到的表示。

    arXiv:2610.20684v1 Announce Type: new  Abstract: Learning informative representations of mechanical Triply Periodic Minimal Surface (TPMS) metamaterials is important for mechanics analysis and structural design, yet existing methods have rarely explicitly supervised learned representations with information about global structural connectivity that is relevant to mechanical response. We propose a topology-aware representation learning framework that integrates persistent homology into a DMAE backbone to explicitly encode global structural connectivity. A persistence-guided masking strategy prioritizes patches associated with persistent topological features, while an auxiliary persistent-homology decoder encourages the latent representation to retain information about topological feature counts across connected components, tunnels, and voids. We evaluate the learned representations on two complementary TPMS datasets using family classification, mechanical-property classification, unsuper
    
[^42]: 稀疏自编码器中数据的收益递减：功能行为并不能保证特征可复现性

    Diminishing Returns of Data in Sparse Autoencoders: Functional Behavior does not Guarantee Feature Reproducibility

    [https://arxiv.org/abs/2610.20680](https://arxiv.org/abs/2610.20680)

    该论文发现稀疏自编码器的功能表现（如重构）在远小于常规规模的数据量下即可接近全量训练水平，但特征的一对一可复现性会随数据量增加持续提升，表明功能行为良好并不能保证特征可复现，数据规模和数据选择方式对特征复现至关重要。

    

    稀疏自编码器（SAE）通常在数亿到数十亿个激活token上进行训练，但很少有工作检验这种规模是否必要。我们通过在三个语言模型上使用从500万token到5亿token参考预算的不同数据量训练四种SAE架构，研究了SAE到底需要多少激活数据。我们衡量了两种收敛概念：标准评测上的功能行为，以及所学特征的一对一可复现性。功能层面的数据需求强烈依赖于所用的指标：重构性能往往在明显更小的数据预算下就接近全量数据的参考水平，而特征可复现性则随数据量增加持续改善。对于Gemma-2-2B BatchTopK，严格匹配特征的比例从500万token时的2%上升到5000万token时的29%。数据选择同样重要：随机采样能更好地保留原始激活分布，通常更有利于重构，而基于多样性的……（摘要原文在此处被截断）

    arXiv:2610.20680v1 Announce Type: new  Abstract: Sparse autoencoders (SAEs) are typically trained on hundreds of millions to billions of activation tokens, yet little work has tested whether this scale is necessary. We study how much activation data SAEs need by training four SAE architectures on three language models with budgets ranging from 5M tokens to a 500M-token reference. We measure two notions of convergence: functional behavior on standard evaluations and one-to-one reproducibility of learned features. Functional data requirements are strongly metric-dependent: reconstruction often approaches the full-data reference at substantially smaller budgets, whereas feature reproducibility continues to improve with data. For Gemma-2-2B BatchTopK, the fraction of strictly matched features rises from 2\% at 5M tokens to 29\% at 50M. Data selection also matters: random sampling better preserves the original activation distribution and generally favors reconstruction, whereas diversity-ba
    
[^43]: 二次网络中的谱不可区分性与学习曲线可辨识性

    Spectral Indistinguishability and Learning-Curve Identifiability in Quadratic Networks

    [https://arxiv.org/abs/2610.20678](https://arxiv.org/abs/2610.20678)

    该论文构造了谱测度完全一致（谱不可区分）但学习曲线仍可区分的二次网络对，证明谱摘要无法唯一辨识网络，而标量学习曲线可以实现辨识，且其查询复杂度随循环长度呈指数增长。

    

    我们研究谱摘要和标量学习曲线能揭示中心化二次网络的哪些信息。我们构造了具有精确高斯单坐标和双坐标边缘分布的解析正输入密度。它们的完整四阶矩算子被一个固定整个对角教师代数的正交映射所交织，因此相应的锚定谱测度完全一致。对于具有非零循环不变量的选定常见教师，由于该映射不保持矩阵乘法，其因子化损失曲线则有所不同。循环长度 $r$ 可以任意大，对于每个固定构造，在训练时间内一致地给出 $O(\varepsilon^r)$ 的对比度。对于给定的网络对，带噪声的标量损失查询需要 $\varepsilon^{-2r}$ 量级的观测，而带标量梯度读出的有界初始化干预则需要 $\varepsilon^{-2}$ 量级的观测。一个等于配对梯度差的共享输入分数，能够达到……

    arXiv:2610.20678v1 Announce Type: cross  Abstract: We study what spectral summaries and scalar learning curves reveal about centered quadratic networks. We construct analytic positive input densities with exactly Gaussian one- and two-coordinate marginals. Their complete fourth-moment operators are intertwined by an orthogonal map fixing the entire diagonal teacher algebra, so the corresponding anchored spectral measures agree. For selected common teachers with a nonzero cycle invariant, their factorized loss curves differ because the map does not preserve matrix multiplication. Cycle length $r$, which can be arbitrarily large, gives $O(\varepsilon^r)$ contrast uniformly in training time for each fixed construction. For a supplied pair, noisy scalar loss queries require order $\varepsilon^{-2r}$ observations, while bounded initialization interventions with scalar gradient readouts require order $\varepsilon^{-2}$. A shared-input score, equal to a paired-gradient difference, attains the
    
[^44]: 关于流映射训练的稳定性与收敛性

    On the Stability and Convergence of Flow Map Training

    [https://arxiv.org/abs/2610.20674](https://arxiv.org/abs/2610.20674)

    本文通过分析MeanFlow与拉格朗日损失诱导的函数梯度流，首次从理论上解释了流映射训练中停止梯度算子和自适应加权等技巧的必要性，并证明了训练会收敛到真实流映射。

    

    流映射是近来的一种概率生成方法，它将扩散轨迹摊销为单步采样器，从而实现高效生成。尽管在提升其实证性能方面已取得进展，但对其成功背后的机制仍缺乏理解。具体而言，许多流映射学习方法将停止梯度算子作为其训练目标的关键组成部分。自适应加权也极大地提升了流映射的生成能力。尽管已有许多解释被提出以支持这些启发式技巧的引入，但尚无工作分析这些假设对流映射自身训练动态的影响。在本工作中，我们通过提供有依据的理论来解释为什么需要这些技巧，从而解决了这一问题。通过分析由 MeanFlow 损失和拉格朗日损失所诱导的函数梯度流，我们证明了其收敛到真实流映射……

    arXiv:2610.20674v1 Announce Type: cross  Abstract: Flow maps are a recent method of probabilistic generation that amortize diffusion trajectories into one-step samplers, unlocking efficient generation. While progress has been made in improving their empirical performance, there is still a lack of understanding behind the mechanisms underlying their success. Specifically, many flow map learning methods employ stop gradient operators as a crucial part of their training target. Adaptive weighting also drastically improves the generation capabilities of flow maps. Although many interpretations have been proposed to support the inclusion of these heuristics, none have analyzed the impact of these assumptions on the training dynamics of flow maps themselves. In this work, we address this issue by providing grounded theory explaining why such tricks are required. Analyzing the functional gradient flow induced by the MeanFlow and Lagrangian losses, we show their convergence to the true flow ma
    
[^45]: 解耦特征、拓扑与交互的图加性模型

    Graph Additive Models disentangling feature, topology, and interactions

    [https://arxiv.org/abs/2610.20662](https://arxiv.org/abs/2610.20662)

    提出AMIGA模型，通过挖掘的图模体表示拓扑结构，并将节点贡献分解为特征、结构和交互三个加性通道，从而解决了图神经加性网络中虚假相关性传递和效应纠缠的问题。

    

    图神经加性网络（GNANs）已成为一种有效的、具备内建可解释性的图学习架构。它们通过将逐特征的形状函数与简单的基于距离的重加权相结合来实现透明性。然而，GNANs存在三个局限性。第一，节点级相关性得分聚合了来自图中所有节点的信息，使得无关节点可能从相关节点继承虚假的相关性。第二，拓扑仅通过最短路径距离来表示，因此结构不同的图可能具有完全相同的距离特征。第三，特征与距离信息以乘法方式结合，导致特征、距离和交互效应被纠缠在同一个可缩放项中。我们提出AMIGA，它通过挖掘出的模体词汇表来表示拓扑结构，并将每个节点的贡献分解为三个加性通道：特征、结构以及特征-结构交互。这些通道被中心化……

    arXiv:2610.20662v1 Announce Type: cross  Abstract: Graph Neural Additive Networks (GNANs) have emerged as effective, interpretable-by-design architectures for graph learning. They achieve transparency by combining per-feature shape functions with simple distance-based reweighting. However, GNANs have three limitations. First, node-wise relevance scores aggregate information from all graph nodes, allowing irrelevant nodes to inherit spurious relevance from relevant ones. Second, topology is represented only through shortest-path distances, so structurally distinct graphs may share identical distance profiles. Third, feature and distance information are combined multiplicatively, entangling feature, distance, and interaction effects in a single rescalable term. We introduce AMIGA, which represents topology through a vocabulary of mined motifs and decomposes each node contribution into three additive channels: feature, structure, and feature-structure interaction. The channels are centere
    
[^46]: 基于方差保持桥的稀疏观测随机场潜在退火后验采样

    Latent Annealed Posterior Sampling with a Variance-Preserving Bridge for Sparsely Observed Random Fields

    [https://arxiv.org/abs/2610.20659](https://arxiv.org/abs/2610.20659)

    提出LAPS方法，利用流匹配学习潜在先验并在潜在空间中进行退火后验采样，结合方差保持桥控制协方差膨胀，从而高效解决稀疏观测随机场后验推断中空间不确定性刻画与计算成本增长的难题。

    

    从稀疏观测中对随机场进行后验推断面临两大挑战：如何准确刻画复杂的空间不确定性，以及如何避免条件采样成本随空间网格加密而大幅增长。为解决这些挑战，我们提出了潜在退火后验采样（LAPS）。LAPS首先将随机场映射到降维后的潜在空间，同时利用由Branch网络和Trunk网络组成的坐标解码器，实现对任意空间位置的场值评估。随后采用流匹配学习潜在先验分布。基于该学习到的先验，后验采样完全在潜在空间中进行，观测似然通过退火过程逐步融入采样。为控制局部协方差膨胀，我们引入了一种方差保持（VP）桥，在传输的潜在状态与目标状态之间分配协方差预算。

    arXiv:2610.20659v1 Announce Type: cross  Abstract: Posterior inference for random fields from sparse observations presents two major challenges: accurately characterizing complex spatial uncertainty and preventing conditional sampling costs from growing substantially as the spatial grid is refined. To address these challenges, we propose latent annealed posterior sampling (LAPS). LAPS first maps the random field into a latent space with reduced dimensionality, while a coordinate decoder composed of Branch and Trunk networks enables field evaluation at arbitrary spatial locations. Flow matching is then used to learn the latent prior distribution. Based on this learned prior, posterior sampling is performed entirely in the latent space, with the observation likelihood incorporated progressively through an annealing procedure. To control local covariance inflation, we introduce a variance-preserving (VP) bridge that allocates a covariance budget between the transported latent state and a 
    
[^47]: 列表可复现性的一个多对数界

    A polylogarithmic bound for list replicability

    [https://arxiv.org/abs/2610.20657](https://arxiv.org/abs/2610.20657)

    本文为任意二元概念类的列表可复现性数 O(log²M log N) 这一首个多对数上界提供了一个简短且自包含的新证明。

    

    Zhang [Zha26] 最近发布了一份手稿，宣布对于定义在 N≥2 个定义域点上的任意一类 M≥2 个二元概念，其列表可复现性数（list-replicability number）为 O(log²M log N)，这将是关于该量的首个多对数上界。在这篇简短笔记中，我们遵循该手稿的高层思路，但采用了 considerably 更为简化的论证，给出了这一界的一个简短且自包含的证明。

    arXiv:2610.20657v1 Announce Type: cross  Abstract: Zhang [Zha26] recently posted a manuscript announcing that the list-replicability number of every class of $M\ge 2$ binary concepts on $N\ge 2$ domain points is $O(\log^2 M\log N)$, which would be the first polylogarithmic upper bound on this quantity. In this note, we present a short, self-contained proof of this bound, following the high-level approach of the manuscript but with considerably simpler arguments.
    
[^48]: 线性赌博机的扰动跟随领导者算法：一般凸集上的 $\sqrt{T}$ 遗憾

    Follow the Perturbed Leader for Linear Bandits: $\sqrt{T}$ Regret over General Convex Sets

    [https://arxiv.org/abs/2610.17818](https://arxiv.org/abs/2610.17818)

    本文提出了首个基于FTPL的对抗性线性赌博机算法，通过一个适应估计奖励的扰动分布，在一般凸动作集上实现了 $\sqrt{T}$ 遗憾，改进了此前 $T^{2/3}$ 的遗憾界，同时比连续指数权重方法具有更好的计算效率和更简单的实现。

    

    我们提出了首个基于扰动跟随领导者（Follow-the-Perturbed-Leader, FTPL）的对抗性线性赌博机算法，该算法在一般凸动作集上实现了 $\sqrt{T}$ 遗憾。我们的算法仅使用线性优化预言机即可达到 $\widetilde{\mathcal{O}}(d^{5/4}\sqrt{T})$ 的遗憾；当进一步可获得最优设计探索分布时，遗憾可达到接近最优的 $\widetilde{\mathcal{O}}(d\sqrt{T})$。这改进了 Dani 和 Hayes（2006）的 FTPL 算法在一般凸集上 $T^{2/3}$ 的遗憾，并将 Lévy 等人（2026）在两个特殊动作集（超立方体和欧几里得球）上的 $\sqrt{T}$ 遗憾结果进行了推广。与之前基于连续指数权重的预言机高效线性赌博机算法相比，FTPL 方法通常具有更好的计算效率，且实现起来要简单得多。关键的算法要素是一个能适应估计奖励的扰动分布。

    arXiv:2610.17818v1 Announce Type: cross  Abstract: We give the first adversarial linear bandit algorithm based on Follow-the-Perturbed-Leader (FTPL) that achieves $\sqrt{T}$ regret over general convex action sets. Our algorithm attains $\widetilde{\mathcal{O}}(d^{5/4}\sqrt{T})$ regret using only linear optimization oracles, and the near-optimal $\widetilde{\mathcal{O}}(d\sqrt{T})$ regret when an optimal-design exploration distribution is further available. This improves on the $T^{2/3}$ regret of the FTPL algorithm of Dani and Hayes (2006) for general convex sets, and generalizes the $\sqrt{T}$ regret of L\'evy et al. (2026) over two special action sets (hypercube and Euclidean ball). Compared to previous oracle-efficient linear bandit algorithms based on continuous exponential weights, the FTPL approach generally has better computational efficiency, and is much simpler to implement.   The key algorithmic element is a perturbation distribution that adapts to the estimated rewards. Its 
    
[^49]: 耗散知识动力学模型的可辨识性：设计激励下的精确恢复与观测数据上的退化

    Identifiability of a dissipative knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational data

    [https://arxiv.org/abs/2610.09889](https://arxiv.org/abs/2610.09889)

    该论文将人类学习建模为参数具有机制含义的耗散常微分方程组，证明了在设计激励条件下模型参数可从数据中被精确恢复（双概念情形有闭式解），而在仅依赖观测数据时可辨识性会退化，并提出了数值精确等价且速度大幅提升的半隐式L-稳定批量求解器。

    

    人类学习是一个耗散动力学过程：熟练度通过练习积累，因遗忘而衰减，并在相互依赖的概念之间传播。我们将其建模为一个非线性耗散常微分方程组，其参数具有机制层面的含义（编码先修耦合关系的概念传递矩阵、各概念的遗忘率，以及饱和的练习-响应增益），并研究这些参数何时能够真正从数据中被恢复。我们在显式激励条件下证明了相应逆问题的结构可辨识性定理，针对双概念情形给出了构造性的闭式恢复方法，并同时给出了单调性、鲁棒性与L-稳定性结果。我们为耗散子系统推导了一种半隐式L-稳定数值格式，以及一个与逐轨迹公式数值等价的批量求解器（预测逐位一致，梯度误差达 $10^{-10}$），同时速度提升两个数量级。

    arXiv:2610.09889v1 Announce Type: new  Abstract: Human learning is a dissipative dynamical process: mastery accumulates through practice, decays through forgetting, and propagates across interdependent concepts. We model it as a nonlinear dissipative system of ordinary differential equations whose parameters are mechanistically meaningful (a concept-transfer matrix encoding prerequisite coupling, per-concept forgetting rates, and a saturating practice-response gain), and we study when those parameters can actually be recovered from data. We prove a structural identifiability theorem for the associated inverse problem under explicit excitation conditions, with constructive closed-form recovery for the two-concept case, together with monotonicity, robustness and L-stability results. We derive a semi-implicit L-stable scheme for the dissipative subsystem and a batched solver numerically equivalent to the per-trajectory formulation (bit-exact predictions, gradients to $10^{-10}$) yet two o
    
[^50]: LASER：用于支撑约束熵正则化离线强化学习的潜在空间伴随匹配

    LASER: Latent Space Adjoint Matching for Support-Constrained Entropy-Regularized Offline RL

    [https://arxiv.org/abs/2610.08989](https://arxiv.org/abs/2610.08989)

    LASER通过潜在空间伴随匹配实现熵正则化的潜在空间离线强化学习，既防止了策略坍缩成单一脆弱模式，又避免了时间反向传播，在40个不同数据质量的OGBench任务上取得了优异表现。

    

    虽然离线强化学习（RL）能够在无需昂贵的在线交互的情况下从静态数据集中进行策略优化，但其性能仍然受到执行分布外（OOD）动作风险的制约。近期的方法通过流匹配学习一个行为克隆策略，然后在其受约束的潜在空间内执行强化学习，从而缓解了这一问题。然而，朴素地优化潜在策略很容易导致策略坍缩成脆弱的单一模式，或利用学习到的评论家网络中尖锐的伪影。在这项工作中，我们发现熵正则化对于在潜在空间强化学习中应对这些挑战至关重要。我们提出了LASER，一种新颖的离线强化学习算法，它应用潜在空间伴随匹配来实现基于强表达能力的流策略的熵正则化潜在空间强化学习，同时避免了时间反向传播。通过在40个具有不同数据集质量的具有挑战性的OGBench任务上的全面实验，我们展示了（摘要在此处截断）

    arXiv:2610.08989v1 Announce Type: new  Abstract: While offline reinforcement learning (RL) enables policy optimization from static datasets without costly online interaction, it remains bottlenecked by the risk of executing out-of-distribution (OOD) actions. Recent approaches mitigate this by learning a behavior-cloning policy through flow matching and then performing RL within its constrained latent space. However, naively optimizing the latent policy can easily cause the policy to collapse into a brittle mode or exploit sharp artifacts of the learned critic. In this work, we find that entropy regularization is essential in latent-space RL for addressing these challenges. We introduce LASER, a novel offline RL algorithm that applies latent-space adjoint matching to achieve entropy-regularized latent-space RL with expressive flow policies while avoiding backpropagation through time. Through comprehensive experiments on 40 challenging OGBench tasks with varying dataset qualities, we sho
    
[^51]: 滑动窗口占据计数谱隙的最优状态空间阶

    Optimal State-Space Order for Spectral Gaps of Sliding-Window Occupation Counts

    [https://arxiv.org/abs/2609.27836](https://arxiv.org/abs/2609.27836)

    本文证明了滑动窗口占据计数所定义的投影计数核的谱隙与原马尔可夫核谱隙之比的最优常数 $c_m^\star$ 满足 $\Theta(1/m)$ 的阶，给出了普适下界 $1/(1080m)$ 和形如 $(m+\log m+O(1))^{-1}$ 的上界，但精确值仍待确定。

    

    设 $P$ 是定义在 $m$ 个状态的空间 $\Omega$ 上的不可约可逆马尔可夫核，其右谱隙记为 $\gamma=1-\lambda_2(P)$。基于平稳轨迹，令 $K_t$ 为从时刻 $t$ 开始的长度为 $n$ 的窗口所对应的占据计数向量。平稳对 $(K_0,K_1)$ 定义了一个可逆的投影计数核 $\widetilde P_n$。对每个 $m\ge2$，定义 $c_m^\star=\inf_{P,\;n\ge2}\frac{n\Gap(\widetilde P_n)}{\Gap(P)}$。我们证明 $\frac{1}{1080m}\le c_m^\star\le q_{m-2}$，其中 $q_0=\frac14$，$q_{r+1}=q_r(1-q_r)$。我们还证明 $q_{m-2}=(m+\log m+O(1))^{-1}$，从而推出 $c_m^\star=\Theta(m^{-1})$。因此，最优比较系数的阶为 $m$，但其精确值仍是未决问题。该下界在 $n=1$ 时同样成立，且对 $P$ 一致成立，包括稀疏核与周期核。其证明将短窗口去相关与平均锚定-游走分解相结合……

    arXiv:2609.27836v1 Announce Type: cross  Abstract: Let $P$ be an irreducible reversible Markov kernel on a $m$-state space $\Omega$, and denote its right spectral gap $\gamma=1-\lambda_2(P)$. From a stationary trajectory, let $K_t$ be the occupation-count vector of the length-$n$ window beginning at time $t$. The stationary pair $(K_0,K_1)$ defines a reversible projected count kernel $\widetilde P_n$. For every $m\ge2$, let \[   c_m^\star=   \inf_{\substack{ P,\; n \ge 2}}   \frac{n\Gap(\widetilde P_n)}{\Gap(P)}. \] We prove \[   \frac1{1080m}\le c_m^\star\le q_{m-2},   \qquad   q_0=\frac14,\quad q_{r+1}=q_r(1-q_r). \] We also show $q_{m-2}=(m+\log m+O(1))^{-1}$, which implies that $c_m^\star=\Theta(m^{-1})$. Thus, the optimal comparison coefficient has order $m$, although its exact value remains open. The lower bound also holds for $n=1$ and is uniform in $P$, including sparse and periodic kernels. Its proof combines short-window decorrelation with an averaged anchor-excursion decompo
    
[^52]: ICON分解：用于模型审计的深度表示多变量概念级解释

    ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing

    [https://arxiv.org/abs/2608.26083](https://arxiv.org/abs/2608.26083)

    ICON分解通过多变量分析，在控制其他概念和结果后精确量化每个概念对模型表示的独特贡献，从而有效识别捷径学习并提高解释的准确性。

    

    arXiv:2608.26083v1 公告类型：新 摘要：深度神经网络经常利用训练数据中的虚假关联，这种失败被称为捷径学习。基于概念的可解释性方法通过测试诸如患者性别或扫描仪设置等概念是否能从网络层中解码来筛选捷径。由于每个概念是单独评估的，这些方法可能会将概念之间的相关性误认为是模型使用它们的证据。我们引入了ICON分解，它转而量化每个概念在考虑所有其他概念和结果后所解释的层方差的比例。在具有已知真实标签的合成数据上，ICON比七种替代基线方法更准确地恢复了概念重要性。在皮肤病变和脑成像模型中，它隔离了模型真正依赖的概念，量化了任何提供的概念未解释的表示部分，并产生了我们验证过的稀疏解释。

    arXiv:2608.26083v1 Announce Type: new  Abstract: Deep neural networks often exploit spurious associations in their training data, a failure known as shortcut learning. Concept-based explainability methods screen for shortcuts by testing whether concepts such as a patient's sex or scanner settings can be decoded from a network layer. Because each concept is evaluated in isolation, these methods can mistake correlations between concepts as evidence that the model uses them. We introduce ICON decomposition, which instead quantifies how much of a layer's variance each concept explains after accounting for all other concepts and the outcome. On synthetic data with known ground truth, ICON recovers concept importance more accurately than seven alternative baseline methods. On skin-lesion and brain-imaging models, it isolates the concepts on which a model genuinely relies, quantifies the representation unexplained by any of the supplied concepts, and yields sparse explanations that we validat
    
[^53]: 混合来源大型语言模型文本中的最优水印定位

    Optimal Watermark Localization in Mixed-Source Large Language Model Texts

    [https://arxiv.org/abs/2608.14906](https://arxiv.org/abs/2608.14906)

    本文提出了混合来源LLM文本中水印定位的渐近最优框架，明确了全局检测、发现和分类的相变边界，并证明发现任务难度高于分类。

    

    水印提供了一种有原则的方式来认证由大型语言模型（LLMs）生成的文本。然而，在实践中，最终文本可能是混合来源的，经过改写、插入、删除或释义后，水印证据仅存留在部分标记位置。尽管先前的研究已经探讨了水印信号的全局检测，但何时能对这些信号进行定位仍不清楚。我们将水印定位问题表述为基于关键统计量的标记级多重检验问题，其中包含一个潜在指示符，记录每个位置的水印依赖是否存活。在由信号稀疏性、下一标记浓度和有效词汇增长指数所索引的渐近框架下，我们推导出全局检测的尖锐边界，以及在坐标级基于关键统计的定位规则类别内的发现和分类相变。我们表明，发现严格比分类更难，并提供了最优规则。

    arXiv:2608.14906v1 Announce Type: cross  Abstract: Watermarking provides a principled way to authenticate text generated by large language models (LLMs). In practice, however, the final text may be mixed-source, with watermark evidence surviving at only a subset of token positions after rewriting, insertion, deletion, or paraphrasing. Although prior work has studied global detection of watermark signals, when such signals can be localized remains unclear. We formulate watermark localization as a token-level multiple-testing problem based on pivotal statistics, with a latent indicator recording whether watermark dependence survives at each position. Under an asymptotic regime indexed by exponents for signal sparsity, next-token concentration, and effective-vocabulary growth, we derive a sharp boundary for global detection and phase transitions for discovery and classification within the class of coordinatewise pivot-based localization rules. We show that discovery is strictly harder tha
    
[^54]: 局部正则性下基于一阶增广拉格朗日方法的非凸复合泛函约束优化

    Nonconvex Composite Functional Constraints via First-Order Augmented Lagrangian Methods under Local Regularity

    [https://arxiv.org/abs/2607.08954](https://arxiv.org/abs/2607.08954)

    该论文在仅需可行集附近局部乘子正则性的弱条件下，无需假设乘子有界，通过带对偶正则化的光滑近端线性增广拉格朗日原始-对偶方法，为非凸复合约束优化问题建立了非渐近KKT收敛保证，达到 $O(K^{-1/3})$ 的KKT误差率。

    

    我们研究了一类非凸约束优化问题在局部乘子正则性条件下的原始-对偶方法的非渐近收敛性。目标函数和泛函不等式约束均为凸Lipschitz外函数与光滑非线性映射的复合。与现有分析要求正则性全局成立或沿整个迭代序列成立不同，我们仅在可行集附近施加乘子正则性。利用具有紧对偶集的辅助极小极大问题，我们推导出Lyapunov估计，在可行初始化和足够大的罚参数下识别出近似可行的迭代点，而无需假设乘子有界。随后，局部正则性给出了相应乘子的界并使对偶截断失效，从而为原问题提供近似的KKT证书。带对偶正则化的光滑近端线性增广拉格朗日方法获得了 $O(K^{-1/3})$ 的KKT

    arXiv:2607.08954v2 Announce Type: replace-cross  Abstract: We study nonasymptotic convergence of primal-dual methods for a class of nonconvex constrained optimization problems under local multiplier regularity. Both the objective and functional inequality constraints are compositions of convex Lipschitz outer functions with smooth nonlinear mappings. Unlike existing analyses requiring regularity globally or along the entire iterate sequence, we impose multiplier regularity only near the feasible set. Using an auxiliary minimax problem with a compact dual set, we derive Lyapunov estimates that identify nearly feasible iterates under feasible initialization and a sufficiently large penalty, without assuming multiplier boundedness. Local regularity then bounds the associated multipliers and makes dual truncation inactive, yielding approximate KKT certificates for the original problem. The smoothed prox-linear augmented Lagrangian method with dual regularization yields an $O(K^{-1/3})$ KKT
    
[^55]: ManifoldFlow：具有可学习奇异谱的SPD松弛Stiefel层

    ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum

    [https://arxiv.org/abs/2607.04535](https://arxiv.org/abs/2607.04535)

    ManifoldFlow通过对固定谱Stiefel层进行最小松弛，将权重分解为 W = Q S^{1/2}，在保持基向量位于Stiefel流形上的同时学习有界的正定奇异谱，使特征值裁剪成为直接的奇异值控制机制，并在序列、表格和图像任务中优于固定谱Stiefel层。

    

    正交层和Stiefel层为神经网络权重提供了精确的谱控制，但它们也带来了很强的建模约束：所有被表示的奇异值都被固定为1。许多受益于正交归一基的场景仍然需要方向相关的衰减或放大。我们提出了ManifoldFlow，这是对固定谱Stiefel层的一种最小松弛方法，它在保持基向量位于Stiefel流形上的同时，通过 W = Q S^{1/2}（其中 Q^T Q = I 且 S 为正定矩阵）学习一个有界的正谱。由于 W^T W = S，S 的特征值恰好是所实现权重的奇异值的平方，这使得特征值裁剪成为一种直接的奇异值控制机制。在配对的序列、表格和图像实验中，可学习的SPD谱在Stiefel先验有效的报告设置中优于固定谱的Stiefel对应方法，其中在循环语言模型投影任务中增益最大（原文摘要在此处截断）。

    arXiv:2607.04535v2 Announce Type: replace-cross  Abstract: Orthogonal and Stiefel layers give neural weights exact spectral control, but they also impose a strong modeling constraint: all represented singular values are fixed at one. Many settings that benefit from an orthonormal basis still need direction-dependent attenuation or amplification. We introduce ManifoldFlow, a minimal relaxation of a fixed-spectrum Stiefel layer that keeps the basis on the Stiefel manifold while learning a bounded positive spectrum through W = Q S^{1/2}, with Q^T Q = I and S positive definite. Since W^T W = S, the eigenvalues of S are exactly the squared singular values of the realized weight, making eigenvalue clipping a direct singular-value control mechanism. Across paired sequence, tabular, and image experiments, the learnable SPD spectrum improves the fixed-spectrum Stiefel counterpart in the reported settings where the Stiefel prior is useful, with the largest gains in recurrent language-model proje
    
[^56]: 随机线性上下文赌博机的集合成员估计

    Set Membership Estimation for Stochastic Linear Contextual Bandits

    [https://arxiv.org/abs/2606.20022](https://arxiv.org/abs/2606.20022)

    本文首次将控制与系统辨识领域成熟的集合成员估计（SME）方法引入线性上下文赌博机，提出基于SME的OFU算法并建立了次线性遗憾界。

    

    上下文赌博机学习由于能够在时变环境中融合辅助/上下文信息，在工程应用中受到越来越多的关注。上下文赌博机中一种广泛使用的方法是不确定性下的乐观主义（OFU），该方法为未知参数构建不确定性集合，并在这些集合上进行乐观决策。在赌博机文献中，不确定性集合通常借助统计学习工具（如置信区间）来推导。相比之下，集合成员估计（SME）——一种在控制与系统辨识领域已经成熟的量化不确定性的方法——在赌博机文献中却很少受到关注。本文以线性上下文赌博机为切入点，研究SME与上下文赌博机之间的联系。我们提出了一种基于SME的OFU算法，并在适当的条件下建立了次线性遗憾界。

    arXiv:2606.20022v2 Announce Type: replace-cross  Abstract: Contextual bandit learning has attracted increasing attention in engineering applications due to its ability to incorporate side/context information in time-varying environments. A widely used approach for contextual bandits is optimism in the face of uncertainty (OFU), in which uncertainty sets for unknown parameters are constructed and decisions are made optimistically over these sets. In the bandit literature, uncertainty sets are typically derived using statistical learning tools, such as confidence intervals. In contrast, set membership estimation (SME), a well-established uncertainty quantification approach in control and system identification, has received little attention in the bandit literature. This paper investigates the connection between SME and contextual bandits by focusing on linear contextual bandits as a starting point. We propose an SME-based OFU algorithm and establish a sublinear regret bound under proper 
    
[^57]: 通过导数约束实现最优分数函数估计

    Optimal score function estimation via derivatives constraints

    [https://arxiv.org/abs/2606.19084](https://arxiv.org/abs/2606.19084)

    该论文建立了统计理论，证明当数据分布支撑在低维流形上时，在导数约束下的经验分数匹配估计可达到极小化极大最优的收敛速率，且该速率取决于流形的内在维度而非环境维度，从而为扩散生成模型的泛化能力而非过拟合提供了理论保证。

    

    我们提供了一个统计理论，解释了当数据分布支撑在低维流形上时，经验分数匹配为何能在扩散生成模型中达到极小化极大最优性能。更准确地说，我们在适当约束的Sobolev函数类上建立了经验风险最小化的收敛速率，并证明在适当的正则性条件下，所得的分数估计器能够使生成分布达到极小化极大最优的Wasserstein-1速率（最多相差对数因子）。我们的结果为基于分数的生成建模能够从数据中泛化而非过拟合提供了充分条件，同时所得的统计速率取决于数据流形的内在维度而非其环境维度。

    arXiv:2606.19084v3 Announce Type: replace-cross  Abstract: We provide a statistical theory explaining why empirical score matching can achieve minimax optimal performance for diffusion generative models when the data distribution is supported on a low-dimensional manifold. More precisely, we establish convergence rates for empirical risk minimization over suitably constrained Sobolev classes and show that, under appropriate regularity conditions, the resulting score estimators lead to minimax optimal Wasserstein-1 rates for the generated distribution, up to logarithmic factors. Our results provide sufficient conditions for score-based generative modeling to generalize from the data instead of overfitting, while the resulting statistical rates depend on the intrinsic dimension of the data manifold rather than its ambient dimension.
    
[^58]: 基于玻尔兹曼边际的kNN分类近指数收敛速率

    Near-Exponential Convergence Rates for kNN Classification based on Boltzmann Margin

    [https://arxiv.org/abs/2606.10361](https://arxiv.org/abs/2606.10361)

    本文提出了一种介于Tsybakov边际与Massart边际之间的新条件——玻尔兹曼边际，并据此首次为kNN分类建立了近指数收敛速率。

    

    分类器的收敛速率分析通常在Tsybakov边际或Massart边际条件下进行。前者是一个相对较弱的条件，通常只能得到多项式速率，而后者则强得多，但可以保证指数速率。在本文中，我们引入了一个称为玻尔兹曼边际的新条件，它弥合了这两种情形之间的差距。该条件弱于Massart边际，通常强于Tsybakov边际，并且在适当条件下可以蕴含二者的许多性质。我们将玻尔兹曼边际应用于kNN分类器的分析，首次建立了kNN分类的近指数收敛速率。我们还给出了主要结果的若干扩展，并提供了支持主要理论结论的数值证据。

    arXiv:2606.10361v2 Announce Type: replace-cross  Abstract: Convergence-rate analysis for classifiers is often conducted under either Tsybakov margin or Massart margin. The former is a relatively weak condition that typically yields polynomial rates, while the latter is substantially stronger but can guarantee exponential rates. In this paper, we introduce a new condition, called Boltzmann margin, that bridges the gap between these two regimes. It is weaker than Massart margin, generally stronger than Tsybakov margin, and can imply many of their properties under suitable conditions. We apply Boltzmann margin to the analysis of kNN classifiers and establish the first near-exponential convergence rates for kNN classification. We also present extensions of the main results and provide numerical evidence supporting the main theoretical implications.
    
[^59]: 无干扰的可检测性：面向单次运行隐私审计的高效金丝雀样本设计

    Detectability without Interference: Efficient Canary Design for Privacy Auditing in One Run

    [https://arxiv.org/abs/2605.27292](https://arxiv.org/abs/2605.27292)

    本文提出IBIS框架，通过基于影响的贪心初始化与双层优化设计相互干扰最小、检测分数最优的金丝雀样本，使隐私审计仅需一次训练运行即可高效完成。

    

    隐私审计旨在利用成员推理攻击实证地评估机器学习模型中的隐私泄露程度，并据此推导差分隐私（DP）参数的下界。标准的隐私审计方法会将一个敏感样本（即金丝雀样本）插入到许多次（远超100次）随机化的训练运行中，并要求审计者检测该样本是否被包含在每次运行中。单次运行审计则在一次训练运行中插入大量金丝雀样本，从而避免了多次运行方法的高昂计算成本。然而，近期的理论研究表明，金丝雀样本之间可能相互干扰，导致单个金丝雀样本的成员推理攻击分数降低，进而削弱审计能力。我们提出了IBIS框架，该框架通过优化金丝雀样本来最大化其自身的分数，同时最小化其对其他样本的影响。IBIS将基于影响的贪心初始化与双层优化相结合，并引入了一种新颖的正则化项，鼓励金丝雀样本在嵌入空间中相互正交……（摘要原文在此处截断）

    arXiv:2605.27292v3 Announce Type: replace  Abstract: Privacy auditing aims to empirically assess privacy leakage in machine learning models using membership inference attacks (MIAs), and to derive lower bounds on differential privacy (DP) parameters. Standard privacy auditing inserts a sensitive sample (a canary) into many ($\gg 100$) randomized training runs, requiring the auditor to detect whether it was included in each run. In one-run auditing, many canaries are included in a single training run, avoiding the high computational cost of multi-run approaches. However, recent theory shows that canaries can interfere with one another, reducing per-canary MIA scores and thereby weakening the audit power. We propose IBIS, a framework that optimizes canaries to maximize their own score while minimizing their impact on others. IBIS combines influence-based greedy initialization with bilevel optimization, introducing a novel regularization term that encourages canaries to be orthogonal in e
    
[^60]: 高维处理效应的监督学习目标

    Supervised Learning Objectives for High-Dimensional Treatment Effects

    [https://arxiv.org/abs/2605.27281](https://arxiv.org/abs/2605.27281)

    该论文将因果推断重新表述为监督学习问题，证明了因果误差可分解为递增阶数的矩平衡误差，并据此设计出能直接改善高维处理效应（如文本干预）因果估计的学习目标，同时可将效应投影到更粗粒度的属性上。

    

    预测具有多种可能变化形式的干预效应（例如，影响心理健康结果的治疗内容，或驱动股价波动的财报电话会议记录文本）在多个领域都非常有用。然而，经典的因果估计方法通常假设所有可能的干预都被观测到，这在干预变化范围极广时是不可行的，例如在所有文本字符串构成的空间中。我们对一种广为人知的方法进行了改进，即将因果推断重新表述为学习问题，以应对此类场景，并假设所学习的函数可实现且不存在未观测的混杂因素。我们证明了因果误差可以分解为阶数递增的矩平衡误差，并利用条件结果在混杂因素中的结构来设计能够直接改善因果估计的学习目标。我们还展示了如何将高维处理的效应投影到更粗粒度的属性上，从而……

    arXiv:2605.27281v2 Announce Type: replace  Abstract: Predicting the effect of interventions with many possible variations, e.g., therapeutic content that affects mental health outcomes or an earnings call transcript that drives movement in share price, is useful across several domains. However, classical causal estimators tend to assume that all possible interventions are observed, which is infeasible when interventions vary widely, for instance, in the space of all text strings. We adapt a well-known approach of recasting causal inference as a learning problem to address such settings, assuming realizability of the learned function and no unobserved confounding. We show that causal error decomposes into moment-balancing errors of increasing order and exploit structure of the conditional outcome in the confounders to design objectives that directly improve causal estimation. We also show how to project the effect of a high-dimensional treatment onto coarser attributes, which allows a s
    
[^61]: 先规划后交易：强化学习交易智能体的推理时优化

    Plan Before You Trade: Inference-Time Optimization for RL Trading Agents

    [https://arxiv.org/abs/2605.12653](https://arxiv.org/abs/2605.12653)

    提出了受模型预测控制启发的FinPILOT框架，利用多步价格预测在推理时对任意预训练的强化学习交易策略进行优化，实现“先规划、后交易”的策略自适应调整。

    

    用于投资组合管理的强化学习智能体通常以静态策略的形式进行训练和部署，缺乏在推理时利用价格预测对策略进行调整的机制。我们提出了FinPILOT（Financial Plugin Inference-time Learning for Optimal Trading，金融插件推理时学习最优交易），这是一个受模型预测控制（MPC）启发的插件式推理时策略优化框架。在价格接受者假设下，投资组合动作不会影响未来市场价格，这使得多步价格预测器能够提供想象的市场轨迹，而无需迭代式的动作条件推演。在每个决策步骤，FinPILOT利用预测器预测的价格轨迹构建基于配置的想象收益目标，并在执行一步交易之前在推理时对预训练策略进行优化。我们的框架与任何预训练策略兼容，并利用预测器的预测对策略进行调整，而无需……

    arXiv:2605.12653v2 Announce Type: replace-cross  Abstract: Reinforcement learning agents for portfolio management are typically trained and deployed as static policies, with no mechanism for policy adjustments using price forecasts at inference time. We propose FinPILOT (Financial Plugin Inference-time Learning for Optimal Trading), a plugin inference-time policy optimization framework inspired by Model Predictive Control (MPC). Under the price-taker assumption, portfolio actions do not affect future market prices, allowing a multi-step price forecaster to provide imagined market trajectories without iterative action-conditioned rollouts. At each decision step, FinPILOT uses the forecaster's predicted price trajectory to construct an allocation-based imagined return objective and optimizes the pretrained policy at inference time before executing one step of the trade. Our framework is compatible with any pretrained policy and adapts the policy using the forecaster's predictions without
    
[^62]: 一种用于复合对数凹采样的近端梯度算法

    A proximal gradient algorithm for composite log-concave sampling

    [https://arxiv.org/abs/2605.12461](https://arxiv.org/abs/2605.12461)

    提出了一种针对复合对数凹分布的近端梯度采样算法，通过结合 $f$ 的梯度评估与 $g$ 的受限高斯预言机，在 $f+g$ 为 $\alpha$-凸且 $f$ 为 $\beta$-光滑的条件下，以 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$ 次预言机调用实现总变差距离下 $\varepsilon$ 精度的采样，匹配了 $g=0$ 情形下的最优结果。

    

    我们提出了一种从 $\mathbb{R}^d$ 上的复合对数凹分布中进行采样的算法，即形如 $\pi\propto e^{-f-g}$ 的密度，假设我们能够获得 $\alpha_f$-凸函数 $f$ 的梯度评估，以及 $\alpha_g$-凸函数 $g$（其中 $\alpha_g \ge 0$）的受限高斯预言机（RGO）。后一要求意味着我们可以容易地从密度 $\text{RGO}_{g,h,y}(x) \propto \exp(-g(x) -\frac{1}{2h}||y-x||^2)$ 中采样，这相当于 $g$ 的近端算子在采样上的类比。如果 $f + g$ 是 $\alpha$-凸的且 $f$ 是 $\beta$-光滑的，我们的采样器在总变差距离下达到 $\varepsilon$ 误差，所需的预言机调用次数为 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$，其中 $\kappa := 1 \vee \beta/\alpha$，$0\le \alpha_g \le \beta$ 且 $\alpha := \alpha_f + \alpha_g > 0$，这与之前 $g=0$ 情形下最先进的结果相匹配。我们进一步将结果扩展到以下情形：(1) ...（摘要在此处被截断）

    arXiv:2605.12461v2 Announce Type: replace-cross  Abstract: We propose an algorithm to sample from composite log-concave distributions over $\mathbb{R}^d$, i.e., densities of the form $\pi\propto e^{-f-g}$, assuming access to gradient evaluations of $\alpha_f$-convex $f$ and a restricted Gaussian oracle (RGO) for $\alpha_g$-convex $g$, where $\alpha_g \ge 0$. The latter requirement means that we can easily sample from the density $\text{RGO}_{g,h,y}(x) \propto \exp(-g(x) -\frac{1}{2h}||y-x||^2)$, which is the sampling analogue of the proximal operator for $g$. If $f + g$ is $\alpha$-convex and $f$ is $\beta$-smooth, our sampler achieves $\varepsilon$ error in total variation distance using $\widetilde{\mathcal O}(\kappa \sqrt d \log^4(1/\varepsilon))$ oracle calls where $\kappa := 1 \vee \beta/\alpha$, with $0\le \alpha_g \le \beta$ and $\alpha := \alpha_f + \alpha_g > 0$, which matches prior state-of-the-art results for the case $g=0$. We further extend our results to cases where (1) $
    
[^63]: 关注差距：偏好学习中的结构感知一致性

    Mind the Gap: Structure-Aware Consistency in Preference Learning

    [https://arxiv.org/abs/2604.27733](https://arxiv.org/abs/2604.27733)

    本文证明了标准替代损失在LLM对齐中无法保证排序一致性，并提出一种结构感知的边际调整方法（SA-DPO），通过利用响应间的语义距离实现稳定且一致的对齐。

    

    将大型语言模型（LLMs）与人类意图对齐，无论是通过显式奖励建模还是直接方法如DPO，从根本上依赖于最小化替代损失作为真实成对排序目标的代理。我们证明了这种依赖对于使用的标准替代损失是有缺陷的：对于神经网络特有的等连续假设集，没有任何标准替代损失能提供有意义的一致性保证。将替代损失最小化到零可能使真实排序误差任意高。为解决此问题，我们将LLM对齐形式化为一个带有边际偏移的排序框架，并推导出$H$-一致性界限，表明强制置信边际$\gamma$不仅有益，而且对一致性是必要的。我们进一步引入了结构感知$H$-一致性及相应目标（SA-DPO），该目标根据响应之间的语义距离调整边际，防止不稳定性。

    arXiv:2604.27733v2 Announce Type: replace  Abstract: Aligning Large Language Models (LLMs) with human intent, whether through explicit reward modeling or direct methods such as DPO, fundamentally relies on minimizing a surrogate loss as a proxy for the true pairwise ranking objective. We prove that this reliance is flawed for the standard surrogate losses used: for the equicontinuous hypothesis sets characteristic of neural networks, no standard surrogate provides a meaningful consistency guarantee. Minimizing the surrogate loss to zero can leave the true ranking error arbitrarily high. To resolve this, we formulate LLM alignment within a margin-shifted ranking framework and derive $H$-consistency bounds showing that enforcing a confidence margin $\gamma$ is not merely beneficial but necessary for consistency. We further introduce Structure-Aware $H$-consistency and a corresponding objective (SA-DPO) that adapts the margin to the semantic distance between responses, preventing instabil
    
[^64]: 面向动力系统的分布到分布神经概率预测框架

    A Distribution-to-Distribution Neural Probabilistic Forecasting Framework for Dynamical Systems

    [https://arxiv.org/abs/2603.25370](https://arxiv.org/abs/2603.25370)

    本文提出了一种分布到分布（D2D）神经概率预测框架，利用核均值嵌入编码输入分布、混合密度网络参数化输出分布，实现了动力系统的递归多步概率预测，避免了求解计算代价高昂的分布演化方程。

    

    概率预测通过预测分布为预测不确定性提供了有原则的表示。使用传统物理模型直接传播预测分布需要求解分布演化方程，而对于高维非线性系统，这一任务在计算上是众所周知的难以承受的。因此，多步预测分布通常使用集合或采样轨迹来近似。本文提出了一种新颖的分布到分布（D2D）神经概率预测框架，该框架在一个可替换的神经预测模块周围配置了兼容的分布编码和解码接口。输入分布采用核均值嵌入表示，输出则由混合密度网络进行参数化，从而能够以分布形式进行递归多步预测。该框架在Lorenz63系统上进行了演示，结果表明D2D模型能够捕获非……

    arXiv:2603.25370v2 Announce Type: replace-cross  Abstract: Probabilistic forecasts provide a principled representation of forecast uncertainty through predictive distributions. Directly propagating predictive distributions with conventional physical models requires solving distributional evolution equations, a task well-known to be computationally prohibitive for high-dimensional nonlinear systems. Consequently, multi-step predictive distributions are commonly approximated using ensembles or sampled trajectories. A novel distribution-to-distribution (D2D) neural probabilistic forecasting framework is proposed with compatible distributional encoding and decoding interfaces around a replaceable neural forecasting module. Kernel mean embeddings represent input distributions, while mixture density networks parameterise outputs, enabling recursive multi-step forecasting in distributional form. The framework is demonstrated on the Lorenz63 system. Results show that the D2D model captures non
    
[^65]: 一种基于模型集成的公平感知预测后处理框架

    A Model Ensemble-Based Post-Processing Framework for Fairness-Aware Prediction

    [https://arxiv.org/abs/2603.18838](https://arxiv.org/abs/2603.18838)

    本文提出一种基于模型集成的后处理框架，该方法独立于模型内部结构、适用于多种学习任务和公平性定义，能够在几乎不损失预测精度的前提下有效提升模型公平性。

    

    在预测性能与公平性之间寻求最优平衡仍然是机器学习中的一项根本性挑战。在本工作中，我们提出了一种后处理框架，通过利用模型集成来促进公平感知预测。该方法的设计独立于任何特定的模型内部结构，因此可广泛应用于各种学习任务、模型架构和公平性定义。通过涵盖分类、回归和生存分析的大量实验，我们证明该框架能够在保持预测精度基本不变或仅受到极小影响的前提下，有效提升公平性。

    arXiv:2603.18838v2 Announce Type: replace  Abstract: Striking an optimal balance between predictive performance and fairness continues to be a fundamental challenge in machine learning. In this work, we propose a post-processing framework that facilitates fairness-aware prediction by leveraging model ensembling. Designed to operate independently of any specific model internals, our approach is widely applicable across various learning tasks, model architectures, and fairness definitions. Through extensive experiments spanning classification, regression, and survival analysis, we demonstrate that the framework effectively enhances fairness while maintaining, or only minimally affecting, predictive accuracy.
    
[^66]: 校准距离的计算与统计难度

    Computational and Statistical Hardness of Calibration Distance

    [https://arxiv.org/abs/2603.18391](https://arxiv.org/abs/2603.18391)

    该论文提出了在均匀边际和无噪声标签假设下精确计算校准距离的高效算法，证明移除任一假设后问题即变为NP难，并给出了适用于一般情形的PTAS以及估计问题所需的Θ(1/ε³)样本复杂度的紧界。

    

    由Błasiok、Gopalan、Hu和Nakkiran（STOC 2023）提出的校准距离，近来已成为衡量概率预测器失准程度的核心指标。我们研究了在给定数据分布的精确描述或仅有数据采样访问的条件下，计算和估计该量的基本问题。当分布具有均匀边际且标签无噪声时，我们给出了一个能精确计算校准距离的高效算法，这改进了Qiao和Zheng（COLT 2024）针对该特殊情形所给出的O(1/√|X|)加性近似。或许令人惊讶的是，当这两个假设中的任意一个被移除时，该问题即变为NP难问题。我们进一步将算法推广为适用于一般情形的多项式时间近似方案（PTAS）。对于估计问题，我们证明了Θ(1/ε³)个样本对于经验校准……（摘要在此处截断）是充分且必要的。

    arXiv:2603.18391v2 Announce Type: replace-cross  Abstract: The distance from calibration, introduced by B{\l}asiok, Gopalan, Hu, and Nakkiran (STOC 2023), has recently emerged as a central measure of miscalibration for probabilistic predictors. We study the fundamental problems of computing and estimating this quantity, given either an exact description of the data distribution or only sample access to it.   We give an efficient algorithm that exactly computes the calibration distance when the distribution has a uniform marginal and noiseless labels, which improves the $O(1/\sqrt{|\mathcal{X}|})$ additive approximation of Qiao and Zheng (COLT 2024) for this special case. Perhaps surprisingly, the problem becomes $\mathsf{NP}$-hard when either of the two assumptions is removed. We extend our algorithm to a polynomial-time approximation scheme for the general case.   For the estimation problem, we show that $\Theta(1/\epsilon^3)$ samples are sufficient and necessary for the empirical cal
    
[^67]: 基于模拟推断中的扩散模型：教程综述

    Diffusion Models in Simulation-Based Inference: A Tutorial Review

    [https://arxiv.org/abs/2512.20685](https://arxiv.org/abs/2512.20685)

    本综述系统梳理了扩散模型在基于模拟推断（SBI）中的最新进展，涵盖训练、推断与评估的设计选择，并讨论了引导、流匹配、一致性模型等概念以及噪声调度、参数化和采样器对效率与统计精度的影响。

    

    扩散模型近来已成为基于模拟推断中强大的学习工具，能够从模拟数据和真实数据中快速且准确地估计潜在参数。其基于分数的公式化为学习参数与观测之间的条件分布或联合分布提供了灵活的方式，从而为各种建模问题提供了通用的解决方案。在本教程综述中，我们综合了扩散模型在SBI领域的最新进展，涵盖训练、推断和评估的设计选择。我们重点介绍了引导、分数组合、流匹配、一致性模型和联合建模等概念所带来的机遇。此外，我们讨论了噪声调度、参数化方法和采样器如何影响效率与统计准确性。最后，我们通过涵盖不同参数维度、模拟预算和模型类型的案例研究来阐释这些概念。

    arXiv:2512.20685v4 Announce Type: replace-cross  Abstract: Diffusion models have recently emerged as powerful learners for simulation-based inference (SBI), enabling fast and accurate estimation of latent parameters from simulated and real data. Their score-based formulation offers a flexible way to learn conditional or joint distributions over parameters and observations, thereby providing a versatile solution to various modeling problems. In this tutorial review, we synthesize recent developments on diffusion models for SBI, covering design choices for training, inference, and evaluation. We highlight opportunities created by various concepts such as guidance, score composition, flow matching, consistency models, and joint modeling. Furthermore, we discuss how efficiency and statistical accuracy are affected by noise schedules, parameterizations, and samplers. Finally, we illustrate these concepts with case studies across parameter dimensionalities, simulation budgets, and model type
    
[^68]: 用于非晶粒子系统的玻尔兹曼生成器

    Boltzmann generators for amorphous particle systems

    [https://arxiv.org/abs/2512.16607](https://arxiv.org/abs/2512.16607)

    本研究针对平衡采样极其困难的非晶材料（玻璃），通过将所需的等变性直接嵌入黎曼结构中，开发了首个专为非晶粒子系统定制的玻尔兹曼生成器。

    

    在热力学平衡态下对构型进行采样是统计物理学中一个长期存在的挑战。玻尔兹曼生成器通过使用生成模型提出独立构型来解决这一问题，随后利用精确的似然评估通过重要性采样对这些构型进行重新加权。近期基于连续归一化流和流匹配的玻尔兹曼生成器在粒子系统和生物分子领域取得了显著成功。然而，这些方法尚未被扩展到非晶材料（玻璃），而这类材料的平衡采样以极其缓慢而著称。由于其无序结构，非晶材料的不变性和几何约束与晶体和生物分子不同，这阻碍了现有生成模型的直接应用。在本研究中，我们通过将所需的等变性直接构建到黎曼（流形）结构中，开发了专为非晶材料定制的玻尔兹曼生成器。

    arXiv:2512.16607v3 Announce Type: replace-cross  Abstract: Sampling configurations in thermodynamic equilibrium is a long-standing challenge in statistical physics. Boltzmann generators address this problem by employing generative models to propose independent configurations, which are then reweighted via importance sampling using exact likelihood evaluations. Recent Boltzmann Generators based on continuous normalizing flows and flow matching have achieved significant success for particle systems and biomolecules. However, these approaches have not been extended to amorphous materials (glasses), for which equilibrium sampling is notoriously slow. Because of their disordered structure, the invariances and geometrical constraints of amorphous materials differ from those of crystals and biomolecules, preventing the direct use of existing generative models. Here, we develop Boltzmann Generators tailored to amorphous materials by building the required equivariances directly into Riemannian 
    
[^69]: 多任务神经扩散过程

    Multi-task Neural Diffusion Processes

    [https://arxiv.org/abs/2510.03419](https://arxiv.org/abs/2510.03419)

    提出多任务神经扩散过程（MT-NDPs），通过任务编码器提取任务表示并条件化扩散模型，实现跨任务信息共享的任务条件化概率回归与少样本自适应。

    

    神经扩散过程为函数分布建模提供了一种可扩展的非高斯方法，但现有形式仅限于单任务推理，无法捕捉相关任务之间的依赖关系。我们提出了多任务神经扩散过程（MT-NDPs），这是一种扩展方法，通过引入任务编码器来实现任务条件化的概率回归以及跨相关函数的少样本自适应。任务编码器从上下文观测中提取低维表示，并以此表示为条件来约束扩散模型，从而实现跨任务的信息共享，同时支持大小可变的上下文集合，并保持对上下文观测顺序的置换不变性。我们通过合成实验以及两个真实世界应用（涉及风力涡轮机数据和学校考试结果）对MT-NDPs进行了评估。结果表明性能得到了改进。

    arXiv:2510.03419v3 Announce Type: replace-cross  Abstract: Neural diffusion processes provide a scalable, non-Gaussian approach to modelling distributions over functions, but existing formulations are limited to single-task inference and do not capture dependencies across related tasks. We propose multi-task neural diffusion processes (MT-NDPs), an extension that incorporates a task encoder to enable task-conditioned probabilistic regression and few-shot adaptation across related functions. The task encoder extracts a low-dimensional representation from context observations and conditions the diffusion model on this representation, allowing information sharing across tasks while accommodating variable-sized context sets and preserving permutation invariance with respect to the ordering of context observations. We evaluate MT-NDPs through synthetic experiments and two real-world applications involving wind turbine data and school examination outcomes. The results demonstrate improved pr
    
[^70]: 用于融合神经算子与经典方法的贪心PDE路由器

    A Greedy PDE Router for Blending Neural Operators and Classical Methods

    [https://arxiv.org/abs/2509.24814](https://arxiv.org/abs/2509.24814)

    本文提出一种近似贪心PDE路由器，在混合迭代求解过程中每步智能选择神经算子或经典数值求解器以融合两者优势，在多种PDE求解任务中持续缩短求解时间。

    

    在求解偏微分方程（PDE）时，经典数值求解器通常计算开销较大，而机器学习方法则可能受到谱偏差的影响，无法捕捉高频分量。设计一种最优的混合迭代求解器——即在每次迭代时从求解器集合中选择一个求解器，以利用它们之间互补的优势——构成了一个具有挑战性的组合优化问题。虽然贪心选择方法因其在Lipschitz假设下对最优解具有常数因子近似保证而备受青睐，但它需要知道每一步的真实误差，而这在实践中是无法获得的。我们通过提出一种近似贪心路由器来解决这一问题，该路由器能够高效地模拟求解器选择中的贪心策略。在泊松方程、对流-扩散方程以及各向异性扩散方程上的实证结果表明，与单一求解器基线和现有方法相比，我们的方法能够持续减少求解时间。

    arXiv:2509.24814v3 Announce Type: replace-cross  Abstract: When solving PDEs, classical numerical solvers are often computationally expensive, while machine learning methods can suffer from spectral bias, failing to capture high-frequency components. Designing an optimal hybrid iterative solver--where, at each iteration, a solver is selected from an ensemble of solvers to leverage their complementary strengths--poses a challenging combinatorial problem. While greedy selection is desirable for its constant-factor approximation guarantee to the optimal solution under Lipschitz assumptions, it requires knowledge of the true error at each step, which is unavailable in practice. We address this by proposing an approximate greedy router that efficiently mimics a greedy approach to solver selection. Empirical results on the Poisson, convection-diffusion, and anisotropic diffusion equations show that our method consistently reduces time-to-solution relative to single-solver baselines and exist
    
[^71]: 粗糙路径的期望签名核

    Expected Signature Kernels for Rough Paths

    [https://arxiv.org/abs/2509.07893](https://arxiv.org/abs/2509.07893)

    本文将Lévy粗糙路径的期望签名核刻画为扩展Goursat型方程组解的标量分量，证明了跳跃情形下无限维系统的适定性并给出显式误差控制的有限维截断，从而无需辅助参考样本即可直接计算单样本签名-MMD检验中的参考依赖项。

    

    期望签名核是路径空间上概率律的一种相似性度量，并直接应用于签名最大均值差异中。基于 Friz、Hager 和 Tapia（Forum Math. Sigma, 2022）以及 Lemercier、Lyons 和 Salvi（SIAM J. Numer. Anal., 2026）的工作，我们将群值非齐次Lévy过程（又称Lévy粗糙路径）的期望签名核刻画为一个扩展Goursat型方程组解的标量分量。我们给出了一个原创性的描述：首先在连续情形下，该有限维方程组是精确的；而在存在跳跃的情形下，该方程组本质上是无限维的，我们证明了其适定性，并推导出具有显式误差控制的有限维截断。这使得在单样本签名-MMD检验中，无需辅助参考样本即可直接计算依赖于参考的项。

    arXiv:2509.07893v2 Announce Type: replace-cross  Abstract: The expected signature kernel is a similarity measure of probability laws on path space and enters directly into signature maximum mean discrepancies. Building on [P. K. Friz, P. P. Hager, and N. Tapia, Forum Math. Sigma, 10 (2022), e42] and [M. Lemercier, T. Lyons, and C. Salvi, SIAM J. Numer. Anal., 64 (2026), pp. 1095-1125], we characterize the expected signature kernel of group-valued inhomogeneous L\'evy processes, aka L\'evy rough paths, as the scalar component of the solution to an extended Goursat-type system. We give an original description, first in the continuous case where the finite-dimensional system is exact. In the presence of jumps, this system is inherently infinite-dimensional; we prove its well-posedness and derive finite-dimensional truncations with explicit error control. This permits direct evaluation of the reference-dependent terms in one-sample signature-MMD tests without an auxiliary reference sample.
    
[^72]: 深度神经网络初始化方差的最优条件：基于SGD动力学的视角

    Optimal Condition for Initialization Variance in Deep Neural Networks: An SGD Dynamics Perspective

    [https://arxiv.org/abs/2508.12834](https://arxiv.org/abs/2508.12834)

    该论文通过将SGD近似为朗之万动力学并利用Fokker-Planck方程与KL散度，建立了期望损失与初始化分布的联系，从而推导出深度神经网络初始化方差的最优条件。

    

    随机梯度下降（SGD）是机器学习中最基础的优化算法之一，通过连续时间近似可以将其重新表述为描述朗之万动力学的Fokker-Planck方程，这一视角已启发了众多理论研究。在该框架下，我们通过Kullback-Leibler（KL）散度研究了由该方程导出的准平稳分布与初始分布之间的关系。由于准稳态分布依赖于期望代价函数，KL散度最终揭示了期望代价函数与初始化分布之间的联系。将该方法应用于深度神经网络（DNN）模型后，我们可以用初始化参数显式地表达期望损失函数的界限。进而，通过最小化该界限，我们得到了高斯（初始化方差）的最优条件……

    arXiv:2508.12834v2 Announce Type: replace-cross  Abstract: Stochastic gradient descent (SGD), one of the most fundamental optimization algorithms in machine learning (ML), can be recast through a continuous-time approximation as a Fokker-Planck equation for Langevin dynamics, a viewpoint that has motivated many theoretical studies. Within this framework, we study the relationship between the quasi-stationary distribution derived from this equation and the initial distribution through the Kullback-Leibler (KL) divergence. As the quasi-steady-state distribution depends on the expected cost function, the KL divergence eventually reveals the connection between the expected cost function and the initialization distribution. By applying this to deep neural network models (DNNs), we can express the bounds of the expected loss function explicitly in terms of the initialization parameters. Then, by minimizing this bound, we obtain an optimal condition of the initialization variance in the Gauss
    
[^73]: 基于总风险先验的正式贝叶斯迁移学习

    Formal Bayesian Transfer Learning via the Total Risk Prior

    [https://arxiv.org/abs/2507.23768](https://arxiv.org/abs/2507.23768)

    该论文提出了一种新的贝叶斯迁移学习方法，其核心创新在于使用以源参数为条件的风险最小化器（而非传统的经验损失最小化器）来构建源数据集与目标数据集所有参数的单一联合先验分布，从而实现完整的贝叶斯不确定性量化以及通过Gibbs采样对源数据集进行选择和模型平均。

    

    现有的迁移学习方法难以应对源数据集有限且无法保证与目标数据集良好对齐的情况。一种典型的策略是将源数据上的经验损失最小化器用作目标参数的先验均值。我们的关键概念性贡献是改为使用以源参数为条件的风险最小化器。这使得我们能够基于源数据集和目标数据集为所有参数构建一个单一的联合先验分布。因此，我们可以获得完整的贝叶斯不确定性量化，并可以通过对控制每个源数据集是否被纳入的指示变量进行吉布斯采样（Gibbs sampling）来执行模型平均。我们展示了该先验的一种特定实例化如何在变换后的坐标系中导出贝叶斯Lasso，并讨论了将我们的方法扩展到中等规模数据集的计算技术。我们讨论了……

    arXiv:2507.23768v2 Announce Type: replace-cross  Abstract: Existing methods for transfer learning struggle to deal with situations where the source datasets are limited and not guaranteed to be well-aligned with the target dataset. A typical strategy is to use the empirical loss minimizer on the source data as a prior mean for the target parameters. Our key conceptual contribution is to use a risk minimizer conditional on source parameters instead. This allows us to construct a single joint prior distribution for all parameters from the source datasets as well as the target dataset. As a consequence, we benefit from full Bayesian uncertainty quantification and can perform model averaging via Gibbs sampling over indicator variables governing the inclusion of each source dataset. We show how a particular instantiation of our prior leads to a Bayesian Lasso in a transformed coordinate system and discuss computational techniques to scale our approach to moderately sized datasets. We discus
    
[^74]: PAC因果可学习性：从识别到有限样本保证

    PAC-Causal Learnability: From Identification to Finite-Sample Guarantees

    [https://arxiv.org/abs/2507.18903](https://arxiv.org/abs/2507.18903)

    该论文提出了PACC因果可学习性框架，首次从PAC学习角度为因果估计提供了有限样本的统一精度-置信度保证，并揭示其样本复杂度由因果识别强度、目标复杂度和干扰复杂度三个关键因素决定。

    

    因果识别探讨的是在理想化的无限数据设定下，能否从观测分布中恢复出因果目标。在实践中，只有有限样本可用，而可识别性本身并不能确定可靠估计需要多少数据。我们引入了“可能近似正确因果”（PACC）可学习性，它在因果模型类上形式化了统一的精度-置信度保证。从PAC学习的视角出发，我们探究需要多少样本才能以至少1-δ的概率将因果目标估计到精度ε。我们证明PACC样本复杂度由三个反复出现的因素决定：因果识别强度、目标复杂度和干扰复杂度，并推导出一个将三者与有限样本因果学习联系起来的通用充分界。我们通过自身对照病例系列、工具变量和倾向性（评分）方法来阐述该框架。

    arXiv:2507.18903v2 Announce Type: replace-cross  Abstract: Causal identification asks whether a causal target can be recovered from the observational distribution in an idealized infinite-data setting. In practice, only finite samples are available, and identifiability alone does not determine how much data are needed for reliable estimation. We introduce Probably Approximately Correct Causal (PACC) learnability, which formalizes uniform accuracy-confidence guarantees over a causal model class. From a PAC-learning perspective, we ask how many samples are sufficient to estimate a causal target to accuracy $\epsilon$ with probability at least $1-\delta$. We show that PACC sample complexity is shaped by three recurring factors: causal identification strength, target complexity, and nuisance complexity, and derive a general sufficient bound linking them to finite-sample causal learning. We illustrate the framework through self-controlled case series, instrumental variables, and propensity-
    
[^75]: 用于万能逼近的并行层归一化

    Parallel Layer Normalization for Universal Approximation

    [https://arxiv.org/abs/2505.13142](https://arxiv.org/abs/2505.13142)

    该论文证明仅含单个层归一化单元的浅层网络无论宽度多大都不具备万能逼近能力，而提出的并行层归一化网络（PLN-Nets）随着并行分组数量的增加即可实现万能逼近，且该机制还可推广到RMSNorm变体、逐位置前馈网络和卷积网络。

    

    层归一化通常因其 在优化中的作用而被研究，但它同时也在仿射层之间充当一种多元非线性。我们研究了当网络宽度增长时，LN 单元的分组方式如何影响神经网络的逼近能力。在并行层归一化网络中，固定维度的 LN 单元被独立地应用于两个仿射层之间的各个组。我们证明，仅含一个 LN 单元的浅层 LN-Net 无论宽度多大都不具备万能逼近性，而 PLN-Nets 则随着并行组数量的增加变得具有万能逼近性。该机制在合适的分组结构下可扩展到深层网络、均方根归一化 变体、逐位置前馈网络以及卷积网络。受控的多层感知机 (MLP) 与物理信息神经网络 (PINN) 实验支持了 PLN 的实际可行性，而 CNN 和 ViT 实验表明组数会塑造有限样本拟合包络，但 d……（原文摘要在此截断）

    arXiv:2505.13142v3 Announce Type: replace  Abstract: Layer Normalization (LN) is usually studied for its role in optimization, but it also acts as a multivariate nonlinearity between affine layers. We investigate how grouping LN units affects the approximation power of neural networks as width grows. In Parallel Layer Normalization networks (PLN-Nets), fixed-dimensional LN units are applied independently to groups between two affine layers. We prove that a shallow LN-Net with one LN unit is non-universal regardless of width, whereas PLN-Nets become universal as the number of parallel groups grows. The mechanism extends to deep networks, Root Mean Square Normalization (RMSNorm) variants, position-wise feed-forward networks, and convolutional networks under suitable grouping structures. Controlled MLP and physics-informed neural network (PINN) experiments support PLN's practical viability, while CNN and ViT experiments show that group count shapes the finite-sample fitting envelope but d
    
[^76]: STAMP：用于按时间顺序划分图域自适应的对称时间平均消息传递

    STAMP: Symmetric Temporal Averaging Message Passing for Domain Adaptation in Chronologically Split Graphs

    [https://arxiv.org/abs/2411.10957](https://arxiv.org/abs/2411.10957)

    提出对称时间平均消息传递（STAMP），通过在时间戳间平衡消息强制时间域间的一阶矩不变性，从而解决按时间顺序划分图中的分布偏移问题，提升半监督节点分类性能。

    

    本文研究按时间顺序划分下图结构数据中的域自适应问题。我们考虑直推式半监督节点分类（SSNC），即利用历史节点的标签来预测较新节点的标签。图连通性中的时间依赖性会引起跨时间的分布偏移，这往往会降低在历史节点上训练的模型的性能。尽管具有重要的实际意义，但按时间顺序划分图下的域自适应问题仍然相对未被充分探索。我们提出对称时间平均消息传递（STAMP），这是一种时间戳平衡算子，用时间对齐的传播取代标准的邻域聚合。我们将其针对异质 ogbn-mag 数据集的专门变体称为 STAMP-mag，而将 STAMP 保留用于一般形式。通过在时间戳之间平衡消息，STAMP 在时间域之间强制实现一阶矩不变性，同时保持……

    arXiv:2411.10957v2 Announce Type: replace-cross  Abstract: This paper addresses domain adaptation in graph-structured data under chronological splits. We consider transductive semi-supervised node classification (SSNC), in which labels of historical nodes are used to predict labels of more recent nodes. Temporal dependencies in graph connectivity induce distribution shifts across time, often degrading the performance of models trained on past nodes. Despite its practical relevance, domain adaptation under chronological graph splits remains relatively underexplored. We propose Symmetric Temporal Averaging Message Passing (STAMP), a timestamp-balanced operator that replaces standard neighborhood aggregation with temporally aligned propagation. We call its specialized variant for the heterogeneous ogbn-mag dataset STAMP-mag, reserving STAMP for the general formulation. By balancing messages across timestamps, STAMP enforces first-moment invariance across temporal domains while remaining s
    
[^77]: 高斯随机场的次大值与精确间距检验

    Second Maximum of a Gaussian Random Field and Exact (t-)Spacing test

    [https://arxiv.org/abs/2406.18397](https://arxiv.org/abs/2406.18397)

    本文提出了黎曼流形上高斯随机场次大值的新概念，借助 Kac-Rice 公式推导出以次大值和 Hessian 矩阵独立部分为条件的全局最大值的显式分布，从而构造出一种可用于检测稀疏备择假设的精确间距检验方法。

    

    本文介绍了在黎曼流形上定义的高斯随机场的次大值这一新概念：当随机场在某点 $t_1$ 达到全局最大值后，次大值指的是在移除场在 $t_1$ 处的值和梯度之后，场在远离 $t_1$ 处所能达到的最大值。这一统计量是刻画全局最大值分布的有力工具。通过采用特定的 Kac-Rice 公式，我们推导出了在给定次大值和黎曼 Hessian 矩阵独立部分（即 $t_1$ 处 Hessian 矩阵中与场在 $t_1$ 处的值和梯度无关的部分）的条件下，全局最大值的显式分布。该方法给出了一种基于这些最大值之间间距的精确检验，我们称之为间距检验。我们研究了该检验在检测高斯对称

    arXiv:2406.18397v4 Announce Type: replace-cross  Abstract: In this article, we introduce the novel concept of the second maximum of a Gaussian random field defined on a Riemannian manifold: Once the field reaches its global maximum at a point $t_1$, the second maximum is the largest value of the field away from $t_1$, after the value and the gradient of the field at $t_1$ have been removed. This statistic serves as a powerful tool for characterizing the distribution of the global maximum. By employing an ad-hoc Kac-Rice formula, we derive the explicit distribution of the maximum, conditioned on the second maximum and the independent part of the Riemannian Hessian (the part of the Hessian at $t_1$ that is independent of the value and the gradient of the field at $t_1$). This approach yields an exact test based on the spacing between these maxima, which we refer to as the spacing test.   We investigate the applicability of this test for detecting sparse alternatives in Gaussian symmetric
    
[^78]: 利用分布鲁棒优化获取可解释的分类模型

    Obtaining Explainable Classification Models using Distributionally Robust Optimization. (arXiv:2311.01994v1 [stat.ML])

    [http://arxiv.org/abs/2311.01994](http://arxiv.org/abs/2311.01994)

    本论文介绍了一种利用分布鲁棒优化获取可解释的分类模型的方法，通过构建稀疏的规则集合来同时解决规则集的稀疏性和预测准确性之间的权衡，从而保证泛化性能并降低计算成本。

    

    对于人类用户来说，模型的可解释性对于理解提议分类器如何根据特征值给数据分配标签至关重要。我们研究使用特征值规则集构建的广义线性模型，该模型可以捕捉非线性依赖和交互作用。规则集的稀疏性和预测准确性之间存在固有的权衡。使用现有方法来找到合适的稀疏度选择（例如通过交叉验证）计算成本很高。我们提出了一种新的公式来学习同时解决这些竞争因素的规则集合。通过利用分布鲁棒优化来确保良好的泛化性能，同时保持低计算成本。该公式利用列生成有效地搜索规则集合的空间并构建稀疏的规则集合，与随机森林或Boosting及其变体等技术相比。我们提出了理论结果来推动这一公式的发展。

    Model explainability is crucial for human users to be able to interpret how a proposed classifier assigns labels to data based on its feature values. We study generalized linear models constructed using sets of feature value rules, which can capture nonlinear dependencies and interactions. An inherent trade-off exists between rule set sparsity and its prediction accuracy. It is computationally expensive to find the right choice of sparsity -- e.g., via cross-validation -- with existing methods. We propose a new formulation to learn an ensemble of rule sets that simultaneously addresses these competing factors. Good generalization is ensured while keeping computational costs low by utilizing distributionally robust optimization. The formulation utilizes column generation to efficiently search the space of rule sets and constructs a sparse ensemble of rule sets, in contrast with techniques like random forests or boosting and their variants. We present theoretical results that motivate an
    

