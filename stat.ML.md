# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Phase Transition from Random Features to Fully Trainable Networks](https://arxiv.org/abs/2610.31663) | 本文为部分可训练的两层神经网络建立了统一的平方 L₂ 逼近误差界 O((κ_P log(m/δ)/m)^α)，揭示了从随机特征到全可训练网络之间存在相变现象：初始改善快速平滑随后收益递减，且初始化失配越大转变越剧烈。 |
| [^2] | [Reliable AI-Assisted Decisions via Selective Auditing](https://arxiv.org/abs/2610.31662) | 提出了一个在线选择性审计框架，在最小化专家审计频率的同时，提供可证明的错误控制保证，使未经审计的AI错误比例始终低于用户设定的阈值。 |
| [^3] | [SympFormer: Inertial Attention Blocks from Damped Hamiltonian Dynamics on Spaces of Probability Measures](https://arxiv.org/abs/2610.31600) | 该论文提出 SympFormer，通过在概率密度空间上建立 Nesterov 型阻尼哈密顿动力学的几何框架（包括局部迁移率几何与商 Stein 几何），从动力学角度推导出具有加速特性的惯性注意力模块。 |
| [^4] | [When to Follow the Teacher: Advantage-Based On-Policy Distillation](https://arxiv.org/abs/2610.31563) | 该论文提出基于优势的在策略蒸馏（AOPD），仅利用学生标准回滚轨迹及其终止奖励来估计教师动作在学生策略下的反事实优势，从而只强化真正对学生有益的教师动作，克服了传统在线策略蒸馏中“始终跟随教师”的局限。 |
| [^5] | [Diffusion Language Models for Perplexity-Aware Black-Box Attacks](https://arxiv.org/abs/2610.31558) | 该论文提出将连续潜空间攻击框架与连续扩散语言模型（dLLM）直接耦合，以学习到的语言生成分布取代人工随机映射，从而生成更自然、更难被防护机制检测且基于自然提示的黑盒攻击提示词。 |
| [^6] | [Nonparametric autoregressive estimation of probability distributions with parallel gradient boosting](https://arxiv.org/abs/2610.31524) | 该论文提出通过自回归因子分解将联合概率分布的估计转化为条件分布估计，并证明在满足次可加性的损失函数下，自回归估计器的收敛速度界可由条件分布估计器的收敛速度界导出，从而将联合分布估计的难度归结为自回归分解中最难因子的估计难度，并借助并行梯度提升实现高效估计。 |
| [^7] | [SABLE: Scalable Affine Bayesian Latent-space Exploration](https://arxiv.org/abs/2610.31495) | SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。 |
| [^8] | [Dimension-free regimes for unadjusted Langevin in ridge-separable models: A subspace robust Wasserstein analysis](https://arxiv.org/abs/2610.31431) | 该论文利用子空间鲁棒Wasserstein距离分析未调整朗之万算法在岭可分模型中的偏差，证明在过参数化和大样本区间下算法复杂度可摆脱对维度的依赖。 |
| [^9] | [Metric Conformal Prediction via the Conditional Expected Local Radius](https://arxiv.org/abs/2610.31336) | 该论文提出条件期望局部半径（CELR）这一新度量，构建了适用于一般度量空间中非欧几里得响应的保形预测方法，结合度量分布随机森林实现高效计算，并从理论上证明了估计量一致性、渐近条件有效性及有限样本边际覆盖率。 |
| [^10] | [Deep Learning for Random Objects in Metric Spaces: Beyond Manifold Assumptions](https://arxiv.org/abs/2610.31328) | 该论文提出了一个直接在响应空间度量结构上运行的随机对象回归深度学习框架，无需流形或欧几里得嵌入假设，同时具备统计一致性与收敛速率保证。 |
| [^11] | [Local Discrimination of Binary Classifiers via Localized AUC Functional](https://arxiv.org/abs/2610.31216) | 本文提出局部化AUC（LAUC）指标及LOCATOR方法，能够识别风险预测模型表现不佳的患者队列，从而克服传统AUC评估忽略性能异质性、可能导致临床决策失误的局限。 |
| [^12] | [Prediction Uncertainty Controlled Computer Model Calibration](https://arxiv.org/abs/2610.31163) | 该论文提出一个两层框架，将条件流匹配摊销后验推断与共形风险控制相结合，直接控制校准后模拟器的预测覆盖率，即使在校准参数弱识别或模型差异被误设的情况下也能保证名义的预测区间水平。 |
| [^13] | [What Comes Next and Why: Interpretable Next-Event Prediction with Neuro-Symbolic Rules](https://arxiv.org/abs/2610.31114) | 提出了神经符号模型RULES及其与残差网络结合的RUNES框架，通过从数据中直接学习可解释的序列规则，在下一事件预测中同时实现了有竞争力的准确率和透明的可解释性。 |
| [^14] | [Transformers with CoT Discover Heavy-Ball Acceleration Through Training](https://arxiv.org/abs/2610.31089) | 本文证明Transformer在梯度下降教师的初始监督下，通过真实回归权重的轨迹级监督训练，能够自发学会实现Heavy-Ball动量加速方法，超越对教师算法的单纯模仿并获得加速效果。 |
| [^15] | [Random Projections for Spectral Algorithms: Optimal Rates in Sobolev Norms](https://arxiv.org/abs/2610.30948) | 本文利用RKHS嵌入性质与算子相似性的精细分析，证明了基于随机投影（随机化草图与Nyström子采样）的谱算法在一般条件下于Sobolev范数中达到匹配极小极大下界的最优学习速率，且不牺牲统计效率。 |
| [^16] | [Continuous-depth Deep Gaussian Processes](https://arxiv.org/abs/2610.30890) | 提出将深度高斯过程的深度视为由随机微分方程刻画的连续演化，并在深度索引的漂移函数族上施加高斯过程先验（CDGP 与 FlowDGP 两种实现），从而缓解传统离散深度 DGP 随深度增长而难以优化的问题。 |
| [^17] | [xVGAE: A Hierarchical Variational Graph Autoencoder for Exchangeable Graphs](https://arxiv.org/abs/2610.30815) | 本文提出 xVGAE，一种受 Aldous-Hoover 定理启发的层次化变分图自编码器，在生成质量与最先进模型相当的同时，将生成速度提升了几个数量级。 |
| [^18] | [Width and Depth Limits Commute During Training](https://arxiv.org/abs/2610.30729) | 本文将残差网络无限宽与无限深极限的可交换性从初始化阶段首次扩展到训练过程，证明了在Depth-μP参数化与梯度裁剪下，线性残差网络在任意有限训练步数内，网络输出沿宽度-深度联合增长路径按 $L^2$ 收敛且两极限可交换。 |
| [^19] | [Residual Calibration via Local Feature-Space Refinement](https://arxiv.org/abs/2610.30652) | 提出了局部自适应概率估计（LAPE），一种即插即用的事后残差校准框架，它以全局校准器输出为锚点，利用相邻校准样本的非参数局部证据对预测进行精细化，从而捕捉特征空间中局部区域的空间变化失准。 |
| [^20] | [Lifetime-valid decisions by continually evolving AI analysts](https://arxiv.org/abs/2610.30562) | 提出“持续验证”方法，在部署前使用可随机化和扩展的零假设模拟数据评估每个AI分析师版本，并证明无论更新如何选择，部署第一类错误率超标版本的终身概率始终有界，从而保障持续演化的AI分析师所有后续决策的终身有效性。 |
| [^21] | [Flow-based Spectral Kernel Learning for Nonstationary Attention](https://arxiv.org/abs/2610.30465) | 提出基于流的谱核注意力（FSKA），利用归一化流参数化非平稳的二元谱密度来学习灵活且可正则化的注意力核，同时支持移除查询-键投影的简化设计，实现参数效率更高的线性注意力架构。 |
| [^22] | [High-Value Training Data from Noisy, Diverse Sources](https://arxiv.org/abs/2610.30421) | 提出一种基于元梯度的可扩展数据加权方法，通过学习可泛化、可迁移的参数化加权规则，为奖励建模等任务从嘈杂多样的数据源中筛选出高价值训练数据。 |
| [^23] | [When Reconstruction Is Not Enough: Doubly Robust Counterfactual and Interventional Mediation Analysis with Learned Multimodal Representations](https://arxiv.org/abs/2610.30220) | 该论文证明了仅基于协变量的训练目标（重建、对比或物理信息）无法保证多模态表示足以支持因果识别，给出了表示保持可识别性的条件，并提出了双稳健的反事实与干预中介分析方法。 |
| [^24] | [AI Mindmeld: A Latent Interconnect Protocol](https://arxiv.org/abs/2610.30215) | 提出潜在互联协议（LIP），通过仅训练轻量级连接器将专家模型的潜在表示直接映射到LLM内部状态，实现无需联合训练、可扩展且可组合的模型间直接知识交换，突破了以文本等多模态数据传递信息的瓶颈。 |
| [^25] | [Differentially Private Synthetic Data via Generalized Moment Matching](https://arxiv.org/abs/2610.30196) | 提出了广义矩匹配这一通用框架，通过私有估计矩、拟合概率测度并采样合成记录来生成差分隐私合成数据，可应用于Wasserstein距离、最大均值差异等多种积分概率度量，并在固定维度下达到相差对数因子的极小极大最优误差率。 |
| [^26] | [Why Constrained Updates Still Forget: A Functional-Path Analysis of Continual SFT](https://arxiv.org/abs/2610.30191) | 该论文提出“函数路径”分析框架，将持续微调导致的遗忘分解为受控方向、梯度陈旧性、探针失配与有限步效应四个因素，并通过匹配的QLoRA干预实验揭示约束放置在Adam之前还是之后对旧能力保留的影响远大于约束粒度本身。 |
| [^27] | [WhatIf-Bench: Benchmarking Intervention-Aware Time Series Forecasting](https://arxiv.org/abs/2610.26582) | 该论文提出了WhatIf-Bench，一个基于数据中心、电网、交通和供水系统机理仿真器构建的半合成基准，用于系统评估时间序列基础模型利用协变量进行干预感知情景预测的能力。 |
| [^28] | [Parallel gradient boosting for flexible estimation of conditional distributions](https://arxiv.org/abs/2607.13550) | 该论文提出了一种称为“并行梯度提升”的梯度提升算法改进方法，通过在每次迭代中仅训练一个基模型即可同时预测多个目标（如整个条件分布），克服了传统提升方法在多输出回归任务中计算开销大、且现有高效变体受限于特定损失函数和决策树基学习器的局限。 |
| [^29] | [Nested Sampling: A Critical and Comprehensive Theoretical Guide](https://arxiv.org/abs/2606.17916) | 本文全面而批判性地阐述了嵌套采样的理论推导与实践挑战，指出其成功的关键在于从受似然约束的先验中高效采样，并系统梳理了它与重要性采样及垂直蒙特卡洛之间的联系。 |
| [^30] | [Evaluating Neural Probabilistic Forecasts of Sparse Earthquake Counts: What Survives a Careful Evaluation](https://arxiv.org/abs/2605.21437) | 对天山-帕米尔地区周地震计数神经概率预报的严格评估表明：泊松类预报（含ETAS）严重低估单周发生3次以上地震的概率（低估8.3倍），引入一个离散参数即可解决；而神经网络相对线性模型的优势在为每个网格加入截距后便不再显著。 |
| [^31] | [Policy Learning with Observational Data: The Case of Hepatitis C Treatment for HIV/HCV Co-Infected Patients](https://arxiv.org/abs/2605.16593) | 本文提出了一种在较弱假设下从观测数据推导多行动政策规则的方法，通过加权K-means算法估计条件平均处理效应并结合标准决策树生成可行政策规则，且能处理治疗依从性不完美的情况，并成功应用于HIV/HCV合并感染患者的丙肝治疗方案选择。 |
| [^32] | [K-Means as the Zero-Temperature Limit of Normalized RBF Assignments: Rates, Trajectories, and Exact Recovery with Entmax](https://arxiv.org/abs/2603.04625) | 本文严格证明了K-均值聚类是归一化RBF分配在零温度下的Γ-极限，从目标函数、优化方法、单次更新和轨迹收敛等多个层面定量刻画了软RBF分配与硬K-均值之间的联系，并利用Entmax实现了精确恢复。 |
| [^33] | [Near-Optimal Regret for KL-Regularized Multi-Armed Bandits](https://arxiv.org/abs/2603.02155) | 该论文通过新颖的分层论证对KL-UCB算法进行精细分析，首次给出了对臂数 $K$ 呈线性依赖的高概率后悔上界 $\tilde{O}(\eta K\log^2T)$，并以首个非常数下界 $\Omega(\eta K\log T)$ 证明了其近紧性。 |
| [^34] | [High-dimensional learning dynamics of multi-pass Stochastic Gradient Descent in multi-index models](https://arxiv.org/abs/2601.21093) | 本文在样本量与维度成比例的高维区间内，首次对非凸多指标模型上的多遍随机梯度下降动力学建立了渐近精确的刻画，证明其收敛于一种新颖的动力学平均场连续时间极限，且适用于任意次线性批量大小及相应的临界学习率缩放。 |
| [^35] | [Penalized Fair Regression for Multiple Groups in Chronic Kidney Disease](https://arxiv.org/abs/2512.17340) | 提出一种面向多群体的惩罚化公平回归通用框架，通过不公平性惩罚与新颖的自动惩罚权重选择评分函数，实现高效求解，并在慢性肾病研究中取得优于现有方法的公平性-准确性平衡。 |
| [^36] | [Stochastic Matching Bandits with Rare Optimization Updates](https://arxiv.org/abs/2509.04194) | 本文提出了MNL选择模型下随机匹配老虎机的批处理算法，通过将昂贵的匹配优化更新次数限制在Θ(log log T)次，在大幅降低计算开销的同时保持了遗憾最优性能。 |
| [^37] | [Multiply Robust Conformal Risk Control with Coarsened Data](https://arxiv.org/abs/2508.15489) | 该论文提出了一个结合半参数理论与保形方法的通用框架，通过RSP和RSCP两种互补程序，将双重稳健估计方法推广到一般粗化数据情形，实现了数据缺失或粗化下具有多重稳健性的保形预测与风险控制。 |
| [^38] | [Meta-learning of Gibbs states for many-body Hamiltonians with applications to Quantum Boltzmann Machines](https://arxiv.org/abs/2507.16373) | 该论文提出了两种元学习算法，通过对有限训练参数值集合进行联合优化，在NISQ设备上高效制备参数化多体哈密顿量的吉布斯态，并可应用于量子玻尔兹曼机。 |
| [^39] | [Diffusion-Based Hypothesis Testing and Change-Point Detection](https://arxiv.org/abs/2506.16089) | 本文将基于分数的假设检验与变点检测方法推广为基于扩散的版本，通过数值优化权重矩阵提升检测能力，并从理论上量化了算法性能及其达到最优的条件。 |
| [^40] | [Eluder-Dimension-Free Regret for Stage-Wise Constrained Contextual Bandits via Offline Regression](https://arxiv.org/abs/2505.07101) | 提出CS-UCCB算法，通过结合离线回归、代价置信集与上反事实置信界，在逐阶段约束的上下文赌博机中首次实现了不含eluder维度项的Õ(√T)遗憾界，同时以高概率保证每一轮的代价可行性。 |
| [^41] | [Reducing QPU Calls in Variational Quantum Algorithms via Classical Offloading of Gradient Computation](https://arxiv.org/abs/2503.16361) | 本文提出HELIA拟设及结合经典增强梯度方法与参数移位规则的混合训练方法，将梯度计算部分卸载到经典硬件，使VQE的逻辑量子评估次数平均减少高达60%，同时提升精度和成功率。 |
| [^42] | [Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms](https://arxiv.org/abs/2502.20115) | 本文提出一种多视图线性结构方程模型及相应算法，通过利用同一系统多个视图间的相关性，在不依赖非高斯性假设的情况下实现了因果发现的可辨识性，并成功应用于脑区间因果图的估计。 |
| [^43] | [Automatic Differentiation for Inference on Partially Observed Markov Process Models](https://arxiv.org/abs/2407.03085) | 该论文提出了一个统一的理论框架，将两种现有的自动微分粒子滤波方法纳入一类新算法，通过偏差-方差权衡显著降低均方误差，并据此开发了仅需可微分模拟器即可对部分可观测马尔可夫过程模型进行似然最大化的新算法。 |
| [^44] | [Optimal Clustering of Multi-layer Networks under Mixture Block Models](https://arxiv.org/abs/2311.15598) | 本文在混合多层随机块模型下建立了由Chernoff信息刻画的极小极大最优网络聚类错误率，并提出了一种结合张量初始化与似然精化的两阶段估计器，在极端稀疏条件下仍能达到该最优率，且可扩展至计数加权网络。 |

# 详细

[^1]: 从随机特征到全可训练网络的相变

    Phase Transition from Random Features to Fully Trainable Networks

    [https://arxiv.org/abs/2610.31663](https://arxiv.org/abs/2610.31663)

    本文为部分可训练的两层神经网络建立了统一的平方 L₂ 逼近误差界 O((κ_P log(m/δ)/m)^α)，揭示了从随机特征到全可训练网络之间存在相变现象：初始改善快速平滑随后收益递减，且初始化失配越大转变越剧烈。

    

    构建精确且计算上可处理的逼近空间是科学计算中的一项基本任务。部分可训练神经网络在固定随机特征与完全自适应表示之间架起了桥梁。本文量化了逼近界中的相变模式，并在训练误差中观察到类似的行为：初始阶段呈现快速而平滑的改善，随后收益递减，且当初始化失配越大时，转变越剧烈。在谱多项式衰减（指数 α>1）与杠杆值兼容性条件下，我们为宽度为 m、可训练比例为 P 的两层网络建立了平方 L₂ 逼近速率 O((κ_P log(m/δ)/m)^α)，其中 κ_P=(P/κ₁\*+(1-P)/κ₀\*)⁻¹。该界在再生核希尔伯特空间（RKHS）的单位球上一致成立，置信概率至少为 1-δ。常数 κ₀\* 与 κ…（摘要原文在此截断）

    arXiv:2610.31663v1 Announce Type: cross  Abstract: Constructing accurate and computationally tractable approximation spaces is a fundamental task in scientific computing. Partially trainable neural networks bridge fixed random features and fully adaptive representations. This paper quantifies a phase transition pattern in approximation bounds and observes analogous behavior in training errors: a rapid but smooth initial improvement followed by diminishing returns, with a sharper transition under greater initialization mismatch. Under polynomial spectral decay with exponent $\alpha>1$ and leverage compatibility, we establish a squared $L_2$ approximation rate $\mathcal{O}((\kappa_P\log(m/\delta)/m)^\alpha)$ for two-layer networks of width $m$ and trainable proportion $P$, where $\kappa_P=(P/\kappa_1^*+(1-P)/\kappa_0^*)^{-1}$. The bound holds uniformly over the unit ball of a reproducing kernel Hilbert space (RKHS) with probability at least $1-\delta$. The constants $\kappa_0^*$ and $\ka
    
[^2]: 通过选择性审计实现可靠的AI辅助决策

    Reliable AI-Assisted Decisions via Selective Auditing

    [https://arxiv.org/abs/2610.31662](https://arxiv.org/abs/2610.31662)

    提出了一个在线选择性审计框架，在最小化专家审计频率的同时，提供可证明的错误控制保证，使未经审计的AI错误比例始终低于用户设定的阈值。

    

    我们考虑对产生一系列决策的AI系统进行在线监控。在每一轮中，系统输出一个提议的决策以及一个用于估计该决策可信度的置信分数。决策和置信分数不必由同一个模型产生：例如，一个智能体可能提出决策，而另一个独立的验证智能体对其进行评估并输出正确性或安全性分数。基于这些信息，我们必须决定是自动接受AI决策，还是咨询专家（例如人类审计员），由其审查该提案并接受或纠正它。我们将此任务表述为一个在线约束优化问题，其目标是最小化专家审计的频率，同时保证未经审计的AI错误比例保持在用户定义的阈值以下。我们推导出一族具有可证明错误控制保证的算法，无需（摘要在此处被截断）

    arXiv:2610.31662v1 Announce Type: cross  Abstract: We consider online monitoring of an AI system that produces a sequence of decisions. At each round, the system outputs a proposed decision together with a confidence score estimating the trustworthiness of the decision. The decision and confidence score need not be produced by the same model: for example, one agent may propose a decision, while a separate verifier agent evaluates it and outputs a correctness or safety score. Based on this information, we must decide whether to accept the AI decision automatically or consult an expert, such as a human auditor, who reviews the proposal and either accepts or corrects it. We formulate this task as an online constrained optimization problem in which the goal is to minimize the frequency of expert auditing while guaranteeing that the fraction of AI errors that go unaudited stays below a user-defined threshold. We derive a family of algorithms with provable error-control guarantees with no as
    
[^3]: SympFormer：源自概率测度空间上阻尼哈密顿动力学的惯性注意力模块

    SympFormer: Inertial Attention Blocks from Damped Hamiltonian Dynamics on Spaces of Probability Measures

    [https://arxiv.org/abs/2610.31600](https://arxiv.org/abs/2610.31600)

    该论文提出 SympFormer，通过在概率密度空间上建立 Nesterov 型阻尼哈密顿动力学的几何框架（包括局部迁移率几何与商 Stein 几何），从动力学角度推导出具有加速特性的惯性注意力模块。

    

    注意力机制是现代大型语言模型的基石，但其动力学原理至今仍仅被部分理解。近期的观点将注意力模块解释为相互作用粒子系统，其平均场极限对应于配备 Wasserstein 型度量的 $\mathbb R^d$ 上概率密度空间中能量泛函的梯度流。我们拓展了这一视角，从这些密度空间上的惯性（Nesterov 型）阻尼哈密顿动力学中推导出加速注意力模块，并发展了使该构造严格化所需的几何与微分演算。特别地，我们为 Otto-Wasserstein 与 softmax-Wasserstein 情形引入了局部迁移率几何，并提出一种商 Stein 几何，解决了线性注意力核的退化问题，从而得到良定义的度量与微分演算。当 softmax 注意力允许梯度流形式时，我们给出了严格的数学基础。

    arXiv:2610.31600v1 Announce Type: cross  Abstract: The attention mechanism underpins modern large language models, yet their dynamics remain only partially understood. Recent perspectives interpret attention blocks as interacting particle systems, whose mean-field limits correspond to gradient flows of energy functionals on spaces of probability densities on $\mathbb R^d$ equipped with Wasserstein-type metrics. We extend this viewpoint by deriving accelerated attention blocks from inertial, Nesterov-type damped Hamiltonian dynamics on these density spaces, developing the geometry and differential calculus needed to make the construction rigorous. In particular, we introduce a local-mobility geometry for the Otto-Wasserstein and softmax-Wasserstein settings, and a quotient Stein geometry that resolves the degeneracy of the linear-attention kernel to yield a well-defined metric and differential calculus. We provide rigorous foundations when softmax attention admits a gradient-flow formul
    
[^4]: 何时跟随教师：基于优势的在策略蒸馏

    When to Follow the Teacher: Advantage-Based On-Policy Distillation

    [https://arxiv.org/abs/2610.31563](https://arxiv.org/abs/2610.31563)

    该论文提出基于优势的在策略蒸馏（AOPD），仅利用学生标准回滚轨迹及其终止奖励来估计教师动作在学生策略下的反事实优势，从而只强化真正对学生有益的教师动作，克服了传统在线策略蒸馏中“始终跟随教师”的局限。

    

    arXiv:2610.31563v1 宣布类型：新 摘要：在线策略蒸馏（OPD）通过直接在学生模型自身采样的轨迹上进行训练，其表现往往优于监督微调（SFT）等离线方法。然而，标准的在线策略蒸馏训练学生在轨迹的每个状态上都跟随教师，这隐含地假设教师的指导总是有益的。我们则区分教师所偏好的行为与真正能帮助当前学生取得成功的行为。我们提出了基于优势的在策略蒸馏（AOPD），该方法根据教师动作在学生策略下的反事实优势来为其加权：能够提升学生期望结果的教师动作会被强化，而不能提升的则被降权。直接计算这些优势需要在每一步尝试替代的教师动作，并从每个替代动作出发进行新的回滚采样。与之不同，我们证明AOPD目标可以仅通过标准的学生回滚轨迹及其终止奖励来估计。我们的估计……（摘要原文在此处截断）

    arXiv:2610.31563v1 Announce Type: new  Abstract: Online Policy Distillation (OPD) often outperforms offline methods such as Supervised Fine-Tuning (SFT) by training on trajectories sampled directly from the student. However, standard OPD trains the student to follow the teacher at every state of the trajectory, assuming that teacher guidance is always beneficial. We instead distinguish between what the teacher prefers and what actually helps the current student succeed. We introduce Advantage-Based On-Policy Distillation (AOPD), which weights teacher actions by their counterfactual advantage under the student. Teacher actions that improve the student's expected outcome are reinforced, while those that do not are downweighted.   Directly computing these advantages requires trying alternative teacher actions at each step and running new rollouts from each. Instead, we show that the AOPD objective can be estimated using only standard student rollouts and their terminal rewards. Our estima
    
[^5]: 面向困惑度感知黑盒攻击的扩散语言模型

    Diffusion Language Models for Perplexity-Aware Black-Box Attacks

    [https://arxiv.org/abs/2610.31558](https://arxiv.org/abs/2610.31558)

    该论文提出将连续潜空间攻击框架与连续扩散语言模型（dLLM）直接耦合，以学习到的语言生成分布取代人工随机映射，从而生成更自然、更难被防护机制检测且基于自然提示的黑盒攻击提示词。

    

    连续空间攻击通过在低维高斯潜空间中进行采样或优化，并将其通过随机映射和最近token查找解码为文本，从而找到能够触发黑盒大语言模型恶意行为（如过度思考）的提示词。由于这种随机映射并非从语言中学习得到，其生成的提示词可能非常不自然，因此容易通过输入防护机制被检测到，且其产生的风险估计也并未建立在自然提示的基础之上。在本文中，我们证明可以将同样的连续潜空间框架直接与从语言中学习得到的生成分布——连续扩散语言模型（dLLM）——相耦合，从而消除人为引入的随机映射，同时保留标准高斯潜空间形式以及下游的采样或优化流程。我们在两种设置下评估了该方法：使用 RareTrap 的基于采样的估计以及…

    arXiv:2610.31558v1 Announce Type: cross  Abstract: \emph{Continuous-space} attacks find prompts that trigger malicious behaviors, such as overthinking, in black-box LLMs by sampling or optimizing over a low-dimensional Gaussian latent space and decoding it to text through a \emph{random mapping} and nearest-token lookup. Because the random mapping is not learned from language, its prompts can be highly unnatural, and as such easily detectable through input guardrails, and the risk estimates they yield are not grounded in natural prompts. In this paper, we demonstrate that the same continuous latent-space framework can be coupled directly to a learned generative distribution over language, a continuous diffusion language model (dLLM), which removes the artificial random mapping while preserving the standard-Gaussian latent formulation and the downstream sampling or optimization procedure. We evaluate this approach under two settings: sampling-based estimation using \emph{RareTrap} and o
    
[^6]: 基于并行梯度提升的概率分布非参数自回归估计

    Nonparametric autoregressive estimation of probability distributions with parallel gradient boosting

    [https://arxiv.org/abs/2610.31524](https://arxiv.org/abs/2610.31524)

    该论文提出通过自回归因子分解将联合概率分布的估计转化为条件分布估计，并证明在满足次可加性的损失函数下，自回归估计器的收敛速度界可由条件分布估计器的收敛速度界导出，从而将联合分布估计的难度归结为自回归分解中最难因子的估计难度，并借助并行梯度提升实现高效估计。

    

    联合概率分布的非参数估计是统计学中的一个核心问题，在高维数据下这一问题以困难著称。类似的挑战也出现在条件分布估计中，由此催生了一批灵活的算法，它们通过利用维数低于环境空间的预测结构来应对“维数灾难”。在本文中，我们研究如何利用这类条件分布估计器，通过以“自回归”方式对联合分布进行因子分解来估计联合分布。我们证明，对于满足次可加性性质的损失函数，自回归估计器的收敛速度界可以从条件分布估计器的收敛速度界推导得出。这些界将估计联合分布的难度与估计自回归分解中最困难因子的难度联系起来。（摘要原文在此处截断）

    arXiv:2610.31524v1 Announce Type: cross  Abstract: Nonparametric estimation of joint probability distributions is a core problem in statistics, which is notoriously challenging with high-dimensional data. Such challenges also arise with conditional distribution estimation and have led to flexible algorithms that aim to tackle the ''curse of dimensionality'' by exploiting predictive structures of lower dimension than the ambient space. In this paper, we study the use of such conditional distribution estimators to estimate joint distributions by factorizing the latter in an ''autoregressive'' way. We show that for loss functions satisfying a subadditivity property, bounds on the convergence rate of an autoregressive estimator can be derived from bounds on the convergence rate of the conditional distribution estimators. These bounds relate the difficulty of estimating the joint distribution to that of estimating the most challenging factor in the autoregressive factorization. When the con
    
[^7]: SABLE：可扩展的仿射贝叶斯潜空间探索

    SABLE: Scalable Affine Bayesian Latent-space Exploration

    [https://arxiv.org/abs/2610.31495](https://arxiv.org/abs/2610.31495)

    SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。

    

    诸如工程设计、蛋白质等复杂对象往往难以优化，因为它们的设计空间呈现高维结构，且目标评估代价高昂。生成模型提供了连续的潜空间表示，但其潜空间可能具有数千至数万维，形成了潜空间维度 D 远超评估预算的情形。我们提出 SABLE，一种针对各向同性分布且预算有限的高维潜空间的贝叶斯优化方法。SABLE 使用贝叶斯仿射代理模型进行预测与采集，并将高阶结构视为代理模型的偏差。在 N 次观测之后，斜率更新的维度至多为 N-1，从而将每个查询分解为一个至多 N 维的跨度内分量（通过期望改进 EI 进行优化）和一个由标量范数支配的跨度外分量。我们表明……

    arXiv:2610.31495v1 Announce Type: new  Abstract: Complex objects such as engineering designs and proteins are often difficult to optimise because their design spaces exhibit high-dimensional structure and objective evaluations are expensive. Generative models provide continuous latent representations, but their latent spaces can have thousands to tens of thousands of dimensions, creating a regime where the latent dimension $D$ greatly exceeds the evaluation budget. We introduce SABLE, a Bayesian optimisation method for high-dimensional latent spaces with an isotropic distribution and limited budget. SABLE uses a Bayesian affine surrogate for prediction and acquisition, treating higher-order structure as surrogate discrepancy. After $N$ observations, the slope update has dimension at most $N-1$, yielding a decomposition of each query into an in-span component optimised by expected improvement (EI) in at most $N$ dimensions and an off-span component governed by a scalar norm. We show tha
    
[^8]: 岭可分模型中未调整朗之万算法的无维度区间：基于子空间鲁棒Wasserstein距离的分析

    Dimension-free regimes for unadjusted Langevin in ridge-separable models: A subspace robust Wasserstein analysis

    [https://arxiv.org/abs/2610.31431](https://arxiv.org/abs/2610.31431)

    该论文利用子空间鲁棒Wasserstein距离分析未调整朗之万算法在岭可分模型中的偏差，证明在过参数化和大样本区间下算法复杂度可摆脱对维度的依赖。

    

    未调整朗之万算法（ULA）被广泛应用于高维贝叶斯推断。基于Wasserstein-2距离的经典分析预测了随维度增长的渐近偏差和混合时间，但这一度量对于许多统计任务而言强于实际需要。我们转而分析子空间鲁棒Wasserstein距离下的偏差，该距离能够控制均值和任意低维边缘分布，因此非常适合实践中通常估计的观测量。对于由独立同分布数据构成的岭可分势函数，我们证明了高概率界，表明当数据点数量N相对维度d变化时，混合过程呈现钟形曲线。在过参数化区间中，有效维度是数据依赖扰动的秩N，使得复杂度依赖于N而非d。在大样本区间中，高斯扰动估计产生了无维度的复杂度……

    arXiv:2610.31431v1 Announce Type: cross  Abstract: The Unadjusted Langevin Algorithm (ULA) is widely used for high-dimensional Bayesian inference. Classical analyses in Wasserstein-2 distance predict an asymptotic bias and a mixing time that grow with dimension, but this metric is stronger than necessary for many statistical tasks. We instead analyze the bias in subspace robust Wasserstein distance, which controls the mean and arbitrary low-dimensional marginals and is therefore well suited to observables typically estimated in practice. For ridge-separable potentials formed with i.i.d. data, we prove high-probability bounds showing a bell-shaped mixing profile as the number of data points $N$ varies relative to the dimension $d$. In the overparameterized regime, the effective dimension is the rank $N$ of the data-dependent perturbation, leading to complexity depending on $N$ rather than $d$. In the large-sample regime, Gaussian-perturbation estimates yield dimension-free complexity on
    
[^9]: 基于条件期望局部半径的度量保形预测

    Metric Conformal Prediction via the Conditional Expected Local Radius

    [https://arxiv.org/abs/2610.31336](https://arxiv.org/abs/2610.31336)

    该论文提出条件期望局部半径（CELR）这一新度量，构建了适用于一般度量空间中非欧几里得响应的保形预测方法，结合度量分布随机森林实现高效计算，并从理论上证明了估计量一致性、渐近条件有效性及有限样本边际覆盖率。

    

    我们提出了条件期望局部半径（CELR），并开发了一种适用于一般度量空间中非欧几里得响应变量的新型保形预测方法。CELR 度量局部条件概率的集中程度，使预测集能够适应异质性和多峰性，而无需依赖密度估计或向量空间结构。其条件分布产生了一个在不同协变量取值之间具有可比性的非一致性得分，并避免了距离轮廓之间代价高昂的传输比较。我们进一步改造了度量分布随机森林，使其能够处理多变量协变量和非线性交互作用，同时单次森林拟合即可提供所有得分组成部分，通过消除重复的局部拟合来降低计算成本。我们在适当的正则性条件下建立了估计量的一致性和渐近条件有效性，并在可交换性假设下获得了有限样本边际覆盖率。

    arXiv:2610.31336v1 Announce Type: cross  Abstract: We introduce the conditional expected local radius (CELR) and develop a new conformal prediction method for non-Euclidean responses in general metric spaces. CELR measures local conditional probability concentration, allowing prediction sets to accommodate heterogeneity and multimodality without requiring densities or a vector space structure. Its conditional distribution yields a nonconformity score that is comparable across covariate values and avoids costly transport comparisons between distance profiles. We further adapt a metric distributional random forest that handles multivariate covariates and nonlinear interactions, while a single forest fit supplies all score components, reducing computation by eliminating repeated local fitting. We establish uniform consistency of the estimators and asymptotic conditional validity under suitable regularity conditions, together with finite-sample marginal coverage under exchangeability. Nume
    
[^10]: 度量空间中随机对象的深度学习：超越流形假设

    Deep Learning for Random Objects in Metric Spaces: Beyond Manifold Assumptions

    [https://arxiv.org/abs/2610.31328](https://arxiv.org/abs/2610.31328)

    该论文提出了一个直接在响应空间度量结构上运行的随机对象回归深度学习框架，无需流形或欧几里得嵌入假设，同时具备统计一致性与收敛速率保证。

    

    深度学习（DL）方法已在众多下游预测任务中显著提升了预测性能，为多个领域带来了新的发现。然而，尽管以概率分布、随机图、函数以及度量空间中其他随机对象形式呈现的现代生物医学结果日益重要，针对非欧几里得随机对象的原则性深度学习方法仍然相对匮乏。受新一代数字与临床生物标志物的启发，我们提出了一种用于度量空间中随机对象回归的深度学习框架。我们的方法直接在响应空间的度量结构上进行操作，避免了现有许多方法所施加的流形或欧几里得嵌入等限制性假设。同时，在既定的统计假设下，该方法保留了现代统计保证。我们为所提出的方法建立了统计一致性与收敛速率（原文摘要在此处截断）。

    arXiv:2610.31328v1 Announce Type: cross  Abstract: Deep learning (DL) methods have substantially improved predictive performance across a wide range of downstream prediction tasks, enabling new discoveries in multiple domains. However, principled DL methods for non-Euclidean random objects remain relatively scarce, despite the growing importance of modern biomedical outcomes taking the form of probability distributions, random graphs, functions, and other random objects in metric spaces. Motivated by a new generation of digital and clinical biomarkers, we propose a deep learning framework for regression with random objects in metric spaces. Our approach operates directly on the metric structure of the response space, avoiding restrictive manifold or Euclidean embedding assumptions imposed by many existing methods. At the same time, it retains modern statistical guarantees under the stated statistical assumptions. We establish statistical consistency and convergence rates for the propos
    
[^11]: 基于局部化AUC泛函的二元分类器局部判别

    Local Discrimination of Binary Classifiers via Localized AUC Functional

    [https://arxiv.org/abs/2610.31216](https://arxiv.org/abs/2610.31216)

    本文提出局部化AUC（LAUC）指标及LOCATOR方法，能够识别风险预测模型表现不佳的患者队列，从而克服传统AUC评估忽略性能异质性、可能导致临床决策失误的局限。

    

    风险预测模型通常使用AUC进行评估，这意味着在评估过程中我们对模型性能的异质性视而不见，这可能导致临床上的不良决策。在这项工作中，我们旨在通过识别风险预测模型可能无法准确预测的患者队列来改变这一现状。为了区分这些队列，我们使用了一种加权AUC的变体，我们称之为局部化AUC（Localized AUC，LAUC）。基于这一指标，我们构建了一种命名为LOCATOR的方法。该方法能够识别出风险预测模型不准确的患者队列。我们在真实世界数据和模拟实验中对该方法进行了测试。此外，我们还为局部化AUC在该应用中的使用建立了统计学理论基础。

    arXiv:2610.31216v1 Announce Type: new  Abstract: Risk prediction models are frequently evaluated with AUC, which means that during evaluation we are blind to performance heterogeneity, which could lead to bad decision-making in clinics. In this work, we aim to change that by identifying patient cohorts for whom the risk prediction model might not be an accurate predictor. To separate these cohorts, we use a version of the weighted AUC, which we call Localized AUC or LAUC. Using this metric, we build a method that we coin LOCATOR. This method is able to identify cohorts of patients for whom the risk prediction model is inaccurate. We test this method both in real-world data and in simulations. We also build a statistical background for the use of Localized AUC for our purposes.
    
[^12]: 预测不确定性可控的计算机模型校准

    Prediction Uncertainty Controlled Computer Model Calibration

    [https://arxiv.org/abs/2610.31163](https://arxiv.org/abs/2610.31163)

    该论文提出一个两层框架，将条件流匹配摊销后验推断与共形风险控制相结合，直接控制校准后模拟器的预测覆盖率，即使在校准参数弱识别或模型差异被误设的情况下也能保证名义的预测区间水平。

    

    计算机模型校准通常是为预测服务的，然而无论是贝叶斯Kennedy-O'Hagan (KOH)校准还是摊销式的基于模拟的推断，其设计目的都不是控制校准后模拟器的预测误差。当参数识别性较弱或数据-模型差异被错误设定时，由名义后验水平导出的模拟器输出区间可能远偏离名义覆盖水平。我们提出了一个直接针对预测覆盖不足（miscoverage）的两层框架。第一层中，一个条件流匹配模型学习校准参数的摊销式后验，其训练数据是由不同结构的高斯过程差异扰动后的模拟器输出。第二层中，共形风险控制将针对目标观测的后验样本组织成嵌套的候选集合，并选出其中最小的集合，使其固定秩的逐点预测区间在合成校准样本上达到规定的时间平均覆盖不足率。

    arXiv:2610.31163v1 Announce Type: cross  Abstract: Computer-model calibration is usually undertaken in service of prediction, yet neither Bayesian Kennedy-O'Hagan (KOH) calibration nor amortized simulation-based inference is built to control the predictive error of the calibrated simulator. When parameters are weakly identified or the data-model discrepancy is misspecified, the simulator-output bands induced by a nominal posterior level can be far from nominal. We propose a two-layer framework that targets predictive miscoverage directly. First, a conditional flow-matching model learns an amortized posterior for the calibration parameters, trained on simulator outputs perturbed with Gaussian-process discrepancies of varying structure. Second, conformal risk control arranges posterior draws for the target observation into nested candidate sets and selects the smallest one whose fixed-rank pointwise bands attain a prescribed time-averaged miscoverage on a synthetic calibration sample. Un
    
[^13]: 接下来会发生什么以及为什么：基于神经符号规则的可解释下一事件预测

    What Comes Next and Why: Interpretable Next-Event Prediction with Neuro-Symbolic Rules

    [https://arxiv.org/abs/2610.31114](https://arxiv.org/abs/2610.31114)

    提出了神经符号模型RULES及其与残差网络结合的RUNES框架，通过从数据中直接学习可解释的序列规则，在下一事件预测中同时实现了有竞争力的准确率和透明的可解释性。

    

    在现实世界的应用中，能够解释预测为何被做出往往与实现高准确率同等重要——甚至更为重要。然而，现有的序列事件预测方法通常以牺牲可解释性为代价来优先追求准确性。神经符号方法通过学习符号规则为提升可解释性提供了一个有前景的方向，但由于时序依赖性的存在，将其扩展到下一事件预测具有挑战性。我们提出了RULES，一个直接从数据中学习序列规则的神经符号模型，并将其与残差网络相结合，形成RUNES（事件序列规则网络）。RULES能够从合成数据中准确恢复真实规则，并在真实数据上发现紧凑且可解释的规则集。RUNES将这些规则以加法方式与残差网络结合，在提供对激活规则及残差作用机制的透明访问的同时，实现了具有竞争力的准确率。

    arXiv:2610.31114v1 Announce Type: new  Abstract: In real-world applications, being able to explain why a prediction is made is often as important - if not more so - than achieving high accuracy. However, existing methods for sequential event prediction typically prioritize accuracy at the expense of interpretability. Neuro-symbolic approaches offer a promising direction to improve interpretability by learning symbolic rules, but extending them to next-event prediction is challenging due to the presence of temporal dependencies. We introduce RULES, a neuro-symbolic model that learns sequential rules directly from data, and combine it with a residual network to form RUNES (Rule Network for Event Sequences). RULES accurately recovers ground truth rules from synthetic data and discovers compact, interpretable rule sets on real data. RUNES combines these rules additively with a residual network, achieving competitive accuracy while providing transparent access to active rules and how residu
    
[^14]: 具有思维链的Transformer通过训练发现动量加速方法

    Transformers with CoT Discover Heavy-Ball Acceleration Through Training

    [https://arxiv.org/abs/2610.31089](https://arxiv.org/abs/2610.31089)

    本文证明Transformer在梯度下降教师的初始监督下，通过真实回归权重的轨迹级监督训练，能够自发学会实现Heavy-Ball动量加速方法，超越对教师算法的单纯模仿并获得加速效果。

    

    现代Transformer通常通过生成和改进中间推理步骤来解决复杂任务。为了对思维链推理建立理论理解，Huang等人（2025）近期的一项重要研究将Transformer视为算法执行器，并证明当Transformer的CoT步骤由梯度下降教师监督时，它们可以学习实现梯度下降，用于线性回归中的上下文权重预测。在本文中，我们研究相同的上下文权重预测任务，但探究CoT推理能否超越对教师算法的单纯模仿。我们证明，在获得梯度下降教师的初始监督之后，进一步使用真实回归权重进行轨迹级监督会使模型实现Heavy-Ball（动量）方法，从而实现对原始教师的加速。值得注意的是，这一加速算法的出现无需来自任何Heavy-Ball教师的监督……

    arXiv:2610.31089v1 Announce Type: new  Abstract: Modern transformers often solve complex tasks by generating and refining intermediate reasoning steps. To develop a theoretical understanding of chain-of-thought (CoT) reasoning, a notable recent study by Huang et al. (2025) viewed transformers as algorithm executors, and showed that when transformers' CoT steps are supervised by a gradient-descent teacher, they can learn to implement gradient descent for in-context weight prediction in linear regression. In this paper, we study the same in-context weight prediction task but ask whether CoT reasoning can go beyond merely imitating the teacher algorithm. We show that, after initial supervision from a gradient-descent teacher, further trajectory-level supervision using the true regression weights leads the model to implement the Heavy-Ball method, thereby achieving acceleration over the original teacher. Notably, this accelerated algorithm emerges without supervision from any Heavy-Ball te
    
[^15]: 谱算法的随机投影：Sobolev范数下的最优速率

    Random Projections for Spectral Algorithms: Optimal Rates in Sobolev Norms

    [https://arxiv.org/abs/2610.30948](https://arxiv.org/abs/2610.30948)

    本文利用RKHS嵌入性质与算子相似性的精细分析，证明了基于随机投影（随机化草图与Nyström子采样）的谱算法在一般条件下于Sobolev范数中达到匹配极小极大下界的最优学习速率，且不牺牲统计效率。

    

    随机投影（RP）提供了一种在保持数据几何结构的同时降低计算与存储开销的有效方法。然而，现有研究主要集中于在设定良好且条件受限的情况下，针对采用随机投影的特定核正则化算法的最优泛化性能。在本文中，我们在不牺牲统计效率的前提下，对基于随机投影的谱算法在一般条件下的泛化性能进行了全面且改进的分析。通过利用再生核希尔伯特空间（RKHS）的嵌入性质以及对算子相似性的精细化分析，我们建立了Sobolev范数下的最优学习速率，其在对数因子范围内与极小极大下界相匹配。对于随机投影的两种常见实现方式——随机化草图和Nyström子采样（均匀采样或基于杠杆得分的采样），我们表明（摘要原文在此处截断）

    arXiv:2610.30948v1 Announce Type: cross  Abstract: Random projections (RP) offer an effective method for reducing computational and storage overhead while preserving the geometric structure of data. However, existing studies have primarily focused on the optimal generalization performance of specific kernel-regularized algorithms with RP in the well-specified setting and under restrictive conditions. In this paper, we provide a comprehensive and improved analysis of the generalization performance of RP-based spectral algorithms under general conditions \change{without sacrificing statistical efficiency}. By leveraging the embedding property of the reproducing kernel Hilbert space (RKHS) and a refined analysis of the operator similarity, we establish optimal learning rates in Sobolev norms that match the minimax lower bounds up to logarithmic factors. For two common implementations of RP, randomized sketches and Nystr\"{o}m sub-sampling (uniform or leverage-based), we show that the proj
    
[^16]: 连续深度的深度高斯过程

    Continuous-depth Deep Gaussian Processes

    [https://arxiv.org/abs/2610.30890](https://arxiv.org/abs/2610.30890)

    提出将深度高斯过程的深度视为由随机微分方程刻画的连续演化，并在深度索引的漂移函数族上施加高斯过程先验（CDGP 与 FlowDGP 两种实现），从而缓解传统离散深度 DGP 随深度增长而难以优化的问题。

    

    深度高斯过程（DGPs）是一类很有吸引力的贝叶斯回归模型，但其标准的离散深度参数化往往随深度增加而变得更难优化。我们认为，与其将深度视为一长串离散堆叠，不如将其视为一种潜在的连续演化。为此，我们提出了一种连续深度的建模形式：隐表示遵循一个随机微分方程，其漂移项是按深度索引的函数族，并在该函数族上配备基于高斯过程的先验。这将核心设计选择重新聚焦于施加在该函数族上的先验，而不是像神经常微分方程（Neural ODE）或神经随机微分方程（Neural SDE）模型那样采用纯神经网络的连续深度参数化。我们研究了这一思想的两种具体实现：CDGP，对状态-深度漂移直接施加高斯过程先验；以及 FlowDGP，一种通过流演化得到的先验，能以更低的训练成本捕捉更丰富的深度依赖关系。为论证这一重新设计的必要性，我们分析了单样本 DGP 目标如何会变得愈发敏感……（原文摘要在此处被截断）

    arXiv:2610.30890v1 Announce Type: cross  Abstract: Deep Gaussian processes (DGPs) are appealing Bayesian regression models, but their standard discrete-depth parameterization often becomes harder to optimize as depth grows. We argue that depth is better treated as a latent continuous evolution than as a long discrete stack. We therefore propose a continuous-depth formulation in which the latent representation follows a stochastic differential equation with a depth-indexed drift family equipped with a GP-based prior. This reframes the core design choice as the prior placed on that family, rather than adopting a purely neural continuous-depth parameterization as in neural ODE or neural SDE models. We study two instantiations of this idea: CDGP, a direct GP prior over the state--depth drift, and FlowDGP, a flow-evolved prior that captures richer depth dependence at lower training cost. To motivate the redesign, we analyze how the single-sample DGP objective can become increasingly sensiti
    
[^17]: xVGAE：一种面向可交换图的层次化变分图自编码器

    xVGAE: A Hierarchical Variational Graph Autoencoder for Exchangeable Graphs

    [https://arxiv.org/abs/2610.30815](https://arxiv.org/abs/2610.30815)

    本文提出 xVGAE，一种受 Aldous-Hoover 定理启发的层次化变分图自编码器，在生成质量与最先进模型相当的同时，将生成速度提升了几个数量级。

    

    图在许多应用中扮演着至关重要的角色，涵盖从药物发现到社交网络建模再到天文学等领域。近年来，图生成模型取得了长足进步，但我们发现，在最先进的模型中，存在一些相对简单的场景，这些模型在标准计算预算内难以收敛，或表现出极其缓慢的生成速度。我们提出了一种生成架构，其生成质量可与最先进的模型相媲美，且生成速度提高了几个数量级。我们的方法受 Aldous-Hoover 定理的启发，这是一个经典的表示定理，它刻画了任何对节点索引置换不变（即可交换）的图的概率分布。该定理表明，三个要素即可满足需求：一个图级潜变量、一组可交换的节点级潜变量，以及一个将图级和节点级变量进行映射的所谓 graphon 函数……

    arXiv:2610.30815v1 Announce Type: cross  Abstract: Graphs play a crucial role in many applications, ranging from drug discovery to social network modeling and astronomy. Generative models for graphs have substantially advanced in recent years, but we identify relatively simple scenarios where state-of-the-art models struggle to converge within a standard compute budget, or exhibit prohibitively slow generation times. We propose a generative architecture that achieves quality competitive with state-of-the-art models and is orders of magnitude faster at generation. Our approach is inspired by the Aldous-Hoover theorem, a classic representation theorem that characterizes any probability distribution of graphs that is invariant to permutations of node indices or exchangeable. This theorem establishes that three ingredients suffice: a graph-level latent variable, a set of exchangeable node-specific latent variables, and a so-called graphon function that maps graph- and node-level variables 
    
[^18]: 训练过程中宽度与深度极限的可交换性

    Width and Depth Limits Commute During Training

    [https://arxiv.org/abs/2610.30729](https://arxiv.org/abs/2610.30729)

    本文将残差网络无限宽与无限深极限的可交换性从初始化阶段首次扩展到训练过程，证明了在Depth-μP参数化与梯度裁剪下，线性残差网络在任意有限训练步数内，网络输出沿宽度-深度联合增长路径按 $L^2$ 收敛且两极限可交换。

    

    近期的研究工作证明了，当残差分支以 $1/\sqrt{深度}$ 进行缩放时，残差网络在初始化时的无限宽与无限深极限是可交换的。特别地，无论宽度与深度是依次趋于无穷还是同时趋于无穷，网络输出都收敛到相同的极限。我们将这一可交换性结果扩展到线性残差网络的训练过程中。在相同的 $1/\sqrt{深度}$ 残差分支缩放（Depth-$\mu$P）和 $\mu$P 宽度参数化下，我们证明：在使用梯度裁剪的条件下，对于任意固定数量的梯度下降步数，当宽度和深度同时趋于无穷时，网络输出沿着每条宽度-深度联合增长路径按 $L^2$ 收敛。因此，无限宽与无限深极限在任何有限训练范围内都是可交换的。在不使用梯度裁剪的情况下，相应的 $L^2$ 收敛一般不成立，但我们建立了较弱的依概率收敛。

    arXiv:2610.30729v1 Announce Type: new  Abstract: Recent work established that the infinite-width and infinite-depth limits of residual networks commute at initialization when residual branches are scaled by $1/\sqrt{depth}$. In particular, the network output converges to the same limit whether width and depth tend to infinity sequentially or jointly. We extend this commutativity result to training in linear residual networks. Under the same $1/\sqrt{depth}$ residual-branch scaling (Depth-$\mu$P) and the $\mu$P width parameterization, we prove that, for any fixed number of gradient-descent steps with gradient clipping, the network output converges in $L^2$ along every joint width--depth growth path as both width and depth tend to infinity. Consequently, the infinite-width and infinite-depth limits commute throughout any finite training horizon. Without gradient clipping, the corresponding $L^2$ convergence does not hold in general, but we establish the weaker convergence in probability.
    
[^19]: 基于局部特征空间精细化的残差校准

    Residual Calibration via Local Feature-Space Refinement

    [https://arxiv.org/abs/2610.30652](https://arxiv.org/abs/2610.30652)

    提出了局部自适应概率估计（LAPE），一种即插即用的事后残差校准框架，它以全局校准器输出为锚点，利用相邻校准样本的非参数局部证据对预测进行精细化，从而捕捉特征空间中局部区域的空间变化失准。

    

    深度神经网络往往难以提供可靠的置信度估计，这使得事后校准对于可靠的决策至关重要。现有的事后校准方法主要依赖于基于分数的变换，这类方法虽然能够改善边际校准，但往往无法捕捉学习到的特征空间中局部区域的、随空间变化的失准现象。为了解决这一问题，我们提出了局部自适应概率估计，这是一种事后残差校准框架。具体而言，LAPE 将全局校准器的输出视为稳定的锚点，并利用来自相邻校准样本的非参数局部证据来对单个预测进行精细化。作为一个即插即用的精细化模块，LAPE 能够有效利用特征空间几何信息，并且可以复用用于构建全局校准器的校准数据。此外，我们通过级联方式将 LAPE 扩展到多视图输入的场景中。

    arXiv:2610.30652v1 Announce Type: cross  Abstract: Deep neural networks often struggle to provide reliable confidence estimates, making post-hoc calibration essential for reliable decision-making. Existing post-hoc methods largely rely on score-based transformations, which improve marginal calibration but often fail to capture spatially varying miscalibration in local regions of the learned feature space. To address this gap, we propose Local Adaptive Probability Estimation (LAPE), a post-hoc residual calibration framework. Specifically, LAPE treats the global calibrator output as a stable anchor and refines the individual prediction by leveraging non-parametric local evidence from neighboring calibration samples. As a plug-and-play refinement module, LAPE effectively utilizes the information about feature-space geometry and can reuse the calibration data that are used to construct the global calibrator. Furthermore, we extend LAPE to the setting with multi-view inputs through Cascade 
    
[^20]: 持续演化的AI分析师实现终身有效的决策

    Lifetime-valid decisions by continually evolving AI analysts

    [https://arxiv.org/abs/2610.30562](https://arxiv.org/abs/2610.30562)

    提出“持续验证”方法，在部署前使用可随机化和扩展的零假设模拟数据评估每个AI分析师版本，并证明无论更新如何选择，部署第一类错误率超标版本的终身概率始终有界，从而保障持续演化的AI分析师所有后续决策的终身有效性。

    

    AI智能体越来越多地扮演分析师的角色，其决策通常可以归结为假设检验。经典检验针对固定程序控制第一类错误率，即拒绝真实零假设的概率。然而，AI分析师引入了内部随机性，并且会随时间不断演化。随着AI分析师的演化，批准第一类错误率超过目标水平的版本的风险会不断累积，而对评估反馈的自适应使用可能导致过拟合。我们提出了持续验证方法，在部署前使用零假设下的模拟数据对每个提出的AI分析师版本进行评估，其测试场景可以随时间随机化和扩展。我们证明，无论更新如何选择或提出的更新数量多少，部署第一类错误率超过允许水平的版本的终身概率始终保持有界。这保证了每个后续决策都能控制第一类错误率。

    arXiv:2610.30562v1 Announce Type: cross  Abstract: AI agents increasingly act as analysts whose decision can often be cast as hypothesis tests. Classical tests control the Type I error rate, the probability of rejecting a true null hypothesis, for fixed procedures. However, AI analysts introduce internal randomness and can evolve continuously over time. As AI analysts evolve, the risk of approving versions with Type I error rates above the target can accumulate, and adaptive use of evaluation feedback can lead to overfitting. We introduce continual validation, which evaluates each proposed AI analyst version before deployment using simulated data under the null hypothesis, with scenarios that can be randomized and expanded over time. We prove that, regardless of how updates are chosen or how many are proposed, the lifetime probability of ever deploying a version whose Type I error rate exceeds the allowed level remains bounded. This guarantees that every subsequent decision controls th
    
[^21]: 面向非平稳注意力的基于流的谱核学习

    Flow-based Spectral Kernel Learning for Nonstationary Attention

    [https://arxiv.org/abs/2610.30465](https://arxiv.org/abs/2610.30465)

    提出基于流的谱核注意力（FSKA），利用归一化流参数化非平稳的二元谱密度来学习灵活且可正则化的注意力核，同时支持移除查询-键投影的简化设计，实现参数效率更高的线性注意力架构。

    

    核化注意力为理解自注意力作为一种基于相似性的聚合机制提供了有用的视角，但现有方法通常依赖于固定或平稳的核，这限制了它们将相似性结构自适应地调整以应对复杂词元交互的能力。在本文中，我们提出了基于流的谱核注意力，这是一个用于学习富有表现力的非平稳注意力核的框架。FSKA 通过谱密度学习来构建核化注意力，并使用归一化流对所得的二元谱密度进行参数化。这种设计实现了灵活且可正则化的核学习，同时保持与可扩展的线性注意力计算的兼容性。我们进一步证明，富有表现力的核学习使得一种简化的注意力设计成为可能，即移除查询-键投影并使用固定的正交值投影，从而得到参数效率更高的架构。

    arXiv:2610.30465v1 Announce Type: cross  Abstract: Kernelized attention provides a useful perspective for understanding self-attention as a similarity-based aggregation mechanism, but existing approaches often rely on fixed or stationary kernels, limiting their ability to adapt the similarity structure to complex token interactions. In this paper, we propose flow-based spectral kernel attention (FSKA), a framework for learning expressive nonstationary attention kernels. FSKA formulates kernelized attention through spectral density learning and parameterizes the resulting bivariate spectral density with normalizing flows. This design enables flexible and regularizable kernel learning while remaining compatible with scalable linear-attention computation. We further show that expressive kernel learning enables a simplified attention design with removed query--key projections and fixed orthogonal value projections, resulting in a more parameter-efficient architecture. Controlled experiment
    
[^22]: 来自嘈杂、多样化来源的高价值训练数据

    High-Value Training Data from Noisy, Diverse Sources

    [https://arxiv.org/abs/2610.30421](https://arxiv.org/abs/2610.30421)

    提出一种基于元梯度的可扩展数据加权方法，通过学习可泛化、可迁移的参数化加权规则，为奖励建模等任务从嘈杂多样的数据源中筛选出高价值训练数据。

    

    训练最先进的模型越来越需要从大型、异构的数据集中学习，这些数据集中的样本在质量、领域相关性和监督来源上各不相同。数据加权——即为训练样本分配不同权重——提供了一种控制此类数据如何影响训练的通用方法，但在大规模下学习有效的权重仍然具有挑战性。我们开发了一种可扩展的数据权重学习方法：给定一个小型、高质量的目标数据集，我们在整个训练过程中使用元梯度（metagradients），直接针对目标域的性能高效地优化训练样本的权重。为了超越固定的数据集和模型进行扩展，我们学习参数化的加权规则，这些规则可以泛化到新样本和流式样本，并能从小模型迁移到规模大得多的模型。我们研究了奖励建模作为典型的应用场景，其中偏好数据可能是嘈杂的、对抗性的、合成的，或与目标域不匹配的。

    arXiv:2610.30421v1 Announce Type: cross  Abstract: Training state-of-the-art models increasingly requires learning from large, heterogeneous datasets whose examples vary in quality, domain relevance, and source of supervision. Dataweighting---assigning different weights to training examples---offers a general way to control how such data influences training, but learning effective weights at scale remains challenging. We develop a scalable approach to learning dataweights: given a small, high-quality target dataset, we use metagradients through the training process to efficiently optimize training-example weights directly for performance on the target domain. To scale beyond a fixed dataset and model, we learn parametric weighting rules that generalize to new and streaming examples and transfer from smaller models to substantially larger ones. We study reward modeling as a prototypical application, where preference data may be noisy, adversarial, synthetic, or mismatched with the targe
    
[^23]: 当重建还不够：基于学习多模态表示的双稳健反事实与干预中介分析

    When Reconstruction Is Not Enough: Doubly Robust Counterfactual and Interventional Mediation Analysis with Learned Multimodal Representations

    [https://arxiv.org/abs/2610.30220](https://arxiv.org/abs/2610.30220)

    该论文证明了仅基于协变量的训练目标（重建、对比或物理信息）无法保证多模态表示足以支持因果识别，给出了表示保持可识别性的条件，并提出了双稳健的反事实与干预中介分析方法。

    

    高维多模态健康记录的学习表示越来越多地被用于调整混杂因素，但它们通常被训练用于重建数据，而非保留因果识别所需的信息。我们研究了替代治疗下的反事实结果，包括表型特异性效应和数字孪生所提供的那种患者层面反事实风险概况，以及暴露诱导的中间混杂下的干预直接与间接效应。我们给出了表示保持可识别性的条件，并表明干预中介分析比反事实比较要求更高。我们证明了没有任何仅基于协变量的训练目标——无论是重建、对比学习还是物理信息驱动——能够证明一个表示是充分的；并且在线性结构模型中，一个表示可能仅损失任意小的重建准确度……（摘要原文在此处截断）

    arXiv:2610.30220v1 Announce Type: new  Abstract: Learned representations of high-dimensional multimodal health records are increasingly used to adjust for confounding, yet they are usually trained to reconstruct the data rather than preserve the information that causal identification requires. We study counterfactual outcomes under alternative treatments, including phenotype-specific effects and patient-level counterfactual risk profiles of the kind digital twins provide, and interventional direct and indirect effects under exposure-induced intermediate confounding. We give conditions under which a representation preserves identification and show that interventional mediation is more demanding than counterfactual comparison. We prove that no covariate-only training objective, whether reconstruction, contrastive, or physics-informed, can certify that a representation is sufficient, and that, in a linear structural model, a representation can lose arbitrarily little reconstruction accura
    
[^24]: AI心智融合：一种潜在互联协议

    AI Mindmeld: A Latent Interconnect Protocol

    [https://arxiv.org/abs/2610.30215](https://arxiv.org/abs/2610.30215)

    提出潜在互联协议（LIP），通过仅训练轻量级连接器将专家模型的潜在表示直接映射到LLM内部状态，实现无需联合训练、可扩展且可组合的模型间直接知识交换，突破了以文本等多模态数据传递信息的瓶颈。

    

    大语言模型（LLM）可以作为科学智能体，使用工具咨询专家模型并运行实验。然而，专家信息必须进入其上下文，通常以文本、图像、音频或视频的形式呈现：这在拥有丰富高维表示的模型之间构成了瓶颈。我们倡导直接的表示交换：一种“心智融合”，传递多种属性，包括LLM不会想到去请求的信息。该方法应可扩展至新的专家模型，灵活应对各种问题和任务，并能在无需联合训练的情况下，在独立连接的专家模型之间实现组合。潜在互联协议学习连接器，将专家模型的表示映射到LLM的内部状态中。仅训练连接器使得该方法在LLM不断增长和演进的情况下依然实用。我们将三个蛋白质专家模型和三个小分子专家模型连接到Gemma 3（4B和12B参数）以及Kimi K2.6（总参数1T，激活参数32B），并冻结所有预训练模型。独立训练的连接……

    arXiv:2610.30215v1 Announce Type: new  Abstract: LLMs can act as scientific agents using tools to consult specialists and run experiments. Yet specialist information must enter their context, generally as text, images, audio or video: a bottleneck between models with rich, high-dimensional representations. We advocate direct representation exchange: a mindmeld conveying many properties, including information an LLM would not think to request. It should be extensible to new specialists, flexible across questions and tasks, and compositional across independently connected specialists without joint training.   The Latent Interconnect Protocol (LIP) learns connectors mapping specialist representations into LLM internal states. Training only connectors keeps this practical as LLMs grow and evolve. We connect three protein and three small molecule specialists to Gemma 3 (4B and 12B parameters) and Kimi K2.6 (1T total, 32B active), freezing all pretrained models. Independently trained connect
    
[^25]: 基于广义矩匹配的差分隐私合成数据

    Differentially Private Synthetic Data via Generalized Moment Matching

    [https://arxiv.org/abs/2610.30196](https://arxiv.org/abs/2610.30196)

    提出了广义矩匹配这一通用框架，通过私有估计矩、拟合概率测度并采样合成记录来生成差分隐私合成数据，可应用于Wasserstein距离、最大均值差异等多种积分概率度量，并在固定维度下达到相差对数因子的极小极大最优误差率。

    

    我们研究了在指定的积分概率度量（IPM）下，针对[-1,1]^d中包含n条记录的任意数据库的(ε,δ)-差分隐私合成数据生成问题。我们提出了广义矩匹配方法，该方法私有地估计矩、拟合概率测度并采样合成记录。当矩误差能够控制目标IPM时，可应用一个一般性误差定理。该框架扩展了针对1-Wasserstein距离和光滑查询的切比雪夫矩匹配方法。我们将该框架应用于采用Matern核、Laplace核以及两层ReLU神经正切核的最大均值差异（MMD），还获得了切片Wasserstein-1以及由范数受限ReLU网络诱导的IPM的保证。在合成记录数量足够多的情况下，对于固定维度，这些误差率在相差对数因子的意义下是极小极大最优的。

    arXiv:2610.30196v1 Announce Type: cross  Abstract: We study $(\varepsilon,\delta)$-differentially private synthetic data for arbitrary databases of $n$ records in $[-1,1]^d$ under a specified integral probability metric (IPM). We introduce generalized moment matching, which estimates moments privately, fits a probability measure, and samples synthetic records. A general error theorem applies when moment error controls the target IPM. This framework extends Chebyshev moment matching for $1$-Wasserstein distance and smooth queries. We apply the framework to maximum mean discrepancy with Matern and Laplace kernels and a two-layer ReLU neural tangent kernel.We also obtain guarantees for sliced Wasserstein-$1$ and the IPM induced by norm-bounded ReLU networks. With sufficiently many synthetic records,these rates are minimax optimal up to logarithmic factors for fixed dimension.
    
[^26]: 为什么受约束的更新仍然会遗忘：持续监督微调的函数路径分析

    Why Constrained Updates Still Forget: A Functional-Path Analysis of Continual SFT

    [https://arxiv.org/abs/2610.30191](https://arxiv.org/abs/2610.30191)

    该论文提出“函数路径”分析框架，将持续微调导致的遗忘分解为受控方向、梯度陈旧性、探针失配与有限步效应四个因素，并通过匹配的QLoRA干预实验揭示约束放置在Adam之前还是之后对旧能力保留的影响远大于约束粒度本身。

    

    监督微调使语言模型适应专门任务，但可能会破坏其先前习得的行为。现有方法通过约束参数、梯度或重放目标来缓解这一问题，然而受保护对象、实际干预轨迹与保留输出之间的联系仍不清晰。我们发展了一种函数路径分析框架，将实际更新所引起的旧损失变化分解为受控方向、梯度陈旧性、探针失配和有限步效应四个组成部分。该框架给出了一个条件性的刷新预算，并识别出信息性重放所提供的恢复方向。通过匹配的QLoRA干预实验，我们将约束粒度与放置位置（位于Adam之前或之后）分离开来：在所研究的代码标签审计中，放置位置强烈影响旧能力的保留效果，而粒度的交互作用则取决于端点状态。检查点生成实验将端点恢复与更高的采样路径准确率区分开来，匹配的重放方案……

    arXiv:2610.30191v1 Announce Type: cross  Abstract: Supervised fine-tuning adapts language models to specialized tasks but can disrupt previously acquired behavior. Existing approaches constrain parameters, gradients, or replay objectives, yet the connection between the protected object, the intervening trajectory, and retained outputs remains unclear. We develop a functional-path account that decomposes old-loss changes along realized updates into the controlled direction, gradient staleness, probe mismatch, and finite-step effects. The account yields a conditional refresh budget and identifies the restoring direction supplied by informative replay. Matched QLoRA interventions separate constraint granularity from placement before or after Adam: placement strongly affects retention across the studied code-label audits, whereas the granularity interaction depends on the endpoint. Checkpointed generation separates endpoint recovery from higher sampled-path accuracy, and matched replay sch
    
[^27]: WhatIf-Bench：面向干预感知时间序列预测的基准测试

    WhatIf-Bench: Benchmarking Intervention-Aware Time Series Forecasting

    [https://arxiv.org/abs/2610.26582](https://arxiv.org/abs/2610.26582)

    该论文提出了WhatIf-Bench，一个基于数据中心、电网、交通和供水系统机理仿真器构建的半合成基准，用于系统评估时间序列基础模型利用协变量进行干预感知情景预测的能力。

    

    近年来，时间序列基础模型（TSFMs）可以在上下文中接收协变量，将零样本预测扩展到由外部因素驱动的问题上。这使得情景分析成为可能：即预测目标变量对那些将协变量设定为特定未来值的事件的响应。然而，这些模型能否可靠地做到这一点仍是一个悬而未决的问题。我们提出了WhatIf-Bench，这是一个用于衡量预测模型利用协变量来预测目标对未来事件响应能力的基准。WhatIf-Bench是半合成的：时间序列来自数据中心、电网、道路交通和供水网络的机理仿真器，其协变量与目标之间的关系在每个系统中受到控制且具有实际依据。每个任务都呈现一段对协变量进行干预的事件历史，然后要求模型预测一个未来事件，该事件要么重放已见过的协变量值（Replay），要么将其设定为未见过的值（Novel）。通过改变协变量数量、历史事件数量和事件长度，共产生了37,665个任务。

    arXiv:2610.26582v2 Announce Type: replace-cross  Abstract: Recent time series foundation models (TSFMs) can consume covariates in-context, extending zero-shot forecasting to problems driven by external factors. This enables scenario analysis: forecasting how a target responds to events that set covariates to chosen future values. Whether they can do this reliably is an open question. We introduce WhatIf-Bench, a benchmark measuring how well forecasters use covariates to forecast a target's response to future events. WhatIf-Bench is semi-synthetic: time series come from mechanistic simulators of data centers, power grids, road traffic, and water networks, with controlled covariate-target relationships grounded in each system. Each task presents a history of events intervening on covariates, then asks to forecast a future event that either replays seen values (Replay) or sets them to unseen ones (Novel). Varying the number of covariates, historical events, and event length yields 37,665 
    
[^28]: 用于条件分布灵活估计的并行梯度提升方法

    Parallel gradient boosting for flexible estimation of conditional distributions

    [https://arxiv.org/abs/2607.13550](https://arxiv.org/abs/2607.13550)

    该论文提出了一种称为“并行梯度提升”的梯度提升算法改进方法，通过在每次迭代中仅训练一个基模型即可同时预测多个目标（如整个条件分布），克服了传统提升方法在多输出回归任务中计算开销大、且现有高效变体受限于特定损失函数和决策树基学习器的局限。

    

    提升方法是标准分类和回归任务中最成功的学习技术之一。近年来，其向多输出预测问题的扩展得到了越来越多的应用。其中一个应用是对整个条件分布而非单一泛函进行预测，这类问题通常可以表述为多输出回归问题，例如多分位数回归。使用经典的提升方法实现来解决此类问题在计算上极具挑战性，因为通常在每次迭代中需要为每个目标分别训练一个基模型。为了加速训练，已有研究提出了更高效的提升方法变体，但它们往往局限于特定的损失函数和特定的基学习器类别（通常是决策树）。在这项工作中，我们研究了梯度提升算法的一种改进版本，我们称之为并行梯度提升，旨在克服上述所有限制……

    arXiv:2607.13550v2 Announce Type: replace-cross  Abstract: Boosting is one of the most successful learning techniques for standard classification and regression tasks. Its extension to multi-output prediction problems has found an increasing number of applications in recent years. Among them is the prediction of entire conditional distributions rather than single functionals, which can often be framed as a multi-output regression problem, for example multiple quantile regression. Addressing such problems with classical implementations of boosting is computationally challenging, because usually one base model is trained for each target at every iteration. More efficient variants of boosting have been proposed to speed up training, but they tend to be tied to specific loss functions and classes of base learners, usually decision trees. In this work, we study a modification of the gradient boosting algorithm, which we call parallel gradient boosting, designed to circumvent all these limit
    
[^29]: 嵌套采样：批判性与全面的理论指南

    Nested Sampling: A Critical and Comprehensive Theoretical Guide

    [https://arxiv.org/abs/2606.17916](https://arxiv.org/abs/2606.17916)

    本文全面而批判性地阐述了嵌套采样的理论推导与实践挑战，指出其成功的关键在于从受似然约束的先验中高效采样，并系统梳理了它与重要性采样及垂直蒙特卡洛之间的联系。

    

    嵌套采样（NS）技术因其在高效探索高似然区域方面的能力而受到广泛关注，尤其在宇宙学和天文学领域——这一特性类似于一种隐式的似然优化，是其成功的基石。尽管NS的完整理论推导较为复杂且涉及多个近似，但其核心挑战在于从受似然约束的先验中进行采样，这对NS的性能至关重要。本工作对NS的推导进行了全面而详细的阐述，澄清了其理论基础与实践挑战。我们对NS的流程进行了透彻的描述，着重强调其优势与潜在的局限性。文中还详细讨论了NS与重要性采样（IS）以及垂直蒙特卡洛之间的联系。通过上述工作，本文旨在加深对该方法的理解，并促进未来方法的发展。

    arXiv:2606.17916v2 Announce Type: replace-cross  Abstract: The nested sampling (NS) technique has gained widespread attention, particularly in cosmology and astronomy, due to its ability to efficiently explore high-likelihood regions - a feature akin to an implicit likelihood optimization that underlies its success. While the full theoretical derivation of NS is complex and involves several approximations, the central challenge lies in sampling from the likelihood-constrained priors, which is crucial for its performance. This work provides a comprehensive and detailed exposition of NS derivation, clarifying both its theoretical foundations and practical challenges. We provide a thorough description of the NS procedure, emphasizing both its strengths and potential limitations. The connections between NS, importance sampling (IS), and vertical Monte Carlo are also discussed in detail. In doing so, this work seeks to deepen understanding of the method and to foster the development of futu
    
[^30]: 评估稀疏地震计数的神经概率预报：哪些结论能经受严格评估的检验

    Evaluating Neural Probabilistic Forecasts of Sparse Earthquake Counts: What Survives a Careful Evaluation

    [https://arxiv.org/abs/2605.21437](https://arxiv.org/abs/2605.21437)

    对天山-帕米尔地区周地震计数神经概率预报的严格评估表明：泊松类预报（含ETAS）严重低估单周发生3次以上地震的概率（低估8.3倍），引入一个离散参数即可解决；而神经网络相对线性模型的优势在为每个网格加入截距后便不再显著。

    

    我们重新审视了一个用于天山与帕米尔地区3×3度网格上每周地震计数的神经负二项模型，并探究其哪些结论能够经受住严格评估的检验。在完整性震级为4.5的条件下，144周的测试期共包含2448个网格-周单元；其中六个网格-周（分布在五个不同的周内）发生了三次及以上的地震事件。有两个结论得以保留。首先，泊松类预报（包括时间性ETAS模型）显著低估了出现这种“繁忙周”的概率：在实际发生六次事件的情况下，ETAS模型的预期仅为0.72次，低估达8.3倍（p = 0.00009，精确检验）。仅需引入一个离散（过散布）参数即可消除这一偏差（带离散参数的ETAS预期为4.45次，p = 0.58）；而在周内模拟余震链虽能再现该尾部特征，却会使总计数超出约四分之一。其次，一旦线性模型为每个网格单元设置了各自的截距项，神经网络相对于线性模型的优势便不复存在：剩余增益仅为偏差的3.4%，不具统计显著性（p = 0.57）。

    arXiv:2605.21437v2 Announce Type: replace-cross  Abstract: We revisit a neural negative binomial model for weekly earthquake counts on a 3 x 3 degree grid over the Tian Shan and Pamir, and ask which of its conclusions survive a careful evaluation. At the completeness magnitude of 4.5, the 144-week test period holds 2448 cell-weeks; six cell-weeks, in five distinct weeks, have three or more events.   Two conclusions survive. First, Poisson forecasts, temporal ETAS included, understate the probability of such a busy week: ETAS expects 0.72 where six occurred, a shortfall by a factor of 8.3 (p = 0.00009, exact test). One dispersion parameter removes it (ETAS with dispersion expects 4.45, p = 0.58); simulating aftershock chains within the week also reproduces the tail but overshoots the total count by about a quarter. Second, the network's advantage over a linear model disappears once that model has an intercept per cell: the remaining gain is 3.4% of deviance, not significant (p = 0.57 ov
    
[^31]: 基于观测数据的政策学习：HIV/HCV合并感染患者丙型肝炎治疗的案例研究

    Policy Learning with Observational Data: The Case of Hepatitis C Treatment for HIV/HCV Co-Infected Patients

    [https://arxiv.org/abs/2605.16593](https://arxiv.org/abs/2605.16593)

    本文提出了一种在较弱假设下从观测数据推导多行动政策规则的方法，通过加权K-means算法估计条件平均处理效应并结合标准决策树生成可行政策规则，且能处理治疗依从性不完美的情况，并成功应用于HIV/HCV合并感染患者的丙肝治疗方案选择。

    

    决策者经常必须从有限的一组备选方案中做出单一选择——例如，医生选择治疗方案、投资者选择投资组合的风险水平，或法官决定量刑。为了改善结果，政策制定者通常会发布政策规则或指南来指导此类选择。在本文中，作者展示了如何在多行动框架下、在对异质性抽样人群底层结构假设相对较弱的情况下，从观测数据中普遍推导政策规则。在假设结果模型在每个同质子群内被正确设定的条件下，通过加权K-means算法一致地估计条件平均处理效应（CATE）。随后，作者使用标准决策树来实现可行的政策规则，并同时考虑了完全依从治疗与不完全依从治疗两种情形。该方法被应用于HIV/HCV合并感染患者的丙型肝炎（HCV）治疗方案选择。

    arXiv:2605.16593v2 Announce Type: replace-cross  Abstract: Decision-makers frequently must choose a single action from a finite set of alternatives -- for example, physicians selecting a treatment, investors choosing a portfolio risk level, or judges determining sentences. To improve outcomes, policymakers often issue policy rules or guidelines to inform such choices. In this paper, I show how to generally derive policy rules from observational data in a multi-action framework under relatively weak assumptions about the underlying structure of the heterogeneous sampled population. Conditional average treatment effects (CATEs) are consistently estimated via a weighted K-means algorithm, assuming the outcome model is correctly specified within each homogeneous subgroup. I then use standard decision trees to implement feasible policy rules, allowing for both perfect and imperfect adherence to treatment. The methodology is applied to treatment options for Hepatitis C (HCV) among patients c
    
[^32]: K-均值作为归一化RBF分配的零温度极限：收敛速率、轨迹与基于Entmax的精确恢复

    K-Means as the Zero-Temperature Limit of Normalized RBF Assignments: Rates, Trajectories, and Exact Recovery with Entmax

    [https://arxiv.org/abs/2603.04625](https://arxiv.org/abs/2603.04625)

    本文严格证明了K-均值聚类是归一化RBF分配在零温度下的Γ-极限，从目标函数、优化方法、单次更新和轨迹收敛等多个层面定量刻画了软RBF分配与硬K-均值之间的联系，并利用Entmax实现了精确恢复。

    

    K-均值通常被视为一种独立于梯度学习的离散过程，而归一化径向基函数（RBF）层则是光滑的。我们精确阐明了K-均值作为此类层零温度极限的含义。通过引入香农熵惩罚将K-均值分配松弛到单纯形上，可以得到归一化高斯RBF责任度以及一个简化的目标函数F_σ，其中0≤J−F_σ≤2nσ²log k，因此F_σ Γ-收敛于K-均值的失真J。我们区分了四个层次的论述：在目标函数层面，最小值与最小化子收敛，且在临界温度σ²=λ_max之上，所有位于均值的质心构成一个严格局部极小值点；在优化层面，交替最小化是关于F_σ的预条件梯度方法，其硬性极限即为Lloyd算法；在单次更新层面，在间隔Δ_min>0的条件下，软质心映射逼近（原文此处截断）

    arXiv:2603.04625v2 Announce Type: replace  Abstract: K-Means is usually treated as a discrete procedure outside gradient-based learning, whereas normalized radial basis function (RBF) layers are smooth. We make precise the sense in which K-Means is the zero-temperature limit of such a layer. Relaxing the K-Means assignments to the simplex with a Shannon entropy penalty yields the normalized Gaussian RBF responsibilities and a reduced objective $F_\sigma$ with $0\le J-F_\sigma\le 2n\sigma^2\log k$, so $F_\sigma$ $\Gamma$-converges to the K-Means distortion $J$. We keep four levels of statement apart. For the objective, minimal values and minimizers converge, and above the critical temperature $\sigma^2=\lambda_{\max}$ all centroids at the mean form a strict local minimizer. For optimization, alternating minimization is a preconditioned gradient method on $F_\sigma$ whose hard limit is Lloyd's algorithm. For one update, under a margin $\Delta_{\min}>0$, the soft centroid map approaches L
    
[^33]: KL正则化多臂老虎机的近最优后悔界

    Near-Optimal Regret for KL-Regularized Multi-Armed Bandits

    [https://arxiv.org/abs/2603.02155](https://arxiv.org/abs/2603.02155)

    该论文通过新颖的分层论证对KL-UCB算法进行精细分析，首次给出了对臂数 $K$ 呈线性依赖的高概率后悔上界 $\tilde{O}(\eta K\log^2T)$，并以首个非常数下界 $\Omega(\eta K\log T)$ 证明了其近紧性。

    

    近年来的研究表明，带有KL正则化目标的强化学习能够获得更快的收敛速度或对数级后悔，这与未正则化设定下经典的 $\sqrt{T}$ 型后悔形成鲜明对比。然而，即使是针对多臂老虎机（MAB）这一特殊情形，关于KL正则化目标的在线学习的统计效率仍然远未被完全刻画。我们通过对KL-UCB算法进行精细分析，并采用一种新颖的分层论证方法，解决了多臂老虎机中的这一问题，得到了 $\tilde{O}(\eta K\log^2T)$ 的上界：这是首个对臂数 $K$ 呈线性依赖的高概率后悔界。其中，$T$ 表示时间范围，$K$ 表示臂数，$\eta^{-1}$ 表示正则化强度，$\tilde{O}$ 隐藏了除涉及 $\log T$ 之外的所有对数因子。我们分析的近紧性由首个非常数下界 $\Omega(\eta K \log T)$ 所证实……

    arXiv:2603.02155v2 Announce Type: replace  Abstract: Recent studies have shown that reinforcement learning with KL-regularized objectives can enjoy faster rates of convergence or logarithmic regret, in contrast to the classical $\sqrt{T}$-type regret in the unregularized setting. However, the statistical efficiency of online learning with respect to KL-regularized objectives remains far from completely characterized, even when specialized to multi-armed bandits (MABs). We address this problem for MABs via a sharp analysis of KL-UCB using a novel peeling argument, which yields a $\tilde{O}(\eta K\log^2T)$ upper bound: the first high-probability regret bound with linear dependence on $K$. Here, $T$ is the time horizon, $K$ is the number of arms, $\eta^{-1}$ is the regularization intensity, and $\tilde{O}$ hides all logarithmic factors except those involving $\log T$. The near-tightness of our analysis is certified by the first non-constant lower bound $\Omega(\eta K \log T)$, which follo
    
[^34]: 多指标模型中多轮随机梯度下降的高维学习动力学

    High-dimensional learning dynamics of multi-pass Stochastic Gradient Descent in multi-index models

    [https://arxiv.org/abs/2601.21093](https://arxiv.org/abs/2601.21093)

    本文在样本量与维度成比例的高维区间内，首次对非凸多指标模型上的多遍随机梯度下降动力学建立了渐近精确的刻画，证明其收敛于一种新颖的动力学平均场连续时间极限，且适用于任意次线性批量大小及相应的临界学习率缩放。

    

    随机梯度下降（SGD）的学习动力学被认为对训练模型的泛化性能有着重要影响。尽管在线（单遍）SGD如何精确优化总体风险的动力学已在广泛的高维问题中得到理解，但对于用于经验风险最小化的多遍SGD，类似的理解在很大程度上仍然缺失。在这项工作中，我们在样本量n与数据维度d成比例增长的高维区间内，为非凸多指标模型的多遍SGD动力学建立了渐近精确的刻画。我们考虑任意次线性批量大小κ ≍ n^α（其中α ∈ [0,1)）以及相应的“临界”学习率缩放，并证明SGD的逐坐标动力学具有一种新颖的连续时间缩放极限，该极限由一组动力学平均场（方程给出）……

    arXiv:2601.21093v3 Announce Type: replace-cross  Abstract: The learning dynamics of Stochastic Gradient Descent (SGD) are believed to play an important role in influencing the generalization properties of trained models. While the precise dynamics of how online (one-pass) SGD optimizes the population risk are now understood in a broad range of high-dimensional problems, a comparable understanding of multi-pass SGD for empirical risk minimization is still largely lacking.   In this work, we develop an asymptotically exact characterization of the dynamics of multi-pass SGD for a non-convex multi-index model, in the high-dimensional regime where sample size $n$ and data dimension $d$ increase proportionally. We consider any sub-linear batch size $\kappa \asymp n^\alpha$ with $\alpha \in [0,1)$ and a commensurate ``critical'' scaling of the learning rate, and show that the coordinate-wise dynamics of SGD admit a novel continuous-time scaling limit, given by a system of dynamical mean-field
    
[^35]: 慢性肾病中面向多群体的惩罚化公平回归

    Penalized Fair Regression for Multiple Groups in Chronic Kidney Disease

    [https://arxiv.org/abs/2512.17340](https://arxiv.org/abs/2512.17340)

    提出一种面向多群体的惩罚化公平回归通用框架，通过不公平性惩罚与新颖的自动惩罚权重选择评分函数，实现高效求解，并在慢性肾病研究中取得优于现有方法的公平性-准确性平衡。

    

    公平回归方法有潜力缓解医疗保健领域的社会偏见问题，但当多个群体同时遭受此类偏见时，关于惩罚化公平回归的研究却十分匮乏。我们提出了一个通用的回归框架，通过对多个群体施加不公平性惩罚来填补这一空白。我们的方法在二元结果上以真阳性率差异惩罚的形式进行了展示，并可通过归约为代价敏感分类问题来高效实现。此外，我们还引入了新颖的评分函数，用于自动选择惩罚权重。我们在仿真实验中对这些惩罚化公平回归方法进行了实证研究，结果显示它们达到了超越现有对比方法的公平性-准确性边界。最后，我们将这些方法应用于一项全国多中心慢性肾病初级保健研究，以开发针对终末期肾病的公平分类器。在该研究中，我们发现子（原文摘要在此处被截断）……

    arXiv:2512.17340v3 Announce Type: replace-cross  Abstract: Fair regression methods have the potential to mitigate societal bias concerns in health care, but there has been little work on penalized fair regression when multiple groups experience such bias. We propose a general regression framework that addresses this gap with unfairness penalties for multiple groups. Our approach is demonstrated for binary outcomes with true positive rate disparity penalties. It can be efficiently implemented through reduction to a cost-sensitive classification problem. We additionally introduce novel score functions for automatically selecting penalty weights. Our penalized fair regression methods are empirically studied in simulations, where they achieve a fairness-accuracy frontier beyond that of existing comparison methods. Finally, we apply these methods to a national multi-site primary care study of chronic kidney disease to develop a fair classifier for end-stage renal disease. There we find subs
    
[^36]: 基于稀疏优化更新的随机匹配老虎机

    Stochastic Matching Bandits with Rare Optimization Updates

    [https://arxiv.org/abs/2509.04194](https://arxiv.org/abs/2509.04194)

    本文提出了MNL选择模型下随机匹配老虎机的批处理算法，通过将昂贵的匹配优化更新次数限制在Θ(log log T)次，在大幅降低计算开销的同时保持了遗憾最优性能。

    

    我们提出了一个在多项Logit（MNL）选择模型下的随机匹配老虎机框架。在该设定中，一侧的 $N$ 个智能体被分配到另一侧的 $K$ 个臂上，每个臂根据未知的偏好从其分配池中随机选择一个智能体，并在时间范围 $T$ 内产生相应的奖励。目标是通过最大化成功匹配的累积收益来最小化遗憾。一种朴素的方法需要在每一轮都求解一个NP难的组合优化问题，导致极其高昂的计算成本。为应对这一挑战，我们提出了批处理算法，将匹配分配的更新次数在整个时间范围内策略性地限制在 $\Theta(\log\log T)$ 次。通过仅在极小比例的轮次中调用昂贵的组合优化，我们的算法大幅降低了总体计算开销，同时仍然实现……

    arXiv:2509.04194v3 Announce Type: replace-cross  Abstract: We introduce a bandit framework for stochastic matching under the multinomial logit (MNL) choice model. In this setting, $N$ agents on one side are assigned to $K$ arms on the other side, where each arm stochastically selects an agent from its assigned pool according to unknown preferences and yields a corresponding reward over a horizon $T$. The objective is to minimize regret by maximizing the cumulative revenue from successful matches. A naive approach requires solving an NP-hard combinatorial optimization problem at every round, resulting in a prohibitive computational cost. To address this challenge, we propose batched algorithms that strategically limit the number of times matching assignments are updated to $\Theta(\log\log T)$ over the entire horizon. By invoking expensive combinatorial optimization only on a vanishing fraction of rounds, our algorithms substantially reduce overall computational overhead while still ach
    
[^37]: 粗化数据下的多重稳健保形风险控制

    Multiply Robust Conformal Risk Control with Coarsened Data

    [https://arxiv.org/abs/2508.15489](https://arxiv.org/abs/2508.15489)

    该论文提出了一个结合半参数理论与保形方法的通用框架，通过RSP和RSCP两种互补程序，将双重稳健估计方法推广到一般粗化数据情形，实现了数据缺失或粗化下具有多重稳健性的保形预测与风险控制。

    

    保形预测（CP）为预测集提供了正式的覆盖率保证，但大多数现有方法假设训练数据和校准数据是完全观测的。然而在许多实际应用中，结果变量、协变量或两者都可能出现缺失或粗化。我们通过将半参数理论与保形方法相结合，开发了一个针对粗化数据预测推断的通用框架。我们引入了两种互补的程序：(i) 稳健分裂预测（RSP）通过构造基于有效影响函数的估计方程并求解其经验类似物，将Yang等人[2024]的双重稳健估计方程方法从经典的缺失结果问题推广到更一般的情形，从而在一般粗化模式下产生多重稳健的预测集。(ii) 稳健分裂保形预测（RSCP）则将半参数理论与保形风险控制相结合。由于有效影响函数是……（原文摘要在此处截断）

    arXiv:2508.15489v2 Announce Type: replace-cross  Abstract: Conformal Prediction (CP) provides prediction sets with formal coverage guarantees, but most existing methods assume fully observed training and calibration data. In many applications, however, outcomes, covariates, or both may be missing or coarsened. We develop a general framework for predictive inference with coarsened data by combining semiparametric theory with conformal methods.   We introduce two complementary procedures. $(i)$ Robust Split Prediction (RSP) generalizes the doubly robust estimating-equation approach of Yang et al. [2024] beyond the classical missing outcome problem by constructing an efficient influence function based estimating equation and solving its empirical analogue. This yields multiply robust prediction sets under general coarsening patterns. $(ii)$ Robust Split Conformal Prediction (RSCP) instead combines semiparametric theory with conformal risk control. Because efficient influence functions are
    
[^38]: 面向多体哈密顿量的吉布斯态元学习及其在量子玻尔兹曼机中的应用

    Meta-learning of Gibbs states for many-body Hamiltonians with applications to Quantum Boltzmann Machines

    [https://arxiv.org/abs/2507.16373](https://arxiv.org/abs/2507.16373)

    该论文提出了两种元学习算法，通过对有限训练参数值集合进行联合优化，在NISQ设备上高效制备参数化多体哈密顿量的吉布斯态，并可应用于量子玻尔兹曼机。

    

    量子吉布斯态的制备是量子计算中的一项基本挑战，对于从开放量子系统建模到量子机器学习等诸多应用至关重要。基于Cervera-Lierta等人提出的元变分量子本征求解器（Meta-VQE）框架和问题驱动的拟设设计，我们提出了两种元学习算法：元变分量子热化器和神经网络元变分量子热化器（NN-Meta VQT），用于在含噪声中等规模量子（NISQ）设备上高效制备参数化哈密顿量的热态。Meta-VQT采用全量子拟设，而NN-Meta VQT则结合了量子-经典混合架构。两者均通过对有限训练参数值集合进行联合优化，在采样参数域内的留出参数值处制备近似吉布斯态。我们在最高若干比特规模的横向场伊辛模型（TFIM）上验证了我们的方法……

    arXiv:2507.16373v3 Announce Type: replace-cross  Abstract: The preparation of quantum Gibbs states is a fundamental challenge in quantum computing, essential for applications ranging from modeling open quantum systems to quantum machine learning. Building on the Meta-Variational Quantum Eigensolver framework proposed by Cervera-Lierta et al. and a problem-driven ansatz design, we introduce two meta-learning algorithms: Meta-Variational Quantum Thermalizer (Meta-VQT) and Neural Network Meta-VQT (NN-Meta VQT) for efficient thermal state preparation of parametrized Hamiltonians on Noisy Intermediate-Scale Quantum (NISQ) devices. Meta-VQT utilizes a fully quantum ansatz, while NN Meta-VQT integrates a quantum--classical hybrid architecture. Both use collective optimization over a finite set of training parameter values to prepare approximate Gibbs states at held-out parameter values within the sampled parameter domain. We validate our methods on Transverse Field Ising Models (TFIM) with up
    
[^39]: 基于扩散的假设检验与变点检测

    Diffusion-Based Hypothesis Testing and Change-Point Detection

    [https://arxiv.org/abs/2506.16089](https://arxiv.org/abs/2506.16089)

    本文将基于分数的假设检验与变点检测方法推广为基于扩散的版本，通过数值优化权重矩阵提升检测能力，并从理论上量化了算法性能及其达到最优的条件。

    

    基于分数的方法近年来在建模与生成领域日益流行。已有研究构建了利用分数函数进行假设检验和变点检测的方法，但这些方法通常不如基于似然的方法强大。近期的工作考虑通过将分数函数与矩阵值函数或权重矩阵相乘进行变换，将基于分数的Fisher散度推广为扩散散度。在本文中，我们将基于分数的假设检验方法和变点检测停止规则扩展为基于扩散的对应方法。此外，我们从理论上量化了这些基于扩散的算法的性能，并研究了可实现最优性能的场景。我们提出了一种数值优化权重矩阵的方法，并通过数值模拟展示了基于扩散算法的优势。

    arXiv:2506.16089v2 Announce Type: replace-cross  Abstract: Score-based methods have recently seen increasing popularity in modeling and generation. Methods have been constructed to perform hypothesis testing and change-point detection with score functions, but these methods are in general not as powerful as their likelihood-based peers. Recent works consider generalizing the score-based Fisher divergence into a diffusion-divergence by transforming score functions via multiplication with a matrix-valued function or a weight matrix. In this paper, we extend the score-based hypothesis test and change-point detection stopping rule into their diffusion-based analogs. Additionally, we theoretically quantify the performance of these diffusion-based algorithms and study scenarios where optimal performance is achievable. We propose a method of numerically optimizing the weight matrix and present numerical simulations to illustrate the advantages of diffusion-based algorithms.
    
[^40]: 基于离线回归实现逐阶段约束上下文赌博机的无Eluder维度遗憾

    Eluder-Dimension-Free Regret for Stage-Wise Constrained Contextual Bandits via Offline Regression

    [https://arxiv.org/abs/2505.07101](https://arxiv.org/abs/2505.07101)

    提出CS-UCCB算法，通过结合离线回归、代价置信集与上反事实置信界，在逐阶段约束的上下文赌博机中首次实现了不含eluder维度项的Õ(√T)遗憾界，同时以高概率保证每一轮的代价可行性。

    

    具有逐阶段约束的上下文赌博机对一类决策问题进行建模，这类决策必须在学习其结果的同时控制风险。我们研究仅能观测到所选动作的奖励与代价的情形，其中奖励函数属于一般函数类，代价函数属于一类凸的非线性函数。我们提出了CS-UCCB算法，该算法结合了离线回归、代价置信集以及上反事实置信界。每次回归拟合都定义了一个有效的代价置信集，而由覆盖熵导出的几何结构则控制着沿采样决策的累积代价不确定性。在两个函数类均具有多对数级覆盖熵且离线估计误差为多对数级的条件下，该有限动作算法以高概率保证每一轮的可行性，并实现Õ(√T)的遗憾界，且不含eluder维度项。在显式的奖励设计条件下，我们还将分析扩展到无限动作的情形……

    arXiv:2505.07101v4 Announce Type: replace-cross  Abstract: Contextual bandits with stage-wise constraints model decisions that must control risk while learning from their outcomes. We study this problem when only the selected action's reward and cost are observed, with a general reward class and a convex class of nonlinear cost functions. We propose CS-UCCB, which combines offline regression, cost confidence sets, and upper counterfactual confidence bounds. Each regression fit defines a valid cost confidence set, while a geometry derived from covering entropy controls cumulative cost uncertainty along the sampled decisions. Under polylogarithmic covering entropy for both classes and polylogarithmic offline estimation error, the finite-action algorithm guarantees, with high probability, feasibility at every round and $\widetilde{\mathcal O}(\sqrt T)$ regret without an eluder-dimension term. We also extend the analysis to infinite actions under an explicit reward-design condition and com
    
[^41]: 通过梯度计算的经典卸载减少变分量子算法中的QPU调用次数

    Reducing QPU Calls in Variational Quantum Algorithms via Classical Offloading of Gradient Computation

    [https://arxiv.org/abs/2503.16361](https://arxiv.org/abs/2503.16361)

    本文提出HELIA拟设及结合经典增强梯度方法与参数移位规则的混合训练方法，将梯度计算部分卸载到经典硬件，使VQE的逻辑量子评估次数平均减少高达60%，同时提升精度和成功率。

    

    变分量子算法（VQAs）是利用近期量子计算机的一种有前景的方法，但其可训练性和量子资源成本仍然是主要挑战。本文通过硬件高效且动态李代数支持的拟设（HELIA）以及两种将经典增强梯度方法与基于量子的参数移位规则（PSR）相结合的训练方法来解决这些挑战。该方法将梯度估计和训练过程分布在经典和量子硬件上。我们使用变分量子本征求解器（VQE）评估了6到18量子比特哈密顿量的基态估计问题，以及最多12量子比特哈密顿量的量子相分类问题。对于VQE，我们的方法提高了精度和成功率，与相同拟设的Full-PSR训练相比，估计的逻辑量子评估次数平均减少高达60%。对于分类任务，最大的改进……

    arXiv:2503.16361v3 Announce Type: replace-cross  Abstract: Variational quantum algorithms (VQAs) are a promising approach for leveraging near-term quantum computers, but trainability and quantum-resource costs remain major challenges. Here we address these challenges with the \textbf{H}ardware \textbf{E}fficient and dynamical \textbf{LI}e algebra supported \textbf{A}nsatz (HELIA) and two training methods combining the classical-enhanced \g method with the quantum-based Parameter-Shift Rule (PSR). This distributes gradient estimation and training across classical and quantum hardware. We evaluate ground-state estimation of 6- to 18-qubit Hamiltonians using the Variational Quantum Eigensolver (VQE) and quantum phase classification for Hamiltonians up to 12 qubits. For VQE, our methods improve accuracy and success rates, with a mean reduction in estimated logical quantum evaluations of up to $60\%$ relative to Full-PSR training of the same ansatz. For classification, the largest improveme
    
[^42]: 无需非高斯性假设的多视图因果发现：可辨识性与算法

    Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms

    [https://arxiv.org/abs/2502.20115](https://arxiv.org/abs/2502.20115)

    本文提出一种多视图线性结构方程模型及相应算法，通过利用同一系统多个视图间的相关性，在不依赖非高斯性假设的情况下实现了因果发现的可辨识性，并成功应用于脑区间因果图的估计。

    

    因果发现是一个困难的问题，通常依赖于对数据生成模型的强假设，例如非高斯性。在实践中，许多现代应用提供了同一系统的多个相关视图，而这一点在因果发现领域很少被考虑。在此，我们利用这种多视图结构，在弱假设条件下实现因果发现。我们提出了一个多视图线性结构方程模型（SEM），该模型通过交替利用视图间的相关性，扩展了著名的非高斯扰动框架。我们证明了该模型在无环SEM情形下的可辨识性。随后，受单视图算法（DirectLiNGAM、PairwiseLiNGAM和ICA-LiNGAM）的启发，我们提出了几种多视图因果发现算法。新方法通过仿真实验和神经影像数据应用得到了验证，在这些应用中，它们能够估计脑区之间的因果图。

    arXiv:2502.20115v4 Announce Type: replace  Abstract: Causal discovery is a difficult problem that typically relies on strong assumptions on the data-generating model, such as non-Gaussianity. In practice, many modern applications provide multiple related views of the same system, which has rarely been considered for causal discovery. Here, we leverage this multi-view structure to achieve causal discovery with weak assumptions. We propose a multi-view linear Structural Equation Model (SEM) that extends the well-known framework of non-Gaussian disturbances by alternatively leveraging correlation over views. We prove the identifiability of the model for acyclic SEMs. Subsequently, we propose several multi-view causal discovery algorithms, inspired by single-view algorithms (DirectLiNGAM, PairwiseLiNGAM, and ICA-LiNGAM). The new methods are validated through simulations and applications on neuroimaging data, where they enable the estimation of causal graphs between brain regions.
    
[^43]: 用于部分可观测马尔可夫过程模型推断的自动微分方法

    Automatic Differentiation for Inference on Partially Observed Markov Process Models

    [https://arxiv.org/abs/2407.03085](https://arxiv.org/abs/2407.03085)

    该论文提出了一个统一的理论框架，将两种现有的自动微分粒子滤波方法纳入一类新算法，通过偏差-方差权衡显著降低均方误差，并据此开发了仅需可微分模拟器即可对部分可观测马尔可夫过程模型进行似然最大化的新算法。

    

    自动微分（AD）推动了机器学习领域的最新进展，包括深度神经网络和哈密顿马尔可夫链蒙特卡罗方法。然而，部分可观测的非线性随机动态系统一直难以应用AD技术，原因在于广泛使用的粒子滤波算法所得到的似然函数估计值作为模型参数的函数是不连续的。为解决这一问题，我们构建了一个理论框架，将两种现有的AD粒子滤波方法嵌入到一个新的算法类别之中。这一新算法类别允许进行偏差-方差权衡，从而使均方误差显著低于现有算法。这使我们能够开发出与AD梯度估计的蒙特卡罗特性相适应的似然最大化算法。我们的算法只需要潜动态系统具备可微分模拟器即可；相比之下，以往大多数针对AD似然最大化的方法……

    arXiv:2407.03085v2 Announce Type: replace-cross  Abstract: Automatic differentiation (AD) has driven recent advances in machine learning, including deep neural networks and Hamiltonian Markov Chain Monte Carlo methods. Partially observed nonlinear stochastic dynamical systems have proved resistant to AD techniques because widely used particle filter algorithms yield an estimated likelihood function that is discontinuous as a function of the model parameters. To resolve this, we create a theoretical framework that embeds two existing AD particle filter methods within a new class of algorithms. This new class permits a bias-variance tradeoff and hence a mean squared error substantially lower than the existing algorithms. This allows us to develop likelihood maximization algorithms suited to the Monte Carlo properties of the AD gradient estimate. Our algorithms require only a differentiable simulator for the latent dynamic system; by contrast, most previous approaches to AD likelihood max
    
[^44]: 混合块模型下多层网络的最优聚类

    Optimal Clustering of Multi-layer Networks under Mixture Block Models

    [https://arxiv.org/abs/2311.15598](https://arxiv.org/abs/2311.15598)

    本文在混合多层随机块模型下建立了由Chernoff信息刻画的极小极大最优网络聚类错误率，并提出了一种结合张量初始化与似然精化的两阶段估计器，在极端稀疏条件下仍能达到该最优率，且可扩展至计数加权网络。

    

    我们研究多层网络的聚类问题，其中每个观测值都是在同一组顶点上测量的无向网络。在混合多层随机块模型（MMSBM）下，每个网络由两个随机块模型之一生成，目标是恢复潜在的网络标签。我们建立了极小极大最优的网络聚类错误率，该错误率呈指数形式，并由各分量网络边概率分布之间的Chernoff信息刻画。我们提出了一种新颖的两阶段估计器，它结合了基于张量的初始化、节点与样本分割以及基于似然的精化方法。该精化步骤利用估计得到的节点社区结构，在极端网络稀疏性的条件下也能以高概率达到极小极大最优率。我们进一步通过混合多层泊松块模型，将似然精化框架扩展到计数加权网络。

    arXiv:2311.15598v2 Announce Type: replace-cross  Abstract: We study clustering for multi-layer networks, where each observation is an undirected network measured on the same set of vertices. Under the mixture multi-layer stochastic block model (MMSBM), each network is generated from one of two stochastic block models, and the goal is to recover the latent network labels. We establish the minimax optimal network clustering error rate, which takes an exponential form and is characterized by the Chernoff information between the edge probability distributions of the component networks. We propose a novel two-stage estimator that combines tensor-based initialization, node and sample splitting, and a likelihood-based refinement. The refinement step exploits estimated node communities and achieves the minimax rate with high probability under extreme network sparsity. We further extend the likelihood-refinement framework to count-weighted networks through a mixture multi-layer Poisson block mo
    

