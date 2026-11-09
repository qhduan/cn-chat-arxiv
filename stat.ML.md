# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quasipolynomial-Time Planning for Decodable POMDPs](https://arxiv.org/abs/2611.09778) | 本文为比可观测性严格更弱的"ρ-可解码"POMDP提出了拟多项式时间的近似规划算法，证明该条件被ℓ₁与ℓ₂可观测性共同蕴含，且最优可解码常数可在多项式时间内计算。 |
| [^2] | [Where Should Learning Live? Regret Analysis for Text, Parameter, and Joint Adaptation](https://arxiv.org/abs/2611.09701) | 该论文将部署阶段对提示与参数适配器的重复选择形式化为在线学习问题，把超额损失分解为候选族内的选择遗憾、候选族限制差距和部署历史影响三部分，并证明在损失不受选择器随机性影响、历史影响有限且限制切换次数的条件下可实现次线性期望遗憾。 |
| [^3] | [A Generalized Bayesian Approach to Clustering with Dimension Reduction](https://arxiv.org/abs/2611.09698) | 该论文提出将自编码器与广义贝叶斯聚类划分联合学习，在学到的低维表示空间中评估聚类损失，从而在不依赖完整生成模型的情况下实现高维数据的概率化聚类并量化分配不确定性。 |
| [^4] | [Online Causal Configuration for Networked Systems via Doubly Robust Steady-State Learning](https://arxiv.org/abs/2611.09665) | 提出CAUSAL-UCB算法，通过干扰感知的裁剪双重稳健稳态学习方法，在测量探索会改变网络化系统稳态的困难条件下正确识别并在线优化配置的目标因果价值，并给出了推理误差与遗憾的理论下界。 |
| [^5] | [Kernel-Based Learning of Analytic Operators and Parametric PDEs](https://arxiv.org/abs/2611.09612) | 该论文建立了基于核方法从有限样本学习解析算子的理论，证明合适的解析核可实现指数级收敛速率，并将其应用于参数化椭圆偏微分方程的解算子学习，给出了分离采样误差与有限元离散化误差的误差界及网格-样本耦合准则。 |
| [^6] | [Locally Adaptive Conformal Prediction for Polynomial Chaos Expansions](https://arxiv.org/abs/2611.09579) | 该论文提出了针对多项式混沌展开的局部自适应jackknife+保形预测框架，通过在嵌套留一残差上训练辅助模型进行归一化，在固定基展开情形下仅需单次模型拟合即可获得具有覆盖保证的自适应预测区间。 |
| [^7] | [Marginal-Nonuniform Multiclass Learning](https://arxiv.org/abs/2611.09543) | 该论文提出了介于经典PAC学习与通用学习之间的边缘非一致学习框架，并通过新提出的非中心化DS-eluder维度完整刻画了最优学习速率，建立了指数、线性与任意慢速率的三分性结论。 |
| [^8] | [k-Volume: a Unified Active Learning Strategy](https://arxiv.org/abs/2611.09388) | 该论文提出了一种新型主动学习策略k-Volume，通过包含样本k近邻的最小球体体积来选择待标注样本，统一融合了密度、多样性和不确定性三大类主动学习策略的优势，理论上被证明渐近等价于均匀采样，并在各种分类器和实验设置中展现出卓越的稳定性。 |
| [^9] | [Calibrating your confidence requires thinking of all the ways you could be wrong](https://arxiv.org/abs/2611.08997) | 本论文提出一种协方差与Fisher信息矩阵零空间对齐的退化高斯分布，在保持贝叶斯不确定性量化质量的同时大幅降低计算开销，实现了高质量且可扩展的贝叶斯深度学习。 |
| [^10] | [Enforcing Symmetries on Pre-trained ML Models for Physics](https://arxiv.org/abs/2611.08992) | 本文将事后群平均方法扩展应用于预训练的物理机器学习模型以强制实施对称性，并针对非紧致群提出了三种替代方案（包括Weyl幺正技巧和数据依赖的正交投影），且不改变模型架构或需要重新训练。 |
| [^11] | [Ordered Policy Optimization](https://arxiv.org/abs/2611.08936) | 提出有序策略优化（OPO），通过在每个小批量中对重要性比率排序、识别影响最大的极端比率并为其分配自适应的逐样本裁剪半径，取代PPO的固定裁剪范围，从而更可靠地控制策略更新。 |
| [^12] | [Geometry-Aware Score-Repellent Monte Carlo](https://arxiv.org/abs/2611.08909) | 本文提出几何感知的分数排斥蒙特卡罗方法（GA-SRMC），用迹归一化的逆 Fisher 矩阵替代单一标量排斥强度，并从理论上证明其能最优地同时应对各向异性能量地形中的陡峭与平坦方向，从而改进 MCMC 采样。 |
| [^13] | [Reweighting for Transfer Classification: Exact Risk Conversion and Minimax Adaptation](https://arxiv.org/abs/2611.08876) | 本文推导出迁移分类中由混合超额风险到目标超额合页风险的精确转换曲线（由密度加权局部投票决定），并在倾斜边界类上建立了将源采样不确定性与关系不确定性分离的极小极大分类率。 |
| [^14] | [Beyond the Lower-Star Filtration: Alternative Filtration Methods for Persistent Homology](https://arxiv.org/abs/2611.08833) | 本文针对下星过滤中高维单纯形与其边界面同时进入导致拓扑特征被立即填充而无法体现在持久图中的局限性，提出了ELSF、LpF和SMF三种替代过滤方法，以更完整地捕捉拓扑特征。 |
| [^15] | [High-Dimensional Off-Policy Evaluation without Sample Splitting using an Onsager Correction](https://arxiv.org/abs/2611.08794) | 本文提出一种经Onsager修正且以目标策略加权的准则，仅利用训练观测数据即可估计全数据奖励模型在目标策略下的样本外风险，从而在高维离线策略评估中无需样本分割即可实现全数据梯度下降轨迹上的早停，克服了留出数据选择可能次优的问题。 |
| [^16] | [Blind Noise Equalization for Large Data Matrices under General Variance Profiles](https://arxiv.org/abs/2611.08784) | 该论文提出了BRINE方法，通过迭代重缩放行与列来均衡未知噪声方差，从而在异方差噪声下恢复大型数据矩阵噪声谱的Marchenko-Pastur定律并放大信号成分，且无需预先了解信号或噪声结构。 |
| [^17] | [On Lipschitz Explosion in Deep Neural Networks with Normalization: Consequences for Optimization and Adversarial Robustness](https://arxiv.org/abs/2611.08775) | 本文证明了带归一化层的深度神经网络（如批归一化网络）的输入和参数Lipschitz常数即使在每个层的范数都受控的情况下，仍会随深度呈指数增长，这种逐层不可见的全局不稳定性对优化保证和对抗鲁棒性具有重要影响。 |
| [^18] | [MEDALLION: Manifold Embedding Distillation and Attributions for Low-dimensional Latent InterpretatiON](https://arxiv.org/abs/2611.08767) | MEDALLION提出了一种结合知识蒸馏与神经特征归因的新框架，能够解释任意降维方法（如t-SNE、UMAP）生成的低维嵌入中哪些原始特征起关键作用。 |
| [^19] | [pESTO: Iterative p-greedy kernel regression with risk-driven Early STOpping](https://arxiv.org/abs/2611.08555) | 该论文提出pESTO方法，通过显式分离逼近误差与噪声放大，并采用Stein无偏风险估计（SURE）作为均方误差的代理，为p-贪婪核回归中节点数量的选择提供了一个有原则且可计算的早停准则。 |
| [^20] | [ELPAC: Endpoint-Anchored Latent Progression with Stage-Varying Multimodal Coordination](https://arxiv.org/abs/2611.08541) | 提出ELPAC贝叶斯框架，通过端点锚定的潜在进展建模与阶段变化的多模态残差协调，从部分观测的横断面数据中联合推断老年队列受试者的潜在阶段、进展亚型及跨模态相互作用。 |
| [^21] | [Drift-admissibility via Optimal Transport](https://arxiv.org/abs/2611.08505) | 该论文提出基于最优传输的结构感知诊断方法，利用 KL 投影间隙与可扩展的 Bregman–Dykstra 算法检验时间序列是否符合指定可预测漂移类，并能区分二阶统计量无法分辨的过程（如同 Hurst 指数的分数布朗运动与 BSS 过程）。 |
| [^22] | [Conformal Prediction for Time-Dependent PDEs](https://arxiv.org/abs/2611.08409) | 本文证明了时间依赖偏微分方程的共形预测覆盖保证在本质上难以实现——函数空间中相近时刻的分布可相互奇异，离散化中全变差距离随网格分辨率指数饱和——并给出了为确保覆盖而对空间离散化施加的必要限制。 |
| [^23] | [Support Vector Machines Directly Minimise KL Divergence](https://arxiv.org/abs/2611.05557) | 本文证明支持向量分类与回归的对偶目标等价于两个高斯测度之间的带约束KL散度最小化，从而为标准核机器建立了统一的信息论解释。 |
| [^24] | [Descending Everest, Part II: Oracle Complexity of Minimizing a Sum of Convex $(L_0,L_1)$-Smooth Functions](https://arxiv.org/abs/2611.04171) | 提出了结合加速近端方法、方差缩减与有界启发式选点策略的 YETI 方法，并给出了最小化凸 $(L_0,L_1)$-光滑函数之和的预言机复杂度的匹配上界与下界（相差对数因子）。 |
| [^25] | [Machine Learning Beyond Statistical Consistency for Hybrid AI](https://arxiv.org/abs/2611.01914) | 该论文提出机器学习的新定义，证明机器学会一个真实概率不仅需要统计一致性（真值条件），还需要机器能在形式系统内部正确推导出自身估计的收敛性（保证条件），从而将学习的标准置于统计一致性之上，为混合人工智能奠定形式化基础。 |
| [^26] | [Simple Extensions of Single-Objective Acquisition Functions and Hedge Strategies for Multi-Objective Bayesian Optimization](https://arxiv.org/abs/2609.31940) | 该论文提出一个基于超体积的简单框架，能够将常用的单目标采集函数与对冲策略直接扩展到多目标贝叶斯优化中，仅需对现有流程做极小修改，无需专门的多目标设计。 |
| [^27] | [Guidance for Prior Change via Density Ratio Estimation](https://arxiv.org/abs/2608.21729) | 提出一种基于密度比估计的无偏测试时引导框架，通过学习得分引导项有效解耦推断过程与先验训练，避免了先验依赖和系统性偏差。 |
| [^28] | [Model-Agnostic FDR Control via Group Gaussian Mirror and Permutation SHAP](https://arxiv.org/abs/2608.00989) | 该论文提出了一个面向分组和序列模型的模型无关FDR控制框架，通过组高斯镜像统计量与置换SHAP块级重要性得分相结合，在低维和高维设定下实现了对原始特征块级别的误发现率控制。 |
| [^29] | [Closed-form fractional radial links for elliptical Mahalanobis discriminant analysis](https://arxiv.org/abs/2607.06089) | 本文针对共享径向生成函数的椭圆分布两类分类问题，直接从类内半径分布推导出闭式分数次幂形式的径向连接函数，并证明了该连接的可识别性、估计量的√n一致性与渐近正态性，以及分类器的渐近贝叶斯最优性。 |
| [^30] | [SGD Provably Prioritizes a Shortcut Spurious Feature in the XOR Model](https://arxiv.org/abs/2606.30444) | 本研究首次对两层ReLU神经网络中虚假特征学习进行端到端理论分析，证明SGD会以指数级速度优先学习捷径虚假特征，且较强的虚假分量会抑制信号特征的学习。 |
| [^31] | [Action-BED: Task-Driven Bayesian Experimental Design with Singly Intractable Objectives](https://arxiv.org/abs/2606.23662) | 该论文提出ACTION-BED框架，通过以下游动作的期望未来损失重新表述贝叶斯实验设计，将其转化为可用随机梯度对设计策略和动作策略进行联合优化的单重不可解目标，从而无需显式的后验或边缘似然估计。 |
| [^32] | [Fine-Tuned Language Models Exhibit Improved Information Conveyance](https://arxiv.org/abs/2605.30844) | 本文提出树冠熵（CE*）这一新度量，从树视角联合刻画语言模型输出长度与生成内容的不确定性，并发现微调后的语言模型具有更优的信息传递能力。 |
| [^33] | [Instance-Optimal Estimation with Multiple LLM Judges on a Budget](https://arxiv.org/abs/2605.23362) | 该论文将固定预算下如何在多个成本与可靠性各异的 LLM 评审器及不同难度的评估实例间分配查询这一形式化为异方差多评审估计问题，并分析了逆方差加权估计器、推导出使误差界最小化的最优分配方案。 |
| [^34] | [Optimizing Computational-Statistical Runtime for Wasserstein Distance Estimation](https://arxiv.org/abs/2605.20122) | 该论文提出“采样-素描-求解”范式，通过对样本进行规则笛卡尔网格素描，在计算-统计运行时间框架下显著优化了光滑测度之间Wasserstein距离估计的运行时间。 |
| [^35] | [Smooth Piecewise Cutting for Neural Operator to Handle Discontinuities and Sharp Transitions](https://arxiv.org/abs/2605.19823) | 提出 Cut-DeepONet 两阶段训练框架，通过将求解域切割为平滑子区域、并将不连续性表示为高维空间中的边界，使神经算子能够高效处理偏微分方程解中的不连续性与尖锐过渡。 |
| [^36] | [Self-Distillation is Optimal Among Spectral Shrinkage Estimators](https://arxiv.org/abs/2605.17778) | 该论文证明在高维线性回归中，当协方差矩阵具有 $s$ 个尖峰时，$s$ 步自蒸馏在谱收缩估计器类中达到最优预测，且证明 $s$ 步是实现最优性的必要条件。 |
| [^37] | [NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning](https://arxiv.org/abs/2605.16757) | 该论文提出 NeuroMAS，将多智能体语言系统视为以 LLM 智能体为节点、文本信号为边的可训练类神经网络架构，通过强化学习让节点自主学会通信、分工与协调，从而将多智能体设计从手工工作流工程转变为可扩展的架构设计。 |
| [^38] | [Training-Free Generative Sampling via Moment-Matched Score Smoothing](https://arxiv.org/abs/2605.14276) | 提出了一种免训练的交互粒子采样器MM-SOLD，通过在整个采样轨迹中强制匹配目标矩，从理论上证明其收敛到与训练GMM均值和协方差一致的吉布斯-玻尔兹曼分布，即使数据秩亏也成立。 |
| [^39] | [Coreset-Induced Conditional Velocity Flow Matching](https://arxiv.org/abs/2605.12951) | 该论文提出CCVFM生成模型，通过熵Sinkhorn核心集构建闭式代理源替代分层整流流中内部流的高斯噪声源，使条件速度规律成为无需神经采样器即可采样的闭式高斯混合，并只需轻量级校正流即可完成训练。 |
| [^40] | [Measuring and Decomposing Mode Separation via the Canonical Diffusion](https://arxiv.org/abs/2605.08777) | 该论文提出以密度固有的唯一常数扩散系数可逆扩散过程（规范扩散）为基准，从其自协方差矩阵中提取量来测量并分解高维分布的模态分离程度，弥补了微分熵、PCA 等无标签工具无法刻画势垒分隔结构、而互信息又依赖潜在标签的不足。 |
| [^41] | [Black-box model classification under the discriminative factorization](https://arxiv.org/abs/2605.07878) | 提出判别式因子分解框架，用于区分黑盒模型级分类中查询集的质量优劣，理论上证明随机分类概率随查询预算呈指数衰减，并实证表明据此估计参数可预测性能衰减率并指导选取高质量查询集。 |
| [^42] | [Ensemble Distributionally Robust Bayesian Optimisation](https://arxiv.org/abs/2605.07565) | 提出了一种计算上可处理的集成分布鲁棒贝叶斯优化算法，能够处理连续上下文并获得次线性遗憾界，改进了当前最先进的理论结果。 |
| [^43] | [Spectrum-Adaptive Generalization Bounds for Trained Deep Transformers](https://arxiv.org/abs/2605.07297) | 该论文提出了针对训练后多层Transformer的谱自适应事后泛化界，通过在训练后逐层、逐矩阵类型灵活选择Schatten指数，在谱复杂性与维度、深度依赖之间实现自适应权衡，克服了现有方法需事先固定范数约束且对深度呈指数依赖的缺陷。 |
| [^44] | [Hypergraph Generation via Structured Stochastic Diffusion](https://arxiv.org/abs/2605.05024) | 提出HEDGE生成模型，通过在关联矩阵上进行结合超图双侧热算子与OU过程的结构化随机扩散，并利用闭式条件目标学习置换等变的逆向漂移场，实现对高阶交互、超边异质性和重叠结构的忠实超图生成。 |
| [^45] | [Sparse Network Inference under Imperfect Detection and its Application to Ecological Networks](https://arxiv.org/abs/2604.18820) | 该论文提出非负低秩分解框架，通过 ℓ_{1/2} 正则化在联合估计检测概率的同时促进生态二部网络中群体内与跨群体图的稀疏性，并设计了自适应惩罚的 ADMM 算法来求解该非凸非光滑问题。 |
| [^46] | [Dynamic Tokenization via Reinforcement Patching: End-to-end Training and Zero-shot Transfer](https://arxiv.org/abs/2603.26097) | ReinPatch 利用强化学习（基于组相对策略梯度 GRPG）将分块边界的选取形式化为离散决策过程，联合优化动态分块策略与下游序列骨干模型，无需连续松弛即可实现端到端训练并支持零样本迁移。 |
| [^47] | [CausalCompass: Evaluating the Robustness of Time-Series Causal Discovery in Misspecified Scenarios](https://arxiv.org/abs/2602.07915) | 提出了 CausalCompass——一个灵活可扩展的基准测试框架，通过八种假设违反场景系统评估时序因果发现方法的鲁棒性，并发现基于深度学习的方法在大多数场景下整体表现最优。 |
| [^48] | [Designing Service Systems from Textual Evidence](https://arxiv.org/abs/2601.21471) | 本研究将基于文本证据的服务系统设计问题形式化为固定置信度下的最佳臂识别问题，利用大语言模型生成的低成本代理评分并结合有选择性的人工审计来校正偏差，从而以最小的人工审计成本高置信度地识别出最优服务配置。 |
| [^49] | [Order-Optimal Sample Complexity of Rectified Flows](https://arxiv.org/abs/2601.20250) | 本文证明了整流流模型在标准神经网络假设下可达到 $\tilde{O}(\varepsilon^{-2})$ 的最优阶样本复杂度，改进了流匹配模型已有的 $O(\varepsilon^{-4})$ 界并匹配均值估计的最优速率。 |
| [^50] | [Matrix Factorization Framework for Community Detection under the Degree-Corrected Block Model](https://arxiv.org/abs/2601.06262) | 该论文将度校正块模型（DCBM）的推断重新表述为带约束的非负矩阵三因子分解问题，据此提出了一种具有理论初始化、可扩展且适用于任意DCBM网络的社区检测方法。 |
| [^51] | [Outlier-Robust Multi-Output Gaussian Processes](https://arxiv.org/abs/2510.26401) | 本文提出R-MOGP，一种对任意单个响应中的离群值具有可证明鲁棒性的多输出高斯过程，其在保持共轭性与计算效率的同时可与稀疏变分推断结合，以远低于现有最先进方法的计算成本达到相当的性能。 |
| [^52] | [Optimal In-context Adaptivity and Distributional Robustness of Transformers](https://arxiv.org/abs/2510.23254) | 本文证明了在多种任务难度混合分布上预训练的 Transformer，即使面对存在分布偏移的固定难度测试分布，也能达到最优收敛速率，展现出最优的上下文自适应性与分布鲁棒性。 |
| [^53] | [Closed-Form Last Layer Optimization](https://arxiv.org/abs/2510.04606) | 该论文提出在平方损失下将神经网络最后一层权重视为主干参数的闭式解函数，仅优化主干参数并在随机梯度下降设定下进行适配，理论上证明了该方法在神经正切核机制下收敛到最优解且优于标准SGD。 |
| [^54] | [Robust Multimodal Learning via Entropy-Gated Contrastive Fusion](https://arxiv.org/abs/2505.15417) | AECF是一种轻量级即插即用的熵门控对比融合层，通过逐实例自适应熵系数、跨模态子集的单调校准以及熵驱动的课程掩码，在输入缺失场景下同时提升多模态推理的鲁棒性与校准性。 |
| [^55] | [Linear Bandits on Ellipsoids: Minimax Optimal Algorithms](https://arxiv.org/abs/2502.17175) | 本文针对动作集为椭球体的线性随机老虎机问题，首次提出极小极大最优算法，通过推导新型信息论遗憾下界，并设计一种结合参数范数估计与探索-承诺策略的非经典（非乐观、非采样）高效算法来实现与该下界匹配。 |
| [^56] | [Identifying Drift, Diffusion, and Causal Structure from Temporal Snapshots](https://arxiv.org/abs/2410.22729) | 该论文提出了首个仅凭时间边缘分布（无需观测个体轨迹）即可联合识别随机微分方程漂移与扩散项的方法，证明了不可辨识性仅在初始分布具有广义旋转对称性时出现，且因果图可从识别出的SDE参数中恢复。 |

# 详细

[^1]: 可解码POMDP的拟多项式时间规划

    Quasipolynomial-Time Planning for Decodable POMDPs

    [https://arxiv.org/abs/2611.09778](https://arxiv.org/abs/2611.09778)

    本文为比可观测性严格更弱的"ρ-可解码"POMDP提出了拟多项式时间的近似规划算法，证明该条件被ℓ₁与ℓ₂可观测性共同蕴含，且最优可解码常数可在多项式时间内计算。

    

    我们研究有限时域、且状态、动作和观测空间均为有限集的部分可观测马尔可夫决策过程（POMDP）的规划问题。众所周知，在此设定下精确规划和近似规划在计算上都是困难的，因此先前的工作探索了使规划变得更可行的自然假设。最近的工作 [Golowich et al., 2022] 为可观测POMDP给出了拟多项式时间的近似规划算法，这类POMDP的观测信道满足 $\ell_1$-可观测性条件，但值得注意的是，其对转移动力学无需任何假设。沿着类似的方向，对于每个固定的 $\rho>0$，我们为 $\rho$-可解码POMDP给出了拟多项式时间的近似规划算法。我们证明可解码性同时被 $\ell_1$-和 $\ell_2$-可观测性所蕴含，并且严格弱于可观测性。此外，我们证明最优的可解码常数可以在多项式时间内计算……

    arXiv:2611.09778v1 Announce Type: cross  Abstract: We study planning for finite-horizon partially observable Markov decision processes (POMDPs) with finite state, action, and observation spaces. Both exact and approximate planning are known to be computationally hard in this setting, so prior work has explored natural assumptions that make planning easier. Recent work [Golowich et al., 2022] gave quasipolynomial-time approximate planning algorithms for observable POMDPs. These are POMDPs whose observation channels satisfy an $\ell_1$-observability condition, but notably, no assumptions are required on the transition dynamics. In a similar vein, for every fixed $\rho>0$, we give quasipolynomial-time approximate planning algorithms for $\rho$-decodable POMDPs. We show that decodability is implied by both $\ell_1$- and $\ell_2$-observability and is strictly weaker than observability. Moreover, we show that the optimal decodability constant can be computed in polynomial time, whereas compu
    
[^2]: 学习应该在哪里发生？文本、参数与联合自适应的遗憾分析

    Where Should Learning Live? Regret Analysis for Text, Parameter, and Joint Adaptation

    [https://arxiv.org/abs/2611.09701](https://arxiv.org/abs/2611.09701)

    该论文将部署阶段对提示与参数适配器的重复选择形式化为在线学习问题，把超额损失分解为候选族内的选择遗憾、候选族限制差距和部署历史影响三部分，并证明在损失不受选择器随机性影响、历史影响有限且限制切换次数的条件下可实现次线性期望遗憾。

    

    语言模型系统在部署过程中可以反复选择提示和参数适配器，并利用反馈来指导后续的选择。早期的选择可能会改变后续配置所继承的上下文或环境。我们研究了在预先构建的候选集合中进行选择的问题，并将累积损失与从头开始就一直使用最佳固定提示-参数对的情况进行比较。这一超额损失可以分解为三项：可用候选族内部的选择遗憾、因限制该候选族而产生的差距，以及实际部署历史所带来的收益或损害。评估选择时使用每个候选者若全程被使用时本会产生的历史，从而将候选者本身的质量与切换的影响分离开来。当损失函数独立于选择器的随机性而固定、过去的选择仅在有限轮数内影响损失、且每个候选者基于自身历史的损失都能被揭示时，限制切换次数可以获得次线性的期望遗憾。

    arXiv:2611.09701v1 Announce Type: new  Abstract: Language-model systems can select prompts and parameter adapters repeatedly during deployment, using feedback to guide later choices. Earlier choices may change the context or environment inherited by later configurations. We study selection among preconstructed candidates and compare cumulative loss with the best fixed prompt--parameter pair used from the beginning. This excess loss separates into three terms: selection regret within the available family, the gap from restricting that family, and the benefit or harm of the actual deployment history. Selection is evaluated using the history each candidate would have produced if used throughout, which separates candidate quality from the effects of switching. When loss functions are fixed independently of selector randomness, past choices affect loss for only a bounded number of rounds, and every candidate's own-history loss is revealed, limiting switches yields sublinear expected regret 
    
[^3]: 一种结合降维的广义贝叶斯聚类方法

    A Generalized Bayesian Approach to Clustering with Dimension Reduction

    [https://arxiv.org/abs/2611.09698](https://arxiv.org/abs/2611.09698)

    该论文提出将自编码器与广义贝叶斯聚类划分联合学习，在学到的低维表示空间中评估聚类损失，从而在不依赖完整生成模型的情况下实现高维数据的概率化聚类并量化分配不确定性。

    

    广义贝叶斯乘积划分模型为基于损失的聚类提供了概率化的对应方法。它们通过指数化的聚类损失（而非似然函数）来更新划分上的先验分布，因此无需为数据建立完整的生成模型即可量化分配的不确定性。然而，其性能取决于损失函数所评估的空间。对于高维观测数据，在原始特征上使用平方欧氏损失可能会被无关坐标主导，或者无法反映位于非线性低维流形附近的聚类结构。我们提出了一种扩展方法，将自编码器与划分联合学习，使聚类损失的评估空间由聚类本身塑造。我们的广义后验将重建损失（保持低维表示忠实于观测数据）与簇内平方和相结合……（摘要至此截断）

    arXiv:2611.09698v1 Announce Type: cross  Abstract: Generalized Bayes product partition models provide a probabilistic counterpart to loss-based clustering. They update a prior over partitions through an exponentiated clustering loss rather than a likelihood, so allocation uncertainty can be quantified without a full generative model for the data. However, their performance depends on the space in which the loss is evaluated. For high-dimensional observations, a squared Euclidean loss on the raw features can be dominated by irrelevant coordinates or fail to reflect cluster structure lying near a nonlinear low-dimensional manifold. We propose an extension in which an autoencoder and the partition are learned together, so that the space in which the clustering loss is evaluated is shaped by the clustering itself. Our generalized posterior combines a reconstruction loss, which keeps the low-dimensional representation faithful to the observed data, with a within-cluster sum of squares in th
    
[^4]: 基于双重稳健稳态学习的网络化系统在线因果配置

    Online Causal Configuration for Networked Systems via Doubly Robust Steady-State Learning

    [https://arxiv.org/abs/2611.09665](https://arxiv.org/abs/2611.09665)

    提出CAUSAL-UCB算法，通过干扰感知的裁剪双重稳健稳态学习方法，在测量探索会改变网络化系统稳态的困难条件下正确识别并在线优化配置的目标因果价值，并给出了推理误差与遗憾的理论下界。

    

    测量探索能够揭示相互作用的本地控制器的行为方式，但可能会改变网络运营商所寻求优化的稳态状态。因此，探索性奖励可能无法识别某一配置在无探索情况下的真实价值：在完全探索条件下，相同的记录分布可能掩盖不同的目标值。识别需要同时考虑行动规律的不匹配以及稳态上下文分布的偏移。我们提出了CAUSAL-UCB算法，它使用目标因果值的乐观界限来选择配置。其干扰感知的裁剪双重稳健评估器产生决策前证书，将统计波动与经过校准的识别、迁移和时间依赖性误差分离。对于任何有限时间范围和预先设定的测量探索率时间表 $\{\eta_t\}$，我们证明了极小极大推理误差的 $\Omega((1+\sum_t\eta_t)^{-1/2})$ 下界以及遗憾下界。

    arXiv:2611.09665v1 Announce Type: cross  Abstract: Measurement exploration reveals how interacting local controllers behave but can alter the steady-state regime a network operator seeks to optimize. Exploratory rewards may therefore fail to identify a configuration's no-exploration value: under full exploration, identical logged distributions can conceal different target values. Identification requires accounting for both action-law mismatch and shifts in the steady-state context distribution. We introduce CAUSAL-UCB, which selects configurations using optimistic bounds on target causal values. Its interference-aware clipped doubly robust evaluator produces predecision certificates separating statistical variation from calibrated identification, transport, and temporal-dependence errors. For any finite horizon and predetermined schedule of measurement-exploration rates $\{\eta_t\}$, we prove lower bounds of $\Omega((1+\sum_t\eta_t)^{-1/2})$ on minimax inference error and $\Omega(\sum_
    
[^5]: 基于核方法的分析算子与参数化偏微分方程学习

    Kernel-Based Learning of Analytic Operators and Parametric PDEs

    [https://arxiv.org/abs/2611.09612](https://arxiv.org/abs/2611.09612)

    该论文建立了基于核方法从有限样本学习解析算子的理论，证明合适的解析核可实现指数级收敛速率，并将其应用于参数化椭圆偏微分方程的解算子学习，给出了分离采样误差与有限元离散化误差的误差界及网格-样本耦合准则。

    

    我们研究基于核方法从有限样本中学习解析算子的问题，重点关注目标的正则性应如何决定核的选择。我们建立了解析算子到向量值再生核希尔伯特空间的嵌入，并推导了有限样本、高概率意义下的理论保证，以量化随机采样带来的不确定性：在无噪声情形下，合适的解析核可获得样本数量分数次幂的指数级收敛速率，而Sobolev型核只能获得多项式级速率。我们将该理论应用于参数化椭圆偏微分方程的解算子，其中训练数据本身是有限元近似，并得到了能够分离采样误差与离散化误差贡献的界，同时规定了网格尺寸应如何与样本数量相耦合。研究结果通过数值实验加以验证。

    arXiv:2611.09612v1 Announce Type: new  Abstract: We study kernel-based learning of analytic operators from finite samples, with a focus on how the regularity of the target should determine the choice of the kernel. We establish embeddings of analytic operators into vector-valued reproducing kernel Hilbert spaces and derive finite-sample, high-probability guarantees that quantify the uncertainty due to random sampling: in the noiseless setting, suitable analytic kernels yield rates exponential in a fractional power of the sample size, while Sobolev-type kernels yield polynomial rates. We apply the theory to the solution operator of a parametric elliptic PDE, where the training data are themselves finite-element approximations, and obtain bounds that separate the sampling and discretization contributions to the error and prescribe how the mesh size should be coupled to the sample size. The results are illustrated numerically.
    
[^6]: 多项式混沌展开的局部自适应保形预测

    Locally Adaptive Conformal Prediction for Polynomial Chaos Expansions

    [https://arxiv.org/abs/2611.09579](https://arxiv.org/abs/2611.09579)

    该论文提出了针对多项式混沌展开的局部自适应jackknife+保形预测框架，通过在嵌套留一残差上训练辅助模型进行归一化，在固定基展开情形下仅需单次模型拟合即可获得具有覆盖保证的自适应预测区间。

    

    本工作为多项式混沌展开开发了一种局部自适应的jackknife+保形预测框架。其中，非一致性得分通过一个辅助多项式混沌展开进行归一化，该辅助模型基于基础模型的对数变换留一残差进行训练。我们证明了如果留一辅助模型是在嵌套的留一残差上训练的，那么jackknife+保形预测的覆盖保证可以延续到自适应区间。对于通过普通最小二乘法拟合的固定基多项式混沌展开，所有所需量均以闭式形式推导得出，且保持覆盖保证的自适应区间可从单次模型拟合中获得，无需任何模型重训练。对于需要多次模型拟合的稀疏多项式混沌展开，我们提出了一种启发式简化方法，以牺牲形式化的覆盖保证为代价来限制自适应性的计算开销。

    arXiv:2611.09579v1 Announce Type: cross  Abstract: This work develops a locally adaptive jackknife+ conformal prediction framework for polynomial chaos expansions. Therein, the non-conformity scores are normalized with an auxiliary polynomial chaos expansion trained on the log-transformed leave-one-out residuals of the base model. We show that the coverage guarantee of jackknife+ conformal prediction carries over to the adaptive intervals if the leave-one-out auxiliary models are trained on nested leave-one-out residuals. For fixed-basis polynomial chaos expansions fitted by ordinary least squares, all required quantities are derived in closed form and the guarantee-preserving adaptive intervals are obtained from a single model fit, without any model retraining. For sparse polynomial chaos expansions, where multiple model fits are necessary, we propose a heuristic simplification that limits the computational overhead of adaptivity at the cost of the formal coverage guarantee. Numerical
    
[^7]: 边缘非一致多分类学习

    Marginal-Nonuniform Multiclass Learning

    [https://arxiv.org/abs/2611.09543](https://arxiv.org/abs/2611.09543)

    该论文提出了介于经典PAC学习与通用学习之间的边缘非一致学习框架，并通过新提出的非中心化DS-eluder维度完整刻画了最优学习速率，建立了指数、线性与任意慢速率的三分性结论。

    

    我们在边缘非一致学习框架下研究多分类问题，其中学习率中的常数可以依赖于生成无标签数据的未知边缘分布，但不能依赖于目标概念。这一概念自然地介于经典PAC学习（其中常数既不依赖于边缘分布也不依赖于目标概念）与较新的通用学习（其中常数可以同时依赖于两者）之间。我们给出了最优边缘非一致学习率的完整刻画，建立了一个基本的三分性：指数速率、线性速率和任意慢速率。这一刻画由一个新的组合维度——非中心化DS-eluder维度所见证。上界证明中的一个关键要素是一组新的分布耗尽引理，这些引理利用了该维度的序贯特性，可能具有独立的研究价值。

    arXiv:2611.09543v1 Announce Type: cross  Abstract: We study the problem of multiclass classification within the framework of marginal-nonuniform learning, where the constants in the learning rate may depend on the unknown marginal distribution generating the unlabeled data, but not on the target concept. This notion naturally lies between classical PAC learning, where the constants are independent of both the marginal distribution and the target concept, and more recent universal learning, where the constants may depend on both. We give a complete characterization of the optimal marginal-nonuniform learning rates, establishing a fundamental trichotomy: exponential, linear, and arbitrarily slow rates. This characterization is witnessed by a new combinatorial dimension, termed the uncentered DS-eluder dimension. A key ingredient in the upper-bound proofs is a set of new distributional exhaustion lemmas that exploit the sequential nature of this dimension and may be of independent interes
    
[^8]: k-Volume：一种统一的主动学习策略

    k-Volume: a Unified Active Learning Strategy

    [https://arxiv.org/abs/2611.09388](https://arxiv.org/abs/2611.09388)

    该论文提出了一种新型主动学习策略k-Volume，通过包含样本k近邻的最小球体体积来选择待标注样本，统一融合了密度、多样性和不确定性三大类主动学习策略的优势，理论上被证明渐近等价于均匀采样，并在各种分类器和实验设置中展现出卓越的稳定性。

    

    在标注样本标签代价高昂的分类场景中，我们提出了一种名为k-Volume（kVol）的新型主动学习（AL）策略。该策略根据包含样本k个最近邻的最小球体体积，随机选择待标注的样本。kVol具有高度通用性：它既可以应用于输入空间，也可以应用于输出空间，并且兼具主动学习三大主要策略家族的理想特性：密度、多样性以及（在输出空间中执行时的）不确定性。从理论角度，我们证明了kVol在渐近意义上等价于在（输入或输出）数据分布支撑集上的均匀采样。基于多种合成数据集、真实表格数据集和图像的实验，我们观察到kVol具有惊人的稳定性：无论使用何种分类器或实验设置，它始终位列最优的主动学习策略之中，而其他竞争方法则难以做到这一点。

    arXiv:2611.09388v1 Announce Type: cross  Abstract: In the classification context where revealing the label of a sample is costly, we propose a new Active Learning (AL) strategy named k-Volume (kVol). It randomly selects a sample to be labeled according to the volume of the smallest ball containing its k nearest neighbors. kVol is highly versatile: it can be applied either in the input space or in the output space, and it enjoys the desirable properties of the three main families of AL strategies: density, diversity, and (when performed in the output space) uncertainty. From a theoretical perspective, we establish that kVol is asymptotically equivalent to uniform sampling over the support of the (input or output) data distribution. Based on experiments on various synthetic and real-life tabular datasets and images, we observe that kVol is surprisingly stable: whatever the classifier or the experimental setup, it still lies among the best AL strategies unlike other competitors that fail 
    
[^9]: 校准你的置信度需要考虑到所有你可能出错的方式

    Calibrating your confidence requires thinking of all the ways you could be wrong

    [https://arxiv.org/abs/2611.08997](https://arxiv.org/abs/2611.08997)

    本论文提出一种协方差与Fisher信息矩阵零空间对齐的退化高斯分布，在保持贝叶斯不确定性量化质量的同时大幅降低计算开销，实现了高质量且可扩展的贝叶斯深度学习。

    

    现代深度学习的一个主要局限是模型置信度的校准失准；传统架构从根本上缺乏量化自身不确定性的机制。更大的模型和更大的数据集可以缓解这一问题，但无法解决其根本的结构性缺陷。贝叶斯方法通过随机参数提供了一个有原则的不确定性感知学习框架。然而，尽管其优势已得到证实，其固有的计算开销往往会使可扩展性或质量受到损害。这种权衡目前阻碍了贝叶斯方法的广泛应用。本论文通过在参数空间引入一种新颖的分布来弥合高质量与低复杂度之间的差距：一种退化高斯分布，其协方差是一个与Fisher信息矩阵零空间对齐的投影矩阵。我们通过两种数值高效的算法展示了这种贝叶斯后验的可扩展性。

    arXiv:2611.08997v1 Announce Type: cross  Abstract: A primary limitation of modern deep learning is the miscalibration of model confidence; traditional architectures fundamentally lack a mechanism to quantify their own uncertainty. Larger models and larger datasets mitigate this issue, but they do not resolve the underlying structural deficiency. Bayesian methods offer a principled uncertainty-aware learning framework through stochastic parameters. Yet, despite the established benefits, the inherent computational overhead often compromises either scalability or quality. This trade-off currently prevents the widespread adoption of Bayesian methods.   This Thesis bridges the gap between high-quality and low-complexity by introducing a novel distribution on parameter space: a degenerate Gaussian whose covariance is a projection matrix aligned with the null space of the Fisher information matrix. We demonstrate the scalability of this Bayesian posterior through two numerically efficient alg
    
[^10]: 在物理领域的预训练机器学习模型上强制实施对称性

    Enforcing Symmetries on Pre-trained ML Models for Physics

    [https://arxiv.org/abs/2611.08992](https://arxiv.org/abs/2611.08992)

    本文将事后群平均方法扩展应用于预训练的物理机器学习模型以强制实施对称性，并针对非紧致群提出了三种替代方案（包括Weyl幺正技巧和数据依赖的正交投影），且不改变模型架构或需要重新训练。

    

    事后群平均（或测试时增强）是一种简单的方法，可以在不改变已训练预测器的架构或重新训练的情况下对其施加对称性。对于有限群和紧致群，通过对群上归一化的哈尔测度进行平均可以得到雷诺兹算子：一个到不变或等变预测器的规范投影，在某些假设下具有已知的风险不增保证。本文在物理仿真模型上进行了数值实验，证明了该方法在这种场景下的有效性。随后，我们将该框架扩展到非紧致群对称性。对于非紧致群，哈尔测度无法归一化，因此无法以相同的方式定义群平均。我们用三种替代方案来取代它：Weyl幺正技巧（需要对空间进行复化并扩展预测器）、数据依赖的正交投影（涉及一个非平凡的优化过程）……

    arXiv:2611.08992v1 Announce Type: cross  Abstract: Post-hoc group averaging (or test-time augmentation) is a simple way to impose symmetry on a trained predictor without changing its architecture or retraining it. For finite and compact groups, averaging over a normalized Haar measure on the group gives the Reynolds operator: a canonical projection onto invariant or equivariant predictors with known non-increasing risk guarantees under certain assumptions. This paper provides numerical experiments on physics emulation models that demonstrate the effectiveness of the approach in this setting. We then extend this framework to non-compact group symmetries. For non-compact groups, the Haar measure cannot be normalized, so group averaging cannot be defined in the same way. We replace it with three alternatives: the Weyl unitary trick (which requires complexifying the space and extending the predictor), a data-dependent orthogonal projection (which involves a non-trivial optimization on the 
    
[^11]: 有序策略优化

    Ordered Policy Optimization

    [https://arxiv.org/abs/2611.08936](https://arxiv.org/abs/2611.08936)

    提出有序策略优化（OPO），通过在每个小批量中对重要性比率排序、识别影响最大的极端比率并为其分配自适应的逐样本裁剪半径，取代PPO的固定裁剪范围，从而更可靠地控制策略更新。

    

    近端策略优化（PPO）通过裁剪新策略与数据采集策略之间的重要性比率来稳定策略更新。标准规则对所有样本使用一个固定的裁剪范围。虽然这种选择简单且有效，但它无法区分统计上有害的极端比率值与较大但可容忍的变化。我们研究了如何使裁剪半径适应所诱导的重要性比率分布的可靠性。我们的案例研究表明，可靠的更新需要控制这些极端比率，而不仅仅是验证单个比率是否落在固定区间内。基于这一观察，我们提出了有序策略优化（OPO）。在每个小批量（minibatch）中，OPO对重要性比率进行排序，选出可能对更新产生最大影响的极端比率，并为它们分配自适应的逐样本半径，而不是使用固定范围……

    arXiv:2611.08936v1 Announce Type: cross  Abstract: Proximal Policy Optimization (PPO) stabilizes policy updates by clipping the importance ratio between the new policy and the data-collecting policy. The standard rule uses one fixed clipping range for all samples. While this choice offers simplicity and effectiveness, it fails to distinguish statistically harmful extreme ratio values from large, yet tolerable changes. We study how to adapt the clipping radius to the reliability of the induced importance ratio distribution. Our case study demonstrates that reliable updates require controlling these extreme ratios, rather than merely verifying that individual ratios fall within a fixed interval. Motivated by this observation, we propose Ordered Policy Optimization (\textsc{OPO}). In each minibatch, \textsc{OPO} sorts the importance ratios, selects the extreme ratios that potentially have the largest effect on the update, and assigns them adaptive sample-wise radii instead of using the fi
    
[^12]: 几何感知的分数排斥蒙特卡罗方法

    Geometry-Aware Score-Repellent Monte Carlo

    [https://arxiv.org/abs/2611.08909](https://arxiv.org/abs/2611.08909)

    本文提出几何感知的分数排斥蒙特卡罗方法（GA-SRMC），用迹归一化的逆 Fisher 矩阵替代单一标量排斥强度，并从理论上证明其能最优地同时应对各向异性能量地形中的陡峭与平坦方向，从而改进 MCMC 采样。

    

    最近提出的分数排斥蒙特卡罗（SRMC）方法通过在 $\mathbb{R}^d$ 中以因子 $\exp(-\alpha\,\theta_n^\top s(x))$ 对目标密度 $\pi$ 进行倾斜来改进 MCMC 采样，其中分数函数 $s(x) = \nabla_x \log \pi(x)$，$\theta_n$ 是来自历史样本的分数平均值。然而，单一的标量排斥强度 $\alpha\ge0$ 无法同时处理各向异性能量地形中的陡峭方向与平坦方向。我们提出几何感知 SRMC（GA-SRMC），将欧氏对齐推广为 $\theta^\top M s(x)$ 的形式，其中 $M$ 为任意对称正定矩阵。在 $M$ 的迹约束条件下，基于随机逼近的中心极限定理证明，迹归一化的逆 Fisher 矩阵 $M_{\rm opt}\propto S^{-1}$ 是最坏方向协方差界的唯一最小化器，从而将 Fisher 信息 $S=\text{Cov}_\pi(s,s)$ 从提提议侧预条件子（例如 FisherM……）提升为排斥机制本身的核心组件。

    arXiv:2611.08909v1 Announce Type: new  Abstract: The recently proposed Score-Repellent Monte Carlo (SRMC) method improves MCMC sampling by tilting the target density $\pi$ in $\mathbb{R}^d$ with the factor $\exp(-\alpha\,\theta_n^\top s(x))$, where the score function $s(x) = \nabla_x \log \pi(x)$, and $\theta_n$ is the score average from past samples. However, a single scalar repellence strength $\alpha\ge0$ cannot handle both the steep and the flat directions in anisotropic energy landscapes at once. We introduce \emph{Geometry-Aware SRMC} (GA-SRMC), which generalizes the Euclidean alignment to $\theta^\top M s(x)$ for any symmetric positive-definite matrix $M$. A stochastic-approximation central limit theorem under a trace constraint on $M$ identifies the trace-normalized inverse Fisher $M_{\rm opt}\propto S^{-1}$ as the unique minimizer of the worst-direction covariance bound, elevating the Fisher information $S=\text{Cov}_\pi(s,s)$ from a proposal-side preconditioner (e.g., FisherM
    
[^13]: 迁移分类的重加权：精确风险转换与极小极大自适应

    Reweighting for Transfer Classification: Exact Risk Conversion and Minimax Adaptation

    [https://arxiv.org/abs/2611.08876](https://arxiv.org/abs/2611.08876)

    本文推导出迁移分类中由混合超额风险到目标超额合页风险的精确转换曲线（由密度加权局部投票决定），并在倾斜边界类上建立了将源采样不确定性与关系不确定性分离的极小极大分类率。

    

    一个源域可以让训练混合数据更容易学习，同时降低其准确性对目标域的价值。我们研究源域造成的训练问题以及学习它的有限样本误差如何共同决定重加权下的目标风险保证。对于二分类问题，针对每一对分布和每个正的目标权重，我们推导出仅由混合超额风险隐含的最尖锐的目标超额合页风险界，且该界在所有截断预测器上一致成立。这一精确转换曲线由密度加权的局部投票决定。加权支持向量机的两样本预言机不等式提供了混合超额风险的估计，而对于高斯核，两者的组合变得显式。在一个倾斜边界类上，我们建立了关于可迁移强度 $s_+$ 和倾斜度 $\tau$ 的统一极小极大分类率 $[n+s_+^2/(m^{-1}+\tau^2)]^{-1}$，将源采样不确定性 ($m^{-1}$) 与关系不确定性分离开来。

    arXiv:2611.08876v1 Announce Type: cross  Abstract: A source can make a training mixture easier to learn while making its accuracy less valuable for the target. We study how the training problem a source creates and the finite-sample error of learning it jointly determine target-risk guarantees under reweighting. For binary classification, for each pair of laws and each positive target weight, we derive the sharpest target excess hinge-risk bound implied by mixture excess alone, uniformly over clipped predictors. This exact conversion curve is determined by density-weighted local votes. A two-sample oracle inequality for weighted support vector machines supplies the mixture excess, and the composition becomes explicit for Gaussian kernels. On a tilted-boundary class, we establish the all-procedure minimax classification rate $[n+s_+^2/(m^{-1}+\tau^2)]^{-1}$ in the transferable strength $s_+$ and the tilt $\tau$, separating source sampling uncertainty ($m^{-1}$) from relation uncertainty
    
[^14]: 超越下星过滤：持久同调的替代过滤方法

    Beyond the Lower-Star Filtration: Alternative Filtration Methods for Persistent Homology

    [https://arxiv.org/abs/2611.08833](https://arxiv.org/abs/2611.08833)

    本文针对下星过滤中高维单纯形与其边界面同时进入导致拓扑特征被立即填充而无法体现在持久图中的局限性，提出了ELSF、LpF和SMF三种替代过滤方法，以更完整地捕捉拓扑特征。

    

    在下星过滤（LSF）中，单纯形根据最大顶点规则被分配过滤值，即每个单纯形在其最高值顶点出现时进入过滤。这种构造的一个局限性在于，某些高维单纯形可能与其边界面同时进入过滤，导致某些拓扑特征在形成之初就被立即填充，因而不会出现在最终的持久图中。为解决这一局限性，我们提出了三种替代过滤方法：ε-下星过滤（ELSF）、L_p过滤（LpF）和softmax过滤（SMF）。ELSF在最大顶点规则中引入了与维度相关的延迟，使高维单纯形在其边界面之后进入过滤；而LpF和SMF则使用依赖于所有顶点值而非仅仅依赖最大值的聚合函数。通过修改（摘要在此处被截断）

    arXiv:2611.08833v1 Announce Type: new  Abstract: In the lower-star filtration (LSF), simplices are assigned filtration values according to the maximum-vertex rule, with each simplex entering the filtration when its highest-valued vertex appears. A limitation of this construction is that some higher-dimensional simplices may enter simultaneously with their boundary faces, causing certain topological features to be filled immediately upon formation and thus not appear in the resulting persistence diagram. To address this limitation, we propose three alternative filtration methods: the \(\varepsilon\)-lower-star filtration (ELSF), the \(L_p\) filtration (LpF), and the softmax filtration (SMF). {ELSF introduces a dimension-dependent delay to the maximum-vertex rule,} allowing higher-dimensional simplices to enter the filtration after their boundary faces, whereas {LpF and SMF use aggregation functions that depend on all vertex values rather} than solely on their maximum. By modifying how f
    
[^15]: 使用翁萨格修正实现无需样本分割的高维离线策略评估

    High-Dimensional Off-Policy Evaluation without Sample Splitting using an Onsager Correction

    [https://arxiv.org/abs/2611.08794](https://arxiv.org/abs/2611.08794)

    本文提出一种经Onsager修正且以目标策略加权的准则，仅利用训练观测数据即可估计全数据奖励模型在目标策略下的样本外风险，从而在高维离线策略评估中无需样本分割即可实现全数据梯度下降轨迹上的早停，克服了留出数据选择可能次优的问题。

    

    用于离线策略评估（OPE）的奖励模型通常需要借助留出数据进行选择。当样本量与上下文维度成比例增长时，在全样本上拟合的模型与在固定比例样本上拟合的模型之间的测试风险差异不一定趋于消失。因此，留出评估只能一致地估计在较小样本上训练的模型的测试风险，但由于候选奖励模型的相对排名在全数据重新拟合后可能发生变化，基于留出数据的选择可能是次优的。我们开发了一种经过Onsager修正、以目标策略加权的选择准则，该准则仅使用训练观测数据即可估计全数据奖励模型在目标策略下的样本外风险。这使得我们可以在全数据梯度下降轨迹上进行早停，而无需样本分割。对于高斯上下文下具有固定深度和隐藏层宽度的神经网络、单指标奖励以及依赖上下文的多指标目标……

    arXiv:2611.08794v1 Announce Type: new  Abstract: Reward models used for off-policy evaluation (OPE) are commonly selected using held-out data. When sample size and context dimension grow proportionally, the difference in test risk between models fitted on the full sample and on a fixed fraction of it need not vanish. Held-out evaluation can therefore consistently estimate the test risk of the model trained on the smaller sample, but because the relative ranking of candidate reward models can change after full-data refitting, held-out selection can be suboptimal. We develop an Onsager-corrected, target-policy-weighted criterion that estimates the full-data reward model's out-of-sample risk under the target policy using only its training observations. This enables early stopping along the full-data gradient-descent trajectory without sample splitting. For neural networks with fixed depth and hidden width under Gaussian contexts, single-index rewards, context-dependent multi-index target 
    
[^16]: 一般方差分布下大型数据矩阵的盲噪声均衡化

    Blind Noise Equalization for Large Data Matrices under General Variance Profiles

    [https://arxiv.org/abs/2611.08784](https://arxiv.org/abs/2611.08784)

    该论文提出了BRINE方法，通过迭代重缩放行与列来均衡未知噪声方差，从而在异方差噪声下恢复大型数据矩阵噪声谱的Marchenko-Pastur定律并放大信号成分，且无需预先了解信号或噪声结构。

    

    低秩近似是从大型含噪数据矩阵中进行推断的基本工具。然而，当噪声方差在各元素之间不尽相同时（这一情形被称为异方差性），噪声会扭曲观测到的谱，产生虚假成分并掩盖微弱信号。为应对这些挑战，我们提出了BRINE（基于预解式的盲迭代噪声均衡器），这是一种数据驱动的方法，通过迭代地对行和列进行重新缩放，以均衡两个维度上的平均噪声方差。在适当条件下，该过程能够恢复噪声谱的Marchenko-Pastur定律，并且无需事先了解信号或噪声结构即可放大信号成分。我们方法的一个关键思想是均衡数据预解式的对角元素，以此作为均衡未知噪声方差的代理。在总体层面，我们识别出一个刻画这种均衡化过程的势函数……

    arXiv:2611.08784v1 Announce Type: new  Abstract: Low-rank approximation is a basic tool for inference from large and noisy data matrices. However, when noise variances differ across entries, a setting known as heteroskedasticity, the noise can distort the observed spectrum, producing spurious components and obscuring weak signals. To address these challenges, we develop BRINE (Blind Resolvent-based Iterative Noise Equalizer), a data-driven procedure that iteratively rescales rows and columns to equalize the average noise variance across both dimensions. Under suitable conditions, this procedure restores the Marchenko-Pastur law for the noise spectrum and can amplify signal components without prior knowledge of the signal or noise structure. A key idea underlying our approach is to equalize the diagonal entries of the data resolvent as a proxy for equalizing the unknown noise variances. At the population level, we identify a potential function that characterizes this equalization and mo
    
[^17]: 关于带归一化层的深度神经网络中的Lipschitz爆炸：对优化与对抗鲁棒性的影响

    On Lipschitz Explosion in Deep Neural Networks with Normalization: Consequences for Optimization and Adversarial Robustness

    [https://arxiv.org/abs/2611.08775](https://arxiv.org/abs/2611.08775)

    本文证明了带归一化层的深度神经网络（如批归一化网络）的输入和参数Lipschitz常数即使在每个层的范数都受控的情况下，仍会随深度呈指数增长，这种逐层不可见的全局不稳定性对优化保证和对抗鲁棒性具有重要影响。

    

    一个网络是否可能在每一层看起来都稳定，但整体却呈指数级不稳定？神经网络的Lipschitz常数是出现在优化保证、泛化界和鲁棒性证书中的基本稳定性度量。深度网络自然存在两种这样的概念：输入Lipschitz性，衡量对数据扰动的敏感性；参数Lipschitz性，衡量对权重扰动的敏感性。我们证明，对于带归一化层的深度网络（以批归一化为典型例子），即使存在强逐层范数控制（如$\|W_\ell\|_2\le 1$），且仅在线性激活函数的情况下，两个Lipschitz常数都可以随深度（即参数维度）呈指数增长。因此，一种被广泛用于稳定训练的机制可能造成严重的最坏情况不稳定性，而这种不稳定性在局部是不可见的，即仅凭逐层范数界无法察觉。参数Lipschitz爆炸……

    arXiv:2611.08775v1 Announce Type: cross  Abstract: Can a network look stable at every layer and still be exponentially unstable as a whole? The Lipschitz constant of a neural network is a basic stability quantity appearing in optimization guarantees, generalization bounds, and robustness certificates. Deep networks naturally admit two such notions, input Lipschitzness, measuring sensitivity to data perturbations, and parameter Lipschitzness, measuring sensitivity to weight perturbations. We prove that, for deep networks with normalization layers, with batch normalization as the canonical example, both Lipschitz constants can grow exponentially with depth and hence with parameter dimension, even under strong per-layer norm control such as $\|W_\ell\|_2\le 1$, and already for linear activations. Thus, a mechanism widely used to stabilize training can create severe worst-case instabilities that are invisible locally, i.e., from layerwise norm bounds alone. Parameter-Lipschitz explosion tu
    
[^18]: MEDALLION：用于低维潜在解释的流形嵌入蒸馏与归因

    MEDALLION: Manifold Embedding Distillation and Attributions for Low-dimensional Latent InterpretatiON

    [https://arxiv.org/abs/2611.08767](https://arxiv.org/abs/2611.08767)

    MEDALLION提出了一种结合知识蒸馏与神经特征归因的新框架，能够解释任意降维方法（如t-SNE、UMAP）生成的低维嵌入中哪些原始特征起关键作用。

    

    降维被广泛用于高维数据可视化和科学发现指导，然而解释哪些原始特征对低维表征有贡献仍然具有挑战性，尤其是对于t-SNE和UMAP等黑盒方法。我们提出了MEDALLION（Manifold Embedding Distillation and Attributions for Low-dimensional Latent InterpretatiON），这是一个通过知识蒸馏和神经特征归因来解释低维嵌入的新型框架。MEDALLION使用神经编码器对任意给定的嵌入进行蒸馏，随后通过多种现有的神经特征归因方法对该编码器进行解释。通过在MNIST上的实验以及单细胞转录组学中的应用，我们展示了MEDALLION能够提供信息丰富的特征归因，为底层嵌入带来额外的见解，并将其性能与现有的可解释性方法进行了比较。

    arXiv:2611.08767v1 Announce Type: cross  Abstract: Dimensionality reduction is widely used to visualize high-dimensional data and guide scientific discovery, yet interpreting which original features contribute to low-dimensional representations remains challenging, particularly for black-box methods such as t-SNE and UMAP. We introduce \textbf{MEDALLION} (Manifold Embedding Distillation and Attributions for Low-dimensional Latent InterpretatiON), a novel framework for interpreting low-dimensional embeddings through knowledge distillation and neural feature attribution. MEDALLION distills any given embedding with a neural encoder which we explain via a broad range of existing neural feature attribution methods. Through experiments on MNIST and applications in single-cell transcriptomics, we show MEDALLION provides informative feature attributions that cast additional insights into the underlying embedding and compare its performance with existing interpretability approaches.
    
[^19]: pESTO：基于风险驱动早停的迭代p-贪婪核回归

    pESTO: Iterative p-greedy kernel regression with risk-driven Early STOpping

    [https://arxiv.org/abs/2611.08555](https://arxiv.org/abs/2611.08555)

    该论文提出pESTO方法，通过显式分离逼近误差与噪声放大，并采用Stein无偏风险估计（SURE）作为均方误差的代理，为p-贪婪核回归中节点数量的选择提供了一个有原则且可计算的早停准则。

    

    基于核的贪婪算法通过最小化逐点误差指标（如幂函数）来选择合适的逼近节点；在此背景下，它们通常被称为p-贪婪方法。在存在噪声数据的情况下，确定性准则可能失效，因为真实解与其估计值之间的偏差同时受到逼近误差和噪声放大两者的影响，而二者之间的平衡是未知的。本研究显式地分离了这两个贡献项，它们分别由幂函数和一个Lebesgue型项所控制。这些界启发了使用Stein无偏风险估计（SURE）作为均方误差的代理，进而为p-贪婪过程中应选择的节点数量提供了一个有原则且可计算的停止准则。数值实验表明，该方法能够有效地平衡确定性精度与统计噪声控制。

    arXiv:2611.08555v1 Announce Type: new  Abstract: Kernel-based greedy algorithms select suitable approximation nodes by minimizing point-wise error indicators, such as the power function; in this context, they are usually referred to as p-greedy approaches. In the presence of noisy data, deterministic criteria may fail, as the discrepancy between the true solution and its estimate is influenced by both approximation error and noise amplification, and the balance between the two is unknown. This study explicitly separates these two contributions, which are governed by the power function and a Lebesgue-type term, respectively. These bounds motivate the use of the Stein's unbiased risk estimator (SURE) as a proxy of the mean squared error, which in turn provides a principled and computable stopping criterion for the number of nodes to select in the p-greedy procedure. Numerical experiments demonstrate that this approach effectively balances deterministic accuracy and statistical noise cont
    
[^20]: ELPAC：端点锚定的潜在进展与阶段变化的多模态协调

    ELPAC: Endpoint-Anchored Latent Progression with Stage-Varying Multimodal Coordination

    [https://arxiv.org/abs/2611.08541](https://arxiv.org/abs/2611.08541)

    提出ELPAC贝叶斯框架，通过端点锚定的潜在进展建模与阶段变化的多模态残差协调，从部分观测的横断面数据中联合推断老年队列受试者的潜在阶段、进展亚型及跨模态相互作用。

    

    老年队列研究日益结合影像、血浆和认知测量数据，但大多数受试者仅提供了部分观测的横断面快照，使得潜在阶段、进展亚型以及随阶段变化的多模态协调需要被联合推断。现有方法主要估计队列层面的衰老过程、在观测协变量下建模协方差，或融合多模态进行预测。然而，尽管受试者分期、亚型分配和多模态相互作用彼此相互依赖，这些方法很少针对端点锚定分期与按进展解析的跨模态残差协调这一联合问题。我们提出了ELPAC，一个包含两个相互衔接构造的贝叶斯框架。第一个构造从临床可靠的终点组出发，沿着编码模态特异性单调方向及可选顺序排列先验的生物学结构化速度分量，反向整合亚型轨迹。

    arXiv:2611.08541v1 Announce Type: new  Abstract: Aging cohorts increasingly combine imaging, plasma, and cognitive measurements, but most subjects provide only a partially observed cross-sectional snapshot, leaving latent stage, progression subtype, and stage-varying multimodal coordination to be inferred jointly. Existing methods estimate the cohort-level aging process, model covariance under observed covariates, or fuse modalities for prediction. However, they rarely target the joint problem of endpoint-anchored staging and progression-resolved residual coordination across modalities, even though subject staging, subtype assignment, and multimodal interplay are mutually dependent. We introduce ELPAC, a Bayesian framework with two interlocking constructions. The first integrates subtype trajectories backward from clinically reliable endpoint groups along biologically structured velocity components encoding modality-specific monotone direction and optional sequential ordering priors. T
    
[^21]: 基于最优传输的漂移可容许性检验

    Drift-admissibility via Optimal Transport

    [https://arxiv.org/abs/2611.08505](https://arxiv.org/abs/2611.08505)

    该论文提出基于最优传输的结构感知诊断方法，利用 KL 投影间隙与可扩展的 Bregman–Dykstra 算法检验时间序列是否符合指定可预测漂移类，并能区分二阶统计量无法分辨的过程（如同 Hurst 指数的分数布朗运动与 BSS 过程）。

    

    我们提出了一种结构感知的最优传输诊断方法，用于评估经验时间序列动态是否与指定的可预测漂移类相容。该方法将随机过程的路径性质转化为经验转移律上的有限维投影问题，把可容许性编码为局部耦合上的行方向矩约束，并通过 Kullback–Leibler 投影间隙 $G_{\lambda,k}$ 来量化偏差。这不仅提供了一个有理论依据的诊断工具，还带来了一种可扩展的 Bregman–Dykstra 算法，该算法在引入结构性约束的同时保留了 Sinkhorn 类方法的计算效率。以布朗运动参考零假设进行校准后，$G_{\lambda,k}$ 能够区分仅凭二阶统计量无法分辨的过程类别——包括具有相同 Hurst 指数的分数布朗运动与 BSS 过程——且无需渐近重采样或连续时间恢复。

    arXiv:2611.08505v1 Announce Type: new  Abstract: We introduce structure-aware optimal transport diagnostics for assessing whether empirical time-series dynamics are compatible with a prescribed predictable-drift class. Our approach translates a pathwise property of stochastic processes into a finite-dimensional projection problem over empirical transition laws, encoding admissibility as row-wise moment constraints on local couplings and quantifying deviations via a Kullback--Leibler projection gap $G_{\lambda,k}$. This yields both a principled diagnostic and a scalable Bregman--Dykstra algorithm that retains the efficiency of Sinkhorn-type methods while incorporating structural constraints. Calibrated against a Brownian reference null, $G_{\lambda,k}$ separates process classes that are indistinguishable by second-order statistics alone---including fractional Brownian motion and BSS processes sharing identical Hurst indices---without asymptotic resampling or continuous-time recovery. Th
    
[^22]: 时间依赖偏微分方程的共形预测

    Conformal Prediction for Time-Dependent PDEs

    [https://arxiv.org/abs/2611.08409](https://arxiv.org/abs/2611.08409)

    本文证明了时间依赖偏微分方程的共形预测覆盖保证在本质上难以实现——函数空间中相近时刻的分布可相互奇异，离散化中全变差距离随网格分辨率指数饱和——并给出了为确保覆盖而对空间离散化施加的必要限制。

    

    不确定性量化在科学机器学习中至关重要，因为模型的输出会为洪水预报、金融风险管理以及机器热控制等安全关键任务提供依据。共形预测能够提供无分布假设的覆盖保证，但在物理和工程中常见的时间依赖场景下，这些保证可能失效，导致系统性的覆盖不足。我们在由偏微分方程描述的时间依赖物理系统的代理模型背景下研究这一问题。我们证明，在函数空间设定下，任意接近的两个时刻的分布可以相互奇异，使得精确的覆盖保证不可能实现。我们进一步表明，在离散化设定下，分布之间的全变差距离往往随网格分辨率的提高而呈指数级饱和，这要求对空间离散化施加限制才能确保覆盖。最后，对于某些……

    arXiv:2611.08409v1 Announce Type: new  Abstract: Uncertainty quantification is crucial in scientific machine learning, where models inform safety-critical tasks such as flood forecasting, financial risk management, and thermal control in machines. Conformal prediction provides distribution-free coverage guarantees, but in time-dependent settings common to physics and engineering, these guarantees can break down, leading to systematic undercoverage. We study this problem in the context of surrogate models for time-dependent physical systems described by partial differential equations. We prove that in a function space setting, distributions at arbitrarily close times can be mutually singular, making exact coverage guarantees impossible. We further show that in discretized settings, the total variation distance between distributions often saturates exponentially fast with the grid resolution, mandating restrictions on the spatial discretization to ensure coverage. Finally, for certain se
    
[^23]: 支持向量机直接最小化KL散度

    Support Vector Machines Directly Minimise KL Divergence

    [https://arxiv.org/abs/2611.05557](https://arxiv.org/abs/2611.05557)

    本文证明支持向量分类与回归的对偶目标等价于两个高斯测度之间的带约束KL散度最小化，从而为标准核机器建立了统一的信息论解释。

    

    支持向量机（SVM）从根本上建立在几何间隔最大化之上，目前缺乏主流的概率解释。这使其在理论上与那些天然可比较、可组合的概率机器学习模型相互隔离。在本文中，我们通过证明支持向量分类和支持向量回归的对偶目标函数都是两个高斯测度之间精确的、带约束的Kullback-Leibler（KL）散度最小化任务，来弥合这一差距。我们进一步证明，当去除定义域约束并施加 $L^2$ 正则化时，同样的散度形式可以恢复核岭回归的精确目标函数。通过证明SVM的对偶问题直接最小化相对熵，我们的结果为标准核机器提供了一个统一的信息论模型，并为在概率框架下统一更广泛的基于核的算法奠定了基础。

    arXiv:2611.05557v1 Announce Type: new  Abstract: Support vector machines (SVMs) are fundamentally grounded in geometric margin maximisation and currently lack a mainstream probabilistic interpretation. This keeps them theoretically isolated from probabilistic machine learning models that are naturally comparable and combinable. In this paper, we bridge this gap by proving that the dual objectives of both support vector classification and support vector regression are exact, constrained Kullback-Leibler (KL) divergence minimisation tasks between two Gaussian measures. We further show that this same divergence formulation recovers the exact objective function for kernel ridge regression when domain constraints are removed and $L^2$ regularisation is applied. By demonstrating that SVM duals directly minimise relative entropy, our results provide a unified information-theoretic model for standard kernel machines and lay the foundation for unifying broader kernel-based algorithms under prob
    
[^24]: 下珠穆朗玛峰（第二部分）：最小化凸 $(L_0,L_1)$-光滑函数之和的预言机复杂度

    Descending Everest, Part II: Oracle Complexity of Minimizing a Sum of Convex $(L_0,L_1)$-Smooth Functions

    [https://arxiv.org/abs/2611.04171](https://arxiv.org/abs/2611.04171)

    提出了结合加速近端方法、方差缩减与有界启发式选点策略的 YETI 方法，并给出了最小化凸 $(L_0,L_1)$-光滑函数之和的预言机复杂度的匹配上界与下界（相差对数因子）。

    

    我们研究 $n$ 个凸 $(L_0,L_1)$-光滑函数平均值的最小化问题，其中每个函数的 Hessian 范数被其梯度范数的仿射函数所界定。我们假设已知从起点到某个极小值点距离的上界 $R$，以及平均函数的最小值与各分量函数下确界的平均值之差的上界 $\delta$。我们推导了在期望意义下函数值达到精度 $\varepsilon$ 所需的预言机复杂度的上界与下界。在对数因子范围内，两个界均为 $n(1+(L_1R)^{2/3}) +\sqrt{\frac{n(L_0+L_1^2\delta)R^2}{\varepsilon}+n(L_1R)^2}$ 阶。该上界由所提出的方法 YETI 达到。该方法将加速近端方法与方差缩减方法的思想相结合，并引入了一种选择附加点的启发式策略。该启发式策略的预言机调用次数是有界的，从而保证了总预言机调用次数的上界仍然成立。

    arXiv:2611.04171v2 Announce Type: replace-cross  Abstract: We study the minimization of the average of $n$ convex $(L_0,L_1)$-smooth functions: the Hessian norm of each function is bounded by an affine function of its gradient norm. We assume known upper bounds $R$ on the distance from the starting point to a minimizer and $\delta$ on the difference between the minimum of the average and the average of the infima of the components. We derive upper and lower bounds on the oracle complexity of reaching accuracy $\varepsilon$ in function value in expectation. Up to logarithmic factors, both bounds are of order $n(1+(L_1R)^{2/3}) +\sqrt{\frac{n(L_0+L_1^2\delta)R^2}{\varepsilon}+n(L_1R)^2}$. The upper bound is attained by the proposed method YETI. The method combines ideas of accelerated proximal methods and variance reduction methods with a heuristic for choosing additional points. The number of oracle calls made by the heuristic is bounded, which preserves the upper bound on the total num
    
[^25]: 面向混合人工智能的超越统计一致性的机器学习

    Machine Learning Beyond Statistical Consistency for Hybrid AI

    [https://arxiv.org/abs/2611.01914](https://arxiv.org/abs/2611.01914)

    该论文提出机器学习的新定义，证明机器学会一个真实概率不仅需要统计一致性（真值条件），还需要机器能在形式系统内部正确推导出自身估计的收敛性（保证条件），从而将学习的标准置于统计一致性之上，为混合人工智能奠定形式化基础。

    

    我们提出了机器学习的一个新定义：当机器成功计算一个函数时，即它对每个输入所返回的值在形式系统中都具有适当的推导时，该机器就学会了这个函数。我们将哥德尔可表示性定理（该定理将计算与推导等同起来）在两个方面扩展至经验目标：一是将该定理扩展到可计算实数，二是将数据作为记录已观察内容的形式化定义纳入系统。我们证明，机器学会一个真实概率当且仅当满足两个条件：真值条件，即其估计收敛于真实概率，这也就是统计一致性；以及保证条件，即机器在系统内部正确推导出自身的收敛性。因此，学习超越了统计一致性，因为它还要求能够从记录中推导出收敛性。

    arXiv:2611.01914v1 Announce Type: cross  Abstract: We propose a new definition of machine learning, according to which a machine learns a function when it computes the function successfully, that is, when the value it returns for each input has a proper derivation in a formal system. We extend G\"odel's Representability Theorem, which equates computation with derivation, to empirical targets in two ways: the theorem is extended to the computable reals, and the data are admitted into the system as empirical definitions that record what has been observed. We prove that a machine learns a true probability if and only if two conditions are met. The Truth Condition is that its estimate converges to the true probability, which is statistical consistency. The Assurance Condition is that the machine properly derives its own convergence inside the system. Learning thus lies beyond statistical consistency, for it requires a derivation of the convergence from the record. A global assumption about
    
[^26]: 面向多目标贝叶斯优化的单目标采集函数与对冲策略的简单扩展

    Simple Extensions of Single-Objective Acquisition Functions and Hedge Strategies for Multi-Objective Bayesian Optimization

    [https://arxiv.org/abs/2609.31940](https://arxiv.org/abs/2609.31940)

    该论文提出一个基于超体积的简单框架，能够将常用的单目标采集函数与对冲策略直接扩展到多目标贝叶斯优化中，仅需对现有流程做极小修改，无需专门的多目标设计。

    

    多目标贝叶斯优化（MOBO）通常通过专门设计的采集函数或标量化方案来实现，这些方法旨在显式地权衡非偏好性目标之间的取舍。在本工作中，我们表明这种复杂性可能是不必要的。我们提出了一个框架，通过基于超体积的方法，将标准的单目标采集函数直接扩展到多目标设置中。我们进一步将采集函数的对冲策略——通常仅在单目标优化中使用——扩展到多目标场景。我们的方法只需对现有贝叶斯优化流程进行极少的修改，并避免了定制化多目标公式的需求。我们展示了大量常用的单目标采集函数和对冲策略如何能够以有原则的方式被改造以处理多个目标，同时保持……（摘要在此处截断）

    arXiv:2609.31940v2 Announce Type: replace-cross  Abstract: Multi-objective Bayesian optimization (MOBO) is commonly approached through specialized acquisition functions or scalarization schemes designed to explicitly account for trade-offs among non-preferential objectives. In this work, we show that such complexity might be unnecessary. We propose a framework that extends standard single-objective acquisition functions directly to the multi-objective setting through a hypervolume-based methodology. We further extend hedge strategies for acquisition functions, which are typically used only in single-objective optimization, to the multi-objective regime. Our approach requires minimal modification to existing Bayesian optimization pipelines and avoids the need for bespoke multi-objective formulations. We demonstrate how a broad class of commonly used single-objective acquisition functions and hedge strategies can be adapted in a principled manner to handle multiple objectives, while pres
    
[^27]: 通过密度比估计进行先验变化引导

    Guidance for Prior Change via Density Ratio Estimation

    [https://arxiv.org/abs/2608.21729](https://arxiv.org/abs/2608.21729)

    提出一种基于密度比估计的无偏测试时引导框架，通过学习得分引导项有效解耦推断过程与先验训练，避免了先验依赖和系统性偏差。

    

    模拟推演推断（SBI）是在模拟器涉及难以处理的似然函数的科学领域中，用于参数推断的重要框架。然而，虽然摊销生成模型能够快速进行后验估计，但它们通常受限于训练期间使用的特定先验，从而随着先验知识的发展限制了其灵活性。为解决这一先验依赖问题，引入了PriorGuide作为一种推理时引导方法，但由于其难以处理的公式形式，它依赖于反向转移核的高斯近似和先验比的高斯混合模型拟合，这两者都会引入系统性偏差。受这些局限性的启发，我们提出了一种无偏的测试时引导框架，利用密度比估计（DRE）学习一个得分引导项，有效将推断过程与先验训练解耦。此外，我们的框架对具体模型规格保持不可知性。

    arXiv:2608.21729v1 Announce Type: new  Abstract: Simulation-Based Inference (SBI) serves as a vital framework for parameter inference in scientific fields where simulators involve intractable likelihoods, yet while amortized generative models offer rapid posterior estimation, they are often restricted by the specific priors used during training, thereby limiting their flexibility as prior knowledge evolves. To address this prior dependency, PriorGuide was introduced as an inference-time guidance method, but due to its intractable formulation, it relies on Gaussian approximations of the reverse transition kernel and Gaussian mixture model fitting for the prior ratio, both of which introduce systematic bias. Motivated by these limitations, we propose an unbiased test-time guidance framework that leverages Density Ratio Estimation (DRE) to learn a score guidance term, effectively decoupling the inference process from the prior training. Moreover, our framework remains agnostic to the spec
    
[^28]: 基于组高斯镜像与置换SHAP的模型无关FDR控制

    Model-Agnostic FDR Control via Group Gaussian Mirror and Permutation SHAP

    [https://arxiv.org/abs/2608.00989](https://arxiv.org/abs/2608.00989)

    该论文提出了一个面向分组和序列模型的模型无关FDR控制框架，通过组高斯镜像统计量与置换SHAP块级重要性得分相结合，在低维和高维设定下实现了对原始特征块级别的误发现率控制。

    

    大多数FDR控制的特征选择方法是为坐标级假设设计的，即每个特征只有单一的权重或重要性得分。这种抽象在序列模型和分组模型中会失效，因为在这些模型中，一个原始特征由一组子特征块表示，例如滞后项、循环状态或基于注意力的交互。我们针对此类场景提出了一个分组特征的FDR控制框架。对于分组线性模型，我们构造了具有矩阵值扰动的零对称块级镜像统计量。对于神经序列模型，我们将置换SHAP导数作为模型无关的块级重要性得分，并结合基于核方法的依赖性度量。该框架在各种网络架构上都是模型无关的，不需要指定协变量分布，且当块大小为1时可退化为高斯镜像或神经高斯镜像方法。我们证明了该方法在低维和高维分组设定下的FDR控制性质。

    arXiv:2608.00989v2 Announce Type: replace  Abstract: Most FDR-controlled feature selection methods are designed for coordinate-wise hypotheses, where each feature has a single weight or importance score. This abstraction fails in sequential and grouped models, where one original feature is represented by a block of sub-features, such as lags, recurrent states, or attention-based interactions. We propose a grouped-feature FDR control framework for such settings. For grouped linear models, we construct null-symmetric block-level mirror statistics with matrix-valued perturbations. For neural sequential models, we combine Permutation SHAP derivatives as model-agnostic block-level importance scores with kernel-based dependence measure. The framework is model-agnostic across network architectures, does not require specifying the covariate distribution, and reduces to Gaussian Mirror or Neural Gaussian Mirror when the block size is one. We prove FDR control for low- and high-dimensional group
    
[^29]: 椭圆马氏判别分析中的闭式分数次径向连接函数

    Closed-form fractional radial links for elliptical Mahalanobis discriminant analysis

    [https://arxiv.org/abs/2607.06089](https://arxiv.org/abs/2607.06089)

    本文针对共享径向生成函数的椭圆分布两类分类问题，直接从类内半径分布推导出闭式分数次幂形式的径向连接函数，并证明了该连接的可识别性、估计量的√n一致性与渐近正态性，以及分类器的渐近贝叶斯最优性。

    

    我们研究两类均服从椭圆分布且共享径向生成函数 g 的二分类问题。此时对数似然比是两个平方马氏半径的加性函数，其径向连接为 φ=log g。对于高斯类，φ 是仿射函数，判别规则退化为二次判别分析（QDA）；否则贝叶斯连接并非恒等函数。不同于用样条平滑器拟合该连接，我们从类内半径分布推导出连接函数族，并估计一个有限的分数次幂投影。我们证明了该连接可由半径分布识别；在有限矩条件下，插入式估计量具有 √n 一致性与渐近正态性；且在迭代筛极限下，所得分类器渐近达到贝叶斯最优。对于 d≥2 且类参数非嵌套的情形，恒等连接仅在仿射（高斯）生成函数下才足够。（注：原文摘要末句似有截断）

    arXiv:2607.06089v2 Announce Type: replace-cross  Abstract: We study binary classification when the two classes are elliptical with a shared radial generator $g$. The log-likelihood ratio is then an additive function of the two squared Mahalanobis radii, with radial link $\varphi=\log g$. For Gaussian classes $\varphi$ is affine and the rule reduces to quadratic discriminant analysis (QDA); otherwise the Bayes link is not the identity. Instead of fitting this link with a spline smoother, we derive its family from the within-class law of the radius and estimate a finite fractional-power projection. We prove that the link is identifiable from the radius law, that the plug-in estimator is $\sqrt{n}$-consistent and asymptotically normal under finite-moment conditions, and that the induced classifier is asymptotically Bayes-optimal in an iterated sieve limit. For $d\ge2$ and non-nested class parameters, the identity link suffices only for affine (Gaussian) generators. The bridge, GAM members
    
[^30]: SGD在XOR模型中可证明地优先学习捷径虚假特征

    SGD Provably Prioritizes a Shortcut Spurious Feature in the XOR Model

    [https://arxiv.org/abs/2606.30444](https://arxiv.org/abs/2606.30444)

    本研究首次对两层ReLU神经网络中虚假特征学习进行端到端理论分析，证明SGD会以指数级速度优先学习捷径虚假特征，且较强的虚假分量会抑制信号特征的学习。

    

    众所周知，神经网络容易过度依赖虚假相关性。然而，模型利用捷径特征的精确机制尚未被完全理解，且缓解这种行为的算法依赖于对学习到的表示尚未得到证实的假设。在这项工作中，我们首次对在逻辑损失上通过在线小批量SGD训练的两层ReLU神经网络的虚假特征学习提供了端到端的理论刻画。我们考虑从高维布尔超立方体中抽取的、具有二次信号函数（即XOR）和线性虚假相关性的数据。我们证明SGD会最先学习虚假特征，且学习速度呈指数级快。此外，优化动力学将虚假特征与信号特征耦合在一起，较强的虚假分量会抑制信号特征的学习。我们的分析揭示了学习动力学中精确的相变现象。

    arXiv:2606.30444v3 Announce Type: replace  Abstract: Neural networks are known to be susceptible to over-reliance on spurious correlations. However, the precise mechanism by which models exploit shortcut features is not fully understood, and algorithms to mitigate this behavior rely on as yet unjustified assumptions about the learned representations. In this work, we provide the first end-to-end theoretical characterization of spurious feature learning for two-layer ReLU neural networks trained by online minibatch SGD on the logistic loss. We consider data drawn from the high-dimensional Boolean hypercube with a quadratic signal function (namely XOR) and a linear spurious correlation. We show that SGD learns the spurious feature first, and exponentially fast. Moreover, the optimization dynamics couple the spurious and signal features, with a stronger spurious component inhibiting signal feature learning. Our analysis reveals precise phase transitions in the learning dynamics. In the fi
    
[^31]: Action-BED：基于单重不可解目标的任务驱动贝叶斯实验设计

    Action-BED: Task-Driven Bayesian Experimental Design with Singly Intractable Objectives

    [https://arxiv.org/abs/2606.23662](https://arxiv.org/abs/2606.23662)

    该论文提出ACTION-BED框架，通过以下游动作的期望未来损失重新表述贝叶斯实验设计，将其转化为可用随机梯度对设计策略和动作策略进行联合优化的单重不可解目标，从而无需显式的后验或边缘似然估计。

    

    贝叶斯实验设计（BED）传统上基于最大化从先验到后验的期望不确定性缩减。这种方法的一个主要缺陷在于它会产生难以优化的双重不可解目标，而且针对特定的下游任务进行定制也十分困难。基于第一性原理的决策理论，我们证明了贝叶斯实验设计可以转而用下游动作的期望未来损失（EFL）来表述，从而提供了一个简单且天然任务驱动的框架。关键的是，我们随后证明所有此类EFL都可以重新整理为单重不可解目标，并可以利用随机梯度同时就设计策略和下游动作策略进行联合优化，我们将这一方法称为ACTION-BED。这一表述还避免了对任何显式后验或边缘似然估计的需求，并且天然……（原文摘要至此截断）

    arXiv:2606.23662v2 Announce Type: replace  Abstract: Bayesian experimental design (BED) has traditionally been based on maximising expected uncertainty reductions from prior to posterior. A major shortfall of this approach is that it leads to doubly intractable objectives that are difficult to optimise, while customising them to particular downstream tasks of interest can also be difficult. Following first principles decision theory, we demonstrate that BED can alternatively be formulated in terms of an expected future loss (EFL) on downstream actions, providing a simple and naturally task-driven framework. Critically, we then show that all such EFLs can be rearranged into singly intractable objectives that can be jointly optimised with respect to both the design policy and a downstream action policy using stochastic gradients, an approach we refer to as ACTION-BED. This formulation further sidesteps the need for any explicit posterior or marginal likelihood estimation and is naturally
    
[^32]: 微调后的语言模型展现出更优的信息传递能力

    Fine-Tuned Language Models Exhibit Improved Information Conveyance

    [https://arxiv.org/abs/2605.30844](https://arxiv.org/abs/2605.30844)

    本文提出树冠熵（CE*）这一新度量，从树视角联合刻画语言模型输出长度与生成内容的不确定性，并发现微调后的语言模型具有更优的信息传递能力。

    

    摘要：微调通常被认为会降低大型语言模型的不确定性和多样性，但现有的分析忽略了输出长度这一关键混杂因素，因此无法捕捉不确定性在整个生成过程中的分布情况。为了解决这一问题，我们提出了树冠熵（Canopy Entropy, CE*），该度量从树的视角看待语言生成过程，其中“树冠”代表所有可能生成路径的空间，使得 CE* 能够自然地量化生成空间的有效大小。CE* 同时捕捉输出长度 N 和生成序列 Y_{1:N} 中的不确定性——事实上，我们证明它等于总香农熵 H(N, Y_{1:N}|X)，其中 X 表示提示词。这一表述产生了可解释的指标，包括长度-熵相关项 ρ(N, r_N)，其中 r_N 是熵率，用于量化信息传递效率。

    arXiv:2605.30844v2 Announce Type: replace  Abstract: Fine-tuning is often believed to reduce uncertainty and diversity in large language models, but existing analyses overlook output length, a key confounder, and therefore fail to capture how uncertainty is distributed across an entire generation rollout. To address this, we propose Canopy Entropy ($\mathrm{CE}^\star$), a measure that views language generation from a tree perspective, where "canopy" represents the space of all possible rollouts, making $\mathrm{CE}^\star$ naturally quantify the effective size of the generation space. $\mathrm{CE}^\star$ jointly captures uncertainty in both the output length $N$ and the generated sequence $Y_{1:N}$ -- indeed, we show that it equals total Shannon entropy $H(N, Y_{1:N}\mid X)$, where $X$ denotes the prompt. This formulation yields interpretable metrics, including a length-entropy correlation term $\rho(N, r_N)$, where $r_N$ is the entropy rate, quantifying information conveyance efficienc
    
[^33]: 预算约束下基于多个大语言模型评审器的实例最优估计

    Instance-Optimal Estimation with Multiple LLM Judges on a Budget

    [https://arxiv.org/abs/2605.23362](https://arxiv.org/abs/2605.23362)

    该论文将固定预算下如何在多个成本与可靠性各异的 LLM 评审器及不同难度的评估实例间分配查询这一形式化为异方差多评审估计问题，并分析了逆方差加权估计器、推导出使误差界最小化的最优分配方案。

    

    评估大语言模型越来越依赖于“LLM 即评审”协议，但这类评估成本高昂：不同评审器具有不同的价格和可靠性，且每个提示-响应对的难度差异很大。这就引出了一个基本的分配问题：在固定预算下，应如何在异构的评审器和评估实例之间分配评估查询，以获得最准确的评分估计？我们将该问题形式化为独立无偏评分观测下的预算约束异方差多评审估计问题。给定 K 个提示-响应对和 J 个成本已知但查询-评审方差未知的评审器，我们的目标是在最小化 ℓ_p 误差的同时估计有界的评分向量。我们的第一个贡献是分析了逆方差加权估计器（IVWE），并推导出使其主导误差界最小化的神谕分配方案。由于该分配依赖于（摘要在此处截断）

    arXiv:2605.23362v2 Announce Type: replace-cross  Abstract: Evaluating large language models increasingly relies on LLM-as-a-judge protocols, but such evaluations remain costly: different judges have different prices and reliabilities, and the difficulty of each prompt-response pair can vary substantially. This raises a basic allocation question: under a fixed budget, how should one distribute evaluation queries across heterogeneous judges and instances to obtain the most accurate score estimates? We formalize this question as budgeted heteroskedastic multi-judge estimation under independent, unbiased score observations. Given $K$ prompt-response pairs and $J$ judges with known costs but unknown query-judge variances, our goal is to estimate the bounded score vector while minimizing an $\ell_p$-error. Our first contribution is to analyze the inverse-variance weighted estimator (IVWE) and to derive the oracle allocation that minimizes its leading error bound. Since this allocation depend
    
[^34]: 优化Wasserstein距离估计的计算-统计运行时间

    Optimizing Computational-Statistical Runtime for Wasserstein Distance Estimation

    [https://arxiv.org/abs/2605.20122](https://arxiv.org/abs/2605.20122)

    该论文提出“采样-素描-求解”范式，通过对样本进行规则笛卡尔网格素描，在计算-统计运行时间框架下显著优化了光滑测度之间Wasserstein距离估计的运行时间。

    

    平方Wasserstein距离是度量概率分布之间差异的常用工具。该距离通常在来自两个底层随机样本、大小为 $n$ 的经验测度之间进行计算。遗憾的是，即使在低维欧氏空间问题（$d \in \{2,3\}$）中，具有近似或精确精度保证的Wasserstein距离计算算法，其运行时间作为 $n$ 和所需精度的函数，扩展性也很差。为此，我们考虑计算-统计运行时间，其目标是从样本中估计可能光滑的测度之间的Wasserstein距离，使其在采样期望意义下达到 $\epsilon$-加性误差；我们允许采集一个样本的计算成本为 $O(1)$。为此，我们开发了“采样-素描-求解”范式，其中我们引入了对样本的规则笛卡尔网格素描。我们证明（尤其是u…（原文摘要至此截断）

    arXiv:2605.20122v2 Announce Type: replace  Abstract: Squared Wasserstein distance is a frequently used tool to measure discrepancy between probability distributions. This distance is typically computed between empirical measures of size $n$ from two underlying random samples. Unfortunately, even in lower dimensional Euclidean space problems $\left( d \in \{2,3\} \right)$, algorithms for Wasserstein distance computation with approximate or exact precision guarantees scale poorly in the runtime as a function of $n$ and the desired precision. In response, we consider the computational-statistical runtime, where the goal is to estimate from samples the Wasserstein distance between potentially smooth measures up to $\epsilon$-additive error in expectation with respect to the sampling; we allow $O(1)$ computational cost for collecting a sample. Towards this, we develop a Sample-Sketch-Solve paradigm where we introduce a regular cartesian grid sketch of the samples. We show that (especially u
    
[^35]: 神经算子的平滑分段切割方法以处理不连续性与尖锐过渡

    Smooth Piecewise Cutting for Neural Operator to Handle Discontinuities and Sharp Transitions

    [https://arxiv.org/abs/2605.19823](https://arxiv.org/abs/2605.19823)

    提出 Cut-DeepONet 两阶段训练框架，通过将求解域切割为平滑子区域、并将不连续性表示为高维空间中的边界，使神经算子能够高效处理偏微分方程解中的不连续性与尖锐过渡。

    

    神经算子在学习偏微分方程（PDE）的解算子方面已取得出色的性能，但其固有的连续表示难以捕捉不连续性和尖锐过渡。现有方法通常在连续函数空间内近似此类特征，往往需要更大的模型容量和高分辨率数据。在本工作中，我们提出 Cut-DeepONet，这是一个两阶段训练框架，在显式建模不连续性的同时降低了学习复杂度。我们的方法通过一种提升策略重新表述该问题，将求解域划分为平滑的子区域，同时将不连续性表示为更高维空间中的边界。这种分离使算子学习任务与神经网络的归纳偏置相契合，并避免了直接近似不连续性。此外，一个额外的网络用于预测依赖于输入的不连续性位置，以……

    arXiv:2605.19823v2 Announce Type: replace-cross  Abstract: Neural operators have achieved strong performance in learning solution operators of partial differential equations (PDEs), but their inherently continuous representations struggle to capture discontinuities and sharp transitions. Existing approaches typically approximate such features within continuous function spaces, often requiring increased model capacity and high-resolution data. In this work, we propose Cut-DeepONet, a two-stage training framework that explicitly models discontinuities while reducing learning complexity. Our approach reformulates the problem via a lifting strategy, partitioning the domain into smooth subregions while representing discontinuities as boundaries in a higher-dimensional space. This separation aligns the operator learning task with the inductive bias of neural networks and avoids directly approximating discontinuities. An additional network predicts input-dependent discontinuity locations for 
    
[^36]: 自蒸馏在谱收缩估计器中是最优的

    Self-Distillation is Optimal Among Spectral Shrinkage Estimators

    [https://arxiv.org/abs/2605.17778](https://arxiv.org/abs/2605.17778)

    该论文证明在高维线性回归中，当协方差矩阵具有 $s$ 个尖峰时，$s$ 步自蒸馏在谱收缩估计器类中达到最优预测，且证明 $s$ 步是实现最优性的必要条件。

    

    自蒸馏已成为提升现代机器学习系统模型性能的一种有前景的技术。我们在高维线性回归中建立了自蒸馏的统计学基础，引入并分析了一类广泛的估计器，即谱收缩估计器。我们同时考虑了尖峰协方差矩阵和一般协方差矩阵。对于前者，我们证明当协方差矩阵具有 $s$ 个不同的尖峰时，$s$ 步自蒸馏在谱收缩估计器中实现了最优预测，优于统计学和机器学习中著名的估计器。此外，我们证明 $s$ 步是达到最优性的必要条件：对于 $1 \leq k \leq s$，任何 $(s-k)$ 步的蒸馏估计器都是严格次优的。对于一般协方差矩阵，我们证明随着蒸馏步数的增加，自蒸馏逼近连续谱收缩估计器中的最优极限预测风险。

    arXiv:2605.17778v2 Announce Type: replace-cross  Abstract: Self-distillation has emerged as a promising technique for improving model performance in modern machine learning systems. We develop the statistical foundations of self-distillation in high-dimensional linear regression by introducing and analyzing a broad class of estimators, namely spectral shrinkage estimators. We consider both spiked and general covariance matrices. For the former, we establish that when the covariance has $s$ distinct spikes, $s$-step self-distillation achieves optimal prediction among spectral shrinkage estimators, outperforming well-known estimators in statistics and machine learning. Moreover, we show that $s$ steps are necessary for optimality: any $(s-k)$-step distilled estimator is strictly suboptimal for $1 \leq k \leq s$. For general covariance matrices, we show that self-distillation approaches the optimal limiting prediction risk among continuous spectral shrinkage estimators as the number of di
    
[^37]: NeuroMAS：将多智能体系统视为神经网络并结合联合强化学习

    NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning

    [https://arxiv.org/abs/2605.16757](https://arxiv.org/abs/2605.16757)

    该论文提出 NeuroMAS，将多智能体语言系统视为以 LLM 智能体为节点、文本信号为边的可训练类神经网络架构，通过强化学习让节点自主学会通信、分工与协调，从而将多智能体设计从手工工作流工程转变为可扩展的架构设计。

    

    多智能体语言系统通常被构建为手工设计的工作流，其中智能体被预先分配语义角色，通信协议也事先指定。我们提出 NeuroMAS，这是一种首次将多智能体语言系统视为可训练、可扩展的类神经网络架构的方法，其中 LLM 智能体作为节点，中间文本信号作为边。在 NeuroMAS 中，智能体节点是角色无关但结构感知的：拓扑结构仅决定信息大体上如何流动，而强化学习训练决定节点如何通信、专门化与协调。这一表述将多智能体设计从工作流工程转向架构设计，使深度、宽度、连接性和增长协议成为可扩展的能力来源。此外，我们从理论上说明了当任务可进行层次化分解时，为什么这种模块化的文本计算更具参数效率。

    arXiv:2605.16757v2 Announce Type: replace  Abstract: Multi-agent language systems are often built as hand-designed workflows, where agents are assigned semantic roles and communication protocols are specified in advance. We propose NeuroMAS, a method that first treats a multi-agent language system as a trainable and scalable neural-network-like architecture with LLM agents as nodes and intermediate textual signals as edges. In NeuroMAS, agent nodes are role-free but structure-aware: the topology only determines how information can flow in general, while reinforcement learning training determines how nodes communicate, specialize, and coordinate. This formulation shifts multi-agent design from workflow engineering toward architecture design, where depth, width, connectivity, and growth protocol become scalable sources of capability. Further, we provide a theoretical perspective showing why such modular textual computation is more parameter-efficient when tasks admit hierarchical decompo
    
[^38]: 通过矩匹配分数平滑实现免训练生成式采样

    Training-Free Generative Sampling via Moment-Matched Score Smoothing

    [https://arxiv.org/abs/2605.14276](https://arxiv.org/abs/2605.14276)

    提出了一种免训练的交互粒子采样器MM-SOLD，通过在整个采样轨迹中强制匹配目标矩，从理论上证明其收敛到与训练GMM均值和协方差一致的吉布斯-玻尔兹曼分布，即使数据秩亏也成立。

    

    扩散模型通过沿扰动目标分布的分数进行去噪来生成样本。在实践中，需要训练一个神经扩散模型，这在计算上代价高昂。近期研究表明，分数匹配会隐式地平滑经验分数，而这种平滑偏差通过捕捉低维数据几何结构来促进泛化。我们提出了矩匹配分数平滑过阻尼朗之万动力学（MM-SOLD），这是一种免训练的交互粒子采样器，可在整个采样轨迹中强制执行目标矩。我们证明，随着粒子数量的增长，理想约束吉布斯平衡的单粒子边缘分布收敛到通过对朴素分数平滑目标进行指数倾斜而得到的吉布斯-玻尔兹曼密度。其均值和协方差与训练所用的GMM相匹配，即使对于秩亏数据也是如此。我们还建立了采样一致性以及针对目标分布的统计误差界（摘要此处被截断）。

    arXiv:2605.14276v2 Announce Type: replace  Abstract: Diffusion models generate samples by denoising along the score of a perturbed target distribution. In practice, one trains a neural diffusion model, which is computationally expensive. Recent work suggests that score matching implicitly smooths the empirical score, and that this smoothing bias promotes generalization by capturing low-dimensional data geometry. We propose moment-matched score-smoothed overdamped Langevin dynamics (MM-SOLD), a training-free interacting particle sampler that enforces the target moments throughout the sampling trajectory. We prove that the one-particle marginal of the ideal constrained Gibbs equilibrium converges, as the number of particles grows, to a Gibbs--Boltzmann density obtained by exponentially tilting the naive score-smoothed target. Its mean and covariance match the training GMM, even for rank-deficient data. We additionally establish sampling consistency, statistical target-error bounds for bo
    
[^39]: 核心集诱导的条件速度流匹配

    Coreset-Induced Conditional Velocity Flow Matching

    [https://arxiv.org/abs/2605.12951](https://arxiv.org/abs/2605.12951)

    该论文提出CCVFM生成模型，通过熵Sinkhorn核心集构建闭式代理源替代分层整流流中内部流的高斯噪声源，使条件速度规律成为无需神经采样器即可采样的闭式高斯混合，并只需轻量级校正流即可完成训练。

    

    我们提出了核心集诱导的条件速度流匹配（CCVFM），这是一种生成模型，它通过由数据信息构建的源分布来增强分层整流流。标准流匹配仅回归速度的条件均值；而分层流匹配则通过速度空间中的第二个内部流来建模完整的条件速度规律，但这个内部流需要从头开始将各向同性高斯噪声传输到多峰的目标速度分布。我们的关键观察是：这个内部源可以被一个由目标核心集构建的闭式代理分布所替代。CCVFM首先使用熵正则化的Sinkhorn核心集将目标压缩为加权原子，并将其提升为高斯混合。由此诱导的条件速度规律是一个闭式高斯混合，无需学习神经采样器即可进行采样。基于该代理源训练的轻量级校正流，t...

    arXiv:2605.12951v3 Announce Type: replace  Abstract: We propose Coreset-Induced Conditional Velocity Flow Matching (CCVFM), a generative model that augments hierarchical rectified flow with a data-informed source distribution. Standard flow matching regresses only the conditional mean of the velocity; hierarchical flow matching instead models the full conditional velocity law with a second, inner flow in velocity space, but this inner flow is asked to transport isotropic Gaussian noise to a multimodal target velocity distribution from scratch. Our key observation is that this inner source can be replaced by a closed-form surrogate built from a coreset of the target. CCVFM first compresses the target into weighted atoms using an entropic Sinkhorn coreset and lifts them to a Gaussian mixture. The induced conditional velocity law is then a closed-form Gaussian mixture that can be sampled without a learned neural sampler. A lightweight correction flow, trained from this surrogate source, t
    
[^40]: 通过规范扩散测量与分解模态分离

    Measuring and Decomposing Mode Separation via the Canonical Diffusion

    [https://arxiv.org/abs/2605.08777](https://arxiv.org/abs/2605.08777)

    该论文提出以密度固有的唯一常数扩散系数可逆扩散过程（规范扩散）为基准，从其自协方差矩阵中提取量来测量并分解高维分布的模态分离程度，弥补了微分熵、PCA 等无标签工具无法刻画势垒分隔结构、而互信息又依赖潜在标签的不足。

    

    模态分离，即一个分布被势垒分隔成簇的锐利程度，是密度的基本几何属性，在高维空间中难以量化。模态往往是数据中有意义的选项，例如分子的构象，或生成模型输出中不同的语义类型。模态分离不同于离散度，而后者正是标准无标签工具所捕获的内容：微分熵随散布程度增大而上升，与是否碎裂无关；PCA 按方差对方向排序，与势垒无关。互信息确实能够捕获模态分离，但前提是给定潜在的成分标签，而这通常是不可获得的。我们通过密度所固有的单一随机过程来测量模态分离，即以 $f$ 为平稳分布且扩散系数为常数的唯一可逆扩散过程（规范扩散）。从其自协方差矩阵中，我们提取出两个……

    arXiv:2605.08777v2 Announce Type: replace  Abstract: Mode separation, namely how sharply a distribution fragments into barrier-separated clusters, is a fundamental geometric property of densities, difficult to quantify in high dimensions. Modes are often the meaningful alternatives in the data, such as the conformations of a molecule or the distinct semantic types among a generative model's outputs. Mode separation is distinct from dispersion, which is what standard label-free tools capture: differential entropy rises with spread regardless of fragmentation, and PCA orders directions by variance regardless of barriers. Mutual information does capture it, but only given the latent component labels, which are usually unavailable. We measure mode separation through a single stochastic process intrinsic to the density, namely the unique reversible diffusion with $f$ as its stationary distribution and a constant scalar diffusion coefficient. From its autocovariance matrix we extract two rea
    
[^41]: 判别式因子分解下的黑盒模型分类

    Black-box model classification under the discriminative factorization

    [https://arxiv.org/abs/2605.07878](https://arxiv.org/abs/2605.07878)

    提出判别式因子分解框架，用于区分黑盒模型级分类中查询集的质量优劣，理论上证明随机分类概率随查询预算呈指数衰减，并实证表明据此估计参数可预测性能衰减率并指导选取高质量查询集。

    

    对现代生成式系统的访问通常仅限于通过API进行查询（即“黑盒”设置），并且在推理时用户对该系统的许多属性一无所知。最近的研究表明，基于模型对一组查询的嵌入响应之间关系所构建的模型低维表示，对于推断模型层面的属性非常有用。然而，这些表示的质量对查询集高度敏感。我们引入了“判别式因子分解”，用于在黑盒模型级分类的情境下区分高质量和低质量的查询集。在该框架下，随机水平分类的概率被一个随查询预算呈指数衰减的项所界定。在三个审计任务中，估计出的因子分解参数能够预测经验性能的衰减率。最后，我们展示了使用估计的因子分解（参数）所选出的查询集……

    arXiv:2605.07878v2 Announce Type: replace-cross  Abstract: Access to modern generative systems is often restricted to querying an API (the ``black-box'' setting) and many properties of the system are unknown to the user at inference time. Recent work has shown that low-dimensional representations of models based on the relationship of their embedded responses to a set of queries are useful for inferring model-level properties. However, the quality of these representations is highly sensitive to the query set. We introduce the \emph{discriminative factorization} to distinguish between high- and low-quality query sets in the context of black-box model-level classification. Under this framework, the probability of chance-level classification is bounded by a term that decays exponentially in the query budget. On three auditing tasks, estimated factorization parameters predict the empirical performance decay rate. We conclude by showing that query sets selected using the estimated factoriza
    
[^42]: 集成分布鲁棒贝叶斯优化

    Ensemble Distributionally Robust Bayesian Optimisation

    [https://arxiv.org/abs/2605.07565](https://arxiv.org/abs/2605.07565)

    提出了一种计算上可处理的集成分布鲁棒贝叶斯优化算法，能够处理连续上下文并获得次线性遗憾界，改进了当前最先进的理论结果。

    

    我们研究了上下文分布不确定性下的零阶优化问题，这一设置通常使用贝叶斯优化（BO）来解决。使BO对数据复杂性和噪声特性更具鲁棒性的一种普遍策略是采用集成模型作为代理模型，从而弥补任何单一模型的弱点。在本研究中，我们提出了一种新颖的集成分布鲁棒贝叶斯优化算法，该算法在处理连续上下文的同时保持计算上的可处理性。我们获得了理论上的次线性遗憾界，改进了当前最先进的结果。我们证明了该方法的实证表现与其理论保证相一致。

    arXiv:2605.07565v3 Announce Type: replace-cross  Abstract: We study zeroth-order optimisation under context distributional uncertainty, a setting commonly tackled using Bayesian optimisation (BO). A prevailing strategy to make BO more robust to the complex and noisy nature of data is to employ an ensemble as the surrogate model, thereby mitigating the weaknesses of any single model. In this study, we propose a novel algorithm for Ensemble Distributionally Robust Bayesian Optimisation that remains computationally tractable while managing continuous context. We obtain theoretical sublinear regret bounds, improving current state-of-the-art results. We show that our method's empirical behaviour aligns with its theoretical guarantees.
    
[^43]: 训练后深度Transformer的谱自适应泛化界

    Spectrum-Adaptive Generalization Bounds for Trained Deep Transformers

    [https://arxiv.org/abs/2605.07297](https://arxiv.org/abs/2605.07297)

    该论文提出了针对训练后多层Transformer的谱自适应事后泛化界，通过在训练后逐层、逐矩阵类型灵活选择Schatten指数，在谱复杂性与维度、深度依赖之间实现自适应权衡，克服了现有方法需事先固定范数约束且对深度呈指数依赖的缺陷。

    

    理解训练后的Transformer为何具有良好的泛化能力是现代机器学习理论中的一个基本问题，而基于复杂度的泛化界为研究这一问题提供了一种有原则的方法。尽管现有的基于范数的Transformer泛化界消除了对隐藏维度的显式多项式依赖，但它们通常强加事先指定的固定范数约束，并且可能对深度表现出不利的指数依赖。在本文中，我们为多层Transformer推导了谱自适应的事后泛化界。在逐层谱范数控制下，这些界以查询-键矩阵、值矩阵和前馈权重矩阵的逐层Schatten量来表示。由于Schatten指数无需事先固定，而是可以在训练后针对每种矩阵类型和每一层分别选择，因此这些界能够在谱复杂性与维度和深度依赖之间进行自适应权衡。

    arXiv:2605.07297v2 Announce Type: replace  Abstract: Understanding why trained Transformers generalize well is a fundamental problem in modern machine learning theory, and complexity-based generalization bounds provide a principled way to study this question. While existing norm-based bounds for Transformers remove the explicit polynomial dependence on the hidden dimension, they typically impose fixed norm constraints specified a priori and can exhibit unfavorable exponential dependence on depth. In this paper, we derive spectrum-adaptive post hoc generalization bounds for multi-layer Transformers. Under layerwise spectral norm control, the bounds are expressed in terms of layerwise Schatten quantities of the query-key, value, and feedforward weight matrices. Since the Schatten indices need not be fixed a priori and can instead be selected after training, separately for each matrix type and layer, the bounds adaptively trade off spectral complexity against the dimension- and depth-depe
    
[^44]: 基于结构化随机扩散的超图生成

    Hypergraph Generation via Structured Stochastic Diffusion

    [https://arxiv.org/abs/2605.05024](https://arxiv.org/abs/2605.05024)

    提出HEDGE生成模型，通过在关联矩阵上进行结合超图双侧热算子与OU过程的结构化随机扩散，并利用闭式条件目标学习置换等变的逆向漂移场，实现对高阶交互、超边异质性和重叠结构的忠实超图生成。

    

    超图用于建模高阶交互，但现实中超图的生成仍然困难，因为关联结构、超边规模的异质性以及重叠结构无法通过成对约简被忠实地刻画。我们提出了HEDGE，一种直接定义在松弛关联矩阵上的生成模型，通过结构化随机扩散实现。其前向过程将超图特有的双侧热算子与Ornstein-Uhlenbeck分量相结合，在数据附近保持结构感知的加噪，同时给出显式的高斯终端分布。在给定观测超图的条件下，该前向过程是线性高斯的，因此条件均值、协方差、得分以及反向漂移目标均可获得闭式解。因此，我们通过向精确的条件目标回归，在关联空间中学习一个置换等变、仅依赖状态的逆向漂移场，并通过模拟学习到的逆时间随机微分方程来生成样本。

    arXiv:2605.05024v2 Announce Type: replace  Abstract: Hypergraphs model higher-order interactions, but realistic hypergraph generation remains difficult because incidence, hyperedge-size heterogeneity, and overlap structure are not faithfully captured by pairwise reductions. We propose \HEDGE, a generative model defined directly on relaxed incidence matrices via a structured stochastic diffusion. The forward process combines a hypergraph-specific two-sided heat operator with an Ornstein--Uhlenbeck component, preserving structure-aware noising near the data while yielding an explicit Gaussian terminal law. Conditional on an observed hypergraph, this forward process is linear-Gaussian, so conditional means, covariances, scores, and reverse-drift targets are available in closed form. We therefore learn a permutation-equivariant state-only reverse-drift field in incidence space by regressing onto exact conditional targets, and generate samples by simulating a learned reverse-time SDE from t
    
[^45]: 不完美检测下的稀疏网络推断及其在生态网络中的应用

    Sparse Network Inference under Imperfect Detection and its Application to Ecological Networks

    [https://arxiv.org/abs/2604.18820](https://arxiv.org/abs/2604.18820)

    该论文提出非负低秩分解框架，通过 ℓ_{1/2} 正则化在联合估计检测概率的同时促进生态二部网络中群体内与跨群体图的稀疏性，并设计了自适应惩罚的 ADMM 算法来求解该非凸非光滑问题。

    

    从计数数据中恢复潜在结构在网络推断领域受到了相当大的关注，尤其是当人们希望同时在二部网络中获取跨群体交互作用和群体内相似性模式时——这类网络在生态学研究中被广泛使用。此类网络通常较为稀疏且易受不完美检测的影响。现有模型主要聚焦于交互关系的恢复，而对由此产生的相似性图的研究则少得多。此外，针对计数不足数据的现有模型即使考虑稀疏性，也只是将其施加于检测模型而非这些图上。为解决这些问题，我们提出了一个非负低秩分解框架，该框架联合估计检测概率，并通过 ℓ_{1/2} 正则化促进群体内与跨群体图的稀疏性。针对由此产生的非凸、非光滑问题，我们开发了一种带自适应惩罚的 ADMM 算法，并建立了渐近可行性与 KKT 条件……（原文摘要在此处截断）

    arXiv:2604.18820v3 Announce Type: replace  Abstract: Recovering latent structure from count data has received considerable attention in network inference, particularly when one seeks both cross-group interactions and within-group similarity patterns in bipartite networks, which are widely used in ecology research. Such networks are often sparse and prone to imperfect detection. Existing models mainly focus on interaction recovery, while the induced similarity graphs are much less studied. Moreover, existing models for under-counted data place sparsity, if at all, on the detection model rather than on these graphs. To address these issues, we propose a nonnegative low-rank factorization framework that jointly estimates detection probabilities and promotes sparse within- and cross-group graphs through $\ell_{1/2}$ regularization. For the resulting nonconvex, nonsmooth problem, we develop an ADMM-based algorithm with adaptive penalization, and establish asymptotic feasibility and KKT stat
    
[^46]: 基于强化学习分块的动态分词：端到端训练与零样本迁移

    Dynamic Tokenization via Reinforcement Patching: End-to-end Training and Zero-shot Transfer

    [https://arxiv.org/abs/2603.26097](https://arxiv.org/abs/2603.26097)

    ReinPatch 利用强化学习（基于组相对策略梯度 GRPG）将分块边界的选取形式化为离散决策过程，联合优化动态分块策略与下游序列骨干模型，无需连续松弛即可实现端到端训练并支持零样本迁移。

    

    高效地聚合空间或时间范围以获得紧凑表示已成为现代深度学习模型的一个统一原则，然而为长程序列数据（尤其是时间序列这类连续序列）学习数据自适应的表示仍然是一个开放性挑战。虽然固定大小的分块方法提升了可扩展性和性能，但以端到端方式发现可变大小的、数据驱动的分块，往往迫使模型依赖软离散化、特定骨干网络或启发式规则。在这项工作中，我们提出了强化分块，这是一种新颖的框架，通过强化学习联合优化序列分块策略及其下游的序列骨干模型。通过将分块边界的放置形式化为一个由组相对策略梯度优化的离散决策过程，ReinPatch 无需连续松弛即可进行动态分块策略优化……

    arXiv:2603.26097v2 Announce Type: replace-cross  Abstract: Efficiently aggregating spatial or temporal horizons to acquire compact representations has become a unifying principle in modern deep learning models, yet learning data-adaptive representations for long-horizon sequence data, especially continuous sequences like time series, remains an open challenge. While fixed-size patching has improved scalability and performance, discovering variable-sized, data-driven patches end-to-end often forces models to rely on soft discretization, specific backbones, or heuristic rules. In this work, we propose Reinforcement Patching (ReinPatch), a novel framework that jointly optimizes a sequence patching policy and its downstream sequence backbone model with reinforcement learning. By formulating patch boundary placement as a discrete decision process optimized via Group Relative Policy Gradient (GRPG), ReinPatch bypasses the need for continuous relaxations and performs dynamic patching policy o
    
[^47]: CausalCompass：评估时序因果发现在模型设定错误场景下的鲁棒性

    CausalCompass: Evaluating the Robustness of Time-Series Causal Discovery in Misspecified Scenarios

    [https://arxiv.org/abs/2602.07915](https://arxiv.org/abs/2602.07915)

    提出了 CausalCompass——一个灵活可扩展的基准测试框架，通过八种假设违反场景系统评估时序因果发现方法的鲁棒性，并发现基于深度学习的方法在大多数场景下整体表现最优。

    

    从时间序列中进行因果发现是机器学习中的一项基础任务。然而，其广泛应用受到两方面的阻碍：一是对不可检验的因果假设的依赖，二是现有基准测试中缺乏面向鲁棒性的评估。为应对这些挑战，我们提出了 CausalCompass，这是一个灵活且可扩展的基准测试框架，旨在评估时序因果发现（TSCD）方法在建模假设被违反情况下的鲁棒性。为展示 CausalCompass 的实际效用，我们在八种假设违反场景下对具有代表性的 TSCD 算法进行了广泛的基准测试。实验结果表明，没有任何单一方法能够在所有设置下始终取得最优性能。尽管如此，在大多数场景中，表现出更优整体性能的是基于深度学习的方法。我们进一步提供了超参数敏感性分析，以加深对该问题的理解。

    arXiv:2602.07915v3 Announce Type: replace-cross  Abstract: Causal discovery from time series is a fundamental task in machine learning. However, its widespread adoption is hindered by a reliance on untestable causal assumptions and by the lack of robustness-oriented evaluation in existing benchmarks. To address these challenges, we propose CausalCompass, a flexible and extensible benchmark framework designed to assess the robustness of time-series causal discovery (TSCD) methods under violations of modeling assumptions. To demonstrate the practical utility of CausalCompass, we conduct extensive benchmarking of representative TSCD algorithms across eight assumption-violation scenarios. Our experimental results indicate that no single method consistently attains optimal performance across all settings. Nevertheless, in most scenarios, the methods exhibiting superior overall performance are deep learning-based approaches. We further provide hyperparameter sensitivity analyses to deepen th
    
[^48]: 基于文本证据的服务系统设计

    Designing Service Systems from Textual Evidence

    [https://arxiv.org/abs/2601.21471](https://arxiv.org/abs/2601.21471)

    本研究将基于文本证据的服务系统设计问题形式化为固定置信度下的最佳臂识别问题，利用大语言模型生成的低成本代理评分并结合有选择性的人工审计来校正偏差，从而以最小的人工审计成本高置信度地识别出最优服务配置。

    

    在服务系统设计中，系统性能的主要证据往往以文本形式呈现，例如客户支持对话记录、投诉叙述和合规审查报告。随着大语言模型（LLM）的日益广泛应用，这些文本轨迹可以被转化为低成本的代理评分，从而支持对候选服务配置的比较和优化。然而，基于大语言模型的评估可能存在系统性偏差，需要通过人工审计进行纠正。为此，本研究探讨如何在最小化昂贵人工审计使用量的同时，以高置信度识别出最佳的服务配置。我们将该问题形式化为固定置信度下的最佳臂识别问题，其中包含始终可观测但可能存在偏差的代理评分，以及有选择性获取的人工验证结果。此外，多个服务配置可以在同一案例上运行。人工审计可以提供案例标签……（原文摘要至此处被截断）

    arXiv:2601.21471v2 Announce Type: replace-cross  Abstract: In service system design, it is often the case that the main evidence of system performance appears in textual form, such as customer support transcripts, complaint narratives, and compliance review reports. With the increasing use of large language models (LLMs), these textual trajectories can be converted into low-cost proxy scores that support comparison and optimization of candidate service configurations. However, LLM-based evaluations can exhibit systematic bias and require human audits for correction. To this end, this research studies how to identify the best service configuration with high confidence while minimizing the use of expensive human audits. We formulate the problem as fixed-confidence best arm identification with an always-observed, potentially biased proxy and selectively acquired human-verified outcomes. Moreover, several configurations can run on one case. A human audit can provide a case label that score
    
[^49]: 整流流的最优阶样本复杂度

    Order-Optimal Sample Complexity of Rectified Flows

    [https://arxiv.org/abs/2601.20250](https://arxiv.org/abs/2601.20250)

    本文证明了整流流模型在标准神经网络假设下可达到 $\tilde{O}(\varepsilon^{-2})$ 的最优阶样本复杂度，改进了流匹配模型已有的 $O(\varepsilon^{-4})$ 界并匹配均值估计的最优速率。

    

    近年来，基于流的生成模型相比扩散模型展现出了更优的效率。本文研究整流流模型，该模型约束从基础分布到数据分布的传输轨迹为线性。这种结构性限制极大地加速了采样过程，通常仅需单步欧拉步即可实现高质量生成。在用于参数化速度场和数据分布的神经网络类的标准假设下，我们证明整流流可以达到 $\tilde{O}(\varepsilon^{-2})$ 的样本复杂度。这改进了流匹配模型已知的最佳 $O(\varepsilon^{-4})$ 界，并达到了均值估计的最优速率。我们的分析利用了整流流的特殊结构：由于模型是沿线性路径以平方损失进行训练的，相关的假设类具有严格可控的局部化 Rademacher 复杂度。

    arXiv:2601.20250v2 Announce Type: replace  Abstract: Recently, flow-based generative models have shown superior efficiency compared to diffusion models. In this paper, we study rectified flow models, which constrain transport trajectories to be linear from the base distribution to the data distribution. This structural restriction greatly accelerates sampling, often enabling high-quality generation with a single Euler step. Under standard assumptions on the neural network classes used to parameterize the velocity field and data distribution, we prove that rectified flows achieve sample complexity $\tilde{O}(\varepsilon^{-2})$. This improves on the best known $O(\varepsilon^{-4})$ bounds for flow matching model and matches the optimal rate for mean estimation. Our analysis exploits the particular structure of rectified flows: because the model is trained with a squared loss along linear paths, the associated hypothesis class admits a sharply controlled localized Rademacher complexity. T
    
[^50]: 度校正块模型下社区检测的矩阵分解框架

    Matrix Factorization Framework for Community Detection under the Degree-Corrected Block Model

    [https://arxiv.org/abs/2601.06262](https://arxiv.org/abs/2601.06262)

    该论文将度校正块模型（DCBM）的推断重新表述为带约束的非负矩阵三因子分解问题，据此提出了一种具有理论初始化、可扩展且适用于任意DCBM网络的社区检测方法。

    

    社区检测是数据分析中的一项基础任务，块模型为识别多样化的社区结构提供了一种途径。度校正块模型（DCBM）是一种成熟的模型，能够刻画节点度的异质性。然而，其推断方法计算代价高昂且对初始化敏感，而谱方法或基于模块度的方法等更经济的替代方案，在恢复某些非同配（non-assortative）结构时可能效果不佳。在本工作中，我们证明DCBM推断可以被重新表述为一个带约束的非负矩阵三因子分解问题。借助这一洞察，我们提出了一种可扩展的社区检测方法，该方法具有理论支撑的初始化，可为推断算法提供初始划分。我们的方法与网络结构无关，适用于任何可用DCBM表示的图。在合成网络与真实世界基准网络上的实验表……（原文在此处截断）

    arXiv:2601.06262v3 Announce Type: replace-cross  Abstract: Community detection is a fundamental task in data analysis, and block models provide an approach for identifying diverse community structures. The degree-corrected block model (DCBM) is an established model that captures node degree heterogeneity. However, inference methods are computationally costly and sensitive to initialization, while cheaper alternatives, such as spectral or modularity-based approaches, may be less effective at recovering certain non-assortative structures. In this work, we show that DCBM inference can be reformulated as a constrained nonnegative matrix trifactorization problem. Leveraging this insight, we propose a scalable community detection method with a theoretically grounded initialization providing an initial partition for inference algorithms. Our approach is agnostic to network structure and applies to any graph representable by a DCBM. Experiments on synthetic and real-world benchmark networks sh
    
[^51]: 离群值鲁棒的多输出高斯过程

    Outlier-Robust Multi-Output Gaussian Processes

    [https://arxiv.org/abs/2510.26401](https://arxiv.org/abs/2510.26401)

    本文提出R-MOGP，一种对任意单个响应中的离群值具有可证明鲁棒性的多输出高斯过程，其在保持共轭性与计算效率的同时可与稀疏变分推断结合，以远低于现有最先进方法的计算成本达到相当的性能。

    

    多输出高斯过程（MOGP）回归能够对多个相关响应变量之间的依赖关系进行建模。与标准高斯过程（GP）类似，MOGP对模型误设和离群值较为敏感，这些离群值可能会扭曲各个输出内部的预测。在多输出的情况下，这一问题可能进一步加剧，因为某一响应中的异常观测可能会对其他相关输出的预测产生不利影响。为应对这一问题，我们提出了R-MOGP，一种对任何单个响应中的离群值均具有可证明鲁棒性的多输出高斯过程。我们的方法保持了共轭性，在不牺牲计算效率的前提下实现了鲁棒性，并可与稀疏变分推断自然结合以实现可扩展性。实验表明，R-MOGP以现有最先进鲁棒MOGP的一小部分计算成本，取得了与之相当的性能。

    arXiv:2510.26401v2 Announce Type: replace  Abstract: Multi-output Gaussian process (MOGP) regression allows modelling dependencies among multiple correlated response variables. Similarly to standard Gaussian processes (GPs), MOGPs are sensitive to model misspecification and outliers, which can distort predictions within individual outputs. In the multi-output case, this situation can be further exacerbated as anomalous observations in one response can adversely influence predictions for other correlated outputs. To handle this situation, we propose R-MOGP, an MOGP that is provably robust to outliers in any individual response. Our approach preserves conjugacy, achieving robustness without sacrificing computational efficiency, and integrates naturally with sparse variational inference for scalability. Empirically, we show that R-MOGP achieves performance comparable to existing state-of-the-art robust MOGPs at a fraction of their computational costs.
    
[^52]: Transformer 的最优上下文自适应性与分布鲁棒性

    Optimal In-context Adaptivity and Distributional Robustness of Transformers

    [https://arxiv.org/abs/2510.23254](https://arxiv.org/abs/2510.23254)

    本文证明了在多种任务难度混合分布上预训练的 Transformer，即使面对存在分布偏移的固定难度测试分布，也能达到最优收敛速率，展现出最优的上下文自适应性与分布鲁棒性。

    

    我们研究上下文学习问题，其中一个 Transformer 在从混合分布 $\pi=\sum_{\alpha\in\mathcal{A}} \lambda_{\alpha} \pi_{\alpha}$（称为预训练先验）中抽取的任务上进行预训练，其中每个混合分量 $\pi_{\alpha}$ 是由 $\alpha$ 索引的特定难度级别任务上的分布。我们的目标是理解该预训练 Transformer 在面对不同测试分布 $\mu$ 进行评估时的性能，该测试分布由固定难度 $\beta\in\mathcal{A}$ 的任务组成，且相对于 $\pi_\beta$ 可能存在分布偏移，其卡方散度满足 $\chi^2(\mu,\pi_{\beta})\le\kappa$。特别地，我们考虑具有随机光滑性的非参数回归问题，以及同时具有随机光滑性和随机有效维数的多指标模型。我们证明，在充足数据上预训练的大型 Transformer 达到了最优收敛速率……

    arXiv:2510.23254v5 Announce Type: replace  Abstract: We study in-context learning problems where a Transformer is pretrained on tasks drawn from a mixture distribution $\pi=\sum_{\alpha\in\mathcal{A}} \lambda_{\alpha} \pi_{\alpha}$, called the pretraining prior, in which each mixture component $\pi_{\alpha}$ is a distribution on tasks of a specific difficulty level indexed by $\alpha$. Our goal is to understand the performance of the pretrained Transformer when evaluated on a different test distribution $\mu$, consisting of tasks of fixed difficulty $\beta\in\mathcal{A}$, and with potential distribution shift relative to $\pi_\beta$, subject to the chi-squared divergence $\chi^2(\mu,\pi_{\beta})$ being at most $\kappa$. In particular, we consider nonparametric regression problems with random smoothness, and multi-index models with both random smoothness and random effective dimension. We prove that a large Transformer pretrained on sufficient data achieves the optimal rate of convergen
    
[^53]: 闭式解最后一层优化

    Closed-Form Last Layer Optimization

    [https://arxiv.org/abs/2510.04606](https://arxiv.org/abs/2510.04606)

    该论文提出在平方损失下将神经网络最后一层权重视为主干参数的闭式解函数，仅优化主干参数并在随机梯度下降设定下进行适配，理论上证明了该方法在神经正切核机制下收敛到最优解且优于标准SGD。

    

    神经网络通常使用随机梯度下降的各种变体进行优化。然而，在平方损失下，线性最后一层权重的最优解是已知的闭式解。我们提出在优化过程中利用这一点，将最后一层视为主干网络参数的函数，仅针对这些参数进行优化。我们证明这等价于在主干网络上进行梯度下降步骤与对最后一层进行闭式更新之间的交替。我们针对随机梯度下降的设定对该方法进行了适配，通过在当前批次上的损失与先前批次积累的信息之间进行权衡。我们提供了理论分析，表明该方法在神经正切核机制下收敛到最优解，并通过一步分析量化了与标准SGD相比的增益。最后，我们证明了我们的方法与SGD相比的有效性。

    arXiv:2510.04606v3 Announce Type: replace-cross  Abstract: Neural networks are typically optimized with variants of stochastic gradient descent. Under a squared loss, however, the optimal solution to the linear last layer weights is known in closed-form. We propose to leverage this during optimization, treating the last layer as a function of the backbone parameters, and optimizing solely for these parameters. We show this is equivalent to alternating between gradient descent steps on the backbone and closed-form updates on the last layer. We adapt the method for the setting of stochastic gradient descent, by trading off the loss on the current batch against the accumulated information from previous batches. We provide theoretical analyses showing convergence of the method to an optimal solution in the neural tangent kernel regime, as well as quantifying the gains compared to standard SGD in a one-step analysis. Finally, we demonstrate the effectiveness of our approach compared with SG
    
[^54]: 基于熵门控对比融合的鲁棒多模态学习

    Robust Multimodal Learning via Entropy-Gated Contrastive Fusion

    [https://arxiv.org/abs/2505.15417](https://arxiv.org/abs/2505.15417)

    AECF是一种轻量级即插即用的熵门控对比融合层，通过逐实例自适应熵系数、跨模态子集的单调校准以及熵驱动的课程掩码，在输入缺失场景下同时提升多模态推理的鲁棒性与校准性。

    

    现实中的多模态系统经常面临输入缺失的场景，例如工厂中的机器人失去音频输入，或临床记录在推理时缺少化验检测结果。标准的融合层要么保持鲁棒性，要么保持校准性，但无法兼顾两者。我们提出了自适应熵门控对比融合（AECF），这是一个轻量级的单一层，它：（i）针对每个实例自适应调整熵系数；（ii）在所有模态子集上强制实现单调校准；（iii）直接利用训练时的熵驱动课程式掩码。在AV-MNIST和MS-COCO数据集上，AECF在50%输入丢弃率下将掩码输入的mAP提升18个百分点，同时将期望校准误差（ECE）最多降低200%，且仅增加1%的运行时间。所有骨干网络保持冻结，使AECF成为一个即插即用的层，可用于鲁棒且校准良好的多模态推理。

    arXiv:2505.15417v2 Announce Type: replace  Abstract: Real-world multimodal systems routinely face missing-input scenarios, and in reality, robots lose audio in a factory or a clinical record omits lab tests at inference time. Standard fusion layers either preserve robustness or calibration but never both. We introduce Adaptive Entropy-Gated Contrastive Fusion (AECF), a single light-weight layer that (i) adapts its entropy coefficient per instance, (ii) enforces monotone calibration across all modality subsets, and (iii) drives a curriculum mask directly from training-time entropy. On AV-MNIST and MS-COCO, AECF improves masked-input mAP by +18 pp at a 50% drop rate while reducing ECE by up to 200%, yet adds 1% run-time. All back-bones remain frozen, making AECF an easy drop-in layer for robust, calibrated multimodal inference.
    
[^55]: 椭球上的线性老虎机：极小极大最优算法

    Linear Bandits on Ellipsoids: Minimax Optimal Algorithms

    [https://arxiv.org/abs/2502.17175](https://arxiv.org/abs/2502.17175)

    本文针对动作集为椭球体的线性随机老虎机问题，首次提出极小极大最优算法，通过推导新型信息论遗憾下界，并设计一种结合参数范数估计与探索-承诺策略的非经典（非乐观、非采样）高效算法来实现与该下界匹配。

    

    我们考虑动作集为椭球体的线性随机老虎机问题。我们提供了该问题的首个已知的极小极大最优算法。我们首先推导出任意算法遗憾值的新型信息论下界，该下界必须至少为 $\Omega(\min(d \sigma \sqrt{T} + d \|\theta\|_{A}, \|\theta\|_{A} T))$，其中 $d$ 是维度，$T$ 是时间范围，$\sigma^2$ 是噪声方差，$A$ 是定义动作集的矩阵，$\theta$ 是未知参数向量。然后我们提供了一种算法，其遗憾值与该下界在一个乘性通用常数范围内匹配。该算法是非经典的，因为它既不是乐观算法，也不是采样算法。其主要思想是结合一种新颖的顺序过程来估计 $\|\theta\|$，随后采用基于该估计的探索-承诺策略。该算法具有很高的计算效率，一次运行仅需时间 $O$……

    arXiv:2502.17175v2 Announce Type: replace  Abstract: We consider linear stochastic bandits where the set of actions is an ellipsoid. We provide the first known minimax optimal algorithm for this problem. We first derive a novel information-theoretic lower bound on the regret of any algorithm, which must be at least $\Omega(\min(d \sigma \sqrt{T} + d \|\theta\|_{A}, \|\theta\|_{A} T))$ where $d$ is the dimension, $T$ the time horizon, $\sigma^2$ the noise variance, $A$ a matrix defining the set of actions and $\theta$ the vector of unknown parameters. We then provide an algorithm whose regret matches this bound to a multiplicative universal constant. The algorithm is non-classical in the sense that it is not optimistic, and it is not a sampling algorithm. The main idea is to combine a novel sequential procedure to estimate $\|\theta\|$, followed by an explore-and-commit strategy informed by this estimate. The algorithm is highly computationally efficient, and a run requires only time $O
    
[^56]: 从时间快照中识别漂移、扩散与因果结构

    Identifying Drift, Diffusion, and Causal Structure from Temporal Snapshots

    [https://arxiv.org/abs/2410.22729](https://arxiv.org/abs/2410.22729)

    该论文提出了首个仅凭时间边缘分布（无需观测个体轨迹）即可联合识别随机微分方程漂移与扩散项的方法，证明了不可辨识性仅在初始分布具有广义旋转对称性时出现，且因果图可从识别出的SDE参数中恢复。

    

    随机微分方程（SDE）是建模动态过程的基本工具，其应用涵盖基因调控网络（GRNs）、污染物传输、金融市场以及图像生成等领域。然而，从数据中学习潜在的SDE是一项具有挑战性的任务，尤其是在无法观测个体轨迹的情况下。受蓬勃发展的单细胞数据集研究的启发，我们提出了首个从时间边缘分布中联合识别SDE漂移与扩散项的系统性方法。在线性漂移和加性扩散的假设下，我们证明了不可辨识性仅会在初始分布具有广义旋转对称性时出现。我们进一步证明，即使该条件成立，漂移和扩散项也几乎总能从边缘分布中被恢复出来。此外，我们还证明了任何具有加性扩散的SDE的因果图都可以从已识别出的SDE参数中恢复。

    arXiv:2410.22729v5 Announce Type: replace  Abstract: Stochastic differential equations (SDEs) are a fundamental tool for modelling dynamic processes, including gene regulatory networks (GRNs), contaminant transport, financial markets, and image generation. However, learning the underlying SDE from data is a challenging task, especially if individual trajectories are not observable. Motivated by burgeoning research in single-cell datasets, we present the first comprehensive approach for jointly identifying the drift and diffusion of an SDE from its temporal marginals. Assuming linear drift and additive diffusion, we show that non-identifiability can only arise if the initial distribution possesses generalized rotational symmetries. We further prove that even if this condition holds, the drift and diffusion can almost always be recovered from the marginals. Additionally, we show that the causal graph of any SDE with additive diffusion can be recovered from the identified SDE parameters. 
    

