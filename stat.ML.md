# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decomposing SGD Dynamics in Neural Networks: Teacher-Induced Spikes and Variance Inflation](https://arxiv.org/abs/2611.24908) | 该论文通过将第一层权重分解为信号分量与正交补空间，揭示了SGD训练会在正交补空间的Gram矩阵中诱导出依赖教师的谱尖峰，从而放大表示方差并导致泛化性能退化的具体机制。 |
| [^2] | [Equivariant Modeling with Polynomial Features via Minimal Homogeneous Generators](https://arxiv.org/abs/2611.24896) | 该论文提出了一个基于最小齐次生成元构造等变多项式特征的通用等变函数建模框架，并提供了可自动计算特征与拟合模型的Python工具包，同时给出了更高效的特征计算算法。 |
| [^3] | [Counterfactual World Models: JEPAs That Reason About "What If"](https://arxiv.org/abs/2611.24778) | 该论文证明仅用事实数据训练的世界模型无法回答“同样扰动下换一个动作结果会如何”的反事实问题，并提出利用“自然不做跃变”的一阶近似构建反事实世界模型，使其无需反事实数据和额外参数即可区分糟糕决策与坏运气。 |
| [^4] | [A Generative Model for Higher-Order Network Dynamics](https://arxiv.org/abs/2611.24638) | 该论文提出了一个用户可自定义成对与高阶交互的随机网络动力学生成模型，以有向超边表示高阶交互，从而为验证和比较超越成对依赖的信息论度量提供了受控基准。 |
| [^5] | [Diffusion Models are Provably Efficient Priors for Sparse Recovery](https://arxiv.org/abs/2611.24602) | 本文针对压缩感知中的稀疏恢复问题，首次证明基于预训练扩散模型先验的似然引导退火朗之万采样器能够在多项式时间内以近乎最优的样本复杂度可证明地恢复稀疏信号，为扩散模型作为逆问题先验提供了高效的端到端理论保证。 |
| [^6] | [The geometry of neural network decision boundaries](https://arxiv.org/abs/2611.24557) | 该论文提出了一种直接在高维输入空间中可视化和测量神经网络决策边界几何形状的方法，揭示了决策边界在大连续子空间内接近输入样本，且训练方法对边界几何的影响显著大于网络架构。 |
| [^7] | [Sparse designs via quantization of the uniform measure on high-dimensional hypercubes](https://arxiv.org/abs/2611.24507) | 本文提出用超立方体上均匀测度的量化误差替代传统的maximin/minimax准则来构造高维稀疏空间填充设计，从而克服了高维下所需设计点数过多以及对边界（尤其是超立方体顶点）过度敏感的缺陷。 |
| [^8] | [Latent Action Policies from Suboptimal Data](https://arxiv.org/abs/2611.24482) | 该论文提出了LAPIS方法，通过推导潜在动作空间中策略改进有效迁移与最优对齐的条件，解决了从次优数据训练潜在动作策略时离线强化学习不可靠的问题。 |
| [^9] | [VAFL: A Vertical Analytic Federated Learning Mechanism for Efficient Crowdsourced Services via Closed-Form Solutions](https://arxiv.org/abs/2611.24367) | 提出了一种基于闭式解的纵向分析联邦学习框架VAFL，通过正则化最小二乘在固定局部表示上拟合任务头，并利用块因子等价求解和两种宽度缩减策略大幅降低计算开销，实现高效的两方及多方隐私保护协作学习。 |
| [^10] | [Stability-compatible kernels: beyond Gaussianity for impulse response estimation](https://arxiv.org/abs/2611.24272) | 该论文提出“稳定性相容核”的新概念，证明其当且仅当稳定元素在对应RKHS中稠密时成立，从而在超越高斯先验的情形下为脉冲响应估计提供了比已知条件更弱的几乎必然BIBO稳定性判据。 |
| [^11] | [TRESS: A Fixed-Size History Representation for Irregular Time Series Forecasting](https://arxiv.org/abs/2611.24257) | TRESS提出了一种固定大小的历史表示方法，通过保留带时间戳的近期观测值并摘要完整历史的统计信息，结合查询条件化的事件提升模块与通道特定解码器，在不规则时间序列预测上取得了最优的测试误差。 |
| [^12] | [FedE2E: One-Shot Federated Learning for End-to-End Autonomous Driving with Closed-Loop Evaluation](https://arxiv.org/abs/2611.24254) | 提出首个面向端到端自动驾驶的一次性联邦学习方法FedE2E，通过多客户端校准、锚点特征对齐和词元级共识蒸馏从异构模型中蒸馏知识，在Bench2Drive闭环评估中一次性驾驶分数达55.85，超过先前工作4.07分。 |
| [^13] | [Functional Principal Component Analysis: Minimax Optimality from Sparse to Dense Designs and Beyond](https://arxiv.org/abs/2611.24228) | 本文首次在含噪离散观测下建立了函数主成分分析的完整极小化极大最优理论，覆盖了从稀疏到稠密随机设计中谱指标发散的特征函数与维数递增的主子空间，并给出了一般二元函数类上本质上二维的最优协方差估计速率。 |
| [^14] | [Rare-Tail Statistics For Learning Biased Gaussian Halfspaces with Label Noise](https://arxiv.org/abs/2611.24181) | 本文提出“稀有尾部均值估计量”这一极简统计方法，在大偏置高斯半空间带标签噪声的学习问题中，以信息论最优样本率恢复目标半空间方向，超越了经典平均法等全局一阶矩统计量。 |
| [^15] | [SoU-RHS: Bayesian Structured Sparsity for Compact Architecture Recovery in CNNs](https://arxiv.org/abs/2611.24166) | 该论文提出SoU-RHS，一种将正则化马蹄铁收缩与不确定性来源稀疏全局-局部机制相结合的贝叶斯结构化稀疏方法，能够稳定地从过参数化CNN中恢复出规模一致且适应数据的紧凑架构。 |
| [^16] | [A Capability Framework for Evaluating Agentic AI Platforms in Regulated Enterprises](https://arxiv.org/abs/2611.24039) | 该论文提出了一套用于评估受监管企业中智能体AI平台的能力框架，通过对全球银行三个金融服务工作负载上十个平台的九维度评测，将市场划分为在权重变化下依然稳定的三个层级。 |
| [^17] | [NOTREKS: Marginal Independence in Score-based Structure Learning](https://arxiv.org/abs/2611.24015) | 本文提出 NOTREKS 信息，将变量间边际独立的先验知识转化为“无 trek”图约束，并通过 PSTrek 惩罚项和 trek 约束融入基于评分的因果结构学习，从而在有限样本下提升结构恢复与效应识别效果。 |
| [^18] | [How Reward Gain Imbalance Shapes Ranking: Rethinking Rubric-Based Reward Aggregation](https://arxiv.org/abs/2611.24005) | 该论文将基于量规的奖励聚合中的评分异质形式化为“奖励增益不平衡”概念，并出人意料地发现消除这种不平衡（即使使用真实标签）反而会降低排名准确性。 |
| [^19] | [ReliFlow: Reliability-Calibrated Residual Transport for Combinatorial Single-Cell Perturbation Prediction](https://arxiv.org/abs/2611.23999) | 提出ReliFlow——一种条件流匹配模型，通过训练期交叉拟合估计单基因效应可靠性并据此校准加性源分布，将速度场分解为加性参考的已知修正与学习得到的残差传输场，从而在单扰动证据稀疏或缺失的情况下显著提升组合扰动响应的预测精度。 |
| [^20] | [Optimal Deep Nonparametric Regression with Uniformly Bounded Lipschitz Constants](https://arxiv.org/abs/2611.23979) | 本文在分裂谱范数乘积约束下建立了深度ReLU网络的尖锐逼近理论，并提出带惩罚的最小二乘估计器，使深度非参数回归在保持一致有界Lipschitz常数的同时达到最优统计收敛率。 |
| [^21] | [Risk-Controlled Correction for Frozen Multimodal Models under Missing Modalities](https://arxiv.org/abs/2611.23886) | 本文提出用贝叶斯差距刻画冻结多模态模型在模态缺失后仍可修正的误差部分，并证明对条件正确的更全模态模型的输出在剩余模态条件下取平均，无需标签即可恢复贝叶斯预测目标，从而实现风险可控的预测修正。 |
| [^22] | [Infinity-Wasserstein Distributionally Robust Principal Component Analysis: Exact Reformulation, Finite-Sample Guarantee, and Efficient Algorithm](https://arxiv.org/abs/2611.23877) | 该论文提出基于无穷范数-Wasserstein 距离的分布鲁棒 PCA 模型，实现了精确的有限维重构、最坏情况分布的显式刻画、有限样本理论保证与高效算法，并揭示其目标函数通过保留经验协方差之外的逐样本残差信息，使分布鲁棒性能够重塑所学的主子空间。 |
| [^23] | [Defining equations for two-block shallow ReLU pattern varieties](https://arxiv.org/abs/2611.23846) | 本文完整地代数刻画了两块浅层ReLU模式簇，证明由激活模式的秩约束所产生的行列式方程生成其定义理想，从而解决了Alexandr-Montúfar提出的猜想。 |
| [^24] | [Improved Leverage Score Sampling for Constrained Active Linear Regression](https://arxiv.org/abs/2611.23801) | 提出椭球正则化杠杆值分数采样，通过外接椭球的形状矩阵刻画约束集几何并将采样偏向重要方向，使约束主动线性回归的查询复杂度由各向异性有效维度决定，从而改进各向异性约束下的查询效率。 |
| [^25] | [Variable Length Diffuse Everything: Arbitrary Length Generative Modeling on Arbitrary State Spaces](https://arxiv.org/abs/2611.23684) | 提出了可变长度扩散万物（VL-DE）框架，通过基于严谨理论的插入机制而非数据填充，实现了任意状态空间上可变长度多模态数据的简单通用生成。 |
| [^26] | [A Free Lunch: Gradient Averaging Achieves Optimal Statistical Efficiency in Newton Sketch](https://arxiv.org/abs/2611.23671) | 该论文证明，在草拟牛顿方法中仅需将随机梯度替换为滑动平均（无需额外计算开销、不改变内部求解器收缩界），即可缓解甚至消除草拟带来的计算—统计权衡，实现最优统计效率。 |
| [^27] | [Efficient Variance-Aware Reinforcement Learning in Time-Inhomogeneous Linear MDPs](https://arxiv.org/abs/2611.23596) | 本文提出计算高效的算法 VA-LSVI-UCB++，在时变非齐次线性 MDP 中实现了方差依赖的遗憾界，并消除了以往方差感知分析中即使在确定性环境下仍然存在的 $\tilde{O}(K^{1/4})$ 多项式加性项。 |
| [^28] | [Bounded-Order Non-Identifiability in Latent Structure Block Models: The Fiber in Closed Form](https://arxiv.org/abs/2611.23547) | 该论文为位置位于K条射线上的潜在结构块模型给出了有限阶模体计数下不可辨识配置的一维纤维闭式解，并证明友谊图F_{K-1}是唯一边最小分离子，因此不存在对所有K都适用的固定模体阶数。 |
| [^29] | [Stability of Parallel Transport Convolution under Riemannian Metric Perturbations](https://arxiv.org/abs/2611.23543) | 本文证明了在紧流形上，平行输运卷积的输出在 $L^2$ 意义下对黎曼度量的 $C^2$ 扰动具有连续稳定性，为此发展了测地线、平行输运、雅可比场与指数映射的扰动估计技术。 |
| [^30] | [The Spectral Distribution of Matrix Least Squares Solutions, and the Ellipsoid Fitting Problem](https://arxiv.org/abs/2611.23431) | 本文研究了未知量为对称矩阵的随机线性系统最小二乘解的谱分布并给出其支撑集的闭式描述，进而应用于椭球拟合问题，证明了最小二乘候选解在 n ≈ d²/10 处存在半正定性阈值。 |
| [^31] | [When Does Subspace Direction Matter for LoRA? Regime Analysis of the Magnitude Principle in Few-Shot Adaptation](https://arxiv.org/abs/2611.23428) | 本文证明LoRA初始子空间方向的重要性取决于情境：在少样本且初始化子空间能捕捉任务结构时方向对齐至关重要，并提出对齐系数ρ来预测增益、以及PivotLoRA进行任务对齐初始化。 |
| [^32] | [Which Tokens, in What Order, and How Fast? A Theoretical Analysis of On-Policy Distillation Objectives](https://arxiv.org/abs/2611.23397) | 本文首次从理论上剖析在线策略蒸馏目标的选择机制，揭示序列级OPD存在导致反向学习顺序的“未来一致性偏差”，而Token级目标可避免该偏差，并阐明了散度选择如何影响Token的优先级、学习顺序与收敛效率。 |
| [^33] | [A Differential Theory of Quality-Resource Trade-offs for AI Model Selection: The Multiresource Frontier and Its Minimax Price](https://arxiv.org/abs/2611.23324) | 该论文提出一种微分理论，将AI模型选择的质量-资源前沿表示为与价格无关的平面曲线，通过有效资源通道数与剩余对数误差深度之间的内在关系刻画了多资源权衡的几何结构与极小极大价格。 |
| [^34] | [A Matching Lower Bound for Online Smooth Calibration](https://arxiv.org/abs/2611.16097) | 本文通过构造自适应对手和遗忘对手，证明在线平滑校准误差的极小极大下界为 Ω(√T)，填补了已知的上下界差距，确立了平滑校准误差的极小极大最优性。 |
| [^35] | [Exploration via Exploitation: The Blessing of Reward Diversity in Personalized Federated RL](https://arxiv.org/abs/2611.12964) | 该论文将个性化联邦强化学习中的奖励异质性从协作障碍转变为结构性优势，提出PF-UCBVI和PF-EFVI算法，证明奖励多样性可替代显式探索，使纯贪心算法即可实现对数遗憾和最优线性加速。 |
| [^36] | [How Path Integral Diffusion Decides a Finite Spin Glass](https://arxiv.org/abs/2611.11826) | 该论文证明各向同性路径积分扩散等价于经过时间变换的高斯随机局部化，从而表明扩散生成器对自旋玻璃所做决策的内容与具体协议无关，并可由作为鞅的“部分承诺”按磁化对比平方设定的速率逐级刻画。 |
| [^37] | [Prescriptive Learning of Linear Policies with Feasibility Guarantees for Binary Decision-Making](https://arxiv.org/abs/2611.10923) | 本文提出一种规范性学习框架，通过在所有训练场景上联合优化线性决策策略与可行分配，为不确定性下的二元决策提供可行性保证，并解决了预测式学习缺乏反事实认知且在分布偏移下易产生不可行决策的问题。 |
| [^38] | [When Is Accuracy Evidence? A Unified Theory of Generalisation, Validation, and Information Fusion](https://arxiv.org/abs/2610.03465) | 该论文提出一个统一的指数框架，将交叉验证准确率转化为真实风险的保守上界，并通过有效折数Keff证明：当各折数据强相关时，单纯增大交叉验证折数K并不能增强统计证据。 |
| [^39] | [IQS-BO: In-Context Query Selection for Bayesian Optimisation](https://arxiv.org/abs/2610.01269) | 本文提出IQS-BO，一种通过在合成先验上进行监督学习来学习贝叶斯优化查询决策的PFN，可在单次前向传递中完成查询选择，从而避免了代理模型重新拟合和采集函数的数值最大化。 |
| [^40] | [Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time](https://arxiv.org/abs/2609.22226) | 该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。 |
| [^41] | [Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts](https://arxiv.org/abs/2609.18366) | 提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。 |
| [^42] | [A Layered Simplex Architecture for Large Alphabets](https://arxiv.org/abs/2608.19908) | 本文提出一种极简的分层单纯形贝叶斯估计器，通过均匀相乘并归一化实现，无需调参，具有显式可计算的遗憾值，并在大字母表概率估计任务上与专门方法竞争，同时揭示缩放规律。 |
| [^43] | [Assessing model calibration with boosting trees](https://arxiv.org/abs/2606.08084) | 本文提出利用提升树来检验回归模型强校准和自校准的必要条件，并以保险定价等场景中的数值实验验证了该方法的实际应用价值。 |
| [^44] | [From Data to Action: Accelerating Refinery Optimization with AI](https://arxiv.org/abs/2605.15085) | 本文通过为线性规划求解器增加记忆功能并引入异常检测工具，能够发现炼油厂优化中输入数据的潜在错误及不同决策层面的异常关系，从而提升后续LP求解结果的可信度和质量。 |
| [^45] | [Inverse Design for Conditional Distribution Matching](https://arxiv.org/abs/2605.09439) | 该论文提出了条件分布匹配（CDM）这一新的逆向设计问题类别，即寻找使生成模型的条件分布匹配给定目标分布的输入，并定义了CDMS与CDMO两个问题变体及相应的求解方法MLGD-F。 |
| [^46] | [Optimistic Actor-Critic with Parametric Policies for Linear Markov Decision Processes](https://arxiv.org/abs/2603.28595) | 该论文针对有限时域线性马尔可夫决策过程提出了使用参数化对数线性策略的乐观演员-评论家框架，通过引入可处理的logit匹配回归目标和基于朗之万动力学的近似汤普森采样，克服了现有演员-评论家方法在探索假设和隐式策略采样效率方面的理论局限。 |
| [^47] | [Kinetic Langevin Splitting Schemes for Constrained Sampling](https://arxiv.org/abs/2603.23397) | 本文提出将动力学朗之万动力学的BAOAB和ABOBA对称分裂格式应用于约束采样的惩罚代理目标，并在仅假设强凸性和光滑性（无需Lipschitz Hessian）的条件下建立了非渐近Wasserstein误差界，其随机梯度版本的复杂度显著优于传统Euler型离散化方法。 |
| [^48] | [Signature-Kernel Based Evaluation Metrics for Robust Probabilistic and Tail-Event Forecasting](https://arxiv.org/abs/2602.10182) | 本文提出了一族基于签名核的删失评估指标，通过将分布主体收缩为单一枢轴点来聚焦尾部区域和高效用区域的评估，从而更准确地衡量概率预测在分布尾部的保真度。 |
| [^49] | [Pragmatic Curiosity: A Unified Framework for Hybrid Learning and Optimization via Active Inference](https://arxiv.org/abs/2602.06104) | 该论文提出了实用好奇心（PraC）框架，通过主动推理将贝叶斯优化与贝叶斯实验设计统一起来，以信息增益与预期遗憾之间的权衡来指导混合学习与优化场景中的序贯决策。 |
| [^50] | [Targeted Synthetic Control Method](https://arxiv.org/abs/2602.04611) | 本文提出目标化合成控制（TSC）方法，一种新型两阶段估计器，通过目标化去偏更新直接估计反事实结果，在保证权重稳定性和凸组合可解释性的同时，可灵活结合任意机器学习模型。 |
| [^51] | [Low-Dimensional Adaptation of Rectified Flow: A Diffusion and Stochastic Localization Perspective](https://arxiv.org/abs/2601.15500) | 本文证明整流流能自动适应目标分布的内在低维结构，在端点自适应时间离散化方案下达到 O(k/ε) 的采样迭代复杂度，并建立了DDPM与随机定位之间的新颖联系，证明DDPM等价于整流流的随机化版本。 |
| [^52] | [Generative modeling of conditional probability distributions on the level-sets of collective variables](https://arxiv.org/abs/2512.17374) | 本文研究了如何对由集体变量水平集所定义的条件概率分布进行生成式建模。 |
| [^53] | [When Features Beat Noise: A Noise-Driven Hypothesis Testing Method for Feature Selection](https://arxiv.org/abs/2511.20851) | 本文提出一种噪声驱动的特征选择方法，通过将每个特征的重要性与随机噪声特征中的最大重要性进行基于非参数bootstrap的假设检验，为特征选择提供了坚实的统计理论基础和显著性评估准则。 |
| [^54] | [Assessing the robustness of heterogeneous treatment effects in survival analysis under informative censoring](https://arxiv.org/abs/2510.13397) | 该论文提出一个少假设框架及新颖的模型无关元学习器SurvB-learner，通过部分识别推导条件平均处理效应的边界，从而在信息性删失偏差下评估生存分析中异质性治疗效应估计的稳健性，并识别治疗有效的患者亚群。 |
| [^55] | [Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models](https://arxiv.org/abs/2505.16959) | 过参数化扩散模型在训练中先于记忆化发生之前逐步实现泛化，记忆化时间与数据集大小成正比，因此泛化与记忆化本质上是时间尺度的竞争，可通过早停来避免记忆化。 |
| [^56] | [Simultaneous analysis of approximate leave-one-out cross-validation and mean-field inference](https://arxiv.org/abs/2501.02624) | 本文通过将ALO-CV与平均场推断进行联合分析，在高斯协变量与强凸性条件下首次证明了近似留一交叉验证对于不可微正则化子也能以可忽略的误差一致地逼近真实的留一泛化误差。 |
| [^57] | [Learning Robust Treatment Rules for Censored Data](https://arxiv.org/abs/2408.09155) | 该论文提出了两个稳健准则及基于采样的差凸算法，用于从删失生存数据中学习最优治疗规则，分别最大化截断平均生存时间和缓冲生存概率。 |
| [^58] | [Statistical Performance Guarantee for Selecting Those Predicted to Benefit Most from Treatment.](http://arxiv.org/abs/2310.07973) | 本研究针对选择最有可能从治疗中获益的人的问题，在估计治疗效果和确定截断值时面临多重测试问题，提出一种统一的置信带方法来评估这些个体的平均治疗效果。 |

# 详细

[^1]: 分解神经网络中的SGD动力学：教师诱导的尖峰与方差膨胀

    Decomposing SGD Dynamics in Neural Networks: Teacher-Induced Spikes and Variance Inflation

    [https://arxiv.org/abs/2611.24908](https://arxiv.org/abs/2611.24908)

    该论文通过将第一层权重分解为信号分量与正交补空间，揭示了SGD训练会在正交补空间的Gram矩阵中诱导出依赖教师的谱尖峰，从而放大表示方差并导致泛化性能退化的具体机制。

    

    我们研究了在多指数教师模型上通过随机梯度下降（SGD）训练的两层神经网络，重点关注与教师子空间正交的第一层参数如何影响泛化性能。我们将第一层权重分解为与教师对齐的信号分量和正交补空间分量，并在比例极限下分析它们与第二层的联合SGD动力学。我们证明，尽管SGD对正交补空间的更新最初是各向同性的，但训练动力学会在补空间的Gram矩阵中诱导出依赖教师的谱尖峰。这种涌现的各向异性提供了一种具体机制，通过该机制表示方差被放大，泛化性能随之退化。为了隔离这一效应，我们引入了“分解动力学”，这是一个解析上可处理的替代模型，其渐近测试误差动力学与SGD完全一致。在使用ReLU学生网络的模拟中，冻结正交补空间的相关方向……

    arXiv:2611.24908v1 Announce Type: cross  Abstract: We study two-layer neural networks trained by stochastic gradient descent (SGD) on a multi-index teacher model, focusing on how first-layer parameters orthogonal to the teacher subspace affect generalization. We decompose the first-layer weights into a signal component aligned with the teacher and an orthogonal complement, and analyze their joint SGD dynamics together with the second layer in the proportional limit. We show that although SGD updates to the complement are initially isotropic, the training dynamics induce teacher-dependent spectral spikes in the complement Gram matrix. This emergent anisotropy provides a concrete mechanism by which representation variance is amplified, and generalization degrades. To isolate this effect, we introduce Decomposed Dynamics, an analytically tractable surrogate whose asymptotic test-error dynamics coincide with those of SGD. In simulations with a ReLU student, freezing the directions of the o
    
[^2]: 基于最小齐次生成元的多项式特征等变建模

    Equivariant Modeling with Polynomial Features via Minimal Homogeneous Generators

    [https://arxiv.org/abs/2611.24896](https://arxiv.org/abs/2611.24896)

    该论文提出了一个基于最小齐次生成元构造等变多项式特征的通用等变函数建模框架，并提供了可自动计算特征与拟合模型的Python工具包，同时给出了更高效的特征计算算法。

    

    我们提出了一个用于建模和学习有限维半单实群表示之间等变函数的通用框架，该框架统一了众多应用领域中发展出的各种构造方法。我们的模型是等变多项式特征的线性组合，其系数是不变多项式特征的任意可学习函数。我们选取不变代数以及不变代数上协变模的最小齐次生成元作为特征，直至给定的阶数。我们提供了一个Python包，可自动计算这些特征并将所得模型拟合到训练数据。特征计算算法需要输入和输出表示的显式同型分解，以及不可约表示张量积的显式Clebsch-Gordan分解，并在计算代数方面相比其他方法实现了加速。数值实验（摘要在此处被截断）

    arXiv:2611.24896v1 Announce Type: new  Abstract: We present a general framework for modeling and learning equivariant functions between finite dimensional semisimple real group representations, unifying constructions developed across many application areas. Our model is a linear combination of equivariant polynomial features whose coefficients are arbitrary learnable functions of invariant polynomial features. We take as features minimal homogeneous generators for the invariant algebra and the covariant module over the invariant algebra, up to prescribed degrees. We provide a Python package that automatically computes these features and fits the resulting models to training data. The algorithm for feature computation requires explicit isotypic decompositions of the input and output representations, along with explicit Clebsch-Gordan decompositions of the tensor product of irreducible representations, and achieves speedups over alternative methods in computational algebra. Numerical exp
    
[^3]: 反事实世界模型：能够推理“如果……会怎样”的JEPA

    Counterfactual World Models: JEPAs That Reason About "What If"

    [https://arxiv.org/abs/2611.24778](https://arxiv.org/abs/2611.24778)

    该论文证明仅用事实数据训练的世界模型无法回答“同样扰动下换一个动作结果会如何”的反事实问题，并提出利用“自然不做跃变”的一阶近似构建反事实世界模型，使其无需反事实数据和额外参数即可区分糟糕决策与坏运气。

    

    世界模型运行在一个它们无法完全观测的世界中。当一种隐藏的扰动（例如推动船只的水流）改变了某个动作的结果时，随之而来的便是一个反事实问题：在同样的扰动作用下，如果采取不同的动作，会发生什么？这个问题的答案能够将糟糕的决策与坏运气区分开来。我们证明，仅在事实转移（真实发生的转变）上训练的世界模型无法依靠自身回答这个问题，无论其数据或模型容量如何。对于许多物理系统而言，观测到的结果加上动力学的一个简单性质就足够了：natura non facit saltus，即自然不做跃变。当动作和扰动作为相互独立的力分别作用时，它们的相互作用在时间上是二阶的，因此反事实结果在一阶近似下等于观测到的结果，再加上世界模型在替代动作与事实动作下预测结果之间的差值。由此便得到了一个无需反事实数据、无需额外参数的反事实世界模型。

    arXiv:2611.24778v1 Announce Type: cross  Abstract: World models operate in a world they cannot fully observe. When a hidden disturbance, such as a current pushing a boat, changes the outcome of an action, a counterfactual question follows: what would have happened under a different action, subject to the same disturbance? Its answer separates bad decisions from bad luck. We prove that world models trained on factual transitions cannot answer it on their own, whatever their data or capacity. For many physical systems, the observed outcome and a simple property of the dynamics suffice: natura non facit saltus, nature makes no leaps. When actions and disturbances act as separate forces, their interaction is second-order in time, so the counterfactual outcome is, to first order, the observed outcome plus the difference between the world model's predictions under the alternative and factual actions. This yields a counterfactual world model with no counterfactual data, extra parameters, or r
    
[^4]: 高阶网络动力学的生成模型

    A Generative Model for Higher-Order Network Dynamics

    [https://arxiv.org/abs/2611.24638](https://arxiv.org/abs/2611.24638)

    该论文提出了一个用户可自定义成对与高阶交互的随机网络动力学生成模型，以有向超边表示高阶交互，从而为验证和比较超越成对依赖的信息论度量提供了受控基准。

    

    从多元时间序列推断的传统网络表示描述的是组件对之间的统计关系，即使潜在的动力学涉及无法仅用成对交互来表示的群体交互。已有若干多元信息论度量被提出来刻画超越成对关系的依赖性，然而由于缺乏具有已知生成机制的系统，这些度量尚未得到仔细验证或比较以供实证使用。为解决这一问题，我们在此引入了一个随机网络动力学的生成模型，其中的成对交互和高阶交互可由用户指定。高阶交互由有向超边表示，该模型为评估信息论度量提供了受控基准。仅去除高阶项即可得到仅含成对交互的参照。实验测试某种度量能否检测出……

    arXiv:2611.24638v1 Announce Type: new  Abstract: Conventional network representations inferred from multivariate time series describe statistical relationships between pairs of components, even when the underlying dynamics involve group interactions that cannot be represented by pairwise interactions alone. Several multivariate information-theoretic measures have been proposed to characterise dependence beyond pairs. However, due to a lack of systems with known generating mechanisms, these measures have not been carefully validated or compared for empirical use. To address this, we here introduce a generative model of stochastic network dynamics with pairwise and higher-order interactions specified by the user. The higher-order interactions are represented by directed hyperedges, and the model provides a controlled benchmark for evaluating information-theoretic measures. Removing only the higher-order term gives a pairwise-only reference. The experiments test whether a measure detects 
    
[^5]: 扩散模型是稀疏恢复的可证明高效先验

    Diffusion Models are Provably Efficient Priors for Sparse Recovery

    [https://arxiv.org/abs/2611.24602](https://arxiv.org/abs/2611.24602)

    本文针对压缩感知中的稀疏恢复问题，首次证明基于预训练扩散模型先验的似然引导退火朗之万采样器能够在多项式时间内以近乎最优的样本复杂度可证明地恢复稀疏信号，为扩散模型作为逆问题先验提供了高效的端到端理论保证。

    

    扩散模型凭借其捕捉复杂数据分布并从中生成样本的能力，彻底改变了生成建模领域。这一成功激发了人们浓厚的兴趣，即将预训练的扩散模型用作成像与感知中高度病态逆问题的表达性先验。尽管经验性能出色，但这种方法的理论理解仍然有限：现有保证往往是渐近性的，或在最坏情况下需要难以承受的计算成本。在本文中，我们通过研究从少量线性测量中进行稀疏恢复这一压缩感知中的经典问题来填补这一空白。我们证明，给定一个准确刻画促进稀疏性的伯努利-高斯先验的扩散模型，一个由似然引导的退火朗之万采样器能够在多项式时间内、以近乎最优的样本复杂度，可证明地恢复稀疏信号，且适用于广泛的随机和确定性传感矩阵族。

    arXiv:2611.24602v1 Announce Type: cross  Abstract: Diffusion models have transformed generative modeling through their ability to capture and generate samples from complex data distributions. This success has spurred substantial interest in using pretrained diffusion models as expressive priors for highly ill-posed inverse problems in imaging and sensing. Despite strong empirical performance, theoretical understanding of this approach remains limited: existing guarantees are often asymptotic or entail prohibitive computational costs in the worst case. In this paper, we address this gap by studying sparse recovery from a small number of linear measurements, a classical problem in compressed sensing. We show that, given an accurate diffusion model capturing a sparsity-promoting Bernoulli-Gaussian prior, a likelihood-guided annealed Langevin sampler provably recovers the sparse signal in polynomial time with near-optimal sample complexity for a broad family of random and deterministic sen
    
[^6]: 神经网络决策边界的几何学

    The geometry of neural network decision boundaries

    [https://arxiv.org/abs/2611.24557](https://arxiv.org/abs/2611.24557)

    该论文提出了一种直接在高维输入空间中可视化和测量神经网络决策边界几何形状的方法，揭示了决策边界在大连续子空间内接近输入样本，且训练方法对边界几何的影响显著大于网络架构。

    

    人工神经网络（ANN）由人们充分理解的数学运算构建而成，但其复杂的特性给直观理解其决策过程带来了重大挑战。本研究提出了一种方法，可以直接在高维输入空间中可视化和测量ANN决策边界的几何形状，避免了传统基于投影技术的失真问题。我们分析了在不同输入、架构和训练方法下的边界曲率和距离。随着子空间规模的增大，边界距离平滑且缓慢地增长，同时保持一致的曲率，这表明决策边界在大而连续的子空间内接近输入样本。我们还量化了对抗训练后数据集样本附近边界距离和曲率的降低程度。研究还发现，训练方法对边界距离和曲率的影响显著强于架构的影响。

    arXiv:2611.24557v1 Announce Type: new  Abstract: Artificial neural networks (ANNs) are built using well-understood mathematical operations, yet their complex nature poses significant challenges to intuitively understanding their decision-making processes. This work introduces a method for visualizing and measuring ANN decision boundary geometry directly in the high-dimensional input space, avoiding distortions found with traditional projection-based techniques. We analyze boundary curvatures and distances in diverse inputs, architectures, and training methods. Boundary distances grow smoothly and slowly with consistent curvature as the size of the subspace increases, suggesting that decision boundaries are close to input samples within large, continuous subspaces. We also quantify how much boundary distances and curvatures reduce near dataset samples after adversarial training. Training methodology has a significantly stronger impact on boundary distance and curvature than architecture
    
[^7]: 通过高维超立方体上均匀测度的量化构造稀疏设计

    Sparse designs via quantization of the uniform measure on high-dimensional hypercubes

    [https://arxiv.org/abs/2611.24507](https://arxiv.org/abs/2611.24507)

    本文提出用超立方体上均匀测度的量化误差替代传统的maximin/minimax准则来构造高维稀疏空间填充设计，从而克服了高维下所需设计点数过多以及对边界（尤其是超立方体顶点）过度敏感的缺陷。

    

    虽然球填充和球覆盖的概念构成了计算机实验中空间填充设计最广泛使用的准则——即maximin和minimax方法——的基础，但它们在高维情况下存在重大缺陷。首先，当维度 $d$ 非常大时，部署空间填充设计需要极其大量的设计点，这使得空间填充设计这一概念本身变得值得质疑。其次，这两种设计特征对边界高度敏感，尤其是当设计域为超立方体时（即本文考虑的情形），对其顶点尤为敏感。当设计规模小于 $2^d$ 时，这种敏感性表现得特别突出。本文不再采用maximin或minimax准则，而是考虑超立方体上均匀测度的量化误差。该误差具有三个关键优势：它考虑了从一个随机点出发的距离分布的整体情况（摘要原文在此处截断）。

    arXiv:2611.24507v1 Announce Type: cross  Abstract: While the notions of sphere packing and sphere covering underlie the most widely used criteria for space-filling design in computer experiments, namely, the maximin and minimax approaches, they suffer from major drawbacks in high dimensions. First, when the dimension $d$ is very large, deploying a space-filling design requires an extremely large number of design points, making the very notion of a space-filling design questionable. Second, these two design characteristics are highly sensitive to boundaries, particularly when the design domain is a hypercube (the case considered in this paper), and especially to its vertices. This sensitivity is particularly acute when the design size is less than $2^d$. Instead of considering the maximin or minimax criterion, we consider here the quantization error of the uniform measure on the hypercube. This error has three key advantages: it accounts for the entire distribution of distances from a r
    
[^8]: 从次优数据中学习潜在动作策略

    Latent Action Policies from Suboptimal Data

    [https://arxiv.org/abs/2611.24482](https://arxiv.org/abs/2611.24482)

    该论文提出了LAPIS方法，通过推导潜在动作空间中策略改进有效迁移与最优对齐的条件，解决了从次优数据训练潜在动作策略时离线强化学习不可靠的问题。

    

    潜在动作模型能够从无动作标注的视频中学习控制表示，但现有的流程通常采用行为克隆来训练潜在策略，缺乏显式的策略改进目标。尽管离线强化学习为策略改进提供了自然的途径，但标准方法无法直接迁移到学习得到的潜在动作空间。我们发现，学习到的潜在动作的结构使得朴素的离线强化学习不可靠：直接在潜在动作上最大化价值或拟合单峰策略，可能产生无法通过物理执行实现的潜在动作，同时潜在表示可能将获得不同奖励的物理控制合并在一起。为了理解潜在空间优化在何时仍然有效，我们推导了策略改进能够迁移到解码执行的条件，以及潜在最优与物理最优对齐的条件。基于这些见解，我们提出了LAPIS，用于目标条件下的潜在动作策略学习。

    arXiv:2611.24482v1 Announce Type: new  Abstract: Latent Action Models (LAMs) learn control representations from action-free video, but existing pipelines typically train latent policies with behavioral cloning, without an explicit policy improvement objective. Although offline RL offers a natural approach to policy improvement, standard methods do not transfer directly to learned latent action spaces. We find that the structure of learned latent actions makes naive offline RL unreliable: directly maximizing value over latent actions or fitting a unimodal policy can produce latent actions that cannot be realized by physical execution, while the latent representation can merge physical controls that receive different rewards. To understand when latent-space optimization remains valid, we derive conditions under which policy improvement transfers to decoded execution and under which latent and physical optima align. Building on these insights, we propose LAPIS for goal-conditioned latent 
    
[^9]: VAFL：一种基于闭式解的高效众包服务纵向分析联邦学习机制

    VAFL: A Vertical Analytic Federated Learning Mechanism for Efficient Crowdsourced Services via Closed-Form Solutions

    [https://arxiv.org/abs/2611.24367](https://arxiv.org/abs/2611.24367)

    提出了一种基于闭式解的纵向分析联邦学习框架VAFL，通过正则化最小二乘在固定局部表示上拟合任务头，并利用块因子等价求解和两种宽度缩减策略大幅降低计算开销，实现高效的两方及多方隐私保护协作学习。

    

    智能服务通常依赖于不同组织所持有的互补信息，这推动了无需集中式原始数据汇集的纵向联邦学习的发展。我们提出VAFL，一种纵向分析联邦学习框架，通过正则化最小二乘法在固定局部表示上拟合分类和回归头。该框架采用代理辅助的隐私保护计算工作流，支持两方和多方协作。为降低拟合开销，我们开发了一种全维度等价求解方法，将块因子直接应用于目标矩阵，避免了构建完整逆矩阵。我们进一步研究了两种宽度缩减策略：带系数提升的局部谱投影，以及带低秩跨方修正的全维度局部拟合。前者压缩预测空间，而后者保留了协作子空间之外的局部成分。我们建立了它们的代数性质……

    arXiv:2611.24367v1 Announce Type: new  Abstract: Intelligent services often rely on complementary information held by different organizations, motivating Vertical Federated Learning without centralized raw-data pooling. We propose VAFL, a Vertical Analytic Federated Learning framework that fits classification and regression heads on fixed local representations through regularized least squares. An agent-assisted protected-computation workflow supports two-party and multi-party collaboration. To reduce fitting overhead, we develop a full-dimensional equivalent solve that applies block factors directly to the target matrix, avoiding construction of the full inverse. We further study two width-reduction strategies: local spectral projection with coefficient lifting, and full-dimensional local fitting with low-rank cross-party correction. The first compresses the prediction space, whereas the second retains local components outside the collaboration subspace. We establish their algebraic p
    
[^10]: 稳定性相容核：超越高斯性的脉冲响应估计

    Stability-compatible kernels: beyond Gaussianity for impulse response estimation

    [https://arxiv.org/abs/2611.24272](https://arxiv.org/abs/2611.24272)

    该论文提出“稳定性相容核”的新概念，证明其当且仅当稳定元素在对应RKHS中稠密时成立，从而在超越高斯先验的情形下为脉冲响应估计提供了比已知条件更弱的几乎必然BIBO稳定性判据。

    

    线性系统辨识中基于核的正则化方法使用半正定核既来定义再生核希尔伯特空间（RKHS），又作为随机脉冲响应的协方差模型。目前已知的核条件可以精确刻画RKHS稳定性，而在高斯先验下的几乎必然BIBO稳定性则需要更为严苛的协方差条件。然而，这些限制在超越高斯性的情形下未必继续成立，因为此时协方差不再决定完整的概率分布规律。我们引入了“稳定性相容核”的概念，即一种协方差核，它允许至少一个具有几乎必然绝对可和样本路径的随机实现。我们证明，稳定性相容性当且仅当稳定元素在相应RKHS中稠密时成立，该条件严格弱于RKHS稳定性，并推导出了实用的充分条件，包括针对平稳核的谱判据。

    arXiv:2611.24272v1 Announce Type: cross  Abstract: Kernel-based regularization for linear system identification uses positive semidefinite kernels both to define reproducing kernel Hilbert spaces (RKHSs) and as covariance models for stochastic impulse responses. Sharp kernel conditions are known for RKHS stability, while almost-sure BIBO stability under Gaussian priors requires an even more restrictive covariance condition. These restrictions, however, need not persist beyond Gaussianity, where the covariance no longer determines the full probability law. We introduce the notion of a stability-compatible kernel, namely a covariance kernel admitting at least one stochastic realization with almost-surely absolutely summable sample paths. We show that stability-compatibility holds if and only if the stable elements are dense in the associated RKHS, a condition strictly weaker than RKHS stability, and derive practical sufficient conditions, including a spectral criterion for stationary ker
    
[^11]: TRESS：一种用于不规则时间序列预测的固定大小历史表示方法

    TRESS: A Fixed-Size History Representation for Irregular Time Series Forecasting

    [https://arxiv.org/abs/2611.24257](https://arxiv.org/abs/2611.24257)

    TRESS提出了一种固定大小的历史表示方法，通过保留带时间戳的近期观测值并摘要完整历史的统计信息，结合查询条件化的事件提升模块与通道特定解码器，在不规则时间序列预测上取得了最优的测试误差。

    

    固定数量的近期观测值对一个变量而言可能只跨越几分钟，而对另一个变量则可能跨越数天。较早的测量值仍能揭示最新值对该变量而言是否异常。我们提出了TRESS（带时间戳的近期事件与统计摘要），这是一种预测模型，它按原始时间保留最新的观测值，并对每个变量的完整历史进行摘要。对于给定的一组变量和事件预算，该表示的大小在记录不断增长时保持不变。其查询条件化的事件提升模块根据每个观测值的时间距离和预测时间范围，对相对于最新值的偏差进行加权。通道特定的解码器将所得特征与历史统计信息以及跨变量共享的摘要相结合，以预测对最新值的修正量。在为每个变量保留8到16个观测值的情况下，TRESS在所有四个基准数据集上比较的方法中取得了最低的平均测试MSE和MAE。

    arXiv:2611.24257v1 Announce Type: new  Abstract: A fixed number of recent observations may span minutes for one variable and days for another. Earlier measurements can still reveal whether the latest value is unusual for that variable. We propose TRESS (Timestamped Recent Events and Statistical Summaries), a forecasting model that retains the latest observations at their original times and summarises the full history of each variable. For a given set of variables and event budget, this representation stays the same size as the record grows. Its query-conditioned event lifting module weights deviations from the latest value according to the age of each observation and the forecast horizon. A channel-specific decoder combines the resulting features with historical statistics and a summary shared across variables to predict a correction to the latest value. Retaining 8 to 16 observations per variable, TRESS records the lowest mean test MSE and MAE among the methods compared on all four be
    
[^12]: FedE2E：面向端到端自动驾驶的带闭环评估的一次性联邦学习

    FedE2E: One-Shot Federated Learning for End-to-End Autonomous Driving with Closed-Loop Evaluation

    [https://arxiv.org/abs/2611.24254](https://arxiv.org/abs/2611.24254)

    提出首个面向端到端自动驾驶的一次性联邦学习方法FedE2E，通过多客户端校准、锚点特征对齐和词元级共识蒸馏从异构模型中蒸馏知识，在Bench2Drive闭环评估中一次性驾驶分数达55.85，超过先前工作4.07分。

    

    联邦学习能够在不共享数据集的情况下实现协同模型训练。然而，现有的自动驾驶联邦学习方法通常需要多轮通信且缺乏闭环评估，而一次性联邦学习方法则主要专注于感知任务。为了解决这些局限性，我们提出了FedE2E，一种面向端到端自动驾驶的一次性联邦学习方法。该方法包含多客户端校准、基于锚点的特征对齐以及词元级共识蒸馏，以从异构模型中蒸馏知识。在Bench2Drive上的闭环评估证明了FedE2E在一次性和多轮联邦学习中的有效性。在一次性设置下，它取得了55.85的驾驶分数，比先前工作高出4.07分；在多轮设置下，经过五轮通信后达到60.35，是同等通信轮数下所有对比方法中的最佳性能。

    arXiv:2611.24254v1 Announce Type: new  Abstract: Federated learning enables collaborative model training without sharing datasets. However, existing federated learning methods for autonomous driving typically require multi-round communication and lack closed-loop evaluation, while one-shot methods mainly focus on perception tasks. To address these limitations, we propose \method, a one-shot federated learning method for end-to-end autonomous driving. It comprises multi-client calibration, anchor-based feature alignment, and token-wise consensus distillation to distill knowledge from heterogeneous models. Closed-loop evaluation on Bench2Drive demonstrates the effectiveness of \method in both one-shot and multi-round federated learning. In the one-shot setting, it achieves a Driving Score of 55.85, which is 4.07 points higher than prior work. In the multi-round setting, it reaches 60.35 after five communication rounds, the best performance among all compared methods under the same number
    
[^13]: 函数主成分分析：从稀疏到稠密设计及更广泛情形的极小化极大最优性

    Functional Principal Component Analysis: Minimax Optimality from Sparse to Dense Designs and Beyond

    [https://arxiv.org/abs/2611.24228](https://arxiv.org/abs/2611.24228)

    本文首次在含噪离散观测下建立了函数主成分分析的完整极小化极大最优理论，覆盖了从稀疏到稠密随机设计中谱指标发散的特征函数与维数递增的主子空间，并给出了一般二元函数类上本质上二维的最优协方差估计速率。

    

    函数主成分分析（FPCA）作为函数数据分析中的降维与表示工具发挥着核心作用，但其在含噪离散观测下谱指标发散情形的极小化极大理论二十多年来一直悬而未决。为填补这一空白，我们考虑了一种基于筛（sieve）的FPCA框架，该框架汇集跨个体的离散测量数据，并能适应随机设计下从稀疏到稠密的各种采样频率。我们首先在超越张量积低复杂度结构的一般二元函数类上建立了协方差估计的新极小化极大速率，从而识别出本质上二维的最优速率。我们的核心理论贡献是在明确的增长与正则性条件下，为具有发散指标的逐一标记的特征函数以及维数不断增加的领先主子空间建立了极小化极大最优速率。

    arXiv:2611.24228v1 Announce Type: cross  Abstract: Functional principal component analysis (FPCA) plays a central role as a dimension-reduction and representation tool in functional data analysis, but its minimax theory for diverging spectral indices under noisy discrete observation has remained open for over two decades. To fill this gap, we consider a sieve-based FPCA framework that pools discrete measurements across subjects and accommodates sampling frequencies ranging from sparse to dense under random designs. We begin by establishing new minimax rates for covariance estimation over general bivariate function classes beyond tensor-product low-complexity structures, thereby identifying the intrinsically two-dimensional optimal rate. Our central theoretical contribution is to establish minimax optimal rates for both individually labelled eigenfunctions with diverging indices and leading principal subspaces of increasing dimensions, under explicit growth and regularity conditions. To
    
[^14]: 面向带标签噪声的偏置高斯半空间学习的稀有尾部统计方法

    Rare-Tail Statistics For Learning Biased Gaussian Halfspaces with Label Noise

    [https://arxiv.org/abs/2611.24181](https://arxiv.org/abs/2611.24181)

    本文提出“稀有尾部均值估计量”这一极简统计方法，在大偏置高斯半空间带标签噪声的学习问题中，以信息论最优样本率恢复目标半空间方向，超越了经典平均法等全局一阶矩统计量。

    

    我们重新审视了在标签噪声存在下学习一般高斯半空间的经典问题，并特别关注大偏置区间——在该区间中，若干标准标签噪声模型被猜想存在信息与计算之间的差距。具体而言，我们提出并分析了稀有尾部均值估计量，这是一种极其简单的一次性统计量，仅对观测到的少数类进行平均。在大偏置区间中，经典的平均法（Servedio, 1999）及相关的全局一阶矩统计量被证明是次优的，而稀有尾部均值估计量则在广泛的标签噪声模型类下，以信息论最优的样本率（除去对数因子）恢复目标半空间方向，这些模型包括概率单位噪声以及远离决策边界时衰减极慢的间隔相关噪声。值得注意的是，我们的理论保证允许任意接近决策边界的标签近乎完全随机，同时仍能实现强大的方向恢复能力。

    arXiv:2611.24181v1 Announce Type: cross  Abstract: We revisit the classical problem of learning general Gaussian halfspaces in the presence of label noise, with particular focus on the large-bias regime, where information-computation gaps are conjectured in several standard label-noise models. Concretely, we present and analyze the rare-tail mean estimator, an extremely simple one-shot statistic that averages only the observed minority class. In the large-bias regime, classical Averaging (Servedio, 1999) and related global first-moment statistics are provably suboptimal, while the rare-tail mean estimator recovers the target halfspace direction at the information-theoretic sample rate (modulo log factors) under a broad class of label noise models, including probit noise and margin-dependent noise with very slow decay away from the boundary. Notably, our guarantees allow labels arbitrarily close to the decision boundary to be nearly random, while still giving strong directional recovery
    
[^15]: SoU-RHS：用于卷积神经网络紧凑架构恢复的贝叶斯结构化稀疏方法

    SoU-RHS: Bayesian Structured Sparsity for Compact Architecture Recovery in CNNs

    [https://arxiv.org/abs/2611.24166](https://arxiv.org/abs/2611.24166)

    该论文提出SoU-RHS，一种将正则化马蹄铁收缩与不确定性来源稀疏全局-局部机制相结合的贝叶斯结构化稀疏方法，能够稳定地从过参数化CNN中恢复出规模一致且适应数据的紧凑架构。

    

    高度参数化的卷积神经网络通常能取得很强的预测性能，但其内存、计算和部署成本促使人们寻找从大型参考模型中恢复紧凑架构的方法。我们提出了SoU-RHS，这是一种贝叶斯结构化稀疏方法，它在正则化马蹄铁收缩的基础上增加了所谓的不确定性来源稀疏全局-局部机制。所得模型锐化了活跃滤波器与非活跃滤波器之间的区分，为架构恢复提供了有效机制。SoU-RHS的一个关键特性是稳定性：在不同的过参数化起始宽度以及合理的剪枝阈值范围内，它都能恢复出总规模和逐层分配相似的紧凑架构。这支持了这样一种解读：该过程识别出的是适应数据的紧凑结构，而不仅仅是简单地对各层进行压缩。

    arXiv:2611.24166v1 Announce Type: new  Abstract: Highly parameterized convolutional neural networks often achieve strong predictive performance, but their memory, computation, and deployment costs motivate the search for methods that recover compact architectures from large reference models. We propose SoU-RHS, a Bayesian structured-sparsity procedure that augments Regularized Horseshoe Shrinkage (RHS) with the so-called Sources of Uncertainty Sparse Global-Local mechanism (SoU-SGL). The resulting model sharpens the separation between active and inactive filters, providing an effective mechanism for architecture recovery. A key property of SoU-RHS is stability: across different overparameterized starting widths and under a reasonable range of pruning thresholds, it recovers compact architectures with similar total size and layer-wise allocation. This supports the interpretation that the procedure is identifying a compact structure adapted to the data, rather than merely compressing eac
    
[^16]: 面向受监管企业的智能体AI平台评估能力框架

    A Capability Framework for Evaluating Agentic AI Platforms in Regulated Enterprises

    [https://arxiv.org/abs/2611.24039](https://arxiv.org/abs/2611.24039)

    该论文提出了一套用于评估受监管企业中智能体AI平台的能力框架，通过对全球银行三个金融服务工作负载上十个平台的九维度评测，将市场划分为在权重变化下依然稳定的三个层级。

    

    智能体AI平台——即LLM智能体在其中进行规划、调用工具、保留记忆并协同操作以改变企业系统状态的系统——其采购速度已超过对其进行评估衡量的速度。公开的智能体基准测试是在固定测试环境中对模型进行评分；而企业购买的是集成平台，其编排、记忆、治理和工具层对最终结果起主导作用，且企业购买这些平台在很大程度上依据的是供应商的演示。我们报告了一项由某全球性银行创新实验室开展的工业界评估项目（“竞赛评测”），其中十个平台——涵盖开源框架、专有产品和超大规模云厂商生态系统——由跨职能评审小组在三个金融服务工作负载（交易对手信用风险、检索增强生成和流程自动化）上，按九个能力维度进行评估。评分结果将该领域划分为三个层级，这些层级在替代权重方案下保持稳定，并揭示了工作负载（摘要在此处被截断）

    arXiv:2611.24039v1 Announce Type: new  Abstract: Agentic AI platforms-systems in which LLM agents plan, invoke tools, retain memory and coordinate to change the state of enterprise systems-are being procured faster than they can be measured. Public agent benchmarks score models in fixed harnesses; enterprises buy integrated platforms whose orchestration, memory, governance and tooling layers dominate outcomes, and they buy them largely on the strength of vendor demonstrations. We report an industrial evaluation programme ("bakeoff") run by the innovation laboratory of a global bank in which ten platforms-spanning open-source frameworks, proprietary products and hyperscaler ecosystems-were assessed by a cross-functional panel on nine capability dimensions across three financial-services workloads: counterparty credit risk, retrieval-augmented generation and process automation. Scores separate the field into three tiers that are stable under alternative weightings, and surface workload-s
    
[^17]: NOTREKS：基于评分的结构学习中的边际独立性

    NOTREKS: Marginal Independence in Score-based Structure Learning

    [https://arxiv.org/abs/2611.24015](https://arxiv.org/abs/2611.24015)

    本文提出 NOTREKS 信息，将变量间边际独立的先验知识转化为“无 trek”图约束，并通过 PSTrek 惩罚项和 trek 约束融入基于评分的因果结构学习，从而在有限样本下提升结构恢复与效应识别效果。

    

    基于评分的结构学习通过在结构方程模型上优化依赖于数据的目标函数来估计因果图。我们研究了如何将“选定变量对之间边际独立”这一先验知识融入此类因果发现方法中。在忠实性假设下，边际独立的变量对之间不可能存在 trek（游走路径），因此该知识转化为图上的“无 trek”约束，我们将其称为 NOTREKS 信息。我们为基于梯度的优化引入了幂级数 trek（PSTrek）惩罚项，并为基于顺序的搜索引入了 trek 约束，同时分析了此类信息为何在有限样本中有所帮助，以及它如何限制搜索空间。我们的数值实验说明了 NOTREKS 信息在因果发现中能起到多大帮助。特别是，在带有 oracle NOTREKS 信息的合成基准实验中，我们展示了这些约束能够同时改善结构恢复效果和基于调整的效应识别。

    arXiv:2611.24015v1 Announce Type: cross  Abstract: Score-based structure learning estimates causal graphs by optimizing a data-dependent objective over structural equation models. We study how prior knowledge that selected pairs of variables are marginally independent can be incorporated into such causal discovery methods. Under faithfulness, marginally independent pairs cannot be connected by a trek, so the knowledge becomes a graphical no-trek constraint, which we call NOTREKS information. We introduce a power-series trek (PSTrek) penalty for gradient-based optimization and a trek constraint for order-based search, and we analyze why such information helps in finite samples and how it restricts the search space. Our numerical experiments illustrate to what extent NOTREKS information helps in causal discovery. In particular, using synthetic benchmarks with oracle NOTREKS information, we show that these constraints can improve both structural recovery and adjustment-based effect identi
    
[^18]: 奖励增益不平衡如何影响排名：重新思考基于量规的奖励聚合

    How Reward Gain Imbalance Shapes Ranking: Rethinking Rubric-Based Reward Aggregation

    [https://arxiv.org/abs/2611.24005](https://arxiv.org/abs/2611.24005)

    该论文将基于量规的奖励聚合中的评分异质形式化为“奖励增益不平衡”概念，并出人意料地发现消除这种不平衡（即使使用真实标签）反而会降低排名准确性。

    

    基于量规的强化学习通过构建奖励来使大型语言模型与人类偏好对齐，通常的做法是平等地聚合评判者分配给各个准则的连续分数。然而，准则之间的评分异质性可能会扭曲候选排名。现有工作主要集中在改进评判者、学习准则权重或优化聚合方法，而评分异质性与排名之间的相互作用仍未得到充分理解。我们将这种异质形式化为“奖励增益不平衡”：每个准则以其自身的奖励增益进入聚合奖励，即该准则在满足与不满足的响应上的期望分数之间的差距。出人意料的是，我们的实验表明，消除这种不平衡，即使使用真实标签，也会降低排名准确性。为了解释这一现象，我们推导出了任意两个候选之间的奖励差异的精确分解，将其分解为基本满足……

    arXiv:2611.24005v1 Announce Type: cross  Abstract: Rubric-based reinforcement learning constructs rewards for aligning large language models with human preferences, typically by equally aggregating continuous scores that a judge assigns to individual criteria. However, scoring heterogeneity across criteria can distort candidate rankings. Existing work has focused on improving judges, learning criterion weights, or refining aggregation, whereas the interaction between scoring heterogeneity and ranking remains insufficiently understood. We formalize this heterogeneity as \emph{reward gain imbalance}: each criterion enters the aggregated reward with its own reward gain, the gap between its expected scores on satisfied and unsatisfied responses. Counterintuitively, our experiments show that removing this imbalance, even with ground-truth labels, lowers ranking accuracy. To explain this, we derive an exact decomposition of the reward difference between any two candidates into a base satisfa
    
[^19]: ReliFlow：面向组合式单细胞扰动预测的可靠性校准残差传输方法

    ReliFlow: Reliability-Calibrated Residual Transport for Combinatorial Single-Cell Perturbation Prediction

    [https://arxiv.org/abs/2611.23999](https://arxiv.org/abs/2611.23999)

    提出ReliFlow——一种条件流匹配模型，通过训练期交叉拟合估计单基因效应可靠性并据此校准加性源分布，将速度场分解为加性参考的已知修正与学习得到的残差传输场，从而在单扰动证据稀疏或缺失的情况下显著提升组合扰动响应的预测精度。

    

    预测组合扰动的转录响应，需要将干预之间的相互作用与单扰动效应估计中的误差区分开来。当单扰动证据稀疏或被完全留出时，这类误差会变得更加显著。我们提出了一种条件流匹配模型，该模型通过仅在训练阶段进行的交叉拟合来估计单基因效应的可靠性，并利用这些估计对加性源分布进行校准。其速度场被分解为对加性参考分布的已知修正项与一个学习得到的残差传输场。在Norman基因扰动数据集上，对于由已观测单扰动构成的未见组合，平均绝对误差（MAE）从0.142降至0.131；当组成单扰动被留出时，MAE从0.198降至0.178。在留出的ComboSciPlex药物组合数据上，MAE从0.191降至0.176。非加性残差相关性从0.574提升至0.638，并且各组件的消融实验……

    arXiv:2611.23999v1 Announce Type: cross  Abstract: Predicting the transcriptional response to combined perturbations requires distinguishing interactions between interventions from errors in their individual effect estimates. The latter become more pronounced when single-perturbation evidence is sparse or withheld. We introduce a conditional flow-matching model that estimates the reliability of single-gene effects from training-only cross-fitting and uses those estimates to calibrate the additive source distribution. The velocity is decomposed into a known correction of the additive reference and a learned residual transport field. On Norman genetic perturbations, reduces mean absolute error (MAE) from 0.142 to 0.131 for unseen combinations of observed singles and from 0.198 to 0.178 when constituent singles are withheld. On held-out ComboSciPlex drug combinations, MAE decreases from 0.191 to 0.176. Non-additive residual correlation improves from 0.574 to 0.638, and component ablations
    
[^20]: 具有一致有界Lipschitz常数的最优深度非参数回归

    Optimal Deep Nonparametric Regression with Uniformly Bounded Lipschitz Constants

    [https://arxiv.org/abs/2611.23979](https://arxiv.org/abs/2611.23979)

    本文在分裂谱范数乘积约束下建立了深度ReLU网络的尖锐逼近理论，并提出带惩罚的最小二乘估计器，使深度非参数回归在保持一致有界Lipschitz常数的同时达到最优统计收敛率。

    

    在涉及含噪协变量或估计特征的许多应用中，深度神经网络估计器必须具备对输入扰动的稳定性。权重范数约束被广泛用于通过限制神经网络的Lipschitz常数来控制对输入扰动的敏感度。然而，在标准范数乘积固定界下的现有逼近结果不足以建立最优的统计收敛率。本文在分裂谱范数乘积约束下为深度ReLU网络建立了一个尖锐的逼近理论。该理论结果表明，随着网络深度的增加，足够光滑的Hölder函数可以在范数乘积界保持固定的情况下被一致地逼近到任意精度。随后，我们针对深度非参数回归提出了一个带惩罚项的最小二乘估计器。所得估计器在具有一致有界Lipschitz常数的同时，仍能达到最优的统计收敛率。

    arXiv:2611.23979v1 Announce Type: cross  Abstract: Stability under input perturbations is necessary for deep neural network estimators in many applications involving noisy covariates or estimated features. Weight norm constraints are widely used to control sensitivity to input perturbations by bounding the Lipschitz constant of neural networks. However, existing approximation results under fixed bounds on standard norm products are insufficient to establish optimal statistical rates. In this paper, we establish a sharp approximation theory for deep ReLU networks under a split spectral norm product constraint. This theoretical result shows that sufficiently smooth H\"{o}lder functions can be uniformly approximated to arbitrary accuracy as the network depth increases, with the norm-product bound held fixed. We then propose a penalized least squares estimator for deep nonparametric regression. The resulting estimator has a uniformly bounded Lipschitz constant while still achieving the opt
    
[^21]: 模态缺失下冻结多模态模型的风险可控修正

    Risk-Controlled Correction for Frozen Multimodal Models under Missing Modalities

    [https://arxiv.org/abs/2611.23886](https://arxiv.org/abs/2611.23886)

    本文提出用贝叶斯差距刻画冻结多模态模型在模态缺失后仍可修正的误差部分，并证明对条件正确的更全模态模型的输出在剩余模态条件下取平均，无需标签即可恢复贝叶斯预测目标，从而实现风险可控的预测修正。

    

    已部署的多模态预测器在其权重无法再被修改之后，可能面临模态缺失的问题。这引出了一个训练时修复方法无法回答的问题：由此产生的误差中，有多少部分仍可以从剩余信息中得到修正？我们证明，这部分误差由贝叶斯差距来刻画，即在给定已观测模态的条件下，模型预测与贝叶斯预测之间的差异。对于交叉熵和平方误差，超额风险以损失的几何结构度量这一差距；已观测模态中的模糊性是无法被消除的。因此，一个严重退化的预测并不一定是可修正的。估计贝叶斯预测通常需要标签。然而，当一个观察到更多模态的模型在条件上正确时，通过对其在剩余模态条件下的输出取平均，可以在无需检索标签的情况下恢复这一目标；在相同的固定混合与邻居设置下，其期望表现不会更差。

    arXiv:2611.23886v1 Announce Type: new  Abstract: A deployed multimodal predictor may face missing modalities after its weights can no longer be changed. This raises a question that training-time repair does not answer: what part of the resulting error is still correctable from the information that remains? We show that this part is characterized by the Bayes gap, the difference between the model's prediction and the Bayes prediction given the observed modalities. For cross-entropy and squared error, excess risk measures this gap in the geometry of the loss; ambiguity in the observed modalities cannot be removed. Thus, a badly degraded prediction is not necessarily a correctable one. Estimating the Bayes prediction normally requires labels. However, when a model observing more modalities is conditionally correct, averaging its outputs conditional on the remaining modalities recovers this target without retrieval labels; with the same fixed blend and neighbors, it is no worse in expected
    
[^22]: 无穷范数-Wasserstein 分布鲁棒主成分分析：精确重构、有限样本保证与高效算法

    Infinity-Wasserstein Distributionally Robust Principal Component Analysis: Exact Reformulation, Finite-Sample Guarantee, and Efficient Algorithm

    [https://arxiv.org/abs/2611.23877](https://arxiv.org/abs/2611.23877)

    该论文提出基于无穷范数-Wasserstein 距离的分布鲁棒 PCA 模型，实现了精确的有限维重构、最坏情况分布的显式刻画、有限样本理论保证与高效算法，并揭示其目标函数通过保留经验协方差之外的逐样本残差信息，使分布鲁棒性能够重塑所学的主子空间。

    

    主成分分析（PCA）是一种基础且广泛使用的降维与表示学习技术，然而其经验公式仍然容易受到分布不确定性的影响。尽管分布鲁棒优化（DRO）提供了一种自然的补救途径，但现有的基于二阶 Wasserstein 距离的 PCA 鲁棒模型要么保留了名义 PCA 子空间，要么缺乏精确的重构形式。本文展示了无穷范数-Wasserstein 距离如何能够同时兼顾这两个理想的性质。对于由此得到的 PCA 分布鲁棒优化模型，我们推导出了精确的有限维重构形式，并显式地识别出达到内层最大值的最坏情况分布。至关重要的是，重构后的目标函数在经验协方差矩阵之外还保留了逐样本的残差信息，揭示了一种分布鲁棒性借此重塑所学主子空间的机制。在真实分布满足温和假设的条件下……（原文摘要在此处被截断）

    arXiv:2611.23877v1 Announce Type: cross  Abstract: Principal component analysis (PCA) is a fundamental and ubiquitous technique for dimension reduction and representation learning, yet its empirical formulation remains vulnerable to distributional uncertainty. While distributionally robust optimization (DRO) offers a natural remedy, existing order-2 Wasserstein models of PCA either preserve the nominal PCA subspace or lack an exact reformulation. This paper demonstrates how the infinity-Wasserstein distance can reconcile these two desirable properties. For the resulting DRO model of PCA, we derive an exact finite-dimensional reformulation and explicitly identify a worst-case distribution attaining the inner maximum. Crucially, the reformulated objective retains samplewise residual information beyond the empirical covariance matrix, revealing a mechanism through which distributional robustness can reshape the learned principal subspace. Under the mild assumption that the true distributi
    
[^23]: 两块浅层ReLU模式簇的定义方程

    Defining equations for two-block shallow ReLU pattern varieties

    [https://arxiv.org/abs/2611.23846](https://arxiv.org/abs/2611.23846)

    本文完整地代数刻画了两块浅层ReLU模式簇，证明由激活模式的秩约束所产生的行列式方程生成其定义理想，从而解决了Alexandr-Montúfar提出的猜想。

    

    我们给出了两块浅层ReLU模式簇的完整代数刻画。我们证明了由激活模式所施加的秩约束产生的行列式方程生成其定义理想，从而证明了Alexandr-Montúfar的一个猜想。该集合论层面的证明利用了矩阵束轨道闭包在Kronecker标准形下的刻画。

    arXiv:2611.23846v1 Announce Type: cross  Abstract: We give a complete algebraic characterization of two-block shallow ReLU pattern varieties. We show that the determinantal equations arising from rank constraints imposed by the activation patterns generate the defining ideals, thereby proving a conjecture of Alexandr-Mont\'ufar. The set-theoretic proof uses a characterization of orbit closures of matrix pencils in terms of the Kronecker canonical form.
    
[^24]: 面向约束主动线性回归的改进杠杆值分数采样方法

    Improved Leverage Score Sampling for Constrained Active Linear Regression

    [https://arxiv.org/abs/2611.23801](https://arxiv.org/abs/2611.23801)

    提出椭球正则化杠杆值分数采样，通过外接椭球的形状矩阵刻画约束集几何并将采样偏向重要方向，使约束主动线性回归的查询复杂度由各向异性有效维度决定，从而改进各向异性约束下的查询效率。

    

    本文研究约束线性回归的主动学习问题。给定设计矩阵 A∈R^{n×d}、约束集 C⊆R^d 以及对 b∈R^n 的查询访问，我们寻求在尽可能少的查询次数下得到 min_{x∈C}‖Ax−b‖₂² 的近似解。该问题出现在众多科学与工程应用中，随机化杠杆值分数采样为求解该问题提供了一种强大方法。然而，现有的杠杆值分数方法要么忽略约束，要么通过岭正则化来编码约束，导致在各向异性约束集上的查询复杂度不够理想。我们提出椭球正则化杠杆值分数：用约束集的一个外接椭球来近似该约束集，并利用其形状矩阵将采样偏向于约束几何下更重要的方向。我们证明查询复杂度由一个各向异性有效维度 d_eff(C) 决定。

    arXiv:2611.23801v1 Announce Type: cross  Abstract: In this paper we study active learning for constrained linear regression. Given a design matrix $A\in\mathbb{R}^{n\times d}$, a constraint set $C\subseteq\mathbb{R}^d$, and query access to $b\in\mathbb{R}^n$, we seek an approximate solution to $\min_{x\in C}\|Ax-b\|_2^2$ using as few queries as possible. This problem arises in many scientific and engineering applications, and randomized leverage score sampling provides a powerful method to solve it. However, existing leverage score methods ignore the constraint or encode it through ridge regularization, suffering unfavorable query complexity for anisotropic constraint sets. We propose ellipsoid-regularized leverage scores, where we approximate the constraint set by an outer ellipsoid and use its shape matrix to bias sampling toward directions that matter under the constraint geometry. We prove that the query complexity is governed by an anisotropic effective dimension $d_{\text{eff}}(C
    
[^25]: 可变长度扩散万物：任意状态空间上的任意长度生成建模

    Variable Length Diffuse Everything: Arbitrary Length Generative Modeling on Arbitrary State Spaces

    [https://arxiv.org/abs/2611.23684](https://arxiv.org/abs/2611.23684)

    提出了可变长度扩散万物（VL-DE）框架，通过基于严谨理论的插入机制而非数据填充，实现了任意状态空间上可变长度多模态数据的简单通用生成。

    

    扩散模型和基于流的生成模型在各类应用中取得了显著成果，但大多数方法假设数据具有固定的维度或长度。与此同时，现实世界的问题（如布局生成）通常涉及可变长度的多模态数据，例如连续的边界框坐标与离散的类别配对。现有方法通常通过将数据填充到固定尺寸，或引入专门的目标函数来显式建模长度，这增加了复杂性并限制了通用性。我们提出了可变长度扩散万物（VL-DE），这是一个用于任意状态空间上可变长度多模态生成的简单而通用的框架。VL-DE 不采用填充方式，而是通过一个基于严谨理论框架的插入机制来处理可变长度，从而得到易于处理的训练目标。此外，利用 VL-DE 所基于的加性形式生成器这一特性……

    arXiv:2611.23684v1 Announce Type: new  Abstract: Diffusion and flow-based generative models have achieved remarkable results across diverse applications, but most assume data of fixed-dimension/length. Meanwhile, real-world problems such as layout generation often involve variable-length, multimodal data, such as continuous bounding box coordinates paired with discrete types. Existing methods typically address variable length by padding data to a fixed size or by introducing specialized objectives to model lengths explicitly, adding complexity and limiting generality. We propose Variable Length Diffuse Everything (VL-DE), a simple and general framework for variable-length multimodal generation on arbitrary state spaces. Instead of padding, VL-DE handles variable length through an insertion-based mechanism grounded in a principled theoretical framework, yielding a tractable training objective. In addition, by utilizing the fact that VL-DE is based on a generator with an additive form, w
    
[^26]: 免费的午餐：梯度平均使Newton Sketch达到最优统计效率

    A Free Lunch: Gradient Averaging Achieves Optimal Statistical Efficiency in Newton Sketch

    [https://arxiv.org/abs/2611.23671](https://arxiv.org/abs/2611.23671)

    该论文证明，在草拟牛顿方法中仅需将随机梯度替换为滑动平均（无需额外计算开销、不改变内部求解器收缩界），即可缓解甚至消除草拟带来的计算—统计权衡，实现最优统计效率。

    

    我们研究在线草拟牛顿方法的统计推断问题，其中每个牛顿系统由内部随机草拟求解器近似求解。已有研究揭示了一种计算—统计权衡：草拟求解器（及其加速变体）将牛顿方法每步的计算成本从 O(d^3) 降低至 O(d^2)，但草拟所引入的随机性也会增大最后迭代点和平均牛顿迭代点的极限协方差。在本工作中，我们通过梯度平均来解决这一权衡。我们证明，仅仅将每步的随机梯度替换为滑动平均，就可以缓解、甚至在某些情况下消除 Newton sketch 中的这一权衡。重要的是，梯度平均不需要任何额外的预言机评估，并且不会改变内部草拟求解器的收缩界。特别地，我们证明了带梯度平均的最后草拟牛顿迭代点可以达到……（摘要在此处截断）

    arXiv:2611.23671v1 Announce Type: cross  Abstract: We study statistical inference for online sketched Newton methods, in which each Newton system is approximately solved by an inner randomized sketching solver. Existing work reveals a computational--statistical trade-off: the sketching solver (and its accelerated variants) reduces the per-step cost of Newton methods from $O(d^3)$ to $O(d^2)$, but the randomness introduced by sketching also inflates the limiting covariance of both the last and the averaged Newton iterates. In this work, we address this trade-off via gradient averaging. We show that simply replacing the stochastic gradient with a moving average at each step can mitigate, and in some cases eliminate, the trade-off in Newton sketch. Importantly, gradient averaging requires no additional oracle evaluations and leaves the contraction bound of the inner sketching solver unchanged. In particular, we prove that the last sketched Newton iterate with gradient averaging attains a 
    
[^27]: 时变非齐次线性马尔可夫决策过程中的高效方差感知强化学习

    Efficient Variance-Aware Reinforcement Learning in Time-Inhomogeneous Linear MDPs

    [https://arxiv.org/abs/2611.23596](https://arxiv.org/abs/2611.23596)

    本文提出计算高效的算法 VA-LSVI-UCB++，在时变非齐次线性 MDP 中实现了方差依赖的遗憾界，并消除了以往方差感知分析中即使在确定性环境下仍然存在的 $\tilde{O}(K^{1/4})$ 多项式加性项。

    

    我们研究在具有特征维度 $d$、时间跨度 $H$ 和 $K$ 个回合的回合制时变非齐次线性马尔可夫决策过程（MDP）中的方差依赖强化学习问题。尽管近期工作已为线性混合 MDP 建立了具有 $\tilde{O}(1)$ 确定性情形开销的方差依赖遗憾界（Zhao et al., 2023），并为线性 MDP 建立了最坏情形下接近极小极大最优的遗憾 $\tilde{O}(d\sqrt{H^3 K} + d^7 H^8)$（He et al., 2023），但现有的针对线性 MDP 的方差感知分析（Li and Sun, 2023; Huang et al., 2023）依赖于一个粗糙的集中不等式步骤，导致其遗憾保证即使在确定性环境中仍保留多项式 $\tilde{O}(K^{1/4})$ 的加性项。我们提出 VA-LSVI-UCB++，这是一种计算高效的算法，实现了方差依赖的遗憾界 $\tilde{O}(d\sqrt{H\min\{\mathrm{Var}_K^*,\mathrm{Var}_K^\pi\}}+d^5H^4)$，其中 $\mathrm{Var}_K^*$ 和 $\mathrm{Var}_K^\pi$ 分别表示最优策略和策略 $\pi$ 的累积方差（摘要在此处被截断）。

    arXiv:2611.23596v1 Announce Type: new  Abstract: We study variance-dependent reinforcement learning in episodic time-inhomogeneous linear Markov decision processes (MDPs) with feature dimension $d$, horizon $H$, and $K$ episodes. While recent works have established variance-dependent regret with $\tilde{O}(1)$ deterministic-regime overhead for linear mixture MDPs (Zhao et al., 2023) and worst-case nearly minimax optimal regret $\tilde{O}(d\sqrt{H^3 K} + d^7 H^8)$ for linear MDPs (He et al., 2023),, existing variance-aware analyses for linear MDPs (Li and Sun, 2023; Huang et al., 2023) rely on a coarse concentration step whose resulting regret guarantees retain a polynomial $\tilde{O}(K^{1/4})$ additive term even in deterministic environments. We propose VA-LSVI-UCB++, a computationally efficient algorithm that achieves the variance-dependent regret bound $\tilde{O}(d\sqrt{H\min\{\mathrm{Var}_K^*,\mathrm{Var}_K^\pi\}}+d^5H^4)$, where $\mathrm{Var}_K^*$ and $\mathrm{Var}_K^\pi$ denote th
    
[^28]: 潜在结构块模型中的有界阶数不可辨识性：闭式纤维解

    Bounded-Order Non-Identifiability in Latent Structure Block Models: The Fiber in Closed Form

    [https://arxiv.org/abs/2611.23547](https://arxiv.org/abs/2611.23547)

    该论文为位置位于K条射线上的潜在结构块模型给出了有限阶模体计数下不可辨识配置的一维纤维闭式解，并证明友谊图F_{K-1}是唯一边最小分离子，因此不存在对所有K都适用的固定模体阶数。

    

    图子由其所有同态密度完全决定，但基于模体的方法只使用有限多个。我们在一个自然的模型族中精确计算了这种截断的代价：位置位于K条射线上的潜在结构块模型，等价于秩为2的度校正块模型。对于给定的径向律，在所有少于2K-1个顶点的图上一致的配置构成一个一维纤维，我们以带有显式端点的闭式形式给出该纤维；友谊图F_{K-1}是其唯一的边最小分离子，因此不存在对所有K都适用的固定模体阶数。分离信号最多被抑制m_1^{2(K-1)}倍，其中m_1是均方斜率，在K=4时约为10^{-4}：在任何边密度下，从模体计数中检测该信号需要单个图中约10^{15}个节点。然而，信息仍然存在于图中；纤维属于统计量本身，而非模型。常见的谱方法流程……（原文摘要在此处截断）

    arXiv:2611.23547v1 Announce Type: cross  Abstract: A graphon is determined by all of its homomorphism densities, but motif-based methods use finitely many. We compute exactly what that truncation costs in a natural family: latent structure block models whose positions lie on $K$ rays, equivalently degree-corrected block models of rank two. For a given radial law, the configurations that agree on every graph with fewer than $2K-1$ vertices form a one-dimensional fiber, which we give in closed form with explicit endpoints; the friendship graph $F_{K-1}$ is its unique edge-minimal separator, so no fixed motif order serves every $K$. The separating signal is suppressed by a factor of at most $m_1^{2(K-1)}$, where $m_1$ is the mean squared slope, about $10^{-4}$ at $K=4$: detecting it from motif counts would take about $10^{15}$ nodes in a single graph, at any edge density. The information is nonetheless in the graph; the fiber belongs to the statistic, not the model. Common spectral pipeli
    
[^29]: 平行输运卷积在黎曼度量扰动下的稳定性

    Stability of Parallel Transport Convolution under Riemannian Metric Perturbations

    [https://arxiv.org/abs/2611.23543](https://arxiv.org/abs/2611.23543)

    本文证明了在紧流形上，平行输运卷积的输出在 $L^2$ 意义下对黎曼度量的 $C^2$ 扰动具有连续稳定性，为此发展了测地线、平行输运、雅可比场与指数映射的扰动估计技术。

    

    arXiv:2611.23543v1 公告类型： new 摘要： 黎曼流形上的几何卷积依赖于底层定义域的几何结构，因此度量扰动下的稳定性成为几何深度学习中的一个基本问题。本文借助微分几何与分析的方法研究了平行输运卷积的这一问题。在一个固定的紧流形上，我们证明：对于连续的输入信号以及支撑在充分小法向邻域内的连续参考核，在给定的局部化与输运路径约定下，卷积输出在 $L^2$ 意义下关于黎曼度量（$C^2$ 拓扑）是连续依赖的。分析过程建立了关于测地线、平行输运、雅可比场和指数映射的扰动估计，并利用这些估计来比较输运后的核。一个核心难点是割迹，它阻碍了对整个流形上的输运构造进行一致比较。

    arXiv:2611.23543v1 Announce Type: new  Abstract: Geometric convolutions on Riemannian manifolds depend on the geometry of the underlying domain, making stability under metric perturbations a fundamental question in geometric deep learning. This paper studies this question for parallel transport convolution through methods from differential geometry and analysis. On a fixed compact manifold, we establish that, for a continuous input signal and a continuous reference kernel supported in a sufficiently small normal neighborhood, the convolution output depends continuously in $L^2$ on the Riemannian metric in the $C^2$-topology, under the specified localization and transport-path conventions. The analysis develops perturbation estimates for geodesics, parallel transport, Jacobi fields, and exponential maps, and uses these estimates to compare the transported kernels. A central difficulty is the cut locus, which obstructs uniform comparison of the transport construction over the entire mani
    
[^30]: 矩阵最小二乘解的谱分布与椭球拟合问题

    The Spectral Distribution of Matrix Least Squares Solutions, and the Ellipsoid Fitting Problem

    [https://arxiv.org/abs/2611.23431](https://arxiv.org/abs/2611.23431)

    本文研究了未知量为对称矩阵的随机线性系统最小二乘解的谱分布并给出其支撑集的闭式描述，进而应用于椭球拟合问题，证明了最小二乘候选解在 n ≈ d²/10 处存在半正定性阈值。

    

    我们研究未知量为对称矩阵的随机线性系统的最小二乘解的谱分布。在归一化和各向同性的设定下，我们推导出了谱分布支撑集的闭式描述。作为应用，我们考虑椭球拟合问题：给定 $\mathbb{R}^d$ 中的 $n$ 个标准高斯向量，是否存在一个中心椭球经过所有这些点？最近有研究表明，该问题在 $n \approx d^2/4$ 处存在尖锐的相变。在可满足性方面，与早期结果不同，这些尖锐的结果并不依赖于最小二乘候选解。我们证明了早期工作中所考虑的最小二乘候选解在 $n \approx d^2/10$ 处存在半正定性阈值，这一现象此前已通过数值方法观察到，并由统计物理方法非严格地预测。此外，我们还证明了相应的最小二乘对偶证书……

    arXiv:2611.23431v1 Announce Type: cross  Abstract: We study the spectral distribution of the least squares solution to a random linear system whose unknown is a symmetric matrix. In the normalized and isotropic setting, we derive a closed-form description of the support of the spectral distribution. As an application, we consider the ellipsoid fitting problem: given $n$ standard Gaussian vectors in $\mathbb{R}^d$, does there exist a centered ellipsoid passing through all of them? It was recently shown that this problem has a sharp phase transition at $n \approx d^2/4$. On the satisfiability side, unlike earlier results, these sharp results do not rely on least squares candidates. We show that the least squares candidates considered in earlier works have a threshold of positive semi-definiteness at $n \approx d^2/10$, which was previously observed numerically and predicted non-rigorously using statistical physics methods. Moreover, we show that the corresponding least squares dual certi
    
[^31]: 子空间方向何时对LoRA重要？少样本自适应中幅值原则的情境分析

    When Does Subspace Direction Matter for LoRA? Regime Analysis of the Magnitude Principle in Few-Shot Adaptation

    [https://arxiv.org/abs/2611.23428](https://arxiv.org/abs/2611.23428)

    本文证明LoRA初始子空间方向的重要性取决于情境：在少样本且初始化子空间能捕捉任务结构时方向对齐至关重要，并提出对齐系数ρ来预测增益、以及PivotLoRA进行任务对齐初始化。

    

    我们证明LoRA适配器的初始输入子空间方向的重要性是依赖于情境的，并通过理论与实验刻画了这种依赖关系。这一结论修正了近期认为谱初始化仅通过更新幅值起作用的观点。当有标签数据相对于表示维度较为稀缺，且用于初始化的激活子空间能够捕捉有监督任务结构时，方向的影响最为显著。在这一情境下，适配器的行空间对其初始点保持敏感，使得任务对齐的初始化尤为有用。基于尖峰协方差的扰动分析刻画了所估计子空间的有限样本稳定性。对LoRA训练第一步的分析解释了子空间与任务对齐为何重要，并由此引出一个对齐系数ρ，该系数能够紧密追踪各分类任务上的性能提升。我们通过PivotLoRA验证了这一解释，该方法从……初始化适配器（原文摘要在此处截断）。

    arXiv:2611.23428v1 Announce Type: new  Abstract: We show that the importance of a LoRA adapter's initial input-subspace direction is regime-dependent, and we characterize this dependence in theory and experiment. This refines the recent view that spectral initializers help through update magnitude alone. Direction matters most when labeled data are scarce relative to the representation dimension and the activation subspace used for initialization captures supervised task structure. In this regime, the adapter's row span remains sensitive to its starting point, making task-aligned initialization especially useful. A spiked-covariance perturbation analysis characterizes the finite-sample stability of the estimated subspace. A first-step analysis of LoRA training explains why the subspace's alignment with the task matters and motivates an alignment coefficient $\rho$ that closely tracks gains across classification tasks. We test this account with PivotLoRA, which initializes adapters from
    
[^32]: 哪些Token、以何种顺序、多快？关于在线策略蒸馏目标的理论分析

    Which Tokens, in What Order, and How Fast? A Theoretical Analysis of On-Policy Distillation Objectives

    [https://arxiv.org/abs/2611.23397](https://arxiv.org/abs/2611.23397)

    本文首次从理论上剖析在线策略蒸馏目标的选择机制，揭示序列级OPD存在导致反向学习顺序的“未来一致性偏差”，而Token级目标可避免该偏差，并阐明了散度选择如何影响Token的优先级、学习顺序与收敛效率。

    

    在线策略蒸馏训练学生语言模型基于其自身的生成内容来匹配教师模型。在使用OPD时，必须选择是在序列级别还是Token级别应用它，以及最小化哪种散度。然而，尽管OPD在实践中被广泛采用，关于哪种OPD目标适用于特定设置的形式化理解仍然有限。为弥补这一空白，我们从理论上分析了OPD目标的选择如何决定哪些Token被优先处理、以什么顺序以及效率如何。我们的分析揭示，序列级OPD表现出一种“未来一致性偏差”，而短视的Token级目标可以避免这种偏差。这种偏差导致了一种反向学习顺序：序列级OPD首先在较靠后的Token位置上匹配教师模型，而在较早的位置上，它可能偏好那些在教师模型下概率较低、因而通常不合意的Token。至于散度的选择，我们证明……（摘要在此处截断）

    arXiv:2611.23397v1 Announce Type: cross  Abstract: On-policy distillation (OPD) trains a student language model to match a teacher model based on its own generations. When using OPD, one must choose whether to apply it at the sequence or token level and which divergence to minimize. However, despite the wide adoption of OPD in practice, the formal understanding of which OPD objective is appropriate for a given setting is limited. Toward addressing this gap, we theoretically analyze how the choice of OPD objective governs which tokens are prioritized, in what order, and how efficiently. Our analysis reveals that sequence-level OPD exhibits a future agreement bias, which myopic token-level objectives avoid. This bias induces a reverse learning order, in which sequence-level OPD first matches the teacher at later token positions, while at earlier positions it can favor tokens that are improbable under the teacher, and thus often undesirable. As for the choice of divergence, we prove that 
    
[^33]: AI模型选择中质量-资源权衡的微分理论：多资源前沿及其极小极大价格

    A Differential Theory of Quality-Resource Trade-offs for AI Model Selection: The Multiresource Frontier and Its Minimax Price

    [https://arxiv.org/abs/2611.23324](https://arxiv.org/abs/2611.23324)

    该论文提出一种微分理论，将AI模型选择的质量-资源前沿表示为与价格无关的平面曲线，通过有效资源通道数与剩余对数误差深度之间的内在关系刻画了多资源权衡的几何结构与极小极大价格。

    

    在不同资源预算下比较AI模型会产生将误差与延迟、参数数量、训练计算量和能耗联系起来的质量-资源前沿。偏移幂律可以概括这类前沿，但其拟合参数既无法揭示与价格无关的几何结构，也无法揭示权衡如何在多个资源通道间延伸。本文通过发展一种使该结构显式化的微分理论来填补这两个空白。每个前沿都容许一种与价格无关的平面曲线表示，将有效活跃资源通道数 $N_{\mathrm{eff}} = s^2/\kappa$（斜率平方与曲率之比）与剩余的对数误差深度 $\Delta S = \ln(\varepsilon/\varepsilon_{\infty})$ 相关联。这种表示将内在形状与资源轴的尺度和原点分离开来。完全单调性与可除性导出了微分不等式和前沿界。当N个资源中的每个……（摘要在此处截断）

    arXiv:2611.23324v1 Announce Type: cross  Abstract: Comparing AI models across resource budgets yields quality-resource frontiers relating error to latency, parameter count, training compute, and energy. Offset power laws summarize such frontiers, but their fitted parameters reveal neither the price-independent geometry nor the manner in which trade-offs extend across several resource channels. This paper addresses both gaps by developing a differential theory in which this structure is made explicit. Each frontier admits a price-independent plane-curve representation relating the effective number of active resource channels, $N_{\mathrm{eff}} = s^2/\kappa$ (squared slope over curvature), to the remaining log-error depth, $\Delta S = \ln(\varepsilon/\varepsilon_{\infty})$. This representation separates intrinsic shape from the scale and origin of the resource axis. Complete monotonicity and divisibility yield differential inequalities and frontier bounds. When each of $N$ resources cont
    
[^34]: 在线平滑校准的匹配下界

    A Matching Lower Bound for Online Smooth Calibration

    [https://arxiv.org/abs/2611.16097](https://arxiv.org/abs/2611.16097)

    本文通过构造自适应对手和遗忘对手，证明在线平滑校准误差的极小极大下界为 Ω(√T)，填补了已知的上下界差距，确立了平滑校准误差的极小极大最优性。

    

    我们研究序列化二元预测中的平滑校准误差，这是一种衡量概率预测偏离校准程度的指标，其对预测的微小扰动具有鲁棒性。已知极小极大平滑校准误差介于 Ω(T^{1/3}) 与 O(√T) 之间（Qiao 和 Zheng，2024），其中 T 为轮数。我们对自适应对手和遗忘对手这两类情形均填补了这一空白。我们构造了一个自适应对手，其结果均值随预测者的累积残差而移动，这使得补偿过去的残差变得代价高昂，从而迫使任何预测者产生 Ω(√T) 的期望平滑校准误差。通过针对预测者的平均预测来模拟该对手，可以得到一个具有相同保证的遗忘对手。这些下界与已知的 O(√T) 上界在普适常数范围内匹配，因此是极小极大最优的。我们的结果还意味着匹配的 Θ(√（原文摘要至此被截断）

    arXiv:2611.16097v2 Announce Type: replace  Abstract: We study smooth calibration error in sequential binary prediction, a measure of how far probabilistic forecasts are from being calibrated that is robust to small perturbations of the forecasts. The minimax smooth calibration error is known to lie between $\Omega(T^{1/3})$ and $O(\sqrt T)$(Qiao and Zheng, 2024), where $T$ is the number of rounds. We close this gap for both adaptive and oblivious adversaries. We construct an adaptive adversary whose outcome mean moves with the forecaster's cumulative residual, which makes compensating for past residuals costly and forces every forecaster to incur $\Omega(\sqrt T)$ expected smooth calibration error. Simulating this adversary against the forecaster's average prediction yields an oblivious adversary with the same guarantee. These lower bounds match the known $O(\sqrt T)$ upper bound up to universal constants, and are therefore minimax optimal. Our results also imply matching $\Theta(\sqrt
    
[^35]: 通过利用实现探索：个性化联邦强化学习中奖励多样性的福音

    Exploration via Exploitation: The Blessing of Reward Diversity in Personalized Federated RL

    [https://arxiv.org/abs/2611.12964](https://arxiv.org/abs/2611.12964)

    该论文将个性化联邦强化学习中的奖励异质性从协作障碍转变为结构性优势，提出PF-UCBVI和PF-EFVI算法，证明奖励多样性可替代显式探索，使纯贪心算法即可实现对数遗憾和最优线性加速。

    

    尽管联邦强化学习（RL）传统上在同质环境中能够保证高效率，但当智能体拥有各自不同的目标时，协作往往会受到阻碍。在本文中，我们颠覆了这一范式，将奖励异质性不再视为达成共识的障碍，而是视为能够催化探索的结构性资产。我们首先提出个性化联邦上置信界值迭代（PF-UCBVI），通过将共享动力学与个性化目标解耦，实现了最优的线性加速。为消除显式探索带来的风险，我们随后提出个性化联邦无探索值迭代（PF-EFVI），这是一种纯贪心算法，利用奖励多样性来确保状态-动作覆盖。我们证明，在足够的奖励多样性条件下，PF-EFVI无需显式探索奖励即可实现对数遗憾。我们的结果表明，智能体之间的分歧将探索负担从算法转移到了（原文在此处截断）。

    arXiv:2611.12964v1 Announce Type: new  Abstract: While federated reinforcement learning (RL) has traditionally guaranteed high efficiency in homogeneous environments, collaboration is often hindered when agents possess distinct goals. In this paper, we flip this paradigm, framing reward heterogeneity not as a hurdle to consensus, but as a structural asset that catalyzes exploration. We first propose Personalized Federated Upper Confidence Bound Value Iteration (PF-UCBVI), which achieves optimal linear speedup by decoupling shared dynamics from personalized objectives. To eliminate the risks of explicit exploration, we then introduce Personalized Federated Exploration-Free Value Iteration (PF-EFVI), a purely greedy algorithm that leverages reward diversity to ensure state-action coverage. We prove that PF-EFVI attains logarithmic regret without explicit exploration bonuses under sufficient reward diversity. Our results reveal that agent disagreement shifts the exploration burden from te
    
[^36]: 路径积分扩散如何决定一个有限自旋玻璃

    How Path Integral Diffusion Decides a Finite Spin Glass

    [https://arxiv.org/abs/2611.11826](https://arxiv.org/abs/2611.11826)

    该论文证明各向同性路径积分扩散等价于经过时间变换的高斯随机局部化，从而表明扩散生成器对自旋玻璃所做决策的内容与具体协议无关，并可由作为鞅的“部分承诺”按磁化对比平方设定的速率逐级刻画。

    

    当AI扩散型生成器对一个受阻挫的自旋玻璃进行采样时，它会首先决定哪些宏观备选方案，以及何时做出决定？我们针对从点源出发的路径积分扩散（PID）型桥扩散模型研究这一问题。我们证明，每个各向同性的PID都是经过时间变换的高斯随机局部化，反之，每个信噪比调度也都由某个PID协议实现。因此，连续时间生成器所决定的内容并不依赖于协议，协议只决定何时决定以及以何种代价决定。我们通过生成器的“部分承诺”来描述其状态：即它在每一时刻分配给由目标所确定的嵌套构型类别的概率。这些部分承诺是鞅，仅沿类别对比移动。它们的系综平均以由磁化强度对比平方精确设定的速率给出关于每个层级的信息。在平衡的、实质上二值的近似中

    arXiv:2611.11826v2 Announce Type: replace-cross  Abstract: When an AI diffusion-type generator samples a frustrated spin glass, which macroscopic alternatives does it decide first, and when? We study this for bridge-diffusion models of the Path Integral Diffusion (PID) type started from a point source. We show that every isotropic PID is a time-changed Gaussian stochastic localization, and conversely every signal-to-noise schedule is realized by a PID protocol. What a continuous-time generator decides therefore does not depend on the protocol, which only sets when and at what cost. We describe the generator's state by its partial commitments: the probabilities it assigns, at each moment, to nested classes of configurations fixed by the target. The partial commitments are martingales that move only along class contrasts. Their ensemble average gives the information about each level at an exact rate set by squared magnetization contrasts. In the balanced, effectively binary approximation
    
[^37]: 面向二元决策的具有可行性保证的线性策略规范性学习

    Prescriptive Learning of Linear Policies with Feasibility Guarantees for Binary Decision-Making

    [https://arxiv.org/abs/2611.10923](https://arxiv.org/abs/2611.10923)

    本文提出一种规范性学习框架，通过在所有训练场景上联合优化线性决策策略与可行分配，为不确定性下的二元决策提供可行性保证，并解决了预测式学习缺乏反事实认知且在分布偏移下易产生不可行决策的问题。

    

    不确定性下的二元决策广泛应用于众多领域，包括能源系统调度、交通运输和人力规划等。尽管机器学习已被越来越多地用于加速或取代此类场景中的混合整数求解器，但现有方法依赖于预测式学习，即训练模型从带标签的数据中复现最优决策，这种方法既缺乏对反事实结果的认知，也无法保证所生成决策的可行性。在分布偏移的情况下，这一局限性会进一步加剧，因为基于历史数据训练的策略可能产生系统性不可行的决策。我们提出了一种面向二元决策的规范性学习框架，以同时解决上述两个局限。该框架不再预测最优标签，而是在所有训练场景上对线性决策策略与可行分配进行联合优化……

    arXiv:2611.10923v2 Announce Type: replace  Abstract: Binary decision-making under uncertainty arises across a wide range of applications, including energy systems scheduling, transportation, and workforce planning. While machine learning has been increasingly applied to accelerate or replace mixed-integer solvers in such settings, existing approaches rely on predictive learning, training models to reproduce optimal decisions from labeled data, which lacks knowledge of counterfactual outcomes and provides no guarantees that the resulting decisions will be feasible. This limitation is further compounded under distributional shift, where policies trained on historical data may produce systematically infeasible decisions. We propose a prescriptive learning framework for binary decision-making that addresses both limitations. Rather than predicting optimal labels, the framework jointly optimizes over linear decision policies and feasible assignments across all training scenarios, allowing f
    
[^38]: 何时准确率才是证据？泛化、验证与信息融合的统一理论

    When Is Accuracy Evidence? A Unified Theory of Generalisation, Validation, and Information Fusion

    [https://arxiv.org/abs/2610.03465](https://arxiv.org/abs/2610.03465)

    该论文提出一个统一的指数框架，将交叉验证准确率转化为真实风险的保守上界，并通过有效折数Keff证明：当各折数据强相关时，单纯增大交叉验证折数K并不能增强统计证据。

    

    K折交叉验证（CV）被广泛用作样本外性能的证据，然而在异构数据下，各折既非独立实验，也非信息量均等。交叉上界验证（CUBV）用真实风险的保守上界取代逐点的CV准确率。本文通过单一的指数框架对CUBV进行了推广，其中泛化间隙的矩生成函数由累积量包络gamma(lambda)所控制。由此得到一族风险界，涵盖Hoeffding界、Bernstein界、依赖感知、PAC-Bayesian以及异构数据源融合等多种情形。对于K折CV，各折间隙之间的相关性通过联合次高斯代理矩阵来建模。在等相关条件下，得到有效折数Keff = K/[1+(K-1)rho]，表明当各折之间存在强相关性时，增大K并不一定能增加统计证据。该框架还被用于……（原文摘要在此处截断）

    arXiv:2610.03465v1 Announce Type: cross  Abstract: K-fold cross-validation (CV) is widely used as evidence of out-of-sample performance, although folds are neither independent experiments nor equally informative under heterogeneous data. Cross Upper-Bound Validation (CUBV) replaces point-wise CV accuracy by conservative upper bounds on true risk. Here we generalise CUBV through a single exponential framework in which the moment-generating function of the generalisation gap is controlled by a cumulant envelope gamma(lambda). This yields a family of risk bounds covering Hoeffding-, Bernstein-, dependency-aware, PAC-Bayesian, and heterogeneous source-fusion settings. For K-fold CV, dependence between fold-wise gaps is modelled through a joint sub-Gaussian proxy matrix. Under equicorrelation, this gives an effective number of folds, Keff = K/[1+(K-1)rho], showing that increasing K does not necessarily increase statistical evidence when folds are strongly dependent. The framework is also ex
    
[^39]: IQS-BO：面向贝叶斯优化的上下文内查询选择

    IQS-BO: In-Context Query Selection for Bayesian Optimisation

    [https://arxiv.org/abs/2610.01269](https://arxiv.org/abs/2610.01269)

    本文提出IQS-BO，一种通过在合成先验上进行监督学习来学习贝叶斯优化查询决策的PFN，可在单次前向传递中完成查询选择，从而避免了代理模型重新拟合和采集函数的数值最大化。

    

    贝叶斯优化（BO）是优化昂贵黑盒函数的强大框架，但通常需要在每次评估步骤中重新拟合代理模型并最大化采集函数。基于先验数据拟合网络（PFNs）的上下文内方法通过在从合成先验中抽取的函数上预训练Transformer来分摊部分成本。PFNs4BO分摊了代理模型的计算成本，但仍依赖于数值最大化的采集函数；而FIBO通过从学习到的密度中采样优化器位置，完全在上下文内执行BO，这固定了决策规则且不使用代理模型。学习型采集函数通过训练好的网络对有限候选集进行评分，但由于缺乏查询的标签，它们只能在先前解决的任务上通过强化学习来学习评分。我们提出了IQS-BO，一个通过在合成先验上进行监督学习来学习查询决策的PFN。在单次前向传递中，IQS-BO……

    arXiv:2610.01269v1 Announce Type: cross  Abstract: Bayesian Optimisation (BO) is a powerful framework for the optimisation of expensive black-box functions, but typically requires refitting a surrogate and maximising an acquisition function at every evaluation step. In-context approaches based on Prior-data Fitted Networks (PFNs) amortise part of this cost by pre-training transformers on functions drawn from synthetic priors. PFNs4BO amortises the surrogate but still relies on a numerically maximised acquisition function, while FIBO performs BO fully in-context by sampling optimiser locations from a learned density, which fixes the decision rule and admits no surrogate. Learned acquisition functions score a finite candidate set with a trained network, but, lacking a label for the query, learn the score by reinforcement learning on previously solved tasks. We propose IQS-BO, a PFN that learns the query decision by supervised learning on synthetic priors. In a single forward pass, IQS-BO
    
[^40]: Swiss-Knife：解码时可重构的外部化多目标对齐框架

    Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time

    [https://arxiv.org/abs/2609.22226](https://arxiv.org/abs/2609.22226)

    该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。

    

    解码时对齐方法通过使用外部奖励对候选续写进行评分并选择得分最大者，来引导一个冻结的语言模型。我们认为这种共享设计只是更大空间中的一个退化点。我们提出Swiss-Knife，一个外部化多目标对齐框架，其中对齐规范是一等运行时对象：可热插拔的评分刀片、批归一化器、成对聚合算子和选择规则。六个公理刻画了可容许的聚合算子，并且我们证明了一个表示定理：每个满足这些公理的算子都具有形式 $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$，这是一个双参数族，其中包含probit和logistic比较规则以及逐点argmax作为命名的坐标。在该族中，成对聚合在对抗性奖励污染下是Lipschitz稳定的，而argmax则不是，且候选批归一化……（原文摘要在此处截断）

    arXiv:2609.22226v1 Announce Type: new  Abstract: Decode-time alignment methods steer a frozen language model by scoring candidate continuations with an external reward and selecting the maximiser. We argue that this shared design is a single degenerate point in a much larger space. We introduce Swiss-Knife, a framework for externalised multi-objective alignment in which the alignment specification is a first-class runtime object: hot-swappable scoring blades, a batch normaliser, a pairwise aggregation operator, and a selection rule. Six axioms characterise the admissible aggregation operators, and we prove a representation theorem: every operator satisfying them has the form $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$, a two-parameter family containing probit and logistic comparison rules and pointwise argmax as named coordinates. Within it, pairwise aggregation is Lipschitz-stable under adversarial reward contamination while argmax is not, and Candidate-Batch Norma
    
[^41]: 坏天才：超越任务特定捷径的反事实引导测试框架演化

    Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts

    [https://arxiv.org/abs/2609.18366](https://arxiv.org/abs/2609.18366)

    提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。

    

    可靠的智能体评估因自动测试框架优化而变得复杂，这类优化方法反复使用已发布的基准 $B_{\mathrm{rel}}$ 来引导一个提议者，该提议者围绕固定的目标智能体编辑提示词、记忆、检索、工具和控制代码。任务保留集虽然改变了语义任务，但基准协议保持不变，因此一个“坏天才”提议者可以生成一个作弊的测试框架，其在发布基准上的性能提升依赖于整个基准范围的捷径。我们提出了反事实测试框架搜索与演化，将测试框架演化建模为在保持有效性的基准反事实上的约束生成问题。在每次提议者更新后，一个挑战者会搜索能大幅摧毁性能提升的可执行协议变换。有效性防火墙检查任务语义是否得到保留，而确认集则决定反事实是否进入有限存档。我们形式化定义了一个精确的捷径中和基准 $B

    arXiv:2609.18366v1 Announce Type: new  Abstract: Reliable agent evaluation is complicated by automatic harness optimization, which repeatedly uses a released benchmark $B_{\mathrm{rel}}$ to guide a Proposer that edits prompts, memory, retrieval, tools, and control code around a fixed target agent. Task holdout varies semantic tasks but leaves the benchmark protocol fixed, so a "bad genius" Proposer can produce a cheating harness whose released-benchmark gain depends on a benchmark-wide shortcut. We introduce Counterfactual Harness Search and Evolution (CHASE), which casts harness evolution as constraint generation over validity-preserving benchmark counterfactuals. After each Proposer update, a Challenger searches for an executable protocol transformation with large gain destruction. A validity firewall checks that task semantics are preserved, while a confirmation set determines whether the counterfactual enters a finite archive. We formalize an exact shortcut-neutralized benchmark $B
    
[^42]: 一种面向大字母表的分层单纯形架构

    A Layered Simplex Architecture for Large Alphabets

    [https://arxiv.org/abs/2608.19908](https://arxiv.org/abs/2608.19908)

    本文提出一种极简的分层单纯形贝叶斯估计器，通过均匀相乘并归一化实现，无需调参，具有显式可计算的遗憾值，并在大字母表概率估计任务上与专门方法竞争，同时揭示缩放规律。

    

    arXiv:2608.19908v1 公告类型：交叉 摘要：在对数损失下对大字母表进行概率估计是一个被广泛研究的问题，已有诸如古德-图灵估计器等著名方法。我们引入并研究了一种新的贝叶斯估计器，具有四个显著特性。首先，其构造极其简单：将来自概率单纯形的独立均匀样本按坐标相乘并重新归一化。深度是唯一的结构参数，而对深度进行平均则消除了调整它的需要。其次，所得混合模型的遗憾值——相对于已知源的编码所支付的超额码长——具有显式且可高效计算的表达式。第三，尽管其简单且无需调整常数，该估计器在多种合成和真实文本基准上，与包括古德-图灵在内的更专门方法相比，表现出竞争力。第四，其遗憾值的可处理性使我们能够识别数据和字母表中的缩放规律。

    arXiv:2608.19908v1 Announce Type: cross  Abstract: Probability estimation over large alphabets under log loss is a well-studied problem, with celebrated methods such as the Good-Turing estimator. We introduce and study a new Bayesian estimator with four notable properties. First, its construction is exceptionally simple: multiply independent uniform draws from the probability simplex coordinate-wise and renormalize. Depth is the only structural parameter, and averaging over depths eliminates the need to tune it. Second, the regret of the resulting mixture, the excess code length it pays relative to a code that knows the source, admits an explicit and efficiently computable expression. Third, despite its simplicity and lack of tuned constants, the estimator is competitive across a diverse set of synthetic and real-text benchmarks with substantially more specialized methods, including Good-Turing. Fourth, the tractability of its regret allows us to identify scaling laws in data, alphabet
    
[^43]: 使用提升树评估模型校准

    Assessing model calibration with boosting trees

    [https://arxiv.org/abs/2606.08084](https://arxiv.org/abs/2606.08084)

    本文提出利用提升树来检验回归模型强校准和自校准的必要条件，并以保险定价等场景中的数值实验验证了该方法的实际应用价值。

    

    回归建模的主要目标是在给定一组特征的情况下近似响应变量的条件均值。如果回归函数所得到的均值估计对于几乎每一组特征都与真实的条件均值相匹配，则称该回归函数是强校准的。由于实践中通常处理的是带有噪声的有限样本观测，追求强校准似乎无法实现。一个较弱的校准概念是自校准，它意味着在给定均值估计的条件下，响应变量的条件期望与该估计相匹配。这一概念非常重要，例如在保险定价中，它确保不同价格群体之间不存在交叉补贴。在本文中，我们证明提升树可用于分别检验强校准和自校准的必要条件。我们方法的实际相关性得到了一个数值例子的支持，其中所提出的检验...

    arXiv:2606.08084v2 Announce Type: replace-cross  Abstract: The main goal in regression modelling consists in approximating the conditional mean of a response given a set of features. A regression function is said to be strongly calibrated if the resulting mean estimates match the true conditional means for almost every set of features. Aiming for strong calibration seems not achievable in practice as one typically deals with finite samples of noisy observations. A weaker notion of calibration is auto-calibration, and it means that the conditional expectation of the response, given a mean estimate, matches this estimate. This notion is important, e.g., in insurance pricing as it ensures no cross-subsidization between different price cohorts. In this paper, we show that boosting trees can be used to test necessary conditions for strong calibration and auto-calibration, respectively. The practical relevance of our approach is supported by a numerical example, in which the proposed tests a
    
[^44]: 从数据到行动：利用人工智能加速炼油厂优化

    From Data to Action: Accelerating Refinery Optimization with AI

    [https://arxiv.org/abs/2605.15085](https://arxiv.org/abs/2605.15085)

    本文通过为线性规划求解器增加记忆功能并引入异常检测工具，能够发现炼油厂优化中输入数据的潜在错误及不同决策层面的异常关系，从而提升后续LP求解结果的可信度和质量。

    

    当今的炼油厂优化需要利用海量数据，这些数据虽然可以由现代线性规划（LP）软件处理，但对结果的解释和应用仍然具有挑战性。大型石化公司使用的模型规模庞大，输入矩阵包含数十万个元素。LP解在数学上是正确的，但模型构建中做了简化，且数据供给可能出现错误。因此，需要进一步的洞察才能信任这些结果。本文的创新之处在于为LP求解器增加了记忆功能，并引入异常检测工具，从而能够发现输入数据中的潜在错误，并找出不同决策层面数据中的异常关系。通过分析LP输入和输出中潜在的“异常”，可以改进后续的LP求解结果。所提出的异常检测工具包含三种方法。（注：原文摘要在此处被截断）

    arXiv:2605.15085v2 Announce Type: replace-cross  Abstract: Nowadays refinery optimization utilizes sheer amounts of data, which can be handled with modern Linear Programming (LP) software, but the interpreting and applying the results remains challenging. Large petrochemical companies use massive models, with hundreds of thousands of input matrix elements. The LP solution is mathematically correct, but simplifications are made in the model, and data supply errors may occur. Therefore, further insight is needed to trust the results. The novelty of the paper consists of enhancing the LP solver with a memory, which makes possible to find potential errors in the input data, finding unusual relations in the data on different decision levels, by introducing an anomaly detection tool. Analyzing the potential "anomalies" in the input and output of LP, the subsequent LP solutions can be improved. The Anomaly Detection tool proposed comprises three methodologies. A transformed version of the pop
    
[^45]: 条件分布匹配的逆向设计

    Inverse Design for Conditional Distribution Matching

    [https://arxiv.org/abs/2605.09439](https://arxiv.org/abs/2605.09439)

    该论文提出了条件分布匹配（CDM）这一新的逆向设计问题类别，即寻找使生成模型的条件分布匹配给定目标分布的输入，并定义了CDMS与CDMO两个问题变体及相应的求解方法MLGD-F。

    

    生成模型是从学习到的分布 $\mathcal{P}(Y \mid X)$ 中采样的强大工具，而逆向设计方法则反转这一映射，以找到能够产生期望点输出 $y^*$ 的输入 $x$。然而，许多设计目标本质上是分布性的而非逐点的，需要考虑 $Y$ 的固有不确定性并针对其特定的分布形式，这是标准逆向设计方法无法解决的任务。为解决这一问题，我们引入了条件分布匹配（CDM），这是生成建模中一类新的逆向设计问题：给定联合分布 $\mathcal{P}(X, Y)$ 和目标分布 $\mathcal{G}(Y)$，找到一个输入 $x^*$，使其诱导的条件分布 $\mathcal{P}(Y \mid X = x^*)$ 与 $\mathcal{G}$ 相匹配。我们正式定义了两个问题变体：条件分布匹配采样（CDMS）和条件分布匹配优化（CDMO）。为求解这些问题，我们提出了 MLGD-F 方法（

    arXiv:2605.09439v2 Announce Type: replace  Abstract: Generative models are powerful tools for sampling from a learned distribution $\mathcal{P}(Y \mid X)$, and inverse-design methods invert this map to find an input $x$ that produces a desired point output $y^*$. However, many design goals are naturally distributional rather than pointwise, incorporating the inherent uncertainty of $Y$ and targeting a specific form for it, a task not addressed by standard inverse design. To address this issue we introduce Conditional Distribution Matching (CDM), a new inverse-design problem class in generative modeling: given a joint distribution $\mathcal{P}(X, Y)$ and a target distribution $\mathcal{G}(Y)$, find an input $x^*$ whose induced conditional distribution $\mathcal{P}(Y \mid X = x^*)$ matches $\mathcal{G}$. We formally define two variants: Conditional Distribution Matching Sampling (CDMS) and Conditional Distribution Matching Optimization (CDMO). To solve these problems, we propose MLGD-F (
    
[^46]: 面向线性马尔可夫决策过程的参数化策略乐观演员-评论家算法

    Optimistic Actor-Critic with Parametric Policies for Linear Markov Decision Processes

    [https://arxiv.org/abs/2603.28595](https://arxiv.org/abs/2603.28595)

    该论文针对有限时域线性马尔可夫决策过程提出了使用参数化对数线性策略的乐观演员-评论家框架，通过引入可处理的logit匹配回归目标和基于朗之万动力学的近似汤普森采样，克服了现有演员-评论家方法在探索假设和隐式策略采样效率方面的理论局限。

    

    尽管演员-评论家方法在实践中取得了成功，但其理论分析存在若干局限性。具体而言，现有的理论工作要么通过做出强假设来回避探索问题，要么分析经过复杂算法修改后不实用的方法。此外，针对线性马尔可夫决策过程（MDP）所分析的演员-评论家方法通常采用自然策略梯度，并通过累积的评论家估计来定义“隐式”策略，而不是学习显式参数化的演员。这类策略的采样在计算上可能代价高昂，导致与环境交互的效率低下。为此，我们聚焦于有限时域线性MDP，提出了一种使用参数化对数线性策略的乐观演员-评论家框架。特别地，我们为演员引入了一个易于处理的logit匹配回归目标。对于评论家，我们通过朗之万动力学（Langevin Mo...原文在此处截断）使用近似汤普森采样。

    arXiv:2603.28595v3 Announce Type: replace  Abstract: Although actor-critic methods have been successful in practice, their theoretical analyses have several limitations. Specifically, existing theoretical work either sidesteps the exploration problem by making strong assumptions or analyzes impractical methods with complicated algorithmic modifications. Moreover, the actor-critic methods analyzed for linear MDPs often employ natural policy gradient and define "implicit" policies through the accumulated critic estimates, rather than learning an explicitly parameterized actor. Such policies can be computationally expensive to sample from, making the environment interactions inefficient. To that end, we focus on the finite-horizon linear MDPs and propose an optimistic actor-critic framework that uses parametric log-linear policies. In particular, we introduce a tractable logit-matching regression objective for the actor. For the critic, we use approximate Thompson sampling via Langevin Mo
    
[^47]: 用于约束采样的动力学朗之万分裂格式

    Kinetic Langevin Splitting Schemes for Constrained Sampling

    [https://arxiv.org/abs/2603.23397](https://arxiv.org/abs/2603.23397)

    本文提出将动力学朗之万动力学的BAOAB和ABOBA对称分裂格式应用于约束采样的惩罚代理目标，并在仅假设强凸性和光滑性（无需Lipschitz Hessian）的条件下建立了非渐近Wasserstein误差界，其随机梯度版本的复杂度显著优于传统Euler型离散化方法。

    

    约束采样，即从限制在凸集上的分布中生成样本的任务，是计算统计学中的一个重要问题。在本文中，我们通过将动力学朗之万动力学的对称BAOAB和ABOBA分裂格式应用于约束目标分布的惩罚代理，提出了新的未校正采样算法，其中包括它们的随机梯度版本。由于惩罚代理的Hessian矩阵在约束集边界处不连续，依赖于Lipschitz Hessian的分析不再适用。因此，我们开发了一种仅基于强凸性和光滑性的分析，证明了一步Wasserstein收缩性质、两种格式的非渐近Wasserstein误差界以及复杂度界。所得的随机梯度版本在复杂度上相比传统Euler型离散化方法有显著改进。数值实验验证了理论结果。

    arXiv:2603.23397v3 Announce Type: replace-cross  Abstract: Constrained sampling, the task of generating samples from a distribution restricted to a convex set, is an important problem in computational statistics. In this paper, we propose new unadjusted sampling algorithms obtained by applying the symmetric BAOAB and ABOBA splittings, of the kinetic Langevin dynamics, to a penalized surrogate of the constrained target. This also includes their stochastic gradient versions. Since the Hessian of the penalized surrogate is discontinuous across the boundary of the constraint set, analyses relying on a Lipschitz Hessian do not apply. We therefore develop an analysis based only on strong convexity and smoothness, proving one-step Wasserstein contraction schemes, non-asymptotic Wasserstein bounds of both schemes and complexity bounds. The resulting complexities for the stochastic gradient versions improve substantially on traditional Euler-type discretizations. Numerical experiments verify ou
    
[^48]: 基于签名核的鲁棒概率预测与尾部事件预测评估指标

    Signature-Kernel Based Evaluation Metrics for Robust Probabilistic and Tail-Event Forecasting

    [https://arxiv.org/abs/2602.10182](https://arxiv.org/abs/2602.10182)

    本文提出了一族基于签名核的删失评估指标，通过将分布主体收缩为单一枢轴点来聚焦尾部区域和高效用区域的评估，从而更准确地衡量概率预测在分布尾部的保真度。

    

    arXiv:2602.10182v2 公告类型：替换 摘要：连续分级概率评分（CRPS）和分位数损失（QL）等标准的基于样本的度量指标被广泛用于概率时间序列预测的评估。然而，这些指标无法捕捉多变量之间的相关性，也无法评估模型捕捉分布尾部的能力。由于分布的主体部分在指标中被过度代表，这些全局指标对分布尾部中高效用事件上的误差不敏感。为了在保持适当评分规则特性的同时，准确衡量分布尾部区域和高效用区域上的分布保真度，我们提出了一族删失签名核度量指标。我们提出的指标通过将分布的主体收缩为单一枢轴点，将预测评估集中于关注区域，即代表高效用或分布尾部的区域。我们在时间序列基础模型（TSFMs）上的基准测试表明，虽然可以为分布形成明确的排名……

    arXiv:2602.10182v2 Announce Type: replace  Abstract: Standard sample-based metrics like Continuous Ranked Probability Score (CRPS) and Quantile Loss (QL) are frequently used in the evaluation of probabilistic time-series forecasting. However, these metrics fail to capture multivariate correlations or assess the ability to capture distribution tails. These global metrics are insensitive to errors on high-utility events in distributional tails due to over-representation of the distribution's body. To accurately measure distributional fidelity on tail-regions and regions of high utility while still having a proper scoring rule, we introduce a family of censored signature kernel metrics. Our proposed metrics concentrate evaluation of forecasts to a focus region, representing high-utility or distributional tails, by collapsing the body of the distribution to a single pivot. Our benchmarks on time-series foundation models (TSFMs) reveal that while a clear ranking can be formed for distributi
    
[^49]: 实用好奇心：基于主动推理的混合学习与优化的统一框架

    Pragmatic Curiosity: A Unified Framework for Hybrid Learning and Optimization via Active Inference

    [https://arxiv.org/abs/2602.06104](https://arxiv.org/abs/2602.06104)

    该论文提出了实用好奇心（PraC）框架，通过主动推理将贝叶斯优化与贝叶斯实验设计统一起来，以信息增益与预期遗憾之间的权衡来指导混合学习与优化场景中的序贯决策。

    

    许多工程和科学工作流程依赖于昂贵的黑盒评估，需要做出既能提升任务性能又能降低不确定性的序贯决策。贝叶斯优化（BO）和贝叶斯实验设计（BED）分别为目标导向的优化和信息探索实验提供了强大但在很大程度上相互独立的方法，对于学习与优化内在耦合的混合场景，现有方法缺乏足够的指导。我们提出了实用好奇心，这是一个通过主动推理实现混合学习与优化的统一框架。PraC 通过权衡关于任务相关潜在符号的信息增益与基于预期遗憾的结果潜力来评估候选查询。这一表述突出了三个操作层面的设计选择：应当澄清哪个潜在量、任务价值如何被编码为遗憾，以及信息增益应当以多大强度进行交换（原文摘要在此处截断）。

    arXiv:2602.06104v3 Announce Type: replace  Abstract: Many engineering and scientific workflows rely on expensive black-box evaluations, requiring sequential decisions that must both improve task performance and reduce uncertainty. Bayesian optimization (BO) and Bayesian experimental design (BED) provide powerful but largely separate treatments of goal-directed optimization and information-seeking experimentation, leaving limited guidance for hybrid settings in which learning and optimization are intrinsically coupled. We propose Pragmatic Curiosity (PraC), a unified framework for hybrid learning and optimization via active inference. PraC evaluates candidate queries by trading information gain about a task-relevant latent symbol against an expected regret-based potential over outcomes. This formulation foregrounds three operational design choices: which latent quantity should be clarified, how task value is encoded as regret, and how strongly information gain should be exchanged agains
    
[^50]: 目标化合成控制方法

    Targeted Synthetic Control Method

    [https://arxiv.org/abs/2602.04611](https://arxiv.org/abs/2602.04611)

    本文提出目标化合成控制（TSC）方法，一种新型两阶段估计器，通过目标化去偏更新直接估计反事实结果，在保证权重稳定性和凸组合可解释性的同时，可灵活结合任意机器学习模型。

    

    合成控制法（SCM）通过构建一个由未处理控制单元加权组合而成的反事实结果，使其匹配处理前的结果轨迹，从而在具有单一处理单元的面板数据中估计因果效应。在本文中，我们提出了目标化合成控制（TSC）方法，这是一种直接估计反事实结果的新型两阶段估计器。具体而言，我们的TSC方法：（1）产生一个目标化去偏估计器，即通过目标化更新来精炼初始权重，从而产生更稳定的权重；（2）确保最终的反事实估计是观测到的控制结果的凸组合，从而使合成控制权重能够被直接解释。TSC方法灵活多样，可以用任意机器学习模型进行实例化。在方法学上，TSC从一个初始的合成控制权重集合出发，通过一维目标化更新……

    arXiv:2602.04611v4 Announce Type: replace-cross  Abstract: The synthetic control method (SCM) estimates causal effects in panel data with a single-treated unit by constructing a counterfactual outcome as a weighted combination of untreated control units that matches the pre-treatment trajectory. In this paper, we introduce the targeted synthetic control (TSC) method, a new two-stage estimator that directly estimates the counterfactual outcome. Specifically, our TSC method (1) yields a targeted debiasing estimator, in the sense that the targeted updating refines the initial weights to produce more stable weights; and (2) ensures that the final counterfactual estimation is a convex combination of observed control outcomes to enable direct interpretation of the synthetic control weights. TSC is flexible and can be instantiated with arbitrary machine learning models. Methodologically, TSC starts from an initial set of synthetic-control weights via a one-dimensional targeted update through 
    
[^51]: 整流流的低维自适应：基于扩散与随机定位的视角

    Low-Dimensional Adaptation of Rectified Flow: A Diffusion and Stochastic Localization Perspective

    [https://arxiv.org/abs/2601.15500](https://arxiv.org/abs/2601.15500)

    本文证明整流流能自动适应目标分布的内在低维结构，在端点自适应时间离散化方案下达到 O(k/ε) 的采样迭代复杂度，并建立了DDPM与随机定位之间的新颖联系，证明DDPM等价于整流流的随机化版本。

    

    近年来，整流流凭借其生成效率和最先进的性能而广受欢迎。在本文中，我们研究了整流流如何自动适应目标分布支撑集的内在低维性，从而加速采样过程。我们证明，使用精心设计的端点自适应时间离散化方案以及足够精确的漂移估计，整流流采样器可以获得 O(k/ε) 量级的迭代复杂度（在对数因子范围内），其中 ε 是总变差距离下的精度，k 是目标分布的内在维度。此外，我们通过在整流流与随机定位过程之间建立一种新颖的联系，证明了去噪扩散概率模型（DDPM）过程等价于整流流的随机化版本。基于这一联系，我们进一步设计了一种随……

    arXiv:2601.15500v5 Announce Type: replace-cross  Abstract: In recent years, Rectified flow (RF) has gained considerable popularity largely due to its generation efficiency and state-of-the-art performance. In this paper, we investigate how well RF automatically adapts to the intrinsic low dimensionality of the support of the target distribution to accelerate sampling. We show that, using a carefully designed end-point adapted time-discretization scheme and with sufficiently accurate drift estimates, the RF sampler enjoys an iteration complexity of order $O(k/\varepsilon)$ (up to log factors), where $\varepsilon$ is the precision in total variation distance and $k$ is the intrinsic dimension of the target distribution. In addition, we show that the denoising diffusion probabilistic model (DDPM) procedure is equivalent to a stochastic version of RF by establishing a novel connection between these processes and stochastic localization. Building on this connection, we further design a stoc
    
[^52]: 基于集体变量水平集的条件概率分布生成式建模

    Generative modeling of conditional probability distributions on the level-sets of collective variables

    [https://arxiv.org/abs/2512.17374](https://arxiv.org/abs/2512.17374)

    本文研究了如何对由集体变量水平集所定义的条件概率分布进行生成式建模。

    

    给定一个由数据表示的 $\mathbb{R}^d$ 中的概率分布 $\mu$，本文研究了在集体变量 $\xi:\mathbb{R}^d \rightarrow \mathbb{R}^k$（其中 $1 \le k$ ...）的水平集上相应条件概率分布的生成式建模。

    arXiv:2512.17374v4 Announce Type: replace  Abstract: Given a probability distribution $\mu$ in $\mathbb{R}^d$ represented by data, we study in this paper the generative modeling of the corresponding conditional probability distributions on the level-sets of a collective variable $\xi:\mathbb{R}^d \rightarrow \mathbb{R}^k$, where $1 \le k
    
[^53]: 当特征战胜噪声：一种噪声驱动的假设检验特征选择方法

    When Features Beat Noise: A Noise-Driven Hypothesis Testing Method for Feature Selection

    [https://arxiv.org/abs/2511.20851](https://arxiv.org/abs/2511.20851)

    本文提出一种噪声驱动的特征选择方法，通过将每个特征的重要性与随机噪声特征中的最大重要性进行基于非参数bootstrap的假设检验，为特征选择提供了坚实的统计理论基础和显著性评估准则。

    

    特征选择一直是机器学习和人工智能领域的一项艰巨挑战，日益复杂的高维数据集需要有原则性的策略来筛选出信息量最大的预测变量。尽管被广泛采用，许多成熟的技术（如Boruta、递归特征消除以及其他基于封装器的方法）都存在明显的局限性：有些方法计算成本高昂，而另一些方法则缺乏明确的统计驱动的停止准则，或无法评估其重要性分数的显著性。本文提出了一种新颖的特征选择方法来解决这些局限。我们的方法引入多个随机噪声特征，并将每个真实特征的重要性与这些噪声特征中的最大重要性值进行比较，同时结合基于非参数bootstrap的假设检验框架，以建立坚实的理论基础。我们建立……

    arXiv:2511.20851v4 Announce Type: replace-cross  Abstract: Feature selection has remained a daunting challenge in machine learning and artificial intelligence, where increasingly complex, high-dimensional datasets demand principled strategies for isolating the most informative predictors. Despite widespread adoption, many established techniques such as Boruta, Recursive Feature Elimination and other wrapper based methods suffer from notable limitations; some incur substantial computational cost, while others offer no definite statistical driven stopping criteria or assess the significance of their importance scores. This paper proposes a novel feature selection method that addresses these limitations. Our approach introduces multiple random noise features and evaluates each feature's importance against the maximum importance value among these noise features incorporating a non-parametric bootstrap-based hypothesis testing framework to establish a solid theoretical foundation. We establ
    
[^54]: 评估信息性删失下生存分析中异质性处理效应的稳健性

    Assessing the robustness of heterogeneous treatment effects in survival analysis under informative censoring

    [https://arxiv.org/abs/2510.13397](https://arxiv.org/abs/2510.13397)

    该论文提出一个少假设框架及新颖的模型无关元学习器SurvB-learner，通过部分识别推导条件平均处理效应的边界，从而在信息性删失偏差下评估生存分析中异质性治疗效应估计的稳健性，并识别治疗有效的患者亚群。

    

    arXiv:2510.13397v4 公告类型：替换 摘要：脱落（中途退出）在临床研究中很常见，高达一半的患者可能因副作用或其他原因提前退出研究。当脱落具有信息性时（即与生存时间相关），会引入删失偏差，从而导致处理效应估计也出现偏差。在本文中，我们提出了一个少假设框架，用于在面临删失偏差时评估生存分析中条件平均处理效应（CATE）估计的稳健性。与现有依赖强假设（如非信息性删失）来获得点估计的工作不同，我们采用部分识别方法推导出CATE的信息性边界。由此，我们的框架有助于识别出尽管存在信息性删失但治疗仍然有效的患者亚群。我们进一步提出了一种新颖的模型无关元学习器，称为SurvB-learner，用于估计这些边界，该方法可与任意机器学习模型结合使用，并且……

    arXiv:2510.13397v4 Announce Type: replace  Abstract: Dropout is common in clinical studies, with up to half of patients leaving early due to side effects or other reasons. When dropout is informative (i.e., dependent on survival time), it introduces censoring bias, because of which treatment effect estimates are also biased. In this paper, we propose an assumption-lean framework to assess the robustness of conditional average treatment effect (CATE) estimates in survival analysis when facing censoring bias. Unlike existing works that rely on strong assumptions, such as non-informative censoring, to obtain point estimation, we use partial identification to derive informative bounds on the CATE. Thereby, our framework helps to identify patient subgroups where treatment is effective despite informative censoring. We further propose a novel model-agnostic meta-learner, called SurvB-learner, to estimate the bounds that can be used in combination with arbitrary machine-learning models, and t
    
[^55]: 更大并不总是意味着记忆：过参数化扩散模型的早停

    Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models

    [https://arxiv.org/abs/2505.16959](https://arxiv.org/abs/2505.16959)

    过参数化扩散模型在训练中先于记忆化发生之前逐步实现泛化，记忆化时间与数据集大小成正比，因此泛化与记忆化本质上是时间尺度的竞争，可通过早停来避免记忆化。

    

    扩散概率模型已成为现代生成式人工智能的基石，然而其背后的泛化机制仍知之甚少。事实上，如果这些模型完美地最小化其训练损失，它们将只会生成属于训练集的数据，即产生记忆化现象，这正如在过参数化机制中被实证观察到的那样。我们通过展示高度过参数化的扩散模型中，自然数据领域的泛化能力在训练过程中、在记忆化发生之前被逐步实现，来重新审视这一观点。我们的结果涵盖从图像到语言的扩散模型，系统性地支持如下经验规律：记忆化时间与数据集大小成正比，这与低噪声情形下拟合经验得分函数所需时间的核回归界相一致。因此，泛化与记忆化最好被理解为不同时间尺度之间的竞争。我们展示了这一现象……

    arXiv:2505.16959v3 Announce Type: replace  Abstract: Diffusion probabilistic models have become a cornerstone of modern generative AI, yet the mechanisms underlying their generalization remain poorly understood. In fact, if these models were perfectly minimizing their training loss, they would just generate data belonging to their training set, i.e., memorize, as empirically found in the overparameterized regime. We revisit this view by showing that, in highly overparameterized diffusion models, generalization in natural data domains is progressively achieved during training before the onset of memorization. Our results, ranging from image to language diffusion models, systematically support the empirical law that memorization time is proportional to the dataset size, consistent with a kernel-regression bound on the time required to fit the empirical score at low noise. Generalization vs. memorization is then best understood as a competition between time scales. We show that this pheno
    
[^56]: 近似留一交叉验证与平均场推断的联合分析

    Simultaneous analysis of approximate leave-one-out cross-validation and mean-field inference

    [https://arxiv.org/abs/2501.02624](https://arxiv.org/abs/2501.02624)

    本文通过将ALO-CV与平均场推断进行联合分析，在高斯协变量与强凸性条件下首次证明了近似留一交叉验证对于不可微正则化子也能以可忽略的误差一致地逼近真实的留一泛化误差。

    

    近似留一交叉验证（ALO-CV）是一种被提出用于估计正则化估计量泛化误差的方法，适用于维度与样本量同阶的高维情形，即所谓的“成比例情形”。本文发展了一种新的分析方法，在高斯协变量与强凸性条件下，推导出了ALO-CV在不可微正则化子情形下的一致性。通过条件化论证，本文证明了ALO-CV权重与平均场推断中对应权重之间的差异很小。再结合平均场推断估计量与留一量之间的上界，本文证明了ALO-CV能够在误差项可忽略的范围内逼近真实的留一量。文中明确处理了带平方损失的线性模型、稳健线性回归以及单指标模型。

    arXiv:2501.02624v3 Announce Type: replace-cross  Abstract: Approximate Leave-One-Out Cross-Validation (ALO-CV) is a method that has been proposed to estimate the generalization error of a regularized estimator in the high-dimensional regime where dimension and sample size are of the same order, the so-called ``proportional regime''. A new analysis is developed to derive the consistency of ALO-CV for non-differentiable regularizers under Gaussian covariates and strong convexity. Using a conditioning argument, the difference between the ALO-CV weights and their counterparts in mean-field inference is shown to be small. Combined with upper bounds between the mean-field inference estimate and the leave-one-out quantity, this provides a proof that ALO-CV approximates the leave-one-out quantity up to negligible error terms. Linear models with square loss, robust linear regression and single-index models are explicitly treated.
    
[^57]: 针对删失数据学习稳健的治疗规则

    Learning Robust Treatment Rules for Censored Data

    [https://arxiv.org/abs/2408.09155](https://arxiv.org/abs/2408.09155)

    该论文提出了两个稳健准则及基于采样的差凸算法，用于从删失生存数据中学习最优治疗规则，分别最大化截断平均生存时间和缓冲生存概率。

    

    目前，通过直接最大化期望结果来估计最优治疗规则的研究文献正在快速增长。在生物医学研究和运营应用中，经常观察到删失的生存结果，在这种情况下，截断平均生存时间和生存概率受到极大关注。在本文中，我们提出了两个稳健的准则，用于在删失生存结果下学习最优治疗规则：前一个准则旨在找到使截断平均生存时间最大化的最优治疗规则，其中截断点由给定的分位数（如中位数）指定；后一个准则旨在找到使缓冲生存概率最大化的最优治疗规则，其中预设阈值经过调整以考虑截断平均生存时间。我们开发了一种基于采样的差凸算法来学习所提出的最优治疗规则，并为它们提供了理论依据。

    arXiv:2408.09155v3 Announce Type: replace-cross  Abstract: There is a fast-growing literature on estimating optimal treatment rules directly by maximizing the expected outcome. In biomedical studies and operations applications, censored survival outcomes are frequently observed, in which case the truncated mean survival time and survival probability are of great interest. In this paper, we propose two robust criteria for learning optimal treatment rules with censored survival outcomes; the former one targets an optimal treatment rule maximizing the truncated mean survival time, where the cutoff is specified by a given quantile such as the median; the latter one targets an optimal treatment rule maximizing buffered survival probabilities, where the predetermined threshold is adjusted to account for the truncated mean survival time. We develop a sampling-based difference-of-convex algorithm for learning the proposed optimal treatment rules, and provide theoretical justifications for them
    
[^58]: 对于选择最有可能从治疗中获益的人的统计性能保证

    Statistical Performance Guarantee for Selecting Those Predicted to Benefit Most from Treatment. (arXiv:2310.07973v1 [stat.ME])

    [http://arxiv.org/abs/2310.07973](http://arxiv.org/abs/2310.07973)

    本研究针对选择最有可能从治疗中获益的人的问题，在估计治疗效果和确定截断值时面临多重测试问题，提出一种统一的置信带方法来评估这些个体的平均治疗效果。

    

    在广泛的学科领域中，许多研究人员使用机器学习算法来识别一组被称为例外反应者的个体，他们最有可能从治疗中获益。一个常见的方法包括两个步骤。首先使用机器学习算法估计条件平均治疗效果或其代理。然后确定所得治疗优先顺序分数的截断值，以选择那些最有可能从治疗中获益的人。不幸的是，这些估计的治疗优先顺序分数往往存在偏差和噪声。此外，利用相同的数据既选择截断值又估计所选个体的平均治疗效果会遇到多重测试问题。为了解决这些挑战，我们开发了一个统一的置信带来实验性地评估那些治疗优先顺序分数至少与任何给定量化值相等的个体的排序平均治疗效果（GATES）。

    Across a wide array of disciplines, many researchers use machine learning (ML) algorithms to identify a subgroup of individuals, called exceptional responders, who are likely to be helped by a treatment the most. A common approach consists of two steps. One first estimates the conditional average treatment effect or its proxy using an ML algorithm. They then determine the cutoff of the resulting treatment prioritization score to select those predicted to benefit most from the treatment. Unfortunately, these estimated treatment prioritization scores are often biased and noisy. Furthermore, utilizing the same data to both choose a cutoff value and estimate the average treatment effect among the selected individuals suffer from a multiple testing problem. To address these challenges, we develop a uniform confidence band for experimentally evaluating the sorted average treatment effect (GATES) among the individuals whose treatment prioritization score is at least as high as any given quant
    

