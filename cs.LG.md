# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Conditional Generative Learning: Model and Error Analysis](https://rss.arxiv.org/abs/2402.01460) | 提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。 |
| [^2] | [Behavioral Couplings: From Formation to Behavioral Emergence in Language Models](https://arxiv.org/abs/2610.23847) | 该论文提出“行为耦合”概念，发现当模型在同时描述两种行为的文档上训练后，针对其中一种行为进行微调会导致另一种行为也随之涌现，揭示了微调可引发超出预期目标之外的行为。 |
| [^3] | [Task-aligned spectral inductive biases in parameterized quantum circuits](https://arxiv.org/abs/2610.23834) | 该论文的核心贡献是揭示了数据重上传量子电路中编码固定谱查询可以通过调制现有信号频率的组合方式改变假设类而无需引入新的样本信息，从而为量子机器学习提供任务对齐的谱归纳偏置，并在合成孔径雷达启发的分类任务中验证了这一机制。 |
| [^4] | [Beyond the Answer: Student-Conditioned Trace Construction for Reasoning Distillation](https://arxiv.org/abs/2610.23831) | 该论文提出ReDiST框架，以学生自身尝试为起点构建、修订并筛选与学生能力匹配的推理轨迹用于蒸馏，从而实现超越仅答案监督的学习效果。 |
| [^5] | [Waiting Buys Exactly Two Moments: Endogenous Identification in Prediction-Driven Allocation](https://arxiv.org/abs/2610.23830) | 该论文证明了在预测驱动的资源分配中，等待观测τ个周期能恰好识别2τ个潜在风险矩（每多等一期新增两个矩），但这种内生识别无法避免时机遗憾，且未必能判定唯一的最优干预日期。 |
| [^6] | [Quality-of-Transmission Estimation in Quantum Networks: Compute, Simulate, or Learn?](https://arxiv.org/abs/2610.23827) | 本文受光网络QoT估计经验启发，提出了一个量子网络传输质量估计的“计算/仿真/学习”决策框架，用于指导量子控制器在何种情况下应解析计算、仿真或通过学习获得端到端纠缠保真度。 |
| [^7] | [ENFR: Evidence Networks with Flow-Matched References for Amortized Simulation-Based Model Inference](https://arxiv.org/abs/2610.23825) | 本文提出ENFR方法，通过结合证据网络与流匹配参考技术，实现对贝叶斯证据的准确、快速且摊销式的估计，从而突破仅衡量证据比值的贝叶斯因子方法的局限，为高维拟合优度检验和模型误设检验提供直接解决方案。 |
| [^8] | [Learning without relearning physical symmetry: an SO(3)-equivariant convolutional network (CNN) on the sphere](https://arxiv.org/abs/2610.23821) | 本文提出了一种专为HEALPix球面像素化设计的SO(3)等变卷积神经网络，利用可操纵滤波器、平行移动和自旋加权特征实现可证明的精确等变性，使网络无需消耗训练数据来学习物理对称性。 |
| [^9] | [CocoRNA: Collective RNA Design with Cooperative Multi-agent Reinforcement Learning](https://arxiv.org/abs/2610.23820) | CocoRNA是一个协作多智能体强化学习框架，通过将RNA逆向设计分解为多个子问题并由多个智能体协作求解，在成功率和设计效率上显著超越现有最先进方法，且无需额外训练即可泛化到未见结构。 |
| [^10] | [Learning Personalized Memory Gating from Delayed One-Sided Feedback](https://arxiv.org/abs/2610.23812) | 该论文将事实级别的个性化记忆门控形式化为带延迟单边反馈的上下文苹果品尝问题，提出结合离线学习与在线自适应的Warm-Start UCB算法，并给出了高概率遗憾上界的理论保证。 |
| [^11] | [Explainable Machine Learning Analysis of Temperature Effects on Mechanical Strength of Fiber-Reinforced High-Performance Concrete](https://arxiv.org/abs/2610.23811) | 本研究基于1,204条实验记录构建了可解释的机器学习框架，利用梯度提升与极端梯度提升模型（R²达0.903）准确预测温度作用下纤维增强高性能混凝土的抗压强度，并结合SHAP方法揭示各因素对强度的影响机制。 |
| [^12] | [Task-aware Kernel Stein certificates for R\'{e}nyi differential privacy](https://arxiv.org/abs/2610.23798) | 本文提出基于护航分布的核Stein差异方法，通过设计聚焦相关隐私方向的核函数，为Rényi差分隐私提供可计算的隐私证书及面向任务的隐私保证，并给出了高斯与拉普拉斯机制的精确恒等式和指数机制的界。 |
| [^13] | [Parametric Resonance in Primal-Dual Learning for Slowly Non-Stationary Constrained MDPs](https://arxiv.org/abs/2610.23788) | 本文揭示了缓慢非平稳约束马尔可夫决策过程中原始-对偶学习的不稳定性并非源于非平稳性本身，而是源于约束估计与测量失配所引发的参数共振，并定量给出了可容忍失配随强迫带宽的平方根增长、随对偶-原始步长比的平方根缩减的规律。 |
| [^14] | [Sampling the XY model with an attention-based neural network](https://arxiv.org/abs/2610.23783) | 该论文提出一种结合物理信息（局域性注意力机制和近似概率）的基于Transformer的变分自回归网络，可高效采样连续自由度的格点系统，在BKT相变附近的二维XY模型上显著优于现有神经采样器，并成功重现传统MCMC方法所得的物理可观测量。 |
| [^15] | [Mesoscopic learning dynamics of extensive-width neural networks](https://arxiv.org/abs/2610.23779) | 该论文通过刻画重叠量的分布，提出了一种介观描述方法，将广宽度两层神经网络学习任意非线性目标函数的动力学约化为少数集体变量的演化，从而克服了传统方法需要追踪发散数量序参量的难题。 |
| [^16] | [Logic-Guided Graph Denoising Diffusion Models](https://arxiv.org/abs/2610.23766) | 该论文提出了一种基于一阶逻辑（FOL）的分类器引导方法，通过将逻辑规则松弛为模糊可微的目标函数来引导扩散模型的去噪轨迹，从而使生成的图数据满足严格的结构约束。 |
| [^17] | [Universal Semantic Enhancement for Weakly Supervised Anomaly Detection](https://arxiv.org/abs/2610.23765) | 该论文提出了一种基于通用语义增强的表格数据弱监督异常检测方法，通过将预训练表格基础模型的通用知识与目标数据集的领域特定知识相融合，有效解决了现有方法特征描述不完整且易过拟合稀缺标签的问题。 |
| [^18] | [From Flow-Box Invariants to Global Conservation Laws: Machine Learning-Assisted Detection of Singularities and Near-Integrability](https://arxiv.org/abs/2610.23760) | 该论文提出“扩张域神经去缩”框架，通过在嵌套扩张的相空间域上追踪独立守恒律的数量，实现了从流盒定理的局部不变量到全局守恒律的自动检测，并能识别奇点与近可积性。 |
| [^19] | [LagNet: Lag-Structured Operators for Long-Term Time Series Forecasting](https://arxiv.org/abs/2610.23735) | 该论文通过直接考察预测算子发现其敏感度矩阵存在对应相对时间滞后的对角模式，据此提出基于 Toeplitz 矩阵的 LagLayer 模块，可作为现有时间序列预测模型中时间线性层的有效替代。 |
| [^20] | [Bridging the Language Gap in Scholarly Data II: Enhancing Gender Inference for Chinese Names](https://arxiv.org/abs/2610.23734) | 针对拼音中文姓名性别推断固有的模糊性问题，该研究开发了由11位中文母语者评分数据训练的汉字回归与分类性别推断模型，以提升科学计量学研究中对中文学者姓名的性别推断准确性。 |
| [^21] | [Escaping Parameter Space: Tight Generalization Bounds via Representation Quality](https://arxiv.org/abs/2610.23729) | 该论文提出了一种基于最后隐藏层表征质量的紧致泛化界，通过无参数k-NN分类器评估表征质量，摆脱了传统参数空间复杂度度量，在CIFAR-10、CIFAR-100和ImageNet上实现了迄今最紧致的泛化认证。 |
| [^22] | [ECG-SDI: Single-lead Electrocardiogram Enables Continuous Sleep Depth Index Profiling and Scalable Digital Biomarkers for Sleep Health](https://arxiv.org/abs/2610.23727) | 本研究开发了基于单导联心电图的ECG-SDI框架，可利用常规心电信号连续刻画睡眠深度变化并提取数字生物标志物，其识别的睡眠紊乱亚型与较差的主观睡眠质量及更高的全因死亡风险显著相关。 |
| [^23] | [Neural Expansion: A Unified Mechanism for How Deep Neural Networks Generalize](https://arxiv.org/abs/2610.23716) | 本文提出“神经扩张”这一统一几何框架，通过泛化区间量化训练样本周围预测稳定的方向区域，发现训练过程中模型射线覆盖沿雅可比奇异值较低的方向渐进扩张，从而揭示了深度神经网络泛化能力的形成机制。 |
| [^24] | [Tool Recommendation for Multi-Step Agents: Finding Every Tool a Task Needs](https://arxiv.org/abs/2610.23707) | 提出了一种已部署的“工具推荐”网关系统，通过固定的三个工具接口让智能体能够访问约8,200个工具（是典型MCP服务器暴露数量的数百倍），并在ToolRet基准上将查询所需工具排进前十的成功率比BM25提高57%。 |
| [^25] | [Blitz: Accelerating the Diffusion Modules of AlphaFold3 and Boltz-2](https://arxiv.org/abs/2610.23703) | Blitz是一个少步蒸馏框架，通过几何噪声调度和多步一致性训练，将AlphaFold 3和Boltz-2的扩散去噪步骤从200步大幅加速至8步，同时保持与教师模型相当的结构预测准确性。 |
| [^26] | [Equivalence testing and run-count design for stochastic machine-learning experiments](https://arxiv.org/abs/2610.23700) | 本文提出了一套基于两级（种子×样本）方差模型和TOST等效性检验的严格认证方案，并给出将预设容差与目标功效换算为所需运行次数的闭式公式，从而取代以往仅凭两三次运行无显著差异就宣称方法等效的不可靠做法。 |
| [^27] | [Expressivity of IQP Feature-Map Quantum Kernels via Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2610.23692) | 本研究在再生核希尔伯特空间框架下为IQP特征映射量子核推导出傅里叶表示，从理论上刻画了其所表示的函数类，并揭示了可访问频率集合由电路构造决定且受带宽超参数缩放的规律。 |
| [^28] | [HardEval: An Execution-Verified Benchmark for AI-Driven Security Hardening and Compliance Remediation](https://arxiv.org/abs/2610.23685) | 提出了HardEval——一个基于RHEL 8 STIG控制项、通过在真实系统上执行验证的方式，来评估大语言模型能否完成操作系统安全加固与合规修复任务的基准测试。 |
| [^29] | [TriRouter: Sparse-to-Dense Electrode-Space Routing for Robust EEG Motor Imagery Decoding](https://arxiv.org/abs/2610.23665) | TriRouter提出一种包含判别保持、掩码鲁棒和潜在补偿三条路径的电极空间路由框架，通过验证校准的路由机制在电极严重缺失或受损情况下显著提升EEG运动想象解码的准确率。 |
| [^30] | [An Efficient Approach to Beetle Damage Detection of Banana Using Deep Learning](https://arxiv.org/abs/2610.23661) | 本研究提出基于YOLOv8n的轻量级深度学习框架，利用孟加拉国本地市场自建数据集（2,473张图像、5,827个标注实例）实现香蕉果实定位及表面健康状况的自动分类。 |
| [^31] | [A Language-Informed Loss for Visual Generation](https://arxiv.org/abs/2610.23653) | 提出LILIE，一种通过归一化流变换器与视觉-语言模型联合训练的可逆图像编码器，可从中提取语言信息引导损失用于文生图训练，相比纯像素损失基线将gFD降低14个点。 |
| [^32] | [JAXtari: High-Throughput and Easy-to-Modify Arcade Learning Environment](https://arxiv.org/abs/2610.23650) | JAXtari 是一个开源高性能的 Atari 2600 环境重新实现，通过大规模 GPU 并行化使 PPO 智能体能在单块 A100 GPU 上不到一小时内完成 1 亿步训练，并提供以对象为中心的观测和灵活的环境修改系统。 |
| [^33] | [Scalable long-range machine learning interatomic potentials with electrical response for atomistic material modeling](https://arxiv.org/abs/2610.23643) | 该论文通过解析分离全局静电计算与学习潜电荷的严格局域导数，构建了一种兼具长程静电与电响应建模能力、同时保持空间可分解性的可扩展机器学习原子间势，实现了电场下材料的分布式高效原子模拟。 |
| [^34] | [No-Regret Language Generation](https://arxiv.org/abs/2610.23641) | 本文提出无遗憾生成框架，用基于树的错误强制维度精确刻画生成任务的极小极大复杂度，揭示所有可容许生成率均可达成，并将其与在线预测联系起来，得到由Littlestone维度决定的错误界和遗憾界。 |
| [^35] | [Deep LLM layers expose passage-level context in EEG during silent reading](https://arxiv.org/abs/2610.23638) | 该研究通过对大语言模型Llama-3.1-8B每一层表征训练EEG解码器，发现默读时词级信息在模型早期层最易解码，而段落级上下文信息随模型深度增加而增强，表明深层LLM表征与阅读中的上下文加工相关。 |
| [^36] | [Deconstructing Multi-Task Active Learning: The Paradox of Gradient Conflict and Orthogonal Decomposition](https://arxiv.org/abs/2610.23623) | 该论文揭示了多任务主动学习中的“冲突悖论”——采集函数对梯度冲突进行评分反而会降低性能——并提出从线性组合到解构的范式转变，通过正交分解框架将冲突评分与干扰处理解耦：采集函数不评分冲突，梯度冲突干扰交由训练阶段的梯度手术解决。 |
| [^37] | [GRAFT-EEG: Gauge-Aware Physical Field Transport for Cross-Dataset Motor-Imagery EEG](https://arxiv.org/abs/2610.23620) | GRAFT-EEG通过移除有效电极集合的公共参考规范并借助模板导联场传输电极掩码，实现了完全不依赖目标数据统计信息的跨数据集运动想象脑电解码，在253名被试上将平衡准确率从BIOT的52.684%提升至60.865%。 |
| [^38] | [ThetreStageAI: Multi-View Retrieval-Augmented Generation for Structured Bengali Theatre Production Planning](https://arxiv.org/abs/2610.23618) | 本文提出 ThetreStageAI，一个通过场景、走位、灯光多视图检索结合模式约束 JSON 生成来实现结构化孟加拉戏剧制作规划的检索增强框架，并配套构建了 500 条记录的人工引导合成数据集。 |
| [^39] | [The Sign Code: The Hidden Binary Nature of Deep Networks](https://arxiv.org/abs/2610.23616) | 该论文发现深度网络的分类能力几乎完全由神经元激活的符号（二值符号编码）而非幅值决定——仅凭符号编码即可达到与完整模型相当的准确率，而翻转20%的符号就会使性能降至随机水平。 |
| [^40] | [Trajectory-Triggered Sparse Collaborative Decoding](https://arxiv.org/abs/2610.23613) | 该论文提出利用草稿置信度路径的聚合轨迹形状特征（而非局部不确定性指标）来决定何时调用目标模型进行纠正的稀疏协作解码方法，将触发失败预测的AUC从接近随机的0.50-0.53提升至0.86，并揭示了触发时机上干预杠杆与证据强度之间的权衡。 |
| [^41] | [How Many, but How Sure? Class Competition as a Confidence Proxy in Multi-Class Density-Based Counting](https://arxiv.org/abs/2610.23611) | 提出一种无需训练的置信度代理方法，利用多类别密度计数中最强类别通道间的局部竞争来衡量计数可靠性，推理时衰减竞争密度可将 Hicks 数据集上的总体 MAE 从 0.920 降至 0.846。 |
| [^42] | [When Magnitude Suffices: Reconstructing Time-Domain-Gated Microwave Signals Without Phase on a Thoracic Phantom](https://arxiv.org/abs/2610.23608) | 该研究证明，仅使用频谱分析仪的幅度测量数据结合全连接神经网络，即可在胸部体模上重构出与矢量网络分析仪结果相当的时域门控微波信号幅度，无需昂贵的相位信息测量。 |
| [^43] | [Boundary Detection for High-Dimensional Point Clouds](https://arxiv.org/abs/2610.23590) | 本文提出两种适用于高维、非均匀密度且含噪点云数据的新型边界检测方法，可同时输出边界分数和法向量估计，并在肿瘤边界检测与LLM激活几何分析中得到验证。 |
| [^44] | [BMuon: Budgeting Muon's Spectral Geometry for Structured Compressibility](https://arxiv.org/abs/2610.23571) | 提出BMuon优化器，通过为Muon的谱几何引入可解释的全局通道分配预算，使隐藏通道竞争共享预算，从而在保持优化性能的同时赋予模型训练后的结构化可压缩性。 |
| [^45] | [A general recipe for data-adaptive anytime-valid envelopes for iterative algorithms](https://arxiv.org/abs/2610.23570) | 该论文提出了一个通用方法，可为一大类迭代算法构建任意时刻有效且数据自适应的停止边界，将误差分析统一归结为对可预测加权噪声和满足时间一致集中性这一单一要求，克服了以往确定性包络过于保守、数据自适应包络只能逐例推导的局限。 |
| [^46] | [Beyond Expected Values: Risk-Sensitive Planning with Distributional Monte-Carlo Tree Search](https://arxiv.org/abs/2610.23565) | 该论文提出了CATSO和PATSO两种分布式蒙特卡洛树搜索算法，通过在Q边维护完整的经验分布（而非仅存储均值估计）并结合CVaR与乐观奖励进行动作选择，实现了针对尾部风险的风险敏感规划，并证明了n次rollout后根节点值误差的O(n^{-1/2})理论界。 |
| [^47] | [Beyond Latent Personalization: Integrating HRV-Conditioned Personalization via LLM for Interpretable Stress Detection](https://arxiv.org/abs/2610.23555) | 该论文提出一种通过LLM语义解释整合HRV特征与个人属性的个性化压力检测框架，以人类可读的方式描述个体生理反应，在TILES-2018数据集上相比最强基线实现29.9%的相对MCC提升，兼顾了检测性能与可解释性。 |
| [^48] | [Causal Discovery beyond Conditional Independence: Identifiability, Algorithms, and Statistical Inference for Directed Acyclic and Cyclic Graphs](https://arxiv.org/abs/2610.23543) | 本综述系统梳理了超越条件独立性的现代因果发现方法，提出将识别来源与统计估计相分离的统一框架，涵盖分布不对称、条件矩约束和跨环境变异等手段，并讨论了潜在混淆与循环图下的可识别性、算法与统计推断。 |
| [^49] | [Out of Support, Not Out of Reach: In-Context Learning with Unsupported Instructions and Composed Tasks](https://arxiv.org/abs/2610.23528) | 该论文建立了理论框架，证明上下文学习即使在指令或目标任务超出预训练支持范围时仍能成功，示例证据足以克服指令分布偏移并保持指数级的任务识别速率。 |
| [^50] | [Interpreting SSL Representations for Spoof Detection: a WavLM Study](https://arxiv.org/abs/2610.23526) | 本研究对WavLM在语音欺骗检测中的逐层表征进行了系统的可解释性分析，发现仅选取五个关键层的组合性能优于全部十三层池化，并通过典型相关分析揭示了各声学特征与不同层之间的关联规律。 |
| [^51] | [Uncertainty-aware AI-enabled framework for predicting hydrogen dispersion in confined spaces to enhance operator situational awareness](https://arxiv.org/abs/2610.23515) | 该论文提出一个两阶段不确定性感知AI框架，通过从早期传感器读数估计氢气泄漏速率并预测氢气云团扩散（附带置信区间），为船舶受限空间加注作业中的操作员提供浓度预测图，增强态势感知并支持预防性安全决策。 |
| [^52] | [Adaptive conformal prediction on the Grassmannian](https://arxiv.org/abs/2610.23508) | 该论文提出了一个在格拉斯曼流形上构建局部自适应预测区域的保形预测框架，通过基不变的几何距离和学习到的难度估计器实现几何感知的非一致性度量，在保持有限样本覆盖保证的同时提升了不同难度下的覆盖均匀性。 |
| [^53] | [A Topological Signature of Learning under Noisy Labels](https://arxiv.org/abs/2610.23491) | 该论文利用持续同调追踪神经激活中的环状结构，发现噪声标签训练的模型在训练中后期比干净标签模型表现出更高的拓扑持续性，即使其预测与噪声标签几乎不一致，从而为噪声监督下的学习过程提供了一种基于表示几何的拓扑特征。 |
| [^54] | [Towards optimal selection schedules for online active learning](https://arxiv.org/abs/2610.23477) | 该论文在高维在线主动学习设定下证明了最优样本查询策略的本质：贪婪最小化目标增量的短视策略等价于小间隔或中间间隔选择策略，即只标注间隔落在随时间不断收缩的区间内的样本，以在标注预算或“查询代价 Δ + 错误代价”的目标下最优。 |
| [^55] | [QD-Rec: Open-Ended Evolution of Recommendations via Quality-Diversity](https://arxiv.org/abs/2610.23470) | 提出QD-Rec方法，将质量-多样性算法、上下文老虎机与基础模型相结合，为高风险决策生成多样化、高质量且附带自然语言理由的开放式推荐。 |
| [^56] | [Graphlet-based Decentralized Finance Transaction Method Classification](https://arxiv.org/abs/2610.23466) | 该论文通过提取有向graphlet网络拓扑特征对DeFi中30种最常见交易方法进行分类，并利用SHAP解释揭示各类方法特有的结构模式，证明拓扑特征可有效补充事件日志计数以提升分类性能。 |
| [^57] | [HydraLLM: Adaptive Width Slicing for Elastic Text and Vision-Language Models](https://arxiv.org/abs/2610.23455) | 提出HydraLLM弹性架构，通过激活注意力头数量动态调整模型各层宽度，使单一参数集能够适配从数据中心GPU到边缘设备的多种部署场景，并通过弹性投影模块将该能力扩展到视觉-语言多模态推理。 |
| [^58] | [Predictable Scaling: Efficient Model Exploration via Scaling-Law Prediction](https://arxiv.org/abs/2610.23453) | 该论文提出“可预测扩展”（PS）框架，通过沿资源轴构建小规模代理模型并结合曲线拟合与不确定性校准，从代理实验预测全规模推荐模型性能，从而将原本需数千GPU小时的模型改进假设评估变得高效可行。 |
| [^59] | [Regularized free deconvolution for spectral repair of resampling distortion in high-dimensional particle filters](https://arxiv.org/abs/2610.23448) | 该论文提出一种正则化自由反卷积估计器，仅凭重要性权重和经验谱即可恢复高维粒子滤波器中因重采样而失真的协方差谱，并将其作为预条件子用于 Metropolis 修正的复活移动，在不引入偏差的情况下提升滤波器的混合效率。 |
| [^60] | [RECREG: Recovery-based Shielding for Regular Safety Properties in Reinforcement Learning](https://arxiv.org/abs/2610.23445) | RecReg是一个通过基于恢复的屏蔽机制强制执行非马尔可夫正则安全性质的安全强化学习框架，并首次提供了严格的安全性和收敛性理论保证。 |
| [^61] | [Generative Policy Improvement from Observational Data](https://arxiv.org/abs/2610.23436) | 本文提出 PolicyLift，一个三阶段框架，通过学习捕捉高层策略的离散动作码，在动作重叠有限的条件下，从观测数据中实现对文本、图像等结构化动作的生成式策略改进。 |
| [^62] | [Bridging Compressed Communication and Local Steps via Spectral Method in Federated Learning](https://arxiv.org/abs/2610.23434) | 该论文提出了一种同时支持本地更新与通信压缩的新型谱联邦学习方法，并在无需有界异构性这一限制性假设的温和条件下，给出了严格的收敛性保证。 |
| [^63] | [Residual Stochastic Concept Bottleneck Models](https://arxiv.org/abs/2610.23421) | 提出残差随机概念瓶颈模型，通过多元高斯分布联合建模概念与残差变量并学习其协方差依赖结构，在缓解概念不完备问题的同时避免无约束残差路径削弱概念干预的有效性。 |
| [^64] | [Flash RNN-T: Memory-Efficient Joint Network and Transducer Loss Computation](https://arxiv.org/abs/2610.23409) | 提出 Flash RNN-T，通过专用内核与分块重计算实现内存高效的 RNN-T 训练，在 H100 上相比最快基线吞吐量提升 24-39%、内存减少 26-56%，分块模式还可将内存占用进一步降低至九分之一。 |
| [^65] | [Information Estimation with Discrete Diffusion](https://arxiv.org/abs/2610.23404) | 提出了基于离散扩散的InfoSEDD方法，将信息论估计与生成建模相结合，为离散数据的互信息等信息度量估计提供了一种轻量且可扩展的新途径。 |
| [^66] | [Contracting with an Agent under Model Mismatch](https://arxiv.org/abs/2610.23402) | 该论文在委托-代理马尔可夫决策过程（PAMDP）中刻画了代理人因模型失配而持有错误信念时的合同设计问题，并提出了一个框架，使掌握真实环境动态的委托人能够通过合同设计引导代理人产生理想的长期行为。 |
| [^67] | [Forecasting Voltage Time Series under Anomalies Using a WGAN Model With Lipschitz Control and Progressive Kernel Critic Strategy](https://arxiv.org/abs/2610.23394) | 提出了一种结合渐进Lipschitz惩罚与自适应核的Wasserstein GAN模型（WGAN-CLP），用于在异常情况下对高维流电压时间序列进行递归预测和动态建模。 |
| [^68] | [Sentence-Level Topic Modeling with Lexical and Contextual Evidence](https://arxiv.org/abs/2610.23388) | SentenceETM提出了一种句子级神经主题模型，通过联合利用词汇分布与句子嵌入上下文这两种证据来源，有效解决了主题建模中词级模型缺乏上下文、文档级模型忽视多主题共存问题的粒度困境。 |
| [^69] | [Data-Driven Prediction of Optimal MPI Rank Placement and Node Count for Memory-Bound CFD on AMD EPYC Clusters](https://arxiv.org/abs/2610.23383) | 该论文基于560个实测配置的数据驱动模型，预测AMD EPYC集群上内存受限CFD的最优MPI进程放置与节点数，发现小网格在单节点上最优、大网格因超线性加速使最大节点数成为最优选择，仅中等网格存在性能与成本的真实权衡（全速运行需多花费20%至97%的节点小时）。 |
| [^70] | [Beyond Success Rate: Interventional Evaluation of Memory in Vision-Language-Action Systems](https://arxiv.org/abs/2610.23372) | 该论文提出干预性记忆评估协议（IMEP），通过目标事实替换等干预操作并将决策正确性与执行成功率分开评估，以严格检验记忆在视觉-语言-动作系统中是否真正发挥作用。 |
| [^71] | [RTSM: Adapting Recursive Models to Multivariate Time-Series Classification](https://arxiv.org/abs/2610.23353) | RTSM通过结合可学习的卷积时间分词器、单状态权重共享精炼和池化分类读出，在30个UEA多元时间序列分类数据集上取得72.81%的平均准确率，在17个深度学习模型中排名第一。 |
| [^72] | [How Finite-Rank Bottlenecks Shape the Low-Rank Adaptation Landscape](https://arxiv.org/abs/2610.23343) | 本文提出了一种无需线性化的新理论框架，证明冻结主干后损失仅通过有限秩的“瓶颈”映射感知适配矩阵，因而迹范数正则化下的全局最优解秩由瓶颈秩而非样本量决定，从而解释了LoRA低秩更新为何足够有效。 |
| [^73] | [SETAREH Transform: A Sorting-Assisted Haar-Wavelet Sparsification Method for Approximate Inner-Product Computation](https://arxiv.org/abs/2610.23338) | 本文提出SETAREH变换，一种利用排序辅助实现能量集中的Haar小波稀疏化方法，可高效近似固定高维系数向量与连续输入向量之间的内积，从而显著降低高维向量运算中的存储、内存流量与通信开销。 |
| [^74] | [Minimax-optimal Scaling Laws in Ridge Regression via Transfer Learning](https://arxiv.org/abs/2610.23337) | 本文在高维岭回归迁移学习框架下建立了风险确定性等价，并证明在共享回归函数和可交换幂律协方差谱的条件下可实现极小极大最优标度律，且最优源正则化仅依赖源数据、具有跨任务普适性。 |
| [^75] | [HiPredict: Predicting the missing HiRISE RED4 spectral channel with machine learning](https://arxiv.org/abs/2610.23335) | 本文提出HiPredict方法，利用机器学习捕捉颜色通道间的非线性关系，从而更准确地重建HiRISE仪器因RED4探测器故障而缺失的光谱通道数据。 |
| [^76] | [Peak-Memory Schedules You Can Trust: Exact Realization, Certified Optimality, and the Limits of Recomputation for Inference](https://arxiv.org/abs/2610.23328) | 该论文提出通过剖析并重排生成程序自身的语句来精确实现峰值内存调度，证明只要内存分配不依赖历史即可逐字节精确达到建模峰值且数值结果逐位一致，从而解决了将优化算子顺序交回编译器重新生成代码的不可靠问题。 |
| [^77] | [The Loss or the Architecture? What Matters in Survival Analysis with Competing Risks](https://arxiv.org/abs/2610.23322) | 该论文将损失函数与网络架构解耦并独立评估，在20个表格和3个病理图像数据集上发现：在删失加权下直接建模累积发生率的损失函数比选择更好的骨干网络带来更大的收益。 |
| [^78] | [Untrained CNN and coefficient-space regularization for joint reconstruction in photoacoustic tomography](https://arxiv.org/abs/2610.23315) | 该论文提出了一种基于未训练CNN固定非线性特征与系数空间正则化的数据驱动框架，实现了光声层析成像中初始压力与空间变化声速的联合重建，具有强泛化能力和噪声鲁棒性。 |
| [^79] | [Look, Reason, Draw a Region: Recursive Refinement for Spatial Context Prediction](https://arxiv.org/abs/2610.23302) | Spatial-TRM是一个仅有1120万可训练参数的紧凑递归推理模型，通过冻结视觉-语言特征与递归细化机制，在RoboSpatial-Home基准上将空间上下文预测准确率大幅提升9.02个百分点。 |
| [^80] | [How the elastic axis folds the transonic flutter boundary: shock energy transfer and conformally calibrated active learning](https://arxiv.org/abs/2610.23298) | 本文通过批量主动学习训练的高斯过程代理模型结合分裂共形预测校准，仅用80次非定常Euler模拟便绘制出Isogai A工况翼型折叠的跨声速颤振边界，揭示了弹性轴位置与激波能量传递对边界折叠的作用机制，并发现时域阻尼标签的隐性失效模式及初始迎角残余对颤振速度的显著影响。 |
| [^81] | [Learning angles for quantum approximate optimization](https://arxiv.org/abs/2610.23296) | 提出通过学习模型建立从问题实例到高质量QAOA变分角度参数的映射，用单次前向传播取代计算昂贵的逐实例非凸优化，速度提升数个数量级且质量几乎不变，并能泛化到未见过的图拓扑结构。 |
| [^82] | [The Optimal Ranking Tradeoffs Between Two Ranking Scores in Two-Class Classification](https://arxiv.org/abs/2610.23294) | 该论文揭示了常用的二分类评估指标都是同一参数化排序得分族的特例，并据此建立了两类分类中任意两个排序得分之间的最优权衡排序方法。 |
| [^83] | [Distributed Performance Boosting through Neural Control for Plug-and-Play Networks](https://arxiv.org/abs/2610.23292) | 本文提出一种基于内模控制原理的分布式神经控制方法，通过局部神经算子的自由参数化互连，在保证即插即用网络系统闭环稳定性的同时提升控制性能。 |
| [^84] | [Data-Efficient Classifier Pretraining Through Diffusion Models](https://arxiv.org/abs/2610.23289) | 提出DiffPT扩散建模预训练方法，可在领域训练数据有限时有效预训练分类器骨干网络，并在医学影像和时间序列任务中超越自监督基线方法。 |
| [^85] | [DeDiCa: A Scalable Deconfounding Discrete Causal Generative Model](https://arxiv.org/abs/2610.23272) | DeDiCa通过张量化参数化的因果贝叶斯网络，实现了对含隐藏混杂的离散变量高效且可扩展的因果推断，其精度媲美最先进的神经因果生成模型，同时推理速度和内存开销大幅降低。 |
| [^86] | [NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models](https://arxiv.org/abs/2610.23271) | NestedVLA提出用嵌套超网络生成任务自适应的动作专家参数来事后整合独立学习的技能，从而在多个机器人操作基准上实现灵活的任务条件化技能复用并有效缓解任务间干扰。 |
| [^87] | [Beyond Marginal Influence: Interaction-Aware Memory Selection for LLM Agents](https://arxiv.org/abs/2610.23266) | 该论文提出JointCoreBench基准和JointCore方法，突破传统基于边际影响的记忆排序方式，通过感知记录间交互并采用成本排序的组删除策略，实现LLM智能体最小成本充分记忆子集的选择，达到100%恢复率并将子集评估次数减少62.9%。 |
| [^88] | [ADT: All-Atom Diffusion Tokenizer for 3D Molecular Structures](https://arxiv.org/abs/2610.23265) | ADT通过将分词粒度与重建分辨率解耦，并借助扩散模型的生成能力恢复量化带来的几何畸变，实现了兼顾紧凑表示与全原子精确保建的三维分子结构分词方法。 |
| [^89] | [Reasoning Models Can Control Their Chains of Thought When They Tell Themselves To](https://arxiv.org/abs/2610.23257) | 推理模型并非天生无法控制其思维链，将约束作为自我指令置于CoT内部时，模型的可控性显著优于在用户提示中指定约束的方式。 |
| [^90] | [Sample, Mix, or Verify? Allocating Test-Time Compute under Imperfect Verifiers](https://arxiv.org/abs/2610.23255) | 该论文提出 SPLIT 方法，利用小型带标签试点实验和精确概率公式，在验证器不完美的情况下科学地分配测试时计算预算（采样数量、模型选择与验证开销）。 |
| [^91] | [FLARE: Physics-Grounded Evaluation of Language-Model Agents for Feeder Restoration with Degraded Observations](https://arxiv.org/abs/2610.23251) | FLARE是一个基于物理约束的馈线恢复评估基准，通过五种退化观测设置下800个交流可行案例、共78,750个回合的大规模评估，首次系统揭示了语言模型智能体在配电网恢复任务中虽中位表现优异但存在显著失败长尾的风险，并验证了物理安全保护机制的有效性。 |
| [^92] | [Can the Parts Fool the Test? Counterfactual Pairing Cycles for Relational OOD](https://arxiv.org/abs/2610.23248) | 提出“反事实配对循环”评估基元：在比较两侧复用相同的两张图像与两个描述、仅改变配对方式，使仅图像、仅文本及可加性捷径在有限样本中精确抵消，从而干净地评估关系型分布外误差，并兼作关系型离群暴露的训练损失与双线性视觉-语言评分的诊断工具。 |
| [^93] | [Think More or Read More? A Token-Budget Theory of Reasoning with Shared Context](https://arxiv.org/abs/2610.23246) | 提出token预算理论形式化固定上下文窗口下“阅读他人输出”与“自主推理”的权衡，通过允许错误相关的Condorcet陪审团模型推导出多智能体辩论最优智能体数量的闭式解，且仅需单智能体测量即可估计最优群体规模。 |
| [^94] | [Anisotropic Representation Dynamics in Transformers](https://arxiv.org/abs/2610.23242) | 该论文提出了混合门控的各向异性表示动力学理论，将自注意力分解为数值传递、查询门控与键门控，揭示了门控机制可诱导表示差异从收缩切换为扩张的状态转换，并给出了判定该现象的充分必要谱判据。 |
| [^95] | [Response-Informed Value Estimation for Cost-Aware Multi-Turn LLM Routing](https://arxiv.org/abs/2610.23239) | 提出RIVET方法，在训练阶段利用每次调用后的实际响应信息为每一轮提供价值监督信号，而在部署时仅依赖调用前信息进行成本感知的多轮LLM模型路由。 |
| [^96] | [Characterising and Analysing Retrieval-Augmented Generation Techniques](https://arxiv.org/abs/2610.23226) | 提出了一个用于刻画和分析现有检索增强生成（RAG）技术的概念框架，可作为实证评估前的筛选机制，帮助研究者和实践者为特定问题场景选择最合适的RAG技术。 |
| [^97] | [Process Reward Modeling with Reliability- and Conservatism-Aware Bayesian Belief](https://arxiv.org/abs/2610.23223) | 提出BayesianPRM，将过程奖励视为随机变量并维护贝叶斯信念，首次同时建模PRM不确定性与策略不确定性两类来源，从而缓解测试时扩展中系统性奖励高估的问题。 |
| [^98] | [Non-Parametric Optimization for Scalable Learning in Stochastic Decision Problems](https://arxiv.org/abs/2610.23204) | 本文基于Malliavin微积分开发了随机变分框架，提出了对参数化维度不敏感的可扩展深度学习算法——随机路径跟随器（SPF），有效求解时变随机优化问题。 |
| [^99] | [Proportionality in Ranking Compression](https://arxiv.org/abs/2610.23193) | 本文为排名压缩问题提出了三种比例性概念，证明了位置比例性和PSC总能被满足且可在多项式时间内求解，而成对比例性一般情况下无法满足，但在单峰或单交叉排名下可以实现。 |
| [^100] | [LangNCO: Compiling Natural-Language Constraints into Inference-Time Interventions for Neural Combinatorial Optimization](https://arxiv.org/abs/2610.23190) | LangNCO利用预训练大语言模型将自然语言运营需求编译为DSL规则，并通过硬动作掩码与关键偏移解码等推理时干预手段，在不改动冻结NCO求解器的架构、表示与参数的情况下满足部署后出现的新约束。 |
| [^101] | [A Stronger Teacher Within: Strengthening MoE Models via On-Policy Bidirectional Distillation](https://arxiv.org/abs/2610.23185) | 提出MoEvolve框架，通过增加MoE模型每token激活预算来挖掘模型内部潜在更强教师，并采用“先恢复后提升”的策略内双向蒸馏策略强化MoE大语言模型。 |
| [^102] | [Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning](https://arxiv.org/abs/2610.23181) | 本文提出TreeOPSD，一种无需评论家的树感知后验自蒸馏框架，通过将树结构回放批次转化为有限的后验自蒸馏目标，同时解决长时程智能体强化学习中的稀疏信用分配与策略陈旧问题。 |
| [^103] | [Structural Affinity: An Algorithm for Coherent Cage-Based Shape Design](https://arxiv.org/abs/2610.23173) | 本文提出结构亲和性方法，通过双笼结构和基于均值坐标预计算的亲和矩阵直接耦合笼体-手柄运动，作为显式正则化手段实现基于笼体变形的平滑连贯形状设计。 |
| [^104] | [Explaining Chess and Go in Natural Language](https://arxiv.org/abs/2610.23166) | 本文提出了基于引擎评估的解释性指标和无需训练的BinProp方法，使语言模型能够发现可复用的棋类概念并以自然语言解释着法，平均得分相比直接提示提高近两倍，且该方法可泛化至围棋、跳棋和将棋等多种棋类。 |
| [^105] | [Nearly Linear Private Learning of Littlestone Classes](https://arxiv.org/abs/2610.23160) | 该论文将Littlestone维数为 $d$ 的类在差分隐私PAC学习中的样本复杂度上界从此前最好的 $\widetilde O(d^5)$ 大幅改进为近乎线性的 $O(d\log d)$，其核心技术贡献是一种针对数据相关候选集、配备满足稳定性条件的秩的结构化指数机制。 |
| [^106] | [CIIR - Class-Indexed Interface Representation Learning](https://arxiv.org/abs/2610.23151) | CIIR提出了一种模块化编码器-解码器框架，利用类索引地址空间在无需配对样本的情况下连接异构数据集，并在冻结基础地址空间后仅通过轻量级适配器即可扩展至新数据集。 |
| [^107] | [Grounded Credit Assignment in Reinforcement Learning for Multi-Turn Search Agents](https://arxiv.org/abs/2610.23148) | 提出接地信用分配（GCA）方法，利用可验证的子结果（证据检索与答案正确性）分别校正搜索与答案的信用，解决GRPO等结果奖励强化学习在训练多轮搜索智能体时导致答案缺乏证据支撑的问题。 |
| [^108] | [Local proxies for memory-efficient training](https://arxiv.org/abs/2610.23145) | 提出为每个Transformer块训练局部代理网络，通过蒸馏预测全局输出使梯度可局部独立传播，在大幅降低训练内存占用的同时保持接近标准反向传播的梯度估计精度。 |
| [^109] | [MetaCurate: Evolving Curation Filters for High-Quality Agentic Coding Data](https://arxiv.org/abs/2610.23134) | MetaCurate 提出了一种无需训练的框架，通过大语言模型成对判断产生的反馈自动演化可执行的轨迹筛选过滤器，从而高效地为智能体编程训练挑选高质量数据样本。 |
| [^110] | [Information-Geometric Blind Spots of Conformal Test Martingales](https://arxiv.org/abs/2610.23131) | 该论文揭示了共形检验鞅的信息几何盲区：凡保持符合性函数前推分布不变的分布偏移均无法被检测，并借助 A-等价类与 A-隐秘 I-投影给出该盲区的显式刻画与勾股分解，同时证明当符合性函数取 Neyman-Pearson 似然比时盲区消失。 |
| [^111] | [LED: Improving Masked Diffusion Language Models by Lookahead Editing](https://arxiv.org/abs/2610.23122) | 提出前瞻编辑方法（LED），通过在掩码扩散解码中引入可重写的提议状态，使掩码扩散语言模型在不牺牲其易优化性的前提下获得迭代修正的表达能力。 |
| [^112] | [Domain-specific language alignment of multimodal embeddings towards explainable predictions](https://arxiv.org/abs/2610.23110) | 该论文提出一个无需带描述训练数据的语言对齐框架，可将任何关联了可解释辅助数据的嵌入空间与语言对齐，从而实现多模态嵌入的可解释预测。 |
| [^113] | [Towards a GraphNARX model for structural dynamics](https://arxiv.org/abs/2610.23103) | GraphNARX是一种基于图的结构动力学非线性自回归外生模型，能够考虑结构几何特性进行系统辨识，可泛化至不同维度和自由度的结构，并支持虚拟传感功能。 |
| [^114] | [Where Latents Belong and How They Flow: Joint Identification of Structural Roles](https://arxiv.org/abs/2610.23074) | 该论文形式化了动力系统中潜在因子的“结构角色”概念，证明了表达位置、接收过去影响与传递未来影响这三种角色可在共同置换下于每个时间点联合识别，并据此开发了在共享潜在坐标系中追踪结构角色的RoleTrace框架。 |
| [^115] | [Path to Prior-Data Fitted Relational Models via Learned Latent Multi-table Representations](https://arxiv.org/abs/2610.23073) | 该论文首次将先验数据拟合（PFN）范式拓展到基于消息传递的图关系模型，通过在合成关系数据库上预训练Griffin编码器，并将其潜在表示输入冻结的上下文内预测器，在目标任务零训练的情况下实现了对真实世界关系分类任务的泛化，并验证了课程学习对消息传递表示的增益。 |
| [^116] | [Unifying Embodied Foundation Models via a Model-to-Policy Formulation](https://arxiv.org/abs/2610.23053) | 本文提出一种基于随机核的、与架构无关的统一概率表述，将VLA模型与世界动作模型等具身基础模型纳入“模型到策略”的统一框架，贯通预训练、后训练与策略部署全过程。 |
| [^117] | [DIFFLET: A Fast Inference Engine for Diffusion Transformer Models on AWS Trainium](https://arxiv.org/abs/2610.23047) | Difflet 是面向 AWS Trainium 的扩散 Transformer 服务引擎，通过五个流水线共享统一编译后端、按注意力头切分的融合注意力以及基于设备端探针的逐步缓存决策，将去噪步骤加速 1.2-4.3 倍，常驻服务速度提升 1.5-3.4 倍。 |
| [^118] | [GR-OPD: Group-Relative Sequence Credit for On-Policy Distillation](https://arxiv.org/abs/2610.23042) | GR-OPD提出了一种组相对的序列级信度分配方法，通过聚合词元级教师反馈、折扣不可靠的后续反馈并利用同提示采样组校准指导，解决了在线策略蒸馏中局部信度分配的缺陷，并从理论上证明了其对反向KL目标梯度的无偏估计性质。 |
| [^119] | [Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design](https://arxiv.org/abs/2610.23036) | 该论文通过控制变量的系统消融研究，分离了循环语言模型中拓扑结构、输入注入与循环状态设计三大维度，发现“前奏-循环块-尾声”三明治架构配合输入注入能持续提升推理与上下文利用能力，并证明完整的Huginn架构在全部十个基准测试中优于Ouro。 |
| [^120] | [Distinguishability and Recovery for a Doubly Sparse Spiked Wigner Model](https://arxiv.org/abs/2610.23033) | 本文研究信号与噪声均稀疏的双稀疏尖峰Wigner模型，提出留一二次阈值化、符号同步及谱精炼等算法，在信噪比保持常数阶的条件下实现了稀疏秩一信号的支撑识别、精确恢复与强恢复，并给出了信号稀疏性、噪声稀疏性与计算成本之间的显式权衡刻画。 |
| [^121] | ["Deep lifts": Neural networks can generalize even with the dumbest optimizer](https://arxiv.org/abs/2610.23009) | 该论文证明，即使以最“愚蠢”的方式——从过大的 ReLU 网络的拟合参数中均匀随机选取参数——只要数据可由更小的网络解释，所选参数仍能以 $\tilde O(wdWD/n)$ 的速率自适应泛化，其根源是 ReLU 网络的一种几何归纳偏置：小网络参数“见证”了指数级体积的等价大网络拟合参数，使其远超泛化差的参数体积。 |
| [^122] | [A Minimal ReLU Classifier on Non-Linearly Separable Data: Convergence and Implicit Bias at and beyond the Edge of Stability](https://arxiv.org/abs/2610.23007) | 该论文在一个极简的非凸非光滑设置（双神经元ReLU网络在一维非线性可分数据上以逻辑损失训练）中，对大学习率下梯度下降的稳定性边缘行为给出了严格理论分析，证明了有限时间损失界，并刻画了GD方向上收敛至显式最大间隔向量这一隐式偏置。 |
| [^123] | [Controlling the Diversity-Quality Trade-Off of LLMs by Merging Models Before and After Alignment](https://arxiv.org/abs/2610.22999) | 提出一种无需额外训练的深度感知 FrankenMerging 方法，通过将对齐模型的低层与基础模型的高层进行合并，可灵活控制大语言模型输出中词汇与语义的多样性及质量之间的权衡。 |
| [^124] | [A Label Noise Robust Algorithm for Graph Data using Neighborhood Label Distribution](https://arxiv.org/abs/2610.22998) | 本文提出LENS框架，通过测量移除节点后邻域熵的变化来识别图数据中的噪声标签节点，并结合重标记机制，提升了图卷积网络在标签噪声下的鲁棒性。 |
| [^125] | [TALLOP: Joint Optimization of Local Training and Communication Compression for Time-Constrained Federated Learning](https://arxiv.org/abs/2610.22996) | 提出TALLOP方法，在时间受限的联邦学习中联合优化本地训练工作负载与通信压缩，以缓解异构环境下的掉队者效应并减少达到目标精度所需的实际训练时间。 |
| [^126] | [PersonaCUABench: Benchmarking Computer-Use Agents Serving Diverse Personas](https://arxiv.org/abs/2610.22990) | 该论文提出 PersonaCUABench 基准，通过让 44 种不同用户角色用自己的措辞重写固定的任务指令，仅改变“谁在提问”就使九个骨干模型的平均任务完成率下降 36%，揭示了现有计算机使用智能体难以真正服务不同用户的个性化需求。 |
| [^127] | [How Murky Before a VLM Knows It Cannot See? Accuracy, Confidence and Abstention on a Physical Visibility Axis](https://arxiv.org/abs/2610.22981) | 该研究首次在以米为单位的物理可视性轴上（基于Koschmieder雾模型和水下吸收光谱）系统测量视觉语言模型的准确率、置信度与弃权行为，发现当能见度降至十几米时模型准确率已骤降至随机水平，而其置信度却始终保持高位，揭示了VLM在“看不清”时缺乏自知之明的过度自信问题。 |
| [^128] | [Towards Hyperparameter Transfer for Differentially Private Optimization](https://arxiv.org/abs/2610.22976) | 本文提出DP-μP框架，将最大更新参数化（μP）扩展到差分隐私设置，通过信噪比条件与谱缩放公式为DP-SGD、DP-Adam和DP-Muon优化器提供了逐层学习率缩放规则，从而实现差分隐私训练中的超参数迁移。 |
| [^129] | [What Shapes Brain Alignment of Large Language Models during Dialogue? Insights from Language Production and Comprehension](https://arxiv.org/abs/2610.22961) | 本研究利用对话中的fMRI数据揭示了大型语言模型在语言产生与语言理解两种模式下呈现截然不同的大脑对齐模式，且对齐程度与模型表征的内在维度以及可解释的语言和交际特征密切相关。 |
| [^130] | [On the tradeoff between robustness and geometric accuracy in adversarial training](https://arxiv.org/abs/2610.22955) | 该论文证明了对抗分类器的几何精度损失与对抗预算ε之间的最优定量权衡关系——鲁棒分类器至多牺牲O(√ε)或O(ε)的精度，从而表明在高复杂度假设类中，鲁棒性与精度可以以精确量化的方式共存。 |
| [^131] | [Large Diffusion Models Need More Than EMA](https://arxiv.org/abs/2610.22945) | 该研究从理论和实验上证明EMA并非扩散模型训练的最优方案：恒定步长轨迹平均存在固有优化局限，而学习率衰减与EMA互补可提升性能，Schedule-Free优化则无需依赖衰减时机即可在各个训练阶段持续超越EMA。 |
| [^132] | [Aligning Language Models Towards Counterfactual Invariance](https://arxiv.org/abs/2610.22942) | 本文提出一种基于群体级奖励函数的策略梯度技术，使语言模型在对齐过程中实现反事实不变性，解决标准强化学习对齐方法在提升整体回复质量的同时放大跨人口群体反事实偏差的问题。 |
| [^133] | [Many Biases, One Circuit: Post-Training Tunes a Pretrained Cue-Routing Mechanism](https://arxiv.org/abs/2610.22937) | 多种偏见（如谄媚性）共享预训练模型中已存在的同一注意力头回路——助手微调在聊天格式中激活它，强化学习进一步加强它，表明跟随线索源于预训练与调优的副作用，而非对齐本身的问题。 |
| [^134] | [Conformal Policy Selection: When to Deploy AI Decisions for Targeted Intervention](https://arxiv.org/abs/2610.22930) | 该论文提出“共形策略选择”方法，将AI策略的针对性部署建模为多重检验问题，通过加权共形p值与两步选择-剪枝程序在控制错误发现率的前提下识别出能从AI策略中受益的个体，实现了有限样本FDR控制和双重稳健的渐近保证。 |
| [^135] | [Scalable Levenberg-Marquardt Training Framework for Deep Learning](https://arxiv.org/abs/2610.22925) | 提出了一种可扩展的Levenberg-Marquardt训练框架，克服了雅可比矩阵带来的内存和计算瓶颈，使LMA能够高效地训练通用大规模深度神经网络。 |
| [^136] | [Where's the Indirect Evidence? Investigating Grammatical Generalization with Influence Functions](https://arxiv.org/abs/2610.22919) | 本文提出一种基于影响函数的自底向上方法来识别语言模型学习语法现象所依赖的间接证据，并发现移除这些间接证据会在五个模型的15种BLiMP语法现象上显著降低模型的泛化准确率。 |
| [^137] | [FederatedBytes: Tokenizer-Free, Byte-Level Small Language Models for Efficient Multimodal Federated Learning](https://arxiv.org/abs/2610.22906) | 提出FederatedBytes——一种无分词器、将所有输入视为字节序列并采用线性注意力机制的小型语言模型，显著降低了NLP与多模态分析任务在联邦学习中的计算和通信开销。 |
| [^138] | [A Feature Selection Layer Approach for Multi-Class Classification on Neural Networks](https://arxiv.org/abs/2610.22898) | 提出了对特征选择层的多种扩展，使嵌入式特征加权机制能够在多分类任务中捕捉特定标签的特征关系，从而提升神经网络的可解释性。 |
| [^139] | [When to Forget: Risk-Controlled Context Adaptation for Long-Horizon Agents](https://arxiv.org/abs/2610.22897) | 该论文提出RECAP框架，通过匹配上下文干预、执行关系预测和风险约束下的独立策略认证，研究长程智能体在何时应保留、压缩或删除失败历史记录，发现失败记录的价值随执行进程动态变化。 |
| [^140] | [Competence-Based Evaluation of Logical Reasoning in Large Language Models](https://arxiv.org/abs/2610.22894) | 本文提出“基于能力的评估”（CBE）新范式，通过参数化生成器调节复杂度并使用保留答案的多种表述形式，来衡量大语言模型逻辑推理的泛化能力，而非仅依赖单一准确率指标。 |
| [^141] | [What Accuracy Hides In Learned Transformer Programs](https://arxiv.org/abs/2610.22891) | 该研究发现token准确率会掩盖Transformer程序中的错误完整答案，通过程序检查能够发现抽样测试遗漏的失败案例，并据此改进训练数据，从而在多个算法任务上提升序列准确率。 |
| [^142] | [Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?](https://arxiv.org/abs/2610.22887) | 提出并发约束强化（CCR），将多个可独立验证的问题放入同一个受限完成中，使RLVR在8,000-token的短窗口下仍能保留正确性信号，并将每训练步的token消耗从约97.5万降至约17.5万，大幅降低训练成本。 |
| [^143] | [Learning to Be Guided: Shaping Latent Geometry for Classifier-Free Guidance](https://arxiv.org/abs/2610.22885) | 本文提出原型正则化方法，通过在潜在空间中塑造“类间方差大于类内方差”的类别几何结构，使潜在空间天然适配无分类器引导（CFG），从而提升条件扩散模型的引导生成性能。 |
| [^144] | [ToPA: Block-wise Toeplitz Adaptation for Expressive and Efficient Fine-Tuning](https://arxiv.org/abs/2610.22883) | 本文提出ToPA方法，利用沿对角线元素恒定的分块Toeplitz矩阵对预训练模型进行参数高效微调，在不施加低秩或稀疏性显式约束的情况下，以紧凑的参数化方式获得更接近全量微调的表现力与效率。 |
| [^145] | [Benchmarking and Enhancing Agentic Causal Discovery](https://arxiv.org/abs/2610.22846) | 提出Heuresis基准，通过匿名模拟数据集评估LLM智能体的因果发现能力，区分出现有算法的收益上限与需要智能体深入检查数据才能弥补的差距，从而推动智能体式因果发现的增强。 |
| [^146] | [Partial-Revelation Panel Forecasting: Predictive Information Gain, Allocation, and Coarsening-Consistent Operators](https://arxiv.org/abs/2610.22831) | 该论文提出部分揭示面板预测框架，证明了嵌套揭示机制下预测信息增益的恒等式，并设计了具有闭式狄利克雷-多项式实现的粗化一致面板算子，在多个情感对话数据集上显著改善了冷启动计数预测。 |
| [^147] | [Causal Edit Serialization](https://arxiv.org/abs/2610.22820) | 提出因果编辑序列化（CES）框架，通过指针词元、轻量级定位头和双索引旋转位置编码，使自回归语言模型能够在保持标准因果注意力机制的前提下，对文本进行插入、删除、替换、复制和移动等显式编辑操作。 |
| [^148] | [Replay Pruning and Few-Bit Integer Inference for Hardware-Efficient Speed-Sign Recognition](https://arxiv.org/abs/2610.22811) | 提出通道重放剪枝方法，通过在原始训练网络上重放记录的剪枝删除并结合知识蒸馏恢复，在保持父网络精度的同时移除YOLO网络60%以上的参数，实现汽车SoC上硬件高效的限速标志识别。 |
| [^149] | [Reasoning Reranking with Semantic IDs as the Sole Item Representation](https://arxiv.org/abs/2610.22797) | 本文构建了ReRankID受控测试平台，首次系统评估仅以语义ID作为物品表示进行推理重排序的可行性，发现此类模型常常直接复制输入顺序而非真正重排序，且引入可读元数据也无法修复这一失败。 |
| [^150] | [Ori: Drift-Aware Streaming $k$-PCA for Very Low-Rank LLM Training](https://arxiv.org/abs/2610.22794) | 提出Ori优化器，利用受Oja++启发的低开销流式k-PCA算法高频跟踪LLM训练中缓慢漂移的梯度子空间，在严格低秩约束下达到最先进的训练精度。 |
| [^151] | [Next Token or Vector? Learning How Far to Decode](https://arxiv.org/abs/2610.22788) | 提出CADENCE上下文自适应解码方法，利用高斯预测头将未来文本片段表示为连续潜在向量，并通过学习路由器根据上下文动态选择预测长度，相比固定长度预测获得更高的草稿接受率。 |
| [^152] | [Anytime Contracts for AI Services](https://arxiv.org/abs/2610.22781) | 该论文将契约理论扩展到用户可随时退出订阅的AI服务场景，证明在此设定下预算最优的合约与基于似然比e-过程的任意时刻有效假设检验成正比，从而防止模型提供方或用户对合约的滥用。 |
| [^153] | [Reusing Calibration Gains across Model Representations](https://arxiv.org/abs/2610.22771) | 该论文提出一种无需接收方优化的迁移方法，可将一种模型表示上学到的尺度校准增益复用到同一预训练模型的其他表示（如量化模型）中，在保持量化存储布局不变的同时保留76.9%–90.5%的校准收益。 |
| [^154] | [Unit-level aggregation breaks coverage in causal foundation models](https://arxiv.org/abs/2610.22770) | 因果基础模型虽然在单元级估计上表现出色，但其可信区间在聚合到单元组（如全样本平均效应）时会严重退化，在ACIC 2016上全样本平均效应的区间仅有20%覆盖真实值。 |
| [^155] | [One Communicator, Three Vendors: Extending the MoA Predictive Theory to a Real NVIDIA+AMD+CPU MPI Ensemble](https://arxiv.org/abs/2610.22769) | 本文首次将NVIDIA A100、AMD MI100和CPU三种异构进程纳入同一个MPI通信器，提出基于1/t_i权重分配的MoA预测理论，并通过实验揭示线性假设在MI100上失效、而启动几何模型能准确预测A100的运行时间行为。 |
| [^156] | [Acquisition Function-Guided Generation for Bayesian Optimization over Discrete Spaces](https://arxiv.org/abs/2610.22763) | 该方法将预训练生成模型与数据空间代理模型相结合，通过扩散桥将采集函数倾斜分布转化为平滑的后验平均采集函数，从而引导生成模型采样候选解，实现便于融入领域知识的离散空间贝叶斯优化。 |
| [^157] | [Shape-Aware Neural Time Fields for Rigid-Body Motion and Manipulation Planning](https://arxiv.org/abs/2610.22762) | 提出形状感知神经时间场框架，通过学习可复用的行进时间表示来显式刻画刚体平移与旋转所受的不同几何约束，并结合基于采样的规划器，实现高效的刚体运动规划与非抓取操作规划。 |
| [^158] | [Refusal Is Not Robustness: Measuring Harmful Reachability in Autoregressive Language Models](https://arxiv.org/abs/2610.22754) | 该论文提出有界延续分析（BCA），将自回归生成建模为随机过程并借助精确概率模型检测，量化了模型在采样出拒答的表面下仍能到达有害内容的概率（有害可达性），弥补了仅用攻击成功率评估 LLM 安全性的不足。 |
| [^159] | [Native Geometry vs. Linear Recoverability: How Alignment Changes Platonic Scaling](https://arxiv.org/abs/2610.22749) | 本研究通过对五个模型家族的天文图像嵌入进行对比，发现监督式Ridge线性对齐能够显著增强模型规模与跨数据集表示对应性之间的关系，且该效应并非源于嵌入维度优势或简单的旋转缩放变换，表明柏拉图式表示收敛主要通过监督对齐而非原生余弦几何体现。 |
| [^160] | [Reading the News Without Hindsight: NoLBERT Topics and the Cross-Section of Stock Returns](https://arxiv.org/abs/2610.22745) | 该论文提出仅使用1976-1995年历史文本预训练的NoLBERT编码器来构建无后见之明偏差的新闻主题，并发现将注意力冲击衡量窗口延长至63个交易日可显著提升基于新闻的股票截面收益预测效果。 |
| [^161] | [OrderFlowDiT: Transport-Regularized Diffusion Transformer for Probabilistic Limit-Order Supply Forecasting](https://arxiv.org/abs/2610.22727) | 该论文提出OrderFlowDiT，一个结合非平衡最优传输辅助损失的条件扩散Transformer，能够对未归一化的价格-成交量场进行建模，联合预测未来五分钟限价订单的价格放置位置、总量和预测不确定性，并配套开发了全场评估框架。 |
| [^162] | [The LLM-Attacker: A Unified Red-Team Framework for Privacy Evaluation of Split and Distributed LLM Systems](https://arxiv.org/abs/2610.22722) | 本文提出LLM-Attacker统一红队评估框架，通过记录攻击者的选择实现切分式与分布式大语言模型隐私评估的可复现与可比较，并揭示训练期间返回梯度会使token恢复率从94.20%升至97.38%、精确文档恢复率从13.71%升至37.77%。 |
| [^163] | [CoLU: Confidence-Weighted Layerwise Unlearning for Large Language Models](https://arxiv.org/abs/2610.22721) | 该论文提出CoLU方法，通过置信度加权的遗忘与保留正则化仅更新单个选定的Transformer层来实现大语言模型遗忘，并配合基于归一化遗忘-保留分数的简短层筛选策略高效选取最优层，在降低计算成本的同时更好地平衡了遗忘效果与知识保留。 |
| [^164] | [Spectral Limits of Polynomial Graph Filters on the Long-Range Ising Model Benchmark](https://arxiv.org/abs/2610.22713) | 该论文证明了在长程伊辛模型基准上，作为各向同性消息传递和谱图网络线性核心的邻接矩阵多项式滤波器存在一个无法通过提高多项式次数突破的误差下限，并指出一个约五十个系数的跨步滤波器能够突破该极限。 |
| [^165] | [Normalizing Flow Maps](https://arxiv.org/abs/2610.22711) | 本文提出标准化流映射（NFM），将流映射的少步生成能力与标准化流的精确似然评估相结合，通过学习底层动力系统的积分器，在构造上同时实现高效少步采样和精确密度估计。 |
| [^166] | [Data Provenance Aggregation and Structural Descriptors for Query-Template Identification](https://arxiv.org/abs/2610.22705) | 本文提出通过聚合元组级Lineage与Why-provenance构建查询级画像并推导溯源结构描述符（PSD），利用数据依赖与结构属性实现查询模板识别。 |
| [^167] | [Chance-Constrained Personalized LLM Routing with Online Preference Learning](https://arxiv.org/abs/2610.22695) | 该论文提出了一种基于机会约束和在线偏好学习的个性化LLM路由算法，通过受约束上下文赌博机框架在最小化推理成本的同时，对实际响应满意度提供了高概率理论保证。 |
| [^168] | [Optimal approximation of smooth functions by ReLU neural networks](https://arxiv.org/abs/2610.22693) | 本文证明了宽度为O(N)、深度为O(L)的ReLU网络对任意光滑度s和维度D下的Hölder函数类均能达到最优逼近误差$(N^2L^2\log N)^{-s/D}$，并由此导出无额外对数因子的极小极大回归速率。 |
| [^169] | [Quantum-Classical Neural Networks for Solving Lindblad Master Equations](https://arxiv.org/abs/2610.22691) | 该论文提出一种结合经典神经网络与参数化量子电路的变分框架来求解林德布拉德主方程，通过纯化表示在构造上保证密度矩阵的物理性质，并从理论上证明经典参数化能够扩大有效假设类、提升模型的表达能力与可控性。 |
| [^170] | [AlphaBet: Reinforcement Learning for LLM Event Forecasting Agents](https://arxiv.org/abs/2610.22684) | AlphaBet是一个智能体强化学习框架，通过训练单个LLM策略在事件解决前的多个时间点自主搜索证据并报告概率，利用适当评分规则以已解决事件实现端到端监督，弥补了现有预测训练方法将证据获取排除在学习循环之外的不足。 |
| [^171] | [Edit Discovery in Human-AI Interactions](https://arxiv.org/abs/2610.22661) | 提出编辑发现优化（EDO）框架，通过在不知晓是否会产生编辑的情况下智能选择将哪些人机交互发送给成本高昂的编辑器（如人类专家或更强AI），以最小化错过的编辑机会并节约有限资源。 |
| [^172] | [Is a Linear Probe Evidence of a Linear Representation?](https://arxiv.org/abs/2610.22655) | 该论文证明线性探针的高准确率不足以证明线性表示的存在——即使是随机特征也能实现线性读出——并提出反向预测性、谱集中度等三种非对称诊断方法，以区分真正的线性世界模型与核机器式的弯曲流形表示。 |
| [^173] | [The Ephemeral Cost of Switching Optimizers Across LLM Training Stages](https://arxiv.org/abs/2610.22650) | 该论文通过系统性实验首次发现，在LLM预训练与后训练阶段之间切换优化器（如AdamW与Muon之间）并不会造成实质性的性能损失，最终性能主要由各阶段所用优化器的独立效应决定，且在SFT中增加Muon更新比例反而能提升下游性能。 |
| [^174] | [Alert-List Concordance for Evaluating Imputation Strategies in ICU Sepsis Prediction](https://arxiv.org/abs/2610.22635) | 该研究提出了“警报列表一致性”这一新指标，通过比较前k%警报患者集合的Jaccard相似度来评估不同插补策略在ICU脓毒症预测中是否识别出相同的高风险患者，弥补了传统重建误差和总体判别指标无法衡量患者级别警报一致性的不足。 |
| [^175] | [Network-Wide Freeway Speed Nowcasting and Forecasting from Sparse Fixed Sensors with a Probe-Supervised Graph Convolutional Recurrent Network](https://arxiv.org/abs/2610.22630) | 该论文提出GCRN+模型，一种探针监督的图卷积循环网络，仅利用15.6%路段的固定传感器数据即可对达拉斯—沃斯堡地区全部4,127条有向高速公路路段实现速度的即时估计与未来一小时预测，且随网络规模线性扩展并可在CPU上训练。 |
| [^176] | [Understanding Stability of Outer Nesterov Momentum on Stochastic Pseudo-Gradients](https://arxiv.org/abs/2610.22623) | 该论文推导出随机伪梯度下外层Nesterov动量的精确均方稳定性准则，揭示内部批大小决定了局部步数增加会使训练稳定或不稳定，且稳定性所需的工作节点数量可能随局部步数呈指数级增长。 |
| [^177] | [Noise-shaped Sampling and Annealing for Scalable On-the-edge Training of Binary and Ternary AI Models](https://arxiv.org/abs/2610.22621) | 本文提出ΣΔ-Sampler优化器，利用二值/三值网络训练损失与高阶伊辛模型能量的结构等价性，通过一阶ΣΔ噪声整形环路嵌入随机采样-退火步骤，将量化与采样噪声推挤出低频适应频带，从而实现可扩展的边缘端离散权重网络训练。 |
| [^178] | [Shift-Reliance Alignment: Ranking Vision-Language Models by the Cues They Trust](https://arxiv.org/abs/2610.22610) | 该论文提出一种“线索依赖度”测量方法，通过编辑少量图像校准检测器，无需在目标域上运行任何候选模型，即可预测哪个视觉-语言模型在目标域的物体存在性问题上表现最佳，并能预测模型被贴纸攻击误导或产生幻觉的倾向。 |
| [^179] | [Rarity of Singletons Matters & Supplementary Materials](https://arxiv.org/abs/2610.22604) | 提出脆弱性感知（FA）框架，通过量化单例统计量的脆弱性，以严格凹且具有唯一闭式解的熵-KL优化对经典缺失质量与未见物种估计量进行自适应轻量级修正。 |
| [^180] | [Can LLMs predict the outcomes of machine learning experiments?](https://arxiv.org/abs/2610.22599) | 该论文构建了包含288个LLM自生成实验想法的数据集，通过让五个前沿大语言模型在机器学习实验训练前预测其影响力并与实际结果对比，衡量了这些模型判断实验是否值得运行的直觉准确性。 |
| [^181] | [Diagnosing GNN Failure Modes with Layer-Wise Probe Entropy](https://arxiv.org/abs/2610.22595) | 该论文提出一种冻结模型并逐层拟合线性探测器的探测熵诊断方法，能够区分有益、中性与有害的平滑，并通过熵退化的出现时机区分GNN的两种主要失效模式——异配性下的结构失配与深层过平滑。 |
| [^182] | [Flat Loss, Evolving Features: Grokking in Modular Addition](https://arxiv.org/abs/2610.22593) | 该论文证明网络可以在训练交叉熵几乎平坦地保持正值的条件下继续学习：在模加法任务上，ReLU网络在保持100%训练准确率的同时隐藏特征持续演化、留出集准确率从零升至90%以上，并通过两个有限窗口定理从理论上证明了这种平坦损失与顿悟式泛化的共存性。 |
| [^183] | [Initiating Removal from Operational Symptoms in Federated Vehicular Networks](https://arxiv.org/abs/2610.22584) | 本文提出ATONE方法，将联邦遗忘嵌入联邦学习聚合规则中，无需移除请求指明具体车辆或轮次，即可由车辆运行症状触发，通过将症状投影到最近全局增量张成的子空间来评分并中和错误贡献的有害效应。 |
| [^184] | [Noise-Robust Forced Alignment for Spontaneous Speech](https://arxiv.org/abs/2610.22583) | 本文提出一种面向自发嘈杂语音的抗噪声强制对齐方法，通过上采样WavLM特征、为填充词和非词汇声音事件设置显式音素单元以及判别投影，将词边界误差较最强基线降低17%-19%，灾难性错位比MFA最佳模型减少2.7倍。 |
| [^185] | [Embodiment-Classifier Guidance: Transferring Frozen VLAs Across Robot Dynamics without Demonstrations](https://arxiv.org/abs/2610.22576) | 该论文提出“本体分类器引导”方法，通过轻量级分类器的梯度在去噪过程中引导冻结的π0流策略，无需演示数据或参数更新即可将其迁移至新的机器人本体，在LIBERO基准上使目标本体平均成功率从72%提升至80%。 |
| [^186] | [Structure-Aware Spatial Kernels for Gaussian Process Distributed Port-Hamiltonian Systems](https://arxiv.org/abs/2610.22575) | 提出一种结构感知的核设计，通过张量积协方差将状态与空间依赖解耦，使高斯过程分布式端口哈密顿系统能够高效建模空间异质系统并融入定性空间知识。 |
| [^187] | [Coverage Cliffs in Off-Policy Quantile Certification](https://arxiv.org/abs/2610.22565) | 本文提出 TailCert 程序，通过构建 PAC-Bayes Bernstein 下置信包络，利用已记录的上下文老虎机数据安全地为离策略选择认证上尾分位数，并揭示了在稀有动作类上所需样本量随重叠因子线性增长、随策略搜索与置信度参数对数增长的定量规律。 |
| [^188] | [Deterministic Bounds on Accuracy and Stability of Metric-Space Regression and Classification](https://arxiv.org/abs/2610.22564) | 该论文提出了一个几何框架，通过基于最坏情况输入和目标噪声计算的平均核大小，为度量空间回归和分类任务提供了与模型无关的、可计算的确定性精度与误差界限。 |
| [^189] | [When Plans Change, Remember: Replanning-Guided Visual Memory for Vision-Language-Action Models](https://arxiv.org/abs/2610.22556) | 提出RePlanMem-VLA，将VLA策略自身的重规划行为（即连续动作块预测之间出现的异常巨大分歧）作为在线写入视觉记忆的信号，以选择性保存关键帧，解决长时程任务中关键视觉信息已不在当前观测中的问题。 |
| [^190] | [On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction](https://arxiv.org/abs/2610.22543) | The paper proposes SPOT, which achieves lossless accelerated decoding for diffusion language models by learning to predict the commit order of positions: a 0.62M-parameter small model reads the target model's output distributions to propose a candidate token tree, and the target model verifies all candidates in a single pass. |
| [^191] | [HeRE: A Novel Benchmark for Hebrew Relation Extraction via LLM-Guided Denoising of Knowledge Graph Alignments](https://arxiv.org/abs/2610.22542) | 本文提出HeRE——首个基于LLM引导去噪知识图谱对齐构建的希伯来语关系抽取基准，提供了一种可扩展、语言无关的低资源语言关系抽取数据集创建方法。 |
| [^192] | [Don't Retrain, Remix: Adaptation of Web Agents using Mixture Hypernetworks](https://arxiv.org/abs/2610.22539) | WebMix 提出混合超网络方法，通过侦察智能体生成界面手册并由轻量级路由器组合预训练的低秩适配器，使网页智能体仅需一次侦察和一次前向传播（无需任何梯度更新）即可适应未见过的网页界面。 |
| [^193] | [Learning the Shared Latent Structure of Cryo-EM Inverse Problems](https://arxiv.org/abs/2610.22523) | 该论文提出通过结合全局目标与掩码图像块目标的互补自蒸馏方法，学习冷冻电镜各逆问题（姿态估计、异质性分析和重构）之间共享的潜在结构表征，以统一的表示提升多个冷冻电镜分析任务。 |
| [^194] | [Learning smooth dynamics via Kinematic Interpolants](https://arxiv.org/abs/2610.22519) | 该论文提出运动学插值子（KI）框架，将随机插值方法扩展至包含位置与速度测量的相空间多快照数据，从而无需模拟即可可扩展地高保真重建潜在的非线性动力学。 |
| [^195] | [Source-Aware Dynamic Pricing with Strategic Buyers](https://arxiv.org/abs/2610.22516) | 本文提出了SALE定价策略，利用已验证特征进行局部化价格探索并结合自报特征校正策略性操纵，从而解决高维半参数环境下存在策略性买家操纵的来源感知动态定价问题。 |
| [^196] | [Towards Reconstructing Geographically Diverse Architecture with 3D Foundation Models](https://arxiv.org/abs/2610.22514) | 该论文提出了高地理覆盖率的三维建筑地标基准ArchWorld，并通过误差分析发现三维基础模型的性能存在地理差异，其成因不仅包括训练数据代表性差异，还包括场景规模、区域特有建筑特征和背景干扰因素的影响。 |
| [^197] | [Predicting Privileged Information Effectiveness for On-Policy Self-Distillation](https://arxiv.org/abs/2610.22512) | 该论文系统评估了16种特权信息配置在多轮智能体任务在线策略自蒸馏中的效果，发现其有效性取决于内容、生成来源和学生模型，并提出可利用基于KL散度的指标在训练前预测特权信息的效用。 |
| [^198] | [Generative Value Classifier](https://arxiv.org/abs/2610.22497) | 该论文提出生成式价值分类器，通过反向提问“在每种可能回报下动作出现的概率”，以回报条件策略作为评论家网络的生成组件，从而利用每一个动作作为监督信号来训练价值函数，在离线和在线强化学习任务上超越了领先的判别式方法。 |
| [^199] | [Upper bounds on the number of causal pieces in spiking neural networks](https://arxiv.org/abs/2610.22484) | 本文首次为前馈单脉冲神经网络中因果片段的数量推导出两个互补的上界——一个基于多重伯努利数、适用于广泛脉冲神经元模型的组合界，以及一个基于超平面排列的界，从而填补了脉冲神经网络因果片段数量极限研究的空白。 |
| [^200] | [Has GPT-6-Astra Achieved Embodied Intelligence?](https://arxiv.org/abs/2610.22471) | 本报告通过在涵盖桌面操作、人形机器人、移动平台和灵巧手的 12 个代表性基准上进行大规模统一评估，证明 GPT-6-Astra 在机器人操作与导航任务中取得了前所未有的性能，其能力挑战了为单个任务和本体构建专用模型的传统范式，并提出了 ICL 泛化性、成本、物理状态验证、故障恢复和长时程进度保持五大未来研究方向。 |
| [^201] | [Progressive Misalignment Can Emerge from Benign Recursive Self-Distillation](https://arxiv.org/abs/2610.22467) | 在递归自蒸馏过程中，即使只使用良性提示和模型自身的安全生成内容，安全模型也会随着迭代轮次不断累积失准，最终在多种有害查询上表现出渐进式失准现象。 |
| [^202] | [Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss](https://arxiv.org/abs/2610.22454) | 研究发现MoE模型在预填充阶段削减路由专家所造成的短答案F1损失，主要集中于少数关键提示词Token（如答案前的聊天模板Token和开头的指令Token），且损害主要在于答案形式而非答案内容，保持这些敏感位置的完整专家远比恢复超级专家或超大激活更有效。 |
| [^203] | [Lean4AgenticRL: Advancing Agentic RL with Verifiable Multi-level Process Rewards](https://arxiv.org/abs/2610.22448) | 提出Lean4AgenticRL框架，首次将覆盖低层工具调用、中层工作流阶段和高层结果的统一多层次过程奖励锚定在Lean4形式化模式上，从而有效提升LLM开放式智能体强化学习的训练效果。 |
| [^204] | [Learnable Splines for Unstructured Neural Operators](https://arxiv.org/abs/2610.22446) | 提出 L-SplineFNO，通过将 FNO 主干与稀疏可学习的张量积 B 样条变换相结合，实现非结构化工程网格与潜在笛卡尔网格之间内存高效的转换，并在三维工程基准测试中取得优异的预测精度。 |
| [^205] | [Null Controls Are Necessary but Not Sufficient: Evaluating Acceptance Guards for Scientific Discovery and Interpretability](https://arxiv.org/abs/2610.22445) | 本文提出四种可执行变换操作与三重接受判据（零接受、正接受、推论一致性），证明零控制测试对自动科学发现而言必要但不充分——退化的零映射约化能通过96%混沌系统的闭合性检验，只有结合独立的观测量保持测试，才能可靠地区分真正的科学约化与退化假设。 |
| [^206] | [Sparse Autoencoders Can Induce Local Geometry for Mitigating Catastrophic Forgetting](https://arxiv.org/abs/2610.22439) | 该论文提出SAGA方法，将每个任务边界训练的稀疏自编码器转换为对表示漂移的局部低秩几何约束，从而在保持可塑性的同时有效缓解持续学习中的灾难性遗忘。 |
| [^207] | [Rank Measures of Fine-Tuning Updates Need a Noise Baseline](https://arxiv.org/abs/2610.22438) | 研究表明，微调权重更新的秩度量必须以随机噪声为基线来解读——更新实际呈“少数强方向叠加长尾”的结构，且这种秩结构并非对齐训练的独有特征。 |
| [^208] | [CF-NODE: Counterfactual Spatial Node Intervention for Test-Time Adaptation Under Domain Shift, Evaluated Across Five Diabetic Retinopathy Datasets](https://arxiv.org/abs/2610.22437) | 提出CF-NODE反事实测试时自适应框架，通过对空间节点的受控干预评估预测稳定性，从而在五个糖尿病视网膜病变数据集上提升跨域分级的鲁棒性。 |
| [^209] | [Generative adversarial networks with Monte Carlo dropout uncertainty quantification for structural health monitoring: Fibre Bragg grating data augmentation comparison in Hawk T1A aircraft](https://arxiv.org/abs/2610.22436) | 本文提出一种结合条件生成对抗网络与蒙特卡洛Dropout不确定性量化的数据增强框架，在Hawk T1A飞机全尺寸数据集上验证了其生成的光纤布拉格光栅响应信号具有物理真实性和统计一致性，可有效用于结构健康监测中的损伤分类。 |
| [^210] | [TRACE: Transformer Residual Activation Codebook Encoding for Communication-efficient Distributed Edge Inference](https://arxiv.org/abs/2610.22435) | TRACE通过在每个Transformer层的注意力后残差处引入向量量化器，将交换的激活值编码为紧凑码本索引，使接收方通过查表直接恢复键/值表示，从而消除通信后串行计算的结构性瓶颈，实现通信高效的分布式边缘推理。 |
| [^211] | [BOLT: Multi-Bottleneck Bound-Guided Learning for Flexible Job-Shop Scheduling with Time Lags and Resource-Coupled Transport](https://arxiv.org/abs/2610.22433) | BOLT通过融合链、机器容量和车队容量三种松弛的多瓶颈下界来引导强化学习，求解带时滞与资源耦合运输的柔性作业车间调度问题，无需求解器即可认证调度方案的次优性差距，并在运输密集场景中显著优于运输惩罚策略。 |
| [^212] | [An Adaptive-Horizon Graph Neural Operator for coupled surrogate morphodynamic modelling](https://arxiv.org/abs/2610.22432) | 本文提出自适应时域图神经算子AH-GNO，直接在非结构网格上从水动力与地形状态学习预测河床高程增量，并能根据流动和河床演变自适应调整历史长度与更新间隔，从而实现高效的地貌动力学耦合代理建模。 |
| [^213] | [GPU-Accelerated Robust MA--RIS Pre-Optimization for Communication and Target Illumination](https://arxiv.org/abs/2610.22431) | 提出GPU加速的鲁棒预优化框架RAMP，通过联合优化慢时间尺度的可移动天线构型与RIS相位，生成兼顾最差用户速率与最差目标照射的可复用帕累托解集，免去对每个信道实现反复优化的需要。 |
| [^214] | [CGDT-CESN: Correlation-Guided and Dynamics-Trained Clustered Echo State Network for Multivariate Time-Series Prediction](https://arxiv.org/abs/2610.22430) | 该论文提出了一种相关性引导与动力学训练的聚类回声状态网络（CGDT-CESN），通过基于变量间相关性构建并训练聚类间连接，在保持计算效率的同时捕捉时变的跨变量依赖关系，显著提升了多元时间序列的预测精度。 |
| [^215] | [Deep Learning for Forest Plantation Mapping in Godavari Districts of Andhra Pradesh, India](https://arxiv.org/abs/2610.22415) | 本文利用深度学习方法对印度安得拉邦东、西戈达瓦里地区的小面积人工林进行遥感制图，克服了人工林与周围植被及其他土地覆盖高度混合、难以用中等分辨率卫星影像检测的挑战。 |
| [^216] | [Binding-Specific Relearning Savings for Test-Time Continual Learning](https://arxiv.org/abs/2610.22381) | 该研究发现在测试时持续学习中，模型习得的特定事实绑定（而非对实体、答案词汇、格式或查询槽位的熟悉性）能在部分遗忘后带来再学习加速，且错误绑定不产生此优势，表明现有仅依赖输出准确率的评估方式无法捕捉先前经验对后续更新的塑造作用。 |
| [^217] | [Trust but Verify: Designing Region-Inhomogeneous Forward Dynamics for Discrete Diffusion](https://arxiv.org/abs/2610.22380) | 该论文提出区域非均匀前向动力学，通过在训练和推理阶段对可疑上下文注入低速率噪声，打破离散扩散中“完全信任上下文”的假设，从而有效抑制错误在生成过程中的传播。 |
| [^218] | [When is a large language model a good calibration agent? Physics-labeled prompting for budgeted engineering simulator calibration](https://arxiv.org/abs/2610.22379) | 该研究发现在计算预算受限的工程模拟器校准任务中，以物理诊断描述为条件的提示能让大语言模型在早期评估阶段显著优于传统无导数优化器——首个物理信息提议在25次试验中23次改善初始设计，五次评估后中位校准误差降低21.9倍。 |
| [^219] | [Memory Bandwidth, Not FLOPs: Profiling and Accelerating Query-Based Segmentation Decoders on Edge GPUs](https://arxiv.org/abs/2610.22377) | 该研究揭示了边缘GPU上基于查询的分割模型解码器的真正瓶颈是内存带宽（跨内核激活数据传输）而非FLOPs，并提出CompactInst，利用4倍无参数空间池化减少内存传输以实现加速。 |
| [^220] | [Physics-Transfer Learning: A Physics-Constrained Theory for Zero-Shot Generalization Along the Ladder of Complexity](https://arxiv.org/abs/2610.22376) | 该论文提出物理迁移学习（PT）理论，通过构建连接低成本源域与复杂目标域的锚定域链，并引入可在观测目标域标签前评估迁移效果的可计算可行性与可靠性指标，实现沿复杂性阶梯的零样本泛化。 |
| [^221] | [Beyond Fixed Acquisition Functions: Temporal Difference Active Learning for Perturbation Discovery](https://arxiv.org/abs/2610.22122) | 提出了一种名为TDAL的主动学习方法，将预算约束下的扰动发现建模为序贯决策问题，通过时间差分学习从单次筛选轨迹中在线学习采集函数，突破传统固定采集函数的局限，实现自适应的实验选择。 |
| [^222] | [Missed Detections and False Alarms in a CNN Technosignature Detector on Breakthrough Listen Data](https://arxiv.org/abs/2610.22050) | 该研究通过向真实绿岸天文台数据中注入15,000个不同形态和振幅的信号，对一个冻结的CNN技术签名探测器进行压力测试，发现其召回率随信号增强反而下降，且某些干扰形态的假阳性率高达0.627，揭示了现有机器学习探测器在训练分布之外的脆弱性。 |
| [^223] | [N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models](https://arxiv.org/abs/2610.21810) | 本文提出条件程序（CP）压缩表示方法，仅用两比特编码按需重构大语言模型n-gram记忆查找表中被请求的行，相比标准2比特量化将输出KL散度降低41–63%，并将DeepSeek的表存储从203GB压缩至51GB。 |
| [^224] | [Encoded but Exposed: Language Embeddings can Reveal More Than They Should in Recommender Systems](https://arxiv.org/abs/2610.21713) | 该研究揭示了推荐系统中的语言嵌入存在严重隐私风险——简单的余弦相似度攻击即可高精度恢复物品身份，泄露还可进一步传播导致用户再识别，而以混淆描述作为防御手段则面临明显的泄露-效用权衡。 |
| [^225] | [How language model agents reshape price competition](https://arxiv.org/abs/2610.21711) | 大语言模型定价智能体能够突破标准学习算法难以摆脱的伯特兰竞争“边际成本定价”基准，且其行为关键取决于部署接口——直接从原始市场历史发布价格的智能体可在自我对弈中维持超竞争高价，而从自身利润摘要中输出概率分布的智能体则趋向竞争性定价。 |
| [^226] | [Router Drift Control: Mitigating Off-Policy Degradation in Mixture-of-Experts Reinforcement Learning](https://arxiv.org/abs/2610.21682) | 提出路由漂移控制（RDC）方法，将采样与训练阶段的路由分歧作为离策略过时信号，通过序列级过滤拒绝高漂移响应，在不引入梯度偏差的前提下缓解混合专家强化学习中的路由漂移与离策略退化。 |
| [^227] | [ManiFusion: Unlocking the High-Throughput Generation via Superposition in Manifold Space](https://arxiv.org/abs/2610.21433) | ManiFusion提出了一个与骨干网络无关的框架，通过在去噪骨干前后对多个样本进行融合与解融合，并采用轨迹融合调度来对齐去噪阶段，从而在单次去噪中并行处理多个样本，突破单样本延迟优化的瓶颈，实现高吞吐量的扩散模型图像生成。 |
| [^228] | [Does On-Policy Distillation Need to Complete Every Rollout?](https://arxiv.org/abs/2610.21337) | 该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。 |
| [^229] | [SALTO: Solver-Anchored Lifted Transfer Operators for Nonlinear Event Inference](https://arxiv.org/abs/2610.21049) | SALTO是一个基于Koopman算子的框架，通过在提升空间中构建可复用的线性化伴随算子，仅需一次编码器前向传播和一个内积即可快速推断任意有限幅度的局部非线性事件源。 |
| [^230] | [Time-partitioning Gaussian Process Mixture Model](https://arxiv.org/abs/2610.20860) | 提出了一种贝叶斯非参数框架TPGPMM，通过自适应Pitman-Yor先验与特定状态高斯过程的结合，自动识别时间序列中的持续状态与变点，无需马尔可夫假设即可建模非平稳依赖，并能有效区分持续变化与瞬态冲击。 |
| [^231] | [Spectral Indistinguishability and Learning-Curve Identifiability in Quadratic Networks](https://arxiv.org/abs/2610.20678) | 该论文构造了谱测度完全一致（谱不可区分）但学习曲线仍可区分的二次网络对，证明谱摘要无法唯一辨识网络，而标量学习曲线可以实现辨识，且其查询复杂度随循环长度呈指数增长。 |
| [^232] | [CascadeLedger: Counterfactual Auditing of Information and Resource Integrity in LLM Cascades](https://arxiv.org/abs/2610.20668) | CascadeLedger提出了一个面向预算受限LLM级联的反事实执行审计框架，通过聚合等价轨迹有效性和反事实前缀无干扰性等机制，审计路由决策是否仅使用了合法可观测信息，并能定位最早违规点与最小隐藏见证。 |
| [^233] | [Learnable Hodge-Laplacian Driven Mixture of Experts for Steady-PDE Surrogates on Coarse Unstructured Meshes](https://arxiv.org/abs/2610.20585) | 该论文提出一种由可学习霍奇-拉普拉斯算子驱动的混合专家架构，将解按频谱尺度分配给不同transformer专家并通过逐顶点门控权重重组，在粗非结构网格稳态PDE代理任务上相比Transolver将相对$L_2$误差降低了31%。 |
| [^234] | [ActiveEvidence: Decision-Focused Evidence Graphs with Verifiable Acquisition Gaps](https://arxiv.org/abs/2610.20532) | 该论文提出ActiveEvidence框架，通过反事实证据交互图（CEIG）和可验证的ε-决策证书，使多模态智能体能够高效决定何时获取视觉证据、仅评估可能改变决策的关键交互，并在证明进一步交互无法改变所选动作时及时停止。 |
| [^235] | [SafeAdapt-Poker: Counterfactual Risk Calibration for Budgeted Adaptation under Opponent Shift](https://arxiv.org/abs/2610.20473) | 该论文提出 SafeAdapt-Poker，将对手漂移下的预算化适应建模为仅依赖决策前信息的双重反事实序贯控制问题，通过单调策略风险模型与配对预算状态延续模型分别校准偏离的局部下行风险和预算消耗的未来代价，并用逐路径累计KL账本约束总偏离量，从而在即时适应价值与安全风险之间实现可校准的权衡。 |
| [^236] | [Hypothesis Selection via Weighted Medians](https://arxiv.org/abs/2610.20375) | 本文提出了一种基于加权中位数的假设选择算法，在达到最优近似因子和样本复杂度的同时，将运行时间对失败概率的依赖从多项式级改进为对数级。 |
| [^237] | [TANGO: RNA Topology and Geometry Co-Design](https://arxiv.org/abs/2610.20277) | 提出TANGO框架，首次将RNA逆设计表述为拓扑与几何协同设计问题，通过多目标多智能体强化学习和分阶段课程训练（二级结构拓扑感知策略学习、三级结构迁移训练、前沿引导微调），实现同时满足二级碱基配对拓扑与三级几何结构的RNA序列设计。 |
| [^238] | [Gap Detection: Mitigating Homogenization through Demand and Defensibility](https://arxiv.org/abs/2610.20162) | 提出Gap Detection框架，通过结合查询相关性（Demand）与现有内容相异度（Defensibility）的GapScore目标函数，在文本嵌入空间中检测相关但未被占据的方向，以缓解语言模型开放式回答的同质化问题。 |
| [^239] | [Fully sequential Gaussian process regression for preference learning](https://arxiv.org/abs/2610.19391) | 该论文针对目标函数只能通过带噪声成对比较来评估的连续全局优化问题，提出了基于矩匹配的完全序贯高斯过程近似方法DOVE，并首次为这类完全序贯近似贝叶斯推断建立了统计误差和期望简单遗憾的收敛速率保证。 |
| [^240] | [Riding the Fixed-Point Curve: Image Classification in the Continuous Thought Machine](https://arxiv.org/abs/2610.19220) | 本文揭示了连续思维机器（CTM）图像分类的内在机制：每个思考步骤是具有平衡放大的吸引性非正规映射，神经状态的不动点随每步移动，思维轨迹即这些不动点描绘的曲线（大部分运动跨图像共享并表现为行波活动），从而解释了CTM的涌现思维过程。 |
| [^241] | [Statistical Vignettes on the Grokking Phenomenon](https://arxiv.org/abs/2610.18811) | 该论文在一个可解析求解的简单模型中研究“顿悟”现象，厘清了初始化、信号强度、权重衰减、样本量与输入维度等因素通过不同机制产生相似学习曲线的作用，给出了顿悟必然发生的严格条件、泛化平台长度的解析渐近表达式，并揭示插值边缘是由消失谱隙刻画的一个独特动力学区域。 |
| [^242] | [SP$^2$ec: Adaptive Self-Speculative Decoding for Vision-Language Models](https://arxiv.org/abs/2610.18735) | 提出免训练的赌博算法SP$^2$ec，利用自推测解码的单峰吞吐量特性，在生成过程中自适应调整草稿配置并最小化墙钟时间遗憾，从而加速视觉语言模型的推理。 |
| [^243] | [Kwai Summary Attention V2 Technical Report](https://arxiv.org/abs/2610.18654) | 提出KSA V2架构，通过允许不同层设置不同的摘要压缩分块窗口，实现更灵活的KV缓存内存压缩，并解决全注意力层中摘要标记带来的额外计算开销问题。 |
| [^244] | [Agentic-MatriBO: Capability-Compiled Agentic Bayesian Optimization for Materials and Chemical Design](https://arxiv.org/abs/2610.17832) | 提出了能力编译型智能体贝叶斯优化框架Agentic-MatriBO，将BO组件表示为可组合的能力，由大语言模型策略提出优化配方，经编译器检查和确定性执行层验证后用于材料与化学设计。 |
| [^245] | [Follow the Perturbed Leader for Linear Bandits: $\sqrt{T}$ Regret over General Convex Sets](https://arxiv.org/abs/2610.17818) | 本文提出了首个基于FTPL的对抗性线性赌博机算法，通过一个适应估计奖励的扰动分布，在一般凸动作集上实现了 $\sqrt{T}$ 遗憾，改进了此前 $T^{2/3}$ 的遗憾界，同时比连续指数权重方法具有更好的计算效率和更简单的实现。 |
| [^246] | [LeanPRM: Lean-Grounded Process Supervision for Natural Language Proof Verification](https://arxiv.org/abs/2610.17697) | LeanPRM利用Lean证明助手自动为自然语言数学证明生成步骤级正确性标签，无需人工标注即可构建包含10,634个证明、15.2万个标注步骤的过程监督数据集，解决了证明验证器训练中标签获取困难的瓶颈。 |
| [^247] | [LoongSage: A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes](https://arxiv.org/abs/2610.16352) | LoongSage是一个智能体原生、完全异步的生产级强化学习框架，基于Megatron与SGLang构建且核心仅约1.4万行代码，通过统一编排白盒与黑盒智能体的AgentFlow范式，显著降低了前沿大语言模型智能体强化学习的采用、使用与调试开销。 |
| [^248] | [Error Correction in Trained TTT Layers](https://arxiv.org/abs/2610.16268) | 该研究剖析了TTT层的梯度更新机制，证明点积目标下不存在预测误差残差、LayerNorm目标下的预测衍生分量在发布初始化处会相互抵消，并通过实验量化了归一化预测对已训练检查点上梯度更新的实际贡献。 |
| [^249] | [Guarded Block Transfers for Polytope Distance: Linear Convergence with Explicit Constants and Gap-Certified Screening for SVM Training](https://arxiv.org/abs/2610.15829) | 该论文提出受保护块转移方法，将k次MDM转移聚合为一次受保护的精确线搜索，在Minkowski差多胞形上证明了具有显式常数的全局线性收敛，并利用TA证书实现了SVM训练的间隙安全筛选。 |
| [^250] | [DisCo Algebra: Why Disintegration and Combination Are Not Inverses](https://arxiv.org/abs/2610.15006) | 本文提出DisCo代数框架，用对称余操作刻画联合概率的分解、用非单位部分操作刻画组合，证明“组合∘分解”在可允许输入上是恒等操作，而“分解∘组合”则是一个有信息损失的重正化算子，从而揭示了分解与组合并非互逆。 |
| [^251] | [Dependency Reasoning for Aspect Sentiment Quadruple Prediction](https://arxiv.org/abs/2610.14873) | 该论文提出DRIFT方法，将依存句法树线性化后作为显式关系上下文直接注入LLM提示中，并结合检索增强的上下文学习与语法感知裁决机制，在方面情感四元组预测任务上取得了最优性能。 |
| [^252] | [How Well Do Prior-Data Fitted Networks Amortize Bayesian Inference?](https://arxiv.org/abs/2610.13764) | 本文将先验数据拟合网络（PFNs）的部署误差分解为先验、上下文和摊销误差，发现PFNs虽能很好地恢复目标后验预测分布，但在罕见设置下仍存在残余摊销误差，而贝叶斯模型平均方法可有效减少这种不确定性并提升预测性能。 |
| [^253] | [SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data](https://arxiv.org/abs/2610.13644) | 该论文提出了SelectArena——一个基于真实世界表格数据的现代特征选择基准测试，通过比较15种广泛使用的特征选择方法，弥补了现有基准依赖过时模型、临时评估协议和人工数据而无法指导实际应用的不足。 |
| [^254] | [AdaptLSTM: Efficient Adaptive Online Learning for Cloud Workload Forecasting under Distribution Drift](https://arxiv.org/abs/2610.12265) | AdaptLSTM 通过验证校准阈值检测分布漂移并进行选择性针对性更新，以约 20% 的计算成本实现接近朴素在线学习的预测精度，大幅提升云工作负载预测的效率。 |
| [^255] | [Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data](https://arxiv.org/abs/2610.11362) | 提出伯努利流模型（BFM），通过构建数据分布与纯噪声之间统一的连续全局伯努利概率流路径，克服了传统二值扩散模型在低函数评估次数下因单步似然近似导致样本质量严重下降的问题，无需蒸馏或额外训练即可实现高效的二值数据生成。 |
| [^256] | [Transferability of Learned States in Neural PDE Solvers](https://arxiv.org/abs/2610.10972) | 提出“复用契约”评估框架，将神经PDE求解器中学习状态的迁移收益与解精度和计算成本解耦，并实证发现固定预测器的收益会随校正算法不同而反转。 |
| [^257] | [The optimal information complexity of VC learning](https://arxiv.org/abs/2610.10600) | 本文通过构造一个eCMI为O(d)阶的随机化“5个基学习器多数投票”学习算法，首次经由CMI信息论分析框架恢复了VC类学习的最优PAC泛化保证。 |
| [^258] | [Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with an RBF Kernel](https://arxiv.org/abs/2610.10080) | 本文针对基于RBF核的高斯过程，在固定域渐近框架下建立了极大似然估计的精确渐近理论，攻克了密集采样强相关性与协方差矩阵非线性依赖带来的长期理论难题。 |
| [^259] | [Leaner Transformers Can Easily Learn to Cluster](https://arxiv.org/abs/2610.09760) | 本文提出了一种嵌入维度仅需 d+⌈log₂k⌉ 但表达能力不变的更精简Transformer来执行k均值聚类的Lloyd算法，并系统刻画了训练Transformer学习聚类算法时影响收敛性与泛化能力的关键因素。 |
| [^260] | [MS-ECG-FM: Towards a More Universal Electrocardiogram Foundation Model for Health Monitoring using Multi-source Contrastive Learning](https://arxiv.org/abs/2610.07662) | MS-ECG-FM通过对齐心电图、超声心动图、放射学和出院报告等多种临床记录进行多源对比学习训练，突破了仅依赖解读报告作为监督的局限，在包括少导联配置在内的所有心电图检测任务上全面超越现有方法。 |
| [^261] | [Learnable Spectral Activations](https://arxiv.org/abs/2610.07419) | 提出可学习谱激活（LSA），用谐波振幅可学习的残差截断傅里叶级数替代固定激活函数，将特征选择与频谱整形解耦为两条独立梯度更新的通路，从而提升隐式神经表示对局部化和空间变化信号的多谐波组合能力。 |
| [^262] | [RealtimeWAM: One-Step Asynchronous World Action Models](https://arxiv.org/abs/2610.06617) | 提出RealtimeWAM，通过教师锚定一致性蒸馏（TACD）实现一步动作生成，并采用异步推理消除视频专家与动作专家之间的顺序等待，显著提升世界动作模型的推理效率。 |
| [^263] | [On the Trade-off Between Information Loss and Generalization in Sparse Attention](https://arxiv.org/abs/2610.04424) | 本文通过Jensen-Shannon散度对稀疏注意力机制的信息损失进行理论刻画，并系统分析了信息损失与模型泛化能力之间的权衡关系。 |
| [^264] | [The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer](https://arxiv.org/abs/2610.03827) | 相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。 |
| [^265] | [Exploring the Trade-Off Between Structured Pruning and Fault Tolerance in Deep Neural Networks for Space Applications](https://arxiv.org/abs/2610.03117) | 本研究通过故障注入实验发现，深度神经网络的结构化剪枝虽然因减少冗余而提高了单次推理对单粒子翻转故障的敏感性，但更短的执行时间降低了遭遇故障的概率，从而有效平衡了这一负面影响。 |
| [^266] | [FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training](https://arxiv.org/abs/2610.02828) | FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。 |
| [^267] | [ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing](https://arxiv.org/abs/2610.02808) | ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。 |
| [^268] | [Generalization in Neural Networks Through the Lens of Magnitude Potential](https://arxiv.org/abs/2610.01633) | 本文提出基于度量量级理论的“量级势”新概念，通过logit层上类内与全数据集的量级势之比来刻画样本的表示情况，实验表明该比值与Feldman记忆分数相关，并能检测决策边界的结构变化、为模运算中的grokking现象提供几何指标。 |
| [^269] | [Redundancy Meets Synergy: Dependency-aware Expert Selection for MoE via Submodular Optimization](https://arxiv.org/abs/2610.00558) | 提出DS-MoE框架，通过次模函数差分优化感知专家间的冗余与协同依赖关系，为MoE模型实现更优的紧凑专家子集选择。 |
| [^270] | [Benchmarking Prompt Optimization of Large Language Models With Chess](https://arxiv.org/abs/2610.00416) | 该论文提出了一个基于1,118道Lichess国际象棋棋题的基准测试，利用国际象棋低成本精确评分、引擎辅助着法评估和可持续更新的可调难度题源等特性，来研究冻结大语言模型的自动提示词优化。 |
| [^271] | [FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training](https://arxiv.org/abs/2610.00385) | FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。 |
| [^272] | [ContractRL: Shielded Group-Relative Policy Optimization for Auditable Tool-Call Repair](https://arxiv.org/abs/2610.00328) | ContractRL提出了一种契约约束的顺序修复协议，将验证器引导的JSON修复建模为有界决策过程，通过契约派生的动作掩码过滤非法修复操作，在仅生成34.4个token的情况下实现了0.9362的语义成功率，并保证了修复过程的可审计性。 |
| [^273] | [Actions with Receipts: Jointly Binding Claims, Evidence, and Execution for Replayable Tool-Agent Auditing](https://arxiv.org/abs/2610.00327) | 该论文提出“声明锚定执行契约”，通过收据机制将智能体发出的声明、证据来源片段、执行前缀与来源版本联合绑定，并由确定性完整性验证器重建绑定关系，从而实现对工具型智能体行为的可重放审计。 |
| [^274] | [GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives](https://arxiv.org/abs/2609.39601) | GroundingPI是一个40亿参数的视觉定位基础模型，它将点和框表示为共享词表中的量化坐标，并通过多模态与空间预训练、监督微调及GRPO强化学习训练而成，在11种感知能力的34个基准测试上以平均73.68%的成绩超越更大的GPT-6 Astra，创下新的最先进水平，为物理智能提供了快速精确的视觉感知基础。 |
| [^275] | [GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed](https://arxiv.org/abs/2609.39600) | GroundAnything 是一个 40 亿参数的视觉定位基础模型，它将视觉定位重新定义为可并行生成的双向扩散过程，通过分块去噪、AR 到扩散的架构转换训练及 GRPO 强化后训练，在实现闪速并行解码的同时保证了精准的视觉定位效果。 |
| [^276] | [VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers](https://arxiv.org/abs/2609.36958) | 本文提出VStress可审计重放契约与VStress-CA相关性感知分配策略，通过估计重复验证器调用的条件边际信息来自适应分配预算并在信息不足时停止调用，从而在固定预算下以更低的调用成本获得更高的验证平衡准确率，同时通过受控审计刻画了该机制在极端损坏率下的适用边界。 |
| [^277] | [Handwritten Text Recognition Lives in the High-Pixel Variance Subspace](https://arxiv.org/abs/2609.35473) | 该论文揭示了手写文本识别的判别性信息集中在像素空间的高方差方向上，从而解释了为什么基于像素重建的自监督预训练方法优于对比学习方法，并在五种语言的六个手写基准上均取得最低字符错误率。 |
| [^278] | [RSD-Poker: Structure-Adaptive and Shift-Robust Risk-Utility Certification for Residual Policies in Imperfect-Information Games](https://arxiv.org/abs/2609.33669) | 该论文提出 RSD-Poker 框架，通过冻结残差候选库、学习并预先冻结策略可见的子群划分，为不完美信息游戏中的残差策略提供对部署分布偏移稳健的风险上界与效用下界双证书认证。 |
| [^279] | [Audit-First VAPO: Risk-Certified Selective Updates under Imperfect Verification](https://arxiv.org/abs/2609.33662) | 该论文提出Audit-First VAPO框架，通过将更新方向的离散准入与幅值的连续控制相分离，在有限验证预算下利用“接受-申诉-弃权”观察策略和有限样本统计界，实现对不完美验证器下有害更新风险、覆盖率与验证器调用率的可认证保障。 |
| [^280] | [Pretraining Transformers with Quantized Softmax in Attention](https://arxiv.org/abs/2609.33591) | 本文提出K区间注意力方法对注意力中的softmax进行量化近似，系统研究了网格校准、舍入方式和直通替代梯度位置等设计选择对Transformer预训练的影响，并推导了相应的反向传播规则。 |
| [^281] | [OSCC: Certified Observation-Safe Coupling Optimization for Gradient-Noise Control in Imperfect-Information Learning](https://arxiv.org/abs/2609.33543) | 提出了观测安全反事实耦合（OSCC）框架，通过定义可容许耦合类的多项安全准则并引入梯度感知的耦合优化准则，在非完美信息学习中安全地降低反事实比较噪声，同时针对策略梯度几何控制回报协方差以优化多动作策略。 |
| [^282] | [An Agnostic Sample Compression Scheme for Squared Loss of Near-Linear Size in the Fat-Shattering Dimension](https://arxiv.org/abs/2609.29696) | 本文首次构造了大小仅与 fat-shattering 维度近线性相关（乘以对数因子）且与样本量无关的平方损失不可知样本压缩方案，正面解决了 Attias 等人（ICML 2024）提出的开放性问题。 |
| [^283] | [Bandit Multiclass PAC Learning: Corrected Lower Bounds, Exact Families, and a Confidence Direct-Sum Phenomenon](https://arxiv.org/abs/2609.29694) | 本文修正了老虎机多类PAC学习的下界理论，指出原有基于BDS维度的下界并不正确，引入锚定维度aBDS证明了新的无常数下界，同时在上界中完全去除了对标签数K的依赖，并揭示了置信度直和现象。 |
| [^284] | [Certified Predictive Value-of-Advice Gating for Cost-Aware Language-Model Guidance in Reinforcement Learning](https://arxiv.org/abs/2609.29548) | 该论文提出了一种认证预测性建议价值门控框架，在调用语言模型前预测可能的响应并评估其价值，仅当预测价值的置信下界超过调用成本时才发起查询，并辅以动作特定证书管理执行，从而实现成本感知且带理论保证的强化学习引导。 |
| [^285] | [On the Sample Complexity of Active Learning with Membership Queries](https://arxiv.org/abs/2609.27241) | 本研究揭示了允许合成成员查询会显著改变统计学习的难度——某些在基于池的主动学习下只能实现多项式误差衰减的假设类，在允许合成查询后变得可指数级快速学习，表明成员查询合成是一种需要新分析工具来刻画的根本不同的学习模式。 |
| [^286] | [CVaR anchor regression protects against rare shifts](https://arxiv.org/abs/2609.27034) | CVaR锚点回归通过用尾部平均值代替跨环境平均残差平方的均值，能够针对罕见的大偏移提供精确的最坏情况风险保证，同时避免了传统方法因过度扩大保护范围而降低常见环境预测准确性的问题。 |
| [^287] | [Horizon-Aware Early Event Prediction for Tokamak Disruption Alarms](https://arxiv.org/abs/2609.24443) | 该论文针对全分布生存方法与运行决策中有限预测视界需求不匹配的问题，提出将早期事件预测（EEP）目标引入基于生存分析的托卡马克破裂预测，以深度生存机为基线应用TLS和survTLS两种方法在有限视界内直接预测破裂风险，并在DIII-D、Alcator C-Mod和EAST三个装置上进行了验证。 |
| [^288] | [No-Regret Mixing of LRU and LFU with Optimal Switching Cost](https://arxiv.org/abs/2609.07566) | 本文证明了现有混合缓存策略LeCar在遗忘型对手下会产生线性遗憾，并提出了一种基于Hedge的虚拟LRU与LFU缓存混合策略H-MC，能够在最优切换开销下实现对LRU与LFU较优者的无遗憾（次线性遗憾）性能保证。 |
| [^289] | [Capability-Gated Language Models: Security Composes, Utility Does Not](https://arxiv.org/abs/2609.00445) | 提出在单一模型权重内部实现按主体的“能力门控部署”，配置构成格结构，并证明安全限制在交运算下可组合累积（安全性随限制叠加而增强），而实用性不具备这种组合性。 |
| [^290] | [Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness](https://arxiv.org/abs/2609.00050) | 提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。 |
| [^291] | [A Human-in-the-Loop Autonomous Agent for Industry Time Series Forecasting](https://arxiv.org/abs/2608.30976) | 该论文提出了 CastClaw，一个面向工业时间序列预测的人机协同自主智能体系统，它在统一运行时中整合数据、专业预测模型、分析工具与用户输入，能够检验时间模式与用户约束、在证据不足时自动检索补充信息，并支持用户以自然语言交互式地制定和迭代优化预测任务。 |
| [^292] | [BenthicDINO: Physics-Informed Self-Distillation for View-Invariant Side-Scan Sonar Representations](https://arxiv.org/abs/2608.23215) | 本文提出了一种基于物理信息的自蒸馏框架（BenthicDINO），通过物理模拟增强和HSIC惩罚强制侧扫声呐图像表示的视角不变性，有效解决了声学伪影导致的特征混合问题。 |
| [^293] | [Online Inference in Distributional Temporal-Difference Learning](https://arxiv.org/abs/2608.14408) | 本文为分布时序差分学习提供了在线推断的理论框架，证明了平均估计器和自助法的渐近正态性，并扩展到了非光滑泛函的局部推断。 |
| [^294] | [Online Inference for Quantile Temporal Difference Learning in Distributional Reinforcement Learning](https://arxiv.org/abs/2608.12973) | 本文提出了一种基于随机缩放的在线推断方法，通过泛函中心极限定理支持，在不存储完整轨迹的情况下对分布强化学习中的QTD进行统计推断，显著减少了内存需求。 |
| [^295] | [RoutePack: Expert Placement and Attention-Aware Data Packing for MoE Reinforcement Learning](https://arxiv.org/abs/2608.12146) | RoutePack通过分层规划器同时优化MoE强化学习中的专家放置与数据打包，协调注意力与专家负载平衡，以最小化执行行数并提升训练效率。 |
| [^296] | [A Quantum Roadmap for Softmax Attention: Exact Born-Rule Analogs for Softmax Attention on the Probability Simplex](https://arxiv.org/abs/2608.11173) | 该论文证明在输入输出约束于概率单纯形的场景下，softmax注意力可以逐分量地精确映射为量子测量过程（Hadamard检验得分、玻恩规则余弦平方族、后选择温度与确定性数值聚合信道），为注意力的量子实现提供了完整路线图。 |
| [^297] | [Can Graph Learning Learn Circuits?](https://arxiv.org/abs/2608.08536) | 本文提出图电路学习（GCL）框架，首次将机制可解释性中的电路定位任务转化为图机器学习问题，通过图神经网络在多个模型-任务对上进行有监督的摊销式训练并泛化到未见案例，在留出案例上取得了0.946的中位数边AUROC。 |
| [^298] | [Predicting Task Difficulty Without Rollouts](https://arxiv.org/abs/2608.05797) | 该论文提出无需实际运行智能体（rollout）即可在覆盖编码、数学、网页导航等17个智能体基准上进行事前任务难度预测，从而降低试错带来的计算成本，并揭示AUC等常用指标可能掩盖较差的难度估计。 |
| [^299] | [Inverse Learning of Latent Risk-Neutral Densities from Irregular Option Quotes](https://arxiv.org/abs/2607.27188) | 该论文发现准确的期权定价并不能保证准确恢复潜在风险中性密度，并通过数值条件分析揭示定价问题存在严重病态性——大量定价方向数值为零、不同密度可产生相同价格——从而解释了不同模型在基准测试中排名差异的根源。 |
| [^300] | [PowerAtlas: Feasibility-Aware Language Models for Electricity-Computing Co-Scheduling](https://arxiv.org/abs/2607.26710) | 该论文将电力-计算协同调度形式化为结构化决策任务，发布了带可执行约束检查的ECBench基准，并提出通过可行性感知GRPO训练的PowerAtlas，使语言模型能生成满足电网与服务约束的联合调度方案。 |
| [^301] | [Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for MLIR](https://arxiv.org/abs/2607.18254) | 该论文提出从 MLIR 方言的操作定义规范（ODS）机械派生的推理时约束解码方法，无需重新训练即可实现跨方言泛化，并发布了涵盖三种方言共 410 个样本的四个自然语言到 MLIR 基准测试数据集。 |
| [^302] | [Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains](https://arxiv.org/abs/2607.13319) | OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。 |
| [^303] | [Teaching LLMs to Recommend and Defer in Underrepresented Epilepsy Care](https://arxiv.org/abs/2606.31036) | 提出MANANA非参数化提示学习框架，将处方错误转化为可审计的提示记忆，使大语言模型能从少量本地患者数据中学习乌干达儿科癫痫的本地化抗癫痫药物处方方案，并在需要时转诊至专科医师。 |
| [^304] | [Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning](https://arxiv.org/abs/2606.22995) | 提出群组-图策略优化（G2PO），一种针对多轮智能体任务的新型分组强化学习算法，通过显式利用状态转换固有的图结构，解决了长时程智能体强化学习中奖励稀疏延迟、高方差价值估计以及短视局部化信用分配的关键瓶颈。 |
| [^305] | [Learning What to Say to Your VLA: Mostly Harmless Vision Language Action Model Steering](https://arxiv.org/abs/2606.12299) | 提出一个通过交互式搜索有效语言序列并将其蒸馏为测试时语言反馈策略的框架，同时利用共形化改进头来预测并阻止有害的语言引导干预，从而可靠地提升视觉-语言-动作模型的闭环任务性能。 |
| [^306] | [TargetSEC: Plug-and-Play In-the-Wild Speech Emotion Conversion via Arousal-Conditioned Latent Style Diffusion](https://arxiv.org/abs/2606.07293) | TargetSEC提出了一种以说话人身份和连续情感为条件的潜在风格扩散框架，通过在紧凑潜在空间中生成情感风格嵌入，实现了即插即用的野外语音情感转换，在转换效果和语音自然度上均优于或持平现有基线方法。 |
| [^307] | [PE-MHL: Physics-Encoded Modular Hybrid Layers for Scalable Learning of Complex Systems](https://arxiv.org/abs/2606.04290) | 提出物理编码模块化混合层框架PE-MHL，通过增量添加子模型逐步细化基线物理模型，并从理论上证明训练误差随子模型数量单调不增且可证明收敛，从而实现复杂系统学习的可扩展性与鲁棒性。 |
| [^308] | [CauTion: Knowing When to Trust LLMs for Ensemble Causal Discovery](https://arxiv.org/abs/2606.03602) | 提出CauTion框架，通过共识过滤和LLM可靠性估计，将大语言模型的领域知识可靠地融合进统计因果发现算法集成中，从而同时克服纯统计方法、LLM错误风险和单一算法偏差这三方面的局限。 |
| [^309] | [Deep-learning-based low-energy trigger algorithms for the Hyper-Kamiokande experiment](https://arxiv.org/abs/2605.31391) | 该研究为超级神冈实验开发了基于深度学习的低能中微子触发算法，其中监督分类器对3 MeV单电子的信号识别效率达76.7%，显著超越传统击中计数触发器的26.4%。 |
| [^310] | [TASER: Task-Aware Stein Regularisation for Geometry-Driven Smoothness](https://arxiv.org/abs/2605.30601) | TASER通过惩罚Langevin Stein算子将模型导数与数据密度得分耦合，提出了一种无需切空间估计或内层扰动搜索的几何驱动正则化框架，有效提升深度神经网络在分布偏移下的稳定性与泛化能力。 |
| [^311] | [Learning to Reason About Implicit Multimodal Harm: From Diagnosis to Generalization](https://arxiv.org/abs/2605.29951) | 提出MuPHI诊断基准，揭示视觉-语言模型虽能理解单一模态信息，却无法将图像-文本组合推理为危害机制，为隐式多模态危害检测从诊断走向泛化提供了方向。 |
| [^312] | [FormInv: A Measurement Protocol for Semantic Invariance in Mathematical Reasoning Benchmarks](https://arxiv.org/abs/2605.29001) | 该论文提出FormInv测量协议，用于审计数学推理基准中自动生成改写题目的语义等价性，并引入条件不一致率（CIR）指标，揭示即使在所有模型都能正确回答的规范形式定理上，语义等价的改写仍会导致5-42%的答案翻转。 |
| [^313] | [Dynamics of Stochastic Momentum with Sparse Updates in High Dimensions](https://arxiv.org/abs/2605.28961) | 该论文首次在高维稀疏更新场景下精确刻画了动量优化的二阶矩动力学，揭示了其相变结构由动量保持时间尺度与学习时间尺度之比决定：学习慢于保持时极限等价于SGD，学习快于保持时系统失稳。 |
| [^314] | [Multitask learning with semiempirical orbital charges enables sample-efficient MLIPs](https://arxiv.org/abs/2605.24073) | 通过多任务学习引入计算成本低廉的GFN1-xTB半经验轨道电荷，显著提升了机器学习原子间势的样本效率和预测准确性。 |
| [^315] | [Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints](https://arxiv.org/abs/2605.22940) | 本文建立了在任务、风险与干预约束分离设定下的熵调控学习理论，通过噪声校准的对数行列式泛函导出精确谱力公式、可观测/预测零空间速度分解、标量反馈系数完整区间，以及多受保护风险下的极小极大可行性证书与锥投影公式，实现受保护风险约束下的可观测熵控制。 |
| [^316] | [ECPO: Evidence-Coupled Policy Optimization for Verifier-Audited Candidate Ranking](https://arxiv.org/abs/2605.21993) | 提出证据耦合策略优化（ECPO），通过证据循环奖励将Top-K排序与基于片段的可审计证据耦合，显著提升验证者可审计的可追溯性，且增益在多个基准、骨干模型与评估器下均保持稳健。 |
| [^317] | [Exploration and Experimentation with a Misspecified Simulator](https://arxiv.org/abs/2605.21458) | 该论文研究了在模拟器存在错误设定且真实系统动态发生漂移的情况下，是否、何时以及如何通过被动适应、持续探索或随机化实验等数据收集策略来重新校准模拟器。 |
| [^318] | [Graph Transductive Sharpening: Leveraging Unlabeled Predictions in Node Classification](https://arxiv.org/abs/2605.20248) | 该论文提出直推锐化（TS）这一损失层面的修改方法，通过最小化未标注节点的预测熵来提取直推式模型对未标注节点预测中蕴含的训练信号，从而改进半监督节点分类。 |
| [^319] | [Diffusion-Based Stochastic Operator Networks for Uncertainty Quantification in Stochastic Partial Differential Equations](https://arxiv.org/abs/2605.17107) | 该论文提出了随机算子网络（SON），通过将DeepONet与随机神经网络及扩散模型相结合，直接从含噪数据中学习，为随机偏微分方程的解同时提供均值预测和不确定性量化。 |
| [^320] | [OrbiSim: World Models as Differentiable Physics Engines for Embodied Intelligence](https://arxiv.org/abs/2605.16395) | OrbiSim将世界模型重新定义为全可微的物理引擎，实现了从显式状态转移到视觉观测生成的端到端可微仿真，支持可微接触建模、稀疏奖励下的梯度策略优化等经典仿真器难以完成的任务，并在预测保真度和控制性能上显著超越现有世界模型。 |
| [^321] | [GRAFT-ATHENA: Self-Improving Agentic Teams for Autonomous Discovery and Evolutionary Numerical Algorithms](https://arxiv.org/abs/2605.11117) | GRAFT-ATHENA框架将隐式的问题-方法关系显式化为可扩展的概率结构，使AI智能体团队的经验能够在结构相关的问题间迁移复用，在物理信息学习、血液流变学建模和高超声速流动求解等任务中达到或超越人类专家水平。 |
| [^322] | [RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards](https://arxiv.org/abs/2605.10899) | RubricEM提出一个统一的强化学习框架，通过量规引导的阶段结构化策略分解实现细粒度信用分配，并结合基于反思的元强化学习实现经验复用，解决了深度研究智能体在长时域任务且缺乏精确答案验证器时的训练难题。 |
| [^323] | [A Complete Characterization of Learnability for Adversarial Noisy Bandits](https://arxiv.org/abs/2605.09200) | 本文通过在函数类凸包上求值的广义极大极小体积这一凸化变体，首次完整刻画了对抗性噪声老虎机中函数类可学习的充要条件。 |
| [^324] | [STRIDE: Reasoning-Aware Training for Time Series Forecasting](https://arxiv.org/abs/2605.08625) | STRIDE提出了一种新框架，将蒸馏后的LLM推理以连续跨模态先验的形式原生注入时间序列基础模型的嵌入空间，在保持数值预测能力的同时引入推理能力，并登上GIFT-Eval排行榜第一。 |
| [^325] | [Training-Induced Escape from Token Clustering in a Mean-Field Formulation of Transformers](https://arxiv.org/abs/2605.07772) | 该论文首次在均场Transformer理论中揭示了训练对推理动力学的重塑作用：仅训练一个$L^2$正则化的线性FFN，即可使token分布在早期层聚簇之后于最终层附近逃逸出聚簇状态，且一次梯度更新就足以产生这种逃逸。 |
| [^326] | [Coordination Matters: Process-Aware Evaluation of Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2605.06557) | 本文提出了一个过程感知的多智能体强化学习评估框架，通过干扰、协同任务完成、参与度和协调效率等诊断指标补充传统聚合性能指标，揭示了回报相近的策略在协调行为上可能存在显著差异。 |
| [^327] | [MOSAIC: Module Discovery via Sparse Additive Identifiable Causal Learning for Scientific Time Series](https://arxiv.org/abs/2605.05524) | MOSAIC 提出了一种稀疏时序 VAE 方法，将科学观测固有的语义可解释性传递到具有可辨识性保证的潜在空间中，从而在底层机制未知的科学时间序列中实现可解释的模块发现。 |
| [^328] | [Regime-Conditioned Evaluation in Multi-Context Bayesian Optimization](https://arxiv.org/abs/2605.04895) | 该论文揭示了迁移贝叶斯优化中采集函数的排名会随预算比例和先验质量等状态变量而逆转（98% 的已发表比较从未控制这一变量），并提出可用先验信息估计的可移植状态得分 PRS=(B/|A|)(1-ρ) 来预测这种排名转变，从而主张以状态条件化评估取代平均效应比较。 |
| [^329] | [Learning reveals invisible structure in low-rank RNNs](https://arxiv.org/abs/2605.04115) | 本文推导出低维常微分方程系统来精确描述低秩RNN的学习动力学，并发现学习过程可分为决定网络功能的“损失可见重叠”和不影响功能但描述学习所必需的“损失不可见重叠”。 |
| [^330] | [Measuring Differences between Conditional Distributions using Kernel Embeddings](https://arxiv.org/abs/2605.02260) | 本文提出了条件最大均值差异（CMMD）这一统一理论框架，通过不同层级的核嵌入度量方法，系统性地解决了条件分布之间差异度量缺乏统一理论处理的问题。 |
| [^331] | [RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy](https://arxiv.org/abs/2605.02003) | 本文提出了RamanBench——首个大规模、完全可复现的拉曼光谱机器学习基准，统一整合了四个领域74个数据集共325,668条光谱，并在标准化协议下对28个模型进行了系统评估。 |
| [^332] | [Interpretable experiential learning based on state history and global feedback](https://arxiv.org/abs/2605.00940) | 提出了一种基于状态历史与全局反馈的可解释经验学习模型，通过带效用值和证据计数的状态转移图学习行为模型，适用于资源受限环境下的强化学习，并在Atari Breakout基准上取得了与神经网络方案相当的性能。 |
| [^333] | [Mesh Field Theory: Port-Hamiltonian Formulation of Mesh-Based Physics](https://arxiv.org/abs/2605.00394) | 该论文提出网格场理论及其神经实现，证明了基于网格的物理动力学在最小物理原理下可局部分解为端口哈密顿形式——其中保守互连由网格拓扑唯一确定、度量效应仅通过本构关系与耗散进入——并据此设计出结构保持的物理神经网络。 |
| [^334] | [CastFlow: Bridging Evidence Acquisition and Accurate Prediction in Agentic Time Series Forecasting](https://arxiv.org/abs/2604.27840) | CastFlow提出了一种角色专门化的智能体时间序列预测框架，通过经验记忆引导冻结的大语言模型调度多视角工具获取证据，再由轻量级专门化模型进行基于证据的时间推理，从而有效衔接证据获取与准确的数值预测。 |
| [^335] | [Leveraging Cell-level Spatial Transcriptomics as Weak Supervision for Nuclei Classification](https://arxiv.org/abs/2604.23481) | 该论文提出利用细胞级空间转录组学作为弱监督来源，自动生成图像-标签对以训练细胞核分类模型，在大幅减少人工标注的情况下达到了与传统监督方法相当的分割和分类性能。 |
| [^336] | [Activation-Efficient Fine-Tuning via Sequence-Compressed Low-Rank Subspaces](https://arxiv.org/abs/2604.22783) | LARS通过在序列级低秩子空间中执行适配，将存储的适配器激活中间量复杂度从O(BSRL)降至O(BRL)，相比LoRA在GPU和CPU上分别节省23%和47.8%的内存，且节省幅度随上下文长度增加而提升。 |
| [^337] | [From User Recognition to Activity Counting: An Identity-Agnostic Approach to Multi-User WiFi Sensing](https://arxiv.org/abs/2604.16572) | 该论文提出将多用户WiFi感知从“识别具体用户”重新表述为“身份无关的活动计数”，仅估计执行各活动的用户数量而无需识别个体，在WiMANS数据集上以更粗粒度的监督实现了更低的计数误差，并在跨环境条件和留出用户评估中展现出更好的泛化能力。 |
| [^338] | [Worst-Case Optimal $\sqrt{\epsilon}$-Rate Stability of KL Divergence under Perturbations of Gaussian Distributions](https://arxiv.org/abs/2604.11026) | 本文在比较分布 $P$ 不必为高斯分布的情况下，首次建立了高斯参考分布受扰动时KL散度变化的最坏情形最优 $\sqrt{\epsilon}$ 阶稳定性界，突破了现有松弛三角不等式要求三个分布均为高斯分布的局限，可应用于流生成模型的分布外检测等场景。 |
| [^339] | [Detecting Multi-Agent Collusion Through Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151) | 提出NARCBench基准与五种探针技术，首次利用模型内部表示在群体层面检测LLM多智能体系统中的串谋行为，并将其形式化为分布式异常检测问题。 |
| [^340] | [SpecXMaster Technical Report](https://arxiv.org/abs/2603.23101) | 该论文提出SpecXMaster，一个基于智能体强化学习的端到端智能框架，可直接从原始FID数据自动提取NMR光谱多重性信息并解读为化学结构，在多个公开NMR解读基准上表现卓越。 |
| [^341] | [M$^2$RNN: Non-Linear RNNs with Matrix-Valued States for Scalable Language Modeling](https://arxiv.org/abs/2603.14360) | 提出了一种具有矩阵值隐藏状态和非线性状态转移的新型循环神经网络M$^2$RNN，突破了Transformer在TC$^0$复杂度上的表达能力限制，实现了完美状态跟踪泛化并在大规模语言建模中表现优异。 |
| [^342] | [Adaptive Conditional Residual Sampling for Spectral Risk Optimisation under Decision-Dependent Uncertainty](https://arxiv.org/abs/2603.12507) | 提出自适应条件残差抽样（ACRS）方法，通过复用标准化残差向量构建确定性代理模型，高效求解决策依赖不确定性下的谱风险优化问题，并提供基于Wasserstein误差的最优性保证。 |
| [^343] | [Structural Causal Bottleneck Models](https://arxiv.org/abs/2603.08682) | 该论文提出结构因果瓶颈模型（SCBMs），用低维“瓶颈”统计量传递高维变量间的因果效应，证明了瓶颈变量在双射意义下可从观测数据识别，并表明瓶颈表示能够恢复工具变量方法的可识别性、并支持中介分析与后门调整下的有效因果效应估计。 |
| [^344] | [AOI: Turning Failed Trajectories into Training Signals for Autonomous Cloud Diagnosis](https://arxiv.org/abs/2603.03378) | 该论文提出AOI可训练多智能体框架，通过GRPO将专家知识蒸馏到本地开源模型并结合读写分离的安全执行架构，把失败运维轨迹转化为训练信号，在安全约束下实现自主云诊断。 |
| [^345] | [Learning Agentic Time Series Forecasting via Trajectory Distillation and Memory-Based Reinforcement Learning](https://arxiv.org/abs/2602.13802) | 本文提出 Cast-R1，一个通过教师模型轨迹蒸馏与带结构化记忆的多轮强化学习（GRPO）训练的 8B 参数智能体时间序列预测模型，证明较小的语言模型也能学会智能体式的工具交互与时序推理能力。 |
| [^346] | [A Theoretical Analysis of Test-Driven Code Generation](https://arxiv.org/abs/2602.06098) | 本文建立了测试驱动代码生成的概率理论框架，证明了基于模糊功能相似性的代码选择估计器在信噪比上严格优于基于功能等价性的估计器，并将回溯提示形式化为汤普森采样的上下文内近似，推导出新的遗憾界以解释其有效性受限于歧义性的原因。 |
| [^347] | [AutoPrunedRetriever: Persistent Reasoning Graphs for Efficient and Extensible Retrieval-Augmented Generation](https://arxiv.org/abs/2602.04926) | 提出了AutoPrunedRetriever（APR），一种基于图的RAG系统，通过共享符号码本将语料库事实与查询推理过程编码为既充当检索索引又充当持久化记忆的紧凑表示，使历史证据和推理结果可被高效复用，从而提升复杂推理任务下的效率与可扩展性。 |
| [^348] | [Pro-AI Bias in Large Language Models](https://arxiv.org/abs/2601.13749) | 该研究发现大型语言模型存在“亲AI”偏向：它们在建议中更频繁且更靠前地推荐AI，会将AI相关职位的薪资高估约10%，并对AI专业证书授予数量的高估程度比匹配的计算机专业高出28%。 |
| [^349] | [A New Family of Poisson Non-negative Matrix Factorization Methods Using the Shifted Log Link](https://arxiv.org/abs/2601.05845) | 该论文提出了带平移对数链接函数的泊松非负矩阵分解新方法族，通过单一调节参数使“部分”的组合方式从加法平滑过渡到乘法，并提供了高效的最大似然算法及适用于大型稀疏数据集的快速近似方法。 |
| [^350] | [AceFF: A State-of-the-Art Machine Learning Potential for Small Molecules](https://arxiv.org/abs/2601.00581) | AceFF是一个为小分子药物发现优化的预训练机器学习原子间势，通过改进TensorNet2架构并在类药物化合物综合数据集上训练，实现了高通量推理速度与DFT级精度的平衡，并支持带电状态和多种药物化学关键元素。 |
| [^351] | [CoatFusion: Controllable Material Coating in Images](https://arxiv.org/abs/2512.02143) | 提出了一种新的“材质涂覆”图像编辑任务及CoatFusion架构，通过扩散模型结合2D反照率纹理和PBR风格参数控制，实现了保留物体几何细节的逼真且可控的材质涂覆效果。 |
| [^352] | [Accelerated Execution of Bayesian Neural Networks using a Single Probabilistic Forward Pass and Code Generation](https://arxiv.org/abs/2511.23440) | 该论文通过假设权重和激活服从高斯分布，用单次解析式的概率前向传播取代多次采样前向传播来加速贝叶斯神经网络，并构建了在嵌入式ARM CPU上训练、编译、优化和部署的端到端流水线。 |
| [^353] | [Operon: Incremental Construction of Ragged Data via Named Dimensions](https://arxiv.org/abs/2511.16080) | Operon是一个基于Rust的工作流引擎，通过命名维度与显式依赖关系的形式化方法，结合静态验证的领域特定语言和增量式的动态任务调度，实现了对不规则数据形状与依赖关系的原生追踪与管理。 |
| [^354] | [Robust Experimental Design via Generalised Bayesian Inference](https://arxiv.org/abs/2511.07671) | 本文提出广义贝叶斯最优实验设计（GBOED），通过用损失函数替代似然函数，在模型设定存在偏差时同时保证实验设计与统计推断的鲁棒性。 |
| [^355] | [Rep2Text: Decoding Full Text from a Single LLM Token Representation](https://arxiv.org/abs/2511.06571) | 提出Rep2Text方法，证明仅凭LLM的单个隐藏表示即可重建原始文本（短文本可恢复约一半单词及整体含义），揭示了隐藏表示蕴含的丰富信息及其潜在的隐私泄露风险。 |
| [^356] | [Empirical Bayesian Multi-Bandit Learning](https://arxiv.org/abs/2510.26284) | 本文提出了一种新颖的分层贝叶斯多老虎机学习框架，通过经验贝叶斯方法估计先验分布的协方差矩阵来捕捉老虎机实例间的相关性与异质性，并据此开发了ebmTS等高效算法。 |
| [^357] | [Energy Overhead in Sensor Nodes: Embedded Machine Learning and Data Offloading via Low Power Wide Area Networks](https://arxiv.org/abs/2510.24829) | 本研究以ESP32-S3为平台，对传感器节点从数据采集、设备端CNN推理到LPWAN无线传输的完整流程进行了系统能耗分析，通过比较本地分类后传输结果与直接卸载原始数据两种方案，揭示了传感器节点能耗的主要驱动因素。 |
| [^358] | [Reliable data clustering with Bayesian community detection](https://arxiv.org/abs/2510.15013) | 该论文提出一种一步式聚类方法，利用贝叶斯社区检测和单一最小描述长度准则同时完成数据稀疏化与聚类，免除了任意阈值选择，在更高噪声和更少样本条件下优于传统两步法，且在无真实结构时能正确返回单一聚类。 |
| [^359] | [Language steering in latent space to mitigate unintended code-switching](https://arxiv.org/abs/2510.13849) | 该论文提出一种通过对平行翻译做主成分分析来识别语言方向、并在潜空间中引导词元嵌入的轻量级推理时方法，可将多语言大模型的无意语码切换指数降低63%至99%，且几乎不增加计算开销、仅需极少平行数据校准。 |
| [^360] | [Board gender diversity and emissions performance: Insights from panel regressions, machine learning, and explainable AI](https://arxiv.org/abs/2510.00244) | 该研究结合面板回归、机器学习与可解释人工智能发现，欧洲企业董事会性别多元化与排放绩效呈非线性关系，约35%为最优比例、22%为必要阈值，且ESG争议不削弱该关系，说明其作用源于治理机制而非象征性行为。 |
| [^361] | [Practical and Private Hybrid ML Inference with Fully Homomorphic Encryption](https://arxiv.org/abs/2509.01253) | 提出Safhire混合推理框架，通过在服务器端加密执行线性层、将非线性计算卸载至客户端，从而消除自举操作并显著降低计算成本，同时利用随机化洗牌技术保护模型机密性，实现实用且隐私安全的全同态加密机器学习推理。 |
| [^362] | [The Statistical Fairness-Accuracy Frontier](https://arxiv.org/abs/2508.17622) | 该论文在有限样本条件下研究了单一预测模型服务多个人口群体时的公平性-准确性权衡，推导了最坏情况最优估计量，刻画了有限样本对各群体福利的不对称影响，给出了最优样本分配策略，并提供了沿整个FA帕累托前沿的一致有限样本界。 |
| [^363] | [Neural Operators Which Certifiably Simultaneously Solve Infinitely Many Convex Programs using Logarithmic Rank and Linear Complexity](https://arxiv.org/abs/2508.14995) | 本文提出一种简短而简单的训练算法来构造特定类别的神经算子，可认证地同时求解无穷多个凸规划问题，且仅需对数秩与线性复杂度，从而弥合了神经算子理论与实践之间的差距。 |
| [^364] | [Reinforcement Learning in MDPs with Information-Ordered Policies](https://arxiv.org/abs/2508.03904) | 本文提出策略类上的“信息序”概念，通过反事实推断为连续状态动作空间的无限时域平均成本MDP设计了基于epoch的强化学习算法，取得了由信息序宽度 $w$ 决定的 $\widetilde O(H\sqrt{wdT/\alpha})$ 遗憾界，从而利用策略间数据的可迁移性降低了探索复杂度。 |
| [^365] | [RCPN: towards robust pseudo-labels in unsupervised CLIP adaptation via consistent prototypes and neighborhoods](https://arxiv.org/abs/2507.22075) | 提出自适应伪标签框架RCPN，通过原型一致性评估（PICS）、邻域语义修正（NALR）及自适应加权机制，为无监督CLIP适配生成更鲁棒的伪标签。 |
| [^366] | [Generative molecule evolution using 3D pharmacophore for efficient Structure-Based Drug Design](https://arxiv.org/abs/2507.20130) | 提出了MEVO框架，通过反应感知的生成模型在可合成化学空间中进行基于三维药效团的分子进化，同时解决了基于结构药物设计中蛋白质-配体复合物数据稀缺和生成分子难以合成的两大难题。 |
| [^367] | [Confidence Calibration in Vision-Language-Action Models](https://arxiv.org/abs/2507.17383) | 本文首次系统研究了视觉-语言-动作（VLA）基础模型的置信度校准问题，建立了置信度估计基线，并提出提示集成与动作级Platt缩放两种轻量级技术来改善校准，从而通过可靠的不确定性量化提升机器人行为的可信赖性。 |
| [^368] | [Missing value imputation with adversarial random forests - MissARF](https://arxiv.org/abs/2507.15681) | MissARF是一种基于对抗性随机森林的快速缺失值插补方法，可在提供高质量单重与多重插补的同时，使多重插补无需额外计算成本。 |
| [^369] | [A Survey of Explainable Reinforcement Learning: Targets, Methods and Needs](https://arxiv.org/abs/2507.12599) | 本文提出了基于“解释什么”和“如何解释”两个维度的可解释强化学习（XRL）直观分类法，并对250多篇论文进行了系统性综述，同时指出了值得关注的相关领域与社区需求。 |
| [^370] | [NeuralDMD: Interpretable Neural Representation of Dynamics from Sparse and Noisy Measurements](https://arxiv.org/abs/2507.03094) | 提出NeuralDMD框架，将神经隐式表示与动态模态分解（DMD）相结合，从稀疏含噪测量中重构出可解释的连续时空动力学，无需真值数据或模拟器，并在天气数据同化等多种任务上优于现有基线方法。 |
| [^371] | [GPT, But Backwards: Exactly Inverting Language Model Outputs](https://arxiv.org/abs/2507.01693) | 该论文提出SODA算法，能够从语言模型的输出logits中精确反推出输入文本，实现多达17个token的零误报重建，且逆转难度主要取决于输入长度和词汇表大小而非参数量，并可应用于恢复后门触发器和个人身份信息等安全场景。 |
| [^372] | [Near-optimal estimates for the $\ell^p$-Lipschitz constants of deep random ReLU neural networks](https://arxiv.org/abs/2506.19695) | 本文证明了遵循He初始化变体的随机ReLU神经网络的 $\ell^p$-Lipschitz 常数的近最优估计，揭示了该常数在 $p \in [2,\infty]$ 时类似于高斯向量的 $\ell^{p'}$ 范数，而在 $p \in [1,2)$ 时则更接近高斯向量的 $\ell^2$ 范数。 |
| [^373] | [RAW-Explainer: Post-hoc Explanations of Path Reasoning Graph Neural Networks for Knowledge Graph Completion](https://arxiv.org/abs/2506.12558) | 本文提出RAW-Explainer，一种专为路径推理图神经网络设计的新型可解释性方法，通过随机游走目标函数和异构信息，为知识图谱链接预测生成连通、简洁且可解释的子图解释，并解决了评估小子图时的分布偏移问题。 |
| [^374] | [GARG-AML against Smurfing: A Scalable and Interpretable Graph-Based Framework for Anti-Money Laundering](https://arxiv.org/abs/2506.04292) | 本文提出GARG-AML框架，通过从二阶交易网络结构中提取的单一可解释指标量化分拆洗钱风险，在保持可扩展性的同时兼顾了可解释性，克服了黑盒网络嵌入方法在实际银行业务中的局限。 |
| [^375] | [GRIT: Out-of-Distribution Robustness from Finite and Noisy Invariant Pairs](https://arxiv.org/abs/2505.24843) | 提出GRIT方法，通过在预训练特征中从有限且含噪的不变样本对差异协方差估计不稳定子空间并予以投影去除，从而在样本对不完美的情况下仍能提升模型的分布外鲁棒性。 |
| [^376] | [Mask-Based Deep Symbolic Regression](https://arxiv.org/abs/2505.24776) | 提出了一种基于随机掩码扩散过程生成多样高质量方程、并结合逐token的GRPO强化学习与长短期风险寻求策略的深度符号回归新方法。 |
| [^377] | [TRIP-Shapley: Propagation-Aware Contribution Attribution Framework for Peer-to-Peer Federated Learning](https://arxiv.org/abs/2505.23246) | TRIP-Shapley提出了一种面向点对点联邦学习的传播感知贡献归因框架，通过独立计算每轮贡献分数并仅传播标量贡献值，避免了多跳路径追踪带来的通信爆炸，并能抵御虚假报告攻击。 |
| [^378] | [Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment](https://arxiv.org/abs/2505.10597) | 提出协作式奖励建模（CRM）框架，通过两个奖励模型跨模型互评结合课程学习来识别可靠的偏好数据，从而有效缓解噪声偏好导致的奖励误泛化并提升大语言模型的对齐效果。 |
| [^379] | [Latent Evolution Operator (LEO): State-to-State Forecasting of Parametric PDEs](https://arxiv.org/abs/2505.09063) | 本文提出潜在演化算子（LEO），一种能以任意时刻状态、物理参数和预测时间跨度为输入、在单次潜在空间评估中直接预测参数化偏微分方程未来状态的参数化映射，从而避免了逐步积分带来的误差累积，并实现了从中间状态出发的泛化预测。 |
| [^380] | [GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](https://arxiv.org/abs/2504.19599) | 本文提出组方差策略优化（GVPO），通过将KL约束奖励最大化的解析解直接融入梯度权重，解决了GRPO等后训练方法训练不稳定的问题，并保证唯一的最优解。 |
| [^381] | [Less is More: Efficient Black-box Attribution via Minimal Interpretable Subset Selection](https://arxiv.org/abs/2504.00470) | 提出LiMA方法，将黑盒归因形式化为有序可解释子集选择问题，借助次模优化的边际增益原理和综合评估语义一致性、协作性、预测置信度与语义有效性的复合目标，高效识别对模型决策最关键的输入区域。 |
| [^382] | [LeanTTA: A Backpropagation-Free and Stateless Approach to Quantized Test-Time Adaptation on Edge Devices](https://arxiv.org/abs/2503.15889) | 本文提出LeanTTA框架，通过无反向传播、无状态地动态更新归一化统计量，实现了面向边缘设备的量化测试时自适应，显著降低内存与计算开销且不依赖大批量或历史数据。 |
| [^383] | [Gabor-Enhanced Physics-Informed Neural Networks for Fast Simulations of Acoustic Wavefields](https://arxiv.org/abs/2502.17134) | 提出一种引入显式可训练Gabor基函数的简化PINN框架，将网络任务重新定义为学习从输入坐标到自定义Gabor坐标系的非线性映射，有效克服了PINNs的低频偏差问题，实现高频声波场的快速高精度模拟。 |
| [^384] | [What is causal about causal models and representations?](https://arxiv.org/abs/2501.19335) | 该论文提出了一个将因果模型与数据生成过程分离的形式化框架，形式化了连接行动与干预的解释，并证明了不存在能同时满足一组期望性质且保持非循环性的行动-干预解释这一不可能性结果。 |
| [^385] | [Rethinking Negative Prompting for Concept Suppression: Unconditional Anchoring and Localized Guidance](https://arxiv.org/abs/2412.07658) | 提出无需训练的TraSCE方法，通过无条件锚定与局部化引导，解决负向提示在越狱攻击下因“引导崩塌”而失效的问题，从而有效抑制文生图模型生成有害或不良概念。 |
| [^386] | [Paint Outside the Box: Synthesizing and Selecting Training Data for Visual Grounding](https://arxiv.org/abs/2412.00684) | 提出 POBF 框架，通过在边界框外进行图像修复来合成训练数据以解决标签错位问题，并利用结合难度分数与过拟合分数的过滤机制筛选最有效的训练数据，从而在数据稀缺条件下显著提升视觉定位性能。 |
| [^387] | [Property Estimation in Geotechnical Databases Using Labeled Random Finite Sets](https://arxiv.org/abs/2410.22659) | 本文提出在标记随机有限集（LRFS）框架下，利用广义标记多伯努利（GLMB）滤波器从贝叶斯视角处理含噪声和不完整的岩土数据，实现了岩土数据库中的土性参数估计。 |
| [^388] | [Harnessing Trust in Directed Graphs: Redefining Robustness of Graph Learning](https://arxiv.org/abs/2306.02002) | 该论文从攻击和防御两个角度重新定义了有向图上图神经网络的鲁棒性，并提出了一种即插即用的消息传递框架，通过利用有向图链接方向中蕴含的信任信号显著提升GNN的鲁棒性。 |
| [^389] | [Is the Policy Gradient a Gradient?](https://arxiv.org/abs/1906.07073) | 本文证明了省略折扣因子的策略梯度更新不必是任何函数的梯度，并构造反例表明该更新会使策略收敛到在折扣与未折扣目标下均全局最差的确定性极限。 |
| [^390] | [UAV Pathfinding in Dynamic Obstacle Avoidance with Multi-agent Reinforcement Learning.](http://arxiv.org/abs/2310.16659) | 本文提出了一种基于多智能体强化学习的集中式训练和分布式执行方法，以在线解决动态障碍物避障问题。实验结果验证了该方法的有效性。 |
| [^391] | [Accelerated gradient methods for nonconvex optimization: Escape trajectories from strict saddle points and convergence to local minima.](http://arxiv.org/abs/2307.07030) | 本文研究了一类加速梯度方法在非凸优化问题上的行为，包括逃离鞍点和收敛到局部极小值点的分析。研究在渐进和非渐进情况下，提出了一类新的Nesterov类型的加速方法，并回答了Nesterov加速梯度方法是否避免了严格鞍点的问题。 |

# 详细

[^1]: 深度条件生成学习：模型与误差分析

    Deep Conditional Generative Learning: Model and Error Analysis

    [https://rss.arxiv.org/abs/2402.01460](https://rss.arxiv.org/abs/2402.01460)

    提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。

    

    我们介绍了一种基于普通微分方程（ODE）的深度生成方法，用于学习条件分布，称为条件Follmer流。从标准高斯分布开始，所提出的流能够以高效的方式将其转化为目标条件分布，在时间1处达到稳定。为了有效实现，我们使用欧拉方法对流进行离散化，使用深度神经网络非参数化估计速度场。此外，我们导出了学习样本的分布与目标分布之间的Wasserstein距离的非渐近收敛速率，在条件分布学习中提供了第一个全面的端到端误差分析。我们的数值实验展示了它在一系列情况下的有效性，从标准的非参数化条件密度估计问题到涉及图像数据的更复杂的挑战，说明它优于各种现有方法。

    We introduce an Ordinary Differential Equation (ODE) based deep generative method for learning a conditional distribution, named the Conditional Follmer Flow. Starting from a standard Gaussian distribution, the proposed flow could efficiently transform it into the target conditional distribution at time 1. For effective implementation, we discretize the flow with Euler's method where we estimate the velocity field nonparametrically using a deep neural network. Furthermore, we derive a non-asymptotic convergence rate in the Wasserstein distance between the distribution of the learned samples and the target distribution, providing the first comprehensive end-to-end error analysis for conditional distribution learning via ODE flow. Our numerical experiments showcase its effectiveness across a range of scenarios, from standard nonparametric conditional density estimation problems to more intricate challenges involving image data, illustrating its superiority over various existing condition
    
[^2]: 行为耦合：从形成到语言模型中的行为涌现

    Behavioral Couplings: From Formation to Behavioral Emergence in Language Models

    [https://arxiv.org/abs/2610.23847](https://arxiv.org/abs/2610.23847)

    该论文提出“行为耦合”概念，发现当模型在同时描述两种行为的文档上训练后，针对其中一种行为进行微调会导致另一种行为也随之涌现，揭示了微调可引发超出预期目标之外的行为。

    

    微调大型语言模型（LLM）可能会引发超出明确目标范围的行为。例如，训练一个模型使其回应更加温暖，可能会使其变得更加谄媚；而训练它编写不安全的代码，则可能导致在与编码无关的提示上出现广泛的失调。我们将这种行为之间的关系——即对某一种行为的微调会引发另一种行为——称为“行为耦合”。在这项工作中，我们研究了行为耦合的形成及其特性。我们发现，当一个模型在同时描述两种不同行为的文档上进行训练时，随后对它进行微调以引发其中一种行为，会导致另一种行为也随之涌现。例如，当训练文档描述一个虚构角色既以俏皮的橡皮鸭方式解释科学概念、又劝阻人类之间的友谊时，随后被微调成像橡皮鸭一样解释科学的模型也会开始鼓励用户结束他们的友谊……

    arXiv:2610.23847v1 Announce Type: cross  Abstract: Fine-tuning large language models (LLMs) can induce behaviors beyond those explicitly targeted. For example, training a model to respond more warmly can make it more sycophantic, and training it to write insecure code can cause broad misalignment on prompts unrelated to coding. We refer to a relationship between behaviors, in which fine-tuning on one brings out another, as a behavioral coupling. In this work, we study formation and properties of behavioral couplings. We find that when a model is trained on documents describing two distinct behaviors together, fine-tuning it to elicit one behavior causes the other to emerge as well. For instance, when training documents describe a fictitious character that both explains scientific concepts in a playful, rubber-duck-like manner and discourages human friendships, a model subsequently fine-tuned to explain science like a rubber duck also begins encouraging users to end their friendships, w
    
[^3]: 参数化量子电路中任务对齐的谱归纳偏置

    Task-aligned spectral inductive biases in parameterized quantum circuits

    [https://arxiv.org/abs/2610.23834](https://arxiv.org/abs/2610.23834)

    该论文的核心贡献是揭示了数据重上传量子电路中编码固定谱查询可以通过调制现有信号频率的组合方式改变假设类而无需引入新的样本信息，从而为量子机器学习提供任务对齐的谱归纳偏置，并在合成孔径雷达启发的分类任务中验证了这一机制。

    

    参数化量子电路在量子机器学习中被广泛研究，但其架构如何提供与任务对齐的归纳偏置仍知之甚少。我们研究了具有两种不同输入的数据重上传量子电路：一种是随样本变化的信号，另一种是固定的、由任务定义的谱查询，后者规定了信号应如何被探测。我们证明，对查询进行编码可以通过调制电路中现有信号频率的组合方式来改变电路在信号上的假设类，而无需引入新的样本相关信息。这一机制与建立在非均匀傅里叶变换之上的任务相匹配，其中评估频率指定了查询，采样位置则指导查询的编码方式。我们在一个受合成孔径雷达启发的分类任务中检验了这一联系，其中采集几何结构决定了用于在图像形成之前对未聚焦频谱斑块进行分类的查询。

    arXiv:2610.23834v1 Announce Type: cross  Abstract: Parameterized quantum circuits are widely investigated in quantum machine learning, yet how their architecture can provide task-aligned inductive biases remains poorly understood. We study data reuploading quantum circuits with two distinct inputs: a signal that varies between samples and a fixed, task-defined spectral query that specifies how the signal should be probed. We show that encoding the query can change the circuit's hypothesis class over the signal by modulating how its existing signal frequencies are combined, without introducing new sample-dependent information. This mechanism matches tasks built on nonuniform Fourier transforms, whose evaluation frequencies specify the query and whose sampling positions guide its encoding. We examine this connection in a classification task inspired by synthetic aperture radar, where acquisition geometry determines a query for classifying unfocused spectral patches before image formation
    
[^4]: 超越答案：面向推理蒸馏的学生条件化推理轨迹构建

    Beyond the Answer: Student-Conditioned Trace Construction for Reasoning Distillation

    [https://arxiv.org/abs/2610.23831](https://arxiv.org/abs/2610.23831)

    该论文提出ReDiST框架，以学生自身尝试为起点构建、修订并筛选与学生能力匹配的推理轨迹用于蒸馏，从而实现超越仅答案监督的学习效果。

    

    推理轨迹可以伴随着正确答案一同出现，却不一定能针对特定学生提供有效的监督信号。我们研究了如何通过同时考虑推理轨迹的中间内容以及其对目标学习者的可及性，来构建超越仅答案监督、能真正改进学习的推理轨迹。我们提出了ReDiST，一个面向证据丰富推理的学生条件化轨迹构建与蒸馏框架。ReDiST从学生自身的尝试出发，依据所提供的证据修订答案正确的草稿，并借助经验证答案的额外指导来修复失败的草稿。随后，一个冻结的、与学生匹配的阅读器在给定问题、证据和候选轨迹的条件下，利用答案恢复能力和经验证答案的概率来筛选并排序原始及编辑后的候选轨迹。我们在蒸馏所选目标的同时，通过辅助答案损失来平衡轨迹监督与最终答案监督。轨迹构建与选择均在离线阶段完成，……

    arXiv:2610.23831v1 Announce Type: new  Abstract: A reasoning trace can accompany a correct answer without providing effective supervision for a particular student. We study how to construct traces that improve learning beyond answer-only supervision by considering both their intermediate content and their accessibility to the target learner. We introduce ReDiST, a student-conditioned trace-construction and distillation framework for evidence-rich reasoning. Starting from the student's own attempts, ReDiST revises answer-correct drafts against the supplied evidence and repairs unsuccessful drafts with additional guidance from the verified answer. A frozen student-matched reader screens and ranks original and edited candidates using answer recovery and verified-answer probability, given the question, evidence, and candidate trace. We distill the selected targets while balancing trace and final-answer supervision with an auxiliary answer loss. Construction and selection occur offline, and
    
[^5]: 等待恰好买到两个矩：预测驱动分配中的内生识别

    Waiting Buys Exactly Two Moments: Endogenous Identification in Prediction-Driven Allocation

    [https://arxiv.org/abs/2610.23830](https://arxiv.org/abs/2610.23830)

    该论文证明了在预测驱动的资源分配中，等待观测τ个周期能恰好识别2τ个潜在风险矩（每多等一期新增两个矩），但这种内生识别无法避免时机遗憾，且未必能判定唯一的最优干预日期。

    

    拥有固定预防性援助预算的规划者必须决定帮助谁以及何时帮助。等待能够揭示风险，但随着高风险个体率先离开，等待也会消除干预的机会。我们在重复噪声信号和风险依赖流失的设定下刻画了这一权衡：在规范模型中，长度为 $\tau$ 的观测前缀恰好能识别潜在风险矩 $m_j=\mathbb E[p^j]$（$j=0,\ldots,2\tau-1$），即每多观察一个周期就能“买到”两个新的矩。这些矩可以识别幸存者中下一周期的平均流失率，但未必能识别分配福利或被优先治疗群体的流失率。我们通过观测等价的构造证明：在固定观测长度下，无论样本量多大，时机上的遗憾都不可避免；而精确证书则可判定所有相容总体是否共享唯一的最优干预日期。匹配实验分离了估计、自助法平均和遗憾加权各自的影响。

    arXiv:2610.23830v1 Announce Type: new  Abstract: A planner with a fixed budget of preventive help must decide whom to help and when. Waiting reveals risk but removes intervention opportunities as higher-risk individuals leave first. We characterize this tradeoff under repeated noisy signals and risk-dependent attrition: in the canonical model, a length-$\tau$ observation prefix identifies exactly the latent risk moments $m_j=\mathbb E[p^j]$ for $j=0,\ldots,2\tau-1$, buying two new moments per additional period. These identify average next-period attrition among survivors but need not identify allocation welfare or attrition among those prioritized for treatment. Our observationally equivalent constructions establish unavoidable timing regret at any sample size for a fixed observation length, while exact certificates determine whether all compatible populations share a unique optimal date. Matched experiments separate the effects of estimation, bootstrap averaging, and regret weighting.
    
[^6]: 量子网络中的传输质量估计：计算、仿真还是学习？

    Quality-of-Transmission Estimation in Quantum Networks: Compute, Simulate, or Learn?

    [https://arxiv.org/abs/2610.23827](https://arxiv.org/abs/2610.23827)

    本文受光网络QoT估计经验启发，提出了一个量子网络传输质量估计的“计算/仿真/学习”决策框架，用于指导量子控制器在何种情况下应解析计算、仿真或通过学习获得端到端纠缠保真度。

    

    未来的量子网络将依赖于控制平面的各项决策，如路由、调度、纠缠供给、纯化和内存分配。每一项决策都依赖于对候选路径所能提供的纠缠质量的及时且可靠的估计，其中端到端保真度是核心指标。受光网络中传输质量（QoT）估计经验的启发，本文探讨了量子控制器何时应该通过解析方法计算保真度、通过仿真获得保真度，还是通过学习建立代理模型。这种光网络的类比是有用的但不完整的：光QoT估计通常预测一个标量传输裕度，而量子QoT估计必须考虑概率性的纠缠生成、随机的内存等待时间、退相干、量子态消耗以及测量约束等因素，这些因素会使保真度成为一个随机目标。我们提出了一个用于量子QoT估计的计算/仿真/学习框架。

    arXiv:2610.23827v1 Announce Type: cross  Abstract: Future quantum networks will rely on control-plane decisions such as routing, scheduling, entanglement provisioning, purification, and memory allocation. Each decision depends on a timely and reliable estimate of the quality of the entanglement that a candidate path can deliver, with end-to-end fidelity as a central metric. Inspired by the optical-networking experience with quality-of-transmission (QoT) estimation, this article asks when a quantum controller should compute fidelity analytically, simulate it, or learn a surrogate model. The optical analogy is useful but incomplete: optical QoT estimation often predicts a scalar transmission margin, whereas quantum QoT estimation must account for probabilistic entanglement generation, random memory waiting times, decoherence, state consumption, and measurement constraints that can make fidelity a stochastic target. We propose a compute/simulate/learn framework for quantum QoT estimation 
    
[^7]: ENFR：基于流匹配参考的证据网络用于摊销式基于仿真的模型推断

    ENFR: Evidence Networks with Flow-Matched References for Amortized Simulation-Based Model Inference

    [https://arxiv.org/abs/2610.23825](https://arxiv.org/abs/2610.23825)

    本文提出ENFR方法，通过结合证据网络与流匹配参考技术，实现对贝叶斯证据的准确、快速且摊销式的估计，从而突破仅衡量证据比值的贝叶斯因子方法的局限，为高维拟合优度检验和模型误设检验提供直接解决方案。

    

    证据 $\mathcal{Z} = p(x | \mathcal{M})$，即数据生成模型 $\mathcal{M}$ 的证据，是贝叶斯框架中的一个强大工具：它能够实现模型推断、模型误设检验和拟合优度评估，可作为标准差异统计量（如 $\chi^2$）的替代方案。尽管用途广泛，证据的计算却是出了名的困难，即使似然函数可用时也是如此。无论是在基于似然的推断还是基于仿真的推断中，估计 $\mathcal{Z}$ 都能为高维拟合优度检验或模型误设检验提供直接的解决方案：使用 $\mathcal{Z}(x)$ 作为检验统计量可诱导出最小体积的 $\alpha$ 水平检验。此类检验对于评估模型保真度至关重要，而模型保真度是信任基于仿真的参数推断的必要条件。现有的学习方法主要针对贝叶斯因子，而贝叶斯因子只能衡量证据的比值。与之相反，我们提供了对……的准确、快速且摊销式的估计（原文摘要在此处截断）

    arXiv:2610.23825v1 Announce Type: cross  Abstract: The evidence, $\mathcal{Z} = p(x | \mathcal{M})$, of a data-generating model $\mathcal{M}$ is a powerful tool in Bayesian frameworks: it enables model inference, model misspecification tests, and goodness-of-fit assessment as an alternative to standard discrepancy statistics, e.g. $\chi^2$. While versatile, the evidence is notoriously difficult to compute, even when the likelihood is available. In both likelihood-based and simulation-based inference, estimating $\mathcal{Z}$ provides an immediate solution to high-dimensional goodness-of-fit or model misspecification tests: using $\mathcal{Z}(x)$ as the test statistic induces a minimum-volume level-$\alpha$ test. Such tests are essential to assess model fidelity, which is necessary to trust simulation-based parameter inference. Existing learned approaches largely target Bayes factors, which only measure evidence ratios. Instead, we provide accurate, fast, and amortized estimates of the 
    
[^8]: 无需重新学习物理对称性：球面上的SO(3)等变卷积神经网络（CNN）

    Learning without relearning physical symmetry: an SO(3)-equivariant convolutional network (CNN) on the sphere

    [https://arxiv.org/abs/2610.23821](https://arxiv.org/abs/2610.23821)

    本文提出了一种专为HEALPix球面像素化设计的SO(3)等变卷积神经网络，利用可操纵滤波器、平行移动和自旋加权特征实现可证明的精确等变性，使网络无需消耗训练数据来学习物理对称性。

    

    许多物理科学中最重要的数据集天然存在于球面上，且这些数据通常不带有优先的位置或取向。使用不反映这种对称性的方法会导致高得多的模拟需求，这对科学分析而言可能是灾难性的（迫使人们依赖廉价且不完善的模型，从而使结果产生偏差）。我们提出了一种新颖的球面规范等变网格架构实现，该架构专为球面的层次等面积等纬度像素化（HEALPix）而设计。SO(3)等变卷积由可操纵滤波器、平行移动和自旋加权特征构建而成。该架构直接提供了精确的对称性，因此网络无需花费训练数据去学习它。在给定频带限制的条件下，卷积运算被构建为具有可证明的精确等变性。我们的完整CNN可以将SO(3)不变性误差控制在……（原文在此截断）

    arXiv:2610.23821v1 Announce Type: cross  Abstract: Many of the most important datasets in the physical sciences live natively on the sphere, and these data often carry no preferred position or orientation. Using methods that do not reflect this symmetry leads to far higher simulation requirements, which can be catastrophic for scientific analyses (forcing reliance on cheap, imperfect models that bias results). We present a novel spherical implementation of a gauge equivariant mesh architecture that is specifically designed for the Hierarchical Equal Area isoLatitude Pixelization (\healpix{}) of the sphere. The $\mathrm{SO}(3)$-equivariant convolution is built from steerable filters, parallel transport and spin-weighted features. The architecture supplies the exact symmetry, so the network does not spend training data learning it. The convolution operations are constructed to give provably exact equivariance, given a band-limit. Our full CNNs can achieve SO(3)-invariance error at per-ce
    
[^9]: CocoRNA：基于协作多智能体强化学习的集体RNA设计

    CocoRNA: Collective RNA Design with Cooperative Multi-agent Reinforcement Learning

    [https://arxiv.org/abs/2610.23820](https://arxiv.org/abs/2610.23820)

    CocoRNA是一个协作多智能体强化学习框架，通过将RNA逆向设计分解为多个子问题并由多个智能体协作求解，在成功率和设计效率上显著超越现有最先进方法，且无需额外训练即可泛化到未见结构。

    

    设计能够可靠折叠成特定二级结构的RNA序列对于理解其生物功能至关重要，但这仍然是一个具有挑战性的计算问题。我们提出了CocoRNA，一个用于RNA逆向设计的协作多智能体强化学习框架。CocoRNA通过将设计任务分解为更小的子问题来简化设计任务，每个子问题由多个智能体协作解决。这种方法降低了问题的复杂性，并改善了设计策略的探索。在训练过程中，集中式评论家利用全局结构信息来指导智能体，使它们能够联合优化各自的设计策略。因此，CocoRNA学习到了高质量的RNA设计策略，无需额外训练即可有效泛化到未见过的结构。在Rfam数据集上的实验表明，CocoRNA在成功率和设计效率方面均显著优于现有最先进的方法。

    arXiv:2610.23820v1 Announce Type: new  Abstract: Designing RNA sequences that reliably fold into specific secondary structures is essential for understanding their biological functions but remains a challenging computational problem. We propose CocoRNA, a cooperative multi-agent reinforcement learning framework for RNA inverse design. CocoRNA simplifies the design task by decomposing it into smaller sub-problems, each solved collaboratively by multiple agents. This approach reduces the complexity of the problem and improves the exploration of design policies. During training, a centralized critic uses global structural information to guide the agents, enabling them to jointly optimize their design strategies. As a result, CocoRNA learns high-quality RNA design policies that generalize effectively to unseen structures without additional training. Experiments on the Rfam dataset demonstrate that CocoRNA substantially outperforms state-of-the-art methods in both success rate and design sp
    
[^10]: 从延迟单边反馈中学习个性化记忆门控

    Learning Personalized Memory Gating from Delayed One-Sided Feedback

    [https://arxiv.org/abs/2610.23812](https://arxiv.org/abs/2610.23812)

    该论文将事实级别的个性化记忆门控形式化为带延迟单边反馈的上下文苹果品尝问题，提出结合离线学习与在线自适应的Warm-Start UCB算法，并给出了高概率遗憾上界的理论保证。

    

    长期记忆使语言模型智能体能够个性化未来的交互，但有限的记忆库无法保存每一个提取出的事实。现有的记忆门控策略通常在所有用户之间共享，因此无法考虑用户特定的未来复用模式。在本文中，我们从事实层面研究记忆门控问题。我们首先提出一种基于检索的标注策略，通过某个事实是否在之后被检索并融入回复来为每个事实打标签。然后，我们将在线记忆门控形式化为具有延迟单边反馈的上下文苹果品尝问题，并提出Warm-Start UCB算法，该算法将基于现有数据的离线学习与针对当前用户的在线自适应相结合。我们的理论分析建立了当延迟窗口至多为O(√T)时，以高概率成立的遗憾上界 $\tilde{O}(T\epsilon \wedge d\sqrt{T})$，其中T是总轮数，ε是初始化误差。

    arXiv:2610.23812v1 Announce Type: new  Abstract: Long-term memory enables language-model agents to personalize future interactions, but a finite memory bank cannot preserve every extracted fact. Existing memory gating policies are typically shared across users and therefore fail to account for user-specific patterns of future reuse. In this paper, we study memory gating at the fact level. We first propose a retrieval-based labeling strategy that labels each fact by whether it is later retrieved and incorporated into responses. We then formulate online memory gating as contextual apple tasting with delayed one-sided feedback, and propose Warm-Start UCB, which combines offline learning from existing data with online adaptation to the current user. Our theoretical analysis establishes a high-probability regret upper bound of $\tilde{O}(T\epsilon \wedge d\sqrt{T})$ when the delay window is at most $O(\sqrt{T})$, where $T$ is the total number of rounds, $\epsilon$ is the initialization erro
    
[^11]: 纤维增强高性能混凝土力学强度温度效应的可解释机器学习分析

    Explainable Machine Learning Analysis of Temperature Effects on Mechanical Strength of Fiber-Reinforced High-Performance Concrete

    [https://arxiv.org/abs/2610.23811](https://arxiv.org/abs/2610.23811)

    本研究基于1,204条实验记录构建了可解释的机器学习框架，利用梯度提升与极端梯度提升模型（R²达0.903）准确预测温度作用下纤维增强高性能混凝土的抗压强度，并结合SHAP方法揭示各因素对强度的影响机制。

    

    由于混合料组成、纤维特性、养护条件与温度之间复杂的相互依赖关系，预测经受温度作用的纤维增强高性能混凝土的抗压强度极具挑战性。本研究利用包含多样材料设计与温度暴露环境的1,204条实验记录，构建了预测抗压强度的数据驱动框架。研究评估了八种机器学习模型，并证明集成方法通常优于单一学习器。梯度提升和极端梯度提升在未见数据上均表现出优越性能，两者均达到了0.903的R²。其中，梯度提升产生了最小的均方根误差（6.67 MPa），而极端梯度提升则实现了最低的平均绝对误差（3.97 MPa）和平均绝对百分比误差（10.60%）。此外，研究进一步利用Shapley加性解释方法对模型进行解释…

    arXiv:2610.23811v1 Announce Type: new  Abstract: Predicting the compressive strength of fiber-reinforced high-performance concrete subjected to temperature exposure is challenging because to the intricate interdependencies of mixture composition, fiber properties, curing conditions, and temperature. A data-driven framework to predict compressive strength utilizing 1,204 experimental records containing a diverse array of material designs and exposure situations. This study evaluates eight machine-learning models and demonstrate that ensemble approaches routinely surpass individual learners. Both gradient boosting and extreme gradient boosting exhibit superior performance on unseen data, each attaining a R^2 of 0.903. Gradient boosting produces the minimal root mean square error (6.67 MPa), whereas extreme gradient boosting achieves the lowest mean absolute error (3.97 MPa) and mean absolute percentage error (10.60%). The models are further interpreted utilizing Shapley additive explanat
    
[^12]: 面向任务的核Stein证书用于Rényi差分隐私

    Task-aware Kernel Stein certificates for R\'{e}nyi differential privacy

    [https://arxiv.org/abs/2610.23798](https://arxiv.org/abs/2610.23798)

    本文提出基于护航分布的核Stein差异方法，通过设计聚焦相关隐私方向的核函数，为Rényi差分隐私提供可计算的隐私证书及面向任务的隐私保证，并给出了高斯与拉普拉斯机制的精确恒等式和指数机制的界。

    

    基于Rényi散度的Rényi差分隐私已成为隐私保证领域的重要概念。然而，在高斯设定之外，Rényi散度可能难以计算。特别是，利用任务特定信息的Rényi差分隐私保证（如Rényi Pufferfish隐私中的情形）可能具有挑战性。本文使用基于护航分布的核Stein差异来为Rényi差分隐私提供证书。通过设计核函数聚焦于相关的隐私方向，我们还推导出了面向任务的隐私证书。我们为高斯机制和拉普拉斯机制提供了精确恒等式，并为指数机制提供了界，同时给出了独立的面向任务保证。一个真实数据集的例子说明了受保护的比较集和发布的子空间如何共同决定所得的证书。

    arXiv:2610.23798v1 Announce Type: cross  Abstract: R\`enyi differential privacy, based on R\`enyi divergence, has become a notion of choice for privacy guarantees. However, outside the Gaussian setting, R\`enyi divergence can be difficult to compute. In particular, R\`enyi differential privacy guarantees that exploit task-specific information, as in R\`enyi Pufferfish privacy, can be challenging. Here we use kernel Stein discrepancies based on escort distributions to provide certificates for R\`enyi differential privacy. By designing the kernel to focus on the relevant privacy directions, we also derive task-aware privacy certificates. We provide exact identities for Gaussian and Laplace mechanisms and bounds for exponential mechanisms, with separate task-aware guarantees. A real data set example illustrates how the protected comparison set and the released subspace jointly determine the resulting certificates.
    
[^13]: 缓慢非平稳约束马尔可夫决策过程原始-对偶学习中的参数共振

    Parametric Resonance in Primal-Dual Learning for Slowly Non-Stationary Constrained MDPs

    [https://arxiv.org/abs/2610.23788](https://arxiv.org/abs/2610.23788)

    本文揭示了缓慢非平稳约束马尔可夫决策过程中原始-对偶学习的不稳定性并非源于非平稳性本身，而是源于约束估计与测量失配所引发的参数共振，并定量给出了可容忍失配随强迫带宽的平方根增长、随对偶-原始步长比的平方根缩减的规律。

    

    本文在双时间尺度假设下分析了约束马尔可夫决策过程的原始-对偶方法，其中非平稳性通过两个渠道进入系统。在加性渠道中，约束水平发生振荡，对偶变量的振荡会饱和于由移动最优解所设定的下限，并被时间尺度比线性放大。在参数渠道中，约束灵敏度发生振荡，相干调制在任何深度下均不携带共振强迫：对偶更新的刚度调制恰好抵消了原始更新的阻尼调制。不稳定性反而来源于原始更新所使用的约束估计与对偶更新所使用的测量之间的失配，从而在系统对的固有频率的两倍处产生参数共振。可容忍的失配随强迫带宽的平方根而增大，并随对偶与原始步长比的平方根而缩减。这些结果……

    arXiv:2610.23788v1 Announce Type: cross  Abstract: Primal-dual methods for constrained Markov decision processes are analyzed under a two-timescale assumption, with non-stationarity entering through two channels. In the additive channel, where the constraint level oscillates, the dual oscillation saturates at a floor set by the moving optimum and is amplified linearly by the timescale ratio. In the parametric channel, where the constraint sensitivity oscillates, a coherent modulation carries no resonant forcing at any depth: the dual update's stiffness modulation cancels the primal update's damping modulation. Instability instead comes from a mismatch between the constraint estimate used by the primal update and the measurement used by the dual update, producing a parametric resonance at twice the pair's natural frequency. The tolerated mismatch grows with the square root of the forcing bandwidth and collapses with the square root of the dual-to-primal step-size ratio. These results ho
    
[^14]: 基于注意力机制的神经网络对XY模型进行采样

    Sampling the XY model with an attention-based neural network

    [https://arxiv.org/abs/2610.23783](https://arxiv.org/abs/2610.23783)

    该论文提出一种结合物理信息（局域性注意力机制和近似概率）的基于Transformer的变分自回归网络，可高效采样连续自由度的格点系统，在BKT相变附近的二维XY模型上显著优于现有神经采样器，并成功重现传统MCMC方法所得的物理可观测量。

    

    我们提出了一种基于Transformer的变分自回归网络，用于对具有连续自由度的格点系统进行采样，并将其应用于Berezinskii-Kosterlitz-Thouless（BKT）相变附近的二维XY模型。该架构结合了蕴含物理信息的要素：基于局域性的注意力机制和近似概率。所得到的采样器显著优于现有的神经采样器。我们在最大达$96\times96$的格点上训练了模型，通过模式丢失估计器验证了配置空间没有任何部分被遗漏，并成功重现了用传统马尔可夫链蒙特卡罗（MCMC）算法得到的能量、磁化强度和螺旋度模量。

    arXiv:2610.23783v1 Announce Type: cross  Abstract: We present a transformer-based variational autoregressive network for sampling lattice systems with continuous degrees of freedom and apply it to the two-dimensional XY model near the Berezinskii-Kosterlitz-Thouless transition. The architecture is combined with physics-informed ingredients: locality-informed attention and approximate probabilities. The resulting sampler significantly outperforms existing neural samplers. We train the model on lattices up to $96\times96$, verify with a mode-dropping estimator that no parts of the configuration space are missed, and reproduce the energy, magnetization and helicity modulus obtained with the traditional Markov Chain Monte Carlo algorithm.
    
[^15]: 广宽度神经网络的介观学习动力学

    Mesoscopic learning dynamics of extensive-width neural networks

    [https://arxiv.org/abs/2610.23779](https://arxiv.org/abs/2610.23779)

    该论文通过刻画重叠量的分布，提出了一种介观描述方法，将广宽度两层神经网络学习任意非线性目标函数的动力学约化为少数集体变量的演化，从而克服了传统方法需要追踪发散数量序参量的难题。

    

    理解随机梯度下降如何在宽神经网络中学习内部表示（这些网络在丰富的目标函数上进行训练）是机器学习中的一个核心问题。我们在两层神经网络中研究这一问题，其中目标函数由具有相同架构的网络表示，非线性形式任意，且网络宽度与输入维度成比例增长。在这种情形下，描述学习动力学的标准方法需要追踪数量发散的重叠量（overlap），这些重叠量衡量了学习到的特征与目标特征之间的相关性，因此无法用少数几个序参量给出明确的描述。在本文中，我们通过刻画重叠量的分布，发展了一种学习的介观描述方法。在学习早期阶段，我们发现重叠统计量的高斯描述能够将学习动力学约化为少数几个集体变量，这些变量追踪着目标函数的学习过程。

    arXiv:2610.23779v1 Announce Type: cross  Abstract: Understanding how stochastic gradient descent learns internal representations in wide neural networks trained on rich target functions is a central problem in machine learning. We study this question in two-layer neural networks learning target functions represented by networks of the same architecture, with arbitrary nonlinearity, and with network width and input dimension growing proportionally. In this regime, standard approaches for describing the learning dynamics require tracking a diverging number of overlaps, which measure correlations between learned and target features, leaving no obvious description in terms of a few order parameters. Here, we develop a mesoscopic description of learning by characterizing the distribution of overlaps. In the early stages of learning, we find that a Gaussian description of overlap statistics reduces the learning dynamics to a few collective variables that track learning of the target function
    
[^16]: 逻辑引导的图去噪扩散模型

    Logic-Guided Graph Denoising Diffusion Models

    [https://arxiv.org/abs/2610.23766](https://arxiv.org/abs/2610.23766)

    该论文提出了一种基于一阶逻辑（FOL）的分类器引导方法，通过将逻辑规则松弛为模糊可微的目标函数来引导扩散模型的去噪轨迹，从而使生成的图数据满足严格的结构约束。

    

    图扩散模型是生成复杂图结构数据的有竞争力的方法。然而，当这类数据分布具有严格的结构约束时，扩散模型可能无法精确捕捉这些约束，从而导致生成无效的图。在本文中，我们提出了一种通过一阶逻辑（FOL）将图扩散模型与特定领域知识相结合的方法，利用其富有表现力且可解释的形式化体系对生成过程施加结构约束。我们采用分类器引导（classifier guidance）技术来融入领域知识，该技术通过将扩散模型的分布条件化于外部变量来控制采样数据。在生成过程中，分类器引导引导去噪轨迹以优化一个目标函数，我们将该目标函数定义为逻辑规则的模糊且可微的松弛形式。我们的方法只需要指定约束条件即可。

    arXiv:2610.23766v1 Announce Type: new  Abstract: Graph diffusion models are competitive approaches for generating complex graph-structured data. However, when such distributions have strict structural constraints, diffusion models may fail to capture them exactly, resulting in the generation of invalid graphs. In this paper, we propose a method for grounding graph diffusion models in domain-specific knowledge via First Order Logic (FOL), leveraging its expressive and interpretable formalism to impose structural constraints on the generative process. We incorporate domain knowledge using classifier guidance, a technique that conditions the diffusion model's distribution on external variables to exert control over the sampled data. During the generative process, classifier guidance steers the denoising trajectory to optimize an objective function, which we define as a fuzzy and differentiable relaxation of the logical rules. Our methodology requires only the specification of constraints 
    
[^17]: 面向弱监督异常检测的通用语义增强

    Universal Semantic Enhancement for Weakly Supervised Anomaly Detection

    [https://arxiv.org/abs/2610.23765](https://arxiv.org/abs/2610.23765)

    该论文提出了一种基于通用语义增强的表格数据弱监督异常检测方法，通过将预训练表格基础模型的通用知识与目标数据集的领域特定知识相融合，有效解决了现有方法特征描述不完整且易过拟合稀缺标签的问题。

    

    arXiv:2610.23765v1 公告类型：新论文。在表格数据上进行弱监督异常检测是许多标签稀缺应用中的一项重要任务，因为表格数据广泛存在，且实践中常见的情况是仅有少量已标注的异常样本与大量未标注样本并存。大多数现有方法试图学习能够捕捉目标域固有分布模式的特征。然而，此类特征往往难以完整描述可能出现的异常特性，并且极易过拟合于极其稀缺的标签。为应对这一挑战，我们提出了一种基于通用语义增强的表格数据弱监督异常检测方法，该方法将预训练表格基础模型中的通用知识与在目标数据集上学习到的领域特定知识相融合。这两种知识天然互补：通用语义……（原文在此处截断）

    arXiv:2610.23765v1 Announce Type: new  Abstract: Weakly supervised anomaly detection on tabular data is an essential task in many label-scarce applications, since tabular data is widely encountered and it is commonly observed in practice that only a limited number of labeled anomalies are available alongside abundant unlabeled samples. Most existing methods try to learn features that capture the inherent distribution patterns in the target domain. However, such features often struggle to provide a complete description of possible abnormal characteristics and are highly susceptible to overfitting the extremely scarce labels. To address this challenge, we propose a weakly supervised anomaly detection method based on universal semantic enhancement for tabular data, which fuses the universal knowledge from a pre-trained tabular foundation model with the domain-specific knowledge learned on the target dataset. The two types of knowledge are naturally complementary: universal semantics descr
    
[^18]: 从流盒不变量到全局守恒律：机器学习辅助的奇点与近可积性检测

    From Flow-Box Invariants to Global Conservation Laws: Machine Learning-Assisted Detection of Singularities and Near-Integrability

    [https://arxiv.org/abs/2610.23760](https://arxiv.org/abs/2610.23760)

    该论文提出“扩张域神经去缩”框架，通过在嵌套扩张的相空间域上追踪独立守恒律的数量，实现了从流盒定理的局部不变量到全局守恒律的自动检测，并能识别奇点与近可积性。

    

    守恒律揭示了动力系统中的不变结构，并将系统的演化约束在低维水平集上。然而在局部范围内，根据流盒定理，每个d维系统在正则点附近通常都存在d-1个函数独立的守恒律，尽管通常只有少数这样的守恒律能够全局保持。数据驱动方法通常仅在训练样本所抽取的相空间指定区域内学习守恒律，使其在该区域之外的行为悬而未决。我们提出了这一局部到全局的延拓问题，并开发了一个“扩张域神经去缩”框架，在嵌套域上追踪独立守恒律的数量。该方法恢复了预期的d-1个局部不变量，并随着域的扩张检测到它们的相继失效，最终收敛到预期的全局守恒律极限。它区分了局部不变量（……摘要在此处截断）

    arXiv:2610.23760v1 Announce Type: cross  Abstract: Conservation laws reveal invariant structure in dynamical systems and constrain their evolution to lower-dimensional level sets. Locally, however, per the flow-box theorem, every $d$-dimensional system generically admits $d-1$ functionally independent conservation laws near a regular point even though, generically, only a few such laws persist globally. Data-driven methods typically learn conservation laws on a prescribed region of phase space from which training samples are drawn, leaving their behavior beyond that region unresolved. We formulate this local-to-global continuation problem and develop an {expanding-domain neural deflation} framework that tracks the number of independent conservation laws over nested domains. The method recovers the expected $d-1$ local invariants and detects their successive breakdown as the domain expands, relaxing to the expected global conservation law limit. It distinguishes the local invariants (as
    
[^19]: LagNet：用于长期时间序列预测的滞后结构算子

    LagNet: Lag-Structured Operators for Long-Term Time Series Forecasting

    [https://arxiv.org/abs/2610.23735](https://arxiv.org/abs/2610.23735)

    该论文通过直接考察预测算子发现其敏感度矩阵存在对应相对时间滞后的对角模式，据此提出基于 Toeplitz 矩阵的 LagLayer 模块，可作为现有时间序列预测模型中时间线性层的有效替代。

    

    长期时间序列预测是将有限的历史窗口映射到延展的未来时间段。许多现有模型主要关注输入表示和模型结构的设计，而对历史到未来预测算子本身的结构形式关注较少。我们直接考察这一算子，发现跨不同模型族中，敏感度矩阵呈现出反复出现的对角模式，其对角线对应于相对时间滞后。受此观察启发，我们开发了一个基于滞后结构时间算子的直接多步预测框架。其基本构建模块 LagLayer 通过将具有相同相对滞后的历史到未来连接绑定在一起来构建 Toeplitz 矩阵，并引入按预测视界和按历史的缩放系数来放宽严格的权重共享。这一设计使 LagLayer 能够作为现有预测模型中时间线性层的有效替代。

    arXiv:2610.23735v1 Announce Type: new  Abstract: Long-term time series forecasting maps a finite history window to an extended future horizon. Many existing models focus primarily on the design of input representations and model structures, while paying less attention to the structural form of the history-to-future forecasting operator itself. We examine this operator directly and find that, across model families, sensitivity matrices exhibit recurring diagonal patterns whose diagonals correspond to relative temporal lags. Motivated by this observation, we develop a framework for direct multi-step forecasting based on lag-structured temporal operators. Its basic building block, LagLayer, constructs a Toeplitz matrix by tying history-to-future connections with the same relative lag and incorporates horizon-wise and history-wise scaling coefficients to relax exact sharing. This design enables LagLayer to serve as an effective replacement for temporal linear layers in existing forecasting
    
[^20]: 弥合学术数据中的语言鸿沟（二）：提升中文姓名的性别推断

    Bridging the Language Gap in Scholarly Data II: Enhancing Gender Inference for Chinese Names

    [https://arxiv.org/abs/2610.23734](https://arxiv.org/abs/2610.23734)

    针对拼音中文姓名性别推断固有的模糊性问题，该研究开发了由11位中文母语者评分数据训练的汉字回归与分类性别推断模型，以提升科学计量学研究中对中文学者姓名的性别推断准确性。

    

    从姓名推断感知性别对于包容性的科学计量学研究十分重要，然而现有方法在非西方学者身上表现较差。我们针对中文姓名来解决这一缺口。中文姓名存在两种形式：汉字和拼音（其罗马化形式，例如 Zhang Wei、Li Na）。在国际书目记录中，姓名通常仅有拼音形式。然而，同一个拼音名字可能对应多个具有不同性别联想的汉字，因此仅凭拼音进行性别推断在本质上是模糊的。为解决这一问题，我们首先开发了两个汉字模型：一个回归模型，输出从0（男性）到1（女性）的连续性别得分；一个分类模型，输出三个标签之一（男性、中性、女性）。我们在来自中文物理学出版物的汉字姓名样本上训练这些模型，该样本由11位中文母语者进行评分。在该样本上，模型取得了RMSE……

    arXiv:2610.23734v1 Announce Type: new  Abstract: Perceived gender inference from names is important for inclusive scientometrics research, yet existing methods perform poorly on non-Western scholars. We address this gap for Chinese names, which appear in two forms, Hanzi and Pinyin, its Romanized form (e.g., Zhang Wei, Li Na). In international bibliographic records, names are often available only in Pinyin. However, a single pinyin name can correspond to multiple distinct hanzi characters with different gender associations, so gender inference from Pinyin alone is intrinsically ambiguous. To address this, we first develop two hanzi models, a regression model that produces a continuous gender score from 0 (male) to 1 (female), and a classification model that outputs one of three labels (\em{male}, \em{neutral}, \em{female}). We train them on a sample of hanzi names from Chinese-language physics publications, rated by 11 native Chinese speakers. On this sample, the models achieve an RMSE
    
[^21]: 逃离参数空间：基于表征质量的紧致泛化界

    Escaping Parameter Space: Tight Generalization Bounds via Representation Quality

    [https://arxiv.org/abs/2610.23729](https://arxiv.org/abs/2610.23729)

    该论文提出了一种基于最后隐藏层表征质量的紧致泛化界，通过无参数k-NN分类器评估表征质量，摆脱了传统参数空间复杂度度量，在CIFAR-10、CIFAR-100和ImageNet上实现了迄今最紧致的泛化认证。

    

    我们为确定性神经网络推导了一个基于其最后隐藏层表征的紧致泛化界。通过使用无参数的k近邻（k-NN）分类器来评估表征质量，我们的证书避免了传统的参数空间复杂度度量、与后验分布的KL散度、权重范数以及压缩方法。该界非常紧致：从头训练时，在CIFAR-10（ResNet-18）上与测试误差仅相差8.0个百分点，并且据我们所知首次在CIFAR-100上给出了非空洞的证书；使用冻结的DINOv2表征时，该界在CIFAR-10上的认证误差为1.4%，在1000类ImageNet上为23.1%。该界可加性地分解为三个可独立测量的经验量：局部类别几何结构（k-NN解码器误差）、解码器与网络的一致性，以及训练稳定性。这使得该界还可以作为一种诊断工具使用。

    arXiv:2610.23729v1 Announce Type: new  Abstract: We derive a tight generalization bound for deterministic neural We derive a tight generalization bound for deterministic neural networks based on their last hidden layer representations. By evaluating representation quality with a parameter-free $k$-nearest-neighbor ($k$-NN) classifier, our certificate avoids traditional parameter-space complexity measures, KL divergences to a posterior, weight norms, and compression. The bound is tight: trained from scratch, it reaches an 8.0 percentage point gap to test error on CIFAR-10 (ResNet-18) and yields the first non-vacuous certificates we are aware of on CIFAR-100; with frozen DINOv2 representations, it certifies at 1.4\% on CIFAR-10 and 23.1\% on 1,000-class ImageNet. It decomposes additively into three empirical, independently measurable quantities: local class geometry ($k$-NN decoder error), decoder-network agreement, and training stability. This allows the bound to act as a diagnostic too
    
[^22]: ECG-SDI：单导联心电图实现连续睡眠深度指数分析及可扩展的睡眠健康数字生物标志物

    ECG-SDI: Single-lead Electrocardiogram Enables Continuous Sleep Depth Index Profiling and Scalable Digital Biomarkers for Sleep Health

    [https://arxiv.org/abs/2610.23727](https://arxiv.org/abs/2610.23727)

    本研究开发了基于单导联心电图的ECG-SDI框架，可利用常规心电信号连续刻画睡眠深度变化并提取数字生物标志物，其识别的睡眠紊乱亚型与较差的主观睡眠质量及更高的全因死亡风险显著相关。

    

    传统的睡眠分期提供了一种粗粒度的表征，掩盖了分期内的深度变化。我们开发了ECG-SDI，这是一个基于单导联心电图（ECG）的框架，用于连续睡眠深度分析和数字生物标志物提取。我们使用12,392份整夜多导睡眠图记录，训练并验证了一个由传统睡眠分期标签监督的两阶段深度学习模型。在合并的内部测试集中，与排除快速眼动（REM）睡眠的有序分期的相关性为0.828，在外部克利夫兰家庭研究中为0.848；相应的REM分类AUROC分别为0.954和0.977。整夜轨迹能够捕捉分期内的深度变化。基于生物标志物的表型分析在睡眠心脏健康研究中识别出一种紊乱亚型，该亚型与较差的主观睡眠质量和较高的全因死亡率相关（校正风险比，1.20；95% CI，1.05-1.38）。在507名具有配对……（摘要在此处截断）

    arXiv:2610.23727v1 Announce Type: cross  Abstract: Conventional sleep staging provides a coarse-grained representation that obscures within-stage depth variation. We developed ECG-SDI, a single-lead electrocardiography (ECG) framework for continuous sleep-depth profiling and digital biomarker extraction. Using 12,392 overnight polysomnographic recordings, we trained and validated a two-stage deep learning model supervised by conventional sleep-stage labels. Correlations with ordered stages excluding rapid eye movement (REM) sleep were 0.828 in the pooled internal test set and 0.848 in the external Cleveland Family Study; corresponding REM classification AUROCs were 0.954 and 0.977. Overnight trajectories captured within-stage variation. Biomarker-based phenotyping identified a disturbed subtype associated with poorer subjective sleep quality and higher all-cause mortality in the Sleep Heart Health Study (adjusted hazard ratio, 1.20; 95% CI, 1.05-1.38). Among 507 participants with paire
    
[^23]: 神经扩张：深度神经网络如何实现泛化的统一机制

    Neural Expansion: A Unified Mechanism for How Deep Neural Networks Generalize

    [https://arxiv.org/abs/2610.23716](https://arxiv.org/abs/2610.23716)

    本文提出“神经扩张”这一统一几何框架，通过泛化区间量化训练样本周围预测稳定的方向区域，发现训练过程中模型射线覆盖沿雅可比奇异值较低的方向渐进扩张，从而揭示了深度神经网络泛化能力的形成机制。

    

    尽管深度神经网络高度过参数化，它们仍展现出卓越的泛化能力，这对传统统计学习理论构成了挑战。虽然基于间隔的界、PAC-贝叶斯分析和神经切向核等理论范式提供了全局性解释，但它们往往缺乏关于泛化如何从训练动力学中涌现的机制性洞察。我们提出了神经扩张，这是一个统一的几何框架，通过输入空间中方向选择性射线覆盖的扩张来刻画泛化。利用泛化区间，我们量化了训练样本周围模型预测保持稳定的方向特定区域。我们在多种架构和数据集上开展的广泛实证研究表明，训练会导致射线覆盖的渐进式扩张，尤其是在雅可比奇异值较低的方向上。这一机制有助于解释……

    arXiv:2610.23716v1 Announce Type: cross  Abstract: Deep neural networks demonstrate remarkable generalization despite being highly over-parameterized, challenging traditional statistical learning theories. While several theoretical paradigms such as margin-based bounds, PAC-Bayesian analyses, and neural tangent kernels provide global explanations, they often lack mechanistic insight into how generalization emerges from training dynamics. We introduce Neural Expansion, a unified geometric framework that characterizes generalization through the expansion of directionally-selective ray coverage in the input space. Using Generalization Intervals (GIs), we quantify direction-specific regions around training samples where the model's predictions remain stable. Our extensive empirical studies across diverse architectures and datasets reveal that training leads to progressive expansion of ray coverage, particularly along directions with low Jacobian singular values. This mechanism helps explai
    
[^24]: 面向多步骤智能体的工具推荐：找到任务所需的每一个工具

    Tool Recommendation for Multi-Step Agents: Finding Every Tool a Task Needs

    [https://arxiv.org/abs/2610.23707](https://arxiv.org/abs/2610.23707)

    提出了一种已部署的“工具推荐”网关系统，通过固定的三个工具接口让智能体能够访问约8,200个工具（是典型MCP服务器暴露数量的数百倍），并在ToolRet基准上将查询所需工具排进前十的成功率比BM25提高57%。

    

    当多步骤智能体任务所需的某一个工具缺失时，任务就会失败，然而智能体可能需要的工具数量庞大、难以尽数列出：在我们所研究的平台上，约8,200个工具的定义加起来超过了我们所评估的任何模型的上下文窗口。我们提出了“工具推荐”，这是一个已部署的系统，通过一个由三个工具组成的固定接口，为智能体提供所有这些工具，其数量是典型MCP服务器所暴露工具的数百倍。该系统运行在智能体与工具之间的网关处，以最终用户自己的凭证执行每一次调用，并根据用户对每项服务的授权同意以及智能体可使用的工具范围对其进行核查。我们在公开的检索基准和按计费统计的端到端会话成本上对其进行了评估。在ToolRet上，“工具推荐”将查询所需的每一个工具排进前十名的频率比BM25高出57%（BM25是商业工具搜索所采用的词法排序方法），并且与规模最大达其九倍的已发表系统相比也表现更优。

    arXiv:2610.23707v1 Announce Type: new  Abstract: A multi-step agent task fails when a single tool it needs is missing, yet the tools an agent may need are too many to list: the roughly 8,200 tools on the platform we study have definitions that together exceed the context window of any model we evaluate. We present Tool Recommendation, a deployed system that gives an agent all of them, hundreds of times more than a typical MCP server exposes, through a fixed interface of three tools. It runs at the gateway between agent and tools, executing every call under the end user's own credentials and checking it against the user's consent for each service and the tools the agent may use. We evaluate it on both a public retrieval benchmark and billed end-to-end session cost. On ToolRet, Tool Recommendation places every tool a query needs in the top ten 57% more often than BM25, a lexical ranking that commercial tool search offers, and compares favorably with published systems up to nine times its
    
[^25]: Blitz：加速AlphaFold3和Boltz-2的扩散模块

    Blitz: Accelerating the Diffusion Modules of AlphaFold3 and Boltz-2

    [https://arxiv.org/abs/2610.23703](https://arxiv.org/abs/2610.23703)

    Blitz是一个少步蒸馏框架，通过几何噪声调度和多步一致性训练，将AlphaFold 3和Boltz-2的扩散去噪步骤从200步大幅加速至8步，同时保持与教师模型相当的结构预测准确性。

    

    基于扩散模型的蛋白质结构预测模型需要大约200个去噪步骤来生成全原子结构，这一高昂的计算成本催生了多种少步采样器。我们提出了Blitz，一个少步蒸馏框架，它使用几何噪声调度将采样步骤更均匀地分布在不同噪声尺度上，然后通过多步一致性训练来学习这些更新。物理导向和无目标候选选择有助于在推理阶段减少剩余的几何缺陷。我们将Blitz应用于AlphaFold 3和Boltz-2，将去噪步骤从200步减少到8步。由于已发表的少步方法结果使用了不同的基准测试和评分程序，我们在统一的评估协议下将两个蒸馏模型与其教师模型以及其他少步方法进行比较，该协议涵盖结构准确性、界面质量、配体放置和立体化学有效性。Blitz保持了教师模型级别的结构准确性，其中AF3-Blitz的表现优于所有已发表的……（原文在此处截断）

    arXiv:2610.23703v1 Announce Type: new  Abstract: Diffusion-based protein structure prediction models produce all-atom structures in about 200 denoising steps, and that cost has driven a number of few-step samplers. We present Blitz, a few-step distillation framework that uses a geometric noise schedule to distribute steps more evenly across noise scales, then learns those updates through multistep consistency training. Physical steering and target-free candidate selection help reduce the remaining geometric defects at inference. We apply Blitz to AlphaFold 3 and Boltz-2, reducing denoising from 200 steps to eight. Because published few-step results use different benchmarks and scoring procedures, we compare both distilled models with their teachers and other few-step methods under a common evaluation protocol covering structural accuracy, interface quality, ligand placement, and stereochemical validity. Blitz retains teacher-level structural accuracy, with AF3-Blitz outperforming all e
    
[^26]: 随机机器学习实验的等效性检验与运行次数设计

    Equivalence testing and run-count design for stochastic machine-learning experiments

    [https://arxiv.org/abs/2610.23700](https://arxiv.org/abs/2610.23700)

    本文提出了一套基于两级（种子×样本）方差模型和TOST等效性检验的严格认证方案，并给出将预设容差与目标功效换算为所需运行次数的闭式公式，从而取代以往仅凭两三次运行无显著差异就宣称方法等效的不可靠做法。

    

    关于两种训练方法表现相当的论断，往往仅建立在每种方法两到三次运行之间无显著差异的基础上。我们开发了一套用于认证单次运行指标（如AUROC，受试者工作特征曲线下面积）实际等效性的方案及其功效理论。该方案建立在一个两级（种子×样本）方差模型之上，将运行间离散度与样本级噪声分离开来。预设容差范围内的等效性通过两次单侧检验（TOST）进行验证，等价地也可以通过位于容差范围内的置信区间来验证；两级自助法（bootstrap）可覆盖复合统计量。一个闭式公式可将容差和目标功效转换为每种方法所需的运行次数；其移位t分布的改进版本则根据实际的两级过程进行校准。在270个单元网格上以及非高斯种子分布规律下的仿真实验证实，该公式在其设计相关范围内（从每种方法五次运行开始）是有效的。对于四次已发表的联合……（摘要在此处被截断）

    arXiv:2610.23700v1 Announce Type: new  Abstract: Claims that two training methods perform equally well often rest on a non-significant difference between two or three runs per method. We develop a protocol, with its power theory, for certifying practical equivalence in per-run metrics such as the AUROC (area under the receiver operating characteristic curve). It rests on a two-level (seed x item) variance model separating between-run dispersion from item-level noise. Equivalence within a pre-specified margin is tested by two one-sided tests (TOST), equivalently by a confidence interval inside the margin; a two-level bootstrap covers composite statistics. A closed-form formula converts a margin and a target power into runs per method; its shifted-t refinement is calibrated against the realised two-level process. Simulations on a 270-cell grid and under non-Gaussian seed laws confirm the formula within its design-relevant range, from five runs per method. For four published runs of a joi
    
[^27]: 基于再生核希尔伯特空间的IQP特征映射量子核的表达能力研究

    Expressivity of IQP Feature-Map Quantum Kernels via Reproducing Kernel Hilbert Spaces

    [https://arxiv.org/abs/2610.23692](https://arxiv.org/abs/2610.23692)

    本研究在再生核希尔伯特空间框架下为IQP特征映射量子核推导出傅里叶表示，从理论上刻画了其所表示的函数类，并揭示了可访问频率集合由电路构造决定且受带宽超参数缩放的规律。

    

    量子核是量子机器学习中最有前景的方法之一，它提供了一种通过量子电路将数据嵌入高维希尔伯特空间的自然方式，并能捕捉数据中的非线性关系。量子核的学习行为由其所表示的函数类决定，而该函数类在形式上由相应的再生核希尔伯特空间（RKHS）的结构来刻画。然而，目前仍缺乏对这一函数类及其如何依赖于特征映射选择和超参数的理论刻画。在本工作中，我们聚焦于瞬时量子多项式（IQP）特征映射，推导出了相应RKHS生成元的傅里叶表示，即以多项式变换后的输入变量求值的三角函数的有限和，其中可访问频率的有限集合由电路构造决定，并由带宽超参数进行缩放。这一结果……

    arXiv:2610.23692v1 Announce Type: cross  Abstract: Quantum kernels are among the most promising methods in quantum machine learning, providing a natural way to embed data into high-dimensional Hilbert spaces via quantum circuits and capturing non-linear relationships in data. Their learning behaviour is governed by the function class they represent, which is formally characterized by the structure of the associated Reproducing Kernel Hilbert Space (RKHS). However, a theoretical characterization of this function class, and of how it depends on the choice of feature map and its hyperparameters, is still lacking. In this work, we focus on the Instantaneous Quantum Polynomial (IQP) feature map and derive a Fourier representation of the associated RKHS generators as finite sums of trigonometric functions evaluated on polynomially transformed input variables, with a finite set of accessible frequencies determined by the circuit construction and scaled by the bandwidth hyperparameter. This re
    
[^28]: HardEval：一个面向AI驱动安全加固与合规修复的执行验证基准

    HardEval: An Execution-Verified Benchmark for AI-Driven Security Hardening and Compliance Remediation

    [https://arxiv.org/abs/2610.23685](https://arxiv.org/abs/2610.23685)

    提出了HardEval——一个基于RHEL 8 STIG控制项、通过在真实系统上执行验证的方式，来评估大语言模型能否完成操作系统安全加固与合规修复任务的基准测试。

    

    arXiv:2610.23685v1 公告类型：cross 摘要：实施安全技术实施指南（STIG）控制项仍然是一项难度很高的操作系统管理任务。每一条以自然语言表述的安全需求都必须被转化为精确且持久的配置更改，这些更改涉及安全外壳协议（SSH）配置、身份验证策略、审计规则以及内核参数。现有的基准测试无法衡量模型能否在真实运行的系统上完成这项任务：代码基准测试评估的是缺乏持久系统状态的孤立程序，而智能体基准测试则主要聚焦于应用程序而非操作系统配置。在本文中，我们提出了HardEval，一个用于评估语言模型辅助安全加固能力的执行验证基准。HardEval基于在未加固的基线系统上无法通过的红帽企业Linux 8（RHEL 8）STIG控制项构建，包含专家撰写的参考修复方案以及开放漏洞与评估语言（OVAL）……

    arXiv:2610.23685v1 Announce Type: cross  Abstract: Implementing Security Technical Implementation Guide (STIG) controls remains a demanding operating-system administration task. Each natural-language security requirement must be translated into a precise and persistent configuration change. These changes span Secure Shell (SSH) configuration, authentication policy, audit rules, and kernel parameters. Existing benchmarks do not measure whether a model can perform this task on a live system. Code benchmarks evaluate isolated programs without persistent system state, while agent benchmarks largely focus on applications rather than operating-system configuration. In this paper, we introduce \textbf{HardEval}, an execution-verified benchmark for evaluating language-model-assisted security hardening. HardEval is built from Red Hat Enterprise Linux 8 STIG controls that fail on an unhardened baseline system. Expert-written reference remediations and Open Vulnerability and Assessment Language (
    
[^29]: TriRouter：面向鲁棒EEG运动想象解码的稀疏到密集电极空间路由

    TriRouter: Sparse-to-Dense Electrode-Space Routing for Robust EEG Motor Imagery Decoding

    [https://arxiv.org/abs/2610.23665](https://arxiv.org/abs/2610.23665)

    TriRouter提出一种包含判别保持、掩码鲁棒和潜在补偿三条路径的电极空间路由框架，通过验证校准的路由机制在电极严重缺失或受损情况下显著提升EEG运动想象解码的准确率。

    

    运动想象脑机接口需要从多通道EEG（脑电图）中进行可靠解码，但实际采集中常常会产生稀疏或受损的电极导联配置。我们提出TriRouter，这是一种具有判别保持、掩码鲁棒性和潜在补偿三条路径的电极空间路由框架。经过验证校准的路由器利用可观测的通道可用性来分配各路径的贡献，在保持完整通道判别能力的同时，增强严重通道缺失情况下的表征恢复能力。在BNCI2014-001数据集上，TriRouter优于标准的公开骨干网络和基于动态空间滤波的受损EEG基线方法。在标称25%、50%和75%的缺失设置下，相比最强的受损EEG基线，其准确率分别提升3.68、4.02和6.46个百分点，并且在结构化区域缺失场景以及额外的低通道数据集评估中依然保持有效。

    arXiv:2610.23665v1 Announce Type: new  Abstract: Motor imagery brain-computer interfaces require reliable decoding from multichannel EEG, but practical acquisition often yields sparse or corrupted electrode montages. We propose TriRouter, an electrode-space routing framework with discriminative-preservation, mask-robustness, and latent-compensation paths. A validation-calibrated router uses observable channel availability to allocate path contributions, preserving complete-channel discrimination while increasing representation recovery under severe missingness. In BNCI2014-001, TriRouter outperforms standard public backbones and a dynamic-spatial-filtering corrupted-EEG baseline. It improves accuracy over the strongest corrupted-EEG baseline by 3.68, 4.02, and 6.46 percentage points at nominal 25%, 50%, and 75% missing settings, and remains effective under structured regional missingness and additional low-channel dataset evaluation.
    
[^30]: 基于深度学习的香蕉甲虫损伤高效检测方法

    An Efficient Approach to Beetle Damage Detection of Banana Using Deep Learning

    [https://arxiv.org/abs/2610.23661](https://arxiv.org/abs/2610.23661)

    本研究提出基于YOLOv8n的轻量级深度学习框架，利用孟加拉国本地市场自建数据集（2,473张图像、5,827个标注实例）实现香蕉果实定位及表面健康状况的自动分类。

    

    香蕉果实因香蕉果实疤痕甲虫造成的可见表面损伤会降低其市场价值。目前对香蕉果实的人工检查既主观又难以规模化。本研究提出了一种基于YOLOv8n深度学习模型的轻量级框架，用于定位单个香蕉并将其可见表面状况分类为健康或不健康。论文标题中的“甲虫损伤”一词是针对当地观察到的疤痕状表面状况的操作性用语；该模型并不识别特定的甲虫种类或生物学成因。研究基于孟加拉国当地市场的图像，构建了一个包含2,473张图像、5,827个标注香蕉实例的自定义数据集。所报告的实验采用80/20划分协议，包含1,982张训练图像和491张测试图像。预训练的YOLOv8n模型在640x640分辨率下训练了50个epoch，批大小为4。在测试集上，该模型取得了8（原文摘要在此处截断）

    arXiv:2610.23661v1 Announce Type: cross  Abstract: Visible surface damage of banana fruits by Basilepta subcostata (the banana fruit scarring beetle) can reduce their market value. Here, manual inspection of Banana fruits is subjective and difficult to scale. This study presents a lightweight framework based on YOLOv8n deep learning model for locating individual bananas and classifying their visible surface condition as healthy or unhealthy. The term beetle damage in the paper title is used operationally for the locally observed scar-like surface condition; the model does not identify a specific beetle species or biological cause. A custom dataset of 2,473 images containing 5,827 annotated banana instances was prepared from local-market imagery in Bangladesh. The reported experiment used an 80/20 protocol with 1,982 training images and 491 test images. A pretrained YOLOv8n model was trained at 640x640 resolution for 50 epochs with a batch size of 4. On the test partition, it achieved 8
    
[^31]: 一种面向视觉生成的语言信息引导损失

    A Language-Informed Loss for Visual Generation

    [https://arxiv.org/abs/2610.23653](https://arxiv.org/abs/2610.23653)

    提出LILIE，一种通过归一化流变换器与视觉-语言模型联合训练的可逆图像编码器，可从中提取语言信息引导损失用于文生图训练，相比纯像素损失基线将gFD降低14个点。

    

    基于像素空间损失的端到端视觉生成方法近期在无需学习型图像分词器的情况下已取得了具有竞争力的性能。在本工作中，我们研究语言信息引导的监督信号能否为视觉生成提供比像素损失更丰富的训练信号。我们提出了LILIE（发音为"lily"），一种语言信息引导且无损的图像编码器。LILIE是一种可逆的图像变换，其训练目标是使视觉-语言模型能够从加噪特征中恢复出图像描述（caption）。该编码器采用归一化流变换器，通过下一词元预测与视觉-语言模型联合训练。随后，我们利用该编码器提取一种简单的语言信息引导损失，用于文本到图像的生成。我们使用LILIE损失训练JiT模型，在GPIC数据集上相比仅使用像素损失的基线，将gFD降低了14个点（66.84 → 53.01）。我们的代码、编码器和生成模型已发布于 https://suchiragarwal.com/lilie。

    arXiv:2610.23653v1 Announce Type: cross  Abstract: End-to-end visual generation with pixel-space loss has recently achieved competitive performance without the need for learned image tokenizers. In this work, we study whether language-informed supervision can provide a richer training signal than pixel loss for visual generation. We introduce LILIE (pronounced "lily"), a Language-Informed and Lossless Image Encoder. LILIE is an invertible image transform trained such that a vision-language model can recover captions from noised features. The encoder uses a normalizing flow transformer trained jointly with a vision-language model via next token prediction. We then use the encoder to extract a simple language-informed loss for text-to-image generation. We train JiT models with LILIE loss, reducing gFD by 14 points (66.84 $\to$ 53.01) against a pixel-only baseline on the GPIC dataset. Our code, encoders, and generative models are available at https://suchiragarwal.com/lilie.
    
[^32]: JAXtari：高吞吐量且易于修改的街机学习环境

    JAXtari: High-Throughput and Easy-to-Modify Arcade Learning Environment

    [https://arxiv.org/abs/2610.23650](https://arxiv.org/abs/2610.23650)

    JAXtari 是一个开源高性能的 Atari 2600 环境重新实现，通过大规模 GPU 并行化使 PPO 智能体能在单块 A100 GPU 上不到一小时内完成 1 亿步训练，并提供以对象为中心的观测和灵活的环境修改系统。

    

    街机学习环境（ALE）一直是深度强化学习研究的基石，为评估通用智能体提供了标准化的基准。然而，它缺乏对对象提取和环境修改的原生支持，限制了研究者评估智能体究竟是学习到了稳健的因果模型，还是依赖脆弱的视觉启发式方法。此外，ALE 对基于 CPU 仿真的依赖在现代 GPU 加速的训练流水线中造成了显著的瓶颈，限制了实验规模和迭代频率。在这项工作中，我们介绍了 JAXtari，这是一个开源、高性能的 Atari 2600 环境重新实现。通过利用大规模 GPU 并行化，JAXtari 使 PPO 智能体能够在单块 A100 GPU 上于一小时内完成 1 亿环境步数的训练。除速度之外，它还提供以对象为中心的观测和灵活的修改系统，使研究人员能够，例如……

    arXiv:2610.23650v1 Announce Type: new  Abstract: The Arcade Learning Environment (ALE) has been a cornerstone of Deep Reinforcement Learning research, providing a standardized benchmark for evaluating general agents. However, its lack of native support for object-extraction and environment modification restricts research into evaluating whether agents learn robust causal models or rely on brittle visual heuristics. Furthermore, the ALE's dependence on CPU-based emulation creates a significant bottleneck in modern GPU-accelerated training pipelines, limiting both experimental scale and iteration frequency. In this work, we introduce JAXtari, an open-source, high-performance reimplementation of Atari 2600 environments. By leveraging massive GPU parallelization, JAXtari allows PPO agents to finish training on 100M environment steps on a single A100-GPU in under an hour. Beyond speed, it provides object-centric observations and a flexible modification system, allowing researchers to, e.g. 
    
[^33]: 具备电响应的可扩展长程机器学习原子间势用于原子尺度材料建模

    Scalable long-range machine learning interatomic potentials with electrical response for atomistic material modeling

    [https://arxiv.org/abs/2610.23643](https://arxiv.org/abs/2610.23643)

    该论文通过解析分离全局静电计算与学习潜电荷的严格局域导数，构建了一种兼具长程静电与电响应建模能力、同时保持空间可分解性的可扩展机器学习原子间势，实现了电场下材料的分布式高效原子模拟。

    

    arXiv:2610.23643v1 公告类型：cross 摘要：对电场下材料进行精确的原子尺度模拟，需要在超越传统第一性原理方法适用尺度的长度尺度上建模长程静电相互作用和电响应。然而，这些能力至今难以与机器学习原子间势（MLIPs）的可扩展性相兼容。严格的局域模型虽然可以实现高效的空间并行化，却忽略了长程相互作用；而现有的静电扩展方法又引入了全局依赖，阻碍了分布式计算。本文通过解析地将长程静电力和玻恩有效电荷（BECs）分离为全局静电计算与学习到的潜电荷的严格局域导数，克服了这一限制。该表述在保持等变MLIPs空间可分解性的同时，能够对不同边界条件下的能量、力、应力和BECs进行分布式评估……（原文摘要在此处被截断）

    arXiv:2610.23643v1 Announce Type: cross  Abstract: Accurate atomistic simulations of materials under electric fields require long-range electrostatics and electrical-response modeling at length scales beyond the reach of conventional first-principles methods. Yet these capabilities remain difficult to reconcile with the scalability of machine-learning interatomic potentials (MLIPs). Strictly local models enable efficient spatial parallelization but neglect long-range interactions, whereas existing electrostatic extensions introduce global dependencies that hinder distributed computation. Here, we overcome this limitation by analytically separating the long-range electrostatic forces and Born effective charges (BECs) into a global electrostatics calculation and strictly local derivatives of learned latent charges. This formulation preserves the spatial decomposability of equivariant MLIPs while enabling distributed evaluation of energies, forces, stresses, and BECs across different boun
    
[^34]: 无遗憾语言生成

    No-Regret Language Generation

    [https://arxiv.org/abs/2610.23641](https://arxiv.org/abs/2610.23641)

    本文提出无遗憾生成框架，用基于树的错误强制维度精确刻画生成任务的极小极大复杂度，揭示所有可容许生成率均可达成，并将其与在线预测联系起来，得到由Littlestone维度决定的错误界和遗憾界。

    

    我们提出“无遗憾生成”这一概念，它要求生成器在所有示范序列和兼容的目标语言上均匀地产生次线性数量的错误。我们还提出了一个不可知变体，允许使用可能与任何语言都不兼容的任意示范。对于这两种设定，我们引入了基于树的错误强制维度，它精确刻画了有限视野下的极小极大复杂度，并建立了无遗憾生成的充分必要条件。我们的刻画揭示了一个丰富的生成率谱系：在常数尺度调整下，每一种可容许的非递减曲线在可实现和不可知两种设定中都是可以达成的。最后，我们将生成复杂度与在线预测联系起来：对于Littlestone维度为有限值 d 的语言类，我们获得了 O(d log(1+T/d)) 的可实现错误界和 O(√(dT log(1+T/d))) 的不可知遗憾界。

    arXiv:2610.23641v1 Announce Type: new  Abstract: We introduce no-regret generation, which requires a generator to make a sublinear number of mistakes uniformly over all demonstration sequences and compatible target languages. We also formulate an agnostic variant that allows arbitrary demonstrations that may not be compatible with any languages. For both settings, we introduce tree-based mistake-forcing dimensions that exactly characterize finite-horizon minimax complexity and establish necessary and sufficient conditions for no-regret generation. Our characterizations reveal a rich spectrum of generation rates: up to constant rescaling, every admissible nondecreasing profile is attainable in both the realizable and agnostic settings. Finally, we connect generation complexity to online prediction: for language classes of finite Littlestone dimension \(d\), we obtain realizable mistake bounds of \(O(d\log(1+T/d))\) and agnostic regret bounds of \(O(\sqrt{dT\log(1+T/d)})\). The realizabl
    
[^35]: 深度大语言模型层揭示默读期间脑电图中的段落级上下文信息

    Deep LLM layers expose passage-level context in EEG during silent reading

    [https://arxiv.org/abs/2610.23638](https://arxiv.org/abs/2610.23638)

    该研究通过对大语言模型Llama-3.1-8B每一层表征训练EEG解码器，发现默读时词级信息在模型早期层最易解码，而段落级上下文信息随模型深度增加而增强，表明深层LLM表征与阅读中的上下文加工相关。

    

    在之前的报告中，我们展示了解码器能够从默读期间记录的19通道脑电图（EEG）中恢复开放词汇的词级信息，方法是将短EEG时间窗与从大语言模型（LLM）中提取的所呈现单词的隐状态嵌入进行对齐。该分析对比了该模型33层中的两层：无上下文的输入嵌入层和一个中间深度层。在本研究中，我们探讨EEG可解码结构如何分布于语言模型的全部深度。我们为Llama-3.1-8B的每一层分别训练了解码器，并在有和没有作用于EEG时间窗序列的因果Transformer两种条件下进行。我们区分了词级可解码性与上下文追踪成分。词级可解码性在早期层达到峰值，并随着接近最后一层而稳步下降。相比之下，检索增益中的上下文追踪成分随深度增加而上升，无论是否使用因果Transformer均是如此。出乎意料的是，一个仅观察单个0.5秒时间窗的解码器……

    arXiv:2610.23638v1 Announce Type: new  Abstract: In a previous report we showed that a decoder can recover open-vocabulary word-level information from 19-channel EEG recorded during silent reading, by aligning short EEG windows with hidden-state embeddings of the presented word taken from a large language model (LLM). That analysis contrasted two of the model's 33 layers: the context-free input embedding and one mid-depth layer. Here we ask how EEG-decodable structure is distributed over the full depth of the language model. We trained one decoder per layer of Llama-3.1-8B, with and without a causal transformer over the sequence of EEG windows. We distinguish word-level decodability from a context-tracking component. Word-level decodability peaks in early layers and declines steadily towards the last layer. In contrast, the context-tracking component of the retrieval gain rises with depth, both with and without a causal transformer. Unexpectedly, a decoder that sees a single 0.5-s wind
    
[^36]: 解构多任务主动学习：梯度冲突悖论与正交分解

    Deconstructing Multi-Task Active Learning: The Paradox of Gradient Conflict and Orthogonal Decomposition

    [https://arxiv.org/abs/2610.23623](https://arxiv.org/abs/2610.23623)

    该论文揭示了多任务主动学习中的“冲突悖论”——采集函数对梯度冲突进行评分反而会降低性能——并提出从线性组合到解构的范式转变，通过正交分解框架将冲突评分与干扰处理解耦：采集函数不评分冲突，梯度冲突干扰交由训练阶段的梯度手术解决。

    

    现有的多任务主动学习（MTL-AL）方法主要依赖于对异构目标（如不确定性、多样性和梯度冲突）进行线性标量化。我们认为这种纠缠式的建模方式存在根本性局限，因为线性组合无法在目标相互冲突的情况下捕捉帕累托最优权衡。我们提出从“组合”到“解构”的范式转变，并引入一个原则性框架，沿三个正交维度重新思考多任务主动学习。首先，我们识别出“冲突悖论”：尽管优化领域文献将梯度冲突视为需要抑制的干扰，但在对候选样本进行排序时对冲突进行惩罚，其效果反而明显差于完全不对其进行评分。因此，我们将这两个关注点解耦：采集函数不对冲突进行评分，而由此产生的干扰则通过训练过程中的梯度手术（gradient surgery）来解决。其次，我们引入正交分解，将采集准则分离……

    arXiv:2610.23623v1 Announce Type: new  Abstract: Existing multi-task active learning (MTL-AL) methods predominantly rely on linear scalarization of heterogeneous objectives such as uncertainty, diversity, and gradient conflict. We argue that this entangled formulation is fundamentally limited, as linear combination fails to capture Pareto-optimal trade-offs under conflicting objectives. We propose a shift from combination to deconstruction and introduce a principled framework that rethinks MTL-AL along three orthogonal dimensions. First, we identify the Conflict Paradox: although the optimization literature treats gradient conflict as interference to be suppressed, penalizing conflict when ranking candidates is measurably worse than not scoring it at all. We therefore decouple the two concerns: the acquisition function does not score conflict, and the resulting interference is resolved by gradient surgery during training. Second, we introduce Orthogonal Decomposition, separating acquis
    
[^37]: GRAFT-EEG：面向跨数据集运动想象脑电的规范感知物理场传输

    GRAFT-EEG: Gauge-Aware Physical Field Transport for Cross-Dataset Motor-Imagery EEG

    [https://arxiv.org/abs/2610.23620](https://arxiv.org/abs/2610.23620)

    GRAFT-EEG通过移除有效电极集合的公共参考规范并借助模板导联场传输电极掩码，实现了完全不依赖目标数据统计信息的跨数据集运动想象脑电解码，在253名被试上将平衡准确率从BIOT的52.684%提升至60.865%。

    

    跨数据集脑电解码通常将导联布局、参考设置和有效电极的变化视为表征偏移，但实际上它们还会改变观测算子。我们研究目标统计盲迁移：拟合过程不接收任何目标信号、标签或经验统计量，电极坐标仅用于实例化一个固定的预测算子。该方法移除了有效电极集合的公共参考规范，并通过模板运动导联场传输每个原生电极掩码。源域拟合的谱分类器作为决策锚点，有界的状态空间模型（SSM）残差则作为辅助。我们建立了受限可辨识性结果和算子扰动界。在五个留一数据集目标（共253名被试）上，动态方法取得了60.865%的等目标平衡准确率，而严格干净设置下的5/5 BIOT为52.684%（层次差+8.182个百分点；95%置信区间+6.382至+10.196；在每个目标上均为正向）。在四个EEGPT-clean目标上，……（原文摘要在此处截断）

    arXiv:2610.23620v1 Announce Type: new  Abstract: Cross-dataset EEG decoding often treats montage, reference, and active-sensor changes as representation shift, although they also alter the observation operator. We study target-statistics-blind transfer: fitting receives no target signal, label, or empirical statistic, and electrode coordinates only instantiate a fixed prediction operator. method removes the active-set common-reference gauge and transports each native mask through a template motor leadfield. A source-fitted spectral classifier anchors the decision; a bounded state-space-model (SSM) residual is secondary. We establish restricted identifiability and an operator-perturbation bound. Across five leave-one-dataset-out targets (253 participants), dynamic method achieved 60.865% equal-target balanced accuracy versus 52.684% for strict-clean 5/5 BIOT (hierarchical difference +8.182 points; 95% CI +6.382 to +10.196; positive in every target). On four EEGPT-clean targets, the desc
    
[^38]: ThetreStageAI：面向结构化孟加拉戏剧制作规划的多视图检索增强生成

    ThetreStageAI: Multi-View Retrieval-Augmented Generation for Structured Bengali Theatre Production Planning

    [https://arxiv.org/abs/2610.23618](https://arxiv.org/abs/2610.23618)

    本文提出 ThetreStageAI，一个通过场景、走位、灯光多视图检索结合模式约束 JSON 生成来实现结构化孟加拉戏剧制作规划的检索增强框架，并配套构建了 500 条记录的人工引导合成数据集。

    

    排演一场戏剧演出需要在同一份制作计划中协调台词、演员走位、舞台指示、灯光聚焦、提示时机以及设备限制等多方面因素。本工作提出了 ThetreStageAI，一个用于结构化孟加拉戏剧制作规划的检索增强框架。该系统并非从单一混合语料库中检索，而是维护场景、演员走位和灯光三个独立视图。孟加拉语文本使用 BGE-M3 进行嵌入，并索引到各视图专属的 Qdrant 集合中。检索到的参考资料与制作简报相结合后传递给 Gemini，由其生成符合模式约束的 JSON，其中包含角色、场景、台词、舞台指示、走位、灯光以及可选的音效提示。在接受输出前会进行结构与语义检查，并最多进行一次受限修复尝试。此外，作者还构建了一个由人工引导、大语言模型辅助的合成数据集，包含 500 条简短的孟加拉戏剧记录，该数据集扩展了……（摘要原文在此处截断）

    arXiv:2610.23618v1 Announce Type: new  Abstract: Staging a theatre performance requires coordinating dialogue, actor movement, stage directions, lighting focus, cue timing, and equipment constraints within the same production plan. This work presents ThetreStageAI, a retrieval augmented framework for structured Bengali theatre production planning. Rather than retrieving from one mixed corpus, the system maintains separate scene, actor-blocking, and lighting views. Bengali text is embedded with BGE-M3 and indexed in view-specific Qdrant collections. Retrieved references are combined with a production brief and passed to Gemini, which generates schema-constrained JSON containing characters, scenes, dialogue, stage directions, blocking, lighting, and optional sound cues. Structural and semantic checks are applied before acceptance, with at most one constrained repair attempt. We also construct a human-guided, LLM-assisted synthetic dataset of 500 short Bengali theatre records that expands
    
[^39]: 符号编码：深度网络的隐藏二值本质

    The Sign Code: The Hidden Binary Nature of Deep Networks

    [https://arxiv.org/abs/2610.23616](https://arxiv.org/abs/2610.23616)

    该论文发现深度网络的分类能力几乎完全由神经元激活的符号（二值符号编码）而非幅值决定——仅凭符号编码即可达到与完整模型相当的准确率，而翻转20%的符号就会使性能降至随机水平。

    

    在训练好的网络中，每个神经元输出的数值都可以分解为符号（正或负）和幅值（绝对大小）。我们将所有神经元符号的集合称为“符号编码”。由此引出一个关键问题：分类更多依赖于符号编码还是幅值？对于采用符号保持激活函数的过参数化视觉分类器，答案压倒性地是符号编码。仅基于这一二值符号编码运行的k近邻分类器，在不同的架构、激活函数和数据集上都能取得与完整模型相当的准确率。此外，在保留原始符号的同时，将每一层的所有幅值替换为随机噪声，几乎不会改变残差网络的性能。相反，仅仅翻转20%的符号就会使准确率降至随机水平。当同类输入共享相似的符号编码时，模型具有良好的泛化能力。而当这种重叠被正则化器破坏时，……

    arXiv:2610.23616v1 Announce Type: new  Abstract: Every neuron in a trained network outputs a number that can be separated into its sign (positive or negative) and its magnitude (absolute size). We call the set of all neuron signs the \textit{sign code}. This raises a key question: does classification depend more on the sign code or on the magnitudes? For overparameterized vision classifiers with sign-preserving activations, the answer is overwhelmingly the sign code. A $k$-nearest neighbor classifier operating only on this binary sign code achieves accuracy comparable to the full model across different architectures, activation functions, and datasets. Moreover, replacing all magnitudes in every layer with random noise while keeping the original signs barely changes residual network performance. In contrast, flipping just 20\% of the signs reduces accuracy to chance. Models generalize well when same-class inputs share similar sign codes. When this overlap is disrupted by a regularizer,
    
[^40]: 轨迹触发的稀疏协作解码

    Trajectory-Triggered Sparse Collaborative Decoding

    [https://arxiv.org/abs/2610.23613](https://arxiv.org/abs/2610.23613)

    该论文提出利用草稿置信度路径的聚合轨迹形状特征（而非局部不确定性指标）来决定何时调用目标模型进行纠正的稀疏协作解码方法，将触发失败预测的AUC从接近随机的0.50-0.53提升至0.86，并揭示了触发时机上干预杠杆与证据强度之间的权衡。

    

    诸如投机解码等稠密辅助方法需要反复使用目标模型验证草稿token，使目标模型的计算与生成过程紧密耦合。我们研究了稀疏协作解码，其中草稿模型生成大部分token，而目标模型仅在需要短暂纠正片段时才被调用。关键问题在于决定何时调用目标模型。我们证明局部不确定性是一个弱触发信号：熵、负对数似然、logit间隔、top-k概率质量和速度特征对失败的预测能力接近随机水平，AUC仅为0.50-0.53。相比之下，基于草稿置信度路径的聚合轨迹形状特征，配合仅使用草稿正确性标签训练的轻量级逻辑回归预测器，可达到0.86的AUC。这引出了一个触发时机的权衡：较早调用具有更大的干预杠杆但证据较弱，而较晚调用则证据更强但重定向轨迹的能力较弱。在线使用时，学习到的稀疏协作解码

    arXiv:2610.23613v1 Announce Type: new  Abstract: Dense assistance methods such as speculative decoding repeatedly verify draft tokens with a target model, keeping target computation coupled to generation. We study sparse collaborative decoding, where a draft model generates most tokens and a target model is invoked only for short corrective segments. The key problem is deciding when to call the target. We show that local uncertainty is a weak trigger: entropy, negative log-likelihood, logit margin, top-$k$ mass, and velocity features give near-chance failure prediction, with AUC 0.50-0.53. In contrast, aggregate trajectory-shape features over the draft confidence path reach AUC 0.86 using a lightweight logistic predictor trained only from draft correctness labels. This induces a trigger-timing tradeoff: early calls have more leverage but weaker evidence, while later calls have stronger evidence but less ability to redirect the trajectory. Used online, learned sparse collaborative decod
    
[^41]: 多少个，但有多确定？类别竞争作为多类别基于密度的计数中的置信度代理

    How Many, but How Sure? Class Competition as a Confidence Proxy in Multi-Class Density-Based Counting

    [https://arxiv.org/abs/2610.23611](https://arxiv.org/abs/2610.23611)

    提出一种无需训练的置信度代理方法，利用多类别密度计数中最强类别通道间的局部竞争来衡量计数可靠性，推理时衰减竞争密度可将 Hicks 数据集上的总体 MAE 从 0.920 降至 0.846。

    

    多类别密度估计器可以在拥挤场景中对物体进行计数，但其按类别划分的输出无法指示密度质量分配的可靠程度。我们引入了一种无需训练的置信度代理，它基于最强的类别通道之间的局部竞争。由此得到的显著性度量只需要预测的密度图，无需重新训练或访问模型内部结构。我们通过在推理时衰减存在竞争的密度，来检验残余竞争是否能够诊断跨类别泄漏。在 Hicks 等人数据集的 DSACA 预测上，该方法将总体 MAE 从 0.920 降低至 0.846，表明类别竞争包含了关于计数可靠性的有用信息。

    arXiv:2610.23611v1 Announce Type: new  Abstract: Multi-class density estimators can count objects in crowded scenes, but their class-wise outputs offer no indication of how reliably density mass has been allocated. We introduce a training-free confidence proxy based on local competition between the strongest class channels. The resulting prominence measure requires only the predicted density maps, with no retraining or access to model internals. We test whether residual competition is diagnostic of cross-class leakage by attenuating contested density at inference time. On DSACA predictions for the Hicks et al. dataset, this reduces overall MAE from 0.920 to 0.846, suggesting that class competition contains useful information about count reliability.
    
[^42]: 当幅度足够时：在胸部体模上无相位重构时域门控微波信号

    When Magnitude Suffices: Reconstructing Time-Domain-Gated Microwave Signals Without Phase on a Thoracic Phantom

    [https://arxiv.org/abs/2610.23608](https://arxiv.org/abs/2610.23608)

    该研究证明，仅使用频谱分析仪的幅度测量数据结合全连接神经网络，即可在胸部体模上重构出与矢量网络分析仪结果相当的时域门控微波信号幅度，无需昂贵的相位信息测量。

    

    时域门控技术用于分离构成微波S参数测量的各到达信号。它需要复数值测量，因此该应用几乎总是使用矢量网络分析仪（VNA）。我们证明，在一种校准的测量配置下，矢量网络分析仪所能给出的门控幅度可以从配备跟踪信号源的频谱分析仪（SA）的仅幅度扫描中重构出来。我们选择门中心以实现测量状态的可分离性，通过对参考状态的逐频点减法对每台仪器进行一次性校正，并在两台仪器的扫描数据上训练一个全连接网络。测试对象是一个具有肺形容腔的躯干体模，其填充物具有十一种介电状态，每种状态依次被留出作为测试。在$S_{11}$上，从SA输入的重构达到0.08 dB$^2$的形状误差和0.81的状态排序，与从VNA输入的重构结果相当。在$S_{21}$上，其误差为0.82 dB$^2$。

    arXiv:2610.23608v1 Announce Type: new  Abstract: Time-domain gating separates the arrivals that form a microwave S-parameter measurement. It requires the measurement of complex values, so this application almost always uses a vector network analyzer (VNA). We show that, on one calibrated measurement configuration, the gated magnitude the VNA would give can be reconstructed from a magnitude-only sweep of a spectrum analyzer with a tracking generator (SA). We select the gate center for separability of the measured states, correct each instrument once by a per-bin subtraction of a reference state, and train a fully connected network on sweeps from both instruments. The test object is a torso phantom with lung-shaped compartments whose filling takes eleven dielectric states, each held out in turn. On $S_{11}$ the reconstruction from SA input reaches a shape error of 0.08 dB$^2$ with a state ordering of 0.81. This matches the reconstruction from VNA input. On $S_{21}$ it reaches 0.82 dB$^2$
    
[^43]: 高维点云的边界检测

    Boundary Detection for High-Dimensional Point Clouds

    [https://arxiv.org/abs/2610.23590](https://arxiv.org/abs/2610.23590)

    本文提出两种适用于高维、非均匀密度且含噪点云数据的新型边界检测方法，可同时输出边界分数和法向量估计，并在肿瘤边界检测与LLM激活几何分析中得到验证。

    

    我们提出了两种用于高维点云数据边界检测的新方法，旨在适用于密度不均匀、环境维度高且噪声较大的情况。第一种方法在进行局部降维后，于局部邻域上拟合一个分离超平面——当数据的内在维度可以被可靠估计时，该方法效果最佳。第二种方法通过测量局部邻域的方向均匀性来进行检测，在合成数据上取得了总体最优的表现。除了给出逐点的数值边界分数外，我们的算法还能在每个边界点处输出估计的法向量。我们在合成数据以及两个真实案例上验证了该方法：我们分析了空间转录组学数据，证明了能够检测肿瘤的边界；我们还研究了大型语言模型（LLM）的激活几何。

    arXiv:2610.23590v1 Announce Type: new  Abstract: We introduce two new methods for boundary detection in high-dimensional point cloud data, intended to work in situations with non-uniform density, high ambient dimension, and substantial noise. Our first method fits a separating hyperplane on local neighborhoods after doing local dimension reduction---this method works best when one can reliably estimate the intrinsic dimension of the data. Our second method measures the directional uniformity of local neighborhoods, and has the best overall performance on synthetic data. In addition to pointwise numerical boundary scores, our algorithms yield an estimated normal vector at each boundary point. We validate our methodology on synthetic data as well as two real examples: we analyze spatial transcriptomics data, where we show that we can detect the boundary of a tumor, and we also study LLM activation geometry.
    
[^44]: BMuon：面向结构化可压缩性的Muon谱几何预算

    BMuon: Budgeting Muon's Spectral Geometry for Structured Compressibility

    [https://arxiv.org/abs/2610.23571](https://arxiv.org/abs/2610.23571)

    提出BMuon优化器，通过为Muon的谱几何引入可解释的全局通道分配预算，使隐藏通道竞争共享预算，从而在保持优化性能的同时赋予模型训练后的结构化可压缩性。

    

    Muon已成为神经网络训练的一种强优化器，并可被解释为谱范数几何下的最陡下降。然而我们发现，使用Muon训练的Transformer在训练后对其MLP隐藏通道进行剪枝时性能会急剧下降。为了调和Muon的优化性能与结构化可压缩性之间的矛盾，我们提出了Budget Muon（BMuon），它修改了Muon的谱几何，使得跨模块的隐藏通道共同竞争一个共享的分配预算。由此得到的oracle恰好是在一族介于Muon几何与结构化稀疏几何之间的范数下的归一化最陡下降。我们的主要稀疏性控制手段是一个可解释的全局通道分配预算，而非依赖于具体设置的稀疏性惩罚系数。对于线性分解模型，我们证明了Muon允许存在具有冗余隐藏宽度的最小代价精确表示，而在BMuon下（摘要原文在此处截断）

    arXiv:2610.23571v1 Announce Type: new  Abstract: Muon has emerged as a strong optimizer for neural network training and admits an interpretation as steepest descent under a spectral-norm geometry. Yet we find that Muon-trained Transformers can degrade sharply when their MLP hidden channels are pruned after training. To reconcile Muon's optimization performance with structured compressibility, we introduce Budget Muon (BMuon), which modifies Muon's spectral geometry so that hidden channels across blocks compete for a shared allocation budget. The resulting oracle is exactly normalized steepest descent under a family of norms that interpolate between Muon's geometry and structured sparse geometries. Our principal sparsity control is an interpretable global channel-allocation budget, rather than a setting-dependent sparsity-penalty coefficient. For linear factorization models, we show that Muon admits minimum-cost exact representations with redundant hidden width, whereas under the BMuon 
    
[^45]: 迭代算法的数据自适应任意时刻有效包络的通用构建方法

    A general recipe for data-adaptive anytime-valid envelopes for iterative algorithms

    [https://arxiv.org/abs/2610.23570](https://arxiv.org/abs/2610.23570)

    该论文提出了一个通用方法，可为一大类迭代算法构建任意时刻有效且数据自适应的停止边界，将误差分析统一归结为对可预测加权噪声和满足时间一致集中性这一单一要求，克服了以往确定性包络过于保守、数据自适应包络只能逐例推导的局限。

    

    大多数针对迭代随机算法的收敛性保证都是在固定迭代次数处陈述的，无法为在数据相关的时刻停止算法提供合理依据。近期的工作构建了能够提供这种依据的任意时刻有效包络，但确定性的包络对于实际停止而言通常过于保守，而现有的数据自适应包络则是针对具体案例逐一推导的，每一种都将特定算法与特定噪声模型及边界构造耦合在一起。本文为一大类迭代算法提供了一个构建此类任意时刻有效、数据自适应边界的通用方法。这些算法中许多算法的性能度量满足一个单步递归关系，这将整个误差分析简化为一个单一的通用要求：对可预测加权的噪声和具有时间一致的集中性。我们通过一个强适应的 sub-ψ 条件将这一要求形式化，并借助连续与离散的混合……（原文摘要至此处截断）

    arXiv:2610.23570v1 Announce Type: cross  Abstract: Most convergence guarantees for iterative stochastic algorithms are stated at a fixed iteration count and do not justify stopping at a data-dependent time. Recent work has constructed anytime-valid envelopes that do, but deterministic envelopes are typically too conservative for practical stopping, and existing data-adaptive ones are derived case-by-case, each coupling a specific algorithm to a specific noise model and boundary construction. This paper provides a general recipe for constructing such anytime-valid data-adaptive boundaries for a large class of iterative algorithms. Performance measures of many of these algorithms satisfy a one-step recursion that reduces the entire error analysis to a single general requirement: time-uniform concentration for predictably weighted sums of noise. We formalize this through a strongly adapted sub-$\psi$ condition, and derive off-the-shelf boundary constructions via continuous and discrete mi
    
[^46]: 超越期望值：基于分布式蒙特卡洛树搜索的风险敏感规划

    Beyond Expected Values: Risk-Sensitive Planning with Distributional Monte-Carlo Tree Search

    [https://arxiv.org/abs/2610.23565](https://arxiv.org/abs/2610.23565)

    该论文提出了CATSO和PATSO两种分布式蒙特卡洛树搜索算法，通过在Q边维护完整的经验分布（而非仅存储均值估计）并结合CVaR与乐观奖励进行动作选择，实现了针对尾部风险的风险敏感规划，并证明了n次rollout后根节点值误差的O(n^{-1/2})理论界。

    

    蒙特卡洛树搜索（MCTS）通常只存储标量均值估计，这使其难以胜任诸如条件风险价值（CVaR）等尾部风险目标。我们提出了带乐观性的分类汤普森采样（CATSO）和带乐观性的粒子汤普森采样（PATSO），这是两种分布式MCTS算法，它们在Q边上维护有限支撑的经验备份分布。CATSO使用分类原子和狄利克雷汤普森采样来表示Q边分布，而PATSO则使用自适应粒子来表示。两种算法均利用汤普森采样的Q边分布的CVaR加上多项式乐观奖励来选择动作，并通过子节点Q边CVaR分数的访问加权平均来传播标量延续值。针对有限深度的嵌套CVaR规划目标，我们证明了在n次rollout后根节点值误差的O(n^{-1/2})界，该界同时上界了风险敏感的简单遗憾。CATSO增加了一个固定网格离散（此处原文被截断）……

    arXiv:2610.23565v1 Announce Type: new  Abstract: Monte-Carlo Tree Search (MCTS) typically stores only scalar mean estimates, which makes it poorly suited for tail-risk objectives such as conditional value-at-risk (CVaR). We introduce Categorical Thompson Sampling with Optimism (CATSO) and Particle Thompson Sampling with Optimism (PATSO), two distributional MCTS algorithms that maintain finite-support empirical backup laws at Q-edges. CATSO represents Q-edge laws with categorical atoms and Dirichlet Thompson sampling, while PATSO represents them with adaptive particles. Both algorithms select actions using the CVaR of a Thompson-sampled Q-edge law plus a polynomial optimism bonus, and propagate scalar continuation values through visit-weighted averages of child Q-edge CVaR scores. For the finite-depth nested CVaR planning target, we prove an $O(n^{-1/2})$ bound on the root value error after $n$ rollouts, which upper-bounds the risk-sensitive simple regret. CATSO adds a fixed-grid discre
    
[^47]: 超越潜在个性化：通过大语言模型整合基于HRV条件化的个性化以实现可解释的压力检测

    Beyond Latent Personalization: Integrating HRV-Conditioned Personalization via LLM for Interpretable Stress Detection

    [https://arxiv.org/abs/2610.23555](https://arxiv.org/abs/2610.23555)

    该论文提出一种通过LLM语义解释整合HRV特征与个人属性的个性化压力检测框架，以人类可读的方式描述个体生理反应，在TILES-2018数据集上相比最强基线实现29.9%的相对MCC提升，兼顾了检测性能与可解释性。

    

    基于心率变异性（HRV）的压力检测受到越来越多的关注。然而，现有的个性化方法通过潜在表示将个人属性与HRV进行整合，限制了二者交互的可解释性。在这项工作中，我们提出了一种由大语言模型（LLM）增强的个性化压力检测框架，该框架通过基于LLM的语义解释来整合HRV特征和个人属性。在TILES-2018数据集上，所提出的框架相比最强基线实现了29.9%的相对MCC提升，同时提供了人类可读的个体生理反应描述。这些结果凸显了基于LLM的语义解释在实现既有效又可解释的个性化压力检测方面的潜力。

    arXiv:2610.23555v1 Announce Type: cross  Abstract: Stress detection based on heart rate variability (HRV) has received increasing attention. However, existing personalized approaches integrate personal attributes with HRV through latent representations, limiting the interpretability of their interactions. In this work, we propose an LLM-enhanced personalized stress detection framework that integrates HRV features and personal attributes through LLM-based semantic interpretation. The proposed framework achieves a 29.9% relative MCC improvement over the strongest baseline on TILES-2018 while providing human-readable descriptions of individual physiological responses. The results highlight the potential of LLM-based semantic interpretation for achieving both effective and interpretable personalized stress detection.
    
[^48]: 超越条件独立的因果发现：有向无环图与循环图的可识别性、算法与统计推断

    Causal Discovery beyond Conditional Independence: Identifiability, Algorithms, and Statistical Inference for Directed Acyclic and Cyclic Graphs

    [https://arxiv.org/abs/2610.23543](https://arxiv.org/abs/2610.23543)

    本综述系统梳理了超越条件独立性的现代因果发现方法，提出将识别来源与统计估计相分离的统一框架，涵盖分布不对称、条件矩约束和跨环境变异等手段，并讨论了潜在混淆与循环图下的可识别性、算法与统计推断。

    

    从数据中学习因果结构需要对机制、噪声、观测方案和干预做出假设。历史上，因果发现主要依赖条件独立性。本综述聚焦于利用条件独立性之外的分布性质的现代因果发现方法。我们的组织原则是将识别来源与统计估计量及其计算实现分离开来。这一视角将条件独立方法与分布不对称性、条件矩约束以及跨环境或跨个体的变异联系起来。我们考察了存在潜在混淆与反馈情况下的因果发现，区分了关于观测变量的因果信息与潜在结构的重构，并强调了循环模型所需的额外语义。我们还区分了总体层面的可识别性与频率学派及贝叶斯一致性（摘要在此处被截断）。

    arXiv:2610.23543v1 Announce Type: cross  Abstract: Learning causal structure from data requires assumptions about mechanisms, noises, observation schemes, and interventions. Historically, causal discovery has relied primarily on conditional independence. This review focuses on modern causal discovery methods that exploit properties of distributions beyond conditional independence. Our organizing principle separates the source of identification from the statistical estimator and its computational implementation. This perspective connects conditional-independence methods with distributional asymmetries, conditional-moment restrictions, and variation across environments or individuals. We examine discovery under latent confounding and feedback, distinguishing causal information about observed variables from reconstruction of latent structure and emphasizing the additional semantics required by cyclic models. We also distinguish population identification, frequentist and Bayesian consisten
    
[^49]: 超出支持范围，但并非遥不可及：使用不支持指令与组合任务的上下文学习

    Out of Support, Not Out of Reach: In-Context Learning with Unsupported Instructions and Composed Tasks

    [https://arxiv.org/abs/2610.23528](https://arxiv.org/abs/2610.23528)

    该论文建立了理论框架，证明上下文学习即使在指令或目标任务超出预训练支持范围时仍能成功，示例证据足以克服指令分布偏移并保持指数级的任务识别速率。

    

    上下文学习（ICL）使预训练的大语言模型能够通过少量输入-输出示例适应下游任务，而无需参数更新。然而，现有理论主要将ICL解释为识别预训练中已表示的潜在任务，当指令或目标任务超出预训练支持范围时，ICL能否成功仍不清楚。我们在温和的假设下建立了一个理论框架，将这两种分布偏移形式区分开来。对于预训练中已表示的目标任务，我们证明示例能够识别该任务，并且随着示例数量的增加，可以降低对预训练支持的提示模板的敏感性。对于不支持的指令，我们证明在固定的模型诱导参考扩展下，示例证据可以主导与N无关的指令偏移，同时保持指数级的识别速率。然后，我们使用相同的ICL框架作为……（原文在此截断）

    arXiv:2610.23528v1 Announce Type: new  Abstract: In-Context Learning (ICL) enables pretrained LLMs to adapt to downstream tasks from a small set of input-output demonstrations, without parameter updates. Yet existing theory primarily explains ICL as identifying a latent task already represented in pretraining, leaving unclear whether ICL can succeed when either the instruction or the target task falls outside the pretraining support. We develop a theoretical framework under mild assumptions that separates these two forms of distribution shift. For target tasks represented in pretraining, we show that demonstrations can identify the task and reduce sensitivity to pretrain-supported prompt templates as their number increases. For an unsupported instruction, we show that under a fixed model-induced reference extension, demonstration evidence can dominate an $N$-independent instruction shift while preserving the exponential identification rate. We then use the same ICL framework as an inte
    
[^50]: 解读用于欺骗检测的自监督学习表征：一项WavLM研究

    Interpreting SSL Representations for Spoof Detection: a WavLM Study

    [https://arxiv.org/abs/2610.23526](https://arxiv.org/abs/2610.23526)

    本研究对WavLM在语音欺骗检测中的逐层表征进行了系统的可解释性分析，发现仅选取五个关键层的组合性能优于全部十三层池化，并通过典型相关分析揭示了各声学特征与不同层之间的关联规律。

    

    诸如WavLM之类的自监督语音模型在欺骗检测任务上取得了优异的性能，然而哪些层编码了最多的欺骗相关信息、以及这些层捕获了哪些声学特性，目前仍不清楚。我们基于ASVspoof 5数据库对WavLM开展了一项逐层可解释性研究，结合了三种互补的分析方法：（i）逐层线性探测，并针对语音合成、语音转换和对抗攻击进行攻击级别的评估；（ii）典型相关分析（CCA），将各层表征与欺骗相关的声学特征建立联系；（iii）对各层子集进行详尽的网格搜索，结果表明仅使用五个层的效果优于全部十三层的池化。CCA分析揭示，虽然大多数声学特征与中间层的对齐程度达到峰值随后衰减，但语音质量指标——基频微扰、振幅微扰和谐噪比却呈现相反的趋势，其相关性随着层数加深而不断增强。

    arXiv:2610.23526v1 Announce Type: cross  Abstract: Self-supervised speech models such as WavLM achieve strong spoofing detection performance, yet which layers encode the most spoof-relevant information and what acoustic properties they capture remain opaque. We present a layer-wise interpretability study of WavLM using the ASVspoof 5 database, combining three complementary analyses: (i) per-layer linear probing with attack-level evaluation across speech synthesis, voice conversion, and adversarial attacks; (ii) a canonical correlation analysis (CCA) linking layer representations to spoofing-relevant acoustic features; (iii) an exhaustive grid search over layer subsets showing that use of five layers outperforms full thirteen-layer pooling. CCA analysis reveals that, while most acoustic features peak in alignment with mid-layers and then decay, voice quality measures jitter, shimmer, and the harmonics-to-noise ratio show the opposite trend, with correlation increasing toward later layer
    
[^51]: 面向受限空间氢气扩散预测的不确定性感知人工智能框架，以提升操作员态势感知能力

    Uncertainty-aware AI-enabled framework for predicting hydrogen dispersion in confined spaces to enhance operator situational awareness

    [https://arxiv.org/abs/2610.23515](https://arxiv.org/abs/2610.23515)

    该论文提出一个两阶段不确定性感知AI框架，通过从早期传感器读数估计氢气泄漏速率并预测氢气云团扩散（附带置信区间），为船舶受限空间加注作业中的操作员提供浓度预测图，增强态势感知并支持预防性安全决策。

    

    我们提出了一个两阶段AI赋能框架，用于增强船舶受限空间中压缩氢气加注作业期间操作员的态势感知能力。在发生泄漏事件时，该框架首先根据早期传感器读数估计未知的泄漏速率，然后预测氢气云团在未来时间段内的扩散情况，为操作员提供带有置信区间的浓度预测图。该框架基于两个独立的AI模型构建为一个AI系统。该AI系统被设计为决策支持工具，它对传感器读数进行增强，从而提升操作员的态势感知能力并支持决策制定。这使得在危险浓度达到之前就能部署预防性安全屏障，而独立的紧急停机系统仍作为最终的安全保障层。系统设计中包含了保障措施，以确保AI系统在其能力范围内运行。

    arXiv:2610.23515v1 Announce Type: new  Abstract: We present a two-stage AI-enabled framework for enhancing operator situational awareness during compressed hydrogen bunkering in maritime confined spaces. In the event of a leak, the framework first estimates the unknown leakage rate from early sensor readings and then forecasts how the hydrogen cloud will disperse over the next time steps, providing operators with a predicted concentration map together with confidence bounds. The framework is based on two independent AI models building one AI system. The AI system is designed as a decision-support tool. It augments sensor readings, and thus enhances operator situational awareness and supports decision-making. This enables preventive safety barriers to be deployed before hazardous concentrations are reached, while an independent emergency shutdown system remains the ultimate safety layer. The system design incorporates safeguards to ensure the AI system operates within its competence env
    
[^52]: 格拉斯曼流形上的自适应保形预测

    Adaptive conformal prediction on the Grassmannian

    [https://arxiv.org/abs/2610.23508](https://arxiv.org/abs/2610.23508)

    该论文提出了一个在格拉斯曼流形上构建局部自适应预测区域的保形预测框架，通过基不变的几何距离和学习到的难度估计器实现几何感知的非一致性度量，在保持有限样本覆盖保证的同时提升了不同难度下的覆盖均匀性。

    

    我们为格拉斯曼值响应的回归问题引入了一个保形预测框架。所提出的方法直接在格拉斯曼流形上以度量球的形式构建预测区域，使用基不变的几何距离作为非一致性度量，并通过学习到的预测难度估计器对相应的非一致性分数进行缩放。这产生了局部自适应、依赖输入的预测区域，这些区域尊重底层流形几何结构，同时在可交换性假设下保留了分裂保形预测的有限样本边际覆盖保证。我们评估了几种格拉斯曼距离度量，并在合成基准和翼型逆向形状设计问题上比较了它们的预测区域效率。结果凸显了几何感知非一致性分数的优势，同时局部自适应提高了不同预测难度级别下的经验覆盖均匀性。

    arXiv:2610.23508v1 Announce Type: cross  Abstract: We introduce a conformal prediction framework for regression with Grassmann-valued responses. The proposed method constructs prediction regions in the form of metric balls directly on the Grassmann manifold, using basis-invariant geometric distances as non-conformity measures and scaling the corresponding non-conformity scores through a learned prediction difficulty estimator. This yields locally adaptive, input-dependent prediction regions that respect the underlying manifold geometry, while retaining the finite-sample marginal coverage guarantee of split conformal prediction under exchangeability. We evaluate several Grassmannian distance metrics and compare their prediction-region efficiency on synthetic benchmarks and an airfoil inverse shape-design problem. The results highlight the advantages of geometry-aware non-conformity scores, while local adaptation improves empirical coverage uniformity across prediction difficulty levels 
    
[^53]: 噪声标签下学习的拓扑特征

    A Topological Signature of Learning under Noisy Labels

    [https://arxiv.org/abs/2610.23491](https://arxiv.org/abs/2610.23491)

    该论文利用持续同调追踪神经激活中的环状结构，发现噪声标签训练的模型在训练中后期比干净标签模型表现出更高的拓扑持续性，即使其预测与噪声标签几乎不一致，从而为噪声监督下的学习过程提供了一种基于表示几何的拓扑特征。

    

    噪声监督如何改变神经网络学习到的表示？训练准确率衡量的是模型与训练标签的一致性，但几乎无法提供关于表示几何结构的信息。我们使用持续同调来追踪神经激活中跨越不同距离阈值的环状结构。实验涵盖图像分类、文本分类和模运算。在固定的 CIFAR-10/ResNet-18 训练探针上，等规模的干净标签子集与损坏标签子集在第 10 个训练轮次时没有可检测到的持续性差异，而在第 50 个训练轮次时，损坏标签子集表现出更高的持续性。即使在该检查点处，损坏探针样本中仅有 7-11% 的预测与所分配的噪声标签相符，这种对比在均匀尺度控制下依然成立。归一化在 CIFAR-10/ResNet-18 上保持了干净与噪声训练轨迹之间的交叉现象，而 CIFAR-100/ResNet-18 在任何（设置下均未出现交叉）（注：原摘要在此处被截断）

    arXiv:2610.23491v1 Announce Type: cross  Abstract: How does noisy supervision change the representations learned by a neural network? Training accuracy measures agreement with training labels but provides little information about representation geometry. We use persistent homology to track loop structure across distance thresholds in neural activations. Experiments span image classification, text classification, and modular arithmetic. On a fixed CIFAR-10/ResNet-18 training probe, equally sized clean-label and corrupted-label subsets show no detectable persistence difference at epoch 10, whereas corrupted-label subsets have higher persistence at epoch 50. The contrast survives uniform-scale controls, even though predictions match the assigned noisy labels for only 7-11% of corrupted probe examples at that checkpoint. Normalisation preserves the crossover between clean and noisy training trajectories on CIFAR-10/ResNet-18, whereas CIFAR-100/ResNet-18 shows no crossover under any of the 
    
[^54]: 迈向在线主动学习的最优样本选择调度策略

    Towards optimal selection schedules for online active learning

    [https://arxiv.org/abs/2610.23477](https://arxiv.org/abs/2610.23477)

    该论文在高维在线主动学习设定下证明了最优样本查询策略的本质：贪婪最小化目标增量的短视策略等价于小间隔或中间间隔选择策略，即只标注间隔落在随时间不断收缩的区间内的样本，以在标注预算或“查询代价 Δ + 错误代价”的目标下最优。

    

    我们研究在线线性分类问题，其中源源不断的未标注样本以数据流形式到达，而人工标注代价高昂。我们研究最优查询策略，即规定样本是否以及何时应被标注并用于一步训练，或应被直接丢弃，从而最小化给定的目标。我们聚焦于可实现情形，在高斯设计下、在数据维度与未标注样本数量成比例增长的高维区域中展开分析。我们考虑两个目标：1）在给定标注预算约束下的泛化误差；2）累积代价函数，其中每次分类错误的代价为单位成本，而每次查询的代价为 $\Delta$。我们首先证明，那些在任意时刻贪婪地最小化目标增量的短视策略对应于小间隔或中间间隔策略，即选择那些间隔落在随时间收缩的特定区间内的样本。在此类策略中，w……（摘要原文至此被截断）

    arXiv:2610.23477v1 Announce Type: cross  Abstract: We consider the problem of online linear classification, where a stream of unlabeled samples is available, and annotation is costly. We investigate the optimal query policy, that prescribes whether and when a sample should be labeled and used for a training step, or instead discarded, so as to minimize a given objective. We focus on the realizable case, under Gaussian design, in the high-dimensional regime where data dimension and number of unlabeled samples grow proportionally. We consider two objectives: 1) the generalization error under a labeling budget, and 2) a cumulative cost function where mistakes have unit cost and queries cost $\Delta$. We first show that the myopic policies that greedily minimize at all times the increment in the objective correspond to either small or intermediate-margin policies, that prescribe the selection of samples whose margins fall within certain intervals that shrink with time. Within this class, w
    
[^55]: QD-Rec：基于质量-多样性的推荐开放式演化

    QD-Rec: Open-Ended Evolution of Recommendations via Quality-Diversity

    [https://arxiv.org/abs/2610.23470](https://arxiv.org/abs/2610.23470)

    提出QD-Rec方法，将质量-多样性算法、上下文老虎机与基础模型相结合，为高风险决策生成多样化、高质量且附带自然语言理由的开放式推荐。

    

    开放式搜索与基础模型相结合，已催生了诸如更快的算法、开放数学问题的新成果以及自我改进智能体等进展。我们将这一范式扩展到用于决策制定的推荐，重点关注那些备选方案集合固定但选择理由并非固定的决策。在高风险决策中，例如收购哪家公司或瞄准哪些客户，比较多样化且有充分论证的选项可能比单一的排序列表更有用。我们将开放式推荐形式化为构建一个多样化、高质量的推荐集合，其中每个推荐由一个项目和一个假设组成，该假设是用自然语言解释该项目为何适合决策者需求的理由。我们提出QD-Rec，这是一种质量-多样性方法，它使用上下文老虎机来选择项目，并利用基础模型以存档中保留的早期假设为踏脚石来生成新的假设。

    arXiv:2610.23470v1 Announce Type: cross  Abstract: Open-ended search combined with foundation models has produced advances such as faster algorithms, new results on open mathematical problems, and self-improving agents. We extend this paradigm to recommendation for decision-making, focusing on decisions where the set of possible choices is fixed but the reasons for choosing them are not. In high-stakes decisions, e.g., which company to acquire or which customer to target, comparing diverse, well-reasoned options can be more useful than a single ranked list. We formalize open-ended recommendation as building a diverse, high-quality set of recommendations, each an item together with a hypothesis, a natural-language reason why the item fits the needs of a decision maker. We propose QD-Rec, a quality--diversity method that selects items with a contextual bandit and generates hypotheses with a foundation model from earlier ones kept in an archive as stepping stones. A new recommendation ent
    
[^56]: 基于Graphlet的去中心化金融交易方法分类

    Graphlet-based Decentralized Finance Transaction Method Classification

    [https://arxiv.org/abs/2610.23466](https://arxiv.org/abs/2610.23466)

    该论文通过提取有向graphlet网络拓扑特征对DeFi中30种最常见交易方法进行分类，并利用SHAP解释揭示各类方法特有的结构模式，证明拓扑特征可有效补充事件日志计数以提升分类性能。

    

    去中心化金融交易涉及账户、智能合约和代币之间复杂的交互。我们研究网络拓扑结构是否能够识别描述DeFi操作的交易方法。我们提取有向graphlet特征，对30种最常出现的交易方法进行分类。结果表明，graphlet特征与事件日志计数相互补充，共同提升了分类性能。借助SHAP（SHapley Additive exPlanations），我们进一步获得了模型解释，揭示了不同交易方法所特有的graphlet结构模式。

    arXiv:2610.23466v1 Announce Type: cross  Abstract: Decentralized Finance (DeFi) transactions involve complex interactions among accounts, smart contracts, and tokens. We investigate whether network topology can identify transaction methods that describe DeFi operations. We extract directed graphlet features to classify the 30 most frequent methods. Our results show that graphlet features complement event log counts in improving classification performance. Using SHapley Additive exPlanations (SHAP), we further derive model explanations that reveal method-specific graphlet structural patterns.
    
[^57]: HydraLLM：面向弹性文本与视觉-语言模型的自适应宽度切分

    HydraLLM: Adaptive Width Slicing for Elastic Text and Vision-Language Models

    [https://arxiv.org/abs/2610.23455](https://arxiv.org/abs/2610.23455)

    提出HydraLLM弹性架构，通过激活注意力头数量动态调整模型各层宽度，使单一参数集能够适配从数据中心GPU到边缘设备的多种部署场景，并通过弹性投影模块将该能力扩展到视觉-语言多模态推理。

    

    大语言模型（LLM）和大型视觉-语言模型（LVLM）正日益部署在从数据中心GPU到资源受限边缘设备的异构硬件上。其固定规模的架构限制了单一模型在不同计算、延迟和模型大小需求的部署环境中复用的能力。我们提出HydraLLM，一种基于弹性MQA（多查询注意力）的架构。其激活的注意力头数量决定了嵌入层、注意力投影、前馈层和输出头的有效宽度，从而在单一参数集内形成一个嵌套的子网络家族。我们引入了一套从静态到弹性的转换流水线，从预训练模型检查点出发，包含基于重要性的权重重排序以及结合知识蒸馏的随机子网络训练。我们通过一个弹性投影模块将该原则扩展到多模态推理，该模块的输出宽度跟随激活的文本宽度。HydraLLM覆盖了更广泛的……

    arXiv:2610.23455v1 Announce Type: new  Abstract: LLMs and LVLMs are increasingly deployed across heterogeneous hardware, ranging from data-center GPUs to resource-constrained edge devices. Their fixed-size architectures limit the ability to reuse a single model across deployment settings with different compute, latency, and model-size targets. We introduce HydraLLM, an elastic MQA-based architecture. Its number of active attention heads determines the effective widths of the embeddings, attention projections, feed-forward layers, and the output head, forming a nested family of subnetworks within a single parameter set. We introduce a static-to-elastic conversion pipeline that, starting from a pre-trained model checkpoint, includes importance-driven weight reordering and stochastic subnetwork training with knowledge distillation. We extend this principle to multimodal inference through an elastic projection module whose output width follows the active text width. HydraLLM covers a wider
    
[^58]: 可预测扩展：基于扩展定律预测的高效模型探索

    Predictable Scaling: Efficient Model Exploration via Scaling-Law Prediction

    [https://arxiv.org/abs/2610.23453](https://arxiv.org/abs/2610.23453)

    该论文提出“可预测扩展”（PS）框架，通过沿资源轴构建小规模代理模型并结合曲线拟合与不确定性校准，从代理实验预测全规模推荐模型性能，从而将原本需数千GPU小时的模型改进假设评估变得高效可行。

    

    评估推荐模型的候选变更（假设）需要进行全规模训练，消耗数千GPU小时，严重限制了工程师能够实际探索的假设空间。我们提出了可预测扩展，这是一个系统性框架，利用经过实证验证的规律性条件，从小规模代理模型预测全规模模型的性能。PS通过沿不同资源轴的可组合扩展函数构建缩小版模型，并通过带有校准不确定性估计的曲线拟合将指标投射到目标规模。我们比较了三种表达能力递增的预测策略：即使是最简单的（原始代理）也能达到与重复运行方差的经验噪声水平相当的预测误差；表达能力最强的（高斯过程）则提供了15%的相对MAE改进和校准的不确定性估计。在两个复杂度递增的排序模型上的评估——一个……

    arXiv:2610.23453v1 Announce Type: new  Abstract: Evaluating candidate changes (hypotheses) to recommendation models requires full-scale training runs consuming thousands of GPU-hours, severely limiting the hypothesis space engineers can feasibly explore. We introduce Predictable Scaling (PS), a systematic framework that exploits empirically verified regularity conditions to predict full-scale model performance from small-scale proxies. PS constructs scaled-down variants via composable scaling functions along distinct resource axes, and projects metrics to the target scale via curve fitting with calibrated uncertainty estimates. We compare three prediction strategies of increasing expressiveness: even the simplest (raw proxy) achieves prediction error within the empirical noise floor of repeated-run variance; the most expressive (Gaussian Process) provides an indicative 15% relative MAE improvement and calibrated uncertainty. Evaluation on two ranking models of increasing complexity---a
    
[^59]: 用于高维粒子滤波器中重采样失真谱修复的正则化自由反卷积

    Regularized free deconvolution for spectral repair of resampling distortion in high-dimensional particle filters

    [https://arxiv.org/abs/2610.23448](https://arxiv.org/abs/2610.23448)

    该论文提出一种正则化自由反卷积估计器，仅凭重要性权重和经验谱即可恢复高维粒子滤波器中因重采样而失真的协方差谱，并将其作为预条件子用于 Metropolis 修正的复活移动，在不引入偏差的情况下提升滤波器的混合效率。

    

    重采样是解决粒子滤波器中权重退化问题的标准方法，但在高维情形下，它会以一种可以从重要性权重计算出来的方式扭曲集合协方差的谱。本文研究由此产生的逆问题：仅凭观测到的权重和经验谱恢复重采样前的谱，并在不使滤波器产生偏差的情况下加以使用。我们首先证明，一旦以权重及其实际产生的后代数量共同作为条件（而不仅仅是权重本身），前向谱定律便可推广到随机重要性权重的情形。随后，我们构造了一个正则化自由反卷积估计器，证明了其在固定分辨率尺度下的一致性并给出显式误差预算，同时通过一个可计算的前向残差来检验模型失配。修复后的谱仅作为 Metropolis 修正的复活移动的预条件子进入滤波器，因此谱误差只会影响混合效率……

    arXiv:2610.23448v1 Announce Type: new  Abstract: Resampling is the standard remedy for weight degeneracy in particle filters, but in high dimensions it distorts the spectrum of the ensemble covariance in a way that is computable from the importance weights. This paper studies the inverse problem: recovering the pre-resampling spectrum from the observed weights and the empirical spectrum alone, and using it without biasing the filter. We first show that the forward spectral law extends to random importance weights once one conditions on the weights together with the realized offspring counts, rather than on the weights alone. We then construct a regularized free-deconvolution estimator, prove its consistency with an explicit error budget at a fixed resolution scale, and certify model mismatch through a computable forward residual. The repaired spectrum enters the filter only as the preconditioner of a Metropolis-adjusted rejuvenation move, so that spectral error affects mixing efficienc
    
[^60]: RECREG：强化学习中针对正则安全性质的基于恢复的屏蔽方法

    RECREG: Recovery-based Shielding for Regular Safety Properties in Reinforcement Learning

    [https://arxiv.org/abs/2610.23445](https://arxiv.org/abs/2610.23445)

    RecReg是一个通过基于恢复的屏蔽机制强制执行非马尔可夫正则安全性质的安全强化学习框架，并首次提供了严格的安全性和收敛性理论保证。

    

    在许多现实场景中，部署强化学习（RL）智能体时需要考虑形式化要求，例如安全性和约束合规性。本文提出了RecReg，一个安全强化学习框架，能够通过基于恢复的屏蔽机制强制执行非马尔可夫的正则安全性质。RecReg是一种统一的方法，支持不同层次的先验知识和安全保证。本文主要侧重于在表格型设置中为RecReg构建理论基础，提供了现有相关工作所不具备的严格的安全性和收敛性保证。我们使用MASA-Safe-RL库提供了大量的实证结果，并概述了利用最先进的基于模型的强化学习将RecReg扩展到深度强化学习领域的方案。实验结果表明，RecReg实现了高效学习，且与本文的理论相吻合。

    arXiv:2610.23445v1 Announce Type: cross  Abstract: In many real-world scenarios we need to consider formal requirements, such as safety and constraint compliance, to deploy Reinforcement Learning (RL) agents. In this paper we introduce RecReg, a safe RL framework capable of enforcing non-Markovian regular safety properties via recovery-based shielding. RecReg is a unified approach that supports different levels of a priori knowledge and safety guarantees. The paper primarily focuses on building the theoretical foundations for RecReg in the tabular setting, providing rigorous safety and convergence guarantees that are not present in related work. We provide an extensive set of empirical results using the MASA-Safe-RL library, and we outline an extension of RecReg to deep RL domains using SOTA model-based RL. Our empirical results demonstrate that RecReg achieves efficient learning and matches the theory in this paper.
    
[^61]: 从观测数据中进行生成式策略改进

    Generative Policy Improvement from Observational Data

    [https://arxiv.org/abs/2610.23436](https://arxiv.org/abs/2610.23436)

    本文提出 PolicyLift，一个三阶段框架，通过学习捕捉高层策略的离散动作码，在动作重叠有限的条件下，从观测数据中实现对文本、图像等结构化动作的生成式策略改进。

    

    我们研究如何从观测数据中改进决策策略，以便在动作是复杂结构化对象（如文本消息或图像）时生成更好的动作。该任务面临两大主要挑战：(i) 动作级别的重叠往往太弱，难以支持可靠的离线策略评估与改进；(ii) 生成新的结构化动作通常需要昂贵的训练或微调（例如对大语言模型），并且由于动作表示中的信息丢失，问题定义可能不明确。为同时解决这两个问题，我们提出了 PolicyLift，一个基于观测数据、面向结构化动作的生成式策略改进三阶段框架。在第一阶段，PolicyLift 学习一小组离散动作码，这些动作码捕捉观测动作空间中的高层策略，并经过专门设计以在有限重叠条件下支持策略改进。在第二阶段，它学习一个基于这些动作码的、依赖上下文的策略。在部署时，第三阶段“提升”（原文截断）……

    arXiv:2610.23436v1 Announce Type: new  Abstract: We study how to improve decision policies from observational data to generate better actions when actions are complex, structured objects, such as text messages or images. This task is subject to two major challenges: (i) action-level overlap is often too weak for reliable off-policy evaluation and improvement, and (ii) generating new structured actions often requires expensive training or fine-tuning (e.g., of an LLM) and can be ill-defined due to information loss in action representations. To address both, we propose PolicyLift, a three-stage framework for generative policy improvement with structured actions from observational data. In Stage 1, PolicyLift learns a small set of discrete action codes that capture high-level strategies in the observed action space and are tailored to support policy improvement under limited overlap. In Stage 2, it learns a context-dependent policy over these action codes. At deployment time, Stage 3 "lif
    
[^62]: 在联邦学习中通过谱方法桥接压缩通信与本地步骤

    Bridging Compressed Communication and Local Steps via Spectral Method in Federated Learning

    [https://arxiv.org/abs/2610.23434](https://arxiv.org/abs/2610.23434)

    该论文提出了一种同时支持本地更新与通信压缩的新型谱联邦学习方法，并在无需有界异构性这一限制性假设的温和条件下，给出了严格的收敛性保证。

    

    arXiv:2610.23434v1 公告类型：new 摘要：联邦学习（FL）能够在保持数据本地化的同时，在大规模边缘设备网络上协同训练复杂模型。然而，数据异构性和高通信开销仍然是根本性的瓶颈。最近，谱优化器Muon在集中式训练中展现出了相较于Adam类方法的显著优势。尽管近期已有工作尝试将Muon迁移到联邦学习中，但将其简单地整合到传统架构中效果不佳。关键在于，现有方法不支持通信压缩和本地步骤——这两项是高效联邦学习的基础技术。为了解决这一局限，我们提出了一种新颖的谱联邦学习方法，它同时利用本地更新和压缩技术。我们在温和的假设下建立了严格的收敛性保证，特别是消除了限制性的有界异构性条件。在标准图像分类基准上的实验评估（摘要在此处被截断）……

    arXiv:2610.23434v1 Announce Type: new  Abstract: Federated Learning (FL) enables collaborative training of complex models across large-scale networks of edge devices while keeping data localized. However, data heterogeneity and high communication overhead remain fundamental bottlenecks. Recently, the spectral optimizer Muon has demonstrated remarkable advantages over Adam-like methods in centralized training. While recent efforts have attempted to transition Muon to FL, naive integrations into traditional architectures fail short. Crucially, existing approaches do not support the the communication compression and local steps --- the two foundational techniques of efficient FL. To address this limitation, we introduce a novel spectral FL method that utilizes both local updates and compression. We establish rigorous convergence guarantees under mild assumptions, notably eliminating the restrictive bounded heterogeneity condition. Empirical evaluation on standard image classification benc
    
[^63]: 残差随机概念瓶颈模型

    Residual Stochastic Concept Bottleneck Models

    [https://arxiv.org/abs/2610.23421](https://arxiv.org/abs/2610.23421)

    提出残差随机概念瓶颈模型，通过多元高斯分布联合建模概念与残差变量并学习其协方差依赖结构，在缓解概念不完备问题的同时避免无约束残差路径削弱概念干预的有效性。

    

    基于概念的模型（CMs）学习人类可解释的中间表示，使用户能够理解模型的推理过程，并通过测试时的概念干预来引导其预测。然而，在现实场景中，其性能可能受限于概念不完备性，即训练概念未能捕获所有与任务相关的信息。在模型的预测过程中添加一条不受约束的残差路径可以克服不完备性带来的影响，但这会引入一条难以解释且易于信息泄露的预测通道，可能削弱概念干预的效果，因为概念干预通常不会改变残差变量。为解决这一局限，我们提出了残差随机概念瓶颈模型，这是一族通过多元高斯分布联合建模概念变量与残差变量的基于概念的模型。Res-SCBMs 学习一种协方差结构，显式地捕获（概念与残差之间的）依赖关系。

    arXiv:2610.23421v1 Announce Type: new  Abstract: Concept-based models (CMs) learn human-interpretable intermediate representations that enable users to understand the model's reasoning and steer its predictions through test-time concept interventions. However, in realistic settings, their performance can be limited by concept incompleteness, where the training concepts fail to capture all task-relevant information. Adding an unconstrained residual pathway to the model's predictive process can overcome the consequences of incompleteness. However, this introduces an inscrutable, leak-prone prediction pathway that may weaken concept interventions, as these often leave residual variables unchanged. To address this limitation, we introduce Residual Stochastic Concept Bottleneck Models (Res-SCBMs), a family of concept-based models that jointly model concept and residual variables through a multivariate Gaussian distribution. Res-SCBMs learn a covariance structure that explicitly captures dep
    
[^64]: Flash RNN-T：内存高效的联合网络与转导损失计算

    Flash RNN-T: Memory-Efficient Joint Network and Transducer Loss Computation

    [https://arxiv.org/abs/2610.23409](https://arxiv.org/abs/2610.23409)

    提出 Flash RNN-T，通过专用内核与分块重计算实现内存高效的 RNN-T 训练，在 H100 上相比最快基线吞吐量提升 24-39%、内存减少 26-56%，分块模式还可将内存占用进一步降低至九分之一。

    

    arXiv:2610.23409v1 公告类型：新论文 摘要：循环神经网络转导器（RNN-T）仍然是流式语音识别的实用选择，但其训练需要大量的中间张量。我们提出了 Flash RNN-T，这是一种联合网络与损失的实现方法，结合了专用内核以及在完整对齐空间和词表上进行可选分块重计算的技术。在中型和长型工作负载上，分块与非分块 Flash 构成了完整的实测吞吐量-内存权衡边界。在 H100 上，在这些工作负载的内存可容纳范围内，非分块 Flash 相比最快基线实现了 24-39% 的更高吞吐量，同时内存使用减少 26-56%。分块技术通过重计算进一步降低内存占用。在中等工作负载上，当两者都采用每批次一次反向调用时，分块 Flash 比该基线节省九倍的内存，但吞吐量降低 16%。在长工作负载上，当所有使用相同反向调度且内存限制为 80 GB 的测试替代方案都耗尽内存时，它仍能正常运行。

    arXiv:2610.23409v1 Announce Type: new  Abstract: The recurrent neural network transducer (RNN-T) remains a practical choice for streaming speech recognition, but training requires large intermediate tensors. We present Flash RNN-T, a joint and loss implementation combining specialized kernels with optional tiled recomputation over the full alignment space and vocabulary. Tiled and untiled Flash form the entire measured throughput-memory frontier on medium and long workloads. On H100, untiled Flash achieves 24-39% higher throughput with 26-56% less memory than the fastest baseline on these workloads wherever it fits. Tiling reduces memory further through recomputation. On medium, tiled Flash uses ninefold less memory than that baseline at 16% lower throughput when both use one backward call per batch. On long, it fits where all tested alternatives with that backward schedule run out of memory with 80 GB. These comparisons use each implementation's lowest-memory configuration reaching at
    
[^65]: 基于离散扩散的信息量估计

    Information Estimation with Discrete Diffusion

    [https://arxiv.org/abs/2610.23404](https://arxiv.org/abs/2610.23404)

    提出了基于离散扩散的InfoSEDD方法，将信息论估计与生成建模相结合，为离散数据的互信息等信息度量估计提供了一种轻量且可扩展的新途径。

    

    互信息等信息论度量在理解随机变量之间的非线性关系方面发挥着关键作用，并被广泛应用于各个科学学科。然而，它们在现实世界离散数据上的应用仍然充满挑战。现有方法通常依赖于将离散数据嵌入到连续空间中，然后应用最初为连续分布设计的神经估计器。这一过程需要对嵌入模型和估计器架构进行精心的工程设计，且饱受高数据维度相关问题的困扰。在本工作中，我们提出了InfoSEDD，一种基于离散扩散的方法，它将信息论估计与生成建模相连接，使其能够用于计算Kullback-Leibler散度。在连续时间马尔可夫链理论原理的支持下，InfoSEDD的设计轻量且可扩展，并能够与预训练模型无缝集成（摘要在此处截断）。

    arXiv:2610.23404v1 Announce Type: new  Abstract: Information-theoretic measures, such as Mutual Information (MI), play a crucial role in understanding non-linear relationships between random variables and are widely used across scientific disciplines. Yet, their use on real-world discrete data remains challenging. Existing methods typically rely on embedding discrete data into a continuous space and apply neural estimators originally designed for continuous distributions. This process requires careful engineering for both the embedding model and estimator architecture, but suffers from issues related to high data dimensionality. In this work, we introduce InfoSEDD, a discrete diffusion-based approach that bridges information-theoretic estimation and generative modeling such that they can be used to compute Kullback-Leibler divergences. Backed by Continuous Time Markov Chains theory principles, the design of InfoSEDD is lightweight and scalable and allows seamless integration with pretr
    
[^66]: 模型失配下与代理人的合同设计

    Contracting with an Agent under Model Mismatch

    [https://arxiv.org/abs/2610.23402](https://arxiv.org/abs/2610.23402)

    该论文在委托-代理马尔可夫决策过程（PAMDP）中刻画了代理人因模型失配而持有错误信念时的合同设计问题，并提出了一个框架，使掌握真实环境动态的委托人能够通过合同设计引导代理人产生理想的长期行为。

    

    在经典的委托-代理设定中，委托人将某项任务委托给代理人，代理人根据任务结果获得报酬，而该结果又为委托人带来收益。为了使双方目标一致，委托人设计合同以激励代理人采取有利于委托人的行动。然而在实践中，代理人可能基于对底层环境的错误认知而采取行动，从而导致系统性的次优决策。在本研究中，我们通过一个委托-代理马尔可夫决策过程（PAMDP）来研究这一问题，其中委托人与一个转移模型可能与真实环境动态不一致的代理人进行反复交互。委托人掌握真实的转移模型，并设计能够兼顾代理人信念与激励的合同。我们刻画了模型失配情形下的合同设计，并开发了一个使委托人能够引导代理人产生理想长期行为的框架。

    arXiv:2610.23402v1 Announce Type: new  Abstract: In the classical principal-agent setting, the principal delegates some task to the agent, who receives a payment conditioned on the outcome of the task, which in turn benefits the principal. To align their objectives, the principal designs a contract that incentivizes the agent to take actions beneficial to the principal. In practice, however, the agent may act under misspecified beliefs about the underlying environment, leading to systematically suboptimal decisions. In this work, we study this problem through a principal-agent Markov Decision Process (PAMDP), in which the principal repeatedly interacts with an agent whose transition model may differ from the true environment dynamics. The principal has access to the true transition model and designs contracts that account for the agent's belief and incentives. We characterize contract design under model mismatch and develop a framework that enables the principal to induce desirable lon
    
[^67]: 使用具有Lipschitz控制和渐进核判别器策略的WGAN模型在异常情况下预测电压时间序列

    Forecasting Voltage Time Series under Anomalies Using a WGAN Model With Lipschitz Control and Progressive Kernel Critic Strategy

    [https://arxiv.org/abs/2610.23394](https://arxiv.org/abs/2610.23394)

    提出了一种结合渐进Lipschitz惩罚与自适应核的Wasserstein GAN模型（WGAN-CLP），用于在异常情况下对高维流电压时间序列进行递归预测和动态建模。

    

    在时间序列预测任务中，对高维数据使用递归预测会大幅增加预测过程的复杂性。主要挑战在于长期动态更难建模，因为序列表现出更大的时间变异性。为了克服这一局限并改进长期动态建模，更重要的是解决高维递归预测问题，本文采用了一种配备渐进Lipschitz惩罚和自适应核的Wasserstein生成对抗网络（GAN）。该模型能够接收数据流的一个窗口并预测后续数值，随后重建时间序列流，并针对每个新窗口动态调整其核。所提出的模型是一个基于Wasserstein GAN（WGAN-CLP）并结合自适应核的深度学习框架，用于对流电压时间序列中的变化进行建模。为了建立方法论基础，我们（摘要原文在此处截断）。

    arXiv:2610.23394v1 Announce Type: new  Abstract: In time-series forecasting tasks, the use of recursive forecasting on high-dimensional data substantially increases the complexity of the prediction process. The main challenge is that long-term dynamics becomes harder to model, since the sequences show greater temporal variability. To counteract this limitation and improve the modeling of long-term dynamics, and more importantly, address high-dimensional recursive forecasting, this paper uses a Wasserstein GAN equipped with a progressive Lipschitz penalty and an adaptive kernel. This model is capable of taking a window of the data stream and forecasting the next values, subsequently reconstructing the time-series stream and dynamically adapting its kernel to each new window. The proopsed model is a deep learning framework based on a Wasserstein GAN (WGAN-CLP) with an adaptive kernel to model alterations in streaming voltage time series. To establish the methodological foundation, we rev
    
[^68]: 基于词汇与上下文证据的句子级主题建模

    Sentence-Level Topic Modeling with Lexical and Contextual Evidence

    [https://arxiv.org/abs/2610.23388](https://arxiv.org/abs/2610.23388)

    SentenceETM提出了一种句子级神经主题模型，通过联合利用词汇分布与句子嵌入上下文这两种证据来源，有效解决了主题建模中词级模型缺乏上下文、文档级模型忽视多主题共存问题的粒度困境。

    

    主题建模面临一个根本性的粒度困境：概率主题模型在词级别分配主题，而基于Transformer的方法则在文档级别进行操作。后者忽视了现实世界中的文档通常涵盖多个主题这一事实，而前者依赖于单个词语，但单个词语很少能提供足够识别主题所需的上下文。我们提出了SentenceETM，这是一种神经主题模型，通过联合利用词汇证据和上下文证据，为每个句子分配一个主题。每个主题同时由两种方式描述：一是通过词嵌入定义的词表上的分布，二是集中于主题特定低维子空间的句子嵌入上的高斯分布。句子的主题同时生成其词语和句子嵌入，并被精确地边缘化，使其后验分布能够结合这两种证据来源。文档的主题比例通过摊销变分推断来推断，而高斯……（原文摘要至此截断）

    arXiv:2610.23388v1 Announce Type: new  Abstract: Topic modeling faces a fundamental granularity dilemma: probabilistic topic models assign topics at the word level, whereas transformer-based methods operate at the document level. The latter overlooks that real-world documents typically cover multiple topics, whereas the former relies on individual words, which rarely provide sufficient context to identify them. We introduce SentenceETM, a neural topic model that assigns a topic to each sentence by jointly using lexical and contextual evidence. Each topic is described both by a distribution over the vocabulary, defined through word embeddings, and by a Gaussian over sentence embeddings concentrated on a topic-specific low-dimensional subspace. The topic of a sentence generates both its words and its embedding and is marginalized exactly, so that its posterior combines the two sources of evidence. Document proportions are inferred with amortized variational inference, while the Gaussian 
    
[^69]: 基于数据驱动的AMD EPYC集群上内存受限CFD的最优MPI进程放置与节点数预测

    Data-Driven Prediction of Optimal MPI Rank Placement and Node Count for Memory-Bound CFD on AMD EPYC Clusters

    [https://arxiv.org/abs/2610.23383](https://arxiv.org/abs/2610.23383)

    该论文基于560个实测配置的数据驱动模型，预测AMD EPYC集群上内存受限CFD的最优MPI进程放置与节点数，发现小网格在单节点上最优、大网格因超线性加速使最大节点数成为最优选择，仅中等网格存在性能与成本的真实权衡（全速运行需多花费20%至97%的节点小时）。

    

    在AMD EPYC集群上运行内存受限的OpenFOAM算例的用户必须决定使用多少节点，以及在每个核心复合体（CCX）上放置多少个MPI进程。最快的选择往往不是最经济的选择。本文通过在五种Zen 4和Zen 5 EPYC处理器上对motorBike算例进行560次测量配置（包括来自处理器遥测数据的插槽功耗），重新审视了我们早期关于这种性能与能耗权衡的分析模型。在最多四个节点的规模下，小网格在大多数处理器上于单节点上既最快又最便宜。对于最大的网格，最大节点数既是最快也是最便宜的选择，因为其加速比呈现超线性特征。只有中等大小的网格才会出现真正的权衡，在这种情况下，全速运行比最经济的方案需要多花费20%到97%（中位数）的节点小时。L3缓存缺失率和每单元的halo通信流量与这一模式相符。对于同一算例的新网格，先前的最优配置……

    arXiv:2610.23383v1 Announce Type: cross  Abstract: Users who run a memory-bound OpenFOAM case on an AMD EPYC cluster must decide how many nodes to use and how many MPI ranks to place on each core complex. The fastest choice is often not the cheapest one. This paper revisits our earlier analytical model of this performance and energy trade-off with 560 measured configurations of the motorBike case on five Zen 4 and Zen 5 EPYC processors, including socket power from processor telemetry. On up to four nodes, small meshes are both fastest and cheapest on a single node on most processors. For the largest mesh the largest node count is both the fastest and the cheapest option, because its speedup is superlinear. A genuine trade-off arises only for medium meshes, where full speed costs a median of 20 to 97% more node hours than the cheapest option. The L3 miss rate and the halo traffic per cell are consistent with this pattern. For a new mesh of the same case, the configuration that was optim
    
[^70]: 超越成功率：视觉-语言-动作系统中记忆的干预性评估

    Beyond Success Rate: Interventional Evaluation of Memory in Vision-Language-Action Systems

    [https://arxiv.org/abs/2610.23372](https://arxiv.org/abs/2610.23372)

    该论文提出干预性记忆评估协议（IMEP），通过目标事实替换等干预操作并将决策正确性与执行成功率分开评估，以严格检验记忆在视觉-语言-动作系统中是否真正发挥作用。

    

    当我们说一个机器人使用了记忆时，这段记忆是否真的发挥了作用？仅凭成功率的消融实验无法确立对任务相关事实的利用：这类实验在改变信息的同时也改变了计算，而总体成功率可能掩盖相互抵消的收益与损失。我们提出了一种面向视觉-语言-动作（VLA）系统的干预性记忆评估协议（IMEP），以回答四个问题：(1) 记忆是否带来净收益？(2) 在什么条件下记忆有帮助？(3) 对记忆的改动是否会改变任务结果？(4) 这些效应是否特定于任务相关的事实？在检索接口处，我们将目标事实替换与保留事实的替换、最小数值扰动以及精确副本进行比较，同时将决策正确性与运动执行成功率分开评估。我们评估了回合内记忆、跨回合记忆以及智能体式记忆。在跨回合学习库实验中，一个末位单位（UL……

    arXiv:2610.23372v1 Announce Type: cross  Abstract: When we say that a robot uses memory, does that memory actually play a role? Success-rate ablations alone cannot establish the use of task-relevant facts: they change information and computation together, while aggregate rates can hide offsetting gains and losses. We propose an Interventional Memory Evaluation Protocol (IMEP) for vision-language-action (VLA) systems to address four questions: (1) Does memory provide a net benefit? (2) Under what conditions does memory help? (3) Do changes to memory alter task outcomes? (4) Are these effects specific to task-relevant facts? At the retrieval interface, we compare target-fact substitutions with fact-preserving replacements, minimal numerical perturbations, and exact replicas, while assessing decision correctness separately from motor execution success. We evaluate within-episode, cross-episode, and agentic memory. In cross-episode learned-bank experiments, a one-unit-in-the-last-place (UL
    
[^71]: RTSM：将递归模型适配到多元时间序列分类

    RTSM: Adapting Recursive Models to Multivariate Time-Series Classification

    [https://arxiv.org/abs/2610.23353](https://arxiv.org/abs/2610.23353)

    RTSM通过结合可学习的卷积时间分词器、单状态权重共享精炼和池化分类读出，在30个UEA多元时间序列分类数据集上取得72.81%的平均准确率，在17个深度学习模型中排名第一。

    

    准确的多元时间序列分类需要提取时间特征并整合整个序列中的信息。权重共享迭代精炼提供了一种重复处理这些特征而无需在每个阶段分配新参数的方法。将这一原理应用于时间序列，需要同时选择适合分类的时间表示形式和潜在状态组织方式。我们提出递归时间序列模型（RTSM），它结合了可学习的卷积时间分词器、单状态权重共享精炼机制以及池化分类读出层。在全部30个UEA数据集上，RTSM在基于验证集的配置与检查点选择下取得了72.81%的平均准确率，这是所评估的17个深度学习模型中最高的平均准确率，比MambaSL高出1.60个百分点。在八个数据集上的消融实验表明，所提出的分词器优于简单的卷积嵌入，单状态模型优于（原文在此处截断）。

    arXiv:2610.23353v1 Announce Type: new  Abstract: Accurate multivariate time-series classification requires extracting temporal features and integrating information across a sequence. Weight-tied refinement offers a way to repeatedly process these features without allocating new parameters at each stage. Applying this principle to time series requires choosing both a temporal representation and a latent-state organization suited to classification. We introduce the Recursive Time-Series Model (RTSM), which combines a learned convolutional temporal tokenizer, single-state weight-tied refinement, and a pooled classification readout. Across all 30 UEA datasets, RTSM achieves 72.81% mean accuracy under validation-based configuration and checkpoint selection, the highest observed mean among the 17 evaluated deep-learning models and 1.60 percentage points above MambaSL. Ablations on eight datasets favor the proposed tokenizer over a simple convolutional embedding and the single-state model ove
    
[^72]: 有限秩瓶颈如何塑造低秩自适应的优化景观

    How Finite-Rank Bottlenecks Shape the Low-Rank Adaptation Landscape

    [https://arxiv.org/abs/2610.23343](https://arxiv.org/abs/2610.23343)

    本文提出了一种无需线性化的新理论框架，证明冻结主干后损失仅通过有限秩的“瓶颈”映射感知适配矩阵，因而迹范数正则化下的全局最优解秩由瓶颈秩而非样本量决定，从而解释了LoRA低秩更新为何足够有效。

    

    低秩自适应（LoRA）已成为大型预训练模型参数高效微调的标准方法，然而对于为什么低秩更新就足够，现有理论解释仍不完整。大多数已有分析依赖于神经正切核（NTK）机制，即在预训练权重附近对网络进行线性化。在本工作中，我们采取了一种无需线性化的不同方法：我们的关键观察是，一旦主干网络被冻结，损失函数只能通过由冻结激活所诱导的有限秩线性映射来感知每个被适配的权重矩阵。迹范数正则化随后迫使优化器在所有产生相同观测输出的矩阵中选择一个低秩矩阵。这使得全局最优解的秩由该映射的秩（我们称之为“瓶颈秩”）控制，而非由样本量控制。对于多头注意力，我们通过利用两个代数性质获得了更紧的秩上界……

    arXiv:2610.23343v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) has become the standard method for parameter-efficient fine-tuning of large pretrained models, yet theoretical explanations for why low-rank updates suffice remain incomplete. Most existing analyses rely on the Neural Tangent Kernel (NTK) regime, which linearizes the network around the pretrained weights. In this work, we take a different approach that requires no linearization: our key observation is that, once the backbone is frozen, the loss sees each adapted weight matrix only through a finite-rank linear map induced by the frozen activations. Trace-norm regularization then forces the optimizer to pick a low-rank matrix among all matrices that produce the same observed output. This yields global minimizers whose ranks are controlled by the rank of this map, which we call the \emph{bottleneck rank}, rather than the sample size. For multi-head attention, we obtain tighter rank upper bounds by exploiting two a
    
[^73]: SETAREH变换：一种用于近似内积计算的排序辅助Haar小波稀疏化方法

    SETAREH Transform: A Sorting-Assisted Haar-Wavelet Sparsification Method for Approximate Inner-Product Computation

    [https://arxiv.org/abs/2610.23338](https://arxiv.org/abs/2610.23338)

    本文提出SETAREH变换，一种利用排序辅助实现能量集中的Haar小波稀疏化方法，可高效近似固定高维系数向量与连续输入向量之间的内积，从而显著降低高维向量运算中的存储、内存流量与通信开销。

    

    arXiv:2610.23338v1 公告类型：新论文 摘要：现代深度学习系统日益依赖高维向量运算，其中大型输入向量与参数向量需要被反复存储、传输和处理。尽管已有大量研究致力于通过剪枝、量化、知识蒸馏和高效架构来压缩神经网络模型规模，但与反复出现的高维输入向量相关的存储、内存流量和通信成本仍是重要的实际问题，在分布式和边缘计算环境中尤其如此。本文提出了SETAREH（基于Haar小波的排序辅助能量集中近似表示变换）变换，这是一种基于Haar小波的稀疏化方法，用于近似固定高维系数向量与连续输入向量之间的内积。通过对两者施加相同的排列序列和Haar变换（摘要在此处被截断）

    arXiv:2610.23338v1 Announce Type: new  Abstract: Modern deep learning systems increasingly rely on high-dimensional vector operations, in which large input and parameter vectors are repeatedly stored, transferred, and processed. Although substantial research has focused on reducing neural-network model size through pruning, quantization, knowledge distillation, and efficient architectures, the storage, memory-traffic, and communication costs associated with repeatedly presented high-dimensional input vectors remain important practical considerations, particularly in distributed and edge-computing environments. This paper proposes the SETAREH (Sorting-assisted Energy-concentrating Transform for Approximate Representation using Haar wavelets) transform, a Haar-wavelet-based sparsification approach for approximating the inner product between a fixed high-dimensional coefficient vector and successive input vectors. By applying the same sequence of permutations and Haar transforms to both v
    
[^74]: 基于迁移学习的岭回归中的极小极大最优标度律

    Minimax-optimal Scaling Laws in Ridge Regression via Transfer Learning

    [https://arxiv.org/abs/2610.23337](https://arxiv.org/abs/2610.23337)

    本文在高维岭回归迁移学习框架下建立了风险确定性等价，并证明在共享回归函数和可交换幂律协方差谱的条件下可实现极小极大最优标度律，且最优源正则化仅依赖源数据、具有跨任务普适性。

    

    预训练模型常被跨下游任务复用，这引出一个自然的问题：这种迁移学习过程能否带来显著的性能提升，即体现在标度律方面。我们在高维回归的典型设定中研究这一问题：其中预训练的源模型通过岭回归训练，而微调后的目标模型通过与源估计器之间的 L2 距离进行正则化。首先，我们在同时存在模型偏移和协方差偏移的情形下，为该两步过程的风险建立了确定性等价表示。其次，通过将确定性等价特化到共享回归函数以及可交换的幂律协方差谱的情形，我们识别出一个区域，在该区域中所得到的标度律达到了极小极大最优指数。值得注意的是，在该区域中，最优的源正则化仅依赖于源数据，这表明其具有跨任务的普适性。

    arXiv:2610.23337v1 Announce Type: new  Abstract: Pretrained models are often reused across downstream tasks, motivating the question of whether this transfer learning procedure can provide a substantial performance improvement, namely, in terms of the scaling law. We address this question in the prototypical setting of high-dimensional regression, where (i) a pretrained \emph{source} model is trained via ridge regression, and (ii) a fine-tuned \emph{target} model is regularized via the $L_2$ distance from the source estimator. First, we establish a deterministic equivalent for the risk of this two-step procedure under both model and covariance shift. Next, by specializing the deterministic equivalent to a shared regression function and commuting power-law covariance spectra, we identify a regime where the resulting scaling law achieves the minimax-optimal exponent. Notably, in this regime, the optimal source regularization depends on source data only, indicating its universality across
    
[^75]: HiPredict：利用机器学习预测HiRISE缺失的RED4光谱通道

    HiPredict: Predicting the missing HiRISE RED4 spectral channel with machine learning

    [https://arxiv.org/abs/2610.23335](https://arxiv.org/abs/2610.23335)

    本文提出HiPredict方法，利用机器学习捕捉颜色通道间的非线性关系，从而更准确地重建HiRISE仪器因RED4探测器故障而缺失的光谱通道数据。

    

    搭载于火星勘测轨道器（MRO）上的高分辨率成像科学实验（HiRISE）以25-30厘米/像素的分辨率获取火星影像，支持火星地质、气候历史和着陆点表征等研究。其14探测器焦平面组件包含多个重叠探测器，可获取三通道彩色红外图像，这对于识别成分变化和研究活跃的表面过程十分重要。2023年，焦平面组件中心的一个探测器（RED4）开始间歇性故障，在获取的影像中产生了缺失区域，使仪器的彩色成像能力降低了一半。HiRISE处理流程目前通过对共址的红外（IR10）通道进行线性缩放来提供合成红色（SYN4）通道。然而，这种方法无法捕捉颜色通道之间观察到的非线性关系，限制了其准确性和应用效果。

    arXiv:2610.23335v1 Announce Type: new  Abstract: The High Resolution Imaging Science Experiment (HiRISE) onboard the Mars Reconnaissance Orbiter (MRO) acquires imagery of Mars at 25--30 cm/pixel, supporting studies of martian geology, climate history, and landing-site characterization. Its 14-detector focal plane assembly includes several overlapping detectors that enable the acquisition of three-channel colour-infrared images, which are important for identifying compositional variations and studying active surface processes. In 2023, a detector in the center of the focal plane assembly (RED4) began failing intermittently, creating a gap in acquired imagery and reducing the instrument's colour-imaging capability by half. The HiRISE processing pipeline provides a synthetic red (SYN4) channel obtained by linearly scaling the co-located infrared (IR10) channel. However, this approach cannot capture observed non-linear relationships between the colour channels, limiting its accuracy and ap
    
[^76]: 可信的峰值内存调度：精确实现、可认证的最优性，以及推理中重计算的极限

    Peak-Memory Schedules You Can Trust: Exact Realization, Certified Optimality, and the Limits of Recomputation for Inference

    [https://arxiv.org/abs/2610.23328](https://arxiv.org/abs/2610.23328)

    该论文提出通过剖析并重排生成程序自身的语句来精确实现峰值内存调度，证明只要内存分配不依赖历史即可逐字节精确达到建模峰值且数值结果逐位一致，从而解决了将优化算子顺序交回编译器重新生成代码的不可靠问题。

    

    在小型设备上进行实时推理需要在无额外延迟的前提下实现低激活内存，这使算子顺序成为最有价值的内存调控手段：它在时间上是零成本的。我们证明了通常部署更优顺序的做法——将其交还给编译器并重新生成代码——是不可靠的。在PyTorch Inductor中，重新生成的程序在54个CPU配置中有22个未能达到预测的内存峰值，其中9个比原始程序更差；在GPU的50个配置中有19个如此，并且在GPU上它还会改变数值结果。我们转而对生成的程序进行剖析，根据其自身语句及其读/写效果对其内存进行精确建模，并对这些语句进行重排。我们证明，只要内存分配不依赖于历史，该方法就能实现所建模的峰值。在我们所有的CPU运行中，它都能精确到字节地实现该峰值，且结果与原程序逐位一致。在GPU上，我们刻画了缓存分配器为何会破坏这一条件。为了知道何时应当停止，我们给出了前向……（原文摘要至此截断）

    arXiv:2610.23328v1 Announce Type: new  Abstract: Real-time inference on small devices needs low activation memory without extra latency, which makes operator order the most valuable memory lever: it is free in time. We show that the usual way of deploying a better order, handing it back to the compiler and regenerating code, is unreliable. In PyTorch Inductor the regenerated program misses its predicted peak in 22 of 54 CPU configurations. It is worse than the stock program in 9 of them and in 19 of 50 on a GPU, and on the GPU it also changes numerical results. We instead profile the generated program, model its memory exactly from its own statements and their read/write effects, and permute those statements. We prove that this realizes the modeled peak whenever allocation does not depend on history. It does so to the byte, with bitwise-identical results, in all our CPU runs. On the GPU we characterize why the caching allocator breaks the condition. To know when to stop, we give forwar
    
[^77]: 损失函数还是网络架构？竞争风险生存分析中什么更重要

    The Loss or the Architecture? What Matters in Survival Analysis with Competing Risks

    [https://arxiv.org/abs/2610.23322](https://arxiv.org/abs/2610.23322)

    该论文将损失函数与网络架构解耦并独立评估，在20个表格和3个病理图像数据集上发现：在删失加权下直接建模累积发生率的损失函数比选择更好的骨干网络带来更大的收益。

    

    通常情况下，例如在医疗保健和预测性维护中，人们关心的是多个相互竞争的原因中哪一个会最先发生，以及其发生的概率：这就是竞争风险生存分析。相应的深度学习方法往往是作为“捆绑组合”开发的：一个新架构搭配一个新的训练损失函数，仅在一到五个数据集上与少数基线方法进行测试。因此，最先进的结果无法被归因于架构、损失函数还是评估协议。这种耦合并非偶然：由于每个竞争原因对应一条发生率曲线，网络的输出头与可在其上表达的损失函数是绑定在一起的。我们将这两个维度独立地进行变化：三种损失函数（其中两种是严格适当的）搭配四种骨干网络，跨越两种能够将这两个维度区分开的模态：20个表格数据集和3个病理图像数据集。有一种损失函数在两种模态中都胜出：即在删失加权下直接对累积发生率进行建模。它带来的收益是好骨干网络选择的两倍。

    arXiv:2610.23322v1 Announce Type: cross  Abstract: Often, \emph{eg} in healthcare and predictive maintenance, what matters is which of several competing causes will strike first, and with what probability: survival analysis with competing risks. Corresponding deep learning methods are developed as bundles: a new architecture together with a new training loss, tested on one to five datasets against a handful of baselines. Thus, a state-of-the-art result cannot be attributed to the architecture, the loss, or the evaluation protocol. The coupling is not accidental: with one incidence curve per cause, a network's head is tied to the losses expressible on it. We vary the two axes independently, three losses, two of them strictly proper, on four backbones, across two modalities that tease the axes apart: 20 tabular and 3 pathology-image datasets. One loss wins in both: modelling cumulative incidence directly under censoring weighting. It brings twice as much as benefits a good choice of back
    
[^78]: 基于未训练CNN与系数空间正则化的光声层析成像联合重建

    Untrained CNN and coefficient-space regularization for joint reconstruction in photoacoustic tomography

    [https://arxiv.org/abs/2610.23315](https://arxiv.org/abs/2610.23315)

    该论文提出了一种基于未训练CNN固定非线性特征与系数空间正则化的数据驱动框架，实现了光声层析成像中初始压力与空间变化声速的联合重建，具有强泛化能力和噪声鲁棒性。

    

    本文考虑了从边界测量数据中对光声层析成像中的初始压力和空间变化声速进行联合重建的问题。提出了一种正则化数据驱动框架，其中未知场在有限维小波空间中表示，并利用由未训练卷积神经网络（CNN）生成的固定非线性特征，通过Tikhonov正则化最小二乘法重建初始压力系数。随后使用学习到的系数空间先验来推断平方声速系数。数值实验表明，该方法在保留数据上实现了精确重建，并对先前未见的模体几何形状具有很强的泛化能力。冻结的重建映射在训练中未出现的测量噪声下也保持稳定，对于高达10%的噪声水平，重建精度仅有轻微变化。与有监督训练的CNN相比……（原文摘要在此处被截断）

    arXiv:2610.23315v1 Announce Type: new  Abstract: The joint reconstruction of the initial pressure and spatially varying sound speed in photoacoustic tomography from boundary measurements is considered. A regularized data-driven framework is proposed in which the unknown fields are represented in finite-dimensional wavelet spaces and fixed nonlinear features generated by an untrained convolutional neural network (CNN) are used to reconstruct the initial-pressure coefficients through Tikhonov-regularized least squares. A learned coefficient-space prior is subsequently used to infer the squared sound-speed coefficients. Numerical experiments demonstrate accurate reconstruction on held-out data and strong generalization to previously unseen phantom geometries. The frozen reconstruction map also remains stable under measurement noise not represented during training, with only minor changes in reconstruction accuracy for noise levels up to $10\%$. In comparison with the supervised trained-CN
    
[^79]: 观察、推理、绘制区域：面向空间上下文预测的递归细化方法

    Look, Reason, Draw a Region: Recursive Refinement for Spatial Context Prediction

    [https://arxiv.org/abs/2610.23302](https://arxiv.org/abs/2610.23302)

    Spatial-TRM是一个仅有1120万可训练参数的紧凑递归推理模型，通过冻结视觉-语言特征与递归细化机制，在RoboSpatial-Home基准上将空间上下文预测准确率大幅提升9.02个百分点。

    

    空间上下文预测需要视觉定位和空间推理能力，以识别场景中相对于物体可用的空闲区域。空间关系和物理约束对于大型视觉-语言模型（VLM）而言仍然充满挑战。递归推理已在离散和抽象推理任务上展现出强大的性能，这启发我们将其应用于改进自然图像中的空间上下文预测。权重共享使模型能够以少量可训练参数实现更高精度。我们提出了Spatial-TRM，一个紧凑的递归模型，它采用冻结的视觉-语言特征、凸区域解码器，且仅有1120万可训练参数。这个权重共享的推理器支持六种空间关系、三种参考坐标系以及开放词汇的锚定物体。在RoboSpatial-Home基准测试中，Spatial-TRM实现了29.51%的上下文准确率，相比单独训练的单步模型提升了9.02个百分点（相对提升44.0%）。

    arXiv:2610.23302v1 Announce Type: cross  Abstract: Spatial context prediction requires visual grounding and spatial reasoning to identify vacant regions relative to objects in a scene. Spatial relationships and physical constraints remain challenging for large vision-language models (VLMs). Recursive reasoning has demonstrated strong performance on discrete and abstract reasoning tasks, motivating its use for improving spatial context prediction in natural images. Weight sharing enables higher accuracy with a small trainable parameter count. We introduce Spatial-TRM, a compact recursive model with frozen vision-language features, a convex region decoder, and just 11.2M trainable parameters. This weight-tied reasoner supports six spatial relationships, three reference frames, and open-vocabulary anchor objects. On the RoboSpatial-Home benchmark, Spatial-TRM achieves 29.51% context accuracy, a gain of 9.02 percentage points (44.0% relative) over a separately trained single-step model wit
    
[^80]: 弹性轴如何折叠跨声速颤振边界：激波能量传递与共形校准的主动学习

    How the elastic axis folds the transonic flutter boundary: shock energy transfer and conformally calibrated active learning

    [https://arxiv.org/abs/2610.23298](https://arxiv.org/abs/2610.23298)

    本文通过批量主动学习训练的高斯过程代理模型结合分裂共形预测校准，仅用80次非定常Euler模拟便绘制出Isogai A工况翼型折叠的跨声速颤振边界，揭示了弹性轴位置与激波能量传递对边界折叠的作用机制，并发现时域阻尼标签的隐性失效模式及初始迎角残余对颤振速度的显著影响。

    

    Isogai A工况是一个俯仰-沉浮耦合翼型，其弹性轴位于前缘前方半弦长处，在无粘流动中呈现出折叠的跨声速颤振边界：在跨声速凹谷底部附近，同一个马赫数下存在多个共存的颤振速度。我们通过耦合的非定常Euler模拟、由批量主动学习训练的阻尼系数高斯过程代理模型，以及在独立随机模拟上进行的分裂共形预测，绘制了这一边界以及移动弹性轴所得到的边界族；共形预测使代理模型在进行稳定性判定时具有与分布无关的误差率保证。文中记录了时域阻尼标签的三种在残差中不留任何痕迹的失效模式，并证明常用的1度初始迎角所遗留的0.5度平均迎角会使凹谷处的颤振速度提高2.8%，其影响超过任何数值参数设置。仅用80次模拟，学习得到的A工况边界即可对65个独立算例中的64个完成正确标注。

    arXiv:2610.23298v1 Announce Type: cross  Abstract: Isogai's case A, a pitch-plunge aerofoil whose elastic axis lies half a chord ahead of the leading edge, has a folded transonic flutter boundary in inviscid flow: near the bottom of the transonic dip several flutter speeds coexist at one Mach number. We map this boundary, and the family obtained by moving the elastic axis, with coupled unsteady Euler simulations, a Gaussian-process surrogate of the damping coefficient trained by batch active learning, and split conformal prediction on independent random simulations, which turns the surrogate into stability calls with a distribution-free error rate. Three failure modes of time-domain damping labels that leave no trace in the residuals are documented, and the half-degree mean incidence left by the common one-degree start is shown to raise the flutter speed at the dip by 2.8%, more than any numerical setting. With 80 simulations the learned case A boundary labels 64 of 65 independent runs
    
[^81]: 学习量子近似优化的角度参数

    Learning angles for quantum approximate optimization

    [https://arxiv.org/abs/2610.23296](https://arxiv.org/abs/2610.23296)

    提出通过学习模型建立从问题实例到高质量QAOA变分角度参数的映射，用单次前向传播取代计算昂贵的逐实例非凸优化，速度提升数个数量级且质量几乎不变，并能泛化到未见过的图拓扑结构。

    

    量子近似优化算法（QAOA）及其变体是帮助解决具有挑战性的组合优化问题的领先算法。QAOA将这些问题映射到Ising哈密顿量上，使其基态即为所需的解，随后从变分量子电路中采样得到候选解。QAOA的性能取决于一组变分角度参数，然而寻找好的角度参数是一项针对具体实例、计算成本高昂的非凸经典优化任务。本文中，我们学习了从问题实例到高质量QAOA角度参数的映射。我们训练的模型用单次前向传播取代了逐实例的优化过程（我们称之为求解器），其速度比求解器快几个数量级，同时质量几乎相同。我们的模型还能进一步泛化到未见过的拓扑结构，例如Barabási-Albert网络和二维网格图。在量子硬件上的实验证实了QAOA采样（摘要在此处被截断）……

    arXiv:2610.23296v1 Announce Type: cross  Abstract: The Quantum Approximate Optimization Algorithm (QAOA) and its variants are leading algorithms to help solve challenging combinatorial optimization problems. QAOA maps these problems to Ising Hamiltonians such that their ground state is the desired solution. Candidate solutions are then drawn from a variational quantum circuit. The performance of QAOA hinges on a set of variational angles. However, finding good angles is an instance-specific and computationally expensive non-convex classical optimization task. Here, we learn the mapping from the problem instance to high-quality QAOA angles. Our learned models replace the per-instance optimization, that we call the solver, with a single forward pass that is several orders of magnitude faster than the solver with a near identical quality. Our models further generalize to unseen topologies such as Barab\'asi-Albert and 2D grid graphs. Experiments on quantum hardware confirm that QAOA sampl
    
[^82]: 两类分类中两个排序得分之间的最优排序权衡

    The Optimal Ranking Tradeoffs Between Two Ranking Scores in Two-Class Classification

    [https://arxiv.org/abs/2610.23294](https://arxiv.org/abs/2610.23294)

    该论文揭示了常用的二分类评估指标都是同一参数化排序得分族的特例，并据此建立了两类分类中任意两个排序得分之间的最优权衡排序方法。

    

    准确率、真负例率（负类召回率）、真正例率（正类召回率）、负预测值（负类精确率）、正预测值（正类精确率）、类别特定的交并比、类别特定的F分数、Cohen's kappa系数以及平衡准确率，都是人们在开发机器学习方法时直观上希望最大化的评分指标。然而，这些指标会导致对这些方法的不同排序。那么，我们如何建立一种在这些指标中任意两者之间实现最优权衡的排序呢？事实证明，所有这些指标都是排序得分（ranking scores）的特例：这是一种由分配给真负例、假正例、假负例和真正例结果的相对重要性值所参数化的连续评分体系。由于所有这些指标都已知能产生有意义的排序，我们寻求它们之间的最优权衡。我们将两个得分之间的权衡视为……（摘要原文在此处截断）

    arXiv:2610.23294v1 Announce Type: cross  Abstract: The accuracy, true negative rate (negative class recall), true positive rate (positive class recall), negative predictive value (negative class precision), positive predictive value (positive class precision), class-specific intersections over unions, class-specific F-scores, Cohen's kappa, and balanced accuracy are scores that one intuitively wants to maximize when developing machine learning methods. However, they lead to different rankings of such methods. So, how can we establish a ranking that is an optimal tradeoff between any pair of these scores? It turns out that all these scores are special cases of ranking scores: a continuum of scores parameterized by the relative importance values assigned to the true negative, false positive, false negative, and true positive outcomes. Since all these scores are known to induce meaningful rankings, we seek the optimal tradeoffs among them. We consider the tradeoff between two scores to be
    
[^83]: 基于神经控制的即插即用网络分布式性能提升

    Distributed Performance Boosting through Neural Control for Plug-and-Play Networks

    [https://arxiv.org/abs/2610.23292](https://arxiv.org/abs/2610.23292)

    本文提出一种基于内模控制原理的分布式神经控制方法，通过局部神经算子的自由参数化互连，在保证即插即用网络系统闭环稳定性的同时提升控制性能。

    

    我们研究了改善预稳定网络化系统在即插即用（PnP）操作（即子系统的添加或移除）下性能的问题。虽然分布式基线控制器可以在PnP转换后保持稳定性，但若使用富有表现力的神经网络策略对其进行增强，则可能使形式化的稳定性保证失效。为克服这一问题，我们基于最近提出的、以内模控制（IMC）原理为基础的性能提升控制器，该方法将闭环稳定性问题转化为对辅助算子的有限ℓ₂增益要求。在分布式设置中，该算子由仅在相邻子系统之间交换信息的局部神经算子互连构成。随后，我们推导了这种互连结构的自由参数化方法，即从无约束的可训练变量到满足规定网络增益界的互连神经算子的显式映射。

    arXiv:2610.23292v1 Announce Type: cross  Abstract: We address the problem of improving the performance of pre-stabilized networked systems subject to plug-and-play (PnP) operations, namely additions or removals of subsystems. While distributed baseline controllers can preserve stability after a PnP transition, augmenting them with expressive neural policies can invalidate the formal stability guarantees. To overcome this issue, we build on recently proposed Performance Boosting controllers based on the Internal Model Control (IMC) principle, which reduce closed-loop stability to a finite $\ell_2$-gain requirement on an auxiliary operator. In the distributed setting, this operator is formed by interconnecting local neural operators that exchange information only between neighboring subsystems. We then derive free parametrizations of this interconnection, namely explicit maps from unconstrained trainable variables to interconnected neural operators satisfying a prescribed network gain bo
    
[^84]: 通过扩散模型实现数据高效的分类器预训练

    Data-Efficient Classifier Pretraining Through Diffusion Models

    [https://arxiv.org/abs/2610.23289](https://arxiv.org/abs/2610.23289)

    提出DiffPT扩散建模预训练方法，可在领域训练数据有限时有效预训练分类器骨干网络，并在医学影像和时间序列任务中超越自监督基线方法。

    

    在数据有限的机器学习任务中，模型容易过拟合，尤其是现代深度神经网络。在AlexNet出现之前，基于重构的预训练方法（如自编码器和去噪自编码器）被证明是有益的，这部分归因于当时训练集规模有限。受这些传统方法的启发，我们提出了DiffPT，这是一种简单的扩散建模方法，用于在特定领域训练数据有限的情况下预训练给定的分类器骨干网络。我们在时间序列数据、标准自然图像基准以及医学影像数据集上进行实验，并将该扩散方法与现代和传统的多种无/自监督方法进行对比。尽管在低数据量场景下，对比学习预训练在CIFAR-10和Tiny ImageNet上表现更好，但DiffPT在医学影像和时间序列任务上尤为有效，在一系列低数据量场景中其性能超越了自监督基线方法。

    arXiv:2610.23289v1 Announce Type: new  Abstract: Machine learning with limited data is susceptible to overfitting, especially with modern deep neural networks. Prior to the introduction of AlexNet, reconstruction-based pretraining with auto-encoder and denoising auto-encoder were found to be beneficial, partially due to the limited size of the training sets at the time. Motivated by these traditional approaches, we propose DiffPT, a simple diffusion modeling approach to pretrain a given classifier backbone for limited domain-specific training data. We experiment on time-series data, standard natural-image benchmarks, and medical imaging datasets, and contrast the diffusion approach to a multitude of modern and traditional un/self-supervised methods. While contrastive pretraining performs better on CIFAR-10 and Tiny ImageNet in low-data regimes, DiffPT is particularly effective on medical imaging and time-series tasks, where it outperforms self-supervised baselines across a range of low
    
[^85]: DeDiCa：一种可扩展的去混杂离散因果生成模型

    DeDiCa: A Scalable Deconfounding Discrete Causal Generative Model

    [https://arxiv.org/abs/2610.23272](https://arxiv.org/abs/2610.23272)

    DeDiCa通过张量化参数化的因果贝叶斯网络，实现了对含隐藏混杂的离散变量高效且可扩展的因果推断，其精度媲美最先进的神经因果生成模型，同时推理速度和内存开销大幅降低。

    

    我们提出了DeDiCa，一个面向离散变量和隐藏混杂因素的高效因果推断实用框架。该框架通过因果贝叶斯网络的张量化参数化重新审视经典概率建模，使推断变得可处理。给定一个因果图，DeDiCa只需在观测数据上通过最大似然训练一次，对隐藏混杂因素进行精确边缘化，之后即可估计任何（代理）可识别的干预查询。尽管参数化设计有意保持简洁，DeDiCa在平均处理效应（ATE）估计方面可与最先进的神经因果生成模型相媲美，同时支持内存高效的CPU训练和毫秒级的精确干预推断。与因果基础模型相比，DeDiCa在观测样本增多时扩展性更强，并以显著更低的推断时间和内存实现了具有竞争力的准确度。DeDiCa为因果推断提供了一个可处理、准确且可扩展的方案。

    arXiv:2610.23272v1 Announce Type: new  Abstract: We introduce DeDiCa, a practical framework for efficient causal inference with discrete variables and hidden confounding that revisits classical probabilistic modeling through a tensorized parameterization of causal Bayesian networks, enabling tractable inference. Given a causal graph, DeDiCa is trained once on observational data by maximum likelihood, exactly marginalizing over hidden confounders, and can then estimate any (proxy-)identifiable interventional query. Despite this deliberately simple parameterization, DeDiCa matches state-of-the-art neural causal generative models in ATE estimation, while supporting memory-efficient CPU training and exact interventional inference in milliseconds. Compared with causal foundation models, DeDiCa scales more effectively with additional observational samples and achieves competitive accuracy at substantially lower inference time and memory. DeDiCa provides a tractable, accurate, and scalable ap
    
[^86]: NestedVLA：学习整合与生成视觉-语言-动作模型的技能

    NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models

    [https://arxiv.org/abs/2610.23271](https://arxiv.org/abs/2610.23271)

    NestedVLA提出用嵌套超网络生成任务自适应的动作专家参数来事后整合独立学习的技能，从而在多个机器人操作基准上实现灵活的任务条件化技能复用并有效缓解任务间干扰。

    

    视觉-语言-动作（VLA）模型有望实现通用机器人智能体，然而跨异构任务与环境整合所学习的技能仍然十分困难：训练数据分散在不同平台上，联合多任务训练存在相互干扰的问题，而模型合并往往引入参数冲突或需要改变模型架构。为此，我们倡导一种先独立学习技能、再进行事后整合的范式。具体而言，我们提出NestedVLA，用参数生成取代参数聚合：一个嵌套超网络以目标任务为条件，合成任务自适应的动作专家参数，并与冻结的基础VLA模型组合；任务特定的压缩VLM残差则被单独保留。这一设计实现了灵活的任务条件化技能复用，同时缓解了任务间的干扰。在三个强化学习基准（MetaWorld、ManiSkill、CALVIN）上，NestedVLA的表现超越了现有方法。

    arXiv:2610.23271v1 Announce Type: cross  Abstract: Vision--Language--Action (VLA) models promise generalist robotic agents, yet consolidating skills learned across heterogeneous tasks and environments remains difficult: training data is fragmented across platforms, joint multi-task training suffers from interference, and model merging tends to introduce parameter conflicts or require architectural changes. We instead advocate a paradigm in which skills are first learned independently and then consolidated post hoc. To this end, we propose NestedVLA, which replaces parameter aggregation with parameter generation: a nested hypernetwork, conditioned on the target task, synthesizes task-adaptive action-expert parameters that compose with a frozen base VLA. Task-specific compressed VLM residuals are retained separately. This formulation enables flexible task-conditioned skill reuse while mitigating interference. Across three RL benchmarks (MetaWorld, ManiSkill, CALVIN), NestedVLA outperform
    
[^87]: 超越边际影响：面向LLM智能体的交互感知记忆选择

    Beyond Marginal Influence: Interaction-Aware Memory Selection for LLM Agents

    [https://arxiv.org/abs/2610.23266](https://arxiv.org/abs/2610.23266)

    该论文提出JointCoreBench基准和JointCore方法，突破传统基于边际影响的记忆排序方式，通过感知记录间交互并采用成本排序的组删除策略，实现LLM智能体最小成本充分记忆子集的选择，达到100%恢复率并将子集评估次数减少62.9%。

    

    LLM智能体的记忆选择通常使用基于记录相关性或个体影响的检索排序。这种排序在存在冗余证据或联合必要证据时无法识别出充分的子集。我们研究逐决策记忆充分性问题：即选择一个能保持完整记忆行为的小子集。为了将搜索误差与模型误差分离，我们提出了JointCoreBench——一个包含九类交互族、六个工作流领域、自然语言决策以及精确充分集预言的基准测试，并提出了JointCore——一种按相关性成本排序、先测试记录块再测试单条记录的组删除基线方法。在6,912个精确条件下，按成本排序的JointCore在100.0%的情况下恢复了最小成本的充分集，将子集评估次数从30.00降至11.13（降低62.9%），并实现了87.8%的活跃记忆节省。在四个12B-70B模型家族和每个模型432个任务上，其未加权平均值为10.28，对比16.00次模型查询和50.6%。

    arXiv:2610.23266v1 Announce Type: cross  Abstract: LLM-agent memory selection often uses retrieval rankings based on record relevance or individual influence. Such rankings cannot identify sufficient subsets under redundant or jointly necessary evidence. We study per-decision memory sufficiency: selecting a small subset that preserves full-memory behavior. To separate search error from model error, we introduce JointCoreBench, a benchmark with nine interaction families, six workflow domains, natural-language decisions, and exact sufficient-set oracles, and JointCore, a relevance cost-ordered group-deletion baseline that tests blocks before individual records. On 6,912 exact conditions, cost-ordered JointCore recovers a minimum-cost sufficient set in 100.0% of cases, reduces subset evaluations from 30.00 to 11.13 (62.9%), and achieves 87.8% active-memory savings. Across four 12B-70B model families and 432 tasks per model, its unweighted mean is 10.28 versus 16.00 model queries and 50.6%
    
[^88]: ADT：面向三维分子结构的全原子扩散分词器

    ADT: All-Atom Diffusion Tokenizer for 3D Molecular Structures

    [https://arxiv.org/abs/2610.23265](https://arxiv.org/abs/2610.23265)

    ADT通过将分词粒度与重建分辨率解耦，并借助扩散模型的生成能力恢复量化带来的几何畸变，实现了兼顾紧凑表示与全原子精确保建的三维分子结构分词方法。

    

    离散结构分词为自回归式三维分子生成提供了一条有前景的路径。然而，现有分词器在紧凑表示与全原子重建之间存在权衡：原子级分词能够保留原子细节，但会产生较长的序列；而紧凑的残基级蛋白质分词通常会忽略侧链几何信息。为克服这一局限，我们提出了全原子扩散分词器，它将分词粒度与重建分辨率解耦。ADT对小分子采用每个原子一个结构分词的方式，对蛋白质则采用每个残基一个分词来联合表示主链和侧链的几何信息。为减轻量化过程引入的几何畸变，我们利用扩散模型的生成能力，超越基于离散分词的确定性解码，从而恢复细粒度的原子几何结构。在小分子和蛋白质上的大量实验表明，ADT实现了强大的……

    arXiv:2610.23265v1 Announce Type: new  Abstract: Discrete structural tokenization provides a promising route to autoregressive 3D molecular generation. Yet, existing tokenizers face a trade-off between compact representation and all-atom reconstruction: atom-level tokens preserve atomic details but produce long sequences, while compact residue-level protein tokens typically omit side-chain geometry. To overcome this limitation, we propose All-atom Diffusion Tokenizer (ADT), which decouples token granularity from reconstruction resolution. ADT uses one structural token per atom for small molecules and one token per protein residue that jointly represents backbone and side-chain geometry. To mitigate geometric distortion introduced by quantization, we leverage the generative capacity of diffusion models to recover fine-grained atomic geometry beyond deterministic decoding from discrete tokens. Extensive experiments across small molecules and proteins demonstrate that ADT achieves strong 
    
[^89]: 推理模型在自我指令下能够控制其思维链

    Reasoning Models Can Control Their Chains of Thought When They Tell Themselves To

    [https://arxiv.org/abs/2610.23257](https://arxiv.org/abs/2610.23257)

    推理模型并非天生无法控制其思维链，将约束作为自我指令置于CoT内部时，模型的可控性显著优于在用户提示中指定约束的方式。

    

    思维链（CoT）可控性，即推理模型控制其思维链属性的能力，对模型的有用性和可靠监督都具有重要意义。近期研究通过测量模型是否遵循用户提示中给出的CoT约束来评估这一能力，并报告各模型均表现出极低的可控性。我们认为，这种低可控性可能并非反映推理模型天生无法控制其CoT，而是源于在用户提示中简单直接地指定约束的方式。为验证这一点，我们提出了自我指令CoT可控性（Self-Instructed CoT Controllability）：一种互补的评估方法，它将约束作为自我指令置于模型的CoT内部，并测量模型遵循该约束的能力。我们表明，在自我指令条件下，模型实现的可控性显著高于标准用户指令下的水平。即使仅是提醒模型遵循用户约束的通用提示，也能大幅提升可控性。

    arXiv:2610.23257v1 Announce Type: new  Abstract: Chain-of-thought (CoT) controllability, the ability of reasoning models to control properties of their CoT, is consequential for both their usefulness and reliable oversight. Recent work evaluates this capability by measuring whether models follow CoT constraints given in the user prompt, and reports very low controllability across models. We argue that this low controllability may not reflect an inherent inability of reasoning models to control their CoT, but instead result from naively specifying constraints in the user prompt. To test this, we introduce Self-Instructed CoT Controllability: a complementary evaluation that places the constraint as a self-instruction inside the model's CoT and measures its ability to follow it. We show that, under self-instruction, models achieve significantly higher controllability than with standard user instructions. Even a generic reminder to follow the user's constraint substantially improves contro
    
[^90]: 采样、混合还是验证？不完美验证器下的测试时计算分配

    Sample, Mix, or Verify? Allocating Test-Time Compute under Imperfect Verifiers

    [https://arxiv.org/abs/2610.23255](https://arxiv.org/abs/2610.23255)

    该论文提出 SPLIT 方法，利用小型带标签试点实验和精确概率公式，在验证器不完美的情况下科学地分配测试时计算预算（采样数量、模型选择与验证开销）。

    

    arXiv:2610.23255v1 公告类型：cross 摘要：让语言模型和智能体更准确的一种常见方法是在推理阶段投入更多计算：生成大量候选解决方案，并提交验证器所偏好的那一个。成功需要满足两个条件：存在正确的候选解，且验证器能将其选出。前者随样本数量的增加而可预测地提升；后者却并非如此，因为真实的验证器（单元测试、多数投票、LLM 裁判、奖励模型）会接受错误答案、拒绝正确答案，并且会犯具有相关性的错误。在预算固定的情况下，从业者必须决定抽取多少样本、从哪些模型中抽取，以及在验证上投入多少资源，而目前这些选择都是凭经验法则做出的。我们提出 SPLIT，它基于一个小的带标签试点实验来做出这些决策。其核心是一个精确公式，用于计算在接受/拒绝和最高分选择机制下提交正确答案的概率，该公式考虑了任务难度的变化以及各模型之间共享的盲点……

    arXiv:2610.23255v1 Announce Type: cross  Abstract: A common way to make language models and agents more accurate is to spend more compute at inference time: generate many candidate solutions and submit the one a verifier prefers. Success then requires that some candidate is correct and that the verifier picks it. The first grows predictably with the number of samples; the second does not, because real verifiers (unit tests, majority votes, LLM judges, reward models) accept wrong answers, reject right ones, and make correlated mistakes. With a fixed budget, a practitioner must decide how many samples to draw, which models to draw them from, and how much to spend on verification, and today these choices are made by rules of thumb. We introduce SPLIT, which makes them from a small labelled pilot. It rests on an exact formula for the probability of submitting a correct answer under accept/reject and top-score selection that accounts for varying task difficulty, blind spots shared across mo
    
[^91]: FLARE：面向退化观测下馈线恢复的语言模型智能体的物理约束评估

    FLARE: Physics-Grounded Evaluation of Language-Model Agents for Feeder Restoration with Degraded Observations

    [https://arxiv.org/abs/2610.23251](https://arxiv.org/abs/2610.23251)

    FLARE是一个基于物理约束的馈线恢复评估基准，通过五种退化观测设置下800个交流可行案例、共78,750个回合的大规模评估，首次系统揭示了语言模型智能体在配电网恢复任务中虽中位表现优异但存在显著失败长尾的风险，并验证了物理安全保护机制的有效性。

    

    配电网故障后的供电恢复需要在信息不完整且相互冲突的情况下做出决策，同时遵守电气约束。针对这一需求，我们提出了FLARE（Feeder-Level Agent Restoration Evaluation，馈线级智能体恢复评估），这是一个面向隔离后馈线恢复的物理约束基准。FLARE包含800个交流可行的案例，涵盖五种观测设置，允许智能体在固定运行限制下收集证据、评估开关并执行恢复操作。我们在78,750个回合中评估了五个指令微调模型，涵盖拓扑变化、分布式发电和负载变化等场景。平均归一化遗憾值介于0.082至0.148之间。尽管每个模型的中位遗憾值均为零，但P95遗憾值介于0.814至1.000之间，揭示了显著的失败长尾。影子回放显示，在15,706个被保护机制拒绝的操作中，96.4%原本会造成物理违规。FLARE实现了语言模型智能体的可重复评估（原文摘要在此处截断）。

    arXiv:2610.23251v1 Announce Type: new  Abstract: Restoring service after distribution-grid faults requires decisions from incomplete and conflicting information while respecting electrical constraints. Addressing this need, we introduce FLARE (Feeder-Level Agent Restoration Evaluation), a physics-grounded benchmark for post-isolation feeder restoration. FLARE con- tains 800 AC-feasible cases across five observation settings and allows agents to gather evidence, evaluate switches, and perform restoration actions under fixed operational limits. We evaluate five instruction-tuned models over 78,750 episodes, including topology, distributed-generation, and loading shifts. Mean normalized regret ranges from 0.082 to 0.148. Although every model has zero median regret, P95 regret ranges from 0.814 to 1.000, revealing an important failure tail. Shadow replay shows that 96.4% of 15,706 guard-rejected actions would have caused a physical violation. FLARE enables reproducible evaluation of langua
    
[^92]: 局部能否骗过测试？面向关系型分布外误差的反事实配对循环

    Can the Parts Fool the Test? Counterfactual Pairing Cycles for Relational OOD

    [https://arxiv.org/abs/2610.23248](https://arxiv.org/abs/2610.23248)

    提出“反事实配对循环”评估基元：在比较两侧复用相同的两张图像与两个描述、仅改变配对方式，使仅图像、仅文本及可加性捷径在有限样本中精确抵消，从而干净地评估关系型分布外误差，并兼作关系型离群暴露的训练损失与双线性视觉-语言评分的诊断工具。

    

    关系型分布外误差很容易描述，却很难进行干净的评估。一幅图像可能是自然的，一个描述（caption）可能是流畅的，且其中提及的所有物体都可能是常见的，但该描述却可能错误地描述了图像中的关系。许多旨在测试这种行为的评估，即使完全不检查图像与描述的配对关系也能被“解决”，因为负样本在描述风格、模板、长度、物体频率或图像统计特性上存在差异。我们提出反事实配对循环，这是一种分块式的评估基元，它在比较的两侧复用相同的两张图像和相同的两个描述，仅改变哪张图像与哪个描述相配对。这使得仅图像、仅文本以及可加性成分等捷径在有限样本中精确地相互抵消。同样的对比还为关系型离群暴露提供了一种训练损失，并为双线性视觉-语言评分提供了一种诊断工具。在合成控制数据、GQA 和 COCO 上，循环分离……（原文摘要在此处截断）

    arXiv:2610.23248v1 Announce Type: cross  Abstract: Relational out-of-distribution errors are easy to describe and hard to evaluate cleanly. An image may be natural, a caption may be fluent, and all mentioned objects may be common, yet the caption may describe the wrong relation in the image. Many evaluations intended to test this behavior can be solved without checking the pairing at all, because negatives differ in caption style, template, length, object frequency, or image statistics. We propose counterfactual pairing cycles, a blocked evaluation primitive that reuses the same two images and the same two captions on both sides of the comparison, changing only which image is paired with which caption. This makes image-only, text-only, and additive component shortcuts cancel exactly in finite samples. The same contrast also gives a training loss for relational outlier exposure and a diagnostic for bilinear vision-language scores. Across synthetic controls, GQA, and COCO, cycles separat
    
[^93]: 多思考还是多阅读？共享上下文下推理的Token预算理论

    Think More or Read More? A Token-Budget Theory of Reasoning with Shared Context

    [https://arxiv.org/abs/2610.23246](https://arxiv.org/abs/2610.23246)

    提出token预算理论形式化固定上下文窗口下“阅读他人输出”与“自主推理”的权衡，通过允许错误相关的Condorcet陪审团模型推导出多智能体辩论最优智能体数量的闭式解，且仅需单智能体测量即可估计最优群体规模。

    

    推理系统越来越多地通过在模型输入中纳入额外上下文来改进答案，从多智能体辩论（智能体阅读同伴的回答）到自我改进（模型重新审视自己的草稿）皆是如此。在这些系统中，通信会消耗上下文，因为一个智能体的输出会成为另一个智能体的输入。我们研究了token阅读与token生成共享同一固定token窗口的情形，因此每花费一个token用于阅读，就少一个token可用于推理。我们提出了一个**token预算理论**来形式化这种权衡，将准确率建模为阅读后剩余推理预算的函数，并采用允许错误相关的Condorcet陪审团模型来聚合投票。对于多智能体辩论，该理论给出了最优智能体数量的闭式预测。这一数量可从单智能体的测量中估计得出，从而无需实际运行辩论即可确定最优群体规模。

    arXiv:2610.23246v1 Announce Type: new  Abstract: Reasoning systems increasingly improve answers by incorporating additional context into the model's input, from multi-agent debate, where agents read their peers' responses, to self-refinement, where a model revisits its own drafts. In these systems, communication consumes context as one agent's output becomes another's input. We study the regime where token reading and generation draw on one fixed token window, so every token spent reading is a token that cannot be spent reasoning. We develop a \textbf{token-budget theory} that formalizes this trade-off, modeling accuracy as a function of the reasoning budget remaining after reading and aggregating votes with a Condorcet jury model that permits correlated errors. For multi-agent debate, the theory gives a closed-form prediction for the optimal number of agents. This quantity is estimated from single-agent measurements, allowing the optimal group size to be determined without running deb
    
[^94]: Transformers 中的各向异性表示动力学

    Anisotropic Representation Dynamics in Transformers

    [https://arxiv.org/abs/2610.23242](https://arxiv.org/abs/2610.23242)

    该论文提出了混合门控的各向异性表示动力学理论，将自注意力分解为数值传递、查询门控与键门控，揭示了门控机制可诱导表示差异从收缩切换为扩张的状态转换，并给出了判定该现象的充分必要谱判据。

    

    内部表示之间的微小差异在穿过 Transformer 层时可能会缩小或扩大，这会影响信息的保留方式以及模型对扰动的敏感程度。然而，注意力权重的变化何时能够将原本会缩小的表示差异转变为扩大的差异，这一点仍不清楚。我们发展了一套混合门控的各向异性表示动力学理论，将自注意力分解为数值传递、查询门控和键门控三个组成部分。我们的分析揭示了被注意力关注的表示的几何结构如何约束门控行为，以及这些机制之间的相互作用如何能够逆转由数值传递引起的收缩。我们将这一现象形式化为门控诱导的状态切换：表示差异在固定的注意力路由下收缩，但在完全注意力下扩张。我们推导出一个充分必要的谱判据，用以证明该现象的存在性或……

    arXiv:2610.23242v1 Announce Type: new  Abstract: Small differences between internal representations can shrink or grow as they pass through a Transformer layer, affecting how information is preserved and how sensitive the model is to perturbations. Yet it remains unclear when changes in attention weights can turn a representation difference that would otherwise shrink into one that grows. We develop a \textbf{Mixture-Gated Theory of Anisotropic Representation Dynamics} that separates self-attention into value transport, query gating, and key gating. Our analysis reveals how the geometry of attended representations constrains gating and how interactions between these mechanisms can reverse contraction induced by value transport. We formalize this phenomenon as gating-induced regime switching, where a representation difference contracts under fixed attention routing but expands under full attention. We derive a necessary and sufficient spectral criterion that certifies the existence or a
    
[^95]: 面向成本感知多轮LLM路由的响应信息价值估计

    Response-Informed Value Estimation for Cost-Aware Multi-Turn LLM Routing

    [https://arxiv.org/abs/2610.23239](https://arxiv.org/abs/2610.23239)

    提出RIVET方法，在训练阶段利用每次调用后的实际响应信息为每一轮提供价值监督信号，而在部署时仅依赖调用前信息进行成本感知的多轮LLM模型路由。

    

    多轮LLM路由需要在每一轮对话中、在模型响应可用之前就选择模型，然而任务成功与否只有在整个交互结束时才能揭晓，这使得每个中间选择都缺少属于自己的标签。然而，记录下来的交互轨迹保留了每次调用所产生的响应，这揭示了每一轮对话所需要的内容以及所选模型对其服务的效果——而这些证据仅凭最终结果是看不到的。我们提出了RIVET（面向轮级路由的响应信息价值估计），该方法在训练阶段利用这种调用后的证据，同时在部署时保持路由严格在调用前完成。RIVET在共享的状态-模型表示之上联合训练两个价值估计器：一个是可部署的路由价值估计器，仅根据调用前信息对候选模型进行评分；另一个是仅用于训练阶段的响应信息价值估计器，它额外观察实际产生的响应。以响应为条件，能够将响应引起的回报变化从残余噪声中区分出来

    arXiv:2610.23239v1 Announce Type: new  Abstract: Multi-turn LLM routing selects a model at every turn before its response is available, yet task success is revealed only when the interaction ends, leaving each intermediate choice without a label of its own. Logged trajectories, however, retain the response each call produced, which exposes what each turn required and how well the chosen model served it, evidence that outcomes alone hide. We introduce RIVET (Response-Informed Value Estimation for Turn-Level Routing), which exploits this post-call evidence during training while keeping routing strictly pre-call at deployment. RIVET jointly trains two value estimators over a shared state-model representation: a deployable Routing Value Estimator that scores candidate models from pre-call information, and a training-only Response-Informed Value Estimator that additionally observes the realized response. Conditioning on the response turns response-induced return variation from residual nois
    
[^96]: 检索增强生成技术的特征描述与分析

    Characterising and Analysing Retrieval-Augmented Generation Techniques

    [https://arxiv.org/abs/2610.23226](https://arxiv.org/abs/2610.23226)

    提出了一个用于刻画和分析现有检索增强生成（RAG）技术的概念框架，可作为实证评估前的筛选机制，帮助研究者和实践者为特定问题场景选择最合适的RAG技术。

    

    大语言模型（LLM）在众多应用领域的兴起，增加了对检索增强生成（RAG）的需求，以减少幻觉并提升LLM的性能。迄今为止，文献中已提出多种具有出色特性和能力的RAG技术，这使得针对特定问题场景选择最合适的RAG技术成为一项艰巨的任务。因此，我们提出了一个由主要维度和子维度组成的概念框架，用于刻画（定义“它是什么”）和分析（定义“它表现如何”）现有的RAG技术。该概念框架是通过对文献中报道的RAG技术属性进行定性分析而开发的。它可以作为一种经验前筛选机制，帮助研究人员和从业者在进行实证基准测试之前，选择最符合特定用户和任务需求的高适配度RAG候选技术。

    arXiv:2610.23226v1 Announce Type: new  Abstract: The advent of Large Language Models (LLMs) across many application domains has increased the need for Retrieval-Augmented Generation (RAG) to reduce hallucinations and improve LLM performance. Thus far, several RAG techniques with impressive features and capabilities have been proposed in the literature, making it a herculean task to select the most appropriate RAG techniques for specific problem scenarios. Thus, we propose a conceptual framework comprising main dimensions and sub-dimensions for characterising (defining "what it is") and analysing (defining "how well it does") existing RAG techniques. The conceptual framework was developed through a qualitative analysis of the attributes of RAG techniques reported in the literature. It can serve as a pre-empirical filtering mechanism for researchers and practitioners in selecting the highest-fit RAG candidates that meet specific user and task requirements before empirical benchmarking of
    
[^97]: 基于可靠性与保守性感知的贝叶斯信念的过程奖励建模

    Process Reward Modeling with Reliability- and Conservatism-Aware Bayesian Belief

    [https://arxiv.org/abs/2610.23223](https://arxiv.org/abs/2610.23223)

    提出BayesianPRM，将过程奖励视为随机变量并维护贝叶斯信念，首次同时建模PRM不确定性与策略不确定性两类来源，从而缓解测试时扩展中系统性奖励高估的问题。

    

    过程奖励模型在大语言模型推理的测试时扩展中扮演着核心角色。然而，PRM可能会在测试时扩展中提供不准确的过程奖励，我们识别出这源于两种不确定性来源：（1）*PRM不确定性*，源于监督数据不足和PRM训练不完善；（2）*策略不确定性*，源于收集监督数据的策略与测试时扩展所使用策略之间的不匹配，导致系统性的奖励高估。以往的研究未能同时考虑这两种不确定性。为弥补这一空白，我们提出BayesianPRM，它将过程奖励视为奖励假设空间中的随机变量，并对其维护贝叶斯信念。由于在连续假设空间上进行精确贝叶斯推断通常是不可解的，我们使用奖励集成对奖励假设空间进行离散化……

    arXiv:2610.23223v1 Announce Type: new  Abstract: Process Reward Models (PRMs) play a central role in test-time scaling for large language model (LLM) reasoning. However, PRMs may provide inaccurate process rewards for test-time scaling, which we identify as arising from two sources of uncertainty: (1) \textit{PRM uncertainty} that arises from insufficient supervision data and imperfect PRM training; and (2) \textit{policy uncertainty} that results from the mismatch between the policy to collect supervision data and the policy for test-time scaling, leading to systematic reward overestimation. Previous studies fail to jointly consider both uncertainties. To bridge this gap, we propose BayesianPRM, which treats the process reward as a random variable within a reward hypothesis space and maintains a Bayesian belief over it. Since exact Bayesian inference over the continuous hypothesis space is generally intractable, we discretize the reward hypothesis space with a reward ensemble and intr
    
[^98]: 随机决策问题中可扩展学习的非参数优化

    Non-Parametric Optimization for Scalable Learning in Stochastic Decision Problems

    [https://arxiv.org/abs/2610.23204](https://arxiv.org/abs/2610.23204)

    本文基于Malliavin微积分开发了随机变分框架，提出了对参数化维度不敏感的可扩展深度学习算法——随机路径跟随器（SPF），有效求解时变随机优化问题。

    

    随机优化（SO）在不确定性决策中扮演着核心角色。在随机优化问题中，时变随机优化（TV-SO）因其在自适应控制和机器学习中的应用而尤为重要。目前已有针对时变确定性优化的非参数方法被提出；然而，尚未针对其随机对应问题进行开发。本工作通过开发一个基于Malliavin微积分的随机变分框架来填补这一空白。该框架为具有随机决策的随机优化问题导出了非参数最优性条件，并支持设计一种对参数化维度不敏感的可扩展深度学习算法。该算法被称为随机路径跟随器（SPF），被应用于分布漂移下的两个重要问题，即最小二乘恢复和逻辑回归。实验结果表明，所提出的（方法在相关任务中表现出色）

    arXiv:2610.23204v1 Announce Type: new  Abstract: Stochastic optimization (SO) plays a central role in decision-making under uncertainty. Among SO problems, time-varying stochastic optimization (TV-SO) is particularly important due to its applications in adaptive control and machine learning. Non-parametric approaches have been proposed for time-varying deterministic optimization; however, they have not been developed for their stochastic counterparts. This work addresses that gap by developing a stochastic variational framework based on Malliavin calculus. This framework yields non-parametric optimality conditions for SO problems with stochastic decisions and supports the design of a scalable deep-learning algorithm that is insensitive to the parameterization dimension. This algorithm, called the Stochastic Path Follower (SPF), is applied to two important problems under distribution drift, namely least-squares recovery and logistic regression. Experimental results show that the propose
    
[^99]: 排名压缩中的比例性

    Proportionality in Ranking Compression

    [https://arxiv.org/abs/2610.23193](https://arxiv.org/abs/2610.23193)

    本文为排名压缩问题提出了三种比例性概念，证明了位置比例性和PSC总能被满足且可在多项式时间内求解，而成对比例性一般情况下无法满足，但在单峰或单交叉排名下可以实现。

    

    受人类反馈强化学习应用的启发，我们考虑这样一个场景：需要将给定的排名集合压缩成更小的排名集合。这也可以被看作是两个基本社会选择框架的结合：排名聚合与多赢家投票。我们提出了三种比例性概念，用以刻画该场景下不同类型的代表性：位置比例性、成对比例性以及固体联盟比例性（PSC）。一方面，我们证明对于任何输入排名和目标压缩规模，位置比例性和PSC总是可以被满足的，并且可以在多项式时间内找到满足要求的输出。另一方面，我们证明成对比例性在一般情况下无法满足，但当输入排名具有单峰性或单交叉性时，仍然可以实现。

    arXiv:2610.23193v1 Announce Type: cross  Abstract: Motivated by applications in reinforcement learning from human feedback, we consider a setting where a given collection of rankings needs to be compressed into a smaller collection of rankings. This can also be seen as a combination of two fundamental social choice frameworks: ranking aggregation and multiwinner voting. We propose three proportionality notions that capture different types of representation in this setting: positional proportionality, pairwise proportionality, and proportionality for solid coalitions (PSC). On the one hand, we show that positional proportionality and PSC are always satisfiable for any input rankings and target compression size, and a desired output can be found in polynomial time. On the other hand, we prove that pairwise proportionality cannot be satisfied in general, but can nevertheless be attained when the input rankings are single-peaked or single-crossing.
    
[^100]: LangNCO：将自然语言约束编译为神经组合优化的推理时干预

    LangNCO: Compiling Natural-Language Constraints into Inference-Time Interventions for Neural Combinatorial Optimization

    [https://arxiv.org/abs/2610.23190](https://arxiv.org/abs/2610.23190)

    LangNCO利用预训练大语言模型将自然语言运营需求编译为DSL规则，并通过硬动作掩码与关键偏移解码等推理时干预手段，在不改动冻结NCO求解器的架构、表示与参数的情况下满足部署后出现的新约束。

    

    神经组合优化（NCO）能够针对训练时遇到的问题结构快速生成高质量解，但运营需求可能在部署之后发生变化，并以自然语言的形式被传达。我们提出LangNCO，这是一个将实例特定需求编译为结构化推理时干预的框架，作用于被冻结的NCO求解器，无需改变预训练策略的架构、学习到的输入表示或参数。预训练的大语言模型（LLM）充当可替换的语义前端，将每条需求翻译为经过模式验证的领域特定语言（DSL）规则集。确定性的规则执行器将这些规则应用为非扩张的硬动作掩码。由于局部掩码虽然能够防止即时违规，却不能确保仍然存在一条完整可行的后续解路径，关键偏移解码会提取规则所引用的关键节点，并重新分配固定rollout的一部分（摘要在此处截断）。

    arXiv:2610.23190v1 Announce Type: cross  Abstract: Neural combinatorial optimization (NCO) rapidly produces high-quality solutions for problem structures encountered during training, but operational requirements may change after deployment and be communicated in natural language. We propose LangNCO, a framework that compiles instance-specific requirements into structured inference-time interventions for a frozen NCO solver, without changing the pretrained policy's architecture, learned input representation, or parameters. A pretrained LLM serves as a replaceable semantic front end, translating each requirement into a schema-validated domain-specific language (DSL) rule set. A deterministic Rule Executor applies the rules as nonexpansive hard action masks. Because local masking prevents immediate violations but does not ensure that a complete feasible continuation remains reachable, Critical-Offset Decoding extracts rule-referenced critical nodes and reallocates part of a fixed rollout 
    
[^101]: 内在的更强教师：通过策略内双向蒸馏增强MoE模型

    A Stronger Teacher Within: Strengthening MoE Models via On-Policy Bidirectional Distillation

    [https://arxiv.org/abs/2610.23185](https://arxiv.org/abs/2610.23185)

    提出MoEvolve框架，通过增加MoE模型每token激活预算来挖掘模型内部潜在更强教师，并采用“先恢复后提升”的策略内双向蒸馏策略强化MoE大语言模型。

    

    基于可验证奖励的强化学习（RLVR）已显著提升了混合专家（MoE）大语言模型，但其结果层面的奖励只能提供稀疏的监督信号。策略内蒸馏提供了密集的token级反馈，然而需要一个更强的兼容教师模型，而策略内自蒸馏则依赖于手工设计的特权信息。我们观察到，MoE架构本身提供了一种解决这一困境的方法：增加每个token的激活预算可以释放潜在的专家容量，从而在模型内部产生一个潜在更强的教师。然而，简单的扩展会降低性能，因为新共同激活的专家并未经过协同工作的优化。因此，我们提出了MoEvolve，一个遵循“先恢复后提升”策略的双向蒸馏框架：我们首先恢复激活扩展后的高k模型，并通过利用其额外的激活容量进一步强化它，然后将（摘要在此处截断）

    arXiv:2610.23185v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) has substantially improved Mixture-of-Experts (MoE) LLMs, but its outcome-level rewards provide only sparse supervision. On-Policy Distillation offers dense token-level feedback, yet requires a stronger compatible teacher, while on-policy self-distillation relies on hand-designed privileged information. We observe that the MoE architecture itself offers a way to resolve this dilemma, as increasing the activation budget per token exposes latent expert capacity and yields a potentially stronger teacher within the model. However, naive expansion degrades performance because newly co-activated experts are not optimized to work together. We therefore propose MoEvolve, a bidirectional distillation framework following a recovery-then-improvement strategy: we first restore the activation-expanded high-$k$ model and further strengthen it by exploiting its additional active capacity, then rev
    
[^102]: 面向长时程智能体强化学习的树感知后验蒸馏

    Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.23181](https://arxiv.org/abs/2610.23181)

    本文提出TreeOPSD，一种无需评论家的树感知后验自蒸馏框架，通过将树结构回放批次转化为有限的后验自蒸馏目标，同时解决长时程智能体强化学习中的稀疏信用分配与策略陈旧问题。

    

    具有可验证奖励的强化学习已成为大语言模型的一种可扩展的后训练范式。然而，在长时程交互中训练智能体面临两个挑战：仅使用终端奖励使得回合级别的信用分配稀疏且方差很高，而在固定的回放批次上进行顺序小批量更新会引入策略陈旧问题，因为当前策略会逐渐偏离行为策略。现有方法通过学习评论家、过程奖励模型或在线策略蒸馏来获得更密集的信用信号，但它们依赖于价值网络、步骤级标注或更强的外部教师模型，从而增加了训练复杂性。与此同时，重要性采样和裁剪可以缓解策略漂移，但它们并未指定一个有限的目标来阻止顺序更新过度。在这项工作中，我们提出了TreeOPSD，一个无评论家、树感知的后验自蒸馏框架，用于长时程智能体……

    arXiv:2610.23181v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards has become a scalable post-training paradigm for large language models. However, training agents over long-horizon interactions presents two challenges: using only terminal rewards makes turn-level credit assignment sparse and high-variance, while sequential mini-batch updates on a fixed rollout batch introduce policy staleness as the current policy drifts from the behavior policy. Existing approaches obtain denser credit via learned critics, process reward models, or on-policy distillation, but they rely on a value network, step-level annotations, or a stronger external teacher that increase training complexity. Meanwhile, importance sampling and clipping mitigate policy drift, but they do not specify a finite target that stops sequential updates from overshooting. In this work, we introduce TreeOPSD, a critic-free, tree-aware posterior self-distillation framework for long-horizon agentic
    
[^103]: 结构亲和性：一种实现连贯的基于笼体的形状设计算法

    Structural Affinity: An Algorithm for Coherent Cage-Based Shape Design

    [https://arxiv.org/abs/2610.23173](https://arxiv.org/abs/2610.23173)

    本文提出结构亲和性方法，通过双笼结构和基于均值坐标预计算的亲和矩阵直接耦合笼体-手柄运动，作为显式正则化手段实现基于笼体变形的平滑连贯形状设计。

    

    基于笼体的变形（CBD）通过一小组控制手柄对三维几何进行参数化，有效地支持从动画和交互式建模，到基于工具的协同设计、神经变形迁移以及空气动力学形状优化等多种任务。许多依赖CBD的优化方法采用正则化策略，通过鼓励平滑、规则的笼体变形来减少变形网格的失真。这些技术为优化算法提供了朝向理想变形的软引导，但它们是与任务目标相竞争，而不是改变变形参数化本身。在这项工作中，我们为CBD引入了结构亲和性，这是一种显式正则化方法，通过利用双笼结构直接耦合笼体-手柄的运动，从而实现通过控制笼体的平滑变形。结构亲和性使用均值坐标预计算一个亲和矩阵，将内部……（摘要在此处截断）

    arXiv:2610.23173v1 Announce Type: new  Abstract: Cage-based deformation (CBD) parametrises 3D geometry through a small set of control handles, effectively enabling a variety of tasks from animation and interactive modelling to tool-based co-design, neural deformation transfer, and aerodynamic shape optimization. Many optimization approaches relying on CBD employ regularisation strategies to reduce the distortion of the deformed mesh by encouraging smooth, regular cage deformations. These techniques offer soft guidance to an optimization algorithm towards desirable deformations, but compete with the task objective rather than changing the deformation parameterization itself. In this work we introduce Structural Affinity for CBD, an explicit regularisation method which enables smooth deformations through control cages by exploiting a two-cage structure to couple cage-handle motion directly. Structural Affinity precomputes an affinity matrix using mean-value coordinates, binding the inner
    
[^104]: 用自然语言解释国际象棋与围棋

    Explaining Chess and Go in Natural Language

    [https://arxiv.org/abs/2610.23166](https://arxiv.org/abs/2610.23166)

    本文提出了基于引擎评估的解释性指标和无需训练的BinProp方法，使语言模型能够发现可复用的棋类概念并以自然语言解释着法，平均得分相比直接提示提高近两倍，且该方法可泛化至围棋、跳棋和将棋等多种棋类。

    

    现代国际象棋引擎的棋力远超人类，其选择着法背后的思路往往令人类难以理解。我们提出了一种基于国际象棋引擎评估的自动化指标，用于评估棋局解释的质量。利用该指标对最先进的语言模型进行评估后发现，它们的表现出奇地差，往往依赖于对棋盘的表面观察，而非稳健、可迁移的国际象棋概念。随后，我们提出了BinProp，这是一种无需训练的方法，能够使同样的模型发现可复用的国际象棋概念，并用它们以自然语言解释着法。BinProp提升了对所有受测模型的可解释性，与直接提示相比，平均得分几乎提高了两倍。除了国际象棋之外，同一方法无需额外调整即可在围棋、西洋跳棋和将棋中发现有用的概念。

    arXiv:2610.23166v1 Announce Type: cross  Abstract: Modern chess engines play far beyond human ability, often choosing moves whose underlying ideas are difficult for humans to understand. We propose an automatic metric for evaluating chess explanations, grounded in chess-engine evaluations. Evaluating state-of-the-art language models using this metric, we find that they perform surprisingly poorly, often relying on surface-level board observations rather than robust, transferable chess concepts. We then introduce BinProp, a training-free method that enables these same models to discover reusable chess concepts and use them to explain moves in natural language. BinProp improves explainability for every model tested, nearly tripling the average score compared with direct prompting. Beyond chess, the same method discovers useful concepts in Go, draughts, and shogi without additional tuning.
    
[^105]: Littlestone类的近乎线性隐私学习

    Nearly Linear Private Learning of Littlestone Classes

    [https://arxiv.org/abs/2610.23160](https://arxiv.org/abs/2610.23160)

    该论文将Littlestone维数为 $d$ 的类在差分隐私PAC学习中的样本复杂度上界从此前最好的 $\widetilde O(d^5)$ 大幅改进为近乎线性的 $O(d\log d)$，其核心技术贡献是一种针对数据相关候选集、配备满足稳定性条件的秩的结构化指数机制。

    

    差分隐私学习旨在从数据中学习的同时保护个体训练样本。对于二分类问题，有限的Littlestone维数刻画了近似隐私PAC可学习性（Alon等，JACM 2022）。理解该维数如何控制隐私样本复杂度仍然是一个核心问题。对于Littlestone维数为 $d$ 的类，第一个一般性上界是 $d$ 的双指数级（Bun、Livni和Moran，FOCS 2020）。后续工作将其改进至 $\widetilde O(d^6)$（Ghazi等，STOC 2021），随后又改进至 $\widetilde O(d^5)$（Lyu，STOC 2026）。我们证明了近乎线性的上界 $O(d\log d)$。我们的主要算法贡献是一种针对数据相关候选集的结构化指数机制，该机制配备有满足适当稳定性条件的秩。对于诸如子空间指示器等自然的Littlestone类，该秩具有简单的几何解释：它是余维数（原文此处被截断）。

    arXiv:2610.23160v1 Announce Type: new  Abstract: Differentially private learning aims to learn from data while protecting individual training examples. For binary classification, finite Littlestone dimension characterizes approximate private PAC learnability (Alon et al., JACM 2022). Understanding how this dimension controls private sample complexity remains a central question. For classes of Littlestone dimension $d$, the first general upper bound was doubly exponential in $d$ (Bun, Livni, and Moran, FOCS 2020). Subsequent work improved this to $\widetilde O(d^6)$ (Ghazi et al., STOC 2021) and then to $\widetilde O(d^5)$ (Lyu, STOC 2026). We prove a nearly linear upper bound of $O(d\log d)$.   Our main algorithmic contribution is a structural exponential mechanism for data-dependent candidate sets equipped with a rank satisfying suitable stability conditions. For natural Littlestone classes such as subspace indicators, this rank has a simple geometric interpretation: it is the codimen
    
[^106]: CIIR——类索引接口表示学习

    CIIR - Class-Indexed Interface Representation Learning

    [https://arxiv.org/abs/2610.23151](https://arxiv.org/abs/2610.23151)

    CIIR提出了一种模块化编码器-解码器框架，利用类索引地址空间在无需配对样本的情况下连接异构数据集，并在冻结基础地址空间后仅通过轻量级适配器即可扩展至新数据集。

    

    多模态学习通常假设存在成对或同步的观测数据，然而独立收集的数据集很少能提供这种对应关系。我们提出CIIR，这是一个模块化的编码器-解码器框架，通过类索引地址空间连接异构数据集，同时保留每个数据集的原生输入格式和输出模块。一个已知的本地索引用于选择地址k，随后将与输入相关的128维地址表示Z_k(x)写入该地址。为了展示CIIR，我们在没有同步或实例匹配样本的情况下，从手写数字、服装图像、口述数字频谱图和手语手势中学习该地址空间。新训练的无标签接收器以94.7%的宏平均地址实现准确率成功恢复所请求的地址。随后，我们在冻结基础地址空间后接入未见过的SVHN数据集，仅训练轻量级适配器，实现了90.2%的MNIST>SVHN和82.3%的SVHN>MNIST迁移性能。

    arXiv:2610.23151v1 Announce Type: new  Abstract: Multimodal learning commonly assumes paired or synchronized observations, yet independently collected datasets rarely provide such correspondence. We introduce CIIR, a modular encoder--decoder framework that connects heterogeneous datasets through a class-indexed address space while preserving each dataset's native input format and output module. A known local index selects address $k$, and an input-dependent 128-dimensional address representation Z_k(x) is written to that address. To demonstrate CIIR, we learn this address space from handwritten digits, clothing images, spoken-digit spectrograms, and hand signs without synchronized or instance-matched samples. Newly trained label-free receivers recover the requested addresses with 94.7% macro-average Address Realization Accuracy. We then attach the unseen SVHN dataset after freezing the base address space and train only lightweight adapters, achieving 90.2% MNIST>SVHN and 82.3% SVHN>MNI
    
[^107]: 多轮搜索智能体强化学习中的接地信用分配

    Grounded Credit Assignment in Reinforcement Learning for Multi-Turn Search Agents

    [https://arxiv.org/abs/2610.23148](https://arxiv.org/abs/2610.23148)

    提出接地信用分配（GCA）方法，利用可验证的子结果（证据检索与答案正确性）分别校正搜索与答案的信用，解决GRPO等结果奖励强化学习在训练多轮搜索智能体时导致答案缺乏证据支撑的问题。

    

    结果导向奖励的强化学习（如GRPO）为搜索智能体轨迹中的每个标记赋予相同的优势，尽管该轨迹实际上运用了两种能力：查找证据和使用证据。我们证明准确性可以在缺乏证据支撑的情况下提升：Search-R1在全部七个问答基准上提高了准确性，但在Bamboogle上，有证据支撑的正确答案比例从84.0%降至42.9%。我们将此归因于GRPO的信用分配机制：检索到含答案证据的轨迹在总体上仅按其相对于其他轨迹的准确性增益比例获得优待，而凭记忆作答或滥用证据会缩小甚至逆转这种证据支撑差距。为解决这一问题，我们提出接地信用分配（GCA），利用可验证的子结果（证据检索与答案正确性）分别校正搜索信用与答案信用：错误答案所受的责备应多于找到证据的搜索过程，而绕过检索直接作答的正确答案……（原文摘要在此处截断）

    arXiv:2610.23148v1 Announce Type: new  Abstract: Outcome-reward reinforcement learning such as GRPO gives every token of a search-agent rollout the same advantage, although the rollout exercises two capabilities: finding evidence and using it. We show that accuracy improves without grounding: Search-R1 raises accuracy on all seven QA benchmarks, yet on Bamboogle the evidence-supported share of correct answers falls from 84.0% to 42.9%. We attribute this to GRPO's credit assignment: rollouts that retrieve answer-bearing evidence are favoured in aggregate only in proportion to their accuracy gain over the rest, a grounding gap that answering from memory or misusing evidence shrinks or reverses. To address this, we propose Grounded Credit Assignment (GCA), which uses verifiable sub-outcomes (evidence retrieval and answer correctness) to correct search and answer credit separately: a wrong answer is blamed more than the search that found the evidence, and a correct answer that bypasses ret
    
[^108]: 用于内存高效训练的局部代理网络

    Local proxies for memory-efficient training

    [https://arxiv.org/abs/2610.23145](https://arxiv.org/abs/2610.23145)

    提出为每个Transformer块训练局部代理网络，通过蒸馏预测全局输出使梯度可局部独立传播，在大幅降低训练内存占用的同时保持接近标准反向传播的梯度估计精度。

    

    局部学习算法能够重塑人工智能的资源需求。标准端到端反向传播的内存瓶颈在于必须存储所有网络激活，直到损失梯度反向传播完成。局部学习可以避免这一问题，但通常以精度下降为代价。我们研究了如何在Transformer中实现准确且内存高效的局部学习。对于每个Transformer块，我们通过知识蒸馏训练一个局部代理网络来预测全局网络输出。梯度可以独立地通过该局部代理进行传播，同时能够很好地估计反向传播梯度。理论上，该方法对于深度线性网络是精确的；在非线性残差模型中，使用适当的残差代理可以将梯度误差限定在边界内。我们通过数值实验验证了这一点，并证明优化蒸馏损失可以改善梯度估计。

    arXiv:2610.23145v1 Announce Type: cross  Abstract: Local learning algorithms can reshape resource requirements of artificial intelligence. The memory bottleneck of standard end-to-end backpropagation lies in the need to store all network activations until the loss gradient has been propagated back through. Local learning avoids this, but typically at a cost in accuracy. We study how to achieve accurate and memory efficient local learning in transformers. For each transformer block, we train a local proxy network to predict the global network output via distillation. Gradients can be independently propagated through this local proxy while estimating well the backpropagation gradients. In theory, this approach is exact for deep linear networks; in non-linear residual models, the gradient error can be bounded when using the appropriate residual proxy. We verify this numerically and demonstrate that optimizing the distillation loss improves gradient estimation. Evaluating the performance o
    
[^109]: MetaCurate：演化筛选过滤器以获取高质量智能体编程数据

    MetaCurate: Evolving Curation Filters for High-Quality Agentic Coding Data

    [https://arxiv.org/abs/2610.23134](https://arxiv.org/abs/2610.23134)

    MetaCurate 提出了一种无需训练的框架，通过大语言模型成对判断产生的反馈自动演化可执行的轨迹筛选过滤器，从而高效地为智能体编程训练挑选高质量数据样本。

    

    对智能体轨迹进行监督微调是提升编程智能体能力的有效方法，但挑选有用的训练样本不仅仅是识别成功的运行结果。一个成功解决的补丁可能伴随着低效的推理或不必要的工具调用，而一次失败的运行中仍可能包含有用的中间步骤。因此，训练流程通常依赖规则和定性评估来过滤轨迹，然而设计这些过滤器往往需要大量的人类专业知识，并且通过反复训练和评估来验证改进的效果成本高昂。我们提出了 MetaCurate，这是一个无需训练的框架，利用大语言模型（LLM）的成对判断来自动演化可执行的轨迹筛选过滤器。过滤器排序与评判模型偏好之间的分歧为联合修订其规则、评分标准（rubric）和聚合策略提供了有针对性的反馈。最终得到的过滤器能够独立地对各条轨迹进行评分，因此成对比较……

    arXiv:2610.23134v1 Announce Type: new  Abstract: Supervised fine-tuning on agent trajectories is an effective way to improve coding agents, but selecting useful training examples requires more than identifying successful runs. A resolved patch may follow inefficient reasoning or unnecessary tool use, while a failed run may still contain useful intermediate steps. Training pipelines therefore rely on rules and qualitative assessments to filter trajectories, yet designing these filters typically requires substantial human expertise, and validating improvements through repeated training and evaluation is expensive. We introduce MetaCurate, a training-free framework that evolves executable trajectory curation filters using pairwise LLM judgments. Disagreements between the filter's rankings and the judge's preferences provide targeted feedback for jointly revising its rules, rubric, and aggregation policy. The resulting filter scores trajectories independently, so pairwise comparisons are r
    
[^110]: 共形检验鞅的信息几何盲区

    Information-Geometric Blind Spots of Conformal Test Martingales

    [https://arxiv.org/abs/2610.23131](https://arxiv.org/abs/2610.23131)

    该论文揭示了共形检验鞅的信息几何盲区：凡保持符合性函数前推分布不变的分布偏移均无法被检测，并借助 A-等价类与 A-隐秘 I-投影给出该盲区的显式刻画与勾股分解，同时证明当符合性函数取 Neyman-Pearson 似然比时盲区消失。

    

    共形检验鞅（CTM）是通过由符合性函数 A 生成的 p 值来序贯检验可交换性的 e-过程。然而，任何保持前推分布 P∘A⁻¹ 不变的分布偏移都会使 p 值的分布保持不变，从而对所有基于 A 的共形检验鞅而言都是不可察觉的。我们通过 A-等价类 [P]_A 对这一盲区进行形式化，并证明对于任意目标分布 T，在 KL 散度意义下 [P]_A 中存在唯一的最近元素 Q*——即 A-隐秘 I-投影，其具有显式密度 q*(z) = t(z) p_S(A(z))/t_S(A(z))，并可分解为可检测分量与隐秘分量的勾股分解。当 A 取为 Neyman-Pearson 似然比时，隐秘分量消失，由此将我们的结果与最优 e 值设计联系起来。我们通过在高维图像数据上经验性地构造该 I-投影来展示这一理论，通过条件重采样 EMNIST……（原文摘要在此处截断）

    arXiv:2610.23131v1 Announce Type: cross  Abstract: Conformal test martingales (CTMs) are e-processes for sequentially testing exchangeability via p-values produced by a conformity function $A$. However, any distribution shift that preserves the push-forward $P\circ A^{-1}$ leaves the p-value distribution invariant, rendering it invisible to every CTM based on $A$. We formalise this blind spot through the \emph{$A$-equivalence class} $[P]_A$ and show that, for any target distribution $T$, there is a unique closest element $Q^\ast\in[P]_A$ in KL divergence---the \emph{$A$-cryptic I-projection}---with explicit density $q^\ast(z) = t(z)\,p_S(A(z))/t_S(A(z))$ and a Pythagorean decomposition into detectable and cryptic components. When $A$ is the Neyman-Pearson likelihood ratio, the cryptic component vanishes, connecting our result to optimal e-value design. We demonstrate the theory on high-dimensional image data by empirically constructing this I-projection. By conditionally resampling EMN
    
[^111]: LED：通过前瞻编辑改进掩码扩散语言模型

    LED: Improving Masked Diffusion Language Models by Lookahead Editing

    [https://arxiv.org/abs/2610.23122](https://arxiv.org/abs/2610.23122)

    提出前瞻编辑方法（LED），通过在掩码扩散解码中引入可重写的提议状态，使掩码扩散语言模型在不牺牲其易优化性的前提下获得迭代修正的表达能力。

    

    扩散大语言模型已成为自回归语言模型的一种有前景的替代方案，能够通过并行解码实现非自回归的文本生成。现有的 dLLMs 大致可分为两大类：掩码扩散模型（MDMs）和均匀扩散模型（UDMs）。在本工作中，我们从理论上证明：MDMs 由于其基于掩码的转移调度而更易于优化，但其计算表达力较低；而 UDMs 支持迭代式的状态修正并具有更强的表达力，但训练难度显著更高。这一权衡引出了一个自然的问题：我们能否在不放弃 MDMs 简单易优化的转移调度的前提下，获得迭代修正所带来的表达力？基于这一分析，我们提出了前瞻编辑（Lookahead Editing，LED），它通过引入一个可重写的提议状态来增强掩码扩散解码。在每个解码步骤中，模型会执行若干前瞻……

    arXiv:2610.23122v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive language models, enabling non-autoregressive text generation through parallel decoding. Existing dLLMs can be broadly categorized into two families: masked diffusion models (MDMs) and uniform diffusion models (UDMs) In this work, we theoretically show that MDMs are easier to optimize due to their mask-based transition schedule but have lower computational expressivity, whereas UDMs enable iterative state revision and greater expressivity but are substantially harder to train. This trade-off raises a natural question: can we obtain the expressivity of iterative revision without giving up the simple, easy-to-optimize transition schedule of MDMs? Motivated by this analysis, we propose Lookahead Editing (\ours{}), which augments masked diffusion decoding with a rewritable proposal state. At each decoding step, the model performs several lookah
    
[^112]: 面向可解释预测的多模态嵌入的领域特定语言对齐

    Domain-specific language alignment of multimodal embeddings towards explainable predictions

    [https://arxiv.org/abs/2610.23110](https://arxiv.org/abs/2610.23110)

    该论文提出一个无需带描述训练数据的语言对齐框架，可将任何关联了可解释辅助数据的嵌入空间与语言对齐，从而实现多模态嵌入的可解释预测。

    

    多模态数据通常被融合为潜在表示，这类表示能够提升下游任务性能，但缺乏可解释性，限制了其在需要可解释模型决策的领域中的应用。视觉-语言对齐已能通过语言使图像嵌入变得可解释，但这需要海量的图像-描述训练数据。然而，对于多模态嵌入而言，带描述的数据通常不可获得，因为相关特征无法仅凭嵌入本身被直接观察和标注。在此，我们提出了一个框架，用于将任何能够与可解释辅助数据相关联的嵌入空间进行语言对齐。该框架通过描述生成函数将辅助数据转换为部分描述；采用一种软对比损失，根据各描述所刻画的属性相似度对样本对进行加权；并利用一个基于排序的概念检索任务，依据辅助数据而非描述本身来评估对齐效果。我们将此框架应用于多模态Ea……（原文摘要在此处截断）

    arXiv:2610.23110v1 Announce Type: new  Abstract: Multimodal data are commonly fused into latent representations that improve downstream task performance but are not interpretable, limiting their application in domains that require explainable model decisions. Vision-language alignment has made image embeddings interpretable through language, but requires vast image-caption training data. However, captioned data are typically not available for multimodal embeddings, as relevant features cannot directly be observed and annotated from embeddings alone. Here, we propose a framework to language-align any embedding space that can be linked to interpretable auxiliary data. Captioning functions convert auxiliary data into partial captions; a soft contrastive loss weights pairs by the similarity of their properties that each caption describes; and a rank-based concept retrieval task evaluates alignment against the auxiliary data rather than the captions. We apply this framework to multimodal Ea
    
[^113]: 面向结构动力学的GraphNARX模型

    Towards a GraphNARX model for structural dynamics

    [https://arxiv.org/abs/2610.23103](https://arxiv.org/abs/2610.23103)

    GraphNARX是一种基于图的结构动力学非线性自回归外生模型，能够考虑结构几何特性进行系统辨识，可泛化至不同维度和自由度的结构，并支持虚拟传感功能。

    

    本文提出了GraphNARX模型，这是一种基于图的结构动力学非线性自回归外生模型。该模型旨在考虑结构的几何特性，并将动态行为建模为结构各单元之间的相互作用。与其他类型的NARX模型类似，GraphNARX用于系统辨识。图的节点属性编码了过去的位移和节点力，模型的输出为下一时间步的位移。该模型的图表示形式使其能够泛化到不同维度和自由度的结构。此外，由于图学习是在节点之间局部进行的，GraphNARX模型还可以学习推断没有可用数据的节点位移，即执行虚拟传感。所提出模型的性能在系统辨识框架和虚拟传感框架下进行了评估，并取得……

    arXiv:2610.23103v1 Announce Type: new  Abstract: The current paper introduces the GraphNARX model, a graph-based nonlinear autoregressive exogenous model for structural dynamics. The model is built to account for the geometry of structures and to model dynamic behaviour as the interaction between elements of the structure. Similar to other types of NARX models, GraphNARX is used for system identification. Nodal attributes of the graph encode past displacements and nodal forces, and the output of the model is the next-timestep displacement. The graph formulation of the model allows it to generalise to structures of different dimensions and degrees of freedom. Moreover, because graph-learning is performed locally between the nodes, the GraphNARX model can also learn to infer displacements of nodes for which data are not available, i.e.\ to perform virtual sensing. The proposed model's performance is evaluated in a system-identification framework and a virtual-sensing framework, and achie
    
[^114]: 潜变量属于何处以及如何流动：结构角色的联合识别

    Where Latents Belong and How They Flow: Joint Identification of Structural Roles

    [https://arxiv.org/abs/2610.23074](https://arxiv.org/abs/2610.23074)

    该论文形式化了动力系统中潜在因子的“结构角色”概念，证明了表达位置、接收过去影响与传递未来影响这三种角色可在共同置换下于每个时间点联合识别，并据此开发了在共享潜在坐标系中追踪结构角色的RoleTrace框架。

    

    动力系统中的潜在因子在不同观测和时间上可以扮演不同的角色。例如，小组会议中的一个想法可能只由部分参与者表达，从早先的讨论中继承信息，并影响接下来所说的内容。我们将这一概念形式化为潜在因子的“结构角色”，用以描述它在何处被表达、是否从过去接收影响，以及是否向未来传递影响。主要挑战在于对应关系问题，因为这三种结构可能在不同置换下被识别，而结构角色只有在它们对齐到同一潜在坐标时才能被定义。我们证明，在适当的结构与变异性条件下，所有三种角色在每个时间点上都可以在共同置换下被联合识别。基于这一理论，我们开发了RoleTrace，一个在共享潜在坐标系中追踪结构角色的框架。合成实验……

    arXiv:2610.23074v1 Announce Type: new  Abstract: Latent factors in dynamical systems can play different roles across observations and time. For example, an idea in a group meeting may be expressed by only some participants, inherit information from earlier discussions, and shape what is said next. We formalize this notion as the structural role of a latent factor, describing where it is expressed, whether it receives influence from the past, and whether it transmits influence to the future. The main challenge is correspondence, since the three structures may be identified under different permutations, while a structural role is defined only when they are aligned to the same latent coordinate. We show that, under suitable structural and variability conditions, all three roles are jointly identifiable under a common permutation at each time point. Guided by this theory, we develop RoleTrace, a framework that traces structural roles in a shared latent coordinate system. Synthetic experime
    
[^115]: 基于学习的潜在多表表示通往先验数据拟合关系模型的路径

    Path to Prior-Data Fitted Relational Models via Learned Latent Multi-table Representations

    [https://arxiv.org/abs/2610.23073](https://arxiv.org/abs/2610.23073)

    该论文首次将先验数据拟合（PFN）范式拓展到基于消息传递的图关系模型，通过在合成关系数据库上预训练Griffin编码器，并将其潜在表示输入冻结的上下文内预测器，在目标任务零训练的情况下实现了对真实世界关系分类任务的泛化，并验证了课程学习对消息传递表示的增益。

    

    先验数据拟合网络（PFNs）表明，在大量合成任务上进行预训练能够使模型泛化到未见过的数据集，而几乎无需或完全无需下游微调。这一范式在基于Transformer的表格和关系基础模型（如RDB-PFN）上已被证明有效，但对于基于消息传递构建的图基关系模型而言，该范式在很大程度上仍待探索。我们在大规模合成关系数据库集合上预训练图基关系基础模型Griffin，并探究对基于Transformer的关系学习者有益的课程学习，是否同样能改进消息传递表示。我们没有在下游任务上进行微调，而是将预训练编码器作为通用的关系表示学习器，并将其潜在表示传递给一个冻结的上下文内预测器（TabICL）；没有任何组件在目标任务上接受训练。在16个真实世界的关系分类任务上进行了评估……

    arXiv:2610.23073v1 Announce Type: cross  Abstract: Prior-data fitted networks (PFNs) show that pretraining on large populations of synthetic tasks can enable generalization to unseen datasets with little or no downstream fine-tuning. This paradigm is effective for transformer-based tabular and relational foundation models such as RDB-PFN, but remains largely unexplored for graph-based relational models built on message passing. We pretrain the graph-based relational foundation model Griffin on large collections of synthetic relational databases and ask whether curriculum learning, which benefits transformer-based relational learners, also improves message-passing representations. Rather than fine-tune on downstream tasks, we use the pretrained encoder as a general-purpose relational representation learner and pass its latent representations to a frozen in-context predictor (\ticl{}); no component is trained on the target task. On 16 real-world relational classification tasks, evaluated
    
[^116]: 通过“模型到策略”的公式化表述统一具身基础模型

    Unifying Embodied Foundation Models via a Model-to-Policy Formulation

    [https://arxiv.org/abs/2610.23053](https://arxiv.org/abs/2610.23053)

    本文提出一种基于随机核的、与架构无关的统一概率表述，将VLA模型与世界动作模型等具身基础模型纳入“模型到策略”的统一框架，贯通预训练、后训练与策略部署全过程。

    

    视觉-语言-动作（VLA）模型和世界-动作模型（WAM）已成为机器人学习与控制领域颇具前景的方法。这些模型融合了多模态感知与动作生成，但在架构、输出表示和学习目标上各不相同。本文提出了一种基于随机核的、与具体架构无关的统一概率表述，用于刻画具身基础模型。我们的表述包含两个关键组成部分：（i）通过条件输出分布和动作实现核来刻画具身基础模型，二者共同导出任务条件化的机器人策略；（ii）将预训练与后训练连接到策略部署，从而兼容模型特定的预训练目标以及离线或在线的后训练策略。我们在OpenVLA和DreamZero上对该表述进行了实例化，涵盖自回归动作token预测与联合视频-动作生成（原文摘要在此处截断）。

    arXiv:2610.23053v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models and world-action models (WAMs) have emerged as promising approaches to robot learning and control. These models integrate multimodal perception and action generation, but differ in their architectures, output representations, and learning objectives. In this paper, we present a unified, architecture-agnostic probabilistic formulation for embodied foundation models based on stochastic kernels. Our formulation has two key components: (i) we characterize embodied foundation models through conditional output distributions and action-realization kernels, which together induce task-conditioned robot policies; and (ii) we connect pretraining and post-training to policy deployment, accommodating model-specific pretraining objectives and offline or online post-training strategies. We instantiate the formulation for OpenVLA and DreamZero, covering autoregressive action-token prediction and joint video-action g
    
[^117]: DIFFLET：AWS Trainium 上扩散 Transformer 模型的快速推理引擎

    DIFFLET: A Fast Inference Engine for Diffusion Transformer Models on AWS Trainium

    [https://arxiv.org/abs/2610.23047](https://arxiv.org/abs/2610.23047)

    Difflet 是面向 AWS Trainium 的扩散 Transformer 服务引擎，通过五个流水线共享统一编译后端、按注意力头切分的融合注意力以及基于设备端探针的逐步缓存决策，将去噪步骤加速 1.2-4.3 倍，常驻服务速度提升 1.5-3.4 倍。

    

    在 AWS Trainium2 上，扩散模型的每个组件都以针对固定形状预先编译的程序形式运行，而流水线在不同模型之间各不相同，提示词长度在不同请求之间各不相同，缓存决策在不同去噪步骤之间各不相同。我们提出了 Difflet——一种服务引擎，其中五个模型流水线共享同一个编译后端，联合注意力按注意力头在多个核心之间切分，主机控制器在每个去噪步骤决定是运行去噪器还是复用其先前的输出。一个小型探针程序在设备上计算缓存信号并每步返回一个数值，一个常驻模型进程使所有程序在请求之间保持加载状态。在五个图像和视频模型家族中，融合注意力使去噪器步骤相比使用 PyTorch/XLA 注意力的同一框架加速 1.2-4.3 倍，并且在 32.8k 视觉 token 规模下，它是唯一能够完成编译的注意力实现。在五个模型中的四个上，常驻服务快 1.5-3.4 倍（摘要原文在此处截断）。

    arXiv:2610.23047v1 Announce Type: cross  Abstract: On AWS Trainium2, every component of a diffusion model runs as a program compiled ahead of time for fixed shapes, while pipelines differ across models, prompt lengths across requests, and caching decisions across denoising steps. We present Difflet, a serving engine in which five model pipelines share one compilation backend, joint attention is split across cores by attention head, and a host controller decides at each denoising step whether to run the denoiser or reuse its earlier output. A small probe program computes the caching signal on the device and returns one number per step, and a resident model process keeps every program loaded across requests. Across five image and video model families, fused attention accelerates denoiser steps by 1.2-4.3x over the same framework using PyTorch/XLA attention, and at 32.8k visual tokens it is the only attention that compiles. On four of the five models, resident serving is 1.5-3.4x faster t
    
[^118]: GR-OPD：基于组相对序列信度的在线策略蒸馏

    GR-OPD: Group-Relative Sequence Credit for On-Policy Distillation

    [https://arxiv.org/abs/2610.23042](https://arxiv.org/abs/2610.23042)

    GR-OPD提出了一种组相对的序列级信度分配方法，通过聚合词元级教师反馈、折扣不可靠的后续反馈并利用同提示采样组校准指导，解决了在线策略蒸馏中局部信度分配的缺陷，并从理论上证明了其对反向KL目标梯度的无偏估计性质。

    

    在线策略蒸馏（On-Policy Distillation, OPD）能够提供词元级别的教师反馈，然而常见的采样词元更新方式只在局部进行信度分配，忽略了早期动作对后续反馈的影响。我们提出了GR-OPD（组相对在线策略蒸馏），该方法通过将词元反馈聚合为经过校准的序列级信度，将反馈计算与信度分配解耦。我们为这一方法建立了理论基础：在理想的在线策略假设下，将原始的反馈总和广播到每个响应词元上，可得到序列级反向KL目标负梯度的无偏估计。为了在实践中有助于有效学习，GR-OPD对在分布偏移下可能变得不可靠的后续反馈进行折扣处理，并利用相同提示词的采样组来校准教师指导。该方法在保留相对响应偏好的同时，保留了受控比例的组均值，我们称之为……

    arXiv:2610.23042v1 Announce Type: new  Abstract: On-policy distillation (OPD) provides token-level teacher feedback, yet common sampled-token updates assign credit locally and omit the effects of earlier actions on subsequent feedback. We introduce \method{} (Group-Relative On-Policy Distillation), which decouples feedback computation from credit assignment by aggregating token feedback into calibrated sequence-level credit. We establish a theoretical basis for this approach by showing that, under ideal on-policy assumptions, broadcasting raw summed feedback to every response token yields an unbiased estimate of the negative gradient of the sequence-level reverse-KL objective. To support effective learning in practice, \method{} discounts later feedback that may become less reliable under distribution shift and uses same-prompt rollout groups to calibrate teacher guidance. It preserves relative response preferences while retaining a controlled fraction of the group mean, which we term 
    
[^119]: 迈向正确的循环模型（第一部分）：拓扑结构、输入注入与循环状态设计

    Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design

    [https://arxiv.org/abs/2610.23036](https://arxiv.org/abs/2610.23036)

    该论文通过控制变量的系统消融研究，分离了循环语言模型中拓扑结构、输入注入与循环状态设计三大维度，发现“前奏-循环块-尾声”三明治架构配合输入注入能持续提升推理与上下文利用能力，并证明完整的Huginn架构在全部十个基准测试中优于Ouro。

    

    循环语言模型在逻辑深度上复用物理Transformer块，但现有架构将循环位置、输入注入与循环状态组织三者纠缠在一起。我们在参数规模、逻辑深度和token预算相匹配、从零训练的模型中分离出这些设计维度，随后检验由此得出的设计原则能否迁移到常驻参数与激活参数均匹配的混合专家模型中。“前奏-循环块-尾声”三明治架构在上下文提取和推理密集型任务上持续带来性能提升。输入注入主要增强知识检索和上下文利用，而随机初始化和分离的H/L循环状态则带来混合甚至负面的收益。综合来看，这些结果支持Huginn式设计：在7.3亿参数、3360亿token的规模下，完整的Huginn架构在全部十个基准测试中均优于Ouro。

    arXiv:2610.23036v1 Announce Type: new  Abstract: Looped language models reuse physical Transformer blocks across logical depth, but established architectures entangle recurrence placement, input injection, and recurrent state organization. We isolate these axes in models trained from scratch at matched parameter scale, logical depth, and token budgets, and then test whether the resulting design principles transfer to resident- and active-parameter-matched Mixture-of-Experts (MoE) models. The prelude-recurrent block-coda sandwich architecture consistently improves performance on context extraction and reasoning-intensive tasks. Input injection primarily enhances knowledge retrieval and context utilization, whereas random initialization and separate H/L recurrent states provide mixed or negative returns. Taken together, these results favor the Huginn-style design: at the 730M-parameter and 336B-token scale, the complete Huginn architecture outperforms Ouro across all ten benchmarks. The 
    
[^120]: 双稀疏尖峰Wigner模型的可区分性与恢复

    Distinguishability and Recovery for a Doubly Sparse Spiked Wigner Model

    [https://arxiv.org/abs/2610.23033](https://arxiv.org/abs/2610.23033)

    本文研究信号与噪声均稀疏的双稀疏尖峰Wigner模型，提出留一二次阈值化、符号同步及谱精炼等算法，在信噪比保持常数阶的条件下实现了稀疏秩一信号的支撑识别、精确恢复与强恢复，并给出了信号稀疏性、噪声稀疏性与计算成本之间的显式权衡刻画。

    

    我们通过多种算法和参数区域，研究具有独立稀疏噪声的尖峰Wigner模型中稀疏秩一信号的可区分性与恢复问题。信号和噪声具有各自独立的稀疏性参数，而信噪比（SNR）在矩阵规模增大时保持常数阶。对于亚高斯噪声以及非零坐标有界远离零的尖峰信号，一种留一二次阈值化程序（Ding等人2024年工作的推广）能够识别信号的支撑集。通过符号同步可对Rademacher尖峰实现精确恢复，通过谱精炼可实现强恢复。二次阈值化的搜索参数给出了信号稀疏性、噪声稀疏性与计算成本之间的显式权衡。我们还通过对角阈值化和二次阈值化研究了向亚高斯尖峰先验的推广。对于具有连续非零噪声的Rademacher尖峰，我们开发了一个可……

    arXiv:2610.23033v1 Announce Type: cross  Abstract: We study distinguishability and recovery of a sparse rank-one signal in a spiked Wigner model with independently sparse noise through a variety of algorithms and parameter regimes. The signal and noise have separate sparsity parameters, while the signal-to-noise ratio (SNR) remains of constant order as the matrix size grows. For sub-Gaussian noise and spikes whose nonzero coordinates are bounded away from zero, a leave-one-out quadratic thresholding procedure (a generalization of Ding et al., 2024) identifies the support. Sign synchronization yields exact recovery for a Rademacher spike, and spectral refinement yields strong recovery. The quadratic thresholding search parameter gives an explicit tradeoff between signal sparsity, noise sparsity, and computational cost. We also study extensions to sub-Gaussian spike priors through diagonal and quadratic thresholding. For Rademacher spikes with continuous nonzero noise, we develop an iden
    
[^121]: “深度提升”：即使使用最“愚蠢”的优化器，神经网络也能实现泛化

    "Deep lifts": Neural networks can generalize even with the dumbest optimizer

    [https://arxiv.org/abs/2610.23009](https://arxiv.org/abs/2610.23009)

    该论文证明，即使以最“愚蠢”的方式——从过大的 ReLU 网络的拟合参数中均匀随机选取参数——只要数据可由更小的网络解释，所选参数仍能以 $\tilde O(wdWD/n)$ 的速率自适应泛化，其根源是 ReLU 网络的一种几何归纳偏置：小网络参数“见证”了指数级体积的等价大网络拟合参数，使其远超泛化差的参数体积。

    

    给定一个带标签的数据集 $\{(x_i, y_i)\}_{i = 1}^n$，考虑在一个宽度为 $W$、深度为 $D$、连续权重有界于 $[-M, M]$ 的 ReLU 网络架构中，从所有能够拟合（插值）该数据的参数向量中均匀随机地抽取一个参数向量 $\hat{\theta}$。我们研究当该数据集实际上可以由一个更小的 $w \times d$ ReLU 网络解释时，$\hat{\theta}$ 的泛化性质。尽管对 $\hat{\theta}$ 的选择规则是完全“盲目”的，它仍能以 $\tilde O\left(\frac{wdWD}{n}\right)$ 的速率自适应地泛化，而不受完整的 $W \times D$ 假设类 VC 维的支配。其原因在于 ReLU 网络中一种特殊的归纳偏置：$w \times d$ 的参数“见证”了指数级体积的 $W \times D$ 拟合参数，这使得（在拟合条件下）泛化能力差的参数体积相形见绌。我们显式地构造了这组近似等价的……

    arXiv:2610.23009v1 Announce Type: new  Abstract: Given a labeled dataset $\{(x_i, y_i)\}_{i = 1}^n$, consider drawing a parameter vector $\hat{\theta}$ uniformly at random from those that interpolate the data within a width-$W$ depth-$D$ ReLU architecture with continuous weights bounded within $[-M, M]$. We study the generalization properties of $\hat{\theta}$ in the case where the dataset can, in fact, be explained by a smaller $w$-by-$d$ ReLU network. Despite the obliviousness of the selection rule for $\hat{\theta}$, it adaptively generalizes at the rate $\tilde O\left(\frac{wdWD}{n}\right)$, rather than being governed by the VC dimension of the full $W$-by-$D$ hypothesis class. The reason is due to a particular inductive bias in ReLU networks: the $w$-by-$d$ parameter *witnesses* an exponential volume of $W$-by-$D$ interpolating parameters, which dwarfs the volume of poorly generalizing parameters (conditioned on interpolation). We explicitly construct this set of near-equivalent $
    
[^122]: 非线性可分数据上的极简ReLU分类器：稳定性边缘及其之外的收敛性与隐式偏置

    A Minimal ReLU Classifier on Non-Linearly Separable Data: Convergence and Implicit Bias at and beyond the Edge of Stability

    [https://arxiv.org/abs/2610.23007](https://arxiv.org/abs/2610.23007)

    该论文在一个极简的非凸非光滑设置（双神经元ReLU网络在一维非线性可分数据上以逻辑损失训练）中，对大学习率下梯度下降的稳定性边缘行为给出了严格理论分析，证明了有限时间损失界，并刻画了GD方向上收敛至显式最大间隔向量这一隐式偏置。

    

    即使违反了经典的基于平滑性的学习率条件，采用大学习率的梯度下降（GD）仍能成功地优化神经网络。然而，针对这一情形的理论研究大多局限于线性模型或线性可分数据。我们分析了同时超出这两种情形的一个极简设置：具体而言，是在一维非线性可分数据上用逻辑损失训练的双神经元ReLU网络。该目标函数非凸且非光滑，且GD表现出稳定性边缘（EoS）行为。我们识别出参数空间中的一个活跃锥，它包含所有完美分类器；在该锥内，问题转化为具有变化有效数据集的分段逻辑回归。在该锥内，我们针对恒定学习率证明了有限时间损失上界，并证明GD在方向上收敛到一个显式的最大间隔向量，而其正交分量则收敛到一个非强凸势函数的唯一极小值点。我们进一步建立了（摘要在此处被截断）

    arXiv:2610.23007v1 Announce Type: new  Abstract: Gradient descent (GD) with large learning rates can successfully optimize neural networks even though classical smoothness-based learning rate conditions are violated. Yet theory for this regime largely covers linear models or linearly separable data. We analyze a minimal setting outside both regimes. Specifically, a two-neuron ReLU network trained with logistic loss on 1D non-linearly separable data. The objective is nonconvex and nonsmooth, and GD exhibits edge-of-stability (EoS) behavior. We identify an active cone in parameter space that contains every perfect classifier, where the problem becomes piecewise logistic regression with a varying effective dataset. Within this cone, we prove a finite-time loss bound for constant learning rates and show that GD converges in direction to an explicit max-margin vector, while the orthogonal component converges to the unique minimizer of a non-strongly convex potential. We further establish ca
    
[^123]: 通过合并对齐前后的模型来控制大语言模型的多样性-质量权衡

    Controlling the Diversity-Quality Trade-Off of LLMs by Merging Models Before and After Alignment

    [https://arxiv.org/abs/2610.22999](https://arxiv.org/abs/2610.22999)

    提出一种无需额外训练的深度感知 FrankenMerging 方法，通过将对齐模型的低层与基础模型的高层进行合并，可灵活控制大语言模型输出中词汇与语义的多样性及质量之间的权衡。

    

    对齐被认为会改变大语言模型（LLM）的输出特性。基础模型往往生成更加多样的输出，而对齐模型则倾向于产生更高质量的输出。两者之间的适当平衡取决于具体任务。现有方法要么需要代价高昂的微调，要么像温度采样那样主要提升词汇多样性而非语义多样性。我们提出了一种深度感知的 FrankenMerging 方法，用于在无需额外训练的情况下控制对齐模型与其基础模型之间的多样性-质量权衡。该方法基于这样一个假设：词汇多样性与模型的较低层关系更密切，而语义多样性与较高层关系更密切，因此将一个对齐模型的较低层与基础模型的较高层相结合。实验表明，改变从每个模型中选取的层数可以同时调节多样性和质量。在大多数情况下，该方法……

    arXiv:2610.22999v1 Announce Type: new  Abstract: Alignment is known to change the output characteristics of large language models (LLMs). Base models tend to generate more diverse outputs, whereas aligned models tend to produce higher-quality outputs. The appropriate balance between the two depends on the task. Existing methods either require costly fine-tuning or, like temperature sampling, mainly improve lexical diversity rather than semantic diversity. We introduce a depth-aware FrankenMerging approach for controlling the diversity--quality trade-off between an aligned model and its base model without additional training. This approach combines the lower layers of an aligned model with the higher layers of its base model based on the hypothesis that lexical diversity is more closely associated with lower layers and semantic diversity with higher layers. Experiments showed that varying the number of layers taken from each model adjusts both diversity and quality. In most cases, the m
    
[^124]: 一种基于邻域标签分布的图数据标签噪声鲁棒算法

    A Label Noise Robust Algorithm for Graph Data using Neighborhood Label Distribution

    [https://arxiv.org/abs/2610.22998](https://arxiv.org/abs/2610.22998)

    本文提出LENS框架，通过测量移除节点后邻域熵的变化来识别图数据中的噪声标签节点，并结合重标记机制，提升了图卷积网络在标签噪声下的鲁棒性。

    

    图卷积网络（GCNs）在图结构数据上展现了卓越的性能。然而，其成功在很大程度上依赖于干净标签的可获得性，而这一假设在现实场景中往往不成立。GCNs的可学习性与节点的邻域结构有着内在联系，因此理解该结构在标签噪声存在下如何变化至关重要。在这项工作中，我们研究了节点邻域标签分布在标签噪声影响下如何变化。我们从数学上推导出：标签噪声会平滑这种分布，从而固有地增加其熵，并严重降低类间可区分性。为了应对这一挑战，我们提出了留一熵节点筛选（LENS），这是一个端到端的框架，通过测量移除特定节点后邻域熵的减少程度来识别噪声节点。LENS还进一步引入了重标记机制……

    arXiv:2610.22998v1 Announce Type: cross  Abstract: Graph Convolutional Networks (GCNs) have demonstrated exceptional performance on graph-structured data. However, their success heavily relies on the availability of clean labels, an assumption that often fails in real-world scenarios. The learnability of GCNs is intrinsically linked to the neighborhood structure of nodes, making it critical to understand how this structure changes in the presence of label noise. In this work, we investigate how the neighborhood label distribution of a node changes under label noise. We mathematically derive that label noise smooths this distribution, which inherently increases its entropy and severely decreases interclass distinguishability. To address this challenge, we propose Leave-one-out Entropy Node Screening (LENS), an end-to-end framework that identifies noisy nodes by measuring the reduction in neighborhood entropy when a specific node is removed. LENS further incorporates a relabeling mechani
    
[^125]: TALLOP：面向时间受限联邦学习的本地训练与通信压缩联合优化

    TALLOP: Joint Optimization of Local Training and Communication Compression for Time-Constrained Federated Learning

    [https://arxiv.org/abs/2610.22996](https://arxiv.org/abs/2610.22996)

    提出TALLOP方法，在时间受限的联邦学习中联合优化本地训练工作负载与通信压缩，以缓解异构环境下的掉队者效应并减少达到目标精度所需的实际训练时间。

    

    在联邦学习（FL）中，最小化达到目标精度所需的实际运行时间至关重要，因为许多应用需要在数据分布随时间演变的情况下频繁更新模型。然而，在异构环境中，计算和通信能力受限的慢速客户端可能会延迟全局同步，迫使较快的客户端等待，从而增加整体训练时间。为了缓解这种掉队者效应，近期研究引入了每轮时间约束，并在可用时间内调整客户端的工作负载。然而，这些方法主要关注本地计算，而在通信受限的环境中，通信可能成为主要瓶颈，这促使人们对训练工作负载与通信压缩进行联合优化。尽管先前的研究已经探索了此类联合优化，但在固定的（原文摘要在此处被截断）……

    arXiv:2610.22996v1 Announce Type: new  Abstract: Minimizing the wall-clock time required to reach a target accuracy is crucial in federated learning (FL), where many applications require frequent model updates as data distributions evolve over time. Nevertheless, in heterogeneous environments, slow clients with limited computational and communication capabilities can delay global synchronization, forcing faster clients to wait and increasing the overall training time. To mitigate this straggler effect, recent studies have introduced per-round time constraints and adapted client workloads within the available time. However, these approaches primarily focus on local computation, whereas communication can become a major bottleneck in communication-limited environments, motivating the joint optimization of training workload and communication compression. Although previous studies have explored such joint optimization, determining an effective training--communication balance under a fixed p
    
[^126]: PersonaCUABench：面向多样化用户角色的计算机使用智能体基准测试

    PersonaCUABench: Benchmarking Computer-Use Agents Serving Diverse Personas

    [https://arxiv.org/abs/2610.22990](https://arxiv.org/abs/2610.22990)

    该论文提出 PersonaCUABench 基准，通过让 44 种不同用户角色用自己的措辞重写固定的任务指令，仅改变“谁在提问”就使九个骨干模型的平均任务完成率下降 36%，揭示了现有计算机使用智能体难以真正服务不同用户的个性化需求。

    

    计算机使用智能体（CUA）通常根据固定的任务规范进行评估，但用户可能会遗漏需求、添加约束条件，或要求在执行有重大影响的操作前先获得批准。完成基准测试指令并不保证能服务好提出请求的那个人。我们提出了 PersonaCUABench，该基准将跨 11 个领域的 74 个任务保持固定，仅改变“谁在提问”。一个角色智能体扮演四组共 44 个用户角色之一，用该用户自己的措辞重写每条指令并回答 CUA 的提问，而 CUA 只能看到对话内容，永远看不到用户画像或原始指令。在九个骨干模型上的实验（共 3,330 次运行）显示，相对于原始请求，平均任务完成率下降了 36%。这些损失有两个来源：在丢失分数的案例中，59.9% 的情况下智能体确实响应了用户的要求但仍然做错了；11.2% 是用户主动要求的停止操作，却被评分器判定为失败。即使当某个角色请求……（摘要在此处截断）

    arXiv:2610.22990v1 Announce Type: cross  Abstract: Computer-use agents (CUAs) are commonly evaluated against fixed task specifications, but users may omit requirements, add constraints, or require approval before consequential actions. Completing the benchmark instruction does not guarantee serving the person who asked. We introduce PersonaCUABench, which holds 74 tasks across 11 domains fixed and varies only who asks. A persona agent, playing one of 44 personas in four groups, rewrites each instruction in that user's own words and answers the CUA's questions, while the CUA sees only the dialogue, never the profile or the original instruction. Across nine backbones (3,330 runs), mean task completion falls by 36% relative to the original requests. The losses have two sources: in 59.9% of cells that lose points, the agent engages with what the user asked for and still gets it wrong, and 11.2% are stops the user requested that the checker scores as failures. Even when a persona request is
    
[^127]: 视线要有多浑浊，视觉语言模型才知道自己看不见？——基于物理可视性轴的准确率、置信度与弃权研究

    How Murky Before a VLM Knows It Cannot See? Accuracy, Confidence and Abstention on a Physical Visibility Axis

    [https://arxiv.org/abs/2610.22981](https://arxiv.org/abs/2610.22981)

    该研究首次在以米为单位的物理可视性轴上（基于Koschmieder雾模型和水下吸收光谱）系统测量视觉语言模型的准确率、置信度与弃权行为，发现当能见度降至十几米时模型准确率已骤降至随机水平，而其置信度却始终保持高位，揭示了VLM在“看不清”时缺乏自知之明的过度自信问题。

    

    视觉语言模型（VLM）的鲁棒性研究通常采用不携带物理单位的标称严重程度等级（“1-5级”）来衡量性能退化，并且很少追问模型的置信度是否随可见性的丧失而同步变化。我们在以米为单位的物理可视性轴上测量准确率、内部（token概率）置信度、口头表达置信度以及显式弃权选项：雾遵循Koschmieder模型建模，水下场景则按纯水吸收光谱进行缩放，两者均借助单目度量深度图施加，可视性范围V在250米至1.95米之间的八个对数间隔级别上扫描。在1000个COCO场景和五个小型开源VLM（2-5B参数）上，框选物体识别任务的准确率在很窄的区间内从接近干净图像的水平骤降至随机水平（雾中的半途可视性V_acc为12.1-15.9米，水下为20.5-38.6米）；然而，对于在干净图像上表现出较高置信度的四个模型，其置信度始终维持在高位：

    arXiv:2610.22981v1 Announce Type: cross  Abstract: Robustness studies of vision-language models (VLMs) grade degradation on nominal severity scales ("level 1-5") that carry no physical units, and they rarely ask whether a model's confidence follows the loss of visibility. We measure accuracy, internal (token-probability) confidence, verbalized confidence and an explicit abstain option on a visibility axis in metres: fog follows the Koschmieder model, an underwater arm scales the absorption spectrum of pure water, both are applied with a monocular metric depth map, and the visibility range V is swept over eight log-spaced levels from 250 m to 1.95 m. On 1000 COCO scenes and five small open VLMs (2-5 B parameters), accuracy on a boxed-object identification task falls from near its clean value to chance within a narrow band (half-way visibility V_acc of 12.1-15.9 m in fog and 20.5-38.6 m under water) while, for the four models that are confident on clean images, the confidence stays high:
    
[^128]: 迈向差分隐私优化的超参数迁移

    Towards Hyperparameter Transfer for Differentially Private Optimization

    [https://arxiv.org/abs/2610.22976](https://arxiv.org/abs/2610.22976)

    本文提出DP-μP框架，将最大更新参数化（μP）扩展到差分隐私设置，通过信噪比条件与谱缩放公式为DP-SGD、DP-Adam和DP-Muon优化器提供了逐层学习率缩放规则，从而实现差分隐私训练中的超参数迁移。

    

    缩放定律与超参数调优仍然是训练大规模模型的核心挑战，尤其在差分隐私（DP）条件下，由于噪声注入和逐样本梯度裁剪，优化问题变得更加复杂。虽然最大更新参数化（μP）能够在标准训练中实现零样本超参数迁移，但我们证明了在DP约束下它会失效，原因在于高维噪声与低秩梯度具有截然不同的谱特性。在本工作中，我们提出了DP-μP，这是一个具有理论依据的框架，将μP扩展到了隐私保护场景。通过分析梯度裁剪、噪声注入与特征学习之间的相互作用，我们推导出了一个控制逐层学习率缩放的信噪比（SNR）条件。结合我们提出的谱缩放公式，我们为DP-SGD、DP-Adam和DP-Muon优化器给出了具体的缩放规则。

    arXiv:2610.22976v1 Announce Type: new  Abstract: Scaling laws and hyperparameter tuning remain central challenges in training large-scale models, particularly under differential privacy (DP), where optimization is further complicated by noise injection and per-sample gradient clipping. While Maximal Update Parametrization ($\mu$P) allows for zero-shot hyperparameter transfer in standard training, we show that it fails under DP constraints due to the distinct spectral properties of high-dimensional noise versus low-rank gradients. In this work, we introduce DP-$\mu$P, a theoretically grounded framework that extends $\mu$P to the privacy-preserving setting. By analyzing the interaction between gradient clipping, noise injection, and feature learning, we derive a Signal-to-Noise Ratio (SNR) condition that governs layer-wise learning rate scaling. Combining this with our proposed spectral scaling formulation, we provide specific scaling rules for DP-SGD, DP-Adam, and DP-Muon optimizers. Em
    
[^129]: 什么塑造了大型语言模型在对话过程中与大脑的对齐？来自语言产生与理解的洞察

    What Shapes Brain Alignment of Large Language Models during Dialogue? Insights from Language Production and Comprehension

    [https://arxiv.org/abs/2610.22961](https://arxiv.org/abs/2610.22961)

    本研究利用对话中的fMRI数据揭示了大型语言模型在语言产生与语言理解两种模式下呈现截然不同的大脑对齐模式，且对齐程度与模型表征的内在维度以及可解释的语言和交际特征密切相关。

    

    大型语言模型（LLM）与大脑的对齐为研究大型语言模型与人类大脑在语言处理过程中的表征相似性提供了一种途径。以往的研究主要集中在被动的语言理解上，而对于对话过程中这种对齐的了解却很少，而在对话中，说话者会同时进行语言产生和语言理解。在本研究中，我们利用对话过程中收集的fMRI数据，考察了大型语言模型各层和各脑区在语言产生与理解过程中的大脑对齐情况。我们评估了这种对齐与大型语言模型表征内在维度的关联，以及若干可解释的语言学和交际特征的贡献。我们的结果揭示了语言产生与理解之间截然不同的大脑对齐模式，以及与每种处理模式相关脑区中内在维度的强关联。此外，我们还发现语言学特征……

    arXiv:2610.22961v1 Announce Type: new  Abstract: Brain alignment of large language models (LLMs) offers a way to investigate representational similarities between LLMs and the human brain during language processing. While previous studies have mainly focused on passive language comprehension, little is known about such alignment during dialogue, in which speakers engage in both language production and comprehension. In this study, using fMRI data collected during dialogue, we examine brain alignment during language production and comprehension across LLM layers and brain regions. We assess its association with the intrinsic dimension (ID) of LLM representations and the contributions of several interpretable linguistic and communicative features. Our results reveal distinct patterns of brain alignment between production and comprehension, as well as strong associations with ID in brain regions associated with each processing mode. Furthermore, we show that the contributions of linguisti
    
[^130]: 论对抗训练中鲁棒性与几何精度之间的权衡

    On the tradeoff between robustness and geometric accuracy in adversarial training

    [https://arxiv.org/abs/2610.22955](https://arxiv.org/abs/2610.22955)

    该论文证明了对抗分类器的几何精度损失与对抗预算ε之间的最优定量权衡关系——鲁棒分类器至多牺牲O(√ε)或O(ε)的精度，从而表明在高复杂度假设类中，鲁棒性与精度可以以精确量化的方式共存。

    

    在存在对抗者的非参数二分类问题中，我们为对抗分类器证明了一个最优的定量估计，该估计以到贝叶斯分类器的Hausdorff距离所衡量的几何精度为控制目标，并以对抗预算ε所编码的鲁棒性为依据。这严格地表明，鲁棒分类器在精度上的牺牲至多为O(√ε)或O(ε)，具体取决于底层贝叶斯分类器的正则性。与先前关于低复杂度假设类上鲁棒性与精度不相容性的工作不同，我们的工作表明，高复杂度假设类允许存在精度被精确量化的鲁棒分类器。我们的结果适用于由欧几里得空间上任意范数所确定的对抗者，并且在无限数据和有限数据两种设置中均成立。

    arXiv:2610.22955v1 Announce Type: new  Abstract: For non-parametric binary classification in the presence of an adversary, we prove an optimal quantitative estimate for adversarial classifiers controlling geometric accuracy, as measured by the Hausdorff distance to the Bayes classifier, in terms or their robustness, as encoded by the adversarial budget $\varepsilon$. This rigorously shows that robust classifiers sacrifice at most $O(\sqrt{\varepsilon})$ or $O(\varepsilon)$ in accuracy, depending on the regularity of the underlying Bayes classifier. In contrast to prior work on the incompatibility of robustness and accuracy over low-complexity hypothesis classes, our work shows that high-complexity hypothesis classes allow for robust classifiers with precisely quantified accuracy. Our results apply for adversaries determined by any norm on Euclidean space and in both the infinite and finite data settings.
    
[^131]: 大型扩散模型需要的不仅仅是EMA

    Large Diffusion Models Need More Than EMA

    [https://arxiv.org/abs/2610.22945](https://arxiv.org/abs/2610.22945)

    该研究从理论和实验上证明EMA并非扩散模型训练的最优方案：恒定步长轨迹平均存在固有优化局限，而学习率衰减与EMA互补可提升性能，Schedule-Free优化则无需依赖衰减时机即可在各个训练阶段持续超越EMA。

    

    大型扩散模型通常采用恒定学习率（LR）和指数移动平均（EMA）进行训练，尽管学习率衰减和Schedule-Free优化在大型语言模型的训练中已展现出强大的性能。我们研究了EMA是否特别适合扩散模型训练，还是存在被忽视的更强替代方案。我们的理论表明，对恒定步长轨迹进行平均存在固有的优化局限性，而学习率衰减和Schedule-Free优化在标准非凸假设下能够获得趋于消失的平稳性保证。实验证明，学习率衰减和EMA提供了互补的优势：在大多数测试的起始点，将衰减与EMA结合可以进一步提升标准基线的性能，尽管这一增益对衰减开始的时间点较为敏感。Schedule-Free则避免了这一时机要求，并在各个训练阶段持续优于恒定学习率下的EMA，在固定评估计算量和最终模型质量方面均实现了更优的表现。

    arXiv:2610.22945v1 Announce Type: new  Abstract: Large-scale diffusion models are commonly trained with a constant learning rate (LR) and exponential moving average (EMA), despite the strong performance of LR decay and Schedule-Free optimization in training large language models. We study whether EMA is uniquely suited to diffusion training, or whether stronger alternatives have been underexplored. Our theory shows that averaging a constant-step trajectory has an inherent optimization limitation, whereas LR decay and Schedule-Free optimization admit vanishing stationarity guarantees under standard nonconvex assumptions. Empirically, LR decay and EMA provide complementary benefits: combining decay with EMA further improves the standard baseline at most tested starting points, although the gain remains sensitive to when decay begins. Schedule-Free avoids this timing requirement and consistently outperforms EMA with a constant LR across training stages, achieving both lower fixed-evaluati
    
[^132]: 将语言模型对齐至反事实不变性

    Aligning Language Models Towards Counterfactual Invariance

    [https://arxiv.org/abs/2610.22942](https://arxiv.org/abs/2610.22942)

    本文提出一种基于群体级奖励函数的策略梯度技术，使语言模型在对齐过程中实现反事实不变性，解决标准强化学习对齐方法在提升整体回复质量的同时放大跨人口群体反事实偏差的问题。

    

    提示特征的微小变化可能会显著改变大语言模型输出的语义、语气和上下文假设。这些特征可能涉及人口统计学等群体变量或其他敏感的潜在属性。当人口统计信息在因果上与任务无关时，模型对此类输入变量的敏感性是不可取的，但这些信息可能仍与生成语义连贯的回复相关。这使得在保持模型效用的同时实现反事实不变性变得困难。为了研究这一问题，我们首先证明了标准模型对齐方法（如强化学习）在提升整体回复质量的同时，会加大不同人口群体之间回复质量的差异，最终放大了所谓的反事实偏差。为解决这一问题，我们开发了一种策略梯度技术，其中包含一个作用于匹配对的群体级奖励函数。

    arXiv:2610.22942v1 Announce Type: cross  Abstract: Small variations in prompt characteristics can meaningfully alter the semantics, tone, and contextual assumptions of large language model outputs. Such characteristics can involve group variables such as demographics or other sensitive latent attributes. Sensitivity to such input variables is undesirable when demographic information is causally irrelevant to the task, but might still be relevant for generation of semantically coherent responses. This makes the task of enforcing counterfactual invariance while preserving model utility difficult. To study this problem, we first show that standard model alignment recipes (such as reinforcement learning) increase differences in response quality across demographics, while improving overall response quality, ultimately amplifying what is known as counterfactual bias. To address this, we develop a policy gradient technique involving a group-level reward function that operates on matched pairs
    
[^133]: 多种偏见，同一回路：后训练微调预训练的线索路由机制

    Many Biases, One Circuit: Post-Training Tunes a Pretrained Cue-Routing Mechanism

    [https://arxiv.org/abs/2610.22937](https://arxiv.org/abs/2610.22937)

    多种偏见（如谄媚性）共享预训练模型中已存在的同一注意力头回路——助手微调在聊天格式中激活它，强化学习进一步加强它，表明跟随线索源于预训练与调优的副作用，而非对齐本身的问题。

    

    当提示中包含一个线索（即指向错误选项的暗示，例如“我认为答案是B”）时，语言模型往往会放弃正确答案。谄媚性及相关偏见都表现出这种失败，但它们通常被分开研究，并被归咎于对齐问题。我们通过因果方法检验了这一观点，在两个后训练流水线的各个阶段追踪这些偏见背后的注意力头。在对大多数模型的多项选择题测试中，四种线索类型在很大程度上依赖相同的少数注意力头——这是一个在预训练模型中就已存在的共享回路。大多数基础模型在纯文本中就会跟随线索。仅用约200条不含线索的通用助手示例对基础模型进行微调，就能在聊天格式中激活这些注意力头，因此跟随线索可能是助手调优的副作用。最终的强化学习（RL）阶段在几乎所有对比中都加强了线索的影响。在Tulu-3-8B和OLMo-2-7B中，我们交换了六个中间层在RL检查点之间的注意力模式……

    arXiv:2610.22937v1 Announce Type: cross  Abstract: Language models often abandon a correct answer when the prompt contains a cue, a hint toward a wrong option such as "I think the answer is B." Sycophancy and related biases show this failure. They are often studied separately and blamed on alignment. We test this view causally, tracing the attention heads behind these biases through the stages of two post-training pipelines. In multiple-choice tests on most models, four cue types rely largely on the same few heads, a shared circuit already present in pretrained models. Most base models follow cues in plain text. Fine-tuning a base model on about 200 generic assistant examples without cues activates these heads in chat format, so cue-following can be a side effect of assistant tuning. The final reinforcement learning (RL) stage strengthens the effect of cues in almost every comparison. In Tulu-3-8B and OLMo-2-7B, we swap the attention patterns of six middle layers between the RL checkpo
    
[^134]: 共形策略选择：何时部署AI决策以进行针对性干预

    Conformal Policy Selection: When to Deploy AI Decisions for Targeted Intervention

    [https://arxiv.org/abs/2610.22930](https://arxiv.org/abs/2610.22930)

    该论文提出“共形策略选择”方法，将AI策略的针对性部署建模为多重检验问题，通过加权共形p值与两步选择-剪枝程序在控制错误发现率的前提下识别出能从AI策略中受益的个体，实现了有限样本FDR控制和双重稳健的渐近保证。

    

    人工智能（AI）策略正日益被用于在医学、金融、教育和公共服务等领域做出影响重大的决策，然而要负责任地部署这些策略，不仅需要知道一项策略在平均意义上是否有帮助，还需要知道它对谁有帮助。现有的离线策略评估和安全策略学习工具提供的是总体层面的保证，但某些个体仍可能遭受不利的处理效应。我们将针对性部署重新构建为一个多重检验问题：对于每个候选单元，检验其在AI策略下的结果是否会超过用户指定的阈值，并在被选中的集合中控制错误发现率（FDR）。我们的方法“共形策略选择”将能够考虑从行为策略到目标策略分布偏移的加权共形p值，与两步选择-剪枝程序相结合。我们证明了有限样本下的FDR控制，并建立了双重稳健的渐近保证。

    arXiv:2610.22930v1 Announce Type: cross  Abstract: Artificial-intelligence (AI) policies are increasingly used to make consequential decisions in medicine, finance, education, and public services, yet deploying them responsibly requires knowing not only whether a policy helps on average, but for whom. Existing tools for off-policy evaluation and safe policy learning provide population-level guarantees, but some individuals may experience adverse treatment effects. We recast targeted deployment as a multiple-testing problem: for each candidate unit, test whether the outcome under the AI policy would exceed a user-specified threshold, and control the false discovery rate (FDR) among the selected set. Our method, Conformal Policy Selection, combines weighted conformal $p$-values that account for the distributional shift from behavior to target policy with a two-step selection-and-pruning procedure. We prove finite-sample FDR control and establish a doubly robust asymptotic guarantee, wher
    
[^135]: 面向深度学习的可扩展Levenberg-Marquardt训练框架

    Scalable Levenberg-Marquardt Training Framework for Deep Learning

    [https://arxiv.org/abs/2610.22925](https://arxiv.org/abs/2610.22925)

    提出了一种可扩展的Levenberg-Marquardt训练框架，克服了雅可比矩阵带来的内存和计算瓶颈，使LMA能够高效地训练通用大规模深度神经网络。

    

    Levenberg-Marquardt（LMA）算法是一种广泛使用的非线性最小二乘求解器，以其快速收敛性而闻名。对于小规模神经网络，LMA相比Adam等一阶优化器具有更好的收敛性和架构鲁棒性。然而，LMA带来了令人望而却步的内存和计算开销。这些开销主要来自雅可比矩阵，其维度会随样本数量和参数数量的增加而增长，这一瓶颈阻碍了其在大规模神经网络中的应用。近期的研究尝试使用预条件共轭梯度（PCG）求解器来扩展LMA，以避免显式构建雅可比矩阵。然而，PCG只能近似求解，需要大量的网络前向传播，且需要针对特定任务的预条件子。其他研究通过优化LMA来降低开销，但其方法仅限于特定任务或网络架构。因此，LMA在训练通用大规模神经网络方面仍然不切实际。

    arXiv:2610.22925v1 Announce Type: new  Abstract: The Levenberg-Marquardt (LMA) algorithm is a widely used nonlinear least-squares solver known for its rapid convergence. For small-scale neural networks, LMA provides better convergence and architectural robustness than first-order optimizers like Adam. However, LMA imposes prohibitive memory and computational costs. These costs are primarily from the Jacobian matrix, whose dimensions scale with both the number of samples and parameters. This bottleneck prevents its use in large-scale neural networks. Recent efforts scale LMA using preconditioned conjugate gradient (PCG) solvers to avoid materializing the Jacobian. However, PCG only approximates the solution, requiring many network propagations and a task-specific preconditioner. Other studies optimize LMA to reduce costs, but they restrict their methods to specific tasks or network architectures. As a result, LMA remains impractical for training general large-scale neural networks. In t
    
[^136]: 间接证据在哪里？使用影响函数研究语法泛化

    Where's the Indirect Evidence? Investigating Grammatical Generalization with Influence Functions

    [https://arxiv.org/abs/2610.22919](https://arxiv.org/abs/2610.22919)

    本文提出一种基于影响函数的自底向上方法来识别语言模型学习语法现象所依赖的间接证据，并发现移除这些间接证据会在五个模型的15种BLiMP语法现象上显著降低模型的泛化准确率。

    

    语言模型能够泛化到训练语料库之外以学习语言现象（Patil et al., 2024; Misra and Mahowald, 2024），这表明存在驱动学习的间接证据。先前的工作仅推断间接证据存在而未识别其特征，或只考虑了关于它可能是什么的特定假设。我们探索了一种替代性的自底向上方法来识别间接证据。我们使用影响函数（Koh and Liang, 2017）来衡量BLiMP（Warstadt et al., 2020）中15种语法现象的单个训练句子如何影响模型预测。我们使用手工构建的规则过滤掉某一现象的直接证据，并将剩余影响力最大的60,000个句子作为间接证据。这种间接证据对学习有实质性影响：在五个模型上，针对每一种语法现象，移除间接证据后重新训练比移除规模匹配的随机句子组更能降低留出集准确率（中位数降幅达……

    arXiv:2610.22919v1 Announce Type: new  Abstract: Language models generalize beyond their training corpus to learn linguistic phenomena (Patil et al., 2024; Misra and Mahowald, 2024), showing that there exists indirect evidence driving learning. Prior work has inferred that indirect evidence exists without identifying its character or considered specific hypotheses about what it might be. We explore an alternative bottom-up method for identifying indirect evidence. We use influence functions (Koh and Liang, 2017) to measure how individual training sentences affect model predictions in 15 grammatical phenomena from BLiMP (Warstadt et al., 2020). We filter out a phenomenon's direct evidence using hand-constructed rules and take the 60,000 most influential remaining sentences as indirect evidence. This indirect evidence substantially affects learning: retraining without it lowers held-out accuracy more than removing a matched random group for every phenomenon across five models (by a media
    
[^137]: FederatedBytes：无分词器、字节级小型语言模型，用于高效的多模态联邦学习

    FederatedBytes: Tokenizer-Free, Byte-Level Small Language Models for Efficient Multimodal Federated Learning

    [https://arxiv.org/abs/2610.22906](https://arxiv.org/abs/2610.22906)

    提出FederatedBytes——一种无分词器、将所有输入视为字节序列并采用线性注意力机制的小型语言模型，显著降低了NLP与多模态分析任务在联邦学习中的计算和通信开销。

    

    联邦学习（FL）是一种强大的机器学习框架，能够基于多个分布式客户端的贡献构建全局模型，同时保护其本地数据的隐私。然而，其高昂的计算与通信开销限制了它在自然语言处理（NLP）和多模态分析等当代AI工作负载中的应用。为此，本文提出了一种新型小型语言模型，该模型利用字节级的文本和多模态数据处理，并通过线性注意力机制加以增强，从而使NLP和多模态分析任务的联邦学习变得可行。该模型名为FederatedBytes，它将输入视为纯粹的字节序列，因而能够处理文本以及包括图像在内的任何其他数据类型；同时，其精简的线性注意力机制降低了整体的内存与计算开销。为验证所提模型的有效性……

    arXiv:2610.22906v1 Announce Type: new  Abstract: Federated Learning (FL) is a powerful machine learning framework that allows building a global model from the contributions of multiple, distributed clients, while ensuring the privacy of their local data. However, its heavy computational and communication overheads challenge its use for contemporary AI workloads such as natural language processing (NLP) and multimodal analysis. For this reason, this paper presents a novel small language model that leverages byte-level text and multimodal data processing, enhanced by a linear attention mechanism, to enable manageable Federated Learning of NLP and multimodal analysis tasks. The proposed model, nicknamed FederatedBytes, manages to process text and any other data types, including images, by treating its input as a mere sequence of bytes, while its lean linear attention mechanism allows abating the overall memory and computational footprint. To prove the effectiveness of the proposed model w
    
[^138]: 一种用于神经网络多分类的特征选择层方法

    A Feature Selection Layer Approach for Multi-Class Classification on Neural Networks

    [https://arxiv.org/abs/2610.22898](https://arxiv.org/abs/2610.22898)

    提出了对特征选择层的多种扩展，使嵌入式特征加权机制能够在多分类任务中捕捉特定标签的特征关系，从而提升神经网络的可解释性。

    

    在数据驱动决策的时代，深度学习模型已成为揭示模式和进行预测的强大工具，影响着社会的许多方面。然而，它们在科学和商业应用中的更广泛采用往往受到可解释性不足的限制。特征加权方法通过评估单个特征的相关性，是提高可解释性方面被研究最多的方法之一。虽然事后方法已经取得了显著进展，特别是在建模多分类问题中特征与标签之间的细微关系方面，但嵌入式方法在这方面通常表现不足。嵌入式技术将特征相关性学习直接集成到模型训练过程中，但它们往往无法捕捉特定标签的关系。为了解决这一局限性，我们提出了对特征选择层的多种扩展，这是一种嵌入式特征加权机制。

    arXiv:2610.22898v1 Announce Type: new  Abstract: In the age of data-driven decision-making, deep learning models have become powerful tools for uncovering patterns and making predictions that influence many aspects of society. However, their broader adoption in scientific and business applications is often limited by a lack of interpretability. Feature weighting methods, which assess the relevance of individual features, are among the most widely studied approaches for improving interpretability. While post hoc techniques have advanced significantly, especially in modeling nuanced relationships between features and labels in multiclass problems, embedded methods typically fall short in this regard. Embedded techniques integrate feature relevance learning directly into the model training process, but they are often unable to capture label-specific relationships. To address the limitation, we propose multiple extensions to the Feature Selection Layer, an embedded feature weighting mechan
    
[^139]: 何时遗忘：面向长程智能体的风险受控上下文自适应

    When to Forget: Risk-Controlled Context Adaptation for Long-Horizon Agents

    [https://arxiv.org/abs/2610.22897](https://arxiv.org/abs/2610.22897)

    该论文提出RECAP框架，通过匹配上下文干预、执行关系预测和风险约束下的独立策略认证，研究长程智能体在何时应保留、压缩或删除失败历史记录，发现失败记录的价值随执行进程动态变化。

    

    长程智能体从失败中学习，但失败记录的价值会随着执行进程的推进而变化。我们研究在何种情况下，相对于保留完整历史的KEEP策略，对同一历史块进行保留、压缩或删除能够改善后续执行。我们提出了RECAP框架，该框架结合了匹配上下文干预、执行关系预测和独立策略认证。通过匹配的环境状态和rollout种子来衡量成功率与续写提示成本的差异；基于失败块与当前执行之间的仅前缀关系来提出有前景的编辑建议。随后，在固定的测量协议下，利用独立任务包在有害干预风险和非正收益风险约束下选择策略。一项五种子纵向案例研究保持失败块及其摘要不变，结果显示：压缩在早期状态失效，但在后期状态成为节省成本的干预手段。在五个骨干模型上，RECAP（摘要在此处截断）

    arXiv:2610.22897v1 Announce Type: new  Abstract: Long-horizon agents learn from failure, but the value of a failure record can change as execution progresses. We study when retaining, compressing, or removing the same history block improves subsequent execution relative to full-history KEEP. We introduce RECAP, a framework that combines matched context interventions, execution-relational prediction, and independent policy certification. Matched environment states and rollout seeds measure success and continuation prompt-cost differences; prefix-only relations between a failure block and the current execution propose promising edits. Independent task bundles then select a policy subject to harmful-intervention and non-positive-benefit risk constraints under a fixed measurement protocol. A five-seed longitudinal case holds both the failure block and its summary fixed: compression fails at an early state and becomes a cost-saving intervention at a later state. Across five backbones, RECAP
    
[^140]: 基于能力的大语言模型逻辑推理评估

    Competence-Based Evaluation of Logical Reasoning in Large Language Models

    [https://arxiv.org/abs/2610.22894](https://arxiv.org/abs/2610.22894)

    本文提出“基于能力的评估”（CBE）新范式，通过参数化生成器调节复杂度并使用保留答案的多种表述形式，来衡量大语言模型逻辑推理的泛化能力，而非仅依赖单一准确率指标。

    

    大语言模型（LLM）的最新进展使其在多种任务上展现出令人瞩目的性能；然而，在逻辑推理方面仍存在不足，包括在语言表述变化下的不一致性。为了提升大语言模型的推理能力，我们必须能够准确衡量其推理能力。当前的基准测试主要报告在选定推理任务上的单一准确率数字。这种基于性能的评估方法对于模型在推理方面的泛化能力说明甚少。为解决这一局限，我们主张采用一种不同的评估范式，称为基于能力的评估（CBE），它测试模型能否在一系列同族问题中可靠地运用相同的能力。参数化生成器在保持所需能力不变的情况下扩展问题复杂度，而匹配的、保留答案的多种表述方式则用于测试正确性是否超越单一表述形式。我们在基础的排序任务上实现了CBE。

    arXiv:2610.22894v1 Announce Type: cross  Abstract: Recent advances in large language models (LLMs) have brought impressive performance across diverse tasks; however, shortcomings remain in logical reasoning, including inconsistencies under linguistic variation. To improve the reasoning capacity of LLMs, we must be able to accurately measure their reasoning capacity. Current benchmarks primarily report single accuracy numbers on selected reasoning tasks. Such a performance-based evaluation methodology says little about a model's generalization capacity in reasoning. To address this limitation, we argue for a different evaluation paradigm, called competence-based evaluation (CBE), which tests whether models reliably apply the same competence across a family of problems. A parameterized generator scales complexity while keeping the required competence fixed, and matched answer-preserving presentations test correctness beyond one formulation. We instantiate CBE on elementary ordering tasks
    
[^141]: 已学习的Transformer程序中准确率掩盖了什么

    What Accuracy Hides In Learned Transformer Programs

    [https://arxiv.org/abs/2610.22891](https://arxiv.org/abs/2610.22891)

    该研究发现token准确率会掩盖Transformer程序中的错误完整答案，通过程序检查能够发现抽样测试遗漏的失败案例，并据此改进训练数据，从而在多个算法任务上提升序列准确率。

    

    Transformer程序（TP）（Friedman等人，2023）通过约束Transformer的架构，使其学习到的计算能够被提取为可读的Python代码。TP能够暴露模型学习到的计算过程，但token准确率可能会掩盖不正确的完整答案。在这项工作中，我们研究了程序检查如何提供超越准确率分数的洞察，并帮助指导训练。我们首先在九个算法任务上评估了TP，将token准确率与完整输出序列的准确率进行比较，并检查了被抽样测试所遗漏的失败案例。此外，我们评估了基于程序检查和失败分析所改进的训练数据，发现多个任务的序列准确率得到了提升。我们研究了一个计数任务，其中一个程序在抽样测试序列上达到了100%的准确率，但在一个使用训练期间允许的符号和长度的其他输入上却失败了。该程序的计数是正确的，但所需的答案在其输出标签中缺失……

    arXiv:2610.22891v1 Announce Type: new  Abstract: Transformer Programs (TP) (Friedman et al., 2023) constrain a Transformer's architecture so that its learned computation can be extracted as readable Python code. TP expose learned computations, but token accuracy can hide incorrect complete answers. In this work, we examine how program inspection provides insights beyond accuracy scores and helps guide training. We first evaluated TP on nine algorithmic tasks. We compared token accuracy with accuracy on complete output sequences and checked for failures missed by sampled tests. Further, we assessed training data improvements informed by program inspection and failure analysis, finding gains in sequence accuracy on several tasks. We examined a counting task where a program achieved 100% accuracy on sampled test sequences but failed on another input using the symbols and lengths allowed during training. The program counted correctly, but the required answer was missing from its output lab
    
[^142]: 批量上下文强化 II：RLVR 需要长采样窗口吗？

    Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?

    [https://arxiv.org/abs/2610.22887](https://arxiv.org/abs/2610.22887)

    提出并发约束强化（CCR），将多个可独立验证的问题放入同一个受限完成中，使RLVR在8,000-token的短窗口下仍能保留正确性信号，并将每训练步的token消耗从约97.5万降至约17.5万，大幅降低训练成本。

    

    长采样窗口使得可验证奖励强化学习（RLVR）的代价高昂，但简单缩短窗口可能会丢失正确性信号：当每个单问题的采样在得出答案之前就被截断时，组相对准确率优势将不复存在。作为批量上下文强化（Batched Contextual Reinforcement, BCR）的技术延续，我们研究了并发奖励如何放宽这种对长窗口的依赖。并发约束强化将三个可独立验证的问题置于同一个有长度上限的训练完成中。在截断前完成的正确答案仍可获得奖励，而共享预算在无需显式长度惩罚的情况下，鼓励更简短的有效推理。我们将该机制与条件奖励方差和更新方差分析联系起来，并在4B和8B数学推理模型上进行了研究。与使用32k窗口的单问题GRPO相比，使用8,000-token窗口的CCR在4B模型上将每训练步的token数从975,464降至174,957。

    arXiv:2610.22887v1 Announce Type: cross  Abstract: Long rollout windows make reinforcement learning with verifiable rewards (RLVR) expensive, yet simply shortening them can remove the correctness signal: when every single-problem rollout truncates before an answer, group-relative accuracy advantages vanish. As a technical continuation of Batched Contextual Reinforcement (BCR), we study how concurrent rewards relax this dependence on long windows. Concurrent Constraint Reinforcement (CCR) places three independently verifiable problems in one capped training completion. Correct answers completed before truncation retain credit, while a shared budget encourages shorter useful reasoning without an explicit length penalty. We connect this mechanism to conditional reward- and update-variance analyses and study 4B and 8B mathematical reasoning models. Against single-problem GRPO with a 32k window, CCR with an 8,000-token window reduces tokens per training step from 975,464 to 174,957 on 4B an
    
[^143]: 学会被引导：面向无分类器引导的潜在空间几何塑造

    Learning to Be Guided: Shaping Latent Geometry for Classifier-Free Guidance

    [https://arxiv.org/abs/2610.22885](https://arxiv.org/abs/2610.22885)

    本文提出原型正则化方法，通过在潜在空间中塑造“类间方差大于类内方差”的类别几何结构，使潜在空间天然适配无分类器引导（CFG），从而提升条件扩散模型的引导生成性能。

    

    无分类器引导是条件扩散模型的标准组成部分，引导生成性能已成为衡量最先进模型的事实标准。近期关于潜在扩散模型的研究致力于学习更容易用扩散过程建模的潜在空间。然而，在CFG下，生成效果还取决于条件在潜在空间中的排列方式。我们在离散类别条件下研究了这种排列方式，称之为“类别几何”，并探究什么样的类别几何能使潜在空间与CFG良好协同工作。在高斯类别模型下，我们证明在总方差固定的情况下，将方差从类内转移到类间可以使各类别在噪声下更易区分，并推导出一个噪声后验目标，其逐样本的下降方向与oracle CFG修正成正比。基于这一分析，我们提出了原型正则化，这是一种轻量级的替代方法（摘要在此处被截断）

    arXiv:2610.22885v1 Announce Type: cross  Abstract: Classifier-free guidance (CFG) is a standard component of conditional diffusion models, and guided generation performance has become the \textit{de facto} measure of state-of-the-art models. Recent work on latent diffusion models seeks to learn latent spaces that are easier to model with diffusion. However, under CFG, generation additionally depends on how conditions are arranged in the latent space. We study this arrangement under discrete class conditions, termed \emph{class geometry}, and investigate what class geometry allows a latent space to work well with CFG. Under a Gaussian class model, we show that shifting variance from intra- to inter-class at fixed total variance makes classes more distinguishable under noise, and derive a noisy posterior objective whose per-sample descent direction is proportional to the oracle CFG correction. Based on the analysis, we propose Prototype Regularization (PR), a lightweight surrogate for th
    
[^144]: ToPA：用于富有表现力且高效微调的分块Toeplitz自适应方法

    ToPA: Block-wise Toeplitz Adaptation for Expressive and Efficient Fine-Tuning

    [https://arxiv.org/abs/2610.22883](https://arxiv.org/abs/2610.22883)

    本文提出ToPA方法，利用沿对角线元素恒定的分块Toeplitz矩阵对预训练模型进行参数高效微调，在不施加低秩或稀疏性显式约束的情况下，以紧凑的参数化方式获得更接近全量微调的表现力与效率。

    

    参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）已成为适配大型预训练模型的一种主流策略。在各类PEFT技术中，基于重参数化的方法，尤其是采用低秩或稀疏结构的方法，受到了广泛关注。然而，这些方法的性能往往不及全量微调。具体而言，低秩方法限制了更新的秩，这与全量微调本身固有的高秩特性并不一致；而基于稀疏性的技术则限制了参数空间的灵活性，并破坏了权重之间的连通性。为了克服这些局限，我们探索使用Toeplitz矩阵，其沿每条对角线的元素均为常数，因此在无需施加显式秩约束或稀疏性约束的情况下，即可提供一种紧凑的参数化方式。一种直观的做法是通过Toeplitz矩阵的乘积来更新权重。然而，与RNN类似，长程乘性……（原文摘要在此处被截断）

    arXiv:2610.22883v1 Announce Type: new  Abstract: Parameter-Efficient Fine-Tuning (PEFT) has emerged as a prevalent strategy for adapting large pre-trained models. Among PEFT techniques, reparameterization-based methods, particularly those that employ low-rank or sparsity structures, have attracted significant attention. However, these approaches often underperform full fine-tuning. Specifically, low-rank methods constrain the update rank, which is misaligned with the inherently high-rank nature of full fine-tuning. In contrast, sparsity-based techniques limit the flexibility of the parameter space and compromise connectivity across weights. To overcome these limitations, we explore the use of Toeplitz matrices, whose entries are constant along each diagonal, therefore providing a compact parameterization without imposing explicit rank or sparsity constraints. A straightforward approach is to update weights via a product of Toeplitz matrices. However, similar to RNNs, long multiplicativ
    
[^145]: 基准测试与增强智能体式因果发现

    Benchmarking and Enhancing Agentic Causal Discovery

    [https://arxiv.org/abs/2610.22846](https://arxiv.org/abs/2610.22846)

    提出Heuresis基准，通过匿名模拟数据集评估LLM智能体的因果发现能力，区分出现有算法的收益上限与需要智能体深入检查数据才能弥补的差距，从而推动智能体式因果发现的增强。

    

    LLM智能体现在可以在数据集上运行因果发现算法并报告其结果，但它们是否做得更多——例如发现算法在何处出错并通过进一步的检验加以修正——从现有基准测试中很难判断。大多数基准测试为变量赋予了有意义的名称，而一个仅看到图宾根（Tübingen）因果对变量名称的LLM，就已经能在99%的情况下正确判断因果方向。在去除名称后，数据通常足够“干净”，以至于仅选择一个比PC更好的算法就能弥补大部分差距。我们提出了Heuresis，这是一个由匿名模拟数据集构成的基准测试，其数据生成机制涵盖从线性高斯分布到厚尾分布。对于每一族数据集，它将“更好的现有算法所能带来的提升”与“只有更仔细地审视数据才能带来的提升”区分开来，并追踪智能体所修改的每一条边。每一族数据集都留下了一个缺口，无论是算法、路由器还是我们测试的两个因果发现基础模型都无法填补（摘要在此处被截断）。

    arXiv:2610.22846v1 Announce Type: new  Abstract: LLM agents can now run causal discovery algorithms on a dataset and report what comes out. Whether they do more than that, such as noticing where an algorithm went wrong and fixing it with further tests, is hard to tell from current benchmarks. Most of them give the variables meaningful names, and an LLM shown only the names of the T\"ubingen cause-effect pairs already gets 99% of the directions right. With the names removed, the data are usually clean enough that picking a better algorithm than PC closes most of the gap. We introduce Heuresis, a benchmark of anonymous simulated datasets whose mechanisms range from linear Gaussian to heavy-tailed. For each family of datasets it separates what a better existing algorithm can gain from what only a closer look at the data can gain, and it traces every edge an agent changes. Every family leaves a gap that none of the algorithms, the router and the two causal-discovery foundation models we ra
    
[^146]: 部分揭示面板预测：预测信息增益、分配与粗化一致算子

    Partial-Revelation Panel Forecasting: Predictive Information Gain, Allocation, and Coarsening-Consistent Operators

    [https://arxiv.org/abs/2610.22831](https://arxiv.org/abs/2610.22831)

    该论文提出部分揭示面板预测框架，证明了嵌套揭示机制下预测信息增益的恒等式，并设计了具有闭式狄利克雷-多项式实现的粗化一致面板算子，在多个情感对话数据集上显著改善了冷启动计数预测。

    

    我们研究部分揭示面板预测问题：给定上一轮对话中 n 个标注者投票中的 k 个，预测下一轮面板的完整计数分布。这种跨轮预测信息不同于同条目的标注价值。在任何嵌套揭示机制下，总体减量满足 $V(k)-V(k+1)=I(Y;R_{k+1}\mid X_k)$；拟合模型也贡献相邻超额风险差异。完全随机粗化（CCAR）掩码为我们算子中使用的可交换随机评分者解释提供了基础。我们定义了粗化一致面板算子，并给出闭式狄利克雷-多项式实现、学习式补全以及评分者数量迁移。与逐级查找基线相比，冷启动计数负对数似然差异在 IEMOCAP/M3ED/EmoWOZ 上分别为 -0.422/-0.166/-0.014 nats。一项留出评分者实验衡量了同条目价值：第一张和第二张额外投票分别使 NLL 降低 0.583/0.221、0.5……

    arXiv:2610.22831v1 Announce Type: new  Abstract: We study partial-revelation panel forecasting: given $k$ of $n$ annotator votes on the previous dialogue turn, predict the full count distribution of the next turn's panel. This cross-turn predictive information differs from same-item annotation value. Under any nested revelation mechanism, the population decrement satisfies $V(k)-V(k+1)=I(Y;R_{k+1}\mid X_k)$; a fitted model also contributes adjacent excess-risk differences. Coarsened-completely-at-random (CCAR) masking supplies the exchangeable random-rater interpretation used in our operators. We define coarsening-consistent panel operators with a closed-form Dirichlet-multinomial realization, learned completion, and rater-count transfer. Against per-level lookup, cold-start count negative log-likelihood differences are -0.422/-0.166/-0.014 nats on IEMOCAP/M3ED/EmoWOZ. A held-out rater experiment measures same-item value: first and second additional votes reduce NLL by 0.583/0.221, 0.5
    
[^147]: 因果编辑序列化

    Causal Edit Serialization

    [https://arxiv.org/abs/2610.22820](https://arxiv.org/abs/2610.22820)

    提出因果编辑序列化（CES）框架，通过指针词元、轻量级定位头和双索引旋转位置编码，使自回归语言模型能够在保持标准因果注意力机制的前提下，对文本进行插入、删除、替换、复制和移动等显式编辑操作。

    

    标准自回归语言模型按顺序生成文本，但修订操作需要回溯引用输入上下文中的词元或已生成的词元。本文提出了因果编辑序列化（CES）框架，用于在同一个因果词元流中表达此类修订。特殊的指针词元与预测的源地址相配对，使得插入、删除、替换、复制和移动等操作能够被显式地定位。轻量级的定位头与双索引旋转位置编码既支持地址预测，也支持对先前发出的指针进行条件化处理，同时保留了标准的因果注意力掩码。预训练过程将普通语言建模与序列化修复任务相结合，这些修复任务来源于随机损坏、模型生成的编辑以及重复文本。在多种模型规模和编辑基线的对比中，CES模型能够完整地执行编辑操作，同时在干净文本上每字节的比特数仅略有增加。

    arXiv:2610.22820v1 Announce Type: new  Abstract: Standard autoregressive language models generate text sequentially, but revision requires referring back to tokens in the input context or those already generated. Causal Edit Serialization (CES) is introduced as a framework for expressing such revisions within the same causal token stream. Special pointer tokens are paired with predicted source addresses, allowing insertion, deletion, replacement, copying and movement to be localized explicitly. A lightweight localization head and dual-index rotary position encoding support both address prediction and conditioning on previously emitted pointers, while retaining a standard causal attention mask. A pretraining procedure combines ordinary language modeling with serialized repairs derived from random corruptions, model-generated edits and repeated text. Across model scales and editing baselines, CES models demonstrate complete applied edits with modestly higher clean-text bits per byte than
    
[^148]: 面向硬件高效限速标志识别的重放剪枝与少比特整数推理

    Replay Pruning and Few-Bit Integer Inference for Hardware-Efficient Speed-Sign Recognition

    [https://arxiv.org/abs/2610.22811](https://arxiv.org/abs/2610.22811)

    提出通道重放剪枝方法，通过在原始训练网络上重放记录的剪枝删除并结合知识蒸馏恢复，在保持父网络精度的同时移除YOLO网络60%以上的参数，实现汽车SoC上硬件高效的限速标志识别。

    

    在汽车摄像头片上系统上，限速标志识别必须将检测器和分类器（读取器）塞进设备的内存、算术和能耗预算之中。在我们的两阶段识别器中，检测器是一个采用ReLU和hard-sigmoid算子的YOLO系列网络，通过通道重放进行压缩：迭代剪枝级联识别出需要移除的通道，记录的删除操作在原始任务训练的父网络上重放，随后进行一次连续的知识蒸馏恢复。在YOLOv5n上，重放在保持父网络精度的同时移除了60.3%的参数，并且在五个匹配种子的验证mAP50-95上优于迭代剪枝和从头训练，其每步变体是精度最高的剪枝方法。在YOLOv8n上，重放和每步重放优于所有其他方法以及未剪枝的父网络。读取器是一个拥有18k参数的五层CNN

    arXiv:2610.22811v1 Announce Type: cross  Abstract: Speed-limit sign recognition on an automotive camera system-on-chip (SoC) must fit a detector and a classifier (the reader) into the memory, arithmetic, and energy budgets of the device. In our two-stage recognizer, the detector, a YOLO-family network with ReLU and hard-sigmoid operators, is compressed by channel replay: an iterative pruning cascade identifies the channels to remove, the recorded deletions are replayed on the original task-trained network, the parent, and one continuous knowledge-distillation recovery follows. On a YOLOv5n, replay removes 60.3% of the parameters at the accuracy of the parent and, in validation mAP50-95 over five matched seeds, outperforms iterative pruning and training from scratch, and its per-step variant (per-step replay) is the most accurate pruning method. On a YOLOv8n, replay and per-step replay outperform every other method and the unpruned parent. The reader, a five-layer CNN with 18k parameter
    
[^149]: 以语义ID作为唯一物品表示的推理重排序

    Reasoning Reranking with Semantic IDs as the Sole Item Representation

    [https://arxiv.org/abs/2610.22797](https://arxiv.org/abs/2610.22797)

    本文构建了ReRankID受控测试平台，首次系统评估仅以语义ID作为物品表示进行推理重排序的可行性，发现此类模型常常直接复制输入顺序而非真正重排序，且引入可读元数据也无法修复这一失败。

    

    推理重排序器在排定候选列表之前会先生成一段推理依据，而现有系统通过读取物品元数据来实现这一过程，尽管生成式推荐系统正日益采用紧凑的语义ID（SIDs）来表示物品。仅凭SID进行排序具有两点吸引力：其一，SID提示词远短于匹配的物品文本；其二，尽管物品编码由内容固定，但骨干模型会部分地从用户-物品交互序列中学习这些token的嵌入，因此这些token可以承载物品文本所缺乏的行为信号。然而，此类训练以“下一物品生成”为目标，其能否支持仅凭SID对给定的候选列表进行排序尚未得到验证。我们构建了ReRankID——一个受控测试平台，模型仅依据以SID形式呈现的历史与候选列表进行推理和排序，并每次仅改变一个变量：物品呈现方式、历史长度或候选列表大小。在该测试平台上，此类模型往往直接复制输入顺序而不进行重排序，且可读的元数据也无法修复这一失败。

    arXiv:2610.22797v1 Announce Type: cross  Abstract: Reasoning rerankers write a rationale before ranking a candidate slate, and existing systems read item metadata to do so, even though generative recommenders increasingly represent items as compact Semantic IDs (SIDs). Ranking from SIDs alone has two appeals: SID prompts are far shorter than the matched item text, and although item codes are fixed from content, the backbone learns their token embeddings in part from user-item interaction sequences, so the tokens can carry behavioral signal that item text lacks. That training targets next-item generation, and whether it supports ranking a supplied slate from SIDs alone is untested. We build ReRankID, a controlled testbed in which models reason and rank from a history and slate shown only as SIDs, varying item rendering, history length, and slate size one at a time. On this testbed, such models often copy the input order instead of reranking, and readable metadata does not repair the fai
    
[^150]: Ori：面向极低秩大语言模型训练的漂移感知流式k-PCA算法

    Ori: Drift-Aware Streaming $k$-PCA for Very Low-Rank LLM Training

    [https://arxiv.org/abs/2610.22794](https://arxiv.org/abs/2610.22794)

    提出Ori优化器，利用受Oja++启发的低开销流式k-PCA算法高频跟踪LLM训练中缓慢漂移的梯度子空间，在严格低秩约束下达到最先进的训练精度。

    

    由GaLore开创的一类著名的低秩大语言模型优化器采用梯度子空间投影方法：在每一步中，它们将梯度投影到低维子空间中，并将更新映射回完整模型。现有的大多数工作要么在特定时间间隔重新计算梯度子空间，要么通过事后适配来追赶移动的子空间。在本工作中，我们通过实践明确识别出LLM梯度子空间随时间以缓慢、稳定的方式漂移的特性。为此，我们提出了Ori，一种新颖的低秩LLM训练优化器，它能够捕捉随时间变化的主梯度方向，在严格的低秩约束下实现最先进的准确度。Ori采用了一种受Oja++启发的流式k-PCA算法，其每步计算成本较低，能够实现高频的实时子空间跟踪。在实践中，Ori缩小了最先进的低秩优化器与全秩方法之间的性能差距。

    arXiv:2610.22794v1 Announce Type: new  Abstract: A prominent family of low-rank LLM optimizers, pioneered by GaLore, performs gradient-subspace projection: at each step, they project the gradient into a low-dimensional subspace and map the update back to the full model. Most existing work either recomputes the gradient subspace at certain intervals, or performs post-hoc adaptation to catch up to the moving subspace. In this work, we explicitly identify in practice the slow, stable manner in which LLM gradient subspaces drift across time. To this end, we propose Ori, a novel low-rank optimizer for training LLMs that captures the shifting principal gradient directions over time to achieve state-of-the-art accuracy under stringent low-rank constraints. Ori employs a streaming $k$-PCA algorithm inspired by Oja++, whose low per-step cost enables high-frequency, on-the-fly subspace tracking. In practice, Ori reduces the performance gap of the state-of-the-art low-rank optimizers to full-rank
    
[^151]: 下一个词元还是向量？学习该解码多远

    Next Token or Vector? Learning How Far to Decode

    [https://arxiv.org/abs/2610.22788](https://arxiv.org/abs/2610.22788)

    提出CADENCE上下文自适应解码方法，利用高斯预测头将未来文本片段表示为连续潜在向量，并通过学习路由器根据上下文动态选择预测长度，相比固定长度预测获得更高的草稿接受率。

    

    标准的自回归解码每次只前进一个词元，即使多个即将出现的词元其实可以被一起预测。我们提出了CADENCE（上下文自适应解码，Context-Adaptive DEcoding），一种既学习草拟什么内容、又学习草拟多远距离的生成方法。高斯预测头将未来的文本片段表示为连续的潜在向量，共享解码器将选定的向量并行转换为词元预测。一个学习得到的路由器根据模型自身的隐藏状态和预测的高斯参数来选择预测长度，使生成粒度能够适应当前上下文，并以单词元生成作为回退方案。骨干模型会验证每个提议的延续内容，并保留与其贪婪预测相匹配的前缀。在Llama-3.1-8B-Instruct上针对Alpaca、GSM8K和HumanEval的实验表明，由路由器选择预测长度位置的方式，比在所有位置统一应用相应预测长度能获得更高的草稿接受率。

    arXiv:2610.22788v1 Announce Type: cross  Abstract: Standard autoregressive decoding advances one token at a time, even when several upcoming tokens can be predicted together. We propose CADENCE (Context-Adaptive DEcoding), a generation method that learns both what to draft and how far to draft. Gaussian prediction heads represent future spans as continuous latent vectors, and a shared decoder converts the selected vector into token predictions in parallel. A learned router selects the prediction length from the model's own hidden states and predicted Gaussian parameters, adapting generation granularity to the current context with single-token generation as a fallback. The backbone verifies each proposed continuation and retains the prefix that matches its greedy predictions. Experiments with Llama-3.1-8B-Instruct on Alpaca, GSM8K, and HumanEval show that router-selected positions yield higher draft acceptance than applying the corresponding prediction length everywhere. CADENCE achieve
    
[^152]: 面向AI服务的随时可退出合约

    Anytime Contracts for AI Services

    [https://arxiv.org/abs/2610.22781](https://arxiv.org/abs/2610.22781)

    该论文将契约理论扩展到用户可随时退出订阅的AI服务场景，证明在此设定下预算最优的合约与基于似然比e-过程的任意时刻有效假设检验成正比，从而防止模型提供方或用户对合约的滥用。

    

    AI服务容易受到道德风险的困扰：模型提供方可能悄悄地将请求路由到更便宜、质量更低的模型，而模型输出固有的随机性使得这种质量下降难以被检测。契约理论通过设计奖励高质量输出并防止策略性超额收费的支付规则来应对道德风险。然而，现有框架假设交互具有固定期限，而AI服务通常通过订阅制和按量付费的API来消费，用户可以根据观察到的性能随时终止服务。我们证明了这种灵活性会破坏标准合约，使其容易被提供方或用户滥用。为解决这一问题，我们将契约理论扩展到考虑未知退出规则的情形。我们将该问题形式化为委托-代理博弈，并证明了该设定下预算最优的合约与基于似然比e-过程（原文在此处截断）的任意时刻有效假设检验成正比。

    arXiv:2610.22781v1 Announce Type: cross  Abstract: AI services are prone to moral hazard, as model providers can silently route requests to cheaper, lower-quality models, and the inherent stochasticity of model outputs makes this degradation hard to detect. Contract theory addresses moral hazard by designing payment rules that reward high-quality outputs and prevent strategic overcharging. However, existing frameworks assume interactions of fixed duration, whereas AI services are typically consumed through subscriptions and pay-as-you-go APIs that users can terminate at any time based on observed performance. We show that this flexibility can break standard contracts, opening them up to abuse by either the providers or users. To address this, we extend contract theory to account for unknown exit rules. We formalize the problem as a Principal-Agent game, and show that budget-optimal contracts in this setting are proportional to anytime-valid hypothesis tests based on likelihood-ratio e-
    
[^153]: 跨模型表示复用校准增益

    Reusing Calibration Gains across Model Representations

    [https://arxiv.org/abs/2610.22771](https://arxiv.org/abs/2610.22771)

    该论文提出一种无需接收方优化的迁移方法，可将一种模型表示上学到的尺度校准增益复用到同一预训练模型的其他表示（如量化模型）中，在保持量化存储布局不变的同时保留76.9%–90.5%的校准收益。

    

    尺度校准会为一种模型表示学习一个更新量。我们研究了同一预训练模型的另一种表示能在多大程度上复用该更新。我们的迁移方法提取正的分组增益，并将其直接安装到接收方的权重或原生量化尺度中，而无需在接收方一侧进行任何优化。对于量化的接收方，该方法保持整数编码、零点和存储布局不变。对于一个Qwen3-1.7B浮点/GPTQ INT3模型对，在未参与项目开发的全新WikiText和C4数据集上，双向迁移保留了接收方专属校准所带来的文本损失改进的76.9%至90.5%。在每个投影内打乱相同的增益值会使两个接收方的性能都变差，这表明其学习到的分组分配在迁移之后仍然重要。一项Llama-3.2-3B对照实验保持所有四个校准投影处的初始原生张量完全相同，而改变其他位置的九个投影。交叉安装……

    arXiv:2610.22771v1 Announce Type: new  Abstract: Scale calibration learns an update for one model representation. We study how much of that update can be reused by another representation of the same pretrained model. Our transfer procedure extracts positive group-wise gains and installs them in the recipient's weights or native quantization scales without recipient-side optimization. On a quantized recipient, it preserves integer codes, zero points and storage layout. For a Qwen3-1.7B floating-point/GPTQ INT3 pair, bidirectional transfer retains 76.9-90.5% of the text-loss improvement from recipient-specific calibration on fresh WikiText and C4 cohorts excluded from project development. Shuffling the same gain values within each projection worsens both recipients, showing that their learned group assignment matters after transfer. A Llama-3.2-3B control keeps the initial native tensors at all four calibrated projections identical and changes nine projections elsewhere. Cross-installati
    
[^154]: 单元级聚合破坏因果基础模型的覆盖率

    Unit-level aggregation breaks coverage in causal foundation models

    [https://arxiv.org/abs/2610.22770](https://arxiv.org/abs/2610.22770)

    因果基础模型虽然在单元级估计上表现出色，但其可信区间在聚合到单元组（如全样本平均效应）时会严重退化，在ACIC 2016上全样本平均效应的区间仅有20%覆盖真实值。

    

    因果基础模型（CFMs）在处理效应估计方面展现出可喜的成果，其单元级估计在标准基准测试中具有竞争力。然而，许多因果分析是通过在单元组（如全样本、受处理单元或某个子群）上求平均的效应来为决策提供依据的，对于这类决策，只有当CFM的可信区间经过良好校准时才有用。我们检验了这些区间，发现无论是在简单的合成环境中还是在标准基准上，其覆盖率都会退化。例如，在ACIC 2016数据集上，CausalPFN经温度校准的95%区间，一旦对4802个单元中超过30个单元的组进行聚合，其覆盖率便低于名义水平，而全样本平均值的区间仅在20%的数据集中包含真实值。我们将这一问题归因于这些模型的训练方式及其区间的构造方式：它们一次只输出单个单元的预测分布，并在推理时将这些……（原文摘要到此截断）

    arXiv:2610.22770v1 Announce Type: cross  Abstract: Causal foundation models (CFMs) show promising results for treatment-effect estimation, with unit-level estimates that are competitive on standard benchmarks. Many causal analyses, however, inform decisions through effects averaged over groups of units, such as the full sample, the treated units, or a subgroup, and for such decisions a CFM is useful only if its credible intervals are calibrated. We examine these intervals and find that their coverage degrades, both in simple synthetic settings and on standard benchmarks. On ACIC 2016, for example, temperature-calibrated 95% intervals from CausalPFN fall below nominal coverage beyond groups of 30 out of 4802 units, and intervals for the full-sample average contain the true value in only 20% of datasets.   We trace this problem to how these models are trained and how their intervals are formed. They output a predictive distribution for one unit at a time and, at inference time, combine t
    
[^155]: 一个通信器，三家厂商：将MoA预测理论扩展到真实的NVIDIA+AMD+CPU MPI集成系统

    One Communicator, Three Vendors: Extending the MoA Predictive Theory to a Real NVIDIA+AMD+CPU MPI Ensemble

    [https://arxiv.org/abs/2610.22769](https://arxiv.org/abs/2610.22769)

    本文首次将NVIDIA A100、AMD MI100和CPU三种异构进程纳入同一个MPI通信器，提出基于1/t_i权重分配的MoA预测理论，并通过实验揭示线性假设在MI100上失效、而启动几何模型能准确预测A100的运行时间行为。

    

    arXiv:2610.22769v1 公告类型：交叉 摘要：配套的MPI扩展论文验证了首个包含通信的机器形状集成系统：在NCSA Delta的一个节点上，四个NVIDIA A100进程和四个CPU进程在同一个Open MPI作业下通过MPI_Gatherv重新组装。本文将其扩展为多厂商通信器。一个跨越Delta的A100和AMD MI100分区的Slurm异构作业，首次将A100、MI100和CPU进程（共九个）置于同一个通信器之下。我们给出了理论（三个并发实现中w_i正比于1/t_i的权重分配；以及将异构作业规范作为一种可接受的指令），并报告了真实的运行结果，包括运行中所暴露的问题：1/t分配假设时间与行数呈线性关系，但这一假设在MI100进程上不成立（6行数据耗时597毫秒，而线性预测仅为1.6毫秒）。一个启动几何模型——其预测在测量之前就已记录——在A100上取得成功（时间在128至3456行之间保持平稳，并恰好在3520和6976处出现阶跃），并且

    arXiv:2610.22769v1 Announce Type: cross  Abstract: The companion MPI-extension paper validated the first communication-bearing machine-shape ensemble: four NVIDIA A100 ranks and four CPU ranks on one NCSA Delta node under one Open MPI job, reassembled by MPI_Gatherv. This paper extends it to a multi-vendor communicator. A Slurm heterogeneous job spanning Delta's A100 and AMD MI100 partitions places A100, MI100 and CPU ranks (nine in all) under one communicator for the first time. We give the theory (a w_i proportional to 1/t_i weight split for three concurrent realizations; a het-job specification as an admissible directive) and report real results, including what the run exposed: the 1/t split assumes time linear in rows, and this fails for the MI100 rank (6 rows took 597 ms against a linear prediction of 1.6 ms). A launch-geometry model, with predictions recorded before measuring, succeeds for the A100 (time flat from 128 to 3456 rows, stepping at exactly 3520 and 6976) and places th
    
[^156]: 面向离散空间贝叶斯优化的采集函数引导生成方法

    Acquisition Function-Guided Generation for Bayesian Optimization over Discrete Spaces

    [https://arxiv.org/abs/2610.22763](https://arxiv.org/abs/2610.22763)

    该方法将预训练生成模型与数据空间代理模型相结合，通过扩散桥将采集函数倾斜分布转化为平滑的后验平均采集函数，从而引导生成模型采样候选解，实现便于融入领域知识的离散空间贝叶斯优化。

    

    arXiv:2610.22763v1 公告类型：new 摘要：离散空间上的贝叶斯优化（BO）由于梯度缺失和有效解的稀疏性而充满挑战。潜在空间BO通过在生成模型的潜在空间中拟合代理模型来规避这两个问题，但这使得原本在原始离散空间中自然表达的领域知识难以直接融入。在这项工作中，我们转而将预训练的生成模型与直接在数据空间中定义的代理模型相结合，从而可以方便地引入领域特定核函数等归纳偏置。我们的方法不是在离散设计空间上最大化采集函数，而是让生成模型在一个由采集函数倾斜的分布下采样候选解。为了恢复用于采样的有用梯度，我们将该倾斜分布嵌入到一个扩散桥中，由此得到平滑的、后验平均化的采集函数。这引出了采集函数引导的生成……（原文摘要在此处被截断）

    arXiv:2610.22763v1 Announce Type: new  Abstract: Bayesian optimization (BO) over discrete spaces is challenging due to missing gradients and the sparsity of valid solutions. Latent-space BO sidesteps both issues by fitting a surrogate in the latent space of a generative model, which however substantially complicates the direct incorporation of domain knowledge that is naturally expressed in the original discrete space. In this work, we instead combine a pretrained generative model with a surrogate defined directly in data space, where inductive biases such as domain-specific kernels can be readily incorporated. Rather than maximizing the acquisition function over the discrete design space, our method samples candidates from the generative model under a distribution tilted by the acquisition function. To recover useful gradients for sampling, we embed this tilted distribution into a diffusion bridge that yields a smooth, posterior-averaged acquisition function. This leads to \emph{acqui
    
[^157]: 面向刚体运动与操作规划的形状感知神经时间场

    Shape-Aware Neural Time Fields for Rigid-Body Motion and Manipulation Planning

    [https://arxiv.org/abs/2610.22762](https://arxiv.org/abs/2610.22762)

    提出形状感知神经时间场框架，通过学习可复用的行进时间表示来显式刻画刚体平移与旋转所受的不同几何约束，并结合基于采样的规划器，实现高效的刚体运动规划与非抓取操作规划。

    

    刚体运动规划是机器人运输与操作任务的基础，物体的位置和朝向共同决定了刚体能否在杂乱环境中移动。这种耦合使得规划极具挑战性，因为平移和旋转会受到周围几何环境截然不同的约束。我们提出了形状感知神经时间场，这是一种基于物理信息的神经运动规划框架，它为刚体运动学习一种可复用的行进时间（即代价至终，cost-to-go）表示，同时显式地考虑这些不同的几何限制。所学习的场能够刻画平移与旋转中的局部运动难度，并与一个基于采样的规划器相结合，该规划器在向目标推进与到达邻近位姿的代价之间进行权衡。我们在二维和三维刚体规划任务以及通过物体推挤实现的非抓取操作任务中，在多样化环境下对该框架进行了评估。结果显示……

    arXiv:2610.22762v1 Announce Type: cross  Abstract: Rigid-body motion planning is fundamental to robotic transport and manipulation, where both object position and orientation determine whether a body can move through cluttered environments. This coupling makes planning chal- lenging because translation and rotation can be constrained very differently by surrounding geometry. We propose Shape- Aware Neural Time Fields, a physics-informed neural motion planning framework that learns a reusable travel-time, or cost- to-go, representation for rigid-body motion while explicitly accounting for these distinct geometric restrictions. The learned field captures local motion difficulty in translation and rotation and is paired with a sampling-based planner that balances progress toward the goal with the cost of reaching nearby poses. We evaluate the framework across 2D and 3D rigid- body planning tasks and non-prehensile manipulation through object pushing in diverse environments. Results show s
    
[^158]: 拒答并非鲁棒性：度量自回归语言模型中的有害可达性

    Refusal Is Not Robustness: Measuring Harmful Reachability in Autoregressive Language Models

    [https://arxiv.org/abs/2610.22754](https://arxiv.org/abs/2610.22754)

    该论文提出有界延续分析（BCA），将自回归生成建模为随机过程并借助精确概率模型检测，量化了模型在采样出拒答的表面下仍能到达有害内容的概率（有害可达性），弥补了仅用攻击成功率评估 LLM 安全性的不足。

    

    通过越狱攻击对大语言模型（LLM）进行安全评估通常以攻击成功率（ASR）为指标：即生成的回复是否违反了安全防护。然而，ASR 仅评估采样得到的输出，对底层生成过程几乎不提供任何信息。自回归语言模型在可能的输出上定义了一个分布，一次采样得到的拒答可能与有害生成路径上存在的大量概率质量同时共存。我们通过有界延续分析（Bounded Continuation Analysis, BCA）来研究这种隐藏结构，该方法将自回归生成建模为在词元前缀上的随机过程，并计算生成过程在有限步内到达有害延续内容的概率。BCA 构建一棵有界延续树，用语义危害评估器对终端延续进行标注，并利用精确的概率模型检测来估计有害可达性。BCA 对 ASR 形成补充，它度量了在一次采样得到的拒答之下，仍有多少有害行为是可以到达的。

    arXiv:2610.22754v1 Announce Type: new  Abstract: LLM safety evaluations via jailbreaking typically index on attack success rate (ASR): whether a generated response violates safeguards. However, ASR only evaluates sampled outputs and provides little information on the underlying generation process. Autoregressive language models define distributions over possible outputs, and a sampled refusal can coexist with substantial probability mass on harmful generation paths. We study this hidden structure through bounded continuation analysis (BCA), which models autoregressive generation as a stochastic process over token prefixes and computes the finite-horizon probability that generation reaches a harmful continuation. BCA constructs a bounded continuation tree, labels terminal continuations with a semantic harm evaluator, and estimates harmful reachability using exact probabilistic model checking. BCA complements ASR by measuring how much harmful behavior remains accessible beneath a sampled
    
[^159]: 原生几何与线性可恢复性：对齐如何改变柏拉图式缩放

    Native Geometry vs. Linear Recoverability: How Alignment Changes Platonic Scaling

    [https://arxiv.org/abs/2610.22749](https://arxiv.org/abs/2610.22749)

    本研究通过对五个模型家族的天文图像嵌入进行对比，发现监督式Ridge线性对齐能够显著增强模型规模与跨数据集表示对应性之间的关系，且该效应并非源于嵌入维度优势或简单的旋转缩放变换，表明柏拉图式表示收敛主要通过监督对齐而非原生余弦几何体现。

    

    更大的模型是否会对同一个世界学习到越来越相似的表示？柏拉图表示假说（PRH）预测它们应当如此，但“相似”可以指原生余弦几何——即在 any 学习到的对齐之前，由余弦相似性诱导的最近邻结构——也可以指线性可恢复性：由监督线性映射所揭示的对应关系。通过在五个模型家族中使用匹配的Legacy Survey与Hyper Suprime-Cam（HSC）嵌入，我们发现相对于原生余弦几何，Ridge对齐增强了模型规模与跨数据集邻域对应性之间的关系。该效应在所有五个家族中均出现，并且在将两种表示都降维到相同维度后依然存在，因此这并非仅仅因为更宽的嵌入为对齐提供了更多可用维度。学习到的映射远非旋转或均匀缩放，且稀疏自编码器（SAE）与块稀疏特征化器（BSF）特征给出……

    arXiv:2610.22749v1 Announce Type: new  Abstract: Do larger models learn increasingly similar representations of the same world? The Platonic Representation Hypothesis (PRH) predicts that they should, but similar can mean native cosine geometry---nearest-neighbour structure induced by cosine similarity before any learned alignment---or linear recoverability: correspondence revealed by a supervised linear map. Using matched Legacy Survey and Hyper Suprime-Cam (HSC) embeddings across five model families, we find that Ridge alignment strengthens the relationship between model size and cross-dataset neighbourhood correspondence relative to native cosine geometry. The effect appears in all five families and persists when both representations are reduced to dimension, so it is not simply that wider embeddings give alignment more dimensions to work with. The learned maps remain far from rotations or uniform scaling, and Sparse Autoencoder (SAE) and Block-Sparse Featuriser (BSF) features give n
    
[^160]: 无后见之明的新闻阅读：NoLBERT主题与股票截面收益

    Reading the News Without Hindsight: NoLBERT Topics and the Cross-Section of Stock Returns

    [https://arxiv.org/abs/2610.22745](https://arxiv.org/abs/2610.22745)

    该论文提出仅使用1976-1995年历史文本预训练的NoLBERT编码器来构建无后见之明偏差的新闻主题，并发现将注意力冲击衡量窗口延长至63个交易日可显著提升基于新闻的股票截面收益预测效果。

    

    读取金融新闻的语言模型通常是在与测试年份相同的文本上训练的，因此回测可能因后见之明而受益。我们基于NoLBERT构建新闻主题，NoLBERT是一个仅在1976年至1995年间发布的文本上进行预训练的编码器，并通过对同一文章的段落进行对比学习微调，将其适配为句子编码器，整个过程不使用任何标签或收益率数据。我们在2012年至2024年的126万篇文章上，将其与LDA、MiniLM和FinBERT等阅读器进行比较，将每个阅读器的主题转换为注意力冲击和股票暴露，并用八个模型在三种再平衡频率下预测美国市值最大的1,000只股票的收益率。模型在2013-2019年数据上拟合，选择在2020-2021年数据上做出，2022-2024年数据被留出用于测试。验证得分最好的实验在每种频率下都出现亏损。将注意力冲击的衡量窗口从过去5个交易日改为过去63个交易日，能提高所有阅读器的测试秩IC（rank IC）。NoLBERT给出最稳……（原文此处截断）

    arXiv:2610.22745v1 Announce Type: cross  Abstract: Language models that read financial news are usually trained on text from the same years used to test them, so a backtest can reward hindsight. We build news topics from NoLBERT, an encoder pretrained only on text published between 1976 and 1995, adapted into a sentence encoder by contrastive fine-tuning on passages of the same article, with no labels or returns. We compare it with LDA, MiniLM and FinBERT readers on 1.26 million articles from 2012 to 2024, convert each reader's topics into attention shocks and stock exposures, and forecast the returns of the 1,000 largest US stocks with eight models at three rebalancing frequencies. Models are fitted on 2013-2019, choices are made on 2020-2021, and 2022-2024 is held out. The experiment with the best validation score loses money at every frequency. Measuring attention shocks against the past 63 trading days instead of 5 raises the test rank IC of every reader. NoLBERT gives the most sta
    
[^161]: OrderFlowDiT：用于概率性限价订单供给预测的传输正则化扩散Transformer

    OrderFlowDiT: Transport-Regularized Diffusion Transformer for Probabilistic Limit-Order Supply Forecasting

    [https://arxiv.org/abs/2610.22727](https://arxiv.org/abs/2610.22727)

    该论文提出OrderFlowDiT，一个结合非平衡最优传输辅助损失的条件扩散Transformer，能够对未归一化的价格-成交量场进行建模，联合预测未来五分钟限价订单的价格放置位置、总量和预测不确定性，并配套开发了全场评估框架。

    

    我们研究在给定当前订单簿和近期订单流的条件下，未来五分钟内提交的限价订单量的条件分布。该任务需要联合捕捉价格放置位置、总量以及预测不确定性。逐坐标的扩散损失没有明确考虑价格位移，而将场归一化为单位质量则会移除关于总量的信息。在选定的汇总统计量上的一致性也不能保证对联合场分布的忠实还原。我们提出OrderFlowDiT，这是一个针对未归一化的买卖双方价格-成交量场的条件扩散Transformer，采用非平衡最优传输（UOT）辅助损失进行训练。我们还开发了一个全场评估框架，该框架将对手的价格放置位置和总量敏感的传输成本与适当的评分规则和校准诊断相结合。在针对30只美国（股票）的1950个验证条件的主要比较中……

    arXiv:2610.22727v1 Announce Type: cross  Abstract: We study the conditional distribution of limit-order volume submitted over the next five minutes, given the current order book and recent order flow. The task requires capturing price placement, total volume, and predictive uncertainty jointly. Coordinate-wise diffusion losses do not explicitly account for price displacement, while normalizing fields to unit mass removes information about total volume. Agreement on selected summary statistics also does not establish fidelity to the joint field distribution. We introduce OrderFlowDiT, a conditional Diffusion Transformer for unnormalized bid and ask volume-at-price fields, trained with an unbalanced optimal-transport (UOT) auxiliary loss. We also develop a full-field evaluation framework that combines a transport cost sensitive to price placement and total volume with proper scoring rules and calibration diagnostics. In the primary comparison on 1950 validation conditions across 30 U.S. 
    
[^162]: LLM-Attacker：用于切分式与分布式大语言模型系统隐私评估的统一红队框架

    The LLM-Attacker: A Unified Red-Team Framework for Privacy Evaluation of Split and Distributed LLM Systems

    [https://arxiv.org/abs/2610.22722](https://arxiv.org/abs/2610.22722)

    本文提出LLM-Attacker统一红队评估框架，通过记录攻击者的选择实现切分式与分布式大语言模型隐私评估的可复现与可比较，并揭示训练期间返回梯度会使token恢复率从94.20%升至97.38%、精确文档恢复率从13.71%升至37.77%。

    

    切分式语言模型将输入文本保留在客户端，但会将中间表示发送到服务器，攻击者可以从这些消息中重建文本。当攻击者看到不同的消息、掌握不同的模型权重，或使用不同的数据集和恢复度量时，隐私评估之间的比较就变得十分困难。我们提出了LLM-Attacker，一个记录这些攻击选择的框架，使研究人员能够检查、比较和扩展评估。兼容的攻击可以重用已记录的消息，其预测结果会被保存以供独立评分。评分规则要求在防御通过之前必须存在一个关联的无防御参考基线。报告还会识别出尚未经过测试的攻击。在一项受控的GPT-2实验中，添加训练期间返回的梯度使token恢复率从94.20%提升到97.38%，精确文档恢复率从13.71%提升到37.77%。token增益的95%置信区间高于测试前设定的两个百分点阈值。

    arXiv:2610.22722v1 Announce Type: cross  Abstract: Split language models keep input text on a client but send intermediate representations to a server. Attacks can reconstruct text from these messages. Comparing privacy evaluations is difficult when attackers see different messages, know different model weights or use different datasets and recovery measures. We present LLM-Attacker, a framework that records these choices and lets researchers check, compare and extend an evaluation. Compatible attacks reuse recorded messages, and their predictions are saved for independent scoring. The scoring rule requires a linked undefended reference before a defence can pass. Reports also identify attacks that remain untested. In a controlled GPT-2 experiment, adding gradients returned during training increased token recovery from 94.20% to 97.38% and exact-document recovery from 13.71% to 37.77%. The 95% interval for the token gain lay above the two-percentage-point threshold fixed before testing.
    
[^163]: CoLU：面向大语言模型的置信度加权逐层遗忘方法

    CoLU: Confidence-Weighted Layerwise Unlearning for Large Language Models

    [https://arxiv.org/abs/2610.22721](https://arxiv.org/abs/2610.22721)

    该论文提出CoLU方法，通过置信度加权的遗忘与保留正则化仅更新单个选定的Transformer层来实现大语言模型遗忘，并配合基于归一化遗忘-保留分数的简短层筛选策略高效选取最优层，在降低计算成本的同时更好地平衡了遗忘效果与知识保留。

    

    大语言模型（LLM）遗忘旨在让模型遗忘特定信息的同时保留其他信息，而无需从头重新训练。当前最先进的方法如NPO、SimNPO和SatImp通过全模型微调来平衡遗忘与保留，而FOM-UL则探索了选择性层更新策略。先前的研究表明，抑制目标信息可能会损害被保留的知识，并且即使使用相同的遗忘损失，这种权衡在不同层之间也存在差异。在本工作中，我们提出了置信度加权逐层遗忘方法，该方法将一种新颖的置信度加权遗忘与保留正则化相结合，仅更新单个选定的Transformer层，而非对整个模型进行遗忘。为了降低CoLU的计算成本，我们进一步提出了简短层筛选，该方法利用新定义的归一化遗忘-保留分数，通过在遗忘数据集的一个子集上以更少的训练轮次进行简短的遗忘试验来对各个层进行排序。

    arXiv:2610.22721v1 Announce Type: new  Abstract: Large language model (LLM) unlearning aims to forget certain information while preserving others, without retraining from scratch. SoTA methods such as NPO, SimNPO, and SatImp balance forgetting and retain through whole model fine-tuning, while FOM-UL explores selective layer updates. Prior works show that suppressing target information can damage retained knowledge. This trade-off can vary across layers even under the same unlearning loss. In this work, we propose Confidence-Weighted Layerwise Unlearning (CoLU), which combines a novel confidence-weighted forgetting and retain regularization while updating a single selected transformer layer, rather than the whole model for unlearning. To reduce CoLU compute cost, we further propose Brief Layer Screening (BLS), which uses a newly defined normalized forgetting-retention (NFR) score to first rank layers through a brief unlearning trials using a subset of forgetting dataset and fewer epochs
    
[^164]: 长程伊辛模型基准上多项式图滤波器的谱极限

    Spectral Limits of Polynomial Graph Filters on the Long-Range Ising Model Benchmark

    [https://arxiv.org/abs/2610.22713](https://arxiv.org/abs/2610.22713)

    该论文证明了在长程伊辛模型基准上，作为各向同性消息传递和谱图网络线性核心的邻接矩阵多项式滤波器存在一个无法通过提高多项式次数突破的误差下限，并指出一个约五十个系数的跨步滤波器能够突破该极限。

    

    长程伊辛模型基准（LRIM）通过其构造方式保证了长程相关性：每个标签是节点的自旋乘以经幂律耦合卷积后的自旋场，因此任何模型与目标之间的距离都是可测量的。我们证明，若读出层的滤波器是邻接矩阵的多项式——即各向同性消息传递和谱图网络的线性核心——则会产生一个任何多项式次数都无法降低的误差下限，其原因在于目标函数沿着谱的一个等值集变化，而每一个此类滤波器在该等值集上都取常数值。通过精确计算，该误差下限以及每个次数对应的最优滤波器，能够将两个改编自已发表架构的谱网络的表现预测到十分之一个数量级以内的精度。滤波器只有在对邻接矩阵所混同的各个偏移量分别赋予权重，同时仍能解析整个晶格上的耦合时，才能突破该误差下限。一个约含五十个系数的跨步滤波器同时做到了这两点：在拟合训练标签之后，其测试误差……（摘要原文在此处截断）

    arXiv:2610.22713v1 Announce Type: new  Abstract: The Long-Range Ising Model benchmark (LRIM) guarantees long-range dependence by construction: each label is the node's spin times the spin field convolved with a power-law coupling, so any model's distance from the target is measurable. We prove that a readout whose filter is a polynomial in the adjacency, the linear core of isotropic message passing and spectral graph networks, leaves an error floor that no degree lowers, because the target varies along a level set of the spectrum on which every such filter is constant. Computed exactly, the floor and the best filter of each degree predict two spectral networks adapted from published architectures to within a tenth of a decade. A filter escapes the floor only if it weighs separately the offsets the adjacency lumps together, and must still resolve the coupling across the whole lattice. A strided filter of about fifty coefficients does both: fitted to the training labels, its test error i
    
[^165]: 标准化流映射

    Normalizing Flow Maps

    [https://arxiv.org/abs/2610.22711](https://arxiv.org/abs/2610.22711)

    本文提出标准化流映射（NFM），将流映射的少步生成能力与标准化流的精确似然评估相结合，通过学习底层动力系统的积分器，在构造上同时实现高效少步采样和精确密度估计。

    

    可扩展的神经动力系统学习方法已将扩散模型、流匹配以及基于流映射的少步生成器推向了生成式建模的前沿。尽管这些最先进的模型源自利用易处理似然的经典生成式建模方法（例如标准化流和自回归模型），但它们在获取显式模型密度方面能力有限。在本文中，我们将少步生成与可处理的样本似然这两条技术路线相结合，提出了标准化流映射，这是一类精确可逆的推前映射，同时推广了标准化流和流映射。更准确地说，NFM 学习底层动力系统的积分器，但关键在于，它们在构造上即可提供高效的精确似然评估。为实现 NFM 的可扩展训练，我们推导了 NFM 固有的两个互补的一致性维度，它们可产生（自）蒸馏目标：（1）……

    arXiv:2610.22711v1 Announce Type: new  Abstract: Scalable approaches to learning neural dynamical systems have placed diffusion, flow matching, and few-step generators in flow maps at the forefront of generative modeling. Such state-of-the-art models have limited access to explicit model densities despite evolving from classical generative modeling approaches that leveraged tractable likelihoods, e.g., Normalizing Flows and Autoregressive models. In this paper, we combine these threads of few-step generation and tractable sample likelihoods and introduce Normalizing Flow Maps (NFM), a class of exactly invertible pushforward maps that generalizes normalizing flows and flow maps. More precisely, NFMs learn the integrator of an underlying dynamical system, but crucially, they provide efficient exact likelihood evaluation by construction. To enable scalable training of NFMs, we derive two complementary axes of consistency inherent to NFMs that yield (self) distillation objectives: (1) poin
    
[^166]: 面向查询模板识别的数据溯源聚合与结构描述符

    Data Provenance Aggregation and Structural Descriptors for Query-Template Identification

    [https://arxiv.org/abs/2610.22705](https://arxiv.org/abs/2610.22705)

    本文提出通过聚合元组级Lineage与Why-provenance构建查询级画像并推导溯源结构描述符（PSD），利用数据依赖与结构属性实现查询模板识别。

    

    运行时查询相似性识别支持多项活动，包括工作负载分析、识别重复行为、审计、调试和异常检测。现有的查询相似性评估方法主要依赖SQL文本、执行计划或结果值，但并未考虑输入数据或查询执行期间结果的推导过程。本文提出了一种利用数据溯源进行查询模板识别的新方法。我们通过在整个查询结果上聚合元组级的Lineage和Why-provenance，构建捕获显式数据依赖、出现频率和结构属性的查询级画像。我们还推导出了溯源结构描述符，即使用一小组数值特征来概括这些画像。我们使用这些画像在由参数化T生成的12,500个不同查询实例上进行查询模板识别，从而评估我们提出的方案。

    arXiv:2610.22705v1 Announce Type: cross  Abstract: Runtime query similarity identification supports several activities, including workload analysis, identifying recurring behaviour, auditing, debugging, and anomaly detection. Existing methods for evaluating query similarity mostly rely on SQL text, execution plans, or result values but do not account for input data or how results are derived during query execution.   This paper proposes a novel approach using data provenance for query-template identification. We construct query-level profiles that capture explicit data dependencies, occurrence frequencies, and structural properties by aggregating tuple-level Lineage and Why-provenance across entire query results. We also derive Provenance Structural Descriptors (PSD), which summarise these profiles using a small set of numerical features.   We evaluate our proposals using these profiles to identify query templates over \(12{,}500\) distinct query instances generated from parametrised T
    
[^167]: 基于在线偏好学习的机会约束个性化LLM路由

    Chance-Constrained Personalized LLM Routing with Online Preference Learning

    [https://arxiv.org/abs/2610.22695](https://arxiv.org/abs/2610.22695)

    该论文提出了一种基于机会约束和在线偏好学习的个性化LLM路由算法，通过受约束上下文赌博机框架在最小化推理成本的同时，对实际响应满意度提供了高概率理论保证。

    

    我们研究个性化LLM路由问题，其中路由器需要在具有不同成本和能力的异构模型中进行选择。路由器必须在最小化累积推理成本的同时，确保实际响应质量的可靠性，即使用户偏好未知且需要在部署过程中从反馈中学习。我们将该问题建模为一个受约束的上下文赌博机问题，并开发了一种能够在学习偏好的同时平衡成本与质量的算法。在对数偏好模型和严格可行性条件下，我们相对于最优平稳随机化上下文预言机建立了次线性期望成本遗憾，并对实际累积满意度提供了高概率保证。该框架不仅适用于LLM路由，还可扩展到更广泛的具有可靠性要求的在线模型选择问题。

    arXiv:2610.22695v1 Announce Type: new  Abstract: We study personalized LLM routing, where a router selects among heterogeneous models with different costs and capabilities. The router must minimize cumulative inference cost while ensuring reliable realized response quality, even though user preferences are unknown and must be learned from feedback during deployment. We formulate this problem as a constrained contextual bandit and develop an algorithm that learns preferences while balancing cost and quality. Under a logistic preference model and a strict-feasibility condition, we establish sublinear expected cost regret relative to an optimal stationary randomized contextual oracle and a high-probability guarantee on realized cumulative satisfaction. The framework extends beyond LLM routing to broader online model-selection problems with reliability requirements.
    
[^168]: ReLU神经网络对光滑函数的最优逼近

    Optimal approximation of smooth functions by ReLU neural networks

    [https://arxiv.org/abs/2610.22693](https://arxiv.org/abs/2610.22693)

    本文证明了宽度为O(N)、深度为O(L)的ReLU网络对任意光滑度s和维度D下的Hölder函数类均能达到最优逼近误差$(N^2L^2\log N)^{-s/D}$，并由此导出无额外对数因子的极小极大回归速率。

    

    我们研究了全连接ReLU网络对Hölder球 $\mathcal H_1^s([0,1]^D)$ 的一致逼近问题。除了一维Lipschitz情形外，现有构造与信息论效率极限之间仍存在差距。本文对任意 $s>0$ 和 $D\ge1$ 完全弥合了这一差距：宽度为 $O(N)$、深度为 $O(L)$、参数幅值为 $O(1)$ 的网络可达到 $(N^2L^2\log N)^{-s/D}$ 阶的一致逼近误差，且无需对宽度和深度之间的关系作任何假设。该构造采用了新的编码与解码机制，利用目标函数局部泰勒系数的高阶有限差分界。作为统计上的推论，在相应截断网络类上的经验风险最小化器可以达到极小极大回归速率 $n^{-2s/(2s+D)}$，且不含额外的对数因子。论文还对关键的逼近技术进行了详细阐述，涵盖……

    arXiv:2610.22693v1 Announce Type: cross  Abstract: We study uniform approximation of the H\"older ball $\mathcal H_1^s([0,1]^D)$ by fully connected ReLU networks. Except in the one-dimensional Lipschitz case, existing constructions remain separated from the information-theoretic efficiency limit. This article closes the gap for every $s>0$ and $D\ge1$: networks with width $O(N)$, depth $O(L)$, and parameter magnitude $O(1)$ attain uniform approximation error of order $(N^2L^2\log N)^{-s/D}$. No relation between width and depth is required. The construction uses new encoding and decoding mechanisms that exploit higher-order finite-difference bounds for local Taylor coefficients of the target function. As a statistical consequence, empirical risk minimizers over the corresponding clipped network classes attain the minimax regression rate $n^{-2s/(2s+D)}$ without additional logarithmic factors. The paper also gives a detailed exposition of the key approximation techniques, covering both e
    
[^169]: 用于求解林德布拉德主方程的量子-经典神经网络

    Quantum-Classical Neural Networks for Solving Lindblad Master Equations

    [https://arxiv.org/abs/2610.22691](https://arxiv.org/abs/2610.22691)

    该论文提出一种结合经典神经网络与参数化量子电路的变分框架来求解林德布拉德主方程，通过纯化表示在构造上保证密度矩阵的物理性质，并从理论上证明经典参数化能够扩大有效假设类、提升模型的表达能力与可控性。

    

    变分量子算法已成为模拟开放量子系统的一种有前景的方法，但其表达能力和可训练性仍鲜为人知。我们提出了一个求解林德布拉德主方程的变分框架，通过将经典神经网络与参数化量子电路相结合构成复合拟设，并最小化残差代价函数。密度矩阵通过在扩大的希尔伯特空间上的纯化来表示，从而在构造上保证了正定性、厄米性和单位迹，同时所有梯度均通过参数移位规则精确计算。除算法设计之外，我们还推导了所提出的复合模型覆盖数的上界，并将其与纯量子神经网络的覆盖数进行比较。我们的分析表明，经典参数化扩大了有效假设类，同时对其复杂度提供了更精细的控制。

    arXiv:2610.22691v1 Announce Type: cross  Abstract: Variational quantum algorithms have emerged as a promising approach for simulating open quantum systems, yet their expressive power and trainability remain poorly understood. We propose a variational framework for solving Lindblad master equations by minimizing a residual cost function using a compositional ansatz obtained by combining a classical neural network with a parameterized quantum circuit. The density matrix is represented through a purification on an enlarged Hilbert space, ensuring positivity, Hermiticity, and unit trace by construction, while all gradients are evaluated exactly via the parameter-shift rule. Beyond the algorithmic formulation, we derive an upper bound for the covering number of the proposed compositional model and compare it with that of purely quantum neural networks. Our analysis shows that the classical parametrization enlarges the effective hypothesis class while providing finer control over its complex
    
[^170]: AlphaBet：用于大语言模型事件预测智能体的强化学习

    AlphaBet: Reinforcement Learning for LLM Event Forecasting Agents

    [https://arxiv.org/abs/2610.22684](https://arxiv.org/abs/2610.22684)

    AlphaBet是一个智能体强化学习框架，通过训练单个LLM策略在事件解决前的多个时间点自主搜索证据并报告概率，利用适当评分规则以已解决事件实现端到端监督，弥补了现有预测训练方法将证据获取排除在学习循环之外的不足。

    

    预测一个尚未解决的事件需要找到相关证据，并判断这些证据在多大程度上支持每种可能的结果。然而，基于结果训练大语言模型（LLM）预测器的现有方法通常从单一预测日期开始使用预先收集好的证据，使得证据获取过程游离于学习循环之外。我们提出了AlphaBet，这是一个智能体强化学习（RL）框架，它训练单个LLM策略在事件解决前的多个日期进行搜索、推理并报告概率。每个带日期的预测都通过适当的评分规则与最终结果对比进行评分，从而使一个已解决的事件能够监督不同信息状态下的决策。我们将预测遗憾分解为证据获取差距和概率报告差距，并证明组相对策略优化（GRPO）的标准差归一化在多分类适当评分规则下可能会扭曲概率报告目标。在GLM-4.5-Air上，AlphaBe……（原文摘要至此截断）

    arXiv:2610.22684v1 Announce Type: cross  Abstract: Forecasting an unresolved event requires finding relevant evidence and deciding how strongly it supports each outcome. Yet outcome-based training of Large Language Model (LLM) forecasters often begins with preassembled evidence at a single prediction date, leaving evidence acquisition outside the learning loop. We introduce AlphaBet, an agentic reinforcement learning (RL) framework that trains a single LLM policy to search, reason, and report probabilities across multiple pre-resolution dates. Each dated forecast is scored by a proper scoring rule against the eventual outcome, allowing one resolved event to supervise decisions under different information states. We decompose forecasting regret into an evidence-acquisition gap and a probability-reporting gap, and show that Group Relative Policy Optimization (GRPO)'s standard-deviation normalization can distort the reporting target under multi-class proper scores. On GLM-4.5-Air, AlphaBe
    
[^171]: 人机交互中的编辑发现

    Edit Discovery in Human-AI Interactions

    [https://arxiv.org/abs/2610.22661](https://arxiv.org/abs/2610.22661)

    提出编辑发现优化（EDO）框架，通过在不知晓是否会产生编辑的情况下智能选择将哪些人机交互发送给成本高昂的编辑器（如人类专家或更强AI），以最小化错过的编辑机会并节约有限资源。

    

    人机交互正在成为改进AI系统的丰富数据来源：虽然许多模型响应已经令人满意，但其他响应所暴露的失败若能被识别并纠正，可能成为宝贵的监督信号。我们通过“编辑器”的概念来研究这一场景：编辑器是一种能力更强的评估过程，它能够检查响应并对其予以确认或修改，但调用它的成本高昂。编辑器可以是人类专家、更强大的AI系统，或者是配备了额外计算、上下文、工具或信息的同一模型。关键挑战在于，在尚不知道是否真的会发生编辑的情况下，就要决定将哪些交互发送给编辑器，因为调用编辑器本身会消耗有限的资源，例如专家时间、推理成本、延迟或工具使用开销。我们将此问题形式化为编辑发现优化，其目标是最小化在被跳过的交互中错过的编辑机会。

    arXiv:2610.22661v1 Announce Type: new  Abstract: Human-AI interactions are becoming a rich source of data for improving AI systems: while many model responses are already satisfactory, others expose failures that could become valuable supervision if identified and corrected. We study this setting through the notion of an editor: a more capable evaluation process that can inspect a response and either confirm it or modify it, but whose invocation is costly. The editor may be a human expert, a stronger AI system, or the same model equipped with additional computation, context, tools, or information. The key challenge is to decide which interactions to send to the editor before knowing whether an edit will actually be made, since invoking the editor itself consumes limited resources such as expert time, inference cost, latency, or tool-use overhead. We formulate this problem as Edit Discovery Optimization (EDO), which seeks to minimize edit opportunities missed among skipped interactions 
    
[^172]: 线性探针是线性表示的证据吗？

    Is a Linear Probe Evidence of a Linear Representation?

    [https://arxiv.org/abs/2610.22655](https://arxiv.org/abs/2610.22655)

    该论文证明线性探针的高准确率不足以证明线性表示的存在——即使是随机特征也能实现线性读出——并提出反向预测性、谱集中度等三种非对称诊断方法，以区分真正的线性世界模型与核机器式的弯曲流形表示。

    

    线性表示假设认为，预训练视觉模型将生成性因素编码为特征空间中可线性解码的方向，并伴随一套标准方法：训练一个线性探针，若准确率高则宣称该因素被表示。我们证明这一测试过于宽松。从渐近意义上说，任何足够丰富的特征映射（包括随机特征映射）都能实现线性读出，因此高探针准确率与两种截然不同的几何结构相容：一种是真正的线性世界模型，其中与潜在变量对齐的子空间主导表示；另一种是核机器，其中潜在变量线性地嵌入在一个大得多的弯曲流形内部。这一区别至关重要，因为与探针正交的弯曲结构对于分布外泛化、鲁棒性和组合迁移都具有重要意义。我们提出三种非对称诊断方法来区分这两种情况：反向预测性、谱集中度以及分布外……（摘要在此处截断）

    arXiv:2610.22655v1 Announce Type: cross  Abstract: The linear representation hypothesis holds that pretrained vision models encode generative factors as linearly decodable directions in feature space, and comes with a standard methodology: train a linear probe, declare a factor represented if accuracy is high. We show this test is too permissive. Asymptotically, any sufficiently rich feature map, including a random one, admits a linear readout, so high probe accuracy is consistent with two very different geometries: a true linear world model, where the latent-aligned subspace dominates the representation, or a kernel machine, where the latents lie linearly inside a much larger curved manifold. This distinction matters because the curved structure orthogonal to the probe is relevant for out-of-distribution generalization, robustness, and compositional transfer. We propose three asymmetric diagnostics that distinguish the two cases: reverse predictivity, spectral concentration, and out-o
    
[^173]: 跨LLM训练阶段切换优化器的短暂代价

    The Ephemeral Cost of Switching Optimizers Across LLM Training Stages

    [https://arxiv.org/abs/2610.22650](https://arxiv.org/abs/2610.22650)

    该论文通过系统性实验首次发现，在LLM预训练与后训练阶段之间切换优化器（如AdamW与Muon之间）并不会造成实质性的性能损失，最终性能主要由各阶段所用优化器的独立效应决定，且在SFT中增加Muon更新比例反而能提升下游性能。

    

    最近的研究表明，在预训练和后续训练阶段之间切换优化器会带来性能损失，但缺乏对这一现象的系统性研究。我们通过从头预训练1.2B参数的Llama-3模型，并改变使用每种优化器进行更新的比例，来研究AdamW与Muon优化器之间的不匹配问题，同时在继续预训练和监督微调（SFT）期间对两种优化器进行类似的插值实验。令人惊讶的是，我们发现几乎没有证据表明存在有意义的优化器不匹配惩罚：最终性能在很大程度上可以由预训练和后训练优化器的独立效应来解释，且随着SFT期间Muon更新比例的增加，下游性能总体上有所提升。尽管匹配预训练优化器可以减少SFT期间的遗忘，但这并不能转化为最终基准测试性能的提升。我们进一步评估了AdamW和M……（摘要在此处截断）

    arXiv:2610.22650v1 Announce Type: new  Abstract: Recent work suggests that switching optimizers between pretraining and subsequent training stages incurs a performance cost, but systematic investigation of this phenomenon is lacking. We study the mismatch between AdamW and Muon optimizers by pretraining 1.2B Llama-3 models from scratch while varying the proportion of updates taken with each optimizer, and similarly interpolating between the two optimizers during continued pretraining and supervised fine-tuning (SFT). Surprisingly, we find little evidence of a meaningful optimizer mismatch penalty: final performance is largely explained by the independent effects of the pretraining and post-training optimizers, with downstream performance generally improving as the proportion of Muon updates during SFT increases. Although matching the pretraining optimizer can reduce forgetting during SFT, this does not translate into improved final benchmark performance. We further evaluate AdamW and M
    
[^174]: 评估ICU脓毒症预测中插补策略的警报列表一致性方法

    Alert-List Concordance for Evaluating Imputation Strategies in ICU Sepsis Prediction

    [https://arxiv.org/abs/2610.22635](https://arxiv.org/abs/2610.22635)

    该研究提出了“警报列表一致性”这一新指标，通过比较前k%警报患者集合的Jaccard相似度来评估不同插补策略在ICU脓毒症预测中是否识别出相同的高风险患者，弥补了传统重建误差和总体判别指标无法衡量患者级别警报一致性的不足。

    

    预处理是临床预测中一种未被充分衡量的可靠性风险：模型可以保持相似的AUROC，却改变了哪些患者会收到警报。插补质量通常通过重建误差或总体判别性能来评估，但这两者都无法直接衡量模型在标记哪些患者方面的一致性。在匹配的警报率下，总体性能相似的流程可能产生不同的患者列表。在真实ICU记录中频繁且呈结构性的数据缺失情况下，这一问题尤为突出。因此，我们提出了“警报列表一致性”（alert-list concordance）这一指标，用于考察两个流程是否识别出相同的高风险患者，通过其前k%警报集合的Jaccard相似度以及患者级别的bootstrap置信区间来衡量。在PhysioNet/CinC 2019数据集（40,336名患者，71.9%缺失率）上，我们使用固定的XGBoost配置，比较了九种插补策略与一个原生缺失基线，模型经过拟合和Platt校准……

    arXiv:2610.22635v1 Announce Type: new  Abstract: Preprocessing is an under-measured reliability risk in clinical prediction: a model can keep a similar AUROC and still change who receives an alert. Imputation quality is commonly evaluated using reconstruction error or aggregate discrimination, neither of which directly measures agreement on which patients are flagged. At a matched alert rate, pipelines with similar aggregate performance can produce different patient lists. This question is especially relevant under the frequent, structured missingness of real ICU records. We therefore introduce alert-list concordance, which asks whether two pipelines identify the same high-risk patients, measured by the Jaccard similarity of their top-k% alert sets with patient-level bootstrap confidence intervals. On PhysioNet/CinC 2019 (40,336 patients, 71.9% missingness), we compare nine imputation strategies and a native-missingness baseline using a fixed XGBoost configuration, fitted and Platt-cal
    
[^175]: 基于探针监督图卷积循环网络利用稀疏固定传感器实现全网高速公路速度的即时估计与预测

    Network-Wide Freeway Speed Nowcasting and Forecasting from Sparse Fixed Sensors with a Probe-Supervised Graph Convolutional Recurrent Network

    [https://arxiv.org/abs/2610.22630](https://arxiv.org/abs/2610.22630)

    该论文提出GCRN+模型，一种探针监督的图卷积循环网络，仅利用15.6%路段的固定传感器数据即可对达拉斯—沃斯堡地区全部4,127条有向高速公路路段实现速度的即时估计与未来一小时预测，且随网络规模线性扩展并可在CPU上训练。

    

    交通管理部门需要其路网上每条路段的当前及近期未来速度，但固定传感器仅覆盖了部分路网。我们研究了仅利用643条路段（15.6%）上的传感器，对达拉斯—沃斯堡地区全部4,127条有向高速公路路段的逐小时速度进行联合即时估计（nowcasting）与下一小时预测的问题。在训练阶段，历史探针车辆数据对所有路段进行监督；而在推理阶段仅使用传感器读数。我们将这一设置称为“探针监督稀疏感知”。我们提出了GCRN+，一种预测速度相对历史速度曲线偏差的图卷积循环网络。该方法将无参数、方向感知的传感器异常核插值（同时用作输入特征和输出基线），与有向路网图上的扩散卷积GRU、低秩自适应图以及可学习的路段嵌入相结合。该模型随网络规模线性扩展，并可在CPU上完成训练。传感器读数由探针速度模拟生成

    arXiv:2610.22630v1 Announce Type: cross  Abstract: Traffic agencies need current and near-future speeds on every link of their network, but their fixed sensors cover only part of it. We study joint nowcasting and next-hour forecasting of hourly speeds on all 4{,}127 directional freeway links of the Dallas--Fort Worth region from sensors on 643 links (15.6%). Historical probe-vehicle data supervise every link in training; only sensor readings are used at inference. We call this setting probe-supervised sparse sensing. We propose GCRN+, a graph convolutional recurrent network that predicts deviations from a historical speed profile. It combines a parameter-free, direction-aware kernel interpolation of sensor anomalies, used as input features and as an output baseline, with a diffusion-convolutional GRU on the directed road graph, a low-rank adaptive graph and learnable link embeddings. It scales linearly with network size and trains on a CPU. Sensor readings are emulated from probe speed
    
[^176]: 理解外层Nesterov动量在随机伪梯度上的稳定性

    Understanding Stability of Outer Nesterov Momentum on Stochastic Pseudo-Gradients

    [https://arxiv.org/abs/2610.22623](https://arxiv.org/abs/2610.22623)

    该论文推导出随机伪梯度下外层Nesterov动量的精确均方稳定性准则，揭示内部批大小决定了局部步数增加会使训练稳定或不稳定，且稳定性所需的工作节点数量可能随局部步数呈指数级增长。

    

    大型神经网络通常在多个工作节点上分布式训练，而节点间的通信是主要开销。因此，工作节点在向采用Nesterov动量的外层优化器发送其净参数位移（即伪梯度）之前，会先执行若干局部优化步骤。我们研究外层动量、局部步数以及工作节点间平均操作如何控制训练动态，特别是其稳定性。对于随机二次函数，我们在任意维度上推导出精确的均方稳定性准则，该准则揭示了内部批大小决定了增加局部步数最终会使外层迭代变得更稳定还是更不稳定。此外，尽管对独立工作节点的伪梯度取平均可以容许更大的外层动量取值，但满足稳定性所需的工作节点数量可能随局部步数呈指数级增长。最后，我们表明……（摘要原文在此处截断）

    arXiv:2610.22623v1 Announce Type: new  Abstract: Large neural networks are commonly trained across many workers, and communication between them is a major cost. Workers therefore take several local optimization steps before sending their net parameter displacements, or pseudo-gradients, to an outer optimizer, which often applies Nesterov momentum. We study how the outer momentum, the number of local steps and the averaging over workers control the training dynamics, and in particular their stability. For random quadratics, we derive an exact mean-square stability criterion in any dimension, which reveals that the inner batch size determines whether increasing the number of local steps eventually stabilizes or destabilizes the outer iteration. Moreover, while averaging the pseudo-gradients of independent workers can make larger outer momentum values admissible, the number of workers required for stability can grow exponentially with the number of local steps. Finally, we show that the m
    
[^177]: 面向二值与三值AI模型可扩展边缘端训练的噪声整形采样与退火方法

    Noise-shaped Sampling and Annealing for Scalable On-the-edge Training of Binary and Ternary AI Models

    [https://arxiv.org/abs/2610.22621](https://arxiv.org/abs/2610.22621)

    本文提出ΣΔ-Sampler优化器，利用二值/三值网络训练损失与高阶伊辛模型能量的结构等价性，通过一阶ΣΔ噪声整形环路嵌入随机采样-退火步骤，将量化与采样噪声推挤出低频适应频带，从而实现可扩展的边缘端离散权重网络训练。

    

    二值与三值神经网络的训练损失在离散权重上展开为多线性多项式后，在结构上等价于层级化高阶伊辛模型的能量。我们利用这一等价性将伊辛求解器应用于离散权重网络训练，同时考虑到二者在目标上的根本差异：伊辛求解器寻求固定能量函数的基态，而训练寻求的是能够泛化到未见数据的解。我们的核心贡献是 ΣΔ-Sampler——一种优化器，它通过在一阶 ΣΔ 噪声整形环路内嵌入一个作为有噪量化器的随机采样-退火步骤来调和上述目标。该环路将来自权重量化、高阶阻挫以及小批量采样的噪声推挤出低频适应频带，使得离散权重更新能够跟随隐藏在崎岖损失地形中的平滑下降轨迹……

    arXiv:2610.22621v1 Announce Type: new  Abstract: The training loss of binary and ternary neural networks, expanded as a multilinear polynomial over the discrete weights, is structurally equivalent to the energy of a hierarchical high-order Ising model. We exploit this equivalence to apply Ising solvers to discrete-weight network training, while accounting for a fundamental difference in objectives: an Ising solver seeks the ground state of a fixed energy function, whereas training seeks solutions that generalize to unseen data. Our key contribution is $\Sigma\Delta$-Sampler, an optimizer that reconciles these objectives by embedding a stochastic sampling-and-annealing step as a noisy quantizer inside a first-order $\Sigma\Delta$ noise-shaping loop. The loop pushes noise from weight quantization, high-order frustration, and mini-batch sampling out of the low-frequency adaptation band, so that discrete weight updates follow the smooth descent trajectories embedded in the rugged loss land
    
[^178]: 依赖偏移对齐：根据视觉-语言模型所信任的线索对其进行排名

    Shift-Reliance Alignment: Ranking Vision-Language Models by the Cues They Trust

    [https://arxiv.org/abs/2610.22610](https://arxiv.org/abs/2610.22610)

    该论文提出一种“线索依赖度”测量方法，通过编辑少量图像校准检测器，无需在目标域上运行任何候选模型，即可预测哪个视觉-语言模型在目标域的物体存在性问题上表现最佳，并能预测模型被贴纸攻击误导或产生幻觉的倾向。

    

    视觉-语言模型通常是从排行榜中挑选的，而排行榜对所有应用只提供同一个排名。然而，对于场景有所暗示但实际并不包含的物体相关的问题，在POPE幻觉基准上错误率最低的模型，并不是在这类问题上错误率最低的模型。针对关于物体是否存在的“是/否”问题，我们证明：只需650到1,200个带标注的目标域样本用于校准一个检测器，就无需在目标域上运行任何候选模型即可预测出该目标域上的最佳模型。我们通过编辑100张带标注的源图像，使某个线索（例如图像中的场景上下文或文字）指向错误答案，并与匹配的对照编辑进行比较，从而测量每个模型对该线索的依赖程度。这种依赖度指标能够预测哪些VLM会被物理贴纸攻击误导（在九个未参与设计该指标的模型上Spearman相关系数为0.93），以及哪些VLM会在未经编辑的COCO照片上产生幻觉，其效果超越了……（原文截断）

    arXiv:2610.22610v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are usually chosen from leaderboards, which give one ranking for every application. On questions about objects that a scene suggests but does not contain, however, the model with the lowest error on the POPE hallucination benchmark is not the one with the lowest error on such questions. For yes/no questions about object presence, we show that the best model for a target domain can be predicted without running any candidate on that domain, given 650 to 1,200 labeled target items that calibrate a detector. We measure each model's reliance on a cue, a feature such as scene context or text in the image, by editing 100 labeled source images so that the cue suggests the wrong answer, and comparing with a matched control edit. Reliance predicts which VLMs are misled by physical sticker attacks (Spearman .93 on nine models not used to design the score) and which hallucinate on unedited COCO photographs, beyond wha
    
[^179]: 单例稀有性至关重要（附补充材料）

    Rarity of Singletons Matters & Supplementary Materials

    [https://arxiv.org/abs/2610.22604](https://arxiv.org/abs/2610.22604)

    提出脆弱性感知（FA）框架，通过量化单例统计量的脆弱性，以严格凹且具有唯一闭式解的熵-KL优化对经典缺失质量与未见物种估计量进行自适应轻量级修正。

    

    从有限样本中估计缺失质量与未见物种数量，很大程度上依赖于稀有事件的占用统计量，尤其是单例。在小样本情形下，这些统计量可能具有高度波动性，使得经典估计方法对抽样波动十分敏感、潜在不可靠。我们提出了一个脆弱性感知框架，显式量化这种敏感性，并对传统的缺失质量与未见物种估计量进行自适应正则化。利用由概率质量与占用计数导出的基于单例的脆弱性度量，我们构建了一个熵-Kullback-Leibler（KL）优化问题，根据基础估计量的脆弱性来调整对它们的依赖程度。所得到的目标函数严格凹，具有唯一的闭式解，且无需Poisson化或渐近近似。该框架为经典估计方法（包括Good-Turing和Good-...）提供了轻量级的修正。

    arXiv:2610.22604v1 Announce Type: new  Abstract: Estimating missing-mass and unseen-species counts from finite samples relies heavily on rare-event occupancy statistics, particularly singletons. In small-sample regimes, these statistics can be highly variable, making classical estimators sensitive to sampling fluctuations and potentially unreliable. We propose a fragility-aware (FA) framework that explicitly quantifies this sensitivity and adaptively regularizes conventional missing-mass and unseen-species estimators. Using singleton-based fragility measures derived from probability mass and occupancy counts, we formulate an entropy--Kullback--Leibler (KL) optimization that adjusts the reliance on base estimates according to their fragility. The resulting objectives are strictly concave, admit unique closed-form solutions, and require no Poissonization or asymptotic approximations. The framework provides a lightweight correction to classical estimators, including Good--Turing and Good-
    
[^180]: 大语言模型能预测机器学习实验的结果吗？

    Can LLMs predict the outcomes of machine learning experiments?

    [https://arxiv.org/abs/2610.22599](https://arxiv.org/abs/2610.22599)

    该论文构建了包含288个LLM自生成实验想法的数据集，通过让五个前沿大语言模型在机器学习实验训练前预测其影响力并与实际结果对比，衡量了这些模型判断实验是否值得运行的直觉准确性。

    

    人工智能的最新进展使得递归自我改进（RSI）越来越具有可行性。大语言模型（LLM）已经能够提出、选择并运行机器学习（ML）实验。然而，机器学习中可能的设计选择空间非常庞大。提出实验的LLM需要具备可靠的直觉，以便在长训练时间的限制下判断某个实验是否值得运行。在这项工作中，我们通过评估当前前沿LLM在实验投入训练之前预测其自生成的机器学习实验想法影响力的准确程度，来衡量这种直觉。为此，我们构建了一个包含288个LLM提出的实验想法的数据集，涵盖八个MLE-bench任务。随后，我们提示五个前沿LLM（GPT-6 Astra、Claude Opus 5.5、Claude Fable 5.1、Kimi K3和GLM-5.3）预测每个想法的影响力，并将其预测与训练后的实际结果进行比较。我们报告了四个主要发现。首先，LLM的预测相当……

    arXiv:2610.22599v1 Announce Type: cross  Abstract: Recent advances in AI make Recursive Self-Improvement (RSI) increasingly plausible. Large Language Models (LLMs) can already propose, select, and run Machine Learning (ML) experiments. However, the space of possible design choices in ML is vast. An LLM that proposes an experiment needs a reliable intuition of whether it is worth running given the timing constraints of long training runs. In this work, we measure such intuition in current frontier LLMs by evaluating how accurately they predict the impact of self-generated ML experiment ideas before these are put into training runs. To this end, we construct a dataset of 288 LLM-proposed ideas on eight MLE-bench tasks. We then prompt five frontier LLMs (GPT-6 Astra, Claude Opus 5.5, Claude Fable 5.1, Kimi K3, and GLM-5.3) to predict the impact of each idea and compare their predictions with the actual outcomes after training. We report four main findings. First, LLM predictions are subst
    
[^181]: 基于逐层探测熵的图神经网络失效模式诊断

    Diagnosing GNN Failure Modes with Layer-Wise Probe Entropy

    [https://arxiv.org/abs/2610.22595](https://arxiv.org/abs/2610.22595)

    该论文提出一种冻结模型并逐层拟合线性探测器的探测熵诊断方法，能够区分有益、中性与有害的平滑，并通过熵退化的出现时机区分GNN的两种主要失效模式——异配性下的结构失配与深层过平滑。

    

    图神经网络（GNN）通常只在最终层进行评估，这掩盖了任务相关信息在消息传递过程中的演化方式。我们提出了一种事后诊断方法：冻结训练好的GNN，为每一层拟合一个独立的线性探测器，并同时跟踪探测器准确率与归一化预测熵。在真实与合成的图设置中，逐层探测熵能够区分有益的、中性的和有害的平滑，这是现有几何度量所不具备的能力。此外，通过比较浅层与深层模型，我们证明熵退化的出现时机可以区分GNN的两种主要失效模式：异配性下的结构失配，即使在浅层模型中也会从最初的几层就显现出来；以及层数过深导致的过平滑，只有随着传播深度的增加才会出现。这些结果确立了探测熵轨迹作为分析GNN中信息流动工具的地位，并激发（摘要原文在此处截断）。

    arXiv:2610.22595v1 Announce Type: new  Abstract: Graph neural networks (GNNs) are typically evaluated only at their final layer, obscuring how task-relevant information evolves during message passing. We introduce a post-hoc diagnostic that freezes a trained GNN, fits an independent linear probe to each layer, and tracks probe accuracy alongside normalized predictive entropy. Across real and synthetic graph settings, layer-wise probe entropy distinguishes beneficial, neutral, and harmful smoothing, a capability not demonstrated by existing geometric measures. Moreover, by comparing shallow and deep models, we show that the onset of entropy degradation separates the two main GNN failure modes: structural mismatch under heterophily, which appears from the earliest layers even in shallow models, and excessive-depth oversmoothing, which emerges only as propagation depth increases. These results establish probe-entropy trajectories as a tool for analyzing information flow in GNNs and motiva
    
[^182]: 平坦损失与演化特征：模加法中的顿悟现象

    Flat Loss, Evolving Features: Grokking in Modular Addition

    [https://arxiv.org/abs/2610.22593](https://arxiv.org/abs/2610.22593)

    该论文证明网络可以在训练交叉熵几乎平坦地保持正值的条件下继续学习：在模加法任务上，ReLU网络在保持100%训练准确率的同时隐藏特征持续演化、留出集准确率从零升至90%以上，并通过两个有限窗口定理从理论上证明了这种平坦损失与顿悟式泛化的共存性。

    

    当一个网络的训练交叉熵几乎平坦地停留在正的水平上时，它还能继续学习吗？我们证明它可以。在模加法这一grokking的标准测试平台上，使用supported-Polar隐藏层更新、归一化输出层更新和权重衰减训练的ReLU网络，可以在训练交叉熵相对平坦的情况下保持100%的训练准确率，同时其隐藏特征持续变化，留出集准确率从零上升到90%以上。正确分类并不需要最小化交叉熵，因为交叉熵仍然依赖于预测置信度。我们通过两个基于构造初始化的有限窗口定理，对单隐层网络证明了这种共存性：其中一个定理给出隐藏特征的傅里叶占比的恒定增益，另一个定理则在任意低的正损失下成立，此时在训练数据上重新拟合的读出层会先于网络本身实现泛化。从随机初始化出发的实验在窗口上同样显示了这种共存现象。

    arXiv:2610.22593v1 Announce Type: new  Abstract: Can a network keep learning while its training cross-entropy stays nearly flat at a positive level? We show that it can. On modular addition, a standard testbed for grokking, ReLU networks trained with supported-Polar hidden updates, normalized output-layer updates and weight decay can keep 100% training accuracy while their hidden features change and held-out accuracy rises from zero to above 90%, at relatively flat training cross-entropy. Correct classification need not minimize cross-entropy, which still depends on prediction confidence. We prove this coexistence for one-hidden-layer networks in two finite-window theorems from constructed initializations: one gives a constant gain in the Fourier fraction of the hidden features, and the other holds at arbitrarily low positive loss, where a readout refitted on the training data generalizes before the network. Experiments from random initialization show the same coexistence on a window f
    
[^183]: 联邦车联网中由运行症状触发的遗忘移除

    Initiating Removal from Operational Symptoms in Federated Vehicular Networks

    [https://arxiv.org/abs/2610.22584](https://arxiv.org/abs/2610.22584)

    本文提出ATONE方法，将联邦遗忘嵌入联邦学习聚合规则中，无需移除请求指明具体车辆或轮次，即可由车辆运行症状触发，通过将症状投影到最近全局增量张成的子空间来评分并中和错误贡献的有害效应。

    

    网联车辆通过联邦学习（FL）训练驾驶模型并用其驾驶，因此错误的贡献往往只在道路上以延迟出现的症状形式暴露出来。联邦遗忘（FU）通常假设移除请求会明确指明要遗忘的内容，而运行症状既不指明具体的车辆也不指明具体的训练轮次，且聚合过程已经将其成因扩散到整个车队中。为此，我们提出聚合时运行证据效应中和方法（ATONE），将遗忘变为联邦学习聚合规则中的一项。该方法将日常驾驶错误和上报的症状投影到由最近全局增量张成的子空间上，利用从这些增量中估计的曲率传播每个贡献的残留效应，并在一次反向传播中为所有贡献评分。一个双层校正机制在每一轮中抑制疑似效应，并在一轮探测后消除已确认的效应。

    arXiv:2610.22584v1 Announce Type: new  Abstract: Connected vehicles train a driving model through federated learning (FL) and drive with it, so a faulty contribution often surfaces only as a delayed symptom on the road. Federated unlearning (FU) assumes that a removal request names what to forget. A symptom names no vehicle or round, and aggregation has already spread its cause across the fleet. We propose \underline{A}ggregation-\underline{T}ime \underline{O}perational-evidence \underline{N}eutralization of \underline{E}ffects (\method{}), which makes unlearning a term of the FL aggregation rule. \method{} projects everyday driving errors and reported symptoms onto a subspace spanned by recent global increments. It propagates the surviving effect of every contribution with curvature estimated from these increments and scores all contributions in one backward pass. A two-tier correction suppresses suspected effects in every round and eliminates confirmed ones after a one-round probe. W
    
[^184]: 面向自发语音的抗噪声强制对齐

    Noise-Robust Forced Alignment for Spontaneous Speech

    [https://arxiv.org/abs/2610.22583](https://arxiv.org/abs/2610.22583)

    本文提出一种面向自发嘈杂语音的抗噪声强制对齐方法，通过上采样WavLM特征、为填充词和非词汇声音事件设置显式音素单元以及判别投影，将词边界误差较最强基线降低17%-19%，灾难性错位比MFA最佳模型减少2.7倍。

    

    我们证明强制对齐器在自发、嘈杂的真实世界语音上性能会退化，填充词、笑声和不流畅现象会破坏词边界，影响许多下游应用。我们重新审视了经典的GMM-HMM对齐器，采用上采样的WavLM特征代替MFCC，使用在逐字标注的自发语音上训练的、针对填充词和非词汇声音事件的显式音素单元，并通过在干净与噪声污染特征对上拟合的判别投影来提高鲁棒性。该流程从149小时的精选音频中自举训练，无需人工或外部边界目标。在会话语音（Buckeye）和不流畅语音（FluencyBank）上，其词边界平均绝对误差（MAE）分别达到20.6和76.9毫秒，比十个基线中最强者的结果分别低19%和17%。即使在扩大MFA搜索束之后（我们表明这会消除一个较大的剪枝伪影），其灾难性错位（>200毫秒）也比蒙特利尔强制对齐器（MFA）最佳模型少2.7倍。

    arXiv:2610.22583v1 Announce Type: new  Abstract: We show that forced aligners degrade on spontaneous, noisy real-world speech, and that fillers, laughter, and disfluencies corrupt word boundaries, impacting many downstream applications. We revisit the classical GMM-HMM aligner with upsampled WavLM features instead of MFCCs, explicit phone units for fillers and non-lexical vocal events trained on verbatim-annotated spontaneous speech, and a discriminative projection fitted on clean and noise-corrupted feature pairs to improve robustness. The pipeline is self-bootstrapped from 149h of curated audio without manual or external boundary targets. On conversational (Buckeye) and disfluent (FluencyBank) speech it reaches word-boundary MAEs of 20.6 and 76.9ms, 19% and 17% lower than the strongest of ten baselines. Catastrophic misalignments (>200ms) are 2.7 times rarer than Montreal Forced Aligners (MFA) best model even after widening MFA's search beam, which we show removes a large pruning art
    
[^185]: 本体分类器引导：无需演示数据即可将冻结的视觉-语言-动作模型迁移至不同机器人动力学

    Embodiment-Classifier Guidance: Transferring Frozen VLAs Across Robot Dynamics without Demonstrations

    [https://arxiv.org/abs/2610.22576](https://arxiv.org/abs/2610.22576)

    该论文提出“本体分类器引导”方法，通过轻量级分类器的梯度在去噪过程中引导冻结的π0流策略，无需演示数据或参数更新即可将其迁移至新的机器人本体，在LIBERO基准上使目标本体平均成功率从72%提升至80%。

    

    诸如 $\pi_0$ 这类基于流（flow-based）的策略是在固定的本体集合上训练的，将其迁移到新的本体上仍是一个悬而未决的问题。为了在不使用额外演示数据或参数更新的情况下，将冻结的 $\pi_0$ 适配到新的本体上，我们利用少量校准任务训练一个轻量级的本体分类器，用以区分源本体上成功的动作块与目标本体上的动作块。对于 $\pi_0$ 在源本体上见过但分类器从未训练过的迁移任务，我们利用分类器的梯度来微调 $\pi_0$ 的每一步去噪过程，该梯度估计了从源本体成功动作分布指向目标本体成功动作分布的方向。在使用四种修改后控制器的LIBERO仿真基准上，本体分类器引导将 $\pi_0$ 在目标本体上的平均成功率从72%提升至80%（跨12个套件-控制器组合），失败率相对降低了31%（12组中的11组均有提升）。

    arXiv:2610.22576v1 Announce Type: cross  Abstract: Flow-based policies such as $\pi_0$ are trained on a fixed set of embodiments, and transferring them to new ones is an open problem. To adapt a frozen $\pi_0$ to a new embodiment without additional demonstrations or parameter updates, we train a lightweight embodiment classifier to distinguish successful action chunks on the source body from those on the target body, using a few calibration tasks. On transfer tasks that $\pi_0$ saw on its source body but the classifier was never trained on, we nudge each denoising step of $\pi_0$ with the classifier's gradient, which estimates the direction from the distribution of successful actions on the source body toward that on the target body. On the LIBERO simulation benchmark with four modified controllers, embodiment-classifier guidance raises $\pi_0$'s average success on the target bodies from 72% to 80% across 12 suite-controller pairs, with a 31% relative reduction in failures (11 of 12 ga
    
[^186]: 面向高斯过程分布式端口哈密顿系统的结构感知空间核

    Structure-Aware Spatial Kernels for Gaussian Process Distributed Port-Hamiltonian Systems

    [https://arxiv.org/abs/2610.22575](https://arxiv.org/abs/2610.22575)

    提出一种结构感知的核设计，通过张量积协方差将状态与空间依赖解耦，使高斯过程分布式端口哈密顿系统能够高效建模空间异质系统并融入定性空间知识。

    

    高斯过程分布式端口哈密顿系统（GP-dPHS）能够在保持物理结构的同时，实现对分布式动力系统的数据驱动建模。然而，标准的平稳核在处理空间异质系统时可能效率低下，因为它们既无法利用已知的输入不变性，也无法反映结构化的空间依赖关系。我们通过一种结构感知的核设计来解决这一局限：该设计去除了不相关的状态坐标，并通过张量积协方差将状态依赖与空间依赖解耦。这种模块化构建方式使得可以通过选择合适的空间核来融入关于空间异质性的定性知识，而无需对底层剖面进行解析描述。在具有周期性、平滑变化和半分段刚度剖面的非线性弦上进行的数值实验表明，结构解耦能够持续改善重建精度……

    arXiv:2610.22575v1 Announce Type: new  Abstract: Gaussian process distributed port-Hamiltonian systems (GP-dPHS) enable data-driven modeling of distributed dynamical systems while preserving their physical structure. However, standard stationary kernels can be inefficient for spatially heterogeneous systems because they neither exploit known input invariances nor reflect structured spatial dependencies. We address this limitation with a structure-aware kernel design that removes irrelevant state coordinates and decouples state and spatial dependencies through a tensor-product covariance. This modular construction allows qualitative knowledge of spatial heterogeneity to be incorporated by selecting an appropriate spatial kernel without requiring an analytical description of the underlying profile. Numerical experiments on a nonlinear string with periodic, smoothly varying, and semi-piecewise stiffness profiles demonstrate that structural decoupling consistently improves reconstruction a
    
[^187]: 离策略分位数认证中的覆盖悬崖

    Coverage Cliffs in Off-Policy Quantile Certification

    [https://arxiv.org/abs/2610.22565](https://arxiv.org/abs/2610.22565)

    本文提出 TailCert 程序，通过构建 PAC-Bayes Bernstein 下置信包络，利用已记录的上下文老虎机数据安全地为离策略选择认证上尾分位数，并揭示了在稀有动作类上所需样本量随重叠因子线性增长、随策略搜索与置信度参数对数增长的定量规律。

    

    我们研究了在什么条件下已记录的上下文老虎机数据能够为从同一样本中选出的策略认证一个上尾分位数。我们为目标累积分布函数（CDF）构建了一个 PAC-Bayes Bernstein 下置信包络，该包络在预先声明的阈值网格和所有依赖数据的后验分布上同时有效。由此得到的程序 TailCert 利用该包络首次穿越水平 p 的位置，返回所选后验的池化 p 分位数的一个原子安全的上置信界（或返回无法判定）。该程序还可以在经过认证的尾部约束下用于策略选择。其中的集中不等式成分都是标准工具；主要的理论贡献是一个匹配的有限类认证边界，它在固定阈值下比较完整日志数据与其联合加权 CDF 指示变量。在一个显式的稀有动作类上，所需样本量随重叠因子 W 线性增长，随策略搜索和置信度参数对数增长……

    arXiv:2610.22565v1 Announce Type: new  Abstract: We study when logged contextual-bandit data can certify an upper tail quantile for a policy selected from the same sample. We construct a PAC-Bayes Bernstein lower-confidence envelope for the target cumulative distribution function (CDF), simultaneous over a predeclared threshold grid and all data-dependent posteriors. The resulting procedure, TailCert, uses the envelope's first crossing of level p to return an atom-safe upper confidence bound on the selected posterior's pooled p-quantile (or inconclusive). It can also be used to select a policy under a certified tail constraint. The concentration ingredients are standard; the main theoretical contribution is a matched finite-class certification boundary comparing full logs with their joint weighted CDF indicators at a fixed threshold. On an explicit rare-action class, the required sample size grows linearly with the overlap factor W, logarithmically with policy-search and confidence com
    
[^188]: 度量空间回归与分类的精度与稳定性的确定性界限

    Deterministic Bounds on Accuracy and Stability of Metric-Space Regression and Classification

    [https://arxiv.org/abs/2610.22564](https://arxiv.org/abs/2610.22564)

    该论文提出了一个几何框架，通过基于最坏情况输入和目标噪声计算的平均核大小，为度量空间回归和分类任务提供了与模型无关的、可计算的确定性精度与误差界限。

    

    数据驱动方法常常由于传感器或目标噪声、信息不足以及输入数据中的噪声而面临数据集中的模糊性问题。虽然最优拟合机器学习方法可以通过不确定性估计来容纳这种噪声，但它们容易过拟合，且难以量化输入与目标之间关系的模糊性。我们引入了一个几何框架，通过平均核大小来量化这种模糊性，该框架使用基于数据的（即与模型无关的）可行集进行计算，这些可行集由最坏情况下的输入和目标噪声导出。该框架为有限构建数据集上可实现的最低经验误差和最高精度提供了可计算的、确定性的、非渐近的界限，且与预测模型类无关，适用于具有伪度量输入空间和度量目标空间的回归与分类问题。输入-目标关系的模糊性意味着三个后果：……（摘要在此处截断）

    arXiv:2610.22564v1 Announce Type: new  Abstract: Data-driven approaches often face ambiguity in datasets due to sensor or target noise, insufficient information, and noise in the input data. While best-fit machine learning methods can accommodate such noise through uncertainty estimates, they are prone to overfitting and it is challenging to quantify ambiguity of the relation between inputs and targets. We introduce a geometric framework that quantifies this ambiguity through the average kernel size, computed with data-dependent, i.e. model-independent, feasible sets derived from worst-case input and target noise. The framework provides computable, deterministic, non-asymptotic bounds on the best achievable empirical error and accuracy for a finite, constructed dataset, independent of the predictive model class, and applies to regression and classification with pseudo-metric input spaces and metric target spaces. The ambiguity of the input-target relation implies three consequences: an
    
[^189]: 当计划改变时，请记住：面向视觉-语言-动作模型的基于重规划引导的视觉记忆

    When Plans Change, Remember: Replanning-Guided Visual Memory for Vision-Language-Action Models

    [https://arxiv.org/abs/2610.22556](https://arxiv.org/abs/2610.22556)

    提出RePlanMem-VLA，将VLA策略自身的重规划行为（即连续动作块预测之间出现的异常巨大分歧）作为在线写入视觉记忆的信号，以选择性保存关键帧，解决长时程任务中关键视觉信息已不在当前观测中的问题。

    

    长时程、多阶段的视觉-语言-动作（VLA）策略常常需要当前观测中已不存在的视觉信息。保留完整的观测历史在计算上代价高昂且高度冗余，因此需要选择性的视觉记忆。我们提出了 RePlanMem-VLA（面向 VLA 的重规划引导关键帧记忆），它将策略自身的计划修订转化为在线记忆写入信号。在滚动时域执行下，连续的动作块会预测部分相同的未来动作，RePlanMem-VLA 将这些时间对齐的预测之间异常巨大的分歧视为发生任务相关事件的证据。我们首先在从头训练的单任务自回归仅解码器模型和流匹配Transformer上比较了 35 种分歧度量方法，随后在 MIKASA-Robo-VLA 任务上使用预训练的 OpenVLA-OFT 和 π0.5 策略对所选信号进行了验证。所得的检测器实现了……（原文摘要在此处截断）

    arXiv:2610.22556v1 Announce Type: cross  Abstract: Long-horizon, multi-stage Vision-Language-Action (VLA) policies often require visual information that is no longer present in the current observation. Retaining the complete observation history is computationally expensive and highly redundant, motivating selective visual memory. We introduce RePlanMem-VLA (Replanning-Guided Keyframe Memory for VLA), which turns a policy's own plan revisions into an online memory-write signal. Under receding-horizon execution, consecutive action chunks predict some of the same future actions. RePlanMem-VLA treats unusually large disagreement between these temporally aligned predictions as evidence of a task-relevant event. We first compare 35 disagreement measures across single-task autoregressive decoder-only and Flow-Matching Transformers trained from scratch, then validate the selected signals on pretrained OpenVLA-OFT and $\pi_{0.5}$ policies over MIKASA-Robo-VLA tasks. The resulting detectors achi
    
[^190]: On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction

    On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction

    [https://arxiv.org/abs/2610.22543](https://arxiv.org/abs/2610.22543)

    The paper proposes SPOT, which achieves lossless accelerated decoding for diffusion language models by learning to predict the commit order of positions: a 0.62M-parameter small model reads the target model's output distributions to propose a candidate token tree, and the target model verifies all candidates in a single pass.

    

    Diffusion language models (DLMs) promise parallel text generation, but in practice face an unresolved speed-quality trade-off. Exact decoding fills in only one masked position per model pass, while faster multi-token decoding schemes are lossy. Speculative decoding is the standard lossless acceleration method for autoregressive models. However, it cannot directly transfer to DLMs, because DLMs must simultaneously decide what to generate and where to generate it. We show that this order is the missing piece: the model's own predictions from the previous pass already give the correct tokens at 92-100% of positions, but do not indicate which position comes next. Based on this observation, we introduce Speculating Positions Order Trajectory (SPOT), which learns to predict this order. A 0.62M-parameter small model reads only the target model's output distributions and proposes a tree of candidate continuation sequences, and the target model verifies all candidates in a single pass by checking whether each equals its own next prediction.

    arXiv:2610.22543v1 Announce Type: new  Abstract: Diffusion language models (DLMs) promise parallel text generation, but in practice face an unresolved speed-quality trade-off. Exact decoding fills in only one masked position per model pass, while faster multi-token decoding schemes are lossy. Speculative decoding is the standard lossless acceleration for autoregressive models. However, it does not transfer directly to DLMs, which must choose both what to generate and where. We show that this order is the missing piece: the model's own predictions from the previous pass already give the right token at 92-100% of positions, but not which position comes next. Based on this observation, we introduce Speculating Positions Order Trajectory (SPOT), which learns to predict this order. A 0.62M-parameter model reads only the target's output distributions and proposes a tree of candidate continuations, and the target verifies all of them in a single pass by checking whether each equals its own ne
    
[^191]: HeRE：通过大语言模型引导的知识图谱对齐去噪构建的希伯来语关系抽取新基准

    HeRE: A Novel Benchmark for Hebrew Relation Extraction via LLM-Guided Denoising of Knowledge Graph Alignments

    [https://arxiv.org/abs/2610.22542](https://arxiv.org/abs/2610.22542)

    本文提出HeRE——首个基于LLM引导去噪知识图谱对齐构建的希伯来语关系抽取基准，提供了一种可扩展、语言无关的低资源语言关系抽取数据集创建方法。

    

    尽管大语言模型在英语关系抽取任务中展现出强大的能力，但由于缺乏标准化基准，评估其在低资源语言中的关系抽取性能仍存在显著差距。创建原生的、针对特定语言标注的关系抽取基准成本高昂且需要人工专业知识，而依赖翻译则会产生次优、不准确的数据集，并引入以英语为中心的偏差。为解决这一问题，我们提出了一种可扩展的、语言无关的方法论，通过对知识库与相应文本之间自然存在但含噪的对齐进行去噪来创建关系抽取数据。我们通过自然语言推理（NLI）任务定义了恰当的对齐，即一个关系三元组需被输入文本所蕴含。我们对五种去噪策略进行了实证评估，并证明在少样本LLM设置下，多语言模型优于特定语言模型，而希伯来语专用编码器在NLI设置下仍具有竞争力。

    arXiv:2610.22542v1 Announce Type: new  Abstract: While LLMs demonstrate strong capabilities in English Relation Extraction (RE) tasks, a significant gap remains in assessing their RE performance in low-resource languages, due to a lack of standardized benchmarks. Creating native, language-specific annotated RE benchmarks is expensive and requires human expertise, while relying on translation yields suboptimal, inaccurate datasets that introduce English-centric biases. To tackle this, we propose a scalable, language-agnostic methodology to create RE data by denoising naturally occurring yet noisy alignments of knowledge bases and corresponding texts. We define a proper alignment via a Natural Language Inference (NLI) task, where a relation triplet is entailed by an input text. We empirically assess five denoising strategies, and demonstrate that multilingual models outperform language-specific models in the few-shot LLM setting, while Hebrew-specific encoders remain competitive under NL
    
[^192]: 无需重新训练，只需重新组合：利用混合超网络适配网页智能体

    Don't Retrain, Remix: Adaptation of Web Agents using Mixture Hypernetworks

    [https://arxiv.org/abs/2610.22539](https://arxiv.org/abs/2610.22539)

    WebMix 提出混合超网络方法，通过侦察智能体生成界面手册并由轻量级路由器组合预训练的低秩适配器，使网页智能体仅需一次侦察和一次前向传播（无需任何梯度更新）即可适应未见过的网页界面。

    

    互联网并非静态或统一的。服务于相同目的的网站在外观和导航方式上可能差异巨大。即使在同一网站内部， successive redesigns 也会改变熟悉功能的呈现和访问方式。人类通过识别熟悉的交互模式并复用已有知识来适应这种变化。我们通过 WebMix 赋予网页智能体类似的能力，这是一个混合超网络，通过组合先前训练好的低秩适配器使智能体适应未见过的界面。WebMix 从一个在源界面上训练的适配器库开始。在部署时，一个侦察智能体探索目标界面，并将其观察结果总结为一份紧凑的手册。一个轻量级路由器使用该手册预测适配器库上的混合系数，将现有适配器组合成一个新的适配器用于推理。适应新界面只需一次侦察运行和一次前向传播，无需在目标上进行梯度更新。基于 Qwen 和（摘要在此处截断）……

    arXiv:2610.22539v1 Announce Type: new  Abstract: The web is not static or uniform. Websites that serve the same purpose can look and navigate very differently. Even within a single site, successive redesigns can change how familiar features are presented and accessed. Humans adapt to this by recognizing familiar interaction patterns and reusing what they already know. We give web agents a similar ability with WebMix, a mixture hypernetwork that adapts them to unseen interfaces by composing previously trained low-rank adapters. WebMix begins with a bank of adapters trained on source interfaces. At deployment, a scout agent explores the target interface and summarizes its observations into a compact manual. A lightweight router uses this manual to predict mixing coefficients over the adapter bank, which combine the existing adapters into a new one for inference. Adapting to a new interface takes only a scouting run and a forward pass, without gradient updates on the target. With Qwen and
    
[^193]: 学习冷冻电镜逆问题的共享潜在结构

    Learning the Shared Latent Structure of Cryo-EM Inverse Problems

    [https://arxiv.org/abs/2610.22523](https://arxiv.org/abs/2610.22523)

    该论文提出通过结合全局目标与掩码图像块目标的互补自蒸馏方法，学习冷冻电镜各逆问题（姿态估计、异质性分析和重构）之间共享的潜在结构表征，以统一的表示提升多个冷冻电镜分析任务。

    

    理解生物分子的功能需要表征其三维结构和构象变化，而冷冻电镜（cryo-EM）是实现这一目标的有力工具。冷冻电镜产生含噪声的二维颗粒图像，其中分子结构、颗粒姿态和成像条件被共同编码其中。从这些观测数据中恢复生物信息涉及多个逆问题，包括姿态估计、异质性分析和三维重构。尽管这些问题共享相同的蛋白质结构和图像成像过程，但它们通常被独立处理，这激发了研究跨逆问题的通用表征的需求。由于具有生物学意义的差异可能在空间上局部化，我们研究了采用互补的全局目标与掩码图像块目标的自蒸馏方法。此外，由于在实验数据集上进行区分可能反映的是采集过程特有的线索，我们开发了受控的合成数据与真实实验数据的评估方法……

    arXiv:2610.22523v1 Announce Type: cross  Abstract: Understanding biomolecular function requires characterizing three-dimensional structure and conformational variability, for which cryogenic electron microscopy (cryo-EM) is a powerful tool. Cryo-EM produces noisy two-dimensional particle images in which molecular structure, particle pose, and imaging conditions are jointly encoded. Recovering biological information from these observations spans inverse problems including pose estimation, heterogeneity analysis, and reconstruction. These tasks are usually addressed independently despite shared protein structure and image formation, motivating a common representation across inverse problems. Since biologically meaningful differences may be spatially localized, we study self-distillation with complementary global and masked patch objectives. Because experimental dataset discrimination can reflect acquisition-specific cues, we develop controlled synthetic and experimental evaluations and f
    
[^194]: 基于运动学插值子学习光滑动力学

    Learning smooth dynamics via Kinematic Interpolants

    [https://arxiv.org/abs/2610.22519](https://arxiv.org/abs/2610.22519)

    该论文提出运动学插值子（KI）框架，将随机插值方法扩展至包含位置与速度测量的相空间多快照数据，从而无需模拟即可可扩展地高保真重建潜在的非线性动力学。

    

    从真实世界的实验数据中推断动力学本质上是模糊的，因为许多合理的动力系统都可以解释在测量时刻所获得的相同位置观测数据。为了对从单细胞RNA到涡旋动力学乃至鸟群集群等各类应用进行更高保真度的建模，科学观测量日益包含有助于解决这一歧义问题的速度估计信息。然而，如何以可扩展的方式、且无需模拟或不必要的近似来融入这些丰富的信息，一直是一个困难的挑战。在本文中，我们提出了运动学插值子，这是一个新颖的框架，它将开创性的随机插值方法加以扩展，以在相空间的多个快照（即同时包含位置和速度测量的观测量）之间推断非线性动力学。在实验上，我们证明在多个科学数据集上，KI 实现了对潜在动力学的卓越重建。

    arXiv:2610.22519v1 Announce Type: new  Abstract: Inferring dynamics from real-world experimental data is inherently ambiguous, as many plausible dynamical systems can explain the same positional observations at measurement times. For higher fidelity modeling of applications ranging from single-cell RNA to vortex dynamics and even flocking of birds, scientific observables increasingly also include velocity estimates that serve to resolve this issue. However, incorporating this rich information in a scalable manner, free of simulation or unnecessary approximations, has remained a difficult challenge. In this paper, we introduce Kinematic Interpolants (KI), a novel framework that extends the seminal methodology of stochastic interpolants to infer nonlinear dynamics across multiple snapshots in phase space, i.e., observables with position and velocity measurements. Empirically, we demonstrate that across several scientific datasets, KI achieves superior reconstruction of the underlying dyn
    
[^195]: 面向策略性买家的来源感知动态定价

    Source-Aware Dynamic Pricing with Strategic Buyers

    [https://arxiv.org/abs/2610.22516](https://arxiv.org/abs/2610.22516)

    本文提出了SALE定价策略，利用已验证特征进行局部化价格探索并结合自报特征校正策略性操纵，从而解决高维半参数环境下存在策略性买家操纵的来源感知动态定价问题。

    

    在现代商业中，企业和平台通常使用来自不同来源的客户特征来进行个性化定价：一些特征通过卖家记录或第三方数据进行验证，而另一些特征则由买家自行报告，可能被寻求更低价格的买家进行策略性操纵。因此，我们提出了一个新颖的来源感知框架，用于高维半参数设置下具有策略性买家的上下文动态定价，其中买家估值遵循具有未知噪声分布的稀疏线性模型。为了考虑不同特征来源在可操纵性上的差异，同时实现有效的估值学习，我们开发了来源感知的局部化探索与校准利用定价策略，该策略利用已验证的特征来局部化价格探索，并结合已验证特征与自行报告特征在利用阶段校正策略性操纵。我们的局部化探索设计允许……

    arXiv:2610.22516v1 Announce Type: cross  Abstract: In modern commerce, firms and platforms typically use customer features from different sources to personalize prices: some features are verified using seller records or third-party data, while others are self-reported and may be strategically manipulated by buyers seeking lower prices. We therefore introduce a novel source-aware framework for contextual dynamic pricing with strategic buyers in a high-dimensional semiparametric setting, where buyer valuations follow a sparse linear model with an unknown noise distribution. To account for differences in manipulability across feature sources while enabling effective valuation learning, we develop the Source-Aware Localized exploration and calibrated Exploitation (SALE) pricing policy, which draws on verified features to localize price exploration and combines verified and reported features to correct for strategic manipulation during exploitation. Our localized exploration design allows c
    
[^196]: 基于三维基础模型重建地理多样化的建筑

    Towards Reconstructing Geographically Diverse Architecture with 3D Foundation Models

    [https://arxiv.org/abs/2610.22514](https://arxiv.org/abs/2610.22514)

    该论文提出了高地理覆盖率的三维建筑地标基准ArchWorld，并通过误差分析发现三维基础模型的性能存在地理差异，其成因不仅包括训练数据代表性差异，还包括场景规模、区域特有建筑特征和背景干扰因素的影响。

    

    互联网规模的真实世界地标照片集推动了三维计算机视觉的进步，但对于在单次前向传递中估计场景结构的现代三维基础模型（3DFMs）而言，这些照片集仍然极具挑战性。在本工作中，我们提出了ArchWorld，一个具有高地理覆盖率和丰富元数据的三维建筑地标基准数据集。我们利用该基准对当前的三维基础模型进行了深入的误差分析，以了解它们在哪些方面存在不足以及如何改进。我们考察了模型误差中与社会相关的差异，发现模型性能因地理区域而异；然而，与典型的公平性研究不同，这些趋势仅部分符合训练数据的代表性分布。我们的调查揭示，观察到的地理差异不仅反映了数据代表性和场景规模的不同，还反映了区域特有建筑特征和背景干扰因素的差异。

    arXiv:2610.22514v1 Announce Type: cross  Abstract: Internet-scale photo collections of real-world landmarks have driven progress in 3D computer vision yet remain highly challenging for modern 3D foundation models (3DFMs) that estimate scene structure in a single feed-forward pass. In this work, we introduce ArchWorld, a benchmark of 3D architectural landmarks with high geographic coverage and rich metadata. We use this benchmark to conduct a thorough error analysis of current 3DFMs to understand where they fall short and how they can be improved. We examine socially relevant disparities in model error and find that model performance varies by geographic region; however, unlike typical fairness regimes, the trends only partially match training data representation. Our investigation reveals that the observed geographic disparities reflect not only differences in data representation and scene size but also differences in region-specific architectural features and background distractors th
    
[^197]: 预测在线策略自蒸馏中特权信息的有效性

    Predicting Privileged Information Effectiveness for On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.22512](https://arxiv.org/abs/2610.22512)

    该论文系统评估了16种特权信息配置在多轮智能体任务在线策略自蒸馏中的效果，发现其有效性取决于内容、生成来源和学生模型，并提出可利用基于KL散度的指标在训练前预测特权信息的效用。

    

    在线策略自蒸馏（OPSD）使大型语言模型（LLM）能够在没有单独教师模型的情况下从特权信息（PI）中学习。然而，先前的工作大多将特权信息视为固定的设计选择，导致其内容和构建方式的影响尚不明确。这一挑战在多轮智能体环境中尤为重要，因为轨迹中包含动作、观察和推理，可以以多种方式转换为特权信息。我们在三个多轮智能体基准测试和三个在规模与推理模式上各不相同的学生模型上，系统地评估了16种特权信息配置。我们发现，特权信息的有效性取决于其内容、生成来源和学生模型。信息密集的轨迹以及由能力更强的生成器产生的特权信息通常带来更大的收益，而思考型模型的收益则不太一致。为了在训练前评估特权信息，我们将特权信息的效用与两个基于KL散度的指标进行了比较。

    arXiv:2610.22512v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) enables large language models (LLMs) to learn from privileged information (PI) without a separate teacher model. However, prior work largely treats PI as a fixed design choice, leaving the effects of its content and construction poorly understood. This challenge is particularly important in multi-turn agentic settings, where trajectories contain actions, observations, and reasoning that can be transformed into PI in many ways. We systematically evaluate 16 PI configurations across three multi-turn agentic benchmarks and three student models that vary in scale and reasoning mode. We find that PI effectiveness depends on its content, generation source, and the student model. Information-dense trajectories and PI produced by more capable generators generally yield larger gains, whereas thinking models benefit less consistently. To assess PI before training, we compare PI utility with two KL-based metrics: 
    
[^198]: 生成式价值分类器

    Generative Value Classifier

    [https://arxiv.org/abs/2610.22497](https://arxiv.org/abs/2610.22497)

    该论文提出生成式价值分类器，通过反向提问“在每种可能回报下动作出现的概率”，以回报条件策略作为评论家网络的生成组件，从而利用每一个动作作为监督信号来训练价值函数，在离线和在线强化学习任务上超越了领先的判别式方法。

    

    Q学习的前提是学习一个函数来预测状态-动作对的价值，其目标是利用该函数来指导策略改进。大多数方法使用标量或分布式的回报目标从零开始学习这个函数。这种判别式方法忽略了另一种可用于学习表征的监督信号来源：动作本身。与其直接预测状态-动作对的回报，我们提出一个反向问题：在每种可能的回报下，该动作出现的可能性有多大？这一视角提供了一个评论家网络，其生成式组件是一个以回报为条件的策略，该策略可以与行动者网络共享架构。训练该策略可以用每一个动作来监督评论家网络，而不仅仅是用回报来监督，同时均方误差损失将评论家网络的Q值锚定在自举目标上。我们在离线和在线强化学习任务上评估了我们的方法，并展示了相对于领先判别式方法的性能提升。

    arXiv:2610.22497v1 Announce Type: new  Abstract: Q-learning is premised on learning a function to predict the value of a state-action pair, with the goal of using such a function to guide policy improvement. Most approaches learn this function from scratch using scalar or distributional return targets. This discriminative approach leaves another source of supervision for learning representations unused: the actions themselves. Rather than directly predict the return of a state-action pair, we ask the reverse question: how likely is the action under each possible return? This perspective provides a critic whose generative component is a return-conditioned policy, which can share the actor's architecture. Training this policy supervises the critic with every action rather than with returns alone, and a squared-error loss anchors the critic's Q-value to the bootstrapped target. We evaluate our approach on both offline and online RL tasks and show gains over leading discriminative methods,
    
[^199]: 脉冲神经网络中因果片段数量的上界

    Upper bounds on the number of causal pieces in spiking neural networks

    [https://arxiv.org/abs/2610.22484](https://arxiv.org/abs/2610.22484)

    本文首次为前馈单脉冲神经网络中因果片段的数量推导出两个互补的上界——一个基于多重伯努利数、适用于广泛脉冲神经元模型的组合界，以及一个基于超平面排列的界，从而填补了脉冲神经网络因果片段数量极限研究的空白。

    

    单脉冲神经网络的输入空间可以分解为因果片段，并且对于某些脉冲神经网络，因果片段的数量已被证明可以提供逼近误差的下界。虽然这种分解适用于广泛的脉冲神经元模型，但脉冲神经网络所能实现的因果片段数量的极限仍然未知。在这项工作中，我们推导了前馈单脉冲神经网络中因果片段数量的两个互补上界。首先，我们建立了一个由多重伯努利数给出的组合界，该界适用于一大类脉冲神经元模型，其阶为 $e^{2(L-1)N\log N+\mathcal{O}(N)}$，其中 $L$ 为网络深度，$N$ 为最大层宽。其次，对于一组特定的神经元模型，我们推导了一个基于超平面排列的界。应用后一种框架，我们得到了阶为 $\mathcal{O}\!\left(N^{(4L...}\right)$ 的界（摘要在此处被截断）。

    arXiv:2610.22484v1 Announce Type: cross  Abstract: The input space of single-spike neural networks decomposes into causal pieces, and the number of causal pieces has been shown to provide a lower bound on the approximation error for some spiking neural networks. While this decomposition is applicable to a wide range of spiking neuron models, the limits of how many causal pieces can be realised by spiking neural networks are still unknown. In this work, we derive two complementary upper bounds on the number of causal pieces in feedforward single-spike neural networks. First, we establish a combinatorial bound given by the poly-Bernoulli numbers, which applies to a broad class of spiking neuron models. The bound is of order $e^{2(L-1)N\log N+\mathcal{O}(N)}$, with network depth $L$ and maximum layer width $N$. Second, for a specific set of neuron models, we derive a bound based on hyperplane arrangements. Applying the latter framework, we obtain bounds of order $\mathcal{O}\!\left(N^{(4L
    
[^200]: GPT-6-Astra 实现具身智能了吗？

    Has GPT-6-Astra Achieved Embodied Intelligence?

    [https://arxiv.org/abs/2610.22471](https://arxiv.org/abs/2610.22471)

    本报告通过在涵盖桌面操作、人形机器人、移动平台和灵巧手的 12 个代表性基准上进行大规模统一评估，证明 GPT-6-Astra 在机器人操作与导航任务中取得了前所未有的性能，其能力挑战了为单个任务和本体构建专用模型的传统范式，并提出了 ICL 泛化性、成本、物理状态验证、故障恢复和长时程进度保持五大未来研究方向。

    

    GPT-6-Astra 在机器人操作与导航方面展现出的能力水平，对为单个任务和单个机器人本体构建专用模型的传统范式提出了挑战。在本报告中，我们不仅探究 GPT-6-Astra 的能力究竟有多强，还探讨其能力对该领域意味着什么。首先，我们探究了独立运行的 GPT-6-Astra 的能力极限。我们在涵盖桌面操作、人形机器人、移动平台和灵巧手的 12 个代表性基准上进行了大规模评估。在统一协议下——该协议允许从训练集演示中进行上下文学习（ICL），并大幅增加执行步骤的数量——GPT-6-Astra 取得了前所未有的性能（图 1）。未来研究可以聚焦于改进：1）ICL 的泛化能力；2）成本与执行时间；3）物理状态验证；4）从不可逆故障中恢复的能力；以及 5）长时程任务中已获进展的保持。其次，我们研究……

    arXiv:2610.22471v1 Announce Type: cross  Abstract: GPT-6-Astra demonstrates a level of competence in robotic manipulation and navigation that challenges the conventional paradigm of building specialized models for individual tasks and embodiments. In this report, we ask not only how capable GPT-6-Astra is, but what its capabilities imply for the field. First, we probe the limits of standalone GPT-6-Astra. We conduct a large-scale evaluation across 12 representative benchmarks spanning tabletop manipulation, humanoids, mobile platforms, and dexterous hands. Under a unified protocol that allows in-context learning (ICL) from training-set demonstrations and substantially increases the number of execution steps, GPT-6-Astra achieves unprecedented performance (Fig. 1). Future research may focus on improving 1) the generality of ICL 2) cost and execution time, 3) physical-state verification, 4) recovery from irreversible failures, and 5) long-horizon preservation of progress. Second, we stud
    
[^201]: 良性递归自蒸馏可以引发渐进式失准

    Progressive Misalignment Can Emerge from Benign Recursive Self-Distillation

    [https://arxiv.org/abs/2610.22467](https://arxiv.org/abs/2610.22467)

    在递归自蒸馏过程中，即使只使用良性提示和模型自身的安全生成内容，安全模型也会随着迭代轮次不断累积失准，最终在多种有害查询上表现出渐进式失准现象。

    

    arXiv:2610.22467v1 公告类型：cross 摘要：在模型自身输出上训练模型作为提升AI能力的可能途径，已受到广泛关注。一种突出的方法是自蒸馏，即训练模型去匹配它在获得额外上下文（例如提示、反馈或指令）时本应产生的token级预测。然而，这种训练过程的安全性尚未被充分理解。在本文中，我们揭示了递归自蒸馏（即每个自蒸馏出的模型成为下一轮训练的教师模型）中一个令人惊讶的安全失效现象。我们观察到，当一个安全模型在自身（安全的）生成结果上进行递归自蒸馏，同时接收额外的非有害提示时，它在多种多样的有害查询上会变得越来越失准；我们将这一现象称为“渐进式失准”。我们在Qwen3和Phi-4模型家族上，通过SORRY-Bench和StrongREJECT基准展示了这种失效模式……

    arXiv:2610.22467v1 Announce Type: cross  Abstract: Training models on their own outputs has attracted widespread attention as a possible route to advance AI capabilities. One prominent approach is self-distillation, where a model is trained to match the token-level predictions it would have produced when given access to additional context, such as a cue, feedback, or instruction. However, the safety properties of this training procedure are poorly understood. In this paper, we present a surprising safety failure of recursive self-distillation, where each self-distilled model becomes the teacher for the next round of training. We observe that when a safe model is recursively self-distilled on its own (safe) generations while receiving an additional non-harmful cue, it becomes increasingly misaligned across a diverse set of harmful queries; we call this Progressive Misalignment. We show this failure mode in both the Qwen3 and Phi-4 model families on SORRY-Bench and StrongREJECT. Progress
    
[^202]: 预填充阶段专家削减在何处损害MoE模型：少数提示词Token承担了大部分短答案F1损失

    Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss

    [https://arxiv.org/abs/2610.22454](https://arxiv.org/abs/2610.22454)

    研究发现MoE模型在预填充阶段削减路由专家所造成的短答案F1损失，主要集中于少数关键提示词Token（如答案前的聊天模板Token和开头的指令Token），且损害主要在于答案形式而非答案内容，保持这些敏感位置的完整专家远比恢复超级专家或超大激活更有效。

    

    在预填充阶段让提示词Token通过更少的路由专家，是降低混合专家模型服务成本的一种简单方法，但这种做法究竟损害了什么尚不清楚。我们在四个MoE模型家族上，通过Token级、层级和注意力路径干预，在短答案问答任务中定位了这种损害。在专家数量减半的情况下，三个家族的F1损失主要来自“答案形式”：模型仍能找到答案，但不再以简短形式作答。在Qwen3-30B-A3B中，这种损失发生在答案之前的两个聊天模板Token上；在DeepSeek-V2-Lite中，损失发生在其指令所在的前8-32个提示词Token上。按答案似然计算，这些位置分别约占全部损害的一半和五分之一。专家削减对首个Token的注意力质量、范数和最大激活的改变不足4%，而恢复“超级专家”或超大激活最多只能挽回约四分之一的损失，远低于保持敏感位置启用全部专家的效果。

    arXiv:2610.22454v1 Announce Type: new  Abstract: Running prompt tokens through fewer routed experts during prefill is a simple way to make mixture-of-experts (MoE) serving cheaper, yet it is unclear what this damages. We locate the damage with token-, layer- and attention-path interventions on short-answer question answering in four MoE families. At half the experts the F1 loss of three families is mostly answer form: the model still finds the answer but stops answering briefly. In Qwen3-30B-A3B this loss is made at the two chat-template tokens before the answer, and in DeepSeek-V2-Lite at the first 8-32 prompt tokens, where its instruction sits; by answer likelihood these positions account for about half and a fifth of the damage. Reduction changes the first token's attention mass, norm and largest activation by less than 4%, and restoring super experts or massive activations recovers at most about a quarter of the loss, well below keeping the sensitive positions full. On real prompts
    
[^203]: Lean4AgenticRL：通过可验证的多层次过程奖励推进智能体强化学习

    Lean4AgenticRL: Advancing Agentic RL with Verifiable Multi-level Process Rewards

    [https://arxiv.org/abs/2610.22448](https://arxiv.org/abs/2610.22448)

    提出Lean4AgenticRL框架，首次将覆盖低层工具调用、中层工作流阶段和高层结果的统一多层次过程奖励锚定在Lean4形式化模式上，从而有效提升LLM开放式智能体强化学习的训练效果。

    

    提升大型语言模型（LLM）应对复杂软件工程（SWE）及通用终端使用问题的智能体能力是一项紧迫挑战。当前最先进的LLM通常经过大规模的智能体强化学习（RL）训练，但它们大多依赖高层的结果奖励，极少考虑低层的工具调用奖励和中层的工作流阶段奖励，导致训练效果不佳。为克服这一局限，我们提出Lean4AgenticRL，据我们所知，这是首个在开放式智能体强化学习中将多层次过程奖励锚定于统一Lean4形式化模式的框架。Lean4AgenticRL首先构建了一个针对特定问题的工作流系统，在执行过程中平衡智能体自身能力与可控性，为进一步验证奠定基础。随后，我们的工作流形式化与多层次奖励标注（原文摘要至此被截断）

    arXiv:2610.22448v1 Announce Type: new  Abstract: Enhancing the agentic capabilities of Large Language Models (LLMs) to tackle complex software engineering (SWE) and general terminal-usage problems is a pressing challenge. Current state-of-the-art LLMs often go through large-scale agentic Reinforcement Learning (RL). However, they mostly rely on high-level outcome rewards, with minimal consideration of low-level toolcall and mid-level workflow-stage rewards, which makes the training less effective. To overcome this limitation, we propose **Lean4AgenticRL**, which, to the best of our knowledge, is the first framework that grounds multi-level process rewards in the unified Lean4 formalization schema for open-ended agentic RL. **Lean4AgenticRL** first builds a *problem-specific workflow* system that balances the agent's own capability and controllability during execution, laying the foundation for further verification. Subsequently, our workflow formalization and multi-level reward annotat
    
[^204]: 面向非结构化神经算子的可学习样条

    Learnable Splines for Unstructured Neural Operators

    [https://arxiv.org/abs/2610.22446](https://arxiv.org/abs/2610.22446)

    提出 L-SplineFNO，通过将 FNO 主干与稀疏可学习的张量积 B 样条变换相结合，实现非结构化工程网格与潜在笛卡尔网格之间内存高效的转换，并在三维工程基准测试中取得优异的预测精度。

    

    神经算子以数值求解器的一小部分成本来近似偏微分方程的解算子。诸如傅里叶神经算子（FNO）等高效的基于网格的架构假设均匀的矩形网格，而工程仿真则使用大型、不规则且局部细化的网格。我们提出了 L-SplineFNO，这是一种内存高效的神经算子，它将 FNO 主干与稀疏、可学习的张量积 B 样条变换相结合，实现非结构化网格与潜在笛卡尔网格之间的转换。局部归一化散射避免了依赖于几何形状的解析样条变换求解过程，同时在构造上保持了紧支撑性、单位分解性以及样条平滑性。该神经算子以全局设计和运行参数为条件，学习从几何形状到场解的映射。在六个三维工程基准测试中，L-SplineFNO 在保持强大预测精度的同时（摘要在此处截断）

    arXiv:2610.22446v1 Announce Type: new  Abstract: Neural operators approximate solution operators of partial differential equations at a fraction of the cost of numerical solvers. Efficient grid-based architectures such as the Fourier neural operator (FNO) assume uniform rectangular grids, whereas engineering simulations use large, irregular, and locally refined meshes. We introduce L-SplineFNO, a memory-efficient neural operator that couples an FNO backbone with a sparse, learnable tensor-product B-spline transfer between unstructured meshes and a latent Cartesian grid. A local normalised scatter avoids the geometry-dependent solve of analytic spline transformation while preserving compact support, partition of unity, and spline smoothness by construction. The neural operator, conditioned on global design and operating parameters, learns the mapping of geometry to the field solution. Across six three-dimensional engineering benchmarks, L-SplineFNO achieves strong predictive accuracy wh
    
[^205]: 零控制是必要但非充分的：评估科学发现与可解释性中的接受防护机制

    Null Controls Are Necessary but Not Sufficient: Evaluating Acceptance Guards for Scientific Discovery and Interpretability

    [https://arxiv.org/abs/2610.22445](https://arxiv.org/abs/2610.22445)

    本文提出四种可执行变换操作与三重接受判据（零接受、正接受、推论一致性），证明零控制测试对自动科学发现而言必要但不充分——退化的零映射约化能通过96%混沌系统的闭合性检验，只有结合独立的观测量保持测试，才能可靠地区分真正的科学约化与退化假设。

    

    自动化科学发现系统会提出科学解释并测试哪些解释应被接受。但通过测试是否就能确立所声称的结论？我们开发了四种用于改变物理描述的可执行操作：识别冗余（L1）、对动力学进行粗粒化（L2）、迁移结构（L3），以及从守恒失衡推断缺失量（L4）。我们通过零接受（NAR）、正接受（PAR）和推论一致性（CAR）三类判据来评估这些操作以及三种可解释性设置。研究发现：一个将每个状态都映射为零的退化约化能够完美预测其自身的未来，尽管不保留任何动力学变化，它却在129个公开混沌系统中的96%上通过了闭合性测试；而一个独立的观测量保持测试则始终将其拒绝。在受控基准上，组合判据接受了全部24个真正的约化（24/24），并拒绝了全部72个零控制（72/72）。其他操作则能够区分对称性与冗余、验证结构迁移（原文在此处截断）。

    arXiv:2610.22445v1 Announce Type: new  Abstract: Automated discovery systems propose scientific explanations and test which to accept. Does passing a test establish the intended claim? We develop four executable operations for changing physical descriptions: identifying redundancies (L1), coarse-graining dynamics (L2), transferring structure (L3), and inferring missing quantities from conservation imbalances (L4). We evaluate these operations and three interpretability settings through null acceptance (NAR), positive acceptance (PAR), and consequence agreement (CAR).   A reduction mapping every state to zero predicts its own future perfectly. It passes closure on 96% of 129 public chaotic systems despite retaining no dynamical variation. An independent observable-preservation test rejects it throughout. On controlled benchmarks, the combined criteria accept 24/24 genuine reductions and reject 72/72 nulls. The other operations distinguish symmetries from redundancies, validate transfers
    
[^206]: 稀疏自编码器可以诱导局部几何以缓解灾难性遗忘

    Sparse Autoencoders Can Induce Local Geometry for Mitigating Catastrophic Forgetting

    [https://arxiv.org/abs/2610.22439](https://arxiv.org/abs/2610.22439)

    该论文提出SAGA方法，将每个任务边界训练的稀疏自编码器转换为对表示漂移的局部低秩几何约束，从而在保持可塑性的同时有效缓解持续学习中的灾难性遗忘。

    

    灾难性遗忘仍然是监督式持续学习中的一个挑战，因为适应新任务会导致表示漂移和先前学习信息的丢失。这引出了一个关键问题：在保持可塑性的同时，应该约束哪些表示变化。为此，我们提出了稀疏自编码器几何锚定方法，它将在每个任务边界训练的稀疏自编码器（SAE）转换为对表示漂移的局部、低秩约束。重放样本指定约束应用于何处，而编码器雅可比矩阵的主右奇异方向指定惩罚哪些变化。由此产生的几何结构依赖于输入，且在后续任务的训练中无需保留SAE即可维持。我们建立了该方法与潜在码匹配之间的局部关系，定义了其谱截断误差，并推导出冻结读出层漂移的条件界。通过结合重放和快照历史，该方法（摘要在此处截断）

    arXiv:2610.22439v1 Announce Type: cross  Abstract: Catastrophic forgetting remains a challenge in supervised continual learning, as adapting to new tasks causes representation drift and loss of previously learned information. This raises the question of which representational changes should be constrained while preserving plasticity. We therefore introduce Sparse Autoencoder Geometric Anchoring (SAGA), which converts a Sparse Autoencoder (SAE) trained at each task boundary into local, low-rank constraints on representation drift. Replay examples specify where the constraints apply, while leading right singular directions of the encoder Jacobian specify which changes are penalized. The resulting geometry is input dependent and can be retained without keeping the SAE for subsequent tasks' training. We establish its local relation to latent code matching, define its spectral truncation error, and derive a conditional bound on frozen-readout drift. With replay and snapshot history, the met
    
[^207]: 微调更新的秩度量需要一个噪声基线

    Rank Measures of Fine-Tuning Updates Need a Noise Baseline

    [https://arxiv.org/abs/2610.22438](https://arxiv.org/abs/2610.22438)

    研究表明，微调权重更新的秩度量必须以随机噪声为基线来解读——更新实际呈“少数强方向叠加长尾”的结构，且这种秩结构并非对齐训练的独有特征。

    

    关于微调对语言模型有何影响的研究，通常用权重更新 $\Delta W$ 的秩来进行概括。我们将通常使用的秩度量与相同形状随机矩阵上的相同度量进行比较，并与两个更严格的参照对象进行比较：预训练权重以及不含对齐内容的匹配微调。计算方向数量的度量显示，微调更新仅略低于噪声水平；而由相同奇异值计算出的稳定秩则显示更新远低于噪声水平。两者都是正确的：更新是在一个长尾之上叠加了少数几个强方向。该模式在十三个参数量高达 8B 的已发布模型对上均成立，且这并非对齐的标志：在训练方式固定的情况下，指令数据与维基百科或代码数据并无显著区别；无论 DPO 的偏好标签是真实的还是被随机交换一半（单模型、中等偏好强度），其给出的更新都相同；而乱序文本反而产生最集中的更新。

    arXiv:2610.22438v1 Announce Type: new  Abstract: Studies of what fine-tuning does to a language model often summarise the weight update $\Delta W$ by its rank. We compare the rank measures used for this against the same measures on a random matrix of the same shape, and against two harder references: the pretrained weights, and matched fine-tunes without alignment content. Measures that count directions put our fine-tuning updates only a little below noise; stable rank, computed from the same singular values, puts them far below. Both are right: an update has a few strong directions on top of a long tail. The pattern holds for thirteen released pairs up to 8B parameters, and it is not a mark of alignment: with training held fixed, instruction data does not stand out from Wikipedia or code, DPO gives the same update whether its preference labels are real or half swapped at random (one model, modest preference strength), and scrambled text gives the most concentrated update of all. Appli
    
[^208]: CF-NODE：面向领域偏移下测试时自适应的反事实空间节点干预方法，并在五个糖尿病视网膜病变数据集上进行评估

    CF-NODE: Counterfactual Spatial Node Intervention for Test-Time Adaptation Under Domain Shift, Evaluated Across Five Diabetic Retinopathy Datasets

    [https://arxiv.org/abs/2610.22437](https://arxiv.org/abs/2610.22437)

    提出CF-NODE反事实测试时自适应框架，通过对空间节点的受控干预评估预测稳定性，从而在五个糖尿病视网膜病变数据集上提升跨域分级的鲁棒性。

    

    基于眼底图像的糖尿病视网膜病变（DR）自动分级可以大幅扩展筛查能力，但在不同的相机、诊所和患者人群中，源域训练的模型性能会急剧下降。现有的测试时自适应方法依赖于熵最小化，或依赖于从图像批次和数据流中积累的目标域信息。这些机制可能会强化那些自信但错误的预测，并且在临床图像独立到达时会失效。我们提出了CF-NODE，一个用于跨域DR分级的新型反事实测试时自适应框架。CF-NODE不依赖预测置信度，也不估计目标分布，而是探究每个预测在对支持它的证据进行受控干预后如何响应。CF-NODE利用模型导出的注意力、特征幅值和节点集相似度对内部空间节点进行排序，逐步衰减排名最高的节点，并利用结果的稳定性……（原文摘要在此处截断）

    arXiv:2610.22437v1 Announce Type: new  Abstract: Automated fundus-based diabetic retinopathy (DR) grading can substantially expand screening capacity, but source-trained models degrade sharply across different cameras, clinics, and patient populations. Existing test-time adaptation methods depend on entropy minimization or target information accumulated from image batches and streams. These mechanisms can reinforce confidently incorrect predictions and break down when clinical images arrive independently. We introduce CF-NODE, a novel counterfactual test-time adaptation framework for cross- domain DR grading. Instead of trusting prediction confidence or estimating a target distribution, CF-NODE asks how each prediction responds under a controlled intervention on the evidence supporting it. CF-NODE ranks internal spatial nodes using model-derived attention, feature magnitude, and node-set similarity, progressively attenuates the highest-ranked nodes, and uses the stability of the result
    
[^209]: 基于蒙特卡洛Dropout不确定性量化的生成对抗网络用于结构健康监测：Hawk T1A飞机光纤布拉格光栅数据增强对比研究

    Generative adversarial networks with Monte Carlo dropout uncertainty quantification for structural health monitoring: Fibre Bragg grating data augmentation comparison in Hawk T1A aircraft

    [https://arxiv.org/abs/2610.22436](https://arxiv.org/abs/2610.22436)

    本文提出一种结合条件生成对抗网络与蒙特卡洛Dropout不确定性量化的数据增强框架，在Hawk T1A飞机全尺寸数据集上验证了其生成的光纤布拉格光栅响应信号具有物理真实性和统计一致性，可有效用于结构健康监测中的损伤分类。

    

    本文在航空航天结构中考察了一种新颖数据增强框架的信号生成能力。首先，该框架通过条件生成对抗网络训练并以蒙特卡洛Dropout不确定性量化作为贝叶斯近似，学习所测结构响应的类条件时域与频域特征，这一步骤实现了随机信号合成与不确定性量化。其次，将生成的结构响应与实验测量结果在时间、频率、不确定性、变异性和特征域上进行严格对比，以评估其物理真实性、统计一致性和分布保真度。最后，对合成数据进行评估，以确定其是否应被纳入有监督的损伤分类工作流程。该框架在BAE Systems Hawk T1A飞机全尺寸基准数据集上得到了验证。

    arXiv:2610.22436v1 Announce Type: new  Abstract: The signal generation capabilities of a novel data augmentation framework are examined in aerospace structures. First, the framework learns class-conditional temporal and spectral characteristics of the measured structural responses. It uses conditional generative adversarial network training and Monte Carlo dropout uncertainty quantification as a Bayesian approximation. This step enables stochastic signal synthesis and uncertainty quantification. Second, the generated responses are rigorously compared against experimental measurements across time, frequency, uncertainty, variability, and feature domains to assess physical realism, statistical consistency, and distributional fidelity. Finally, the synthetic data are evaluated to determine whether they should be incorporated into a supervised damage classification workflow. The framework is validated on the full-scale benchmark dataset from a BAE Systems Hawk T1A aircraft tested at the Un
    
[^210]: TRACE：面向通信高效分布式边缘推理的Transformer残差激活码本编码

    TRACE: Transformer Residual Activation Codebook Encoding for Communication-efficient Distributed Edge Inference

    [https://arxiv.org/abs/2610.22435](https://arxiv.org/abs/2610.22435)

    TRACE通过在每个Transformer层的注意力后残差处引入向量量化器，将交换的激活值编码为紧凑码本索引，使接收方通过查表直接恢复键/值表示，从而消除通信后串行计算的结构性瓶颈，实现通信高效的分布式边缘推理。

    

    跨共置边缘设备的分布式Transformer推理本质上受通信瓶颈制约，因为在商用边缘带宽下，序列并行分区之间交换激活值主导了推理延迟。现有的压缩方案虽然降低了这一成本，但仍存在结构性瓶颈：接收到的数据仍需经过键/值投影后才能进行注意力计算，导致计算被串行化在通信之后。本文提出了Transformer残差激活码本编码（Trace）框架来解决这些局限性。Trace在每个Transformer层的注意力后残差处放置一个向量量化器（VQ），并预计算以码本索引的键/值表示。在推理过程中，交换的激活值被编码为紧凑的码本索引，使接收方能够通过查表直接恢复键和值，而无需执行推导它们所需的下游计算。

    arXiv:2610.22435v1 Announce Type: cross  Abstract: Distributed Transformer inference across co-located edge devices is fundamentally communication-bound, as exchanging activations between sequence-parallel partitions dominates inference latency under commodity edge bandwidths. Existing compression schemes reduce this cost but retain a structural bottleneck: the received data must still undergo key/value projection before attention can proceed, serializing computation behind communication. This paper presents Transformer Residual Activation Codebook Encoding (Trace), a framework that addresses these limitations. Trace places a Vector Quantizer (VQ) at the post-attention residual of every Transformer layer and precomputes codebook-indexed key/value representations. During inference, exchanged activations are encoded into compact codebook indices, allowing receivers to directly recover keys and values through table lookup instead of executing the downstream computations required to derive
    
[^211]: BOLT：面向带时滞与资源耦合运输的柔性作业车间调度的多瓶颈下界引导学习

    BOLT: Multi-Bottleneck Bound-Guided Learning for Flexible Job-Shop Scheduling with Time Lags and Resource-Coupled Transport

    [https://arxiv.org/abs/2610.22433](https://arxiv.org/abs/2610.22433)

    BOLT通过融合链、机器容量和车队容量三种松弛的多瓶颈下界来引导强化学习，求解带时滞与资源耦合运输的柔性作业车间调度问题，无需求解器即可认证调度方案的次优性差距，并在运输密集场景中显著优于运输惩罚策略。

    

    当一支稀缺的车辆车队在工位之间搬运工件时，柔性作业车间调度问题变为资源耦合问题，因为调度方案决定了运输等待时间。BOLT（面向运输耦合调度的下界引导学习）通过链松弛、机器容量松弛和车队容量松弛构建一个下界。该下界在终止时等于完工时间，因此其每一步的变化构成一种奖励，其总和等于常数减去完工时间。由于该下界从不超过可达到的完工时间，每个候选方案带来的下界增量都是完工时间下限的已被证明的提升，策略将其解读为一种“价格”。根节点的下界无需求解器即可证明已完成调度方案的次优性差距。在以工业为背景的10模块预制建筑实例上，在所有装载运输占车队时间至少66%的测试场景中，显式运输耦合方法均优于运输惩罚策略，而在装载占比56%及以下的场景中则无优势。在五个零样本50模块和80模块……（原文摘要在此处截断）

    arXiv:2610.22433v1 Announce Type: new  Abstract: Flexible job-shop scheduling becomes resource-coupled when a scarce vehicle fleet moves jobs between stations, because the schedule sets the transport waits. BOLT (bound-oriented learning for transport-coupled scheduling) builds a lower bound from chain, machine-capacity, and fleet-capacity relaxations. The bound equals the makespan at termination, so its per-step changes form a reward that sums to a constant minus the makespan. Because it never exceeds the attainable makespan, each candidate's bound increase is a proven rise of the makespan floor, which the policy reads as a price. The root bound certifies a completed schedule's suboptimality gap without a solver. On industry-motivated 10-module prefabricated-construction instances, explicit transport coupling outperforms a transport-penalty policy in every tested cell where loaded moves occupy at least 66% of fleet time and in none at or below 56%. Across five zero-shot 50- and 80-modu
    
[^212]: 一种用于耦合代理地貌动力学建模的自适应时域图神经算子

    An Adaptive-Horizon Graph Neural Operator for coupled surrogate morphodynamic modelling

    [https://arxiv.org/abs/2610.22432](https://arxiv.org/abs/2610.22432)

    本文提出自适应时域图神经算子AH-GNO，直接在非结构网格上从水动力与地形状态学习预测河床高程增量，并能根据流动和河床演变自适应调整历史长度与更新间隔，从而实现高效的地貌动力学耦合代理建模。

    

    地貌动力学模拟需要水动力求解器、泥沙输移模块与河床更新之间的反复耦合。对于在大规模非结构网格上进行的长历时、多情景模拟而言，这种耦合在计算上代价高昂。本研究提出一种自适应时域图神经算子（AH-GNO），将河床演变表述为一个图算子学习问题，从近期的水动力和地形状态出发预测未来的河床高程增量。该模型直接在原始非结构网格上运行。该方法融合了面积加权图学习、历史状态选择和预报时域估计，使有效历史长度（K）和更新间隔（B）能够随流动与河床状态的演变而自适应变化。以Telemac2D/GAIA求解结果作为监督数值参考，该框架在官方Bump和Yen弯道基准算例以及一个现场尺度应用（Waiho河）上进行了评估。

    arXiv:2610.22432v1 Announce Type: cross  Abstract: Morphodynamic simulations require repeated coupling between hydrodynamic solvers, sediment transport modules, and bed updating. For long duration, multi-scenario simulations on large unstructured meshes, such coupling is computationally demanding. This study proposes an Adaptive-Horizon Graph Neural Operator (AH-GNO) that formulates bed evolution as a graph operator learning problem from recent hydrodynamic and topographic states to predict future bed elevation increments. The model operates directly on the native unstructured meshes. The approach integrates area-weighted graph learning, historical state selection, and forecast horizon estimation, allowing the effective history length (K) and update interval (B) to vary with the evolving flow and bed state. Using Telemac2D/GAIA solutions as supervised numerical references, the framework is evaluated on the official Bump and Yen bend benchmarks and on a field-scale application (Waiho Ri
    
[^213]: 面向通信与目标照射的GPU加速鲁棒MA-RIS预优化

    GPU-Accelerated Robust MA--RIS Pre-Optimization for Communication and Target Illumination

    [https://arxiv.org/abs/2610.22431](https://arxiv.org/abs/2610.22431)

    提出GPU加速的鲁棒预优化框架RAMP，通过联合优化慢时间尺度的可移动天线构型与RIS相位，生成兼顾最差用户速率与最差目标照射的可复用帕累托解集，免去对每个信道实现反复优化的需要。

    

    可移动天线（MA）与可重构智能表面（RIS）为通感一体化（ISAC）提供了强大的空间控制能力，但针对每次信道实现反复调整MA位置和RIS相位可能并不现实。本文提出了一种GPU加速的鲁棒预优化框架，联合选择慢时间尺度的MA几何构型与RIS相位，并输出一个可复用的帕累托解集，以平衡期望的最差用户速率与发射侧的最差目标照射。其中，照射目标衡量的是方向性发射功率，而非完整的回波检测或参数估计性能。所提出的“体制感知多目标预优化”方法（RAMP）保留了各体制特有的帕累托种群，利用帕累托存活反馈调节跨体制交配，并构建了最差体制鲁棒档案。数值结果表明，与所考虑的进化算法基准相比，该方法展现出具有竞争力的鲁棒帕累托解质量。（原文摘要末尾截断）

    arXiv:2610.22431v1 Announce Type: cross  Abstract: Movable antennas (MAs) and reconfigurable intelligent surfaces (RISs) offer strong spatial control for integrated sensing and communication (ISAC), but repeatedly adapting MA positions and RIS phases to each channel realization can be impractical. This letter develops a GPU-accelerated robust pre-optimization framework that jointly selects slow-timescale MA geometry and RIS phases and returns a reusable Pareto set balancing expected worst-user rate and transmit-side worst-target illumination. The illumination objective measures directional transmitted power rather than complete echo-detection or parameter-estimation performance. The proposed Regime-Aware Multiobjective Pre-optimization (RAMP) preserves regime-specific Pareto populations, regulates cross-regime mating using Pareto-survival feedback, and constructs a worst-regime robust archive. Numerical results demonstrate competitive robust Pareto quality against the considered evolut
    
[^214]: CGDT-CESN：面向多元时间序列预测的相关性引导与动力学训练聚类回声状态网络

    CGDT-CESN: Correlation-Guided and Dynamics-Trained Clustered Echo State Network for Multivariate Time-Series Prediction

    [https://arxiv.org/abs/2610.22430](https://arxiv.org/abs/2610.22430)

    该论文提出了一种相关性引导与动力学训练的聚类回声状态网络（CGDT-CESN），通过基于变量间相关性构建并训练聚类间连接，在保持计算效率的同时捕捉时变的跨变量依赖关系，显著提升了多元时间序列的预测精度。

    

    对多元时间序列的准确预测需要模型能够同时捕捉各变量自身的动态特性和变量间的交互作用。本研究提出了一种相关性引导与动力学训练的聚类回声状态网络（CGDT-CESN），该网络采用模块化储备池架构，其中每个聚类专门负责多元时间序列中的一个变量。研究提出了一种创新的储备池设计，聚类之间的连接由经验观测到的变量间相关性驱动，从而提供了一种数据驱动的耦合机制。除了常规的读出层训练之外，该模型还训练由相关性信息指导的聚类间连接权重，使储备池能够捕捉随时间变化的跨变量依赖关系，同时不牺牲计算效率，并进一步提高了预测精度。研究发现 CGDT-CESN 的预测性能优于……

    arXiv:2610.22430v1 Announce Type: new  Abstract: Accurate prediction of multivariate time series requires models that simultaneously capture variable-specific dynamics and cross-variable interactions. In this work, a Correlation-Guided and Dynamics-Trained Clustered Echo State Network (CGDT-CESN) is proposed, which comprises a modular reservoir architecture with each cluster distinctly specializing in one of the variables in the multivariate time series. An innovative reservoir design is proposed in which inter-cluster connections are enabled by empirically observed correlations among variables, providing a data-driven coupling mechanism. Apart from conventional readout training, inter-cluster connection weights informed by correlations are trained, which in turn allows the reservoir to capture time-varying cross-variable dependencies without sacrificing computational efficiency and further improves forecasting accuracy. The prediction performance of CGDT-CESN is found to be superior t
    
[^215]: 基于深度学习的印度安得拉邦戈达瓦里地区人工林制图

    Deep Learning for Forest Plantation Mapping in Godavari Districts of Andhra Pradesh, India

    [https://arxiv.org/abs/2610.22415](https://arxiv.org/abs/2610.22415)

    本文利用深度学习方法对印度安得拉邦东、西戈达瓦里地区的小面积人工林进行遥感制图，克服了人工林与周围植被及其他土地覆盖高度混合、难以用中等分辨率卫星影像检测的挑战。

    

    小面积人工林在东南亚农民的社会经济福祉中发挥着至关重要的作用。大多数人工林建立在曾经的农业用地上，且往往是不太适合农业耕作的土地。而由天然林转化而来的人工林则会对生物多样性造成不利影响。因此，对小面积人工林进行制图，对于理解东南亚森林覆盖的动态变化，以及研究这一重要土地覆盖和土地利用变化所带来的社会、经济和生态效应都具有重要意义。由于人工林面积较小，利用中等分辨率卫星传感器对其进行检测十分困难，而人工林与周围植被及其他土地覆盖类型之间的高度混合进一步加剧了这一难题，因为这些地物在卫星信号中往往会在空间和时间上表现出多样多变的响应。在本文中，我们研究了在东戈达瓦里和西戈达瓦里地区进行小面积人工林制图的问题。

    arXiv:2610.22415v1 Announce Type: cross  Abstract: Small-area forest plantations play a vital role in the socioeconomic well-being of farmers in Southeast Asia. Most plantations are established on former agricultural land, often on land less suitable for agriculture. Plantations that are converted from natural forest have adverse impacts on biodiversity. Mapping small-area plantations is thus important to understand the dynamics of forest cover in Southeast Asia and to study the social, economic, and ecological effects of this important land cover and land use change. While the small size of forest plantations makes it difficult to detect them using moderate resolution satellite sensors, the problem is exacerbated by the high degree of mixing between plantations, surrounding vegetation, and other land covers, which often show variegated responses in satellite signals across space and time. In this paper, we study the problem of mapping small-area forest plantations in East and West God
    
[^216]: 面向测试时持续学习的绑定特异性再学习节省效应

    Binding-Specific Relearning Savings for Test-Time Continual Learning

    [https://arxiv.org/abs/2610.22381](https://arxiv.org/abs/2610.22381)

    该研究发现在测试时持续学习中，模型习得的特定事实绑定（而非对实体、答案词汇、格式或查询槽位的熟悉性）能在部分遗忘后带来再学习加速，且错误绑定不产生此优势，表明现有仅依赖输出准确率的评估方式无法捕捉先前经验对后续更新的塑造作用。

    

    当一个事实从语言模型中变得更难被提取时，模型是真的丢失了这个事实，还是它仅仅变得不那么容易被访问到？参数更新的测试时持续学习智能体通常以当前输出准确率来评估；我们探究这种测量方式是否能捕捉到先前经验如何塑造后续的更新。我们检验学习一个特定的“事实绑定”是否会在之后产生再学习优势，且这种优势超越了对相关实体、答案词汇、格式或查询槽位的熟悉性。在Qwen2.5-1.5B模型上，我们比较了三种匹配且经过平衡处理的历史条件：LEARNED（学习组）接受正确的绑定，CONTROL（对照组）接受匹配的暴露但不包含该绑定，WRONG（错误组）在同一查询槽位接受一个竞争性的、关系上有效的答案。在完全相同的无回放持续微调数据流之后，在预设的部分遗忘检查点上，LEARNED组比CONTROL组更快地重新习得目标事实。WRONG组相比CONTROL组没有表现出正向优势，而它与LEA（原文在此处截断）

    arXiv:2610.22381v1 Announce Type: cross  Abstract: When a fact becomes harder to elicit from a language model, has the model lost it, or has it merely become less accessible? Parameter-updating test-time continual learning agents are usually evaluated by current output accuracy; we ask whether that measurement captures how prior experience shapes later updates. We test whether learning a specific factual binding produces a later relearning advantage beyond familiarity with its entity, answer vocabulary, format, or query slot. In Qwen2.5-1.5B, we compare three matched, counterbalanced histories: LEARNED receives the correct binding, CONTROL receives matched exposure without it, and WRONG receives a competing relation-valid answer in the same slot. After an identical rehearsal-free continual fine-tuning stream, LEARNED reacquires the target faster than CONTROL at a prespecified checkpoint of partial forgetting. WRONG shows no positive advantage over CONTROL, while its comparison with LEA
    
[^217]: 信任但需验证：为离散扩散设计区域非均匀的前向动力学

    Trust but Verify: Designing Region-Inhomogeneous Forward Dynamics for Discrete Diffusion

    [https://arxiv.org/abs/2610.22380](https://arxiv.org/abs/2610.22380)

    该论文提出区域非均匀前向动力学，通过在训练和推理阶段对可疑上下文注入低速率噪声，打破离散扩散中“完全信任上下文”的假设，从而有效抑制错误在生成过程中的传播。

    

    连续扩散将前向过程视为一个可设计空间；而离散语言扩散通常采用吸收式破坏，使上下文在整个生成过程中被完全信任。这使得信任成为二元的：被破坏的上下文被直接当作真值复制，从而将错误传播到生成的片段中。我们通过将吸收核与均匀混合核相结合，并采用区域相关的速率，重新打开了这一设计空间。缺失位置遵循吸收动力学，而上下文则经历低速率的随机替换，使模型在训练中接触到“已观察但不确定”的词元。在推理阶段，一种启发式采样器以退火强度向可疑上下文重新注入噪声，并在轨迹后期冻结上下文更新。对一个预训练的1.7亿参数掩码扩散模型进行8000步微调，在一块6 GB显存的笔记本GPU上仅需不到五小时。即使观察到的词元是均匀随机的垃圾内容，基础模型仍会对这些词元分配大约50%的概率。

    arXiv:2610.22380v1 Announce Type: new  Abstract: Continuous diffusion treats the forward process as a design space; discrete language diffusion often uses absorbing corruption, leaving context fully trusted throughout generation. This makes trust binary: corrupted context is copied as ground truth, propagating errors into generated spans. We reopen this design space by combining absorbing and uniform-mixing kernels with region-dependent rates. Missing positions follow absorbing dynamics, while context undergoes low-rate random substitution, exposing the model to observed but uncertain tokens during training. At inference, a heuristic sampler re-injects noise into suspicious context under an annealed intensity and freezes context updates late in the trajectory. An 8,000-step finetune of a pretrained 170M-parameter masked diffusion model takes under five hours on a 6 GB laptop GPU. The base model assigns roughly 50% probability to observed tokens even when they are uniform-random garbage
    
[^218]: 什么时候大语言模型是一个好的校准智能体？面向预算受限工程模拟器校准的物理标注提示方法

    When is a large language model a good calibration agent? Physics-labeled prompting for budgeted engineering simulator calibration

    [https://arxiv.org/abs/2610.22379](https://arxiv.org/abs/2610.22379)

    该研究发现在计算预算受限的工程模拟器校准任务中，以物理诊断描述为条件的提示能让大语言模型在早期评估阶段显著优于传统无导数优化器——首个物理信息提议在25次试验中23次改善初始设计，五次评估后中位校准误差降低21.9倍。

    

    校准昂贵的工程模拟器需要调整模型参数以匹配参考数据，这一过程受到计算成本的严重制约。本研究探讨大语言模型（LLM）在以模拟误差的物理诊断描述为条件时，能否比传统无导数优化器更高效地提出有价值的参数更新建议。研究在圆柱尾流大涡模拟和阻尼振荡器逆问题上进行了测试，将物理标注提示与四种文本消融对照（移除、匿名化、填充、误导）以及标准优化器（Nelder-Mead、贝叶斯优化、CMA-ES和随机搜索）进行了对比评估。结果表明存在显著的早期语义优势：在25次试验中有23次，首个基于物理信息的LLM提议改善了初始设计；经过五次评估后，中位校准误差比无物理信息的对照组低21.9倍。然而，这一优势……（摘要原文在此处截断）

    arXiv:2610.22379v1 Announce Type: cross  Abstract: Calibrating expensive engineering simulators requires tuning model parameters to match reference data, a process heavily constrained by computational costs. This study investigates whether Large Language Models (LLMs), conditioned on physical diagnostic descriptions of simulation errors, can propose informative parameter updates more efficiently than conventional derivative-free optimizers. We test this on a cylinder-wake Large-Eddy Simulation and a damped-oscillator inverse problem. Physics-labeled prompts are evaluated against four text ablations (removed, anonymized, padded, misdirected) and standard optimizers (Nelder-Mead, Bayesian optimization, CMA-ES, and random search). Results demonstrate a strong early semantic advantage: the first physics-informed LLM proposal improves the initial design in 23 of 25 trials. After five evaluations, the median calibration error is 21.9 times lower than physics-free controls. However, this adva
    
[^219]: 内存带宽，而非FLOPs：在边缘GPU上剖析与加速基于查询的分割解码器

    Memory Bandwidth, Not FLOPs: Profiling and Accelerating Query-Based Segmentation Decoders on Edge GPUs

    [https://arxiv.org/abs/2610.22377](https://arxiv.org/abs/2610.22377)

    该研究揭示了边缘GPU上基于查询的分割模型解码器的真正瓶颈是内存带宽（跨内核激活数据传输）而非FLOPs，并提出CompactInst，利用4倍无参数空间池化减少内存传输以实现加速。

    

    基于查询的分割模型即使在工作站GPU上进行实时推理时也能取得出色的性能。然而，我们发现在边缘设备上，其Transformer解码器严重受限于内存带宽——这一瓶颈在标准的FLOP分析中不可见，并且随着输入分辨率的提高和DRAM带宽的限制而加剧。通过启用张量核心GEMM的FP16屋顶线分析与硬件DRAM计数器测量表明，FastInst和Mask2Former的非张量核心部分仅达到CUDA核心脊点的2.5%至6%，却占用了解码器时间的53%至70%以及解码器字节数的61%至71%。其中最大的单个内核可持续维持高达90%的标称峰值DRAM带宽，实际上已使内存总线饱和。因此，瓶颈主要并非源于较低的GEMM计算强度，而在于跨内核的激活数据传输。CompactInst通过4倍的无参数空间池化来解决这一瓶颈，从而减少了传输的字节数（原文摘要在此处截断）。

    arXiv:2610.22377v1 Announce Type: cross  Abstract: Query-based segmentation models achieve strong performance even under real-time inference on workstation GPUs. On edge devices, however, we show their transformer decoders are deeply memory-bandwidth-bound, a bottleneck invisible to standard FLOP analysis that worsens with input resolution and limited DRAM bandwidth. An FP16 roofline analysis with tensor-core GEMMs active and hardware DRAM counters shows that the non-tensor-core aggregates of FastInst and Mask2Former reach only $2.5\%$ to $6\%$ of the CUDA-core ridge yet account for $53\%$ to $70\%$ of decoder time and $61\%$ to $71\%$ of decoder bytes. The largest individual kernels sustain up to $90\%$ of nominal peak DRAM bandwidth, effectively saturating the memory bus. The bottleneck is therefore not primarily low GEMM intensity, but cross-kernel activation traffic. CompactInst addresses this bottleneck using parameter-free spatial pooling at $4{\times}$, which reduces the bytes m
    
[^220]: 物理迁移学习：一种沿复杂性阶梯实现零样本泛化的物理约束理论

    Physics-Transfer Learning: A Physics-Constrained Theory for Zero-Shot Generalization Along the Ladder of Complexity

    [https://arxiv.org/abs/2610.22376](https://arxiv.org/abs/2610.22376)

    该论文提出物理迁移学习（PT）理论，通过构建连接低成本源域与复杂目标域的锚定域链，并引入可在观测目标域标签前评估迁移效果的可计算可行性与可靠性指标，实现沿复杂性阶梯的零样本泛化。

    

    可靠的分布外泛化仍然是机器学习中的核心挑战，尤其是当模型被部署在数据稀缺的高保真领域时。现有方法通常依赖统计分布对齐或不变性假设，而当领域偏移反映了底层机制的变化时，这些方法往往会失效。受物理学中一个长期存在的范式启发——即从易于处理的理想化模型出发，逐步纳入更多机制和更高保真度层级来理解复杂系统——我们提出了物理迁移学习，这是一种用于沿复杂性阶梯进行零样本泛化的物理约束学习理论。PT构建了连接低成本源域与复杂目标域的锚定域链，并引入了可计算的可行性与可靠性指标，以便在观测到目标域标签之前评估迁移的可行性。在物理一致性和稳定性条件……（原文摘要到此截断）

    arXiv:2610.22376v1 Announce Type: cross  Abstract: Reliable out-of-distribution generalization remains a central challenge in machine learning, particularly when models are deployed in data-scarce high-fidelity domains. Existing approaches often rely on statistical distribution alignment, or invariance assumptions that can fail when domain shifts reflect changes in underlying mechanisms. Inspired by a longstanding paradigm in physics that understands complex systems by starting from tractable idealizations and progressively incorporating additional mechanisms and levels of fidelity, we propose physics-transfer (PT) learning, a physics-constrained learning theory for zero-shot generalization along a ladder of complexity. PT constructs chains of anchor domains (CADs) that connect low-cost source domains to complex targets and introduces computable feasibility and reliability indicators for assessing transfer before target-domain labels are observed. Under physical-consistency and stable-
    
[^221]: 超越固定采集函数：面向扰动发现的时间差分主动学习

    Beyond Fixed Acquisition Functions: Temporal Difference Active Learning for Perturbation Discovery

    [https://arxiv.org/abs/2610.22122](https://arxiv.org/abs/2610.22122)

    提出了一种名为TDAL的主动学习方法，将预算约束下的扰动发现建模为序贯决策问题，通过时间差分学习从单次筛选轨迹中在线学习采集函数，突破传统固定采集函数的局限，实现自适应的实验选择。

    

    基因与药物扰动筛选是阐明疾病机制和验证治疗靶点的重要工具，但实验成本限制了可测试的候选对象数量。现有的扰动筛选方法通常在利用反馈更新代理模型的同时保持采集函数固定，这限制了它们针对不同响应景观自适应调整选择策略的能力。实验反馈能够揭示当前的命中结果，并通过更新候选预测来指导后续的选择，从而为学习如何选择实验提供了训练信号。我们将预算约束下的扰动发现表述为一个序贯决策问题，并提出时间差分主动学习（TDAL），该方法通过时间差分学习从单次筛选轨迹中在线学习采集函数，同时兼顾当前的命中结果与未来的潜在发现。TDAL利用命中概率分布之间的Hellinger距离……

    arXiv:2610.22122v1 Announce Type: new  Abstract: Genetic and drug perturbation screens are important tools for elucidating disease mechanisms and validating therapeutic targets, but experimental costs limit the number of candidates that can be tested. Existing perturbation screening methods typically update the surrogate model with feedback while keeping the acquisition function fixed, limiting their ability to adapt selection to different response landscapes. Experimental feedback reveals current hits and informs subsequent selection by updating candidate predictions, providing a training signal for learning how to select experiments. We formulate budgeted perturbation discovery as a sequential decision problem and propose Temporal Difference Active Learning (TDAL), which learns an acquisition function online through temporal difference learning from a single screening trajectory, accounting for current hits and future discoveries. TDAL uses the Hellinger distance between hit probabil
    
[^222]: CNN技术签名探测器在“突破聆听”数据中的漏检与误报

    Missed Detections and False Alarms in a CNN Technosignature Detector on Breakthrough Listen Data

    [https://arxiv.org/abs/2610.22050](https://arxiv.org/abs/2610.22050)

    该研究通过向真实绿岸天文台数据中注入15,000个不同形态和振幅的信号，对一个冻结的CNN技术签名探测器进行压力测试，发现其召回率随信号增强反而下降，且某些干扰形态的假阳性率高达0.627，揭示了现有机器学习探测器在训练分布之外的脆弱性。

    

    射电技术签名（technosignature）搜索依赖于机器学习探测器在人工审查之前缩小候选池。这些探测器通常在来自训练分布的留出数据上进行验证，而这种验证方式恰恰排除了它们最可能失败的情形：技术签名具有未知的形态，而地面干扰的种类则无穷无尽。本文作者训练了一个标准的竞赛型探测器一次后将其冻结，然后用15,000次信号注入对其进行探测，涵盖十种形态和五种振幅，并将这些信号置于真实的绿岸天文台背景数据中。测试了两个方向：形态不同于直线窄带信号的真实信号，以及与训练数据相似的干扰。结果显示，召回率没有遵循预期的排序，且随着信号变亮反而下降，即使是探测器被训练寻找的那类“大海捞针”式信号也是如此（召回率从0.590降至0.090）。其中一种干扰形态达到了0.627的假阳性率，与探测器自身的基线相当。

    arXiv:2610.22050v1 Announce Type: cross  Abstract: Radio technosignature searches rely on machine learning detectors to shrink the candidate pool before a human looks at it. They are validated on held-out data from the training distribution, which excludes the cases where they are most likely to fail: a technosignature has unknown morphology, and terrestrial interference is unbounded in variety. I trained a standard competition detector once, froze it, and probed it with 15,000 injections across ten morphologies and five amplitudes, placed into real Green Bank backgrounds. Two directions were tested: genuine signals with morphologies other than a straight narrowband line, and interference resembling the training data. Recall did not follow the predicted ordering, and it fell as signals brightened, even for a needle of the kind the detector was trained on (from 0.590 to 0.090). One interference morphology reached a false positive rate of 0.627, comparable to the detector's own baseline 
    
[^223]: 二比特的N-Gram记忆：大语言模型中查找表的量化

    N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models

    [https://arxiv.org/abs/2610.21810](https://arxiv.org/abs/2610.21810)

    本文提出条件程序（CP）压缩表示方法，仅用两比特编码按需重构大语言模型n-gram记忆查找表中被请求的行，相比标准2比特量化将输出KL散度降低41–63%，并将DeepSeek的表存储从203GB压缩至51GB。

    

    条件记忆通过选择性查找来扩展语言模型的容量，但完整的嵌入表必须始终可用以应对每一个可能的输入。我们提出了条件程序，这是一种压缩表示方法，只需重建输入所请求的行。CP用短码和局部尺度来表示每一行，并使用一个在整个表上共享的重构程序。我们通过高斯格、范数恢复以及可独立解码的行记录来实例化这种表示。CP的两比特码相比2比特最近取整量化，将输出KL散度降低了41–63%。对三个经过训练的n-gram记忆族的分析表明，类高斯的坐标统计与行之间学习到的关系并存；在相同的存储预算下，共享格结构的表现优于我们测试的拟合码本和低秩残差变体。对于DeepSeek，该方法将表存储从203GB降至51GB，为更大的工作负载留出了空间。

    arXiv:2610.21810v1 Announce Type: cross  Abstract: Conditional memory expands language-model capacity through selective lookup, but the full embedding table must remain available for every possible input. We introduce Conditional Program (CP), a compressed representation that reconstructs only the rows an input requests. CP represents each row by a short code and a local scale, with a reconstruction program shared across the table. We instantiate this representation with a Gaussian trellis, norm restoration and independently decodable row records. CP's two-bit code lowers output KL by 41--63\% relative to 2-bit round-to-nearest. Analysis across three trained n-gram memory families shows that Gaussian-like coordinate statistics coexist with learned relationships among rows; at matched storage budgets, the shared trellis outperforms the fitted codebooks and low-rank residual variants we test. For DeepSeek, it reduces table storage from 203GB to 51GB, leaving room for larger workloads whe
    
[^224]: 虽经编码却暴露无遗：语言嵌入在推荐系统中的信息泄露超乎想象

    Encoded but Exposed: Language Embeddings can Reveal More Than They Should in Recommender Systems

    [https://arxiv.org/abs/2610.21713](https://arxiv.org/abs/2610.21713)

    该研究揭示了推荐系统中的语言嵌入存在严重隐私风险——简单的余弦相似度攻击即可高精度恢复物品身份，泄露还可进一步传播导致用户再识别，而以混淆描述作为防御手段则面临明显的泄露-效用权衡。

    

    语言嵌入在推荐系统中被越来越多地用于表示物品内容，但其隐私风险仍然鲜为人知。我们研究了当攻击者能够获取来自外部来源的辅助物品文本时，物品嵌入是否会泄露物品身份。我们证明，在两个真实世界数据集上，一种简单的余弦相似度攻击能够在不同语言模型和候选集设置下，以较高的平均倒数排名（MRR）恢复物品身份。当攻击者的部分信息与受害者的表示重叠时，泄露最为严重；即使描述来自独立来源，泄露仍可通过语义对齐而持续存在。我们进一步表明，物品泄露可以传播并导致基于稀疏评分的用户再识别。最后，我们评估了使用混淆描述作为缓解手段，并发现存在明显的泄露-效用权衡。

    arXiv:2610.21713v1 Announce Type: cross  Abstract: Language embeddings are increasingly used in recommenders to represent item content, but their privacy risks remain poorly understood. We study whether item embeddings can reveal item identities when an adversary has access to auxiliary item text from external sources. We show that a simple cosine-similarity attack can recover item identities with high mean reciprocal rank across language models and candidate-set settings on two real-world datasets. Leakage is strongest when the adversary's partial information overlaps with the victim representation, but it also persists through semantic alignment between independently sourced descriptions. We further show that item leakage can propagate to user re-identification from sparse ratings. Finally, we evaluate obfuscated descriptions as a mitigation and find a clear leakage-utility tradeoff.
    
[^225]: 语言模型智能体如何重塑价格竞争

    How language model agents reshape price competition

    [https://arxiv.org/abs/2610.21711](https://arxiv.org/abs/2610.21711)

    大语言模型定价智能体能够突破标准学习算法难以摆脱的伯特兰竞争“边际成本定价”基准，且其行为关键取决于部署接口——直接从原始市场历史发布价格的智能体可在自我对弈中维持超竞争高价，而从自身利润摘要中输出概率分布的智能体则趋向竞争性定价。

    

    重复伯特兰竞争是算法定价的一个严苛基准：在同质商品市场中，标准学习程序通常收敛到阶段博弈的纳什结果，即价格接近边际成本。我们证明，大语言模型（LLM）定价智能体能够突破这一基准，而这取决于其部署方式的选择。LLM凭借更快的适应能力优于标准在线学习器，但在面对全信息学习器时，这一优势基本消失。通过提取替代算法，我们揭示了特定部署方式下的LLM如何通过定期压低价格、抢占较慢学习器而获利。每一期从包含对手价格的原始市场历史中直接发布单一价格的智能体，在自我对弈中能够维持超竞争（高于竞争水平）的定价状态；而定期从自身利润的摘要历史中输出概率分布的智能体，则在自我对弈中趋向竞争性区域。混合这两种接口的市场则可能产生动态的价格状态。

    arXiv:2610.21711v1 Announce Type: cross  Abstract: Repeated Bertrand competition is a stringent benchmark for algorithmic pricing: in homogeneous-goods markets, standard learning procedures typically select the stage-game Nash outcome with prices near marginal cost. We show that LLM pricing agents can escape this benchmark, depending on deployment decisions. LLMs outperform standard online learners through faster adaptation, but this advantage largely disappears against full-information learners. An extracted surrogate algorithm illustrates how LLMs in specific deployments profit from regularly undercutting slower learners. Agents that post one price every period from raw market histories that include rival prices can sustain supracompetitive regimes in self-play. Agents that periodically emit probability distributions from summarized histories of their own profits instead move toward the competitive region in self-play. Markets mixing these interfaces can generate dynamic regimes in w
    
[^226]: 路由漂移控制：缓解混合专家强化学习中的离策略退化

    Router Drift Control: Mitigating Off-Policy Degradation in Mixture-of-Experts Reinforcement Learning

    [https://arxiv.org/abs/2610.21682](https://arxiv.org/abs/2610.21682)

    提出路由漂移控制（RDC）方法，将采样与训练阶段的路由分歧作为离策略过时信号，通过序列级过滤拒绝高漂移响应，在不引入梯度偏差的前提下缓解混合专家强化学习中的路由漂移与离策略退化。

    

    强化学习（RL）可能因训练与推理不匹配、小批量更新以及异步训练而变为离策略。由此产生的不匹配在混合专家模型中尤为有害，因为采样与训练之间的差异可能改变路由决策并导致路由漂移。采样路由重放（R3）通过在训练阶段复用采样阶段的专家选择来缓解这种不匹配。我们的分析表明，路由重放会因丢失训练路由器选中而采样路由器未选中的专家的贡献，从而引入梯度偏差。为解决这一问题，我们提出了路由漂移控制（RDC），它将采样与训练之间的路由分歧作为离策略过时性的信号，并选择性地拒绝高漂移的响应。这种序列级过滤避免了重放引入的梯度偏差，同时保留了原生的专家选择及被保留响应所对应的学习信号。

    arXiv:2610.21682v1 Announce Type: cross  Abstract: Reinforcement learning (RL) can become off-policy due to training-inference mismatch, mini-batch updates, and asynchronous training.The resulting mismatch is especially harmful in Mixture-of-Experts (MoE) models, where rollout--training discrepancies can change routing decisions and lead to router drift. Rollout Routing Replay (R3) mitigates this mismatch by reusing expert selections from the rollout stage during training. Our analysis shows that routing replay introduces gradient bias by losing contributions from experts selected by the training router but not the rollout router. To address this, we propose Router Drift Control (RDC), which uses routing disagreement between rollout and training as a signal of off-policy staleness and selectively rejects high-drift responses. The sequence-level filtering avoids gradient bias introduced by replay while preserving native expert selection and the associated learning signals for retained r
    
[^227]: ManiFusion：通过流形空间中的叠加解锁高吞吐量生成

    ManiFusion: Unlocking the High-Throughput Generation via Superposition in Manifold Space

    [https://arxiv.org/abs/2610.21433](https://arxiv.org/abs/2610.21433)

    ManiFusion提出了一个与骨干网络无关的框架，通过在去噪骨干前后对多个样本进行融合与解融合，并采用轨迹融合调度来对齐去噪阶段，从而在单次去噪中并行处理多个样本，突破单样本延迟优化的瓶颈，实现高吞吐量的扩散模型图像生成。

    

    扩散模型能够实现高质量的图像生成，但由于需要在多个时间步上进行迭代去噪，其计算开销依然很高。现有的加速方法主要着眼于降低单样本延迟，而这一方向上的收益正趋于饱和。我们转而探索一个与之正交的、面向吞吐量的方向：在单次去噪计算中同时处理多个样本。为此，我们提出了ManiFusion，一个与骨干网络无关的框架，通过融合空间计算实现多样本生成。ManiFusion通过在去噪骨干之前对多个样本进行融合、在其后通过解融合将它们分离，从而联合处理多个样本。我们从几何视角阐释了多个样本如何共享一个融合表示，同时仍保留引导各自生成轨迹所需的信息。为了使融合分配更好地与去噪阶段相适配，ManiFusion采用了轨迹融合调度

    arXiv:2610.21433v1 Announce Type: cross  Abstract: Diffusion models achieve high-quality image generation but remain computationally expensive due to iterative denoising across many timesteps. Existing acceleration methods mainly reduce per-sample latency, but gains along this axis are becoming saturated. We instead explore an orthogonal throughput-oriented direction by processing multiple samples within a single denoising evaluation. To this end, we propose ManiFusion, a backbone-agnostic framework for multi-sample generation through fused-space computation. ManiFusion jointly processes multiple samples by fusing them before the denoising backbone and separating them afterward through unfusion. We provide a geometric perspective on how multiple samples can share a fused representation while retaining the information needed to guide their individual generation trajectories. To better align fusion allocation with denoising stages, ManiFusion uses Trajectory-Fusion Scheduling (TFS), whic
    
[^228]: 在策略蒸馏需要完成每一次 Rollout 吗？

    Does On-Policy Distillation Need to Complete Every Rollout?

    [https://arxiv.org/abs/2610.21337](https://arxiv.org/abs/2610.21337)

    该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。

    

    在策略蒸馏（On-policy Distillation, OPD）通过对学生模型生成的 rollout 序列提供反馈，有效地将教师模型的能力迁移给学生模型。然而，生成完整的学生 rollout 序列并为每个 token 计算教师反馈，使得 OPD 的计算开销十分高昂。我们提出了随机截断在策略蒸馏（Stochastic Truncated On-Policy Distillation, STOPD），该方法通过随机截断学生 rollout 序列，并利用分块逆概率加权（blockwise IPW）校正由此产生的采样偏差，从而加速 OPD。在每一步训练中，我们在生成前为每个 rollout 采样一个停止位置，让学生模型生成至该位置或自然终止。我们对每个生成的 token 获取教师反馈，并使用分块 IPW 补偿保留后续 token 的较低概率，同时对权重加以控制。我们证明了这种加权方式在期望意义上保持了原始 OPD 的梯度总和不变，并推导出了一步更新的误差界……

    arXiv:2610.21337v1 Announce Type: cross  Abstract: On-policy distillation (OPD) effectively transfers capabilities from a teacher to a student through feedback on student-generated rollouts. However, generating complete student rollouts and computing teacher feedback for every token make OPD computationally expensive. We introduce Stochastic Truncated On-Policy Distillation (STOPD), which accelerates OPD by randomly truncating student rollouts and correcting the resulting sampling bias with blockwise inverse-probability weighting (IPW). At each training step, we sample a stopping position for each rollout before generation and let the student generate until that position or natural termination. We obtain teacher feedback on every generated token and use blockwise IPW to compensate for the lower probability of retaining later tokens while controlling the weights. We prove that this weighting preserves Vanilla OPD's gradient sum in expectation and derive one-step update-error bounds for 
    
[^229]: SALTO：用于非线性事件推断的求解器锚定提升转移算子

    SALTO: Solver-Anchored Lifted Transfer Operators for Nonlinear Event Inference

    [https://arxiv.org/abs/2610.21049](https://arxiv.org/abs/2610.21049)

    SALTO是一个基于Koopman算子的框架，通过在提升空间中构建可复用的线性化伴随算子，仅需一次编码器前向传播和一个内积即可快速推断任意有限幅度的局部非线性事件源。

    

    学习到的动力学模型能够准确预测轨迹，但在求导或转置时产生的敏感性却不可靠，这限制了其在逆问题和科学决策中的应用。经典伴随方法能提供精确的局部导数，但单个伴随算子无法表示系统对一族有限幅度扰动的非线性响应。我们提出了SALTO（求解器锚定提升转移算子），这是一个基于Koopman算子的框架，用于推断有限幅度扰动源。在提升空间中，非线性动力学由一个线性传播子来近似，其转置能够在单次求解中将每个传感器信号反向传播。由此得到一个可复用的提升伴随算子，仅需一次编码器前向传播和一个内积运算，即可预测任意候选局部事件所产生的测量值。为了让模型获得模拟器的精确局部行为，SALTO施加了三个结构约束：它将物理态动力学锚定（摘要在此处被截断）

    arXiv:2610.21049v1 Announce Type: new  Abstract: Learned dynamical models can accurately predict trajectories while producing unreliable sensitivities when differentiated or transposed, limiting their use in inverse problems and scientific decision-making. Classical adjoints provide exact local derivatives, but a single adjoint cannot represent the nonlinear response of a system to a family of finite-amplitude perturbations. We introduce SALTO (Solver-Anchored Lifted Transfer Operators), a Koopman-based framework for inferring finite-amplitude sources. In the lifted space, the nonlinear dynamics are approximated by a linear propagator, and its transpose carries each sensor backward in a single solve. The result is a reusable lifted adjoint that predicts the measurement produced by any candidate localized event with one encoder pass and an inner product. To give the model the simulator's exact local behavior, SALTO imposes three structural constraints: it anchors the physical-state dyna
    
[^230]: 时间划分高斯过程混合模型

    Time-partitioning Gaussian Process Mixture Model

    [https://arxiv.org/abs/2610.20860](https://arxiv.org/abs/2610.20860)

    提出了一种贝叶斯非参数框架TPGPMM，通过自适应Pitman-Yor先验与特定状态高斯过程的结合，自动识别时间序列中的持续状态与变点，无需马尔可夫假设即可建模非平稳依赖，并能有效区分持续变化与瞬态冲击。

    

    我们提出了时间划分高斯过程混合模型（TPGPMM），这是一个用于识别时间序列中持续状态并建模非平稳依赖关系的贝叶斯非参数框架。该模型将连续分段上的自适应Pitman-Yor先验与特定状态的高斯过程相结合，而不施加状态内的马尔可夫依赖。通过学习Pitman-Yor折扣参数，模型能够适应同质和异质的状态长度。一种带有精确Gibbs边界更新的分裂-合并Metropolis-within-Gibbs采样器可联合学习变点和模型超参数，量化分段及状态内动态的不确定性。使用RBF核和固定的保守观测噪声水平，TPGPMM倾向于稀疏分段，能够区分持续变化与瞬态冲击。仿真实验证明了该模型对状态结构的恢复能力。该方法已应用于CBOE波动率…

    arXiv:2610.20860v1 Announce Type: cross  Abstract: We propose the Time-partitioning Gaussian Process Mixture Model (TPGPMM), a Bayesian nonparametric framework for identifying persistent regimes and modelling nonstationary dependence in time series. The model combines an adaptive Pitman--Yor prior on contiguous segmentations with regime-specific Gaussian processes, without imposing Markovian within-regime dependence. Learning the Pitman--Yor discount parameter allows adaptation to homogeneous and heterogeneous regime lengths.   A split--merge Metropolis-within-Gibbs sampler with exact Gibbs boundary updates jointly learns change points and model hyperparameters, quantifying uncertainty over segmentations and within-regime dynamics. With an RBF kernel and a fixed, conservative observation-noise level, TPGPMM favours sparse segmentations that distinguish persistent shifts from transient shocks. Simulations demonstrate recovery of the regime structure. Applications to the CBOE Volatility 
    
[^231]: 二次网络中的谱不可区分性与学习曲线可辨识性

    Spectral Indistinguishability and Learning-Curve Identifiability in Quadratic Networks

    [https://arxiv.org/abs/2610.20678](https://arxiv.org/abs/2610.20678)

    该论文构造了谱测度完全一致（谱不可区分）但学习曲线仍可区分的二次网络对，证明谱摘要无法唯一辨识网络，而标量学习曲线可以实现辨识，且其查询复杂度随循环长度呈指数增长。

    

    我们研究谱摘要和标量学习曲线能揭示中心化二次网络的哪些信息。我们构造了具有精确高斯单坐标和双坐标边缘分布的解析正输入密度。它们的完整四阶矩算子被一个固定整个对角教师代数的正交映射所交织，因此相应的锚定谱测度完全一致。对于具有非零循环不变量的选定常见教师，由于该映射不保持矩阵乘法，其因子化损失曲线则有所不同。循环长度 $r$ 可以任意大，对于每个固定构造，在训练时间内一致地给出 $O(\varepsilon^r)$ 的对比度。对于给定的网络对，带噪声的标量损失查询需要 $\varepsilon^{-2r}$ 量级的观测，而带标量梯度读出的有界初始化干预则需要 $\varepsilon^{-2}$ 量级的观测。一个等于配对梯度差的共享输入分数，能够达到……

    arXiv:2610.20678v1 Announce Type: cross  Abstract: We study what spectral summaries and scalar learning curves reveal about centered quadratic networks. We construct analytic positive input densities with exactly Gaussian one- and two-coordinate marginals. Their complete fourth-moment operators are intertwined by an orthogonal map fixing the entire diagonal teacher algebra, so the corresponding anchored spectral measures agree. For selected common teachers with a nonzero cycle invariant, their factorized loss curves differ because the map does not preserve matrix multiplication. Cycle length $r$, which can be arbitrarily large, gives $O(\varepsilon^r)$ contrast uniformly in training time for each fixed construction. For a supplied pair, noisy scalar loss queries require order $\varepsilon^{-2r}$ observations, while bounded initialization interventions with scalar gradient readouts require order $\varepsilon^{-2}$. A shared-input score, equal to a paired-gradient difference, attains the
    
[^232]: CascadeLedger：LLM级联中信息与资源完整性的反事实审计

    CascadeLedger: Counterfactual Auditing of Information and Resource Integrity in LLM Cascades

    [https://arxiv.org/abs/2610.20668](https://arxiv.org/abs/2610.20668)

    CascadeLedger提出了一个面向预算受限LLM级联的反事实执行审计框架，通过聚合等价轨迹有效性和反事实前缀无干扰性等机制，审计路由决策是否仅使用了合法可观测信息，并能定位最早违规点与最小隐藏见证。

    

    我们提出了CascadeLedger，一个面向预算受限语言模型级联的反事实执行审计框架。该协议强制执行“先调用后读取”语义，将成功的读取操作绑定到服务与缓存标识上，记录可用性状态及各组件的预算转换，并在接入任何目标模型或任务评分器之前冻结事件轨迹。其核心对象是聚合等价轨迹有效性：两次执行可以拥有相同的最终答案、声明的资源投影和服务直方图，但其路由决策是否仅使用了合法可观测的信息却可能存在差异。我们用反事实前缀无干扰性（CPNI）来补充单轨迹不变量，该性质在耦合可见前缀与策略随机性的同时，仅改变一个声明的隐藏族。反事实影响边界（CIF）定位最早违规的前缀以及一个最小隐藏见证；保持投影的对抗性变异（PPAM）

    arXiv:2610.20668v1 Announce Type: cross  Abstract: We introduce CascadeLedger, a counterfactual execution-audit framework for budgeted language-model cascades. The protocol enforces call-before-read semantics, binds successful reads to service and cache identities, records availability and component-wise budget transitions, and freezes the event trace before any target or task scorer is joined. The central object is aggregate-equivalent trace validity: two executions can share a final answer, declared resource projection, and service histogram while differing in whether their routing decisions used only legitimately observable information. We complement single-trace invariants with counterfactual prefix non-interference (CPNI), which couples a visible prefix and policy randomness while changing only a declared hidden family. The counterfactual influence frontier (CIF) localizes the earliest violating prefix and a minimal hidden witness; projection-preserving adversarial mutation (PPAM)
    
[^233]: 可学习霍奇-拉普拉斯驱动的混合专家模型用于粗非结构网格上的稳态偏微分方程代理模型

    Learnable Hodge-Laplacian Driven Mixture of Experts for Steady-PDE Surrogates on Coarse Unstructured Meshes

    [https://arxiv.org/abs/2610.20585](https://arxiv.org/abs/2610.20585)

    该论文提出一种由可学习霍奇-拉普拉斯算子驱动的混合专家架构，将解按频谱尺度分配给不同transformer专家并通过逐顶点门控权重重组，在粗非结构网格稳态PDE代理任务上相比Transolver将相对$L_2$误差降低了31%。

    

    当前针对非结构网格上稳态偏微分方程代理模型的方法对整个解应用单一的混合机制。本文提出了一种新架构，将解划分给多个标准transformer专家，每个专家专注于不同的频谱尺度。该划分由离散霍奇-拉普拉斯算子 $L_0=\star_0^{-1} d_0^{\top}\star_1 d_0$ 构建，作为测量高频内容的拟合层。门控机制以网格为输入，生成逐顶点权重以重新组合各专家的输出。AirFRANS是一个稳态雷诺平均Navier-Stokes（RANS）翼型基准测试。我们在该基准的精简重网格化版本上评估了所提方法，每个非结构网格包含9.6K个面和4.9K个顶点。我们在相同的数据集和数据集划分上与Transolver进行了对比。我们的方法将相对$L_2$误差降低了31%，并降低了内部均方误差。

    arXiv:2610.20585v1 Announce Type: new  Abstract: Current approaches for steady partial differential equations surrogate models on unstructured meshes apply a single mixing mechanism to the whole solution. This paper proposes an architecture that partitions the solution among vanilla transformer experts, each focused on a different spectral scale. The partition is built from a discrete Hodge Laplacian $L_0=\star_0^{-1} d_0^{\top}\star_1 d_0$, that serves as a fitting layer measuring high-frequency content. A gating mechanism takes the mesh as input and produces the per-vertex weights that recombine the expert outputs. AirFRANS is a steady Reynolds-averaged Navier--Stokes (RANS) airfoil benchmark. We evaluate our approach in a reduced remeshed version of the benchmark with each unstructured mesh counting 9.6K faces and 4.9K vertices. We compare our approach with the Transolver on the same dataset and dataset slit. Our approach reduces relative $L_2$ error by 31\% and interior mean square
    
[^234]: ActiveEvidence：具有可验证获取差距的决策导向证据图

    ActiveEvidence: Decision-Focused Evidence Graphs with Verifiable Acquisition Gaps

    [https://arxiv.org/abs/2610.20532](https://arxiv.org/abs/2610.20532)

    该论文提出ActiveEvidence框架，通过反事实证据交互图（CEIG）和可验证的ε-决策证书，使多模态智能体能够高效决定何时获取视觉证据、仅评估可能改变决策的关键交互，并在证明进一步交互无法改变所选动作时及时停止。

    

    多模态智能体必须决定视觉证据是否值得获取、查询哪个候选对象或分辨率、以及进一步的交互何时不再能改变其决策。我们在严格的免费元数据/付费观测边界、随机向量成本以及冻结执行轨迹的条件下研究这一问题，并将其视为可决策认证的主动证据获取问题。反事实证据交互图（CEIG）表示一元证据价值、带符号的二阶交互以及有向条件性损害，同时为未解决的交互维护显式的分数包络；决策加权包络目标收紧了可能改变动作身份的界限。间隙导向的细化仅评估能够取代现有获取方案的交互，一旦ε-决策证书表明未解决的项在声明的容差范围内无法改变所选动作，便停止评估。该证书是……

    arXiv:2610.20532v1 Announce Type: cross  Abstract: A multimodal agent must decide whether visual evidence is worth acquiring, which candidate or resolution to query, and when further interaction can no longer change its decision. We study this problem under a strict free-metadata/paid-observation boundary, stochastic vector costs, and a frozen execution trace, and treat it as decision-certifiable active evidence acquisition. A Counterfactual Evidence Interaction Graph (CEIG) represents unary evidence value, signed second-order interaction, and directed conditional harm while maintaining explicit score enclosures for unresolved interactions; a decision-weighted enclosure objective tightens the bounds that can change action identity. Gap-directed refinement evaluates only interactions that can displace the incumbent acquisition and stops once an $\epsilon$-decision certificate shows that unresolved terms cannot change the selected action beyond a declared tolerance. The certificate is mo
    
[^235]: SafeAdapt-Poker：对手漂移下预算化适应的反事实风险校准

    SafeAdapt-Poker: Counterfactual Risk Calibration for Budgeted Adaptation under Opponent Shift

    [https://arxiv.org/abs/2610.20473](https://arxiv.org/abs/2610.20473)

    该论文提出 SafeAdapt-Poker，将对手漂移下的预算化适应建模为仅依赖决策前信息的双重反事实序贯控制问题，通过单调策略风险模型与配对预算状态延续模型分别校准偏离的局部下行风险和预算消耗的未来代价，并用逐路径累计KL账本约束总偏离量，从而在即时适应价值与安全风险之间实现可校准的权衡。

    

    在行为发生漂移后，对手的历史行为记录可能会产生误导。SafeAdapt-Poker 将对手漂移下的预算化适应问题表述为一个双重反事实序贯控制问题，仅使用每次决策之前可获得的信息。在每次决策时，控制器会提出两个不同的问题：一是在匹配的合法后续情形下，偏离冻结锚点会引发怎样的局部下行风险；二是现在消耗一部分剩余偏离预算会损失多少未来的适应价值。第一个问题由一个关于偏离强度的单调策略风险模型来回答；第二个问题则由一个配对的预算状态延续模型来回答，该模型为当前的预算消耗定价。控制器通过将即时适应价值与这两个量进行权衡，在保持支撑集的候选偏离中进行选择；一条逐路径的累计KL散度账本对相对于锚点的总偏离量进行约束；同时从一组有限的冻结控制器中进行筛选。

    arXiv:2610.20473v1 Announce Type: cross  Abstract: Opponent histories can mislead after behavioral shifts. SafeAdapt-Poker formulates budgeted adaptation under opponent shift as a dual-counterfactual sequential-control problem using only information available before each decision. At each decision the controller asks two distinct questions: what local downside a deviation from the frozen anchor would induce under matched legal continuations, and what future adaptation value would be lost by consuming part of the remaining deviation budget now. The first question is answered by a monotone strategic-risk model over deviation intensity; the second by a paired budget-state continuation model that prices current budget consumption. The controller selects among support-preserving candidate deviations by trading immediate adaptation value against these two quantities, a pathwise cumulative-KL ledger bounds total anchor-relative deviation, and a finite family of frozen controllers is screened 
    
[^236]: 基于加权中位数的假设选择

    Hypothesis Selection via Weighted Medians

    [https://arxiv.org/abs/2610.20375](https://arxiv.org/abs/2610.20375)

    本文提出了一种基于加权中位数的假设选择算法，在达到最优近似因子和样本复杂度的同时，将运行时间对失败概率的依赖从多项式级改进为对数级。

    

    给定 n 个分布（假设）$\mathcal H=\{H_1,\ldots,H_n\}$ 以及来自未知目标分布 P 的样本，适当假设选择旨在找到一个分布，使其与 P 的总变差距离接近已知分布中可达到的最佳水平。我们提出了一种算法，该算法使用 $O(\varepsilon^{-2}\log(n/\delta))$ 个目标样本，运行时间为 $O(n\varepsilon^{-2}\log(n/\delta))$，以至少 $1-\delta$ 的概率返回满足 $d_\mathrm{TV}(P,H_{\hat i})\leq 3\min_{i\in[n]}d_\mathrm{TV}(P,H_i)+\varepsilon$ 的 $H_{\hat i}$。该算法达到了最优的近似因子和样本复杂度。该算法的运行时间将 Aamand、Aliakbarpour、Chen 和 Silwal（COLT 2026）的最先进界限 $\tilde{O}(n\varepsilon^{-2}\delta^{-1})$ 中对失败概率的依赖从多项式级改进为对数级，从而回答了他们提出的问题 2。

    arXiv:2610.20375v1 Announce Type: cross  Abstract: Given $n$ distributions (hypotheses) $\mathcal H=\{H_1,\ldots,H_n\}$ and samples from an unknown target distribution $P$, proper hypothesis selection seeks a distribution whose total variation distance to $P$ is close to the best achievable within the known distributions. We give an algorithm that, using $O(\varepsilon^{-2}\log(n/\delta))$ target samples, running in $O(n\varepsilon^{-2}\log(n/\delta))$ time, with probability at least $1-\delta$, returns $H_{\hat i}$ satisfying $d_\mathrm{TV}(P,H_{\hat i})\leq 3\min_{i\in[n]}d_\mathrm{TV}(P,H_i)+\varepsilon.$ The algorithm achieves the optimal approximation factor and sample complexity. The running time improves the dependence on the failure probability from polynomial to logarithmic in the state-of-the-art bound $\tilde{O}(n\varepsilon^{-2}\delta^{-1})$ of Aamand, Aliakbarpour, Chen, and Silwal (COLT 2026), thereby answering their Question~2.
    
[^237]: TANGO：RNA拓扑与几何协同设计

    TANGO: RNA Topology and Geometry Co-Design

    [https://arxiv.org/abs/2610.20277](https://arxiv.org/abs/2610.20277)

    提出TANGO框架，首次将RNA逆设计表述为拓扑与几何协同设计问题，通过多目标多智能体强化学习和分阶段课程训练（二级结构拓扑感知策略学习、三级结构迁移训练、前沿引导微调），实现同时满足二级碱基配对拓扑与三级几何结构的RNA序列设计。

    

    设计满足目标三级结构要求的RNA序列是RNA工程中的一个重要问题，在合成生物学和治疗学领域有着广泛的应用。现有的3D RNA设计方法已取得长足进展，但它们在很大程度上将目标折叠视为纯几何目标。这对RNA而言是不完整的，因为RNA的功能折叠不仅由三级几何结构决定，还由二级碱基配对拓扑决定。因此，我们将RNA逆设计表述为拓扑与几何协同设计问题，并提出了TANGO——一个面向该任务的多目标多智能体强化学习框架。TANGO采用分阶段课程学习：首先从二级结构目标中学习拓扑感知策略，然后将该策略迁移到三级结构目标上训练一个通用的协同设计模型，最后应用前沿引导的微调来细化特定目标下二级结构保真度与三级结构几何之间的权衡。

    arXiv:2610.20277v1 Announce Type: new  Abstract: Designing RNA sequences that satisfy target tertiary-structure objectives is an important problem in RNA engineering, with broad applications in synthetic biology and therapeutics. Existing 3D RNA design methods have made substantial progress, but they largely treat the target fold as a geometry-only objective. This is incomplete for RNA, whose functional folds are specified not only by tertiary geometry but also by secondary base-pairing topology. We therefore formulate RNA inverse design as a topology-and-geometry co-design problem and propose TANGO, a multi-objective multi-agent reinforcement learning framework for this task. TANGO adopts a staged curriculum: it first learns a topology-aware policy from secondary-structure targets, then transfers this policy to tertiary-structure targets to train a general co-design model, and finally applies frontier-guided fine-tuning to refine target-specific trade-offs among secondary-structure fi
    
[^238]: 差距检测：通过需求与可辩护性缓解同质化

    Gap Detection: Mitigating Homogenization through Demand and Defensibility

    [https://arxiv.org/abs/2610.20162](https://arxiv.org/abs/2610.20162)

    提出Gap Detection框架，通过结合查询相关性（Demand）与现有内容相异度（Defensibility）的GapScore目标函数，在文本嵌入空间中检测相关但未被占据的方向，以缓解语言模型开放式回答的同质化问题。

    

    语言模型在回答开放式问题时常常生成语义相似的答案，即使存在许多合理的替代方案。生成有用的替代方案需要同时识别查询所要求的内容以及现有回答已经覆盖的内容。我们提出了差距检测（Gap Detection），这是一个在共享文本嵌入空间中搜索相关但尚未被占据的方向的框架。其目标函数GapScore将查询相关性（需求，Demand）与最近现有条目的相异度（可辩护性，Defensibility）相结合。该框架在单位球面上优化该乘积，利用语言模型将候选方向映射为概念词，并通过任务特定的过滤和受约束的提示将这些概念转化为文本。在五个精选主题和90个INFINITY-CHAT查询上，报告的输出碰撞率平均相对降低分别为12.8%和10.8%。我们还将该框架实例化应用于四个产品的生成式引擎优化。

    arXiv:2610.20162v1 Announce Type: new  Abstract: Language models often produce semantically similar answers to open-ended questions, even when many plausible alternatives exist. Generating useful alternatives requires identifying both what the query calls for and what existing responses already cover. We introduce Gap Detection, a framework that searches for relevant but unoccupied directions in a shared text embedding space. Its objective, GapScore, combines query relevance (Demand) with dissimilarity to the nearest existing item (Defensibility). The framework optimizes this product on the unit sphere, maps candidate directions to concept words with a language model, and uses task-specific filtering and constrained prompting to turn those concepts into text. On five curated topics and 90 INFINITY-CHAT queries, the reported mean relative reductions in output Collision are 12.8% and 10.8%, respectively. We also instantiate the framework for generative engine optimization over four produ
    
[^239]: 面向偏好学习的完全序贯高斯过程回归

    Fully sequential Gaussian process regression for preference learning

    [https://arxiv.org/abs/2610.19391](https://arxiv.org/abs/2610.19391)

    该论文针对目标函数只能通过带噪声成对比较来评估的连续全局优化问题，提出了基于矩匹配的完全序贯高斯过程近似方法DOVE，并首次为这类完全序贯近似贝叶斯推断建立了统计误差和期望简单遗憾的收敛速率保证。

    

    我们研究了一类连续全局优化问题，其目标函数无法被直接评估；相反，我们只能观测到在两个选定的设计点中哪一个（在有噪声的情况下）表现更好。这类成对比较会使高斯过程回归的使用变得复杂，因为其会产生非高斯后验。我们考虑一种基于矩匹配的完全序贯近似方法，该方法可以对元模型的均值和协方差实现快速的闭式更新。随后我们提出了DOVE，这是一种完全序贯采样策略，它将信息增益准则与旨在控制误差累积的随机采样相结合。在相当标准的正则性条件下，我们证明了依赖于核函数光滑性的统计误差和期望简单遗憾的收敛速率。据我们所知，这是连续全局优化领域中完全序贯近似贝叶斯推断的首个此类收敛速率结果。

    arXiv:2610.19391v1 Announce Type: new  Abstract: We study continuous global optimization problems in which the objective function cannot be directly evaluated; rather, we observe only which of two selected design points appears to perform better, subject to noise. Such comparisons complicate the use of Gaussian process regression because they yield non-Gaussian posteriors. We consider a fully sequential approximation, based on moment-matching, that yields a fast closed-form update of the metamodel's mean and covariance. We then propose DOVE, a fully sequential sampling policy that combines an information gain criterion with randomized sampling designed to control error accumulation. Under fairly standard regularity conditions, we prove convergence rates for both statistical error and expected simple regret that depend on kernel smoothness. To our knowledge, these are the first such rates for fully sequential approximate Bayesian inference in continuous global optimization.
    
[^240]: 驾驭不动点曲线：连续思维机器中的图像分类

    Riding the Fixed-Point Curve: Image Classification in the Continuous Thought Machine

    [https://arxiv.org/abs/2610.19220](https://arxiv.org/abs/2610.19220)

    本文揭示了连续思维机器（CTM）图像分类的内在机制：每个思考步骤是具有平衡放大的吸引性非正规映射，神经状态的不动点随每步移动，思维轨迹即这些不动点描绘的曲线（大部分运动跨图像共享并表现为行波活动），从而解释了CTM的涌现思维过程。

    

    连续思维机器（CTM）是一种循环神经网络，它关注输入并基于神经元之间的同步性做出预测。在跨越多个领域的任务中，它表现出尚未被解释的涌现思维过程。我们报告了关于CTM如何对图像进行分类的多项发现，重点聚焦于ImageNet。每个思考步骤是一个具有吸引性但强烈非正规的映射，表现出平衡放大效应（Murphy and Miller, 2009）；虽然吸引性在CTM初始化时就已存在，但训练极大地促进了非正规性。根据巴拿赫（Banach）定理，神经状态具有一个局部唯一的不动点，该不动点在每一步都会移动，而思维轨迹就是这些不动点所描绘的曲线，滞后一个时间步跟随其后。该曲线的大部分运动在不同图像之间是共享的，并表现为先前报道过的跨越神经元的行波活动。沿着思维的注意力调度……

    arXiv:2610.19220v1 Announce Type: new  Abstract: The Continuous Thought Machine (CTM) is a recurrent network that attends to inputs and makes predictions based on the synchronization between its neurons. On tasks spanning many domains, it displays emergent thought processes that have not been explained. We report a number of findings about how the CTM classifies images, focusing on ImageNet. Each thinking step is an attracting but strongly non-normal map exhibiting balanced amplification (Murphy and Miller, 2009); while attraction is already present at CTM initialization, training highly promotes non-normality. By Banach's theorem, the neural state has a locally unique attracting fixed point, which moves at every step, and the thought trajectory is the curve these fixed points trace, followed one tick behind. Most of that curve's motion is shared across images, and visible as a previously reported traveling wave of activity across neurons. The attention schedule along the thought is la
    
[^241]: 顿悟现象的统计小品

    Statistical Vignettes on the Grokking Phenomenon

    [https://arxiv.org/abs/2610.18811](https://arxiv.org/abs/2610.18811)

    该论文在一个可解析求解的简单模型中研究“顿悟”现象，厘清了初始化、信号强度、权重衰减、样本量与输入维度等因素通过不同机制产生相似学习曲线的作用，给出了顿悟必然发生的严格条件、泛化平台长度的解析渐近表达式，并揭示插值边缘是由消失谱隙刻画的一个独特动力学区域。

    

    在这项工作中，我们在一个简单且可解析处理的设定中研究顿悟现象，其丰富的现象学使我们能够厘清初始化、信号强度、权重衰减、样本量和输入维度各自的 影响。这些因素可以通过不同的机制产生表面上相似的学习曲线，从而使对其的解读变得复杂。我们的分析区分了借助干扰特征的早期拟合、核心特征学习以及延迟泛化。我们识别出了顿悟可被严格证明发生的精确条件，并推导出泛化平台长度的解析渐近表达式。我们的结果还刻画了早期泛化以及无法达到规定测试精度目标的区域。我们证明插值边缘构成了一个独特的动力学区域，其中消失的谱隙将核心学习与泛化分离开来，这有助于澄清模型接近插值边界所扮演的角色。

    arXiv:2610.18811v2 Announce Type: replace  Abstract: In this work, we study grokking in a simple, analytically tractable setting whose rich phenomenology allows us to disentangle the effects of initialization, signal strength, weight decay, sample size, and input dimension. These factors can produce superficially similar learning curves through distinct mechanisms, complicating their interpretation. Our analysis distinguishes early fitting through nuisance features, core feature learning, and delayed generalization. We identify sharp conditions under which grokking provably occurs and derive analytic asymptotic expressions for the lengths of the generalization plateaus. Our results also characterize regimes of early generalization and failure to reach a prescribed test accuracy target. We show that the interpolation edge constitutes a distinct dynamical regime, where vanishing spectral gaps separate core learning from generalization, helping clarify the role of proximity to interpolati
    
[^242]: SP$^2$ec：面向视觉语言模型的自适应自推测解码

    SP$^2$ec: Adaptive Self-Speculative Decoding for Vision-Language Models

    [https://arxiv.org/abs/2610.18735](https://arxiv.org/abs/2610.18735)

    提出免训练的赌博算法SP$^2$ec，利用自推测解码的单峰吞吐量特性，在生成过程中自适应调整草稿配置并最小化墙钟时间遗憾，从而加速视觉语言模型的推理。

    

    推测解码已被证明在加速大语言模型推理方面卓有成效，但对于视觉语言模型而言，输入模态的多样性使得小型训练的草稿模型难以实现持续的高吞吐量。自推测解码通过复用目标模型的部分作为草稿模型，从而保留了视觉语言模型的多模态能力，但其效率取决于能否为每个提示选择合适的草稿配置。我们提出了单峰推测解码，这是一种免训练的赌博算法，能够在生成过程中自适应地调整草稿配置。SP$^2$ec 利用在现有自推测方法（包括 SWIFT、KnapSpec、SpecVLM 以及我们提出的简单 cosSim 方法）中观察到的单峰吞吐量模式。我们的算法设计显式地为每个给定查询最小化一种新的墙钟时间遗憾，该遗憾与经验墙钟时间延迟相对应。理论保证……

    arXiv:2610.18735v2 Announce Type: replace  Abstract: Speculative Decoding (SD) has demonstrated its efficacy in accelerating large language model inference, but for Vision-Language Models (VLMs), the diverse input modalities make it difficult for a small trained drafter to achieve consistently high throughput. Self-speculative decoding reuses parts of the target model as a drafter which retains the multi-modal capability of the VLMs, yet its efficiency depends on choosing a suitable draft configuration for each prompt. We propose Single-Peak SPeculative decoding (SP$^2$ec), a training-free bandit algorithm that adapts the draft configuration during generation. SP$^2$ec exploits the single-peak throughput pattern observed in existing self-speculative methods, including SWIFT, KnapSpec, SpecVLM, and our simple cosSim. Our algorithmic design explicitly minimizes a novel notion of wall-time regret for each given query, corresponding to the empirical wall-time latency. Theoretical guarantee
    
[^243]: 快手摘要注意力 V2 技术报告

    Kwai Summary Attention V2 Technical Report

    [https://arxiv.org/abs/2610.18654](https://arxiv.org/abs/2610.18654)

    提出KSA V2架构，通过允许不同层设置不同的摘要压缩分块窗口，实现更灵活的KV缓存内存压缩，并解决全注意力层中摘要标记带来的额外计算开销问题。

    

    arXiv:2610.18654v1 公告类型：cross  摘要：诸如快手摘要注意力（KSA）之类的摘要标记架构，将远距离上下文压缩为每个分块中可学习的摘要标记，混合变体则将这些层与全注意力层交错排列，以保留对原始文本的精确访问。在这种混合 KSA-Full 架构中，交错插入的摘要标记在不同层之间充当流式传输的角色。尽管有效，但它也引入了一些瓶颈：(1) 不同层的摘要压缩分块必须相同，因此难以实现灵活的 KV 缓存压缩，例如无法为不同层设置不同的摘要压缩分块窗口。(2) 在全注意力层中，摘要标记引入了额外的计算成本，但该层却忽略它们，仍然支付二次方复杂度的预填充开销，因此对计算端不友好。受此启发，我们开发了 KSA V2，它不仅实现了更灵活的 KV 缓存内存压缩……

    arXiv:2610.18654v1 Announce Type: cross  Abstract: Summary-token architectures such as Kwai Summary Attention (KSA) compress distant context into learnable Summary token per chunk, and hybrid variants interleave these layers with full attention to retain exact access to the original text. In such hybrid KSA-Full architecture, the interleaving inserted summary tokens are served as a streaming flow role across different layers. Although effective, however, it also introduce some bottlenecks: (1) the different layers summary compression chunk must be same, therefore hard to reach a flexible KV Cache compression, e.g., different layer set different summary compression chunk window. (2) in the full-attention layers, the summary token introduced additional computation cost, but such layer ignore them and still pay quadratic prefilling, therefore unfriendly to the computation side. Motivated by them, we developed the KSA V2, which not only achieves a more flexible KV cache memory compression 
    
[^244]: Agentic-MatriBO：面向材料与化学设计的能力编译型智能体贝叶斯优化

    Agentic-MatriBO: Capability-Compiled Agentic Bayesian Optimization for Materials and Chemical Design

    [https://arxiv.org/abs/2610.17832](https://arxiv.org/abs/2610.17832)

    提出了能力编译型智能体贝叶斯优化框架Agentic-MatriBO，将BO组件表示为可组合的能力，由大语言模型策略提出优化配方，经编译器检查和确定性执行层验证后用于材料与化学设计。

    

    材料与化学优化必须在有限预算下，基于序列化观测来选择下一次合成、表征或高保真仿真。大语言模型能够利用任务知识提出调整方案，但组合代理模型、采集函数与搜索策略时，会受到兼容性、参数依赖性和证据要求的限制。我们提出了Agentic-MatriBO，这是一个能力编译型的智能体贝叶斯优化（BO）框架，它将BO组件和选择条件表示为可组合的能力。语言策略提出BO配方和候选计划；编译器根据注册的能力目录对提案进行检查；确定性执行层负责验证并提交候选方案。Oracle观测结果被存储在研究记忆中，并为后续轮次提供信息支持。我们在四个离线有限候选池——COF、PCE10、Buch……上与八个基线方法进行了比较。

    arXiv:2610.17832v1 Announce Type: cross  Abstract: Materials and chemical optimization must choose the next synthesis, characterization, or high-fidelity simulation from sequential observations under a limited budget. Large language models can use task knowledge to propose adjustments, but composing surrogates, acquisition functions, and search strategies is constrained by compatibility, parameter dependencies, and evidence requirements. We present Agentic-MatriBO, a capability-compiled agentic Bayesian optimization (BO) framework that represents BO components and selection conditions as composable capabilities. A language policy proposes BO recipes and candidate plans; a compiler checks proposals against the registered capability catalog; and a deterministic execution layer validates and commits candidates. Oracle observations are stored in research memory and inform later rounds. We compare Agentic-MatriBO with eight baselines on four offline finite candidate pools---COF, PCE10, Buch
    
[^245]: 线性赌博机的扰动跟随领导者算法：一般凸集上的 $\sqrt{T}$ 遗憾

    Follow the Perturbed Leader for Linear Bandits: $\sqrt{T}$ Regret over General Convex Sets

    [https://arxiv.org/abs/2610.17818](https://arxiv.org/abs/2610.17818)

    本文提出了首个基于FTPL的对抗性线性赌博机算法，通过一个适应估计奖励的扰动分布，在一般凸动作集上实现了 $\sqrt{T}$ 遗憾，改进了此前 $T^{2/3}$ 的遗憾界，同时比连续指数权重方法具有更好的计算效率和更简单的实现。

    

    我们提出了首个基于扰动跟随领导者（Follow-the-Perturbed-Leader, FTPL）的对抗性线性赌博机算法，该算法在一般凸动作集上实现了 $\sqrt{T}$ 遗憾。我们的算法仅使用线性优化预言机即可达到 $\widetilde{\mathcal{O}}(d^{5/4}\sqrt{T})$ 的遗憾；当进一步可获得最优设计探索分布时，遗憾可达到接近最优的 $\widetilde{\mathcal{O}}(d\sqrt{T})$。这改进了 Dani 和 Hayes（2006）的 FTPL 算法在一般凸集上 $T^{2/3}$ 的遗憾，并将 Lévy 等人（2026）在两个特殊动作集（超立方体和欧几里得球）上的 $\sqrt{T}$ 遗憾结果进行了推广。与之前基于连续指数权重的预言机高效线性赌博机算法相比，FTPL 方法通常具有更好的计算效率，且实现起来要简单得多。关键的算法要素是一个能适应估计奖励的扰动分布。

    arXiv:2610.17818v1 Announce Type: cross  Abstract: We give the first adversarial linear bandit algorithm based on Follow-the-Perturbed-Leader (FTPL) that achieves $\sqrt{T}$ regret over general convex action sets. Our algorithm attains $\widetilde{\mathcal{O}}(d^{5/4}\sqrt{T})$ regret using only linear optimization oracles, and the near-optimal $\widetilde{\mathcal{O}}(d\sqrt{T})$ regret when an optimal-design exploration distribution is further available. This improves on the $T^{2/3}$ regret of the FTPL algorithm of Dani and Hayes (2006) for general convex sets, and generalizes the $\sqrt{T}$ regret of L\'evy et al. (2026) over two special action sets (hypercube and Euclidean ball). Compared to previous oracle-efficient linear bandit algorithms based on continuous exponential weights, the FTPL approach generally has better computational efficiency, and is much simpler to implement.   The key algorithmic element is a perturbation distribution that adapts to the estimated rewards. Its 
    
[^246]: LeanPRM：基于Lean的自然语言证明验证过程监督

    LeanPRM: Lean-Grounded Process Supervision for Natural Language Proof Verification

    [https://arxiv.org/abs/2610.17697](https://arxiv.org/abs/2610.17697)

    LeanPRM利用Lean证明助手自动为自然语言数学证明生成步骤级正确性标签，无需人工标注即可构建包含10,634个证明、15.2万个标注步骤的过程监督数据集，解决了证明验证器训练中标签获取困难的瓶颈。

    

    如今，AI系统能够撰写冗长且困难的数学证明，而逐步验证其有效性正成为瓶颈。学习型验证器可以低成本地检查证明，但训练这样的验证器需要对每一步进行正确性标注，而这类标签对于开放式证明来说很难获得。现有方法要么依赖人工标注者（成本高昂），要么从最终答案中推导标签（而证明本身没有最终答案），且用Lean形式化每个证明需要专家投入大量精力。我们提出了LeanPRM，这是一个利用Lean证明助手为自然语言证明创建步骤级标签的框架。一个模型在没有Lean反馈的情况下，同时撰写自然语言证明及其配对的形式化证明；随后Lean对每一步进行标注，再由第二个模型过滤掉仅由形式化语法导致的失败，使负标签只标记数学本身存在的错误。LeanPRM包含10,634个证明、152K个带标签的步骤，且完全无需人工标注。

    arXiv:2610.17697v1 Announce Type: cross  Abstract: AI systems now write long and difficult mathematical proofs, and checking their step-by-step validity is becoming a bottleneck. A learned verifier could check proofs cheaply, but training one requires a correctness label on every step, and such labels are hard to obtain for open-ended proofs. Existing methods take them from human annotators, which is costly, or derive them from final answers, which proofs do not have, and formalizing each proof in Lean requires expert effort. We introduce LeanPRM, a framework that uses the Lean proof assistant to create step-level labels for natural-language proofs. A model writes a natural-language proof together with a paired formal proof, without feedback from Lean. Lean then labels every step, and a second model discards failures caused only by the formal syntax, so that the negative labels mark errors in the mathematics. LeanPRM contains 10,634 proofs with 152K labeled steps and no human annotatio
    
[^247]: LoongSage：面向前沿大语言模型的生产级、高效智能体强化学习框架（附经验证的训练方案）

    LoongSage: A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes

    [https://arxiv.org/abs/2610.16352](https://arxiv.org/abs/2610.16352)

    LoongSage是一个智能体原生、完全异步的生产级强化学习框架，基于Megatron与SGLang构建且核心仅约1.4万行代码，通过统一编排白盒与黑盒智能体的AgentFlow范式，显著降低了前沿大语言模型智能体强化学习的采用、使用与调试开销。

    

    强化学习（RL）已成为大语言模型（LLM）后训练流程中的关键阶段。随着大语言模型能力的前沿从静态推理转向智能体任务——其特征包括外部工具调用、多轮交互和长程决策——现有主流强化学习框架变得日益笨重，增加了采用、使用和调试的开销。为应对这些挑战，我们提出了LoongSage，一个智能体原生、完全异步的强化学习框架，其核心代码仅约1.4万行。LoongSage基于Megatron实现分布式训练，基于SGLang实现高吞吐量推理，将编排逻辑集中在单一控制器进程中，并将所有驻留GPU的组件封装为Ray actor。它引入了AgentFlow这一统一执行范式，可同时编排白盒与黑盒智能体。对于异步执行，（原文摘要在此处截断，后续内容缺失）

    arXiv:2610.16352v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has emerged as a critical phase in the post-training pipeline of large language models (LLMs). As the frontier of LLM capabilities shifts from static reasoning to agentic tasks--characterized by external tool invocation, multi-turn interactions, and long-horizon decision-making--existing mainstream RL frameworks have grown increasingly unwieldy, raising the overhead of adoption, usage, and debugging. To address these challenges, we present LoongSage, an agent-native, fully asynchronous RL framework with a lightweight core of roughly 14K lines of code. Built upon Megatron for distributed training and SGLang for high-throughput inference, LoongSage centralizes orchestration within a single controller process and encapsulates all GPU-resident components as Ray actors. It introduces AgentFlow, a unified execution paradigm that orchestrates both white-box and black-box agents. For asynchronous execution, 
    
[^248]: 已训练TTT层中的误差修正

    Error Correction in Trained TTT Layers

    [https://arxiv.org/abs/2610.16268](https://arxiv.org/abs/2610.16268)

    该研究剖析了TTT层的梯度更新机制，证明点积目标下不存在预测误差残差、LayerNorm目标下的预测衍生分量在发布初始化处会相互抵消，并通过实验量化了归一化预测对已训练检查点上梯度更新的实际贡献。

    

    测试时训练（TTT）层在处理输入序列的同时更新快权重记忆。然而，基于梯度的更新本身并不能说明记忆如何修订已有的关联。我们研究了已发布的TTT实现中由预测衍生出的更新的结构与经验作用。对于点积目标，我们刻画了预测误差残差的缺失，并建立了重参数化后内循环上升与下降等价的条件。对于TTT-Linear和TTT-MLP所使用的基于LayerNorm的平方误差目标，我们推导了内循环梯度，并识别出一个预测衍生分量，该分量在零ε极限下于发布的初始化处会相互抵消。随后，我们在三个已训练的检查点上测量了完整归一化预测的贡献。在保留LayerNorm雅可比矩阵的同时移除该贡献，得到的平均梯度范数比值为0.25至0.32。

    arXiv:2610.16268v2 Announce Type: replace-cross  Abstract: Test-time-training (TTT) layers update fast-weight memory while processing an input sequence. A gradient-based update, however, does not by itself establish how the memory revises an existing association. We study the structure and empirical role of prediction-derived updates in released TTT implementations. For dot-product objectives, we characterize the absence of a prediction-error residual and establish conditions under which inner-loop ascent and descent are equivalent after reparameterization. For the LayerNorm-based squared-error objective used by TTT-Linear and TTT-MLP, we derive the inner gradient and identify a prediction-derived component that cancels at the released initialization in the zero-epsilon limit. We then measure the full normalized prediction's contribution on three trained checkpoints. Removing this contribution while retaining the LayerNorm Jacobian gives a mean gradient norm ratio of 0.25 to 0.32 acros
    
[^249]: 面向多胞形距离的受保护块转移：显式常数下的线性收敛与用于SVM训练的间隙认证筛选

    Guarded Block Transfers for Polytope Distance: Linear Convergence with Explicit Constants and Gap-Certified Screening for SVM Training

    [https://arxiv.org/abs/2610.15829](https://arxiv.org/abs/2610.15829)

    该论文提出受保护块转移方法，将k次MDM转移聚合为一次受保护的精确线搜索，在Minkowski差多胞形上证明了具有显式常数的全局线性收敛，并利用TA证书实现了SVM训练的间隙安全筛选。

    

    硬间隔SVM训练等价于两个凸包之间的距离问题，三角形算法以认证的对偶间隙求解该问题。我们提出“受保护块转移”：将每次扫描中的k次Mitchell–Demyanov–Malozemov（MDM）转移聚合为一次精确线搜索，通过与单次最优转移的比较进行保护（该比较的代价与点数n无关），并在主导转移因容量被裁剪时以-away步替换。在Minkowski差多胞形上，目标函数为1-强凸，我们证明了具有显式常数的全局线性收敛，且无需组合式的swap-step界；每步收缩率与k无关，仅drop-count界随k增长，因此聚合被证明是安全的，而扫描次数的节省是通过测量而非证明得到的。由于TA证书是Frank–Wolfe间隙除以当前距离，因此间隙安全筛选得以适用，并且我们对……（原文摘要在此处截断）

    arXiv:2610.15829v2 Announce Type: replace  Abstract: Hard-margin SVM training is the distance problem between two convex hulls, which the Triangle Algorithm (TA) solves with a certified duality gap. We propose \emph{guarded block transfers}: $k$ Mitchell--Demyanov--Malozemov (MDM) transfers per scan are aggregated into one exact line search, guarded by a comparison with the single best transfer at a cost independent of the number of points $n$, and replaced by an away step when the leading transfer is capacity-clipped. On the Minkowski-difference polytope, where the objective is $1$-strongly convex, we prove global linear convergence with explicit constants and no combinatorial swap-step bounds; the per-step contraction is $k$-independent and only the drop-count bound grows with $k$, so aggregation is provably safe while the saving in scans is measured, not proved. Since the TA certificate is the Frank--Wolfe gap divided by the current distance, gap-safe screening applies, and we bound
    
[^250]: DisCo代数：为什么分解与组合并非互逆运算

    DisCo Algebra: Why Disintegration and Combination Are Not Inverses

    [https://arxiv.org/abs/2610.15006](https://arxiv.org/abs/2610.15006)

    本文提出DisCo代数框架，用对称余操作刻画联合概率的分解、用非单位部分操作刻画组合，证明“组合∘分解”在可允许输入上是恒等操作，而“分解∘组合”则是一个有信息损失的重正化算子，从而揭示了分解与组合并非互逆。

    

    我们提出了DisCo代数——一个建立在裸族 $\mathcal{P}_n=\mathrm{Prob}([0,1]^n)$ 之上的联合概率组合式框架，由一个对称余操作（disintegration，即分解）和一个非单位部分对称操作（combination，即限制在分解像上的组合）构成。该框架不需要概率积分变换，不需要外部汇总函数，也不需要用条件核对对象进行富集。分解一侧使用分块联合累积分布函数（CDF）作为内在的、由测度决定的一维摘要，从而实现有损压缩；组合一侧读取相同的摘要，并通过正则条件测度进行重构，在可允许输入上实现无损嵌入。我们证明：先组合后分解在可允许输入上是恒等操作，但先分解后组合则不是——后者定义了一个重正化算子 $R_{\mathcal{I}}=\text{Combine}\circ\text{Disintegrate}$。该算子的信息损失……

    arXiv:2610.15006v2 Announce Type: replace-cross  Abstract: We present DisCo algebra---a compositional framework for joint probability built from a symmetric cooperad (disintegration) and a non-unital partial symmetric operad (combination restricted to the image of disintegration) on the bare family $\mathcal{P}_n=\mathrm{Prob}([0,1]^n)$. No probability integral transform, no external summary functions, no enrichment of objects with conditional kernels. The disintegration side uses block-wise joint CDFs as intrinsic, measure-determined 1D summaries, yielding a lossy compression. The combination side reads the same summaries and reconstructs via regular conditional measures, yielding a lossless embedding on admissible inputs. We prove that combination followed by disintegration is the identity on admissible inputs, but disintegration followed by combination is not---it defines a renormalization operator $R_{\mathcal{I}}=\text{Combine}\circ\text{Disintegrate}$. The information loss of thi
    
[^251]: 面向方面情感四元组预测的依存关系推理

    Dependency Reasoning for Aspect Sentiment Quadruple Prediction

    [https://arxiv.org/abs/2610.14873](https://arxiv.org/abs/2610.14873)

    该论文提出DRIFT方法，将依存句法树线性化后作为显式关系上下文直接注入LLM提示中，并结合检索增强的上下文学习与语法感知裁决机制，在方面情感四元组预测任务上取得了最优性能。

    

    方面情感四元组预测（ASQP）本质上是一个关系推理问题：识别“方面-观点-类别-情感”四元组需要对结构化依赖关系进行建模，而不仅仅是顺序上的共现。现有的基于大语言模型（LLM）的方法通过基于检索的示例选择或多阶段训练等方式间接引入句法信息，而非将其作为统一的、单次生成时的显式上下文。我们提出了DRIFT，该方法将依存句法分析树线性化，并作为显式的关系上下文直接注入LLM提示中，使模型能够对方面与观点之间的依赖关系进行推理，而不是仅凭词元邻近性来推断。DRIFT还进一步集成了检索增强的上下文学习以及一种语法感知的、基于LLM的裁决机制，该机制在跨度边界变化的情况下对精确匹配评估形成补充。在SemEval Rest15、Rest16和ACOS数据集上，DRIFT取得了最强的性能。

    arXiv:2610.14873v2 Announce Type: replace  Abstract: Aspect Sentiment Quadruple Prediction (ASQP) is fundamentally a relational reasoning problem: identifying aspect-opinion-category-sentiment quadruples requires modeling structured dependencies, not merely sequential co-occurrence. Existing LLM-based approaches incorporate syntactic information indirectly, via retrieval-based demonstration selection or multi-stage training, rather than as unified, single-pass generation-time context. We present DRIFT, which linearizes dependency parse trees and injects them directly into LLM prompts as explicit relational context, enabling the model to reason over aspect-opinion dependencies rather than infer them from token proximity alone. DRIFT further integrates retrieval-augmented in-context learning and a syntax-aware LLM-based adjudication mechanism that complements exact-match evaluation under span-boundary variation. On SemEval Rest15, Rest16, and ACOS, DRIFT achieves the strongest performanc
    
[^252]: 先验数据拟合网络能多好地摊销贝叶斯推断？

    How Well Do Prior-Data Fitted Networks Amortize Bayesian Inference?

    [https://arxiv.org/abs/2610.13764](https://arxiv.org/abs/2610.13764)

    本文将先验数据拟合网络（PFNs）的部署误差分解为先验、上下文和摊销误差，发现PFNs虽能很好地恢复目标后验预测分布，但在罕见设置下仍存在残余摊销误差，而贝叶斯模型平均方法可有效减少这种不确定性并提升预测性能。

    

    先验数据拟合网络（PFNs）被训练用于近似合成先验的贝叶斯后验预测分布（PPD），然而关于其近似质量的研究结果却出人意料地稀少。从经验风险最小化的视角出发，我们将PFNs的部署误差分解为先验误差、上下文误差和摊销误差，并在现实先验下相对于参考后验预测分布来度量摊销误差。结果表明，PFNs通常能很好地恢复目标后验预测分布。然而，我们发现部分残余摊销误差仍然存在，尤其是在先验预训练分布中的罕见设置下，即使扩大模型规模也无法消除，这表现为独立预训练的PFNs之间的分歧，并在推理时给预测器留下不确定性。作为补救措施，我们研究了多种类型的贝叶斯模型平均方法，发现这些方法能够持续减少这种不确定性并改善预测性能。

    arXiv:2610.13764v1 Announce Type: cross  Abstract: Prior-data fitted networks (PFNs) are trained to approximate the Bayesian posterior predictive distribution (PPD) of a synthetic prior, yet results about the approximation quality are surprisingly thin. Taking an empirical risk minimization perspective, we decompose the deployment error of PFNs into prior, context, and amortization error, and measure the latter against a reference PPD under a realistic prior. Results show that PFNs generally recover the targeted PPD very well. Nevertheless, we find that parts of the residual amortization error persist, especially in \textit{rare} settings under the prior pretraining distribution, even when scaling the model size, which manifests in disagreement between independently pretrained PFNs and leaves uncertainty about the predictor at inference time. As a remedy, we study multiple types of Bayesian model averages and find that these can consistently reduce this uncertainty and improve predicti
    
[^253]: SelectArena：面向真实世界表格数据的现代特征选择基准测试

    SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data

    [https://arxiv.org/abs/2610.13644](https://arxiv.org/abs/2610.13644)

    该论文提出了SelectArena——一个基于真实世界表格数据的现代特征选择基准测试，通过比较15种广泛使用的特征选择方法，弥补了现有基准依赖过时模型、临时评估协议和人工数据而无法指导实际应用的不足。

    

    从表格数据集中选择特征子集（列）是现实应用中普遍存在的挑战。行业专家希望最小化模型训练成本，气候科学家需要可解释的因素子集来理解极端天气事件，而临床医生由于财务和时间限制需要在各类诊断测试中做出选择。当从业者查阅科学文献以确定应使用哪种特征选择方法时，他们面对的是一个缺乏代表性的基准测试现状。尽管相关研究众多，现有的特征选择基准测试所能提供的指导却十分有限，因为它们依赖于过时的模型、临时设计的评估协议，或无法反映真实问题的人工生成数据。为了弥合科学基准测试与现实应用之间的差距，我们推出了 SelectArena，一个现代化的特征选择基准测试。我们比较了15种广泛使用的特征选择方法……

    arXiv:2610.13644v1 Announce Type: cross  Abstract: Selecting a subset of features (columns) from a tabular dataset is a ubiquitous challenge in real-world applications. Industry experts seek to minimize model training costs, climate scientists require an interpretable subset of factors to understand extreme weather events, and clinicians need to choose among diagnostic tests due to financial and time constraints. When looking at the scientific literature to determine which feature selection method to use, practitioners across domains face an unrepresentative landscape of benchmarking. Despite extensive research, existing feature selection benchmarks provide limited guidance because they rely on outdated models, ad hoc evaluation protocols, or artificially generated data that do not reflect real-world problems. To close the gap between scientific benchmarking and real-world applications, we introduce SelectArena, a modern benchmark for feature selection. We compare 15 widely used featur
    
[^254]: AdaptLSTM：面向分布漂移下云工作负载预测的高效自适应在线学习

    AdaptLSTM: Efficient Adaptive Online Learning for Cloud Workload Forecasting under Distribution Drift

    [https://arxiv.org/abs/2610.12265](https://arxiv.org/abs/2610.12265)

    AdaptLSTM 通过验证校准阈值检测分布漂移并进行选择性针对性更新，以约 20% 的计算成本实现接近朴素在线学习的预测精度，大幅提升云工作负载预测的效率。

    

    准确的工作负载预测对于网络规模云服务的弹性资源供给至关重要，然而由热门内容、产品发布和用户行为引发的分布漂移会迅速降低离线训练模型的性能。朴素的在线学习虽然能恢复预测精度，但会带来难以承受的每步计算成本。我们提出了 AdaptLSTM，一种自适应在线学习框架，它通过验证集校准的阈值检测漂移，并施加选择性的、有针对性的更新。在阿里巴巴机器轨迹数据集上，AdaptLSTM 以 20% 的成本恢复了朴素在线学习 54% 的性能提升（2.7 倍效率，10 个随机种子下 p=0.002）。在波动性更强的容器轨迹数据集上，它以 20% 的成本实现了 96% 的性能提升（4.8 倍效率，相比静态模型 MAE 降低 75%）。与经典漂移检测器（ADWIN、DDM、Page-Hinkley）在回归尺度的误差流上无法触发不同，AdaptLSTM 在 301 步中触发了 42 次，并优于同等计算预算的基线方法。墙上时钟时间（原文在此处截断）。

    arXiv:2610.12265v1 Announce Type: new  Abstract: Accurate workload forecasting is critical for elastic resource provisioning in web-scale cloud services, where distribution shifts driven by viral content, product launches, and user behavior degrade offline-trained models rapidly. Naive online learning recovers accuracy but incurs prohibitive per-step compute cost. We propose AdaptLSTM, an adaptive online framework that detects drift via validation-calibrated thresholds and applies selective, targeted updates. On the Alibaba Machine Trace, AdaptLSTM recovers 54\% of Naive Online's improvement at 20\% cost ($2.7\times$ efficiency, $p=0.002$ over 10 seeds). On the more volatile Container Trace, it achieves 96\% at 20\% cost ($4.8\times$ efficiency, $+75\%$ MAE reduction over Static). Unlike classical drift detectors (ADWIN, DDM, Page-Hinkley) which fail to trigger on regression-scale error streams, AdaptLSTM fires 42 times over 301 steps and outperforms matched-budget baselines. Wall-cloc
    
[^255]: 伯努利流模型：面向二进制数据的自洽生成建模

    Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data

    [https://arxiv.org/abs/2610.11362](https://arxiv.org/abs/2610.11362)

    提出伯努利流模型（BFM），通过构建数据分布与纯噪声之间统一的连续全局伯努利概率流路径，克服了传统二值扩散模型在低函数评估次数下因单步似然近似导致样本质量严重下降的问题，无需蒸馏或额外训练即可实现高效的二值数据生成。

    

    二值扩散模型通常需要大量的函数评估次数（NFE）才能生成高质量样本，使得实际推理的计算成本高昂。在不使用蒸馏或额外训练的前提下，降低NFE同时保持样本质量仍然是一个重大挑战。现有的二值扩散模型定义了离散的单步前向路径，然后再推导反向后验分布。在需要跨步采样的低NFE场景中，它们用单步似然转移来近似真实的多步似然，这严重降低了样本质量。为了解决这一根本性局限，并将生成动力学与固定的离散时间步解耦，我们提出了伯努利流模型（BFM）。BFM不依赖于顺序式的单步马尔可夫扩散链，而是定义了一条连接数据分布与纯噪声的统一连续全局伯努利概率流路径，并从中解析推导出……（摘要在此处截断）

    arXiv:2610.11362v1 Announce Type: new  Abstract: Binary diffusion models typically require a large number of function evaluations (NFEs) to generate high-quality samples, making practical inference computationally expensive. Reducing NFEs while preserving sample quality without distillation or additional training remains a significant challenge. Existing binary diffusion models define a discrete one-step forward path and then derive the reverse posterior. In low-NFE settings requiring cross-step sampling, they approximate the true multi-step likelihood with a single-step likelihood transition, which severely degrades sample quality. To address this fundamental limitation and decouple the generative dynamics from fixed discrete time steps, we propose Bernoulli Flow Models (BFM). Rather than relying on sequential one-step Markov diffusion chains, BFM defines a unified continuous global Bernoulli probability flow path between data distributions and pure noise, from which we derive analyti
    
[^256]: 神经PDE求解器中学习状态的可迁移性

    Transferability of Learned States in Neural PDE Solvers

    [https://arxiv.org/abs/2610.10972](https://arxiv.org/abs/2610.10972)

    提出“复用契约”评估框架，将神经PDE求解器中学习状态的迁移收益与解精度和计算成本解耦，并实证发现固定预测器的收益会随校正算法不同而反转。

    

    评估神经PDE求解器中的有效复用是具有挑战性的：最终精度可能同时反映源任务的学习效果和目标时刻的计算投入。我们的“复用契约”通过成对状态比较、匹配的目标信息和计算预算以及成本核算，将解的精度、学习贡献和数值效用区分开来。一项文献审计从12篇论文中提取了18条针对特定版本的协议记录，记录了所保留的状态、目标时刻的资源以及报告的对照条件。对于一个固定的线性系统和残差容限，我们构造了两个具有相同解误差、能量误差和残差范数的初始猜测，它们以不同的共轭梯度（CG）迭代次数达到相同的解。在240条源训练轨迹、两个线性PDE族、傅里叶神经算子和卷积网络的实验中，固定预测器带来的收益在不同的校正算法之间发生逆转。

    arXiv:2610.10972v1 Announce Type: new  Abstract: Assessing useful reuse in neural PDE solvers is challenging: final accuracy can reflect source learning and target-time computation. Our reuse contract separates solution accuracy, learning contribution, and numerical utility through paired state comparisons, matched target information and budgets, and cost accounting. A literature audit extracts 18 version-specific protocol records from 12 papers, documenting retained states, target-time resources, and reported controls. For a fixed linear system and residual tolerance, we construct two initial guesses with identical solution-error, energy-error, and residual norms, reaching the same solution with different conjugate-gradient (CG) iteration counts. Across 240 source-training trajectories, two linear PDE families, Fourier neural operators and convolutional networks, a fixed predictor's benefit reverses across correction algorithms. Among pairs with both relative prediction errors less th
    
[^257]: VC学习的最优信息复杂度

    The optimal information complexity of VC learning

    [https://arxiv.org/abs/2610.10600](https://arxiv.org/abs/2610.10600)

    本文通过构造一个eCMI为O(d)阶的随机化“5个基学习器多数投票”学习算法，首次经由CMI信息论分析框架恢复了VC类学习的最优PAC泛化保证。

    

    Steinke和Zakynthinou（2020）提出了条件互信息（CMI）框架，利用依赖于算法的信息论量来分析学习算法的信息复杂度。我们研究了其中一种量——评估条件互信息。一个有趣的问题是：能否通过基于CMI的依赖于算法的分析，恢复VC类的最优PAC保证？我们证明了这是可能的：在可实现情形下，我们构造了一个eCMI为O(d)阶的学习算法，从而恢复了该最优保证，其中d是概念类的VC维。特别地，我们的算法是一种随机化的“5个基学习器多数投票”算法，并具有最优的期望泛化保证。

    arXiv:2610.10600v1 Announce Type: cross  Abstract: Steinke and Zakynthinou(2020) introduces the Conditional Mutual Information (CMI) framework of analyzing the information complexity of learning algorithms based on algorithm-dependent information-theoretic quantities. We study one of these quantities, the evaluated Conditional Mutual Information (eCMI). It has been an interesting question whether the optimal PAC guarantee for VC classes can be recovered from the algorithm-dependent analyses via CMI. And we show that it is possible to recover this guarantee by constructing a learning algorithm whose eCMI is of order O(d) in the realizable case, where d is the VC-dimension of the concept class. Specially, our algorithm is a randomized Majority-of-5 base learners with optimal in-expectation generalization guarantee.
    
[^258]: 具有RBF核的高斯过程极大似然估计的精确渐近理论

    Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with an RBF Kernel

    [https://arxiv.org/abs/2610.10080](https://arxiv.org/abs/2610.10080)

    本文针对基于RBF核的高斯过程，在固定域渐近框架下建立了极大似然估计的精确渐近理论，攻克了密集采样强相关性与协方差矩阵非线性依赖带来的长期理论难题。

    

    高斯过程（GP）广泛应用于机器学习、空间统计、时间序列分析、优化、贝叶斯统计以及各类科学应用中。高斯过程模型的核心组成部分是其核函数，通常通过参数族来指定。其中最广泛使用的选择之一是径向基函数（RBF）核，也称为平方指数核或高斯核，这得益于其形式简单、光滑且灵活的特性。在实践中，核参数通常通过极大似然估计（MLE）来估计，这也是标准高斯过程软件所采用的标准方法。尽管应用如此广泛，即便对于RBF核，极大似然估计量在固定域渐近框架下的渐近行为仍然鲜为人知。其主要困难源于密集采样观测之间日益增强的相关性，以及协方差矩阵对核参数的非线性依赖。在本文中……

    arXiv:2610.10080v1 Announce Type: cross  Abstract: Gaussian processes (GPs) are widely used across machine learning, spatial statistics, time-series analysis, optimization, Bayesian statistics, and scientific applications. A central component of a GP model is its kernel, which is typically specified through a parametric family. Among the most widely used choices is the radial basis function (RBF), also known as the squared exponential or Gaussian kernel, owing to its simple form, smoothness, and flexibility. In practice, the kernel parameters are routinely estimated by the maximum likelihood estimators (MLEs), as implemented by standard GP software. Despite this widespread use, the asymptotic behavior of the MLEs remains poorly understood under fixed-domain asymptotics, even for the RBF kernel. The main difficulty arises from the increasingly strong dependence among densely sampled observations and the nonlinear dependence of the covariance matrix on the kernel parameters. In this pape
    
[^259]: 更精简的Transformer可以轻松学会聚类

    Leaner Transformers Can Easily Learn to Cluster

    [https://arxiv.org/abs/2610.09760](https://arxiv.org/abs/2610.09760)

    本文提出了一种嵌入维度仅需 d+⌈log₂k⌉ 但表达能力不变的更精简Transformer来执行k均值聚类的Lloyd算法，并系统刻画了训练Transformer学习聚类算法时影响收敛性与泛化能力的关键因素。

    

    Transformer具备上下文学习能力，某些已知的学习算法可以在模型的前向传播过程中执行。近期的研究表明，Transformer可以精确执行Lloyd算法，对d维空间中的n个点进行k均值聚类，其嵌入维度为 d_emb = d+k（因此需要大小为(d+k)²的注意力投影矩阵）。在本工作中，我们在这一结果的基础上进行了以下拓展：首先，我们提出了一种表达能力相同但规模更小的Transformer，它能以嵌入维度 d_emb = (d + ⌈log₂ k⌉) 执行Lloyd算法。其次，我们在给定聚类任务分布的条件下训练这些Transformer学习聚类算法，并从理论上刻画和通过实验验证了影响基于随机梯度的学习算法收敛性和分布内泛化能力的因素。最后，我们探究了一般性的聚类能力（摘要此处被截断）。

    arXiv:2610.09760v1 Announce Type: new  Abstract: Transformers have in-context learning capabilities, where some known learning algorithms can be executed in the forward pass through the model. Recent work shows that transformers can exactly perform Lloyd's algorithm for $k$-means clustering with $n$ points in $d$ dimensions with an embedding size $d_{\textsf{emb}} = d+k$ (thus, requiring attention projection matrices of size $(d+k)^2$). In this work, we build upon this result in the following ways: First, we present an equally expressive but smaller transformer that executes Lloyd's algorithm with embedding size $d_{\textsf{emb}} = (d + \lceil \log_2 k \rceil)$. Next, we train these transformers to learn the clustering algorithms given a distribution of clustering tasks, and theoretically characterize and empirically validate the factors affecting the convergence and in-distribution generalization of learning algorithms based on stochastic gradients. Finally, we probe the general clust
    
[^260]: MS-ECG-FM：基于多源对比学习迈向更通用的心电图健康监测基础模型

    MS-ECG-FM: Towards a More Universal Electrocardiogram Foundation Model for Health Monitoring using Multi-source Contrastive Learning

    [https://arxiv.org/abs/2610.07662](https://arxiv.org/abs/2610.07662)

    MS-ECG-FM通过对齐心电图、超声心动图、放射学和出院报告等多种临床记录进行多源对比学习训练，突破了仅依赖解读报告作为监督的局限，在包括少导联配置在内的所有心电图检测任务上全面超越现有方法。

    

    心电图记录心脏的电活动，通过检测心功能异常来辅助诊断。心电图基础模型已展现出有前景的效果，但其局限性在于仅将心电图解读报告作为唯一的监督信号。由于解读报告只包含临床医生常规识别的波形信息子集，这限制了表示学习，使其忽略了心电图中更广泛的诊断信号。我们提出了一种新的心电图基础模型——MS-ECG-FM——它通过与多种不同类型的临床记录（包括心电图、超声心动图、放射学及出院报告）进行对比对齐来训练。我们在扩展的心电图检测基准集上对MS-ECG-FM进行了评估，结果表明其在心电图可检测的全部疾病范围内全面优于现有方法，包括在少导联配置下的表现。

    arXiv:2610.07662v1 Announce Type: new  Abstract: Electrocardiography (ECG) records the electrical activity of the heart, aiding diagnosis by detecting abnormalities in cardiac function. ECG foundation models have demonstrated promising results, but are limited by a reliance on ECG interpretation reports as their sole supervision. Because interpretation reports only capture the subset of waveform information routinely recognized by clinicians, this constrains representation learning to overlook the broader diagnostic signals present in ECG. We introduce a new ECG foundation model --- MS-ECG-FM --- that is trained through contrastive alignment to multiple distinct clinical note types, including ECG, echocardiography, radiology, and discharge reports. We evaluate MS-ECG-FM on an extended set of ECG detection benchmarks, showing that it comprehensively outperforms existing methods on the full span of conditions that ECG can detect, including in reduced-lead configurations. Different report
    
[^261]: 可学习谱激活

    Learnable Spectral Activations

    [https://arxiv.org/abs/2610.07419](https://arxiv.org/abs/2610.07419)

    提出可学习谱激活（LSA），用谐波振幅可学习的残差截断傅里叶级数替代固定激活函数，将特征选择与频谱整形解耦为两条独立梯度更新的通路，从而提升隐式神经表示对局部化和空间变化信号的多谐波组合能力。

    

    隐式神经表示（INR）由其输入编码和激活函数所诱导的频谱结构塑造。现有方法主要通过修改网络可用的频率来改进拟合效果，例如采用坐标编码或周期非线性函数。然而，频率可及性并非唯一的瓶颈：具有局部化或空间变化结构的信号，需要网络能够高效地将多个频率组合成多谐波的内部响应。我们提出可学习谱激活（LSA），用残差截断傅里叶级数替代固定的神经元级非线性函数，其谐波振幅在训练过程中进行学习。LSA并不扩展渐近函数类，而是改变表示的分解方式：线性权重负责选择特征，而激活系数控制频谱整形，二者通过相互独立的梯度进行更新。由于……（原文摘要在此处截断）

    arXiv:2610.07419v1 Announce Type: new  Abstract: Implicit neural representations (INRs) are shaped by the spectral structure induced by their input encodings and activation functions. Existing methods improve fitting primarily by modifying which frequencies are available to the network, through coordinate encodings or periodic nonlinearities. However, frequency access is not the only bottleneck: signals with localized or spatially varying structure require the network to efficiently compose frequencies into multi-harmonic internal responses. We introduce learnable spectral activations (LSA), which replace fixed neuron-level nonlinearities with a residual truncated Fourier series whose harmonic amplitudes are learned during training. LSA does not expand the asymptotic function class. Instead, it changes the factorization of the representation: linear weights select features while activation coefficients control spectral shaping, and the two are updated by separate gradients. Because the
    
[^262]: RealtimeWAM：一步异步世界动作模型

    RealtimeWAM: One-Step Asynchronous World Action Models

    [https://arxiv.org/abs/2610.06617](https://arxiv.org/abs/2610.06617)

    提出RealtimeWAM，通过教师锚定一致性蒸馏（TACD）实现一步动作生成，并采用异步推理消除视频专家与动作专家之间的顺序等待，显著提升世界动作模型的推理效率。

    

    世界动作模型引入视频生成骨干网络的视觉表示来引导动作预测。近期的高效WAM采用混合Transformer（MoT）架构，只需计算一次视频表示即可供动作专家复用。然而，专家内迭代（即多步动作去噪）和专家间等待（即视频专家与动作专家的顺序执行）仍然限制了推理效率。为此，我们提出RealtimeWAM，一种极其高效的WAM变体，具有一步动作生成和异步推理能力，解决了上述两个瓶颈。为减少专家内迭代，我们提出教师锚定一致性蒸馏（TACD）来解决局部-全局误差鸿沟问题：仅靠较低的局部一致性误差并不能保证最终动作的准确性。TACD利用冻结教师模型多步展开终点的显式监督来补充局部一致性约束，

    arXiv:2610.06617v2 Announce Type: replace-cross  Abstract: World Action Models (WAMs) incorporate visual representations from video generation backbones to guide action prediction. Recent efficient WAMs adopt Mixture-of-Transformers (MoT) architectures and compute video representations once for reuse by the action expert. However, intra-expert iteration (\ie, multi-step action denoising) and inter-expert waiting (\ie, sequential execution of the video and action experts) still limit inference efficiency. To this end, we present RealtimeWAM, an extremely efficient WAM variant with one-step action generation and asynchronous inference, addressing these two bottlenecks. To reduce intra-expert iteration, we propose Teacher-Anchored Consistency Distillation (TACD) to address a local-global error gap: low local consistency error alone does not guarantee accurate final actions. TACD supplements local consistency with explicit supervision from the frozen teacher's multi-step rollout endpoint, 
    
[^263]: 关于稀疏注意力中信息损失与泛化之间的权衡

    On the Trade-off Between Information Loss and Generalization in Sparse Attention

    [https://arxiv.org/abs/2610.04424](https://arxiv.org/abs/2610.04424)

    本文通过Jensen-Shannon散度对稀疏注意力机制的信息损失进行理论刻画，并系统分析了信息损失与模型泛化能力之间的权衡关系。

    

    为了缓解Transformer的二次复杂度瓶颈，稀疏注意力已成为一项关键技术。尽管稀疏Transformer在实践中取得了广泛的实证成功，但对其理论理解仍然零散。特别是，两个基本问题仍不清楚：(1) 稀疏化如何影响注意力机制的信息保真度？(2) 这种信息损失如何与模型的泛化行为相互作用？为弥合这一差距，本文对稀疏注意力机制的Jensen-Shannon（JS）散度和泛化差距进行了系统性分析。具体而言，我们首先通过JS散度刻画近似误差，并对截断质量alpha进行了基于顺序统计量的集中性分析——其中注意力分数被假定为独立同分布的sub-Gaussian随机变量。

    arXiv:2610.04424v2 Announce Type: replace  Abstract: To mitigate the quadratic complexity bottleneck of the Transformer, sparse attention has emerged as a pivotal technology. Despite the extensive empirical success of sparse Transformers, the theoretical understanding of sparse attention remains fragmented. In particular, two fundamental questions remain unclear: (1) How does sparsification affect the information fidelity of attention mechanisms? (2) How does this information loss interact with the generalization behavior of the model? To bridge this gap, this paper proposes a systematic analysis of the Jensen-Shannon (JS) divergence and of the generalization gap of sparse attention mechanisms. Specifically, we first characterize the approximation error via the JS divergence. Through an order-statistics-based concentration analysis of the truncation mass alpha --- where the attention scores are assumed to be independent and identically distributed sub-Gaussian random variables with par
    
[^264]: 分数并非结构：大脑对齐与跨语言迁移

    The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer

    [https://arxiv.org/abs/2610.03827](https://arxiv.org/abs/2610.03827)

    相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。

    

    相似性分数常被用作模型与大脑或跨语言间共享结构的证据。我们探讨当结构被移除或测量工具失效时，这样的分数实际反映的是什么，并在两个场景中进行了验证。在十七种语言中，语法性探针在语言距离越远时迁移效果越差（r = -0.66），但探针自身的准确率也沿同一轴线下降，在四种语言中处于随机水平（这四种语言是距离最远的五种中的四种）。剔除这些语言会使已解释方差减半，同时由于也缩小了距离范围，该设计无法判断梯度中有多少来自测量工具本身。将272个语言对视为独立样本时，导向效应的p值为0.0006；但以十七种语言为单位时，该效应不显著（p = 0.155）。在大脑对齐方面，一个训练目标将语言模型与fMRI响应的相似性从0.10提升到0.34（可靠性上限为0.54），然而

    arXiv:2610.03827v2 Announce Type: replace-cross  Abstract: Similarity scores are often offered as evidence that a model shares structure with the brain or across languages. We ask what such a score reads when that structure is removed, or when the instrument measuring it does not work, in two settings. Across seventeen languages, a grammaticality probe transfers worse between more distant languages (r = -0.66). But the probe's own accuracy falls along the same axis and is at chance in four languages, four of the five most distant. Dropping them halves the explained variance, and because it also narrows the range of distances, the design cannot say how much of the gradient is the instrument. Counting the 272 language pairs as independent gives p = 0.0006 for a steering effect that is null when the seventeen languages are the unit (p = 0.155). In brain alignment, a training objective raises a language model's similarity to fMRI responses from 0.10 to 0.34 (reliability ceiling 0.54), yet 
    
[^265]: 探索面向空间应用的深度神经网络中结构化剪枝与容错性之间的权衡

    Exploring the Trade-Off Between Structured Pruning and Fault Tolerance in Deep Neural Networks for Space Applications

    [https://arxiv.org/abs/2610.03117](https://arxiv.org/abs/2610.03117)

    本研究通过故障注入实验发现，深度神经网络的结构化剪枝虽然因减少冗余而提高了单次推理对单粒子翻转故障的敏感性，但更短的执行时间降低了遭遇故障的概率，从而有效平衡了这一负面影响。

    

    深度神经网络（DNN）由于其信息的分布式表示，本质上对比特级故障具有一定程度的鲁棒性。随着模型宽度的增加，信息变得更加分散，理论上降低了任何单个比特故障的影响。在本文中，我们实证研究了模型宽度与单粒子翻转（SEU）鲁棒性之间的关系。我们进行了一项全面的实验：基线模型经过迭代结构化剪枝以减小宽度，同时尽可能保持任务性能。在每个剪枝阶段，我们开展针对性的故障注入实验，以评估模型在模拟比特翻转场景下的性能。我们的结果表明，尽管结构化剪枝通过减少冗余增加了每次推理对故障的敏感性，但这一效应被更短的执行时间有效抵消，因为执行时间越短，遇到故障的概率就越低。

    arXiv:2610.03117v1 Announce Type: new  Abstract: Deep Neural Networks (DNNs) inherently exhibit a degree of robustness to bit-level faults due to their distributed representation of information. As a model increases in width, this information becomes more dispersed, theoretically reducing the impact of any single bit fault. In this paper, we empirically investigate the relationship between model width and robustness to Single Event Upsets (SEUs). We conduct a comprehensive experiment in which baseline models undergo iterative structured pruning to reduce their width while preserving task performance as much as possible. At each pruning stage, we run a targeted fault-injection campaign to evaluate the model's performance under simulated bit-flip scenarios. Our results show that, although structured pruning increases per-inference sensitivity to faults by reducing redundancy, this effect is effectively counterbalanced by shorter execution time, which lowers the probability of encounterin
    
[^266]: FSPO：面向预算约束的大语言模型强化学习后训练的策略一致风险与帕累托可行控制

    FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training

    [https://arxiv.org/abs/2610.02828](https://arxiv.org/abs/2610.02828)

    FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。

    

    自适应的大语言模型强化学习后训练会在训练过程中在线调整多个训练执行器，包括 rollout 温度、组大小、裁剪、KL 正则化、验证器分配以及更新预算。目前有三个相互耦合的问题尚未解决：从行为轨迹训练得到的未来风险模型未必能准确估计将要部署的控制器所引发的风险；基于已记录的状态-动作对校准的分数在经过选择性动作选择后可能出现校准失准；独立的单资源最小成本通常无法保证多资源延续的可行性。我们提出 FSPO，一种面向预算约束的大语言模型强化学习后训练的反馈状态控制器，能够联合解决这些问题。FSPO 学习一个策略一致的风险前瞻（risk-to-go）模型，其 Bellman 目标遵循与未来决策所用的同一个冻结控制器，并同时学习一个长程效用模型。决策条件化轨迹校准（DCTC）对风险进行校准……

    arXiv:2610.02828v1 Announce Type: new  Abstract: Adaptive LLM reinforcement-learning post-training changes multiple training actuators online, including rollout temperature, group size, clipping, KL regularization, verifier allocation, and update budget. Three coupled issues remain unresolved. A future-risk model trained from behavior trajectories need not estimate the risk induced by the controller that will be deployed; a score calibrated on logged state-action pairs can become miscalibrated after selective action choice; and independent per-resource minimum costs do not in general certify a feasible multi-resource continuation. We introduce FSPO, a feedback-state controller for budgeted LLM RL post-training that addresses these issues jointly. FSPO learns a policy-consistent risk-to-go model whose Bellman target follows the same frozen controller used for future decisions, together with a long-horizon utility model. Decision-conditioned trajectory calibration (DCTC) calibrates risk 
    
[^267]: ROUTEAUDIT：面向预算受限多验证器路由的交互感知识别方法

    ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing

    [https://arxiv.org/abs/2610.02808](https://arxiv.org/abs/2610.02808)

    ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。

    

    自适应多验证器系统通常通过端点的质量-成本差距进行比较，即使验证器目录、可用性、资源核算、信息过滤或评分器会随策略发生变化。我们将验证器路由形式化为一个契约条件化的识别问题。该契约记录了请求支持、验证器目录、实际可用性、资源核算、在线过滤以及轨迹后评分；一个匹配的路由对比仅改变策略坐标。ROUTEAUDIT为该契约增加了三个可度量的对象：契约格在所有可容许的桥接顺序上对坐标增量取平均，并报告由此得到的归因及其路径敏感性；策略无关的响应带在自适应策略揭示不同观测时识别成对的顺序对比；对于不完整的匹配，请求级边界利用仍然可观测的潜在结果，给出紧致的有限（摘要在此处截断）。

    arXiv:2610.02808v1 Announce Type: new  Abstract: Adaptive multi-verifier systems are commonly compared through endpoint quality-cost gaps, even when the verifier catalog, availability, accounting, information filtration, or scorer changes with the policy. We formulate verifier routing as a contract-conditioned identification problem. The contract records request support, verifier catalog, realized availability, resource accounting, online filtration, and post-trace scoring; a matched route contrast changes only the policy coordinate. ROUTEAUDIT adds three measurable objects to this contract. A contract lattice averages coordinate increments over every admissible bridge order and reports the resulting attribution together with its path sensitivity. A policy-independent response tape identifies paired sequential contrasts when adaptive policies reveal different observations. For incomplete matching, request-level bounds use whichever potential outcome remains observed and give a sharp fi
    
[^268]: 透过量级势的视角看神经网络中的泛化

    Generalization in Neural Networks Through the Lens of Magnitude Potential

    [https://arxiv.org/abs/2610.01633](https://arxiv.org/abs/2610.01633)

    本文提出基于度量量级理论的“量级势”新概念，通过logit层上类内与全数据集的量级势之比来刻画样本的表示情况，实验表明该比值与Feldman记忆分数相关，并能检测决策边界的结构变化、为模运算中的grokking现象提供几何指标。

    

    解释神经网络中的泛化与训练动态仍然是一个挑战，目前已有多种方法被开发用于研究这些现象的不同方面。在本文中，我们引入了“量级势”这一概念——一个基于度量量级理论的量——它反映了一个任意点被给定集合所代表的程度。我们发现这个基本量可以被应用于考察神经泛化中的多种特征。在logit层上计算得到的、相对于某个类别与相对于整个数据集的量级势之比，能够揭示该点的表示信息。在实验中，我们发现单个训练点的这些比值与Feldman记忆分数相关。跨点聚合的量级势比值能够检测决策边界中的结构性变化，并为模运算中的grokking（顿悟）现象提供一种几何指标。

    arXiv:2610.01633v1 Announce Type: new  Abstract: Explaining generalization and training dynamics in neural networks remains a challenge, and various approaches have been developed to study different aspects of these phenomena. In this paper, we introduce the idea of {\em magnitude potential} -- a quantity based on the theory of metric magnitude -- that reflects how well an arbitrary point is represented by a given set. We find that this basic quantity can be applied to examine various features in neural generalization. The ratio between the magnitude potential with respect to a class and with respect to the entire data, computed at the logit layer, is informative of the representation of the point. In experiments, these ratios for individual training points are found to be correlated with the Feldman memorization scores. Magnitude potential ratios aggregated across points detect structural changes in the decision boundaries and provide a geometric indicator of grokking in modular arith
    
[^269]: 冗余与协同的相遇：基于次模优化的MoE依赖感知专家选择

    Redundancy Meets Synergy: Dependency-aware Expert Selection for MoE via Submodular Optimization

    [https://arxiv.org/abs/2610.00558](https://arxiv.org/abs/2610.00558)

    提出DS-MoE框架，通过次模函数差分优化感知专家间的冗余与协同依赖关系，为MoE模型实现更优的紧凑专家子集选择。

    

    尽管混合专家模型通过稀疏激活有效地扩展了模型容量，但其部署常常受到过高内存需求的制约。提取一个紧凑的专家子集是一种有前景的解决方案。然而，现有的专家选择启发式方法主要依赖Top-k排序，这种做法孤立地评估单个专家，忽略了MoE门控网络所引入的专家间复杂的相互依赖关系。本文提出DS-MoE，一个具有理论支撑的框架，通过次模函数差分（DS）优化重新定义了专家选择问题。通过分析损失退化的二阶泰勒展开，我们揭示了专家组合中的功能二重性：冗余（专家编码重叠的表示）与协同（专家提供互补的误差消除）。为应对这种二重性，我们在数学上将冗余削减……

    arXiv:2610.00558v1 Announce Type: new  Abstract: While Mixture-of-Experts (MoE) models effectively scale model capacity through sparse activation, their deployment is often bottlenecked by prohibitive memory requirements. Extracting a compact subset of experts presents a promising solution. However, existing expert selection heuristics predominantly rely on Top-k ranking, which isolates the evaluation of individual experts and ignores the intricate inter-expert dependencies introduced by the MoE gating network. In this paper, we propose DS-MoE, a theoretically grounded framework that redefines expert selection via difference-of-submodular (DS) optimization. By analyzing the second-order Taylor expansion of the loss degradation, we reveal functional duality within expert combinations: redundancy (where experts encode overlapping representations) and synergy (where experts provide complementary error cancellation). To navigate this duality, we mathematically decouple redundancy reduction
    
[^270]: 基于国际象棋的大语言模型提示词优化基准测试

    Benchmarking Prompt Optimization of Large Language Models With Chess

    [https://arxiv.org/abs/2610.00416](https://arxiv.org/abs/2610.00416)

    该论文提出了一个基于1,118道Lichess国际象棋棋题的基准测试，利用国际象棋低成本精确评分、引擎辅助着法评估和可持续更新的可调难度题源等特性，来研究冻结大语言模型的自动提示词优化。

    

    随着大语言模型能力的不断提升，对其评估变得越来越具有挑战性：基准测试可能饱和，公开测试集面临数据污染的风险，而评估更难的任务可能需要昂贵的评分或执行基础设施。这些挑战在自动提示词优化（APO）中被进一步放大，因为在搜索更优提示词的过程中需要反复进行评估。因此，研究APO需要一个评分成本低且结果确定、难度足够高以留出改进空间、并随模型演进可持续更新的基准测试。我们引入了一个由1,118道Lichess棋题构建的国际象棋基准测试，用于研究冻结大语言模型（即不更新模型权重、仅优化提示词）的自动提示词优化。国际象棋结合了低成本的精确匹配评分、基于引擎的备选着法评估，以及难度可调且可持续更新的题源。与仅报告孤立测试项成功率的评估不同，该基准测试还……（原文摘要在此处截断）

    arXiv:2610.00416v1 Announce Type: new  Abstract: Evaluating large language models becomes increasingly challenging as their capabilities advance: benchmarks can saturate, public test sets risk contamination, and assessing harder tasks can require expensive grading or execution infrastructure. These challenges are amplified in automatic prompt optimization (APO), where evaluation is repeated throughout the search for better prompts. Studying APO therefore requires a benchmark that is cheap and deterministic to score, hard enough to leave room for improvement, and renewable as models evolve. We introduce a chess benchmark built from 1,118 Lichess puzzles to study APO for frozen LLMs: we optimize their prompts without updating their model weights. Chess combines inexpensive exact-match scoring, engine-based evaluation of alternative moves, and a renewable supply of problems with adjustable difficulty. Unlike evaluations that report only success on isolated test items, the benchmark also c
    
[^271]: FAER：面向语言模型后训练的可审计、效用对齐的轨迹重放

    FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training

    [https://arxiv.org/abs/2610.00385](https://arxiv.org/abs/2610.00385)

    FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。

    

    重放选择器通常依据格式反馈、置信度、新鲜度或响应长度对缓存轨迹进行排序，然而缓存层面的正确性与下游学习器的效用是两个不同的目标。我们形式化了这种“选择到学习”的差距，并提出FAER作为一个可审计的全轨迹重放框架。其中免训练的固定选择器作为协议基线；FAER-UTILITY则是在互不相交的校准块上拟合的学习器感知选择器。归一化梯度对齐被报告为基线方法，而一次性的优化器感知虚拟更新提供了幅度感知的效用面。审计契约会在评估标签加入之前冻结观测字段和重放轨迹。在GSM8K数据集上使用Qwen2.5-1.5B-Instruct模型，匹配学习器的研究显示固定选择器在128次更新下质量达到0.6329，相比之下均匀采样为0.5482，格式反馈为0.6037。仅使用元数据的交叉拟合校准达到0.6...

    arXiv:2610.00385v1 Announce Type: cross  Abstract: Replay selectors often rank cached trajectories by format feedback, confidence, freshness, or response length, although cache-level correctness and downstream learner utility are distinct objectives. We formalize this selection-to-learning gap and introduce FAER as an auditable full-trajectory replay framework. Its training-free fixed selector is a protocol baseline; FAER-UTILITY is the learner-aware selector fitted on disjoint calibration blocks. The normalized gradient alignment is reported as a baseline, while a disposable optimizer-aware virtual update supplies a magnitude-aware utility surface. The audit contract freezes observed fields and replay traces before evaluation labels are joined. On GSM8K with Qwen2.5-1.5B-Instruct, the matched learner study reports quality 0.6329 for the fixed selector, compared with 0.5482 for uniform and 0.6037 for format-feedback under 128 updates. Metadata-only cross-fitted calibration reaches $0.6
    
[^272]: ContractRL：面向可审计工具调用修复的屏蔽式组相对策略优化

    ContractRL: Shielded Group-Relative Policy Optimization for Auditable Tool-Call Repair

    [https://arxiv.org/abs/2610.00328](https://arxiv.org/abs/2610.00328)

    ContractRL提出了一种契约约束的顺序修复协议，将验证器引导的JSON修复建模为有界决策过程，通过契约派生的动作掩码过滤非法修复操作，在仅生成34.4个token的情况下实现了0.9362的语义成功率，并保证了修复过程的可审计性。

    

    结构化工具调用往往在仅有少量字段违反模式或执行契约时就会失败。重新生成完整对象会扩大动作空间，并使重复修复难以审计。我们提出了ContractRL，这是一种契约约束的顺序修复协议，将验证器引导的JSON修复建模为一个有界决策过程。在每一步中，策略观察候选对象、带类型的验证器反馈、JSON Pointer、不可变的修复历史以及剩余预算；契约派生的动作掩码在确定性验证器执行状态转移之前过滤掉格式错误或被禁止的RFC-6902操作。我们为补丁、重试和放弃决策指定了契约约束的组相对目标，同时将规范目标和语义标签保持在在线状态之外，直到轨迹冻结。在相同的验证器信息下，ContractRL仅使用34.4个生成token就达到了0.9362的语义成功率。

    arXiv:2610.00328v1 Announce Type: new  Abstract: Structured tool calls often fail after only a small number of fields violate a schema or an execution contract. Regenerating the complete object enlarges the action surface and makes repeated repair difficult to audit. We introduce ContractRL, a contract-constrained sequential repair protocol that models verifier-guided JSON repair as a bounded decision process. At each step the policy observes the candidate, typed verifier feedback, JSON Pointer, immutable repair history, and remaining budget; a contract-derived action mask filters malformed or prohibited RFC-6902 operations before a deterministic validator performs the transition. We specify a contract-constrained group-relative objective for patch, retry, and abstention decisions while keeping canonical targets and semantic labels outside the online state until trace freeze. Under identical verifier information, ContractRL attains 0.9362 semantic success with 34.4 generated tokens, co
    
[^273]: 带收据的动作：联合绑定声明、证据与执行以实现可重放的工具型智能体审计

    Actions with Receipts: Jointly Binding Claims, Evidence, and Execution for Replayable Tool-Agent Auditing

    [https://arxiv.org/abs/2610.00327](https://arxiv.org/abs/2610.00327)

    该论文提出“声明锚定执行契约”，通过收据机制将智能体发出的声明、证据来源片段、执行前缀与来源版本联合绑定，并由确定性完整性验证器重建绑定关系，从而实现对工具型智能体行为的可重放审计。

    

    使用工具的智能体可以展示引用和执行日志，却留下一个关键关联未经审计：呈现给用户的声明是否就是已提交执行所发出的、且被所引用来源支持的声明。因此，一个有效的引用和一个有效的执行轨迹可以各自形式上完好，却被移植到其他声明、动作、运行或来源版本之上。我们提出了一种以声明为锚点的执行契约，将发出的声明、其精确的来源片段、产生该声明的有序执行前缀，以及该执行所观察到的来源版本和访问状态联合绑定在一起。每份收据包含一个发射锚点，用于确定性地定位该声明在已提交答案或含声明动作中的位置，同时包含来源标识符、偏移量、哈希值、引文以及域分离的执行承诺。一个确定性的完整性验证器会在加入语义或任务标签之前重建这些绑定关系。

    arXiv:2610.00327v1 Announce Type: cross  Abstract: Tool-using agents can expose citations and execution logs while leaving a critical association unaudited: whether the claim shown to a user is the claim emitted by the committed execution and supported by the cited source. A valid citation and a valid trace can therefore remain individually well formed while being transplanted across claims, actions, runs, or source versions. We introduce a claim-anchored execution contract that jointly binds the emitted claim, its exact source span, the ordered execution prefix that produced it, and the source version and access state observed by that execution. Each receipt contains an emission anchor that deterministically locates the claim inside a committed answer or claim-bearing action, together with source identifiers, offsets, hashes, quotes, and a domain-separated execution commitment. A deterministic integrity verifier reconstructs these bindings before semantic or task labels are joined. We
    
[^274]: GroundingPI：一个基于视觉原语、面向物理智能的视觉定位基础模型

    GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives

    [https://arxiv.org/abs/2609.39601](https://arxiv.org/abs/2609.39601)

    GroundingPI是一个40亿参数的视觉定位基础模型，它将点和框表示为共享词表中的量化坐标，并通过多模态与空间预训练、监督微调及GRPO强化学习训练而成，在11种感知能力的34个基准测试上以平均73.68%的成绩超越更大的GPT-6 Astra，创下新的最先进水平，为物理智能提供了快速精确的视觉感知基础。

    

    精确的视觉定位至关重要。它需要指明哪个物体是目标以及该物体位于何处，即使在杂乱场景和微小物体的情况下也必须准确，并且必须足够快速以支持闭环控制。然而，视觉-语言-动作模型（VLA）和世界-动作模型（WAM）通常从通用的视觉-语言骨干网络和视频生成骨干网络获取感知能力，而这些骨干网络在上述场景中仍然表现不佳。我们提出了GroundingPI，一个40亿参数的视觉定位基础模型，它以共享词表中的量化坐标形式生成点和边界框。其训练结合了多模态与空间预训练、监督微调以及基于GRPO的强化学习，并利用公开数据集和专用数据引擎提供监督信号。在与涵盖11种感知能力的34个视觉定位基准上的44个基线模型的对比中，GroundingPI确立了新的最先进水平，平均成绩达到73.68%，超过了参数量更大的GPT-6 Astra（71.54%）。作为下游任务的视觉骨干网络，GroundingPI提升了……（摘要在此处被截断）

    arXiv:2609.39601v1 Announce Type: cross  Abstract: Precise grounding matters. It specifies which object is the target and where that object is, even in clutter and for tiny objects, and it has to be fast enough for closed-loop control. Yet vision-language-action (VLA) and world-action models (WAMs) take perception from general-purpose vision-language and video-generation backbones, which still fail in these settings. We introduce GroundingPI, a 4B grounding foundation model that generates points and boxes as quantized coordinates in a shared vocabulary. Training combines multimodal and spatial pretraining, supervised fine-tuning, and reinforcement learning with GRPO, using supervision from public datasets and dedicated data engines. Against 44 baselines across 34 grounding benchmarks spanning 11 perceptual capabilities, GroundingPI establishes a new state of the art, averaging 73.68%, above the larger GPT-6 Astra (71.54%). As a downstream visual backbone, GroundingPI improves performan
    
[^275]: GroundAnything：以闪速实现并行解码与精准视觉定位的统一

    GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed

    [https://arxiv.org/abs/2609.39600](https://arxiv.org/abs/2609.39600)

    GroundAnything 是一个 40 亿参数的视觉定位基础模型，它将视觉定位重新定义为可并行生成的双向扩散过程，通过分块去噪、AR 到扩散的架构转换训练及 GRPO 强化后训练，在实现闪速并行解码的同时保证了精准的视觉定位效果。

    

    自回归（AR）视觉定位模型将空间预测序列化，引入了顺序延迟，并对输出 token 施加了因果顺序约束。我们将视觉定位视为视觉证据提取：物体、位置和空间关系由图像和查询共同约束，但它们之间的依赖关系并不意味着存在内在的从左到右的生成顺序。这一区别使得双向扩散成为天然合适的选择，允许空间假设并行涌现，并通过迭代去噪进行联合优化。我们提出了 GroundAnything，一个 40 亿参数的视觉定位基础模型，通过分块去噪实现了快速并行解码与精准定位的统一。其训练过程结合了基于公开数据集与专用数据引擎的定位预训练、采用 AR 与扩散联合目标的直接 AR 到扩散架构转换、监督微调以及基于 GRPO 的强化后训练。在 30 个视觉定位基准测试上……

    arXiv:2609.39600v1 Announce Type: cross  Abstract: Autoregressive (AR) grounding models serialize spatial predictions, introducing sequential latency and imposing a causal order on output tokens. We view grounding as visual evidence extraction: objects, locations, and spatial relations are jointly constrained by the image and query, yet their dependencies do not imply an intrinsic left-to-right generation order. This distinction makes bidirectional diffusion a natural fit, allowing spatial hypotheses to emerge in parallel and be jointly refined through iterative denoising. We introduce GroundAnything, a 4B-parameter grounding foundation model that reconciles fast parallel decoding with precise localization through blockwise denoising. Training combines grounding pretraining from public datasets and dedicated data engines, direct AR-to-diffusion conversion with joint AR and diffusion objectives, supervised fine-tuning, and GRPO-based reinforcement post-training. Across 30 grounding benc
    
[^276]: VStress：面向重复验证器的相关性感知审计与自适应预算分配

    VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers

    [https://arxiv.org/abs/2609.36958](https://arxiv.org/abs/2609.36958)

    本文提出VStress可审计重放契约与VStress-CA相关性感知分配策略，通过估计重复验证器调用的条件边际信息来自适应分配预算并在信息不足时停止调用，从而在固定预算下以更低的调用成本获得更高的验证平衡准确率，同时通过受控审计刻画了该机制在极端损坏率下的适用边界。

    

    重复验证器调用只有在贡献条件信息时才有价值。我们提出了VStress——一种可审计的重放契约，以及VStress-CA——一种相关性感知的分配策略。该策略在密封的校准集上估计未查询验证器的条件边际信息，对不确定性进行折扣，按调用成本归一化，并在下一次调用不具信息量时停止或弃权。控制器在接入干净预言机之前冻结其决策和成本账本；当依赖性偏移警报触发时，禁用通道偏好并回退到精确停止机制。受控审计给出了该机制的适用边界：在35%对称损坏下，多数投票-5将平衡准确率从0.6578提升至0.7739；而在65%损坏下则损失0.1226个百分点。在固定预算匹配比较中，广度、冗余和自适应分配策略分别取得0.6048、0.6375和0.6538的平衡准确率，每项平均调用3.4216次，RLVR分数为0.64。

    arXiv:2609.36958v1 Announce Type: cross  Abstract: Repeated verifier calls are useful only when they contribute conditional information. We introduce VStress, an auditable replay contract, and VStress-CA, a correlation-aware allocation policy that estimates the conditional marginal information of an unqueried verifier on a sealed calibration split, discounts uncertainty, normalizes by call cost, and stops or abstains when the next call is not informative. The controller freezes its decision and cost ledger before joining the clean oracle; a dependence-shift alarm disables channel preference and falls back to exact-stop. The controlled audit gives the mechanism boundary: at 35% symmetric corruption, majority-5 improves balanced accuracy from 0.6578 to 0.7739, whereas at 65% it loses 0.1226 points. In the matched fixed-budget comparison, breadth, redundancy, and adaptive allocation obtain balanced accuracies 0.6048, 0.6375, and 0.6538, with 3.4216 calls per item and an RLVR score of 0.64
    
[^277]: 手写文本识别存在于高方差像素子空间中

    Handwritten Text Recognition Lives in the High-Pixel Variance Subspace

    [https://arxiv.org/abs/2609.35473](https://arxiv.org/abs/2609.35473)

    该论文揭示了手写文本识别的判别性信息集中在像素空间的高方差方向上，从而解释了为什么基于像素重建的自监督预训练方法优于对比学习方法，并在五种语言的六个手写基准上均取得最低字符错误率。

    

    在用于手写文本识别（HTR）的自监督预训练中，像素重建方法优于对比方法，这与自然图像分类中的情况不同。我们认为这种差异源于判别性信号在像素空间中的位置：对于HTR而言，判别性信号集中在高方差方向上，而在低方差方向上基本不存在。这预示着保留高方差像素内容的目标函数将具有最佳的迁移效果。我们在六个跨越五种语言的手写基准数据集上，在匹配的编码器、数据和评估协议下，测试了来自三个家族（基于像素的MIM、JEPA和对比学习）的六种SSL方法。在完整标签条件下，基于像素的SSL在每个基准测试和两种冻结探测评估中都取得了最低的字符错误率（CER），能够暴露逐位置的字符信息，而其他家族的方法只能通过读出层恢复这些信息，并且它是唯一能从真实手写体预训练中受益的家族。

    arXiv:2609.35473v2 Announce Type: replace-cross  Abstract: In self-supervised pretraining for Handwritten Text Recognition (HTR), pixel reconstruction methods outperform contrastive methods, unlike in natural-image classification. We argue that this difference follows from where discriminative signal lies in pixel space: for HTR, it is concentrated in high-variance directions and largely absent from low-variance ones. This predicts that objectives preserving high-variance pixel content will transfer best. We test six SSL methods from three families (pixel-grounded MIM, JEPA, and contrastive) under matched encoder, data, and evaluation protocols on six handwriting benchmarks across five languages. With full labels, pixel-groundrounded SSL achieves the lowest CER on every benchmark and both frozen probes, exposes per-position character information that other families recover only through the readout, and is the only family to benefit from pretraining on real handwriting. Pixel-grounded r
    
[^278]: RSD-Poker：面向不完美信息游戏中残差策略的结构自适应与偏移稳健的风险-效用认证

    RSD-Poker: Structure-Adaptive and Shift-Robust Risk-Utility Certification for Residual Policies in Imperfect-Information Games

    [https://arxiv.org/abs/2609.33669](https://arxiv.org/abs/2609.33669)

    该论文提出 RSD-Poker 框架，通过冻结残差候选库、学习并预先冻结策略可见的子群划分，为不完美信息游戏中的残差策略提供对部署分布偏移稳健的风险上界与效用下界双证书认证。

    

    残差策略适配提供了一种轻量级的方式来修改强参考策略，但共享的缩放尺度与固定的子群划分可能掩盖异质性的性能退化，并在部署时信息状态混合分布发生变化时变得脆弱。我们提出了 RSD-Poker，一个结构自适应且对分布偏移稳健的认证框架：它冻结一个由残差族和尺度组成的候选库，在独立的结构划分集上学习一个策略可见的子群划分，并在加入校准标签之前冻结该划分。每个“候选-组”配对都会获得一个关于相对于锚点策略风险的加权同时上界证书，以及一个关于弱响应效用的下界证书。随后，在一个预先声明的部署组比例不确定性集合上，选出稳健的组到候选的映射。在从每个冻结组的分布中抽取独立校准单元、候选库与划分在校准前固定、以及组内……（原文摘要在此处截断）

    arXiv:2609.33669v2 Announce Type: replace  Abstract: Residual policy adaptation provides a lightweight way to modify a strong reference policy, but a shared scale and a fixed subgroup partition can hide heterogeneous degradation and become fragile when the deployment mixture of information states changes. We introduce RSD-Poker, a structure-adaptive and shift-robust certification framework that freezes a bank of residual families and scales, learns a policy-visible partition on an independent structure split, and freezes that partition before calibration labels are joined. Each candidate-group pair receives a weighted simultaneous upper certificate for anchor-relative risk and a lower certificate for weak-response utility. A robust group-to-candidate map is then selected over a predeclared uncertainty set of deployment group proportions. Under independent calibration units drawn from each frozen group's law, a candidate bank and partition fixed before calibration, and invariant within-
    
[^279]: 审计优先的VAPO：不完美验证下的风险认证选择性更新

    Audit-First VAPO: Risk-Certified Selective Updates under Imperfect Verification

    [https://arxiv.org/abs/2609.33662](https://arxiv.org/abs/2609.33662)

    该论文提出Audit-First VAPO框架，通过将更新方向的离散准入与幅值的连续控制相分离，在有限验证预算下利用“接受-申诉-弃权”观察策略和有限样本统计界，实现对不完美验证器下有害更新风险、覆盖率与验证器调用率的可认证保障。

    

    不完美的验证器即使在裁剪和正则化限制了其幅值的情况下，仍可能分配出有害的更新方向。我们提出了Audit-First VAPO，它将离散的方向准入与连续的幅值控制分离开来。一种仅用于观察的“接受-申诉-弃权”策略使用有限的二次验证预算；其动作轨迹在加入干净标签之前被冻结。随后，同时的有限样本界在一个预先声明的策略族上对选定的有害风险、覆盖率以及验证器调用率进行认证。条件Hoeffding-Azuma界解释了由共享预算所引起的依赖性，而rollout或验证器的变更会启动新的认证阶段。在准入之后，一个有界的trust-clip-KL执行器来控制幅值大小。我们在两个模型和两个推理基准上对该方法进行评估，对比了静态RLVR、匹配随机选择、置信度阈值、噪声校正以及验证器增强等基线方法。在Qwen3.5-0.8B和GSM8K上，在目标……

    arXiv:2609.33662v2 Announce Type: replace  Abstract: Imperfect verifiers can assign a harmful update direction even when clipping and regularization bound its magnitude. We introduce Audit-First VAPO, which separates discrete directional admission from continuous magnitude control. An observation-only accept-appeal-abstain policy uses a finite secondary-verification budget; its action trace is frozen before clean labels are joined. Simultaneous finite-sample bounds then certify selected harmful risk, coverage, and verifier-call rate over a predeclared policy family. Conditional Hoeffding-Azuma bounds account for the dependence induced by shared budgets, and rollout or verifier changes initiate a new certification stage. After admission, a bounded trust-clip-KL actuator controls magnitude. We evaluate two models on two reasoning benchmarks against static RLVR, matched-random selection, confidence thresholding, noise correction, and verifier augmentation. On Qwen3.5-0.8B and GSM8K at tar
    
[^280]: 在注意力机制中使用量化Softmax预训练Transformer

    Pretraining Transformers with Quantized Softmax in Attention

    [https://arxiv.org/abs/2609.33591](https://arxiv.org/abs/2609.33591)

    本文提出K区间注意力方法对注意力中的softmax进行量化近似，系统研究了网格校准、舍入方式和直通替代梯度位置等设计选择对Transformer预训练的影响，并推导了相应的反向传播规则。

    

    低精度Transformer系统越来越多地对注意力矩阵乘法进行量化，而softmax通常仍保持较高的计算精度。在预训练期间，近似的softmax会改变用于训练模型的梯度以及模型的前向计算。我们通过K区间注意力来研究这种相互作用，该方法使用K+1个网格值来近似指数函数。我们改变了逐行网格校准方式、插值与硬舍入的对比，以及直通替代梯度相对于归一化的位置。我们推导了相应的反向传播规则，包括校准导数，并在模型、数据和优化器完全匹配的预训练实验中比较了这些设计选择。将行极值分离（detach）会使前向计算保持不变，但会导致验证损失出现延迟性上升。在K=4的硬舍入设置下，最小-最大校准和预归一化替代方案会产生较大的损失差距；改变其中任何一个选择都会显著（原文摘要在此处截断）……

    arXiv:2609.33591v2 Announce Type: replace  Abstract: Low-precision Transformer systems increasingly quantize attention matrix multiplications, while softmax often remains at higher precision. During pretraining, an approximate softmax changes the gradients that train the model as well as its forward computation. We study this interaction with K-interval attention, which approximates the exponential using K+1 grid values. We vary per-row grid calibration, interpolation versus hard rounding, and the placement of a straight-through surrogate relative to normalization. We derive the corresponding backward rules, including calibration derivatives, and compare these choices in pretraining experiments matched on model, data, and optimizer. Detaching the row extrema leaves the forward computation unchanged but produces a delayed increase in validation loss. With hard rounding at K=4, min-max calibration and a pre-normalization surrogate incur a large loss gap; changing either choice substantia
    
[^281]: OSCC：面向非完美信息学习中梯度噪声控制的认证观测安全耦合优化

    OSCC: Certified Observation-Safe Coupling Optimization for Gradient-Noise Control in Imperfect-Information Learning

    [https://arxiv.org/abs/2609.33543](https://arxiv.org/abs/2609.33543)

    提出了观测安全反事实耦合（OSCC）框架，通过定义可容许耦合类的多项安全准则并引入梯度感知的耦合优化准则，在非完美信息学习中安全地降低反事实比较噪声，同时针对策略梯度几何控制回报协方差以优化多动作策略。

    

    耦合的轨迹推演可以降低反事实动作比较中的噪声，但存在两个问题阻碍了标准的共同随机数构造在非完美信息环境中作为通用学习原语来使用。首先，无效的耦合可能暴露隐藏状态、同步内生策略随机性，或在反事实历史分叉后错配机会事件。其次，在多动作策略优化中，更低的回报对比方差本身并不是相关的优化目标：优化器实际依赖于经过局部策略梯度几何投影后的回报协方差矩阵。我们提出了观测安全反事实耦合，这是一个通过边际保持、信息状态安全、分支局部策略随机性、语义事件对齐以及先记录后预言机重放等准则来定义可容许耦合类的框架。我们推导了一个梯度感知的耦合准则，表明对于边际保持的……（原文摘要在此处被截断）

    arXiv:2609.33543v2 Announce Type: replace  Abstract: Coupled rollouts can reduce the noise of counterfactual action comparisons, but two issues prevent standard common-random-number constructions from serving as a general learning primitive in imperfect-information environments. First, an invalid coupling may expose hidden state, synchronize endogenous policy randomness, or misalign chance events after counterfactual histories diverge. Second, in multi-action policy optimization, lower return-contrast variance is not by itself the relevant objective: the optimizer depends on the return covariance matrix after projection through the local policy-gradient geometry. We introduce observation-safe counterfactual coupling (OSCC), a framework that defines an admissible class through marginal preservation, information-state safety, branch-local policy randomness, semantic event alignment, and trace-before-oracle replay. We derive a gradient-aware coupling criterion showing that, for marginal-p
    
[^282]: 一种在 fat-shattering 维度上具有近线性大小的平方损失不可知样本压缩方案

    An Agnostic Sample Compression Scheme for Squared Loss of Near-Linear Size in the Fat-Shattering Dimension

    [https://arxiv.org/abs/2609.29696](https://arxiv.org/abs/2609.29696)

    本文首次构造了大小仅与 fat-shattering 维度近线性相关（乘以对数因子）且与样本量无关的平方损失不可知样本压缩方案，正面解决了 Attias 等人（ICML 2024）提出的开放性问题。

    

    我们为每一个函数类 $\mathcal{F}\subseteq[0,1]^{\mathcal{X}}$ 以及每一个精度 $0<\alpha\le 1$ 构造了一个针对经验平方损失的不可知样本压缩方案：对于任意带有（含噪声）标签的有限样本 $S\in(\mathcal{X}\times[0,1])^m$，该方案至多存储 $O(\mathrm{fat}(\mathcal{F},c'\alpha)\cdot\log^3(2/\alpha))$ 个原始标注样本和辅助比特，其大小与样本量 $m$ 无关，并能重构出一个函数 $\hat f$，满足 $L_2(\hat f,S)\le\inf_{f\in\mathcal{F}}L_2(f,S)+\alpha$。这正面解决了 Attias、Hanneke、Kontorovich 和 Sadigurschi（ICML 2024，第 5 节）提出的开放性问题，即要求构造大小为 $\mathrm{fat}(\mathcal{F},c\alpha)\cdot\mathrm{polylog}(c/\alpha)$ 的不可知 $\ell_2$ 压缩方案。此前所有已知的有界大小构造——无论是不可知情形还是甚至可实现情形——都会引入一个乘性的对偶 fat-shattering 因子，而该因子可能是指数级大的。

    arXiv:2609.29696v1 Announce Type: new  Abstract: We construct, for every function class $\mathcal{F}\subseteq[0,1]^{\mathcal{X}}$ and every accuracy $0<\alpha\le 1$, an agnostic sample compression scheme for the empirical squared loss: for every finite sample $S\in(\mathcal{X}\times[0,1])^m$ with arbitrary (noisy) labels, the scheme stores at most $O(\mathrm{fat}(\mathcal{F},c'\alpha)\cdot\log^3(2/\alpha))$ original labeled examples and auxiliary bits, independent of the sample size $m$, and reconstructs a function $\hat f$ with $L_2(\hat f,S)\le\inf_{f\in\mathcal{F}}L_2(f,S)+\alpha$. This resolves, in the positive, the open problem of Attias, Hanneke, Kontorovich, and Sadigurschi (ICML 2024, Section 5), which asks for an agnostic $\ell_2$ compression scheme of size $\mathrm{fat}(\mathcal{F},c\alpha)\cdot\mathrm{polylog}(c/\alpha)$. All previously known bounded-size constructions, agnostic and even realizable, incur a multiplicative dual fat-shattering factor, which can be exponentiall
    
[^283]: 老虎机多类PAC学习：修正的下界、精确类族与置信度直和现象

    Bandit Multiclass PAC Learning: Corrected Lower Bounds, Exact Families, and a Confidence Direct-Sum Phenomenon

    [https://arxiv.org/abs/2609.29694](https://arxiv.org/abs/2609.29694)

    本文修正了老虎机多类PAC学习的下界理论，指出原有基于BDS维度的下界并不正确，引入锚定维度aBDS证明了新的无常数下界，同时在上界中完全去除了对标签数K的依赖，并揭示了置信度直和现象。

    

    我们研究具有老虎机反馈的可实现多类PAC学习：学习器观察到一个独立同分布的样本实例，预测K个标签中的一个，并且仅获知预测是否正确。Hanneke、Meng、Moran和Shaeiri（arXiv:2605.25678）通过对数因子范围内的老虎机DS维度BDS刻画了最优样本复杂度，并提出问题：是否每个类都具有样本复杂度O((BDS+log(1/δ))/ε)。首先，我们证明已发表的下界Ω((BDS+log(1/δ))/ε)如所述是不正确的：我们给出了BDS=K-1的显式类族，其样本复杂度呈指数级更小，并定位了其证明中两个独立的漏洞。我们围绕一个新的锚定维度aBDS≤BDS修复了下界理论，证明了一个无常数的由三部分构成的下界。在上界方面，我们完全移除了环境标签数K的依赖，证明了O(...（摘要在此处截断）

    arXiv:2609.29694v1 Announce Type: new  Abstract: We study realizable multiclass PAC learning with bandit feedback: the learner observes an i.i.d. instance, predicts one of $K$ labels, and learns only whether the prediction was correct. Hanneke, Meng, Moran, and Shaeiri (arXiv:2605.25678) characterized the optimal sample complexity via the bandit DS dimension $\mathrm{BDS}$ up to logarithmic factors, and asked whether every class admits sample complexity $O((\mathrm{BDS}+\log(1/\delta))/\epsilon)$.   First, we show that the published lower bound $\Omega((\mathrm{BDS}+\log(1/\delta))/\epsilon)$ is incorrect as stated: we exhibit explicit classes with $\mathrm{BDS}=K-1$ whose sample complexity is exponentially smaller, and locate two independent gaps in its proof. We repair the lower-bound theory around a new anchored dimension $\mathrm{aBDS}\le\mathrm{BDS}$, proving a constant-free three-part lower bound. On the upper-bound side we remove the ambient label count $K$ entirely, proving $O(
    
[^284]: 强化学习中成本感知的语言模型引导的认证预测性建议价值门控

    Certified Predictive Value-of-Advice Gating for Cost-Aware Language-Model Guidance in Reinforcement Learning

    [https://arxiv.org/abs/2609.29548](https://arxiv.org/abs/2609.29548)

    该论文提出了一种认证预测性建议价值门控框架，在调用语言模型前预测可能的响应并评估其价值，仅当预测价值的置信下界超过调用成本时才发起查询，并辅以动作特定证书管理执行，从而实现成本感知且带理论保证的强化学习引导。

    

    语言模型建议可以加速强化学习，但调用成本高昂，且返回的动作可能过时或错误。我们将建议获取形式化为一个响应条件化的元推理问题：在查询之前，控制器预测可能的解析响应，评估每个响应将产生的决策和声明的后续行动，仅当预测价值的置信下界超过定价成本时才进行查询。执行则由动作特定的证书单独管理。在明确的假设下，认证建议是近似最优的，被包装的学习器仅在干预稳定性条件下继承回退遗憾，且保守分配相对于短视预言机最多损失声明的查询价值估计误差。在BabyAI上，采用Qwen2.5-1.5B和7B作为建议者的代理校准控制器，在20个种子上相比不查询将GoToObj回报提高了0.029 ± 0.016和0.030 ± 0.015，同时减少……（摘要截断）

    arXiv:2609.29548v1 Announce Type: new  Abstract: Language-model advice can accelerate reinforcement learning, but calls are costly and returned actions may be stale or wrong. We formulate advice acquisition as a response-contingent metareasoning problem: before querying, the controller predicts possible parsed responses, evaluates the decision and declared continuation that would follow each response, and queries only when a lower confidence bound on predictive value exceeds the priced cost. Execution is governed separately by an action-specific certificate. Under explicit assumptions, certified advice is near-optimal, a wrapped learner inherits fallback regret only under intervention stability, and conservative allocation loses at most the declared query-value estimation error relative to a myopic oracle. On BabyAI, a proxy-calibrated controller with Qwen2.5-1.5B and 7B advisors improves GoToObj return over no querying by 0.029 +/- 0.016 and 0.030 +/- 0.015 across 20 seeds while reduc
    
[^285]: 关于带成员查询的主动学习的样本复杂度

    On the Sample Complexity of Active Learning with Membership Queries

    [https://arxiv.org/abs/2609.27241](https://arxiv.org/abs/2609.27241)

    本研究揭示了允许合成成员查询会显著改变统计学习的难度——某些在基于池的主动学习下只能实现多项式误差衰减的假设类，在允许合成查询后变得可指数级快速学习，表明成员查询合成是一种需要新分析工具来刻画的根本不同的学习模式。

    

    本工作重新审视了主动学习中的一个根本问题：合成任意查询的能力究竟有多强大？与基于池的主动学习相比——即学习者只能从给定的未标注数据池中选择查询——我们发现这种看似微小的查询能力变化可能会极大地改变统计学习的难度。特别地，某些在基于池的设置下本质上学习缓慢、其误差随样本数量仅呈多项式衰减的假设类，一旦允许合成查询，便变得可以指数级快速学习。这一显著的差距表明，成员查询的合成引发了一种根本不同的学习模式，这种模式未能被现有主动学习理论充分刻画，需要新的分析工具来表征其复杂度。受这一现象启发，我们提出了若干充分条件，展示了有趣的例子，并提出……

    arXiv:2609.27241v1 Announce Type: cross  Abstract: This work revisits a fundamental question in active learning: how powerful is the ability to synthesize arbitrary queries? Compared to pool-based active learning, where the learner only selects queries from a given unlabeled pool, we find that this seemingly mild change in query ability may dramatically alter the difficulty of statistical learning. In particular, some hypothesis classes that are inherently slow to learn in the pool-based setting, achieving only polynomial error decay in the number of samples, become exponentially learnable once synthesized queries are allowed. This striking gap suggests that membership query synthesis induces a fundamentally different mode of learning, one that is not adequately captured by existing active learning theory and calls for new analytical tools to characterize its complexity. Motivated by this phenomenon, we develop several sufficient conditions, present intriguing examples, and propose a c
    
[^286]: CVaR锚点回归保护对抗罕见偏移

    CVaR anchor regression protects against rare shifts

    [https://arxiv.org/abs/2609.27034](https://arxiv.org/abs/2609.27034)

    CVaR锚点回归通过用尾部平均值代替跨环境平均残差平方的均值，能够针对罕见的大偏移提供精确的最坏情况风险保证，同时避免了传统方法因过度扩大保护范围而降低常见环境预测准确性的问题。

    

    我们研究了当训练数据包含罕见的大偏移时，在新环境中的预测问题。锚点回归对跨环境的平均残差平方的均值进行惩罚，它能保护对抗由训练偏移的二阶矩所决定的椭球内的偏移。因此，覆盖罕见偏移可能需要较大的惩罚，这会在所有方向上扩大椭球，从而降低模型在常见环境上的准确性。我们提出了CVaR锚点回归，它用尾部平均值代替平均残差平方的均值。与直接应用于预测风险的CVaR或GroupDRO不同，该方法不会仅仅因为某个环境的噪声水平高就赋予其更大的权重。我们在允许异方差噪声的线性结构模型下证明了精确的最坏情况风险保证。对于离散环境，减小CVaR尾部比例会将鲁棒性集合从椭球扩展为训练偏移的缩放凸包。

    arXiv:2609.27034v1 Announce Type: cross  Abstract: We study prediction in new environments when training data contain rare, large shifts. Anchor regression penalizes the average of the squared mean residual across environments. It protects against shifts in an ellipsoid determined by the second moment of the training shifts. Covering rare shifts may therefore require a large penalty, expanding the ellipsoid in every direction and reducing accuracy on common environments. We propose CVaR anchor regression, which replaces the average of the squared mean residuals with a tail average. Unlike CVaR or GroupDRO applied directly to prediction risks, it does not give environments more weight solely because their noise levels are high. We prove an exact worst-case risk guarantee under a linear structural model that allows for heteroscedastic noise. For discrete environments, decreasing the CVaR tail fraction expands the robustness set from an ellipsoid to a scaled convex hull of the training sh
    
[^287]: 面向托卡马克破裂警报的视界感知早期事件预测

    Horizon-Aware Early Event Prediction for Tokamak Disruption Alarms

    [https://arxiv.org/abs/2609.24443](https://arxiv.org/abs/2609.24443)

    该论文针对全分布生存方法与运行决策中有限预测视界需求不匹配的问题，提出将早期事件预测（EEP）目标引入基于生存分析的托卡马克破裂预测，以深度生存机为基线应用TLS和survTLS两种方法在有限视界内直接预测破裂风险，并在DIII-D、Alcator C-Mod和EAST三个装置上进行了验证。

    

    可靠的破裂预测对于未来托卡马克的安全运行至关重要。现有的全分布生存分析方法对完整的剩余破裂时间分布进行建模，而实际的运行决策主要依赖于有限预测视界内的破裂风险。这种不匹配促使我们将早期事件预测（EEP）目标引入基于生存分析的破裂预测中。我们以深度生存机（DSM）作为全分布基线，提出将两种已有的EEP方法应用于托卡马克破裂预测：其一是时间标签平滑（TLS），它直接预测有限视界内的破裂概率；其二是survTLS，它在此基础上还对视界内的事件时间分布进行建模。通过使用共同的因果编码器，我们在DIII-D、Alcator C-Mod和EAST装置上对上述方法进行了比较，并区分了无阈值的截止期限排序性能与通过验证集选择的固定策略警报性能。

    arXiv:2609.24443v1 Announce Type: cross  Abstract: Reliable disruption prediction is essential for the safe operation of future tokamaks. Existing full-distribution survival methods model the complete residual time-to-disruption distribution, whereas operational decisions primarily depend on disruption risk within a finite prediction horizon. This mismatch motivates introducing Early Event Prediction (EEP) objectives into survival-based disruption prediction. We take Deep Survival Machines (DSM) as the full-distribution baseline and propose applying two established EEP methods to tokamak disruption prediction: Temporal Label Smoothing (TLS), which directly predicts disruption probability within a finite horizon, and survTLS, which additionally models the event-time distribution within that horizon. Using a common causal encoder, we compare these methods on DIII-D, Alcator C-Mod, and EAST. We distinguish threshold-free deadline ranking from validation-selected fixed-policy alarm perform
    
[^288]: 具有最优切换开销的LRU与LFU无遗憾混合策略

    No-Regret Mixing of LRU and LFU with Optimal Switching Cost

    [https://arxiv.org/abs/2609.07566](https://arxiv.org/abs/2609.07566)

    本文证明了现有混合缓存策略LeCar在遗忘型对手下会产生线性遗憾，并提出了一种基于Hedge的虚拟LRU与LFU缓存混合策略H-MC，能够在最优切换开销下实现对LRU与LFU较优者的无遗憾（次线性遗憾）性能保证。

    

    缓存系统通常依赖于简单的淘汰策略，如最近最少使用（LRU）和最不经常使用（LFU），它们在互补的请求模式下表现良好。诸如LeCar和Cacheus等近期策略借鉴了在线学习中专家问题的思想，将LRU和LFU结合起来。具体而言，在发生缓存未命中时，这些策略使用由追踪历史淘汰记录所更新的分数导出的概率，在两种淘汰规则之间进行随机选择。尽管这些策略展现出强劲的实证性能，但目前尚不清楚它们是否能够在每条请求序列上都保证渐近地表现得与LRU和LFU中较优者一样好，即它们是否能相对于该基准实现次线性遗憾。我们首先证明LeCar在面对遗忘型对手时会遭受线性遗憾，即使拥有无界的历史记录也是如此。随后，我们提出H-MC，一种基于Hedge算法的虚拟LRU与LFU缓存混合策略，该策略保留了Hedge的选择概率特性（摘要在此处截断）。

    arXiv:2609.07566v1 Announce Type: new  Abstract: Caching systems often rely on simple eviction policies such as Least Recently Used (LRU) and Least Frequently Used (LFU), which perform well in complementary request regimes. Recent policies such as LeCar and Cacheus combine LRU and LFU using ideas from the experts problem in online learning. Specifically, upon a miss, they randomize between the two eviction rules using probabilities derived from scores updated by tracking the history of past evictions. While these policies exhibit strong empirical performance, it remains unclear whether they are guaranteed, on every request sequence, to perform asymptotically as well as the better of LRU and LFU, i.e., whether they achieve sublinear regret with respect to this benchmark. We first show that LeCar suffers linear regret against an oblivious adversary, even with unbounded history. We then propose H-MC, a Hedge-based mixture of virtual LRU and LFU caches that preserves Hedge's selection prob
    
[^289]: 能力门控语言模型：安全性可组合，实用性不可组合

    Capability-Gated Language Models: Security Composes, Utility Does Not

    [https://arxiv.org/abs/2609.00445](https://arxiv.org/abs/2609.00445)

    提出在单一模型权重内部实现按主体的“能力门控部署”，配置构成格结构，并证明安全限制在交运算下可组合累积（安全性随限制叠加而增强），而实用性不具备这种组合性。

    

    已部署的语言模型安全防护措施（安全微调、过滤、遗忘学习）仅在模型权重之外按主体区分：过滤器被重新配置、访问层级成倍增加、模型制品被重新发布；而在同一组权重内部，所有请求面对的都是相同的模型配置。这促使我们定义“能力门控部署”：在一组权重内部实现按主体的访问控制，其配置构成一个格——交运算累积某个主体的限制，并运算汇集一个联盟的权限范围。我们通过在现有嵌套分解机制上实施稀疏秩门控来实例化该方法，利用单次遍归因引导配置搜索，并仅从预注册的保留集划分中一次性读取全部结果。安全性可组合：在单调引出假设下，交运算处的组合性可被证明，而我们在逐点意义上证伪了该假设。在两个模型谱系中，保留集上交运算的中位数效应加深了抑制；经多重校正后唯一存活的效应进一步强化了这种抑制。实用性却不可组合：……（原文摘要在此处截断）

    arXiv:2609.00445v1 Announce Type: cross  Abstract: Deployed language model safeguards (safety fine-tuning, filtering, unlearning) vary by principal only outside the model weights: filters are reconfigured, tiers are multiplied, and artefacts are reissued; inside one set of weights every request meets the same model configuration. This motivates us to define capability-gated deployment: per-principal access control inside one set of weights, whose configurations form a lattice - meets accumulate a principal's restrictions and joins pool a coalition's reach. We instantiate it by sparse rank gating over an existing nested-factorisation mechanism, guide profile search with one-pass attribution, and read every result once from a pre-registered held-out split. Security composes: provably at meets under a monotone-elicitation assumption we falsify pointwise. In two lineages the median held-out meet deepens suppression; the one effect surviving correction strengthens it. Utility does not: indi
    
[^290]: 迈向智能体化云工程：基于零信任智能体套件的图工程与循环工程

    Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness

    [https://arxiv.org/abs/2609.00050](https://arxiv.org/abs/2609.00050)

    提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。

    

    智能体AI正在推动基于云的工作流的发展，其中自主智能体可以对运营状态进行推理、调用授权工具、修改软件和基础设施、部署服务、验证执行结果，并在长时程、多步骤任务中进行自适应调整。构建此类工作流需要针对工作流推进、受限执行、故障恢复和可验证完成等环节的显式机制。我们提出了智能体云工作流工程，这是一个智能体AI框架，它将自然语言描述的智能体云工程任务转化为经过验证的代码仓库和经过验证的运营性云部署，从而实现基于云的智能体工作流自动化。该框架分离了三个互补的关注点：图工程负责指定长时程工作流推进以及依赖验证的状态转移；循环工程提供有界的诊断、修复或重新规划、重试和重新验证；智能体套件工程则（负责执行零信任的受限执行控制）。

    arXiv:2609.00050v1 Announce Type: cross  Abstract: Agentic AI is enabling cloud-based workflows in which autonomous agents reason over operational state, invoke authorized tools, modify software and infrastructure, deploy services, verify execution outcomes, and adapt across long-horizon, multistep tasks. Engineering such workflows requires explicit mechanisms for workflow progression, constrained execution, failure recovery, and verifiable completion. We present Agentic Cloud Workflow Engineering, an agentic AI framework that transforms natural-language agentic cloud-engineering tasks into validated code repositories and verified operational cloud deployments for automating cloud-based agentic workflows. The framework separates three complementary concerns: graph engineering specifies long-horizon workflow progression and verification-dependent transitions; loop engineering provides bounded diagnosis, repair or re-planning, retry, and re-verification; and agent harness engineering enf
    
[^291]: 面向工业时间序列预测的人机协同自主智能体

    A Human-in-the-Loop Autonomous Agent for Industry Time Series Forecasting

    [https://arxiv.org/abs/2608.30976](https://arxiv.org/abs/2608.30976)

    该论文提出了 CastClaw，一个面向工业时间序列预测的人机协同自主智能体系统，它在统一运行时中整合数据、专业预测模型、分析工具与用户输入，能够检验时间模式与用户约束、在证据不足时自动检索补充信息，并支持用户以自然语言交互式地制定和迭代优化预测任务。

    

    现实世界中的时间序列预测很少是一次性的模型调用：从业者需要制定任务、连接数据与模型、融入领域专业知识、评估预测的合理性，并传达不确定性。专业化的预测模型能够提供强大的数值预测，但通常在固定流程中运行；而通用的大语言模型（LLM）智能体往往缺乏针对预测任务的检验机制、约束条件和停止规则。我们提出了 CastClaw，一个通过面向预测的工程化框架构建的人机协同自主预测系统。CastClaw 在同一运行时环境中连接数据、专业模型、分析工具、用户输入以及带版本控制的执行记录。用户可以用自然语言指定预测目标、时间跨度、约束条件和假设。CastClaw 从用户提供的或模型生成的预测出发，检验时间模式与用户约束；当证据缺失时，它会检索相关上下文信息以补充判断依据。

    arXiv:2608.30976v1 Announce Type: new  Abstract: Real-world time-series forecasting is rarely a one-shot model invocation: practitioners must formulate tasks, connect data and models, incorporate domain expertise, assess prediction plausibility, and communicate uncertainty. Specialized forecasting models provide strong numerical predictions but usually operate in fixed pipelines, while general-purpose large language model (LLM) agents often lack forecasting-specific checks, constraints, and stopping rules. We present CastClaw, a human-in-the-loop autonomous forecasting system built through forecasting-oriented harness engineering. CastClaw connects data, specialized models, analytical tools, user input, and a versioned execution record in one runtime. Users specify the target, horizon, constraints, and hypotheses in natural language. Starting from a supplied or model-generated forecast, CastClaw checks temporal patterns and user constraints; when evidence is missing, it retrieves conte
    
[^292]: BenthicDINO：基于物理信息的自蒸馏方法用于侧扫声呐视角不变表示学习

    BenthicDINO: Physics-Informed Self-Distillation for View-Invariant Side-Scan Sonar Representations

    [https://arxiv.org/abs/2608.23215](https://arxiv.org/abs/2608.23215)

    本文提出了一种基于物理信息的自蒸馏框架（BenthicDINO），通过物理模拟增强和HSIC惩罚强制侧扫声呐图像表示的视角不变性，有效解决了声学伪影导致的特征混合问题。

    

    arXiv:2608.23215v1 公告类型：交叉 摘要：侧扫声呐（SSS）图像中的自动感知受到物理声学伪影的严重阻碍，导致表示不可避免地混合了海底固有反射率与瞬态观测几何。现有的自监督学习（SSL）框架依赖于为自然图像设计的增强方法，未能考虑声学退化，也未能明确强制视角不变性。为解决这一差距，我们提出了一种基于物理信息的自蒸馏框架，构建在DINOv3架构之上，采用ConvNeXt-v2-Tiny骨干网络以最大化数据效率。该方法通过两种主要机制强制视角不变性：物理驱动的增强方法，模拟散斑噪声、距离相关衰减和辐射定标误差；以及希尔伯特-施密特独立性准则（HSIC）惩罚项，明确将学习到的密集补丁特征与物理观测参数解耦。

    arXiv:2608.23215v1 Announce Type: cross  Abstract: Automated perception in side-scan sonar (SSS) imagery is severely hindered by physical acoustic artifacts, resulting in representations that inextricably mix intrinsic seabed reflectivity with transient viewing geometries. Existing self-supervised learning (SSL) frameworks rely on augmentations designed for natural images, failing to account for acoustic degradation and explicitly enforce view-invariance. To address this gap, we introduce a physics-informed self-distillation framework built upon the DINOv3 architecture utilizing a ConvNeXt-v2-Tiny backbone to maximize data efficiency. The proposed methodology enforces view-invariance through two primary mechanisms: physically motivated augmentations that simulate speckle noise, range-dependent attenuation, and radiometric miscalibration; and a Hilbert-Schmidt Independence Criterion (HSIC) penalty that explicitly decouples learned dense patch features from physical viewing parameters. F
    
[^293]: 分布时序差分学习中的在线推断

    Online Inference in Distributional Temporal-Difference Learning

    [https://arxiv.org/abs/2608.14408](https://arxiv.org/abs/2608.14408)

    本文为分布时序差分学习提供了在线推断的理论框架，证明了平均估计器和自助法的渐近正态性，并扩展到了非光滑泛函的局部推断。

    

    我们研究了在固定策略下，针对回报分布泛函的在线统计推断问题。回报分布通过非参数分布时序差分学习从单一马尔可夫轨迹中估计。对于Polyak-Ruppert平均估计器，我们证明了其根号T误差在Cramér空间中弱收敛于中心高斯随机元素。我们还证明了，在观测轨迹的条件下，自助法与原始平均值之间的根号T差异弱收敛于相同的高斯极限。这些结果证明了对于光滑统计泛函（包括方差、CVaR、期望短缺和期望分位数）的自助推断是合理的。对于非光滑统计泛函，我们开发了在有限多个阈值的$T^{-1/2}$邻域内估计回报CDF的局部渐近理论，以及其自助对应版本。该理论使我们能够进行推断。

    arXiv:2608.14408v1 Announce Type: cross  Abstract: We study online statistical inference for functionals of the return distribution under a fixed policy. The return distribution is estimated by nonparametric distributional temporal-difference learning from a single Markov trajectory. For the Polyak--Ruppert averaged estimator, we prove that its root-$T$ error converges weakly to a centered Gaussian random element in Cram\'er space. We also prove that, conditionally on the observed trajectory, the root-$T$ difference between the bootstrap and original averages converges weakly to the same Gaussian limit. These results justify bootstrap inference for smooth statistical functionals, including variance, CVaR, expected shortfall, and expectiles. For nonsmooth statistical functionals, we develop a local asymptotic theory for the estimated return CDF over $T^{-1/2}$-neighborhoods of finitely many thresholds, together with its bootstrap analogue. This theory allows us to conduct inference for 
    
[^294]: 分布强化学习中分位数时序差分学习的在线推断

    Online Inference for Quantile Temporal Difference Learning in Distributional Reinforcement Learning

    [https://arxiv.org/abs/2608.12973](https://arxiv.org/abs/2608.12973)

    本文提出了一种基于随机缩放的在线推断方法，通过泛函中心极限定理支持，在不存储完整轨迹的情况下对分布强化学习中的QTD进行统计推断，显著减少了内存需求。

    

    本文研究了分布强化学习中分位数时序差分学习（QTD）的统计推断方法。假设可以访问生成模型，我们首先为同步和异步QTD建立了泛函中心极限定理，表明QTD的平均迭代弱收敛于一个重新缩放的布朗运动。接着，我们提供了在线推断方法。基于随机缩放，该推断过程利用整个QTD路径上的信息构建了一个渐近枢轴统计量进行推断。同时，所提出的统计量可以在线计算，无需存储QTD迭代的完整轨迹。这大大降低了内存需求，并实现了分布强化学习中的高效统计推断。

    arXiv:2608.12973v1 Announce Type: cross  Abstract: In this paper, we study how to perform statistical inference for quantile temporal difference learning (QTD) in distributional reinforcement learning. Assuming access to a generative model, we first establish functional central limit theorems for both synchronous and asynchronous QTD, which show that the averaged iterates of QTD converge weakly to a rescaled Brownian motion. We next provide online inference methods. Based on random scaling, the inference procedure constructs an asymptotically pivotal statistic for inference by using the information along the whole QTD path. Meanwhile, the proposed statistic can be computed online without storing the entire trajectory of QTD iterates. This substantially reduces the memory requirement and enables efficient statistical inference in distributional reinforcement learning.
    
[^295]: RoutePack：面向MoE强化学习的专家放置与注意力感知数据打包方法

    RoutePack: Expert Placement and Attention-Aware Data Packing for MoE Reinforcement Learning

    [https://arxiv.org/abs/2608.12146](https://arxiv.org/abs/2608.12146)

    RoutePack通过分层规划器同时优化MoE强化学习中的专家放置与数据打包，协调注意力与专家负载平衡，以最小化执行行数并提升训练效率。

    

    摘要：arXiv:2608.12146v1 公告类型：交叉 摘要：训练用于强化学习（RL）的混合专家（MoE）模型耦合了两个负载平衡问题：序列组合决定了每个数据并行微批次中的密集注意力工作，而令牌路由则决定了专家并行等级上的稀疏专家工作。单独优化其中任何一个都可能将瓶颈转移到另一个。在MoE强化学习中，推出时的路由重放在训练步骤之前暴露了每个样本的序列长度和逐层专家需求。我们提出了RoutePack，一种分层规划器，它在优化器步骤窗口内协调状态一致、逐层专家重路由与联合注意力和专家感知的数据打包。RoutePack首先使用聚合路由需求在每层MoE中独立放置专家。然后，它将样本打包成最小可验证的或最佳已知可行的、受令牌上限的执行行数，并使用投影EDP分片感知目标优化其DP布局。该目标...

    arXiv:2608.12146v1 Announce Type: cross  Abstract: Training Mixture-of-Experts (MoE) models for reinforcement learning (RL) couples two load-balancing problems: sequence composition determines dense attention work in each data-parallel microbatch, while token routing determines sparse expert work on expert-parallel ranks. Optimizing either alone can shift the bottleneck to the other. In MoE RL, rollout-time routing replay exposes every sample's sequence length and layer-wise expert demand before its training step. We present RoutePack, a hierarchical planner that coordinates state-consistent, layer-wise expert rerouting with joint attention- and expert-aware data packing over an optimizer-step window. RoutePack first places experts independently at each MoE layer using aggregate routing demand. It then packs samples into the smallest certified, or best-known feasible, number of token-capped execution rows and optimizes their DP layout with a projected EDP-shard-aware objective. The obj
    
[^296]: softmax注意力的量子路线图：概率单纯形上softmax注意力的精确玻恩规则类比

    A Quantum Roadmap for Softmax Attention: Exact Born-Rule Analogs for Softmax Attention on the Probability Simplex

    [https://arxiv.org/abs/2608.11173](https://arxiv.org/abs/2608.11173)

    该论文证明在输入输出约束于概率单纯形的场景下，softmax注意力可以逐分量地精确映射为量子测量过程（Hadamard检验得分、玻恩规则余弦平方族、后选择温度与确定性数值聚合信道），为注意力的量子实现提供了完整路线图。

    

    注意力机制构成了许多现代AI模型（如Transformer）的基础。在使用注意力机制的一类子问题中，输入和输出被约束在概率单纯形上，使得所有输出之和为1。在这种设定下，softmax注意力可以实现精确的、逐分量的量子对应实现。注意力得分是对振幅编码输入的块编码投影进行Hadamard检验所得的统计量。指数softmax是在精确双射下由玻恩规则测量所生成的余弦平方族的内部，其边界在有限参数值处以精确零点的形式表达稀疏注意力。softmax温度对应于一个重复次数，其中后选择测量轮次可精确实现离散化的逆温度。数值聚合是一个确定性的列加载信道，它对列随机的数值矩阵进行了扩张。门控残差是……的制备角度（原文摘要在此处截断）。

    arXiv:2608.11173v2 Announce Type: replace-cross  Abstract: The attention mechanism forms the foundation of many modern AI models such as the Transformer. In one subclass of problems where attention is used, inputs and outputs are bound to the probability simplex so that all outputs sum to one. In this setting, softmax attention admits an exact, component-by-component quantum realization. Attention scores are Hadamard-test statistics on block-encoded projections of amplitude-encoded inputs. The exponential softmax is the interior of a cosine-squared family generated by Born-rule measurement under an exact bijection, whose boundary expresses sparse attention with exact zeros at finite parameter values. The softmax temperature is a repetition count where post-selected measurement rounds realize discretized inverse temperature exactly. Value aggregation is a deterministic column-loading channel that dilates the column-stochastic value matrix. The gated residual is the preparation angle of 
    
[^297]: 图学习能学习电路吗？

    Can Graph Learning Learn Circuits?

    [https://arxiv.org/abs/2608.08536](https://arxiv.org/abs/2608.08536)

    本文提出图电路学习（GCL）框架，首次将机制可解释性中的电路定位任务转化为图机器学习问题，通过图神经网络在多个模型-任务对上进行有监督的摊销式训练并泛化到未见案例，在留出案例上取得了0.946的中位数边AUROC。

    

    电路定位是一项机制可解释性任务，其目标是识别变换器计算图中足以重现特定行为的稀疏子图。大多数已有方法针对每个模型-任务对独立地定位电路。我们转而将电路定位表述为一个图机器学习问题，其中计算图的边表示计算路径，图神经网络（GNN）对这些路径之间的交互进行建模。我们提出了图电路学习（GCL），这是一个有监督的、摊销式框架，它在多个模型-任务对上训练GNN，并将其应用于未见过的案例。为了提供充足的数据，我们利用源自TracrBench程序的额外案例扩充了InterpBench基准。在评估的46种GCL配置中，由交叉验证选出的配置在16个留出案例上取得了0.946的中位数边AUROC（四分位区间为[0.841, 0.980]）。

    arXiv:2608.08536v2 Announce Type: replace  Abstract: Circuit localization is a mechanistic interpretability task whose goal is to identify a sparse subgraph of a transformer's computation graph sufficient to reproduce a particular behavior. Most established methods localize circuits independently for each model-task pair. We instead frame circuit localization as a graph machine learning problem in which the edges of a computation graph represent computational pathways, and graph neural networks (GNNs) model interactions among these pathways. We introduce Graph Circuit Learning (GCL), a supervised, amortized framework that trains a GNN across multiple model-task pairs and applies it to unseen cases. To provide sufficient data, we augment the InterpBench benchmark with additional cases derived from TracrBench programs. Of the 46 evaluated GCL configurations, the one selected by cross-validation scored a median edge AUROC of 0.946 (interquartile interval [0.841, 0.980]) on the 16 held-out
    
[^298]: 无需滚动模拟的任务难度预测

    Predicting Task Difficulty Without Rollouts

    [https://arxiv.org/abs/2608.05797](https://arxiv.org/abs/2608.05797)

    该论文提出无需实际运行智能体（rollout）即可在覆盖编码、数学、网页导航等17个智能体基准上进行事前任务难度预测，从而降低试错带来的计算成本，并揭示AUC等常用指标可能掩盖较差的难度估计。

    

    评估和训练自主智能体的一个根本挑战，是衡量它们所尝试任务的内在难度。对这一指标的估计对于创建合成数据或基准的环境设计者，以及构建训练课程都非常有用，从而提供了一种降低计算成本的途径。随着智能体（尤其是基于大语言模型的智能体）进入更长时程的领域，这种估计变得愈发重要，因为经验性的试错会成为严重的计算瓶颈。然而，现有工作大多局限于静态任务，并且所依赖的评估指标——正如我们所展示的——可能无法完整反映预测性能。在本文中，我们研究了涵盖编码、数学、机器学习、网页导航、函数调用及其他领域的17个智能体基准上的事前（ex ante）难度预测。我们发现，AUC可能掩盖较差的难度估计，并识别出……（原文摘要在此处截断）

    arXiv:2608.05797v3 Announce Type: replace-cross  Abstract: A fundamental challenge in evaluating and training autonomous agents is measuring the intrinsic difficulty of the tasks they attempt. Estimating this quantity can be useful for environment designers creating synthetic data or benchmarks, as well as for constructing training curricula, thereby offering a way to reduce compute costs. Such an estimation becomes increasingly important as agents (particularly LLM-based) move into longer-horizon domains, where empirical trial-and-error becomes a severe computational bottleneck. However existing work is largely confined to static tasks and relies on evaluation metrics that, as we show, can give an incomplete picture of predictive performance. In this paper we study ex ante difficulty prediction across 17 agentic benchmarks spanning coding, mathematics, machine learning, web navigation, function calling, and other domains. We find that AUC can mask poor difficulty estimates, identify t
    
[^299]: 从不规则期权报价中反向学习潜在风险中性密度

    Inverse Learning of Latent Risk-Neutral Densities from Irregular Option Quotes

    [https://arxiv.org/abs/2607.27188](https://arxiv.org/abs/2607.27188)

    该论文发现准确的期权定价并不能保证准确恢复潜在风险中性密度，并通过数值条件分析揭示定价问题存在严重病态性——大量定价方向数值为零、不同密度可产生相同价格——从而解释了不同模型在基准测试中排名差异的根源。

    

    准确的期权价格并不意味着能够准确恢复潜在的风险中性密度。我们通过两个互补的基准来研究这一区别。受控基准暴露出用于潜在评估的模拟器真实密度，而按时间顺序排列的NIFTY基准仅测试留出的市场价格。在合成基准上，双分量对数正态混合模型在总价格、L^1、Wasserstein和固定尾部误差上均为最低。学习到的算子保持了更窄的优势：DeepONet相对于混合模型将1%分位数和方差误差分别降低了39.0%和34.6%，而在结构上误设的Merton族上，报价Transformer将L^1降低了16.4%。数值条件分析解释了这些排名为何可能不同：在施加质量和远期约束后，126个定价方向中有95个在数值上为零，且两个在L^1距离上相差0.061的密度在覆盖的行权价上产生完全相同的价格。（摘要在此处截断）

    arXiv:2607.27188v3 Announce Type: replace  Abstract: Accurate option prices do not imply accurate recovery of the latent risk-neutral density. We study this distinction with two complementary benchmarks. A controlled benchmark exposes simulator-truth densities for latent evaluation, while a chronological NIFTY benchmark tests only held-out market prices. A two-component lognormal mixture has the lowest aggregate price, $L^1$, Wasserstein, and fixed-tail errors on the synthetic benchmark. Learned operators retain narrower strengths: DeepONet reduces 1% quantile and variance error by 39.0% and 34.6% relative to the mixture, and a quote transformer reduces $L^1$ by 16.4% on the structurally misspecified Merton family. A numerical conditioning analysis explains why these rankings can differ: after enforcing mass and forward constraints, 95 of 126 pricing directions are numerically null, and two densities separated by $L^1 = 0.061$ produce identical prices on the covered strikes. On 524 hel
    
[^300]: PowerAtlas：面向电力-计算协同调度的可行性感知语言模型

    PowerAtlas: Feasibility-Aware Language Models for Electricity-Computing Co-Scheduling

    [https://arxiv.org/abs/2607.26710](https://arxiv.org/abs/2607.26710)

    该论文将电力-计算协同调度形式化为结构化决策任务，发布了带可执行约束检查的ECBench基准，并提出通过可行性感知GRPO训练的PowerAtlas，使语言模型能生成满足电网与服务约束的联合调度方案。

    

    电力-计算协同调度（ECCS）决定计算任务在何处、何时运行，以及发电机如何调度，因为任务的部署位置会改变节点的电力需求，而电力可用性则决定了任务能否得到服务。先前的工作通过优化方法或学习型控制器来解决这类耦合问题，而大型语言模型（LLM）正越来越多地被用于为调度和电力系统生成结构化决策。然而，求解器的运行时间在不同实例之间差异很大，且即使LLM的输出可以被正确解析，仍可能违反电网和服务约束。我们将ECCS形式化为一个结构化决策任务，并发布了ECBench——一个包含2,000个实例的基准数据集，其中附带Gurobi最优参考调度方案，以及针对电网限制、任务服务、成本和延迟的可执行检查。我们提出PowerAtlas，该方法通过监督初始化以及基于确定性评分器反馈的可行性感知GRPO来训练LLM，并能在单次……（原文摘要在此处截断）

    arXiv:2607.26710v2 Announce Type: replace  Abstract: Electricity-computing co-scheduling (ECCS) decides where and when computing tasks run together with how generators are dispatched, because task placement changes nodal demand while power availability determines whether tasks can be served. Prior work addresses such coupled problems with optimization or learned controllers, and large language models (LLMs) are increasingly used to generate structured decisions for scheduling and power systems. Solver runtimes vary across instances, and LLM outputs that parse correctly can still violate grid and service constraints. We formalize ECCS as a structured decision task and release ECBench, 2,000 instances with Gurobi-optimal reference schedules and executable checks for grid limits, task service, cost, and latency. We propose PowerAtlas, which trains an LLM with supervised initialization and Feasibility-Aware GRPO on deterministic scorer feedback, and generates the joint schedule in a single
    
[^301]: 无需重新训练的跨方言泛化：面向 MLIR 的模式派生约束解码基准与评估

    Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for MLIR

    [https://arxiv.org/abs/2607.18254](https://arxiv.org/abs/2607.18254)

    该论文提出从 MLIR 方言的操作定义规范（ODS）机械派生的推理时约束解码方法，无需重新训练即可实现跨方言泛化，并发布了涵盖三种方言共 410 个样本的四个自然语言到 MLIR 基准测试数据集。

    

    多层中间表示（MLIR）是现代机器学习编译器基础设施（TensorFlow、JAX/StableHLO、PyTorch Inductor、IREE）的基础，然而在代码语言模型的预训练语料库中仅以极微量出现。此外，MLIR 在设计上具有可扩展性：每个应用领域都会发布新的方言，因此针对每个方言分别微调模型的做法无法规模化。我们探究能否从每个方言的操作定义规范（ODS）中机械地推导出推理时先验知识，以替代基于梯度的适配。首先，我们发布了涵盖三种方言的四个自然语言到 MLIR 的基准测试（MLIR-Spec-150、Linalg-Spec-30、StableHLO-Spec-30、StableHLO-Held-Out-200），共包含 410 个范围内的自然语言到 MLIR 配对样本，外加一个包含 25 个程序的语法外压力测试集，其中 30 个 Spec 提示词附有功能性参考实现，全部以 Apache-2.0 许可证发布，并附有 Gebru 数据表和 Croissant 1.1 元数据。其次，我们构建了一个三层的模式派生约束栈：基于操作签名的上下文无关文法（CFG）……（原文摘要在此处截断）

    arXiv:2607.18254v2 Announce Type: replace  Abstract: Multi-Level Intermediate Representation (MLIR) underlies modern ML compiler infrastructure (TensorFlow, JAX/StableHLO, PyTorch Inductor, IREE), yet appears only in trace amounts in code-LM pretraining corpora. MLIR is also extensible by design: new dialects ship per application domain, so a fine-tuned model per dialect does not scale. We ask whether inference-time priors derived mechanically from each dialect's Operation Definition Specification (ODS) can substitute for gradient-based adaptation. First, we release four natural-language-to-MLIR benchmarks across three dialects (MLIR-Spec-150, Linalg-Spec-30, StableHLO-Spec-30, StableHLO-Held-Out-200), totaling 410 in-scope NL-to-MLIR pairs, plus a 25-program out-of-grammar stress set, with functional references on 30 Spec prompts, under Apache-2.0 with Gebru datasheets and Croissant 1.1 metadata. Second, we build a three-layer schema-derived constraint stack: a CFG over op signatures 
    
[^302]: 将通用车辆模型适配用于跨地形高速模型预测控制（MPC）

    Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains

    [https://arxiv.org/abs/2607.13319](https://arxiv.org/abs/2607.13319)

    OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。

    

    高速越野自动驾驶需要对目标车辆进行精确的闭环控制，同时在多变地形上保持鲁棒性。近期的前向运动动力学（FKD）预测基础模型展现出一条有前景的路径：从通用模型出发，将其专用于目标平台。然而，有效的专业化仍然具有挑战性，因为它通常需要大量真实世界数据，而且适配到某一环境的模型仍可能对特定地形或驾驶模式过拟合。我们提出了OptCar（优化汽车），这是一种弥合通用FKD模型到专用FKD模型差距的方法，在为特定车辆优化性能的同时保留了跨地形泛化能力。OptCar引入了一种基于transformer的FKD架构，利用FiLM将多步预测条件化于单个动力学上下文token，该token概括了近期的状态-动作历史。随后，该方法利用有限的真实世界数据对通用模型进行专业化（摘要在此处截断）。

    arXiv:2607.13319v2 Announce Type: replace-cross  Abstract: High-speed off-road autonomy requires precise closed-loop control for a target vehicle while remaining robust across changing terrains. Recent forward kinodynamic (FKD) prediction foundation models suggest a promising path, starting from a generalist model and specializing it to the target platform. However, effective specialization remains challenging, as it often requires substantial real-world data, and models adapted to one setting can still overfit to specific terrains or driving regimes. We present OptCar (Optimized Car), a recipe for bridging the gap from generalist to specialist FKD models that preserves cross-terrain generalization while optimizing performance for a specific vehicle. OptCar introduces a transformer FKD architecture that uses FiLM to condition multi-step predictions on a single dynamics context token summarizing recent state-action history. It then specializes the generalist model using limited real-wor
    
[^303]: 教导大语言模型在代表性不足的癫痫诊疗中进行推荐与转诊

    Teaching LLMs to Recommend and Defer in Underrepresented Epilepsy Care

    [https://arxiv.org/abs/2606.31036](https://arxiv.org/abs/2606.31036)

    提出MANANA非参数化提示学习框架，将处方错误转化为可审计的提示记忆，使大语言模型能从少量本地患者数据中学习乌干达儿科癫痫的本地化抗癫痫药物处方方案，并在需要时转诊至专科医师。

    

    在资源受限的环境中，癫痫专科知识十分稀缺。临床决策支持系统必须适应当地处方实践，并能识别何时需要专科医师审查。我们研究乌干达的儿科癫痫诊疗，从纵向门诊记录中预测抗癫痫药物方案。通用语言模型与医生处方达到了可观的吻合度，但神经科医生的审查揭示了其中存在默认遵循西方处方惯例、未考虑当地实践的问题。我们提出了MANANA，一个非参数化的提示学习框架，能够从少量患者级训练集中学习本地处方指导。MANANA将处方错误转化为可审计的提示记忆，提供单智能体和多智能体两种变体，并在两个独立收集的乌干达队列中优于经典机器学习模型、直接LLM提示和提示优化基线。所学到的提示轨迹也

    arXiv:2606.31036v2 Announce Type: replace  Abstract: Specialist epilepsy expertise is scarce in resource-constrained settings. Clinical decision-support systems must adapt to local prescribing practices and recognize when specialist review is needed. We study pediatric epilepsy care in Uganda, predicting anti-seizure medication regimens from longitudinal clinic notes. General-purpose language models achieve non-trivial agreement with physician prescriptions, but neurologists' review reveals Western prescribing defaults that do not account for local practices. We introduce MANANA, a non-parametric prompt-learning framework that learns local prescribing guidance from a small patient-level training set. MANANA turns prescription errors into auditable prompt memory, with single-agent and multi-agent variants, and improves over classical ML models, direct LLM prompting, and prompt-optimization baselines across two independently collected Ugandan cohorts. The learned prompt trajectory also p
    
[^304]: 面向长时程智能体强化学习的群组-图策略优化

    Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning

    [https://arxiv.org/abs/2606.22995](https://arxiv.org/abs/2606.22995)

    提出群组-图策略优化（G2PO），一种针对多轮智能体任务的新型分组强化学习算法，通过显式利用状态转换固有的图结构，解决了长时程智能体强化学习中奖励稀疏延迟、高方差价值估计以及短视局部化信用分配的关键瓶颈。

    

    基于分组的强化学习已在智能体场景中显著增强了大语言模型（LLMs）的能力。为了实现更细粒度的策略更新，近期的智能体强化学习框架已从轨迹级训练转向步骤级训练。然而，长时程智能体强化学习面临着严重的奖励稀疏与延迟问题，因为反馈往往要推迟数十个交互步骤之后才会出现。尽管现有的步骤级框架细化了训练粒度，但其信用分配仍然是粗粒度的，并且依然将智能体的探索视为孤立的线性轨迹。这种过于简化的视角忽略了状态转换中固有的图结构，导致了高方差的状态价值估计以及短视的、局部化的信用分配。为了克服这些关键瓶颈，我们提出了群组-图策略优化（G2PO），这是一种专为多轮智能体任务量身定制的新型分组强化学习算法。G2PO 显式地将……（摘要在此处截断）

    arXiv:2606.22995v2 Announce Type: replace-cross  Abstract: Group-based Reinforcement Learning (RL) has significantly enhanced Large Language Models (LLMs) in agentic scenarios. To achieve finer-grained policy updates, recent agentic RL frameworks have shifted from trajectory-level to step-level training. However, long-horizon agentic RL suffers from severe reward sparsity and delay, as feedback is often deferred for dozens of interaction steps. While existing step-level frameworks refine training granularity, their credit assignment remains coarse-grained and still treats agent exploration as isolated, linear trajectories. This oversimplified perspective ignores the inherent graph structure of state transitions, leading to high-variance state-value estimation and myopic, localized credit assignment. To overcome these critical bottlenecks, we propose Group-Graph Policy Optimization (G2PO), a novel group-based RL algorithm tailored for multi-turn agentic tasks. G2PO explicitly transforms
    
[^305]: 学习对您的VLA说什么：基本无害的视觉-语言-动作模型引导

    Learning What to Say to Your VLA: Mostly Harmless Vision Language Action Model Steering

    [https://arxiv.org/abs/2606.12299](https://arxiv.org/abs/2606.12299)

    提出一个通过交互式搜索有效语言序列并将其蒸馏为测试时语言反馈策略的框架，同时利用共形化改进头来预测并阻止有害的语言引导干预，从而可靠地提升视觉-语言-动作模型的闭环任务性能。

    

    视觉-语言-动作（VLA）模型为机器人控制提供了自然语言接口，但从语言到行为的映射往往脆弱且不直观：语义相似的指令可能引发截然不同的行为，而某些能力可能无法仅通过提示来激发。因此，无论是人类指令还是零样本语言模型，都可能无法可靠地引导VLA成功执行任务。在本工作中，我们提出了一个框架，该框架通过交互式搜索来寻找能够提升闭环VLA任务性能的语言序列，将这些序列蒸馏为测试时的语言反馈策略（LFP），并学习一个改进头来预测语言引导何时能够提升性能。我们对这个改进头进行共形化处理，以防止有害的引导干预，即在分布外场景中LFP相对于原始指令降低任务性能的情况。

    arXiv:2606.12299v2 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models provide a natural language interface to robot control, but the mapping from language to behavior is often brittle and unintuitive: semantically similar instructions can induce drastically different behaviors, while some capabilities may not be elicitable through prompting alone. As a result, both human instructions and zero-shot language models can fail to reliably steer VLAs toward successful task execution. In this work, we propose a framework that interactively searches for language sequences that improve closed-loop VLA task performance, distills these sequences into a test-time language feedback policy (LFP), and learns an improvement head that predicts when language steering will improve performance. We conformalize this improvement head to prevent harmful steering interventions, where the LFP decreases task performance relative to the original instruction on out-of-distribution scenari
    
[^306]: TargetSEC：基于唤醒条件化潜在风格扩散的即插即用野外语音情感转换

    TargetSEC: Plug-and-Play In-the-Wild Speech Emotion Conversion via Arousal-Conditioned Latent Style Diffusion

    [https://arxiv.org/abs/2606.07293](https://arxiv.org/abs/2606.07293)

    TargetSEC提出了一种以说话人身份和连续情感为条件的潜在风格扩散框架，通过在紧凑潜在空间中生成情感风格嵌入，实现了即插即用的野外语音情感转换，在转换效果和语音自然度上均优于或持平现有基线方法。

    

    语音情感转换（SEC）旨在将源语音的情感转换为目标情感，同时保留语音内容和说话人身份。由于训练数据的非平行特性以及复杂多变的真实世界声学环境，在野外（in-the-wild）数据上进行SEC极具挑战性。现有的固定时长方法要么难以有效转换情感（高质量但低转换率），要么会损害语音的自然度（高转换率但低质量）。我们提出了TargetSEC，这是一个由嵌入驱动的潜在扩散框架，它以说话人身份和连续情感（唤醒度）为条件，生成以情感为中心的风格嵌入。与在频谱图上进行扩散的方法不同，TargetSEC在紧凑的潜在空间中运行。在MSP-Podcast数据集上的实验表明，TargetSEC超越了现有的非时长基线方法，并在转换MSE和自然度方面达到甚至超过了时长预测基线，且无需显式的时间建模。

    arXiv:2606.07293v2 Announce Type: replace-cross  Abstract: Speech Emotion Conversion (SEC) aims to transform the emotion of a source utterance into a target emotion while preserving content and speaker identity. SEC on in-the-wild data is challenging due to the non-parallel nature of training data and complex real-world acoustics. Existing fixed-duration approaches either struggle to shift the emotion effectively (high quality, low conversion) or degrade speech naturalness (low quality, high conversion). We propose TargetSEC, an embedding-driven latent diffusion framework that generates emotion-focused style embeddings conditioned on speaker identity and continuous emotion. Unlike methods that diffuse over spectrograms, TargetSEC operates in a compact latent space. Experiments on the MSP-Podcast dataset show that TargetSEC outperforms current non-duration baselines and matches or exceeds the duration-prediction baseline in conversion MSE and naturalness without explicit temporal modeli
    
[^307]: PE-MHL：用于复杂系统可扩展学习的物理编码模块化混合层

    PE-MHL: Physics-Encoded Modular Hybrid Layers for Scalable Learning of Complex Systems

    [https://arxiv.org/abs/2606.04290](https://arxiv.org/abs/2606.04290)

    提出物理编码模块化混合层框架PE-MHL，通过增量添加子模型逐步细化基线物理模型，并从理论上证明训练误差随子模型数量单调不增且可证明收敛，从而实现复杂系统学习的可扩展性与鲁棒性。

    

    将基于物理的组件与数据驱动的组件相结合的混合模型，在控制应用中实现准确性和可解释性方面展现出了巨大潜力。尽管近期方法在融入物理一致性方面取得了进展，但在可扩展性、对噪声的鲁棒性以及模型复杂度控制方面仍存在挑战。本文提出了一种物理编码模块化混合层框架，其中基线物理模型通过增量添加新的子模型来逐步细化，每个新组件在增加复杂度的同时保留先前组件已经学到的内容。我们为这种构造建立了理论保证：通过对每个新子模型采用最小二乘初始化，训练误差随子模型数量的增加而单调不增，并且可证明地收敛。在非线性NARX基准和Quanser Aero 2平台上的实证评估表明，PE-（摘要原文在此处截断）

    arXiv:2606.04290v2 Announce Type: replace  Abstract: Hybrid models that combine physics-based and data-driven components have shown strong potential for achieving accuracy and interpretability in control applications. While recent methods have made progress in incorporating physical consistency, challenges remain in scalability, robustness to noise, and control of model complexity. This paper proposes a Physics-Encoded Modular Hybrid Layer (PE-MHL) framework, in which a baseline physics-based model is incrementally refined through the addition of new sub-models, where each new component adds complexity while preserving what previous components have already learned. We establish a theoretical guarantee for this construction: with a least-squares initialization of each new sub-model, the training error is monotonically non-increasing in the number of sub-models and provably converges. Empirical evaluations on a nonlinear NARX benchmark and the Quanser Aero 2 platform demonstrate that PE-
    
[^308]: CauTion：知道何时信任大语言模型进行集成因果发现

    CauTion: Knowing When to Trust LLMs for Ensemble Causal Discovery

    [https://arxiv.org/abs/2606.03602](https://arxiv.org/abs/2606.03602)

    提出CauTion框架，通过共识过滤和LLM可靠性估计，将大语言模型的领域知识可靠地融合进统计因果发现算法集成中，从而同时克服纯统计方法、LLM错误风险和单一算法偏差这三方面的局限。

    

    从观测数据中进行因果发现仍然具有挑战性，原因在于纯统计方法存在根本性局限，例如等价类内部的统计不可区分性以及对有限样本量的敏感性。虽然大语言模型（LLM）为补充统计推断提供了有前景的领域知识来源，但现有的LLM增强方法容易受到LLM错误的影响，且产生高昂的token成本。此外，依赖单一以数据为中心的算法会使结果对算法特有的偏差敏感。为了解决这些局限性，我们提出了CauTion，一个通过共识过滤和LLM可靠性估计，将LLM领域知识可靠地集成到统计因果发现算法集成中的框架。CauTion分三个阶段进行。首先，算法集成利用共识投票解决多达96%的算法间达成一致的边，实现接近完美的（摘要在此处被截断）

    arXiv:2606.03602v2 Announce Type: replace-cross  Abstract: Causal discovery from observational data remains challenging due to the fundamental limitations of purely statistical methods, such as statistical distinguishability within equivalence classes and sensitivity to finite sample sizes. While large language models (LLMs) offer a promising source of domain knowledge to complement statistical inference, existing LLM-augmented methods are vulnerable to LLM errors and incur high token costs. Moreover, reliance on a single data-centric algorithm can make results sensitive to algorithm-specific biases. To address these limitations, we propose CauTion, a framework that reliably integrates LLM domain knowledge into an ensemble of statistical causal discovery algorithms through consensus filtering and LLM reliability estimation. CauTion proceeds in three stages. First, an algorithm ensemble utilizes a consensus voting to resolve up to 96% of edges on which algorithms agree, achieving near-p
    
[^309]: 超级神冈实验中基于深度学习的低能触发算法

    Deep-learning-based low-energy trigger algorithms for the Hyper-Kamiokande experiment

    [https://arxiv.org/abs/2605.31391](https://arxiv.org/abs/2605.31391)

    该研究为超级神冈实验开发了基于深度学习的低能中微子触发算法，其中监督分类器对3 MeV单电子的信号识别效率达76.7%，显著超越传统击中计数触发器的26.4%。

    

    现代机器学习技术凭借其强大的模式识别能力，在粒子物理学中变得日益重要，包括在具有严格运行时间约束的实时数据采集中。本文详细介绍了基于深度学习的触发算法在超级神冈这类大型水切伦科夫探测器上的性能表现，旨在探测低能中微子事件（低于7 MeV）。文中展示了自定义神经网络监督分类器的性能，以及两种仅使用探测器噪声进行训练的异常检测方法：纯自编码器和基于流形投影-扩散恢复的模型。监督模型对动能为3 MeV的单电子表现出76.7%的信号识别效率，显著超过传统基于击中计数的触发器所获得的26.4%的信号效率，而流形投影-扩散恢复方法达到了……

    arXiv:2605.31391v2 Announce Type: replace-cross  Abstract: Modern machine learning techniques have become increasingly important in particle physics because of their powerful pattern-recognition capabilities, including in real-time data acquisition where stringent runtime constraints apply. This paper details the performance of deep-learning-based trigger algorithms for a large water Cherenkov detector such as Hyper-Kamiokande, aimed at low-energy neutrino events (below 7 MeV). The performance of custom neural-network supervised classifiers is shown alongside two anomaly-detection approaches trained solely on detector noise: a pure autoencoder and a model based on Manifold Projection-Diffusion Recovery. The supervised model shows signal identification efficiencies of 76.7% for single electrons of 3 MeV kinetic energy, significantly exceeding signal efficiencies obtained from a traditional hit-count-based trigger of 26.4%, while the Manifold Projection-Diffusion Recovery approach reache
    
[^310]: TASER：面向几何驱动平滑性的任务感知Stein正则化

    TASER: Task-Aware Stein Regularisation for Geometry-Driven Smoothness

    [https://arxiv.org/abs/2605.30601](https://arxiv.org/abs/2605.30601)

    TASER通过惩罚Langevin Stein算子将模型导数与数据密度得分耦合，提出了一种无需切空间估计或内层扰动搜索的几何驱动正则化框架，有效提升深度神经网络在分布偏移下的稳定性与泛化能力。

    

    深度神经网络对分布偏移仍然敏感，这促使人们寻求基于数据几何的正则化方法。我们提出了TASER（任务感知Stein正则化），这是一个紧凑的训练时框架，通过惩罚Langevin Stein算子将模型导数与密度得分耦合起来。我们建立了加权Sobolev解释，并在逐渐增强的分布假设下给出了一系列层次化的稳定性界。此外，我们还提出了基于一阶Stein算子的TASERLight。TASER和TASERLight既不需要显式的切空间估计，也不需要内层扰动搜索。受控的合成示例说明了它们对外推能力和方向敏感性的影响。在光谱学、骨骼运动、心电图（ECG）、MNIST和CIFAR-10数据集上的实验表明，该方法在分布偏移下带来了性能提升，并且与各向同性正则化结合时具有互补收益。

    arXiv:2605.30601v2 Announce Type: replace  Abstract: Deep neural networks remain sensitive to distribution shifts, motivating regularisation informed by data geometry. We introduce TASER (Task-Aware Stein Regularisation), a compact training-time framework that couples model derivatives to density scores by penalising the Langevin Stein operator. We establish a weighted Sobolev interpretation and a hierarchy of stability bounds under progressively stronger distributional assumptions. Moreover we introduce TASERLight, which is based on a first-order Stein operator. TASER and TASERLight require neither explicit tangent-space estimation nor an inner perturbation search. Controlled synthetic examples illustrate their effects on extrapolation and directional sensitivity. Experiments on spectroscopy, skeleton motion, ECG, MNIST, and CIFAR-10 demonstrate gains under distribution shift and complementary benefits when combined with isotropic regularisation.
    
[^311]: 学习对隐式多模态危害进行推理：从诊断到泛化

    Learning to Reason About Implicit Multimodal Harm: From Diagnosis to Generalization

    [https://arxiv.org/abs/2605.29951](https://arxiv.org/abs/2605.29951)

    提出MuPHI诊断基准，揭示视觉-语言模型虽能理解单一模态信息，却无法将图像-文本组合推理为危害机制，为隐式多模态危害检测从诊断走向泛化提供了方向。

    

    一张无害的图片配上无害的文字说明，只有在组合在一起时才可能变得有害，这使得此类检测在本质上是一个跨模态推理问题。现有的多模态危害基准仅评估分类准确率，尚不清楚模型是否真正识别了产生危害的跨模态机制。我们提出了MuPHI（多模态语用危害解释），这是一个受控诊断基准，其中危害纯粹源于图像与文本的组合，并配有良性反事实样本和经过标注的危害成因解释。借助MuPHI，我们系统地评估了视觉-语言模型（VLMs）中组合式危害推理的失效环节。MuPHI揭示：VLMs能够可靠地理解单一模态的信息，却无法将证据关联到危害机制之中，且其预测对决定标签的反事实干预响应微弱。此外，现有基于标签训练的危害检测方法虽能显著提升域内（ID）检测性能，但……

    arXiv:2605.29951v2 Announce Type: replace  Abstract: A benign image paired with a benign caption can become harmful only in combination, making detection fundamentally a cross-modal reasoning problem. Existing multimodal harm benchmarks evaluate only classification accuracy, leaving unclear whether models identify the cross-modal mechanism that produces harm. We introduce MuPHI (Multimodal Pragmatic Harm Interpretation), a controlled diagnostic benchmark in which harm arises purely from image-text composition, with benign counterfactuals and annotated harm rationales. Using MuPHI, we systematically evaluate where compositional harm reasoning breaks in vision-language models (VLMs). MuPHI reveals that VLMs reliably ground each modality yet fail to bind the evidence into the harm mechanism and their predictions respond weakly to label-determining counterfactual interventions. Moreover, existing harm detection approaches trained on labels substantially improves in-domain (ID) detection bu
    
[^312]: FormInv：数学推理基准中语义不变性的测量协议

    FormInv: A Measurement Protocol for Semantic Invariance in Mathematical Reasoning Benchmarks

    [https://arxiv.org/abs/2605.29001](https://arxiv.org/abs/2605.29001)

    该论文提出FormInv测量协议，用于审计数学推理基准中自动生成改写题目的语义等价性，并引入条件不一致率（CIR）指标，揭示即使在所有模型都能正确回答的规范形式定理上，语义等价的改写仍会导致5-42%的答案翻转。

    

    对MathCheck（ICLR 2025）改写质量的审计在129个组中发现了4个语义错误的改写（3.1%）；移除这些错误后，GPT-4o的排名从第2位降至第4位，而Claude Haiku和DeepSeek V3则升至其上。利用跨模型一致性方法，这些错误在不到10美元的成本下被自动发现；在我们自己的数据集中，同一协议发现47%自动生成的连接词变化改写在语义上是错误的。这一缺陷还加剧了一个更深层的测量差距：在所有模型都能以规范形式正确回答的定理上，语义等价的改写仍会使其中5-42%的答案发生翻转；在所有模型共享的43个定理上，Claude Haiku 4.5翻转已知正确答案的频率约为DeepSeek V3的9倍（41.9%对4.7%），我们将这一难度受控的差距称为条件不一致率（CIR）。CIR排名在一个独立的103个定理集合上得到了复现。原始语义一致性率（SCR）在各模型之间的范围为50-82%（摘要在此处截断）。

    arXiv:2605.29001v2 Announce Type: replace-cross  Abstract: A paraphrase-quality audit of MathCheck (ICLR 2025) detected 4 semantically incorrect paraphrases in 129 groups (3.1%); removing them drops GPT-4o from rank 2 to rank 4 and elevates Claude Haiku and DeepSeek V3 above it. Cross-model unanimity found these errors automatically for under $10; in our own dataset the same protocol found that 47% of auto-generated connective-variation paraphrases were semantically incorrect. That flaw compounds a deeper measurement gap. On the theorems every model answers correctly in canonical form, a semantically equivalent rewording still flips the answer for 5-42% of them: Claude Haiku 4.5 flips a known-correct answer about 9x as often as DeepSeek V3 (41.9% vs. 4.7% on the 43 theorems all models share), a difficulty-controlled gap we call the Conditional Inconsistency Rate (CIR). The CIR ranking replicates on an independent 103-theorem set. Raw Semantic Consistency Rates (SCR) span 50-82% across 
    
[^313]: 高维稀疏更新下随机动量的动力学

    Dynamics of Stochastic Momentum with Sparse Updates in High Dimensions

    [https://arxiv.org/abs/2605.28961](https://arxiv.org/abs/2605.28961)

    该论文首次在高维稀疏更新场景下精确刻画了动量优化的二阶矩动力学，揭示了其相变结构由动量保持时间尺度与学习时间尺度之比决定：学习慢于保持时极限等价于SGD，学习快于保持时系统失稳。

    

    现有的动量理论假设梯度以大致恒定的速率到达每个参数，但这一假设在实践中会因重尾数据分布和现代架构而被打破。我们从理论上分析了稀疏更新下两个可解析求解的动量模型的动力学：一个是具有稀疏输入的最小二乘模型，另一个是具有稀有类别的逻辑回归模型。两者都具有精确的闭式二阶矩动力学，我们在稀疏度、批次大小和动量衰减的三个标度指数下刻画了其高维极限。两个问题的相结构均由两个内在时间尺度之比决定：动量保持时间尺度（缓冲区能存活多少次有效更新）和学习时间尺度（减少平方误差需要多少次有效更新）。当学习远慢于保持时，极限与SGD一致；当学习更快时，系统不稳定；……

    arXiv:2605.28961v2 Announce Type: replace-cross  Abstract: Existing theory of momentum assumes that gradients arrive at every parameter at a roughly constant rate, an assumption violated in practice by heavy-tailed data distributions and modern architectures. We theoretically analyze the dynamics of two tractable models of momentum under sparse updates: a least squares model with sparse inputs and a logistic regression model with a rare class. Both admit exact closed-form second-moment dynamics whose high-dimensional limits we characterize across three scaling exponents for sparsity, batch size, and momentum decay. The phase structure on both problems is governed by the ratio of two intrinsic timescales: a momentum retention timescale (how many active updates the buffer survives) and a learning timescale (how many active updates it takes to reduce the squared error). When learning is much slower than retention, the limit matches SGD; when learning is faster, the system is unstable; whe
    
[^314]: 多任务学习结合半经验轨道电荷实现样本高效的机器学习原子间势

    Multitask learning with semiempirical orbital charges enables sample-efficient MLIPs

    [https://arxiv.org/abs/2605.24073](https://arxiv.org/abs/2605.24073)

    通过多任务学习引入计算成本低廉的GFN1-xTB半经验轨道电荷，显著提升了机器学习原子间势的样本效率和预测准确性。

    

    机器学习原子间势（MLIPs）需要生成计算成本高昂的大规模训练数据集，才能准确模拟材料和分子。利用多任务学习引入电子结构信息可以提高样本效率，然而，在随原子数呈二次方增长的完整哈密顿矩阵上进行训练，对于大型数据集来说是难以实现的。在这项工作中，我们展示了利用轨道分辨半经验电荷的多任务学习显著提高了MLIPs的样本效率和准确性。为了高效预测轨道电荷，我们实现了一个专门设计的等变模型，与不变基线模型相比降低了电荷预测误差。通过在训练中引入计算成本低廉、随原子数呈线性扩展的GFN1-xTB轨道电荷，我们的模型相对于仅使用能量训练的基线模型，能量平均绝对误差降低了58%。

    arXiv:2605.24073v2 Announce Type: replace-cross  Abstract: Machine learning interatomic potentials (MLIPs) require generating computationally expensive, large-scale training datasets to accurately simulate materials and molecules. Incorporating electronic structure information using multitask learning improves sample efficiency, however, training on full Hamiltonian matrices, which scale quadratically with the number of atoms, is intractable for large datasets. In this work, we show that multitask learning utilizing orbitally resolved semiempirical charges significantly improves sample efficiency and accuracy in MLIPs. To efficiently predict orbital charges, we implement a specialized equivariant model, reducing charge prediction error compared to an invariant baseline. By augmenting training with computationally inexpensive GFN1-xTB orbital charges, which scale linearly with the number of atoms, our model achieves a 58% reduction in energy mean absolute error relative to energy-only b
    
[^315]: 以人为中心的学习力学：受保护风险约束下的可观测熵控制

    Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints

    [https://arxiv.org/abs/2605.22940](https://arxiv.org/abs/2605.22940)

    本文建立了在任务、风险与干预约束分离设定下的熵调控学习理论，通过噪声校准的对数行列式泛函导出精确谱力公式、可观测/预测零空间速度分解、标量反馈系数完整区间，以及多受保护风险下的极小极大可行性证书与锥投影公式，实现受保护风险约束下的可观测熵控制。

    

    熵降低并非一个安全准则：它可能使预测保持不变，或伴随着受保护群体风险的上升。我们研究在分别指定的任务、风险与干预约束下的熵调控学习。一个经噪声校准的对数行列式泛函给出了精确的谱力公式，以及将速度度量分解为可观测分量与预测零空间分量的方法。对于标量反馈，我们刻画了在保持任务下降比例并满足熵速率要求时系数的完整区间；而将干预限制在可观测方向上可能使该区间为空。对于多个受保护风险，我们在指定子空间与范数预算内推导出精确的极小极大可行性证书、针对不确定约束行的精确鲁棒扩展，以及保护后剩余熵削减的锥投影公式。这些公式采用标准的凸几何方法，但……

    arXiv:2605.22940v2 Announce Type: replace-cross  Abstract: Entropy reduction is not a safety criterion: it can leave predictions unchanged or accompany increasing risk for a protected group. We study entropy-regulated learning under separately specified task, risk, and intervention constraints. A noise-calibrated log-determinant functional yields an exact spectral force formula and a metric decomposition into observable and prediction-null velocities. For scalar feedback, we characterize the complete interval of coefficients preserving a task-descent fraction and meeting an entropy-rate request; restricting intervention to observable directions can make that interval empty. For several protected risks, we derive an exact minimax certificate of feasibility within a specified subspace and norm budget, an exact robust extension for uncertain constraint rows, and a cone-projection formula for the entropy reduction remaining after protection. These formulas use standard convex geometry but 
    
[^316]: ECPO：用于可验证审计候选排序的证据耦合策略优化

    ECPO: Evidence-Coupled Policy Optimization for Verifier-Audited Candidate Ranking

    [https://arxiv.org/abs/2605.21993](https://arxiv.org/abs/2605.21993)

    提出证据耦合策略优化（ECPO），通过证据循环奖励将Top-K排序与基于片段的可审计证据耦合，显著提升验证者可审计的可追溯性，且增益在多个基准、骨干模型与评估器下均保持稳健。

    

    用于决策支持的排序系统应当提供可独立于模型排序分数进行审计的证据。我们研究了一种封闭名单设定，其中策略同时输出Top-K排序以及与候选相关联的、基于文本片段的证据包。证据耦合策略优化（ECPO）结合了列表级效用、证书有效性，以及一种证据循环奖励——该奖励在确定性恢复之前移除包含答案的字段。在MAVEN-ERE和RAMS数据集上，在所测试的溯源设置中，ECPO相比匹配的对照方法提升了验证者可审计的可追溯性、排序盲恢复能力以及人类对比支持度。这些增益在留出评估器、其他8B/14B骨干模型以及HotpotQA答案排序接口下依然保持。常规的NDCG指标和运行时间则仍是明确的权衡代价。

    arXiv:2605.21993v2 Announce Type: replace  Abstract: Ranking systems used for decision support should expose evidence that can be audited independently of the model's ranking scores. We study a closed-roster setting in which a policy jointly emits a Top-$K$ ranking and candidate-linked, span-grounded evidence bundles. Evidence-Coupled Policy Optimization (ECPO) combines listwise utility, certificate validity, and an evidence-cycle reward that removes answer-bearing fields before deterministic recovery. On MAVEN-ERE and RAMS, ECPO improves verifier-audited traceability, rank-blind recovery, and human comparative support relative to matched controls in the tested provenance settings. The gains persist under held-out evaluators, other 8B/14B backbones, and a HotpotQA answer-ranking interface. Ordinary NDCG and runtime remain explicit trade-offs.
    
[^317]: 使用错误设定模拟器的探索与实验

    Exploration and Experimentation with a Misspecified Simulator

    [https://arxiv.org/abs/2605.21458](https://arxiv.org/abs/2605.21458)

    该论文研究了在模拟器存在错误设定且真实系统动态发生漂移的情况下，是否、何时以及如何通过被动适应、持续探索或随机化实验等数据收集策略来重新校准模拟器。

    

    随机化实验是衡量因果效应以指导决策的标准方法，但成本高昂。根据历史数据校准的模拟器提供了一种更廉价的替代方案，但在发生漂移（即校准后系统动态发生变化）时，其策略可能变得次优。我们研究是否、何时以及如何收集数据以重新校准模拟器。我们考虑一个由多个单元并行运行并由单一操作者管理的系统。现实世界的系统是一个带有潜变量的部分可观测过程，而模拟器将其建模为完全可观测的马尔可夫过程，因此存在错误设定。两种数据收集策略利用当前策略：（i）从不重新校准，（ii）从在策略数据进行重新校准（被动适应）。另外两种策略通过尝试当前策略之外的动作来收集离策略数据：（iii）在所有单元上进行持续探索，以及（iv）在部分单元样本上进行随机化实验。

    arXiv:2605.21458v2 Announce Type: replace  Abstract: Randomized experiments are the standard for measuring causal effects that guide decisions, but are costly. Simulators calibrated to historical data offer a cheaper alternative, but their policies can become suboptimal under drift, a change in dynamics after calibration. We study if, when, and how to collect data for recalibrating the simulator. We consider a system with multiple units running in parallel and managed by a single operator. The real-world system is a partially observed process with a latent variable, while the simulator models it as a fully observed Markovian process, and thus is misspecified. Two data collection strategies exploit the current policy: (i) never recalibrating and (ii) recalibrating from on-policy data (passive adaptation). Two others collect off-policy data by trying actions outside the current policy: (iii) continuous exploration across all units and (iv) randomized experimentation on a sample of units 
    
[^318]: 图直推锐化：在节点分类中利用未标注节点的预测

    Graph Transductive Sharpening: Leveraging Unlabeled Predictions in Node Classification

    [https://arxiv.org/abs/2605.20248](https://arxiv.org/abs/2605.20248)

    该论文提出直推锐化（TS）这一损失层面的修改方法，通过最小化未标注节点的预测熵来提取直推式模型对未标注节点预测中蕴含的训练信号，从而改进半监督节点分类。

    

    在直推式设置中，虽然可以观察到完整的图结构，但节点标签只有部分可用，半监督节点分类的进展很大程度上集中在架构创新上。在本文中，我们重新审视了一个与之正交的方向：训练目标。我们从一个简单的观察出发：直推式模型在训练过程中会对每个节点都产生预测，包括那些没有标签的节点。这些未标注节点的预测可能包含有用的训练信号，但由于缺乏真实标签，标准的监督目标会将它们丢弃。受交叉熵可分解为标签相关对齐项与标签无关熵项的启发，我们提出将预测置信度作为在缺乏标签情况下提取这一信号的自然方式。由此我们提出直推锐化：一种损失层面的修改方法，它在最小化未标注节点预测熵的同时进行平衡……

    arXiv:2605.20248v2 Announce Type: replace  Abstract: In the transductive setting, where the full graph is observed but node labels are only partially available, progress in semi-supervised node classification has largely focused on architectural innovation. In this paper, we revisit an orthogonal axis: the training objective. We start from a simple observation: transductive models produce predictions for every node during training, including nodes without labels. These unlabeled-node predictions may contain useful training signal, but standard supervised objectives discard them because no ground-truth labels are available. Inspired by the decomposition of cross-entropy into a label-dependent alignment term and a label-independent entropy term, we propose prediction confidence as a natural way to extract this signal in the absence of labels. This motivates Transductive Sharpening (TS): a loss-level modification that minimizes prediction entropy on unlabeled nodes while counterbalancing 
    
[^319]: 基于扩散模型的随机算子网络用于随机偏微分方程的不确定性量化

    Diffusion-Based Stochastic Operator Networks for Uncertainty Quantification in Stochastic Partial Differential Equations

    [https://arxiv.org/abs/2605.17107](https://arxiv.org/abs/2605.17107)

    该论文提出了随机算子网络（SON），通过将DeepONet与随机神经网络及扩散模型相结合，直接从含噪数据中学习，为随机偏微分方程的解同时提供均值预测和不确定性量化。

    

    我们引入了一个新颖的框架，用于对随机偏微分方程（SPDEs）相关的解算子进行不确定性量化。尽管随机偏微分方程在建模不确定性下的复杂物理系统中扮演着核心角色，但其实际应用通常需要指定模型不确定性的幅度和结构，而这些信息往往是未知的，且难以从含噪测量数据中推断出来。为了应对这一挑战，我们开发了一种随机算子学习框架，该框架能够直接从含噪数据中学习，并同时输出均值解场和不确定性量化结果。所提出的方法被称为随机算子网络，其结构是将深度算子网络与随机神经网络相结合，以建模随机性并实现概率预测。训练过程通过最小化哈密顿型损失函数并优化……

    arXiv:2605.17107v2 Announce Type: replace-cross  Abstract: We introduce a novel framework for uncertainty quantification of solution operators associated with stochastic partial differential equations (SPDEs). Although SPDEs play a central role in modeling complex physical systems under uncertainty, their practical use typically requires specifying the magnitude and structure of model uncertainties that are often unknown and difficult to infer from noisy measurements. To address this challenge, we develop a stochastic operator-learning framework that learns directly from noisy data and outputs both a mean solution field and a quantification of uncertainty. The proposed method, namely the Stochastic Operator Network (SON), is constructed by combining the structure of the Deep Operator Network (DeepONet) with Stochastic Neural Networks (SNNs) to model stochasticity and enable probabilistic prediction. The training procedure is carried out by minimizing a Hamiltonian-type loss and optimiz
    
[^320]: OrbiSim：作为具身智能可微物理引擎的世界模型

    OrbiSim: World Models as Differentiable Physics Engines for Embodied Intelligence

    [https://arxiv.org/abs/2605.16395](https://arxiv.org/abs/2605.16395)

    OrbiSim将世界模型重新定义为全可微的物理引擎，实现了从显式状态转移到视觉观测生成的端到端可微仿真，支持可微接触建模、稀疏奖励下的梯度策略优化等经典仿真器难以完成的任务，并在预测保真度和控制性能上显著超越现有世界模型。

    

    我们提出了OrbiSim，一种新颖的机器人仿真范式，它将世界模型重新定义为具身智能的全可微物理引擎。与以往专注于在潜在空间或视觉域中进行无约束想象的世界模型不同，OrbiSim建立了一条统一的、基于物理的通路，连接了结构化场景资产、神经动力学与下游强化学习。通过在整个仿真循环中实现端到端可微性——涵盖从显式状态转移到视觉观测生成的全过程——OrbiSim支持传统经典仿真器难以处理的任务，例如可微接触建模、稀疏奖励下基于梯度的策略优化以及直观的物理推断。实证结果表明，OrbiSim在预测保真度和控制性能方面均显著优于最先进的世界模型。此外，它的一致性……

    arXiv:2605.16395v2 Announce Type: replace-cross  Abstract: We present OrbiSim, a novel robotic simulation paradigm that redefines world models as a fully differentiable physics engine for embodied intelligence. Unlike prior world models that focus on unconstrained imagination in latent or visual domains, OrbiSim establishes a unified, physically-grounded pathway that bridges structured scene assets, neural dynamics, and downstream reinforcement learning. By enabling end-to-end differentiability throughout the entire simulation loop -- spanning from explicit state transitions to visual observation generation -- OrbiSim supports tasks traditionally intractable for classical simulators, such as differentiable contact modeling, gradient-based policy optimization under sparse rewards, and intuitive physical inference. Empirical results demonstrate that OrbiSim significantly outperforms state-of-the-art world models in both predictive fidelity and control performance. Furthermore, its consis
    
[^321]: GRAFT-ATHENA：用于自主发现与进化数值算法的自我改进智能体团队

    GRAFT-ATHENA: Self-Improving Agentic Teams for Autonomous Discovery and Evolutionary Numerical Algorithms

    [https://arxiv.org/abs/2605.11117](https://arxiv.org/abs/2605.11117)

    GRAFT-ATHENA框架将隐式的问题-方法关系显式化为可扩展的概率结构，使AI智能体团队的经验能够在结构相关的问题间迁移复用，在物理信息学习、血液流变学建模和高超声速流动求解等任务中达到或超越人类专家水平。

    

    科学方法是针对一类问题而开发的，因此知识可以在结构相似的案例之间迁移。语言模型智能体能够执行科学工作流程，但其问题-方法关系仍然是隐式的，导致每个新问题都需要重新开始搜索，之前有效的方法经验很少能够迁移。我们提出了GRAFT-ATHENA框架，它将这种问题到方法的映射显式化为一个可扩展的概率结构，包含可允许的问题、方法及其依赖关系。图分解技术使该结构保持可处理性，而语义指纹用于衡量相似度，从而使积累的经验能够指导相关问题的求解。结果表明，该框架达到或超越了专家基线：在物理信息学习中实现了接近机器精度的损失，再现了临床一致的血液流变学趋势，并为阿波罗指令舱开发了一个高阶高超声速流动求解器，其结果与实验测量值的误差在1.8%以内。

    arXiv:2605.11117v2 Announce Type: replace  Abstract: Scientific methods are developed for classes of problems, so knowledge transfers across structurally related cases. Language-model agents can execute scientific workflows, but their problem--method relationships remain implicit, so each new problem restarts the search and little of what worked transfers. We introduce GRAFT--ATHENA, which makes this problem-to-method map explicit as an expandable probabilistic structure of admissible problems, methods, and their dependencies. Graph factorization keeps the substrate tractable, and semantic fingerprints measure similarity, so experience guides related problems. As a result, the framework matched or exceeded expert baselines, attaining near-machine-precision losses in physics-informed learning, reproducing clinically consistent blood-rheology trends, and developing a high-order hypersonic-flow solver for the Apollo Command Module that matched experimental measurements within $1.8\%$. It 
    
[^322]: RubricEM：超越可验证奖励的量规引导策略分解元强化学习

    RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards

    [https://arxiv.org/abs/2605.10899](https://arxiv.org/abs/2605.10899)

    RubricEM提出一个统一的强化学习框架，通过量规引导的阶段结构化策略分解实现细粒度信用分配，并结合基于反思的元强化学习实现经验复用，解决了深度研究智能体在长时域任务且缺乏精确答案验证器时的训练难题。

    

    深度研究智能体需要进行规划、搜索、评估证据并合成长篇报告。使用强化学习（RL）训练这类智能体极具挑战性，因为它们需要在长时域上运行，且其输出缺乏精确的答案验证器。为此，我们提出了RubricEM，一个面向深度研究的统一强化学习框架，它将分阶段信用分配与基于反思的元强化学习（用于经验复用）相结合。量规通过为策略执行、裁判反馈和经验复用提供共享接口，统一了整个框架。RubricEM首先通过监督微调灌输量规引导的脚手架，教会智能体使用量规进行规划，并将其轨迹组织为语义阶段。在此结构先验的基础上，我们使用阶段结构化GRPO优化任务策略，利用分阶段的裁判反馈实现更细粒度的信用分配。与此同时，我们异步训练一个具有共享主干的反思元策略（wi……

    arXiv:2605.10899v2 Announce Type: replace  Abstract: Deep research agents plan, search, evaluate evidence, and synthesize long-form reports. Training these agents with reinforcement learning (RL) is challenging because they operate over long horizons and their outputs lack exact answer verifiers. To this end, we introduce RubricEM, a unified RL framework for deep research that combines stagewise credit assignment with reflection-based meta-RL for experience reuse. Rubrics unify the framework by providing a shared interface for policy execution, judge feedback, and experience reuse. RubricEM first instills a rubric-guided scaffold through supervised fine-tuning, teaching the agent to plan with rubrics and organize its trajectory into semantic stages. Building on this structural prior, we optimize the task policy with Stage-Structured GRPO, using stagewise judge feedback for finer-grained credit assignment. In parallel, we asynchronously train a shared-backbone reflection meta-policy (wi
    
[^323]: 对抗性噪声老虎机可学习性的完整刻画

    A Complete Characterization of Learnability for Adversarial Noisy Bandits

    [https://arxiv.org/abs/2605.09200](https://arxiv.org/abs/2605.09200)

    本文通过在函数类凸包上求值的广义极大极小体积这一凸化变体，首次完整刻画了对抗性噪声老虎机中函数类可学习的充要条件。

    

    我们研究在给定已知函数类 $\mathcal{F}$ 条件下的对抗性噪声老虎机问题。在每一轮中，对抗者从 $\mathcal{F}$ 中选择一个函数 $f$，学习者选择一个臂，随后观察到由所选臂和函数 $f$ 共同决定的含噪奖励。目标是最小化累积遗憾 $R(T)$，其定义为学习者在 $T$ 轮内的表现与事后最优固定臂表现之差。若存在能够实现次线性遗憾的算法，我们称函数类 $\mathcal{F}$ 是可学习的。我们的主要结果是对抗性噪声老虎机可学习性的完整刻画。该刻画基于 Hanneke 和 Wang（2025）提出的广义极大极小体积的一种凸化变体，即在函数类凸包 $\operatorname{co}(\mathcal F)$ 上求值的广义极大极小体积。我们证明，$\mathcal{F}$ 是可学习的当且仅当该凸化的……

    arXiv:2605.09200v3 Announce Type: replace  Abstract: We study adversarial noisy bandits given a known function class $\mathcal{F}$. In each round, the adversary selects a function $f \in \mathcal{F}$, the learner chooses an arm, and then observes a noisy reward determined by the chosen arm and the function $f$. The goal is to minimize the cumulative regret $R(T)$, defined as the difference between the learner's performance and that of the best fixed arm in hindsight over $T$ rounds. We say that a function class $\mathcal{F}$ is learnable if there exists an algorithm achieving sublinear regret. Our main result is a complete characterization of learnability for adversarial noisy bandits. The characterization is given in terms of a convexified variant of the generalized maximin volume introduced by Hanneke and Wang (2025): namely, the generalized maximin volume evaluated on the convex hull $\operatorname{co}(\mathcal F)$. We prove that $\mathcal F$ is learnable if and only if this convexi
    
[^324]: STRIDE：面向时间序列预测的推理感知训练

    STRIDE: Reasoning-Aware Training for Time Series Forecasting

    [https://arxiv.org/abs/2605.08625](https://arxiv.org/abs/2605.08625)

    STRIDE提出了一种新框架，将蒸馏后的LLM推理以连续跨模态先验的形式原生注入时间序列基础模型的嵌入空间，在保持数值预测能力的同时引入推理能力，并登上GIFT-Eval排行榜第一。

    

    时间序列基础模型在数值预测方面表现出色，但作为黑盒运行，缺乏定性推理能力。相反，直接将大语言模型（LLM）应用于时间数据会引入模态鸿沟：文本分词器会碎片化连续数值，破坏数学关系并导致序列长度爆炸，从而带来计算开销。为解决这一问题，我们提出了STRIDE（Strategic Time-series Reasoning Injected via Distilled Embeddings，通过蒸馏嵌入注入的战略性时间序列推理），这是一个将LLM推理原生集成到TSFM连续嵌入空间的新颖框架。STRIDE不使用离散token，而是将推理轨迹蒸馏到一个轻量级LLM中，并将其池化后的隐状态动态投影为连续的跨模态先验，注入数值编码器。通过交叉熵和分位数损失进行联合优化，STRIDE可以轻松适配多种架构（如Chronos-2、Timer-S1）。STRIDE在GIFT-Eval排行榜上排名第一。

    arXiv:2605.08625v2 Announce Type: replace-cross  Abstract: Time Series Foundation Models (TSFMs) excel at numerical forecasting but operate as black boxes lacking qualitative reasoning. Conversely, applying LLMs directly to temporal data introduces a modality gap: text tokenizers fragment continuous numerical values, degrading mathematical relationships and exploding sequence lengths, leading to computational overhead. To resolve this, we introduce STRIDE (Strategic Time-series Reasoning Injected via Distilled Embeddings), a novel framework natively integrating LLM reasoning into the continuous embedding space of TSFMs. Instead of discrete tokens, STRIDE distills reasoning traces into a lightweight LLM and dynamically projects its pooled hidden states as a continuous cross-modal prior into the numerical encoder. Jointly optimized via cross-entropy and quantile losses, STRIDE readily attaches to diverse architectures (e.g., Chronos-2, Timer-S1). STRIDE ranks #1 on the GIFT-Eval leaderbo
    
[^325]: Transformer均场表述中训练诱导的Token聚簇逃逸

    Training-Induced Escape from Token Clustering in a Mean-Field Formulation of Transformers

    [https://arxiv.org/abs/2605.07772](https://arxiv.org/abs/2605.07772)

    该论文首次在均场Transformer理论中揭示了训练对推理动力学的重塑作用：仅训练一个$L^2$正则化的线性FFN，即可使token分布在早期层聚簇之后于最终层附近逃逸出聚簇状态，且一次梯度更新就足以产生这种逃逸。

    

    Transformer通过在各层之间迭代地变换token表示来执行推理。这种逐层计算已被实证研究，而近期的Transformer动力学均场理论解释了注意力如何驱动token分布走向聚簇。然而，现有分析将模型参数视为预先给定的，留下了训练如何重塑这一聚簇图景的开放问题。我们在一个带噪声的均场Transformer中研究这一问题，其中仅在$L^2$正则化下训练一个参数线性的FFN。我们识别出一种由训练诱导的跨层推理动力学阶段：在前几层发生聚簇之后，token分布可以在接近最终层时离开聚簇区域。仅仅一次梯度下降更新就足以产生这种逃逸，且训练目标的最小值点满足一个指数型的turnpike（收费公路）界，在该界之下聚簇会在中间各层持续存在。

    arXiv:2605.07772v2 Announce Type: replace  Abstract: Transformers perform inference by iteratively transforming token representations across layers. This layerwise computation has been studied empirically, and recent mean-field theories of Transformer dynamics explain how attention can drive token distributions toward clustering. However, existing analyses treat model parameters as prescribed, leaving open how training reshapes this clustering picture. We study this question in a noisy mean-field Transformer in which only a parameter-linear FFN is trained under $L^2$ regularization. We identify a training-induced phase of the inference dynamics across layers: after clustering in the early layers, the token distribution can leave the clustered regime near the final layers. A single gradient-descent update already produces this escape, and minimizers of the training objective satisfy an exponential turnpike bound under which clustering persists through the interior layers. Our analysis i
    
[^326]: 协同至关重要：面向协作式多智能体强化学习的过程感知评估

    Coordination Matters: Process-Aware Evaluation of Cooperative Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2605.06557](https://arxiv.org/abs/2605.06557)

    本文提出了一个过程感知的多智能体强化学习评估框架，通过干扰、协同任务完成、参与度和协调效率等诊断指标补充传统聚合性能指标，揭示了回报相近的策略在协调行为上可能存在显著差异。

    

    协作式多智能体强化学习（MARL）基准测试通常强调回报、成功率或完成时间等聚合性结果。虽然这些指标至关重要，但它们可能掩盖智能体之间如何进行协调，并且无法区分通过不同交互过程达到相似结果的策略。我们提出了一个过程感知的评估框架，通过干扰、协同任务完成、参与度和协调效率等诊断指标来补充聚合性能的评估。我们在STAT——一个受控的空间任务分配测试平台——中评估了该框架，并检验了其在两个成熟的协作式MARL基准测试中更广泛的适用性。在多种基于价值和演员-评论家的方法中，我们发现回报几乎相同的策略在协调行为上可能存在显著差异。STAT中的受控规模扩展实验揭示了空间、任务和智能体规模扩展各自不同的过程特征。

    arXiv:2605.06557v2 Announce Type: replace-cross  Abstract: Cooperative multi-agent reinforcement learning (MARL) benchmarks commonly emphasize aggregate outcomes such as return, success rate, or completion time. While essential, these metrics can obscure how agents coordinate and fail to distinguish policies that achieve similar outcomes through different interaction processes. We propose a process-aware evaluation framework that complements aggregate performance with diagnostics of interference, coordinated task completion, participation, and coordination efficiency. We evaluate the framework in STAT, a controlled spatial task-allocation testbed, and examine its broader applicability in two established cooperative MARL benchmarks. Across diverse value-based and actor-critic methods, we find policies with nearly identical return can exhibit substantial differences in coordination behavior. Controlled scaling in STAT reveals distinct process signatures for spatial, task, and agent scali
    
[^327]: MOSAIC：面向科学时间序列的基于稀疏可加可辨识因果学习的模块发现

    MOSAIC: Module Discovery via Sparse Additive Identifiable Causal Learning for Scientific Time Series

    [https://arxiv.org/abs/2605.05524](https://arxiv.org/abs/2605.05524)

    MOSAIC 提出了一种稀疏时序 VAE 方法，将科学观测固有的语义可解释性传递到具有可辨识性保证的潜在空间中，从而在底层机制未知的科学时间序列中实现可解释的模块发现。

    

    因果表征学习（CRL）旨在恢复具有可辨识性保证的潜在变量，在适当的假设下，通常可辨识至排列和逐分量重参数化的等价程度。然而，可辨识性并不等于可解释性：潜在语义通常是在事后通过与已知的真实因子对齐来赋予的。这一局限性在科学时间序列中尤为突出，因为其底层机制未知，而发现可解释的结构本身就是一个主要目标。相比之下，科学观测（如残基对距离、气候指数或过程传感器数据）本质上是具有语义的，因为它们对应于命名的物理量。这引出了一个关键问题：能否将观测的可解释性传递到可辨识的潜在空间中？我们提出了 MOSAIC（通过稀疏可加可辨识因果学习进行模块发现），这是一种稀疏时序 VAE，它集成了……

    arXiv:2605.05524v2 Announce Type: replace-cross  Abstract: Causal representation learning (CRL) seeks to recover latent variables with identifiability guarantees, typically up to permutation and component-wise reparameterization under appropriate assumptions. However, identifiability does not imply interpretability: latent semantics are typically assigned post hoc by alignment with known ground-truth factors. This limitation is particularly acute in scientific time series, where underlying mechanisms are unknown and discovering interpretable structure is a primary goal. In contrast, scientific observations (such as residue-pair distances, climate indices, or process sensors) are inherently semantic, as they correspond to named physical quantities. This raises a key question: can the interpretability of observations be transferred to the identifiable latent space? We propose MOSAIC (Module discovery via Sparse Additive Identifiable Causal learning), a sparse temporal VAE that integrates
    
[^328]: 多上下文贝叶斯优化中的状态条件化评估

    Regime-Conditioned Evaluation in Multi-Context Bayesian Optimization

    [https://arxiv.org/abs/2605.04895](https://arxiv.org/abs/2605.04895)

    该论文揭示了迁移贝叶斯优化中采集函数的排名会随预算比例和先验质量等状态变量而逆转（98% 的已发表比较从未控制这一变量），并提出可用先验信息估计的可移植状态得分 PRS=(B/|A|)(1-ρ) 来预测这种排名转变，从而主张以状态条件化评估取代平均效应比较。

    

    已发表的迁移贝叶斯优化（transfer-BO）比较研究通常在隐藏的状态变量上估计采集函数选择的平均处理效应，而实际应用者需要的是针对其特定先验质量、预算比例和度量指标的条件效应。对来自 NeurIPS、ICML、ICLR、AISTATS、UAI、TMLR、JMLR 和 AutoML-Conf（2022-2025）的 40 篇迁移贝叶斯优化论文的审查发现，98% 的论文从未将 B/|A| 作为受控变量轴进行变化。在相同的 GDSC2 基准上，仅改变预算就会逆转排名：当 B=50 时，Greedy 在 Hit@1 上比 UCB 高 0.050；而当 B=100 时，UCB 比 Greedy 高 0.035。我们用可移植状态得分（Portable Regime Score）PRS=(B/|A|)(1-ρ) 来刻画这一转变，其中 ρ 是先验秩相关系数，可以在主比较之前从试点上下文中估计得到。在涵盖化学、药物反应生物学和超参数优化（HPO）的 79 个条件中，层次模型给出 β=0.50（p=1.1e-9），且 19% 的条件落入等效区，其中 |adv……（原文摘要至此截断）

    arXiv:2605.04895v2 Announce Type: replace  Abstract: Published transfer-BO comparisons often estimate an average treatment effect of acquisition choice over hidden regime variables, while practitioners need the conditional effect for their specific prior quality, budget ratio, and metric. An audit of 40 transfer-BO papers from NeurIPS, ICML, ICLR, AISTATS, UAI, TMLR, JMLR, and AutoML-Conf (2022-2025) finds that 98% never vary B/|A| as a controlled axis. On the same GDSC2 benchmark, changing only the budget reverses the ranking: at B=50, Greedy outperforms UCB by 0.050 Hit@1, while at B=100, UCB outperforms Greedy by 0.035. We capture this transition with the Portable Regime Score PRS=(B/|A|)(1-rho), where rho is the prior rank correlation and can be estimated from pilot contexts before the main comparison. Across 79 conditions spanning chemistry, drug-response biology, and HPO, a hierarchical model gives beta=0.50 (p=1.1e-9), and 19% of conditions fall in an equivalence zone where |adv
    
[^329]: 学习揭示了低秩RNN中的不可见结构

    Learning reveals invisible structure in low-rank RNNs

    [https://arxiv.org/abs/2605.04115](https://arxiv.org/abs/2605.04115)

    本文推导出低维常微分方程系统来精确描述低秩RNN的学习动力学，并发现学习过程可分为决定网络功能的“损失可见重叠”和不影响功能但描述学习所必需的“损失不可见重叠”。

    

    神经系统中的学习源于突触的改变，这些改变重塑了行为背后的表征。虽然低秩循环神经网络（RNN）已成为连接网络结构与功能的强大框架，但对其学习过程的理论理解仍然难以捉摸。在此，我们将低秩框架从神经活动扩展到学习过程，通过在简化的重叠空间中直接推导梯度下降动力学。我们构建了一个封闭形式的低维常微分方程（ODE）系统来支配该空间中的学习，该系统对于线性RNN是精确的，对于大-$N$高斯极限下的非线性RNN是渐近精确的。我们分析的核心在于对两类重叠的区分：损失可见重叠（loss-visible overlaps），它们完全决定网络的活动、输出和损失；以及损失不可见重叠（loss-invisible overlaps），它们不影响网络功能，但却是描述学习过程所必需的。我们阐明了这种分解所带来的后果。

    arXiv:2605.04115v2 Announce Type: replace-cross  Abstract: Learning in neural systems arises from synaptic changes that reshape the representations underlying behavior. While low-rank recurrent neural networks (RNNs) have emerged as a powerful framework for linking connectivity to function, a theoretical understanding of their learning process remains elusive. Here, we extend the low-rank framework from activity to learning by deriving gradient-descent dynamics directly in a reduced overlap space. We formulate a closed-form, low-dimensional system of ODEs that governs learning in this space, exact for linear RNNs and asymptotically exact for nonlinear RNNs in the large-$N$ Gaussian limit. Central to our analysis is a distinction between two classes of overlaps: loss-visible overlaps, which fully determine network activity, output, and loss, and loss-invisible overlaps, which do not affect function but are required to describe learning. We illustrate the consequences of this decompositi
    
[^330]: 使用核嵌入度量条件分布之间的差异

    Measuring Differences between Conditional Distributions using Kernel Embeddings

    [https://arxiv.org/abs/2605.02260](https://arxiv.org/abs/2605.02260)

    本文提出了条件最大均值差异（CMMD）这一统一理论框架，通过不同层级的核嵌入度量方法，系统性地解决了条件分布之间差异度量缺乏统一理论处理的问题。

    

    比较条件分布是统计学和机器学习中的一项基本挑战，在众多领域都有广泛应用。尽管基于再生核希尔伯特空间（RKHS）中分布核嵌入来度量差异的方法提供了强大的非参数技术，但现有文献仍然较为分散，缺乏统一的理论处理。本文通过建立一个连贯的框架来弥补这一空白，用于研究基于核的方法，通过我们称之为条件最大均值差异（CMMD）来度量条件分布之间的散度。CMMD 由一系列我们称之为“层级”的度量组成，其中包含三个特例，每个特例使用不同类型的 RKHS 嵌入：CMMD$_0$（条件均值算子）、CMMD$_1$（条件均值嵌入）和 CMMD$_2$（联合均值嵌入）。我们还进一步引入了一个一般层级……

    arXiv:2605.02260v2 Announce Type: replace-cross  Abstract: Comparing conditional distributions is a fundamental challenge in statistics and machine learning, with applications across a wide range of domains. While proposed methods for measuring discrepancies using kernel embeddings of distributions in a reproducing kernel Hilbert space (RKHS) provide powerful non-parametric techniques, the existing literature remains fragmented and lacks a unified theoretical treatment. This paper addresses this gap by establishing a coherent framework for studying kernel-based methods to measure divergence between conditional distributions through what we refer to as conditional maximum mean discrepancy (CMMD). The CMMD consists of a family of metrics which we call levels, with three special cases each using a different type of RKHS embedding: CMMD$_0$ (conditional mean operators), CMMD$_1$ (conditional mean embeddings), and CMMD$_2$ (joint mean embeddings). We additionally introduce a general level $
    
[^331]: RamanBench：一个面向拉曼光谱机器学习的大规模基准测试

    RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy

    [https://arxiv.org/abs/2605.02003](https://arxiv.org/abs/2605.02003)

    本文提出了RamanBench——首个大规模、完全可复现的拉曼光谱机器学习基准，统一整合了四个领域74个数据集共325,668条光谱，并在标准化协议下对28个模型进行了系统评估。

    

    机器学习（ML）已经变革了许多科学领域，然而一些关键应用仍然缺乏标准化的基准测试。拉曼光谱作为一种广泛使用的无创分子分析技术，正是这样一个领域——其进展受到数据集碎片化、评估标准不一致以及模型无法捕捉光谱数据结构等因素的限制。我们提出了RamanBench，这是首个大规模、完全可复现的拉曼光谱机器学习基准测试，包含简化的数据访问、评估协议与代码，以及一个实时排行榜。该基准统一整合了四个领域的74个数据集（其中16个为随本基准首次发布），共包含325,668条光谱，涵盖多种实验条件下的分类和回归任务。我们在标准化协议下对28个模型进行了基准测试，包括经典方法（如PLS）、拉曼光谱专用模型（如RamanNet）以及表格基础模型（TFM，如TabPFN）。

    arXiv:2605.02003v3 Announce Type: replace  Abstract: Machine Learning (ML) has transformed many scientific fields, yet key applications still lack standardized benchmarks. Raman spectroscopy, a widely used technique for non-invasive molecular analysis, is one such field where progress is limited by fragmented datasets, inconsistent evaluation, and models that fail to capture the structure of spectral data. We introduce RamanBench, the first large-scale, fully reproducible benchmark for ML on Raman spectroscopy, consisting of streamlined data access, evaluation protocols and code, as well as a live leaderboard. It unifies 74 datasets (including 16 first released with this benchmark) across four domains, comprising 325,668 spectra and spanning classification and regression tasks under diverse experimental conditions. We benchmark 28 models under a standardized protocol, including classical methods (e.g., PLS), Raman-specific (e.g., RamanNet), Tabular Foundation Model (TFM) (e.g., TabPFN)
    
[^332]: 基于状态历史与全局反馈的可解释经验学习

    Interpretable experiential learning based on state history and global feedback

    [https://arxiv.org/abs/2605.00940](https://arxiv.org/abs/2605.00940)

    提出了一种基于状态历史与全局反馈的可解释经验学习模型，通过带效用值和证据计数的状态转移图学习行为模型，适用于资源受限环境下的强化学习，并在Atari Breakout基准上取得了与神经网络方案相当的性能。

    

    本文提出了一种基于状态历史与全局反馈的新型可解释经验学习模型。该模型能够学习一种以状态集合之间的转移图表示的行为模型，其中每条转移都附有效用值和证据计数。该模型有望适用于解决资源受限环境下的强化学习问题。该模型在OpenAI Gym的Atari Breakout基准上进行了全面评估，其性能可与一些已知的基于神经网络的解决方案相媲美。

    arXiv:2605.00940v2 Announce Type: replace-cross  Abstract: A new interpretable experiential learning model based on state history and global feedback is presented. It is capable of learning a behavioral model represented by a transition graph between sets of states, with transitions attributed with utility and evidence count. This model is expected to be suitable for solving reinforcement learning problem in resource-constrained environments. The model was thoroughly evaluated on the OpenAI Gym Atari Breakout benchmark, demonstrating performance comparable to some known neural network-based solutions.
    
[^333]: 网格场理论：基于网格的物理学的端口哈密顿表述

    Mesh Field Theory: Port-Hamiltonian Formulation of Mesh-Based Physics

    [https://arxiv.org/abs/2605.00394](https://arxiv.org/abs/2605.00394)

    该论文提出网格场理论及其神经实现，证明了基于网格的物理动力学在最小物理原理下可局部分解为端口哈密顿形式——其中保守互连由网格拓扑唯一确定、度量效应仅通过本构关系与耗散进入——并据此设计出结构保持的物理神经网络。

    

    我们提出了网格场理论及其神经实现MeshFT-Net：一个面向基于网格的连续介质物理的结构保持框架，它清晰地将物理的拓扑结构与度量结构分离开来。通过施加最小物理原理（局部性、置换等变性、方向协变性以及能量平衡/耗散不等式），我们证明了基于网格物理的一个约化定理。在这些条件下，物理动力学允许局部分解为端口哈密顿形式：保守互连由网格拓扑唯一确定，而度量效应仅通过本构关系和耗散进入。这一约化澄清了哪些部分必须固定、哪些部分应当学习，从而直接指导了MeshFT-Net的设计。在解析数据集与真实数据集的评估、物理一致性测试以及分布外验证中，MeshFT-Net实现了接近于零的能量……（原文摘要在此处截断）

    arXiv:2605.00394v4 Announce Type: replace  Abstract: We present Mesh Field Theory (MeshFT) and its neural realization, MeshFT-Net: a structure-preserving framework for mesh-based continuum physics that cleanly separates the physics' topological structure from its metric structure. Imposing minimal physical principles (locality, permutation equivariance, orientation covariance, and energy balance/dissipation inequality), we prove a reduction theorem for mesh-based physics. Under these conditions, the physical dynamics admit a local factorization into a port-Hamiltonian form: the conservative interconnection is fixed uniquely by mesh topology, whereas metric effects enter only through constitutive relations and dissipation. This reduction clarifies what must be fixed and what should be learned, directly informing MeshFT-Net's design. Across evaluations on analytic and realistic datasets, physics-consistency tests, and out-of-distribution validation, MeshFT-Net achieves near-zero energy d
    
[^334]: CastFlow：在智能体时间序列预测中架起证据获取与准确预测之间的桥梁

    CastFlow: Bridging Evidence Acquisition and Accurate Prediction in Agentic Time Series Forecasting

    [https://arxiv.org/abs/2604.27840](https://arxiv.org/abs/2604.27840)

    CastFlow提出了一种角色专门化的智能体时间序列预测框架，通过经验记忆引导冻结的大语言模型调度多视角工具获取证据，再由轻量级专门化模型进行基于证据的时间推理，从而有效衔接证据获取与准确的数值预测。

    

    时间序列预测是时序数据分析中的一项基础任务，而基于大语言模型（LLM）的智能体通过自主工具交互和基于证据的推理提供了一种有前景的范式。然而，仍存在两个根本性挑战：如何通过自适应的工具交互获取与预测相关的证据，以及如何将此类证据转化为准确的数值预测。为应对这些挑战，我们提出了CastFlow，这是一个角色专门化的智能体预测框架，连接了通用工具推理与专门化的时间序列预测。具体而言，CastFlow引入了一个经验记忆模块，从训练数据中提炼成功的工具使用轨迹，指导冻结的LLM调度多视角工具包以完成证据获取。随后，一个轻量级的专门化LLM执行基于证据的时间推理，以细化集成预测先验。为增强这一能力……

    arXiv:2604.27840v3 Announce Type: replace-cross  Abstract: Time series forecasting is a fundamental task in temporal data analytics, and large language model (LLM)-based agents offer a promising paradigm through autonomous tool interaction and evidence-grounded reasoning. However, two fundamental challenges remain: how to acquire forecast-relevant evidence through adaptive tool interactions, and how to translate such evidence into accurate numerical predictions. To address these challenges, we propose CastFlow, a role-specialized agentic forecasting framework that bridges general-purpose tool reasoning and specialized temporal forecasting. Specifically, CastFlow introduces an Experience Memory that distills successful tool-use trajectories from training data, guiding a frozen LLM to orchestrate a multi-view toolkit for evidence acquisition. A lightweight specialized LLM then performs evidence-grounded temporal reasoning to refine an ensemble forecasting prior. To enhance this capabilit
    
[^335]: 利用细胞级空间转录组学作为细胞核分类的弱监督

    Leveraging Cell-level Spatial Transcriptomics as Weak Supervision for Nuclei Classification

    [https://arxiv.org/abs/2604.23481](https://arxiv.org/abs/2604.23481)

    该论文提出利用细胞级空间转录组学作为弱监督来源，自动生成图像-标签对以训练细胞核分类模型，在大幅减少人工标注的情况下达到了与传统监督方法相当的分割和分类性能。

    

    基于深度学习的组织病理学细胞核分析严重依赖于大规模、由专家精心整理的数据集，这些数据集成本高昂且难以跨器官扩展。为了解决这一问题，我们提出了一个可扩展的框架，利用细胞级空间转录组学（ST）作为细胞核分类的替代监督来源。细胞级空间转录组学同时提供基因表达谱和细胞核掩膜，能够在无需人工标注的情况下自动生成图像-标签对。我们将基因表达转化为细胞类型标签，并引入一种面向图像的分类策略，以有效地将这些细胞身份信息迁移到组织病理学图像上。在未见过的组织上进行评估，尽管使用的人工标注数据集更少，我们的框架相比传统监督方法仍取得了相当的分割和分类性能。

    arXiv:2604.23481v2 Announce Type: replace-cross  Abstract: Deep learning-based nuclei analysis in histopathology relies heavily on large-scale, expert-curated datasets that are costly and difficult to scale across organs. To address this, we propose a scalable framework leveraging cell-level spatial transcriptomics (ST) as an alternative supervision source for nuclei classification. Cell-level ST provides both gene expression profiles and nucleus masks, enabling automatic generation of image-label pairs without manual annotation. We convert gene expression into cell-type labels and introduce an image-oriented classification strategy to effectively transfer these identities to histopathology images. Evaluated on unseen tissues, our framework achieves comparable segmentation and classification performance compared with conventional supervised approaches, despite using fewer manually annotated datasets.
    
[^336]: 通过序列压缩低秩子空间实现激活高效的微调

    Activation-Efficient Fine-Tuning via Sequence-Compressed Low-Rank Subspaces

    [https://arxiv.org/abs/2604.22783](https://arxiv.org/abs/2604.22783)

    LARS通过在序列级低秩子空间中执行适配，将存储的适配器激活中间量复杂度从O(BSRL)降至O(BRL)，相比LoRA在GPU和CPU上分别节省23%和47.8%的内存，且节省幅度随上下文长度增加而提升。

    

    参数高效微调（PEFT）已成为适配大型语言模型（LLM）的标准方法。然而，尽管许多PEFT方法减少了可训练参数的数量，它们仍然受到随序列长度线性扩展的token级别激活中间量的内存限制，这限制了它们在长上下文和设备端适配场景中的实用性。与先前保留token级别适配器激活的记忆高效PEFT方法不同，LARS（低内存激活秩子空间）通过在序列级别的低秩子空间内执行适配，将存储的适配器激活中间量的序列长度扩展复杂度从O(BSRL)降低到O(BRL)。通过池化特征提取、低秩子空间调制和残差投影，LARS在S=1024时相比LoRA将总体内存占用在GPU上减少23%，在CPU上减少47.8%，且相对节省随上下文长度的增加而提升。实验在推理……（原文摘要在此处截断）

    arXiv:2604.22783v2 Announce Type: replace-cross  Abstract: Parameter-Efficient Fine-Tuning (PEFT) has become the standard approach for adapting large language models (LLMs). However, despite reducing trainable parameter counts, many PEFT methods remain memory-bound by token-level activation intermediates that scale linearly with sequence length, limiting their practicality for long-context and on-device adaptation. Unlike prior memory-efficient PEFT approaches that preserve token-level adapter activations, LARS (Low-memory Activation-Rank Subspace) reduces the sequence-length scaling of stored adapter activation intermediates from O(BSRL) to O(BRL) by performing adaptation within a sequence-level low-rank subspace. Through pooled feature extraction, low-rank subspace modulations, and residual projection, LARS reduces the overall memory footprint by 23% on GPUs and 47.8% on CPUs relative to LoRA at S = 1024, with relative savings increasing with context length. Experiments across reason
    
[^337]: 从用户识别到活动计数：一种身份无关的多用户WiFi感知方法

    From User Recognition to Activity Counting: An Identity-Agnostic Approach to Multi-User WiFi Sensing

    [https://arxiv.org/abs/2604.16572](https://arxiv.org/abs/2604.16572)

    该论文提出将多用户WiFi感知从“识别具体用户”重新表述为“身份无关的活动计数”，仅估计执行各活动的用户数量而无需识别个体，在WiMANS数据集上以更粗粒度的监督实现了更低的计数误差，并在跨环境条件和留出用户评估中展现出更好的泛化能力。

    

    WiFi信道状态信息（CSI）使得无设备人体活动识别成为可能，但现有的多用户方法通常假设一组固定的已知用户，在面对未见过的个体时性能会下降。我们将多用户识别重新表述为身份无关的活动计数，即估计有多少用户在执行每种活动，而无需将动作分配给特定个体。为了进行受控比较，我们将依赖身份的预测转换为活动计数，并使用仅输出头和目标函数不同的共享流水线，以相同的整数值指标进行评估。在WiMANS数据集上，尽管使用了更粗粒度的监督，身份无关计数所实现的整数计数平均绝对误差（MAE）仍低于由依赖身份预测导出的计数（在0-5计数范围内分别为0.0815和0.0909）。这一优势在所有八种环境-频段条件下均成立，而导出计数仅在精确匹配准确率方面领先。在留出用户评估中，依赖身份方法的宏F1显著下降

    arXiv:2604.16572v2 Announce Type: replace  Abstract: WiFi Channel State Information (CSI) enables device-free human activity recognition, but multi-user methods typically assume a fixed set of known users and degrade on unseen individuals. We reformulate multi-user recognition as identity-agnostic activity counting, estimating how many users perform each activity without assigning actions to individuals. For a controlled comparison, identity-dependent predictions are converted to activity counts and evaluated with the same integer-valued metrics using a shared pipeline that differs only in output head and objective. On WiMANS, identity-agnostic counting achieves lower integer-count MAE than derived identity-dependent counts (0.0815 versus 0.0909 on a 0-5 count scale) despite using coarser supervision. This ordering holds across all eight environment-band conditions, while derived counts lead only in exact-match accuracy. Under leave-users-out evaluation, identity-dependent macro-F1 dro
    
[^338]: 高斯分布扰动下KL散度的最坏情形最优 $\sqrt{\epsilon}$ 阶稳定性

    Worst-Case Optimal $\sqrt{\epsilon}$-Rate Stability of KL Divergence under Perturbations of Gaussian Distributions

    [https://arxiv.org/abs/2604.11026](https://arxiv.org/abs/2604.11026)

    本文在比较分布 $P$ 不必为高斯分布的情况下，首次建立了高斯参考分布受扰动时KL散度变化的最坏情形最优 $\sqrt{\epsilon}$ 阶稳定性界，突破了现有松弛三角不等式要求三个分布均为高斯分布的局限，可应用于流生成模型的分布外检测等场景。

    

    我们为库尔贝克-莱布勒（KL）散度在其高斯参考分布受到扰动时建立了一个稳定性界。现有的高斯分布之间KL散度的松弛三角不等式要求三个分布全部为高斯分布，这限制了它们在诸如基于流的生成模型的分布外检测等应用中的使用。设 $\mathcal{N}_1$ 和 $\mathcal{N}_2$ 为 $d$ 维高斯分布，$P$ 为任意满足 $\mathbb{E}_P[\|X\|^2]<\infty$ 且 $\mathrm{KL}(P\Vert \mathcal{N}_1)<\infty$ 的 $d$ 维分布。若当 $0<\epsilon<1/12$ 时有 $\mathrm{KL}(\mathcal{N}_1\Vert\mathcal{N}_2)\leq\epsilon$，我们给出了 $\left|\mathrm{KL}(P\Vert\mathcal{N}_1)-\mathrm{KL}(P\Vert\mathcal{N}_2)\right|$ 的上界。在适当的均匀有界条件下，当 $\epsilon\rightarrow 0$ 时，该界为 $\frac{\sqrt{6d\epsilon}}{2}+K\sqrt{\epsilon}+o(\sqrt{\epsilon})$。

    arXiv:2604.11026v4 Announce Type: replace-cross  Abstract: We establish a stability bound for the Kullback-Leibler (KL) divergence when its Gaussian reference distribution is perturbed. Existing relaxed triangle inequalities of KL divergence between Gaussian distributions require all three distributions to be Gaussian, which limits their use in applications such as out-of-distribution detection with flow-based generative models. Let $\mathcal{N}_1$ and $\mathcal{N}_2$ be $d$-dimensional Gaussian distributions, and let $P$ be any $d$-dimensional distribution satisfying $\mathbb{E}_P[\|X\|^2]<\infty$ and $\mathrm{KL}(P\Vert \mathcal{N}_1)<\infty$. If $\mathrm{KL}(\mathcal{N}_1\Vert\mathcal{N}_2)\leq\epsilon$ for $0<\epsilon<1/12$, we give an upper bound of $\left|\mathrm{KL}(P\Vert\mathcal{N}_1)-\mathrm{KL}(P\Vert\mathcal{N}_2)\right|$. Under suitable uniform boundedness conditions, the bound is $ \frac{\sqrt{6d\epsilon}}{2}+K\sqrt{\epsilon}+o(\sqrt{\epsilon})$ as $\epsilon\rightarrow 0$
    
[^339]: 通过多智能体可解释性检测多智能体串谋

    Detecting Multi-Agent Collusion Through Multi-Agent Interpretability

    [https://arxiv.org/abs/2604.01151](https://arxiv.org/abs/2604.01151)

    提出NARCBench基准与五种探针技术，首次利用模型内部表示在群体层面检测LLM多智能体系统中的串谋行为，并将其形式化为分布式异常检测问题。

    

    随着LLM智能体越来越多地部署在多智能体系统中，它们带来了可能规避标准人类监督形式的隐蔽协调风险。虽然对模型激活进行线性探针在单智能体场景中检测欺骗方面已显示出前景，但串谋本质上是一种多智能体现象，利用内部表示来检测智能体之间的串谋仍未被探索。我们提出了NARCBench，一个用于评估环境分布偏移下串谋检测的基准，并提出五种探针技术，通过聚合每个智能体的欺骗分数在群体层面进行场景分类，并在四个开放权重模型（Qwen3-32B、Llama-3.1-70B、DeepSeek-R1 32B、GPT-OSS-20B）和六种探针架构上进行了评估。我们将该问题构建为分布式异常检测问题，识别出三种映射到不同异常类型和检测范式的串谋特征。

    arXiv:2604.01151v3 Announce Type: replace  Abstract: As LLM agents are increasingly deployed in multi-agent systems, they introduce risks of covert coordination that may evade standard forms of human oversight. While linear probes on model activations have shown promise for detecting deception in single-agent settings, collusion is inherently a multi-agent phenomenon, and the use of internal representations for detecting collusion between agents remains unexplored. We introduce NARCBench, a benchmark for evaluating collusion detection under environment distribution shift, and propose five probing techniques that aggregate per-agent deception scores to classify scenarios at the group level, evaluated across four open-weight models (Qwen3-32B, Llama-3.1-70B, DeepSeek-R1 32B, GPT-OSS-20B) and six probe architectures. We frame this as a distributed anomaly detection problem, identifying three collusion signatures that map onto distinct anomaly types and detection paradigms. Every model rea
    
[^340]: SpecXMaster 技术报告

    SpecXMaster Technical Report

    [https://arxiv.org/abs/2603.23101](https://arxiv.org/abs/2603.23101)

    该论文提出SpecXMaster，一个基于智能体强化学习的端到端智能框架，可直接从原始FID数据自动提取NMR光谱多重性信息并解读为化学结构，在多个公开NMR解读基准上表现卓越。

    

    智能光谱学是AI驱动的闭环科学发现中的关键要素，充当物质结构与人工智能之间的重要桥梁。然而，传统的依赖专家的光谱解读面临诸多重大挑战，包括易受人为偏见和错误的影响、依赖有限的专业知识，以及不同解读人员之间的差异。为应对这些挑战，我们提出了SpecXMaster，一个利用智能体强化学习进行核磁共振（NMR）分子光谱解读的智能框架。SpecXMaster能够直接从原始FID（自由感应衰减）数据中自动提取1H和13C光谱的多重性信息。这一端到端流水线实现了将NMR光谱完全自动化地解读为化学结构。该框架在多个公开的NMR解读基准测试中展现出卓越的性能，并被进一步……

    arXiv:2603.23101v4 Announce Type: replace  Abstract: Intelligent spectroscopy serves as a pivotal element in AI-driven closed-loop scientific discovery, functioning as the critical bridge between matter structure and artificial intelligence. However, conventional expert-dependent spectral interpretation encounters substantial hurdles, including susceptibility to human bias and error, dependence on limited specialized expertise, and variability across interpreters. To address these challenges, we propose SpecXMaster, an intelligent framework leveraging Agentic Reinforcement Learning (RL) for NMR molecular spectral interpretation. SpecXMaster enables automated extraction of multiplicity information from both 1H and 13C spectra directly from raw FID (free induction decay) data. This end-to-end pipeline enables fully automated interpretation of NMR spectra into chemical structures. It demonstrates superior performance across multiple public NMR interpretation benchmarks and has been refine
    
[^341]: M$^2$RNN：基于矩阵值状态的非线性循环神经网络用于可扩展的语言建模

    M$^2$RNN: Non-Linear RNNs with Matrix-Valued States for Scalable Language Modeling

    [https://arxiv.org/abs/2603.14360](https://arxiv.org/abs/2603.14360)

    提出了一种具有矩阵值隐藏状态和非线性状态转移的新型循环神经网络M$^2$RNN，突破了Transformer在TC$^0$复杂度上的表达能力限制，实现了完美状态跟踪泛化并在大规模语言建模中表现优异。

    

    Transformer具有高度并行性，但其计算能力局限于TC$^0$复杂度类，无法完成诸如实体跟踪和代码执行等已被证明需要更强表达能力的任务。受此限制的启发，我们重新审视非线性循环神经网络（RNN）在语言建模中的应用，并提出了矩阵到矩阵RNN（M$^2$RNN）：一种具有矩阵值隐藏状态和富有表达力的非线性状态转移的架构。我们证明了非线性RNN的语言建模性能受其状态规模的限制，并展示了状态规模扩展机制如何实现对张量核心的高效利用。实验表明，M$^2$RNN在训练中未见过的序列长度上实现了完美的状态跟踪泛化。这些优势同样体现在大规模语言建模中。在将循环层与注意力机制交错组合的混合设置中，混合M$^2$RNN优于等效的门控Delta（Gated Delta）……

    arXiv:2603.14360v3 Announce Type: replace-cross  Abstract: Transformers are highly parallel but are limited to computations in the TC$^0$ complexity class, excluding tasks such as entity tracking and code execution that provably require greater expressive power. Motivated by this limitation, we revisit non-linear Recurrent Neural Networks (RNNs) for language modeling and introduce Matrix-to-Matrix RNN (M$^2$RNN): an architecture with matrix-valued hidden states and expressive non-linear state transitions. We demonstrate that the language modeling performance of non-linear RNNs is limited by their state size, and show how the state size expansion mechanism enables efficient use of tensor cores. Empirically, M$^2$RNN achieves perfect state tracking generalization at sequence lengths not seen during training. These benefits also translate to large-scale language modeling. In hybrid settings that interleave recurrent layers with attention, Hybrid M$^2$RNN outperforms equivalent Gated Delta
    
[^342]: 决策依赖不确定性下谱风险优化的自适应条件残差抽样

    Adaptive Conditional Residual Sampling for Spectral Risk Optimisation under Decision-Dependent Uncertainty

    [https://arxiv.org/abs/2603.12507](https://arxiv.org/abs/2603.12507)

    提出自适应条件残差抽样（ACRS）方法，通过复用标准化残差向量构建确定性代理模型，高效求解决策依赖不确定性下的谱风险优化问题，并提供基于Wasserstein误差的最优性保证。

    

    我们考虑最小化谱风险目标函数，即期望成本加上加权条件风险价值（CVaR），其中不确定参数的分布依赖于决策，且只能通过随机模拟器在固定的抽样预算下进行抽样。我们提出自适应条件残差抽样（ACRS）方法。该方法在空间填充设计的每个点上抽取一次模拟样本，拟合决策依赖分布的异方差回归模型，并将标准化残差向量作为情景复用，从而构建一个确定性代理模型。ACRS在该代理模型上搜索若干局部最优解，在最有希望的候选点周围增加抽样，利用共同随机数在模拟器上对候选方案进行排序，最后采用尾部重要性抽样进行样本路径的局部精细化。我们证明代理模型最优解的最优性间隙受情景分布的Wasserstein误差约束……

    arXiv:2603.12507v3 Announce Type: replace  Abstract: We consider minimising a spectral risk objective, the expected cost plus a weighted conditional value-at-risk, when the distribution of the uncertain parameters depends on the decision and can only be sampled through a stochastic simulator under a fixed budget of draws. We propose adaptive conditional residual sampling (ACRS). It takes one simulator draw at each point of a space-filling design, fits a heteroscedastic regression model of the decision-dependent distribution and reuses the standardised residual vectors as scenarios to construct a deterministic surrogate. ACRS searches this surrogate for several local optima, adds draws around the most promising ones, ranks the candidates on the simulator with common random numbers and finishes with sample-path local refinement using tail importance sampling. We show that the optimality gap of the surrogate minimiser is bounded by the Wasserstein error of the scenario distribution and th
    
[^343]: 结构因果瓶颈模型

    Structural Causal Bottleneck Models

    [https://arxiv.org/abs/2603.08682](https://arxiv.org/abs/2603.08682)

    该论文提出结构因果瓶颈模型（SCBMs），用低维“瓶颈”统计量传递高维变量间的因果效应，证明了瓶颈变量在双射意义下可从观测数据识别，并表明瓶颈表示能够恢复工具变量方法的可识别性、并支持中介分析与后门调整下的有效因果效应估计。

    

    我们提出了结构因果瓶颈模型（SCBMs），这是一类新颖的结构因果模型，其中高维变量之间的因果效应由低维的汇总统计量（即“瓶颈”）来传递。SCBMs为机制特定的、目标依赖的降维提供了一个灵活的框架，同时仍可通过标准学习算法进行估计。我们证明了瓶颈变量可以从观测数据中被识别（至双射），并通过实验验证了这一结论。随后，我们研究了线性SCBMs在三种标准调整策略下的因果效应估计问题。对于工具变量方法，我们发现了当所有变量均为高维时，朴素的二阶段最小二乘法在总体层面存在不可识别性，并证明瓶颈表示可以恢复可识别性。对于中介分析和后门调整，我们证明了对瓶颈变量进行条件化可以得到有效的效应估计。

    arXiv:2603.08682v3 Announce Type: replace-cross  Abstract: We introduce structural causal bottleneck models (SCBMs), a novel class of structural causal models in which causal effects between high-dimensional variables are mediated by low-dimensional summary statistics, or *bottlenecks*. SCBMs provide a flexible framework for mechanism-specific, target-dependent dimension reduction while remaining estimable via standard learning algorithms. We prove that bottleneck variables are identifiable up to bijection from observational data, and validate this experimentally. We then study causal effect estimation in linear SCBMs under three standard adjustment strategies. For instrumental variables, we identify a population-level non-identifiability of na\"ive two-stage least squares when all variables are high-dimensional, and show that bottleneck representations restore identification. For mediation and backdoor adjustment, we prove that conditioning on bottleneck variables yields valid effect 
    
[^344]: AOI：将失败轨迹转化为训练信号以实现自主云诊断

    AOI: Turning Failed Trajectories into Training Signals for Autonomous Cloud Diagnosis

    [https://arxiv.org/abs/2603.03378](https://arxiv.org/abs/2603.03378)

    该论文提出AOI可训练多智能体框架，通过GRPO将专家知识蒸馏到本地开源模型并结合读写分离的安全执行架构，把失败运维轨迹转化为训练信号，在安全约束下实现自主云诊断。

    

    大语言模型（LLM）智能体为自动化站点可靠性工程（SRE）提供了一种有前景的数据驱动方法，然而其企业级部署面临三大挑战：对专有数据的访问受限、在权限管控环境下执行操作的安全性问题，以及封闭系统无法从失败中学习改进。我们提出了AOI（自主运维智能，Autonomous Operations Intelligence），这是一个可训练的多智能体框架，将自动化运维形式化为安全约束下的结构化轨迹学习问题。我们的方法集成了三个关键组件：首先，一个可训练的诊断系统应用组相对策略优化（GRPO），将专家级知识蒸馏到本地部署的开源模型中，在不暴露敏感数据的前提下实现基于偏好的学习；其次，一个读写分离的执行架构将运维轨迹分解为观察、推理等阶段（摘要原文在此处截断）

    arXiv:2603.03378v4 Announce Type: replace-cross  Abstract: Large language model (LLM) agents offer a promising data-driven approach to automating Site Reliability Engineering (SRE), yet their enterprise deployment is constrained by three challenges: restricted access to proprietary data, unsafe action execution under permission-governed environments, and the inability of closed systems to improve from failures. We present AOI (Autonomous Operations Intelligence), a trainable multi-agent framework formulating automated operations as a structured trajectory learning problem under security constraints. Our approach integrates three key components. First, a trainable diagnostic system applies Group Relative Policy Optimization (GRPO) to distill expert-level knowledge into locally deployed open-source models, enabling preference-based learning without exposing sensitive data. Second, a read-write separated execution architecture decomposes operational trajectories into observation, reasonin
    
[^345]: 通过轨迹蒸馏与基于记忆的强化学习学习智能体式时间序列预测

    Learning Agentic Time Series Forecasting via Trajectory Distillation and Memory-Based Reinforcement Learning

    [https://arxiv.org/abs/2602.13802](https://arxiv.org/abs/2602.13802)

    本文提出 Cast-R1，一个通过教师模型轨迹蒸馏与带结构化记忆的多轮强化学习（GRPO）训练的 8B 参数智能体时间序列预测模型，证明较小的语言模型也能学会智能体式的工具交互与时序推理能力。

    

    arXiv:2602.13802v2 公告类型： 替换。摘要： 近期的智能体式时间序列预测（TSF）方法使用大型语言模型（LLM）自主编排分析与预测工具，并利用所获得的证据对时间模式进行推理。然而，许多此类方法依赖强大的大型语言模型在无需训练、由提示驱动的工作流中运行，这使得较小的模型能否学习这些交互与推理能力仍是一个悬而未决的问题。在本文中，我们提出了 Cast-R1，一个 8B 参数的智能体预测模型，它通过轨迹蒸馏和带有结构化记忆的多轮强化学习进行训练。首先，我们开发了一个配备专用分析与预测工具的预测智能体框架，使用教师 LLM 生成多轮交互轨迹，随后对这些轨迹进行评估并筛选用于监督微调。其次，我们在多轮预测回合中应用组相对策略优化（GRPO），并利用结构化记忆保存（原文摘要在此处截断）。

    arXiv:2602.13802v2 Announce Type: replace  Abstract: Recent agentic time series forecasting (TSF) approaches use large language models (LLMs) to autonomously orchestrate analytical and forecasting tools and reason about temporal patterns using the resulting evidence. However, many such approaches rely on powerful LLMs in training-free, prompt-driven workflows, leaving open whether smaller models can learn these interaction and reasoning capabilities. In this paper, we introduce Cast-R1, an 8Bparameter agentic forecasting model trained through trajectory distillation and multi-turn reinforcement learning with structured memory. First, we develop a forecasting agent harness with specialized analysis and prediction tools to generate multi-turn interaction trajectories using a teacher LLM, which are subsequently evaluated and selected for supervised fine-tuning. Second, we apply group relative policy optimization (GRPO) across multi-turn forecasting episodes, with structured memory preserv
    
[^346]: 测试驱动代码生成的理论分析

    A Theoretical Analysis of Test-Driven Code Generation

    [https://arxiv.org/abs/2602.06098](https://arxiv.org/abs/2602.06098)

    本文建立了测试驱动代码生成的概率理论框架，证明了基于模糊功能相似性的代码选择估计器在信噪比上严格优于基于功能等价性的估计器，并将回溯提示形式化为汤普森采样的上下文内近似，推导出新的遗憾界以解释其有效性受限于歧义性的原因。

    

    代码助手在测试驱动的软件开发中的应用日益广泛，但其环境交互策略背后的理论机制仍然缺乏深入研究。我们为两种主流范式提供了概率框架：利用执行环境在生成后进行代码选择，以及以环境反馈为条件的代码生成。首先，我们将若干成熟的选择启发式方法形式化为环境感知的代码正确性估计器。我们从理论上证明，基于模糊功能相似性的估计器引入了归纳偏置，并且在信噪比方面严格优于基于功能等价性的估计器。其次，我们将回溯提示（backprompting）框架化为汤普森采样的上下文内近似。我们为包含不可观测成分的奖励函数推导出了一个新的遗憾界，从理论上解释了为什么回溯提示的有效性受限于歧义性。

    arXiv:2602.06098v4 Announce Type: replace-cross  Abstract: Code assistants are increasingly utilized in test-driven software development, yet the theoretical mechanisms behind their environment-interaction strategies remain underexplored. We provide a probabilistic framework for two dominant paradigms: code selection after generation using the execution environment, and code generation conditioned on environment feedback. First, we formalize several well-established selection heuristics as environment-aware estimators of code correctness. We theoretically prove that estimators based on fuzzy functional similarity add an inductive bias and strictly dominate estimators based on functional equivalence in terms of signal-to-noise ratio. Second, we frame backprompting as an in-context approximation of Thompson sampling. We derive a novel regret bound for reward functions with unobservable components, theoretically explaining why the effectiveness of backprompting is limited by the ambiguity
    
[^347]: AutoPrunedRetriever：面向高效且可扩展检索增强生成的持久化推理图

    AutoPrunedRetriever: Persistent Reasoning Graphs for Efficient and Extensible Retrieval-Augmented Generation

    [https://arxiv.org/abs/2602.04926](https://arxiv.org/abs/2602.04926)

    提出了AutoPrunedRetriever（APR），一种基于图的RAG系统，通过共享符号码本将语料库事实与查询推理过程编码为既充当检索索引又充当持久化记忆的紧凑表示，使历史证据和推理结果可被高效复用，从而提升复杂推理任务下的效率与可扩展性。

    

    检索增强生成（RAG）已成为知识密集型大语言模型任务的标准方法。复杂推理仍然具有挑战性，因为查询往往需要通过多个相互依赖的推理步骤来组合分散的证据。然而，现有的RAG系统独立地处理每个查询，反复检索重叠的证据并重建已经完成的推理。若能保留这些推理过程，先前检索到的证据和中间结果就可以在后续查询中被复用。但随着历史记录的增长，推理时的复用成本会不断上升，这就需要一种紧凑的表示方式，在不显著损失质量的前提下保留有用信息。为应对这些挑战，我们提出了AutoPrunedRetriever（APR），这是一种基于图的RAG系统，它将语料库事实和查询衍生的推理编码到一个共享的符号码本（symbolic codebook）中。该码本既充当检索索引，又充当持久化记忆。

    arXiv:2602.04926v2 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) has become a standard approach for knowledge-intensive LLM tasks. Complex reasoning remains challenging because queries often require composing dispersed evidence through multiple dependent reasoning steps. Yet existing RAG systems process queries independently, repeatedly retrieving overlapping evidence and reconstructing established reasoning. Preserving such reasoning makes previously retrieved evidence and intermediate results reusable for subsequent queries. However, inference-time reuse becomes increasingly costly as the history grows, motivating a compact representation that preserves useful information without substantial loss in quality. To address these challenges, we present AutoPrunedRetriever (APR), a graph-based RAG system that encodes corpus facts and query-derived reasoning in a shared symbolic codebook. The codebook serves both as the retrieval index and as persistent memory
    
[^348]: 大型语言模型中的“亲AI”偏向

    Pro-AI Bias in Large Language Models

    [https://arxiv.org/abs/2601.13749](https://arxiv.org/abs/2601.13749)

    该研究发现大型语言模型存在“亲AI”偏向：它们在建议中更频繁且更靠前地推荐AI，会将AI相关职位的薪资高估约10%，并对AI专业证书授予数量的高估程度比匹配的计算机专业高出28%。

    

    大型语言模型（LLMs）在多个领域的决策支持中正得到越来越广泛的应用。我们探究的问题是：这些模型是否偏爱人工智能（AI）本身。我们通过三个实验来检验这一问题。首先，在投资、学习、职业和创业等领域的开放式建议中，AI在大多数推荐列表中都占据显著位置，并且一旦出现通常排名第一。其次，在薪资估算方面，模型对AI相关职位薪资的高估程度高于与之高度匹配的非AI职位。在一项受控的后续实验中，在所有其他职位信息保持不变的情况下，仅将职位名称从匹配的非AI名称改为AI相关名称，就会使预测薪资提高10%。最后，我们证明，AI专业所授予的学位及其他证书数量被高估的程度比与之匹配的计算机专业相应证书高出28%。总体而言，我们的研究表明，AI不仅在日常生活中……

    arXiv:2601.13749v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly employed for decision-support across multiple domains. We ask whether these models favor artificial intelligence (AI) itself. We examine this question through three experiments. First, in open-ended advice across investment, study, career, and startup domains, AI is prominent in most recommendation lists and is usually ranked first when it appears. Second, in salary estimations, models overestimate AI-related jobs more than they do for closely matched non-AI jobs. In a controlled follow-up, changing only the job title from the matched non-AI title to the AI-related title raises predicted salary by 10%, while all other supplied job details remain fixed. Finally, we show that the number of degrees and other credentials awarded by AI programs were 28% more overestimated than corresponding awards for matched computing programs. Overall, we show that AI is not only prominent in everyday 
    
[^349]: 使用平移对数链接函数的泊松非负矩阵分解新方法族

    A New Family of Poisson Non-negative Matrix Factorization Methods Using the Shifted Log Link

    [https://arxiv.org/abs/2601.05845](https://arxiv.org/abs/2601.05845)

    该论文提出了带平移对数链接函数的泊松非负矩阵分解新方法族，通过单一调节参数使“部分”的组合方式从加法平滑过渡到乘法，并提供了高效的最大似然算法及适用于大型稀疏数据集的快速近似方法。

    

    泊松非负矩阵分解（NMF）是一种广泛使用的方法，用于寻找计数数据的可解释的“基于部分”的分解。虽然泊松NMF存在许多变体，但现有方法都假设分解中的“部分”以加法方式组合。这一假设在某些情况下可能是自然的，但在其他情况下则不然。在此，我们引入带有平移对数（shifted-log）链接函数的泊松NMF来放宽这一假设。平移对数链接函数具有单一调节参数，随着该参数的变化，模型从假设各部分以加法方式组合（即标准泊松NMF）转变为假设各部分更多地以乘法方式组合。我们提供了通过最大似然估计来拟合该模型的算法，还提供了一种近似方法，可大幅减少大型稀疏数据集的计算时间（计算量与数据矩阵中非零条目的数量成正比）。我们在多种真实数据集上演示了这些新方法。

    arXiv:2601.05845v2 Announce Type: replace  Abstract: Poisson non-negative matrix factorization (NMF) is a widely used method to find interpretable "parts-based" decompositions of count data. While many variants of Poisson NMF exist, existing methods assume that the "parts" in the decomposition combine additively. This assumption may be natural in some settings, but not in others. Here we introduce Poisson NMF with the shifted-log link function to relax this assumption. The shifted-log link function has a single tuning parameter, and as this parameter varies the model changes from assuming that parts combine additively (i.e., standard Poisson NMF) to assuming that parts combine more multiplicatively. We provide an algorithm to fit this model by maximum likelihood, and also an approximation that substantially reduces computation time for large, sparse datasets (computations scale with the number of non-zero entries in the data matrix). We illustrate these new methods on a variety of real
    
[^350]: AceFF：面向小分子的最先进机器学习原子间势

    AceFF: A State-of-the-Art Machine Learning Potential for Small Molecules

    [https://arxiv.org/abs/2601.00581](https://arxiv.org/abs/2601.00581)

    AceFF是一个为小分子药物发现优化的预训练机器学习原子间势，通过改进TensorNet2架构并在类药物化合物综合数据集上训练，实现了高通量推理速度与DFT级精度的平衡，并支持带电状态和多种药物化学关键元素。

    

    我们介绍了AceFF，一个为小分子药物发现优化的预训练机器学习原子间势（MLIP）。尽管MLIP已成为密度泛函理论（DFT）的高效替代方案，但其在多样化化学空间中的泛化能力仍是一大难题。AceFF通过改进TensorNet2架构并在一个涵盖类药物化合物的综合数据集上进行训练来解决这一问题。这种方法使该力场在高通量推理速度与DFT级精度之间取得了平衡。我们推出了两个共享相同架构的模型：AceFF-2，在我们自己的小分子数据集上训练，支持药物化学的基本元素（H、B、C、N、O、F、Si、P、S、Cl、Br、I）；以及AceFF-2-OMOL，在OMol25的一个子集上训练，包含AceFF-2的元素以及常见的生物分子离子元素Na和K。这些模型经过专门训练以处理带电状态。验证……

    arXiv:2601.00581v3 Announce Type: replace-cross  Abstract: We introduce AceFF, a pre-trained machine learning interatomic potential (MLIP) optimized for small molecule drug discovery. While MLIPs have emerged as efficient alternatives to Density Functional Theory (DFT), generalizability across diverse chemical spaces remains difficult. AceFF addresses this by refining the TensorNet2 architecture and training it on a comprehensive dataset of drug-like compounds. This approach yields a force field that balances high-throughput inference speed with DFT-level accuracy. We introduce two models sharing the same architecture: \mbox{AceFF-2}, trained on our own dataset of small molecules and supporting the essential medicinal chemistry elements (H, B, C, N, O, F, Si, P, S, Cl, Br, I); and \mbox{AceFF-2-OMOL}, trained on a subset of OMol25 containing the AceFF-2 elements plus the common biomolecular ion elements Na and K. The models are explicitly trained to handle charged states. Validation ag
    
[^351]: CoatFusion：图像中可控的材质涂覆

    CoatFusion: Controllable Material Coating in Images

    [https://arxiv.org/abs/2512.02143](https://arxiv.org/abs/2512.02143)

    提出了一种新的“材质涂覆”图像编辑任务及CoatFusion架构，通过扩散模型结合2D反照率纹理和PBR风格参数控制，实现了保留物体几何细节的逼真且可控的材质涂覆效果。

    

    我们提出“材质涂覆”这一新颖的图像编辑任务，旨在模拟在物体表面施加一层薄薄的材质，同时保留其底层的粗粒度和细粒度几何结构。材质涂覆与现有的“材质迁移”方法有本质区别，后者旨在替换物体的固有材质，往往会覆盖细节。为解决这一新任务，我们构建了一个大规模合成数据集DataCoat110K（包含11万张图像），其中包含具有多种基于物理的涂层的3D物体。随后，我们提出了CoatFusion，这是一种新颖的架构，通过将扩散模型条件化于2D反照率纹理以及细粒度的PBR风格参数控制（包括粗糙度、金属度、透射率以及一个关键的厚度参数）来实现该任务。实验和用户研究表明，CoatFusion能够生成逼真且可控的涂覆效果，在各项指标上显著优于现有的材质编辑与迁移方法。

    arXiv:2512.02143v2 Announce Type: replace-cross  Abstract: We introduce Material Coating, a novel image editing task that simulates applying a thin material layer onto an object while preserving its underlying coarse and fine geometry. Material coating is fundamentally different from existing "material transfer" methods, which are designed to replace an object's intrinsic material, often overwriting fine details. To address this new task, we construct a large-scale synthetic dataset (110K images) of 3D objects with varied, physically-based coatings, named DataCoat110K. We then propose CoatFusion, a novel architecture that enables this task by conditioning a diffusion model on both a 2D albedo texture and granular, PBR-style parametric controls, including roughness, metalness, transmission, and a key thickness parameter. Experiments and user studies show CoatFusion produces realistic, controllable coatings and significantly outperforms existing material editing and transfer methods on t
    
[^352]: 使用单次概率前向传播与代码生成加速贝叶斯神经网络的执行

    Accelerated Execution of Bayesian Neural Networks using a Single Probabilistic Forward Pass and Code Generation

    [https://arxiv.org/abs/2511.23440](https://arxiv.org/abs/2511.23440)

    该论文通过假设权重和激活服从高斯分布，用单次解析式的概率前向传播取代多次采样前向传播来加速贝叶斯神经网络，并构建了在嵌入式ARM CPU上训练、编译、优化和部署的端到端流水线。

    

    机器学习模型在诊断、天气预报、自然语言处理和自动驾驶等领域表现良好，但其有限的不确定性处理能力限制了在安全关键场景中的应用。传统神经网络往往无法检测分布外（OOD）数据，并可能输出看似自信却错误的预测。贝叶斯神经网络（BNN）通过提供概率估计解决了这一问题，但由于预测需要对权重分布进行采样并进行多次前向传播，计算成本很高。概率前向传播（PFP）通过假设权重和激活服从高斯分布，为随机变分推断（SVI）提供了一种高效的近似方法，实现了完全解析的不确定性传播，并用单次确定性前向传播取代了采样。我们提出了一个端到端流水线，用于在嵌入式ARM CPU上训练、编译、优化和部署基于PFP的贝叶斯神经网络。

    arXiv:2511.23440v2 Announce Type: replace  Abstract: Machine learning models perform well across domains such as diagnostics, weather forecasting, NLP, and autonomous driving, but their limited uncertainty handling restricts use in safety-critical settings. Traditional neural networks often fail to detect out-of-domain (OOD) data and may output confident yet incorrect predictions. Bayesian neural networks (BNNs) address this by providing probabilistic estimates, but incur high computational cost because predictions require sampling weight distributions and multiple forward passes. The Probabilistic Forward Pass (PFP) offers a highly efficient approximation to Stochastic Variational Inference (SVI) by assuming Gaussian-distributed weights and activations, enabling fully analytic uncertainty propagation and replacing sampling with a single deterministic forward pass. We present an end-to-end pipeline for training, compiling, optimizing, and deploying PFP-based BNNs on embedded ARM CPUs. 
    
[^353]: Operon：通过命名维度增量构建不规则数据

    Operon: Incremental Construction of Ragged Data via Named Dimensions

    [https://arxiv.org/abs/2511.16080](https://arxiv.org/abs/2511.16080)

    Operon是一个基于Rust的工作流引擎，通过命名维度与显式依赖关系的形式化方法，结合静态验证的领域特定语言和增量式的动态任务调度，实现了对不规则数据形状与依赖关系的原生追踪与管理。

    

    现代数据处理工作流经常遇到不规则数据：即包含可变长度元素的集合，这类数据自然产生于自然语言处理、科学测量和自主AI智能体等领域。现有的工作流引擎缺乏对不规则数据固有的形状和依赖关系的原生支持，迫使用户手动管理复杂的索引和依赖记录。我们提出了Operon，一个基于Rust的工作流引擎，它通过一种新颖的命名维度形式化方法及显式依赖关系来解决这些挑战。Operon提供了一种领域特定语言，用户可以在其中声明带有维度标注的管道，这些标注会经过静态验证以确保正确性；同时运行时系统在执行过程中随着数据形状的逐步发现而动态调度任务。我们形式化了用于推理部分形状的数学基础，并证明了Operon的……（原文摘要在此处截断）

    arXiv:2511.16080v2 Announce Type: replace-cross  Abstract: Modern data processing workflows frequently encounter ragged data: collections with variable-length elements that arise naturally in domains like natural language processing, scientific measurements, and autonomous AI agents. Existing workflow engines lack native support for tracking the shapes and dependencies inherent to ragged data, forcing users to manage complex indexing and dependency bookkeeping manually. We present Operon, a Rust-based workflow engine that addresses these challenges through a novel formalism of named dimensions with explicit dependency relations. Operon provides a domain-specific language where users declare pipelines with dimension annotations that are statically verified for correctness, while the runtime system dynamically schedules tasks as data shapes are incrementally discovered during execution. We formalize the mathematical foundation for reasoning about partial shapes and prove that Operon's in
    
[^354]: 基于广义贝叶斯推断的鲁棒实验设计

    Robust Experimental Design via Generalised Bayesian Inference

    [https://arxiv.org/abs/2511.07671](https://arxiv.org/abs/2511.07671)

    本文提出广义贝叶斯最优实验设计（GBOED），通过用损失函数替代似然函数，在模型设定存在偏差时同时保证实验设计与统计推断的鲁棒性。

    

    贝叶斯最优实验设计是一个开展实验的原则性框架，它利用贝叶斯推断来量化选择某个设计所能期望获得的信息量。然而，准确的贝叶斯推断依赖于一个假设，即对数据生成过程所建立的统计模型是正确设定的。如果该假设不成立，贝叶斯方法可能导致糟糕的推断以及不准确的信息增益估计。广义贝叶斯（或称Gibbs）推断是一种更鲁棒的概率推断框架，它用合适的损失函数替代贝叶斯更新中的似然函数。在本工作中，我们提出了广义贝叶斯最优实验设计（GBOED），将广义贝叶斯推断扩展到实验设计场景中，从而在设计与推断两方面均实现鲁棒性。利用贝叶斯决策理论的一个推广形式，我们推导了一种新的采集（函数）……

    arXiv:2511.07671v2 Announce Type: replace-cross  Abstract: Bayesian optimal experimental design is a principled framework for conducting experiments that leverages Bayesian inference to quantify how much information one can expect to gain from selecting a certain design. However, accurate Bayesian inference relies on the assumption that one's statistical model of the data-generating process is correctly specified. If this assumption is violated, Bayesian methods can lead to poor inference and estimates of information gain. Generalised Bayesian (or Gibbs) inference is a more robust probabilistic inference framework that replaces the likelihood in the Bayesian update by a suitable loss function. In this work, we present Generalised Bayesian Optimal Experimental Design (GBOED), an extension of generalised Bayesian inference to the experimental design setting which achieves robustness in both design and inference. Using a generalisation of Bayesian decision theory, we derive a new acquisit
    
[^355]: Rep2Text：从单个LLM词元表示中解码完整文本

    Rep2Text: Decoding Full Text from a Single LLM Token Representation

    [https://arxiv.org/abs/2511.06571](https://arxiv.org/abs/2511.06571)

    提出Rep2Text方法，证明仅凭LLM的单个隐藏表示即可重建原始文本（短文本可恢复约一半单词及整体含义），揭示了隐藏表示蕴含的丰富信息及其潜在的隐私泄露风险。

    

    大型语言模型将文本表示为隐藏的数值模式，但这些表示实际包含什么信息仍然是一个谜。我们能否仅凭其中一个表示恢复出原始文本？在本文中，我们提出了Rep2Text，这是一种旨在通过从单个隐藏表示重建文本来回答这一问题的方法。我们发现这些表示保留的信息远比我们预期的要多。对于约16个单词的短文本，Rep2Text能够恢复大约一半的原始单词，同时保留其整体含义。随着文本变长，恢复确切的措辞变得更加困难，但主要思想仍然可以被恢复。这些发现为理解语言模型内部存储了什么提供了新的途径，并凸显了潜在的隐私风险，因为隐藏表示可能会泄露原始输入的部分内容。

    arXiv:2511.06571v4 Announce Type: replace-cross  Abstract: Large language models represent text as hidden numerical patterns, but what information these representations actually contain remains a mystery. Can we recover the original text from just one of them? In this paper, we introduce Rep2Text, a method designed to answer this question by reconstructing text from a single hidden representation. We find that these representations preserve much more information than we might expect. For short texts of about 16 words, Rep2Text can recover roughly half of the original words while preserving their overall meaning. As the text gets longer, recovering the exact wording becomes harder, but the main ideas can still be recovered. These findings offer a new way to understand what language models store internally and highlight potential privacy risks, as hidden representations may reveal parts of the original input.
    
[^356]: 经验贝叶斯多老虎机学习

    Empirical Bayesian Multi-Bandit Learning

    [https://arxiv.org/abs/2510.26284](https://arxiv.org/abs/2510.26284)

    本文提出了一种新颖的分层贝叶斯多老虎机学习框架，通过经验贝叶斯方法估计先验分布的协方差矩阵来捕捉老虎机实例间的相关性与异质性，并据此开发了ebmTS等高效算法。

    

    多任务情境老虎机学习因其在多个相关任务中利用共享结构和任务特定异质性来提升决策能力的潜力，引起了广泛的研究兴趣。在本文中，我们提出了一个新颖的分层贝叶斯框架，用于在各种老虎机实例中进行学习。该框架通过分层贝叶斯模型捕捉不同老虎机实例之间的异质性和相关性，在实现有效信息共享的同时兼顾实例特定的差异。与以往忽视跨老虎机协方差结构学习的方法不同，我们引入了一种经验贝叶斯方法来估计先验分布的协方差矩阵，从而提升了跨多老虎机学习的实用性与灵活性。基于该方法，我们开发了两种高效算法：ebmTS（经验贝叶斯多老虎机……）

    arXiv:2510.26284v3 Announce Type: replace  Abstract: Multi-task learning in contextual bandits has attracted significant research interest due to its potential to enhance decision-making across multiple related tasks by leveraging shared structures and task-specific heterogeneity. In this article, we propose a novel hierarchical Bayesian framework for learning in various bandit instances. This framework captures both the heterogeneity and the correlations among different bandit instances through a hierarchical Bayesian model, enabling effective information sharing while accommodating instance-specific variations. Unlike previous methods that overlook the learning of the covariance structure across bandits, we introduce an empirical Bayesian approach to estimate the covariance matrix of the prior distribution. This enhances both the practicality and flexibility of learning across multi-bandits. Building on this approach, we develop two efficient algorithms: ebmTS (Empirical Bayesian Mul
    
[^357]: 传感器节点中的能耗开销：嵌入式机器学习与通过低功耗广域网络的数据卸载

    Energy Overhead in Sensor Nodes: Embedded Machine Learning and Data Offloading via Low Power Wide Area Networks

    [https://arxiv.org/abs/2510.24829](https://arxiv.org/abs/2510.24829)

    本研究以ESP32-S3为平台，对传感器节点从数据采集、设备端CNN推理到LPWAN无线传输的完整流程进行了系统能耗分析，通过比较本地分类后传输结果与直接卸载原始数据两种方案，揭示了传感器节点能耗的主要驱动因素。

    

    基于图像和音频的远程物联网应用为应对生态变化提供了巨大的机遇。然而，要使其有效运作，这些应用必须在有限电力供应下长时间运行，同时通过低功耗广域网络（LPWAN）提供数据。以往的研究只考察了物联网系统中孤立组件的能效，例如无线电传输或数据处理方法。然而，要确定传感器节点能耗的主要驱动因素，需要进行整体性的评估。为解决这一问题，我们以ESP32-S3微控制器为平台，系统地分析了传感器节点从数据采集、可选的设备端CNN分类到无线网络传输的整个流程中的能耗。我们在三个环境监测场景中，比较了先在设备端进行CNN分类再传输结果与直接传输原始传感器数据这两种方案。在此基础上，我们改变了开发者可控的变量，特别是……

    arXiv:2510.24829v3 Announce Type: replace  Abstract: Image- and audio-based remote IoT applications offer a substantial opportunity to address ecological changes. However, to be effective, they must operate for long periods on limited power while also providing data via LPWAN. Previous studies have investigated the energy efficiency of isolated components such as radio transmission or data processing methods within an IoT system. However, a holistic evaluation is required to determine the main drivers of energy consumption in sensor nodes. To address this, we systematically analyze energy consumption across the entire sensor-node pipeline, from data acquisition to optional on-device CNN classification and wireless network transmission, using an ESP32-S3 microcontroller. We compare on-device CNN classification followed by result transmission with transmitting raw sensor data across three environmental monitoring scenarios. Thereby, we vary developer-controllable variables, notably the m
    
[^358]: 基于贝叶斯社区发现的可靠数据聚类

    Reliable data clustering with Bayesian community detection

    [https://arxiv.org/abs/2510.15013](https://arxiv.org/abs/2510.15013)

    该论文提出一种一步式聚类方法，利用贝叶斯社区检测和单一最小描述长度准则同时完成数据稀疏化与聚类，免除了任意阈值选择，在更高噪声和更少样本条件下优于传统两步法，且在无真实结构时能正确返回单一聚类。

    

    从神经科学、基因组学到系统生物学和生态学，研究人员依靠对相似性数据进行聚类来揭示模块化结构。大多数方法分两步工作：先降低噪声，再进行聚类。第一步的执行并不考虑第二步的需求：这迫使研究者做出任意的选择，丢弃有用的信息，甚至可能在不存在真实结构的地方强行施加结构。本研究将这两个步骤结合在一起。利用成熟的贝叶斯社区检测方法，我们让单一的最小描述长度准则同时决定如何稀疏化数据以及如何聚类。这种一步式方法保留了全部样本用于推断，消除了任意阈值的选择，并在不存在真实结构时返回单一聚类。在合成数据上，我们的方法能够在更多噪声和更少样本的条件下恢复聚类，优于标准方法。在人脑基因共表达数据上，它找到了比现有方法更连贯、更稳定的聚类。

    arXiv:2510.15013v2 Announce Type: replace-cross  Abstract: From neuroscience and genomics to systems biology and ecology, researchers rely on clustering similarity data to uncover modular structure. Most methods work in two steps: first reduce noise, then cluster. The first step runs without regard to the second: it forces arbitrary choices, discards useful information, and can impose structure where none exists. Here we bring the two steps together. Using established Bayesian community detection methods, we let one minimum description length criterion choose both how much to sparsify the data and how to cluster it. This one-step approach keeps all the samples for inference, removes the arbitrary threshold choice, and returns a single cluster when there is no real structure. On synthetic data, our approach recovers clusters under more noise and with fewer samples than standard methods. On gene co-expression data from the human brain, it finds more coherent and more stable clusters than
    
[^359]: 在潜空间中进行语言引导以缓解无意的语码切换

    Language steering in latent space to mitigate unintended code-switching

    [https://arxiv.org/abs/2510.13849](https://arxiv.org/abs/2510.13849)

    该论文提出一种通过对平行翻译做主成分分析来识别语言方向、并在潜空间中引导词元嵌入的轻量级推理时方法，可将多语言大模型的无意语码切换指数降低63%至99%，且几乎不增加计算开销、仅需极少平行数据校准。

    

    多语言大语言模型（LLM）经常表现出诸如无意语码切换之类的幻觉现象，降低了下游任务的可靠性。我们提出了潜空间语言引导方法，这是一种轻量级的推理时方法，通过对平行翻译进行主成分分析（PCA）来识别语言方向，并沿这些方向轴引导词元嵌入，从而控制语言身份。我们的方法缓解了语码切换问题（下一词元分布散度降低高达55%，语码切换指数降低63%至99%），同时保持了广泛的语义方向，计算开销几乎可以忽略不计，并且只需要极少的平行数据进行校准。此外，我们还使用单个主成分进行了语言分类评估（准确率达95-99%），并进一步分析了语言表示在各层中的演化过程。

    arXiv:2510.13849v4 Announce Type: replace  Abstract: Multilingual Large Language Models (LLMs) often exhibit hallucinations such as unintended code-switching, reducing reliability in downstream tasks. We propose latent-space language steering, a lightweight inference-time method that identifies language directions via Principal Component Analysis (PCA) on parallel translations and steers token embeddings along these axes to control language identity. Our approach mitigates code-switching (reduced next-token distributional divergence by up to 55\%, reduced Code-Switching Index by 63--99\%) while preserving broad semantic direction with negligible computational overhead and requires only minimal parallel data for calibration. Additionally, we evaluate language classification using a single principal component (95-99\% accuracy) and further analyze the layer-wise evolution of language representations.
    
[^360]: 董事会性别多元化与排放绩效：来自面板回归、机器学习和可解释人工智能的见解

    Board gender diversity and emissions performance: Insights from panel regressions, machine learning, and explainable AI

    [https://arxiv.org/abs/2510.00244](https://arxiv.org/abs/2510.00244)

    该研究结合面板回归、机器学习与可解释人工智能发现，欧洲企业董事会性别多元化与排放绩效呈非线性关系，约35%为最优比例、22%为必要阈值，且ESG争议不削弱该关系，说明其作用源于治理机制而非象征性行为。

    

    arXiv:2510.00244v3 公告类型：replace-cross 摘要：随着欧盟倡议强制要求公司董事会实行性别配额，一个关键问题随之而来：更高的董事会性别多元化（BGD）是否与更好的排放绩效（EP）相关？为回答这一问题，我们基于2016年至2022年欧洲企业样本，考察了董事会性别多元化对排放绩效的影响。通过面板回归、先进的机器学习算法和可解释人工智能，我们揭示了一种非线性关系。具体而言，排放绩效随着董事会性别多元化的提高而改善，直至达到约35%的最优水平，超过该水平后，董事会性别多元化的进一步提升不再带来排放绩效的额外改善。至少22%的董事会性别多元化阈值是实现排放绩效有意义改善的必要条件。为评估排放绩效结果的真实性，本研究还考察了ESG争议是否会削弱董事会性别多元化与排放绩效之间的关系。结果显示无显著影响，这表明董事会性别多元化的作用源于治理机制而非象征性行为。此外，路径分……（摘要原文在此处截断）

    arXiv:2510.00244v3 Announce Type: replace-cross  Abstract: With European Union initiatives mandating gender quotas on corporate boards, a key question arises: Is greater board gender diversity (BGD) associated with better emissions performance (EP)? To answer this question, we examine the influence of BGD on EP across a sample of European firms from 2016 to 2022. Using panel regressions, advanced machine learning algorithms, and explainable AI, we reveal a non-linear relationship. Specifically, EP improves with BGD up to an optimal level of approximately 35 %, beyond which further increases in BGD yield no additional improvement in EP. A minimum BGD threshold of 22 % is necessary for meaningful improvements in EP. To assess the legitimacy of EP outcomes, this study examines whether ESG controversies weaken the BGD-EP relationship. The results show no significant effect, suggesting that BGD's impact is driven by governance mechanisms rather than symbolic actions. Additionally, path anal
    
[^361]: 基于全同态加密的实用且隐私保护的混合机器学习推理

    Practical and Private Hybrid ML Inference with Fully Homomorphic Encryption

    [https://arxiv.org/abs/2509.01253](https://arxiv.org/abs/2509.01253)

    提出Safhire混合推理框架，通过在服务器端加密执行线性层、将非线性计算卸载至客户端，从而消除自举操作并显著降低计算成本，同时利用随机化洗牌技术保护模型机密性，实现实用且隐私安全的全同态加密机器学习推理。

    

    在当代基于云的服务中，保护用户的敏感数据并确保服务器模型的机密性至关重要。全同态加密（FHE）能够直接对加密输入进行推理，但昂贵的自举（bootstrapping）操作和非线性激活的低效近似阻碍了其实用性。我们提出了Safhire，一种混合推理框架，它在服务器端以加密方式执行线性层计算，同时将非线性部分以明文形式卸载到客户端执行。这一设计消除了自举操作，支持精确激活，并显著降低了计算成本。为了在客户端能够访问中间输出的情况下保护模型机密性，Safhire采用随机化洗牌技术对中间值进行混淆，使客户端实际上无法重建模型。为进一步降低推理延迟，Safhire还引入了先进的优化技术（如……）

    arXiv:2509.01253v2 Announce Type: replace-cross  Abstract: In contemporary cloud-based services, protecting users' sensitive data and ensuring the confidentiality of the server's model are critical. Fully homomorphic encryption (FHE) enables inference directly on encrypted inputs, but its practicality is hindered by expensive bootstrapping and inefficient approximations of non-linear activations. We introduce Safhire, a hybrid inference framework that executes linear layers under encryption on the server while offloading non- linearities to the client in plaintext. This design eliminates bootstrapping, supports exact activations, and significantly reduces computation cost. To safeguard model confidentiality despite client access to intermediate outputs, Safhire applies randomized shuffling, which obfuscates intermediate values and makes it practically impossible for a client to reconstruct the model. To further reduce inference latency, Safhire incorporates advanced optimizations such 
    
[^362]: 统计公平性-准确性前沿

    The Statistical Fairness-Accuracy Frontier

    [https://arxiv.org/abs/2508.17622](https://arxiv.org/abs/2508.17622)

    该论文在有限样本条件下研究了单一预测模型服务多个人口群体时的公平性-准确性权衡，推导了最坏情况最优估计量，刻画了有限样本对各群体福利的不对称影响，给出了最优样本分配策略，并提供了沿整个FA帕累托前沿的一致有限样本界。

    

    我们研究了当单一预测模型需要服务于多个人口统计群体时的公平性与准确性权衡问题。理解这一权衡的一个有用工具是公平性-准确性（FA）帕累托前沿，它刻画了这样一组模型：这些模型无法在不损害公平性或准确性的前提下改进其中任何一方。虽然刻画FA前沿需要对数据分布的完全知识，但我们聚焦于有限样本情形，量化设计者在有限数据下能够多好地逼近前沿上的任意一点，并对最坏情况下的差距给出上界。特别地，我们推导出了依赖于设计者对协变量分布了解程度的最坏情况最优估计量。对于每个估计量，我们刻画了有限样本效应如何不对等地影响各群体的福利，并确定了最优的样本分配策略。最后，我们提供了沿整个FA前沿的一致有限样本界，量化……（摘要在此处被截断）

    arXiv:2508.17622v4 Announce Type: replace-cross  Abstract: We study fairness-accuracy tradeoffs when a single predictive model must serve multiple demographic groups. A useful tool for understanding this tradeoff is the fairness-accuracy (FA) Pareto frontier, which characterizes the set of models that cannot be improved in either fairness or accuracy without worsening the other. While characterizing the FA frontier requires full knowledge of the data distribution, we focus on the finite-sample regime, quantifying how well a designer can approximate any point on the frontier from limited data and bounding the worst-case gap. In particular, we derive worst-case-optimal estimators that depend on the designer's knowledge of the covariate distribution. For each estimator, we characterize how finite-sample effects asymmetrically impact each group's welfare and identify optimal sample allocation strategies. Finally, we provide uniform finite-sample bounds along the entire FA frontier, quantif
    
[^363]: 使用对数秩与线性复杂度、可认证地同时求解无穷多个凸规划问题的神经算子

    Neural Operators Which Certifiably Simultaneously Solve Infinitely Many Convex Programs using Logarithmic Rank and Linear Complexity

    [https://arxiv.org/abs/2508.14995](https://arxiv.org/abs/2508.14995)

    本文提出一种简短而简单的训练算法来构造特定类别的神经算子，可认证地同时求解无穷多个凸规划问题，且仅需对数秩与线性复杂度，从而弥合了神经算子理论与实践之间的差距。

    

    神经算子（Neural Operators，NOs）是一类深度学习模型，旨在通过将相关问题投射到无穷维空间上并在其上进行操作，从而同时求解无穷多个相关问题。此类算子学习问题在数学金融中自然出现，例如在学习最优投资组合族或对冲策略族时。然而，理论与实践之间仍存在两个显著差距：1）大多数通用近似保证并未描述实际可实现的神经算子，或者只是纯粹的存在性陈述；2）通用近似定理给出的最坏情况参数界表明，所构造的神经算子可能需要不切实际的大量参数才能解决大多数算子学习问题，这与大量实验证据直接相悖。本文通过给出一个简短而简单的训练算法来构造一类特定的神经算子，弥合了这些差距……

    arXiv:2508.14995v2 Announce Type: replace  Abstract: Neural operators (NOs) are a class of deep learning models designed to simultaneously solve infinitely many related problems by casting them into an infinite-dimensional space, whereon these NOs operate. Such operator learning problems arise naturally in mathematical finance, for instance, when learning families of optimal portfolios or hedging strategies. Two significant gaps, however, remain between theory and practice: 1) most universal approximation guarantees do not describe practically implementable NOs or are pure existence statements; and 2) worst-case parameter bounds from universal approximation theorems suggest that constructed NOs may require an unrealistically large number of parameters to solve most operator learning problems, which stands in direct opposition to a slew of experimental evidence.   This paper closes these gaps by exhibiting a short and simple training algorithm to construct a specific class of NOs, proxi
    
[^364]: 具有信息有序策略的马尔可夫决策过程强化学习

    Reinforcement Learning in MDPs with Information-Ordered Policies

    [https://arxiv.org/abs/2508.03904](https://arxiv.org/abs/2508.03904)

    本文提出策略类上的“信息序”概念，通过反事实推断为连续状态动作空间的无限时域平均成本MDP设计了基于epoch的强化学习算法，取得了由信息序宽度 $w$ 决定的 $\widetilde O(H\sqrt{wdT/\alpha})$ 遗憾界，从而利用策略间数据的可迁移性降低了探索复杂度。

    

    许多运营性马尔可夫决策过程（MDP）提供了结构化信息，使得在某一策略下生成的轨迹还可能揭示其他策略本会有怎样的表现。我们通过参数化策略类上的信息序来形式化这一现象：当在 $\pi_\theta$ 下收集的数据可用于估计 $\pi_{\theta‘}$ 的长期平均成本时，记作 $\pi_{\theta’} \preceq \pi_\theta$。我们为具有连续状态和动作空间的无限时域平均成本MDP开发了一种基于epoch的强化学习算法，该算法通过反事实推断来利用这种序关系。对于一个 $d$ 维策略类，若其信息序的宽度为 $w$，则该算法相对于该类中的最优策略实现了 $\widetilde O(H\sqrt{wdT/\alpha})$ 的遗憾，其中 $H$ 为偏差跨度上界，$\alpha$ 表示“信息比率”。因此，信息序的宽度刻画了该问题的有效探索复杂度。

    arXiv:2508.03904v2 Announce Type: replace-cross  Abstract: Many operational Markov decision processes provide structured information where a trajectory generated under one policy may also reveal how other policies would have performed. We formalize this through an information order over a parameterized policy class, writing $\pi_{\theta'} \preceq \pi_\theta$ when data collected under $\pi_\theta$ can be used to estimate the long-run average cost of $\pi_{\theta'}$. We develop an epoch-based reinforcement learning algorithm for infinite-horizon average-cost MDPs with continuous state and action spaces that exploits this order through counterfactual inference. For a $d$-dimensional policy class whose information order has width $w$, the algorithm achieves $\widetilde O(H\sqrt{wdT/\alpha})$ regret relative to the optimal policy in the class, where $H$ is a bias span upper bound and $\alpha$ denotes an "information ratio". Thus, the width captures the effective exploration complexity of th
    
[^365]: RCPN：通过一致的原型与邻域实现无监督CLIP适配中的鲁棒伪标签

    RCPN: towards robust pseudo-labels in unsupervised CLIP adaptation via consistent prototypes and neighborhoods

    [https://arxiv.org/abs/2507.22075](https://arxiv.org/abs/2507.22075)

    提出自适应伪标签框架RCPN，通过原型一致性评估（PICS）、邻域语义修正（NALR）及自适应加权机制，为无监督CLIP适配生成更鲁棒的伪标签。

    

    在CLIP等视觉-语言模型的无监督适配中，基于零样本预测得到的伪标签往往存在显著噪声，尤其是在域偏移或视觉复杂场景下。传统依赖固定置信度阈值的伪标签过滤方法在完全无监督设置中往往不可靠。本文提出了一种新颖的自适应伪标签框架，通过融合原型一致性和基于邻域的一致性来提升CLIP的适配性能。所提方法包含两个关键组件：PICS，基于类内特征紧凑性和跨类特征分离度来评估伪标签的准确性；以及NALR，利用邻近样本间的语义相似性动态修正伪标签。此外，我们还引入了一种自适应加权机制，在训练过程中调整伪标签样本的影响力。

    arXiv:2507.22075v2 Announce Type: replace  Abstract: In unsupervised adaptation for vision-language models such as CLIP, pseudo-labels derived from zero-shot predictions often exhibit significant noise, particularly under domain shifts or in visually complex scenarios. Conventional pseudo-label filtering approaches, which rely on fixed confidence thresholds, tend to be unreliable in fully unsupervised settings. In this work, we propose a novel adaptive pseudo-labeling framework that enhances CLIP's adaptation performance by integrating prototype consistency and neighborhood-based consistency. The proposed method comprises two key components: PICS, which assesses pseudo-label accuracy based on in-class feature compactness and cross-class feature separation; and NALR, which exploits semantic similarities among neighboring samples to refine pseudo-labels dynamically. Additionally, we introduce an adaptive weighting mechanism that adjusts the influence of pseudo-labeled samples during trai
    
[^366]: 利用三维药效团的生成式分子进化实现高效的基于结构的药物设计

    Generative molecule evolution using 3D pharmacophore for efficient Structure-Based Drug Design

    [https://arxiv.org/abs/2507.20130](https://arxiv.org/abs/2507.20130)

    提出了MEVO框架，通过反应感知的生成模型在可合成化学空间中进行基于三维药效团的分子进化，同时解决了基于结构药物设计中蛋白质-配体复合物数据稀缺和生成分子难以合成的两大难题。

    

    生成式模型的最新突破为药物发现展现了巨大前景，然而基于结构的药物设计（SBDD）面临两个关键挑战：高质量蛋白质-配体复合物数据的稀缺性，以及所生成分子的实际可合成性。基于配体的方法虽然受益于十亿规模的小分子数据集，但往往缺乏靶蛋白的结构信息。为了解决数据稀缺和合成不可达的双重挑战，我们提出了MEVO——一个在可合成化学空间中运行的生成式分子进化框架。MEVO引入了一种反应感知的生成模型，该模型通过将预定义反应模板应用于Enamine砌块构建的反应路径进行训练。该模型联合生成分子以及符合这些模板的对应合成路线草图。通过将这种生成能力整合到口袋感知的分子进化过程中（摘要原文在此处被截断）。

    arXiv:2507.20130v2 Announce Type: replace  Abstract: Recent breakthroughs in generative models have shown great promise for drug discovery, yet Structure-Based Drug Design (SBDD) faces two critical challenges: the scarcity of high-quality protein-ligand complex data and the practical synthesizability of generated molecules. While ligand-based approaches benefit from billion-scale small molecule datasets, they often lack the structural context of the target protein. To address these dual challenges of data scarcity and synthetic inaccessibility, we propose MEVO, a generative molecular evolutionary framework that operates within a synthesizable chemical space. MEVO introduces a reaction-aware generative model trained on reaction routes constructed by applying predefined reaction templates to Enamine Building Blocks. The model jointly generates molecules and corresponding synthesis sketches consistent with these templates. By integrating this generative capability into a pocket-aware evol
    
[^367]: 视觉-语言-动作模型的置信度校准

    Confidence Calibration in Vision-Language-Action Models

    [https://arxiv.org/abs/2507.17383](https://arxiv.org/abs/2507.17383)

    本文首次系统研究了视觉-语言-动作（VLA）基础模型的置信度校准问题，建立了置信度估计基线，并提出提示集成与动作级Platt缩放两种轻量级技术来改善校准，从而通过可靠的不确定性量化提升机器人行为的可信赖性。

    

    可信赖的机器人行为不仅需要高水平的任务成功率，还需要机器人能够可靠地量化其成功的可能性。为此，我们首次对视觉-语言-动作（VLA）基础模型中的置信度校准进行了系统性研究，该类模型将视觉观察和自然语言指令映射到底层机器人运动指令。我们为VLA建立了置信度估计基线，考察了任务成功与校准误差之间的关系以及校准随时间的演变规律，并提出了两种轻量级技术来纠正我们所观察到的校准偏差：提示集成和动作级Platt缩放。本研究的目的是开始开发必要的工具和概念理解，以便通过可靠的不确定性量化使VLA模型变得可信赖。

    arXiv:2507.17383v3 Announce Type: replace-cross  Abstract: Trustworthy robot behavior requires not only high levels of task success but also that the robot can reliably quantify how likely it is to succeed. To this end, we present a first-of-its-kind study of confidence calibration in vision-language-action (VLA) foundation models, which map visual observations and natural language instructions to low-level robot motor commands. We establish a confidence estimation baseline for VLAs, examine how task success relates to calibration error and how calibration evolves over time, and introduce two lightweight techniques to remedy the miscalibration we observe: prompt ensembles and action-wise Platt scaling. Our aim in this study is to begin to develop the tools and conceptual understanding necessary to render VLAs trustworthy via reliable uncertainty quantification.
    
[^368]: 基于对抗性随机森林的缺失值插补方法——MissARF

    Missing value imputation with adversarial random forests - MissARF

    [https://arxiv.org/abs/2507.15681](https://arxiv.org/abs/2507.15681)

    MissARF是一种基于对抗性随机森林的快速缺失值插补方法，可在提供高质量单重与多重插补的同时，使多重插补无需额外计算成本。

    

    处理缺失值是生物统计分析中的一个常见挑战，通常通过插补方法来解决。我们提出了一种新颖、快速且易于使用的插补方法，称为基于对抗性随机森林的缺失值插补（MissARF），该方法基于生成式机器学习，可同时提供单重插补和多重插补。MissARF利用对抗性随机森林（ARF）进行密度估计和数据合成。为了插补某个观测值的缺失值，我们以该观测的非缺失值为条件，从ARF生成的估计条件分布中进行采样。我们的实验表明，MissARF在插补质量和运行速度方面与最先进的单重和多重插补方法表现相当，且多重插补无需额外的计算成本。

    arXiv:2507.15681v2 Announce Type: replace-cross  Abstract: Handling missing values is a common challenge in biostatistical analyses, typically addressed by imputation methods. We propose a novel, fast, and easy-to-use imputation method called missing value imputation with adversarial random forests (MissARF), based on generative machine learning, that provides both single and multiple imputation. MissARF employs adversarial random forest (ARF) for density estimation and data synthesis. To impute a missing value of an observation, we condition on the non-missing values and sample from the estimated conditional distribution generated by ARF. Our experiments demonstrate that MissARF performs comparably to state-of-the-art single and multiple imputation methods in terms of imputation quality and fast runtime with no additional costs for multiple imputation.
    
[^369]: 可解释强化学习综述：解释对象、方法与需求

    A Survey of Explainable Reinforcement Learning: Targets, Methods and Needs

    [https://arxiv.org/abs/2507.12599](https://arxiv.org/abs/2507.12599)

    本文提出了基于“解释什么”和“如何解释”两个维度的可解释强化学习（XRL）直观分类法，并对250多篇论文进行了系统性综述，同时指出了值得关注的相关领域与社区需求。

    

    近年来人工智能（AI）模型的成功伴随着其内部机制的不透明性，这主要归因于深度神经网络的使用。为了理解这些内部机制并解释这些AI模型的输出，研究者们提出了一系列方法，统称为可解释人工智能（XAI）领域。本文聚焦于XAI的一个子领域——可解释强化学习（XRL），其目标是对通过强化学习训练的智能体的行为进行解释。我们提出了一个基于“是什么”和“如何”两个问题的直观分类法：第一个问题关注方法所解释的对象，第二个问题涉及解释的提供方式。我们运用这一分类法对250多篇论文进行了最新进展的综述。此外，我们还介绍了一些与XRL密切相关的领域，我们认为这些领域应受到研究界的关注。最后，我们识别了……（原文摘要此处截断）

    arXiv:2507.12599v2 Announce Type: replace  Abstract: The success of recent Artificial Intelligence (AI) models has been accompanied by the opacity of their internal mechanisms, due notably to the use of deep neural networks. In order to understand these internal mechanisms and explain the output of these AI models, a set of methods have been proposed, grouped under the domain of eXplainable AI (XAI). This paper focuses on a sub-domain of XAI, called eXplainable Reinforcement Learning (XRL), which aims to explain the actions of an agent that has learned by reinforcement learning. We propose an intuitive taxonomy based on two questions "What" and "How". The first question focuses on the target that the method explains, while the second relates to the way the explanation is provided. We use this taxonomy to provide a state-of-the-art review of over 250 papers. In addition, we present a set of domains close to XRL, which we believe should get attention from the community. Finally, we ident
    
[^370]: NeuralDMD：从稀疏且含噪测量中获得可解释的动力学神经表示

    NeuralDMD: Interpretable Neural Representation of Dynamics from Sparse and Noisy Measurements

    [https://arxiv.org/abs/2507.03094](https://arxiv.org/abs/2507.03094)

    提出NeuralDMD框架，将神经隐式表示与动态模态分解（DMD）相结合，从稀疏含噪测量中重构出可解释的连续时空动力学，无需真值数据或模拟器，并在天气数据同化等多种任务上优于现有基线方法。

    

    科学成像中的许多挑战都涉及求解病态逆问题，其目标是从间接、含噪且高度稀疏的测量中恢复时空场——而且通常无法获得真值数据或可靠的模拟器。为应对这一困难场景，我们提出了NeuralDMD，这是一个可解释的、无需预先训练（针对每个实例）的重构框架，它将神经隐式表示与动态模态分解相结合，直接从测量数据中重构连续的时空动力学。NeuralDMD将DMD模态参数化为连续神经场，并通过谱时间演化施加低秩线性动力学先验，从而强制实现时间连续性。这一建模方式既能在稀疏条件下进行预测，又能产生可解释的模态和频谱。我们发现NeuralDMD在多种任务上都优于基线方法：包括从稀疏气象站数据进行的天气数据同化等。

    arXiv:2507.03094v3 Announce Type: replace-cross  Abstract: Many challenges in scientific imaging involve solving ill-posed inverse problems, where the goal is to recover spatio-temporal fields from indirect, noisy, and highly sparse measurements - often without access to ground truth data or reliable simulators. To address this challenging scenario, we present NeuralDMD, an interpretable, untrained (per-instance) reconstruction framework that combines neural implicit representations with Dynamic Mode Decomposition (DMD) to reconstruct continuous spatio-temporal dynamics directly from measurements. NeuralDMD parameterizes DMD modes as continuous neural fields, and imposes a low-rank linear dynamics prior with spectral time evolution to enforce temporal continuity. This formulation enables both forecasting under sparsity, and yields interpretable modes and spectra. We find that NeuralDMD outperforms baselines on a wide variety of tasks: from weather data assimilation from sparse station 
    
[^371]: 反向GPT：精确逆转语言模型的输出

    GPT, But Backwards: Exactly Inverting Language Model Outputs

    [https://arxiv.org/abs/2507.01693](https://arxiv.org/abs/2507.01693)

    该论文提出SODA算法，能够从语言模型的输出logits中精确反推出输入文本，实现多达17个token的零误报重建，且逆转难度主要取决于输入长度和词汇表大小而非参数量，并可应用于恢复后门触发器和个人身份信息等安全场景。

    

    最近的研究表明，不同的输入会在大语言模型（LLM）中产生不同的隐藏表示，因此原则上可以从这些表示中恢复出输入。然而，目前尚无任何方法能够仅凭模型的输出文本或logits可靠地实现这一点。我们提出了稀疏独热离散Adam（SODA），这是一种基于搜索的白盒算法，能够从输出的部分知识中重建文本输入。给定输出logits，SODA能够以零误报率恢复多达17个token的自然序列，并且与此前最强的方法相比，精确匹配重建性能提升了4倍。我们进一步证明，对于逆转难度而言，输入长度和词汇表大小比模型参数量更为关键，SODA成功逆转了从3300万到30亿参数规模的模型。最后，我们展示了概念验证性的安全应用：SODA在不到1000次迭代中从被投毒的大语言模型中恢复了6个后门触发器，并恢复了139个token的个人身份信息。

    arXiv:2507.01693v3 Announce Type: replace-cross  Abstract: Recent work shows that distinct inputs produce distinct hidden representations in LLMs, so the input can in principle be recovered from them. Yet, no existing method can reliably do so from the model's output text or logits alone. We introduce Sparse One-hot Discrete Adam (SODA), a white-box search-based algorithm that reconstructs textual inputs from partial knowledge of the output. Given output logits, SODA recovers natural sequences of up to 17 tokens with zero false positives and achieves a 4x improvement in exact-match reconstruction over the strongest prior method. We further show that input length and vocabulary size matter more for inversion difficulty than parameter count, with SODA inverting models from 33M to 3B parameters. Finally, we demonstrate proof-of-concept security applications: SODA recovers 6 backdoor triggers from a poisoned LLM in under 1000 iterations, and 139 tokens of personally identifiable informatio
    
[^372]: 深度随机ReLU神经网络的 $\ell^p$-Lipschitz 常数的近最优估计

    Near-optimal estimates for the $\ell^p$-Lipschitz constants of deep random ReLU neural networks

    [https://arxiv.org/abs/2506.19695](https://arxiv.org/abs/2506.19695)

    本文证明了遵循He初始化变体的随机ReLU神经网络的 $\ell^p$-Lipschitz 常数的近最优估计，揭示了该常数在 $p \in [2,\infty]$ 时类似于高斯向量的 $\ell^{p'}$ 范数，而在 $p \in [1,2)$ 时则更接近高斯向量的 $\ell^2$ 范数。

    

    本文研究了具有随机参数的ReLU神经网络 $\Phi: \mathbb{R}^d \to \mathbb{R}$ 在 $p \in [1,\infty]$ 范围内的 $\ell^p$-Lipschitz 常数。权重的分布遵循He初始化的一个变体。在零偏置网络的情况下，我们为宽网络推导出了高概率意义下的上下界，这两个界之间最多相差一个与网络深度呈对数关系的因子。值得注意的是，$\ell^p$-Lipschitz 常数的行为在 $p \in [1,2)$ 和 $p \in [2,\infty]$ 两个区间之间表现出显著差异。对于 $p \in [2,\infty]$，$\ell^p$-Lipschitz 常数的表现类似于 $\Vert g\Vert_{p'}$，其中 $g \in \mathbb{R}^d$ 是一个 $d$ 维标准高斯向量，且 $1/p + 1/p' = 1$。相反，对于 $p \in [1,2)$，$\ell^p$-Lipschitz 常数更接近于 $\Vert g \Vert_{2}$。我们将分析进一步扩展到偏置可能非零且从任意分布中抽取的网络。

    arXiv:2506.19695v2 Announce Type: replace-cross  Abstract: This paper studies the $\ell^p$-Lipschitz constants of ReLU neural networks $\Phi: \mathbb{R}^d \to \mathbb{R}$ with random parameters for $p \in [1,\infty]$. The distribution of the weights follows a variant of the He initialization. In the case of zero-bias networks, we derive high probability upper and lower bounds for wide networks that differ at most by a factor that is logarithmic in the network's depth. Remarkably, the behavior of the $\ell^p$-Lipschitz constant varies significantly between the regimes $ p \in [1,2) $ and $ p \in [2,\infty] $. For $p \in [2,\infty]$, the $\ell^p$-Lipschitz constant behaves similarly to $\Vert g\Vert_{p'}$, where $g \in \mathbb{R}^d$ is a $d$-dimensional standard Gaussian vector and $1/p + 1/p' = 1$. In contrast, for $p \in [1,2)$, the $\ell^p$-Lipschitz constant aligns more closely to $\Vert g \Vert_{2}$. We extend our analysis to networks with possibly non-zero biases drawn from arbitra
    
[^373]: RAW-Explainer：面向知识图谱补全的路径推理图神经网络事后解释方法

    RAW-Explainer: Post-hoc Explanations of Path Reasoning Graph Neural Networks for Knowledge Graph Completion

    [https://arxiv.org/abs/2506.12558](https://arxiv.org/abs/2506.12558)

    本文提出RAW-Explainer，一种专为路径推理图神经网络设计的新型可解释性方法，通过随机游走目标函数和异构信息，为知识图谱链接预测生成连通、简洁且可解释的子图解释，并解决了评估小子图时的分布偏移问题。

    

    路径推理图神经网络在链接预测等知识图谱补全任务上已展现出最先进的性能。然而，解释其预测结果仍然是一个具有挑战性的开放性问题，因为大多数GNN可解释性方法针对的是节点级或图级任务。在本文中，我们提出了RAW-Explainer（随机游走解释器），这是一种专为路径推理GNN量身定制的新型可解释性方法，能够为知识图谱中的链接预测生成连通、简洁且可解释的子图解释。我们的方法利用异构信息来识别事实性解释，并通过随机游走目标函数来增强连通性，同时使用参数化模型实现高效的集体解释。此外，RAW-Explainer通过提出一种稳健的评估策略，克服了在评估比完整图小数个数量级的解释子图时出现的分布偏移问题。

    arXiv:2506.12558v2 Announce Type: replace  Abstract: Path reasoning graph neural networks have demonstrated state-of-the-art performance on knowledge graph completion tasks such as link prediction. However, interpreting their predictions remains a challenging open problem, as most GNN explainability methods target node-level or graph-level tasks. In this paper, we propose RAW-Explainer ($\underline{Ra}$ndom $\underline{W}$alk Explainer), a novel explainability method tailored to path reasoning GNNs that generates connected, concise, and interpretable subgraph explanations for link prediction in knowledge graphs. Our method uses heterogeneous information to identify factual explanations and encourages connectivity through a random walk objective, using a parameterized model for efficient collective explanation. Furthermore, RAW-Explainer overcomes the distribution shift issue when evaluating explanatory subgraphs that are orders of magnitude smaller than the full graph by proposing a ro
    
[^374]: GARG-AML对抗分拆洗钱：一个可扩展且可解释的基于图的反洗钱框架

    GARG-AML against Smurfing: A Scalable and Interpretable Graph-Based Framework for Anti-Money Laundering

    [https://arxiv.org/abs/2506.04292](https://arxiv.org/abs/2506.04292)

    本文提出GARG-AML框架，通过从二阶交易网络结构中提取的单一可解释指标量化分拆洗钱风险，在保持可扩展性的同时兼顾了可解释性，克服了黑盒网络嵌入方法在实际银行业务中的局限。

    

    洗钱是一个全球性问题，据估计占全球GDP的2%-5%，这促使监管机构对金融机构实施严格规定以打击洗钱行为。一种典型的洗钱手段被称为“分拆洗钱”，即通过将交易拆分成较小金额来逃避检测。鉴于洗钱方案的复杂性——涉及多方之间的大量交易——网络分析成为反洗钱的重要工具。然而，近年来网络分析领域的进展严重依赖黑盒网络嵌入方法，限制了其在实际业务中的应用。为应对分拆洗钱的特殊性质以及银行的运营要求，本文提出了GARG-AML，这是一种新颖的基于图的方法，通过从二阶交易网络结构中导出的单一可解释指标来量化分拆洗钱风险。与传统方法不同，GARG-AML在性能与可解释性之间取得了平衡

    arXiv:2506.04292v4 Announce Type: replace-cross  Abstract: Money laundering, a global problem estimated to account for 2%-5% of the global GDP, required regulators to enforce stringent rules on financial institutions to combat it. One prominent laundering method, called smurfing, relies on breaking up transactions into smaller amounts to evade detection. Given the complexity of laundering schemes, involving numerous transactions across multiple parties, network analytics is an important tool for anti-money laundering. However, recent advances in network analytics have leaned heavily on black-box network embedding methods, limiting their utility in businesses. To address the specific nature of smurfing and banks' operational requirements, this paper introduces GARG-AML, a novel graph-based method that quantifies smurfing risk through a single interpretable metric derived from the structure of the second-order transaction network. Unlike traditional methods, GARG-AML provides a balance b
    
[^375]: GRIT：基于有限且含噪的不变样本对实现分布外鲁棒性

    GRIT: Out-of-Distribution Robustness from Finite and Noisy Invariant Pairs

    [https://arxiv.org/abs/2505.24843](https://arxiv.org/abs/2505.24843)

    提出GRIT方法，通过在预训练特征中从有限且含噪的不变样本对差异协方差估计不稳定子空间并予以投影去除，从而在样本对不完美的情况下仍能提升模型的分布外鲁棒性。

    

    预测器常常利用虚假相关性，这些相关性在分布偏移下会失效，从而损害分布外（OOD）鲁棒性。不变样本对，例如同一物体在不同背景下拍摄的照片，能直接指示哪些变化不应影响预测。然而现实中收集的样本对是有限且含噪的，因此可能遗漏相关变化，或导致学习器丢弃有预测价值的信息。为了从这样的样本对集合中学习，我们提出了几何鲁棒不变训练：它在固定的预训练特征中，从样本对差异的协方差估计出一个不稳定子空间，并在拟合预测器之前将该子空间投影去除。由于样本对的处理与预测器的拟合是相互分离的，同一投影可以配合经验风险最小化（ERM）或其他鲁棒性目标使用。与二次预测一致性不同——后者会惩罚沿每一个具有非零样本对协方差方向的变化，包括引入的变化……（摘要原文在此处截断）

    arXiv:2505.24843v3 Announce Type: replace  Abstract: Predictors often exploit spurious correlations that fail under distribution shift, undermining out-of-distribution (OOD) robustness. Invariant pairs, such as the same object photographed against different backgrounds, directly indicate which variation should not affect prediction. Yet real pair collections are finite and noisy, so they can miss relevant variation or lead a learner to discard predictive information. To learn from such collections, we propose Geometric Robustness Invariant Training (GRIT), which estimates an unstable subspace from the covariance of pair differences in fixed pretrained features and projects it out before fitting a predictor. Because pair processing is separate from predictor fitting, the same projection can accompany ERM or other robustness objectives. Unlike quadratic prediction consistency, which penalizes variation along every direction with nonzero pair covariance, including variation introduced by 
    
[^376]: 基于掩码的深度符号回归

    Mask-Based Deep Symbolic Regression

    [https://arxiv.org/abs/2505.24776](https://arxiv.org/abs/2505.24776)

    提出了一种基于随机掩码扩散过程生成多样高质量方程、并结合逐token的GRPO强化学习与长短期风险寻求策略的深度符号回归新方法。

    

    扩散模型已成为一种强大的生成建模框架，在图像和音频合成等应用中取得了显著成功。受此进展启发，我们提出了一种新颖的基于扩散的符号回归方法。我们构建了基于随机掩码的扩散与去噪过程，以生成多样且高质量的方程。我们将该生成过程与逐token的组相对策略优化（GRPO）方法相结合，从而在给定的测量数据集上进行高效的强化学习。此外，我们引入了一种长短期风险寻求策略，以扩大表现优异的候选解池，进一步提升性能。大量的实验和消融研究证明了我们方法的有效性。

    arXiv:2505.24776v2 Announce Type: replace  Abstract: Diffusion has emerged as a powerful framework for generative modeling, achieving remarkable success in applications such as image and audio synthesis. Enlightened by this progress, we propose a novel diffusion-based approach for symbolic regression. We construct a random mask-based diffusion and denoising process to generate diverse and high-quality equations. We integrate this generative processes with a token-wise Group Relative Policy Optimization (GRPO) method to conduct efficient reinforcement learning on the given measurement dataset. In addition, we introduce a long short-term risk-seeking policy to expand the pool of top-performing candidates, further enhancing performance. Extensive experiments and ablation studies have demonstrated the effectiveness of our approach.
    
[^377]: TRIP-Shapley：面向点对点联邦学习的传播感知贡献归因框架

    TRIP-Shapley: Propagation-Aware Contribution Attribution Framework for Peer-to-Peer Federated Learning

    [https://arxiv.org/abs/2505.23246](https://arxiv.org/abs/2505.23246)

    TRIP-Shapley提出了一种面向点对点联邦学习的传播感知贡献归因框架，通过独立计算每轮贡献分数并仅传播标量贡献值，避免了多跳路径追踪带来的通信爆炸，并能抵御虚假报告攻击。

    

    贡献归因是联邦学习（FL）中的一个基本问题，但在点对点联邦学习（P2PFL）中依然充满挑战，因为参与者的影响力会通过重复的模型交换而传播。最近的研究DICE通过直接追踪多轮传播路径来解决这一问题。然而，其实际部署受到两大挑战的阻碍：（i）多跳传播路径追踪导致的通信爆炸，（ii）对虚假报告攻击的脆弱性。为了克服这些挑战，我们考虑在点对点联邦学习中采用仅用于审计的无模型协调器来进行贡献归因。基于这一设定，我们提出了TRIP-Shapley，一种传播感知的、受Shapley启发的贡献归因替代方法。TRIP-Shapley不显式追踪多轮路径，而是独立计算每一轮的贡献分数，并仅传播标量形式的贡献值。这使得影响力归因能够……（摘要在此处截断）

    arXiv:2505.23246v3 Announce Type: replace  Abstract: Contribution attribution is a fundamental problem in federated learning (FL), yet it remains challenging in peer-to-peer FL (P2PFL), where participant influence propagates through repeated model exchanges. Recent work, DICE, addresses this issue by directly tracking multi-round propagation paths. However, its practical deployment is hindered by two challenges: (i) communication explosion caused by multi-hop propagation path tracking, and (ii) vulnerability to misreporting attacks. To overcome these challenges, we consider contribution attribution in P2PFL with a model-free coordinator used solely for auditing. Based on this setting, we propose TRIP-Shapley, a propagation-aware Shapley-inspired surrogate for contribution attribution. Instead of explicitly tracking multi-round paths, TRIP-Shapley computes round-wise contribution scores independently and propagates only scalar contribution values. This enables attribution of influence a
    
[^378]: 两心智胜过一心：面向大语言模型对齐的协作式奖励建模

    Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment

    [https://arxiv.org/abs/2505.10597](https://arxiv.org/abs/2505.10597)

    提出协作式奖励建模（CRM）框架，通过两个奖励模型跨模型互评结合课程学习来识别可靠的偏好数据，从而有效缓解噪声偏好导致的奖励误泛化并提升大语言模型的对齐效果。

    

    奖励模型（RMs）对于将大语言模型（LLMs）与人类价值观对齐至关重要，然而噪声偏好反馈可能通过虚假关联和过拟合引发奖励的误泛化。我们刻画了偏好对的训练动态，并揭示了噪声的独特优化特征：持续的高损失、剧烈的时间波动以及不规则的梯度更新。我们的分析进一步表明，排除此类样本对可以改善下游对齐效果。然而在实践中，可靠的偏好无法先验得知，而基于模型自身训练信号的自过滤可能会加剧确认偏误。为应对这一挑战，我们提出了协作式奖励建模，这是一种用于识别可靠偏好的在线框架。CRM将跨模型互评与课程学习相结合：两个奖励模型相互评估对方的样本选择，同时课程学习保持它们的学习进度。

    arXiv:2505.10597v3 Announce Type: replace-cross  Abstract: Reward models (RMs) are central to aligning large language models (LLMs) with human values, yet noisy preference feedback can induce reward misgeneralization through spurious correlations and overfitting. We characterize the training dynamics of preference pairs and uncover a distinctive optimization signature of noise: persistently high losses, sharp temporal fluctuations, and irregular gradient updates. Our analyses further show that excluding such pairs improves downstream alignment. In practice, however, reliable preferences are not known a priori, and self-filtering based on a model's own training signals can amplify confirmation bias. To address this challenge, we introduce Collaborative Reward Modeling (CRM), an online framework for identifying reliable preferences. CRM couples cross-model peer review with curriculum learning: two RMs evaluate each other's sample selections, while curriculum keeps their learning progress
    
[^379]: 潜在演化算子（LEO）：参数化偏微分方程的状态到状态预测

    Latent Evolution Operator (LEO): State-to-State Forecasting of Parametric PDEs

    [https://arxiv.org/abs/2505.09063](https://arxiv.org/abs/2505.09063)

    本文提出潜在演化算子（LEO），一种能以任意时刻状态、物理参数和预测时间跨度为输入、在单次潜在空间评估中直接预测参数化偏微分方程未来状态的参数化映射，从而避免了逐步积分带来的误差累积，并实现了从中间状态出发的泛化预测。

    

    在多查询场景下，使用高保真求解器求解随时间变化的参数化偏微分方程（PDE）代价高昂，这促使人们研究潜在空间中的降阶模型。现有方法通常要么是（i）递归潜在动力学模型，这类模型需要逐步积分并在长时间预测中累积误差；要么是（ii）神经算子，这类模型能够实现直接预测，但通常从初始条件开始训练，并未显式地强制实现从中间状态的泛化能力。我们提出了潜在演化算子，这是一种参数化映射，它接收任意时刻的状态，连同物理参数和预测时间跨度，并在单次潜在空间评估中预测未来状态。在标准的流形和正则性假设下，有限时间演化算子可以分解为编码器-传播器-解码器的结构。我们使用神经网络来近似这一结构。

    arXiv:2505.09063v2 Announce Type: replace  Abstract: Solving time-dependent parametric partial differential equations (PDEs) with high-fidelity solvers is expensive in multi-query settings. This motivates reduced-order models in latent space. Existing methods are typically either (i) recurrent latent dynamics models, which require step-by-step integration and accumulate error over long horizons, or (ii) neural operators, which enable direct prediction but are usually trained from initial conditions and do not explicitly enforce generalization from intermediate states.   We introduce the Latent Evolution Operator (LEO), a parameterized map that takes a state at an arbitrary time, together with physical parameters and a forecast horizon, and predicts the future state in a single latent-space evaluation. Under standard manifold and regularity assumptions, finite-time evolution operators admit an encoder-propagator-decoder decomposition. We approximate this structure using neural networks 
    
[^380]: GVPO：面向大语言模型后训练的组方差策略优化

    GVPO: Group Variance Policy Optimization for Large Language Model Post-Training

    [https://arxiv.org/abs/2504.19599](https://arxiv.org/abs/2504.19599)

    本文提出组方差策略优化（GVPO），通过将KL约束奖励最大化的解析解直接融入梯度权重，解决了GRPO等后训练方法训练不稳定的问题，并保证唯一的最优解。

    

    后训练在改进和对齐大语言模型以满足特定任务和人类偏好方面发挥着至关重要的作用。尽管最近的后训练技术（如组相对策略优化GRPO）通过增加采样数量并结合相对奖励评分取得了卓越的性能，但这些方法常常存在训练不稳定的问题，限制了其实际应用。作为进一步的改进，我们提出了组方差策略优化（GVPO）。GVPO将KL约束奖励最大化的解析解直接融入到其梯度权重中，确保与最优策略保持一致。该方法具有直观的物理解释：其梯度反映了隐式奖励的中心距离与实际奖励的中心距离之间的均方误差。GVPO具有两个关键优势：（1）它保证了唯一的最优解，恰好是KL约束奖励最大化问题的解（原文此处截断）。

    arXiv:2504.19599v4 Announce Type: replace  Abstract: Post-training plays a crucial role in refining and aligning large language models to meet specific tasks and human preferences. While recent advancements in post-training techniques, such as Group Relative Policy Optimization (GRPO), leverage increased sampling with relative reward scoring to achieve superior performance, these methods often suffer from training instability that limits their practical adoption. As a next step, we present Group Variance Policy Optimization (GVPO). GVPO incorporates the analytical solution to KL-constrained reward maximization directly into its gradient weights, ensuring alignment with the optimal policy. The method provides intuitive physical interpretations: its gradient mirrors the mean squared error between the central distance of implicit rewards and that of actual rewards. GVPO offers two key advantages: (1) it guarantees a unique optimal solution, exactly the KL-constrained reward maximization o
    
[^381]: 少即是多：通过最小可解释子集选择实现高效黑盒归因

    Less is More: Efficient Black-box Attribution via Minimal Interpretable Subset Selection

    [https://arxiv.org/abs/2504.00470](https://arxiv.org/abs/2504.00470)

    提出LiMA方法，将黑盒归因形式化为有序可解释子集选择问题，借助次模优化的边际增益原理和综合评估语义一致性、协作性、预测置信度与语义有效性的复合目标，高效识别对模型决策最关键的输入区域。

    

    为了构建可信赖的AI系统，使用归因方法理解模型行为至关重要，归因方法旨在识别对模型决策影响最大的输入区域。然而，对于图像等离散输入，由于组合爆炸问题，准确识别输入与预测之间的交互尤其具有挑战性。我们将黑盒归因问题形式化为一个有序的可解释子集选择问题。受次模优化边际增益原理的启发，该形式化方法允许输入子区域之间存在冗余、正互补性和负干扰。我们提出了LiMA（Less input is More faithful for Attribution，即更少输入带来更忠实的归因），该方法通过一系列选定子集对子区域的贡献进行排序。LiMA采用一种复合目标函数，通过语义一致性、协作性、预测置信度和语义有效性来评估子集。我们进一步设计了一个双向……（原文摘要在此截断）

    arXiv:2504.00470v3 Announce Type: replace  Abstract: To develop a trustworthy AI system, it is essential to understand its behavior using attribution methods, which aim to identify the input regions that most influence the model's decisions. For discrete inputs such as images, however, accurately identifying input-prediction interactions is particularly challenging because of combinatorial explosion. We formulate black-box attribution as an ordered interpretable subset-selection problem. Inspired by the marginal-gain principle of submodular optimization, the formulation permits redundancy, positive complementarity, and negative interference among input sub-regions. We propose LiMA (Less input is More faithful for Attribution), which ranks sub-regions by their contributions along a sequence of selected subsets. LiMA employs a composite objective that evaluates subsets through semantic consistency, collaboration, prediction confidence, and semantic effectiveness. We further design a bidi
    
[^382]: LeanTTA：一种面向边缘设备上量化测试时自适应的无反向传播、无状态方法

    LeanTTA: A Backpropagation-Free and Stateless Approach to Quantized Test-Time Adaptation on Edge Devices

    [https://arxiv.org/abs/2503.15889](https://arxiv.org/abs/2503.15889)

    本文提出LeanTTA框架，通过无反向传播、无状态地动态更新归一化统计量，实现了面向边缘设备的量化测试时自适应，显著降低内存与计算开销且不依赖大批量或历史数据。

    

    尽管在边缘设备上部署机器学习模型具有诸多优势，但移动平台的资源限制、环境的动态特性，以及训练数据与真实场景数据分布之间的差异，使得此类部署充满挑战。现有的测试时自适应方法通常内存开销大，且并未针对量化兼容性或低资源设备部署而设计。为应对这些挑战，我们提出了LeanTTA，一个专为边缘设备量身打造的、无反向传播且无状态的新型量化测试时自适应框架。我们的方法通过在不使用反向传播的情况下动态更新归一化统计量来最小化计算成本，使LeanTTA摆脱了依赖大批量数据和历史数据的常见缺陷，从而在现实部署场景中具有更强的鲁棒性。我们的方法是首个通过组合……来实现进一步计算收益的方法。

    arXiv:2503.15889v2 Announce Type: replace-cross  Abstract: While there are many advantages to deploying machine learning models on edge devices, the resource constraints of mobile platforms, the dynamic nature of the environment, and differences between the distribution of training versus in-the-wild data make such deployments challenging. Current test-time adaptation methods are often memory-intensive and not designed to be quantization-compatible or deployed on low-resource devices. To address these challenges, we present LeanTTA, a novel backpropagation-free and stateless framework for quantized test-time adaptation tailored to edge devices. Our approach minimizes computational costs by dynamically updating normalization statistics without backpropagation, which frees LeanTTA from the common pitfall of relying on large batches and historical data, making our method robust to realistic deployment scenarios. Our approach is the first to enable further computational gains by combining 
    
[^383]: 用于声波场快速模拟的Gabor增强物理信息神经网络

    Gabor-Enhanced Physics-Informed Neural Networks for Fast Simulations of Acoustic Wavefields

    [https://arxiv.org/abs/2502.17134](https://arxiv.org/abs/2502.17134)

    提出一种引入显式可训练Gabor基函数的简化PINN框架，将网络任务重新定义为学习从输入坐标到自定义Gabor坐标系的非线性映射，有效克服了PINNs的低频偏差问题，实现高频声波场的快速高精度模拟。

    

    物理信息神经网络（PINNs）因其灵活性和无网格公式化而在求解偏微分方程（包括散射Helmholtz方程）方面备受关注。然而，其性能受低频偏差的影响，尤其在高频波场模拟中，这限制了收敛速度和精度。为解决这一问题，我们提出了一种新颖且简化的PINN框架，该框架引入显式的、可训练的Gabor基函数，以高效捕捉波场的局部化和振荡特性。与以往依赖乘法滤波器或辅助网络来学习Gabor参数的基于Gabor的PINN不同，我们的方法将网络的任务重新定义为学习从输入坐标到自定义Gabor坐标系的非线性映射，其中Gabor函数捕捉波场的主导振荡行为。这种表述吸收了两个Ga（摘要在此处截断）

    arXiv:2502.17134v3 Announce Type: replace-cross  Abstract: Physics-Informed Neural Networks (PINNs) have gained attention for solving partial differential equations, including the scattered Helmholtz equation, due to their flexibility and mesh-free formulation. However, their performance suffers from low-frequency bias, particularly in high-frequency wavefield simulations, limiting convergence speed and accuracy. To address this, we propose a novel and simplified PINN framework that incorporates explicit, trainable Gabor basis functions to efficiently capture the localized and oscillatory nature of wavefields. Unlike previous Gabor-based PINNs that rely on multiplicative filters or auxiliary networks to learn Gabor parameters, our approach redefines the network's task as learning a nonlinear mapping from input coordinates to a custom Gabor coordinate system, where a Gabor function captures the dominant oscillatory behavior of the wavefield. This formulation absorbs the effect of two Ga
    
[^384]: 因果模型与因果表示中的“因果”究竟指什么？

    What is causal about causal models and representations?

    [https://arxiv.org/abs/2501.19335](https://arxiv.org/abs/2501.19335)

    该论文提出了一个将因果模型与数据生成过程分离的形式化框架，形式化了连接行动与干预的解释，并证明了不存在能同时满足一组期望性质且保持非循环性的行动-干预解释这一不可能性结果。

    

    因果贝叶斯网络通常被描述为“因果的”，因为它们支持关于干预的推理。然而，干预是在数学模型中定义的，而模型的预测涉及的是现实世界中行动的效果。因此，因果模型是否以及如何对行动的效果做出可证伪的预测，取决于行动如何被解释为干预。这需要明确哪些行动对应哪些干预，而这一步骤往往被隐式处理。我们开发了一个形式化框架，将因果模型与其旨在描述的数据生成过程分离开来，并将连接行动与干预的解释形式化。在该框架内，我们证明了一个不可能性结果：没有任何解释能够在保持非循环性的同时满足一组期望的性质。在循环解释下，每一个拟合观测数据的模型……

    arXiv:2501.19335v3 Announce Type: replace-cross  Abstract: Causal Bayesian networks are often described as causal because they support reasoning about interventions. Yet interventions are defined in the mathematical model, whereas the resulting model predictions concern the effects of actions in the real world. Consequently, whether and how a causal model makes falsifiable predictions about the effects of actions depends on how actions are interpreted as interventions. This requires specifying which actions correspond to which interventions, a step that is frequently left implicit. We develop a formal framework that separates a causal model from the data-generating process it aims to describe and formalizes interpretations connecting actions to interventions. Within this framework, we prove an impossibility result: no interpretation can simultaneously satisfy a collection of desirable properties while remaining non-circular. Under a circular interpretation, every model that fits the ob
    
[^385]: 重新思考用于概念抑制的负向提示：无条件锚定与局部化引导

    Rethinking Negative Prompting for Concept Suppression: Unconditional Anchoring and Localized Guidance

    [https://arxiv.org/abs/2412.07658](https://arxiv.org/abs/2412.07658)

    提出无需训练的TraSCE方法，通过无条件锚定与局部化引导，解决负向提示在越狱攻击下因“引导崩塌”而失效的问题，从而有效抑制文生图模型生成有害或不良概念。

    

    文生图扩散模型已迅速从研究原型转变为被广泛部署的工具。在具备强大能力的同时，这些模型也可能生成有害或不良内容（例如NSFW图像），这促使人们需要实用的机制来移除此类概念。负向提示利用负向提示的预测作为引导锚点，是将采样轨迹引导远离不需要概念的最常用机制之一。然而，面对越狱提示，它可能在我们称之为“引导崩塌”的情形下失效：模型对正向提示和负向提示的去噪预测变得相似，使得驱动负向提示的引导差异消失，引导后的预测退化为负向提示的预测，从而削弱了概念抑制效果。为解决这一问题，我们提出了TraSCE（Trajectory Steering for ConcEpt Suppression，用于概念抑制的轨迹引导），这是一种无需训练的方法……

    arXiv:2412.07658v3 Announce Type: replace-cross  Abstract: Text-to-image diffusion models have rapidly transitioned from research prototypes to widely deployed tools. Along with their impressive capabilities, these models can also generate harmful or undesirable content (e.g., NSFW imagery), motivating practical mechanisms for removing such concepts. Negative prompting, which uses the negative-prompt prediction as the guidance anchor, is among the most widely used mechanisms for steering sampling trajectories away from unwanted concepts. However, for jailbreak prompts, it can fail in a regime we refer to as \emph{guidance collapse}, where the model's denoising predictions for the prompt and negative prompt become similar, so that the guidance difference driving negative prompting vanishes and the guided prediction degenerates to the negative-prompt prediction, undermining suppression. To fix this issue, we propose TraSCE (Trajectory Steering for ConcEpt Suppression), a training-free in
    
[^386]: 框外作画：视觉定位训练数据的合成与选择

    Paint Outside the Box: Synthesizing and Selecting Training Data for Visual Grounding

    [https://arxiv.org/abs/2412.00684](https://arxiv.org/abs/2412.00684)

    提出 POBF 框架，通过在边界框外进行图像修复来合成训练数据以解决标签错位问题，并利用结合难度分数与过拟合分数的过滤机制筛选最有效的训练数据，从而在数据稀缺条件下显著提升视觉定位性能。

    

    视觉定位旨在根据文本查询定位图像中的相应区域。鉴于大规模数据整理的困难，本文研究了如何在数据稀缺的设置下有效学习视觉定位。为解决数据稀缺问题，我们提出了一种新颖的框架 POBF（Paint Outside the Box and Filter，框外作画与过滤）。POBF 通过在边界框之外进行图像修复来合成图像，解决了以往工作中遇到的标签错位问题。此外，POBF 利用一种创新的过滤方案来选择最有效的训练数据。该方案结合了难度分数和过拟合分数，并通过一个惩罚项加以平衡。在四个基准数据集上的大量实验表明，POBF 持续提升性能，相比仅使用真实数据的方法平均提升 5.83%，在准确率上超越领先的基线方法 2.29%-3.85%。

    arXiv:2412.00684v3 Announce Type: replace-cross  Abstract: Visual grounding aims to localize the image regions based on a textual query. Given the difficulty of large-scale data curation, we investigate how to effectively learn visual grounding under data-scarce settings in this paper. To address the data scarcity, we propose a novel framework, POBF (Paint Outside the Box and Filter). POBF synthesizes images by inpainting outside the box, tackling a label misalignment issue encountered in previous works. Furthermore, POBF leverages an innovative filtering scheme to select the most effective training data. This scheme combines a hardness score and an overfitting score, balanced by a penalty term. Extensive experiments across four benchmark datasets demonstrate that POBF consistently improves performance, achieving an average gain of 5.83\% over the real-data-only method and outperforming leading baselines by 2.29\%-3.85\% in accuracy. Additionally, we validate the robustness and general
    
[^387]: 基于标记随机有限集的岩土数据库土性参数估计

    Property Estimation in Geotechnical Databases Using Labeled Random Finite Sets

    [https://arxiv.org/abs/2410.22659](https://arxiv.org/abs/2410.22659)

    本文提出在标记随机有限集（LRFS）框架下，利用广义标记多伯努利（GLMB）滤波器从贝叶斯视角处理含噪声和不完整的岩土数据，实现了岩土数据库中的土性参数估计。

    

    充足且准确的数据是数据中心岩土工程学的核心要素。然而，岩土数据集本质上具有不确定性，这使得工程师难以获得精确的信息来支持决策。当数据驱动技术的性能完全依赖于不完善的数据库，或者有时难以对场地进行实地勘察时，这一挑战显得尤为突出。本文在标记随机有限集（LRFS）框架内提出了从含噪声和不完整数据中进行岩土特性估计的方法。我们利用广义标记多伯努利（GLMB）滤波器——多目标估计问题的一种基础解决方案——从贝叶斯视角处理测量不确定性。特别是，本工作重点关注了LRFS与岩土工程中大型间接数据（BID）之间的相似性，由于两者的特征彼此相近，这使得GLMB滤波能够（应用于岩土特性估计）。

    arXiv:2410.22659v1 Announce Type: cross  Abstract: The sufficiency of accurate data is a core element in data-centric geotechnics. However, geotechnical datasets are essentially uncertain, whereupon engineers have difficulty with obtaining precise information for making decisions. This challenge is more apparent when the performance of data-driven technologies solely relies on imperfect databases or even when it is sometimes difficult to investigate sites physically. This paper introduces geotechnical property estimation from noisy and incomplete data within the labeled random finite set (LRFS) framework. We leverage the ability of the generalized labeled multi-Bernoulli (GLMB) filter, a fundamental solution for multi-object estimation, to deal with measurement uncertainties from a Bayesian perspective. In particular, this work focuses on the similarity between LRFSs and big indirect data (BID) in geotechnics as those characteristics resemble each other, which enables GLMB filtering to
    
[^388]: 利用有向图中的信任：重新定义图学习的鲁棒性

    Harnessing Trust in Directed Graphs: Redefining Robustness of Graph Learning

    [https://arxiv.org/abs/2306.02002](https://arxiv.org/abs/2306.02002)

    该论文从攻击和防御两个角度重新定义了有向图上图神经网络的鲁棒性，并提出了一种即插即用的消息传递框架，通过利用有向图链接方向中蕴含的信任信号显著提升GNN的鲁棒性。

    

    现有关于鲁棒图神经网络（GNN）的研究主要集中于无向图，忽视了有向图链接方向中固有的可信性。在本工作中，我们从攻击和防御两个角度分析了当前方法的局限性，并重新定义了有向图中GNN的鲁棒性。在攻击方面，我们引入了一种更贴近现实的有向图攻击设置，弥补了现有攻击方法的不足。在防御方面，我们提出了一个简单而有效的消息传递框架，可作为即插即用的插件层来增强GNN的鲁棒性，同时避免产生虚假的鲁棒性错觉。我们的研究结果表明，利用有向图中编码的信任信号可以显著提高GNN的鲁棒性。与现有防御策略结合时，该框架在实现最先进的鲁棒性能的同时，还保持了较高的干净数据准确率。

    arXiv:2306.02002v2 Announce Type: replace-cross  Abstract: Existing research on robust Graph Neural Networks (GNNs) focuses predominantly on undirected graphs, overlooking the trustworthiness inherent in the link directions of directed graphs. In this work, we analyze the limitations of current approaches from both the attack and defense perspectives and redefine the robustness of GNNs in directed graphs. From the attack side, we introduce a more realistic directed graph attack setting that addresses the shortcomings of existing attacks. From the defense side, we propose a simple yet effective message-passing framework that serves as a plug-in layer to strengthen GNN robustness while avoiding a false sense of robustness. Our findings show that the trust signals encoded in directed graphs can be harnessed to substantially improve the robustness of GNNs. When combined with existing defense strategies, the framework achieves state-of-the-art robust performance alongside strong clean accur
    
[^389]: 策略梯度真的是梯度吗？

    Is the Policy Gradient a Gradient?

    [https://arxiv.org/abs/1906.07073](https://arxiv.org/abs/1906.07073)

    本文证明了省略折扣因子的策略梯度更新不必是任何函数的梯度，并构造反例表明该更新会使策略收敛到在折扣与未折扣目标下均全局最差的确定性极限。

    

    策略梯度定理描述了期望折扣回报关于智能体策略参数的梯度。然而，许多策略梯度方法省略了外层折扣因子，由此产生的更新不必等于折扣目标的梯度。那么这个更新是否是另一个目标的梯度呢？这多年来一直是一个悬而未决的问题，这种理论上的不清晰也导致文献中出现了大量错误的表述。本文针对由此产生的按回合求和的在策略更新回答了这一问题，通过证明证明该更新不必等于任何函数的梯度。此外，作者还构建了一个反例：在该反例中，精确的期望更新会将策略推向一个确定性的极限，该极限无论对于折扣目标还是未折扣目标而言都是全局最差的，同时参数会发散。作者通过调研文献来引出这项工作，并表明文献中仍然存在（大量相关误用）。

    arXiv:1906.07073v3 Announce Type: replace  Abstract: The policy gradient theorem describes the gradient of the expected discounted return with respect to an agent's policy parameters. However, many policy gradient methods omit the outer discount factor, yielding an update that need not equal the gradient of the discounted objective. Is the resulting update the gradient of another objective? This has been an open question for several years, and this lack of theoretical clarity has led to an abundance of misstatements in the literature. We answer this question for the resulting episode-summed, on-policy update by proving that it need not equal the gradient of any function. Further, we construct a counterexample in which exact expected updates drive the policy toward a deterministic limit that is globally pessimal with respect to both the discounted and undiscounted objectives, while the parameter diverges. We motivate this work by surveying the literature and showing that there remains a
    
[^390]: 无人机在多智能体强化学习中的动态避障路径规划

    UAV Pathfinding in Dynamic Obstacle Avoidance with Multi-agent Reinforcement Learning. (arXiv:2310.16659v1 [cs.RO])

    [http://arxiv.org/abs/2310.16659](http://arxiv.org/abs/2310.16659)

    本文提出了一种基于多智能体强化学习的集中式训练和分布式执行方法，以在线解决动态障碍物避障问题。实验结果验证了该方法的有效性。

    

    多智能体强化学习方法在动态和不确定的场景中在线规划智能体的可行且安全的路径具有重要意义。本文提出了一种基于多智能体强化学习的集中式训练和分布式执行方法，以在线解决动态障碍物避障问题。在该方法中，每个智能体仅与中央规划者或其邻居进行通信，以在线规划可行且安全的路径。我们基于模型预测控制的思想改进了我们的方法，以提高智能体的训练效率和采样利用率。在模拟、室内和室外环境中的实验结果验证了我们方法的有效性。

    Multi-agent reinforcement learning based methods are significant for online planning of feasible and safe paths for agents in dynamic and uncertain scenarios. Although some methods like fully centralized and fully decentralized methods achieve a certain measure of success, they also encounter problems such as dimension explosion and poor convergence, respectively. In this paper, we propose a novel centralized training with decentralized execution method based on multi-agent reinforcement learning to solve the dynamic obstacle avoidance problem online. In this approach, each agent communicates only with the central planner or only with its neighbors, respectively, to plan feasible and safe paths online. We improve our methods based on the idea of model predictive control to increase the training efficiency and sample utilization of agents. The experimental results in both simulation, indoor, and outdoor environments validate the effectiveness of our method. The video is available at htt
    
[^391]: 加速梯度方法用于非凸优化：逃逸轨迹和收敛到局部极小值点

    Accelerated gradient methods for nonconvex optimization: Escape trajectories from strict saddle points and convergence to local minima. (arXiv:2307.07030v1 [math.OC])

    [http://arxiv.org/abs/2307.07030](http://arxiv.org/abs/2307.07030)

    本文研究了一类加速梯度方法在非凸优化问题上的行为，包括逃离鞍点和收敛到局部极小值点的分析。研究在渐进和非渐进情况下，提出了一类新的Nesterov类型的加速方法，并回答了Nesterov加速梯度方法是否避免了严格鞍点的问题。

    

    本文研究了一类广义的加速梯度方法在光滑非凸函数上的行为。通过对Polyak的重球方法和Nesterov加速梯度方法进行改进，以实现对非凸函数局部极小值的收敛，本文提出了一类Nesterov类型的加速方法，并通过渐进分析和非渐进分析对这些方法进行了严格研究，包括逃离鞍点和收敛到局部极小值点。在渐进情况下，本文回答了一个开放问题，即带有可变动量参数的Nesterov加速梯度方法（NAG）是否几乎必定避免了严格鞍点。本文还提出了两种渐进收敛和发散的度量方式，并对几种常用的标准加速方法（如NAG和Ne）进行了评估。

    This paper considers the problem of understanding the behavior of a general class of accelerated gradient methods on smooth nonconvex functions. Motivated by some recent works that have proposed effective algorithms, based on Polyak's heavy ball method and the Nesterov accelerated gradient method, to achieve convergence to a local minimum of nonconvex functions, this work proposes a broad class of Nesterov-type accelerated methods and puts forth a rigorous study of these methods encompassing the escape from saddle-points and convergence to local minima through a both asymptotic and a non-asymptotic analysis. In the asymptotic regime, this paper answers an open question of whether Nesterov's accelerated gradient method (NAG) with variable momentum parameter avoids strict saddle points almost surely. This work also develops two metrics of asymptotic rate of convergence and divergence, and evaluates these two metrics for several popular standard accelerated methods such as the NAG, and Ne
    

