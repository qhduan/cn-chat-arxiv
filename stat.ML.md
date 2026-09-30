# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ReCIRC: Rectified Conformal Risk Control](https://arxiv.org/abs/2609.38112) | ReCIRC通过将每个输入的估计局部风险曲线求逆，把校准阈值重新参数化为代表共同目标条件风险的风险预算，在无论曲线估计准确与否都保持有限样本边际保证的同时，实现近似的条件风险控制，解决了传统共形风险控制对简单样本保护过度、对困难样本保护不足的问题。 |
| [^2] | [Traversing the solution space of neural networks with Hessian Null Space Continuation](https://arxiv.org/abs/2609.38081) | 本文提出Hessian零空间延拓（HNC）方法，利用局部曲率遍历权重空间中保持网络功能不变的区域，首次揭示了局部模式连通区域内存在多种不同的内部计算机制。 |
| [^3] | [Latent Inference-Time Guidance of Time Series Foundation Models](https://arxiv.org/abs/2609.38058) | 本文提出潜空间推理时引导方法，通过具有独立分量的时间相关潜空间自适应集成多个时间序列基础模型的预测，在保持开箱即用特性的同时提供了可识别性与重构保证。 |
| [^4] | [When do data mixtures improve scaling laws? Insights from high-dimensional regression](https://arxiv.org/abs/2609.38011) | 该论文通过构建共享回归函数、异构协方差与噪声水平的高维混合数据回归模型，建立了极小极大风险界并推导出岭回归测试误差的确定性等价，从理论上刻画了辅助数据何时能真正改进缩放定律而非仅仅提供更多样本。 |
| [^5] | [Mutual Information Constrained Chernoff Bottleneck](https://arxiv.org/abs/2609.37994) | 本文提出互信息约束下的Chernoff瓶颈问题，刻画了最优误差指数 $C(R)$ 随速率 $R$ 的变化规律（严格递增至 $H(V)$ 后保持不变、且不必为凹函数），证明 $k+1$ 个输出足以达到最优值，并给出了交替优化算法。 |
| [^6] | [A shape-similarity latent space for fluid interfaces: invertible reduced-order modelling of droplet morphology](https://arxiv.org/abs/2609.37947) | 本文提出 SHROM 框架，通过结合平方根速度函数形状表示、形状空间邻居图和自编码器，构建了一个低维、可逆且忠实于形状相似性的潜在空间，实现了液滴形态的可逆降阶建模，从而从稀疏的实验记录中恢复单次拍摄所遗漏的液滴中间形态状态。 |
| [^7] | [Identifiability Guarantees for Drivers and Dynamics of Delayed Physical Systems](https://arxiv.org/abs/2609.37944) | 本文提出一种有理论支撑的方法，证明在宽松假设下随机时滞微分方程的结构驱动项与漂移项是可辨识的，并在驱动项可辨识性与动力学物理一致性基准上优于现有方法。 |
| [^8] | [Post-Anomaly Detection Inference for Deep SVDD](https://arxiv.org/abs/2609.37935) | 本文提出PADI框架，利用选择性推断方法为冻结的Deep SVDD异常检测器的异常判定提供条件于“被识别为异常”事件下的严格统计有效性推断，解决了异常决策缺乏统计保证的问题。 |
| [^9] | [Search Dimension in Unlabeled Projection Pursuit: A Scaling Law for Subspace Restriction](https://arxiv.org/abs/2609.37917) | 该论文证明在无标签投影追踪中搜索维度本身是关键的统计量：过大的高斯补空间会让无信号方向最小化经验目标，而将搜索限制在与算子判别增益相匹配的子空间可把所需样本量的标度从 ς⁻⁸ 改善到 ς⁻⁴。 |
| [^10] | [Counterfactual Probing for Parallel Unmasking with Hidden Forest Structure](https://arxiv.org/abs/2609.37841) | 该论文提出一种反事实探测方法，在依赖关系未知的隐藏森林结构下实现并行去掩码采样，使得包括依赖发现在内的总模型评估次数和串行深度均相对序列长度 $N$ 达到亚线性复杂度，同时保证有界的采样误差。 |
| [^11] | [When Noise Estimation Hides Basis Misspecification in Repeated Bayesian Inverse Problems](https://arxiv.org/abs/2609.37762) | 本文揭示在贝叶斯逆问题中估计观测噪声方差会掩盖基底误设所导致的覆盖性损失，并提出利用John球形度统计量检验投影补空间中残差样本谱的形状，以在重复问题中检测这种隐藏的失效。 |
| [^12] | [SPARK: A General Goodness-of-Fit Assessment via Residual Projection](https://arxiv.org/abs/2609.37705) | 本文提出SPARK框架，通过去偏策略将初始拟合的残差投影到近似正交方向并采用基于核的投影方法提取残留信号，从而实现对传统统计模型和黑箱学习器的通用拟合优度检验。 |
| [^13] | [The double descent and Runge phenomena in overparametrized polynomial interpolation](https://arxiv.org/abs/2609.37657) | 该论文系统研究了单项式、切比雪夫和勒让德三种多项式基下具有最小范数系数的过参数化多项式插值，揭示了龙格现象与机器学习中双重下降现象之间的联系。 |
| [^14] | [A Finslerian Approach for Embedding Directed Data](https://arxiv.org/abs/2609.37649) | 该论文提出将有向数据建模为距离依赖方向的芬斯勒流形上的样本，并证明由这种非对称距离构造的核算子可分解为收敛于加权拉普拉斯算子的对称部分（恢复扩散映射）和收敛于一阶传输算子的反对称部分，从而将有向数据的几何结构与方向信息分离开来。 |
| [^15] | [RLTL;DR: Self-improvement by Internalizing Self-generated Feedback](https://arxiv.org/abs/2609.37633) | 提出RLTL;DR方法，让策略在每次失败后根据验证器输出撰写自己的TL;DR见解，并将其作为后续尝试的上下文条件，同时通过反向传播将这些见解内化为任务到见解的直接映射，从而在无教师模型、任务难度极高（Pass@128=0）的场景下实现自我提升。 |
| [^16] | [XU-RS: Explaining Credal Width in Random-Set Language Models](https://arxiv.org/abs/2609.37594) | 提出XU-RS框架，利用期望梯度将随机集语言模型中答案的信度宽度（即认知不确定性）归因到输入词元，从而解释模型不确定性的来源。 |
| [^17] | [Ornstein-Uhlenbeck Is Hard to Beat, Yet Superlinear Drift Ships Lower Transport Costs](https://arxiv.org/abs/2609.37579) | 本文通过基于分数的图像生成实验证明，超线性朗之万扩散模型在经验Wasserstein距离上几乎全面优于奥恩斯坦-乌伦贝克基线且波动更小，表明OU扩散“难以被超越”的结论并非普遍成立。 |
| [^18] | [Why Adaptive Optimizers Underestimate Rare Tokens](https://arxiv.org/abs/2609.37535) | 本文揭示了Adam等逐坐标自适应优化器因将更新除以在词元出现后达到峰值的幅度运行估计，从而系统性地低估稀有词元，并精确刻画了哪些优化器（如SGD、Shampoo、Muon）能保持输出层平均嵌入、哪些（如Adam、Adafactor、Lion、符号下降）不能。 |
| [^19] | [Physical Muon: Orthogonalization as an Equilibrium Computation](https://arxiv.org/abs/2609.37525) | 该论文提出Physical Muon优化器，将Muon的正交化步骤转化为连续时间流的平衡态计算，仅需矩阵-向量乘积、倒数读取和局部秩1写入等模拟硬件友好的操作，从而在保持训练性能的同时实现物理可实现的优化。 |
| [^20] | [Probability Contracts: Accuracy, Coherence, and Decisions Across LLM Interfaces](https://arxiv.org/abs/2609.37470) | 该论文提出“概率契约”基准，通过将精确有限世界后验、经验证的事件变换与失效感知决策评估相结合，系统揭示了大语言模型不同概率接口在准确性、一致性和决策损失上的显著差异，并证明接口间分歧不足以认证真实误差。 |
| [^21] | [A stochastic subgradient method with optimal failure exponent](https://arxiv.org/abs/2609.37425) | 本文提出一种采用调和型均匀平均步长调度的随机次梯度方法，通过指数上鞅论证与相匹配的不可能性结果证明，其在次高斯梯度噪声下最小化超出目标精度的失败概率时达到了最优失败指数。 |
| [^22] | [High-Dimensional Simulation-Based Inference in Latent Spaces](https://arxiv.org/abs/2609.37381) | 提出将基于模拟的推断与潜在生成式建模相融合的新框架，通过学习模拟器参数的低维表示、在潜在空间中直接进行后验推断再将样本映射回原始参数空间，从而解决高维参数空间的推断问题，并给出了潜在空间推断恢复目标后验分布的理论条件。 |
| [^23] | [A Sharp Transition in Data Reconstruction under Differential Privacy](https://arxiv.org/abs/2609.37344) | 本文在零集中差分隐私下建立了数据重建的急剧相变：当隐私预算 ρ 低于数据维度 d 量级时，即使攻击者知道其余全部训练数据，任何机制和攻击都无法在信息论意义上实现精确重建，从而为隐私预算的选择提供了理论边界。 |
| [^24] | [Interacting particle guidance for sampling reward-tilted generative priors](https://arxiv.org/abs/2609.37227) | 提出相互作用粒子引导（IPG），利用由Feynman–Kac偏微分方程导出的漂移项驱动粒子输运，取代传统的重加权机制，从而克服序贯蒙特卡罗方法中的权重退化和粒子坍缩问题，实现从奖励倾斜生成先验中的高效采样。 |
| [^25] | [Pointwise or Pairwise: When Do Pairwise Losses Help Reward Learning, Provably?](https://arxiv.org/abs/2609.37209) | 该论文在允许每个上下文包含多个动作的分组离线上下文老虎机设置下，首次从理论上刻画了成对损失（价值差分回归VDR）相对于逐点价值回归（VR）何时以及为何能带来可证明的优势，并在包含与动作无关的上下文干扰项的半参数模型下给出了有限样本回归保证和离线遗憾界。 |
| [^26] | [Interpretable intrinsic dimension estimation through componentwise calibration of distance and angle](https://arxiv.org/abs/2609.37114) | 该论文将DANCo内在维度估计方法重构为可分别校准与解释的距离分量和角度分量，并为Gride估计器推导出闭式KL散度，从而在含噪与幅度异质的实际数据上显著降低估计误差（如24个流形上40%噪声水平下平均百分比误差从27.7%降至17.6%）并增强可解释性。 |
| [^27] | [Identifying ODEs from Unstructured Data with Causal Representation Learning](https://arxiv.org/abs/2609.37083) | 提出SPEED-AE框架，将预训练的因果表征学习与逐分量自编码器结合，首次从图像等非结构化高维数据中可证明地学习到适合稀疏ODE发现的变量表示，从而实现对动力系统控制方程的识别。 |
| [^28] | [Iterative Exact Discrete Guidance for Energy-Based Sampling](https://arxiv.org/abs/2609.37043) | 提出了一种迭代精确离散引导框架（IEDG），沿退火路径逐步学习玻尔兹曼倾斜的阶段局部后验修正，并以相对有效样本量自适应调节步长，从而实现对大型离散状态空间上非归一化多峰目标分布的精确采样。 |
| [^29] | [Scalable Diffusion SBI for Compositional Inference under Simulator Misspecification](https://arxiv.org/abs/2609.36950) | 该论文提出了可扩展的扩散式模拟推断方法：通过考虑观测数量的连续时间扩散系数扩展组合式分数推断，并引入层次化分块扩散采样（HBDS），使单一预训练模型无需重训即可在不同观测集合与分组下推断共享参数和分组潜在状态，同时借助路径正则化缓解模拟器误设问题。 |
| [^30] | [Beyond Conditional Independence: Root Cause Analysis with Deep Causal Models](https://arxiv.org/abs/2609.36771) | 该论文提出基于深度因果模型的根因分析方法，通过建立分布约束检验与根因分析之间的隐式联系，突破了传统方法对条件独立性和无混杂性强假设的依赖，从而能够在存在潜变量混杂的情况下对任意因果模型生成的数据进行根因识别。 |
| [^31] | [Federated Clustering with Unknown Local and Global Cluster Cardinalities](https://arxiv.org/abs/2609.36762) | 本文提出了一个局部与全局聚类数量均未知的两阶段联邦聚类框架，通过自适应分裂-合并（ASM）算法让每个客户端从自身数据估计局部聚类数量，供 FedGEM 等需要局部数量的聚合器使用。 |
| [^32] | [Into the danger zone: stable extrapolation in high-dimensional function and operator learning](https://arxiv.org/abs/2609.36709) | 该论文识别出某些全纯函数与算子类别，证明即使在大幅分布偏移下其分布外泛化误差仍能以代数速率收敛，这一现象源于高阶坐标平滑性的增加，被称为“高维之福”。 |
| [^33] | [When Is Coarse Supervision Worth It? Cost-Aware Learning under Unknown Aggregation](https://arxiv.org/abs/2609.36704) | 该论文研究了在聚合规则未知的情况下精细标签与廉价粗粒度标签之间的成本感知权衡，证明粗监督的价值由成本、噪声和可识别性共同决定，推导出粗监督的闭式盈亏平衡条件，并提出一种达到最优累积风险的“估计-追踪”在线策略。 |
| [^34] | [Understanding Private Evolution as Learning-Augmented Clustering](https://arxiv.org/abs/2609.36678) | 本文将私有进化（PE）重新表述为生成模型增强的Wasserstein学习，证明利用生成模型可获得更好的性能界（如样本复杂度取决于内在维度而非环境维度），并针对标准PE在良好聚类实例上不收敛的问题，提出了具有可证明收敛性的几何感知新算法。 |
| [^35] | [Second-Moment Stochastic Approximation Methods](https://arxiv.org/abs/2609.36600) | 本文提出基于一阶矩与二阶矩估计器的二阶矩随机逼近方法（将Adam、Muon等现代优化器纳入统一框架），通过矩阵方程最优预条件化视角推导方法，并建立两阶段收敛分析框架，证明了实用方法几乎必然收敛到目标解的邻域，且邻域大小由矩估计器的偏差和方差决定。 |
| [^36] | [Optimal detection of general moment changes: Simultaneous mean and covariance change detection and beyond](https://arxiv.org/abs/2609.36594) | 提出了一种基于张量表示的多元时间序列多阶矩变化点检测方法，可在统一框架下同时检测均值、协方差及高阶矩的变化，并在适当条件下达到极小极大最优的定位误差率。 |
| [^37] | [Hierarchical Utility Calibration for Structured Multiclass Decisions](https://arxiv.org/abs/2609.36532) | 该论文提出层次化效用校准（HUC），通过将效用误差精确分解为标签树各内部节点的贡献之和，解决了传统效用校准中正负贡献相互抵消、掩盖层次结构局部效用误差的问题。 |
| [^38] | [Optimal Multi-Reward Reinforcement Learning](https://arxiv.org/abs/2609.36486) | 该论文研究了多奖励函数的有限时域强化学习问题，提出了一个无需额外预烧成本的算法，达到了与信息论下界仅相差多对数因子的最优样本复杂度 $O(SAH^3\log M/\epsilon^2)$。 |
| [^39] | [LOCO-AdaMP: Built-in LOCO Inference for Adaptive Minipatch Ensembles with Enhanced Prediction](https://arxiv.org/abs/2609.36396) | 提出了LOCO-AdaMP框架，通过LOCO重要性引导的自适应特征采样构建小补丁集成，在无需数据拆分的情况下实现渐近有效的特征重要性推断，同时显著提升高维稀疏场景下的预测性能。 |
| [^40] | [Finite-Sample Theory for Fitted Q-Iteration When Actions Are Functions](https://arxiv.org/abs/2609.36390) | 本文首次建立了动作空间为函数（如放射治疗通量图、机器人运动轨迹）时拟合Q迭代的有限样本理论，通过提出无需动作密度的“评论家相对覆盖条件”以及平滑正则化策略搜索，克服了函数型动作空间中覆盖刻画困难、传统覆盖要求过严和贪婪优化难以实施这三大难题。 |
| [^41] | [Adapting Linear-Time Architectures for Tabular In-Context Learning](https://arxiv.org/abs/2609.36337) | 该研究发现因果线性序列混合器DeltaNet最适合表格上下文学习，性能甚至超过非因果线性注意力，但在超出预训练上下文长度2-4倍时性能退化，且实验表明这一退化并非源于隐藏状态容量限制。 |
| [^42] | [Cheap and Powerful Tests for Supervised Subspaces: Per-Component Inference for PLS](https://arxiv.org/abs/2609.36307) | 本文提出两种基于留出OLS重拟合的廉价高效检验方法（Nadeau-Bengio校正渐近t检验与置换检验），为PLS监督子空间提供统计推断，并通过固定序列检验实现对各成分的逐一推断。 |
| [^43] | [MoRE: Scaling mixture of experts with hardware-aware low-rank routing](https://arxiv.org/abs/2609.36301) | 提出MoRE方法，通过将MoE路由器权重矩阵低秩分解，把路由成本从Θ(Mh)降至O((h+M)r)，在证明可保持路由表达能力与负载均衡的同时，支持Θ(h/r)倍的更多专家，并考虑硬件实际加速。 |
| [^44] | [GNA: Granular Neighbor Assembly for Retrieval-Augmented Multivariate Time-Series Forecasting](https://arxiv.org/abs/2609.36281) | 该论文提出GNA，一个用于多元时间序列预测的检索增强层，它在完整窗口和单变量两个粒度上检索并组装相似历史邻居，并通过可学习门控将其与骨干网络预测及持续性预测融合，从而突破了固定长度回看窗口的局限。 |
| [^45] | [Stochastic Optimization Under Power-Law Spectra: Tight Bounds and Shuffling Analysis](https://arxiv.org/abs/2609.36271) | 本文将幂律谱收敛理论推广至随机梯度下降，并精确证明对各向同性高斯数据而言，单次洗牌的采样策略严格优于交替洗牌和IID采样。 |
| [^46] | [OTROPE: Optimal Transport-based Robust Off-policy Evaluation for Large Language Models](https://arxiv.org/abs/2609.36264) | 提出 OTROPE，一种基于最优传输、无需似然值的大语言模型离线策略评估方法，通过在语义空间对齐行为策略与目标策略样本，实现无需密度比估计和策略建模的双重鲁棒式评估，适用于黑盒大语言模型。 |
| [^47] | [The Signed Geometry of One-Shot Recourse: On-Path Validity and the Signed-Curvature Criterion](https://arxiv.org/abs/2609.36252) | 该论文证明单次解析式反事实补救能否一步成功由路径曲率 $\kappa=\hat g^\top\nabla^2 f(x)\hat g$ 的符号决定（非负则有效），给出仅凭分数与梯度的规则不可避免存在 $Kd_p^2/\|\nabla f(x)\|$ 量级过冲的下界，并证明在利普希茨曲率下于承诺点评估一次分数即可达到极小极大最优有效性。 |
| [^48] | [Can Representation Learning Decouple from Loss Minimization? Polar Updates Have an Answer](https://arxiv.org/abs/2609.36240) | 该论文证明，在 Muon 极化更新引发的损失平台期与振荡阶段，表示学习并未停止——权重持续移动、特征持续与教师子空间对齐，AGOP 主特征空间甚至在损失下降之前就已精确恢复教师子空间。 |
| [^49] | [One-Step Next-Latent Prediction Is Not a World Model](https://arxiv.org/abs/2609.36227) | 该论文从理论上证明单步下一隐变量预测（如LeNEPA）只识别出条件均值而非可展开的世界模型转移核，其多步开环误差随预测时域增长，且在非线性条件均值或非单射观测情形下无法通过复合得到正确的多步预测。 |
| [^50] | [Copula Active Subspaces I: A Score-Covariance Method for Reduced-Order Non-Gaussian Density Estimation](https://arxiv.org/abs/2609.36142) | 提出 Copula 活跃子空间（CAS）方法，利用 copula 得分协方差的主特征向量识别非高斯噪声分布中依赖结构的变化方向，从而实现贝叶斯推断中非高斯噪声密度的降阶表示与估计。 |
| [^51] | [Graph-Split Bayesian Causal Forest for Spatial Heterogeneous Treatment Effect Estimation](https://arxiv.org/abs/2609.36046) | 提出图分割贝叶斯因果森林（GSBCF），通过将图分割贝叶斯加性回归树与贝叶斯因果森林倾向得分回归框架相结合，克服了传统轴对齐分割规则无法建模空间结构的局限，实现了空间异质性处理效应的估计。 |
| [^52] | [Intrinsic Associative Memory on Riemannian Manifolds: Curvature, Capacity, and Emergent Modes](https://arxiv.org/abs/2609.35948) | 该论文在黎曼流形上构建了内蕴的稠密联想记忆，证明曲率从根本上决定记忆的存亡——正曲率可抹除记忆而负曲率会强化记忆，并给出了容量随核重叠概率的标度律，同时揭示了模式重叠能够涌现出新记忆模式的机制。 |
| [^53] | [FluxLite: Inference-Time Proposal Control for Discrete Diffusion Models](https://arxiv.org/abs/2609.35947) | FluxLite 提出一个无需训练的轻量级提议控制框架，通过在 Feynman-Kac 势中加入 q_t 加权的图散度项精确补偿稀疏跳跃速率扰动，避免 SMC 权重退化，并实例化为单跳局部重分配（HEU）与非负二次修正两种实用采样器。 |
| [^54] | [Wasserstein Causal Forests for Distribution-Valued Outcomes](https://arxiv.org/abs/2609.35898) | 本文提出Wasserstein因果森林（WCF）用于处理结果为概率分布的因果推断问题，定义了包含参考距离对比的分布型处理效应，在多数模拟设计中条件分布估计最为准确，并应用于Project STAR项目揭示小班教学对成绩分布（而不仅是均值）的影响。 |
| [^55] | [Learning from the Gap Between Pass@K and Pass@1](https://arxiv.org/abs/2609.35793) | 提出 GapFT 方法，通过在 Pass@K 与 Pass@1 的差距（即单样本失败但 K 个样本内可解决的问题）上进行微调，将测试时搜索带来的能力吸收进模型，从而提升单样本解码的性能。 |
| [^56] | [Serverless gossip training of LSTM failure detectors: A matched-protocol comparison with federated, local and centralized learning on NASA C-MAPSS](https://arxiv.org/abs/2609.35792) | 该论文首次在严格匹配协议下量化了无服务器Gossip训练用于LSTM故障检测的效果，证明其在NASA C-MAPSS上可达到与需要中央服务器的联邦学习（FedAvg）几乎相同的F1性能，并明显优于本地独立训练。 |
| [^57] | [Quantifying Behavioral Tails in Black-Box Language Models](https://arxiv.org/abs/2609.33638) | RareTrap框架通过代理LLM构建几何感知映射来诱导可复现的提示词分布，并结合序列稀有事件模拟技术，有效估计了黑盒大语言模型发生严重行为的概率。 |
| [^58] | [Which Self-Improvements Should We Trust? Reliable Self-Improvement When Agents Reuse Their Benchmarks](https://arxiv.org/abs/2609.33180) | 提出REUSE框架，通过认证式风险控制评估解决递归自我改进中智能体反复重用固定基准测试所导致的自适应过拟合问题，确保经验改进能真实反映任务分布上的群体性提升。 |
| [^59] | [Byzantine-Robust Federated RAG via Aligned Calibration and Fixed-Membership Conformal Prediction](https://arxiv.org/abs/2609.33037) | 该论文提出一种对拜占庭节点鲁棒的联邦检索增强生成方法，利用“校准与查询两个阶段中诚实节点相同”这一关键观察，通过对齐校准与固定成员保形预测，在部分节点于两个阶段均可能虚假上报评分的情况下，仍能以预设概率保证返回的答案集合包含正确答案。 |
| [^60] | [Bayesian Deck-of-cards-based Ordinal Regression with Sequential Preference Elicitation](https://arxiv.org/abs/2609.23212) | 该论文提出B-DOR，将基于纸牌的序数回归概率化为贝叶斯框架，通过累积链接似然将空白纸牌数量与潜在价值差异关联，并提供哈密顿蒙特卡洛采样和约束凸优化两种推断算法以支持序贯偏好引出。 |
| [^61] | [Exact Regret Frontiers and Externality Scheduling in Centralized Serial-Dictatorship Bandits](https://arxiv.org/abs/2609.19963) | 该论文精确刻画了集中式序列独裁匹配老虎机中的可达对数遗憾前沿：匹配级 Graves-Lai 约束可归结为有限个成对探索配额并可通过多项式规模的线性规划求解，同时揭示相同的探索配额经不同调度会产生截然不同的遗憾。 |
| [^62] | [Performative Privacy: When Differential Privacy Maximizes Utility](https://arxiv.org/abs/2608.28198) | 该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。 |
| [^63] | [Gromov-Monge Flow Matching for Equivariant Graph Generation](https://arxiv.org/abs/2608.26961) | 本文提出了一种基于Gromov-Monge距离的流匹配方法，通过商空间几何和等变架构实现高效的图生成，并利用松弛和下界解决对齐难题。 |
| [^64] | [Comparing Corrupted Constrained Learning Problems](https://arxiv.org/abs/2608.25745) | 本文发现经典数据处理不等式在约束学习问题中失效，并提出广义版本以修正这一局限。 |
| [^65] | [Characterizing Full Nonequilibrium Dynamics of Simple Exclusion Processes](https://arxiv.org/abs/2608.25606) | 本文利用变分自回归网络系统表征了从一维到三维的简单排斥过程的非平衡动力学，首次提供了三维有限时间分析，并揭示了一维有限时间动力学活性与TASEP三相稳态组织之间的直接对应关系。 |
| [^66] | [Tight Nonasymptotic Local Convergence of Sinkhorn-Knopp](https://arxiv.org/abs/2608.11760) | 本文首次提供了Sinkhorn-Knopp算法的非渐近局部收敛分析，证明了其在特定条件下为多项式时间算法，并显著提升了稠密矩阵缩放问题的复杂度上界。 |
| [^67] | [SR-OPSD: Self-Referenced On-Policy Self-Distillation](https://arxiv.org/abs/2608.09745) | 提出SR-OPSD方法，通过将自教师模型与冻结的初始策略构建归一化几何目标并最小化正向Rényi散度，实现了对在线策略自蒸馏中密集监督信号的精确可控调节。 |
| [^68] | [Theoretical Guarantees for SMC-Guided Diffusion Sampling](https://arxiv.org/abs/2607.04780) | 本文为SMC引导的扩散采样建立了首个非渐近误差理论保证，刻画了有限粒子涨落以及扩散模型、数值实现与引导机制中各类局部误差经前向平滑核传播的规律。 |
| [^69] | [A functional central limit theorem for kernel gradient flow and infinitesimal gradient boosting](https://arxiv.org/abs/2606.25494) | 本文在与softmax梯度树基学习器相关的再生核希尔伯特空间中，借助巴拿赫空间上ODE的一般随机扰动分析，为无穷小梯度提升和核梯度流建立了函数中心极限定理，证明重缩放偏差依分布收敛于高斯过程。 |
| [^70] | [Bentkus-type asymptotic e-values](https://arxiv.org/abs/2606.06332) | 本文借助Bentkus的近最优集中不等式框架提出了Bentkus型渐近e值，成功消除了现有渐近e值中的“缺失因子”，从而在事后推断和多重检验中实现了比现有方法更锐利的推断。 |
| [^71] | [Reasoning with Sampling: Cutting at Decision Points](https://arxiv.org/abs/2605.30327) | 该研究表明从基础模型的幂分布中采样即可达到媲美强化学习训练的推理能力，并提出应在推理轨迹中的关键决策点处进行切割重采样，以实现高效的混合采样。 |
| [^72] | [Certified Adaptive Refresh: Anytime-Valid Monitoring for Federated Conformal RAG](https://arxiv.org/abs/2605.29139) | 提出Anytime-FC-RAG框架，通过三条可审计规则，使联邦保形RAG系统在持续检查与反复模型升级的情况下仍能实现任意时刻有效、经过认证的漏报率监控。 |
| [^73] | [Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](https://arxiv.org/abs/2605.21648) | 该论文建立了混沌边缘处 dropout 的平均场理论，揭示出不同激活函数的普适类与标度律，并证明将 dropout 随深度调度且集中于靠近输入层的位置能最大化正则化收益，在 MLP 等模型上表现最为一致。 |
| [^74] | [Optimization Risk Bounds for Kolmogorov-Arnold Networks Trained by DP-SGD with Correlated Noise](https://arxiv.org/abs/2605.12648) | 本文首次为使用时间相关噪声DP-SGD训练的两层Kolmogorov-Arnold网络（KAN）建立了优化风险界，并显式刻画了其对时间相关性、裁剪、小批量采样和网络宽度的依赖关系。 |
| [^75] | [Empirical Bayes 1-bit matrix completion](https://arxiv.org/abs/2605.09509) | 本文提出了一种受Efron-Morris估计量启发的经验贝叶斯1比特矩阵补全方法，该方法利用二值矩阵的低秩结构并将奇异值向零收缩，实现了具有竞争力的预测精度和良好的预测校准性。 |
| [^76] | [Non-Myopic Active Feature Acquisition via Pathwise Policy Gradients](https://arxiv.org/abs/2605.05511) | 该论文提出对特征获取过程进行连续松弛并结合直通式前向模拟，实现了贯穿完整获取轨迹的非短视路径策略梯度，从而对主动特征获取策略进行低方差的端到端优化。 |
| [^77] | [Amortized Optimal Transport from Sliced Potentials](https://arxiv.org/abs/2604.15114) | 本文提出基于切片最优传输势能的两种摊销方法（回归式RA-OT和目标式OA-OT），可高效解决多对测度之间的重复最优传输问题。 |
| [^78] | [PAC-CF: Calibrating Irreversible Frontier Pruning in LLM-Guided Search](https://arxiv.org/abs/2604.14345) | 提出PAC-CF方法，将LLM引导搜索中的前沿剪枝形式化为具有PAC保证的保形决策问题，通过Native-Trace路径校准得到保形边际来过滤候选，避免因不可约评估偏差误删所有可通向有效解的分支，从而在多种领域和预算下提升搜索效用。 |
| [^79] | [Query Lower Bounds for Diffusion Sampling](https://arxiv.org/abs/2604.10857) | 本文首次建立了扩散采样的分数查询下界，证明在多项式精度分数估计下任何采样算法至少需要 $\widetilde{\Omega}(\sqrt{d})$ 次自适应分数查询，从而从信息论角度正式解释了实践中多尺度噪声调度不可或缺的原因。 |
| [^80] | [Learning to Recorrupt: Noise Distribution Agnostic Self-Supervised Image Denoising](https://arxiv.org/abs/2603.25869) | 提出了一种自监督去噪框架 L2R，通过可学习的再污染器与去噪器以极小极大鞍点目标联合优化，无需精确了解噪声分布即可在多种非常规和重尾噪声分布下实现最先进的去噪性能。 |
| [^81] | [Cross-Fitting-Free Debiased Machine Learning with Multiway Dependence](https://arxiv.org/abs/2602.11333) | 本文提出了一种无需交叉拟合的去偏机器学习方法，通过结合Neyman正交矩条件和局部化经验过程，在多重聚类依赖下实现有效的渐近推断。 |
| [^82] | [From Seeds to Semantics: Measuring Semantic Accessibility in Deterministic Diffusion Models](https://arxiv.org/abs/2602.06155) | 该论文提出“语义可达性”这一概念，通过在DDIM确定性采样轨迹的多个位置训练探针分类器，量化最终图像的语义信息（如类别标签和属性）能从初始噪声种子和中间状态中被提取预测的程度。 |
| [^83] | [On the Nonasymptotic Scaling Guarantee of Hyperparameter Estimation in Inhomogeneous, Weakly-Dependent Complex Network Dynamical Systems](https://arxiv.org/abs/2601.15603) | 本文从测度输运视角提出了基于均值型观测的超参数估计理论框架，并为非齐次、弱依赖复杂网络动力系统建立了超参数估计偏差关于网络规模大小的非渐近界，证明在固定观测时长和通用优化算法下估计随网络增大依然可靠。 |
| [^84] | [Persistent Tri-State Message Passing](https://arxiv.org/abs/2601.01207) | 本文提出持久三状态消息传递（P3MP），理论分析了边角色跨层持久性对权重平均的影响，推导出局部权重与共享权重预激活排序发生逆转的条件，并证明共享权重仅保留1/K的目标状态反馈。 |
| [^85] | [High-Dimensional Partial Least Squares: Spectral Analysis and Fundamental Limitations](https://arxiv.org/abs/2512.15684) | 本文利用随机矩阵理论对高维偏最小二乘法（PLS-SVD）进行谱分析，首次定量刻画了估计潜在方向与真实方向的对齐程度，从而解释了该方法的重构性能并揭示了其表现反直觉或失效的基本局限。 |
| [^86] | [Fooling Algorithms in Non-Stationary Bandits using Belief Inertia](https://arxiv.org/abs/2511.05620) | 本文利用“信念惯性”机制构造确定性对抗实例，对滑动窗口UCB算法给出了最坏情况动态遗憾的下界，证明即使是专门设计用于遗忘过时观测的算法，也会被非平稳环境的变点欺骗而遭受显著遗憾。 |
| [^87] | [CPATTA: Conformal Supervision Allocation For Active Test-Time Adaptation](https://arxiv.org/abs/2509.25692) | 该论文提出CPATTA，首次将带覆盖率感知在线校准的保形预测不确定性引入主动测试时自适应，通过平滑保形分数、伪覆盖率驱动的在线权重更新、领域偏移检测和分阶段更新方案，使准确率持续超越现有最先进方法约5%。 |
| [^88] | [Data-Efficient Time-Dependent PDE Surrogates: Graph Neural Simulators vs. Neural Operators](https://arxiv.org/abs/2509.06154) | 提出图神经模拟器（GNS），通过消息传递结合数值时间步进格式建模瞬时时间导数，克服神经算子依赖大数据的缺陷，实现时间依赖PDE的数据高效代理建模。 |
| [^89] | [Cryo-EM as a Stochastic Inverse Problem](https://arxiv.org/abs/2509.05541) | 本文将冷冻电镜三维重建创新性地表述为概率测度空间上的随机逆问题，通过最小化观测与模拟图像分布间的统计距离（KL散度、最大均值差异），并借助Wasserstein梯度流的粒子数值方法求解，从而突破了传统离散构象假设、实现了连续结构变化的恢复。 |
| [^90] | [Quantum Geometry of Data](https://arxiv.org/abs/2507.21135) | 本文首次完整建立了量子认知机器学习（QCML）的矩阵几何框架及其数学基础，将数据编码为希尔伯特空间中的量子几何，直接从数据导出内蕴维度、量子度规与贝里曲率等几何拓扑结构，并提出矩阵拉普拉斯算子作为避免维度灾难、保持几何性质的图嵌入替代方法。 |
| [^91] | [Causal pieces: analysing and improving spiking neural networks piece by piece](https://arxiv.org/abs/2504.14015) | 提出“因果片段”新概念来分析脉冲神经网络，证明片段内输出脉冲时间对输入和参数局部Lipschitz连续，且因果片段数量可作为衡量SNN逼近能力的有效指标。 |
| [^92] | [Estimating the Causal Effects of T Cell Receptors](https://arxiv.org/abs/2410.14127) | 该论文提出一种利用V(D)J重组生成的预选择TCR库作为自然实验来校正未观察混杂因素的方法，结合半参数层次因果模型与置换不变神经网络，从观察性TCR测序数据中推断T细胞受体序列对患者疾病预后的因果效应。 |

# 详细

[^1]: ReCIRC：校正的共形风险控制

    ReCIRC: Rectified Conformal Risk Control

    [https://arxiv.org/abs/2609.38112](https://arxiv.org/abs/2609.38112)

    ReCIRC通过将每个输入的估计局部风险曲线求逆，把校准阈值重新参数化为代表共同目标条件风险的风险预算，在无论曲线估计准确与否都保持有限样本边际保证的同时，实现近似的条件风险控制，解决了传统共形风险控制对简单样本保护过度、对困难样本保护不足的问题。

    

    许多黑箱预测模型的应用需要控制与任务相关的错误率，例如图像分割中遗漏的病灶像素或多标签分类中遗漏的标签。共形风险控制（CRC；Angelopoulos et al., arXiv:2208.02814）为此类损失提供了无分布保证，但它校准的是所有输入共享的单一阈值。由于条件风险随输入而变化，这种边际保证往往对简单样本保护过度，而对困难样本保护不足。我们提出ReCIRC（校正共形风险控制），它通过对每个输入估计的局部风险曲线求逆，将校准后的阈值重新参数化为代表共同目标条件风险的风险预算 $a$，然后对由此得到的阈值族不加修改地应用CRC。无论估计曲线的准确性如何，ReCIRC都能保持CRC的有限样本边际保证，而当曲线估计准确时，还可实现近似的条件风险控制……

    arXiv:2609.38112v1 Announce Type: cross  Abstract: Many applications of black-box predictive models require controlling task-relevant error rates, such as missed lesion pixels in segmentation or missed labels in multilabel classification. Conformal risk control (CRC; Angelopoulos et al., arXiv:2208.02814) gives distribution-free guarantees for such losses, but it calibrates a single threshold shared by all inputs. Because conditional risk varies with the input, this marginal guarantee often overprotects easy cases and underprotects hard ones. We propose ReCIRC (Rectified Conformal Risk Control), which inverts each input's estimated local risk curve to reparameterize the calibrated threshold as a risk budget $a$ representing a common target conditional risk, and then applies CRC unchanged to the resulting family. ReCIRC retains CRC's finite-sample marginal guarantee regardless of the accuracy of the estimated curves, while accurate curves yield approximate conditional risk control and, 
    
[^2]: 基于Hessian零空间延拓法遍历神经网络的解空间

    Traversing the solution space of neural networks with Hessian Null Space Continuation

    [https://arxiv.org/abs/2609.38081](https://arxiv.org/abs/2609.38081)

    本文提出Hessian零空间延拓（HNC）方法，利用局部曲率遍历权重空间中保持网络功能不变的区域，首次揭示了局部模式连通区域内存在多种不同的内部计算机制。

    

    在单个任务上，深度网络可以学习到许多不同的解，这取决于其优化器、训练数据、架构和超参数。其中许多解是模式连通的：它们在权重空间中并非孤立的点，而是由低损失区域相互连接。然而，在这些区域内网络的内部计算如何变化尚不清楚。另一条平行的研究路线揭示了神经表征的退化性：许多网络能够以不同的内部结构达到相似的训练损失。然而，这些解在权重空间中是如何相互关联的仍不明确。我们统一了这两个子领域，并首次证明在权重空间的一个局部模式连通区域内存在许多不同的内部机制。为此，我们提出了Hessian零空间延拓（HNC），这是一种可扩展的方法，利用局部曲率来遍历权重空间中保持网络功能不变的区域，并且可以被引导至具有特定属性的解。

    arXiv:2609.38081v1 Announce Type: new  Abstract: On a single task, deep networks can learn many solutions, depending on their optimizer, training data, architecture, and hyperparameters. Many of these solutions are mode-connected: rather than isolated points in weight space, they are connected by low-loss regions. Yet how their internal computation varies within these regions is unknown. A parallel line of work has identified the degeneracy of neural representations: many networks reach similar training loss with distinct internal structures. However, it is unclear how these solutions are related in weight space. We unify these subfields and show for the first time that many different internal mechanisms exist within a local mode-connected region in weight space. To do so, we introduce Hessian Null Space Continuation (HNC), a scalable method that uses local curvature to traverse regions of weight space that preserve network function, and can be steered toward solutions with specified p
    
[^3]: 时间序列基础模型的潜空间推理时引导

    Latent Inference-Time Guidance of Time Series Foundation Models

    [https://arxiv.org/abs/2609.38058](https://arxiv.org/abs/2609.38058)

    本文提出潜空间推理时引导方法，通过具有独立分量的时间相关潜空间自适应集成多个时间序列基础模型的预测，在保持开箱即用特性的同时提供了可识别性与重构保证。

    

    时间序列基础模型（TSFMs）目前在预测任务中提供了最先进的结果。它们开箱即用，并依靠上下文学习进行预测，这使得其性能质量对用户选择的回看窗口、协变量、预测范围和训练数据分布高度敏感。在实践中，这些预测的质量参差不齐但具有互补性，这凸显了需要一种有原则的集成方法，而不是简单地选择最佳上下文。本文提出了针对TSFMs的潜空间推理时引导方法，通过一个具有独立分量的时间相关潜空间，自适应地组合一组TSFM预测。该框架具备可识别性和重构保证，同时保持了基础模型开箱即用的特性。我们在多种频率、多个领域的数据集上进行了实验：这些实验表明……

    arXiv:2609.38058v1 Announce Type: cross  Abstract: Time Series Foundation Models (TSFMs) currently provide state-of-the-art results in forecasting tasks. They are available out-of-the-box and rely on in-context learning to make their predictions, which makes the quality of their performance highly sensitive to the user-selected lookback, covariates, horizon and training data distributions. In practise, the quality of the forecasts are variable but complementary, which highlights the need for a principled ensembling approach, rather than selecting the best context. This paper introduces Latent Inference-Time Guidance for TSFMs, which adaptively combines a pool of TSFM forecasts through a time-dependent latent space with independent components. The framework comes equipped with identifiability and reconstruction guarantees, whilst maintaining the off-the-shelf aspect of foundation models. We provide experiments on datasets at various frequencies and from multiple domains: these show that
    
[^4]: 数据混合何时能改进缩放定律？来自高维回归的洞见

    When do data mixtures improve scaling laws? Insights from high-dimensional regression

    [https://arxiv.org/abs/2609.38011](https://arxiv.org/abs/2609.38011)

    该论文通过构建共享回归函数、异构协方差与噪声水平的高维混合数据回归模型，建立了极小极大风险界并推导出岭回归测试误差的确定性等价，从理论上刻画了辅助数据何时能真正改进缩放定律而非仅仅提供更多样本。

    

    现代机器学习系统在不同领域的数据混合上进行训练，选择合适的数据混合比例可以显著提升下游性能。尽管已有大量关于数据混合与重新加权的文献，但现有工作大多停留在经验层面，尚不清楚辅助数据何时能真正改进缩放定律，而不仅仅是提供更多的样本。为了深入理解这一问题，我们研究了一个高维混合数据回归模型，该模型具有共享的回归函数、异构的协方差和噪声水平，以及可能以不同速率增长的数据集规模。我们在一般协方差结构下、椭球参数约束下建立了极小极大风险，并在可交换协方差条件下推导出岭回归测试误差的确定性等价。随后，我们专门研究了具有对齐幂律协方差谱的目标域和辅助域，该理论给出……（摘要原文在此处截断）

    arXiv:2609.38011v1 Announce Type: new  Abstract: Modern machine learning systems are trained on mixtures of data from different domains, and choosing the right mixture can substantially improve downstream performance. Despite an extensive literature on data mixing and reweighting, existing work is largely empirical and it remains unclear when auxiliary data genuinely improves scaling laws rather than merely providing more samples. To gain insight into this question, we study a high-dimensional mixed-data regression model with a shared regression function, heterogeneous covariances and noise levels, and dataset sizes that may grow at different rates. We establish the minimax risk under an ellipsoidal parameter constraint for the general covariance structure and derive deterministic equivalents for the test error of ridge regression under commutative covariances. We then specialize to a target domain and an auxiliary domain with aligned power-law covariance spectra, where the theory yiel
    
[^5]: 互信息约束下的切尔诺夫瓶颈

    Mutual Information Constrained Chernoff Bottleneck

    [https://arxiv.org/abs/2609.37994](https://arxiv.org/abs/2609.37994)

    本文提出互信息约束下的Chernoff瓶颈问题，刻画了最优误差指数 $C(R)$ 随速率 $R$ 的变化规律（严格递增至 $H(V)$ 后保持不变、且不必为凹函数），证明 $k+1$ 个输出足以达到最优值，并给出了交替优化算法。

    

    经典的信息瓶颈（IB）通过 $I(U;Y)$ 来衡量 $X$ 的表示 $U$ 与目标 $Y$ 之间的相关性，但这并不能直接刻画下游决策的误差。对于一个由多路分别编码的观测所推断的二元假设 $Y$，其最优误差指数是给定 $Y$ 时 $U$ 的两个条件分布之间的 Chernoff 信息。我们研究了互信息约束下的 Chernoff 瓶颈问题，即在速率约束 $I(U;X) \leq R$ 下寻找使该 Chernoff 信息最大化的编码器。我们证明其最优值 $C(R)$ 在 $R = H(V)$ 之前严格递增，其中 $V$ 是将 $X$ 中具有相同似然比的符号合并后得到的变量；超过该点后，$C(R)$ 保持在未压缩时的误差指数水平；而且与 IB 曲线不同，$C(R)$ 不一定是凹的。我们进一步证明 $k+1$ 个输出足以达到 $C(R)$，其中 $k$ 是 $V$ 的基数。我们提出了一种交替算法来更新……

    arXiv:2609.37994v1 Announce Type: cross  Abstract: The classical information bottleneck (IB) measures the relevance of a representation $U$ of $X$ to a target $Y$ by $I(U;Y)$, which does not directly characterize the error of downstream decisions. For a binary hypothesis $Y$ inferred from many separately encoded observations, the optimal error exponent is the Chernoff information between the two conditional distributions of $U$ given $Y$. We study the mutual information constrained Chernoff bottleneck, which seeks an encoder that maximizes this Chernoff information subject to a rate constraint $I(U;X) \leq R$. We show that its optimal value $C(R)$ increases strictly up to $R = H(V)$, where $V$ merges the symbols of $X$ with equal likelihood ratio, remains at the uncompressed exponent beyond, and, unlike the IB curve, need not be concave. We further show that $k+1$ outputs suffice to attain $C(R)$, where $k$ is the cardinality of $V$. We propose an alternating algorithm that updates the
    
[^6]: 流体界面的形状相似性潜在空间：液滴形态的可逆降阶建模

    A shape-similarity latent space for fluid interfaces: invertible reduced-order modelling of droplet morphology

    [https://arxiv.org/abs/2609.37947](https://arxiv.org/abs/2609.37947)

    本文提出 SHROM 框架，通过结合平方根速度函数形状表示、形状空间邻居图和自编码器，构建了一个低维、可逆且忠实于形状相似性的潜在空间，实现了液滴形态的可逆降阶建模，从而从稀疏的实验记录中恢复单次拍摄所遗漏的液滴中间形态状态。

    

    arXiv:2609.37947v1 公告类型：cross。摘要：液滴在数十微秒内破裂，而一次拍摄大约只能捕获十几帧。中间状态若不重复实验便无法恢复，而模拟它们的成本过高，难以遍历整个操作范围。然而，这些状态存在于整个实验数据集中：一项跨越设备驱动范围的实验活动所产生的形态，与任何单次记录所遗漏的形态相似。要利用这一点，需要一个低维、可逆且忠实于形状（而非采样坐标）的表示。本征正交分解（POD）提供了前两个性质，但它在采样坐标中度量距离；流形学习满足第三个性质，却缺乏映射回形状的手段；弹性形状分析提供了形状度量，却没有降阶坐标。SHROM 将这三者结合在一起：界面由其平方根速度函数表示，在该形状空间上构建邻居图，并训练自编码器以重建……（摘要在此处截断）

    arXiv:2609.37947v1 Announce Type: cross  Abstract: A droplet breaks up in tens of microseconds, and a recording captures perhaps a dozen frames. The states in between cannot be recovered without repeating the experiment, and simulating them is too costly to sweep an operating envelope. Yet they are present in the corpus as a whole: a campaign spanning a device's actuation range produces morphologies resembling those any single recording missed. Exploiting that requires a representation that is low-dimensional, invertible, and faithful to shape rather than to sampling. Proper orthogonal decomposition supplies the first two but measures distance in sampled coordinates; manifold learning supplies the third but no map back to a shape; elastic shape analysis supplies a shape metric but no reduced coordinates. SHROM composes all three. Interfaces are represented by their square-root velocity functions, a neighbour graph is built over that shape space, and an autoencoder is trained to reconst
    
[^7]: 时滞物理系统驱动项与动力学的可辨识性保证

    Identifiability Guarantees for Drivers and Dynamics of Delayed Physical Systems

    [https://arxiv.org/abs/2609.37944](https://arxiv.org/abs/2609.37944)

    本文提出一种有理论支撑的方法，证明在宽松假设下随机时滞微分方程的结构驱动项与漂移项是可辨识的，并在驱动项可辨识性与动力学物理一致性基准上优于现有方法。

    

    目前已有大量方法被提出，包括物理信息神经网络（功能强大但不保证动力学的可辨识性）、符号回归（需要一组预先计算好的操作）以及因果发现（更具原则性，但通常依赖于物理系统可能违反的强假设）。在本工作中，我们开发了一种有理论支撑的方法，并证明在一组宽松的假设下，随机时滞微分方程的结构驱动项和漂移项是可辨识的。我们的方法在驱动项可辨识性基准测试中优于其他方法，并在第二个用于评估所学动力学物理一致性的基准测试中也表现更佳。

    arXiv:2609.37944v1 Announce Type: cross  Abstract: A wide range of methods have been proposed, including physics-informed neural networks, which are powerful but do not guarantee identifiability of the dynamics, symbolic regression, which requires a set of precomputed operations, and causal discovery, which is more principled but usually relies on strong assumptions that physical systems may violate. In this work, we develop a theory-grounded method and prove that under a set of permissive assumptions, the structural drivers and drift of stochastic delayed differential equations are identifiable. Our method outperforms others on a benchmark for driver identifiability, and on a second benchmark to evaluate physical consistency of the learned dynamics.
    
[^8]: 针对Deep SVDD的异常检测后推断

    Post-Anomaly Detection Inference for Deep SVDD

    [https://arxiv.org/abs/2609.37935](https://arxiv.org/abs/2609.37935)

    本文提出PADI框架，利用选择性推断方法为冻结的Deep SVDD异常检测器的异常判定提供条件于“被识别为异常”事件下的严格统计有效性推断，解决了异常决策缺乏统计保证的问题。

    

    深度支持向量数据描述已成为一种主流的无监督异常检测框架，它通过学习将正常数据紧凑地围绕一个中心进行表征的潜在表示来实现检测。尽管其在实践中取得了成功，但Deep SVDD产生的异常决策通常仅基于异常分数作出，缺乏严格的统计保证，从而限制了其在必须严格控制误报的安全关键和高风险应用中的可靠性。在本文中，我们提出了PADI（异常检测后推断），这是一个新颖的框架，通过利用选择性推断框架，为已训练并冻结的Deep SVDD检测器配备具有统计有效性的推断。具体而言，PADI在测试实例被Deep SVDD识别为异常这一事件的条件下进行推断，从而能够对异常决策进行严格的统计评估。基于这一构建，

    arXiv:2609.37935v1 Announce Type: cross  Abstract: Deep Support Vector Data Description (Deep SVDD) has become a prominent framework for unsupervised anomaly detection by learning latent representations that compactly characterize normal data around a center. Despite its empirical success, anomaly decisions produced by Deep SVDD are typically made solely based on anomaly scores without rigorous statistical guarantees, thereby limiting their reliability in safety-critical and high-stakes applications where false positives must be strictly controlled. In this paper, we propose PADI (Post-Anomaly Detection Inference), a novel framework that equips a trained and frozen Deep SVDD detector with statistically valid inference by leveraging the Selective Inference framework. Specifically, PADI performs inference conditional on the event that a test instance is identified as anomalous by Deep SVDD, thereby enabling rigorous statistical assessment of anomaly decisions. Based on this formulation, 
    
[^9]: 无标签投影追踪中的搜索维度：子空间限制的标度律

    Search Dimension in Unlabeled Projection Pursuit: A Scaling Law for Subspace Restriction

    [https://arxiv.org/abs/2609.37917](https://arxiv.org/abs/2609.37917)

    该论文证明在无标签投影追踪中搜索维度本身是关键的统计量：过大的高斯补空间会让无信号方向最小化经验目标，而将搜索限制在与算子判别增益相匹配的子空间可把所需样本量的标度从 ς⁻⁸ 改善到 ς⁻⁴。

    

    投影追踪寻找使数据看起来最偏离高斯分布的方向。当观测空间中存在较大的高斯补空间时，经验目标函数可能被一个完全不携带信号的方向最小化，其经验峰度可低至与真实方向相当。样本拆分只会暴露而非修复这一失败。附加与潜在机制独立的坐标会使搜索退化，但贝叶斯可恢复性保持不变。将搜索限制在已知前向算子的列空间内，可以在负峰度分支上精确地消除这一失败。从数据中估计主子空间则是另一种替代途径。在一个受控的两分量模型中，主导的充分标度取决于算子传递判别量的增益：协方差谱峰估计为 ς⁻⁴，而四阶矩搜索为 ς⁻⁸。在固定搜索维度下，实测的阈值比率与理论预测一致（原文在此处截断）。

    arXiv:2609.37917v1 Announce Type: new  Abstract: Projection pursuit searches for a direction along which the data look least Gaussian. When the observation space contains a large Gaussian complement, the empirical objective can be minimized by a direction that carries no signal, with empirical kurtosis as low as at the truth. Sample splitting exposes rather than repairs this failure. Appending coordinates independent of the latent regime degrades the search while leaving Bayes recoverability unchanged. Restricting the search to the column space of a known forward operator removes the failure exactly on the negative-kurtosis branch. Estimating a principal subspace from the data is the alternative. In a controlled two-component model, the leading sufficient scalings differ in the gain with which the operator transmits the discriminant: $\varsigma^{-4}$ for covariance-spike estimation and $\varsigma^{-8}$ for fourth-moment search. At fixed search dimension, the measured threshold ratio co
    
[^10]: 基于隐藏森林结构的反事实探测并行去掩码方法

    Counterfactual Probing for Parallel Unmasking with Hidden Forest Structure

    [https://arxiv.org/abs/2609.37841](https://arxiv.org/abs/2609.37841)

    该论文提出一种反事实探测方法，在依赖关系未知的隐藏森林结构下实现并行去掩码采样，使得包括依赖发现在内的总模型评估次数和串行深度均相对序列长度 $N$ 达到亚线性复杂度，同时保证有界的采样误差。

    

    掩码生成模型提供了并行的词元预测能力，但准确的并行采样必须考虑词元之间的依赖关系。当依赖关系未知时，寻找安全的并行批次同样需要消耗模型评估次数。我们研究总的评估次数（包括依赖发现过程在内）能否在序列长度 $N$ 上达到亚线性，进而使串行深度也相应达到亚线性。我们考虑具有隐藏森林结构的离散分布，并通过固定的近似条件分布预言机对其进行访问。在明确的正则性条件和一致 Hellinger 误差界下，对于任意固定的目标精度 $\varepsilon\in (0,1/8]$ 以及足够大的 $N$，我们的采样器实现了按种子平均的总变差误差不超过 $\varepsilon$，且总掩码状态提交次数与串行深度均被 $O(N^C \varepsilon^a)$ 界定（其中常数 $0<C<1$ 与 $a>0$）。这些保证依赖于多项式规模的词表大小，以及由 $N$ 和 $\varepsilon$ 共同决定的边响应下界。

    arXiv:2609.37841v1 Announce Type: new  Abstract: Masked generative models offer parallel token prediction, but accurate parallel sampling must account for dependencies among tokens. When dependencies are unknown, finding safe batches also costs model evaluations. We study whether total evaluations, including discovery, can be sublinear in sequence length $N$; sublinear sequential depth then follows. We consider discrete distributions with hidden forest structure, accessed through a fixed approximate conditional oracle. Under explicit regularity conditions and uniform Hellinger error bounds, for any fixed target accuracy $\varepsilon\in (0,1/8]$ and sufficiently large $N$, our sampler achieves seed-averaged total-variation error at most $\varepsilon$, with total masked-state submissions and sequential depth both bounded by $O(N^C \varepsilon^a)$ for constants $0 <1$ and $a > 0$. These guarantees use polynomial vocabulary size and an edge-response lower bound set by $N$ and $\varepsilon$
    
[^11]: 当噪声估计掩盖重复贝叶斯逆问题中的基底误设时

    When Noise Estimation Hides Basis Misspecification in Repeated Bayesian Inverse Problems

    [https://arxiv.org/abs/2609.37762](https://arxiv.org/abs/2609.37762)

    本文揭示在贝叶斯逆问题中估计观测噪声方差会掩盖基底误设所导致的覆盖性损失，并提出利用John球形度统计量检验投影补空间中残差样本谱的形状，以在重复问题中检测这种隐藏的失效。

    

    在贝叶斯逆问题中，当真值包含基底之外的成分时，受基底约束的先验可能会失去覆盖性。我们证明，对观测噪声方差的估计可能会掩盖这种覆盖性损失。在线性前向模型下，当张成空间内的先验方差占主导时，最大似然噪声估计会吸收位于模型值域补空间中的基底外能量。此时，残差幅度与观测覆盖性检验仍接近名义水平，而场覆盖性却下降。我们研究共享同一前向算子和同一基底的重复问题，且该基底独立于被检验数据固定。在投影到补空间之后，并在给定拟合噪声尺度的条件下，每个精确检验都是对残差的无量纲方向的检验。我们用John球形度统计量检验残差样本谱的形状。在高斯噪声下，其零假设模型在有限样本量下是精确的，我们推导了其在零假设下的均值以及在预设备择假设下的功效。

    arXiv:2609.37762v1 Announce Type: cross  Abstract: Basis-restricted priors in Bayesian inverse problems can lose coverage when the truth has components outside the basis. We show that estimating the observation-noise variance can hide this loss. Under a linear forward model, when the in-span prior variance dominates the noise, the maximum-likelihood noise estimate absorbs the out-of-basis energy in the complement of the model range. Residual-magnitude and observation-coverage checks then stay near nominal while field coverage falls. We study repeated problems sharing one forward operator and one basis, fixed independently of the tested data. After projection onto the complement, and conditionally on the fitted noise scale, every exact test is a test of the scale-free direction of the residuals. We test the shape of their sample spectrum with John's sphericity statistic. Under Gaussian noise its null model is exact at finite sample size, and we derive its null mean and its power at prop
    
[^12]: SPARK：一种基于残差投影的通用拟合优度评估方法

    SPARK: A General Goodness-of-Fit Assessment via Residual Projection

    [https://arxiv.org/abs/2609.37705](https://arxiv.org/abs/2609.37705)

    本文提出SPARK框架，通过去偏策略将初始拟合的残差投影到近似正交方向并采用基于核的投影方法提取残留信号，从而实现对传统统计模型和黑箱学习器的通用拟合优度检验。

    

    拟合优度检验是评估拟合程序是否已捕获协变量中所含系统性信息的基本工具。传统理论主要关注参数回归模型，而现代数据分析日益依赖灵活的黑箱学习器，其预测上的成功本身并不足以评估模型的准确性。本文提出了SPARK，一个通用的拟合优度检验框架，适用于传统统计模型和一般的黑箱学习程序、连续型和二元响应变量，以及低维和高维预测变量。基于去偏策略，该方法将初始学习程序拟合所得的残差投影到近似正交的方向上，以提取其中残留的信号。为了捕获所有投影方向上的信息，我们提出了一种基于核的投影方法，并建立了该方法的渐近性质和一致性理论。

    arXiv:2609.37705v1 Announce Type: cross  Abstract: Goodness-of-fit testing is a basic tool for assessing whether a fitted procedure has captured the systematic information contained in the covariates. While traditional theory has largely focused on parametric regression models, modern data analysis increasingly relies on flexible black-box learners, whose predictive success alone is insufficient to assess model accuracy. In this paper, we propose SPARK, a general framework for goodness-of-fit testing that applies to traditional statistical models and general black-box learning procedures, continuous and binary responses, and low- and high-dimensional predictors. Based on a debiasing strategy, the residuals from an initial fit of a learning procedure are projected onto nearly orthogonal directions to extract any remaining signal. To capture information across all projection directions, we propose a kernel-based projection method and establish both its asymptotic properties and the consi
    
[^13]: 过参数化多项式插值中的双重下降与龙格现象

    The double descent and Runge phenomena in overparametrized polynomial interpolation

    [https://arxiv.org/abs/2609.37657](https://arxiv.org/abs/2609.37657)

    该论文系统研究了单项式、切比雪夫和勒让德三种多项式基下具有最小范数系数的过参数化多项式插值，揭示了龙格现象与机器学习中双重下降现象之间的联系。

    

    多项式插值中的龙格现象通常被认为是机器学习中双重下降现象的经典类比。在本短文中，我们探讨了三种常用多项式基（单项式基、切比雪夫基和勒让德基）下的过参数化多项式插值，其中系数取 $\ell^2$ 范数最小（对于单项式基，还考虑了 $\ell^1$ 范数最小的系数）。我们的结果主要针对等距数据点和切比雪夫数据点给出，但许多结果并不依赖于采样的具体形式。

    arXiv:2609.37657v1 Announce Type: cross  Abstract: The Runge phenomenon in polynomial interpolation is often considered a classical analogue of the double descent phenomenon in machine learning. In this note, we explore overparameterized polynomial interpolation in three popular polynomial bases: Monomial, Chebyshev and Legendre basis with coefficients that are minimal in the $\ell^2$-norm (and, for the monomial basis, also those minimal in the $\ell^1$-norm). We present our results primarily for equidistant and Chebyshev data points, but many results are independent of the exact form of sampling.
    
[^14]: 一种用于嵌入有向数据的芬斯勒方法

    A Finslerian Approach for Embedding Directed Data

    [https://arxiv.org/abs/2609.37649](https://arxiv.org/abs/2609.37649)

    该论文提出将有向数据建模为距离依赖方向的芬斯勒流形上的样本，并证明由这种非对称距离构造的核算子可分解为收敛于加权拉普拉斯算子的对称部分（恢复扩散映射）和收敛于一阶传输算子的反对称部分，从而将有向数据的几何结构与方向信息分离开来。

    

    许多数据集具有内在的方向性：引用总是指向更早的文献，细胞沿谱系分化，交通沿偏好的路线流动。谱嵌入方法，包括大多数针对有向图的扩展，都丢弃了这一信息：它们将数据对称化并映射到无法表示不对称性的欧几里得空间中。我们转而将有向数据建模为从芬斯勒流形上采样得到，该流形的距离依赖于行进方向，并研究由这种非对称距离构建的核算子。通过对该算子的矩展开，我们证明其对称部分与反对称部分将几何结构与方向性分离开来。当核的带宽趋于零时，对称部分收敛到一个加权拉普拉斯算子，在黎曼情形下恢复了扩散映射；而反对称部分则收敛到一个编码方向性的一阶传输算子。我们证明……

    arXiv:2609.37649v1 Announce Type: cross  Abstract: Many datasets carry an intrinsic directionality: citations point backward in time, cells differentiate along lineages, and traffic follows preferred routes. Spectral embedding methods, including most of their extensions to directed graphs, discard this information: they symmetrize the data and map it into a Euclidean space where asymmetry cannot be represented. We instead model directed data as sampled from a Finsler manifold, whose distance depends on the direction of travel, and study the kernel operator built from this asymmetric distance. Through a moment expansion of this operator, we show that its symmetric and antisymmetric parts separate geometry from direction. As the bandwidth of the kernel vanishes, the symmetric part converges to a weighted Laplacian, recovering diffusion maps in the Riemannian case, while the antisymmetric part converges to a first-order transport operator that encodes the directionality. We prove that the
    
[^15]: RLTL;DR：通过内化自生成反馈实现自我提升

    RLTL;DR: Self-improvement by Internalizing Self-generated Feedback

    [https://arxiv.org/abs/2609.37633](https://arxiv.org/abs/2609.37633)

    提出RLTL;DR方法，让策略在每次失败后根据验证器输出撰写自己的TL;DR见解，并将其作为后续尝试的上下文条件，同时通过反向传播将这些见解内化为任务到见解的直接映射，从而在无教师模型、任务难度极高（Pass@128=0）的场景下实现自我提升。

    

    带可验证奖励的强化学习（RLVR）的常见范式是让智能体对任务进行多次尝试，并朝着成功的尝试方向进行优化。这在自我提升的场景中会出现问题：任务极其困难，智能体成功概率很低甚至为零，且没有教师模型或示例解答可供蒸馏。在本文中，我们提出了 RLTL;DR。在每次失败尝试后，我们向策略展示验证器的输出，并让它以一条 TL;DR（一句话见解）的形式撰写自己的反馈。下一次 rollout 会以所有先前的见解为条件，我们依次采样 rollout，直到找到解决方案。此外，我们对上下文中的见解启用反向传播，以将“任务到见解”的直接映射内化到模型中。在具有挑战性的工具调用和代码数据集上（过滤至 Pass@128=0），对 Qwen 3.5 9B Thinking 策略进行标准 GRPO 训练的表现始终持平……

    arXiv:2609.37633v1 Announce Type: cross  Abstract: The common paradigm of reinforcement learning with verifiable rewards (RLVR) is to let agents make multiple attempts at a task, and optimize towards the successful ones. This becomes problematic in the realms of self-improvement, where tasks are so difficult that the agent has a low or even no chance of success, and where there are no teacher models or example solutions to distill from. In this paper, we introduce RLTL;DR. After each failed attempt, we show the policy the verifier outputs and let it write its own feedback, in the form of a single TL;DR insight. The next rollout is conditioned on all previous insights, and we sequentially sample rollouts until a solution is found. Moreover, we enable backpropagation on the in-context insights to internalize a direct task to insight mapping. On challenging tool-calling and coding datasets (filtered to Pass@128=0), standard GRPO training of a Qwen 3.5 9B Thinking policy stays flat at a Pa
    
[^16]: XU-RS：解释随机集语言模型中的信度宽度

    XU-RS: Explaining Credal Width in Random-Set Language Models

    [https://arxiv.org/abs/2609.37594](https://arxiv.org/abs/2609.37594)

    提出XU-RS框架，利用期望梯度将随机集语言模型中答案的信度宽度（即认知不确定性）归因到输入词元，从而解释模型不确定性的来源。

    

    不确定性估计只能告诉我们模型有多不确定，却无法说明原因。如果不知道输入的哪些部分影响了模型的不确定性，我们就无法判断该不确定性分数是否依赖于与任务相关的输入特征。我们在基于预训练语言模型构建的随机集分类器中研究这一问题。这类分类器为单个答案以及答案组分配概率，从而为每个答案产生下概率和上概率；二者之间的差值被称为信度宽度，用于表示由训练数据有限所引发的关于某个答案的认知不确定性。我们提出了XU-RS，这是一个将答案的信度宽度归因到输入给语言模型的词元（单词或词片段）上的框架。XU-RS使用期望梯度（一种标准的特征归因方法）来估计输入词元对信度宽度的贡献。该框架在MedQA数据集上进行了评估。

    arXiv:2609.37594v1 Announce Type: new  Abstract: Uncertainty estimates tell us how unsure a model is, but not why. Without knowing which parts of an input influences a model's uncertainty, we cannot tell whether that uncertainty score depends on input features that are relevant for the task. We study this problem in randomset classifiers built using pretrained language models. These classifiers assign probability to individual answers and to groups of answers, producing lower and upper probabilities for each answer; The difference between these probabilities, called credal width, is used to represent epistemic uncertainty about an answer arising from limited training data. We propose XU-RS, a framework that attributes an answer's credal width to the input tokens (words or word pieces) supplied to a language model. XU-RS uses Expected Gradients (a standard feature attribution method) to estimate how input tokens contribute to credal width. The proposed framework is evaluated on a MedQA 
    
[^17]: 奥恩斯坦-乌伦贝克过程难以被超越，然而超线性漂移可带来更低的输运成本

    Ornstein-Uhlenbeck Is Hard to Beat, Yet Superlinear Drift Ships Lower Transport Costs

    [https://arxiv.org/abs/2609.37579](https://arxiv.org/abs/2609.37579)

    本文通过基于分数的图像生成实验证明，超线性朗之万扩散模型在经验Wasserstein距离上几乎全面优于奥恩斯坦-乌伦贝克基线且波动更小，表明OU扩散“难以被超越”的结论并非普遍成立。

    

    Brešar和Mijatović证明，在排除超线性漂移的假设条件下，奥恩斯坦-乌伦贝克（OU）扩散在前向收敛方面难以被超越。我们转而针对基于分数的图像生成测试超线性朗之万扩散，并通过Fokker-Planck方程数值计算其条件分数。在我们的实验中，超线性模型在经验Wasserstein距离上于几乎整个测试网格范围内击败了奥恩斯坦-乌伦贝克基线，且在不同的扩散时域上表现出更小的波动。因此，Brešar和Mijatović所得出的“OU扩散难以被超越”的结论并不具有普适性。

    arXiv:2609.37579v1 Announce Type: cross  Abstract: Bre\v{s}ar and Mijatovi\'c \cite{bresar2025} show that Ornstein--Uhlenbeck diffusion is hard to beat in forward convergence under assumptions that exclude superlinear drift. We instead test superlinear Langevin diffusions for score-based image generation, computing their conditional scores numerically from a Fokker--Planck equation. In our experiments, the superlinear models beat the Ornstein--Uhlenbeck baseline on empirical Wasserstein distance across nearly the entire tested grid and show less variation across diffusion horizons. The ``hard to beat'' verdict of \cite{bresar2025} thus fails to be universal.
    
[^18]: 为什么自适应优化器会低估稀有词元

    Why Adaptive Optimizers Underestimate Rare Tokens

    [https://arxiv.org/abs/2609.37535](https://arxiv.org/abs/2609.37535)

    本文揭示了Adam等逐坐标自适应优化器因将更新除以在词元出现后达到峰值的幅度运行估计，从而系统性地低估稀有词元，并精确刻画了哪些优化器（如SGD、Shampoo、Muon）能保持输出层平均嵌入、哪些（如Adam、Adafactor、Lion、符号下降）不能。

    

    在softmax输出层中，稀有词元在大多数训练步上会收到一个很小的正logit梯度，而在它作为目标词元的少数步上会收到一个大得多的负梯度。SGD只是简单地将这些贡献相加，而Adam、RMSProp和符号下降等逐坐标自适应方法则将每次更新除以其幅度的运行估计值，且该估计值恰好在词元刚出现后达到最大。这种不平衡带来两方面影响。在整体输出层层面，我们刻画了哪些优化器能够保持平均输出嵌入不变：任何更新对过去梯度呈线性的方法都能保持，Kronecker分解与正交化方法（如Shampoo和Muon）也能保持；而Adam、Adafactor、Lion和符号下降则不能，对于这些方法我们推导出了该变化的精确的逐步表达式。在单个稀有词元层面，同样的归一化会移动训练的不动点。在unigram模型中，符号下…（摘要在此处被截断）

    arXiv:2609.37535v1 Announce Type: new  Abstract: In the softmax output layer, a rare token receives a small positive logit gradient on most steps and a much larger negative gradient on the few steps when it is the target. SGD simply adds these contributions. Coordinate-wise adaptive methods such as Adam, RMSProp, and sign descent instead divide each update by a running estimate of its magnitude, and that estimate is largest immediately after the token appears. This imbalance has two effects. At the level of the whole output layer, we characterize which optimizers preserve the mean output embedding: every method whose update is linear in past gradients does, as do Kronecker-factored and orthogonalized methods such as Shampoo and Muon. Adam, Adafactor, Lion, and sign descent do not, and for these methods we obtain an exact step-by-step expression for the change. At the level of an individual rare token, the same normalization shifts the training fixed point. In the unigram model, sign de
    
[^19]: 物理Muon：作为平衡态计算的正交化

    Physical Muon: Orthogonalization as an Equilibrium Computation

    [https://arxiv.org/abs/2609.37525](https://arxiv.org/abs/2609.37525)

    该论文提出Physical Muon优化器，将Muon的正交化步骤转化为连续时间流的平衡态计算，仅需矩阵-向量乘积、倒数读取和局部秩1写入等模拟硬件友好的操作，从而在保持训练性能的同时实现物理可实现的优化。

    

    物理神经网络与模拟存内计算有望降低神经网络训练的能耗。然而，实现这一潜力需要优化器既能有效学习又具备物理可实现性。SGD适合局部模拟更新，但在transformer上表现不佳，而Adam系列优化器对模拟偏差并不稳定。Muon虽然具有出色的训练性能，但其Newton–Schulz正交化依赖于密集的矩阵-矩阵乘积。为解决这一障碍，我们提出Physical Muon，它将正交化计算为一个连续时间流的平衡态。随机探针利用矩阵-向量乘积、倒数读取和局部秩1写入来近似该流。为验证这种替代能否保持训练性能，我们在一个1095万参数的transformer上进行了评估。在每种方法九个随机种子下，密集流的平均验证交叉熵比Newton–Schulz高0.0085；探针实现的……

    arXiv:2609.37525v1 Announce Type: new  Abstract: Physical neural networks and analog in-memory computing could reduce the energy cost of neural network training. Realizing this potential, however, requires optimizers that combine effective learning with physical implementability. SGD fits local analog updates but struggles on transformers, while Adam family is unstable against analog bias. Muon offers strong training performance, but its Newton--Schulz orthogonalization relies on dense matrix-matrix products. To address this obstacle, we introduce Physical Muon, which computes the orthogonalization as the equilibrium of a continuous-time flow. Random probes approximate the flow using matrix-vector products, reciprocal reads, and local rank-1 writes. To test whether this replacement preserves training performance, we evaluate it on a 10.95M-parameter transformer. The dense flow's mean validation cross-entropy is 0.0085 above Newton--Schulz across nine seeds per method; the probe impleme
    
[^20]: 概率契约：跨大语言模型接口的准确性、一致性与决策

    Probability Contracts: Accuracy, Coherence, and Decisions Across LLM Interfaces

    [https://arxiv.org/abs/2609.37470](https://arxiv.org/abs/2609.37470)

    该论文提出“概率契约”基准，通过将精确有限世界后验、经验证的事件变换与失效感知决策评估相结合，系统揭示了大语言模型不同概率接口在准确性、一致性和决策损失上的显著差异，并证明接口间分歧不足以认证真实误差。

    

    用于决策的概率在等价的请求中应指向同一事件。我们提出了“概率契约”，这是一个将精确的有限世界后验分布、经过验证的事件变换以及失效感知的决策评估联系在一起的基准。我们在1,000个世界上评估了四种模型-接口配置。这些配置在准确性、一致性和决策损失方面的表现各不相同：Kev的总规范后验误差低于Jev，但其互补残差和粗化残差更大，且准确性排序随数据分层而变化。在延迟成本为0.10时，Jev的Event接口与Choice接口在32.8%的有效配对上会引发不同的二元动作。事后分析发现，接口间的分歧只能认证各配置中平均二元配对误差的11%至52%。一个基本的动作区域刻画解释了相对于随机选择单一接口，对接口输出取平均何时会改变决策损失。尽管取平均不会恶化该基线……（原文摘要在此处截断）

    arXiv:2609.37470v1 Announce Type: new  Abstract: A probability used for a decision should refer to the same event across equivalent requests. We introduce probability contracts, a benchmark connecting exact finite-world posteriors, validated event transformations, and failure-aware decision evaluation. Four model-interface configurations are evaluated on 1,000 worlds. Their assessments differ across accuracy, coherence, and decision loss: Kev has lower aggregate canonical posterior error than Jev, but larger complement and coarsening residuals, with accuracy ordering varying by stratum. Jev's Event and Choice interfaces induce different binary actions on 32.8% of valid pairs at defer cost 0.10. A post-hoc analysis finds that disagreement certifies only 11-52% of mean binary pair error across configurations. An elementary action-region characterization explains when averaging changes decision loss relative to randomly selecting one interface. Although averaging cannot worsen that baseli
    
[^21]: 具有最优失败指数的随机次梯度方法

    A stochastic subgradient method with optimal failure exponent

    [https://arxiv.org/abs/2609.37425](https://arxiv.org/abs/2609.37425)

    本文提出一种采用调和型均匀平均步长调度的随机次梯度方法，通过指数上鞅论证与相匹配的不可能性结果证明，其在次高斯梯度噪声下最小化超出目标精度的失败概率时达到了最优失败指数。

    

    固定目标精度 $\varepsilon$、梯度噪声水平 $s$ 和时间范围 $N$。我们希望设计能最小化“次优间隙超过目标精度”这一事件发生概率的算法，即 $\mathcal{E}_N = -\log \sup_{f,P} \mathbb{P}_P(f(x_A) - f_\star \ge \varepsilon)$，其中噪声分布 $P$ 仅已知为次高斯分布。我们挑选出一种调和型（形如 $h_k = R^2/(\varepsilon (N+m-k))$）的均匀平均步长调度，并通过优化的指数上鞅论证证明它达到了最优指数 $\mathcal{E}_N^\star = \varepsilon^2 N (1+o(1))/(2R^2 s^2)$。最优性由一个相匹配的不可能性结果加以证明：在高斯噪声下，一种掩蔽梯度的测度变换将所有算法的指数都限制在相同的主阶上。在小噪声极限下，我们的设定退化为 Gönsgens 与 van Parys（2025）针对次梯度方法的对抗性误差模型。

    arXiv:2609.37425v1 Announce Type: cross  Abstract: Fix a target accuracy $\varepsilon$, a gradient-noise level $s$, and a horizon $N$. We wish to design algorithms which minimize the probability of observing a suboptimality gap which exceeds the target accuracy, i.e., $\mathcal{E}_N = -\log \sup_{f,P} \mathbb{P}_P(f(x_A) - f_\star \ge \varepsilon)$, with the noise law $P$ known only to be sub-Gaussian. We single out a uniformly averaged schedule which is harmonic (of the form $h_k = R^2/(\varepsilon (N+m-k))$) and prove, via an optimized exponential supermartingale argument, that it attains the optimal exponent $\mathcal{E}_N^\star = \varepsilon^2 N (1+o(1))/(2R^2 s^2)$. Optimality is certified by a matching impossibility result: under Gaussian noise, a gradient-masking change of measure caps the exponent of every algorithm at the same leading order. In a small-noise limit, our setting degenerates into the adversarial-error model of G\"osgens and van Parys (2025) for subgradient method
    
[^22]: 潜在空间中的高维基于模拟的推断

    High-Dimensional Simulation-Based Inference in Latent Spaces

    [https://arxiv.org/abs/2609.37381](https://arxiv.org/abs/2609.37381)

    提出将基于模拟的推断与潜在生成式建模相融合的新框架，通过学习模拟器参数的低维表示、在潜在空间中直接进行后验推断再将样本映射回原始参数空间，从而解决高维参数空间的推断问题，并给出了潜在空间推断恢复目标后验分布的理论条件。

    

    神经基于模拟的推断（SBI）已在从可能高维的观测（如图像或时间序列）中推断相对少量可解释参数方面取得了广泛成功。相应地，SBI中的表示学习几乎完全专注于压缩用于对后验进行条件化的观测数据。然而，近年来SBI开始面向日益高维的参数空间，这引出了一个互补的问题：推断目标本身是否也应该被压缩。我们的答案是 将SBI与潜在生成式建模进行实用的融合，即学习模拟器参数的低维表示，直接在该潜在空间中执行后验推断，并将后验样本映射回原始参数空间。我们刻画了潜在空间推断能够恢复目标后验分布的条件，并系统地研究了其有效性。

    arXiv:2609.37381v1 Announce Type: new  Abstract: Neural simulation-based inference (SBI) has been widely successful in inferring a relatively small number of interpretable parameters from potentially high-dimensional observations, such as images or time series. Accordingly, representation learning in SBI has focused almost exclusively on compressing the observations used to condition the posterior. More recently, however, SBI has begun to target increasingly high-dimensional parameter spaces, raising the complementary question of whether the inference target itself should be compressed. Our answer is a practical merger of SBI and latent generative modeling, which learns a low-dimensional representation of the simulator parameters, performs posterior inference directly in this latent space, and maps posterior samples back to the original parameter space. We characterize the conditions under which latent-space inference recovers the desired target posterior and systematically study its e
    
[^23]: 差分隐私下数据重建的急剧转变

    A Sharp Transition in Data Reconstruction under Differential Privacy

    [https://arxiv.org/abs/2609.37344](https://arxiv.org/abs/2609.37344)

    本文在零集中差分隐私下建立了数据重建的急剧相变：当隐私预算 ρ 低于数据维度 d 量级时，即使攻击者知道其余全部训练数据，任何机制和攻击都无法在信息论意义上实现精确重建，从而为隐私预算的选择提供了理论边界。

    

    数据重建攻击已在从学习到的模型中恢复训练样本方面取得了经验性的成功，这引发了隐私担忧，并推动了对具有防御保证的防御措施的研究，这些保证在面对未来威胁时仍然有效。虽然差分隐私（DP）提供了形式化的保护，但如何选择隐私预算仍然是一个挑战：较小的预算会严重降低模型效用，但很难量化在不允许精确重建的前提下，预算最大可以设置到多少。在这项工作中，我们研究了一类知情的攻击者，他们的目标是从一个 ρ-零集中差分隐私模型中重建单个 d 维训练样本，并且知道所有其他训练数据。我们的主要贡献是建立了数据重建在 ρ ≍ d 处的急剧转变：一方面，我们针对任何隐私机制和任何攻击方法推导了基于熵的下界，刻画了一组目标先验分布，对于这些先验分布，重建在信息论上是不可能的

    arXiv:2609.37344v1 Announce Type: cross  Abstract: Data reconstruction attacks have empirically been successful in recovering training samples from learned models, raising privacy concerns and motivating defenses with guarantees that remain valid against future threats. While differential privacy (DP) provides formal protection, choosing the privacy budget remains a challenge: small budgets severely reduce utility, but it is hard to quantify how large the budget can be without allowing accurate reconstruction. In this work, we study informed attackers who aim to reconstruct a single $d$-dimensional training sample from a $\rho$-zero-concentrated DP model, knowing all other training data. Our main contribution is to establish a sharp transition at $\rho \asymp d$ for data reconstruction: on the one hand, we derive entropy-based lower bounds for any private mechanism and any attack, characterizing a set of target priors for which reconstruction is information-theoretically impossible for
    
[^24]: 用于采样奖励倾斜生成先验的相互作用粒子引导方法

    Interacting particle guidance for sampling reward-tilted generative priors

    [https://arxiv.org/abs/2609.37227](https://arxiv.org/abs/2609.37227)

    提出相互作用粒子引导（IPG），利用由Feynman–Kac偏微分方程导出的漂移项驱动粒子输运，取代传统的重加权机制，从而克服序贯蒙特卡罗方法中的权重退化和粒子坍缩问题，实现从奖励倾斜生成先验中的高效采样。

    

    推理时引导技术能够在无需重新训练的情况下，将预训练的扩散模型和基于流的模型适配到新任务中，例如生成来自条件分布的样本或具有期望属性的样本。这一问题可以被形式化为从奖励倾斜的生成先验中采样。由于从该分布进行精确采样是难以处理的，基于引导的方法依赖于近似，从而产生有偏的样本；而序贯蒙特卡罗（SMC）方法则通过重要性权重来纠正这种偏差。然而，尽管SMC在大粒子数量极限下是精确的，但在实际应用中存在权重退化和粒子坍缩的问题。我们提出了相互作用粒子引导（IPG），用粒子输运取代重加权机制。粒子通过一个额外的漂移项相互作用，该漂移项由Feynman–Kac偏微分方程推导得出，用于抵消重加权项，且粒子始终保持无权重状态。当将该漂移项限制在再生核希尔伯特空间（RKHS）中选取时，可以得到一个计算成本低廉的闭式解。

    arXiv:2609.37227v1 Announce Type: new  Abstract: Inference-time steering adapts pretrained diffusion and flow-based models to new tasks, e.g., to generate samples from a conditional distribution or samples with desired properties, without retraining. This can be formalized as sampling from a reward-tilted generative prior. As exact sampling from this distribution is intractable, guidance-based methods rely on approximations producing biased samples, and sequential Monte Carlo (SMC) methods correct for this bias using importance weights. However, while exact in the large particle limit, SMC suffers from weight degeneracy and particle collapse in practice. We propose interacting particle guidance (IPG), which replaces reweighting with transport. The particles interact through an additional drift, derived from the Feynman--Kac PDE to cancel the reweighting term, and remain unweighted. Choosing the drift in a reproducing kernel Hilbert space yields a closed-form solution that is cheap to c
    
[^25]: 逐点还是成对：成对损失何时能（以可证明的方式）帮助奖励学习？

    Pointwise or Pairwise: When Do Pairwise Losses Help Reward Learning, Provably?

    [https://arxiv.org/abs/2609.37209](https://arxiv.org/abs/2609.37209)

    该论文在允许每个上下文包含多个动作的分组离线上下文老虎机设置下，首次从理论上刻画了成对损失（价值差分回归VDR）相对于逐点价值回归（VR）何时以及为何能带来可证明的优势，并在包含与动作无关的上下文干扰项的半参数模型下给出了有限样本回归保证和离线遗憾界。

    

    即使在能够观测到逐点奖励的情况下，成对损失也越来越多地被用于奖励学习，但其实证结果好坏参半。成对损失何时以及为何会优于逐点损失？我们在一个允许每个上下文包含多个动作的分组离线上下文老虎机（contextual-bandit）设置中研究这一问题，该设置涵盖了许多奖励学习场景。我们比较了价值回归（VR）——对观测到的奖励进行逐点回归——与价值差分回归（VDR）——对在同一上下文下采样得到的一对动作之间的奖励差进行回归。我们考虑一个半参数模型，其中平均奖励等于一个可学习的动作相关分量与一个任意的上下文相关但与动作无关的干扰项之和，从而刻画了上下文特定的扰动。利用统一化的局部化分析，我们对有限函数类和线性函数类证明了有限样本回归保证，并将其转化为离线遗憾界。对于有限类……

    arXiv:2609.37209v1 Announce Type: new  Abstract: Pairwise losses are increasingly used for reward learning even when pointwise rewards are observed, with mixed empirical results. When and why do pairwise losses outperform pointwise losses? We study this question in a grouped offline contextual-bandit setting allowing multiple actions per context, capturing many reward learning scenarios. We compare Value Regression (VR), which regresses observed rewards pointwise, with Value Difference Regression (VDR), which regresses reward differences between a pair of actions sampled under the same context. We consider a semiparametric model where the mean reward is the sum of a learnable action-dependent component and an arbitrary context-dependent yet action-independent nuisance, capturing context-specific disturbances. Using a unified localized analysis, we prove finite-sample regression guarantees for finite and linear function classes and translate them into offline-regret bounds. For finite c
    
[^26]: 通过距离与角度的分量级校准实现可解释的内在维度估计

    Interpretable intrinsic dimension estimation through componentwise calibration of distance and angle

    [https://arxiv.org/abs/2609.37114](https://arxiv.org/abs/2609.37114)

    该论文将DANCo内在维度估计方法重构为可分别校准与解释的距离分量和角度分量，并为Gride估计器推导出闭式KL散度，从而在含噪与幅度异质的实际数据上显著降低估计误差（如24个流形上40%噪声水平下平均百分比误差从27.7%降至17.6%）并增强可解释性。

    

    DANCo（Dimensionality from Angle and Norm Concentration，基于角度与范数集中性的维度估计方法）通过联合校准最近邻距离与角度统计量，在干净的内在维度（ID）基准上持续达到最先进的精度。然而，实际数据会引入邻域相对噪声与样本幅度异质性，从而可能扭曲这些几何信号。我们对DANCo进行了分量级重构，保留各自独立的距离差异曲线与角度差异曲线，使估计结果的来源能够被识别与解释。对于距离分量，我们为广义比值内在维度估计器（Gride）的通用阶比值推导出了闭式Kullback-Leibler散度；当两个角度参数均匹配（Full设置）时，在24个流形上、噪声水平为典型近邻间距40%的条件下，Gride将平均百分比误差从27.7%降至17.6%。对于角度分量，两种采样机制促使我们在对齐平均方向的同时……（原文摘要在此处截断）

    arXiv:2609.37114v1 Announce Type: new  Abstract: DANCo (Dimensionality from Angle and Norm Concentration) jointly calibrates nearest-neighbor distance and angular statistics and consistently reaches state-of-the-art accuracy on clean intrinsic-dimension (ID) benchmarks. Practical data, however, introduce neighborhood-relative noise and sample-amplitude heterogeneity that can distort these geometric signals. We reformulate DANCo componentwise, retaining separate distance and angular discrepancy curves so that the source of an estimate can be identified and interpreted. For the distance component, we derive a closed-form Kullback-Leibler divergence for the generic-order ratios of the generalized ratios ID estimator (Gride); when both angular parameters are matched (Full), Gride reduces mean percentage error from $27.7\%$ to $17.6\%$ at noise equal to $40\%$ of typical neighbor spacing on 24 manifolds. For the angular component, two sampling regimes motivate aligning mean direction while 
    
[^27]: 基于因果表征学习从非结构化数据中识别常微分方程

    Identifying ODEs from Unstructured Data with Causal Representation Learning

    [https://arxiv.org/abs/2609.37083](https://arxiv.org/abs/2609.37083)

    提出SPEED-AE框架，将预训练的因果表征学习与逐分量自编码器结合，首次从图像等非结构化高维数据中可证明地学习到适合稀疏ODE发现的变量表示，从而实现对动力系统控制方程的识别。

    

    我们研究从诸如图像等非结构化、高维观测数据中恢复动力系统的控制常微分方程（ODE）的问题。现有的ODE发现方法通常假设可以直接测量变量，或者无法为学习到的变量和方程提供理论保证。尽管因果表征学习（CRL）方法能够保证从高维观测中识别变量（直至逐分量微分同胚的等价性），但我们证明这些变量通常不能直接用作方程发现方法的输入，因为这类方法通常假设变量能够导出稀疏方程。为此，我们提出了稀疏等价方程发现自编码器（SPEED-AE），这是一个将预训练的CRL方法与逐分量自编码器相结合的框架，该自编码器学习适用于稀疏ODE发现的变量变换。我们证明，对于多项式ODE，这一附加的变换……（摘要在此处截断）

    arXiv:2609.37083v1 Announce Type: cross  Abstract: We study the problem of recovering the governing ODE of a dynamical system from unstructured, high-dimensional observations such as images. Existing methods for ODE discovery typically assume direct measurements of the variables, or do not provide theoretical guarantees on the learned variables and equations. While Causal Representation Learning (CRL) methods provide guarantees on identifying variables from high-dimensional observations up to component-wise diffeomorphisms, we show that in general these variables cannot be used directly as input to equation discovery methods, which typically assume that the variables will lead to sparse equations. So we introduce SParse Equivalent Equation Discovery AutoEncoder (SPEED-AE), a framework that combines a pretrained CRL method with a component-wise autoencoder that learns transformations of variables that are amenable to sparse ODE discovery. We show that for polynomial ODEs, this additiona
    
[^28]: 面向基于能量采样的迭代精确离散引导

    Iterative Exact Discrete Guidance for Energy-Based Sampling

    [https://arxiv.org/abs/2609.37043](https://arxiv.org/abs/2609.37043)

    提出了一种迭代精确离散引导框架（IEDG），沿退火路径逐步学习玻尔兹曼倾斜的阶段局部后验修正，并以相对有效样本量自适应调节步长，从而实现对大型离散状态空间上非归一化多峰目标分布的精确采样。

    

    当多峰目标分布远离一个易于处理的参考分布时，从大型离散状态空间上的非归一化分布中采样会变得非常困难。我们提出了迭代精确离散引导，这是一个面向非归一化离散目标的、种群精确且逐轨迹的引导框架。IEDG 并非一步学会从参考分布到目标分布的完整修正，而是沿一条退火轨迹引入全局玻尔兹曼倾斜。每一阶段针对当前源分布的增量玻尔兹曼倾斜学习一个阶段局部的后验修正，同时这些所得修正是相对于一个固定的解析后验进行累积的。在种群最优处，精确的阶段后验能够恢复正确的反向动力学，而对反向动力学的精确模拟即可重现目标分布。IEDG 通过相对有效样本量来选择阶段增量，该指标可控制 Rényi-2 位移，并使步长局部自适应于热力学几何（摘要原文至此处截断）。

    arXiv:2609.37043v1 Announce Type: new  Abstract: Sampling from unnormalized distributions over large discrete state spaces becomes difficult when a multimodal target is far from a tractable reference. We introduce Iterative Exact Discrete Guidance (IEDG), a population-exact, trajectory-wise guidance framework for unnormalized discrete targets. Rather than learn the full reference-to-target correction in one step, IEDG introduces a global Boltzmann tilt along an annealing trajectory. Each stage learns a stage-local posterior correction for an incremental Boltzmann tilt of the current source, while the resulting corrections are accumulated relative to a fixed analytic posterior. At the population optimum, exact stage posteriors recover the correct reverse dynamics, whose exact simulation reproduces the target distribution. IEDG chooses stage increments by relative effective sample size (rESS), which controls R\'enyi-2 displacement and locally adapts the step size to the thermodynamic geo
    
[^29]: 面向模拟器误设情形下组合推断的可扩展扩散式模拟推断（SBI）

    Scalable Diffusion SBI for Compositional Inference under Simulator Misspecification

    [https://arxiv.org/abs/2609.36950](https://arxiv.org/abs/2609.36950)

    该论文提出了可扩展的扩散式模拟推断方法：通过考虑观测数量的连续时间扩散系数扩展组合式分数推断，并引入层次化分块扩散采样（HBDS），使单一预训练模型无需重训即可在不同观测集合与分组下推断共享参数和分组潜在状态，同时借助路径正则化缓解模拟器误设问题。

    

    当需要对大量异构观测进行组合、需要保留层次化潜在结构、且模拟器相对于观测数据存在误设时，基于模拟的推断会变得非常困难。我们针对“设计条件化”场景中的基于扩散的推断开发了采样与微调方法；在该场景下，同一模拟器会在不同实验条件 $\xi$ 下被反复调用。我们扩展了组合式基于分数的推断，引入了一个能够考虑观测数量的连续时间扩散系数，从而避免了雅可比修正和辅助协方差修正。我们提出了层次化分块扩散采样（Hierarchical Blockwise Diffusion Sampling, HBDS），该方法仅使用单个预训练模型即可推断共享参数与各组特定的潜在状态，且层次结构只需在采样阶段指定。这些方法结合在一起，可以在无需重新训练的情况下支持可变的观测集合与分组方式。为应对模拟器误设问题，我们引入了路径正则化（摘要在此处被截断）。

    arXiv:2609.36950v1 Announce Type: new  Abstract: Simulation-based inference is challenging when many heterogeneous observations must be composed, hierarchical latent structure must be preserved, and the simulator is misspecified relative to observed data. We develop sampling and fine-tuning methods for diffusion-based inference in design-conditional settings, where the same simulator is queried across different experimental conditions $\xi$. We extend compositional score-based inference with a continuous-time diffusion coefficient that accounts for the number of observations, avoiding Jacobian and auxiliary-covariance corrections. We introduce Hierarchical Blockwise Diffusion Sampling (HBDS), which infers shared parameters and group-specific latent states using a single pretrained model, with the hierarchy specified only at sampling time. Together, these methods support variable observation sets and groupings without retraining. To address misspecification, we introduce path-regularize
    
[^30]: 超越条件独立性：基于深度因果模型的根因分析

    Beyond Conditional Independence: Root Cause Analysis with Deep Causal Models

    [https://arxiv.org/abs/2609.36771](https://arxiv.org/abs/2609.36771)

    该论文提出基于深度因果模型的根因分析方法，通过建立分布约束检验与根因分析之间的隐式联系，突破了传统方法对条件独立性和无混杂性强假设的依赖，从而能够在存在潜变量混杂的情况下对任意因果模型生成的数据进行根因识别。

    

    根因分析（RCA）是许多现实场景中的关键问题。RCA通过将异常观测与相应的参考（即正常）观测进行比较，能够识别系统中出现故障或失效的机制。然而，现有方法要么依赖启发式方法，要么依赖带有强无混杂性假设的条件独立性检验，因此无法在存在潜变量的情况下利用其他复杂的分布约束。为了放宽这些假设，我们将底层系统建模为一个因果模型，并将异常系统建模为同一因果模型中结构函数的变化。具体而言，为了处理未观测的混杂因素，我们建立了分布约束检验与根因分析之间的隐式联系。为了使我们的方法能够适应由任意因果模型生成的数据，我们采用了深度因果模型（DCM）框架……

    arXiv:2609.36771v1 Announce Type: cross  Abstract: Root cause analysis (RCA) is a critical problem in many real-world scenarios. RCA enables the identification of faulty or failing mechanisms in a system by comparing anomalous observations with corresponding reference (i.e., regular) observations. However, existing approaches rely either on heuristic methods or on conditional independence tests with a strong unconfoundedness assumption, and thus fail to exploit other complicated distributional constraints in the presence of latent variables. To relax these assumptions, we model the underlying system as a causal model and the anomalous system as a change in the structural functions of the same causal model. Specifically, to handle unobserved confounders, we establish an implicit connection between distributional constraint testing and root cause analysis. To adapt our approach to data generated from arbitrary causal models, we employ the deep causal model (DCM) framework, in which we de
    
[^31]: 具有未知局部与全局聚类数量的联邦聚类

    Federated Clustering with Unknown Local and Global Cluster Cardinalities

    [https://arxiv.org/abs/2609.36762](https://arxiv.org/abs/2609.36762)

    本文提出了一个局部与全局聚类数量均未知的两阶段联邦聚类框架，通过自适应分裂-合并（ASM）算法让每个客户端从自身数据估计局部聚类数量，供 FedGEM 等需要局部数量的聚合器使用。

    

    不需要全局聚类数量 $K$ 的联邦聚类方法仍然假设每个客户端知道其自身的局部聚类数量 $K_g$。当客户端对其数据的了解并不比服务器更多时，这一假设难以成立，例如在跨独立运营的工业现场进行故障诊断的场景中。我们提出了一个两阶段框架，其中两个数量均未知：每个客户端首先从其自身数据估计 $K_g$，然后需要局部数量的聚合器（如 FedGEM）用这些估计值代替真实值。在第一阶段，我们引入了自适应分裂-合并算法，它通过 BIC 驱动的分裂来增长球面高斯混合模型，然后合并多余的成分。ASM 不使用任何标签，仅在客户端的留出数据上选择超参数，并且不对聚类在客户端之间的共享方式做任何假设。我们推导了一个闭式分裂准则，其临界聚类大小随各向异性……（原文摘要至此截断）

    arXiv:2609.36762v1 Announce Type: new  Abstract: Federated clustering methods that do not require the global number of clusters $K$ still assume that each client knows its local number $K_g$. This assumption is hard to justify when clients know no more about their data than the server does, as in fault diagnosis across independently operated industrial sites. We propose a two-phase framework in which neither count is known: each client first estimates $K_g$ from its own data, and an aggregator that requires local counts, such as FedGEM, then uses these estimates in place of the true values. For the first phase we introduce Adaptive Split--Merge (ASM), which grows a spherical Gaussian mixture by BIC-driven splitting and then merges excess components. ASM uses no labels, selects its hyperparameters on held-out client data only, and makes no assumption about how clusters are shared across clients. We derive a closed-form split criterion whose critical cluster size falls with anisotropy an
    
[^32]: 进入危险区域：高维函数与算子学习中的稳定外推

    Into the danger zone: stable extrapolation in high-dimensional function and operator learning

    [https://arxiv.org/abs/2609.36709](https://arxiv.org/abs/2609.36709)

    该论文识别出某些全纯函数与算子类别，证明即使在大幅分布偏移下其分布外泛化误差仍能以代数速率收敛，这一现象源于高阶坐标平滑性的增加，被称为“高维之福”。

    

    分布外泛化是科学机器学习中的核心挑战。我们研究了测试分布与训练分布不同的回归问题，并探讨：在对目标函数或算子施加何种假设下稳定外推是可能的，以及可以在训练域之外外推多远？现有理论通过衡量训练分布与测试分布之间差异的加性惩罚项来控制测试误差。这类保证显示了对小幅分布偏移的鲁棒性，但与经验上观察到的分布外性能相比可能非常悲观。我们识别出了若干全纯函数和算子的类别，即使存在大幅分布偏移，其分布外泛化误差仍能以代数速率收敛。这一现象源于高阶指标坐标平滑性的增加，我们将其称为“高维之福”。

    arXiv:2609.36709v1 Announce Type: cross  Abstract: Out-of-distribution (OOD) generalization is a central challenge in scientific machine learning. We study regression problems in which the test distribution differs from the training distribution and ask: under what assumptions on the target function or operator is stable extrapolation possible, and how far beyond the training domain can one extrapolate? Existing theory controls the test error through additive penalties measuring the discrepancy between the training and test distributions. Such guarantees show robustness to small distribution shifts, but can very pessimistic in comparison to OOD performance observed empirically. We identify classes of holomorphic functions and operators for which the OOD generalization error converges at algebraic rates even in the presence of large distribution shifts. This phenomenon stems from the increasing smoothness of higher-index coordinates, leading to what we term a `blessing of high dimension
    
[^33]: 粗粒度监督何时值得？未知聚合下的成本感知学习

    When Is Coarse Supervision Worth It? Cost-Aware Learning under Unknown Aggregation

    [https://arxiv.org/abs/2609.36704](https://arxiv.org/abs/2609.36704)

    该论文研究了在聚合规则未知的情况下精细标签与廉价粗粒度标签之间的成本感知权衡，证明粗监督的价值由成本、噪声和可识别性共同决定，推导出粗监督的闭式盈亏平衡条件，并提出一种达到最优累积风险的“估计-追踪”在线策略。

    

    现代学习系统通常需要在多个分辨率下获取监督信号，在标注成本与信息含量之间进行权衡。我们研究了成本感知的双分辨率学习问题：其中昂贵的精细标签揭示一个向量响应，而较廉价的粗粒度标签揭示一个由未知权重形成的标量聚合，同时学习目标始终是完整的响应。挑战在于，未知的聚合方式会改变粗粒度数据所能识别的方向，因此粗粒度监督的价值同时取决于成本、噪声和可识别性。我们刻画了这一信息几何结构，并开发了一种“估计-追踪”策略，用以学习聚合规则并追踪最优的分辨率组合。我们推导出了粗粒度监督的闭式盈亏平衡条件，并证明该在线策略能够达到最优的首阶累积风险系数，同时给出了匹配的局部渐近极小极大下界。合成实验验证了所预测的全精细/（原文摘要至此截断）

    arXiv:2609.36704v1 Announce Type: new  Abstract: Modern learning systems often acquire supervision at multiple resolutions, trading annotation cost against information content. We study cost-aware two-resolution learning, where expensive fine labels reveal a vector response and cheaper coarse labels reveal a scalar aggregate formed with unknown weights, while the target remains the full response. The challenge is that unknown aggregation changes which directions coarse data can identify, so the value of coarse supervision depends jointly on cost, noise, and identification. We characterize this information geometry and develop an estimate-and-track policy that learns the aggregation rule and tracks the optimal resolution mix. We derive a closed-form break-even condition for coarse supervision and prove that the online policy attains the optimal leading cumulative-risk coefficient, with a matching local asymptotic minimax lower bound. Synthetic experiments support the predicted all-fine/
    
[^34]: 理解私有进化作为学习增强的聚类方法

    Understanding Private Evolution as Learning-Augmented Clustering

    [https://arxiv.org/abs/2609.36678](https://arxiv.org/abs/2609.36678)

    本文将私有进化（PE）重新表述为生成模型增强的Wasserstein学习，证明利用生成模型可获得更好的性能界（如样本复杂度取决于内在维度而非环境维度），并针对标准PE在良好聚类实例上不收敛的问题，提出了具有可证明收敛性的几何感知新算法。

    

    私有进化（PE）是一种用于合成数据生成的差分隐私算法。虽然它可以被视为一种Wasserstein学习算法，但它在实践中的表现远好于最坏情况Wasserstein分析所预测的结果。我们将PE重新表述为生成模型增强的Wasserstein学习。我们从理论上证明，当考虑到使用能够捕捉真实分布某些特性的生成模型时，可以获得更好的性能界。例如，如果生成器在与分布相同的低维空间中给出样本，那么样本复杂度取决于内在维度而非环境维度。我们还证明了PE的标准变体在简单的良好聚类实例上可能无法收敛，并提出了一种新的几何感知版本的PE，该版本在此类实例上具有可证明的收敛性。实验表明，我们的新算法与标准基线相比具有竞争力。

    arXiv:2609.36678v1 Announce Type: new  Abstract: Private Evolution (PE) is a differentially private algorithm for synthetic data generation. While it can be viewed as a Wasserstein learning algorithm, it performs much better in practice than worst-case Wasserstein analyses would predict. We recast PE as generative model-augmented Wasserstein learning. We show theoretically that when we take into account the use of a generative model that is able to capture something about the true distribution, then we can obtain much better performance bounds. For example, if the generator gives samples in the same low-dimensional space as the distribution, then sample complexity depends on intrinsic, not ambient, dimension. We also show that standard variants of PE can fail to converge on simple well-clustered instances, and propose a new geometry-aware version of PE with provable convergence on such instances. Experimentally, we show that our new algorithm is competitive with standard baselines and 
    
[^35]: 二阶矩随机逼近方法

    Second-Moment Stochastic Approximation Methods

    [https://arxiv.org/abs/2609.36600](https://arxiv.org/abs/2609.36600)

    本文提出基于一阶矩与二阶矩估计器的二阶矩随机逼近方法（将Adam、Muon等现代优化器纳入统一框架），通过矩阵方程最优预条件化视角推导方法，并建立两阶段收敛分析框架，证明了实用方法几乎必然收敛到目标解的邻域，且邻域大小由矩估计器的偏差和方差决定。

    

    经典的随机逼近方法依赖于随机回归函数的一阶矩（均值）估计器。我们研究了同时采用一阶矩和二阶矩估计器的方法，其中包括Adam和Muon等现代深度学习优化器作为特例。我们从求解矩阵方程的最优预条件化视角推导出二阶矩随机逼近方法，并为其收敛性分析开发了一个两阶段框架。第一阶段重点分析依赖精确一阶矩和二阶矩的概念性（不实用的）方法。在第二阶段，我们用各自的估计器替代精确矩，并借助Dvoretzky定理证明所得到的实用方法几乎必然收敛到目标解的一个邻域。该邻域的大小取决于一阶矩和二阶矩估计器的偏差和方差。

    arXiv:2609.36600v1 Announce Type: cross  Abstract: Classical stochastic approximation methods rely on estimators of the first moment (mean) of a random regression function. We study methods that employ estimators of both the first and the second moments, which include modern deep-learning optimizers such as Adam and Muon as special cases. We derive second-moment stochastic approximation methods through the lens of optimal preconditioning for solving matrix equations, and develop a two-stage framework for their convergence analysis. The first stage focuses on the analysis of conceptual (impractical) methods that rely on the exact first and second moments. In the second stage, we replace the exact moments with their respective estimators, and invoke Dvoretzky's theorem to show that the resulting practical methods converge almost surely to a neighborhood of the target solution. The size of the neighborhood depends on the biases and variances of the first- and second-moment estimators. We 
    
[^36]: 一般矩变化的最优检测：均值与协方差变化的同时检测及更广泛的扩展

    Optimal detection of general moment changes: Simultaneous mean and covariance change detection and beyond

    [https://arxiv.org/abs/2609.36594](https://arxiv.org/abs/2609.36594)

    提出了一种基于张量表示的多元时间序列多阶矩变化点检测方法，可在统一框架下同时检测均值、协方差及高阶矩的变化，并在适当条件下达到极小极大最优的定位误差率。

    

    我们研究多元时间序列中的多变点检测问题，其中分布以分段常数的方式发生变化。分布变化可以体现在不同阶数的矩上，从均值和协方差的偏移到高阶矩的变化。高阶矩能够捕捉越来越丰富的分布特征，但在高维情形下却变得难以估计。我们的张量表示将不同阶数的矩统一在一个共同的线性代数框架内，由此提出一种新方法，能够检测直至预设固定阶数 $p$ 的所有阶矩的变化。所提出的检测程序能够适应时间相依性，并允许时间序列的维度随样本量增长。在适当的正则条件下，所提出的程序达到了与新建立的极小极大下界相匹配的定位误差率。我们进一步推导了在非消失情形和……（原文此处截断）下的极限分布。

    arXiv:2609.36594v1 Announce Type: cross  Abstract: We study multiple change-point detection in multivariate time series whose distributions change in a piecewise constant manner. Distributional changes can manifest across different moment orders, from shifts in the mean and covariance to changes in higher-order moments. Higher-order moments capture increasingly rich distributional features but become difficult to estimate in high dimensions. Our tensor representation unifies moments of different orders within a common linear algebraic framework, enabling a new method to detect changes in moments of all orders up to a prescribed fixed order $p$. The resulting procedure accommodates temporal dependence and allows the dimension of the time series to grow with the sample size. Under suitable regularity conditions, the proposed procedure achieves a localization error rate that matches a newly developed minimax lower bound. We further derive limiting distributions under both nonvanishing and
    
[^37]: 结构化多类别决策的层次化效用校准

    Hierarchical Utility Calibration for Structured Multiclass Decisions

    [https://arxiv.org/abs/2609.36532](https://arxiv.org/abs/2609.36532)

    该论文提出层次化效用校准（HUC），通过将效用误差精确分解为标签树各内部节点的贡献之和，解决了传统效用校准中正负贡献相互抵消、掩盖层次结构局部效用误差的问题。

    

    在多类别概率预测中，效用校准（Utility Calibration, UC）通过将审计聚焦于指定的效用，近年来作为一种在控制计算与样本需求的同时保障下游决策的方法而受到关注。与此同时，一些多类别问题具有有意义的标签层次结构，这些层次结构在医学和图像分类中发挥着重要作用，然而UC如何在层次结构中评估效用仍缺乏充分理解。我们证明，实际效用与预测平均效用之间的差异可以精确分解为标签树各内部节点贡献之和。这一分解表明，来自不同节点的正负贡献可能相互抵消，并且即使UC很小，层次结构某些部分中残留的效用误差也未必很小。为解决这一问题，我们提出了层次化效用校准，它评估每个……

    arXiv:2609.36532v1 Announce Type: cross  Abstract: In multiclass probabilistic prediction, Utility Calibration (UC), which focuses auditing on specified utilities, has recently received attention as a way to guarantee downstream decisions while controlling computational and sample requirements. At the same time, some multiclass problems have meaningful label hierarchies that play important roles in medicine and image classification, yet how UC evaluates utility within a hierarchy remains insufficiently understood. We show that the difference between realized utility and predicted mean utility admits an exact decomposition into a sum of contributions from the internal nodes of the label tree. This decomposition shows that positive and negative contributions from different nodes can cancel, and that even when UC is small, the utility errors remaining in parts of the hierarchy need not be small. To address this problem, we propose Hierarchical Utility Calibration (HUC), which evaluates ea
    
[^38]: 最优多奖励强化学习

    Optimal Multi-Reward Reinforcement Learning

    [https://arxiv.org/abs/2609.36486](https://arxiv.org/abs/2609.36486)

    该论文研究了多奖励函数的有限时域强化学习问题，提出了一个无需额外预烧成本的算法，达到了与信息论下界仅相差多对数因子的最优样本复杂度 $O(SAH^3\log M/\epsilon^2)$。

    

    我们研究了一个转移动态未知的有限时域马尔可夫决策过程（MDP），其中包含有限个已知的奖励函数集合 $\{r^1, r^2, \ldots, r^M\}$。目标是仅通过在线回合式交互，为每一个奖励函数输出一个 $\epsilon$-最优策略。性能通过策略误差 $V_{0}^{*, m} - V_{0}^{\widehat\pi^{m}, m}$ 来衡量，其中 $m\in [M]$ 表示奖励函数，$V_{0}^{*, m}=\mathbb{E}_{s_1\sim \mu}[V_{1}^{*, m}(s_1)]$。在此设定下，我们设计了一个可证明高效的算法，建立了最小最大（minimax）样本复杂度界 $O\left(\frac{SAH^3}{\epsilon^2}\log M \mathrm{polylog}\left(\frac{SAH\log M}{\min\{\epsilon, 1\}\delta}\right)\right)$ 个回合，且无需额外的预烧（burn-in）成本。该结果与信息论下界相比仅相差 $\mathrm{polylog}(SAH\log M/(\min\{\epsilon, 1\}\delta))$ 因子。我们的方法结合了三项技术要素。首先，我们……（摘要至此截断）

    arXiv:2609.36486v1 Announce Type: new  Abstract: We study an unknown-transition finite-horizon Markov decision process (MDP) with a finite collection of known reward functions $\{r^1, r^2, \ldots, r^M\}$. The goal is to output an $\epsilon$-optimal policy for every reward using online episodic interaction only. Performance is measured by the policy error $V_{0}^{*, m} - V_{0}^{\widehat\pi^{m}, m}$ where $m\in [M]$ represents the reward function and $V_{0}^{*, m}=\mathbb{E}_{s_1\sim \mu}[V_{1}^{*, m}(s_1)]$. Under this setting, we design a provably efficient algorithm to establish a minimax sample complexity bound of $$ O\left(\frac{SAH^3}{\epsilon^2}\log M \mathrm{polylog}\left(\frac{SAH\log M}{\min\left\{\epsilon, 1\right\}\delta}\right)\right)$$ episodes, with no additional burn-in cost. This matches the information-theoretic lower bound up to a factor of $ \mathrm{polylog}(SAH\log M/(\min\left\{\epsilon, 1\right\}\delta))$. Our method combines three technical ingredients. First, we 
    
[^39]: LOCO-AdaMP：面向自适应小补丁集成的内置LOCO推断与增强预测

    LOCO-AdaMP: Built-in LOCO Inference for Adaptive Minipatch Ensembles with Enhanced Prediction

    [https://arxiv.org/abs/2609.36396](https://arxiv.org/abs/2609.36396)

    提出了LOCO-AdaMP框架，通过LOCO重要性引导的自适应特征采样构建小补丁集成，在无需数据拆分的情况下实现渐近有效的特征重要性推断，同时显著提升高维稀疏场景下的预测性能。

    

    随着黑盒机器学习模型日益普遍，提取带有不确定性量化的解释已成为一项关键挑战。一种流行的解释类型是留一协变量法（LOCO）特征重要性，而先前的LOCO推断方法通常需要数据拆分或模型重拟合。最近的一种集成框架LOCO-MP通过对观测值和特征同时进行子采样的小补丁来解决这些挑战，但大规模的特征子采样可能会在高维稀疏设置中损害预测性能。受这一局限性的启发，我们考虑了由LOCO重要性引导的自适应特征采样的小补丁集成方法，并提出了LOCO-AdaMP，它能够对所得的自适应小补丁集成实现免费的LOCO推断。我们证明，LOCO-AdaMP在保留渐近有效的特征重要性推断（无需数据拆分）的同时，产生了预测性能显著提升的模型……

    arXiv:2609.36396v1 Announce Type: cross  Abstract: As black-box machine learning models become increasingly common, extracting interpretations with uncertainty quantification has become a critical challenge. One popular type of interpretation is leave-one-covariate-out (LOCO) feature importance, while prior LOCO inference methods often require data-splitting or model-refitting. A recent ensemble framework, LOCO-MP, addresses these challenges using minipatches that subsample both observations and features, but massive feature subsampling can hurt prediction in high-dimensional sparse settings. Motivated by this limitation, we consider minipatch ensembles with adaptive feature sampling guided by LOCO importance, and propose LOCO-AdaMP, which enables free LOCO inference for the resulting adaptive minipatch ensemble. We show that LOCO-AdaMP yields substantially improved predictive models while retaining asymptotically valid feature importance inference without data-splitting, despite the c
    
[^40]: 当动作空间为函数时的拟合Q迭代有限样本理论

    Finite-Sample Theory for Fitted Q-Iteration When Actions Are Functions

    [https://arxiv.org/abs/2609.36390](https://arxiv.org/abs/2609.36390)

    本文首次建立了动作空间为函数（如放射治疗通量图、机器人运动轨迹）时拟合Q迭代的有限样本理论，通过提出无需动作密度的“评论家相对覆盖条件”以及平滑正则化策略搜索，克服了函数型动作空间中覆盖刻画困难、传统覆盖要求过严和贪婪优化难以实施这三大难题。

    

    离线强化学习旨在从先前收集的数据中寻找最优决策规则。在某些应用中，决策本身可以是一个完整的函数，例如放射治疗中的剂量通量图或机器人技术中的平滑运动轨迹。本文研究了在折扣无限时域设定下，具有函数型动作的拟合Q迭代（FQI）的有限样本理论。在这一设定中出现了三大困难：首先，函数型动作缺乏勒贝格概率密度，使得覆盖性的刻画变得复杂；其次，传统的覆盖性要求可能过于苛刻；第三，庞大的函数型动作空间使得FQI中的贪婪优化面临挑战。为解决这些困难，我们在“评论家相对覆盖条件”下研究了平滑正则化的策略搜索。该条件衡量的是已记录数据区分相关动作值差异的能力，而无需假设动作存在概率密度。我们的……（摘要截断）

    arXiv:2609.36390v1 Announce Type: cross  Abstract: Offline reinforcement learning seeks optimal decision rules from previously collected data. In some applications, a decision can be an entire function, such as a fluence map in radiation therapy or a smooth movement trajectory in robotics. In this paper, we study the finite-sample theory for fitted Q-iteration (FQI) with functional actions in a discounted infinite-horizon setting. Three major difficulties arise in this setting: first, the absence of a Lebesgue probability density for functional actions complicates coverage descriptions; second, conventional coverage requirements can be restrictive; and third, the large functional action space makes greedy optimization in FQI challenging. To address these difficulties, we study smoothness-regularized policy search under a critic-relative coverage condition. This condition measures how well logged data distinguish relevant action-value differences without requiring an action density. Our
    
[^41]: 面向表格上下文学习的线性时间架构适配

    Adapting Linear-Time Architectures for Tabular In-Context Learning

    [https://arxiv.org/abs/2609.36337](https://arxiv.org/abs/2609.36337)

    该研究发现因果线性序列混合器DeltaNet最适合表格上下文学习，性能甚至超过非因果线性注意力，但在超出预训练上下文长度2-4倍时性能退化，且实验表明这一退化并非源于隐藏状态容量限制。

    

    表格基础模型通过在上下文中条件化于带标签的示例而取得了优异的性能，但softmax注意力机制限制了其在大规模数据集上的应用。然而，现有的线性时间替代方案大多是因果的，它们在表格上下文学习（ICL）中的潜力仍未得到充分探索。为解决这一问题，我们（1）重新审视了因果训练设置，（2）比较了各种线性序列混合器，并（3）研究了它们在超出预训练上下文长度时的ICL泛化能力。首先，我们证明因果模型的最佳训练设置类似于下一词元预测。随后，令人惊讶的是，最有前景的线性序列混合器竟是因果的：DeltaNet甚至优于非因果的线性注意力。然而，当序列长度超过预训练上下文长度的2-4倍时，其性能会出现退化，而现有的缓解策略（如双向化）充其量只是延迟了这一问题的发生。隐藏状态oracle实验表明，这并非状态容量问题。相反，我们的……

    arXiv:2609.36337v1 Announce Type: new  Abstract: Tabular foundation models achieve strong performance by conditioning on labelled examples in context, but softmax attention limits their use on large datasets. Existing linear-time alternatives, however, are mostly causal, and their potential for tabular in-context learning (ICL) remains underexplored. To address this, we (1) revisit causal training setups, (2) compare linear sequence mixers, and (3) investigate their ICL generalisation beyond the pretraining context length. First, we show that the best training setup for causal models resembles next-token prediction. Then, perhaps surprisingly, the most promising linear sequence mixer is causal: DeltaNet outperforms even non-causal linear attention. However, it degrades beyond $2$-$4\times$ the pretraining context length, and existing mitigation strategies such as bidirectionality defer the problem at best. A hidden-state oracle shows that this is not a capacity problem. Instead, our an
    
[^42]: 用于监督子空间的廉价而高效的检验：PLS的逐成分推断

    Cheap and Powerful Tests for Supervised Subspaces: Per-Component Inference for PLS

    [https://arxiv.org/abs/2609.36307](https://arxiv.org/abs/2609.36307)

    本文提出两种基于留出OLS重拟合的廉价高效检验方法（Nadeau-Bengio校正渐近t检验与置换检验），为PLS监督子空间提供统计推断，并通过固定序列检验实现对各成分的逐一推断。

    

    偏最小二乘（PLS）回归在高维X中提取少数几个与结果变量对齐的方向，在应用科学领域被广泛使用，但对所得拟合的统计推断要么代价高昂、有偏且不被推荐，要么完全缺失。我们将推断问题简化为对监督子空间的留出OLS重拟合——这是PLS、监督PCA和线性探针共有的基本操作——并提供了两种基于留出相关性的检验方法：一种是经过Nadeau-Bengio校正的渐近t检验作为快速近似，另一种是功效相当、在结果与预测变量独立且样本行独立同分布条件下具有有限样本有效性的置换检验。留出预测在监督张成空间进行任何正交重构后保持不变，因此诸如varimax这样的可解释基可以继承联合声明，但无法获得逐轴的p值；逐成分的统计声明则来自对PLS提取顺序的固定序列检验。我们在合成几何结构、两个近红外（NIR）化学计量数据集以及交叉验证场景上进行了验证。

    arXiv:2609.36307v1 Announce Type: new  Abstract: Partial Least Squares (PLS) regression extracts a few outcome-aligned directions in a high-dimensional X and is widely used across applied science, but inference on the resulting fit is either expensive, biased and discouraged, or absent. We reduce inference to held-out OLS refits of the supervised subspace, a primitive shared by PLS, supervised PCA, and linear probes, and supply two tests using held-out correlations: a Nadeau-Bengio corrected asymptotic t-test as a fast approximation, and a permutation test with comparable power, finite-sample valid under outcome-predictor independence and iid rows. Held-out predictions are unchanged under any orthogonal rebasing of the supervised span, so an interpretable basis such as varimax inherits the joint claim but not a per-axis p-value; per-component claims come from a fixed-sequence test on the PLS extraction order. We validate on synthetic geometries, two NIR chemometric datasets, and cross-
    
[^43]: MoRE：通过硬件感知的低秩路由扩展专家混合模型

    MoRE: Scaling mixture of experts with hardware-aware low-rank routing

    [https://arxiv.org/abs/2609.36301](https://arxiv.org/abs/2609.36301)

    提出MoRE方法，通过将MoE路由器权重矩阵低秩分解，把路由成本从Θ(Mh)降至O((h+M)r)，在证明可保持路由表达能力与负载均衡的同时，支持Θ(h/r)倍的更多专家，并考虑硬件实际加速。

    

    专家混合层是前沿语言模型的核心组件，而近期的架构正朝着更多、更小的专家方向发展。在这种模式下，标准的线性路由器成为瓶颈：当有 $M$ 个专家和隐藏维度 $h$ 时，其每 token 成本 $\Theta(Mh)$ 在 $M$ 较大时会主导 MoE 层的开销。我们提出 MoRE（秩约简路由的专家混合），它在秩 $r$ 处对路由器权重矩阵进行分解，将路由成本降低至 $O((h+M)r)$。我们证明，当激活专家数量固定时，与 $M$ 呈对数关系的秩足以保证路由的表达能力，且在不计精度因子的意义下该秩下界是必要的。我们还证明，对数秩在高斯记忆模型中能够保持负载均衡，并且在合成电话簿任务上的训练表明低秩不会损害记忆能力。在匹配的激活 FLOPs 下，该分解允许专家数量增加 $\Theta(h/r)$ 倍。为了在实际运行时间中实现这一收益……

    arXiv:2609.36301v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) layers are central to frontier language models, and recent architectures push toward more and smaller experts. In this regime, the standard linear router becomes a bottleneck: with $M$ experts and hidden dimension $h$, its per-token cost $\Theta(Mh)$ dominates the MoE layer once $M$ is large. We introduce MoRE (Mixture of Rank-reduced-routed Experts), which factorizes the router weight matrix at rank $r$ and reduces the routing cost to $O((h + M)r)$. We prove that rank logarithmic in $M$ suffices for routing expressivity when the number of active experts is fixed, and is necessary up to precision factors. We also prove that logarithmic rank preserves load balance in a Gaussian memorization model, and training on a synthetic phonebook task shows that low rank does not hurt memorization. At matched active FLOPs, the factorization allows a factor of $\Theta(h/r)$ more experts. To realize this gain in wall-clock ti
    
[^44]: GNA：面向检索增强多元时间序列预测的细粒度邻居组装

    GNA: Granular Neighbor Assembly for Retrieval-Augmented Multivariate Time-Series Forecasting

    [https://arxiv.org/abs/2609.36281](https://arxiv.org/abs/2609.36281)

    该论文提出GNA，一个用于多元时间序列预测的检索增强层，它在完整窗口和单变量两个粒度上检索并组装相似历史邻居，并通过可学习门控将其与骨干网络预测及持续性预测融合，从而突破了固定长度回看窗口的局限。

    

    深度预测模型从固定长度的回看窗口出发进行预测，而延长窗口带来的收益递减，成本却不断攀升。检索增强方法则向模型展示相似的过去情形是如何延续发展的。检索完整的过去窗口会为每个变量提供同一过去时刻的延续；然而在多元时间序列中，最佳的历史匹配因变量而异。我们提出了GNA（细粒度邻居组装，Granular Neighbor Assembly），这是一个用于预测骨干网络的检索层，它在两个粒度上组装邻居：完整的过去窗口（保持变量之间的连贯性）以及按变量的邻居（每个变量从其自身最匹配的过去中获取未来值）。一个可学习的门控机制会针对每个预测步和每个变量，决定在骨干网络自身预测之外，对这些检索得到的未来与持续性预测的信任程度。候选样本来自一个经过训练、用于预测每个窗口未来的嵌入表示，且检索严格满足因果性：过去的……（原文摘要在此处截断）

    arXiv:2609.36281v1 Announce Type: new  Abstract: Deep forecasters predict from a fixed-length lookback window, and lengthening it gives diminishing returns at a growing cost. Retrieval augmentation instead shows the model how similar past situations continued. Retrieving a whole past window gives every variate the continuation of the same past moment. In multivariate series, however, the best past match differs from variate to variate. We present GNA (Granular Neighbor Assembly), a retrieval layer for forecasting backbones that assembles neighbors at two granularities: whole past windows, which keep the variates coherent, and per-variate neighbors, in which each variate takes its future from its own best-matching past. A learned gate decides, per forecast step and variate, how much to trust these futures against a persistence forecast, next to the backbone's own forecast. Candidates come from an embedding trained to predict each window's future, and retrieval is strictly causal: a past
    
[^45]: 幂律谱下的随机优化：紧致收敛界与洗牌分析

    Stochastic Optimization Under Power-Law Spectra: Tight Bounds and Shuffling Analysis

    [https://arxiv.org/abs/2609.36271](https://arxiv.org/abs/2609.36271)

    本文将幂律谱收敛理论推广至随机梯度下降，并精确证明对各向同性高斯数据而言，单次洗牌的采样策略严格优于交替洗牌和IID采样。

    

    近期的研究已经证明，数据上的幂律谱条件能够为确定性梯度下降提供紧致的收敛界，从而化解了经典指数界与实际观察到的幂律学习曲线之间的矛盾。在本工作中，我们将这一结果扩展到高维机器学习的随机情形。我们提供了两项主要贡献：(1) 我们将幂律谱理论推广到随机梯度下降（SGD），证明相同的谱指数支配着随机动力学；(2) 针对各向同性高斯数据这一基本情形，我们对数据洗牌策略进行了精确分析，推导出精确常数，证明单次洗牌严格优于交替洗牌和IID采样。我们的结果弥合了抽象谱理论与实际随机训练策略选择之间的差距，为数据几何如何驱动优化速度提供了统一的图景。

    arXiv:2609.36271v1 Announce Type: new  Abstract: Recent work has established that power-law spectral conditions on data enable tight convergence bounds for deterministic gradient descent, resolving the conflict between classical exponential bounds and observed power-law learning curves. In this work, we extend this result to the stochastic regime of high-dimensional machine learning. We provide two main contributions: (1) We generalize the power-law spectral theory to Stochastic Gradient Descent (SGD), showing that the same spectral exponents govern stochastic dynamics; (2) For the fundamental case of isotropic Gaussian data, we provide a precise analysis of data shuffling, deriving exact constants that prove Single Shuffle is strictly superior to Flip-Flop and IID sampling. Our results bridge the gap between abstract spectral theory and practical stochastic training choices, offering a unified picture of how data geometry drives optimization speed.
    
[^46]: OTROPE：基于最优传输的大语言模型鲁棒离线策略评估方法

    OTROPE: Optimal Transport-based Robust Off-policy Evaluation for Large Language Models

    [https://arxiv.org/abs/2609.36264](https://arxiv.org/abs/2609.36264)

    提出 OTROPE，一种基于最优传输、无需似然值的大语言模型离线策略评估方法，通过在语义空间对齐行为策略与目标策略样本，实现无需密度比估计和策略建模的双重鲁棒式评估，适用于黑盒大语言模型。

    

    对大语言模型（LLM）进行可靠的评估对其开发与部署至关重要，然而在线评估往往成本高昂、风险较大且难以安全进行。我们研究大语言模型的离线策略评估问题，即利用来自行为模型的有限人工标注数据来评估一个更新的目标大语言模型。这一设定极具挑战性，原因在于标注数据稀缺、行为模型与目标模型之间的分布偏移普遍存在，而且对于黑盒大语言模型而言，其响应的似然值通常不可获得。我们提出了基于最优传输的鲁棒离线策略评估方法（OTROPE），这是一种无需似然值的评估方法，通过最优传输在语义空间中进行分布校正，使有标注的行为策略样本与无标注的目标策略样本相互对齐。OTROPE 将校正后的人工标注残差与代理预测器相结合，形成一种双重鲁棒风格的评估方式，且无需对行为策略建模或进行密度比估计。我们在理论上刻画了……

    arXiv:2609.36264v1 Announce Type: new  Abstract: Reliable evaluation of large language models (LLMs) is essential for their development and deployment, yet is often costly, risky, and difficult to perform safely online. We study off-policy evaluation for LLMs, where limited human-labeled data from a behavior model are used to evaluate a newer target LLM. This setting is challenging because labels are scarce, behavior--target distribution shift is common, and response likelihoods are often unavailable for black-box LLMs. We propose the Optimal Transport-based Robust Off-Policy Evaluation (OTROPE), a likelihood-free evaluation that performs distributional correction in a semantic space via optimal transport to align labeled behavior-policy samples with unlabeled target-policy samples. OTROPE combines corrected human-labeled residuals with proxy predictors, yielding a doubly robust-style evaluation without behavior-policy modeling or density-ratio estimation. We theoretically characterize
    
[^47]: 单次反事实补救的有符号几何：路径上有效性与带符号曲率判据

    The Signed Geometry of One-Shot Recourse: On-Path Validity and the Signed-Curvature Criterion

    [https://arxiv.org/abs/2609.36252](https://arxiv.org/abs/2609.36252)

    该论文证明单次解析式反事实补救能否一步成功由路径曲率 $\kappa=\hat g^\top\nabla^2 f(x)\hat g$ 的符号决定（非负则有效），给出仅凭分数与梯度的规则不可避免存在 $Kd_p^2/\|\nabla f(x)\|$ 量级过冲的下界，并证明在利普希茨曲率下于承诺点评估一次分数即可达到极小极大最优有效性。

    

    闭式（解析式）补救方法将一个被分类器拒绝的用户沿着分类器分数 $f$ 的单位梯度 $\hat g$ 移动承诺距离 $d_p=|f(x)|/\|\nabla f(x)\|$，在该处线性化后的分数恰好降为零。我们研究这一单步操作何时会成功，以及额外的模型查询能带来什么改变。在主阶近似下，该步恰好落在有利一侧当且仅当路径曲率 $\kappa=\hat g^\top\nabla^2 f(x)\,\hat g$ 非负。在80个浅层模型上，被拒用户中一步落在有利侧的比例与 $\kappa\ge0$ 的比例相关系数高达 $r=0.985$，但在 Fashion-MNIST 上前者平均比后者低8.2个百分点。任何仅使用分数值和梯度的规则，都不可能对所有路径曲率以 $K$ 为界的分数都保持有效，除非对其中某些分数产生 $Kd_p^2/\|\nabla f(x)\|$ 量级的过冲。当曲率还是利普希茨连续且步长较短时，在承诺点处对 $f$ 的一次评估即可达到极小极大（minimax）最优。

    arXiv:2609.36252v1 Announce Type: new  Abstract: Closed-form recourse moves a rejected user along the unit gradient $\hat g$ of the classifier score $f$ by the promised distance $d_p=|f(x)|/\|\nabla f(x)\|$, at which the linearized score reaches zero. We ask when this one-shot step succeeds and what additional model queries change. To leading order the step ends on the favorable side exactly when the path curvature $\kappa=\hat g^\top\nabla^2 f(x)\,\hat g$ is nonnegative. Across 80 shallow models, the fraction of rejected users whose step ends there and the fraction with $\kappa\ge0$ correlate at $r=0.985$, although on Fashion-MNIST the first falls below the second by 8.2 points on average. No rule that uses only the score value and gradient can be valid for every score with path curvature bounded by $K$ without overshooting some by order $Kd_p^2/\|\nabla f(x)\|$. When the curvature is also Lipschitz and the step is short, one evaluation of $f$ at the promised point attains the minimax
    
[^48]: 表示学习能否与损失最小化解耦？极化更新给出了答案

    Can Representation Learning Decouple from Loss Minimization? Polar Updates Have an Answer

    [https://arxiv.org/abs/2609.36240](https://arxiv.org/abs/2609.36240)

    该论文证明，在 Muon 极化更新引发的损失平台期与振荡阶段，表示学习并未停止——权重持续移动、特征持续与教师子空间对齐，AGOP 主特征空间甚至在损失下降之前就已精确恢复教师子空间。

    

    arXiv:2609.36240v1 公告类型：新论文 摘要：当训练损失停止改善时，表示学习是否也随之停止？我们针对矩阵 Muon 优化器研究了这一问题，其极化归一化更新的步长由梯度的秩而非范数决定。在稳定边缘附近，全批量 Muon 在教师-学生问题上会进入近似周期为 2 的损失振荡，并持续数千步：周期平均损失保持平稳甚至上升，但权重仍在持续移动，且学到的特征继续与教师子空间对齐。对于线性教师-学生学习的玩具模型，我们推导出显式的周期与对齐公式，以及条件性的平台期与衰减界。对于一个总体平均场 ReLU 模型，我们证明，在给定的维度、初始化和小头部条件下，平均梯度外积（AGOP）的主特征空间会在损失平台期内精确恢复教师子空间，随后损失才下降。在全部 33 个 ReLU、GELU 和 SiLU …

    arXiv:2609.36240v1 Announce Type: new  Abstract: Does representation learning stop when the training loss stops improving? We study this question for matrix Muon, whose polar-normalised updates have a step length set by the gradient's rank rather than its norm. Near the edge of stability, full-batch Muon on teacher-student problems enters approximately period-2 loss oscillations that persist for thousands of steps: the cycle-mean loss stays flat or rises, yet the weights keep moving and the learned features continue to align with the teacher subspace. For linear teacher-student learning toys, we derive explicit cycle and alignment formulas and conditional plateau and decay bounds. For a population mean-field ReLU model, we prove that, under stated dimension, initialisation and small-head conditions, the leading eigenspace of the average gradient outer product (AGOP) recovers the teacher subspace exactly during a loss plateau, before the loss later drops. In all 33 ReLU, GELU and SiLU t
    
[^49]: 单步下一隐变量预测并非世界模型

    One-Step Next-Latent Prediction Is Not a World Model

    [https://arxiv.org/abs/2609.36227](https://arxiv.org/abs/2609.36227)

    该论文从理论上证明单步下一隐变量预测（如LeNEPA）只识别出条件均值而非可展开的世界模型转移核，其多步开环误差随预测时域增长，且在非线性条件均值或非单射观测情形下无法通过复合得到正确的多步预测。

    

    下一隐变量预测拟合一个从当前嵌入到下一嵌入的映射。LeNEPA将这一目标引入时间序列，用LeJEPA的各向同性惩罚取代下一嵌入预测中的停止梯度。世界模型是一个可以展开的转移核，而单步回归所识别的是条件均值，均值仅在特殊情况下才构成转移核。对于线性高斯马尔可夫隐变量，均值转移与新息协方差由单步问题唯一确定，且在预测时域K处的开环平方误差等于前推新息协方差之和的迹；即使单步拟合完全精确，该误差仍随K增长。若条件均值是非线性的，对其复合并不能得到多步条件均值；若观测是马尔可夫状态的非单射函数，无记忆的单步映射无法确定未来观测，而一个短窗口却可以。各向同性惩罚……（摘要原文在此处被截断）

    arXiv:2609.36227v1 Announce Type: cross  Abstract: Next-latent prediction fits a map from the current embedding to the next one. LeNEPA carries this objective to time series, replacing the stop-gradient of next-embedding prediction with the isotropy penalty of LeJEPA. A world model is a transition kernel that can be rolled out. The one-step regression identifies a conditional mean, and a mean is a kernel only in special cases. For a linear-Gaussian Markov latent, the mean transition and the innovation covariance are fixed by the one-step problem, and the open-loop squared error at horizon $K$ equals the trace of the sum of the pushed-forward innovation covariances. That error grows with $K$ after the one-step fit is exact. If the conditional mean is nonlinear, composing it is not the multi-step conditional mean. If the observation is a non-injective function of a Markov state, a memoryless one-step map does not determine future observations, while a short window can. An isotropy penalt
    
[^50]: Copula (连接函数) 活跃子空间 I：一种用于降阶非高斯密度估计的得分协方差方法

    Copula Active Subspaces I: A Score-Covariance Method for Reduced-Order Non-Gaussian Density Estimation

    [https://arxiv.org/abs/2609.36142](https://arxiv.org/abs/2609.36142)

    提出 Copula 活跃子空间（CAS）方法，利用 copula 得分协方差的主特征向量识别非高斯噪声分布中依赖结构的变化方向，从而实现贝叶斯推断中非高斯噪声密度的降阶表示与估计。

    

    在具有非高斯观测噪声的贝叶斯推断问题中，后验分布的准确性完全取决于噪声密度的准确性，而基于梯度的采样器需要该密度及其梯度可以逐点求值——无论是通过显式表达式还是通过代码，且不需要内部求解。我们提出 Copula 活跃子空间（CAS）来表示这种噪声密度。通过逐分量的秩变换将噪声分布的依赖结构隔离在其 copula 中，再通过秩为 r 的降阶仅保留依赖结构发生变化的方向。这些方向是 copula 得分协方差 C := Cov_{π_Z}(∇log c^Z) 的主特征向量，这正是使该降阶成为 copula 活跃子空间的原因。由于当各坐标相互独立时 C 为零，这些方向即为依赖方向，而数据的协方差未必能识别出这些方向。（摘要在此处被截断）

    arXiv:2609.36142v1 Announce Type: cross  Abstract: In Bayesian inference problems with non-Gaussian observation noise, the posterior is only as accurate as the noise density, and gradient-based samplers need that density and its gradient evaluable pointwise, whether from an explicit expression or from code, and without an inner solve. We propose Copula Active Subspaces (CAS) to represent this noise density. A componentwise rank transform isolates the noise law's dependence in its copula, and a rank-$r$ reduction keeps only the directions along which that dependence varies. These directions are the leading eigenvectors of the copula score covariance $\boldsymbol{C} := \mathrm{Cov}_{\pi_{\boldsymbol{Z}}}(\nabla\log c^{Z})$, which is what makes the reduction a copula active subspace. Because $\boldsymbol{C}$ vanishes when the coordinates are independent, these are directions of dependence, which the covariance of the data need not identify. From this construction follow a Gaussian-referen
    
[^51]: 图分割贝叶斯因果森林用于空间异质性处理效应估计

    Graph-Split Bayesian Causal Forest for Spatial Heterogeneous Treatment Effect Estimation

    [https://arxiv.org/abs/2609.36046](https://arxiv.org/abs/2609.36046)

    提出图分割贝叶斯因果森林（GSBCF），通过将图分割贝叶斯加性回归树与贝叶斯因果森林倾向得分回归框架相结合，克服了传统轴对齐分割规则无法建模空间结构的局限，实现了空间异质性处理效应的估计。

    

    在空间观察性研究中，处理分配和结果往往表现出空间依赖模式，且由于已测量和未测量的空间结构混杂因素，处理效应可能在空间和不同亚群之间存在差异。在估计异质性处理效应（HTEs）的同时考虑空间依赖性是空间因果推断的核心任务。因果贝叶斯加性回归树方法是建模和估计异质性处理效应的流行非参数方法。尽管这些方法具有灵活性和不确定性量化的优点，但这些模型中通常采用的轴对齐分割规则并不适合建模空间结构。我们提出了一种空间结构感知的贝叶斯非参数方法，称为图分割贝叶斯因果森林（GSBCF），该方法将图分割贝叶斯加性回归树（GS-BART）与贝叶斯因果森林的倾向得分回归框架相结合，用于空间异质性……

    arXiv:2609.36046v1 Announce Type: cross  Abstract: In spatial observational studies, treatment assignment and outcomes often exhibit spatial dependence patterns, and treatment effects may vary across space and subpopulations due to both measured and unmeasured spatially structured confounders. Accounting for spatial dependence while estimating heterogeneous treatment effects (HTEs) is a central task in spatial causal inference. Causal Bayesian additive regression tree methods are popular nonparametric methods for modeling and estimating HTEs. Despite their flexibility and uncertainty quantification, the axis-aligned split rules often adopted in these models are not suitable for modeling spatial structures. We propose a spatial structure-aware Bayesian nonparametric method, called Graph-Split Bayesian Causal Forest (GSBCF), that integrates graph-split Bayesian additive regression trees (GS-BART) with the Bayesian causal forest propensity-score regression framework for spatial heterogene
    
[^52]: 黎曼流形上的内蕴联想记忆：曲率、容量与涌现模式

    Intrinsic Associative Memory on Riemannian Manifolds: Curvature, Capacity, and Emergent Modes

    [https://arxiv.org/abs/2609.35948](https://arxiv.org/abs/2609.35948)

    该论文在黎曼流形上构建了内蕴的稠密联想记忆，证明曲率从根本上决定记忆的存亡——正曲率可抹除记忆而负曲率会强化记忆，并给出了容量随核重叠概率的标度律，同时揭示了模式重叠能够涌现出新记忆模式的机制。

    

    几何对联想记忆的作用远不止于约束：曲率决定了记忆体记住什么，以及它创造出哪些状态。我们通过将记忆建模为 Epanechnikov 核密度寻模问题，在黎曼流形上构建了内蕴的稠密联想记忆。我们比较了测地线能量与体积校正能量，并证明曲率使二者的行为产生分化。我们证明，测地线记忆总能保留孤立的模式，而校正记忆则服从一个严格的 Ricci 曲率阈值：在高维情形下，正曲率可能抹除记忆，而负曲率则会强化记忆。我们推导出测地线容量的标度律：保留全部模式时容量为 $q_\beta^{-1/2}$，保留典型模式时为 $q_\beta^{-1}$，其中 $q_\beta$ 为成对核重叠概率。我们展示了模式重叠如何“创造”出新的记忆：精心设计的 $N$ 个模式构型可以实现全部 $2^N-1$ 个子集模式，而处于存储阈值处的随机数据则仅产生 P……（摘要在此处被截断）

    arXiv:2609.35948v1 Announce Type: cross  Abstract: Geometry does more than constrain an associative memory: curvature determines what it remembers and which states it creates. We develop intrinsic dense associative memories on Riemannian manifolds by casting memory as Epanechnikov kernel-density mode seeking. We compare geodesic and volume-corrected energies and show that curvature separates their behavior. We prove that geodesic memory always retains an isolated pattern, while corrected memory obeys a sharp Ricci-curvature threshold: positive curvature can erase memories in high dimensions, while negative curvature reinforces them. We derive geodesic capacity scalings of $q_\beta^{-1/2}$ for retaining every pattern and $q_\beta^{-1}$ for a typical one, where $q_\beta$ is the pairwise kernel-overlap probability. We show how overlap \emph{creates} novel memories: designed $N$-pattern configurations realize all $2^N-1$ subset modes, but random data at the storage threshold yield only a P
    
[^53]: FluxLite：面向离散扩散模型的推理时提议控制

    FluxLite: Inference-Time Proposal Control for Discrete Diffusion Models

    [https://arxiv.org/abs/2609.35947](https://arxiv.org/abs/2609.35947)

    FluxLite 提出一个无需训练的轻量级提议控制框架，通过在 Feynman-Kac 势中加入 q_t 加权的图散度项精确补偿稀疏跳跃速率扰动，避免 SMC 权重退化，并实例化为单跳局部重分配（HEU）与非负二次修正两种实用采样器。

    

    对于预训练离散扩散模型和扩散语言模型，许多推理时任务都可以归结为从预训练分布的倾斜版本中抽取样本。Feynman-Kac 序贯蒙特卡洛（SMC）在原理上能够精确完成这一修正，但当提议动力学与倾斜分布不匹配时，其规定的权重常常会退化，从而限制了增加粒子数量所带来的实际收益。我们提出 FluxLite，一个轻量级、无需训练的离散扩散提议控制框架。在由预训练反向速率构成的稀疏有向图上，任何稀疏的跳跃速率扰动都可以通过 Feynman-Kac 势中一个以 q_t 加权的图散度项得到精确补偿；因此目标路径得以保持，而剩余的重加权方差则成为一个局部凸优化目标。我们将这一原理实例化为两个实用的采样器：单跳局部重分配规则（HEU）和一个小型非负二次修正。

    arXiv:2609.35947v1 Announce Type: new  Abstract: Many inference-time tasks for pretrained discrete diffusion models and diffusion language models reduce to drawing samples from a tilted version of the pretrained distribution. Feynman-Kac sequential Monte Carlo (SMC) makes this correction exact in principle, but its prescribed weights routinely degenerate when the proposal dynamics are misaligned with the tilt, capping the practical gains from additional particles. We introduce FluxLite, a lightweight, training-free proposal-control framework for discrete diffusion. On the sparse directed graph of pretrained reverse rates, any sparse jump-rate perturbation can be exactly compensated by a $q_t$-weighted graph-divergence term in the Feynman-Kac potential; the target path is therefore preserved while the residual reweighting variance becomes a local convex objective. We instantiate this principle as two practical samplers: a one-hop local reallocation rule (HEU) and a small nonnegative qua
    
[^54]: 面向分布型结果变量的Wasserstein因果森林

    Wasserstein Causal Forests for Distribution-Valued Outcomes

    [https://arxiv.org/abs/2609.35898](https://arxiv.org/abs/2609.35898)

    本文提出Wasserstein因果森林（WCF）用于处理结果为概率分布的因果推断问题，定义了包含参考距离对比的分布型处理效应，在多数模拟设计中条件分布估计最为准确，并应用于Project STAR项目揭示小班教学对成绩分布（而不仅是均值）的影响。

    

    本文提出了Wasserstein因果森林（WCF），适用于每个研究单元的结果本身是一个概率分布的场景。本研究还定义了基于有限网格变换的平均处理效应和条件平均处理效应，其中包括一种参考距离对比方法，用于考察处理是否使单元级分布向预先设定的基准靠近。模拟实验涵盖了零效应、位置与形状变化、有限重叠、均值相同但分布规律不同、异质性效应、多峰性以及结构性零点等多种情形。结果表明，WCF在大多数实验设计中于条件分布度量上最为准确，并在主要的位置与形状设置中显著改善了参考效应的估计，但在多峰分布设置下其准确性不及森林基线方法。WCF被应用于著名的“Project STAR”项目，揭示出小班教学所改变的不只是均值：它提升了同年级数学成绩……（原文摘要在此处截断）

    arXiv:2609.35898v1 Announce Type: cross  Abstract: This paper proposes Wasserstein Causal Forests (WCF) for settings in which each unit's outcome is itself a probability distribution. This study also defines finite-grid transformed average and conditional average treatment effects, including a reference-distance contrast that asks whether treatment moves unit-level distributions toward a prespecified benchmark. Simulations cover null effects, location and shape changes, limited overlap, equal-mean but different laws, heterogeneous effects, multimodality, and structural zeros. WCF is most accurate on the conditional-law metric in most reported designs and sharply improves reference-effect estimation in the principal location-and-shape settings, but it is less accurate than the forest baselines for multimodal settings. WCF is applied to the famous Project STAR \citep{word1990state}, revealing that small classes alter more than the mean: they raise within-grade mathematics achievement by 
    
[^55]: 从 Pass@K 与 Pass@1 之间的差距中学习

    Learning from the Gap Between Pass@K and Pass@1

    [https://arxiv.org/abs/2609.35793](https://arxiv.org/abs/2609.35793)

    提出 GapFT 方法，通过在 Pass@K 与 Pass@1 的差距（即单样本失败但 K 个样本内可解决的问题）上进行微调，将测试时搜索带来的能力吸收进模型，从而提升单样本解码的性能。

    

    大语言模型越来越多地采用基于可验证奖励的强化学习（RLVR）进行训练。精确的验证器还可以通过从多个样本中挑选出一个通过的响应来支持测试时扩展，而其他部署方式则使用束搜索、自适应采样或工具。我们研究单样本解码——即每个查询只获得一个响应而不进行搜索——以探究搜索中暴露出的行为能否被吸收进模型之中。现有的基于验证响应的后训练方法通常不会区分那些在首次解码时就已经解决的问题与在 K 个样本内才得以恢复的失败问题。在固定预算下，这可能导致训练样例被浪费在重复部署策略已经具备的行为上。我们提出 GapFT，它根据源检查点的单样本结果来选择训练证据，并在 Pass@K 与 Pass@1 之间的差距上进行微调：即策略在单样本上失败但在 K 个样本内能够解决的问题。我们匹配训练样例，

    arXiv:2609.35793v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly trained with reinforcement learning from verifiable rewards (RLVR). An exact verifier can also support test-time scaling by selecting a passing response from multiple samples, while other deployments use beam search, adaptive sampling, or tools. We study single-sample decoding, where each query receives one response without search, to ask whether search-exposed behavior can be absorbed into the model. Existing verified-response post-training recipes do not generally distinguish problems already solved on the first decode from failures recovered within K samples. Under a fixed budget, this can spend examples repeating behavior the deployed policy already has. We introduce GapFT, which selects training evidence by the source checkpoint's single-sample outcome and fine-tunes on the Pass@K-Pass@1 gap: problems the policy fails on one sample but solves within K samples. We match training examples,
    
[^56]: 无服务器Gossip训练的LSTM故障检测器：在NASA C-MAPSS上与联邦学习、本地学习和集中式学习的匹配协议对比

    Serverless gossip training of LSTM failure detectors: A matched-protocol comparison with federated, local and centralized learning on NASA C-MAPSS

    [https://arxiv.org/abs/2609.35792](https://arxiv.org/abs/2609.35792)

    该论文首次在严格匹配协议下量化了无服务器Gossip训练用于LSTM故障检测的效果，证明其在NASA C-MAPSS上可达到与需要中央服务器的联邦学习（FedAvg）几乎相同的F1性能，并明显优于本地独立训练。

    

    工业预测性维护越来越依赖于从分布在不同站点、传感器数据难以集中汇总的设备中进行学习。联邦平均（FedAvg）通过中央聚合服务器解决这一问题；Gossip学习则去除了服务器，但其在受控条件下针对循环（递归）故障检测模型的表现尚未得到测量。我们在NASA C-MAPSS涡扇发动机基准数据集上，针对检测即将发生故障的堆叠LSTM模型，比较了同步环形Gossip训练与FedAvg、隔离的本地训练以及集中式参考方案。所有方法共享同一开源实现、架构、初始化、优化器、数据划分和训练预算，主要评估指标采用每个测试发动机仅使用一个终端窗口的方式，以避免重叠窗口带来的统计相关性。在FD001子集（五个随机种子）上，Gossip方法的终端窗口F1达到89.6 ± 1.3%，FedAvg为89.9 ± 1.1%，本地训练为83.6 ± 6.7%，集中式训练为9……（摘要在此处截断）

    arXiv:2609.35792v1 Announce Type: new  Abstract: Industrial predictive maintenance increasingly depends on learning from equipment spread across sites whose sensor data cannot easily be pooled. Federated averaging (FedAvg) solves this with a central aggregation server; gossip learning removes the server, but its behaviour for recurrent failure-detection models has not been measured under controlled conditions. We compare synchronous ring gossip with FedAvg, isolated local training and a centralized reference for a stacked LSTM that detects imminent failure on the NASA C-MAPSS turbofan benchmark. All methods share one open implementation, architecture, initialization, optimizer, data split and training budget, and the primary endpoint uses one terminal window per test engine to avoid the statistical dependence of overlapping windows. On FD001 (five seeds), gossip reached a terminal-window F1 of 89.6 +/- 1.3%, compared with 89.9 +/- 1.1% for FedAvg, 83.6 +/- 6.7% for local training and 9
    
[^57]: 量化黑盒语言模型的行为尾部

    Quantifying Behavioral Tails in Black-Box Language Models

    [https://arxiv.org/abs/2609.33638](https://arxiv.org/abs/2609.33638)

    RareTrap框架通过代理LLM构建几何感知映射来诱导可复现的提示词分布，并结合序列稀有事件模拟技术，有效估计了黑盒大语言模型发生严重行为的概率。

    

    我们提出了RareTrap，一个用于估计黑盒大语言模型（LLM）中严重行为发生概率的框架。概率估计的一个关键挑战是在输入空间上定义一个可处理的分布。为实现这一目标，RareTrap使用一个代理LLM，并构建了一种几何感知的映射，将低维潜在参考空间映射到其token嵌入空间，从而在输入提示词上诱导出明确且可复现的分布。通过在响应上应用响应级别的性能函数来量化行为严重程度。这使得序列稀有事件模拟成为可能，即将评估集中在逐渐更严重的行为上，同时在诱导的提示词分布下保持概率——否则这种概率将难以测量。在10个开放权重模型和两个前沿模型（GPT-5.4和Claude Sonnet 4.6）上，我们发现RareTrap成功诱导了严重的资源共…（原文摘要在此处截断）

    arXiv:2609.33638v2 Announce Type: replace-cross  Abstract: We introduce RareTrap, a framework for estimating the probability of severe behaviors in black box large language models (LLMs). A key challenge for probability estimation is defining a tractable distribution over the input space. To accomplish that, RareTrap uses a surrogate LLM and constructs a geometry-aware mapping from a lower-dimensional latent reference space into its token-embedding space to induce an explicit and reproducible distribution over input prompts. A response-level performance function is utilized on the response to quantify behavior severity. This enables sequential rare event simulation that concentrates evaluations on progressively more severe behaviors while preserving probability under the induced prompt distribution, which would otherwise be prohibitive to measure. Across 10 open-weight and two frontier models (GPT-5.4 and Claude Sonnet 4.6), we find that RareTrap successfully induces severe resource co
    
[^58]: 哪些自我改进值得信赖？当智能体重用其基准测试时的可靠自我改进

    Which Self-Improvements Should We Trust? Reliable Self-Improvement When Agents Reuse Their Benchmarks

    [https://arxiv.org/abs/2609.33180](https://arxiv.org/abs/2609.33180)

    提出REUSE框架，通过认证式风险控制评估解决递归自我改进中智能体反复重用固定基准测试所导致的自适应过拟合问题，确保经验改进能真实反映任务分布上的群体性提升。

    

    随着递归自我改进（RSI）的迅速发展，可靠的评估对于指导自适应搜索变得至关重要。RSI通常依赖有限的评估资源（如固定基准测试）来决定保留哪些修改以及下一步提出什么方案。然而，当这些有限资源被反复重用时，新的候选方案是基于同一评估集的反馈提出的，因此搜索轨迹可能会自适应地过拟合，经验上的改进可能无法反映智能体在底层任务分布上真正的群体性提升。一些现有方法考虑了多重比较问题，但它们假设候选方案是独立于评估集选择的，因此无法控制这种自适应依赖性。为了解决这一问题，我们提出了REUSE（Risk-controlled Evaluation Under Sequential Evolution，序列演化下的风险控制评估），一个经过认证保证的评估与晋升框架，允许固定的评估集支持重复……

    arXiv:2609.33180v2 Announce Type: replace-cross  Abstract: As recursive self-improvement (RSI) rapidly advances, reliable evaluation becomes critical for guiding adaptive search. RSI typically relies on finite evaluation resources, such as fixed benchmarks, to determine which modifications are retained and what is proposed next. However, when these finite resources are repeatedly reused, new candidates are proposed based on feedback from the same evaluation set, so the search trajectory can adaptively overfit and empirical improvement may not reflect genuine population improvement on the underlying task distribution. Some existing methods account for multiple comparisons but assume that candidates are chosen independently of the evaluation set, and therefore do not control this adaptive dependence. To address this, we propose REUSE (Risk-controlled Evaluation Under Sequential Evolution), a certified evaluation and promotion framework that allows a fixed evaluation set to support repeat
    
[^59]: 拜占庭鲁棒的联邦检索增强生成：基于对齐校准与固定成员保形预测

    Byzantine-Robust Federated RAG via Aligned Calibration and Fixed-Membership Conformal Prediction

    [https://arxiv.org/abs/2609.33037](https://arxiv.org/abs/2609.33037)

    该论文提出一种对拜占庭节点鲁棒的联邦检索增强生成方法，利用“校准与查询两个阶段中诚实节点相同”这一关键观察，通过对齐校准与固定成员保形预测，在部分节点于两个阶段均可能虚假上报评分的情况下，仍能以预设概率保证返回的答案集合包含正确答案。

    

    检索增强生成（RAG）通过查阅相关文档，使语言模型能够更准确地回答问题。许多有价值的文档集合（如医疗记录）因隐私法规而无法集中汇总。联邦RAG将每个文档集合保留在其所有者（即节点）处，各节点根据自身文档对候选答案进行评分，再由中央枢纽汇总这些评分。其中部分节点（称为拜占庭节点）可能已被攻破、发生故障、或被隐藏在文档中的指令误导，从而上报任意评分。保形预测以预先设定的概率返回一个包含正确答案的答案集合，其做法是在校准步骤中对已知答案的问题确定一个截止阈值。一组规模不超过事先声明上限的未知节点，可能在校准阶段和查询阶段均进行虚假上报。现有方法要么假设所有节点都是诚实的，要么仅保护校准步骤。我们观察到，在这两个步骤中诚实节点是同一批节点。因此，中央枢纽……

    arXiv:2609.33037v2 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) lets language models answer questions more accurately by consulting relevant documents. Many valuable collections, such as medical records, cannot be pooled because of privacy rules. Federated RAG leaves each collection with its owner, or node, which scores candidate answers from its own documents; a central hub combines the scores. Some nodes, called Byzantine, may be compromised, faulty, or misled by instructions hidden in documents, and report arbitrary scores. Conformal prediction returns a set containing the correct answer with a chosen probability, using a cutoff set in a calibration step on questions with known answers. An unknown group of nodes, no larger than a declared bound, may misreport both in this step and at query time. Existing methods assume every node is honest or protect only the calibration step. We observe that the honest nodes are the same in both steps. The hub theref
    
[^60]: 基于纸牌的贝叶斯序数回归与序贯偏好引出方法

    Bayesian Deck-of-cards-based Ordinal Regression with Sequential Preference Elicitation

    [https://arxiv.org/abs/2609.23212](https://arxiv.org/abs/2609.23212)

    该论文提出B-DOR，将基于纸牌的序数回归概率化为贝叶斯框架，通过累积链接似然将空白纸牌数量与潜在价值差异关联，并提供哈密顿蒙特卡洛采样和约束凸优化两种推断算法以支持序贯偏好引出。

    

    基于纸牌的序数回归（DOR）从参考备选方案的排序中推断价值函数，其中决策者（DM）在相邻等级之间插入空白纸牌以表达偏好强度。DOR及其随机扩展（SMAA-DOR）将这些回答视为硬约束，从而定义一组兼容的价值函数。我们提出B-DOR，这是DOR的一种概率化重构，其中每对相邻等级产生一个序数观测，即声明的偏好方向和纸牌数量，并通过累积链接似然函数建模，将空白纸牌数量与备选方案之间的潜在价值差异联系起来。论文提出了两种贝叶斯推断算法：BAYES-DOR通过哈密顿蒙特卡洛（HMC）采样整个后验分布；FTRL-DOR通过约束凸优化跟踪最大后验估计。此外，通过多步骤的引出过程，引出……

    arXiv:2609.23212v1 Announce Type: cross  Abstract: The Deck-of-cards-based Ordinal Regression (DOR) infers a value function from a ranking of reference alternatives in which the Decision Maker (DM) inserts blank cards between consecutive levels to express preference intensity. DOR, and its stochastic extension (SMAA-DOR), treat these answers as hard constraints defining a set of compatible value functions. We propose B-DOR, a probabilistic reformulation of DOR in which each pair of adjacent levels yields an ordinal observation, the declared direction and the number of cards, modelled through a cumulative-link likelihood that relates the number of blank cards to the latent value difference between alternatives. Two Bayesian inference algorithms are proposed: BAYES-DOR samples the whole posterior distribution by Hamiltonian Monte Carlo; FTRL-DOR tracks the maximum a posteriori estimate by constrained convex optimization. Moreover, through a multi-step elicitation process, elicitation can
    
[^61]: 集中式序列独裁老虎机中的精确遗憾前沿与外部性调度

    Exact Regret Frontiers and Externality Scheduling in Centralized Serial-Dictatorship Bandits

    [https://arxiv.org/abs/2609.19963](https://arxiv.org/abs/2609.19963)

    该论文精确刻画了集中式序列独裁匹配老虎机中的可达对数遗憾前沿：匹配级 Graves-Lai 约束可归结为有限个成对探索配额并可通过多项式规模的线性规划求解，同时揭示相同的探索配额经不同调度会产生截然不同的遗憾。

    

    在集中式序列独裁匹配老虎机中，探索必须使用完整匹配，因此学习某一个“玩家-臂”对可能会给其他对带来遗憾。我们在已知公共优先级顺序和单位方差高斯奖励的设定下研究这种外部性。我们证明匹配层面的 Graves-Lai 约束可归结为有限多个成对探索配额，并且在“首选分离”的实例上可得到一个多项式规模的边际线性规划。在这些实例上，期望对数遗憾系数的精确可达集为 G(θ)𝒳(θ)，其中 𝒳 为可行的匹配分配集合，G 将分配映射为玩家遗憾。通常的上封闭 Graves-Lai 区域虽然具有相同的帕累托极小边界，却可能严格更大。我们进一步证明，相同的探索配额通过不同的调度方式会产生截然不同的遗憾。最后，我们构造了估计-求解-跟踪策略……（原文摘要在此处截断）

    arXiv:2609.19963v2 Announce Type: cross  Abstract: Exploration in centralized serial-dictatorship matching bandits must use complete matchings, so learning one player-arm pair can impose regret on others. We study this externality under a known common priority order and Gaussian rewards with unit variance. We show that the matching-level Graves-Lai constraints reduce to finitely many pairwise exploration quotas and, at top-choice-separated instances, yield a polynomial-size marginal linear program. At these instances, the exact attainable set of expected logarithmic regret coefficients is $G(\theta)\mathcal{X}(\theta)$, where $\mathcal{X}$ is the feasible matching-allocation set and $G$ maps allocations to player regret. The usual upper-closed Graves-Lai region can be strictly larger despite having the same Pareto-minimal boundary. We further show that identical exploration quotas can induce very different regret through their scheduling. Finally, we construct estimate-solve-track poli
    
[^62]: 表演性隐私：差分隐私何时能最大化效用

    Performative Privacy: When Differential Privacy Maximizes Utility

    [https://arxiv.org/abs/2608.28198](https://arxiv.org/abs/2608.28198)

    该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。

    

    保护隐私的学习通常源于这样一种理念：保护用户数据可以维持信任，从而保持用户参与，进而在长期内提升效用。然而，这一论点迄今为止尚未被形式化。与此同时，表演性学习为研究部署行为会影响其后续观测数据的学习系统提供了一个框架。在本工作中，我们将这两种视角结合起来，提出了“表演性隐私”的概念，即数据泄露会降低未来的用户参与度。我们研究了一个简单模型：智能体反复贡献数据用于均值估计，但当其数据被泄露时可能会退出系统。隐私通过差分隐私机制来实现，从而在估计噪声与未来参与度之间形成权衡。通过对该动态过程的理论研究和数值实验，我们证明了在某些条件下，有限的隐私预算在长期内可以优于非隐私估计。

    arXiv:2608.28198v1 Announce Type: new  Abstract: Privacy-preserving learning is often motivated by the idea that protecting users' data can preserve trust and thus participation, improving utility in the long term. However, this claim has not been formalized so far. In parallel, performative learning provides a framework for studying learning systems whose deployment affects the data they later observe. In this work, we bring these two perspectives together and introduce \emph{performative privacy}, where data leakage reduces future participation. We study a simple model where agents repeatedly contribute data for mean estimation but may leave the system when their data is leaked. Privacy is implemented through differentially private mechanisms, creating a trade-off between estimation noise and future participation. We show, through a theoretical study of the dynamics and numerical experiments, that a finite privacy budget can outperform non-private estimation in the long term when the
    
[^63]: 图生成中的Gromov-Monge流匹配与等变架构

    Gromov-Monge Flow Matching for Equivariant Graph Generation

    [https://arxiv.org/abs/2608.26961](https://arxiv.org/abs/2608.26961)

    本文提出了一种基于Gromov-Monge距离的流匹配方法，通过商空间几何和等变架构实现高效的图生成，并利用松弛和下界解决对齐难题。

    

    图在节点排列下具有不变性，这促使在生成模型中使用排列等变架构。然而，在流匹配中，对称性也可能进入源-目标耦合：一旦图对在节点重标记下进行比较，自然的Wasserstein几何就是图商空间的几何。该空间的欧几里得商度量与Gromov-Monge距离一致，该距离通过最优重标记节点获得。我们从理论上发展了这一视角，表明商耦合可以无额外成本地提升到对齐的代表，并且对称化产生等变流匹配最小化器，包括用于分类端点预测的情况。在实践中，精确的Gromov-Monge对齐是难以处理的，因此我们使用高效的Gromov-Wasserstein型松弛和内部节点对齐的下界来构造小批量耦合，可选地结合外部分配。

    arXiv:2608.26961v1 Announce Type: new  Abstract: Graphs are invariant under node permutations, motivating the use of permutation-equivariant architectures in generative models. In flow matching, however, symmetry may also enter the source--target coupling: once graph pairs are compared up to node relabeling, the natural Wasserstein geometry is that of the graph quotient space. The Euclidean quotient metric of this space coincides with the Gromov--Monge distance, obtained by optimally relabeling the nodes. We develop this perspective theoretically, showing that quotient couplings can be lifted to aligned representatives without additional cost and that symmetrization yields equivariant flow-matching minimizers, including for categorical endpoint prediction. In practice, exact Gromov--Monge alignment is intractable, so we construct minibatch couplings using efficient Gromov--Wasserstein-type relaxations and lower bounds for the inner node alignment, optionally combined with an outer assi
    
[^64]: 比较受污染的约束学习问题

    Comparing Corrupted Constrained Learning Problems

    [https://arxiv.org/abs/2608.25745](https://arxiv.org/abs/2608.25745)

    本文发现经典数据处理不等式在约束学习问题中失效，并提出广义版本以修正这一局限。

    

    arXiv:2608.25745v1 公告类型：新 摘要：统计学中的一个关键结果是数据处理不等式，最初由布莱克威尔（1951）证明，后来由德格鲁特（1962）在统计不确定性的角度下进行了完善。它指出，通过随机修改另一个实验获得的统计实验的贝叶斯风险不能低于原始实验的贝叶斯风险，无论选择何种损失函数或先验。在机器学习中，这一结果支撑了信息瓶颈原理和某些特征学习技术的应用。然而，机器学习问题是约束学习问题：所使用的模型类并不包括所有可测函数。我们提供了一个简单的反例，表明经典数据处理不等式在这种情况下不成立。因此，我们提出了一种广义数据处理不等式，要求联合分布（相对于损失函数）的约束贝叶斯风险满足特定条件。

    arXiv:2608.25745v1 Announce Type: new  Abstract: A key result in statistics is the data processing inequality, originally proved by Blackwell (1951) and later refined by DeGroot (1962) in terms of statistical uncertainty. It states that the Bayes risk of a statistical experiment obtained by stochastically modifying another experiment cannot be lower than the Bayes risk of the original experiment, regardless of the loss function or prior chosen. In machine learning, this result underlies applications such as the information bottleneck principle and some feature learning techniques. However, machine learning problems are constrained learning problems: the model class used does not include all measurable functions. We present a simple counterexample showing that the classical data processing inequality fails to hold in such a setting. Hence, we formulate a generalized data processing inequality, requiring the constrained Bayes risk of a joint distribution (with respect to a loss function 
    
[^65]: 简单排斥过程完全非平衡动力学的表征

    Characterizing Full Nonequilibrium Dynamics of Simple Exclusion Processes

    [https://arxiv.org/abs/2608.25606](https://arxiv.org/abs/2608.25606)

    本文利用变分自回归网络系统表征了从一维到三维的简单排斥过程的非平衡动力学，首次提供了三维有限时间分析，并揭示了一维有限时间动力学活性与TASEP三相稳态组织之间的直接对应关系。

    

    arXiv:2608.25606v1 公告类型：交叉 摘要：简单排斥过程（SEP）是非平衡输运的典范模型，然而其在指数级庞大的构型空间上的时间依赖联合分布的丰富动力学仍然难以处理。在这里，我们利用变分自回归网络系统地表征了从一维到三维的对称（SSEP）、非对称（ASEP）和完全非对称（TASEP）情况下的非平衡动力学。我们首先通过重现一维SSEP的先前有限时间结果和二维SSEP的长时间张量网络结果来验证该方法，然后提供一维和二维中SSEP、ASEP和TASEP更丰富的有限时间动力学，以及三维中的新有限时间分析。具体而言，在一维中，我们揭示了有限时间动力学活性图直接对应于经典的三相TASEP稳态组织，并且在长时间极限下，边界和体效应分别发挥作用。

    arXiv:2608.25606v1 Announce Type: cross  Abstract: The simple exclusion process (SEP) is a paradigmatic model for nonequilibrium transport, yet the rich dynamics of its time-dependent joint distribution over an exponentially large configuration space remain notoriously intractable. Here, we leverage variational autoregressive networks to systematically characterize the nonequilibrium dynamics of symmetric (SSEP), asymmetric (ASEP), and totally asymmetric (TASEP) cases from one to three dimensions. We first validate the approach by reproducing the previous finite-time results for the 1D SSEP and long-time tensor-network results for the 2D SSEP, and then provide richer finite-time dynamics of the SSEP, ASEP, and TASEP in 1D and 2D, and a new finite-time analysis in 3D. Specifically, in 1D, we reveal that finite-time dynamical-activity maps directly correspond to the classical three-phase TASEP steady-state organization, and, in the long-time limit, boundary and bulk effects separately go
    
[^66]: Sinkhorn-Knopp算法的紧致非渐近局部收敛性

    Tight Nonasymptotic Local Convergence of Sinkhorn-Knopp

    [https://arxiv.org/abs/2608.11760](https://arxiv.org/abs/2608.11760)

    本文首次提供了Sinkhorn-Knopp算法的非渐近局部收敛分析，证明了其在特定条件下为多项式时间算法，并显著提升了稠密矩阵缩放问题的复杂度上界。

    

    我们重新审视了矩阵缩放问题中的Sinkhorn-Knopp（SK）算法。尽管已有大量关于SK及其变体全局收敛性的文献，但其局部线性收敛行为仍未被充分理解。我们通过提供SK的首个非渐近局部分析来填补这一空白，该分析匹配了基于现有渐近雅可比论证所获得的收敛速率。我们证明，在特定连通性条件下，SK是双随机矩阵缩放的多项式时间算法。利用所开发的工具，我们展示了SK的局部次优性，并提供了加速变体。最后，对于稠密矩阵，我们将现有的一阶矩阵缩放算法的复杂度从$O(\tfrac{n^{7/3}}{\varepsilon^{2/3}})$改进为$O(\tfrac{n^{9/4}}{\sqrt{\varepsilon}})$。

    arXiv:2608.11760v1 Announce Type: cross  Abstract: We revisit the Sinkhorn-Knopp (SK) algorithm for the matrix scaling problem. Despite extensive literature on the global convergence of SK and its variants, its local linear convergence behavior remains less understood. We address this gap by providing the first nonasymptotic local analysis of SK that matches the rate obtained from existing asymptotic Jacobian-based arguments. We show that under certain connectivity conditions, SK is a polynomial-time algorithm for doubly stochastic matrix scaling. With the developed tools, we showcase the local suboptimality of SK and provide accelerated variants. Finally, for dense matrices, we improve the complexity of existing first-order matrix scaling algorithms from $O(\tfrac{n^{7/3}}{\varepsilon^{2/3}})$ to $O(\tfrac{n^{9/4}}{\sqrt{\varepsilon}})$.
    
[^67]: SR-OPSD：自参照在线策略自蒸馏

    SR-OPSD: Self-Referenced On-Policy Self-Distillation

    [https://arxiv.org/abs/2608.09745](https://arxiv.org/abs/2608.09745)

    提出SR-OPSD方法，通过将自教师模型与冻结的初始策略构建归一化几何目标并最小化正向Rényi散度，实现了对在线策略自蒸馏中密集监督信号的精确可控调节。

    

    在线策略自蒸馏（OPSD）将反馈转化为学生模型生成轨迹上的密集token级监督信号，与依赖稀疏结果奖励的强化学习形成互补。其自教师模型由学生模型的当前参数或指数移动平均参数导出，并以额外的上下文为条件，与学生模型及其采样上下文分布共同演化。修改这一移动目标所带来的收益，取决于目标与学生之间的概率失配如何转化为参数更新。我们提出了自参照在线策略自蒸馏（SR-OPSD），该方法从自教师模型和冻结的初始策略构建归一化几何目标，随后最小化从该目标到学生模型的正向Rényi散度。插值系数控制自教师模型的贡献程度，而Rényi散度的阶数则控制梯度中目标到学生概率比的幂次加权方式。

    arXiv:2608.09745v2 Announce Type: replace-cross  Abstract: On-policy self-distillation (OPSD) converts feedback into dense token-level supervision on student-generated trajectories, complementing reinforcement learning with sparse outcome rewards. Its self-teacher, derived from the student's current or exponentially averaged parameters and conditioned on additional context, evolves alongside the student and its rollout context distribution. The benefit of modifying this moving target depends on how target--student probability mismatches translate into updates. We propose \emph{Self-Referenced On-Policy Self-Distillation (SR-OPSD)}, which constructs a normalized geometric target from the self-teacher and a frozen initial policy, then minimizes the forward R\'enyi divergence from this target to the student. The interpolation coefficient controls the self-teacher's contribution, while the R\'enyi order controls the power weighting of target-to-student probability ratios in the gradient. F
    
[^68]: SMC引导扩散采样的理论保证

    Theoretical Guarantees for SMC-Guided Diffusion Sampling

    [https://arxiv.org/abs/2607.04780](https://arxiv.org/abs/2607.04780)

    本文为SMC引导的扩散采样建立了首个非渐近误差理论保证，刻画了有限粒子涨落以及扩散模型、数值实现与引导机制中各类局部误差经前向平滑核传播的规律。

    

    对预训练扩散模型进行事后条件化可以借助序贯蒙特卡洛（SMC）方法来实现。SMC引导的扩散采样器通过演化一个相互作用的粒子系统，将无条件反向扩散动力学与序贯重加权相结合，以近似目标条件分布。然而，即使处于无穷粒子极限下，由于扩散模型本身、其数值实现以及引导机制中的误差，实际实现的采样器仍可能与理想条件目标存在偏差。我们刻画了这些局部误差如何通过前向平滑核进行传播，该核同时涵盖了反向动力学和剩余的条件信息。由此得到了非渐近误差界，它同时刻画了有限粒子涨落，以及来自初始化、数值积分、得分近似和势函数设计的近似误差。在此过程中，我们扩展了稳定性……

    arXiv:2607.04780v2 Announce Type: replace-cross  Abstract: Post-hoc conditioning of pretrained diffusion models can be addressed using Sequential Monte Carlo (SMC) methods. By evolving an interacting particle system, SMC-guided diffusion samplers combine unconditional reverse-diffusion dynamics with sequential reweighting to approximate conditional distributions. Nevertheless, even in the infinite-particle limit, the implemented sampler may differ from the ideal conditional target because of errors in the diffusion model, its numerical implementation, and the guidance mechanism. We characterize how these local errors propagate through forward-smoothing kernels, which jointly account for the reverse dynamics and the remaining conditioning information. This yields non-asymptotic error bounds that capture both finite-particle fluctuations and approximation errors arising from initialization, numerical integration, score approximation, and potential design. In doing so, we extend stability
    
[^69]: 核梯度流与无穷小梯度提升的函数中心极限定理

    A functional central limit theorem for kernel gradient flow and infinitesimal gradient boosting

    [https://arxiv.org/abs/2606.25494](https://arxiv.org/abs/2606.25494)

    本文在与softmax梯度树基学习器相关的再生核希尔伯特空间中，借助巴拿赫空间上ODE的一般随机扰动分析，为无穷小梯度提升和核梯度流建立了函数中心极限定理，证明重缩放偏差依分布收敛于高斯过程。

    

    在Dombry和Duchamps（2024）对无穷小梯度提升的大样本分析基础上，我们研究了该过程在其确定性极限附近的波动，并建立了一个函数中心极限定理：经过重缩放的偏差依分布收敛于一个高斯过程。该分析在与softmax梯度树基学习器自然关联的再生核希尔伯特空间（RKHS）中进行，其中提升过程被刻画为一个自治常微分方程（ODE）的解。证明依赖于巴拿赫空间中常微分方程的一般随机扰动分析，该分析本身具有独立的研究价值：只要一个向量场序列收敛且满足中心极限定理，其对应的ODE解也同样满足中心极限定理。我们首先在更简单的核梯度流设定中阐述这一扰动方法，在该设定下高斯极限具有显式的刻画……

    arXiv:2606.25494v2 Announce Type: replace-cross  Abstract: Building on the large-sample analysis of infinitesimal gradient boosting (Dombry and Duchamps, 2024), we study the fluctuations of the process around its deterministic limit and establish a functional central limit theorem: the rescaled deviations converge in distribution to a Gaussian process. The analysis is carried out in a reproducing kernel Hilbert space (RKHS) naturally associated with the softmax gradient tree base learner, in which the boosting process is characterized as the solution of an autonomous ordinary differential equation (ODE). The proof rests on a general stochastic perturbation analysis of ODEs in Banach spaces, which is of independent interest: whenever a sequence of vector fields converges and satisfies a central limit theorem, so does the associated ODE solution. We first illustrate this perturbation approach in the simpler setting of kernel gradient flow, where the Gaussian limit admits an explicit char
    
[^70]: Bentkus型渐近e值

    Bentkus-type asymptotic e-values

    [https://arxiv.org/abs/2606.06332](https://arxiv.org/abs/2606.06332)

    本文借助Bentkus的近最优集中不等式框架提出了Bentkus型渐近e值，成功消除了现有渐近e值中的“缺失因子”，从而在事后推断和多重检验中实现了比现有方法更锐利的推断。

    

    渐近e值正逐渐成为渐近p值的一种有力替代方案，特别是在事后推断和多重检验中，因为这些场景下的显著性水平可能是依赖于数据的。然而，现有的渐近e值存在“缺失因子”问题，这是一种缩放上的低效，导致推断过于保守。借助Bentkus在2000年代提出的近最优集中不等式框架，我们引入了Bentkus型渐近e值，并证明其成功消除了缺失因子。我们还从理论和实证两方面证明，Bentkus型e值始终能提供比现有替代方法更锐利的推断，从而带来更紧凑的事后置信区间以及多重检验程序中更高的拒绝率。

    arXiv:2606.06332v2 Announce Type: replace-cross  Abstract: Asymptotic e-values are emerging as a powerful alternative to asymptotic p-values, particularly in post-hoc inference and multiple testing, where significance levels may be data-dependent. Existing asymptotic e-values, however, suffer from the ``missing factor,'' a scaling inefficiency resulting in overly conservative inference. Drawing on the framework of near-optimal concentration inequalities developed by Bentkus in the 2000s, we introduce Bentkus-type asymptotic e-values and prove that they successfully eliminate the missing factor. We also demonstrate both theoretically and empirically that Bentkus-type e-values consistently deliver sharper inference than existing alternatives, leading to tighter post-hoc confidence intervals and higher rejection rates in multiple testing procedures.
    
[^71]: 基于采样的推理：在决策点处切割

    Reasoning with Sampling: Cutting at Decision Points

    [https://arxiv.org/abs/2605.30327](https://arxiv.org/abs/2605.30327)

    该研究表明从基础模型的幂分布中采样即可达到媲美强化学习训练的推理能力，并提出应在推理轨迹中的关键决策点处进行切割重采样，以实现高效的混合采样。

    

    前沿推理模型是通过强化学习对基础语言模型进行后训练而得到的。最近的研究对这一做法提出了挑战，表明从基础模型分布的锐化版本（即所谓的幂分布）中进行采样，无需额外训练、精选数据集或验证器，即可引发相当水平的推理能力。然而，要使这一方法实用化，需要能够高效地从幂分布中采样。采样器需要“混合”到幂分布，这要求在目标分布的众数之间移动；直观地说，例如尝试不同的推理策略。先前工作提出的采样器反复地在当前推理轨迹中均匀随机地选择一个“切割”位置，并从该位置起重新采样后缀。然而，推理轨迹通常只包含少数几个关键性的决策（例如证明策略或算法的选择），我们观察到均匀随机的切割方式并不理想（摘要原文在此处截断）。

    arXiv:2605.30327v2 Announce Type: replace-cross  Abstract: Frontier reasoning models are produced by post-training base language models with reinforcement learning. Recent work has challenged this by showing that sampling from a sharpened version of the base model's distribution, a so-called power distribution, elicits comparable reasoning without additional training, curated datasets, or verifiers. However, making this method practical requires efficiently sampling from the power distribution. A sampler needs to "mix" to the power distribution, which necessitates moving between modes of the target distribution; intuitively, e.g., trying different reasoning strategies. The samplers proposed in prior works repeatedly select a "cut" position in the current reasoning trace uniformly at random and resample the suffix from that position onward. However, reasoning traces typically contain a few consequential decisions (e.g., the choice of proof strategy or algorithm), and we observe that a u
    
[^72]: 认证自适应刷新：联邦保形RAG的任意时刻有效监控

    Certified Adaptive Refresh: Anytime-Valid Monitoring for Federated Conformal RAG

    [https://arxiv.org/abs/2605.29139](https://arxiv.org/abs/2605.29139)

    提出Anytime-FC-RAG框架，通过三条可审计规则，使联邦保形RAG系统在持续检查与反复模型升级的情况下仍能实现任意时刻有效、经过认证的漏报率监控。

    

    基于检索增强生成（RAG）构建的问答服务——即语言模型根据检索到的文档进行回答——需要被持续检查并反复升级，因此其可靠性保证必须在这两种情况下依然成立。我们研究联邦保形RAG：持有私有语料库的节点使用共享语言模型对候选答案进行评分，并将压缩后的评分发送到中心节点，由中心节点返回答案集合；一次“漏报”即遗漏了真实答案。我们将监控问题形式化为一个序贯检验：当漏报率超过经过认证的界限（即容许值）时发出警报。但保形证书只覆盖某一个冻结配置在事先固定的某一次查看，因此无法提供任意时刻有效的警报；为某一模型校准的阈值对其替代模型不提供任何保证；而以完整错误水平对每次升级重新认证会使失败不断累积。我们提出Anytime-FC-RAG，包含三条可审计的规则：在每个配置的全新……（摘要在此处被截断）

    arXiv:2605.29139v2 Announce Type: replace-cross  Abstract: Question-answering services built on retrieval-augmented generation (RAG), in which a language model answers from retrieved documents, are inspected continuously and upgraded repeatedly, so their reliability guarantee must survive both. We study federated conformal RAG: nodes holding private corpora score candidate answers with a shared language model and send compressed scores to a hub that returns an answer set; a miss omits the true answer. We formulate monitoring as a sequential test: alarm when misses exceed a certified bound on the miss rate (the allowance). But a conformal certificate covers one frozen configuration at one look fixed in advance, so it gives no anytime-valid alarm; a threshold calibrated for one model certifies nothing about its replacement; and re-certifying each upgrade at full error level compounds failures. We propose Anytime-FC-RAG with three auditable rules: pick each configuration before its fresh 
    
[^73]: Dropout 普适性：混沌边缘处的标度律与最优调度

    Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos

    [https://arxiv.org/abs/2605.21648](https://arxiv.org/abs/2605.21648)

    该论文建立了混沌边缘处 dropout 的平均场理论，揭示出不同激活函数的普适类与标度律，并证明将 dropout 随深度调度且集中于靠近输入层的位置能最大化正则化收益，在 MLP 等模型上表现最为一致。

    

    我们探究将 dropout 作为静态超参数的标准处理方式是否最优，或者通过让其随网络深度变化是否能提升其效用。为此，我们发展了混沌边缘附近 dropout 的平均场理论，识别出平滑激活函数与折角（kinked）激活函数所属的不同普适类，并给出相应的标度指数。由此得到的信号传播理论，在以最大化 dropout 所提供的正则化效果为约束的条件下，启发我们应将 dropout 集中施加在靠近输入层的位置。在视觉、语音和金融时间序列上的实验表明，该方法在多层感知机（MLP）中带来最一致的收益，而在 Transformer 中收益较小。

    arXiv:2605.21648v3 Announce Type: replace  Abstract: We ask whether the standard treatment of dropout as a static hyperparameter is optimal, or whether its utility can be improved by letting it vary over depth. We answer this by developing a mean-field theory of dropout near the edge of chaos, identifying distinct universality classes for smooth and kinked activations, together with their scaling exponents. The resulting propagation theory, constrained by maximizing the regularization delivered by dropout, motivates concentrating dropout near the input. Experiments on vision, speech and financial time series show gains most consistently in MLPs, with smaller gains in Transformers.
    
[^74]: 使用相关噪声DP-SGD训练的Kolmogorov-Arnold网络的优化风险界

    Optimization Risk Bounds for Kolmogorov-Arnold Networks Trained by DP-SGD with Correlated Noise

    [https://arxiv.org/abs/2605.12648](https://arxiv.org/abs/2605.12648)

    本文首次为使用时间相关噪声DP-SGD训练的两层Kolmogorov-Arnold网络（KAN）建立了优化风险界，并显式刻画了其对时间相关性、裁剪、小批量采样和网络宽度的依赖关系。

    

    对于采用时间相关噪声的差分隐私随机梯度下降（DP-SGD）的理论理解仍然有限，尤其是在非凸神经网络训练方面。作为第一步，我们研究了两层Kolmogorov-Arnold网络（KANs），这是一种近期提出的、具有可学习样条边函数的架构。我们在该设定下首次建立了具有相关噪声的裁剪小批量DP-SGD的优化风险界，并显式刻画了其对时间相关性、裁剪、小批量采样和网络宽度的依赖。现有论证方法失效的原因有三：时间依赖性破坏了条件中心化步骤；投影阻碍了相关扰动在跨迭代过程中的相互抵消；主动裁剪破坏了经验梯度结构。我们通过平移动态与辅助动态、加权经验损失以及高概率局部化论证来解决这些问题。

    arXiv:2605.12648v2 Announce Type: replace  Abstract: The theoretical understanding of differentially private stochastic gradient descent (DP-SGD) with temporally correlated noise remains limited, particularly for non-convex neural network training. As a first step, we study two-layer Kolmogorov-Arnold Networks (KANs), a recently introduced architecture with learnable spline-based edge functions. We establish the first optimization risk bounds for clipped mini-batch DP-SGD with correlated noise in this setting, with explicit dependence on temporal correlation, clipping, mini-batch sampling, and network width. Existing arguments fail for three reasons: temporal dependence breaks the conditional-centering step; projection obstructs the cross-iteration cancellation of correlated perturbations; and active clipping breaks the empirical-gradient structure. We address these issues through shifted and auxiliary dynamics, a weighted empirical loss, and a high-probability localization argument. O
    
[^75]: 经验贝叶斯1比特矩阵补全

    Empirical Bayes 1-bit matrix completion

    [https://arxiv.org/abs/2605.09509](https://arxiv.org/abs/2605.09509)

    本文提出了一种受Efron-Morris估计量启发的经验贝叶斯1比特矩阵补全方法，该方法利用二值矩阵的低秩结构并将奇异值向零收缩，实现了具有竞争力的预测精度和良好的预测校准性。

    

    预测二值矩阵中未观测元素的问题，被称为1比特矩阵补全，在推荐系统等领域有着广泛的应用。在本研究中，我们受Efron-Morris估计量（一种将奇异值向零收缩的James-Stein估计量的矩阵推广形式）的启发，开发了一种用于1比特矩阵补全的经验贝叶斯方法。所提出的方法利用了二值矩阵的潜在低秩结构，并与多维项目反应理论相呼应。模拟研究和实际数据应用表明，所提出的方法实现了具有竞争力的预测精度和良好的预测校准性。

    arXiv:2605.09509v2 Announce Type: replace-cross  Abstract: The problem of predicting unobserved entries in a binary matrix, known as 1-bit matrix completion, has found diverse applications in fields such as recommendation systems. In this study, we develop an empirical Bayes method for 1-bit matrix completion motivated by the Efron--Morris estimator, a matrix generalization of the James--Stein estimator that shrinks singular values toward zero. The proposed method exploits the underlying low-rank structure of binary matrices, drawing parallels with multidimensional item response theory. Simulation studies and real-data applications demonstrate that the proposed method achieves competitive predictive accuracy and favorable predictive calibration.
    
[^76]: 基于路径策略梯度的非短视主动特征获取

    Non-Myopic Active Feature Acquisition via Pathwise Policy Gradients

    [https://arxiv.org/abs/2605.05511](https://arxiv.org/abs/2605.05511)

    该论文提出对特征获取过程进行连续松弛并结合直通式前向模拟，实现了贯穿完整获取轨迹的非短视路径策略梯度，从而对主动特征获取策略进行低方差的端到端优化。

    

    主动特征获取（AFA）考虑这样一类预测问题：特征的获取成本高昂，学习者需要自适应地决定针对每个实例获取哪些特征值，以及何时停止获取并进行预测。在本文中，我们引入了获取过程的一种连续松弛方法，使得非短视的路径策略梯度（NM-PPG）能够贯穿整个获取轨迹进行计算，从而避免了标准得分函数策略梯度的高方差问题，同时允许对获取策略进行端到端优化。为了使训练与部署阶段更好地保持一致，我们开发了一种直通式前向模拟方法，在前向传播中遵循离散的特征获取行为，同时通过相应的软松弛进行反向传播。我们推导了该梯度估计器方差的一个针对AFA的平均情形上界，该上界刻画了估计器的不稳定性，并启发了分阶段温度锐化策略。在合成数据集与（原文摘要在此处截断）上的实验……

    arXiv:2605.05511v2 Announce Type: replace  Abstract: Active feature acquisition (AFA) considers prediction problems in which features are costly to obtain and the learner adaptively decides which feature values to acquire for each instance and when to stop and predict. In this paper, we introduce a continuous relaxation of the acquisition process that enables non-myopic pathwise policy gradients (NM-PPG) through the full acquisition trajectory, avoiding the high variance of standard score-function policy gradients while allowing end-to-end optimization of the acquisition policy. To better align training with deployment, we develop a straight-through rollout that follows discrete feature acquisitions in the forward pass while backpropagating through the corresponding soft relaxation. We derive an AFA-specific average-case upper bound on the variance of this gradient estimator, which characterizes instability and motivates staged temperature sharpening. Experiments on both synthetic and 
    
[^77]: 基于切片势能的摊销最优传输

    Amortized Optimal Transport from Sliced Potentials

    [https://arxiv.org/abs/2604.15114](https://arxiv.org/abs/2604.15114)

    本文提出基于切片最优传输势能的两种摊销方法（回归式RA-OT和目标式OA-OT），可高效解决多对测度之间的重复最优传输问题。

    

    我们提出了一种新颖的摊销优化方法，通过利用从切片最优传输中导出的Kantorovich势能，来预测多对测度之间的最优传输（OT）方案。我们引入了两种摊销策略：基于回归的摊销（RA-OT）和基于目标的摊销（OA-OT）。在RA-OT中，我们构建了一个函数回归模型，将原始OT问题的Kantorovich势能作为响应变量，将从切片OT获得的势能作为预测变量，并通过最小二乘法估计这些模型。在OA-OT中，我们通过优化Kantorovich对偶目标来估计函数模型的参数。在这两种方法中，预测的OT方案随后均从估计的势能中恢复得到。作为摊销OT方法，RA-OT和OA-OT都能通过重用从先前实例中学习到的信息，高效地解决跨不同测度对的重复OT问题，从而快速求解。

    arXiv:2604.15114v2 Announce Type: replace-cross  Abstract: We propose a novel amortized optimization method for predicting optimal transport (OT) plans across multiple pairs of measures by leveraging Kantorovich potentials derived from sliced OT. We introduce two amortization strategies: regression-based amortization (RA-OT) and objective-based amortization (OA-OT). In RA-OT, we formulate a functional regression model that treats Kantorovich potentials from the original OT problem as responses and those obtained from sliced OT as predictors, and estimate these models via least-squares methods. In OA-OT, we estimate the parameters of the functional model by optimizing the Kantorovich dual objective. In both approaches, the predicted OT plan is subsequently recovered from the estimated potentials. As amortized OT methods, both RA-OT and OA-OT enable efficient solutions to repeated OT problems across different measure pairs by reusing information learned from prior instances to rapidly ap
    
[^78]: PAC-CF：在LLM引导搜索中校准不可逆前沿剪枝

    PAC-CF: Calibrating Irreversible Frontier Pruning in LLM-Guided Search

    [https://arxiv.org/abs/2604.14345](https://arxiv.org/abs/2604.14345)

    提出PAC-CF方法，将LLM引导搜索中的前沿剪枝形式化为具有PAC保证的保形决策问题，通过Native-Trace路径校准得到保形边际来过滤候选，避免因不可约评估偏差误删所有可通向有效解的分支，从而在多种领域和预算下提升搜索效用。

    

    LLM引导的搜索通常通过基于评估器分数对top-$K$候选进行排序和剪枝来解决复杂任务。然而，即使采用重复采样等流行方法来降低方差，不可约偏差依然存在。因此，剪枝可能会误删所有能够到达有效解的后续分支。在本文中，我们提出了可能近似正确保形过滤（PAC-CF），它将树剪枝形式化为一个具有PAC保证的决策问题。理论分析阐明了不可约偏差如何降低用于认证性淘汰的分数区分度。Native-Trace路径校准从保留的Native轨迹上验证器有效后续分支的分数差距中推导出一个保形边际。在部署阶段，PAC-CF在直接的分数差距过滤规则中使用这一校准后的边际。在多个领域和最先进的控制器上，PAC-CF在各种预算下均提升了效用。

    arXiv:2604.14345v4 Announce Type: replace-cross  Abstract: LLM-guided search is usually adopted to solve complex tasks by ranking and pruning top-$K$ candidates based on evaluator scores. However, irreducible bias still exists even if popular methods, such as repeated sampling, are applied to reduce variance. Consequently, pruning may remove every continuation that can reach a valid solution. In this paper, we propose Probably Approximately Correct Conformal Filtering (PAC-CF), which formulates tree pruning as a PAC-guaranteed decision problem. Theoretical analysis establishes how irreducible bias reduces the score separation for certified elimination. Native-Trace path calibration derives a conformal margin from the score deficit of verifier-valid continuations on held-out Native traces. During deployment, PAC-CF uses this calibrated margin in a direct score-gap filtering rule. Across diverse domains and state-of-the-art controllers, PAC-CF improves utility at various budgets while re
    
[^79]: 扩散采样的查询下界

    Query Lower Bounds for Diffusion Sampling

    [https://arxiv.org/abs/2604.10857](https://arxiv.org/abs/2604.10857)

    本文首次建立了扩散采样的分数查询下界，证明在多项式精度分数估计下任何采样算法至少需要 $\widetilde{\Omega}(\sqrt{d})$ 次自适应分数查询，从而从信息论角度正式解释了实践中多尺度噪声调度不可或缺的原因。

    

    扩散模型通过迭代查询学习到的分数估计来生成样本。目前快速增长的大量文献专注于通过最小化分数评估次数来加速采样，然而这种加速的信息论极限仍不清楚。在这项工作中，我们建立了扩散采样的首个分数查询下界。我们证明，对于 $d$ 维分布，在给定多项式精度 $\varepsilon=d^{-O(1)}$（在任何 $L^p$ 意义下）的分数估计访问权限的情况下，任何采样算法都需要 $\widetilde{\Omega}(\sqrt{d})$ 次自适应分数查询。特别地，我们的证明表明，在任何多项式总查询预算内，成功的采样需要搜索 $\widetilde{\Omega}(\sqrt{d})$ 个不同的噪声水平，为实践中为什么需要多尺度噪声调度提供了正式的理论解释。

    arXiv:2604.10857v2 Announce Type: replace-cross  Abstract: Diffusion models generate samples by iteratively querying learned score estimates. A rapidly growing literature focuses on accelerating sampling by minimizing the number of score evaluations, yet the information-theoretic limits of such acceleration remain unclear.   In this work, we establish the first score query lower bounds for diffusion sampling. We prove that for $d$-dimensional distributions, given access to score estimates with polynomial accuracy $\varepsilon=d^{-O(1)}$ (in any $L^p$ sense), any sampling algorithm requires $\widetilde{\Omega}(\sqrt{d})$ adaptive score queries. In particular, our proof shows that, within any polynomial total-query budget, successful sampling requires searching over $\widetilde{\Omega}(\sqrt{d})$ distinct noise levels, providing a formal explanation for why multiscale noise schedules are necessary in practice.
    
[^80]: 学习再污染：噪声分布无关的自监督图像去噪

    Learning to Recorrupt: Noise Distribution Agnostic Self-Supervised Image Denoising

    [https://arxiv.org/abs/2603.25869](https://arxiv.org/abs/2603.25869)

    提出了一种自监督去噪框架 L2R，通过可学习的再污染器与去噪器以极小极大鞍点目标联合优化，无需精确了解噪声分布即可在多种非常规和重尾噪声分布下实现最先进的去噪性能。

    

    自监督图像去噪方法传统上依赖于架构约束、伪配对构建或专门的损失函数来避免平凡的恒等映射。其中，诸如 Noisier2Noise 或 R2R 等方法通过向含噪图像添加合成噪声来构建训练对。虽然有效，但这些基于再污染的方法需要对噪声分布有精确的了解，而这通常是不可获得的。我们提出了 Learning to Recorrupt（L2R），这是一个不需要精确了解噪声分布的自监督框架。我们的方法引入了一个可学习的再污染器，通过极小极大鞍点目标与去噪器联合优化。所提出的方法在无需噪声分布先验知识的方法中，对于非常规和重尾噪声分布（如对数伽马分布和拉普拉斯分布）以及空间相关噪声等情况，取得了最先进的性能。

    arXiv:2603.25869v2 Announce Type: replace-cross  Abstract: Self-supervised image denoising methods have traditionally relied on architectural constraints, pseudo-pair constructions, or specialized loss functions to avoid the trivial identity mapping. Among these, approaches such as Noisier2Noise or R2R create training pairs by adding synthetic noise to noisy images. While effective, these recorruption-based approaches require precise knowledge of the noise distribution, which is often unavailable. We present Learning to Recorrupt (L2R), a self-supervised framework that does not require exact knowledge of the noise distribution. Our method introduces a learnable recorruptor jointly optimized with the denoiser through a min--max saddle-point objective. The proposed method achieves state-of-the-art performance among methods without prior knowledge of the noise distribution across unconventional and heavy-tailed noise distributions, such as log-gamma and Laplace, as well as spatially corre
    
[^81]: 无需交叉拟合的多重依赖去偏机器学习方法

    Cross-Fitting-Free Debiased Machine Learning with Multiway Dependence

    [https://arxiv.org/abs/2602.11333](https://arxiv.org/abs/2602.11333)

    本文提出了一种无需交叉拟合的去偏机器学习方法，通过结合Neyman正交矩条件和局部化经验过程，在多重聚类依赖下实现有效的渐近推断。

    

    arXiv:2602.11333v3 公告类型：替换 摘要：本文针对广义矩估计（GMM）模型中具有一般多重聚类依赖的两步去偏机器学习（DML）估计量，开发了一种渐近理论，且不依赖交叉拟合。虽然交叉拟合被广泛使用，但当第一阶段学习器复杂且有效样本量由独立聚类数量决定时，它在统计上可能低效且计算负担沉重。我们证明，通过结合Neyman正交矩条件和基于局部化的经验过程方法，可以在不进行样本分割的情况下实现有效推断，并允许任意数量的聚类维度。结果表明，在多重聚类依赖下，所得的去偏GMM估计量具有渐近线性和渐近正态性。本文的一个核心技术贡献是为一般类别推导出新的全局和局部极大不等式。

    arXiv:2602.11333v3 Announce Type: replace  Abstract: This paper develops an asymptotic theory for two-step debiased machine learning (DML) estimators in generalised method of moments (GMM) models with general multiway clustered dependence, without relying on cross-fitting. While cross-fitting is commonly employed, it can be statistically inefficient and computationally burdensome when first-stage learners are complex and the effective sample size is governed by the number of independent clusters. We show that valid inference can be achieved without sample splitting by combining Neyman-orthogonal moment conditions with a localisation-based empirical process approach, allowing for an arbitrary number of clustering dimensions. The resulting debiased GMM estimators are shown to be asymptotically linear and asymptotically normal under multiway clustered dependence. A central technical contribution of the paper is the derivation of novel global and local maximal inequalities for general clas
    
[^82]: 从种子到语义：测量确定性扩散模型中的语义可达性

    From Seeds to Semantics: Measuring Semantic Accessibility in Deterministic Diffusion Models

    [https://arxiv.org/abs/2602.06155](https://arxiv.org/abs/2602.06155)

    该论文提出“语义可达性”这一概念，通过在DDIM确定性采样轨迹的多个位置训练探针分类器，量化最终图像的语义信息（如类别标签和属性）能从初始噪声种子和中间状态中被提取预测的程度。

    

    扩散模型通过一系列学习到的去噪步骤生成样本，近期的研究已经探讨了语义结构如何在这一采样过程中逐步显现。我们在确定性采样器中通过测量“语义可达性”来研究这一问题：即关于最终语义属性（例如图像的类别标签或属性）的信息，有多少能够从产生该样本的轨迹上的种子及中间状态中提取出来。使用DDIM采样（其中每个初始噪声种子唯一确定一条轨迹和最终图像），我们在轨迹上的多个位置分别训练独立的分类器（探针）来预测最终图像的语义属性，并使用top-1准确率和归一化互信息来衡量该属性在给定点的状态下可被预测的程度。在MNIST、Fashion-MNIST、CIFAR-10和CelebA数据集上，类别标签和图像属性可以从种子及早期状态中以高于随机水平的方式被预测（摘要在此处截断）。

    arXiv:2602.06155v2 Announce Type: replace  Abstract: Diffusion models generate samples through a sequence of learned denoising steps, and recent work has studied how semantic structure appears along this sampling process. We study this question in deterministic samplers by measuring semantic accessibility: how much information about a final semantic property, such as an image class label or attribute, can be extracted from the seed and intermediate states along the trajectory that produces the sample. Using DDIM sampling, for which each initial noise seed determines a unique trajectory and final image, we train separate classifiers (probes) at several points along the trajectory to predict a semantic property of the final image. We measure how well such a property can be predicted from the state at a given point using top-1 accuracy and normalized mutual information. Across MNIST, Fashion-MNIST, CIFAR-10, and CelebA, class labels and image attributes can be predicted above chance from 
    
[^83]: 论非齐次、弱依赖复杂网络动力系统中超参数估计的非渐近标度保证

    On the Nonasymptotic Scaling Guarantee of Hyperparameter Estimation in Inhomogeneous, Weakly-Dependent Complex Network Dynamical Systems

    [https://arxiv.org/abs/2601.15603](https://arxiv.org/abs/2601.15603)

    本文从测度输运视角提出了基于均值型观测的超参数估计理论框架，并为非齐次、弱依赖复杂网络动力系统建立了超参数估计偏差关于网络规模大小的非渐近界，证明在固定观测时长和通用优化算法下估计随网络增大依然可靠。

    

    层次贝叶斯模型通过将参数建模为从由超参数控制的分布中抽取的样本，越来越多地被应用于大规模、非齐次的复杂网络动力系统中。然而，随着网络规模的增长，这些估计量的理论保证一直缺失。一个关键问题是，超参数估计在更大规模网络下可能发散，从而损害模型的可靠性。通过从测度输运的角度刻画系统的演化，我们提出了一个基于均值型观测（这类观测在许多科学应用中普遍存在）的超参数估计理论框架。我们的主要贡献是建立了非齐次复杂网络动力系统中超参数估计偏差关于网络规模大小的非渐近界，该界在固定观测时长内对一大类通用优化算法均成立。对于系统……（摘要截断）

    arXiv:2601.15603v2 Announce Type: replace-cross  Abstract: Hierarchical Bayesian models are increasingly used in large, inhomogeneous complex network dynamical systems by modeling parameters as draws from a hyperparameter-governed distribution. However, theoretical guarantees for these estimates as the network population size grows have been lacking. A critical concern is that hyperparameter estimation may diverge for larger networks, undermining the model's reliability. Formulating the system's evolution in a measure transport perspective, we propose a theoretical framework for estimating hyperparameters with mean-type observations, which are prevalent in many scientific applications. Our primary contribution is a nonasymptotic bound for the deviation of estimate of hyperparameters in inhomogeneous complex network dynamical systems with respect to network population size, which is established for a general family of optimization algorithms within a fixed observation duration. For syst
    
[^84]: 持久三状态消息传递

    Persistent Tri-State Message Passing

    [https://arxiv.org/abs/2601.01207](https://arxiv.org/abs/2601.01207)

    本文提出持久三状态消息传递（P3MP），理论分析了边角色跨层持久性对权重平均的影响，推导出局部权重与共享权重预激活排序发生逆转的条件，并证明共享权重仅保留1/K的目标状态反馈。

    

    在随机消息传递中，一条边的采样角色会改变后续层中用于计算自适应权重的节点状态。因此，权重平均取决于边角色是跨层持久保持还是在每一层重新采样。我们在持久三状态消息传递（P3MP）中研究了这种依赖关系，该方法将持久的加性、减性和非激活三种角色与根据每个样本的端点状态计算的权重相结合。对于两层情形，我们将目标状态反馈与权重源协方差分离，并推导出局部权重与共享权重逆转其预激活排序的条件。对于K个样本，共享权重仅保留1/K比例的目标状态反馈。对于仅基于源的评分器，该交互作用为零。在样本之间置换完整权重向量可以在保持其经验分布的同时量化相同的角色-状态依赖性。精确枚举与仅特征检查点测量结果与理论相符（原文在此截断）。

    arXiv:2601.01207v2 Announce Type: replace  Abstract: In stochastic message passing, an edge's sampled role changes the node states used to compute adaptive weights at later layers. Weight averaging therefore depends on whether edge roles persist across layers or are resampled at each layer. We study this dependence in Persistent Tri-State Message Passing (P3MP), which combines persistent additive, subtractive, and inactive roles with weights computed from each sample's endpoint states. For two layers, we separate target-state feedback from weight-source covariance and derive the condition under which local and shared weights reverse their preactivation ordering. Shared weights retain a $1/K$ fraction of the feedback for $K$ samples. The interaction is zero for a source-only scorer. Permuting complete weight vectors between samples quantifies the same role-state dependence while preserving their empirical distribution. Exact enumeration and feature-only checkpoint measurements agree wit
    
[^85]: 高维偏最小二乘法：谱分析与基本局限性

    High-Dimensional Partial Least Squares: Spectral Analysis and Fundamental Limitations

    [https://arxiv.org/abs/2512.15684](https://arxiv.org/abs/2512.15684)

    本文利用随机矩阵理论对高维偏最小二乘法（PLS-SVD）进行谱分析，首次定量刻画了估计潜在方向与真实方向的对齐程度，从而解释了该方法的重构性能并揭示了其表现反直觉或失效的基本局限。

    

    偏最小二乘法（PLS）是一种广泛使用的数据整合方法，旨在从成对的高维数据集中提取共享的潜在成分。尽管该方法在实践中已成功应用数十年，但对其在高维情形下行为的精确理论理解仍然有限。在本文中，我们研究了一种数据整合模型，其中两个高维数据矩阵共享一个低秩的公共潜在结构，同时各自包含个体特有的成分。我们利用随机矩阵理论的工具分析了相关交叉协方差矩阵的奇异向量，并推导出估计的潜在方向与真实潜在方向之间对齐程度的渐近刻画。这些结果为基于奇异值分解的偏最小二乘法变体（PLS-SVD）的重构性能提供了定量解释，并识别出该方法表现出反直觉或受限行为的区域。

    arXiv:2512.15684v2 Announce Type: replace-cross  Abstract: Partial Least Squares (PLS) is a widely used method for data integration, designed to extract latent components shared across paired high-dimensional datasets. Despite decades of practical success, a precise theoretical understanding of its behavior in high-dimensional regimes remains limited. In this paper, we study a data integration model in which two high-dimensional data matrices share a low-rank common latent structure while also containing individual-specific components. We analyze the singular vectors of the associated cross-covariance matrix using tools from random matrix theory and derive asymptotic characterizations of the alignment between estimated and true latent directions. These results provide a quantitative explanation of the reconstruction performance of the PLS variant based on Singular Value Decomposition (PLS-SVD) and identify regimes where the method exhibits counter-intuitive or limiting behavior. Buildi
    
[^86]: 利用信念惯性欺骗非平稳老虎机算法

    Fooling Algorithms in Non-Stationary Bandits using Belief Inertia

    [https://arxiv.org/abs/2511.05620](https://arxiv.org/abs/2511.05620)

    本文利用“信念惯性”机制构造确定性对抗实例，对滑动窗口UCB算法给出了最坏情况动态遗憾的下界，证明即使是专门设计用于遗忘过时观测的算法，也会被非平稳环境的变点欺骗而遭受显著遗憾。

    

    我们研究了特定多臂老虎机算法在至多含一个变点的分段平稳实例上的最坏情况动态遗憾。我们的构造利用了信念惯性机制：在环境变化发生之前收集的观测数据会使算法难以快速修正其决策所依据的经验排序。我们首先针对 Explore-Then-Commit、ε-贪婪和 UCB 算法说明这一机制，对于这些算法，确定性的单变点实例即可产生线性遗憾。我们的主要结果针对标准的滑动窗口 UCB（SW-UCB），该算法本旨在遗忘过时的观测数据。对于每个 K≥2 以及每个窗口 K≤τ≤T，我们证明了其最坏情况遗憾的下界为 (1/20)·min{T, (K ln T)^(1/3) T^(2/3)}。该证明结合了两个确定性障碍：一个使重复遗忘代价高昂的平稳小间隙实例，以及一个造成严格后……

    arXiv:2511.05620v2 Announce Type: replace  Abstract: We study worst-case dynamic regret of specific multi-armed bandit algorithms on piecewise-stationary instances with at most one breakpoint. Our constructions exploit belief inertia: observations collected before a change can make an algorithm slow to revise the empirical ordering on which its decisions are based. We first illustrate this mechanism for Explore-Then-Commit, $\epsilon$-greedy, and UCB, for which deterministic one-breakpoint instances can produce linear regret. Our principal result concerns standard sliding-window UCB (SW-UCB), which was designed to forget outdated observations. For every $K\geq 2$ and every window $K\leq\tau\leq T$, we prove the finite gap-free lower bound $\frac{1}{20}\min\{T,(K\ln T)^{1/3}T^{2/3}\}$ on its worst-case regret. The proof combines two deterministic obstructions: a stationary small-gap instance that makes repeated forgetting costly, and a one-breakpoint instance that creates a strict post-
    
[^87]: CPATTA：面向主动测试时自适应的保形监督分配

    CPATTA: Conformal Supervision Allocation For Active Test-Time Adaptation

    [https://arxiv.org/abs/2509.25692](https://arxiv.org/abs/2509.25692)

    该论文提出CPATTA，首次将带覆盖率感知在线校准的保形预测不确定性引入主动测试时自适应，通过平滑保形分数、伪覆盖率驱动的在线权重更新、领域偏移检测和分阶段更新方案，使准确率持续超越现有最先进方法约5%。

    

    主动测试时自适应（Active Test-Time Adaptation, ATTA）通过在部署阶段有选择地查询人工标注来提升模型在领域偏移下的鲁棒性，但现有方法采用启发式的不确定性度量，数据选择效率低下，浪费了人工标注预算。我们提出了保形预测主动测试时自适应方法（Conformal Prediction Active TTA, CPATTA），首次将具有原则性、基于覆盖率感知在线校准的保形不确定性引入ATTA。CPATTA采用结合top-K确定性度量的平滑保形分数、由伪覆盖率驱动的在线权重更新算法、自适应调整人工监督的领域偏移检测器，以及平衡人工标注与模型标注数据的分阶段更新方案。大量实验表明，CPATTA在准确率上持续超越最先进的ATTA方法约5%。

    arXiv:2509.25692v2 Announce Type: replace-cross  Abstract: Active Test-Time Adaptation (ATTA) improves model robustness under domain shift by selectively querying human annotations at deployment, but existing methods use heuristic uncertainty measures and suffer from low data selection efficiency, wasting human annotation budget. We propose Conformal Prediction Active TTA (CPATTA), which first brings principled, conformal uncertainty with coverage-aware online calibration into ATTA. CPATTA employs smoothed conformal scores with a top-$K$ certainty measure, an online weight-update algorithm driven by pseudo coverage, a domain-shift detector that adapts human supervision, and a staged update scheme that balances human-labeled and model-labeled data. Extensive experiments demonstrate that CPATTA consistently outperforms the state-of-the-art ATTA methods by around 5% in accuracy.
    
[^88]: 数据高效的时间依赖偏微分方程代理模型：图神经模拟器与神经算子的对比

    Data-Efficient Time-Dependent PDE Surrogates: Graph Neural Simulators vs. Neural Operators

    [https://arxiv.org/abs/2509.06154](https://arxiv.org/abs/2509.06154)

    提出图神经模拟器（GNS），通过消息传递结合数值时间步进格式建模瞬时时间导数，克服神经算子依赖大数据的缺陷，实现时间依赖PDE的数据高效代理建模。

    

    开发精确且数据高效的代理模型是推进科学人工智能（AI for Science）的核心。神经算子（NOs）通过传统神经网络架构来近似无穷维函数空间之间的映射，作为偏微分方程（PDE）驱动系统的代理模型已广受欢迎。然而，它们对大型数据集的依赖以及在低数据情况下泛化能力的局限阻碍了其实际应用。我们认为这些限制源于其对数据的全局处理方式，未能充分利用物理系统的局部离散化结构。为解决这一问题，我们提出图神经模拟器（GNS）作为时间依赖偏微分方程的一种有原则的代理建模范式。GNS利用消息传递机制结合数值时间步进格式，通过建模瞬时时间导数来学习PDE动力学。这种设计模仿了传统数值求解器，从而实现稳定的（摘要在此处截断）

    arXiv:2509.06154v3 Announce Type: replace  Abstract: Developing accurate, data-efficient surrogate models is central to advancing AI for Science. Neural operators (NOs), which approximate mappings between infinite-dimensional function spaces using conventional neural architectures, have gained popularity as surrogates for systems driven by partial differential equations (PDEs). However, their reliance on large datasets and limited ability to generalize in low-data regimes hinder their practical utility. We argue that these limitations arise from their global processing of data, which fails to exploit the local, discretized structure of physical systems. To address this, we propose Graph Neural Simulators (GNS) as a principled surrogate modeling paradigm for time-dependent PDEs. GNS leverages message-passing combined with numerical time-stepping schemes to learn PDE dynamics by modeling the instantaneous time derivatives. This design mimics traditional numerical solvers, enabling stable
    
[^89]: 冷冻电镜：一个随机逆问题

    Cryo-EM as a Stochastic Inverse Problem

    [https://arxiv.org/abs/2509.05541](https://arxiv.org/abs/2509.05541)

    本文将冷冻电镜三维重建创新性地表述为概率测度空间上的随机逆问题，通过最小化观测与模拟图像分布间的统计距离（KL散度、最大均值差异），并借助Wasserstein梯度流的粒子数值方法求解，从而突破了传统离散构象假设、实现了连续结构变化的恢复。

    

    冷冻电子显微镜（Cryo-EM）能够对生物分子进行高分辨率成像，但结构异质性仍然是三维重建中的主要挑战。传统方法假设构象为一个离散集合，这限制了其恢复连续结构变化的能力。在这项工作中，我们将冷冻电镜重建表述为概率测度空间上的一个随机逆问题（SIP），其中观测到的图像被建模为分子结构上未知分布经由随机前向算子的推前映射。我们将重建问题表述为观测图像分布与模拟图像分布之间变分差异的最小化问题，并采用KL散度和最大均值差异（Maximum Mean Discrepancy）等统计距离作为度量。由此产生的优化问题通过Wasserstein梯度流在概率测度空间上进行求解，我们使用粒子来数值表示并求解该测度……

    arXiv:2509.05541v2 Announce Type: replace-cross  Abstract: Cryo-electron microscopy (Cryo-EM) enables high-resolution imaging of biomolecules, but structural heterogeneity remains a major challenge in 3D reconstruction. Traditional methods assume a discrete set of conformations, limiting their ability to recover continuous structural variability. In this work, we formulate cryo-EM reconstruction as a stochastic inverse problem (SIP) over probability measures, where the observed images are modeled as the push-forward of an unknown distribution over molecular structures via a random forward operator. We pose the reconstruction problem as the minimization of a variational discrepancy between observed and simulated image distributions, using statistical distances such as the KL divergence and the Maximum Mean Discrepancy. The resulting optimization is performed over the space of probability measures via a Wasserstein gradient flow, which we numerically solve using particles to represent an
    
[^90]: 数据的量子几何

    Quantum Geometry of Data

    [https://arxiv.org/abs/2507.21135](https://arxiv.org/abs/2507.21135)

    本文首次完整建立了量子认知机器学习（QCML）的矩阵几何框架及其数学基础，将数据编码为希尔伯特空间中的量子几何，直接从数据导出内蕴维度、量子度规与贝里曲率等几何拓扑结构，并提出矩阵拉普拉斯算子作为避免维度灾难、保持几何性质的图嵌入替代方法。

    

    我们展示了量子认知机器学习如何将数据编码为量子几何。在QCML中，数据的特征由学习得到的厄米矩阵表示，数据点被映射为希尔伯特空间中的态。量子几何描述为数据集赋予了丰富的几何与拓扑结构——包括内蕴维度、量子度规和贝里曲率——这些均直接从数据中导出。QCML能够捕捉数据的全局性质，同时避免了局部方法固有的维度灾难。本工作首次完整阐述了QCML作为一个矩阵几何框架，建立了其数学基础，并演示了如何将量子几何中的算符级工具直接应用于数据。我们引入了矩阵拉普拉斯算子及其特征映射，作为保持几何性质的图嵌入替代方案，并展示了如何将切尔类等拓扑不变量应用于数据。

    arXiv:2507.21135v2 Announce Type: replace  Abstract: We demonstrate how Quantum Cognition Machine Learning (QCML) encodes data as quantum geometry. In QCML, features of the data are represented by learned Hermitian matrices, and data points are mapped to states in Hilbert space. The quantum geometry description endows the dataset with rich geometric and topological structure---including intrinsic dimension, quantum metric, and Berry curvature---derived directly from the data. QCML captures global properties of data, while avoiding the curse of dimensionality inherent in local methods. The present work provides the first full exposition of QCML as a framework for \emph{matrix geometry}, establishing its mathematical foundation and demonstrating how operator-level tools from quantum geometry can be directly applied to data. We introduce the matrix Laplacian and its eigenmaps as a geometry-preserving alternative to graph-based embeddings, and we show how topological invariants such as Che
    
[^91]: 因果片段：逐片分析和改进脉冲神经网络

    Causal pieces: analysing and improving spiking neural networks piece by piece

    [https://arxiv.org/abs/2504.14015](https://arxiv.org/abs/2504.14015)

    提出“因果片段”新概念来分析脉冲神经网络，证明片段内输出脉冲时间对输入和参数局部Lipschitz连续，且因果片段数量可作为衡量SNN逼近能力的有效指标。

    

    我们提出了“因果片段”，这是一个用于分析脉冲神经网络（SNN）的新概念，其灵感来源于人工神经网络（ANN）中用于研究表达能力和可训练性的“线性片段”。因果片段将具有单脉冲编码的前馈SNN的输入和参数空间划分为不同的区域，在每个区域中，相同的子网络导致输出脉冲的产生。对于由具有大膜时间常数的基于电流的泄漏积分发放（LIF）神经元构成的网络，我们证明了在每个因果片段内，输出脉冲时间相对于输入和网络参数是局部Lipschitz连续的。我们进一步证明了依赖于因果片段数量的逼近误差下界。因此，因果片段的数量是衡量SNN逼近能力的一个度量指标，该指标在脉冲时间不连续的情况下仍然有效，并且适用于同时具有兴奋性和抑制性突触的网络。

    arXiv:2504.14015v2 Announce Type: replace-cross  Abstract: We introduce "causal pieces", a novel concept for analysing spiking neural networks (SNNs), inspired by "linear pieces" used to study expressivity and trainability in artificial neural networks (ANNs). Causal pieces partition the input and parameter space of a feedforward SNN with single-spike coding into distinct regions where the same subnetwork causes the output spikes. For networks of current-based leaky integrate-and-fire (LIF) neurons with large membrane time constants, we show that within each causal piece, output spike times are locally Lipschitz continuous with respect to inputs and network parameters. We further prove a lower bound on the approximation error that depends on the number of causal pieces. Thus, the number of causal pieces is a measure of the approximation capabilities of SNNs, which is valid despite spike-time discontinuities and applies to networks with both excitatory and inhibitory synapses. Empirical
    
[^92]: 估计T细胞受体的因果效应

    Estimating the Causal Effects of T Cell Receptors

    [https://arxiv.org/abs/2410.14127](https://arxiv.org/abs/2410.14127)

    该论文提出一种利用V(D)J重组生成的预选择TCR库作为自然实验来校正未观察混杂因素的方法，结合半参数层次因果模型与置换不变神经网络，从观察性TCR测序数据中推断T细胞受体序列对患者疾病预后的因果效应。

    

    人类免疫学的一个核心问题是患者的T细胞受体如何影响疾病。在此，我们提出一种方法，利用观察性TCR测序数据和临床结果数据来推断T细胞受体（TCR）序列对患者预后的因果效应。我们的方法利用患者的预选择TCR库来校正未观察到的混杂因素，例如患者的环境和生活史。这种预选择库由一个程序化的随机重组过程——V(D)J重组——生成，这提供了一个自然实验。我们首先推导出一个因果识别结果，利用生物学理论以半参数方式约束层次因果模型，从而实现因果推断。然后，我们开发了一种因果估计策略，该策略使用置换不变神经网络和表示学习，可扩展到来自数百名患者的数百万条序列。

    arXiv:2410.14127v2 Announce Type: replace-cross  Abstract: A central question in human immunology is how a patient's T cell receptors impacts disease. Here, we introduce a method to infer the causal effects of T cell receptor (TCR) sequences on patient outcomes using observational TCR sequencing data and clinical outcomes data. Our approach corrects for unobserved confounders, such as a patient's environment and life history, using the patient's pre-selection TCR repertoire. This pre-selection repertoire is generated by a programmed stochastic recombination process, V(D)J recombination, which provides a natural experiment. We first derive a causal identification result that leverages biological theory to semiparametrically constrain a hierarchical causal model, enabling causal inference. We then develop a causal estimation strategy that uses permutation invariant neural networks and representation learning to scale to millions of sequences from hundreds of patients. Given sequence data
    

