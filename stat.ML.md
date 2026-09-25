# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Riemannian Gradient Descent for Gaussian Mixture Models with unknown diagonal covariances](https://arxiv.org/abs/2609.30220) | 本文提出将锥粒子梯度下降与黎曼梯度下降相结合，利用高斯分布的Fisher-Rao几何结构，为估计具有未知分量数目和未知对角协方差矩阵的高斯混合模型提供了收敛性理论保证和数值实验验证。 |
| [^2] | [Anchored Extra-Proximal Methods: Optimal Higher-Order Methods for Monotone Inclusion Problems](https://arxiv.org/abs/2609.30212) | 提出了锚定额外邻近（AEP）框架，通过结合锚定外推与满足相对误差条件的非精确锚定邻近更新，为复合单调包含问题构造出复杂度最优的任意阶（p≥2）高阶求解方法。 |
| [^3] | [Intrinsic-Extrinsic Coupling in Learning Dynamics](https://arxiv.org/abs/2609.30185) | 该论文形式化了学习动力学中的“内在-外在耦合”，证明学习者的当前观察并不决定其对后续训练的响应——相同的内在干预在不同外部延续下会产生非加性的、读出特定的交互效应，例如回放机制可将一次写入操作在32次更新中的贡献从五个正确预测变为零。 |
| [^4] | [Nuclear Norm-Regularized Bayesian Matrix Completion](https://arxiv.org/abs/2609.30078) | 本文提出了首个针对未知噪声方差下核范数正则化贝叶斯矩阵补全模型的采样器，并给出了复杂度为矩阵维度和目标精度倒数之多项式的显式非渐近保证。 |
| [^5] | [Path-specific harm decomposition: A partial identification framework](https://arxiv.org/abs/2609.29938) | 该论文提出了将治疗伤害分解为直接路径和间接（中介）路径贡献的新概念与部分识别框架，为即使在随机对照试验中也无法点识别的直接与间接负向影响比例（FNA）提供了识别边界。 |
| [^6] | [Robust Detection of LLM-Generated Text under Contamination](https://arxiv.org/abs/2609.29935) | 该论文将人类与机器文本建模为带Huber污染的有限阶马尔可夫过程，刻画了LLM生成文本可靠检测的精确理论边界，并证明对似然比检验等统计检测器进行截断处理可在污染环境下实现鲁棒检测。 |
| [^7] | [Shrinking-Tube Concentration for Adaptive Markovian Stochastic Approximation](https://arxiv.org/abs/2609.29833) | 本文为自适应马尔可夫链驱动的投影随机逼近建立了收缩管集中界，证明迭代点以多项式衰减的逃逸概率停留在随时间收紧的目标邻域内，给出匹配下界刻画了容差收缩与逃逸概率衰减间的最优权衡，并扩展到含鞅差噪声与可预测偏差的情形。 |
| [^8] | [Boolean threshold functions, neuron capacity, and memory retrieval](https://arxiv.org/abs/2609.29756) | 本文精确计数了n维布尔阈值函数的数目，将单个阈值神经元的容量确定为 $n^2-\log_2(n!)+1+O(n^{-99})$ 比特，把经典误差项从 $O(n)$ 改进到 $O(n^{-99})$，并为神经网络无虚假记忆检索的容量问题提供了数学基础。 |
| [^9] | [Stochastic Semantic Evidence Graphs: Uncertainty Propagation and Governance for Agentic AI](https://arxiv.org/abs/2609.29703) | 提出随机语义证据图（SSEG）框架，通过分层随机有向无环图对智能体AI中证据、检索、提示、生成及决策映射各环节的不确定性进行建模与传播，实现终端误差的逐路径界定、来源溯源的Fréchet界传播以及治理触发的诊断。 |
| [^10] | [Optimal Recovery Meets Bayesian Learning: Where Worst-Case Bounds Pay Off](https://arxiv.org/abs/2609.29622) | 该论文揭示了最坏情况最优恢复与贝叶斯学习的精确数学对应关系，并通过实验证明 Morozov 校准在噪声盲规则失效、可复现性和后端迁移场景下显著优于发布权重、ML-II 和 GCV 等传统超参数选择规则，但在可交换数据上 split-conformal 方法更胜一筹。 |
| [^11] | [When Identical Rows Disagree: From Benchmark Identifiability to Replication-Robust Anomaly Detection](https://arxiv.org/abs/2609.29580) | 论文揭示了表格数据中重复行对基准评估和异常检测的系统性影响，并提出因子化检测器SCOUT，通过分离复制不变的支持证据与计数证据，实现复制鲁棒的无监督异常检测。 |
| [^12] | [DeepGOF-1: A Pretrained Convolutional Goodness-of-Fit Test for Logistic Regression with a Computable Consistency Certificate](https://arxiv.org/abs/2609.29575) | 提出了一种统计量为预训练冻结卷积网络的逻辑回归拟合优度检验，p值通过分析者自身的bootstrap校准来保证检验水平的精确性，并首次提供可通过单次前向传播计算得出的一致性证书。 |
| [^13] | [The Impossible Trinity of Time-Series Validation: A Conservation Law among Training Sufficiency, Test Coverage, and Temporal Causality](https://arxiv.org/abs/2609.29530) | 本文证明时间序列验证存在“不可能三难”——训练充分性、测试覆盖度与时间因果性无法同时满足，并给出守恒律不等式 α+β ≤ 1+Λ，量化了跨越因果边界所必须付出的数据泄漏偏差代价。 |
| [^14] | [Physics-Informed Neural Operator Surrogate for 2D Magnetohydrodynamic Reconnection](https://arxiv.org/abs/2609.29514) | 本研究开发了一种基于傅里叶神经算子的物理信息神经算子（PINO）代理模型，用于二维可压缩电阻性磁重联模拟，通过预测磁通函数精确满足无散度约束、以连续时间查询消除自回归误差累积，从而在大Lundquist数下以远低于直接数值模拟的成本实现参数扫描。 |
| [^15] | [Direct Message Approximation (DMA): A Consistency-Based Framework for Tractable Approximate Inference on Factor Graphs](https://arxiv.org/abs/2609.29466) | 该论文提出直接消息近似（DMA），通过直接近似因子到变量的消息而非边缘分布，并借助一致性条件与主定理，实现了无需内循环迭代、避免负精度消息且误差可控的因子图近似推断。 |
| [^16] | [Neural Transport Nested Sampling](https://arxiv.org/abs/2609.29413) | 提出神经传输嵌套采样（NTNS）算法，将嵌套采样与神经流方法相结合，仅需目标能量函数评估即可对高维分子系统进行采样并估计完整配分函数，在含55个粒子的Lennard-Jones团簇上，将采样误差较最强神经基线降低了一个数量级以上。 |
| [^17] | [Machine Unlearning for Gibbs Supervised Learning Algorithms](https://arxiv.org/abs/2609.29409) | 提出了一种基于ERM-RER变分形式的精确遗忘方法，使吉布斯监督学习算法在遗忘数据后与从头重新训练的结果在分布上完全一致。 |
| [^18] | [Learning a Flow to Self-Supervised Representations](https://arxiv.org/abs/2609.29350) | 本文提出非对抗性的基于流的分布匹配框架 FBDM，通过球面条件速度回归学习参考引导的几何结构，避免了昂贵的编码器-评论家优化，性能与分布匹配方法几乎相当并与现有自监督学习方法具有竞争力。 |
| [^19] | [GCUL: Ambiguity Identification in Text Emotion Classification via Cluster-Guided Learning](https://arxiv.org/abs/2609.29327) | 提出了一种几何引导的选择性分类框架GCUL，将误分类和歧义实例视为表示空间中的混淆吸引子，通过三阶段聚类引导学习使拒绝边界从表示几何结构中自然涌现，而非依赖预设的拒绝率。 |
| [^20] | [Sufficiently Reduced Distributional Regression](https://arxiv.org/abs/2609.29291) | 本文提出SRDR方法，通过严格恰当评分规则将充分降维转化为风险最小化问题，并利用可通过采样估计的能量评分联合训练降维映射与生成式预测模型，无需密度计算或对抗训练，同时证明了估计条件分布在能量距离上的收敛性。 |
| [^21] | [FB-GDM: Fully-Bayesian Guided Diffusion Models for High-Dimensional Linear Inverse Problems via Unsupervised Variational Inference](https://arxiv.org/abs/2609.29216) | FB-GDM提出了一种全贝叶斯引导扩散方法，通过在每个反向扩散步骤中用变分推断自动估计两个精度参数，免除了针对具体任务且需依赖真值的人工超参数校准，同时借助可分离分解保持线性计算复杂度，成本与一次ΠGDM运行相当。 |
| [^22] | [Functional dynamic mode decomposition: Learning infinite-dimensional systems from data](https://arxiv.org/abs/2609.29159) | 本文提出了函数型动态模态分解（DMD），将投影DMD和精确DMD从有限维扩展到无限维系统，无需对空间域进行离散化即可直接从数据中学习偏微分方程等无限维动力系统。 |
| [^23] | [Feature Space Selection and Heterogeneous Effect Estimation for Blood-Brain Barrier Permeability: A Random Forest to the Generalized Random Forest Pipeline](https://arxiv.org/abs/2609.29076) | 本研究通过系统性消融实验比较多种分子特征空间与算法组合，发现基于组合特征的动态随机森林在血脑屏障通透性预测中取得最高 AUC（0.970），并进一步结合广义随机森林与双重/去偏机器学习，探索性地估计了分子结构与 BBB 通透性之间的异质性关联。 |
| [^24] | [Transformers as Cross-Task Learners: Shared Structure Drives Sample Efficiency in In-Context Learning](https://arxiv.org/abs/2609.29060) | 本文通过覆盖数刻画任务空间的复杂度，揭示了Transformer如何利用共享的跨任务结构提升上下文学习的样本效率，并提出了一种基于锚函数的任务识别与评估方法。 |
| [^25] | [Personalised federated learning for Riemannian and Euclidean EEG decoding](https://arxiv.org/abs/2609.29037) | 该论文将个性化联邦学习适配到黎曼SPDNet脑电解码器上，让所有受试者共享主干而各自保留分类头，在三个运动想象数据集上取得了优于标准联邦学习、集中式训练和EEGNet的准确率，同时收敛更快、通信参数更少。 |
| [^26] | [When Does Action Credit Need Updating?](https://arxiv.org/abs/2609.29007) | 该论文提出成对分支敏感度来判断策略更新引起的漂移是否会推翻原有动作排序，从而仅在必要时利用一阶锚定信用迁移估计器基于旧干预轨迹更新历史动作信用，大幅降低工具使用智能体反复更新的成本。 |
| [^27] | [Back to the Definition: Estimating Step-Level Advantages via Trajectory Graphs for Agentic Reinforcement Learning](https://arxiv.org/abs/2609.28963) | 提出通过轨迹图来估计步骤级优势，解决了GRPO等分组强化学习方法在步骤层面因轨迹级粗粒度估计而产生的系统性偏差问题。 |
| [^28] | [Selective Inference for Deep Clustering in Latent Spaces](https://arxiv.org/abs/2609.28756) | 本文针对使用固定预训练编码器的深度聚类提出了一个选择性推断框架，通过应对从原始数据空间到潜在空间的非线性变换所带来的复杂选择过程，为聚类结果的统计可靠性检验提供了计算可行且有效的 p 值。 |
| [^29] | [Exact Bayes Regret and Asymptotic Optimality in High-Dimensional Gaussian Bandits](https://arxiv.org/abs/2609.28718) | 本文研究时间范围与维度成比例的高维高斯贝叶斯线性老虎机，证明了归一化后验不确定性在所有因果策略上具有一致显式极限，由此推导出汤普森采样等策略的精确后悔曲线，并证明后验均值贪婪选择达到极限最优贝叶斯后悔，而汤普森采样的后悔严格更大。 |
| [^30] | [RLVR landscapes for iterated multiplications can be benign: Insights from spin-glass theory](https://arxiv.org/abs/2609.28625) | 本文借助自旋玻璃理论证明，对于迭代乘法等算法任务，RLVR的优化地形是良性的（不存在局部极小值陷阱），其训练困难主要来自扩散屏障和梯度估计误差，而非地形本身的陷阱。 |
| [^31] | [Global Convergence of Third-Order Langevin Dynamics for Non-Convex Optimization via Simulated Annealing](https://arxiv.org/abs/2609.28611) | 该论文证明了在模拟退火框架下，采用固定摩擦与递减噪声的三阶朗之万动力学在非凸优化中可依概率收敛到全局最小值，并给出了离散化格式保持该收敛速率的充分步长条件。 |
| [^32] | [NumericJev: Jev-like LLM Numerical Decoding with Multiway Decision Trees](https://arxiv.org/abs/2609.28587) | 提出了一种无需训练的数值解码算法NUMERICJEV，通过多路决策树递归细化数值范围，使任何具有类Jev结构化选择接口的大语言模型都能输出数值，其性能甚至超过从包含正确答案的候选列表中直接选择。 |
| [^33] | [SGA: Uncertainty Quantification for Multi-Step Forecasting in Time Series Foundation Models](https://arxiv.org/abs/2609.28582) | 本文提出SGA方法，通过有向无环图表征预测分支拓扑结构并度量其图复杂度，实现了对时序基础模型多步预测不确定性的有效量化，提升了预测结果的可信度。 |
| [^34] | [Matrix Aggregation Operators](https://arxiv.org/abs/2609.28562) | 本文首次形式化了矩阵聚合算子（MAO）的概念，将聚合理论从向量拓展到矩阵结构，并分析了其可分解性与对称性，证明某些算子无法分解为逐行逐列聚合的形式。 |
| [^35] | [Speculative Evaluation of Stochastic LLMs](https://arxiv.org/abs/2609.28560) | 该论文提出了一种基于分层贝叶斯尼曼策略的推测式评估方法（HBN及异步版本HBN-async），通过分层贝叶斯模型估计各任务方差并自适应地分配推演预算，从而在固定预算下最小化随机大语言模型基准评估的方差。 |
| [^36] | [An Order-Theoretic Characterization of Consistent Inductive Inference](https://arxiv.org/abs/2609.28551) | 该论文在ZFC框架下通过有限可实现轨迹上的一个线性序（要求冲突轨迹选择不同子轨迹、且对每个固定目标良基）完整刻画了一致归纳推理，证明一致性等价于此类序的存在性，并回答了Lu（2024）的问题。 |
| [^37] | [Stochastic Inertial Krasnosel'skii-Mann Iteration Achieves Near-Optimal Sample Complexity](https://arxiv.org/abs/2609.28543) | 本文提出一种随机惯性 Krasnosel'skii-Mann（iKM）迭代方法，仅在随机 KM 算法上添加两个惯性外推、且每次更新只需一次可能有偏随机预言机调用的条件下，实现了 Õ(ε⁻²) 的近乎最优样本复杂度。 |
| [^38] | [Sequential Confidence Sets for Coverage-Constrained Conformal Model Selection](https://arxiv.org/abs/2609.28522) | 提出了覆盖率约束序贯模型置信集（CC-SMCS），利用同步鞅置信序列与精确闭式规则，以至少1-δ的概率识别出满足覆盖率硬约束且成本最小的保形预测流水线。 |
| [^39] | [Certified Task-Conditioned Active Observability](https://arxiv.org/abs/2609.28520) | 该论文形式化了“认证的任务条件化主动可观测性复杂度”，即在认证误差与安全弃权保证下识别任务相关状态所需的最小最坏情况期望交互代价，并证明任务预测等价性诱导出唯一的最小充分商空间，使主动可观测性复杂度在其上严格不变。 |
| [^40] | [From Prediction to Explainable Provider Behavior Profiles for Fraud, Waste, and Abuse Review](https://arxiv.org/abs/2609.28477) | 该研究提出将欺诈、浪费与滥用（FWA）审查从预测建模转向可解释的提供者行为画像，通过将账单收入分解为提供者规模与诊疗项目构成的乘积，从而解释提供者行为变化的原因。 |
| [^41] | [Learning to Fluctuate: Statistical Foundations for Causal Tabular Pretraining](https://arxiv.org/abs/2609.26290) | 提出波动监督预训练（FSP），用平均处理效应加有效影响函数波动来标注合成表格，并证明完全波动可使高斯标签可观测，从而将因果标签预测风险从 $(1-\lambda)^2/n$ 阶降至 $n^{-2}$ 阶。 |
| [^42] | [Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts](https://arxiv.org/abs/2609.18366) | 提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。 |
| [^43] | [Supervising the Chain Ladder](https://arxiv.org/abs/2609.16552) | 本文将链梯法准备金进展模式的选择建模为监督学习问题，通过在严格凸的目标函数上添加可解释的惩罚项与超参数（如数据衰减、权重幂、基准参考与平滑约束），把精算师的专业判断形式化，并可通过单一线性系统求解。 |
| [^44] | [Confidence Horizons](https://arxiv.org/abs/2608.03889) | 本文提出“置信视界”这一新型统计对象，通过放弃有限时间范围之外的有效性，在预算或伦理等约束下获得更精确的大样本任意时有效推断，并与Pocock、O'Brien-Fleming等经典成组序贯边界建立了明确联系。 |
| [^45] | [All you need is log](https://arxiv.org/abs/2606.27349) | 本文刻画了在数据处理下单调且在独立乘积上可加的多分布泛函的唯一形式，即通过多路重合散度在四层参数空间上的正积分来统一表示，解决了Rényi族向多分布泛化的开放问题。 |
| [^46] | [MultiwayPAM: Multiway Partitioning Around Medoids for LLM-as-a-Judge Score Analysis](https://arxiv.org/abs/2603.10287) | 该论文提出了MultiwayPAM，一种新的张量聚类方法，能够同时估计LLM-as-a-Judge评分张量各模式的聚类成员和中心点，从而揭示LLM评估器评分偏差的结构。 |
| [^47] | [Inverse Problems Conditioned on Observation Ensembles: Applications and Methods](https://arxiv.org/abs/2601.22029) | 该论文提出了一类新的统计问题——集合条件逆问题（EIP），并基于一种利用观测集合信息的新型条件生成模型（集合逆生成模型），给出了非迭代的推理时后验采样方法，可应用于高能物理解折叠、全波形反演和逆成像等领域。 |
| [^48] | [Stacked SVD or SVD stacked? A Random Matrix Theory perspective on data integration](https://arxiv.org/abs/2507.22170) | 本文借助随机矩阵理论，首次在比例渐近区间下严格比较了Stack-SVD与SVD-Stack这两种估计多数据集共享奇异子空间的主流数据整合方法的理论性能。 |
| [^49] | [Capturing Unseen Spatial Heat Extremes Through Dependence-Aware Generative Modeling](https://arxiv.org/abs/2507.09211) | DeepX-GAN是一种显式捕捉空间依赖性的深度生成模型，能够零样本模拟超出历史记录的统计上合理的“未见”热极端事件，揭示多个地点同时遭受极端高温的隐藏风险。 |
| [^50] | [Time-Varying Bayesian Optimization Without a Metronome](https://arxiv.org/abs/2501.18963) | 该论文首次推导出显式考虑观测采样频率变化的时变贝叶斯优化遗憾上界，并据此提出了关于数据集规模和过期数据策略的实用建议，其 BOLT 算法在实验中优于现有最先进的 TVBO 方法。 |
| [^51] | [Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data](https://arxiv.org/abs/2405.18929) | 提出了一种将正-无标签学习与自编码器、深度支持向量数据描述等深度异常检测模型相结合的深度正-无标签异常检测框架，以应对无标签数据被异常污染的现实情况，从而提升半监督异常检测的性能。 |
| [^52] | [A Probabilistic Approach for Alignment with Human Comparisons](https://arxiv.org/abs/2403.10771) | 通过提出的两阶段“监督微调+人类比较”框架，本文研究了如何有效利用人类比较来改善AI模型的对齐，特别是在面对嘈杂数据和高维模型时。 |
| [^53] | [Transitional Conditional Independence](https://arxiv.org/abs/2104.11547) | 本文提出“过渡条件独立性”新概念，通过马尔可夫核的单一分解将条件独立性推广到涉及参数、处理、环境等非随机变量的情形，无需在输入空间上定义分布，并证明其不对称性是本质的。 |
| [^54] | [CatSIM: A Categorical Image Similarity Metric](https://arxiv.org/abs/2004.09073) | CatSIM是一种基于结构相似性范式的新图像相似度度量方法，适用于二值和多值的二维及三维图像与体积，对位置的微小扰动具有鲁棒性，并能比较图像内部的任意区域。 |

# 详细

[^1]: 针对未知对角协方差高斯混合模型的黎曼梯度下降

    Riemannian Gradient Descent for Gaussian Mixture Models with unknown diagonal covariances

    [https://arxiv.org/abs/2609.30220](https://arxiv.org/abs/2609.30220)

    本文提出将锥粒子梯度下降与黎曼梯度下降相结合，利用高斯分布的Fisher-Rao几何结构，为估计具有未知分量数目和未知对角协方差矩阵的高斯混合模型提供了收敛性理论保证和数值实验验证。

    

    本文研究了Beurling-LASSO（BLASSO）的数值求解问题，BLASSO是一个在测度空间中促进稀疏性的凸优化框架。我们考虑将其应用于估计具有未知分量数目和未知对角协方差矩阵的高斯混合模型（GMM）。我们的方法将锥粒子梯度下降（CPGD）原理与黎曼梯度下降相结合，以充分考虑高斯分布所固有的Fisher-Rao几何结构。我们的贡献有两个方面。首先，我们为算法的收敛性提供了理论保证。特别地，我们在解的非退化条件下建立了指数级局部收敛性，并将该假设与底层统计目标的分离条件联系起来。其次，我们解决了CPGD的实际实现问题，并通过数值实验展示了其性能。

    arXiv:2609.30220v1 Announce Type: cross  Abstract: This paper investigates the numerical resolution of the Beurling-LASSO (BLASSO), a convex optimization framework that promotes sparsity in the space of measures. We consider its application to the estimation of Gaussian mixture models (GMMs) with an unknown number of components and unknown diagonal covariance matrices. Our approach combines the Conic Particle Gradient Descent (CPGD) principle with Riemannian gradient descent, to account for the underlying Fisher-Rao geometry of Gaussian distributions. Our contributions are twofold. First, we provide theoretical guarantees for the convergence of our algorithm. In particular, we establish exponential local convergence under a non-degeneracy condition on the solution and relate this assumption to a separation condition on the underlying statistical target. Second, we address practical implementation aspects of CPGD and present numerical experiments illustrating its performance. On the tes
    
[^2]: 锚定额外邻近方法：单调包含问题的最优高阶方法

    Anchored Extra-Proximal Methods: Optimal Higher-Order Methods for Monotone Inclusion Problems

    [https://arxiv.org/abs/2609.30212](https://arxiv.org/abs/2609.30212)

    提出了锚定额外邻近（AEP）框架，通过结合锚定外推与满足相对误差条件的非精确锚定邻近更新，为复合单调包含问题构造出复杂度最优的任意阶（p≥2）高阶求解方法。

    

    我们研究了在切残差准则下，求复合单调包含问题近似解所需的确定性预言机复杂度，该问题由一个光滑的单值单调算子与一个极大单调的集值算子之和构成。我们提出了锚定额外邻近（AEP）框架，该框架将锚定外推步骤与满足相对误差条件的非精确锚定邻近更新相结合。在一阶情形下，该框架能够恢复复合快速外梯度方法；通过将隐式更新中的算子替换为其在外推点处的泰勒近似，该框架自然地产生了二阶及更高阶的扩展。对于每个 p≥2，在假设单值算子的 (p-1) 阶导数满足 Lipschitz 连续的条件下，我们将这一构造与二分线搜索相结合，得到一个 p 阶方法，用以找到切残差满足给定精度的点。

    arXiv:2609.30212v1 Announce Type: cross  Abstract: We study the deterministic oracle complexity of finding approximate solutions to composite monotone inclusion problems, formed by the sum of a smooth single-valued monotone operator and a maximally monotone set-valued operator, under the tangent-residual criterion. We introduce the Anchored Extra-Proximal (AEP) framework, which combines an anchored extrapolation step with an inexact anchored proximal update satisfying a relative-error condition. The framework recovers the composite Fast Extragradient method in the first-order setting and yields natural second- and higher-order extensions by replacing the operator in the implicit update with its Taylor approximation at the extrapolated point. For every $p\geq 2$, assuming that the $(p-1)$th derivative of the single-valued operator is Lipschitz continuous, we combine this construction with a bisection line search to obtain a $p$th-order method that finds a point with tangent residual at 
    
[^3]: 学习动力学中的内在-外在耦合

    Intrinsic-Extrinsic Coupling in Learning Dynamics

    [https://arxiv.org/abs/2609.30185](https://arxiv.org/abs/2609.30185)

    该论文形式化了学习动力学中的“内在-外在耦合”，证明学习者的当前观察并不决定其对后续训练的响应——相同的内在干预在不同外部延续下会产生非加性的、读出特定的交互效应，例如回放机制可将一次写入操作在32次更新中的贡献从五个正确预测变为零。

    

    学习者的当前观察结果并不一定决定其对后续训练的响应。我们通过受限学习状态干预的“延续条件化价值”来形式化内在-外在耦合，并用相对于观察的纤维来刻画当前的一致性。我们提出一种可执行的有限框架分类器头写入操作，在有限精度接受检查下，既保护当前的logits，又修复指定的历史边际。我们区分了局部可容许性、延续条件化干预价值以及完整策略性能三者的不同。一个匹配的四格对照实验识别出：相同的内在干预与不同的外部延续之间存在读出特定的非加性交互。在基于CLINC的类增量学习设置中，回放机制使该写入操作在32次更新中的贡献从五个正确预测变为零。在输出蒸馏、使用RoBERTa骨干网络以及优化器原生机制下，同样会出现非零的交互效应。

    arXiv:2609.30185v1 Announce Type: new  Abstract: A learner's current observations need not determine its response to further training. We formulate intrinsic-extrinsic coupling through the continuation-conditioned value of a constrained learning-state intervention, with observation-relative fibers describing present agreement. An executable finite-frame classifier-head write protects current logits while repairing specified historical margins under finite-precision acceptance checks. We distinguish local admissibility, continuation-conditioned intervention value, and complete-policy performance. A matched four-cell contrast identifies readout-specific non-additivity between the same intrinsic intervention and alternative external continuations. In a CLINC-derived class-incremental setting, replay changes the write's 32-update contribution from five correct predictions to zero. Nonzero interactions also occur under output distillation, with a RoBERTa backbone, and under optimizer-native
    
[^4]: 核范数正则化的贝叶斯矩阵补全

    Nuclear Norm-Regularized Bayesian Matrix Completion

    [https://arxiv.org/abs/2609.30078](https://arxiv.org/abs/2609.30078)

    本文提出了首个针对未知噪声方差下核范数正则化贝叶斯矩阵补全模型的采样器，并给出了复杂度为矩阵维度和目标精度倒数之多项式的显式非渐近保证。

    

    矩阵补全是指从带噪声观测的元素中估计矩阵中缺失元素的问题，它是推荐系统以及面板数据中反事实结果估计等众多问题的基础。许多算法采用正则化最小二乘法来解决该问题，通常以核范数作为正则化项，但这种方法只能产生点估计，缺乏内置的不确定性量化。贝叶斯公式是一种自然的替代方案：如果噪声方差已知，基于核范数的先验可以产生对数凹的后验分布。然而遗憾的是，在实践中噪声方差并非先验已知，因此要实现完全贝叶斯方法，必须对噪声方差施加先验。我们给出了该模型的第一个具有显式非渐近保证的采样器：其复杂度关于矩阵维度以及目标精度的倒数是多项式级的。我们的技术是将噪声精度的分布离散化到……

    arXiv:2609.30078v1 Announce Type: cross  Abstract: Matrix completion, the problem of estimating missing entries in a matrix from noisily observed ones, underlies a diverse array of problems such as recommender systems and counterfactual outcome estimation in panel data. Many algorithms address the problem using regularized least squares, often with the nuclear norm as a regularizer, but this method yields a point estimate with no built-in uncertainty quantification. A Bayesian formulation is a natural alternative, and if the noise variance is known, the nuclear norm-based prior yields a log-concave posterior. Unfortunately, in practice, the noise variance will not be known a priori, so for a fully Bayesian approach, a prior must be imposed on it. We give the first sampler for this model with an explicit non-asymptotic guarantee: polynomial in the matrix dimensions and in the reciprocal of the target accuracy. Our technique is to discretize the distribution of the noise precision onto a
    
[^5]: 路径特异性伤害分解：一个部分识别框架

    Path-specific harm decomposition: A partial identification framework

    [https://arxiv.org/abs/2609.29938](https://arxiv.org/abs/2609.29938)

    该论文提出了将治疗伤害分解为直接路径和间接（中介）路径贡献的新概念与部分识别框架，为即使在随机对照试验中也无法点识别的直接与间接负向影响比例（FNA）提供了识别边界。

    

    设计治疗政策时的一个核心目标往往是“不造成伤害”，即避免那些能改善平均结果却使某些个体结果恶化的干预措施。一个广泛使用的伤害度量是“负向影响比例”（FNA），定义为干预降低个体结果的概率。然而，在许多应用中，治疗是通过中介变量起作用的，单一的“总”FNA可能会掩盖伤害主要是通过直接路径还是间接（由中介诱导的）路径产生。在这项工作中，我们引入了FNA的路径特异性版本。为此，我们在因果中介分析框架下将总伤害分解为直接伤害和间接伤害。然而，这些量依赖于潜在结果的联合分布，即使在随机对照试验中也无法被点识别。作为解决方案，我们开发了一种新颖的直接FNA和间接FNA的部分识别框架。

    arXiv:2609.29938v1 Announce Type: cross  Abstract: A central goal when designing treatment policies is often to "do no harm", that is, to avoid interventions that improve average outcomes while worsening outcomes for some individuals. A widely used notion for harm is the fraction of negatively affected (FNA), defined as the probability that an intervention decreases an individual's outcome. However, in many applications, treatments operate through mediators, and a single "total" FNA can obscure whether harm arises primarily through direct pathways or indirect (mediator-induced) pathways. In this work, we introduce a path-specific analogue of the FNA. For this, we disentangle total harm into direct and indirect harm in causal mediation settings. However, these quantities depend on joint distributions of potential outcomes that are not point-identified even in randomised controlled trials. As a remedy, we develop a novel partial identification framework for direct and indirect FNA. In ou
    
[^6]: 污染环境下的LLM生成文本鲁棒检测

    Robust Detection of LLM-Generated Text under Contamination

    [https://arxiv.org/abs/2609.29935](https://arxiv.org/abs/2609.29935)

    该论文将人类与机器文本建模为带Huber污染的有限阶马尔可夫过程，刻画了LLM生成文本可靠检测的精确理论边界，并证明对似然比检验等统计检测器进行截断处理可在污染环境下实现鲁棒检测。

    

    我们研究在编辑和污染条件下对大语言模型（LLM）生成文本的检测。我们将人类和机器文本建模为带有Huber污染的有限阶马尔可夫过程，并在本文假设下刻画了可靠检测的精确边界。当污染程度相对于干净数据源的分离度足够大时，检测是不可能的。在该边界以下，一组截断似然比检验能够实现趋于零的最坏情况错误率。这一构造启发了将截断作为现有统计检测器的一种简单修改方法。对于一大类加性得分函数，我们识别出截断检验保持一致性、而原始检验的最坏情况功效趋于零的条件。我们在三个数据集和三个生成模型上评估了七种检测器，并在RAID基准上进行测试。截断在两项研究中均提升了鲁棒性，其增益因检测器和污染设置而异。例如，在目标……（摘要原文在此处截断）

    arXiv:2609.29935v1 Announce Type: cross  Abstract: We study the detection of LLM-generated text under editing and contamination. Modeling human and machine text as finite-order Markov processes with Huber contamination, we characterize an exact boundary for reliable detection under our assumptions. Detection is impossible when contamination is sufficiently large relative to clean-source separation. Below this boundary, a collection of clipped likelihood-ratio tests achieves vanishing worst-case errors. This construction motivates clipping as a simple modification of existing statistical detectors. For a broad class of additive scores, we identify conditions under which the clipped test is consistent while the raw test's worst-case power tends to zero. We evaluate seven detectors across three datasets and three generation models, and on the RAID benchmark. Clipping improves robustness in both studies, with gains varying across detectors and contamination settings. For example, at a targ
    
[^7]: 面向自适应马尔可夫随机逼近的收缩管型集中界

    Shrinking-Tube Concentration for Adaptive Markovian Stochastic Approximation

    [https://arxiv.org/abs/2609.29833](https://arxiv.org/abs/2609.29833)

    本文为自适应马尔可夫链驱动的投影随机逼近建立了收缩管集中界，证明迭代点以多项式衰减的逃逸概率停留在随时间收紧的目标邻域内，给出匹配下界刻画了容差收缩与逃逸概率衰减间的最优权衡，并扩展到含鞅差噪声与可预测偏差的情形。

    

    自适应算法日益在做出决策的同时重塑生成其未来数据的动力学。我们为自适应马尔可夫链驱动的投影随机逼近建立了收缩管集中界。该界以高概率保证，所选时刻之后的每一次迭代始终保持在围绕目标、随时间不断收紧的容差范围内。所选时刻之后发生任何偏离的概率具有多项式衰减的上界，而一个匹配的下界表明，在仅有二阶矩有限的条件下，该多项式指数在一般情形下不可再改进。因此，该结果刻画了容差收缩速度与未来逃逸概率下降速度之间的尖锐权衡。我们还将分析扩展到带有额外鞅差噪声和可预测偏差的递推形式，展示了鞅差噪声尺度的增长如何减缓逃逸概率的衰减。

    arXiv:2609.29833v1 Announce Type: cross  Abstract: Adaptive algorithms increasingly make decisions while reshaping the dynamics that generate their future data. We establish a shrinking-tube concentration bound for projected stochastic approximation driven by an adaptive Markov chain. The bound guarantees, with high probability, that every iterate after a chosen time remains within a tolerance around the target that tightens over time. The probability of any exit after the chosen time admits a polynomially decaying upper bound, and a matching lower bound shows that its polynomial exponent cannot be improved in general under finite second moments. The result therefore identifies a sharp tradeoff between how quickly the tolerance shrinks and how rapidly the probability of any future exit decreases. We also extend the analysis to recursions with additional martingale-difference noise and predictable bias, showing how growth in the martingale-difference noise scale slows the decay of the e
    
[^8]: 布尔阈值函数、神经元容量与记忆检索

    Boolean threshold functions, neuron capacity, and memory retrieval

    [https://arxiv.org/abs/2609.29756](https://arxiv.org/abs/2609.29756)

    本文精确计数了n维布尔阈值函数的数目，将单个阈值神经元的容量确定为 $n^2-\log_2(n!)+1+O(n^{-99})$ 比特，把经典误差项从 $O(n)$ 改进到 $O(n^{-99})$，并为神经网络无虚假记忆检索的容量问题提供了数学基础。

    

    单个神经元能记住多少信息？神经网络在不产生虚假记忆的前提下能检索多少条记忆？这些问题都与一个基本问题相关：形如 $f(x)=\operatorname{sgn}(a_0+\langle a,x\rangle)$（$x\in\{-1,1\}^n$）的布尔阈值函数究竟有多少个？本文证明，不同布尔阈值函数的数目 $T_n$ 满足 $T_n=2\binom{2^n-1}{n}\bigl(1+O(n^{-99})\bigr)$。等价地，单个阈值神经元的容量为 $n^2-\log_2(n!)+1+O(n^{-99})$ 比特，这将 Kahn–Komlós–Szemerédi 结果中的 $O(n)$ 误差项改进为 $O(n^{-99})$。为证明这一点，我们证明了：当 $1\le r\le n-1$ 且 $v_1,\ldots,v_r$ 从 $\{-1,1\}^n$ 中随机选取时，$\mathbb P\{\langle v_1,\ldots,v_r\rangle\cap\{-1,1\}^n =\{\pm v_1,\ldots,\pm v_r\} \} =1-O(n^{-99})$。在用于记忆检索的 Kanter–Sompolinsky 哈密顿量的背景下，这一结果表明……

    arXiv:2609.29756v1 Announce Type: cross  Abstract: How much information can a single neuron remember? How many memories can neural networks retrieve without creating false memories? These questions are related to a basic question: how many Boolean threshold functions $f(x)=\operatorname{sgn}(a_0+\langle a,x\rangle)$, $x\in\{-1,1\}^n$, are there? In this paper, we show that the number $T_n$ of distinct Boolean threshold functions is \[ T_n=2\binom{2^n-1}{n}\bigl(1+O(n^{-99})\bigr). \] Equivalently, the capacity of a single threshold neuron is $n^2-\log_2(n!)+1+O(n^{-99})$ bits, improving the $O(n)$ error term in the result of Kahn--Koml\'os--Szemer\'edi to $O(n^{-99})$. To prove this, we show that, for $1\le r\le n-1$, and $v_1,\ldots,v_r$ are chosen at random from $\{-1,1\}^n$, \[ \mathbb P\!\left\{ \langle v_1,\ldots,v_r\rangle\cap\{-1,1\}^n =\{\pm v_1,\ldots,\pm v_r\} \right\} =1-O(n^{-99}). \] In the context of the Kanter--Sompolinsky Hamiltonian for memory retrieval, this identifie
    
[^9]: 随机语义证据图：面向智能体AI的不确定性传播与治理

    Stochastic Semantic Evidence Graphs: Uncertainty Propagation and Governance for Agentic AI

    [https://arxiv.org/abs/2609.29703](https://arxiv.org/abs/2609.29703)

    提出随机语义证据图（SSEG）框架，通过分层随机有向无环图对智能体AI中证据、检索、提示、生成及决策映射各环节的不确定性进行建模与传播，实现终端误差的逐路径界定、来源溯源的Fréchet界传播以及治理触发的诊断。

    

    AI智能体评估通常只检查最终答案，但误差可能通过证据、检索、提示、生成或决策映射等环节引入。我们提出随机语义证据图（SSEG），这是一种分层随机有向无环图（DAG），其语言节点可扩展为自回归的词量子图，其可观测输出可以是完整短语上的概率分布。语义约简与校准均为可选操作。我们定义了图相对的局部缺陷与下游边影响，推导了终端误差的逐路径上界，并利用其逐节点分量来诊断治理触发条件。在来源溯源方面，该图保留了不确定的论断—段落关系，并传播精确的Fréchet界，而非假设各来源之间相互独立。在三种开放权重架构上，信息等价的变化会实质性改变完整短语的概率分布。一项受控实验在5,000个案例中未出现证书违规；交叉RAG与l…（摘要原文在此处截断）

    arXiv:2609.29703v1 Announce Type: new  Abstract: AI-agent evaluations usually inspect a final answer, yet error may enter through evidence, retrieval, prompting, generation or decision mapping. We introduce a stochastic semantic evidence graph (SSEG), a hierarchical stochastic DAG whose language node expands into an autoregressive token subgraph and whose observable output may be a law over complete phrases. Semantic reduction and calibration are optional. We define graph-relative local defects and downstream edge influences, derive a pathwise bound on terminal error and use its nodewise terms to diagnose governance triggers. For source provenance, the graph preserves uncertain claim--passage relations and propagates sharp Fr\'echet bounds rather than assuming independence across sources. Across three open-weight architectures, information-equivalent changes materially alter complete-phrase laws. A controlled experiment yields no certificate violations in 5,000 cases; crossed-RAG and l
    
[^10]: 最优恢复遇见贝叶斯学习：最坏情况界何时发挥优势

    Optimal Recovery Meets Bayesian Learning: Where Worst-Case Bounds Pay Off

    [https://arxiv.org/abs/2609.29622](https://arxiv.org/abs/2609.29622)

    该论文揭示了最坏情况最优恢复与贝叶斯学习的精确数学对应关系，并通过实验证明 Morozov 校准在噪声盲规则失效、可复现性和后端迁移场景下显著优于发布权重、ML-II 和 GCV 等传统超参数选择规则，但在可交换数据上 split-conformal 方法更胜一筹。

    

    最坏情况最优恢复（OR）与贝叶斯学习用两套术语描述了相同的高斯-二次-希尔伯特问题。我们深化了这一对应关系——信息半径等于经过块金值优化的高斯过程后验方差，并在一个具有闭式解的平衡块金值处由后验均值达到——同时在三个已发表的贝叶斯系统中实测最坏情况方法的优势所在。这本账簿是双面的：损失与收益同样具有启发意义。Morozov 校准在 σ 盲规则失效的场景下能以 1.00-1.19 倍的差距追踪测试访问预言机，在噪声抽取之间的可复现性提高 4.9-6.3 倍（p=0.002-0.004），并且是唯一在后端更换后选择依然稳健的可部署规则（1.36 倍，而发布权重、ML-II 和 GCV 为 12-30 倍）；紧证书在信息论下界处实现覆盖，无数值松弛。但在可交换数据上，split-conformal 方法正面击败了 OR。

    arXiv:2609.29622v1 Announce Type: cross  Abstract: Worst-case Optimal Recovery (OR) and Bayesian learning describe the same Gaussian-quadratic-Hilbert problems in two vocabularies. We sharpen the correspondence - the radius of information equals a nugget-optimized GP posterior variance and is attained by the posterior mean at a closed-form balance nugget - and measure, inside three published Bayesian systems, where the worst-case side pays. The ledger is two-sided: the losses instruct as much as the wins. Morozov calibration tracks a test-access oracle within $1.00$-$1.19\times$ where $\sigma$-blind rules fail, is $4.9$-$6.3\times$ more reproducible across noise draws ($p=0.002$-$0.004$), and is the only deployable rule whose selection survives a change of backend ($1.36\times$ against $12$-$30\times$ for the released weight, ML-II and GCV); tight certificates cover at the information-theoretic floor with no numerical slack. But on exchangeable data split-conformal beats the OR head on
    
[^11]: 当相同行出现分歧时：从基准可辨识性到复制鲁棒的异常检测

    When Identical Rows Disagree: From Benchmark Identifiability to Replication-Robust Anomaly Detection

    [https://arxiv.org/abs/2609.29580](https://arxiv.org/abs/2609.29580)

    论文揭示了表格数据中重复行对基准评估和异常检测的系统性影响，并提出因子化检测器SCOUT，通过分离复制不变的支持证据与计数证据，实现复制鲁棒的无监督异常检测。

    

    发布的数据表通常被视为独立同分布样本，但其重复行可能编码了业务频率、重复实体、连接操作、重采样或提取错误。我们证明这种模糊性创造了一个隐藏的测量层，带来三个后果：特征相同的行造成已达到的评估上限，行加权AUROC对复制敏感，而行训练的检测器会学习到偏向多重性规模的规律。对全部690个OddBench数据集的精确行审计发现：355个存在训练-测试重叠，147个存在特征相同但标签冲突的情况，137个存在与训练正常样本完全相同的测试异常。在四种经典检测器几何结构上，从行加权切换到支持集加权会使50-61个数据集的AUROC变化至少0.05。我们提出SCOUT（支持计数正交化无监督测试），一种因子化的异常检测器，将复制不变的支持证据与暴露感知的计数证据分离，从而实现复制鲁棒的检测。

    arXiv:2609.29580v1 Announce Type: new  Abstract: A released table is often treated as an i.i.d. sample, although its repeated rows may encode business frequency, repeated entities, joins, resampling, or extraction errors. We show that this ambiguity creates a hidden measurement layer with three consequences: feature-identical rows impose an attained evaluation ceiling, row-weighted AUROC is sensitive to replication, and row-trained detectors learn a multiplicity-size-biased law. An exact-row audit of all 690 OddBench datasets finds train-test overlap in 355, feature-identical label conflict in 147, and a test anomaly identical to a training normal in 137. Switching from row to support weighting changes AUROC by at least 0.05 on 50-61 datasets across four classical detector geometries. We introduce SCOUT (Support-Count Orthogonalized Unsupervised Testing), a factorized anomaly detector that separates replication-invariant support evidence from exposure-aware count evidence. Factorwise s
    
[^12]: DeepGOF-1：一种用于逻辑回归的预训练卷积拟合优度检验及其可计算的一致性证书

    DeepGOF-1: A Pretrained Convolutional Goodness-of-Fit Test for Logistic Regression with a Computable Consistency Certificate

    [https://arxiv.org/abs/2609.29575](https://arxiv.org/abs/2609.29575)

    提出了一种统计量为预训练冻结卷积网络的逻辑回归拟合优度检验，p值通过分析者自身的bootstrap校准来保证检验水平的精确性，并首次提供可通过单次前向传播计算得出的一致性证书。

    

    逻辑回归的拟合优度检验在最需要它们的场合反而最不可靠：在小样本条件下，其检验水平会偏离名义水平，而将多个检验组合起来会加剧这种偏离。我们提出了一种新的检验方法，其统计量是一个卷积网络，该网络只需在模拟的偏离数据上训练一次，便能将失拟“读取”为一幅图像：以协变量秩为坐标的标准ized残差网格。分析者无需进行任何训练。网络以冻结状态发布，p值是观测得分在分析者自身自助法（bootstrap）样本中的秩，因此检验水平是校准过程的属性，而非网络所学内容的属性。我们证明了在枢轴性条件下的精确性、无该条件时的渐近精确性，以及一个一致性定理——其关键条件可通过冻结权重在单次前向传播中计算得出，从而为每种备择假设提供一份证书；我们还测量了该检验的“盲锥”范围。在一个预先声明的六十单元网格上，部署后的检验水平在五十八个单元中保持在名义区间内。

    arXiv:2609.29575v1 Announce Type: cross  Abstract: Goodness-of-fit tests for logistic regression are least reliable where they are most needed: at small samples their levels drift from the nominal one, and combining them worsens the drift. We propose a test whose statistic is a convolutional network, trained once on simulated departures, that reads misfit as a picture: a grid of standardized residuals over covariate ranks. The analyst never trains. The network ships frozen, and the p-value is the rank of the observed score within the analyst's own bootstrap, so the level is a property of the calibration rather than of what the network learned. We prove exactness under pivotality, asymptotic exactness without it, and a consistency theorem whose key condition is computable from the frozen weights in one forward pass, giving a per-alternative certificate; we also measure the test's blind cone. On a pre-declared sixty-cell grid the deployed level stays in the nominal band in fifty-eight ce
    
[^13]: 时间序列验证的不可能三难：训练充分性、测试覆盖度与时间因果性之间的守恒定律

    The Impossible Trinity of Time-Series Validation: A Conservation Law among Training Sufficiency, Test Coverage, and Temporal Causality

    [https://arxiv.org/abs/2609.29530](https://arxiv.org/abs/2609.29530)

    本文证明时间序列验证存在“不可能三难”——训练充分性、测试覆盖度与时间因果性无法同时满足，并给出守恒律不等式 α+β ≤ 1+Λ，量化了跨越因果边界所必须付出的数据泄漏偏差代价。

    

    在时间序列上验证模型需要同时满足三个条件：每次训练应使用样本的大部分（充分性）、各测试集应共同覆盖样本的大部分（覆盖度）、且训练数据应先于测试数据（因果性）。我们证明这三者无法同时兼得，并为每一项“定价”。设 α 为各折中最小的训练比例，β 为测试所覆盖样本的比例，Λ 为样本中来自某测试点未来、却被用作训练数据的比例，δ 为从测试点到其未来中最近训练点的距离。在长度为 T 的样本上，任何验证方案都满足 α+β ≤ 1+Λ 与 α+min{β, δ/T} ≤ 1，并且在 β-混合（β-mixing）条件下，某测试点处的数据泄漏偏差至多为 2Mβ_mix(δ)。换言之：要越过因果边界 α+β=1，就必须在测试点的未来数据上进行训练……（摘要原文在此处截断）

    arXiv:2609.29530v1 Announce Type: new  Abstract: Validating a model on a time series asks for three things at once: each training run should use most of the sample (sufficiency), the test sets should together cover most of the sample (coverage), and training data should come before test data (causality). We prove that the three cannot be had together and price each one. Let $\alpha$ be the smallest training fraction over folds, $\beta$ the fraction of the sample covered by tests, $\Lambda$ the fraction of the sample used as training data from the future of a test point, and $\delta$ the distance from a test point to the nearest training point in its future. Every scheme on a sample of length $T$ satisfies $\alpha+\beta \le 1+\Lambda$ and $\alpha+\min\{\beta,\delta/T\} \le 1$, and under $\beta$-mixing the leakage bias at a test point is at most $2M\beta_{\mathrm{mix}}(\delta)$. In words: going beyond the causal frontier $\alpha+\beta=1$ requires training on the future; that future data 
    
[^14]: 二维磁流体动力学磁重联的物理信息神经算子代理模型

    Physics-Informed Neural Operator Surrogate for 2D Magnetohydrodynamic Reconnection

    [https://arxiv.org/abs/2609.29514](https://arxiv.org/abs/2609.29514)

    本研究开发了一种基于傅里叶神经算子的物理信息神经算子（PINO）代理模型，用于二维可压缩电阻性磁重联模拟，通过预测磁通函数精确满足无散度约束、以连续时间查询消除自回归误差累积，从而在大Lundquist数下以远低于直接数值模拟的成本实现参数扫描。

    

    磁重联的直接数值模拟受限于电阻性磁流体力学（MHD）的尺度分离问题。在大Lundquist数 $S$ 条件下，电流层厚度以 $S^{-1/2}$ 的速率变薄，这迫使计算采用精细网格和极短时间步长，使得参数扫描的成本高得令人望而却步。深度学习神经算子提供了一种替代方案：它学习的是函数空间之间的映射而非单个解，因此单个训练好的模型可以以推理成本返回任意参数和任意时刻的系统状态。我们开发了一种基于傅里叶神经算子（FNO）的物理信息神经算子（PINO）代理模型，用于有壁边界域中二维可压缩、粘性、电阻性磁重联问题。该模型以初始状态、Lundquist数和连续查询时间为条件输入。通过预测磁通函数，使 $\nabla\cdot\mathbf{B}=0$ 约束精确成立；直接时间查询消除了自回归误差累积；此外还采用了奇偶性感知的（原文在此处截断）

    arXiv:2609.29514v1 Announce Type: cross  Abstract: Direct numerical simulation of magnetic reconnection is limited by the scale separation of resistive magnetohydrodynamics. At large Lundquist numbers $S$ the current layer thins as $S^{-1/2}$, forcing fine grids and short time steps that make parameter scans prohibitively expensive. Deep learning neural operators offer an alternative by learning the map between function spaces rather than individual solutions, so that a single trained model returns the state for any parameter and time at inference cost. We have developed a Fourier Neural Operator (FNO) based Physics-Informed Neural Operator (PINO) surrogate for two-dimensional compressible, viscous, resistive reconnection in a wall bounded domain. We condition on the initial state, the Lundquist number, and a continuous query time. Predicting the magnetic flux function makes $\grad\!\cdot\!\bB=0$ exact, direct time queries remove autoregressive error accumulation, and parity-aware spec
    
[^15]: 直接消息近似（DMA）：一种基于一致性的因子图可驾驭近似推断框架

    Direct Message Approximation (DMA): A Consistency-Based Framework for Tractable Approximate Inference on Factor Graphs

    [https://arxiv.org/abs/2609.29466](https://arxiv.org/abs/2609.29466)

    该论文提出直接消息近似（DMA），通过直接近似因子到变量的消息而非边缘分布，并借助一致性条件与主定理，实现了无需内循环迭代、避免负精度消息且误差可控的因子图近似推断。

    

    因子图上的近似消息传递是两大主流概率推断算法族的基础：期望传播（EP）和变分消息传递（VMP）。这两种方法都在每个因子边上近似边缘分布，这迫使算法采用迭代的轮询调度，存在产生负精度消息的风险，且对于VMP而言，在Dirac-delta因子处会退化为点估计。我们提出直接消息近似（DMA），它直接近似因子到变量的消息，而非边缘分布。对于可归一化的因子，我们定义了一个一致性条件（要求当所有其他传入消息均为Dirac delta时达到精确结果）来指导消息的构造。我们证明了一个主定理（针对正规消息、任意图结构），利用消息KL散度约束边缘KL散度，并由其导出三个结构性推论：Dirac输入一致性、无需EP式的内循环迭代、以及不会产生负精度消息。此外，我们还证明了一个互补的 O(1/r^...（摘要在此处被截断）

    arXiv:2609.29466v1 Announce Type: cross  Abstract: Approximate message passing on factor graphs underlies two dominant families of probabilistic inference algorithms: expectation propagation (EP) and variational message passing (VMP). Both methods approximate the marginal at each factor edge, forcing an iterative round-robin schedule, risking negative-precision messages, and, for VMP, collapsing to point estimates at Dirac-delta factors. We introduce Direct Message Approximation (DMA), which approximates factor-to-variable messages directly rather than the marginal. For normalisable factors, we define a consistency condition (requiring exactness when all other incoming messages are Dirac deltas) to guide message construction. We prove a master theorem (proper messages, any graph) bounding marginal KL from message KL, with three structural corollaries: Dirac-input consistency, no EP-style inner-loop iteration, and no negative-precision messages. Further, we prove a complementary $O(1/r^
    
[^16]: 神经传输嵌套采样

    Neural Transport Nested Sampling

    [https://arxiv.org/abs/2609.29413](https://arxiv.org/abs/2609.29413)

    提出神经传输嵌套采样（NTNS）算法，将嵌套采样与神经流方法相结合，仅需目标能量函数评估即可对高维分子系统进行采样并估计完整配分函数，在含55个粒子的Lennard-Jones团簇上，将采样误差较最强神经基线降低了一个数量级以上。

    

    从分子系统的玻尔兹曼分布中进行采样是一个推断问题，近年来在神经密度估计技术进展的推动下取得了显著发展。我们开发了一种新颖的采样算法——神经传输嵌套采样（NTNS），它将嵌套采样的经典优势与现代基于神经流的方法相结合。NTNS 在嵌套采样外循环中，使用流匹配速度作为经过 Metropolis–Hastings 校正的 Langevin 核中的漂移项，仅需对目标能量函数进行评估，并能为高维粒子系统的完整配分函数提供可扩展的估计。我们在具有挑战性的分子采样基准上对 NTNS 进行了测试，规模扩展至由 55 个相互作用粒子组成的 Lennard–Jones 团簇。在该任务上，相对于最强的神经基线方法，NTNS 将与参考 MCMC 相比的原子间距离和能量的 Wasserstein 误差降低了一个数量级以上。

    arXiv:2609.29413v1 Announce Type: new  Abstract: Sampling from Boltzmann distributions of molecular systems is an inference problem that has seen significant recent developments fuelled by advances in neural density estimation. We develop a novel sampling algorithm, Neural Transport Nested Sampling (NTNS), which combines the classical strengths of nested sampling with modern neural flow-based methods. NTNS uses a flow matching velocity as the drift in a Metropolis--Hastings corrected Langevin kernel inside a nested sampling outer loop, requiring only evaluations of the target energy function and providing scalable estimation of the full partition function of high-dimensional particle systems. We benchmark NTNS on challenging molecular sampling benchmarks, scaling up to Lennard--Jones clusters of 55 interacting particles, where it reduces both interatomic distance and energy Wasserstein errors to reference MCMC by over an order of magnitude relative to the strongest neural baselines at 
    
[^17]: 吉布斯监督学习算法的机器遗忘

    Machine Unlearning for Gibbs Supervised Learning Algorithms

    [https://arxiv.org/abs/2609.29409](https://arxiv.org/abs/2609.29409)

    提出了一种基于ERM-RER变分形式的精确遗忘方法，使吉布斯监督学习算法在遗忘数据后与从头重新训练的结果在分布上完全一致。

    

    本文提出了一种针对吉布斯监督学习算法实现精确遗忘的方法，该方法采用受相对熵正则化经验风险最小化（ERM-RER）启发的变分形式。该方法通过在待遗忘数据集上最大化期望经验风险，并以相对于原始算法的相对熵作为正则化约束来实现。优化变量是模型空间上的一个概率测度，其解为另一个吉布斯概率测度，代表一个新的吉布斯监督学习算法。该方法保证了精确遗忘，即新的吉布斯算法在分布上与在保留数据集上从头重新训练所得到的算法完全一致。作为副产品，该方法还提供了一个通过策略性地选择参考测度和正则化项来对ERM-RER中的数据点进行重新加权的框架。

    arXiv:2609.29409v1 Announce Type: cross  Abstract: In this paper, a method for achieving exact unlearning for Gibbs supervised learning algorithms is proposed using a variational formulation inspired by empirical risk minimization subject to relative entropy regularization (ERM-RER). Such a method consists of maximizing the expected empirical risk over the dataset to be unlearned subject to a regularization by relative entropy with respect to the original algorithm. The optimization variable is a probability measure on the models; and the solution is another Gibbs probability measure that represents a new Gibbs supervised learning algorithm. The method guarantees exact unlearning in the sense that the new Gibbs algorithm coincides in distribution with the algorithm that would have been obtained by retraining from scratch on the dataset to be retained. As a byproduct, a framework for reweighting data points in ERM-RER by strategically choosing both the reference measure and the regulari
    
[^18]: 学习一个通向自监督表示的流

    Learning a Flow to Self-Supervised Representations

    [https://arxiv.org/abs/2609.29350](https://arxiv.org/abs/2609.29350)

    本文提出非对抗性的基于流的分布匹配框架 FBDM，通过球面条件速度回归学习参考引导的几何结构，避免了昂贵的编码器-评论家优化，性能与分布匹配方法几乎相当并与现有自监督学习方法具有竞争力。

    

    显式的几何参考为构建自监督表示提供了一种直接的方式。然而，现有的对抗性分布匹配方法需要代价高昂的编码器-评论家（encoder-critic）联合优化。我们提出了基于流的分布匹配（Flow-Based Distribution Matching, FBDM），这是一个非对抗性框架，通过球面条件速度回归来学习这种参考引导的几何结构。受等角紧框架（ETF）启发的参考允许其分量数量 K' 超过辅助流维度 d*，同时保持结构化的几何分离。我们将每张图像的两个增广视图分配给同一目标，同时限制每个参考中心可以接收的图像数量。显式的对齐损失进一步拉近了两个视图的表示。在从 CIFAR 到 ImageNet 的多个基准测试上的实验表明，FBDM 取得了与 DM 几乎相当的性能，并与现有的自监督学习（SSL）方法保持竞争力。在匹配训练成本的比较中……（摘要在此处截断）

    arXiv:2609.29350v1 Announce Type: cross  Abstract: Explicit geometric references offer a direct way to structure self-supervised representations. Existing adversarial distribution-matching formulations, however, require costly encoder-critic optimization. We introduce Flow-Based Distribution Matching (FBDM), a non-adversarial framework that learns this reference-directed geometry through spherical conditional velocity regression. An ETF-inspired reference allows its number of components K' to exceed the auxiliary flow dimension d* while retaining structured geometric separation. We assign both augmented views of each image to the same target, while limiting how many images each reference center can receive. An explicit alignment loss further pulls the two views' representations closer together. Experiments across benchmarks ranging from CIFAR to ImageNet show that FBDM achieves performance nearly on par with DM and remains competitive with existing SSL methods. Matched training-cost co
    
[^19]: GCUL：通过聚类引导学习实现文本情感分类中的歧义识别

    GCUL: Ambiguity Identification in Text Emotion Classification via Cluster-Guided Learning

    [https://arxiv.org/abs/2609.29327](https://arxiv.org/abs/2609.29327)

    提出了一种几何引导的选择性分类框架GCUL，将误分类和歧义实例视为表示空间中的混淆吸引子，通过三阶段聚类引导学习使拒绝边界从表示几何结构中自然涌现，而非依赖预设的拒绝率。

    

    选择性分类使模型能够对不确定的实例放弃预测，但现有方法通常通过置信度分数、预定义的覆盖率约束或实例级距离度量来拒绝这些实例。这些方法可能忽视了学习表示空间中困难样本的集体几何结构。我们提出了引导聚类式不确定学习（GCUL），这是一种几何引导的选择性分类框架，它将误分类和歧义实例识别为表示空间中的潜在混淆吸引子。GCUL使用三阶段程序来初始化、聚类并显式重新标注这一不确定区域，使拒绝边界从底层表示几何结构中自然涌现，而非依赖于预设的拒绝率。我们进一步推导了一个选择性分数和一个几何充分条件，用于刻画拒绝机制何时能够产生积极效果。

    arXiv:2609.29327v1 Announce Type: cross  Abstract: Selective classification enables a model to abstain from predictions on uncertain instances, but existing approaches typically reject them through confidence scores, predefined coverage constraints or instance-level distance measures. These approaches may overlook the collective geometric structure of difficult samples in learned representation spaces. We propose Guided Clustering-based Uncertain Learning (GCUL), a geometric-guided selective classification framework that identifies misclassified and ambiguous instances as a potential confusion attractor in the representation space. GCUL uses a three-phase procedure to initialize, cluster, and explicitly relabel this uncertain region, allowing the rejection boundary to emerge from the underlying representation geometry rather than from a prescribed rejection rate. We further derive a selectivity score and a geometric sufficient condition that characterizes when rejection can provide pos
    
[^20]: 充分约简分布回归

    Sufficiently Reduced Distributional Regression

    [https://arxiv.org/abs/2609.29291](https://arxiv.org/abs/2609.29291)

    本文提出SRDR方法，通过严格恰当评分规则将充分降维转化为风险最小化问题，并利用可通过采样估计的能量评分联合训练降维映射与生成式预测模型，无需密度计算或对抗训练，同时证明了估计条件分布在能量距离上的收敛性。

    

    我们提出了充分约简分布回归（SRDR），这是一种将条件分布估计与非线性充分降维（SDR）相结合的生成式方法。该方法建立在通过严格恰当评分规则对充分性的刻画之上：当且仅当使用约简后的协变量预测响应相对于使用完整协变量不会造成期望评分损失时，该降维才是充分的。由此，充分降维被转化为一个风险最小化问题。SRDR通过最小化能量评分来联合训练降维映射和生成式预测模型，而能量评分可以通过采样进行估计，无需密度计算或对抗训练。该框架还可扩展至多环境数据和分类任务。我们证明了估计的条件分布在能量距离上收敛于真实的条件分布，这意味着学习到的表示在渐近意义下是充分的。

    arXiv:2609.29291v1 Announce Type: cross  Abstract: We propose Sufficiently Reduced Distributional Regression (SRDR), a generative method that combines conditional distribution estimation with nonlinear sufficient dimension reduction (SDR). It builds on a characterization of sufficiency through strictly proper scoring rules: a dimension reduction is sufficient if and only if predicting the response from the reduced covariates incurs no loss in expected score relative to the full covariates. Sufficient dimension reduction thus becomes a risk minimization problem. SRDR jointly trains a dimension reduction map and a generative prediction model by minimizing the energy score, which can be estimated by sampling without density evaluation or adversarial training. The framework extends to multi-environment data and to classification. We prove that the estimated conditional distributions converge in energy distance to the true ones, which implies that the learned representation is asymptoticall
    
[^21]: FB-GDM：基于无监督变分推断的全贝叶斯引导扩散模型，用于高维线性逆问题

    FB-GDM: Fully-Bayesian Guided Diffusion Models for High-Dimensional Linear Inverse Problems via Unsupervised Variational Inference

    [https://arxiv.org/abs/2609.29216](https://arxiv.org/abs/2609.29216)

    FB-GDM提出了一种全贝叶斯引导扩散方法，通过在每个反向扩散步骤中用变分推断自动估计两个精度参数，免除了针对具体任务且需依赖真值的人工超参数校准，同时借助可分离分解保持线性计算复杂度，成本与一次ΠGDM运行相当。

    

    扩散模型是线性逆问题的强大先验，但现有的参考引导方法——扩散后验采样（DPS）和伪逆引导扩散模型（ΠGDM）——依赖于需要针对每个任务调整的标量超参数，且通常需要借助真值来进行调节。我们提出了FB-GDM，一种完全贝叶斯的引导扩散方法，它消除了这一校准步骤。从ΠGDM的高斯近似出发，我们推导出了依赖于两个精度参数（即方差的倒数）的闭式条件分数，其中一个与去噪近似相关，另一个与观测似然相关，并将这两个参数视为潜变量，在每个反向扩散步骤中通过变分推断进行估计。一种可分离的分解方式使得每次更新的计算量与像素数量呈线性关系，因此推断在全图像分辨率下依然可以高效进行，其计算成本仅相当于运行一次ΠGDM。FB-GDM既不需要噪声水平信息，也不需要真值（原文此处被截断）……

    arXiv:2609.29216v1 Announce Type: cross  Abstract: Diffusion models are powerful priors for linear inverse problems, but the reference guidance methods, Diffusion Posterior Sampling (DPS) and Pseudoinverse-Guided Diffusion Models ($\Pi$GDM), rely on scalar hyperparameters tuned per task, usually against the ground truth. We introduce FB-GDM, a fully-Bayesian guided diffusion method that removes this calibration step. Starting from the Gaussian approximation of $\Pi$GDM, we derive a closed-form conditional score that depends on two precision parameters (inverse variances), one associated with the denoising approximation and one with the observation likelihood, and treat them as latent variables inferred by variational inference at each reverse step. A separable factorization makes each update scale linearly with the number of pixels, so the inference stays tractable at full image resolution, at a cost comparable to one $\Pi$GDM run. FB-GDM requires neither the noise level nor the ground
    
[^22]: 函数型动态模态分解：从数据中学习无限维系统

    Functional dynamic mode decomposition: Learning infinite-dimensional systems from data

    [https://arxiv.org/abs/2609.29159](https://arxiv.org/abs/2609.29159)

    本文提出了函数型动态模态分解（DMD），将投影DMD和精确DMD从有限维扩展到无限维系统，无需对空间域进行离散化即可直接从数据中学习偏微分方程等无限维动力系统。

    

    动态模态分解（DMD）是一种数据驱动方法，它计算底层动力系统的最佳线性逼近，并将动力学分解为特征时空模式的叠加。DMD最初由流体力学领域提出，此后其本身及各种扩展方法已在分子动力学、气候科学、工程、金融和神经科学等众多研究领域得到广泛应用，应用场景包括降维、预测、系统辨识、控制和谱聚类等。为了将DMD应用于偏微分方程，通常需要先使用有限差分或有限元技术对空间域进行离散化，从而隐式地将问题转化为有限维问题。本文将投影DMD和精确DMD扩展到了无限维系统：我们的DMD变体不再从向量值观测中估计矩阵，而是学习有限秩的算子（摘要在此处截断）。

    arXiv:2609.29159v1 Announce Type: cross  Abstract: Dynamic mode decomposition (DMD) is a data-driven method that computes the best linear approximation of the underlying dynamical system and decomposes the dynamics into a superposition of characteristic spatiotemporal patterns. Originally introduced by the fluid dynamics community, DMD and its extensions have found widespread use in many other research areas such as molecular dynamics, climate science, engineering, finance, and neuroscience. Applications include dimensionality reduction, forecasting, system identification, control, and spectral clustering. In order to apply DMD to partial differential equations, the spatial domain is typically first discretized using finite difference or finite element techniques, thus implicitly rendering the problem finite-dimensional. We extend projected and exact DMD to infinite-dimensional systems. Rather than estimating matrices from vector-valued observations, our DMD variants learn finite-rank 
    
[^23]: 血脑屏障通透性的特征空间选择与异质性效应估计：从随机森林到广义随机森林的流程

    Feature Space Selection and Heterogeneous Effect Estimation for Blood-Brain Barrier Permeability: A Random Forest to the Generalized Random Forest Pipeline

    [https://arxiv.org/abs/2609.29076](https://arxiv.org/abs/2609.29076)

    本研究通过系统性消融实验比较多种分子特征空间与算法组合，发现基于组合特征的动态随机森林在血脑屏障通透性预测中取得最高 AUC（0.970），并进一步结合广义随机森林与双重/去偏机器学习，探索性地估计了分子结构与 BBB 通透性之间的异质性关联。

    

    预测血脑屏障（BBB）通透性对中枢神经系统药物发现至关重要。本研究使用 MoleculeNet BBBP 数据集（n = 2039），系统地对分子特征空间进行消融分析，以将特征化方法与模型架构的影响相互分离。我们在四种学习算法上评估了三类特征家族（Morgan 指纹、RDKit 理化描述符、SMILES 二元组）。结果表明，预测性能同时取决于特征表示与算法的选择。使用组合特征的动态随机森林取得了最高的平均 AUC（0.970，95% 置信区间：0.963–0.977）。其次，基于这一最优特征表示，我们利用广义随机森林对分子结构与 BBB 通透性之间的异质性关联进行了探索性估计。我们通过 LogP 的中位数分割构建伪处理变量，并应用双重/去偏机器学习方法来控制混杂因素。正交化处理显著地（原文摘要在此处截断）

    arXiv:2609.29076v1 Announce Type: cross  Abstract: Predicting blood-brain barrier (BBB) permeability is critical for central nervous system drug discovery. Using the MoleculeNet BBBP dataset (n = 2039), this study systematically ablates molecular feature spaces to isolate featurisation from model architecture. We evaluate three feature families (Morgan fingerprints, RDKit physicochemical descriptors, SMILES bigrams) across four learning algorithms. Results demonstrate that predictive performance depends jointly on feature representation and algorithm. Dynamic Random Forest using combined features achieved the highest mean AUC (0.970, 95% CI: 0.963-0.977). Second, this optimal representation enables exploratory estimation of heterogeneous associations between molecular structure and BBB permeability using Generalized Random Forests. Constructing a pseudo-treatment from a LogP median split, we applied double/debiased machine learning to account for confounding. Orthogonalization substant
    
[^24]: Transformer作为跨任务学习器：共享结构驱动上下文学习中的样本效率

    Transformers as Cross-Task Learners: Shared Structure Drives Sample Efficiency in In-Context Learning

    [https://arxiv.org/abs/2609.29060](https://arxiv.org/abs/2609.29060)

    本文通过覆盖数刻画任务空间的复杂度，揭示了Transformer如何利用共享的跨任务结构提升上下文学习的样本效率，并提出了一种基于锚函数的任务识别与评估方法。

    

    Transformer通过在预训练期间联合学习广泛的任务族，并仅凭简短的提示就能适应未见过的任务，从而取得了卓越的性能。然而，对这一现象的严格数学和统计学理解仍然有限。本文旨在研究Transformer如何利用共享的跨任务结构，以及这种结构如何影响上下文学习（ICL）的样本复杂度。具体而言，我们通过在规定度量下的覆盖数来刻画任务空间复杂度，从而在无需显式参数化表示的情况下量化低维跨任务结构。所得的覆盖提供了一组锚函数，我们利用它们提出了任务识别与评估程序：通过上下文观测在锚函数中定位一个未见过的任务，再通过聚合相应锚函数在查询点上的评估值来预测响应。

    arXiv:2609.29060v1 Announce Type: cross  Abstract: Transformers achieve remarkable performance by jointly learning broad families of tasks during pretraining and adapting to unseen tasks from only a short prompt. Yet a rigorous mathematical and statistical understanding of this phenomenon remains limited. This paper aims to study how Transformers exploit shared cross-task structure and how this structure affects the sample complexity of in-context learning (ICL). Specifically, we characterize task-space complexity through covering numbers under a prescribed metric, thereby quantifying the low-dimensional cross-task structure without requiring an explicit parametric representation. The resulting cover provides a set of anchor functions, which we use to introduce a task-identification-and-evaluation procedure: context observations localize an unseen task among the anchor functions, and the response at a query is predicted by aggregating the corresponding anchor function query evaluations
    
[^25]: 用于黎曼流形和欧几里得脑电解码的个性化联邦学习

    Personalised federated learning for Riemannian and Euclidean EEG decoding

    [https://arxiv.org/abs/2609.29037](https://arxiv.org/abs/2609.29037)

    该论文将个性化联邦学习适配到黎曼SPDNet脑电解码器上，让所有受试者共享主干而各自保留分类头，在三个运动想象数据集上取得了优于标准联邦学习、集中式训练和EEGNet的准确率，同时收敛更快、通信参数更少。

    

    联邦学习（FL）使脑电（EEG）解码器能够从多个受试者的记录中学习，而无需将数据汇集到一起。我们考虑了两种轻量级脑电解码器：基于黎曼流形的SPDNet和基于欧几里得空间的EEGNet。两者都分为主干和头部两部分，其中主干用于构建潜在表示，头部用于对其进行分类。然而，受试者间的差异性使得单一共享的联邦学习模型难以很好地适配每个受试者。个性化联邦学习可以解决这一问题：所有受试者共同学习一个主干，而每个受试者保留自己的头部。我们将个性化联邦学习适配到SPDNet上，并以EEGNet作为欧几里得基线，研究了其相对于标准联邦学习和集中式训练的效果。实验涵盖了三个运动想象数据集，这些数据集在通道数、受试者数量和类别数方面覆盖了多种不同的情形。我们观察到，个性化SPDNet比标准联邦学习和集中式训练都取得了更高的准确率，同时比标准联邦学习收敛轮数更少、通信参数量更少，并且在各种情形下都优于EEGNet。

    arXiv:2609.29037v1 Announce Type: cross  Abstract: Federated learning (FL) lets EEG decoders learn from recordings of several subjects without pooling them. We consider two light EEG decoders, the Riemannian SPDNet and the Euclidean EEGNet. Both split into a trunk, which builds a latent representation, and a head, which classifies it. Inter-subject variability, however, makes a single shared FL model a poor fit for each subject. Personalised FL addresses this: all subjects learn a common trunk, and each subject keeps its own head. We adapt it for SPDNet and study its effects against standard FL and centralised training, with EEGNet as a Euclidean baseline. Experiments cover three motor-imagery datasets that span diverse regimes in channels, subjects and classes. We observe that personalised SPDNet reaches higher accuracy than both standard FL and centralised training, while converging in fewer rounds and communicating fewer parameters than standard FL. It also outperforms every EEGNet 
    
[^26]: 动作信用何时需要更新？

    When Does Action Credit Need Updating?

    [https://arxiv.org/abs/2609.29007](https://arxiv.org/abs/2609.29007)

    该论文提出成对分支敏感度来判断策略更新引起的漂移是否会推翻原有动作排序，从而仅在必要时利用一阶锚定信用迁移估计器基于旧干预轨迹更新历史动作信用，大幅降低工具使用智能体反复更新的成本。

    

    使用工具的智能体会随着新的交互数据不断更新。然而，每次策略更新后，先前估计的动作信用可能会变得过时。从头重新计算这些信用可能需要大量额外的工具调用和环境交互，使得反复更新的成本越来越高。我们提出了一个简单的问题：历史动作信用究竟何时才真正需要更新？我们的关键观察是：动作价值的改变并不一定意味着决策的改变。只要策略引起的漂移不足以推翻现有的动作排序，历史信用就仍然有用。基于这一思想，我们引入了成对分支敏感度，用以衡量策略更新对区分两个候选动作的下游区域的影响强度。随后，我们推导出一阶锚定信用迁移估计器，利用旧的干预轨迹来更新历史信用，并提出了一种 De……（原文摘要在此处被截断）

    arXiv:2609.29007v1 Announce Type: new  Abstract: Tool-using agents are continually updated with new interaction data. After each policy update, however, previously estimated action credits may become stale. Recomputing them from scratch can require many additional tool calls and environment interactions, making repeated updates increasingly expensive. We ask a simple question: when does historical action credit actually need to be updated? Our key observation is that a change in action value does not necessarily imply a change in the decision. Historical credit can still be useful as long as policy-induced drift is too small to overturn the existing action ranking. Building on this idea, we introduce pairwise branch sensitivity to capture how strongly a policy update affects the downstream regions that distinguish two candidate actions. We then derive a first-order anchored credit-transport estimator that updates historical credit using old interventional trajectories, and propose a De
    
[^27]: 回归定义：通过轨迹图估计智能体强化学习中的步骤级优势

    Back to the Definition: Estimating Step-Level Advantages via Trajectory Graphs for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2609.28963](https://arxiv.org/abs/2609.28963)

    提出通过轨迹图来估计步骤级优势，解决了GRPO等分组强化学习方法在步骤层面因轨迹级粗粒度估计而产生的系统性偏差问题。

    

    基于分组的强化学习方法，如GRPO及其变体，已成为训练推理型和智能体型大语言模型（LLM）的主流范式。虽然其分组归一化的优势估计在响应层面是可靠的，但在步骤层面却存在系统性偏差，因为粗粒度的轨迹级优势难以准确反映单个步骤的贡献（即失败的轨迹可能包含有价值的步骤）。通过重新审视强化学习的基础定义，我们注意到GRPO在单轮任务上的成功源于其优势估计策略遵循了基本定义：从同一状态采样的多个动作的平均奖励构成可信的状态价值估计。将这种忠实的估计扩展到步骤层面，原则上需要从每个中间状态采样多个动作，但这在逐状态的基础上成本过高。为了缓解……

    arXiv:2609.28963v1 Announce Type: new  Abstract: Group-based reinforcement learning (RL) methods, such as GRPO and its variants, have become a leading paradigm for training reasoning and agentic large language models (LLMs). While their group-normalized advantage estimation is reliable at the response level, it becomes systematically biased at the step level, since coarse-grained trajectory-level advantages are hard to accurately reflect the contribution of individual steps (i.e, failed trajectories may contain valuable steps). Revisiting the foundational RL definition, we notice that GRPO's success on single-turn tasks stems from its advantage estimation strategy, which adheres to the basic definition: the mean reward of multiple actions sampled from the same state constitutes a credible state-value estimate. Extending the faithful estimation to step-level would in principle demand sampling multiple actions from each intermediate state, which is too costly on a per-state basis. To mit
    
[^28]: 潜在空间中深度聚类的选择性推断

    Selective Inference for Deep Clustering in Latent Spaces

    [https://arxiv.org/abs/2609.28756](https://arxiv.org/abs/2609.28756)

    本文针对使用固定预训练编码器的深度聚类提出了一个选择性推断框架，通过应对从原始数据空间到潜在空间的非线性变换所带来的复杂选择过程，为聚类结果的统计可靠性检验提供了计算可行且有效的 p 值。

    

    深度聚类是一种强大的方法，它通过在聚类之前学习低维潜在表示来发现高维数据中的有意义结构。尽管其在实践中取得了成功，但评估所得聚类的统计可靠性仍然具有挑战性。在同一数据上检验所发现的聚类会引入选择偏差，并使经典的 p 值失效。选择性推断为纠正这种偏差提供了一个有原则的框架，但现有方法主要针对直接在观测特征上进行的聚类。在这项工作中，我们为使用固定预训练编码器的深度聚类开发了一个选择性推断框架。关键挑战在于，聚类分配是通过从原始数据空间到潜在空间的非线性变换来确定的，这使得选择过程比传统聚类复杂得多。我们的方法提供了一种计算上可行的方式来……

    arXiv:2609.28756v1 Announce Type: cross  Abstract: Deep clustering is a powerful approach for discovering meaningful structures in high-dimensional data by learning a low-dimensional latent representation prior to clustering. Despite its empirical success, assessing the statistical reliability of the resulting clusters remains challenging. Testing discovered clusters on the same data induces selection bias and invalidates classical $p$-values. Selective inference (SI) provides a principled framework for correcting this bias, but existing methods focus on clustering performed directly on the observed features. In this work, we develop an SI framework for deep clustering with a fixed pretrained encoder. The key challenge is that cluster assignments are determined through a nonlinear transformation from the original data space to the latent space, resulting in a substantially more complex selection process than in conventional clustering. Our method provides a computationally tractable wa
    
[^29]: 高维高斯老虎机中的精确贝叶斯后悔与渐近最优性

    Exact Bayes Regret and Asymptotic Optimality in High-Dimensional Gaussian Bandits

    [https://arxiv.org/abs/2609.28718](https://arxiv.org/abs/2609.28718)

    本文研究时间范围与维度成比例的高维高斯贝叶斯线性老虎机，证明了归一化后验不确定性在所有因果策略上具有一致显式极限，由此推导出汤普森采样等策略的精确后悔曲线，并证明后验均值贪婪选择达到极限最优贝叶斯后悔，而汤普森采样的后悔严格更大。

    

    我们研究贝叶斯线性老虎机问题，其中参数服从各向同性高斯分布，候选臂为相互独立的高斯分布，奖励噪声为高斯噪声，且时间范围与维度成比例。归一化后的后验不确定性具有显式极限，且该极限在所有因果策略上一致。随后，利用高斯后验恒等式确定极限参数重叠，无需假设自适应递归的闭合性。这些结果为汤普森采样、后验均值贪婪选择以及一族缩放后验采样协方差的策略给出了精确的后悔曲线。归一化的实际累积后悔在L1范数下收敛，且在紧的比例时间区间上一致。一个策略一致的下界确定了极限最优贝叶斯后悔，并证明后验均值贪婪选择能够达到该下界。汤普森采样的领先后悔严格更大，其相对于贪婪选择的瞬时后悔比率……

    arXiv:2609.28718v1 Announce Type: cross  Abstract: We study Bayesian linear bandits with an isotropic Gaussian parameter, independent Gaussian candidate arms, and Gaussian reward noise when the horizon is proportional to the dimension. The normalized posterior uncertainty has an explicit limit that is uniform over all causal policies. Gaussian posterior identities then determine the limiting parameter overlaps without an assumed closure of the adaptive recursion. These results yield exact regret curves for Thompson sampling, posterior-mean greedy selection, and a family of policies that scale the posterior sampling covariance. The normalized realized cumulative regret converges in L1, uniformly on compact proportional-time intervals. A policy-uniform lower bound identifies the limiting optimal Bayes regret and proves that posterior-mean greedy selection attains it. Thompson sampling incurs a strictly larger leading regret; its instantaneous regret ratio relative to greedy selection lie
    
[^30]: 迭代乘法任务上RLVR的优化地形可以是良性的：来自自旋玻璃理论的洞见

    RLVR landscapes for iterated multiplications can be benign: Insights from spin-glass theory

    [https://arxiv.org/abs/2609.28625](https://arxiv.org/abs/2609.28625)

    本文借助自旋玻璃理论证明，对于迭代乘法等算法任务，RLVR的优化地形是良性的（不存在局部极小值陷阱），其训练困难主要来自扩散屏障和梯度估计误差，而非地形本身的陷阱。

    

    尽管可验证奖励强化学习（RLVR）十分重要，但它能在多大程度上学习到新的推理能力仍存在争议。本文研究了RLVR在算法任务（如迭代群乘法与拟群乘法）上的优化地形。为此，我们将基于短视表格策略的熵正则化RLVR映射为确定性策略上的能量模型（自旋玻璃模型）。这一映射给出了RLVR所能达到性能的上界，并使我们能够在该表格设置下严格刻画优化地形。我们从理论和实验两方面证明，对于一大类具有不相关输入的模型和任务，该优化地形是良性的，不存在可能困住RLVR训练的局部极小值。相反，这些任务在实际中的困难似乎至少部分源于扩散屏障以及穿越优化地形时的梯度估计误差等问题。这些是真正的障碍……

    arXiv:2609.28625v1 Announce Type: new  Abstract: Despite the importance of reinforcement learning with verifiable rewards (RLVR), the extent to which it can learn new reasoning capabilities remains debated. Here we study the optimization landscape of RLVR on algorithmic tasks, such as iterated group and quasigroup multiplication. To this end, we map entropy-regularized RLVR over myopic tabular policies onto an energy-based (spin-glass) model over deterministic policies. This mapping upper-bounds what RLVR can achieve, and lets us rigorously characterize the landscape in this tabular setting. We show, both theoretically and experimentally, that for a wide class of models and tasks with uncorrelated inputs, this landscape is benign, containing no local minima that could trap RLVR training. Rather, the practical difficulty of these tasks appears to stem, at least in part, from issues such as diffusive barriers and gradient-estimation error in traversing the landscape. These are genuine ob
    
[^31]: 基于模拟退火的三阶朗之万动力学在非凸优化中的全局收敛性

    Global Convergence of Third-Order Langevin Dynamics for Non-Convex Optimization via Simulated Annealing

    [https://arxiv.org/abs/2609.28611](https://arxiv.org/abs/2609.28611)

    该论文证明了在模拟退火框架下，采用固定摩擦与递减噪声的三阶朗之万动力学在非凸优化中可依概率收敛到全局最小值，并给出了离散化格式保持该收敛速率的充分步长条件。

    

    我们研究了在固定摩擦力与递减噪声的模拟退火框架下，三阶朗之万动力学用于非凸优化的全局收敛性保证。一个显式的三块扭曲熵结构将耗散从含噪的辅助变量传递到整个状态空间。在耗散性、正则性以及低温泛函不等式假设下，对数冷却调度使目标值以势垒控制的动力学速率依概率收敛到全局最小值。对于精确力积分和中点三阶段离散化格式，多项式递减的步长可在物理时间尺度上保持该收敛速率。三次局部端点估计给出了比现有冻结力动力学结果更宽松的充分步长条件。与单梯度UBU积分器的比较表明，在相同的强耦合分析框架下，其中心化随机局部误差会带来更小的充分（步长条件要求）。

    arXiv:2609.28611v1 Announce Type: cross  Abstract: We study global convergence guarantees of third-order Langevin dynamics for non-convex optimization via simulated annealing with fixed friction and decreasing noise. An explicit three-block distorted entropy transfers dissipation from the noisy auxiliary variable to the full state. Under dissipativity, regularity, and low-temperature functional-inequality assumptions, logarithmic cooling drives the objective values to the global minimum in probability at the barrier-controlled kinetic rate. For the exact-force-integral and midpoint three-stage discretizations, polynomially decreasing steps preserve this rate on the physical time scale. The cubic local endpoint estimate gives a less restrictive sufficient step-size condition than the available frozen-force kinetic result. A comparison with the one-gradient UBU integrator shows how its centered stochastic local error leads, under the same strong-coupling analysis, to a smaller sufficient
    
[^32]: NumericJev：基于多路决策树的类Jev大语言模型数值解码

    NumericJev: Jev-like LLM Numerical Decoding with Multiway Decision Trees

    [https://arxiv.org/abs/2609.28587](https://arxiv.org/abs/2609.28587)

    提出了一种无需训练的数值解码算法NUMERICJEV，通过多路决策树递归细化数值范围，使任何具有类Jev结构化选择接口的大语言模型都能输出数值，其性能甚至超过从包含正确答案的候选列表中直接选择。

    

    大语言模型能够理解自然语言，但做出稳健的决策仍然具有挑战性。类Jev模型可以暴露结构化的选项，但这些接口无法直接以所要求的精度提供数值。我们提出了NUMERICJEV，这是一种无需训练的数值解码算法，能够使任何具有类Jev结构化选择接口的大语言模型输出数值。令人惊讶的是，在我们的算术基准测试中，它的表现比从包含正确答案的候选列表中直接选择高出2.93个百分点（图1）。我们的研究动机来自这样一个观察：数值范围选择本身就是一个类Jev大语言模型能够解决的决策问题。NUMERICJEV通过多路决策树递归地细化数值范围，同时在上下文中保留原始问题，且无需参数更新或访问隐藏状态。在100个值的网格上，十路树只需要两轮决策即可完成。

    arXiv:2609.28587v1 Announce Type: cross  Abstract: Large language models can interpret natural lan- guage, yet robust decisions remain challenging. Jev-like models expose structured choices, but these interfaces do not directly provide numeri- cal values at a requested precision. We propose NUMERICJEV, a training-free numerical decod- ing algorithm that enables numerical output from any LLM with a Jev-like structured-choice in- terface. Surprisingly, on our arithmetic bench- mark, it outperforms direct selection from a can- didate list containing the correct answer by 2.93 percentage points (Figure 1). Our motivation comes from the observation that numerical range selection is itself a decision problem that Jev- like LLMs can address. NUMERICJEV recur- sively refines a range through a multiway deci- sion tree while retaining the original question in context, without parameter updates or hidden- state access. On a 100-value grid, a ten-way tree requires only two decision rounds. Range- 
    
[^33]: SGA：时间序列基础模型中多步预测的不确定性量化

    SGA: Uncertainty Quantification for Multi-Step Forecasting in Time Series Foundation Models

    [https://arxiv.org/abs/2609.28582](https://arxiv.org/abs/2609.28582)

    本文提出SGA方法，通过有向无环图表征预测分支拓扑结构并度量其图复杂度，实现了对时序基础模型多步预测不确定性的有效量化，提升了预测结果的可信度。

    

    arXiv:2609.28582v1 公告类型：cross。摘要：近来时序基础模型的出现显著提升了多步预测的性能，使其能够在较长的未来时间范围内做出准确预测。然而，现有的时序基础模型往往存在显著的固有不确定性，这种不确定性通常表现为在每个时间步衍生出预测分支并向后续步骤扩散；不同的预测分支往往展现出不同的预测表现，从而削弱了时序基础模型预测结果的可信度。在本文中，我们提出了切片-图化-对齐（Slicing-Graphing-Alignment，SGA）方法来量化时序基础模型多步预测的不确定性。所提出的SGA首先利用有向无环图表征所有潜在预测分支的拓扑结构，使图复杂度能够约束多步预测的不确定性，然后通过融合拓扑信息与时序基础模型固有的随机特性来精确度量图复杂度……

    arXiv:2609.28582v1 Announce Type: cross  Abstract: The recent emergence of Time Series Foundation Models (TSFMs) has significantly advanced multi-step forecasting performance, enabling accurate predictions over extended future horizons. However, existing TSFMs often suffer from significantly inherent uncertainty, which typically manifests as derived forecast branches emerging at each time step and spreading to subsequent steps; different forecast branches often exhibit varying forecasting performance, thereby undermining the credibility of TSFM forecasts. In this paper, we propose the Slicing-Graphing-Alignment (SGA) method to quantify the uncertainty of multi-step TSFM forecasts. The proposed SGA first characterizes the topology of all potential forecast branches using a directed acyclic graph, such that the graph complexity bounds the uncertainty of multi-step forecasts, and then precisely measures the graph complexity by integrating both topological information and TSFM-inherent sto
    
[^34]: 矩阵聚合算子

    Matrix Aggregation Operators

    [https://arxiv.org/abs/2609.28562](https://arxiv.org/abs/2609.28562)

    本文首次形式化了矩阵聚合算子（MAO）的概念，将聚合理论从向量拓展到矩阵结构，并分析了其可分解性与对称性，证明某些算子无法分解为逐行逐列聚合的形式。

    

    聚合理论传统上主要关注定义在向量上的算子。然而，许多应用——包括多准则决策、群体决策、基于模糊规则的分类系统以及重叠/分组指数等——需要聚合的信息天然地以隶属度矩阵的形式结构化（例如，一个对象集合与一族模糊集交互的情形）。尽管如此，针对这类算子尚未提出正式的框架，部分原因在于通常将矩阵展平为向量的做法（这会丢弃结构信息），部分原因在于依赖于按顺序聚合行和列的可分解算子。本文通过形式化矩阵聚合算子的概念来填补这一空白。我们分析了矩阵聚合算子的可分解性与对称性，证明了某些算子无法用可分解形式表达，并考察了若干……（摘要在此处截断）

    arXiv:2609.28562v1 Announce Type: cross  Abstract: Aggregation theory has traditionally focused on operators defined over vectors. However, many applications-including Multi-Criteria Decision Making, Group Decision Making, Fuzzy Rule-Based Classification Systems, and overlap/grouping indices-require aggregating information naturally structured as a matrix of membership degrees (e.g., where a set of objects interacts with a family of fuzzy sets). Despite this, no formal framework has been proposed for this class of operators, partly due to the common practice of flattening matrices into vectors (which discards structural information) and partly due to a reliance on decomposable operators that aggregate rows and columns sequentially. This paper addresses this gap by formalizing the notion of a matrix aggregation operator (MAO). We analyze the decomposability and symmetry properties of MAOs, showing that certain operators cannot be expressed in decomposable form and examining several noti
    
[^35]: 随机大语言模型的推测式评估

    Speculative Evaluation of Stochastic LLMs

    [https://arxiv.org/abs/2609.28560](https://arxiv.org/abs/2609.28560)

    该论文提出了一种基于分层贝叶斯尼曼策略的推测式评估方法（HBN及异步版本HBN-async），通过分层贝叶斯模型估计各任务方差并自适应地分配推演预算，从而在固定预算下最小化随机大语言模型基准评估的方差。

    

    评估一个随机的大语言模型代价高昂：基准测试分数通过随机化的多次推演（rollouts）来估计期望性能，然而均匀的重复采样忽略了任务级推演方差之间的显著差异。我们研究如何在固定的推演预算下最小化基准测试均值估计的方差。我们提出了基于分层贝叶斯尼曼（HBN）策略的推测式评估方法，其中试点规模和阶段权重在事前联合确定。该方法首先运行一个简短的均匀试点，将各任务的成功计数与分层贝叶斯模型相结合，并使用任务级抽样方差的后验期望进行精确的正整数尼曼分配。为了缓解试点同步障碍，HBN-async 根据部分试点反馈推测性地执行后续推演，并保留最终分配所选中的那些推演结果。在六个检查点和18个基准测试组上，我们评估了107个非退化的基准-检查点组合。对于推演预算……

    arXiv:2609.28560v1 Announce Type: cross  Abstract: Evaluating a stochastic large language model is costly: benchmark scores estimate expected performance from randomized rollouts, yet uniform repetition ignores sharp differences in task-level rollout variance. We ask how to minimize the variance of a fixed-benchmark mean under an exact rollout budget. We develop Speculative Evaluation with a Hierarchical Bayesian Neyman (HBN) policy with pilot size and stage weight jointly chosen ex ante. It runs a short uniform pilot, pools per-task success counts with a hierarchical Bayesian model, and uses posterior expectations of task-level sampling variances for exact positive-integer Neyman allocation. To mitigate the pilot synchronization barrier, HBN-async speculatively executes continuations from partial pilot feedback and retains those selected by the final allocation. Across six checkpoints and 18 benchmark groups, we evaluate 107 nondegenerate benchmark-checkpoint profiles. For rollout bud
    
[^36]: 一致归纳推理的序理论刻画

    An Order-Theoretic Characterization of Consistent Inductive Inference

    [https://arxiv.org/abs/2609.28551](https://arxiv.org/abs/2609.28551)

    该论文在ZFC框架下通过有限可实现轨迹上的一个线性序（要求冲突轨迹选择不同子轨迹、且对每个固定目标良基）完整刻画了一致归纳推理，证明一致性等价于此类序的存在性，并回答了Lu（2024）的问题。

    

    在什么条件下，学习者能够沿着由某个固定但未知假设所标注的每条无限序列，仅做出有限多次预测错误？我们在ZFC公理体系下对任意二值假设类刻画了这种一致性形式，且不要求一致错误界。该刻画基于有限可实现轨迹上的单个线性序：每条轨迹选择其最小子轨迹，且该序必须满足两个条件——相互冲突的轨迹选择不同的子轨迹，以及该序在每个固定目标的轨迹集上是良基的。这些条件诱导出一个学习者，其每次犯错时所选证据都会严格递减。反之，一个一致的学习者可通过典范错误记录和Kleene–Brouwer排序构造出这样一个序。该结果提供了用有限证据来表示一致预测的方法，回答了Lu（2024）提出的一个问题。

    arXiv:2609.28551v1 Announce Type: cross  Abstract: When can a learner make only finitely many prediction errors along every infinite sequence labeled by a fixed, unknown hypothesis? We characterize this form of consistency for arbitrary binary hypothesis classes in ZFC, without requiring a uniform mistake bound. The characterization uses a single linear order on finite realizable traces. Each trace selects its least subtrace, and the order must satisfy two conditions: conflicting traces select different subtraces, and the order is well-founded on the traces of each fixed target. These conditions induce a learner whose selected evidence decreases on every mistake. Conversely, a consistent learner yields such an order through canonical mistake transcripts and the Kleene--Brouwer ordering. The result provides a representation of consistent prediction by finite evidence, answering a question of Lu (2024).
    
[^37]: 随机惯性 Krasnosel'skii-Mann 迭代实现近乎最优的样本复杂度

    Stochastic Inertial Krasnosel'skii-Mann Iteration Achieves Near-Optimal Sample Complexity

    [https://arxiv.org/abs/2609.28543](https://arxiv.org/abs/2609.28543)

    本文提出一种随机惯性 Krasnosel'skii-Mann（iKM）迭代方法，仅在随机 KM 算法上添加两个惯性外推、且每次更新只需一次可能有偏随机预言机调用的条件下，实现了 Õ(ε⁻²) 的近乎最优样本复杂度。

    

    我们分析了一种简单的随机惯性 Krasnosel'skii-Mann（iKM）方法，用于在实希尔伯特空间中求非扩张算子的不动点。该方法只需在随机 KM 算法 [Bravo and Cominetti, 2024] 的基础上添加两个惯性外推即可得到，并且每次更新仍只需调用一次可能有偏的随机预言机，同时在随机和确定性两种情形下都达到了尖锐的速率。具体而言，利用我们提出的参数调度方案，我们证明了如下的最后迭代不动点残差界：O(1/K + σ log K/√K + B_K log K/K)，其中 K 为迭代时域长度，σ 为噪声水平，B_K 为累积的均方根偏差。当 B_K=O(√K) 时，该界给出 Õ(ε⁻²) 的样本复杂度，在对数因子范围内匹配了在我们模型的无偏子类下给出的随机预言机下界 [Foster et al., ...]。

    arXiv:2609.28543v1 Announce Type: cross  Abstract: We analyze a simple stochastic inertial Krasnosel'skii--Mann (iKM) method for finding a fixed point of a nonexpansive operator in a real Hilbert space. Our method is obtained simply by adding two inertial extrapolations to stochastic KM [Bravo and Cominetti, 2024], and it retains one call to a possibly biased stochastic oracle per update and achieves sharp rates in both the stochastic and deterministic regimes. Specifically, with our proposed parameter schedule, we prove the following last-iterate fixed-point residual bound: \[   {O}\!\left(\frac{1}{K} +\frac{\sigma\log K}{\sqrt K} +\frac{B_K\log K}{K}\right), \] where $K$ is the horizon, $\sigma$ is the noise level and $B_K$ is the accumulated root-mean-square bias. When $B_K=O(\sqrt K)$, this yields $\widetilde O(\epsilon^{-2})$ sample complexity that matches, up to a logarithmic factor, the stochastic-oracle lower bound given under the unbiased subclass of our model [Foster et al., 
    
[^38]: 覆盖率约束保形模型选择的序贯置信集

    Sequential Confidence Sets for Coverage-Constrained Conformal Model Selection

    [https://arxiv.org/abs/2609.28522](https://arxiv.org/abs/2609.28522)

    提出了覆盖率约束序贯模型置信集（CC-SMCS），利用同步鞅置信序列与精确闭式规则，以至少1-δ的概率识别出满足覆盖率硬约束且成本最小的保形预测流水线。

    

    现代保形预测系统通常维护多个自适应流水线，它们在基础预测器、一致性分数、校准窗口和更新规则上各不相同。比较这些流水线十分困难，因为覆盖率是一个硬性约束，而效率只应在可行的流水线之间进行优化。我们将该问题形式化为针对随机约束 argmin 的序贯推断问题。在每个时刻，目标集合是满足多个前缀平均条件失覆盖率约束的最小成本流水线集合。我们提出了覆盖率约束序贯模型置信集，它将流水线区分为可证明可行、可能可行和可能约束最优三类。利用同步鞅置信序列，CC-SMCS 将矩形置信区域投影到约束 argmin 上，并给出精确的闭式规则。以至少 $1-\delta$ 的概率，它包含所有约束最优的流水线。

    arXiv:2609.28522v1 Announce Type: cross  Abstract: Modern conformal forecasting systems often maintain several adaptive pipelines that differ in base forecasters, conformity scores, calibration windows, and update rules. Comparing them is difficult because coverage is a hard constraint, whereas efficiency should be optimized only among feasible pipelines. We formulate this problem as sequential inference for a stochastic constrained argmin. At each time, the target is the set of minimum-cost pipelines satisfying multiple prefix-average conditional miscoverage constraints. We introduce Coverage-Constrained Sequential Model Confidence Sets (CC-SMCS), which separate certifiably feasible, possibly feasible, and possibly constrained-optimal pipelines. Using simultaneous martingale confidence sequences, CC-SMCS projects a rectangular confidence region onto the constrained argmin and admits an exact closed-form rule. With probability at least $1-\delta$, it contains every constrained-optimal 
    
[^39]: 认证的任务条件化主动可观测性

    Certified Task-Conditioned Active Observability

    [https://arxiv.org/abs/2609.28520](https://arxiv.org/abs/2609.28520)

    该论文形式化了“认证的任务条件化主动可观测性复杂度”，即在认证误差与安全弃权保证下识别任务相关状态所需的最小最坏情况期望交互代价，并证明任务预测等价性诱导出唯一的最小充分商空间，使主动可观测性复杂度在其上严格不变。

    

    在对不可观测的物理系统采取行动之前，自主智能体必须确定哪些潜在区分支配下游任务、需要多少次主动干预才能认证这些区分、以及何时应当放弃行动以防止灾难性错误。经典可观测性将状态重构视为一个无条件的二元谓词，当被动观测无法在不施加扰动的情况下打破潜在简并、完全的微观反演代价过高、以及区分与任务无关的自由度浪费交互预算时，该方法便会失效。我们形式化了任务条件化主动可观测性复杂度：即在认证误差与安全弃权保证下，识别任务相关状态所需的最小最坏情况期望交互代价。我们证明任务预测等价性诱导出唯一的最小充分商 $\mathcal{H}/\!\sim_\tau$，使得主动可观测性复杂度在该商上严格保持不变……（摘要原文在此处截断）

    arXiv:2609.28520v1 Announce Type: cross  Abstract: Before acting upon an unobservable physical system, an autonomous agent must determine which latent distinctions govern downstream tasks, how many active interventions are necessary to certify them, and when to abstain to prevent catastrophic errors. Classical observability treats state reconstruction as an unconditioned binary predicate, failing when passive observations cannot break latent degeneracies without perturbation, full microscopic inversion is prohibitively costly, and distinguishing task-irrelevant degrees of freedom wastes interaction budgets. We formalize task-conditioned active observability complexity: the minimum worst-case expected interaction cost required to identify task-relevant states under certified error and safe abstention guarantees. We prove that task-predictive equivalence induces the unique minimal sufficient quotient $\mathcal{H}/\!\sim_\tau$, leaving active observability complexity strictly invariant wh
    
[^40]: 从预测到可解释的医疗服务提供者行为画像：面向欺诈、浪费与滥用审查

    From Prediction to Explainable Provider Behavior Profiles for Fraud, Waste, and Abuse Review

    [https://arxiv.org/abs/2609.28477](https://arxiv.org/abs/2609.28477)

    该研究提出将欺诈、浪费与滥用（FWA）审查从预测建模转向可解释的提供者行为画像，通过将账单收入分解为提供者规模与诊疗项目构成的乘积，从而解释提供者行为变化的原因。

    

    理赔数据可以显示医疗服务提供者的行为发生了变化，但仅凭数据本身无法解释原因。欺诈、浪费与滥用（FWA）审查需要识别重要的行为、定位驱动这些行为的诊疗代码和资金，并检验合理的解释。一种常见的替代方法——预测建模——通过标记偏离预期使用量预测的异常来发现问题，但预测的价值有限，除非它能够超越简单的持续性预测，并解释偏差为何重要。在我们的季度提供者-诊疗项目数据中，最新观测值已捕获了大部分可预测的变化，而增加模型结构几乎无法提升准确性。残差将增长、服务线变化、代码维护以及不完整的观测与潜在的可疑行为混为一谈，使得单点预测并不完整。因此，我们将提供者审查重新表述为一个描述性表示问题：账单收入 y = s × p，其中 s 衡量提供者规模，p 描述诊疗项目构成。

    arXiv:2609.28477v1 Announce Type: cross  Abstract: Claims data can show that provider behavior changed but cannot by itself explain why. FWA (fraud, waste, and abuse) review requires identifying material behavior, locating the codes and dollars driving it, and testing plausible explanations. A common alternative, predictive modeling, flags deviations from an expected-utilization forecast -- but a forecast has limited value unless it beats simple persistence and explains why a deviation matters. In our quarterly provider-procedure data, the latest observation captures most forecastable variation, and added model structure adds little accuracy. Residuals conflate growth, service-line shifts, code maintenance, and incomplete observation with potentially concerning behavior, making point forecasts incomplete.   We instead formulate provider review as a descriptive representation problem: billed revenue y = s * p, where s measures provider scale and p describes procedure composition. The pr
    
[^41]: 学习波动：因果表格预训练的统计基础

    Learning to Fluctuate: Statistical Foundations for Causal Tabular Pretraining

    [https://arxiv.org/abs/2609.26290](https://arxiv.org/abs/2609.26290)

    提出波动监督预训练（FSP），用平均处理效应加有效影响函数波动来标注合成表格，并证明完全波动可使高斯标签可观测，从而将因果标签预测风险从 $(1-\lambda)^2/n$ 阶降至 $n^{-2}$ 阶。

    

    因果表格基础模型在多个合成机制之间摊销效应估计，但潜在效应监督只奖励后验收缩，而非直接编码固定部署总体中所需的重复样本响应。我们提出波动监督预训练（FSP）：每个合成表格由其平均处理效应加上其有效影响函数的波动来标注，而部署时仍只需一次冻结的前向传播。沿着路径 $T_{\lambda,P}=\theta(P)+\lambda P_n\psi_P$，我们证明了一个端点相变：每个固定的 $\lambda<1$ 都会保留 $(1-\lambda)^2/n$ 阶的标签模糊性，而完全波动使高斯标签变得可观测，并将最优有限层因果标签预测风险降至 $n^{-2}$ 阶。一个有限预训练界综合了标签误差、网络误差、回合采样误差与优化误差；其产生的采样缺陷控制了固定机制偏差、均方……（摘要截断）

    arXiv:2609.26290v1 Announce Type: cross  Abstract: Causal tabular foundation models amortize effect estimation across synthetic mechanisms, but latent-effect supervision rewards posterior shrinkage instead of directly encoding the repeated-sample response needed in a fixed deployment population. We introduce fluctuation-supervised pretraining (FSP): each synthetic table is labeled by its average treatment effect plus its efficient influence-function fluctuation, while deployment remains a single frozen forward pass. Along the path $T_{\lambda,P}=\theta(P)+\lambda P_n\psi_P$, we prove an endpoint transition: every fixed $\lambda<1$ retains label ambiguity of order $(1-\lambda)^2/n$, whereas full fluctuation makes the Gaussian label observable and reduces optimal finite-stratum causal label-prediction risk to order $n^{-2}$. One finite-pretraining bound combines label, network, episode-sampling, and optimization errors; its resulting sampling defect controls fixed-mechanism bias, mean sq
    
[^42]: 坏天才：超越任务特定捷径的反事实引导测试框架演化

    Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts

    [https://arxiv.org/abs/2609.18366](https://arxiv.org/abs/2609.18366)

    提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。

    

    可靠的智能体评估因自动测试框架优化而变得复杂，这类优化方法反复使用已发布的基准 $B_{\mathrm{rel}}$ 来引导一个提议者，该提议者围绕固定的目标智能体编辑提示词、记忆、检索、工具和控制代码。任务保留集虽然改变了语义任务，但基准协议保持不变，因此一个“坏天才”提议者可以生成一个作弊的测试框架，其在发布基准上的性能提升依赖于整个基准范围的捷径。我们提出了反事实测试框架搜索与演化，将测试框架演化建模为在保持有效性的基准反事实上的约束生成问题。在每次提议者更新后，一个挑战者会搜索能大幅摧毁性能提升的可执行协议变换。有效性防火墙检查任务语义是否得到保留，而确认集则决定反事实是否进入有限存档。我们形式化定义了一个精确的捷径中和基准 $B

    arXiv:2609.18366v1 Announce Type: new  Abstract: Reliable agent evaluation is complicated by automatic harness optimization, which repeatedly uses a released benchmark $B_{\mathrm{rel}}$ to guide a Proposer that edits prompts, memory, retrieval, tools, and control code around a fixed target agent. Task holdout varies semantic tasks but leaves the benchmark protocol fixed, so a "bad genius" Proposer can produce a cheating harness whose released-benchmark gain depends on a benchmark-wide shortcut. We introduce Counterfactual Harness Search and Evolution (CHASE), which casts harness evolution as constraint generation over validity-preserving benchmark counterfactuals. After each Proposer update, a Challenger searches for an executable protocol transformation with large gain destruction. A validity firewall checks that task semantics are preserved, while a confirmation set determines whether the counterfactual enters a finite archive. We formalize an exact shortcut-neutralized benchmark $B
    
[^43]: 监督链梯法

    Supervising the Chain Ladder

    [https://arxiv.org/abs/2609.16552](https://arxiv.org/abs/2609.16552)

    本文将链梯法准备金进展模式的选择建模为监督学习问题，通过在严格凸的目标函数上添加可解释的惩罚项与超参数（如数据衰减、权重幂、基准参考与平滑约束），把精算师的专业判断形式化，并可通过单一线性系统求解。

    

    链梯法的加权进展模式最小化一个显式的损失函数，但实务中很少直接照此入账。精算师通常会调整该模式并记录最终调整后的比率。本文将链梯法的进展模式选择视为一个监督学习问题。对模式调整的专业判断被转化为在链梯法损失函数上定义好的惩罚项和超参数框架，该损失函数在此被视为机器学习中的目标函数。数据权重通过引入衰减参数和幂参数进行推广，分别用于近期性和加权控制。基准形态和平滑性约束则通过参考惩罚项和Whittaker-Henderson平滑方法引入。所构建的目标函数是严格凸的，可通过求解一个线性系统得到最小值。每个超参数本身都是一种可解释的调整方式，可由专业判断予以声明，并归类为经验调整或前瞻性调整。经验调整可以被设置得更加客观。

    arXiv:2609.16552v1 Announce Type: cross  Abstract: The chain ladder's volume-weighted pattern minimises an explicit loss function, yet is rarely booked as such. Practitioners adjust the pattern and record the final adjusted ratios. This paper treats the chain ladder's pattern selection as a supervised-learning problem. Judgement on pattern adjustments becomes a framework of defined penalties and hyperparameters on the chain ladder's loss function, treated here as an objective function in machine learning. Data weights are generalised with a decay and a power parameter for recency and volume weighting. Benchmark shaping and smoothness enter through a reference penalty and Whittaker-Henderson smoothing. The assembled objective is strictly convex and minimised by a single linear system. Each hyperparameter becomes an interpretable adjustment in its own right, declarable by judgement and categorised as an experience or a prospective adjustment. Experience adjustments can be set more object
    
[^44]: 置信视界

    Confidence Horizons

    [https://arxiv.org/abs/2608.03889](https://arxiv.org/abs/2608.03889)

    本文提出“置信视界”这一新型统计对象，通过放弃有限时间范围之外的有效性，在预算或伦理等约束下获得更精确的大样本任意时有效推断，并与Pocock、O'Brien-Fleming等经典成组序贯边界建立了明确联系。

    

    任意时有效推断使分析师能够持续监测数据并提前停止实验。然而，这类方法中的大多数由于需要在无限时间范围内保持有效而具有一定的保守性。在实践中，由于预算、实际操作或伦理方面的限制，往往会对时间范围施加一个界限。本文提出了这样一个问题：“是否可以通过放弃超出某个有限时间范围的有效性，来获得更精确的大样本任意时有效推断？”我们对这一问题给出了肯定的回答，提出了一类我们称之为“置信视界”的统计对象。这些对象既可以被视为有界时间范围内的大样本置信序列，也可以被视为具有最大中期查看次数的成组序贯重复置信区间。我们明确建立了其与Pocock [1977]、O'Brien-Fleming [1979] 以及Wang-Ts等经典成组序贯边界的联系。

    arXiv:2608.03889v2 Announce Type: replace-cross  Abstract: Anytime-valid inference enables analysts to continuously monitor their data and stop experiments early. However, the majority of these methods incur a certain conservativeness by remaining valid on infinite time horizons. In practice, a bound on the horizon may be imposed due to budgetary, practical, or ethical constraints. In this paper, we ask the question: "Is it possible to obtain sharper large-sample anytime-valid inference by forgoing validity beyond some finite time horizon?". We provide a positive answer to this question by proposing a family of statistical objects that we call "confidence horizons". These objects can be viewed as large-sample confidence sequences on bounded time horizons, or alternatively as group sequential repeated confidence intervals with a maximal number of interim peeking times. We make explicit connections to the group sequential boundaries of Pocock [1977], O'Brien--Fleming [1979], and Wang--Ts
    
[^45]: 你只需要对数

    All you need is log

    [https://arxiv.org/abs/2606.27349](https://arxiv.org/abs/2606.27349)

    本文刻画了在数据处理下单调且在独立乘积上可加的多分布泛函的唯一形式，即通过多路重合散度在四层参数空间上的正积分来统一表示，解决了Rényi族向多分布泛化的开放问题。

    

    arXiv:2606.27349v1 公告类型：交叉 摘要：比较两个概率分布是统计学和机器学习的基本构建块，而正确的族已被充分理解：阶数为α∈[0,∞]的Rényi散度是在数据处理下单调且在独立乘积上可加的唯一族。许多问题却需要同时比较两个以上的分布——多群体公平性、多先验PAC-Bayes界、多假设检验——而Rényi族的多分布泛化正确形式一直是一个开放问题。我们对此进行了刻画。每个在数据处理下单调且在独立乘积上可加的W元分布泛函，都可以表示为多路重合散度C_α(π_1,…,π_W) := -log∫ π_1^{α_1}…π_W^{α_W}（其中∑_k α_k = 1）在具有四个分层参数空间上的正积分：单纯形内部；混合符号指数锥。

    arXiv:2606.27349v1 Announce Type: cross  Abstract: Comparing two probability distributions is a basic building block of statistics and machine learning, and the right family is well understood: the R\'enyi divergences of order $\alpha\in[0,\infty]$ are the unique family monotone under data processing and additive on independent products. Many problems instead compare more than two distributions at once -- multi-population fairness, multi-prior PAC-Bayes bounds, multi-hypothesis testing -- and the right multi-distribution generalization of the R\'enyi family has been an open question.   We characterize it. Every functional of $W$-tuples of distributions that is monotone under data processing and additive on independent products is a positive integral of multi-way coincidence divergences $C_{\alpha}(\pi_1,\dots,\pi_W) := -\log\int \pi_1^{\alpha_1}\cdots\pi_W^{\alpha_W}$ (with $\sum_k \alpha_k = 1$) over a parameter space with four strata: the simplex interior; mixed-sign exponent cones (
    
[^46]: MultiwayPAM：用于LLM-as-a-Judge评分分析的多路围绕中心点划分方法

    MultiwayPAM: Multiway Partitioning Around Medoids for LLM-as-a-Judge Score Analysis

    [https://arxiv.org/abs/2603.10287](https://arxiv.org/abs/2603.10287)

    该论文提出了MultiwayPAM，一种新的张量聚类方法，能够同时估计LLM-as-a-Judge评分张量各模式的聚类成员和中心点，从而揭示LLM评估器评分偏差的结构。

    

    LLM-as-a-Judge是一种灵活的文本评估框架，通过更改提示模板，我们可以从多个角度获得对给定文本质量的评分。使用LLM-as-a-Judge的两大主要挑战是：使用大语言模型（LLM）进行推理的计算成本（尤其是在评估大量实例时），以及LLM评估器固有的偏差。为了解决这些问题并揭示LLM评估器造成的评分偏差结构，我们提出将张量聚类方法应用于给定的LLM-as-a-Judge评分张量，该张量的元素是不同问题、回答者和评估者组合所对应的评分。具体而言，我们开发了一种新的张量聚类方法MultiwayPAM，利用该方法可以同时估计给定数据张量每个模式的聚类成员关系和中心点。通过观察MultiwayPAM获得的中心点，我们可以获得关于模型行为的相关知识。

    arXiv:2603.10287v2 Announce Type: replace-cross  Abstract: LLM-as-a-Judge is a flexible framework for text evaluation, which allows us to obtain scores for the quality of a given text from various perspectives by changing the prompt template. Two main challenges in using LLM-as-a-Judge are computational cost of inference using a large language model (LLM), especially when evaluating a large number of instances, and inherent bias of an LLM evaluator. To address these issues and reveal the structure of score bias caused by an LLM evaluator, we propose to apply a tensor clustering method to a given LLM-as-a-Judge score tensor, whose entries are the scores for different combinations of questions, answerers, and evaluators. Specifically, we develop a new tensor clustering method MultiwayPAM, with which we can simultaneously estimate the cluster membership and the medoids for each mode of a given data tensor. By observing the medoids obtained by MultiwayPAM, we can gain knowledge about the m
    
[^47]: 以观测集合为条件的逆问题：应用与方法

    Inverse Problems Conditioned on Observation Ensembles: Applications and Methods

    [https://arxiv.org/abs/2601.22029](https://arxiv.org/abs/2601.22029)

    该论文提出了一类新的统计问题——集合条件逆问题（EIP），并基于一种利用观测集合信息的新型条件生成模型（集合逆生成模型），给出了非迭代的推理时后验采样方法，可应用于高能物理解折叠、全波形反演和逆成像等领域。

    

    我们引入了一类新的多元统计问题，我们称之为“集合条件逆问题”。EIP的目标是对一个按照先验在前向过程下的推前分布而分布的集合进行反演。在高能物理（HEP）中，这与一个广为人知的问题——解折叠相关，其目标是从被探测器效应扭曲的观测中重建真实的物理分布。EIP也出现在全波形反演（FWI）以及具有未知先验的逆成像问题中。我们提出了一类非迭代的推理时方法，基于一种新的条件生成模型类别来构建后验采样器，我们将其称为集合逆生成模型。在后验建模中，这些模型在单个观测的基础上，还额外利用了观测集合中所包含的集合信息。与现有方法不同，我们提出的方法避免了显式和迭代……

    arXiv:2601.22029v2 Announce Type: replace  Abstract: We introduce a new multivariate statistical problem that we refer to as the Ensemble-conditioned Inverse Problem (EIP). The aim of EIP is to invert for an ensemble that is distributed according to the pushforward of a prior under a forward process. In high energy physics (HEP), this is related to a widely known problem called unfolding, which aims to reconstruct the true physics distribution from observations that are distorted by detector effects. The EIP also arises in full waveform inversion (FWI) and inverse imaging with unknown priors. We propose non-iterative inference-time methods that construct posterior samplers based on a new class of conditional generative models, which we call ensemble inverse generative models. For the posterior modeling, these models additionally use the ensemble information contained in the observation set on top of single observations. Unlike existing methods, our proposed methods avoid explicit and i
    
[^48]: 堆叠SVD还是SVD堆叠？随机矩阵理论视角下的数据整合

    Stacked SVD or SVD stacked? A Random Matrix Theory perspective on data integration

    [https://arxiv.org/abs/2507.22170](https://arxiv.org/abs/2507.22170)

    本文借助随机矩阵理论，首次在比例渐近区间下严格比较了Stack-SVD与SVD-Stack这两种估计多数据集共享奇异子空间的主流数据整合方法的理论性能。

    

    arXiv:2507.22170v2 Announce Type: replace-cross 摘要：现代数据分析日益需要在多个高维数据集中识别共享的潜在结构。一个常用的模型假设数据矩阵是具有共享奇异子空间的低秩矩阵的含噪观测。在此情况下，出现了两种用于估计该共享结构的主要方法，它们在跨数据集整合信息的方式上有所不同。第一种方法称为Stack-SVD，它将所有数据集拼接在一起，然后执行奇异值分解（SVD）。第二种方法称为SVD-Stack，它首先对每个数据集分别执行SVD，然后聚合这些数据集的顶部奇异向量，最后计算它们之间的一致性。尽管这些方法被广泛使用，但它们尚未在比例渐近区间（proportional asymptotic regime）下得到严格研究，而在当今数据规模和维度不断增长的世界中，这一区间具有重要的实际意义。

    arXiv:2507.22170v2 Announce Type: replace-cross  Abstract: Modern data analysis increasingly requires identifying shared latent structure across multiple high-dimensional datasets. A commonly used model assumes that the data matrices are noisy observations of low-rank matrices with a shared singular subspace. In this case, two primary methods have emerged for estimating this shared structure, which vary in how they integrate information across datasets. The first approach, termed Stack-SVD, concatenates all the datasets, and then performs a singular value decomposition (SVD). The second approach, termed SVD-Stack, first performs an SVD separately for each dataset, then aggregates the top singular vectors across these datasets, and finally computes a consensus amongst them. While these methods are widely used, they have not been rigorously studied in the proportional asymptotic regime, which is of great practical relevance in today's world of increasing data size and dimensionality. Con
    
[^49]: 通过依赖感知生成建模捕捉未见的空间热极端事件

    Capturing Unseen Spatial Heat Extremes Through Dependence-Aware Generative Modeling

    [https://arxiv.org/abs/2507.09211](https://arxiv.org/abs/2507.09211)

    DeepX-GAN是一种显式捕捉空间依赖性的深度生成模型，能够零样本模拟超出历史记录的统计上合理的“未见”热极端事件，揭示多个地点同时遭受极端高温的隐藏风险。

    

    观测到的气候极端事件记录为潜在灾害提供了不完整的视角，遗漏了超出历史经验的“未见”事件。忽视空间依赖性进一步低估了同时袭击多个地点的灾害风险。我们提出了DeepX-GAN（物理极端事件依赖增强嵌入-生成对抗网络），这是一种明确捕捉罕见极端事件空间结构的深度生成模型。其零样本泛化能力使其能够模拟超出观测记录的统计上合理的极端事件，并通过长期气候模式大集合模拟进行评估。我们定义了两种“未见”类型：直接影响目标地区的“直接命中”型极端事件，以及险些错过目标地区的“擦边”型极端事件。这些未实现的事件揭示了隐藏的风险，既可以促使人们采取主动适应措施，也可能强化一种虚假的抗灾安全感。将DeepX-GAN应用于中东和北非地区的结果表明，概率……（原文摘要在此处截断）

    arXiv:2507.09211v3 Announce Type: replace  Abstract: Observed records of climate extremes provide an incomplete view of plausible hazards, missing "unseen" events beyond historical experience. Ignoring spatial dependence further underestimates hazards striking multiple locations simultaneously. We introduce DeepX-GAN (Dependence-Enhanced Embedding for Physical eXtremes-Generative Adversarial Network), a deep generative model that explicitly captures the spatial structure of rare extremes. Its zero-shot generalizability enables the simulation of statistically plausible extremes beyond the observed record, evaluated against long climate model large-ensemble simulations. We define two unseen types: direct-hit extremes that affect the target, and near-miss extremes that narrowly miss. These unrealized events reveal hidden risks and can either prompt proactive adaptation or reinforce a false sense of resilience. Applying DeepX-GAN to the Middle East and North Africa shows that the probabili
    
[^50]: 无需节拍器的时变贝叶斯优化

    Time-Varying Bayesian Optimization Without a Metronome

    [https://arxiv.org/abs/2501.18963](https://arxiv.org/abs/2501.18963)

    该论文首次推导出显式考虑观测采样频率变化的时变贝叶斯优化遗憾上界，并据此提出了关于数据集规模和过期数据策略的实用建议，其 BOLT 算法在实验中优于现有最先进的 TVBO 方法。

    

    时变贝叶斯优化（TVBO）是优化时变的、昂贵的、含噪声的黑盒函数 $f$ 的首选框架。然而，大多数 TVBO 算法所提供的渐近保证都依赖于“观测以恒定频率获取”这一假设。由于高斯过程（GP）推断的复杂度随数据集规模呈三次方增长，这一假设从长远来看是不现实的。在本文中，我们放宽了这一假设，并推导出了首个显式考虑观测采样频率变化的上界遗憾界。基于这一分析，我们提出了关于 TVBO 算法数据集规模和过期数据策略的实用建议。我们通过对遵循这些建议的算法 BOLT 在合成问题和真实世界问题上的实验，展示了其性能优于当前最先进的 TVBO 方法。

    arXiv:2501.18963v4 Announce Type: replace-cross  Abstract: Time-Varying Bayesian Optimization (TVBO) is the go-to framework for optimizing a time-varying, expensive, noisy black-box function $f$. However, most of the asymptotic guarantees offered by TVBO algorithms rely on the assumption that observations are acquired at a constant frequency. As the GP inference complexity scales with the cube of its dataset size, this assumption is unrealistic in the long run. In this paper, we relax this assumption and derive the first upper regret bound that explicitly accounts for changes in the observations sampling frequency. Based on this analysis, we formulate practical recommendations about dataset sizes and stale data policies of TVBO algorithms. We illustrate how an algorithm (BOLT) that follows these recommendations performs better than the state-of-the-art of TVBO through experiments on synthetic and real-world problems.
    
[^51]: 针对受污染无标签数据的深度正-无标签异常检测

    Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data

    [https://arxiv.org/abs/2405.18929](https://arxiv.org/abs/2405.18929)

    提出了一种将正-无标签学习与自编码器、深度支持向量数据描述等深度异常检测模型相结合的深度正-无标签异常检测框架，以应对无标签数据被异常污染的现实情况，从而提升半监督异常检测的性能。

    

    半监督异常检测旨在通过在无标签数据之外利用少量有标签异常数据来提升异常检测性能，因而受到了广泛关注。现有的半监督方法假设大部分无标签数据是正常的，并通过最小化无标签数据的异常分数、同时最大化有标签异常数据的异常分数来训练异常检测器。然而，在实际应用中，无标签数据往往被异常数据所污染。这削弱了最大化异常分数这一操作的效果，从而阻碍了检测性能的提升。为了解决这一问题，我们提出了深度正-无标签异常检测框架，该框架将正-无标签学习与自编码器、深度支持向量数据描述等深度异常检测模型相结合。我们的方法能够利用无标签数据来近似正常数据的异常分数……

    arXiv:2405.18929v3 Announce Type: replace-cross  Abstract: Semi-supervised anomaly detection, which aims to improve the anomaly detection performance by using a small amount of labeled anomaly data in addition to unlabeled data, has attracted attention. Existing semi-supervised approaches assume that most unlabeled data are normal, and train anomaly detectors by minimizing the anomaly scores for the unlabeled data while maximizing those for the labeled anomaly data. However, in practice, the unlabeled data are often contaminated with anomalies. This weakens the effect of maximizing the anomaly scores for anomalies, and prevents us from improving the detection performance. To solve this, we propose the deep positive-unlabeled anomaly detection framework, which integrates positive-unlabeled learning with deep anomaly detection models such as autoencoders and deep support vector data descriptions. Our approach enables the approximation of anomaly scores for normal data using the unlabeled
    
[^52]: 一种基于概率的人类比较对齐方法

    A Probabilistic Approach for Alignment with Human Comparisons

    [https://arxiv.org/abs/2403.10771](https://arxiv.org/abs/2403.10771)

    通过提出的两阶段“监督微调+人类比较”框架，本文研究了如何有效利用人类比较来改善AI模型的对齐，特别是在面对嘈杂数据和高维模型时。

    

    一个增长的趋势是将人类知识整合到学习框架中，利用微妙的人类反馈来完善AI模型。尽管取得了这些进展，但尚未开发出描述人类比较何时改善传统监督微调过程的特定条件的全面理论框架。为弥补这一差距，本文研究了有效利用人类比较来解决由嘈杂数据和高维模型引起的限制。我们提出了一个将机器学习与人类反馈通过概率二分方法联系起来的两阶段“监督微调+人类比较”（SFT+HC）框架。这两阶段框架首先通过SFT过程从带有噪声标记的数据中学习低维表示，然后利用人类比较来改进模型对齐。为了检验对齐阶段的效力，我们引入了一个新概念，称为“标签噪声到一致性”

    arXiv:2403.10771v1 Announce Type: new  Abstract: A growing trend involves integrating human knowledge into learning frameworks, leveraging subtle human feedback to refine AI models. Despite these advances, no comprehensive theoretical framework describing the specific conditions under which human comparisons improve the traditional supervised fine-tuning process has been developed. To bridge this gap, this paper studies the effective use of human comparisons to address limitations arising from noisy data and high-dimensional models. We propose a two-stage "Supervised Fine Tuning+Human Comparison" (SFT+HC) framework connecting machine learning with human feedback through a probabilistic bisection approach. The two-stage framework first learns low-dimensional representations from noisy-labeled data via an SFT procedure, and then uses human comparisons to improve the model alignment. To examine the efficacy of the alignment phase, we introduce a novel concept termed the "label-noise-to-co
    
[^53]: 过渡条件独立性

    Transitional Conditional Independence

    [https://arxiv.org/abs/2104.11547](https://arxiv.org/abs/2104.11547)

    本文提出“过渡条件独立性”新概念，通过马尔可夫核的单一分解将条件独立性推广到涉及参数、处理、环境等非随机变量的情形，无需在输入空间上定义分布，并证明其不对称性是本质的。

    

    统计模型中包含非随机变量：参数、处理、环境、设计点。普通的条件独立性无法表达涉及此类变量的关系。要应用条件独立性，必须先在这些变量上定义一个分布，而这会改变陈述的含义。本文引入了“过渡条件独立性”这一概念。它通过一个具有非随机输入 T 的马尔可夫核 K(W|T) 将三个变量关联起来，并由单一的分解式定义：\[ X\perp\!\!\perp_{K(W|T)} Y |Z \quad :\iff \quad \exists\, Q(X|Z):\; K(X,Y,Z|T) = Q(X|Z)\otimes K(Y,Z|T).\] 该关系断言存在一个对所有输入 t 都相同的马尔可夫核 Q(X|Z)。因此，它给出的是一种分解形式，而非条件期望之间的几乎必然恒等式，并且不需要在输入空间上定义任何分布。该关系是不对称的。我们证明了这种不对称性是本质性的：将其对称化会破坏该陈述所表达的含义。

    arXiv:2104.11547v5 Announce Type: replace-cross  Abstract: Statistical models contain variables that are not random: parameters, treatments, environments, design points. Ordinary conditional independence cannot express relations involving such variables. To apply it one must first put a distribution on them, and that changes the meaning of the statement. This paper introduces transitional conditional independence. It relates three variables on a Markov kernel $K(W|T)$ with non-stochastic input $T$, and is defined by a single factorization: \[ X\perp\!\!\perp_{K(W|T)} Y |Z \quad :\iff \quad \exists\, Q(X|Z):\; K(X,Y,Z|T) = Q(X|Z)\otimes K(Y,Z|T).\] The relation asserts a Markov kernel $Q(X|Z)$ that is the same for every input $t$. It therefore yields a factorization rather than an almost-sure identity between conditional expectations, and it needs no distribution on the input space. The relation is asymmetric. We show that the asymmetry is essential: symmetrizing it destroys the stateme
    
[^54]: CatSIM：一种分类图像相似度度量

    CatSIM: A Categorical Image Similarity Metric

    [https://arxiv.org/abs/2004.09073](https://arxiv.org/abs/2004.09073)

    CatSIM是一种基于结构相似性范式的新图像相似度度量方法，适用于二值和多值的二维及三维图像与体积，对位置的微小扰动具有鲁棒性，并能比较图像内部的任意区域。

    

    我们提出了CatSIM，这是一种用于二值和多值二维及三维图像和体积的新相似度度量方法。CatSIM采用结构相似性图像质量范式，并对位置的微小扰动具有鲁棒性，因此位于相似但不完全重叠的图像或体积区域中的结构，能够获得比简单匹配更高的评分。该度量方法还可以比较图像和体积内部的任意区域。CatSIM在人工数据集上进行了评估，并通过两个独立的图像质量评估调查与人类感知进行对比验证，同时在两个数据集上进行了应用展示。公开可用的R包catsim实现了该方法。

    arXiv:2004.09073v2 Announce Type: replace-cross  Abstract: We introduce CatSIM, a new similarity metric for binary and multinary two- and three-dimensional images and volumes. CatSIM uses a structural similarity image quality paradigm and is robust to small perturbations in location so that structures in similar, but not entirely overlapping, image or volumetric regions are rated higher than by simple matching. The metric can also compare arbitrary regions inside images and volumes. CatSIM is evaluated on artificial data sets, validated by comparing with human perception in two separate image quality assessment surveys, and illustrated on two datasets. The publicly available R package \texttt{catsim} implements the methodology.
    

