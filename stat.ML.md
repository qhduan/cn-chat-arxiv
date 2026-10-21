# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks](https://arxiv.org/abs/2610.28543) | 该论文提出二元离散扩散，将扩散过程的含噪状态表示为每个位置候选词元集合的二进制隶属向量，通过添加与删除候选实现信息的渐进丢失与恢复，从而能显式建模部分不确定性并有效解决数独、规划等逻辑与推理任务。 |
| [^2] | [Do Neural Models Prefer Their Submodels?](https://arxiv.org/abs/2610.28439) | 该论文通过引入几何框架研究神经模型的几何结构，证明除少数模型边界上的例外情况外，损失函数的局部极小值点通常不会落在由不活跃神经元或低秩结构构成的子模型中。 |
| [^3] | [Confessions of a Masked MDP](https://arxiv.org/abs/2610.28360) | 本文首次研究了动作掩码未知的线性MDP中策略优化与掩码学习相结合的问题，提出了一个基于确定性乐观优化的最小最大最优且计算高效的算法，并进一步提出了一个能主动规划探索策略、在非退化实例上表现更优的算法。 |
| [^4] | [Stabilizing Filters for Graph Signal Generation](https://arxiv.org/abs/2610.28318) | 该论文推导了连续时间图生成模型在图扰动下的Wasserstein稳定性界，并据此提出了一个原则性的稳定图滤波器设计框架，使图信号生成过程对图结构误差具有鲁棒性。 |
| [^5] | [Debiased Prior-Data Fitted Networks for Amortized Bayesian Semi-Parametric Inference](https://arxiv.org/abs/2610.28314) | 该论文为去偏先验数据拟合网络建立了理论基础，揭示了理想PFN的后验集中速率会隐式依赖干扰函数并随先验多样化而变慢，并基于有效影响函数提出标签校正与先验校正两种去偏策略以加速目标后验的集中速率。 |
| [^6] | [Geometry-Aware Block Selection for Sparse Variational Gaussian Processes](https://arxiv.org/abs/2610.28256) | 本文提出一种几何感知的块选择方法，通过在固定诱导点预算下选取能收紧变分下界的块对角缩放矩阵中的块，从而提升稀疏变分高斯过程的测试RMSE和NLL性能。 |
| [^7] | [Where Has the Foreign Signal Gone?](https://arxiv.org/abs/2610.28175) | 该研究发现Twitter情绪噪声对下一季度股票损失的预测关系在本国股票上比外国股票更加稳定，并用“生产者-知识错配假说”解释：外国股票的英语Twitter贡献者池缺乏拥有信息的母国市场投资者，导致其信息流中非基本面噪声成分更大。 |
| [^8] | [Predicting silent failures of learned inference under restricted access: what measurement physics and prior information explain](https://arxiv.org/abs/2610.28155) | 该论文提出“盲区集合”概念用于刻画学习推理中不依赖真值诊断无法察觉的静默失效，证明线性高斯模型下静默失效风险可精确分解为非检测与非覆盖，并通过线性化VAE先验构建评分，在真实X射线CT数据的前瞻性研究中成功预测了学习推理系统的实际盲性（Spearman ρ=0.79和0.84）。 |
| [^9] | [Near-Optimal Regret Bounds for UCB-Style Gaussian Process Bandits via Data Thinning](https://arxiv.org/abs/2610.28136) | 本文提出在高斯噪声假设下，通过数据拆分技术使UCB式高斯过程赌博机算法获得更紧的置信区间，从而实现近最优的简单遗憾和累积遗憾上界。 |
| [^10] | [Quantifying the Threat: A Bayesian Probabilistic Framework for Heavy Metal Health Risks among Children in the Densu Basin, Ghana](https://arxiv.org/abs/2610.28064) | 该研究提出贝叶斯概率框架评估加纳登苏流域地下水重金属对儿童的非致癌健康风险，发现约31%的采样点危害指数超过1的概率大于0.5，其中铅、砷和镉是主要风险贡献者。 |
| [^11] | [When Does Training on Synthetic Data Provably Work?](https://arxiv.org/abs/2610.28036) | 该论文从理论上刻画了合成数据训练何时可证明有效：在密度比受控时给出有限样本超额误差界，但若真实分布在合成分布支撑之外且假设类足够有表达能力，则仅凭合成风险无法保证真实世界的误差。 |
| [^12] | [Optimal aggregation in multiclass classification](https://arxiv.org/abs/2610.28025) | 本文将二分类的聚合理论推广到多分类，证明在0/1损失下最优聚合速率惊人地与类别数K无关，且经验风险最小化（ERM）聚合器达到最优速率。 |
| [^13] | [Annealed correlation and quenched noise sensitivity in biased deep neural networks](https://arxiv.org/abs/2610.27983) | 本文证明在给深度神经网络加入独立高斯偏置项后，淬火噪声敏感性在广泛的深度范围内依然成立，且只需扰动输入与原输入相等的概率趋于零这一更弱的假设。 |
| [^14] | [Practical Advantages of Online Conformal Prediction for Data Streams](https://arxiv.org/abs/2610.27740) | 该研究系统比较了四种在线保形回归方法，证明同时自适应更新回归模型和保形阈值可显著提升非平稳数据流下的覆盖率与区间宽度权衡，其中基于梯度的分位数跟踪方法每次观测仅需 O(p²) 的计算与内存成本。 |
| [^15] | [Privacy Accounting for Empirical Denoisers](https://arxiv.org/abs/2610.27710) | 本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。 |
| [^16] | [Pattern Intelligence: Unsupervised, LLM-Interpreted Discovery of Systematic Underpayment and Denial Patterns in Healthcare Revenue Cycle Management](https://arxiv.org/abs/2610.27684) | 该论文提出Pattern Intelligence系统：通过基于裁决行为（刻意排除付款方身份）的130维索赔嵌入与HDBSCAN无监督聚类，再将付款方与合同信息作为事后分割重新引入，并借助LLM进行解读，从而自动发现医疗收入周期管理中可纠正的系统性少付与拒付模式。 |
| [^17] | [BARRR: Bidimensional Augmented Reduced Rank Regression for Multi-View Multi-Cohort Data](https://arxiv.org/abs/2610.27647) | 提出二维增广降秩回归方法，可同时整合多队列与多视图数据，同时建模协变量相关与无关效应，并在两个维度上分解全局、部分共享和个体特有三个层级的效应。 |
| [^18] | [Divide-and-Conquer Neural Bayes Estimation for Spatial and Spatiotemporal Point Process Models](https://arxiv.org/abs/2610.27619) | 本文提出一种分治式神经贝叶斯估计框架，可直接在原始点模式数据上高效推断空间与时空点过程模型，克服了传统贝叶斯方法中似然难处理与计算昂贵的瓶颈。 |
| [^19] | [Ask to Learn: Query-Based Collaborative Reinforcement Learning](https://arxiv.org/abs/2610.27606) | 该论文提出了基于查询的协作强化学习新框架，使仅有离线数据的学习者无需直接与环境交互，即可通过自适应地向持有互补信息的协作者查询状态-时间对来突破离线数据覆盖范围的限制，并针对有限视界表格型问题给出了所需查询次数的实例依赖信息论下界。 |
| [^20] | [From POMDP Theory to Deep RL with Particle Filters](https://arxiv.org/abs/2610.27575) | 本文提出了一种结合序贯蒙特卡洛粒子滤波、在线潜在动力学模型学习和近似在线规划的实用端到端深度强化学习算法，并首次为部分可观测环境下的深度强化学习提供了将模型估计、粒子滤波和蒙特卡洛等各项近似误差显式分解的端到端遗憾界理论保证。 |
| [^21] | [Bayesian Neural Networks via Dirichlet Process Priors on the Data-Generating Distribution for Randomized Value Functions in Deep Reinforcement Learning](https://arxiv.org/abs/2610.27570) | 该论文提出在数据生成分布上（而非网络权重上）施加狄利克雷过程先验来构建新型贝叶斯神经网络DP-BNN，并据此开发了DP-DQN算法，通过随机化价值函数实现高效的汤普森采样，从而解决深度强化学习中认知不确定性量化的难题。 |
| [^22] | [Active Pilot Sampling for Multi-fidelity Estimation in Stochastic Optimization](https://arxiv.org/abs/2610.27563) | 本文提出一种主动先导采样框架，通过学习模型输出协方差矩阵的概率代理模型，利用协方差在设计域上的平滑变化并在优化查询间共享信息，从而降低迭代随机优化中多保真度估计的采样成本。 |
| [^23] | [What Moved the Alarm Score? Causal Attribution with Exact Computation and Scarce-Incident Estimation](https://arxiv.org/abs/2610.27534) | 该论文将警报分数的变化建模为混合结构因果模型状态上的Shapley博弈，把结构函数与外生规律作为不同的博弈参与者，从而在事件数据稀少的条件下实现精确的因果归因，并以函数/规律分辨的细粒度方式解释各类型变化如何传播并影响警报分数。 |
| [^24] | [On the Accuracy of Approximate Leave-One-Out with Blockwise Nonsmooth Regularization in High-dimensions](https://arxiv.org/abs/2610.27502) | 该论文为高维比例渐近机制下处理分块稀疏正则化的启发式近似留一法提供了理论准确性保证。 |
| [^25] | [MiCLA: Mixture-to-Constituent Latent Alignment for Polyphonic Audio Representation Learning](https://arxiv.org/abs/2610.27491) | MiCLA 提出了一种两阶段自监督预训练框架，通过合成对数梅尔混合和双向潜在对齐来显式建模混合音频与其组成成分的关系，在 AudioSet 下游任务上取得了领先的 mAP 性能。 |
| [^26] | [Beyond Scalar Leaderboards: Adaptive Sampling for Pareto Frontier Identification in Multi-Objective LLM Arenas](https://arxiv.org/abs/2610.27447) | 本文提出了固定置信度的自适应采样算法VB-EGE，能在多目标LLM竞技场评估中以尽可能少的比较高效识别帕累托前沿上的模型，突破了单一标量排行榜的局限。 |
| [^27] | [A Bayesian machine learning approach to direct and spillover effects in cluster-randomized trials with noncompliance](https://arxiv.org/abs/2610.27425) | 该论文提出一种将贝叶斯可加回归树与簇级随机效应相结合、并引入同伴治疗巧妙协变量的贝叶斯机器学习方法，在贝叶斯g-计算框架下估计存在不依从性的整群随机试验中的直接效应与溢出效应，并在模型误设时显著降低偏差。 |
| [^28] | [Communication Complexity of Universal Features](https://arxiv.org/abs/2610.27402) | 受机器学习特征概念启发，本文利用新颖的势函数论证和贪婪函数选择方法，构造了一种仅需 $O(d^*\log(1/\varepsilon))$ 比特（由对偶 VC 维决定）的通用数据表示，使得解码器能够在任意指定任务下以不超过 $\varepsilon$ 的误差完成预测。 |
| [^29] | [A plug-in estimator of the non-iterative conditional expectation g-formula using long short-term memory models](https://arxiv.org/abs/2610.27354) | 该论文首次为基于长短期记忆（LSTM）模型的g公式非迭代条件期望（NICE）代入估计量的统计推断有效性提供了原理性证明，用于在存在时变混杂和治疗-混杂反馈的情况下估计持续治疗策略的因果效应。 |
| [^30] | [VampPrior Unleashed](https://arxiv.org/abs/2610.27322) | 该论文通过解耦VampPrior的先验与后验协方差，并用狄利克雷过程取代固定的均匀混合权重，显著提升了模型证据并实现了稳健的非参数聚类。 |
| [^31] | [Solving the Ho--Nguyen Conjecture for Gaussian Mixtures: Applications to Mixtures of Experts](https://arxiv.org/abs/2610.27265) | 本文解决了Ho--Nguyen猜想，证明对所有聚类大小M≥2的高斯混合，实数非零权重系统的密度抵消阶数恰为2M，为过拟合高斯混合模型和专家混合模型的参数学习速率提供了严格的理论保证。 |
| [^32] | [Diffusion Score Estimation with Shallow Neural Networks: Regularity, Approximation, and Finite-Sample Rates](https://arxiv.org/abs/2610.27206) | 该论文为用浅层 ReLU 网络通过去噪分数匹配估计 Ornstein–Uhlenbeck 扩散分数建立了显式的有限样本理论，给出了时间一致的 Sobolev 正则性界、基于路徑范数的逼近误差控制，以及无界输入下经验 DSM 最小化器的高概率泛化风险界。 |
| [^33] | [Learned Control Variates for Rare Grid-Outage Estimation at Electric Vehicle Charging Stations](https://arxiv.org/abs/2610.27159) | 本文提出利用基于廉价可观测特征（如温度和负荷预测）训练的停电分类器构建控制变量，将任意二分类学习器的召回率和假阳性率与方差缩减显式关联，从而大幅减少电动汽车充电站罕见电网停电概率估计所需的昂贵蒙特卡洛模拟次数。 |
| [^34] | [Modular Bayesian Inference for Partially Misspecified Mixtures with Uncertain Class Labels](https://arxiv.org/abs/2610.27085) | 提出一种模块化贝叶斯框架，通过固定类别概率将分类与下游推断解耦，并隔离可信度较低总体对似然的贡献，从而防止模型部分误设对目标参数后验推断的扭曲，同时大幅简化推断过程。 |
| [^35] | [Hilbert-Scale Stability of Posterior Diffusion Priors for Ill-Posed Signal Recovery](https://arxiv.org/abs/2610.27030) | 该论文在希尔伯特尺度框架下为线性逆问题的后验扩散采样建立了稳定性理论，在正向算子平滑条件、真值源条件与分数误差弱范数控制下，证明了学习到的扩散先验在不适定信号恢复中能够给出将噪声、分数近似与反向时间效应分离的非渐近恢复保证。 |
| [^36] | [Gaussian Approximation of Besov Functions for Logistic Classification on Manifolds](https://arxiv.org/abs/2610.27022) | 该论文针对紧致光滑流形上的正则化逻辑斯蒂分类，借助常数线性依赖预测器上确界范数的Bernstein型不等式得到无需有界贝叶斯假设的快速预言机不等式，并通过多带宽高斯核积分算子的带符号组合与实插值方法，构造出同时利用内在维度与目标正则性且无饱和现象的Besov函数高斯逼近。 |
| [^37] | [Reliable Estimation of Differential Privacy under Selection Bias](https://arxiv.org/abs/2610.27021) | 本文揭示了以往差分隐私审计实现在运行成员推断攻击时引入选择偏差、从而产生统计上无效且被高估的隐私泄露下界这一问题，并提出了在选择偏差下可靠估计差分隐私的方法。 |
| [^38] | [Robust Latent Variable Graphical Models: Exact Recovery and Fundamental Limits](https://arxiv.org/abs/2610.27010) | 本文提出了一种用测度变换协方差替代样本协方差、并用合同变换核范数惩罚低秩部分的凸估计方法，用于鲁棒估计具有“稀疏减低秩”结构的高斯精度矩阵，并在受限可逆性与不可表示条件下证明了精确恢复保证，包括解的唯一性、无虚假边、边符号恢复正确以及秩和惯性的准确恢复。 |
| [^39] | [Dynamic Dual-Feedback Conformal Inference for Time Series Forecasting](https://arxiv.org/abs/2610.26987) | 该论文提出动态双反馈保形推断（DDCI），通过同时利用误覆盖指示的实际反馈和基于理想保形方法的估计反馈来自适应调整预测阈值，从而在分布变化环境下实现更可靠的时间序列预测不确定性量化。 |
| [^40] | [Quadratic Sample Complexity for Strictly Replicable Realizable Learning with Unknown Marginals](https://arxiv.org/abs/2610.26944) | 该论文首次证明，在分布及其边缘均未知的严格可复制可实现学习中，样本复杂度具有二次方下界Ω(d²/(ερ²))，并通过泊松化球采样算法为完整布尔类P_d给出了匹配的Θ(d²/(ερ²)+ε⁻¹log(1/δ))上界，完整刻画了该问题的复杂度。 |
| [^41] | [A Simple and General Approach to top-$k$ Selection](https://arxiv.org/abs/2610.26407) | 本文提出一种通过最小化期望机会成本上界导出的替代目标来自适应分配采样批次的简单通用方法，用于解决贝叶斯top-$k$选择问题，并证明其后验期望机会成本的衰减速率至少达到任意采样规则最佳速率的一半。 |
| [^42] | [Selecting the Best Few with Human Evaluators: Adaptively Combining Ratings and Pairwise Comparisons](https://arxiv.org/abs/2610.26380) | 本文提出一种自适应评估分配方法，先利用人工评分对候选项目进行粗筛，再借助两两比较精化优秀项目的排序，从而在有限评估预算下高效地从大量候选中选出最优的 $k$ 个项目。 |
| [^43] | [Truncate or Not? A Scaling Study of Looped Transformers with Truncated Backpropagation](https://arxiv.org/abs/2610.26030) | 通过近600次训练拟合的统一缩放律表明，在联合优化前向和后向循环次数时，完整BPTT训练的循环Transformer始终优于相同算力下使用截断BPTT的训练方式。 |
| [^44] | [Certifiably Correct Fr\'echet Means in Orbit Spaces](https://arxiv.org/abs/2610.24191) | 该论文针对轨道空间中计算NP难的Fréchet均值问题，借鉴群同步思想提出可求解的谱松弛与半定松弛方法，并设计了可证明全局最优性的半定证书，该证书甚至在两两对齐失效的高噪声场景下依然有效。 |
| [^45] | [Towards Hyperparameter Transfer for Differentially Private Optimization](https://arxiv.org/abs/2610.22976) | 本文提出DP-μP框架，将最大更新参数化（μP）扩展到差分隐私设置，通过信噪比条件与谱缩放公式为DP-SGD、DP-Adam和DP-Muon优化器提供了逐层学习率缩放规则，从而实现差分隐私训练中的超参数迁移。 |
| [^46] | [Complete or Sparse: A Tale of Two Identifiabilities](https://arxiv.org/abs/2610.20851) | 本文证明在生成式与判别式两种因果方向下，表示维度充足时最优编码器是潜在变量的线性函数，而结合非负激活与能量效率约束可使每个神经元恰好编码一个潜在变量，为“祖母神经元”现象提供了精确的理论解释。 |
| [^47] | [Measuring Variable Importance Using Proportional Inclusive Marginal Effects](https://arxiv.org/abs/2610.17053) | 提出了PRIME（比例包含边际效应）这一新的变量重要性度量方法，通过无模型变量选择技术仅纳入活跃变量进行计算，克服了PME仅适用于确定性模型且计算成本高的局限，能从含噪声数据中准确可靠地量化变量重要性。 |
| [^48] | [Some remarks on the connection between generative diffusion and optimal transport](https://arxiv.org/abs/2610.13614) | 本文揭示了生成扩散与最优传输之间的三个简单联系：数据加噪的联合分布在每个时刻都是熵正则化最优传输问题的解（因此反向扩散等价于逐渐降低熵正则化的退火过程），且扩散与流匹配中分离数据模式的动力学不稳定性，在最优传输中表现为生成初期一个被熵正则化平滑的指数级薄边界层。 |
| [^49] | [FERPO: Forward Entropy-Regularized Policy Optimization](https://arxiv.org/abs/2610.02198) | FERPO提出了一种无需对评论家求动作导数的在策略最大熵强化学习算法，通过熵与KL散度正则化的策略改进目标推导最优动作分布，并借助自归一化重要性采样以最小化前向KL来拟合Actor，从而避免了因价值预测不准导致的不可靠策略更新。 |
| [^50] | [Multi-Marginal Inverse Optimal Transport for Contrastive Learning Via Explicit Anchor-Positive-Negative Coupling](https://arxiv.org/abs/2609.33741) | 该论文提出Neg-MMIOT-CL方法，通过显式建模锚点-正样本-负样本三元组的多边际逆向最优传输耦合，克服了现有逆向OT方法因不利用负样本而导致的维度坍塌问题，提升了对比学习的下游性能。 |
| [^51] | [What Does the Rank Buy? A Spectral and Distributional Analysis of Low-Rank Adaptation](https://arxiv.org/abs/2609.32002) | 该论文证明在硬性逐因子范数预算下，LoRA的秩上限无法起到容量控制作用——所有复杂度与位移泛函均由秩一更新最大化、Rademacher复杂度与秩无关且分布位移上界与秩无关且紧致——并指出了秩真正发挥作用的两个位置。 |
| [^52] | [Sharp spectral norm concentration of sparse random tensors](https://arxiv.org/abs/2609.20520) | 本文证明了稀疏伯努利随机张量谱范数的锐利集中不等式，即当 $np\ge c\log n$ 时谱范数以高概率被 $C\sqrt{np}$ 控制，消除了以往结果中的对数因子，并将结果推广至非齐次采样情形及随机超图模型的无对数第二特征值界。 |
| [^53] | [A General Kernel Framework for Non-CND Distance Measures Using |D|-Dimensional Sparse Landmark Embeddings](https://arxiv.org/abs/2609.19083) | 提出稀疏地标嵌入（SLE）核框架，通过紧支撑凸块函数将输入嵌入为稀疏特征向量，使得任意距离度量（包括非条件负定度量）都能构造出可证明半正定的核矩阵，从而完全摆脱核方法（如高斯过程）对希尔伯特距离条件的依赖。 |
| [^54] | [ARISE: An adaptive residual-informed stability ensemble for feature selection in small-sample biomedical omics](https://arxiv.org/abs/2608.14866) | ARISE通过自适应加权组合多种相关性信号和残差信息冗余控制，在小样本生物医学组学数据中显著提升了特征选择的预测性和稳定性。 |
| [^55] | [$H^2(\Omega)$ A Priori Error Bounds for Boundary-Adapted PINNs Solving Elliptic Dirichlet Problems](https://arxiv.org/abs/2607.19167) | 本文通过乘以适当的到边界距离函数精确施加边界条件，构造了“边界自适应”神经网络，证明精确施加边界条件（而非 $L^2$ 边界惩罚）可以实现椭圆狄利克雷问题的 $H^2(\Omega)$ 误差控制，并推导出显式的高概率收敛速率。 |
| [^56] | [Service-Induced Congestion in Memory-Constrained LLM Serving](https://arxiv.org/abs/2606.15555) | 该论文将内存受限的LLM服务建模为离散时间动力学系统，揭示了服务过程本身会内生地累积显存压力并引发请求驱逐，并证明了在同质负载下无驱逐平衡态不稳定，系统通常会收敛到带驱逐的极限环。 |
| [^57] | [A Deep Zero-Inflated Model of North Atlantic Right Whale Presence To Support Blue Economy Management in the U.S. East Coast](https://arxiv.org/abs/2606.14403) | 本文提出了一种深度零膨胀伯努利模型，联合建模物种存在性与检测概率，有效处理被动声学监测数据中的零膨胀和复杂依赖，为濒危物种保护与蓝色经济管理提供新工具。 |
| [^58] | [The price of multi-group transductive learning](https://arxiv.org/abs/2606.04423) | 该论文揭示了在留一直推设定下，多群体学习器的错误率可能遭受随群体数量线性增长（最大达样本量平方根）的乘性惩罚，而在群体可实现的统计设定下，该惩罚仅为与群体数量无关的多对数因子。 |
| [^59] | [Learning Generative Dynamics with Soft Path-Law Constraints: A McKean-Vlasov FBSDE Approach](https://arxiv.org/abs/2605.08928) | 本文提出一个统一框架，将软分布律约束与路径依赖监督相结合，把从分布观测和离散轨迹学习随机动力学的问题表述为离散路径依赖的McKean-Vlasov控制问题，其最优性系统为McKean-Vlasov正倒向随机微分方程（FBSDE）。 |
| [^60] | [Scalable Posterior Uncertainty for Flexible Density-Based Clustering](https://arxiv.org/abs/2603.03188) | 该论文将基于密度的聚类与鞅后验分布相结合，提出了一种完全并行化、可扩展的贝叶斯不确定性量化框架，兼容归一化流等可微密度估计器，并能对任意聚类相关量进行推断，同时具备严格的频率派收敛性保证。 |
| [^61] | [A Jointly Efficient and Near-Optimal Algorithm for Heteroskedastic Generalized Linear Bandits with Adversarial Corruptions](https://arxiv.org/abs/2602.10971) | 本文提出将基于在线镜像下降的估计器与基于Hessian的置信度权重相结合的HCW-GLB-OMD算法，为带对抗性污染的异方差广义线性赌博机实现了联合高效且近最优的高概率遗憾保证。 |
| [^62] | [Robust Residual Correction via Selective Deployment for Time Series Forecasting](https://arxiv.org/abs/2512.22428) | CRC框架通过验证集校准的选择性部署规则，仅在校正能带来净误差降低时才应用混合残差校正，从而在时间序列预测中实现鲁棒的误差降低，尤其在复杂场景下避免了过拟合问题。 |
| [^63] | [FLEXI-Haz: Semiparametric Deep Survival Inference Beyond Proportional Hazards](https://arxiv.org/abs/2512.10570) | 提出 FLEXI-Haz 半参数深度神经网络模型，在突破比例风险假设限制的同时保留可解释的回归参数，并针对右删失数据实现了极小化极大最优的收敛速率及渐近有效的半参数统计推断。 |
| [^64] | [Differential Privacy as a Perk: Federated Learning over Multiple-Access Fading Channels with a Multi-Antenna Base Station](https://arxiv.org/abs/2510.23463) | 该论文研究了基于多天线基站的多址接入衰落信道上的空中联邦学习，巧妙地将信道噪声从性能损害转化为差分隐私保护的天然随机性来源，突破了现有工作在信道模型和损失函数假设上的限制，实现了隐私保护与训练性能的协同优化。 |
| [^65] | [Wasserstein Robust Classification Error: Surrogate Loss Misalignment and Distributional Attacks](https://arxiv.org/abs/2510.10000) | 本文证明了仅当损失函数是0-1损失的正倍数加标签相关偏移时，WDRO风险的排序才与鲁棒准确率的排序一致（交叉熵等常用损失存在失配问题），并提出Wasserstein分布攻击（WDA）来直接评估鲁棒准确率。 |
| [^66] | [Bayesian Nonparametric Dynamical Clustering of Time Series](https://arxiv.org/abs/2510.06919) | 提出一种基于层次狄利克雷过程与高斯过程先验的贝叶斯非参数方法，通过在未知数量的线性动态机制间切换来建模时间序列簇的演化，以原则性方式避免簇的无序增殖，并可通过变分推断实现离线与在线的高效学习，在心电图分析中验证了其有效性。 |
| [^67] | [Evaluating utility of clinical prediction models: A comparison of decision curve analysis and cost curves](https://arxiv.org/abs/2509.24608) | 本文比较了临床预测模型评估中的决策曲线分析与成本曲线两种方法，证明决策曲线与Brier曲线密切相关且两者间的参考线可相互转换，并提出上包络决策曲线可作为模型决策曲线的有用比较基准。 |
| [^68] | [Federated Learning: From Theory to Practice](https://arxiv.org/abs/2505.19183) | 本书是一本联邦学习的实践入门指南，其核心贡献是提出通过联邦学习网络的加权边编码设备间相似性来实现个性化联邦学习，使每个设备在训练自己模型的同时仍能受益于与相关设备的协作。 |
| [^69] | [Weight Space Learning for Certifiable Few-shot Transfer Learning](https://arxiv.org/abs/2502.06970) | 该论文提出了一种新型迁移学习方法，通过在上游任务上训练PEFT参数的扩散模型分布，再利用采样-评估程序为下游任务选择最优参数，从而即使在少样本情况下也能获得可认证的非空洞学习理论泛化保证。 |
| [^70] | [Help Me Help You: The Aggregate Value of Source and Target Data in Transfer Learning](https://arxiv.org/abs/2408.16189) | 该论文揭示了监督迁移学习中的两种机制二分现象：弱迁移机制下最优收敛率仅由源数据或目标数据之一决定，而强迁移机制下需联合利用源与目标数据的互补信息才能实现快得多的目标收敛率。 |
| [^71] | [On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers.](http://arxiv.org/abs/2310.14421) | 本文研究了针对AI分类器的对抗鲁棒性度量的存在性、唯一性和可扩展性，提出了可以验证的数学条件，并在合成基准测试和生物医学应用中进行了实际计算和解释。 |

# 详细

[^1]: 二元离散扩散：面向逻辑与规划任务的候选集扩散

    Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks

    [https://arxiv.org/abs/2610.28543](https://arxiv.org/abs/2610.28543)

    该论文提出二元离散扩散，将扩散过程的含噪状态表示为每个位置候选词元集合的二进制隶属向量，通过添加与删除候选实现信息的渐进丢失与恢复，从而能显式建模部分不确定性并有效解决数独、规划等逻辑与推理任务。

    

    离散扩散模型通过迭代去噪由文本等分类数据构成的含噪序列，实现任意顺序的生成，并具备自校正与填充能力，这使其成为推理、规划以及代码、数学等受约束生成任务的有前景的候选方案。然而，掩码扩散和均匀扩散会在单次腐蚀事件中抹去词元的身份信息，因此其含噪状态无法显式表示那种多个备选词元仍然合理的部分解析不确定性。我们提出二元离散扩散来填补这一空白：这是一族连续时间离散扩散过程，其含噪状态是每个位置上的一组候选词元，以二进制隶属向量表示。二元扩散通过添加和删除操作来改变候选集，使信息能够逐渐丢失并逐渐恢复，同时扩散状态本身就表示部分不确定性。在Sudoku-Extreme、Boxoban和gr

    arXiv:2610.28543v1 Announce Type: cross  Abstract: Discrete diffusion models iteratively denoise a noisy sequence of categorical data such as text, enabling any-order generation with self-correction and in-filling capabilities. This makes them promising candidates for reasoning, planning, and constrained-generation tasks like coding and math. Yet masked and uniform diffusion erase a token's identity in a single corruption event, so their noisy states cannot explicitly represent partially resolved uncertainty in which several alternatives remain plausible. We close this gap with Binary Discrete Diffusion, a family of continuous-time discrete-diffusion processes whose noisy state is a set of candidate tokens per position, represented as a binary membership vector. Binary diffusion changes candidate sets through additions and removals, allowing information to be lost and recovered gradually while the diffusion state itself represents partial uncertainty. On Sudoku-Extreme, Boxoban, and gr
    
[^2]: 神经模型是否偏好其子模型？

    Do Neural Models Prefer Their Submodels?

    [https://arxiv.org/abs/2610.28439](https://arxiv.org/abs/2610.28439)

    该论文通过引入几何框架研究神经模型的几何结构，证明除少数模型边界上的例外情况外，损失函数的局部极小值点通常不会落在由不活跃神经元或低秩结构构成的子模型中。

    

    神经模型是指由给定神经网络架构所能表达的函数集合。神经模型包含天然的子模型，例如通过将某些神经元设为不活跃状态而得到的子模型。我们研究神经模型的几何结构，并探究它们是否偏好其子模型，即损失函数的局部极小值点是否位于给定的子模型中。我们引入了一个几何框架来判断这种情况是否发生。对于浅层多层感知机，我们证明了含有不活跃隐藏神经元的子网络永远不会被偏好。对于深层多层感知机，我们给出了含有不活跃隐藏神经元的子网络不被偏好的条件。对于闪电自注意力机制，低秩子模型通常不被偏好，但我们也发现了一些例外情况，即低秩子模型因位于模型边界上而被偏好。我们的结果表明，除少数例外情况外，对子模型的偏好无法用几何结构来解释。

    arXiv:2610.28439v1 Announce Type: new  Abstract: A neural model is the set of functions that can be expressed by a given neural network architecture. Neural models contain natural submodels, such as those obtained by setting some neurons to be inactive. We study the geometry of neural models and ask whether they prefer their submodels. By this we mean whether local minimizers of the loss function lie in a given submodel. We introduce a geometric framework to determine if this occurs. For shallow multilayer perceptrons, we show that subnetworks with inactive hidden neurons are never preferred. For deep multilayer perceptrons, we give conditions under which subnetworks with inactive hidden neurons are not preferred. For lightning self-attention, lower-rank submodels are usually not preferred, but we identify examples where they are because they lie on a boundary of the model. Our results show that, aside from a few exceptions, preference for submodels cannot be explained by the geometry 
    
[^3]: 掩码MDP的自白

    Confessions of a Masked MDP

    [https://arxiv.org/abs/2610.28360](https://arxiv.org/abs/2610.28360)

    本文首次研究了动作掩码未知的线性MDP中策略优化与掩码学习相结合的问题，提出了一个基于确定性乐观优化的最小最大最优且计算高效的算法，并进一步提出了一个能主动规划探索策略、在非退化实例上表现更优的算法。

    

    在强化学习中，人们通常假设动作集是预先已知的，因此必须在计算开始之前就知道哪些状态-动作对是不被允许的。在某些应用中，确定这个动作掩码是主要的计算瓶颈。为了将这一问题隔离出来以突出其核心性质，我们研究了在线性MDP模型下如何将最优策略的搜索与掩码本身的学习相结合，在该模型中，每次尝试一个动作都会揭示决定其有效性的线性分数的精确值。在这一框架下，我们首先构建了一个基于确定性乐观优化的算法，并证明它是最小最大最优的，同时我们也表明它在计算上是高效的。然而，正如我们所展示的，该算法在非退化实例上是次优的。我们提出了另一种算法，它同样在计算上是高效的，并且能够主动规划其探索策略。该算法具有一个形式化的解释……（摘要在此处截断）

    arXiv:2610.28360v1 Announce Type: cross  Abstract: In reinforcement learning, one presumes the action set to be known ahead of time, and, thus, one must know if any state-action pairs are not permissible   ahead of computations. In some applications, determining this action mask is the main computational bottleneck. Isolating this problem to highlight its core   properties, we study how to combine the search for an optimal policy with the learning of the mask itself, under a linear MDP model in which each attempted   action reveals the exact value of a linear score determining its validity. In this framework, we first construct an algorithm based on deterministic   optimistic optimization and show that it is minimax optimal, which we show is also computationally efficient. However, it is sub-optimal on non-degenerate   instances, as we show. We propose an alternative, which is also computationally efficient, which plans its exploration strategy. This algorithm admits a   formal interpr
    
[^4]: 用于图信号生成的稳定化滤波器

    Stabilizing Filters for Graph Signal Generation

    [https://arxiv.org/abs/2610.28318](https://arxiv.org/abs/2610.28318)

    该论文推导了连续时间图生成模型在图扰动下的Wasserstein稳定性界，并据此提出了一个原则性的稳定图滤波器设计框架，使图信号生成过程对图结构误差具有鲁棒性。

    

    生成图信号是机器学习中的一项重要任务，将底层图结构融入生成模型可以实现结构感知的分布建模。然而，该图结构往往带有噪声或估计不完美，结构误差会影响生成过程以及由此产生的信号分布。虽然近期的工作已经为图神经常微分方程建立了稳定性保证，但拓扑薛定谔桥模型中所使用的图感知参考动力学的鲁棒性在很大程度上仍未被探索。在本文中，我们研究了一类连续时间生成模型的结构稳定性，其动力学将图滤波器与学习得到的图神经网络相结合。我们首先推导了一个Wasserstein稳定性界，用于量化图扰动对生成信号分布的影响。基于这一分析，我们随后提出了一个有原则的图滤波器设计框架

    arXiv:2610.28318v1 Announce Type: cross  Abstract: Generating graph signals is an important task in machine learning, and incorporating the underlying graph into a generative model enables structurally-informed distribution modeling. However, said graph is often noisy or imperfectly estimated, and structural errors can affect the generative process and the resulting signal distribution. While recent work has established stability guarantees for graph neural ordinary differential equations, the robustness of graph-aware reference dynamics, as used in topological Schr\"odinger bridge models, remains largely unexplored. Here, we investigate the structural stability of continuous-time generative models whose dynamics combine a graph filter with a learned graph neural network. We first derive a Wasserstein stability bound that quantifies the effect of graph perturbations on the generated signal distributions. Informed by this analysis, we then propose a principled framework for designing gr
    
[^5]: 用于摊销贝叶斯半参数推断的去偏先验数据拟合网络

    Debiased Prior-Data Fitted Networks for Amortized Bayesian Semi-Parametric Inference

    [https://arxiv.org/abs/2610.28314](https://arxiv.org/abs/2610.28314)

    该论文为去偏先验数据拟合网络建立了理论基础，揭示了理想PFN的后验集中速率会隐式依赖干扰函数并随先验多样化而变慢，并基于有效影响函数提出标签校正与先验校正两种去偏策略以加速目标后验的集中速率。

    

    先验数据拟合网络（PFNs）作为端到端的摊销贝叶斯推断方法，近年来广受欢迎。然而，它们针对半参数目标（如缺失数据下的均值或因果效应）的统计行为仍知之甚少。在这项工作中，我们为去偏PFN建立了理论基础，使其能在一般半参数设置下对有限维目标进行推断。首先，我们证明，尽管理想的PFN相对于目标是贝叶斯最优的，其后验集中速率可能隐式依赖于（潜在）干扰函数的集中速率，并且随着先验变得更加多样化而变慢。随后，为了加快目标后验集中速率，我们基于有效影响函数提出了两种通用的去偏策略：(i) 标签校正和 先验校正。最后，我们证明去偏PFN 具备……

    arXiv:2610.28314v1 Announce Type: cross  Abstract: Prior-data fitted networks (PFNs) have recently gained popularity as an end-to-end approach to amortized Bayesian inference. Yet, their statistical behavior for semi-parametric targets, such as means under missingness or causal effects, remains poorly understood. In this work, we develop theoretical foundations for debiased PFNs to perform inference on finite-dimensional targets in general semi-parametric settings. First, we show that, despite being Bayes-optimal with respect to the target, an idealized PFN can exhibit a posterior concentration rate that implicitly depends on the concentration rates of (latent) nuisance functions and can become slower as the prior becomes more diverse. Then, to speed up the target posterior concentration rates, we introduce two generic debiasing strategies based on efficient influence functions: (i) a label correction and (ii) a prior correction. Finally, we show that the debiased PFNs (i) & (ii) posse
    
[^6]: 面向稀疏变分高斯过程的几何感知块选择方法

    Geometry-Aware Block Selection for Sparse Variational Gaussian Processes

    [https://arxiv.org/abs/2610.28256](https://arxiv.org/abs/2610.28256)

    本文提出一种几何感知的块选择方法，通过在固定诱导点预算下选取能收紧变分下界的块对角缩放矩阵中的块，从而提升稀疏变分高斯过程的测试RMSE和NLL性能。

    

    稀疏变分高斯过程（SVGPs）是一种被广泛使用的、有原则的可扩展高斯过程建模方法。其成功源于其理论上可解释的结构：它们在支持可扩展的随机优化的同时，为完整高斯过程的边缘似然提供一个下界。近期的工作探索了对后验条件分布的修改，并证明此类扩展能够改进这一成熟的框架。特别是，引入块对角结构的缩放矩阵可以得到更紧的高斯过程边缘似然下界。我们研究了在固定诱导点预算下，选择能够收紧变分目标的块是否可以改善测试RMSE和NLL。由于缩放矩阵是通过完整残差协方差矩阵对数行列式的上界对变分下界做出贡献的，我们的工作做出了以下贡献……

    arXiv:2610.28256v1 Announce Type: cross  Abstract: Sparse variational Gaussian processes (SVGPs) are a widely used, principled approach to scalable Gaussian process modelling. Their success stems from their theoretically interpretable structure: they provide a lower bound on the marginal likelihood of the full GP while allowing scal- able stochastic optimization. Recent work has explored modifications of the posterior conditional distribution and has shown that such extensions can improve upon this well-established framework. In particular, introducing a block-diagonal structured scal- ing matrix leads to a tighter lower bound on the GP marginal likelihood. We investigate whether selecting blocks that tighten the variational objective improves test RMSE and NLL at a fixed inducing-point budget. Since the scaling matrix contributes to the variational lower bound through an upper bound on the log-determinant of the full residual covariance matrix, our work makes the following contributio
    
[^7]: 外国信号去哪儿了？

    Where Has the Foreign Signal Gone?

    [https://arxiv.org/abs/2610.28175](https://arxiv.org/abs/2610.28175)

    该研究发现Twitter情绪噪声对下一季度股票损失的预测关系在本国股票上比外国股票更加稳定，并用“生产者-知识错配假说”解释：外国股票的英语Twitter贡献者池缺乏拥有信息的母国市场投资者，导致其信息流中非基本面噪声成分更大。

    

    我们比较了Twitter信息流中的情绪噪声与本国公司及外国公司股票损失之间的关系。下一季度损失对所测噪声的斜率的季度均值“国内-国外”差异为-1.40（Newey-West t = -2.99）。对于本国股票和外国股票，该斜率分别在78%和48%的季度中为负（块自助法 p = 0.016）。因此，这一关系在时间上对本国股票更加稳定。我们运用一个噪声理性预期模型，通过贡献者池的构成来解释这些模式。与本国公司相比，外国公司股票的信息流具有更高的重复作者集中度，以及更大的模型隐含的非基本面成分。我们通过“生产者-知识错配假说”来解读这些差异：对于外国公司，英语贡献者池低估了往往拥有信息优势的母国市场投资者……（原文摘要至此截断）

    arXiv:2610.28175v1 Announce Type: new  Abstract: We compare the relation between sentiment noise in the Twitter feed and losses for stocks of domestic and foreign firms. The mean quarterly domestic-foreign difference in the slope of next-quarter losses on measured noise is -1.40 (Newey-West t = -2.99). This slope is negative for 78 and 48 percent of the quarters for domestic and foreign stocks, respectively (block-bootstrap p = 0.016). Thus, this relationship is temporally more stable for domestic stocks. We use a noisy rational expectations model to explain these patterns through the composition of the contributor pool. The feeds for stocks of foreign firms have higher repeat-author concentration and a larger model-implied nonfundamental component than those for domestic firms. We interpret these differences through the producer-knowledge mismatch hypothesis. For foreign firms, the English-language contributor pool underrepresents the home-market investors who tend to possess informat
    
[^8]: 预测受限访问条件下学习推理的静默失效：测量物理与先验信息能解释什么

    Predicting silent failures of learned inference under restricted access: what measurement physics and prior information explain

    [https://arxiv.org/abs/2610.28155](https://arxiv.org/abs/2610.28155)

    该论文提出“盲区集合”概念用于刻画学习推理中不依赖真值诊断无法察觉的静默失效，证明线性高斯模型下静默失效风险可精确分解为非检测与非覆盖，并通过线性化VAE先验构建评分，在真实X射线CT数据的前瞻性研究中成功预测了学习推理系统的实际盲性（Spearman ρ=0.79和0.84）。

    

    学习先验被用于重建测量几乎无法到达的区域，并在那里报告不确定性。它们可能会静默地失败：所报告的置信区间排除了真实值，而所有不依赖真值的诊断都保持在噪声范围内。我们将这类错误称为“盲区集合”，并探究是否能在看到目标数据之前预测它们。对于一个在先验上平均后规范正确的线性高斯模型，静默失效的风险可以精确地分解为非检测与非覆盖两个因子，而这一性质对于一般的学习系统并不成立；我们通过线性化VAE先验构建了一个近似该参考基准的评分。在一项针对真实X射线CT物体（2DeteCT）在模拟受限访问条件下的前瞻性研究中——所有预测在任何试验开始之前都以哈希形式提交——该评分对两个学习推理系统（潜在拉普拉斯方法和集合卡尔曼反演）的实际盲性进行了排序，在240次试验中Spearman相关系数ρ分别为0.79和0.84……

    arXiv:2610.28155v1 Announce Type: cross  Abstract: Learned priors are used to reconstruct regions that measurements barely reach, and to report uncertainty there. They can fail silently: the reported interval excludes the truth while every ground-truth-free diagnostic stays within its noise. We call such errors the blind set and ask whether they can be predicted before the target data are seen. For a correctly specified linear-Gaussian model averaged over the prior, the risk of a silent failure factorises exactly into non-detection and non-coverage, which does not hold for learned systems in general; we build a score that approximates this reference by linearising a VAE prior. In a prospective study on real X-ray CT objects (2DeteCT) under simulated restricted access, with predictions committed by hash before any trial, the score ranked the realised blindness of two learned inference systems (latent Laplace and ensemble Kalman inversion) with Spearman $\rho = 0.79$ and 0.84 over 240 pr
    
[^9]: 通过数据拆分技术实现UCB式高斯过程赌博机的近最优遗憾界

    Near-Optimal Regret Bounds for UCB-Style Gaussian Process Bandits via Data Thinning

    [https://arxiv.org/abs/2610.28136](https://arxiv.org/abs/2610.28136)

    本文提出在高斯噪声假设下，通过数据拆分技术使UCB式高斯过程赌博机算法获得更紧的置信区间，从而实现近最优的简单遗憾和累积遗憾上界。

    

    高斯过程置信上界算法已被广泛研究，但其已知的遗憾上界是次优的。为了获得近最优的遗憾上界，现有研究主要集中于非自适应方法，例如不确定性采样，但其实际效果往往有限。本文表明，在高斯噪声假设下，UCB式算法可以通过数据拆分——一种从单个数据集创建独立数据集的统计技术——实现近最优的遗憾上界。首先，我们证明数据拆分可以在赌博机设置中产生更紧的置信区间，并且GP-UCB的直接扩展可以实现近最优的简单遗憾上界。此外，通过利用更紧的置信区间并将GP-UCB与分阶段消除算法相结合，我们证明了近最优的累积遗憾上界。最后，我们展示了UCB式算法的有效性。

    arXiv:2610.28136v1 Announce Type: cross  Abstract: Gaussian process upper confidence bound (GP-UCB) has been widely studied, though its known regret upper bound is suboptimal. To obtain near-optimal regret upper bounds, existing studies mainly focus on non-adaptive methods, such as uncertainty sampling, whose practical effectiveness is often limited. This paper shows that, under a Gaussian noise assumption, UCB-style algorithms can achieve near-optimal regret upper bounds via data thinning, a statistical technique for creating independent datasets from a single dataset. First, we show that data thinning can yield tighter confidence intervals in the bandit setting, and that a direct extension of GP-UCB achieves a near-optimal simple regret upper bound. Furthermore, by leveraging the tighter confidence intervals and combining GP-UCB with a phased elimination algorithm, we show a near-optimal cumulative regret upper bound. Finally, we demonstrate the effectiveness of UCB-style algorithms 
    
[^10]: 量化威胁：加纳登苏流域儿童重金属健康风险的贝叶斯概率框架

    Quantifying the Threat: A Bayesian Probabilistic Framework for Heavy Metal Health Risks among Children in the Densu Basin, Ghana

    [https://arxiv.org/abs/2610.28064](https://arxiv.org/abs/2610.28064)

    该研究提出贝叶斯概率框架评估加纳登苏流域地下水重金属对儿童的非致癌健康风险，发现约31%的采样点危害指数超过1的概率大于0.5，其中铅、砷和镉是主要风险贡献者。

    

    加纳登苏河流域的地下水仍然是重要的生活用水来源，但多种金属污染威胁着儿童的健康。以往的确定性危害指数评估对不确定性、共同暴露和空间变异性的处理有限。本研究应用贝叶斯概率框架进行非致癌风险评估，使用94个地下水样本以及对铁、锰、镍、铅、镉和砷的测量。贝叶斯回归模型传播参数不确定性并考虑了污染物的共存情况。采用自适应哈密顿蒙特卡洛方法进行后验推断，同时使用反距离加权方法生成空间连续的风险曲面。铅、砷和镉主导了累积风险。后验预测结果表明，约31%的采样点其危害指数（HI）超过1的概率大于0.5。风险曲面高度异质，后验均值为0.37。

    arXiv:2610.28064v1 Announce Type: cross  Abstract: Groundwater in Ghana's Densu River Basin remains a key domestic water source, but multi-metal contamination threatens children's health. Previous deterministic hazard index assessments provide limited treatment of uncertainty, co-exposure, and spatial variability. This study applies a Bayesian probabilistic framework for non-carcinogenic risk assessment using 94 groundwater samples and measurements of Fe, Mn, Ni, Pb, Cd, and As. A Bayesian regression model propagates parameter uncertainty and accounts for contaminant co-occurrence. Adaptive Hamiltonian Monte Carlo is used for posterior inference, while inverse distance weighting generates spatially continuous risk surfaces. Pb, As, and Cd dominate cumulative risk. Posterior predictive results indicate that approximately 31% of sampling sites have a probability greater than 0.5 of exceeding a hazard index (HI) of 1. The risk surface is highly heterogeneous, with a posterior mean of 0.37
    
[^11]: 合成数据训练何时才能被证明有效？

    When Does Training on Synthetic Data Provably Work?

    [https://arxiv.org/abs/2610.28036](https://arxiv.org/abs/2610.28036)

    该论文从理论上刻画了合成数据训练何时可证明有效：在密度比受控时给出有限样本超额误差界，但若真实分布在合成分布支撑之外且假设类足够有表达能力，则仅凭合成风险无法保证真实世界的误差。

    

    高质量训练数据是机器学习中的一大瓶颈，这促使人们越来越多地使用人工生成的数据。我们研究了在共享条件标签分布的前提下，在合成数据上训练的二元分类器何时能够成功迁移到真实分布上。首先，当真实协变量分布在受控的密度比条件下关于合成分布绝对连续时，我们推导了有限样本超额误差界。其次，我们研究了部分奇异与完全奇异的情形，即真实分布在合成分布支撑之外存在概率质量。如果假设类具有足够的表达能力，那么仅凭合成风险无法一致地控制真实世界的误差：即使两个分布在 Wasserstein 距离上可以任意接近，一个分类器也可以在合成数据上几乎达到最优，同时在未被覆盖的真实区域上接近最坏情况的误差。对于假设函数与其对数…（原文摘要至此截断）

    arXiv:2610.28036v1 Announce Type: new  Abstract: High-quality training data are a major bottleneck in machine learning, motivating increasing use of artificially generated data. We study when a binary classifier trained on synthetic data transfers successfully to a real distribution, under a shared conditional label distribution. First, we derive finite-sample excess-error bounds when the real covariate distribution is absolutely continuous with respect to the synthetic distribution under a controlled density ratio. Second, we study the partially and fully singular case, where the real distribution has mass outside the synthetic one. If the hypothesis class is sufficiently expressive, synthetic risk alone cannot uniformly control real-world error: a classifier can be nearly optimal on synthetic data while approaching worst-case error on the uncovered real region, even when the distributions are arbitrarily close in Wasserstein distance. For function classes where hypotheses and the log
    
[^12]: 多分类中的最优聚合

    Optimal aggregation in multiclass classification

    [https://arxiv.org/abs/2610.28025](https://arxiv.org/abs/2610.28025)

    本文将二分类的聚合理论推广到多分类，证明在0/1损失下最优聚合速率惊人地与类别数K无关，且经验风险最小化（ERM）聚合器达到最优速率。

    

    本文研究多分类中的聚合问题，将二分类的已有结果加以推广。给定一个由M个评分函数构成的有限字典F={f_1,...,f_M}和一个训练样本，目标是构造一个数据驱动的聚合器，使其在给定损失函数下的风险尽可能接近span(F)中某个指定子集内最优聚合器（oracle）的风险。特别地，我们在0/1损失和多项逻辑斯蒂损失下研究了模型选择与凸聚合问题。在0/1损失下，我们证明最优聚合速率 remarkably（惊人地）不依赖于类别数K，相应的经验风险最小化（ERM）聚合器在模型选择中达到速率最优，在凸聚合中接近最优（仅多出一个sqrt(ln(K))因子）。而在多项逻辑斯蒂损失下，情况则更为微妙。尽管最优聚合速率仍然与……（原文摘要在此处被截断）

    arXiv:2610.28025v1 Announce Type: cross  Abstract: This paper studies the problem of aggregation in multiclass classification, generalizing existing results for binary classification. Given a finite dictionary of score functions F={f_1,...,f_M} and a training sample, the goal is to construct a data-driven aggregate whose risk is as close as possible to that of the best aggregate (oracle) in a prescribed subset of span(F) with respect to a given loss. In particular, we investigate model selection and convex aggregation under the 0/1 and multinomial logistic losses.   For the 0/1 loss, we show that the optimal aggregation rates remarkably do not depend on the number of classes K, and the corresponding empirical risk minimization (ERM) aggregates are rate-optimal for model selection and nearly-optimal for convex aggregation (up to an additional sqrt(ln(K)) factor). The situation under the multinomial logistic loss is more subtle. Although the optimal aggregation rates remain independent o
    
[^13]: 含偏置深度神经网络中的退火相关性与淬火噪声敏感性

    Annealed correlation and quenched noise sensitivity in biased deep neural networks

    [https://arxiv.org/abs/2610.27983](https://arxiv.org/abs/2610.27983)

    本文证明在给深度神经网络加入独立高斯偏置项后，淬火噪声敏感性在广泛的深度范围内依然成立，且只需扰动输入与原输入相等的概率趋于零这一更弱的假设。

    

    在Jonasson、Steif和Zetterqvist（2023）的早期工作中，已经证明了每层宽度为 $n$、权重为独立同分布标准高斯、使用符号激活且不含偏置项的全连接深度神经网络，对于输入 $\boldsymbol{\omega}$ 及其扰动 $\boldsymbol{\eta}$ 上任何满足温和正相关条件且 $\mathbb{Q}^{(n)}(\boldsymbol{\eta} \in \{\boldsymbol{\omega},-\boldsymbol{\omega}\}) \to 0$ 的联合概率分布 $\mathbb{Q}^{(n)}$ 而言，都是退火和淬火噪声敏感的。本文中，我们加入了自然数量级的独立高斯偏置项，并证明在相当广泛的深度范围内淬火噪声敏感性依然保持，实际上只需稍弱的假设 $\mathbb{Q}^{(n)}(\boldsymbol{\omega} = \boldsymbol{\eta}) \to 0$。因此，在正相关条件下，淬火噪声敏感性在最弱要求下即成立，即扰动输入渐近……

    arXiv:2610.27983v1 Announce Type: cross  Abstract: In earlier work by Jonasson, Steif and Zetterqvist (2023), fully connected deep neural networks with layers of width $n$, i.i.d.\ standard Gaussian weights, sign activations and no bias terms were shown to be annealed and quenched noise sensitive with respect to any joint probability distribution $\mathbb{Q}^{(n)}$ on the input $\boldsymbol{\omega}$ and its perturbation $\boldsymbol{\eta}$ satisfying a mild positive-correlation condition and $\mathbb{Q}^{(n)}(\boldsymbol{\eta} \in \{\boldsymbol{\omega},-\boldsymbol{\omega}\}) \to 0$. Here we add independent Gaussian biases of the natural order and show that quenched noise sensitivity is retained over a broad range of depths, indeed under the slightly weaker assumption that $\mathbb{Q}^{(n)}(\boldsymbol{\omega} = \boldsymbol{\eta}) \to 0$. Thus, subject to the positive-correlation condition, quenched noise sensitivity holds under the minimal requirement that the perturbed input is asymp
    
[^14]: 在线保形预测在数据流场景下的实用优势

    Practical Advantages of Online Conformal Prediction for Data Streams

    [https://arxiv.org/abs/2610.27740](https://arxiv.org/abs/2610.27740)

    该研究系统比较了四种在线保形回归方法，证明同时自适应更新回归模型和保形阈值可显著提升非平稳数据流下的覆盖率与区间宽度权衡，其中基于梯度的分位数跟踪方法每次观测仅需 O(p²) 的计算与内存成本。

    

    完全保形预测在可交换性假设下能够提供有限样本边际覆盖保证，但其精确计算代价可能很高，且其经典保证无法直接扩展到非平稳数据流。我们在经验覆盖率、预测区间宽度和计算成本三个维度上，将四种顺序保形回归方法与加权完全保形基准进行了比较。这些方法将普通最小二乘或遗忘因子最小二乘与均匀加权或指数加权的经验残差分位数相结合，或与基于在线梯度的分位数跟踪方法相结合。实验涵盖一个独立同分布对照组、两个合成分布漂移场景以及四个真实世界数据流。结果表明，同时自适应调整回归模型和保形阈值能够显著改善漂移条件下的覆盖率-区间宽度权衡。分位数跟踪方法每次观测仅需 $O(p^2)$ 的时间与内存开销。

    arXiv:2610.27740v1 Announce Type: cross  Abstract: Full conformal prediction offers finite-sample marginal coverage guarantees under exchangeability, but exact computation can be costly, while its classical guarantees do not directly extend to non-stationary data streams. We compare four sequential conformal regression procedures with a weighted full-conformal benchmark in terms of empirical coverage, prediction interval width, and computational cost. The procedures combine ordinary or forgetting-factor least squares with uniform or exponentially weighted empirical residual quantiles, or with online gradient-based quantile tracking. Experiments cover an independent and identically distributed control, two synthetic distribution-shift scenarios, and four real-world data streams. Adapting the regression model as well as the conformal threshold substantially improves the coverage--width tradeoff under drift. The quantile-tracking procedure requires $O(p^2)$ time and memory per observation
    
[^15]: 经验去噪器的隐私核算

    Privacy Accounting for Empirical Denoisers

    [https://arxiv.org/abs/2610.27710](https://arxiv.org/abs/2610.27710)

    本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。

    

    扩散模型已成为现代生成系统的基础，但其强大的记忆能力引发了隐私方面的担忧。虽然差分隐私（DP）训练能够提供形式化的保证，但其代价是样本质量的下降。在本工作中，我们研究了一个与之互补的问题：在采样过程本身中会产生多少隐私损失？直接分析现代神经去噪器是具有挑战性的，因为它们对单个训练样本的敏感性通常是难以计算的。为了克服这一挑战，我们构建并分析了一种经过截断的隐私可问责经验去噪器——这是一种无需训练的贝叶斯最优去噪器估计器，其每一步的敏感性都可以被显式计算，从而使每个去噪步骤都能成为一个高斯机制。这一视角使得在以下情况下进行隐私核算成为可能：当每个训练样本对每个去噪步骤都有贡献时，可在高斯DP下进行隐私核算；当每个……（摘要在此处被截断）

    arXiv:2610.27710v1 Announce Type: new  Abstract: Diffusion models have emerged as the foundation of modern generative systems, yet their high memorization capacity raises privacy concerns. While differentially private (DP) training provides formal guarantees, it comes at a cost in sample quality. In this work, we study a complementary question: how much privacy loss is incurred during the sampling process itself? Directly analyzing modern neural denoisers is challenging because their sensitivity to a single training example is generally intractable. To overcome this challenge, we construct and analyze a clipped privacy-accountable empirical denoiser, a training-free estimator of the Bayes-optimal denoiser, whose per-step sensitivity can be computed explicitly, allowing each denoising step to be a Gaussian mechanism. This perspective enables privacy accounting under Gaussian DP when each training example contributes to each denoising step, and under $f$-DP trade-off functions when each 
    
[^16]: 模式智能：医疗收入周期管理中系统性少付与拒付模式的无监督发现与LLM解读

    Pattern Intelligence: Unsupervised, LLM-Interpreted Discovery of Systematic Underpayment and Denial Patterns in Healthcare Revenue Cycle Management

    [https://arxiv.org/abs/2610.27684](https://arxiv.org/abs/2610.27684)

    该论文提出Pattern Intelligence系统：通过基于裁决行为（刻意排除付款方身份）的130维索赔嵌入与HDBSCAN无监督聚类，再将付款方与合同信息作为事后分割重新引入，并借助LLM进行解读，从而自动发现医疗收入周期管理中可纠正的系统性少付与拒付模式。

    

    医疗提供方因付款方系统性少付与索赔拒付而损失大量收入，然而识别其背后反复出现、可纠正的模式仍主要依赖人工，且无法扩展到医院级别的索赔量。我们提出Pattern Intelligence——一个已部署的系统，它通过基于索赔的裁决行为（而非索赔身份）进行聚类，以无监督方式发现此类模式。每个索赔被嵌入为一个130维向量，其中结合了逐行的预期与实际支付比率、索赔调整原因（CARC）代码、拒付与支付信号以及服务构成。我们刻意将付款方与合同身份排除在距离度量之外——我们证明它们会主导几何结构，使聚类退化为按付款方划分的平凡分组——并且仅在事后作为分割维度（按合同和就诊类型）重新引入，从而使每个被呈现出的模式保持连贯且可操作。我们使用HDBSCAN进行聚类……（原文摘要在此截断）

    arXiv:2610.27684v1 Announce Type: new  Abstract: Healthcare providers lose substantial revenue to systematic payer underpayments and claim denials, yet identifying the recurring, correctable patterns behind them remains largely manual and does not scale to hospital claim volumes. We present Pattern Intelligence, a deployed system that discovers such patterns without supervision by clustering claims on their adjudication behaviour rather than their identity. Each claim is embedded in a 130-dimensional vector combining per-line expected-versus-actual payment ratios, claim adjustment reason (CARC) codes, denial and payment signals, and service composition. Payer and contract identity are deliberately excluded from the distance metric - we show they dominate the geometry and collapse clusters into trivial per-payer groups - and are reintroduced only as a post-hoc segmentation (by contract and visit type) that keeps every surfaced pattern coherent and actionable. We cluster with HDBSCAN, re
    
[^17]: BARRR：面向多视图多队列数据的二维增广降秩回归

    BARRR: Bidimensional Augmented Reduced Rank Regression for Multi-View Multi-Cohort Data

    [https://arxiv.org/abs/2610.27647](https://arxiv.org/abs/2610.27647)

    提出二维增广降秩回归方法，可同时整合多队列与多视图数据，同时建模协变量相关与无关效应，并在两个维度上分解全局、部分共享和个体特有三个层级的效应。

    

    在生物医学研究中，跨多个队列（即样本组）和多个视图（即数据来源或模态）的二维数据整合，对于从异构数据集中获得整体性洞察至关重要。然而，现有的大多数整合方法仅关注单一维度，例如跨不同数据来源的多视图整合，或跨不同样本组的多队列整合。针对二维整合的现有方法十分稀缺，且以无监督方法为主，忽视了协变量所能提供的宝贵信息。为填补这一空白，我们提出了二维增广降秩回归方法。我们的方法能够同时建模与协变量相关和无关的效应，并在两个维度上对效应层级进行分解（全局、部分共享和个体特有）。大量模拟实验表明，我们的方法在信号恢复、结果预测以及缺失……（原文摘要在此处截断）方面展现出优势。

    arXiv:2610.27647v1 Announce Type: cross  Abstract: Bidimensional data integration across multiple cohorts (i.e., sample groups) and views (i.e., sources or modalities) is crucial in biomedical research to derive holistic insights from heterogeneous datasets. However, most existing integrative methods focus on a single dimension, such as multi-view integration across distinct sources or multi-cohort integration across distinct sample groups. Existing approaches for bidimensional integration are scarce and predominantly unsupervised, overlooking the valuable insights offered by covariates. To address this gap, we introduce the Bidimensional Augmented Reduced Rank Regression (baRRR) method. Our approach models covariate-related and unrelated effects concurrently, while also decomposing effect levels (global, partially shared, and individual) across both dimensions. Extensive simulations demonstrate the advantages of our method with respect to signal recovery, outcome prediction, and missi
    
[^18]: 面向空间与时空点过程模型的分治式神经贝叶斯估计

    Divide-and-Conquer Neural Bayes Estimation for Spatial and Spatiotemporal Point Process Models

    [https://arxiv.org/abs/2610.27619](https://arxiv.org/abs/2610.27619)

    本文提出一种分治式神经贝叶斯估计框架，可直接在原始点模式数据上高效推断空间与时空点过程模型，克服了传统贝叶斯方法中似然难处理与计算昂贵的瓶颈。

    

    空间与时空点过程模型被广泛应用于多个科学领域的事件数据分析，但由于似然函数难以处理、潜在结构维度过高以及推断依赖事件历史等原因，基于似然的贝叶斯推断在计算上往往难以承受。现有的近似方法和无似然方法虽能缓解其中部分挑战，但往往以统计效率降低、不确定性量化不完善、或因依赖预定义汇总统计量而造成信息损失为代价。近期一种有前景的方法利用了神经网络，但此前仅在纯空间或纯时间场景中被探索，且对于大范围区域或稠密点模式，其训练阶段计算代价高昂。本文提出了一种针对空间与时空点过程模型的分治式神经贝叶斯估计框架，该方法直接在原始点模式实现数据上进行操作……

    arXiv:2610.27619v1 Announce Type: cross  Abstract: Spatial and spatiotemporal point process models are widely used to analyze event data across many scientific disciplines, but likelihood-based Bayesian inference is often computationally prohibitive due to intractable likelihoods, high-dimensional latent structures, and dependence on event history. Existing approximate and likelihood-free methods mitigate some of these challenges, but often at the cost of reduced statistical efficiency, imperfect uncertainty quantification, or information loss from reliance on predefined summary statistics. A promising recent approach involves neural networks, but it has only been explored in purely spatial or temporal settings and requires a computationally expensive training stage for large domains or dense point patterns. Here, we propose a divide-and-conquer neural Bayes estimation framework for spatial and spatiotemporal point process models. Our method operates directly on raw point pattern reali
    
[^19]: 问而学：基于查询的协作强化学习

    Ask to Learn: Query-Based Collaborative Reinforcement Learning

    [https://arxiv.org/abs/2610.27606](https://arxiv.org/abs/2610.27606)

    该论文提出了基于查询的协作强化学习新框架，使仅有离线数据的学习者无需直接与环境交互，即可通过自适应地向持有互补信息的协作者查询状态-时间对来突破离线数据覆盖范围的限制，并针对有限视界表格型问题给出了所需查询次数的实例依赖信息论下界。

    

    离线强化学习从根本上受限于可用数据的覆盖范围，而通过在线探索收集额外轨迹可能代价高昂、不安全或不可行。我们研究能否在不与底层环境直接交互的情况下克服这一局限。受协作者中介模拟等实际场景的启发——其中协作者可以提供与离线数据互补的信息——我们提出了基于查询的协作强化学习，以研究如何高效地获取这些信息。学习者从一个固定的离线数据集出发，自适应地查询状态-时间对；协作者根据一个固定的、可能是次优的策略采样动作，模拟器返回相应的奖励和下一状态。对于有限视界的表格型问题，我们推导了关于额外查询次数的实例依赖的信息论下界……

    arXiv:2610.27606v1 Announce Type: cross  Abstract: Offline reinforcement learning is fundamentally limited by the coverage of the available data, while collecting additional trajectories through online exploration may be costly, unsafe, or infeasible. We study whether this limitation can be overcome without directly interacting with the underlying environment. Motivated by practical settings such as collaborator-mediated simulation, where a collaborator can provide information complementary to the offline data, we introduce query-based collaborative reinforcement learning to study how to acquire this information efficiently. The learner starts with a fixed offline dataset and adaptively queries state-time pairs; the collaborator samples an action according to a fixed, possibly suboptimal policy, and the simulator returns the corresponding reward and next state. For finite-horizon tabular problems, we derive an instance-dependent information-theoretic lower bound on the additional queri
    
[^20]: 从POMDP理论到基于粒子滤波的深度强化学习

    From POMDP Theory to Deep RL with Particle Filters

    [https://arxiv.org/abs/2610.27575](https://arxiv.org/abs/2610.27575)

    本文提出了一种结合序贯蒙特卡洛粒子滤波、在线潜在动力学模型学习和近似在线规划的实用端到端深度强化学习算法，并首次为部分可观测环境下的深度强化学习提供了将模型估计、粒子滤波和蒙特卡洛等各项近似误差显式分解的端到端遗憾界理论保证。

    

    在部分可观测环境中的基于模型的强化学习需要推断隐藏状态、学习潜在动力学，并基于由此产生的信念进行规划。实用的算法通常只解决上述流程中的某一个环节，而对其他环节依赖的近似往往缺乏深入理解；同时，现有的POMDP理论通常分析的是计算效率低下的理论算法，与现代深度强化学习实践相距甚远。为填补这一空白，我们提出了一种实用的端到端算法，该算法通过序贯蒙特卡洛（SMC）方法近似解决隐藏状态滤波问题，在线更新潜在动力学模型，并在用状态价值评论家（一种QMDP近似）对叶节点进行自举之前，执行 $d$ 步近似在线规划。我们分析了每种近似方法，并证明了端到端的遗憾界，将总误差分解为模型估计误差、粒子滤波误差、蒙特卡洛规划误差等部分。

    arXiv:2610.27575v1 Announce Type: new  Abstract: Model-based reinforcement learning in partially observable environments requires inferring hidden states, learning latent dynamics, and planning under the resulting belief. Practical algorithms often solve one aspect of the above pipeline while relying on approximations for the others that are often poorly understood, while existing POMDP theory typically analyzes computationally inefficient theoretical algorithms far from modern deep RL practice. To fill this gap, we propose a practical end-to-end algorithm that approximately solves the hidden state filtering problem via sequential Monte Carlo (SMC), updates a latent dynamics model online, and performs approximate online planning for $d$ steps before bootstrapping leaves with a state-value critic (a QMDP approximation). We analyze each approximation and prove an end-to-end regret bound decomposing the error into model estimation error, particle filter error, Monte Carlo rollout error, a
    
[^21]: 基于数据生成分布上狄利克雷过程先验的贝叶斯神经网络在深度强化学习随机化价值函数中的应用

    Bayesian Neural Networks via Dirichlet Process Priors on the Data-Generating Distribution for Randomized Value Functions in Deep Reinforcement Learning

    [https://arxiv.org/abs/2610.27570](https://arxiv.org/abs/2610.27570)

    该论文提出在数据生成分布上（而非网络权重上）施加狄利克雷过程先验来构建新型贝叶斯神经网络DP-BNN，并据此开发了DP-DQN算法，通过随机化价值函数实现高效的汤普森采样，从而解决深度强化学习中认知不确定性量化的难题。

    

    贝叶斯神经网络（BNN）为量化认知不确定性提供了一个有原则的框架。然而，标准的权重空间先验会产生复杂的高维神经网络后验，从中采样在计算上代价极高，且流行的近似推断方案往往无法有效捕捉认知不确定性——这种失效模式在强化学习（RL）中尤为突出。为解决这一挑战，我们引入了一类新的BNN，它通过在底层数据生成分布上（而非网络参数上）直接施加贝叶斯非参数狄利克雷过程（DP）先验来量化不确定性。在此基础上，我们开发了DP-DQN，这是一种强化学习算法，它利用DP-BNN来维护价值函数上的后验分布，并执行高效的汤普森采样。通过将随机化价值函数建立在贝叶斯非参……

    arXiv:2610.27570v1 Announce Type: cross  Abstract: Bayesian neural networks (BNNs) offer a principled framework for quantifying epistemic uncertainty. However, standard weight-space priors yield complex, high-dimensional neural network posteriors that are computationally prohibitive to sample from, and popular approximate inference schemes often fail to capture epistemic uncertainty effectively---a failure mode particularly acute in reinforcement learning (RL) \citep{osband2018randomized}. To resolve this challenge, we introduce a new class of BNNs that quantify uncertainty by placing Bayesian nonparametric Dirichlet Process (DP) priors directly on the underlying data-generating distribution rather than on network parameters. Building on this foundation, we develop DP-DQN, a reinforcement learning algorithm that leverages DP-BNNs to maintain posterior distributions over value functions and perform efficient Thompson sampling. By grounding randomized value functions in Bayesian nonparam
    
[^22]: 面向随机优化中多保真度估计的主动先导采样方法

    Active Pilot Sampling for Multi-fidelity Estimation in Stochastic Optimization

    [https://arxiv.org/abs/2610.27563](https://arxiv.org/abs/2610.27563)

    本文提出一种主动先导采样框架，通过学习模型输出协方差矩阵的概率代理模型，利用协方差在设计域上的平滑变化并在优化查询间共享信息，从而降低迭代随机优化中多保真度估计的采样成本。

    

    使用高保真非线性模型的随机优化在计算上往往难以承受，因为每次目标函数评估都需要在外层优化循环内部进行基于采样的不确定性量化。多保真度估计器（如近似控制变量法 ACV）可以通过利用更廉价低保真模型来降低这一成本，但它们需要模型输出的协方差信息来确定控制变量的权重和样本分配。这些协方差通常是未知的，必须通过先导采样来估计。现有的先导采样策略主要针对一次性估计，未能利用迭代随机优化的结构特点，即协方差信息在设计域上可能平滑变化，并且可以在不同的优化查询之间共享。我们提出了一种主动先导采样框架，利用概率方法学习模型输出协方差矩阵的概率代理模型……

    arXiv:2610.27563v1 Announce Type: cross  Abstract: Stochastic optimization with high-fidelity nonlinear models is often computationally prohibitive because each objective evaluation requires sampling-based uncertainty quantification inside an outer optimization loop. Multi-fidelity estimators, such as approximate control variates (ACV), can reduce this cost by exploiting cheaper low-fidelity models, but they require model-output covariance information to determine control-variate weights and sample allocations. These covariances are generally unknown and must be estimated through pilot sampling. Existing pilot-sampling strategies primarily address one-off estimation and do not exploit the structure of iterative stochastic optimization, where covariance information may vary smoothly across the design domain and can be shared across optimization queries. We propose an active pilot-sampling framework that learns a probabilistic emulator of the model-output covariance matrix using a positi
    
[^23]: 什么改变了警报分数？基于精确计算与稀疏事件估计的因果归因

    What Moved the Alarm Score? Causal Attribution with Exact Computation and Scarce-Incident Estimation

    [https://arxiv.org/abs/2610.27534](https://arxiv.org/abs/2610.27534)

    该论文将警报分数的变化建模为混合结构因果模型状态上的Shapley博弈，把结构函数与外生规律作为不同的博弈参与者，从而在事件数据稀少的条件下实现精确的因果归因，并以函数/规律分辨的细粒度方式解释各类型变化如何传播并影响警报分数。

    

    工业系统和科学仪器中的警报往往需要在事件数据稀少的情况下进行即时诊断。为了指导干预，解释必须追溯数据生成过程中的变化如何通过因果结构传播并改变警报分数，而不仅仅是标记那些可能只是下游效应的偏移。然而，对相互作用的因果组件进行精确归因似乎面临着组合爆炸的难题，而现有方法要么假设因果模型保持不变，要么需要足够的事件数据才能可靠地估计变化后的系统。我们通过将警报分数的变化表述为混合结构因果模型（SCM）状态上的Shapley博弈来应对这一场景。将结构函数和外生规律视为不同的参与者，从而产生更细粒度的、可区分函数与规律的报告，说明每类变化如何传播至警报分数。同一框架还定义了已实现事件报告和总体报告。

    arXiv:2610.27534v1 Announce Type: cross  Abstract: Alarms in industrial systems and scientific instruments often demand immediate diagnosis despite scarce incident data. To guide intervention, an explanation must trace how changes in the data-generating process propagate through the causal structure to alter the alarm score, rather than merely flagging shifts that may be downstream effects. Yet exact attribution across interacting causal components appears combinatorial, while existing methods often either assume an unchanged causal model or require enough incident data to estimate the changed system reliably. We address this setting by formulating the alarm-score change as a Shapley game over hybrid structural causal model (SCM) states. Treating structural functions and exogenous laws as distinct players yields a finer-grained, function/law-resolved report of how each type of change propagates to the alarm score. The same framework defines realized-event and population reports, with a
    
[^24]: 高维背景下分块非光滑正则化的近似留一法准确性研究

    On the Accuracy of Approximate Leave-One-Out with Blockwise Nonsmooth Regularization in High-dimensions

    [https://arxiv.org/abs/2610.27502](https://arxiv.org/abs/2610.27502)

    该论文为高维比例渐近机制下处理分块稀疏正则化的启发式近似留一法提供了理论准确性保证。

    

    近似留一法是一类能够以计算高效且统计准确的方式近似留一风险估计的方法。由于其与机器遗忘和数据估值等其他重要问题的关联，这类方法近年来受到越来越多的关注。在经验风险最小化中，ALO方法通常依赖于少数几步牛顿迭代，因此要求损失函数和正则项是光滑的。然而在高维问题中，正则项往往是非光滑的。这促使研究者针对非光滑正则化开发了启发式的ALO方法，其经验准确性通常表现良好，但理论支撑仍然有限。本文针对一类广泛的不可微罚项（该类罚项可促进分块稀疏性）研究了启发式ALO方法在比例高维渐近机制下的准确性。在假设

    arXiv:2610.27502v1 Announce Type: cross  Abstract: Approximate leave-one-out (ALO) refers to methods that provide computationally efficient and statistically accurate approximations to leave-one-out risk estimation. These methods have recently attracted increased attention because of their connections to other important problems, including machine unlearning and data valuation. For empirical risk minimization, ALO methods often rely on a few Newton steps and therefore require smooth loss functions and regularizers. In high-dimensional problems, however, regularizers are frequently non-smooth. This has motivated the development of heuristic ALO methods for non-smooth regularization, whose empirical accuracy is often strong but whose theoretical justification remains limited.   In this paper, we study the accuracy of a heuristic ALO method for a broad class of non-differentiable penalties that promote block sparsity in the proportional high-dimensional regime. Assuming uniformly bounded 
    
[^25]: MiCLA：面向多声源音频表示学习的混合-成分潜在对齐

    MiCLA: Mixture-to-Constituent Latent Alignment for Polyphonic Audio Representation Learning

    [https://arxiv.org/abs/2610.27491](https://arxiv.org/abs/2610.27491)

    MiCLA 提出了一种两阶段自监督预训练框架，通过合成对数梅尔混合和双向潜在对齐来显式建模混合音频与其组成成分的关系，在 AudioSet 下游任务上取得了领先的 mAP 性能。

    

    现实世界的音频通常包含同时发生的声音事件，然而标准的自监督方法将每个音频片段视为孤立的训练单元，没有显式建模混合音频与组成成分之间的关系。我们提出了混合-成分潜在对齐，这是一个两阶段自监督预训练框架，利用合成的对数梅尔频谱混合将混合音频的表示与其组成片段进行对齐。第一阶段通过掩码自蒸馏和对比正则化来构建单片段表示的结构。第二阶段引入双向对齐机制：对于合成混合音频，MiCLA 抑制与其中一个成分对齐的分量，并将残差与另一个成分对齐，具体做法是在 patch token 上采用投影残差化，在全局 class token 上采用样本自适应软残差化。在下游评估中，MiCLA 在微调设置下于 AS-20K 数据集上达到 43.2% 的平均精度均值，在 AS-2M 数据集上达到 50.3% 的 mAP，并且（摘要在此处截断）

    arXiv:2610.27491v1 Announce Type: cross  Abstract: Real-world audio typically contains concurrent sound events, yet standard self-supervised methods treat each clip as an isolated training unit without explicitly modeling mixture--constituent relationships. We propose Mixture-to-Constituent Latent Alignment (MiCLA), a two-stage self-supervised pre-training framework that uses synthetic log-mel mixing to align mixture representations with their constituent clips. Stage 1 structures single-clip representations via masked self-distillation and contrastive regularization. Stage 2 introduces bidirectional alignment: for a synthetic mixture, MiCLA suppresses the component aligned with one constituent and aligns the residual with the other, using projective residualization on patch tokens and sample-adaptive soft residualization on the global class token. In downstream evaluations, MiCLA achieves 43.2% mean average precision (mAP) on AS-20K and 50.3% mAP on AS-2M under fine-tuning, and achiev
    
[^26]: 超越标量排行榜：多目标LLM竞技场中帕累托前沿识别的自适应采样算法

    Beyond Scalar Leaderboards: Adaptive Sampling for Pareto Frontier Identification in Multi-Objective LLM Arenas

    [https://arxiv.org/abs/2610.27447](https://arxiv.org/abs/2610.27447)

    本文提出了固定置信度的自适应采样算法VB-EGE，能在多目标LLM竞技场评估中以尽可能少的比较高效识别帕累托前沿上的模型，突破了单一标量排行榜的局限。

    

    arXiv:2610.27447v1 公告类型：cross 摘要：大语言模型（LLM）的评估本质上是多目标的，因为不同模型可能在数学、代码、写作等性能维度上呈现不同的权衡关系。由于此类评估通常依赖成本高昂且耗时的人工反馈，因此如何用尽可能少的比较得出可靠结论至关重要。本文提出了向量-Borda经验差距消除算法（VB-EGE），这是一种固定置信度的自适应采样算法，用于在竞技场式评估中高效识别帕累托前沿上的模型，其中每次比较会针对选定的性能维度在某个提示上对一对模型进行评估。一旦积累了足够的统计证据以证实某模型的帕累托状态，VB-EGE就会将其从活跃集合中逐步消除。在多目标Bradley-Terry模型下，我们证明了VB-EGE能够以高概率返回正确的帕累托前沿

    arXiv:2610.27447v1 Announce Type: cross  Abstract: The evaluation of large language models (LLMs) is inherently multi-objective, as different models may offer different tradeoffs across performance dimensions such as math, code, and writing. Because such evaluations are often based on costly and time-consuming human feedback, it is important to draw reliable conclusions using as few comparisons as possible. In this paper, we present Vector-Borda Empirical-Gap Elimination (VB-EGE), a fixed-confidence adaptive sampling algorithm for efficiently identifying models on the Pareto frontier in arena-style evaluation, where each comparison evaluates a pair of models on a prompt with respect to a selected performance dimension. VB-EGE progressively eliminates models from an active set once sufficient statistical evidence has been accumulated to certify their Pareto status. Under a multi-objective Bradley-Terry model, we show that VB-EGE returns the correct Pareto frontier with high probability 
    
[^27]: 一种用于存在不依从性的整群随机试验中直接效应与溢出效应估计的贝叶斯机器学习方法

    A Bayesian machine learning approach to direct and spillover effects in cluster-randomized trials with noncompliance

    [https://arxiv.org/abs/2610.27425](https://arxiv.org/abs/2610.27425)

    该论文提出一种将贝叶斯可加回归树与簇级随机效应相结合、并引入同伴治疗巧妙协变量的贝叶斯机器学习方法，在贝叶斯g-计算框架下估计存在不依从性的整群随机试验中的直接效应与溢出效应，并在模型误设时显著降低偏差。

    

    我们开发了一种贝叶斯机器学习方法，用于在存在治疗不依从性的整群随机试验中估计直接效应和溢出效应。为了在考虑簇内相关性的同时灵活地建模结局和治疗接受情况，我们将贝叶斯可加回归树与簇级随机效应相结合。我们通过一个同伴治疗巧妙协变量来增强结局回归，该协变量定义为由初步接受模型预测的同伴接受治疗比例的均值，以减轻正则化所引起的效应分解失真。在贝叶斯g-计算框架内，我们估计了簇平均意向治疗效应和簇平均局部平均治疗效应，并检验了对其分解所依据的跨世界限制的敏感性。模拟研究表明，在模型误设情况下偏差有所降低，而正确设定的参数模型则表现出预期的频率学派性质。

    arXiv:2610.27425v1 Announce Type: cross  Abstract: We develop a Bayesian machine learning approach for estimating direct and spillover effects in cluster-randomized trials with treatment noncompliance. To flexibly model outcomes and treatment receipt while accounting for within-cluster dependence, we combine Bayesian additive regression trees with cluster-level random effects. We augment the outcome regression with a peer-treatment clever covariate, defined as the predicted mean proportion of treated peers from a preliminary receipt model, to mitigate regularization-induced distortion of the effect decomposition. Within a Bayesian g-computation framework, we estimate cluster-average intent-to-treat effects and cluster-average local average treatment effects and examine sensitivity to the cross-world restrictions underlying their decomposition. Simulation studies show reduced bias under model misspecification, while the correctly specified parametric models show expected frequentist pro
    
[^28]: 通用特征的通信复杂度

    Communication Complexity of Universal Features

    [https://arxiv.org/abs/2610.27402](https://arxiv.org/abs/2610.27402)

    受机器学习特征概念启发，本文利用新颖的势函数论证和贪婪函数选择方法，构造了一种仅需 $O(d^*\log(1/\varepsilon))$ 比特（由对偶 VC 维决定）的通用数据表示，使得解码器能够在任意指定任务下以不超过 $\varepsilon$ 的误差完成预测。

    

    受到机器学习中“特征”概念的启发，我们研究了表示数据的复杂度，使得该数据表示能够用于给定函数族的评估。具体而言，我们考虑一个确定性编码器，它观测 $X\sim D$，并在任务 $f\in\mathcal{F}\subseteq\{0,1\}^{\mathcal{X}}$ 被指定之前产生一个有限编码。对于每个任务，解码器必须在分布 $D$ 下以不超过 $\varepsilon$ 的误差预测 $f(X)$。我们构造了一个编码器，它输出一个长度为 $O(d^{*}\log(1/\varepsilon))$ 比特的表示，其中 $d^{*}$ 是函数类 $\mathcal{F}$ 的对偶 VC 维。我们的编码器通过贪婪方式从函数类中选取 $O((d^{*}/\varepsilon)\log(1/\varepsilon))$ 个函数，并利用它们的联合标签模式来产生编码。上界证明的关键思想是一种新颖的势函数论证，即我们将每次贪婪选择所获得的熵与对偶 VC 维所允许的模式数量进行比较。我们展示了……

    arXiv:2610.27402v1 Announce Type: cross  Abstract: Motivated by the notion of features from machine learning, we study the complexity of representing data that allows for evaluation with a given family of functions. Concretely, we consider a deterministic encoder that observes $X\sim D$ and produces a finite encoding before a task $f\in\mathcal{F}\subseteq\{0,1\}^{\mathcal{X}}$ is specified. For each task, a decoder must predict $f(X)$ with error at most $\varepsilon$ under $D$. We construct an encoder that outputs a representation with $O(d^{*}\log(1/\varepsilon))$ bits, where $d^{*}$ is the dual VC dimension of $\mathcal{F}$. Our encoder greedily selects $O((d^{*}/\varepsilon)\log(1/\varepsilon))$ functions from the class and uses their joint label pattern to produce the encoding. The key idea in our upper bound is a novel potential function argument, where we compare the entropy gained from each greedy selection with the number of patterns allowed by the dual VC dimension. We presen
    
[^29]: 基于长短期记忆模型的非迭代条件期望g公式的代入估计量

    A plug-in estimator of the non-iterative conditional expectation g-formula using long short-term memory models

    [https://arxiv.org/abs/2610.27354](https://arxiv.org/abs/2610.27354)

    该论文首次为基于长短期记忆（LSTM）模型的g公式非迭代条件期望（NICE）代入估计量的统计推断有效性提供了原理性证明，用于在存在时变混杂和治疗-混杂反馈的情况下估计持续治疗策略的因果效应。

    

    在可交换性、正性和一致性假设下，g公式的非迭代条件期望（NICE）代入估计量可用于估计持续治疗策略的因果效应，即使存在治疗与混杂因子之间的反馈关系也是如此。在实际应用中，参数模型通常被用于估计混杂因子、治疗和结局随时间的联合分布。作为参数模型的替代方案，长短期记忆（LSTM）模型专为捕获序列数据中的短期模式和长期依赖关系而设计，可能更好地适应时变混杂因子、治疗和结局之间复杂的时序依赖关系。然而，基于LSTM模型的代入估计量统计推断的有效性尚未得到研究。本文提供了原理性证明，即在一种类似于真实数据的复杂数据生成机制下……（原文摘要在此处截断）

    arXiv:2610.27354v1 Announce Type: cross  Abstract: The non-iterative conditional expectation (NICE) plug-in estimator of the g-formula can be used to estimate the causal effects of sustained treatment strategies, even in the presence of treatment-confounder feedback, under the assumptions of exchangeability, positivity, and consistency. In practical applications, parametric models are commonly used to estimate the joint distribution of confounders, treatments, and outcomes over time. As an alternative to parametric models, long short-term memory (LSTM) models, designed to capture both short-term patterns and long-range dependencies in sequential data, may better accommodate complex temporal dependencies among time-varying confounders, treatments, and outcomes. However, the validity of statistical inferences of a plug-in estimator based on LSTM models has not yet been investigated. Here, we provide proof of principle that, under a complex data generation mechanism that resembles data fr
    
[^30]: 释放VampPrior的全部潜力

    VampPrior Unleashed

    [https://arxiv.org/abs/2610.27322](https://arxiv.org/abs/2610.27322)

    该论文通过解耦VampPrior的先验与后验协方差，并用狄利克雷过程取代固定的均匀混合权重，显著提升了模型证据并实现了稳健的非参数聚类。

    

    深度潜变量模型（DLVMs）支撑着许多现代分析流程，例如在单细胞基因组学中，但其先验分布却鲜受关注。VampPrior近似最优先验（聚合后验），通常能获得较高的模型证据。在此，我们识别出VampPrior的两个关键局限：它将先验与后验的不确定性绑定在同一个摊销协方差上，并且由于其固定的均匀混合权重而导致聚类效果不佳。我们通过解耦先验与后验协方差来释放VampPrior的潜力，解决了用同一组共享参数同时产生高先验熵和低后验熵的冲突，并显著提升了模型证据。随后，我们用狄利克雷过程取代均匀混合。两者结合后，我们的方法相比VampPrior提供了更高的模型证据以及稳健的非参数聚类。将我们的[方法]与单细胞变分推断（scVI）相结合……

    arXiv:2610.27322v1 Announce Type: new  Abstract: Deep latent variable models (DLVMs) underpin many modern analysis pipelines, for example in single-cell genomics, yet their prior distributions have received markedly little attention. The VampPrior approximates the optimal prior (the aggregate posterior) and often attains high model evidence. Here, we identify two key limitations of the VampPrior: it ties prior and posterior uncertainty to a single amortized covariance and clusters poorly due to its fixed uniform mixture weights. We unleash the VampPrior by decoupling its prior and posterior covariances, resolving the conflict of generating high prior and low posterior entropy with the same shared parameters and substantially improving model evidence. We then replace the uniform mixture with a Dirichlet process. When combined, our proposals provide higher model evidence than the VampPrior and robust non-parametric clustering. Augmenting single cell Variational Inference (scVI) with our 
    
[^31]: 解决高斯混合模型中的Ho--Nguyen猜想：在专家混合模型中的应用

    Solving the Ho--Nguyen Conjecture for Gaussian Mixtures: Applications to Mixtures of Experts

    [https://arxiv.org/abs/2610.27265](https://arxiv.org/abs/2610.27265)

    本文解决了Ho--Nguyen猜想，证明对所有聚类大小M≥2的高斯混合，实数非零权重系统的密度抵消阶数恰为2M，为过拟合高斯混合模型和专家混合模型的参数学习速率提供了严格的理论保证。

    

    过拟合的高斯混合模型可能包含许多密度几乎相同的参数配置，这使得密度抵消的阶数成为其参数学习速率的核心问题。Ho和Nguyen将这些抵消编码为一个多项式系统，并猜想其精确阶数是聚类大小的函数。解决该猜想的关键是一种共同的正方差平移，它将系数方程转化为高斯矩匹配问题。对于每个聚类大小 $\Mhn\ge2$，我们证明了实数、非零权重系统的阶数恰为 $2\Mhn$：一个锐利的符号变化界在该阶数处强制刚性，而正Gauss--Hermite求积则构造出直至 $2\Mhn-1$ 阶的非平凡解。以110位有效数字精度进行的总体密度计算恢复了两到六个分量情形下预测的首项；有限多重启动搜索在两到五个分量的情形下恢复出相同的观测幂次，较小的有限规模……

    arXiv:2610.27265v1 Announce Type: new  Abstract: Over-fitted Gaussian mixtures can contain many parameter configurations with nearly the same density, making the order of density cancellation central to their parameter-learning rates. Ho and Nguyen encoded these cancellations in a polynomial system and conjectured its exact order as a function of cluster size. The key to resolving the conjecture is a common positive-variance shift that turns the coefficient equations into Gaussian moment matching. For every cluster size $\Mhn\ge2$, we prove that the real, nonzero-weight system has order $2\Mhn$: a sharp sign-change bound forces rigidity at that order, while positive Gauss--Hermite quadrature constructs a non-trivial solution through order $2\Mhn-1$. Population-density calculations at 110-digit working precision recover the predicted leading term for two to six components; finite multistart searches recover the same observed powers for two to five components, with smaller finite-scale c
    
[^32]: 基于浅层神经网络的扩散分数估计：正则性、逼近与有限样本率

    Diffusion Score Estimation with Shallow Neural Networks: Regularity, Approximation, and Finite-Sample Rates

    [https://arxiv.org/abs/2610.27206](https://arxiv.org/abs/2610.27206)

    该论文为用浅层 ReLU 网络通过去噪分数匹配估计 Ornstein–Uhlenbeck 扩散分数建立了显式的有限样本理论，给出了时间一致的 Sobolev 正则性界、基于路徑范数的逼近误差控制，以及无界输入下经验 DSM 最小化器的高概率泛化风险界。

    

    我们研究了在 $\mathbb R^d$ 中使用有限宽度浅层 ReLU 网络、通过去噪分数匹配（DSM）来估计 Ornstein–Uhlenbeck 扩散分数的样本复杂度，及其对逆过程采样的意义。在初始密度的归一化导数满足弱可微性和多项式增长条件下，我们将来自高斯转移核和初始密度的互补估计相结合，得到了在每个固定球上关于时间一致的显式 Sobolev 界。将这些界与现有的浅层网络逼近理论相结合，可对逼近误差进行显式控制，并给出充分的路徑范数预算。随后，我们利用截断技术和局部化 Rademacher 复杂度，为路径范数受限的经验 DSM 最小化器（其输入和标签均可无界）建立了高概率风险界。在额外的有界密度和次高斯假设下，适当选择网络的……（摘要在此处截断）

    arXiv:2610.27206v1 Announce Type: cross  Abstract: We study the sample complexity of estimating Ornstein--Uhlenbeck diffusion scores in $\mathbb R^d$ using denoising score matching (DSM) with finite-width shallow ReLU networks, and its implications for reverse-process sampling. Under weak differentiability and polynomial-growth conditions on normalized derivatives of the initial density, we combine complementary estimates from the Gaussian transition kernel and the initial density to obtain explicit Sobolev bounds that are uniform in time on each fixed ball. Combining these bounds with existing shallow-network approximation theory gives explicit control of the approximation error and a sufficient path-norm budget. We then use truncation and localized Rademacher complexity to establish high-probability risk bounds for path-norm-constrained empirical DSM minimizers with unbounded inputs and labels. Under additional bounded-density and sub-Gaussian assumptions, suitable choices of network
    
[^33]: 用于电动汽车充电站罕见电网停电估计的学习控制变量

    Learned Control Variates for Rare Grid-Outage Estimation at Electric Vehicle Charging Stations

    [https://arxiv.org/abs/2610.27159](https://arxiv.org/abs/2610.27159)

    本文提出利用基于廉价可观测特征（如温度和负荷预测）训练的停电分类器构建控制变量，将任意二分类学习器的召回率和假阳性率与方差缩减显式关联，从而大幅减少电动汽车充电站罕见电网停电概率估计所需的昂贵蒙特卡洛模拟次数。

    

    电动汽车（EV）的公共充电为配电馈线带来了大量且同步的负荷。在大多数电网中，馈线同时过载的情况是罕见的，但一旦发生，后果可能非常严重：过载的馈线会使变压器承受巨大压力，并可能在整个城市范围内引发连锁停电事故。因此，追求高可靠性的电网规划者需要估计极小的停电概率。为此，使用昂贵的电网模拟器进行普通蒙特卡洛估计可能需要多得令人望而却步的运行次数。为了减少运行次数，我们提出了一种由停电分类器构建的控制变量（CV），该分类器使用温度和负荷预测等廉价的、可观测的特征进行训练。我们证明，对于任何二分类学习器，方差缩减是其召回率和假阳性率的显式函数。此外，我们还证明了基于高斯代理的控制变量……

    arXiv:2610.27159v1 Announce Type: new  Abstract: Public charging of electric vehicles (EVs) adds a large and synchronized load to the distribution feeders. In most networks, it is rare to overload a feeder at the same time. When it does happen, the consequences can be severe: an overloaded feeder stresses its transformers and can trigger cascading outages across a city. Planners who design for high reliability therefore need to estimate very small outage probabilities. To that end, plain Monte Carlo estimation with an expensive grid simulator can require a prohibitively large number of runs for this task. To reduce the number of runs, we propose a control variate (CV) built from a learned classifier of the outage, trained on cheap observable features such as temperature and load forecasts. We show that for any binary learner, the variance reduction is an explicit function of its recall and false-positive rate. Furthermore, we show that control variate based on a Gaussian proxy of the l
    
[^34]: 面向类别标签不确定且部分误设混合模型的模块化贝叶斯推断

    Modular Bayesian Inference for Partially Misspecified Mixtures with Uncertain Class Labels

    [https://arxiv.org/abs/2610.27085](https://arxiv.org/abs/2610.27085)

    提出一种模块化贝叶斯框架，通过固定类别概率将分类与下游推断解耦，并隔离可信度较低总体对似然的贡献，从而防止模型部分误设对目标参数后验推断的扭曲，同时大幅简化推断过程。

    

    科学推断常常涉及来自多个潜在总体的观测数据，这些数据的类别归属是未知的，需要从可用数据中推断。虽然完整的贝叶斯混合模型为这一场景下的联合不确定性量化提供了一种有原则的方法，但当模型对于非推断目标的总体存在误设或可信度较低时，可能会扭曲对感兴趣参数的后验推断。我们提出了一个模块化贝叶斯框架来防范此类偏差。我们利用固定的类别概率（例如来自在带标签数据子集上训练的分类器）将分类与下游推断解耦；进而通过阻止可信度较低的总体的似然贡献影响推断结果，来保护对感兴趣参数的推断。这种有原则的模块化的一个关键优势在于它极大地简化了推断过程。

    arXiv:2610.27085v1 Announce Type: cross  Abstract: Scientific inference often involves observations from multiple latent populations whose class memberships are uncertain and must be inferred from available data. While full Bayesian mixture modelling provides a principled approach to joint uncertainty quantification in this setting, often the model is misspecified or less trustworthy for populations that are not the target of the inference, which can distort posterior inference for the parameters of interest. We propose a modular Bayesian framework that safeguards against such biases. We decouple classification from downstream inference using fixed class probabilities, for example from a classifier trained on a subset of labelled data. We then protect inference on the parameters of interest by preventing the likelihood contributions of less trustworthy populations from influencing them. A key benefit of this principled modularization is that it greatly simplifies inference, reducing a 
    
[^35]: 希尔伯特尺度上后验扩散先验用于不适定信号恢复的稳定性

    Hilbert-Scale Stability of Posterior Diffusion Priors for Ill-Posed Signal Recovery

    [https://arxiv.org/abs/2610.27030](https://arxiv.org/abs/2610.27030)

    该论文在希尔伯特尺度框架下为线性逆问题的后验扩散采样建立了稳定性理论，在正向算子平滑条件、真值源条件与分数误差弱范数控制下，证明了学习到的扩散先验在不适定信号恢复中能够给出将噪声、分数近似与反向时间效应分离的非渐近恢复保证。

    

    当正向算子会把未知信号平滑掉时，间接且带噪声的观测难以被反演。在这种情况下，不稳定的模态恰恰是数据揭示信息最少的模态，因此任何重构方法的成功都取决于先验信息与算子几何结构在不同正则性尺度上的相互作用方式。扩散先验在这类问题中已被广泛使用；然而，现有理论仍无法确定：在不适定正向算子作用下，学习到的分数函数何时仍能保持有效性。本文针对希尔伯特尺度上的线性逆问题研究后验扩散，并在一个显式的分数放大条件下推导出稳定性理论。在正向映射满足平滑条件、真值满足源条件、以及分数误差具有弱范数控制的假设下，我们推导出了希尔伯特空间 \(H^s\) 中的非渐近恢复界，该界将观测噪声、分数近似误差与反向时间（原文摘要在此处截断）的影响分离开来。

    arXiv:2610.27030v1 Announce Type: new  Abstract: Indirect and noisy observations are difficult to invert when the forward operator smooths out the unknown signal. In this case, the unstable modes are precisely the modes that the data reveals least, thus the success of any reconstruction method depends on how prior information interacts with the geometry of the operator across regularity scales. Diffusion priors have become ubiquitous for such problems; however, the current theory still does not determine when a learned score remains informative under ill-posed forward operators. In this paper, we study posterior diffusion for linear inverse problems on Hilbert scales and derive a stability theory under an explicit score-amplification condition. Under a smoothing condition on the forward map, a source condition on the truth, and weak-norm control of score error, we derive non-asymptotic recovery bounds in \(\Hs^s\) that separate observation noise, score approximation, and reverse-time d
    
[^36]: 流形上逻辑斯蒂分类的Besov函数高斯逼近

    Gaussian Approximation of Besov Functions for Logistic Classification on Manifolds

    [https://arxiv.org/abs/2610.27022](https://arxiv.org/abs/2610.27022)

    该论文针对紧致光滑流形上的正则化逻辑斯蒂分类，借助常数线性依赖预测器上确界范数的Bernstein型不等式得到无需有界贝叶斯假设的快速预言机不等式，并通过多带宽高斯核积分算子的带符号组合与实插值方法，构造出同时利用内在维度与目标正则性且无饱和现象的Besov函数高斯逼近。

    

    我们研究了连通、紧致、光滑的嵌入流形 $X\subset\mathbb R^d$（无边界，内在维度为 $\varrho$）上的高斯核逼近与正则化逻辑斯蒂分类问题。两个主要挑战分别是：在不假设贝叶斯函数有界的前提下推导快速的预言机不等式，以及构造既能利用内在维度又能利用目标函数正则性、且不出现饱和现象的高斯逼近器。对于第一个挑战，我们通过一个Bernstein型不等式加以解决，该不等式中的常数与预测器的上确界范数呈线性依赖关系。结合局部化的复杂度估计，由此得到了可分赋范函数空间上的预言机不等式。针对第二个挑战，我们建立了一个一致的Sobolev算子展开，并利用不同带宽下高斯核积分算子的带符号组合来抵消其低阶偏差项。随后通过实插值方法，进而得到Besov函数的 $L_2(X)$ 逼近（摘要原文在此处截断）。

    arXiv:2610.27022v1 Announce Type: cross  Abstract: We study Gaussian kernel approximation and regularized logistic classification on a connected compact smooth embedded manifold $X\subset\mathbb R^d$ without boundary and of intrinsic dimension $\varrho$. Two main challenges are to derive a fast oracle inequality without assuming a bounded Bayes function and to construct Gaussian approximants that exploit both the intrinsic dimension and the target's regularity without saturation. We address the first challenge through a Bernstein-type inequality whose constant depends linearly on the supremum norm of the predictor. Combined with localized complexity estimates, this yields an oracle inequality over separable normed function spaces. To address the second challenge, we establish a uniform Sobolev operator expansion and cancel its lower-order bias terms using a signed combination of Gaussian kernel integral operators at different bandwidths. Real interpolation then gives an $L_2(X)$ approx
    
[^37]: 选择偏差下差分隐私的可靠估计

    Reliable Estimation of Differential Privacy under Selection Bias

    [https://arxiv.org/abs/2610.27021](https://arxiv.org/abs/2610.27021)

    本文揭示了以往差分隐私审计实现在运行成员推断攻击时引入选择偏差、从而产生统计上无效且被高估的隐私泄露下界这一问题，并提出了在选择偏差下可靠估计差分隐私的方法。

    

    arXiv:2610.27021v1 公告类型：交叉。摘要：统计隐私审计对于验证差分隐私（DP）机制的理论隐私保证至关重要。审计人员通常使用成员推断攻击（MIA）在指定置信水平下估计DP机制真实隐私泄露的下界，该下界不应超过DP所保证的理论隐私上界。然而，我们发现以往工作中的许多审计实现在运行MIA时引入了选择偏差，导致统计上无效且被高估的下界。由于大多数审计本身并不紧致，这种偏差至今未被察觉：估计出的下界仍远低于理论上界。尽管如此，准确评估这一下界至关重要，因为存在缺陷的统计评估可能导致对正确机制的误报，或对隐私保证紧致性的错误评估。本工作采用了来自统……（摘要不完整）

    arXiv:2610.27021v1 Announce Type: cross  Abstract: Statistical privacy audits are essential for verifying the theoretical privacy guarantees of differentially private (DP) mechanisms. Auditors typically use Membership Inference Attacks (MIAs) to estimate a lower bound on the true privacy leakage of DP mechanisms at a specified confidence level, which should not exceed the theoretical privacy upper bound guaranteed by DP. However, we found that many auditing implementations in prior work introduce a selection bias when running MIAs, which leads to statistically invalid and overestimated lower bounds. Because most audits are not tight, this bias has thus far gone unnoticed: the estimated lower bounds remained well below the theoretical upper bounds. Nevertheless, accurately evaluating this lower bound is critical, as a flawed statistical assessment could lead to false alarms on correct mechanisms or an incorrect evaluation of the guarantee's tightness. This work employs tools from the st
    
[^38]: 鲁棒潜变量图模型：精确恢复与基本极限

    Robust Latent Variable Graphical Models: Exact Recovery and Fundamental Limits

    [https://arxiv.org/abs/2610.27010](https://arxiv.org/abs/2610.27010)

    本文提出了一种用测度变换协方差替代样本协方差、并用合同变换核范数惩罚低秩部分的凸估计方法，用于鲁棒估计具有“稀疏减低秩”结构的高斯精度矩阵，并在受限可逆性与不可表示条件下证明了精确恢复保证，包括解的唯一性、无虚假边、边符号恢复正确以及秩和惯性的准确恢复。

    

    我们研究高斯精度矩阵的鲁棒估计问题，该矩阵可分解为一个稀疏矩阵减去一个对称的、可能不定的低秩矩阵，这种结构出现在潜变量边缘化与线性测量条件化相结合的情形中。所提出的凸估计器用中心化的高斯测度变换（MT）协方差替代样本协方差，该协方差在标称模型下能够保留稀疏分量的带符号图结构，并通过一个合同变换的核范数惩罚低秩分量，这是一种编码了结构侧信息的原子范数。我们的主要结果是一个原始-对偶见证定理：在Fisher信息满足受限可逆性与不可表示条件的情况下，凸优化的原始解是唯一的，不产生虚假边，能够恢复所有强度足够高的边的符号，并且恰好具有真实的秩与惯性。这些条件可归结为图形化拉（截断）……

    arXiv:2610.27010v1 Announce Type: cross  Abstract: We study robust estimation of a Gaussian precision matrix that decomposes as a sparse matrix minus a symmetric, possibly indefinite, low-rank matrix, a structure that arises when latent-variable marginalization is combined with conditioning on linear measurements. The proposed convex estimator replaces the sample covariance by the centered Gaussian measure-transformed (MT) covariance, which preserves the signed graph of the sparse component under the nominal model, and penalizes the low-rank component by the nuclear norm of a congruence transform, an atomic norm that encodes structural side information. Our main result is a primal--dual witness theorem: under restricted invertibility and irrepresentability conditions on the Fisher information, the raw convex optimizer is unique, has no false edges, recovers the signs of all sufficiently strong edges, and has exactly the true rank and inertia. These conditions reduce to the graphical-la
    
[^39]: 用于时间序列预测的动态双反馈保形推断

    Dynamic Dual-Feedback Conformal Inference for Time Series Forecasting

    [https://arxiv.org/abs/2610.26987](https://arxiv.org/abs/2610.26987)

    该论文提出动态双反馈保形推断（DDCI），通过同时利用误覆盖指示的实际反馈和基于理想保形方法的估计反馈来自适应调整预测阈值，从而在分布变化环境下实现更可靠的时间序列预测不确定性量化。

    

    arXiv:2610.26987v1 公告类型：cross 摘要：时间序列预测中的不确定性量化要求尤为苛刻，因为序列数据具有时间依赖性，且容易发生分布变化。保形推断通过构造预测集合，已成为一种强大的不确定性量化方法。近期的研究进展引入了在线保形方法，通过反馈机制自适应地调整预测阈值。然而，现有的反馈机制通常仅依赖误覆盖指示器（实际反馈），即在每个时间步真实标签是否落在区间内，而忽视了由理想保形方法推导出的经验预测阈值（估计反馈）。在本文中，我们提出了动态双反馈保形推断（DDCI），它融合了由实际反馈和估计反馈组成的双反馈机制。前者驱动主要的调整……

    arXiv:2610.26987v1 Announce Type: cross  Abstract: Quantifying uncertainty in time series forecasting is particularly demanding because sequential data exhibit temporal dependence and are prone to distributional changes. Conformal inference has emerged as a powerful uncertainty quantification approach through the construction of prediction sets. Recent advances have introduced online conformal methods that adaptively adjust prediction thresholds through feedback mechanisms. However, the existing feedback mechanism typically relies solely on miscoverage indicators (actual feedback), whether the true label falls within the interval at each time step, while overlooking the empirical prediction threshold (estimated feedback) that is derived from the oracle conformal method. In this paper, we propose Dynamic Dual-feedback Conformal Inference (DDCI), which incorporates a dual-feedback mechanism consisting of actual feedback and estimated feedback. The former drives the primary adjustment of 
    
[^40]: 未知边缘分布下严格可复制可实现学习的二次样本复杂度

    Quadratic Sample Complexity for Strictly Replicable Realizable Learning with Unknown Marginals

    [https://arxiv.org/abs/2610.26944](https://arxiv.org/abs/2610.26944)

    该论文首次证明，在分布及其边缘均未知的严格可复制可实现学习中，样本复杂度具有二次方下界Ω(d²/(ερ²))，并通过泊松化球采样算法为完整布尔类P_d给出了匹配的Θ(d²/(ερ²)+ε⁻¹log(1/δ))上界，完整刻画了该问题的复杂度。

    

    在量词∃A∀D的框架下，即可实现分布及其边缘分布均未知的情况下，我们确定了严格可复制性（即两个独立样本和一个共享随机种子必须产生完全相同的函数）的有限类样本复杂度指数。当ε=δ=ρ=10⁻⁴时，在|H|≤M上的最坏情况复杂度为Θ((log M)²)，且不含迭代对数项。对于d≥1、0<ε≤1/16以及0<δ,ρ≤10⁻³，任何在整个显式锚定可实现类上有效的随机学习器（即使允许非proper学习）都需要Ω(d²/(ερ²))个样本。对于完整布尔类P_d（d≥2），在相同的参数范围内，我们借助一个proper的泊松化球采样算法证明了匹配的刻画结果Θ(d²/(ερ²)+ε⁻¹log(1/δ))。此外，我们还为每个有限可测布尔类给出了显式的proper Gibbs算法上界。

    arXiv:2610.26944v1 Announce Type: cross  Abstract: Under the quantifier $\exists A\,\forall D$, where the realizable distribution and its marginal are unknown, we determine the finite-class sample-complexity exponent for strict replicability: two independent samples and one shared seed must produce identical functions. For $\varepsilon=\delta=\rho=10^{-4}$, the worst case over $|H|\le M$ is $\Theta((\log M)^2)$, without iterated logarithms. For $d\ge1$, $0<\varepsilon\le1/16$, and $0<\delta,\rho\le10^{-3}$, every randomized, possibly improper learner valid throughout an explicit anchored realizable family requires $\Omega(d^2/(\varepsilon\rho^2))$ samples. For the full Boolean class $P_d$, $d\ge2$, in the same parameter range, we prove the matching characterization $\Theta(d^2/(\varepsilon\rho^2)+\varepsilon^{-1}\log(1/\delta))$ using a proper Poissonized ball-sampling algorithm. We also give an explicit proper Gibbs upper bound for every finite measurable Boolean class. Our lower boun
    
[^41]: 一种简单而通用的top-$k$选择方法

    A Simple and General Approach to top-$k$ Selection

    [https://arxiv.org/abs/2610.26407](https://arxiv.org/abs/2610.26407)

    本文提出一种通过最小化期望机会成本上界导出的替代目标来自适应分配采样批次的简单通用方法，用于解决贝叶斯top-$k$选择问题，并证明其后验期望机会成本的衰减速率至少达到任意采样规则最佳速率的一半。

    

    本文研究了贝叶斯top-$k$识别问题，其中决策者的任务是从一组质量未知的物品中选出$k$个物品。为了评估这些物品，决策者需要自适应地分配固定的采样预算，每次对某个物品的采样都会产生对其质量的一个含噪观测，目标是最小化所选集合的期望机会成本。我们提出了一种方法，通过最小化由期望机会成本上界导出的替代目标来分配每一批采样。该方法具有通用性：采样可以按批次进行分配，选择数量$k$可以是任意的，并且适用于多种采样分布。该方法同时也简单直观。我们证明了该方法的后验期望机会成本以指数速率衰减，其速率至少达到任何采样规则所能达到的最佳速率的一半。在广泛的仿真设置中……

    arXiv:2610.26407v2 Announce Type: replace  Abstract: This paper considers the Bayesian top-$k$ identification problem, wherein a decision maker is tasked with selecting $k$ items from a pool of items of unknown quality. To assess the items, the decision maker adaptively allocates a fixed sampling budget, where each sample of an item yields a noisy observation of its quality, with the objective of minimizing the expected opportunity cost of the selected set. We propose an approach that allocates each batch of samples by minimizing a surrogate objective derived from an upper bound on the expected opportunity cost. The approach is general: samples can be allocated in batches, the selection size $k$ is arbitrary, and it applies to many sampling distributions. It is also simple and intuitive. We prove that the method's posterior expected opportunity cost decays at an exponential rate of at least half the best rate achievable by any sampling rule. In simulations across a wide range of settin
    
[^42]: 借助人工评估者选出最优少数：自适应地结合评分与两两比较

    Selecting the Best Few with Human Evaluators: Adaptively Combining Ratings and Pairwise Comparisons

    [https://arxiv.org/abs/2610.26380](https://arxiv.org/abs/2610.26380)

    本文提出一种自适应评估分配方法，先利用人工评分对候选项目进行粗筛，再借助两两比较精化优秀项目的排序，从而在有限评估预算下高效地从大量候选中选出最优的 $k$ 个项目。

    

    本文研究这样一个决策者问题：需要从大量候选者（如求职者、大语言模型、奖项提名人或新产品创意）中识别出最优的 $k$ 个项目。决策者通过征集人工评估来了解各项目的质量。由于每次评估都可能成本高昂，决策者希望通过自适应地分配评估来高效利用有限的评估预算。一个关键考虑因素是所征集的评估形式，这种形式应当是评估者乐于提供的。两种流行的形式是评分和两两比较，我们对两者都加以考虑，因为在预算有限的情况下，二者的结合可能尤其有用。评分可以在早期使用，以大致了解哪些项目好、哪些项目不好。随着这一图景逐渐成形，两两比较可以细化优秀项目之间的排序，从而聚焦于最终的 top-$k$ 选择。为了解决这一分配问题，

    arXiv:2610.26380v2 Announce Type: replace-cross  Abstract: This paper considers a decision-maker tasked with identifying the $k$ best items from a large pool of candidates, such as job applicants, LLMs, award nominees, or new product ideas. The decision-maker learns about the items' quality by soliciting human evaluations. Each evaluation can be costly, so the decision-maker aims to use the limited evaluation budget efficiently by allocating evaluations adaptively. A key consideration is the form of evaluation to solicit, which should be something evaluators are comfortable providing. Two popular forms are ratings and pairwise comparisons, and we allow for both, as they may be especially useful in combination given the limited budget. Ratings can be used early on to get a coarse picture of which items are good and which are not. As that picture forms, comparisons can refine the ordering of the good items, narrowing in on the ultimate top-$k$ selection. To solve the allocation problem, 
    
[^43]: 截断与否？截断反向传播下循环Transformer的缩放律研究

    Truncate or Not? A Scaling Study of Looped Transformers with Truncated Backpropagation

    [https://arxiv.org/abs/2610.26030](https://arxiv.org/abs/2610.26030)

    通过近600次训练拟合的统一缩放律表明，在联合优化前向和后向循环次数时，完整BPTT训练的循环Transformer始终优于相同算力下使用截断BPTT的训练方式。

    

    循环Transformer通过复用共享的层块来增加计算深度，而无需增加可训练参数的数量。它们通过时间反向传播（BPTT）进行训练。为了节省训练算力，BPTT通常被截断为只传播最后几次循环，但截断会产生有偏梯度，这可能削弱循环带来的好处。在这项工作中，我们以系统化的方式研究了截断在循环Transformer训练中的价值：我们联合变化模型规模、训练token数量以及前向和后向循环次数，进行了近600次训练运行，并拟合出一个统一的缩放律，通过有效参数刻画循环与截断的影响。尽管截断能够节省训练算力，但在联合优化前向和后向循环次数时，我们拟合的缩放律更倾向于完整的BPTT。在算力匹配的实验中，采用更少循环次数的完整BPTT始终优于相同算力下的截断设置。

    arXiv:2610.26030v2 Announce Type: replace  Abstract: Looped transformers reuse a shared block of layers to increase computational depth without increasing the number of trainable parameters. They are trained with backpropagation through time (BPTT). To save training compute, BPTT is often truncated to the last few loops, but truncation yields biased gradients that may weaken the benefit of looping. In this work, we study the value of truncation in looped transformer training in a systematic way: we jointly vary model size, training tokens, and forward and backward loop counts across nearly 600 training runs and fit a unified scaling law that captures the effects of looping and truncation through effective parameters. Despite the training-compute savings of truncation, our fitted law favors full BPTT when forward and backward loop counts are jointly optimized. In matched-compute runs, full BPTT through fewer loops consistently outperforms truncated settings of the same compute. The law 
    
[^44]: 轨道空间中可证明正确的Fréchet均值

    Certifiably Correct Fr\'echet Means in Orbit Spaces

    [https://arxiv.org/abs/2610.24191](https://arxiv.org/abs/2610.24191)

    该论文针对轨道空间中计算NP难的Fréchet均值问题，借鉴群同步思想提出可求解的谱松弛与半定松弛方法，并设计了可证明全局最优性的半定证书，该证书甚至在两两对齐失效的高噪声场景下依然有效。

    

    我们研究了有限正交群作用下轨道空间中的Fréchet均值。由于计算这些均值在一般情况下是NP难的，我们受群同步问题的启发，开发了易于求解的谱松弛和半定规划松弛方法。两两对齐可以验证全局最优性，我们还进一步开发了一种半定证书，能够将最优性验证扩展到该条件之外的情形。对于模符号的高斯数据，我们证明了在两两对齐方法失效的高噪声环境下，我们的证书仍然能够成功验证最优性。数值实验在多种设置下评估了这些方法，包括总结选区划分方案集合以及对时间序列进行聚类。

    arXiv:2610.24191v2 Announce Type: replace-cross  Abstract: We study Fr\'echet means in orbit spaces of finite orthogonal group actions. Since computing these means is NP-hard in general, we take inspiration from group synchronization to develop tractable spectral and semidefinite relaxations. Pairwise alignment certifies global optimality and we develop a semidefinite certificate that can extend beyond this condition. For Gaussian data modulo sign, we prove that our certificate succeeds in a high-noise regime where alignment fails. Numerical experiments evaluate these methods in various settings, including summarizing ensembles of districting plans and clustering time series
    
[^45]: 迈向差分隐私优化的超参数迁移

    Towards Hyperparameter Transfer for Differentially Private Optimization

    [https://arxiv.org/abs/2610.22976](https://arxiv.org/abs/2610.22976)

    本文提出DP-μP框架，将最大更新参数化（μP）扩展到差分隐私设置，通过信噪比条件与谱缩放公式为DP-SGD、DP-Adam和DP-Muon优化器提供了逐层学习率缩放规则，从而实现差分隐私训练中的超参数迁移。

    

    缩放定律与超参数调优仍然是训练大规模模型的核心挑战，尤其在差分隐私（DP）条件下，由于噪声注入和逐样本梯度裁剪，优化问题变得更加复杂。虽然最大更新参数化（μP）能够在标准训练中实现零样本超参数迁移，但我们证明了在DP约束下它会失效，原因在于高维噪声与低秩梯度具有截然不同的谱特性。在本工作中，我们提出了DP-μP，这是一个具有理论依据的框架，将μP扩展到了隐私保护场景。通过分析梯度裁剪、噪声注入与特征学习之间的相互作用，我们推导出了一个控制逐层学习率缩放的信噪比（SNR）条件。结合我们提出的谱缩放公式，我们为DP-SGD、DP-Adam和DP-Muon优化器给出了具体的缩放规则。

    arXiv:2610.22976v1 Announce Type: new  Abstract: Scaling laws and hyperparameter tuning remain central challenges in training large-scale models, particularly under differential privacy (DP), where optimization is further complicated by noise injection and per-sample gradient clipping. While Maximal Update Parametrization ($\mu$P) allows for zero-shot hyperparameter transfer in standard training, we show that it fails under DP constraints due to the distinct spectral properties of high-dimensional noise versus low-rank gradients. In this work, we introduce DP-$\mu$P, a theoretically grounded framework that extends $\mu$P to the privacy-preserving setting. By analyzing the interaction between gradient clipping, noise injection, and feature learning, we derive a Signal-to-Noise Ratio (SNR) condition that governs layer-wise learning rate scaling. Combining this with our proposed spectral scaling formulation, we provide specific scaling rules for DP-SGD, DP-Adam, and DP-Muon optimizers. Em
    
[^46]: 完整还是稀疏：两种可辨识性的故事

    Complete or Sparse: A Tale of Two Identifiabilities

    [https://arxiv.org/abs/2610.20851](https://arxiv.org/abs/2610.20851)

    本文证明在生成式与判别式两种因果方向下，表示维度充足时最优编码器是潜在变量的线性函数，而结合非负激活与能量效率约束可使每个神经元恰好编码一个潜在变量，为“祖母神经元”现象提供了精确的理论解释。

    

    理解神经网络何时学习到可解释的表示，是机制可解释性研究的核心问题。我们通过可辨识性来研究这一问题：监督学习何时能够恢复数据背后真实的潜在变量？我们引入了两种以因果方向相区分的数据生成过程：一种是由标签导致潜在变量（生成式），另一种是由潜在变量导致标签（判别式）。当表示的维度不少于潜在空间的维度时，我们证明最优编码器是潜在变量的线性函数；相反，当维度不足时，可辨识性通常会失效，编码器变为非线性。随后我们进一步证明，额外的约束能够恢复单个潜在变量，而不仅仅是它们的线性混合。在生成式设定中，非负激活与能量效率迫使每个神经元恰好编码一个潜在变量，从而为“选择性祖母神经元”现象提供了精确的理论解释。

    arXiv:2610.20851v1 Announce Type: cross  Abstract: Understanding when neural networks learn interpretable representations is central to mechanistic interpretability. We study this through identifiability: when can supervised learning recover the true latent variables underlying the data? We introduce two data generating processes distinguished by their causal direction where either labels cause latents (generative) or latents cause labels (discriminative). When the representation has at least as many dimensions as the latent space, we prove that the optimal encoders are linear in the latents. Conversely, when dimensions are insufficient, identifiability generically fails and encoders become nonlinear. We then show that additional constraints yield recovery of individual latent variables, not just linear mixtures. In the generative setting, non-negative activations and energy efficiency force each neuron to encode exactly one latent, providing a precise account of selective grandmother 
    
[^47]: 基于比例包含边际效应的变量重要性度量

    Measuring Variable Importance Using Proportional Inclusive Marginal Effects

    [https://arxiv.org/abs/2610.17053](https://arxiv.org/abs/2610.17053)

    提出了PRIME（比例包含边际效应）这一新的变量重要性度量方法，通过无模型变量选择技术仅纳入活跃变量进行计算，克服了PME仅适用于确定性模型且计算成本高的局限，能从含噪声数据中准确可靠地量化变量重要性。

    

    基于Shapley的度量方法在统计学和机器学习中被广泛用于衡量变量重要性。然而，这些度量方法存在一个严重问题——当一个非活跃变量与活跃变量相关时，该非活跃变量可能获得正的重要性度量值。Herin等人最近提出了一种称为比例边际效应（PME）的度量方法来缓解这种虚假效应。然而，PME是为确定性模型开发的，且计算成本很高。本文提出了一种称为比例包含边际效应（PRIME）的新度量方法，克服了PME的这两个局限性。PRIME的关键思想是使用一种新颖的无模型变量选择技术来识别活跃变量，并仅将这些变量纳入重要性度量的计算中。对测试函数和真实数据集的分析表明，PRIME能够从含噪声的输入-输出数据中准确、可靠地量化变量重要性。

    arXiv:2610.17053v1 Announce Type: new  Abstract: Shapley-based measures are widely used for measuring variable importance in statistics and machine learning. However, these measures have a serious issue--a nonactive variable can get a positive importance measure when it is correlated with an active variable. Herin et al. recently proposed a measure called proportional marginal effects (PME) that mitigates this spurious effect. However, PME is developed for deterministic models and has high computational cost. In this article, we propose a new measure called proportional inclusive marginal effects (PRIME) that overcomes these two limitations of PME. The key idea in PRIME is to use a novel model-free variable selection technique to identify the active variables and include only them in the computation of the importance measure. Analysis of test functions and real datasets show that PRIME can accurately and reliably quantify variable importance from noisy input-output data.
    
[^48]: 关于生成扩散与最优传输之间联系的一些注记

    Some remarks on the connection between generative diffusion and optimal transport

    [https://arxiv.org/abs/2610.13614](https://arxiv.org/abs/2610.13614)

    本文揭示了生成扩散与最优传输之间的三个简单联系：数据加噪的联合分布在每个时刻都是熵正则化最优传输问题的解（因此反向扩散等价于逐渐降低熵正则化的退火过程），且扩散与流匹配中分离数据模式的动力学不稳定性，在最优传输中表现为生成初期一个被熵正则化平滑的指数级薄边界层。

    

    我们讨论了生成扩散与最优传输之间的三个简单联系。首先，数据与其 Ornstein-Uhlenbeck 加噪过程的联合分布在每一时刻都是某个熵正则化最优传输问题的解，其温度由噪声水平决定。因此，反向扩散可以被解读为一种逐渐降低熵正则化的退火过程。其次，我们研究了物种形成转变（speciation transition）——即数据分布的各个模式在生成过程中如何浮现——并比较了它在扩散模型、流匹配与最优传输中的发生方式。以高斯混合分布作为高维例子，我们表明扩散与流匹配借以分离各模式的动力学不稳定性，在最优传输情形下变为生成初始阶段一个指数级薄的边界层，而该边界层会被熵正则化所平滑。第三，考虑一个带有一般前向动力学的……（摘要原文在此处截断）

    arXiv:2610.13614v1 Announce Type: cross  Abstract: We discuss three simple connections between generative diffusion and optimal transport. First, the joint law of the data and of their Ornstein-Uhlenbeck noising is, at every time, the solution of an entropic optimal-transport problem, at a temperature set by the noise level. Backward diffusion can consequently be read as an annealing which gradually decreases the entropic regularization. Second, we study the speciation transition--how the modes of the data distribution emerge during generation--and compare how it happens in diffusion and flow matching with respect to optimal transport. Using Gaussian mixtures as high-dimensional example, we show that the dynamical instability through which diffusion and flow matching separate the modes becomes, for optimal transport, an exponentially thin boundary layer at the beginning of generation, which is rounded by entropic regularization. Third, considering a generic forward dynamics with a gene
    
[^49]: FERPO：前向熵正则化策略优化

    FERPO: Forward Entropy-Regularized Policy Optimization

    [https://arxiv.org/abs/2610.02198](https://arxiv.org/abs/2610.02198)

    FERPO提出了一种无需对评论家求动作导数的在策略最大熵强化学习算法，通过熵与KL散度正则化的策略改进目标推导最优动作分布，并借助自归一化重要性采样以最小化前向KL来拟合Actor，从而避免了因价值预测不准导致的不可靠策略更新。

    

    在连续控制的在线强化学习中，若干最先进的方法利用学习到的评论家对动作的梯度来改进策略。然而，评论家通常被训练用于预测回报，准确的价值预测并不一定能产生准确的动作导数，这可能导致不可靠的策略更新。我们提出前向熵正则化策略优化（FERPO），这是一种在策略的最大熵强化学习算法，它利用评论家的价值进行策略改进，而无需对评论家关于动作进行求导。FERPO从由熵和Kullback-Leibler（KL）散度正则化的策略改进目标中推导出最优的目标动作分布。随后，我们通过最小化前向KL目标将Actor拟合到该目标分布上，该目标使用自归一化重要性采样（SNIS）进行估计，动作从rollout策略中抽取。通过限制目标分布……

    arXiv:2610.02198v1 Announce Type: cross  Abstract: Several state-of-the-art methods for online reinforcement learning in continuous control improve policies using action gradients of a learned critic. However, critics are typically trained to predict returns, and accurate value predictions do not necessarily yield accurate action derivatives, potentially leading to unreliable policy updates. We propose Forward Entropy-Regularized Policy Optimization (FERPO), an on-policy maximum entropy reinforcement learning algorithm that performs policy improvement using critic values without differentiating the critic with respect to actions. FERPO derives an optimal target action distribution from a policy-improvement objective regularized by entropy and Kullback-Leibler (KL) divergence. We then fit the actor to this target by minimizing a forward-KL objective, estimated using self-normalized importance sampling (SNIS) with actions drawn from the rollout policy. By limiting the target distribution
    
[^50]: 基于显式锚点-正样本-负样本耦合的多边际逆向最优传输对比学习

    Multi-Marginal Inverse Optimal Transport for Contrastive Learning Via Explicit Anchor-Positive-Negative Coupling

    [https://arxiv.org/abs/2609.33741](https://arxiv.org/abs/2609.33741)

    该论文提出Neg-MMIOT-CL方法，通过显式建模锚点-正样本-负样本三元组的多边际逆向最优传输耦合，克服了现有逆向OT方法因不利用负样本而导致的维度坍塌问题，提升了对比学习的下游性能。

    

    基于逆向最优传输（OT）的表示学习方法，通过学习表示使得表示空间中一对数据边缘分布之间的全局OT耦合集中在正样本对上，这与以往主要关注成对匹配的方法形成对比。然而，这些方法并未利用负样本对，因此其方式并非真正意义上的对比学习。我们证明这会导致维度坍塌问题，进而降低下游任务性能。为缓解这一问题，我们开发了一种新颖的多边际（MM）逆向最优传输（IOT）对比学习（CL）方法，称为Neg-MMIOT-CL，该方法学习表示，使得三个数据边缘分布之间的全局多边际最优传输（MMOT）耦合——相对于表示空间中数据点三元组之间精心设计的基础代价函数——集中在锚点-正样本-负样本三元组上。

    arXiv:2609.33741v2 Announce Type: replace  Abstract: Inverse Optimal Transport (OT) based methods for representation learning learn representations such that the global OT coupling between a pair of data marginals in the representation space, concentrates on the positive pairs. This is in contrast to previous methods that primarily focused on pairwise matching. However, these methods $\textit{DO NOT}$ utilize negative pairs and hence are not truly contrastive in their approach. We show that this leads to issues of dimensional collapse and hence degraded downstream performance. To alleviate this, we develop a novel multi-marginal (MM) inverse OT (IOT) contrastive learning (CL) approach called Neg-MMIOT-CL, which learns representations such that the global multi-marginal OT (MMOT) coupling between a triple of data marginals, with respect to a carefully designed ground-cost between triplets of data points in the representation space, concentrates on the anchor-positive-negative $\textit{t
    
[^51]: 秩能买到什么？低秩适配（LoRA）的谱与分布分析

    What Does the Rank Buy? A Spectral and Distributional Analysis of Low-Rank Adaptation

    [https://arxiv.org/abs/2609.32002](https://arxiv.org/abs/2609.32002)

    该论文证明在硬性逐因子范数预算下，LoRA的秩上限无法起到容量控制作用——所有复杂度与位移泛函均由秩一更新最大化、Rademacher复杂度与秩无关且分布位移上界与秩无关且紧致——并指出了秩真正发挥作用的两个位置。

    

    LoRA中的秩 $r$ 被广泛视为一种容量控制手段：人们通常假设较小的秩会带来更简单、泛化能力更强的模型。我们证明，在硬性逐因子范数预算——即实践中权重衰减与范数控制的理想化——之下，这一直觉会失效。其原因在于结构层面：在这类预算下，LoRA所能实现的更新恰好是核范数球内秩至多为 $r$ 的矩阵集合，而我们分析的每一个复杂度与位移泛函在该集合上均由秩一更新取得最大值——因此秩上限从未真正起到约束作用。由此可直接得出若干结论：我们所研究的线性读出模型类对所有 $r \ge 1$ 都是完全相同的，其Rademacher复杂度不依赖于 $r$，且适配所能使源分布产生的位移服从一个与秩无关的上界，我们证明该上界是紧的。如果秩并不控制容量，那它的作用体现在哪里？我们确定了两个位置。（原文摘要在此处被截断）

    arXiv:2609.32002v2 Announce Type: replace  Abstract: The rank $r$ in LoRA is widely treated as a capacity control: a smaller rank is assumed to yield a simpler model that generalizes better. We show that, under hard per-factor norm budgets---the idealization of the weight decay and norm control used in practice---this intuition breaks down. The reason is structural: under such budgets, the updates LoRA can reach are exactly the matrices of rank at most $r$ inside a nuclear-norm ball, and every complexity and displacement functional we analyze is maximized over this set by a rank-one update---so the rank cap never binds. The consequences follow directly. The linear-readout model class we study is identical for every $r \ge 1$, its Rademacher complexity carries no dependence on $r$, and the distance the adaptation can move the source distribution obeys a rank-independent upper bound that we show is sharp. If rank does not control capacity, where does it act? We identify two places. Stati
    
[^52]: 稀疏随机张量谱范数的锐利集中性

    Sharp spectral norm concentration of sparse random tensors

    [https://arxiv.org/abs/2609.20520](https://arxiv.org/abs/2609.20520)

    本文证明了稀疏伯努利随机张量谱范数的锐利集中不等式，即当 $np\ge c\log n$ 时谱范数以高概率被 $C\sqrt{np}$ 控制，消除了以往结果中的对数因子，并将结果推广至非齐次采样情形及随机超图模型的无对数第二特征值界。

    

    我们证明了具有独立伯努利项的稀疏随机张量谱范数的锐利集中不等式。设 $T$ 是维度为 $n\times\cdots\times n$ 的 $k$ 阶张量，其各项独立服从伯努利$(p)$ 分布，其中 $k$ 固定。对于任意 $c,r>0$，我们证明当 $np\ge c\log n$ 时，$\|T-\mathbb E T\|\le C_{k,r,c}\sqrt{np}$ 以至少 $1-n^{-r}$ 的概率成立。我们将该界推广到具有确定性逐项权重的非齐次伯努利采样。这一结果消除了 Zhou 和 Zhu（2021）工作中的对数因子。证明遵循 Kahn–Szemerédi 轻-重分解方法，并对重元组部分给出了精细估计。我们还为 Friedman 和 Wigderson（1995）的随机超图模型得到了一个不含对数因子的第二特征值界。

    arXiv:2609.20520v1 Announce Type: cross  Abstract: We prove a sharp concentration inequality for the spectral norm of sparse random tensors with independent Bernoulli entries. Let $T$ be an order-$k$ tensor of dimension $n\times\cdots\times n$ with independent Bernoulli$(p)$ entries, where $k$ is fixed. For any $c,r>0$, we show that $\|T-\mathbb E T\|\le C_{k,r,c}\sqrt{np}$ with probability at least $1-n^{-r}$ whenever $np\ge c\log n$. We extend this bound to inhomogeneous Bernoulli sampling with deterministic entrywise weights. This removes the logarithmic factor in the work of Zhou and Zhu (2021). The proof follows the Kahn--Szemer\'edi light--heavy decomposition with a refined estimate on the heavy tuple part. We also obtain a log-free second eigenvalue bound for the random hypergraph model of Friedman and Wigderson (1995).
    
[^53]: 基于|D|维稀疏地标嵌入的非CND距离度量通用核框架

    A General Kernel Framework for Non-CND Distance Measures Using |D|-Dimensional Sparse Landmark Embeddings

    [https://arxiv.org/abs/2609.19083](https://arxiv.org/abs/2609.19083)

    提出稀疏地标嵌入（SLE）核框架，通过紧支撑凸块函数将输入嵌入为稀疏特征向量，使得任意距离度量（包括非条件负定度量）都能构造出可证明半正定的核矩阵，从而完全摆脱核方法（如高斯过程）对希尔伯特距离条件的依赖。

    

    核方法，尤其是高斯过程（GP），需要希尔伯特距离度量（即其平方为条件负定（CND）的度量）来保证核矩阵的半正定性（PSD）；而这一条件在许多自然输入空间上并不成立，包括光滑流形和概率分布空间。我们提出了稀疏地标嵌入（SLE）核，彻底消除了这一要求。每个输入通过以全部|D|个训练点为中心的紧支撑凸块函数被嵌入为稀疏特征向量；在该嵌入空间中应用任何标准PSD核，即可得到对任意距离度量均可证明为半正定的核。紧支撑特性自动控制了嵌入的稀疏性，使得尽管环境维度很高，核矩阵仍保持良态且计算上可行。我们对PSD性质、稀疏性、稳定性以及普适性提供了理论保证。

    arXiv:2609.19083v1 Announce Type: cross  Abstract: Kernel methods, and Gaussian Processes (GPs) in particular, require a Hilbertian distance measure---one whose square is conditionally negative definite (CND)---to guarantee positive semi-definiteness (PSD) of the kernel matrix; a condition that fails for many natural input spaces, including smooth manifolds and spaces of probability distributions. We propose the Sparse Landmark Embedding (SLE) kernel, which eliminates this requirement entirely. Each input is embedded into a sparse feature vector via compactly supported bump functions centered at all |D| training points; applying any standard PSD kernel in this embedding space yields a kernel that is provably PSD for arbitrary distance measures. The compact support automatically controls embedding sparsity, keeping kernel matrices well-conditioned and computationally tractable despite the high ambient dimension. We provide theoretical guarantees on PSD, sparsity, stability, and universa
    
[^54]: ARISE：一种自适应残差信息稳定性集成方法，用于小样本生物医学组学中的特征选择

    ARISE: An adaptive residual-informed stability ensemble for feature selection in small-sample biomedical omics

    [https://arxiv.org/abs/2608.14866](https://arxiv.org/abs/2608.14866)

    ARISE通过自适应加权组合多种相关性信号和残差信息冗余控制，在小样本生物医学组学数据中显著提升了特征选择的预测性和稳定性。

    

    摘要：目标：小样本分子分类需要能够为二元和多类结果识别预测性、稳定性和非冗余子集的特征选择器。我们提出了ARISE（自适应残差信息稳定性集成），它整合了互补的相关性信号、类平衡的稳定性评估、残差信息冗余控制以及多类配对覆盖。方法：ARISE通过15个预定义配置文件组合七个百分位归一化的相关性成分，并通过嵌套内部交叉验证自适应加权。它在五个分子数据集、八个特征集大小、三个固定分类器（k近邻、支持向量机和随机森林）以及六个过滤比较器上进行了评估。泛化性能通过五次外部交叉验证重复50次，使用平衡准确率、宏观F1分数和Cohen's kappa进行估计。结果：在210,000次留出评估中，ARISE排名第一。

    arXiv:2608.14866v1 Announce Type: cross  Abstract: Objective: Small-sample molecular classification requires feature selectors that identify predictive, stable, and nonredundant subsets for binary and multiclass outcomes. We propose ARISE (Adaptive Residual-Informed Stability Ensemble), which integrates complementary relevance signals, class-balanced stability assessment, residual-informed redundancy control, and multiclass pairwise coverage.   Methods: ARISE combines seven percentile-normalized relevance components through 15 predefined profiles, adaptively weighted by nested inner cross-validation. It was evaluated on five molecular datasets, eight feature-set sizes, three fixed classifiers (k-nearest neighbours, support vector machine, and random forest), and six filter comparators. Generalization was estimated by five-fold outer cross-validation repeated 50 times using balanced accuracy, macro-F1, and Cohen's kappa.   Results: Across 210,000 held-out assessments, ARISE ranked first
    
[^55]: 求解椭圆狄利克雷问题的边界自适应物理信息神经网络的 $H^2(\Omega)$ 先验误差界

    $H^2(\Omega)$ A Priori Error Bounds for Boundary-Adapted PINNs Solving Elliptic Dirichlet Problems

    [https://arxiv.org/abs/2607.19167](https://arxiv.org/abs/2607.19167)

    本文通过乘以适当的到边界距离函数精确施加边界条件，构造了“边界自适应”神经网络，证明精确施加边界条件（而非 $L^2$ 边界惩罚）可以实现椭圆狄利克雷问题的 $H^2(\Omega)$ 误差控制，并推导出显式的高概率收敛速率。

    

    我们研究通过在神经网络假设空间中最小化偏微分方程残差来数值求解一致椭圆狄利克雷边值问题（BVPs），其中神经网络通过乘以用户定义的“到边界距离近似”函数 $\rho$ 来精确满足边界条件。虽然 $L^2(\partial\Omega)$ 边界惩罚通常无法保证 $H^2(\Omega)$ 误差控制，但我们证明了只要 $\rho$ 选择得当，精确施加边界条件可以实现这种控制。更确切地说，我们确定了 $\rho$ 上保持 $H^2(\Omega)$ 误差控制所需正则性的一个尖锐充分条件，为一般光滑区域 $\Omega$ 提供了此类 $\rho$ 的实用构造，并将由此得到的网络称为“边界自适应”网络。对于边界自适应的整流二次单元网络和双曲正切网络，我们推导出了显式的高概率 $H^2(\Omega)$ 收敛速率。

    arXiv:2607.19167v2 Announce Type: replace-cross  Abstract: We study the numerical resolution of uniformly elliptic Dirichlet boundary value problems (BVPs) by minimization of the PDE residual over a hypothesis space of Neural Networks made to satisfy exactly the boundary conditions through multiplication by a user-defined \emph{distance-to-boundary approximation} $\rho$. While an $L^2(\partial\Omega)$ boundary penalty cannot generally guarantee $H^2(\Omega)$ error control, we show that exact enforcement can, provided $\rho$ is chosen appropriately. More precisely, we identify a sharp sufficient condition on $\rho$ preserving the regularity needed for $H^2(\Omega)$ error control, provide practical constructions of such $\rho$ for general smooth domains $\Omega$, and refer to the resulting networks as \emph{boundary-adapted}. For boundary-adapted Rectified Quadratic Unit (ReQU) and hyperbolic tangent (tanh) networks, we then derive explicit high-probability $H^2(\Omega)$ convergence rate
    
[^56]: 内存受限大语言模型服务中的服务诱导型拥塞

    Service-Induced Congestion in Memory-Constrained LLM Serving

    [https://arxiv.org/abs/2606.15555](https://arxiv.org/abs/2606.15555)

    该论文将内存受限的LLM服务建模为离散时间动力学系统，揭示了服务过程本身会内生地累积显存压力并引发请求驱逐，并证明了在同质负载下无驱逐平衡态不稳定，系统通常会收敛到带驱逐的极限环。

    

    在大语言模型（LLM）服务中，每个请求在服务过程中会持续占用GPU显存，因为其键值缓存会随着每个生成令牌而不断增长。在高并发情况下，总体内存使用量因此会随时间内生地增长：服务过程本身会制造未来的容量压力。当超出内存容量时，系统会驱逐活跃请求，丢弃已缓存的状态并在之后重新启动它们，这会浪费计算资源并降低吞吐量。我们建立了一个内存受限LLM推理的离散时间动力学模型，用以刻画连续批处理下的请求准入、内存增长与驱逐行为。在饱和输入状态下，系统既存在无驱逐的不动点，也存在带驱逐的极限环。对于同质工作负载，我们证明无驱逐平衡态是不稳定的，并且除了一个勒贝格测度为零的精确捕获集合之外，系统均收敛于……（摘要原文在此处截断）

    arXiv:2606.15555v3 Announce Type: replace-cross  Abstract: In large language model (LLM) serving, each request accumulates persistent graphics processing unit (GPU) memory during service as its key-value cache grows with every generated token. Under high concurrency, aggregate memory usage therefore increases endogenously over time: the service process itself creates future capacity pressure. When memory capacity is exceeded, systems evict active requests, discarding cached state and restarting them later, which wastes computation and reduces throughput. We develop a discrete-time dynamical model of memory-constrained LLM inference that captures admission, memory growth, and eviction under continuous batching. In the saturated-input regime, the system admits both eviction-free fixed points and limit cycles with evictions. For homogeneous workloads, we show that the eviction-free equilibrium is unstable and that, except for a Lebesgue-measure-zero exact-capture set, the system converges
    
[^57]: 一种深度零膨胀模型用于美国东海岸北大西洋露脊鲸存在性建模以支持蓝色经济管理

    A Deep Zero-Inflated Model of North Atlantic Right Whale Presence To Support Blue Economy Management in the U.S. East Coast

    [https://arxiv.org/abs/2606.14403](https://arxiv.org/abs/2606.14403)

    本文提出了一种深度零膨胀伯努利模型，联合建模物种存在性与检测概率，有效处理被动声学监测数据中的零膨胀和复杂依赖，为濒危物种保护与蓝色经济管理提供新工具。

    

    arXiv:2606.14403v2 公告类型：替换-交叉  摘要：对濒危海洋哺乳动物物种（如北大西洋露脊鲸）的有效建模，对于在日益增长的蓝色经济中平衡海洋保护至关重要。由自主水下航行器收集的被动声学监测数据为局部海洋物种检测和海洋学感知提供了新机会，但也引入了复杂的统计挑战，如零膨胀、不完全检测和复杂依赖结构。为此，我们提出了深度零膨胀伯努利（DeepZIB）模型——一种深度统计方法，该方法联合建模潜在物种存在性和条件检测概率，同时从异构协变量信息中学习复杂的栖息地关系。我们建立了模型结构性质的理论结果，并进行了模拟实验以证明其恢复底层参数和潜在存在场的能力。应用...

    arXiv:2606.14403v2 Announce Type: replace-cross  Abstract: Effective modeling of endangered marine mammal species, such as the North Atlantic Right Whale, is critical for balancing marine conservation with the growing blue economy. Passive acoustic monitoring data collected by autonomous underwater vehicles provide new opportunities for localized marine species detection and oceanographic sensing, but introduce complex statistical challenges such as zero inflation, imperfect detection, and intricate dependence structures. In response, we propose the Deep Zero-Inflated Bernoulli (DeepZIB) model--a deep statistical method which jointly models latent species presence and conditional detection probabilities while learning complex habitat relationships from heterogeneous covariate information. We establish theoretical results on the model's structural properties and conduct simulation experiments to demonstrate its ability to recover underlying parameters and latent presence fields. Applica
    
[^58]: 多群体直推学习的代价

    The price of multi-group transductive learning

    [https://arxiv.org/abs/2606.04423](https://arxiv.org/abs/2606.04423)

    该论文揭示了在留一直推设定下，多群体学习器的错误率可能遭受随群体数量线性增长（最大达样本量平方根）的乘性惩罚，而在群体可实现的统计设定下，该惩罚仅为与群体数量无关的多对数因子。

    

    我们证明了在留一直推设定下，任何多群体学习器在某些群体上的错误率相对于单群体设定下可达到的错误率，都可能遭受出人意料的大幅乘性惩罚。事实上，该惩罚可以随群体数量线性增长，最大可达约样本量的平方根。我们还证明了在类似的（群体可实现的）统计设定中，该惩罚始终至多为样本量的多对数因子，且与群体数量无关。

    arXiv:2606.04423v2 Announce Type: replace  Abstract: We show every multi-group learner in the leave-one-out transductive setting may incur a surprisingly large multiplicative penalty in its error rate on some group relative to the error rate achievable in the single-group setting. Indeed, the penalty can increase linearly with the number of groups, up to roughly the square-root of the sample size. We also show that in the analogous (group-realizable) statistical setting, the penalty is always at most a poly-logarithmic factor in the sample size, independent of the number of groups.
    
[^59]: 基于软路径律约束的学习生成动力学：一种McKean-Vlasov正倒向随机微分方程方法

    Learning Generative Dynamics with Soft Path-Law Constraints: A McKean-Vlasov FBSDE Approach

    [https://arxiv.org/abs/2605.08928](https://arxiv.org/abs/2605.08928)

    本文提出一个统一框架，将软分布律约束与路径依赖监督相结合，把从分布观测和离散轨迹学习随机动力学的问题表述为离散路径依赖的McKean-Vlasov控制问题，其最优性系统为McKean-Vlasov正倒向随机微分方程（FBSDE）。

    

    我们提出了一个从分布观测和离散观测轨迹中学习随机动力学的统一框架。其关键贡献是将软分布律约束（用于对经验分布拟合进行正则化）与路径依赖监督（用于捕捉仅凭边际分布无法识别的时间结构）相结合。我们将该问题表述为一个离散路径依赖的McKean-Vlasov控制问题，在单一目标函数中对终端点分布、中间边际分布以及联合采样路径分布施加惩罚。该问题的最优性系统呈现为McKean-Vlasov正倒向随机微分方程的形式，其中边际项随时间连续作用，而路径律项则在观测时刻引发条件伴随跳。我们针对二次控制代价和常数扩散系数的情形推导了简化的FBSDE，并对基于最大均值差的惩罚系统建立了局部适定性……（摘要原文在此处截断）

    arXiv:2605.08928v2 Announce Type: replace-cross  Abstract: We introduce a unified framework for learning stochastic dynamics from distributional observations and discretely observed trajectories. The key contribution is to combine soft law constraints, which regularize fitting to empirical distributions, with path-dependent supervision, which captures temporal structure that marginal distributions alone cannot identify. We formulate the problem as a discrete path-dependent McKean-Vlasov control problem, penalizing endpoint, intermediate marginal, and joint sampled-path laws within a single objective. Its optimality system takes the form of a McKean-Vlasov forward-backward stochastic differential equation (FBSDE), in which the marginal terms act continuously in time while the path-law term instead induces conditional adjoint jumps at the observation dates. We derive the reduced FBSDE for quadratic control cost and constant diffusion, establish local well-posedness for a Maximum Mean Dis
    
[^60]: 面向灵活的基于密度的聚类的可扩展后验不确定性量化

    Scalable Posterior Uncertainty for Flexible Density-Based Clustering

    [https://arxiv.org/abs/2603.03188](https://arxiv.org/abs/2603.03188)

    该论文将基于密度的聚类与鞅后验分布相结合，提出了一种完全并行化、可扩展的贝叶斯不确定性量化框架，兼容归一化流等可微密度估计器，并能对任意聚类相关量进行推断，同时具备严格的频率派收敛性保证。

    

    基于密度的聚类是一种成熟的框架，它通过数据生成密度的拓扑或几何特征来定义聚类，由此产生的灵活方法在大规模和复杂数据集上已被证明是有效的。我们通过将基于密度的聚类与鞅后验分布相结合，解决了获得可扩展的贝叶斯式不确定性量化的问题。密度的不确定性首先通过一种完全并行化的预测重采样方案来量化，该方案由模型得分函数驱动，自然地兼容诸如归一化流等现代可微密度估计器。随后，通过将从鞅后验中采样的每个随机密度进行基于密度的聚类，获得聚类的后验样本，从而能够对任何与聚类相关的量进行推断。将估计目标表示为密度泛函还使得严格的频率派收敛分析成为可能。

    arXiv:2603.03188v3 Announce Type: replace-cross  Abstract: Density-based clustering is a well-established framework that defines clusters through topological or geometric features of the data-generating density, yielding flexible procedures that have proven effective on large and complex datasets. We address the problem of obtaining scalable Bayesian-style uncertainty quantification by combining density-based clustering with martingale posterior distributions. Density uncertainty is first quantified through a fully parallelized predictive resampling scheme driven by model score functions, naturally accommodating modern differentiable density estimators such as normalizing flows. Posterior samples of the clustering are then obtained by applying density-based clustering to each random density sampled from the martingale posterior, enabling inference on any clustering-related quantity. Casting the estimand as a density functional also enables a rigorous frequentist convergence analysis. W
    
[^61]: 针对带对抗性污染的异方差广义线性赌博机的联合高效且近最优算法

    A Jointly Efficient and Near-Optimal Algorithm for Heteroskedastic Generalized Linear Bandits with Adversarial Corruptions

    [https://arxiv.org/abs/2602.10971](https://arxiv.org/abs/2602.10971)

    本文提出将基于在线镜像下降的估计器与基于Hessian的置信度权重相结合的HCW-GLB-OMD算法，为带对抗性污染的异方差广义线性赌博机实现了联合高效且近最优的高概率遗憾保证。

    

    我们研究了带有对抗性污染的异方差广义线性赌博机（GLBs），并提出了一种简单算法 HCW-GLB-OMD，该算法将基于在线镜像下降（OMD）的估计器与基于Hessian矩阵的置信度权重相结合。尽管这两个组成部分都已被充分研究，但要为该组合算法建立理论保证仍需要非平凡的分析。我们利用先前分析中不需要的技术工具（包括加权似然上鞅和自约束论证）建立了抗污染的置信序列。在遗憾分析方面，我们推导了一个新的Hessian势函数引理，以处理对抗性污染与自适应置信度加权之间的相互作用。对于自协和（self-concordant）广义线性模型，我们证明了高概率遗憾界 $\widetilde{\mathcal{O}}\bigl( d \sqrt{\sum_t g(\tau_t) \dot\mu_{t,\star}} + d^2 g_{\max} \kappa + d(1+\kappa)(C\vee1) \bigr)$，其中 $\dot\mu_{t,\star}$ 是……

    arXiv:2602.10971v3 Announce Type: replace  Abstract: We consider heteroskedastic generalized linear bandits (GLBs) with adversarial corruptions and propose \texttt{HCW-GLB-OMD}, a simple algorithm that combines an online mirror descent (OMD)-based estimator with Hessian-based confidence weights. Although both components are well studied, establishing guarantees for the combined algorithm requires a nontrivial analysis. We establish a corruption-robust confidence sequence using technical tools not required in prior analyses, including a weighted likelihood supermartingale and a self-bounding argument. For the regret analysis, we derive a new Hessian potential lemma to handle the interaction between adversarial corruption and adaptive confidence weighting. For self-concordant GLMs, we prove a high-probability regret bound of $\widetilde{\mathcal{O}}\bigl( d \sqrt{\sum_t g(\tau_t) \dot\mu_{t,\star}} + d^2 g_{\max} \kappa + d(1+\kappa)(C\vee1) \bigr)$, where $\dot\mu_{t,\star}$ is the slop
    
[^62]: 基于选择性部署的鲁棒残差校正时间序列预测方法

    Robust Residual Correction via Selective Deployment for Time Series Forecasting

    [https://arxiv.org/abs/2512.22428](https://arxiv.org/abs/2512.22428)

    CRC框架通过验证集校准的选择性部署规则，仅在校正能带来净误差降低时才应用混合残差校正，从而在时间序列预测中实现鲁棒的误差降低，尤其在复杂场景下避免了过拟合问题。

    

    事后残差校正可以降低预测误差，但持续开启的更新往往难以泛化：当残差信号微弱或充满噪声时，表达能力强的校正器可能会过拟合，甚至增大整体预测误差。我们提出了CRC，一个鲁棒的残差校正框架，它将混合校正器与经过验证集校准的选择性部署规则相结合。CRC通过保守的岭回归“底线”加上条件性非线性精炼来生成校正，但只有在留出数据证据表明相对于冻结的基线预测器能够净减少误差时才部署这些校正。CRC将“何时修正预测”视为一个经验证集校准的部署决策，而不仅仅是“如何拟合残差”。在标准长序列基准测试和多个骨干模型上，CRC在大多数数据集-骨干模型组合中提升了对齐平均的MSE和MAE，在复杂数据集上收益尤为显著（例如，在Traffic数据集上相对MSE降低超过20%）。

    arXiv:2512.22428v3 Announce Type: replace  Abstract: Post-hoc residual correction can reduce forecasting error, but always-on updates often fail to generalize: when residual signals are weak or noisy, expressive correctors can overfit and even increase the overall forecasting errors. We introduce CRC, a robust residual correction framework that couples a hybrid corrector with a validation-calibrated selective deployment rule. CRC proposes corrections via a conservative ridge "floor" plus conditional nonlinear refinement, but deploys them only when held-out evidence indicates net error reduction over a frozen baseline forecaster. CRC treats when to revise a forecast as a validation-calibrated deployment decision, not only how to fit residuals. Across standard long-sequence benchmarks and multiple backbones, CRC improves horizon-averaged MSE and MAE in most dataset-backbone pairs, with particularly large gains on complex datasets (e.g., over 20% relative MSE reduction on Traffic). We add
    
[^63]: FLEXI-Haz：超越比例风险假设的半参数深度生存推断

    FLEXI-Haz: Semiparametric Deep Survival Inference Beyond Proportional Hazards

    [https://arxiv.org/abs/2512.10570](https://arxiv.org/abs/2512.10570)

    提出 FLEXI-Haz 半参数深度神经网络模型，在突破比例风险假设限制的同时保留可解释的回归参数，并针对右删失数据实现了极小化极大最优的收敛速率及渐近有效的半参数统计推断。

    

    灵活的生存预测本身并不能保证对协变量效应或个体特异性生存概率进行有效的统计推断。我们提出了 FLEXI-Haz，一种针对右删失数据的半参数深度神经网络模型，它将对数风险中可解释的线性成分与关于时间及调整协变量的灵活函数相结合。这种构造在允许存在非比例的讨厌参数效应的同时，保留了可解释的回归参数。在正则性条件下，我们建立了在复合 Hölder 函数类上直至对数因子意义下达到极小化极大最优的讨厌参数收敛速率，并实现了回归参数的 √n 一致估计、渐近正态性以及半参数有效性。对于取值有限的调整协变量，我们推导了累积风险的显式影响函数，并构造了交叉拟合的一步估计量，该估计量在逐点意义上具有渐近有效性。

    arXiv:2512.10570v3 Announce Type: replace-cross  Abstract: Flexible survival prediction does not by itself ensure valid inference for covariate effects or subject-specific survival probabilities. We develop FLEXI-Haz, a semiparametric deep neural network model for right-censored data that combines an interpretable linear component in the log-hazard with a flexible function of time and adjustment covariates. This formulation permits nonproportional nuisance effects while retaining interpretable regression parameters. Under regularity conditions, we establish nuisance convergence rates that are minimax optimal up to logarithmic factors over composite Holder classes, together with root-n-consistent, asymptotically normal, and semiparametrically efficient estimation of the regression parameter. For adjustment covariates with finite support, we derive the explicit influence function for cumulative hazard and construct a cross-fitted one-step estimator yielding asymptotically valid pointwise
    
[^64]: 差分隐私作为额外收益：基于多天线基站的多址接入衰落信道上的联邦学习

    Differential Privacy as a Perk: Federated Learning over Multiple-Access Fading Channels with a Multi-Antenna Base Station

    [https://arxiv.org/abs/2510.23463](https://arxiv.org/abs/2510.23463)

    该论文研究了基于多天线基站的多址接入衰落信道上的空中联邦学习，巧妙地将信道噪声从性能损害转化为差分隐私保护的天然随机性来源，突破了现有工作在信道模型和损失函数假设上的限制，实现了隐私保护与训练性能的协同优化。

    

    联邦学习（FL）是一种分布式学习范式，它通过在训练过程中无需交换原始数据来保护隐私。在其典型的边缘部署实例中，无线传输由模拟空中计算（AirComp）实现，称为空中联邦学习（AirFL），其中固有的信道噪声扮演着一种独特的“亦敌亦友”的角色：一方面，它因带噪的全局聚合而降低训练性能；另一方面，它为隐私保护机制提供了天然的随机性来源，而这类隐私保护可以通过差分隐私（DP）进行形式化量化。然而，有效利用这种信道损伤仍然具有挑战性，因为现有工作大多在简单信道模型或受限损失函数类型的假设下，仅考虑单轮或非收敛隐私损失界限下的（本地）差分隐私增强。在本文中，我们研究了多址接入衰落信道上的空中联邦学习

    arXiv:2510.23463v4 Announce Type: replace  Abstract: Federated Learning (FL) is a distributed learning paradigm that preserves privacy by eliminating the need to exchange raw data during training. In its prototypical edge instantiation with underlying wireless transmissions enabled by analog over-the-air computing (AirComp), referred to as \emph{over-the-air FL (AirFL)}, the inherent channel noise plays a unique role of \emph{frenemy} in the sense that it degrades training due to noisy global aggregation while providing a natural source of randomness for privacy-preserving mechanisms, formally quantified by \emph{differential privacy (DP)}. It remains, nevertheless, challenging to effectively harness such channel impairments, as prior arts, under assumptions of either simple channel models or restricted types of loss functions, mostly considering (local) DP enhancement with a single-round or non-convergent bound on privacy loss. In this paper, we study AirFL over multiple-access fading
    
[^65]: Wasserstein鲁棒分类误差：代理损失失配与分布攻击

    Wasserstein Robust Classification Error: Surrogate Loss Misalignment and Distributional Attacks

    [https://arxiv.org/abs/2510.10000](https://arxiv.org/abs/2510.10000)

    本文证明了仅当损失函数是0-1损失的正倍数加标签相关偏移时，WDRO风险的排序才与鲁棒准确率的排序一致（交叉熵等常用损失存在失配问题），并提出Wasserstein分布攻击（WDA）来直接评估鲁棒准确率。

    

    Wasserstein分布鲁棒优化（WDRO）提供了一个通过参考分布周围Wasserstein球内最坏情况期望来评估模型鲁棒性的框架。我们证明，按WDRO风险递增对假设排序等价于按鲁棒准确率递减对假设排序，当且仅当损失函数是0-1损失的正倍数加上一个标签相关的偏移量。对于包括交叉熵在内的一类损失函数，我们证明了一个鲁棒准确率完美的假设反而可能比鲁棒准确率接近于零的假设具有更高的WDRO风险。为了直接评估鲁棒准确率，我们开发了Wasserstein分布攻击（WDA），它在候选支撑点搜索之后增加一个权重生成阶段，以在Wasserstein约束下优化经验质量（概率质量）的分配，而不是将每个数据点的全部质量分配给单一返回的攻击点。

    arXiv:2510.10000v3 Announce Type: replace  Abstract: Wasserstein distributionally robust optimization (WDRO) provides a framework for evaluating model robustness through the worst-case expectation in the Wasserstein ball around a reference distribution. We prove that ordering hypotheses by increasing WDRO risk is equivalent to ordering them by decreasing robust accuracy if and only if the loss function is a positive multiple of the zero-one loss plus a label-dependent offset. For a class of loss functions including cross-entropy, we show that a hypothesis with perfect robust accuracy can conversely have a higher WDRO risk than another hypothesis with near zero robust accuracy. To evaluate robust accuracy directly, we develop Wasserstein distributional attacks (WDA), which add a weight generation stage after candidate support search to optimize the allocation of empirical mass under the Wasserstein constraint, rather than assigning each data point's full mass to one returned attack poin
    
[^66]: 时间序列的贝叶斯非参数动态聚类

    Bayesian Nonparametric Dynamical Clustering of Time Series

    [https://arxiv.org/abs/2510.06919](https://arxiv.org/abs/2510.06919)

    提出一种基于层次狄利克雷过程与高斯过程先验的贝叶斯非参数方法，通过在未知数量的线性动态机制间切换来建模时间序列簇的演化，以原则性方式避免簇的无序增殖，并可通过变分推断实现离线与在线的高效学习，在心电图分析中验证了其有效性。

    

    我们提出了一种方法，通过在未知数量的线性动态机制之间切换，来建模无界数量的时间序列簇的演化。我们开发了一种贝叶斯非参数方法，使用层次狄利克雷过程作为切换线性动态系统（Switching Linear Dynamical System）参数的先验，并使用高斯过程先验来建模每个簇内振幅和时间对齐的统计变化。通过对时间序列模式的演化进行建模，该方法以有原则的方式避免了簇的不必要增殖。我们通过为离线和在线场景制定变分下界来执行推断，从而通过优化实现高效学习。我们通过使用公开数据库进行的多项心电图分析案例研究，展示了该方法的通用性和有效性。

    arXiv:2510.06919v2 Announce Type: replace-cross  Abstract: We present a method that models the evolution of an unbounded number of time series clusters by switching among an unknown number of regimes with linear dynamics. We develop a Bayesian non-parametric approach using a hierarchical Dirichlet process as a prior on the parameters of a Switching Linear Dynamical System and a Gaussian process prior to model the statistical variations in amplitude and temporal alignment within each cluster. By modeling the evolution of time series patterns, the method avoids unnecessary proliferation of clusters in a principled manner. We perform inference by formulating a variational lower bound for off-line and on-line scenarios, enabling efficient learning through optimization. We illustrate the versatility and effectiveness of the approach through several case studies of electrocardiogram analysis using publicly available databases.
    
[^67]: 评估临床预测模型的效用：决策曲线分析与成本曲线的比较

    Evaluating utility of clinical prediction models: A comparison of decision curve analysis and cost curves

    [https://arxiv.org/abs/2509.24608](https://arxiv.org/abs/2509.24608)

    本文比较了临床预测模型评估中的决策曲线分析与成本曲线两种方法，证明决策曲线与Brier曲线密切相关且两者间的参考线可相互转换，并提出上包络决策曲线可作为模型决策曲线的有用比较基准。

    

    分类模型通常预测二元结果，并使用一个或多个决策阈值来产生分类结果或为临床决策提供信息。决策曲线和成本曲线是评估模型在不同决策阈值下净收益和损失的模型评价方法。本文对决策曲线和成本曲线进行了比较，证明了决策曲线与一种特定类型的成本曲线——Brier曲线密切相关。使用这两种方法的用户都在假设模型分数已经过校准的前提下设置分类阈值，并依据正确阳性分类与正确阴性分类的相对临床重要性进行权衡。给定的决策曲线都存在一条与之对应的Brier曲线，使得每种空间中常见的参考线可以被纳入另一种空间中（例如，“不治疗”决策曲线可以转换为等效的“不治疗”Brier曲线）。我们建议将上包络决策曲线作为与模型决策曲线进行比较的有用基准。

    arXiv:2509.24608v2 Announce Type: replace  Abstract: Classification models typically predict a binary outcome and use one or more decision thresholds to produce a classification or inform a clinical decision. Decision curves and cost curves are model evaluation approaches that assess net benefit and loss across decision thresholds. In this paper we compare decision curves and cost curves, demonstrating that decision curves are closely related to a specific type of cost curve called a Brier curve. Users of both approaches set the classification threshold assuming calibrated model scores, using the relative clinical importance of correct positive and negative classifications. A given decision curve has a corresponding Brier curve such that reference lines common in each space can be included in either (e.g. the 'treat none' decision curve can be translated into an equivalent 'treat none' Brier curve). We suggest the upper envelope decision curve as a useful comparison to a model decision
    
[^68]: 联邦学习：从理论到实践

    Federated Learning: From Theory to Practice

    [https://arxiv.org/abs/2505.19183](https://arxiv.org/abs/2505.19183)

    本书是一本联邦学习的实践入门指南，其核心贡献是提出通过联邦学习网络的加权边编码设备间相似性来实现个性化联邦学习，使每个设备在训练自己模型的同时仍能受益于与相关设备的协作。

    

    本书提供了构建和理解联邦学习（FL）系统的实践性介绍。联邦学习使多个设备（如智能手机、传感器或本地计算机）能够协同训练机器学习（ML）模型，同时保持数据私密并存储在本地。当由于隐私、法规或技术原因无法或不应该将数据集中化时，这是一种强大的解决方案。本书面向希望学习如何设计可扩展的、保护隐私的联邦学习系统的学生、工程师和研究人员。我们的主要关注点是个性化：使每个设备能够训练自己的模型，同时仍能从与相关设备的协作中受益。这是通过利用设备之间（与其关联的学习任务之间）的相似性来实现的，这些相似性由联邦学习（FL）网络的加权边（或链接）进行编码。其核心思想是将现实世界中的联邦学习系统表示为设备网络。

    arXiv:2505.19183v4 Announce Type: replace  Abstract: This book offers a hands-on introduction to building and understanding federated learning (FL) systems. FL enables multiple devices -- such as smartphones, sensors, or local computers -- to collaboratively train machine learning (ML) models, while keeping their data private and local. It is a powerful solution when data cannot or should not be centralized due to privacy, regulatory, or technical reasons. The book is designed for students, engineers, and researchers who want to learn how to design scalable, privacy preserving FL systems. Our main focus is on personalization: enabling each device to train its own model while still benefiting from collaboration with relevant devices. This is achieved by leveraging similarities between (the learning tasks associated with) devices that are encoded by the weighted edges (or links) of a federated learning network (FL network). The key idea is to represent real-world FL systems as networks o
    
[^69]: 面向可认证少样本迁移学习的权重空间学习

    Weight Space Learning for Certifiable Few-shot Transfer Learning

    [https://arxiv.org/abs/2502.06970](https://arxiv.org/abs/2502.06970)

    该论文提出了一种新型迁移学习方法，通过在上游任务上训练PEFT参数的扩散模型分布，再利用采样-评估程序为下游任务选择最优参数，从而即使在少样本情况下也能获得可认证的非空洞学习理论泛化保证。

    

    在当代深度学习中，解决低数据问题的一个普遍且有效的工作流程是通过参数高效微调（PEFT）将强大的预训练基础模型（FM）适配到新任务上。然而，尽管这些方法在经验上有效，所得到的解决方案却缺乏能够认证其准确性的泛化保证——而在高重要性应用部署之前，出于伦理或法律方面的原因，可能需要这种保证。在本文中，我们开发了一种新颖的迁移学习方法，旨在即使在少样本情况下，也能为下游任务提供非空洞的学习理论泛化保证。具体而言，我们首先利用上游任务训练一个PEFT参数的分布。然后，我们通过一个“采样-评估”程序来学习下游任务——从训练好的扩散模型中采样合理的PEFT参数，并选择在下游数据上似然最高的那个。至关重要的是，这种配置（原文在此处截断）

    arXiv:2502.06970v3 Announce Type: replace  Abstract: In contemporary deep learning, a prevalent and effective workflow for solving low-data problems is adapting powerful pre-trained foundation models (FMs) to new tasks via parameter-efficient fine-tuning (PEFT). However, while empirically effective, the resulting solutions lack generalisation guarantees to certify their accuracy - which may be required for ethical or legal reasons prior to deployment in high-importance applications. In this paper we develop a novel transfer learning approach that is designed to facilitate non-vacuous learning theoretic generalisation guarantees for downstream tasks, even in the low-shot regime. Specifically, we first use upstream tasks to train a distribution over PEFT parameters. We then learn the downstream task by a sample-and-evaluate procedure -- sampling plausible PEFTs from the trained diffusion model and selecting the one with the highest likelihood on the downstream data. Crucially, this confi
    
[^70]: 帮我帮你：迁移学习中源数据与目标数据的聚合价值

    Help Me Help You: The Aggregate Value of Source and Target Data in Transfer Learning

    [https://arxiv.org/abs/2408.16189](https://arxiv.org/abs/2408.16189)

    该论文揭示了监督迁移学习中的两种机制二分现象：弱迁移机制下最优收敛率仅由源数据或目标数据之一决定，而强迁移机制下需联合利用源与目标数据的互补信息才能实现快得多的目标收敛率。

    

    迁移学习（无论是回归还是分类任务）所关注的问题是：如何利用来自源分布 P 的不完美数据来改进针对目标分布 Q 的预测。尽管目前对这一问题在无监督设置（即没有带标签目标数据可用）下已有大量理论认识，但带有标签目标数据的监督设置仍然研究不足。我们识别并刻画了监督设置下可实现的收敛率中一个有趣的二分现象：存在一个弱迁移机制，其中最优收敛率由源数据集或目标数据集中某一方的信息决定，而非两者共同决定；还存在一个强迁移机制，在该机制下可以实现快得多的目标收敛率，这需要同时考虑源数据和目标数据中的互补信息。此外，虽然先前针对无监督设置的理论工具可以扩展以刻画弱迁移机制，但我们认为它们无法捕捉（原文摘要在此处截断）

    arXiv:2408.16189v4 Announce Type: replace-cross  Abstract: Transfer learning, in regression or classification, concerns problems where one aims to use imperfect data from a source distribution P to improve prediction w.r.t. a target distribution Q. While there is now much theoretical understanding of the unsupervised setting of this problem (where no labeled target data is available), the supervised setting with labeled target data remains under-studied.   We identify and characterize an interesting dichotomy in the rates achievable in the supervised setting: there is a weak transfer regime where optimal rates are determined by information in either the source or target dataset but not both, and a strong transfer regime where much faster target rates are achievable, accounting for complementary information in source and target data. Furthermore, while previous theoretical tools for the unsupervised setting can be extended to capture the weak transfer regime, we argue that they cannot c
    
[^71]: 对AI分类器的对抗鲁棒性度量的存在性，唯一性和可扩展性研究

    On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers. (arXiv:2310.14421v1 [stat.ML])

    [http://arxiv.org/abs/2310.14421](http://arxiv.org/abs/2310.14421)

    本文研究了针对AI分类器的对抗鲁棒性度量的存在性、唯一性和可扩展性，提出了可以验证的数学条件，并在合成基准测试和生物医学应用中进行了实际计算和解释。

    

    本文提出并证明了针对（局部）唯一可逆分类器、广义线性模型（GLM）和熵AI（EAI）具有最小对抗路径（MAP）和最小对抗距离（MAD）的存在性、唯一性和明确的分析计算的简单可验证的数学条件。在常见的合成基准测试数据集上，针对神经网络、提升随机森林、GLM和EAI等各类AI工具进行MAP和MAD的实际计算、比较和解释，包括双卷状螺旋线及其扩展以及两个生物医学数据问题（用于健康保险理赔预测和心脏病发作致死率分类）。在生物医学应用中，展示了MAP如何在预定义的可访问控制变量子集中提供唯一的最小患者特定风险缓解干预措施。

    Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.
    

