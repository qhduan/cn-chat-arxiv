# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Geometric Algebra for Neural Networks: A Survey](https://arxiv.org/abs/2610.30121) | 本综述系统梳理了几何代数神经网络的基本构件、等变性、应用与实证表现，指出基于几何代数的架构在具有显式几何对称性和异质几何实体的场景中优势最为显著。 |
| [^2] | [The role of locality and weight sharing in convolutional neural networks](https://arxiv.org/abs/2610.30118) | 本研究通过师生模型的精确动力学分析揭示：卷积网络中权重共享带来的学习加速与输入位置数 $P$ 成正比，而局部性则决定样本复杂度，二者的影响均依赖于输入位置间的空间相关性。 |
| [^3] | [Mitigating Model Collapse through Dreaming Learning](https://arxiv.org/abs/2610.30009) | 该论文从谱分析角度揭示低温采样（Ts < 1）会使转移算子锐化、谱隙收窄从而导致模型坍塌，并提出“梦境学习”正则化方法——在训练中对模型以高温生成的自身样本施加额外损失项——来缓解这一退化。 |
| [^4] | [The Price of Current-Effect Inference in Nonstationary Multi-Armed Bandits](https://arxiv.org/abs/2610.29982) | 该论文揭示了非平稳多臂老虎机中收益最大化与现时效应推断之间的内在权衡，证明了以有限宽度置信区间同时覆盖所有现时效应所需付出的动态遗憾极小极大代价，并提出 TRACE 算法来同时兼顾这两个目标。 |
| [^5] | [Mastering Model Fit: Applications of the InterModel Vigorish for Interpretable Machine Learning](https://arxiv.org/abs/2610.29960) | 本文将“模型间赌利（IMV）”引入机器学习领域，通过 SHAP-IMV、Multi-IMV 和 Ablate-IMV 等可解释工具，将模型拟合差异量化为相对于参考模型的期望投注优势，从而揭示哪些特征、模型组件和训练阶段真正改善了概率拟合。 |
| [^6] | [Silent Alignment in Two-Layer Linear Networks: A High-Dimensional Perspective](https://arxiv.org/abs/2610.29864) | 该论文在高维比例极限下借助随机矩阵理论精确刻画了双层线性网络梯度流的谱演化，并发现小初始化时存在“静默对齐”现象——网络奇异方向在一阶时间尺度上先完成对齐而损失保持不变，可见的学习随后才在对数时间尺度上以鞍点到鞍点的方式发生。 |
| [^7] | [How Do DiTs Denoise? An Aggregator-Predictor View of Attention and MLPs](https://arxiv.org/abs/2610.29854) | 论文提出将扩散Transformer视为“聚合器-预测器”架构的新解释框架——AdaLN调制的注意力充当依赖输入与噪声的上下文聚合器，MLP充当可学习键-值表示的patch预测器，并从理论上证明基于局部上下文的聚合能以指数衰减的误差逼近全局贝叶斯去噪器。 |
| [^8] | [Asymmetric Likelihood-Free Hypothesis Testing](https://arxiv.org/abs/2610.29798) | 本文刻画了非对称无似然假设检验中可行的样本预算三元组（至多相差对数因子），证明对参考分布 P 和 Q 分配不等的样本预算可带来多项式级别的节省——例如将来自 D 的所需样本量从 Ω(√n) 降至 Õ(n^(1/4))——并将其应用于非自适应分布 2-聚类问题。 |
| [^9] | [Post-hoc Epistemic Uncertainty for Vision-Language-Action Policies](https://arxiv.org/abs/2610.29730) | 提出了一种后验核方差头方法，无需训练多个策略或修改原策略，即可从单一VLA策略的嵌入中估计认知不确定性，其表现与双策略集成方法相当。 |
| [^10] | [Lifting PCA in the Wasserstein Space](https://arxiv.org/abs/2610.29679) | 本文提出了“提升PCA”，通过凸延拓提升泛函的框架，首次将欧氏主成分分析规范化地扩展到Wasserstein空间，并将其归结为两个等价的、具有广义代价的Wasserstein重心最小化问题。 |
| [^11] | [Near Minimax Rate for Quantile Distributional Reinforcement Learning](https://arxiv.org/abs/2610.29595) | 本文在生成模型假设下，为分位数Bellman不动点估计建立了在所有分辨率和所有Wasserstein阶数上一致成立的 $\sqrt{n}$ 高概率误差界，达到了分位数分布策略评估的近极小极大最优速率。 |
| [^12] | [Fragment-Level Graph Encoding for Interpretable Molecular Property Prediction](https://arxiv.org/abs/2610.29589) | 该论文提出用BRICS分解将分子编码为片段级图取代原子级图，在不改变模型结构的前提下使GNN能够给出功能团级别的可解释预测，并在约五分之一大小的图上达到与原子级编码相当的预测精度。 |
| [^13] | [Private Online Prediction from Experts with Small Losses](https://arxiv.org/abs/2610.29575) | 该论文提出了首个遗憾度达到 $\tilde{O}(\sqrt{L^\star})$ 的差分隐私专家在线预测算法，在低损失情形下显著优于现有 $\tilde{O}(\sqrt{T})$ 界限，并证明在忘却输入下隐私代价很小。 |
| [^14] | [Residual-Based Causal Discovery via Pretrained In-Context Quantile Regression](https://arxiv.org/abs/2610.29530) | 该论文提出Loscar方法，将RESIT框架从加性噪声模型扩展至更广泛的位置-尺度噪声模型，通过分位数回归非参数化地估计条件均值和尺度，并支持将预训练的表格基础模型作为即插即用估计器，从而避免了现有方法对参数化假设的依赖。 |
| [^15] | [BRIDGE: An Acceptance Gate to Block False Improvements in Model Updates, with a Retrospective Clinical Evaluation](https://arxiv.org/abs/2610.29515) | BRIDGE 提出了一种模型无关的接受门控机制，通过交叉拟合与 bootstrap 置信下界检验来拦截模型更新中的虚假改进，将虚假改进率从约 20% 降至接近零，并经 12 项回顾性临床任务验证有效。 |
| [^16] | [A unified approach for estimation of unnormalized models via Stein operators](https://arxiv.org/abs/2610.29510) | 本文建立了一个连接适当评分规则与Stein方法的统一框架，证明分数匹配、比率匹配、伪似然和噪声对比估计等评分规则可自然诱导Stein算子，并借助由此产生的几何视角统一刻画了相应估计量的渐近效率。 |
| [^17] | [Generative Modeling via Nonlinear Schr\"odinger Bridges and Branching Diffusions](https://arxiv.org/abs/2610.29476) | 该论文提出了基于前向-后向偏微分方程的非线性薛定谔桥生成建模框架SBNL，并利用分支扩散设计出具有高度并行性和高维可扩展性的高效数值算法。 |
| [^18] | [Density-Level Backward Error for Low-NFE Generative Flows](https://arxiv.org/abs/2610.29475) | 本文提出密度层面的后向误差框架，分离出真正改变边缘分布的求解器误差分量，从而为低NFE生成流导出感知求解器阶数的计算分配规则以及基于规范自由度的速度场选择目标。 |
| [^19] | [One Act, Many Masks: Effect-Marginal Policy Optimization for Long-Horizon Agents](https://arxiv.org/abs/2610.29427) | 该论文发现了多轮强化学习中信用信号与具体文本响应绑定所导致的“信用更新错位”问题，并提出效应边缘策略优化（EMPO），通过效应级别的代理目标将策略更新与执行层面的等价性对齐，从而改进长程LLM智能体的策略优化。 |
| [^20] | [When Thoughts Become Memory: Unifying In-Context Learning and Fine-Tuning](https://arxiv.org/abs/2610.29419) | 该论文首次提出基于信息价值精确定律的认证规则，统一判定未见数据应放入预训练模型的上下文还是其权重，并通过随时有效的检验在多种模型与任务上验证了其正确性。 |
| [^21] | [The Predictive--Optimization Gap in GuanDan Self-Play](https://arxiv.org/abs/2610.29411) | 在掼蛋自博弈中，对局进度虽被证明具有真实预测价值，但以冻结胜负预测器的变化 ΔΦ 作回合级监督干预，尽管在个别训练谱系中奏效，五个预注册实验在方法层面均未复现增益，揭示了预测有效性与优化效用之间的差距。 |
| [^22] | [Cascading Bandits with Noisy Observations: Algorithms and Regret Bounds](https://arxiv.org/abs/2610.29381) | 该论文提出了一种探测成本与位置相关且反馈含噪声的随机级联赌博机模型，刻画了最优短列表可通过按每臂分数排序求得的条件，并给出了实例相关遗憾匹配理论下界的UCB算法。 |
| [^23] | [GaP-MoE : Explicit Expert Specialization in Mixture-of-Experts via Orthogonal Gradient Accumulation and Projection](https://arxiv.org/abs/2610.29355) | 提出GaP-MoE方法，通过正交梯度累积与投影直接操纵专家输出梯度而非修改损失函数，从而在稀疏专家混合模型中诱导显式的专家专业化，解决表示坍缩问题。 |
| [^24] | [ExpEvolve: Structuring and Reusing Experience for Open-Ended Discovery](https://arxiv.org/abs/2610.29303) | 提出ExpEvolve框架，将思想层级的提取与控制注入进化式编码代理，通过发现能够解释程序间评分差异的高层级思想因素并复用经验，提升开放式科学发现的样本效率。 |
| [^25] | [Approximate Solomonoff Induction: On the Robustness of Learned Universal Predictors](https://arxiv.org/abs/2610.29289) | 本文研究了实际机器学习算法产生的近似最优预测器的鲁棒性问题，给出了近似对预测误差最坏情况影响的紧致界，从而论证了通过对数损失最小化学习到的通用预测器在近似条件下依然保持稳健。 |
| [^26] | [Schr\"odinger-Bass Bridge Barycenter Models for Measure-Valued Data](https://arxiv.org/abs/2610.29227) | 本文利用薛定谔-巴斯桥的方向不对称性与对偶漂移-扩散控制，提出四种新型测度值数据重心模型，并从理论上证明了这些重心的存在性，且证明高斯输入下解码器形式存在高斯最优解，可将重心计算化为有限维优化。 |
| [^27] | [Curvature-Scaled Sharpness-Aware Minimization: Convergence Beyond Smoothness](https://arxiv.org/abs/2610.29192) | 该论文利用相对梯度失真揭示了SAM在广义平滑条件下失真可能无界、SAM会循环而USAM会发散的现象，建立了梯度扰动统一失真控制的必要条件，并由此提出曲率缩放的锐度感知最小化方法，实现了超越平滑性假设的收敛性保证。 |
| [^28] | [Internal Guidance as a Sampling Method for Tilted Diffusion Models](https://arxiv.org/abs/2610.29086) | 本文提出“内部引导”方法，通过改变基础去噪器的求值输入点而非修改其输出来采样倾斜密度，并利用随机对角重缩放改进训练算法，解决了所需求值点超出训练分布的问题。 |
| [^29] | [A Jump-Modified Wasserstein Gradient Flow for Maximum Mean Discrepancy with Energy Kernels](https://arxiv.org/abs/2610.29007) | 本文提出一种带跳跃项的修正Wasserstein梯度流来最小化能量核生成的最大均值差异（MMD），证明了该流的全局适定性、仅凭有限二阶矩即可收敛到目标分布，并构造了能量单调不增的全局无碰撞粒子过程。 |
| [^30] | [From Common Information to Canonical Correlation Analysis and Beyond: Bridging Information Theory, Signal Processing, and Machine Learning](https://arxiv.org/abs/2610.28978) | 本教程首次将统计学中的典型相关分析、信息论中的两种公共信息概念以及工程领域的多视图子空间方法置于统一框架下，为无监督恢复多观测中公共分量的跨学科问题建立了桥梁，并厘清了不同理论之间的精确等价与类比关系。 |
| [^31] | [Is In-Context Reinforcement Learning A Posterior Sampler? An Empirical Study](https://arxiv.org/abs/2610.28977) | 本文通过建立贝叶斯后验采样器必须满足的两个必要条件——充分统计量不变性和鞅性质，并据此构造统计假设检验，实证考察了上下文强化学习（特别是DPT及其变体）在实践中是否真正表现为贝叶斯后验采样器。 |
| [^32] | [Auditing Conformal Prediction for Language Models Under Inference-Time Model Shift](https://arxiv.org/abs/2610.28937) | 该论文首次提出并形式化了“推理时模型偏移”现象——即部署与校准阶段推理配置（如量化模式和推理温度）的差异会引起保形分数分布变化并破坏分割保形可交换性，并通过8个语言模型、4个基准和12种推理配置的大规模实验系统评估了其对语言模型保形预测可靠性的影响。 |
| [^33] | [Backtracking Direct, Indirect and Total Effects](https://arxiv.org/abs/2610.28910) | 本文提出基于回溯反事实的因果效应新定义，无需干预处理变量或中介变量，而是通过回溯条件分布构造满足处理-中介约束的替代外生状态，从而得到回溯直接效应、间接效应和总效应。 |
| [^34] | [Actually, All Learnable Distribution Classes Are Privately Learnable](https://arxiv.org/abs/2610.28878) | 本文证明了所有在总变差距离下可学习的分布类在（ε, δ）-差分隐私下同样可学习，从而证明了 Ashtiani 的猜想。 |
| [^35] | [Depth Separation for Neural Networks under Gaussian Measure](https://arxiv.org/abs/2610.28800) | 该论文在高斯测度下建立了神经网络的深度分离结果：存在一个径向函数可被多项式规模的3层ReLU网络以小L²误差逼近，但2层网络要逼近它需要指数级规模，其核心工具是关于径向函数Hermite高频展开的不可逼近性定理，且全部证明均经Lean 4形式化验证。 |
| [^36] | [ReAPO: Retry-Aware Policy Optimization for Efficient Exploration in LLM Reasoning](https://arxiv.org/abs/2610.28770) | 提出基于随机最短路径框架的信用分配方法 ReAPO，通过为失败轨迹分配额外生成预算来实现重试感知的策略优化，从而提升大语言模型推理中探索的效率。 |
| [^37] | [A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners](https://arxiv.org/abs/2610.28747) | 该论文证明了一个“逆向戈德定理”：借助先天递归对比算子自我生成结构受限的负例扰动，并以InfoNCE式对比学习训练的受限评分器，在忠实性条件下可仅凭正数据实现高概率的PAC条件语言识别，且样本复杂度仅为多项式级。 |
| [^38] | [Borrowing Strength Across Stores: Transfer Learning for Choice-Based Revenue Management](https://arxiv.org/abs/2610.28708) | 该论文提出了一种面向MNL需求模型的收缩估计量，通过迁移学习在门店本地需求估计与跨门店信息共享之间取得平衡，从而在本地数据有限的情况下提升零售收益管理中品类与定价决策的可靠性。 |
| [^39] | [Privacy Accounting for Empirical Denoisers](https://arxiv.org/abs/2610.27710) | 本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。 |
| [^40] | [Hierarchical Gaussian Processes for Survey Weighting: Local Equivalent Weights in Structured Populations](https://arxiv.org/abs/2610.20321) | 本文提出基于层次高斯过程回归的调查加权新框架，通过结构化组级效应与核平滑在稀疏后分层单元格间共享信息，在降低计算成本的同时为非线性模型构建了局部等价权重并给出了理论收敛保证。 |
| [^41] | [Asymptotically Exact Negative Guidance of Diffusion Models via Positive-Unlabeled Learning](https://arxiv.org/abs/2610.19802) | 提出一种从显式目标密度采样的扩散模型负引导新方法，利用正-未标记学习在线估计样本属于不良条件的概率，从而实现渐近精确、偏差可通过增加计算量系统降低的负引导。 |
| [^42] | [Proximal Balancing for Causal Effect Estimation under Unmeasured Confounding](https://arxiv.org/abs/2609.40051) | 提出邻近平衡方法，将经典协变量平衡思想扩展到仅能通过代理变量观测的混杂因素，无需指定代理角色、求解逆问题或依赖潜变量模型，即可实现未测量混杂下的因果效应估计。 |
| [^43] | [Causal Discovery via Transformed Low-Rank Quantile Surfaces](https://arxiv.org/abs/2609.16931) | 本文提出低秩分位数曲面（LRQS）二元因果模型，证明在因果方向上变换后的条件分位数曲面具有低秩结构从而保证因果方向的通用可识别性，并提出了一种交替进行秩约束近似与单调变换保序估计的非参数因果发现方法。 |
| [^44] | [Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction](https://arxiv.org/abs/2608.23356) | 本文提出了一种分层指数-高斯混合模型，通过分层分解和正则化技术解决了现有EGMN模型的方差崩溃和组件冗余问题，从而提升了短视频推荐中观看时间预测的准确性。 |
| [^45] | [Learning reshapes power-law anisotropy in internal representations](https://arxiv.org/abs/2608.15239) | 该论文通过精确求解两层线性网络的学习动态，揭示了幂律各向异性在内部表示中的形成机制，并发现特征学习机制下指数呈现非单调且多阶段演化，而惰性机制下行为不同。 |
| [^46] | [Generalised Robust Bayes for Joint Inference of Model and Contamination](https://arxiv.org/abs/2607.25665) | 提出Hölder–Bayes方法，通过对缩放模型施加Hölder散度来构建模型参数与污染比例的联合广义后验，在无需指定离群点分布的情况下同时实现鲁棒参数推断与离群点检测，并提供了后验影响函数一致有界、有限样本n^{-1/2}风险界及Bernstein–von Mises定理等理论保证。 |
| [^47] | [To Retain or to Adapt? Generalizing Continual Learning](https://arxiv.org/abs/2607.05609) | 该论文挑战了持续学习以“保留知识”为中心的传统假设，将持续学习重新形式化为以最小化平均终身误差为目标的在线优化问题，并提出“迁移效率”作为衡量不稳定性与瞬态误差之间张力的定量指标。 |
| [^48] | [Sequential sparse Gaussian process quantile regression](https://arxiv.org/abs/2606.31284) | 本文提出一种序贯稀疏高斯过程分位数回归框架，通过诱导变量和拉普拉斯近似实现高效后验推断，并将预测不确定性分解为条件先验与后验诱发两个分量，以此驱动诱导输入填充和数据采集两种自适应机制，实现计算资源的动态分配。 |
| [^49] | [Gaussian Processes with Sample Paths in Reproducing Kernel Banach Spaces](https://arxiv.org/abs/2605.28106) | 该论文揭示了高斯过程与再生核巴拿赫空间中高斯随机元素之间的联系，证明此类空间上的协方差算子由对称正定函数唯一确定，并据此将Driscoll定理首次推广到了再生核巴拿赫空间。 |
| [^50] | [Interdomain Attention: Beyond Token-Level Key-Value Memory](https://arxiv.org/abs/2605.24330) | 提出跨域注意力（IA）这一新型混合架构，在同一层内将键值信息压缩到SSM参数化的循环记忆中并隐式重建注意力矩阵，从而在保持SSM计算效率的同时近似点积注意力。 |
| [^51] | [FRESH: Information-Geometric Calibration of Patient-Level Models to Aggregate Evidence](https://arxiv.org/abs/2605.16246) | FRESH提出了一种基于信息几何的校准方法，通过对患者级生成模型进行最小扰动，使其匹配人群级汇总证据（如已发表试验和登记数据），从而实现患者级数据与聚合数据的统一、高效建模以支持临床决策。 |
| [^52] | [The Sample Complexity of Multiple Change Point Identification under Bandit Feedback](https://arxiv.org/abs/2605.13252) | 该论文提出了一种在老虎机反馈下识别多个变化点的自适应算法，并证明了其样本预算的非渐近上下界，同时揭示在一般置信水平下样本复杂度不仅取决于跳跃幅度，先前仅由跳跃幅度决定复杂度的认识是不完整的。 |
| [^53] | [A Composite Activation Function for Learning Stable Binary Representations](https://arxiv.org/abs/2605.11558) | 提出了一种由sigmoid与tanh复合而成的重尾激活函数（HTAF），作为Heaviside函数的平滑近似，在零输入附近保持较大梯度且尾部梯度衰减更慢，从而实现了脉冲神经网络、二值神经网络和深度Heaviside神经网络的稳定梯度训练。 |
| [^54] | [Adaptive auditing of AI systems with anytime-valid guarantees](https://arxiv.org/abs/2605.07002) | 提出了一个具有任意时有效统计保证的假设检验框架，使高度自适应的AI系统审计——即使样本量小、采样与停止规则在过程中灵活决定——也能得出统计学上严谨的结论。 |
| [^55] | [Model-agnostic information transfer and fusion for classification with label noise](https://arxiv.org/abs/2604.25845) | 本文提出了一个通用的模型无关非参数信息迁移与融合框架，用于带标签噪声的分类问题，克服了现有统计迁移学习方法因噪声与干净数据间分布偏移及参数化模型限制所带来的不足，并可广泛应用于各类分类器。 |
| [^56] | [Generalization and Scaling Laws for Mixture-of-Experts Transformers](https://arxiv.org/abs/2604.09175) | 该论文为专家混合Transformer建立了泛化与缩放理论，证明其泛化误差由活跃参数预算主导并附加路由开销，且近似误差可通过扩展活跃容量或增加专家数量来降低，具体取决于主导瓶颈。 |
| [^57] | [Active Learning under Simulated Class Imbalance and Label Shift in Binary Semantic Segmentation](https://arxiv.org/abs/2601.06209) | 该论文的贡献不是新的主动学习采集策略，而是提出了一个基于两个开源数据集的可控、可复现的模拟协议，用于系统研究类别不平衡与标签偏移这两种不平衡形式对二元语义分割中主动学习算法的影响。 |
| [^58] | [Covariate-assisted graph matching](https://arxiv.org/abs/2512.11761) | 本文提出两种利用节点和边协变量信息的带种子图匹配方法——基于二次分配问题的全局方法和利用种子邻域的可扩展局部方法，显著提升了图匹配在结构噪声和不可识别匹配情形下的灵活性与鲁棒性。 |
| [^59] | [Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](https://arxiv.org/abs/2510.16511) | 提出了OracleAD，一个简单可解释的无监督多元时间序列异常检测框架，它将每个变量的时间动态编码为因果嵌入，通过自注意力机制建模动态空间关系，并与代表正常状态的稳定潜在结构对齐以检测异常。 |
| [^60] | [Memorizing Long-tail Data Can Help Generalization Through Composition](https://arxiv.org/abs/2510.16322) | 记忆长尾样本与特征组合能力相结合，可以使模型对训练中从未出现过的长尾特征组合的稀有测试样本做出正确预测，且这种组合能力依赖于模型架构。 |
| [^61] | [A Computable Measure of Suboptimality for Entropy-Regularised Variational Objectives](https://arxiv.org/abs/2509.10393) | 本文提出可显式计算的核梯度差异（KGD）作为熵正则化变分目标次优性的度量，在贝叶斯情形下与核 Stein 差异（KSD）一致，从而使研究者在无法获得未归一化密度的情况下也能开发和比较新型采样算法。 |
| [^62] | [Can synthetic data reproduce real-world findings in epidemiology? A replication study using adversarial random forests](https://arxiv.org/abs/2508.14936) | 该研究提出对抗性随机森林（ARF）作为合成表格型流行病学数据的高效便捷方法，并通过复制六项流行病学研究的统计分析，验证了合成数据能在保护隐私的同时可靠重现真实世界的研究发现。 |
| [^63] | [The dynamic generalized covariance measure for conditional independence testing with nonstationary time series](https://arxiv.org/abs/2504.21647) | 本文提出了首个基于动态广义协方差度量的条件独立性检验框架，可对非平稳非线性时间序列的单次观测数据进行检验，能够检测出期望条件协方差在某些时刻非零的依赖关系。 |
| [^64] | [Exploiting Concavity Information in Gaussian Process Contextual Bandit Optimization](https://arxiv.org/abs/2503.10836) | 本文提出一种将保持凹性的回归样条基与受凹性约束的高斯过程后验相结合的新型奖励模型，并据此构建UCB算法，从而在上下文老虎机优化中利用已知的凹性结构提升序贯决策的优化效率。 |
| [^65] | [Optimized Conformal Selection: Powerful Selective Inference After Data-Dependent Score Construction](https://arxiv.org/abs/2411.17983) | 该论文提出了优化共形选择（OptCS）通用框架，突破了现有方法要求评分函数独立于校准和测试数据的限制，在数据依赖的评分构建和大量数据复用的情况下仍能保证有限样本的错误发现率（FDR）控制。 |
| [^66] | [Multi-Label Sentiment Analysis on 100 Languages with Dynamic Weighting for Label Imbalance](https://arxiv.org/abs/2008.11573) | 该论文提出了一种动态加权方法来应对多标签情感分析中的标签不平衡问题，并结合适配的多标签焦点损失与线性时间复杂度的最优阈值选择方法，在100种语言的情感分析任务中于9个指标中的7个上达到了最先进水平。 |
| [^67] | [The Limits of Post-Selection Generalization](https://arxiv.org/abs/1806.06100) | 本文揭示了事后泛化算法的能力极限，证明了任何满足事后泛化并回答自适应选择统计查询的算法其误差都存在紧的下界，为选择后数据分析的进展设置了一个根本性障碍。 |

# 详细

[^1]: 面向神经网络的几何代数：综述

    Geometric Algebra for Neural Networks: A Survey

    [https://arxiv.org/abs/2610.30121](https://arxiv.org/abs/2610.30121)

    本综述系统梳理了几何代数神经网络的基本构件、等变性、应用与实证表现，指出基于几何代数的架构在具有显式几何对称性和异质几何实体的场景中优势最为显著。

    

    几何代数（GA）在单一的代数框架中统一了几何基元和几何变换。围绕该框架设计的神经网络通常将特征表示为多重向量（multivectors），并约束层运算以遵循代数结构，从而能够对几何变换进行有原则的处理，且在许多情况下内置等变性。本综述回顾了几何代数神经网络，包括其基本构件、等变性质、应用领域、实证性能和软件支持。我们提供了代数选择指南、基于GA的架构与其他等变架构的实证比较，以及涵盖训练病态、部署约束和性能局限的失效模式分析。纵观所调研的文献，基于GA的架构最有力的证据出现在具有显式几何对称性和异质几何实体的场景中；证据……

    arXiv:2610.30121v1 Announce Type: cross  Abstract: Geometric algebra (GA) unifies geometric primitives and transformations in a single algebraic framework. Neural networks designed around this framework often represent features as multivectors and constrain layer operations to respect the algebraic structure, enabling principled treatment of geometric transformations and, in many cases, built-in equivariance. This survey reviews geometric algebra neural networks, including their building blocks, equivariant properties, application domains, empirical performance, and software support. We provide an algebra-selection guide, an empirical comparison of GA-based and other equivariant architectures, and a failure-mode analysis covering training pathologies, deployment constraints, and performance limitations. Across the surveyed literature, the strongest evidence for GA-based architectures appears in settings with explicit geometric symmetries and heterogeneous geometric entities; evidence f
    
[^2]: 局部性与权重共享在卷积神经网络中的作用

    The role of locality and weight sharing in convolutional neural networks

    [https://arxiv.org/abs/2610.30118](https://arxiv.org/abs/2610.30118)

    本研究通过师生模型的精确动力学分析揭示：卷积网络中权重共享带来的学习加速与输入位置数 $P$ 成正比，而局部性则决定样本复杂度，二者的影响均依赖于输入位置间的空间相关性。

    

    卷积神经网络的成功通常被归因于局部性和权重共享。然而，这两个特性如何分别塑造学习过程，以及它们的影响如何依赖于自然图像的空间相关性，仍然知之甚少。我们在一个师生模型中研究这些问题，其中一个卷积教师作用于 $P$ 个输入位置，位置之间的相关性为 $c$。我们比较了通过在线随机梯度下降训练的卷积学生与局部连接学生，并推导出在大特征维度 $D$ 极限下能精确描述其动力学的常微分方程。对于不相关的位置（$c = 0$），两个学生遵循相同的学习轨迹，而卷积学生的学习速度恰好快 $P$ 倍。我们还发现局部性会影响样本复杂度：与教师核大小相匹配的学生能够从 $\mathcal{O}_{D}(D)$ 个样本中恢复教师……

    arXiv:2610.30118v1 Announce Type: new  Abstract: The success of convolutional neural networks is often attributed to locality and weight sharing. However, how these two properties separately shape learning, and how their effects depend on the spatial correlations of natural images, remain poorly understood. We address these questions in a teacher-student model with a convolutional teacher acting on $P$ input positions with correlation $c$ between positions. We compare convolutional and locally connected students trained by online stochastic gradient descent, and derive ordinary differential equations that describe their dynamics exactly in the limit of large feature dimension $D$. For uncorrelated positions ($c = 0$), the two students follow the same learning trajectory, with the convolutional student learning exactly $P$ times faster. We also find that locality affects the sample complexity: a student that matches the kernel size of the teacher recovers it from $\mathcal{O}_{D}(D)$ sa
    
[^3]: 通过梦境学习缓解模型坍塌

    Mitigating Model Collapse through Dreaming Learning

    [https://arxiv.org/abs/2610.30009](https://arxiv.org/abs/2610.30009)

    该论文从谱分析角度揭示低温采样（Ts < 1）会使转移算子锐化、谱隙收窄从而导致模型坍塌，并提出“梦境学习”正则化方法——在训练中对模型以高温生成的自身样本施加额外损失项——来缓解这一退化。

    

    在合成数据上训练生成模型会导致多样性丧失和性能下降，这一现象被称为模型坍塌。我们在受控的马尔可夫环境中从谱分析的角度研究这种退化。学习到的转移算子的特征值谱为该过程提供了直接的观察视角，因为次主导特征值决定了马尔可夫链的混合速度，进而决定了模型对状态空间的持续探索程度。我们发现，在温度 Ts < 1 下采样对转移算子的作用类似于反复锐化，会将其推向一个确定性系统：次主导特征值趋近于单位圆，谱隙收窄。当 Ts < 1 时，即使是数据累积似乎也无法阻止这种漂移。随后我们分析了梦境学习，这是一种正则化方法，它在常规训练的同时，对模型以较高温度 Td > 1 采样生成的自身样本添加一个损失项。由于高温样本会平滑……

    arXiv:2610.30009v1 Announce Type: new  Abstract: Training generative models on synthetic data causes loss of diversity and a decrease in performance, a phenomenon known as model collapse. We study this degeneration spectrally in a controlled Markov setting. The eigenvalue spectrum of the learned transition operator offers a direct view of the process, since the subdominant eigenvalues set how fast the chain mixes and therefore how much of the state space it continues to explore. We find that sampling at temperature $T_s < 1$ acts on the operator like repeated sharpening, pushing it toward a deterministic system: the subdominant eigenvalues approach the unit circle and the spectral gap narrows. When $T_s < 1$, even data accumulation does not appear to arrest this drift. We then analyze Dreaming Learning, a regulariser that, in parallel with regular training, adds a loss term on the model's own samples drawn at a raised temperature $T_d > 1$. Because high-temperature samples flatten the 
    
[^4]: 非平稳多臂老虎机中现时效应推断的代价

    The Price of Current-Effect Inference in Nonstationary Multi-Armed Bandits

    [https://arxiv.org/abs/2610.29982](https://arxiv.org/abs/2610.29982)

    该论文揭示了非平稳多臂老虎机中收益最大化与现时效应推断之间的内在权衡，证明了以有限宽度置信区间同时覆盖所有现时效应所需付出的动态遗憾极小极大代价，并提出 TRACE 算法来同时兼顾这两个目标。

    

    自适应实验利用老虎机算法来偏好具有更高期望收益的干预措施。然而，决策者还需要对现时处理效应进行可靠的推断：即在当前时刻干预措施与参照臂平均收益之间的差异，而非历史平均。这两个目标相互冲突：收益最大化倾向于选择更优的干预，而现时效应推断则要求随着结果随时间变化，持续探索可能次优的干预。我们针对平均收益具有指数 β∈(0,1] 和常数 L 的 Hölder 平滑性的 K 臂老虎机刻画了这一权衡。在要求置信区间以不超过 w 的宽度、以高概率同时覆盖所有现时效应的条件下，我们在期望动态遗憾上建立了（在对数因子范围内）匹配的极小极大界。在时间跨度 T 内，我们的算法 TRACE（Tracking Rewards and Current Effects）实现了遗憾……（原文摘要至此截断）

    arXiv:2610.29982v1 Announce Type: cross  Abstract: Adaptive experiments use bandit algorithms to favor interventions with higher expected rewards. However, decision-makers also require reliable inference on current treatment effects: differences between intervention and reference-arm mean rewards at the present time, rather than historical averages. These objectives conflict: reward maximization favors better interventions, whereas current-effect inference requires continued exploration of potentially suboptimal interventions as outcomes change over time. We characterize this tradeoff for $K$-arm bandits with H\"older-smooth mean rewards of exponent $\beta\in(0,1]$ and constant $L$. Requiring high-probability simultaneous coverage of all current effects by confidence intervals of width at most $w$, we establish matching minimax bounds, up to logarithmic factors, on expected dynamic regret. Over horizon $T$, our algorithm TRACE (Tracking Rewards and Current Effects) achieves regret of o
    
[^5]: 掌握模型拟合：InterModel Vigorish（模型间赌利）在可解释机器学习中的应用

    Mastering Model Fit: Applications of the InterModel Vigorish for Interpretable Machine Learning

    [https://arxiv.org/abs/2610.29960](https://arxiv.org/abs/2610.29960)

    本文将“模型间赌利（IMV）”引入机器学习领域，通过 SHAP-IMV、Multi-IMV 和 Ablate-IMV 等可解释工具，将模型拟合差异量化为相对于参考模型的期望投注优势，从而揭示哪些特征、模型组件和训练阶段真正改善了概率拟合。

    

    arXiv:2610.29960v1 公告类型：交叉。仅有准确率几乎无法说明哪些特征或模型组件改善了概率拟合、哪些结果从中受益，以及拟合在训练过程中如何变化。我们将模型间赌利（InterModel Vigorish, IMV）引入机器智能领域，以将拟合差异表达为相对于参考模型的期望投注优势。SHAP-IMV 通过 Shapley 加权的子集比较将增益归因于各个特征，Multi-IMV 评估一对多及成对的类别判别能力，Ablate-IMV 则通过消融评估架构组件和特征组。我们使用表格数据、图像、传感器序列和图数据，在十个随机种子上对这些应用进行了评估。结果显示，特征排序在不同估计器之间存在差异；一对多准确率很高时，相对于零基准的改进却可能微不足道；而在固定训练预算下，消融可以保持甚至改善拟合。在六个图分类基准上，我们对 IMV 进行了超过 200 个训练（周期）的时序剖析……

    arXiv:2610.29960v1 Announce Type: cross  Abstract: Accuracy alone gives little information about which features or model components improve probabilistic fit, which outcomes benefit, or how fit changes during training. We introduce the InterModel Vigorish (IMV) into the Machine Intelligence space in order to express differences in fit as an expected betting advantage over a reference model. SHAP-IMV attributes gains to features through Shapley-weighted subset comparisons, Multi-IMV evaluates one-vs-rest and pairwise class discrimination, and Ablate-IMV assesses architectural components and feature groups by ablation. We evaluate these applications over ten random seeds using tabular data, images, sensor sequences, and graphs. Feature rankings vary across estimators. High one-vs-rest accuracy can accompany negligible improvement over a null, while ablation can preserve or improve fit under a fixed training budget. On six graph-classification benchmarks, we time-profile IMV over 200 trai
    
[^6]: 双层线性网络中的静默对齐：高维视角

    Silent Alignment in Two-Layer Linear Networks: A High-Dimensional Perspective

    [https://arxiv.org/abs/2610.29864](https://arxiv.org/abs/2610.29864)

    该论文在高维比例极限下借助随机矩阵理论精确刻画了双层线性网络梯度流的谱演化，并发现小初始化时存在“静默对齐”现象——网络奇异方向在一阶时间尺度上先完成对齐而损失保持不变，可见的学习随后才在对数时间尺度上以鞍点到鞍点的方式发生。

    

    我们研究了双层线性网络在高维情形下的梯度流，其中输入维度、隐藏层内部维度与输出维度成比例地发散。在白化输入和各向同性失衡的条件下，我们将动力学的闭式解与已有的随机矩阵理论结果相结合，推导出该动力学的高维极限。我们的方法给出了权重矩阵在任意时刻的精确渐近谱分布，并刻画了所学子空间的演化。在小初始化极限下，我们识别出一种“静默对齐”现象：网络的奇异方向在一阶时间尺度上完成对齐，而奇异值和经验损失却保持在其初始值不变。可见的学习随后才在（对数）时间尺度上发生，此时奇异值逐渐脱离零点，损失随之下降。这一阶段以“鞍点到鞍点”的动力学为特征，该动力学进而转变为一个连续的过程……

    arXiv:2610.29864v1 Announce Type: cross  Abstract: We study gradient flow in a two-layer linear network in the high-dimensional regime where the input, inner and output dimensions diverge proportionally. Under whitened inputs and isotropic imbalance, we combine a closed-form solution of the dynamics with prior random matrix theory results to derive a high-dimensional limit for the dynamics. Our method yields exact asymptotic spectral distributions for the weight matrices at all times and characterizes the evolution of the learned subspace. In the small-initialization limit, we identify a silent alignment phenomenon: the singular directions of the network align on an order-one timescale, while the singular values and empirical loss remain at their initial values. Visible learning occurs later, on a logarithmic timescale, as singular values progressively escape zero and the loss decreases. This phase is characterized by a saddle-to-saddle dynamics, which becomes a continuous process of s
    
[^7]: DiT如何去噪？从聚合器-预测器视角理解注意力与MLP

    How Do DiTs Denoise? An Aggregator-Predictor View of Attention and MLPs

    [https://arxiv.org/abs/2610.29854](https://arxiv.org/abs/2610.29854)

    论文提出将扩散Transformer视为“聚合器-预测器”架构的新解释框架——AdaLN调制的注意力充当依赖输入与噪声的上下文聚合器，MLP充当可学习键-值表示的patch预测器，并从理论上证明基于局部上下文的聚合能以指数衰减的误差逼近全局贝叶斯去噪器。

    

    扩散Transformer（DiT）如何表示图像内容，并学习复用这些内容的规则以实现泛化？我们提出了一种“聚合器-预测器”视角，将DiT视为输入自适应的局部去噪器，其中上下文聚合与patch预测为理解注意力和MLP提供了功能性角色。我们首先证明，对于高斯马尔可夫随机场（GMRF）先验，全局贝叶斯去噪器可以仅使用上下文patch的一个子集来近似，且误差随上下文半径呈指数衰减。我们进一步表明，常被用于解释扩散模型泛化能力的逐patch线性去噪器，在聚合信息足够充分时能够恢复经验局部去噪器。在这一视角下，经AdaLN调制的注意力自然地参数化了一个依赖输入和噪声的聚合器，而MLP则提供了一个可学习的键-值表示，用于逐token的patch预测。我们的实验支持这一观点。在真实数据上训练的DiT……

    arXiv:2610.29854v1 Announce Type: cross  Abstract: How do diffusion transformers (DiTs) represent image content and learn rules for reusing it to generalize? We develop an aggregator--predictor view of DiTs as input-adaptive local denoisers, in which context aggregation and patch prediction provide functional roles for interpreting attention and MLPs. We first establish that, for Gaussian Markov random field (GMRF) priors, the global Bayes denoiser can be approximated using only a subset of context patches, with error decaying exponentially in context radius. We further show that patchwise linear denoisers, often invoked to explain diffusion generalization, recover the empirical local denoiser with sufficiently informative aggregation. Within this view, AdaLN-modulated attention naturally parameterizes an input- and noise-dependent aggregator, while the MLP admits a learned key--value representation for tokenwise patch prediction. Our experiments support this view. DiTs trained on real
    
[^8]: 非对称无似然假设检验

    Asymmetric Likelihood-Free Hypothesis Testing

    [https://arxiv.org/abs/2610.29798](https://arxiv.org/abs/2610.29798)

    本文刻画了非对称无似然假设检验中可行的样本预算三元组（至多相差对数因子），证明对参考分布 P 和 Q 分配不等的样本预算可带来多项式级别的节省——例如将来自 D 的所需样本量从 Ω(√n) 降至 Õ(n^(1/4))——并将其应用于非自适应分布 2-聚类问题。

    

    在无似然假设检验（LFHT）中，给定来自定义在 [n] 上、且满足全变差距离 d_TV(P,Q)≥ε 的两个未知分布 P 和 Q 的样本，以及来自第三个分布 D∈{P,Q} 的样本，目标是判别 D=P 还是 D=Q。我们研究了非对称无似然假设检验，其中 P 和 Q 的样本预算 s_P 和 s_Q 可以不同，并在对数因子范围内刻画了可行的三元组 (s_P, s_Q, s_D)。我们证明了对参考分布采用不相等的样本预算可以带来多项式级别的节省。例如，在常数分离度下，从 P 采集 n 个样本、从 Q 采集 n^(3/4) 个样本，即可仅用 Õ(n^(1/4)) 个来自 D 的样本完成检验，而现有的两类基线方法——对称无似然假设检验与非等样本接近性检验——都需要来自 D 的 Ω(√n) 个样本。我们进一步将结果应用于非自适应分布 2-聚类问题，该问题中 k 个分布之间的样本分配必须在观测之前固定。

    arXiv:2610.29798v1 Announce Type: new  Abstract: In likelihood-free hypothesis testing (LFHT), given samples from two unknown distributions $P,Q$ on $[n]$ with $\mathrm{d_{TV}}(P,Q)\geq\varepsilon$, and from a third distribution $D\in\{P,Q\}$, the goal is to identify whether $D=P$ or $D=Q$. We study asymmetric likelihood-free hypothesis testing, where the sample budgets $s_P$ and $s_Q$ for $P$ and $Q$ may differ, and characterize the feasible triples $(s_P,s_Q,s_D)$ up to logarithmic factors. We establish that unequal reference budgets can give a polynomial saving. For instance, for constant separation, $n$ samples from $P$ and $n^{3/4}$ from $Q$ allow testing with $\widetilde O(n^{1/4})$ samples from $D$, whereas both existing baselines---symmetric LFHT and unequal-sample closeness testing---require $\Omega(\sqrt n)$ samples from $D$.   We further apply our results to non-adaptive distribution 2-clustering, where the sample allocation among $k$ distributions must be fixed before obser
    
[^9]: 视觉-语言-动作策略的后验认知不确定性估计

    Post-hoc Epistemic Uncertainty for Vision-Language-Action Policies

    [https://arxiv.org/abs/2610.29730](https://arxiv.org/abs/2610.29730)

    提出了一种后验核方差头方法，无需训练多个策略或修改原策略，即可从单一VLA策略的嵌入中估计认知不确定性，其表现与双策略集成方法相当。

    

    视觉-语言-动作（VLA）策略需要识别其训练知识中的空白，以便在新情况下请求有用的示范或协助，尤其是在失败代价高昂的场景中。独立训练的策略集合内部的分歧可以指导这类决策，但这需要训练和运行多个策略。我们提出了后验核方差头，能够从单一策略的嵌入中估计认知不确定性。每个头计算以训练示范的嵌入为条件的高斯过程条件方差。该方法的拟合不需要回归目标或梯度更新，且不改变原有策略。在SmolVLA于LIBERO-10上的实验中，核方差头在与任务成功的任务级相关性、主动学习期间的示范获取、以及向更强监督策略转介等方面，与双策略集成分数相比表现具有竞争力。

    arXiv:2610.29730v1 Announce Type: cross  Abstract: Vision-language-action (VLA) policies need to recognize gaps in their training knowledge to request useful demonstrations or assistance in new situations, especially when failures are costly. Disagreement within an ensemble of independently trained policies can guide these decisions, but requires training and running multiple policies. We propose post-hoc kernel-variance heads that estimate epistemic uncertainty from the embeddings of a single policy. Each head computes the conditional variance of a Gaussian process conditioned on embeddings from the training demonstrations. Fitting requires no regression targets or gradient updates and leaves the policy unchanged. In experiments with SmolVLA on LIBERO-10, the kernel variance heads perform competitively with two-policy ensemble scores in task-level correlation with task success, in demonstration acquisition during active learning, and in referral to a stronger supervisor policy as a pr
    
[^10]: 沃瑟斯坦空间中的PCA提升

    Lifting PCA in the Wasserstein Space

    [https://arxiv.org/abs/2610.29679](https://arxiv.org/abs/2610.29679)

    本文提出了“提升PCA”，通过凸延拓提升泛函的框架，首次将欧氏主成分分析规范化地扩展到Wasserstein空间，并将其归结为两个等价的、具有广义代价的Wasserstein重心最小化问题。

    

    主成分分析（PCA）是统计降维的基本工具，然而在最优传输的框架下，针对分布值数据的规范化扩展一直未曾出现。虽然Wasserstein重心和Wasserstein最小二乘回归分别为欧氏平均和线性回归在Wasserstein空间中提供了规范化的提升，但PCA的类似原理却始终未被探索。本文提出了提升PCA（Lifted PCA），这是欧氏PCA在Wasserstein空间中的一种规范化扩展，其通过凸延拓提升概率测度空间中泛函的框架获得。从Stiefel流形和Grassmann流形上PCA的经典变分形式出发，我们推导出两个等价的最小化问题，这些问题定义在支撑于上述流形上的概率测度之上，并对应于具有广义代价的Wasserstein重心。我们建立了（摘要在此处截断）

    arXiv:2610.29679v1 Announce Type: cross  Abstract: Principal Component Analysis (PCA) is a fundamental tool for statistical dimension reduction, yet no canonical extension to distribution-valued data has emerged within the framework of optimal transport. While Wasserstein barycenters and Wasserstein least-squares regression provide canonical liftings of Euclidean averaging and linear regression in the Wasserstein space, an analogous principle for PCA has remained unexplored. In this paper, we introduce Lifted PCA, a canonical extension of Euclidean PCA to Wasserstein space, obtained through the framework of lifting functionals in the space of probability measures by convex extension. Starting from the classical variational formulations of PCA on the Stiefel and Grassmann manifolds, we derive two equivalent minimization problems over probability measures supported on these manifolds that correspond to Wasserstein barycenters with generalized costs. We establish an exact analogue of the 
    
[^11]: 分位数分布强化学习的近极小极大速率

    Near Minimax Rate for Quantile Distributional Reinforcement Learning

    [https://arxiv.org/abs/2610.29595](https://arxiv.org/abs/2610.29595)

    本文在生成模型假设下，为分位数Bellman不动点估计建立了在所有分辨率和所有Wasserstein阶数上一致成立的 $\sqrt{n}$ 高概率误差界，达到了分位数分布策略评估的近极小极大最优速率。

    

    本文研究分位数分布策略评估问题，其中给定策略下的回报分布由分位数Bellman不动点 $\bm\eta_m$ 来近似。我们在Wasserstein距离下，研究了跨不同分位数分辨率估计 $\bm\eta_m$ 的统计精度。假设可以访问生成模型，我们从经验马尔可夫决策过程构建了分位数不动点估计量 $\bm\eta_m^{(n)}$，并建立了Wasserstein距离下的非渐近误差界。当奖励密度一致下有界（即一致远离零）时，我们获得了在所有分辨率 $m$ 和所有Wasserstein阶数 $1\le p\le\infty$ 上同时成立的 $\sqrt{n}$ 阶高概率界。当奖励密度在支撑集端点处趋近于零时，我们证明仅需连续性和端点单调性即可在 $1\le p\le2$ 时保持相同的与分辨率无关的速率。对于 $p>2$，我们建立了带有系数的 $\sqrt{n}$ 阶界……

    arXiv:2610.29595v1 Announce Type: cross  Abstract: In this paper, we study quantile distributional policy evaluation, where the return distribution under a given policy is approximated by the quantile Bellman fixed point $\bm\eta_m$. We investigate the statistical accuracy of estimating $\bm\eta_m$ across quantile resolutions under Wasserstein distances. Assuming access to a generative model, we construct a quantile fixed point estimator $\bm\eta_m^{(n)}$ from an empirical Markov decision process and establish non-asymptotic error bounds under Wasserstein distances. When reward densities are uniformly bounded away from zero, we obtain a root-$n$ high-probability bound simultaneously over all resolutions $m$ and all Wasserstein orders $1\le p\le\infty$. When reward densities vanish at the support endpoints, we show that continuity and endpoint monotonicity suffice to preserve the same resolution-independent rate for $1\le p\le2$. For $p>2$, we establish a root-$n$ bound with coefficient
    
[^12]: 基于片段级图编码的可解释分子性质预测

    Fragment-Level Graph Encoding for Interpretable Molecular Property Prediction

    [https://arxiv.org/abs/2610.29589](https://arxiv.org/abs/2610.29589)

    该论文提出用BRICS分解将分子编码为片段级图取代原子级图，在不改变模型结构的前提下使GNN能够给出功能团级别的可解释预测，并在约五分之一大小的图上达到与原子级编码相当的预测精度。

    

    图神经网络（GNN）是分子性质预测的标准工具之一，但其预测的原子级解释对药物化学家而言难以直接采用。我们提出改变输入表示而非设计新模型：将分子通过BRICS分解编码为片段级图，其中每个节点是一个具有合成意义的片段，其特征为其原子特征的逐元素求和。由于仅改变了表示方式，任何消息传递GNN和任何事后归因方法都能在功能团分辨率而非单个原子的层面上产生解释。在骨架划分下，跨越四种架构、四个预测任务和两种解释器的实验表明，片段级编码在约五分之一大小的图上达到了与原子级相同的测试精度。积分梯度归因在两个酶靶点上以高于随机水平的准确率恢复了已知药效团，其中在BACE上达到1.5倍的……

    arXiv:2610.29589v1 Announce Type: new  Abstract: Graph neural networks (GNNs) are among the standard tools for molecular property prediction, but the atom-level explanations of their predictions are hard for medicinal chemists to act on. We propose a change of input representation rather than a new model: molecules are encoded as fragment-level graphs from BRICS decomposition, where each node is a synthetically meaningful fragment whose features are the element-wise sum of its atoms' features. Because only the representation changes, any message-passing GNN and any post-hoc attribution method produces explanations at the resolution of functional groups instead of single atoms. Across four architectures, four prediction tasks, and two explainers under scaffold splits, fragment-level encoding matches atom-level test accuracy on roughly five-fold smaller graphs. Integrated-Gradients attributions recover known pharmacophores above chance on two enzyme targets, namely BACE at 1.5 times the 
    
[^13]: 小损失下的差分隐私专家在线预测

    Private Online Prediction from Experts with Small Losses

    [https://arxiv.org/abs/2610.29575](https://arxiv.org/abs/2610.29575)

    该论文提出了首个遗憾度达到 $\tilde{O}(\sqrt{L^\star})$ 的差分隐私专家在线预测算法，在低损失情形下显著优于现有 $\tilde{O}(\sqrt{T})$ 界限，并证明在忘却输入下隐私代价很小。

    

    我们研究机器学习中的一个基本问题——差分隐私约束下的专家在线预测。现有的隐私保护算法实现了接近最优的遗憾度 $\tilde{O}(\sqrt{T})$，其中 $T$ 为时间范围。然而，当最优专家的累计损失 $L^\star$ 远小于 $T$ 时，该界限变得次优。在这项工作中，我们提出了首个遗憾度为 $\tilde{O}(\sqrt{L^\star})$ 的差分隐私算法，在低损失情形下带来了显著的改进。对于常数隐私参数，我们的遗憾度界限在多对数因子范围内与非隐私下界相匹配，表明在此设置下隐私只需付出很小的代价。我们的结果假设输入序列是忘却的，即预先固定且独立于算法的输出。我们还证明了对于自适应生成的输入，$\tilde{\Omega}(\sqrt{T})$ 的遗憾度是不可避免的。

    arXiv:2610.29575v1 Announce Type: new  Abstract: We consider online prediction from experts, a fundamental problem in machine learning, under differential privacy constraints. Existing private algorithms achieve near-optimal regret $\tilde{O}(\sqrt{T})$, where $T$ is the time horizon. However, this bound becomes suboptimal when $L^\star$, the cumulative loss of the best expert, is significantly less than $T$. In this work, we present the first differentially private algorithm with regret $\tilde{O}(\sqrt{L^\star})$, offering a substantial improvement in the low-loss regime. Our regret bound matches the non-private lower bound up to poly-logarithmic factors for constant privacy parameters, demonstrating that privacy incurs only a small cost in this setting. Our result assumes that the input sequence is oblivious, i.e., it is fixed in advance and independent of the algorithm's outputs. We also prove that an $\tilde{\Omega}(\sqrt{T})$ regret is inevitable for adaptively generated inputs, 
    
[^14]: 基于预训练上下文分位数回归的残差因果发现

    Residual-Based Causal Discovery via Pretrained In-Context Quantile Regression

    [https://arxiv.org/abs/2610.29530](https://arxiv.org/abs/2610.29530)

    该论文提出Loscar方法，将RESIT框架从加性噪声模型扩展至更广泛的位置-尺度噪声模型，通过分位数回归非参数化地估计条件均值和尺度，并支持将预训练的表格基础模型作为即插即用估计器，从而避免了现有方法对参数化假设的依赖。

    

    基于残差的因果发现假设每个变量由其父节点和一个独立的噪声源共同生成。该方法通过估计这一噪声并检验其与候选原因之间的独立性来推断因果关系。对于可识别的函数类别，如加性噪声模型（ANM）及更广泛的位置-尺度噪声模型（LSNM），该原理能够识别因果方向，因为只有正确的因果模型才满足这种独立性。现有的LSNM方法依赖参数化假设来获得标准化残差，存在模型误设的风险。相比之下，RESIT框架允许对残差进行非参数估计，从而迭代地识别汇节点，但该框架最初是为ANM提出的。为弥合这一差距，我们提出了Loscar，这是RESIT在LSNM上的扩展方法，通过分位数回归估计条件均值和条件尺度，使预训练的表格基础模型能够作为即插即用的估计器。我们将其与可行的广义最小二乘法进行对比……

    arXiv:2610.29530v1 Announce Type: cross  Abstract: Residual-based causal discovery assumes that each variable is generated from its parents and an independent noise source. It estimates this noise and tests its independence from candidate causes. For identifiable functions such as additive-noise models (ANMs) and the broader class of location-scale noise models (LSNMs), this principle identifies the causal direction, as only the causal model admits this independence. Existing LSNM methods rely on parametric assumptions to obtain standardized residuals, risking model misspecification. In contrast, the RESIT-framework allows for nonparametric estimation of residuals to iteratively identify sink nodes, but was proposed for ANMs. To close this gap, we propose Loscar, an extension of RESIT to LSNMs that estimates the conditional mean and scale via quantile regression, enabling pretrained tabular foundation models as plug-in estimators. We compare it with a feasible generalized least squares
    
[^15]: BRIDGE：一种阻止模型更新中虚假改进的接受门控，附回顾性临床评估

    BRIDGE: An Acceptance Gate to Block False Improvements in Model Updates, with a Retrospective Clinical Evaluation

    [https://arxiv.org/abs/2610.29515](https://arxiv.org/abs/2610.29515)

    BRIDGE 提出了一种模型无关的接受门控机制，通过交叉拟合与 bootstrap 置信下界检验来拦截模型更新中的虚假改进，将虚假改进率从约 20% 降至接近零，并经 12 项回顾性临床任务验证有效。

    

    模型会被反复更新，而在验证数据上看似更好但在新数据上并非如此的更新是一种“虚假改进”，可能误导临床决策。朴素的规则会接受每一个正向的验证增益，而当同一批数据既用于选择又用于评判更新时，结果会过于乐观。BRIDGE 是一个模型无关的接受门控，它基于预测结果和可用标签、以固定的设置来决定是否接受每次更新。数据的每一折评判在其余折上选出的更新（交叉拟合），以减少选择偏差；接受条件要求增益的 bootstrap 置信下界超过一个裕度。实验涵盖四种设置：(A) 11 个新数据集，(B) 合成任务，(C) 12 项回顾性临床任务，以及 (D) 39 个支持性数据集。虚假改进率从朴素方法的 19.1% 降至 BRIDGE 的 1.1% (A)，从 20.7% 降至 1.7% (B)，从 20.0% 降至 0% (C)，从 12.5% 降至 0.5% (D)；所有关于数据集或任务上降低幅度的 95% 置信区间均不包含零。

    arXiv:2610.29515v1 Announce Type: new  Abstract: Models are updated repeatedly, and an update that looks better on validation data but not on new data is a false improvement that can misguide clinical decisions. The naive rule accepts every positive validation gain and is optimistic when the same data choose and judge the update. BRIDGE is a model-agnostic acceptance gate that decides each update from predictions and the available labels, with fixed settings. Each fold of the data judges the update chosen on the other folds (cross-fitting) to reduce selection bias; acceptance requires the gain's lower bootstrap bound to exceed a margin. The four settings are (A) 11 new datasets, (B) synthetic tasks, (C) 12 retrospective clinical tasks, and (D) 39 supporting datasets. The false-improvement rate falls from 19.1% (naive) to 1.1% (BRIDGE) in (A), 20.7% to 1.7% (B), 20.0% to 0% (C), and 12.5% to 0.5% (D); every 95% confidence interval of the reduction over datasets or tasks excludes zero (n
    
[^16]: 基于Stein算子的非归一化模型估计的统一方法

    A unified approach for estimation of unnormalized models via Stein operators

    [https://arxiv.org/abs/2610.29510](https://arxiv.org/abs/2610.29510)

    本文建立了一个连接适当评分规则与Stein方法的统一框架，证明分数匹配、比率匹配、伪似然和噪声对比估计等评分规则可自然诱导Stein算子，并借助由此产生的几何视角统一刻画了相应估计量的渐近效率。

    

    本文开发了一个统一的框架，将适当评分规则与Stein方法相联系，用于非归一化模型的估计。我们构建了一类适当评分规则，它涵盖了分数匹配、比率匹配、伪似然和噪声对比估计。我们证明了这些适当评分规则会自然地诱导出Stein算子。这一联系为Stein算子的构造提供了新的原理，并使我们能够通过所诱导的Stein算子相关的几何结构来研究基于这些评分规则的估计量。利用这一几何视角，我们对所得估计量的渐近效率给出了统一的刻画。

    arXiv:2610.29510v1 Announce Type: cross  Abstract: In this paper, we develop a unified framework connecting proper scoring rules and Stein's method for the estimation of unnormalized models. We construct a class of proper scoring rules that encompasses score matching, ratio matching, pseudolikelihood, and noise contrastive estimation. We show that these proper scoring rules naturally induce Stein operators. This connection yields a new construction principle for Stein operators, and enables us to study estimators based on these scoring rules through the geometry associated with the induced Stein operators. Using this geometric perspective, we provide a unified characterization of the asymptotic efficiency of the resulting estimators.
    
[^17]: 基于非线性薛定谔桥与分支扩散的生成建模

    Generative Modeling via Nonlinear Schr\"odinger Bridges and Branching Diffusions

    [https://arxiv.org/abs/2610.29476](https://arxiv.org/abs/2610.29476)

    该论文提出了基于前向-后向偏微分方程的非线性薛定谔桥生成建模框架SBNL，并利用分支扩散设计出具有高度并行性和高维可扩展性的高效数值算法。

    

    我们提出了一个基于前向-后向偏微分方程（PDE）的全新生成建模框架。这些方程自然地产生于平均场博弈（MFG）理论中，并与非线性薛定谔系统密切相关。我们的主要贡献是在生成建模的背景下重新解释这一联系，并为其实现开发高效的数值方法。具体而言，我们通过一个耦合的偏微分方程组推导并刻画了经典薛定谔桥问题的非线性扩展，由此得到的模型我们称之为SBNL（非线性薛定谔桥）。为计算其解，我们设计了一种基于分支扩散和半线性偏微分方程概率表示的模拟算法。该方法具有“尴尬并行”（embarrassingly parallel）的特性，可扩展至高维情形，并适用于一大类非线性相互作用泛函。我们通过……

    arXiv:2610.29476v1 Announce Type: new  Abstract: We propose a new generative modeling framework grounded in forward-backward partial differential equations (PDEs). These equations arise naturally in mean-field game (MFG) theory, and are closely related to nonlinear Schr\"odinger systems. Our main contribution is to reinterpret this connection within the setting of generative modeling and to develop an efficient numerical methodology for its practical implementation. Specifically, we derive and characterize a nonlinear extension of the classical Schr\"odinger bridge problem through a coupled system of PDEs, yielding a model we call SBNL (Schr\"odinger Bridge NonLinear). To compute its solution, we design a simulation algorithm based on branching diffusions and the probabilistic representation of semilinear PDEs. The resulting method is embarrassingly parallel, scalable to high dimensions, and applicable to a broad class of nonlinear interaction functionals. We validate the approach thro
    
[^18]: 面向低NFE生成流的密度层面后向误差

    Density-Level Backward Error for Low-NFE Generative Flows

    [https://arxiv.org/abs/2610.29475](https://arxiv.org/abs/2610.29475)

    本文提出密度层面的后向误差框架，分离出真正改变边缘分布的求解器误差分量，从而为低NFE生成流导出感知求解器阶数的计算分配规则以及基于规范自由度的速度场选择目标。

    

    低NFE（网络函数评估次数）生成式ODE采样不仅受求解器阶数的限制，还受固定的网络评估预算如何沿流分布的影响。我们开发了一个密度层面的后向误差框架，该框架能够识别求解器修正方程缺陷中会改变演化边缘分布的分量，同时排除在Wasserstein切几何中不可见的螺线轨迹畸变。所得残差诱导出一种感知求解器阶数的计算分配规则，并借助规范自由度，提供了一个用于从同一密度路径的多个速度代表中选择数值有利者的目标函数。这将我们的方法与由轨迹或端点替代量驱动的调度搜索以及高阶求解器设计区分开来。精确的高斯模型和混合模型确立了该机制，包括基于曲率的分配失效以及连续最优传输并非离散化最优的情形。在预训练的CIFAR-10

    arXiv:2610.29475v1 Announce Type: new  Abstract: Low-NFE generative ODE sampling is constrained not only by solver order, but also by how a fixed budget of network evaluations is distributed along the flow. We develop a density-level backward-error framework that identifies the component of a solver's modified-equation defect that changes the evolving marginal, while excluding solenoidal trajectory distortion that is invisible in Wasserstein tangent geometry. The resulting residual induces a solver-order-aware compute-allocation rule and, through gauge freedom, an objective for selecting numerically favorable velocity representatives of the same density path. This separates our approach from schedule search driven by trajectory or endpoint surrogates and from higher-order solver design. Exact Gaussian and mixture models establish the mechanism, including cases where curvature-based allocation fails and continuum-optimal transport is not discretization-optimal. On a pretrained CIFAR-10 
    
[^19]: 一幕多面：面向长程智能体的效应边缘策略优化

    One Act, Many Masks: Effect-Marginal Policy Optimization for Long-Horizon Agents

    [https://arxiv.org/abs/2610.29427](https://arxiv.org/abs/2610.29427)

    该论文发现了多轮强化学习中信用信号与具体文本响应绑定所导致的“信用更新错位”问题，并提出效应边缘策略优化（EMPO），通过效应级别的代理目标将策略更新与执行层面的等价性对齐，从而改进长程LLM智能体的策略优化。

    

    多轮强化学习（RL）的最新进展显著提升了LLM智能体在复杂、长程交互任务上的性能。然而，尽管在细粒度信用分配方面已取得进展，策略优化仍通过针对特定响应的概率比来应用这些学习信号。不同的文本响应可以产生相同的可执行效应，但由后续结果产生的信用差异仍然与各自的文本实现绑定在一起。我们将这种不匹配识别为“信用更新错位”。我们的分析揭示，即使策略上下文和可执行效应保持不变，信用与文本的配对方式也会直接塑造策略更新的方向。受这一发现启发，我们提出了效应边缘策略优化（EMPO），它通过一个效应级别的代理目标将不同文本实现之间的更新耦合起来，使策略优化与执行层面的等价性保持一致……

    arXiv:2610.29427v1 Announce Type: new  Abstract: Recent advances in multi-turn reinforcement learning (RL) have substantially improved the performance of LLM agents on complex, long-horizon interactive tasks. However, despite progress in fine-grained credit assignment, policy optimization still applies these learning signals through response-specific probability ratios. Different textual responses can induce the same executable effect, yet credit differences arising from subsequent outcomes remain tied to their individual textual realizations. We identify this mismatch as credit-update misalignment. Our analysis reveals that credit-text pairing directly shapes policy-update directions, even when the policy context and executable effect are held fixed. Motivated by this finding, we propose Effect-Marginal Policy Optimization (EMPO), which aligns policy optimization with execution-level equivalence through an effect-level surrogate that couples updates across alternative textual realizat
    
[^20]: 当思想成为记忆：统一上下文学习与微调

    When Thoughts Become Memory: Unifying In-Context Learning and Fine-Tuning

    [https://arxiv.org/abs/2610.29419](https://arxiv.org/abs/2610.29419)

    该论文首次提出基于信息价值精确定律的认证规则，统一判定未见数据应放入预训练模型的上下文还是其权重，并通过随时有效的检验在多种模型与任务上验证了其正确性。

    

    我们首次给出了一个认证规则，用于判定未见数据应属于预训练模型的上下文还是其权重，该规则建立在其信息价值的精确定律之上。这一信息价值——即在留出查询上节省的编码长度——在期望上等于关于查询所存储的比特数，加上先验修复，再减去解码损失，且该定律对任何模型、架构、数据集和任务均成立。该定律的上界可通过一次测量来认证这一选择，而一个随时有效检验则可基于留出查询做出选择，且无论何时停止都能控制误差。在训练过的transformer、TabPFN、占用网络和立体匹配网络上，该上界颁发的每一份证书都是正确的。该检验在翼型压力预测任务上选择TabPFN优于XGBoost，选择训练后的代理模型优于上下文学习器；在32组比较、共64,000个查询流中，有61,948次正确判定GPT-2应使用缓存还是微调，且在零假设流中的错误率至多3.3%。该定律还能诊断何时学习（原文在此截断）

    arXiv:2610.29419v1 Announce Type: cross  Abstract: We are the first to give a certified rule for whether unseen data belong in a pretrained model's context or its weights, built on an exact law for their information value. That value, the code length saved on held-out queries, equals in expectation the bits stored about the query, plus a prior repair, minus a decoding loss, for any model, architecture, dataset and task. The law's ceilings certify the choice from one measurement, and an anytime-valid test makes it from held-out queries, with error controlled whenever it stops. Every certificate the ceilings issue on a trained transformer, TabPFN, an occupancy network and a stereo network is correct. The test picks TabPFN over XGBoost and a trained surrogate over an in-context learner on airfoil pressure, calls the better of GPT-2's cache and fine-tune in 61,948 of 64,000 query streams over 32 comparisons, and errs in at most 3.3% of null streams. The law also diagnoses when learning in 
    
[^21]: 掼蛋自博弈中的“预测—优化”差距

    The Predictive--Optimization Gap in GuanDan Self-Play

    [https://arxiv.org/abs/2610.29411](https://arxiv.org/abs/2610.29411)

    在掼蛋自博弈中，对局进度虽被证明具有真实预测价值，但以冻结胜负预测器的变化 ΔΦ 作回合级监督干预，尽管在个别训练谱系中奏效，五个预注册实验在方法层面均未复现增益，揭示了预测有效性与优化效用之间的差距。

    

    arXiv:2610.29411v1，公告类型：new。结果预测信号在强化学习中常被视为有前景的优化目标，但预测有效性与优化效用是两个不同的命题。我们在掼蛋——一种多回合不完全信息博弈，其中单回合结果的含义取决于公开的对局状态——中揭示了这一区别。留出集分析表明，状态—收益交互作用在恒定的带符号收益效应之外提供了额外的预测价值，证明对局进度在预测意义上是有意义的。随后，我们将冻结的对局胜负预测器的变化量 ΔΦ 作为受控的回合级监督干预手段，应用于匹配的 DanZero 风格蒙特卡洛自博弈协议中。该干预在个别训练谱系中可以成功：一个选定的进度增强检查点在锁定评估上显著优于其匹配的基线。然而，方法层面的结果逆转了这一印象：五个预注册（的实验中没有一个……）（摘要在此处截断）。

    arXiv:2610.29411v1 Announce Type: new  Abstract: Outcome-predictive signals are often treated as promising optimization targets in reinforcement learning, but predictive validity and optimization utility are different claims. We expose this distinction in GuanDan, a multi-round imperfect-information game in which the implication of a round outcome depends on the public match state. Held-out analysis shows that state--gain interactions provide incremental predictive value beyond a constant signed-gain effect, establishing that match progress is predictively meaningful. We then use changes in a frozen match-win predictor, $\Delta\Phi$, as a controlled round-level supervision intervention in a matched DanZero-style Monte Carlo self-play protocol. The intervention can succeed in an individual lineage: a selected progress-augmented checkpoint substantially outperforms its matched Baseline on a locked evaluation. Yet the method-level result reverses this impression: none of five preregistere
    
[^22]: 带噪声观测的级联赌博机：算法与遗憾界

    Cascading Bandits with Noisy Observations: Algorithms and Regret Bounds

    [https://arxiv.org/abs/2610.29381](https://arxiv.org/abs/2610.29381)

    该论文提出了一种探测成本与位置相关且反馈含噪声的随机级联赌博机模型，刻画了最优短列表可通过按每臂分数排序求得的条件，并给出了实例相关遗憾匹配理论下界的UCB算法。

    

    我们提出了一种新颖的随机级联赌博机设置，其中探测从L个臂中选取K个臂构成的有序短列表会产生与位置相关的成本，而学习者只能观察到关于每个臂真实输出的噪声反馈。我们识别出一类噪声模型的子类，对于这些模型，可以通过依据每个臂的分数对臂进行排序来找到最优短列表：当信道不存在假阳性时，这样的分数对选择和排序同时存在；当测量无成本时，这样的分数仅对排序存在（但对选择不存在）；然而在一般情况下，一旦噪声和成本同时存在，选择和排序都无法分离。在选择和排序分数都存在的情形下，我们提出了一种基于UCB的算法，在时间范围T上具有O((L-K) log T)的实例相关遗憾，该遗憾与算法无关的下界相匹配（在常数因子内）。对于分数不存在的更具挑战性的场景，我们开发了一种新算法……

    arXiv:2610.29381v1 Announce Type: new  Abstract: We propose a novel stochastic cascading bandit setup where probing an ordered shortlist of $K$ out of $L$ arms incurs position-dependent costs, and the learner observes only noisy feedback regarding each arm's true output. We identify a sub-class of noise models for which the optimal shortlist can be found by sorting arms according to a per-arm score: such a score exists for both selection and ordering when the channel has no false positives, and for ordering alone (but not selection) when measurements are costless; however in general neither selection nor ordering is separable once noise and cost are both present. For the case where both selection and ordering scores exist, we provide a UCB-based algorithm with $O((L-K)\log T)$ instance-dependent regret over a time horizon $T$, which matches an algorithm-agnostic lower bound (up to constant factors). For the more challenging scenario where scores don't exist, we develop a new algorithmi
    
[^23]: GaP-MoE：通过正交梯度累积与投影实现专家混合模型中的显式专家专业化

    GaP-MoE : Explicit Expert Specialization in Mixture-of-Experts via Orthogonal Gradient Accumulation and Projection

    [https://arxiv.org/abs/2610.29355](https://arxiv.org/abs/2610.29355)

    提出GaP-MoE方法，通过正交梯度累积与投影直接操纵专家输出梯度而非修改损失函数，从而在稀疏专家混合模型中诱导显式的专家专业化，解决表示坍缩问题。

    

    稀疏专家混合模型通过将输入令牌仅路由到专家网络的一个子集来提升可扩展性，但这类模型常常遭受表示坍缩问题，即多个专家收敛到相似的函数并丧失其预期的多样性。先前的工作要么通过改进路由器架构，要么通过引入隐式惩罚专家之间相似性的辅助损失项来解决这一局限。然而，基于路由器的方法仍然容易受到路由坍缩的影响，而基于损失的方法在优化过程中可能与任务原始目标发生冲突。在本工作中，我们提出了GaP-MoE（MoE中的梯度累积与投影），该方法通过直接操纵专家输出梯度而非修改损失函数来诱导专家专业化。在每个训练步骤中，GaP-MoE基于归一化路由分数，从TopK路由的专家中随机选择一个专家作为领导者……

    arXiv:2610.29355v1 Announce Type: new  Abstract: Sparse Mixture-of-Experts (SMoE) models improve scalability by routing input tokens to only a subset of expert networks, but they often suffer from representation collapse, in which multiple experts converge toward similar functions and lose their intended diversity. Prior work has addressed this limitation either by improving router architectures or by introducing auxiliary loss terms that implicitly penalize similarity among experts. However, router-based approaches remain vulnerable to routing collapses, and loss-based approaches can conflict with the task's original objective during optimization. In this work, we propose GaP-MoE (Gradient Accumulation and Projection in MoE), which induces expert specialization by directly manipulating expert output gradients rather than modifying the loss function. At each training step, GaP-MoE stochastically selects one of the $TopK$ routed experts as a leader based on normalized routing scores, an
    
[^24]: ExpEvolve：面向开放式发现的经验结构化与复用

    ExpEvolve: Structuring and Reusing Experience for Open-Ended Discovery

    [https://arxiv.org/abs/2610.29303](https://arxiv.org/abs/2610.29303)

    提出ExpEvolve框架，将思想层级的提取与控制注入进化式编码代理，通过发现能够解释程序间评分差异的高层级思想因素并复用经验，提升开放式科学发现的样本效率。

    

    诸如AlphaEvolve这样的进化编码代理是利用大语言模型（LLM）构建AI科学家的显著成功案例之一。这些代理通过迭代改进和演化程序来解决开放式科学问题，利用LLM的先验知识和推理能力。尽管取得了成功，但对程序的变异大多是基于表面启发式的低层级操作，缺乏持久的控制单元。相比之下，科学是通过思想实现复利式积累的。在这项工作中，我们提出了ExpEvolve，它将思想层级的提取与控制注入进化过程：它从程序中发现并利用能够解释不同程序之间评分差异的高层级因素。我们刻画了利用思想因素进行发现何时能提升样本效率的理论条件。我们还构建了一个实用框架，以展示提取出的高层级思想因素有助于指导基于经验的利用。

    arXiv:2610.29303v1 Announce Type: cross  Abstract: Evolutionary coding agents such as AlphaEvolve are one of the notable successes in using Large Language Models (LLMs) to build AI Scientists. These agents tackle open-ended scientific problems by iteratively improving and evolving programs, leveraging the prior knowledge and reasoning capabilities of LLMs. Despite their success, mutations over programs are mostly low-level upon surface heuristics, without a persistent unit of control. In contrast, science compounds through ideas. In this work, we propose ExpEvolve, which injects idea-level extraction and control into evolution: it discovers and leverages the high-level factors from programs that explain scoring differences among different programs. We characterize theoretical conditions on when discovery with idea factors improves sample efficiency. We also build a practical framework to show that the extracted high-level idea factors are helpful for guiding exploitation from experienc
    
[^25]: 近似所罗门诺夫归纳：论学习得到的通用预测器的鲁棒性

    Approximate Solomonoff Induction: On the Robustness of Learned Universal Predictors

    [https://arxiv.org/abs/2610.29289](https://arxiv.org/abs/2610.29289)

    本文研究了实际机器学习算法产生的近似最优预测器的鲁棒性问题，给出了近似对预测误差最坏情况影响的紧致界，从而论证了通过对数损失最小化学习到的通用预测器在近似条件下依然保持稳健。

    

    通过在大规模多样化数据集上进行对数损失最小化训练的大型语言模型为何展现出广泛的能力和可泛化的技能，这一问题尚未被完全理解。Grau-Moya等人（ICML 2024）通过元学习的视角提供了一个概念性解释：若序列生成环境 $\nu$ 从先验 $w_{\nu}$ 中抽取，则最小化总体对数损失可以恢复相应的贝叶斯混合预测器 $\xi=\sum_\nu w_\nu\nu$。在 $w$ 为通用先验的理想化极限下，最优预测器对应于所罗门诺夫通用归纳。然而，实用的机器学习算法通常只能产生近似最优的预测器。这引出了一个自然的问题：预测性能在近似条件下是否依然保持鲁棒？在本文中，我们首先给出了近似对预测误差的最坏情况影响的紧致界。我们证明任何满足 $ 的预测器 $\rho$……（摘要原文在此处被截断）

    arXiv:2610.29289v1 Announce Type: new  Abstract: Why large language models trained via log-loss minimization over massive and diverse datasets exhibit broad capabilities and generalizable skills is not yet fully understood. Grau-Moya et al. (ICML 2024) offer a conceptual explanation through the lens of meta-learning: if sequence-generating environments $\nu$ are drawn from a prior $w_{\nu}$, minimizing population log-loss recovers the corresponding Bayesian mixture predictor $\xi=\sum_\nu w_\nu\nu$. In the idealized limit in which $w$ is a universal prior, the optimal predictor corresponds to Solomonoff universal induction. Practical machine learning algorithms, however, generally produce only \emph{approximately} optimal predictors. This raises a natural question: \emph{Does the prediction performance remain robust under approximation}? In this paper, we first provide a tight bound on the worst-case effect of approximation on prediction error. We show any predictor $\rho$ satisfying $
    
[^26]: 面向测度值数据的薛定谔-巴斯桥重心模型

    Schr\"odinger-Bass Bridge Barycenter Models for Measure-Valued Data

    [https://arxiv.org/abs/2610.29227](https://arxiv.org/abs/2610.29227)

    本文利用薛定谔-巴斯桥的方向不对称性与对偶漂移-扩散控制，提出四种新型测度值数据重心模型，并从理论上证明了这些重心的存在性，且证明高斯输入下解码器形式存在高斯最优解，可将重心计算化为有限维优化。

    

    arXiv:2610.29227v1 公告类型：交叉 摘要：基于概率测度之间随机与确定性最优传输桥的重心模型，已被证明对（概率）测度值数据的有监督和无监督学习十分有用。在本文中，我们利用最近提出的两测度间薛定谔-巴斯桥中出现的方向不对称性及对偶漂移-扩散控制，提出了SBB-编码器、SBB-解码器、SBB-对称和SBB-马尔可夫四种重心模型。针对每一个新模型，我们的第一个理论结果在温和假设下证明了SBB重心的存在性。我们的第二个理论结果刻画了高斯分布的解码器SBB重心：我们证明对于高斯输入测度，解码器形式存在高斯极小化器，从而将重心计算归结为有限维优化，并得到一个紧凑的算法。对于一般的…

    arXiv:2610.29227v1 Announce Type: cross  Abstract: Barycenter models, based on stochastic and deterministic optimal transport bridges between probability measures, have proven to be useful for supervised and unsupervised learning for (probability) measure-valued data. In this paper, we utilize the directional asymmetry and dual drift-diffusion control arising in the recently proposed Schr\"odinger-Bass Bridge (SBB) between two measures, and propose \textit{SBB-encoder}, \textit{SBB-decoder}, \textit{SBB-symmetric}, and \textit{SBB-Markovian} barycenter models. For each of these novel models, our first theoretical result establishes the existence of the SBB-barycenter under mild assumptions. Our second theoretical result characterizes the decoder SBB-barycenter of Gaussians where we show that for Gaussian input measures, the decoder formulation admits a Gaussian minimizer, reducing barycenter computation to a finite-dimensional optimization and yielding a compact algorithm. For general 
    
[^27]: 曲率缩放的锐度感知最小化：超越平滑性的收敛性

    Curvature-Scaled Sharpness-Aware Minimization: Convergence Beyond Smoothness

    [https://arxiv.org/abs/2610.29192](https://arxiv.org/abs/2610.29192)

    该论文利用相对梯度失真揭示了SAM在广义平滑条件下失真可能无界、SAM会循环而USAM会发散的现象，建立了梯度扰动统一失真控制的必要条件，并由此提出曲率缩放的锐度感知最小化方法，实现了超越平滑性假设的收敛性保证。

    

    锐度感知最小化（SAM）已成为训练深度神经网络的一种广泛使用的方法。尽管其应用广泛，但现有的SAM类方法的收敛性保证都要求平滑性这一限制性假设。广义平滑条件最近在梯度方法的分析中被提出，但对于SAM仍未得到充分探索。为了分析SAM的扰动在广义平滑性下的影响，我们使用相对梯度失真来量化锐度感知方向偏离原始梯度的程度。在$(L_0,L_1)$-平滑性条件下，我们证明这种失真对于SAM在接近平稳点时可能变得无界，对于其未归一化变体（USAM）在大梯度时可能变得无界，并给出了SAM发生循环、USAM发散的具体例子。随后，我们针对沿梯度方向的扰动（其非负乘子仅依赖于梯度范数）建立了实现统一失真控制的必要条件。

    arXiv:2610.29192v1 Announce Type: cross  Abstract: Sharpness-Aware Minimization (SAM) has established itself as a widely used method for training deep neural networks. Despite its popularity, existing convergence guarantees for SAM-type methods require smoothness, a restrictive assumption. Generalized smoothness conditions have recently been proposed in the analysis of gradient methods but remain underexplored for SAM. To analyze the effect of SAM's perturbation under generalized smoothness, we use relative gradient distortion to quantify how the sharpness-aware direction departs from the original gradient. Under $(L_0,L_1)$-smoothness, we show that this distortion can become unbounded near stationarity for SAM and at large gradients for its unnormalized variant (USAM), with examples where SAM cycles and USAM diverges. We then establish a necessary condition for uniform distortion control among perturbations along the gradient whose nonnegative multipliers depend only on its norm. Moti
    
[^28]: 内部引导：一种用于倾斜扩散模型的采样方法

    Internal Guidance as a Sampling Method for Tilted Diffusion Models

    [https://arxiv.org/abs/2610.29086](https://arxiv.org/abs/2610.29086)

    本文提出“内部引导”方法，通过改变基础去噪器的求值输入点而非修改其输出来采样倾斜密度，并利用随机对角重缩放改进训练算法，解决了所需求值点超出训练分布的问题。

    

    奖励引导方法使用仅由基础分布样本训练的扩散模型，来对基础密度的指数倾斜分布进行采样。现有方法采用我们所说的“外部引导”：即在漂移项中加入加性项，或在转移概率上加入乘性项。我们提出了“内部引导”，它不修改基础去噪器的输出，而是改变对其求值的输入点。对于嵌入在超立方体 $\{-1,1\}^{d}$ 中的离散分布，任何线性倾斜密度的去噪器恰好等于基础去噪器在经过平移和重新缩放的输入上的求值结果。在实践中，基础去噪器是在基础分布的加噪样本上训练的神经网络。对倾斜密度进行采样所需的求值点，相对于基础分布的基本加噪过程而言超出了训练分布范围。我们通过对数据进行随机对角重缩放来修改训练算法，使去噪器能够（摘要在此处被截断）

    arXiv:2610.29086v1 Announce Type: cross  Abstract: Reward guidance methods sample an exponential tilt of a base density using a diffusion model trained with only samples from the base distribution. Existing methods use what we call external guidance: additive terms in the drift or multiplicative terms on the transition probabilities. We propose internal guidance which leaves the output of the base denoiser unmodified and instead changes the point at which it is evaluated. For discrete distributions embedded in the hypercube $\{-1,1\}^{d}$, the denoiser of any linearly tilted density is exactly the base denoiser evaluated at a shifted and rescaled input. In practice the base denoiser is a neural network trained on noised samples of the base distribution. The required evaluation points for sampling the tilted density would be out-of-distribution for basic noising of the base distribution. We modify the training algorithm with random diagonal rescalings of the data, so that the denoiser n
    
[^29]: 基于能量核的最大均值差异的跳跃修正Wasserstein梯度流

    A Jump-Modified Wasserstein Gradient Flow for Maximum Mean Discrepancy with Energy Kernels

    [https://arxiv.org/abs/2610.29007](https://arxiv.org/abs/2610.29007)

    本文提出一种带跳跃项的修正Wasserstein梯度流来最小化能量核生成的最大均值差异（MMD），证明了该流的全局适定性、仅凭有限二阶矩即可收敛到目标分布，并构造了能量单调不增的全局无碰撞粒子过程。

    

    我们研究由能量核 $K(z)=-|z|^q$（在每一维 $d\ge2$ 时 $0<q<2$，以及一维情形 $1\le q<2$）生成的平方最大均值差异（MMD）的跳跃修正Wasserstein梯度流。在次临界密度和有限矩假设下，我们证明了全局适定性。所引入的跳跃项将质量转移向一阶变分较小的取值，并与输运项耗散相同的MMD能量。在我们对目标测度和跳跃速率的假设下，每个具有有限MMD能量且跳跃耗散为零的Borel概率测度都等于目标测度。将这一刚性结果与紧性及二阶矩恒等式相结合，可以从仅具有有限二阶矩的初始数据得到向目标测度的收敛，而无需额外的一致时间密度或矩假设。我们还构造了一个全局无碰撞的粒子过程，其能量沿每条路径单调不增。定量平均场估计适用于……

    arXiv:2610.29007v1 Announce Type: cross  Abstract: We study a jump-modified Wasserstein gradient flow of the squared maximum mean discrepancy generated by the energy kernels $K(z)=-|z|^q$, $0<2$, in every dimension $d\ge2$ as well as the one-dimensional range $1\le q<2$. Under subcritical density and finite-moment assumptions, we prove global well-posedness. The added jumps transfer mass toward smaller values of the first variation and dissipate the same MMD energy as the transport term. Under our hypotheses on the target and jump rates, every Borel probability measure with finite MMD energy and zero jump dissipation is equal to the target. Combining this rigidity with compactness and a second-moment identity yields convergence to the target from finite-second-moment data, without additional uniform-in-time density or moment assumptions. We also construct a global collision-free particle process whose energy is nonincreasing along each path. A quantitative mean-field estimate applies t
    
[^30]: 从公共信息到典型相关分析及其延伸：连接信息论、信号处理与机器学习

    From Common Information to Canonical Correlation Analysis and Beyond: Bridging Information Theory, Signal Processing, and Machine Learning

    [https://arxiv.org/abs/2610.28978](https://arxiv.org/abs/2610.28978)

    本教程首次将统计学中的典型相关分析、信息论中的两种公共信息概念以及工程领域的多视图子空间方法置于统一框架下，为无监督恢复多观测中公共分量的跨学科问题建立了桥梁，并厘清了不同理论之间的精确等价与类比关系。

    

    信号处理、机器学习和神经成像中的许多问题提供了对同一过程的若干联合观测，每个观测将所有观测共有的分量与自身私有的分量混合在一起，并需要在无监督条件下恢复公共分量。这个问题在多个研究群体中得到了研究，且在很大程度上是相互独立的：统计学家提出了由Hotelling于1936年引入的典型相关分析（CCA）；信息论学者提出了两种公共信息的概念，分别源于Gács和Körner以及Wyner；工程师们为传感器阵列和多用户通信开发了子空间方法，以设计多视图接收器。本教程在单一的阐述中呈现了各研究群体为这一问题开发的主要结果和工具，并解释了它们交汇处涌现的联系，区分了精确等价关系与特殊情况和类比。我们发展了信息论……（原文摘要截断于此）

    arXiv:2610.28978v1 Announce Type: cross  Abstract: Many problems in signal processing, machine learning, and neuroimaging provide several jointly observed measurements of one process, each mixing a component common to all of them with a component private to itself, and require the common component to be recovered without supervision. This problem has been studied in several communities, largely independently. Statisticians proposed canonical correlation analysis (CCA), introduced by Hotelling in 1936. Information theorists proposed two notions of common information, due to G\'acs and K\"orner and to Wyner. Engineers developed subspace methods for sensor arrays and multiuser communications to design multiview receivers. This tutorial presents the main results and tools each community developed for this problem in a single exposition, and explains the connections that emerge where they meet, distinguishing exact equivalences from special cases and analogies. We develop the information-th
    
[^31]: 上下文强化学习是后验采样器吗？一项实证研究

    Is In-Context Reinforcement Learning A Posterior Sampler? An Empirical Study

    [https://arxiv.org/abs/2610.28977](https://arxiv.org/abs/2610.28977)

    本文通过建立贝叶斯后验采样器必须满足的两个必要条件——充分统计量不变性和鞅性质，并据此构造统计假设检验，实证考察了上下文强化学习（特别是DPT及其变体）在实践中是否真正表现为贝叶斯后验采样器。

    

    上下文强化学习利用大型语言模型令人印象深刻的上下文学习能力，在推理时零样本泛化到新的、未见过的任务，而无需任何参数更新。在众多方法中，决策预训练Transformer（DPT）因其理论基础而尤为引人注目：在模型具有充分表达能力、预训练数据无限等理想化假设下，DPT可以执行贝叶斯后验采样。然而这一理论刻画在实践中是否仍然成立仍是悬而未决的问题。因此，我们提出这样一个问题：上下文强化学习，具体而言是DPT和我们提出的DPT变体，在实践中是否是一个贝叶斯后验采样器？为了研究这一问题，我们建立了贝叶斯后验采样器必须满足的两个必要条件：充分统计量不变性和鞅性质。随后我们构造了统计假设检验，以统计方法评估实践中训练的（摘要在此处截断）

    arXiv:2610.28977v1 Announce Type: cross  Abstract: In-context reinforcement learning (ICRL) exploits the impressive in-context learning ability of large language models to zero-shot generalize to new, unseen tasks at inference time without any parameter updates. Among many approaches, the Decision-Pretrained Transformer (DPT) is particularly appealing for its theoretical grounding: under idealized assumptions such as a sufficiently expressive model and infinite pretraining data, DPT can perform Bayesian posterior sampling. Whether this characterization survives in practice remains open. Therefore, we ask: \textit{is ICRL, specifically DPT and our proposed DPT variants, a Bayesian posterior sampler in practice?} To approach this question, we establish two necessary conditions that a Bayesian posterior sampler must satisfy: \textit{sufficient-statistic invariance} and the \textit{martingale property}. We then construct statistical hypothesis tests to statistically assess if practically-t
    
[^32]: 审计推理时模型偏移下语言模型的保形预测

    Auditing Conformal Prediction for Language Models Under Inference-Time Model Shift

    [https://arxiv.org/abs/2610.28937](https://arxiv.org/abs/2610.28937)

    该论文首次提出并形式化了“推理时模型偏移”现象——即部署与校准阶段推理配置（如量化模式和推理温度）的差异会引起保形分数分布变化并破坏分割保形可交换性，并通过8个语言模型、4个基准和12种推理配置的大规模实验系统评估了其对语言模型保形预测可靠性的影响。

    

    语言模型的保形预测（CP）通常在匹配协议下进行评估，即校准和部署使用相同的训练后推理配置，例如量化模式和推理温度。而在实际部署中，为了满足内存或吞吐量限制或服务策略的要求，这些配置选择往往与校准时不同。我们认为这暴露了当前语言模型CP评估协议的一个局限性。我们将这一现象形式化为“推理时模型偏移”：即由推理配置而非输入-标签分布引起的保形分数分布变化，并从理论上刻画了即使任务分布和已训练模型保持不变，这种偏移也会破坏分割保形可交换性。随后，我们在8个语言模型、4个有限标签基准和12种推理配置上评估了其影响。我们发现匹配评估...

    arXiv:2610.28937v1 Announce Type: new  Abstract: Conformal prediction (CP) for language models is typically evaluated under a matched protocol, in which calibration and deployment use the same post-training inference configuration, such as quantization mode and inference temperature. In deployment, these choices often differ from those used at calibration to meet memory or throughput constraints or serving policies. We argue that this exposes a limitation in current CP evaluation protocols for language models. We formalize the underlying phenomenon as inference-time model shift: a change in the conformal score distribution induced by inference configuration rather than by the input--label distribution, and theoretically characterize how it breaks split-conformal exchangeability even when the task distribution and trained model are unchanged. We then evaluate its consequences across 8 language models, 4 finite-label benchmarks, and 12 inference configurations. We find that matched evalu
    
[^33]: 回溯直接效应、间接效应与总效应

    Backtracking Direct, Indirect and Total Effects

    [https://arxiv.org/abs/2610.28910](https://arxiv.org/abs/2610.28910)

    本文提出基于回溯反事实的因果效应新定义，无需干预处理变量或中介变量，而是通过回溯条件分布构造满足处理-中介约束的替代外生状态，从而得到回溯直接效应、间接效应和总效应。

    

    自然直接效应和间接效应是通过跨世界干预反事实来定义的。我们提出了一种基于回溯反事实的替代性构造方法。我们不再对处理变量或中介变量进行干预，而是利用回溯条件分布来构造一个满足所需处理-中介约束的替代性外生状态。由此得到了回溯直接效应和回溯间接效应，以及回溯总效应。同样地，我们也对人群干预效应进行了相应的改造。

    arXiv:2610.28910v1 Announce Type: cross  Abstract: Natural direct and indirect effects are defined through cross-world interventional counterfactuals. We propose an alternative construction based on backtracking counterfactuals. Rather than intervening on the treatment or mediator, we use a backtracking conditional distribution to construct an alternative exogenous state satisfying a desired treatment-mediator constraint. This yields the backtracking direct and indirect effects, and also the backtracking total effect. Likewise, we adapt the population intervention effects.
    
[^34]: 实际上，所有可学习的分布类都是可隐私学习的

    Actually, All Learnable Distribution Classes Are Privately Learnable

    [https://arxiv.org/abs/2610.28878](https://arxiv.org/abs/2610.28878)

    本文证明了所有在总变差距离下可学习的分布类在（ε, δ）-差分隐私下同样可学习，从而证明了 Ashtiani 的猜想。

    

    我们证明，每一个在总变差距离下能够以有限个样本学习到任意小目标误差的分布类，同样可以在（$\varepsilon, \delta$）-差分隐私条件下以有限个样本学习到任意小的目标误差。这证明了 Ashtiani 提出的一个猜想。Bun、Kamath、Mouzakis 和 Singhal 之前的工作 [arXiv:2402.00267] 曾对该猜想给出了弱反驳，表明当可学习性相对于固定目标误差来定义时，这一蕴含关系并不成立。与之相反，我们证明，在可学习性的标准定义下，该猜想是成立的。

    arXiv:2610.28878v1 Announce Type: new  Abstract: We show that every class of distributions that is learnable up to arbitrarily small target error in total variation distance with finitely many samples is also learnable under ($\varepsilon, \delta$)-differential privacy up to arbitrarily small target error with finitely many samples. This proves a conjecture of Ashtiani. Previous work by Bun, Kamath, Mouzakis, and Singhal [arXiv:2402.00267] gave a weak refutation of this conjecture, showing that this implication is not true when learnability is defined with respect to a fixed target error. In contrast, we show that, under the standard definition of learnability, the conjecture is true.
    
[^35]: 高斯测度下神经网络的深度分离

    Depth Separation for Neural Networks under Gaussian Measure

    [https://arxiv.org/abs/2610.28800](https://arxiv.org/abs/2610.28800)

    该论文在高斯测度下建立了神经网络的深度分离结果：存在一个径向函数可被多项式规模的3层ReLU网络以小L²误差逼近，但2层网络要逼近它需要指数级规模，其核心工具是关于径向函数Hermite高频展开的不可逼近性定理，且全部证明均经Lean 4形式化验证。

    

    我们重新审视了由Eldan和Shamir、Daniely等人研究过的神经网络深度分离问题。具体而言，我们考虑在d维标准高斯测度下径向函数的L²逼近这一自然设定。我们证明，存在一个标准高斯测度下的L²径向函数，它可以被一个规模关于d为多项式的深度为3的ReLU神经网络以固定的足够小的L²误差逼近，但对于权重具有亚指数上界的深度为2的网络而言，若要逼近该函数，其规模必须至少关于d呈指数级。为此，我们证明了一个一般性结果：如果某个径向函数在其Hermite展开的高频部分具有可观的质量，那么它无法被宽度较小且权重有界的深度为2的神经网络很好地逼近。所有定理、引理、命题和推论均已在Lean 4中得到形式化验证。

    arXiv:2610.28800v1 Announce Type: cross  Abstract: We revisit the problem of depth separation for neural networks, as studied by Eldan and Shamir, Daniely, and others. Specifically, we consider the natural setting of $L^2$ approximation of radial functions under the standard Gaussian measure in $d$ dimensions. We show that there is a radial function in $L^2$ under the standard Gaussian measure that can be approximated to a fixed sufficiently small $L^2$ error by a depth-3 ReLU neural network of polynomial size in $d$ but can only be approximated by depth-2 networks with subexponentially bounded weights if their size is at least exponential in $d$. For that, we prove a general result showing that if a radial function has substantial mass in the high frequencies of its Hermite expansion, then it cannot be well approximated by a depth-2 neural network with small width and bounded weights. All theorems, lemmas, propositions, and corollaries are verified in Lean~4.
    
[^36]: ReAPO：面向大语言模型推理高效探索的重试感知策略优化

    ReAPO: Retry-Aware Policy Optimization for Efficient Exploration in LLM Reasoning

    [https://arxiv.org/abs/2610.28770](https://arxiv.org/abs/2610.28770)

    提出基于随机最短路径框架的信用分配方法 ReAPO，通过为失败轨迹分配额外生成预算来实现重试感知的策略优化，从而提升大语言模型推理中探索的效率。

    

    RLVR（可验证奖励强化学习）提升了语言模型的推理能力，但其探索通常依赖独立的随机采样，这会导致反复采样到相似的失败策略。已有工作通过在后续尝试的上下文中包含错误解法，在采样过程中引入依赖关系来引导探索，并取得了成功。然而，这类方法引入了一个额外的问题：如何为每次尝试分配信用。我们提出从随机最短路径的视角来审视信用分配问题——这是经典强化学习中的一个框架，旨在最小化到达终止状态（在我们的场景中即经过验证的成功回答）的期望累积成本。我们证明，在适当的简化假设下，这一视角自然地引出了对 GRPO 的一个简单修改，我们称之为重试感知策略优化。ReAPO 会为失败的轨迹分配额外的生成资源……

    arXiv:2610.28770v1 Announce Type: new  Abstract: RLVR improves language-model reasoning, but commonly relies on independent stochastic rollouts for exploration, which can repeatedly sample similar failed strategies. Prior work has seen success with approaches that guide exploration by introducing dependence into the sampling process by including incorrect solutions in the context for subsequent attempts. Still, these methods introduce the additional question of how to assign credit to each attempt. We propose to view the credit assignment through the lens of Stochastic Shortest Paths, a framework in classical reinforcement learning that seeks to minimize the expected cumulative cost to reach a terminal state (in our case, a verified successful response). We show that, under suitable simplifying assumptions, this perspective naturally leads to a simple modification to GRPO, which we call Retry-Aware Policy Optimization (ReAPO). ReAPO allocates additional generation to failed trajectorie
    
[^37]: 逆向戈德定理：基于基底受限对比学习器的正数据语言识别

    A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners

    [https://arxiv.org/abs/2610.28747](https://arxiv.org/abs/2610.28747)

    该论文证明了一个“逆向戈德定理”：借助先天递归对比算子自我生成结构受限的负例扰动，并以InfoNCE式对比学习训练的受限评分器，在忠实性条件下可仅凭正数据实现高概率的PAC条件语言识别，且样本复杂度仅为多项式级。

    

    戈德定理表明，任何学习器都无法仅凭正例呈现来识别超有限类中的任意语言。我们证明了一个形态不同的正面结果。固定一个先天的“递归对比算子”$R$，它将任意字符串映射到其结构上受限的有限扰动集合；再考虑一个容量受限的判别式评分器，它通过在自我生成的扰动上进行InfoNCE式对比估计训练。在算子、评分器与目标语言之间满足忠实性条件的前提下，训练后的评分器能够仅凭正数据实现“PAC条件识别”：以高概率，它会把除$\varepsilon$比例之外的所有正例都排在其自身轨道中每个语言外扰动之上，且样本复杂度关于模型容量、轨道大小、分数界限和$1/\varepsilon$呈多项式级。负例由学习器自行构造，而非由外部提供……

    arXiv:2610.28747v1 Announce Type: cross  Abstract: Gold's theorem shows that no learner can identify an arbitrary language in a superfinite class from positive presentations alone. We prove a positive result of a different shape. Fix an innate \emph{recursive contrast operator} $R$ mapping any string to a finite set of structurally bounded perturbations of it, and a discriminative scorer of bounded capacity trained by InfoNCE-style contrastive estimation \citep{vandenoord2018,macollins2018} on self-generated perturbations. Under a faithfulness condition relating operator, scorer, and target, the trained scorer achieves \emph{PAC conditional identification} from positive data alone: with high probability it ranks all but an $\varepsilon$-fraction of positives above every out-of-language perturbation in their own orbit, with sample complexity polynomial in the capacity, the orbit size, the score bound, and $1/\varepsilon$. The negatives are constructed by the learner, not supplied by an 
    
[^38]: 跨门店借力：基于选择模型的收益管理迁移学习

    Borrowing Strength Across Stores: Transfer Learning for Choice-Based Revenue Management

    [https://arxiv.org/abs/2610.28708](https://arxiv.org/abs/2610.28708)

    该论文提出了一种面向MNL需求模型的收缩估计量，通过迁移学习在门店本地需求估计与跨门店信息共享之间取得平衡，从而在本地数据有限的情况下提升零售收益管理中品类与定价决策的可靠性。

    

    大型零售连锁企业必须根据当地顾客偏好来调整品类和定价决策，然而即使是拥有大量销售记录的零售商，在估计门店或本地市场层面的需求时，数据往往有限，尤其是在销售周期较短的产品类别中。因此，本地估计（为每个门店单独拟合需求模型）可能不可靠，而跨门店汇集数据虽能提高估计精度，却以忽略本地需求异质性为代价。我们研究了零售商如何在多项Logit（MNL）需求模型下通过迁移学习来平衡这两种相互冲突的效应。我们提出了一种收缩估计量，将门店特定的需求估计与从其他门店汇集的信息相结合。与运营管理领域现有信息传递理论文献所基于的线性或二次设定不同，非线性MNL似然函数无法直接……（摘要在此处截断）

    arXiv:2610.28708v1 Announce Type: cross  Abstract: Large retail chains must tailor assortment and pricing decisions to local customer preferences, yet even retailers with extensive sales records often have limited data for estimating demand at the store or local market level, particularly in product categories with short selling windows. Local estimation (fitting a demand model for each store on its own) can therefore be unreliable, while pooling data across stores improves estimation precision at the cost of ignoring local demand heterogeneity. We study how retailers can balance these competing effects through transfer learning under a multinomial logit (MNL) demand model. We propose a shrinkage estimator that combines store-specific demand estimates with information pooled from other stores. Unlike the linear or quadratic settings underlying much of the existing theoretical literature on information transfer in operations, the nonlinear MNL likelihood does not admit a straightforward
    
[^39]: 经验去噪器的隐私核算

    Privacy Accounting for Empirical Denoisers

    [https://arxiv.org/abs/2610.27710](https://arxiv.org/abs/2610.27710)

    本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。

    

    扩散模型已成为现代生成系统的基础，但其强大的记忆能力引发了隐私方面的担忧。虽然差分隐私（DP）训练能够提供形式化的保证，但其代价是样本质量的下降。在本工作中，我们研究了一个与之互补的问题：在采样过程本身中会产生多少隐私损失？直接分析现代神经去噪器是具有挑战性的，因为它们对单个训练样本的敏感性通常是难以计算的。为了克服这一挑战，我们构建并分析了一种经过截断的隐私可问责经验去噪器——这是一种无需训练的贝叶斯最优去噪器估计器，其每一步的敏感性都可以被显式计算，从而使每个去噪步骤都能成为一个高斯机制。这一视角使得在以下情况下进行隐私核算成为可能：当每个训练样本对每个去噪步骤都有贡献时，可在高斯DP下进行隐私核算；当每个……（摘要在此处被截断）

    arXiv:2610.27710v1 Announce Type: new  Abstract: Diffusion models have emerged as the foundation of modern generative systems, yet their high memorization capacity raises privacy concerns. While differentially private (DP) training provides formal guarantees, it comes at a cost in sample quality. In this work, we study a complementary question: how much privacy loss is incurred during the sampling process itself? Directly analyzing modern neural denoisers is challenging because their sensitivity to a single training example is generally intractable. To overcome this challenge, we construct and analyze a clipped privacy-accountable empirical denoiser, a training-free estimator of the Bayes-optimal denoiser, whose per-step sensitivity can be computed explicitly, allowing each denoising step to be a Gaussian mechanism. This perspective enables privacy accounting under Gaussian DP when each training example contributes to each denoising step, and under $f$-DP trade-off functions when each 
    
[^40]: 用于调查加权的层次高斯过程：结构化总体中的局部等价权重

    Hierarchical Gaussian Processes for Survey Weighting: Local Equivalent Weights in Structured Populations

    [https://arxiv.org/abs/2610.20321](https://arxiv.org/abs/2610.20321)

    本文提出基于层次高斯过程回归的调查加权新框架，通过结构化组级效应与核平滑在稀疏后分层单元格间共享信息，在降低计算成本的同时为非线性模型构建了局部等价权重并给出了理论收敛保证。

    

    基于回归的调查加权通过等价权重将基于模型的总体估计与基于设计的推断联系起来。对于非线性模型，等价权重可以通过后分层总体估计量的敏感性进行局部定义。然而，传统回归模型可能无法充分捕捉稀疏后分层单元格中的非线性关系和复杂交互作用。我们开发了一个基于层次高斯过程（HGP）回归的调查加权框架，将结构化的组级效应与基于核的平滑相结合，以在相关单元格之间共享信息。我们推导了利用结构化协方差表示来降低计算成本的高斯后验预测公式。我们还在给定的正则性条件下建立了后验预测均值的均匀收敛结果，并通过匹配一阶敏感性来构建局部等价权重。

    arXiv:2610.20321v1 Announce Type: cross  Abstract: Regression-based survey weighting connects model-based population estimation with design-based inference through equivalent weights. For nonlinear models, equivalent weights can be defined locally through the sensitivity of a poststratified population estimator. However, conventional regression models may inadequately capture nonlinear relationships and complex interactions across sparse poststratification cells. We develop a survey-weighting framework based on hierarchical Gaussian process (HGP) regression, combining structured group-level effects with kernel-based smoothing to share information across related cells. We derive Gaussian posterior prediction formulas that exploit the structured covariance representation to reduce computational costs. We also establish a uniform convergence result for the posterior predictive mean under stated regularity conditions and construct local equivalent weights by matching the first-order sensit
    
[^41]: 基于正-未标记学习的扩散模型渐近精确负引导

    Asymptotically Exact Negative Guidance of Diffusion Models via Positive-Unlabeled Learning

    [https://arxiv.org/abs/2610.19802](https://arxiv.org/abs/2610.19802)

    提出一种从显式目标密度采样的扩散模型负引导新方法，利用正-未标记学习在线估计样本属于不良条件的概率，从而实现渐近精确、偏差可通过增加计算量系统降低的负引导。

    

    扩散模型的负引导旨在生成避开某种不期望条件的样本，该条件既可以由参考样本指定，也可以由条件得分与无条件模型共同指定。现有方法在每个样本处注入由局部信息推导出的排斥项，因此所得采样器并不以某个显式密度为目标，其偏差也无法通过增加计算量而被系统地降低。与之相反，我们提出了一种从显式目标密度采样的负引导方法，该方法能够统一处理基于参考样本和基于得分的两种条件指定方式，并以一个参数化目标密度本身的回避强度来抑制不期望条件附近的概率质量。实现这一目标需要在每个扩散时间步估计每个样本属于不期望条件的可能性。我们将这一估计问题构造为一个正-未标记学习问题，该问题支持轻量级的在线训练。

    arXiv:2610.19802v1 Announce Type: new  Abstract: Negative guidance of diffusion models generates samples that avoid an undesired condition, specified either by reference samples or by a conditional score alongside an unconditional model. Existing methods inject repulsive terms derived from local information at each sample, so the resulting sampler does not target an explicit density, and its bias cannot be systematically reduced by increasing computation. We instead present a negative-guidance method that samples from an explicit target density, which uniformly handles both reference-sample and score-based specifications and suppresses probability mass near the undesired condition with an avoidance strength that parametrizes the target density itself. Realizing this target requires estimating, at each diffusion timestep, how likely each sample is to belong to the undesired condition. We cast this estimation as a Positive-Unlabeled learning problem that admits lightweight online trainin
    
[^42]: 面向未测量混杂下因果效应估计的邻近平衡方法

    Proximal Balancing for Causal Effect Estimation under Unmeasured Confounding

    [https://arxiv.org/abs/2609.40051](https://arxiv.org/abs/2609.40051)

    提出邻近平衡方法，将经典协变量平衡思想扩展到仅能通过代理变量观测的混杂因素，无需指定代理角色、求解逆问题或依赖潜变量模型，即可实现未测量混杂下的因果效应估计。

    

    从观测数据中估计因果效应是科学与政策领域的核心问题，但当混杂因素未被测量时，因果效应无法被识别。邻近因果推断通过未测量混杂因素的代理变量来应对这一问题。然而，现有的基于代理的方法要么指定代理角色并求解一个逆问题，该问题是不适定的，且在高维代理变量下难以估计；要么使用潜变量模型，该模型假设学习到的潜变量与隐藏的混杂因素相匹配，一旦不匹配就会产生偏差。为了应对这些挑战，我们提出了邻近平衡方法。它将经典的协变量平衡思想推广到只能通过代理变量观测的混杂因素：它学习协变量和代理变量的一个低维摘要表示，使处理组之间具有可比性，然后对该摘要进行调整。该方法不需要指定代理角色、求解逆问题或使用潜变量模型。我们给出（原文摘要在此处截断）

    arXiv:2609.40051v1 Announce Type: new  Abstract: Estimating causal effects from observational data is central to science and policy, but the effects are not identified when confounders are unmeasured. Proximal causal inference addresses this problem with proxies of the unmeasured confounders. However, existing proxy-based approaches either designate proxy roles and solve an inverse problem, which is ill-posed and hard to estimate with high-dimensional proxies, or use a latent-variable model, which assumes that the learned latent variable matches the hidden confounder and leaves bias when it does not. To address these challenges, we introduce proximal balancing. It carries the classical idea of covariate balancing to confounders that are observed only through proxies: it learns a low-dimensional summary of the covariates and proxies that makes the treatment groups comparable, and then adjusts for this summary. It needs no designated proxy roles, inverse problem, or latent model. We give
    
[^43]: 基于变换低秩分位数曲面的因果发现

    Causal Discovery via Transformed Low-Rank Quantile Surfaces

    [https://arxiv.org/abs/2609.16931](https://arxiv.org/abs/2609.16931)

    本文提出低秩分位数曲面（LRQS）二元因果模型，证明在因果方向上变换后的条件分位数曲面具有低秩结构从而保证因果方向的通用可识别性，并提出了一种交替进行秩约束近似与单调变换保序估计的非参数因果发现方法。

    

    我们提出了低秩分位数曲面（LRQS），这是一种二元因果模型，在因果方向上，条件分位数曲面的未知单调变换具有低秩函数分解形式。LRQS 涵盖了位置-尺度噪声模型和后非线性异方差噪声模型，同时允许多个分位数基函数来表示超出位置-尺度效应的分布变化。我们证明了 LRQS 的通用可识别性：变换后的分位数曲面在因果方向上是低秩的，而在相应约束下，反方向的可表示性仅出现在例外的、经过精细调整的原因边缘分布中。我们提供了一个简单而强大的因果评分方法，采用非参数拟合程序，在离散化分位数曲面的秩约束近似与未知单调变换的保序估计之间交替进行。在具有更高秩分布变化的合成机制上的实验表……（原文摘要在此处被截断）

    arXiv:2609.16931v1 Announce Type: cross  Abstract: We propose Low-Rank Quantile Surfaces (LRQS), a bivariate causal model in which, in the causal direction, an unknown monotone transformation of the conditional quantile surface admits a low-rank functional decomposition. LRQS subsumes location-scale noise models and post-nonlinear heteroscedastic noise models, while allowing multiple quantile bases to represent changes beyond location-scale effects. We prove generic identifiability of LRQS: the transformed quantile surface is low rank in the causal direction, whereas reverse representability under the corresponding constraints occurs only for exceptional, fine-tuned cause marginals. We provide a simple-yet-powerful causal score using a nonparametric fitting procedure that alternates between rank-constrained approximation of discretized quantile surfaces and isotonic estimation of the unknown monotone transformation. Experiments on synthetic mechanisms with higher-rank distributional sh
    
[^44]: 分层指数-高斯混合模型用于观看时间分布预测

    Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction

    [https://arxiv.org/abs/2608.23356](https://arxiv.org/abs/2608.23356)

    本文提出了一种分层指数-高斯混合模型，通过分层分解和正则化技术解决了现有EGMN模型的方差崩溃和组件冗余问题，从而提升了短视频推荐中观看时间预测的准确性。

    

    准确的观看时间（WT）预测是短视频推荐的重要需求。然而，WT分布具有接近零膨胀、长尾和多模态的特点。最近的指数-高斯混合网络（EGMN）对完整的条件WT分布进行建模，而非单一的点估计，并达到了最先进的性能。我们的大规模复现研究表明，EGMN容易遭受方差崩溃、组件冗余和无效组件的问题。我们提出了一种分层指数-高斯混合（HEGM）模型，通过分层跳过观看分解、基于KL的方差正则化、结构化初始化、去除强制高斯偏移和熵正则化来解决这些失败模式。在公开和大规模工业数据集上，HEGM提高了排序准确性和阈值事件预测，同时保持了竞争力的点估计准确性，并显著提升了整体性能。

    arXiv:2608.23356v1 Announce Type: new  Abstract: Accurate watch-time (WT) prediction is an important requirement for short-video recommendations. Yet WT distributions are near-zero-inflated, long-tailed and multimodal. The recent Exponential-Gaussian Mixture Network (EGMN) models the full conditional WT distribution rather than a single point estimate and achieves state-of-the-art performance. Our large-scale reproduction study reveals that EGMN is vulnerable to variance collapse, component redundancy, and inactive components. We propose a Hierarchical Exponential-Gaussian Mixture (HEGM) model that addresses these failure modes through a hierarchical skip-watch decomposition, KL-based variance regularization, structured initialization, removing the forced Gaussian shift and the entropy regularizer. Across public and large-scale industrial datasets, HEGM improves ranking accuracy and threshold-event prediction, while maintaining competitive point-estimation accuracy and substantially im
    
[^45]: 学习重塑内部表示中的幂律各向异性

    Learning reshapes power-law anisotropy in internal representations

    [https://arxiv.org/abs/2608.15239](https://arxiv.org/abs/2608.15239)

    该论文通过精确求解两层线性网络的学习动态，揭示了幂律各向异性在内部表示中的形成机制，并发现特征学习机制下指数呈现非单调且多阶段演化，而惰性机制下行为不同。

    

    摘要：在从最先进的语言模型到小鼠大脑皮层等广泛的生物和人工神经系统中，内部表示中的幂律各向异性已被观察到。这种各向异性是高维信息处理的关键几何特性，并支撑着多种理论分析。然而，其从输入结构和任务驱动学习中涌现的机制仍不清楚。在此，我们通过精确求解一个宽两层线性神经网络在教师-学生设置下，具有幂律输入和教师结构的学习动态，来刻画这一形成过程。我们表明，在特征学习机制中，内部表示谱的局部幂律指数在训练过程中非单调演化，并在模式和训练时间上表现出多达四个不同的渐近区域。相比之下，在惰性机制中，该指数重新...

    arXiv:2608.15239v1 Announce Type: new  Abstract: Power-law anisotropy in internal representations has been observed across a wide range of biological and artificial neural systems, from state-of-the-art language models to the mouse cerebral cortex. This anisotropy is a key geometric property of high-dimensional information processing and underlies a variety of theoretical analyses. However, the mechanism by which it emerges from input structure and task-driven learning has remained unclear. Here, we characterize this formation process by exactly solving the learning dynamics of a wide two-layer linear neural network in a teacher--student setting with power-law input and teacher structures. We show that, in the feature-learning regime, the local power-law exponent of the internal-representation spectrum evolves nonmonotonically over the course of training and exhibits up to four distinct asymptotic regimes across modes and training times. By contrast, in the lazy regime, the exponent re
    
[^46]: 用于模型与污染联合推断的广义鲁棒贝叶斯方法

    Generalised Robust Bayes for Joint Inference of Model and Contamination

    [https://arxiv.org/abs/2607.25665](https://arxiv.org/abs/2607.25665)

    提出Hölder–Bayes方法，通过对缩放模型施加Hölder散度来构建模型参数与污染比例的联合广义后验，在无需指定离群点分布的情况下同时实现鲁棒参数推断与离群点检测，并提供了后验影响函数一致有界、有限样本n^{-1/2}风险界及Bernstein–von Mises定理等理论保证。

    

    鲁棒广义贝叶斯方法能够保护参数推断免受污染的影响，但通常无法推断污染的比例或识别具体的离群点。我们提出了Hölder–Bayes方法，该方法将Hölder散度应用于缩放后的模型，从而在模型参数与污染比例上构建联合广义后验分布，且无需指定离群点分布。每一次后验抽样会标记出与拟合模型最不相容的相应数量的观测值；对这些选择取平均，即可将联合不确定性传播到离群点检测得分中。在固定且不可忽略的污染下，我们建立了后验影响函数的一致有界性、后验平均超额风险的有限样本 n^{-1/2} 界，以及Bernstein–von Mises定理。离群点分布与拟合模型之间的尾部重叠程度控制着总体风险最小化器的偏差以及极限后验的相应变化。

    arXiv:2607.25665v2 Announce Type: replace-cross  Abstract: Robust generalised Bayesian methods protect parameter inference from contamination but do not generally infer its proportion or identify individual outliers. We propose H\"older--Bayes, which applies the H\"older divergence to a scaled model to construct a joint generalised posterior over model parameters and the contamination proportion, without specifying an outlier distribution. Each posterior draw flags a corresponding number of observations least compatible with the fitted model; averaging selections propagates joint uncertainty to outlier-detection scores. Under fixed, non-negligible contamination, we establish uniform boundedness of the posterior influence function, a finite-sample $n^{-1/2}$ bound on posterior-averaged excess risk, and a Bernstein--von Mises theorem. Tail overlap between the outlier distribution and the fitted model controls the population risk minimiser's bias and the associated change in limiting post
    
[^47]: 保留还是适应？持续学习的泛化

    To Retain or to Adapt? Generalizing Continual Learning

    [https://arxiv.org/abs/2607.05609](https://arxiv.org/abs/2607.05609)

    该论文挑战了持续学习以“保留知识”为中心的传统假设，将持续学习重新形式化为以最小化平均终身误差为目标的在线优化问题，并提出“迁移效率”作为衡量不稳定性与瞬态误差之间张力的定量指标。

    

    持续学习领域的文献长期以来一直以缓解灾难性遗忘为目标。这一目标建立在一个通常未被言明的假设之上：即终身学习者应逼近联合任务学习（JTL）的解，并保留所有先前获得的知识。我们挑战这一以保留为中心的前提，认为在非平稳环境中，优先考虑保留可能会阻碍实时适应。我们将焦点转向平均终身误差（ALE），将持续学习形式化为一个由环境动态与学习动态之间相互作用所支配的在线优化问题。我们引入迁移效率作为定量度量，用以刻画不稳定性（继承自相互冲突的过去经验的偏差）与瞬态误差（从头学习新任务所产生的优化成本）之间的张力。在温和的收敛条件下，该分解在线性模型和神经网络模型中均成立。

    arXiv:2607.05609v2 Announce Type: replace-cross  Abstract: The Continual Learning (CL) literature has long been driven by the goal of mitigating catastrophic forgetting. This objective rests on an often unstated assumption: that a lifelong learner should approximate the Joint-Task Learning (JTL) solution and retain all previously acquired knowledge. We challenge this retention-centered premise, arguing that in non-stationary environments prioritizing retention can impede real-time adaptation. Shifting the focus to the Average Lifelong Error (ALE), we formalize CL as an online optimization problem governed by the interaction between environmental and learning dynamics. We introduce Transfer Efficiency as a quantitative measure of the tension between Instability, the bias inherited from conflicting past experience, and Transient Error, the optimization cost of learning new tasks from scratch. Under mild convergence conditions, holding across linear and neural network models, this decompo
    
[^48]: 序贯稀疏高斯过程分位数回归

    Sequential sparse Gaussian process quantile regression

    [https://arxiv.org/abs/2606.31284](https://arxiv.org/abs/2606.31284)

    本文提出一种序贯稀疏高斯过程分位数回归框架，通过诱导变量和拉普拉斯近似实现高效后验推断，并将预测不确定性分解为条件先验与后验诱发两个分量，以此驱动诱导输入填充和数据采集两种自适应机制，实现计算资源的动态分配。

    

    分位数回归旨在从观测数据中估计响应变量的条件分位数。在贝叶斯框架下，高斯过程分位数回归能够提供不确定性量化，但由于非对称拉普拉斯似然函数的非共轭性以及后验推断的计算代价，该方法面临显著的计算挑战。我们开发了一个稀疏高斯过程框架，其中分位数函数通过缩减的诱导变量集合来表示，并使用拉普拉斯近似进行后验推断。随后，通过将预测不确定性分解为条件先验方差分量和后验诱发方差分量，驱动了两种互补的自适应机制：诱导输入填充和数据采集。这些机制被结合到一个序贯算法中，该算法将计算资源分配给预测不确定性的主要来源，并自适应地控制

    arXiv:2606.31284v2 Announce Type: replace  Abstract: Quantile regression aims to estimate the conditional quantiles of a response variable from observed data. In a Bayesian setting, Gaussian process quantile regression provides uncertainty quantification but faces significant computational challenges due to the nonconjugacy of the asymmetric Laplace likelihood and the cost of posterior inference. We develop a sparse Gaussian process framework in which the quantile function is represented through a reduced set of inducing variables and posterior inference is performed using a Laplace approximation. A decomposition of the predictive uncertainty into conditional-prior and posterior-induced variance components is then exploited to drive two complementary adaptive mechanisms: inducing-input infilling and data acquisition. These mechanisms are combined within a sequential algorithm that allocates computational effort toward the dominant source of predictive uncertainty and adaptively control
    
[^49]: 样本路径位于再生核巴拿赫空间中的高斯过程

    Gaussian Processes with Sample Paths in Reproducing Kernel Banach Spaces

    [https://arxiv.org/abs/2605.28106](https://arxiv.org/abs/2605.28106)

    该论文揭示了高斯过程与再生核巴拿赫空间中高斯随机元素之间的联系，证明此类空间上的协方差算子由对称正定函数唯一确定，并据此将Driscoll定理首次推广到了再生核巴拿赫空间。

    

    我们研究了高斯过程与再生核巴拿赫空间中的高斯随机元素之间的联系。我们证明了此类空间上弱二阶Radon概率测度的协方差算子由一个对称正定函数唯一确定。在高斯情形下，我们借助γ-可Radon化算子刻画了哪些函数可以由协方差算子产生。基于这些结果，我们将Driscoll定理推广到了再生核巴拿赫空间的情形。

    arXiv:2605.28106v2 Announce Type: replace-cross  Abstract: We investigate the connection between Gaussian processes and Gaussian random elements in reproducing kernel Banach spaces. We show that the covariance operator of a weak second-order Radon probability measure on such a space is uniquely determined by a symmetric positive-definite function. In the Gaussian case, we characterize those functions that arise from covariance operators in terms of $\gamma$-radonifying operators. Building on these results, we extend Driscoll's theorem to the reproducing kernel Banach space setting.
    
[^50]: 跨域注意力：超越词元级键值记忆

    Interdomain Attention: Beyond Token-Level Key-Value Memory

    [https://arxiv.org/abs/2605.24330](https://arxiv.org/abs/2605.24330)

    提出跨域注意力（IA）这一新型混合架构，在同一层内将键值信息压缩到SSM参数化的循环记忆中并隐式重建注意力矩阵，从而在保持SSM计算效率的同时近似点积注意力。

    

    基于Softmax注意力的Transformer和基于循环记忆的模型（如状态空间模型（SSM））构成了现代序列模型的基础。混合架构旨在结合注意力的表达能力与循环记忆的计算效率，然而到目前为止，常见的做法只是简单地将注意力层和循环层堆叠在一起。我们提出了跨域注意力（Interdomain Attention, IA），这是一种新颖的混合注意力架构，它在同一层中融合注意力和循环记忆。受softmax注意力核回归视角的启发，IA将值以及键的核特征压缩到由SSM参数化的循环记忆中。然后，IA通过从记忆状态中隐式重建注意力矩阵和值来近似点积注意力；重要的是，这种构建方式保持了SSM的计算效率优势。我们通过设计和预训练IA网络架构来展示IA的有效性……

    arXiv:2605.24330v2 Announce Type: replace  Abstract: Softmax attention-based transformers and recurrent memory-based models (e.g., state space models (SSMs)) form the foundation of modern sequence models. Hybrid architectures aim to combine the expressivity from attention and the computational efficiency from recurrent memory, yet so far common practices simply stack both attention and recurrent layers. We propose Interdomain Attention (IA) as a novel hybrid attention architecture that mixes attention and recurrent memory in the same layer. Motivated by a kernel regression view of softmax attention, IA compresses values and kernel features of keys into a recurrent memory parameterized by an SSM. Then IA approximates dot-product attention by implicitly reconstructing the attention matrix and values from memory states; importantly this construction maintains SSM's computational efficiency advantage. We demonstrate IA's efficacy by designing and pre-training IA network architectures, wher
    
[^51]: FRESH：基于信息几何的患者级模型对聚合证据的校准方法

    FRESH: Information-Geometric Calibration of Patient-Level Models to Aggregate Evidence

    [https://arxiv.org/abs/2605.16246](https://arxiv.org/abs/2605.16246)

    FRESH提出了一种基于信息几何的校准方法，通过对患者级生成模型进行最小扰动，使其匹配人群级汇总证据（如已发表试验和登记数据），从而实现患者级数据与聚合数据的统一、高效建模以支持临床决策。

    

    本文介绍了FRESH（近期证据与个体历史融合，Fusion of Recent Evidence and Subject Histories），这是一种将人群层面的汇总结果——包括已发表的临床试验、登记系统汇总数据、既往自然史研究以及经同行评审的间接比较——纳入基于患者级数据训练的预测模型的方法。该方法提供了一种有原则的方式，将患者级和聚合级两种数据类型结合到统一的、数据高效的模型中，以支持临床决策。FRESH假设可以访问在患者级数据源（如临床试验或真实世界数据）上训练的生成模型。该方法通过对模型进行重新校准，使其匹配目标人群的一组指定汇总统计量，从而产生患者级预测。这可以理解为聚合数据来源在患者层面的重现——其关键特性在于重新校准是对原始联合分布的最小扰动。

    arXiv:2605.16246v2 Announce Type: replace-cross  Abstract: This note introduces FRESH (Fusion of Recent Evidence and Subject Histories), a method for incorporating population-level summary results -- published clinical trials, registry summaries,   prior natural-history studies, and peer-reviewed indirect comparisons -- into predictive models trained on patient-level data. This method provides a principled means of combining both   patient-level and aggregate-level data types into a unified data-efficient model for clinical decision making.   FRESH assumes access to a generative model trained on patient-level data sources (e.g. clinical trial or real-world data). The method produces patient-level predictions from a re-calibrated   model that matches a set of specified aggregate statistics for a target population. This can be understood as a patient-level recapitulation of the aggregate source -- with the key property   that the recalibration is a minimal perturbation of the original jo
    
[^52]: 老虎机反馈下多变化点识别的样本复杂度

    The Sample Complexity of Multiple Change Point Identification under Bandit Feedback

    [https://arxiv.org/abs/2605.13252](https://arxiv.org/abs/2605.13252)

    该论文提出了一种在老虎机反馈下识别多个变化点的自适应算法，并证明了其样本预算的非渐近上下界，同时揭示在一般置信水平下样本复杂度不仅取决于跳跃幅度，先前仅由跳跃幅度决定复杂度的认识是不完整的。

    

    我们研究在老虎机反馈下的多变化点定位问题。一个定义在紧致区间上的未知分段常数函数可以在自适应选择的输入处被顺序查询，每次查询返回该函数的一个含噪评估值。目标是在目标精度 η 和置信水平 1-δ 下识别指定数量的不连续点（称为变化点），同时使用尽可能少的样本。我们提出了一种自适应算法，该算法首先检测可能包含变化点的区间，然后将这些变化点的位置精化到精度 η。我们建立了该算法样本预算的非渐近上界，并给出了相应的下界。先前的研究表明，当 δ→0 时，仅跳跃幅度决定了渐近样本复杂度。我们揭示了这一图景在该范围之外是不完整的。我们从实证和理论两方面证明，对于一般的 δ（摘要在此处截断）

    arXiv:2605.13252v2 Announce Type: replace-cross  Abstract: We study multiple change point localization under bandit feedback. An unknown piecewise-constant function on a compact interval can be queried sequentially at adaptively chosen inputs, and each query returns a noisy evaluation of the function. The goal is to identify a prescribed number of discontinuities, known as change points, within a target precision $\eta$ and confidence level $1-\delta$, while using as few samples as possible. We propose an adaptive algorithm that first detects intervals likely to contain change points and then refines their locations to precision $\eta$. We establish non-asymptotic upper bounds on its sample budget, together with corresponding lower bounds. Prior work shows that jump magnitudes alone determine the asymptotic sample complexity as $\delta\to 0$. We reveal that this picture is incomplete beyond this regime. We demonstrate, both empirically and theoretically, that for general $\delta$ and $
    
[^53]: 一种用于学习稳定二值表示的复合激活函数

    A Composite Activation Function for Learning Stable Binary Representations

    [https://arxiv.org/abs/2605.11558](https://arxiv.org/abs/2605.11558)

    提出了一种由sigmoid与tanh复合而成的重尾激活函数（HTAF），作为Heaviside函数的平滑近似，在零输入附近保持较大梯度且尾部梯度衰减更慢，从而实现了脉冲神经网络、二值神经网络和深度Heaviside神经网络的稳定梯度训练。

    

    激活函数通过塑造内部表示在神经网络中发挥着核心作用。近年来，学习二值激活表示因其在计算和内存效率以及可解释性方面的优势而受到广泛关注。然而，使用Heaviside（阶跃）激活函数训练神经网络仍然具有挑战性，因为其不可微性阻碍了标准的基于梯度的优化。在本文中，我们提出了重尾激活函数，它是Heaviside函数的一种平滑近似，能够实现基于梯度优化的稳定训练。我们将HTAF构造为sigmoid与双曲正切函数的复合函数，并从理论上证明它在零输入附近保持了较大的梯度质量，同时在尾部区域表现出更慢的梯度衰减。我们证明了脉冲神经网络、二值神经网络和深度Heaviside神经网络都可以得到稳定的训练。

    arXiv:2605.11558v3 Announce Type: replace  Abstract: Activation functions play a central role in neural networks by shaping internal representations. Recently, learning binary activation representations has attracted significant attention due to their advantages in computational and memory efficiency, as well as interpretability. However, training neural networks with Heaviside activations remains challenging, as their non-differentiability obstructs standard gradient-based optimization. In this paper, we propose Heavy Tailed Activation Function (HTAF), a smooth approximation to the Heaviside function that enables stable training with gradient-based optimization. We construct HTAF as a sigmoid hyperbolic tangent composite function and theoretically show that it maintains a large gradient mass around zero inputs while exhibiting slower gradient decay in the tail regions. We show that Spiking Neural Networks, Binary Neural Networks and Deep Heaviside neural Networks can be trained stably
    
[^54]: 具有任意时有效保证的AI系统自适应审计

    Adaptive auditing of AI systems with anytime-valid guarantees

    [https://arxiv.org/abs/2605.07002](https://arxiv.org/abs/2605.07002)

    提出了一个具有任意时有效统计保证的假设检验框架，使高度自适应的AI系统审计——即使样本量小、采样与停止规则在过程中灵活决定——也能得出统计学上严谨的结论。

    

    刻画生成式AI系统失效模式的一个主要瓶颈是标注和评估的成本与时间。因此，自适应测试范式日益流行：人们根据过往结果机会主义地决定标注哪些案例以及标注多少案例。虽然这一框架非常实用，但其极端灵活性使得难以得出统计学上严谨的结论，因为它违反了经典统计假设：观测数量通常有限（往往只有10到50个案例），且关于采样和停止的决策是在数据收集过程中临时做出的，而非基于预先指定的规则。为了刻画从高度自适应的审计中能够得出哪些统计推断，我们从两种“对峙”的视角引入了一个假设检验框架：(i) 模型的原假设，即断言不存在性能低于目标阈值的失效模式；与 (ii) 审计者……

    arXiv:2605.07002v2 Announce Type: replace-cross  Abstract: A major bottleneck in characterizing the failure modes of generative AI systems is the cost and time of annotation and evaluation. Consequently, adaptive testing paradigms have gained popularity, where one opportunistically decides which cases and how many to annotate based on past results. While this framework is highly practical, its extreme flexibility makes it difficult to draw statistically rigorous conclusions, as it violates classical assumptions: the number of observations is typically limited (often 10 to 50 cases) and decisions regarding sampling and stopping are made in the midst of data collection rather than based a pre-specified rule. To characterize what statistical inferences can be drawn from highly adaptive audits, we introduce a hypothesis testing framework from two "dueling" perspectives: (i) the model's null that asserts there is no failure mode with performance below a target threshold versus (ii) the audi
    
[^55]: 面向标签噪声分类的模型无关信息迁移与融合

    Model-agnostic information transfer and fusion for classification with label noise

    [https://arxiv.org/abs/2604.25845](https://arxiv.org/abs/2604.25845)

    本文提出了一个通用的模型无关非参数信息迁移与融合框架，用于带标签噪声的分类问题，克服了现有统计迁移学习方法因噪声与干净数据间分布偏移及参数化模型限制所带来的不足，并可广泛应用于各类分类器。

    

    标签噪声是现代机器学习中一个根本性的挑战，尤其是当大规模数据集通过自动化过程生成时。一种日益普遍且重要的数据范式，特别是在医学成像等领域，涉及从一个带有粗糙、噪声标签的大型数据集进行学习，并辅以一个小型的、经过专家验证的干净数据集。这一设定构成了一个典型的信息迁移与融合问题。然而，噪声数据与干净数据之间显著的分布偏移违背了现有统计迁移学习方法所依赖的核心总体参数相似性假设，同时它们对参数化模型的依赖也使其不适用于图像等复杂数据。为解决这些局限性，本文开发了一个通用的模型无关非参数框架，用于带标签噪声的分类，该框架适用于广泛的分类器类别。我们的方法利用小型的干净数据集……

    arXiv:2604.25845v2 Announce Type: replace-cross  Abstract: Label noise presents a fundamental challenge in modern machine learning, especially when large-scale datasets are generated via automated processes. An increasingly common and important data paradigm, particularly in domains like medical imaging, involves learning from a large dataset with coarse, noisy labels supplemented by a small, expert-verified, clean dataset. This setting constitutes a typical information transfer and fusion problem. However, the significant distribution shift between the noisy and clean data violates the core overall parametric similarity assumptions of existing statistical transfer learning methods, while their reliance on parametric models is ill-suited for complex data like images. To address these limitations, this paper develops a generic model-agnostic nonparametric framework for classification with label noise, which applies to a broad class of classifiers. Our approach leverages the small clean 
    
[^56]: 专家混合Transformer的泛化与缩放定律

    Generalization and Scaling Laws for Mixture-of-Experts Transformers

    [https://arxiv.org/abs/2604.09175](https://arxiv.org/abs/2604.09175)

    该论文为专家混合Transformer建立了泛化与缩放理论，证明其泛化误差由活跃参数预算主导并附加路由开销，且近似误差可通过扩展活跃容量或增加专家数量来降低，具体取决于主导瓶颈。

    

    我们为专家混合Transformer建立了一套泛化与缩放理论，该理论清晰地将每个输入的*活跃*容量与路由组合学分离开来。通过以固定路由模式为条件，并对这些模式进行并集界定，我们推导出了一个上确界范数覆盖数界限，其度量熵随活跃参数预算而扩展，并产生MoE特有的路由开销。结合针对平方损失的标准ERM分析，我们在d维流形数据模型和C^β目标函数下得到了一个泛化界限，表明一旦适当考虑活跃参数，近似与估计之间的权衡与稠密网络中的一致。我们进一步为MoE架构证明了一个构造性近似定理，表明在近似构造下，误差可以通过扩展活跃容量或增加专家数量来降低，具体取决于主导瓶颈。我们

    arXiv:2604.09175v2 Announce Type: replace  Abstract: We develop a theory of generalization and scaling for Mixture-of-Experts (MoE) Transformers that cleanly separates \emph{active} per-input capacity from routing combinatorics. By conditioning on fixed routing patterns and union-bounding across them, we derive a sup-norm covering-number bound whose metric entropy scales with the active parameter budget and incurs a MoE-specific routing overhead. Combined with a standard ERM analysis for squared loss, this yields a generalization bound under a $d$-dimensional manifold data model and $C^\beta$ targets, showing that approximation and estimation trade off as in dense networks once active parameters are accounted for appropriately. We further prove a constructive approximation theorem for MoE architectures, showing that, under the approximation construction, error can decrease either by scaling active capacity or by increasing the number of experts, depending on the dominant bottleneck. Fr
    
[^57]: 二元语义分割中模拟类别不平衡与标签偏移下的主动学习

    Active Learning under Simulated Class Imbalance and Label Shift in Binary Semantic Segmentation

    [https://arxiv.org/abs/2601.06209](https://arxiv.org/abs/2601.06209)

    该论文的贡献不是新的主动学习采集策略，而是提出了一个基于两个开源数据集的可控、可复现的模拟协议，用于系统研究类别不平衡与标签偏移这两种不平衡形式对二元语义分割中主动学习算法的影响。

    

    主动学习的目标是从未标注的数据集中挑选出信息量最大的样本。这在数据量大且标注成本高昂的场景中非常有用，例如机器视觉或医学成像领域。机器视觉有两个特点：其一，产生的图像大多没有缺陷；其二，产生的图像数量极其庞大，以至于我们无法存储所有采集到的图像。这一方面导致缺陷分布上存在强烈的类别不平衡，另一方面，有限的存储空间又可能引发标签偏移。为了理解这两种不平衡形式如何影响主动学习算法，我们基于两个开源数据集提出了一项模拟研究。我们的贡献并非提出一种新的采集策略，而是提供一个可控且可复现的协议，通过合成方式构建类别不平衡程度和标签偏移程度均被固定设定的数据集……

    arXiv:2601.06209v2 Announce Type: replace-cross  Abstract: The aim of Active Learning is to select the most informative samples from an unlabelled set of data. This is useful in cases where the amount of data is large and labelling is expensive, such as in machine vision or medical imaging. Two particularities of machine vision are first, that most of the images produced are free of defects, and second, that the amount of images produced is so big that we cannot store all acquired images. This results, on the one hand, in a strong class imbalance in defect distribution and, on the other hand, in a potential label shift caused by limited storage. To understand how these two forms of imbalance affect active learning algorithms, we propose a simulation study based on two open-source datasets. Our contribution is not a new acquisition strategy but a controlled and reproducible protocol that synthetically constructs datasets in which the levels of class imbalance and label shift are fixed a
    
[^58]: 协变量辅助的图匹配

    Covariate-assisted graph matching

    [https://arxiv.org/abs/2512.11761](https://arxiv.org/abs/2512.11761)

    本文提出两种利用节点和边协变量信息的带种子图匹配方法——基于二次分配问题的全局方法和利用种子邻域的可扩展局部方法，显著提升了图匹配在结构噪声和不可识别匹配情形下的灵活性与鲁棒性。

    

    数据集成通常需要在没有唯一标识符的情况下对齐数据集；对于网络数据，这通常通过图匹配来解决。尽管在许多应用中节点和边协变量广泛存在，但大多数现有方法仅依赖两个图之间的结构相似性，这限制了它们在不可识别匹配和结构噪声下的灵活性与鲁棒性。在本文中，我们研究了带种子的图匹配问题，即在已知一组节点（种子）部分对齐的情况下，同时利用信息丰富的节点和边协变量。我们提出了两种协变量辅助方法：一种基于二次分配问题（QAP）的全局方法，以及一种利用种子邻域的可扩展局部方法。两种方法都基于一个灵活的条件建模框架，能够使用从广义线性模型开始的各种预测模型来整合协变量。

    arXiv:2512.11761v2 Announce Type: replace-cross  Abstract: Data integration often requires aligning datasets without unique identifiers; for network data this is commonly addressed through graph matching. Despite the widespread availability of node and edge covariates in many applications, most existing methods rely solely on the structural similarity between the two graphs, limiting their flexibility and robustness to non-identifiable matches and structural noise. In this article, we study the seeded graph matching problem, where a partial alignment for a set of nodes (seeds) is known, in the presence of informative node and edge covariates. We propose two covariate-assisted methods: a global approach based on a quadratic assignment problem (QAP), and a scalable local approach that leverages the neighborhood of the seeds. Both methods are based on a flexible conditional modeling framework that enables covariate integration using predictive models ranging from generalized linear models
    
[^59]: 用于可解释多元时间序列异常检测的结构化时间因果性

    Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection

    [https://arxiv.org/abs/2510.16511](https://arxiv.org/abs/2510.16511)

    提出了OracleAD，一个简单可解释的无监督多元时间序列异常检测框架，它将每个变量的时间动态编码为因果嵌入，通过自注意力机制建模动态空间关系，并与代表正常状态的稳定潜在结构对齐以检测异常。

    

    现实世界中的多元时间序列异常是罕见的，且通常缺乏标签。此外，主流方法依赖于针对基准数据集调优的日益复杂的架构，只能检测到异常片段的部分内容，并夸大了检测性能。在本文中，我们提出了OracleAD，一个简单且可解释的无监督多元时间序列异常检测框架。OracleAD将每个变量的过去序列编码为单一的因果嵌入，用于联合预测当前时间点并重建输入窗口，从而有效建模时间动态。随后，这些嵌入经过自注意力机制，被投影到共享潜在空间中以捕获变量间的空间关系。这些关系并非静态，而是通过从每个变量的时间动态中涌现的特性来建模的。投影后的嵌入与代表正常状态关系的稳定潜在结构对齐。

    arXiv:2510.16511v2 Announce Type: replace  Abstract: Real-world multivariate time series anomalies are rare and often unlabeled. Additionally, prevailing methods rely on increasingly complex architectures tuned to benchmarks, detecting only fragments of anomalous segments and overstating performance. In this paper, we introduce OracleAD, a simple and interpretable unsupervised framework for multivariate time series anomaly detection. OracleAD encodes each variable's past sequence into a single causal embedding to jointly predict the present time point and reconstruct the input window, effectively modeling temporal dynamics. These embeddings then undergo a self-attention mechanism to project them into a shared latent space and capture spatial relationships. These relationships are not static, since they are modeled by a property that emerges from each variable's temporal dynamics. The projected embeddings are aligned to a Stable Latent Structure (SLS) representing normal-state relations
    
[^60]: 记忆长尾数据可以通过组合帮助泛化

    Memorizing Long-tail Data Can Help Generalization Through Composition

    [https://arxiv.org/abs/2510.16322](https://arxiv.org/abs/2510.16322)

    记忆长尾样本与特征组合能力相结合，可以使模型对训练中从未出现过的长尾特征组合的稀有测试样本做出正确预测，且这种组合能力依赖于模型架构。

    

    深度学习促使研究人员重新思考记忆与泛化之间的关系。在许多情况下，由于隐式正则化的存在，记忆并不会损害泛化能力，并且通过记忆长尾样本还可能有所助益。在本文中，我们考虑了记忆与简单组合之间的协同作用——即在长尾特征的组合上做出正确预测的能力。从理论上讲，我们证明了在线性设置中，记忆与组合相结合可以帮助模型在需要长尾特征组合的稀有测试样本上做出正确预测，即使这些组合在训练数据中从未出现过。在简单数据上针对神经网络架构的实验表明，这一理论见解可以推广到线性设置之外，我们还进一步观察到，模型的组合能力取决于其架构。

    arXiv:2510.16322v2 Announce Type: replace  Abstract: Deep learning has led researchers to rethink the relationship between memorization and generalization. In many settings, memorization does not hurt generalization due to implicit regularization and may help by memorizing long-tailed examples. In this paper, we consider the synergy between memorization and simple composition -- the ability to make correct prediction on a combination of long-tailed features. Theoretically, we show that for a linear setting, memorization together with composition can help the model make correct predictions on rare test examples that require a combination of long-tailed features, even if such combinations were never observed in the training data. Experiments on neural network architecture on simple data show that the theoretical insight extends beyond the linear setting, and we further observe that the composition capability of the model depends on its architecture.
    
[^61]: 熵正则化变分目标次优性的可计算度量

    A Computable Measure of Suboptimality for Entropy-Regularised Variational Objectives

    [https://arxiv.org/abs/2509.10393](https://arxiv.org/abs/2509.10393)

    本文提出可显式计算的核梯度差异（KGD）作为熵正则化变分目标次优性的度量，在贝叶斯情形下与核 Stein 差异（KSD）一致，从而使研究者在无法获得未归一化密度的情况下也能开发和比较新型采样算法。

    

    统计学和机器学习中的多种方法都以最小化熵正则化变分目标所得到的概率分布为目标。这种更高的灵活性带来了计算上的挑战，因为此时无法获得目标分布的显式未归一化密度。为缓解这一困难，我们提出了一种称为“梯度差异”的次优性新度量，特别是提出了一种可显式计算的核梯度差异。在贝叶斯统计的背景下，KGD 与核 Stein 差异相吻合，并且我们获得了 KSD 的一种新颖刻画，即它度量的是变分梯度的大小。在这一熟悉场景之外，即使无法获得未归一化密度，KGD 也使得新型采样算法的开发与比较成为可能。为说明这一点，我们提出并研究了若干新颖算法，包括对 Stein 的一种自然推广

    arXiv:2509.10393v5 Announce Type: replace-cross  Abstract: Several methods in statistics and machine learning target a probability distribution for which an entropy-regularised variational objective is minimised. This increased flexibility introduces a computational challenge, as one loses access to an explicit unnormalised density for the target. To mitigate this difficulty, we introduce a novel measure of suboptimality called 'gradient discrepancy', and in particular a kernel gradient discrepancy (KGD) that can be explicitly computed. In the Bayesian statistics context, KGD coincides with the kernel Stein discrepancy (KSD), and we obtain a novel characterisation of KSD as measuring the size of a variational gradient. Outside this familiar setting, KGD enables novel sampling algorithms to be developed and compared, even when unnormalised densities cannot be obtained. To illustrate this point several novel algorithms are proposed and studied, including a natural generalisation of Stein
    
[^62]: 合成数据能否重现流行病学中的真实世界研究发现？一项基于对抗性随机森林的复制研究

    Can synthetic data reproduce real-world findings in epidemiology? A replication study using adversarial random forests

    [https://arxiv.org/abs/2508.14936](https://arxiv.org/abs/2508.14936)

    该研究提出对抗性随机森林（ARF）作为合成表格型流行病学数据的高效便捷方法，并通过复制六项流行病学研究的统计分析，验证了合成数据能在保护隐私的同时可靠重现真实世界的研究发现。

    

    由于数据访问受限和隐私问题，合成数据在应对流行病学中的实际挑战方面具有巨大潜力。然而，目前许多方法存在质量有限、计算需求高以及对非专业人员而言过于复杂等问题。此外，常用的合成数据评估策略往往无法直接反映统计效用，也不能充分衡量隐私风险。在此背景下，一个关键且尚未被充分探索的问题是：合成数据能否在保护隐私的同时，可靠地重现流行病学研究的关键发现。我们提出对抗性随机森林（ARF）作为一种高效且便捷的方法，用于合成表格型流行病学数据。为评估其性能，我们复制了六篇流行病学出版物的统计分析，涵盖血压、人体测量、心肌梗死、加速度计测量、孤独感和糖尿病，数据来自德国的

    arXiv:2508.14936v4 Announce Type: replace-cross  Abstract: Synthetic data hold substantial potential to address practical challenges in epidemiology due to restricted data access and privacy concerns. However, many current methods suffer from limited quality, high computational demands, and complexity for non-experts. Furthermore, common evaluation strategies for synthetic data often fail to directly reflect statistical utility and measure privacy risks sufficiently. Against this background, a critical underexplored question is whether synthetic data can reliably reproduce key findings from epidemiological research while preserving privacy. We propose adversarial random forests (ARF) as an efficient and convenient method for synthesizing tabular epidemiological data. To evaluate its performance, we replicated statistical analyses from six epidemiological publications covering blood pressure, anthropometry, myocardial infarction, accelerometry, loneliness, and diabetes, from the German 
    
[^63]: 用于非平稳时间序列条件独立性检验的动态广义协方差度量

    The dynamic generalized covariance measure for conditional independence testing with nonstationary time series

    [https://arxiv.org/abs/2504.21647](https://arxiv.org/abs/2504.21647)

    本文提出了首个基于动态广义协方差度量的条件独立性检验框架，可对非平稳非线性时间序列的单次观测数据进行检验，能够检测出期望条件协方差在某些时刻非零的依赖关系。

    

    识别随机过程之间的关系是经济学等许多领域的核心目标。虽然标准的多变量时间序列分析工具具有诸多优势，但使用线性向量自回归模型往往难以捕捉非线性动态。这一困难推动了针对非线性时间序列的因果发现与变量选择方法的发展，这些方法通常采用条件独立性检验。在本文中，我们提出了首个适用于非平稳非线性过程单次观测序列的条件独立性检验框架。所提出的检验方法旨在针对期望条件协方差在某些时刻非零的备择假设具有检验功效。我们还讨论了一种能够对更广泛的备择假设获得检验功效的方法。我们框架的关键技术要素是时变非线性……（摘要在此处截断）

    arXiv:2504.21647v5 Announce Type: replace-cross  Abstract: Identifying relationships among stochastic processes is a core objective in many fields, such as economics. While the standard toolkit for multivariate time series analysis has many advantages, it can be difficult to capture nonlinear dynamics using linear vector autoregressive models. This difficulty has motivated the development of methods for causal discovery and variable selection for nonlinear time series, which routinely employ tests for conditional independence. In this paper, we introduce the first framework for conditional independence testing that works with a single realization of a nonstationary nonlinear process. The proposed test is designed to have power against alternatives in which the expected conditional covariance is non-zero for at least some times. We also discuss an approach for gaining power against a broader range of alternatives. The key technical ingredients of our framework are time-varying nonlinear
    
[^64]: 在高斯过程上下文老虎机优化中利用凹性信息

    Exploiting Concavity Information in Gaussian Process Contextual Bandit Optimization

    [https://arxiv.org/abs/2503.10836](https://arxiv.org/abs/2503.10836)

    本文提出一种将保持凹性的回归样条基与受凹性约束的高斯过程后验相结合的新型奖励模型，并据此构建UCB算法，从而在上下文老虎机优化中利用已知的凹性结构提升序贯决策的优化效率。

    

    上下文老虎机对序贯决策问题进行建模，其中奖励同时取决于所选动作和观测到的上下文。在医学、商业和工程等许多领域中，先验知识提供了关于奖励函数的结构信息，可以利用这些信息来提高优化效率。本文研究这样一种设置：对于每个可观测的上下文，条件平均奖励已知关于动作变量是凹的。为了利用这一结构，我们开发了一种上下文老虎机算法，该算法在凹性约束条件下对贝叶斯高斯过程后验进行条件化。我们提出了一种新颖的奖励模型，将保持凹性的回归样条基与受约束的高斯过程后验相结合，得到一个易于处理的形状约束估计器。基于该模型，我们构建了UCB算法，并为受约束后验建立了新的后验集中不等式。

    arXiv:2503.10836v2 Announce Type: replace-cross  Abstract: The contextual bandit models sequential decision-making problems in which rewards depend on both the chosen action and observed context. In many domains such as medicine, business, and engineering, prior knowledge provides structural information about the reward function that can be exploited to improve optimization efficiency. This paper studies settings where, for each observable context, the conditional mean reward is known to be concave with respect to the action variable. To leverage this structure, we develop a contextual bandit algorithm that conditions a Bayesian Gaussian Process posterior on concavity constraints. We propose a novel reward model that combines a concavity-preserving regression spline basis with a constrained GP posterior, yielding a tractable shape-constrained estimator. Building on this model, we construct a UCB algorithm and establish new posterior concentration inequalities for the constrained poster
    
[^65]: 优化共形选择：数据依赖评分构建后的强效选择性推断

    Optimized Conformal Selection: Powerful Selective Inference After Data-Dependent Score Construction

    [https://arxiv.org/abs/2411.17983](https://arxiv.org/abs/2411.17983)

    该论文提出了优化共形选择（OptCS）通用框架，突破了现有方法要求评分函数独立于校准和测试数据的限制，在数据依赖的评分构建和大量数据复用的情况下仍能保证有限样本的错误发现率（FDR）控制。

    

    我们研究共形选择中的数据驱动评分构建问题，该方法利用共形p值和多重检验，从无标签数据池中筛选出具有较大未观测标签的“有趣”实例，同时在有限样本下控制错误发现率（FDR）。现有解决方案要求评分函数的构建必须独立于校准数据和测试数据，然而，当面临众多评分选择和有限的标注数据时，更理想的做法是以数据驱动的方式选择或构建强有力的符合性评分，并利用所有标注数据进行评分训练。我们提出了优化共形选择，这是一个通用框架，允许在灵活的数据驱动评分构建之后进行有效的多重检验。我们给出了一般性条件，使得OptCS在存在大量数据复用的情况下仍能构建有效的共形p值，并处理复杂的p值依赖关系，以维持有限样本的FDR控制。

    arXiv:2411.17983v2 Announce Type: replace-cross  Abstract: We study data-driven score construction in conformal selection, which uses conformal p-values and multiple testing to select "interesting" instances with large unobserved labels from a pool of unlabeled data, while controlling the FDR in finite sample. While existing solutions require the score function to be constructed independent of the calibration and test data, it is desirable to select or construct powerful conformity score in a data-driven manner and to leverage all labeled data for score training when presented with many score choices and limited labeled data. We propose Optimized Conformal Selection (OptCS), a general framework that allows valid multiple testing after flexible data-driven score construction. We introduce general conditions under which OptCS constructs valid conformal p-values despite substantial data reuse and handles complex p-value dependencies to maintain finite-sample FDR control. We instantiate th
    
[^66]: 基于动态加权的100种语言多标签情感分析（针对标签不平衡问题）

    Multi-Label Sentiment Analysis on 100 Languages with Dynamic Weighting for Label Imbalance

    [https://arxiv.org/abs/2008.11573](https://arxiv.org/abs/2008.11573)

    该论文提出了一种动态加权方法来应对多标签情感分析中的标签不平衡问题，并结合适配的多标签焦点损失与线性时间复杂度的最优阈值选择方法，在100种语言的情感分析任务中于9个指标中的7个上达到了最先进水平。

    

    我们研究跨语言情感分析，该领域因其在市场研究、政治和社会科学等多个领域的应用而受到广泛关注。特别是，由于情感遵循普拉奇克情感轮模型，我们引入了一个多标签设置下的情感分析框架。我们提出了一种新颖的动态加权方法，在训练过程中平衡每个类别的贡献，这不同于以往根据类别频率分配固定权重的静态加权方法。此外，我们将单标签目标识别文献中偏向较难样本的焦点损失（focal loss）适配到我们的多标签设置中。进一步地，我们推导出一种在线性时间复杂度内选择最优类别特定阈值以最大化宏平均F1分数的方法。通过大量实验，我们证明我们的方法在3个不同（数据集）的9个指标中有7个取得了最先进的性能。

    arXiv:2008.11573v2 Announce Type: replace-cross  Abstract: We investigate cross-lingual sentiment analysis, which has attracted significant attention due to its applications in various areas including market research, politics and social sciences. In particular, we introduce a sentiment analysis framework in multi-label setting as it obeys Plutchik's wheel of emotions. We introduce a novel dynamic weighting method that balances the contribution from each class during training, unlike previous static weighting methods that assign non-changing weights based on their class frequency. Moreover, we adapt the focal loss that favors harder instances from single-label object recognition literature to our multi-label setting. Furthermore, we derive a method to choose optimal class-specific thresholds that maximize the macro-f1 score in linear time complexity. Through an extensive set of experiments, we show that our method obtains the state-of-the-art performance in 7 of 9 metrics in 3 differen
    
[^67]: 选择后泛化的极限

    The Limits of Post-Selection Generalization

    [https://arxiv.org/abs/1806.06100](https://arxiv.org/abs/1806.06100)

    本文揭示了事后泛化算法的能力极限，证明了任何满足事后泛化并回答自适应选择统计查询的算法其误差都存在紧的下界，为选择后数据分析的进展设置了一个根本性障碍。

    

    尽管统计学和机器学习提供了许多确保泛化的方法，但这些方法在存在自适应性的情况下往往会失效——自适应性是指一种常见的做法，即分析方式的选择取决于之前与同一数据集的交互。近期的一系列工作引入了功能强大的通用算法来确保事后泛化（也称为鲁棒泛化或选择后泛化），其含义是：给定算法的输出后，很难找到任何一个统计量使得数据与其来源总体之间存在显著差异。在这项工作中，我们展示了满足事后泛化的算法在能力上的若干局限。首先，我们对任何满足事后泛化并回答自适应选择的统计查询的算法的误差给出了紧的下界，这表明选择后数据分析的进展存在一个强大的障碍。其次，我们证明了事后泛化……

    arXiv:1806.06100v2 Announce Type: replace  Abstract: While statistics and machine learning offer numerous methods for ensuring generalization, these methods often fail in the presence of adaptivity---the common practice in which the choice of analysis depends on previous interactions with the same dataset. A recent line of work has introduced powerful, general purpose algorithms that ensure post hoc generalization (also called robust or post-selection generalization), which says that, given the output of the algorithm, it is hard to find any statistic for which the data differs significantly from the population it came from.   In this work we show several limitations on the power of algorithms satisfying post hoc generalization. First, we show a tight lower bound on the error of any algorithm that satisfies post hoc generalization and answers adaptively chosen statistical queries, showing a strong barrier to progress in post selection data analysis. Second, we show that post hoc genera
    

