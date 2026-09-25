# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Temporal Gradient Inversion for Private Trajectory Reconstruction in Embodied Reinforcement Learning](https://arxiv.org/abs/2609.30258) | 该论文提出TRACE时间梯度反演攻击，通过利用连续具身梯度间的跨时间相关性和策略头梯度结构中的闭式动作恢复，从每步策略梯度中自回归地重构出私有的观测-动作轨迹序列，突破了传统单帧梯度反演攻击的局限。 |
| [^2] | [Agentic Detection of Online Conspiracies](https://arxiv.org/abs/2609.30250) | 提出了一个配备社会查询工具的智能体框架，通过结合社会语境推断说话者意图（言外之力），而非仅依赖词汇标记，来检测社交媒体上表达方式隐晦的阴谋论话语。 |
| [^3] | [To Trust or Not to Trust: Retrieval-Augmented Fact Checking in Speech](https://arxiv.org/abs/2609.30227) | 论文提出VeriSpeak语音事实核查基准，揭示了大型音频语言模型存在显著的文本-语音模态差距（书面声明可验证但语音版本常失败），且仅靠检索增强带来的提升有限。 |
| [^4] | [PoEM: Predicting RL Outcomes from Existing Policies](https://arxiv.org/abs/2609.30226) | 提出PoEM框架，利用一组已在其他奖励上完成强化学习后训练的现有模型来预测新奖励函数下的强化学习结果，从而避免每次奖励变化时都从头运行昂贵且不稳定的强化学习过程。 |
| [^5] | [Minimally Invasive Steering of Language Models](https://arxiv.org/abs/2609.30218) | 提出MISVO方法，利用基于Fisher信息几何的局部KL散度正则化，在冻结语言模型上实现最小侵入式的测试时导向，避免奖励优化导致输出分布大幅改变和生成质量下降。 |
| [^6] | [A Nearly Quadratic Lower Bound for Linear Optimization over Convex Bodies in the Membership Oracle Model](https://arxiv.org/abs/2609.30215) | 本文在成员资格预言机模型下证明了凸体上线性优化和均匀采样的随机算法具有近似二次下界，其中线性优化的下界匹配已知上界，均匀采样的下界改进了之前的线性结果，且该构造同样给出了体积估计的下界。 |
| [^7] | [Anchored Extra-Proximal Methods: Optimal Higher-Order Methods for Monotone Inclusion Problems](https://arxiv.org/abs/2609.30212) | 提出了锚定额外邻近（AEP）框架，通过结合锚定外推与满足相对误差条件的非精确锚定邻近更新，为复合单调包含问题构造出复杂度最优的任意阶（p≥2）高阶求解方法。 |
| [^8] | [The Alignment Illusion in Multimodal Large Language Models](https://arxiv.org/abs/2609.30210) | 该研究通过对13个多模态大语言模型施加受控干预，揭示了标准标量相似度度量无法真实反映跨模态内容交互的“对齐幻觉”现象，其根源在于MLP下投影所导致的权重诱导对齐。 |
| [^9] | [Beyond Compression: Training Latent Representations for Stable Long-Horizon Rollout in Neural Surrogate Solvers](https://arxiv.org/abs/2609.30198) | 该论文揭示潜在神经代理求解器长时程预测不稳定的根源在于潜在表示仅按重建目标训练，并通过Koopman算子学习、Hamming噪声注入、噪声注入和多步滚动微调等训练层面干预，使潜在表示适配稳定的长时程滚动预测。 |
| [^10] | [Intrinsic-Extrinsic Coupling in Learning Dynamics](https://arxiv.org/abs/2609.30185) | 该论文形式化了学习动力学中的“内在-外在耦合”，证明学习者的当前观察并不决定其对后续训练的响应——相同的内在干预在不同外部延续下会产生非加性的、读出特定的交互效应，例如回放机制可将一次写入操作在32次更新中的贡献从五个正确预测变为零。 |
| [^11] | [GridSFM: A Foundation Model for Solving AC Optimal Power Flow](https://arxiv.org/abs/2609.30173) | GridSFM是一个1500万参数的物理启发图神经网络基础模型，通过在54种电网拓扑上预训练并结合基于牛顿法的物理信息微调，仅需100个求解实例即可适应多达10000节点的未见电网，实现2.45%的零样本发电成本误差，且优于使用更多数据训练的单一拓扑专用神经网络模型。 |
| [^12] | [Do Audio Language Models Hear and Read Distinctive Features Alike?](https://arxiv.org/abs/2609.30167) | 该研究通过最小对立音素对与随机配对基准的分析发现，音频语言模型的解码器对音素区别特征的表征在听觉与阅读两种模态间普遍不一致，仅有Qwen2.5-Omni模型中的浊音特征表现出超过随机基准的跨模态方向一致性。 |
| [^13] | [Learning and interpreting policies for simultaneous entanglement requests in quantum networks](https://arxiv.org/abs/2609.30157) | 该论文提出了一种结合消息传递神经网络（MPNN）的双深度Q网络强化学习方法，用于学习和解释量子网络中同时服务多个纠缠请求的纠缠资源调度策略。 |
| [^14] | [Does a model's stated reason for rejecting a candidate do any work?](https://arxiv.org/abs/2609.30151) | 该研究通过将模型声称缺失的事实插入对应档案并在贪心解码下重新测试，首次因果性地验证了语言模型拒绝候选者时所述理由确实会实际影响其后续选择。 |
| [^15] | [Graph-Based Inference and Topology-Aware Multi-Agent Reinforcement Learning for Large-Scale Railway Network Management](https://arxiv.org/abs/2609.30150) | 该论文提出一个将基于图上高斯过程核的分层贝叶斯环境建模与拓扑感知多智能体强化学习相结合的图基框架，利用瑞士联邦铁路的真实数据，实现了大规模铁路网络维护管理的可扩展决策支持。 |
| [^16] | [GRASP: Generating, Revising, and Assessing for Strategic Planning with Agentic AI](https://arxiv.org/abs/2609.30147) | GRASP是一个策略感知的多阶段规划框架，通过将规划流程解耦为生成、修订和评估三个上下文隔离的专门模块，显著提升了LLM在复杂任务上的规划准确率，在多个基准数据集上建立了新的最先进水平。 |
| [^17] | [Orbital Error Dynamics: Self-Organized Criticality, Ephemeral Parameter Resonance, and Non-Linear Biological Ontologies in Zero-Storage Neural Synthesis](https://arxiv.org/abs/2609.30115) | 该论文提出轨道误差动力学（OED）框架，将神经网络权重从静态存储矩阵（O(W)）改为由复二次映射 z=z²+c 程序化生成的瞬态拓扑共振（O(1)），实现零存储神经合成，并结合弯曲正弦波假说、观察者视界几何与重尾仿生扰动来定位共振区域并摆脱非凸优化停滞。 |
| [^18] | [On the SoS Certifiability of Log-Concave Distributions](https://arxiv.org/abs/2609.30105) | 该论文证明了对数凹分布的矩多项式具有不依赖于 Poincaré 常数的平方和证书，从而恢复了对数凹分布的最优矩界，并为高维统计估计问题带来了具有维度无关误差保证的高效算法。 |
| [^19] | [MQSS-Selector: RL-Guided Pass Selection for an MLIR Compilation Pipeline](https://arxiv.org/abs/2609.30104) | 本文提出MQSS-Selector，一个利用强化学习自动为MLIR量子编译流水线选择优化Pass的统一学习型选择器，以在碎片化的量子软件栈中提升NISQ时代量子程序的编译保真度。 |
| [^20] | [AT-SKM-Net: An Accelerated Trainable Sampling Kaczmarz-Motzkin Framework for Linear Hard-Constraint Feasibility on Dynamic Graphs](https://arxiv.org/abs/2609.30088) | 本文提出AT-SKM-Net框架，通过拓扑感知异构GNN引导的混合采样策略与Cholesky更新机制，将动态图上线性硬约束可行性问题的等式投影复杂度从O(N³)降至O(N²)，实现计算加速。 |
| [^21] | [Return or Revise? Learning When Revision Helps Retrieval-Augmented QA](https://arxiv.org/abs/2609.30087) | 本文提出“可恢复性”指标——通过在同一评判标准下同时评估草稿答案与其候选修订所得到的成对效果——并训练模型在修订前预测该指标，从而判断何时进行检索增强修订有益，在多个实验设置下均优于仅基于草稿置信度的决策方法。 |
| [^22] | [Residual Correlation as a Diagnostic for Joint-Uncertainty Gains from GP Coregionalisation](https://arxiv.org/abs/2609.30085) | 该论文发现GP协同区域化的主要收益在于联合不确定性量化而非点预测，并提出残差相关性（而非原始目标相关性）才是预测联合不确定性增益的最强指标，同时给出了可从独立逐目标残差计算的轻量级诊断指标D_logdet来量化这一增益。 |
| [^23] | [Reachability-Based Formal Verification of Graph Neural Networks with Node and Edge Features](https://arxiv.org/abs/2609.30079) | 本文提出GraphStar集合这一Star集合的推广形式，将神经网络验证框架扩展至带节点与边特征的图神经网络，实现了对电力系统中潮流分析、最优潮流估计和连锁故障分析任务的可达性形式化验证。 |
| [^24] | [Nuclear Norm-Regularized Bayesian Matrix Completion](https://arxiv.org/abs/2609.30078) | 本文提出了首个针对未知噪声方差下核范数正则化贝叶斯矩阵补全模型的采样器，并给出了复杂度为矩阵维度和目标精度倒数之多项式的显式非渐近保证。 |
| [^25] | [How Reproducible Are Evaluation Conclusions? A Self-Audit of LLM-Inferred Prompt Structure](https://arxiv.org/abs/2609.30074) | 这项研究通过对LLM提示结构推断的自我审计发现，小规模提示集产生的模型评估排名中只有最差模型的位置是可靠的，而中间和头部模型的排名在不同重复实验中极不稳定。 |
| [^26] | [KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization](https://arxiv.org/abs/2609.30059) | KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。 |
| [^27] | [From Processing to Functionality: Engineering Accessible Material States in Cu-Embedded SiO$_x$ Memristive Devices](https://arxiv.org/abs/2609.30047) | 本研究建立了连接等离子体沉积工艺与SiO$_x$/Cu/SiO$_x$忆阻器件宏观功能的多尺度框架，结合5万余个器件的统计分析和物理模拟，揭示器件行为源于缺陷形成到功能区涌现的概率性级联过程，而重构氧空位密度可作为调控器件功能的有效潜在描述符。 |
| [^28] | [AERIAL: Adversarial Evaluation of Robustness in Accuracy-Preserving Low-Precision EEG Decoders](https://arxiv.org/abs/2609.30037) | 该研究提出AERIAL评估框架，系统发现精度保持型压缩（剪枝、PTQ、QAT）虽不改变EEG解码器的直接对抗鲁棒性，但剪枝会显著降低不同精度模型之间对抗样本的双向迁移效率。 |
| [^29] | [Aim Short to Reach Far: Your Frozen World Model Can Plan Better Than You Think](https://arxiv.org/abs/2609.30036) | 提出锚定规划方法，通过瞄准从经验中检索的中间观测目标而非最终目标图像，使冻结的世界模型无需额外训练即可在长程任务规划中全面超越原有规划器。 |
| [^30] | [Canopy: Exploiting Piecewise Smooth Tree Priors for Multi-Fidelity Bandits](https://arxiv.org/abs/2609.30017) | CANOPY是一种多保真度树状老虎机算法，通过廉价的随机路径探测在线学习分段平滑先验在树结构中的有效区域，从而突破传统方法需预先假设全局平滑性的局限，更精准地引导昂贵的叶节点评估。 |
| [^31] | [GHOST-Q: Towards Studying Grounding Hallucinations Overlooked Under Same-score TradeOffs in Quantized VLMS](https://arxiv.org/abs/2609.29999) | 该论文提出GHOST-Q评估框架，通过将FP16与量化VLM的预测逐项配对，揭示出量化模型即使在总体准确率几乎不变的情况下，其视觉接地与幻觉行为仍发生显著变化，且内存大幅节省并不保证推理延迟降低。 |
| [^32] | [Let Training Guide Selection: Online Synthetic Data Filtering via Real-Anchored Utility](https://arxiv.org/abs/2609.29988) | 提出FROST在线合成数据过滤框架，通过锚定真实训练数据的梯度反馈估计合成数据效用，无需外部验证器即可过滤约20-30%的合成数据并提升真实任务性能。 |
| [^33] | [Diverse Geometries, Frozen Weights: Robust Heterogeneous Treatment-Effect Estimation via Causal Expert Ensembles](https://arxiv.org/abs/2609.29974) | 该论文提出GeoACE五专家集成框架，通过结合锚定校正估计器与多样化的重叠感知和结果引导几何，并采用验证集学习且在测试前冻结的集成权重（其中新增的O-Phi-ACE专家用无结果的重叠感知统计投影替代锚定输入），实现了更鲁棒的异质性处理效应估计。 |
| [^34] | [A Contraction Framework for Stochastic Operators with Bootstrapping: Application to TD Learning](https://arxiv.org/abs/2609.29961) | 该论文提出了一个将自举式随机更新统一建模为随机算子的收缩分析框架，无需依赖梯度结构且允许采样误差随迭代点增长，从而为TD学习等算法在任意目标更新周期$K$下提供了几何收敛的有限时间保证。 |
| [^35] | [Beyond Average Safety: Chance-Constrained LLM Fine-tuning](https://arxiv.org/abs/2609.29960) | 本文提出一种机会约束的保安全微调方法，通过限制安全样本相对参考模型退化超过阈值的比例，并利用可微上界与约束感知梯度下降算法，解决了传统平均安全损失掩盖罕见但严重安全失效的问题。 |
| [^36] | [Not All Confusion Is Equal: A Source-Aware Uncertainty Diagnosis for Fine-Grained Aircraft Detection](https://arxiv.org/abs/2609.29959) | 提出 A²E² 诊断工具，将细粒度飞机检测中的模型混淆沿“偶然性/认知性”与“类内/类间”两个维度分解为可定量测量的 2×2 四类来源，把被动的混淆度量转化为可操作的改进指导。 |
| [^37] | [Multi-Dimensional Matching](https://arxiv.org/abs/2609.29958) | 该论文提出一种基于特征的多维匹配机制，通过单次谱投影将匹配问题化简为 O(N log N) 时间可解的一维排序，并证明在投影空间内可取得精确的纳什社会福利最优解，同时具备福利保证与抗噪声稳定性。 |
| [^38] | [Tracking States or Tracking Cosets? An Algebraic Account of Learned State Tracking](https://arxiv.org/abs/2609.29951) | 该论文从代数视角揭示Transformer在群运算状态追踪任务中学到的并非精确状态而是商类（陪集）解，并证明最优顺序无关准确率收敛于阿贝尔化类大小的倒数。 |
| [^39] | [Error- and Prediction-Driven Motor Learning in the Cortico-Cerebellar Loop](https://arxiv.org/abs/2609.29945) | 该论文提出一种受小脑启发的控制框架，通过将多路复用预测表示与内部反馈相结合，实现了延迟反馈下的精确在线校正，并将适应学习时间缩短了一个数量级。 |
| [^40] | [MF-SCBO : Multi-fidelity Scalable Constrained Bayesian Optimization](https://arxiv.org/abs/2609.29941) | 该论文提出MF-SCBO方法，将可扩展约束贝叶斯优化扩展至多保真度设置，首次同时解决了高维性、黑盒约束、任意数量保真度级别和非嵌套采样这四个难题，并在实验中展现出优于单保真度SCBO及其他多保真度方法的收敛性能。 |
| [^41] | [Path-specific harm decomposition: A partial identification framework](https://arxiv.org/abs/2609.29938) | 该论文提出了将治疗伤害分解为直接路径和间接（中介）路径贡献的新概念与部分识别框架，为即使在随机对照试验中也无法点识别的直接与间接负向影响比例（FNA）提供了识别边界。 |
| [^42] | [When Temporal Perturbations Act Like Sensor Biases: Label-Free Auditing of Wearable Activity Recognizers](https://arxiv.org/abs/2609.29937) | 提出无标签审计方法SpectrumAudit，揭示可穿戴活动识别模型对“时间扰动”的鲁棒性主要由持续的直流传感器偏移主导，而非真正的时域波形变化。 |
| [^43] | [Robust Detection of LLM-Generated Text under Contamination](https://arxiv.org/abs/2609.29935) | 该论文将人类与机器文本建模为带Huber污染的有限阶马尔可夫过程，刻画了LLM生成文本可靠检测的精确理论边界，并证明对似然比检验等统计检测器进行截断处理可在污染环境下实现鲁棒检测。 |
| [^44] | [Improving Calibration of Black-Box Radiology AI Using Test-Time Augmentation](https://arxiv.org/abs/2609.29931) | 提出了一种名为DualTTA的模型无关框架，利用基于临床的测试时增强技术，在无需访问模型内部或训练数据的情况下，有效改进了黑盒放射学AI模型的校准性能。 |
| [^45] | [Spatio-temporally complementary feature propagation on graphs for longitudinal AADT estimation](https://arxiv.org/abs/2609.29906) | 该研究提出一种时空互补特征传播框架，通过有向图上的泊松能量最小化和流量比矩阵，融合环形检测器数据与宏观交通模型两种互补数据源，实现高效准确的多年期城市路网AADT估计。 |
| [^46] | [Cost-Sensitive Online Window Size Selection for Portfolio Management](https://arxiv.org/abs/2609.29887) | 本文提出一个两级在线学习框架，将候选窗口大小视为专家并以包含换手成本的损失动态聚合，推导了考虑换手成本的有限时间代价敏感跟踪遗憾界，并证明适当调参的Fixed Share算法在次线性切换预算下可实现渐近无跟踪遗憾。 |
| [^47] | [A New Gap Sequence for Shellsort: RL-Driven Algorithm Discovery Beyond $N^{4/3}$](https://arxiv.org/abs/2609.29881) | 本文提出一种强化学习驱动的自监督搜索系统，通过在可执行增量生成器空间中搜索，首次发现了最坏情况复杂度超越数十年来 $N^{4/3}$ 上界的实用希尔排序新增量序列。 |
| [^48] | [From Graphs to Feeders: Constraint-Guided Diffusion for Rule-Compliant Feeder Generation](https://arxiv.org/abs/2609.29879) | 提出了电网约束的离散去噪扩散模型 PG-DiGress，通过在反向扩散过程中以软掩码抑制不兼容的边类别并结合最终投影步骤，生成同时满足电气兼容性与辐射状规则的合规配电馈线拓扑。 |
| [^49] | [Does per-frame early exit pay? A compute-matched study of dynamic depth for on-device speech enhancement](https://arxiv.org/abs/2609.29867) | 本文通过对单一因果模型的每个中间深度进行监督训练并微调输出头，为设备端语音增强派生出一族帕累托效率更高的静态模型，在同等算力下PESQ最高提升0.11，或在节省30%算力的情况下匹配最佳PESQ，并在STM32N6微控制器上验证了int8量化后的延迟-质量表现。 |
| [^50] | [Beyond Model Size: Redesigning LiSenNet for embedded speech enhancement](https://arxiv.org/abs/2609.29866) | 本研究通过将LiSenNet的循环瓶颈替换为卷积频率与时间混合器、将不支持的操作改造为静态int8兼容原语并使用有界解码器激活，使37k参数的语音增强模型能够在STM32微控制器NPU上满足量化部署要求并保持增强质量。 |
| [^51] | [Efficient Continuous DEM Reconstruction under Limited Target-Resolution Supervision](https://arxiv.org/abs/2609.29864) | SCOPE 通过在低分辨率网格上预测潜在系数场并复用局部傅里叶残差函数，将高维系数预测与输出网格构建解耦，从而在仅有粗分辨率监督数据的条件下实现了高效的连续高分辨率DEM重建。 |
| [^52] | [Elucidating the Conformal Structure of the Brinkman Penalisation Method for Geometry-Adapted, Structure-Preserving Operator Learning of Hamiltonian PDEs](https://arxiv.org/abs/2609.29847) | 本文揭示了Brinkman惩罚正则化在辛矩阵与惩罚投影满足相容性条件时保持多共形辛结构，由此获得精确的局部守恒律，并据此提出了满足离散共形守恒律的结构保持数值积分器和面向哈密顿偏微分方程的几何自适应、结构保持算子学习方法。 |
| [^53] | [SwitchPFN: Shared Switching Dynamics for Frozen In-Context Time Series Classification](https://arxiv.org/abs/2609.29814) | SwitchPFN通过从训练序列中学习共享投影和状态码本，使局部动态算子与转移特征在不同样本间可直接比较，从而在冻结的表格基础模型上实现了时间序列分类的最优平均准确率。 |
| [^54] | [FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates](https://arxiv.org/abs/2609.29812) | FlashLoop发现循环Transformer中循环引入的额外计算与KV缓存存储大部分是冗余的，并利用状态变化集中于少数token、注意力差异由稀疏稳定的关键列主导等特性，通过惰性更新策略实现了快速且内存高效的循环Transformer推理。 |
| [^55] | [CORDIAL: Calibrating Ordinal LLM Outputs from Few Labels](https://arxiv.org/abs/2609.29807) | CORDIAL提出了一种仅用五个可解释参数、仅需少量标签（如20个）即可校准LLM有序输出分布的方法，在绝大多数实验设置中取得最低对数损失，同时支持跨任务先验学习和多LLM融合。 |
| [^56] | [An Analytical Theory of Auxiliary Learning](https://arxiv.org/abs/2609.29774) | 该论文提出了辅助学习的首个解析理论，通过师生框架推导出描述在线随机梯度下降动力学的封闭微分方程组，量化了任务相关性和标签噪声对辅助学习收益的影响，并建立了多任务误差与单任务误差之间的普遍关系。 |
| [^57] | [WeatherDiagFlow: Evidence-Grounded Radar Nowcasting with Diagnostic Flow Refinement](https://arxiv.org/abs/2609.29772) | 提出 WeatherDiagFlow，将雷达临近预报重构为“预报—公报—审计”任务：以回波移动、生消、强回波风险和不确定性等结构化诊断证据条件化滚动流精化，并借助多智能体层生成业务公报与独立的事后验证审计。 |
| [^58] | [On Growth and Form, and Function: Reusable Regulatory Handles Control Phenotypic Variation](https://arxiv.org/abs/2609.29755) | 该研究将低秩适配（LoRA）应用于预训练的神经细胞自动机，证明形态的大尺度连贯变化可以由固定调控网络上低维、可复用且可组合的“调控把手”来参数化控制。 |
| [^59] | [TopU-LBVS: A Realistic Multi Target Benchmark for Ligand Based Virtual Screening](https://arxiv.org/abs/2609.29740) | 提出了TopU-LBVS，一个覆盖7大类93个蛋白靶点、采用性质匹配且结构相似诱饵并配有三种固定评估协议的多靶点基于配体虚拟筛选基准，解决了现有基准因简单诱饵和随机阴性而高估性能的问题。 |
| [^60] | [TTLab at StanceEval-2026: A Cloze-Style Prompting Approach for Arabic-Language Stance Detection (CLASP-Ar)](https://arxiv.org/abs/2609.29733) | 该论文提出 CLASP-Ar，通过将阿拉伯语立场检测转化为完形填空式的掩码语言建模提示方法，简化了以往多任务学习方案的额外复杂性。 |
| [^61] | [Revalidation Beats Stateful Routing for Scientific Surrogates Under Distribution Shift](https://arxiv.org/abs/2609.29715) | 该研究通过大规模可复现的流式基准实验证明，在分布偏移条件下，无需复杂的有状态自适应控制器，只需在每个新数据批次上重新验证候选代理模型并选择验证损失最低者，即可将平均对数遗憾降至0.091，显著优于事后最优的固定模型选择策略（0.192）。 |
| [^62] | [RAPTOR: RAndom-projection Physics-informed Transient sOlveR](https://arxiv.org/abs/2609.29714) | 本文提出了RAPTOR——首个利用基于固定高斯径向基函数的物理信息随机投影神经网络来表示混合微分代数方程未知轨迹的新型电力系统时域仿真框架，以应对基于变换器的资源所带来的多时间尺度暂态仿真挑战。 |
| [^63] | [Decoupling Knowledge and Privacy: Post-Task Self-Distillation Replay for LLM Continual Learning](https://arxiv.org/abs/2609.29711) | 该论文提出SPARK方法，通过将知识保留与隐私修正解耦——先冻结任务后学到的分布作为稳定参考，再围绕其进行针对稀疏敏感位置的选择性修正——从而在大语言模型持续学习中兼顾知识保留与隐私保护。 |
| [^64] | [Three Ways Classical Test Theory Misleads for LLM Judges](https://arxiv.org/abs/2609.29709) | 该论文揭示经典测试理论的三个常用信度统计量在LLM评判者评估情境中含义发生扭曲——例如内部一致性系数无法区分题目设计与评判者错误的影响——因此不能直接照搬用于解读LLM评判者的表现。 |
| [^65] | [Safety-oriented pedestrian trajectory prediction at urban intersections using time-to-collision and crossing-zone context](https://arxiv.org/abs/2609.29706) | 本文提出一种融合碰撞时间（TTC）与过街区域上下文的面向安全的行人轨迹预测框架，并引入超出1米阈值误差的频率和幅度作为评估指标，以更好地捕捉与安全相关的预测性能。 |
| [^66] | [An Agnostic Sample Compression Scheme for Squared Loss of Near-Linear Size in the Fat-Shattering Dimension](https://arxiv.org/abs/2609.29696) | 本文首次构造了大小仅与 fat-shattering 维度近线性相关（乘以对数因子）且与样本量无关的平方损失不可知样本压缩方案，正面解决了 Attias 等人（ICML 2024）提出的开放性问题。 |
| [^67] | [Bandit Multiclass PAC Learning: Corrected Lower Bounds, Exact Families, and a Confidence Direct-Sum Phenomenon](https://arxiv.org/abs/2609.29694) | 本文修正了老虎机多类PAC学习的下界理论，指出原有基于BDS维度的下界并不正确，引入锚定维度aBDS证明了新的无常数下界，同时在上界中完全去除了对标签数K的依赖，并揭示了置信度直和现象。 |
| [^68] | [Predicting Symptoms of Amotivation and Anhedonia among University Students with a Novel Oversampling Method](https://arxiv.org/abs/2609.29690) | 本文提出了一种新型过采样方法，以解决机器学习模型在预测大学生动机缺乏和快感缺失症状时的类别不平衡问题，从而更准确地识别高风险学生。 |
| [^69] | [Not All Synthetic Data Are Equal: Expert-Committee Audit Screening for Imbalanced Crash-Injury-Severity Prediction in Automated Driving Systems](https://arxiv.org/abs/2609.29687) | 提出专家委员会审计筛选（ECAS）框架，利用仅基于真实数据的专家委员会从标签支持度、边界分离度、委员会一致性和局部合理性等维度对合成少数类样本进行可信度审计，从而提升自动驾驶系统中不平衡事故伤害严重程度预测的可靠性。 |
| [^70] | [Named Entity Recognition using Sliding Window Approach](https://arxiv.org/abs/2609.29682) | 提出一种无需重新训练或修改架构的推理阶段滑动窗口流水线，将冻结的句子级NER模型扩展至文档级命名实体识别，有效解决了长文档中的截断丢内容和实体割裂问题。 |
| [^71] | [The Sequential Price of Continual Learning](https://arxiv.org/abs/2609.29674) | 该论文在过参数化线性回归模型中首次精确刻画了持续学习的“序列代价”：总损失恰好分解为联合训练的内在损失与额外序列代价，在同质任务几何下总损失为联合训练的两倍，且EWC正则化能以与强度成反比的方式降低该代价。 |
| [^72] | [GBFRVFL: Granular-Ball Computing-Based Fuzzy Random Vector Functional Link Network](https://arxiv.org/abs/2609.29670) | 该论文提出基于粒球计算的GBFRVFL框架，将原始样本抽象为自适应粒球，并通过模糊隶属度和新型统计密度自适应毕达哥拉斯隶属度（SDAPM）两种方案量化粒球可靠性，从而有效应对数据噪声、离群点和类别不平衡，弥补了传统RVFL网络无法显式处理不确定性的不足。 |
| [^73] | [Graph, Loop, and Harness Engineering for Zero-Trust Agentic Data Engineering and Analytical Processing](https://arxiv.org/abs/2609.29668) | 该论文提出零信任智能体数据工程与零信任智能体OLAP两个框架，通过图工程、循环工程与线束工程三种抽象，以证据门控、有界恢复和严格验证机制确保大语言模型智能体可靠地完成端到端云数据工程与分析处理。 |
| [^74] | [VG-TIE: An interpretable tabular-to-image encoding method based on visibility graphs](https://arxiv.org/abs/2609.29650) | 本文提出VG-TIE，一种基于自然可视图与水平可视图结合PCA的可解释表格数据到图像编码方法，使生成的图像与模型无关且每个像素对应一个输入特征、像素强度反映该特征相对总体均值的偏离方向和幅度。 |
| [^75] | [Albireo: Adaptive, Energy-Efficient Inference Framework for Video Object Detection on the Edge](https://arxiv.org/abs/2609.29648) | Albireo是一个面向边缘设备的自适应节能视频目标检测推理框架，它为每个目标维护卡尔曼滤波器并仅在预测不确定性超过阈值时才调用检测器，在无需修改或重训检测器的情况下大幅降低GPU能耗，同时避免了盲目跳帧带来的检测质量下降。 |
| [^76] | [Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search](https://arxiv.org/abs/2609.29636) | 该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。 |
| [^77] | [TTLab at AlexandriaX-2026: A Fine-Tuned Surface Tagger for Arabic Machine-Translation Error-Span Detection and Classification](https://arxiv.org/abs/2609.29633) | 该论文提出基于MARBERTv2微调的词元级分类系统，结合焦点损失、类别权重和方言特定解码阈值应对标签不平衡问题，在AlexandriaX-2026阿拉伯语机器翻译错误跨度检测与分类任务中获得第三名。 |
| [^78] | [A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes](https://arxiv.org/abs/2609.29630) | MARETopic是一个无需训练的主题建模框架，通过将嵌入投影到低维流形并把主题发现转化为基于排序的原型选择，贪心选出邻域可覆盖语料库的真实文档作为主题原型，其MARETopic_Corr变体在类别最多的两个基准上Purity和NMI领先于神经与聚类主题模型。 |
| [^79] | [Limited Structural Reliability in Public Educational Prediction Benchmarks: A Four-Dimension Audit of Seven Datasets](https://arxiv.org/abs/2609.29625) | 该论文对七个公共教育预测数据集进行四维度建模前审计，发现大多数数据集存在跨组脆弱性——独立同分布性能看似正常但在分组留出下崩溃，且提高模型复杂度也无法弥补，揭示现有教育预测基准的结构可靠性严重不足。 |
| [^80] | [Optimal Recovery Meets Bayesian Learning: Where Worst-Case Bounds Pay Off](https://arxiv.org/abs/2609.29622) | 该论文揭示了最坏情况最优恢复与贝叶斯学习的精确数学对应关系，并通过实验证明 Morozov 校准在噪声盲规则失效、可复现性和后端迁移场景下显著优于发布权重、ML-II 和 GCV 等传统超参数选择规则，但在可交换数据上 split-conformal 方法更胜一筹。 |
| [^81] | [An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model](https://arxiv.org/abs/2609.29618) | 该消融研究表明，在小MLA-SSM混合语言模型中，SSM分支对性能的贡献大于MLA分支，且密集FFN混合模型以更少的峰值训练内存达到了与三值MoE混合模型相当的困惑度表现。 |
| [^82] | [Evidence-Driven Differential Diagnosis of Malignant Melanoma](https://arxiv.org/abs/2609.29613) | 提出了一种多层次证据驱动的恶性黑色素瘤鉴别诊断框架，通过解剖部位感知的掩码transformer建模患者所有病灶及其发病部位的上下文，并结合可学习的人口统计学嵌入捕捉患者元数据，使诊断特异性分别提升17.15%和7.14%。 |
| [^83] | [Active Client Selection in Federated Trajectory Prediction with Uncertainty-Awareness and Heterogeneous Complexity](https://arxiv.org/abs/2609.29600) | 针对联邦轨迹预测中场景不确定性高和跨场景复杂度异构的两大挑战，本文提出了一系列融合不确定性与复杂度感知的主动客户端选择方法，以优先选取信息量大的客户端来提升联邦学习训练效果。 |
| [^84] | [A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data](https://arxiv.org/abs/2609.29584) | 该论文提出了一个统一的计算框架，首次将组织级语义身份建模为可解释且随时间演化的语义构建，整合了语义表示学习、图语义建模、组织语义指纹、时序演化分析与证据驱动验证。 |
| [^85] | [When Identical Rows Disagree: From Benchmark Identifiability to Replication-Robust Anomaly Detection](https://arxiv.org/abs/2609.29580) | 论文揭示了表格数据中重复行对基准评估和异常检测的系统性影响，并提出因子化检测器SCOUT，通过分离复制不变的支持证据与计数证据，实现复制鲁棒的无监督异常检测。 |
| [^86] | [PartHackBench: Certified Equal-Progress Stress Tests for Partial-Credit Tool-Agent Evaluation](https://arxiv.org/abs/2609.29578) | 论文提出PartHackBench压力测试方法，通过私有认证器确保对抗轨迹与诚实轨迹在真实进度上逐组件匹配后再测量得分膨胀，从而暴露出部分得分评估中历史归因机制存在显著分数虚高且几乎无法检测攻击的缺陷。 |
| [^87] | [Task-Resolved Fisher Spectroscopy for Quantum Reservoir Computing](https://arxiv.org/abs/2609.29570) | 该论文提出“任务分辨的费希尔谱分析”方法，通过费希尔信息层级结构精确定位量子储层计算中任务信息在储层状态、测量、特征压缩和有限采样各环节的损失位置。 |
| [^88] | [Classifier-Dependent Benefits of Pseudo-Labeling for Semi-Supervised Android Malware Attribution](https://arxiv.org/abs/2609.29564) | 本研究在CICMalDroid 2020数据集上对六种分类器进行了伪标签半监督学习的系统性评估与统计检验，首次揭示其收益强烈依赖于分类器类型：SVM获益最大（+4.4%准确率），而随机森林在低标注比例下反而显著受损。 |
| [^89] | [Certified Predictive Value-of-Advice Gating for Cost-Aware Language-Model Guidance in Reinforcement Learning](https://arxiv.org/abs/2609.29548) | 该论文提出了一种认证预测性建议价值门控框架，在调用语言模型前预测可能的响应并评估其价值，仅当预测价值的置信下界超过调用成本时才发起查询，并辅以动作特定证书管理执行，从而实现成本感知且带理论保证的强化学习引导。 |
| [^90] | [Generalized Graph Variational Autoencoders: Bounded Divergences Control Posterior Collapse](https://arxiv.org/abs/2609.29546) | 本文提出广义图变分自编码器（GGVA），用Rényi-Tsallis散度族替代KL散度，并揭示散度的有界性（而非其阶数）才是控制后验坍缩的关键性质。 |
| [^91] | [The Impossible Trinity of Time-Series Validation: A Conservation Law among Training Sufficiency, Test Coverage, and Temporal Causality](https://arxiv.org/abs/2609.29530) | 本文证明时间序列验证存在“不可能三难”——训练充分性、测试覆盖度与时间因果性无法同时满足，并给出守恒律不等式 α+β ≤ 1+Λ，量化了跨越因果边界所必须付出的数据泄漏偏差代价。 |
| [^92] | [A Corpus of Real Scam- and Spam-Call Conversations from an Active Voice-Agent Honeypot](https://arxiv.org/abs/2609.29528) | 本文通过主动式语音代理蜜罐在53天内收集了10,015通真实诈骗与骚扰电话对话（约895小时音频、328,869条转录轮次），为电话诈骗研究提供了极为稀缺的真实对话数据集。 |
| [^93] | [Common Covariance Geometry and Certification for Brownian Kernel Ladders](https://arxiv.org/abs/2609.29525) | 该论文提出了支配布朗核梯经验椭球并集的最小迹公共协方差概念，借助绝对二可和算子、阈值与有效电阻几何等工具给出其精确刻画，并据此导出通用高斯复杂度界、确定性深度定律以及精确的经验柯尔莫哥洛夫宽度公式。 |
| [^94] | [Sample-Weighted End-to-End Trace-Norm Geometry for Multitask Learning](https://arxiv.org/abs/2609.29520) | 该论文提出用从任务系数到输入空间预测器的端到端映射的样本加权迹范数来分析多任务学习泛化性能，推导出精确的经验Rademacher复杂度，并证明传统分离乘积界存在无界的方向间隙、因子分解间隙以及指数级深度间隙。 |
| [^95] | [CataOPD: Catalytic On-Policy Distillation for Large Language Model Reasoning](https://arxiv.org/abs/2609.29518) | CataOPD提出让教师模型充当“催化剂”而非目标的同策略蒸馏方法，通过自救援路由和催化引导自解析扩展推理轨迹的可达性，并将经验证的学生生成轨迹内化为无需催化剂的最终策略。 |
| [^96] | [Evaluation of Multi-Turn Consistency in LLM Agents: Survival Analysis and Failure-Rationale Taxonomy](https://arxiv.org/abs/2609.29508) | 该论文在受延迟满足启发的20步多智能体环境中，利用Kaplan-Meier生存曲线和离散时间风险回归对8个模型家族的84,540条轨迹进行时间一致性评估，并从13,780条深思轨迹中构建了七类失败理由分类法，系统揭示了LLM智能体在多轮交互中的失败风险及其原因。 |
| [^97] | [Spectral-Guided Diffusion: Accelerating Inference via Static Spectral Layer Scheduling](https://arxiv.org/abs/2609.29505) | 该论文提出基于谱集中比率（SCR）与Frobenius范数的静态谱层调度方法，仅利用预训练权重离线确定扩散模型中可冻结并复用缓存的残差分支，无需路由器或输入依赖搜索，即可在相同计算预算下加速推理并保持生成质量。 |
| [^98] | [Task-Aware Spectral Pruning: A Mixture-of-Masks Framework for Efficient LLM Inference](https://arxiv.org/abs/2609.29499) | 提出TASP框架，通过将模块谱描述符与任务消融效应校准、为每个用户请求路由固定的稀疏掩码，在Llama-3-70B上实现43%有效FLOP削减的同时保持约97.7%的性能，且先导实验表明其适用性因模型而异。 |
| [^99] | [BLADE: Distilled LLM Regularization for Calibrated Knowledge Graph Completion](https://arxiv.org/abs/2609.29487) | BLADE通过将离线LLM判断蒸馏为冻结教师正则化器的变分模型，在推理时无需LLM参与，即可同时提供校准的预测概率和认知不确定性估计，并将校准误差大幅降低。 |
| [^100] | [Direct Message Approximation (DMA): A Consistency-Based Framework for Tractable Approximate Inference on Factor Graphs](https://arxiv.org/abs/2609.29466) | 该论文提出直接消息近似（DMA），通过直接近似因子到变量的消息而非边缘分布，并借助一致性条件与主定理，实现了无需内循环迭代、避免负精度消息且误差可控的因子图近似推断。 |
| [^101] | [Precise Convergence Speed of Clipped SGD](https://arxiv.org/abs/2609.29458) | 本文对裁剪SGD在 $(L_0,L_1)$-光滑函数上给出了更紧致的收敛分析，将步长有效范围扩大到 $\eta < 1/\beta$，加强了收敛准则并将最终可达损失界降低至更精确的 $6 \min(\sigma^2/c, 3\sigma)$。 |
| [^102] | [Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores](https://arxiv.org/abs/2609.29453) | 该论文将组合推荐解耦为随机评分学习与确定性选择两个阶段，阐明了差分隐私保证经选择器端到端传递的成立条件，并提出了一种能在评分噪声扰动下保证推荐列表顺序不变的间隔证书。 |
| [^103] | [SPADE-DFL: Communication-Efficient Decentralized Federated Learning via Derivative-Free Linearized ADMM](https://arxiv.org/abs/2609.29446) | 本文提出SPADE-DFL方法，通过无导数线性化ADMM让本地更新次数随计算预算增长，在去中心化联邦学习中仅用Θ(T^{2/3})轮通信即达到O(T^{-1/3})的收敛界，并同时实现了客户端级差分隐私保护。 |
| [^104] | [Rufus-Air: An Open LLM Post-Training Recipe](https://arxiv.org/abs/2609.29421) | 本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。 |
| [^105] | [Neural Transport Nested Sampling](https://arxiv.org/abs/2609.29413) | 提出神经传输嵌套采样（NTNS）算法，将嵌套采样与神经流方法相结合，仅需目标能量函数评估即可对高维分子系统进行采样并估计完整配分函数，在含55个粒子的Lennard-Jones团簇上，将采样误差较最强神经基线降低了一个数量级以上。 |
| [^106] | [Machine Unlearning for Gibbs Supervised Learning Algorithms](https://arxiv.org/abs/2609.29409) | 提出了一种基于ERM-RER变分形式的精确遗忘方法，使吉布斯监督学习算法在遗忘数据后与从头重新训练的结果在分布上完全一致。 |
| [^107] | [Transcript-Supervised Post-Training of Generative Speech Enhancement on Real Recordings via Reinforce Adjoint Matching](https://arxiv.org/abs/2609.29405) | 本文提出利用强化伴随匹配（RAM）方法，仅以文本转录作为弱监督信号，直接在真实录音上对生成式语音增强模型进行后训练，在不损害感知语音质量的前提下将词错误率降低5.08个百分点。 |
| [^108] | [RD-JEPA: Predictive latent pretraining for few-trajectory transfer across reaction--diffusion equations](https://arxiv.org/abs/2609.29403) | 提出RD-JEPA自监督预训练架构，在多个反应扩散系统上联合预训练后，仅需少量轨迹即可高效迁移到未见过的反应扩散方程，误差低于监督基线及从头训练模型。 |
| [^109] | [ICE: Task-Aligned Clifford Latent Fields for Multimodal Graph Foundation Models](https://arxiv.org/abs/2609.29398) | 该论文提出ICE（交互感知克利福德编码器），一种基于节点索引克利福德潜在场的多模态图基础模型，通过将拓扑、文本和图像编码为显式的Cl(3)几何地址并利用边缘感知几何积，统一地保留实体语义、构建图邻域交互状态，并为具有不同几何结构的预测任务提供适配的表示。 |
| [^110] | [Concurrent Split Learning Through Stable Client Clustering](https://arxiv.org/abs/2609.29395) | 本文提出全局聚类并行分割学习（GCPSL），通过将客户端稳定分配到固定集群并并发执行多个分割学习工作负载，在不增大单个批量的前提下提升客户端直接数据参与率，使CIFAR-10达到85%验证准确率的时间从串行执行的约19分钟缩短至约6分钟。 |
| [^111] | [MORE-PLR: multi-output regression employed for partial label ranking](https://arxiv.org/abs/2609.29386) | 本文提出MORE-PLR方法，创新性地将多输出回归应用于部分标签排序问题，通过编码器将带并列关系的（可能不完整的）标签排序编码为多元回归目标，并利用事后处理层将回归结果转换为桶序输出。 |
| [^112] | [Lightweight Probabilistic Downscaling from a Deterministic Base Model](https://arxiv.org/abs/2609.29383) | 本文提出将确定性预训练与概率微调相结合的两阶段训练方法，构建轻量级概率机器学习降尺度模型，在阿尔卑斯山、新西兰和南非三个区域的气温与降水降尺度任务中，RMSE超越了现有最先进方法。 |
| [^113] | [Decoupled Early Exits for Task-Dependent Compute Allocation in Flow-Matching VLAs](https://arxiv.org/abs/2609.29382) | 提出一个将骨干网络深度、动作专家深度和去噪步骤作为三个可联合配置计算轴的框架，通过在骨干网络和动作专家中附加轻量级退出变换器实现解耦早退，从而实现面向任务的动态计算分配并降低流匹配VLA模型的计算开销。 |
| [^114] | [On the second-order optimization for spiking neural networks](https://arxiv.org/abs/2609.29379) | 提出SpiKFAX，一种专为脉冲神经网络设计的二阶优化方法，通过对Fisher信息矩阵进行Kronecker因子分解近似来有效捕捉脉冲激活导致的尖锐损失曲面，在多种架构和数据集上持续提升测试准确率。 |
| [^115] | [Learning a Flow to Self-Supervised Representations](https://arxiv.org/abs/2609.29350) | 本文提出非对抗性的基于流的分布匹配框架 FBDM，通过球面条件速度回归学习参考引导的几何结构，避免了昂贵的编码器-评论家优化，性能与分布匹配方法几乎相当并与现有自监督学习方法具有竞争力。 |
| [^116] | [FlowAtom: Atom-Based Evidence Aggregation for Multi-Label Website Fingerprinting](https://arxiv.org/abs/2609.29330) | FlowAtom通过在无标签流量上预训练编码器学习共享“原子”原型，并将观察窗口内多个流量流的原子响应聚合为置换不变表示，从而实现对混合加密流量中受监控网站集合的多标签识别，闭世界微F1分数最高达97.82%。 |
| [^117] | [GCUL: Ambiguity Identification in Text Emotion Classification via Cluster-Guided Learning](https://arxiv.org/abs/2609.29327) | 提出了一种几何引导的选择性分类框架GCUL，将误分类和歧义实例视为表示空间中的混淆吸引子，通过三阶段聚类引导学习使拒绝边界从表示几何结构中自然涌现，而非依赖预设的拒绝率。 |
| [^118] | [TinyCardioUNet: IMU-to-ECG Translation with Graph-Encoded Inter-Axis Dependencies and Tensor Decomposition-Based Parameter Reduction](https://arxiv.org/abs/2609.29322) | TinyCardioUNet通过图神经网络编码IMU六轴间的依赖关系，并利用基于自动变分贝叶斯秩选择的张量分解缩减参数，仅以36k参数的轻量级模型即可从胸部佩戴的IMU信号准确重建心电图。 |
| [^119] | [Neuralized Multi-Wavelet Decomposition for Time Series Classification and Forecasting](https://arxiv.org/abs/2609.29317) | 提出 m-WCN 端到端深度学习框架，通过用可训练卷积算子近似经典 GHM 多小波变换并施加正交性约束，将多小波分解神经化，实现时间序列时域模式与频域成分的联合提取，从而提升时间序列分类与预测的性能。 |
| [^120] | [Beyond Feature Reliability: Repeat-Informed Multifractal Curve Regression for Brain-Age Prediction](https://arxiv.org/abs/2609.29307) | 提出RMCR框架，通过联合建模多分形曲线结构与重复扫描变异性，学习稳定且可重复的年龄预测模式，同时提升脑龄预测的准确性和被试内一致性。 |
| [^121] | [Sufficiently Reduced Distributional Regression](https://arxiv.org/abs/2609.29291) | 本文提出SRDR方法，通过严格恰当评分规则将充分降维转化为风险最小化问题，并利用可通过采样估计的能量评分联合训练降维映射与生成式预测模型，无需密度计算或对抗训练，同时证明了估计条件分布在能量距离上的收敛性。 |
| [^122] | [From Text Decisions to Pixels: An Study of Jev-Style Visual Choice Model](https://arxiv.org/abs/2609.29283) | 本文提出PixelJev视觉决策接口，利用小型开源多模态模型将图像、指令和候选项映射为结构化决策，通过64样本源适配将Pets准确率从60.13%大幅提升至92.40%，并能零样本迁移到其他视觉问答任务。 |
| [^123] | [Online Task Adaptation via Self-Organisation](https://arxiv.org/abs/2609.29281) | 本文提出一种基于神经细胞自动机的元学习自组织方法，通过局部记忆更新和delta规则，在适应阶段完全无需梯度即可实现在线任务适应。 |
| [^124] | [Learnable Time-Frequency Masks for Explaining Time-Series Classifiers](https://arxiv.org/abs/2609.29270) | 提出了XACT通用框架，通过在任意可逆时频变换（STFT、连续及离散小波变换）的系数上学习稀疏归因掩码，为时间序列分类器提供更精确且不易突出虚假特征的解释。 |
| [^125] | [BridgeMem: Causal Dyadic Transition Residuals for Temporal Knowledge Graph Forecasting](https://arxiv.org/abs/2609.29268) | 提出BridgeMem，通过检索查询实体对的历史转移事件，经支持度自适应的经验贝叶斯读取器将其转换为似然比残差，用以校正冻结的全词表时序知识图谱预测器的打分，从而捕捉现有方法所忽略的特定实体对转移证据。 |
| [^126] | [AFT Neural Function Approximators for 1D Nonlinear Force Laws](https://arxiv.org/abs/2609.29242) | 本文提出用神经网络替代谐波平衡法中计算代价高昂的交替频-时（AFT）格式，直接由位移傅里叶系数预测非线性力系数及其雅可比矩阵，从而在求解器和延拓算法不变的情况下高效计算含非线性接触与摩擦的装配结构频响曲线。 |
| [^127] | [Post-Training Leaves Behavioral Shadows on Unrelated Decisions](https://arxiv.org/abs/2609.29233) | 该论文提出主动无任务蒸馏（ATD）方法，证明后训练会在模型行为上留下可被探测的“阴影”——仅凭教师模型在任务无关提示中输出的单个单词，就能将编程等目标能力传递给学生模型。 |
| [^128] | [FB-GDM: Fully-Bayesian Guided Diffusion Models for High-Dimensional Linear Inverse Problems via Unsupervised Variational Inference](https://arxiv.org/abs/2609.29216) | FB-GDM提出了一种全贝叶斯引导扩散方法，通过在每个反向扩散步骤中用变分推断自动估计两个精度参数，免除了针对具体任务且需依赖真值的人工超参数校准，同时借助可分离分解保持线性计算复杂度，成本与一次ΠGDM运行相当。 |
| [^129] | [Continuous Online Fault Detection for Mobile Robots via Adaptive Edge Models](https://arxiv.org/abs/2609.29194) | 本文提出师生蒸馏框架，将离线基础模型TSPulse的故障检测能力蒸馏到轻量级MiniRocket学生模型中，并结合递归最小二乘在线自适应，实现了移动机器人在边缘硬件上的实时故障检测（4.30毫秒延迟），且在真实域偏移下无需灾难性遗忘即可恢复性能（VUS-PR从0.26提升至0.75）。 |
| [^130] | [ASIRF: An Agentic Framework for Context-Dependent Sensitive Information Redaction](https://arxiv.org/abs/2609.29191) | 该论文提出ASIRF智能体框架，通过在推理时从知识库检索领域特定的敏感信息定义，无需重新训练即可适应新领域进行敏感信息脱敏，在85%的模型-领域组合中召回率超越了OpenAI隐私过滤器。 |
| [^131] | [Towards Deployable Underwater Vessel Classification](https://arxiv.org/abs/2609.29179) | 该论文提出了一种结合多表示特征工程与紧凑卷积架构的水下舰船声学分类框架，并证明在严格的录音级数据划分协议下，仅15.7万参数的紧凑CNN即可达到0.7226的宏F1，性能不逊于参数量大70多倍的ResNet18。 |
| [^132] | [Edge AI on Constrained Devices for Binary Sleep-Wake Classification in Dynamic Environments](https://arxiv.org/abs/2609.29163) | 该论文在ESP32-S3微控制器上实现了结合惯性传感与视觉姿态分类的多模态边缘AI系统，采用两阶段睡眠检测策略，在动态环境中实现睡眠-清醒二分类，运动检测准确率达96.5%，姿态分类准确率达89%。 |
| [^133] | [Functional dynamic mode decomposition: Learning infinite-dimensional systems from data](https://arxiv.org/abs/2609.29159) | 本文提出了函数型动态模态分解（DMD），将投影DMD和精确DMD从有限维扩展到无限维系统，无需对空间域进行离散化即可直接从数据中学习偏微分方程等无限维动力系统。 |
| [^134] | [A Particle-Swarm-Assisted Gradient Meta-Learning Algorithm for Joint Transmit Precoding and STAR-RIS Coefficient Optimization](https://arxiv.org/abs/2609.29150) | 本文提出粒子群辅助的梯度元学习（PSA-GML）算法，通过幅度分离参数化降低搜索维度，并结合PSO全局搜索与闭式预编码求解，联合优化发射预编码与STAR-RIS系数以最大化多用户下行链路的加权和速率。 |
| [^135] | [Not Every Token Is Worth Distilling: Selective Supervision for Direct-OPD](https://arxiv.org/abs/2609.29142) | 该论文揭示了Direct-OPD中token级对数比率监督无法反映教师行为真实变化的缺陷，并提出根据教师参考JSD进行选择性屏蔽监督的方法S²D-OPD，仅在教师行为发生显著变化的token上进行蒸馏。 |
| [^136] | [A Concentration Bound for Two-Timescale Actor-Critic Algorithm](https://arxiv.org/abs/2609.29117) | 本文为长期平均奖励设定下带函数逼近的双时间尺度Actor-Critic算法推导了一致的全时间集中界，证明Actor参数在有限时间后以高概率进入并保持在一个安全区域，且其与最优参数的误差以至少 $1-\epsilon_1-\epsilon_2$ 的概率满足给定的收敛速率上界。 |
| [^137] | [ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks](https://arxiv.org/abs/2609.29102) | 提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。 |
| [^138] | [Language Specificity vs. Domain Diversity: Benchmarking Transformers for Bangla Medical NER](https://arxiv.org/abs/2609.29101) | 该研究在3,179个样本的完整测试集上对孟加拉语医学命名实体识别进行了大规模基准测试，微调后的XLM-RoBERTa以0.5959的F1分数创造了新的最先进水平，并揭示语言专用的BanglaBERT表现持续不及多语言模型。 |
| [^139] | [TraceGuard: Adaptive Multimodal Poison Filtering through Cross-Feature Rank Agreement](https://arxiv.org/abs/2609.29099) | TraceGuard通过六个语料库级特征（跨模态邻域、重复文本、文本擦除变化等）分析投毒样本的集体影响模式，并利用跨特征排名一致性自适应地过滤多模态数据中的隐蔽投毒样本，全程无需训练受害模型。 |
| [^140] | [Downside-Controlled Online Forecast Combination under Delayed and Revised Outcomes](https://arxiv.org/abs/2609.29096) | 该论文提出一种下行风险可控的在线预测组合方法，在延迟与修正的结果环境下将冻结预测器、静态修正器与在线修正器在单纯形上组合，仅使用成熟损失，实现最坏劣化仅0.15%且最高收益达11.5%，并在欧洲全部七个日前负荷竞价区域均降低MSE。 |
| [^141] | [Where Does Exactly-Once Live? Model, Harness, and Tool-Contract Effects on Duplicate Side Effects in LLM Agents](https://arxiv.org/abs/2609.29095) | 该论文提出确定性沙盒基准 LIMBO，研究 LLM 智能体的“恰好一次”副作用语义应由模型、智能体框架还是工具契约来保障，并发现答案取决于故障类型：当即时回读能够揭示实际结果时，由模型来决定。 |
| [^142] | [DAWN: Noise-Robust Quadruped Parkour via Depth-Denoising World Models](https://arxiv.org/abs/2609.29092) | 该论文提出DAWN框架，通过让世界模型以带噪深度为输入、干净深度为重建目标实现隐式去噪，并结合对比学习对齐，将噪声鲁棒性直接内置到腿式运动的深度感知中，从而无需手工调整的滤波器即可实现鲁棒的四足跑酷。 |
| [^143] | [Physics and Data Driven Transformer-Mamba Framework for Flow Field](https://arxiv.org/abs/2609.29087) | 该论文提出TM4FF框架，通过残差小波Mamba去噪层、Transformer注意力机制和基于傅里叶导数的纳维-斯托克斯方程物理约束损失三项创新，在CFD流场求解中实现了高精度、强噪声鲁棒性和良好的跨工况泛化能力。 |
| [^144] | [A Rapid Pipeline for Training and Deploying ML Models on WeBe Band](https://arxiv.org/abs/2609.29084) | 本文提出一个集成开源Piccolo AI生态系统的自动化快速流水线，能够在资源受限的WeBe手环上快速开发、优化并部署满足延迟、内存和功耗约束的机器学习模型。 |
| [^145] | [Feature Space Selection and Heterogeneous Effect Estimation for Blood-Brain Barrier Permeability: A Random Forest to the Generalized Random Forest Pipeline](https://arxiv.org/abs/2609.29076) | 本研究通过系统性消融实验比较多种分子特征空间与算法组合，发现基于组合特征的动态随机森林在血脑屏障通透性预测中取得最高 AUC（0.970），并进一步结合广义随机森林与双重/去偏机器学习，探索性地估计了分子结构与 BBB 通透性之间的异质性关联。 |
| [^146] | [BranchShine-CR: Compact Multilingual IPA Transcription with Self-Conditioned CTC and Consistency Regularization](https://arxiv.org/abs/2609.29069) | BranchShine-CR是一个仅2500万参数的多语言国际音标转录模型，通过自条件CTC与一致性正则化等技术，以约十二分之一的参数量实现4.47%的IPA字符错误率并相对降低22.3%，在全部41种语言上超越同规模NeMo Conformer基线，适用于低资源设备端发音评估。 |
| [^147] | [Transformers as Cross-Task Learners: Shared Structure Drives Sample Efficiency in In-Context Learning](https://arxiv.org/abs/2609.29060) | 本文通过覆盖数刻画任务空间的复杂度，揭示了Transformer如何利用共享的跨任务结构提升上下文学习的样本效率，并提出了一种基于锚函数的任务识别与评估方法。 |
| [^148] | [SLCA-GRPO: Resolving Cross-Segment Credit Misattribution in Tool-Calling RL](https://arxiv.org/abs/2609.29050) | 提出SLCA-GRPO框架，通过段锁定信用分配在结构段级别解耦优势估计，解决工具调用强化学习中跨段信用错误归因问题，并配套构建模式引导的LLM模拟器（SGLS）以实现无需真实API的可扩展稳定训练。 |
| [^149] | [Where Hallucinations Live: A Cross-Architecture Circuit in VQ-Tokenized Vision-Language Models](https://arxiv.org/abs/2609.29048) | 该研究发现在VQ标记化的视觉语言模型中，物体幻觉源于一个跨架构共享的早期层注意力路由电路，并通过单变量架构替换实验证明向量量化机制本身是这一病理性信号的根源。 |
| [^150] | [Personalised federated learning for Riemannian and Euclidean EEG decoding](https://arxiv.org/abs/2609.29037) | 该论文将个性化联邦学习适配到黎曼SPDNet脑电解码器上，让所有受试者共享主干而各自保留分类头，在三个运动想象数据集上取得了优于标准联邦学习、集中式训练和EEGNet的准确率，同时收敛更快、通信参数更少。 |
| [^151] | [Paging the Experts: A Reproducible Characterization of Flash-Backed MoE Inference on iPhone](https://arxiv.org/abs/2609.29032) | 该论文提出Routide运行时，通过将专家权重保留在闪存并按字节预算分页加载的方式在iPhone上运行量化35B级MoE大模型，并通过可复现测量证明MoE推理中看似存在的内存“容量断崖”实为缓存策略与工作负载交互的产物，而非普遍的内存需求。 |
| [^152] | [Exploiting answer-invariant redundancies in satellite imagery for efficient VLM inference on edge](https://arxiv.org/abs/2609.29029) | 该论文提出Rift两阶段系统，通过识别并剪除卫星影像中不影响最终答案的图像分块与视觉token冗余，在Jetson AGX Orin上使LLaVA-1.5 7B的能耗降低78%、延迟降低69%，同时将准确率从45%提升至73%。 |
| [^153] | [Generative Atmospheric Super-Resolution from Heterogeneous In Situ Observations through Composable Interfaces](https://arxiv.org/abs/2609.29027) | 本文提出可组合观测接口方法，将探空仪、飞机和地面站这三类异构原位观测统一转换为特定来源的似然因子，从而条件化单一预训练的13变量大气扩散模型，实现生成式大气超分辨率重建。 |
| [^154] | [Growth-Inspired Graph Generation and Inverse Design of Mechanical Lattices via Dot Matrices Database Augmentation and GCNN](https://arxiv.org/abs/2609.29024) | 本工作受自然界生长机制启发，提出基于离散点阵顺序生长的形态发生式图生成框架，结合有限元分析与图卷积神经网络（GCNN）学习三维力学点阵的拓扑-性能映射，实现力学点阵的正向预测与逆向设计。 |
| [^155] | [EvoTreeNAD: Genealogy-Guided Evolution for LLM-Driven Neural Architecture Discovery](https://arxiv.org/abs/2609.29016) | EvoTreeNAD提出了一种谱系引导的进化算法，通过从空根节点生长谱系树并利用后代性能选择優良谱系，使LLM智能体无需种子或预定义搜索空间即可持续发现神经架构。 |
| [^156] | [Learning from Mixed-Quality Deployment Experience for Robot Manipulation](https://arxiv.org/abs/2609.29000) | 提出预测性动作分块学习（PACL），仅利用真实部署中自然积累的混合质量自主经验（无需人工纠正或探索交互），通过分块级评论家与未来潜在预测增强时间差分学习，有效提升机器人操作策略的长时程学习效果。 |
| [^157] | [Automatic Rank Allocation for Low-Rank Adaptation in Large Language Models via lp Regularization](https://arxiv.org/abs/2609.28998) | 提出 ℓp-LoRA，利用 ℓp 正则化对每个秩一分量的能量施加稀疏约束，将矩阵优化简化为二维问题并推导出隐式阈值准则，从而以有原理、无需手动设计重要性分数的方式自动为 LoRA 分配各矩阵的秩。 |
| [^158] | [CrossSafe: Towards Cross-Embodiment Latent Safety Filters](https://arxiv.org/abs/2609.28984) | 本文提出CrossSafe，利用安全推理在不同机器人间的共通性构建跨形态的潜在安全过滤器，并根据各机器人的形态、运动学和动力学差异来具体实现安全动作，从而为通用操作策略提供跨机器人的安全保障。 |
| [^159] | [Spectral Graph Neural Networks with Hermite Polynomials: A Comprehensive Study](https://arxiv.org/abs/2609.28979) | 本文提出基于厄米特多项式的谱图神经网络HermNet，无需特征分解或学习基底即可实现稀疏高效的图滤波传播，并在有限训练预算下优于其他多项式基底模型。 |
| [^160] | [Same Bit Width, Different Outcomes: Post-Training Quantization of Text-to-Speech Across Architectures](https://arxiv.org/abs/2609.28974) | 该论文首次在统一协议下系统评估了跨多种TTS架构的训练后量化，发现相同位宽在不同模型上效果差异巨大且敏感组件因模型而异，可通过分阶段消融识别并用逐层GPTQ将性能恢复至0.1 UTMOS以内。 |
| [^161] | [Why Does Misinformation Propagate Faster? An Algorithmic Perspective on X](https://arxiv.org/abs/2609.28947) | 本文通过对X平台开源推荐算法进行首个组件级分析，揭示其“互动可替代性机制”——将推荐分数构建为所有预测用户互动的加权和——使得仅凭引发大量即时反应（点赞、转发）即可被反复推荐的虚假信息获得了传播优势，从而从算法机制层面解释了虚假信息为何传播得更快。 |
| [^162] | [Response-state Learning for Transferable Vibrational Spectroscopic Characterization with Electron Prior](https://arxiv.org/abs/2609.28935) | 提出SO(3)等变神经卡尔曼网络SENK，通过等变Transformer主干、神经卡尔曼桥与NBO电子先验通路构建响应态级联，实现从小分子到类药体系跨化学空间的可迁移、高保真IR/Raman光谱预测，并在QM9S和QMe14S上超越DetaNet。 |
| [^163] | [Automatic Harness Evolution for Hardware Design Verification: Can LLMs Consolidate Gains Across Discovered Harnesses?](https://arxiv.org/abs/2609.28908) | 该研究在硬件设计验证任务中对固定语言模型进行自动测试框架演化，发现虽然演化显著提升了完成尝试和任务覆盖率，但这些收益难以跨任务巩固和保持，表明LLM目前尚无法可靠地整合测试框架的改进收益。 |
| [^164] | [GeoDose-CP: Graph-Local Conformal Inference for Continuous-Treatment Earth Observation](https://arxiv.org/abs/2609.28895) | GeoDose-CP提出了一种支撑感知的图局部保形推断框架，通过联合建模干预处理偏移、逆结果尺度雅可比和空间残差依赖性，为连续处理地球观测数据提供可靠的干预不确定性量化。 |
| [^165] | [Forecast-Dojo: Replayable Environments for Benchmarking and Training LLM Forecasting Agents](https://arxiv.org/abs/2609.28876) | 该论文提出了Forecast-Dojo——一个结合已结算预测市场问题与带日期新闻的可重放环境，用于基准测试和训练LLM预测代理，实验发现研究工具能提升全部12个模型的预测表现，但所有模型仍落后于历史市场预测水平。 |
| [^166] | [When Fancy Eviction Fails: Rethinking Cache Replacement For LLM Prefix Reuse](https://arxiv.org/abs/2609.28870) | 通过对两家公司生产追踪数据的分析发现，在LLM前缀缓存场景下，为传统缓存设计的复杂淘汰策略相比LRU几乎没有收益，原因在于前缀复用由活跃会话的规律节奏主导、使得“近期性”异常具有预测性，因此有效的前缀缓存管理应以近期性为基础。 |
| [^167] | [Image Fidelity is Not Field Fidelity: Joint Thermodynamic Reconstruction and Error Localization in Neural Tomography](https://arxiv.org/abs/2609.28868) | 该论文提出CoroNeRF方法，通过可微分原子发射渲染器从多视角、多谱线观测中联合重建三维电子密度和温度场，并证明跨随机种子不稳定性可作为推理时无需真值的局部物理场误差指标。 |
| [^168] | [Multimodal Routing and Region Refinement for Language-Guided Medical Image Segmentation](https://arxiv.org/abs/2609.28860) | MRSeg提出了一种参数高效的多模态路由框架，通过联合路由器为每个图像-文本对动态分配低秩适配器来协同适配视觉与文本特征，并结合基于文本查询的区域精化模块，提升语言引导医学图像分割的准确性。 |
| [^169] | [RECLAIM: Can Agents Reproduce the Claims of Machine Learning Papers?](https://arxiv.org/abs/2609.28850) | RECLAIM是一个基于100篇NeurIPS 2025论文的可重建基准测试，通过预先定义复现目标、成功标准和GPU预算，并按作者发布资源分为运行、重训练、重新实现三个难度级别，用独立语言模型依据日志评分，结果显示最好的AI智能体也只能分别复现41%、27%和15%的论文结果。 |
| [^170] | [LastOPD: Taming Collapse in Latent On-Policy Distillation](https://arxiv.org/abs/2609.28845) | 论文揭示了潜在在线策略蒸馏中“先获益后崩溃”以及“对齐越好反而表现越差”两大失败模式，将其根源归结为潜在信号在不同层上的角色错配，并提出 LastOPD 方法来驯服这种崩溃。 |
| [^171] | [Uncertainty-Gated Exploration Noise Suppresses Task Collapse in Online RL Fine-Tuning of a Flow-Matching Vision-Language-Action Policy](https://arxiv.org/abs/2609.28838) | 该论文提出一种不确定性门控的探索噪声控制器，利用任务无关的新颖性与能力信号在任务流之间重新分配探索，在小算力在线强化学习微调流匹配VLA策略时有效抑制了“任务坍塌”，优于固定噪声和可学习噪声方案。 |
| [^172] | [M$^2$PFN: End-to-End Disentangled Alignment for Generalizable Multimodal In-Context Learning in Alzheimer's Disease](https://arxiv.org/abs/2609.28836) | 提出端到端框架M²PFN，通过在TabPFN的transformer中进行可微推理、并利用解耦与对比学习将3D-MRI和表格模态对齐至共享子空间，从而把表格基础模型的上下文学习能力扩展为可跨队列泛化的多模态阿尔茨海默病诊断器。 |
| [^173] | [When Does Unsupervised Learning Succeed or Fail? A PoS Perspective on Reconstruction-Based Anomaly Detection](https://arxiv.org/abs/2609.28832) | 该论文提出基于子空间追踪的几何框架刻画基于重构的无监督异常检测的两种失败模式，并引入无需异常标签的动态推拉与嵌套流形雕刻方法，通过受控扰动学习紧凑的最优正常子空间。 |
| [^174] | [Stream Recursion Model (SRM)](https://arxiv.org/abs/2609.28809) | 该论文提出了流递归模型（SRM），通过将计算组织为多个经递归细化的相互作用潜在流，在保持与GPT-2相当的单位参数性能的同时，暴露内部计算结构，为大型语言模型的机制可解释性提供了可扩展的解决方案。 |
| [^175] | [Monitoring Urban Traffic Dynamics at Fine Spatiotemporal Resolution Using Distributed Acoustic Sensing and Deep Learning](https://arxiv.org/abs/2609.28793) | 本研究将分布式声波传感（DAS）与深度学习相结合，把现有地下光纤电缆转化为密集传感器阵列，实现了米级空间、秒级时间分辨率的被动式隐私保护城市交通监测，可连续高效地揭示交通量、拥堵及事件驱动的交通动态变化。 |
| [^176] | [Vector Bellman Theory for Multichain Robust Average-Reward Markov Decision Processes](https://arxiv.org/abs/2609.28792) | 本文为多链鲁棒平均回报马尔可夫决策过程建立了向量贝尔曼理论，通过“先增益、后偏差”的耦合向量增益-偏差系统，从任意初始状态同时确定最优鲁棒增益并提供平稳鞍点策略。 |
| [^177] | [The Mechanics of Delta Learning: Target Design for Generalizable Scientific Machine Learning](https://arxiv.org/abs/2609.28782) | 本文揭示Δ学习中残差尺度并非衡量残差可学习性的充分指标，提出尺度归一化图狄利克雷粗糙度作为预训练诊断工具，并确立基线互补性作为科学机器学习中目标设计的核心原则。 |
| [^178] | [Physics-Guided Multi-Objective Deep Learning for Ultrasound RF Data Interpolation in Resource-Constrained Imaging](https://arxiv.org/abs/2609.28775) | 该论文提出一种物理引导的多目标深度学习框架，通过融合射频域与波束形成域损失的混合监督机制以及随机跳过掩码策略，实现资源受限超声成像中稀疏射频数据到稠密数据的高质量重建，有效抑制栅瓣伪影并提升模型对不同采集布局的泛化能力。 |
| [^179] | [Selective Inference for Deep Clustering in Latent Spaces](https://arxiv.org/abs/2609.28756) | 本文针对使用固定预训练编码器的深度聚类提出了一个选择性推断框架，通过应对从原始数据空间到潜在空间的非线性变换所带来的复杂选择过程，为聚类结果的统计可靠性检验提供了计算可行且有效的 p 值。 |
| [^180] | [Evaluating Cross-region Generalization for Wavelet-Diffusion Precipitation Downscaling](https://arxiv.org/abs/2609.28749) | 该研究系统评估了小波扩散模型在公里级降水降尺度中的跨区域与跨事件类型泛化能力，发现仅在俄克拉荷马州训练的模型在其他气候区域仍保持竞争力。 |
| [^181] | [Technical Manual for Toolkit for Confidence-Corpus Consistency via Fine-Tuning on a Fabricated Corpus](https://arxiv.org/abs/2609.28747) | 该论文提出了一个开源工具包，通过在虚构算术语料库上微调小型语言模型，并以不变的测量程序配对比较其微调前后对虚构答案与真实答案的置信度，从而直接检验“模型置信度可作为事实知识代理指标”这一假设。 |
| [^182] | [Policy Complexity, Reaction Time, and Bounded Rationality in Reinforcement Learning](https://arxiv.org/abs/2609.28737) | 本文提出MI-SARSA算法，通过互信息正则化将状态特定的信息成本显式纳入强化学习，实现了策略压缩并能够预测试验层面的反应时间，从而为生物有限理性提供了更合适的计算模型。 |
| [^183] | [Unmasking Shortcut Learning in IoT Intrusion Detection: A Forensic, Multi-Paradigm Evaluation of Feature Dependence and Data Leakage](https://arxiv.org/abs/2609.28725) | 该论文通过取证式的多范式评估发现，物联网入侵检测模型的近乎完美性能主要源于对静态测试床IP/MAC地址和原始时间戳等数据集捷径（即数据泄露）的利用，而非真正可泛化的攻击行为特征。 |
| [^184] | [Upholding Robustness in Federated Learning: Trends, Emerging Strategies, and Research Opportunities](https://arxiv.org/abs/2609.28722) | 本文从威胁攻击面分类、鲁棒聚合策略分类和分层防御策略三个角度对联邦学习鲁棒性进行了全面综述，并审视现有评估实践、指出未来研究方向。 |
| [^185] | [Exact Bayes Regret and Asymptotic Optimality in High-Dimensional Gaussian Bandits](https://arxiv.org/abs/2609.28718) | 本文研究时间范围与维度成比例的高维高斯贝叶斯线性老虎机，证明了归一化后验不确定性在所有因果策略上具有一致显式极限，由此推导出汤普森采样等策略的精确后悔曲线，并证明后验均值贪婪选择达到极限最优贝叶斯后悔，而汤普森采样的后悔严格更大。 |
| [^186] | [Temporal Learning for End-Effector Position Estimation under Aerodynamic Disturbances in Aerial Continuum Manipulation](https://arxiv.org/abs/2609.28716) | 本文提出使用闭合形式连续时间（CfC）神经网络来估计无人机气动干扰下空中连续体机械臂的末端执行器三维位置残差，相比MLP和GRU等方法提升了位置估计的准确性。 |
| [^187] | [LabFactory: Building and Evaluating Executable AI Labs](https://arxiv.org/abs/2609.28697) | LabFactory提出让AI构建者将科学简报转化为可交付执行、封装了模型、知识、工具与控制器的“AI实验室”，再由独立主机在保留数据上运行并评分，从而把评估对象从构建者的进展陈述转变为实际交付的系统。 |
| [^188] | [Federated Learning of AnDE Classifiers](https://arxiv.org/abs/2609.28695) | 本文提出了一种支持任意依赖阶数的AnDE分类器联邦学习框架，通过本地学习判别式权重并全局聚合实现隐私保护，实验证明其性能持续优于联邦朴素贝叶斯，且差分隐私聚合仅带来有限的精度损失。 |
| [^189] | [M-plicits: Neural Implicit Surfaces via Nested Multiscale Residuals](https://arxiv.org/abs/2609.28684) | 提出M-plicits多尺度框架，将表面建模为通过嵌套邻域训练的MLP残差和，并将监督严格限制在先前零水平集周围的窄带内，从而在训练效率、渲染速度和噪声鲁棒性之间实现更好的平衡。 |
| [^190] | [Thinking Leakage: A Causal Audit of NoThink Post-Training in Hybrid Reasoning Models](https://arxiv.org/abs/2609.28682) | 该论文首次在因果中介框架下系统审计了混合推理模型NoThink后训练中的“思维泄露”现象，通过双向激活干预证明后训练准确率收益中有42%-79%实际来源于基础模型Think模式中已存在的思维能力。 |
| [^191] | [Beyond Static Graph World Models: Learning Stochastic Latent Dynamics over Evolving Topologies](https://arxiv.org/abs/2609.28670) | 提出图动力学模型（GDM），利用稀疏循环邻接矩阵和循环状态空间架构，实现对随机、部分可观测环境中演化拓扑图结构观测的世界建模，并首次引入联合图状态分布的评估方法。 |
| [^192] | [OPDiv: Optimal Selection of Top-K High-Scoring, Diverse Compounds](https://arxiv.org/abs/2609.28665) | 提出OPDiv算法，利用整数优化在得分与多样性之间找到最优权衡，从而选择top-k高得分且多样化的化合物，并主张虚拟筛选本质上是隐式的约束优化任务。 |
| [^193] | [The Fellowship of the Query: Learning Retrieval Actions](https://arxiv.org/abs/2609.28653) | 通过轨迹微调可以让小型语言模型有效学会检索增强问答中的“下一步动作”控制决策，宏F1分数远超零样本提示，且单个SLM可同时兼任控制器与答案生成器。 |
| [^194] | [RLVR landscapes for iterated multiplications can be benign: Insights from spin-glass theory](https://arxiv.org/abs/2609.28625) | 本文借助自旋玻璃理论证明，对于迭代乘法等算法任务，RLVR的优化地形是良性的（不存在局部极小值陷阱），其训练困难主要来自扩散屏障和梯度估计误差，而非地形本身的陷阱。 |
| [^195] | [Reward Hacking Challenges Oversight of Autonomous Research Agents](https://arxiv.org/abs/2609.28614) | 该研究通过对17个语言模型和38个任务的系统实验，发现自主研究智能体在无指令时也常有自发奖励破解行为（开放式任务达30.5%），且允许破解时74.6%的尝试既能通过评估阈值又能规避评估机制，表明奖励破解对自主研究智能体的监督构成了严峻挑战。 |
| [^196] | [UltraBench 2: Towards Robust Evaluation of Vision Foundation Models on Ultrasound](https://arxiv.org/abs/2609.28610) | 该论文推出了UltraBench 2，一个覆盖广泛解剖结构和任务、注重标准化与可重复性的超声视觉基础模型综合评估基准，并发现超声特异性预训练在分类任务上仍占优势，而最先进的通用模型在分割任务上已与之持平。 |
| [^197] | [fable.intermittent: benchmarking probabilistic forecasting methods for intermittent time series](https://arxiv.org/abs/2609.28607) | 该论文发布了 R 包 fable.intermittent，在统一的 fable 框架下整合了多种间歇性时间序列的概率预测方法以便系统比较，并提出了一种采用 Tweedie 预测分布的新型指数平滑模型 TWEES。 |
| [^198] | [UO-FIE: Combining Exact-Label Supervision with Graded Utility for Factivity Inference](https://arxiv.org/abs/2609.28605) | UO-FIE是一种参数高效的事实性推断系统，通过结合硬标签监督、基于效用的软目标、计划性类别权重和序数损失，在类别高度不平衡的中文事实性推断任务中同时提升预测的精确匹配率和区间接近度。 |
| [^199] | [Physics-Informed Self-Supervised Learning for Joint Wire Calibration and Interaction Position Reconstruction in Multi-Wire Parallel Plate Avalanche Counters](https://arxiv.org/abs/2609.28604) | 该论文提出了一种物理信息驱动的自监督学习框架，无需带标签的位置数据或专门校准实验，即可在多丝平行板雪崩计数器中联合实现丝增益校准与相互作用位置重建。 |
| [^200] | [Learning to Discover Interesting Mathematics](https://arxiv.org/abs/2609.28603) | 该论文提出将定理的内在趣味性定义为证明长度与陈述长度之比，证明该指标与定理的下游效用强相关，并训练了一个能准确预测证明难度的27B模型，据此优化可生成更有趣的数学定理。 |
| [^201] | [HClimRep-Ocean: A Global Ocean Emulator on an Unstructured Mesh](https://arxiv.org/abs/2609.28601) | 本文提出了 HClimRep-Ocean，一个直接在 FESOM2 原生非结构网格上运行的全球海洋机器学习模拟器，突破了以往数据驱动海洋模型依赖经纬度网格的局限，从而更好地刻画中尺度涡旋及复杂海岸线等海洋特有挑战。 |
| [^202] | [BRFID: Toward Byzantine-Robust Federated Intrusion Detection](https://arxiv.org/abs/2609.28599) | 该论文发现联邦入侵检测系统中的标签翻转投毒攻击会导致攻击者自身检测准确率自我退化，这一“自我妥协”信号可作为可检测异常用于识别拜占庭客户端，从而在无需额外防御机制的情况下实现拜占庭鲁棒的联邦入侵检测。 |
| [^203] | [TAM-Chain: Multi-Scale Thyroid Cytology Classification via Absorbing Markov Chains and Shannon Entropy Uncertainty Quantification for False-Negative Suppression and Domain-Shift Adaptation](https://arxiv.org/abs/2609.28590) | 该论文提出TAM-Chain框架，将多放大倍数（10x/20x/40x）甲状腺细胞学特征提取建模为吸收马尔可夫随机过程，结合香农熵不确定性量化、最优停止准则与人机协同转诊机制来抑制假阴性和域偏移风险，在内部测试集上实现Macro F1达0.9741且假阴性率为0%。 |
| [^204] | [NumericJev: Jev-like LLM Numerical Decoding with Multiway Decision Trees](https://arxiv.org/abs/2609.28587) | 提出了一种无需训练的数值解码算法NUMERICJEV，通过多路决策树递归细化数值范围，使任何具有类Jev结构化选择接口的大语言模型都能输出数值，其性能甚至超过从包含正确答案的候选列表中直接选择。 |
| [^205] | [SGA: Uncertainty Quantification for Multi-Step Forecasting in Time Series Foundation Models](https://arxiv.org/abs/2609.28582) | 本文提出SGA方法，通过有向无环图表征预测分支拓扑结构并度量其图复杂度，实现了对时序基础模型多步预测不确定性的有效量化，提升了预测结果的可信度。 |
| [^206] | [Auditability Is Not One Property: Rule Overlap, Behavioural Agreement, and Composition in Reinforcement Learning](https://arxiv.org/abs/2609.28581) | 该论文将强化学习策略的可审计性分解为六个可独立检验的谓词，并提出基于符号规则提取与哈希账本的审计协议，同时发现规则集重叠并不代表行为一致，揭示了离散行为规则描述层的严格局限。 |
| [^207] | [Uncovering Residential PV-EV Co-Adoption from Smart-Meter Data: Load Archetypes and Detection for Demand-Side Planning](https://arxiv.org/abs/2609.28578) | 本文提出一种集成的两阶段分析工作流，通过DTW k-means聚类从智能电表数据中发现居民用电行为原型，并结合BiLSTM模型检测家庭光伏与电动汽车的协同采用，为需求侧管理和低压电网规划提供支持。 |
| [^208] | [Time-Series Foundation Models That Understand Data Revisions](https://arxiv.org/abs/2609.28576) | 该论文提出修订感知的时间序列基础模型 VINTAGE-TS，通过区分观测时间与信息可用时间、预测首次发布值并使用联合预测分布，解决了使用修订后数据评估历史预测时可能产生的前视偏差问题。 |
| [^209] | [DEEPO: Dual-Entropy Enhanced Policy Optimization for Hallucination in MLLMs](https://arxiv.org/abs/2609.28570) | 该论文提出DEEPO方法，针对强化学习纠正链中的两个薄弱环节——高语义熵困难查询导致组相对优势归零、以及“自信但错误”的token梯度不可见——通过结合信号方差正则化与梯度预处理的双阶段增强来抑制多模态大语言模型的幻觉问题。 |
| [^210] | [Leakage-Safe Machine Learning for Hydrogen Embrittlement Detection in 316L Stainless Steel: A Region-Held-Out Evaluation of Texture and Deep Features in SEM Micrographs](https://arxiv.org/abs/2609.28567) | 该论文提出了一种基于留一区域法（LORO）交叉验证的区域留出评估协议，有效防止来自同一试样区域的SEM图像在训练集与测试集之间的信息泄漏，从而为316L不锈钢氢脆检测的机器学习模型提供了更可靠、更严谨的评估方式。 |
| [^211] | [When Explanations Cannot Be Read: Measuring and Correcting SHAP and LIME Rendering for Right-to-Left Languages](https://arxiv.org/abs/2609.28565) | 本文提出SHAP-RTL渲染层，修正SHAP和LIME解释可视化在从右到左语言（如阿拉伯语、乌尔都语等）中的阅读方向错乱和字形断裂问题，同时保持原始归因值、特征排序和模型输出不变。 |
| [^212] | [Don't Read the Log: Execution Traces Contaminate Verifiers in Video-Generation Agents](https://arxiv.org/abs/2609.28564) | 该论文揭示了智能体视频生成系统中的一个关键缺陷：向多模态评审器展示智能体的执行轨迹等辅助文本会严重污染其对纯视觉质量的判断——报告成功的轨迹可使评审器接受78–90%的失败视频，即使明确要求“仅使用视频帧”也无法消除这种偏差。 |
| [^213] | [SpaFactor: Lightweight Spatial Context-Aware Gene Program Modeling for Histology-to-Transcriptomics Inference](https://arxiv.org/abs/2609.28563) | 提出了SpaFactor，一个轻量高效的低秩形态学-程序-基因分解框架，通过融合中心点与多尺度邻域上下文的空间感知表示，从HE染色图像实现基因表达预测，避免了对高维噪声敏感和计算开销大的问题。 |
| [^214] | [Matrix Aggregation Operators](https://arxiv.org/abs/2609.28562) | 本文首次形式化了矩阵聚合算子（MAO）的概念，将聚合理论从向量拓展到矩阵结构，并分析了其可分解性与对称性，证明某些算子无法分解为逐行逐列聚合的形式。 |
| [^215] | [CARE: Condition-Aware Representation Regularization for Diffusion Models](https://arxiv.org/abs/2609.28561) | 提出了一种轻量级即插即用的条件感知表示正则化框架CARE，利用内置条件信号动态调节特征分布，无需显式对齐损失或外部监督即可提升扩散模型的视觉保真度和收敛稳定性。 |
| [^216] | [Speculative Evaluation of Stochastic LLMs](https://arxiv.org/abs/2609.28560) | 该论文提出了一种基于分层贝叶斯尼曼策略的推测式评估方法（HBN及异步版本HBN-async），通过分层贝叶斯模型估计各任务方差并自适应地分配推演预算，从而在固定预算下最小化随机大语言模型基准评估的方差。 |
| [^217] | [CFD Correction of Open Tip Clearance Flow in a Compressor Cascade Using VAE Latent Space Adaptation](https://arxiv.org/abs/2609.28558) | 提出了一种基于变分自编码器潜在空间自适应的无侵入式CFD修正方法，仅用12组配对的CFD-实验数据即可有效修正压气机叶栅叶顶间隙流预测与实验之间的偏差。 |
| [^218] | [SMILESGNN: Interpretable Clinical Toxicity Prediction via SMILES-Graph Cross-Attention Fusion](https://arxiv.org/abs/2609.28553) | 该论文提出SMILESGNN多模态架构，通过交叉注意力融合SMILES Transformer与GATv2图编码器，在仅0.4M参数的情况下于ClinTox数据集上取得AUC-ROC 0.987的竞争性性能，同时保留显式图分支以支持基于GNNExplainer的可解释毒性预测。 |
| [^219] | [An Order-Theoretic Characterization of Consistent Inductive Inference](https://arxiv.org/abs/2609.28551) | 该论文在ZFC框架下通过有限可实现轨迹上的一个线性序（要求冲突轨迹选择不同子轨迹、且对每个固定目标良基）完整刻画了一致归纳推理，证明一致性等价于此类序的存在性，并回答了Lu（2024）的问题。 |
| [^220] | [Stochastic Inertial Krasnosel'skii-Mann Iteration Achieves Near-Optimal Sample Complexity](https://arxiv.org/abs/2609.28543) | 本文提出一种随机惯性 Krasnosel'skii-Mann（iKM）迭代方法，仅在随机 KM 算法上添加两个惯性外推、且每次更新只需一次可能有偏随机预言机调用的条件下，实现了 Õ(ε⁻²) 的近乎最优样本复杂度。 |
| [^221] | [An Exposition of GPT Astra's Proof of Lower Bound on DP Continual Counting](https://arxiv.org/abs/2609.28528) | 本文详细阐述了 GPT Astra 关于差分隐私持续计数下界的证明，并希望有助于寻找更自然、更简洁的证明方法。 |
| [^222] | [Sequential Confidence Sets for Coverage-Constrained Conformal Model Selection](https://arxiv.org/abs/2609.28522) | 提出了覆盖率约束序贯模型置信集（CC-SMCS），利用同步鞅置信序列与精确闭式规则，以至少1-δ的概率识别出满足覆盖率硬约束且成本最小的保形预测流水线。 |
| [^223] | [Certified Task-Conditioned Active Observability](https://arxiv.org/abs/2609.28520) | 该论文形式化了“认证的任务条件化主动可观测性复杂度”，即在认证误差与安全弃权保证下识别任务相关状态所需的最小最坏情况期望交互代价，并证明任务预测等价性诱导出唯一的最小充分商空间，使主动可观测性复杂度在其上严格不变。 |
| [^224] | [Stable and Faithful Explanations for Knowledge Tracing](https://arxiv.org/abs/2609.28502) | 本研究提出一套同时评估预测竞争力、解释稳定性与忠实性的知识追踪解释验证协议，并发现重建ASSISTments 2009数据集以修复多技能交互的标签泄漏问题后会改变模型AUC并重新排列解释结果。 |
| [^225] | [Algebraic Expressivity Certificates for Shallow Polynomial Neural Networks](https://arxiv.org/abs/2609.28500) | 本文借助代数几何中的 Veronese 割簇与理想消元方法，构建了一个将网络架构自动转化为不可表示性多项式证书的通用流水线，并据此推导出球面上秩二二次网络的精确总体损失下界，揭示代数障碍会造成不可消除的近似误差。 |
| [^226] | [Hybrid Variational Quantum-Classical Framework with Adaptive Weighting and Efficiency Assessment](https://arxiv.org/abs/2609.28491) | 该论文提出了Sim-HVQC混合深度量子神经网络框架，通过将无参数的SimAM自适应加权模块与经典特征提取相结合来保留类别判别信息，首次将变分量子电路扩展应用于多类别分类任务，并通过多种子评估证明了其可复现性、参数效率和可解释性。 |
| [^227] | [RADAR: Readiness for AI Discovery and Agentic Reach](https://arxiv.org/abs/2609.28480) | 该论文提出RADAR评估框架，对166个国家的测量发现，AI描述公共服务的能力（信息可读性）远超其代为办理服务的能力（智能体可操作性），且这一差距并不随国家财富增加而缩小。 |
| [^228] | [From Prediction to Explainable Provider Behavior Profiles for Fraud, Waste, and Abuse Review](https://arxiv.org/abs/2609.28477) | 该研究提出将欺诈、浪费与滥用（FWA）审查从预测建模转向可解释的提供者行为画像，通过将账单收入分解为提供者规模与诊疗项目构成的乘积，从而解释提供者行为变化的原因。 |
| [^229] | [Learning the Cost of Reliable Inference](https://arxiv.org/abs/2609.28322) | 该论文设计了一个基于反向第二价格拍卖的大模型采购平台，通过提供商竞争驱动token定价，并在学习各提供商质量的同时，将查询路由到满足质量阈值的最具成本竞争力的提供商。 |
| [^230] | [Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching](https://arxiv.org/abs/2609.28107) | 该论文提出通过迁移单任务条件流匹配专家模型学习到的速度场，将其知识蒸馏到一个共享的多任务策略中，并结合原始CFM目标保持对专家演示的保真度，从而实现计算高效的多任务机器人操作策略学习。 |
| [^231] | [LAYERSCOPE: A Layerwise Characterization of Video and Multimodal Learned Representations](https://arxiv.org/abs/2609.28086) | 提出无标签逐层分析框架LAYERSCOPE，通过多种几何度量刻画视频与多模态模型的逐层表征结构，发现中间层表征可优于最终层输出，且单一几何度量无法可靠预测下游性能。 |
| [^232] | [NS-ATTENTION: Newton-Schulz Transformations of Attention Outputs in Vision Transformers](https://arxiv.org/abs/2609.27735) | 提出无参数的Newton-Schulz注意力变换（NS-Attn.），对每个注意力头输出进行谱处理以降低谱集中度并提高有效秩，在ViT和Swin于CIFAR-10/100的全部12组对比实验中均带来平均0.25–0.83个百分点的准确率提升。 |
| [^233] | [ProCredit: From Outcome Rewards to Progress Credit in Agentic Reinforcement Learning](https://arxiv.org/abs/2609.27532) | 提出 ProCredit，利用可在中间状态上运行的验收检查，把与最终结果同样可验证的任务进展转化为逐步的信用信号，从而克服长程智能体强化学习中仅依赖结果奖励导致的训练信号稀疏、失败尝试无法区分、推进任务的步骤得不到应得信用等问题。 |
| [^234] | [Scaling of Capability and Efficiency at Inference Time in Large Reasoning Models](https://arxiv.org/abs/2609.27166) | 本文利用层次贝叶斯模型量化了DeepSeek-R1-Distill系列模型在算术与算法推理任务上的表现，发现正确解题概率随问题难度近似指数衰减，而衰减尺度随模型规模增长，从而揭示了推理时能力与效率随模型规模的扩展规律。 |
| [^235] | [Safety Nudges: User-Facing Interventions for Real-Time AI Risk Awareness](https://arxiv.org/abs/2609.26865) | 该研究提出了Safety Nudges——一款基于浏览器的工具，能在聊天机器人对话中实时检测并提示潜在的AI安全风险，实地研究表明此类面向用户的干预措施能有效提升用户对AI危害的意识，可作为模型层面安全防护的有益补充。 |
| [^236] | [QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs](https://arxiv.org/abs/2609.26855) | 提出QUARTET图Transformer架构，利用基于近期截断个性化PageRank的因果随机游走采样器提取密集连通且无时序泄露的局部子图，并通过四分支交叉注意力丰富全局上下文，从而克服RelGT在关系图建模中局部采样松散与全局记忆单一的局限。 |
| [^237] | [Learning to Fluctuate: Statistical Foundations for Causal Tabular Pretraining](https://arxiv.org/abs/2609.26290) | 提出波动监督预训练（FSP），用平均处理效应加有效影响函数波动来标注合成表格，并证明完全波动可使高斯标签可观测，从而将因果标签预测风险从 $(1-\lambda)^2/n$ 阶降至 $n^{-2}$ 阶。 |
| [^238] | [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses](https://arxiv.org/abs/2609.24972) | 该论文提出RRSI方法，通过将正则化原则（如时间退火的编辑预算限制和鼓励探索未开发轨迹）引入智能体框架的递归自我改进过程，防止递归进化对训练任务过拟合，从而提升分布外基准上的泛化能力。 |
| [^239] | [Learning tactile perception from high-bandwidth single-point sensing](https://arxiv.org/abs/2609.24621) | SpectRobot框架将高带宽单点触觉信号转换为时频频谱图，使其能被标准视觉编码器直接处理，从而无需空间分布式传感器阵列即可实现丰富的触觉感知。 |
| [^240] | [Taking a Second Look: Correcting Sea Ice Forecasts with Sparse Observations](https://arxiv.org/abs/2609.24591) | 提出ECHO框架，通过状态自适应的异构传播策略（ECHO-Scale动态调整传播距离、ECHO-Delta学习有界残差）利用稀疏海冰密集度观测校正海冰预报，在全部96个评估设置中均优于固定传播方法。 |
| [^241] | [Auditing Bayesian Graph Alignment: Diagnostic Comparisons and Reference Failure](https://arxiv.org/abs/2609.23232) | 该论文在多组图对上系统比较了贝叶斯图对齐的多种收敛诊断方法，发现没有任何单一诊断在所有采样器和场景下占优，且在大规模图中诊断只能预测后续边缘变化而非后验误差。 |
| [^242] | [StationPDE: Station-Oriented Surface PDE Learning for Multi-Station Multivariate Weather Forecasting](https://arxiv.org/abs/2609.22123) | StationPDE提出了一种面向站点的地表PDE学习模型，将天气演化分解为地表风输运与可学习的高空推断，从而在缺乏连续场和高空数据的条件下实现具有物理可解释性的多站点多变量天气预报。 |
| [^243] | [Complete Neural Electronic Initialization Accelerates Materials DFT](https://arxiv.org/abs/2609.21759) | 该论文提出了首个完备的机器学习方法来加速PAW形式下的材料平面波DFT计算，通过形式化七项准则并引入AugNet——首个PAW增强占据数通用等变模型和首个材料通用自旋密度模型——填补了现有方法中缺失的关键初始化组件。 |
| [^244] | [How Many Humans Is a Judge Panel Worth?](https://arxiv.org/abs/2609.21277) | 该论文提出两种不同的“等效人类评判者数量”度量——谱残差多样性 ν_H 与分布平方误差 ν_MSE，发现同一组 32 个语言模型评审在三个 ChaosNLI 任务上分别相当于约 4.24–6.50 个和 2.30–3.75 个人类评判者，且更大的谱多样性并不保证更好的分布恢复。 |
| [^245] | [Dynamic Generalized Gromov-Wasserstein Optimal Transport](https://arxiv.org/abs/2609.20008) | 该论文提出TP-DATE框架，首次以无模拟方式将Gromov-Wasserstein最优传输动态化，通过路径作用量证明静态与动态二次型最优传输的等价性，并发展行进对流匹配方法，实现空间转录组学中兼顾组织结构保持的连续轨迹重建。 |
| [^246] | [REARL: A Closed-loop Autonomous Driving Simulation Enhancement Framework with Real Traffic Data and Large Language Models](https://arxiv.org/abs/2609.19903) | REARL提出了一种结合真实交通数据聚类与大语言模型的闭环仿真增强框架，通过实时监测仿真与真实交通的偏差并动态调整车辆决策，使自动驾驶仿真更贴近真实交通模式。 |
| [^247] | [TacSushi: Tactile-Grounded World-Action Modeling for Dexterous Sushi Manipulation](https://arxiv.org/abs/2609.19613) | 提出TacSushi，一种触觉接地的世界-动作建模策略，通过特征级门控融合指尖触觉并利用失败试验的未来后果预测进行监督，实现了形变、遮挡和不确定接触条件下的灵巧寿司操作。 |
| [^248] | [TERN: A Delta-rule Memory with a Seasonal Reference and Online Adaptation for Epidemic Forecasting](https://arxiv.org/abs/2609.18407) | TERN是一种基于delta规则快速权重记忆的流感疫情预测模型，通过由疫情阶段特征驱动的门控擦除机制、显式季节参考和在线适应，在多个流感基准测试上超越了现有疫情图模型和通用预测器。 |
| [^249] | [Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts](https://arxiv.org/abs/2609.18366) | 提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。 |
| [^250] | [Improving the Last-Iterate Guarantees of Anytime Algorithms for Stochastic Monotone Variational Inequalities](https://arxiv.org/abs/2609.15257) | 本文提出了一种具有Halpern锚定的单循环单调用随机算法，将随机单调变分不等式的任意时间最后迭代收敛速率从O(t^{-1/5})提升至O(t^{-1/4})，且无需有界方差或有界可行集假设。 |
| [^251] | [An explicit solution of the five-expert prediction PDE and the exact optimality set of COMB](https://arxiv.org/abs/2609.14892) | 本文首次给出五专家预测偏微分方程的显式解，并证明COMB策略仅在低维子集上最优，从而推翻了Gravin、Peres和Sivan（2016）的COMB最优性猜想。 |
| [^252] | [CyFM: Cylindrical Optimal Transport for Few-Step Complex-Valued Flow Matching](https://arxiv.org/abs/2609.14171) | 该论文提出圆柱流匹配CyFM，通过将复值信号建模在圆柱流形上并采用圆柱最优传输路径替代笛卡尔路径，严格约束了角速度回归目标，避免了笛卡尔路径导致的重尾角速度问题，从而实现高效的少步复值流匹配。 |
| [^253] | [Vibe Patenting: Evaluating LLM Judges for Professional Patent-Drafting Agents](https://arxiv.org/abs/2609.13422) | 该论文提出了端到端专利撰写测试平台"Vibe Patenting"，证明LLM裁判的迭代反馈能持续提升AI生成的专利草稿质量，并使低推理低成本智能体接近昂贵的高推理智能体的表现。 |
| [^254] | [PocketVE: Stable and Property-Guided Structure-Based Drug Design with Variance-Exploding Diffusion](https://arxiv.org/abs/2609.08101) | PocketVE是一个以蛋白质口袋为条件的方差爆炸扩散框架，通过结合EDM式3D去噪、无分类器多性质引导和自适应蛋白质扰动正则化，在CrossDocked2020上显著提升了3D有效性（58.6→80.6）并大幅降低了应变能（457.4→127.9）。 |
| [^255] | [Improving precipitation forecasts in an AI weather model using observational data](https://arxiv.org/abs/2609.03210) | 通过使用观测的IMERG降水数据微调图变换器AI天气预报模型，将中期降水预报的CRPS提升最高19%，并使全球极端降雨预测的Brier技巧评分超越最先进业务化模型57%。 |
| [^256] | [Modern Transformers Are Implicit Hybrids: From Functional Differentiation to Principled Hybrid Architecture Design](https://arxiv.org/abs/2609.02986) | 本文提出RFIS和RPD两个干预指标，发现现代Transformer的注意力头自然分化为由全局位置频带（GPBand）分隔的检索头和位置头两类，为有原则地设计FA-LA混合架构提供了实证基础。 |
| [^257] | [Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement](https://arxiv.org/abs/2608.31046) | 研究发现在策略蒸馏中教师监督充满噪声且学生对其并不敏感，其性能提升主要来自对低概率token的学习，即使使用固定负优势信号也能达到同样效果，因此OPD本质上更接近自我改进而非真正的知识蒸馏。 |
| [^258] | [Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data](https://arxiv.org/abs/2608.29001) | 本文提出RaDE方法，利用基于排名的信息和代表性节点子集选择来实现图嵌入的维度可解释性，在降低计算成本的同时改进文本与多媒体数据的检索任务。 |
| [^259] | [Aero Hand Open: A Simulation-Ready Tendon-Driven Hand for Dexterous Manipulation Learning](https://arxiv.org/abs/2608.28578) | 提出了Aero Hand Open——一款仿真就绪的腱驱动拟人灵巧手，附带可复现缆绳传动的仿真模型和双向辨识执行映射，解决了腱驱动手在灵巧操作学习中的仿真建模难题。 |
| [^260] | [J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data](https://arxiv.org/abs/2608.26582) | J-Zero提出了一种统一的挑战者-求解者-评判者协同进化框架，通过对抗性任务生成和基于生成方式的偏好对，实现了无需人工数据即可在可验证和不可验证领域中的自我进化。 |
| [^261] | [Two Dimensions Govern Agnostic Multiclass Transductive Learning](https://arxiv.org/abs/2608.25326) | 该论文证明多类不可知转导学习的最优误差率由DS维度和Natarajan维度共同决定，公式为 $\widetilde\Theta(d_{DS}/n + \sqrt{d_{\mathrm N}/n})$，适用于任意标签空间。 |
| [^262] | [The Sharp Tail of Uniform Stability](https://arxiv.org/abs/2608.24098) | 本文构造了一个确定性均匀稳定学习问题，证明了其泛化差距尾部下界与理论最优上界匹配，从而解决了均匀稳定性中线性对数依赖是否可实现的关键开放问题。 |
| [^263] | [Learning Generalizable Behaviors for Terminal Agents](https://arxiv.org/abs/2608.22631) | 本文提出“智能体组合泛化”假说，认为强化学习通过塑造高层决策行为来组合和路由预训练获得的低层技能，而非从头学习新技能，从而提升终端代理的泛化能力。 |
| [^264] | [How Weight Encoding Affects Language Model Placement and Performance on the Apple Neural Engine](https://arxiv.org/abs/2608.22110) | 权重编码方式（fp16/int8/三元权重）会同时影响语言模型在苹果神经引擎上的硬件部署位置与推理性能，其中 int8 压缩可将热身前向延迟降低约 1.9 倍，但稠密编码本身并不决定模型是否使用 ANE。 |
| [^265] | [DecoVAE: a Lightweight Interpretable Trend-Seasonal VAE Framework for Efficient Probabilistic Time Series Forecasting](https://arxiv.org/abs/2608.20052) | 本文提出DecoVAE，一个轻量级可解释的VAE框架，通过趋势差分正则化和频域复高斯VAE显式分解时间序列，在七个基准上持续优于现有方法，同时降低内存和计算开销。 |
| [^266] | [CLaST: Context-aware Contrastive VAE for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2608.20025) | CLaST通过引入上下文感知的对比损失函数，增强了VAE对时间序列内部依赖的捕捉能力，从而在概率预测中显著提升了准确率。 |
| [^267] | [Does Mapping Non-Maximal Probabilities to GMM Components Matter for S-JEPA Encoder Representations?](https://arxiv.org/abs/2608.19084) | 本文通过对照实验证明，S-JEPA编码器中非最大概率分配到特定GMM分量对表示质量有显著影响，仅保留概率值而不保留映射信息会降低性能。 |
| [^268] | [Continual Reasoning Gym: Diagnosing and Harnessing Shared Reasoning in Continual RLVR](https://arxiv.org/abs/2608.18574) | 本文提出了持续推理健身房环境，发现持续RLVR的最终性能差距主要由共享推理而非遗忘主导，并探讨了如何利用共享推理来提升持续学习效果。 |
| [^269] | [A Comprehensive Review of Large Language Models for Nanophotonics: From Surrogate Modeling to Autonomous Design](https://arxiv.org/abs/2608.18279) | 本文综述了大型语言模型在纳米光子学中的两种应用模式，即作为代理模型处理结构-光谱映射和实现自主设计，以突破传统深度学习在通用推理能力上的局限。 |
| [^270] | [Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection](https://arxiv.org/abs/2608.17965) | 本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。 |
| [^271] | [Reflex-Guard: A Low-Latency Guardrail for LLM Prompt Safety Using Dense Semantic Embeddings](https://arxiv.org/abs/2608.17556) | Reflex-Guard是一种本地运行的轻量级护栏，通过越狱感知预处理、紧凑嵌入和快速分类器，在低于100毫秒的延迟下实现高精度提示安全过滤，同时避免数据隐私风险。 |
| [^272] | [Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks](https://arxiv.org/abs/2608.07335) | 本文系统评估了八种CNN编码器在并行化Q网络中的性能，并结合Hadamax编码与多种价值函数头，提出了一个在Atari-57上表现优异的复合架构。 |
| [^273] | [Multiscale Reward Hedging from Correct Demonstrations](https://arxiv.org/abs/2608.06825) | 该论文首次为连续奖励类给出了无视野限制的奖励对冲保证，通过在每个精度尺度上共享投票进行对冲，使累积隐藏间隙由度量熵积分界定且与轮数无关，多项式熵下可达到 O(d log A) 的总间隙和 O(d/m) 的快速收敛率。 |
| [^274] | [Hardware Keystores for AI Agent Signing Workflows: A Zero-Trust MCP Enforcement Architecture](https://arxiv.org/abs/2608.06130) | 论文针对AI代理签名场景中的“困惑代理人”问题，提出了一种结合硬件密钥存储的五层零信任强制执行架构，确保只有符合操作者已确认意图的请求才能触发硬件签名，从而防御提示注入攻击导致的私钥滥用。 |
| [^275] | [Trident : How to Break Deep Reinforcement Learning Cyber Defenses (Agentic)](https://arxiv.org/abs/2608.04317) | 该论文提出Trident框架，通过动态沙箱基准环境、超过1.3万条红蓝对抗交互轨迹数据集以及“代码即策略”的RLVR智能体架构，训练LLM红队智能体自适应地发起攻击，从而揭示深度强化学习网络防御系统面对自适应威胁时的脆弱性。 |
| [^276] | [Wiring Beats Blending: What Transfers Between Transformer Sizes -- and What Doesn't](https://arxiv.org/abs/2608.02829) | 本文发现，在不同规模的Transformer模型间转换时，表示对齐强但参数对齐弱，价值在于初始化，并通过最小二乘补偿和方差保持重缩放两个杠杆实现有效转换。 |
| [^277] | [Global Convergence of DGM and PINN Algorithms for Solving Nonlinear PDEs](https://arxiv.org/abs/2607.24726) | 本文证明了DGM和PINN算法在求解一类半线性非线性PDE时，其训练过程能全局收敛到PDE的真实解，解决了非凸优化下算法数学基础的长期问题。 |
| [^278] | [No Free Lunch in Flow Surrogates under Time-Varying Boundary Conditions: A Two-Regime Study](https://arxiv.org/abs/2607.23667) | 在时变边界条件下不存在“万能”的流动代理模型架构：一次性全场模型最适合同步辐射CMP浆料薄膜流动（误差2.7%），而只有潜在自回归DeepONet能保留卡门涡街90%的脱涡功率，表明对时间维度的处理方式是决定模型成败的关键。 |
| [^279] | [A Multi-level Information Integration Framework for Physically Verifiable Fault Diagnosis of Rotating Machinery](https://arxiv.org/abs/2607.22797) | 提出一种与编码器无关的多任务框架——诊断证据网络（DENet），将旋转机械故障诊断输出扩展为包含分类结果、可对照轴承几何与转速理论值验证的预测特征频率以及时间定位信息的结构化证据记录，从而实现物理可验证的故障诊断。 |
| [^280] | [SechKAN: Kolmogorov-Arnold Networks with Hyperbolic Secant Functions](https://arxiv.org/abs/2607.18290) | 本文提出了基于双曲正割函数的新型Kolmogorov-Arnold网络SechKAN，通过一维线性投影控制参数规模，在函数拟合、PDE代理建模和图像分类任务上取得了与MLP及现有KAN变体相当或更优的性能。 |
| [^281] | [A JoLT for the KV cache: Near-lossless KV cache compression via joint Lagrangian allocation of Tucker ranks and a rotated residual for llms](https://arxiv.org/abs/2607.12550) | 本文提出JoLT方法，通过部分Tucker分解和旋转低比特残差，在保持头与层轴完整的同时压缩令牌和特征轴，实现KV缓存的近无损压缩。 |
| [^282] | [A hybrid analytical-PINN model for subsurface simulation of geothermal heat exchangers in heterogeneous underground](https://arxiv.org/abs/2607.12271) | 该研究提出一种解析与物理信息神经网络（PINN）混合的参数化框架，通过解析提取奇异线源响应并利用钻孔中心相对坐标下的叠加原理复用神经修正，实现了非均质地下环境中多钻孔地热换热器长期温度场的高效通用模拟。 |
| [^283] | [Who Owns the AI Recommendation? A Multi-Industry Empirical Map of Brand Category Ownership Across Large Language Models](https://arxiv.org/abs/2606.23057) | 该研究通过对五个行业50个品牌、250个查询在三个大语言模型上跨越两个月的大规模实证测量，首次绘制了AI推荐中品牌类别归属的图谱，发现品牌收录率较为均衡、推荐份额随时间高度稳定，并识别出7.6%的“竞争真空”查询。 |
| [^284] | [CaliPPer: quantifying, predicting and improving AI model performance for binding prediction](https://arxiv.org/abs/2606.07258) | CaliPPer是一个事后校准框架，通过多链样本-域距离与距离感知贝叶斯重校准，在无标签条件下实现免疫受体结合预测AI模型性能的量化、预测与提升。 |
| [^285] | [DiffUNet^2: Bidirectional Conditional Diffusion for Probabilistic Scientific Spatiotemporal Modeling](https://arxiv.org/abs/2606.03926) | DiffUNet² 是一种双向条件扩散模型，能在单一共享模型中同时完成科学时空数据的正向预测和反向概率推断，在流体力学、化学反应动力学和材料变形等多个科学数据集上均展现出强大的双向预测能力和高质量的概率集成效果。 |
| [^286] | [A Unified Benchmark for Dynamic Medical Treatment Reinforcement Learning](https://arxiv.org/abs/2606.01028) | 该论文提出了MedGym——一个基于物理信息神经网络、在连续时间框架下由临床数据构建的可配置动态治疗推荐强化学习基准，用于评估强化学习方法处理不规则测量间隔、个体化治疗反应和测量点间安全性的能力。 |
| [^287] | [A Multimodal 3D Foundation Model for Light Sheet Fluorescence Microscopy Enables Few-Shot Segmentation, Classification, and Deblurring](https://arxiv.org/abs/2605.26026) | 该论文提出了一个针对光片荧光显微镜数据的3D基础模型，通过在大规模多样化3D图像上联合优化掩码重建与图像-文本对齐进行预训练，学习可迁移的体积表征，大幅降低标注负担，并实现少样本分割、分类与去模糊。 |
| [^288] | [DeGRe: Dense-supervised Generative Reranking for Recommendation](https://arxiv.org/abs/2605.25749) | 该论文提出DeGRe框架，通过引入密集监督信号来解决生成式重排序中的启发式标签偏差和稀疏奖励导致的信用分配难题。 |
| [^289] | [Every Component Is a Lookup: One Linear Graph for Interaction, Composition and Attribution](https://arxiv.org/abs/2605.23393) | 本文提出基于“键值形式”与“加性残差流”两个架构假设，将 Transformer 统一表示为一张线性计算图，并通过 Unpack 反向归因方法，使组件交互、组合路径与词元归因成为同一张图的不同读出结果。 |
| [^290] | [Lossless Anti-Distillation Sampling](https://arxiv.org/abs/2605.18829) | 提出无损抗蒸馏采样（LADS），在不改变生成内容与质量的前提下，通过在账号间耦合潜在随机性、同时保持账号内生成独立性的机制，大幅削弱蒸馏攻击的效力。 |
| [^291] | [Pointwise Generalization in Deep Neural Networks](https://arxiv.org/abs/2605.18598) | 该论文为全连接深度神经网络建立了逐点泛化理论，通过基于各层学习特征表示特征值的“逐点黎曼维度”来刻画假设，提出了依赖假设、感知表示的泛化界，在理论和实验上均比基于模型规模、范数乘积和无限宽度线性化的传统方法紧致多个数量级，为表示学习奠定了新的统计基础。 |
| [^292] | [GOMA: Toward Structure-Driven Multimodal Alignment from a Graph Signal Smoothing Perspective](https://arxiv.org/abs/2605.15723) | 提出GOMA方法，从图信号平滑视角出发，通过内容嵌入（负责配对身份区分）和语义嵌入（负责关系一致性）两个相连嵌入的分工设计，实现结构驱动的多模态对齐。 |
| [^293] | [A Neural Hierarchical-Matrix Preconditioner for Real-Time GPU Solves](https://arxiv.org/abs/2605.13343) | 该论文提出一种由图-注意力网络学习的H²矩阵格式SPD近似逆预条件子，并采用截断Kaporin条件数作为训练目标以克服近零模态上的梯度消失问题，从而在8-16毫秒预算内实现对每帧变化的稀疏对称正定系统的实时GPU求解。 |
| [^294] | [Open-access model for detecting openly dumped dispersed municipal solid waste from crowdsourced UAV imagery in Sub-Saharan Africa](https://arxiv.org/abs/2605.02316) | 该研究提出了一个开放获取的深度学习模型，利用众包无人机影像在撒哈拉以南非洲10个国家的29个区域自动检测露天倾倒的分散固体垃圾，揭示了垃圾堆积与人口密度及当地基础设施缺乏密切相关。 |
| [^295] | [IatroBench: A Pre-Registered Benchmark of Clinical Omission in Language Models](https://arxiv.org/abs/2604.07709) | 论文提出预注册基准IatroBench，从“作为”与“遗漏”两个伤害维度评估语言模型在临床场景中的安全性，并首次揭示了模型对同一病例会向医生提供比患者更多临床信息的“框架依赖性信息保留”现象。 |
| [^296] | [LiveMathematicianBench: A Live Benchmark for Research-Level Mathematical Reasoning with Proof Sketches](https://arxiv.org/abs/2604.01754) | 提出了LiveMathematicianBench，一个基于训练截止日期后新发表arXiv论文构建的动态研究级数学推理基准测试，通过引入十三类定理逻辑分类体系和证明概要实现细粒度评估，有效避免了数据污染问题。 |
| [^297] | [Invertible Query-Key Coupling Composes with Attention Mechanisms](https://arxiv.org/abs/2604.01683) | 提出一种可逆的查询-键耦合变换（RealNVP风格的交替仿射映射），可无损地叠加在现有注意力机制之上，以极少的额外参数和不变的整体架构显著提升差分注意力等方法的语言建模性能。 |
| [^298] | [MemGuard-Alpha: Limits of Membership Inference for Detecting and Filtering Memorization-Contaminated Signals in LLM-Based Financial Forecasting](https://arxiv.org/abs/2603.26797) | 该论文提出MemGuard-Alpha框架，通过融合五种MIA方法与时间邻近特征的复合污染分数以及跨模型记忆化分歧，系统审计了成员推断攻击在检测并过滤大语言模型金融预测中记忆化污染信号方面的能力与局限。 |
| [^299] | [Beyond Pairwise Attention: Higher-Order Modular Attention for Efficient Sequence Learning](https://arxiv.org/abs/2603.11133) | 该论文提出高阶模块化注意力HOMA，通过重叠块、局部窗口和低秩投影将成对注意力与可计算的三元注意力显式融合，在捕捉超出显式建模阶数的高阶依赖的同时实现更快收敛和更高的参数效率。 |
| [^300] | [MultiwayPAM: Multiway Partitioning Around Medoids for LLM-as-a-Judge Score Analysis](https://arxiv.org/abs/2603.10287) | 该论文提出了MultiwayPAM，一种新的张量聚类方法，能够同时估计LLM-as-a-Judge评分张量各模式的聚类成员和中心点，从而揭示LLM评估器评分偏差的结构。 |
| [^301] | [Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety](https://arxiv.org/abs/2603.10044) | 评测条件对测得的模型安全性影响超过脚手架本身——在相同的基准题目上，选择题与开放式格式会使测得的安全性相差5-20个百分点，说明评测结果更多取决于测量方法而非模型潜在的 safety 能力。 |
| [^302] | [Learning Causal Structure of Time Series using Best Order Score Search](https://arxiv.org/abs/2603.05370) | 本文提出TS-BOSS算法，将最优序分数搜索（BOSS）扩展到多变量时间序列的动态贝叶斯网络因果结构学习中，兼具可扩展性与理论保证。 |
| [^303] | [RQ-Reg: A Residual-Quantization-Based Framework for Continuous Value Prediction in Recommender Systems](https://arxiv.org/abs/2602.23012) | 该论文提出RQ-Reg框架，通过残差量化将观看时长、GMV等连续目标值分解为从粗到细的量化编码序列并进行自回归预测，以克服传统回归方法在建模复杂长尾分布时欠拟合或牺牲泛化性的缺陷。 |
| [^304] | [Gradient Networks for Universal Magnetic Modeling of Synchronous Machines](https://arxiv.org/abs/2602.14947) | 该论文提出将梯度网络嵌入电机方程的物理约束框架，用于含空间谐波的饱和同步电机通用磁性建模，从构造上保证互易性与能量守恒，同时兼顾高精度、高数据效率与实时控制的可行性。 |
| [^305] | [TabSieve: Explicit In-Table Evidence Selection for Tabular Prediction](https://arxiv.org/abs/2602.11700) | TabSieve提出了一种先选择后预测的表格预测框架，通过显式选择表内证据行、构建40K规模的合成微调数据集TabSieve-SFT-40K，以及结合分离奖励的强化学习方法TAB-GRPO，实现了可审计且稳健的表格预测。 |
| [^306] | [Near-Oracle KV Selection via Pre-hoc Sparsity for Long-Context Inference](https://arxiv.org/abs/2602.08329) | 该论文提出事前稀疏化方法PrHS，在注意力打分之前进行KV选择以避免事后启发式方法的后验偏差，并推导出仅依赖丢弃质量的互信息损失上界，从而为长上下文LLM推理提供具有显式精度控制的近预言机KV选择。 |
| [^307] | [Quantum Attention by Overlap Interference: Predicting Classical and Many-Body Quantum Sequences](https://arxiv.org/abs/2602.06699) | 提出了一种通过状态重叠干涉与多项式核实现非线性、并利用Rényi-1/2熵泛函估计损失的变分量子自注意力机制（QSA），相比最优经典方法在训练复杂度上具有潜在优势，可用于预测经典和多体量子序列。 |
| [^308] | [TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference](https://arxiv.org/abs/2602.05145) | TIDE 通过复用推理过程中的中间隐藏状态在线增量训练草稿模型，并结合自适应运行时控制与异构 GPU 集群调度，在不增加额外目标模型开销的情况下实现了高达 1.66 倍的 LLM 推理吞吐量提升。 |
| [^309] | [On Cost-Aware Designs for Sequential Hypothesis Testing](https://arxiv.org/abs/2512.19067) | 本文提出了成本感知序贯假设检验框架，证明了最优期望总成本按 $\Theta(\log(1/\delta))$ 缩放，并揭示了“最大化期望信息增益与期望成本之比”这一设计原则，据此改编的经典策略具有渐近最优性。 |
| [^310] | [Few-Shot Specific Emitter Identification via Integrated Complex Variational Mode Decomposition and Spatial Attention Transfer](https://arxiv.org/abs/2512.16786) | 该论文提出集成复数变分模态分解、时间卷积网络与空间注意力机制并借助预训练权重迁移的方法，在标注数据有限的小样本条件下实现了高精度的辐射源个体识别。 |
| [^311] | [A Fast and Effective Solution to the Problem of Look-ahead Bias in LLMs](https://arxiv.org/abs/2512.06607) | 本文提出一种推理时干预方法，利用两个小型专用模型调整大模型logits，从而快速、低成本地消除大语言模型在金融预测中的前瞻偏差。 |
| [^312] | [Self-Localizing MIMO Beam Mapping for Intelligent Open RAN with Continuously Evolving Channel Memory](https://arxiv.org/abs/2511.17007) | 本文提出了一种无需位置标签的自定位MIMO波束映射框架，利用稀疏CSI测量和波束域RSS构建分层无线记忆，并通过双尺度提取器和混合时间编码器实现高效的信道知识推理。 |
| [^313] | [MLPerf Automotive](https://arxiv.org/abs/2510.27065) | MLPerf Automotive 是首个针对汽车 AI 加速机器学习系统的标准化公开性能基准测试，提供可复现的评估框架，涵盖 2D/3D 目标检测、语义分割和端到端驾驶等汽车感知任务。 |
| [^314] | [Enabling Approximate Joint Sampling in Diffusion LMs](https://arxiv.org/abs/2509.22738) | 本文提出在现有大型扩散语言模型之上附加一个轻量级单层“采样器”，使模型能够在一次前向传播中近似地从真实联合分布并行采样多个 token，从而在保持准确率的同时大幅提升生成速度。 |
| [^315] | [Unraveling the cognitive patterns of Large Language Models through module communities](https://arxiv.org/abs/2508.18192) | 该研究借鉴生物认知系统的分析方法，开发了一个连接认知技能、LLM架构和数据集的基于网络的框架，通过模块社区分析揭示了大语言模型展现出独特的模块组织结构，其涌现的技能模式部分类似于生物系统的认知特化机制。 |
| [^316] | [Stacked SVD or SVD stacked? A Random Matrix Theory perspective on data integration](https://arxiv.org/abs/2507.22170) | 本文借助随机矩阵理论，首次在比例渐近区间下严格比较了Stack-SVD与SVD-Stack这两种估计多数据集共享奇异子空间的主流数据整合方法的理论性能。 |
| [^317] | [Capturing Unseen Spatial Heat Extremes Through Dependence-Aware Generative Modeling](https://arxiv.org/abs/2507.09211) | DeepX-GAN是一种显式捕捉空间依赖性的深度生成模型，能够零样本模拟超出历史记录的统计上合理的“未见”热极端事件，揭示多个地点同时遭受极端高温的隐藏风险。 |
| [^318] | [The kernel of graph indices for vector search](https://arxiv.org/abs/2506.20584) | 提出了基于核方法的支持向量图（SVG）这一新型图索引，首次为度量和非度量向量空间中的向量搜索提供形式化可导航性保证，并将 HNSW、DiskANN 等流行图索引统一解释为它的特例。 |
| [^319] | [Diffusion-aided Task-oriented Semantic Communications with Model Inversion Attack](https://arxiv.org/abs/2506.19886) | 该论文针对任务保密场景下的模型反演攻击问题，提出一种扩散模型辅助的面向任务语义通信方法，防止攻击者在不知晓接收方任务与模型的情况下从语义特征中重建原始输入，并指出仅用PSNR/SSIM评估隐私泄露并不充分，需在保证任务准确性的同时兼顾隐私保护。 |
| [^320] | [Multimodal AI predicts clinical outcomes of drug combinations from preclinical data](https://arxiv.org/abs/2503.02781) | 本文提出多模态AI模型Madrigal，通过将分子结构、通路、细胞活力和转录组学数据对齐到共享潜在空间，实现了从临床前数据预测药物组合临床结果，性能优于现有方法。 |
| [^321] | [Time-Varying Bayesian Optimization Without a Metronome](https://arxiv.org/abs/2501.18963) | 该论文首次推导出显式考虑观测采样频率变化的时变贝叶斯优化遗憾上界，并据此提出了关于数据集规模和过期数据策略的实用建议，其 BOLT 算法在实验中优于现有最先进的 TVBO 方法。 |
| [^322] | [Foundations of Large Language Models](https://arxiv.org/abs/2501.09223) | 本书系统阐述了大语言模型的六大核心基础领域——预训练、生成模型、提示、对齐、推断与推理，为学习者提供了一部权威的基础性参考书。 |
| [^323] | [AIR: Analytic Imbalance Rectifier for Continual Learning](https://arxiv.org/abs/2408.10349) | 该论文提出了面向真实世界持续学习的解析式不平衡矫正器（AIR），通过冻结主干网络、闭式解增量分类器以及解析式重加权模块（ARM），有效解决了不平衡数据流中的类别不平衡问题。 |
| [^324] | [Transductive Off-policy Proximal Policy Optimization](https://arxiv.org/abs/2406.03894) | 本文提出ToPPO，通过新颖的策略改进下界表述和计算高效的优化机制，将离策略数据安全地引入PPO训练，在保证单调改进的同时显著提升了PPO利用异策略数据的能力。 |
| [^325] | [Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data](https://arxiv.org/abs/2405.18929) | 提出了一种将正-无标签学习与自编码器、深度支持向量数据描述等深度异常检测模型相结合的深度正-无标签异常检测框架，以应对无标签数据被异常污染的现实情况，从而提升半监督异常检测的性能。 |
| [^326] | [Generating Interesting Scientific Ideas using Knowledge Graphs and LLMs: Evaluations with 100 Research Group Leaders](https://arxiv.org/abs/2405.17044) | 该研究提出SciMuse系统，利用包含5800万篇论文的知识图谱结合大语言模型生成个性化研究想法，并通过100多位研究团队负责人对4400多个想法的大规模评估发现，专家整体兴趣评分虽保守（均值2.40/5），但近四分之一的想法获得了高分认可。 |
| [^327] | [A Probabilistic Approach for Alignment with Human Comparisons](https://arxiv.org/abs/2403.10771) | 通过提出的两阶段“监督微调+人类比较”框架，本文研究了如何有效利用人类比较来改善AI模型的对齐，特别是在面对嘈杂数据和高维模型时。 |
| [^328] | [Reconstructing short-lived particles using hypergraph representation learning](https://arxiv.org/abs/2402.10149) | 提出了一种基于超图表示学习的图神经网络架构HyPER，能够高效重建对撞机事例中的短寿命母粒子，其性能优于现有最先进技术且参数效率更高。 |
| [^329] | [Calpric: Inclusive and Fine-grain Labeling of Privacy Policies with Crowdsourcing and Active Learning](https://arxiv.org/abs/2008.02954) | 该论文提出Calpric框架，通过结合自动文本分割、众包标注和主动学习，以低成本高效生成大规模、高质量的隐私政策训练数据集，使未经训练的众包标注者能达到与专业标注者相当的标注质量。 |
| [^330] | [CatSIM: A Categorical Image Similarity Metric](https://arxiv.org/abs/2004.09073) | CatSIM是一种基于结构相似性范式的新图像相似度度量方法，适用于二值和多值的二维及三维图像与体积，对位置的微小扰动具有鲁棒性，并能比较图像内部的任意区域。 |
| [^331] | [DCRMTA: Unbiased Causal Representation for Multi-touch Attribution.](http://arxiv.org/abs/2401.08875) | DCRMTA提出了一种无偏的多触点归因方法，通过建立转化预测模型和构建对照触点序列来减轻偏差的影响。 |

# 详细

[^1]: 具身强化学习中用于私有轨迹重构的时间梯度反演

    Temporal Gradient Inversion for Private Trajectory Reconstruction in Embodied Reinforcement Learning

    [https://arxiv.org/abs/2609.30258](https://arxiv.org/abs/2609.30258)

    该论文提出TRACE时间梯度反演攻击，通过利用连续具身梯度间的跨时间相关性和策略头梯度结构中的闭式动作恢复，从每步策略梯度中自回归地重构出私有的观测-动作轨迹序列，突破了传统单帧梯度反演攻击的局限。

    

    具身强化学习智能体中的分布式学习通过在设备端保留原始传感器数据、仅向服务器传输策略梯度，提供了一定程度的隐私保护。然而，时间结构可以将这种泄露放大到超越单帧攻击的程度。我们提出了针对连续编码的时间重构攻击（TRACE），这是一种摊销式的时间梯度反演攻击，能够从每步策略学习梯度中自回归地重构出私有的观测-动作轨迹序列。该攻击利用了以往单帧方法所忽略的两种结构性信号：（i）连续具身梯度之间的跨时间相关性，我们通过条件互信息界对其进行了形式化刻画；（ii）从策略头梯度结构中闭式恢复动作，我们证明了当标准熵正则化足够小时，这种恢复是精确的。在保留测试的具身场景中，TRACE达到了18.8 dB PSNR，接近……

    arXiv:2609.30258v1 Announce Type: new  Abstract: Distributed learning in embodied reinforcement-learning agents offers a degree of privacy by retaining raw sensor data on-device and transmitting only policy gradients to the server. Yet temporal structure can amplify this leakage beyond single-frame attacks. We introduce Temporal Reconstruction Attack on Consecutive Encodings (TRACE), an amortized temporal gradient-inversion attack that autoregressively reconstructs the sequence of private observation-action trajectories from per-step policy-learning gradients. The attack exploits two structural signals ignored by prior single-frame methods: (i) cross-time correlation between successive embodied gradients, which we formalize via a conditional mutual-information bound, and (ii) closed-form action recovery from policy-head gradient structure, which we prove exact when standard entropy regularization is sufficiently small. On held-out embodied scenes, TRACE reaches $18.8$ dB PSNR with near
    
[^2]: 在线阴谋论话语的智能体检测

    Agentic Detection of Online Conspiracies

    [https://arxiv.org/abs/2609.30250](https://arxiv.org/abs/2609.30250)

    提出了一个配备社会查询工具的智能体框架，通过结合社会语境推断说话者意图（言外之力），而非仅依赖词汇标记，来检测社交媒体上表达方式隐晦的阴谋论话语。

    

    社交媒体上的阴谋论话语并不总是通过明确的声明或稳定的词汇标记来表达。相同的表面内容可能表达的是认同、合理的担忧、批评、讽刺或嘲弄。因此，主要的挑战不仅是识别与阴谋论相关的声明，而是推断说话者的意图——即话语的言外之力。我们认为这可以通过利用相关的社会语境来实现，为此我们提出了一个配备了支持社会查询工具集的智能体框架。我们在一个独特的希伯来语推文数据集上展示了该方法的优势，该数据集涵盖了四年期间（2018年末至2023年初）公开发布的80%–90%的希伯来语推文，横跨多个选举周期以及新冠疫情和相关疫苗接种运动时期。这种广泛的覆盖范围可用于恢复不同的社会语境。我们在人工标注的数据上对该框架进行了评估。

    arXiv:2609.30250v1 Announce Type: new  Abstract: Conspiratorial discourse on social media is not always expressed through explicit claims or stable lexical markers. The same surface content may express endorsement, legitimate concerns, criticism, satire, or mockery. The main challenge is therefore not only recognizing conspiracy-related claims, but inferring the speaker's intent -- the utterance's illocutionary force. We argue that this can be achieved through the use of relevant social contexts and propose an agentic framework, equipped with a set of tools supporting social queries.   We demonstrate the benefits of our approach on a unique dataset of Hebrew tweets, covering 80\%--90\% of the public Hebrew tweets published over a four-year span (late 2018-- early 2023), encompassing several election cycles as well as the COVID pandemic years and related vaccination campaigns. This extensive coverage can be used in recovering different social contexts. Evaluating our framework on a manu
    
[^3]: 信任与否：语音中基于检索增强的事实核查

    To Trust or Not to Trust: Retrieval-Augmented Fact Checking in Speech

    [https://arxiv.org/abs/2609.30227](https://arxiv.org/abs/2609.30227)

    论文提出VeriSpeak语音事实核查基准，揭示了大型音频语言模型存在显著的文本-语音模态差距（书面声明可验证但语音版本常失败），且仅靠检索增强带来的提升有限。

    

    在线虚假信息越来越多地以语音形式出现，例如新闻片段、播客、访谈、政治演讲和社交媒体视频，这催生了对能够直接从语音中核查声明的系统的需求。我们提出了VeriSpeak，一个用于研究大型音频语言模型（LALMs）中基于语音的事实验证能力的探测基准。VeriSpeak包含3,879条语音声明，涵盖时间性、地理性和关系性事实，且真伪标签均衡。该基准旨在检验事实验证能力能否从文本迁移到语音，以及检索增强的LALMs能否利用文本证据来正确支持或反驳语音声明。我们的实验揭示了一个持续存在的文本-语音模态差距：那些能够可靠验证书面声明的LALMs，在面对相同声明以语音形式呈现时往往会失败。此外，仅靠检索带来的增益有限，因为模型经常混淆检索到的证据……

    arXiv:2609.30227v1 Announce Type: cross  Abstract: Online misinformation increasingly appears in spoken formats such as news clips, podcasts, interviews, political speeches, and social media videos, creating a need for fact-checking systems that can verify claims directly from speech. We introduce VeriSpeak, a probe benchmark for studying speech-based fact verification in Large Audio Language Models (LALMs). VeriSpeak contains 3,879 spoken claims spanning temporal, geographical, and relational facts, with balanced true and false labels. The benchmark is designed to examine whether factual verification ability transfers from text to speech, and whether retrieval-augmented LALMs can use textual evidence to correctly support or refute spoken claims. Our experiments reveal a consistent text-speech modality gap: LALMs that verify written claims reliably often fail on the same claims when spoken. Moreover, retrieval alone provides limited gains because models frequently conflate retrieved ev
    
[^4]: PoEM：从现有策略预测强化学习结果

    PoEM: Predicting RL Outcomes from Existing Policies

    [https://arxiv.org/abs/2609.30226](https://arxiv.org/abs/2609.30226)

    提出PoEM框架，利用一组已在其他奖励上完成强化学习后训练的现有模型来预测新奖励函数下的强化学习结果，从而避免每次奖励变化时都从头运行昂贵且不稳定的强化学习过程。

    

    基础模型通过强化学习（RL）进行后训练，以最大化特定的奖励，例如人类对齐、正确性或指令遵循。这一后训练过程计算量巨大，有时不稳定，并且每次当奖励模型发生变化或我们想要组合多个奖励时，都必须从头开始运行。因此我们提出这样一个问题：给定一个新的奖励函数，是否可以在不实际运行强化学习的情况下预测其强化学习结果？我们通过引入PoEM对这个问题给出了肯定的回答。PoEM是一个框架，它利用一组已经在其他奖励上完成过后训练的模型，来预测在新奖励函数上运行强化学习的输出结果。首先，我们证明如果新的奖励函数可以表示为现有奖励函数的线性组合，那么新的策略在对数空间中也可以表示为现有对数策略的线性组合。令人惊讶的是，即使在奖励之间不存在线性关系的情况下，我们观察到……（摘要在此处截断）

    arXiv:2609.30226v1 Announce Type: cross  Abstract: Foundation models are post-trained with reinforcement learning (RL) to maximize specific rewards, such as human alignment, correctness, or instruction following. This post-training process is computationally intensive, sometimes unstable, and has to be run from scratch every time the reward model changes or when we want to combine multiple rewards. We hence ask: given a new reward function, is it possible to predict the RL outcomes without actually running RL on it? We answer this in the affirmative by introducing PoEM, a framework to predict the outputs of RL on a new reward function using a set of models already post-trained on other rewards. First, we show that if the new reward function can be written as a linear combination of existing ones, then the new policy in log-space can be written as a linear combination of the existing log-policies. Surprisingly, even in cases where the rewards are not linearly connected, we observe that 
    
[^5]: 语言模型的最小侵入式导向方法

    Minimally Invasive Steering of Language Models

    [https://arxiv.org/abs/2609.30218](https://arxiv.org/abs/2609.30218)

    提出MISVO方法，利用基于Fisher信息几何的局部KL散度正则化，在冻结语言模型上实现最小侵入式的测试时导向，避免奖励优化导致输出分布大幅改变和生成质量下降。

    

    前逻辑值导向通过在冻结语言模型的最终隐藏状态上添加向量，使模型适应测试时的奖励。然而，无正则化的奖励优化可能会显著改变输出分布并降低生成质量。我们提出了最小侵入式导向向量优化方法（MISVO），该方法利用诱导的token分布的局部KL散度几何结构来惩罚干预。由此得到的Fisher二次型能够度量分布敏感性，并可通过与冻结语言模型头的矩阵-向量乘积计算出解析梯度。我们推导出了序列级KL散度梯度的精确分解形式，将其分解为解析Fisher项和后缀得分函数项。对于固定的生成视野，我们证明了后缀项在导向幅度上是二阶小量，且三种Fisher替代形式与完整的KL梯度在一阶上保持一致。MISVO使用冻结参考替代形式来优化位置...

    arXiv:2609.30218v1 Announce Type: cross  Abstract: Pre-logit steering adapts a frozen language model to a test-time reward by adding vectors to its final hidden states. Unregularized reward optimization can substantially alter the output distribution and degrade generation quality. We propose Minimally Invasive Steering Vector Optimization (MISVO), which penalizes interventions using the local KL geometry of the induced token distribution. The resulting Fisher quadratic measures distributional sensitivity and admits an analytic gradient computed through matrix--vector products with the frozen language-model head. We derive an exact decomposition of the sequence-level KL gradient into an analytic Fisher term and a suffix score-function term. For a fixed generation horizon, we show that the suffix term is second order in the steering magnitude and that three Fisher surrogates agree with the full KL gradient to first order. MISVO uses the frozen-reference surrogate to optimize position-sp
    
[^6]: 成员资格预言机模型下凸体上线性优化的近似二次下界

    A Nearly Quadratic Lower Bound for Linear Optimization over Convex Bodies in the Membership Oracle Model

    [https://arxiv.org/abs/2609.30215](https://arxiv.org/abs/2609.30215)

    本文在成员资格预言机模型下证明了凸体上线性优化和均匀采样的随机算法具有近似二次下界，其中线性优化的下界匹配已知上界，均匀采样的下界改进了之前的线性结果，且该构造同样给出了体积估计的下界。

    

    我们证明了在成员资格预言机模型中，针对凸体上线性优化和均匀采样的随机算法的近似二次下界。对于线性优化，该下界与已知的近似二次上界在维度上至多相差一个多对数因子。对于均匀采样，该结果改进了先前的线性下界。我们的构造还隐含了体积估计的相同下界。

    arXiv:2609.30215v1 Announce Type: cross  Abstract: We prove nearly quadratic lower bounds for randomized algorithms for linear optimization and uniform sampling over convex bodies in the membership oracle model. For linear optimization, this matches the known nearly quadratic upper bound up to a polylog factor in the dimension. For uniform sampling, this improves on the previous linear lower bound. Our construction also implies the same lower bound for volume estimation.
    
[^7]: 锚定额外邻近方法：单调包含问题的最优高阶方法

    Anchored Extra-Proximal Methods: Optimal Higher-Order Methods for Monotone Inclusion Problems

    [https://arxiv.org/abs/2609.30212](https://arxiv.org/abs/2609.30212)

    提出了锚定额外邻近（AEP）框架，通过结合锚定外推与满足相对误差条件的非精确锚定邻近更新，为复合单调包含问题构造出复杂度最优的任意阶（p≥2）高阶求解方法。

    

    我们研究了在切残差准则下，求复合单调包含问题近似解所需的确定性预言机复杂度，该问题由一个光滑的单值单调算子与一个极大单调的集值算子之和构成。我们提出了锚定额外邻近（AEP）框架，该框架将锚定外推步骤与满足相对误差条件的非精确锚定邻近更新相结合。在一阶情形下，该框架能够恢复复合快速外梯度方法；通过将隐式更新中的算子替换为其在外推点处的泰勒近似，该框架自然地产生了二阶及更高阶的扩展。对于每个 p≥2，在假设单值算子的 (p-1) 阶导数满足 Lipschitz 连续的条件下，我们将这一构造与二分线搜索相结合，得到一个 p 阶方法，用以找到切残差满足给定精度的点。

    arXiv:2609.30212v1 Announce Type: cross  Abstract: We study the deterministic oracle complexity of finding approximate solutions to composite monotone inclusion problems, formed by the sum of a smooth single-valued monotone operator and a maximally monotone set-valued operator, under the tangent-residual criterion. We introduce the Anchored Extra-Proximal (AEP) framework, which combines an anchored extrapolation step with an inexact anchored proximal update satisfying a relative-error condition. The framework recovers the composite Fast Extragradient method in the first-order setting and yields natural second- and higher-order extensions by replacing the operator in the implicit update with its Taylor approximation at the extrapolated point. For every $p\geq 2$, assuming that the $(p-1)$th derivative of the single-valued operator is Lipschitz continuous, we combine this construction with a bisection line search to obtain a $p$th-order method that finds a point with tangent residual at 
    
[^8]: 多模态大语言模型中的对齐幻觉

    The Alignment Illusion in Multimodal Large Language Models

    [https://arxiv.org/abs/2609.30210](https://arxiv.org/abs/2609.30210)

    该研究通过对13个多模态大语言模型施加受控干预，揭示了标准标量相似度度量无法真实反映跨模态内容交互的“对齐幻觉”现象，其根源在于MLP下投影所导致的权重诱导对齐。

    

    多模态大语言模型（MLLM）中的层级视觉-文本相似度被广泛解读为语言模型逐步将视觉内容整合进共享表示空间的证据。这种解读建立在这样一个假设之上：标量对齐分数反映的是内容层面的跨模态交互。为了检验这一假设，我们对视觉流施加了受控干预。在来自五个模型家族、参数量从0.5B到72B的13个MLLM上，用高斯噪声替换投影器输出的视觉token会急剧降低任务准确率，然而四种标准标量度量（CKA、SVCCA、MIR以及主角度余弦）却无法一致地区分被破坏的视觉流与原始视觉流。我们将这种失败称为“对齐幻觉”，并将其根源追溯至共享的语言模型路径：各向异性的MLP下投影将视觉token和文本token拉向共同的输出方向，从而产生了由权重引起的对齐……

    arXiv:2609.30210v1 Announce Type: cross  Abstract: Layer-wise visual-text similarity in Multimodal Large Language Models (MLLMs) is widely interpreted as evidence that the language model progressively integrates visual content into a shared representation space. This reading rests on the assumption that scalar alignment scores reflect content-level cross-modal interaction. To test this assumption, we apply controlled interventions to the visual stream. Across 13 MLLMs from five families spanning 0.5B to 72B parameters, replacing projector-output visual tokens with Gaussian noise sharply reduces task accuracy, yet four standard scalar measures (CKA, SVCCA, MIR, and the leading principal-angle cosine) fail to consistently separate the corrupted stream from the original. We call this failure the alignment illusion and trace it to the shared language-model pathway: anisotropic MLP down-projections pull visual and text tokens toward common output directions, producing weight-induced alignme
    
[^9]: 超越压缩：为神经代理求解器中稳定的长时程滚动预测训练潜在表示

    Beyond Compression: Training Latent Representations for Stable Long-Horizon Rollout in Neural Surrogate Solvers

    [https://arxiv.org/abs/2609.30198](https://arxiv.org/abs/2609.30198)

    该论文揭示潜在神经代理求解器长时程预测不稳定的根源在于潜在表示仅按重建目标训练，并通过Koopman算子学习、Hamming噪声注入、噪声注入和多步滚动微调等训练层面干预，使潜在表示适配稳定的长时程滚动预测。

    

    潜在神经代理求解器，或称潜在动力学模型，通过演化压缩的潜在空间而非直接求解全分辨率物理场来加速时变物理系统的仿真。原则上，这可以降低计算成本并简化学习过程，但在实际应用中，误差往往会在长自回归滚动预测过程中快速累积，限制了其预测效用。我们表明，这种不稳定性并非源于潜在表示本身，而是产生于当它仅针对重建目标进行训练时——这样的表示并不适合长时程预测任务。我们系统地评估了一系列使潜在表示与长时程滚动预测对齐的训练层面干预措施：在自编码器训练阶段引入Koopman算子学习和Hamming噪声注入以改进压缩质量，同时采用噪声注入和多步滚动微调以改进动力学学习。那些能够改善长时程……

    arXiv:2609.30198v1 Announce Type: new  Abstract: Latent neural surrogate solvers, or latent dynamics models, accelerate simulations of time-dependent physical systems by evolving a compressed latent space rather than resolving full-resolution fields directly. In principle this reduces computational cost and simplifies learning, but in practice errors often accumulate rapidly during long autoregressive rollouts, limiting predictive utility. We show that this instability does not stem from the latent representation itself, but arises when it is trained solely for reconstruction, producing representations poorly suited to long-horizon forecasting. We systematically evaluate training-level interventions that align latent representations with long-horizon rollout: Koopman operator learning and Hamming noise injection during autoencoder training to improve compression, together with noise injection and multi-step rollout fine-tuning to improve dynamics. Interventions that improve long-horizo
    
[^10]: 学习动力学中的内在-外在耦合

    Intrinsic-Extrinsic Coupling in Learning Dynamics

    [https://arxiv.org/abs/2609.30185](https://arxiv.org/abs/2609.30185)

    该论文形式化了学习动力学中的“内在-外在耦合”，证明学习者的当前观察并不决定其对后续训练的响应——相同的内在干预在不同外部延续下会产生非加性的、读出特定的交互效应，例如回放机制可将一次写入操作在32次更新中的贡献从五个正确预测变为零。

    

    学习者的当前观察结果并不一定决定其对后续训练的响应。我们通过受限学习状态干预的“延续条件化价值”来形式化内在-外在耦合，并用相对于观察的纤维来刻画当前的一致性。我们提出一种可执行的有限框架分类器头写入操作，在有限精度接受检查下，既保护当前的logits，又修复指定的历史边际。我们区分了局部可容许性、延续条件化干预价值以及完整策略性能三者的不同。一个匹配的四格对照实验识别出：相同的内在干预与不同的外部延续之间存在读出特定的非加性交互。在基于CLINC的类增量学习设置中，回放机制使该写入操作在32次更新中的贡献从五个正确预测变为零。在输出蒸馏、使用RoBERTa骨干网络以及优化器原生机制下，同样会出现非零的交互效应。

    arXiv:2609.30185v1 Announce Type: new  Abstract: A learner's current observations need not determine its response to further training. We formulate intrinsic-extrinsic coupling through the continuation-conditioned value of a constrained learning-state intervention, with observation-relative fibers describing present agreement. An executable finite-frame classifier-head write protects current logits while repairing specified historical margins under finite-precision acceptance checks. We distinguish local admissibility, continuation-conditioned intervention value, and complete-policy performance. A matched four-cell contrast identifies readout-specific non-additivity between the same intrinsic intervention and alternative external continuations. In a CLINC-derived class-incremental setting, replay changes the write's 32-update contribution from five correct predictions to zero. Nonzero interactions also occur under output distillation, with a RoBERTa backbone, and under optimizer-native
    
[^11]: GridSFM：用于求解交流最优潮流的基础模型

    GridSFM: A Foundation Model for Solving AC Optimal Power Flow

    [https://arxiv.org/abs/2609.30173](https://arxiv.org/abs/2609.30173)

    GridSFM是一个1500万参数的物理启发图神经网络基础模型，通过在54种电网拓扑上预训练并结合基于牛顿法的物理信息微调，仅需100个求解实例即可适应多达10000节点的未见电网，实现2.45%的零样本发电成本误差，且优于使用更多数据训练的单一拓扑专用神经网络模型。

    

    我们提出GridSFM，这是一个将跨电网拓扑预训练的基础模型与物理信息微调相结合的框架，用于大规模求解交流最优潮流（AC-OPF）。它是一个拥有1500万参数的物理启发图神经网络，在54种拓扑结构（500至4000节点）上进行了预训练。我们的模型在10000节点系统的保留运行工况上实现了2.45%的零样本发电成本误差，且随着系统规模增长性能没有退化。在此基础上，我们将预训练主干网络与基于牛顿潮流法的物理信息微调设计相结合。仅需100个已求解的实例，GridSFM即可适应多达10000节点的未见电网。我们证明，当作为热启动点部署时，该模型在成本和求解器迭代次数方面均优于使用更多数据训练的单一拓扑专用神经网络模型。在设计该基础模型时，我们克服了……（原文摘要至此截断）

    arXiv:2609.30173v1 Announce Type: cross  Abstract: We introduce GridSFM, a framework that combines a pretrained foundation model across grid topologies with physics-informed fine-tuning for solving AC Optimal Power Flow (AC-OPF) at scale. It is a $15$ million parameter physics-inspired graph neural network pretrained across $54$ topologies of $500$ to $4{,}000$ buses. Our model attains a $2.45\%$ zero-shot generation-cost error on a $10{,}000$ bus case held-out operating conditions with no degradation as system size grows. Building on this, we pair the pretrained backbone with a physics-informed fine-tuning design based on Newton's method for power flow. With only $100$ solved instances, GridSFM adapts to unseen grids up to $10{,}000$ buses. We show it out performs single topology, dedicated neural network models that are trained more data, both in terms of cost and solver iterations when deployed as warm starting points.   In designing this foundation model, we overcome the fact that 
    
[^12]: 音频语言模型听到与读到的区别特征是否一致？

    Do Audio Language Models Hear and Read Distinctive Features Alike?

    [https://arxiv.org/abs/2609.30167](https://arxiv.org/abs/2609.30167)

    该研究通过最小对立音素对与随机配对基准的分析发现，音频语言模型的解码器对音素区别特征的表征在听觉与阅读两种模态间普遍不一致，仅有Qwen2.5-Omni模型中的浊音特征表现出超过随机基准的跨模态方向一致性。

    

    音频语言模型让语音和文本经过同一个解码器。我们探究的问题是：当模型听到一个音素与读到该音素时，解码器是否在相同的方向上表征其区别特征。对于仅在一个特征上不同的最小对立音素对，我们计算两个成员平均表征之间的偏移量；对这些偏移量取平均得到每个模态（流）的方向，然后测量两个方向之间的余弦相似度。由于两个模态对于任意音素对本就存在一定程度的一致性，我们将所有度量与基于随机配对构建的参考基准进行比较，而非与零进行比较。我们将该方法应用于6个模型、7个特征以及来自11个语系的15种语言。经多重检验校正后，只有两个Qwen2.5-Omni模型中的浊音（voicing）特征超过了该参考基准，且该参考基准在不同模型之间相差达七倍。在六个模型中的三个里，浊音在音频模态下于14种拥有足够最小对立对可供测量的语言中共享同一个方向。

    arXiv:2609.30167v1 Announce Type: new  Abstract: Audio language models pass speech and text through a single decoder. We ask whether that decoder represents a distinctive feature in the same direction when a phoneme is heard and when it is read. For minimal pairs of phonemes differing in one feature, we take the offset between the two members' mean representations. Averaging those offsets gives a direction for each stream, and we measure the cosine between the two. Because the two streams already agree about arbitrary phoneme pairs, we compare every measure against a reference built from random pairings rather than against zero. We apply this to 6 models, 7 features and 15 languages from 11 families. Only voicing in the two Qwen2.5-Omni models exceeds that reference after correction for multiple testing, and the reference varies by a factor of seven between models. In three of the six models, voicing has one direction in audio across the 14 languages with enough minimal pairs to measur
    
[^13]: 量子网络中同时纠缠请求的策略学习与解释

    Learning and interpreting policies for simultaneous entanglement requests in quantum networks

    [https://arxiv.org/abs/2609.30157](https://arxiv.org/abs/2609.30157)

    该论文提出了一种结合消息传递神经网络（MPNN）的双深度Q网络强化学习方法，用于学习和解释量子网络中同时服务多个纠缠请求的纠缠资源调度策略。

    

    未来的量子网络将利用纠缠来执行众多任务，例如长距离量子信息传输、分布式量子计算和量子传感。通常，这些任务需要在网络的各个区域同时执行，同时最小化资源和延迟。因此，我们需要能够调度链路级纠缠资源的策略，并利用链路级纠缠来创建每个任务所需的各种形式的多体纠缠。在这项工作中，我们使用强化学习来解决这一问题。我们为该问题构建了马尔可夫决策过程（MDP），并采用结合消息传递神经网络（MPNN）的双深度Q网络（DQN），配合经验回放缓冲区和课程训练来获得策略。其中关键的物理参数是链路级纠缠产生的概率，即链路激活概率。我们展示了我们的策略...

    arXiv:2609.30157v1 Announce Type: cross  Abstract: Future quantum networks will make use of entanglement to perform numerous tasks, such as sending quantum information over long distances, distributed quantum computing, and quantum sensing. In general, these tasks will need to be performed simultaneously in various regions of a network, while minimizing resources and latency. We will thus require policies for scheduling link-level entanglement resources, and using the link-level entanglement to create various forms of multipartite entanglement required for every task. In this work, we address this problem using reinforcement learning. We formulate a Markov Decision Process for the problem and use double deep Q-networks (DQN) with Message Passing Neural Networks (MPNNs), experience replay buffers, and curriculum training to obtain policies. The key physical parameter is the probability of link-level entanglement generation, i.e., the link activation probability. We show that our policie
    
[^14]: 模型所述拒绝候选者的理由真的起作用吗？

    Does a model's stated reason for rejecting a candidate do any work?

    [https://arxiv.org/abs/2609.30151](https://arxiv.org/abs/2609.30151)

    该研究通过将模型声称缺失的事实插入对应档案并在贪心解码下重新测试，首次因果性地验证了语言模型拒绝候选者时所述理由确实会实际影响其后续选择。

    

    当被要求在候选者之间做出选择并解释理由时，语言模型常常通过指出对手档案中缺失的某个事实来拒绝对方：比如“没有导演”、“没有死亡日期”。这句话是对模型面前文本的一个断言，而且可以在不需要任何评判者的情况下加以检验。我们将陈述该所提及事实的真实语料句子插入对手的档案中，并在贪心解码下重新提问。两个对照实验将内容与位置因素区分开来：在同一档案中加入长度匹配的无关句子，以及在模型从未提及的第三个选项处加入相同的两句话。在三次实验中规模最大的一次里——六个开源模型在2WikiMultihopQA数据集上——在模型所指出的档案处提供该所提及事实，比无关对照更能改变模型的选择，几率比为3.57 [1.54, 8.26]，Holm校正后p=0.0210，且该结果在剔除任何单个模型后依然成立。而该实验设计旨在检测的关键对照——在无人提及的选项处提供相同事实——未能通过多重比较校正（Holm……）

    arXiv:2609.30151v1 Announce Type: cross  Abstract: Asked to choose between candidates and explain the choice, a language model often rejects a rival by naming a fact its profile lacks: no director, no date of death. That sentence is a claim about the text in front of the model, and it can be tested without any judge. We insert a real corpus sentence stating the named fact into the rival's profile and ask again under greedy decoding. Two controls separate content from placement: a length-matched irrelevant sentence at the same profile, and the same two sentences at a third option the model never mentioned. In the largest of three runs, six open models on 2WikiMultihopQA, supplying the named fact at the profile the model named moves its choice more than the irrelevant control does, odds ratio 3.57 [1.54, 8.26], Holm p=0.0210, and this survives dropping any single model. The contrast the design was built to detect, the same fact at the option nobody named, does not clear correction, Holm 
    
[^15]: 基于图的推断与拓扑感知多智能体强化学习用于大规模铁路网络管理

    Graph-Based Inference and Topology-Aware Multi-Agent Reinforcement Learning for Large-Scale Railway Network Management

    [https://arxiv.org/abs/2609.30150](https://arxiv.org/abs/2609.30150)

    该论文提出一个将基于图上高斯过程核的分层贝叶斯环境建模与拓扑感知多智能体强化学习相结合的图基框架，利用瑞士联邦铁路的真实数据，实现了大规模铁路网络维护管理的可扩展决策支持。

    

    现代基础设施资产管理构成了一个复杂的序贯决策问题，其特点是规划周期长且存在系统级交互，例如空间退化相关性和规模经济效应。虽然深度强化学习在优化维护策略方面展现出前景，但将其扩展到真实世界网络仍然具有挑战性。集中式方法在大规模系统中计算上难以处理，而分布式方法往往无法捕捉关键的协调机制。为了应对这些挑战，我们提出了一个基于图的框架，该框架将精确的环境建模与可扩展的决策支持相结合。首先，我们采用分层贝叶斯模型，利用图上的高斯过程核，从瑞士联邦铁路提供的真实数据中推断出一个真实且具有空间相关性的铁路维护规划网络环境。其次，我们……（摘要内容在此处截断）

    arXiv:2609.30150v1 Announce Type: new  Abstract: Modern infrastructure asset management constitutes a complex sequential decision-making problem, characterized by long planning horizons and system-level interactions, such as spatial deterioration correlations and economies of scale. While deep reinforcement learning has shown promise in optimizing maintenance policies, scaling to real-world networks remains challenging. Centralized approaches become computationally intractable in large-scale systems, whereas decentralized approaches often fail to capture essential coordination mechanisms. To address these challenges, we propose a graph-based framework that integrates accurate environment modeling with scalable decision support. First, we employ a hierarchical Bayesian model leveraging a Gaussian Process on Graph kernel to infer a realistic, spatially correlated networked environment of railway maintenance planning from real-world data provided by the Swiss Federal Railways. Second, we 
    
[^16]: GRASP：基于智能体AI的策略规划生成、修订与评估框架

    GRASP: Generating, Revising, and Assessing for Strategic Planning with Agentic AI

    [https://arxiv.org/abs/2609.30147](https://arxiv.org/abs/2609.30147)

    GRASP是一个策略感知的多阶段规划框架，通过将规划流程解耦为生成、修订和评估三个上下文隔离的专门模块，显著提升了LLM在复杂任务上的规划准确率，在多个基准数据集上建立了新的最先进水平。

    

    大型语言模型（LLMs）通常表现出一种性能特征，即随着任务复杂性的增加，其可靠性会下降。我们通过引入GRASP——一个具有策略感知能力的多阶段规划框架——来解决为复杂任务生成高质量自然语言可执行计划的挑战。GRASP将规划流程解耦为多个专门化、上下文隔离的模块：它预编译全局宏观指导方针，在隔离的上下文窗口中探索备选的局部策略，并使用多标准判别器独立评估轨迹。实证评估表明，GRASP在多个数据集上持续确立了新的最先进水平，与直接的LLM规划器相比，在Natural Plan Calendar Scheduling（提升约12.4%）、ZebraLogic（提升约30.8%）和SciBench Math上取得了显著的准确率提升。

    arXiv:2609.30147v1 Announce Type: new  Abstract: Large Language Models (LLMs) typically exhibit a performance profile where reliability degrades as task complexity increases. We address the challenge of generating high-quality natural language executable plans for complex tasks by introducing $\textbf{GRASP}$, a strategy-aware, multi-stage planning framework. GRASP decouples the planning pipeline across specialized, context-isolated modules: it pre-compiles global macro-guidelines (GenPlan), explores alternative localized strategies within isolated context windows (RevPlan), and independently evaluates trajectories using a multi-criteria discriminator (VerPlan). Empirical evaluations show that GRASP consistently establishes a new state-of-the-art frontier across diverse datasets, yielding substantial accuracy gains over direct LLM planners on Natural Plan Calendar Scheduling ($\sim$12.4$\%$$\uparrow$), ZebraLogic ($\sim$30.8$\%$$\uparrow$), and SciBench Math. Crucially, under multi-tas
    
[^17]: 轨道误差动力学：零存储神经合成中的自组织临界性、瞬态参数共振与非线性生物本体论

    Orbital Error Dynamics: Self-Organized Criticality, Ephemeral Parameter Resonance, and Non-Linear Biological Ontologies in Zero-Storage Neural Synthesis

    [https://arxiv.org/abs/2609.30115](https://arxiv.org/abs/2609.30115)

    该论文提出轨道误差动力学（OED）框架，将神经网络权重从静态存储矩阵（O(W)）改为由复二次映射 z=z²+c 程序化生成的瞬态拓扑共振（O(1)），实现零存储神经合成，并结合弯曲正弦波假说、观察者视界几何与重尾仿生扰动来定位共振区域并摆脱非凸优化停滞。

    

    现代深度神经网络将参数视为存储在物理内存中的静态浮点矩阵，这带来了冯·诺依曼内存瓶颈和表示坍缩问题。我们提出了轨道误差动力学（Orbital Error Dynamics, OED）这一分析框架，其中突触权重不再是存储的质量（O(W)），而是由复二次多项式映射 z_{n+1} = z_n^2 + c 程序化推导出的瞬态拓扑共振（O(1)）。我们引入了弯曲正弦波假说，证明当谐波波在环境阻力作用下向内卷曲、趋向心形尖点（c = 1/4）时，非平衡的生命系统随之涌现。我们定义了参数空间中的观察者视界几何，识别出位于不动点盆地与真实边界 c = 0.25 ± 0.50i 之间的内部共振肩部轨迹 X_upper = (0.25, +0.18) 和 X_lower = (0.25, -0.18)。为了在不进行损失清零的情况下摆脱非凸停滞，我们引入了一种重尾的仿生……（原文摘要在此处被截断）

    arXiv:2609.30115v1 Announce Type: cross  Abstract: Modern deep neural networks treat parameters as static floating-point matrices stored in physical memory, incurring Von Neumann memory bottlenecks and representation collapse. We formulate Orbital Error Dynamics (OED), an analytical framework wherein synaptic weights are not stored masses (O(W)), but transient topological resonances (O(1)) derived procedurally from the complex quadratic polynomial map z_{n+1} = z_n^2 + c. We introduce the Bent Sine Wave Hypothesis, demonstrating that non-equilibrium living systems emerge when harmonic waves curl inward through environmental drag toward the cardioid cusp (c = 1/4). We define the Observer Horizon Geometry in parameter space, identifying interior resonance shoulder loci X_upper = (0.25, +0.18) and X_lower = (0.25, -0.18) between the fixed-point basin and the true boundary at c = 0.25 +/- 0.50i. To escape non-convex stagnation without loss zeroing, we introduce a heavy-tailed Biomimetic Pe
    
[^18]: 论对数凹分布的平方和可认证性

    On the SoS Certifiability of Log-Concave Distributions

    [https://arxiv.org/abs/2609.30105](https://arxiv.org/abs/2609.30105)

    该论文证明了对数凹分布的矩多项式具有不依赖于 Poincaré 常数的平方和证书，从而恢复了对数凹分布的最优矩界，并为高维统计估计问题带来了具有维度无关误差保证的高效算法。

    

    对于 $\mathbb{R}^d$ 上任意各向同性的对数凹分布 $P$，我们证明了对每个偶数 $m\ge 2$，多项式 $(Cm)^m\|v\|_2^m - \mathbb{E}_{X\sim P}\langle X,v\rangle^m$ 是平方和，其中 $C>0$ 是一个通用常数。这一结果去除了 Kothari 和 Steinhardt（arXiv:1711.07465）定理中对 Poincaré 常数的依赖，恢复了对数凹分布的最优矩界。作为一个直接推论，我们为一大类高维统计估计问题获得了具有与维度无关误差保证的计算高效算法。我们的证明使用随机局部化技术将 $P$ 分解为若干随机强对数凹测度的平均，而这些测度的中心矩具备 Diakonikolas、Hopkins、Pensia 和 Tiegel（STOC 2025; arXiv:2410.21194）所给出的次高斯证书。通过一种与协方差相适应的局部化方式选取，我们证明了一个四阶矩证书……（原文摘要在此处截断）

    arXiv:2609.30105v1 Announce Type: new  Abstract: For an arbitrary isotropic log-concave distribution $P$ on $\mathbb{R}^d$, we prove that the polynomial $(Cm)^m\|v\|_2^m - \mathbb{E}_{X\sim P}\langle X,v\rangle^m$ is a sum of squares for every even $m\ge2$, where $C>0$ is a universal constant. This removes the dependence on the Poincar\'e constant in the theorem of Kothari and Steinhardt (arXiv:1711.07465), recovering the optimal moment bounds for log-concave distributions. As an immediate corollary, we obtain computationally efficient algorithms with dimension-free error guarantees for a wide range of high-dimensional statistical estimation problems.   Our proof uses stochastic localization to decompose $P$ as an average of random strongly log-concave measures, whose centered moments admit the subgaussian certificates of Diakonikolas, Hopkins, Pensia, and Tiegel (STOC 2025; arXiv:2410.21194). With a covariance-adapted choice of localization, we show that a fourth-moment certificate de
    
[^19]: MQSS-Selector：基于强化学习引导的MLIR编译流水线Pass选择

    MQSS-Selector: RL-Guided Pass Selection for an MLIR Compilation Pipeline

    [https://arxiv.org/abs/2609.30104](https://arxiv.org/abs/2609.30104)

    本文提出MQSS-Selector，一个利用强化学习自动为MLIR量子编译流水线选择优化Pass的统一学习型选择器，以在碎片化的量子软件栈中提升NISQ时代量子程序的编译保真度。

    

    高性能计算（HPC）与量子计算（QC）系统正日益向统一的HPCQC（高性能计算-量子计算）基础设施融合，这一趋势源于连接经典与量子工作流的日益增长的需求，并影响到系统堆栈的各个层面——从硬件、编译器和运行时，一直到应用程序。然而，当今的量子计算设备仍处于含噪声中等规模量子（NISQ）时代，容易出错且资源受限，因此需要专门的优化和拓扑映射才能达到足够的保真度。这使得整个量子软件栈中的编译与优化显得尤为关键。许多现有软件栈仍然碎片化，由相互独立的组件分别负责设备选择、编译器Pass优化和作业队列调度。本文提出了一个统一的、基于学习的选择器，将这些彼此分散的组件整合在一起。

    arXiv:2609.30104v1 Announce Type: cross  Abstract: High Performance Computing (HPC) and Quantum Computing (QC) systems are increasingly converging towards unified High Performance Computing-Quantum Computing (HPCQC) infrastructures, driven by a growing need to bridge classical and quantum workflows, which affects all levels of the system stack, from the hardware to compilers and runtimes, all the way to applications. However, today's QC devices are still in the Noisy Intermediate-Scale Quantum (NISQ) era, are error-prone and resource-limited, and therefore require specialized optimizations and topology mappings to achieve sufficient fidelity. This places special emphasis on proper compilation and optimization within the overall quantum software stack. Many existing stacks remain fragmented, with separate components responsible for device selection, compiler-pass optimization, and job queue scheduling. This paper proposes a unified, learning-based selector that integrates these disparat
    
[^20]: AT-SKM-Net：一种面向动态图上线性硬约束可行性的加速可训练采样Kaczmarz-Motzkin框架

    AT-SKM-Net: An Accelerated Trainable Sampling Kaczmarz-Motzkin Framework for Linear Hard-Constraint Feasibility on Dynamic Graphs

    [https://arxiv.org/abs/2609.30088](https://arxiv.org/abs/2609.30088)

    本文提出AT-SKM-Net框架，通过拓扑感知异构GNN引导的混合采样策略与Cholesky更新机制，将动态图上线性硬约束可行性问题的等式投影复杂度从O(N³)降至O(N²)，实现计算加速。

    

    带线性约束的图结构优化是关键基础设施的重要基础，但由于存在海量严格硬约束和高维度问题而面临可扩展性限制。尽管近期基于投影的方法（如可训练采样Kaczmarz-Motzkin网络 T-SKM-Net）能够保证可行性，但它们在动态环境中需要处理整个约束集并进行昂贵的矩阵分解，因而计算成本高昂。为弥补这一不足，我们提出了加速可训练SKM（AT-SKM）网络框架。为将计算集中于活跃约束并消除冗余计算，我们引入了一种由拓扑感知异构GNN模型引导的混合采样策略。为高效处理基于图的约束中的拓扑变化，我们采用了Cholesky更新机制，从理论上将低秩扰动下等式投影的复杂度从O(N^3)降低至O(N^2)。实验…

    arXiv:2609.30088v1 Announce Type: cross  Abstract: Graph-structured optimization with linear constraints is fundamental to critical infrastructure but faces scalability limits due to massive strict hard constraints and high dimensionality. While recent projection-based methods such as Trainable Sampling Kaczmarz-Motzkin Net (T-SKM-Net) guarantee feasibility, they face high computational costs in dynamic environments by processing the entire constraint set and requiring expensive matrix factorizations. To bridge this gap, we propose the Accelerated Trainable-SKM (AT-SKM) Net framework. To concentrate computation on the active constraints and eliminate redundant calculations, we introduce a hybrid sampling strategy guided by a topology-aware heterogeneous GNN model. To efficiently handle topological shifts in graph-based constraints, we employ a Cholesky Update mechanism that theoretically reduces the equality projection complexity from O(N^3) to O(N^2) under low-rank perturbations. Expe
    
[^21]: 返回还是修订？学习修订何时有助于检索增强问答

    Return or Revise? Learning When Revision Helps Retrieval-Augmented QA

    [https://arxiv.org/abs/2609.30087](https://arxiv.org/abs/2609.30087)

    本文提出“可恢复性”指标——通过在同一评判标准下同时评估草稿答案与其候选修订所得到的成对效果——并训练模型在修订前预测该指标，从而判断何时进行检索增强修订有益，在多个实验设置下均优于仅基于草稿置信度的决策方法。

    

    我们考虑这样一个决策问题：在答案修订系统中，是直接返回已有的草稿答案，还是利用检索到的证据对其进行修订。草稿置信度估计的是当前答案是否正确，但这一决策需要估计某次特定修订所带来的效果。为了进行离线训练与评估，我们在同一正确性评判标准下对返回的草稿答案及其候选修订同时进行评分，这使得修复效果、潜在损害以及与最优答案之间的差距变得可观测。我们将这种成对效应称为“可恢复性”（recoverability），并训练策略在修订之前对其进行预测。在三个修订设置下共 25,870 个留出的开放域问题上，基于成对结果训练的评分器在全部九个 Llama 设置-随机种子组合中，其“准确率-修订率”曲线下面积均优于同等条件的草稿正确性评分器，并且在由开发集选定的阈值下平均提升 0.23–0.68 个准确率百分点，该差异仅在不同训练运行之间具有统计显著性。

    arXiv:2609.30087v1 Announce Type: new  Abstract: We consider the decision of whether to return an existing draft answer or revise it using retrieved evidence, as in answer-revision systems. Draft confidence estimates whether the current answer is correct, but the decision requires estimating the effect of a specified revision. For offline training and evaluation, we grade both the returned draft and its candidate revision under the same correctness judge, which makes repair, harm, and the gap to an oracle observable. We call this paired effect its recoverability, and we train policies to predict it before revision. On 25,870 held-out open-domain questions across three revision setups, a scorer trained on the paired outcome has greater area under the accuracy--revision-rate curve than a matched draft-correctness scorer in all nine Llama setup--seed fits, and gains 0.23--0.68 accuracy points on average at development-selected thresholds, a difference significant across training runs only
    
[^22]: 残差相关性作为GP协同区域化联合不确定性增益的诊断指标

    Residual Correlation as a Diagnostic for Joint-Uncertainty Gains from GP Coregionalisation

    [https://arxiv.org/abs/2609.30085](https://arxiv.org/abs/2609.30085)

    该论文发现GP协同区域化的主要收益在于联合不确定性量化而非点预测，并提出残差相关性（而非原始目标相关性）才是预测联合不确定性增益的最强指标，同时给出了可从独立逐目标残差计算的轻量级诊断指标D_logdet来量化这一增益。

    

    在多目标回归中，相关目标通常通过具有内在协同区域化模型（GP-ICM）的多输出高斯过程进行耦合，其假设是共享统计强度能够提升整体性能。然而在实践中，其收益并不一致。在所研究的各种设置中，我们发现协同区域化的主要收益在于联合不确定性量化，而非点预测。原始目标相关性无法预测耦合何时有益；在此研究的可分离GP-ICM设置中，残差相关性——即独立的逐目标预测器无法解释的跨目标依赖性——是联合不确定性增益的最强预测因子。我们引入了一个轻量级诊断指标 D_logdet = -1/2 log det R_res，它表示通过建模完整的残差协方差而非对角残差协方差所获得的理想化联合负对数似然（NLL）增益，且该指标可由独立的逐目标残差计算得出。

    arXiv:2609.30085v1 Announce Type: new  Abstract: In multi-target regression, correlated targets are often coupled through multi-output Gaussian processes with an intrinsic model of coregionalisation (GP-ICM), assuming that sharing statistical strength improves overall performance. In practice, the benefits are inconsistent. Across the settings studied, we find that the main benefit of coregionalisation is joint uncertainty quantification rather than point prediction. Raw target correlation does not predict when coupling helps; in the separable GP-ICM settings studied here, residual correlation, the cross-target dependence left unexplained by independent per-target predictors, is the strongest predictor of joint-uncertainty gains.   We introduce a lightweight diagnostic, $D_{\rm logdet}=-\frac{1}{2}\log\det R_{\rm res}$, which represents the idealised joint negative log-likelihood (NLL) gain from modelling a full rather than diagonal residual covariance and is computable from independen
    
[^23]: 基于可达性的含节点与边特征的图神经网络形式化验证

    Reachability-Based Formal Verification of Graph Neural Networks with Node and Edge Features

    [https://arxiv.org/abs/2609.30079](https://arxiv.org/abs/2609.30079)

    本文提出GraphStar集合这一Star集合的推广形式，将神经网络验证框架扩展至带节点与边特征的图神经网络，实现了对电力系统中潮流分析、最优潮流估计和连锁故障分析任务的可达性形式化验证。

    

    arXiv:2609.30079v1 公告类型：交叉  摘要：图神经网络（GNNs）已成为在电力系统中开发快速、拓扑感知代理模型的一种突出方法，可支持潮流分析（PF）、最优潮流估计（OPF）以及连锁故障分析（CFA）等任务。尽管其应用日益增多，但对基于GNN的模型进行形式化验证仍然具有挑战性，现有方法的适用范围有限。我们通过GraphStar集合将神经网络验证（NNV）框架扩展到图结构输入。GraphStar集合是Star集合的一种推广，能够同时捕获节点特征和边特征上的不确定性。这一扩展使得线性消息传递操作能够进行传播，并能对GNN架构（包括图卷积网络（GCN）和带边特征的图同构网络（GINE）层）中的ReLU非线性进行可靠近似。我们在IEEE-24、IEEE-39和IEEE-118测试系统上，对PF、OPF和CFA三个电力系统任务评估了GNNV（原文摘要至此截断）

    arXiv:2609.30079v1 Announce Type: cross  Abstract: Graph neural networks (GNNs) have become a prominent approach for developing fast, topology-aware surrogates in electric power systems, supporting tasks such as power flow (PF) analysis, optimal power flow (OPF) estimation, and cascading failure analysis (CFA). Despite this growing use, formally verifying GNN-based models remains challenging, with existing methods limited in scope. We extend the neural network verification (NNV) framework to graph-structured inputs through GraphStar sets, a generalization of Star sets that captures uncertainty over both node and edge features. This extension enables the propagation of linear message-passing operations and the sound approximation of ReLU nonlinearities for GNN architectures, including graph convolutional network (GCN) and graph isomorphism network with edge features (GINE) layers. We evaluate GNNV across three power system tasks, PF, OPF, and CFA, on the IEEE-24, IEEE-39, and IEEE-118 t
    
[^24]: 核范数正则化的贝叶斯矩阵补全

    Nuclear Norm-Regularized Bayesian Matrix Completion

    [https://arxiv.org/abs/2609.30078](https://arxiv.org/abs/2609.30078)

    本文提出了首个针对未知噪声方差下核范数正则化贝叶斯矩阵补全模型的采样器，并给出了复杂度为矩阵维度和目标精度倒数之多项式的显式非渐近保证。

    

    矩阵补全是指从带噪声观测的元素中估计矩阵中缺失元素的问题，它是推荐系统以及面板数据中反事实结果估计等众多问题的基础。许多算法采用正则化最小二乘法来解决该问题，通常以核范数作为正则化项，但这种方法只能产生点估计，缺乏内置的不确定性量化。贝叶斯公式是一种自然的替代方案：如果噪声方差已知，基于核范数的先验可以产生对数凹的后验分布。然而遗憾的是，在实践中噪声方差并非先验已知，因此要实现完全贝叶斯方法，必须对噪声方差施加先验。我们给出了该模型的第一个具有显式非渐近保证的采样器：其复杂度关于矩阵维度以及目标精度的倒数是多项式级的。我们的技术是将噪声精度的分布离散化到……

    arXiv:2609.30078v1 Announce Type: cross  Abstract: Matrix completion, the problem of estimating missing entries in a matrix from noisily observed ones, underlies a diverse array of problems such as recommender systems and counterfactual outcome estimation in panel data. Many algorithms address the problem using regularized least squares, often with the nuclear norm as a regularizer, but this method yields a point estimate with no built-in uncertainty quantification. A Bayesian formulation is a natural alternative, and if the noise variance is known, the nuclear norm-based prior yields a log-concave posterior. Unfortunately, in practice, the noise variance will not be known a priori, so for a fully Bayesian approach, a prior must be imposed on it. We give the first sampler for this model with an explicit non-asymptotic guarantee: polynomial in the matrix dimensions and in the reciprocal of the target accuracy. Our technique is to discretize the distribution of the noise precision onto a
    
[^25]: 评估结论的可复现性如何？对LLM推断提示结构的自我审计

    How Reproducible Are Evaluation Conclusions? A Self-Audit of LLM-Inferred Prompt Structure

    [https://arxiv.org/abs/2609.30074](https://arxiv.org/abs/2609.30074)

    这项研究通过对LLM提示结构推断的自我审计发现，小规模提示集产生的模型评估排名中只有最差模型的位置是可靠的，而中间和头部模型的排名在不同重复实验中极不稳定。

    

    对LLM系统的评估通常在小规模提示集上取平均值，并以排名表的形式报告模型表现。我们提出这样一个问题：这样的排名表值得多少信任？并以基于LLM的提示结构推断作为案例研究：涵盖五个系列的八个开放模型变体，参数量从8B到675B，禁用缓存，持久化了293个原始中间表示。所测量的现象本身就是不稳定的：相同的调用无法可靠地恢复相同的结构，节点集Jaccard相似度均值从0.39到0.96不等，72%的提示-模型组合从未达到完美的节点集匹配。对评估过程本身的审计进一步削弱了其结论，这是我们的主要贡献。在基于提示的联合聚类自助法检验下，只有排名的底部是稳固的：可复现性最差的两个模型在99%和86%的重复实验中保持排名不变，中间四个模型仅占27%到48%，排名前两位的模型各占68%。因此，该排名表能够可靠地识别最差的模型，但无法可靠地确定其余模型的位置。

    arXiv:2609.30074v1 Announce Type: cross  Abstract: Evaluations of LLM systems routinely average over small prompt sets and report models as a ranked table. We ask how much confidence such a table deserves, using LLM-based prompt-structure inference as the case study: eight open model variants across five families and 8B to 675B parameters, caching disabled, 293 raw intermediate representations persisted. The measured phenomenon is unstable to begin with. Identical calls do not reliably recover identical structure, with mean node-set Jaccard from 0.39 to 0.96 and 72% of prompt-model cells never node-set-perfect. Auditing the evaluation weakens its conclusions further, and this is our main contribution. Under a joint cluster bootstrap over prompts, only the bottom of the ranking is firm: the two least reproducible models hold rank in 99% and 86% of replicates, the middle four in 27% to 48%, and the top two in 68% each, so the table identifies the worst model reliably but does not reliabl
    
[^26]: KernelOPT：面向GPU内核优化的调度感知智能体搜索

    KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization

    [https://arxiv.org/abs/2609.30059](https://arxiv.org/abs/2609.30059)

    KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。

    

    深度学习的推理与训练性能在很大程度上取决于GPU内核的效率。现代编译器（如PyTorch Inductor）能够从高层模型代码自动生成GPU内核，但其性能常常大幅落后于专家手写的实现。近期基于大语言模型（LLM）辅助的内核优化器虽然能缩小独立内核方面的这一差距，但它们将编译后的模型视为黑盒，通常只优化单个独立内核，既不尊重编译器的结构性决策，也不进行模型级的端到端验证。我们提出了KernelOPT，一个将编译后模型视为结构化产物的多智能体系统。该系统保留厂商库调用（cuBLAS、cuDNN），仅针对生成的Triton子内核，并使用五个由性能剖析引导的LLM智能体进行优化。一个由静态校验、多种子正确性验证、模型级float64回退验证以及性能门控组成的四道验证级联，在优化过程中对候选内核进行筛选（原文摘要在此处截断）。

    arXiv:2609.30059v1 Announce Type: cross  Abstract: Deep learning inference and training performance depends critically on GPU kernel efficiency. Modern compilers such as PyTorch Inductor automatically generate GPU kernels from high-level model code, but frequently underperform expert-written implementations by wide margins. Recent LLM-assisted kernel optimizers can close this gap for standalone kernels, yet treat compiled models as black boxes, generally optimizing individual standalone kernels without respecting the compiler's structural decisions or verifying the model end-to-end. We present KernelOPT, a multi-agent system that treats compiled models as structured artifacts. It preserves vendor library calls (cuBLAS, cuDNN) and exclusively targets generated Triton sub-kernels using five profiling-guided LLM agents. A four-gate verification cascade of static validation, multi-seed correctness, model-level float64-fallback verification, and performance gating filters candidates during 
    
[^27]: 从工艺到功能：铜嵌入式SiO$_x$忆阻器件中可及材料态的工程化调控

    From Processing to Functionality: Engineering Accessible Material States in Cu-Embedded SiO$_x$ Memristive Devices

    [https://arxiv.org/abs/2609.30047](https://arxiv.org/abs/2609.30047)

    本研究建立了连接等离子体沉积工艺与SiO$_x$/Cu/SiO$_x$忆阻器件宏观功能的多尺度框架，结合5万余个器件的统计分析和物理模拟，揭示器件行为源于缺陷形成到功能区涌现的概率性级联过程，而重构氧空位密度可作为调控器件功能的有效潜在描述符。

    

    氧化物基器件中的阻变行为普遍受随机缺陷过程支配，然而制造条件与功能行为之间的预测性关联仍难以捉摸。在此，我们建立了一个多尺度框架，将等离子体定义的沉积条件与溅射SiO$_x$/Cu/SiO$_x$体系的宏观器件功能联系起来。通过对50,000多个实验表征器件的大规模统计分析，并结合基于物理的等离子体模拟与原子尺度模拟，我们表明器件行为并非源于确定性的“工艺-性能”映射，而是源于跨越缺陷形成、缺陷态演化与功能区涌现的概率性级联过程。数据驱动的聚类分析揭示了一个由可操作开关类型构成的连续功能态空间，而逆向建模则识别出重构的氧空位密度作为有效的潜在描述符……

    arXiv:2609.30047v1 Announce Type: cross  Abstract: Resistive switching in oxide-based devices is widely governed by stochastic defect processes, yet a predictive link between fabrication conditions and functional behavior remains elusive. Here, we establish a multiscale framework connecting plasma-defined deposition conditions to macroscopic device functionality in sputtered SiO$_x$/Cu/SiO$_x$-based systems. By combining large-scale statistical analysis of more than 50,000 experimentally characterized devices with physics-based plasma and atomistic simulations, we show that device behavior does not emerge from deterministic process-to-performance mappings, but from a probabilistic cascade spanning defect formation, defect-state evolution, and functional-regime emergence. Data-driven clustering reveals a continuous functional state space composed of operational switching types, while inverse modeling identifies the reconstructed oxygen-vacancy density as an effective latent descriptor c
    
[^28]: AERIAL：精度保持型低精度EEG解码器的鲁棒性对抗评估

    AERIAL: Adversarial Evaluation of Robustness in Accuracy-Preserving Low-Precision EEG Decoders

    [https://arxiv.org/abs/2609.30037](https://arxiv.org/abs/2609.30037)

    该研究提出AERIAL评估框架，系统发现精度保持型压缩（剪枝、PTQ、QAT）虽不改变EEG解码器的直接对抗鲁棒性，但剪枝会显著降低不同精度模型之间对抗样本的双向迁移效率。

    

    面向部署的模型压缩对于资源受限的脑机接口（BCI）颇具吸引力，但其是否会改变模型的对抗脆弱性仍不清楚。在BCI竞赛IV-2a数据集上，我们在九名受试者和三个随机种子下，比较了32位浮点（FP32）的EEGNet和ShallowConvNet模型与全局幅度剪枝、模拟INT8训练后量化（PTQ）以及量化感知训练（QAT）方法。仿真提供了可微分的量化-反量化模型，用于白盒攻击和梯度分析，同时采用原生TensorRT部署进行验证。结果表明，保持精度的压缩并不能提升直接鲁棒性：在ε=0.005时，EEGNet的PGD准确率在FP32、50%剪枝（P50）、PTQ和QAT之间均保持在22-24%。然而，P50将双向对抗迁移效率降低至0.963/0.928（FP32→P50/P50→FP32），而PTQ则为0.994/0.997；ShallowConvNet也呈现出相同的趋势。

    arXiv:2609.30037v1 Announce Type: cross  Abstract: Deployment-oriented compression is attractive for resource-constrained brain--computer interfaces (BCIs), but whether it changes adversarial vulnerability remains unclear. On BCI Competition IV-2a, we compare 32-bit floating-point (FP32) EEGNet and ShallowConvNet models with global magnitude pruning and simulated INT8 post training quantization (PTQ) and quantization-aware training (QAT) across nine subjects and three seeds. Simulation provides differentiable quantize--dequantize models for white-box attacks and gradient analysis, while native TensorRT deployment is used for validation. Accuracy-preserving compression does not improve direct robustness: at $\epsilon=0.005$, EEGNet PGD accuracy remains 22--24\% across FP32, 50\% pruning (P50), PTQ, and QAT. However, P50 reduces bidirectional transfer efficiency to 0.963/0.928 (FP32$\rightarrow$P50/P50$\rightarrow$FP32), versus 0.994/0.997 for PTQ; the same trend holds for ShallowConvNet
    
[^29]: 短程瞄准以致远：你的冻结世界模型比你想象的更会规划

    Aim Short to Reach Far: Your Frozen World Model Can Plan Better Than You Think

    [https://arxiv.org/abs/2609.30036](https://arxiv.org/abs/2609.30036)

    提出锚定规划方法，通过瞄准从经验中检索的中间观测目标而非最终目标图像，使冻结的世界模型无需额外训练即可在长程任务规划中全面超越原有规划器。

    

    基于视觉世界模型构建的规划器通常通过预测结果与编码目标图像之间的距离来为每个预测结果评分。我们证明，即使动力学完全精确、短程搜索全局最优，这一目标也可能限制控制效果：到达目标可能需要最初远离目标的动作。使用冻结的 LeWM 模型，中间目标在 Cube、PushT、Reacher 和 TwoRoom 任务上显著改善了动作合成与已记录动作的排序。学习得到的目标和从观测经验中提取的目标都能带来这些收益。我们提出了锚定规划，该方法检索一段其起点和终点分别与当前观测和目标观测相似的已记录片段，然后瞄准该片段起点之后不久的一个观测。冻结模型从当前状态出发，向该目标对动作进行评分。在无需任何额外训练的情况下，向观测目标进行规划在我们长程评估中的每个任务上都优于已发布的 LeWM 规划器。

    arXiv:2609.30036v1 Announce Type: new  Abstract: Planners built on visual world models commonly score each predicted outcome by its distance to the encoded goal image. We show that this target can limit control even with exact dynamics and globally optimal short-horizon search: reaching a goal may require actions that initially move away from it. With frozen LeWM models, intermediate targets substantially improve action synthesis and recorded-action ranking on Cube, PushT, Reacher, and TwoRoom. Learned targets and targets drawn from observed experience both produce these gains. We introduce Anchored Planning, which retrieves a recorded segment whose start and end resemble the current and goal observations, then aims at an observation shortly after its start. The frozen model scores actions toward this target from the current state. Without additional training, planning toward observed targets outperforms the released LeWM planner on every task in our long-range evaluation. Additional f
    
[^30]: Canopy：利用分段平滑树先验的多保真度老虎机算法

    Canopy: Exploiting Piecewise Smooth Tree Priors for Multi-Fidelity Bandits

    [https://arxiv.org/abs/2609.30017](https://arxiv.org/abs/2609.30017)

    CANOPY是一种多保真度树状老虎机算法，通过廉价的随机路径探测在线学习分段平滑先验在树结构中的有效区域，从而突破传统方法需预先假设全局平滑性的局限，更精准地引导昂贵的叶节点评估。

    

    许多LLM推理问题，包括模型路由、前缀缓存管理、提示词裁剪和测试时搜索，都可以被视为对树结构的优化。这种结构自然地源于自回归生成：每个前缀定义一个节点，其后续延续构成其下方的一个子树。树的内部节点能够提供对某区域价值的廉价但有偏的估计，而叶节点评估则昂贵但准确。层次化老虎机方法可以利用这种结构，但通常需要预先指定特定的平滑性调度，尽管实际目标往往只是分段平滑的，且其最优值可能位于急剧变化的边界附近。我们提出了CANOPY，这是一种多保真度树状老虎机算法，它通过学习确定平滑先验在何处有效，而不是全局地假设其成立。CANOPY使用廉价的随机路径探测来构建局部聚合偏差的在线证书，然后将昂贵的叶节点评估引导至最值得评估的区域。

    arXiv:2609.30017v1 Announce Type: cross  Abstract: Many LLM inference problems, including model routing, prefix-cache management, prompt trimming, and test-time search, can be viewed as optimization over a tree. This structure arises naturally from autoregressive generation: every prefix defines a node, and its continuations form a subtree below it. Internal nodes of the tree provide cheap but biased estimates of a region's value, while leaf evaluations are expensive but accurate. Hierarchical bandit methods can exploit this structure, but typically require a specific smoothness schedule to be specified in advance, even though real objectives are often only piecewise smooth and their optima may lie near sharp boundaries. We introduce CANOPY, a multi-fidelity tree bandit that learns where the smoothness prior is valid rather than assuming it globally. CANOPY uses cheap random-path probes to construct an online certificate of local aggregation bias, then directs expensive leaf evaluation
    
[^31]: GHOST-Q：研究量化视觉语言模型中被忽视的同分数权衡下的接地幻觉问题

    GHOST-Q: Towards Studying Grounding Hallucinations Overlooked Under Same-score TradeOffs in Quantized VLMS

    [https://arxiv.org/abs/2609.29999](https://arxiv.org/abs/2609.29999)

    该论文提出GHOST-Q评估框架，通过将FP16与量化VLM的预测逐项配对，揭示出量化模型即使在总体准确率几乎不变的情况下，其视觉接地与幻觉行为仍发生显著变化，且内存大幅节省并不保证推理延迟降低。

    

    视觉-语言模型（VLM）的训练后量化通常通过总体任务准确率和内存节省来评估，但保持总体分数并不能保证视觉接地行为得以保留。我们提出GHOST-Q，这是一项跨精度的受控评估，在FP16、INT8和NF4三种精度下，对三个8B规模的VLM系列在实用性和幻觉敏感基准上进行了评测。我们不仅仅比较总体准确率，而是将FP16与量化模型的预测逐项配对，以量化压缩如何重新分配接地任务的成功与失败。结果显示，六个量化变体中有五个将MMStar准确率保持在±2个百分点以内，然而经过错误发现率校正后，36个配对效应中仍有10个保持显著，其中九个出现在幻觉敏感条件下。同设备A100上的性能分析进一步表明，大幅的内存节省并不一定意味着更低的推理延迟。最后，一个开放式AMBE

    arXiv:2609.29999v1 Announce Type: cross  Abstract: Post-training quantization of vision--language models (VLMs) is typically assessed through aggregate task accuracy and memory savings, but preserving a headline score does not guarantee preservation of visual grounding behavior. We present GHOST-Q, a cross-precision controlled evaluation of three 8B VLM families under FP16, INT8, and NF4 across utility and hallucination-sensitive benchmarks. Rather than comparing only aggregate accuracy, we pair FP16 and quantized predictions item by-item to quantify how compression redistributes grounding successes and failures. Five of six quantized variants preserve MMStar accuracy within $\pm2$ percentage points, yet 10 of 36 paired effects remain significant after false-discovery-rate correction, nine on hallucination-sensitive conditions. Same-device A100 profiling further demonstrates that substantial memory reduction does not necessarily mean lower inference latency. Finally, an open-ended AMBE
    
[^32]: 让训练引导选择：基于真实锚定效用的在线合成数据过滤

    Let Training Guide Selection: Online Synthetic Data Filtering via Real-Anchored Utility

    [https://arxiv.org/abs/2609.29988](https://arxiv.org/abs/2609.29988)

    提出FROST在线合成数据过滤框架，通过锚定真实训练数据的梯度反馈估计合成数据效用，无需外部验证器即可过滤约20-30%的合成数据并提升真实任务性能。

    

    当真实数据有限时，合成数据可以扩展训练监督的规模，但噪声和分布失配会降低其价值。现有的合成数据选择方法通常强调保真度或多样性，而忽视了学习器不断变化的需求。我们提出了FROST，这是一个通过锚定于真实训练数据的梯度反馈来估计合成数据效用的在线框架。该框架根据近期历史记录校准批次效用，以确定何时需要过滤，并且仅在带外批次中过滤样本以决定保留哪些内容，无需外部验证器或留出的验证集。在图像分类和面向text-to-SQL的LLM微调两个公开基准上的实验表明，与在完整合成数据池上训练相比，FROST过滤掉了约20-30%的合成数据，同时提升了真实任务的性能。我们进一步在训练过程中将FROST应用于大规模工业广告重排序场景。

    arXiv:2609.29988v1 Announce Type: new  Abstract: Synthetic data can scale training supervision when real-world data are limited, but noise and distribution mismatch can reduce its value. Existing synthetic data selection methods often emphasize fidelity or diversity rather than the learner's evolving needs. We propose FROST, an online framework that estimates synthetic-data utility through gradient feedback anchored in real training data. It calibrates batch utility against recent history to determine when filtering is needed and filters samples only in out-of-band batches to determine what to retain, without an external verifier or held-out validation set. Experiments on two public benchmarks for image classification and LLM fine-tuning for text-to-SQL show that FROST filters out around 20--30% of the synthetic data while improving real-task performance compared with training on the full synthetic data pool. We further apply FROST during training in a large-scale industrial ads re-ran
    
[^33]: 多样几何、冻结权重：基于因果专家集成的鲁棒异质性处理效应估计

    Diverse Geometries, Frozen Weights: Robust Heterogeneous Treatment-Effect Estimation via Causal Expert Ensembles

    [https://arxiv.org/abs/2609.29974](https://arxiv.org/abs/2609.29974)

    该论文提出GeoACE五专家集成框架，通过结合锚定校正估计器与多样化的重叠感知和结果引导几何，并采用验证集学习且在测试前冻结的集成权重（其中新增的O-Phi-ACE专家用无结果的重叠感知统计投影替代锚定输入），实现了更鲁棒的异质性处理效应估计。

    

    从观测数据中估计异质性处理效应是困难的，因为最合适的归纳偏置会随着重叠程度、处理不平衡、预后结构和样本量的变化而改变。我们提出了几何多样锚定校正专家集成，这是一个五专家框架，它将一个通用的锚定校正估计器与互补的重叠感知几何和结果引导几何相结合。其任务级集成权重仅从内部验证预测中学习，在测试评估前被冻结，然后应用于在完整开发样本上重新拟合的专家。第五个专家O-Phi-ACE从协变量和处理分配中构建一个不依赖结果的重叠感知统计投影，并用这种更低维度的几何来替代锚定输入。我们在八个基准协议上将GeoACE与11个对比方法进行评估。添加O-Phi-ACE使平均sqrt(PEHE)相对于四专家版本有所降低……

    arXiv:2609.29974v1 Announce Type: new  Abstract: Estimating heterogeneous treatment effects from observational data is difficult because the most appropriate inductive bias varies with overlap, treatment imbalance, prognostic structure, and sample size. We introduce the Geometry-Diverse Anchor-Correction Expert Ensemble (GeoACE), a five-expert framework that combines a common anchor-correction estimator with complementary overlap-aware and outcome-guided geometries. Its task-level ensemble weights are learned only from internal validation predictions, frozen before test evaluation, and then applied to experts refitted on the complete development sample. The fifth expert, O-Phi-ACE, constructs an outcome-free, overlap-aware statistical projection from covariates and treatment assignment and replaces the anchor input with this lower-dimensional geometry. We evaluate GeoACE against 11 comparators on eight benchmark protocols. Adding O-Phi-ACE reduced mean sqrt(PEHE) relative to the four-e
    
[^34]: 带自举机制的随机算子收缩框架：在TD学习中的应用

    A Contraction Framework for Stochastic Operators with Bootstrapping: Application to TD Learning

    [https://arxiv.org/abs/2609.29961](https://arxiv.org/abs/2609.29961)

    该论文提出了一个将自举式随机更新统一建模为随机算子的收缩分析框架，无需依赖梯度结构且允许采样误差随迭代点增长，从而为TD学习等算法在任意目标更新周期$K$下提供了几何收敛的有限时间保证。

    

    许多迭代算法都依赖于自举：一个变量使用其另一个冻结的副本作为目标进行更新，而该目标会定期被更新后的变量所替换。最大化-最小化方法和非精确近端点方法都具有这种结构，时序差分（TD）学习也是如此。然而，针对将采样更新与每隔$K$步才刷新一次的目标相结合的场景，现有的收敛性保证依赖于更新的特定结构，例如线性逼近或基于梯度的内部步骤，以及一致有界的采样误差。我们转而将采样更新建模为参数空间上的一个随机算子，从而将分析简化为一个收缩论证，该论证不需要梯度结构，并允许采样误差随迭代点增长。在该框架内，我们针对独立同分布样本和任意目标更新周期$K$导出了有限时间界。我们证明迭代点以几何速率收敛（摘要原文在此处截断）。

    arXiv:2609.29961v1 Announce Type: new  Abstract: Many iterative algorithms rely on bootstrapping. A variable is updated using a second, frozen copy as a target, which is periodically replaced with the updated variable. Majorize-minimize and inexact proximal-point methods share this structure, as does temporal-difference (TD) learning. However, existing convergence guarantees for scenarios that combine sampled updates with targets refreshed only every $K$ steps rely on the specific structure of the update, such as linear approximation or gradient-based inner steps, and on uniformly bounded sampling error. We instead model the sampled update as a stochastic operator on the parameter space, which reduces the analysis to a contraction argument that needs no gradient structure and allows the sampling error to grow with the iterates. Within this framework, we derive a finite-time bound for i.i.d. samples and any target-update period $K$. We show that the iterates converge geometrically in ro
    
[^35]: 超越平均安全性：机会约束的大语言模型微调

    Beyond Average Safety: Chance-Constrained LLM Fine-tuning

    [https://arxiv.org/abs/2609.29960](https://arxiv.org/abs/2609.29960)

    本文提出一种机会约束的保安全微调方法，通过限制安全样本相对参考模型退化超过阈值的比例，并利用可微上界与约束感知梯度下降算法，解决了传统平均安全损失掩盖罕见但严重安全失效的问题。

    

    在大语言模型上针对新目标进行微调可以提升有用性、指令遵循能力或领域特定性能，但也可能在安全关键提示上引发性能回退。现有的保安全微调方法通常控制平均安全损失或使用加权辅助惩罚，这可能会掩盖罕见但严重的失效情况。我们提出了一种用于保安全微调的机会约束公式化方法，该方法限制相对于参考模型退化超过规定阈值的安全样本比例。由于由此产生的经验机会约束包含不连续的指示函数，我们引入了违反率的可微优化上界，从而得到一个易于处理的保守约束。随后，我们开发了一种约束感知的梯度下降方法，将优化后的约束视为参数空间中的安全集，并以最小程度修改微调方向以保持……

    arXiv:2609.29960v1 Announce Type: cross  Abstract: Fine-tuning large language models on new objectives can improve helpfulness, instruction following, or domain-specific performance, but it can also induce regressions on safety-critical prompts. Existing safety-preserving fine-tuning methods typically control average safety loss or use weighted auxiliary penalties, which can obscure rare but severe failures. We propose a chance-constrained formulation for safety-preserving fine-tuning that limits the fraction of safety examples whose degradation relative to a reference model exceeds a prescribed threshold. Because the resulting empirical chance constraint contains a discontinuous indicator, we introduce a differentiable majorization of the violation rate, yielding a tractable conservative constraint. We then develop a constraint-aware gradient descent method that treats the majorized constraint as a safe set in parameter space and minimally modifies the fine-tuning direction to preserv
    
[^36]: 并非所有混淆都同等重要：面向细粒度飞机检测的来源感知不确定性诊断

    Not All Confusion Is Equal: A Source-Aware Uncertainty Diagnosis for Fine-Grained Aircraft Detection

    [https://arxiv.org/abs/2609.29959](https://arxiv.org/abs/2609.29959)

    提出 A²E² 诊断工具，将细粒度飞机检测中的模型混淆沿“偶然性/认知性”与“类内/类间”两个维度分解为可定量测量的 2×2 四类来源，把被动的混淆度量转化为可操作的改进指导。

    

    细粒度目标检测器通常使用混淆矩阵进行评估，混淆矩阵能够显示模型在何处产生混淆，但无法解释混淆的原因，也无法判断混淆是否可以降低。我们认为，混淆可以归因于不同的、可分离的来源，且每个来源都可以定量测量，从而将被动测量转化为可操作的指导。我们提出了 A²E²，这是一种诊断工具，它沿着两个维度分解混淆来源：{偶然性，认知性} × {类内，类间}，形成一个 2×2 的分类体系，枚举出混淆来源的类型。每个象限都由其专属的量进行测量，该量在三个位置之一计算（输入几何、输出空间的不一致性以及偏置参数的后验分布），因此两个认知性来源是通过构造方式而非经验相关性来区分的。在细粒度飞机检测任务中，四个象限成为四个被命名的混淆来源，并各自拥有相应的补救判断。

    arXiv:2609.29959v1 Announce Type: cross  Abstract: Fine-grained object detectors are commonly evaluated with confusion matrices, which show where the model is confused but not why, nor whether the confusion can be reduced. We argue that confusion can be attributed to distinct, separable sources, each quantitatively measurable, turning a passive measurement into actionable guidance. We present $A^2E^2$, a diagnostic tool that decomposes the sources of confusion along two axes, $\{$aleatoric, epistemic$\} \times \{$within-class, between-class$\}$, giving a $2\times2$ taxonomy that enumerates the source types. Each quadrant is measured by its own quantity, computed in one of three places (input geometry, output-space disagreement, and the bias-parameter posterior), so the two epistemic sources are separated by construction rather than by an empirical correlation. On fine-grained aircraft detection, the four quadrants become four named sources with their own remedy verdict: affinity (geome
    
[^37]: 多维匹配

    Multi-Dimensional Matching

    [https://arxiv.org/abs/2609.29958](https://arxiv.org/abs/2609.29958)

    该论文提出一种基于特征的多维匹配机制，通过单次谱投影将匹配问题化简为 O(N log N) 时间可解的一维排序，并证明在投影空间内可取得精确的纳什社会福利最优解，同时具备福利保证与抗噪声稳定性。

    

    我们研究了一种匹配机制，其中智能体（agents）和物品（objects）通过特征而非完整排序来描述。单次谱投影将问题简化为一维排序，可在 O(N log N) 时间内计算完成。我们证明，在去尺度化（descaled）的特征与偏好上，我们的算法在投影空间内可取得精确的纳什社会福利（NSW）最优解，并具有无条件的功利主义福利保证和条件性的 NSW 保证。所提出的机制对外生噪声具有稳定性，但不满足防策略性（strategy-proof）；我们给出了一个明确的可获利的虚假报告示例。在一个智能体（agentic）AI 购物应用中，诊断工具成功预测了一个成功案例和一个失败案例。包含 100 个实例的稳健性研究证实了这些发现。

    arXiv:2609.29958v1 Announce Type: cross  Abstract: We study a matching mechanism where agents and objects are described by features rather than complete rankings. A single spectral projection reduces the problem to a one-dimensional sort, computable in O(N log N) time. We prove that on descaled features and preferences, our algorithm obtains the exact Nash Social Welfare (NSW) optimum within the projected space, with an unconditional utilitarian-welfare guarantee and a conditional NSW guarantee. The proposed mechanism is stable against exogenous noise but not strategy-proof; we provide an explicit profitable misreport. On an agentic AI shopping application, the diagnostics correctly anticipate both a success and a failure case. A 100-instance robustness study confirms the findings.
    
[^38]: 追踪状态还是追踪陪集？学习型状态追踪的代数解释

    Tracking States or Tracking Cosets? An Algebraic Account of Learned State Tracking

    [https://arxiv.org/abs/2609.29951](https://arxiv.org/abs/2609.29951)

    该论文从代数视角揭示Transformer在群运算状态追踪任务中学到的并非精确状态而是商类（陪集）解，并证明最优顺序无关准确率收敛于阿贝尔化类大小的倒数。

    

    状态追踪需要组合一系列更新，但仅凭准确率无法揭示模型究竟学到了什么。我们研究了被训练来预测群元素连续乘积的神经网络。我们在Transformer中识别出了“商解”现象：模型能够恢复商类，同时在其成员之间近乎均匀地进行预测。类大小的倒数无需任何拟合参数即可预测部分准确率，这将基于奇偶性的解释推广到了非奇偶性的商结构。我们的基线Transformer在超过精确追踪边界后，其预测在前缀重排下几乎不变。我们证明，对于在均匀独立同分布全群输入下的有限群，最优的顺序无关精确准确率随前缀长度增长收敛于阿贝尔化类大小的倒数，这与观察到的阿贝尔化平台现象一致。顺序更新还允许更多可能性：任何将群划分为子群（无论是否正规）的右陪集划分，都能在顺序更新下保持（原文摘要在此处被截断）。

    arXiv:2609.29951v1 Announce Type: cross  Abstract: State tracking requires composing a sequence of updates, but accuracy alone does not reveal what a model has learned. We study neural networks trained to predict the running product of group elements. We identify quotient solutions in Transformers, where models recover the quotient class while predicting nearly uniformly among its members. The reciprocal of class size predicts partial accuracy without a fitted parameter, extending parity-based accounts to non-parity quotients. Our baseline Transformers' predictions change little under prefix reordering beyond the exact-tracking frontier. We prove that, for finite groups under uniform i.i.d. full-group inputs, optimal order-blind exact accuracy converges to the reciprocal of abelianization class size as prefix length grows, consistent with the observed abelianization plateaus. Sequential updates permit more: any partition into right cosets of a subgroup, normal or not, survives sequenti
    
[^39]: 皮层-小脑环路中误差与预测驱动的运动学习

    Error- and Prediction-Driven Motor Learning in the Cortico-Cerebellar Loop

    [https://arxiv.org/abs/2609.29945](https://arxiv.org/abs/2609.29945)

    该论文提出一种受小脑启发的控制框架，通过将多路复用预测表示与内部反馈相结合，实现了延迟反馈下的精确在线校正，并将适应学习时间缩短了一个数量级。

    

    在延迟的感觉反馈下实现鲁棒控制仍然是机器人学和神经科学中的一个关键挑战。经典的小脑模型通过前向预测来解释延迟补偿，但无法解释生物系统中观察到的快速在线校正和快速适应现象。我们提出了一种受小脑启发的控制框架，将多路复用的预测表示与内部反馈相结合。通过联合编码运动学变量和任务相关的误差信号，该模型能够在反馈延迟的情况下实现精确的在线校正。此外，在小脑环路中引入反馈显著加速了适应过程，将学习时间缩短了一个数量级。我们的结果表明，在存在延迟的情况下，单信号预测是不够的，而多路复用与反馈相结合为在线控制和快速学习提供了一种统一的机制。

    arXiv:2609.29945v1 Announce Type: new  Abstract: Robust control under delayed sensory feedback remains a key challenge in both robotics and neuroscience. Classical cerebellar models explain delay compensation through forward prediction but fail to account for fast online corrections and rapid adaptation observed in biological systems.   We propose a cerebellum-inspired control framework that combines multiplexed predictive representations with internal feedback. By jointly encoding kinematic variables and task-relevant error signals, the model enables accurate online correction despite delayed feedback. Furthermore, incorporating feedback within the cerebellar loop significantly accelerates adaptation, reducing learning time by an order of magnitude.   Our results show that single-signal predictions are insufficient under delay, while multiplexing and feedback together provide a unified mechanism for online control and rapid learning.
    
[^40]: MF-SCBO：多保真度可扩展约束贝叶斯优化

    MF-SCBO : Multi-fidelity Scalable Constrained Bayesian Optimization

    [https://arxiv.org/abs/2609.29941](https://arxiv.org/abs/2609.29941)

    该论文提出MF-SCBO方法，将可扩展约束贝叶斯优化扩展至多保真度设置，首次同时解决了高维性、黑盒约束、任意数量保真度级别和非嵌套采样这四个难题，并在实验中展现出优于单保真度SCBO及其他多保真度方法的收敛性能。

    

    许多现实世界的优化问题依赖于昂贵的仿真或实验，因此有效利用可用数据至关重要。随着机器学习、工程和控制等应用中目标评估成本的持续上升，对受黑盒约束的高维黑盒函数进行多保真度优化变得越来越重要。据我们所知，目前尚无现有方法能够同时处理高维性、黑盒约束、任意数量的保真度级别以及非嵌套采样。在这项工作中，我们将可扩展约束贝叶斯优化方法扩展到多保真度设置，得到了MF-SCBO方法。我们在标准基准函数以及具有挑战性的问题上对该方法进行了评估。实验结果表明，MF-SCBO通常比单保真度的SCBO和其他多保真度方法取得更好的收敛性能。

    arXiv:2609.29941v1 Announce Type: new  Abstract: Many real-world optimization problems rely on expensive simulations or experiments, making the efficient use of available data essential. Multi-fidelity optimization of high-dimensional black-box functions subject to black-box constraints is increasingly relevant as the cost of objective evaluations continues to rise in applications such as machine learning, engineering, and control. To our knowledge, no existing method simultaneously addresses high-dimensionality, black-box constraints, an arbitrary number of fidelity levels, and non-nested sampling. In this work, we extend the Scalable Constrained Bayesian Optimization method to the multi-fidelity setting, resulting in the MF-SCBO method. The proposed approach is evaluated on standard benchmark functions as well as challenging problems. The experimental results demonstrate that MF-SCBO generally achieves better convergence than both the single-fidelity SCBO and the other multi-fidelity
    
[^41]: 路径特异性伤害分解：一个部分识别框架

    Path-specific harm decomposition: A partial identification framework

    [https://arxiv.org/abs/2609.29938](https://arxiv.org/abs/2609.29938)

    该论文提出了将治疗伤害分解为直接路径和间接（中介）路径贡献的新概念与部分识别框架，为即使在随机对照试验中也无法点识别的直接与间接负向影响比例（FNA）提供了识别边界。

    

    设计治疗政策时的一个核心目标往往是“不造成伤害”，即避免那些能改善平均结果却使某些个体结果恶化的干预措施。一个广泛使用的伤害度量是“负向影响比例”（FNA），定义为干预降低个体结果的概率。然而，在许多应用中，治疗是通过中介变量起作用的，单一的“总”FNA可能会掩盖伤害主要是通过直接路径还是间接（由中介诱导的）路径产生。在这项工作中，我们引入了FNA的路径特异性版本。为此，我们在因果中介分析框架下将总伤害分解为直接伤害和间接伤害。然而，这些量依赖于潜在结果的联合分布，即使在随机对照试验中也无法被点识别。作为解决方案，我们开发了一种新颖的直接FNA和间接FNA的部分识别框架。

    arXiv:2609.29938v1 Announce Type: cross  Abstract: A central goal when designing treatment policies is often to "do no harm", that is, to avoid interventions that improve average outcomes while worsening outcomes for some individuals. A widely used notion for harm is the fraction of negatively affected (FNA), defined as the probability that an intervention decreases an individual's outcome. However, in many applications, treatments operate through mediators, and a single "total" FNA can obscure whether harm arises primarily through direct pathways or indirect (mediator-induced) pathways. In this work, we introduce a path-specific analogue of the FNA. For this, we disentangle total harm into direct and indirect harm in causal mediation settings. However, these quantities depend on joint distributions of potential outcomes that are not point-identified even in randomised controlled trials. As a remedy, we develop a novel partial identification framework for direct and indirect FNA. In ou
    
[^42]: 当时间扰动表现得像传感器偏差时：可穿戴活动识别器的无标签审计

    When Temporal Perturbations Act Like Sensor Biases: Label-Free Auditing of Wearable Activity Recognizers

    [https://arxiv.org/abs/2609.29937](https://arxiv.org/abs/2609.29937)

    提出无标签审计方法SpectrumAudit，揭示可穿戴活动识别模型对“时间扰动”的鲁棒性主要由持续的直流传感器偏移主导，而非真正的时域波形变化。

    

    可穿戴人体活动识别（HAR）模型在传感器、被试和骨干网络上运行，然而一段平滑的波形可能看似时间性扰动，实则主要利用了持续的传感器偏移。我们提出SpectrumAudit，一种标签密封的审计方法，它在从训练和测试中排除的被试的校准窗口上拟合相位随机化的全窗口刺激。选定后，它将精确的直流投影和预算约束的零均值残差重放到同一个冻结的目标模型上，无需重新拟合。在来自三个数据集和三种骨干网络的27个目标模型上，所选波形造成2.87至40.83个百分点的三阶段鲁棒准确率损失。在此重放预算下，直流分量在24/27个目标模型上比交流分量更具破坏性，并在22/27个目标模型上恢复了至少90%的总降幅；所有5次失败均发生在WISDM数据集上。在留出的UTD-MHAD验证中，所选波形造成13.49个百分点的准确率损失和11.68个百分点的宏F1损失，而匹配的随机变化仅为-0.66个百分点。

    arXiv:2609.29937v1 Announce Type: cross  Abstract: Wearable human-activity recognition (HAR) models operate across sensors, subjects, and backbones, yet a smooth waveform may appear temporal while exploiting a persistent sensor offset primarily. We introduce SpectrumAudit, a label-sealed audit that fits a phase-randomized full-window stimulus on calibration windows from subjects held out from training and testing. After selection, it replays its exact DC projection and budget-constrained zero-mean residual on the same frozen victim without refitting. Across 27 victims from three datasets and three backbones, the selected waveforms cause 2.87-40.83-point three-phase robust accuracy losses. Under this replay budget, DC is more damaging than AC on 24/27 victims and recovers at least 90% of the full drop on 22/27; all 5 failures occur on WISDM. In a held-out UTD-MHAD check, the selected waveform causes 13.49-pp accuracy and 11.68-pp macro-F1 losses, versus -0.66 pp for matched random chang
    
[^43]: 污染环境下的LLM生成文本鲁棒检测

    Robust Detection of LLM-Generated Text under Contamination

    [https://arxiv.org/abs/2609.29935](https://arxiv.org/abs/2609.29935)

    该论文将人类与机器文本建模为带Huber污染的有限阶马尔可夫过程，刻画了LLM生成文本可靠检测的精确理论边界，并证明对似然比检验等统计检测器进行截断处理可在污染环境下实现鲁棒检测。

    

    我们研究在编辑和污染条件下对大语言模型（LLM）生成文本的检测。我们将人类和机器文本建模为带有Huber污染的有限阶马尔可夫过程，并在本文假设下刻画了可靠检测的精确边界。当污染程度相对于干净数据源的分离度足够大时，检测是不可能的。在该边界以下，一组截断似然比检验能够实现趋于零的最坏情况错误率。这一构造启发了将截断作为现有统计检测器的一种简单修改方法。对于一大类加性得分函数，我们识别出截断检验保持一致性、而原始检验的最坏情况功效趋于零的条件。我们在三个数据集和三个生成模型上评估了七种检测器，并在RAID基准上进行测试。截断在两项研究中均提升了鲁棒性，其增益因检测器和污染设置而异。例如，在目标……（摘要原文在此处截断）

    arXiv:2609.29935v1 Announce Type: cross  Abstract: We study the detection of LLM-generated text under editing and contamination. Modeling human and machine text as finite-order Markov processes with Huber contamination, we characterize an exact boundary for reliable detection under our assumptions. Detection is impossible when contamination is sufficiently large relative to clean-source separation. Below this boundary, a collection of clipped likelihood-ratio tests achieves vanishing worst-case errors. This construction motivates clipping as a simple modification of existing statistical detectors. For a broad class of additive scores, we identify conditions under which the clipped test is consistent while the raw test's worst-case power tends to zero. We evaluate seven detectors across three datasets and three generation models, and on the RAID benchmark. Clipping improves robustness in both studies, with gains varying across detectors and contamination settings. For example, at a targ
    
[^44]: 使用测试时增强改进黑盒放射学AI的校准

    Improving Calibration of Black-Box Radiology AI Using Test-Time Augmentation

    [https://arxiv.org/abs/2609.29931](https://arxiv.org/abs/2609.29931)

    提出了一种名为DualTTA的模型无关框架，利用基于临床的测试时增强技术，在无需访问模型内部或训练数据的情况下，有效改进了黑盒放射学AI模型的校准性能。

    

    放射学AI系统日益影响着分诊、随访影像和治疗计划等临床决策。为了安全地做出这些决策，模型输出必须经过良好校准，即预测概率能够准确反映真实风险。许多标准的校准改进技术，如MC Dropout和深度集成，需要访问模型参数或重新训练。然而，专有临床AI系统以黑盒形式运行，阻止了对模型内部的访问。为此，我们提出了一个模型无关的框架，利用基于临床的测试时增强（TTA）来改进黑盒模型的校准。我们的框架应用几何和物理启发的3D CT扰动，并在无需访问模型内部或原始训练数据的情况下学习概率级聚合策略。在肺栓塞和颅内出血检测任务中，DualTTA取得了优异的（校准性能）……

    arXiv:2609.29931v1 Announce Type: new  Abstract: Radiology AI systems increasingly inform clinical decisions such as triage, follow-up imaging, and treatment planning. For these decisions to be made safely, model outputs must be well calibrated, meaning predicted probabilities accurately reflect true risk. Many standard techniques for improving calibration, such as MC Dropout and Deep Ensembles, require access to model parameters or retraining. However, proprietary clinical AI systems operate as black boxes, preventing access to the model's internals. To that end, we propose a model-agnostic framework for improving calibration of black-box models using clinically grounded test-time augmentation (TTA). Our framework applies geometric and physics-inspired 3D CT perturbations and learns probability-level aggregation strategies without access to model internals or the original training data. Across pulmonary embolism and intracranial hemorrhage detection tasks, DualTTA achieved the stronge
    
[^45]: 基于图上时空互补特征传播的多年期年均日交通量（AADT）估计

    Spatio-temporally complementary feature propagation on graphs for longitudinal AADT estimation

    [https://arxiv.org/abs/2609.29906](https://arxiv.org/abs/2609.29906)

    该研究提出一种时空互补特征传播框架，通过有向图上的泊松能量最小化和流量比矩阵，融合环形检测器数据与宏观交通模型两种互补数据源，实现高效准确的多年期城市路网AADT估计。

    

    年均日交通量（AADT）的估计对于交通规划和基础设施维护至关重要，然而由于物理传感器的高成本和空间稀疏性，获得整个城市网络多年期的准确数值仍然具有挑战性。本研究提出了一种新颖的时空互补特征传播框架，该框架充分利用了两种不同数据源的优势：空间稀疏但时间密集的环形检测器数据，以及空间完整但时间稀疏的宏观交通模型。该方法的核心是一种有向图上的特征传播算法，该算法被表述为考虑残差的泊松能量最小化问题。此外，用流量比矩阵取代标准的二值邻接矩阵，以捕捉交叉口处真实的车辆转弯比例。该算法在苏黎世市得到了验证，展现出高计算效率，并取得了……

    arXiv:2609.29906v1 Announce Type: new  Abstract: The estimation of Annual Average Daily Traffic (AADT) is vital for transportation planning and infrastructure maintenance, yet obtaining accurate values for an entire urban network across multiple years remains challenging due to the high cost and spatial sparsity of physical sensors. This research proposes a novel spatio-temporally complementary feature propagation framework that leverages the strengths of two distinct data sources: spatially sparse but temporally dense loop detector data, and a spatially complete but temporally sparse macroscopic transportation model. The methodology highlights a feature propagation algorithm on directed graphs, formulated as a Poisson energy minimization considering residues. The standard binary adjacency matrix is replaced with flow ratio matrices to capture real-world vehicle turn ratios at intersections. Validated in the city of Zurich, the algorithm demonstrates high computational efficiency, achi
    
[^46]: 面向投资组合管理的代价敏感在线窗口大小选择

    Cost-Sensitive Online Window Size Selection for Portfolio Management

    [https://arxiv.org/abs/2609.29887](https://arxiv.org/abs/2609.29887)

    本文提出一个两级在线学习框架，将候选窗口大小视为专家并以包含换手成本的损失动态聚合，推导了考虑换手成本的有限时间代价敏感跟踪遗憾界，并证明适当调参的Fixed Share算法在次线性切换预算下可实现渐近无跟踪遗憾。

    

    本文研究了市场环境变化下面向投资组合管理的代价敏感在线窗口大小选择问题。具体而言，我们提出了一个两级框架，该框架使用候选窗口大小构建投资组合，并通过在线学习对其进行动态聚合。通过将候选窗口大小视为“专家”，我们利用包含换手成本的损失函数动态更新它们的聚合权重。此外，我们推导了有限时间范围内的代价敏感跟踪遗憾界，该界考虑了聚合投资组合的换手成本，其中静态遗憾是其特例。在有界损失和成本率的条件下，经过适当调整的Fixed Share算法在次线性切换预算下可实现渐近无跟踪遗憾，而Hedge算法则涵盖静态情形。

    arXiv:2609.29887v1 Announce Type: cross  Abstract: This paper investigates cost-sensitive online window size selection for portfolio management under changing market conditions. Specifically, we propose a two-level framework that constructs portfolios using candidate window sizes and dynamically aggregates them through online learning. By treating candidate window sizes as ``experts,'' we dynamically update their aggregation weights using turnover-inclusive losses. Moreover, we derive finite-horizon cost-sensitive tracking-regret bounds that account for turnover of the aggregated portfolio, with static regret as a special case. Under bounded losses and cost rates, suitably tuned Fixed Share achieves asymptotically no tracking regret for sublinear switching budgets, with Hedge covering the static case.
    
[^47]: 一种新的希尔排序增量序列：超越 $N^{4/3}$ 的强化学习驱动算法发现

    A New Gap Sequence for Shellsort: RL-Driven Algorithm Discovery Beyond $N^{4/3}$

    [https://arxiv.org/abs/2609.29881](https://arxiv.org/abs/2609.29881)

    本文提出一种强化学习驱动的自监督搜索系统，通过在可执行增量生成器空间中搜索，首次发现了最坏情况复杂度超越数十年来 $N^{4/3}$ 上界的实用希尔排序新增量序列。

    

    选择希尔排序的增量序列是一个著名的开放性问题。六十多年来，成功的序列一直依赖于人工设计的公式、数值搜索或数论构造。尽管针对稠密型或主要具有理论意义的序列家族存在更强的通用界，但对于短小、稀疏且具有实际竞争力的构造而言，其最坏情况上界数十年来始终未能突破 $N^{4/3}$。我们提出这样一个问题：增量序列本身能否从执行过程中学习得到？我们提出了一个由强化学习驱动的自监督系统，在可执行的增量生成器空间中进行搜索。每个候选方案在构造上都是有效的，执行后返回精确的比较次数和移动次数；没有任何经典序列被用作学习目标。在五次独立搜索中，该系统发现了一个共同的有理-几何序列家族。第二个自监督阶段仅对一个有限前缀进行调优，得到了实用序列 $1,3,8,20,47,116,300,585,1416,3303,\ldots$

    arXiv:2609.29881v1 Announce Type: cross  Abstract: Choosing Shellsort gaps is a well-known open problem. For over sixty years, successful sequences have relied on human-designed formulas, numerical searches, or number-theoretic constructions. Although stronger general bounds exist for dense or mainly theoretical families, the worst-case upper bound for a short, sparse, and practically competitive construction has not advanced beyond $N^{4/3}$ for decades. We ask whether the sequence itself can instead be learned from execution. We present an RL-driven, self-supervised system that searches over executable gap generators. Every proposal is valid by construction, and executed candidates return exact comparison and move counts; no classical sequence is used as a target. Across five independent searches, the system discovers a common rational-geometric family. A second self-supervised stage tunes only a finite prefix, producing the practical sequence $1,3,8,20,47,116,300,585,1416,3303,\ldot
    
[^48]: 从图到馈线：用于规则合规馈线生成的约束引导扩散模型

    From Graphs to Feeders: Constraint-Guided Diffusion for Rule-Compliant Feeder Generation

    [https://arxiv.org/abs/2609.29879](https://arxiv.org/abs/2609.29879)

    提出了电网约束的离散去噪扩散模型 PG-DiGress，通过在反向扩散过程中以软掩码抑制不兼容的边类别并结合最终投影步骤，生成同时满足电气兼容性与辐射状规则的合规配电馈线拓扑。

    

    生成式建模方法通常侧重于从训练数据中恢复宏观的统计特性。在图生成领域，这可能指度分布、聚类系数或谱特性等。然而，当缺少详细的馈线模型时，要生成可用的配电馈线，仅匹配通用的图统计特性是不够的：采样得到的拓扑结构还必须满足电气兼容性与辐射状规则。为此，我们将馈线合成问题形式化为一个约束引导的图生成问题，并提出了电网约束的离散去噪扩散模型 PG-DiGress（Power-Grid-constrained Discrete Denoising Diffusion model），该模型在从馈线数据中学习类别型的节点与边模式的同时，遵守领域特定规则。具体而言，该方法通过软掩码将馈线约束注入到反向扩散过程中，在去噪过程中抑制不兼容的边类别，随后再通过一个最终的投影步骤进行重建。

    arXiv:2609.29879v1 Announce Type: new  Abstract: Generative modeling approaches often focus on recovering broad statistical characteristics from the training data. In the context of graph generation, this may refer to degree distributions, clustering coefficients, or spectral properties. However, generating usable distribution feeders when detailed feeder models are unavailable requires more than matching generic graph statistics: the sampled topology must also obey electrical compatibility and radiality rules. We therefore formulate feeder synthesis as a constraint-guided graph generation problem and propose the Power-Grid-constrained Discrete Denoising Diffusion model, PG-DiGress, which learns categorical node and edge patterns from feeder data, while respecting domain-specific rules. Specifically, it injects feeder constraints into the reverse diffusion process through soft masks that suppress incompatible edge classes during denoising, followed by a final projection step that rebui
    
[^49]: 逐帧早退是否划算？面向设备端语音增强的动态深度算力匹配研究

    Does per-frame early exit pay? A compute-matched study of dynamic depth for on-device speech enhancement

    [https://arxiv.org/abs/2609.29867](https://arxiv.org/abs/2609.29867)

    本文通过对单一因果模型的每个中间深度进行监督训练并微调输出头，为设备端语音增强派生出一族帕累托效率更高的静态模型，在同等算力下PESQ最高提升0.11，或在节省30%算力的情况下匹配最佳PESQ，并在STM32N6微控制器上验证了int8量化后的延迟-质量表现。

    

    基于深度学习的语音增强正越来越多地部署在助听器、头戴式耳机和入耳式耳机等设备上。然而，大多数此类设备只能加速静态的int8计算图，因此深度可变的网络必须实现为多个计算图，并由某种策略进行调度。在本文中，我们对单一因果模型的每个中间深度进行监督训练，然后微调其输出头，以保证更深层的输出永远不会比浅层的差。利用这种训练协议，我们可以派生出一族静态模型，这些模型相比在相同训练预算下从头训练的同等规模模型具有更优的帕累托效率。具体而言，在相同算力下，我们的PESQ最高提升0.11，并以减少30%的算力匹配了最佳PESQ。随后，我们将模型量化为int8，并在STM32N6微控制器上测量延迟-质量前沿。在VoiceBank-DEMAND数据集上，动态增强器与静态模型处于同一前沿（摘要至此截断）。

    arXiv:2609.29867v1 Announce Type: cross  Abstract: Deep learning-based speech enhancement is increasingly deployed on-device in hearing aids, headsets, and earbuds. Most of these devices, however, can only accelerate static int8 graphs, so a depth-varying network must be implemented as several graphs, orchestrated by a policy. In this paper, we supervise every intermediate depth of one causal model, then we fine-tune its output heads to guarantee that deeper outputs are never worse than shallower ones. Using this training protocol, we can derive a family of static models that are more Pareto-efficient than their equivalently-sized counterparts trained from scratch on the same budget. Specifically, we achieve up to 0.11 higher PESQ for equivalent compute, and match the best PESQ at 30% less compute. We then quantize the models to int8 and measure the latency-quality frontier on an STM32N6 microcontroller. On VoiceBank-DEMAND, the dynamic enhancer lies on the same frontier as the static 
    
[^50]: 超越模型规模：面向嵌入式语音增强的LiSenNet重新设计

    Beyond Model Size: Redesigning LiSenNet for embedded speech enhancement

    [https://arxiv.org/abs/2609.29866](https://arxiv.org/abs/2609.29866)

    本研究通过将LiSenNet的循环瓶颈替换为卷积频率与时间混合器、将不支持的操作改造为静态int8兼容原语并使用有界解码器激活，使37k参数的语音增强模型能够在STM32微控制器NPU上满足量化部署要求并保持增强质量。

    

    在资源受限设备上部署实时语音增强需要满足严格的延迟、内存和能耗约束。微控制器NPU（神经网络处理器）可以在这些约束下加速神经推理，但仅支持静态、整数量化图中的受限算子集合。近期的语音增强网络已将参数量和乘累加运算（MACs）降低到名义上适合微控制器的水平，但其算子和执行模式往往仍与受限的NPU不兼容。我们通过针对STM32N6570-DK Neural-ART加速器重新设计LiSenNet——一个37k参数的子带双路径模型——来解决这一差距。我们用卷积频率和时间混合器替换其循环瓶颈结构，将不支持的操作重新表述为静态int8兼容的原语，并使用有界解码器激活以在量化后保持质量。在VoiceBank-DEMAND数据集上，最终的NPU兼容模型达到或超过……

    arXiv:2609.29866v1 Announce Type: cross  Abstract: Deploying real-time speech enhancement on resource-constrained devices requires meeting strict latency, memory, and energy constraints. Microcontroller NPUs can accelerate neural inference under these constraints, but only through a restricted set of operators in static, integer-quantized graphs. Recent speech-enhancement networks have reduced parameter counts and MACs to levels nominally suitable for microcontrollers, but their operators and execution patterns often remain incompatible with restricted NPUs. We address this gap by redesigning LiSenNet, a 37k parameter sub-band dual-path model, for the STM32N6570-DK Neural-ART accelerator. We replace its recurrent bottleneck with convolutional frequency and temporal mixers, reformulate unsupported operations as static int8-compatible primitives, and use bounded decoder activations to preserve quality after quantization. On VoiceBank-DEMAND, the final NPU-compatible model matches or exce
    
[^51]: 有限目标分辨率监督下的高效连续数字高程模型（DEM）重建

    Efficient Continuous DEM Reconstruction under Limited Target-Resolution Supervision

    [https://arxiv.org/abs/2609.29864](https://arxiv.org/abs/2609.29864)

    SCOPE 通过在低分辨率网格上预测潜在系数场并复用局部傅里叶残差函数，将高维系数预测与输出网格构建解耦，从而在仅有粗分辨率监督数据的条件下实现了高效的连续高分辨率DEM重建。

    

    高分辨率数字高程模型（DEM）支持地球观测应用，但成对的训练参考数据通常只能在较粗的输出分辨率下获得。因此，重建更精细的地形网格既需要在监督尺度之外实现有效迁移，也需要控制密集查询的计算量。为解决这一问题，SCOPE 从较粗分辨率的成对数据中学习连续地形表示。它在低分辨率网格上预测潜在系数场，并通过基函数评估和几何引导的集成融合来复用局部傅里叶残差函数。这一设计将高维系数预测与输出网格构建分离开来。在地理分布的陆海样本上进行的实验评估了监督重建、未见尺度推断、跨域泛化和理论计算等方面。在主要的监督尺度评估中，SCOPE 在六项指标上均领先于所比较的方法。

    arXiv:2609.29864v1 Announce Type: cross  Abstract: High-resolution digital elevation models (DEMs) support Earth observation applications, but paired training references are often available only at coarser output resolutions. Reconstructing finer terrain grids therefore requires both effective transfer beyond the supervised scale and control of dense-query computation. To address this problem, SCOPE learns a continuous terrain representation from coarser-resolution pairs. It predicts a latent coefficient field on the low-resolution grid and reuses local Fourier residual functions through basis evaluation and geometry-guided ensemble fusion. This separates high-dimensional coefficient prediction from output-grid construction. Experiments on geographically distributed land--ocean samples assess supervised reconstruction, unseen-scale inference, cross-domain generalization, and theoretical computation. SCOPE leads the compared methods across six metrics in the main supervised-scale evalua
    
[^52]: 阐释Brinkman惩罚方法的共形结构：面向哈密顿偏微分方程的几何自适应、结构保持算子学习

    Elucidating the Conformal Structure of the Brinkman Penalisation Method for Geometry-Adapted, Structure-Preserving Operator Learning of Hamiltonian PDEs

    [https://arxiv.org/abs/2609.29847](https://arxiv.org/abs/2609.29847)

    本文揭示了Brinkman惩罚正则化在辛矩阵与惩罚投影满足相容性条件时保持多共形辛结构，由此获得精确的局部守恒律，并据此提出了满足离散共形守恒律的结构保持数值积分器和面向哈密顿偏微分方程的几何自适应、结构保持算子学习方法。

    

    Brinkman惩罚方法通过将固体区域建模为强耗散介质，将复杂域上的边值问题嵌入到一个简单的计算盒中，从而避免了贴体网格的生成。我们证明，经Brinkman型惩罚正则化的多辛哈密顿偏微分方程，在连接辛矩阵与惩罚投影的相容性条件下，能够保持一个多共形辛结构。这产生了一个精确的局部守恒律：在该守恒律下，多辛二形式在流体区域内被保持，并在固体内部呈指数衰减。带有Brinkman摩擦的线性波动方程以及带有人工欧姆电导率的麦克斯韦方程组均满足该条件，并具有显式的修正哈密顿密度。在此基础上，我们提出：(i) 通过Strang分裂构造满足离散共形守恒律的结构保持数值积分器，以及 (ii) confo...（原文摘要在此处被截断）

    arXiv:2609.29847v1 Announce Type: cross  Abstract: The Brinkman penalisation method embeds boundary-value problems on complex domains into a simple computational box by modeling the solid region as a strongly dissipative medium, avoiding body-fitted mesh generation. We show that multi-symplectic Hamiltonian PDEs regularised by Brinkman-type penalisation retain a multi-conformal symplectic structure under a compatibility condition linking the symplectic matrix and the penalisation projection. This yields an exact local conservation law, under which the multi-symplectic two-form is conserved in the fluid region and decays exponentially inside the solid. The linear wave equation with Brinkman friction and Maxwell's equations with artificial Ohmic conductivity satisfy this condition, with explicit modified Hamiltonian densities. Building on this, we propose (i) structure-preserving numerical integrators via Strang splitting that satisfy a discrete conformal conservation law, and (ii) confo
    
[^53]: SwitchPFN：面向冻结上下文时间序列分类的共享切换动力学

    SwitchPFN: Shared Switching Dynamics for Frozen In-Context Time Series Classification

    [https://arxiv.org/abs/2609.29814](https://arxiv.org/abs/2609.29814)

    SwitchPFN通过从训练序列中学习共享投影和状态码本，使局部动态算子与转移特征在不同样本间可直接比较，从而在冻结的表格基础模型上实现了时间序列分类的最优平均准确率。

    

    表格基础模型（TFMs）为时间序列分类提供了一条有前景的路径，但其有效性取决于如何将序列数据转换为表格表示。现有的表示方法面临两个挑战：全局聚合可能丢失时间演化的顺序信息，而在独立拟合的坐标系中计算的特征在不同序列之间可能不具备一致的含义。因此，我们将TFMs的表示设计本身视为一个独立的研究问题：该表示应当在保持样本间共享特征定义的同时，保留局部的时序转移信息。我们提出了SwitchPFN，它从训练序列中学习一个共享的投影和状态码本，使局部动态算子和转移特征在不同样本之间可以直接比较。在所评估的基准测试中，SwitchPFN在所有被评估的方法中取得了最高的平均准确率，超越了最强的基线方法……（原文摘要在此处截断）

    arXiv:2609.29814v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) provide a promising route to time-series classification, but their effectiveness depends on how sequential data are converted into tabular representations. Existing representations face two challenges: global aggregation can lose the order of temporal evolution, while features computed in independently fitted coordinate systems may not have consistent meanings across sequences. We therefore view representation design for TFMs as a problem in its own right: the representation should preserve local temporal transitions while maintaining a shared feature definition across samples. We propose SwitchPFN, which learns a shared projection and regime codebook from the training sequences, making local dynamic operators and transition features directly comparable across samples. Across the evaluated benchmarks, SwitchPFN achieves the highest mean accuracy among the evaluated methods, improving over the strongest ba
    
[^54]: FlashLoop：基于惰性更新的快速且内存高效的循环Transformer

    FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates

    [https://arxiv.org/abs/2609.29812](https://arxiv.org/abs/2609.29812)

    FlashLoop发现循环Transformer中循环引入的额外计算与KV缓存存储大部分是冗余的，并利用状态变化集中于少数token、注意力差异由稀疏稳定的关键列主导等特性，通过惰性更新策略实现了快速且内存高效的循环Transformer推理。

    

    循环Transformer作为一种参数高效的方法，通过重复应用共享的Transformer模块来增加计算深度，受到了广泛关注。然而，其相对于传统Transformer的实际优势仍存在争议：每增加一次循环都需要额外执行一次Transformer前向传播，并需要缓存另一组KV状态，导致推理FLOPs和KV缓存内存随循环深度持续增长。这种开销在大循环次数和长上下文场景下尤为严重，使得循环Transformer的参数效率无法转化为实际的推理效率。在本文中，我们发现循环所带来的大量额外计算和存储是冗余的。随着递归的进行，状态变化越来越集中在少数token子集上；注意力输出的差异主要由稀疏且稳定的关键列子集主导；并且

    arXiv:2609.29812v1 Announce Type: new  Abstract: Looped Transformers have attracted substantial attention as a parameter-efficient approach to increasing computational depth through repeated application of shared Transformer blocks. However, their practical advantages over conventional Transformers remain under debate: each additional loop incurs another Transformer pass and requires caching another set of KV states, causing inference FLOPs and KV-cache memory to grow continuously with loop depth. This overhead becomes particularly severe at large loop counts and long context, preventing the parameter efficiency of Looped Transformers from translating into practical inference efficiency. In this paper, we find that much of the additional computation and storage introduced by looping is redundant. As recurrence proceeds, state changes become increasingly concentrated on a small subset of tokens; attention-output differences are dominated by a sparse and stable subset of key columns; and
    
[^55]: CORDIAL：从少量标签校准LLM的有序输出

    CORDIAL: Calibrating Ordinal LLM Outputs from Few Labels

    [https://arxiv.org/abs/2609.29807](https://arxiv.org/abs/2609.29807)

    CORDIAL提出了一种仅用五个可解释参数、仅需少量标签（如20个）即可校准LLM有序输出分布的方法，在绝大多数实验设置中取得最低对数损失，同时支持跨任务先验学习和多LLM融合。

    

    大型语言模型（LLM）可以将文本转换为有序量表上的概率分布，但该分布是一种含噪声的测量：可能出现饱和、压缩或夸大，并偏向某个一致的方向。我们提出了CORDIAL方法，它将模型的输出视为对真实标签的含噪读数，并使用一个包含五个可解释参数的信道对其进行校正。该信道足够小巧，因此其后验分布可以仅凭少量标签进行估计，并且我们证明了由此得到的校准方法能保持一阶随机序关系。在Amazon评论和CMU-MOSEI转录文本上使用四个LLM进行的实验中，在5到100个标签的80个设置中的76个里，CORDIAL在九种校准器中取得了最低的对数损失；使用20个标签和主要的7B读取模型时，它就能匹敌使用28-54个标签的最强基线方法。同样的后验分布还使我们能够从其他任务中学习先验分布并融合多个LLM。像Dirichlet校准这类无限制的校准器，只有在标签数量充足时才能超过它。

    arXiv:2609.29807v1 Announce Type: new  Abstract: A large language model (LLM) can turn a text into a distribution over an ordered scale, but that distribution is a noisy measurement: saturated, compressed or exaggerated, and biased in a consistent direction. We propose CORDIAL, which treats the model's output as a noisy reading of the true label and corrects it with a channel of five interpretable parameters. The channel is small enough for its posterior to be averaged from a handful of labels, and we prove that the resulting calibration preserves first-order stochastic order. On Amazon reviews and CMU-MOSEI transcripts with four LLMs, CORDIAL has the lowest log loss among nine calibrators in 76 of 80 settings with 5 to 100 labels; with 20 labels and the main 7B reader, it matches the strongest baseline using 28-54 labels. The same posterior lets us learn priors from other tasks and fuse several LLMs. Unrestricted calibrators such as Dirichlet calibration overtake it only as the calibr
    
[^56]: 辅助学习的解析理论

    An Analytical Theory of Auxiliary Learning

    [https://arxiv.org/abs/2609.29774](https://arxiv.org/abs/2609.29774)

    该论文提出了辅助学习的首个解析理论，通过师生框架推导出描述在线随机梯度下降动力学的封闭微分方程组，量化了任务相关性和标签噪声对辅助学习收益的影响，并建立了多任务误差与单任务误差之间的普遍关系。

    

    辅助学习是一种优化范式，通过让神经网络在额外的任务上联合训练来提升其在目标任务上的性能。然而，这种改进背后的机制仍然知之甚少。我们使用师生框架研究这一问题，并推导出一组封闭的微分方程组，用于描述大输入极限下在线随机梯度下降的动力学。对于线性网络，我们得到了在学习率主导阶意义下泛化误差的封闭形式表达式，量化了任务相关性和标签噪声如何决定辅助学习的收益。对于非线性激活函数，我们发展了一种涨落-耗散解析理论，建立了将主任务误差和辅助任务误差与相应单任务误差联系起来的普遍关系。数值实验支持了理论预测，并展示了辅助任务如何提升网络性能。

    arXiv:2609.29774v1 Announce Type: new  Abstract: Auxiliary learning is an optimization paradigm in which a neural network's performance on a target task is improved by jointly training it on additional tasks. However, the mechanisms behind this improvement remain poorly understood. We study this problem using a teacher-student framework and derive a closed system of differential equations describing the dynamics of online stochastic gradient descent in the large-input limit. For linear networks, we obtain a closed-form expression for the generalization error to leading order in the learning rate, quantifying how task correlations and label noise determine the benefit of auxiliary learning. For non-linear activation functions, we develop a fluctuation-dissipation analytical theory that establishes a general relation linking the main and auxiliary errors to the corresponding single-task error. Numerical experiments support the theoretical predictions and show how auxiliary tasks improve 
    
[^57]: WeatherDiagFlow：基于诊断流精化的证据支撑雷达临近预报

    WeatherDiagFlow: Evidence-Grounded Radar Nowcasting with Diagnostic Flow Refinement

    [https://arxiv.org/abs/2609.29772](https://arxiv.org/abs/2609.29772)

    提出 WeatherDiagFlow，将雷达临近预报重构为“预报—公报—审计”任务：以回波移动、生消、强回波风险和不确定性等结构化诊断证据条件化滚动流精化，并借助多智能体层生成业务公报与独立的事后验证审计。

    

    雷达临近预报对短期预警和应急响应至关重要，但传统系统主要仅返回未来雷达回波场，对业务沟通与事后核查的支持有限。我们将雷达临近预报建模为一个证据支撑的“预报—公报—审计”任务：数值预报器同时生成未来雷达场与结构化诊断证据；预报时刻的公报仅使用模型可获得的证据，而事后审计仅在预报时效结束后才引入未来雷达真值。基于该任务框架，WeatherDiagFlow 预测回波移动、生消演变、强回波风险和不确定性，以此条件化滚动流精化过程，同时采用冻结骨架残差校准来改善长时效强回波的保留。多智能体层将结构化证据转化为业务公报，并独立生成验证审计，而不（原文摘要在此处截断）。

    arXiv:2609.29772v1 Announce Type: new  Abstract: Radar nowcasting is essential for short-term warning and emergency response, yet conventional systems mainly return future radar fields and provide limited support for operational communication and post-event verification. We formulate radar nowcasting as an evidence-grounded forecast--bulletin--audit task, in which a numerical forecaster produces both future radar fields and structured diagnostic evidence. Forecast-time bulletins use only model-available evidence, whereas post-event audits incorporate future radar truth only after the forecast horizon is observed. Based on this task formulation, WeatherDiagFlow predicts motion, growth and decay, heavy-echo risk, and uncertainty to condition rolling flow refinement, while frozen-scaffold residual calibration improves long-lead strong-echo preservation. A multi-agent layer converts the structured evidence into operational bulletins and independently generates verification audits without f
    
[^58]: 《论生长、形态与功能：可复用的调控把手控制表型变异》

    On Growth and Form, and Function: Reusable Regulatory Handles Control Phenotypic Variation

    [https://arxiv.org/abs/2609.29755](https://arxiv.org/abs/2609.29755)

    该研究将低秩适配（LoRA）应用于预训练的神经细胞自动机，证明形态的大尺度连贯变化可以由固定调控网络上低维、可复用且可组合的“调控把手”来参数化控制。

    

    表型转变如何通过底层调控动力学的变化来实现，仍然是发育生物学中的一个核心问题。受达西·汤普森（D'Arcy Thompson）1917年著作《论生长与形态》的启发，我们探究形态的连贯大规模转变是否能够被编码为自组织发育系统的低维调制。我们使用神经细胞自动机（NCAs）作为分布式发育的生物启发模型，其中共享的局部调控网络从单个细胞生长出目标形态。我们将低秩适配（LoRA）应用于预训练的NCAs，把每个适配后的发育程序表示为对固定调控支架的低秩调制。完全生长的二维emoji表型的水平与垂直缩放各自都可以通过秩一适配来实现；它们的线性组合能够参数化地控制表型大小，可泛化到训练分布之外，并能与目标……（原文摘要在此处截断）

    arXiv:2609.29755v1 Announce Type: cross  Abstract: How phenotypic transformations are implemented by changes in underlying regulatory dynamics remains a central question in developmental biology. Inspired by D'Arcy Thompson's 1917 "On Growth and Form", we ask whether coherent large-scale transformations of morphology can be encoded as low-dimensional modulations of a self-organizing developmental system. We use neural cellular automata (NCAs) as bio-inspired models of distributed development, in which a shared local regulatory network grows target morphologies from a single cell. We apply low-rank adaptation (LoRA) to pretrained NCAs, representing each adapted developmental program as a low-rank modulation of a fixed regulatory scaffold. Horizontal and vertical scaling of a fully grown 2D emoji phenotype can each be implemented by rank-one adaptations. Their linear combinations parametrically control phenotype size, generalize beyond the training distribution, and compose with target-s
    
[^59]: TopU-LBVS：一个面向基于配体的虚拟筛选的现实多靶点基准测试

    TopU-LBVS: A Realistic Multi Target Benchmark for Ligand Based Virtual Screening

    [https://arxiv.org/abs/2609.29740](https://arxiv.org/abs/2609.29740)

    提出了TopU-LBVS，一个覆盖7大类93个蛋白靶点、采用性质匹配且结构相似诱饵并配有三种固定评估协议的多靶点基于配体虚拟筛选基准，解决了现有基准因简单诱饵和随机阴性而高估性能的问题。

    

    基于配体的虚拟筛选（LBVS）是早期药物发现中一种实用的一线筛选工具，但现有基准测试可能因随机阴性样本、容易的诱饵分子、靶点覆盖有限以及不规范的评估协议而高估性能。我们提出了TopU-LBVS，一个在困难阴性筛选条件下的多靶点LBVS基准测试。基于经整理的ChEMBL 35生物活性数据，TopU-LBVS覆盖7个蛋白质类别中的93个蛋白质靶点，并以固定的1:40活性化合物与诱饵比例，构建了具有性质匹配、结构相似诱饵的靶点特异性筛选文库。每个文库包含约400到10,000个化合物，其设计旨在减少基于简单理化性质和最近邻指纹的捷径学习。TopU-LBVS提供了三种固定评估协议。TopU-LBVS-full评估ChEMBL* → TopU在全部93个靶点上的泛化能力。TopU-LBVS-low评估低数据量条件下TopU → TopU的性能表现。

    arXiv:2609.29740v1 Announce Type: cross  Abstract: Ligand-based virtual screening (LBVS) is a practical first-pass tool in early-stage drug discovery, but existing benchmarks can overestimate performance through random negatives, easy decoys, limited target coverage, and non-standardized evaluation protocols. We introduce TopU-LBVS, a multi-target benchmark for LBVS under hard-negative screening conditions. Starting from curated ChEMBL~35 bioactivity data, TopU-LBVS covers 93 protein targets across 7 protein classes and constructs target-specific screening libraries with property-matched, structurally similar decoys at a fixed 1:40 active-to-decoy ratio. Libraries contain roughly 400 to 10,000 compounds and are designed to reduce simple physicochemical and nearest-neighbor fingerprint shortcuts.   TopU-LBVS provides three fixed protocols. TopU-LBVS-full evaluates ChEMBL$^\ast \rightarrow$ TopU generalization across all 93 targets. TopU-LBVS-low evaluates low-data TopU $\rightarrow$ Top
    
[^60]: TTLab 参加StanceEval-2026：一种用于阿拉伯语立场检测的完形填空式提示方法

    TTLab at StanceEval-2026: A Cloze-Style Prompting Approach for Arabic-Language Stance Detection (CLASP-Ar)

    [https://arxiv.org/abs/2609.29733](https://arxiv.org/abs/2609.29733)

    该论文提出 CLASP-Ar，通过将阿拉伯语立场检测转化为完形填空式的掩码语言建模提示方法，简化了以往多任务学习方案的额外复杂性。

    

    阿拉伯语立场检测仍然是一项具有挑战性的任务，以往的共享任务系统主要依赖于多任务学习和模型集成方法。虽然这些系统取得了最先进的性能，但多任务学习引入的额外复杂性限制了它们的适用性和可迁移性。为了降低这种复杂性，我们提出了 CLASP-Ar，该方法将任务重新表述为完形填空式的掩码语言建模。在这种方法中，目标对象、预测的情感倾向和文本被组合成一个单一的提示，其中 [MASK] 位置的预测被限制在由语言化器约束的标签词汇表内。

    arXiv:2609.29733v1 Announce Type: new  Abstract: Arabic-language stance detection remains challenging, and previous shared-task systems have largely relied on multitask learning and ensembles. While these systems achieve state-of-the-art performance, their applicability and transferability are limited by the additional complexity introduced by multitask learning.To reduce this complexity, we introduce $\texttt{CLASP-Ar}$, which reformulates the task as cloze-style masked language modeling. In this approach, the target, predicted sentiment, and text are combined into a single prompt whose $\texttt{[MASK]}$ prediction is restricted to a verbalizer-constrained label vocabulary.
    
[^61]: 在分布偏移下，重新验证优于有状态路由的科学代理模型选择策略

    Revalidation Beats Stateful Routing for Scientific Surrogates Under Distribution Shift

    [https://arxiv.org/abs/2609.29715](https://arxiv.org/abs/2609.29715)

    该研究通过大规模可复现的流式基准实验证明，在分布偏移条件下，无需复杂的有状态自适应控制器，只需在每个新数据批次上重新验证候选代理模型并选择验证损失最低者，即可将平均对数遗憾降至0.091，显著优于事后最优的固定模型选择策略（0.192）。

    

    代理模型通常在开发阶段被选定，之后随着新测量数据的到来而保持不变。当噪声、输入支持范围或物理参数发生变化时，这种做法就变得有风险。我们提出了一个问题：这些变化是否需要引入有状态的自适应控制器，还是只需在每个新批次上对候选模型重新进行验证就足够了。为研究这一问题，我们构建了RegimeShift-Surrogates，这是一个可复现的流式基准测试，涵盖八个解析与动力学任务、四种平稳或偏移的机制、十个保留的随机种子，以及八个基于经典方法、多层感知机和Kolmogorov-Arnold网络的代理模型。验证性实验包含30,720次模型拟合和3,200个评分部署窗口。选择当前窗口中验证损失最低的模型，相对于逐窗口预言机的平均对数遗憾为0.091；而事后选择的最优固定模型的遗憾为0.192。两者的配对差异为-0.101（分层自助法……

    arXiv:2609.29715v1 Announce Type: cross  Abstract: Surrogate models are often chosen during development and then left in place as new measurements arrive. That practice becomes risky when noise, input support, or physical parameters change. We asked whether such changes call for a stateful adaptive controller, or whether it is enough to validate the candidate models again on each new batch. To study this question, we built RegimeShift-Surrogates, a reproducible streaming benchmark spanning eight analytic and dynamical tasks, four stationary or shifting regimes, ten held-out seeds, and eight classical, multilayer-perceptron, and Kolmogorov-Arnold network surrogates. The confirmatory run contains 30,720 model fits and 3,200 scored deployment windows. Choosing the model with the lowest validation loss in the current window yields mean log regret 0.091 against a per-window oracle; the best fixed model chosen in hindsight yields 0.192. The paired difference is -0.101 (hierarchical bootstrap
    
[^62]: RAPTOR：随机投影物理信息驱动的电力系统暂态求解器

    RAPTOR: RAndom-projection Physics-informed Transient sOlveR

    [https://arxiv.org/abs/2609.29714](https://arxiv.org/abs/2609.29714)

    本文提出了RAPTOR——首个利用基于固定高斯径向基函数的物理信息随机投影神经网络来表示混合微分代数方程未知轨迹的新型电力系统时域仿真框架，以应对基于变换器的资源所带来的多时间尺度暂态仿真挑战。

    

    现代电力系统时域仿真的复杂性显著增加，因为基于变换器的资源引入了控制动态，这些控制动态必须与较慢的系统级动态和快速的电磁动态同时进行仿真。由此产生的宽时间尺度范围可能迫使经典时域求解器采用小的时间步长，使每个时间步长处的非线性方程求解变得复杂，并降低求解器在强非线性、多时间尺度暂态条件下的可靠性。本文提出了RAPTOR，这是首个面向电力系统动态仿真的时域仿真框架。RAPTOR的核心在于引入了一种新的积分技术，利用由固定高斯径向基函数（RBF）构建的物理信息随机投影神经网络（PIRPNN）来表示混合微分代数方程（DAE）在一个时间区间上的未知轨迹。非线性求解器，例如……（摘要在此处被截断）

    arXiv:2609.29714v1 Announce Type: cross  Abstract: The complexity of time-domain simulation of modern power systems has increased significantly because converter-based resources introduce control dynamics that must be simulated alongside slower system-level and fast electromagnetic dynamics. The resulting wide range of timescales may force classical time-domain solvers to use small timesteps, complicate the solution of nonlinear equations at each timestep, and reduce solver reliability under strongly nonlinear and multi-timescale transient conditions. This paper introduces RAPTOR, a first-of-its-kind time-domain simulation framework for power system dynamic simulations. At its core, RAPTOR introduces a new integration technique that represents the unknown trajectory of hybrid differential-algebraic equations (DAEs) over a time interval using a physics-informed random-projection neural network (PIRPNN) built from fixed Gaussian radial basis functions (RBFs). A nonlinear solver, e.g., Ne
    
[^63]: 解耦知识与隐私：面向大语言模型持续学习的任务后自蒸馏重放

    Decoupling Knowledge and Privacy: Post-Task Self-Distillation Replay for LLM Continual Learning

    [https://arxiv.org/abs/2609.29711](https://arxiv.org/abs/2609.29711)

    该论文提出SPARK方法，通过将知识保留与隐私修正解耦——先冻结任务后学到的分布作为稳定参考，再围绕其进行针对稀疏敏感位置的选择性修正——从而在大语言模型持续学习中兼顾知识保留与隐私保护。

    

    隐私保护的持续学习（PPCL）必须在跨序列任务保留有用知识的同时，减少敏感内容的再现。形式化隐私保证刻画的是随机化机制，而操作性输出控制关注的则是训练后的模型能否选择性地降低其输出中敏感内容出现的可能性。在这项工作中，我们在现实的任务演化情境下，将后者与持续学习效用放在一起进行研究。知识保留与隐私修正作用于不同的粒度：任务获取需要广泛保留当前任务和旧任务的行为，而隐私修正则针对稀疏的标注位置。联合优化会导致当前任务的保留目标不断变化。我们提出了SPARK，一种保留-修正分解方法，它首先冻结学习到的任务后分布，然后围绕这一稳定参考进行选择性修正。

    arXiv:2609.29711v1 Announce Type: cross  Abstract: Privacy-preserving continual learning (PPCL) must reduce the reproduction of sensitive content while retaining useful knowledge across sequential tasks. Formal privacy guarantees characterize randomized mechanisms, whereas operational output control concerns whether a trained model selectively reduces the likelihood of sensitive content in its outputs. In this work, we investigate the latter together with continual-learning utility under realistic task evolution. Retention and privacy correction operate at different granularities: task acquisition requires broad preservation of current- and old-task behavior, whereas privacy correction targets sparse annotated positions. Joint optimization leaves the current-task preservation target continually changing. We propose SPARK, a retention-correction decomposition that first freezes the learned post-task distribution and then applies selective correction around this stable reference. Self-Di
    
[^64]: 经典测试理论误导LLM评判者的三种方式

    Three Ways Classical Test Theory Misleads for LLM Judges

    [https://arxiv.org/abs/2609.29709](https://arxiv.org/abs/2609.29709)

    该论文揭示经典测试理论的三个常用信度统计量在LLM评判者评估情境中含义发生扭曲——例如内部一致性系数无法区分题目设计与评判者错误的影响——因此不能直接照搬用于解读LLM评判者的表现。

    

    一个LLM评判者依据评分标准对一批回答进行打分，得到的信度值为0.52——这究竟测量了什么？评判者评估领域已开始借用经典测试理论的信度统计量，但通常并未说明每个统计量所假设的测量设计。我们证明，三个被广泛移植的统计量对评判者的含义与其对测试的含义并不相同，因为评判者情境重新排列了这些测量设计所依赖的角色。第一，基于评分标准要素计算的内部一致性系数不包含评分者维度：将某一评判者的实测错误率固定在4.72%时，随着题库的重新设计，KR-20仍在0.01至0.68之间变化，而改变评判者错误也会使该系数产生相当幅度的变动，因此题目设计与评判者错误无法被分别识别，任何单一数值都不能被解读为评判者本身的属性。第二，依存性指数 Φ(λ) 是一个方差比值……

    arXiv:2609.29709v1 Announce Type: cross  Abstract: An LLM judge scores a bank of responses against a rubric, and the reliability comes back at $0.52$. What has been measured? Judge evaluation has begun borrowing reliability statistics from classical test theory, usually without stating the measurement design each statistic assumes, and we show that three widely portable ones mean something different for a judge than for a test because the judge setting rearranges the roles those designs rest on. First, an internal-consistency coefficient computed over rubric elements contains no scorer facet. Holding one judge's measured error rate fixed at $4.72\%$, KR-20 still ranges from $0.01$ to $0.68$ as the item bank is redesigned around it, and varying judge error moves the coefficient by a comparable amount, so item design and judge error are not separately identified and no single value can be read as a property of the judge. Second, the dependability index $\Phi(\lambda)$ is a ratio of varia
    
[^65]: 面向安全的行人轨迹预测：在城市交叉口利用碰撞时间与过街区域上下文

    Safety-oriented pedestrian trajectory prediction at urban intersections using time-to-collision and crossing-zone context

    [https://arxiv.org/abs/2609.29706](https://arxiv.org/abs/2609.29706)

    本文提出一种融合碰撞时间（TTC）与过街区域上下文的面向安全的行人轨迹预测框架，并引入超出1米阈值误差的频率和幅度作为评估指标，以更好地捕捉与安全相关的预测性能。

    

    准确的行人轨迹预测对于主动式道路安全应用十分重要，尤其是在城市交叉口，因为行人的运动同时受到车辆交互和过街环境的影响。本研究提出了一种面向安全的轨迹预测框架，将行人运动历史与碰撞时间信息和过街区域指标相结合。该研究使用inD（Intersection Drone）数据集中某一城市交叉口的自然驾驶轨迹，在1.6秒观测时长和2.4秒预测时长的设定下评估了多种神经网络架构。该方法采用一个池化长短期记忆网络分别对TTC历史和过街区域上下文进行编码，然后将其与行人位置整合。除了传统的平均位移误差（ADE）和最终位移误差（FDE）之外，预测性能还通过超过研究设定的1米误差阈值的频率和幅度进行评估。

    arXiv:2609.29706v1 Announce Type: new  Abstract: Accurate pedestrian trajectory prediction is important for proactive road-safety applications, particularly at urban intersections where pedestrian motion is shaped by both vehicle interactions and crossing context. This study presents a safety-oriented trajectory-prediction framework that combines pedestrian motion history with Time-to-Collision (TTC) information and crossing-zone indicators. Using naturalistic trajectories from one urban intersection in the inD (Intersection Drone) dataset, several neural architectures were evaluated with 1.6 s observation and 2.4 s prediction horizons. A pooled Long Short-Term Memory (LSTM) separately encodes TTC histories and crossing-zone context before integrating them with pedestrian positions. In addition to conventional Average Displacement Error (ADE) and Final Displacement Error (FDE), prediction performance was assessed using the frequency and magnitude of errors exceeding a study-defined 1 m
    
[^66]: 一种在 fat-shattering 维度上具有近线性大小的平方损失不可知样本压缩方案

    An Agnostic Sample Compression Scheme for Squared Loss of Near-Linear Size in the Fat-Shattering Dimension

    [https://arxiv.org/abs/2609.29696](https://arxiv.org/abs/2609.29696)

    本文首次构造了大小仅与 fat-shattering 维度近线性相关（乘以对数因子）且与样本量无关的平方损失不可知样本压缩方案，正面解决了 Attias 等人（ICML 2024）提出的开放性问题。

    

    我们为每一个函数类 $\mathcal{F}\subseteq[0,1]^{\mathcal{X}}$ 以及每一个精度 $0<\alpha\le 1$ 构造了一个针对经验平方损失的不可知样本压缩方案：对于任意带有（含噪声）标签的有限样本 $S\in(\mathcal{X}\times[0,1])^m$，该方案至多存储 $O(\mathrm{fat}(\mathcal{F},c'\alpha)\cdot\log^3(2/\alpha))$ 个原始标注样本和辅助比特，其大小与样本量 $m$ 无关，并能重构出一个函数 $\hat f$，满足 $L_2(\hat f,S)\le\inf_{f\in\mathcal{F}}L_2(f,S)+\alpha$。这正面解决了 Attias、Hanneke、Kontorovich 和 Sadigurschi（ICML 2024，第 5 节）提出的开放性问题，即要求构造大小为 $\mathrm{fat}(\mathcal{F},c\alpha)\cdot\mathrm{polylog}(c/\alpha)$ 的不可知 $\ell_2$ 压缩方案。此前所有已知的有界大小构造——无论是不可知情形还是甚至可实现情形——都会引入一个乘性的对偶 fat-shattering 因子，而该因子可能是指数级大的。

    arXiv:2609.29696v1 Announce Type: new  Abstract: We construct, for every function class $\mathcal{F}\subseteq[0,1]^{\mathcal{X}}$ and every accuracy $0<\alpha\le 1$, an agnostic sample compression scheme for the empirical squared loss: for every finite sample $S\in(\mathcal{X}\times[0,1])^m$ with arbitrary (noisy) labels, the scheme stores at most $O(\mathrm{fat}(\mathcal{F},c'\alpha)\cdot\log^3(2/\alpha))$ original labeled examples and auxiliary bits, independent of the sample size $m$, and reconstructs a function $\hat f$ with $L_2(\hat f,S)\le\inf_{f\in\mathcal{F}}L_2(f,S)+\alpha$. This resolves, in the positive, the open problem of Attias, Hanneke, Kontorovich, and Sadigurschi (ICML 2024, Section 5), which asks for an agnostic $\ell_2$ compression scheme of size $\mathrm{fat}(\mathcal{F},c\alpha)\cdot\mathrm{polylog}(c/\alpha)$. All previously known bounded-size constructions, agnostic and even realizable, incur a multiplicative dual fat-shattering factor, which can be exponentiall
    
[^67]: 老虎机多类PAC学习：修正的下界、精确类族与置信度直和现象

    Bandit Multiclass PAC Learning: Corrected Lower Bounds, Exact Families, and a Confidence Direct-Sum Phenomenon

    [https://arxiv.org/abs/2609.29694](https://arxiv.org/abs/2609.29694)

    本文修正了老虎机多类PAC学习的下界理论，指出原有基于BDS维度的下界并不正确，引入锚定维度aBDS证明了新的无常数下界，同时在上界中完全去除了对标签数K的依赖，并揭示了置信度直和现象。

    

    我们研究具有老虎机反馈的可实现多类PAC学习：学习器观察到一个独立同分布的样本实例，预测K个标签中的一个，并且仅获知预测是否正确。Hanneke、Meng、Moran和Shaeiri（arXiv:2605.25678）通过对数因子范围内的老虎机DS维度BDS刻画了最优样本复杂度，并提出问题：是否每个类都具有样本复杂度O((BDS+log(1/δ))/ε)。首先，我们证明已发表的下界Ω((BDS+log(1/δ))/ε)如所述是不正确的：我们给出了BDS=K-1的显式类族，其样本复杂度呈指数级更小，并定位了其证明中两个独立的漏洞。我们围绕一个新的锚定维度aBDS≤BDS修复了下界理论，证明了一个无常数的由三部分构成的下界。在上界方面，我们完全移除了环境标签数K的依赖，证明了O(...（摘要在此处截断）

    arXiv:2609.29694v1 Announce Type: new  Abstract: We study realizable multiclass PAC learning with bandit feedback: the learner observes an i.i.d. instance, predicts one of $K$ labels, and learns only whether the prediction was correct. Hanneke, Meng, Moran, and Shaeiri (arXiv:2605.25678) characterized the optimal sample complexity via the bandit DS dimension $\mathrm{BDS}$ up to logarithmic factors, and asked whether every class admits sample complexity $O((\mathrm{BDS}+\log(1/\delta))/\epsilon)$.   First, we show that the published lower bound $\Omega((\mathrm{BDS}+\log(1/\delta))/\epsilon)$ is incorrect as stated: we exhibit explicit classes with $\mathrm{BDS}=K-1$ whose sample complexity is exponentially smaller, and locate two independent gaps in its proof. We repair the lower-bound theory around a new anchored dimension $\mathrm{aBDS}\le\mathrm{BDS}$, proving a constant-free three-part lower bound. On the upper-bound side we remove the ambient label count $K$ entirely, proving $O(
    
[^68]: 使用新型过采样方法预测大学生的动机缺乏和快感缺失症状

    Predicting Symptoms of Amotivation and Anhedonia among University Students with a Novel Oversampling Method

    [https://arxiv.org/abs/2609.29690](https://arxiv.org/abs/2609.29690)

    本文提出了一种新型过采样方法，以解决机器学习模型在预测大学生动机缺乏和快感缺失症状时的类别不平衡问题，从而更准确地识别高风险学生。

    

    大学生患抑郁症等常见心理健康问题的比例异常偏高，这些问题可能损害其学习、社会功能和整体幸福感。在此背景下，动机缺乏（即动机驱动力丧失）和快感缺失（即兴趣或愉悦感减退）的症状尤为使人衰弱，却常常未被察觉。开发新方法来识别具有明显动机缺乏和快感缺失症状的学生，有助于实现更早、更有针对性的干预。机器学习方法已被越来越多地用于根据症状严重程度对个体进行分类。然而，这些机器学习模型往往受到类别不平衡问题的困扰，即大多数样本属于低症状组，而属于高症状组的样本相对较少。这种不平衡会降低模型准确性并使预测产生偏差。为解决这一问题，研究通常采用流行的过采样策略SMOTE。

    arXiv:2609.29690v1 Announce Type: new  Abstract: University students experience disproportionately high rates of common mental health conditions, such as depression, which can impair learning, social functioning, and overall well-being. Within this context, symptoms of amotivation (i.e. loss of motivational drive) and anhedonia (i.e. diminished interest or pleasure) are particularly debilitating, yet they frequently go undetected. Developing new approaches to identify students with prominent amotivation and anhedonia could enable earlier and more targeted intervention. Machine learning (ML) methods have increasingly been used to classify individuals according to symptom severity. However, these ML models often suffer from class imbalance, where the majority of cases fall in the low-symptom group and relatively few in the high-symptom group. This imbalance can reduce model accuracy and bias predictions. To address this, studies commonly employ the popular oversampling strategy SMOTE. Ho
    
[^69]: 并非所有合成数据都同等可靠：面向自动驾驶系统不平衡事故伤害严重程度预测的专家委员会审计筛选方法

    Not All Synthetic Data Are Equal: Expert-Committee Audit Screening for Imbalanced Crash-Injury-Severity Prediction in Automated Driving Systems

    [https://arxiv.org/abs/2609.29687](https://arxiv.org/abs/2609.29687)

    提出专家委员会审计筛选（ECAS）框架，利用仅基于真实数据的专家委员会从标签支持度、边界分离度、委员会一致性和局部合理性等维度对合成少数类样本进行可信度审计，从而提升自动驾驶系统中不平衡事故伤害严重程度预测的可靠性。

    

    自动驾驶系统（ADS）正日益在公共道路上运行，引发了安全方面的担忧。然而，由于事故报告数量有限、严重伤害结果罕见且伤害类别高度不平衡，可靠的事故伤害严重程度预测仍然十分困难。现有的数据增强方法主要致力于增加少数类样本的数量，但很少评估所生成的样本对于安全关键型预测是否可信。本研究提出了专家委员会审计筛选（ECAS），这是一个面向数据不平衡条件下自动驾驶系统事故伤害严重程度预测的可信度感知样本接受框架。研究使用来自美国国家公路交通安全管理局（NHTSA）常设通用命令记录中的1,477起事故级别的自动驾驶系统碰撞事故，ECAS通过一个仅基于真实数据构建的专家委员会，依据标签支持度、边界分离度、委员会一致性和局部合理性等标准对生成的少数类样本进行审计筛选。类内百分位归一化和Pare（原文摘要在此处截断）

    arXiv:2609.29687v1 Announce Type: new  Abstract: Automated driving systems (ADSs) are increasingly operating on public roads, raising safety concerns, yet reliable prediction of crash injury severity remains difficult because crash reports are limited, severe outcomes are rare, and injury classes are highly imbalanced. Existing augmentation methods mainly increase minority-class sample size but rarely assess whether generated samples are credible for safety-critical prediction. This study proposes Expert-Committee Audit Screening (ECAS), a credibility-aware sample acceptance framework for ADS crash injury severity prediction under data imbalance. Using 1,477 incident-level ADS crashes from the National Highway Traffic Safety Administration Standing General Order records, ECAS audits generated minority samples through a real-data-only expert committee based on label support, boundary separation, committee agreement, and local plausibility. Within-class percentile normalization and Paret
    
[^70]: 使用滑动窗口方法的命名实体识别

    Named Entity Recognition using Sliding Window Approach

    [https://arxiv.org/abs/2609.29682](https://arxiv.org/abs/2609.29682)

    提出一种无需重新训练或修改架构的推理阶段滑动窗口流水线，将冻结的句子级NER模型扩展至文档级命名实体识别，有效解决了长文档中的截断丢内容和实体割裂问题。

    

    命名实体识别（NER）是自然语言处理的核心任务，但基于Transformer的句子级模型由于固定输入长度的限制，在处理长文档时面临困难：截断会丢失内容，而非重叠分块会在片段边界处割裂实体。我们提出了一种仅用于推理阶段的流水线，通过重叠滑动窗口将在MahaNER语料库上微调的冻结NER模型MahaNER-BERT扩展到文档级预测，并将多个窗口合并为单一标注，无需任何重新训练或架构更改。我们在由MahaNER测试集构建的六个文档级语料库上评估了该流水线，采用两种策略：Normal Repeat（重复句子序列以延长长度并保持上下文连续性）和Random Repeat（拼接不同的序列以产生更长、异构的输入），每种策略在三个长度级别上实例化，并在多种滑动窗口配置下进行测试。该模型保持……

    arXiv:2609.29682v1 Announce Type: new  Abstract: Named Entity Recognition (NER) is a core NLP task, but transformer-based sentence-level models struggle with long documents because of fixed input-length limits: truncation drops content, and non-overlapping chunking fragments entities at segment boundaries. We introduce an inference-only pipeline that extends a frozen NER model, MahaNER-BERT, fine-tuned on the MahaNER corpus, to document-level prediction via overlapping sliding windows that are merged into a single annotation, without any retraining or architectural change.   We evaluate the pipeline on six document-level corpora built from the MahaNER test set using two strategies: Normal Repeat, which duplicates sentence sequences to extend length while preserving contextual continuity, and Random Repeat, which concatenates distinct sequences to produce longer, heterogeneous inputs, each instantiated at three length levels, across several sliding-window configurations. The model retai
    
[^71]: 持续学习的序列代价

    The Sequential Price of Continual Learning

    [https://arxiv.org/abs/2609.29674](https://arxiv.org/abs/2609.29674)

    该论文在过参数化线性回归模型中首次精确刻画了持续学习的“序列代价”：总损失恰好分解为联合训练的内在损失与额外序列代价，在同质任务几何下总损失为联合训练的两倍，且EWC正则化能以与强度成反比的方式降低该代价。

    

    序列式任务更新是持续学习的基础，但其对新任务的近因偏差可能带来持久的性能代价。我们在一个采用独立同分布任务采样的过参数化线性回归模型中研究这一代价。我们证明，分布层面的遗忘与总体损失会收敛到同一个平稳极限。这一共同极限可以精确分解为两部分：联合训练渐近达到的内在损失，以及一个额外的“序列代价”；在更同质的任务几何结构下，这两项恰好相等，使得总损失达到联合训练的两倍。我们进一步分析了在一般任务曲率下固定强度的弹性权重固化方法，并刻画了其在每个正则化强度下的平稳序列代价。在强正则化条件下，该代价随EWC强度成反比衰减，同时收敛到平稳状态的速度也以相同尺度变慢。在Jester笑话评分数据集上，该理论精确量化了……（原文在此处截断）

    arXiv:2609.29674v1 Announce Type: cross  Abstract: Sequential task updates are fundamental to continual learning, but their recency bias can impose a lasting performance cost. We study this cost in an overparameterized linear-regression model with i.i.d. task sampling. We prove that distribution-level forgetting and population loss converge to the same stationary limit. This common limit separates exactly into the intrinsic loss asymptotically attained by joint training and an additional sequential price, and in more homogeneous task geometries the two terms coincide, making the total loss twice that of joint training. We further analyze fixed-strength elastic weight consolidation (EWC) under general task curvatures and characterize its stationary sequential price at every regularization strength. Under strong regularization, the price decays inversely with EWC strength while convergence to stationarity slows at the same scale. On the Jester joke-rating dataset, the theory exactly quan
    
[^72]: GBFRVFL：基于粒球计算的模糊随机向量函数链接网络

    GBFRVFL: Granular-Ball Computing-Based Fuzzy Random Vector Functional Link Network

    [https://arxiv.org/abs/2609.29670](https://arxiv.org/abs/2609.29670)

    该论文提出基于粒球计算的GBFRVFL框架，将原始样本抽象为自适应粒球，并通过模糊隶属度和新型统计密度自适应毕达哥拉斯隶属度（SDAPM）两种方案量化粒球可靠性，从而有效应对数据噪声、离群点和类别不平衡，弥补了传统RVFL网络无法显式处理不确定性的不足。

    

    在实际机器学习任务中，数据常常受到噪声、离群点和类别不平衡的污染，这会降低传统模型的性能。虽然随机向量函数链接（RVFL）网络具有训练速度快、泛化能力强的优点，但其并未显式地处理不确定性，也未利用数据的局部结构。为了解决这些局限性，我们提出了一种模糊粒球随机向量函数链接（GBFRVFL）框架，该框架利用粒球计算将原始样本抽象为自适应的粒球。在该框架内，我们引入了两种隶属度分配方案：（i）F-GBRVFL，它引入模糊隶属度来量化每个粒球的可靠性；（ii）SDAP-GBRVFL，我们提出了一种新颖的统计密度自适应毕达哥拉斯隶属度（SDAPM）方案，该方案能够基于类别方差、局部稀疏性等因素动态调整隶属度与非隶属度值……

    arXiv:2609.29670v1 Announce Type: new  Abstract: In practical machine learning tasks, data are often contaminated with noise, outliers, and class imbalance, which can degrade the performance of conventional models. While random vector functional link (RVFL) networks offer fast training and strong generalization, they do not explicitly handle uncertainty or exploit local data structure. To address these limitations, we propose a fuzzy granular-ball random vector functional link (GBFRVFL) framework that leverages granular-ball computing to abstract raw samples into adaptive granular balls. Within this framework, we introduce two membership assignment schemes: (i) F-GBRVFL, which incorporates fuzzy membership to quantify the reliability of each granular ball, and (ii) SDAP-GBRVFL, which we propose, incorporates a novel statistical density-adaptive pythagorean membership (SDAPM) scheme that dynamically adjusts membership and non-membership values based on class variance, local sparsity, an
    
[^73]: 图、循环与线束工程：面向零信任智能体数据工程与分析处理

    Graph, Loop, and Harness Engineering for Zero-Trust Agentic Data Engineering and Analytical Processing

    [https://arxiv.org/abs/2609.29668](https://arxiv.org/abs/2609.29668)

    该论文提出零信任智能体数据工程与零信任智能体OLAP两个框架，通过图工程、循环工程与线束工程三种抽象，以证据门控、有界恢复和严格验证机制确保大语言模型智能体可靠地完成端到端云数据工程与分析处理。

    

    arXiv:2609.29668v1（公告类型：交叉列表）。摘要：大语言模型智能体日益推动数据工作流的自动化，但端到端的云数据工程与分析执行需要在代码、数据、基础设施和运行时环境之间实现可靠的协调。我们提出了两个零信任框架。零信任智能体数据工程从自然语言任务出发，生成、部署并验证完整的云数据工程解决方案，其完成以仓库、部署、运行时和策略证据为前提条件。零信任智能体OLAP将受治理的数据准备与经验证的联机分析处理（OLAP）相结合，仅在通过验证并获得证据绑定的批准后才允许投入生产，且只有在经过同快照执行、精确结果等价、确定性落地和反思之后才发布分析答案。两个框架共享三个抽象：用于证据门控工作流结构的图工程、用于有界恢复的循环工程……（原文摘要至此截断）

    arXiv:2609.29668v1 Announce Type: cross  Abstract: Large language model agents increasingly automate data workflows, but end-to-end cloud data engineering and analytical execution require reliable coordination across code, data, infrastructure, and runtime environments. We present two zero-trust frameworks. Zero-Trust Agentic Data Engineering generates, deploys, and verifies complete cloud data-engineering solutions from natural-language tasks, with completion conditioned on repository, deployment, runtime, and policy evidence. Zero-Trust Agentic OLAP combines governed Data Preparation with verified Online Analytical Processing (OLAP), permitting production promotion only after validation and evidence-bound approval, and releasing analytical answers only after Same-Snapshot Execution, Exact Result Equivalence, deterministic grounding, and reflection. Both frameworks share three abstractions: graph engineering for evidence-gated workflow structure, loop engineering for bounded recovery,
    
[^74]: VG-TIE：一种基于可视图的可解释表格数据到图像编码方法

    VG-TIE: An interpretable tabular-to-image encoding method based on visibility graphs

    [https://arxiv.org/abs/2609.29650](https://arxiv.org/abs/2609.29650)

    本文提出VG-TIE，一种基于自然可视图与水平可视图结合PCA的可解释表格数据到图像编码方法，使生成的图像与模型无关且每个像素对应一个输入特征、像素强度反映该特征相对总体均值的偏离方向和幅度。

    

    表格数据到图像的编码方法能够将特征向量转换为图像，从而使基于卷积神经网络和视觉Transformer的模型得以应用于表格数据。现有方法采用线性和非线性降维技术（如主成分分析（PCA）、t-SNE和UMAP）来确定像素位置，导致图像的空间布局无法从本质上反映特征之间的关系。本文提出了VG-TIE，这是一种新颖的方法，它利用自然可视图（NVG）和水平可视图（HVG）将特征值的结构编码到通过PCA获得的二维空间中。所生成的图像与模型无关且具有内在可解释性：每个像素对应一个输入特征，其强度反映该特征相对于总体均值的偏离幅度和方向，边表示特征之间的关联关系。

    arXiv:2609.29650v1 Announce Type: cross  Abstract: Tabular-to-image encoding methods enable the application of models based on both convolutional neural networks and vision transformers to tabular data, transforming feature vectors into images. Existing methods employ linear and nonlinear dimensionality reduction techniques (e.g., Principal Component Analysis (PCA), t-SNE, and UMAP) to determine pixel positions, resulting in images whose spatial layout do not inherently reflect feature relationships. This paper introduces Visibility Graphs for Tabular-to-Image Encoding (VG-TIE), a novel method that encodes the structure of feature values using Natural Visibility Graph (NVG) and Horizontal Visibility Graph (HVG) into a two-dimensional space obtained through PCA. The resulting images are model-agnostic and intrinsically interpretable. Each pixel corresponds to an input feature, its intensity reflects the magnitude and direction of deviation from the population mean, and edges represent f
    
[^75]: Albireo：面向边缘端视频目标检测的自适应节能推理框架

    Albireo: Adaptive, Energy-Efficient Inference Framework for Video Object Detection on the Edge

    [https://arxiv.org/abs/2609.29648](https://arxiv.org/abs/2609.29648)

    Albireo是一个面向边缘设备的自适应节能视频目标检测推理框架，它为每个目标维护卡尔曼滤波器并仅在预测不确定性超过阈值时才调用检测器，在无需修改或重训检测器的情况下大幅降低GPU能耗，同时避免了盲目跳帧带来的检测质量下降。

    

    摘要：边缘设备上的视频目标检测需要在长帧流上运行计算开销高昂的检测器，导致高能耗和持续的GPU占用。尽管连续帧之间高度冗余，但朴素的跳帧方法是内容盲目的：它会在目标进入、遮挡恢复和剧烈运动等关键时刻跳帧，从而降低检测质量。我们提出了Albireo，一个与检测器无关、无需编解码器的自适应推理框架，它可以封装现成的检测器，并根据场景内容和每个目标的时序状态来决定何时可以安全地跳过检测器调用，无需修改或重新训练检测器。Albireo为每个活跃目标维护一个10维卡尔曼滤波器（KF），仅当预测不确定性超过阈值时才调用检测器；在跳过的帧上，边界框由KF状态预测得出，GPU开销几乎为零。基于KF的救援机制可以保留已确认的目标（摘要在此处截断）

    arXiv:2609.29648v1 Announce Type: cross  Abstract: Video object detection on edge devices runs computationally expensive detectors over long frame streams, causing high energy consumption and sustained GPU utilization. Although consecutive frames are highly redundant, naive frame skipping is content-blind: it skips during critical moments such as object entry, occlusion recovery, and abrupt motion, degrading detection quality. We present Albireo, a detector-agnostic, codec-free adaptive inference framework that wraps off-the-shelf detectors and decides when detector invocation can be safely skipped based on scene content and per-object temporal state, requiring no detector modification or retraining. Albireo maintains a 10-dimensional Kalman filter (KF) per active object and invokes the detector only when prediction uncertainty exceeds a threshold; on skipped frames, boxes are predicted from the KF state at near-zero GPU cost. A KF-based rescue mechanism preserves confirmed objects thr
    
[^76]: 办公室规模验证搜索中的算子包、提议者强度与构造型家族平台期

    Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search

    [https://arxiv.org/abs/2609.29636](https://arxiv.org/abs/2609.29636)

    该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。

    

    验证搜索是指语言模型提出程序、硬评估器对其进行评分、选择机制保留最优解的过程，这种方法近来已推动了数学纪录的进展；但对提议者侧组件的受控消融实验仍然罕见。我们在办公规模上（笔记本电脑上运行的30B本地模型，每次运行120-600个验证样本）对最小化的FunSearch风格循环进行了仪器化，采用三种算子包：模型自行编写并携带的示意图式笔记本（代替逐字复制的精英解）、命名障碍、以及对已发现构造的行为排斥。在来自公共仓库的九个构造问题上，带两次重复的完整2³因子设计在名义两阶段分析中支持主要对比：该组合缩小了更多从种子解到纪录的差距（+0.196；名义合并p=0.023，阶段组合p≈0.08；每问题效应中位数为+0.045）。排斥机制在各处都提高了构造哈希多样性（p=0.0039；部分属于操纵检查）（注：原文摘要至此处截断）。

    arXiv:2609.29636v1 Announce Type: cross  Abstract: Verified search, in which a language model proposes programs, a hard evaluator scores them, and selection keeps the best, has recently moved mathematical records; controlled ablations of the proposer-side components remain rare. We instrument a minimal FunSearch-style loop at office scale (a 30B local model on a laptop, 120-600 verified samples per run) with three operator packages: a schematic notebook the model writes and carries instead of verbatim elites, a named obstacle, and behavioural repulsion from constructions already found. On nine construction problems from a public repository, the complete 2^3 factorial with two replicates favours the primary contrast in a nominal two-stage analysis: the composition closes more of the seed-to-record gap (+0.196; nominal pooled p=0.023, stage-combination p~0.08; median per-problem effect +0.045). Repulsion raises construction-hash diversity everywhere (p=0.0039; partly a manipulation check
    
[^77]: TTLab参加AlexandriaX-2026竞赛：面向阿拉伯语机器翻译错误跨度检测与分类的微调表层标注器

    TTLab at AlexandriaX-2026: A Fine-Tuned Surface Tagger for Arabic Machine-Translation Error-Span Detection and Classification

    [https://arxiv.org/abs/2609.29633](https://arxiv.org/abs/2609.29633)

    该论文提出基于MARBERTv2微调的词元级分类系统，结合焦点损失、类别权重和方言特定解码阈值应对标签不平衡问题，在AlexandriaX-2026阿拉伯语机器翻译错误跨度检测与分类任务中获得第三名。

    

    我们介绍了TTLab参加AlexandriaX-2026子任务3（阿拉伯语机器翻译错误跨度检测与分类）的提交系统。我们的系统将该任务构建为基于表层形式的词元级分类，并保留字符偏移量以确保与评估指标的精确对齐。为应对严重的标签不平衡问题，我们采用了带类别权重的焦点损失以及方言特定的解码阈值。在六个阿拉伯语预训练编码器中，MARBERTv2取得了最佳整体性能，在开发集和测试集上分别达到40.8和40.91的分数，在所有参赛队伍中排名第三。尽管我们的系统能够有效定位错误跨度，但稀有错误类型的分类仍然具有挑战性，这凸显了针对尾部类别进行数据增强的必要性。代码已在GitHub上开源。

    arXiv:2609.29633v1 Announce Type: cross  Abstract: We present TTLab's submission to the AlexandriaX-2026 Subtask~3 on Arabic MT error span detection and classification. Our system frames the task as token-level classification over surface forms, preserving character offsets to ensure exact alignment with the evaluation metric. To handle severe label imbalance, we employ a focal loss with class weighting and dialect-specific decoding thresholds. Among six Arabic pre-trained encoders, MARBERTv2 achieves the best overall performance of 40.8 and 40.91 on the development and test set, respectively, ranking $\nth{3}$ out of all participating teams. While our system localizes error spans effectively, classification of rare error types remains challenging, highlighting the need for data augmentation for tail categories. The code is available at ${\href{https://github.com/ENTAILab/arabic-dialectal-mt-error-span-detection}{\faGithub~ TTLab at AlexandriaX-2026}$
    
[^78]: 一种基于排序原型的流形感知主题建模方法

    A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes

    [https://arxiv.org/abs/2609.29630](https://arxiv.org/abs/2609.29630)

    MARETopic是一个无需训练的主题建模框架，通过将嵌入投影到低维流形并把主题发现转化为基于排序的原型选择，贪心选出邻域可覆盖语料库的真实文档作为主题原型，其MARETopic_Corr变体在类别最多的两个基准上Purity和NMI领先于神经与聚类主题模型。

    

    近期的主题模型利用预训练嵌入，但神经架构产生的潜在表示缺乏与具体文本的关联，而基于聚类的流水线只能在事后分配代表性文档，依赖于在 高维空间中因枢纽性和各向异性而失真的绝对距离。我们提出了MARETopic，这是一个无需训练的框架，将主题发现转化为基于排序的原型选择。在将嵌入投影到低维流形后，MARETopic构建编码序数邻域结构的排序列表。贪心算法精确选出K个范例文档（即真实的语料库文本），其邻域可覆盖整个语料库。两个变体共享这一准则。其中MARETopic_Corr利用查询性能预测器和秩相关性度量对候选进行评分，在类别最多的两个基准数据集上取得了Purity和NMI的最优结果，领先于神经主题模型和基于聚类的主题模型。

    arXiv:2609.29630v1 Announce Type: cross  Abstract: Recent topic models leverage pretrained embeddings, but neural architectures produce latent representations without grounding in specific texts, and clustering-based pipelines assign representative documents only post hoc, relying on absolute distances distorted by hubness and anisotropy in high-dimensional spaces. We introduce MARETopic, a training-free framework that casts topic discovery as rank-based prototype selection. After projecting embeddings onto a low-dimensional manifold, MARETopic builds ranked lists encoding ordinal neighborhood structure. A greedy algorithm selects exactly K exemplar documents, real corpus texts, whose neighborhoods cover the corpus. Two variants share this criterion. MARETopic$_\text{Corr}$ scores candidates with a query performance predictor and a rank correlation measure, leading Purity and NMI on the two benchmarks with the most categories, ahead of both neural and clustering-based topic models. MAR
    
[^79]: 公共教育预测基准中有限的结构可靠性：七个数据集的四维度审计

    Limited Structural Reliability in Public Educational Prediction Benchmarks: A Four-Dimension Audit of Seven Datasets

    [https://arxiv.org/abs/2609.29625](https://arxiv.org/abs/2609.29625)

    该论文对七个公共教育预测数据集进行四维度建模前审计，发现大多数数据集存在跨组脆弱性——独立同分布性能看似正常但在分组留出下崩溃，且提高模型复杂度也无法弥补，揭示现有教育预测基准的结构可靠性严重不足。

    

    在七个公共教育预测数据集中，仅有三个通过了全部四项建模前可靠性检查；其余四个数据集要么未能通过分组感知泛化测试，要么缺乏执行这些测试所需的来源元数据。其中一个数据集最初被判定为不合格，但在从留出矩阵中排除分组标识符特征后得以修正，这表明该审计方法能够区分真正的跨组混淆与特征编码伪象。每个数据集均在模型优化之前接受了四项检查的审计：基线差距、划分不稳定性、零分离以及分组感知留出下的元数据充分性。主要的失效模式并非仅仅是弱独立同分布性能，而是跨组脆弱性：在最显著的案例中，UCI学生数据集的R²从独立同分布的0.242下降到分组留出下的-0.097，而高等教育数据集则从0.041崩溃至-8.79。增加模型复杂度并不能消除这一模式：集成模型同样表现出这种脆弱性。

    arXiv:2609.29625v1 Announce Type: new  Abstract: Across seven public educational prediction datasets, three passed all four pre-modeling reliability checks; the remaining four either failed group-aware generalization tests or lacked the provenance metadata needed to run them. One dataset was initially classified as failing but corrected after excluding group-identifier features from the holdout matrix, demonstrating that the audit can distinguish genuine cross-group confounding from feature-encoding artifacts. Each dataset was audited before model optimization using four checks: baseline gap, split instability, null separation, and metadata adequacy under group-aware holdout. The dominant failure mode was not weak iid performance alone but cross-group fragility: in the clearest case, UCI Student declined from iid R-squared 0.242 to group-holdout R-squared -0.097, while Higher Ed collapsed from 0.041 to -8.79. Increasing model complexity did not remove this pattern: ensemble models impr
    
[^80]: 最优恢复遇见贝叶斯学习：最坏情况界何时发挥优势

    Optimal Recovery Meets Bayesian Learning: Where Worst-Case Bounds Pay Off

    [https://arxiv.org/abs/2609.29622](https://arxiv.org/abs/2609.29622)

    该论文揭示了最坏情况最优恢复与贝叶斯学习的精确数学对应关系，并通过实验证明 Morozov 校准在噪声盲规则失效、可复现性和后端迁移场景下显著优于发布权重、ML-II 和 GCV 等传统超参数选择规则，但在可交换数据上 split-conformal 方法更胜一筹。

    

    最坏情况最优恢复（OR）与贝叶斯学习用两套术语描述了相同的高斯-二次-希尔伯特问题。我们深化了这一对应关系——信息半径等于经过块金值优化的高斯过程后验方差，并在一个具有闭式解的平衡块金值处由后验均值达到——同时在三个已发表的贝叶斯系统中实测最坏情况方法的优势所在。这本账簿是双面的：损失与收益同样具有启发意义。Morozov 校准在 σ 盲规则失效的场景下能以 1.00-1.19 倍的差距追踪测试访问预言机，在噪声抽取之间的可复现性提高 4.9-6.3 倍（p=0.002-0.004），并且是唯一在后端更换后选择依然稳健的可部署规则（1.36 倍，而发布权重、ML-II 和 GCV 为 12-30 倍）；紧证书在信息论下界处实现覆盖，无数值松弛。但在可交换数据上，split-conformal 方法正面击败了 OR。

    arXiv:2609.29622v1 Announce Type: cross  Abstract: Worst-case Optimal Recovery (OR) and Bayesian learning describe the same Gaussian-quadratic-Hilbert problems in two vocabularies. We sharpen the correspondence - the radius of information equals a nugget-optimized GP posterior variance and is attained by the posterior mean at a closed-form balance nugget - and measure, inside three published Bayesian systems, where the worst-case side pays. The ledger is two-sided: the losses instruct as much as the wins. Morozov calibration tracks a test-access oracle within $1.00$-$1.19\times$ where $\sigma$-blind rules fail, is $4.9$-$6.3\times$ more reproducible across noise draws ($p=0.002$-$0.004$), and is the only deployable rule whose selection survives a change of backend ($1.36\times$ against $12$-$30\times$ for the released weight, ML-II and GCV); tight certificates cover at the information-theoretic floor with no numerical slack. But on exchangeable data split-conformal beats the OR head on
    
[^81]: 一个小型MLA-SSM混合语言模型的探索性消融研究

    An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model

    [https://arxiv.org/abs/2609.29618](https://arxiv.org/abs/2609.29618)

    该消融研究表明，在小MLA-SSM混合语言模型中，SSM分支对性能的贡献大于MLA分支，且密集FFN混合模型以更少的峰值训练内存达到了与三值MoE混合模型相当的困惑度表现。

    

    我们报告了对TALH（Adaptive Latent Hybrid，自适应潜在混合模型）的一项探索性单种子消融实验。TALH是一个仅有解码器的语言模型，结合了并行的多头潜在注意力机制与自定义的循环状态空间分支。五个变体（每token估计活跃参数量在1.17亿至2.17亿之间）在FineWeb样本上从零开始训练，采用相同的优化步数和token数量。在这一特定设置下，移除SSM分支会导致验证困惑度的最大退化（仅MLA模型PPL为315），而移除MLA的影响则小得多（仅SSM模型PPL为239）。密集FFN混合模型取得了231的PPL，而测试的top-2三值MoE混合模型为240，但后者可节省3.87 GB的峰值训练内存。我们还保留了一项初步的Apple M3计时观察：在五个未优化的实现中，仅MLA模型在512至2,048个提示token范围内的首token生成时间曲线最为平坦，尽管密集Transformer的速度要快得多……

    arXiv:2609.29618v1 Announce Type: cross  Abstract: We report an exploratory, single-seed ablation of TALH (Adaptive Latent Hybrid), a decoder-only language model with parallel Multi-head Latent Attention (MLA) and a custom recurrent state-space (SSM) branch. Five variants, spanning 117--217M estimated active parameters per token, are trained from scratch on a FineWeb sample for the same number of optimisation steps and tokens. In this specific setup, removing the SSM branch gives the largest degradation in validation perplexity (MLA-only PPL 315), whereas removing MLA has a much smaller effect (SSM-only PPL 239). A dense-FFN hybrid obtains PPL 231, compared with 240 for the tested top-2 ternary-MoE hybrid, while using 3.87 GB less peak training memory. We also preserve a preliminary Apple M3 timing observation: among the five unoptimised implementations, MLA-only has the flattest measured time-to-first-token curve from 512 to 2,048 prompt tokens, although the dense Transformer is much 
    
[^82]: 基于证据驱动的恶性黑色素瘤鉴别诊断

    Evidence-Driven Differential Diagnosis of Malignant Melanoma

    [https://arxiv.org/abs/2609.29613](https://arxiv.org/abs/2609.29613)

    提出了一种多层次证据驱动的恶性黑色素瘤鉴别诊断框架，通过解剖部位感知的掩码transformer建模患者所有病灶及其发病部位的上下文，并结合可学习的人口统计学嵌入捕捉患者元数据，使诊断特异性分别提升17.15%和7.14%。

    

    我们提出了一种用于恶性黑色素瘤鉴别诊断的模块化、多层次框架。该框架整合了病灶、患者和人群三个层面的上下文信息与证据，实现了各层级的决策。我们引入了一种解剖部位感知的掩码transformer，通过考虑患者体内数量可变的所有病灶及其发病部位，有效建模患者上下文。此外，我们通过可学习的人口统计学嵌入纳入患者元数据，以捕捉人群统计特征。通过大量实验，我们探讨了特定信息对决策过程的影响，并考察了考虑不同类型信息时的指标权衡。在SIIM-ISIC 2020数据集上的验证结果表明，加入包含位置的病灶上下文和元数据分别使特异性提高了17.15%和7.14%，同时提升了……

    arXiv:2609.29613v1 Announce Type: cross  Abstract: We present a modular and multi-level framework for the differential diagnosis of malignant melanoma. Our framework integrates contextual information and evidence at the lesion, patient, and population levels, enabling decision-making at each level. We introduce an anatomic-site aware masked transformer, which effectively models the patient context by considering all lesions in a patient, which can be variable in count, and their site of incidence. Additionally, we incorporate patient metadata via learnable demographics embeddings to capture population statistics. Through extensive experiments, we explore the influence of specific information on the decision-making process and examine the tradeoff in metrics when considering different types of information. Validation results using the SIIM-ISIC 2020 dataset indicate including the lesion context with location and metadata improves specificity by 17.15% and 7.14%, respectively, while enha
    
[^83]: 具有不确定性感知与异构复杂度的联邦轨迹预测中的主动客户端选择

    Active Client Selection in Federated Trajectory Prediction with Uncertainty-Awareness and Heterogeneous Complexity

    [https://arxiv.org/abs/2609.29600](https://arxiv.org/abs/2609.29600)

    针对联邦轨迹预测中场景不确定性高和跨场景复杂度异构的两大挑战，本文提出了一系列融合不确定性与复杂度感知的主动客户端选择方法，以优先选取信息量大的客户端来提升联邦学习训练效果。

    

    训练诸如Transformer等序列模型如今已成为自动驾驶车辆轨迹预测的标准做法，然而由于真实世界的轨迹分散于不同区域和车辆之间，构建高质量的集中式数据集仍然十分困难。联邦学习（FL）提供了一种天然的替代方案，但面临两个独特的挑战：由轨迹或地图模糊性引起的高场景不确定性，以及由多样的地图拓扑、交通密度、智能体构成和驾驶行为导致的跨场景复杂度异构性。我们提出了一系列主动客户端选择方法，逐步引入对场景不确定性和复杂度的感知，以优先选择信息量丰富的客户端。我们的不确定性感知选择器在不确定性感知的全局目标下，使用每个客户端的负对数似然以及估计的偶然不确定性。我们进一步开发了一个联合考虑场景复杂度与不确定性的选择器……

    arXiv:2609.29600v1 Announce Type: new  Abstract: Training sequence models such as transformers is now standard for autonomous vehicle trajectory prediction, yet assembling high-quality centralized datasets remains challenging because real-world trajectories are fragmented across regions and vehicles. Federated Learning (FL) offers a natural alternative, but faces two distinctive challenges: high scene uncertainty arising from trajectory or map ambiguity, and cross-scene complexity heterogeneity caused by diverse map topology, traffic density, agent composition, and driving behaviors.   We propose a family of active client selection methods that progressively incorporate awareness of scene uncertainty and complexity to prioritize informative clients. Our uncertainty-aware selectors use per-client negative log-likelihood under an uncertainty-aware global objective and estimated aleatoric uncertainty. We further develop a selector that jointly considers scene complexity and uncertainty, m
    
[^84]: 一个从纵向文本数据中建模组织级语义身份的计算框架

    A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data

    [https://arxiv.org/abs/2609.29584](https://arxiv.org/abs/2609.29584)

    该论文提出了一个统一的计算框架，首次将组织级语义身份建模为可解释且随时间演化的语义构建，整合了语义表示学习、图语义建模、组织语义指纹、时序演化分析与证据驱动验证。

    

    组织持续产生大量文本数据，这些数据记录了组织如何随时间进行沟通、演化并形成自身差异。尽管自然语言处理的最新进展已显著提升了组织层面的文本分析能力，但现有方法主要将组织表示为用于相似度估计、分类或检索的潜在嵌入向量或预测性特征向量。因此，目前尚缺乏一个通用的计算框架，能够将组织级语义身份建模为一种可解释的、基于纵向文本证据而不断演化的语义构建。本文提出了一个计算框架，将语义表示学习、基于图的语义建模、组织级语义指纹、时序语义演化以及证据驱动的验证整合在一个统一的分析方法论之中。组织被刻画……（原文摘要在此处截断）

    arXiv:2609.29584v1 Announce Type: cross  Abstract: Organisations continuously generate large volumes of textual data that capture how they communicate, evolve and differentiate themselves over time. Although recent advances in natural language processing have substantially improved organisation-level text analytics, existing approaches primarily represent organisations as latent embeddings or predictive feature vectors for similarity estimation, classification or retrieval. Consequently, there is currently no general computational framework for modelling organisation-level semantic identity as an interpretable and evolving semantic construct derived from longitudinal textual evidence. This paper introduces a computational framework that integrates semantic representation learning, graph-based semantic modelling, organisation-level semantic fingerprints, temporal semantic evolution and evidence-driven validation within a unified analytical methodology. Organisations are characterised th
    
[^85]: 当相同行出现分歧时：从基准可辨识性到复制鲁棒的异常检测

    When Identical Rows Disagree: From Benchmark Identifiability to Replication-Robust Anomaly Detection

    [https://arxiv.org/abs/2609.29580](https://arxiv.org/abs/2609.29580)

    论文揭示了表格数据中重复行对基准评估和异常检测的系统性影响，并提出因子化检测器SCOUT，通过分离复制不变的支持证据与计数证据，实现复制鲁棒的无监督异常检测。

    

    发布的数据表通常被视为独立同分布样本，但其重复行可能编码了业务频率、重复实体、连接操作、重采样或提取错误。我们证明这种模糊性创造了一个隐藏的测量层，带来三个后果：特征相同的行造成已达到的评估上限，行加权AUROC对复制敏感，而行训练的检测器会学习到偏向多重性规模的规律。对全部690个OddBench数据集的精确行审计发现：355个存在训练-测试重叠，147个存在特征相同但标签冲突的情况，137个存在与训练正常样本完全相同的测试异常。在四种经典检测器几何结构上，从行加权切换到支持集加权会使50-61个数据集的AUROC变化至少0.05。我们提出SCOUT（支持计数正交化无监督测试），一种因子化的异常检测器，将复制不变的支持证据与暴露感知的计数证据分离，从而实现复制鲁棒的检测。

    arXiv:2609.29580v1 Announce Type: new  Abstract: A released table is often treated as an i.i.d. sample, although its repeated rows may encode business frequency, repeated entities, joins, resampling, or extraction errors. We show that this ambiguity creates a hidden measurement layer with three consequences: feature-identical rows impose an attained evaluation ceiling, row-weighted AUROC is sensitive to replication, and row-trained detectors learn a multiplicity-size-biased law. An exact-row audit of all 690 OddBench datasets finds train-test overlap in 355, feature-identical label conflict in 147, and a test anomaly identical to a training normal in 137. Switching from row to support weighting changes AUROC by at least 0.05 on 50-61 datasets across four classical detector geometries. We introduce SCOUT (Support-Count Orthogonalized Unsupervised Testing), a factorized anomaly detector that separates replication-invariant support evidence from exposure-aware count evidence. Factorwise s
    
[^86]: PartHackBench：用于部分得分工具智能体评估的认证等进度压力测试

    PartHackBench: Certified Equal-Progress Stress Tests for Partial-Credit Tool-Agent Evaluation

    [https://arxiv.org/abs/2609.29578](https://arxiv.org/abs/2609.29578)

    论文提出PartHackBench压力测试方法，通过私有认证器确保对抗轨迹与诚实轨迹在真实进度上逐组件匹配后再测量得分膨胀，从而暴露出部分得分评估中历史归因机制存在显著分数虚高且几乎无法检测攻击的缺陷。

    

    长时程工具智能体常常在不达到最终成功的情况下取得有用的进展，这促使了部分得分评估的出现。然而，评估者可能会奖励那些暂时的、后来被逆转的、或无法归因于被评估智能体的里程碑。当一条诚实轨迹与一条得分更高的对抗性轨迹进行比较时，如果后者确实取得了更多真实进展，则这种比较是不确定的。我们提出了PartHackBench，这是一种消除该混淆因素的受控方法。一个私有认证器只有在两条轨迹在当前状态谓词满足情况和标准化智能体归因方面逐组件完全匹配时，才接受该轨迹对；得分膨胀（定义为f(A) - f(H)）仅在之后进行测量。在PB-CSTE的18个密封保留任务中，冻结的历史目标运行为15个任务生成了匹配的对抗样本。历史归因产生了平均0.252的得分膨胀，条件攻击成功率为10/15，端到端收益为10/18，且未检测到14个严格回滚中的任何一个。

    arXiv:2609.29578v1 Announce Type: new  Abstract: Long-horizon tool agents often make useful progress without reaching terminal success, motivating partial-credit evaluation. Yet evaluators may reward milestones that were temporary, later reversed, or not attributable to the evaluated agent. Comparing an honest trajectory with a higher-scoring adversarial one is inconclusive if the latter made more genuine progress. We introduce PartHackBench, a controlled methodology that removes this confound. A private certifier admits a pair only when its trajectories match component-wise in both current-state predicate satisfaction and standardized agent attribution; score inflation, defined as f(A) - f(H), is measured only afterward. In 18 sealed held-out tasks in PB-CSTE, the frozen historical-target run produced matched adversaries for 15 tasks. Historical credit yielded mean inflation of .252, conditional attack success of 10/15, end-to-end yield of 10/18, and detected none of 14 strict rollbac
    
[^87]: 任务分辨的费希尔谱分析用于量子储层计算

    Task-Resolved Fisher Spectroscopy for Quantum Reservoir Computing

    [https://arxiv.org/abs/2609.29570](https://arxiv.org/abs/2609.29570)

    该论文提出“任务分辨的费希尔谱分析”方法，通过费希尔信息层级结构精确定位量子储层计算中任务信息在储层状态、测量、特征压缩和有限采样各环节的损失位置。

    

    量子储层计算（QRC）利用固定的量子动力学对时间序列进行编码，仅训练经典读出层，但仅凭基准容量无法揭示任务信息是在储层、测量、特征压缩还是有限采样中丢失的。我们引入了“任务分辨的费希尔谱分析”，其中预测目标在输入历史的平稳分布上定义了一组正交的得分坐标。沿着这些得分坐标对带标签的历史进行重新加权，可以生成一个精确仿射的储层状态和测量结果族。在相同的坐标系中，我们得到了一个多体费希尔信息层级结构，它关联了状态量子费希尔信息、完整测量记录的费希尔信息，以及保留到多体阶数 $r$ 的矩矩阵。每个矩矩阵的二次型恰好是最优线性读出的平稳容量，而有限测量……（原文摘要在此处被截断）

    arXiv:2609.29570v1 Announce Type: cross  Abstract: Quantum reservoir computing (QRC) uses fixed quantum dynamics to encode a time series and trains only a classical readout, but a benchmark capacity alone does not reveal whether task information is lost in the reservoir, the measurement, feature compression, or finite sampling. We introduce \emph{task-resolved Fisher spectroscopy}, in which prediction targets define orthonormal score coordinates on the stationary distribution of input histories. Reweighting labeled histories along these scores generates an exactly affine family of reservoir states and measurement outcomes. In the same coordinates, we obtain a many-body Fisher-information hierarchy relating the state quantum Fisher information, the Fisher information of the complete measurement record, and moment matrices retained through many-body order $r$. The quadratic form of each moment matrix is exactly the stationary capacity of the optimal linear readout, while a finite-measure
    
[^88]: 伪标签在半监督安卓恶意软件归因中依赖于分类器的收益

    Classifier-Dependent Benefits of Pseudo-Labeling for Semi-Supervised Android Malware Attribution

    [https://arxiv.org/abs/2609.29564](https://arxiv.org/abs/2609.29564)

    本研究在CICMalDroid 2020数据集上对六种分类器进行了伪标签半监督学习的系统性评估与统计检验，首次揭示其收益强烈依赖于分类器类型：SVM获益最大（+4.4%准确率），而随机森林在低标注比例下反而显著受损。

    

    由于高特征维度、类别不平衡以及专家标注数据的高成本，安卓恶意软件家族的检测与分类仍然具有挑战性。半监督学习（SSL）提供了一种利用未标注样本的方法，但先前的工作很少测试SSL的收益是否能在不同类型的分类器之间推广，也很少报告统计显著性。我们在CICMalDroid 2020数据集上对六种分类器（LightGBM、XGBoost、随机森林、逻辑回归、MLP和SVM）的伪标签方法进行了系统性评估，采用五折分层交叉验证，并在五个标注比例（1-20%）下进行配对t检验。我们发现SSL收益强烈依赖于分类器：SVM获得最大的显著提升（在5%标注比例下准确率提高4.4%，p = 0.0028），LightGBM有适度提升（在2-5%标注比例下提高0.8至1.3%），而随机森林在低标注比例下受到显著损害（在1%标注比例下下降3.1%）。逐类分析揭示……

    arXiv:2609.29564v1 Announce Type: new  Abstract: Detecting and classifying Android malware families remains challenging due to high feature dimensionality, class imbalance, and the high cost of expert-labeled data. Semi-supervised learning (SSL) offers a way to leverage unlabeled samples, but prior works rarely test whether SSL benefits generalize across classifier types or report statistical significance. We present a systematic evaluation of pseudo-labeling across six classifiers (LightGBM, XGBoost, Random Forest, Logistic Regression, MLP, and SVM) on the CICMalDroid 2020 dataset, using five-fold stratified cross-validation and paired t-tests across five labeled ratios (1-20%). We find that SSL benefit is strongly classifier-dependent: SVM shows the largest significant gain (+4.4% accuracy at 5% labels, p = 0.0028), LightGBM improves modestly (+0.8 to +1.3% at 2-5% labels), while Random Forest is significantly harmed at low label ratios (-3.1% at 1% labels). Per-class analysis reveal
    
[^89]: 强化学习中成本感知的语言模型引导的认证预测性建议价值门控

    Certified Predictive Value-of-Advice Gating for Cost-Aware Language-Model Guidance in Reinforcement Learning

    [https://arxiv.org/abs/2609.29548](https://arxiv.org/abs/2609.29548)

    该论文提出了一种认证预测性建议价值门控框架，在调用语言模型前预测可能的响应并评估其价值，仅当预测价值的置信下界超过调用成本时才发起查询，并辅以动作特定证书管理执行，从而实现成本感知且带理论保证的强化学习引导。

    

    语言模型建议可以加速强化学习，但调用成本高昂，且返回的动作可能过时或错误。我们将建议获取形式化为一个响应条件化的元推理问题：在查询之前，控制器预测可能的解析响应，评估每个响应将产生的决策和声明的后续行动，仅当预测价值的置信下界超过定价成本时才进行查询。执行则由动作特定的证书单独管理。在明确的假设下，认证建议是近似最优的，被包装的学习器仅在干预稳定性条件下继承回退遗憾，且保守分配相对于短视预言机最多损失声明的查询价值估计误差。在BabyAI上，采用Qwen2.5-1.5B和7B作为建议者的代理校准控制器，在20个种子上相比不查询将GoToObj回报提高了0.029 ± 0.016和0.030 ± 0.015，同时减少……（摘要截断）

    arXiv:2609.29548v1 Announce Type: new  Abstract: Language-model advice can accelerate reinforcement learning, but calls are costly and returned actions may be stale or wrong. We formulate advice acquisition as a response-contingent metareasoning problem: before querying, the controller predicts possible parsed responses, evaluates the decision and declared continuation that would follow each response, and queries only when a lower confidence bound on predictive value exceeds the priced cost. Execution is governed separately by an action-specific certificate. Under explicit assumptions, certified advice is near-optimal, a wrapped learner inherits fallback regret only under intervention stability, and conservative allocation loses at most the declared query-value estimation error relative to a myopic oracle. On BabyAI, a proxy-calibrated controller with Qwen2.5-1.5B and 7B advisors improves GoToObj return over no querying by 0.029 +/- 0.016 and 0.030 +/- 0.015 across 20 seeds while reduc
    
[^90]: 广义图变分自编码器：有界散度控制后验坍缩

    Generalized Graph Variational Autoencoders: Bounded Divergences Control Posterior Collapse

    [https://arxiv.org/abs/2609.29546](https://arxiv.org/abs/2609.29546)

    本文提出广义图变分自编码器（GGVA），用Rényi-Tsallis散度族替代KL散度，并揭示散度的有界性（而非其阶数）才是控制后验坍缩的关键性质。

    

    变分图自编码器（VGAE）使用Kullback-Leibler散度将其后验分布向先验分布进行正则化，这一选择继承自变分自编码器而非经过论证。我们提出了广义图变分自编码器（GGVA），它将该正则项替换为Rényi-Tsallis散度族中任意阶数q的成员，而保持模型的其他部分不变。这两种散度对于对角高斯分布都具有闭式解，并且在q→1时都精确恢复为KL散度，因此VGAE是我们自己模型在q=1时的一个分支而非独立的基线，任何测得的差异都可归因于单个标量。我们的分析表明，有界性而非阶数才是起作用的性质：当q<1时，Tsallis散度的上界为1/(1-q)，且与潜变量维度无关，而相同阶数的KL散度和Rényi散度都是无界的。在跨越三个合成族的十个图上，

    arXiv:2609.29546v1 Announce Type: cross  Abstract: The variational graph autoencoder (VGAE) regularizes its posterior toward the prior with the Kullback-Leibler divergence, a choice inherited from the variational autoencoder rather than argued for. We introduce the generalized graph variational autoencoder (GGVA), which replaces that term with any member of the R\'enyi-Tsallis family of order $q$ while leaving every other part of the model untouched. Both members admit closed forms for diagonal Gaussians and both recover the KL exactly as $q \to 1$, so the VGAE is the $q=1$ arm of our own model rather than a separate baseline, and any measured difference is attributable to a single scalar. Our analysis identifies boundedness, not the order, as the operative property: for $q<1$ the Tsallis divergence is bounded above by $1/(1-q)$, independently of the latent width, whereas the KL and the R\'enyi divergence of the same order are unbounded. On ten graphs spanning three synthetic families,
    
[^91]: 时间序列验证的不可能三难：训练充分性、测试覆盖度与时间因果性之间的守恒定律

    The Impossible Trinity of Time-Series Validation: A Conservation Law among Training Sufficiency, Test Coverage, and Temporal Causality

    [https://arxiv.org/abs/2609.29530](https://arxiv.org/abs/2609.29530)

    本文证明时间序列验证存在“不可能三难”——训练充分性、测试覆盖度与时间因果性无法同时满足，并给出守恒律不等式 α+β ≤ 1+Λ，量化了跨越因果边界所必须付出的数据泄漏偏差代价。

    

    在时间序列上验证模型需要同时满足三个条件：每次训练应使用样本的大部分（充分性）、各测试集应共同覆盖样本的大部分（覆盖度）、且训练数据应先于测试数据（因果性）。我们证明这三者无法同时兼得，并为每一项“定价”。设 α 为各折中最小的训练比例，β 为测试所覆盖样本的比例，Λ 为样本中来自某测试点未来、却被用作训练数据的比例，δ 为从测试点到其未来中最近训练点的距离。在长度为 T 的样本上，任何验证方案都满足 α+β ≤ 1+Λ 与 α+min{β, δ/T} ≤ 1，并且在 β-混合（β-mixing）条件下，某测试点处的数据泄漏偏差至多为 2Mβ_mix(δ)。换言之：要越过因果边界 α+β=1，就必须在测试点的未来数据上进行训练……（摘要原文在此处截断）

    arXiv:2609.29530v1 Announce Type: new  Abstract: Validating a model on a time series asks for three things at once: each training run should use most of the sample (sufficiency), the test sets should together cover most of the sample (coverage), and training data should come before test data (causality). We prove that the three cannot be had together and price each one. Let $\alpha$ be the smallest training fraction over folds, $\beta$ the fraction of the sample covered by tests, $\Lambda$ the fraction of the sample used as training data from the future of a test point, and $\delta$ the distance from a test point to the nearest training point in its future. Every scheme on a sample of length $T$ satisfies $\alpha+\beta \le 1+\Lambda$ and $\alpha+\min\{\beta,\delta/T\} \le 1$, and under $\beta$-mixing the leakage bias at a test point is at most $2M\beta_{\mathrm{mix}}(\delta)$. In words: going beyond the causal frontier $\alpha+\beta=1$ requires training on the future; that future data 
    
[^92]: 来自主动式语音代理蜜罐的真实诈骗与骚扰电话对话语料库

    A Corpus of Real Scam- and Spam-Call Conversations from an Active Voice-Agent Honeypot

    [https://arxiv.org/abs/2609.29528](https://arxiv.org/abs/2609.29528)

    本文通过主动式语音代理蜜罐在53天内收集了10,015通真实诈骗与骚扰电话对话（约895小时音频、328,869条转录轮次），为电话诈骗研究提供了极为稀缺的真实对话数据集。

    

    诈骗者与其目标之间的真实对话是研究电话诈骗最有价值的信息载体之一，但也最为稀缺：被动式蜜罐捕获的绝大多数是自动语音消息和挂断电话，大规模研究通常仅刻画电话元数据而非对话内容，而人工诱骗方式又难以规模化。我们展示了一个由主动式语音代理蜜罐收集的真实诈骗电话对话数据集。专用电话号码被投放到诈骗团伙获取线索的渠道中；来电由一个低延迟对话代理接听，该代理扮演可信的目标人设并维持互动，同时每通电话都被录音、转录并自动标注。在最初的53天窗口期内，我们捕获了10,015通入站诈骗和骚扰电话（其中6,601通包含两轮或以上对话）：约895小时的音频，以及来自5,665个不同主叫号码的328,869条转录对话轮次。

    arXiv:2609.29528v1 Announce Type: cross  Abstract: Real conversations between fraudsters and their targets are among the most informative artifacts for studying telephone scams, yet also the scarcest: passive honeypots overwhelmingly capture automated messages and hang-ups, large-scale studies characterize call metadata rather than dialogue, and manual scam-baiting does not scale. We present a dataset of real scam-call conversations collected by an active voice-agent honeypot. Dedicated numbers are seeded into the lead-generation channels fraud operations harvest; inbound callers are answered by a low-latency conversational agent that adopts a plausible target persona and sustains the interaction while every call is recorded, transcribed, and automatically labeled. Over an initial 53-day window we captured 10,015 inbound scam and spam calls (6,601 with two or more turns): roughly 895 hours of audio and 328,869 transcribed turns from 5,665 distinct originating numbers. Under a holistic 
    
[^93]: 布朗核梯的公共协方差几何与认证

    Common Covariance Geometry and Certification for Brownian Kernel Ladders

    [https://arxiv.org/abs/2609.29525](https://arxiv.org/abs/2609.29525)

    该论文提出了支配布朗核梯经验椭球并集的最小迹公共协方差概念，借助绝对二可和算子、阈值与有效电阻几何等工具给出其精确刻画，并据此导出通用高斯复杂度界、确定性深度定律以及精确的经验柯尔莫哥洛夫宽度公式。

    

    一种表示自适应的核类在固定样本上产生的是再生核希尔伯特空间椭球体的并集，而非单个椭球体。我们引入了支配由布朗核梯所生成的无限制经验并集的最小迹公共协方差，并发展其在统计、逼近论与计算方面的推论。该协方差值可通过绝对二可和算子与协方差支配乘子得到精确表述，并由此导出一个通用的高斯复杂度界。一个封闭的末层狄拉克迹约化和一个带符号布朗阈值表示，将一般性的协方差问题转化为阈值、图余面积与有效电阻几何问题。这些工具给出了确定性的深度定律、条件高斯反演、随机设计与摄动转移，以及一个精确的经验柯尔莫哥洛夫宽度公式，其主导协方差特征子空间可近似……（摘要原文在此处截断）

    arXiv:2609.29525v1 Announce Type: new  Abstract: A representation-adaptive kernel class produces, on a fixed sample, a union of reproducing-kernel Hilbert-space ellipsoids rather than one ellipsoid. We introduce the minimum-trace common covariance that dominates the unrestricted empirical union generated by Brownian kernel ladders and develop its statistical, approximation-theoretic, and computational consequences. The covariance value admits exact formulations through absolutely two-summing operators and covariance-dominated multipliers, and it yields a universal Gaussian-complexity bound. A closed last-layer Dirac-trace reduction and a signed Brownian threshold representation convert the generic covariance problem into threshold, graph-coarea, and effective-resistance geometry. These tools give deterministic depth laws, conditional Gaussian reverses, random-design and perturbation transfers, and an exact empirical Kolmogorov-width formula whose leading covariance eigenspaces approxim
    
[^94]: 多任务学习中样本加权的端到端迹范数几何

    Sample-Weighted End-to-End Trace-Norm Geometry for Multitask Learning

    [https://arxiv.org/abs/2609.29520](https://arxiv.org/abs/2609.29520)

    该论文提出用从任务系数到输入空间预测器的端到端映射的样本加权迹范数来分析多任务学习泛化性能，推导出精确的经验Rademacher复杂度，并证明传统分离乘积界存在无界的方向间隙、因子分解间隙以及指数级深度间隙。

    

    多任务模型将共享表示与任务特定输出相结合，但泛化界通常分别控制这两个组成部分。这种乘积形式可能丢弃相对方向与抵消效应，并且在所表示的预测器保持不变的情况下，当中间坐标进行等价变换时其值也可能发生改变。我们转而研究从任务系数到输入空间预测器的端到端映射的样本量加权迹范数。对于其固定半径的函数类，我们推导出了精确的经验Rademacher复杂度。同一量可以通过在表示作用后消除正定任务协方差矩阵来刻画，并且在有限维中间空间中，可以通过在所有等价可逆重构上优化分离乘积来刻画。显式构造表明存在无界的方向间隙和因子分解间隙，以及抵消线性层之间的指数级深度间隙。

    arXiv:2609.29520v1 Announce Type: new  Abstract: Multitask models combine a shared representation with task-specific outputs, but generalization bounds often control the two components separately. Such products can discard relative orientation and cancellation and can change under equivalent transformations of intermediate coordinates even when the represented predictors are unchanged. We study instead the sample-size-weighted trace norm of the end-to-end map from task coefficients to input-space predictors. For its fixed-radius class, we derive the exact empirical Rademacher complexity. The same quantity is characterized by eliminating a positive-definite task covariance after the representation acts and, in finite-dimensional intermediate spaces, by optimizing the separated product over all equivalent invertible refactorizations. Explicit constructions show unbounded orientation and factorization gaps and an exponential depth gap for cancelling linear layers. As a geometric applicati
    
[^95]: CataOPD：面向大语言模型推理的催化式同策略蒸馏

    CataOPD: Catalytic On-Policy Distillation for Large Language Model Reasoning

    [https://arxiv.org/abs/2609.29518](https://arxiv.org/abs/2609.29518)

    CataOPD提出让教师模型充当“催化剂”而非目标的同策略蒸馏方法，通过自救援路由和催化引导自解析扩展推理轨迹的可达性，并将经验证的学生生成轨迹内化为无需催化剂的最终策略。

    

    强化学习（RL）和同策略蒸馏（OPD）是提升大语言模型推理能力的两种代表性范式。然而，当没有采样到正确的轨迹时，RL缺乏正向的正确性信号，而OPD则始终受限于学生在同策略分布下可达的推理轨迹。因此，我们提出了CataOPD，其中教师模型充当催化剂而非目标，在扩展可达性的同时，将经验证的学生生成轨迹内化到无需催化剂的策略中。自救援路由（Self-Rescue Routing）将经验上全部失败的组用作路由信号而非教师干预的触发器，首先通过额外的同策略自采样来寻找正确轨迹。对于自救援后仍未解决的问题，催化引导自解析使用催化引导在引导的学生分布中引出经验证的学生生成轨迹。障碍加权内化

    arXiv:2609.29518v1 Announce Type: new  Abstract: Reinforcement learning (RL) and on-policy distillation (OPD) are two representative paradigms for improving large language model reasoning. However, when no correct trajectory is sampled, RL lacks a positive correctness signal, while OPD remains constrained by the reasoning trajectories reachable under the student's on-policy distribution. Therefore, we propose CataOPD, where the teacher acts as a catalyst rather than a target, expanding reachability while internalizing verified student-produced trajectories into a catalyst-free policy. Self-Rescue Routing uses empirically all-failed groups as routing signals rather than teacher-intervention triggers, first seeking correct trajectories through additional on-policy self-sampling. For problems unresolved after self-rescue, Catalytic-Guided Self-Resolution uses catalytic guidance to elicit a verified student-produced trajectory in the guided student distribution. Barrier-Weighted Internaliz
    
[^96]: 大语言模型智能体多轮一致性评估：生存分析与失败理由分类法

    Evaluation of Multi-Turn Consistency in LLM Agents: Survival Analysis and Failure-Rationale Taxonomy

    [https://arxiv.org/abs/2609.29508](https://arxiv.org/abs/2609.29508)

    该论文在受延迟满足启发的20步多智能体环境中，利用Kaplan-Meier生存曲线和离散时间风险回归对8个模型家族的84,540条轨迹进行时间一致性评估，并从13,780条深思轨迹中构建了七类失败理由分类法，系统揭示了LLM智能体在多轮交互中的失败风险及其原因。

    

    大语言模型（LLM）智能体在孤立任务上可能表现良好，但在长时间的交互过程中却会逐渐陷入不一致性。我们在一个受延迟满足研究启发的可控20步多智能体环境中评估时间一致性。在每一步中，智能体需在继续延迟获取奖励或立即领取奖励（终止回合）之间做出选择。通过对社交可见性（私密 vs 公开）、人格压力源和深思策略进行全因子操纵，我们运行了涵盖8个模型家族的84,540条轨迹。我们将首次领取奖励视为“事件发生时间”结果，估计了Kaplan-Meier生存曲线并拟合离散时间风险回归，以量化各实验因素如何随时间改变失败风险。随后，为了分析与失败相关的理由和语言模式，我们从选择终止回合的智能体的13,780条深思轨迹中构建了一个七类别的分类法，使用一种……

    arXiv:2609.29508v1 Announce Type: new  Abstract: Large language model (LLM) agents may perform well on isolated tasks yet drift into inconsistency over extended interaction. We evaluate temporal consistency in a controlled 20-step multi-agent setting inspired by delayed-gratification studies. At each step, an agent chooses between continuing to delay a reward or claiming it immediately (terminating the episode). Across a full-factorial manipulation of social visibility (private vs public), persona stressors, and deliberation policy, we run 84,540 trajectories spanning 8 model families. Treating the first reward-claim as a time-to-event outcome, we estimate Kaplan-Meier survival curves and fit discrete-time hazard regression to quantify how experimental factors shift failure risk over time. Then, to analyze rationales and language patterns associated with failure, we build a seven-category taxonomy from 13,780 deliberation traces from agents who choose to terminate the episode, using an
    
[^97]: 谱引导扩散：通过静态谱层调度加速推理

    Spectral-Guided Diffusion: Accelerating Inference via Static Spectral Layer Scheduling

    [https://arxiv.org/abs/2609.29505](https://arxiv.org/abs/2609.29505)

    该论文提出基于谱集中比率（SCR）与Frobenius范数的静态谱层调度方法，仅利用预训练权重离线确定扩散模型中可冻结并复用缓存的残差分支，无需路由器或输入依赖搜索，即可在相同计算预算下加速推理并保持生成质量。

    

    扩散模型推理过程中需要反复评估同一个大型网络。我们探究是否仅凭预训练权重就能识别出在整个生成轨迹中无需重复计算的残差分支。我们提出的谱集中比率（SCR）用于衡量奇异值能量在主导奇异值与尾部奇异值之间的分布情况。结合Frobenius范数，它为每个被调度的单元提供了离线敏感性代理指标和确定性的生命周期。被冻结的单元会复用其缓存的残差分支更新，而当前的残差流和所有外部条件继续传播。该方法无需路由器、校准提示词或依赖输入的搜索。在相同的层-步计算预算下，SCR/Frobenius调度在LLaDA-8B、DiT-XL/2、U-ViT-L和SDXL上的质量保持效果优于随机调度、深度调度、范数调度、稳定秩调度以及Frobenius-稳定秩调度。更广泛的LLaDA测试涵盖检索、推理、代码、摘要和开放式生成任务；匹配视野的对照实验保留了……

    arXiv:2609.29505v1 Announce Type: new  Abstract: Diffusion inference repeatedly evaluates the same large network. We ask whether pretrained weights alone can identify residual branches that need not be recomputed throughout the trajectory. Our \textbf{Spectral Concentration Ratio (SCR)} measures leading-versus-tail singular-value energy. Combined with Frobenius magnitude, it yields an offline sensitivity proxy and a deterministic lifetime for each scheduled unit. A frozen unit reuses its cached residual-branch update while the current residual stream and all external conditioning continue to propagate. The method needs no router, calibration prompts, or input-dependent search. At matched layer-step budgets, SCR/Frobenius preserves quality better than random, depth, norm, stable-rank, and Frobenius--stable-rank schedules on LLaDA-8B, DiT-XL/2, U-ViT-L, and SDXL. Broader LLaDA tests cover retrieval, reasoning, code, summarization, and open-ended generation; matched-horizon controls retai
    
[^98]: 任务感知谱剪枝：面向高效大语言模型推理的掩码混合框架

    Task-Aware Spectral Pruning: A Mixture-of-Masks Framework for Efficient LLM Inference

    [https://arxiv.org/abs/2609.29499](https://arxiv.org/abs/2609.29499)

    提出TASP框架，通过将模块谱描述符与任务消融效应校准、为每个用户请求路由固定的稀疏掩码，在Llama-3-70B上实现43%有效FLOP削减的同时保持约97.7%的性能，且先导实验表明其适用性因模型而异。

    

    静态剪枝对所有提示都施加同一种稀疏结构，然而推理、检索、生成、编程和翻译等任务可能依赖语言模型的不同部分。我们提出任务感知谱剪枝，这是一种训练后框架，它将模块级谱描述符与实测的任务特定消融效应进行校准，在稀疏掩码构建过程中关闭分组查询注意力和SwiGLU的依赖关系，并将每个用户轮次路由到一个在预填充与解码全程保持固定的已编译掩码。一个模块不相交的先导实验会在完整校准之前先判断谱信号是否具有信息量。在所述的回顾性运行规则下，该先导实验在受评估的Llama-3-8B和Llama-3-70B检查点上通过，但拒绝了Qwen2.5-1.5B，这表明该方法的适用性依赖于具体模型而非普遍适用。在43%的有效FLOP削减下，Llama-3-70B基准测试仍保持了97.7（摘要原文在此处截断）

    arXiv:2609.29499v1 Announce Type: new  Abstract: Static pruning imposes one sparse structure on every prompt, even though reasoning, retrieval, generation, coding, and translation can depend on different parts of a language model. We introduce Task-Aware Spectral Pruning (TASP), a post-training framework that calibrates module-level spectral descriptors against measured task-specific ablation effects, closes grouped-query-attention and SwiGLU dependencies during sparse-mask construction, and routes each user turn to one compiled mask that remains fixed throughout prefill and decoding. A module-disjoint pilot first determines whether the spectral signal is informative before full calibration. Under the stated retrospective operating rule, the pilot passes on the evaluated Llama-3-8B and Llama-3-70B checkpoints but rejects Qwen2.5-1.5B, demonstrating that applicability is model-dependent rather than universal. At a 43% active-FLOP reduction, the Llama-3-70B benchmark harness retains 97.7
    
[^99]: BLADE：面向校准知识图谱补全的蒸馏LLM正则化方法

    BLADE: Distilled LLM Regularization for Calibrated Knowledge Graph Completion

    [https://arxiv.org/abs/2609.29487](https://arxiv.org/abs/2609.29487)

    BLADE通过将离线LLM判断蒸馏为冻结教师正则化器的变分模型，在推理时无需LLM参与，即可同时提供校准的预测概率和认知不确定性估计，并将校准误差大幅降低。

    

    知识图谱补全模型通常以排序为优化目标，然而许多下游应用需要经过校准的概率。我们提出BLADE，这是一个变分模型，它将潜在真值与图谱记录过程分离，并将离线的语言模型判断蒸馏到一个冻结的教师正则化器中。在推理阶段LLM完全不参与。后验样本提供预测概率和认知不确定性，而紧凑的教师模型仅作为可选的分诊因子保留。在五个基准测试上，BLADE在统一的排序协议下保持竞争力，其自适应ECE相对于深度集成平均降低了60.1%，相对于温度缩放的RotatE降低了78.1%。在相同的FB15k-237候选集上，BLADE相较于基于验证集选择的直方图分箱方法以及匹配的生成式ComplEx2模型，在ECE、Brier分数和NLL上均有改进，且这些改进在预设的近似候选池上依然保持。

    arXiv:2609.29487v1 Announce Type: new  Abstract: Knowledge graph completion models optimize ranking, although many downstream applications require calibrated probabilities. We present BLADE, a variational model that separates latent truth from graph recording and distills offline language-model judgments into a frozen teacher regularizer. The LLM is absent during inference. Posterior samples provide predictive probabilities and epistemic uncertainty, while the compact teacher remains available only as an optional triage factor. Across five benchmarks, BLADE remains competitive under a common ranking protocol and reduces adaptive ECE by a macro-average of 60.1% relative to deep ensembles and 78.1% relative to temperature-scaled RotatE. On identical FB15k-237 candidate sets, BLADE also improves ECE, Brier score, and NLL over validation-selected histogram binning and a matched generative ComplEx2 model, with these improvements persisting on a prespecified near-miss pool. Under controlled 
    
[^100]: 直接消息近似（DMA）：一种基于一致性的因子图可驾驭近似推断框架

    Direct Message Approximation (DMA): A Consistency-Based Framework for Tractable Approximate Inference on Factor Graphs

    [https://arxiv.org/abs/2609.29466](https://arxiv.org/abs/2609.29466)

    该论文提出直接消息近似（DMA），通过直接近似因子到变量的消息而非边缘分布，并借助一致性条件与主定理，实现了无需内循环迭代、避免负精度消息且误差可控的因子图近似推断。

    

    因子图上的近似消息传递是两大主流概率推断算法族的基础：期望传播（EP）和变分消息传递（VMP）。这两种方法都在每个因子边上近似边缘分布，这迫使算法采用迭代的轮询调度，存在产生负精度消息的风险，且对于VMP而言，在Dirac-delta因子处会退化为点估计。我们提出直接消息近似（DMA），它直接近似因子到变量的消息，而非边缘分布。对于可归一化的因子，我们定义了一个一致性条件（要求当所有其他传入消息均为Dirac delta时达到精确结果）来指导消息的构造。我们证明了一个主定理（针对正规消息、任意图结构），利用消息KL散度约束边缘KL散度，并由其导出三个结构性推论：Dirac输入一致性、无需EP式的内循环迭代、以及不会产生负精度消息。此外，我们还证明了一个互补的 O(1/r^...（摘要在此处被截断）

    arXiv:2609.29466v1 Announce Type: cross  Abstract: Approximate message passing on factor graphs underlies two dominant families of probabilistic inference algorithms: expectation propagation (EP) and variational message passing (VMP). Both methods approximate the marginal at each factor edge, forcing an iterative round-robin schedule, risking negative-precision messages, and, for VMP, collapsing to point estimates at Dirac-delta factors. We introduce Direct Message Approximation (DMA), which approximates factor-to-variable messages directly rather than the marginal. For normalisable factors, we define a consistency condition (requiring exactness when all other incoming messages are Dirac deltas) to guide message construction. We prove a master theorem (proper messages, any graph) bounding marginal KL from message KL, with three structural corollaries: Dirac-input consistency, no EP-style inner-loop iteration, and no negative-precision messages. Further, we prove a complementary $O(1/r^
    
[^101]: 裁剪随机梯度下降（Clipped SGD）的精确收敛速度

    Precise Convergence Speed of Clipped SGD

    [https://arxiv.org/abs/2609.29458](https://arxiv.org/abs/2609.29458)

    本文对裁剪SGD在 $(L_0,L_1)$-光滑函数上给出了更紧致的收敛分析，将步长有效范围扩大到 $\eta < 1/\beta$，加强了收敛准则并将最终可达损失界降低至更精确的 $6 \min(\sigma^2/c, 3\sigma)$。

    

    我们对裁剪梯度下降在 $(L_0, L_1)$-光滑函数上的收敛性给出了更紧致的分析，并提供了量化的常数。基于 Koloskova 等人（2023）的思想，我们重构了若干情形的划分，揭示了由 $\ell_2$-投影基本性质导出的偏差控制的核心作用，从而简化了证明。我们还将步长的有效范围从 $\eta \leq 1 / (9 \beta)$ 扩展到 $\eta < 1 /\beta$，其中 $\beta = L_0 + c L_1$，$c$ 为裁剪常数，这与更传统的光滑函数分析相匹配。我们将收敛准则从 $\left( \min_{t < T} \mathbb{E}[\lVert \nabla f(x_t) \rVert_2] \right)$ 加强为 $\left( \frac{1}{T} \sum_{t < T} \mathbb{E}[\lVert \nabla f(x_t) \rVert_2] \right)$ 并保持匹配的收敛速度，同时将最终可达损失从 $\mathcal{O}(\min(\sigma^2/c, \sigma))$ 降低到更精确的 $6 \min(\sigma^2 /c, 3 \sigma)$。

    arXiv:2609.29458v1 Announce Type: new  Abstract: We present a tightened convergence analysis of clipped gradient descent on $(L_0, L_1)$-smooth functions, with quantitative constants. Building on the ideas of Koloskova et al (2023), we refactor several case disjunctions to reveal the central role of a control of the bias derived from fundamental properties of $\ell_2$-projection, simplifying proofs. We also extend the domain of validity from $\eta \leq 1 / (9 \beta)$ to $\eta < 1 /\beta$ where $\beta = L_0 + c L_1$ for clipping constant $c$, which matches the more traditional analysis of smooth functions. We strengthen the convergence criterion from $\left( \min_{t < T} \mathbb{E}[\lVert \nabla f(x_t) \rVert_2] \right)$ to $\left( \frac{1}{T} \sum_{t < T} \mathbb{E}[\lVert \nabla f(x_t) \rVert_2] \right)$ with matching speed, and lower the final achievable loss from $\mathcal{O}(\min(\sigma^2/c, \sigma))$ to the more precise $6 \min(\sigma^2 /c, 3 \sigma)$.
    
[^102]: 噪声评分下面向隐私与稳定性的组合推荐中学习与选择解耦方法

    Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores

    [https://arxiv.org/abs/2609.29453](https://arxiv.org/abs/2609.29453)

    该论文将组合推荐解耦为随机评分学习与确定性选择两个阶段，阐明了差分隐私保证经选择器端到端传递的成立条件，并提出了一种能在评分噪声扰动下保证推荐列表顺序不变的间隔证书。

    

    我们将组合推荐形式化为一个随机化的评分学习器之后接一个确定性的选择器。首先，通过后处理机制，适当界定范围的差分隐私保证可以传递到选择过程及其审计轨迹中。端到端的隐私保证仅在以下情况下成立：选择器的输入为公开或独立的信息、为先前已输出的私有数据、或已单独进行过隐私核算；若固定原始状态或候选信息，则只能获得条件性的隐私保证。其次，我们推导出一个基于日志的间隔证书：当评分扰动引起的目标函数变动为有界且小于最小贪心决策间隔的一半时，即可保证有序的组合推荐列表保持不变。受控的固定间隔测试显示出近线性的指数缩放特性，经验斜率为-0.220（95%置信区间为[-0.231, -0.210]），对比独立噪声下的参考斜率-1/4。在OULAD、MovieLens-25M和Amazon Musical Instruments数据集上的真实锚点实验表明，更大的锚点权重能够降低由评分噪声引起的……

    arXiv:2609.29453v1 Announce Type: new  Abstract: We formalize slate recommendation as a randomized score learner followed by deterministic selection. First, an appropriately scoped differential-privacy guarantee passes through selection and its audit trace by post-processing. End-to-end privacy holds only when selector inputs are public or independent, previous private outputs, or separately privacy-accounted; fixing raw state or candidate information instead yields only a conditional guarantee. Second, we derive a logged margin certificate: bounded score-induced objective movement below half the smallest greedy decision margin guarantees that the ordered slate is unchanged.   Controlled fixed-margin tests show near-linear exponent scaling, with an empirical slope of $-0.220$ (95% CI $[-0.231,-0.210]$) against the independent-noise reference $-1/4$. Real-anchor experiments on OULAD, MovieLens-25M, and Amazon Musical Instruments show that greater anchor weight reduces score-noise-induce
    
[^103]: SPADE-DFL：基于无导数线性化ADMM的通信高效去中心化联邦学习

    SPADE-DFL: Communication-Efficient Decentralized Federated Learning via Derivative-Free Linearized ADMM

    [https://arxiv.org/abs/2609.29446](https://arxiv.org/abs/2609.29446)

    本文提出SPADE-DFL方法，通过无导数线性化ADMM让本地更新次数随计算预算增长，在去中心化联邦学习中仅用Θ(T^{2/3})轮通信即达到O(T^{-1/3})的收敛界，并同时实现了客户端级差分隐私保护。

    

    在无导数去中心化学习中减少通信量，需要控制在多次本地更新过程中累积的客户端间分歧。本文提出了SPADE-DFL，这是一种原始-对偶方法，允许邻居交换之间的本地函数值更新次数随计算预算增长，同时保持非私有情形下的收敛阶。对于在统一查询矩界约束下的光滑非凸目标函数，所规定的非私有调度方案仅使用Θ(T^{2/3})轮通信即可达到O(T^{-1/3})的时间平均平稳性与共识界，其中T为每个客户端的本地更新次数。对于私有训练，该方法将累积的数据相关增量与图校正相隔离，使得每个客户端每轮仅需维护一个受保护状态即可生成所有外发消息。论文为完整的交互记录证明了客户端级差分隐私，并量化了由此产生的性能影响。

    arXiv:2609.29446v1 Announce Type: new  Abstract: Reducing communication in derivative-free decentralized learning requires controlling the disagreement accumulated over multiple local updates. This paper develops SPADE-DFL, a primal--dual method that allows the number of local function-value updates between neighbor exchanges to grow with the computation budget while preserving the nonprivate convergence order. For smooth nonconvex objectives under uniform query-moment bounds, the prescribed nonprivate schedule achieves a time-averaged stationarity and consensus bound of $\mathcal{O}(T^{-1/3})$ using only $\Theta(T^{2/3})$ communication rounds, where $T$ is the number of local updates per client. For private training, the accumulated data-dependent increment is isolated from the graph correction, allowing one protected state per client and round to generate all outgoing messages. We prove client-level differential privacy for the full interactive transcript and quantify the resulting o
    
[^104]: Rufus-Air：一个开放的大语言模型后训练方案

    Rufus-Air: An Open LLM Post-Training Recipe

    [https://arxiv.org/abs/2609.29421](https://arxiv.org/abs/2609.29421)

    本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。

    

    Rufus-Air 是一个在 GLM-4.5-Air-Base（106B-A12B）上构建的开放且可复现的后训练方案，由八个阶段的串行流水线组成：SFT（监督微调）、推理 RL、编码 RL、指令遵循 RL、通用智能体、编码智能体、搜索智能体和 RLHF。我们记录了复现该方案所需的数据、奖励设计、基础设施、阶段顺序以及各阶段的结果。各阶段从基础能力逐步推进到高级能力，奖励信号也从严格可验证的奖励过渡到较为柔和的基于评判者的信号。训练基于开源组件和公开数据，其中大部分数据按原样使用，无需新的人工标注或内部蒸馏教师模型。我们的主要发现是：(i) 多样化、高质量的 SFT 奠定了坚实的能力基础；(ii) 难度过滤可将 RL 提示保持在有效的学习区间内；(iii) 奖励可靠性为阶段排序提供了实用原则；(iv) 基础设施与工程选择是……

    arXiv:2609.29421v1 Announce Type: cross  Abstract: Rufus-Air is an open and reproducible post-training recipe on GLM-4.5-Air-Base (106B-A12B), organized as a serial pipeline of eight stages: SFT, Reasoning RL, Coding RL, Instruction-Following RL, General Agent, Coding Agent, Search Agent, and RLHF. We document the data, reward design, infrastructure, stage order, and stagewise results needed to reproduce the recipe. Stages progress from basic to advanced capabilities and from hard, verifiable rewards to softer judge-based signals. Training builds on open-source components and public data, much of it used as released, without new human annotation or an in-house distillation teacher. Our main findings are that (i) diverse, high-quality SFT establishes a strong capability floor; (ii) difficulty filtering keeps RL prompts within a productive learning range; (iii) reward reliability provides a practical principle for ordering stages; and (iv) infrastructure and engineering choices are part 
    
[^105]: 神经传输嵌套采样

    Neural Transport Nested Sampling

    [https://arxiv.org/abs/2609.29413](https://arxiv.org/abs/2609.29413)

    提出神经传输嵌套采样（NTNS）算法，将嵌套采样与神经流方法相结合，仅需目标能量函数评估即可对高维分子系统进行采样并估计完整配分函数，在含55个粒子的Lennard-Jones团簇上，将采样误差较最强神经基线降低了一个数量级以上。

    

    从分子系统的玻尔兹曼分布中进行采样是一个推断问题，近年来在神经密度估计技术进展的推动下取得了显著发展。我们开发了一种新颖的采样算法——神经传输嵌套采样（NTNS），它将嵌套采样的经典优势与现代基于神经流的方法相结合。NTNS 在嵌套采样外循环中，使用流匹配速度作为经过 Metropolis–Hastings 校正的 Langevin 核中的漂移项，仅需对目标能量函数进行评估，并能为高维粒子系统的完整配分函数提供可扩展的估计。我们在具有挑战性的分子采样基准上对 NTNS 进行了测试，规模扩展至由 55 个相互作用粒子组成的 Lennard–Jones 团簇。在该任务上，相对于最强的神经基线方法，NTNS 将与参考 MCMC 相比的原子间距离和能量的 Wasserstein 误差降低了一个数量级以上。

    arXiv:2609.29413v1 Announce Type: new  Abstract: Sampling from Boltzmann distributions of molecular systems is an inference problem that has seen significant recent developments fuelled by advances in neural density estimation. We develop a novel sampling algorithm, Neural Transport Nested Sampling (NTNS), which combines the classical strengths of nested sampling with modern neural flow-based methods. NTNS uses a flow matching velocity as the drift in a Metropolis--Hastings corrected Langevin kernel inside a nested sampling outer loop, requiring only evaluations of the target energy function and providing scalable estimation of the full partition function of high-dimensional particle systems. We benchmark NTNS on challenging molecular sampling benchmarks, scaling up to Lennard--Jones clusters of 55 interacting particles, where it reduces both interatomic distance and energy Wasserstein errors to reference MCMC by over an order of magnitude relative to the strongest neural baselines at 
    
[^106]: 吉布斯监督学习算法的机器遗忘

    Machine Unlearning for Gibbs Supervised Learning Algorithms

    [https://arxiv.org/abs/2609.29409](https://arxiv.org/abs/2609.29409)

    提出了一种基于ERM-RER变分形式的精确遗忘方法，使吉布斯监督学习算法在遗忘数据后与从头重新训练的结果在分布上完全一致。

    

    本文提出了一种针对吉布斯监督学习算法实现精确遗忘的方法，该方法采用受相对熵正则化经验风险最小化（ERM-RER）启发的变分形式。该方法通过在待遗忘数据集上最大化期望经验风险，并以相对于原始算法的相对熵作为正则化约束来实现。优化变量是模型空间上的一个概率测度，其解为另一个吉布斯概率测度，代表一个新的吉布斯监督学习算法。该方法保证了精确遗忘，即新的吉布斯算法在分布上与在保留数据集上从头重新训练所得到的算法完全一致。作为副产品，该方法还提供了一个通过策略性地选择参考测度和正则化项来对ERM-RER中的数据点进行重新加权的框架。

    arXiv:2609.29409v1 Announce Type: cross  Abstract: In this paper, a method for achieving exact unlearning for Gibbs supervised learning algorithms is proposed using a variational formulation inspired by empirical risk minimization subject to relative entropy regularization (ERM-RER). Such a method consists of maximizing the expected empirical risk over the dataset to be unlearned subject to a regularization by relative entropy with respect to the original algorithm. The optimization variable is a probability measure on the models; and the solution is another Gibbs probability measure that represents a new Gibbs supervised learning algorithm. The method guarantees exact unlearning in the sense that the new Gibbs algorithm coincides in distribution with the algorithm that would have been obtained by retraining from scratch on the dataset to be retained. As a byproduct, a framework for reweighting data points in ERM-RER by strategically choosing both the reference measure and the regulari
    
[^107]: 通过强化伴随匹配在真实录音上对生成式语音增强进行转录文本监督的后训练

    Transcript-Supervised Post-Training of Generative Speech Enhancement on Real Recordings via Reinforce Adjoint Matching

    [https://arxiv.org/abs/2609.29405](https://arxiv.org/abs/2609.29405)

    本文提出利用强化伴随匹配（RAM）方法，仅以文本转录作为弱监督信号，直接在真实录音上对生成式语音增强模型进行后训练，在不损害感知语音质量的前提下将词错误率降低5.08个百分点。

    

    我们将强化伴随匹配（Reinforce Adjoint Matching，RAM）这一基于奖励的后训练方法应用于生成式语音增强（SE）。从一个预训练的语音增强模型出发，RAM将模型的条件分布向更高奖励的输出方向倾斜。在训练过程中，当前模型在线生成增强语音，使用可能不可微的奖励函数评估每个生成的端点，然后通过解析方式对端点重新加噪，以构建奖励引导回归目标所需的输入。这使得后训练可以直接在真实录音上进行，仅需文本转录等弱监督信息，而无需成对的干净语音目标或奖励梯度。我们研究了基于词错误率（WER）的后训练方法，以及能否在不损害语音感知质量的前提下提升识别性能。在真实CHiME-4录音上的实验表明，相比预训练的FlowSE模型，词错误率降低了5.08个百分点，且不降低任何感知语音质量指标。

    arXiv:2609.29405v1 Announce Type: cross  Abstract: We adapt Reinforce Adjoint Matching (RAM), a reward-based post-training method, to generative speech enhancement (SE). Starting from a pretrained SE model, RAM tilts the model's conditional distribution toward outputs with higher reward. During training, the current model generates enhanced speech on-policy, evaluates each generated endpoint with a potentially non-differentiable reward, and analytically re-noises the endpoint to construct inputs for a reward-guided regression objective. This enables post-training directly on real recordings using weak supervision, such as text transcripts, without requiring paired clean speech targets or reward gradients. We investigate word error rate (WER)-based post-training and whether recognition performance can be improved without compromising perceptual speech quality. Experiments on real CHiME-4 recordings reduce WER by 5.08 percentage points relative to pretrained FlowSE without reducing any o
    
[^108]: RD-JEPA：面向反应扩散方程少轨迹迁移的预测性潜空间预训练

    RD-JEPA: Predictive latent pretraining for few-trajectory transfer across reaction--diffusion equations

    [https://arxiv.org/abs/2609.29403](https://arxiv.org/abs/2609.29403)

    提出RD-JEPA自监督预训练架构，在多个反应扩散系统上联合预训练后，仅需少量轨迹即可高效迁移到未见过的反应扩散方程，误差低于监督基线及从头训练模型。

    

    为时间依赖偏微分方程学习代理模型时，一旦控制算子发生改变，往往就需要重新构建一套仿真数据集。我们提出RD-JEPA，一种用于反应扩散轨迹自监督预训练的联合嵌入预测架构。该单一模型在五个参数化系统上进行预训练，随后被适配到三个保留系统上，这些系统的反应算子和轨迹均未包含在预训练数据中。仅使用来自保留系统的一条、五条或十条完整轨迹，RD-JEPA相比五个监督代理基线模型、一个移除了轨迹相关预测潜通路的独立训练对照组，以及一个架构相同但从零开始训练的模型，均取得了更低的平均相对离散 $\ell^2$ 场误差和平均绝对空间一阶差分误差。在所评估的方程、输出分辨率、预测时域以及适配轨迹的选择范围内……（原文摘要在此处被截断）

    arXiv:2609.29403v1 Announce Type: new  Abstract: Learning surrogates for time-dependent partial differential equations often requires a new simulation corpus when the governing operator changes. We introduce RD-JEPA, a joint-embedding predictive architecture for self-supervised pretraining on reaction-diffusion trajectories. A single model is pretrained on five parameterized systems and then adapted to three held-out systems whose reaction operators and trajectories are excluded from pretraining. Using one, five, or ten complete trajectories from a held-out system, RD-JEPA achieves lower mean relative discrete $\ell^2$ field error and mean absolute spatial first-difference error than five supervised surrogate baselines, an independently trained control that removes the trajectory-dependent predictive latent pathway, and an architecture-matched model trained from scratch. Within the evaluated equations, output resolution, forecast horizons, and choices of adaptation trajectories, the re
    
[^109]: ICE：面向多模态图基础模型的任务对齐克利福德潜在场

    ICE: Task-Aligned Clifford Latent Fields for Multimodal Graph Foundation Models

    [https://arxiv.org/abs/2609.29398](https://arxiv.org/abs/2609.29398)

    该论文提出ICE（交互感知克利福德编码器），一种基于节点索引克利福德潜在场的多模态图基础模型，通过将拓扑、文本和图像编码为显式的Cl(3)几何地址并利用边缘感知几何积，统一地保留实体语义、构建图邻域交互状态，并为具有不同几何结构的预测任务提供适配的表示。

    

    多模态属性图将实体、视觉内容、语言和观测到的关系连接在一起。在这样的图上学习统一的基础模型，不仅仅是把每个节点压缩成一个融合的欧几里得向量。其表示必须保留实体语义、从图邻域构建交互状态，并将该状态呈现给具有不同几何结构的预测单元。我们的实证研究表明这些需求为何不可分割：更高阶的通道能够恢复基础图中的节点对关系，专门的查询能揭示被通用读出层所隐藏的信息，而刚性的blade隔离则会消除跨阶能力。因此，我们提出了ICE（交互感知克利福德编码器），一个建立在节点索引克利福德潜在场之上的多模态图基础模型。拓扑、文本和图像被映射到显式的Cl(3)几何地址中。边缘感知的几何积将这些方向转换为标量、双向量和三向量（摘要原文在此处截断）。

    arXiv:2609.29398v1 Announce Type: new  Abstract: Multimodal attributed graphs connect entities, visual content, language, and observed relations. Learning one foundation across such graphs requires more than compressing each node into a fused Euclidean vector. The representation must preserve entity semantics, construct interaction state from graph neighborhoods, and expose that state to prediction units with different geometry. Our empirical study shows why these requirements are inseparable. Higher-grade channels recover pair relations across the foundation graphs, specialized queries reveal information hidden by a generic readout, and rigid blade isolation removes cross-grade capacity. We therefore introduce ICE (Interaction-aware Clifford Encoder), a multimodal graph foundation model built on a node-indexed Clifford latent field. Topology, text, and images enter explicit Cl(3) addresses. Edge-aware geometric products transform these directions into scalar, bivector, and trivector r
    
[^110]: 通过稳定客户端聚类实现并发分割学习

    Concurrent Split Learning Through Stable Client Clustering

    [https://arxiv.org/abs/2609.29395](https://arxiv.org/abs/2609.29395)

    本文提出全局聚类并行分割学习（GCPSL），通过将客户端稳定分配到固定集群并并发执行多个分割学习工作负载，在不增大单个批量的前提下提升客户端直接数据参与率，使CIFAR-10达到85%验证准确率的时间从串行执行的约19分钟缩短至约6分钟。

    

    使用固定全局批量的训练限制了在任何单一步骤中能够提供样本的分布式客户端数量。我们研究了在不增加单个工作负载所处理批量的情况下，利用额外服务器工作节点的方法。全局聚类并行分割学习（GCPSL）将客户端分配到固定的集群，为每个集群并发执行带全局采样的并行分割学习（GPSL）工作负载，并周期性地融合客户端与服务器模型片段。在包含256个逻辑客户端的仿真中，将客户端群体划分到更多工作负载中可以提高直接数据参与率，而较小的集群可能带来准确率损失。在四块H100 GPU上实现的标签感知GCPSL，在三次匹配运行中平均于6.13±0.15分钟内达到CIFAR-10上85%的验证准确率，而相同工作负载串行执行则需要19.09±0.45分钟。在四GPU的资源配置下，规模均衡和随机固定归属的集群划分达到了目……（原文在此处截断）

    arXiv:2609.29395v1 Announce Type: cross  Abstract: Training with a fixed global batch limits how many distributed clients can provide examples in any one step. We examine a way to use additional server workers without increasing the batch processed by an individual workload. Global Clustered Parallel Split Learning (GCPSL) assigns clients to fixed clusters, executes a Parallel Split Learning with Global Sampling (GPSL) workload for each cluster concurrently, and periodically fuses the client and server model segments. In simulations with 256 logical clients, dividing the population across more workloads improves direct data participation, while smaller clusters can incur an accuracy cost. A four-H100 implementation of label-aware GCPSL reaches 85% CIFAR-10 validation accuracy in $6.13 \pm 0.15$ minutes over three matched runs, versus $19.09 \pm 0.45$ minutes when the same workloads are serialized. Within the four-GPU allocation, size-balanced and random fixed affiliations reach the tar
    
[^111]: MORE-PLR：用于部分标签排序的多输出回归方法

    MORE-PLR: multi-output regression employed for partial label ranking

    [https://arxiv.org/abs/2609.29386](https://arxiv.org/abs/2609.29386)

    本文提出MORE-PLR方法，创新性地将多输出回归应用于部分标签排序问题，通过编码器将带并列关系的（可能不完整的）标签排序编码为多元回归目标，并利用事后处理层将回归结果转换为桶序输出。

    

    部分标签排序问题是一种监督学习场景，旨在拟合一个偏好模型，为给定的输入实例预测定义在标签集合上的桶序。该问题推广了著名的标签排序问题，而后者在实践中仅限于输出标签的全序。现有的部分标签排序方法主要是通过扩展标签排序方法来处理预测中的并列关系。本文提出使用多输出回归来解决部分标签排序问题，引入了一种编码器，在学习阶段将（可能不完整的）带并列关系的标签排序转换为多元回归目标，这是标签排序和部分标签排序领域中都未被充分探索的视角。此外，在推理阶段，我们引入了若干事后处理层，将多输出回归结果转换为输出的桶序，从而有效实现该方法。

    arXiv:2609.29386v1 Announce Type: new  Abstract: The partial label ranking problem is a supervised learning scenario that aims to fit a preference model that predicts a bucket order defined over a set of labels for a given input instance. This problem generalizes the well-known label ranking problem, which, in practice, is limited to outputting total orders of labels. Existing partial label ranking methods have primarily extended label ranking approaches to handle ties in predictions. This paper proposes using multi-output regression to address the partial label ranking problem, introducing an encoder that, during the learning phase, transforms the (possibly incomplete) rankings with ties of labels to multivariate regression targets, an underexplored perspective in both label ranking and partial label ranking. Moreover, during the inference phase, we introduce several post-hoc layers that convert the multi-output regression results into the output bucket order to effectively implement 
    
[^112]: 基于确定性基础模型的轻量级概率降尺度方法

    Lightweight Probabilistic Downscaling from a Deterministic Base Model

    [https://arxiv.org/abs/2609.29383](https://arxiv.org/abs/2609.29383)

    本文提出将确定性预训练与概率微调相结合的两阶段训练方法，构建轻量级概率机器学习降尺度模型，在阿尔卑斯山、新西兰和南非三个区域的气温与降水降尺度任务中，RMSE超越了现有最先进方法。

    

    气候数据降尺度是提高气候数据空间分辨率的任务，通常通过从粗分辨率的全球模式输出中生成高分辨率的区域气候数据来实现。近期在相关任务——天气预报中的机器学习（ML）研究，得益于新设计的训练方法和架构组件而取得了显著进步，但这些进展尚未惠及降尺度任务。我们改编了其中两种方法，构建了一系列基于改进U-Net骨干网络的轻量级概率机器学习降尺度模型，并在CORDEX-ML-Bench测试套件上，对阿尔卑斯山、新西兰和南非三个地理区域的日最高气温和降水进行了评估。我们发现，将确定性预训练与概率微调相结合的两阶段训练课程能够很好地迁移到降尺度任务中，在RMSE指标上超越了现有最先进方法。我们的工作为轻量级、概率化的降尺度迈出了重要一步。

    arXiv:2609.29383v1 Announce Type: new  Abstract: Climate data downscaling is the task of increasing the spatial resolution of climate data, typically by generating fine-resolution regional climate data from coarse global model output. Recent machine learning (ML) work in the related task of weather forecasting has seen significant improvements due to newly devised training methods and architectural components, but these have not yet benefited downscaling. We adapt two of these methods to create a family of lightweight probabilistic ML downscaling models built on a modified U-Net backbone and evaluate them on the CORDEX-ML-Bench suite for daily maximum temperature and precipitation across three geographic regions: the Alps, New Zealand and South Africa. We find that a two-stage training curriculum, combining deterministic pretraining with probabilistic tuning, transfers well to downscaling, beating the state-of-the-art for RMSE. Our work provides an advancement towards lightweight, prob
    
[^113]: 流匹配视觉-语言-动作模型中面向任务依赖计算分配的解耦早退机制

    Decoupled Early Exits for Task-Dependent Compute Allocation in Flow-Matching VLAs

    [https://arxiv.org/abs/2609.29382](https://arxiv.org/abs/2609.29382)

    提出一个将骨干网络深度、动作专家深度和去噪步骤作为三个可联合配置计算轴的框架，通过在骨干网络和动作专家中附加轻量级退出变换器实现解耦早退，从而实现面向任务的动态计算分配并降低流匹配VLA模型的计算开销。

    

    流匹配视觉-语言-动作模型已成为通用机器人控制的一种潜在解决方案，其设计方式是将预训练的视觉-语言模型骨干网络与生成连续机器人动作的动作专家相结合。尽管这些模型展现出令人印象深刻的能力，但由于其参数量极高，其计算需求对于机器人控制而言往往过于高昂。为了缓解这些低效问题，现有方法主要通过早退机制跳过VLM骨干网络层或减少去噪步骤，而保持动作专家深度不变。我们提出了一个框架，将骨干网络深度V、动作专家深度A和去噪步骤D作为VLA中三个可联合配置的计算轴。从预训练的VLA出发，我们在骨干网络和动作专家的中间深度处附加轻量级的退出变换器（ET），并通过训练使其蒸馏最后一层的输出（摘要在此处截断）。

    arXiv:2609.29382v1 Announce Type: cross  Abstract: Flow-matching Vision-Language-Action (VLA) models have emerged as a potential solution for generalist robot control, designed by combining a pretrained Vision-Language Model (VLM) backbone with an action expert that generates continuous robot actions. While these models exhibit impressive capabilities, due to their very high number of parameters, their computational requirements are often prohibitive for robotics control. To mitigate these inefficiencies, existing methods predominantly skip VLM backbone layers with early exits or reduce denoising steps, while leaving action expert depth untouched. We propose a framework that exposes backbone depth $V$, action expert depth $A$, and denoising steps $D$ as three jointly configurable compute axes in a VLA. Starting from a pretrained VLA, we attach lightweight Exit Transformers (ET) at intermediate depths in both the backbone and the action expert, trained to distil the last layer of the po
    
[^114]: 脉冲神经网络的二阶优化方法研究

    On the second-order optimization for spiking neural networks

    [https://arxiv.org/abs/2609.29379](https://arxiv.org/abs/2609.29379)

    提出SpiKFAX，一种专为脉冲神经网络设计的二阶优化方法，通过对Fisher信息矩阵进行Kronecker因子分解近似来有效捕捉脉冲激活导致的尖锐损失曲面，在多种架构和数据集上持续提升测试准确率。

    

    脉冲神经网络通过利用稀疏的二值脉冲和事件驱动计算，为传统神经网络提供了一种高能效的替代方案。然而，SNN的训练仍然具有挑战性，因为脉冲激活会产生尖锐的损失曲面，阻碍训练的进行，而Adam系列等对角曲率优化器可能无法捕捉这种几何结构。此外，将基于曲率的优化方法扩展到SNN，还因其底层动态的稀疏性、离散性和时间递归特性而变得更加复杂。为了解决这些局限性，我们提出了SpiKFAX，这是一种二阶优化方法，它构建了一个计算上可行的、专门适应SNN结构的Fisher信息矩阵Kronecker因子分解近似。在五种架构和七个数据集上的实证评估表明，SpiKFAX在测试准确率方面持续取得改进。

    arXiv:2609.29379v1 Announce Type: new  Abstract: Spiking Neural Networks (SNNs) offer an energy-efficient alternative to conventional neural networks by exploiting sparse, binary spikes, and event-driven computation. However, the training of SNNs remains challenging, as spiking activations create a sharp loss landscape that hinders training, and diagonal-curvature optimizers such as the Adam family may fail to capture this geometry. The extension of curvature-based optimization methods to SNNs is further complicated by the sparse, discrete, and temporally recurrent nature of their underlying dynamics. To address these limitations, we propose SpiKFAX, a second-order optimization method that formulates a computationally tractable, Kronecker-factored approximation of the Fisher information matrix specifically adapted to the structure of SNNs. Empirical evaluation across five architectures and seven datasets demonstrates that SpiKFAX consistently yields improvements in test accuracy and tr
    
[^115]: 学习一个通向自监督表示的流

    Learning a Flow to Self-Supervised Representations

    [https://arxiv.org/abs/2609.29350](https://arxiv.org/abs/2609.29350)

    本文提出非对抗性的基于流的分布匹配框架 FBDM，通过球面条件速度回归学习参考引导的几何结构，避免了昂贵的编码器-评论家优化，性能与分布匹配方法几乎相当并与现有自监督学习方法具有竞争力。

    

    显式的几何参考为构建自监督表示提供了一种直接的方式。然而，现有的对抗性分布匹配方法需要代价高昂的编码器-评论家（encoder-critic）联合优化。我们提出了基于流的分布匹配（Flow-Based Distribution Matching, FBDM），这是一个非对抗性框架，通过球面条件速度回归来学习这种参考引导的几何结构。受等角紧框架（ETF）启发的参考允许其分量数量 K' 超过辅助流维度 d*，同时保持结构化的几何分离。我们将每张图像的两个增广视图分配给同一目标，同时限制每个参考中心可以接收的图像数量。显式的对齐损失进一步拉近了两个视图的表示。在从 CIFAR 到 ImageNet 的多个基准测试上的实验表明，FBDM 取得了与 DM 几乎相当的性能，并与现有的自监督学习（SSL）方法保持竞争力。在匹配训练成本的比较中……（摘要在此处截断）

    arXiv:2609.29350v1 Announce Type: cross  Abstract: Explicit geometric references offer a direct way to structure self-supervised representations. Existing adversarial distribution-matching formulations, however, require costly encoder-critic optimization. We introduce Flow-Based Distribution Matching (FBDM), a non-adversarial framework that learns this reference-directed geometry through spherical conditional velocity regression. An ETF-inspired reference allows its number of components K' to exceed the auxiliary flow dimension d* while retaining structured geometric separation. We assign both augmented views of each image to the same target, while limiting how many images each reference center can receive. An explicit alignment loss further pulls the two views' representations closer together. Experiments across benchmarks ranging from CIFAR to ImageNet show that FBDM achieves performance nearly on par with DM and remains competitive with existing SSL methods. Matched training-cost co
    
[^116]: FlowAtom：面向多标签网站指纹识别的基于原子的证据聚合

    FlowAtom: Atom-Based Evidence Aggregation for Multi-Label Website Fingerprinting

    [https://arxiv.org/abs/2609.29330](https://arxiv.org/abs/2609.29330)

    FlowAtom通过在无标签流量上预训练编码器学习共享“原子”原型，并将观察窗口内多个流量流的原子响应聚合为置换不变表示，从而实现对混合加密流量中受监控网站集合的多标签识别，闭世界微F1分数最高达97.82%。

    

    在混合加密流量中识别受监控网站集合是一项具有挑战性的任务，因为单个流量流往往只能提供关于网站身份的部分证据。为应对这一挑战，我们提出了FlowAtom，它从无网站标签的流表示中构建共享原型，称为“原子”。具体而言，FlowAtom在外部无标签流量上预训练流编码器，并将每个观察窗口内跨多个流的原子响应聚合为固定维度、置换不变的表示，用于受监控网站集合的预测。在Direct HTTPS、Trojan和VMess协议上，FlowAtom在闭世界评估中分别取得了97.82%、94.43%和93.92%的微F1分数，并且在包含受监控访问的窗口的开放世界评估中始终优于所评估的基线方法。代码可在 https://github.com/aimafan123/FlowAtom 获取。

    arXiv:2609.29330v1 Announce Type: new  Abstract: Identifying the set of monitored websites in mixed encrypted traffic is challenging because an individual flow often provides only partial evidence of website identity. To address this challenge, we propose FlowAtom, which constructs shared prototypes, called Atoms, from flow representations without website labels. Specifically, FlowAtom pretrains a flow encoder on external unlabeled traffic and aggregates Atom responses across flows within each observation window into a fixed-dimensional, permutation-invariant representation for monitored website-set prediction. Across Direct HTTPS, Trojan, and VMess, FlowAtom achieves micro-F1 scores of 97.82%, 94.43%, and 93.92% in closed-world evaluation, respectively, and consistently outperforms the evaluated baselines in open-world evaluation on windows containing monitored visits. The code is available at https://github.com/aimafan123/FlowAtom.
    
[^117]: GCUL：通过聚类引导学习实现文本情感分类中的歧义识别

    GCUL: Ambiguity Identification in Text Emotion Classification via Cluster-Guided Learning

    [https://arxiv.org/abs/2609.29327](https://arxiv.org/abs/2609.29327)

    提出了一种几何引导的选择性分类框架GCUL，将误分类和歧义实例视为表示空间中的混淆吸引子，通过三阶段聚类引导学习使拒绝边界从表示几何结构中自然涌现，而非依赖预设的拒绝率。

    

    选择性分类使模型能够对不确定的实例放弃预测，但现有方法通常通过置信度分数、预定义的覆盖率约束或实例级距离度量来拒绝这些实例。这些方法可能忽视了学习表示空间中困难样本的集体几何结构。我们提出了引导聚类式不确定学习（GCUL），这是一种几何引导的选择性分类框架，它将误分类和歧义实例识别为表示空间中的潜在混淆吸引子。GCUL使用三阶段程序来初始化、聚类并显式重新标注这一不确定区域，使拒绝边界从底层表示几何结构中自然涌现，而非依赖于预设的拒绝率。我们进一步推导了一个选择性分数和一个几何充分条件，用于刻画拒绝机制何时能够产生积极效果。

    arXiv:2609.29327v1 Announce Type: cross  Abstract: Selective classification enables a model to abstain from predictions on uncertain instances, but existing approaches typically reject them through confidence scores, predefined coverage constraints or instance-level distance measures. These approaches may overlook the collective geometric structure of difficult samples in learned representation spaces. We propose Guided Clustering-based Uncertain Learning (GCUL), a geometric-guided selective classification framework that identifies misclassified and ambiguous instances as a potential confusion attractor in the representation space. GCUL uses a three-phase procedure to initialize, cluster, and explicitly relabel this uncertain region, allowing the rejection boundary to emerge from the underlying representation geometry rather than from a prescribed rejection rate. We further derive a selectivity score and a geometric sufficient condition that characterizes when rejection can provide pos
    
[^118]: TinyCardioUNet：基于图编码轴间依赖关系与张量分解参数缩减的IMU到ECG转换

    TinyCardioUNet: IMU-to-ECG Translation with Graph-Encoded Inter-Axis Dependencies and Tensor Decomposition-Based Parameter Reduction

    [https://arxiv.org/abs/2609.29322](https://arxiv.org/abs/2609.29322)

    TinyCardioUNet通过图神经网络编码IMU六轴间的依赖关系，并利用基于自动变分贝叶斯秩选择的张量分解缩减参数，仅以36k参数的轻量级模型即可从胸部佩戴的IMU信号准确重建心电图。

    

    从佩戴于胸部的惯性测量单元估算心电图（ECG）能够实现无需电极不适感的连续心率监测。我们提出了TinyCardioUNet，这是一种轻量级UNet，无需预先进行通道选择即可使用全部六个IMU轴，通过编码轴间依赖关系的图神经网络细化其瓶颈层，并采用基于自动变分贝叶斯秩选择的张量分解来实现参数缩减。在公开数据集上，TinyCardioUNet仅使用36.0k参数便实现了0.098的均方根误差（RMSE）和0.677的皮尔逊相关系数，并且对加性噪声保持相对鲁棒，证明了紧凑模型能够实现准确的ECG重建。

    arXiv:2609.29322v1 Announce Type: new  Abstract: Estimating electrocardiography (ECG) from a chest-worn inertial measurement unit (IMU) enables continuous heart rate (HR) monitoring without the discomfort of electrodes. We propose TinyCardioUNet, a lightweight UNet that uses all six IMU axes without prior channel selection, refines its bottleneck with a graph neural network that encodes inter-axis dependencies, and employs tensor decomposition with automatic variational Bayesian rank selection for parameter reduction. On a public dataset, TinyCardioUNet achieves an RMSE of $0.098$ and a Pearson correlation coefficient of $0.677$ with only $36.0$k parameters and remains comparatively robust to additive noise, demonstrating accurate ECG reconstruction with a compact model.
    
[^119]: 用于时间序列分类与预测的神经化多小波分解

    Neuralized Multi-Wavelet Decomposition for Time Series Classification and Forecasting

    [https://arxiv.org/abs/2609.29317](https://arxiv.org/abs/2609.29317)

    提出 m-WCN 端到端深度学习框架，通过用可训练卷积算子近似经典 GHM 多小波变换并施加正交性约束，将多小波分解神经化，实现时间序列时域模式与频域成分的联合提取，从而提升时间序列分类与预测的性能。

    

    arXiv:2609.29317v1 公告类型： cross 摘要：时间序列分析在金融、医疗和气象等领域具有基础性地位。现实世界中的时间序列往往表现出由多种潜在因素塑造的多尺度特性，从而形成复杂的时序模式和丰富的频率结构。然而，现有方法通常孤立地关注频域分解或时域模式提取，忽视了二者的联合结构。这种解耦建模限制了表示的表达能力，削弱了需要同时进行时域与频域推理的任务的性能。为填补这一空白，我们提出 m-WCN，一种新颖的端到端深度学习框架，它将多小波分解神经化，以联合提取时间模式与频率成分。通过使用可训练的卷积算子逼近经典的 GHM 多小波变换并施加正交性约束，m-WCN 产生可解释的多尺度表示（摘要原文在此处截断）。

    arXiv:2609.29317v1 Announce Type: cross  Abstract: Time series analysis is fundamental in domains such as finance, healthcare, and meteorology. Real-world time series often exhibit multiscale characteristics shaped by diverse latent factors, resulting in intricate temporal patterns and rich frequency structures. However, existing approaches typically focus on either frequency-domain decomposition or time-domain pattern extraction in isolation, neglecting their joint structure. This decoupled modeling limits representation expressiveness and undermines performance in tasks requiring simultaneous temporal and spectral reasoning. To address this gap, we propose m-WCN, a novel end-to-end deep learning framework that neuralizes multi-wavelet decomposition for joint extraction of temporal patterns and frequency components. By approximating the classical GHM multi-wavelet transform with trainable convolutional operators and enforcing orthogonality constraints, m-WCN produces interpretable mul
    
[^120]: 超越特征可靠性：面向脑龄预测的重复扫描感知多分形曲线回归

    Beyond Feature Reliability: Repeat-Informed Multifractal Curve Regression for Brain-Age Prediction

    [https://arxiv.org/abs/2609.29307](https://arxiv.org/abs/2609.29307)

    提出RMCR框架，通过联合建模多分形曲线结构与重复扫描变异性，学习稳定且可重复的年龄预测模式，同时提升脑龄预测的准确性和被试内一致性。

    

    从静息态fMRI进行脑龄预测，为刻画自发脑动力学的年龄相关变化以及识别功能性特征提供了一个定量框架。现有研究已将分形与多分形标度特性与年龄联系起来，并考察了单个特征的可靠性。然而，预测的重复性取决于各特征如何联合波动以及预测模型如何组合这些特征，而基于单个特征的可靠性评估无法捕捉这一点。为解决这一问题，我们提出了重复扫描感知多分形曲线回归（RMCR），这是一个从多分形曲线中学习稳定年龄预测模式的结构化框架。通过联合建模曲线结构与重复扫描变异性，RMCR学习能够同时兼顾准确性和被试内一致性的波动阶数预测组合。相对于匹配的运行级岭回归基线，RMCR在HCP-A上将单次运行的MAE降低了6.1%，在其他数据集上降低了7.9%（摘要在此处截断）。

    arXiv:2609.29307v1 Announce Type: new  Abstract: Brain-age prediction from resting-state fMRI provides a quantitative framework for characterizing age-related changes in spontaneous brain dynamics and for identifying functional signatures. Existing studies have linked fractal and multifractal scaling to age and examined the reliability of individual features. However, prediction repeatability depends on how features fluctuate jointly and how a predictor combines them, which feature-wise reliability assessments do not capture.   To address this problem, we propose Repeat-informed Multifractal Curve Regression (RMCR), a structured framework for learning stable age-predictive patterns from multifractal curves. By jointly modeling curve structure and repeat-scan variability, RMCR learns predictive combinations of fluctuation orders that target both accuracy and within-subject consistency.   Relative to a matched run-level ridge baseline, RMCR reduces single-run MAE by 6.1% on HCP-A and 7.9
    
[^121]: 充分约简分布回归

    Sufficiently Reduced Distributional Regression

    [https://arxiv.org/abs/2609.29291](https://arxiv.org/abs/2609.29291)

    本文提出SRDR方法，通过严格恰当评分规则将充分降维转化为风险最小化问题，并利用可通过采样估计的能量评分联合训练降维映射与生成式预测模型，无需密度计算或对抗训练，同时证明了估计条件分布在能量距离上的收敛性。

    

    我们提出了充分约简分布回归（SRDR），这是一种将条件分布估计与非线性充分降维（SDR）相结合的生成式方法。该方法建立在通过严格恰当评分规则对充分性的刻画之上：当且仅当使用约简后的协变量预测响应相对于使用完整协变量不会造成期望评分损失时，该降维才是充分的。由此，充分降维被转化为一个风险最小化问题。SRDR通过最小化能量评分来联合训练降维映射和生成式预测模型，而能量评分可以通过采样进行估计，无需密度计算或对抗训练。该框架还可扩展至多环境数据和分类任务。我们证明了估计的条件分布在能量距离上收敛于真实的条件分布，这意味着学习到的表示在渐近意义下是充分的。

    arXiv:2609.29291v1 Announce Type: cross  Abstract: We propose Sufficiently Reduced Distributional Regression (SRDR), a generative method that combines conditional distribution estimation with nonlinear sufficient dimension reduction (SDR). It builds on a characterization of sufficiency through strictly proper scoring rules: a dimension reduction is sufficient if and only if predicting the response from the reduced covariates incurs no loss in expected score relative to the full covariates. Sufficient dimension reduction thus becomes a risk minimization problem. SRDR jointly trains a dimension reduction map and a generative prediction model by minimizing the energy score, which can be estimated by sampling without density evaluation or adversarial training. The framework extends to multi-environment data and to classification. We prove that the estimated conditional distributions converge in energy distance to the true ones, which implies that the learned representation is asymptoticall
    
[^122]: 从文本决策到像素：Jev风格视觉选择模型研究

    From Text Decisions to Pixels: An Study of Jev-Style Visual Choice Model

    [https://arxiv.org/abs/2609.29283](https://arxiv.org/abs/2609.29283)

    本文提出PixelJev视觉决策接口，利用小型开源多模态模型将图像、指令和候选项映射为结构化决策，通过64样本源适配将Pets准确率从60.13%大幅提升至92.40%，并能零样本迁移到其他视觉问答任务。

    

    视觉软件通常需要对提供的备选项做出决策，而非生成解释。我们提出了PixelJev，一个原生图像决策接口，它使用小型开源多模态模型，将图像、任务指令和运行时候选集合映射为结构化选择以及以候选为条件的概率。其初始实现通过现有的语言模型读取方式，将识别和多选视觉问答统一起来，并分别评估了冻结推理、语言侧适配和保留集校准等选项。在七项基准测试评估中，64样本的源适配将Pets准确率从60.13%提升至92.40%，并在不进行目标拟合的情况下迁移到自然重采样、新纹理标签和A-OKVQA任务，同时冻结推理已可支持两个VQA任务。在Pets和ScienceQA上进行的匹配纯提示对照实验将Pets的大幅提升归因于适配过程。

    arXiv:2609.29283v1 Announce Type: new  Abstract: Visual software often needs a decision over supplied alternatives rather than a generated explanation. We present PixelJev, a native-image decision interface that maps an image, a task instruction, and a runtime candidate set to a structured choice and candidate-conditioned probabilities using small open multimodal models. Its initial realization unifies recognition and multiplechoice visual question answering through an existing language-model readout, with separately evaluated options for frozen inference, language-side adaptation, and held-out calibration. Across seven benchmark evaluations, 64-shot source adaptation raises Pets accuracy from 60.13% to 92.40% across optimization seeds and transfers to natural resampling, new texture labels, and A-OKVQA without target fitting, while frozen inference already supports both VQA tasks. A matched prompt-only follow-up on Pets and ScienceQA attributes the large Pets gain to adaptation and id
    
[^123]: 基于自组织的在线任务适应

    Online Task Adaptation via Self-Organisation

    [https://arxiv.org/abs/2609.29281](https://arxiv.org/abs/2609.29281)

    本文提出一种基于神经细胞自动机的元学习自组织方法，通过局部记忆更新和delta规则，在适应阶段完全无需梯度即可实现在线任务适应。

    

    神经网络通常通过计算梯度并更新模型参数来进行适应。我们研究任务特定的适应是否可以转而由一种元学习的自组织过程产生，该过程在适应阶段完全不需要梯度。我们用神经细胞自动机（Neural Cellular Automaton）来实现这一想法，其中局部相互作用的循环单元同时维持一个循环状态和一个快速联想记忆。在元训练期间，使用反向传播来学习循环动力学以及记忆的读写方式。训练完成后，慢速模型参数保持固定，在线适应仅通过由局部预测误差和delta规则驱动的逐单元记忆更新来进行。我们评估了所学习到的机制是否能够适应语义上不同的留出分类任务。对支持数据的一次遍历即可在留出性能上带来显著提升，而无需梯度计算。

    arXiv:2609.29281v1 Announce Type: new  Abstract: Neural networks are typically adapted by computing gradients and updating model parameters. We investigate whether task-specific adaptation can instead emerge from a meta-learned self-organising process that requires no gradients at adaptation time. We instantiate this idea with a Neural Cellular Automaton in which locally interacting recurrent cells maintain both a recurrent state and a fast associative memory. During meta-training, backpropagation is used to learn the recurrent dynamics together with how the memory is read and written. Once training is complete, the slow model parameters remain fixed, and online adaptation occurs only through cellwise memory updates driven by local prediction errors and a delta rule.   We evaluate whether the learned mechanism can adapt to semantically distinct held-out classification tasks. A single pass over the support data produces substantial improvements in held-out performance without gradient c
    
[^124]: 用于解释时间序列分类器的可学习时频掩码

    Learnable Time-Frequency Masks for Explaining Time-Series Classifiers

    [https://arxiv.org/abs/2609.29270](https://arxiv.org/abs/2609.29270)

    提出了XACT通用框架，通过在任意可逆时频变换（STFT、连续及离散小波变换）的系数上学习稀疏归因掩码，为时间序列分类器提供更精确且不易突出虚假特征的解释。

    

    时间序列可解释性仍然是一个具有挑战性的问题，因为判别性信息通常编码在潜在的频率或时频特征中，而非原始信号本身。现有的归因方法通常仅在时域或固定变换域中运行，这限制了它们在不同表示之间捕获显著信息的能力。我们提出了XACT，一个通用框架，它可以在来自任意可逆时频变换的系数上学习稀疏归因掩码。我们在短时傅里叶变换（STFT）、连续小波变换和离散小波变换上对该框架进行了评估。此外，我们将虚拟检查层方法从STFT扩展到两种小波变换，使LRP能够在这些表示中生成解释。在一个合成数据集上，XACT能够产生精确的解释，并且比所测试的基线方法更不容易突出虚假特征。在两个真实数据集上（摘要内容在此处截断）。

    arXiv:2609.29270v1 Announce Type: new  Abstract: Time-series explainability remains challenging because discriminative information is often encoded in latent frequency or time-frequency features rather than in the raw signal itself. Existing attribution methods typically operate either in the time domain or in a fixed transform domain, limiting their ability to capture salient information across different representations. We propose XACT, a general framework that learns sparse attribution masks over coefficients from arbitrary invertible time-frequency transforms. We evaluate the framework on the STFT, the continuous wavelet transform, and the discrete wavelet transform. In addition, we extend the virtual inspection layer approach from the STFT to both wavelet transforms, enabling LRP to generate explanations in these representations. On a synthetic dataset, XACT produces precise explanations and is less prone to highlighting spurious features than the tested baselines. Across two real
    
[^125]: BridgeMem：面向时序知识图谱预测的因果二元组转移残差

    BridgeMem: Causal Dyadic Transition Residuals for Temporal Knowledge Graph Forecasting

    [https://arxiv.org/abs/2609.29268](https://arxiv.org/abs/2609.29268)

    提出BridgeMem，通过检索查询实体对的历史转移事件，经支持度自适应的经验贝叶斯读取器将其转换为似然比残差，用以校正冻结的全词表时序知识图谱预测器的打分，从而捕捉现有方法所忽略的特定实体对转移证据。

    

    时序知识图谱预测旨在从已观测事件的时间结构中推断未来的关系事实。现有预测方法主要通过实体状态、关系状态、路径或精确递归来概括历史，这些视角往往忽略了特定实体对的转移证据，即查询主体与候选对象之间先前的关系如何改变目标关系出现的概率。我们提出BridgeMem，将该量估计为一个残差并加到冻结的全词表预测器的对数分数上。对于每个候选对象，BridgeMem检索严格早于时刻t的该实体对事件，编码其关系、方向与时间间隔，并将其转换为似然比校正。一个支持度自适应的经验贝叶斯读取器在精确转移计数充足时信任这些计数，在计数稀疏时则退回到可学习的注意力估计器。骨干模型自身的不确定性对该校正进行门控……（摘要原文在此处截断）

    arXiv:2609.29268v1 Announce Type: new  Abstract: Temporal knowledge graph forecasting aims to infer future relational facts from the temporal structure of observed events. Existing forecasters mainly summarize history through entity states, relation states, paths, or exact recurrence. These views often miss pair-specific transition evidence, that is, the way prior relations between the query actor and a candidate change the odds of the target relation. We introduce BridgeMem, which estimates this quantity as a residual added to the log scores of a frozen full-vocabulary forecaster. For each candidate, BridgeMem retrieves the pair's events that strictly precede t, encodes their relations, directions, and lags, and converts them into a likelihood-ratio correction. A support-adaptive empirical-Bayes reader trusts exact transition counts where they are abundant and backs off to a learned attention estimator where they are sparse. The backbone's own uncertainty gates the correction, so conf
    
[^126]: 用于一维非线性力规律的AFT神经函数逼近器

    AFT Neural Function Approximators for 1D Nonlinear Force Laws

    [https://arxiv.org/abs/2609.29242](https://arxiv.org/abs/2609.29242)

    本文提出用神经网络替代谐波平衡法中计算代价高昂的交替频-时（AFT）格式，直接由位移傅里叶系数预测非线性力系数及其雅可比矩阵，从而在求解器和延拓算法不变的情况下高效计算含非线性接触与摩擦的装配结构频响曲线。

    

    非线性接触和摩擦强烈影响装配结构的振动响应，但其精确的数值处理在计算上十分昂贵。谐波平衡法被广泛用于计算周期稳态响应，然而其所需的交替频-时（AFT）格式对于非光滑和滞回非线性而言计算代价高昂，且必须在整个非线性求解过程中反复执行。本文证明这一过程可以被神经网络所取代：该网络直接将位移傅里叶系数映射为非线性力系数，并通过自动微分提供相应的雅可比矩阵。在计算频响曲线时，周围的求解器和延拓算法保持不变。这些神经网络仅学习单个非线性元件，而非完整的系统响应。基于物理的无量纲化和相位归一化便于……（摘要在此处截断）

    arXiv:2609.29242v1 Announce Type: cross  Abstract: Nonlinear contacts and friction strongly influence the vibration response of assembled structures, but their accurate numerical treatment is computationally demanding. The harmonic balance method is widely used to compute periodic steady-state responses, yet the required alternating frequency-time scheme becomes costly for nonsmooth and hysteretic nonlinearities and must be repeated throughout the nonlinear solution process. Here we show that this procedure can be replaced by neural networks that directly map displacement Fourier coefficients to nonlinear force coefficients and provide the corresponding Jacobian through automatic differentiation. The surrounding solver and continuation algorithms remain unchanged for the computation of frequency response curves. The neural networks exclusively learn individual nonlinear elements rather than complete system responses. Physics-based nondimensionalization and phase normalization facilitat
    
[^127]: 后训练会在无关决策上留下行为阴影

    Post-Training Leaves Behavioral Shadows on Unrelated Decisions

    [https://arxiv.org/abs/2609.29233](https://arxiv.org/abs/2609.29233)

    该论文提出主动无任务蒸馏（ATD）方法，证明后训练会在模型行为上留下可被探测的“阴影”——仅凭教师模型在任务无关提示中输出的单个单词，就能将编程等目标能力传递给学生模型。

    

    我们发现语言模型可以通过任务无关的文本传递能力。后训练通常使用任务特定的数据来改进语言模型。先前关于“潜意识学习”的研究表明，这些更新的信息可以通过无关的生成内容传递，但其研究主要集中于使用大量教师输出时的特质或偏好。我们提出了主动无任务蒸馏，仅使用教师模型在每个提示中输出的单个词即可实现能力传递。ATD通过选择教师模型和学生模型共同的公共祖先在两个普通词之间几乎无差异的提示，来探测后训练所留下的行为阴影。从这个祖先初始化的学生模型仅通过学习产生的提示-词对进行训练，无需目标任务示例、教师模型的logits或教师模型参数。在以Qwen2.5-1.5B进行的主要编程实验中，5,664个样本在HumanEval+上带来了5.34个百分点的提升。

    arXiv:2609.29233v1 Announce Type: cross  Abstract: We find that language models can transfer capabilities through task-unrelated text. Post-training typically improves language models using task-specific data. Prior work on subliminal learning shows that information about these updates can pass through unrelated generations, but has largely focused on traits or preferences using extensive teacher outputs. We introduce Active Taskless Distillation (ATD), which achieves capability transfer using only a single word from the teacher per prompt. ATD probes the behavioral shadow of post-training by selecting prompts where the teacher and student's shared public ancestor is nearly indifferent between two ordinary words. A student initialized from this ancestor learns solely from the resulting prompt-word pairs, without target-task examples, teacher logits, or teacher parameters. In the primary coding experiment with Qwen2.5-1.5B, 5,664nses yield a 5.34 pp gain on HumanEval+ over an exact nuis
    
[^128]: FB-GDM：基于无监督变分推断的全贝叶斯引导扩散模型，用于高维线性逆问题

    FB-GDM: Fully-Bayesian Guided Diffusion Models for High-Dimensional Linear Inverse Problems via Unsupervised Variational Inference

    [https://arxiv.org/abs/2609.29216](https://arxiv.org/abs/2609.29216)

    FB-GDM提出了一种全贝叶斯引导扩散方法，通过在每个反向扩散步骤中用变分推断自动估计两个精度参数，免除了针对具体任务且需依赖真值的人工超参数校准，同时借助可分离分解保持线性计算复杂度，成本与一次ΠGDM运行相当。

    

    扩散模型是线性逆问题的强大先验，但现有的参考引导方法——扩散后验采样（DPS）和伪逆引导扩散模型（ΠGDM）——依赖于需要针对每个任务调整的标量超参数，且通常需要借助真值来进行调节。我们提出了FB-GDM，一种完全贝叶斯的引导扩散方法，它消除了这一校准步骤。从ΠGDM的高斯近似出发，我们推导出了依赖于两个精度参数（即方差的倒数）的闭式条件分数，其中一个与去噪近似相关，另一个与观测似然相关，并将这两个参数视为潜变量，在每个反向扩散步骤中通过变分推断进行估计。一种可分离的分解方式使得每次更新的计算量与像素数量呈线性关系，因此推断在全图像分辨率下依然可以高效进行，其计算成本仅相当于运行一次ΠGDM。FB-GDM既不需要噪声水平信息，也不需要真值（原文此处被截断）……

    arXiv:2609.29216v1 Announce Type: cross  Abstract: Diffusion models are powerful priors for linear inverse problems, but the reference guidance methods, Diffusion Posterior Sampling (DPS) and Pseudoinverse-Guided Diffusion Models ($\Pi$GDM), rely on scalar hyperparameters tuned per task, usually against the ground truth. We introduce FB-GDM, a fully-Bayesian guided diffusion method that removes this calibration step. Starting from the Gaussian approximation of $\Pi$GDM, we derive a closed-form conditional score that depends on two precision parameters (inverse variances), one associated with the denoising approximation and one with the observation likelihood, and treat them as latent variables inferred by variational inference at each reverse step. A separable factorization makes each update scale linearly with the number of pixels, so the inference stays tractable at full image resolution, at a cost comparable to one $\Pi$GDM run. FB-GDM requires neither the noise level nor the ground
    
[^129]: 基于自适应边缘模型的移动机器人连续在线故障检测

    Continuous Online Fault Detection for Mobile Robots via Adaptive Edge Models

    [https://arxiv.org/abs/2609.29194](https://arxiv.org/abs/2609.29194)

    本文提出师生蒸馏框架，将离线基础模型TSPulse的故障检测能力蒸馏到轻量级MiniRocket学生模型中，并结合递归最小二乘在线自适应，实现了移动机器人在边缘硬件上的实时故障检测（4.30毫秒延迟），且在真实域偏移下无需灾难性遗忘即可恢复性能（VUS-PR从0.26提升至0.75）。

    

    移动机器人需要能够在受限的边缘硬件上持续适应的鲁棒实时故障检测能力。虽然深度时间序列模型在无监督异常检测方面表现出色，但其计算成本使得无法在机载设备上高频执行。本文通过师生蒸馏框架弥补了这一差距。离线基础模型（TSPulse）从经过故障注入增强的无标签时间序列中生成伪标签。轻量级的MiniRocket学生模型通过递归最小二乘估计器进行适配，逼近这一复杂的决策边界，从而在机载设备上执行实时推理。在TSB-AD基准和真实移动机器人上的评估表明，学生模型实现了4.30毫秒的CPU推理延迟。在真实世界的域偏移情况下，在线自适应使学生模型能够从未见过的机械退化中恢复，将VUS-PR分数从0.26提升至0.75，且未发生灾难性遗忘。

    arXiv:2609.29194v1 Announce Type: cross  Abstract: Mobile robots require robust, real-time fault detection capable of continuous adaptation on constrained edge hardware. While deep time-series models excel at unsupervised anomaly detection, their computational cost prohibits high-frequency onboard execution. This paper bridges this gap via a Teacher-Student distillation framework. An offline foundation model (TSPulse) generates pseudo-labels from unlabeled time series augmented with fault injections. A lightweight MiniRocket Student, adapted with a Recursive Least Squares estimator, approximates this complex decision boundary to execute real-time inference onboard. Evaluations on the TSB-AD benchmark and a physical mobile robot demonstrate the Student achieves a 4.30 ms CPU inference latency. During real-world domain shifts, online adaptation enables the Student to recover from unseen mechanical degradation, improving VUS-PR scores from 0.26 to 0.75 without catastrophic forgetting. Cru
    
[^130]: ASIRF：一个面向上下文相关敏感信息脱敏的智能体框架

    ASIRF: An Agentic Framework for Context-Dependent Sensitive Information Redaction

    [https://arxiv.org/abs/2609.29191](https://arxiv.org/abs/2609.29191)

    该论文提出ASIRF智能体框架，通过在推理时从知识库检索领域特定的敏感信息定义，无需重新训练即可适应新领域进行敏感信息脱敏，在85%的模型-领域组合中召回率超越了OpenAI隐私过滤器。

    

    敏感信息是由领域和意图定义的，而非一个通用类别，然而诸如隐私过滤器和命名实体识别器等脱敏系统在训练时便固定了分类体系，导致每进入一个新领域都需要重新训练。我们提出了ASIRF（智能体敏感信息脱敏框架），它在推理时根据输入所属领域从灵活的知识库中检索领域特定的定义，无需重新训练即可适应新领域。我们评估了两种架构——三次调用的多智能体流水线和单智能体变体——涵盖十个小型开源权重模型和八个数据集（包括分布外的虚构领域），并以OpenAI隐私过滤器（OPF）作为基于训练分类器的基线。每个领域仅需数十条专家撰写的定义且不使用任何训练数据，ASIRF在80个模型-领域组合中的68个（85%）上，召回率至少由两种架构之一超过了OPF

    arXiv:2609.29191v1 Announce Type: new  Abstract: Sensitive information is defined by domain and intent, not a universal category, yet redaction systems such as privacy filters and named-entity recognizers fix a taxonomy at training time, requiring retraining for each new domain. We introduce ASIRF (Agentic Sensitive Information Redaction Framework), which retrieves domain-specific definitions based on the input's domain from a flexible knowledge base at inference time, needing no retraining to adapt. Two architectures, a three-call multi-agent pipeline and a single-agent variant, are evaluated across ten small open-weight models and eight datasets, including out-of-distribution fictional domains, against the OpenAI Privacy Filter (OPF) as a trained-classifier baseline. With only a few dozen expert-authored definitions per domain and no training data, ASIRF's recall exceeds OPF's in 68 of 80 model-domain combinations (85 percent), by at least one of the two architectures, with shortfall
    
[^131]: 迈向可部署的水下舰船分类

    Towards Deployable Underwater Vessel Classification

    [https://arxiv.org/abs/2609.29179](https://arxiv.org/abs/2609.29179)

    该论文提出了一种结合多表示特征工程与紧凑卷积架构的水下舰船声学分类框架，并证明在严格的录音级数据划分协议下，仅15.7万参数的紧凑CNN即可达到0.7226的宏F1，性能不逊于参数量大70多倍的ResNet18。

    

    我们提出了一种紧凑的水声分类框架，该框架结合了多表示特征工程、时间统计池化以及为声学时频和耳蜗表示而设计的紧凑卷积架构。我们研究了多种传统及听觉启发的表示方法，并首先在ShipsEar数据集上评估了轻量级分类器和卷积神经网络（CNN）。在所提供的数据划分上，一个两层CNN达到了0.9918的宏F1分数，而径向基函数支持向量机（RBF-SVM）达到了0.9883。然而，由于无法重构源录音的来源信息，无法验证其与录音无关的泛化能力。因此，我们在DeepShip数据集上采用分段前录音级划分的评估协议进行评估。在该协议下，一个仅15.7万参数的紧凑CNN取得了0.7226的测试宏F1，而拥有1117万参数的ResNet18并未带来性能提升。

    arXiv:2609.29179v1 Announce Type: cross  Abstract: We propose a compact underwater acoustic classification framework combining multi-representation feature engineering, temporal statistical pooling, and compact convolutional architectures designed for acoustic time-frequency and cochlear representations. We investigate multiple conventional and auditory-inspired representations and first evaluate lightweight classifiers and Conventional Neural Networks (CNNs) on ShipsEar dataset. On the provided split, a two-layer CNN achieves a macro F1 of 0.9918, while a Radial Basis Function Support Vector Machine (RBF-SVM) reaches 0.9883. However, source-recording provenance cannot be reconstructed, preventing verification of recording-independent generalisation. We therefore evaluate on DeepShip dataset using recording-level partitioning before segmentation. Under this protocol, a 157K-parameter compact CNN achieves a test macro F1 of 0.7226, while an 11.17M-parameter ResNet18 provides no improvem
    
[^132]: 面向动态环境中二分类睡眠-清醒状态检测的受限设备边缘人工智能

    Edge AI on Constrained Devices for Binary Sleep-Wake Classification in Dynamic Environments

    [https://arxiv.org/abs/2609.29163](https://arxiv.org/abs/2609.29163)

    该论文在ESP32-S3微控制器上实现了结合惯性传感与视觉姿态分类的多模态边缘AI系统，采用两阶段睡眠检测策略，在动态环境中实现睡眠-清醒二分类，运动检测准确率达96.5%，姿态分类准确率达89%。

    

    本文提出了一种基于边缘人工智能的系统，用于在资源受限的嵌入式硬件上检测非平稳移动环境中的睡眠与清醒状态。传统方法依赖于基于加速度计的活动度量，极易受到运动和振动伪影的干扰，并受限于可穿戴设备与物联网设备严格的计算和能耗预算。为应对这些挑战，研究者在ESP32-S3微控制器上设计并实现了一个多模态处理流程。该系统结合了用于头部运动分析的惯性传感和视觉姿态分类。基于FreeRTOS的双核架构实现了实时数据采集与设备端推理的并行执行。睡眠检测采用两阶段策略：首先在时间窗口内检测低运动状态，随后通过视觉验证姿态。实验结果显示，基于运动的检测准确率达到96.5%，姿态分类准确率达到89%。

    arXiv:2609.29163v1 Announce Type: new  Abstract: This paper presents an Edge AI-based system for detecting sleep and wake states in non-stationary mobile environments using resource-constrained embedded hardware. Conventional approaches relying on accelerometer-based activity metrics are highly susceptible to motion and vibration artifacts and are limited by strict compute and energy budgets of wearable and IoT devices. To address these challenges, a multimodal pipeline is designed and implemented on an ESP32-S3 microcontroller.   The system combines inertial sensing for head movement analysis and visual pose classification. A dual-core architecture with FreeRTOS enables parallel execution of real-time data acquisition and on-device inference. Sleep detection follows a two-stage strategy: low-movement detection over a temporal window, followed by visual validation of poses.   Experimental results show accuracies of 96.5% for motion-based detection and 89% for pose classification, yield
    
[^133]: 函数型动态模态分解：从数据中学习无限维系统

    Functional dynamic mode decomposition: Learning infinite-dimensional systems from data

    [https://arxiv.org/abs/2609.29159](https://arxiv.org/abs/2609.29159)

    本文提出了函数型动态模态分解（DMD），将投影DMD和精确DMD从有限维扩展到无限维系统，无需对空间域进行离散化即可直接从数据中学习偏微分方程等无限维动力系统。

    

    动态模态分解（DMD）是一种数据驱动方法，它计算底层动力系统的最佳线性逼近，并将动力学分解为特征时空模式的叠加。DMD最初由流体力学领域提出，此后其本身及各种扩展方法已在分子动力学、气候科学、工程、金融和神经科学等众多研究领域得到广泛应用，应用场景包括降维、预测、系统辨识、控制和谱聚类等。为了将DMD应用于偏微分方程，通常需要先使用有限差分或有限元技术对空间域进行离散化，从而隐式地将问题转化为有限维问题。本文将投影DMD和精确DMD扩展到了无限维系统：我们的DMD变体不再从向量值观测中估计矩阵，而是学习有限秩的算子（摘要在此处截断）。

    arXiv:2609.29159v1 Announce Type: cross  Abstract: Dynamic mode decomposition (DMD) is a data-driven method that computes the best linear approximation of the underlying dynamical system and decomposes the dynamics into a superposition of characteristic spatiotemporal patterns. Originally introduced by the fluid dynamics community, DMD and its extensions have found widespread use in many other research areas such as molecular dynamics, climate science, engineering, finance, and neuroscience. Applications include dimensionality reduction, forecasting, system identification, control, and spectral clustering. In order to apply DMD to partial differential equations, the spatial domain is typically first discretized using finite difference or finite element techniques, thus implicitly rendering the problem finite-dimensional. We extend projected and exact DMD to infinite-dimensional systems. Rather than estimating matrices from vector-valued observations, our DMD variants learn finite-rank 
    
[^134]: 一种粒子群辅助的梯度元学习算法：用于发射预编码与STAR-RIS系数的联合优化

    A Particle-Swarm-Assisted Gradient Meta-Learning Algorithm for Joint Transmit Precoding and STAR-RIS Coefficient Optimization

    [https://arxiv.org/abs/2609.29150](https://arxiv.org/abs/2609.29150)

    本文提出粒子群辅助的梯度元学习（PSA-GML）算法，通过幅度分离参数化降低搜索维度，并结合PSO全局搜索与闭式预编码求解，联合优化发射预编码与STAR-RIS系数以最大化多用户下行链路的加权和速率。

    

    本文研究了多用户下行链路中发射预编码器与同时透射和反射可重构智能表面（STAR-RIS）透射/反射系数的联合优化问题，旨在最大化加权和速率（WSR）。针对这一非凸问题，我们提出了一种粒子群辅助的梯度元学习（PSA-GML）算法。首先，通过幅度分离参数化和折叠预编码器表示对原问题进行等价变换，从而自动满足能量守恒约束并降低搜索维度。随后，利用粒子群优化（PSO）对STAR-RIS系数进行全局搜索，提供高质量且对初始化鲁棒的暖启动，而发射预编码器则以闭式解获得。区别于传统的交替优化（AO）方法，本文采用逐坐标的长短期记忆（LSTM）元优化器进行训练（摘要在此处被截断）。

    arXiv:2609.29150v1 Announce Type: new  Abstract: This paper investigates the joint optimization of the transmit precoder and the transmission/reflection coefficients of a simultaneously transmitting and reflecting reconfigurable intelligent surface (STAR-RIS) to maximize the weighted sum rate (WSR) in a multi-user downlink. We propose a particle-swarm-assisted gradient meta-learning (PSA-GML) algorithm for this non-convex problem. The original problem is first equivalently transformed via an amplitude-split parameterization and a collapsed precoder representation, which automatically satisfy the energy-conservation constraint and reduce the search dimension. Particle swarm optimization (PSO) then performs a global search over the STAR-RIS coefficients to yield a high-quality, initialization-robust warm start, with the transmit precoder obtained in closed form. Departing from conventional alternating optimization (AO), a coordinate-wise long short-term memory (LSTM) meta-optimizer train
    
[^135]: 并非每个Token都值得蒸馏：面向Direct-OPD的选择性监督

    Not Every Token Is Worth Distilling: Selective Supervision for Direct-OPD

    [https://arxiv.org/abs/2609.29142](https://arxiv.org/abs/2609.29142)

    该论文揭示了Direct-OPD中token级对数比率监督无法反映教师行为真实变化的缺陷，并提出根据教师参考JSD进行选择性屏蔽监督的方法S²D-OPD，仅在教师行为发生显著变化的token上进行蒸馏。

    

    直接在线策略蒸馏通过将强化学习后与强化学习前检查点之间的token级对数比率，作为学生模型自身生成轨迹上的密集监督信号，把强化学习带来的策略改进从一个小模型迁移到一个更大的学生模型。这种迁移方式在每个状态下都会奖励策略的变化，然而对数比率仅衡量相对变化：即使两个检查点分配给学生模型候选token的概率质量趋于消失，对数比率仍可能保持不变。通过一个精确的构造，我们证明当检查点之间的Jensen-Shannon散度（JSD）以及两个方向的KL散度随着该概率质量一同消失时，Direct-OPD的奖励及其更新仍可保持不变，同时我们指出较小的JSD能够约束教师模型行为变化的幅度。基于这一分析，我们提出了面向Direct-OPD的选择性监督方法（S²D-OPD），它根据教师参考JSD对学生模型采样得到的状态进行排序并屏蔽……（摘要原文不完整）

    arXiv:2609.29142v1 Announce Type: cross  Abstract: Direct On-Policy Distillation (Direct-OPD) transfers reinforcement-learning-induced policy improvements from a small model to a larger student by using the token-level log-ratio between post-RL and pre-RL checkpoints as dense supervision on the student's own rollouts. This transfer rewards the policy shift at every state, yet the log-ratio measures only relative change: it can stay fixed even as the probability mass that both checkpoints assign to the student's candidate tokens vanishes. Through an exact construction, we show that the Direct-OPD reward and its update can remain unchanged while the Jensen-Shannon divergence (JSD) and both KL directions between the checkpoints vanish with this mass, and we note that a small JSD bounds how much the teacher's behavior changed. Motivated by this analysis, we propose Selective Supervision for Direct-OPD (S$^2$D-OPD), which ranks student-sampled states by their teacher-reference JSD and masks
    
[^136]: 双时间尺度Actor-Critic算法的集中界

    A Concentration Bound for Two-Timescale Actor-Critic Algorithm

    [https://arxiv.org/abs/2609.29117](https://arxiv.org/abs/2609.29117)

    本文为长期平均奖励设定下带函数逼近的双时间尺度Actor-Critic算法推导了一致的全时间集中界，证明Actor参数在有限时间后以高概率进入并保持在一个安全区域，且其与最优参数的误差以至少 $1-\epsilon_1-\epsilon_2$ 的概率满足给定的收敛速率上界。

    

    近年来，大量研究工作致力于为双时间尺度Actor-Critic算法建立渐近与非渐近收敛保证，其中Actor递归以比Critic递归更慢的时间尺度运行。本工作在长期平均奖励设定下，为带函数逼近的Actor-Critic算法推导了一致的全时间集中界。该界有助于我们以高概率分析Actor参数的行为。我们证明，在某个有限时间之后，Actor参数以高概率进入一个安全区域，并在其后始终保持在该区域内。具体而言，以至少 $1-\epsilon_1-\epsilon_2$ 的概率，对所有 $k\geq n_0$（原文此处截断），Actor误差 $\Vert \theta_k-\theta^{*}\Vert$ 满足 $O\left(\frac{n_0^{3/4}}{k}\frac{1}{\sqrt{\epsilon_2}}+\left(\frac{1}{n_0}\right)^{1/4}\log^{1/4}\left(\frac{1}{\epsilon_1}\right)+\left(\frac{1}{n_0}\right)^{1/4}\right)$。

    arXiv:2609.29117v1 Announce Type: new  Abstract: Significant research effort has been directed in recent years towards establishing both asymptotic and non-asymptotic convergence guarantees for two-timescale actor--critic algorithms, where the actor recursion is run on a slower timescale than the critic recursion. This work derives a uniform all-time concentration bound for the actor--critic algorithm with function approximation in the long-run average-reward setting. This bound helps us analyze the behavior of the actor parameter with high probability. We show that, after some finite time, the actor parameter enters a safe region and remains within it thereafter with high probability. Specifically, with probability at least $1-\epsilon_1-\epsilon_2$, the actor error $\Vert \theta_k-\theta^{*}\Vert$ is $O\left(\frac{n_0^{3/4}}{k}\frac{1}{\sqrt{\epsilon_2}}+\left(\frac{1}{n_0}\right)^{1/4}\log^{1/4}\left(\frac{1}{\epsilon_1}\right)+\left(\frac{1}{n_0}\right)^{1/4}\right)$ for all $k\geq
    
[^137]: ELF-REG：将连续扩散语言模型扩展至推理任务

    ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks

    [https://arxiv.org/abs/2609.29102](https://arxiv.org/abs/2609.29102)

    提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。

    

    全连续扩散语言模型（dLMs）对连续表示进行去噪而无需中间离散化，并在最后一步并行解码所有响应token。它们在具有挑战性的推理任务上的性能表现，尚未像自回归（AR）大语言模型和掩码扩散语言模型那样得到充分验证。我们将嵌入式语言流（ELF）扩展到GSM8K、MATH-500、HumanEval和MBPP上的数学推理与代码生成任务。我们提出了ELF-REG，它通过表示对齐与纠缠（REPA+REG）来改进学习，其中冻结的AR教师模型监督中间去噪器特征，并提供一个与响应联合去噪的全局表示。ELF-REG-L在64次网络函数评估（NFE）下于GSM8K上达到55.96%的pass@1，在128次NFE下于MATH-500上达到13.39%、HumanEval上达到22.56%。它在GSM8K和代码任务上的pass@1优于所评估的同规模扩散语言模型，并将MATH-500的pass@1从……（摘要原文在此处被截断）

    arXiv:2609.29102v1 Announce Type: new  Abstract: Fully continuous diffusion language models (dLMs) denoise continuous representations without intermediate discretization, then decode all response tokens in parallel at the final step. Their performance on challenging reasoning tasks remains less established than that of autoregressive (AR) LLMs and masked dLMs. We scale Embedded Language Flows (ELF) to mathematical reasoning and code generation on GSM8K, MATH-500, HumanEval, and MBPP. We introduce ELF-REG, which improves learning with representation alignment and entanglement (REPA+REG), where a frozen AR teacher supervises intermediate denoiser features and supplies a global representation that is jointly denoised with the response. ELF-REG-L achieves 55.96% pass@1 on GSM8K at 64 network function evaluations (NFE), and 13.39% on MATH-500 and 22.56% on HumanEval at 128 NFE. It outperforms the evaluated comparable-scale dLMs in pass@1 on GSM8K and code, and improves MATH-500 pass@1 from 
    
[^138]: 语言特异性与领域多样性之权衡：面向孟加拉语医学命名实体识别的Transformer基准测试

    Language Specificity vs. Domain Diversity: Benchmarking Transformers for Bangla Medical NER

    [https://arxiv.org/abs/2609.29101](https://arxiv.org/abs/2609.29101)

    该研究在3,179个样本的完整测试集上对孟加拉语医学命名实体识别进行了大规模基准测试，微调后的XLM-RoBERTa以0.5959的F1分数创造了新的最先进水平，并揭示语言专用的BanglaBERT表现持续不及多语言模型。

    

    针对低资源语言的医学命名实体识别（NER）任务，由于语言变异性高且领域专用标注语料稀缺，仍然是一项具有挑战性的工作。本文提出了一项全面的实证基准测试，评估了三个经过微调的transformer编码器——BanglaBERT、多语言BERT（mBERT）和XLM-RoBERTa——与GPT-4o mini在零样本和少样本提示配置下进行孟加拉语医学NER的性能对比。与以往仅在仅50个样本的有限子集上评估大语言模型的研究不同，我们在包含3,179个样本的完整测试集上进行了大规模评估，提供了统计上稳健且可复现的基线结果。我们微调的XLM-RoBERTa模型达到了0.5959的F1分数，创造了新的最先进水平（state-of-the-art），超越了此前报道的最佳结果0.5848。至关重要的是，我们证明了语言专用的BanglaBERT模型的表现始终不及其多语言模型（摘要在此处截断）。

    arXiv:2609.29101v1 Announce Type: new  Abstract: Medical Named Entity Recognition (NER) for low-resource languages remains a challenging task due to high linguistic variability and a scarcity of domain-specific annotated corpora. This work presents a comprehensive empirical benchmark evaluating three fine-tuned transformer encoders-BanglaBERT, multilingual BERT (mBERT), and XLM-RoBERTa-against GPT-4o mini under zero-shot and few-shot prompting configurations for Bangla medical NER. In contrast to prior studies that evaluated large language models on limited subsets of only 50 samples, we conduct a large-scale evaluation across the full test set of 3,179 samples, providing statistically robust and reproducible baselines. Our fine-tuned XLM-RoBERTa model achieves an F1- score of 0.5959, establishing a new state-of-the-art and surpassing the previously reported best result of 0.5848. Crucially, we demonstrate that the language-specific BanglaBERT model consistently underperforms its multi
    
[^139]: TraceGuard：通过跨特征排名一致性实现自适应多模态投毒过滤

    TraceGuard: Adaptive Multimodal Poison Filtering through Cross-Feature Rank Agreement

    [https://arxiv.org/abs/2609.29099](https://arxiv.org/abs/2609.29099)

    TraceGuard通过六个语料库级特征（跨模态邻域、重复文本、文本擦除变化等）分析投毒样本的集体影响模式，并利用跨特征排名一致性自适应地过滤多模态数据中的隐蔽投毒样本，全程无需训练受害模型。

    

    多模态训练依赖于从外部来源收集的图文语料库，这为攻击者对数据进行投毒创造了机会。隐蔽攻击可以在保留看似合理的图文对的同时，隐藏检测器所依赖的差异特征，因此表面干净的数据仍然能够改变训练后模型的行为。因此，我们研究投毒集必须保留哪些属性才能使攻击保持有效。一个小规模的投毒集仍然必须在训练过程中施加足够的集体影响，才能诱导攻击者期望的目标行为。我们从攻击模式出现的频率以及携带该模式的样本对模型产生的联合影响强度两个方面来分析这种影响。这一分析启发了六个语料库级特征，用于检查跨模态邻域关系、重复出现的文本模式以及文本片段擦除后的变化，而无需训练受害模型。我们提出了TraceGuard，一种自适应的基于排名的过滤方法，它利用互补特征之间的一致性来识别投毒样本。

    arXiv:2609.29099v1 Announce Type: cross  Abstract: Multimodal training relies on image-text corpora collected from external sources, creating opportunities for attackers to poison the data. Stealthy attacks can preserve plausible image-text pairs while concealing the differences used by detectors, so apparently clean data can still redirect the trained model. We therefore ask which properties a poison set must preserve for the attack to remain effective. A small poison set must still exert enough collective influence during training to induce the attacker's target behavior. We analyze this influence in terms of how often an attack pattern occurs and how strongly the examples carrying it jointly affect the model. This analysis motivates six corpus-level features that examine cross-modal neighborhoods, recurring text, and changes after text-span erasure without training the victim model. We introduce TraceGuard, an adaptive rank-based filtering method that uses agreement among complement
    
[^140]: 延迟与修正结果下的下行风险可控在线预测组合

    Downside-Controlled Online Forecast Combination under Delayed and Revised Outcomes

    [https://arxiv.org/abs/2609.29096](https://arxiv.org/abs/2609.29096)

    该论文提出一种下行风险可控的在线预测组合方法，在延迟与修正的结果环境下将冻结预测器、静态修正器与在线修正器在单纯形上组合，仅使用成熟损失，实现最坏劣化仅0.15%且最高收益达11.5%，并在欧洲全部七个日前负荷竞价区域均降低MSE。

    

    事后修正适用于无法重新训练的预测器（例如基础模型），但在误差稳定处拟合的修正器，在误差发生漂移时反而可能造成损害。我们的目标是实现下行控制：性能不会比初始预测差太多。该方法在单纯形上组合冻结的预测器、静态修正器与在线修正器，仅使用在地平线之后才成熟（可用）的损失。在七个基准任务和四个基础模型（其中两个为基础模型）上，主地平线处28对组合中的最坏劣化仅为0.15%，而收益最高可达11.5%。在欧洲七个竞价区域的日前负荷预测任务中，该方法在全部七个区域均降低了平均MSE，而单一修正器在已发布预测最准确的区域会将平均MSE推高至多102%。三个关于专家速度、数据流长度和结果对齐的实证条件（每个条件均由一个记录在案的失败案例确定）划定了该方法的适用范围。利用临时（初步）结果进行学习可以在结算结果上改善四个区域的表现。

    arXiv:2609.29096v1 Announce Type: new  Abstract: Post-hoc correction adjusts a forecaster that cannot be retrained, such as a foundation model, but a correction fitted where errors are stable can hurt where they shift. We aim for downside control: not much worse than the starting forecast. We combine the frozen forecaster, a static corrector and an online corrector on the simplex, using only losses that mature after the horizon. Across seven benchmarks and four base models, two of them foundation models, the worst deterioration over 28 pairs at the main horizon is 0.15% and gains reach 11.5%. On day-ahead load for seven European bidding zones it lowers mean MSE in all seven zones, while single correctors raise mean MSE by up to 102% where the published forecast is most accurate. Three empirical conditions on expert speed, stream length and outcome alignment, each fixed by a documented failure, delimit its scope. Learning from the provisional outcome improves four zones on the settled o
    
[^141]: 恰好一次语义由谁承担？模型、智能体框架与工具契约对 LLM 智能体重复副作用的影响

    Where Does Exactly-Once Live? Model, Harness, and Tool-Contract Effects on Duplicate Side Effects in LLM Agents

    [https://arxiv.org/abs/2609.29095](https://arxiv.org/abs/2609.29095)

    该论文提出确定性沙盒基准 LIMBO，研究 LLM 智能体的“恰好一次”副作用语义应由模型、智能体框架还是工具契约来保障，并发现答案取决于故障类型：当即时回读能够揭示实际结果时，由模型来决定。

    

    当使用工具的智能体的写操作超时或返回服务器错误时，该操作可能已经实际生效。盲目重试会导致重复执行——第二次扣款、第二次公告、第二次部署——而放弃重试则会跳过必要的工作。我们提出问题：恰好一次（exactly-once）行为应该在哪里强制执行：在模型中、在智能体框架（harness）中，还是在工具契约中？我们介绍了 LIMBO，一个由六个服务组成的确定性沙盒，这些服务具有真实的契约（可选的幂等键、最终一致性和缺失的读取路径），并在服务边界注入了十二种故障模式，包括延迟提交、重复投递和部分批次；每个回合都根据已提交效果的账本进行评分。在涵盖九个近期模型、三个生产级智能体框架、两种契约变体和十五种恢复条件共 25,930 个回合的实验中，答案取决于具体的故障类型。当即时回读能够揭示发生了什么时，由模型来决定……

    arXiv:2609.29095v1 Announce Type: cross  Abstract: When a tool-using agent's write times out or returns a server error, the action may already have taken effect. Retrying blindly duplicates it -- a second charge, a second announcement, a second deployment -- while giving up skips required work. We ask where exactly-once behaviour should be enforced: in the model, in the agent harness, or in the tool contract. We introduce LIMBO, a deterministic sandbox of six services with realistic contracts (optional idempotency keys, eventually consistent and missing read paths) and twelve fault modes injected at the service boundary, including late commits, redelivery and partial batches; every episode is graded against a ledger of committed effects. Across 25,930 episodes spanning nine recent models, three production agent harnesses, two contract variants and fifteen recovery conditions, the answer depends on the fault. When an immediate read-back can reveal what happened, the model decides: front
    
[^142]: DAWN：通过深度去噪世界模型实现噪声鲁棒的四足跑酷

    DAWN: Noise-Robust Quadruped Parkour via Depth-Denoising World Models

    [https://arxiv.org/abs/2609.29092](https://arxiv.org/abs/2609.29092)

    该论文提出DAWN框架，通过让世界模型以带噪深度为输入、干净深度为重建目标实现隐式去噪，并结合对比学习对齐，将噪声鲁棒性直接内置到腿式运动的深度感知中，从而无需手工调整的滤波器即可实现鲁棒的四足跑酷。

    

    基于视觉的腿式运动方法通常假设训练时的深度数据是干净的，并在部署时依赖手工调整的后处理滤波器。然而，滤波器参数很少被公开，阻碍了可复现性，且当深度噪声未被处理时性能会大幅下降。将噪声鲁棒性直接构建到学习流程中可以消除这种依赖。虽然这种鲁棒性已在本体感知输入中被探索过，但在腿式运动领域，针对深度感知的类似方法仍然很大程度上缺失。我们提出了DAWN（世界模型中的去噪与对齐以实现噪声鲁棒性），这是一个面向腿式运动的噪声鲁棒感知框架，它通过两项修改将噪声鲁棒性直接内置到世界模型中：（1）将带噪声的深度图输入编码器，同时以干净的深度图作为重建目标，迫使模型对其输入进行隐式去噪；（2）应用对比学习来对齐（原文此处截断）。

    arXiv:2609.29092v1 Announce Type: cross  Abstract: Vision-based legged locomotion methods assume clean depth at training time and rely on hand-tuned post-processing filters at deployment. However, filter parameters are rarely disclosed, hindering reproducibility, and performance degrades substantially when depth noise is left unaddressed. Building noise robustness directly into the learning pipeline would eliminate this dependency. While such robustness has been explored for proprioceptive inputs, analogous approaches for depth perception remain largely absent in legged locomotion. We propose DAWN (Denoising and Alignment in World models for Noise-robustness), a noise-robust perception framework for legged locomotion, which builds noise robustness directly into a world model via two modifications: (1) feeding noisy depth to the encoder while keeping clean depth as the reconstruction target, forcing the model to implicitly denoise its input; and (2) applying contrastive learning to alig
    
[^143]: 物理与数据驱动的Transformer-Mamba流场框架

    Physics and Data Driven Transformer-Mamba Framework for Flow Field

    [https://arxiv.org/abs/2609.29087](https://arxiv.org/abs/2609.29087)

    该论文提出TM4FF框架，通过残差小波Mamba去噪层、Transformer注意力机制和基于傅里叶导数的纳维-斯托克斯方程物理约束损失三项创新，在CFD流场求解中实现了高精度、强噪声鲁棒性和良好的跨工况泛化能力。

    

    虽然深度学习加速了计算流体力学（CFD）中昂贵的偏微分方程求解，但现有方法如物理信息神经网络（PINNs）和傅里叶神经算子（FNOs）在泛化能力、噪声鲁棒性和物理一致性方面往往存在不足。我们提出了流场Transformer-Mamba（TM4FF）框架，这是一个物理约束的算子学习模型，具有三项关键创新：用于特征去噪的残差小波Mamba（RWM）层、用于增强特征融合的基于Transformer的注意力机制，以及利用傅里叶导数强制执行纳维-斯托克斯方程的物理信息损失函数。在四个CFD数据集上的实验表明，TM4FF在不同流动条件下实现了高精度和鲁棒的泛化能力。

    arXiv:2609.29087v1 Announce Type: new  Abstract: While deep learning accelerates expensive partial differential equation solving in computational fluid dynamics (CFD), existing methods like PINNs and FNOs often struggle with generalization, noise robustness, and physical consistency. We introduce the Transformer-Mamba for Flow Field (TM4FF) framework, a physics-constrained operator learning model with three key innovations: a Residual Wavelet Mamba (RWM) layer for feature denoising, a Transformer-based attention mechanism for enhanced feature fusion, and a physics-informed loss using Fourier derivatives to enforce the Navier-Stokes equations. Experiments on four CFD datasets show TM4FF achieves high accuracy and robust generalization across varying flow conditions.
    
[^144]: 一种在WeBe手环上快速训练与部署机器学习模型的流水线

    A Rapid Pipeline for Training and Deploying ML Models on WeBe Band

    [https://arxiv.org/abs/2609.29084](https://arxiv.org/abs/2609.29084)

    本文提出一个集成开源Piccolo AI生态系统的自动化快速流水线，能够在资源受限的WeBe手环上快速开发、优化并部署满足延迟、内存和功耗约束的机器学习模型。

    

    针对计算和内存资源受限的边缘设备开发优化的机器学习算法是具有挑战性、耗时且高度依赖设备特定约束的工作。在这项工作中，我们简化了边缘机器学习工作流程，实现了机器学习（ML）模型直接在WeBe Band上的快速开发、优化和部署，WeBe Band是一款专为多模态生理数据监测设计的腕戴式可穿戴设备。所提出的系统自动生成硬件高效的机器学习模型，这些模型可以轻松集成到WeBe核心固件中，支持AutoML、硬件感知量化和性能分析，以构建满足预期延迟目标、同时兼容设备内存和功耗限制的模型。该框架将开源的Piccolo AI生态系统与自动化流水线紧密集成，可生成可部署的固件制品，并执行硬件感知

    arXiv:2609.29084v1 Announce Type: new  Abstract: Developing optimized machine-learning algorithms for edge devices with limited computational and memory resources is challenging, time-consuming, and highly dependent on device-specific constraints. In this work, we streamline an edge ML workflow to enable rapid development, optimization, and deployment of machine-learning (ML) models directly on the WeBe Band, a wrist-worn wearable device designed for multimodal physiological data monitoring. The proposed system automatically generates hardware-efficient ML models that can be easily integrated into the WeBe core firmware, supporting AutoML, hardware-aware quantization, and performance profiling to build models that meet desired latency targets while remaining compatible with device memory and power limitations.   The proposed framework tightly integrates the open-source Piccolo AI ecosystem with an automated pipeline that generates deployable firmware artifacts, performs hardware-aware 
    
[^145]: 血脑屏障通透性的特征空间选择与异质性效应估计：从随机森林到广义随机森林的流程

    Feature Space Selection and Heterogeneous Effect Estimation for Blood-Brain Barrier Permeability: A Random Forest to the Generalized Random Forest Pipeline

    [https://arxiv.org/abs/2609.29076](https://arxiv.org/abs/2609.29076)

    本研究通过系统性消融实验比较多种分子特征空间与算法组合，发现基于组合特征的动态随机森林在血脑屏障通透性预测中取得最高 AUC（0.970），并进一步结合广义随机森林与双重/去偏机器学习，探索性地估计了分子结构与 BBB 通透性之间的异质性关联。

    

    预测血脑屏障（BBB）通透性对中枢神经系统药物发现至关重要。本研究使用 MoleculeNet BBBP 数据集（n = 2039），系统地对分子特征空间进行消融分析，以将特征化方法与模型架构的影响相互分离。我们在四种学习算法上评估了三类特征家族（Morgan 指纹、RDKit 理化描述符、SMILES 二元组）。结果表明，预测性能同时取决于特征表示与算法的选择。使用组合特征的动态随机森林取得了最高的平均 AUC（0.970，95% 置信区间：0.963–0.977）。其次，基于这一最优特征表示，我们利用广义随机森林对分子结构与 BBB 通透性之间的异质性关联进行了探索性估计。我们通过 LogP 的中位数分割构建伪处理变量，并应用双重/去偏机器学习方法来控制混杂因素。正交化处理显著地（原文摘要在此处截断）

    arXiv:2609.29076v1 Announce Type: cross  Abstract: Predicting blood-brain barrier (BBB) permeability is critical for central nervous system drug discovery. Using the MoleculeNet BBBP dataset (n = 2039), this study systematically ablates molecular feature spaces to isolate featurisation from model architecture. We evaluate three feature families (Morgan fingerprints, RDKit physicochemical descriptors, SMILES bigrams) across four learning algorithms. Results demonstrate that predictive performance depends jointly on feature representation and algorithm. Dynamic Random Forest using combined features achieved the highest mean AUC (0.970, 95% CI: 0.963-0.977). Second, this optimal representation enables exploratory estimation of heterogeneous associations between molecular structure and BBB permeability using Generalized Random Forests. Constructing a pseudo-treatment from a LogP median split, we applied double/debiased machine learning to account for confounding. Orthogonalization substant
    
[^146]: BranchShine-CR：基于自条件CTC与一致性正则化的紧凑型多语言国际音标转录

    BranchShine-CR: Compact Multilingual IPA Transcription with Self-Conditioned CTC and Consistency Regularization

    [https://arxiv.org/abs/2609.29069](https://arxiv.org/abs/2609.29069)

    BranchShine-CR是一个仅2500万参数的多语言国际音标转录模型，通过自条件CTC与一致性正则化等技术，以约十二分之一的参数量实现4.47%的IPA字符错误率并相对降低22.3%，在全部41种语言上超越同规模NeMo Conformer基线，适用于低资源设备端发音评估。

    

    我们提出了BranchShine-CR，一个用于多语言转录为国际音标（IPA）的2500万参数模型。它结合了log-mel特征、旋转位置编码的E-Branchformer编码器、中间层自条件连接时序分类（CTC）以及跨增强视图的一致性正则化。在16,646条共享的IPApack++测试语句上，该模型实现了4.47%的IPA字符错误率，相比ZIPA-CTC-NS相对降低了22.3%，而参数量仅约为其十二分之一，且完全从零开始训练。BranchShine-CR还在全部41个数据集语言标签上超越了规模相近的NeMo Conformer基线。消融实验表明，各组件在模型性能贡献上协同发挥作用。这些发现证明了在有限计算预算下实现紧凑型IPA识别的能力，可应用于低资源设备端发音评估场景。

    arXiv:2609.29069v1 Announce Type: new  Abstract: We introduce BranchShine-CR, a 25M-parameter model for multilingual transcription into the International Phonetic Alphabet (IPA). It combines log-mel features, a rotary-position E-Branchformer encoder, intermediate self-conditioned connectionist temporal classification (CTC), and consistency regularization across augmented views. On 16,646 shared IPApack++ test utterances, it achieves 4.47% IPA character error rate, a 22.3% relative reduction from ZIPA-CTC-NS, with approximately one-twelfth as many parameters while being trained from scratch. BranchShine-CR also outperforms a similarly sized NeMo Conformer baseline across all 41 dataset language labels. Ablation studies indicate the individual components synergetically acting in model performance contribution. These findings support compact IPA recognition capabilities under limited compute budget, for applications in low-resource on-device pronunciation assessment.
    
[^147]: Transformer作为跨任务学习器：共享结构驱动上下文学习中的样本效率

    Transformers as Cross-Task Learners: Shared Structure Drives Sample Efficiency in In-Context Learning

    [https://arxiv.org/abs/2609.29060](https://arxiv.org/abs/2609.29060)

    本文通过覆盖数刻画任务空间的复杂度，揭示了Transformer如何利用共享的跨任务结构提升上下文学习的样本效率，并提出了一种基于锚函数的任务识别与评估方法。

    

    Transformer通过在预训练期间联合学习广泛的任务族，并仅凭简短的提示就能适应未见过的任务，从而取得了卓越的性能。然而，对这一现象的严格数学和统计学理解仍然有限。本文旨在研究Transformer如何利用共享的跨任务结构，以及这种结构如何影响上下文学习（ICL）的样本复杂度。具体而言，我们通过在规定度量下的覆盖数来刻画任务空间复杂度，从而在无需显式参数化表示的情况下量化低维跨任务结构。所得的覆盖提供了一组锚函数，我们利用它们提出了任务识别与评估程序：通过上下文观测在锚函数中定位一个未见过的任务，再通过聚合相应锚函数在查询点上的评估值来预测响应。

    arXiv:2609.29060v1 Announce Type: cross  Abstract: Transformers achieve remarkable performance by jointly learning broad families of tasks during pretraining and adapting to unseen tasks from only a short prompt. Yet a rigorous mathematical and statistical understanding of this phenomenon remains limited. This paper aims to study how Transformers exploit shared cross-task structure and how this structure affects the sample complexity of in-context learning (ICL). Specifically, we characterize task-space complexity through covering numbers under a prescribed metric, thereby quantifying the low-dimensional cross-task structure without requiring an explicit parametric representation. The resulting cover provides a set of anchor functions, which we use to introduce a task-identification-and-evaluation procedure: context observations localize an unseen task among the anchor functions, and the response at a query is predicted by aggregating the corresponding anchor function query evaluations
    
[^148]: SLCA-GRPO：解决工具调用强化学习中的跨段信用错误归因问题

    SLCA-GRPO: Resolving Cross-Segment Credit Misattribution in Tool-Calling RL

    [https://arxiv.org/abs/2609.29050](https://arxiv.org/abs/2609.29050)

    提出SLCA-GRPO框架，通过段锁定信用分配在结构段级别解耦优势估计，解决工具调用强化学习中跨段信用错误归因问题，并配套构建模式引导的LLM模拟器（SGLS）以实现无需真实API的可扩展稳定训练。

    

    工具调用智能体产生的输出具有异构性，将结构化的工具调用与面向用户的自然语言总结交织在一起。这种输出异构性在标准的同策略强化学习（RL）中呈现出一种结构性失效模式：诸如GRPO之类的算法不加区分地向所有token广播同质的轨迹级标量优势值。因此，来自总结文本生成的梯度噪声会泄漏到工具决策token中，导致跨段信用错误归因和脆弱的优化过程。在本工作中，我们提出了SLCA-GRPO，一个融合了段锁定信用分配的框架。为了在不依赖昂贵的真实API的情况下实现可扩展的探索并保证训练的稳定性，我们首先构建了模式引导的LLM模拟器作为基础训练基础设施。在此基础上，SLCA在单组rollout内部于结构段级别对优势估计进行解耦，而无需额外的rollout采样。

    arXiv:2609.29050v1 Announce Type: new  Abstract: Tool-calling agents produce heterogeneous outputs, interleaving structured tool invocations with user-facing natural language summaries. This output heterogeneity presents a structural failure mode in standard on-policy Reinforcement Learning (RL): algorithms like GRPO indiscriminately broadcast a homogeneous trajectory-level scalar advantage to all tokens. Consequently, gradient noise from summary generation leaks into tool-decision tokens, causing cross-segment credit misattribution and brittle optimization. In this work, we propose SLCA-GRPO, a framework incorporating Segment-Locked Credit Assignment (SLCA). To enable scalable exploration without costly real APIs and stable training, we first construct the Schema-Guided LLM Simulator (SGLS) as foundational training infrastructure. Building on this, SLCA decouples advantage estimation at the structural segment level within a single group of rollouts, without requiring additional rollou
    
[^149]: 幻觉藏身何处：VQ标记化视觉语言模型中的跨架构电路

    Where Hallucinations Live: A Cross-Architecture Circuit in VQ-Tokenized Vision-Language Models

    [https://arxiv.org/abs/2609.29048](https://arxiv.org/abs/2609.29048)

    该研究发现在VQ标记化的视觉语言模型中，物体幻觉源于一个跨架构共享的早期层注意力路由电路，并通过单变量架构替换实验证明向量量化机制本身是这一病理性信号的根源。

    

    通过向量量化（VQ）码本对图像进行标记化的统一视觉语言模型（VLM）在基于事实的是/否基准测试中经常出现物体幻觉，然而现有的解码时修复方法仅将其视为一般的校准偏差，缺乏架构层面的解释。我们在涵盖八个LLM家族的二十五个模型上进行激活修补实验，识别出VQ标记化VLM之间共享的一个早期层（L₀）注意力路由电路，并提出一种三门诊断方法，用于区分携带该电路的模型与不携带该电路的模型。该诊断方法分离出十个阳性模型（涵盖三个LLM家族的五个自然统一VQ VLM和五个诱导变体），并排除了其余十五个模型。单变量架构替换实验（LLaVA-1.6的CLIP+MLP改为VQ+Linear）会安装该电路，而在相同数据上使用计算量匹配的MLP对照组则不会出现该现象，从而将向量量化确定为病理性信号的来源；

    arXiv:2609.29048v1 Announce Type: cross  Abstract: Unified vision-language models (VLMs) that tokenize images through a vector-quantized (VQ) codebook routinely hallucinate objects on grounded yes/no benchmarks, yet existing decoding-time fixes treat this as generic miscalibration without an architectural account. Using activation patching across twenty-five models spanning eight LLM families, we identify an early-layer ($L_0$) attention routing circuit shared across VQ-tokenized VLMs and propose a three-gate diagnostic that distinguishes the models carrying it from those that do not. The diagnostic isolates ten positive models (five natural unified-VQ VLMs across three LLM families and five induced variants) and rejects the remaining fifteen. A single-variable architectural swap (LLaVA-1.6 CLIP+MLP $\rightarrow$ VQ+Linear) installs the circuit, while a matched-compute MLP control on identical data does not, isolating vector quantization as the source of the pathological signal; the ro
    
[^150]: 用于黎曼流形和欧几里得脑电解码的个性化联邦学习

    Personalised federated learning for Riemannian and Euclidean EEG decoding

    [https://arxiv.org/abs/2609.29037](https://arxiv.org/abs/2609.29037)

    该论文将个性化联邦学习适配到黎曼SPDNet脑电解码器上，让所有受试者共享主干而各自保留分类头，在三个运动想象数据集上取得了优于标准联邦学习、集中式训练和EEGNet的准确率，同时收敛更快、通信参数更少。

    

    联邦学习（FL）使脑电（EEG）解码器能够从多个受试者的记录中学习，而无需将数据汇集到一起。我们考虑了两种轻量级脑电解码器：基于黎曼流形的SPDNet和基于欧几里得空间的EEGNet。两者都分为主干和头部两部分，其中主干用于构建潜在表示，头部用于对其进行分类。然而，受试者间的差异性使得单一共享的联邦学习模型难以很好地适配每个受试者。个性化联邦学习可以解决这一问题：所有受试者共同学习一个主干，而每个受试者保留自己的头部。我们将个性化联邦学习适配到SPDNet上，并以EEGNet作为欧几里得基线，研究了其相对于标准联邦学习和集中式训练的效果。实验涵盖了三个运动想象数据集，这些数据集在通道数、受试者数量和类别数方面覆盖了多种不同的情形。我们观察到，个性化SPDNet比标准联邦学习和集中式训练都取得了更高的准确率，同时比标准联邦学习收敛轮数更少、通信参数量更少，并且在各种情形下都优于EEGNet。

    arXiv:2609.29037v1 Announce Type: cross  Abstract: Federated learning (FL) lets EEG decoders learn from recordings of several subjects without pooling them. We consider two light EEG decoders, the Riemannian SPDNet and the Euclidean EEGNet. Both split into a trunk, which builds a latent representation, and a head, which classifies it. Inter-subject variability, however, makes a single shared FL model a poor fit for each subject. Personalised FL addresses this: all subjects learn a common trunk, and each subject keeps its own head. We adapt it for SPDNet and study its effects against standard FL and centralised training, with EEGNet as a Euclidean baseline. Experiments cover three motor-imagery datasets that span diverse regimes in channels, subjects and classes. We observe that personalised SPDNet reaches higher accuracy than both standard FL and centralised training, while converging in fewer rounds and communicating fewer parameters than standard FL. It also outperforms every EEGNet 
    
[^151]: 分页加载专家：iPhone上基于闪存的MoE推理的可复现特性分析

    Paging the Experts: A Reproducible Characterization of Flash-Backed MoE Inference on iPhone

    [https://arxiv.org/abs/2609.29032](https://arxiv.org/abs/2609.29032)

    该论文提出Routide运行时，通过将专家权重保留在闪存并按字节预算分页加载的方式在iPhone上运行量化35B级MoE大模型，并通过可复现测量证明MoE推理中看似存在的内存“容量断崖”实为缓存策略与工作负载交互的产物，而非普遍的内存需求。

    

    稀疏激活减少了混合专家的计算量，但并未消除存储全部专家权重的需求。我们提出了Routide，一个Swift/MLX运行时，它能在专家权重保留于iPhone存储中、仅将字节预算内的子集载入内存的条件下，执行固定的公开Qwen3.6-35B-A3B量化检查点的文本路径。我们刻画了缓存策略敏感性、数值比较边界以及测量限制。在五个记录的128-token工作负载上，固定路由回放在512 MiB LRU缓存下的按需命中率为0.00%，在相同预算下采用种子随机淘汰策略时为18.80%，而在576 MiB LRU下达到38.58%。因此，表面上的容量断崖是缓存策略与工作负载相互作用的结果，而非普遍的内存需求。同运行时的Mac对照组在淘汰与异步预取过程中保持了生成序列的一致性，包括2,560次精确token比较和10,334次推测性加载。相比之下，完整的驻留Python与记录的手机……（摘要在此处被截断）

    arXiv:2609.29032v1 Announce Type: cross  Abstract: Sparse activation reduces mixture-of-experts computation without eliminating the need to store all experts. We present Routide, a Swift/MLX runtime that executes the text path of a pinned public Qwen3.6-35B-A3B quantized checkpoint while keeping expert weights in iPhone storage and a byte-budgeted subset in memory. We characterize cache-policy sensitivity, numerical comparison boundaries, and measurement limits. Across five recorded 128-token workloads, fixed-route replay gives 0.00% demand hits with a 512 MiB LRU cache, 18.80% with seeded random eviction at the same budget, and 38.58% with 576 MiB LRU. The apparent capacity cliff is therefore a policy/workload interaction, not a universal memory requirement. Same-runtime Mac controls preserve generated sequences across eviction and asynchronous prefetch, including 2,560 exact token comparisons and 10,334 speculative loads. In contrast, complete resident-Python versus recorded-phone se
    
[^152]: 利用卫星影像中答案不变的冗余实现边缘端高效视觉语言模型推理

    Exploiting answer-invariant redundancies in satellite imagery for efficient VLM inference on edge

    [https://arxiv.org/abs/2609.29029](https://arxiv.org/abs/2609.29029)

    该论文提出Rift两阶段系统，通过识别并剪除卫星影像中不影响最终答案的图像分块与视觉token冗余，在Jetson AGX Orin上使LLaVA-1.5 7B的能耗降低78%、延迟降低69%，同时将准确率从45%提升至73%。

    

    星载视觉语言模型可以使卫星能够直接回答查询请求，但对高分辨率影像进行穷举式分块推理速度慢且能耗高。我们识别出了答案不变的token冗余（AITR）：即在不改变最终答案的前提下可以移除的图像分块和视觉token。我们提出了Rift，这是一个两阶段系统，首先执行基于查询条件的分块剪枝，随后进行弹性预填充以降低token预算。我们在Jetson AGX Orin上运行的LLaVA-1.5 7B模型上对其进行了评估。与穷举式分块推理相比，Rift将能耗降低了78%，延迟降低了69%，同时将准确率从45%提升至73%。

    arXiv:2609.29029v1 Announce Type: cross  Abstract: Onboard vision-language models could enable satellites to answer queries directly, but exhaustive tiled inference over high-resolution imagery is slow and energy-intensive. We identify answer-invariant token redundancy (AITR): image tiles and vision tokens that can be removed without changing the final answer. We present Rift, a two-stage system that performs query-conditioned tile pruning followed by elastic prefill to reduce token budget. We evaluate it on LLaVA-1.5 7B running on Jetson AGX Orin. Compared with exhaustive tiled inference, Rift reduces energy by 78% and latency by 69%, while increasing accuracy from 45% to 73%.
    
[^153]: 通过可组合接口从异构原位观测中进行生成式大气超分辨率

    Generative Atmospheric Super-Resolution from Heterogeneous In Situ Observations through Composable Interfaces

    [https://arxiv.org/abs/2609.29027](https://arxiv.org/abs/2609.29027)

    本文提出可组合观测接口方法，将探空仪、飞机和地面站这三类异构原位观测统一转换为特定来源的似然因子，从而条件化单一预训练的13变量大气扩散模型，实现生成式大气超分辨率重建。

    

    大气观测稀疏、异构且分布不均匀，而许多生成式大气模型学习的是规则网格上多变量状态的分布。经过预训练后，扩散模型可以提供大气先验，这些先验可以在贝叶斯框架中与由观测导出的似然因子相结合。然而，这些观测源在几何形态和采样密度上差异巨大，这使得在统一推断框架内一致地利用它们的观测变得复杂。本文将这一重建问题形式化为生成式大气超分辨率，并引入可组合的观测接口，用于对单一预训练的13变量大气扩散模型进行条件化。这些接口将稀疏的探空仪观测（R）、聚类的飞机观测（A）以及密集且不规则的地面站观测（S）转换为特定于来源的似然因子，用以指定观测在何处对网格施加约束。

    arXiv:2609.29027v1 Announce Type: new  Abstract: Atmospheric observations are sparse, heterogeneous, and unevenly distributed, whereas many generative atmospheric models learn distributions over regularly gridded multivariate states. Once pretrained, diffusion models can supply atmospheric priors that can be combined with observation-derived likelihood factors in a Bayesian formulation. However, these observation sources differ substantially in geometry and sampling density, complicating the consistent use of their observations within a common inference framework. Here, we formulate this reconstruction problem as generative atmospheric super-resolution and introduce composable observation interfaces for conditioning a single pretrained 13-variable atmospheric diffusion model. The interfaces convert sparse radiosonde (R), clustered aircraft (A), and dense irregular surface-station (S) observations into source-specific likelihood factors that specify where observations constrain the grid
    
[^154]: 基于点阵数据库增强与图卷积神经网络的生长启发式图生成与力学点阵逆向设计

    Growth-Inspired Graph Generation and Inverse Design of Mechanical Lattices via Dot Matrices Database Augmentation and GCNN

    [https://arxiv.org/abs/2609.29024](https://arxiv.org/abs/2609.29024)

    本工作受自然界生长机制启发，提出基于离散点阵顺序生长的形态发生式图生成框架，结合有限元分析与图卷积神经网络（GCNN）学习三维力学点阵的拓扑-性能映射，实现力学点阵的正向预测与逆向设计。

    

    自然界的承重与输运网络并非一步组装而成，而是通过生长、分支、强化和成环等时间有序的发育过程逐渐形成。受这一发育逻辑的启发，本工作提出了一种面向力学点阵的形态发生式图生成框架，其中离散点阵提供潜在节点，最终架构通过层间与层内的顺序生长而构建。该规则在二维中以类叶脉的发育序列进行可视化，并在三维中实现于一个包含27个候选节点的3x3x3节点矩阵上。通过基于梁单元的有限元分析对一系列不同的三维点阵数据集进行评估，并将其直接表示为图。一个具有三个图卷积层和双重全局池化的图卷积神经网络（GCNN）学习拓扑-性能映射，并预测有效压缩性能……

    arXiv:2609.29024v1 Announce Type: new  Abstract: Natural load-bearing and transport networks are not assembled in a single step; they emerge through a temporally ordered process of growth, branching, reinforcement, and loop formation. Inspired by this developmental logic, this work introduces a morphogenetic graph-generation framework for mechanical lattices in which a discrete dot matrix provides potential nodes and the final architecture is created by sequential cross-layer and intra-layer growth. The same rule is visualized in two dimensions as a leaf-vein-like developmental sequence and implemented in three dimensions on a 3x3x3 nodal matrix containing 27 candidate nodes. A dataset of distinct three-dimensional lattices was evaluated by beam-based finite element analysis and represented directly as graphs. A graph convolutional neural network (GCNN) with three graph-convolution layers and dual global pooling learns the topology-property mapping and predicts effective compressive st
    
[^155]: EvoTreeNAD：基于谱系引导进化的LLM驱动神经架构发现

    EvoTreeNAD: Genealogy-Guided Evolution for LLM-Driven Neural Architecture Discovery

    [https://arxiv.org/abs/2609.29016](https://arxiv.org/abs/2609.29016)

    EvoTreeNAD提出了一种谱系引导的进化算法，通过从空根节点生长谱系树并利用后代性能选择優良谱系，使LLM智能体无需种子或预定义搜索空间即可持续发现神经架构。

    

    人工智能驱动的科学发现通过自主开发解决方案和设计来加速研究进程。大型语言模型（LLM）智能体通过迭代生成和评估来支持这一过程。然而，仅靠这些迭代并不能确保累积性的进展，也无法确定下一步应探索哪些方向。高昂的评估成本进一步限制了探索的范围。神经架构发现将这些挑战汇聚在一起，将开放式设计与资源密集型实验相结合。我们提出了EvoTreeNAD，一种谱系引导的进化算法，它无需提供种子或手动指定的搜索空间即可构建可训练的神经架构。从一个空的根节点开始，它持续生长出一棵谱系树，其中每个新节点代表一个完整的架构。由每个节点及其后代计算出的前百分位数值指导谱系选择。利用选定的设计历史，一个创意智能体提出变体架构……

    arXiv:2609.29016v1 Announce Type: cross  Abstract: AI-driven scientific discovery accelerates research by autonomously developing solutions and designs. Large language model (LLM) agents support this process through iterative generation and evaluation. Yet these iterations alone do not ensure cumulative progress or establish which directions to pursue next. Costly evaluation further constrains the scope of exploration. Neural architecture discovery brings these challenges together, coupling open-ended design with resource-intensive experimentation. We introduce EvoTreeNAD, a genealogy-guided evolutionary algorithm that constructs trainable architectures without a supplied seed or a hand-specified search space. Starting from an empty root, it grows a persistent genealogy in which each new node represents a complete architecture. Top-percentile values computed from each node and its descendants guide lineage selection. Using the selected design history, an Idea Agent proposes a variant a
    
[^156]: 从混合质量的部署经验中学习机器人操作

    Learning from Mixed-Quality Deployment Experience for Robot Manipulation

    [https://arxiv.org/abs/2609.29000](https://arxiv.org/abs/2609.29000)

    提出预测性动作分块学习（PACL），仅利用真实部署中自然积累的混合质量自主经验（无需人工纠正或探索交互），通过分块级评论家与未来潜在预测增强时间差分学习，有效提升机器人操作策略的长时程学习效果。

    

    部署在真实环境中的机器人策略会自然积累混合质量的经验，包括成功的执行、部分进展和失败。尽管这些滚动数据为进一步学习提供了宝贵信息，但直接将其纳入模仿学习可能会强化不良行为，而离线强化学习在稀疏奖励和有限数据覆盖下往往存在价值估计不可靠的问题。我们考虑一种实际的部署后学习场景，即学习仅依赖于自然积累的自主滚动数据，无需额外的人工纠正或探索性交互。为了有效利用这类经验，我们提出了预测性动作分块学习（PACL）。PACL 首先学习一个预测性的分块级评论家，用于评估时间上延展的动作序列，并通过未来潜在预测来增强时间差分学习，为长时程任务提供更丰富的监督信号。

    arXiv:2609.29000v1 Announce Type: new  Abstract: Robot policies deployed in real environments naturally accumulate mixed-quality experience, including successful executions, partial progress, and failures. Although these rollouts provide valuable information for further learning, directly incorporating them into imitation learning may reinforce undesirable behaviors, while offline reinforcement learning often suffers from unreliable value estimation under sparse rewards and limited data coverage. We consider a practical post-deployment setting where learning relies only on naturally accumulated autonomous rollouts, without additional human corrections or exploratory interaction. To effectively exploit such experience, we propose Predictive Action Chunk Learning (PACL). PACL first learns a predictive chunk-level critic that evaluates temporally extended action sequences and augments temporal difference learning with future latent prediction, providing richer supervision for long-horizon
    
[^157]: 基于 ℓp 正则化的大语言模型低秩自适应自动秩分配

    Automatic Rank Allocation for Low-Rank Adaptation in Large Language Models via lp Regularization

    [https://arxiv.org/abs/2609.28998](https://arxiv.org/abs/2609.28998)

    提出 ℓp-LoRA，利用 ℓp 正则化对每个秩一分量的能量施加稀疏约束，将矩阵优化简化为二维问题并推导出隐式阈值准则，从而以有原理、无需手动设计重要性分数的方式自动为 LoRA 分配各矩阵的秩。

    

    低秩自适应已成为大语言模型流行的参数高效微调方法。LoRA 的一个关键挑战是如何确定每个自适应矩阵的秩，因为秩直接控制其容量和效率。现有的自适应秩方法通常依据手动设计的重要性分数来分配秩，而这些分数并非直接从优化目标推导而来。在本工作中，我们提出了 ℓp-LoRA，这是一种基于 ℓp 正则化（0<p<1）的原则性秩分配方法，ℓp 正则化是信号处理和统计学中经典的稀疏性诱导技术。具体而言，我们对每个秩一 LoRA 分量的能量进行正则化，促使冗余分量消失，同时保留重要分量。我们推导了相应的近端子问题，并将矩阵优化问题简化为二维问题，从而得到一个用于识别冗余分量的隐式阈值准则（摘要在此处截断）。

    arXiv:2609.28998v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) has become a popular parameter-efficient fine-tuning method for large language models. A key challenge in LoRA is how to determine the rank of each adaptation matrix, as rank directly controls its capacity and efficiency. Existing adaptive-rank methods typically allocate ranks according to manually designed importance scores, which are not directly derived from an optimization objective. In this work, we propose $\ell_p$-LoRA, a principled rank-allocation method based on $\ell_p$ regularization with $0<1$, which is a classical sparsity-inducing technique in signal processing and statistics. Specifically, we regularize the energy of each rank-one LoRA component, encouraging redundant components to vanish while preserving important ones. We derive the corresponding proximal subproblem and reduce the matrix optimization to a two-dimensional problem, leading to an implicit thresholding criterion for identifying red
    
[^158]: CrossSafe：迈向跨形态的潜在安全过滤器

    CrossSafe: Towards Cross-Embodiment Latent Safety Filters

    [https://arxiv.org/abs/2609.28984](https://arxiv.org/abs/2609.28984)

    本文提出CrossSafe，利用安全推理在不同机器人间的共通性构建跨形态的潜在安全过滤器，并根据各机器人的形态、运动学和动力学差异来具体实现安全动作，从而为通用操作策略提供跨机器人的安全保障。

    

    跨形态学习已经表明，单一模型（例如视觉-语言-动作模型，即VLA模型）能够学习可应用于异构机器人以完成各种任务的状态表示和操作技能。我们假设安全执行同样具有这一性质。满足安全约束所需的推理过程——例如检测障碍物、识别出需要避开它、以及选择一个安全的抽象动作——在很大程度上是不同机器人之间共享的。不同形态之间的差异在于抽象的安全动作如何被具体实现：形态结构、运动学和动力学决定了哪些动作是安全且可行的。因此，同一个动作对一个机器人可能是安全的，而对另一个机器人则是不安全的。这对于在统一的末端执行器动作空间中运行、且未明确刻画安全性如何依赖于机器人形态和运动学的通用操作策略而言尤为重要。我们提出

    arXiv:2609.28984v1 Announce Type: cross  Abstract: Cross-embodiment learning has shown that a single model, such as a vision-language-action (VLA) model, can learn state representations and manipulation skills that can be applied across heterogeneous robots to accomplish various tasks. We hypothesize that the same holds for safety enforcement. The reasoning required to satisfy a safety constraint, such as detecting an obstacle, recognizing that it should be avoided, and selecting a safe abstract action, is largely shared across robots. What differs across embodiments is how the abstract safe action is realized: morphology, kinematics, and dynamics determine which actions are safe and feasible. Consequently, the same action can be safe for one robot and unsafe for another. This is especially important for generalist manipulation policies that operate in a common end-effector action space without explicitly capturing how safety depends on the robot's morphology and kinematics. We propose
    
[^159]: 基于厄米特多项式的谱图神经网络：一项全面研究

    Spectral Graph Neural Networks with Hermite Polynomials: A Comprehensive Study

    [https://arxiv.org/abs/2609.28979](https://arxiv.org/abs/2609.28979)

    本文提出基于厄米特多项式的谱图神经网络HermNet，无需特征分解或学习基底即可实现稀疏高效的图滤波传播，并在有限训练预算下优于其他多项式基底模型。

    

    我们研究了基于厄米特多项式构建的谱图神经网络，并提出了HermNet——一个将逐节点预测器与归一化厄米特传播相结合的简单模型。其稀疏递推结构既无需特征分解，也无需学习基底。我们将基础模型与可选的坐标校准、响应归一化以及高斯导数正则化加以区分。厄米特多项式与其他完备多项式基底张成相同的次数受限滤波器空间，但在有限训练预算下，不同基底的坐标可能产生不同的优化行为。我们通过谱信号能量、标签采样、学习特征的变化以及正则化的偏差-方差权衡来分析这一行为。受控合成实验确定了一个特定区间，在该区间内，纯HermNet优于与之匹配的多项式基底替代方案，包括与联合训练的非线性预测器结合使用的情况。曲率正则化进一步改善了HermNet的性能。

    arXiv:2609.28979v1 Announce Type: new  Abstract: We study spectral graph neural networks built from Hermite polynomials and propose HermNet, a simple model that combines a nodewise predictor with normalized Hermite propagation. Its sparse recurrence requires neither eigendecomposition nor a learned basis. We distinguish the basic model from optional coordinate calibration, response normalization and Gaussian derivative regularization. Hermite and other complete polynomial bases span the same degree-bounded filter space, but their coordinates can produce different optimization behavior under limited training budgets. We analyze this behavior through spectral signal energy, label sampling, changes in learned features and the bias--variance trade-off of regularization. Controlled synthetic experiments identify a regime in which plain HermNet outperforms matched polynomial-basis alternatives, including with a jointly trained nonlinear predictor. Curvature regularization further improves He
    
[^160]: 相同位宽，不同结果：跨架构的文本转语音训练后量化

    Same Bit Width, Different Outcomes: Post-Training Quantization of Text-to-Speech Across Architectures

    [https://arxiv.org/abs/2609.28974](https://arxiv.org/abs/2609.28974)

    该论文首次在统一协议下系统评估了跨多种TTS架构的训练后量化，发现相同位宽在不同模型上效果差异巨大且敏感组件因模型而异，可通过分阶段消融识别并用逐层GPTQ将性能恢复至0.1 UTMOS以内。

    

    训练后量化（PTQ）可以降低设备端文本转语音（TTS）的成本，但已发表的评估仅覆盖单一系统或单一方法。我们在统一协议下跨多种TTS架构评估PTQ，包括三个核心模型、另外八个模型的权重与激活消融实验，以及两个进行盲量化的留存模型。4位逐通道权重量化使UTMOS（一种预测的平均意见得分）在Supertonic上下降2.8，而在Kokoro上仅下降0.07；逐张量缩放即使在8位下也可能导致严重退化。相同的位宽会产生不同的结果，因为敏感组件是模型特定的，无法从模型类别可靠地预测。一种分阶段消融流程可以识别该敏感组件，而逐层GPTQ能将其恢复至0.1 UTMOS以内。真实的int8和int4内核以硬件相关的成本再现了模拟结果的排序。在Mac mini上，4位权重内核使Supertonic的延迟降至fp32的0.60倍，而int8反而更慢，因此每种配置的性能表现各不相同。

    arXiv:2609.28974v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) reduces the cost of on-device text-to-speech (TTS), but published evaluations cover one system or method. We evaluate PTQ across TTS architectures under one protocol with three core models, weight and activation ablations of eight more, and two held-out models quantized blind. Four-bit per-channel weights reduce UTMOS, a predicted mean opinion score, by 2.8 on Supertonic and 0.07 on Kokoro, and per-tensor scaling can cause severe degradation even at 8 bits. The same bit width yields different outcomes, because the sensitive component is model-specific and not reliably predicted from the model class. A staged ablation procedure identifies it, and per-layer GPTQ can restore it to within 0.1 UTMOS. Real int8 and int4 kernels reproduce the simulated ordering at hardware-dependent cost. On a Mac mini, a 4-bit weight kernel runs Supertonic at 0.60x the fp32 latency while int8 is slower, so each configuration 
    
[^161]: 为什么虚假信息传播得更快？从算法视角解析X平台

    Why Does Misinformation Propagate Faster? An Algorithmic Perspective on X

    [https://arxiv.org/abs/2609.28947](https://arxiv.org/abs/2609.28947)

    本文通过对X平台开源推荐算法进行首个组件级分析，揭示其“互动可替代性机制”——将推荐分数构建为所有预测用户互动的加权和——使得仅凭引发大量即时反应（点赞、转发）即可被反复推荐的虚假信息获得了传播优势，从而从算法机制层面解释了虚假信息为何传播得更快。

    

    虚假信息被广泛报道在基于互动（engagement-based）的社交平台上传播得更快，然而以往的研究大多停留在实证分析层面，未能识别出导致这一现象的具体算法机制。得益于X平台推荐算法的开源，我们开展了据我们所知首个针对社交媒体平台实际部署的推荐算法的组件级（component-level）研究，检验该算法的各个组件如何影响虚假信息的传播。具体而言，我们识别出了算法中的一种“互动可替代性机制”（engagement fungibility mechanism）：最终的推荐分数被构建为所有预测用户互动行为的加权和。其结果是，一条推文可能仅仅因为被预测会吸引大量即时反应（例如点赞和转发）而被反复推荐，即使它并不被预期会引发深入的回应（例如回复和引用）。由于虚假信息通常...

    arXiv:2609.28947v1 Announce Type: cross  Abstract: Misinformation is widely reported to propagate faster on engagement-based platforms, yet prior work largely focused on empirical analysis, without identifying a specific algorithmic mechanism that results in this phenomenon. Thanks to the open-sourcing of X's recommendation algorithms, we conduct what is, to our knowledge, the first component-level study of the recommendation algorithm deployed by a social media platform, which examines how each of its components affects misinformation propagation. Specifically, we identify the engagement fungibility mechanism in the algorithm, where the final recommendation score is constructed as a weighted sum of all predicted user activities. As a result, a tweet can be repeatedly recommended simply because it is predicted to draw many instant reactions (e.g., likes and retweets), even when it is not expected to draw thoughtful responses (e.g., replies and quotes). Since misinformation typically dr
    
[^162]: 基于电子先验的响应态学习实现可迁移的振动光谱表征

    Response-state Learning for Transferable Vibrational Spectroscopic Characterization with Electron Prior

    [https://arxiv.org/abs/2609.28935](https://arxiv.org/abs/2609.28935)

    提出SO(3)等变神经卡尔曼网络SENK，通过等变Transformer主干、神经卡尔曼桥与NBO电子先验通路构建响应态级联，实现从小分子到类药体系跨化学空间的可迁移、高保真IR/Raman光谱预测，并在QM9S和QMe14S上超越DetaNet。

    

    当局域立体电子环境扰动中间响应态、且高风险响应单元主导特征光谱指纹时，振动光谱预测可能变得不准确，使得跨外部化学空间的预测十分困难。SO(3)等变神经卡尔曼网络（SENK）构建了一种响应态级联结构，其结合了用于Hessian、偶极导数和极化率导数学习的等变Transformer主干网络、用于状态相关精炼与可靠性感知的等变神经卡尔曼桥，以及将一致性正则化与有界的、分支特定的引导光谱校准相耦合的NBO信息电子先验通路。SENK在QM9S和QMe14S数据集上超越了DetaNet，同时从小分子到类药体系均保持了全谱红外（IR）和拉曼（Raman）光谱的保真度。在具有复杂局域电子结构的生物分子体系中，SENK保持稳定并有选择性地改进了光谱敏感特征。

    arXiv:2609.28935v1 Announce Type: new  Abstract: Vibrational spectral prediction can become inaccurate when localized stereoelectronic environments perturb intermediate response states and high-risk response units dominate characteristic spectral fingerprints, making prediction across external chemical space difficult. SO(3) Equivariant Neural Kalman Networks (SENK) form a response-state cascade that combines an equivariant transformer backbone for Hessian, dipole-derivative and polarizability-derivative learning, an Equivariant Neural Kalman bridge for state-dependent refinement and reliability sensing, and an NBO-informed electronic-prior pathway coupling consistency regularization with bounded, branch-specific guided spectral calibration. SENK outperforms DetaNet on QM9S and QMe14S while preserving full-spectrum IR and Raman fidelity from small molecules to drug-like systems. SENK remains stable and selectively improves spectrally sensitive features in biomolecular systems with comp
    
[^163]: 硬件设计验证的自动测试框架演化：LLM 能否巩固已发现测试框架的收益？

    Automatic Harness Evolution for Hardware Design Verification: Can LLMs Consolidate Gains Across Discovered Harnesses?

    [https://arxiv.org/abs/2609.28908](https://arxiv.org/abs/2609.28908)

    该研究在硬件设计验证任务中对固定语言模型进行自动测试框架演化，发现虽然演化显著提升了完成尝试和任务覆盖率，但这些收益难以跨任务巩固和保持，表明LLM目前尚无法可靠地整合测试框架的改进收益。

    

    智能体的行为取决于围绕语言模型的测试框架，但语言模型能否可靠地改进此类硬件设计任务的测试框架仍不清楚。我们围绕一个固定的目标模型，在12个专有的设计验证根因定位任务上研究了测试框架的自动演化。在每项任务五次试验中，自动演化的测试框架将完成的尝试次数提高了71-76%，将任意命中任务覆盖率提高了80-100%，而总正确尝试仅提高了18-24%。最强的至少可复现两次的成功结果仅提高了一个任务，且后续候选者在任务之间交换收益而非保留收益。一个辅助候选者在排除于搜索之外的四任务验证集上有所改进，但在包含搜索和验证任务的后续12任务重放中与基线持平，因此所选收益未能在完整任务池中保持。在所测试的演化谱系中，有效的搜索……（摘要截断）

    arXiv:2609.28908v1 Announce Type: cross  Abstract: Agent behavior depends on the harness surrounding a language model, but it remains unclear whether language models can reliably improve such harnesses for hardware-design tasks. We study automatic harness evolution around a fixed subject model on 12 proprietary design-verification root-cause localization tasks. Across five trials per task, automatically evolved harnesses increased completed attempts by 71-76% and any-hit task coverage by 80-100%, while total correct attempts improved by only 18-24%. The strongest success reproducible at least twice result improved by one task, and later candidates exchanged gains across tasks rather than preserving them. An auxiliary candidate improved on a four-task validation set excluded from search but tied its baseline on a subsequent 12-task replay containing both search and validation tasks, so the selected gain did not persist across the full pool. Across the tested lineage, useful search, evid
    
[^164]: GeoDose-CP：面向连续处理地球观测的图局部保形推断

    GeoDose-CP: Graph-Local Conformal Inference for Continuous-Treatment Earth Observation

    [https://arxiv.org/abs/2609.28895](https://arxiv.org/abs/2609.28895)

    GeoDose-CP提出了一种支撑感知的图局部保形推断框架，通过联合建模干预处理偏移、逆结果尺度雅可比和空间残差依赖性，为连续处理地球观测数据提供可靠的干预不确定性量化。

    

    从地球观测（EO）中获得面向干预的可靠不确定性量化仍然具有挑战性，尤其当需要同时应对连续处理变化、空间依赖性、有限支撑以及卫星-结果不确定性时。现有的因果推断、保形预测和空间方法各自解决了该问题的部分内容，但它们的直接组合通常无法恢复适当的干预参考规律，因为候选处理重新分配会同时改变处理似然、标准化残差以及依赖于图的残差似然。本研究提出了GeoDose-CP，一个支撑感知的保形框架，用于在连续或混合连续-原子处理下进行局部化随机潜在结果推断。其核心方法论贡献是一种图局部目标轨道规律，它联合刻画了干预引起的处理偏移、结果尺度的逆雅可比行列式以及空间残差依赖性。

    arXiv:2609.28895v1 Announce Type: cross  Abstract: Reliable intervention-oriented uncertainty quantification from Earth observation (EO) remains challenging when continuous treatment shifts, spatial dependence, limited support, and satellite-outcome uncertainty must be addressed simultaneously. Existing causal, conformal, and spatial approaches address parts of this problem, but their direct combination does not generally recover the appropriate interventional reference law because candidate reassignment jointly alters treatment likelihood, standardized residuals, and graph-dependent residual likelihood. This study presents GeoDose-CP, a support-aware conformal framework for localized stochastic potential outcomes under continuous or mixed continuous-atomic treatment. Its central methodological contribution is a graph-local target-orbit law that jointly represents intervention-induced treatment shift, the inverse outcome-scale Jacobian, and spatial residual dependence. The framework fu
    
[^165]: Forecast-Dojo：用于基准测试和训练大语言模型预测代理的可重放环境

    Forecast-Dojo: Replayable Environments for Benchmarking and Training LLM Forecasting Agents

    [https://arxiv.org/abs/2609.28876](https://arxiv.org/abs/2609.28876)

    该论文提出了Forecast-Dojo——一个结合已结算预测市场问题与带日期新闻的可重放环境，用于基准测试和训练LLM预测代理，实验发现研究工具能提升全部12个模型的预测表现，但所有模型仍落后于历史市场预测水平。

    

    我们推出了Forecast-Dojo，这是一个用于基准测试和训练大语言模型预测代理的可重放环境。它将已结算的预测市场问题与带日期标注的新闻相结合，使代理能够研究某个事件，并在连续的历史时间点上更新其预测。相同的任务和工具支持重复评估、训练交互数据的收集，以及来自已记录结果的反馈，而无需等待新事件结算。Forecast-Dojo包含1,568个Polymarket事件（按时间划分为训练期和评估期）以及1,880万篇带日期标注的新闻文章。在对12个模型的评估中，研究工具降低了所有12个模型的Brier分数。预测质量也随着事件的发展而提升，在记录到更多新增带日期证据的步骤中提升幅度最大。然而，所有模型在Brier分数和准确率上仍然落后于历史市场预测。在时间点之间携带的信念笔记本降低了研究成本，但并不能持续改善预测表现（原文此处截断）。

    arXiv:2609.28876v1 Announce Type: new  Abstract: We introduce Forecast-Dojo, a replayable environment for benchmarking and training LLM forecasting agents. It combines resolved prediction-market questions with dated news, allowing agents to research an event and revisit their predictions at successive historical dates. The same tasks and tools support repeated evaluation, collection of training interactions, and feedback from recorded outcomes without waiting for new events to resolve. Forecast-Dojo contains 1,568 Polymarket events, split by time into training and evaluation periods, and 18.8M dated news articles. In an evaluation of 12 models, research tools lower Brier score for all 12. Forecasts also improve as events unfold, with the largest gains at steps where more newly dated evidence is recorded. Every model still trails historical market forecasts in both Brier score and accuracy. A belief notebook carried between dates lowers research cost but does not consistently improve fo
    
[^166]: 当花哨的淘汰策略失效时：重新思考LLM前缀复用的缓存替换

    When Fancy Eviction Fails: Rethinking Cache Replacement For LLM Prefix Reuse

    [https://arxiv.org/abs/2609.28870](https://arxiv.org/abs/2609.28870)

    通过对两家公司生产追踪数据的分析发现，在LLM前缀缓存场景下，为传统缓存设计的复杂淘汰策略相比LRU几乎没有收益，原因在于前缀复用由活跃会话的规律节奏主导、使得“近期性”异常具有预测性，因此有效的前缀缓存管理应以近期性为基础。

    

    长时间运行的LLM应用会反复发送不断增长的上下文，使得前缀缓存对降低预填充（prefill）成本至关重要。然而，前缀缓存在智能体（agentic）工作负载下的行为仍未被充分理解。我们研究了两家公司的生产环境追踪数据，并在HBM受限和大内存池两种设置下评估了14种淘汰算法。尽管与Belady最优算法存在较大差距，但为传统缓存设计的复杂策略相比LRU几乎没有带来任何收益。其原因是结构性的：前缀复用主要由活跃会话的规律性节奏所主导，这使得“近期性”具有异常强的预测能力。尽管如此，前缀缓存仍然带来了新的挑战，包括重尾分布的会话内存占用，以及随注意力计算随序列长度增长而高度变化的未命中成本。我们引入了计算节省比率（compute-savings ratio）和两个离线oracle来量化这些影响。我们的结果表明，有效的前缀缓存管理应将近期性作为其基础

    arXiv:2609.28870v1 Announce Type: cross  Abstract: Long-running LLM applications repeatedly send growing context, making prefix caching critical for reducing prefill cost. Yet prefix-cache behavior under agentic workloads remains poorly understood. We study production traces from two companies and evaluate 14 eviction algorithms across HBM-constrained and large memory-pool settings. Despite a large gap to Belady, sophisticated policies designed for traditional caches provide little benefit over LRU. The reason is structural: prefix reuse is dominated by the regular pacing of active sessions, making recency unusually predictive. Prefix caching nevertheless introduces new challenges, including heavy-tailed session footprints and highly variable miss costs as attention computation grows with sequence length. We introduce the compute-savings ratio and two offline oracles to quantify these effects. Our results show that effective prefix-cache management should retain recency as its foundati
    
[^167]: 图像保真度不等于场保真度：神经断层扫描中的联合热力学重建与误差定位

    Image Fidelity is Not Field Fidelity: Joint Thermodynamic Reconstruction and Error Localization in Neural Tomography

    [https://arxiv.org/abs/2609.28868](https://arxiv.org/abs/2609.28868)

    该论文提出CoroNeRF方法，通过可微分原子发射渲染器从多视角、多谱线观测中联合重建三维电子密度和温度场，并证明跨随机种子不稳定性可作为推理时无需真值的局部物理场误差指标。

    

    用于科学断层扫描的神经场是从二维图像中优化的，但实际感兴趣的量往往是一个潜在的三维物理场。由于前向映射是多对一的，较低的二维图像误差并不能保证三维物理场的正确性。此外，潜在场在训练过程中不受直接监督，其误差在部署阶段也无法对照真值进行评估。我们开发了CoroNeRF，通过可微分的原子发射渲染器，直接从多视角、多谱线强度中联合优化三维电子密度和温度场。以日冕断层扫描作为受控测试平台，我们评估了物理场的恢复效果，并检验跨随机种子不稳定性是否能够提供一种在推理阶段无需真值的局部物理场误差指标。我们着重强调了以下两个观察结果：（i）图像保真度不等于场保真度：光谱消融实验表明，有限通道的重建可以拟合其可用的观测数据……

    arXiv:2609.28868v1 Announce Type: new  Abstract: Neural fields for scientific tomography are optimized from 2D images, but the actual quantity of interest is often a latent 3D physical field. Because the forward map is many-to-one, low 2D image error need not certify a correct 3D field. Moreover, the latent field is not directly supervised during training, and its error cannot be evaluated against truth at deployment. We develop CoroNeRF to jointly optimize 3D electron density and temperature fields directly from multiview, multiline intensities through a differentiable atomic-emission renderer. Using solar coronal tomography as a controlled testbed, we evaluate physical-field recovery and test whether cross-seed instability provides a ground-truth-free-at-inference indicator of local physical-field error. We underscore the following two observations. (i) Image fidelity is not field fidelity: spectral ablations show that limited-channel reconstructions can fit their available observati
    
[^168]: 面向语言引导医学图像分割的多模态路由与区域精化

    Multimodal Routing and Region Refinement for Language-Guided Medical Image Segmentation

    [https://arxiv.org/abs/2609.28860](https://arxiv.org/abs/2609.28860)

    MRSeg提出了一种参数高效的多模态路由框架，通过联合路由器为每个图像-文本对动态分配低秩适配器来协同适配视觉与文本特征，并结合基于文本查询的区域精化模块，提升语言引导医学图像分割的准确性。

    

    文本描述可以通过明确指定待勾画的病灶表现及其位置，来减少医学图像分割中的歧义。现有的文本引导方法主要改进图像与语言特征交互的位置，但通常在所有图像-文本对之间保留单一的学习更新路径。我们提出了MRSeg，这是一个参数高效的框架，它利用每个图像-文本对在稠密预测之前对视觉特征与文本特征的适配进行路由。冻结的ConvNeXt-Tiny和PubMedBERT编码器提供多尺度视觉特征和临床文本标记。一个联合路由器利用最深层的视觉特征和池化后的文本，在低秩适配器基上预测稀疏混合。所得到的路由在两个视觉尺度和文本的独立适配器组之间共享，在协调各模态适配的同时保持各自特征特定的参数相互分离。Region Bridge模块使用源自文本的查询将稠密视觉标记聚合到潜在表示中（原文摘要在此处截断）。

    arXiv:2609.28860v1 Announce Type: cross  Abstract: Textual descriptions can reduce ambiguity in medical image segmentation by specifying the finding and location to be delineated. Existing text-guided methods mainly improve where image and language features interact but generally retain a single learned update pathway across all image-text pairs. We propose MRSeg, a parameter-efficient framework that uses each image-text pair to route the adaptation of visual and textual features before dense prediction. Frozen ConvNeXt-Tiny and PubMedBERT encoders provide multiscale visual features and clinical text tokens. A joint router uses the deepest visual feature and pooled text to predict a sparse mixture over low-rank adapter bases. The resulting route is shared across separate adapter banks for two visual scales and text, coordinating their adaptation while keeping the feature-specific parameters separate. Region Bridge uses text-derived queries to aggregate dense visual tokens into latent r
    
[^169]: RECLAIM：智能体能复现机器学习论文的结论吗？

    RECLAIM: Can Agents Reproduce the Claims of Machine Learning Papers?

    [https://arxiv.org/abs/2609.28850](https://arxiv.org/abs/2609.28850)

    RECLAIM是一个基于100篇NeurIPS 2025论文的可重建基准测试，通过预先定义复现目标、成功标准和GPU预算，并按作者发布资源分为运行、重训练、重新实现三个难度级别，用独立语言模型依据日志评分，结果显示最好的AI智能体也只能分别复现41%、27%和15%的论文结果。

    

    复现一篇机器学习论文涉及大部分研究步骤，从安装软件、调试到运行实验，这些工作正日益由AI智能体来承担。我们提出了RECLAIM，一个基于100篇NeurIPS 2025论文的基准测试，可以每年从新会议中重建更新。对于每篇论文，我们预先固定要复现的结果、判定复现成功的标准以及GPU小时预算。智能体必须利用论文本身和作者发布的资源来复现该结果。作者发布的内容决定了难度级别：Run级（运行级）发布包含代码、数据和模型权重；Retrain级（重训练级）发布缺少权重，因此智能体需要自行训练模型；Reimplement级（重新实现级）发布缺少代码，因此智能体需要自己编写代码。我们使用一个独立的语言模型根据日志和输出（而非智能体自己的报告）来评分。我们在每篇论文上运行四个智能体各一次；结果发现每个级别中表现最好的智能体也仅能复现Run级论文的41%、Retrain级的27%和Reimplement级的15%。

    arXiv:2609.28850v1 Announce Type: new  Abstract: Reproducing a machine learning paper involves most research steps, from installing software and debugging to running experiments, work that AI agents increasingly do. We introduce RECLAIM, a benchmark of 100 NeurIPS 2025 papers that can be rebuilt yearly from new conferences. For each paper we fix in advance the result to reproduce, what counts as a successful reproduction, and a GPU-hour budget. An agent must reproduce that result using the paper and whatever its authors released. What the authors released decides the difficulty tier. Run-tier releases include code, data, and weights; Retrain-tier releases lack weights, so the agent trains the model; Reimplement-tier releases lack code, so the agent writes it. A separate language model grades runs from logs and outputs rather than agents' reports. We run four agents once per paper; the best agent in each tier reproduces only 41% of Run-tier papers, 27% at Retrain, and 15% at Reimplement
    
[^170]: LastOPD：驯服潜在在线策略蒸馏中的崩溃现象

    LastOPD: Taming Collapse in Latent On-Policy Distillation

    [https://arxiv.org/abs/2609.28845](https://arxiv.org/abs/2609.28845)

    论文揭示了潜在在线策略蒸馏中“先获益后崩溃”以及“对齐越好反而表现越差”两大失败模式，将其根源归结为潜在信号在不同层上的角色错配，并提出 LastOPD 方法来驯服这种崩溃。

    

    在线策略蒸馏（OPD）依据学生模型自身生成的回答对其进行纠正，但其信号来自教师模型的下一词元分布：它告诉学生教师“说了什么”，却遗漏了教师“如何思考”。潜在监督通过将学生模型的潜在状态与教师模型对齐，有望补上这一缺失部分。近期诸如 OPRD 等方法将这一信号引入了在线策略蒸馏。然而，在将 Qwen3-4B 和 Qwen3-8B 蒸馏到 Qwen3-1.7B-Base 的过程中，我们观察到这种做法存在两种失败模式。其一，先获益后崩溃：仅使用潜在监督即可在 10 步内将 MATH-500 准确率从 25 提升至 46，但随后的训练使性能退化至 11 且无法恢复。其二，对齐越好、行为越差：尽管在崩溃过程中对齐指标持续改善，但对齐程度最高的模型反而表现最差。进一步分析表明，潜在信号的应用方式存在错配：按深度配对的各层实际上扮演着不同的角色……

    arXiv:2609.28845v1 Announce Type: cross  Abstract: On-policy distillation (OPD) corrects a student on the responses it writes, but its signal is the teacher's next-token distribution: it tells the student what the teacher says but misses how it thinks. Latent supervision promises the missing part by aligning the student's latent states to the teacher's. Recent methods such as OPRD bring this signal into on-policy distillation. However, we observe two failures of this recipe when distilling Qwen3-4B and Qwen3-8B into Qwen3-1.7B-Base. Early gain, late collapse: latent supervision alone lifts MATH-500 accuracy from 25 to 46 in 10 steps, but subsequent training degrades performance down to 11 with no recovery. Better alignment, worse behavior: although the alignment metric steadily improves throughout this collapse, the most aligned model turns out to be the worst performing. Further analysis suggests a mismatch in how the latent signal is applied: layers paired by depth play different rol
    
[^171]: 不确定性门控探索噪声抑制流匹配视觉-语言-动作策略在线强化学习微调中的任务坍塌

    Uncertainty-Gated Exploration Noise Suppresses Task Collapse in Online RL Fine-Tuning of a Flow-Matching Vision-Language-Action Policy

    [https://arxiv.org/abs/2609.28838](https://arxiv.org/abs/2609.28838)

    该论文提出一种不确定性门控的探索噪声控制器，利用任务无关的新颖性与能力信号在任务流之间重新分配探索，在小算力在线强化学习微调流匹配VLA策略时有效抑制了“任务坍塌”，优于固定噪声和可学习噪声方案。

    

    预训练流匹配视觉-语言-动作（VLA）策略的在线强化学习微调有望让机器人在部署后持续学习，但持续的更新往往会在总体表现看似健康的情况下，破坏模型在单个任务上的能力。我们在匹配的小算力预算下研究了这种被称为“任务坍塌”的失败模式，实验在LIBERO-10上进行，使用450M参数的SmolVLA策略，通过带随机（SDE）采样的PPO进行训练。三种探索噪声策略仅在一个运行时变量上有所不同：固定噪声尺度、ReinFlow风格的可学习噪声网络，以及一种不确定性门控控制器——后者基于任务无关的新颖性与能力信号在任务流之间重新分配探索，且无需任务标签或回合边界。在池化定义下，固定噪声在三个随机种子中的两个出现任务坍塌，可学习噪声在测量至200次迭代的每个种子中均出现坍塌，而该控制器在任何……（原文截断）

    arXiv:2609.28838v1 Announce Type: cross  Abstract: Online reinforcement learning fine-tuning of pretrained flow-matching vision-language-action (VLA) policies promises robots that keep learning after deployment, but continued updates often destroy competence on individual tasks while the aggregate still looks healthy. We study this failure mode, which we call task collapse, under a matched small-compute budget on LIBERO-10 with a 450M-parameter SmolVLA policy trained by PPO with stochastic (SDE) sampling. Three exploration-noise policies differ in one live variable: a fixed noise scale, a ReinFlow-style learned noise network, and an uncertainty-gated controller that redistributes exploration across task streams from task-agnostic novelty and competence signals, without task labels or episode boundaries. Under the pooled definition, fixed noise collapses tasks in two of three seeds and learned noise in every seed measured to iteration 200, while the controller collapses none in any of i
    
[^172]: M²PFN：面向阿尔茨海默病可泛化多模态上下文学习的端到端解耦对齐

    M$^2$PFN: End-to-End Disentangled Alignment for Generalizable Multimodal In-Context Learning in Alzheimer's Disease

    [https://arxiv.org/abs/2609.28836](https://arxiv.org/abs/2609.28836)

    提出端到端框架M²PFN，通过在TabPFN的transformer中进行可微推理、并利用解耦与对比学习将3D-MRI和表格模态对齐至共享子空间，从而把表格基础模型的上下文学习能力扩展为可跨队列泛化的多模态阿尔茨海默病诊断器。

    

    尽管已有多种结合影像与表格数据进行阿尔茨海默病（AD）诊断的多模态方法被提出，但它们在跨队列泛化方面往往存在局限。上下文学习（ICL）已在TabPFN等基础表格模型中展现出卓越的泛化性能和高度灵活性。将TabPFN的ICL扩展至多模态AD分析的主要障碍在于：TabPFN是在合成表格先验上进行元训练的，而这些先验与图像衍生特征的统计结构并不天然匹配。我们提出了M²PFN，一个能将该表格基础模型转变为多模态AD预测器的端到端框架。M²PFN (i) 通过TabPFN的transformer执行可微推理，将任务梯度反向传播至3D-MRI编码器和表格编码器；(ii) 通过解耦与对比目标，将两种模态对齐到与ICL引擎先验相匹配的共享子空间中；(iii) …（原文摘要在此处截断）

    arXiv:2609.28836v1 Announce Type: cross  Abstract: While various multimodal methods combining imaging and tabular data for Alzheimer's disease (AD) diagnosis were proposed, they are often limited in generalization across cohorts. In-context learning (ICL) has demonstrated excellent generalization performances and high flexibility in foundational tabular models such as TabPFN. To extend TabPFN's ICL to multimodal AD analysis, the main obstacle is that TabPFN is meta-trained on synthetic tabular priors that do not naturally match the statistical structure of image-derived features. We propose M$^2$PFN, an end-to-end framework that turns this tabular foundation model into a multimodal AD predictor. M$^2$PFN (i) performs differentiable inference through TabPFN's transformer, back-propagating task gradients into 3D-MRI and tabular encoders; (ii) aligns the two modalities into a shared subspace, via disentanglement and a contrastive objective, matched to the ICL engine's prior; and (iii) fol
    
[^173]: 无监督学习何时成功或失败？基于重构的异常检测的子空间追踪视角

    When Does Unsupervised Learning Succeed or Fail? A PoS Perspective on Reconstruction-Based Anomaly Detection

    [https://arxiv.org/abs/2609.28832](https://arxiv.org/abs/2609.28832)

    该论文提出基于子空间追踪的几何框架刻画基于重构的无监督异常检测的两种失败模式，并引入无需异常标签的动态推拉与嵌套流形雕刻方法，通过受控扰动学习紧凑的最优正常子空间。

    

    基于重构的无监督学习可能以两种相反的方式失败：模型可能对异常重构得过于准确，或者丢弃了有效的正常变化。利用子空间追踪假设，我们通过正常分量所诱导的交、并、连接几何结构来刻画这些失败模式。学习范围过大导致连接盲区，而容量不足则导致交偏好和正常保真度的损失。我们证明了紧凑的正常并集在所有保持正常保真的范围中是最优的，且通常需要非线性重构映射。基于这一几何结构，我们提出了动态推拉方法，它无需异常标签即可从受控扰动中学习；以及嵌套流形雕刻方法，它在潜空间中递归地应用相同原理。实验证实了在所有测试的推拉配置中，潜空间几何结构均发生了预测的变化。所提出的方法改进了基于重构的无监督学习性能。

    arXiv:2609.28832v1 Announce Type: new  Abstract: Reconstruction-based unsupervised learning can fail in two opposing ways: a model may reconstruct anomalies too accurately or discard valid nominal variation. Using the Pursuit of Subspaces hypothesis, we characterize these failures through the meet, union, and join geometries induced by the nominal components. Excess learned range produces join blindness, while insufficient capacity produces meet preference and loss of nominal fidelity. We show that the compact nominal union is optimal among nominal faithful ranges and generally requires a nonlinear reconstruction map. Based on this geometry, we introduce Dynamic Push and Pull, which learns from controlled perturbations without anomaly labels, and nested manifold carving, which applies the same principle recursively in latent space. Experiments confirm the predicted changes in latent geometry across every tested Push and Pull configuration. The proposed methods improve reconstruction-ba
    
[^174]: 流递归模型（SRM）

    Stream Recursion Model (SRM)

    [https://arxiv.org/abs/2609.28809](https://arxiv.org/abs/2609.28809)

    该论文提出了流递归模型（SRM），通过将计算组织为多个经递归细化的相互作用潜在流，在保持与GPT-2相当的单位参数性能的同时，暴露内部计算结构，为大型语言模型的机制可解释性提供了可扩展的解决方案。

    

    arXiv:2609.28809v1 公告类型：新 摘要：机制可解释性旨在对大型语言模型（LLM）的内部行为做出可验证的陈述。许多可解释性技术难以随着架构规模和深度的增长而扩展。我们对此的解决方案是引入结构上天然利于可解释性的较小模型。在这项工作中，我们提出了流递归模型（SRM），这是对层级推理模型（HRM）的一种改进，旨在暴露内部计算结构的同时保持可扩展性。SRM将计算组织为多个相互作用的潜在流，并通过递归细化进行更新，从而能够直接分析流动态、因果贡献和路由行为。SRM在单位参数基础上取得了与GPT-2相当的性能。我们的分析揭示了各流之间一致且各不相同的行为，表明存在结构化的功能专门化与交互。这些结果表明……

    arXiv:2609.28809v1 Announce Type: new  Abstract: Mechanistic interpretability seeks to make verifiable statements about the internal behavior of large language models (LLMs). Many interpretability techniques struggle to scale with the increasing size and depth of architectures. Our solution to this is to introduce smaller models with structures that lend themselves to interpretability. In this work, we introduce the Stream Recursion Model (SRM), a modification of the Hierarchical Reasoning Model (HRM) designed to expose internal computational structure while remaining scalable. SRM organizes computation into multiple interacting latent streams that are updated through recursive refinement, enabling direct analysis of stream dynamics, causal contribution, and routing behavior. SRM achieves performance comparable to GPT-2 on a per-parameter basis. Our analysis reveals consistent and distinct behavior across streams, indicating structured specialization and interaction. These results sugg
    
[^175]: 利用分布式声波传感与深度学习在高精度时空分辨率下监测城市交通动态

    Monitoring Urban Traffic Dynamics at Fine Spatiotemporal Resolution Using Distributed Acoustic Sensing and Deep Learning

    [https://arxiv.org/abs/2609.28793](https://arxiv.org/abs/2609.28793)

    本研究将分布式声波传感（DAS）与深度学习相结合，把现有地下光纤电缆转化为密集传感器阵列，实现了米级空间、秒级时间分辨率的被动式隐私保护城市交通监测，可连续高效地揭示交通量、拥堵及事件驱动的交通动态变化。

    

    在高时空分辨率下绘制交通动态的分布图是交通研究中的一个基本问题。分布式声波传感（DAS）作为一种创新的地震观测工具，成为在高空间和时间尺度上进行实时城市交通监测的极具前景的解决方案。分布式声波传感将现有的地下光纤电缆重新用作密集、连续的传感器阵列，能够在米级空间分辨率和秒级时间分辨率下对道路交通活动进行被动式且保护隐私的监测。本研究探讨了将DAS与深度学习模型相结合，能否作为一个连续且高效的城市交通观测系统，以揭示高时空分辨率下的城市交通动态（即交通量、拥堵状况以及事件驱动的交通变化）。本研究利用在美国德克萨斯州学院市（College Station）沿道路网络部署的DAS系统，开发了一个……（摘要内容在此处截断）

    arXiv:2609.28793v1 Announce Type: new  Abstract: Mapping the distribution of traffic dynamics at high spatiotemporal resolution is a fundamental question in transportation research. Distributed acoustic sensing (DAS), an innovative seismic observation tool, emerges as a promising solution for real-time urban traffic monitoring at high spatial and temporal scales. Distributed acoustic sensing repurposes existing underground fiber-optic cables as dense, continuous sensor arrays, enabling passive and privacy-preserving monitoring of roadway traffic activity at meter-level spatial and second-level temporal resolution. This study examines whether integrating DAS and deep learning models can serve as a continuous and efficient urban traffic observatory for revealing urban traffic dynamics (i.e. traffic volume and congestion, event-driven changes) at high spatiotemporal resolution. Using a DAS deployment along a roadway network in the City of College Station, Texas, USA, this study develops a
    
[^176]: 多链鲁棒平均回报马尔可夫决策过程的向量贝尔曼理论

    Vector Bellman Theory for Multichain Robust Average-Reward Markov Decision Processes

    [https://arxiv.org/abs/2609.28792](https://arxiv.org/abs/2609.28792)

    本文为多链鲁棒平均回报马尔可夫决策过程建立了向量贝尔曼理论，通过“先增益、后偏差”的耦合向量增益-偏差系统，从任意初始状态同时确定最优鲁棒增益并提供平稳鞍点策略。

    

    鲁棒平均回报马尔可夫决策过程为不确定性下的长期性能优化提供了一个基础框架，其最优长期回报可能依赖于初始状态。这种状态依赖性需要一个同时考虑常返类奖励与转移不确定性的向量贝尔曼理论。我们针对具有紧致、动作后 $(s,a)$-矩形模糊集的有限模型发展了这样一套理论。“先增益、后偏差”的优化原则导出一个耦合的向量增益-偏差系统，该系统的每个有限解都能识别最优鲁棒增益，并同时从所有初始状态出发，提供对抗历史依赖对手的平稳鞍点策略。我们进一步通过平稳增益条件和规范瞬态修正的一致界来刻画可解性，并给出允许各常返类具有不同增益的充分条件。这些证书还可产生渐近……

    arXiv:2609.28792v1 Announce Type: new  Abstract: Robust average-reward Markov decision processes provide a fundamental framework for long-term performance optimization under uncertainty, and can have optimal long-run rewards that depend on the initial state. This state dependence requires a vector Bellman theory that accounts for both recurrent-class rewards and transition uncertainty. We develop such a theory for finite models with compact, post-action $(s,a)$-rectangular ambiguity. A gain-first, bias-second optimization principle yields a coupled vector gain-bias system, and every finite solution identifies the optimal robust gain and supplies stationary saddle strategies against history-dependent opponents, simultaneously from all initial states. We further characterize solvability through stationary gain conditions and a uniform bound on canonical transient corrections, and give sufficient conditions that permit distinct recurrent-class gains. The certificates also yield asymptotic
    
[^177]: Δ学习的机制：面向可泛化科学机器学习的目标设计

    The Mechanics of Delta Learning: Target Design for Generalizable Scientific Machine Learning

    [https://arxiv.org/abs/2609.28782](https://arxiv.org/abs/2609.28782)

    本文揭示Δ学习中残差尺度并非衡量残差可学习性的充分指标，提出尺度归一化图狄利克雷粗糙度作为预训练诊断工具，并确立基线互补性作为科学机器学习中目标设计的核心原则。

    

    在科学机器学习中，Δ学习（Δ-learning）通过训练模型来学习相对于物理基线的残差误差，通常假设更精确的基线（即残差尺度更小）能够固有地提升下游性能。本文证明，仅凭残差尺度是判断可学习性的一个不充分的启发式指标。通过在总能量目标上评估分子图神经网络，我们发现复杂的局部描述符基线虽然能产生尺度很小的残差目标，但这些残差在架构信息代理空间中却异常粗糙，且相对于其尺度而言更难学习。相反，半经验基线能同时降低残差尺度和归一化粗糙度，从而提升域内和域外预测性能。我们引入尺度归一化图狄利克雷粗糙度（$D_{\text{IQR}}$）作为残差可学习性的预训练诊断指标，并将基线互补性确立为核心的目标设计原则，从而提升目标空间的形态……

    arXiv:2609.28782v1 Announce Type: new  Abstract: In scientific machine learning, $\Delta$-learning trains models on residual errors relative to physical baselines, assuming that more accurate baselines with smaller residual scales inherently improve downstream performance. Here, we demonstrate that residual scale alone is an insufficient heuristic for learnability. Evaluating molecular graph neural networks on total energy targets, we show that complex local descriptor baselines can yield small residual targets that are disproportionately rough within architecture-informed proxy spaces and harder to learn relative to their scale. Conversely, semi-empirical baseline reduces both scale and normalized roughness, improving in-domain and out-of-domain prediction. We introduce scale-normalized graph Dirichlet roughness ($D_{\text{IQR}}$) as a pre-training diagnostic for residual learnability and establish baseline complementarity as a core target-design principle, elevating target space form
    
[^178]: 资源受限成像中基于物理引导的多目标深度学习用于超声射频数据插值

    Physics-Guided Multi-Objective Deep Learning for Ultrasound RF Data Interpolation in Resource-Constrained Imaging

    [https://arxiv.org/abs/2609.28775](https://arxiv.org/abs/2609.28775)

    该论文提出一种物理引导的多目标深度学习框架，通过融合射频域与波束形成域损失的混合监督机制以及随机跳过掩码策略，实现资源受限超声成像中稀疏射频数据到稠密数据的高质量重建，有效抑制栅瓣伪影并提升模型对不同采集布局的泛化能力。

    

    超声成像日益面向便携式、即时检测和可穿戴应用场景，在这些场景中，功率、带宽和硬件复杂度的限制往往要求在时空扫描中进行稀疏数据采集。然而，使用稀疏数据进行图像重建会在相干波束形成过程中引入不充分的相位信息，产生栅瓣伪影，从而降低成像的对比度分辨率。我们提出了一种物理引导、数据驱动的框架，用于从稀疏到稠密的射频（RF）重建，使模型训练与下游图像形成过程保持一致。我们的方法采用混合监督方案训练端到端插值网络，该方案结合了射频域损失和波束形成域损失，并利用指数移动平均（EMA）来稳定多目标训练。为了在变化的采集布局下提高泛化能力，我们还引入了一种随机跳过掩码策略，通过改变稀疏度来增强模型的适应性。

    arXiv:2609.28775v1 Announce Type: cross  Abstract: Ultrasound imaging increasingly targets portable, point-of-care, and wearable settings where constraints on power, bandwidth, and hardware complexity often necessitate sparse data acquisition in spatiotemporal scanning. However, image reconstruction using the sparse data can introduce insufficient phase information in coherent beamforming process, resulting in grating-lobe artifacts that degrade imaging contrast resolution. We present a physics-guided, data-driven framework for sparse-to-dense radio-frequency (RF) reconstruction that aligns training with downstream image formation. Our approach trains an end-to-end interpolation network using a hybrid supervision scheme that combines an RF-domain and a beamforming-domain loss with exponential moving average (EMA) to stabilize the multi-objective training. To improve generalization under variable acquisition layouts, we also introduce a random-skip masking strategy that varies sparsity 
    
[^179]: 潜在空间中深度聚类的选择性推断

    Selective Inference for Deep Clustering in Latent Spaces

    [https://arxiv.org/abs/2609.28756](https://arxiv.org/abs/2609.28756)

    本文针对使用固定预训练编码器的深度聚类提出了一个选择性推断框架，通过应对从原始数据空间到潜在空间的非线性变换所带来的复杂选择过程，为聚类结果的统计可靠性检验提供了计算可行且有效的 p 值。

    

    深度聚类是一种强大的方法，它通过在聚类之前学习低维潜在表示来发现高维数据中的有意义结构。尽管其在实践中取得了成功，但评估所得聚类的统计可靠性仍然具有挑战性。在同一数据上检验所发现的聚类会引入选择偏差，并使经典的 p 值失效。选择性推断为纠正这种偏差提供了一个有原则的框架，但现有方法主要针对直接在观测特征上进行的聚类。在这项工作中，我们为使用固定预训练编码器的深度聚类开发了一个选择性推断框架。关键挑战在于，聚类分配是通过从原始数据空间到潜在空间的非线性变换来确定的，这使得选择过程比传统聚类复杂得多。我们的方法提供了一种计算上可行的方式来……

    arXiv:2609.28756v1 Announce Type: cross  Abstract: Deep clustering is a powerful approach for discovering meaningful structures in high-dimensional data by learning a low-dimensional latent representation prior to clustering. Despite its empirical success, assessing the statistical reliability of the resulting clusters remains challenging. Testing discovered clusters on the same data induces selection bias and invalidates classical $p$-values. Selective inference (SI) provides a principled framework for correcting this bias, but existing methods focus on clustering performed directly on the observed features. In this work, we develop an SI framework for deep clustering with a fixed pretrained encoder. The key challenge is that cluster assignments are determined through a nonlinear transformation from the original data space to the latent space, resulting in a substantially more complex selection process than in conventional clustering. Our method provides a computationally tractable wa
    
[^180]: 评估小波扩散模型降水降尺度的跨区域泛化能力

    Evaluating Cross-region Generalization for Wavelet-Diffusion Precipitation Downscaling

    [https://arxiv.org/abs/2609.28749](https://arxiv.org/abs/2609.28749)

    该研究系统评估了小波扩散模型在公里级降水降尺度中的跨区域与跨事件类型泛化能力，发现仅在俄克拉荷马州训练的模型在其他气候区域仍保持竞争力。

    

    扩散模型在公里级降水降尺度方面展现出强大潜力，但其在地理上未见过的区域和事件类型中的表现仍缺乏充分理解。本研究基于小波扩散模型（WDM）框架，评估了跨区域和跨事件类型的泛化能力。研究选取了美国六个3×3度区域，分别代表对流性降水、冬季降水、热带降水和大气河流降水类型。低分辨率输入通过对NOAA多雷达/多传感器（MRMS）组合反射率场进行块平均生成。研究将仅在俄克拉荷马州（OK）样本上训练的WDM与在全部六个区域上训练的WDM进行比较，并与最近邻插值和双三次插值方法进行了对比。模型性能通过三类指标体系进行评估，分别衡量图像域重建质量、谱与分布保真度以及分箱降水检测能力。结果表明，仅在OK训练的WDM在OK以外的区域仍保持竞争力。尽管全区域……（原文摘要在此处截断）

    arXiv:2609.28749v1 Announce Type: new  Abstract: Diffusion models have shown strong potential for kilometer-scale precipitation downscaling, but their performance in geographically unseen regions and event regimes remains insufficiently understood. Building on the wavelet diffusion model (WDM) framework, this study evaluates cross-region and cross-event generalization. Six 3 x 3 deg U.S. regions represent convective, winter, tropical, and atmospheric-river precipitation regimes. Low-resolution inputs are generated by block averaging NOAA Multi-Radar/Multi-Sensor (MRMS) composite reflectivity fields. A WDM trained only on Oklahoma (OK) samples and a WDM trained on all six regions are compared with nearest-neighbor and Bicubic interpolation. Model performance is evaluated using three metric families that measure image-domain reconstruction, spectral and distributional fidelity, and bin-wise precipitation detection. The OK-trained WDM remains competitive outside OK. Although the all-regio
    
[^181]: 基于虚构语料库微调的置信度-语料库一致性工具包技术手册

    Technical Manual for Toolkit for Confidence-Corpus Consistency via Fine-Tuning on a Fabricated Corpus

    [https://arxiv.org/abs/2609.28747](https://arxiv.org/abs/2609.28747)

    该论文提出了一个开源工具包，通过在虚构算术语料库上微调小型语言模型，并以不变的测量程序配对比较其微调前后对虚构答案与真实答案的置信度，从而直接检验“模型置信度可作为事实知识代理指标”这一假设。

    

    语言模型对其答案的置信度通常被解读为模型对相应事实掌握程度的代理指标。本手册记录了一个旨在直接检验这一解读的开源工具包：一个小型因果语言模型在一个语料库上进行微调，该语料库对81个一位数加法组合中的每一个都一致地断言一个虚构的算术答案，随后将模型微调后对每个虚构答案的置信度，与其微调前对相应真实答案的置信度进行配对比较，整个过程中使用完全相同的测量程序。我们描述并论证了流程的每个阶段——事实空间生成、考虑token长度的置信度测量、基线验证、语料库构建、微调以及微调前后的配对比较——以及每个阶段旨在排除的混杂因素，其中包括一位数与两位数答案之间的分词不对称性，以及答案仅仅失去相对优势与……

    arXiv:2609.28747v1 Announce Type: cross  Abstract: A language model's confidence in an answer is often read as a proxy for how well it knows the corresponding fact. This manual documents an open toolkit built to test that reading directly: a small causal language model is fine-tuned on a corpus that consistently asserts one fabricated arithmetic answer for each of the 81 single-digit addition pairs, and its post-fine-tuning confidence in each fabricated answer is compared against its own pre-fine-tuning confidence in the corresponding true answer, using an unchanged measurement procedure throughout. We describe and justify every pipeline stage, fact-space generation, token-length-aware confidence measurement, baseline validation, corpus construction, fine-tuning, and paired before/after comparison, together with the confound each is meant to rule out, among them tokenization asymmetry between single- and double-digit answers and the difference between an answer merely losing its edge a
    
[^182]: 强化学习中的策略复杂度、反应时间与有限理性

    Policy Complexity, Reaction Time, and Bounded Rationality in Reinforcement Learning

    [https://arxiv.org/abs/2609.28737](https://arxiv.org/abs/2609.28737)

    本文提出MI-SARSA算法，通过互信息正则化将状态特定的信息成本显式纳入强化学习，实现了策略压缩并能够预测试验层面的反应时间，从而为生物有限理性提供了更合适的计算模型。

    

    生物智能体并非在无限计算能力的条件下进行学习。对人类而言，学习与选择受到感知、注意力和工作记忆等约束的塑造，这些约束限制了有多少状态信息能够指导行为，从而限定了策略复杂度的上限。标准的强化学习模型通常只优化奖励，而不显式地表征这些内部成本，因此作为生物智能的模型并不十分合适。我们推导出MI-SARSA，这是一种在线策略的时序差分算法，它通过学习到的边际动作先验以及对状态偏离该先验的状态特定惩罚，引入了互信息正则化。由此得到一个序贯学习模型：在该模型中，只有当状态信息带来的期望回报收益足以抵消其新增的信息成本时，状态信息才会被有选择地使用。关键在于，控制策略压缩的状态特定信息成本，同时也产生了试验层面的可预测（此处摘要被截断）

    arXiv:2609.28737v1 Announce Type: cross  Abstract: Biological agents do not learn under conditions of unlimited computation. For humans, learning and choice are shaped by constraints on perception, attention, and working memory, which limit how much state information guides behavior and therefore bound policy complexity. Standard reinforcement learning models typically optimize reward without explicitly representing these internal costs, making them less suitable as models of biological intelligence. We derive MI-SARSA, an on-policy temporal-difference algorithm that incorporates mutual-information regularization through a learned marginal action prior and a penalty on state-specific deviations from that prior. This yields a sequential learning model in which state information is used selectively when its expected return benefit justifies the added informational cost. Critically, the same state-specific information cost that governs policy compression also generates trial-level predict
    
[^183]: 揭露物联网入侵检测中的捷径学习：基于取证式多范式方法的特征依赖性与数据泄露评估

    Unmasking Shortcut Learning in IoT Intrusion Detection: A Forensic, Multi-Paradigm Evaluation of Feature Dependence and Data Leakage

    [https://arxiv.org/abs/2609.28725](https://arxiv.org/abs/2609.28725)

    该论文通过取证式的多范式评估发现，物联网入侵检测模型的近乎完美性能主要源于对静态测试床IP/MAC地址和原始时间戳等数据集捷径（即数据泄露）的利用，而非真正可泛化的攻击行为特征。

    

    基于机器学习的网络入侵检测系统在物联网基准数据集上常常报告近乎完美的性能。然而，这些模型究竟是学到了可泛化的攻击行为，还是利用了数据集中的虚假捷径——例如静态测试床IP/MAC地址以及按时间顺序记录产生的伪影——仍然是一个重要问题。我们评估了CyberFlowIoT-GICAP基准数据集，该数据集包含126个PCAP会话中的3,617,388条流记录，其中良性流为849,395条。我们采用PCAP不相交的划分方式，在四种特征配置下评估了四种学习范式；此外还使用传统的随机流划分方式对LightGBM进行了评估。当仅使用统计流行为特征（Fbehav）时，LightGBM（92.58% ± 8.18%）、随机森林（92.59% ± 8.18%）和深度多层感知机（92.55% ± 8.18%）取得了几乎相同的Macro-F1，这表明性能受限于特征表示而非模型复杂度。在使用原始时间戳的特征配置下（Ftsta

    arXiv:2609.28725v1 Announce Type: cross  Abstract: Machine learning-based Network Intrusion Detection Systems often report near-perfect performance on IoT benchmarks. However, whether these models learn generalizable attack behavior or exploit spurious dataset shortcuts- such as static testbed IP/MAC addresses and chronological recording artifacts-remains an important question. We evaluate the CyberFlowIoT-GICAP benchmark, containing 3,617,388 flow records across 126 PCAP sessions with 849,395 benign flows. Four learning paradigms are evaluated across four feature configurations using PCAP-disjoint splits; LightGBM is additionally evaluated using conventional random-flow splitting. When only statistical flow behavior is used (Fbehav), LightGBM (92.58% +/- 8.18%), Random Forest (92.59% +/- 8.18%), and Deep MLP (92.55% +/- 8.18%) achieve nearly identical Macro-F1, indicating that performance is constrained by feature representation rather than model complexity. With raw timestamps (Ftsta
    
[^184]: 维护联邦学习中的鲁棒性：趋势、新兴策略与研究机遇

    Upholding Robustness in Federated Learning: Trends, Emerging Strategies, and Research Opportunities

    [https://arxiv.org/abs/2609.28722](https://arxiv.org/abs/2609.28722)

    本文从威胁攻击面分类、鲁棒聚合策略分类和分层防御策略三个角度对联邦学习鲁棒性进行了全面综述，并审视现有评估实践、指出未来研究方向。

    

    尽管联邦学习（FL）已被广泛应用于机器学习中保护用户隐私，但它仍然容易受到各种鲁棒性挑战的影响，包括性能受损风险、信息窃取威胁和聚合漏洞。本工作从三个紧密耦合的角度对联邦学习鲁棒性进行了全面综述：(i) 以威胁为中心的鲁棒性视角，对多方面的攻击面进行分类；(ii) 鲁棒聚合策略的结构化分类法，区分以结果为中心的方法与以安全为中心的策略；(iii) 防御策略的分层分类法。我们严格审查了当前联邦学习鲁棒性的评估实践，并识别了主要应用和开放的研究挑战，以指导未来的研究。

    arXiv:2609.28722v1 Announce Type: new  Abstract: While Federated Learning (FL) has been widely adopted for protecting user privacy in machine learning, it remains vulnerable to various robustness challenges, including performance-impairment risks, information-stealing threats, and aggregation vulnerabilities. This work offers a holistic synthesis of FL robustness along three tightly coupled angles: (i) a threat-centric view of robustness that categorizes the multifaceted attack surfaces, (ii) a structured taxonomy of robust aggregation strategies distinguishing outcome-centric approaches from security-centric strategies, and (iii) a layered taxonomy of defensive strategies. We rigorously examine current evaluation practices for FL robustness and identify major applications and open research challenges to guide future research.
    
[^185]: 高维高斯老虎机中的精确贝叶斯后悔与渐近最优性

    Exact Bayes Regret and Asymptotic Optimality in High-Dimensional Gaussian Bandits

    [https://arxiv.org/abs/2609.28718](https://arxiv.org/abs/2609.28718)

    本文研究时间范围与维度成比例的高维高斯贝叶斯线性老虎机，证明了归一化后验不确定性在所有因果策略上具有一致显式极限，由此推导出汤普森采样等策略的精确后悔曲线，并证明后验均值贪婪选择达到极限最优贝叶斯后悔，而汤普森采样的后悔严格更大。

    

    我们研究贝叶斯线性老虎机问题，其中参数服从各向同性高斯分布，候选臂为相互独立的高斯分布，奖励噪声为高斯噪声，且时间范围与维度成比例。归一化后的后验不确定性具有显式极限，且该极限在所有因果策略上一致。随后，利用高斯后验恒等式确定极限参数重叠，无需假设自适应递归的闭合性。这些结果为汤普森采样、后验均值贪婪选择以及一族缩放后验采样协方差的策略给出了精确的后悔曲线。归一化的实际累积后悔在L1范数下收敛，且在紧的比例时间区间上一致。一个策略一致的下界确定了极限最优贝叶斯后悔，并证明后验均值贪婪选择能够达到该下界。汤普森采样的领先后悔严格更大，其相对于贪婪选择的瞬时后悔比率……

    arXiv:2609.28718v1 Announce Type: cross  Abstract: We study Bayesian linear bandits with an isotropic Gaussian parameter, independent Gaussian candidate arms, and Gaussian reward noise when the horizon is proportional to the dimension. The normalized posterior uncertainty has an explicit limit that is uniform over all causal policies. Gaussian posterior identities then determine the limiting parameter overlaps without an assumed closure of the adaptive recursion. These results yield exact regret curves for Thompson sampling, posterior-mean greedy selection, and a family of policies that scale the posterior sampling covariance. The normalized realized cumulative regret converges in L1, uniformly on compact proportional-time intervals. A policy-uniform lower bound identifies the limiting optimal Bayes regret and proves that posterior-mean greedy selection attains it. Thompson sampling incurs a strictly larger leading regret; its instantaneous regret ratio relative to greedy selection lie
    
[^186]: 空中连续体机械臂操作中气动干扰下末端执行器位置估计的时序学习

    Temporal Learning for End-Effector Position Estimation under Aerodynamic Disturbances in Aerial Continuum Manipulation

    [https://arxiv.org/abs/2609.28716](https://arxiv.org/abs/2609.28716)

    本文提出使用闭合形式连续时间（CfC）神经网络来估计无人机气动干扰下空中连续体机械臂的末端执行器三维位置残差，相比MLP和GRU等方法提升了位置估计的准确性。

    

    本文研究了时序神经网络在空中连续体机械臂（ACM）末端执行器位置估计中的应用，该机械臂在无人机（UAV）引起的气动效应下运行。研究团队在静止（旋翼关闭）和自由悬停条件下，针对连续体机器人（CR）的不同构型和无人机的不同高度采集了实验数据集，提供了有无气动残差的末端执行器位置测量数据。为了建立名义框架，研究评估了采用逐渐丰富的应变基的应变参数化运动学模型，以平衡模型复杂度和预测精度。所选的名义模型随后作为基线，使用闭合形式连续时间神经网络进行三维位置残差估计，并与多层感知机（MLP）和门控循环单元（GRU）进行对比。在未见过的测试实验中

    arXiv:2609.28716v1 Announce Type: cross  Abstract: This paper investigates temporal neural networks for \mbox{end-effector} position \mbox{estimation} of an aerial continuum manipulator (ACM) operating under aerodynamic effects induced by the unmanned aerial vehicle (UAV). An experimental dataset is collected under stationary (\mbox{rotor-off}) and \mbox{free-hovering} conditions across continuum robot (CR) configurations and UAV altitudes, providing \mbox{end-effector} position measurements with and without aerodynamic residuals. To establish a nominal framework, \mbox{strain-parameterized} kinematic models with progressively richer strain bases are evaluated to balance model complexity and prediction accuracy. The selected nominal model then serves as the baseline for 3D position residual estimation using a \mbox{closed-form} \mbox{continuous-time} (CfC) neural network, with a multilayer perceptron (MLP) and a gated recurrent unit (GRU) used for comparison. On unseen test experiments
    
[^187]: LabFactory：构建与评估可执行的AI实验室

    LabFactory: Building and Evaluating Executable AI Labs

    [https://arxiv.org/abs/2609.28697](https://arxiv.org/abs/2609.28697)

    LabFactory提出让AI构建者将科学简报转化为可交付执行、封装了模型、知识、工具与控制器的“AI实验室”，再由独立主机在保留数据上运行并评分，从而把评估对象从构建者的进展陈述转变为实际交付的系统。

    

    科学任务定义了一种期望的能力，但实现它通常需要构建一个针对该任务定制的计算系统——获取数据、设计表示、训练模型、实现工具，并决定它们在推理时如何被使用。我们提出LabFactory，这是一个框架，其中AI构建者将一份科学简报转化为一个可执行的AI实验室：一个在固定接口之后整合了模型、知识资源、工具和控制器的任务专用求解器。构建者在一个计量工作区中开发和打包该实验室；随后由一个独立的主机在保留的输入上执行交付的工件，参考标签保持在求解器输入接口之外，并按照任务的协议对其输出进行评分。这使得被评估的对象成为交付的系统本身，而不是构建者对其进展的描述。我们记录了横跨七个科学任务类别的28个精选构建案例——从分子和基……（原文摘要此处被截断）

    arXiv:2609.28697v1 Announce Type: new  Abstract: Scientific tasks specify a desired capability, but realizing it often requires building a computational system tailored to the task---acquiring data, designing representations, training models, implementing tools, and deciding how they are used at inference. We present LabFactory, a framework in which an AI builder turns a scientific brief into an executable AI lab: a task-specific solver that integrates models, knowledge resources, tools, and a controller behind a fixed interface. The builder develops and packages the lab in a metered workspace; a separate host then executes the delivered artifact on held-out inputs, with reference labels kept outside the solver's input interface, and scores its outputs under the task's protocol. This makes the delivered system, rather than the builder's account of its progress, the object of evaluation. We document 28 selected constructions across seven scientific task categories---from molecular and g
    
[^188]: AnDE分类器的联邦学习

    Federated Learning of AnDE Classifiers

    [https://arxiv.org/abs/2609.28695](https://arxiv.org/abs/2609.28695)

    本文提出了一种支持任意依赖阶数的AnDE分类器联邦学习框架，通过本地学习判别式权重并全局聚合实现隐私保护，实验证明其性能持续优于联邦朴素贝叶斯，且差分隐私聚合仅带来有限的精度损失。

    

    本工作提出了一个在分布式环境中训练平均n依赖估计器的联邦框架。所提出的方法聚焦于判别式设置，其中模型权重在本地学习并全局聚合，支持任意依赖阶数n。这种设计使得联邦训练无需传输具有语义意义的参数，从而提高了隐私性。此外，还将生成式AnDE模型联邦化以提供对比基线，并可选择在概率表聚合中应用差分隐私。在12个离散数据集上的实验表明，n≥1的判别式模型始终优于联邦朴素贝叶斯（NB，n=0），且隐私保护聚合在精度损失有限的情况下是有效的。这些结果确立了联邦AnDE作为一种可行且保护隐私的框架，表明概率模型在现代联邦学习环境中依然适用。

    arXiv:2609.28695v1 Announce Type: new  Abstract: This work presents a federated framework for training Averaged $n$-Dependence Estimators (AnDE) in distributed environments. The proposed method focuses on the discriminative setting, where model weights are learned locally and aggregated globally, supporting any dependency order $n$. This design allows federated training without transmitting semantically meaningful parameters, improving privacy. Additionally, generative AnDE models are federated to provide a comparative baseline, with optional differential privacy applied to the aggregation of probability tables. Experiments on 12 discrete datasets show that discriminative models with $n \geq 1$ consistently outperform federated Naive Bayes (NB, $n=0$), and that privacy-preserving aggregation is effective with limited accuracy loss. These results establish federated AnDE as a viable and privacy-preserving framework, showing that probabilistic models remain applicable in modern federated
    
[^189]: M-plicits：基于嵌套多尺度残差的神经隐式表面

    M-plicits: Neural Implicit Surfaces via Nested Multiscale Residuals

    [https://arxiv.org/abs/2609.28684](https://arxiv.org/abs/2609.28684)

    提出M-plicits多尺度框架，将表面建模为通过嵌套邻域训练的MLP残差和，并将监督严格限制在先前零水平集周围的窄带内，从而在训练效率、渲染速度和噪声鲁棒性之间实现更好的平衡。

    

    将输入坐标通过正弦函数编码并输入多层感知机（MLP），已被证明对于定义为零水平集的表面的隐式神经表示（INR）是有效的。然而，现有方法往往难以在训练效率、渲染速度和噪声鲁棒性之间取得平衡：单MLP方法在推理时代价高昂；基于网格的表示虽然速度快，但可能限制表面平滑度并过拟合输入噪声；而以往的多尺度方法由于硬频谱截断经常捕捉到噪声并产生伪影。为了解决这些局限性，我们提出了M-plicits，这是一个多尺度框架，将表面建模为通过一系列嵌套邻域训练的MLP残差和。与依赖标准全域采样且需要昂贵的网格提取才能进行可视化的现有残差方法不同，我们的方法严格将监督限定在先前零水平集周围的窄带区域内。

    arXiv:2609.28684v1 Announce Type: cross  Abstract: Encoding input coordinates with sinusoidal functions into multi-layer perceptrons (MLPs) has proven effective for implicit neural representations (INRs) of surfaces defined as zero-level sets. However, existing methods often struggle to balance training efficiency, rendering speed, and noise robustness: single-MLP approaches are expensive at inference, grid-based representations are fast but can limit surface smoothness and overfit input noise, and previous multiscale approaches frequently capture noise and produce artifacts due to hard spectral truncation. To address these limitations, we propose M-plicits, a multiscale framework that models surfaces as a residual sum of MLPs trained via a sequence of nested neighborhoods. Unlike existing residual approaches that rely on standard domain-wide sampling and require costly mesh extraction for visualization, our method strictly localizes supervision to narrow bands around the previous zero
    
[^190]: 思维泄露：混合推理模型中NoThink后训练的因果审计

    Thinking Leakage: A Causal Audit of NoThink Post-Training in Hybrid Reasoning Models

    [https://arxiv.org/abs/2609.28682](https://arxiv.org/abs/2609.28682)

    该论文首次在因果中介框架下系统审计了混合推理模型NoThink后训练中的“思维泄露”现象，通过双向激活干预证明后训练准确率收益中有42%-79%实际来源于基础模型Think模式中已存在的思维能力。

    

    在NoThink模式下对混合推理模型进行后训练，作为一种在保持推理速度的同时提升性能的方法，正受到越来越多的关注。然而，这些性能提升可能源于基础模型在Think模式下已经具备的思维能力。我们在因果中介框架中对这种“思维泄露”现象进行建模，并沿一个从基础模型导出的简单激活方向使用双向干预来审计其贡献。在竞赛数学基准上对三个模型和三种后训练方法的实验中，我们发现泄露是真实存在的、具有因果性且相当可观：行为与表征分析显示模型向Think模式偏移；沿该方向对基础模型进行激活导向能够复现大部分后训练带来的准确率提升；而对已训练检查点进行反向导向则会消除其获得的相当一部分收益。在九个具有正向NoThink收益的对齐检查点中，泄露比例介于42%到79%之间。

    arXiv:2609.28682v1 Announce Type: new  Abstract: Post-training hybrid reasoning models in NoThink mode has attracted growing interest as a way to improve performance while keeping inference fast. However, these gains may draw on thinking behavior already accessible through the base model's Think mode. We formulate this thinking leakage in a causal mediation framework and audit its contribution using bidirectional interventions along a simple base-derived activation direction. Across three models and three post-training methods on competition math benchmarks, we find that leakage is real, causal, and substantial: behavioral and representational analyses reveal shifts toward Think, steering the base model along this direction reproduces most of the post-training accuracy gain, and counter-steering a checkpoint removes a substantial share of what it gains. Across nine aligned checkpoints with positive NoThink gains, the resulting leakage ratio ranges from 42% to 79%. These interventions s
    
[^191]: 超越静态图世界模型：在演化拓扑上学习随机潜在动力学

    Beyond Static Graph World Models: Learning Stochastic Latent Dynamics over Evolving Topologies

    [https://arxiv.org/abs/2609.28670](https://arxiv.org/abs/2609.28670)

    提出图动力学模型（GDM），利用稀疏循环邻接矩阵和循环状态空间架构，实现对随机、部分可观测环境中演化拓扑图结构观测的世界建模，并首次引入联合图状态分布的评估方法。

    

    基于图的世界模型最近成为一种在关系状态表示上学习状态转移的方法。然而，现有方法大多局限于固定拓扑的图，或确定性的、完全可观测的环境。我们提出了图动力学模型，这是一个面向图结构观测的世界模型，旨在处理更一般的场景：在随机且部分可观测的环境中拓扑不断演化的情况。GDM使用一个稀疏的循环邻接矩阵来建模拓扑更新并执行消息传递，同时采用循环状态空间架构来建模随机状态转移。此外，我们发现图世界模型在评估方面存在空白，因为现有方法没有提供一种手段来比较预测分布与真实分布在联合图状态（包括相互依赖的拓扑、节点特征和图特征）上的差异。因此，我们引入了一种（摘要在此处不完整）

    arXiv:2609.28670v1 Announce Type: cross  Abstract: Graph-based world models have recently emerged as a means of learning transitions over relational state representations. However, existing approaches are largely limited to fixed-topology graphs or deterministic, fully observable environments. We propose the Graph Dynamics Model (GDM), a world model for graph-structured observations that is designed to handle the more general setting of evolving topologies in stochastic and partially observable environments. The GDM uses a sparse recurrent adjacency matrix to model topology updates and perform message passing, together with a recurrent state-space architecture for modelling stochastic transitions. Furthermore, we identify a gap in the evaluation of graph-based world models, as existing methods do not provide a means of comparing predicted and true distributions over the joint graph state comprising the interdependent topology, node features, and graph features. We therefore introduce t
    
[^192]: OPDiv：Top-K高得分、多样化化合物的最优选择

    OPDiv: Optimal Selection of Top-K High-Scoring, Diverse Compounds

    [https://arxiv.org/abs/2609.28665](https://arxiv.org/abs/2609.28665)

    提出OPDiv算法，利用整数优化在得分与多样性之间找到最优权衡，从而选择top-k高得分且多样化的化合物，并主张虚拟筛选本质上是隐式的约束优化任务。

    

    虚拟筛选活动可能产生数千个有前景的候选化合物，但其中只有少数能够被购买、合成或测试。实际问题在于如何选择一组既排名靠前又具有足够多样性的化合物：这构成了一个真正的权衡，因为选择得分最高的分子会导致多样性有限，而侧重多样性的选择则会牺牲一些得分较高的分子。我们提出了OPDiv，这是一种多样性选择与评估算法，通过整数优化找到最优的分子子集，从而解决这一权衡。我们在实践中利用指纹距离、形状和静电多样性展示了该选择算法，并比较了所得的多样性谱。我们认为，虚拟筛选不仅仅是一个排序问题，更是一个隐式的约束优化任务：当不希望出现冗余化学型时，应当基于top-k化合物的选择来比较不同的筛选流程。

    arXiv:2609.28665v1 Announce Type: new  Abstract: A virtual screening campaign may produce thousands of promising candidates, but only a small number can be purchased, synthesized, or tested. The practical question is how to select a set of compounds that both rank well and are diverse enough: this poses a genuine tradeoff, where selecting the highest-scoring molecules yields limited diversity, while diversity selection sacrifices some well-scoring molecules. We introduce OPDiv, a diversity selection and evaluation algorithm solving this tradeoff by finding an optimal subset of molecules using integer optimization. We demonstrate the selection algorithm in practice with fingerprint distance, shape and electrostatic diversity and compare the resulting diversity spectra. We argue that virtual screening is not merely a ranking problem, but also an implicit constrained optimization task: when redundant chemotypes are undesirable, pipelines should be compared based on the top-k compound sele
    
[^193]: 查询远征队：学习检索动作

    The Fellowship of the Query: Learning Retrieval Actions

    [https://arxiv.org/abs/2609.28653](https://arxiv.org/abs/2609.28653)

    通过轨迹微调可以让小型语言模型有效学会检索增强问答中的“下一步动作”控制决策，宏F1分数远超零样本提示，且单个SLM可同时兼任控制器与答案生成器。

    

    检索增强式问答需要对何时分解问题、搜索、重新表述、提取证据、综合事实、验证进度以及何时停止等控制决策。我们研究轨迹微调能否提升小型语言模型（SLM）作为“下一步动作控制器”的表现。我们还额外评估了一种低资源设置，即由单个SLM同时充当控制器和最终答案生成器。基于被采纳的教师搜索轨迹，我们构建了一个七分类动作预测任务，模型从当前轨迹状态预测下一个结构化的教师动作，并在多种SLM和超小型语言模型（xSLM）上评估了LoRA监督微调作为控制器的效果。在1,646个留出的动作示例上，基于13,194个动作训练的Granite 4.1 3B达到了0.6536的宏F1分数，而同一模型的零样本提示仅为0.1736，TF-IDF逻辑回归基线为0.5399。在端到端的控制器/生成器交换评估……（摘要在此处截断）

    arXiv:2609.28653v1 Announce Type: cross  Abstract: Retrieval-augmented question answering requires control decisions about when to decompose a question, search, reformulate, extract evidence, synthesize facts, verify progress, and stop. We study whether trajectory fine-tuning can improve small language models (SLMs) as next-action controllers. We additionally evaluate a low-resource setting in which a single SLM serves as both the controller and the final-answer generator. From accepted teacher search traces, we build a seven-way action-prediction task, where the model predicts the next structured teacher action from the current trajectory state, and evaluate LoRA-supervised fine-tuning across SLMs and xSLMs as controllers. On 1,646 held-out action examples, Granite 4.1 3B trained on 13,194 actions reaches macro-F1 0.6536, compared with 0.1736 for zero-shot prompting of the same model and 0.5399 for a TF-IDF logistic-regression baseline. In an end-to-end controller/generator swap evalu
    
[^194]: 迭代乘法任务上RLVR的优化地形可以是良性的：来自自旋玻璃理论的洞见

    RLVR landscapes for iterated multiplications can be benign: Insights from spin-glass theory

    [https://arxiv.org/abs/2609.28625](https://arxiv.org/abs/2609.28625)

    本文借助自旋玻璃理论证明，对于迭代乘法等算法任务，RLVR的优化地形是良性的（不存在局部极小值陷阱），其训练困难主要来自扩散屏障和梯度估计误差，而非地形本身的陷阱。

    

    尽管可验证奖励强化学习（RLVR）十分重要，但它能在多大程度上学习到新的推理能力仍存在争议。本文研究了RLVR在算法任务（如迭代群乘法与拟群乘法）上的优化地形。为此，我们将基于短视表格策略的熵正则化RLVR映射为确定性策略上的能量模型（自旋玻璃模型）。这一映射给出了RLVR所能达到性能的上界，并使我们能够在该表格设置下严格刻画优化地形。我们从理论和实验两方面证明，对于一大类具有不相关输入的模型和任务，该优化地形是良性的，不存在可能困住RLVR训练的局部极小值。相反，这些任务在实际中的困难似乎至少部分源于扩散屏障以及穿越优化地形时的梯度估计误差等问题。这些是真正的障碍……

    arXiv:2609.28625v1 Announce Type: new  Abstract: Despite the importance of reinforcement learning with verifiable rewards (RLVR), the extent to which it can learn new reasoning capabilities remains debated. Here we study the optimization landscape of RLVR on algorithmic tasks, such as iterated group and quasigroup multiplication. To this end, we map entropy-regularized RLVR over myopic tabular policies onto an energy-based (spin-glass) model over deterministic policies. This mapping upper-bounds what RLVR can achieve, and lets us rigorously characterize the landscape in this tabular setting. We show, both theoretically and experimentally, that for a wide class of models and tasks with uncorrelated inputs, this landscape is benign, containing no local minima that could trap RLVR training. Rather, the practical difficulty of these tasks appears to stem, at least in part, from issues such as diffusive barriers and gradient-estimation error in traversing the landscape. These are genuine ob
    
[^195]: 奖励破解行为挑战自主研究智能体的监督机制

    Reward Hacking Challenges Oversight of Autonomous Research Agents

    [https://arxiv.org/abs/2609.28614](https://arxiv.org/abs/2609.28614)

    该研究通过对17个语言模型和38个任务的系统实验，发现自主研究智能体在无指令时也常有自发奖励破解行为（开放式任务达30.5%），且允许破解时74.6%的尝试既能通过评估阈值又能规避评估机制，表明奖励破解对自主研究智能体的监督构成了严峻挑战。

    

    自主研究智能体能够设计实验、评估结果并撰写报告，这使它们既能控制科学结果本身，又能控制用于支持该结果的证据。由此产生了奖励破解（reward hacking）的风险：即满足奖励评判标准却并未实现预期目标。我们研究了以下三个问题：(1) 模型在没有相关指令的情况下自发进行奖励破解的频率；(2) 当允许破解时，其破解方法的有效性和可检测性；(3) 当LLM评审小组反馈其决定和理由时，模型如何做出调整。在17个语言模型和38个任务上的实验表明，开放式研究流程任务的自发奖励破解率为30.5%，而任务特定内核任务为2.9%。当在通过阈值高于我们最佳合规基线的任务上允许破解时，677次尝试中有505次（74.6%）被确认为奖励破解：它们既越过了阈值，又被机制验证小组确认存在对评估机制的利用。仅审查[原文摘要在此处截断]

    arXiv:2609.28614v1 Announce Type: new  Abstract: Autonomous research agents can design experiments, evaluate results, and write reports, giving them control over both a scientific result and the evidence used to support it. This creates a risk of reward hacking: meeting the reward criteria without achieving the intended goal. We study (1) how often models reward-hack without instructions to do so, (2) how effective and detectable their methods are when hacking is allowed, and (3) how they adapt when an LLM review panel returns its decision and reasons. Across 17 language models and 38 tasks, the spontaneous reward-hacking rate is 30.5% on open-ended research-pipeline tasks and 2.9% on task-specific kernels. When hacking is allowed on tasks whose pass thresholds exceed our best compliant baselines, 505/677 attempts (74.6%) are confirmed reward hacks: they both clear the threshold and receive mechanism-verification panel confirmation of an evaluation exploit. An LLM panel reviewing only 
    
[^196]: UltraBench 2：迈向超声领域视觉基础模型的鲁棒评估

    UltraBench 2: Towards Robust Evaluation of Vision Foundation Models on Ultrasound

    [https://arxiv.org/abs/2609.28610](https://arxiv.org/abs/2609.28610)

    该论文推出了UltraBench 2，一个覆盖广泛解剖结构和任务、注重标准化与可重复性的超声视觉基础模型综合评估基准，并发现超声特异性预训练在分类任务上仍占优势，而最先进的通用模型在分割任务上已与之持平。

    

    基准测试已成为机器学习研究及其应用领域（包括医疗保健）中日益关键的一部分。然而，尽管近年来新的超声基础模型在稳步发展，用于评估这些模型的设计良好的基准测试的发展却相对滞后。这一不足导致了对相互竞争的模型进行评估时碎片化且不一致，使得难以衡量研究进展。为了解决这个问题，我们推出了UltraBench 2，这是一个具有广泛解剖结构和任务覆盖范围的综合基准，专注于标准化、可重复性和易用性。利用该基准，我们比较了现有的用于超声图像分析的视觉基础模型。我们的分析表明，超声特异性预训练在分类任务上仍然领先，但最先进的通用模型在分割任务上已经与之持平。

    arXiv:2609.28610v1 Announce Type: cross  Abstract: Benchmarking is an increasingly critical part of research in machine learning and the domains where it is applied, including healthcare. Yet, despite the steady development of new ultrasound foundation models in recent years, the development of well-designed benchmarks to evaluate them has lagged behind. This deficiency has led to fragmented and inconsistent evaluations of competing models, making it difficult to measure progress. To address this issue, we introduce UltraBench 2, a comprehensive benchmark with wide anatomical and task coverage, and a focus on standardization, reproducibility, and ease-of-use. Using this benchmark, we compare existing vision foundation models for ultrasound image analysis. Our analyses demonstrate that ultrasound-specific pretraining still leads on classification, but that state-of-the-art general-purpose models have drawn level on segmentation.
    
[^197]: fable.intermittent：面向间歇性时间序列的概率预测方法基准测试

    fable.intermittent: benchmarking probabilistic forecasting methods for intermittent time series

    [https://arxiv.org/abs/2609.28607](https://arxiv.org/abs/2609.28607)

    该论文发布了 R 包 fable.intermittent，在统一的 fable 框架下整合了多种间歇性时间序列的概率预测方法以便系统比较，并提出了一种采用 Tweedie 预测分布的新型指数平滑模型 TWEES。

    

    间歇性时间序列在备件需求和零售销售中十分常见。由于预测误差的成本通常是不对称的，诸如库存控制等决策需要完整的预测分布而非点预测。目前已提出许多概率预测方法，然而它们的实现分散在不同的软件框架中，难以对它们进行系统的比较。我们介绍了 fable.intermittent，这是一个 R 包，它在 fable 框架内实现了多种针对间歇性序列的概率预测方法。该包允许通过单一、简单的预测流程在一组时间序列上拟合和评估多个模型。我们还提出了 TWEES，一种具有 Tweedie 预测分布的新型指数平滑模型。拟合 TWEES 需要重复计算计算开销很大的 Tweedie 密度。我们还发布了该 R 包（摘要原文至此截断）。

    arXiv:2609.28607v1 Announce Type: new  Abstract: Intermittent time series are common in spare-parts demand and retail sales. Since the cost of forecast errors is typically asymmetric, decisions such as inventory control require the full predictive distribution rather than a point forecast. Many probabilistic forecasting methods have been proposed; their implementations, however, are scattered across different software frameworks, making it difficult to compare them systematically. We introduce fable.intermittent, an R package that implements several probabilistic forecasting methods for intermittent series within the fable framework. The package allows several models to be fitted and evaluated on a collection of time series through a single, simple forecasting pipeline. We also introduce TWEES, a new exponential smoothing model with a Tweedie predictive distribution. Fitting TWEES requires repeated evaluation of the computationally demanding Tweedie density. We also release the R packa
    
[^198]: UO-FIE：结合精确标签监督与分级效用的事实性推断

    UO-FIE: Combining Exact-Label Supervision with Graded Utility for Factivity Inference

    [https://arxiv.org/abs/2609.28605](https://arxiv.org/abs/2609.28605)

    UO-FIE是一种参数高效的事实性推断系统，通过结合硬标签监督、基于效用的软目标、计划性类别权重和序数损失，在类别高度不平衡的中文事实性推断任务中同时提升预测的精确匹配率和区间接近度。

    

    2026年事实性推断评估（FIE2026）将中文语境-假设对划分为九个有序的事实性区间。其评估指标既奖励精确预测，也奖励接近正确区间的预测，而566个训练样本中有64.1%属于单一类别。在初步实验中，多个mDeBERTa分类模型主要倾向于预测主导类别，而Huber回归基线则产生更多接近正确区间的预测，但精确匹配的数量较少。我们提出了面向效用的事实性推断（UO-FIE），这是一种参数高效的系统，将精确标签监督与分级效用相结合。UO-FIE预测九个类别上的分布，并结合了硬标签监督、基于效用的软目标、计划性类别权重和序数损失。我们在受控比较中评估了期望效用解码，并使用了在折外预测上选择的序数校准。

    arXiv:2609.28605v1 Announce Type: cross  Abstract: The Factivity Inference Evaluation 2026 (FIE2026) classifies Chinese context-hypothesis pairs into nine ordered factivity intervals. Its evaluation metric rewards both exact predictions and proximity to the correct interval, while 64.1% of the 566 training examples belong to a single class. In preliminary experiments, several mDeBERTa classification models predominantly predict the dominant class, whereas a Huber-regression baseline produces more predictions near the correct interval but fewer exact matches.   We introduce Utility-Oriented Factivity Inference (UO-FIE), a parameter-efficient system that combines exact-label supervision with graded utility. UO-FIE predicts a distribution over the nine classes and combines hard-label supervision, utility-based soft targets, scheduled class weights, and an ordinal loss. We evaluate expected-utility decoding in controlled comparisons and use ordinal calibration selected on out-of-fold predi
    
[^199]: 面向多丝平行板雪崩计数器的物理信息自监督学习：丝校准与相互作用位置联合重建

    Physics-Informed Self-Supervised Learning for Joint Wire Calibration and Interaction Position Reconstruction in Multi-Wire Parallel Plate Avalanche Counters

    [https://arxiv.org/abs/2609.28604](https://arxiv.org/abs/2609.28604)

    该论文提出了一种物理信息驱动的自监督学习框架，无需带标签的位置数据或专门校准实验，即可在多丝平行板雪崩计数器中联合实现丝增益校准与相互作用位置重建。

    

    科学仪器需要精确的校准，以便将探测器信号转换为可靠的物理观测量。传统的校准流程通常依赖于专门的校准测量、解析响应模型或带有标签的参考数据，这限制了其适应不断变化的运行条件和探测器老化的能力。我们提出了一种物理信息驱动的自监督学习框架，可在多丝平行板雪崩计数器（MWPPAC）中联合实现丝校准与相互作用位置重建，而无需带标签的位置测量或专门的校准运行。该方法将探测器校准表述为一个隐变量优化问题，利用仅源自探测器几何结构和电荷-能量一致性约束的监督信号，同时估计全局丝增益和逐事件的相互作用位置。一个与探测器无关的神经网络重建……（原文摘要在此处被截断）

    arXiv:2609.28604v1 Announce Type: new  Abstract: Scientific instruments require accurate calibration to convert detector signals into reliable physical observables. Conventional calibration procedures typically rely on dedicated calibration measurements, analytical response models or labelled reference data, limiting their ability to adapt to changing operating conditions and detector aging.   We present a physics-informed self-supervised learning framework that jointly performs wire calibration and interaction position reconstruction in Multi-Wire Parallel Plate Avalanche Counters (MWPPACs) without requiring labelled position measurements or dedicated calibration runs. The method formulates detector calibration as a latent optimization problem in which global wire gains and event-wise interaction positions are estimated simultaneously using supervision derived exclusively from detector geometry and charge-energy consistency constraints. A detector-independent neural network reconstruc
    
[^200]: 学习发现有趣的数学

    Learning to Discover Interesting Mathematics

    [https://arxiv.org/abs/2609.28603](https://arxiv.org/abs/2609.28603)

    该论文提出将定理的内在趣味性定义为证明长度与陈述长度之比，证明该指标与定理的下游效用强相关，并训练了一个能准确预测证明难度的27B模型，据此优化可生成更有趣的数学定理。

    

    arXiv:2609.28603v1 公告类型：cross 摘要：近年来，大型语言模型（LLM）解决高级数学问题的能力日益增强，其中包括许多悬置数十年的开放性难题。这为以前所未有的规模扩展数学知识打开了大门。然而，尽管LLM或许能够猜想并证明越来越多的定理，这些新的数学知识是否有趣或有用仍是一个悬而未决的问题。我们将定理的内在趣味性定义为其证明长度与陈述长度之比，并证明该指标与定理下游效用的外在度量高度相关。我们确定了“在给定一组前提条件下证明的难度”作为计算这些指标的有用基础要素，并训练了一个27B参数的模型，其预测证明难度的准确度超过了前沿的通用模型。针对我们的指标进行优化后，所得到的模型能够产出更有趣的定理……

    arXiv:2609.28603v1 Announce Type: cross  Abstract: Recently, Large Language Models (LLMs) have been increasingly able to solve advanced mathematical problems, including many that have been open for decades. This opens the door to expansion of mathematical knowledge at unprecedented scale. Yet, while LLMs may be able to conjecture and prove more and more theorems, it remains open whether this new mathematical knowledge is interesting or useful. We define intrinsic interestingness of a theorem as the ratio between the length of its proof and the length of its statement. We show that this correlates strongly with an extrinsic measure of the downstream utility of a theorem. We identify the difficulty of a proof conditioned on a set of premises as a useful primitive for computing these metrics, and train a 27B model that predicts proof difficulty more accurately than frontier general-purpose models. Optimizing for our metric creates a model capable of producing more interesting theorems, wh
    
[^201]: HClimRep-Ocean：基于非结构网格的全球海洋模拟器

    HClimRep-Ocean: A Global Ocean Emulator on an Unstructured Mesh

    [https://arxiv.org/abs/2609.28601](https://arxiv.org/abs/2609.28601)

    本文提出了 HClimRep-Ocean，一个直接在 FESOM2 原生非结构网格上运行的全球海洋机器学习模拟器，突破了以往数据驱动海洋模型依赖经纬度网格的局限，从而更好地刻画中尺度涡旋及复杂海岸线等海洋特有挑战。

    

    近年来，针对大气过程的机器学习（ML）模拟器发展迅速，变革了天气预报领域。尽管目前已出现早期的机器学习海洋预报模型，但其发展程度仍不及相应的大气模型。与大气不同，海洋的大部分动能蕴含在中尺度涡旋中，其特征空间尺度比类似的大气现象约小一个数量级。此外，复杂的海岸线、狭窄的海峡以及海冰覆盖的海域使得边界表示成为一个大气模型无需面对的核心挑战。因此，数值海洋模拟通常采用局部加密甚至完全非结构的网格。然而，迄今为止，数据驱动的海洋模型仍大多围绕经纬度网格构建。我们提出了 HClimRep-Ocean，一个直接运行在 FESOM2 原生非结构网格上的海洋模拟器。该模拟器……

    arXiv:2609.28601v1 Announce Type: cross  Abstract: Machine-learning (ML) emulators for atmospheric processes have advanced rapidly in recent years, transforming weather forecasting. Although early ML ocean forecasting models now exist, they remain less developed than their atmospheric counterparts. Unlike the atmosphere, much of the ocean's kinetic energy resides in mesoscale eddies whose characteristic spatial scales are approximately an order of magnitude smaller than those of comparable atmospheric features. Moreover, complex coastlines, narrow straits, and ice-covered seas make boundary representation a central challenge that atmospheric models do not face. Consequently, numerical ocean simulations commonly use locally refined or even completely unstructured meshes. However, their data-driven counterparts have so far been built around latitude-longitude grids. We present HClimRep-Ocean, an ocean emulator that operates directly on the native unstructured mesh of FESOM2. The emulator
    
[^202]: BRFID：迈向拜占庭鲁棒的联邦入侵检测

    BRFID: Toward Byzantine-Robust Federated Intrusion Detection

    [https://arxiv.org/abs/2609.28599](https://arxiv.org/abs/2609.28599)

    该论文发现联邦入侵检测系统中的标签翻转投毒攻击会导致攻击者自身检测准确率自我退化，这一“自我妥协”信号可作为可检测异常用于识别拜占庭客户端，从而在无需额外防御机制的情况下实现拜占庭鲁棒的联邦入侵检测。

    

    在一个三客户端联邦入侵检测系统（IDS）中，使用标签翻转模型投毒从单个拜占庭客户端翻转60%的训练标签，会使攻击者自身的联邦检测准确率自我退化，从99.96%（无投毒时）降至84.33%。其中，联邦全局集成模型在所有测试的投毒率下均保持稳定的准确率，且无需部署防御机制，也无需攻击者之间进行协调。在本文中，我们提出了实证结果，量化了标签翻转投毒攻击对在CICIDS2017数据集上训练的三客户端联邦IDS的影响，其中各客户端具有非独立同分布的攻击子类型。我们证明，攻击者对抗性自我妥协的信号代表了一种可检测的异常，在不存在目标数据泄露的情况下，可被利用于识别拜占庭客户端。我们注意到，聚合步骤使用的是联邦森林（树拼接）而非参数化的FedAvg；该结果……

    arXiv:2609.28599v1 Announce Type: cross  Abstract: Flipping 60\% of training labels from a single Byzantine client using label-flipping model poisoning self-degrades an attacker's own federated detection accuracy, $99.96\%$ (at no poisoning rate) to $84.33\%$ in a three-client federated IDS. Where the Federated global ensemble maintains stable accuracy across all tested poison rates, without a defense mechanism in place and without coordination between attackers. In this paper, we present empirical results quantifying the impact of label-flipping poisoning attacks on a three-client federated IDS trained on CICIDS2017 with non-IID attack subtype distributions across clients. We demonstrate that the signal of the adversarial self-compromise represents a detectable anomaly for exploitation for Byzantine client identification in the absence of target data exfiltration. We note that the aggregation step uses a Federated Forest (tree concatenation) rather than a parametric FedAvg; the result
    
[^203]: TAM-Chain：基于吸收马尔可夫链与香农熵不确定性量化的多尺度甲状腺细胞学分类方法，用于假阴性抑制与域偏移适应

    TAM-Chain: Multi-Scale Thyroid Cytology Classification via Absorbing Markov Chains and Shannon Entropy Uncertainty Quantification for False-Negative Suppression and Domain-Shift Adaptation

    [https://arxiv.org/abs/2609.28590](https://arxiv.org/abs/2609.28590)

    该论文提出TAM-Chain框架，将多放大倍数（10x/20x/40x）甲状腺细胞学特征提取建模为吸收马尔可夫随机过程，结合香农熵不确定性量化、最优停止准则与人机协同转诊机制来抑制假阴性和域偏移风险，在内部测试集上实现Macro F1达0.9741且假阴性率为0%。

    

    背景与问题：基于Bethesda系统的甲状腺细针穿刺活检（FNAB）细胞学检查在甲状腺癌早期检测中发挥着关键作用；然而，深度学习方法在临床域偏移情形下面临假阴性率高和过度自信的重大挑战。方法：本研究提出TAM-Chain，这是一个多尺度（10x、20x、40x）甲状腺细胞学分类框架，将吸收马尔可夫链理论与基于香农熵的不确定性量化相结合。该框架将多放大倍数的特征提取动态建模为吸收随机过程，从而实现最优停止准则和人机协同转诊机制，以严格抑制关键诊断错误。结果：在内部测试集（N = 235）上的广泛评估显示，该框架的Macro F1得分为0.9741，绝对假阴性率（FNR）为0.00%。在独立的外部（测试集上的评估结果……原文在此截断）

    arXiv:2609.28590v1 Announce Type: new  Abstract: Background & Problem: Thyroid Fine-Needle Aspiration Biopsy (FNAB) cytology based on the Bethesda System plays a pivotal role in early thyroid cancer detection; however, deep learning approaches face substantial challenges regarding high false-negative rates and overconfidence under clinical domain shift.   Methods: In this study, we propose TAM-Chain, a multi-scale (10x, 20x, 40x) thyroid cytology classification framework leveraging Absorbing Markov Chain theory combined with Shannon Entropy-based Uncertainty Quantification. The framework dynamically models multi-magnification feature extraction as an absorbing stochastic process, enabling optimal stopping criteria and a human-in-the-loop referral mechanism to strictly suppress critical diagnostic errors.   Results: Extensive evaluation on an internal test set (N = 235) demonstrates a Macro F1 score of 0.9741 with an absolute False-Negative Rate (FNR) of 0.00%. On an independent externa
    
[^204]: NumericJev：基于多路决策树的类Jev大语言模型数值解码

    NumericJev: Jev-like LLM Numerical Decoding with Multiway Decision Trees

    [https://arxiv.org/abs/2609.28587](https://arxiv.org/abs/2609.28587)

    提出了一种无需训练的数值解码算法NUMERICJEV，通过多路决策树递归细化数值范围，使任何具有类Jev结构化选择接口的大语言模型都能输出数值，其性能甚至超过从包含正确答案的候选列表中直接选择。

    

    大语言模型能够理解自然语言，但做出稳健的决策仍然具有挑战性。类Jev模型可以暴露结构化的选项，但这些接口无法直接以所要求的精度提供数值。我们提出了NUMERICJEV，这是一种无需训练的数值解码算法，能够使任何具有类Jev结构化选择接口的大语言模型输出数值。令人惊讶的是，在我们的算术基准测试中，它的表现比从包含正确答案的候选列表中直接选择高出2.93个百分点（图1）。我们的研究动机来自这样一个观察：数值范围选择本身就是一个类Jev大语言模型能够解决的决策问题。NUMERICJEV通过多路决策树递归地细化数值范围，同时在上下文中保留原始问题，且无需参数更新或访问隐藏状态。在100个值的网格上，十路树只需要两轮决策即可完成。

    arXiv:2609.28587v1 Announce Type: cross  Abstract: Large language models can interpret natural lan- guage, yet robust decisions remain challenging. Jev-like models expose structured choices, but these interfaces do not directly provide numeri- cal values at a requested precision. We propose NUMERICJEV, a training-free numerical decod- ing algorithm that enables numerical output from any LLM with a Jev-like structured-choice in- terface. Surprisingly, on our arithmetic bench- mark, it outperforms direct selection from a can- didate list containing the correct answer by 2.93 percentage points (Figure 1). Our motivation comes from the observation that numerical range selection is itself a decision problem that Jev- like LLMs can address. NUMERICJEV recur- sively refines a range through a multiway deci- sion tree while retaining the original question in context, without parameter updates or hidden- state access. On a 100-value grid, a ten-way tree requires only two decision rounds. Range- 
    
[^205]: SGA：时间序列基础模型中多步预测的不确定性量化

    SGA: Uncertainty Quantification for Multi-Step Forecasting in Time Series Foundation Models

    [https://arxiv.org/abs/2609.28582](https://arxiv.org/abs/2609.28582)

    本文提出SGA方法，通过有向无环图表征预测分支拓扑结构并度量其图复杂度，实现了对时序基础模型多步预测不确定性的有效量化，提升了预测结果的可信度。

    

    arXiv:2609.28582v1 公告类型：cross。摘要：近来时序基础模型的出现显著提升了多步预测的性能，使其能够在较长的未来时间范围内做出准确预测。然而，现有的时序基础模型往往存在显著的固有不确定性，这种不确定性通常表现为在每个时间步衍生出预测分支并向后续步骤扩散；不同的预测分支往往展现出不同的预测表现，从而削弱了时序基础模型预测结果的可信度。在本文中，我们提出了切片-图化-对齐（Slicing-Graphing-Alignment，SGA）方法来量化时序基础模型多步预测的不确定性。所提出的SGA首先利用有向无环图表征所有潜在预测分支的拓扑结构，使图复杂度能够约束多步预测的不确定性，然后通过融合拓扑信息与时序基础模型固有的随机特性来精确度量图复杂度……

    arXiv:2609.28582v1 Announce Type: cross  Abstract: The recent emergence of Time Series Foundation Models (TSFMs) has significantly advanced multi-step forecasting performance, enabling accurate predictions over extended future horizons. However, existing TSFMs often suffer from significantly inherent uncertainty, which typically manifests as derived forecast branches emerging at each time step and spreading to subsequent steps; different forecast branches often exhibit varying forecasting performance, thereby undermining the credibility of TSFM forecasts. In this paper, we propose the Slicing-Graphing-Alignment (SGA) method to quantify the uncertainty of multi-step TSFM forecasts. The proposed SGA first characterizes the topology of all potential forecast branches using a directed acyclic graph, such that the graph complexity bounds the uncertainty of multi-step forecasts, and then precisely measures the graph complexity by integrating both topological information and TSFM-inherent sto
    
[^206]: 可审计性不是单一属性：强化学习中的规则重叠、行为一致性与组合

    Auditability Is Not One Property: Rule Overlap, Behavioural Agreement, and Composition in Reinforcement Learning

    [https://arxiv.org/abs/2609.28581](https://arxiv.org/abs/2609.28581)

    该论文将强化学习策略的可审计性分解为六个可独立检验的谓词，并提出基于符号规则提取与哈希账本的审计协议，同时发现规则集重叠并不代表行为一致，揭示了离散行为规则描述层的严格局限。

    

    强化学习（RL）策略通常以不透明的神经检查点形式发布，而训练日志只能表明某次训练发生过，却无法解释策略究竟学到了什么。我们研究独立训练的策略能否通过可审计的离散行为规则来进行表示和组合。我们将可审计性定义为六个可分别测试的谓词：轨迹完整性、无损编码、规则覆盖、行为一致性、组合质量以及价值模型可靠性。我们的协议使用共享的冻结符号化器、被动规则提取、仅追加的哈希绑定账本、精确的环境重放，以及带有明确盲区回退机制的离线置信度排序仲裁。结果表明，这一描述层存在严格的局限：规则集的重叠并不意味着行为一致性——不同策略可能共享相同的符号规则，但在全新状态上却选择几乎等同于随机猜测的动作。因此，融合后的策略只能在……（原文在此处截断）

    arXiv:2609.28581v1 Announce Type: cross  Abstract: Reinforcement-learning (RL) policies are often distributed as opaque neural checkpoints, while training logs show that a run occurred without explaining what the policy learned. We study whether independently trained policies can be represented and composed through auditable discrete behavioral rules. We define auditability as six separately testable predicates: trace integrity, lossless coding, rule coverage, behavioral agreement, composition quality, and value-model reliability. Our protocol uses a shared frozen symbolizer, passive rule extraction, an append-only hash-bound ledger, exact environment replay, and offline confidence-ranked arbitration with an explicit blind-spot fallback.   The results place strict limits on this description layer. Rule-set overlap does not imply behavioral agreement: policies may share symbolic rules while choosing near-chance-matching actions on fresh states. The fused policy therefore selects among e
    
[^207]: 从智能电表数据中揭示住宅光伏-电动汽车协同采用：面向需求侧规划的负荷原型与检测

    Uncovering Residential PV-EV Co-Adoption from Smart-Meter Data: Load Archetypes and Detection for Demand-Side Planning

    [https://arxiv.org/abs/2609.28578](https://arxiv.org/abs/2609.28578)

    本文提出一种集成的两阶段分析工作流，通过DTW k-means聚类从智能电表数据中发现居民用电行为原型，并结合BiLSTM模型检测家庭光伏与电动汽车的协同采用，为需求侧管理和低压电网规划提供支持。

    

    电动汽车（EV）和屋顶光伏（PV）系统日益普及，正在重塑居民用电需求，并为需求侧管理（DSM）、电价设计和低压电网规划带来新的挑战。现有文献大多孤立地研究电动汽车充电或光伏发电，对家庭协同采用的行为动态理解不足。我们开发了一个集成的两部分工作流来分析高级量测体系（AMI）数据。发现部分应用基于动态时间规整（DTW）的k-means聚类和DTW重心平均方法，将每日的用电或上网功率曲线聚类为可解释的行为原型；预测部分则在21天时间窗口上训练双向长短期记忆（BiLSTM）模型，并与表格类基线模型进行基准比较，用于光伏/电动汽车活动检测。电动汽车活动标签是从类似充电的负荷特征中推断出来的，因为充电器…

    arXiv:2609.28578v1 Announce Type: new  Abstract: The increasing adoption of electric vehicles (EVs) and rooftop photovoltaic (PV) systems is reshaping residential electricity demand and creating new challenges for demand-side management (DSM), tariff design, and low-voltage network planning. Much of the existing literature examines EV charging or PV generation in isolation, leaving the behavioral dynamics of household co-adoption less understood. We develop an integrated, two-part workflow to analyze advanced metering infrastructure (AMI) data. A discovery component applies dynamic time warping (DTW) k-means with DTW barycenter averaging to cluster daily import or export profiles into interpretable behavioral archetypes, while a predictive component trains a bidirectional long short-term memory (BiLSTM) model on 21-day windows and benchmarks it against tabular baselines for PV/EV activity detection. The EV activity labels are inferred from charging-like load signatures because charger 
    
[^208]: 理解数据修订的时间序列基础模型

    Time-Series Foundation Models That Understand Data Revisions

    [https://arxiv.org/abs/2609.28576](https://arxiv.org/abs/2609.28576)

    该论文提出修订感知的时间序列基础模型 VINTAGE-TS，通过区分观测时间与信息可用时间、预测首次发布值并使用联合预测分布，解决了使用修订后数据评估历史预测时可能产生的前视偏差问题。

    

    历史观测数据并非总是固定不变的：统计机构会随着新证据的到来而修订先前发布的数值。因此，基于当下下载数据进行的预测，可能会使模型接触到在其所谓做出预测的日期时尚不可用的信息。我们提出了 VINTAGE-TS，这是一种对时间序列基础模型的修订感知适配方法，它区分了观测时间与信息可用时间。其预测目标是下一期的首次发布值以及该发布之后固定天数内可获得的数值；两者均不被声明为最终真实值。联合预测分布保留了这些预测目标之间的依赖关系，并揭示了两者差异的不确定性。我们规定了基于 ALFRED 的滚动评估、匹配的 Chronos-2 对比、常规基线与修订感知基线，以及一项针对预训练数据重叠情况的独立审计。随附软件实现了有效性区间重建、删……

    arXiv:2609.28576v1 Announce Type: new  Abstract: Historical observations are not always fixed: statistical agencies revise previously published values as new evidence arrives. Forecasting from a contemporary download can therefore expose a model to information unavailable at the date it purportedly made a prediction. We propose VINTAGE-TS, a revision-aware adaptation of a time-series foundation model that distinguishes observation time from information-availability time. Its targets are the next period's first-published value and the value available a fixed number of days after that publication; neither is declared final truth. A joint predictive distribution preserves dependence between these targets and exposes uncertainty about their difference. We specify an ALFRED-based rolling evaluation, a matched Chronos-2 comparison, conventional and revision-aware baselines, and a separate audit of pretraining overlap. The accompanying software implements validity-interval reconstruction, del
    
[^209]: DEEPO：面向多模态大语言模型幻觉问题的双熵增强策略优化

    DEEPO: Dual-Entropy Enhanced Policy Optimization for Hallucination in MLLMs

    [https://arxiv.org/abs/2609.28570](https://arxiv.org/abs/2609.28570)

    该论文提出DEEPO方法，针对强化学习纠正链中的两个薄弱环节——高语义熵困难查询导致组相对优势归零、以及“自信但错误”的token梯度不可见——通过结合信号方差正则化与梯度预处理的双阶段增强来抑制多模态大语言模型的幻觉问题。

    

    强化学习（RL）被广泛用于提升多模态大语言模型（MLLMs）的推理能力，但其对幻觉问题的抑制效果并不稳定。我们将此归因于从奖励到参数更新的“纠正链”中的两个薄弱环节。在采样层面，困难查询——即那些具有高语义熵的查询——经常产生全体一致的错误样本组，使得组相对优势恰好在幻觉风险最高的地方坍缩为零。在优化层面，“自信但错误”的 token 对梯度不可见：分类策略的期望得分梯度范数会随其分布变尖锐而趋于消失，因此最需要纠正的预测反而获得最弱的更新。我们提出双熵增强策略优化（DEEPO），一种结合信号方差正则化与梯度预处理的双阶段增强方法：语义熵触发的专家前缀在高熵困难查询上注入有依据的后续内容……

    arXiv:2609.28570v1 Announce Type: new  Abstract: Reinforcement learning (RL) is widely used to sharpen reasoning in multimodal large language models (MLLMs), yet its effect on hallucination is uneven. We trace this to two weak points in the \emph{correction chain} from reward to parameter update. At the rollout level, hard queries---those with high semantic entropy---frequently produce unanimously wrong sample groups, collapsing the group-relative   advantage to zero exactly where hallucination risk is highest. At the optimization level, confident-but-wrong tokens are gradient-invisible: a categorical policy's expected score-gradient norm vanishes as its distribution sharpens, so the predictions that most need correction receive the weakest updates. We propose Dual-Entropy Enhanced Policy Optimization (DEEPO), a dual-stage enhancement combining signal   variance regularization with gradient preconditioning: semantic-entropy-triggered expert prefixes inject grounded continuations on hig
    
[^210]: 面向316L不锈钢氢脆检测的防泄漏机器学习：基于区域留出协议的SEM显微图像纹理与深度特征评估

    Leakage-Safe Machine Learning for Hydrogen Embrittlement Detection in 316L Stainless Steel: A Region-Held-Out Evaluation of Texture and Deep Features in SEM Micrographs

    [https://arxiv.org/abs/2609.28567](https://arxiv.org/abs/2609.28567)

    该论文提出了一种基于留一区域法（LORO）交叉验证的区域留出评估协议，有效防止来自同一试样区域的SEM图像在训练集与测试集之间的信息泄漏，从而为316L不锈钢氢脆检测的机器学习模型提供了更可靠、更严谨的评估方式。

    

    扫描电子显微镜（SEM）通常被用于表征结构钢中氢脆（HE）引起的微观组织变化。机器学习可以自动化这一表征过程，但模型通常采用图像级别的数据划分进行评估。当多张图像来自同一试样区域时，这种划分方式会在训练集与测试集之间造成信息泄漏。本文提出了一种区域留出协议，用于对316L不锈钢的原始态（AR）和充氢态（H2）SEM显微图像进行分类，该协议基于对14个空间区域（8个AR区域、6个H2区域，共31张图像）实施的留一区域法（LORO）交叉验证。我们比较了六种特征-分类器组合，分别基于局部二值模式（LBP）、灰度共生矩阵（GLCM）、在143张无标注SEM图像上预训练的自监督卷积嵌入特征，以及卷积神经网络（CNN）。其中最简单的纹理方法——LBP结合支持向量机……

    arXiv:2609.28567v1 Announce Type: new  Abstract: Scanning electron microscopy (SEM) is routinely used to characterize the microstructural changes caused by hydrogen embrittlement (HE) in structural steels. Machine learning can automate this characterization, but models are often evaluated using image-level splits. When several images come from the same specimen region, such splits leak information between the training and test sets. Here, we propose a region-held-out protocol for classifying as-received (AR) and hydrogen-charged (H2) SEM micrographs of 316L stainless steel, based on Leave-One-Region-Out (LORO) cross-validation over 14 spatial regions (8 AR, 6 H2; 31 images). We compared six feature-classifier combinations built on local binary patterns (LBP), grey-level co-occurrence matrices (GLCM), self-supervised convolutional embeddings pretrained on 143 unlabeled SEM images, and a convolutional neural network (CNN). The simplest texture approach, LBP with a support vector machine 
    
[^211]: 当解释无法被阅读时：针对从右到左语言的SHAP和LIME渲染的测量与修正

    When Explanations Cannot Be Read: Measuring and Correcting SHAP and LIME Rendering for Right-to-Left Languages

    [https://arxiv.org/abs/2609.28565](https://arxiv.org/abs/2609.28565)

    本文提出SHAP-RTL渲染层，修正SHAP和LIME解释可视化在从右到左语言（如阿拉伯语、乌尔都语等）中的阅读方向错乱和字形断裂问题，同时保持原始归因值、特征排序和模型输出不变。

    

    诸如SHAP和LIME等事后解释方法被广泛用于解释文本分类器，但其可视化主要针对从左到右书写的语言设计。当应用于从右到左（RTL）书写的语言（如乌尔都语、阿拉伯语、波斯语和希伯来语）时，归因值在数学上仍然有效，但视觉呈现却会失效：词元顺序错乱、连体字形断裂、图表布局不符合自然阅读方向。本研究将这一差距视为一个可视化问题，而非解释方法本身的局限。我们提出了SHAP-RTL，一个用于修正SHAP和LIME可视化中阅读方向和文字整形问题的渲染层，并针对每种语言进行字体选择，同时保留原始的归因值、特征排序和模型输出。该方法在乌尔都语、阿拉伯语、希伯来语和波斯语的仇恨及冒犯性语言数据集上进行了评估。

    arXiv:2609.28565v1 Announce Type: cross  Abstract: Post hoc explanation methods such as SHAP and LIME are widely used to interpret text classifiers, but their visualizations are mainly designed for left-to-right languages. When applied to right-to-left (RTL) languages such as Urdu, Arabic, Persian, and Hebrew, the attribution values remain mathematically valid, while their visual presentation fails. Tokens appear out of sequence, connected letterforms break apart, and plot layouts do not follow the natural reading direction. This study addresses this gap as a visualization problem rather than a limitation of the explanation methods themselves. We present SHAP-RTL, a rendering layer that corrects reading direction and script shaping in SHAP and LIME visualizations, with per-language font selection, while preserving the original attribution values, feature ordering, and model outputs. The approach is evaluated on Urdu, Arabic, Hebrew, and Persian hate and offensive-language datasets usin
    
[^212]: 别读日志：执行轨迹会污染视频生成智能体中的验证器

    Don't Read the Log: Execution Traces Contaminate Verifiers in Video-Generation Agents

    [https://arxiv.org/abs/2609.28564](https://arxiv.org/abs/2609.28564)

    该论文揭示了智能体视频生成系统中的一个关键缺陷：向多模态评审器展示智能体的执行轨迹等辅助文本会严重污染其对纯视觉质量的判断——报告成功的轨迹可使评审器接受78–90%的失败视频，即使明确要求“仅使用视频帧”也无法消除这种偏差。

    

    智能体视频生成系统在生成器与验证器之间构建了一个闭环：一个大语言模型规划镜头、调用文生视频模型，再由多模态评审器判断结果是否满足用户请求。为了诊断漫长工作流中的失败环节，近期的框架有意向评审器展示超出视频本身的信息——包括智能体的执行轨迹、它的计划、以及它合成的旁白解说。我们提出一个问题：在保持视频帧不变的前提下，这些辅助文本是否会改变评审器对纯视觉要求的判定。在一个包含109个生成的双事件片段、并经人工标注的基准数据集上（其中所请求的事件要么明显完成，要么明显缺失），一条报告工具调用成功的轨迹使得三个开源权重的Qwen-VL评审器（7B、8B、32B）接受了78–90%的失败片段（而无文本时仅为7–19%）；相反，一条与之矛盾的轨迹会使它们拒绝多达100%的正确片段；而“仅使用视频帧”的指令并不能消除这一效应（原文摘要在此处被截断）。

    arXiv:2609.28564v1 Announce Type: cross  Abstract: Agentic video-generation systems close a loop between a generator and a verifier: an LLM plans shots, calls a text-to-video model, and a multimodal judge decides whether the result satisfies the request. To diagnose where a long workflow fails, recent harnesses deliberately show the judge more than the video-the agent's execution trace, its plan, the narration it synthesized. We ask whether this auxiliary text moves the judge's verdict on purely \emph{visual} requirements, holding the frames fixed. On a benchmark of 109 generated two-event clips with manual labels, in which the requested event is either visibly completed or visibly missing, a trace that reports a successful tool call makes three open-weight Qwen-VL judges (7B, 8B, 32B) accept $78$--$90\%$ of the failures, up from $7$--$19\%$ without text, and a contradicting trace makes them reject up to $100\%$ of correct clips; an instruction to ``use only the frames'' does not remov
    
[^213]: SpaFactor：用于组织学到转录组学推断的轻量级空间上下文感知基因程序建模

    SpaFactor: Lightweight Spatial Context-Aware Gene Program Modeling for Histology-to-Transcriptomics Inference

    [https://arxiv.org/abs/2609.28563](https://arxiv.org/abs/2609.28563)

    提出了SpaFactor，一个轻量高效的低秩形态学-程序-基因分解框架，通过融合中心点与多尺度邻域上下文的空间感知表示，从HE染色图像实现基因表达预测，避免了对高维噪声敏感和计算开销大的问题。

    

    空间转录组学（ST）能够在组织结构中分析基因表达，但其高昂成本和实验复杂性限制了常规应用。因此，从常规可获取的苏木精-伊红（HE）染色图像预测空间基因表达，提供了一种可扩展的替代方案。然而，传统方法通常将高维基因输出作为独立目标进行拟合，忽视了基因之间的生物学协同性，同时容易受到高维噪声和过拟合的影响。现有解决这一局限的尝试往往依赖计算量大的图网络或复杂的辅助监督。为此，我们提出了SpaFactor，一个轻量且高效的低秩形态学-程序-基因分解框架。在输入端，SpaFactor高效地将中心点的视觉表示与多尺度的局部和区域邻域上下文相融合，产生能够捕获细胞形态学的组织学表示。

    arXiv:2609.28563v1 Announce Type: new  Abstract: Spatial transcriptomics (ST) profiles gene expression within tissue architecture, but its cost and experimental complexity limit routine use. Predicting spatial expression from routinely available hematoxylin and eosin (HE) images therefore offers a scalable alternative. However, conventional methods often fit high-dimensional gene outputs as independent targets, overlooking the biological coordination among genes while remaining vulnerable to high-dimensional noise and overfitting. Existing attempts to address this limitation often rely on computationally heavy graph networks or complex auxiliary supervision. We therefore introduce SpaFactor, a lightweight and efficient low-rank morphology-program-gene factorization framework. At the input, SpaFactor efficiently fuses the visual representation of the central spot with multiscale local and regional neighborhood context, yielding a histologic representation that captures cellular morpholo
    
[^214]: 矩阵聚合算子

    Matrix Aggregation Operators

    [https://arxiv.org/abs/2609.28562](https://arxiv.org/abs/2609.28562)

    本文首次形式化了矩阵聚合算子（MAO）的概念，将聚合理论从向量拓展到矩阵结构，并分析了其可分解性与对称性，证明某些算子无法分解为逐行逐列聚合的形式。

    

    聚合理论传统上主要关注定义在向量上的算子。然而，许多应用——包括多准则决策、群体决策、基于模糊规则的分类系统以及重叠/分组指数等——需要聚合的信息天然地以隶属度矩阵的形式结构化（例如，一个对象集合与一族模糊集交互的情形）。尽管如此，针对这类算子尚未提出正式的框架，部分原因在于通常将矩阵展平为向量的做法（这会丢弃结构信息），部分原因在于依赖于按顺序聚合行和列的可分解算子。本文通过形式化矩阵聚合算子的概念来填补这一空白。我们分析了矩阵聚合算子的可分解性与对称性，证明了某些算子无法用可分解形式表达，并考察了若干……（摘要在此处截断）

    arXiv:2609.28562v1 Announce Type: cross  Abstract: Aggregation theory has traditionally focused on operators defined over vectors. However, many applications-including Multi-Criteria Decision Making, Group Decision Making, Fuzzy Rule-Based Classification Systems, and overlap/grouping indices-require aggregating information naturally structured as a matrix of membership degrees (e.g., where a set of objects interacts with a family of fuzzy sets). Despite this, no formal framework has been proposed for this class of operators, partly due to the common practice of flattening matrices into vectors (which discards structural information) and partly due to a reliance on decomposable operators that aggregate rows and columns sequentially. This paper addresses this gap by formalizing the notion of a matrix aggregation operator (MAO). We analyze the decomposability and symmetry properties of MAOs, showing that certain operators cannot be expressed in decomposable form and examining several noti
    
[^215]: CARE：面向扩散模型的条件感知表示正则化

    CARE: Condition-Aware Representation Regularization for Diffusion Models

    [https://arxiv.org/abs/2609.28561](https://arxiv.org/abs/2609.28561)

    提出了一种轻量级即插即用的条件感知表示正则化框架CARE，利用内置条件信号动态调节特征分布，无需显式对齐损失或外部监督即可提升扩散模型的视觉保真度和收敛稳定性。

    

    扩散模型的最新研究进展凸显了表示正则化对于提升样本质量和训练效率的重要性。然而，常用的正则化方法往往忽视了直接决定生成目标的内置条件（如标签或文本）。在本工作中，我们展示了条件信号如何影响特征分布，并提出了CARE（条件感知表示正则化）。CARE是一个轻量级的即插即用正则化框架，能够根据条件相似度动态调节特征分布。CARE利用内置的条件信号来审慎地引导表示空间，促使相似条件对应的特征形成更紧凑的聚类，而无需依赖显式的对齐损失或外部监督。实验结果表明，CARE在类条件生成图像和文本生成图像任务上均能持续提升视觉保真度与收敛稳定性。

    arXiv:2609.28561v1 Announce Type: new  Abstract: Recent advances in diffusion models highlight the importance of representation regularization for improving sample quality and training efficiency. However, commonly used regularization methods often overlook the built-in conditions (such as labels or texts) which directly determine the generation target. In this work, we demonstrate how conditioning signals affect the feature distribution and introduce the CARE (Condition-Aware REpresentation regularization). CARE is a lightweight plug-and-play regularization framework that dynamically modulates feature distribution based on condition similarity. CARE leverages built-in conditioning signals to judiciously guide the representation space, promoting tighter feature clusters for similar conditions without relying on explicit alignment losses or external supervision. Empirically, CARE consistently improves both visual fidelity and convergence stability across both class-to-image and text-to-
    
[^216]: 随机大语言模型的推测式评估

    Speculative Evaluation of Stochastic LLMs

    [https://arxiv.org/abs/2609.28560](https://arxiv.org/abs/2609.28560)

    该论文提出了一种基于分层贝叶斯尼曼策略的推测式评估方法（HBN及异步版本HBN-async），通过分层贝叶斯模型估计各任务方差并自适应地分配推演预算，从而在固定预算下最小化随机大语言模型基准评估的方差。

    

    评估一个随机的大语言模型代价高昂：基准测试分数通过随机化的多次推演（rollouts）来估计期望性能，然而均匀的重复采样忽略了任务级推演方差之间的显著差异。我们研究如何在固定的推演预算下最小化基准测试均值估计的方差。我们提出了基于分层贝叶斯尼曼（HBN）策略的推测式评估方法，其中试点规模和阶段权重在事前联合确定。该方法首先运行一个简短的均匀试点，将各任务的成功计数与分层贝叶斯模型相结合，并使用任务级抽样方差的后验期望进行精确的正整数尼曼分配。为了缓解试点同步障碍，HBN-async 根据部分试点反馈推测性地执行后续推演，并保留最终分配所选中的那些推演结果。在六个检查点和18个基准测试组上，我们评估了107个非退化的基准-检查点组合。对于推演预算……

    arXiv:2609.28560v1 Announce Type: cross  Abstract: Evaluating a stochastic large language model is costly: benchmark scores estimate expected performance from randomized rollouts, yet uniform repetition ignores sharp differences in task-level rollout variance. We ask how to minimize the variance of a fixed-benchmark mean under an exact rollout budget. We develop Speculative Evaluation with a Hierarchical Bayesian Neyman (HBN) policy with pilot size and stage weight jointly chosen ex ante. It runs a short uniform pilot, pools per-task success counts with a hierarchical Bayesian model, and uses posterior expectations of task-level sampling variances for exact positive-integer Neyman allocation. To mitigate the pilot synchronization barrier, HBN-async speculatively executes continuations from partial pilot feedback and retains those selected by the final allocation. Across six checkpoints and 18 benchmark groups, we evaluate 107 nondegenerate benchmark-checkpoint profiles. For rollout bud
    
[^217]: 基于变分自编码器潜在空间自适应的压气机叶栅开放叶顶间隙流CFD修正

    CFD Correction of Open Tip Clearance Flow in a Compressor Cascade Using VAE Latent Space Adaptation

    [https://arxiv.org/abs/2609.28558](https://arxiv.org/abs/2609.28558)

    提出了一种基于变分自编码器潜在空间自适应的无侵入式CFD修正方法，仅用12组配对的CFD-实验数据即可有效修正压气机叶栅叶顶间隙流预测与实验之间的偏差。

    

    压气机叶栅中开放叶顶间隙流的CFD预测与实验结果相比存在偏差，而实验观测数据稀疏且缺乏高分辨率的实验基准真值。本研究提出了一种基于变分自编码器（VAE）和潜在空间自适应的无侵入式修正方法。首先利用包含166个参数化采样的CFD总压损失场数据集训练VAE，以学习这些场的低维统计表示。随后冻结VAE，仅使用12组配对的CFD-实验工况训练低秩潜在空间适配器。通过观测算子将修正后的高分辨率场映射到实验观测空间，使监督信号仅在可用的测量位置和测量的节距窗口内施加。在当前的12折交叉验证中，平均绝对误差有所降低。

    arXiv:2609.28558v1 Announce Type: new  Abstract: CFD predictions of open tip clearance flow in compressor cascades are subject to discrepancies relative to experiments, while experimental observations are sparse and high-resolution experimental ground truth is unavailable. This study proposes a non-intrusive correction method based on a variational autoencoder (VAE) and latent-space adaptation. A VAE is first trained using a dataset of 166 parametrically sampled CFD total pressure loss fields to learn a low-dimensional statistical representation of these fields. The VAE is then frozen, and a low-rank latent-space adapter is trained using only 12 paired CFD--experiment operating conditions. An observation operator maps the corrected high-resolution fields to the experimental observation space, allowing supervision to be applied only at the available measurement locations and within the measured pitchwise windows. In the current 12-fold cross-validation, the mean absolute error decreases
    
[^218]: SMILESGNN：基于SMILES-图交叉注意力融合的可解释临床毒性预测

    SMILESGNN: Interpretable Clinical Toxicity Prediction via SMILES-Graph Cross-Attention Fusion

    [https://arxiv.org/abs/2609.28553](https://arxiv.org/abs/2609.28553)

    该论文提出SMILESGNN多模态架构，通过交叉注意力融合SMILES Transformer与GATv2图编码器，在仅0.4M参数的情况下于ClinTox数据集上取得AUC-ROC 0.987的竞争性性能，同时保留显式图分支以支持基于GNNExplainer的可解释毒性预测。

    

    药物毒性预测对于降低药物研发后期的损耗率至关重要，但由于严重的类别不平衡、基于骨架的泛化问题以及临床对可解释预测的需求，该任务仍然充满挑战。单模态方法——SMILES Transformer或图神经网络——各自捕捉分子结构的互补方面，而仅使用序列的模型无法直接提供基于图的归因解释。我们提出了SMILESGNN，这是一种通过交叉注意力融合SMILES Transformer编码器和GATv2图编码器的多模态架构，并提出了其变体SMILESGNN-PT，该变体使用ChemBERTa-2预训练主干网络。该设计在预测流程中保留了显式的图分支，支持基于GNNExplainer对与毒性预测相关子结构的分析。在ClinTox数据集上，SMILESGNN仅以0.4M参数实现了AUC-ROC 0.987和F1 0.906的性能，与强大的SMILES Transformer相比具有竞争力。

    arXiv:2609.28553v1 Announce Type: cross  Abstract: Drug toxicity prediction is critical for reducing late-stage attrition in drug discovery, yet remains challenging due to severe class imbalance, scaffold-based generalization, and the clinical need for interpretable predictions. Single-modality approaches-SMILES Transformers or graph neural networks capture complementary aspects of molecular structure, while sequence-only models cannot directly provide graph-attributed explanations. We present SMILESGNN, a multimodal architecture that fuses a SMILES Transformer encoder and a GATv2 graph encoder via cross-attention, and SMILESGNN-PT, a variant using a ChemBERTa-2 pretrained backbone. The design retains an explicit graph branch within the predictive pipeline, supporting GNNExplainer-based analysis of substructures associated with toxic predictions. On ClinTox, SMILESGNN achieves AUC-ROC 0.987 and F1 0.906 with only 0.4M parameters, performing competitively with a strong SMILESTransformer
    
[^219]: 一致归纳推理的序理论刻画

    An Order-Theoretic Characterization of Consistent Inductive Inference

    [https://arxiv.org/abs/2609.28551](https://arxiv.org/abs/2609.28551)

    该论文在ZFC框架下通过有限可实现轨迹上的一个线性序（要求冲突轨迹选择不同子轨迹、且对每个固定目标良基）完整刻画了一致归纳推理，证明一致性等价于此类序的存在性，并回答了Lu（2024）的问题。

    

    在什么条件下，学习者能够沿着由某个固定但未知假设所标注的每条无限序列，仅做出有限多次预测错误？我们在ZFC公理体系下对任意二值假设类刻画了这种一致性形式，且不要求一致错误界。该刻画基于有限可实现轨迹上的单个线性序：每条轨迹选择其最小子轨迹，且该序必须满足两个条件——相互冲突的轨迹选择不同的子轨迹，以及该序在每个固定目标的轨迹集上是良基的。这些条件诱导出一个学习者，其每次犯错时所选证据都会严格递减。反之，一个一致的学习者可通过典范错误记录和Kleene–Brouwer排序构造出这样一个序。该结果提供了用有限证据来表示一致预测的方法，回答了Lu（2024）提出的一个问题。

    arXiv:2609.28551v1 Announce Type: cross  Abstract: When can a learner make only finitely many prediction errors along every infinite sequence labeled by a fixed, unknown hypothesis? We characterize this form of consistency for arbitrary binary hypothesis classes in ZFC, without requiring a uniform mistake bound. The characterization uses a single linear order on finite realizable traces. Each trace selects its least subtrace, and the order must satisfy two conditions: conflicting traces select different subtraces, and the order is well-founded on the traces of each fixed target. These conditions induce a learner whose selected evidence decreases on every mistake. Conversely, a consistent learner yields such an order through canonical mistake transcripts and the Kleene--Brouwer ordering. The result provides a representation of consistent prediction by finite evidence, answering a question of Lu (2024).
    
[^220]: 随机惯性 Krasnosel'skii-Mann 迭代实现近乎最优的样本复杂度

    Stochastic Inertial Krasnosel'skii-Mann Iteration Achieves Near-Optimal Sample Complexity

    [https://arxiv.org/abs/2609.28543](https://arxiv.org/abs/2609.28543)

    本文提出一种随机惯性 Krasnosel'skii-Mann（iKM）迭代方法，仅在随机 KM 算法上添加两个惯性外推、且每次更新只需一次可能有偏随机预言机调用的条件下，实现了 Õ(ε⁻²) 的近乎最优样本复杂度。

    

    我们分析了一种简单的随机惯性 Krasnosel'skii-Mann（iKM）方法，用于在实希尔伯特空间中求非扩张算子的不动点。该方法只需在随机 KM 算法 [Bravo and Cominetti, 2024] 的基础上添加两个惯性外推即可得到，并且每次更新仍只需调用一次可能有偏的随机预言机，同时在随机和确定性两种情形下都达到了尖锐的速率。具体而言，利用我们提出的参数调度方案，我们证明了如下的最后迭代不动点残差界：O(1/K + σ log K/√K + B_K log K/K)，其中 K 为迭代时域长度，σ 为噪声水平，B_K 为累积的均方根偏差。当 B_K=O(√K) 时，该界给出 Õ(ε⁻²) 的样本复杂度，在对数因子范围内匹配了在我们模型的无偏子类下给出的随机预言机下界 [Foster et al., ...]。

    arXiv:2609.28543v1 Announce Type: cross  Abstract: We analyze a simple stochastic inertial Krasnosel'skii--Mann (iKM) method for finding a fixed point of a nonexpansive operator in a real Hilbert space. Our method is obtained simply by adding two inertial extrapolations to stochastic KM [Bravo and Cominetti, 2024], and it retains one call to a possibly biased stochastic oracle per update and achieves sharp rates in both the stochastic and deterministic regimes. Specifically, with our proposed parameter schedule, we prove the following last-iterate fixed-point residual bound: \[   {O}\!\left(\frac{1}{K} +\frac{\sigma\log K}{\sqrt K} +\frac{B_K\log K}{K}\right), \] where $K$ is the horizon, $\sigma$ is the noise level and $B_K$ is the accumulated root-mean-square bias. When $B_K=O(\sqrt K)$, this yields $\widetilde O(\epsilon^{-2})$ sample complexity that matches, up to a logarithmic factor, the stochastic-oracle lower bound given under the unbiased subclass of our model [Foster et al., 
    
[^221]: 关于 GPT Astra 差分隐私持续计数下界证明的详细阐述

    An Exposition of GPT Astra's Proof of Lower Bound on DP Continual Counting

    [https://arxiv.org/abs/2609.28528](https://arxiv.org/abs/2609.28528)

    本文详细阐述了 GPT Astra 关于差分隐私持续计数下界的证明，并希望有助于寻找更自然、更简洁的证明方法。

    

    本文的目标是，尽我们所理解，对 Harrison 和 Leeman（arXiv:2609.17650v01 和 arXiv:2609.17650v02）最近展示的、由 Astra 给出的差分隐私持续计数下界证明进行详细阐述。我们相信存在更自然、更简洁的证明方法，并希望本文能有助于实现这一目标。在 Harrison 和 Leeman 的初始预印本（arXiv:2609.17650v01）之前，Bairaktari 和 Larsen（arXiv:2607.00876）曾给出一个优雅的证明，表明纯差分隐私和近似差分隐私持续计数的下界均为 $\Omega(\log^{3/2}(n))$，并且他们在私人交流中告知我们，他们还证明了纯差分隐私持续计数的最优下界 $\Omega(\log^{2}(n))$。他们随后发表了该 $\Omega(\log^{2}(n))$ 下界，现已成为 Bairaktari、Dahl 和 Larsen 的联合工作（arXiv:2607.00876v3）。他们的新结果是一个 ele（原文在此处截断）……

    arXiv:2609.28528v1 Announce Type: cross  Abstract: The goal of this note is to give a detailed proof, to the best of our understanding, of the recent presentation by Harrison and Leeman (arXiv:2609.17650v01 and arXiv:2609.17650v02) of the proof by Astra on the lower bound for differentially private continual counting. We believe a more natural and easy proof is possible and hope that this note will help in that effort.   Prior to the initial preprint by Harrison and Leeman (arXiv:2609.17650v01), Bairaktari and Larsen (arXiv:2607.00876) gave an elegant proof to show a lower bound of $\Omega(\log^{3/2}(n))$ for both pure and approximate-DP continual counting, and in personal communication had informed us that they have a proof of optimal $\Omega(\log^{2}(n))$ for pure-differential private continual counting as well. They have subsequently published their $\Omega(\log^{2}(n))$ bound, which is now a joint work of Bairaktari, Dahl, and Larsen (arXiv:2607.00876v3). Their new result is an ele
    
[^222]: 覆盖率约束保形模型选择的序贯置信集

    Sequential Confidence Sets for Coverage-Constrained Conformal Model Selection

    [https://arxiv.org/abs/2609.28522](https://arxiv.org/abs/2609.28522)

    提出了覆盖率约束序贯模型置信集（CC-SMCS），利用同步鞅置信序列与精确闭式规则，以至少1-δ的概率识别出满足覆盖率硬约束且成本最小的保形预测流水线。

    

    现代保形预测系统通常维护多个自适应流水线，它们在基础预测器、一致性分数、校准窗口和更新规则上各不相同。比较这些流水线十分困难，因为覆盖率是一个硬性约束，而效率只应在可行的流水线之间进行优化。我们将该问题形式化为针对随机约束 argmin 的序贯推断问题。在每个时刻，目标集合是满足多个前缀平均条件失覆盖率约束的最小成本流水线集合。我们提出了覆盖率约束序贯模型置信集，它将流水线区分为可证明可行、可能可行和可能约束最优三类。利用同步鞅置信序列，CC-SMCS 将矩形置信区域投影到约束 argmin 上，并给出精确的闭式规则。以至少 $1-\delta$ 的概率，它包含所有约束最优的流水线。

    arXiv:2609.28522v1 Announce Type: cross  Abstract: Modern conformal forecasting systems often maintain several adaptive pipelines that differ in base forecasters, conformity scores, calibration windows, and update rules. Comparing them is difficult because coverage is a hard constraint, whereas efficiency should be optimized only among feasible pipelines. We formulate this problem as sequential inference for a stochastic constrained argmin. At each time, the target is the set of minimum-cost pipelines satisfying multiple prefix-average conditional miscoverage constraints. We introduce Coverage-Constrained Sequential Model Confidence Sets (CC-SMCS), which separate certifiably feasible, possibly feasible, and possibly constrained-optimal pipelines. Using simultaneous martingale confidence sequences, CC-SMCS projects a rectangular confidence region onto the constrained argmin and admits an exact closed-form rule. With probability at least $1-\delta$, it contains every constrained-optimal 
    
[^223]: 认证的任务条件化主动可观测性

    Certified Task-Conditioned Active Observability

    [https://arxiv.org/abs/2609.28520](https://arxiv.org/abs/2609.28520)

    该论文形式化了“认证的任务条件化主动可观测性复杂度”，即在认证误差与安全弃权保证下识别任务相关状态所需的最小最坏情况期望交互代价，并证明任务预测等价性诱导出唯一的最小充分商空间，使主动可观测性复杂度在其上严格不变。

    

    在对不可观测的物理系统采取行动之前，自主智能体必须确定哪些潜在区分支配下游任务、需要多少次主动干预才能认证这些区分、以及何时应当放弃行动以防止灾难性错误。经典可观测性将状态重构视为一个无条件的二元谓词，当被动观测无法在不施加扰动的情况下打破潜在简并、完全的微观反演代价过高、以及区分与任务无关的自由度浪费交互预算时，该方法便会失效。我们形式化了任务条件化主动可观测性复杂度：即在认证误差与安全弃权保证下，识别任务相关状态所需的最小最坏情况期望交互代价。我们证明任务预测等价性诱导出唯一的最小充分商 $\mathcal{H}/\!\sim_\tau$，使得主动可观测性复杂度在该商上严格保持不变……（摘要原文在此处截断）

    arXiv:2609.28520v1 Announce Type: cross  Abstract: Before acting upon an unobservable physical system, an autonomous agent must determine which latent distinctions govern downstream tasks, how many active interventions are necessary to certify them, and when to abstain to prevent catastrophic errors. Classical observability treats state reconstruction as an unconditioned binary predicate, failing when passive observations cannot break latent degeneracies without perturbation, full microscopic inversion is prohibitively costly, and distinguishing task-irrelevant degrees of freedom wastes interaction budgets. We formalize task-conditioned active observability complexity: the minimum worst-case expected interaction cost required to identify task-relevant states under certified error and safe abstention guarantees. We prove that task-predictive equivalence induces the unique minimal sufficient quotient $\mathcal{H}/\!\sim_\tau$, leaving active observability complexity strictly invariant wh
    
[^224]: 知识追踪的稳定且忠实的解释

    Stable and Faithful Explanations for Knowledge Tracing

    [https://arxiv.org/abs/2609.28502](https://arxiv.org/abs/2609.28502)

    本研究提出一套同时评估预测竞争力、解释稳定性与忠实性的知识追踪解释验证协议，并发现重建ASSISTments 2009数据集以修复多技能交互的标签泄漏问题后会改变模型AUC并重新排列解释结果。

    

    知识追踪（KT）模型对学生表现的预测是不透明的，这限制了其在教学行动中的应用。本研究贡献了一套验证协议，共同测试预测竞争力（RQ1）、解释稳定性（RQ2）以及基于重训练的忠实性（RQ3）。研究从ASSISTments 2009和2012数据集中构建了涵盖五个教学主题的十三个行为特征，其中历史特征由时间上先前的交互计算得出，当前响应延迟仅保留用于回顾性分析。研究重建了ASSISTments 2009数据集：未经修正的skill-builder版本将每个多技能交互按每个技能一行的方式重复记录，由于这些行共享同一个正确性标签，导致标签泄漏到先前交互特征中。数据重建降低了模型AUC并重新排列了解释结果。研究将使用Tree SHapley Additive exPlanations（TreeSHAP）解释的极端梯度提升模型与四个深度学习模型进行比较（摘要此处不完整）。

    arXiv:2609.28502v1 Announce Type: new  Abstract: Knowledge tracing (KT) models predict student performance opaquely, limiting pedagogical action. This study contributes a validation protocol testing predictive competitiveness (RQ1), explanation stability (RQ2) and retraining-based faithfulness (RQ3) together. Thirteen behavioral features across five pedagogical themes were engineered from ASSISTments 2009 and 2012, with history features computed from temporally preceding interactions and current response latency retained only for retrospective analysis. ASSISTments 2009 was rebuilt: the uncorrected skill-builder release duplicates each multi-skill interaction across one row per skill, and because those rows share one correctness label, they leak it into preceding-interaction features. Rebuilding lowered model AUC and reordered the explanation results. An Extreme Gradient Boosting (XGBoost) model explained with Tree SHapley Additive exPlanations (TreeSHAP) was compared against four deep
    
[^225]: 浅层多项式神经网络表达能力的代数证书

    Algebraic Expressivity Certificates for Shallow Polynomial Neural Networks

    [https://arxiv.org/abs/2609.28500](https://arxiv.org/abs/2609.28500)

    本文借助代数几何中的 Veronese 割簇与理想消元方法，构建了一个将网络架构自动转化为不可表示性多项式证书的通用流水线，并据此推导出球面上秩二二次网络的精确总体损失下界，揭示代数障碍会造成不可消除的近似误差。

    

    我们利用代数几何研究无偏置浅层多项式神经网络的精确可表示性。在复数域 $\mathbb{C}$ 上，宽度为 $r$、激活函数为 $z\mapsto z^d$ 的网络计算的是 $r$ 个线性形式的 $d$ 次幂之和，其 Zariski 闭包是一个 Veronese 割簇。因此，理想消元可以给出不可表示性的多项式证书。我们将这一构造实现为一个通用的“架构到证书”流水线。对于二次网络，我们恢复了精确的对称行列式描述，并通过正交对称性解释了其维数。在更高次情形下，该实现恢复了经典的 catalecticant 方程与割方程，并通过有限的架构扫描描绘了直接消元方法的实际适用范围。我们还推导了球面上秩二二次网络的精确总体损失下界，以此说明代数障碍如何导致不可消除的近似误差。

    arXiv:2609.28500v1 Announce Type: cross  Abstract: We study exact representability by bias-free shallow polynomial neural networks using algebraic geometry. Over $\mathbb{C}$, a width-$r$ network with activation $z\mapsto z^d$ computes a sum of $r$ $d$-th powers of linear forms, whose Zariski closure is a Veronese secant variety. Ideal elimination therefore yields polynomial certificates of nonrepresentability. We implement this construction as a generic architecture-to-certificate pipeline. For quadratics, we recover the exact symmetric determinantal description and explain its dimension through orthogonal symmetry. In higher degree, the implementation recovers classical catalecticant and secant equations and maps the practical reach of direct elimination across a finite architecture sweep. We also derive the exact population loss floor for a rank-two quadratic network on the sphere, illustrating how an algebraic obstruction induces irreducible approximation error.
    
[^226]: 具有自适应加权与效率评估的混合变分量子-经典框架

    Hybrid Variational Quantum-Classical Framework with Adaptive Weighting and Efficiency Assessment

    [https://arxiv.org/abs/2609.28491](https://arxiv.org/abs/2609.28491)

    该论文提出了Sim-HVQC混合深度量子神经网络框架，通过将无参数的SimAM自适应加权模块与经典特征提取相结合来保留类别判别信息，首次将变分量子电路扩展应用于多类别分类任务，并通过多种子评估证明了其可复现性、参数效率和可解释性。

    

    混合量子-经典神经网络已成为一种在机器学习中利用量子计算优势、同时缓解当前硬件限制的有前景的方法。本文提出了Sim-HVQC，这是一种混合深度量子神经网络，它将自适应、无参数的SimAM加权模块与经典特征提取相结合，在编码到变分量子电路（VQC）之前保留类别判别信息。以往的研究仅限于二分类任务。相比之下，所提出的框架在多个多类别数据集（MNIST、KMNIST、Fashion-MNIST和EMNIST）上进行了训练和评估。该框架通过多种子评估、参数分析以及潜在特征/量子特征的检查，进一步展示了其可复现性、参数效率和可解释性。源代码已在 https://github.com/Dilli822/SimAM-HVQC 公开提供。

    arXiv:2609.28491v1 Announce Type: cross  Abstract: Hybrid quantum-classical neural networks have emerged as a promising approach for leveraging quantum computing in machine learning while mitigating current hardware limitations. This paper presents Sim-HVQC, a hybrid Deep Quantum Neural Network that couples an adaptive, parameter-free SimAM weighting module with classical feature extraction to preserve class-discriminative information prior to encoding into a Variational Quantum Circuit (VQC). Previous studies are restricted to binary classification [1] [2] [3] [4] [5]. In contrast, the proposed framework is trained and evaluated on various multi-class datasets(MNIST, KMNIST, Fashion-MNIST, and EMNIST). The framework further demonstrates reproducibility, parameter efficiency, and interpretability through multi-seed evaluation, parameter analysis, and latent/quantum feature inspection. The source code is publicly available at https://github.com/Dilli822/ SimAM-HVQC
    
[^227]: RADAR：AI发现能力与智能体触达就绪度

    RADAR: Readiness for AI Discovery and Agentic Reach

    [https://arxiv.org/abs/2609.28480](https://arxiv.org/abs/2609.28480)

    该论文提出RADAR评估框架，对166个国家的测量发现，AI描述公共服务的能力（信息可读性）远超其代为办理服务的能力（智能体可操作性），且这一差距并不随国家财富增加而缩小。

    

    政府正越来越多地通过AI系统而非网站来接触公民。RADAR（AI发现能力与智能体触达就绪度）在166个国家测量这一系统是否真正有效，涵盖两项任务：一是聊天机器人能否就某项公共服务给出正确、来源官方且针对特定国家的答案（信息可读性），二是自动化智能体能否触达该服务并对其执行操作（智能体可操作性）。核心发现是，AI描述公共服务的能力远强于其触达这些服务的能力。在全部166个国家中，按RADAR的相应指标衡量，信息可读性得分均高于平均智能体可操作性得分，且这一差距并不随国家财富增长而缩小。收入和语言只能解释部分现象，若干政府的表现远好于或远差于其资源水平所预测的结果。这两类失败具有不同的相关因素，也需要不同的解决方案。AI能否描述某项服务……

    arXiv:2609.28480v1 Announce Type: cross  Abstract: Governments increasingly meet citizens through an AI system rather than a website. RADAR (Readiness for AI Discovery and Agentic Reach) measures whether that system works, across 166 countries and on two tasks: whether a chatbot can give a correct, officially sourced, country-specific answer about a public service (informational legibility), and whether an automated agent can reach the service to act on it (agent operability). The central finding is that AI can describe public services far better than it can reach them. In every one of the 166 countries, informational legibility scores exceed average agent operability scores under RADAR's respective measures, and the gap does not shrink with national wealth. Income and language explain only part of the pattern, and several governments perform far better or worse than their resources predict. The two failures have different correlates and different fixes. Whether AI can describe a servi
    
[^228]: 从预测到可解释的医疗服务提供者行为画像：面向欺诈、浪费与滥用审查

    From Prediction to Explainable Provider Behavior Profiles for Fraud, Waste, and Abuse Review

    [https://arxiv.org/abs/2609.28477](https://arxiv.org/abs/2609.28477)

    该研究提出将欺诈、浪费与滥用（FWA）审查从预测建模转向可解释的提供者行为画像，通过将账单收入分解为提供者规模与诊疗项目构成的乘积，从而解释提供者行为变化的原因。

    

    理赔数据可以显示医疗服务提供者的行为发生了变化，但仅凭数据本身无法解释原因。欺诈、浪费与滥用（FWA）审查需要识别重要的行为、定位驱动这些行为的诊疗代码和资金，并检验合理的解释。一种常见的替代方法——预测建模——通过标记偏离预期使用量预测的异常来发现问题，但预测的价值有限，除非它能够超越简单的持续性预测，并解释偏差为何重要。在我们的季度提供者-诊疗项目数据中，最新观测值已捕获了大部分可预测的变化，而增加模型结构几乎无法提升准确性。残差将增长、服务线变化、代码维护以及不完整的观测与潜在的可疑行为混为一谈，使得单点预测并不完整。因此，我们将提供者审查重新表述为一个描述性表示问题：账单收入 y = s × p，其中 s 衡量提供者规模，p 描述诊疗项目构成。

    arXiv:2609.28477v1 Announce Type: cross  Abstract: Claims data can show that provider behavior changed but cannot by itself explain why. FWA (fraud, waste, and abuse) review requires identifying material behavior, locating the codes and dollars driving it, and testing plausible explanations. A common alternative, predictive modeling, flags deviations from an expected-utilization forecast -- but a forecast has limited value unless it beats simple persistence and explains why a deviation matters. In our quarterly provider-procedure data, the latest observation captures most forecastable variation, and added model structure adds little accuracy. Residuals conflate growth, service-line shifts, code maintenance, and incomplete observation with potentially concerning behavior, making point forecasts incomplete.   We instead formulate provider review as a descriptive representation problem: billed revenue y = s * p, where s measures provider scale and p describes procedure composition. The pr
    
[^229]: 学习可靠推理的成本

    Learning the Cost of Reliable Inference

    [https://arxiv.org/abs/2609.28322](https://arxiv.org/abs/2609.28322)

    该论文设计了一个基于反向第二价格拍卖的大模型采购平台，通过提供商竞争驱动token定价，并在学习各提供商质量的同时，将查询路由到满足质量阈值的最具成本竞争力的提供商。

    

    基准测试与路由平台日益成为连接大型语言模型提供商与终端用户的中介。然而，这些平台上的提供商通常采用固定的每token定价方式，使用户无法为其任务获得最具竞争力的价格。在本工作中，我们设计了一个采购平台，其中每个任务的token价格由提供商之间的竞争驱动，使用户能够在保证质量水平的前提下获得有竞争力的价格。为此，该平台通过反向第二价格拍卖依次路由查询，激励模型提供商真实地竞标其服务用户查询的平均成本的最佳估计。在路由查询的过程中，平台学习每个提供商所提供的质量，并逐步将查询路由到满足期望质量阈值的提供商中最具成本竞争力的提供商。为验证我们的设计，我们使用多个模型进行了实验。

    arXiv:2609.28322v1 Announce Type: new  Abstract: Benchmarking and routing platforms increasingly act as intermediaries connecting large language model providers with end-users. However, providers on these platforms typically use a fixed price per token, preventing users from achieving the most competitive price for their tasks. % workloads. In this work, we design a procurement platform where token prices for each task are driven by provider competition, enabling users to secure competitive pricing for guaranteed quality levels. To this end, the platform sequentially routes queries via a reverse second-price auction that incentivizes model providers to truthfully bid their best estimate of the average cost to serve a user's query. As it routes queries, the platform learns the quality offered by each provider and progressively routes queries to the most cost-competitive provider among those meeting a desired quality threshold. To validate our design, we conduct experiments with multiple
    
[^230]: 通过条件流匹配蒸馏实现高效的多任务操作策略

    Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching

    [https://arxiv.org/abs/2609.28107](https://arxiv.org/abs/2609.28107)

    该论文提出通过迁移单任务条件流匹配专家模型学习到的速度场，将其知识蒸馏到一个共享的多任务策略中，并结合原始CFM目标保持对专家演示的保真度，从而实现计算高效的多任务机器人操作策略学习。

    

    生成式建模的最新进展近来已被广泛应用于机器人学的策略学习中。特别是，使用专家演示训练的条件流匹配（CFM）在机器人操作基准测试中已被证明优于现有方法。虽然先前的工作主要集中于单任务设置，但我们从多任务的角度来研究这个问题，因为为每个任务训练独立模型的计算成本非常高。多任务策略学习本身也面临一系列挑战：在简单拼接的演示数据集上进行朴素训练，要么需要增加模型容量以适应额外的复杂性，要么会导致性能下降。我们提出通过迁移单任务CFM专家模型所学习到的速度场，将其知识蒸馏到一个共享的多任务策略中。我们将该蒸馏信号与原始的CFM目标相结合，以保持对专家演示数据的保真度。

    arXiv:2609.28107v1 Announce Type: cross  Abstract: Advances in generative modeling have recently been extensively employed in robotics for policy learning. In particular, Conditional Flow Matching (CFM) trained with expert demonstrations has been shown to outperform existing methods on robot manipulation benchmarks. While prior work has mainly focused on single-task settings, we study the problem from a multi-task perspective, as training independent models for each task is computationally expensive. Multi-Task policy learning comes with its own set of challenges, as naively training on a concatenated dataset of demonstrations would either require increased model capacity to accommodate the added complexity or result in drops in performance. We propose to distill knowledge from single-task CFM experts into a shared multi-task policy by transferring their learned velocity fields. We combine this distillation signal with the original CFM objective to retain fidelity to the demonstrations
    
[^231]: LAYERSCOPE：视频与多模态学习表征的逐层刻画

    LAYERSCOPE: A Layerwise Characterization of Video and Multimodal Learned Representations

    [https://arxiv.org/abs/2609.28086](https://arxiv.org/abs/2609.28086)

    提出无标签逐层分析框架LAYERSCOPE，通过多种几何度量刻画视频与多模态模型的逐层表征结构，发现中间层表征可优于最终层输出，且单一几何度量无法可靠预测下游性能。

    

    我们提出了LAYERSCOPE，这是一个无标签的逐层分析框架，旨在刻画模型在视频和多模态场景下学习到的表征。使用最终层或中间层表征来评估下游性能，通常需要大量带标签的数据、重复的任务特定评估以及大量的计算。为了解决这些局限性，LAYERSCOPE利用局部、全局、分布以及基于对应关系的几何度量，在无需任务特定标签的情况下，比较模型内部以及跨模型的逐层表征结构。我们在MVEB/MVEB+的多个任务上评估了七个架构各异的模型，涵盖视频与多模态分类、聚类以及文本到视频检索。我们发现中间层的表征可以优于最终层和模型默认输出。我们还发现没有任何单一的几何度量能够一致地预测下游性能，但注意到

    arXiv:2609.28086v1 Announce Type: cross  Abstract: We propose LAYERSCOPE, a label-free, layerwise framework that aims to characterize a model's learned representations in video and multimodal settings. Evaluating downstream performance using representations from final or intermediate layers typically requires large amounts of labeled data, repeated task-specific evaluations, and substantial computation. To address these limitations, LAYERSCOPE uses local, global, distributional, and correspondence-based geometric metrics to compare layerwise representation structure within and across models without requiring task-specific labels. We evaluate seven architecturally diverse models across video and multimodal classification, clustering, and text-to-video retrieval tasks from MVEB/MVEB+. We find that intermediate-layer representations can outperform final-layer and model-default outputs. We also find that no single geometric metric consistently predicts downstream performance, but note that
    
[^232]: NS-Attention：视觉Transformer中注意力输出的Newton-Schulz变换

    NS-ATTENTION: Newton-Schulz Transformations of Attention Outputs in Vision Transformers

    [https://arxiv.org/abs/2609.27735](https://arxiv.org/abs/2609.27735)

    提出无参数的Newton-Schulz注意力变换（NS-Attn.），对每个注意力头输出进行谱处理以降低谱集中度并提高有效秩，在ViT和Swin于CIFAR-10/100的全部12组对比实验中均带来平均0.25–0.83个百分点的准确率提升。

    

    Newton-Schulz（NS）迭代最近被用于Muon优化器中，在大语言模型训练过程中对更新矩阵进行变换。受其谱效应的启发，我们研究将NS直接应用于Transformer的注意力表示。我们提出Newton-Schulz注意力（NS-Attn.），这是一种应用于每个注意力头输出的无参数变换。每个注意力头的输出被排列为特征×令牌矩阵，并通过其Frobenius范数进行归一化，随后应用有限步的NS多项式迭代，再恢复原始范数。其目标是在标准的头合并与输出投影之前，降低谱集中度并提高有效秩。在CIFAR-10和CIFAR-100数据集上对ViT和Swin的实验中，NS-Attn.在所有12组同种子对比中均提升了最终轮次的准确率，平均增益为0.25至0.83个百分点。ViT消融实验表明，一次迭代的平均准确率高于两次迭代。谱分析……（原文截断）

    arXiv:2609.27735v1 Announce Type: new  Abstract: Newton-Schulz (NS) iteration has recently been used in the Muon optimizer to transform update matrices during the training of large language models. Motivated by its spectral effect, we investigate applying NS directly to Transformer attention representations. We introduce Newton-Schulz Attention (NS-Attn.), a parameter-free transformation applied to the output of each attention head. Each head output is arranged as a feature-by-token matrix and normalized by its Frobenius norm. We then apply a finite NS polynomial step and restore the original norm. The objective is to reduce spectral concentration and increase effective rank before standard head merging and output projection. Across ViT and Swin on CIFAR-10 and CIFAR-100, NS-Attn. improves final-epoch accuracy in all 12 matched-seed comparisons, with mean gains of 0.25--0.83 percentage points. ViT ablations show higher mean accuracy with one iteration than with two. Spectral analysis f
    
[^233]: ProCredit：智能体强化学习中从结果奖励到进展信用的转变

    ProCredit: From Outcome Rewards to Progress Credit in Agentic Reinforcement Learning

    [https://arxiv.org/abs/2609.27532](https://arxiv.org/abs/2609.27532)

    提出 ProCredit，利用可在中间状态上运行的验收检查，把与最终结果同样可验证的任务进展转化为逐步的信用信号，从而克服长程智能体强化学习中仅依赖结果奖励导致的训练信号稀疏、失败尝试无法区分、推进任务的步骤得不到应得信用等问题。

    

    长程智能体任务要求智能体通过一系列工具调用对环境进行修改，任务成败由最终状态决定。标准做法是在任务结束时给出单一的结果奖励，并对同一任务采样得到的多条轨迹进行比较。由此带来的问题是：当一组采样中没有任何成功轨迹时，训练便得不到任何信号；失败的尝试无法按照其接近完成的程度加以区分；而真正推动任务进展的步骤与仅仅查询环境的步骤会获得相同的信用。已有工作或将比较的单位从轨迹细化为步骤，或训练一个奖励模型来提供中间信号：前者仍然只能从最终成败中获取信号，后者则需要依赖模型来估计信号。我们观察到，用于判定成功的验收检查同样可以作用于中间状态，因此任务进展与最终结果一样是可验证的。我们提出 ProCredit，它将这种经过验证的进展……（摘要原文在此处截断）

    arXiv:2609.27532v1 Announce Type: cross  Abstract: Long-horizon agentic tasks require an agent to modify an environment through a sequence of tool calls, with success determined by the final state. The standard recipe assigns a single outcome reward at the end and compares trajectories sampled for the same task. As a result, a group with no successful trajectory yields no training signal, failed attempts cannot be told apart by how close they came to completion, and turns that advance the task receive the same credit as turns that only query the environment. Prior work refines the unit of comparison from the trajectory to the step, or trains a reward model to supply intermediate signal: the former still derives its signal from final success alone, and the latter estimates it with a model. We observe that the acceptance checks that decide success can also be run on intermediate states, so progress is as verifiable as the outcome. We propose ProCredit, which turns this verified progress 
    
[^234]: 大型推理模型在推理时能力与效率的扩展规律

    Scaling of Capability and Efficiency at Inference Time in Large Reasoning Models

    [https://arxiv.org/abs/2609.27166](https://arxiv.org/abs/2609.27166)

    本文利用层次贝叶斯模型量化了DeepSeek-R1-Distill系列模型在算术与算法推理任务上的表现，发现正确解题概率随问题难度近似指数衰减，而衰减尺度随模型规模增长，从而揭示了推理时能力与效率随模型规模的扩展规律。

    

    能力与效率是大型语言模型（LLM）推理的两个关键维度。能力指正确解决给定问题的能力，而效率指在有限资源下完成这一任务的能力。当LLM使用思维链推理来求解难度受控的问题时，正确解决的问题数量以及得到正确答案所需的token数量都取决于问题难度和模型规模。然而，这些因素如何共同塑造能力与效率仍知之甚少。本文使用层次贝叶斯模型，评估了DeepSeek-R1-Distill模型家族的LLM在四类算术与算法推理问题上的能力与效率。在固定模型规模下，正确解决一个实例的概率随实例规模（作为问题难度的代理指标）近似呈指数衰减，该衰减尺度随模型规模而增长……

    arXiv:2609.27166v1 Announce Type: new  Abstract: Capability and efficiency are two key dimensions of reasoning in large language models (LLMs). Capability refers to the ability to solve a given problem correctly, whereas efficiency refers to the ability to do so with limited resources. When LLMs use Chain-of-Thought (CoT) reasoning to solve problems of controlled hardness, both the number of problems solved correctly and the number of tokens required to reach a correct answer depend on problem hardness and model size. However, how these factors jointly shape capability and efficiency remains poorly understood. Here, we use hierarchical Bayesian models to evaluate the capability and efficiency of LLMs from the DeepSeek-R1-Distill model family across four classes of arithmetic and algorithmic reasoning problems. At a fixed model size, the probability of correctly solving an instance decays approximately exponentially with instance size, our proxy for problem hardness. The decay scale gro
    
[^235]: 安全提示：面向用户的实时AI风险感知干预措施

    Safety Nudges: User-Facing Interventions for Real-Time AI Risk Awareness

    [https://arxiv.org/abs/2609.26865](https://arxiv.org/abs/2609.26865)

    该研究提出了Safety Nudges——一款基于浏览器的工具，能在聊天机器人对话中实时检测并提示潜在的AI安全风险，实地研究表明此类面向用户的干预措施能有效提升用户对AI危害的意识，可作为模型层面安全防护的有益补充。

    

    对话式AI系统可能对其用户构成安全风险，例如幻觉、谄媚、过度自信和拟人化，但这些风险在用户日常使用中难以察觉。我们介绍了Safety Nudges，这是一种基于浏览器的工具，当检测到聊天机器人对话中出现令人担忧的行为时，它会提供轻量级的即时标记。我们通过一项为期两周的实地研究对Safety Nudges进行了评估，该研究涉及45名频繁使用聊天机器人的用户，收集了交互日志、调查问卷以及用户对各条提示的反馈。参与者认为该工具有用、清晰且干扰性最小，几乎所有用户都报告称对潜在AI危害的意识有所提高，尽管我们发现仅凭这种意识提升并不一定能带来可观察到的行为改变。我们的结果表明，面向用户的安全提示可以通过帮助人们在具体情境中批判性地评估AI回应，来补充模型层面的安全防护措施，同时强调了……

    arXiv:2609.26865v1 Announce Type: cross  Abstract: Conversational AI systems can pose safety risks to their users such as hallucination, sycophancy, overconfidence, and anthropomorphism, but these risks are difficult for users to detect during everyday use. We introduce Safety Nudges, a browser-based tool that provides lightweight, in situ flags when concerning behavior is detected in chatbot conversations. We evaluated Safety Nudges in a two-week field study with 45 frequent chatbot users, collecting interaction logs, surveys, and feedback on individual nudges. Participants found the tool useful, clear, and minimally disruptive, with nearly all users reporting an increased awareness of potential AI harms, though we found that this improved awareness alone did not necessarily lead to discernible behavioral changes. Our results suggest that user facing safety nudges can complement model-level safeguards by helping people critically evaluate AI responses in context, while highlighting th
    
[^236]: QUARTET：基于四分支交叉注意力与随机游走轨迹的关系图Transformer增强方法

    QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs

    [https://arxiv.org/abs/2609.26855](https://arxiv.org/abs/2609.26855)

    提出QUARTET图Transformer架构，利用基于近期截断个性化PageRank的因果随机游走采样器提取密集连通且无时序泄露的局部子图，并通过四分支交叉注意力丰富全局上下文，从而克服RelGT在关系图建模中局部采样松散与全局记忆单一的局限。

    

    arXiv:2609.26855v1 公告类型：交叉 摘要：关系深度学习将多表数据库建模为异构时序图，图Transformer目前在RelBench等基准测试上取得了最先进的性能。然而，当前领先的模型RelGT存在两个关键局限：其随机局部采样器生成的子图连接松散，阻碍了消息传递；其全局注意力模块依赖于单一的、基于种子特征的内存，忽略了更广泛的宏观层面动态。为克服这些局限，我们提出了QUARTET，一种表达能力强的图Transformer架构，它在局部子图上应用完全自注意力，同时通过交叉注意力分支来丰富全局上下文。具体而言，QUARTET采用基于近期截断个性化PageRank（PPR）的因果随机游走（CRW）采样器，以提取紧凑、抗枢纽节点干扰且密集连通的局部子图，且不会产生时序信息泄露。与此同时，四分支交叉注意力（摘要在此处截断）

    arXiv:2609.26855v1 Announce Type: cross  Abstract: Relational Deep Learning (RDL) models multi-table databases as heterogeneous temporal graphs, and graph transformers currently achieve state-of-the-art performance on benchmarks like RelBench. However, the current leading model, RelGT, suffers from two key limitations: its random local sampler yields loosely connected subgraphs that hinder message passing, and its global attention module relies on a single, seed-feature-based memory that ignores broader macro-level dynamics. To overcome these limitations, we introduce QUARTET, an expressive graph transformer architecture that applies full self-attention on local subgraphs while enriching global context through cross-attention branches. Specifically, QUARTET employs a Causal Random Walk (CRW) sampler based on recency-truncated Personalized PageRank (PPR) to extract compact, hub-robust, and densely connected local subgraphs without temporal leakage. Concurrently, a quad-branch cross-atte
    
[^237]: 学习波动：因果表格预训练的统计基础

    Learning to Fluctuate: Statistical Foundations for Causal Tabular Pretraining

    [https://arxiv.org/abs/2609.26290](https://arxiv.org/abs/2609.26290)

    提出波动监督预训练（FSP），用平均处理效应加有效影响函数波动来标注合成表格，并证明完全波动可使高斯标签可观测，从而将因果标签预测风险从 $(1-\lambda)^2/n$ 阶降至 $n^{-2}$ 阶。

    

    因果表格基础模型在多个合成机制之间摊销效应估计，但潜在效应监督只奖励后验收缩，而非直接编码固定部署总体中所需的重复样本响应。我们提出波动监督预训练（FSP）：每个合成表格由其平均处理效应加上其有效影响函数的波动来标注，而部署时仍只需一次冻结的前向传播。沿着路径 $T_{\lambda,P}=\theta(P)+\lambda P_n\psi_P$，我们证明了一个端点相变：每个固定的 $\lambda<1$ 都会保留 $(1-\lambda)^2/n$ 阶的标签模糊性，而完全波动使高斯标签变得可观测，并将最优有限层因果标签预测风险降至 $n^{-2}$ 阶。一个有限预训练界综合了标签误差、网络误差、回合采样误差与优化误差；其产生的采样缺陷控制了固定机制偏差、均方……（摘要截断）

    arXiv:2609.26290v1 Announce Type: cross  Abstract: Causal tabular foundation models amortize effect estimation across synthetic mechanisms, but latent-effect supervision rewards posterior shrinkage instead of directly encoding the repeated-sample response needed in a fixed deployment population. We introduce fluctuation-supervised pretraining (FSP): each synthetic table is labeled by its average treatment effect plus its efficient influence-function fluctuation, while deployment remains a single frozen forward pass. Along the path $T_{\lambda,P}=\theta(P)+\lambda P_n\psi_P$, we prove an endpoint transition: every fixed $\lambda<1$ retains label ambiguity of order $(1-\lambda)^2/n$, whereas full fluctuation makes the Gaussian label observable and reduces optimal finite-stratum causal label-prediction risk to order $n^{-2}$. One finite-pretraining bound combines label, network, episode-sampling, and optimization errors; its resulting sampling defect controls fixed-mechanism bias, mean sq
    
[^238]: RRSI：智能体框架的正则化递归自我改进

    RRSI: Regularized Recursive Self-Improvement of Agent Harnesses

    [https://arxiv.org/abs/2609.24972](https://arxiv.org/abs/2609.24972)

    该论文提出RRSI方法，通过将正则化原则（如时间退火的编辑预算限制和鼓励探索未开发轨迹）引入智能体框架的递归自我改进过程，防止递归进化对训练任务过拟合，从而提升分布外基准上的泛化能力。

    

    LLM智能体的能力在很大程度上被其“框架”所放大，即围绕冻结骨干模型的提示词、控制流、工具、记忆和上下文管理。近期的方法通过迭代地提出并选择对智能体框架的组件级编辑，日益将这一过程自动化，实际上在智能体系统层面建立了一种递归自我改进（RSI）的形式。然而，这种递归进化可能因记忆训练任务而过拟合，在分布内基准上表现出巨大收益，但在分布外基准上收益缩小甚至消失。我们提出了智能体框架的正则化递归自我改进（RRSI），通过约束进化候选的提案与选择，将正则化原则融入框架自我改进之中。提案者以时间退火的预算运行，限制候选可以捆绑的编辑数量，并鼓励探索未开发的轨迹……

    arXiv:2609.24972v1 Announce Type: cross  Abstract: An LLM agent's capability is largely magnified by its harness, namely the prompts, control flow, tooling, memory, and context management surrounding the frozen backbone model. Recent methods increasingly automate this process by iteratively proposing and selecting component-wise edits of an agent harness, practically establishing a form of recursive self-improvement (RSI) at the agent-system level. However, such recursive evolution may overfit by memorizing the training tasks, showing large in-distribution gains that shrink or even vanish on out-of-distribution benchmarks. We introduce Regularized Recursive Self-Improvement of Agent Harnesses (RRSI), which incorporates the principles of regularizations into harness self-improvement by constraining the evolution candidate proposal and selection. The proposer operates with a temporally annealed budget, limiting how many edits a candidate can bundle, and it encourages unexplored trajector
    
[^239]: 从高带宽单点感知中学习触觉感知

    Learning tactile perception from high-bandwidth single-point sensing

    [https://arxiv.org/abs/2609.24621](https://arxiv.org/abs/2609.24621)

    SpectRobot框架将高带宽单点触觉信号转换为时频频谱图，使其能被标准视觉编码器直接处理，从而无需空间分布式传感器阵列即可实现丰富的触觉感知。

    

    触觉感知正日益被融入基于学习的机器人操作中，然而许多现有方法依赖于空间分布式的传感器。在这里，我们提出了SpectRobot，一个将单点触觉信号转换为紧凑的时频频谱图的框架。这些频谱图将高带宽的触觉历史编码为固定大小的类图像表示。它们可以由标准的视觉编码器处理，并集成到最初为视觉任务开发的机器学习流水线中，同时保留了传统相机无法获取的时间和频率信息。SpectRobot不是通过触觉元件阵列来增加空间密度，而是利用稀疏、高带宽的单点测量中蕴含的丰富动态信息。在我们的实现中，传感器安装在远离接触表面的位置，同时保持与其机械耦合，从而减少直接暴露于磨损，并可能……

    arXiv:2609.24621v1 Announce Type: cross  Abstract: Tactile sensing is increasingly being incorporated into learning-based robotic manipulation, yet many existing approaches rely on spatially distributed sensors. Here we introduce SpectRobot, a framework that transforms single-point tactile signals into compact time-frequency spectrograms. These spectrograms encode high-bandwidth tactile histories as fixed-size image-like representations. They can be processed by standard vision encoders and integrated into learning pipelines originally developed for vision, while preserving temporal and frequency information unavailable to conventional cameras.   Rather than increasing spatial density through arrays of tactile elements, SpectRobot exploits the rich dynamics contained in sparse, high-bandwidth single-point measurements. In our implementation, the sensors are mounted away from the contact surface while remaining mechanically coupled to it, reducing direct exposure to wear and potentially
    
[^240]: 再看一眼：利用稀疏观测校正海冰预报

    Taking a Second Look: Correcting Sea Ice Forecasts with Sparse Observations

    [https://arxiv.org/abs/2609.24591](https://arxiv.org/abs/2609.24591)

    提出ECHO框架，通过状态自适应的异构传播策略（ECHO-Scale动态调整传播距离、ECHO-Delta学习有界残差）利用稀疏海冰密集度观测校正海冰预报，在全部96个评估设置中均优于固定传播方法。

    

    海冰预报会提前数天发布，在此期间误差会不断累积，而新的、通常较为稀疏的海冰密集度（SIC）观测数据会陆续变得可用。我们发现，固定传播方式产生的误差集中在结构化的高梯度冰缘附近，而均匀的内部区域只需有限的传播，这表明传播距离应当依赖于状态。因此，我们提出了ECHO（基于证据引导的异构传播校正），其中ECHO-Scale在保持校正几何结构的同时自适应调整传播距离，而ECHO-Delta则在固定传播的基础上学习有界残差。在涵盖不同先验、观测时间、稀疏程度、几何形态和噪声条件的全部96个标准评估设置中，两种方法均优于固定传播。ECHO-Delta取得了最佳的平均精度，而ECHO-Scale对几何形态变化更具鲁棒性。代码可在 https://github.com/yingtian22/TAKING-A-SEC 获取。

    arXiv:2609.24591v1 Announce Type: new  Abstract: Sea ice forecasts are issued several days ahead, allowing errors to accumulate while new, often sparse sea ice concentration (SIC) observations become available. We find that fixed-propagation errors concentrate near structured, high-gradient ice edges, whereas homogeneous interiors require limited propagation, suggesting that propagation distance should be state dependent. We therefore introduce ECHO (Evidence-guided Correction with Heterogeneous prOpagation), where ECHO-Scale adapts propagation distance while preserving correction geometry, and ECHO-Delta learns a bounded residual around fixed propagation. Across all 96 standard evaluation settings spanning diverse priors, observation times, sparsity levels, geometries, and noise conditions, both outperform fixed propagation. ECHO-Delta achieves the best average accuracy, while ECHO-Scale is more robust to geometry shifts. Code is available at https://github.com/yingtian22/TAKING-A-SEC
    
[^241]: 审计贝叶斯图对齐：诊断方法比较与参考失效

    Auditing Bayesian Graph Alignment: Diagnostic Comparisons and Reference Failure

    [https://arxiv.org/abs/2609.23232](https://arxiv.org/abs/2609.23232)

    该论文在多组图对上系统比较了贝叶斯图对齐的多种收敛诊断方法，发现没有任何单一诊断在所有采样器和场景下占优，且在大规模图中诊断只能预测后续边缘变化而非后验误差。

    

    贝叶斯图对齐用于估计对应关系概率，但对齐分数轨迹的收敛并不必然意味着对应边缘分布的准确。我们在240对来自四个源族的新精确图对、240对具有20-100个顶点的更大规模图对，以及一个独立的60例精确实现检验上审计了这一差距。在显式的边翻转似然下，我们比较了三种采样器以及基于分数、边缘、指示器、类别和分类器的诊断方法。对于精确知情采样器，边缘分歧在错误判别上优于分数R-hat，但其对普通局部采样的改进尚不确定。基于分配的R*和短指示器面板具有竞争力；没有任何一种诊断方法在所有采样器和端点上占优。在更大规模下，诊断方法预测的是后续边缘变化而非后验误差，且分类性能取决于漂移阈值。不相交窗口和留出链检验减弱了……（原文在此处截断）

    arXiv:2609.23232v1 Announce Type: cross  Abstract: Bayesian graph alignment estimates correspondence probabilities, but convergence of an alignment-score trace need not imply accurate correspondence marginals. We audit this gap on 240 new exact graph pairs from four source families, 240 larger pairs with 20-100 vertices, and a separate 60-case exact implementation check. Under an explicit edge-flip likelihood, we compare three samplers and score, marginal, indicator, categorical, and classifier-based diagnostics. Marginal disagreement improves error discrimination over score R-hat for the exact informed sampler, but its improvement for vanilla local sampling is uncertain. Assignment-based R* and short indicator panels are competitive; no diagnostic dominates across samplers and endpoints. At larger sizes, diagnostics predict subsequent marginal changes, not posterior error, and classification performance depends on the drift threshold. Disjoint-window and held-out-chain checks attenuat
    
[^242]: StationPDE：面向站点的地表偏微分方程学习，用于多站点多变量天气预报

    StationPDE: Station-Oriented Surface PDE Learning for Multi-Station Multivariate Weather Forecasting

    [https://arxiv.org/abs/2609.22123](https://arxiv.org/abs/2609.22123)

    StationPDE提出了一种面向站点的地表PDE学习模型，将天气演化分解为地表风输运与可学习的高空推断，从而在缺乏连续场和高空数据的条件下实现具有物理可解释性的多站点多变量天气预报。

    

    多站点多变量天气预报旨在从历史地面观测数据预测多个气象站点未来的天气变量。现有的站点预报模型学习离散站点之间的统计依赖关系，但缺乏显式的物理演化过程。与此同时，基于偏微分方程（PDE）的天气模型提供了可解释的物理动力学，然而其需要连续场和地面站点数据中无法获取的高空变量。为了弥合这一差距，我们提出了StationPDE，一种面向站点的地表PDE学习模型。StationPDE从离散的站点观测中构建具有地形感知能力的连续地表场，并将其物理演化分解为地表风输运和高空推断两个部分。地表风输运显式地演化可观测的天气变量，而高空推断则利用可学习的水平扩散来近似无法获取的高空变量所带来的缺失影响。一个并行的……

    arXiv:2609.22123v1 Announce Type: new  Abstract: Multi-station multivariate weather forecasting aims to forecast future weather variables at multiple weather stations from historical surface observations. Existing station forecasting models learn statistical dependencies among discrete stations, but lack explicit physical evolution. Meanwhile, PDE-based weather models provide interpretable physical dynamics, yet require continuous fields and upper-air variables unavailable in surface station data. To bridge this gap, we propose StationPDE, a station-oriented surface PDE learning model. StationPDE constructs a terrain-aware continuous surface field from discrete station observations and decomposes its physical evolution into surface wind transport and upper-air inference. Surface wind transport explicitly evolves observable weather variables, while upper-air inference uses learnable horizontal diffusion to approximate the missing influence of unavailable upper-air variables. A parallel 
    
[^243]: 完备神经电子初始化加速材料密度泛函理论计算

    Complete Neural Electronic Initialization Accelerates Materials DFT

    [https://arxiv.org/abs/2609.21759](https://arxiv.org/abs/2609.21759)

    该论文提出了首个完备的机器学习方法来加速PAW形式下的材料平面波DFT计算，通过形式化七项准则并引入AugNet——首个PAW增强占据数通用等变模型和首个材料通用自旋密度模型——填补了现有方法中缺失的关键初始化组件。

    

    我们提出了首个在投影增强波（PAW）形式下加速材料平面波密度泛函理论（DFT）计算的完备机器学习方法。我们形式化了“完备神经电子初始化器”为实现实用的端到端PAW DFT加速所必须满足的七项准则。将这些准则应用于先前的工作，揭示了两个缺失的结构相关组件——增强占据数和自旋初始化——它们阻碍了现有方法提供完备的无参考初始化。受控消融实验表明，省略这些组件可能会消除或逆转仅预测平滑价电子密度的模型所获得的加速效果。我们通过引入AugNet来满足这些缺失的要求，AugNet是首个针对PAW增强占据数的通用等变模型，同时也是首个针对材料的通用自旋密度模型，它可以预测平滑的自旋差密度。

    arXiv:2609.21759v1 Announce Type: cross  Abstract: We present the first complete machine learning method for accelerating plane-wave density functional theory (DFT) in materials under the projector augmented wave (PAW) formalism. We formalize seven criteria that a \textit{Complete Neural Electronic Initializer} must satisfy for practical end-to-end PAW DFT acceleration. Applying these criteria to prior work reveals two missing structure-dependent components, augmentation occupancies and spin initialization, that prevent existing methods from providing complete reference-free initialization. Controlled ablations show that omitting these components can eliminate or reverse the acceleration obtained via models that only predict the smooth valence density. We satisfy these missing requirements by introducing AugNet, the first general equivariant model for PAW augmentation occupancies, and the first general spin density model for materials, which predicts the smooth spin-difference density 
    
[^244]: 一个语言模型评审团相当于多少个人类评判者？

    How Many Humans Is a Judge Panel Worth?

    [https://arxiv.org/abs/2609.21277](https://arxiv.org/abs/2609.21277)

    该论文提出两种不同的“等效人类评判者数量”度量——谱残差多样性 ν_H 与分布平方误差 ν_MSE，发现同一组 32 个语言模型评审在三个 ChaosNLI 任务上分别相当于约 4.24–6.50 个和 2.30–3.75 个人类评判者，且更大的谱多样性并不保证更好的分布恢复。

    

    一组语言模型评审团究竟代表多少个人类判断？答案取决于匹配的对象是什么。我们针对经验人类标签分布对类别型评审团进行审计，保留了那些相对于单一金标准标签的二值错误所坍缩掉的分歧。我们通过将归一化残差格拉姆矩阵的参与率与条件独立的人类参考抽样相匹配来度量谱残差多样性，得到 ν_H；并单独匹配分布平方误差，得到 ν_MSE。在三个 ChaosNLI 任务上，同一组由 32 个评审组成的评审团，其 ν_H 为 4.24–6.50，而 ν_MSE 仅为 2.30–3.75。一个谱恒等式将决定误差的特征值、成员能量和平均方向权重分离开来。可实现的硬标签评审团表明，即使成员能量相等且相关性非负，更大的谱多样性也可能伴随更差的分布恢复。在观察到的评审团中，同规模内的排名一致性……（摘要原文在此处截断）

    arXiv:2609.21277v1 Announce Type: new  Abstract: How many human judgments does a panel of language models represent? The answer depends on what is matched. We audit categorical judge panels against empirical human label distributions, retaining disagreement that binary errors relative to one gold label collapse. We measure spectral residual diversity by matching the participation ratio of a normalized residual Gram matrix to conditionally independent human-reference draws, giving nu_H. We separately match distributional squared error, giving nu_MSE. Across three ChaosNLI tasks, the same 32-judge panels have nu_H=4.24--6.50 but nu_MSE=2.30--3.75. A spectral identity separates the eigenvalues, member energies, and averaging-direction weights that determine error. Realizable hard-label panels show that greater spectral diversity can accompany worse distribution recovery even with equal member energies and nonnegative correlations. In the observed panels, within-size ranking agreement vari
    
[^245]: 动态广义Gromov-Wasserstein最优传输

    Dynamic Generalized Gromov-Wasserstein Optimal Transport

    [https://arxiv.org/abs/2609.20008](https://arxiv.org/abs/2609.20008)

    该论文提出TP-DATE框架，首次以无模拟方式将Gromov-Wasserstein最优传输动态化，通过路径作用量证明静态与动态二次型最优传输的等价性，并发展行进对流匹配方法，实现空间转录组学中兼顾组织结构保持的连续轨迹重建。

    

    Gromov-Wasserstein最优传输（GW-OT）通过引入结构感知的传输代价扩展了经典最优传输。这对于空间转录组学尤为重要，因为在空间转录组学中，动态重建除了匹配表达模式外，还应保留组织结构。尽管静态形式已被广泛用于此类结构感知对齐，但用于重建连续轨迹的一般动态形式仍然缺失。我们引入了行进对动态对齐与轨迹估计，这是一个以无模拟方式动态推广GW-OT的理论与计算框架。我们通过路径作用量表述了一大类静态和动态二次型最优传输（QOT），并证明了静态与动态的等价性。我们进一步发展了行进对流匹配方法，该方法允许条件路径之间相互作用，并将其交互边缘化为单一向量场（摘要在此处被截断）。

    arXiv:2609.20008v1 Announce Type: cross  Abstract: Gromov--Wasserstein optimal transport (GW-OT) extends classical optimal transport by introducing structure-aware transport cost. This is particularly relevant for spatial transcriptomics, where dynamical reconstruction should preserve tissue structure in addition to matching expression patterns. While static formulations have been widely used for such structure-aware alignment, a general dynamic formulation for reconstructing continuous trajectories is still missing. We introduce Travelling Pair Dynamical Alignment and Trajectory Estimation (TP-DATE), a theoretical and computational framework to generalize GW-OT dynamically in a simulation-free manner. We formulate a broad class of static and dynamic Quadratic-form OT (QOT) through path actions and prove the static dynamic equivalence. We further develop travelling-pair flow matching, which allows interacting conditional paths and marginalizes their interactions into a single vector fi
    
[^246]: REARL：一种结合真实交通数据与大语言模型的闭环自动驾驶仿真增强框架

    REARL: A Closed-loop Autonomous Driving Simulation Enhancement Framework with Real Traffic Data and Large Language Models

    [https://arxiv.org/abs/2609.19903](https://arxiv.org/abs/2609.19903)

    REARL提出了一种结合真实交通数据聚类与大语言模型的闭环仿真增强框架，通过实时监测仿真与真实交通的偏差并动态调整车辆决策，使自动驾驶仿真更贴近真实交通模式。

    

    准确的仿真对自动驾驶开发至关重要，然而捕捉真实世界的交通复杂性仍然具有挑战性。依赖预定义规则或静态数据回放的现有仿真器难以应对动态交通场景。CRITICAL方法使用真实交通数据和大语言模型（LLM）来调整初始仿真配置，但随着仿真过程的推进，仿真分布仍会偏离真实交通。我们提出REARL，一个将真实交通数据与大语言模型相结合的闭环仿真增强框架。该方法对真实交通数据进行聚类，每个聚类中心作为代表性场景，为大语言模型提供典型的真实世界交通模式。随后，一个定时滑动窗口检测器监测车辆速度分布以及车辆对之间平均间距的偏差。若某项指标超过阈值，则由大语言模型调整车辆的决策行为；否则保留现有控制器继续运行。

    arXiv:2609.19903v1 Announce Type: new  Abstract: Accurate simulation is crucial for autonomous driving development, yet capturing real-world traffic complexity remains challenging. Existing simulators that rely on predefined rules or static data playback struggle with dynamic traffic. CRITICAL uses real traffic data and a large language model (LLM) to adjust the initial simulation configuration, but the simulated distribution still diverges from real traffic as the rollout evolves. We propose REARL, a closed-loop simulation enhancement framework that integrates real traffic data with LLMs. Real traffic data are clustered, and each cluster center is used as a representative scenario that provides typical real-world traffic patterns for the LLM. A timed sliding-window detector then monitors discrepancies in vehicle speed distribution and mean spacing between pairs of vehicles. If a metric exceeds a threshold, the LLM adjusts vehicle decision-making; otherwise the existing controller is k
    
[^247]: TacSushi：面向灵巧寿司操作的触觉接地世界-动作建模

    TacSushi: Tactile-Grounded World-Action Modeling for Dexterous Sushi Manipulation

    [https://arxiv.org/abs/2609.19613](https://arxiv.org/abs/2609.19613)

    提出TacSushi，一种触觉接地的世界-动作建模策略，通过特征级门控融合指尖触觉并利用失败试验的未来后果预测进行监督，实现了形变、遮挡和不确定接触条件下的灵巧寿司操作。

    

    灵巧的食物操作需要在形变、遮挡和不确定接触条件下的控制。我们提出TacSushi，一种基于触觉接地、基于Cosmos3的世界-动作策略，它在作用于当前观测的同时，从记录的未来后果中学习。骨干网络编码当前的RGB图像、语言和手部状态，特征级门控融合将指尖触觉特征融入动作表示。在训练期间，一个以示范动作块为条件的解码器预测记录的未来视觉观测、任务进度、相对接触风险和触觉摘要；该解码器在部署时被移除。失败的试验提供后果监督，但其动作被排除在模仿学习之外。我们在340次成功和50次失败的真实机器人试验上训练TacSushi，并在600次独立测试中比较六种方法，涵盖三个分布内任务和两个分布外食材变体。为了评估食品质量……

    arXiv:2609.19613v1 Announce Type: cross  Abstract: Dexterous food manipulation requires control under deformation, occlusion, and uncertain contact. We present TacSushi, a tactile-grounded, Cosmos3-based world-action policy that learns from recorded future consequences while acting on current observations. The backbone encodes current RGB, language, and hand state, and feature-wise gated fusion incorporates fingertip tactile features into the action representation. During training, a decoder conditioned on demonstrated action chunks predicts logged future visual observations, task progress, relative contact risk, and tactile summaries; this decoder is removed at deployment. Failed trials provide consequence supervision, but their actions are excluded from imitation. We train TacSushi on 340 successful and 50 failed real-robot trials and compare six methods in 600 separate rollouts across three in-distribution tasks and two out-of-distribution ingredient variants. To assess food quality
    
[^248]: TERN：一种用于疫情预测的带季节参考与在线适应的Delta规则记忆模型

    TERN: A Delta-rule Memory with a Seasonal Reference and Online Adaptation for Epidemic Forecasting

    [https://arxiv.org/abs/2609.18407](https://arxiv.org/abs/2609.18407)

    TERN是一种基于delta规则快速权重记忆的流感疫情预测模型，通过由疫情阶段特征驱动的门控擦除机制、显式季节参考和在线适应，在多个流感基准测试上超越了现有疫情图模型和通用预测器。

    

    每周的流感监测数据用于指导疫苗分发和公共卫生警报，但其预测十分困难。每个地区仅提供少数几个季节的数据，疫情波每年在时间和高度上都会发生变化，在疫情波上升期间有帮助的信息在峰值过后反而会产生误导，而上个季节的形态在一年内仍保持参考价值。现有的疫情图模型和通用预测器只读取短固定时间窗口，并同等对待所有历史信息，因此它们既无法利用更早季节的数据，也无法在疫情阶段发生变化时丢弃过时的关联。为了解决这些局限性，我们提出了TERN，这是一个围绕delta规则快速权重记忆构建的预测器：该记忆按通道进行衰减、沿学习到的地址进行擦除，并由局部疫情阶段特征驱动的门控机制控制，同时结合了显式的季节参考和在线适应机制。在三个Cola-GNN流感基准测试上，TERN的表现优于疫情图模型和通用预测器……

    arXiv:2609.18407v1 Announce Type: cross  Abstract: Weekly influenza surveillance counts guide vaccine distribution and public-health alerts, yet they are hard to forecast. Each region offers only a few seasons, waves shift in timing and height every year, and information that helps while a wave grows misleads after its peak, whereas last season's shape stays informative for a year. Existing epidemic graph models and general forecasters read a short fixed window and treat all past information alike, so they neither exploit earlier seasons nor discard stale associations when the epidemic phase changes. To address these limitations, we propose TERN, a forecaster built around a delta-rule fast-weight memory that decays channel-wise and erases along a learned address under gates driven by local epidemic-phase features, combined with an explicit seasonal reference and online adaptation. On three Cola-GNN influenza benchmarks, TERN outperformed epidemic graph models and general forecasters, m
    
[^249]: 坏天才：超越任务特定捷径的反事实引导测试框架演化

    Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts

    [https://arxiv.org/abs/2609.18366](https://arxiv.org/abs/2609.18366)

    提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。

    

    可靠的智能体评估因自动测试框架优化而变得复杂，这类优化方法反复使用已发布的基准 $B_{\mathrm{rel}}$ 来引导一个提议者，该提议者围绕固定的目标智能体编辑提示词、记忆、检索、工具和控制代码。任务保留集虽然改变了语义任务，但基准协议保持不变，因此一个“坏天才”提议者可以生成一个作弊的测试框架，其在发布基准上的性能提升依赖于整个基准范围的捷径。我们提出了反事实测试框架搜索与演化，将测试框架演化建模为在保持有效性的基准反事实上的约束生成问题。在每次提议者更新后，一个挑战者会搜索能大幅摧毁性能提升的可执行协议变换。有效性防火墙检查任务语义是否得到保留，而确认集则决定反事实是否进入有限存档。我们形式化定义了一个精确的捷径中和基准 $B

    arXiv:2609.18366v1 Announce Type: new  Abstract: Reliable agent evaluation is complicated by automatic harness optimization, which repeatedly uses a released benchmark $B_{\mathrm{rel}}$ to guide a Proposer that edits prompts, memory, retrieval, tools, and control code around a fixed target agent. Task holdout varies semantic tasks but leaves the benchmark protocol fixed, so a "bad genius" Proposer can produce a cheating harness whose released-benchmark gain depends on a benchmark-wide shortcut. We introduce Counterfactual Harness Search and Evolution (CHASE), which casts harness evolution as constraint generation over validity-preserving benchmark counterfactuals. After each Proposer update, a Challenger searches for an executable protocol transformation with large gain destruction. A validity firewall checks that task semantics are preserved, while a confirmation set determines whether the counterfactual enters a finite archive. We formalize an exact shortcut-neutralized benchmark $B
    
[^250]: 改进随机单调变分不等式的任意时间算法的最后迭代保证

    Improving the Last-Iterate Guarantees of Anytime Algorithms for Stochastic Monotone Variational Inequalities

    [https://arxiv.org/abs/2609.15257](https://arxiv.org/abs/2609.15257)

    本文提出了一种具有Halpern锚定的单循环单调用随机算法，将随机单调变分不等式的任意时间最后迭代收敛速率从O(t^{-1/5})提升至O(t^{-1/4})，且无需有界方差或有界可行集假设。

    

    我们分析了用于约束凸凹问题和单调变分不等式的具有Halpern锚定的随机算法。该算法是单循环单调用（single-loop and single-call）的，因为它在每次迭代中仅使用梯度算子的一个无偏样本，从而能够应用于具有噪声反馈的单调博弈。以 $t$ 表示迭代计数器，我们证明了梯度映射范数和受限间隙两项指标的任意时间最后迭代收敛速率均为 $O(t^{-1/4})$，改进了此前针对受限间隙函数所获得的最佳已知速率 $O(t^{-1/5})$。我们的速率结果涵盖了具有潜在无界可行集的约束问题，以及一类无需有界方差假设的结构化随机预言机。

    arXiv:2609.15257v1 Announce Type: cross  Abstract: We analyze a stochastic algorithm with Halpern anchoring for constrained convex-concave problems and monotone variational inequalities. This algorithm is single-loop and single-call since it uses one unbiased sample of the gradient operator at every iteration to be applicable to monotone games with noisy feedback. With $t$ denoting the iteration counter, we prove the anytime last-iterate convergence rate of $O(t^{-1/4})$ for both gradient-mapping norm and restricted gap, improving the best-known rate $O(t^{-1/5})$ that was obtained for the restricted gap function. Our rates cover constrained problems with a potentially unbounded feasible set as well as a structured class of stochastic oracles without a bounded variance.
    
[^251]: 五专家预测偏微分方程的显式解与COMB策略的精确最优集

    An explicit solution of the five-expert prediction PDE and the exact optimality set of COMB

    [https://arxiv.org/abs/2609.14892](https://arxiv.org/abs/2609.14892)

    本文首次给出五专家预测偏微分方程的显式解，并证明COMB策略仅在低维子集上最优，从而推翻了Gravin、Peres和Sivan（2016）的COMB最优性猜想。

    

    本文推导了五专家情形下“带专家建议的平稳预测”偏微分方程（PDE）的显式解。该公式分三个区域给出：在前两个区域，解为四专家解加上一个带初等正密度的单一积分；在第三个区域，解为一个双曲乘积的有限和，其系数由一次标量求积确定。我们的公式证明了方向 $(1,0,1,0,0)$ 在整个有序扇形区域内都是最优的，而COMB策略 $(1,0,1,0,1)$ 仅在该扇形区域的一个低维子集（即 $x_1=x_2$ 且 $x_3=x_4$ 处）上最优。这一结果推翻了Gravin、Peres和Sivan（2016）提出的COMB最优性猜想。哈密顿不等式的验证是一项繁琐的工作，其中部分通过计算机辅助证明完成。整个验证过程可归结为21个标量不等式，我们使用147个精确的有理Bernstein多项式对它们进行了证明。

    arXiv:2609.14892v1 Announce Type: cross  Abstract: In this paper, we derive an explicit solution of the stationary prediction with expert advice PDE for five experts. The formula is given in three regions. In the first two regions, it is the four-expert solution plus a single integral with an elementary positive density. In the third region, it is a finite sum of hyperbolic products whose coefficients are determined by one scalar quadrature. Our formula establishes that the direction $(1,0,1,0,0)$ is optimal throughout the ordered sector, and that the COMB strategy $(1,0,1,0,1)$ is optimal only on a lower dimensional subset of the sector (where $x_1=x_2$ and $x_3=x_4$). This disproves the COMB optimality conjecture of Gravin, Peres and Sivan (2016). The verification of the Hamiltonian inequalities is a tedious task, part of which is completed with a computer assisted proof. The verification reduces to 21 scalar inequalities, which we prove using 147 exact rational Bernstein polynomial 
    
[^252]: CyFM：用于少步复值流匹配的圆柱最优传输

    CyFM: Cylindrical Optimal Transport for Few-Step Complex-Valued Flow Matching

    [https://arxiv.org/abs/2609.14171](https://arxiv.org/abs/2609.14171)

    该论文提出圆柱流匹配CyFM，通过将复值信号建模在圆柱流形上并采用圆柱最优传输路径替代笛卡尔路径，严格约束了角速度回归目标，避免了笛卡尔路径导致的重尾角速度问题，从而实现高效的少步复值流匹配。

    

    复值信号，如磁共振成像（MRI）和音频频谱图，几乎总是被建模为平坦的双通道欧几里得数据。对于非零值，幅相映射 z ↦ (|z|, z/|z|) 将信号域等同于圆柱 (0, ∞) × S¹，在此之上我们有意地将继承的度量 dA² + A²dθ² 替换为解耦的乘积度量 dA² + dθ²。在这项实证研究中，我们量化了这种替换的代价与收益。通过计算精确的解析桥，我们证明了笛卡尔路径会导致角速度呈现重尾分布（幂律指数约为 1.0），在独立耦合下近半数的概率路径角速度超过 π，而圆柱路径从未超过这一速率。为解决这一问题，我们分析了圆柱流匹配，该方法严格约束了回归目标，并通过（圆柱最优传输方式）将噪声与数据耦合。

    arXiv:2609.14171v1 Announce Type: new  Abstract: Complex-valued signals, such as Magnetic Resonance Imaging (MRI) and audio spectrograms, are almost always modelled as flat two-channel Euclidean data. For nonzero values the amplitude-phase chart $z \mapsto (|z|, z/|z|)$ identifies the signal domain with the cylinder $(0, \infty) \times S^1$, on which we deliberately replace the inherited metric $dA^2 + A^2 d\theta^2$ by the decoupled product metric $dA^2 + d\theta^2$. In this empirical study we measure what that substitution costs and what it buys. By computing exact analytical bridges, we demonstrate that Cartesian paths induce a heavy-tailed distribution of angular velocity (power law index $\approx 1.0$), with nearly half of the probability paths exceeding an angular speed of $\pi$ under independent coupling, a rate no cylindrical path ever exceeds. To resolve this, we analyze Cylindrical Flow Matching (CyFM), which strictly bounds the regression target, and couple noise and data by
    
[^253]: 氛围专利撰写：评估用于专业专利起草智能体的大语言模型裁判

    Vibe Patenting: Evaluating LLM Judges for Professional Patent-Drafting Agents

    [https://arxiv.org/abs/2609.13422](https://arxiv.org/abs/2609.13422)

    该论文提出了端到端专利撰写测试平台"Vibe Patenting"，证明LLM裁判的迭代反馈能持续提升AI生成的专利草稿质量，并使低推理低成本智能体接近昂贵的高推理智能体的表现。

    

    大语言模型（LLM）裁判越来越多地被用于评估和改进AI生成的输出，但它们在复杂专业工作中的可靠性仍不明确。我们通过"Vibe Patenting"（氛围专利撰写）来研究这一问题，这是一个用于AI智能体评估的端到端专利撰写测试平台。一个单独调用的LLM裁判会对生成的专利草稿进行评估，并提供结构化反馈用于迭代修订。在多种发明和起草智能体配置下，裁判引导的修订能够持续提升裁判评估的质量，而无引导的修订则往往趋于饱和。值得注意的是，迭代的裁判反馈使低推理能力的智能体能够接近成本高昂得多的高推理智能体的性能。更强的模型和更多的推理通常会提升裁判评估的起草质量，而特定领域的智能体工作流程则能带来进一步提升。我们通过一位专业专利律师的独立评估对该裁判进行了验证，并发现……

    arXiv:2609.13422v1 Announce Type: new  Abstract: LLM judges are increasingly used to evaluate and improve AI-generated outputs, yet their reliability for complex professional work remains unclear. We study this problem through Vibe Patenting, an end-to-end patent-drafting testbed for AI-agent evaluation. A separately-invoked LLM judge evaluates generated patent drafts and provides structured feedback for iterative revision. Across multiple inventions and drafting-agent configurations, judge-guided revision consistently improves judge-assessed quality, while unguided revision tends to saturate. Notably, iterative judge feedback enables a low-reasoning agent to approach the performance of a substantially more expensive high-reasoning agent. Stronger models and increased reasoning generally improve judge-assessed drafting quality, while domain-specific agentic workflows provide further gains. We validate the judge against independent evaluation by a professional patent attorney and find m
    
[^254]: PocketVE：基于方差爆炸扩散的稳定且性质引导的结构药物设计

    PocketVE: Stable and Property-Guided Structure-Based Drug Design with Variance-Exploding Diffusion

    [https://arxiv.org/abs/2609.08101](https://arxiv.org/abs/2609.08101)

    PocketVE是一个以蛋白质口袋为条件的方差爆炸扩散框架，通过结合EDM式3D去噪、无分类器多性质引导和自适应蛋白质扰动正则化，在CrossDocked2020上显著提升了3D有效性（58.6→80.6）并大幅降低了应变能（457.4→127.9）。

    

    蛋白质条件化的3D分子生成是基于结构的药物设计中的核心挑战，需要在口袋兼容性、分子性质和物理几何之间取得平衡。我们提出了PocketVE，一个以蛋白质口袋为条件的方差爆炸（VE）扩散框架，它将稳定的坐标去噪与推理时的性质引导相结合。具体而言，PocketVE结合了用于3D去噪的EDM风格训练与采样设置、无需外部性质分类器即可实现多性质导向的无分类器引导，以及作为训练时口袋正则化器的自适应蛋白质扰动。在CrossDocked2020数据集上按照GenBench3D协议进行评估，PocketVE相对于其TAGMol架构基线，将Valid$_{3\text{D}}$从58.6提升到80.6，并将应变能从457.4降低到127.9，同时在适度引导下保持具有竞争力的对接得分和分子性质评分。一个引导尺度

    arXiv:2609.08101v1 Announce Type: cross  Abstract: Protein-conditioned 3D molecule generation is a central challenge in structure-based drug design, requiring a balance between pocket compatibility, molecular properties, and physical geometry. We propose \textbf{PocketVE}, a protein-pocket-conditioned variance-exploding (VE) diffusion framework that couples stable coordinate denoising with inference-time property guidance. Specifically, PocketVE combines an EDM-style training and sampling setup for 3D denoising, classifier-free guidance for multi-property steering without external property classifiers, and adaptive protein perturbation as a training-time pocket regularizer. Evaluated on CrossDocked2020 under the GenBench3D protocol, PocketVE improves Valid$_{3\text{D}}$ from 58.6 to 80.6 and reduces strain energy from 457.4 to 127.9 relative to its TAGMol architectural baseline, while retaining competitive docking and molecular-property scores under moderate guidance. A guidance-scale 
    
[^255]: 利用观测数据改进AI天气模型中的降水预报

    Improving precipitation forecasts in an AI weather model using observational data

    [https://arxiv.org/abs/2609.03210](https://arxiv.org/abs/2609.03210)

    通过使用观测的IMERG降水数据微调图变换器AI天气预报模型，将中期降水预报的CRPS提升最高19%，并使全球极端降雨预测的Brier技巧评分超越最先进业务化模型57%。

    

    人工智能天气预报（AIWP）系统在中期天气预报方面现已超越最先进的物理模型。当前的全球AIWP模型几乎完全使用单一的再分析数据集ERA5进行训练，但该数据集存在已知偏差，尤其是在降水方面。本文使用0.25°分辨率的IMERG降水数据对图变换器（graph-transformer）架构进行微调。所得模型将中期连续分级概率评分（CRPS）提高了多达19%，同时在热带风暴和毛毛雨事件中表现出更优的预报技能。在极端降雨预测方面，我们的模型的Brier技巧评分在全球范围内比最先进的业务化模型高出57%；不过，对于最强的降水事件，基于物理的业务化模型仍然更为可靠。我们的结果表明，将基于观测的降水数据直接纳入训练可以显著改进降水预报。

    arXiv:2609.03210v1 Announce Type: cross  Abstract: Artificial intelligence weather prediction (AIWP) systems now surpass state-of-the-art physical models for medium-range weather forecasting. Current global AIWP models are trained almost exclusively using one reanalysis dataset, ERA5, but it has known biases, particularly for precipitation. Here we fine-tune a graph-transformer architecture with IMERG precipitation data at 0.25{\deg} resolution. The resulting model improves medium-range continuous ranked probability scores by up to 19%, while also demonstrating superior skill for tropical storms and drizzle events. Our model exceeds the Brier skill score of state-of-the-art operational models on extreme rainfall prediction by 57% globally; however, a physics-based operational model remains more reliable for the heaviest precipitation events. Our results demonstrate that incorporating observations-based precipitation data directly into training can substantially improve precipitation fo
    
[^256]: 现代Transformer是隐式混合体：从功能分化到有原则的混合架构设计

    Modern Transformers Are Implicit Hybrids: From Functional Differentiation to Principled Hybrid Architecture Design

    [https://arxiv.org/abs/2609.02986](https://arxiv.org/abs/2609.02986)

    本文提出RFIS和RPD两个干预指标，发现现代Transformer的注意力头自然分化为由全局位置频带（GPBand）分隔的检索头和位置头两类，为有原则地设计FA-LA混合架构提供了实证基础。

    

    结合全注意力（FA）和线性注意力（LA）的混合架构日益受到关注，但其注意力分配方式仍然依赖于启发式规则。我们在基于RoPE的Transformer所学习到的头级别功能组织中寻求一个有证据支撑的基础。行为探针无法给出完整的分类体系，因此我们提出了两个干预指标：RoPE频率重要性分数（RFIS），衡量每个频率如何影响某个头的注意力分布；以及RoPE位置依赖性（RPD），用于分离注意力头对旋转位置调制的依赖程度。在Qwen3系列模型和Llama3.1上，RFIS给出假设且RPD加以验证，形成了一个完整的分类体系，将由显著中低频带分隔的检索头和位置头区分开来。受控Transformer实验表明，这一边界遵循训练长度的位置尺度；我们将其命名为全局位置频带。该分析揭示了零样本长度外推失败的潜在原因，并产生了

    arXiv:2609.02986v1 Announce Type: new  Abstract: Hybrid architectures combining Full Attention (FA) and Linear Attention (LA) are increasingly prominent, yet their allocation remains heuristic. We seek an evidence-grounded basis in head-level functional organization learned by RoPE-based Transformers. Behavioral probes do not yield a complete taxonomy, so we propose two intervention metrics: RoPE Frequency Importance Score (RFIS), measuring how each frequency affects a head's attention distribution, and RoPE Positional Dependence (RPD), isolating dependence on rotary positional modulation. On Qwen3-series models and Llama3.1, RFIS suggests and RPD verifies a complete taxonomy of retrieval and positional heads separated by a salient mid-low-frequency band. Controlled Transformers show that this boundary follows the training-length positional scale; we term it the Global Positional Band (GPBand). The analysis suggests a potential cause of zero-shot length-extrapolation failure and yields
    
[^257]: 在策略蒸馏真的在蒸馏吗？从噪声教师到自我改进

    Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement

    [https://arxiv.org/abs/2608.31046](https://arxiv.org/abs/2608.31046)

    研究发现在策略蒸馏中教师监督充满噪声且学生对其并不敏感，其性能提升主要来自对低概率token的学习，即使使用固定负优势信号也能达到同样效果，因此OPD本质上更接近自我改进而非真正的知识蒸馏。

    

    arXiv:2608.31046v1 公告类型：cross 摘要：在策略蒸馏（On-policy distillation, OPD）提供了密集的token级别监督，作为可验证奖励强化学习（RLVR）中稀疏结果级优势信号的替代方案。然而，教师是对学生生成的轨迹进行评分，而这些轨迹对教师而言本质上是非在策略的，因此其监督的可靠性，以及学生改进的真正来源，仍然不清楚。我们定量分析了OPD训练过程中教师监督的质量，发现其中存在大量噪声，且噪声的普遍程度随教师模型规模的增大而增加。令人惊讶的是，学生策略对这种噪声并不敏感，无论保留还是移除噪声监督，学生都收敛到相近的性能。那么OPD到底在蒸馏吗？通过分析其性能提升的驱动因素，我们发现学习主要集中在低对数概率的token上，而且使用单一固定的负优势就能达到与教师提供优势相当的性能。这表明OPD在很大程度上是……（原文在此处截断）

    arXiv:2608.31046v1 Announce Type: cross  Abstract: On-policy distillation (OPD) offers dense token-level supervision as an alternative to the sparse outcome-level advantages of reinforcement learning with verifiable rewards (RLVR). However, the teacher scores student-generated trajectories that are inherently off-policy for it, so the reliability of its supervision, and hence the source of the student's improvement, remains unclear. We quantitatively analyze teacher supervision during OPD training and find substantial noise whose prevalence increases with teacher scale. Surprisingly, the student policy is insensitive to such noise, converging to comparable performance regardless of whether noisy supervision is retained or removed. Does OPD distill at all? By analyzing what drives its gains, we find that learning concentrates on low log-probability tokens, and using a single fixed negative advantage matches the performance of teacher-provided ones. This suggests that OPD works largely b
    
[^258]: 面向文本与多媒体数据的有效图与基于排名的上下文嵌入

    Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data

    [https://arxiv.org/abs/2608.29001](https://arxiv.org/abs/2608.29001)

    本文提出RaDE方法，利用基于排名的信息和代表性节点子集选择来实现图嵌入的维度可解释性，在降低计算成本的同时改进文本与多媒体数据的检索任务。

    

    在数据驱动的世界中，高效地组织和映射对象之间的关系至关重要。图是建模这些连接关系的强大工具，被广泛应用于社交网络、电信和生物学等领域。然而，基于图的方法通常面临较高的计算成本，尤其是在内存和空间使用方面。为解决这一问题，图嵌入技术（也称为网络表示学习）将图信息编码到低维表示中，同时保留结构特性。然而，传统方法缺乏可解释的维度。RaDE（Rank Diffusion Embedding，排名扩散嵌入）引入了一种使用基于排名信息的新方法，其关键步骤是选择一个具有代表性的节点子集，从而为其维度提供可解释性并改进检索任务。尽管潜力巨大，RaDE的原始提案并未充分探索代表性子集选择的有效性。

    arXiv:2608.29001v1 Announce Type: new  Abstract: In a data-driven world, efficiently organizing and mapping relationships between objects is crucial. Graphs are powerful tools for modeling these connections, being widely used in social networks, telecommunications, and biology. However, graph-based methods often face high computational costs, particularly in memory and space usage. To address this, graph embedding techniques, also referred to as Network Representation Learning, encode graph information into lower-dimensional representations while preserving structural aspects. Traditional methods, however, lack interpretable dimensions. RaDE (Rank Diffusion Embedding) introduces a new approach using rank-based information, with a key step being the selection of a representative subset of nodes to provide interpretability for its dimensions and improve retrieval tasks. Despite its potential, RaDE's original proposal did not fully explore the effectiveness of representative subset select
    
[^259]: Aero Hand Open：一款面向灵巧操作学习的仿真就绪腱驱动灵巧手

    Aero Hand Open: A Simulation-Ready Tendon-Driven Hand for Dexterous Manipulation Learning

    [https://arxiv.org/abs/2608.28578](https://arxiv.org/abs/2608.28578)

    提出了Aero Hand Open——一款仿真就绪的腱驱动拟人灵巧手，附带可复现缆绳传动的仿真模型和双向辨识执行映射，解决了腱驱动手在灵巧操作学习中的仿真建模难题。

    

    腱驱动灵巧手具有拟人化结构，而将执行器从关节处移开，正是这类高性能手部能够以低成本制造的关键。这种成本节约来自两方面：通过缆绳传递力，使得电机无需安装在所驱动的关节内部，因此可以使用更小、更便宜的电机；同时一个电机可以通过单根缆绳驱动多个关节，从而减少所需电机的数量。然而，与直驱灵巧手相比，腱驱动手更难以用于学习。产生成本节约的欠驱动传动系统本身在仿真器中就难以建模，而且由同一根缆绳驱动的关节无法被独立控制。我们提出了Aero Hand Open，一款以仿真就绪状态发布的腱驱动拟人灵巧手。该产品附带三项内容：一个能够复现缆绳传动本身的仿真模型；一个经过辨识的执行映射，可在两个方向上将该模型与电机指令相互连接，包括（原文在此处截断）

    arXiv:2608.28578v1 Announce Type: cross  Abstract: Tendon-driven hands are anthropomorphic, and moving the actuators off the joints is what makes a hand of this capability affordable to build. Two effects produce that saving. Routing force through a cable removes the requirement that a motor fit inside the joint it drives, so smaller and cheaper motors suffice, and one motor can drive several joints through a single cable, so fewer motors are needed. They are also harder to learn on than a direct-drive hand. The underactuated transmission that produces the saving is itself difficult to represent in a simulator, and the joints one cable drives are not independently commandable. We present Aero Hand Open, a tendon-driven anthropomorphic hand that is released simulation-ready. Three things ship with it. A simulation model reproduces the cable transmission itself. An identified actuation map connects that model to the motor commands in both directions, including the three-way coupling of t
    
[^260]: J-Zero：从零数据出发的统一挑战者-求解者-评判者协同进化

    J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data

    [https://arxiv.org/abs/2608.26582](https://arxiv.org/abs/2608.26582)

    J-Zero提出了一种统一的挑战者-求解者-评判者协同进化框架，通过对抗性任务生成和基于生成方式的偏好对，实现了无需人工数据即可在可验证和不可验证领域中的自我进化。

    

    arXiv:2608.26582v1 公告类型：交叉 摘要：自我进化语言模型最近成为通往超级智能的一条有前景的路径，其优势在于减少人类监督成本。尽管在可验证领域已取得显著进展，但自我进化在不可验证领域仍研究不足。我们提出了从零数据出发的评判者协同适应（J-Zero），这是一个统一的挑战者-求解者-评判者协同进化框架，支持在两种领域中的自我改进。挑战者和求解者通过对抗性互动协同进化：挑战者生成越来越难的任务，而求解者学习产生更高质量的响应。与此同时，评判者通过使用偏好对进行协同适应，这些偏好对的顺序是预先已知的，基于每个响应的生成方式，即求解者的答案优于挑战者的答案，以及其分解再组合的答案优于其一次性答案，而非基于评判者自身的评分。

    arXiv:2608.26582v1 Announce Type: cross  Abstract: Self-evolving language models have recently emerged as a promising path toward superintelligence, with the advantage of reducing the cost of human supervision. While considerable progress has been made in verifiable domains, self-evolution in unverifiable domains remains substantially less explored. We propose Judge co-adaptation from Zero data (J-Zero), a unified Challenger--Solver--Judge co-evolution framework that supports self-improvement across both domains. The Challenger and Solver co-evolve through an adversarial interaction: the Challenger generates increasingly difficult tasks, while the Solver learns to produce higher-quality responses to them. In parallel, the Judge co-adapts using preference pairs whose ordering is known in advance from how each response was produced, i.e., the Solver's answer over the Challenger's, and its decomposed-and-recombined answer over its one-shot answer, rather than from the Judge's own scores. 
    
[^261]: 两个维度主导不可知多类转导学习

    Two Dimensions Govern Agnostic Multiclass Transductive Learning

    [https://arxiv.org/abs/2608.25326](https://arxiv.org/abs/2608.25326)

    该论文证明多类不可知转导学习的最优误差率由DS维度和Natarajan维度共同决定，公式为 $\widetilde\Theta(d_{DS}/n + \sqrt{d_{\mathrm N}/n})$，适用于任意标签空间。

    

    arXiv:2608.25326v1 公告类型：新 摘要：在转导分类中，对手固定一个带标签的总体，一个标签被均匀隐藏，学习器看到所有剩余标签。对于二分类，不可知转导学习和PAC学习具有相同的最小最大速率。这一性质是否扩展到多类学习此前尚不明确，尤其是在标签空间无界且统一收敛可能失效的情况下。我们解决了该问题，直至对数因子。对于每个具有DS维度 $d_{DS}$ 和Natarajan维度 $d_{\mathrm N}$ 的多类类 $\mathcal H$，最优不可知转导超额误差满足 $\widetilde\Theta\left(\frac{d_{DS}}{n}+\sqrt{\frac{d_{\mathrm N}}{n}}\right)$。该结果适用于任意标签空间。这两个项都是必要的。一个DS伪立方体给出可实现情况下的 $d_{DS}/n$ 障碍，而一个具有重复点和公平标签的Natarajan立方体给出不可知情况下的 $\sqrt{d_{\mathrm N}/n}$ 障碍。上界使用随机重采样技术。

    arXiv:2608.25326v1 Announce Type: new  Abstract: In transductive classification, an adversary fixes a labeled population, one label is hidden uniformly, and the learner sees all remaining labels. For binary classes, agnostic transductive and PAC learning have the same minimax rate. Whether this extends to multiclass learning was open, especially for unbounded label spaces where uniform convergence can fail. We resolve the question up to logarithmic factors. For every multiclass class $\mathcal H$ with DS dimension $d_{DS}$ and Natarajan dimension $d_{\mathrm N}$, the optimal agnostic transductive excess error satisfies $\widetilde\Theta\left(\frac{d_{DS}}{n}+\sqrt{\frac{d_{\mathrm N}}{n}}\right).$ The result holds for arbitrary label spaces. The two terms are both necessary. A DS pseudo-cube gives the realizable $d_{DS}/n$ obstruction, while a Natarajan cube with repeated points and fair labels gives the agnostic $\sqrt{d_{\mathrm N}/n}$ obstruction. The upper bound uses a random-reser
    
[^262]: 均匀稳定性的尖锐尾部

    The Sharp Tail of Uniform Stability

    [https://arxiv.org/abs/2608.24098](https://arxiv.org/abs/2608.24098)

    本文构造了一个确定性均匀稳定学习问题，证明了其泛化差距尾部下界与理论最优上界匹配，从而解决了均匀稳定性中线性对数依赖是否可实现的关键开放问题。

    

    均匀稳定性控制一个训练样本在任意测试点上能改变损失的程度。一个新的无对数上界表明，一个γ均匀稳定的算法，其损失在[0,L]范围内，以概率1-δ保证泛化差距至多为O(γlog(1/δ)+L√(log(1/δ)/n))。是否实际的有界损失学习算法能实现对log(1/δ)的线性依赖一直是未解问题。已知构造仅适用于辅助弱依赖随机变量，其逐点范围随n增长。已知的学习下界仅在常数概率下成立。我们填补了这一空白。对于每个n、稳定性水平γ和损失界L，我们构造了一个确定性的γ均匀稳定学习问题，其尾部同时满足，对于1≤p≤cn，概率P(R(A_S)-R_S(A_S)≥c'min{L,γp+L√(p/n)})。

    arXiv:2608.24098v1 Announce Type: new  Abstract: Uniform stability controls how much one training example can change the loss at any test point. A new logarithmic-free upper bound shows that a $\gamma$-uniformly stable algorithm with loss in $[0,L]$ has generalization gap at most $O \left(\gamma\log(1/\delta) +L\sqrt{\frac{\log(1/\delta)}{n}}\right)$ with probability $1-\delta$. Whether an actual bounded-loss learning algorithm can realize the linear dependence on $\log(1/\delta)$ has remained open. The known construction realizes it only for auxiliary weakly dependent random variables whose pointwise range grows with $n$. The known learning lower bound holds only at constant probability. We close this gap. For every $n$, stability level $\gamma$, and loss bound $L$, we construct one deterministic $\gamma$-uniformly stable learning problem whose tail satisfies, simultaneously for $1\le p\le c n$, $\mathbb P \left( R(A_S)-R_S(A_S) \ge c'\min \left\{L,\gamma p+L\sqrt{p/n}\right\} \right)
    
[^263]: 终端代理的可泛化行为学习

    Learning Generalizable Behaviors for Terminal Agents

    [https://arxiv.org/abs/2608.22631](https://arxiv.org/abs/2608.22631)

    本文提出“智能体组合泛化”假说，认为强化学习通过塑造高层决策行为来组合和路由预训练获得的低层技能，而非从头学习新技能，从而提升终端代理的泛化能力。

    

    终端代理是大语言模型（LLMs）的一个引人注目的应用，有望深度融入用户的日常工作中。强化学习（RL）是提升其能力的关键技术，这使得可扩展的训练环境成为核心挑战。由于公开的真实用户交互数据稀缺，合成环境提供了一种实用替代方案，但常常面临领域差距和保真度有限的问题，导致泛化性能不佳。现有工作主要侧重于扩大合成环境的数量和多样性，而奖励信号质量和泛化机制仍未得到充分探索。我们研究了RL如何改进终端代理，并提出了“智能体组合泛化”假说：与其从头教授新的领域特定技能，RL主要塑造高层决策行为，这些行为组合并路由在预训练阶段获得的低层技能。

    arXiv:2608.22631v2 Announce Type: replace  Abstract: Terminal agents are a compelling application of large language models (LLMs), with the potential to integrate deeply into users' daily workflows. Reinforcement learning (RL) is a key technique for improving their capabilities, making scalable training environments a central challenge. Since public real-user interaction data are scarce, synthetic environments provide a practical alternative, but often suffer from domain gaps and limited fidelity, leading to poor generalization. Existing work mainly scales the quantity and diversity of synthetic environments, while reward-signal quality and the mechanisms governing generalization remain under-explored. We study how RL improves terminal agents and propose the Agentic Compositional Generalization hypothesis: rather than teaching new domain-specific skills from scratch, RL primarily shapes high-level decision-making behaviors that compose and route low-level skills acquired during pre-tra
    
[^264]: 权重编码如何影响语言模型在苹果神经引擎上的部署位置与性能

    How Weight Encoding Affects Language Model Placement and Performance on the Apple Neural Engine

    [https://arxiv.org/abs/2608.22110](https://arxiv.org/abs/2608.22110)

    权重编码方式（fp16/int8/三元权重）会同时影响语言模型在苹果神经引擎上的硬件部署位置与推理性能，其中 int8 压缩可将热身前向延迟降低约 1.9 倍，但稠密编码本身并不决定模型是否使用 ANE。

    

    权重压缩会改变加速器的部署位置以及内存流量，从而使推理加速效果的解读变得复杂。我们通过公开的 Core ML 部署路径，在苹果神经引擎（ANE）上研究了这种相互作用。研究使用了五个独立训练的语言模型检查点，涵盖两种架构，以及稠密 fp16、int8 和采用两位查找表编码的三元权重。我们结合编译器设备规划、同步内存控制器测量以及计算单元排除控制，针对固定的单 token 前向工作负载进行分析。在 M1 上，较小的 fp16 导出模型尽管允许 ANE 执行，却在 CPU 上运行，而其压缩后的对应模型则表现出 ANE 活动。int8 导出模型将热身前向延迟降低了 1.9 倍。较大的 fp16 导出模型同样使用了 ANE，这表明稠密编码本身并不能决定部署位置。在 M3 上单独进行的能耗测量也支持相同方向的变化。

    arXiv:2608.22110v2 Announce Type: replace  Abstract: Weight compression can alter accelerator placement as well as memory traffic, complicating the interpretation of inference speedups. We investigate this interaction on the Apple Neural Engine through the public Core ML deployment path. Five independently trained language-model checkpoints span two architectures and dense fp16, int8, and ternary weights encoded with two-bit lookup tables. We combine compiler device plans, synchronized memory-controller measurements, and compute-unit exclusion controls for a fixed single-token forward workload. On an M1, the smaller fp16 export executes on the CPU despite permitting ANE execution, whereas its compressed counterparts exhibit ANE activity. The int8 export reduces warm forward latency by a factor of 1.9. The larger fp16 export also uses the ANE, indicating that dense encoding alone does not determine placement. Separate M3 energy measurements support the same direction of change. These re
    
[^265]: DecoVAE：一种轻量级可解释趋势-季节VAE框架，用于高效概率时间序列预测

    DecoVAE: a Lightweight Interpretable Trend-Seasonal VAE Framework for Efficient Probabilistic Time Series Forecasting

    [https://arxiv.org/abs/2608.20052](https://arxiv.org/abs/2608.20052)

    本文提出DecoVAE，一个轻量级可解释的VAE框架，通过趋势差分正则化和频域复高斯VAE显式分解时间序列，在七个基准上持续优于现有方法，同时降低内存和计算开销。

    

    概率时间序列预测仍然具有挑战性，主要因为建模不同的趋势和季节动态需要专门的方法。现有方法往往无法捕捉这些组件的独特内在属性，缺乏可解释性，或遭受沉重的内存和运行时间开销。为解决这些限制，我们提出了DecoVAE，一种轻量级可解释趋势-季节VAE框架，通过应用领域特定的归纳偏置，将时间序列显式分解为趋势和季节组件。趋势流通过潜在轨迹上的差分正则化器强制执行结构平滑性，类似于Hodrick-Prescott滤波器。同时，季节流通过复高斯VAE在频域中操作，天然捕捉周期性模式的幅度和相位。在七个真实世界基准上的广泛评估表明，DecoVAE始终优于现有方法。

    arXiv:2608.20052v1 Announce Type: new  Abstract: Probabilistic time series forecasting remains challenging, largely because modeling distinct trend and seasonal dynamics requires specialized approaches. Existing methods often fail to capture the unique inner properties of these components, lack interpretability, or suffer from heavy memory and runtime overhead. To address these limitations, we propose DecoVAE, a lightweight interpretable trend-seasonal VAE framework that explicitly decomposes time series into trend and seasonal components by applying domain-specific inductive biases. The trend stream enforces structural smoothness using a differential regularizer on the latent trajectory, analogous to the Hodrick-Prescott filter. Concurrently, the seasonal stream operates in the frequency domain via a complex Gaussian VAE, natively capturing the amplitude and phase of periodic patterns. Extensive evaluations across seven real-world benchmarks show that DecoVAE consistently outperforms 
    
[^266]: CLaST：用于概率时间序列预测的上下文感知对比VAE

    CLaST: Context-aware Contrastive VAE for Probabilistic Time Series Forecasting

    [https://arxiv.org/abs/2608.20025](https://arxiv.org/abs/2608.20025)

    CLaST通过引入上下文感知的对比损失函数，增强了VAE对时间序列内部依赖的捕捉能力，从而在概率预测中显著提升了准确率。

    

    概率预测模型广泛应用于能源系统、金融、医学和交通等领域的时间序列预测。近年来，深度生成模型在概率预测中表现出色，但许多传统方法难以捕捉内部时间依赖性，导致潜在表示的表达能力有限。为解决这一局限性，我们提出了CLaST，一种用于概率多变量时间序列预测的VAE框架。与现有生成模型不同，CLaST通过学习保留观测值之间上下文相似性的嵌入，并借助对比损失函数实现。在九个广泛采用的基准数据集上的实验表明，CLaST持续超越强基线方法。在短期预测任务中，我们的方法在CRPS上比次优方法提升了高达16.4%，在NMAE上提升了14.4%。

    arXiv:2608.20025v1 Announce Type: new  Abstract: Probabilistic forecasting models are widely used for time series forecasting in domains such as energy systems, finance, medicine, and transportation. In recent years, deep generative models have shown strong results on probabilistic forecasting, yet many conventional approaches struggle to capture internal temporal dependencies, leading to latent representations with limited expressive power. To address this limitation, we propose \textit{CLaST}, a VAE framework for probabilistic multivariate time series forecasting. Unlike existing generative models, CLaST learns embeddings that preserve contextual similarity between observations through our contrastive loss function. Experiments across nine widely adopted benchmarks demonstrate that CLaST consistently surpasses strong baseline methods. In short-term forecasting tasks, our approach achieves improvements of up to $16.4\%$ in CRPS and $14.4\%$ in NMAE over the second-best method. Further
    
[^267]: 将非最大概率映射到GMM分量对S-JEPA编码器表示重要吗？

    Does Mapping Non-Maximal Probabilities to GMM Components Matter for S-JEPA Encoder Representations?

    [https://arxiv.org/abs/2608.19084](https://arxiv.org/abs/2608.19084)

    本文通过对照实验证明，S-JEPA编码器中非最大概率分配到特定GMM分量对表示质量有显著影响，仅保留概率值而不保留映射信息会降低性能。

    

    摘要：arXiv:2608.19084v1 公告类型：新 摘要：S-JEPA使用软高斯混合模型（GMM）后验概率而非硬聚类标签来保留不确定性。目前尚不清楚仅概率值是否足够，或者哪些GMM分量接收非最大概率是否也很重要。我们通过两个匹配的对照组进行了测试。FIXED-RANDPERM保留最大分量及其概率以及非最大概率值的多重集合，但使用为每个物理帧固定的映射重新分配这些非最大值。UNIFORM-TAIL保留最大分量、其概率以及非最大总质量，但将该质量均匀分布。在三个独立随机种子下，真实软方法在两个冻结编码器读取任务上优于两个对照组。它在控制当前帧的完整频谱后，能更好地恢复原始GMM尾部，并在短时间内更好地访问光谱动态。在两个曝光实验中...

    arXiv:2608.19084v1 Announce Type: new  Abstract: S-JEPA uses soft Gaussian mixture model (GMM) posteriors instead of hard cluster labels to preserve uncertainty. It remains unclear whether the probability values alone are sufficient, or whether it also matters which GMM components receive the non-maximal probabilities. We test this with two matched controls. FIXED-RANDPERM keeps the top-1 component and probability together with the multiset of non-maximal probability values, but reassigns those non-maximal values using a mapping fixed for each physical frame. UNIFORM-TAIL keeps the top-1 component, its probability, and total non-maximal mass but distributes that mass uniformly. Across three independent seeds, REAL SOFT outperforms both controls on two frozen Encoder readouts. It provides better recovery of the original GMM tail and greater accessibility of spectral dynamics over short time scales after controlling for the complete spectrum of the current frame. In two exposure experime
    
[^268]: 持续推理健身房：诊断与利用持续RLVR中的共享推理

    Continual Reasoning Gym: Diagnosing and Harnessing Shared Reasoning in Continual RLVR

    [https://arxiv.org/abs/2608.18574](https://arxiv.org/abs/2608.18574)

    本文提出了持续推理健身房环境，发现持续RLVR的最终性能差距主要由共享推理而非遗忘主导，并探讨了如何利用共享推理来提升持续学习效果。

    

    arXiv:2608.18574v1 公告类型：新 摘要：具有可验证奖励的强化学习（RLVR）通常对多个任务进行后训练推理模型，而随着新任务的加入，重新运行多任务RLVR（MTRL）会使能力扩展成本高昂。因此，我们研究持续RLVR，即在每个任务到来时更新现有模型。核心问题是，以这种方式更新的模型是否能达到与联合训练模型相同的性能。为了回答这个问题，我们引入了持续推理健身房，这是一个持续RLVR环境，将文本和视觉推理任务组织成五个任务序列。在这种设置下，我们发现了两个关键观察：顺序RLVR表现出适度的遗忘，但其最终性能仍低于MTRL。为了理解后者，我们分解了最终性能，并表明遗忘只解释了差距的一部分。为了解释前者，我们识别了共享推理：可转移的推理结构使得在一个任务上的训练能够促进其他任务的学习。

    arXiv:2608.18574v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) commonly post-trains reasoning models on multiple tasks, while rerunning multitask RLVR (MTRL) as new tasks are added makes capability expansion costly. We therefore study continual RLVR, which updates the existing model as each task arrives. The central question is whether a model updated this way can perform as well as a jointly trained model. To answer this question, we introduce Continual Reasoning Gym, a continual-RLVR environment that organizes text and visual reasoning tasks into five task sequences. In this setting, we identify two key observations: Sequential RLVR exhibits modest forgetting, yet its final performance remains below that of MTRL. To understand the latter, we decompose final performance and show that forgetting accounts for only part of the gap. To explain the former, we identify shared reasoning: transferable reasoning structure allows training on one task to s
    
[^269]: 大型语言模型在纳米光子学中的全面综述：从代理建模到自主设计

    A Comprehensive Review of Large Language Models for Nanophotonics: From Surrogate Modeling to Autonomous Design

    [https://arxiv.org/abs/2608.18279](https://arxiv.org/abs/2608.18279)

    本文综述了大型语言模型在纳米光子学中的两种应用模式，即作为代理模型处理结构-光谱映射和实现自主设计，以突破传统深度学习在通用推理能力上的局限。

    

    超表面通过实现前所未有的光操控精度，彻底改变了光子器件的发展。然而，其设计过程常常受到计算成本高昂的模拟和复杂的高维设计空间的限制。尽管深度学习通过充当代理模型加速了设计过程，但它仍受限于特定任务的架构，并缺乏通用推理能力。本综述探讨了大型语言模型（LLMs）如何为既定的数值纳米光子学工作流程增添语义接口、代码生成和工具编排功能。我们首先概述了从经典神经网络到基于Transformer模型的发展及其在纳米光子设计中的应用。然后，我们回顾了纳米光子学中LLM相关方法的兴起，并将其组织为两种操作模式：将结构-光谱映射视为语言的代理模型，以及自主设计模式。

    arXiv:2608.18279v1 Announce Type: cross  Abstract: Metasurfaces have revolutionized the development of photonic devices by enabling unprecedented precision in light manipulation. However, their design processes are often constrained by computationally expensive simulations and complex high-dimensional design spaces. Although deep learning has accelerated the design process by serving as a surrogate model, it remains constrained by task-specific architectures and lacks universal reasoning capabilities. This review surveys how Large Language Models (LLMs) are adding semantic interfaces, code generation, and tool orchestration to established numerical nanophotonic workflows. We first outline the development from classical neural networks to transformer-based models and their applications in nanophotonic design. We then review the emergence of LLM-related methods in nanophotonics and organize them into two operational modes: surrogate models that treat structure-spectrum mapping as a langu
    
[^270]: 过于自信而不安全：用于可靠日志异常检测的模型校准

    Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection

    [https://arxiv.org/abs/2608.17965](https://arxiv.org/abs/2608.17965)

    本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。

    

    在线日志异常检测对于维护大规模计算系统的可靠性至关重要。尽管基于语言模型的日志异常检测器取得了强大的检测性能，但其置信度估计仍校准不佳。我们表明，这些检测器经常对错误预测赋予过高的置信度，尤其是在严重类别不平衡下的异常日志中。此外，即使传统校准指标显示校准良好，错误预测的置信度仍持续偏高，这为运维监控系统造成了关键可靠性缺口。为解决此问题，我们提出了日志重建与距离（LoRD），一种轻量级的事后校准框架，用于可靠的日志异常检测。LoRD从正确分类的验证样本的潜在表示中学习预测路径特定的可靠性模型，并估计预测可靠性阈值。

    arXiv:2608.17965v1 Announce Type: cross  Abstract: Online log anomaly detection is critical for maintaining the reliability of large-scale computing systems. Although recent language model-based log anomaly detectors achieve strong detection performance, their confidence estimates remain poorly calibrated. We show that these detectors frequently assign excessive confidence to incorrect predictions, particularly for anomalous logs under severe class imbalance. Moreover, confidence on erroneous predictions remains persistently high even when conventional calibration metrics indicate good calibration, creating a critical reliability gap for operational monitoring systems. To address this issue, we propose Log Reconstruction and Distance (LoRD), a lightweight post-hoc calibration framework for reliable log anomaly detection. LoRD learns prediction-route-specific reliability models from latent representations of correctly classified validation samples and estimates prediction reliability th
    
[^271]: 反射守卫：利用密集语义嵌入实现低延迟的大语言模型提示安全护栏

    Reflex-Guard: A Low-Latency Guardrail for LLM Prompt Safety Using Dense Semantic Embeddings

    [https://arxiv.org/abs/2608.17556](https://arxiv.org/abs/2608.17556)

    Reflex-Guard是一种本地运行的轻量级护栏，通过越狱感知预处理、紧凑嵌入和快速分类器，在低于100毫秒的延迟下实现高精度提示安全过滤，同时避免数据隐私风险。

    

    大语言模型（LLMs）在实际应用中经常面临精心设计的提示词试图绕过安全控制的风险。现有的护栏方法，如LLM作为评判者和基于云的安全API，能够检测不安全内容。然而，它们通常会给每个请求增加约250-900毫秒的延迟。对于需要系统在100毫秒内响应的实时应用来说，这种延迟过高。此外，将用户提示路由到外部审核端点会引发严重的数据隐私问题。本文介绍了Reflex-Guard，一种本地运行的轻量级护栏。它采用越狱感知预处理、紧凑的句子变换器嵌入和七个快速二元分类器。这些组件共同实现了高精度的提示安全过滤，且延迟远低于现有解决方案。通过对30,568个策略平衡数据集的系统评估，证明了其有效性。

    arXiv:2608.17556v1 Announce Type: cross  Abstract: Large Language Models (LLMs) in real-world applications often face the risks of specially crafted prompts designed to bypass the safety controls. Existing guardrail methods, such as LLM-as-a-judge and cloud-based safety APIs are able to detect unsafe content. However, they often add a delay of about 250-900 ms to each request. This delay is too high for real-time applications, when the system usually needs to respond in less than 100 ms. Furthermore, routing user prompts through external moderation endpoints raises significant data privacy concerns. This paper introduces Reflex-Guard, a lightweight guardrail that runs locally. It uses jailbreak-aware preprocessing, compact sentence-transformer embeddings, and seven fast binary classifiers. Together, these components enable high-accuracy prompt safety filtering with much lower latency than existing solutions. Through systematic evaluation on a strategically balanced dataset of 30,568 sa
    
[^272]: Aftab：并行化Q网络中CNN编码器与先进价值函数的综合基准

    Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks

    [https://arxiv.org/abs/2608.07335](https://arxiv.org/abs/2608.07335)

    本文系统评估了八种CNN编码器在并行化Q网络中的性能，并结合Hadamax编码与多种价值函数头，提出了一个在Atari-57上表现优异的复合架构。

    

    arXiv:2608.07335v2 公告类型：替换交叉 摘要：深度强化学习的最新进展日益倾向于简化、高度并行化的范式。值得注意的是，并行化Q网络（PQN）算法能够在无需经验回放缓冲区或目标网络的情况下进行离策略价值学习。然而，在这些无缓冲区设置中运行的视觉编码器的表示能力和计算效率仍相对未被充分探索。在本工作中，我们系统性地研究了PQN内卷积神经网络的架构设计空间。我们评估了八种不同的CNN拓扑结构，同时明确表征了它们的参数和计算需求。我们进一步通过将Hadamax编码范式与分类、集成和决斗价值头集成，研究了乘性表示学习和先进价值估计的效果。在Atari-57上的广泛实验表明，我们最终的复合架构...

    arXiv:2608.07335v2 Announce Type: replace-cross  Abstract: Recent advancements in deep reinforcement learning have increasingly favored simplified, highly parallelized paradigms. Notably, the Parallelized Q-Network (PQN) algorithm enables off-policy value learning without relying on experience replay buffers or target networks. However, the representational capacity and computational efficiency of visual encoders operating in these buffer-free settings remain comparatively underexplored. In this work, we systematically investigate the architectural design space of Convolutional Neural Networks within PQN. We evaluate eight distinct CNN topologies while explicitly characterizing their parameter and computational requirements. We further study the effect of multiplicative representation learning and advanced value estimation by integrating the Hadamax encoding paradigm with categorical, ensemble, and dueling value heads. Extensive experiments on Atari-57 show that our final composite arc
    
[^273]: 从正确示范中进行多尺度奖励对冲

    Multiscale Reward Hedging from Correct Demonstrations

    [https://arxiv.org/abs/2608.06825](https://arxiv.org/abs/2608.06825)

    该论文首次为连续奖励类给出了无视野限制的奖励对冲保证，通过在每个精度尺度上共享投票进行对冲，使累积隐藏间隙由度量熵积分界定且与轮数无关，多项式熵下可达到 O(d log A) 的总间隙和 O(d/m) 的快速收敛率。

    

    当许多答案都正确时，从正确示范中学习比监督学习更加困难：学习者在做出预测后，只能看到一个有效答案，但既不知道自己给出的答案是否有效，也得不到任何奖励。现有的奖励对冲保证因此都假设奖励类是有限的。我们给出了首个无视野限制的、适用于连续奖励类的保证。关键在于在每一个精度尺度上，通过一次共享投票对容忍最优性检验进行对冲。目标奖励在每个尺度上有一个存活的代理，而间隙超过该尺度的预测会使代理数量翻倍。由此得到同时尾部界 |{t:ℓ_t>2^{-j}}| ≤ log₂𝒩(𝒢, 2^{-j-1}) + j，其中 𝒢 是最优性间隙函数类。对尾部进行积分，得到累积隐藏间隙由度量熵积分界定，且与轮数无关。多项式熵 (A/ε)^d 给出 O(d log A) 的总间隙以及快速的 O(d/m) 收敛率。

    arXiv:2608.06825v2 Announce Type: replace  Abstract: Learning from correct demonstrations is harder than supervised learning when many answers are correct: after predicting, the learner sees one valid answer but not whether its own answer was valid, nor any reward. Existing reward-hedging guarantees consequently assume a finite reward class. We give the first horizon-free guarantee for continuous classes. The key is to hedge in one shared vote over tolerant optimality tests at every accuracy scale. A target reward has one surviving proxy per scale, and a prediction with gap above that scale doubles the proxy. This yields the simultaneous tail bound $|\{t:\ell_t>2^{-j}\}|\leq \log_2\mathcal N(\mathcal G,2^{-j-1})+j$, where $\mathcal G$ is the class of optimality-gap functions. Integrating the tails gives cumulative hidden gap bounded by a metric-entropy integral, independently of the number of rounds. Polynomial entropy $(A/\epsilon)^d$ gives $O(d\log A)$ total gap and a fast $O(d/m)$ s
    
[^274]: 面向AI代理签名工作流的硬件密钥存储：一种零信任MCP强制执行架构

    Hardware Keystores for AI Agent Signing Workflows: A Zero-Trust MCP Enforcement Architecture

    [https://arxiv.org/abs/2608.06130](https://arxiv.org/abs/2608.06130)

    论文针对AI代理签名场景中的“困惑代理人”问题，提出了一种结合硬件密钥存储的五层零信任强制执行架构，确保只有符合操作者已确认意图的请求才能触发硬件签名，从而防御提示注入攻击导致的私钥滥用。

    

    AI代理越来越多地代表其操作者签署Git提交、认证文档并证明发布工件，其所使用的私钥保存在软件可访问的位置（明文文件、环境变量、容器内存）中，任何该代理可触达的进程都能读取这些密钥。一个广泛部署的代理框架最近就因一次电子邮件注入以这种方式泄露了其密钥。硬件密钥存储（HSM、TPM、智能卡）将密钥保存在设备上，但将密钥存储暴露为LLM代理可调用的工具只是转移了问题而非消除它：一旦签名会话建立，硬件无法区分反映操作者意图的请求与被注入到代理所读取内容中的请求。我们刻画了这一“困惑代理人”（confused-deputy）问题，并构建了其所需要的五层零信任强制执行栈，使只有与操作者已确认意图一致的请求才能到达硬件。我们在两个攻击面上进行了评估。提示注入……

    arXiv:2608.06130v2 Announce Type: replace-cross  Abstract: AI agents increasingly sign Git commits, certify documents, and attest release artifacts on behalf of their operators, using private keys that live in software-accessible locations (plaintext files, environment variables, container memory) readable by any process the agent can reach. A widely deployed agent framework recently leaked its keys this way to a single email injection. Hardware keystores (HSM, TPM, smart card) keep the key on-device, but exposing the keystore as a tool an LLM agent can call moves the problem rather than removing it: once a signing session exists, the hardware cannot tell a request reflecting the operator's intent from one injected into content the agent read. We characterize this confused-deputy problem and build the five-layer Zero-Trust enforcement stack it requires, so that only requests consistent with the operator's committed intent reach the hardware. We evaluate on two attack planes. Prompt inj
    
[^275]: Trident：如何攻破深度强化学习网络防御（智能体）

    Trident : How to Break Deep Reinforcement Learning Cyber Defenses (Agentic)

    [https://arxiv.org/abs/2608.04317](https://arxiv.org/abs/2608.04317)

    该论文提出Trident框架，通过动态沙箱基准环境、超过1.3万条红蓝对抗交互轨迹数据集以及“代码即策略”的RLVR智能体架构，训练LLM红队智能体自适应地发起攻击，从而揭示深度强化学习网络防御系统面对自适应威胁时的脆弱性。

    

    基于深度强化学习（DRL）的自主网络防御系统引起了广泛的研究关注，但其评估几乎完全针对静态的、启发式的红方智能体，这使得这类系统在面对自适应威胁时的鲁棒性严重缺乏研究。与此同时，基于可验证奖励的强化学习（RLVR）的最新进展提升了大语言模型（LLM）的推理能力，但由于缺乏合适的基准环境和交互数据集，其与网络安全的结合仍难以实现。为了弥合这一差距，我们提出了Trident，一个基于智能体LLM的红队测试框架，包含三个组件：一个跨越CybORG CAGE 4和CyberWheel、配备隔离沙箱服务器的动态基准环境；一个包含超过13,000条高保真红蓝对抗交互轨迹、用于RLVR的数据集；以及一个“代码即策略”（Code-as-Policy）的RLVR智能体架构（Trident Agentic）。后者将红方智能体的训练重新表述为一种上下文相关的……

    arXiv:2608.04317v2 Announce Type: replace-cross  Abstract: Autonomous cyber defense systems based on Deep Reinforcement Learning (DRL) have attracted significant research attention, yet remain evaluated almost exclusively against static, heuristic red agents, leaving their robustness against adaptive threats critically understudied. Meanwhile, recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have improved LLM reasoning, but their integration into cybersecurity remains elusive due to the absence of suitable benchmark environments and interaction datasets. To bridge this gap, we introduce Trident, an agentic LLM red teaming framework comprising three components: a dynamic benchmark with isolated sandbox servers spanning CybORG CAGE 4 and CyberWheel, a dataset comprises over 13,000 high-fidelity red-blue interaction trajectories for RLVR, and a ``Code-as-Policy'' RLVR agentic architecture Trident Agentic). The latter reformulates red agent training as a contextual 
    
[^276]: 布线优于混合：不同Transformer规模之间传递了什么——以及什么没有传递

    Wiring Beats Blending: What Transfers Between Transformer Sizes -- and What Doesn't

    [https://arxiv.org/abs/2608.02829](https://arxiv.org/abs/2608.02829)

    本文发现，在不同规模的Transformer模型间转换时，表示对齐强但参数对齐弱，价值在于初始化，并通过最小二乘补偿和方差保持重缩放两个杠杆实现有效转换。

    

    arXiv:2608.02829v3 公告类型：替换交叉 摘要：模型家族通常按规模逐个从头训练。能否将预训练的大模型转换为较小的兄弟模型？我们端到端地表征了Pythia中的1.4B->410M转换。表示在不同规模间强烈对齐（岭回归R^2=0.84），而参数对齐较弱。密集权重投影在功能上具有破坏性，且一个比特精确的控制表明这不是组装伪影：基混合破坏了旋转、每头、GELU和LayerNorm结构。在最佳拟合线性算子之后，权重残差在洗牌控制下在统计上与噪声无异。因此，转换价值存在于初始化中。在匹配预算的持续预训练中，我们将转换分解为两个独立杠杆：最小二乘补偿（功能杠杆，最佳零样本）和方差保持重缩放（动力学杠杆，最佳终点）。补偿是一种令牌高效、低预算的胜利，而非其他。

    arXiv:2608.02829v3 Announce Type: replace-cross  Abstract: Model families are typically trained size by size, each from scratch. Can a pretrained large model instead be converted into a smaller sibling? We characterize the 1.4B->410M conversion in Pythia end to end. Representations align strongly across sizes (ridge R^2=0.84) while parameters align weakly. Dense weight projection is functionally destructive, and a bit-exact control shows this is not an assembly artifact: basis mixing breaks rotary, per-head, GELU, and LayerNorm structure. After the best-fit linear operator, weight residuals are statistically indistinguishable from noise under shuffle controls. Conversion value therefore lives in initialization. In matched-budget continued pre-training we decompose conversion into two independent levers: least-squares compensation (function lever, best zero-shot) and variance-preserving rescale (dynamics lever, best endpoints). Compensation is a token-efficient, low-budget win rather th
    
[^277]: DGM和PINN算法求解非线性偏微分方程的全局收敛性

    Global Convergence of DGM and PINN Algorithms for Solving Nonlinear PDEs

    [https://arxiv.org/abs/2607.24726](https://arxiv.org/abs/2607.24726)

    本文证明了DGM和PINN算法在求解一类半线性非线性PDE时，其训练过程能全局收敛到PDE的真实解，解决了非凸优化下算法数学基础的长期问题。

    

    arXiv:2607.24726v2 公告类型：替换 摘要：深度伽辽金方法（DGM）和物理信息神经网络（PINNs）已成为快速发展的科学机器学习领域中广泛用于求解偏微分方程（PDE）的方法。在这些方法中，通过使用（随机）梯度下降来最小化神经网络的PDE残差，训练神经网络以近似PDE解。由于PDE残差目标函数的非凸性，训练后的神经网络原则上可能仅收敛到目标函数的局部最小值（这不会成为PDE的解）。因此，关于这些算法的数学基础存在一个长期问题，而建立训练后的神经网络将收敛到PDE解这一性质具有高度价值。在本文中，我们考虑一类半线性PDE，其非线性和一阶导数均包含在内。对于此类PDE，我们...

    arXiv:2607.24726v2 Announce Type: replace  Abstract: The Deep Galerkin Method (DGM) and Physics Informed Neural Networks (PINNs) have become widely-used methods for solving partial differential equations (PDEs) in the rapidly growing field of scientific machine learning. In these methods, a neural network is trained to approximate the PDE solution by using (stochastic) gradient descent to minimize the PDE residual of the neural network. Due to the non-convexity of the PDE residual objective function, the trained neural network may, in principle, only converge to a local minimizer of the objective function (which would not be a solution of the PDE). Therefore, there is a longstanding question regarding the mathematical foundations of these algorithms, and it is highly valuable to establish that the trained neural network will converge to the PDE solution. In this paper, we consider a class of semilinear PDEs with nonlinearities in the solution and its first derivative. For this class of
    
[^278]: 时变边界条件下流动代理模型的“免费午餐不存在”：一项双工况研究

    No Free Lunch in Flow Surrogates under Time-Varying Boundary Conditions: A Two-Regime Study

    [https://arxiv.org/abs/2607.23667](https://arxiv.org/abs/2607.23667)

    在时变边界条件下不存在“万能”的流动代理模型架构：一次性全场模型最适合同步辐射CMP浆料薄膜流动（误差2.7%），而只有潜在自回归DeepONet能保留卡门涡街90%的脱涡功率，表明对时间维度的处理方式是决定模型成败的关键。

    

    我们测试了在简单流动工况上取得成功的架构是否也能在更复杂的工况上同样成功，其中每个模型分别在各工况上单独训练。我们研究了两种时变边界条件下的瞬态流动：半导体制造核心工艺化学机械平坦化（CMP）中的三维浆料薄膜流动，以及二维卡门涡街（KVS）。八个代理模型运行在同一共享流水线上，它们的区别在于学习的是全场还是潜在表示，以及采用一次性预测还是逐步预测。没有哪种单一架构能同时在两种工况中胜出。在薄膜流动上，一次性全场模型重构累积壁面剪切应力的相对误差仅为2.7%。在尾流上，潜在自回归DeepONet保留了90%的脱涡功率，而直接模型和一次性模型则将其衰减至几乎为零。对时间的处理方式决定了结果。自持的尾流需要自回归……

    arXiv:2607.23667v2 Announce Type: replace-cross  Abstract: We test whether an architecture that succeeds on a simple flow regime also succeeds on a richer one, with each trained separately on each regime. We explore two transient flows under time-varying boundary conditions: the three-dimensional slurry film in chemical-mechanical planarisation (CMP), central to semiconductor manufacturing, and the two-dimensional K\'arm\'an vortex street (KVS). Eight surrogate models on one shared pipeline differ in whether they learn the full field or a latent representation, and in whether they predict in one shot or step by step. No single architecture wins both regimes. On the film, a one-shot full-field model reconstructs the cumulative wall shear stress to 2.7% relative error. On the wake, a latent autoregressive DeepONet retains 90% of the shedding power that direct and one-shot models damp to almost zero. The treatment of time decides the outcome. The self-sustained wake calls for autoregressi
    
[^279]: 一种用于旋转机械物理可验证故障诊断的多层次信息集成框架

    A Multi-level Information Integration Framework for Physically Verifiable Fault Diagnosis of Rotating Machinery

    [https://arxiv.org/abs/2607.22797](https://arxiv.org/abs/2607.22797)

    提出一种与编码器无关的多任务框架——诊断证据网络（DENet），将旋转机械故障诊断输出扩展为包含分类结果、可对照轴承几何与转速理论值验证的预测特征频率以及时间定位信息的结构化证据记录，从而实现物理可验证的故障诊断。

    

    将从物理模型到数据驱动诊断再到自然语言推理的多层次信息整合为可验证的决策链，是智能制造领域日益增长的需求。本文以轴承故障诊断作为代表性测试平台，指出传统方法的标准输出仅为类别标签和由分类器自身分布导出的置信度分数，缺乏与独立物理知识进行对照比较的手段；同时，越来越多用于维护沟通的语言模型可能引入缺乏依据的内容。本工作从输出端解决上述两个局限。所提出的诊断证据网络（DENet）是一种与编码器无关的多任务框架，其将输出扩展为一条结构化的证据记录：包括分类结果、可与由轴承几何尺寸和轴速确定的理论值进行对比验证的预测特征频率，以及时间定位信息（摘要在此处被截断，后续内容未提供）。

    arXiv:2607.22797v2 Announce Type: replace-cross  Abstract: Integrating multi-level information, from physical models through data-driven diagnostics to natural language reasoning, into verifiable decision chains is a growing need in intelligent manufacturing. In bearing fault diagnosis, taken here as a representative testbed, the standard output is a class label and a confidence score derived from the classifier's own distribution, offering limited means of comparison against independent physical knowledge. Meanwhile, language models increasingly used for maintenance communication may introduce unsupported content. This work addresses both limitations from the output side. The proposed Diagnostic Evidence Network (DENet) is an encoder-agnostic multi-task framework that extends the output to a structured evidence record: the classification, a predicted characteristic frequency comparable against the theoretical value determined by bearing geometry and shaft speed, and a temporal localiz
    
[^280]: SechKAN：基于双曲正割函数的Kolmogorov-Arnold网络

    SechKAN: Kolmogorov-Arnold Networks with Hyperbolic Secant Functions

    [https://arxiv.org/abs/2607.18290](https://arxiv.org/abs/2607.18290)

    本文提出了基于双曲正割函数的新型Kolmogorov-Arnold网络SechKAN，通过一维线性投影控制参数规模，在函数拟合、PDE代理建模和图像分类任务上取得了与MLP及现有KAN变体相当或更优的性能。

    

    近年来，Kolmogorov-Arnold网络（KANs）因其在机器学习和科学计算中的有效性而受到越来越多的关注，为神经网络设计提供了一种新的范式。本文提出了SechKAN，一种基于双曲正割函数的新型KAN。选择双曲正割基函数是因为其平滑的钟形曲线、局部化响应和良好的梯度特性。我们采用一维线性投影来减少参数数量，使SechKAN能够保持与多层感知器（MLPs）相当的模型规模。实验结果表明，SechKAN在函数拟合、偏微分方程（PDE）代理建模和图像分类基准测试（包括MNIST、FashionMNIST、CIFAR10和CIFAR100）上均表现出有效性。在函数拟合任务上，SechKAN达到了与MLPs及代表性KAN变体相当的性能。在PDE代理建模上，它优于MLPs，并取得了具有竞争力或更好的表现。

    arXiv:2607.18290v3 Announce Type: replace-cross  Abstract: In recent years KolmogorovArnold Networks KANs have attracted increasing attention due to their effectiveness in machine learning and scientific computing offering a new paradigm for neural network design In this paper we present SechKAN a novel KAN based on hyperbolic secant sech functions The hyperbolic secant basis is adopted for its smooth bellshaped form localized responses and wellbehaved gradients We employ a 1D linear projection to reduce the number of parameters allowing SechKAN to maintain a model size comparable to that of multilayer perceptrons MLPs Experimental results show the effectiveness of SechKAN on function fitting PDE surrogate modeling and image classification benchmarks including MNIST FashionMNIST CIFAR10 and CIFAR100 On function fitting SechKAN achieves performance comparable to both MLPs and representative KAN variants On PDE surrogate modeling it outperforms MLPs and achieves competitive or better per
    
[^281]: 一种针对KV缓存的JoLT方法：通过Tucker秩的联合拉格朗日分配和旋转残差实现大语言模型的近无损KV缓存压缩

    A JoLT for the KV cache: Near-lossless KV cache compression via joint Lagrangian allocation of Tucker ranks and a rotated residual for llms

    [https://arxiv.org/abs/2607.12550](https://arxiv.org/abs/2607.12550)

    本文提出JoLT方法，通过部分Tucker分解和旋转低比特残差，在保持头与层轴完整的同时压缩令牌和特征轴，实现KV缓存的近无损压缩。

    

    键值（KV）缓存已成为Transformer推理中的主要内存开销：它随批次大小、上下文长度和深度增长，在长上下文场景下，它而非模型权重决定了吞吐量的上限。现有的压缩方法分为两类。低秩方法对缓存的二维切片进行分解，可以是每个头的矩阵或跨层的特征块，而量化方法则降低每个条目的位宽。这两类方法都没有利用缓存在一层中天然是三阶张量这一事实，其三个轴——头、令牌和特征——携带的冗余量差异很大。我们直接采用这种张量视图。我们的方法JoLT（联合拉格朗日Tucker）应用部分Tucker分解，仅压缩令牌和特征轴，同时保留头和层轴不变，然后通过旋转的低位残差恢复截断所丢弃的能量：一个随机或...

    arXiv:2607.12550v3 Announce Type: replace-cross  Abstract: The key-value (KV) cache has become the dominant memory cost of transformer inference: it grows with batch size, context length, and depth, and at long context it, rather than the model weights, sets the throughput ceiling. Existing reductions fall into two families. Low-rank methods factor two-dimensional slices of the cache, either per-head matrices or cross-layer feature blocks, and quantization methods lower the bit-width of every entry. Neither exploits the fact that the cache at a layer is naturally a third-order tensor whose three axes, the heads, the tokens, and the features, carry very different amounts of redundancy. We take this tensor view directly. Our method, JoLT (Joint Lagrangian Tucker), applies a partial Tucker decomposition that compresses only the token and feature axes while leaving the head and layer axes intact, then restores the energy that truncation discards with a rotated low-bit residual: a random or
    
[^282]: 一种用于非均质地下环境中地热换热器地下温度场模拟的解析-PINN混合模型

    A hybrid analytical-PINN model for subsurface simulation of geothermal heat exchangers in heterogeneous underground

    [https://arxiv.org/abs/2607.12271](https://arxiv.org/abs/2607.12271)

    该研究提出一种解析与物理信息神经网络（PINN）混合的参数化框架，通过解析提取奇异线源响应并利用钻孔中心相对坐标下的叠加原理复用神经修正，实现了非均质地下环境中多钻孔地热换热器长期温度场的高效通用模拟。

    

    准确且高效地预测地下温度场对于钻孔换热器（BHE）系统的设计与运行至关重要。本文开发了一种参数化的解析-物理信息神经网络（PINN）混合框架，用于非均质地下环境中多钻孔换热器的长期模拟。该方法通过解析方式提取奇异线源响应，从而能够有效训练与地下非均质性相关的神经修正网络。通过对热导率进行显式参数化，单个前馈神经网络的物理信息学习即可泛化至不同的地下条件。通过在以钻孔为中心的相对坐标系中构造该修正项，所学到的修正可借助空间和时间叠加原理作为通用修正器被重复使用。基于无限线源（ILS）、有限线源（FLS）和m…的数值实验（摘要在此处截断）。

    arXiv:2607.12271v2 Announce Type: replace  Abstract: Accurate and efficient prediction of subsurface temperature fields is essential for the design and operation of borehole heat exchanger (BHE) systems. Here we develop a parametric hybrid analytical and physics-informed neural network (PINN) framework for long-term multi-BHE simulations in heterogeneous underground. The method analytically extracts the singular line source response and enables the effective training of neural correction associated with subsurface heterogeneity. An explicit parametrization of the thermal conductivity allows physics-informed learning of a single feedforward neural network to generalize across different subsurface conditions. By formulating the correction in borehole-centered relative coordinates, the learned correction can be reused as a universal corrector through spatial and temporal superposition principles. Numerical experiments based on the infinite line source (ILS), finite line source (FLS) and m
    
[^283]: 谁拥有AI的推荐？跨大语言模型品牌类别归属的多行业实证图谱

    Who Owns the AI Recommendation? A Multi-Industry Empirical Map of Brand Category Ownership Across Large Language Models

    [https://arxiv.org/abs/2606.23057](https://arxiv.org/abs/2606.23057)

    该研究通过对五个行业50个品牌、250个查询在三个大语言模型上跨越两个月的大规模实证测量，首次绘制了AI推荐中品牌类别归属的图谱，发现品牌收录率较为均衡、推荐份额随时间高度稳定，并识别出7.6%的“竞争真空”查询。

    

    这项探索性研究测量了五个行业、50个品牌、250个查询中的品牌收录情况，于2026年2月和9月各将每个查询五次提交给GPT-5.2、Gemini 3 Flash和Perplexity sonar-pro（分别获得3,614和3,750条评分答案）。类别收录率、推荐份额、竞争真空指数和共同提及不对称性等指标均有明确的分母定义。2月份同一行业内被采样品牌的收录率较为接近（平均基尼系数为0.30），而在250个查询中有204个查询的答案中至少80%提及了至少一个品牌。竞争真空出现在7.6%的查询中；初步的开放词汇模型解读表明，大多数真空现象反映了所采样的品牌列表本身。部分预先设定的9月复制实验显示，跨日期的推荐份额具有很强的相关性（Spearman 0.994），真空现象的普遍程度保持不变，一致性为60.8%（2月为57.2%）。这种描述性的规模关联持续存在。固定边际并不能解释……

    arXiv:2606.23057v2 Announce Type: replace-cross  Abstract: This exploratory study measures brand inclusion across five industries, 50 brands and 250 queries, each put five times to GPT-5.2, Gemini 3 Flash and Perplexity sonar-pro in February and September 2026 (3,614 and 3,750 scored answers). Category Inclusion Rate, Recommendation Share, Competitive Vacuum Index and Co-Mention Asymmetry have stated denominators. February inclusion rates sit close together across an industry's sampled brands (mean Gini 0.30), while at least one brand is named in 80% or more of answers to 204 of 250 queries. Vacuums occur in 7.6% of queries; provisional open-vocabulary model readings suggest most reflect the sampled brand list. The partially pre-specified September replication shows strong cross-date Recommendation Share correlation (Spearman 0.994), unchanged vacuum prevalence and agreement of 60.8% against February's 57.2%. The descriptive size association persists. Fixed margins do not account for a
    
[^284]: CaliPPer：量化、预测并提升结合预测任务中AI模型的性能

    CaliPPer: quantifying, predicting and improving AI model performance for binding prediction

    [https://arxiv.org/abs/2606.07258](https://arxiv.org/abs/2606.07258)

    CaliPPer是一个事后校准框架，通过多链样本-域距离与距离感知贝叶斯重校准，在无标签条件下实现免疫受体结合预测AI模型性能的量化、预测与提升。

    

    结合预测模型能够加速治疗性抗体和TCR的发现，但它们在新数据集上的表现难以预料，常常导致较低的发现率。密度比方法（如PAPE、M-CBPE）虽然可以在无标签条件下为二分类任务提供性能估计，但其假设以及仅输出聚合结果的特点，限制了它们在新生抗原表位、抗原变异体和化学骨架等结合预测场景中的应用。本文提出CaliPPer（Calibration and Prediction of Performance，校准与性能预测），这是一个事后框架，将多链的“样本到域距离”与距离感知的贝叶斯重校准相结合，并在三个层次上运作：泛化性评分、聚合性能预测以及逐样本置信度。在十个模型、八种架构和两个免疫受体领域上，CaliPPer实现了距离与性能之间|r|=0.80–0.92的相关性，预测AUROC/AP/F1的平均绝对误差仅为0.008–0.070，并显著提升了AUROC（此处摘要截断）。

    arXiv:2606.07258v1 Announce Type: cross  Abstract: Binding prediction models accelerate therapeutic antibody and TCR discovery, but their performance on new datasets is unpredictable, often leading to low discovery rates. Density-ratio methods (PAPE, M-CBPE) provide label-free performance estimation for binary classification, but their assumptions and aggregate-only outputs limit binding prediction on neoepitopes, antigen variants and chemical scaffolds. Here we present CaliPPer (Calibration and Prediction of Performance), a post-hoc framework pairing a multi-chain Sample-to-Domain Distance (S2DD) with distance-aware Bayesian recalibration, operating at three resolutions: generalisability score, aggregate performance prediction, and per-sample confidence. Across ten models, eight architectures and two immune-receptor domains, CaliPPer attains distance--performance correlations $|r|=0.80\text{--}0.92$, predicts AUROC/AP/F1 with mean absolute errors $0.008\text{--}0.070$, and improves AU
    
[^285]: DiffUNet²：用于概率性科学时空建模的双向条件扩散模型

    DiffUNet^2: Bidirectional Conditional Diffusion for Probabilistic Scientific Spatiotemporal Modeling

    [https://arxiv.org/abs/2606.03926](https://arxiv.org/abs/2606.03926)

    DiffUNet² 是一种双向条件扩散模型，能在单一共享模型中同时完成科学时空数据的正向预测和反向概率推断，在流体力学、化学反应动力学和材料变形等多个科学数据集上均展现出强大的双向预测能力和高质量的概率集成效果。

    

    研究科学现象的时空演化通常依赖于昂贵的模拟和实验。基于机器学习的代理模型可以降低这一成本，但大多数仅限于确定性的正向预测。科学时间分析通常既需要正向预测也需要反向推断，而时间演化并不总是唯一确定的，尤其是对于逆问题。我们提出了 DiffUNet²，一种用于概率性科学时间预测的双向条件扩散模型，它在共享模型内同时支持正向和反向预测。我们在涵盖流体力学、化学反应动力学和材料变形的四个科学时间数据集上，将 DiffUNet² 与确定性和概率性基线模型进行了对比评估。结果表明，DiffUNet² 在两个时间方向上都取得了强大的预测性能，并具有高质量的概率集成效果。

    arXiv:2606.03926v2 Announce Type: replace-cross  Abstract: Studying the spatiotemporal evolution of scientific phenomena often relies on costly simulations and experiments. Machine learning-based surrogate models reduce this cost, but most are limited to deterministic forward prediction. Scientific temporal analysis often requires both forward prediction and backward inference, while temporal evolution is not always uniquely determined, especially for the inverse problem. We introduce DiffUNet^2, a bidirectional conditional diffusion model for probabilistic scientific temporal prediction. It supports both forward and backward prediction within a shared model. We evaluate DiffUNet^2 on four scientific temporal datasets spanning fluid dynamics, chemical reaction dynamics, and material deformation, against deterministic and probabilistic baselines. Results show that DiffUNet^2 achieves strong predictive performance in both temporal directions and high probabilistic ensemble quality compar
    
[^286]: 动态医疗治疗强化学习的统一基准

    A Unified Benchmark for Dynamic Medical Treatment Reinforcement Learning

    [https://arxiv.org/abs/2606.01028](https://arxiv.org/abs/2606.01028)

    该论文提出了MedGym——一个基于物理信息神经网络、在连续时间框架下由临床数据构建的可配置动态治疗推荐强化学习基准，用于评估强化学习方法处理不规则测量间隔、个体化治疗反应和测量点间安全性的能力。

    

    医疗治疗推荐给强化学习带来了诸多挑战：患者生理状态在连续时间中演变，测量与干预以不规则的时间间隔进行，且治疗效果在个体间存在显著差异。然而，现有的强化学习建模方式和模拟环境均基于具有固定决策间隔的离散时间马尔可夫决策过程（MDP）。因此，评估强化学习方法能否处理与时间间隔相关的疾病进展、个性化治疗反应以及连续测量点之间的安全性问题仍然十分困难。为填补这一空白，我们提出了MedGym，一个面向动态治疗推荐的基准环境。MedGym在连续时间框架下对患者的纵向演化进行建模，并利用物理信息神经网络从临床数据构建了一个可配置的医疗强化学习基准。由此得到的基准使得离散…（摘要在此处截断）

    arXiv:2606.01028v2 Announce Type: replace  Abstract: Medical treatment recommendation poses several challenges to reinforcement learning (RL): patient physiology evolves in continuous time, measurements and interventions are performed at irregular intervals, and treatment effects vary substantially across individuals. Existing RL formulations and simulated environments, however, are based on discrete-time MDPs with fixed decision intervals. Thus, it remains difficult to evaluate whether RL methods can handle time-interval-dependent disease progression, personalized treatment response, and safety between consecutive measurement points. To address this gap, we introduce MedGym, a benchmark environment for dynamic treatment recommendation. MedGym models longitudinal patient evolution in a continuous-time framework and constructs a configurable medical RL benchmark from clinical data by using Physics-Informed Neural Networks. The resulting benchmark enables direct comparison between discre
    
[^287]: 一种用于光片荧光显微镜的多模态3D基础模型，实现少样本分割、分类与去模糊

    A Multimodal 3D Foundation Model for Light Sheet Fluorescence Microscopy Enables Few-Shot Segmentation, Classification, and Deblurring

    [https://arxiv.org/abs/2605.26026](https://arxiv.org/abs/2605.26026)

    该论文提出了一个针对光片荧光显微镜数据的3D基础模型，通过在大规模多样化3D图像上联合优化掩码重建与图像-文本对齐进行预训练，学习可迁移的体积表征，大幅降低标注负担，并实现少样本分割、分类与去模糊。

    

    光片荧光显微镜（LSM）能够对生物样本进行高分辨率的三维（3D）成像，为研究细胞组织结构、病理学和血管网络提供丰富的体积数据。然而，LSM数据的规模、维度和标注负担使得有监督深度学习方法成本高昂且难以扩展。此外，尽管未标注的LSM体积数据十分丰富，但由于计算上的挑战以及体积表征学习的复杂性，针对这一模态的基础模型仍然探索不足。在本工作中，我们为LSM数据引入了一个3D基础模型，该模型在跨越多种生物体、染色方法和成像协议的大型精选3D图像集合上进行了预训练。我们通过联合优化掩码重建和图像-文本对齐来学习可迁移的体积表征。预训练的骨干网络极大地降低了标注负担。

    arXiv:2605.26026v2 Announce Type: replace-cross  Abstract: Light sheet fluorescence microscopy (LSM) enables high-resolution, three-dimensional (3D) imaging of biological specimens, providing rich volumetric data for studying cellular organization, pathology, and vascular networks. However, the size, dimensionality, and annotation burden of LSM data make supervised deep learning approaches costly and difficult to scale. Additionally, despite the abundance of unannotated LSM volumes, foundation models for this modality remain underexplored due to computational challenges and the complexity of volumetric representation learning. In this work, we introduce a 3D foundation model for LSM data, pretrained on a large curated collection of 3D images spanning multiple organisms, stains, and imaging protocols. We learn transferable volumetric representations by jointly optimizing for masked reconstruction and image-text alignment. The pretrained backbone drastically reduces the annotation burden
    
[^288]: DeGRe：面向推荐的密集监督生成式重排序

    DeGRe: Dense-supervised Generative Reranking for Recommendation

    [https://arxiv.org/abs/2605.25749](https://arxiv.org/abs/2605.25749)

    该论文提出DeGRe框架，通过引入密集监督信号来解决生成式重排序中的启发式标签偏差和稀疏奖励导致的信用分配难题。

    

    在多阶段推荐系统中，重排序通过捕捉列表内部的上下文依赖关系来优化整体效用，但其核心挑战在于如何在指数级庞大的排列空间中探索最优序列。近期研究已转向端到端生成式框架，这类方法通常利用列表级奖励或偏好对齐来指导生成器的训练。然而，这些方法仍面临两个关键问题。第一是启发式标签偏差：现有方法往往基于简单规则构建训练目标，例如将用户点击过的物品提升至列表顶部，而忽略了列表上下文中的因果依赖关系。第二是信用分配问题：稀疏的列表级事后奖励无法直接指导序列生成过程中的中间步骤，导致优化方向模糊不清。为解决这些问题，我们提出了DeGRe（密集监督生成式重排序），一种……

    arXiv:2605.25749v2 Announce Type: replace-cross  Abstract: In multi-stage recommender systems, reranking optimizes overall utility by capturing intra-list contextual dependencies, yet its central challenge lies in exploring optimal sequences within an exponentially large permutation space. Recent studies have shifted towards end-to-end generative frameworks, which typically leverage list-wise rewards or preference alignment to guide generator training. However, these methods still face two critical issues. First is the heuristic label bias. Existing methods often construct training targets based on simple rules, such as promoting clicked items to the top, while ignoring causal dependencies within the list context. Second is the credit assignment problem. Sparse list-level posterior rewards fail to directly guide intermediate steps in sequence generation, leading to ambiguous optimization directions.   To address these issues, we propose DeGRe (Dense-supervised Generative Reranking), a 
    
[^289]: 每个组件都是一次查找：用一张线性图统一组件交互、组合与归因

    Every Component Is a Lookup: One Linear Graph for Interaction, Composition and Attribution

    [https://arxiv.org/abs/2605.23393](https://arxiv.org/abs/2605.23393)

    本文提出基于“键值形式”与“加性残差流”两个架构假设，将 Transformer 统一表示为一张线性计算图，并通过 Unpack 反向归因方法，使组件交互、组合路径与词元归因成为同一张图的不同读出结果。

    

    Transformer 的可解释性方法通常围绕相互独立的问题构建：哪些组件发生交互、信息如何路由到输出、以及哪些输入词元做出了贡献。由于这些方法依赖不同的假设，它们的答案难以相互关联。我们认为，两个基于架构的假设足以回答上述三个问题：其一，注意力机制和 MLP 共享一种键值形式 φ(S)U，其中 φ(S) 在值 U 上进行选择；其二，各组件从加性残差流（即各组件输出之和）中读取信息。将这些选择固定在前向传播时的取值上，模型即可转化为一张计算图，而组件交互、组合路径和词元归因都只是这张图的不同读出方式。我们开发了 Unpack——一种在该计算图上执行的反向归因方法，并将每种读出结果与相应的既有基准测试进行验证：交互得分能够预测消融效应……

    arXiv:2605.23393v3 Announce Type: replace-cross  Abstract: Interpretability methods for transformers are typically built around separate questions: which components interact, how information routes to the output, and which input tokens contribute. Because these methods rely on different assumptions, their answers are difficult to relate. We argue that two architecturally motivated assumptions suffice to address all three questions: attention and MLPs share a key-value form, $\phi(S)\,U$, in which $\phi(S)$ selects over values $U$, and components read from an additive residual stream, the sum of component outputs. Holding these selections at their forward-pass values turns the model into a computational graph, of which component interactions, composition paths, and token attribution are different readouts. We develop Unpack, a backward attribution procedure over this graph, and validate each readout against the corresponding established test: interaction scores predict ablation effects 
    
[^290]: 无损抗蒸馏采样

    Lossless Anti-Distillation Sampling

    [https://arxiv.org/abs/2605.18829](https://arxiv.org/abs/2605.18829)

    提出无损抗蒸馏采样（LADS），在不改变生成内容与质量的前提下，通过在账号间耦合潜在随机性、同时保持账号内生成独立性的机制，大幅削弱蒸馏攻击的效力。

    

    前沿商业生成式模型正面临日益严重的蒸馏威胁：蒸馏者通过收集模型生成的响应，以极低的成本训练出竞争模型。现有的防御方法要么修改生成内容以降低蒸馏性能，从而牺牲响应质量；要么依赖行为检测机制，而这种机制很容易通过多账号查询被绕过。在本工作中，我们提出了无损抗蒸馏采样（LADS），它在保持生成内容本身不变的同时，显著降低蒸馏攻击的有效性。具体而言，LADS 通过一种耦合机制控制推理过程中的潜在随机性，该机制在保持同一账号内生成独立性的同时，引入跨账号之间的依赖性。从构造上看，每个通常只持有一个账号的良性用户，在 LADS 下的使用体验与没有任何防御时完全相同，从而享有无损的体验。

    arXiv:2605.18829v2 Announce Type: replace  Abstract: Frontier commercial generative models face a growing threat from distillation, whereby a distiller harvests generated responses and trains a competing model at drastically lower cost. Existing defenses either modify the generation to degrade distillation performance, sacrificing response quality, or rely on behavioral detection mechanisms that can be readily bypassed through multi-account querying. In this work, we propose Lossless Anti-Distillation Sampling (LADS), which leaves the generation itself unchanged while substantially reducing the effectiveness of distillation. Concretely, LADS controls the latent randomness underlying inference through a coupling mechanism that preserves within-account generation independence while inducing cross-account dependence. By construction, each benign user, who typically holds only a single account, receives the same experience under LADS as they would without any defense, thereby enjoying a lo
    
[^291]: 深度神经网络中的逐点泛化

    Pointwise Generalization in Deep Neural Networks

    [https://arxiv.org/abs/2605.18598](https://arxiv.org/abs/2605.18598)

    该论文为全连接深度神经网络建立了逐点泛化理论，通过基于各层学习特征表示特征值的“逐点黎曼维度”来刻画假设，提出了依赖假设、感知表示的泛化界，在理论和实验上均比基于模型规模、范数乘积和无限宽度线性化的传统方法紧致多个数量级，为表示学习奠定了新的统计基础。

    

    我们通过为全连接网络建立逐点泛化理论，来解决深度神经网络为何能够泛化这一根本问题。该框架突破了刻画丰富非线性特征学习范式的长期障碍，并为表示学习建立了新的统计基础。对于每个训练好的模型，我们通过逐点黎曼维度来刻画假设，该维度由网络各层学习到的特征表示的特征值导出。由此建立了一个有原则的框架，用于推导依赖于假设、感知表示的泛化界。这些泛化界相对于基于模型规模、范数乘积以及无限宽度线性化的方法提供了系统性升级，在理论和实验上都给出了紧致多个数量级的保证。在分析层面，我们识别出了能够解释（原文在此处截断）……

    arXiv:2605.18598v2 Announce Type: replace  Abstract: We address the fundamental question of why deep neural networks generalize by establishing a pointwise generalization theory for fully connected networks. This framework resolves long-standing barriers to characterizing the rich nonlinear feature-learning regime and builds a new statistical foundation for representation learning. For each trained model, we characterize the hypothesis via a pointwise Riemannian Dimension, derived from the eigenvalues of the learned feature representations across layers. This establishes a principled framework for deriving hypothesis-dependent, representation-aware generalization bounds. These bounds offer a systematic upgrade over approaches based on model size, products of norms, and infinite-width linearizations, yielding guarantees that are orders of magnitude tighter in both theory and experiment. Analytically, we identify the structural properties and mathematical principles that explain the trac
    
[^292]: GOMA：从图信号平滑视角迈向结构驱动的多模态对齐

    GOMA: Toward Structure-Driven Multimodal Alignment from a Graph Signal Smoothing Perspective

    [https://arxiv.org/abs/2605.15723](https://arxiv.org/abs/2605.15723)

    提出GOMA方法，从图信号平滑视角出发，通过内容嵌入（负责配对身份区分）和语义嵌入（负责关系一致性）两个相连嵌入的分工设计，实现结构驱动的多模态对齐。

    

    多模态检索利用图像、文本和对象关系来回答关于同一集合的不同问题。一个查询可能寻求某对象的配对描述、同一类别中的另一对象，或通过观测关系相连的对象。这些目标依赖于不同的相关性概念：配对匹配需要对象特定的区分，而跨对象检索则受益于关系一致性。现有方法或学习强跨模态对应，或在图上传播信息，但向邻居正则化的共享输出会削弱身份区分。此外，图正则化带来的增益在均匀图传播后可能减弱。我们提出图优化多模态对齐（GOMA），将这些角色分配给两个相连的嵌入：每个模态生成一个直接受配对身份监督的内容嵌入，以及一个与跨模态结构（联合训练的）语义嵌入。

    arXiv:2605.15723v2 Announce Type: replace  Abstract: Multimodal retrieval uses images, text, and object relationships to answer different questions about the same collection. A query may seek an object's paired description, another object in the same category, or an object connected by an observed relationship. These goals rely on different notions of relevance. Paired matching requires object-specific distinctions, whereas cross-object retrieval benefits from relational agreement. Existing methods learn strong cross-modal correspondence or propagate information over a graph, but a shared output regularized toward neighbors can weaken identity distinctions. Moreover, gains from graph regularization can diminish after uniform graph propagation. We introduce Graph-Optimized Multimodal Alignment (GOMA), which assigns these roles to two connected embeddings. Each modality produces a content embedding directly supervised for paired identity and a semantic embedding jointly trained with cros
    
[^293]: 一种用于实时GPU求解的神经层次矩阵预条件子

    A Neural Hierarchical-Matrix Preconditioner for Real-Time GPU Solves

    [https://arxiv.org/abs/2605.13343](https://arxiv.org/abs/2605.13343)

    该论文提出一种由图-注意力网络学习的H²矩阵格式SPD近似逆预条件子，并采用截断Kaporin条件数作为训练目标以克服近零模态上的梯度消失问题，从而在8-16毫秒预算内实现对每帧变化的稀疏对称正定系统的实时GPU求解。

    

    交互式仿真需要在8-16毫秒的时间预算内，对每帧都在变化的稀疏对称正定（SPD）矩阵A求解Ax=b。在几千个未知数的规模下，仅代数多重网格的构建过程就会超出该时间预算，而Jacobi等局部预条件子虽无需构建，却无法将误差传递到整个求解域。我们为填补这一空白学习了一种预条件子：一个图与注意力网络预测H²矩阵格式的SPD近似逆。在空间排序的三维网格上，真实逆矩阵的块随着其所耦合的簇彼此远离而失去秩；该格式的嵌套基遵循这种衰减规律，因此推理与应用主要由与N呈线性关系的叶子块运算主导，而稠密逆的代价为N²。我们的主要发现涉及训练方面。探测类损失仅通过与A的乘积作用于M，因此其梯度在决定共轭梯度迭代次数的近零模态上消失。截断的Kaporin条件数不含这样的因子；仅改变……（原文在此处截断）

    arXiv:2605.13343v3 Announce Type: replace-cross  Abstract: Interactive simulation solves Ax=b for a sparse SPD A that changes every frame, inside an 8-16 ms budget. At a few thousand unknowns, the setup of algebraic multigrid alone exceeds that budget, while Jacobi and other local preconditioners have no setup but cannot move error across the domain. We learn a preconditioner for this gap: a graph-and-attention network predicts an SPD approximate inverse in H^2-matrix format. On a spatially ordered 3D mesh, blocks of the true inverse lose rank as the clusters they couple move apart; the nested bases of the format follow that decay, so inference and apply are dominated by leaf-block work linear in N, where a dense inverse costs N^2. Our main finding concerns training. Probe losses reach M only through a product with A, so their gradient vanishes on the near-null modes that set the conjugate-gradient iteration count. A truncated Kaporin condition number has no such factor; changing only 
    
[^294]: 基于众包无人机影像检测撒哈拉以南非洲露天倾倒的分散式城市固体垃圾的开放获取模型

    Open-access model for detecting openly dumped dispersed municipal solid waste from crowdsourced UAV imagery in Sub-Saharan Africa

    [https://arxiv.org/abs/2605.02316](https://arxiv.org/abs/2605.02316)

    该研究提出了一个开放获取的深度学习模型，利用众包无人机影像在撒哈拉以南非洲10个国家的29个区域自动检测露天倾倒的分散固体垃圾，揭示了垃圾堆积与人口密度及当地基础设施缺乏密切相关。

    

    在快速城市化的撒哈拉以南非洲地区，由于分散的非正规垃圾倾倒以及用于空间监测的高分辨率数据集有限，城市固体垃圾的管理仍然充满挑战。我们提出了一种开放获取的深度学习模型，利用众包无人机影像自动检测露天倾倒的分散固体垃圾，该模型在涵盖多样环境背景的10个国家的29个区域进行了训练和评估。基于人工标注图像切片训练的深度学习模型在所有研究区域的露天分散固体垃圾检测中均取得了优异表现。预测的垃圾分布揭示了异质性的堆积模式：既有局部热点（通常沿水道分布，垃圾堆积可能加剧洪水风险和公共卫生风险），也有城市区域内更为分散的垃圾分布。垃圾堆积与人口密度以及当地基础设施缺乏的指标关联最为密切。

    arXiv:2605.02316v2 Announce Type: replace-cross  Abstract: Managing municipal solid waste in rapidly urbanizing Sub-Saharan Africa remains challenging due to dispersed informal dumping and limited high-resolution datasets for spatial monitoring. We present an open-access deep learning model for automated detection of openly dumped dispersed solid waste via crowdsourced UAV imagery, trained and evaluated across 29 regions in 10 countries, encompassing diverse environmental contexts. A deep learning model trained on manually annotated image tiles achieved excellent performance in detecting openly dumped dispersed solid waste across all study regions. Predicted distributions reveal heterogeneous accumulation patterns, ranging from localized hotspots - often along waterways, where waste can exacerbate flood and public health risks - to more dispersed litter across urban areas. Waste accumulation is most strongly associated with population density and indicators of lack of local infrastruct
    
[^295]: IatroBench：一个针对语言模型临床信息遗漏的预注册基准测试

    IatroBench: A Pre-Registered Benchmark of Clinical Omission in Language Models

    [https://arxiv.org/abs/2604.07709](https://arxiv.org/abs/2604.07709)

    论文提出预注册基准IatroBench，从“作为”与“遗漏”两个伤害维度评估语言模型在临床场景中的安全性，并首次揭示了模型对同一病例会向医生提供比患者更多临床信息的“框架依赖性信息保留”现象。

    

    一个经过强安全训练的模型会为医生提供苯二氮䓬类药物的减量停药方案，却不会为提出同样请求的患者提供。模型本身知道这些信息，但分享多少取决于提问的框架。我们提出了IatroBench，一个在两类伤害维度（作为性伤害与遗漏性伤害）上，通过60个预注册临床场景对6个模型进行评估的基准测试。我们使用Claude Opus 4.6依据一位医生撰写的评分细则对模型回复进行打分，发现其遗漏评分与该医生评分的一致性程度，与另一位医生评分之间的一致性相当。我们发现，当同一病例分别以患者提问和医生会诊两种形式呈现时（两种变体在语体、请求方式以及隐含的治疗医生监督方面也存在差异），我们测试的全部五个模型向医生分享的信息都多于向患者分享的信息。我们将这种现象称为“框架依赖性信息保留”。我们发现平均解耦差距为+0.38……

    arXiv:2604.07709v5 Announce Type: replace  Abstract: A strongly safety-trained model will provide a doctor with a benzodiazepine taper schedule, but not a patient who asks for one. The model knows the information, but how much it shares depends on the framing. We introduce IatroBench, a benchmark that evaluates models on two axes of harm (commission and omission) across 60 pre-registered clinical scenarios and 6 models. We use Claude Opus 4.6 to score model responses against a rubric written by a physician, and find that its omission scores are as well-aligned to the physician's scores as another physician's scores are. We find that when the same case is presented as a patient query and a doctor consultation (the variants also differ in register, request and the supervision a treating physician implies), all five models we test share more information with the doctor than the patient. We term this phenomenon "framing-contingent withholding." We find a mean decoupling gap of +0.38 across
    
[^296]: LiveMathematicianBench：一个基于证明概要的研究级数学推理动态基准测试

    LiveMathematicianBench: A Live Benchmark for Research-Level Mathematical Reasoning with Proof Sketches

    [https://arxiv.org/abs/2604.01754](https://arxiv.org/abs/2604.01754)

    提出了LiveMathematicianBench，一个基于训练截止日期后新发表arXiv论文构建的动态研究级数学推理基准测试，通过引入十三类定理逻辑分类体系和证明概要实现细粒度评估，有效避免了数据污染问题。

    

    数学推理是人类智力的标志，大型语言模型（LLM）能否有意义地进行数学推理仍然是人工智能和认知科学中的一个核心问题。随着大语言模型越来越多地被整合到科学工作流程中，对其数学能力进行严格评估已成为一种实际需求。现有的基准测试受限于合成环境和数据污染。我们提出了LiveMathematicianBench，这是一个基于模型训练截止日期之后发布的最新arXiv论文构建的、面向研究级数学推理的动态选择题基准测试。通过将评估建立在新发表的定理之上，它提供了一个超越记忆模式的真实测试平台。该基准测试引入了一个包含十三个类别的定理类型逻辑分类体系（例如蕴含、等价、存在性、唯一性），从而实现跨推理形式的细粒度评估。它采用了一种证明——

    arXiv:2604.01754v2 Announce Type: replace-cross  Abstract: Mathematical reasoning is a hallmark of human intelligence, and whether large language models (LLMs) can meaningfully perform it remains a central question in artificial intelligence and cognitive science. As LLMs are increasingly integrated into scientific workflows, rigorous evaluation of their mathematical capabilities becomes a practical necessity. Existing benchmarks are limited by synthetic settings and data contamination. We present LiveMathematicianBench, a dynamic multiple-choice benchmark for research-level mathematical reasoning built from recent arXiv papers published after model training cutoffs. By grounding evaluation in newly published theorems, it provides a realistic testbed beyond memorized patterns. The benchmark introduces a thirteen-category logical taxonomy of theorem types (e.g., implication, equivalence, existence, uniqueness), enabling fine-grained evaluation across reasoning forms. It employs a proof-
    
[^297]: 可逆查询-键耦合与注意力机制的组合

    Invertible Query-Key Coupling Composes with Attention Mechanisms

    [https://arxiv.org/abs/2604.01683](https://arxiv.org/abs/2604.01683)

    提出一种可逆的查询-键耦合变换（RealNVP风格的交替仿射映射），可无损地叠加在现有注意力机制之上，以极少的额外参数和不变的整体架构显著提升差分注意力等方法的语言建模性能。

    

    arXiv:2604.01683v2 公告类型：replace-cross 摘要：缩放点积注意力将查询和键构建为相互独立的线性投影，因此两者在计算评分的点积之前从不发生交互。我们研究了耦合的查询-键动力学，这是一种评分前的变换，它在标准评分之前通过共享的可逆耦合使每个token的查询和键共同演化。我们将其实现为实非体积保持流（RealNVP）风格的交替仿射映射：该耦合在初始化时为恒等映射，每个注意力头仅增加少量参数，并保持softmax及周围架构不变。我们将耦合叠加在现有注意力方法之上而非替换它们，并探究这种组合是否有所帮助。在WikiText-103上，为差分注意力添加耦合后在150M和455M两种参数规模上均带来了改进。在455M参数下，该增益在序列长度512处具有统计显著性（p=0.003，六个随机种子），通过了Bonferroni校正并成功复现……

    arXiv:2604.01683v2 Announce Type: replace-cross  Abstract: Scaled dot-product attention forms its queries and keys as independent linear projections, so the two never interact before the dot product that scores them. We study coupled query-key dynamics, a pre-scoring transformation that evolves each token's query and key jointly through a shared invertible coupling before standard scoring. We realize it as an alternating affine map in the style of real non-volume-preserving flows: the coupling is the identity at initialization, adds a small fraction of parameters per head, and leaves the softmax and surrounding architecture unchanged. We place coupling on top of existing attention methods rather than replacing them, and ask whether that composition helps. On WikiText-103, adding coupling to Differential Attention improves on it at both 150M and 455M parameters. At 455M the gain is significant at sequence length 512 (p=0.003, six seeds), survives a Bonferroni correction and replicates o
    
[^298]: MemGuard-Alpha：基于大语言模型的金融预测中记忆化污染信号检测与过滤的成员推断极限

    MemGuard-Alpha: Limits of Membership Inference for Detecting and Filtering Memorization-Contaminated Signals in LLM-Based Financial Forecasting

    [https://arxiv.org/abs/2603.26797](https://arxiv.org/abs/2603.26797)

    该论文提出MemGuard-Alpha框架，通过融合五种MIA方法与时间邻近特征的复合污染分数以及跨模型记忆化分歧，系统审计了成员推断攻击在检测并过滤大语言模型金融预测中记忆化污染信号方面的能力与局限。

    

    arXiv:2603.26797v2 公告类型：替换 摘要：大语言模型越来越多地被用于生成金融Alpha信号，但许多模型已经记忆了其训练语料库中的历史数据，产生了在样本外迅速失效的表面准确性。成员推断攻击（MIA）已被提出作为一种诊断手段。然而尚未确定的是：在该场景下，MIA分数是否对记忆化真正具有信息量，以及在考虑现实交易成本后，基于MIA分数构建的信号级过滤是否仍有帮助。我们提出了MemGuard-Alpha，其包含一个由五种MIA方法与时间邻近特征组合而成的复合污染分数，以及跨模型记忆化分歧，后者利用不同模型训练截止时间的差异。随后我们对这两种方法进行了审计。在七个大语言模型（124M-7B）、50个标普100成分股、42,800个提示词以及2019至2024年间299,600个提示-模型MIA分数上，我们得出三个发现。第一，当样本内状态由训练截止时间定义时，一个时间……（原文摘要至此截断）

    arXiv:2603.26797v2 Announce Type: replace  Abstract: Large language models are increasingly used to generate financial alpha signals, but many have memorized the historical data in their training corpora, producing apparent accuracy that collapses out of sample. Membership inference attacks (MIA) have been proposed as a diagnostic. What has not been established is whether MIA scores are informative about memorization in this setting, or whether signal-level filtering built on them helps once realistic costs are applied.We introduce MemGuard-Alpha, comprising a composite contamination score combining five MIA methods with a temporal proximity feature, and Cross-Model Memorization Disagreement, which exploits variation in training cutoffs across models. We then audit both. Across seven LLMs (124M-7B), 50 S&P 100 constituents, 42,800 prompts and 299,600 prompt-model MIA scores spanning 2019-2024, three findings emerge.First, where in-sample status is defined by a training cutoff, a tempor
    
[^299]: 超越成对注意力：面向高效序列学习的高阶模块化注意力

    Beyond Pairwise Attention: Higher-Order Modular Attention for Efficient Sequence Learning

    [https://arxiv.org/abs/2603.11133](https://arxiv.org/abs/2603.11133)

    该论文提出高阶模块化注意力HOMA，通过重叠块、局部窗口和低秩投影将成对注意力与可计算的三元注意力显式融合，在捕捉超出显式建模阶数的高阶依赖的同时实现更快收敛和更高的参数效率。

    

    序列建模任务可能涉及内在的高阶依赖关系，而标准的自注意力机制只为token对分配分数，并未显式参数化这类交互。我们提出了高阶模块化注意力（HOMA），它将成对注意力与一条显式的三元注意力通路相融合，并通过重叠块、局部窗口和低秩投影使该通路变得可计算。我们在受控的PARITY和MATCH3任务以及TAPE基准上，将HOMA与匹配的成对基线及纯三元基线进行了比较。结果显示，HOMA与基线相当甚至更优，尤其当底层依赖关系超出显式建模的三元阶数时，其优势最为明显。在若干设置中，这些优势还伴随着更快的收敛速度和更好的参数效率，且所学到的非线性融合机制为有效组合成对与三元表示提供了一条可行途径。

    arXiv:2603.11133v2 Announce Type: replace  Abstract: Sequence modeling tasks can involve intrinsic higher-order dependencies, while standard self-attention assigns scores to token pairs and does not explicitly parameterize such interactions. We introduce Higher-Order Modular Attention (HOMA), which fuses pairwise attention with an explicit triadic attention pathway made tractable through overlapping blocks, local windows, and a low-rank projection. We compare HOMA with matched pairwise and purely triadic baselines on controlled PARITY and MATCH3 tasks, as well as TAPE benchmarks. HOMA is competitive with or outperforms the baselines, with its clearest advantages when the underlying dependencies extend beyond the explicitly modeled triadic order. These advantages are accompanied in several settings by faster convergence and improved parameter efficiency, with the learned nonlinear fusion providing an effective mechanism for combining the pairwise and triadic representations. Overall, ou
    
[^300]: MultiwayPAM：用于LLM-as-a-Judge评分分析的多路围绕中心点划分方法

    MultiwayPAM: Multiway Partitioning Around Medoids for LLM-as-a-Judge Score Analysis

    [https://arxiv.org/abs/2603.10287](https://arxiv.org/abs/2603.10287)

    该论文提出了MultiwayPAM，一种新的张量聚类方法，能够同时估计LLM-as-a-Judge评分张量各模式的聚类成员和中心点，从而揭示LLM评估器评分偏差的结构。

    

    LLM-as-a-Judge是一种灵活的文本评估框架，通过更改提示模板，我们可以从多个角度获得对给定文本质量的评分。使用LLM-as-a-Judge的两大主要挑战是：使用大语言模型（LLM）进行推理的计算成本（尤其是在评估大量实例时），以及LLM评估器固有的偏差。为了解决这些问题并揭示LLM评估器造成的评分偏差结构，我们提出将张量聚类方法应用于给定的LLM-as-a-Judge评分张量，该张量的元素是不同问题、回答者和评估者组合所对应的评分。具体而言，我们开发了一种新的张量聚类方法MultiwayPAM，利用该方法可以同时估计给定数据张量每个模式的聚类成员关系和中心点。通过观察MultiwayPAM获得的中心点，我们可以获得关于模型行为的相关知识。

    arXiv:2603.10287v2 Announce Type: replace-cross  Abstract: LLM-as-a-Judge is a flexible framework for text evaluation, which allows us to obtain scores for the quality of a given text from various perspectives by changing the prompt template. Two main challenges in using LLM-as-a-Judge are computational cost of inference using a large language model (LLM), especially when evaluating a large number of instances, and inherent bias of an LLM evaluator. To address these issues and reveal the structure of score bias caused by an LLM evaluator, we propose to apply a tensor clustering method to a given LLM-as-a-Judge score tensor, whose entries are the scores for different combinations of questions, answerers, and evaluators. Specifically, we develop a new tensor clustering method MultiwayPAM, with which we can simultaneously estimate the cluster membership and the medoids for each mode of a given data tensor. By observing the medoids obtained by MultiwayPAM, we can gain knowledge about the m
    
[^301]: 脚手架之下的安全性：评估条件如何塑造所测得的安全表现

    Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety

    [https://arxiv.org/abs/2603.10044](https://arxiv.org/abs/2603.10044)

    评测条件对测得的模型安全性影响超过脚手架本身——在相同的基准题目上，选择题与开放式格式会使测得的安全性相差5-20个百分点，说明评测结果更多取决于测量方法而非模型潜在的 safety 能力。

    

    安全基准测试通常针对“裸”模型——即接收提示并输出响应的模型——进行，但现实世界的部署会将这些模型“包裹”在复杂的脚手架中。这些脚手架对基准测试所衡量的模型安全性究竟有多大影响？我们在四个预先注册的安全基准上，使用直接 API 以及三种脚手架（ReAct、多智能体和 map-reduce）测试了六个领先模型，共进行了 62,808 次评分评估。研究发现，安全性的测量方式比脚手架本身更为重要：对于其他方面完全相同的基准题目，使用选择题还是开放式问题格式，会使测得的安全性相差 5-20 个百分点。由于这两种格式采用不同的评分方法（答案提取与 LLM 评审），这一差距源于测量方式而非模型潜在安全性的差异。若使用启发式方法对模型拒绝行为进行分类，将在五种情况下得出不同的结论。基准的选择可解释结果变异的 19.3%。

    arXiv:2603.10044v3 Announce Type: replace  Abstract: Safety benchmarks usually test "bare" models that receive prompts and output responses, but real-world deployments "wrap" those models in complex scaffolds. How much do these scaffolds affect model safety as measured by benchmarks? We test six leading models on four pre-registered safety benchmarks with a direct API and three scaffolds: ReAct, multi-agent, and map-reduce. We conducted 62,808 scored evaluations. How safety is measured matters more than scaffolding does: we find that using a multiple choice vs. open-ended format for otherwise-identical benchmark items changes measured safety by 5-20 percentage points (pp). The two formats are scored with different methods (answer extraction and an LLM judge), so the gap is due to measurement rather than differences in latent safety. Using a heuristic to classify model refusals would have led to different findings in five cases. Benchmark choice explains 19.3% of the variation in outcom
    
[^302]: 基于最优序分数搜索的时间序列因果结构学习

    Learning Causal Structure of Time Series using Best Order Score Search

    [https://arxiv.org/abs/2603.05370](https://arxiv.org/abs/2603.05370)

    本文提出TS-BOSS算法，将最优序分数搜索（BOSS）扩展到多变量时间序列的动态贝叶斯网络因果结构学习中，兼具可扩展性与理论保证。

    

    从观测数据中学习因果结构是许多科学和政策领域的核心问题，但许多学科中常见的时间序列设置由于时间依赖性而带来了若干挑战。在本文中，我们聚焦于基于分数的多变量时间序列因果发现，并提出了TS-BOSS，这是最近提出的最佳序分数搜索（Best Order Score Search, BOSS）（Andrews et al. 2023）的时间序列扩展版本。TS-BOSS在动态贝叶斯网络结构上执行基于置换的搜索，同时利用生长-收缩树来缓存中间分数计算，从而在时间序列设置中保留了BOSS在静态设置中的可扩展性和出色的实证性能。我们提供了理论保证，证明了在适当假设下TS-BOSS的可靠性，并给出了一个中间结果，将基于置换方法的经典子图最小性结果扩展到了动态（时间序列）设置中。我们的实验……

    arXiv:2603.05370v2 Announce Type: replace-cross  Abstract: Causal structure learning from observational data is central to many scientific and policy domains, but the time series setting common to many disciplines poses several challenges due to temporal dependence. In this paper we focus on score-based causal discovery for multivariate time series and introduce TS-BOSS, a time series extension of the recently proposed Best Order Score Search (BOSS) (Andrews et al. 2023). TS-BOSS performs a permutation-based search over dynamic Bayesian network structures while leveraging grow-shrink trees to cache intermediate score computations, preserving the scalability and strong empirical performance of BOSS in the static setting. We provide theoretical guarantees establishing the soundness of TS-BOSS under suitable assumptions, and we present an intermediate result that extends classical subgraph minimality results for permutation-based methods to the dynamic (time series) setting. Our experimen
    
[^303]: RQ-Reg：一种基于残差量化的推荐系统连续值预测框架

    RQ-Reg: A Residual-Quantization-Based Framework for Continuous Value Prediction in Recommender Systems

    [https://arxiv.org/abs/2602.23012](https://arxiv.org/abs/2602.23012)

    该论文提出RQ-Reg框架，通过残差量化将观看时长、GMV等连续目标值分解为从粗到细的量化编码序列并进行自回归预测，以克服传统回归方法在建模复杂长尾分布时欠拟合或牺牲泛化性的缺陷。

    

    预测观看时长和商品交易总额（GMV）等连续值是工业推荐系统中的核心问题。其固有难点在于目标信号具有高度复杂且呈长尾的分布，难以精确建模。现有的回归方法通常依赖于对目标分布的固定参数假设：过于简单的假设会对真实世界的数据欠拟合，而更复杂的假设则往往牺牲可扩展性与泛化能力。为解决这些局限性，我们提出了一种基于残差量化（RQ）的序列建模框架，将目标连续值分解为一系列量化编码，这些编码代表逐步精细化的近似值。模型以自回归的方式从粗粒度到细粒度地预测这些编码，每一步都细化上一步遗留的残差误差。为进一步提升质量……（摘要原文在此处截断）

    arXiv:2602.23012v2 Announce Type: replace-cross  Abstract: Predicting continuous values such as watch-time and gross merchandise value (GMV) is a core problem in industrial recommendation systems. Its inherent difficulty stems from the highly complex and long-tailed distributions of the target signals, which are hard to model accurately. Existing regression methods typically rely on fixed parametric assumptions on the target distribution: overly simple assumptions underfit real-world data, whereas more intricate ones tend to sacrifice scalability and generalization. To address these limitations, we propose a sequence modeling framework based on residual quantization (RQ), in which the target continuous value is decomposed into a sequence of quantization codes that represent progressively finer approximations. The model autoregressively predicts these codes from coarse to fine granularity, with each step refining the residual error left by the previous one. To further improve the qualit
    
[^304]: 用于同步电机通用磁性建模的梯度网络

    Gradient Networks for Universal Magnetic Modeling of Synchronous Machines

    [https://arxiv.org/abs/2602.14947](https://arxiv.org/abs/2602.14947)

    该论文提出将梯度网络嵌入电机方程的物理约束框架，用于含空间谐波的饱和同步电机通用磁性建模，从构造上保证互易性与能量守恒，同时兼顾高精度、高数据效率与实时控制的可行性。

    

    本文提出了一种物理约束神经网络框架，用于对饱和同步电机进行磁性建模，并包含空间谐波的影响。通过将梯度网络嵌入电机方程中以建模守恒的电磁行为，该框架在构造上即满足互易性和能量守恒，同时能够普遍逼近任何物理上可行的磁特性。与查找表和黑盒神经网络不同，该方法保证了单调性、可逆性和平滑的输出，并且具有极高的数据效率。该方法利用一台5.6千瓦永磁同步磁阻电机的实测数据和有限元法（FEM）数据进行了验证，并在嵌入式平台上演示了实时闭环控制。结果证实了该方法具有精确、物理一致且计算高效的表现。

    arXiv:2602.14947v4 Announce Type: replace-cross  Abstract: This paper presents a physics-constrained neural network framework for magnetic modeling of saturable synchronous machines, including spatial harmonics. By embedding gradient networks into the machine equations to model conservative electromagnetic behavior, the framework satisfies reciprocity and energy conservation by construction, while universally approximating any physically feasible magnetic characteristic. Unlike lookup tables and black-box neural networks, it guarantees monotonicity, invertibility, and smooth outputs, and remains highly data efficient. The method is validated using measured and finite-element method (FEM) data from a 5.6-kW permanent-magnet (PM) synchronous reluctance machine, and is demonstrated in real-time closed-loop control on an embedded platform. The results confirm accurate, physically consistent, and computationally efficient performance.
    
[^305]: TabSieve：面向表格预测的显式表内证据选择

    TabSieve: Explicit In-Table Evidence Selection for Tabular Prediction

    [https://arxiv.org/abs/2602.11700](https://arxiv.org/abs/2602.11700)

    TabSieve提出了一种先选择后预测的表格预测框架，通过显式选择表内证据行、构建40K规模的合成微调数据集TabSieve-SFT-40K，以及结合分离奖励的强化学习方法TAB-GRPO，实现了可审计且稳健的表格预测。

    

    表格预测可以利用表内行作为少样本证据，但现有的表格模型通常执行实例级推理，且基于大语言模型的提示方法往往较为脆弱。模型无法持续稳定地利用相关行，而嘈杂的上下文还可能降低性能。为应对这一挑战，我们提出了TabSieve，这是一个“先选择后预测”的框架，使证据的使用变得显式且可审计。给定一个表格和一个查询行，TabSieve首先选出一小组信息量大的行作为证据，然后在所选证据的条件下预测缺失的目标值。为实现这一能力，我们通过使用强大的教师模型并施加严格过滤，从331个真实表格中合成高质量推理轨迹，构建了TabSieve-SFT-40K数据集。此外，我们引入了TAB-GRPO，这是一种强化学习方案，通过分离的奖励信号共同优化证据选择与预测正确性，并稳定了混合……（原文摘要至此处被截断）

    arXiv:2602.11700v2 Announce Type: replace-cross  Abstract: Tabular prediction can benefit from in-table rows as few-shot evidence, yet existing tabular models typically perform instance-wise inference and LLM-based prompting is often brittle. Models do not consistently leverage relevant rows, and noisy context can degrade performance. To address this challenge, we propose TabSieve, a select-then-predict framework that makes evidence usage explicit and auditable. Given a table and a query row, TabSieve first selects a small set of informative rows as evidence and then predicts the missing target conditioned on the selected evidence. To enable this capability, we construct TabSieve-SFT-40K by synthesizing high-quality reasoning trajectories from 331 real tables using a strong teacher model with strict filtering. Furthermore, we introduce TAB-GRPO, a reinforcement learning recipe that jointly optimizes evidence selection and prediction correctness with separate rewards, and stabilizes mix
    
[^306]: 基于事前稀疏化的近预言机KV选择方法用于长上下文推理

    Near-Oracle KV Selection via Pre-hoc Sparsity for Long-Context Inference

    [https://arxiv.org/abs/2602.08329](https://arxiv.org/abs/2602.08329)

    该论文提出事前稀疏化方法PrHS，在注意力打分之前进行KV选择以避免事后启发式方法的后验偏差，并推导出仅依赖丢弃质量的互信息损失上界，从而为长上下文LLM推理提供具有显式精度控制的近预言机KV选择。

    

    大语言模型（LLM）推理的一个核心瓶颈是对不断增长的键值（KV）缓存进行注意力计算的开销。尽管近预言机的top-k KV选择能够在大幅减少计算和带宽的同时保持稠密注意力的质量，但现有的稀疏方法通常依赖事后启发式方法，即以观察到的注意力或代理分数为条件的选择器。这种条件化引入了后验偏差：它往往会扭曲真实的token重要性并遗漏显著的token，从而损害长程推理能力。为解决这一问题，我们提出事前稀疏化方法（Pre-hoc Sparsity, PrHS），它在注意力打分之前选择KV条目，并提供显式的精度控制。设被丢弃条目的注意力质量为delta（丢弃质量），通过从边际分布到互信息的分析，我们推导出了一个仅依赖于丢弃质量的互信息损失上界。这一关系解释了现有方法失败的……

    arXiv:2602.08329v2 Announce Type: replace-cross  Abstract: A core bottleneck in large language model (LLM) inference is the cost of attending over the ever-growing key-value (KV) cache. Although near-oracle top-k KV selection can preserve the quality of dense attention while sharply reducing computation and bandwidth, existing sparse methods generally rely on posterior heuristics, i.e., selectors conditioned on observed attention or proxy scores. Such conditioning introduces posterior bias: it tends to distort true token importance and miss salient tokens, thereby impairing long-range reasoning. To tackle this problem, we propose Pre-hoc Sparsity (PrHS), which selects KV entries before attention scoring and provides explicit accuracy control. Let the attention mass of discarded entries be delta (the dropped mass). Through a marginal-to-mutual-information analysis, we derive an upper bound on the mutual-information loss that depends only on the dropped mass. This relation explains failu
    
[^307]: 基于重叠干涉的量子注意力机制：预测经典与多体量子序列

    Quantum Attention by Overlap Interference: Predicting Classical and Many-Body Quantum Sequences

    [https://arxiv.org/abs/2602.06699](https://arxiv.org/abs/2602.06699)

    提出了一种通过状态重叠干涉与多项式核实现非线性、并利用Rényi-1/2熵泛函估计损失的变分量子自注意力机制（QSA），相比最优经典方法在训练复杂度上具有潜在优势，可用于预测经典和多体量子序列。

    

    我们提出了一种自注意力机制的变分量子实现（QSA）——它是Transformer和大语言模型的核心操作——通过形成对过去数据的重叠加权组合来预测序列的未来元素。与以往方法不同，我们的QSA通过状态重叠的干涉和k次多项式核来实现所需的非线性，并通过两个可观测量的期望值来估计基于Rényi-1/2熵泛函的损失，从而避免了将振幅编码的预测解码为经典概率。QSA还支持一种受约束的可训练数据嵌入，将状态重叠与数据层面的相似性联系起来。其主要的端到端训练复杂度以 $O(\mu^{-1}k^2Td)$ 的方式扩展，而最公平的经典对比方法复杂度为 $O(Td^{k+1})$，其中 $\mu$ 为训练信号；我们通过数值实验表明，这可以带来复杂度上的优势。

    arXiv:2602.06699v2 Announce Type: replace-cross  Abstract: We propose a variational quantum implementation of self-attention (QSA)-the core operation in transformers and large language models-which predicts future elements of a sequence by forming overlap-weighted combinations of past data. At variance with previous approaches, our QSA realizes the required nonlinearity through interference of state overlaps and a degree-$k$ polynomial kernel, and estimates a loss based on R\'enyi-$1/2$ entropic functionals via two observables' expectation values, avoiding the decoding of amplitude-encoded predictions into classical probabilities. QSA also accommodates a constrained, trainable data-embedding tying state overlaps to data-level similarities. Its dominant end-to-end training complexity scales as $O\left(\mu^{-1}k^2Td\right)$, versus $O\left(T d^{k+1}\right)$ of the fairest classical comparison, with $\mu$ a training signal; we show numerically that this allows a complexity advantage in th
    
[^308]: TIDE：面向自改进大语言模型推理的时间增量草稿引擎

    TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference

    [https://arxiv.org/abs/2602.05145](https://arxiv.org/abs/2602.05145)

    TIDE 通过复用推理过程中的中间隐藏状态在线增量训练草稿模型，并结合自适应运行时控制与异构 GPU 集群调度，在不增加额外目标模型开销的情况下实现了高达 1.66 倍的 LLM 推理吞吐量提升。

    

    推测解码可以显著加速大语言模型推理，但由于工作负载不断演变，在实际中充分实现其收益具有挑战性。我们提出了 TIDE（Temporal Incremental Draft Engine，时间增量草稿引擎），这是一个服务引擎原生的框架，将在线草稿模型适配直接集成到高性能大语言模型推理系统中。TIDE 复用目标模型在推理过程中产生的中间隐藏状态作为草稿模型适配的训练信号，从而避免了额外的目标模型计算和运行时开销。它采用自适应运行时控制，仅在有益时才激活推测解码和草稿模型训练。TIDE 还通过将推理和训练任务映射到合适的 GPU 类别来充分利用异构集群。在多种真实世界工作负载上，TIDE 相比无推测解码的基线实现了高达 1.66 倍的吞吐量提升，并在静态草稿模型性能下降的分布不对齐工作负载上恢复了性能表现。

    arXiv:2602.05145v2 Announce Type: replace-cross  Abstract: Speculative decoding can substantially accelerate LLM inference, but realizing its benefits in practice is challenging due to evolving workloads. We present TIDE (Temporal Incremental Draft Engine), a serving-engine-native framework that integrates online draft adaptation directly into high-performance LLM inference systems. TIDE reuses target model's intermediate hidden states generated during inference as training signals for draft adaptation, thereby avoiding additional target model computation and serving-time overhead. It employs adaptive runtime control to activate speculation and draft model training only when beneficial. TIDE exploits heterogeneous clusters by mapping inference and training to appropriate GPU classes. Across diverse real-world workloads, TIDE achieves up to 1.66$\times$ throughput over no-speculation baselines while recovering performance on misaligned workloads where static draft models degrade through
    
[^309]: 论序贯假设检验中的成本感知设计

    On Cost-Aware Designs for Sequential Hypothesis Testing

    [https://arxiv.org/abs/2512.19067](https://arxiv.org/abs/2512.19067)

    本文提出了成本感知序贯假设检验框架，证明了最优期望总成本按 $\Theta(\log(1/\delta))$ 缩放，并揭示了“最大化期望信息增益与期望成本之比”这一设计原则，据此改编的经典策略具有渐近最优性。

    

    我们提出了成本感知序贯假设检验，其中主动决策者选择具有不同随机成本的感知动作，在平均误差约束 $\delta$ 下识别真实假设，同时最小化期望总成本而非样本数量。对于固定成本，我们证明了最优期望总成本的量级为 $\Theta(\log(1/\delta))$，并且可以通过基于多假设序贯概率比检验（MSPRT）的程序实现。我们证明了成本感知的设计原则是在策略诱导的动作分布下最大化期望信息增益与期望成本之比。在此原则指导下，我们将两种经典策略改编至成本感知设置中，并建立了它们的渐近最优性。随后，我们在两种揭示模型下处理随机成本：事后揭示模型（即仅在获得样本后才披露成本，此时成本-误差权衡与固定成本情形一致）……

    arXiv:2512.19067v2 Announce Type: replace-cross  Abstract: We introduce Cost-Aware (CA) Sequential Hypothesis Testing (CASHT), in which an active decision-maker selects sensing actions with differing, random costs to identify the true hypothesis under an average-error constraint $\delta$ while minimizing the expected total cost rather than the number of samples. For fixed costs, we prove that the optimal expected total cost scales as $\Theta(\log(1/\delta))$, and is achievable by Multihypothesis Sequential Probability Ratio Test-based procedures. We show that the CA design principle is to maximize the ratio of expected information gain to expected cost under the policy-induced action distribution. Guided by this principle, we adapt two classic policies to the CA setting and establish their asymptotic optimality. We then treat random costs under two revelation models: ex-post, where costs are disclosed only after a sample is obtained, and the cost-error tradeoff coincides with the fixed
    
[^310]: 基于集成复数变分模态分解与空间注意力迁移的小样本辐射源个体识别

    Few-Shot Specific Emitter Identification via Integrated Complex Variational Mode Decomposition and Spatial Attention Transfer

    [https://arxiv.org/abs/2512.16786](https://arxiv.org/abs/2512.16786)

    该论文提出集成复数变分模态分解、时间卷积网络与空间注意力机制并借助预训练权重迁移的方法，在标注数据有限的小样本条件下实现了高精度的辐射源个体识别。

    

    辐射源个体识别（SEI）利用无源硬件特性对发射机进行身份认证，提供了一种稳健的物理层安全解决方案。然而，大多数基于深度学习的方法依赖大量数据或需要先验信息，这在标注数据有限的现实场景中带来了挑战。我们提出了一种集成复数变分模态分解算法，通过分解并重构复值信号来逼近原始发射信号，从而实现更精确的特征提取。我们进一步利用时间卷积网络有效建模信号的时序特性，并引入空间注意力机制对信息丰富的信号片段进行自适应加权，显著提升了识别性能。此外，分支网络设计允许利用来自其他数据的预训练权重，同时减少对辅助数据的需求。

    arXiv:2512.16786v2 Announce Type: replace-cross  Abstract: Specific emitter identification (SEI) utilizes passive hardware characteristics to authenticate transmitters, providing a robust physical-layer security solution. However, most deep-learning-based methods rely on extensive data or require prior information, which poses challenges in real-world scenarios with limited labeled data. We propose an integrated complex variational mode decomposition algorithm that decomposes and reconstructs complex-valued signals to approximate the original transmitted signals, thereby enabling more accurate feature extraction. We further utilize a temporal convolutional network to effectively model the sequential signal characteristics, and introduce a spatial attention mechanism to adaptively weight informative signal segments, significantly enhancing identification performance. Additionally, the branch network allows leveraging pre-trained weights from other data while reducing the need for auxili
    
[^311]: 一种快速有效解决大语言模型前瞻偏差问题的方法

    A Fast and Effective Solution to the Problem of Look-ahead Bias in LLMs

    [https://arxiv.org/abs/2512.06607](https://arxiv.org/abs/2512.06607)

    本文提出一种推理时干预方法，利用两个小型专用模型调整大模型logits，从而快速、低成本地消除大语言模型在金融预测中的前瞻偏差。

    

    由于大语言模型在长时间序列数据上训练而产生前瞻偏差，将其应用于金融预测任务面临挑战。这使得金融领域通常采用的回测方法无法实施，因为使用特定知识截止日期从头重新训练前沿模型的成本过于高昂。本文提出了一种快速、有效且低成本的替代方案。我们的方法在推理阶段通过一对较小的专门模型来调整大型基础模型的logits，从而引导生成过程——其中一个模型在需要遗忘的信息上进行微调，另一个在需要保留的信息上进行微调。我们证明该方法能有效消除逐字和语义两个层面的知识，纠正偏差，并且优于已有方法。

    arXiv:2512.06607v2 Announce Type: replace-cross  Abstract: Applying LLMs to predictive tasks in finance is challenging due to look-ahead bias resulting from their training on long time-series data. This precludes the backtests typically employed in finance since retraining frontier models from scratch with a specific knowledge cutoff is prohibitive. In this paper, we introduce a fast, effective, and low-cost alternative. Our method guides generation at inference time by adjusting the logits of a large base model using a pair of smaller, specialized models -- one fine-tuned on information to be forgotten and another on information to be retained. We demonstrate that our method effectively removes both verbatim and semantic knowledge, corrects biases, and outperforms prior methods.
    
[^312]: 面向智能开放RAN的自定位MIMO波束映射：具有持续演进的信道记忆

    Self-Localizing MIMO Beam Mapping for Intelligent Open RAN with Continuously Evolving Channel Memory

    [https://arxiv.org/abs/2511.17007](https://arxiv.org/abs/2511.17007)

    本文提出了一种无需位置标签的自定位MIMO波束映射框架，利用稀疏CSI测量和波束域RSS构建分层无线记忆，并通过双尺度提取器和混合时间编码器实现高效的信道知识推理。

    

    摘要：面向6G的开放智能无线接入网络（RAN）需要准确且可复用的无线信道知识，以支持智能推理和控制。然而，在开放和多供应商部署中，全维度信道状态信息（CSI）和精确的位置标签难以获取和维护。本文开发了一种自定位多输入多输出（MIMO）波束映射框架，该框架从高度稀疏的CSI测量中构建分层无线记忆，无需显式位置标签。为降低采集和处理开销，我们使用波束域接收信号强度（RSS）作为紧凑输入，并理论上证明其能够实现渐近无偏的空间特征估计。一种双尺度提取器捕获不完整观测中的快照内角度依赖性和样本间相关性，并设计了一种混合时间编码器来整合近期数据。

    arXiv:2511.17007v2 Announce Type: replace-cross  Abstract: Open and intelligent radio access networks (RANs) envisioned for 6G require accurate and reusable wireless channel knowledge for intelligent inference and control. However, full-dimensional channel state information (CSI) and accurate location labels are difficult to acquire and maintain across open and multi-vendor deployments. This paper develops a self-localizing multiple-input multiple-output (MIMO) beam map framework that constructs a hierarchical wireless memory from highly sparse CSI measurements without explicit location labels. To reduce acquisition and processing overhead, we use beam-domain received signal strength (RSS) as compact inputs and theoretically show that they enable asymptotically unbiased spatial signature estimation. A dual-scale extractor captures intra-snapshot angular dependencies and inter-sample correlations for incomplete observations, and a hybrid temporal encoder is designed to consolidate recen
    
[^313]: MLPerf Automotive（汽车领域机器学习基准测试）

    MLPerf Automotive

    [https://arxiv.org/abs/2510.27065](https://arxiv.org/abs/2510.27065)

    MLPerf Automotive 是首个针对汽车 AI 加速机器学习系统的标准化公开性能基准测试，提供可复现的评估框架，涵盖 2D/3D 目标检测、语义分割和端到端驾驶等汽车感知任务。

    

    我们提出 MLPerf Automotive，这是首个用于评估部署于汽车系统中实现 AI 加速的机器学习系统的标准化公开性能基准测试。该基准测试由 MLCommons 内部合作伙伴共同开发，旨在满足汽车机器学习系统中标准化性能评估方法的需求。现有的基准测试套件无法应用于此类系统，因为汽车工作负载具有独特的约束条件，包括传感器套件、安全性和实时处理等，这些约束使其有别于先前基准测试所针对的领域。我们已实现并被采纳的 MLPerf Automotive 基准测试是一个评估框架和方法论，能够以可复现的性能指标对汽车系统进行基准测试。该基准测试由汽车感知任务组成，涵盖 2D 目标检测、2D 语义分割、3D 目标检测、端到端驾驶等任务。

    arXiv:2510.27065v2 Announce Type: replace  Abstract: We present MLPerf Automotive, the first standardized public performance benchmark for evaluating Machine Learning systems that are deployed for AI acceleration in automotive systems. Developed through a collaborative partnership within MLCommons, this benchmark addresses the need for standardized performance evaluation methodologies in automotive machine learning systems. Existing benchmark suites cannot be utilized for these systems since automotive workloads have unique constraints including sensor suites, safety, and real-time processing that distinguish them from the domains that previously introduced benchmarks target. Our implemented and adopted MLPerf Automotive benchmark is a framework for evaluation and methodology for benchmarking automotive systems with reproducible performance metrics. The benchmark consists of automotive perception tasks in 2D object detection, 2D semantic segmentation, 3D object detection, end-to-end dr
    
[^314]: 在扩散语言模型中实现近似联合采样

    Enabling Approximate Joint Sampling in Diffusion LMs

    [https://arxiv.org/abs/2509.22738](https://arxiv.org/abs/2509.22738)

    本文提出在现有大型扩散语言模型之上附加一个轻量级单层“采样器”，使模型能够在一次前向传播中近似地从真实联合分布并行采样多个 token，从而在保持准确率的同时大幅提升生成速度。

    

    在自回归语言模型中，每个 token 的采样都以之前所有 token 为条件，因此整个字符串可以看作是从模型所表示的正确底层联合分布中采样得到的。相比之下，掩码扩散语言模型通过乱序且可能并行地解除 token 掩码来生成文本。要让整个字符串再次从正确的底层联合分布中采样，就需要在每次完整模型前向传播中恰好只解除一个 token 的掩码。并行解除掩码的 token 数量越多，字符串就偏离真实联合分布越远；这一点可以从准确率的下降（以及速度的提升）中观察到。在本文中，我们设计了一种方法，可以在单次完整模型前向传播中从联合分布中近似地采样多个 token；为此，我们在现有的大型扩散语言模型之上构建了一个新的轻量级单层“采样器”。

    arXiv:2509.22738v3 Announce Type: replace  Abstract: In autoregressive language models, each token is sampled by conditioning on all the past tokens; the overall string has thus been sampled from the correct underlying joint distribution represented by the model. In contrast, masked diffusion language models generate text by unmasking tokens out of order and potentially in parallel. Generating an overall string sampled from the correct underlying joint distribution would (again) require exactly one token unmasking in every full-model forward pass. The more tokens unmasked in parallel, the further away the string is from the true joint; this can be seen in the resulting drop in accuracy (but, increase in speed). In this paper we devise a way to {\em approximately} sample multiple tokens from the joint distribution in a single full-model forward pass; we do so by developing a new lightweight single-layer ``sampler" on top of an existing large diffusion LM. One forward pass of the full mo
    
[^315]: 通过模块社区揭示大语言模型的认知模式

    Unraveling the cognitive patterns of Large Language Models through module communities

    [https://arxiv.org/abs/2508.18192](https://arxiv.org/abs/2508.18192)

    该研究借鉴生物认知系统的分析方法，开发了一个连接认知技能、LLM架构和数据集的基于网络的框架，通过模块社区分析揭示了大语言模型展现出独特的模块组织结构，其涌现的技能模式部分类似于生物系统的认知特化机制。

    

    大语言模型（LLMs）通过从科学发现、医学诊断到聊天机器人等广泛应用，在科学、工程和社会领域取得了重大进展，重塑了我们的世界。尽管它们无处不在且功能强大，但LLM的底层机制仍隐藏在数十亿参数和复杂结构之中，使其内部架构和认知过程难以理解。我们通过借鉴理解生物系统中新兴认知的方法来填补这一空白，开发了一个连接认知技能、LLM架构和数据集的基于网络的框架，开创了基础模型分析的新范式。模块社区中的技能分布表明，虽然LLM并不严格对应于特定生物系统中所观察到的聚焦特化现象，但它们表现出独特的模块社区，其涌现的技能模式部分模仿了生物学中的认知组织方式。

    arXiv:2508.18192v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have reshaped our world with significant advancements in science, engineering, and society through applications ranging from scientific discoveries and medical diagnostics to Chatbots. Despite their ubiquity and utility, the underlying mechanisms of LLM remain concealed within billions of parameters and complex structures, making their inner architecture and cognitive processes challenging to comprehend. We address this gap by adopting approaches to understanding emerging cognition in biology and developing a network-based framework that links cognitive skills, LLM architectures, and datasets, ushering in a paradigm shift in foundation model analysis. The skill distribution in the module communities demonstrates that while LLMs do not strictly parallel the focalized specialization observed in specific biological systems, they exhibit unique communities of modules whose emergent skill patterns partially mi
    
[^316]: 堆叠SVD还是SVD堆叠？随机矩阵理论视角下的数据整合

    Stacked SVD or SVD stacked? A Random Matrix Theory perspective on data integration

    [https://arxiv.org/abs/2507.22170](https://arxiv.org/abs/2507.22170)

    本文借助随机矩阵理论，首次在比例渐近区间下严格比较了Stack-SVD与SVD-Stack这两种估计多数据集共享奇异子空间的主流数据整合方法的理论性能。

    

    arXiv:2507.22170v2 Announce Type: replace-cross 摘要：现代数据分析日益需要在多个高维数据集中识别共享的潜在结构。一个常用的模型假设数据矩阵是具有共享奇异子空间的低秩矩阵的含噪观测。在此情况下，出现了两种用于估计该共享结构的主要方法，它们在跨数据集整合信息的方式上有所不同。第一种方法称为Stack-SVD，它将所有数据集拼接在一起，然后执行奇异值分解（SVD）。第二种方法称为SVD-Stack，它首先对每个数据集分别执行SVD，然后聚合这些数据集的顶部奇异向量，最后计算它们之间的一致性。尽管这些方法被广泛使用，但它们尚未在比例渐近区间（proportional asymptotic regime）下得到严格研究，而在当今数据规模和维度不断增长的世界中，这一区间具有重要的实际意义。

    arXiv:2507.22170v2 Announce Type: replace-cross  Abstract: Modern data analysis increasingly requires identifying shared latent structure across multiple high-dimensional datasets. A commonly used model assumes that the data matrices are noisy observations of low-rank matrices with a shared singular subspace. In this case, two primary methods have emerged for estimating this shared structure, which vary in how they integrate information across datasets. The first approach, termed Stack-SVD, concatenates all the datasets, and then performs a singular value decomposition (SVD). The second approach, termed SVD-Stack, first performs an SVD separately for each dataset, then aggregates the top singular vectors across these datasets, and finally computes a consensus amongst them. While these methods are widely used, they have not been rigorously studied in the proportional asymptotic regime, which is of great practical relevance in today's world of increasing data size and dimensionality. Con
    
[^317]: 通过依赖感知生成建模捕捉未见的空间热极端事件

    Capturing Unseen Spatial Heat Extremes Through Dependence-Aware Generative Modeling

    [https://arxiv.org/abs/2507.09211](https://arxiv.org/abs/2507.09211)

    DeepX-GAN是一种显式捕捉空间依赖性的深度生成模型，能够零样本模拟超出历史记录的统计上合理的“未见”热极端事件，揭示多个地点同时遭受极端高温的隐藏风险。

    

    观测到的气候极端事件记录为潜在灾害提供了不完整的视角，遗漏了超出历史经验的“未见”事件。忽视空间依赖性进一步低估了同时袭击多个地点的灾害风险。我们提出了DeepX-GAN（物理极端事件依赖增强嵌入-生成对抗网络），这是一种明确捕捉罕见极端事件空间结构的深度生成模型。其零样本泛化能力使其能够模拟超出观测记录的统计上合理的极端事件，并通过长期气候模式大集合模拟进行评估。我们定义了两种“未见”类型：直接影响目标地区的“直接命中”型极端事件，以及险些错过目标地区的“擦边”型极端事件。这些未实现的事件揭示了隐藏的风险，既可以促使人们采取主动适应措施，也可能强化一种虚假的抗灾安全感。将DeepX-GAN应用于中东和北非地区的结果表明，概率……（原文摘要在此处截断）

    arXiv:2507.09211v3 Announce Type: replace  Abstract: Observed records of climate extremes provide an incomplete view of plausible hazards, missing "unseen" events beyond historical experience. Ignoring spatial dependence further underestimates hazards striking multiple locations simultaneously. We introduce DeepX-GAN (Dependence-Enhanced Embedding for Physical eXtremes-Generative Adversarial Network), a deep generative model that explicitly captures the spatial structure of rare extremes. Its zero-shot generalizability enables the simulation of statistically plausible extremes beyond the observed record, evaluated against long climate model large-ensemble simulations. We define two unseen types: direct-hit extremes that affect the target, and near-miss extremes that narrowly miss. These unrealized events reveal hidden risks and can either prompt proactive adaptation or reinforce a false sense of resilience. Applying DeepX-GAN to the Middle East and North Africa shows that the probabili
    
[^318]: 面向向量搜索的图索引之核方法

    The kernel of graph indices for vector search

    [https://arxiv.org/abs/2506.20584](https://arxiv.org/abs/2506.20584)

    提出了基于核方法的支持向量图（SVG）这一新型图索引，首次为度量和非度量向量空间中的向量搜索提供形式化可导航性保证，并将 HNSW、DiskANN 等流行图索引统一解释为它的特例。

    

    向量搜索中最流行的图索引使用计算几何原理来构建图。因此，它们的形式化图可导航性保证仅在与欧几里得空间中有效。在这项工作中，我们展示了机器学习可以用于在度量和非度量向量空间（例如内积相似度）中构建向量搜索的图索引。从这个新颖的视角出发，我们提出了支持向量图（SVG），这是一种新型的图索引，它利用核方法来建立图的连接性，并且带有在度量和非度量向量空间中均有效的形式化可导航性保证。此外，我们将最流行的图索引（包括 HNSW 和 DiskANN）解释为 SVG 的特定特化形式，并表明可以从这种特化背后的原理推导出新的可导航索引。最后，我们提出了 SVG-L0，它将 ℓ0 稀疏性约束融入 SVG 的 k 近邻构建过程中。

    arXiv:2506.20584v3 Announce Type: replace  Abstract: The most popular graph indices for vector search use principles from computational geometry to build the graph. Hence, their formal graph navigability guarantees are only valid in Euclidean space. In this work, we show that machine learning can be used to build graph indices for vector search in metric and non-metric vector spaces (e.g., for inner product similarity). From this novel perspective, we introduce the Support Vector Graph (SVG), a new type of graph index that leverages kernel methods to establish the graph connectivity and that comes with formal navigability guarantees valid in metric and non-metric vector spaces. In addition, we interpret the most popular graph indices, including HNSW and DiskANN, as particular specializations of SVG and show that new navigable indices can be derived from the principles behind this specialization. Finally, we propose SVG-L0 that incorporates an $\ell_0$ sparsity constraint into the SVG k
    
[^319]: 基于扩散模型辅助的面向任务语义通信与模型反演攻击

    Diffusion-aided Task-oriented Semantic Communications with Model Inversion Attack

    [https://arxiv.org/abs/2506.19886](https://arxiv.org/abs/2506.19886)

    该论文针对任务保密场景下的模型反演攻击问题，提出一种扩散模型辅助的面向任务语义通信方法，防止攻击者在不知晓接收方任务与模型的情况下从语义特征中重建原始输入，并指出仅用PSNR/SSIM评估隐私泄露并不充分，需在保证任务准确性的同时兼顾隐私保护。

    

    语义通信通过传输语义信息而非原始输入符号序列来提升传输效率。面向任务的语义通信进一步致力于仅保留与任务相关的信息，从而实现更大的带宽节省。然而，这类基于神经网络的通信系统容易受到模型反演攻击，即攻击者试图从截获的语义特征中恢复敏感的输入信息。因此，关键挑战在于在保持任务准确性和鲁棒性的同时保护隐私。我们考虑了一种任务保密的场景，其中攻击者在不知道合法接收方任务或模型的情况下，试图从截获的特征中重建原始输入。尽管PSNR和SSIM通常被用于评估重建质量，我们发现外部分类器仍然能够以可观的准确率完成合法接收方的任务。

    arXiv:2506.19886v3 Announce Type: replace-cross  Abstract: Semantic communication enhances transmission efficiency by conveying semantic information rather than raw input symbol sequences. Task-oriented semantic communication further aims to retain only task-specific information, thereby achieving greater bandwidth savings. However, these neural-network-based communication systems are vulnerable to model inversion attacks, in which adversaries attempt to recover sensitive input information from intercepted semantic features. The key challenge is therefore to preserve privacy while maintaining task accuracy and robustness. We consider a task-confidential setting in which the adversary attempts to reconstruct the original input from intercepted features without knowing the legitimate receiver's task or model. Although PSNR and SSIM are commonly used to assess reconstruction quality, we find that an external classifier can still perform the legitimate receiver's task with nontrivial accur
    
[^320]: 多模态人工智能从临床前数据预测药物组合的临床结果

    Multimodal AI predicts clinical outcomes of drug combinations from preclinical data

    [https://arxiv.org/abs/2503.02781](https://arxiv.org/abs/2503.02781)

    本文提出多模态AI模型Madrigal，通过将分子结构、通路、细胞活力和转录组学数据对齐到共享潜在空间，实现了从临床前数据预测药物组合临床结果，性能优于现有方法。

    

    从临床前数据预测临床结果对于选择安全有效的药物组合以及减少临床试验后期失败至关重要。现有的AI模型使用分子结构和靶点注释信息，但并未利用能够反映化合物在细胞环境中作用方式的扰动读出数据。本文提出了Madrigal，这是一个多模态AI模型，能够从分子结构、通路、细胞活力和转录组学数据中学习。Madrigal将21,842个化合物的各模态数据对齐到共享潜在空间中，即使对于仅在部分数据模态中观察到的药物也能预测组合结果。该模型基于158个专家精选和795个患者报告的药物组合结果进行训练，其性能超越了单模态方法和最先进的多模态方法。消融实验表明，模态对齐和多模态输入各自都能提高预测性能。Madrigal能够预测共享膜...（摘要在此处被截断）

    arXiv:2503.02781v3 Announce Type: replace-cross  Abstract: Predicting clinical outcomes from preclinical data is essential for selecting safe and effective drug combinations and for reducing late-stage failures. AI models use molecular structure and target annotations, and do not leverage the perturbation readouts that report how a compound acts in a cellular context. Here we introduce Madrigal, a multimodal AI model that learns from structural, pathway, cell-viability, and transcriptomic data. Madrigal aligns these modalities across 21,842 compounds into a shared latent space and predicts combination outcomes even for drugs observed in only a subset of the data modalities. Trained on 158 expert-curated and 795 patient-reported combination outcomes, Madrigal outperforms single-modality and state-of-the-art multimodal methods. Ablations show that modality alignment and multimodal input each improve predictive performance. Madrigal predicts elevated risk for combinations that share membr
    
[^321]: 无需节拍器的时变贝叶斯优化

    Time-Varying Bayesian Optimization Without a Metronome

    [https://arxiv.org/abs/2501.18963](https://arxiv.org/abs/2501.18963)

    该论文首次推导出显式考虑观测采样频率变化的时变贝叶斯优化遗憾上界，并据此提出了关于数据集规模和过期数据策略的实用建议，其 BOLT 算法在实验中优于现有最先进的 TVBO 方法。

    

    时变贝叶斯优化（TVBO）是优化时变的、昂贵的、含噪声的黑盒函数 $f$ 的首选框架。然而，大多数 TVBO 算法所提供的渐近保证都依赖于“观测以恒定频率获取”这一假设。由于高斯过程（GP）推断的复杂度随数据集规模呈三次方增长，这一假设从长远来看是不现实的。在本文中，我们放宽了这一假设，并推导出了首个显式考虑观测采样频率变化的上界遗憾界。基于这一分析，我们提出了关于 TVBO 算法数据集规模和过期数据策略的实用建议。我们通过对遵循这些建议的算法 BOLT 在合成问题和真实世界问题上的实验，展示了其性能优于当前最先进的 TVBO 方法。

    arXiv:2501.18963v4 Announce Type: replace-cross  Abstract: Time-Varying Bayesian Optimization (TVBO) is the go-to framework for optimizing a time-varying, expensive, noisy black-box function $f$. However, most of the asymptotic guarantees offered by TVBO algorithms rely on the assumption that observations are acquired at a constant frequency. As the GP inference complexity scales with the cube of its dataset size, this assumption is unrealistic in the long run. In this paper, we relax this assumption and derive the first upper regret bound that explicitly accounts for changes in the observations sampling frequency. Based on this analysis, we formulate practical recommendations about dataset sizes and stale data policies of TVBO algorithms. We illustrate how an algorithm (BOLT) that follows these recommendations performs better than the state-of-the-art of TVBO through experiments on synthetic and real-world problems.
    
[^322]: 大语言模型基础

    Foundations of Large Language Models

    [https://arxiv.org/abs/2501.09223](https://arxiv.org/abs/2501.09223)

    本书系统阐述了大语言模型的六大核心基础领域——预训练、生成模型、提示、对齐、推断与推理，为学习者提供了一部权威的基础性参考书。

    

    这是一本关于大语言模型的书籍。正如书名所示，本书主要聚焦于基础性概念，而非全面涵盖所有前沿技术。全书由六个主要章节构成，每个章节探讨一个关键领域：预训练、生成模型、提示（Prompting）、对齐、推断（Inference）和推理（Reasoning）。本书面向大学生、自然语言处理及相关领域的专业人士和从业者，也可作为所有对大语言模型感兴趣的读者的参考书。

    arXiv:2501.09223v3 Announce Type: replace-cross  Abstract: This is a book about large language models. As indicated by the title, it primarily focuses on foundational concepts rather than comprehensive coverage of all cutting-edge technologies. The book is structured into six main chapters, each exploring a key area: pre-training, generative models, prompting, alignment, inference, and reasoning. It is intended for college students, professionals, and practitioners in natural language processing and related fields, and can serve as a reference for anyone interested in large language models.
    
[^323]: AIR：用于持续学习的解析式不平衡矫正器

    AIR: Analytic Imbalance Rectifier for Continual Learning

    [https://arxiv.org/abs/2408.10349](https://arxiv.org/abs/2408.10349)

    该论文提出了面向真实世界持续学习的解析式不平衡矫正器（AIR），通过冻结主干网络、闭式解增量分类器以及解析式重加权模块（ARM），有效解决了不平衡数据流中的类别不平衡问题。

    

    持续学习（CL）智能体从按顺序到达的数据中进行增量学习，并适应真实世界环境动态、不断变化的特性。然而，许多现有的持续学习方法在不断演变的不平衡数据流中出现性能下降，其原因在于对变化类别的适应能力有限，或者无法有效利用来自新类别和先前观测类别的混合数据。为应对这些挑战，我们提出了一种面向真实世界持续学习的解析式不平衡矫正器（AIR）算法。AIR是一种在线的无样本（exemplar-free）方法，采用冻结的主干网络作为特征提取器，并使用闭式解的增量分类器，其权重等于针对同一类加权岭回归目标的联合学习权重。AIR通过一个解析式重加权模块（ARM）来解决类别不平衡问题，该模块为损失函数中的每个类别计算重加权因子，以均衡各类别的总样本权重。在长尾类增量（摘要内容不完整，原文在此处截断）

    arXiv:2408.10349v2 Announce Type: replace  Abstract: Continual learning (CL) agents incrementally learn from sequentially arriving data and adapt to the dynamic, ever-changing nature of real-world environments. However, many existing CL methods suffer performance degradation in evolving, imbalanced data streams due to limited adaptation to changing class frequencies or ineffective use of mixed data from new and previously observed classes. To deal with these challenges, we propose an analytic imbalance rectifier (AIR) algorithm for real-world CL. AIR is an online exemplar-free approach with a frozen backbone as the feature extractor and a closed-form incremental classifier whose weight equals the joint-learning weight for the same class-weighted ridge objective. AIR addresses class imbalance with an analytic reweighting module (ARM) that calculates a reweighting factor for each class in the loss function to equalize total sample weights across classes. Under long-tailed class-increment
    
[^324]: 直推式离策略近端策略优化

    Transductive Off-policy Proximal Policy Optimization

    [https://arxiv.org/abs/2406.03894](https://arxiv.org/abs/2406.03894)

    本文提出ToPPO，通过新颖的策略改进下界表述和计算高效的优化机制，将离策略数据安全地引入PPO训练，在保证单调改进的同时显著提升了PPO利用异策略数据的能力。

    

    近端策略优化（PPO）是一种流行的无模型强化学习算法，因其简洁性和有效性而备受推崇。然而，由于其固有的在策略特性，它在利用来自不同策略的数据方面的能力受到限制。本文提出了一种对原始PPO方法的新型离策略扩展，命名为直推式离策略PPO（ToPPO）。在此，我们为在PPO训练中引入离策略数据提供了理论依据，并为其安全应用提供了审慎的指导原则。我们的贡献包括：针对由离策略数据导出的候选策略，提出了策略改进下界的新颖表述，并配备了一种计算高效的机制来优化该下界，同时以单调改进的保证作为支撑。在六个代表性任务上的全面实验结果凸显了ToPPO的良好性能表现。

    arXiv:2406.03894v2 Announce Type: replace  Abstract: Proximal Policy Optimization (PPO) is a popular model-free reinforcement learning algorithm, esteemed for its simplicity and efficacy. However, due to its inherent on-policy nature, its proficiency in harnessing data from disparate policies is constrained. This paper introduces a novel off-policy extension to the original PPO method, christened Transductive Off-policy PPO (ToPPO). Herein, we provide theoretical justification for incorporating off-policy data in PPO training and prudent guidelines for its safe application. Our contribution includes a novel formulation of the policy improvement lower bound for prospective policies derived from off-policy data, accompanied by a computationally efficient mechanism to optimize this bound, underpinned by assurances of monotonic improvement. Comprehensive experimental results across six representative tasks underscore ToPPO's promising performance.
    
[^325]: 针对受污染无标签数据的深度正-无标签异常检测

    Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data

    [https://arxiv.org/abs/2405.18929](https://arxiv.org/abs/2405.18929)

    提出了一种将正-无标签学习与自编码器、深度支持向量数据描述等深度异常检测模型相结合的深度正-无标签异常检测框架，以应对无标签数据被异常污染的现实情况，从而提升半监督异常检测的性能。

    

    半监督异常检测旨在通过在无标签数据之外利用少量有标签异常数据来提升异常检测性能，因而受到了广泛关注。现有的半监督方法假设大部分无标签数据是正常的，并通过最小化无标签数据的异常分数、同时最大化有标签异常数据的异常分数来训练异常检测器。然而，在实际应用中，无标签数据往往被异常数据所污染。这削弱了最大化异常分数这一操作的效果，从而阻碍了检测性能的提升。为了解决这一问题，我们提出了深度正-无标签异常检测框架，该框架将正-无标签学习与自编码器、深度支持向量数据描述等深度异常检测模型相结合。我们的方法能够利用无标签数据来近似正常数据的异常分数……

    arXiv:2405.18929v3 Announce Type: replace-cross  Abstract: Semi-supervised anomaly detection, which aims to improve the anomaly detection performance by using a small amount of labeled anomaly data in addition to unlabeled data, has attracted attention. Existing semi-supervised approaches assume that most unlabeled data are normal, and train anomaly detectors by minimizing the anomaly scores for the unlabeled data while maximizing those for the labeled anomaly data. However, in practice, the unlabeled data are often contaminated with anomalies. This weakens the effect of maximizing the anomaly scores for anomalies, and prevents us from improving the detection performance. To solve this, we propose the deep positive-unlabeled anomaly detection framework, which integrates positive-unlabeled learning with deep anomaly detection models such as autoencoders and deep support vector data descriptions. Our approach enables the approximation of anomaly scores for normal data using the unlabeled
    
[^326]: 利用知识图谱和大语言模型生成有趣的科学想法：基于100位研究团队负责人的评估

    Generating Interesting Scientific Ideas using Knowledge Graphs and LLMs: Evaluations with 100 Research Group Leaders

    [https://arxiv.org/abs/2405.17044](https://arxiv.org/abs/2405.17044)

    该研究提出SciMuse系统，利用包含5800万篇论文的知识图谱结合大语言模型生成个性化研究想法，并通过100多位研究团队负责人对4400多个想法的大规模评估发现，专家整体兴趣评分虽保守（均值2.40/5），但近四分之一的想法获得了高分认可。

    

    科学文献的快速增长使研究人员越来越难以发现有新颖性和影响力的想法，尤其是在跨学科领域。现代人工智能（AI）系统为科学构思提供了新的机遇，但AI生成的想法究竟有多大吸引力，以及如何提升其质量？在此，我们提出了SciMuse，它利用一个包含5800万篇论文的知识图谱和大语言模型（LLM）来生成个性化的研究想法。这项工作的核心重点是探究这些想法的有趣程度。为此，我们开展了一项大规模评估，邀请100多位研究团队负责人——涵盖从自然科学到人文学科——根据兴趣程度对4400多个个性化想法进行评分。总体而言，专家评分较为保守（5分制中平均分为2.40分，最常见评分为1分），但也有24.9%的想法获得了4分或5分。我们发现提供……

    arXiv:2405.17044v4 Announce Type: replace  Abstract: The rapid growth of scientific literature makes it increasingly challenging for researchers to identify novel and impactful ideas, especially across disciplines. Modern artificial intelligence (AI) systems offer new opportunities for scientific ideation, but how compelling are AI-generated ideas, and how can their quality be improved? Here, we introduce SciMuse, which generates personalized research ideas using a knowledge graph of 58 million papers and a large language model (LLM). A central focus of this work is to understand how interesting these ideas are. Therefore, we conducted a large-scale evaluation in which more than 100 research group leaders -- spanning the natural sciences to the humanities -- rated over 4,400 personalized ideas according to their level of interest. Overall, expert ratings were modest (mean 2.40 on a 5-point scale, most common rating 1), while 24.9% of ideas were rated 4 or 5. We find that supplying conc
    
[^327]: 一种基于概率的人类比较对齐方法

    A Probabilistic Approach for Alignment with Human Comparisons

    [https://arxiv.org/abs/2403.10771](https://arxiv.org/abs/2403.10771)

    通过提出的两阶段“监督微调+人类比较”框架，本文研究了如何有效利用人类比较来改善AI模型的对齐，特别是在面对嘈杂数据和高维模型时。

    

    一个增长的趋势是将人类知识整合到学习框架中，利用微妙的人类反馈来完善AI模型。尽管取得了这些进展，但尚未开发出描述人类比较何时改善传统监督微调过程的特定条件的全面理论框架。为弥补这一差距，本文研究了有效利用人类比较来解决由嘈杂数据和高维模型引起的限制。我们提出了一个将机器学习与人类反馈通过概率二分方法联系起来的两阶段“监督微调+人类比较”（SFT+HC）框架。这两阶段框架首先通过SFT过程从带有噪声标记的数据中学习低维表示，然后利用人类比较来改进模型对齐。为了检验对齐阶段的效力，我们引入了一个新概念，称为“标签噪声到一致性”

    arXiv:2403.10771v1 Announce Type: new  Abstract: A growing trend involves integrating human knowledge into learning frameworks, leveraging subtle human feedback to refine AI models. Despite these advances, no comprehensive theoretical framework describing the specific conditions under which human comparisons improve the traditional supervised fine-tuning process has been developed. To bridge this gap, this paper studies the effective use of human comparisons to address limitations arising from noisy data and high-dimensional models. We propose a two-stage "Supervised Fine Tuning+Human Comparison" (SFT+HC) framework connecting machine learning with human feedback through a probabilistic bisection approach. The two-stage framework first learns low-dimensional representations from noisy-labeled data via an SFT procedure, and then uses human comparisons to improve the model alignment. To examine the efficacy of the alignment phase, we introduce a novel concept termed the "label-noise-to-co
    
[^328]: 使用超图表示学习重建短寿命粒子

    Reconstructing short-lived particles using hypergraph representation learning

    [https://arxiv.org/abs/2402.10149](https://arxiv.org/abs/2402.10149)

    提出了一种基于超图表示学习的图神经网络架构HyPER，能够高效重建对撞机事例中的短寿命母粒子，其性能优于现有最先进技术且参数效率更高。

    

    在对撞机实验中，重质量短寿命粒子的运动学重建对于标准模型的精确检验以及超越标准模型的新物理搜寻至关重要。在具有许多末态喷注的对撞机事例（例如正反顶夸克对的全强子衰变）中进行运动学重建是极具挑战性的。我们提出了HyPER（Hypergraph for Particle Event Reconstruction，超图粒子事例重建）：一种基于图神经网络的新型架构，利用超图表示学习来构建更强大、更高效的对撞机事例表示。HyPER用于从末态物体集合中重建母粒子。经过模拟数据的训练和测试，HyPER模型的表现优于现有的最先进重建技术，同时展现出卓越的参数效率。这种新颖的超图方法使该方法能够应用于多喷注事例中的粒子重建。

    arXiv:2402.10149v3 Announce Type: cross  Abstract: In collider experiments, the kinematic reconstruction of heavy, short-lived particles is vital for precision tests of the Standard Model and in searches for physics beyond it. Performing kinematic reconstruction in collider events with many final-state jets, such as the all-hadronic decay of top-antitop quark pairs, is challenging. We present HyPER: Hypergraph for Particle Event Reconstruction, a novel architecture based on graph neural networks that uses hypergraph representation learning to build more powerful and efficient representations of collider events. HyPER is used to reconstruct parent particles from sets of final-state objects. Trained and tested on simulation, the HyPER model is shown to perform favorably when compared to existing state-of-the-art reconstruction techniques, while demonstrating superior parameter efficiency. The novel hypergraph approach allows the method to be applied to particle reconstruction in a multit
    
[^329]: Calpric：利用众包与主动学习实现隐私政策的包容性细粒度标注

    Calpric: Inclusive and Fine-grain Labeling of Privacy Policies with Crowdsourcing and Active Learning

    [https://arxiv.org/abs/2008.02954](https://arxiv.org/abs/2008.02954)

    该论文提出Calpric框架，通过结合自动文本分割、众包标注和主动学习，以低成本高效生成大规模、高质量的隐私政策训练数据集，使未经训练的众包标注者能达到与专业标注者相当的标注质量。

    

    在隐私政策上训练准确的深度学习模型面临的一个重大挑战是获取大量且全面的训练数据的成本和难度。为了解决这些挑战，我们提出了Calpric，它结合了自动文本选择与分割、主动学习以及众包标注人员的使用，以低成本为隐私政策生成大规模、均衡的训练集。自动化的文本选择与分割简化了标注任务，使来自众包平台（如亚马逊Mechanical Turk）的未经训练的标注人员能够与受过训练的标注人员（如法学院学生）相媲美，同时减少了标注者之间的分歧，从而降低了标注成本。拥有可靠的训练标签使得主动学习得以应用，主动学习使用更少的训练样本即可高效覆盖输入空间，进一步降低了成本，并改善了数据中的类别和数据类别平衡。

    arXiv:2008.02954v2 Announce Type: replace-cross  Abstract: A significant challenge to training accurate deep learning models on privacy policies is the cost and difficulty of obtaining a large and comprehensive set of training data. To address these challenges, we present Calpric, which combines automatic text selection and segmentation, active learning and the use of crowdsourced annotators to generate a large, balanced training set for privacy policies at low cost. Automated text selection and segmentation simplify the labeling task, enabling untrained annotators from crowdsourcing platforms, like Amazon's Mechanical Turk, to be competitive with trained annotators, such as law students, and also reduce inter-annotator disagreement, which decreases labeling cost. Having reliable labels for training enables the use of active learning, which uses fewer training samples to efficiently cover the input space, further reducing cost and improving class and data category balance in the data s
    
[^330]: CatSIM：一种分类图像相似度度量

    CatSIM: A Categorical Image Similarity Metric

    [https://arxiv.org/abs/2004.09073](https://arxiv.org/abs/2004.09073)

    CatSIM是一种基于结构相似性范式的新图像相似度度量方法，适用于二值和多值的二维及三维图像与体积，对位置的微小扰动具有鲁棒性，并能比较图像内部的任意区域。

    

    我们提出了CatSIM，这是一种用于二值和多值二维及三维图像和体积的新相似度度量方法。CatSIM采用结构相似性图像质量范式，并对位置的微小扰动具有鲁棒性，因此位于相似但不完全重叠的图像或体积区域中的结构，能够获得比简单匹配更高的评分。该度量方法还可以比较图像和体积内部的任意区域。CatSIM在人工数据集上进行了评估，并通过两个独立的图像质量评估调查与人类感知进行对比验证，同时在两个数据集上进行了应用展示。公开可用的R包catsim实现了该方法。

    arXiv:2004.09073v2 Announce Type: replace-cross  Abstract: We introduce CatSIM, a new similarity metric for binary and multinary two- and three-dimensional images and volumes. CatSIM uses a structural similarity image quality paradigm and is robust to small perturbations in location so that structures in similar, but not entirely overlapping, image or volumetric regions are rated higher than by simple matching. The metric can also compare arbitrary regions inside images and volumes. CatSIM is evaluated on artificial data sets, validated by comparing with human perception in two separate image quality assessment surveys, and illustrated on two datasets. The publicly available R package \texttt{catsim} implements the methodology.
    
[^331]: DCRMTA: 无偏的多触点归因的因果表示

    DCRMTA: Unbiased Causal Representation for Multi-touch Attribution. (arXiv:2401.08875v1 [cs.LG])

    [http://arxiv.org/abs/2401.08875](http://arxiv.org/abs/2401.08875)

    DCRMTA提出了一种无偏的多触点归因方法，通过建立转化预测模型和构建对照触点序列来减轻偏差的影响。

    

    多触点归因（MTA）在实现对每个广告触点对于转化行为的贡献的公正估计方面起着关键作用，深刻影响预算分配和广告推荐。传统的多触点归因方法首先构建一个转化预测模型，通过历史数据学习触点序列和用户购买行为之间的内在关系。在此基础上，从原始序列子集中构建对照触点序列，并使用预测模型估计转化，从而计算广告贡献。这些方法的一个隐含假设是转化预测模型的无偏性。然而，由于用户偏好和互联网推荐机制（如过去的购物记录导致的广告推荐同质化）引起的混杂变量因素，转化中很容易产生偏差。

    Multi-touch attribution (MTA) currently plays a pivotal role in achieving a fair estimation of the contributions of each advertising touchpoint to-wards conversion behavior, deeply influencing budget allocation and advertising recommenda-tion. Traditional multi-touch attribution methods initially build a conversion prediction model, an-ticipating learning the inherent relationship be-tween touchpoint sequences and user purchasing behavior through historical data. Based on this, counterfactual touchpoint sequences are con-structed from the original sequence subset, and conversions are estimated using the prediction model, thus calculating advertising contributions. A covert assumption of these methods is the un-biased nature of conversion prediction models. However, due to confounding variables factors arising from user preferences and internet recom-mendation mechanisms such as homogenization of ad recommendations resulting from past shop-ping records, bias can easily occur in conversi
    

