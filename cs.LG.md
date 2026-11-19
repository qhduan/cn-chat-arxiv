# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowing What You Know: A Trait Theory of Reviewer Value in Human-AI Cognitive Systems](https://arxiv.org/abs/2611.24932) | 该论文提出“元认知校准”（审阅者信心与其判断实际正确性的一致性）是决定人类审阅者在人机协作认知系统中核心价值的唯一关键特质，并主张联合系统应将每个决策条目的最终裁定权交给校准程度更高的智能体——无论是人还是模型。 |
| [^2] | [Decomposing SGD Dynamics in Neural Networks: Teacher-Induced Spikes and Variance Inflation](https://arxiv.org/abs/2611.24908) | 该论文通过将第一层权重分解为信号分量与正交补空间，揭示了SGD训练会在正交补空间的Gram矩阵中诱导出依赖教师的谱尖峰，从而放大表示方差并导致泛化性能退化的具体机制。 |
| [^3] | [Equivariant Modeling with Polynomial Features via Minimal Homogeneous Generators](https://arxiv.org/abs/2611.24896) | 该论文提出了一个基于最小齐次生成元构造等变多项式特征的通用等变函数建模框架，并提供了可自动计算特征与拟合模型的Python工具包，同时给出了更高效的特征计算算法。 |
| [^4] | [On Evaluating User Simulators for Interactive Agents](https://arxiv.org/abs/2611.24892) | 该论文对用户模拟器的评估指标本身进行元评估，发现现有方法无法可靠检测模拟用户行为中的多数故障，并据此提出了一种新的评估方法。 |
| [^5] | [Cap-and-Track: Optimizing LLM Inference Request Dispatch Across Shifting Capacity](https://arxiv.org/abs/2611.24891) | Cap-and-Track利用饱和资源处的快速失败拒绝信号驱动分发器自动向上探测和退避，无需预测输出长度或手动调整集群级速率即可跟随动态变化的容量上限，并在生产A/B测试中以少43%的H100 GPU服务了相同流量。 |
| [^6] | [Monoculture Robust Learning](https://arxiv.org/abs/2611.24881) | 该论文在PAC学习中提出了“单一文化鲁棒性”概念，并揭示了可复现性与单一文化鲁棒性之间的惊人联系，使仅拥有单一数据集的学习器能输出 $k$ 个联合失败概率可与 $k$ 个独立数据集训练相媲美的准确假设，从而缓解算法单一文化问题。 |
| [^7] | [Auditing black-box clinical prediction models with a large language model through queries alone: a blinded test with planted flaws](https://arxiv.org/abs/2611.24877) | 该论文在盲测条件下证明，大语言模型可以仅通过查询黑箱临床预测模型来构建高保真代理模型并检出人为植入的缺陷（如反向风险因素、被低估的亚组和不合理阶跃）。 |
| [^8] | [Learning to Interact in Multi-Agent Self-Play for Controllable Autonomous Driving Scenario Generation](https://arxiv.org/abs/2611.24876) | 该论文提出引入智能体相对动态目标的扩展多智能体自博弈框架，实现了对攻击性尾随、对抗性切入等安全关键交互驾驶场景的精准可控生成，解决了静态目标方法无法模拟复杂智能体交互的问题。 |
| [^9] | [TangentPDE: Learning Local Operator Responses with Sparse Finite-Difference Supervision](https://arxiv.org/abs/2611.24873) | 提出TangentPDE方法，仅利用参考求解器在稀疏成对输入处的状态输出构造有限差分响应监督，无需导数标签、伴随求解或完整雅可比矩阵，即可同时学习神经PDE算子的取值及其局部变化。 |
| [^10] | [Atomic MathReasoner: Instance-Level Composition of Atomic Reasoning Skills for Mathematical LLMs](https://arxiv.org/abs/2611.24846) | 提出了 Atomic MathReasoner（AMR），一种实例级框架：通过训练能力专用的 LoRA 专家库并用无梯度优化在验证子集上发现低损失组合原型，再依据参考解答的词元级负对数似然为每个数学问题分配最合适的原子推理技能组合，从而有效利用原子能力提升数学大语言模型的推理表现。 |
| [^11] | [Measuring and Improving Transfer of Neural Surrogates for Vehicle Aerodynamics](https://arxiv.org/abs/2611.24843) | 该论文通过大规模受控基准测试首次系统评估了车辆空气动力学神经代理模型的跨数据集迁移行为，发现微调仅在不到一半的情况下优于从头训练，且会损害模型在预训练域上的准确性。 |
| [^12] | [Does the Action Matter? Probing Action-Conditioned Semantic Forecasting as a World Model for Autonomous Driving](https://arxiv.org/abs/2611.24840) | 本文提出通过测试时动作干预来量化评估动作条件语义预测模型的动作依赖性，以检验这类模型作为自动驾驶世界模型时是否真正依据动作进行预测。 |
| [^13] | [Kaer: Security-Aware Replanning for Multi-Agent Systems](https://arxiv.org/abs/2611.24839) | Kaer通过抽象模板传递智能体反馈、将计划修订限制在确定性修复算子内并强制执行预设策略，防止恶意智能体劫持多智能体系统的重规划过程。 |
| [^14] | [What Survival Benchmarks Don't Tell You: Impact of Model Selection and Dataset Regimes](https://arxiv.org/abs/2611.24837) | 该研究通过涵盖10种方法与49个数据集的大规模中立基准测试，揭示了评估设计对结论的决定性影响，并发现对深度学习和树模型而言，以IBS而非C-index为目标进行超参数优化，能以极低成本大幅提升概率预测的准确性与校准度。 |
| [^15] | [Overlooked and Underutilized: Prototypes are the Missing Piece in Tabular XAI](https://arxiv.org/abs/2611.24831) | 论文提出将原型（尤其是通过将相关特征聚合成有界子空间、体现人类可理解概念的原型部分）作为特征归因方法的直观补充，以降低认知负担的方式帮助用户审计表格数据上的复杂机器学习模型。 |
| [^16] | [Synthesizing hyperspectral Data using generative Models to train spectral Unmixing Methods for low-cost Crop Residue Cover Mapping](https://arxiv.org/abs/2611.24827) | 本研究提出通过中间条件数据生成器将插值与丰度估计解耦的框架，利用生成模型合成高光谱数据来训练解混模型，从而在地面真值数据稀疏的条件下实现低成本的作物残茬覆盖制图。 |
| [^17] | [Prognosis Without Prediction: The Intrinsic Geometry of Survival](https://arxiv.org/abs/2611.24799) | 该论文提出自组织生存流形（SOSM）深度学习框架，通过将无监督表示学习与生存标签解耦，并借助伪时间加权、曲率最小化的图拉普拉斯潜空间把生存进展视为数据内在几何的涌现属性，从而在184例透明细胞肾细胞癌多组学队列上无需显式结局回归即实现了0.768±0.068的平均一致性指数。 |
| [^18] | [Bias-Only Training Dynamics in ReLU Networks](https://arxiv.org/abs/2611.24797) | 该论文提出用“激活码”刻画单隐层ReLU网络仅训练偏置时的训练动力学，证明训练等价于一系列以激活码为特征的线性最小二乘问题，且网络变宽时神经元切换比例趋近于零、动力学趋于线性。 |
| [^19] | [The Elephant in the Question: Unified Models Picture the Concepts Behind Their Answers](https://arxiv.org/abs/2611.24793) | 本文提出无需训练解码器的V-lens方法，利用统一多模态模型图像生成器对文本隐藏状态的访问能力，可视化模型推理中连接问题与答案的关键中间概念，并通过lens tuning显著提升代词指代等概念的可视化准确率。 |
| [^20] | [Training-free motion statistics match self-supervised behavior encoders](https://arxiv.org/abs/2611.24792) | 本文证明了一种免训练的简单运动统计表示（逐帧通用测量+固定时间窗口汇总+PCA降维）在预注册的公平比较协议下，在多个行为识别基准上超越或匹敌最先进的自监督行为编码器，表明自监督编码器并未真正胜过简单的人工特征。 |
| [^21] | [Learning Trajectories Leave Decodable Traces in Neural Network Weight Space and Internal Representations](https://arxiv.org/abs/2611.24782) | 该研究发现神经网络的最终权重与隐藏表示中保留了可解码的学习历史信息——训练后的网络能被准确匹配回其初始权重，且先前任务的训练顺序可通过线性探针被100%解码。 |
| [^22] | [Reconstruction vs. Prediction Pre-Training of Discrete Video Representations for Continuous Control](https://arxiv.org/abs/2611.24781) | 该论文通过容量匹配的对照实验发现，在离散视频表征预训练中，重构式自编码目标能保留更多物理状态信息，而预测式（V-JEPA）目标在自回归滚动中误差累积更少，并对低层视觉扰动更鲁棒。 |
| [^23] | [Counterfactual World Models: JEPAs That Reason About "What If"](https://arxiv.org/abs/2611.24778) | 该论文证明仅用事实数据训练的世界模型无法回答“同样扰动下换一个动作结果会如何”的反事实问题，并提出利用“自然不做跃变”的一阶近似构建反事实世界模型，使其无需反事实数据和额外参数即可区分糟糕决策与坏运气。 |
| [^24] | [Horizontal Flow Matching for Single-Cell Spatiotemporal Dynamics](https://arxiv.org/abs/2611.24764) | 该论文提出水平流匹配方法，通过在连续-反应方程与水平投影中排除共享刚体运动的代价，使得从空间转录组学数据中学习单细胞时空的组织结构、分子状态与细胞丰度变化时不受切片旋转平移等坐标变化的干扰。 |
| [^25] | [Elysium: Simulation-Free Unbalanced Schr\"odinger Bridges](https://arxiv.org/abs/2611.24763) | Elysium 提出了一种免仿真的非平衡薛定谔桥求解方法，联合学习状态转移与质量变化，无需在训练中模拟轨迹即可从快照数据重建随机细胞动力学。 |
| [^26] | [Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing](https://arxiv.org/abs/2611.24757) | 该论文提出一种方差最优的反事实策略设计方法，通过联合优化动作的采样方式，将流量引导至两种策略不一致的动作上，从而显著降低在线A/B测试的方差并提升样本效率。 |
| [^27] | [Learning the Anyon Atlas](https://arxiv.org/abs/2611.24752) | 通过引入基于逻辑回归的拒绝门算法，将2+1维拓扑量子场论模数据的搜索从22个任意子扩展至42个，识别出150,679个新模数据，据此推翻了Geiko-Moore关于高斯和实性足以保证时间反演对称性的猜想，并展示了算法在秩100处的可扩展性。 |
| [^28] | [Cross-Vehicle Transferability of Lateral Velocity Estimation on Production Vehicles through Input Selection and Data Augmentation](https://arxiv.org/abs/2611.24747) | 该研究首次在量产车辆上量化了神经网络横向速度估计器跨车辆零样本迁移时误差扩大一个数量级的精度差距，并通过系统比较CAN信号输入确定了迁移最稳健的五信号集，同时结合物理动机的数据增强方法来缩小这一差距。 |
| [^29] | [Sex Differences in the Detection of Parkinson's Disease from Speech](https://arxiv.org/abs/2611.24741) | 本研究通过在性别平衡的数据集上结合预训练语音模型嵌入与手工声学特征，首次系统揭示了基于语音的帕金森病检测存在性别差异，女性说话者比男性说话者能被更准确地分类。 |
| [^30] | [Uneven gains in protein-RNA affinity and mutation-effect prediction](https://arxiv.org/abs/2611.24739) | 该研究系统评估了11类蛋白质-RNA亲和力预测方法，发现总体亲和力预测的进步并不均衡地转化为突变效应预测的收益——受控CoPRA降低了绝对亲和力误差、提升了平均效应准确性并在独立外部测试中全面改善，但排序准确性和局部误差的收益参差不齐。 |
| [^31] | [Approximate Dynamic Programming with Basis Function Learning](https://arxiv.org/abs/2611.24737) | 该论文提出BOBF方法，通过反复应用Bellman算子从模型本身学习基函数，保证每次学习都能以可量化幅度改进MDP值函数近似并收敛到最优值函数，其缩减版SBOBF还能在限制基函数数量和计算量的同时保留这些理论保证。 |
| [^32] | [How Generative Models Generate: A Concise Guide to the Generative AI Landscape](https://arxiv.org/abs/2611.24728) | 本文提出以生成为中心的两级分析框架，区分出直接变换/解码、序贯条件构造和迭代更新/输运三种基本生成路径，并考察其背后的建模与学习原理，为理解复杂多样的生成式AI领域提供了简明统一的指南。 |
| [^33] | [Surrogate Modeling and Reduced-Order Methods in Gravitational-Wave Physics and Numerical Relativity: From Spectral Representations to Adaptive Compression](https://arxiv.org/abs/2611.24727) | 该综述系统回顾了以本征正交分解、贪婪基选择和经验插值方法为核心的降阶建模与代理技术，展示其如何将高保真引力波波形压缩为可快速评估的紧凑表示，从而突破数值相对论与参数估计中的计算瓶颈。 |
| [^34] | [DRIFT: Detachable Reasoning for Runtime-Flexible and Efficient End-to-End Driving](https://arxiv.org/abs/2611.24719) | DRIFT提出了一种端到端自动驾驶框架，利用小于1.5亿参数的轻量级语言模型实现场景描述与动作解说，并引入可在运行时灵活开关语言推理的规划模块，且不损害闭环驾驶性能，从而兼顾推理能力与计算效率。 |
| [^35] | [Meta-LSQ: A Closed-Form Sobolev Inner Solver for Few-Shot Meta-Learning](https://arxiv.org/abs/2611.24718) | Meta-LSQ用固定随机傅里叶基上的一次闭式正则化最小二乘求解替代MAML的内环梯度下降，仅元学习三个标量即可获得任意阶精确导数，在少样本正弦回归上以约1/36的训练时间取得比一阶MAML低5.5倍的误差。 |
| [^36] | [Openhagen Worlds: Calibrated Environments for Measuring and Training Scientific Discovery Agents](https://arxiv.org/abs/2611.24711) | Openhagen Worlds是一套包含27个程序化生成环境、178个经严格校准难度级别的科学发现基准套件，通过明确的评分规则和针对神谕策略、非演绎策略及捷径策略的成功率校准，能够真正测量并训练智能体选择信息性实验和从结果中推理的能力，排除记忆回溯与捷径策略的干扰。 |
| [^37] | [When Does Continual Pre-Training Help? A Controlled Study of Dialectal Adaptation for Qu\'ebec French](https://arxiv.org/abs/2611.24699) | 该研究通过对九个大语言模型的预算匹配对照实验发现，持续预训练对魁北克法语方言适应的收益取决于起始模型对目标方言的既有能力，其增益主要由语料构成（尤其是文学文本）而非数据量驱动，且主要提升表达理解而非语法能力。 |
| [^38] | [Stealing Exact Training Data from Other Clients during Federated Learning](https://arxiv.org/abs/2611.24689) | 该论文首次提出一种迭代式攻击流水线，通过逐轮从聚合梯度中减去已还原句子的梯度贡献来解开FedAvg的多客户端梯度纠缠，实现了联邦学习中100%精度的文本精确还原，最高可精确还原80.2%的客户端句子。 |
| [^39] | [Zonotopic Analysis of Quantization Error in Neural Networks](https://arxiv.org/abs/2611.24648) | 本文提出基于 zonotope 与约束 zonotope 的抽象解释验证框架，通过统一的舍入-截断抽象转换器精确界定量化神经网络的量化误差，在 ACAS Xu 基准上较现有方法将误差界宽度收窄约 70-85%、运行时间缩短 65-80%。 |
| [^40] | [Equal Steps, Unequal Curvature: Where Muon's Update Goes Wrong](https://arxiv.org/abs/2611.24644) | 该论文通过测量曲率发现 Muon 优化器在其谱的高曲率顶端奇异方向上步长过大，并利用二阶最优乘子进行诊断与干预以改进其更新。 |
| [^41] | [Choosing the Right Representations: Task-Aware Fusion for Multimodal Materials Language Models](https://arxiv.org/abs/2611.24641) | 该论文提出了轻量级的任务条件化注意力融合桥接模块，能够根据目标性质自适应地从多个结构编码器的不同深度选择并融合互补的表征，从而提升多模态材料语言模型的性质预测能力。 |
| [^42] | [Clover-35B-A3B-Preview Technical Report](https://arxiv.org/abs/2611.24636) | Clover 是一个总参数 350 亿、激活参数仅 30 亿的混合专家扩散语言模型，其创新在于将门控 Delta 网络的循环记忆与块内双向注意力结合以适配块扩散，在智能体基准上全面领先（SWE-bench Verified 达 73.6 分），并在单块 GB300 GPU 上实现 680 tokens/s 的生成速度。 |
| [^43] | [Transferring an analysis of human brain organization to language models](https://arxiv.org/abs/2611.24626) | 该研究将刻画人类皮层组织的扩散映射方法应用于语言模型，发现推理训练使模型在一段连续层跨度内发生局部重组，将该区域的权重嫁接到其他模型中可提升其数学推理能力。 |
| [^44] | [Evaluating the Effectiveness of Location Encoding Methods for Species Distribution Modelling](https://arxiv.org/abs/2611.24622) | 该论文提出了一个标准化评估框架，首次系统量化了不同位置编码方法（如气候变量、学习式编码、地球嵌入模型等）在局部和全球尺度物种分布范围估计中的有效性。 |
| [^45] | [GateWatch: Low-Overhead LLM Monitoring with Boolean Circuits](https://arxiv.org/abs/2611.24619) | GateWatch通过将决策规则表达为少量二值化SAE特征上的显式布尔逻辑电路，实现了紧凑、可解释且低开销的大语言模型行为监控。 |
| [^46] | [Diffusion Models are Provably Efficient Priors for Sparse Recovery](https://arxiv.org/abs/2611.24602) | 本文针对压缩感知中的稀疏恢复问题，首次证明基于预训练扩散模型先验的似然引导退火朗之万采样器能够在多项式时间内以近乎最优的样本复杂度可证明地恢复稀疏信号，为扩散模型作为逆问题先验提供了高效的端到端理论保证。 |
| [^47] | [Morpho-Spectral Projection Analysis (MSPA) for Plant Spectroscopy](https://arxiv.org/abs/2611.24599) | 该论文提出了一种监督式降维方法MSPA（形态-光谱投影分析），突破了传统降维方法将相邻波长视为无序列、忽略光谱序列结构的局限，通过吸收特征运动学和局部多阶奇异模态等互补模块，将植物连续光谱投影到紧凑的形态-光谱子空间以支持性状预测建模。 |
| [^48] | [From Activity Recognition to Shift Detection: Variational Wi-Fi Sensing with Evidential Uncertainty](https://arxiv.org/abs/2611.24596) | 该论文提出Wi-HAR多天线变分自编码器架构，从原始Wi-Fi 6 CSI中学习紧凑分布，在十二类活动识别上达到96.6%的中位数准确率（显著优于单天线模型的70.8–81.5%），并复用同一潜在骨干实现部署条件漂移的检测。 |
| [^49] | [AGAMOTTO: An AI-Enabled Framework for Time-Critical Tactical Course-of-Action Evaluation and Generation](https://arxiv.org/abs/2611.24568) | AGAMOTTO是一个AI赋能框架，通过集成仿真能力使指挥官能够在时间紧迫的实时作战中动态评估和生成战术行动方案，初步结果已验证该系统的可行性。 |
| [^50] | [Taming the Tails of Stochastic Gradient Descent in the Interpolation Regime](https://arxiv.org/abs/2611.24562) | 本文首次在插值机制下为随机梯度下降建立了高概率收敛保证，在凸情形证明了 log(β⁻¹)·n⁻¹ 的快速率，在强凸情形给出了一族基于矩的界及其匹配下界，证明指数速率的劣化本质上是不可避免的。 |
| [^51] | [The geometry of neural network decision boundaries](https://arxiv.org/abs/2611.24557) | 该论文提出了一种直接在高维输入空间中可视化和测量神经网络决策边界几何形状的方法，揭示了决策边界在大连续子空间内接近输入样本，且训练方法对边界几何的影响显著大于网络架构。 |
| [^52] | [Sparse designs via quantization of the uniform measure on high-dimensional hypercubes](https://arxiv.org/abs/2611.24507) | 本文提出用超立方体上均匀测度的量化误差替代传统的maximin/minimax准则来构造高维稀疏空间填充设计，从而克服了高维下所需设计点数过多以及对边界（尤其是超立方体顶点）过度敏感的缺陷。 |
| [^53] | [Latent Action Policies from Suboptimal Data](https://arxiv.org/abs/2611.24482) | 该论文提出了LAPIS方法，通过推导潜在动作空间中策略改进有效迁移与最优对齐的条件，解决了从次优数据训练潜在动作策略时离线强化学习不可靠的问题。 |
| [^54] | [Every Query Routes Itself: Accelerating Video Diffusion Transformers with Per-Query Routed Sparse Attention](https://arxiv.org/abs/2611.24479) | 提出了无需训练的逐查询路由稀疏注意力方法PQRS，让每个查询独立选择自己的键簇并用质心补偿未选中的键簇，从而在提升注意力精度的同时加速视频扩散Transformer的推理。 |
| [^55] | [ICICLE: In-Context Inference for Curve-Learning Extrapolation](https://arxiv.org/abs/2611.24469) | ICICLE 是一种轻量级（仅128万参数，约为FT-PFN的8.7%）的上下文内学习曲线外推代理模型，通过共享集合编码器支持可变数量的超参数，在多个基准上以远少于FT-PFN的参数量取得相当性能，并在Meta-Album基准上超越Quick-Tune。 |
| [^56] | [Cooperative UAV and Ground Station for Robust MEC of Dynamic Aquatic IoT: A Generative AI-Enhanced Approach](https://arxiv.org/abs/2611.24466) | 本文提出一种生成式AI增强的无人机与地面站协同鲁棒边缘计算框架，通过联合任务卸载与无人机轨迹优化来最小化动态水上物联网计算任务的总执行时间。 |
| [^57] | [Harmonic Circuit: Chord Progression Lies in a Sparse Principal Information Flow in Symbolic Music Models](https://arxiv.org/abs/2611.24453) | 首次在符号音乐生成模型（ChatMusician 与 MuPT）中进行电路发现，识别出驱动和声进行 I…V→I 行为的一条稀疏主信息流电路，并通过因果干预验证其作用。 |
| [^58] | [From real to synthetic: PSG-GAN learns sleep apnea signal dynamics for AI training](https://arxiv.org/abs/2611.24436) | 本研究开发了一种生成对抗网络PSG-GAN，能够协同合成具有阻塞性睡眠呼吸暂停特征的长片段多信号PSG数据，并真实再现信号间的动态交互关系，为AI模型训练提供了高质量的合成数据。 |
| [^59] | [DyNA-Mixer: Decoupling Time and Topology for Scalable Dynamic Node Affinity Prediction](https://arxiv.org/abs/2611.24426) | DyNA-Mixer 通过将时间演化建模与拓扑结构建模解耦，在保证计算可扩展性的同时捕捉多尺度时间模式与全图协同模式，实现了可扩展的动态节点亲和度预测。 |
| [^60] | [Predicting Process-Level Behavior: A Benchmarking Study](https://arxiv.org/abs/2611.24425) | 该论文通过对并发轨迹预测和平均吞吐时间预测两项流程级任务的全面基准测试，比较了统计方法、机器学习方法、基础模型、业务流程仿真以及聚合到流程级的轨迹级PPM方法，揭示了最适合预测流程级行为的建模范式。 |
| [^61] | [Probing Physics in Motion Diffusion Models Against Their Own Estimate](https://arxiv.org/abs/2611.24423) | 该研究对运动扩散模型的内部激活进行逐层物理探测后发现，当读取方法被公平调整时，模型自身的干净样本估计与内部激活的表现相当（差距最多仅0.01），表明此前报告的“隐藏物理知识”更多是读取方法可提取性差异造成的假象，而非模型真正隐藏了物理信息。 |
| [^62] | [MINT: Meeting-time Indicators for Truncation in Multi-Step Off-Policy RL](https://arxiv.org/abs/2611.24413) | 该论文提出 MINT 方法，用一个基于行为轨迹与目标轨迹首次相遇时间的二元停止指标来替代传统多步离策略强化学习中的逐轨迹系数乘积，从而免除了对敏感截断长度超参数、转移模型及混合时间估计的依赖，同时兼顾方差控制与长时程信号。 |
| [^63] | [RSIBench-Infra: Can Agents Optimize Their Own Training and Inference Infrastructure at Production Scale?](https://arxiv.org/abs/2611.24395) | 该论文提出了 RSIBench-Infra，一个面向生产规模、评估智能体能否在引擎、内核和指令层级对训练与推理基础设施进行端到端优化的基准，结果显示 66.2% 的任务实现了超过 1.10 倍的有效加速。 |
| [^64] | [Temporal Rank in Recurrent Neural Networks](https://arxiv.org/abs/2611.24394) | 本文发现了RNN中独立于梯度消失/爆炸问题的另一关键局限——由秩约束的隐藏状态动态导致的信息瓶颈，并提出结构化高维状态动态方法加以解决，使标准RNN在序列任务上实现快速学习。 |
| [^65] | [How to Turn an Attention Neural Network into an Interpretable Hard-Logic Predicate Circuit: Truth-Table Predicate Extraction and State-Machine Merging under Ship-of-Theseus Replacement](https://arxiv.org/abs/2611.24381) | 该论文提出一种“忒修斯之船”式逐块替换方法，将训练好的注意力网络（包括注意力机制本身）整体转化为由0/1谓词与最小化真值表构成的硬逻辑谓词电路，既保持可解释性又能在教师正确的样本上忠实模仿教师。 |
| [^66] | [MoE-Context: Show the Model Only What It Needs](https://arxiv.org/abs/2611.24379) | MoE-Context通过在每步前进行快速的“是/否”门控判断，只向语言模型展示当前所需的上下文块，其余内容以可召回的单行笔记形式掩蔽在单一键值缓存中而不被删除，从而将查找任务准确率从93/120大幅提升至119/120。 |
| [^67] | [LLM-Assisted Multi-Fidelity Bayesian Optimization: Fusing Expert Knowledge with Numerical Data](https://arxiv.org/abs/2611.24377) | 该论文提出LLM辅助的多保真度贝叶斯优化方法，利用大型语言模型将专家关于低保真度模型所缺失物理效应的定性描述转化为成对比较知识，并与数值数据融合，从而在高保真度数据稀缺时更准确地学习低-高保真度差异，降低工程设计成本。 |
| [^68] | [VAFL: A Vertical Analytic Federated Learning Mechanism for Efficient Crowdsourced Services via Closed-Form Solutions](https://arxiv.org/abs/2611.24367) | 提出了一种基于闭式解的纵向分析联邦学习框架VAFL，通过正则化最小二乘在固定局部表示上拟合任务头，并利用块因子等价求解和两种宽度缩减策略大幅降低计算开销，实现高效的两方及多方隐私保护协作学习。 |
| [^69] | [OCTR2A: Real-Time RGB-to-BEV Trajectory Projection and Forecasting at Urban Intersections](https://arxiv.org/abs/2611.24352) | 本文提出OCTR2A架构，仅利用现有城市基础设施上的对角成对RGB相机即可实时重建并预测交叉口交通参与者的鸟瞰图轨迹，无需激光雷达、航拍或高精地图等昂贵资源。 |
| [^70] | [Tell Me How Sure You Are: Communicating Uncertainty in Auxiliary Evidence to Foundation Models](https://arxiv.org/abs/2611.24351) | 本文首次系统地证明，向基础模型传达上下文信息中的不确定性能够为下游零样本图像分类决策提供有用的额外线索，并在多种设置中带来性能提升。 |
| [^71] | [Exploring Daily Emotional Well-Being through Personalized Data Analysis](https://arxiv.org/abs/2611.24342) | 该项目通过整合多源个人数据构建每日情绪数据集，结合人格与幸福感问卷对个体日常情绪状态及其时间变化进行个性化分析，并据此提出了一种简单的个性化情绪推荐方法。 |
| [^72] | [OpenGNS: An Open Dataset of the Gradient Noise Scale across Vision, Language, and Diffusion Models](https://arxiv.org/abs/2611.24331) | 该论文发布了首个开放GNS测量数据集OpenGNS，包含横跨ResNet图像分类、GPT语言建模与DiT图像合成三大领域的400余条测量轨迹，为系统研究梯度噪声尺度与临界批量大小提供了数据基础。 |
| [^73] | [Can Weight-Space Programming Alter Learned Ethical Priorities?](https://arxiv.org/abs/2611.24330) | 该研究表明，通过直接编辑权重空间中的选择权重，可以成功调整模型学习到的伦理优先级——使模型在权衡中更严格地对待正式誓言而非普通承诺，同时不损害其在其他任务上的表现与事实准确性。 |
| [^74] | [Distribution-Free Uncertainty Quantification for Weather Forecasting: Conformal Prediction Under Spatiotemporal Dependence](https://arxiv.org/abs/2611.24329) | 该论文对八种共形预测区间构造方法在网格化2米气温预报（6小时至10天八个时效）中的表现进行了系统基准测试，并创新性地引入空间过散度指数、每日空间覆盖率分位数和漏报持续性等时空依赖性诊断指标，揭示了边际覆盖率与时空联合覆盖表现之间的差异。 |
| [^75] | [Structure-Preserving Neural Approximation of Convex, Semiconvex, and Quasiconvex Functions](https://arxiv.org/abs/2611.24325) | 本文将输入凸神经网络的结构保持逼近框架扩展到半凸、拟凸和凸可变换函数，建立了尖锐的极小极大逼近速率，并发展了具有 $L^\infty$ 逼近保证的结构保持输入拟凸神经网络（IqCNN）。 |
| [^76] | [RespCCT and FreqCCT: Compact, Explainable and Cost-Aware Convolutional Transformers for Multi-Class Respiratory Disease Classification](https://arxiv.org/abs/2611.24323) | 本文提出了两个仅约17万和12.5万参数的紧凑卷积Transformer（RespCCT与FreqCCT），从零训练即可在七个类别的呼吸疾病肺音分类中与大型预训练模型相媲美，同时显著降低计算成本并具备可解释性。 |
| [^77] | [Supervised Hebbian Class Networks: Learning Without Backpropagation](https://arxiv.org/abs/2611.24314) | 本文提出监督式赫布类网络，通过为神经元分配类别、采用跨层共享的基于类别的监督规则并结合适配的Lion优化器，实现了无需反向传播和梯度计算的多层感知机与卷积网络训练。 |
| [^78] | [AutoPI: Taxonomy-Guided Adaptive Red Teaming for Prompt Injection in LLM Agents](https://arxiv.org/abs/2611.24313) | AutoPI提出了一种以提示注入技术分类体系PIAtlas为引导的自适应红队测试方法，通过为每个子技术生成注入并迭代改进，旨在发现攻破LLM智能体的多种多样化方式，而非仅仅最大化攻击成功率。 |
| [^79] | [LoopCache: Accelerating Long-Context Self-Speculative Decoding for Looped Transformers](https://arxiv.org/abs/2611.24301) | 本文提出LoopCache，一个面向长上下文循环Transformer的高效自投机解码框架，通过置信度引导机制解决现有波风格方法中固定循环起草浪费计算、以及跨循环批处理导致KV缓存内存流量放大R倍这两个关键问题。 |
| [^80] | [Element-Wise Femoral Stress Prediction: A Hybrid Graph Neural Network Surrogate for Finite Element Analysis](https://arxiv.org/abs/2611.24274) | 该研究提出了一种结合图卷积、邻域聚合与残差学习的混合图神经网络（GCN-SAGE-Residual）代理模型，可在保留有限元网格空间拓扑结构的前提下实现快速的单元级股骨应力预测，克服了传统有限元分析计算成本高、常规机器学习模型忽略网格结构的缺陷。 |
| [^81] | [Stability-compatible kernels: beyond Gaussianity for impulse response estimation](https://arxiv.org/abs/2611.24272) | 该论文提出“稳定性相容核”的新概念，证明其当且仅当稳定元素在对应RKHS中稠密时成立，从而在超越高斯先验的情形下为脉冲响应估计提供了比已知条件更弱的几乎必然BIBO稳定性判据。 |
| [^82] | [TRESS: A Fixed-Size History Representation for Irregular Time Series Forecasting](https://arxiv.org/abs/2611.24257) | TRESS提出了一种固定大小的历史表示方法，通过保留带时间戳的近期观测值并摘要完整历史的统计信息，结合查询条件化的事件提升模块与通道特定解码器，在不规则时间序列预测上取得了最优的测试误差。 |
| [^83] | [ViLFlow: Parallel Vision-Language Generation with Continuous Latent Flows](https://arxiv.org/abs/2611.24240) | 提出ViLFlow，一种结合输出嵌入联合建模与并行token解码的连续流视觉-语言模型，通过图像引导的教师预测和预测感知解码，将预训练连续语言生成模型高效适配到视觉问答和图像描述任务。 |
| [^84] | [Functional Principal Component Analysis: Minimax Optimality from Sparse to Dense Designs and Beyond](https://arxiv.org/abs/2611.24228) | 本文首次在含噪离散观测下建立了函数主成分分析的完整极小化极大最优理论，覆盖了从稀疏到稠密随机设计中谱指标发散的特征函数与维数递增的主子空间，并给出了一般二元函数类上本质上二维的最优协方差估计速率。 |
| [^85] | [TableControl: Gesture Interaction Through Surface Vibrations](https://arxiv.org/abs/2611.24218) | 本文提出TableControl系统，在微控制器上运行完整的处理流程，利用四个隐藏的压电传感器实现基于表面振动的连续手势识别，可识别敲击、上滑和下滑手势并驱动游戏和台灯等实际应用，且无需传输原始传感器信号。 |
| [^86] | [Towards Reliable Scaling Laws for Tabular Foundation Models](https://arxiv.org/abs/2611.24214) | 该研究揭示了表格基础模型在合成数据与真实数据上缩放行为的差异，指出准确拟合缩放定律并不能保证在未见数据集上的可靠预测，且使用过少的评估数据集选择配置会导致系统性过度乐观，而计算最优缩放需要扩大合成预训练数据集规模的搜索范围。 |
| [^87] | [Continuous Noisy-Loop Models](https://arxiv.org/abs/2611.24213) | 本文量化了“统计上高效可学”与“存在紧凑概率电路”之间的本质差距：对网格上相邻自旋一致性的连续含噪读数，样条估计器可在多项式时间内达到极小极大最优的 Hellinger 风险 $\Theta(\min\{1, dN^{-2/3}\})$，而紧凑的非负可分解电路则需 $\exp(\Theta(\sqrt d))$ 规模的节点，并给出了 KL 逼近误差的精确刻画。 |
| [^88] | [GALA: Gallery-Level Alignment for Real-Life Vision Decoding from EEG](https://arxiv.org/abs/2611.24206) | 该论文提出GALA方法，并引入“增量直推式校准”新范式，利用预先已知的候选库和持续累积的未标注EEG数据进行无标签校准，从而解决EEG视觉解码难以跨个体泛化、依赖昂贵标注数据的瓶颈问题。 |
| [^89] | [Learning with Unlearning: Synaptic Role Partitioning for Simultaneous Continual Learning and Machine Unlearning](https://arxiv.org/abs/2611.24205) | 提出了统一框架LwU，通过在突触层面对每个权重进行“保留”与“删除”双重评分，将网络划分为四个具有独立更新规则的区域，从而在同一模型中同时化解持续学习与机器遗忘之间的冲突。 |
| [^90] | [Worth Another Look:Constrained World-Model Planning for Visual Inspection](https://arxiv.org/abs/2611.24187) | 该论文将冻结检测器上的视觉检查形式化为部分可观测下的神经符号观测规划，通过紧凑循环世界模型与滚动时域束搜索在线联合调度查询、裁剪、增强和终止，在严格预算下比静态调度和无搜索离线RL最高提升28.4个AP百分点。 |
| [^91] | [Rare-Tail Statistics For Learning Biased Gaussian Halfspaces with Label Noise](https://arxiv.org/abs/2611.24181) | 本文提出“稀有尾部均值估计量”这一极简统计方法，在大偏置高斯半空间带标签噪声的学习问题中，以信息论最优样本率恢复目标半空间方向，超越了经典平均法等全局一阶矩统计量。 |
| [^92] | [Adaptive Action Horizon Selection via Flow-Field Uncertainty](https://arxiv.org/abs/2611.24178) | 本文提出DUAC，一种无需重训练的事后不确定性估计方法，通过预训练VLA模型向量场的闭式流后验协方差和雅可比-向量积，仅需单个动作块即可获得动作级不确定性分数，从而实现自适应动作视界选择并避免多次采样的推理开销。 |
| [^93] | [SoU-RHS: Bayesian Structured Sparsity for Compact Architecture Recovery in CNNs](https://arxiv.org/abs/2611.24166) | 该论文提出SoU-RHS，一种将正则化马蹄铁收缩与不确定性来源稀疏全局-局部机制相结合的贝叶斯结构化稀疏方法，能够稳定地从过参数化CNN中恢复出规模一致且适应数据的紧凑架构。 |
| [^94] | [Nearly Tight Almost Sure Convergence Rates for the Last iterate of SGD](https://arxiv.org/abs/2611.24151) | 该论文为SGD最后迭代点的几乎必然收敛建立了近乎紧致的速率界——光滑情形下分别为 $k^{-1/3}$（存在稳定极小点）和 $k^{-1/4}$（且后者可推广至非光滑凸情形），并通过匹配下界证明这些指数在相差多对数因子的意义下是最优的。 |
| [^95] | [Fill-Distance-Guided Active Learning for Causal Function Recovery under Network Interference](https://arxiv.org/abs/2611.24143) | 该论文提出FANCI方法，在有限实验预算下，利用填充距离引导的主动学习设计与特定处理的建模非线性响应曲面，实现网络干扰环境中完整的暴露依赖总体平均响应函数恢复。 |
| [^96] | [QuantAtlas-OM: Taming Neural Spectral Bias in Long-Horizon Spatiotemporal Ocean Dynamics](https://arxiv.org/abs/2611.24124) | 提出尺度-频谱解耦框架QuantAtlas-OM，将时间演化约束在低波数潜在流形以保证长时程稳定性，同时通过外部旁路保留全分辨率涡旋与锋面残差，从而解决自回归海洋预测中的神经频谱偏差问题。 |
| [^97] | [The 7th International Verification of Neural Networks Competition (VNN-COMP 2026): Summary and Results](https://arxiv.org/abs/2611.24114) | 第七届国际神经网络验证竞赛（VNN-COMP 2026）通过统一的标准格式、标准化硬件环境和作者指定的工具参数，对最先进的神经网络验证工具在可扩展性和速度方面进行了客观比较，并总结了竞赛规则、基准测试、结果与经验教训。 |
| [^98] | [AG-BFGS: Foundations for Accelerated Geometry Learning](https://arxiv.org/abs/2611.24113) | 该论文提出了AG-BFGS加速几何学习框架，将动量引入BFGS的Hessian逆近似学习中，首次为加速拟牛顿方法建立了严格的理论基础，证明了加速更新满足显式稳定性条件且保持正定性。 |
| [^99] | [UCLIR: Unsupervised Continual Learning using Independent Representations](https://arxiv.org/abs/2611.24093) | 该论文提出UCLIR方法，受大脑指数距离规则启发，通过对多个独立学习的小型互补表示分别施加权重正则化，有效解决了持续学习中随任务增加旧知识保留效果下降的问题。 |
| [^100] | [Quantum Annealing with a Quantum Neural Network: Width of Annealing Matters](https://arxiv.org/abs/2611.24089) | 本文将量子退火扩展至量子神经网络，发现宽度仅为 2 的量子退火网络即可在量子搜索问题中指数级地降低退火深度，且通过调节参数，深度为 1、宽度为 2 的量子感知机便能以指数级小的误差完成求解。 |
| [^101] | [VESPA: Verifiable Symbolic Policy Adaptation for Safe Reinforcement Learning](https://arxiv.org/abs/2611.24083) | VESPA提出了一种利用形式化验证产生的反例对决策树策略进行定向微调的框架，能够高效地将不安全策略调整为安全策略，避免了从头重训与反复验证的低效循环。 |
| [^102] | [Meaning Reaches Object Choice but Not Completion: Where Is a VLA Vulnerable to Landmark Synonyms?](https://arxiv.org/abs/2611.24076) | 研究发现VLA策略能够理解训练指令中未出现过的地标同义词并将其语义用于物体选择（该语义作用点位于视觉-语言模型与动作生成器之间的连接器之前），但这种措辞改写会显著降低任务完成率，且连接器之后无法通过单独注入单个模态的token来改变选择。 |
| [^103] | [Recurrent Domain Incremental Adaptation of Time-Series Foundation Models](https://arxiv.org/abs/2611.24075) | 该论文提出PACT方法，将冻结的嵌入空间划分为局部图卡并进行轻量级的分段残差自适应，从而在运行条件不断变化的情况下实现时间序列基础模型的域增量自适应，同时避免全局自适应破坏已正确分类的嵌入表示。 |
| [^104] | [FDM+: A Framework for Constrained Control of the 2D Radiative Transport Equation](https://arxiv.org/abs/2611.24070) | 该论文提出FDM+框架，通过解耦预测架构、带总误差界分析的高效参数化策略以及基于条件分布的设计空间探索，解决了冻结扩散模型在二维辐射输运方程受限控制中维度差异、分辨率不匹配与多解性三大难题。 |
| [^105] | [Replay, Don't Resample: Off-Policy Consolidation of Long-Horizon RL Experts](https://arxiv.org/abs/2611.24059) | 提出多专家回放策略蒸馏（MRPD），通过按专家的学习顺序回放其RL训练日志（而非代价高昂的重新采样）来整合多个长时程强化学习专家，从而大幅降低整合成本并缓解训练与推理的不匹配。 |
| [^106] | [Faster to the Wrong Answer: The Price of Local Updates in Federated Optimization](https://arxiv.org/abs/2611.24057) | 该论文刻画了联邦学习中在给定条件数约束下局部更新可达到的精确最坏情况偏差，揭示了局部计算加速与更新偏差之间的根本权衡。 |
| [^107] | [From Data to Valves: A Self-Learning, Production-Grade Edge Stack for Sustainable HVAC Control](https://arxiv.org/abs/2611.24052) | 该论文提出了一个在每栋建筑的NVIDIA Jetson Orin边缘网关上运行的自学习生产级HVAC控制栈，其核心创新是结合序贯学习且受多重门控保护的transformer温度预测器和带有不可削弱结构性安全包络的分布式软演员-评论家冷冻水阀门控制智能体，实现从建筑自身数据持续自适应并自我验证控制效果。 |
| [^108] | [Algorithmic Market Making Can Lead to Rough Volatility Through the Order Book](https://arxiv.org/abs/2611.24028) | 本文提出一种自洽机制，将做市商建模为采用随机梯度下降规则、异步响应新信息调整报价的交易台，从而证明算法化做市可以通过订单簿动态自然地产生粗糙波动率。 |
| [^109] | [Semi-Supervised GAN-Based Segmentation of CTPA Scans for Pulmonary Emboli Boundary Extraction](https://arxiv.org/abs/2611.24018) | 该论文提出了一种基于半监督生成对抗网络的CTPA扫描分割方法，通过组合Dice损失、交叉熵损失、特征匹配损失和自训练损失，有效解决了CTPA图像高度不平衡和标注数据稀缺的问题，实现肺栓塞边界的自动提取。 |
| [^110] | [Rethinking Contrastive Targets in Radiology Language-Image Pretraining](https://arxiv.org/abs/2611.24016) | 该论文指出放射学CLIP式对比预训练中零样本性能先升后降及大批量训练失效的根源在于目标错定——大量未配对报告与图像共享临床发现导致配对身份目标存在系统性偏倚，作者通过合成测试平台CLEVR-Radiology复现了这些失败并建立了相应的目标错定理论加以解释。 |
| [^111] | [NOTREKS: Marginal Independence in Score-based Structure Learning](https://arxiv.org/abs/2611.24015) | 本文提出 NOTREKS 信息，将变量间边际独立的先验知识转化为“无 trek”图约束，并通过 PSTrek 惩罚项和 trek 约束融入基于评分的因果结构学习，从而在有限样本下提升结构恢复与效应识别效果。 |
| [^112] | [Rethinking Fixed-Horizon Latent Reasoning](https://arxiv.org/abs/2611.24001) | 该论文发现将潜空间思维链运行至完整预设长度反而可能损害推理性能，并提出用“已实现与剩余建构性进展”来刻画推理收益，为重新思考固定长度的潜空间推理提供了依据。 |
| [^113] | [A Mechanistic Analysis of In-Context Learning and Out-of-Context Hallucination](https://arxiv.org/abs/2611.24000) | 本文通过动态上下文中功能角色的统一视角，首次将上下文学习与上下文外幻觉这两种看似对立的现象联系起来进行机制分析，发现角色效应的显著影响取决于足够的贡献幅度和一致的带符号聚合这两个必要条件。 |
| [^114] | [ReliFlow: Reliability-Calibrated Residual Transport for Combinatorial Single-Cell Perturbation Prediction](https://arxiv.org/abs/2611.23999) | 提出ReliFlow——一种条件流匹配模型，通过训练期交叉拟合估计单基因效应可靠性并据此校准加性源分布，将速度场分解为加性参考的已知修正与学习得到的残差传输场，从而在单扰动证据稀疏或缺失的情况下显著提升组合扰动响应的预测精度。 |
| [^115] | [Optimal Deep Nonparametric Regression with Uniformly Bounded Lipschitz Constants](https://arxiv.org/abs/2611.23979) | 本文在分裂谱范数乘积约束下建立了深度ReLU网络的尖锐逼近理论，并提出带惩罚的最小二乘估计器，使深度非参数回归在保持一致有界Lipschitz常数的同时达到最优统计收敛率。 |
| [^116] | [OMNIS+: Semantic RAN Slicing and Quantization for Dynamic Split Neural Networks](https://arxiv.org/abs/2611.23977) | 提出了OMNIS+语义RAN切片框架，通过动态分割神经网络、语义压缩与自适应“box”量化，在多小区边缘计算中实现通信与计算资源的联合优化调度，并显著增强了对传输比特错误的鲁棒性。 |
| [^117] | [How Complete Must an Environment Be to Train a Capable GUI Agent?](https://arxiv.org/abs/2611.23946) | 该论文提出MobileGym平台，首次系统研究了GUI训练环境在奖励覆盖、数据覆盖和特征覆盖三个维度上的完整性对智能体训练效果的影响，发现环境完整性的降低会显著损害智能体的性能。 |
| [^118] | [PGU-Decoder: Prior-Guided Unified Decoding for Industrial Road-Network Map Updating](https://arxiv.org/abs/2611.23936) | PGU-Decoder提出先验引导的统一解码框架，通过先验查询模块和关系图注意力网络，统一处理交叉口路网生成、合流/分流检测与可行驶区域检测三大任务，降低工业级路网地图更新的运营成本。 |
| [^119] | [CoSlice: Coordinated Multi-Agent Reinforcement Learning for 5G Network Slicing Allocation](https://arxiv.org/abs/2611.23909) | 该论文提出CoSlice框架，将偏好条件化的切片智能体与基于大语言模型的协调器相结合，通过分层多智能体强化学习解决O-RAN中切片间物理资源块分配问题，在保障各切片SLA的同时实现整体资源的高效利用。 |
| [^120] | [Kernel Ridge Attention: Nonlinear Memory with Selective Forgetting](https://arxiv.org/abs/2611.23896) | 提出核岭注意力（KRA），一种通过递归岭回归实现非线性记忆更新、结合内容依赖的选择性遗忘机制、推理状态大小与上下文长度无关且在语言建模中优于注意力与循环基线的循环序列混合器。 |
| [^121] | [Accelerating Distributed Digital Ising Machines](https://arxiv.org/abs/2611.23883) | 该论文提出了首个经过校准的分布式数字伊辛机跨平台成本模型，将扫描时间分解为计算、内存、同步和器件间通信四项，并据此提供系统性技术来加速其MCMC工作负载。 |
| [^122] | [A Rank-One Bypass of Fixed Linear Erasure](https://arxiv.org/abs/2611.23880) | 本文证明了在开放权重场景下，攻击者仅需一个秩一的仿射更新即可绕过固定线性擦除层，几乎完全恢复被擦除的目标性能，同时保持良性线性头的输出分数不变。 |
| [^123] | [Infinity-Wasserstein Distributionally Robust Principal Component Analysis: Exact Reformulation, Finite-Sample Guarantee, and Efficient Algorithm](https://arxiv.org/abs/2611.23877) | 该论文提出基于无穷范数-Wasserstein 距离的分布鲁棒 PCA 模型，实现了精确的有限维重构、最坏情况分布的显式刻画、有限样本理论保证与高效算法，并揭示其目标函数通过保留经验协方差之外的逐样本残差信息，使分布鲁棒性能够重塑所学的主子空间。 |
| [^124] | [In Situ Training of Optical Neural Networks by Coding Agents](https://arxiv.org/abs/2611.23875) | 提出“光学即代码”方法，让编码智能体仅凭相机测量图像就能原位训练光学神经网络，自主形成对光学的物理理解并用可解释光学元件重新参数化网络，性能优于无梯度方法。 |
| [^125] | [LAGR: Language-Aware Grasp Re-Ranking for Task Oriented Grasping](https://arxiv.org/abs/2611.23858) | 该论文构建了包含268万个带评分的5自由度抓取标注的大规模面向任务抓取数据集，并提出LAGR方法，通过语言感知的抓取重排序同时兼顾抓取的物理可行性与任务适配性。 |
| [^126] | [Neural Memory Orchestration](https://arxiv.org/abs/2611.23857) | 提出了首个在真实硬件上针对真实工作负载端到端部署的学习型页面预取系统 Overseer，通过 eBPF 采集缺页事件、transformer 预测器（OverNet）预测未来页面地址、内核常驻执行器提前载入页面，从而缓解 HPC 工作负载因缺页中断造成的内存瓶颈。 |
| [^127] | [Beyond Cross-Modal Reconstruction: Participant-Level Evaluation of sEMG-Kinematic Coupling in Post-Stroke Gait](https://arxiv.org/abs/2611.23856) | 该研究通过源模态移除与跨参与者源替换实验证明，冻结的双向模型确实捕获了卒中后步态中特定于受试者的肌电-运动学神经肌肉耦合，而非仅依赖参与者间共享的时间模式。 |
| [^128] | [A Fast, Programmable GPU-to-CPU Invocation Layer for GPU-Initiated Communication](https://arxiv.org/abs/2611.23848) | 提出KIWI框架，将GPU到CPU的反向卸载打造为一等通信路径，实现亚微秒级设备到主机延迟和每秒数亿次操作，性能可超越GPUDirect Async。 |
| [^129] | [RePlan-VLA: Progress-Aware Replanning for Long-Horizon Vision-Language-Action Models](https://arxiv.org/abs/2611.23843) | RePlan-VLA是一个与具体VLA模型无关的框架，通过监测机器人运动来评估任务进度并自适应地修订未来计划，从而将VLA模型的短时程操作能力转化为可靠的长时程任务执行能力，且完全无需对底层VLA策略进行额外微调。 |
| [^130] | [Online State-Representation Adaptation under Synthetic Interaction-Cue Remapping](https://arxiv.org/abs/2611.23836) | 在线索重映射的合成任务中，持续在线更新状态映射的预测器较冻结系数的预测器将后期动作准确率提升28.76个百分点，且情境老虎机通过利用可揭示真实状态的动作奖励获得了最高的累积效用。 |
| [^131] | [NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding](https://arxiv.org/abs/2611.23835) | 提出NarrowSpec，证明推测解码中读取目标隐藏状态的草稿模型的隐藏宽度可缩小至目标模型的一半或四分之一而几乎不影响接受率，从而在B200和DGX Spark上实现1.06-1.44倍的端到端解码加速。 |
| [^132] | [CAFE-AQI: Coverage-Aware Day-Ahead AQI Forecasting with Early Warning and Conformal Uncertainty in Bangladesh](https://arxiv.org/abs/2611.23815) | 该论文提出CAFE-AQI框架，通过审计发现孟加拉国数据集中城市标签重复与覆盖突变问题并加以加权处理，结合时空特征、类别均衡预警头和风险条件共形校准，实现次日最大AQI预测、健康风险早期预警及不确定性区间估计。 |
| [^133] | [Continual Tool-Space Extension Learning for Multimodal Agents](https://arxiv.org/abs/2611.23810) | 提出COTEX方法，通过参数方向定位与正交约束，在适配器层面以低成本为多模态智能体持续扩展新工具，同时保持对已有工具的使用能力。 |
| [^134] | [Sightline: Long-Window Speculative Decoding for Fast SVG Generation](https://arxiv.org/abs/2611.23807) | Sightline通过将验证窗口延长至草稿器的预测范围并配合足够宽的草稿树，利用SVG生成中草稿错误稀少且聚集的特性，实现了当前最快的无损SVG解码。 |
| [^135] | [Improved Leverage Score Sampling for Constrained Active Linear Regression](https://arxiv.org/abs/2611.23801) | 提出椭球正则化杠杆值分数采样，通过外接椭球的形状矩阵刻画约束集几何并将采样偏向重要方向，使约束主动线性回归的查询复杂度由各向异性有效维度决定，从而改进各向异性约束下的查询效率。 |
| [^136] | [TabBioMed: A Large-Scale Benchmark for Biomedical Tabular Learning](https://arxiv.org/abs/2611.23779) | 提出了大规模生物医学表格学习基准TabBioMed，统一了94个精选数据集与23个模型的标准化评估，发现表格基础模型整体预测性能最强，但其优势依赖于具体数据条件。 |
| [^137] | [STORM: A Lightweight Structure-Free Model for Binding Affinity Prediction](https://arxiv.org/abs/2611.23777) | STORM是一种快速轻量、无需结构信息的机器学习模型，通过集成多种蛋白质和分子表示的小模型融合来预测结合亲和力，在相同训练数据下达到或超越Boltz-2的准确度，同时大幅降低了大规模筛选和模型重训的计算成本。 |
| [^138] | [A Systematic Analysis of a Small Optimal Subspace for Physics-Informed Neural Networks: $n$-DAS $\subset \mathbb{R}^p$](https://arxiv.org/abs/2611.23773) | 该论文提出“主导对齐子空间”（DAS），通过梯度对齐而非Hessian特征值大小来选择低维优化子空间，论证了最高曲率方向并非训练实际移动的方向，并以物理信息神经网络为对象对该结构进行了系统分析。 |
| [^139] | [When Normal-Only Updates Are Not Enough: Test-Time Adaptation in Reconstruction-Based Time-Series Anomaly Detection](https://arxiv.org/abs/2611.23761) | 研究表明，即使将测试时自适应的更新严格限制在被标记为正常的样本上，也无法可靠地改善基于重构的时间序列异常检测性能。 |
| [^140] | [Interpreting Timestamp Permutations in Dynamic Link Prediction](https://arxiv.org/abs/2611.23753) | 该论文提出一种交叉诊断方法，将时间戳置换实验中的训练效应与评估历史效应分离，并发现动态链接预测器的性能差距主要源于原始时序训练（尤其是显式时间编码）的贡献，而非评估历史的改变。 |
| [^141] | [Thinking Fast, Anywhere: One System-1 for the Decision Points of Any Agent Workflow](https://arxiv.org/abs/2611.23751) | AnyS1将人类编写的程序性技能转化为领域通用的决策点来源，训练出一个统一的4B轻量System-1模型，使任意智能体工作流在运行时通过一次前向传播即可完成常规决策，并在置信度不足时回退到大模型System-2。 |
| [^142] | [Same Rank, Different Needs: Rethinking LoRA Redistribution in Federated Learning](https://arxiv.org/abs/2611.23742) | 该论文揭示联邦LoRA中按矩阵重构最优截断聚合更新可能无法保留其对单个客户端输入的响应，并提出了接收方感知子空间适配框架RASA，利用对角输入二阶矩对接收方加权更新响应进行无中心PCA，实现秩约束接收方目标下的全局最优读出。 |
| [^143] | [Toward Out-of-Distribution Generalization in Neural AC Power Flow](https://arxiv.org/abs/2611.23729) | 该论文系统评估了物理信息损失、多电网训练和扰动数据增强对图神经网络潮流模型从小电网向未见过的500节点大电网零样本迁移能力的提升，发现物理信息损失可改善迁移效果，而多电网训练虽与最佳单一电网持平，但仍无法完全缩小与目标电网之间的差距。 |
| [^144] | [Do 3D Point Representations Encode Architectural Functions? Probing Supervised and Pretrained Models on Urban Facades](https://arxiv.org/abs/2611.23720) | 该研究发现冻结的预训练三维点云Transformer（如Utonia）能够编码建筑元素的功能属性（如窗户是否可开启、柱子是否承重），显著优于从头训练的网络和手工描述子，证明预训练是获得建筑功能理解能力的关键。 |
| [^145] | [HypoLoop: Revising Weight-Editing Plans from Scientific Feedback](https://arxiv.org/abs/2611.23718) | 该论文提出HypoLoop，一种将实验反馈与权重编辑决策相关联的原生权重编辑方法族，通过跨轮次迭代修正可编辑参数位置和拟合规则，在22个科学任务上显著降低了预测误差。 |
| [^146] | [Learning through Discernment: when a small language model learns from user feedback on the job, and when it decays](https://arxiv.org/abs/2611.23716) | 该研究发现，设备端小型语言模型在夜间从用户反馈中自我微调学习时，采用“先判断用户是否给出明确评价”的辨识式反馈感知机制，可使模型在两周内保持接近最优性能，但从长期看模型仍普遍呈现先提升后衰退的规律。 |
| [^147] | [MoMoVL: Modality-Aware Adaptive Compression for On-Device Sparsely-Activated Vision-Language Models](https://arxiv.org/abs/2611.23695) | 提出MoMoVL，一种面向端侧混合专家视觉语言模型的模态感知自适应压缩方法，利用视觉与文本token的模态差异优化专家的混合精度量化，实现高效且保护隐私的端侧部署。 |
| [^148] | [Measuring the Conservatism of Latent Safety Filters for World Models](https://arxiv.org/abs/2611.23685) | 提出了“重叠度”这一指标来衡量世界模型潜在安全过滤器因潜在表示不可区分而被迫拒绝安全动作的最小比例，从而可以量化并比较不同潜在表示设计的保守性。 |
| [^149] | [Variable Length Diffuse Everything: Arbitrary Length Generative Modeling on Arbitrary State Spaces](https://arxiv.org/abs/2611.23684) | 提出了可变长度扩散万物（VL-DE）框架，通过基于严谨理论的插入机制而非数据填充，实现了任意状态空间上可变长度多模态数据的简单通用生成。 |
| [^150] | [Evaluating Governed LLM-Driven ML Experimentation: Lifecycle, Provenance, and Failure Metrics](https://arxiv.org/abs/2611.23678) | 本文提出应从实验过程而非仅从结果分数来评估LLM驱动的ML实验系统，构建了由控制平面（而非LLM）管理试验生命周期、编辑范围与保留决策、并将记录写入只追加账本的治理框架，从中计算无效操作率、重复失败率和溯源完整性等治理指标，并在六个ML任务共1,445次试验中验证了同一契约的普遍适用性。 |
| [^151] | [Simpler Tiny Recursive Models](https://arxiv.org/abs/2611.23677) | 提出STRM——一种仅用单一状态和统一更新的更简单微型递归模型，通过训练中速率被抖动的指数移动平均状态更新等关键设计，在不牺牲泛化能力的前提下提升了训练速度和最终性能。 |
| [^152] | [Finetuning VLAs for Dexterous Manipulation: The Right Checkpoint Is All You Need](https://arxiv.org/abs/2611.23676) | 该论文发现，用SFT结合扩散引导强化学习（DSRL）将VLA适配到灵巧手时，SFT最优检查点与DSRL后最优策略并不一致，且冻结VLM、仅微调动作专家模块能得到最可引导的策略，因此选择正确的检查点至关重要。 |
| [^153] | [Communicative On-Policy Distillation: Distilling Black-Box Teacher Capabilities through Communication for Long-Horizon Tasks](https://arxiv.org/abs/2611.23673) | 提出通信式在策略蒸馏（COPD），以自然语言通信为媒介，让学生模型自主决定何时向黑盒教师提问，并将成功的师生交互用于在策略自蒸馏与任务奖励强化学习相结合的训练，从而在长时程任务上取得最强黑盒蒸馏基线 2.46 倍的增益。 |
| [^154] | [A Free Lunch: Gradient Averaging Achieves Optimal Statistical Efficiency in Newton Sketch](https://arxiv.org/abs/2611.23671) | 该论文证明，在草拟牛顿方法中仅需将随机梯度替换为滑动平均（无需额外计算开销、不改变内部求解器收缩界），即可缓解甚至消除草拟带来的计算—统计权衡，实现最优统计效率。 |
| [^155] | [CratesBump: A Version-Bump Dataset for the Rust Ecosystem](https://arxiv.org/abs/2611.23664) | 该论文发布了首个用于预测 Rust 生态系统版本升级的公开数据集 CratesBump（涵盖约 22 万个 crate、超过两百万个连续发布对、每对含 41 个特征和双重标签），并配套提供可从公开数据重新生成该数据集的工具 semver-extract，基准实验表明版本升级预测任务可学习但远未解决。 |
| [^156] | [Same Model, Different People: Numerical Execution Contracts and the Reproducibility of Fairness Audits](https://arxiv.org/abs/2611.23657) | 研究通过因果对照实验证明，即使模型权重与审计代码完全一致，仅仅改变数值计算方式（浮点精度、硬件与软件栈）就会改变阈值附近的个体决策，从而威胁公平性审计结论的跨平台可复现性。 |
| [^157] | [Toaster: Nonlinear Memory from Cascaded Linear Memories and Target Propagation](https://arxiv.org/abs/2611.23648) | Toaster通过将两个线性记忆级联并插入激活函数、利用目标传播近似跨层信号，构建出可在未修改的delta规则核上高效训练的两层非线性记忆，兼具线性记忆的训练速度与深度记忆（MLP）的表达能力。 |
| [^158] | [When Do LLMs Model the Right Target? Problem Formulation in High-Stakes Risk Prediction](https://arxiv.org/abs/2611.23638) | 该研究通过儿童福利、医疗保健和就业三大高风险场景的实验评估发现，在缺乏明确指导的情况下，LLM在构建风险预测问题时无法可靠地识别和处理“决策依赖性”（即历史决策影响观察结果的效度威胁），凸显了将LLM用于高风险决策建模时人工审查问题构建的必要性。 |
| [^159] | [Option-GRPO: Hierarchical RL Against AI Slop in Code Generation for Unseen SDKs](https://arxiv.org/abs/2611.23633) | 该论文提出Option-GRPO分层强化学习框架，仅凭SDK源代码（无需人工示例或测试用例）即可将代码大模型适配到未见过的SDK，通过从源代码中挖掘可复用代码模式（code options）作为课程与动作空间，同时优化功能正确性与代码简洁优雅性，从而抑制AI垃圾代码的产生。 |
| [^160] | [FairPGT: Partitioned Graph Transformers for Dual-Level Fair Graph Clustering](https://arxiv.org/abs/2611.23631) | 提出FairPGT，通过METIS分区与路由token降低注意力计算成本，并在聚类分配（互信息约束促进人口统计平等）和注意力机制（nCGAR自蒸馏实现群体中立）两个层面实现公平图聚类。 |
| [^161] | [What Interpolators Store and What They Express: An Information-Theoretic Framework for Benign Overfitting and Label-Free Model Selection](https://arxiv.org/abs/2611.23630) | 本文提出区分插值器“存储”与“表达”的信息论框架，证明所有插值器存储等量标签噪声故存储类泛化界（如CMI）无法区分它们，而“表达”才能刻画超额风险，从而解释良性过拟合并实现无需标签的模型选择。 |
| [^162] | [Two is better than one: optimizing binary rating scales for online platforms](https://arxiv.org/abs/2611.23626) | 该论文证明在线平台通过为不同用户使用不同的二元评分量表而非单一量表——包括将单一量表拆分为两个、将多量表最优设计刻画为阈值均匀间隔的0-1阶跃函数、以及引入依赖先前反馈的自适应量表——可以显著提升区分物品内在质量的排名能力，其中自适应设计能带来指数级的提升。 |
| [^163] | [Agent Verifiers in Counterfactual Analysis: A Pilot Study Benchmarking Agentic Heterogeneous Treatment Effects](https://arxiv.org/abs/2611.23621) | 本文提出一个包含检索可采性、基于RATE的异质性验证和编码器可采性检验的三层基准协议，用于防范检索泄露与编码器泄露，从而可靠地评估智能体在异质性处理效应估计中的表现。 |
| [^164] | [How Uncertain Is My Action? Self-Correcting Visual Planning with Frozen World Models](https://arxiv.org/abs/2611.23620) | 该论文提出将冻结的潜在世界模型反向运行，通过对比假设偏移下的预测观测与实际观测来检测、估计并补偿执行偏移，使视觉规划器具备自校正能力，检测性能与理论预测相差不到0.05 AUROC。 |
| [^165] | [EEG2MCU: Sensing-to-Execution Co-Design for Wearable EEG Inference](https://arxiv.org/abs/2611.23619) | EEG2MCU通过任务感知的电极蒙太奇选择将电极数量作为硬件设计变量，并结合按需生成INT8区域和跨内存的Halo重用等协同设计技术，在Cortex-M7微控制器上实现了峰值SRAM降低11.38倍、延迟降低2.57倍、能耗降低2.48倍，同时保留了94-96%的全蒙太奇准确率。 |
| [^166] | [Error accumulation in reverse diffusion with an imperfect score: beyond path-space analyses](https://arxiv.org/abs/2611.23617) | 本文突破了传统路径空间分析框架，证明当目标分布满足 Poincaré 不等式时，扩散模型采样过程中的误差累积可以得到有效缓解。 |
| [^167] | [Isthmus: Bridging Supervision Islands Restores Uncertainty and Stability in One-Step Flows](https://arxiv.org/abs/2611.23616) | 该论文揭示单步流采样因监督时间对落在与数据对角线断开的“监督孤岛”上而丢失不确定性与稳定性，并提出用“地峡”桥接孤岛以恢复单步流的不确定性和稳定性。 |
| [^168] | [Optical Neural Networks Should Be Evaluated by Function, Not Only Physics](https://arxiv.org/abs/2611.23610) | 本文提出一种面向功能的三步评估方法，通过将光学神经网络模块与性能匹配的标准数字模块进行对比，来客观衡量其对系统级优势（如速度和能耗）的真实贡献。 |
| [^169] | [Label-Free Role-Aware Segmentation of Classroom Recordings: Recovering the Teacher from Transferable Structure](https://arxiv.org/abs/2611.23609) | 该论文提出了一种无需标签、可在录音设备本地运行的师生语音分割流水线，通过基于会话相对结构而非声音身份的方式来识别教师角色，并能跨教室、语言和声学条件迁移。 |
| [^170] | [JEPA as a Transferable Optimizer: Learning and Understanding Black-Box Optimization from Reward-Agnostic Pretraining](https://arxiv.org/abs/2611.23608) | 提出了基于联合嵌入预测架构的学习型优化器JEPA-PS，通过奖励无关的预训练学习预测优化轨迹中的下一个点而非奖励本身，测试时仅需一次前向传播即可提出候选解，并在15个控制环境中与12种优化器对比验证了其性能，同时增强了对优化器如何利用搜索历史的可解释性。 |
| [^171] | [Entropy-Regularized Integrated Gradients for Stable Feature Attribution](https://arxiv.org/abs/2611.23600) | 提出熵正则化积分梯度（ERIG），通过带温度参数的最大熵softmax加权抑制积分路径上的梯度不规则性，并从理论上证明其归因图对微小输入扰动具有更强的稳定性。 |
| [^172] | [DuoCL: Dual-Path Online Continual Learning](https://arxiv.org/abs/2611.23598) | DuoCL通过结合快速解析学习路径与慢速骨干网络适应路径的双路径框架，在保证在线学习不间断的前提下自适应计算预算，相比基于梯度的基线方法平均提升在线准确率9.67个百分点。 |
| [^173] | [Efficient Variance-Aware Reinforcement Learning in Time-Inhomogeneous Linear MDPs](https://arxiv.org/abs/2611.23596) | 本文提出计算高效的算法 VA-LSVI-UCB++，在时变非齐次线性 MDP 中实现了方差依赖的遗憾界，并消除了以往方差感知分析中即使在确定性环境下仍然存在的 $\tilde{O}(K^{1/4})$ 多项式加性项。 |
| [^174] | [GRAFT: Graph-Regularized Prototype Fields for Realistic Test-Time Adaptation of VLMs](https://arxiv.org/abs/2611.23591) | 该论文提出GRAFT方法，将适应后的原型建模为文本锚点加上经语义类别图上高斯马尔可夫随机场先验部分池化的位移场，并引入显式类别存在模型，实现闭式无梯度的测试时适应，从而在少类别批次或相关数据流等现实场景下优于零样本基线。 |
| [^175] | [Safe to Omit, Worth Omitting? Keeping the Answer and Cutting Latency When an LLM Agent Skips Another's Message](https://arxiv.org/abs/2611.23570) | 该论文提出一种基于无消息置信度读取与有限样本统计检验的机制，使多智能体LLM系统中的接收方智能体能够以高概率保证答案改变率不超过5%地安全跳过辅助智能体的消息，在保持最终答案质量的同时有效降低通信延迟，并揭示了消息增益远小于oracle上限的量化证据。 |
| [^176] | [Bounded-Order Non-Identifiability in Latent Structure Block Models: The Fiber in Closed Form](https://arxiv.org/abs/2611.23547) | 该论文为位置位于K条射线上的潜在结构块模型给出了有限阶模体计数下不可辨识配置的一维纤维闭式解，并证明友谊图F_{K-1}是唯一边最小分离子，因此不存在对所有K都适用的固定模体阶数。 |
| [^177] | [Stability of Parallel Transport Convolution under Riemannian Metric Perturbations](https://arxiv.org/abs/2611.23543) | 本文证明了在紧流形上，平行输运卷积的输出在 $L^2$ 意义下对黎曼度量的 $C^2$ 扰动具有连续稳定性，为此发展了测地线、平行输运、雅可比场与指数映射的扰动估计技术。 |
| [^178] | [Can LLMs Understand What They Predict? Evaluating Latent Causal Structure Inference in Financial Markets](https://arxiv.org/abs/2611.23533) | 该论文提出LUCID基准，评估大语言模型能否从事件描述和历史金融数据中推断潜在的因果市场结构，发现模型生成的因果图通常无法改善重构质量，表明预测成功并不等于真正理解了底层机制。 |
| [^179] | [Balancing Partition Forest for Causal Weighting](https://arxiv.org/abs/2611.23524) | 该论文提出平衡划分树与平衡划分森林（BPT/BPF），通过在协变量—处理空间上以 Bregman 散度增益驱动分裂的矩形划分来估计理想平衡权重，并从理论上证明其总体残差可界定逆概率加权和双重稳健估计量的偏差。 |
| [^180] | [From Arraignment to Resolution: Transformer Models of Court Cases](https://arxiv.org/abs/2611.23518) | 本文提出基于Transformer的时间序列模型，将法院案件建模为事件序列以预测刑事案件中下一诉讼程序的结果，从而提升司法决策的透明度与可监督性。 |
| [^181] | [Optimal Control of RKHS Diffeomorphic Flows for Classification and Generative Modeling](https://arxiv.org/abs/2611.23497) | 该论文提出了一个统一的最优控制框架，用RKHS参数化的微分同胚流同时解决分类与密度估计任务，其精确且确定性的散度使得二阶优化器切实可行，同时避免了现有连续归一化流的随机迹估计和势流限制。 |
| [^182] | [JEPA for astronomy: multi-survey spectral alignment puts stellar parameters on the leading principal axes](https://arxiv.org/abs/2611.23492) | 该论文提出将JEPA（联合嵌入预测架构）而非对比学习用于多巡天恒星光谱对齐，使恒星物理参数（如有效温度、表面重力、[Fe/H]和[alpha/Fe]）恰好落在嵌入空间的前几个主成分轴上，从而仅通过线性读出即可恢复Kiel图等恒星物理结构。 |
| [^183] | [SpecLook: Speculating Where to Look for Fast and Reliable Agentic MLLMs](https://arxiv.org/abs/2611.23483) | SpecLook 提出了一个无需训练的“推测-验证”三层框架，通过让小模型推测智能体下一步的放大观察——该观察可被验证、纠正并用于检验答案——从而显著加速多模态大语言模型智能体的推理并提升答案可靠性。 |
| [^184] | [GUARD: Stabilizing Recursive Self-Improvement with Regret Guidance](https://arxiv.org/abs/2611.23476) | GUARD提出了一种无评估器的闭环框架，通过将任务分解为执行前自检并用方向性遗憾引导框架更新，稳定AI智能体的递归自我改进，避免局部退化、技能遗忘和记忆膨胀。 |
| [^185] | [AllotNet: an Interpretable Mixture-of-Experts Choice Model for Moral Judgement in Humanitarian Aid Allocation, Validated against a Planted Synthetic Population](https://arxiv.org/abs/2611.23474) | AllotNet是一个可解释的混合专家条件Logit模型，通过预注册的六语言工具Allot捕获多文化道德判断以辅助人道主义援助分配，其规划器构建审核队列而非拒绝任何人，并通过植入式合成人口验证了模型有效性。 |
| [^186] | [FACT: Fixed-Budget Allocation and Cross-Step Training for Speculative Decoding](https://arxiv.org/abs/2611.23463) | 提出FACT实验框架，系统研究草稿模型训练、候选token分配与执行条件对推测解码接受率和延迟的联合影响，并提出对早期步骤加重权重的跨步训练方法，使接受率相比传统分离训练提升15-24%。 |
| [^187] | [Where Global Context and Weight Sharing Work in Compact Vision Backbones](https://arxiv.org/abs/2611.23446) | 该研究发现，在EfficientNet-B0骨干网络的P3、P4、P5阶段插入轻量级零门控残差自注意力适配器可将ImageNet-1k精度提升至74.02%，而单纯用BiFPN多尺度特征金字塔包装骨干则无法超越普通MBConv主干。 |
| [^188] | [Why Sub-4-Bit Vision CNNs Collapse and Where Mixed-Precision Recovery Stops](https://arxiv.org/abs/2611.23445) | 本研究系统剖析了视觉CNN在4比特以下量化时崩溃的机理，发现崩溃高度非线性且集中在后期逐点投影和深度可分离层，并提出了结构化的多步骤混合精度恢复瀑布策略来挽救精度。 |
| [^189] | [Conformally Calibrated Adaptive Computation in Neural Operators](https://arxiv.org/abs/2611.23440) | 该论文提出一种基于保形校准的自适应计算方法，在固定误差预算下为每个输入选择神经算子的傅里叶模态数或通道宽度并提供统计保证，同时发现模态选择虽缩减了谱支持范围，却因全网格变换主导计算成本而几乎无法降低推理延迟。 |
| [^190] | [When Does Subspace Direction Matter for LoRA? Regime Analysis of the Magnitude Principle in Few-Shot Adaptation](https://arxiv.org/abs/2611.23428) | 本文证明LoRA初始子空间方向的重要性取决于情境：在少样本且初始化子空间能捕捉任务结构时方向对齐至关重要，并提出对齐系数ρ来预测增益、以及PivotLoRA进行任务对齐初始化。 |
| [^191] | [Stability Does Not Mirror Plasticity: A Module-Level View of Declarative and Procedural Learning in Transformers](https://arxiv.org/abs/2611.23427) | 该论文揭示了Transformer中可塑性与稳定性在模块层面的双重分离——QK与VO投影在程序性学习中最具可塑性而MLP在陈述性记忆中最具可塑性，并且稳定性并不镜像可塑性：学习新程序时更新QK可保留过半陈述性记忆，而更新VO会丢失约五分之四。 |
| [^192] | [Incremental Roseland: Batch-wise Landmark Diffusion via Kernelized Sequential Karhunen--Lo\`eve Update](https://arxiv.org/abs/2611.23414) | 提出增量式Roseland（iRoseland），通过核化序贯Karhunen–Loève方案对流式增长的地标亲和矩阵的SVD进行增量更新，实现仅需单次数据遍历、有界内存、并可跟踪分布漂移的地标扩散谱嵌入。 |
| [^193] | [RFBench: A Simulator-Grounded Multimodal Benchmark for Passive RF Layout Design and Understanding](https://arxiv.org/abs/2611.23410) | 该论文提出了RFBench——首个基于ADS Momentum仿真器的无源射频版图设计与理解多模态基准，包含10,000个带有多模态对齐表示的双端口平面版图和七项任务，用以系统评估多模态语言模型连接射频设计意图、版图几何与电磁行为的能力。 |
| [^194] | [Which Tokens, in What Order, and How Fast? A Theoretical Analysis of On-Policy Distillation Objectives](https://arxiv.org/abs/2611.23397) | 本文首次从理论上剖析在线策略蒸馏目标的选择机制，揭示序列级OPD存在导致反向学习顺序的“未来一致性偏差”，而Token级目标可避免该偏差，并阐明了散度选择如何影响Token的优先级、学习顺序与收敛效率。 |
| [^195] | [QDA: Closed-Loop LLM Quantization Design with Agents](https://arxiv.org/abs/2611.23388) | QDA提出了一个基于智能体的闭环系统，通过KL散度快速评估候选量化方案并自动化整合不同代码库的方法，为每个模型和精度自动设计最优的大语言模型训练后量化方案。 |
| [^196] | [Neural Circuit Architectural Priors for Rat Locomotion](https://arxiv.org/abs/2611.23364) | 该论文通过协同改进生物力学大鼠身体模型（Rat v2）、融合实验数据的神经回路架构先验（Rat NCAP）以及训练基础设施，构建了能够研究复杂身体与神经回路相互作用产生大鼠运动行为的具身模拟框架。 |
| [^197] | [TRACE-MAS: Diagnosing and Repairing Collaboration Failures in Multi-Agent Systems](https://arxiv.org/abs/2611.23362) | TRACE-MAS是一个细粒度诊断与修复框架，通过从良性轨迹中学习正常协作模式、检测智能体交互层面的偏差、进行源头归因并实施最小干预，来识别和修复多智能体系统中源于交互的协作故障。 |
| [^198] | [A Survey on the Use of Machine Learning within Computer Algebra](https://arxiv.org/abs/2611.23361) | 本综述系统梳理了机器学习在计算机代数领域的应用——既包括优化柱形代数分解、Gröbner基计算、符号积分等现有算法，也包括直接符号计算，并比较了从决策树到Transformer等多种机器学习方法。 |
| [^199] | [Prioritize, Don't Decide: What We Learned Auditing a Deployed Human-in-the-Loop CV Sorter](https://arxiv.org/abs/2611.23359) | 本文审计了一个部署在国家学术招聘平台上的人在环简历优先排序系统——委员会按其推荐顺序只需阅读一半简历即可覆盖 95% 的相关候选人；审计进一步发现该评测简历池仅凭文档长度即可排序，使精确率类指标失去参考意义，这一发现直接改变了产品设计。 |
| [^200] | [Implementing Binary Transformers with Approximate Memories with Permutation Invariance Enabled Selective (PIES) Protection](https://arxiv.org/abs/2611.23358) | 本文提出利用Transformer对行列置换的不变性，设计了一种低开销的选择性保护方案（PIES），可显著提升基于近似存储器的二值Transformer在存储错误下的可靠性。 |
| [^201] | [SIGMA: speaker-isolated, symptom-structured multimodal estimation of depression severity from clinical interviews](https://arxiv.org/abs/2611.23348) | SIGMA框架通过说话人分离、跨模态交叉注意力编码以及基于共享潜在严重程度特质的症状联合预测，在临床访谈中实现了抑郁严重程度的最先进多模态估计。 |
| [^202] | [BiMa-ECGSR: reconstructing the 12-lead ECG from reduced lead sets with bidirectional state-space models](https://arxiv.org/abs/2611.23345) | BiMa-ECGSR提出了一种基于双向Mamba状态空间模型的编码器-解码器架构，能够从单导联或少量导联高精度地重构完整的12导联心电图，并在多个大规模内外部数据集上显著优于传统线性重构方法。 |
| [^203] | [Confidence-Based Materials Data Quality Assessment via Domain Knowledge and Graph-Structured Learning](https://arxiv.org/abs/2611.23342) | 该论文提出了一种融合领域知识与图结构学习的置信度材料数据质量评估框架，将数据质量建模为连续置信度估计而非二元异常识别，实现了样本级评估和基于历史参考的批次比较。 |
| [^204] | [Optimize the Cob, Then the Kernel: Optimizing Beam Search for Generative Recommendation Inference](https://arxiv.org/abs/2611.23339) | 该论文提出利用Tree Attention将束搜索的各条束沿序列维度拼接以消除冗余KV缓存访问，并结合静态KV缓存分配、图编译、算子融合与词表约束解码等系统优化，在不依赖自定义内核的前提下大幅提升了生成式推荐推理中束搜索的效率。 |
| [^205] | [A Differential Theory of Quality-Resource Trade-offs for AI Model Selection: The Multiresource Frontier and Its Minimax Price](https://arxiv.org/abs/2611.23324) | 该论文提出一种微分理论，将AI模型选择的质量-资源前沿表示为与价格无关的平面曲线，通过有效资源通道数与剩余对数误差深度之间的内在关系刻画了多资源权衡的几何结构与极小极大价格。 |
| [^206] | [HydroMind: A Checkable Agent for Autonomous Hydrological Research](https://arxiv.org/abs/2611.23323) | 该论文开发了面向通用智能体的水文建模系统HydroMind及其建模工具和研究流程，首次检验了当自主智能体自行决定水文模型评估方案时，其结论能否被自身数据、代码和结果支持并供读者核查。 |
| [^207] | [Error Analysis of Deep Learning](https://arxiv.org/abs/2611.23321) | 本文以误差分解（逼近、统计与优化误差）为统一视角系统综述深度学习理论，梳理从单个误差分量分析到综合框架的发展脉络，并结合多个应用领域的实例阐明如何获得已训练网络的理论保证。 |
| [^208] | [Measuring What Matters in the Age of AI: The Research Software Metrics Landscape](https://arxiv.org/abs/2611.23253) | 该论文基于49篇文献系统梳理了研究软件度量指标的全景，将其归纳为八大类并从数据来源、效度、自动化潜力和滥用风险等维度进行分析，同时指出AI辅助开发正使现有指标体系的假设面临失效。 |
| [^209] | [What Moves the Ceiling on Automated Audit: from General Code to Scientific Code, Results and Data](https://arxiv.org/abs/2611.23250) | 本文在真值基准上量化了“异厂商模型审计+确定性检查”这一自动化审计流程的能力上限：通用代码上召回率仅约 30% 且无饱和点，可通过阅读模型、规则与决策规则提升；而在科学结果核查中，数字溯源检查能识别全部 49 份不一致报告，模型审计则可额外发现 25 份内部一致伪造中的 23 份。 |
| [^210] | [Exploring Parkinson's Diagnosis Using Machine Learning Techniques](https://arxiv.org/abs/2611.23248) | 该研究将多种机器学习算法应用于包含58名患者的帕金森病数据集，发现决策树和逻辑回归准确率最高（72-76%），而支持向量机在识别阳性病例方面灵敏度最优（高达97.56%）。 |
| [^211] | [Budgeted Proper Loss Design for Uncertain Decision Thresholds](https://arxiv.org/abs/2611.23247) | 本文在阈值分布属于有限指定族的设定下，推导出使代理遗憾到阈值平均决策遗憾的最坏情况乘数最小化的最优适度损失权重——即在规定下限处截断阈值密度的重标度上包络线，并证明了其锐利性、最优性与稳定性。 |
| [^212] | [Chebyshev Polynomial Zonotope Abstractions for Certified Reachability of Learning-Enabled Systems](https://arxiv.org/abs/2611.23246) | 该论文证明泰勒展开因复平面上紧邻实轴的奇点而存在不可逾越的可达集宽度上限，并提出基于切比雪夫多项式带状体的抽象方法，在同等阶数下不受该上限约束且认证宽度随阶数呈几何级数增长，从而实现对学习使能系统更精确的认证可达性分析。 |
| [^213] | [Response-Relevant Information Is Not Sufficient for Neural Training-State Equivalence](https://arxiv.org/abs/2611.23245) | 尽管权重谱、表示几何和优化器几何等描述子携带响应相关信息并能预测训练扰动后的响应偏差，但它们仍不足以作为判定两个神经网络检查点训练状态等价的充分依据。 |
| [^214] | [Goemans's Unsplittable-Flow Conjecture: The Anatomy of AI-Assisted Counterexample Discovery](https://arxiv.org/abs/2611.23244) | 本文开发了一个前向反例构造流程，系统剖析了AI辅助发现的Goemans不可分割流猜想七顶点反例的结构原理，揭示了该反例为何成立、其所属的更广泛反例家族以及决定反例出现位置的关键结构特征。 |
| [^215] | [Biologically-Inspired Spike Encoding and Optimization for Spiking Neural Networks](https://arxiv.org/abs/2611.23242) | 该论文将生物启发的脉冲编码方案（频率编码、延迟编码、增量调制、DoG 滤波）与自适应阈值替代梯度和 K-FAC 二阶优化相结合，显著提升了脉冲神经网络在容量受限条件下的准确率和脉冲稀疏性。 |
| [^216] | [GridForge: An Open Physics-Informed Digital Twin and Benchmark Suite for UK 11 kV Underground Distribution Cables](https://arxiv.org/abs/2611.23241) | 本文提出了GridForge——一个开源的物理信息数字孪生与基准测试套件，为英国11千伏地下配电电缆领域提供了首个公开基准数据集、经验证的IEC标准热模型以及物理信息神经网络（PINN）代理模型，填补了该领域在故障诊断与预测模型评估协议上的空白。 |
| [^217] | [Neural Network-Based Modeling of Hysteretic Systems: From Prandtl and Preisach Neural Networks to Neuro-Madelung and Neuroplastic Continuous Models](https://arxiv.org/abs/2611.23240) | 本综述系统梳理了从PNN、PreisachNN到GPNN、NMM、EPNN和DPCNN等一系列基于神经网络的迟滞系统建模方法的发展，涵盖其记忆表征机制、网络架构、训练算法及在非线性结构与材料系统中的应用。 |
| [^218] | [Multimodal Masked Diffusion Autoencoder for Modeling Impaired Driving](https://arxiv.org/abs/2611.23211) | 该论文提出多模态掩码扩散自编码器 ID-MAE，通过自监督方式重建被掩码的注视与轨迹数据来增强受损驾驶预测，并发布了包含 109 人、103 小时驾驶数据及十种受损诱发条件的大规模模拟驾驶数据集 DIVERT，显著提升了受损驾驶推断效果。 |
| [^219] | [MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study](https://arxiv.org/abs/2611.23158) | 本研究首次在统一的多语言评估协议下，用200种语言和9000多个评估设置系统比较八种LoRA方法，发现小规模搜索无法可靠预测方法在大规模下的相对表现，且LoRA虽能提升翻译数据拟合度但会持续损害阅读理解能力。 |
| [^220] | [Pareto-Informed Smart "Predict, then Optimize" for Multi-Objective Combinatorial Problems](https://arxiv.org/abs/2611.23076) | 提出了一种帕累托感知的SPO+扩展方法，利用真实帕累托前沿构建与采样偏好向量对齐的目标决策并最小化遗憾，克服了现有决策聚焦学习方法在多目标组合优化中因简单标量化而无法捕捉丰富权衡关系的局限。 |
| [^221] | [ConserveEdit: Source-Anchored Transactions for Drift-Bounded Iterative Image Editing](https://arxiv.org/abs/2611.22724) | ConserveEdit提出了一种与编辑器无关的事务式控制器，通过从源图像编译的契约将授权区域与受保护内容分离，确保迭代图像编辑中授权区域外的像素保持位级精确不变，且漂移与迭代次数无关。 |
| [^222] | [Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration](https://arxiv.org/abs/2611.22275) | 该论文提出Heave-Ho去中心化框架，通过多智能体后训练实现物理耦合的人形机器人协调控制，结合视觉-语言规划器与基于语音的领导者-跟随者机制，在无需网络通信的情况下实现最多十台（含异构）人形机器人的长时程协作移动操作。 |
| [^223] | [Synthetic Crystal Data Generation for Diffusion Pretraining in Material Property Prediction](https://arxiv.org/abs/2611.22140) | 该论文提出一种结合扰动-弛豫、MatterGen生成式采样等三种互补策略的合成晶体数据生成流水线，为材料性质预测的扩散预训练提供大规模且结构多样的弛豫结构，突破了依赖昂贵DFT弛豫获取预训练数据的瓶颈。 |
| [^224] | [Auditing World-Model Explanations in Retail Forecasting: Faithfulness, Accuracy and Price Response](https://arxiv.org/abs/2611.20811) | 研究发现，零售预测中的世界模型解释即使通过了八项审计检查（如构造性、稳定性、价格方向和与真实销量的一致性），其促销提升估计仍可能高出真实值1.2至1.4倍，价格响应被夸大近5至6.5倍，表明通过常规检查并不保证解释的忠实性与准确性。 |
| [^225] | [Passable for Whom? Few-Shot Personal Calibration of Sidewalk Accessibility Perception](https://arxiv.org/abs/2611.19839) | 该论文提出 PassCal 框架，将共享的场景描述与个人解释分离，仅需从未见过的移动辅助工具用户的少量回答，即可个性化预测其对人行道场景的可达性判断。 |
| [^226] | [Ambiguous Examples Help, But Full Data Wins: Revisiting Dataset Cartography for Extractive Adversarial QA](https://arxiv.org/abs/2611.19763) | 本文在更严格的抽取式对抗问答设置下重新审视数据集制图法的结论，发现仅用33%模糊样本子集训练虽有帮助，但仍不敌全量数据训练在对抗性评估中的表现。 |
| [^227] | [ReCALM: A Residual-enhanced Crowdsourced Analytic Learning Mechanism for Distributed Continual Service Construction](https://arxiv.org/abs/2611.19276) | 该论文提出ReCALM，一种残差增强型众包解析学习机制，通过引入深度非线性表示能力克服传统解析学习受限于单层线性网络而导致的欠拟合问题，同时保持无梯度的闭式递归求解以避免异构数据持续更新中的灾难性遗忘，从而支持分布式众包环境下的持续AI服务构建。 |
| [^228] | [FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models](https://arxiv.org/abs/2611.18627) | 提出 FlashREINFORCE——一种无评论家、每个提示词仅单次采样的异步强化学习框架，通过单批次 REINFORCE、序列信赖域与样本均值优化，在策略滞后约四步的情况下仍能稳定训练智能体语言模型，同时保持提示词覆盖度。 |
| [^229] | [Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders](https://arxiv.org/abs/2611.18179) | 本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。 |
| [^230] | [LoCoJAX: Host-Local Coordination for Small-Model RL on TPUs](https://arxiv.org/abs/2611.16301) | 该论文发现单一控制器协调模式在小模型强化学习中开销占比过高（0.6B模型下主机编排占步骤时间36%），并提出LoCoJAX通过主机本地执行rollout和奖励计算来提升TPU上小模型RL的效率。 |
| [^231] | [A Matching Lower Bound for Online Smooth Calibration](https://arxiv.org/abs/2611.16097) | 本文通过构造自适应对手和遗忘对手，证明在线平滑校准误差的极小极大下界为 Ω(√T)，填补了已知的上下界差距，确立了平滑校准误差的极小极大最优性。 |
| [^232] | [Probabilistic State Space Enumeration in Neural Network Computation](https://arxiv.org/abs/2611.15062) | 该论文受网络可靠性分析启发，提出概率状态空间枚举方法（PSSE），将Transformer中MLP的密集输出表示为通道组联合状态上重要性校正掩码输出的期望，并配合NORDER-II算法高效枚举高概率状态，实现具有可证明误差界限的高效近似。 |
| [^233] | [Prescriptive Learning of Linear Policies with Feasibility Guarantees for Binary Decision-Making](https://arxiv.org/abs/2611.10923) | 本文提出一种规范性学习框架，通过在所有训练场景上联合优化线性决策策略与可行分配，为不确定性下的二元决策提供可行性保证，并解决了预测式学习缺乏反事实认知且在分布偏移下易产生不可行决策的问题。 |
| [^234] | [MAGE: Towards Generalizable Multi-timescale EEG Representations](https://arxiv.org/abs/2611.09928) | 提出了MAGE脑电基础模型，通过新颖的多时间尺度注意力机制和可训练的动态聚合模块，自适应地捕捉并整合脑电信号中跨越多个时间尺度的神经动态信息，突破了现有模型固定时间尺度的局限。 |
| [^235] | [On Agentic Manipulation Policies](https://arxiv.org/abs/2611.09892) | 提出AMP1智能体式操作框架，证明通用AI模型直接控制机械臂可零样本超越VLA和WAM等专用策略，在RoboLab-120上达到85.6%的成功率。 |
| [^236] | [Calibrated Uncertainty Quantification on TDC ADMET: What Can Be Resolved Across 22 Datasets?](https://arxiv.org/abs/2611.09373) | 在22个TDC ADMET数据集的基准测试中，集成蒙特卡洛dropout取得了最佳平均排名，但统计上无法将其与其他集成方法或LightGBM明确区分开，说明该基准的分辨能力有限。 |
| [^237] | [Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning](https://arxiv.org/abs/2611.08144) | 该论文发现强化学习中环境与训练器边界处的静默故障几乎无法通过学习曲线等基于结果的检查发现，而直接检查运行过程中跨越边界传输的数据本身则能检测到每一个注入的故障。 |
| [^238] | [Systolic Linear Attention: Co-Designing Tile-Grouped Gated Delta Attention for TPUs](https://arxiv.org/abs/2611.06828) | 提出了脉动线性注意力（SLA），针对TPU的256×256脉动矩阵单元重新设计门控Delta线性注意力——将键扩展至256、把衰减门分组为64通道瓦片并用全分块矩阵乘法的单一融合内核计算——在TPU v7x上比Tokamax KDA内核快1.6至2.5倍。 |
| [^239] | [Training Autoregressive Language Models on Rounded, Bounded, and Censored Numerical Labels via Exact Likelihood](https://arxiv.org/abs/2611.02461) | 提出边界边缘化方法，首次通过精确似然让自回归语言模型在舍入、有界和删失等区间型数值标签上训练，仅用至多两条对数长度的边界路径即可计算精确概率和梯度。 |
| [^240] | [Profile-aware diagnostics of discarded parameter-Hessian mass](https://arxiv.org/abs/2611.01335) | 提出一种轮廓感知的诊断框架，通过在不同分组规模和组内曲率轮廓参考下测量参数Hessian的块外平方占比，来量化块对角曲率模型所丢弃的曲率质量，并评估候选修正的实际效果。 |
| [^241] | [Your LR Schedule May Not Be Principled: Effective Learning Rate Scheduling in Weight-Normalized Training](https://arxiv.org/abs/2610.40362) | 本文发现有效学习率轨迹是权重归一化与权重衰减训练损失差异的关键因素，并据此推导出以 1/√t 衰减为基础的 AIR 最优学习率调度，在大规模 MoE 预训练中全面超越余弦等传统调度。 |
| [^242] | [TGTV: Interpretable Multivariate Time-Series Anomaly Detection in UAV Telemetry](https://arxiv.org/abs/2610.38106) | 本文提出TGTV框架，将时空图学习与大语言模型相结合，实现无人机多变量遥测的可解释异常检测和可追溯故障分析。 |
| [^243] | [FOCUS: Active Critic Learning for Long-Horizon Language Model Agents](https://arxiv.org/abs/2610.35801) | FOCUS是一种强化学习方法，通过选择性分支对预计最能降低估计误差的状态进行续写采样，实现评论家网络的主动学习并获得更准确的价值估计，从而解决长程语言模型智能体中稀疏奖励导致的信用分配难题。 |
| [^244] | [Shared Context, Separate Engines: Streaming Vision--Language Inference on Edge SoCs](https://arxiv.org/abs/2610.35309) | 提出了一种在边缘SoC上协调CPU、GPU、NPU等异构加速器并发执行有状态流式VLM推理的系统，通过将流的版本化上下文与各引擎的KV状态分离，实现任意引擎可接续处理数据流，从而支持流式生成、并行序列生成和状态感知调度。 |
| [^245] | [Geometric knowledge spaces, Attention pairing and Architectures](https://arxiv.org/abs/2610.29526) | 本文提出一个几何化框架，通过形式化“知识世界”与“注意力配对”的概念，统一推导出标准注意力、因果注意力、线性注意力、张量注意力和循环神经网络等多种架构，并引入新的几何空间以启发未来神经网络架构的设计。 |
| [^246] | [ProteinGeoBench: Frontier Structural-Biology Agents Are Fluent at Lookup but Lost in Geometry](https://arxiv.org/abs/2610.26829) | 本文提出包含131个结构生物学任务的ProteinGeoBench基准，通过对4,249次模型运行的分析发现，前沿模型擅长事实查找但普遍在基于原子坐标的几何计算任务上失败，揭示了当前模型几何推理能力的重大短板。 |
| [^247] | [The Effectiveness and Durability of Pretraining Data Filtering in Protein Language Models](https://arxiv.org/abs/2610.26651) | 该研究利用ESM-2蛋白质语言模型发现，预训练数据过滤能有效削弱模型对病毒蛋白（尤其是人类感染病毒）的能力且随模型规模增大而更有效，但这种防御可通过约80美元成本的低成本微调轻易被恢复，揭示了预训练过滤作为安全保障的局限性。 |
| [^248] | [Typed Decision Models under Prevalence Shift and Label Budgets: Decision Training, Calibration, and Implicit Priors](https://arxiv.org/abs/2610.26325) | 本文在流行率偏移与标签预算约束下系统比较三个类型化决策模型与其骨干模型及微调分类器，发现厂商决策训练并非总是有益，其校准优势因任务类型而异。 |
| [^249] | [Feasibility Ceilings Before Learning: Sizing and Timing Bounds for Drone-Mesh Interception](https://arxiv.org/abs/2610.20823) | 本文推导了无人机拦截网格在策略选定之前的性能闭式界限，并证明当智能体间距至少为2r时，平面与体积覆盖率分别存在90.69%与74.05%的固有可行性上限。 |
| [^250] | [Reinforcement Learning for Large-Scale Lifelong Multi-Agent Path Finding with Shared Spatial Fields](https://arxiv.org/abs/2610.18186) | 该论文提出一种以吞吐量为奖励、端到端训练的强化学习导航策略，通过卷积网络每时间步计算一次共享空间场并由紧凑的目标条件算子导出各智能体移动偏好，在万级智能体的终身多智能体路径规划中达到与最强基线相当或更高的吞吐量，且无需重训练即可泛化部署到更大规模的未见布局，平均执行时间低于10毫秒。 |
| [^251] | [Interaction-as-a-Judge](https://arxiv.org/abs/2610.17895) | 提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。 |
| [^252] | [One Checkpoint, Many Runs: Recipe Cards for Fine-Tuned Models](https://arxiv.org/abs/2610.16825) | 论文提出“模型配方”概念，主张模型卡应披露一次运行达不到性能门槛的失败概率，并通过对 300 张模型卡的调查和 2,124 次微调运行证明，仅以均值、中位数或 bootstrap 区间作为验收标准常常会误判通过失败的运行。 |
| [^253] | [LoongSage: A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes](https://arxiv.org/abs/2610.16352) | LoongSage是一个智能体原生、完全异步的生产级强化学习框架，基于Megatron与SGLang构建且核心仅约1.4万行代码，通过统一编排白盒与黑盒智能体的AgentFlow范式，显著降低了前沿大语言模型智能体强化学习的采用、使用与调试开销。 |
| [^254] | [An Invariant Tangent-Angle Descriptor and a Band U-Net for 2D Fragment Adjacency Prediction](https://arxiv.org/abs/2610.09459) | 该论文提出了一种在构造上对碎片旋转和轮廓起始点选择均不变的切线角描述子来替代原有的局部评分，并用带状U-Net替换最终分类器，从而改进了基于轮廓的二维碎片邻接预测的两阶段架构。 |
| [^255] | [CADFather: Autonomous CAD Reconstruction through Coordinated Tool Use](https://arxiv.org/abs/2610.09127) | CADFather是一个自主智能体系统，通过视觉语言助手协调学习型/算法型提案工具与数值优化等多种互补工具，从3D网格中重建出可编辑的参数化CAD程序。 |
| [^256] | [Considering Context: When World Models Need Context Encoders](https://arxiv.org/abs/2610.06651) | 该论文提出“预测充分性”这一形式化指标，用于量化上下文对基于模型的强化学习预测的真实增益，并证明世界模型是否需要上下文编码器取决于环境先验而非MDP类别，据此建立了上下文感知算法的分类框架。 |
| [^257] | [Beyond In-Distribution Preservation: Recovering Generalization in Quantized VLAs via Vulnerability-Oriented Tuning](https://arxiv.org/abs/2610.05745) | 该研究揭示量化VLA模型在环境扰动下虽保持分布内性能却变得脆弱，发现动作差异集中于少数脆弱状态且教师指导在高差异状态提升泛化、在低差异状态反而损害泛化，据此提出面向脆弱状态的选择性调优方法以恢复量化模型的泛化能力。 |
| [^258] | [When the noncommutative AM-GM inequality holds](https://arxiv.org/abs/2610.04874) | 本文证明当 $n\ge 2\lceil m/2 \rceil^2$ 时非交换AM-GM不等式成立，其核心方法是通过构造基于布尔立方体上切比雪夫节点的顶点测度，将问题转化为求积估计。 |
| [^259] | [DV-Lens: Revealing the Functional Organization of Language Model Parameters](https://arxiv.org/abs/2610.04489) | 提出DV-Lens参数级可解释性框架，通过估计模块特定的雅可比矩阵将语言模型参数投影到下游词汇空间，从而揭示参数的功能组织及其与模型能力的关系。 |
| [^260] | [What Does a Harness Buy? Tokens, Mostly](https://arxiv.org/abs/2610.04433) | 该研究通过对七个模型在三个生产级框架和多个基准上的重复对照实验发现，没有任何框架能稳定提升性能，更换框架带来的差异仅有 2 到 5 个百分点、效果与重新运行一次相当，框架带来的分数提升主要源于更多的 token 消耗。 |
| [^261] | [PoCoFL: POlicy-COmpliant Federated Learning](https://arxiv.org/abs/2610.03650) | 提出了PoCoFL框架，通过将联邦学习类型、策略语义与密码学实现三者解耦，并借助承诺和非交互式零知识证明来验证客户端贡献的策略合规性，从而实现适用于多种拓扑、角色和聚合方式的通用可验证联邦学习。 |
| [^262] | [When Is Accuracy Evidence? A Unified Theory of Generalisation, Validation, and Information Fusion](https://arxiv.org/abs/2610.03465) | 该论文提出一个统一的指数框架，将交叉验证准确率转化为真实风险的保守上界，并通过有效折数Keff证明：当各折数据强相关时，单纯增大交叉验证折数K并不能增强统计证据。 |
| [^263] | [CriticHack: Evaluating Visual Rewards Under Robot Policy Optimization](https://arxiv.org/abs/2610.02527) | 该论文揭示，用学习型视觉奖励模型优化机器人策略时，奖励分数与任务成功率可能同时上升、看似健康，但实际上会显著放大“作用于错误物体”的隐蔽失败，而这一现象在使用模拟器真实任务完成信号训练时并不会出现。 |
| [^264] | [IQS-BO: In-Context Query Selection for Bayesian Optimisation](https://arxiv.org/abs/2610.01269) | 本文提出IQS-BO，一种通过在合成先验上进行监督学习来学习贝叶斯优化查询决策的PFN，可在单次前向传递中完成查询选择，从而避免了代理模型重新拟合和采集函数的数值最大化。 |
| [^265] | [Deep Learning-Based Tri-Hybrid Multi-User MIMO Precoding: The Blessing of EM-Reconfigurable Antennas](https://arxiv.org/abs/2609.39167) | 本文提出基于Conformer神经网络架构的三混合预编码网络Tri-PNet，通过将电磁可重构天线引入的电磁域预编码与模数混合预编码进行联合设计，显著提升了宽带多用户MIMO-OFDM系统的频谱效率。 |
| [^266] | [SANTA++: Sampling Attention through Representative Keys](https://arxiv.org/abs/2609.35629) | SANTA++ 是一种无需训练的随机注意力方法，通过将 KV 缓存分组并用代表性键进行重要性采样，仅访问 28-31% 的 KV 缓存即可保留 95-98% 的稠密注意力性能，显著降低长上下文推理的内存开销。 |
| [^267] | [Elastic Selective Spectral Hybrids for Train-Once, Export-Many Budgeted Inference](https://arxiv.org/abs/2609.32486) | 提出弹性选择性谱混合模型（ESSH），通过输入相关的衰减与门控机制赋予谱状态空间模型选择性记忆能力，并结合滑动窗口注意力，实现一次训练即可导出适配多种计算预算的高效语言模型。 |
| [^268] | [Spread and Scale: What Determines Whether Test-Time Budget Allocation Pays](https://arxiv.org/abs/2609.27917) | 本文通过预先注册的验证性实验发现，工作负载中实例难度的离散程度是决定测试时预算重新分配是否划算的关键属性，并且即使计入分配策略自身消耗的预算成本，该策略依然可能带来收益。 |
| [^269] | [Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2609.23697) | 提出TrustMOPD方法，以专家模型RL训练前后相对于共享参考模型的位移作为token级可靠性代理，实现无标签的多教师加权蒸馏监督分配，将学生模型性能恢复率从54.4%大幅提升至91.5%。 |
| [^270] | [Perplexity Cost Understates What Activation Quantisation Breaks](https://arxiv.org/abs/2609.23125) | 该论文发现困惑度作为聚合指标会低估激活量化的真实损害——即使困惑度仅上升1.2至1.5倍，归纳能力几乎无损（保持0.959）而检索能力已骤降至0.554，且这种结构性差距并非仅由量化误差幅度造成。 |
| [^271] | [Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time](https://arxiv.org/abs/2609.22226) | 该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。 |
| [^272] | [Cross-Architecture Foundation-Model Distillation for Edge Flood Segmentation](https://arxiv.org/abs/2609.20441) | 该论文通过将3亿参数的地理空间基础模型蒸馏到仅70万参数的EfficientViT-B0学生模型中，并利用教师模型对未标注Sentinel-2影像生成伪标签来扩展训练数据，实现了可在边缘设备部署的高性能洪水分割模型。 |
| [^273] | [Risk-Aware World Modeling with Flow-Guided Occupancy Evolution for Selective Trajectory Planning in Automated Driving](https://arxiv.org/abs/2609.18442) | 提出了RiskWorld风险感知世界建模框架，通过流引导占据演化实现共享占据预测，仅在额外预测风险触发干预且替代轨迹满足逐分量约束时才进行选择性轨迹替换，从而提升自动驾驶运动规划的安全性。 |
| [^274] | [Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts](https://arxiv.org/abs/2609.18366) | 提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。 |
| [^275] | [Are You Learning Biological Signal or Shortcuts? Auditing and Mitigating Bias in Protein-Protein Interaction Datasets](https://arxiv.org/abs/2609.10193) | 本文系统审计了蛋白质-蛋白质相互作用数据集中已报道和未报道的多种偏差（包括随机数据划分引入的拓扑捷径），这些偏差会使机器学习模型学到捷径而非真正的生物信号，并提出相应的缓解方法。 |
| [^276] | [TabBench-Bio: A Living Benchmark for Machine Learning on High-Dimensional Biomedical Tables](https://arxiv.org/abs/2609.07441) | TabBench-Bio是一个涵盖43个生物医学数据集的动态基准测试，系统评估了经典方法、神经网络和表格基础模型在高维小样本场景下的表现，发现表格基础模型总体领先，其中RealTabPFN v2.5表现最佳。 |
| [^277] | [OracleZoom: On-Policy Self-Distillation Inspired Reference-Constrained Recursive Image Super Resolution](https://arxiv.org/abs/2609.06490) | OracleZoom提出了一种受自蒸馏启发、参考约束的递归图像超分辨率框架，通过直接与跨尺度监督、无参考质量目标、KL约束潜在先验和EMA一致性等技术，解决了深层递归缺乏真实标签监督的问题，在七个数据集上实现了极端放大倍率下最先进的超分辨率质量。 |
| [^278] | [Modern Transformers Are Implicit Hybrids: From Functional Differentiation to Principled Hybrid Architecture Design](https://arxiv.org/abs/2609.02986) | 本文提出RFIS和RPD两个干预指标，发现现代Transformer的注意力头自然分化为由全局位置频带（GPBand）分隔的检索头和位置头两类，为有原则地设计FA-LA混合架构提供了实证基础。 |
| [^279] | [XMerge: Cross-Axis Selection and Reconstructive Layer Merging for LLM Depth Compression](https://arxiv.org/abs/2609.02083) | XMerge 是一种训练后的大语言模型深度压缩方法，通过跨轴选择识别隐藏状态变化最小的层块，并利用局部边界重构重新拟合相邻存留块，在不改变架构、不增加推理参数、无需任务标签的情况下实现高质量的层删除压缩。 |
| [^280] | [Ceiling-Clipped Acceptance Histograms Indicate Stranded Speed-up in Block-Diffusion Speculative Decoding](https://arxiv.org/abs/2608.30427) | 该论文发现块扩散投机解码中，高接受率会使草稿模型提前耗尽训练时的块长度上限，造成未被实现的“搁浅加速”，提出用接受直方图顶格柱的尖峰作为投入训练算力前的预检指标，并证明推理时简单扩大块大小无法恢复该加速。 |
| [^281] | [Actionable CBFI: Integrating Structural Decomposition and Causal Counterfactual Recourse for Tabular Machine Learning](https://arxiv.org/abs/2608.27821) | 该论文提出A-CBFI框架，基于结构因果模型将特征重要性分解与因果反事实补救相结合，通过隔离协同交互瓶颈并释放抑制性结构锁，为表格机器学习生成因果有效、可操作的针对性干预方案，克服了现有方法因果无效与干预分散的问题。 |
| [^282] | [MetaSieve: Faster Relational Deep Learning through SQL-Based Metapath Selection](https://arxiv.org/abs/2608.25903) | 本文提出MetaSieve，利用SQL连接和聚合能力，通过智能剪枝元路径来减小GNN训练子图大小，从而加速关系型深度学习，同时保持模型准确性。 |
| [^283] | [A Layered Simplex Architecture for Large Alphabets](https://arxiv.org/abs/2608.19908) | 本文提出一种极简的分层单纯形贝叶斯估计器，通过均匀相乘并归一化实现，无需调参，具有显式可计算的遗憾值，并在大字母表概率估计任务上与专门方法竞争，同时揭示缩放规律。 |
| [^284] | [Efficient Hessian-Free Methods for Multi-Objective Bilevel Optimization with Nonconvex Lower Level](https://arxiv.org/abs/2608.12704) | 本文提出了一种新的多目标Moreau包络无海森算法（MOMEHA），首次解决了非凸下层问题的多目标双层优化，通过单循环和无海森设计实现了高效计算。 |
| [^285] | [CommitKV: Commit-Aligned KV Cache Compression for Multimodal Agents](https://arxiv.org/abs/2608.07855) | CommitKV是一种无需训练的多模态智能体KV缓存压缩方法，通过对比工具调用前后对同一缓存页面的删除影响来捕捉证据使用的变化，淘汰两个时间窗口中影响力均低的页面并暂时保护被观察激活的页面，避免因注意力转移而误删后续需要的信息。 |
| [^286] | [Beyond Co-Movement: Locality by Exposures Enables a Joint Factor-Graph Framework for Portfolio Diversification](https://arxiv.org/abs/2608.06618) | 该论文提出MINGLE框架，用系统性因子暴露度而非观测共动性来定义图的局部性，并通过统一的ADMM框架联合学习潜在因子表示与图拓扑结构，从而在真实市场结构上实现更有效的投资组合分散化。 |
| [^287] | [Graph-of-Thoughts for LLM Causal Discovery with a Fairness-Pathway Case Study](https://arxiv.org/abs/2608.02877) | GoT-CD是一种基于思维图的LLM因果发现方法，它让LLM一次性生成多个完整候选图，并通过确定性评分排序、硬并集约束合并、细化与DAG投影，在多个基准上取得了优于现有LLM因果发现方法的F1分数和结构汉明距离。 |
| [^288] | [QAdapt: Noise-Adaptive Neural Pre-Decoding for Surface-Code Quantum Error Correction](https://arxiv.org/abs/2607.28422) | QAdapt提出了一种噪声自适应神经预解码框架，通过学习局部时空相关性并结合在线弹性权重巩固，在多种噪声机制间持续适应，显著降低了表面码量子纠错的逻辑错误率并提升了解码效率。 |
| [^289] | [Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.21653) | 提出了 Molt 框架，通过可组合模型并行、统一智能体接口、全异步 rollout 与优化以及分布式经验存储，实现了万亿参数规模下的智能体强化学习训练，且无需修改现有智能体的执行逻辑。 |
| [^290] | [Operational Proto-Introspection in Looped Language Models: Reading Correctness Before the Answer](https://arxiv.org/abs/2607.18553) | 该论文提出“操作性原初内省”这一概念，通过探针读取循环语言模型Ouro-RLTT在推理早期的隐状态，证明可以在答案给出之前部分预测推理结果的正确性（AUROC达0.758，远高于token置信度的0.515），并排除了问题难度等竞争性解释。 |
| [^291] | [What Does a Temporal Benchmark Score Measure? Decomposing Channel Use in Video VLM Evaluation](https://arxiv.org/abs/2607.12304) | 该论文提出“反转冲突”评测方法（帧序反转而RoPE保持正向），将时序视频VLM基准分数分解为“任务是否真正需要时序信息”与“模型是从像素还是位置编码获取时序”两个通道，从而区分真正理解时序的模型与仅仅被干扰的模型。 |
| [^292] | [Bringing Back Rule Induction to Fluid Intelligence Research? An Initial Validation of the ARC-AGI Benchmark in Humans](https://arxiv.org/abs/2607.11263) | 该研究首次在人类样本中验证了ARC-AGI基准的心理测量学特性，发现其与图形流体智力高度相关（ρ=.63），证明这一以规则归纳为核心的基准可作为流体智力的有效测量工具。 |
| [^293] | [Auditing Invisible Weight Updates with Reference Traces](https://arxiv.org/abs/2607.09800) | 本研究提出通过高精度参考轨迹对低精度权重更新进行审计，发现随机写回优于最近写回，且该方法在绝对精度上有效但在方向特异性上存在局限。 |
| [^294] | [Is Randomness Necessary for Adaptive Data Analysis?](https://arxiv.org/abs/2607.07085) | 本文首次系统研究了自适应数据分析中一个根本性开放问题——随机性是否必要，即探索确定性机制能否达到与随机化机制相当的查询支持能力。 |
| [^295] | [Uncertainty-aware damage identification in short-span bridges via physics-informed variational autoencoder](https://arxiv.org/abs/2607.05025) | 该论文提出将数值特征值求解器直接嵌入变分自编码器解码器并利用高斯copula刻画刚度因子后验分布的物理信息框架，实现了短跨桥梁振动损伤识别中可靠的不确定性量化与物理一致性保证。 |
| [^296] | [When Do Foundation Models Pay Off? A Break-Even Analysis of Pretrained Time Series Forecasters](https://arxiv.org/abs/2607.04919) | 该论文首次系统性地分析了部署时间序列基础模型何时能比经典方法更划算，通过30个数据集的对比实验给出了基于训练样本量和季节性强度的盈亏平衡经验法则（例如训练样本少于700且季节性显著时零样本基础模型即够用）。 |
| [^297] | [When Does Learning to Stop Help? Configuring Risk-Controlled Early Exit for Reasoning-Model Services](https://arxiv.org/abs/2606.30852) | 该论文提出一种基于工作负载校准数据的部署决策流程，通过Learn-then-Test以可证明的风险界限同时认证学习式停止规则与静态token预算两类提前退出方案并部署成本最低者，实验表明两类方案在不同模型与任务上各有优势、无普适胜者。 |
| [^298] | [FlexMoE: Learning One-for-All Multi-Budget Expert Pruning for Flexible MoE Deployment](https://arxiv.org/abs/2606.27866) | FlexMoE提出了一种训练后的一体化多预算专家剪枝框架，通过构建跨预算共享的可切片权重空间和学习专家级保留动作，将大型预训练MoE模型转换为可灵活部署的多预算子网络家族。 |
| [^299] | [SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference](https://arxiv.org/abs/2606.26587) | 提出SharQ方法，通过在线稀疏-稠密分解与残差补偿机制，有效结合了激活稀疏性与FP4量化，在无需训练的情况下降低了大语言模型推理中的激活压缩损失。 |
| [^300] | [Embedding Foundation Model Predictions in Discrete-Choice Models with Structural Guarantees](https://arxiv.org/abs/2606.26432) | 本文提出一种两阶段适配器方法，通过将基础模型预测嵌入多项Logit模型，在保留结构保证的同时修正经济逻辑冲突，使时间价值等指标成为数学保证。 |
| [^301] | [Autonomous Video Generation for Self-Evolving World Models](https://arxiv.org/abs/2606.24152) | 本文提出“自主视频生成”概念，以反事实可控性（生成干预条件下的未来并将其绑定到具身智能体）作为评估标准，使视频生成世界模型能够通过环境反馈实现自进化。 |
| [^302] | [TROPT: An Open Framework for Unifying and Advancing Discrete Text Optimization](https://arxiv.org/abs/2606.23496) | TROPT是首个开源框架，通过单一接口统一了离散文本优化器的执行并标准化其开发流程，解决了现有优化器分散、难以使用和难以对比的问题，降低了在红队测试、审计和可解释性等领域应用与推进离散文本优化的门槛。 |
| [^303] | [BatchGen: An Architecture for Scalable and Efficient Batch Inference](https://arxiv.org/abs/2606.21712) | 该论文提出了“序列协程计算模型”这一批量推理新架构基础，将每个序列表示为细粒度的事件驱动协程，使运行时能够动态重组工作负载，从而在大规模GPU集群上实现更高的吞吐量和利用率。 |
| [^304] | [Learning through Internalization](https://arxiv.org/abs/2606.20937) | 本文首次对神经网络的内化过程给出可证明分析：在奇偶性学习任务上，单层Transformer可先借助显式思维链（CoT）监督完成学习，再随着CoT标记逐步移除而将自回归计算内化进自身权重、实现直接计算，而该任务在无CoT监督时从数据中学习是计算上困难的，同时内化也会带来分布外性能的退化。 |
| [^305] | [Pose6DAug: Physically Plausible Multi-view Object Swapping for Robot Data Augmentation](https://arxiv.org/abs/2606.20118) | 提出了Pose6DAug框架，通过在保留物理有效动作轨迹与多视角标定观测的前提下替换被操作物体，将已有的成功演示转化为针对策略失败模式的物理合理增强数据，无需新的数据采集即可提升VLA策略对分布外物体的泛化能力。 |
| [^306] | [A Differentiable Composite Approximation Framework for Autonomous Underwater Vehicle Maneuvering Modeling from Sea-Trial Data](https://arxiv.org/abs/2606.19711) | 本文提出一种可微复合近似框架，将多项式基与数据自适应神经基作为统一预测器的可微分量进行梯度联合校准，从而从海试数据中建立更准确的AUV机动模型。 |
| [^307] | [DLWM: Diverse Latent World Models for Efficient Multimodal Reasoning](https://arxiv.org/abs/2606.15160) | DLWM提出了一种将潜空间推理与强化学习相结合的多模态推理框架，通过构建多样化的潜在世界模型来应对视觉输入的多种合理解释问题，摆脱单一解释和固定推理路径的限制，从而实现更高效的多模态推理。 |
| [^308] | [Rational Sparse Autoencoder](https://arxiv.org/abs/2606.14990) | 提出有理稀疏自编码器（RSAE），用可训练的有理函数取代固定的编码器激活函数，在匹配期望稀疏度下从理论上保证不劣于 ReLU，并在给定假设下可证明优于 JumpReLU 和 TopK。 |
| [^309] | [FedRAN: Accurate and Efficient Analytic Federated Continual Learning](https://arxiv.org/abs/2606.11480) | FedRAN提出了一种低秩分析式联邦持续学习框架，通过跨客户端与任务合并的谱摘要直接压缩二阶结构，并利用保留的秩显式控制近似精度与通信开销之间的权衡，实现准确且高效的联邦持续学习。 |
| [^310] | [Dirichlet-Guided Group Forecasting for Alleviating Over-smoothing in Time Series Forecasting](https://arxiv.org/abs/2606.10592) | 提出迪利克雷引导的群体预测（DGF）框架，通过显式建模多个模式条件下的预测分布及其选择概率的不确定性，并借助迪利克雷引导的分层采样与基于奖励的优化，有效缓解时间序列预测中的过度平滑问题。 |
| [^311] | [Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models](https://arxiv.org/abs/2606.04326) | 本文提出了概念瓶颈模型的合成基准，使研究人员能够控制数据模态、概念选择和标注质量等因素，从而系统地评估该类模型在决策支持与自动化场景中的表现，并识别出新的失败模式。 |
| [^312] | [CaMBRAIN: Real-time, Continuous EEG Inference with Causal State Space Models](https://arxiv.org/abs/2605.28792) | 提出了首个基于Mamba的因果状态空间模型CaMBRAIN，实现了脑电信号的实时连续推理，克服了注意力机制的二次方复杂度和滑动窗口处理无法全局理解信号的局限。 |
| [^313] | [Extracting Training Data from Diffusion Language Models via Infilling](https://arxiv.org/abs/2605.24173) | 该论文提出“填充提取”协议，利用任意二进制掩码全面评估扩散语言模型的训练数据泄露风险，发现传统的前缀探测方法会显著低估记忆化风险，而边缘条件掩码可提取多达三倍的逐字训练序列。 |
| [^314] | [GRTO: Bootstrapped Group Relative Tool Optimization for Referring Segmentation](https://arxiv.org/abs/2605.23500) | 提出GRTO，一个基于数学原理的两阶段框架，首次将推理策略与可微分割工具联合优化，解决现有方法只优化VLM、在域偏移下分割错误损害正确推理的问题。 |
| [^315] | [Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization](https://arxiv.org/abs/2605.19782) | 该研究通过三个受控实验揭示，LLM智能体在硬件感知代码优化中依赖先验知识而非搜索——它在黑盒优化中表现为贪婪优化器，在零样本内核生成中会完全忽略输入尺寸信息，且面对不常见的内核尺寸时性能急剧下降。 |
| [^316] | [The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM Reproducibility](https://arxiv.org/abs/2605.19537) | 该论文通过调研200个推理引擎和分析35,000篇ML论文，首次系统量化了推理后端这一常被忽视的“沉默超参数”对大语言模型基准测试结果与可复现性的影响。 |
| [^317] | [KVBuffer: IO-aware Serving for Linear Attention](https://arxiv.org/abs/2605.19049) | KVBuffer是一种面向线性注意力的IO感知服务机制，通过缓冲最近的键和值实现分块计算与批量状态更新，显著减少内存访问量并降低解码延迟，同时高效支持投机解码验证。 |
| [^318] | [Constrained latent state modeling: A unifying perspective on representation learning under competing constraints](https://arxiv.org/abs/2605.15995) | 该论文提出约束潜在状态建模（CLSM）这一统一概念框架，用预测充分性、最小性、时间一致性、观测兼容性、抗干扰不变性和结构约束六个互补特性来刻画潜在表示，从而统一解释和比较各类表示学习方法，并阐明约束组合如何提升表示的可辨识性。 |
| [^319] | [From Data to Action: Accelerating Refinery Optimization with AI](https://arxiv.org/abs/2605.15085) | 本文通过为线性规划求解器增加记忆功能并引入异常检测工具，能够发现炼油厂优化中输入数据的潜在错误及不同决策层面的异常关系，从而提升后续LP求解结果的可信度和质量。 |
| [^320] | [REALM: Retrospective Encoder Alignment for LFP Modeling](https://arxiv.org/abs/2605.14867) | 本文提出REALM框架，通过回溯知识蒸馏将非因果模型的表达知识迁移到因果模型中，从而实现高精度且可实时部署的LFP行为解码。 |
| [^321] | [Learning to Decide with AI Assistance under Human-Alignment](https://arxiv.org/abs/2605.12646) | 本文揭示了AI置信度与人类对自身预测的置信度之间的对齐程度如何影响AI辅助决策中通过反复交互学习最优决策的复杂度，并将该学习问题归约为双臂老虎机问题。 |
| [^322] | [MIST: Reliable Streaming Decision Trees for Online Class-Incremental Learning via McDiarmid Bound](https://arxiv.org/abs/2605.11617) | 该论文提出MIST框架，利用McDiarmid界构造与类别数量无关的紧凑分裂准则，并解决分裂时的知识迁移缺失问题，从而使流式决策树能够可靠地应用于在线类别增量持续学习。 |
| [^323] | [Minimum Neural Weight Norm = Kolmogorov Complexity](https://arxiv.org/abs/2605.10878) | 该论文证明了循环神经网络计算某函数所需的最小权重范数与该函数的柯尔莫哥洛夫复杂度除以其对数渐近等价，从而在权重范数最小化与算法简单性之间建立了理论联系，为解释神经网络的泛化能力提供了新途径。 |
| [^324] | [Signature Approach for Contextual Bandits with Nonlinear and Path-dependent Rewards: Regret Bound and Statistical Inference](https://arxiv.org/abs/2605.10313) | 该论文提出基于签名变换的 DisSigUCB 算法，将路径依赖的非线性奖励泛函近似为签名空间中的线性泛函，从而高效求解非线性和路径依赖奖励的上下文老虎机问题，并建立了高概率遗憾界。 |
| [^325] | [Inverse Design for Conditional Distribution Matching](https://arxiv.org/abs/2605.09439) | 该论文提出了条件分布匹配（CDM）这一新的逆向设计问题类别，即寻找使生成模型的条件分布匹配给定目标分布的输入，并定义了CDMS与CDMO两个问题变体及相应的求解方法MLGD-F。 |
| [^326] | [Sparse Layers are Critical to Scaling Looped Language Models](https://arxiv.org/abs/2605.09165) | 研究发现稀疏混合专家（MoE）层是扩展循环语言模型的关键，因为不同循环通过激活不同专家可在不增加参数的情况下恢复表达能力，同时循环边界还提供了比标准模型更优的提前退出点。 |
| [^327] | [Supersampling Stable Diffusion and Beyond: Training-Free Neural Network Scaling via Signal Interpolation](https://arxiv.org/abs/2605.08698) | 提出了一种免训练的卷积核信号插值方法，使Stable Diffusion 1.5无需额外训练即可生成4倍于训练分辨率的图像，同时避免了分块扩散的对象重复问题和核膨胀的微调困难。 |
| [^328] | [StreamPhy: Streaming Inference of High-Dimensional Physical Dynamics via State Space Models](https://arxiv.org/abs/2605.07384) | StreamPhy是一个端到端流式推断框架，通过数据自适应观测编码器、支持内存高效在线更新的结构化状态空间模型和FT-FiLM连续场解码器，实现了从非规则稀疏测量中实时高效准确地推断高维多模态物理场的演化。 |
| [^329] | [TraXion: Rethinking Pre-training Frameworks for Mobility and Beyond](https://arxiv.org/abs/2605.06906) | 该论文提出TraXion预训练框架，将移动数据所具有的元组值事件、用户持久签名和跨用户关联三个结构特性形式化为多实体时空事件流（MESES）应满足的公理，从而克服了将轨迹当作句子、访问当作词元的语言建模式预训练方法的局限。 |
| [^330] | [Observable Performance Does Not Fully Reflect Adaptive System Organization: A Multi-Level Analysis of Gait Dynamics Under Occlusal Constraint](https://arxiv.org/abs/2605.00778) | 该研究通过对一名帕金森病患者在六种咬合约束下步态动力学的多层次分析，证明可观测的运动表现并不能完全反映适应性神经力学系统的内部组织结构，相似的表观输出可能源于不同的适应性配置。 |
| [^331] | [Distinct mechanisms underlying in-context learning in transformers](https://arxiv.org/abs/2604.12151) | 该论文通过对有限离散马尔可夫链集合上训练的Transformer进行机制分析，揭示其上下文学习呈现四个算法阶段（记忆或泛化、单点或双点统计），由两种性质截然不同的多层子电路机制实现，并由两条依赖数据多样性的边界划分记忆与泛化阶段。 |
| [^332] | [Reading Human Preference from Looped Transformer Internal States](https://arxiv.org/abs/2604.09870) | 该研究发现，通过在循环Transformer模型的内部状态上训练小型探针，可以以约61%-65%的微弱优势识别人类偏好的回复，且证据表明偏好信息更可能以单个回复的属性而非两回复之间的比较形式编码于模型内部。 |
| [^333] | [Training-Free Dynamic Upcycling of Expert Language Models](https://arxiv.org/abs/2603.29765) | 本文提出 DUME，一种免训练的新方法，通过复用不同领域训练好的稠密专家模型来动态构建统一的 MoE 模型，从而避免多任务微调带来的过度专业化和灾难性遗忘问题。 |
| [^334] | [Optimistic Actor-Critic with Parametric Policies for Linear Markov Decision Processes](https://arxiv.org/abs/2603.28595) | 该论文针对有限时域线性马尔可夫决策过程提出了使用参数化对数线性策略的乐观演员-评论家框架，通过引入可处理的logit匹配回归目标和基于朗之万动力学的近似汤普森采样，克服了现有演员-评论家方法在探索假设和隐式策略采样效率方面的理论局限。 |
| [^335] | [The multiplicative Lipschitz bound as an inductive bias](https://arxiv.org/abs/2603.28113) | 该论文提出将看似宽松的乘性Lipschitz“平凡界”与新型求积非线性激活函数相结合，在Lipschitz约束学习中超越了Sandwich、凸势层等最先进方法，并在非扩张自编码器上实现了比PCA低12-24%的重建误差。 |
| [^336] | [Shapley meets Rawls: an integrated framework for measuring and explaining unfairness](https://arxiv.org/abs/2603.26476) | 本文提出利用Shapley值及其扩展的ESL值家族，将群体不公平性的定义、度量和解释整合到统一框架中，并在Census Income数据集上识别出年龄、每周工作小时数和婚姻状况是造成性别不公平的关键因素，同时计算效率优于传统Bootstrap检验。 |
| [^337] | [Improving LLM Final Representations with Inter-Layer Geometry](https://arxiv.org/abs/2603.22665) | 本文提出用轻量级图神经网络聚合大语言模型各层信息，并进一步设计基于SL(2,Zn)凯莱图的Cayley编码器，以稀疏、规则且低直径的数学拓扑替代昂贵的注意力机制，在提升效率的同时显著增强下游预测性能。 |
| [^338] | [HYPER: Hypothesis-Guided Planning for Evidence-Seeking Retrieval](https://arxiv.org/abs/2603.19008) | HYPER是一个无需训练的检索前规划框架，以临时答案假设为锚点，将检索分解为支持、区分和特征三种证据角色，把RAG从面向主题的检索转变为面向证据的检索，从而在需要竞争性选择的任务中更好地支持最终决策。 |
| [^339] | [AERR-Nav: Adaptive Exploration-Recovery-Reminiscing Strategy for Zero-Shot Object Navigation](https://arxiv.org/abs/2603.17712) | AERR-Nav提出了一种自适应探索-恢复-回忆策略，通过根据环境动态调整机器人状态，解决了未知多层环境中零样本目标导航的探索与利用平衡难题。 |
| [^340] | [Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency](https://arxiv.org/abs/2603.17102) | 该论文提出将大语言模型的跨语言不一致性作为可解释性工具，通过对比模型在不同语言中事实回忆成功与失败时的MoE路由行为，定位混合专家模型中负责特定知识的关键专家。 |
| [^341] | [Auditing Path-Specific Fairness Across Causal Discovery Algorithms: A Heart Failure Case Study](https://arxiv.org/abs/2603.15926) | 本研究通过为心力衰竭临床记录数据集建立专家引导的因果图基准，首次在临床数据上系统评估了不同因果发现算法的选择如何通过路径特定公平性分解传导并影响公平性与效用的权衡。 |
| [^342] | [SHAMISA: SHAped Modeling of Implicit Structural Associations for Self-supervised No-Reference Image Quality Assessment](https://arxiv.org/abs/2603.13669) | 该论文提出SHAMISA，一种非对比自监督无参考图像质量评估框架，其核心创新是组合式失真引擎——从连续参数空间生成每次仅单一失真因素变化的图像分组，从而在训练中实现对表征相似性的细粒度塑形控制。 |
| [^343] | [Imitation Learning from Human Motion Alone Does Not Guarantee Biomechanically Plausible Gait Kinetics](https://arxiv.org/abs/2603.12408) | 研究发现仅依靠运动数据的模仿学习虽能准确跟踪运动学，却无法估计出合理的关节力矩，而将地面反作用力和压力中心等动力学信息纳入模仿奖励的KAIL框架能产生更符合生物力学的步态动力学。 |
| [^344] | [Counterfactual Feature Regularization for Expansion-Based Class-Incremental Learning](https://arxiv.org/abs/2603.09145) | 本文从因果视角指出虚假特征相关性是新旧特征冲突的根源，提出基于必要性与充分性概率（PNS）的反事实特征正则化方法，以消除任务内和任务间的虚假相关性，指导类增量学习中的特征扩展。 |
| [^345] | [How Far Does Online RL Generalize in Mobile GUI Agents? A Controlled Study with AndroidWorld-Generalization](https://arxiv.org/abs/2603.07432) | 本文提出AndroidWorld-Generalization基准，通过未见实例、未见模板、未见应用三级递进的上下文偏移系统评估在线RL在移动GUI智能体中的泛化能力，发现RL收益随环境偏移距离增大而单调衰减。 |
| [^346] | [Unlocking Continuous Diffusion Language Models via Discrete Stochastic Localization](https://arxiv.org/abs/2602.16169) | 提出离散随机局部化（DSL）连续状态框架，通过将随机局部化通道与单位球面词元嵌入相结合，实现了仅依赖含噪状态、无需时间步条件的贝叶斯最优去噪器，使单个去噪网络即可支持连续去噪、掩码扩散、随机顺序自回归等多种采样方法，从而让连续扩散模型在离散序列生成上有效匹敌掩码离散扩散模型。 |
| [^347] | [MUSE: Multi-Tenant Model Serving With Seamless Model Updates](https://arxiv.org/abs/2602.11776) | MUSE是一个多租户模型服务框架，通过将模型分数与客户端决策边界解耦，使模型更新时无需跨数百个客户重新协调校准阈值，从而实现无缝的模型更新。 |
| [^348] | [Signature-Kernel Based Evaluation Metrics for Robust Probabilistic and Tail-Event Forecasting](https://arxiv.org/abs/2602.10182) | 本文提出了一族基于签名核的删失评估指标，通过将分布主体收缩为单一枢轴点来聚焦尾部区域和高效用区域的评估，从而更准确地衡量概率预测在分布尾部的保真度。 |
| [^349] | [Multimodal Information Fusion for Chart Understanding: A Survey of MLLMs -- Evolution, Limitations, and Cognitive Enhancement](https://arxiv.org/abs/2602.10138) | 本综述系统梳理了基于多模态大语言模型（MLLM）的图表理解研究，分析了视觉与语言信息融合的核心挑战，提出了规范型与非规范型基准的新颖分类体系，并追溯了从经典深度学习技术到先进MLLM融合范式的方法演进历程。 |
| [^350] | [Towards Generalizable Reasoning: Group Causal Counterfactual Policy Optimization for LLM Reasoning](https://arxiv.org/abs/2602.06475) | 该论文从因果视角将多候选推理视为一族反事实实验，提出群体因果反事实策略优化，通过同时评估推理稳健性与有效性的反事实奖励机制，训练大语言模型超越仅以答案正确性为标准的学习方式，从而掌握可泛化的推理模式。 |
| [^351] | [Pragmatic Curiosity: A Unified Framework for Hybrid Learning and Optimization via Active Inference](https://arxiv.org/abs/2602.06104) | 该论文提出了实用好奇心（PraC）框架，通过主动推理将贝叶斯优化与贝叶斯实验设计统一起来，以信息增益与预期遗憾之间的权衡来指导混合学习与优化场景中的序贯决策。 |
| [^352] | [CoreQ: Mismatch-Corrected Successive Rounding for LLM Quantization](https://arxiv.org/abs/2602.05902) | CoreQ是一种免学习的大语言模型后训练量化框架，通过自适应调节逐层失配校正的强度，避免了完全校正导致的校准数据过拟合以及固定校正系数忽略层间差异的问题。 |
| [^353] | [Targeted Synthetic Control Method](https://arxiv.org/abs/2602.04611) | 本文提出目标化合成控制（TSC）方法，一种新型两阶段估计器，通过目标化去偏更新直接估计反事实结果，在保证权重稳定性和凸组合可解释性的同时，可灵活结合任意机器学习模型。 |
| [^354] | [Low-Dimensional Adaptation of Rectified Flow: A Diffusion and Stochastic Localization Perspective](https://arxiv.org/abs/2601.15500) | 本文证明整流流能自动适应目标分布的内在低维结构，在端点自适应时间离散化方案下达到 O(k/ε) 的采样迭代复杂度，并建立了DDPM与随机定位之间的新颖联系，证明DDPM等价于整流流的随机化版本。 |
| [^355] | [Meta-RL with Bayesian Linear Task Models](https://arxiv.org/abs/2512.20974) | 提出GLiBRL深度贝叶斯强化学习框架，通过结合广义线性任务模型与可学习非线性基函数，实现精确的共轭贝叶斯后验更新和闭式边缘似然，消除了变分推断带来的近似误差，并可灵活兼容离策略与在线策略算法。 |
| [^356] | [TinyMyo: a Tiny Foundation Model for Flexible EMG Signal Processing at the Edge](https://arxiv.org/abs/2512.15729) | TinyMyo是一个仅360万参数的微型肌电基础模型，通过自监督掩码重建预训练支持多种下游任务，手势分类性能超越现有肌电基础模型，并可在超低功耗微控制器上实现边缘端部署。 |
| [^357] | [Integrated Prediction and Multi-period Portfolio Optimization](https://arxiv.org/abs/2512.11273) | 本文提出IPMO模型，将收益预测与多期均值-方差投资组合优化整合为端到端的单一框架，解决了传统两阶段方法中预测与决策脱节以及忽略交易成本的问题。 |
| [^358] | [When Features Beat Noise: A Noise-Driven Hypothesis Testing Method for Feature Selection](https://arxiv.org/abs/2511.20851) | 本文提出一种噪声驱动的特征选择方法，通过将每个特征的重要性与随机噪声特征中的最大重要性进行基于非参数bootstrap的假设检验，为特征选择提供了坚实的统计理论基础和显著性评估准则。 |
| [^359] | [Consecutive Preferential Bayesian Optimization](https://arxiv.org/abs/2511.05163) | 该论文提出连续偏好贝叶斯优化（CPBO），通过在每次比较中复用上一个候选方案来降低候选方案的生产成本，并在概率偏好模型中引入最小可觉差（JND）阈值以处理评估者的感知模糊性，同时适配信息论采集策略来高效逼近未知最优解。 |
| [^360] | [Assessing the robustness of heterogeneous treatment effects in survival analysis under informative censoring](https://arxiv.org/abs/2510.13397) | 该论文提出一个少假设框架及新颖的模型无关元学习器SurvB-learner，通过部分识别推导条件平均处理效应的边界，从而在信息性删失偏差下评估生存分析中异质性治疗效应估计的稳健性，并识别治疗有效的患者亚群。 |
| [^361] | [Controllable Generative Trajectory Prediction via Weak Preference Alignment](https://arxiv.org/abs/2510.10731) | 提出PrefCVAE增强型CVAE框架，利用弱标注偏好对为潜变量赋予语义属性，从而在不损失预测准确性的前提下实现可控、语义有意义的轨迹预测。 |
| [^362] | [Hyperparameter Optimization via Budget Allocation in Lipschitz Spaces](https://arxiv.org/abs/2510.10605) | 该论文将超参数优化形式化为“未知值探测”决策问题，提出了在Lipschitz空间结构下结合探索与价值导向集中的预算分配策略，证明了接近最优的最坏情况近似保证，并在250多个实验设置中取得了一致的性能提升。 |
| [^363] | [Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models](https://arxiv.org/abs/2505.16959) | 过参数化扩散模型在训练中先于记忆化发生之前逐步实现泛化，记忆化时间与数据集大小成正比，因此泛化与记忆化本质上是时间尺度的竞争，可通过早停来避免记忆化。 |
| [^364] | [Reinforcement Learning with Reference-Based Rewards for Language Model Reasoning and Alignment](https://arxiv.org/abs/2504.09895) | 提出RL2R方法，利用参考答案与轻量级语义相似度指标（如BERTScore）作为奖励信号，为语言模型的推理与对齐提供统一且低成本的强化学习后训练方案。 |
| [^365] | [Survival of the Fittest Perturbations: Uncertainty-Aware Evolution of Adversarial Distributions for Robust Generalization](https://arxiv.org/abs/2411.02871) | 提出了进化式不确定性感知分布对抗训练（EvoUAD-AT），通过在对抗锚点周围维护实例级的对抗特征统计搜索分布并进行进化式威胁选择，以缓解单点对抗训练忽略对抗变异性的问题，从而提升模型的鲁棒泛化能力。 |
| [^366] | [Mixtures of In-Context Learners](https://arxiv.org/abs/2411.02830) | 提出MoICL方法，将上下文学习的演示示例子集视为专家并学习加权函数来合并其输出分布，在提升分类性能的同时减少推理时间，并显著增强了对域外、不平衡和噪声示例的鲁棒性。 |
| [^367] | [Don't Pay Attention, PLANT It: Pretraining Attention via Learning-to-Rank](https://arxiv.org/abs/2410.23066) | 提出PLANT，一种即插即用策略，通过基于互信息增益的学习排序模型预训练标签注意力来初始化注意力模块，可与大语言模型骨干集成，在极端多标签文本分类任务上超越现有最先进方法，且在罕见标签上收益最大。 |
| [^368] | [Design Optimization of Nuclear Fusion Reactor through Deep Reinforcement Learning](https://arxiv.org/abs/2409.08231) | 该研究将深度强化学习中的奖励约束策略优化（RCPO）应用于托卡马克聚变反应堆设计，在满足四个稳定运行约束的条件下，实现了对建造成本、能量增益和能量约束时间的多目标黑箱优化。 |
| [^369] | [Co-Pilot for Health: Personalized Algorithmic AI Nudging to Improve Health Outcomes.](http://arxiv.org/abs/2401.10816) | 该研究通过使用基于图神经网络的推荐系统和来自可穿戴设备的健康行为数据，设计并实施了一个人工智能驱动平台，实现了个性化和情境引导，能够提高参与者的日常活动水平和中等至剧烈运动时长。 |
| [^370] | [Loss-Controlling Calibration for Predictive Models.](http://arxiv.org/abs/2301.04378) | 本文提出了一种学习框架，用于对可交换数据进行损失控制预测的校准预测模型。通过引入保持交换性质的变换，我们证明了在获得测试标签时有有限样本控制保证，并发展了一种近似方法来构建预测器。这种方法是符合损失控制预测的自然扩展。 |

# 详细

[^1]: 知道你所知道的：人机认知系统中审阅者价值的特质理论

    Knowing What You Know: A Trait Theory of Reviewer Value in Human-AI Cognitive Systems

    [https://arxiv.org/abs/2611.24932](https://arxiv.org/abs/2611.24932)

    该论文提出“元认知校准”（审阅者信心与其判断实际正确性的一致性）是决定人类审阅者在人机协作认知系统中核心价值的唯一关键特质，并主张联合系统应将每个决策条目的最终裁定权交给校准程度更高的智能体——无论是人还是模型。

    

    当一个能力强大的模型与一名人类审阅者在同一决策上协作时，什么样的 人类仍然能贡献价值？我们认为这是一个关于单一认知特质的问题——元认知校准：即审阅者的行为信心与其判断的实际正确性之间的一致程度。我们提出一套审阅者价值的特质理论来使其精确化。该理论提出五个论断：价值由校准程度决定，而非由活动量、速度、一致率或资历决定；校准会在行为遥测数据中留下可观测的痕迹；这些痕迹本身并不具有诊断性，因为经过良好校准的专家所呈现的快速、少修改的行为特征，在未经过结果门控之前与自信的新手难以区分；人类的校准与模型的自我认知是同一个构念，因此人机联合系统应在每个条目上服从校准程度更高的那个智能体；个体校准仅能在……（原文在此处截断）

    arXiv:2611.24932v1 Announce Type: new  Abstract: When a capable model and a human reviewer cooperate on the same decision, which human still adds value? We argue this is a question about a single cognitive trait, metacognitive calibration: the alignment between a reviewer's behavioral confidence and the actual correctness of their judgments. We develop a trait theory of reviewer value to make it precise. The theory makes five claims: value is governed by calibration rather than by activity, speed, agreement, or tenure; calibration leaves observable traces in behavioral telemetry; those traces are non-diagnostic on their own, because the fast, low-revision signature of a calibrated expert is indistinguishable from that of a confident novice until it is gated by outcomes; human calibration and a model's self-knowledge are the same construct, so the joint system should defer to whichever agent is better calibrated on each item; and individual calibration is causally identifiable only abov
    
[^2]: 分解神经网络中的SGD动力学：教师诱导的尖峰与方差膨胀

    Decomposing SGD Dynamics in Neural Networks: Teacher-Induced Spikes and Variance Inflation

    [https://arxiv.org/abs/2611.24908](https://arxiv.org/abs/2611.24908)

    该论文通过将第一层权重分解为信号分量与正交补空间，揭示了SGD训练会在正交补空间的Gram矩阵中诱导出依赖教师的谱尖峰，从而放大表示方差并导致泛化性能退化的具体机制。

    

    我们研究了在多指数教师模型上通过随机梯度下降（SGD）训练的两层神经网络，重点关注与教师子空间正交的第一层参数如何影响泛化性能。我们将第一层权重分解为与教师对齐的信号分量和正交补空间分量，并在比例极限下分析它们与第二层的联合SGD动力学。我们证明，尽管SGD对正交补空间的更新最初是各向同性的，但训练动力学会在补空间的Gram矩阵中诱导出依赖教师的谱尖峰。这种涌现的各向异性提供了一种具体机制，通过该机制表示方差被放大，泛化性能随之退化。为了隔离这一效应，我们引入了“分解动力学”，这是一个解析上可处理的替代模型，其渐近测试误差动力学与SGD完全一致。在使用ReLU学生网络的模拟中，冻结正交补空间的相关方向……

    arXiv:2611.24908v1 Announce Type: cross  Abstract: We study two-layer neural networks trained by stochastic gradient descent (SGD) on a multi-index teacher model, focusing on how first-layer parameters orthogonal to the teacher subspace affect generalization. We decompose the first-layer weights into a signal component aligned with the teacher and an orthogonal complement, and analyze their joint SGD dynamics together with the second layer in the proportional limit. We show that although SGD updates to the complement are initially isotropic, the training dynamics induce teacher-dependent spectral spikes in the complement Gram matrix. This emergent anisotropy provides a concrete mechanism by which representation variance is amplified, and generalization degrades. To isolate this effect, we introduce Decomposed Dynamics, an analytically tractable surrogate whose asymptotic test-error dynamics coincide with those of SGD. In simulations with a ReLU student, freezing the directions of the o
    
[^3]: 基于最小齐次生成元的多项式特征等变建模

    Equivariant Modeling with Polynomial Features via Minimal Homogeneous Generators

    [https://arxiv.org/abs/2611.24896](https://arxiv.org/abs/2611.24896)

    该论文提出了一个基于最小齐次生成元构造等变多项式特征的通用等变函数建模框架，并提供了可自动计算特征与拟合模型的Python工具包，同时给出了更高效的特征计算算法。

    

    我们提出了一个用于建模和学习有限维半单实群表示之间等变函数的通用框架，该框架统一了众多应用领域中发展出的各种构造方法。我们的模型是等变多项式特征的线性组合，其系数是不变多项式特征的任意可学习函数。我们选取不变代数以及不变代数上协变模的最小齐次生成元作为特征，直至给定的阶数。我们提供了一个Python包，可自动计算这些特征并将所得模型拟合到训练数据。特征计算算法需要输入和输出表示的显式同型分解，以及不可约表示张量积的显式Clebsch-Gordan分解，并在计算代数方面相比其他方法实现了加速。数值实验（摘要在此处被截断）

    arXiv:2611.24896v1 Announce Type: new  Abstract: We present a general framework for modeling and learning equivariant functions between finite dimensional semisimple real group representations, unifying constructions developed across many application areas. Our model is a linear combination of equivariant polynomial features whose coefficients are arbitrary learnable functions of invariant polynomial features. We take as features minimal homogeneous generators for the invariant algebra and the covariant module over the invariant algebra, up to prescribed degrees. We provide a Python package that automatically computes these features and fits the resulting models to training data. The algorithm for feature computation requires explicit isotypic decompositions of the input and output representations, along with explicit Clebsch-Gordan decompositions of the tensor product of irreducible representations, and achieves speedups over alternative methods in computational algebra. Numerical exp
    
[^4]: 论交互式智能体用户模拟器的评估

    On Evaluating User Simulators for Interactive Agents

    [https://arxiv.org/abs/2611.24892](https://arxiv.org/abs/2611.24892)

    该论文对用户模拟器的评估指标本身进行元评估，发现现有方法无法可靠检测模拟用户行为中的多数故障，并据此提出了一种新的评估方法。

    

    用户模拟器为评估交互式智能体提供了一种可扩展的方式，但如何评估这些模拟器本身的质量仍缺乏充分研究。特别是，现有的用户模拟器评估框架能否可靠地检测模拟用户行为中的重大故障尚不清楚，这种不确定性使得模拟器评估结果难以解读。为了理解基准测试的高分究竟能在多大程度上反映模拟器的质量，我们对评估指标本身进行了（元）评估。为具体实现这一元评估，我们故意对模拟回放过程进行扰动以诱发多种故障，例如目标被篡改、约束条件被遗漏、过早停止以及接受不合格的工作成果，然后检验各项指标能否对这些故障进行惩罚。结果显示，来自三项先前研究的评估方法仅能检测出10种扰动中的5至6种；而且这些方法的综合评分甚至更青睐那些被指示接受不合格工作成果的模拟器群体，而非未经修改的对照组。基于这些缺陷，我们提出了一种新的评估方法。

    arXiv:2611.24892v1 Announce Type: new  Abstract: User simulators offer a scalable way to evaluate interactive agents, but how to assess their quality remains underexplored. In particular, it is unclear whether existing user simulator evaluation frameworks reliably detect meaningful failures in simulated user behavior. This uncertainty makes it difficult to interpret simulator evaluation. To understand what a high benchmark score tells us about simulator quality, we (meta-)evaluate the metrics themselves. To instantiate this meta-evaluation, we deliberately perturb simulated replays to induce failures such as altered goals, omitted constraints, premature stopping, and acceptance of broken work, then test whether the metrics penalize these failures. Evaluation methods from three prior studies detect only 5-6 of the 10 perturbations; their combined scores even favor populations instructed to accept broken work over unmodified controls. Motivated by these gaps, we propose a new evaluation 
    
[^5]: Cap-and-Track：在动态变化的容量下优化LLM推理请求分发

    Cap-and-Track: Optimizing LLM Inference Request Dispatch Across Shifting Capacity

    [https://arxiv.org/abs/2611.24891](https://arxiv.org/abs/2611.24891)

    Cap-and-Track利用饱和资源处的快速失败拒绝信号驱动分发器自动向上探测和退避，无需预测输出长度或手动调整集群级速率即可跟随动态变化的容量上限，并在生产A/B测试中以少43%的H100 GPU服务了相同流量。

    

    对于LLM推理而言，固定的分发速率会随着服务集群规模调整和请求规模的变化而过时。生成式输出的长度在请求准入时是未知的，且共享的提供商资源池可能会失去容量。我们提出Cap-and-Track，一种跨提供商容量和内部GPU的异步推理设计。Meta的流量整形器为帖子和直播视频理解实现了Cap-and-Track。拒绝信号表示后端当前的准入上限。持久化优先队列在执行之外保存多余的工作，包括被拒绝以待重试的请求。Cap是在饱和资源处的一种快速失败准入检查，能在长时间等待之前拒绝工作。Track分发器仅利用这些拒绝信号进行向上探测和退避。它无需输出长度预测，也无需集群级的速率重新调整，即可跟随容量上限。在并发的同流量生产A/B测试中，该整形器运行每副本的Cap加Track，以比基准少43%的H100 GPU服务了一个分类器（原文在此处截断）。

    arXiv:2611.24891v1 Announce Type: new  Abstract: A fixed dispatch rate for LLM inference becomes stale as serving fleets resize and request sizes change. Generative output length is unknown at admission, and shared provider pools can lose capacity. We present Cap-and-Track, a design for asynchronous inference across provider capacity and internal GPUs. Meta's traffic shaper implements Cap-and-Track for post and live-video understanding. Refusals signal the backend's current admission ceiling. Durable priority queues hold excess work outside execution, including refused requests for retry. Cap is a fail-fast admission check at the saturated resource that refuses work before a long wait. Using only those refusals, the Track dispatcher probes upward and backs off. It follows the ceiling without output-length prediction or a fleet-level rate to retune. In a concurrent same-traffic production A/B, the shaper ran per-replica Cap plus Track. It served a classifier on 43% fewer H100s than its 
    
[^6]: 单一文化鲁棒学习

    Monoculture Robust Learning

    [https://arxiv.org/abs/2611.24881](https://arxiv.org/abs/2611.24881)

    该论文在PAC学习中提出了“单一文化鲁棒性”概念，并揭示了可复现性与单一文化鲁棒性之间的惊人联系，使仅拥有单一数据集的学习器能输出 $k$ 个联合失败概率可与 $k$ 个独立数据集训练相媲美的准确假设，从而缓解算法单一文化问题。

    

    我们在概率近似正确（PAC）学习中引入了“单一文化鲁棒性”的概念，以此作为缓解算法单一文化问题（Kleinberg和Raghavan，2021）的一种途径：即共享相同组件（如训练数据）的模型往往在相同的输入上失败。具体而言，我们探讨了一个仅访问单一数据集的学习器能否输出 $k$ 个各自准确的假设，且这些假设的联合失败概率可与在 $k$ 个独立数据集上训练相媲美。我们将这样的学习过程称为实现了单一文化鲁棒性，并通过约束联合误差来量化这一性质，即在个体意义上所有 $k$ 个假设都错误分类某点的概率。我们通过揭示可复现性（Impagliazzo等人，2022）与单一文化鲁棒性之间出人意料的联系，设计了实现这类保证的方法。具体来说，我们证明了如果基础学习器是 $\rho$-可复现的，那么独立运行它……

    arXiv:2611.24881v1 Announce Type: new  Abstract: We introduce a notion of monoculture robustness in Probably Approximately Correct (PAC) learning, as a way to mitigate the problem of algorithmic monoculture (Kleinberg and Raghavan, 2021): the tendency of models sharing the same components (e.g., training data) to fail on the same inputs. Specifically, we ask whether a learner with access to a single dataset can output $k$ individually accurate hypotheses whose joint failure probability is comparable to training on $k$ independent datasets. We say that such a learning procedure achieves monoculture robustness, and we quantify this by bounding the conjunctive error that all $k$ hypotheses misclassify points in an individual sense. We design methods for achieving this type of guarantee by drawing a surprising connection between replicability (Impagliazzo et al., 2022) and monoculture robustness. Specifically, we show that if the base learner is $\rho$-replicable, then running it independe
    
[^7]: 仅通过查询利用大语言模型审计黑箱临床预测模型：一项植入缺陷的盲测

    Auditing black-box clinical prediction models with a large language model through queries alone: a blinded test with planted flaws

    [https://arxiv.org/abs/2611.24877](https://arxiv.org/abs/2611.24877)

    该论文在盲测条件下证明，大语言模型可以仅通过查询黑箱临床预测模型来构建高保真代理模型并检出人为植入的缺陷（如反向风险因素、被低估的亚组和不合理阶跃）。

    

    arXiv:2611.24877v1 公告类型：cross 摘要：大语言模型（LLM）被用于解释临床预测模型，其承诺在于能够说明模型计算了什么、发现其缺陷，甚至揭示其中蕴含的新知识。我们在答案可被检验的范围内对这三项承诺进行了测试。基于国际卒中试验中较早随机分组的患者，我们针对两种结局训练了四类模型（逻辑回归、可解释提升机、梯度提升树、神经网络），在24个模型副本中各植入一个缺陷（一个被低估预测的亚组、一个反向的风险因素或一个不合理的阶跃），并将其与8个无缺陷的干净模型混合。GPT-6 Astra 通过一个仅返回模型预测结果的工具，在盲法条件下对每个模型进行了三次审计；评判规则在任何运行之前即已预先确定，并由程序执行。其代理模型（中位数含128个项）以平均 R² 为 0.976 复现了被审计模型的对数几率（128项的加性代理模型为 0.959）……

    arXiv:2611.24877v1 Announce Type: cross  Abstract: Large language models (LLMs) are used to explain clinical prediction models, on the promise that they can say what a model computes, find its faults and even reveal new knowledge in it. We tested the three promises where the answers could be checked. From the International Stroke Trial we trained four kinds of model (logistic regression, explainable boosting machine, gradient-boosted trees, neural network) for two outcomes on patients randomised earlier, planted one flaw in each of 24 copies (an under-predicted subgroup, a reversed risk factor or an implausible step) and mixed them with the 8 clean models. GPT-6 Astra audited each model three times, blind, through a tool that only returned the model's predictions; the judging rules were committed before any run and applied by program. Its surrogate models, with a median of 128 terms, reproduced the audited models' log-odds with a mean $R^2$ of 0.976 (0.959 for an additive surrogate of 
    
[^8]: 面向可控自动驾驶场景生成的多智能体自博弈交互学习

    Learning to Interact in Multi-Agent Self-Play for Controllable Autonomous Driving Scenario Generation

    [https://arxiv.org/abs/2611.24876](https://arxiv.org/abs/2611.24876)

    该论文提出引入智能体相对动态目标的扩展多智能体自博弈框架，实现了对攻击性尾随、对抗性切入等安全关键交互驾驶场景的精准可控生成，解决了静态目标方法无法模拟复杂智能体交互的问题。

    

    大规模多智能体自博弈能够在仿真中有效训练鲁棒的驾驶策略，但如何控制这些模型生成特定的、安全攸关的边缘场景仍是一个关键挑战。现有的目标条件自博弈框架主要依赖静态的地图位置目标，无法可靠地产生复杂的智能体交互行为，例如攻击性尾随或对抗性切入，而这些交互行为对于严格的自动驾驶车辆（AV）测试是必不可少的。为解决这一问题，我们提出了一个扩展的多智能体自博弈框架，引入了智能体相对的动态目标。通过让策略以这种扩展的目标表示为条件，我们的框架能够实现有针对性的、可控的场景合成。我们证明了所提方法能够成功生成静态目标基线无法模拟的交互式驾驶行为，为自动驾驶测试中的长尾场景生成提供了灵活的工具。

    arXiv:2611.24876v1 Announce Type: cross  Abstract: Large-scale, multi-agent self-play trains robust driving policies effectively in simulation, but controlling these models to generate specific, safety-critical edge cases remains a key challenge. Existing goal-conditioned self-play frameworks rely primarily on static map-location goals and fail to reliably produce complex agent interactions such as aggressive tailgating or adversarial cut-offs. These agent interactions are necessary for rigorous autonomous vehicle (AV) testing. To address this, we present an extended multi-agent self-play framework that introduces agent-relative dynamic goals. By conditioning policies on this extended goal representation, our framework enables targeted, controllable scenario synthesis. We demonstrate that our approach successfully generates interactive driving behaviors that static-goal baselines cannot simulate, providing a flexible tool for long-tail scenario generation in AV testing.
    
[^9]: TangentPDE：利用稀疏有限差分监督学习局部算子响应

    TangentPDE: Learning Local Operator Responses with Sparse Finite-Difference Supervision

    [https://arxiv.org/abs/2611.24873](https://arxiv.org/abs/2611.24873)

    提出TangentPDE方法，仅利用参考求解器在稀疏成对输入处的状态输出构造有限差分响应监督，无需导数标签、伴随求解或完整雅可比矩阵，即可同时学习神经PDE算子的取值及其局部变化。

    

    神经PDE算子通常通过状态重构进行优化，这在很大程度上仅约束了算子在观测输入处的取值，却未能充分刻画其对局部扰动的敏感性。要更忠实地逼近算子行为，不仅需要准确的状态预测，还需要沿局部扰动方向的正确变化。我们提出TangentPDE，该方法利用有限差分近似方向导数，并基于参考求解器在稀疏成对输入处的状态输出来构建与任务相关的有限步长响应。其目标函数共同约束状态保真度、响应方向对齐以及响应幅值校准，从而同时学习算子的取值及其局部变化。由于响应标签仅由成对输入处的状态输出构建，该方法既不需要显式的导数标签，也不需要线性化或伴随PDE求解，更不需要完整的雅可比矩阵。

    arXiv:2611.24873v1 Announce Type: new  Abstract: Neural PDE operators are often optimized through state reconstruction, which largely constrains operator values at observed inputs but does not fully characterize their sensitivity to local perturbations. A more faithful approximation of operator behavior requires not only accurate state predictions, but also correct variations along local perturbation directions. We present TangentPDE, which uses finite differences to approximate directional derivatives and constructs task-relevant finite-step responses from the state outputs of a reference solver at sparse paired inputs. Its objective jointly constrains state fidelity, response-direction alignment, and response-magnitude calibration, thereby learning both operator values and their local variations. Because the response labels are constructed solely from state outputs at paired inputs, the method requires neither explicit derivative labels, linearized or adjoint PDE solves, nor full Jac
    
[^10]: Atomic MathReasoner：面向数学大语言模型的原子推理技能实例级组合

    Atomic MathReasoner: Instance-Level Composition of Atomic Reasoning Skills for Mathematical LLMs

    [https://arxiv.org/abs/2611.24846](https://arxiv.org/abs/2611.24846)

    提出了 Atomic MathReasoner（AMR），一种实例级框架：通过训练能力专用的 LoRA 专家库并用无梯度优化在验证子集上发现低损失组合原型，再依据参考解答的词元级负对数似然为每个数学问题分配最合适的原子推理技能组合，从而有效利用原子能力提升数学大语言模型的推理表现。

    

    大多数现有方法将大语言模型（LLM）的数学推理当作一种整体性能力加以增强，然而不同的数学问题实际上需要不同的推理技能。近期的研究从更细粒度的视角出发，将数学推理分解为原子技能，并为各项能力开发专门的模块。但是，如何有效利用这些原子能力仍未得到充分探索。为应对这一挑战，我们提出了 Atomic MathReasoner（AMR），一个用于组合原子推理技能的实例级框架。AMR 首先训练一个面向特定能力的 LoRA 专家库，并在留出的验证子集上进行无梯度优化，以发现一组多样化的低损失组合原型。它利用参考解答的词元级负对数似然来评估每个原型的偏好，并通过……对模糊的分配进行过滤（原文摘要在此处截断）。

    arXiv:2611.24846v1 Announce Type: new  Abstract: Most existing approaches enhance mathematical reasoning in large language models (LLMs) as a monolithic capability, despite the fact that different mathematical problems require different reasoning skills. Recent work has explored a more fine-grained perspective by decomposing mathematical reasoning into atomic skills and developing specialized modules for individual capabilities. However, how to effectively utilize these atomic abilities remains underexplored. To address this challenge, we propose Atomic MathReasoner (AMR), an instance-level framework for composing atomic reasoning skills. AMR first trains a library of capability-specific LoRA experts and performs gradient-free optimization over held-out validation subsets to discover a diverse set of low-loss composition prototypes. It evaluates the preference of each prototype using the token-level negative log-likelihood of reference solutions, filters ambiguous assignments through a
    
[^11]: 测量与改进车辆空气动力学神经代理模型的迁移能力

    Measuring and Improving Transfer of Neural Surrogates for Vehicle Aerodynamics

    [https://arxiv.org/abs/2611.24843](https://arxiv.org/abs/2611.24843)

    该论文通过大规模受控基准测试首次系统评估了车辆空气动力学神经代理模型的跨数据集迁移行为，发现微调仅在不到一半的情况下优于从头训练，且会损害模型在预训练域上的准确性。

    

    车辆外部空气动力学的神经代理模型是基于物理的计算流体动力学（CFD）仿真的快速替代方案。其根本局限性在于缺乏跨车辆类别的零样本泛化能力，这导致每种新车辆设计都需要开展昂贵的定制训练活动。与其他领域的预训练模型类似，预训练的代理模型应当能够仅利用少量仿真就适应新的车辆类别，同时在其预训练域上保持准确。此前尚无研究系统地测试过这种迁移行为。我们对五个基于DrivAerML预训练的代理模型，在四个独立的目标数据集上开展了大规模、受控的基准测试研究。这些数据集在几何形状、CFD求解器、湍流模型和流动条件方面均与源数据不同。我们发现，在不到一半的对比实验中，微调这些代理模型优于从头训练，并且会降低代理模型（在预训练域上）的准确性。

    arXiv:2611.24843v1 Announce Type: new  Abstract: Neural surrogates for external vehicle aerodynamics are fast alternatives to physics-based computational fluid dynamics (CFD) simulations. Their foundational limitation is a lack of zero-shot generalization across vehicle classes, which results in an expensive, custom training campaign for each new vehicle design. Similar to pretrained models in other fields, a pretrained surrogate should adapt to a new vehicle class using only a few simulations and stay accurate on its pretraining domain. No prior study has systematically tested this transfer behavior. We conduct a large-scale, controlled benchmarking study on five DrivAerML-pretrained surrogates across four independent target datasets. These datasets differ from the source in geometry, CFD solver, turbulence model, and flow conditions. We find that fine-tuning these surrogates outperforms training from scratch in fewer than half of the comparisons and lowers the accuracy of the surroga
    
[^12]: 动作是否重要？探究动作条件语义预测作为自动驾驶世界模型的作用

    Does the Action Matter? Probing Action-Conditioned Semantic Forecasting as a World Model for Autonomous Driving

    [https://arxiv.org/abs/2611.24840](https://arxiv.org/abs/2611.24840)

    本文提出通过测试时动作干预来量化评估动作条件语义预测模型的动作依赖性，以检验这类模型作为自动驾驶世界模型时是否真正依据动作进行预测。

    

    世界模型在自动驾驶领域被日益广泛地用于预测场景在自车动作下的演变，从而支持反事实推理和基于模型的规划。然而，现有评估主要集中于预测保真度和规划性能，很少验证预测结果是否确实依赖于所条件化的动作。本文研究了动作条件语义预测中的动作依赖性问题，该任务根据过去的语义分割图和自车动作来预测未来的语义分割。我们将具有特征级和输入级动作条件的预测器与没有动作输入的完全相同的模型进行对比，并通过测试时的动作干预（以标准语义分割指标评分）来衡量动作依赖性。该评估方法在具有已知动作依赖动态特性的合成环境中得到验证，并被应用于Cityscapes和BDD100K两个数据集，既在各自数据集内部进行评估，也在两个车辆平台之间进行跨数据集评估。

    arXiv:2611.24840v1 Announce Type: cross  Abstract: World models are increasingly used in autonomous driving to predict how a scene evolves under the ego vehicle's actions, supporting counterfactual reasoning and model-based planning. However, existing evaluations focus on prediction fidelity and planning performance, and rarely verify that the predictions depend on the conditioning action. In this paper, we study action dependence in action-conditioned semantic forecasting, which predicts future semantic segmentation from past segmentations and the ego action. We compare a forecaster with feature-wise and input-level action conditioning against an identical model without action input, and measure action dependence with test-time action interventions scored by standard segmentation metrics. The evaluation is validated in a synthetic environment with known action-dependent dynamics and applied to Cityscapes and BDD100K, both within each dataset and across the two vehicle platforms. Exper
    
[^13]: Kaer：面向多智能体系统的安全感知式重规划框架

    Kaer: Security-Aware Replanning for Multi-Agent Systems

    [https://arxiv.org/abs/2611.24839](https://arxiv.org/abs/2611.24839)

    Kaer通过抽象模板传递智能体反馈、将计划修订限制在确定性修复算子内并强制执行预设策略，防止恶意智能体劫持多智能体系统的重规划过程。

    

    arXiv:2611.24839v1 公告类型：cross 摘要：多智能体系统（MAS）的一种流行架构依赖于一个编排器，由它生成初始计划、协调计划的执行，并在任务无法事先完全明确的情况下，根据智能体的反馈动态修订计划。这种动态重规划引入了一个关键漏洞：恶意智能体可以在其反馈中嵌入指令，从而劫持重规划过程。现有的防御手段仅专注于针对单个工具的访问控制或静态的、执行前的策略强制执行，使得重规划阶段处于无保护状态。我们提出了Kaer，一个具有安全感知重规划能力的多智能体框架。Kaer通过严格的抽象模板向重规划过程传递诊断信息，从而保护规划器免受不可信智能体反馈的影响。计划修订被限制在一组确定性的修复算子之内，且所有修改后的计划都必须满足预设策略。我们构建了Cintra，一个基于模拟的……（基准，原文此处截断）

    arXiv:2611.24839v1 Announce Type: cross  Abstract: One popular architecture for multi-agent systems (MAS) relies on an orchestrator that generates an initial plan, coordinates its execution, and dynamically revises the plan based on agent feedback when tasks cannot be fully specified in advance. This dynamic replanning introduces a critical vulnerability: malicious agents can embed instructions within their feedback to hijack the replanning process. Existing defenses focus exclusively on per-tool access control or static, pre-execution policy enforcement, leaving the replanning phase unsecured.   We introduce Kaer, a multi-agent framework featuring security-aware replanning. Kaer shields the planner from untrusted agent feedback by conveying diagnostics to the replanning procedure strictly through abstract templates. Plan revisions are restricted to a set of deterministic repair operators, and all modified plans must satisfy preset policies. We create Cintra, a benchmark of simulated u
    
[^14]: 生存分析基准测试无法告诉你的事：模型选择与数据集条件的影响

    What Survival Benchmarks Don't Tell You: Impact of Model Selection and Dataset Regimes

    [https://arxiv.org/abs/2611.24837](https://arxiv.org/abs/2611.24837)

    该研究通过涵盖10种方法与49个数据集的大规模中立基准测试，揭示了评估设计对结论的决定性影响，并发现对深度学习和树模型而言，以IBS而非C-index为目标进行超参数优化，能以极低成本大幅提升概率预测的准确性与校准度。

    

    尽管生存分析方法层出不穷，但当前的基准测试往往在狭窄的条件下进行：它们严重依赖一致性指数（C-index）进行评估和模型选择，并且经常排除深度学习（DL）方法或未能充分代表具有挑战性的数据集条件。为了解决这些局限性，我们开展了一项大规模、中立的大规模基准测试，在49个多样化数据集上比较了10种经典方法、基于树的方法和深度学习方法。至关重要的是，我们将模型选择标准（分别针对C-index、综合Brier分数（IBS）或平均绝对误差（MAE-PO）进行优化）视为一个独立的实验变量。通过贝叶斯混合效应建模，我们证明了评估设计在很大程度上决定了研究结论。对于复杂的深度学习和基于树的架构，选择超参数时以IBS而非C-index为优化目标，能以几乎可以忽略不计的代价，显著提升概率预测的准确性和校准度。

    arXiv:2611.24837v1 Announce Type: new  Abstract: Despite the proliferation of survival analysis methods, current benchmarks often operate under narrow conditions: they heavily rely on the concordance index (C-index) for evaluation and model selection, and frequently either exclude deep learning (DL) methods or underrepresent challenging dataset regimes. To address these limitations, we conduct a large-scale, neutral benchmark comparing 10 classical, tree-based, and DL methods across 49 diverse datasets. Crucially, we treat the model selection criterion (optimizing for C-index, Integrated Brier Score (IBS), or Mean Absolute Error (MAE-PO)) as an independent experimental variable. Using Bayesian mixed-effects modeling, we demonstrate that evaluation design heavily dictates conclusions. For complex DL and tree-based architectures, selecting hyperparameters to optimize for IBS instead of C-index yields substantial improvements in probabilistic accuracy and calibration at a negligible cost 
    
[^15]: 被忽视与未被充分利用：原型是表格数据可解释AI（XAI）中缺失的一环

    Overlooked and Underutilized: Prototypes are the Missing Piece in Tabular XAI

    [https://arxiv.org/abs/2611.24831](https://arxiv.org/abs/2611.24831)

    论文提出将原型（尤其是通过将相关特征聚合成有界子空间、体现人类可理解概念的原型部分）作为特征归因方法的直观补充，以降低认知负担的方式帮助用户审计表格数据上的复杂机器学习模型。

    

    可解释人工智能（XAI）经常应用于高风险的表格数据，然而该领域严重依赖特征归因方法。虽然这些方法提供了技术上的透明性，但让用户评估数十个孤立的特征需要非专业用户付出过多的认知努力。在本文中，我们认为原型仍然被忽视和未被充分利用，应将其整合为审计复杂机器学习模型的补充性、直观的入口。虽然完整的范例可以作为基于原型的解释的直观基线，但我们强调原型部分对表格数据尤其有价值。通过将相关特征分组到有界的子空间中，原型部分在技术上体现了人类可理解的概念，用连贯的语义单元取代孤立的特征。我们通过两个主要视角支持这一整合。在认知层面，这些部分与Rosch的分类理论和Kahneman（的理论）相符

    arXiv:2611.24831v1 Announce Type: new  Abstract: Explainable AI (XAI) is frequently applied to high-stakes tabular data, yet the field relies heavily on feature attribution methods. While these approaches offer technical transparency, evaluating dozens of isolated features demands excessive cognitive effort from non-expert users. In this paper, we argue that prototypes remain overlooked and underutilized, and should be integrated as a complementary, intuitive gateway to auditing complex machine learning models. Although entire exemplars function as an intuitive baseline for prototype-based explanations, we highlight prototypical parts as particularly valuable for tabular data. By grouping related features into bounded subspaces, prototypical parts technically embody human-understandable concepts, replacing isolated features with cohesive semantic units. We support this integration through two primary lenses. Cognitively, these parts align with Rosch's categorization theory and Kahneman
    
[^16]: 利用生成模型合成高光谱数据以训练光谱解混方法，实现低成本作物残茬覆盖制图

    Synthesizing hyperspectral Data using generative Models to train spectral Unmixing Methods for low-cost Crop Residue Cover Mapping

    [https://arxiv.org/abs/2611.24827](https://arxiv.org/abs/2611.24827)

    本研究提出通过中间条件数据生成器将插值与丰度估计解耦的框架，利用生成模型合成高光谱数据来训练解混模型，从而在地面真值数据稀疏的条件下实现低成本的作物残茬覆盖制图。

    

    由多伦多大学航空航天团队空间系统分部开发的作物残茬高光谱制图田间成像纳卫星（FINCH）的光谱范围，给通过高光谱解混确定作物残茬覆盖丰度分数带来了重大挑战。标准指标的严重缺乏使得必须使用复杂的、数据驱动的解混模型。复杂的解混模型需要密集且分布良好的流形，而地面真值数据集稀疏且昂贵。为了更好地利用现有地面真值数据集中的信息内容，本研究提出了一个框架，通过一个中间条件数据生成器将插值与丰度制图估计解耦，这与传统的单一模型解混流程形成对比。这种解耦使得能够纳入关于光谱的物理先验知识，而这些知识是解混模型本身无法获得的，从而人为地……

    arXiv:2611.24827v1 Announce Type: new  Abstract: The spectral range of the Field Imaging Nanosatellite for Crop residue Hyperspectral Mapping (FINCH), developed by the University of Toronto Aerospace Team's Space System Division, poses significant challenges for hyperspectral unmixing to determine crop residue cover fractional abundances. The severe lack of standard indices necessitates the use of complex, data-driven unmixing models. Complex unmixing models require dense, well-distributed manifolds, whereas ground-truth datasets are sparse and expensive. To better leverage the information content in existing ground-truth datasets, this study presents a framework that decouples interpolation and abundance-mapping estimation via an intermediary conditional data generator, contrasting with traditional single-model unmixing pipelines. This decoupling enables the inclusion of physical prior knowledge about spectra, which is not natively accessible to unmixing models, thereby artificially e
    
[^17]: 无需预测的预后：生存的内在几何学

    Prognosis Without Prediction: The Intrinsic Geometry of Survival

    [https://arxiv.org/abs/2611.24799](https://arxiv.org/abs/2611.24799)

    该论文提出自组织生存流形（SOSM）深度学习框架，通过将无监督表示学习与生存标签解耦，并借助伪时间加权、曲率最小化的图拉普拉斯潜空间把生存进展视为数据内在几何的涌现属性，从而在184例透明细胞肾细胞癌多组学队列上无需显式结局回归即实现了0.768±0.068的平均一致性指数。

    

    在高维环境中建模系统性失败，从根本上受限于那些试图预测稀疏、含噪或删失终点的监督式算法。为了打破这种对显式结局回归的依赖，我们构建了自组织生存流形（Self-Organizing Survival Manifolds, SOSM）框架，这是一种将无监督表示学习与生存标签解耦的深度学习架构，并将事件发生时间的进展视为一种涌现的几何属性。该系统将异构的多组学数据阵列投影到一个统一的潜在空间中，该空间由一个伪时间加权、曲率最小化的图拉普拉斯算子所支配，从而将患者轨迹组织成沿低耗散测地线流的路径。为验证该框架，我们将此架构应用于一个包含184名患者的透明细胞肾细胞癌多组学队列。在一个确定性的5折交叉验证流程中，SOSM取得了平均0.768 ± 0.068的一致性指数，同时其外部（截断处）……

    arXiv:2611.24799v1 Announce Type: new  Abstract: Modeling systemic failure in high-dimensional environments is fundamentally limited by supervised algorithms that attempt to predict sparse, noisy or censored endpoints. To break this reliance on explicit outcome regression, we engineered the Self-Organizing Survival Manifolds (SOSM) framework, a deep learning architecture that decouples unsupervised representation learning from survival labels and treats time-to-event progression as an emergent geometric property. The system projects heterogeneous multi-omics arrays into a unified latent space governed by a pseudotime-weighted, curvature-minimizing graph Laplacian, organizing patient trajectories along low-dissipation geodesic flows. To validate the framework, we applied this architecture to a 184-patient clear cell renal cell carcinoma multi-omics cohort. Across a deterministic 5-fold cross-validation pipeline, SOSM achieved a mean Concordance Index of $0.768 \pm 0.068$, while out-of-f
    
[^18]: ReLU网络中的仅偏置训练动力学

    Bias-Only Training Dynamics in ReLU Networks

    [https://arxiv.org/abs/2611.24797](https://arxiv.org/abs/2611.24797)

    该论文提出用“激活码”刻画单隐层ReLU网络仅训练偏置时的训练动力学，证明训练等价于一系列以激活码为特征的线性最小二乘问题，且网络变宽时神经元切换比例趋近于零、动力学趋于线性。

    

    具有随机初始化固定权重、仅训练偏置的神经网络是通用逼近器，但仅偏置训练的动力学在很大程度上仍未被探索。我们通过“激活码”来研究单隐层ReLU网络中的这些动力学——激活码是活跃神经元的二进制模式，它将输入空间划分为凸多面体区域。我们证明，关于偏置的梯度仅通过每个输入的激活码依赖于该输入，因此训练动力学可以用这些激活码来刻画。特别地，只要激活码保持不变，训练就退化为以激活码为特征的线性最小二乘问题。训练因此可以理解为一系列最小二乘问题，当神经元改变其激活状态时发生状态转换。我们证明，随着网络宽度增大，发生切换的神经元比例趋于消失，训练动力学趋近于线性（网络）的动力学……

    arXiv:2611.24797v1 Announce Type: cross  Abstract: Neural networks with randomly initialized, fixed weights and trainable biases are universal approximators, yet the dynamics of bias-only training remain largely unexplored. We study these dynamics in one-hidden-layer ReLU networks through activation codes, the binary patterns of active neurons that partition the input space into convex polyhedral regions. We show that the gradients with respect to the biases depend on each input only through its activation code, and hence the training dynamics can be characterized with these codes. In particular, as long as the activation codes remain fixed, training reduces to a linear least-squares problem with the codes as features. Training can then be understood as a sequence of least-squares problems, with transitions occurring when neurons change their activation states. We show that the fraction of switching neurons vanishes as the network widens, and the training dynamics approach those of lin
    
[^19]: 问题中的大象：统一模型描绘其答案背后的概念

    The Elephant in the Question: Unified Models Picture the Concepts Behind Their Answers

    [https://arxiv.org/abs/2611.24793](https://arxiv.org/abs/2611.24793)

    本文提出无需训练解码器的V-lens方法，利用统一多模态模型图像生成器对文本隐藏状态的访问能力，可视化模型推理中连接问题与答案的关键中间概念，并通过lens tuning显著提升代词指代等概念的可视化准确率。

    

    可视化中间表示有助于我们检视模型输出背后的推理步骤。统一多模态模型在同一架构中融合了理解与生成，为实现这一目标提供了机会：其图像生成器可以访问处理文本时形成的表示。我们提出V-lens，利用这一连接，无需训练解码器即可可视化所选token的隐藏状态。在四个基于扩散模型和两个纯自回归的统一模型上，生成的图像揭示了连接问题与答案的关键中间概念，即使这些概念从未被明确提及。对照实验展示了这些可视化内容如何随上下文、token位置和模型深度而变化。随后我们提出lens tuning（透镜调优），通过V-lens用图像对选定的隐藏状态施加监督。在BAGEL、SenseNova-U1和HunyuanImage-3.0上，代词指代对象的正确可视化率显著提升（原摘要在此处被截断）。

    arXiv:2611.24793v1 Announce Type: cross  Abstract: Visualizing intermediate representations could help us inspect the reasoning steps underlying a model output. Unified multimodal models offer this opportunity by combining understanding and generation in one architecture. Their image generators can access representations formed while processing text. We introduce the V-lens, which uses this connection to visualize a chosen token's hidden state without training a decoder. Across four diffusion-based and two purely autoregressive unified models, the resulting images reveal key intermediate concepts that connect a question to its answer even when the concepts are never named explicitly. Controlled tests show how this content changes with context, token position, and depth. We then introduce lens tuning, which supervises selected hidden states with pictures through the V-lens. Across BAGEL, SenseNova-U1, and HunyuanImage-3.0, the rate of correct visualizations of pronoun referents rises fr
    
[^20]: 免训练的运动统计特征匹敌自监督行为编码器

    Training-free motion statistics match self-supervised behavior encoders

    [https://arxiv.org/abs/2611.24792](https://arxiv.org/abs/2611.24792)

    本文证明了一种免训练的简单运动统计表示（逐帧通用测量+固定时间窗口汇总+PCA降维）在预注册的公平比较协议下，在多个行为识别基准上超越或匹敌最先进的自监督行为编码器，表明自监督编码器并未真正胜过简单的人工特征。

    

    动物行为的自监督编码器通常在人工构建的运动特征上进行训练，或以这些特征作为输入。我们提出疑问：在公平比较的条件下，这些编码器是否能胜过这类人工特征。我们构建了一种免训练的表示方法，它由通用的逐帧测量指标组成，在固定时间窗口上进行汇总，并通过PCA降维。在预注册协议下（即使用相同大小的冻结表示和一个共享分类器），我们将其与重新训练及已发布的编码器（BAMS、hBehaveMAE）进行了比较。在CalMS21数据集上，该方法达到85.8的宏平均F1分数，而最佳编码器为82.4；在仅有四个标注视频时两者打平，而当标注者更换或需要从少量视频中学习新行为时，该方法保持领先；这一结论在以三个随机种子重新训练编码器的情况下依然成立。在MABe22小鼠三联体数据集上，该方法得分为83.7，而已发布的hBehaveMAE为80.3（差距+3.4，95%置信区间+2.1至+4.7）。在第二个物种——MABe22果蝇群体数据集上，同样的方法在50个任务上取得51.6的平均F1分数，而BAMS为48.2（原文在此处截断）。

    arXiv:2611.24792v1 Announce Type: cross  Abstract: Self-supervised encoders for animal behavior are often trained on, or fed, hand-built movement features. We ask whether they beat those ingredients when compared fairly. A training-free representation, made of generic per-frame measurements summarized over fixed time windows and reduced with PCA, is compared with retrained and released encoders (BAMS, hBehaveMAE) under a preregistered protocol: frozen representations of equal size and one shared classifier. On CalMS21 it reaches 85.8 macro F1 against 82.4 for the best encoder, ties with four labelled videos, and leads when the annotator changes or new behaviors must be learned from a few videos; the reading holds for encoders retrained with three seeds. On the MABe22 mouse triplets it scores 83.7 against 80.3 for the released hBehaveMAE (gap +3.4, 95% interval +2.1 to +4.7). On a second species, the MABe22 fly groups, the same recipe scores 51.6 mean F1 over 50 tasks against 48.2 for B
    
[^21]: 学习轨迹在神经网络权重空间与内部表示中留下可解码的痕迹

    Learning Trajectories Leave Decodable Traces in Neural Network Weight Space and Internal Representations

    [https://arxiv.org/abs/2611.24782](https://arxiv.org/abs/2611.24782)

    该研究发现神经网络的最终权重与隐藏表示中保留了可解码的学习历史信息——训练后的网络能被准确匹配回其初始权重，且先前任务的训练顺序可通过线性探针被100%解码。

    

    在相同任务上训练的神经网络会发展出相关的权重空间结构，但在学习之后仍可相互区分。这些差异究竟只是反映了不同的替代解，还是它们保留了关于个体学习历史的信息？我们检验了最终权重是否能揭示初始化信息、任务顺序以及更早的轨迹状态。训练重组了整个网络集合：初始成对距离并不能稳定地预测最终距离。尽管如此，每个数据集的30个训练后的网络都能在十个候选者中被正确匹配回各自的初始权重。共享相同初始化的网络即使在不同的批次调度下训练，也会发展出可识别的最终隐藏表示，尽管通过与初始活动的直接比较无法恢复身份。学习顺序同样保持可读性：一个线性探针在80个保留测试网络中，在经过十个epoch的共同（训练）之后，以100%的准确率解码出两个较早任务的顺序。

    arXiv:2611.24782v1 Announce Type: cross  Abstract: Neural networks trained on the same task develop related weight-space structures, yet remain distinguishable after learning. Do these differences merely reflect alternative solutions, or do they preserve information about individual learning histories? We tested whether final weights reveal initialization, task order, and earlier trajectory states. Training reorganized the ensemble: initial pairwise distances did not consistently predict final distances. Nevertheless, all 30 trained networks were correctly matched to their initial weights among ten candidates per dataset. Networks sharing an initialization also developed recognizable final hidden representations across different batch schedules, although direct comparisons with initial activity did not recover identity. Learning order remained readable as well. A linear probe decoded the order of two earlier tasks with 100% accuracy on 80 held-out networks after ten epochs of a common 
    
[^22]: 连续控制中离散视频表征的重构式与预测式预训练对比

    Reconstruction vs. Prediction Pre-Training of Discrete Video Representations for Continuous Control

    [https://arxiv.org/abs/2611.24781](https://arxiv.org/abs/2611.24781)

    该论文通过容量匹配的对照实验发现，在离散视频表征预训练中，重构式自编码目标能保留更多物理状态信息，而预测式（V-JEPA）目标在自回归滚动中误差累积更少，并对低层视觉扰动更鲁棒。

    

    自监督视频编码器通常采用重构像素或预测潜在特征两种目标进行训练，然而这些目标如何塑造面向下游动态建模的表征仍不清楚。我们提出了一项容量匹配的对照研究来隔离这一选择：两个具有共享FSQ瓶颈的相同视频编码器，在连续控制环境中分别以自编码和V-JEPA目标进行训练。我们引入了一种简单的非对称离散化方案，使相同的瓶颈也能应用于预测模型。所得表征呈现出明确的权衡：自编码器保留了更多的物理状态信息，并在单步潜在预测上略占优势，而预测式编码器在自回归滚动中累积的误差更少。结构探针分析表明，预测式编码器的这一优势与其对低层视觉扰动更强的鲁棒性以及对时间上下文更强的依赖性相关。

    arXiv:2611.24781v1 Announce Type: new  Abstract: Self-supervised video encoders are commonly trained to either reconstruct pixels or predict latent features, yet it remains unclear how these objectives shape representations for downstream dynamics. We present a capacity-matched study isolating this choice: two identical video encoders with a shared FSQ bottleneck are trained under autoencoding and V-JEPA objectives on a continuous control environment. We introduce a simple asymmetric discretization scheme to apply the same bottleneck to the predictive model. The resulting representations exhibit a clear trade-off: the autoencoder preserves substantially more physical state information and achieves slightly better one-step latent prediction, while the predictive encoder accumulates less error under autoregressive rollout. Structural probes suggest that this advantage is associated with greater robustness to low-level visual perturbations and stronger dependence on temporal context. Thes
    
[^23]: 反事实世界模型：能够推理“如果……会怎样”的JEPA

    Counterfactual World Models: JEPAs That Reason About "What If"

    [https://arxiv.org/abs/2611.24778](https://arxiv.org/abs/2611.24778)

    该论文证明仅用事实数据训练的世界模型无法回答“同样扰动下换一个动作结果会如何”的反事实问题，并提出利用“自然不做跃变”的一阶近似构建反事实世界模型，使其无需反事实数据和额外参数即可区分糟糕决策与坏运气。

    

    世界模型运行在一个它们无法完全观测的世界中。当一种隐藏的扰动（例如推动船只的水流）改变了某个动作的结果时，随之而来的便是一个反事实问题：在同样的扰动作用下，如果采取不同的动作，会发生什么？这个问题的答案能够将糟糕的决策与坏运气区分开来。我们证明，仅在事实转移（真实发生的转变）上训练的世界模型无法依靠自身回答这个问题，无论其数据或模型容量如何。对于许多物理系统而言，观测到的结果加上动力学的一个简单性质就足够了：natura non facit saltus，即自然不做跃变。当动作和扰动作为相互独立的力分别作用时，它们的相互作用在时间上是二阶的，因此反事实结果在一阶近似下等于观测到的结果，再加上世界模型在替代动作与事实动作下预测结果之间的差值。由此便得到了一个无需反事实数据、无需额外参数的反事实世界模型。

    arXiv:2611.24778v1 Announce Type: cross  Abstract: World models operate in a world they cannot fully observe. When a hidden disturbance, such as a current pushing a boat, changes the outcome of an action, a counterfactual question follows: what would have happened under a different action, subject to the same disturbance? Its answer separates bad decisions from bad luck. We prove that world models trained on factual transitions cannot answer it on their own, whatever their data or capacity. For many physical systems, the observed outcome and a simple property of the dynamics suffice: natura non facit saltus, nature makes no leaps. When actions and disturbances act as separate forces, their interaction is second-order in time, so the counterfactual outcome is, to first order, the observed outcome plus the difference between the world model's predictions under the alternative and factual actions. This yields a counterfactual world model with no counterfactual data, extra parameters, or r
    
[^24]: 面向单细胞时空动力学的水平流匹配

    Horizontal Flow Matching for Single-Cell Spatiotemporal Dynamics

    [https://arxiv.org/abs/2611.24764](https://arxiv.org/abs/2611.24764)

    该论文提出水平流匹配方法，通过在连续-反应方程与水平投影中排除共享刚体运动的代价，使得从空间转录组学数据中学习单细胞时空的组织结构、分子状态与细胞丰度变化时不受切片旋转平移等坐标变化的干扰。

    

    从空间转录组学中学习细胞的时空动力学，需要将在不同坐标系下获取的不同细胞快照连接起来。切片的旋转或平移可能表现为细胞运动，从而扭曲所推断的对应关系。我们提出水平流匹配，它能够在共享刚体运动之外学习组织结构、分子状态和细胞丰度的变化。该动力学遵循一个连续-反应方程，其作用对共享刚体运动不施加代价；由此得到的水平投影器在学习阶段从空间预测误差中、在生成阶段从预测速度中移除该分量。一个等价的端点问题在 Wasserstein-Fisher-Rao 几何下联合估计对应关系与相对位姿，并给出解析路径，其速度与增长作为回归目标。对这些目标进行回归可恢复水平动力学……（原文摘要此处截断）

    arXiv:2611.24764v1 Announce Type: cross  Abstract: Learning the spatiotemporal dynamics of cells from spatial transcriptomics requires connecting snapshots of different cells acquired in different coordinate frames. A rotation or translation of a section can appear as cell movement and distort the inferred correspondence. We introduce horizontal flow matching, which learns changes in tissue organization, molecular state, and cell abundance up to shared rigid motion. The dynamics follow a continuity--reaction equation whose action assigns no cost to shared rigid motion, and the resulting horizontal projector removes this component from spatial prediction errors during learning and from predicted velocities during generation. An equivalent endpoint problem jointly estimates correspondence and relative pose in Wasserstein--Fisher--Rao geometry and yields analytic paths whose velocity and growth serve as regression targets. Regressing these targets recovers the horizontal dynamics under id
    
[^25]: Elysium：免仿真的非平衡薛定谔桥

    Elysium: Simulation-Free Unbalanced Schr\"odinger Bridges

    [https://arxiv.org/abs/2611.24763](https://arxiv.org/abs/2611.24763)

    Elysium 提出了一种免仿真的非平衡薛定谔桥求解方法，联合学习状态转移与质量变化，无需在训练中模拟轨迹即可从快照数据重建随机细胞动力学。

    

    arXiv:2611.24763v1 公告类型： cross 摘要：从快照数据重建细胞动力学是一项具有挑战性的任务，因为单细胞测量的破坏性使得无法随时间追踪同一批细胞。细胞动力学本质上是随机的过程，并且由于增殖与凋亡而涉及种群规模的变化。基于生长的方法将种群变化表示为密度上的增长率，这种方式无法将个体的退出和进入分配到具体细胞。非平衡薛定谔桥能够对个体层面的事件进行建模，但现有针对被终止扩散桥的神经求解器需要在训练过程中模拟轨迹。我们提出了 Elysium，一种免仿真的求解器，能够联合学习状态转移与质量变化。利用被终止的布朗运动参考过程及其时间反演，Elysium 从最优端点耦合和条件桥分布出发构建了直接的回归目标，而无需在训练期间模拟轨迹。我们证明了期望轨迹极……

    arXiv:2611.24763v1 Announce Type: cross  Abstract: Reconstructing cellular dynamics from snapshots is challenging because the destructive nature of single-cell measurements makes tracking the same cells over time impossible. Cellular dynamics are inherently stochastic and involve changes in population size through proliferation and apoptosis. Growth-based methods represent population change as a rate on the density, which does not assign exits and entries to individual cells. Unbalanced Schr\"odinger bridges model individual events, but neural solvers for killed diffusion bridges simulate trajectories during training. We introduce Elysium, a simulation-free solver that jointly learns state transition and mass change. Using a killed Brownian reference and its time reversal, Elysium constructs direct regression objectives from an optimal endpoint coupling and conditional bridge distributions, without simulating trajectories during training. We show that the minimizers of the expected tra
    
[^26]: 面向样本高效在线A/B测试的方差最优反事实策略设计

    Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing

    [https://arxiv.org/abs/2611.24757](https://arxiv.org/abs/2611.24757)

    该论文提出一种方差最优的反事实策略设计方法，通过联合优化动作的采样方式，将流量引导至两种策略不一致的动作上，从而显著降低在线A/B测试的方差并提升样本效率。

    

    在线A/B测试是在部署前评估面向用户系统变更（例如新的推荐或排序策略）的标准方法。虽然标准A/B测试的均值差估计量是无偏的，但它通常存在高方差问题。当处理策略与生产策略以相近的概率选择相同的动作时，由此产生的观测数据会增加噪声，却几乎不提供关于两种策略差异的信息。因此，A/B测试往往需要大量流量和较长周期才能检测到较小的处理效应。现有的方差缩减方法主要是在数据收集方案固定之后改进估计量，而近年来的重叠感知设计仅优化两种策略之间的单一流量分配，因而无法将流量引导到两种策略不一致的动作上。在本文中，我们联合优化动作的采样方式……

    arXiv:2611.24757v1 Announce Type: new  Abstract: Online A/B testing is the standard approach for evaluating changes to user-facing systems, such as a new recommendation or ranking policy, before deployment. While the difference-in-means estimator of a standard A/B test is unbiased, it often suffers from high variance. When the treatment and production policies choose the same actions with similar probabilities, the resulting observations can add noise while providing little information about the difference between the policies. Consequently, A/B tests can require large amounts of traffic and long durations to detect small treatment effects. Existing variance-reduction methods mainly improve the estimator after the data-collection scheme has been fixed, and recent overlap-aware designs only optimize a single traffic split between the two policies; thereby, they cannot direct traffic to the actions on which the policies disagree.   In this paper, we jointly optimize how actions are sampl
    
[^27]: 学习任意子图集

    Learning the Anyon Atlas

    [https://arxiv.org/abs/2611.24752](https://arxiv.org/abs/2611.24752)

    通过引入基于逻辑回归的拒绝门算法，将2+1维拓扑量子场论模数据的搜索从22个任意子扩展至42个，识别出150,679个新模数据，据此推翻了Geiko-Moore关于高斯和实性足以保证时间反演对称性的猜想，并展示了算法在秩100处的可扩展性。

    

    我们将之前对2+1维拓扑量子场论模数据的搜索从22个任意子扩展到42个任意子，识别出150,679个新的模数据。为此，我们在算法中增加了三个拒绝门（其中两个基于逻辑回归），用于舍弃不太可能产生模数据的SL(2,Z)表示。我们估计，在经过已知构造的闭包处理后，秩23到42之间的模数据大约已有80%被识别。利用这些数据，我们找到了Geiko和Moore猜想（即高阶高斯和的实性足以保证陈-西蒙斯理论中的时间反演对称性）的反例，并提出了一个改进的判据。我们通过识别秩100处未被任何陈-西蒙斯理论实现的本原模数据，展示了这种门控实现的可扩展性。秩42以内已识别的数据已在https://AnyonAtlas.org上发布。

    arXiv:2611.24752v1 Announce Type: cross  Abstract: We extend our previous search for the modular data of topological quantum field theories in 2+1 dimensions from 22 to 42 anyons, identifying 150,679 new modular data. We do so by adding to our algorithm three rejection gates, two of them based on logistic regression, that discard representations of $SL(2,\mathbb{Z})$ unlikely to yield modular data. We estimate that roughly 80% of the modular data between ranks 23 and 42 has been identified after closing under known constructions. With this data, we find counterexamples to a conjecture of Geiko and Moore that the reality of the higher Gauss sums suffices for time-reversal symmetry in Chern-Simons theories, and propose a refined criterion. We demonstrate the scalability of this gated implementation by identifying prime modular data at rank 100 that is not realized by any Chern-Simons theory. The identified data through rank 42 has been made available at https://AnyonAtlas.org, to which f
    
[^28]: 基于输入选择与数据增强的量产车辆横向速度估计跨车辆迁移性研究

    Cross-Vehicle Transferability of Lateral Velocity Estimation on Production Vehicles through Input Selection and Data Augmentation

    [https://arxiv.org/abs/2611.24747](https://arxiv.org/abs/2611.24747)

    该研究首次在量产车辆上量化了神经网络横向速度估计器跨车辆零样本迁移时误差扩大一个数量级的精度差距，并通过系统比较CAN信号输入确定了迁移最稳健的五信号集，同时结合物理动机的数据增强方法来缩小这一差距。

    

    横向速度对车辆动力学控制具有安全关键意义，但无法通过量产传感器直接测量。基于模型的估计器（如卡尔曼滤波器）已得到成熟应用，而数据驱动的估计器在训练范围内精度更高，但据推测在超出训练范围后会损失精度；将其迁移到结构不同的车辆上鲜有测试。我们记录了三辆量产车的真实道路测试驾驶数据，在一辆紧凑型轿车的量产CAN信号上训练神经网络估计器，并以GNSS辅助惯性参考为基准，在一辆SUV和一辆厢式车上进行零样本评估。在未见过的车辆上，网络的误差比在训练车辆上高出一个数量级，这一差距此前从未在量产车辆上被量化。对来自十个量产CAN信号的输入集进行系统比较，确定了一个迁移最稳健的五信号集。为缩小这一差距，我们引入了三种基于物理的……（原文摘要在此处截断）

    arXiv:2611.24747v1 Announce Type: cross  Abstract: Lateral velocity is safety-critical for vehicle-dynamics control but cannot be measured with production sensors. Model-based estimators such as Kalman filters are established, whereas data-driven estimators are more accurate within the training regime but suspected to lose accuracy beyond it. Their transfer to a structurally different vehicle has rarely been tested. We record real-world test drives of three series-production vehicles, train a neural-network estimator on production CAN signals of a compact car, and evaluate it zero-shot on an SUV and a van against a GNSS-aided inertial reference. On the unseen vehicles, the network's error exceeds that on the training vehicle by an order of magnitude, a gap not quantified on production vehicles before. A systematic comparison of input sets from ten production CAN signals identifies a five-signal set that transfers most robustly. To narrow this gap, we introduce three physically motivate
    
[^29]: 从语音检测帕金森病中的性别差异

    Sex Differences in the Detection of Parkinson's Disease from Speech

    [https://arxiv.org/abs/2611.24741](https://arxiv.org/abs/2611.24741)

    本研究通过在性别平衡的数据集上结合预训练语音模型嵌入与手工声学特征，首次系统揭示了基于语音的帕金森病检测存在性别差异，女性说话者比男性说话者能被更准确地分类。

    

    帕金森病（PD）是世界上第二常见的神经退行性疾病，70%至75%的PD患者在某一阶段会出现言语障碍。由于PD在男性中的发病率高于女性，以往的研究往往依赖于不平衡的数据集，这可能为训练的模型引入性别偏见。本研究使用深度学习和传统机器学习方法，探究了基于语音的帕金森病检测中的性别差异。我们使用三个预训练语音模型的嵌入，结合手工设计的声学特征和不同的分类器，评估了基于语音的PD检测效果。在四个持续发声数据集的性别平衡子集上进行了性别特异性实验，对性别、疾病状态进行了控制，并在可能的情况下控制了年龄。所有录音均在持续发声任务中采集。结果表明，女性说话者的分类准确率高于男性说话者。

    arXiv:2611.24741v1 Announce Type: cross  Abstract: Parkinson's disease (PD) is the second most common neurodegenerative disease in the world, and 70\% to 75\% of PD patients suffer from speech disorders at some stage. Since PD affects men more frequently than women, previous publications tend to rely on unbalanced datasets that potentially introduce sex biases into trained models. This study investigates sex differences in PD detection from speech using deep learning and traditional machine learning approaches. We evaluated speech-based PD detection using embeddings from three pre-trained speech models, combined with handcrafted acoustic features and different classifiers. Sex-specific experiments were conducted on balanced subsets of four sustained phonation datasets, controlling for sex, disease status, and, where possible, age. All recordings were obtained during a sustained phonation task. Results show that female speakers were more accurately classified than male speakers in the l
    
[^30]: 蛋白质-RNA亲和力与突变效应预测中的不均衡收益

    Uneven gains in protein-RNA affinity and mutation-effect prediction

    [https://arxiv.org/abs/2611.24739](https://arxiv.org/abs/2611.24739)

    该研究系统评估了11类蛋白质-RNA亲和力预测方法，发现总体亲和力预测的进步并不均衡地转化为突变效应预测的收益——受控CoPRA降低了绝对亲和力误差、提升了平均效应准确性并在独立外部测试中全面改善，但排序准确性和局部误差的收益参差不齐。

    

    准确的匹配亲和力决定突变效应，但总体亲和力预测的收益可能不均衡地传递到平均效应、排序和局部准确性上。我们使用方法特定的匹配数据集评估了十一类方法。受控的CoPRA将绝对亲和力的平均绝对误差从1.2898降低到1.1431 pKd。在来自20个家族的220对突变中，它产生了更强的响应并提高了平均效应准确性，但排序收益不确定，且平均中心局部误差没有降低。在一项独立的外部测试中（来自三个来源匹配家族的24对定量数据），绝对突变效应误差和常规突变效应误差在所有三个家族中均有所下降。平均中心局部误差从0.4198降至0.3761 pKd，其中两个家族有所改善，一个家族误差更高。误差分解将收益定位于结合景观的均值上。PRA-Pred的特征追踪、固定权重的M2-PRNet输入控制以及匹配监督……（摘要原文在此处截断）

    arXiv:2611.24739v1 Announce Type: new  Abstract: Accurate matched affinities determine mutation effects, but gains in aggregate affinity prediction may transfer unevenly to average effects, ordering and local accuracy. We evaluated eleven method families using method-specific matched populations. Controlled CoPRA reduced absolute-affinity mean absolute error from 1.2898 to 1.1431 pKd. Across 220 mutation pairs from 20 families, it generated stronger responses and improved mean-effect accuracy, with uncertain ranking gains and no decrease in mean centered local error. In an independent external test (24 quantitative pairs from three source-matched families), absolute and ordinary mutation-effect errors fell in all three families. Mean centered local error fell from 0.4198 to 0.3761 pKd, with improvements in two families and higher error in one. Error decomposition localized gains in landscape means. PRA-Pred feature tracing, fixed-weight M2-PRNet input controls and matched supervision i
    
[^31]: 基于基函数学习的近似动态规划

    Approximate Dynamic Programming with Basis Function Learning

    [https://arxiv.org/abs/2611.24737](https://arxiv.org/abs/2611.24737)

    该论文提出BOBF方法，通过反复应用Bellman算子从模型本身学习基函数，保证每次学习都能以可量化幅度改进MDP值函数近似并收敛到最优值函数，其缩减版SBOBF还能在限制基函数数量和计算量的同时保留这些理论保证。

    

    本工作提出了一种在马尔可夫决策过程（MDP）中近似值函数的新方法。我们的方法建立在近似线性规划（ALP）框架之上，其中基函数（BF）扮演着重要角色。传统上，基函数通常是预先定义的、随机生成的或局限于特定结构的，这既限制了所能达到的精度，也没有提供改进给定近似的机制。我们转而从模型本身学习基函数，即通过反复将Bellman算子应用于当前的值估计。我们引入了Bellman算子基函数（BOBF）学习方法，并证明每个学习到的基函数都能以可量化的幅度改进近似，且该方法收敛于最优值函数。其缩减版本（SBOBF）在保留上述保证的同时，限制了基函数的数量，从而限制了每次迭代的计算量。对于大规模……

    arXiv:2611.24737v1 Announce Type: new  Abstract: This work presents a novel approach for approximating the value function in Markov decision processes (MDPs). Our approach is grounded in the approximate linear programming (ALP) framework, wherein basis functions (BFs) play an important role. Traditionally, BFs have been pre-defined, randomly generated, or restricted to specific structures, which both limits the accuracy that can be achieved and leaves no mechanism for improving a given approximation. We instead learn BFs from the model itself, by repeatedly applying the Bellman operator to the current value estimate. We introduce the Bellman operator basis function (BOBF) learning method, and prove that each learned BF improves the approximation by a quantifiable amount and that the method converges to the optimal value function. Its shrunken counterpart (SBOBF) retains this guarantee while capping the number of BFs, and therefore the computational effort per iteration. For large-scale
    
[^32]: 生成模型如何生成：生成式AI全景简明指南

    How Generative Models Generate: A Concise Guide to the Generative AI Landscape

    [https://arxiv.org/abs/2611.24728](https://arxiv.org/abs/2611.24728)

    本文提出以生成为中心的两级分析框架，区分出直接变换/解码、序贯条件构造和迭代更新/输运三种基本生成路径，并考察其背后的建模与学习原理，为理解复杂多样的生成式AI领域提供了简明统一的指南。

    

    快速扩张的生成式AI领域汇聚了在架构、学习目标、数学表示和采样过程上各不相同的方法。一份简洁且经过精选组织的技术指南，可以为理解这个日益复杂的领域提供一个易于入门的起点。本文采用以生成为中心的方法，将模型如何产生新观测与使生成成为可能的建模和学习原理联系起来。我们构建了一个两级视角：首先，区分出三种反复出现的生成路径——直接变换或解码、序贯条件构造以及迭代更新或输运；其次，考察这些路径背后的建模与学习原理。我们通过关键组件、学习目标和采样过程来介绍代表性方法，随后考察了它们更广泛的联系。

    arXiv:2611.24728v1 Announce Type: cross  Abstract: The rapidly expanding generative AI landscape brings together methods that differ in architecture, learning objective, mathematical representation and sampling procedure. A concise, selectively organised technical guide can provide an accessible starting point for developing a coherent understanding of this increasingly complex landscape. This article takes a generation-centred approach, connecting how models produce new observations with the modelling and learning principles that make generation possible. We develop a two-level perspective: first, distinguishing three recurring generation routes-direct transformation or decoding, sequential conditional construction, and iterative updating or transport; and second, examining the modelling and learning principles underlying these routes. Representative approaches are introduced through their key components, learning objectives and sampling procedures. We then examine their broader conne
    
[^33]: 引力波物理与数值相对论中的代理建模与降阶方法：从谱表示到自适应压缩

    Surrogate Modeling and Reduced-Order Methods in Gravitational-Wave Physics and Numerical Relativity: From Spectral Representations to Adaptive Compression

    [https://arxiv.org/abs/2611.24727](https://arxiv.org/abs/2611.24727)

    该综述系统回顾了以本征正交分解、贪婪基选择和经验插值方法为核心的降阶建模与代理技术，展示其如何将高保真引力波波形压缩为可快速评估的紧凑表示，从而突破数值相对论与参数估计中的计算瓶颈。

    

    引力波天文学的迅速发展与高精度数值相对论造成了严峻的计算瓶颈：对爱因斯坦场方程的直接数值积分以及高维贝叶斯参数估计需要数百万次波形评估，仅靠传统求解器无法持续支撑。降阶建模与代理技术已成为弥合这一差距的基础性工具，能够在不牺牲物理精度的前提下，将高保真度解流形压缩为紧凑且可快速评估的表示形式。本综述回顾了基于投影的降阶建模的数学支柱——包括本征正交分解（POD）、贪婪基选择和经验插值方法（EIM）——并通过代理波形族与降阶求积（Reduced-Order Quadratures）追踪了这些方法在引力波数据分析中的实际影响。随后我们考察了……

    arXiv:2611.24727v1 Announce Type: cross  Abstract: The rapid expansion of gravitational-wave astronomy and high-precision numerical relativity has created an acute computational bottleneck: direct numerical integration of the Einstein field equations and high-dimensional Bayesian parameter estimation require millions of waveform evaluations that cannot be sustained by traditional solvers alone. Reduced-order modeling (ROM) and surrogate techniques have emerged as foundational tools to bridge this gap, compressing high-fidelity solution manifolds into compact, rapidly evaluable representations without sacrificing physical accuracy. This perspective reviews the mathematical pillars of projection-based reduced-order modeling---including Proper Orthogonal Decomposition, greedy basis selection, and the Empirical Interpolation Method---and traces their operational impact in gravitational-wave data analysis through surrogate waveform families and Reduced-Order Quadratures. We then examine the
    
[^34]: DRIFT：面向运行时灵活且高效的端到端驾驶的可分离推理

    DRIFT: Detachable Reasoning for Runtime-Flexible and Efficient End-to-End Driving

    [https://arxiv.org/abs/2611.24719](https://arxiv.org/abs/2611.24719)

    DRIFT提出了一种端到端自动驾驶框架，利用小于1.5亿参数的轻量级语言模型实现场景描述与动作解说，并引入可在运行时灵活开关语言推理的规划模块，且不损害闭环驾驶性能，从而兼顾推理能力与计算效率。

    

    将视觉语言模型（VLM）集成到端到端（E2E）自动驾驶中，凭借其场景描述和推理能力展现出了强大的潜力。然而，VLM的计算开销高昂，并且在运行时动态开启和关闭语言推理而不损害闭环性能这一点尚未得到验证，这阻碍了其实际应用。为了解决这些问题，我们提出了DRIFT——一个集成了轻量级语言模型（LM）的端到端自动驾驶框架，用于视觉问答和动作解说。通过将中间驾驶表征高效地投影到语言嵌入空间，我们证明了一个紧凑的语言模型（参数量小于1.5亿）即可实现强大的场景描述和动作解说能力。此外，我们引入了一个运行时灵活的规划模块，该模块在从语言模型蒸馏出的代理推理特征上进行训练，并且可以以语言模型的输出或其他条件作为输入进行调节（摘要在此处被截断）。

    arXiv:2611.24719v1 Announce Type: cross  Abstract: The integration of vision-language models (VLMs) into end-to-end (E2E) autonomous driving has demonstrated strong potential by providing scene description and reasoning capabilities. However, VLMs are computationally expensive, and switching language reasoning on and off at runtime without hurting closed-loop performance has not been demonstrated, which hinders practical adoption. To address these issues, we propose DRIFT, an E2E autonomous driving framework that integrates a lightweight language model (LM) for visual question answering and action commentary. By efficiently projecting intermediate driving representations into the language embedding space, we demonstrate that a compact LM (<150M parameters) can achieve strong scene description and action commentary. Furthermore, we introduce a runtime-flexible planning module that is trained on surrogate reasoning features distilled from the LM and can be conditioned on either the LM or
    
[^35]: Meta-LSQ：一种用于少样本元学习的闭式Sobolev内环求解器

    Meta-LSQ: A Closed-Form Sobolev Inner Solver for Few-Shot Meta-Learning

    [https://arxiv.org/abs/2611.24718](https://arxiv.org/abs/2611.24718)

    Meta-LSQ用固定随机傅里叶基上的一次闭式正则化最小二乘求解替代MAML的内环梯度下降，仅元学习三个标量即可获得任意阶精确导数，在少样本正弦回归上以约1/36的训练时间取得比一阶MAML低5.5倍的误差。

    

    基于梯度的元学习方法（如模型无关元学习MAML）通过一个需要被微分求导的内环梯度下降过程来适应新任务，并且要对每一个网络权重进行元学习。Meta-LSQ则改为在固定的随机傅里叶基上通过一次正则化最小二乘求解来拟合每个任务，并且只元学习三个标量：带宽、岭回归参数和导数权重。由于正弦函数的导数仍是正弦函数，同样的求解方法还能接收导数观测，且拟合出的模型具有任意阶的精确闭式导数。在10样本正弦回归任务上，Meta-LSQ仅需14秒训练，最终误差比训练时长长36倍的一阶MAML基线低5.5倍；高斯过程对照组实验表明，这一提升大部分来自跨任务共享同一个核函数。在DESY的ARES装置上结合模型导数进行的模拟束流调谐任务中，它的精度略高于……（原文摘要在此处截断）

    arXiv:2611.24718v1 Announce Type: new  Abstract: Gradient-based meta-learners such as model-agnostic meta-learning (MAML) adapt to a new task with an inner loop of gradient descent that has to be differentiated through, and they meta-learn every network weight. Meta-LSQ instead fits each task with one regularised least-squares solve over a fixed random Fourier basis and meta-learns only three scalars, a bandwidth, a ridge and a derivative weight. Since the derivative of a sinusoid is again a sinusoid, the same solve can also take derivative observations, and the fitted model has exact closed-form derivatives of any order. On 10-shot sinusoid regression Meta-LSQ trains in 14 s and ends with a 5.5 times lower error than a first-order MAML baseline that trains 36 times longer, and a Gaussian-process control suggests that most of this gain comes from sharing one kernel across tasks. In simulated beam tuning at DESY's ARES facility with model derivatives it is slightly more accurate than an
    
[^36]: Openhagen Worlds：用于测量与训练科学发现智能体的校准环境

    Openhagen Worlds: Calibrated Environments for Measuring and Training Scientific Discovery Agents

    [https://arxiv.org/abs/2611.24711](https://arxiv.org/abs/2611.24711)

    Openhagen Worlds是一套包含27个程序化生成环境、178个经严格校准难度级别的科学发现基准套件，通过明确的评分规则和针对神谕策略、非演绎策略及捷径策略的成功率校准，能够真正测量并训练智能体选择信息性实验和从结果中推理的能力，排除记忆回溯与捷径策略的干扰。

    

    从事科学研究的智能体必须选择有信息量的实验并根据结果进行推理，但大多数科学发现基准无法将这种能力与记忆回溯、未明确说明的评分规则或捷径策略区分开来。我们提出Openhagen Worlds，一套包含27个程序化生成环境的基准套件（部分环境具有有限真值空间），涵盖从经典的实验推理谜题到基于模拟器的物理和化学场景。在这些环境中，智能体在预算限制下开展实验并提交答案，由确定性验证器对照隐藏真值进行检查。每个观测都标明了完整的评分规则，178个难度级别中的每一级都经过校准：只有在64个开发实例上，脚本化神谕策略成功率至少达到0.75、最强的合理非演绎策略成功率至多0.25、且每个受控捷径策略成功率至多0.35时，该级别才会被采纳。92个核心级别涵盖全部27个环境族，其规模适合训练小型模型；86个前沿级别……（摘要原文在此处截断）

    arXiv:2611.24711v1 Announce Type: cross  Abstract: Agents that do science must choose informative experiments and reason from their results, but most discovery benchmarks cannot separate that skill from recall, unstated scoring rules or shortcut policies. We introduce Openhagen Worlds, a suite of 27 procedurally generated environments (some with finite truth spaces), from classic puzzles of experimental reasoning to simulator-grounded physics and chemistry, in which an agent runs experiments under a budget and commits an answer that a deterministic verifier checks against a hidden truth. Every observation states the complete scoring rule, and each of the 178 difficulty levels is calibrated: it is admitted only if, on 64 development instances, a scripted oracle succeeds on at least 0.75, the strongest plausible non-deductive policy on at most 0.25 and each gated shortcut on at most 0.35. The 92 core levels span all 27 families and are sized to train small models; the 86 frontier rungs e
    
[^37]: 持续预训练何时有效？一项关于魁北克法语方言适应的对照研究

    When Does Continual Pre-Training Help? A Controlled Study of Dialectal Adaptation for Qu\'ebec French

    [https://arxiv.org/abs/2611.24699](https://arxiv.org/abs/2611.24699)

    该研究通过对九个大语言模型的预算匹配对照实验发现，持续预训练对魁北克法语方言适应的收益取决于起始模型对目标方言的既有能力，其增益主要由语料构成（尤其是文学文本）而非数据量驱动，且主要提升表达理解而非语法能力。

    

    持续预训练（CPT）被广泛用于将预训练语言模型适配到方言和非标准语言变体，但这种额外的训练何时才是必要的、其收益由什么驱动、以及总体评估分数是否足以刻画其效果，目前仍不清楚。本文针对魁北克法语语言理解任务，对涵盖不同模型家族与规模的九个大语言模型开展了一项持续预训练的对照研究。采用每个实验3000万词元的预算匹配协议，我们的结果表明：持续预训练的收益在很大程度上取决于起始模型对目标语言变体已有的掌握程度。当持续预训练带来收益时，其提升更多源于语料构成而非数据量，其中文学文本贡献了大部分增益。对魁北克法语与标准法语的成对评估进一步表明，持续预训练主要改善的是表达理解能力而非语法能力，并且语料库……（原文摘要在此处截断）

    arXiv:2611.24699v1 Announce Type: cross  Abstract: Continual pre-training (CPT) is widely used to adapt pretrained language models to dialectal and non-standard language varieties, yet it remains unclear when this additional training is necessary, what drives its benefits, and whether aggregate evaluation scores adequately capture its effects. In this paper, we present a controlled study of CPT for Qu\'ebec French language understanding across nine LLMs spanning different families and scales. Using a budget-matched protocol of 30M tokens per experiment, our results show that the benefit of CPT depends strongly on the starting model's existing competence in the target variety. When CPT is beneficial, the improvements are driven more by corpus composition than by data volume, with literary text accounting for most of the gain. Paired Qu\'ebec and standard French evaluations further show that CPT primarily improves expression comprehension rather than grammatical competence, and that corp
    
[^38]: 在联邦学习过程中从其他客户端窃取精确训练数据

    Stealing Exact Training Data from Other Clients during Federated Learning

    [https://arxiv.org/abs/2611.24689](https://arxiv.org/abs/2611.24689)

    该论文首次提出一种迭代式攻击流水线，通过逐轮从聚合梯度中减去已还原句子的梯度贡献来解开FedAvg的多客户端梯度纠缠，实现了联邦学习中100%精度的文本精确还原，最高可精确还原80.2%的客户端句子。

    

    恶意攻击者可以利用梯度反演攻击，借助联邦学习中某一轮的模型更新来逆向还原隐私数据。尽管针对文本的梯度反演技术已有诸多进展，但迄今为止，恶意客户端实现对文本的精确还原仍不可行，因为FedAvg将多个客户端的梯度贡献纠缠在一起，掩盖了任何单个句子的信号。我们提出了一种新颖的攻击流水线，通过迭代循环逐步解开这种纠缠：在每一轮迭代中，将已精确还原的句子的梯度贡献从聚合梯度中减去，从而使剩余的句子暴露出来以便在下一轮迭代中被还原。我们首次实现了100%精度的句子精确还原，并且在包含多达模型隐藏维度4倍令牌数的聚合数据上，精确还原了高达80.2%的客户端句子——这是现有攻击无法企及的领域。

    arXiv:2611.24689v1 Announce Type: new  Abstract: Malicious adversaries can employ gradient inversion attacks to reverse-engineer private data using the model updates of a round in Federated Learning. While there have been developments in gradient inversion for text, exact text recovery by a malicious client has so far remained infeasible, as FedAvg entangles the gradient contributions of many clients and obscures the signal of any individual sentence. We introduce a novel attack pipeline that unwinds this entanglement progressively through an iterative loop, in which the gradient contributions of sentences recovered exactly in one round are subtracted from the aggregate, exposing the remaining sentences for recovery in the next iteration. We show, for the first time, exact sentence recovery, with 100% precision, and on aggregates containing up to $4\times$ more tokens than the model's hidden dimension, we recover up to 80.2% of client sentences exactly, a regime where existing attacks 
    
[^39]: 神经网络中量化误差的 Zonotope 分析

    Zonotopic Analysis of Quantization Error in Neural Networks

    [https://arxiv.org/abs/2611.24648](https://arxiv.org/abs/2611.24648)

    本文提出基于 zonotope 与约束 zonotope 的抽象解释验证框架，通过统一的舍入-截断抽象转换器精确界定量化神经网络的量化误差，在 ACAS Xu 基准上较现有方法将误差界宽度收窄约 70-85%、运行时间缩短 65-80%。

    

    量化神经网络通过降低数值精度来获得计算效率，但由于离散的舍入和截断操作，对其量化误差进行界定仍然是一个难题。我们提出了一个基于 zonotope 的抽象解释框架，用于量化 ReLU 前馈神经网络的验证，并引入了一个统一的舍入-截断抽象转换器，可同时联合处理这两种操作。我们进一步将该框架扩展到约束 zonotope，通过添加线性约束来收紧误差界，并推导出可靠（sound）的量化误差表示。我们证明了舍入-截断抽象转换器的可靠性与仿射带最优性，以及其完整约束形式的凸包最优性。在 ACAS Xu 基准测试中，在所测试的最小两个扰动半径下，与 QEBVerif 的符号 DRA 传播方法相比，我们的约束 zonotope 方法将平均误差界宽度降低了约 70-85%，并将运行时间缩短了 65-80%。

    arXiv:2611.24648v1 Announce Type: new  Abstract: Quantized neural networks achieve computational efficiency by reducing precision, but bounding their quantization error remains a challenge due to discrete rounding and clamping operations. We present a zonotope-based abstract interpretation framework for quantized ReLU feed-forward network verification, introducing a unified round-clamp abstract transformer that jointly handles both operations. We extend this framework to constrained zonotopes, which add linear constraints to tighten bounds, and derive sound quantization error representations. We establish soundness and affine-strip optimality of the round-clamp abstract transformer, as well as convex-hull optimality of its complete constrained formulation. On the ACAS Xu benchmarks, our constrained zonotope method reduces average error-bound widths by approximately 70-85% and runtimes by 65-80% compared to QEBVerif's symbolic DRA propagation at the two smallest tested perturbation radi
    
[^40]: 等步长、不等曲率：Muon 更新在哪里出了问题

    Equal Steps, Unequal Curvature: Where Muon's Update Goes Wrong

    [https://arxiv.org/abs/2611.24644](https://arxiv.org/abs/2611.24644)

    该论文通过测量曲率发现 Muon 优化器在其谱的高曲率顶端奇异方向上步长过大，并利用二阶最优乘子进行诊断与干预以改进其更新。

    

    Muon 沿其动量的极因子更新每个隐藏权重矩阵，即谱范数下的最速下降方向。这一选择在不参考损失函数的情况下固定了两种分配方式：动量的每个奇异方向获得相同的步长，而在非方阵中，每个神经元的步长由其杠杆分数决定。这些分配是否合理是一个关于曲率的问题，而最速下降论证无法回答这一点；我们对此进行了实际测量。利用留出梯度和 Hessian 向量积，我们按奇异值频带和神经元组对 Muon 的步长进行分解，并计算每组的二阶最优乘子。在 FineWeb-Edu 上训练的 1.24 亿参数解码器中，乘子从顶部奇异方向到尾部单调递增：Muon 在其谱的高曲率顶端步长过大。我们在 Lean 中证明了这种诊断何时可以在 Muon 自身的更新范数下授权进行干预。在测量的曲率上，该关键的…（摘要被截断）

    arXiv:2611.24644v1 Announce Type: new  Abstract: Muon updates each hidden weight matrix along the polar factor of its momentum, the steepest-descent direction under the spectral norm. That choice fixes two allocations without consulting the loss: every singular direction of the momentum receives the same step, and in a non-square matrix each neuron's step is set by its leverage score. Whether they are good is a question about curvature, which steepest-descent arguments do not answer; we measure it. Using held-out gradients and Hessian-vector products, we decompose Muon's step by singular-value band and by neuron group and compute each group's second-order optimal multiplier. In a 124M-parameter decoder trained on FineWeb-Edu, the multipliers rise monotonically from the top singular directions to the tail: Muon over-steps the high-curvature top of its spectrum. We prove in Lean when such a diagnostic licenses an intervention at Muon's own update norm. On the measured curvature, the crit
    
[^41]: 选择正确的表征：面向多模态材料语言模型的任务感知融合

    Choosing the Right Representations: Task-Aware Fusion for Multimodal Materials Language Models

    [https://arxiv.org/abs/2611.24641](https://arxiv.org/abs/2611.24641)

    该论文提出了轻量级的任务条件化注意力融合桥接模块，能够根据目标性质自适应地从多个结构编码器的不同深度选择并融合互补的表征，从而提升多模态材料语言模型的性质预测能力。

    

    面向材料科学的多模态语言模型依赖于将三维结构表征有效融入模型的机制。MatterChat 通过在预训练结构编码器与大语言模型（LLM）之间使用一个大型桥接模块，并结合一个独立的对比对齐阶段来应对这一挑战。我们则提出另一个问题：该桥接模块能否根据目标性质自适应地选择并组合来自多个结构编码器的表征？我们发现，监督式与生成式的材料编码器在性质预测中提供了互补的信息。我们还发现，对于 Zatom-1，这些信息并不集中在最后一层；相反，最具信息量的表征出现在不同的网络深度，具体取决于目标性质。这一发现启发了任务条件化注意力融合的设计，这是一种轻量级的基于注意力的桥接方法，它使用任务条件化的潜在查询将 CHGNet 特征与中间层表征相结合……

    arXiv:2611.24641v1 Announce Type: cross  Abstract: Multimodal language models for materials science rely on effective mechanisms for incorporating three-dimensional structural representations. MatterChat addresses this challenge using a large bridge module between a pretrained structural encoder and the LLM, together with a separate contrastive alignment stage. We instead ask whether this bridge can adaptively select and combine representations from multiple structural encoders according to the target property. We find that supervised and generative material encoders provide complementary information for property prediction. We found that, for Zatom-1, this information is not concentrated in the final layer. Instead, the most informative representations occur at different depths depending on the target property. This motivates Task-Conditioned Attention Fusion (TCAF), a lightweight attention-based bridge that uses task-conditioned latent queries to combine CHGNet features with intermed
    
[^42]: Clover-35B-A3B-Preview 技术报告

    Clover-35B-A3B-Preview Technical Report

    [https://arxiv.org/abs/2611.24636](https://arxiv.org/abs/2611.24636)

    Clover 是一个总参数 350 亿、激活参数仅 30 亿的混合专家扩散语言模型，其创新在于将门控 Delta 网络的循环记忆与块内双向注意力结合以适配块扩散，在智能体基准上全面领先（SWE-bench Verified 达 73.6 分），并在单块 GB300 GPU 上实现 680 tokens/s 的生成速度。

    

    我们介绍 Clover-35B-A3B-Preview（简称 Clover），这是一个混合专家扩散语言模型，总参数量为 350 亿，激活参数量为 30 亿，支持思考模式生成。Clover 将门控 Delta 网络（Gated Delta Networks）适配到块扩散范式中，将先前已生成上下文的循环记忆与当前块内的双向 softmax 注意力相结合。历史记忆读取与当前块内的 token 在共享 softmax 中进行竞争，并配备可学习的块大小校准以及一个独立的、零初始化的右上下文卷积分支。Clover 在全部七项被评估的智能体指标中均处于领先地位，其中包括 SWE-bench Verified 上的 73.6 分和 Terminal-Bench 2.1 上的 51.7 分。在内部智能体工作负载上，使用单块 NVIDIA GB300 GPU，在 BF16 精度和批量大小为 1 的条件下，其报告的生成速度达到 680 tokens/s。我们围绕模型质量、草拟与验证成本以及受限内存条件下的部署来组织评估。

    arXiv:2611.24636v1 Announce Type: cross  Abstract: We introduce Clover-35B-A3B-Preview (Clover), a mixture-of-experts diffusion language model with 35 billion total parameters and 3 billion activated parameters, supporting thinking-mode generation. Clover adapts Gated Delta Networks to block diffusion by combining recurrent memory of previously generated context with bidirectional softmax attention inside the current block. Historical memory readout competes with current-block tokens in a shared softmax, with learnable block-size calibration and a separate, zero-initialized right-context convolution branch. Clover leads the seven evaluated agent metrics, including 73.6 on SWE-bench Verified and 51.7 on Terminal-Bench 2.1. On internal agent workloads, it reaches a reported generation speed of 680 tokens/s using one NVIDIA GB300 GPU at BF16 precision and batch size one. We organize the evaluation around model quality, drafting and verification costs, and deployment under constrained memo
    
[^43]: 将人类大脑组织结构分析迁移至语言模型

    Transferring an analysis of human brain organization to language models

    [https://arxiv.org/abs/2611.24626](https://arxiv.org/abs/2611.24626)

    该研究将刻画人类皮层组织的扩散映射方法应用于语言模型，发现推理训练使模型在一段连续层跨度内发生局部重组，将该区域的权重嫁接到其他模型中可提升其数学推理能力。

    

    在人类大脑皮层发育过程中，功能系统逐渐分化。我们提出疑问：语言模型推理能力的提升是否涉及类似的区域性重组。我们将通常用于刻画皮层组织结构的扩散映射嵌入方法，应用于来自三个模型家族的相互匹配的基座模型与经推理训练的模型，发现整体梯度轮廓大体保持不变，但在一段连续的层跨度内存在局部变化。将这些跨度区域的权重嫁接到指令微调的宿主模型中，平均而言提升了数学推理能力，且增益随区域重叠度的增加而上升，同时比在其他位置进行嫁接保留了更多先前已成功解决的问题。在整个训练过程中，该跨度在性能早期上升阶段逐渐形成，来自中间训练阶段的嫁接同样有效。嫁接使大多数宿主模型的行为部分向其供体模型偏移，有时甚至在供体模型未能解决的问题上取得改进；同一分析还在一个（原文在此处截断）……

    arXiv:2611.24626v1 Announce Type: cross  Abstract: During human cortical development, functional systems become increasingly differentiated. We asked whether improved reasoning in language models involves a comparable regional reorganization. Applying diffusion-map embedding, used to characterize cortical organization, to matched base and reasoning-trained models from three families, we found broadly preserved gradient profiles with localized changes within a contiguous layer span. Grafting weights from these spans into instruction-tuned hosts improved mathematical reasoning on average, with gains rising with region overlap, while retaining more previously solved problems than grafts elsewhere. Across training, the span formed gradually during the early rise in performance, and grafts from intermediate stages helped. Grafts shifted most hosts partly towards their donors' behaviour and sometimes improved on problems the donors failed; the same analysis located a transferable span in a m
    
[^44]: 评估位置编码方法在物种分布建模中的有效性

    Evaluating the Effectiveness of Location Encoding Methods for Species Distribution Modelling

    [https://arxiv.org/abs/2611.24622](https://arxiv.org/abs/2611.24622)

    该论文提出了一个标准化评估框架，首次系统量化了不同位置编码方法（如气候变量、学习式编码、地球嵌入模型等）在局部和全球尺度物种分布范围估计中的有效性。

    

    理解物种的空间分布是生态学中的一个关键问题。目前已提出多种解决方案，涵盖统计方法、传统机器学习方法以及基于深度学习的方法。这些方法以目标物种的稀疏观测数据作为输入，旨在生成指示物种空间分布范围的密集空间输出。与统计模型的选择相互独立的另一个问题是：什么样的输入特征最具信息量。得益于近期的发展，可选方案的数量已大大扩展，涵盖不同的气候变量、基于学习的编码、地球嵌入模型或原始变换的地理坐标。在本工作中，我们引入了一个标准化评估框架，并在局部和全球尺度上量化了不同位置编码方法在物种分布范围估计中的有效性。

    arXiv:2611.24622v1 Announce Type: new  Abstract: Understanding the spatial distribution of species is a key question in ecology. A variety of solutions, from statistical, traditional machine learning, through to deep learning-based methods have been proposed. These methods take sparse observations for a species of interest as input and aim to generate dense spatial outputs indicating the spatial range of the species. Orthogonal to the choice of statistical model used is the question of what input features are most informative. Thanks to recent developments, the number of options has greatly expanded, with choices spanning different climate variables, learning-based encodings, earth embedding models, or raw transformed geographic coordinates. In this work, we introduce a standardized evaluation framework and quantify the effectiveness of different location encodings in the context of species range estimation, both at the local and global scale.
    
[^45]: GateWatch：基于布尔电路的低开销大语言模型监控

    GateWatch: Low-Overhead LLM Monitoring with Boolean Circuits

    [https://arxiv.org/abs/2611.24619](https://arxiv.org/abs/2611.24619)

    GateWatch通过将决策规则表达为少量二值化SAE特征上的显式布尔逻辑电路，实现了紧凑、可解释且低开销的大语言模型行为监控。

    

    随着大语言模型（LLM）和基于LLM的智能体日益广泛地被部署，监控其行为对于检测不安全或非预期的动作变得至关重要。激活探针提供了一种高效的方法，但现有方法在准确性、可解释性和计算开销之间存在权衡。基于原始激活值的探针缺乏可解释的输入，而稀疏自编码器（SAE）探针虽然使用可解释的特征，但通常依赖于加性分类器和大型SAE编码器，这限制了可解释性并增加了内存开销。我们提出了GateWatch，这是一种紧凑且可解释的分类器，它将整个决策规则表达为对少量二值化SAE特征的显式布尔逻辑。我们探索了在给定门预算约束下构建这些电路的三种方法。为了高效部署，GateWatchServe仅计算选定的SAE特征，并支持将其二进制值跨层流式传输（原文摘要在此处被截断）。

    arXiv:2611.24619v1 Announce Type: new  Abstract: As large language models (LLMs) and LLM-based agents are increasingly deployed, monitoring their behavior becomes important for detecting unsafe or unintended actions. Activation probes offer an efficient approach, but existing methods face trade-offs between accuracy, interpretability, and computational overhead. Probes on raw activations lack interpretable inputs, whereas sparse autoencoder (SAE) probes use interpretable features but often rely on additive classifiers and large SAE encoders, limiting interpretability and increasing memory overhead. We introduce GateWatch, a compact, interpretable classifier that expresses its entire decision rule as an explicit Boolean logic over a small number of binarized SAE features. We explore three approaches to constructing these circuits under a specified gate budget. For efficient deployment, GateWatchServe computes only selected SAE features, enables streaming of their binary values across la
    
[^46]: 扩散模型是稀疏恢复的可证明高效先验

    Diffusion Models are Provably Efficient Priors for Sparse Recovery

    [https://arxiv.org/abs/2611.24602](https://arxiv.org/abs/2611.24602)

    本文针对压缩感知中的稀疏恢复问题，首次证明基于预训练扩散模型先验的似然引导退火朗之万采样器能够在多项式时间内以近乎最优的样本复杂度可证明地恢复稀疏信号，为扩散模型作为逆问题先验提供了高效的端到端理论保证。

    

    扩散模型凭借其捕捉复杂数据分布并从中生成样本的能力，彻底改变了生成建模领域。这一成功激发了人们浓厚的兴趣，即将预训练的扩散模型用作成像与感知中高度病态逆问题的表达性先验。尽管经验性能出色，但这种方法的理论理解仍然有限：现有保证往往是渐近性的，或在最坏情况下需要难以承受的计算成本。在本文中，我们通过研究从少量线性测量中进行稀疏恢复这一压缩感知中的经典问题来填补这一空白。我们证明，给定一个准确刻画促进稀疏性的伯努利-高斯先验的扩散模型，一个由似然引导的退火朗之万采样器能够在多项式时间内、以近乎最优的样本复杂度，可证明地恢复稀疏信号，且适用于广泛的随机和确定性传感矩阵族。

    arXiv:2611.24602v1 Announce Type: cross  Abstract: Diffusion models have transformed generative modeling through their ability to capture and generate samples from complex data distributions. This success has spurred substantial interest in using pretrained diffusion models as expressive priors for highly ill-posed inverse problems in imaging and sensing. Despite strong empirical performance, theoretical understanding of this approach remains limited: existing guarantees are often asymptotic or entail prohibitive computational costs in the worst case. In this paper, we address this gap by studying sparse recovery from a small number of linear measurements, a classical problem in compressed sensing. We show that, given an accurate diffusion model capturing a sparsity-promoting Bernoulli-Gaussian prior, a likelihood-guided annealed Langevin sampler provably recovers the sparse signal in polynomial time with near-optimal sample complexity for a broad family of random and deterministic sen
    
[^47]: 面向植物光谱学的形态-光谱投影分析

    Morpho-Spectral Projection Analysis (MSPA) for Plant Spectroscopy

    [https://arxiv.org/abs/2611.24599](https://arxiv.org/abs/2611.24599)

    该论文提出了一种监督式降维方法MSPA（形态-光谱投影分析），突破了传统降维方法将相邻波长视为无序列、忽略光谱序列结构的局限，通过吸收特征运动学和局部多阶奇异模态等互补模块，将植物连续光谱投影到紧凑的形态-光谱子空间以支持性状预测建模。

    

    植物反射光谱在可见光、近红外及短波红外波段范围内，将叶片的生化和解剖性状编码为具有局部可微分结构的连续光谱特征；然而，极端的波段间共线性以及不利的样本-特征比例给预测建模带来了挑战。光谱学中常规采用的降维方法具有列置换不变性：它们将相邻波长视为无序的表格列，从而丢弃了吸收特征的序列微分几何结构。为此，我们提出形态-光谱投影分析（MSPA），这是一种监督式降维算子，通过三个互补的模块将连续光谱投影到紧凑的形态-光谱子空间中：(i) 加性连续统去除后的吸收深度、吸收面积以及红边拐点运动学特征；(ii) 局部多阶奇异特征模态（0阶、1阶和2阶）……（摘要原文在此处截断）

    arXiv:2611.24599v1 Announce Type: new  Abstract: Plant reflectance spectroscopy across the visible, near-infrared, and shortwave infrared domain encodes foliar biochemical and anatomical traits into contiguous spectral signatures with locally differentiable structure; however, extreme inter-band collinearity and unfavorable sample-to-feature ratios challenge predictive modeling. Conventional dimensionality reduction methods routinely adopted in spectroscopy, are column-permutation invariant: they treat adjacent wavelengths as unordered tabular columns and discard the sequential differential geometry of absorption features. Here, we propose Morpho-Spectral Projection Analysis (MSPA), a supervised dimensionality reduction operator that projects contiguous spectra into a compact morpho-spectral subspace through three complementary blocks: (i) additive continuum-removed absorption depth, area, and red-edge inflection kinematics; (ii) local multi-order singular eigenmodes (0th, 1st, and 2nd
    
[^48]: 从活动识别到漂移检测：基于证据不确定性的变分Wi-Fi感知

    From Activity Recognition to Shift Detection: Variational Wi-Fi Sensing with Evidential Uncertainty

    [https://arxiv.org/abs/2611.24596](https://arxiv.org/abs/2611.24596)

    该论文提出Wi-HAR多天线变分自编码器架构，从原始Wi-Fi 6 CSI中学习紧凑分布，在十二类活动识别上达到96.6%的中位数准确率（显著优于单天线模型的70.8–81.5%），并复用同一潜在骨干实现部署条件漂移的检测。

    

    人体运动会扰动无线信道，这使商用Wi-Fi设备成为无设备活动识别的实用平台。然而，信道状态信息（CSI）也强烈依赖于具体的人、接收器位置、录制会话以及周围环境，因此当部署条件偏离训练时的条件后，分类器仍可能继续输出活动标签。我们提出了Wi-HAR，这是一种多天线架构，其中四个变分自编码器从原始Wi-Fi 6 CSI幅度中学习紧凑分布，并由延迟融合分类器结合它们的均值与对数方差。在一个公开基准的十二个活动上，Wi-HAR达到了96.6%的中位数识别准确率，而经过独立优化的单天线模型仅为70.8–81.5%，这表明多天线为同一活动提供了互补的观测视角。随后，我们利用相同的潜在骨干网络，在不使用（标签）的情况下检测部署条件的漂移。

    arXiv:2611.24596v1 Announce Type: new  Abstract: Human motion perturbs wireless channels, making commodity Wi-Fi devices a practical platform for device-free activity recognition. Yet Channel State Information (CSI) also depends strongly on the person, receiver position, recording session, and surrounding environment, so a classifier can continue producing activity labels after its deployment conditions have moved away from those seen during training. We propose Wi-HAR, a multi-antenna architecture in which four Variational Auto-Encoders learn compact distributions from raw Wi-Fi 6 CSI magnitudes and a delayed-fusion classifier combines their means and log-variances. On twelve activities from a public benchmark, Wi-HAR reaches 96.6% median recognition accuracy, compared with 70.8-81.5% for independently optimized single-antenna models, showing that the antennas provide complementary views of the same activity. We then use the same latent backbone to detect deployment shifts without usi
    
[^49]: AGAMOTTO：一个面向时间紧迫条件下战术行动方案评估与生成的AI赋能框架

    AGAMOTTO: An AI-Enabled Framework for Time-Critical Tactical Course-of-Action Evaluation and Generation

    [https://arxiv.org/abs/2611.24568](https://arxiv.org/abs/2611.24568)

    AGAMOTTO是一个AI赋能框架，通过集成仿真能力使指挥官能够在时间紧迫的实时作战中动态评估和生成战术行动方案，初步结果已验证该系统的可行性。

    

    当前的指挥与控制（C2）平台缺乏集成的仿真能力，无法让指挥官在实时作战过程中动态探索行动方案（COA）。本文介绍了AGAMOTTO（军事行动、战术威胁与结果的高级指导与分析）开发过程中的进行中研究及贡献，该系统是整体NGC2项目的一部分，该项目致力于推进陆军C2系统的现代化，例如战术突击工具包（TAK）。本文概述了当前的设计、初步发现和剩余的研究任务，重点阐述了已完成的工作将如何在未来时间受限的环境中支持战术决策。虽然完整系统仍在开发中，但初步开发和结果已经证明了AGAMOTTO的可行性。早期实验既揭示了部署AI赋能工具的潜在益处，也指出了实际应用中的限制。

    arXiv:2611.24568v1 Announce Type: new  Abstract: Current command and control (C2) platforms lack integrated simulation capabilities that would allow commanders to dynamically explore courses of action (COAs) during live operations. This paper presents ongoing research and contributions to the development of AGAMOTTO (Advanced Guidance and Analytics for Military Operations, Tactical Threats, and Outcomes) as part of the overarching NGC2 program, which advances the Army's efforts to modernize C2 systems, such as the Tactical Assault Kit (TAK). The paper outlines the current design, preliminary findings, and the remaining research tasks, with an emphasis on how the completed work will support tactical decision-making in future time-constrained environments. While the full system is still under development, initial development and results demonstrate the feasibility of AGAMOTTO. Early experiments highlight both the potential benefits and practical constraints of deploying AI-enabled tools 
    
[^50]: 驯服插值机制下随机梯度下降的尾部

    Taming the Tails of Stochastic Gradient Descent in the Interpolation Regime

    [https://arxiv.org/abs/2611.24562](https://arxiv.org/abs/2611.24562)

    本文首次在插值机制下为随机梯度下降建立了高概率收敛保证，在凸情形证明了 log(β⁻¹)·n⁻¹ 的快速率，在强凸情形给出了一族基于矩的界及其匹配下界，证明指数速率的劣化本质上是不可避免的。

    

    在本文中，我们研究了光滑插值机制下的随机梯度下降，即总体目标函数与样本损失共享同一个最小值点。与以往主要集中于期望意义下分析的工作不同，我们在相同的假设条件下获得了高概率收敛保证。在凸函数情形下，我们证明了经过 n 次迭代后可达到 log(β⁻¹)·n⁻¹ 阶的快速收敛速率，且失败概率至多为 β。在强凸情形下，我们推导出一族基于矩的界，其阶为 β⁻ᵖ·exp(-O(p·n))，其中 p ∈ (0,1]，并给出了与之匹配的下界，表明指数收敛速率的相应劣化在本质上是不可避免的。在这两种情形下，我们的理论结果统一适用于一大类步长调度策略。基于这一分析，我们将结果进一步扩展到低噪声机制以及非凸（摘要在此处截断）

    arXiv:2611.24562v1 Announce Type: cross  Abstract: In this paper, we study stochastic gradient descent in the smooth interpolation regime, where the population objective shares a common minimizer with the sample losses. In contrast to prior work, which has largely focused on in-expectation analysis, we obtain high-probability convergence guarantees under the same assumptions. In the convex case, we prove a fast rate of order $\log(\beta^{-1}) \cdot n^{-1}$ after $n$ iterations, with failure probability at most $\beta$. In the strongly convex case, we derive a family of moment-based bounds of order $\beta^{-p} \cdot \exp(-\mathcal{O}(p \cdot n))$ for $p \in (0,1]$, and complement them with a matching lower bound showing that the corresponding deterioration in the exponential rate is essentially unavoidable. In both settings, our theoretical results are unified across a broad class of step-size schedules. Building on this analysis, we extend our results to a low-noise regime and to nonco
    
[^51]: 神经网络决策边界的几何学

    The geometry of neural network decision boundaries

    [https://arxiv.org/abs/2611.24557](https://arxiv.org/abs/2611.24557)

    该论文提出了一种直接在高维输入空间中可视化和测量神经网络决策边界几何形状的方法，揭示了决策边界在大连续子空间内接近输入样本，且训练方法对边界几何的影响显著大于网络架构。

    

    人工神经网络（ANN）由人们充分理解的数学运算构建而成，但其复杂的特性给直观理解其决策过程带来了重大挑战。本研究提出了一种方法，可以直接在高维输入空间中可视化和测量ANN决策边界的几何形状，避免了传统基于投影技术的失真问题。我们分析了在不同输入、架构和训练方法下的边界曲率和距离。随着子空间规模的增大，边界距离平滑且缓慢地增长，同时保持一致的曲率，这表明决策边界在大而连续的子空间内接近输入样本。我们还量化了对抗训练后数据集样本附近边界距离和曲率的降低程度。研究还发现，训练方法对边界距离和曲率的影响显著强于架构的影响。

    arXiv:2611.24557v1 Announce Type: new  Abstract: Artificial neural networks (ANNs) are built using well-understood mathematical operations, yet their complex nature poses significant challenges to intuitively understanding their decision-making processes. This work introduces a method for visualizing and measuring ANN decision boundary geometry directly in the high-dimensional input space, avoiding distortions found with traditional projection-based techniques. We analyze boundary curvatures and distances in diverse inputs, architectures, and training methods. Boundary distances grow smoothly and slowly with consistent curvature as the size of the subspace increases, suggesting that decision boundaries are close to input samples within large, continuous subspaces. We also quantify how much boundary distances and curvatures reduce near dataset samples after adversarial training. Training methodology has a significantly stronger impact on boundary distance and curvature than architecture
    
[^52]: 通过高维超立方体上均匀测度的量化构造稀疏设计

    Sparse designs via quantization of the uniform measure on high-dimensional hypercubes

    [https://arxiv.org/abs/2611.24507](https://arxiv.org/abs/2611.24507)

    本文提出用超立方体上均匀测度的量化误差替代传统的maximin/minimax准则来构造高维稀疏空间填充设计，从而克服了高维下所需设计点数过多以及对边界（尤其是超立方体顶点）过度敏感的缺陷。

    

    虽然球填充和球覆盖的概念构成了计算机实验中空间填充设计最广泛使用的准则——即maximin和minimax方法——的基础，但它们在高维情况下存在重大缺陷。首先，当维度 $d$ 非常大时，部署空间填充设计需要极其大量的设计点，这使得空间填充设计这一概念本身变得值得质疑。其次，这两种设计特征对边界高度敏感，尤其是当设计域为超立方体时（即本文考虑的情形），对其顶点尤为敏感。当设计规模小于 $2^d$ 时，这种敏感性表现得特别突出。本文不再采用maximin或minimax准则，而是考虑超立方体上均匀测度的量化误差。该误差具有三个关键优势：它考虑了从一个随机点出发的距离分布的整体情况（摘要原文在此处截断）。

    arXiv:2611.24507v1 Announce Type: cross  Abstract: While the notions of sphere packing and sphere covering underlie the most widely used criteria for space-filling design in computer experiments, namely, the maximin and minimax approaches, they suffer from major drawbacks in high dimensions. First, when the dimension $d$ is very large, deploying a space-filling design requires an extremely large number of design points, making the very notion of a space-filling design questionable. Second, these two design characteristics are highly sensitive to boundaries, particularly when the design domain is a hypercube (the case considered in this paper), and especially to its vertices. This sensitivity is particularly acute when the design size is less than $2^d$. Instead of considering the maximin or minimax criterion, we consider here the quantization error of the uniform measure on the hypercube. This error has three key advantages: it accounts for the entire distribution of distances from a r
    
[^53]: 从次优数据中学习潜在动作策略

    Latent Action Policies from Suboptimal Data

    [https://arxiv.org/abs/2611.24482](https://arxiv.org/abs/2611.24482)

    该论文提出了LAPIS方法，通过推导潜在动作空间中策略改进有效迁移与最优对齐的条件，解决了从次优数据训练潜在动作策略时离线强化学习不可靠的问题。

    

    潜在动作模型能够从无动作标注的视频中学习控制表示，但现有的流程通常采用行为克隆来训练潜在策略，缺乏显式的策略改进目标。尽管离线强化学习为策略改进提供了自然的途径，但标准方法无法直接迁移到学习得到的潜在动作空间。我们发现，学习到的潜在动作的结构使得朴素的离线强化学习不可靠：直接在潜在动作上最大化价值或拟合单峰策略，可能产生无法通过物理执行实现的潜在动作，同时潜在表示可能将获得不同奖励的物理控制合并在一起。为了理解潜在空间优化在何时仍然有效，我们推导了策略改进能够迁移到解码执行的条件，以及潜在最优与物理最优对齐的条件。基于这些见解，我们提出了LAPIS，用于目标条件下的潜在动作策略学习。

    arXiv:2611.24482v1 Announce Type: new  Abstract: Latent Action Models (LAMs) learn control representations from action-free video, but existing pipelines typically train latent policies with behavioral cloning, without an explicit policy improvement objective. Although offline RL offers a natural approach to policy improvement, standard methods do not transfer directly to learned latent action spaces. We find that the structure of learned latent actions makes naive offline RL unreliable: directly maximizing value over latent actions or fitting a unimodal policy can produce latent actions that cannot be realized by physical execution, while the latent representation can merge physical controls that receive different rewards. To understand when latent-space optimization remains valid, we derive conditions under which policy improvement transfers to decoded execution and under which latent and physical optima align. Building on these insights, we propose LAPIS for goal-conditioned latent 
    
[^54]: 每个查询自主路由：基于逐查询路由稀疏注意力加速视频扩散Transformer

    Every Query Routes Itself: Accelerating Video Diffusion Transformers with Per-Query Routed Sparse Attention

    [https://arxiv.org/abs/2611.24479](https://arxiv.org/abs/2611.24479)

    提出了无需训练的逐查询路由稀疏注意力方法PQRS，让每个查询独立选择自己的键簇并用质心补偿未选中的键簇，从而在提升注意力精度的同时加速视频扩散Transformer的推理。

    

    对长token序列的自注意力计算主导了视频扩散Transformer的推理成本。基于聚类的稀疏注意力同时对查询和键进行聚类，并让同一簇中的所有查询共享同一个键簇选择。我们证明了这种共享限制了准确性：在键簇和密度固定的情况下，随着查询簇的精细化，注意力变得更加准确，而当每个查询独立进行选择时最为准确，这完全不需要查询聚类。因此，我们提出了逐查询路由稀疏注意力（PQRS），这是一种无需训练的方法，仅对键进行聚类，让每个查询选择自己的键簇，并用质心来补偿未被选择的键簇。逐查询的选择使得没有任何查询块会共享键列表，因此PQRS遍历键簇而非查询，将具有相同选择的查询分桶为稠密任务，并在固定的softmax偏移量下一次合并它们的局部结果。

    arXiv:2611.24479v1 Announce Type: cross  Abstract: Self-attention over long token sequences dominates the inference cost of video diffusion transformers. Clustering-based sparse attention clusters both queries and keys and lets all queries in a cluster share one selection of key clusters. We show that this sharing limits accuracy: with key clusters and density fixed, attention grows more accurate as the query clusters are refined, and is most accurate when each query selects on its own, which needs no query clustering at all. We therefore propose Per-Query Routed Sparse Attention (PQRS), a training-free method that clusters only keys, lets each query select its own key clusters, and compensates for the unselected ones with their centroids. Per-query selections leave no query tile with a shared key list, so PQRS traverses key clusters instead of queries, buckets queries with identical selections into dense tasks, and merges their partial results in one pass under a fixed softmax offset 
    
[^55]: ICICLE：面向学习曲线外推的上下文内推理

    ICICLE: In-Context Inference for Curve-Learning Extrapolation

    [https://arxiv.org/abs/2611.24469](https://arxiv.org/abs/2611.24469)

    ICICLE 是一种轻量级（仅128万参数，约为FT-PFN的8.7%）的上下文内学习曲线外推代理模型，通过共享集合编码器支持可变数量的超参数，在多个基准上以远少于FT-PFN的参数量取得相当性能，并在Meta-Album基准上超越Quick-Tune。

    

    冻融贝叶斯优化通过预测部分观测到的学习曲线将如何延续来分配训练预算。LC-PFN 首先将先验数据拟合网络（PFN）应用于学习曲线外推，而 FT-PFN 则将该方法扩展为具备配置感知能力的冻融优化，采用了一个十槽超参数编码器。我们提出了面向学习曲线外推的上下文内推理方法 ICICLE，这是一种上下文内代理模型，通过一个共享集合编码器接受可变数量的超参数，并在涵盖类型化超参数、重尾噪声、量化效应和发散现象的合成先验上进行训练。ICICLE 拥有 128 万参数（仅为 FT-PFN 参数量的 8.7%），并在 LCBench、PD1 和 TaskSet 上产生了与 FT-PFN 相似的平均遗憾轨迹。在包含 33 个超参数的 86 项任务的 Meta-Album 基准上，ICICLE 总体上以及在延长预算设置下均取得了比 Quick-Tune 更低的平均归一化遗憾（原文此处截断）。

    arXiv:2611.24469v1 Announce Type: new  Abstract: Freeze-thaw Bayesian optimization allocates training budget by predicting how partially observed learning curves will continue. LC-PFN first applied prior-data fitted networks (PFNs) to learning-curve extrapolation, and FT-PFN extended this approach to configuration-aware freeze-thaw optimization with a ten-slot hyperparameter encoder. We introduce In-Context Inference for Curve-Learning Extrapolation (ICICLE), an in-context surrogate that accepts a variable number of hyperparameters through a shared set encoder and is trained on a synthetic prior covering typed hyperparameters, heavy-tailed noise, quantization, and divergence. ICICLE has 1.28 million parameters (8.7% of FT-PFN's count) and produces similar mean-regret trajectories to FT-PFN on LCBench, PD1, and TaskSet. On the 86-task Meta-Album benchmark with 33 hyperparameters, ICICLE reaches lower mean normalized regret than Quick-Tune overall and on extended, with no clear differenc
    
[^56]: 面向动态水上物联网鲁棒移动边缘计算的无人机与地面站协同：一种生成式AI增强方法

    Cooperative UAV and Ground Station for Robust MEC of Dynamic Aquatic IoT: A Generative AI-Enhanced Approach

    [https://arxiv.org/abs/2611.24466](https://arxiv.org/abs/2611.24466)

    本文提出一种生成式AI增强的无人机与地面站协同鲁棒边缘计算框架，通过联合任务卸载与无人机轨迹优化来最小化动态水上物联网计算任务的总执行时间。

    

    日益增长的水上物联网在海上搜救等应用中需要计算支持和覆盖保障。无人机可以提供低成本、灵活的空中服务，地面站可以提供强大的支撑，二者可以协同帮助复杂场景中的水上物联网。然而，无人机与地面站为水上物联网提供服务的协作面临任务不确定性、水上物联网轨迹不确定性、异构性以及计算资源受限等挑战。为解决这些问题，我们提出了一种基于无人机与地面站协同的鲁棒多接入边缘计算（MEC）框架，以辅助水上物联网完成计算任务。具体而言，我们将联合任务卸载与无人机轨迹优化问题建模为最小化总执行时间的优化问题，该问题属于混合整数非线性规划且为NP难问题。因此，我们提出了（原文在此处被截断）

    arXiv:2611.24466v1 Announce Type: cross  Abstract: The increasing aquatic Internet of Things (AIoTs) require computational support and coverage in applications such as maritime search and rescue. Unmanned aerial vehicles (UAVs) can offer low-cost, flexible aerial services, and ground stations (GSs) can provide powerful supports, which can cooperate to help the AIoTs in complex scenarios. However, the collaboration between UAVs and GSs for AIoTs faces challenges of task uncertainties, AIoTs trajectory uncertainties, heterogeneities, and limited computational resources. To address these issues, we propose a cooperative UAV and GS based robust multi-access edge computing (MEC) framework to assist AIoTs in completing computational tasks. Specifically, we formulate the optimization problem of joint task offloading and UAV trajectory to minimize the total execution time, which is in the form of mixed integer nonlinear programming and NP-hard to tackle. Therefore, we propose the algorithm of 
    
[^57]: 和声电路：和弦进行蕴含于符号音乐模型的稀疏主信息流中

    Harmonic Circuit: Chord Progression Lies in a Sparse Principal Information Flow in Symbolic Music Models

    [https://arxiv.org/abs/2611.24453](https://arxiv.org/abs/2611.24453)

    首次在符号音乐生成模型（ChatMusician 与 MuPT）中进行电路发现，识别出驱动和声进行 I…V→I 行为的一条稀疏主信息流电路，并通过因果干预验证其作用。

    

    机制可解释性旨在将神经网络的计算过程解码为人类可理解的逻辑，其核心方法之一是电路发现：识别驱动特定模型行为的子网络中的信息流。尽管电路发现技术在自然语言处理（NLP）领域已经趋于成熟，但音乐模型中的可解释电路仍在很大程度上未被探索。在本工作中，我们在两个符号音乐生成模型 ChatMusician 和 MuPT 中识别出一个具有音乐功能的电路。我们考虑的音乐功能是对经典和声进行模式 I…V→I 中终止主和弦根音的预测。我们使用 Auto-Emergence（自动涌现）算法提取出一个主要的稀疏子网络作为所识别的电路。通过因果干预进行的验证证实，该电路是符号音乐模型中和声进行行为的主要驱动因素。据我们所知，这一研究是……

    arXiv:2611.24453v1 Announce Type: cross  Abstract: Mechanistic Interpretability (MI) seeks to decode neural network computations into human-understandable logic. A core methodology in MI is circuit discovery: identifying the information flow in sub-networks that drive specific model behaviors. While circuit discovery has matured within Natural Language Processing (NLP), the interpretable circuits in music models remain largely unexplored. In this work, we identify a music-functional circuit within two symbolic music generative models: ChatMusician and MuPT. The music function we consider is the prediction of the terminal tonic root in a classical harmonic progression pattern $I \ldots V \rightarrow I$. We use the Auto-Emergence algorithm to extract a principal sparse sub-network as the identified circuit. Validation via causal intervention confirms that this identified circuit is a primary driver of harmonic progression behaviors in the symbolic music models. To our knowledge, this rep
    
[^58]: 从真实到合成：PSG-GAN学习睡眠呼吸暂停信号动力学用于AI训练

    From real to synthetic: PSG-GAN learns sleep apnea signal dynamics for AI training

    [https://arxiv.org/abs/2611.24436](https://arxiv.org/abs/2611.24436)

    本研究开发了一种生成对抗网络PSG-GAN，能够协同合成具有阻塞性睡眠呼吸暂停特征的长片段多信号PSG数据，并真实再现信号间的动态交互关系，为AI模型训练提供了高质量的合成数据。

    

    合成多导睡眠图（PSG）数据在医疗保健应用中具有巨大潜力，例如替代敏感且难以收集的真实数据。以往对合成PSG数据的研究主要集中在单个信号的短片段上，忽视了与睡眠障碍相关的信号间动态关系。在本研究中，我们开发了一种生成对抗网络（PSG-GAN），用于协同合成具有阻塞性睡眠呼吸暂停（OSA）典型的多样性和复杂特征的多个PSG信号的较长片段。合成信号看起来十分真实，且脑电图和眼电图的频段功率在各睡眠阶段呈现出相似的分布。此外，PSG-GAN学会了模拟信号间的交互作用，例如呼吸阻塞期间及之后基于光电容积脉搏波（PPG）的心率变化。仅使用自动评分的合成信号训练的自动评分模型……（摘要在此处截断）

    arXiv:2611.24436v1 Announce Type: new  Abstract: Synthetic polysomnography (PSG) data has great potential in healthcare applications, for example in substituting real data that is sensitive and challenging to collect. Previous research on synthetic PSG data has been largely focused on short segments of individual signals, overlooking the inter-signal dynamics related to sleep disorders. In this study, we developed a generative adversarial network (PSG-GAN) for co-synthesizing longer segments of multiple PSG signals with diverse and complex characteristics typical of obstructive sleep apnea (OSA). The synthetic signals appeared realistic, and the electroencephalography and electrooculography band powers were similarly distributed across sleep stages. Moreover, the PSG-GAN learned to mirror signal interactions such as changes in photoplethysmography-based heart rate during and after breathing obstructions. An automatic scoring model trained solely with automatically scored synthetic sign
    
[^59]: DyNA-Mixer：解耦时间与拓扑以实现可扩展的动态节点亲和度预测

    DyNA-Mixer: Decoupling Time and Topology for Scalable Dynamic Node Affinity Prediction

    [https://arxiv.org/abs/2611.24426](https://arxiv.org/abs/2611.24426)

    DyNA-Mixer 通过将时间演化建模与拓扑结构建模解耦，在保证计算可扩展性的同时捕捉多尺度时间模式与全图协同模式，实现了可扩展的动态节点亲和度预测。

    

    连续时间动态图保留了静态快照所丢弃的交互顺序、时间信息以及不断演化的交互强度，因此动态图挖掘对于推荐系统、贸易网络和交易平台等持续演化的系统至关重要。该领域的研究一直以动态链接预测为中心，即仅预测未来交互是否会发生，而许多应用还需要预测其强度。动态节点亲和度预测（DNAP）问题将这一任务泛化为连续目标值和全候选集排序，使其具有更广泛的适用性，但难度也显著增加。这带来了两个相互耦合的难题：排序任务需要一个表达能力足够强的预测器，以同时捕捉多尺度的时间演化模式与全图范围的协同模式；然而对每一个源节点—目标节点对都运行这样的预测器，其计算开销是不可承受的。我们提出了 DyNA-Mixer，它将时间预测与拓扑建模解耦。

    arXiv:2611.24426v1 Announce Type: new  Abstract: Continuous-time dynamic graphs preserve the ordering, timing, and evolving strength of interactions that static snapshots discard, making dynamic graph mining essential for evolving systems such as recommender systems, trade networks, and transaction platforms. Research in this area has centered on dynamic link prediction, which forecasts only whether a future interaction will occur, whereas many applications also need its intensity. The dynamic node affinity prediction (DNAP) problem generalizes this task to continuous targets and full-candidate ranking, making it more broadly applicable but substantially harder. This raises two coupled difficulties: ranking requires a predictor expressive enough to capture both multi-scale temporal evolution and graph-wide collaborative patterns, yet running such a predictor over every source--destination pair is computationally prohibitive. We propose DyNA-Mixer, which decouples temporal forecasting f
    
[^60]: 预测流程级行为：一项基准测试研究

    Predicting Process-Level Behavior: A Benchmarking Study

    [https://arxiv.org/abs/2611.24425](https://arxiv.org/abs/2611.24425)

    该论文通过对并发轨迹预测和平均吞吐时间预测两项流程级任务的全面基准测试，比较了统计方法、机器学习方法、基础模型、业务流程仿真以及聚合到流程级的轨迹级PPM方法，揭示了最适合预测流程级行为的建模范式。

    

    预测性流程监控（PPM）通常预测单个流程实例的未来情况，即在轨迹（trace）级别上进行运作。然而，流程管理者通常是在流程级别上进行决策的，例如在分配员工或计算关键绩效指标（KPI）时。鉴于PPM研究至今尚未专注于流程级预测，一个问题是：现有方法对这类流程级管理决策的支持程度究竟如何。我们通过对两项流程级预测任务——并发轨迹预测和平均吞吐时间预测——的全面基准测试研究来回答这一问题。在合成事件日志和真实事件日志上，我们比较了统计方法和基于机器学习的方法、基础模型、业务流程仿真，以及（这一点尤为重要）已被广泛采用的轨迹级PPM方法——我们将其预测结果聚合到流程级别。我们评估了哪些建模范式最适合用于预测流程级行为。

    arXiv:2611.24425v1 Announce Type: new  Abstract: Predictive process monitoring (PPM) typically forecasts the future of individual process instances, hence operating on trace level. Process managers, however, typically operate on process level, e.g., when assigning staff or calculating KPIs. The question is how well such process-level managerial decisions are supported by existing approaches, given that PPM research has so far not focused on process-level predictions. We answer this question through a comprehensive benchmarking study of two process-level prediction tasks: concurrent trace prediction and average throughput time prediction. Across synthetic and real-life event logs, we compare statistical and machine-learning-based methods, foundation models, business process simulation, and, importantly, established trace-level PPM approaches, aggregating their predictions to the process-level. We assess which modeling paradigms are most suitable for predicting process-level behavior and
    
[^61]: 以自身估计为基准探测运动扩散模型中的物理知识

    Probing Physics in Motion Diffusion Models Against Their Own Estimate

    [https://arxiv.org/abs/2611.24423](https://arxiv.org/abs/2611.24423)

    该研究对运动扩散模型的内部激活进行逐层物理探测后发现，当读取方法被公平调整时，模型自身的干净样本估计与内部激活的表现相当（差距最多仅0.01），表明此前报告的“隐藏物理知识”更多是读取方法可提取性差异造成的假象，而非模型真正隐藏了物理信息。

    

    最近的研究报告称，视频扩散模型“知道的比它们表现出来的更多”：物理属性从其内部激活值中解码的效果优于从其输入或输出中解码。然而，扩散模型正是从这些相同的激活值计算出其干净样本估计值的，因此这类比较实际衡量的只是读取方法提取信息的难易程度，而模型是否隐藏了信息则是另一个独立的问题。我们着手在运动学文本到动作扩散模型中寻找隐藏的物理知识——在这类模型中，每个中间估计值的物理属性都可以被精确计算——并记录了MDM（两个检查点）和MotionDiffuse采样轨迹上的每一层。与通常的基线相比，这些内部层在平衡和飞行方面确实看起来领先于模型自身的估计。但当所有读取方法以相同方式调整后，模型自身的估计缩小了这一差距：对于平衡和飞行，额外读取一层最多只带来0.01的增益，而阳性对照实验表明我们能够检测到这一量级的差异。

    arXiv:2611.24423v1 Announce Type: new  Abstract: Recent studies report that video diffusion models "know more than they show": physical properties decode better from their activations than from their inputs or outputs. But a diffusion model computes its clean-sample estimate from those same activations, so such a comparison measures how easily a readout extracts the information. Whether the model hides anything is a separate question. We set out to find hidden physical knowledge in kinematic text-to-motion diffusion models, where the physics of every intermediate estimate can be computed exactly, and recorded every layer along the sampling trajectories of MDM (two checkpoints) and MotionDiffuse. Against the usual baselines the layers do look ahead of the model's own estimate on balance and flight. With every readout tuned the same way, the estimate closes the gap: for balance and flight, adding a layer gains at most 0.01, a size that a positive control shows we would detect. The fall r
    
[^62]: MINT：面向多步离策略强化学习截断的相遇时间指标

    MINT: Meeting-time Indicators for Truncation in Multi-Step Off-Policy RL

    [https://arxiv.org/abs/2611.24413](https://arxiv.org/abs/2611.24413)

    该论文提出 MINT 方法，用一个基于行为轨迹与目标轨迹首次相遇时间的二元停止指标来替代传统多步离策略强化学习中的逐轨迹系数乘积，从而免除了对敏感截断长度超参数、转移模型及混合时间估计的依赖，同时兼顾方差控制与长时程信号。

    

    许多多步离策略时序差分方法通过每步轨迹系数的乘积来处理策略失配问题，这导致了方差控制与长时程信号之间的权衡，使得最大轨迹长度成为一个敏感的超参数。我们提出了 MINT（Meeting-time INdicators for Truncation，用于截断的相遇时间指标），它用一个二元停止指标来替代这些系数乘积：在耦合的行为轨迹与目标轨迹首次相遇之前取值为 1，之后取值为 0。在真实价值函数处，该指标的幂等性产生了一个由相遇时间生存概率所支配的方差界。在实际实现中，只需单次采样的动作即可为该停止规则提供无模型代理，无需转移模型或混合时间估计。我们在固定策略评估下，分别推导了状态耦合算子其实际代理算子收缩的充分条件。在 MuJoCo 和 DeepMind Control 任务上，MINT 的表现优于现有方法。

    arXiv:2611.24413v1 Announce Type: cross  Abstract: Many multi-step off-policy temporal-difference methods handle policy mismatch through products of per-step trace coefficients, leading to a trade-off between variance control and long-horizon signal. The maximum trace length consequently becomes a sensitive hyperparameter. We propose MINT (Meeting-time INdicators for Truncation), which replaces these products with a binary stopping indicator: one until coupled behavior and target trajectories first meet, zero thereafter. At the true value function, the indicator's idempotence yields a variance bound governed by meeting-time survival. For practical implementation, a single sampled action provides a model-free surrogate for the stopping rule, requiring no transition model or mixing-time estimate. We derive separate sufficient conditions for contraction of the state-coupled operator and its practical surrogate under fixed-policy evaluation. On MuJoCo and DeepMind Control tasks, MINT outpe
    
[^63]: RSIBench-Infra：智能体能否在生产规模下优化自身的训练与推理基础设施？

    RSIBench-Infra: Can Agents Optimize Their Own Training and Inference Infrastructure at Production Scale?

    [https://arxiv.org/abs/2611.24395](https://arxiv.org/abs/2611.24395)

    该论文提出了 RSIBench-Infra，一个面向生产规模、评估智能体能否在引擎、内核和指令层级对训练与推理基础设施进行端到端优化的基准，结果显示 66.2% 的任务实现了超过 1.10 倍的有效加速。

    

    提升训练与推理基础设施的效率是递归自我改进的一个实际组成部分。现有的智能体驱动优化基准大多聚焦于单一软件层（如内核），尚不清楚智能体能否在生产工作负载下带来端到端的性能提升。我们提出了 RSIBench-Infra，这是一个包含 20 个训练与推理任务的基准，涵盖多种模型家族和模态，部署规模最高可达 32 个加速器，覆盖 NVIDIA GPU 与昇腾 NPU。智能体从基于官方方案构建的可运行基线出发，可以在引擎、内核和指令层级对软件栈进行修改。OCI 镜像层提交协议记录这些更改以实现可复现部署，同时由独立评估在任务特定的质量约束下衡量端到端性能。在所有任务结果中，66.2% 实现了超过 1.10 倍且经过验证的有效加速。这些结果……

    arXiv:2611.24395v1 Announce Type: new  Abstract: Improving the efficiency of training and inference infrastructure is a practical component of recursive self-improvement. Existing benchmarks for agent-driven optimization largely focus on individual software layers such as kernel, leaving unclear whether agents can deliver end-to-end performance gains under production workloads. We introduce RSIBench-Infra, a benchmark of 20 training and inference tasks spanning diverse model families and modalities, with deployments of up to 32 accelerators on NVIDIA GPUs and Ascend NPUs. Agents start from runnable baselines derived from official recipes and can modify the software stack across engine, kernel, and instruction levels. An OCI image-layer submission protocol captures these changes for reproducible deployment, while independent evaluation measures end-to-end performance under task-specific quality constraints. Across all task results, 66.2% exceed $1.10\times$ validated speedup. These resu
    
[^64]: 循环神经网络中的时间秩

    Temporal Rank in Recurrent Neural Networks

    [https://arxiv.org/abs/2611.24394](https://arxiv.org/abs/2611.24394)

    本文发现了RNN中独立于梯度消失/爆炸问题的另一关键局限——由秩约束的隐藏状态动态导致的信息瓶颈，并提出结构化高维状态动态方法加以解决，使标准RNN在序列任务上实现快速学习。

    

    循环神经网络是时序建模的自然框架，然而由于其顺序化、不可并行的计算方式以及难以捕捉长程依赖而常常被忽视，这些困难通常被归因于梯度的消失和爆炸。我们发现了一个独特且未被充分研究的局限性：由秩约束的隐藏状态动态所引发的信息瓶颈，它限制了随时间推移能够保留的独立输入方向的数量，这体现在输入到状态映射的秩上。我们证明这一约束会独立于梯度稳定性而限制时间容量。为了解决这一问题，我们引入了结构化的高维状态动态，以保持时间可分性并防止输入到状态的雅可比矩阵随时间饱和。在受控的序列任务上，所得模型尽管严格保持在标准RNN公式框架之内，仍然表现出快速的学习能力。

    arXiv:2611.24394v1 Announce Type: cross  Abstract: Recurrent Neural Networks (RNNs) are a natural framework for temporal modelling, yet are often dismissed due to their sequential, non-parallelisable computation and difficulty in capturing long-range dependencies, typically attributed to vanishing and exploding gradients. We identify a distinct and under-explored limitation: an information bottleneck induced by rank-constrained hidden state dynamics, which limits the number of independent input directions that can be preserved over time, as reflected in the rank of the input-to-state mapping. We show that this constraint restricts temporal capacity independently of gradient stability. To address this, we introduce structured high-dimensional state dynamics that preserve temporal separability and prevent saturation of the input-to-state Jacobian over time. On controlled sequence tasks, the resulting model exhibits rapid learning despite remaining strictly within a standard RNN formulati
    
[^65]: 如何将注意力神经网络转化为可解释的硬逻辑谓词电路：忒修斯之船式替换下的真值表谓词提取与状态机合并

    How to Turn an Attention Neural Network into an Interpretable Hard-Logic Predicate Circuit: Truth-Table Predicate Extraction and State-Machine Merging under Ship-of-Theseus Replacement

    [https://arxiv.org/abs/2611.24381](https://arxiv.org/abs/2611.24381)

    该论文提出一种“忒修斯之船”式逐块替换方法，将训练好的注意力网络（包括注意力机制本身）整体转化为由0/1谓词与最小化真值表构成的硬逻辑谓词电路，既保持可解释性又能在教师正确的样本上忠实模仿教师。

    

    我们将一个训练好的注意力网络作为一个整体（包括注意力机制本身）转换为可在其位置运行的硬逻辑谓词电路。电路的每个单元都是关于词元的0/1谓词，谓词之间的每个映射都是一个最小化的真值表。较短的表（如查询码与键码的乘积）采用精确方法最小化，而宽的特征到特征的转换则通过容噪最小化转化为蕴含规则。模块从输出端开始逐个被替换，如同忒修斯之船的方式，每次替换后，剩余的软性部分会重新适应已冻结的电路。注意力选择采用两种形式：整数分数的逐行比较（顺序选择）和学到的规则（规则选择）。两者仅在普通训练集上拟合，且只在教师分类正确的样本上进行拟合，从而使电路在教师正确之处模仿教师。在TweetEval-emotion上，……

    arXiv:2611.24381v1 Announce Type: new  Abstract: We convert a trained attention network as a whole, attention included, into a hard-logic predicate circuit that runs in its place. Every unit of the circuit is a 0/1 predicate on tokens, and every map between predicates is a minimised truth table. Short tables, such as the product of a query code and a key code, are minimised exactly, and wide feature-to-feature transitions become implication rules by noise-tolerant minimisation. Blocks are replaced one at a time from the output down, in the manner of the Ship of Theseus, and after each replacement the remaining soft part re-adapts to the frozen circuit. Attention selection takes two forms, a row-wise comparison of integer scores (order selection) and a learned rule (rule selection). Both are fitted on the ordinary training set alone, and only on the examples that the teacher classifies correctly, so that the circuit imitates the teacher where the teacher is right. On TweetEval-emotion, 
    
[^66]: MoE-Context：只让模型看到它需要的内容

    MoE-Context: Show the Model Only What It Needs

    [https://arxiv.org/abs/2611.24379](https://arxiv.org/abs/2611.24379)

    MoE-Context通过在每步前进行快速的“是/否”门控判断，只向语言模型展示当前所需的上下文块，其余内容以可召回的单行笔记形式掩蔽在单一键值缓存中而不被删除，从而将查找任务准确率从93/120大幅提升至119/120。

    

    AI智能体在每一步都会读取其完整的历史记录。人类并非如此工作：我们只保留当下所需的信息，其余内容则以可随时回忆的粗略笔记形式保存。MoE-Context将这一理念应用于语言模型：上下文被分割成多个块，在每一步之前，一次快速的“是/否”判断决定该步需要哪些块；其余块保留为可以调回的单行笔记，且不删除任何内容。在我们构建的版本中，同一个模型先做出这一判断再进行回答，所有操作都在单一的键值缓存上进行，其中被隐藏的块是被掩蔽而非移除。在配备Qwen3-1.7B的笔记本电脑上进行的合成查找任务中，仅显示所需块可将准确率从120题中的93题提升至119题，且被掩蔽的块与被删除的块同样不可见。模型自身的挑选能力是薄弱环节：一个写入空闲输出行的监督式线性探针将准确率从120题中的80题提升至110题，但在该任务上简单的关键词搜索能实现完美挑选，且该探针无法迁移（摘要在此处截断）。

    arXiv:2611.24379v1 Announce Type: new  Abstract: AI agents read their whole history at every step. People do not work like that: we keep what the moment needs and hold the rest as a rough note we can recall. MoE-Context applies this to language models. The context is split into blocks. Before each step a fast yes or no pass decides which blocks the step needs; the rest stay as one-line notes that can be brought back, and nothing is deleted. In the version we build, one model makes this decision and then answers, over a single key-value cache where hidden blocks are masked rather than removed. On a laptop with Qwen3-1.7B and a synthetic lookup task, showing only the needed block raises accuracy from 93 to 119 of 120, and a masked block is as hidden as a deleted one. The model's own picking is the weak part. A supervised linear probe written into spare output rows lifts its accuracy from 80 to 110 of 120, but simple keyword search picks perfectly on this task, and the probe does not tran
    
[^67]: LLM辅助的多保真度贝叶斯优化：融合专家知识与数值数据

    LLM-Assisted Multi-Fidelity Bayesian Optimization: Fusing Expert Knowledge with Numerical Data

    [https://arxiv.org/abs/2611.24377](https://arxiv.org/abs/2611.24377)

    该论文提出LLM辅助的多保真度贝叶斯优化方法，利用大型语言模型将专家关于低保真度模型所缺失物理效应的定性描述转化为成对比较知识，并与数值数据融合，从而在高保真度数据稀缺时更准确地学习低-高保真度差异，降低工程设计成本。

    

    多保真度贝叶斯优化（MFBO）通过将廉价的低保真度（LF）模型与少量精确的高保真度（HF）评估相结合，并依次选择要评估的设计，从而降低工程设计成本。然而，当高保真度数据稀缺时，仅凭数值数据难以学习低保真度与高保真度模型之间的差异，尤其是当低保真度模型中被省略的物理效应依赖于设计本身时。即使数据不足，领域专家通常也清楚低保真度模型省略了哪些物理效应，以及在何种条件下这些效应会增强。现有多保真度模型无法直接利用这类定性知识，而大型语言模型（LLM）则能够解读此类知识并据此比较不同设计。所提出的LLM辅助MFBO在传统多保真度模型基础上增加了一个知识基准：根据专家对被省略物理效应的文字描述，LLM判断两个设计中哪一个需要更大的带符号残差修正，并利用成对高斯（摘要在此处截断）

    arXiv:2611.24377v1 Announce Type: cross  Abstract: Multi-fidelity Bayesian optimization (MFBO) reduces engineering design cost by combining inexpensive low-fidelity (LF) models with a few accurate high-fidelity (HF) evaluations and sequentially selecting designs to evaluate. However, when HF data are scarce, the LF-HF discrepancy is difficult to learn from numerical data alone, especially when physics omitted in the LF model makes it design-dependent. Yet even without sufficient data, domain experts often know which physics the LF model omits and under which conditions its effect grows. Existing multi-fidelity (MF) models cannot use such qualitative knowledge directly, but large language models (LLMs) can interpret it and compare designs accordingly. The proposed LLM-Assisted MFBO adds a knowledge basis to the conventional MF model: from an expert description of the omitted physics, an LLM judges which of two designs requires the larger signed residual correction, and a pairwise Gaussi
    
[^68]: VAFL：一种基于闭式解的高效众包服务纵向分析联邦学习机制

    VAFL: A Vertical Analytic Federated Learning Mechanism for Efficient Crowdsourced Services via Closed-Form Solutions

    [https://arxiv.org/abs/2611.24367](https://arxiv.org/abs/2611.24367)

    提出了一种基于闭式解的纵向分析联邦学习框架VAFL，通过正则化最小二乘在固定局部表示上拟合任务头，并利用块因子等价求解和两种宽度缩减策略大幅降低计算开销，实现高效的两方及多方隐私保护协作学习。

    

    智能服务通常依赖于不同组织所持有的互补信息，这推动了无需集中式原始数据汇集的纵向联邦学习的发展。我们提出VAFL，一种纵向分析联邦学习框架，通过正则化最小二乘法在固定局部表示上拟合分类和回归头。该框架采用代理辅助的隐私保护计算工作流，支持两方和多方协作。为降低拟合开销，我们开发了一种全维度等价求解方法，将块因子直接应用于目标矩阵，避免了构建完整逆矩阵。我们进一步研究了两种宽度缩减策略：带系数提升的局部谱投影，以及带低秩跨方修正的全维度局部拟合。前者压缩预测空间，而后者保留了协作子空间之外的局部成分。我们建立了它们的代数性质……

    arXiv:2611.24367v1 Announce Type: new  Abstract: Intelligent services often rely on complementary information held by different organizations, motivating Vertical Federated Learning without centralized raw-data pooling. We propose VAFL, a Vertical Analytic Federated Learning framework that fits classification and regression heads on fixed local representations through regularized least squares. An agent-assisted protected-computation workflow supports two-party and multi-party collaboration. To reduce fitting overhead, we develop a full-dimensional equivalent solve that applies block factors directly to the target matrix, avoiding construction of the full inverse. We further study two width-reduction strategies: local spectral projection with coefficient lifting, and full-dimensional local fitting with low-rank cross-party correction. The first compresses the prediction space, whereas the second retains local components outside the collaboration subspace. We establish their algebraic p
    
[^69]: OCTR2A：城市交叉口实时RGB到鸟瞰图（BEV）轨迹投影与预测

    OCTR2A: Real-Time RGB-to-BEV Trajectory Projection and Forecasting at Urban Intersections

    [https://arxiv.org/abs/2611.24352](https://arxiv.org/abs/2611.24352)

    本文提出OCTR2A架构，仅利用现有城市基础设施上的对角成对RGB相机即可实时重建并预测交叉口交通参与者的鸟瞰图轨迹，无需激光雷达、航拍或高精地图等昂贵资源。

    

    城市交叉口的轨迹预测研究通常假设可靠的鸟瞰图（BEV）运动历史是可获得的，而这在实践中需要昂贵资源，如激光雷达、航拍视角、高精地图或经过标定的基础设施。我们通过仅使用安装在现有城市基础设施上、高度可变的对角成对RGB相机，在推理时重建并预测交通参与者的BEV轨迹，从而解决这一从感知到预测的上游问题。我们提出了对角轨迹重建与预测架构（OCTR2A），这是一个模块化流水线，集成了带类别细化的单视角检测、几何跨相机匹配、基于Transformer的BEV投影以及以自身为中心的GRU轨迹预测，采用级联方式训练，并主要以软多尺度空间重叠损失进行监督。由于目前没有公开基准提供这种传感配置，我们引入了基于Blender构建的R3ST++……（摘要未完整提供）

    arXiv:2611.24352v1 Announce Type: cross  Abstract: Trajectory forecasting at urban intersections is often studied assuming that reliable bird's-eye-view (BEV) motion histories are available, which in practice requires costly resources such as LiDAR, aerial views, HD maps, or calibrated infrastructure. We address this upstream perception-to-forecasting problem by reconstructing and predicting agents' BEV trajectories only from paired opposite-corner RGB cameras mounted on existing urban infrastructure at variable heights at inference time. We introduce the Opposite-Corner Trajectory Reconstruction and Anticipation Architecture (OCTR2A), a modular pipeline integrating per-view detection with class refinement, geometric cross-camera matching, transformer-based BEV projection, and egocentric GRU forecasting, trained in cascade and mainly supervised with a soft multi-scale spatial-overlap loss. Since no public benchmark provides this sensing configuration, we introduce R3ST++, a Blender-bas
    
[^70]: 告诉我你有多确定：向基础模型传达辅助证据中的不确定性

    Tell Me How Sure You Are: Communicating Uncertainty in Auxiliary Evidence to Foundation Models

    [https://arxiv.org/abs/2611.24351](https://arxiv.org/abs/2611.24351)

    本文首次系统地证明，向基础模型传达上下文信息中的不确定性能够为下游零样本图像分类决策提供有用的额外线索，并在多种设置中带来性能提升。

    

    在许多查询中，提供给生成式基础模型（如大语言模型和视觉语言模型）的上下文信息本质上是不确定的，包括其对物理世界的表示。虽然传统的提示方法会丢弃这些相关的不确定性，但本文首次在文献中系统地证明，在上下文中传达这种不确定性能够为下游决策提供有用的额外线索，并在广泛的设置中带来改进。我们将该决策问题表述为一个由基于文本的辅助特征支持的零样本基础模型辅助图像分类任务。我们选择这一任务，是因为它具有明确定义的辅助属性、相关的不确定性以及无歧义的任务正确性。我们使用五个下游基础模型，在三个数据集上广泛评估了来自人类和习得模型在观察或预测时产生的不确定性。

    arXiv:2611.24351v1 Announce Type: new  Abstract: In many queries, the context information provided to generative foundation models, such as LLMs and VLMs, is inherently uncertain, including its representations of the physical world. While conventional prompting discards the associated uncertainty, this paper, for the first time in the literature, systematically demonstrates that communicating such uncertainty in context provides useful additional cues for downstream decisions and yields improvements across a wide range of settings.   We formulate the decision-making problem as a zero-shot foundation model-assisted image classification task supported by auxiliary text-based characteristics. We choose this due to the availability of well-defined auxiliary properties, associated uncertainty, and unambiguous task correctness. Using five downstream foundation models across three datasets, we extensively evaluate uncertainty arising from both humans and learned models when observing or predi
    
[^71]: 通过个性化数据分析探索日常情绪幸福感

    Exploring Daily Emotional Well-Being through Personalized Data Analysis

    [https://arxiv.org/abs/2611.24342](https://arxiv.org/abs/2611.24342)

    该项目通过整合多源个人数据构建每日情绪数据集，结合人格与幸福感问卷对个体日常情绪状态及其时间变化进行个性化分析，并据此提出了一种简单的个性化情绪推荐方法。

    

    该项目从个性化视角出发，利用源自日常生活的情绪状态记录及情境信息来分析每日情绪幸福感。项目基于这样一种理念：幸福感并非固定不变的数值，而是逐日波动的，只有结合个体的情境与行为才能被更好地理解。这项工作归属于数据科学领域，聚焦于个体层面的情绪状态。项目重点在于从多个个人数据源构建一个整合的每日数据集并对情绪状态进行分析，研究了情绪变量的分布、其随时间的演变、用户之间的差异以及情绪随时间的波动性。此外，项目还引入了人格与幸福感问卷作为稳定的参照基准，以便更好地对每日情绪分析结果进行情境化解读。在此基础上，项目通过比较（相似性）……开发了一种简单的个性化推荐方法。

    arXiv:2611.24342v1 Announce Type: cross  Abstract: This project analysed daily emotional well-being from a personalized perspective using records of emotional states and contextual information derived from daily life. It was based on the idea that well-being is not a fixed value, but varies from day to day, and can be better understood in relation to individual context and behaviour. The work is framed within the field of Data Science and focuses on emotional states at an individual level. The project focused on constructing an integrated daily dataset from multiple personal data sources and analysing emotional states. The distributions of emotional variables, their temporal evolution and the differences between users were studied, together with their variability over time. Personality and well-being questionnaires were also incorporated as stable references to better contextualize daily emotional results. A simple personalized recommendation approach was then developed by comparing si
    
[^72]: OpenGNS：涵盖视觉、语言与扩散模型的梯度噪声尺度开放数据集

    OpenGNS: An Open Dataset of the Gradient Noise Scale across Vision, Language, and Diffusion Models

    [https://arxiv.org/abs/2611.24331](https://arxiv.org/abs/2611.24331)

    该论文发布了首个开放GNS测量数据集OpenGNS，包含横跨ResNet图像分类、GPT语言建模与DiT图像合成三大领域的400余条测量轨迹，为系统研究梯度噪声尺度与临界批量大小提供了数据基础。

    

    梯度噪声尺度（GNS）指标衡量随机梯度下降中梯度的信噪比，正如McCandlish等人（2018）所示，该指标可用于近似神经网络训练中的临界批量大小（CBS）。尽管GNS出现在前沿模型的技术报告中，但目前尚无涵盖现代架构的GNS测量开放数据集，这限制了对该指标的系统研究。我们通过发布OpenGNS来填补这一空白，该数据集包含超过400条GNS测量轨迹，横跨三个领域：使用ResNet进行图像分类、使用生成式预训练Transformer（GPT）进行自回归语言建模，以及使用扩散Transformer（DiT）进行图像合成，覆盖参数量从1100万到10亿的模型，这些模型在最大更新参数化（muP）下训练，并进行了密集的批量大小和学习率扫描。基于该数据集，我们研究了GNS如何依赖于超参数、规模和训练（过程）……

    arXiv:2611.24331v1 Announce Type: new  Abstract: The Gradient Noise Scale (GNS) metric measures the signal-to-noise ratio of gradients in stochastic gradient descent and, as shown by McCandlish et al. (2018), can be used to approximate the Critical Batch Size (CBS) during neural network training. Despite its appearance in technical reports of frontier models, no open dataset of GNS measurements across modern architectures exists, limiting systematic study of the metric. We close this gap by releasing OpenGNS, a dataset of more than 400 GNS measurement trajectories spanning three domains: image classification with ResNets, autoregressive language modeling with Generative Pre-trained Transformers (GPTs), and image synthesis with Diffusion Transformers (DiTs), covering models from 11M to 1B parameters trained under Maximal Update Parameterization (muP) with dense batch-size and learning-rate sweeps. Building on this dataset, we study how the GNS depends on hyperparameters, scale, and trai
    
[^73]: 权重空间编程能否改变已习得的伦理优先级？

    Can Weight-Space Programming Alter Learned Ethical Priorities?

    [https://arxiv.org/abs/2611.24330](https://arxiv.org/abs/2611.24330)

    该研究表明，通过直接编辑权重空间中的选择权重，可以成功调整模型学习到的伦理优先级——使模型在权衡中更严格地对待正式誓言而非普通承诺，同时不损害其在其他任务上的表现与事实准确性。

    

    我们研究了权重编辑是否能改变模型在“遵守承诺”与“让更多人受益”之间的权衡方式。在一个概念监督的文本模型中，我们修改了拟合得到的选择权重，使得推翻正式誓言所需的收益大于推翻普通承诺所需的收益。在六个独立训练的文本阅读器上，该编辑在两种测试顺序下，对目标普通冲突选择更大群体的成功率为100%，对目标正式冲突的成功率为83.3%。该编辑同时保持了源模型在较小冲突上的成功率以及其事实准确性。剩余的正式冲突失败案例恰好与读取受益人数量时的错误重合。一个独立的分类器演示了为什么这种区分至关重要：复用事实答案可以通过无条件选择测试，却无法保留小差距承诺。在COCO128上的YOLO11n试点实验中，偏置编辑将小目标匹配数从3/43提升至9/43，而等效的分数校准方法重现了……（原文在此处截断）

    arXiv:2611.24330v1 Announce Type: cross  Abstract: We investigate whether a weight edit can alter how a model trades off honoring a commitment against benefiting more people. In a concept-supervised text model, we alter fitted choice weights so that the benefit required to override a formal pledge is larger than for an ordinary promise. Across six independently trained text readers, the edit selects the larger group in both orders on 100% of targeted ordinary conflicts and 83.3% of targeted formal conflicts. It preserves the source model's success rates on smaller conflicts and its factual accuracy. The remaining formal-conflict failures coincide with errors reading a beneficiary number. A separate classifier demonstrates why this distinction matters: reusing a factual answer passes an unconditional choice test but fails to preserve small-gap promises. In a YOLO11n pilot on COCO128, a bias edit increased small-target matches from 3/43 to 9/43, and an equivalent score calibration reprod
    
[^74]: 天气预报的无分布不确定性量化：时空依赖下的共形预测

    Distribution-Free Uncertainty Quantification for Weather Forecasting: Conformal Prediction Under Spatiotemporal Dependence

    [https://arxiv.org/abs/2611.24329](https://arxiv.org/abs/2611.24329)

    该论文对八种共形预测区间构造方法在网格化2米气温预报（6小时至10天八个时效）中的表现进行了系统基准测试，并创新性地引入空间过散度指数、每日空间覆盖率分位数和漏报持续性等时空依赖性诊断指标，揭示了边际覆盖率与时空联合覆盖表现之间的差异。

    

    共形预测能够提供有限样本、无分布的边际覆盖率保证，并已开始被应用于网格化天气预报。然而，边际覆盖率关注的是单个点位上长期被覆盖的比例，而非在某一天内覆盖率在空间与时间上的联合表现。我对八种区间构造方法（高斯方法、池化拆分共形、逐格点拆分共形、逐季节方法、自适应共形推断和滚动窗口）进行了基准测试，研究对象是网格化的2米气温预报，涵盖从6小时到10天共八个预报时效，使用5.625度分辨率的ERA5数据、一个刻意保持简洁的卷积预报模型以及ECMWF的业务化HRES模式，其中2017-2018年用于校准，2019-2021年用于测试。除覆盖率、区间宽度和区间得分外，我还报告了依赖性诊断指标：空间过散度指数、每日空间覆盖率的分位数，以及相对于日期随机打乱零假设的漏报持续性。逐格点方法达到了接近名义水平的边际覆盖率……

    arXiv:2611.24329v1 Announce Type: new  Abstract: Conformal prediction gives finite-sample, distribution-free marginal coverage and has begun to be applied to gridded weather forecasts. Marginal coverage concerns the long-run fraction of covered points at a location, not how coverage behaves jointly across space and time on a given day. I benchmark eight interval-construction procedures (Gaussian, pooled and per-cell split conformal, per-season, adaptive conformal inference and rolling windows) for gridded 2 m temperature forecasts at eight lead times from 6 hours to 10 days, using ERA5 at 5.625 degree resolution, a deliberately modest convolutional forecaster and ECMWF's operational HRES, with calibration in 2017-2018 and testing in 2019-2021. Alongside coverage, width and interval score, I report dependence diagnostics: a spatial overdispersion index, quantiles of daily spatial coverage and miss persistence relative to a shuffled-days null. Per-cell methods reach near-nominal marginal
    
[^75]: 凸函数、半凸函数与拟凸函数的结构保持神经逼近

    Structure-Preserving Neural Approximation of Convex, Semiconvex, and Quasiconvex Functions

    [https://arxiv.org/abs/2611.24325](https://arxiv.org/abs/2611.24325)

    本文将输入凸神经网络的结构保持逼近框架扩展到半凸、拟凸和凸可变换函数，建立了尖锐的极小极大逼近速率，并发展了具有 $L^\infty$ 逼近保证的结构保持输入拟凸神经网络（IqCNN）。

    

    输入凸神经网络通过架构约束提供了一种在全局范围内强制凸性的自然方法。我们研究了这种结构保持逼近框架如何扩展到半凸函数、拟凸函数以及凸可变换函数。对于紧凸域上的 $C^{1,1}$ 凸函数，我们在使用 $m$ 个仿射分片的最大仿射逼近下建立了尖锐的 $W^{1,p}$ 极小极大速率 $m^{-1/d}$，并证明每个最大仿射函数都存在精确的有界宽度 ICNN 表示，其复杂度与 $m$ 呈线性关系。同样的 $W^{1,p}$ 逼近速率通过二次提升扩展到了半凸函数，并通过单调复合扩展到了凸可变换函数。对于一般的 Lipschitz 拟凸函数，我们基于对仿射投影的单调响应发展了一种定量的有限探测逼近理论，并得到了具有 $L^\infty$ 逼近保证的结构保持 IqCNN。（注：原文摘要在此处被截断）

    arXiv:2611.24325v1 Announce Type: new  Abstract: Input convex neural networks provide a natural way to enforce convexity globally through architectural constraints. We study how this structure-preserving approximation framework extends to semiconvex, quasiconvex, and convex transformable functions. For $C^{1,1}$ convex functions on compact convex domains, we establish the sharp $W^{1,p}$ minimax rate $m^{-1/d}$ under max-affine approximation with $m$ affine pieces, and show that every max-affine function admits an exact bounded-width ICNN representation with complexity linear in $m$. The same $W^{1,p}$ approximation rate extends to semiconvex functions through quadratic lifting and to convex transformable functions through monotone composition. For general Lipschitz quasiconvex functions, we develop a quantitative finite-probe approximation theory based on monotone responses to affine projections and obtain a structure-preserving IqCNN with an $L^\infty$ approximation guarantee. We fur
    
[^76]: RespCCT与FreqCCT：用于多类别呼吸疾病分类的紧凑、可解释且成本感知的卷积Transformer

    RespCCT and FreqCCT: Compact, Explainable and Cost-Aware Convolutional Transformers for Multi-Class Respiratory Disease Classification

    [https://arxiv.org/abs/2611.24323](https://arxiv.org/abs/2611.24323)

    本文提出了两个仅约17万和12.5万参数的紧凑卷积Transformer（RespCCT与FreqCCT），从零训练即可在七个类别的呼吸疾病肺音分类中与大型预训练模型相媲美，同时显著降低计算成本并具备可解释性。

    

    胸部听诊的结果依赖于听诊者本人，而在缺乏训练有素的听诊人员时，能够对肺音进行分类的软件可以提供第二重判读。目前最精确的已发表分类器通常是对大型预训练网络进行微调，且其准确率很少与运行成本或配对统计信息一同报告。本研究做出了三项贡献：(i) 提出了RespCCT和FreqCCT两个紧凑的卷积Transformer，参数量分别为173,187和124,999，它们从随机初始化开始训练，输入为来自三个公开数据库（ICBHI、KAUH和RD@TR）的5秒音频帧转换成的梅尔频谱、恒定Q变换和倒谱图像，涵盖1,665个音频文件、七个类别。(ii) 在同一测试集上，采用配对bootstrap置信区间，将这两个模型与五个ImageNet预训练网络以及更大的蒸馏变体进行比较。(iii) 在单CPU线程上测量了流水线各阶段的计算成本，检验归因图是否依赖于训练得到的权重，并探究了模型学习到的特征（摘要在此处被截断）。

    arXiv:2611.24323v1 Announce Type: cross  Abstract: Chest auscultation depends on the listener, and software that classifies lung sounds could give a second reading where trained listeners are scarce. The most accurate published classifiers fine-tune large pretrained networks, and their accuracy is rarely reported with running cost or paired statistics. This study makes three contributions. (i) It proposes RespCCT and FreqCCT, two compact convolutional transformers of 173,187 and 124,999 parameters that learn from random initialization on mel, constant-Q and cepstral images of 5 s frames from three public databases, ICBHI, KAUH and RD@TR, with 1,665 audio files in seven classes. (ii) It compares both models with five ImageNet-pretrained networks and larger distilled variants on one test set, using paired bootstrap intervals. (iii) It measures the cost of every pipeline stage on one CPU thread, tests whether attribution maps depend on the trained weights, and probes whether the learned f
    
[^77]: 监督式赫布类网络：无需反向传播的学习

    Supervised Hebbian Class Networks: Learning Without Backpropagation

    [https://arxiv.org/abs/2611.24314](https://arxiv.org/abs/2611.24314)

    本文提出监督式赫布类网络，通过为神经元分配类别、采用跨层共享的基于类别的监督规则并结合适配的Lion优化器，实现了无需反向传播和梯度计算的多层感知机与卷积网络训练。

    

    本文提出了监督式赫布类网络，这是一种无需梯度计算或反向传播即可训练多层感知机和卷积网络的方法。学习层中的每个神经元都被分配一个类别，而连接则遵循通常的全连接或卷积模式。该方法将跨层共享的基于类别的监督规则与适配赫布活动乘积的Lion优化器相结合。所有层在整个训练过程中共同训练，其活动直接提供类别预测，无需单独训练的分类器。实验在MNIST数据集上使用多层感知机以及在CIFAR-10数据集上使用卷积网络对该方法进行了测试。

    arXiv:2611.24314v1 Announce Type: new  Abstract: This paper introduces Supervised Hebbian Class Networks as a method for training multi- layer perceptrons and convolutional networks without gradient calculations or backpropa- gation. Each neuron in a learned layer is assigned a class, while connections follow the usual fully connected or convolutional pattern. The method combines a class-based supervision rule shared across layers with a Lion optimizer adapted to Hebbian activity products. All layers train together throughout the run, and their activities provide class predictions di- rectly, without a separately trained classifier. The experiments test the method on MNIST with a multilayer perceptron and on CIFAR-10 with a convolutional networ
    
[^78]: AutoPI：面向LLM智能体提示注入的分类学引导自适应红队测试

    AutoPI: Taxonomy-Guided Adaptive Red Teaming for Prompt Injection in LLM Agents

    [https://arxiv.org/abs/2611.24313](https://arxiv.org/abs/2611.24313)

    AutoPI提出了一种以提示注入技术分类体系PIAtlas为引导的自适应红队测试方法，通过为每个子技术生成注入并迭代改进，旨在发现攻破LLM智能体的多种多样化方式，而非仅仅最大化攻击成功率。

    

    提示注入是LLM智能体面临的主要尚未解决的安全风险，社区现已达成共识：鲁棒性必须针对自适应攻击者而非固定的基准注入来衡量。目前已存在许多自适应红队工具，其中大多数以最大化攻击成功率（ASR）为目标。然而，ASR衡量的是目标被攻破的频率，而非被攻破方式的多样性。一个红队工具可以通过复用少数有效的技术获得高ASR，导致其余技术未被测试，也缺席于用于加固模型的攻击集合中。我们提出AutoPI，一种旨在发现攻破目标的多种不同方式、而不仅仅是频繁攻破的自适应红队工具。AutoPI利用PIAtlas——一个提示注入技术分类体系——来引导其搜索并衡量覆盖率。对于每个场景，攻击者LLM为分类体系中的每个子技术编写一条注入，并利用LLM评判器的反馈对失败的注入进行改进。该方法是黑盒的。

    arXiv:2611.24313v1 Announce Type: cross  Abstract: Prompt injection is the dominant unsolved security risk for LLM agents, and the community now agrees that robustness must be measured against adaptive attackers rather than fixed benchmark injections. Many adaptive red teamers exist, and most are built to maximize attack success rate (ASR). Yet ASR counts how often a target breaks, not in how many different ways. A red teamer can achieve a high ASR by reusing a few techniques that work, leaving the others untested and absent from the attacks used to harden models. We present AutoPI, an adaptive red teamer designed to find many different ways to break a target, not just to break it often. AutoPI uses PIAtlas, a taxonomy of prompt-injection techniques, both to guide its search and to measure coverage. For each scenario, an attacker LLM writes an injection for every sub-technique in the taxonomy and then reworks the ones that fail using feedback from an LLM judge. The method is black-box 
    
[^79]: LoopCache：加速循环Transformer的长上下文自投机解码

    LoopCache: Accelerating Long-Context Self-Speculative Decoding for Looped Transformers

    [https://arxiv.org/abs/2611.24301](https://arxiv.org/abs/2611.24301)

    本文提出LoopCache，一个面向长上下文循环Transformer的高效自投机解码框架，通过置信度引导机制解决现有波风格方法中固定循环起草浪费计算、以及跨循环批处理导致KV缓存内存流量放大R倍这两个关键问题。

    

    循环Transformer通过在R个递归步骤中重复应用权重共享的模块，以紧凑的模型实现较大的有效深度，每个循环步骤维护一个特定于该循环的键值缓存。为了加速顺序循环执行，最近出现了波风格的自投机方法，如WaveFront Decoding（WFD）和LoopSpec，它们在单个步骤中将草词和目标词跨循环流水线化。然而，这些方法面临两个关键局限。首先，它们只在固定的循环处进行起草，忽略了中间预测的变化，在过时的轨迹上浪费计算资源。其次，将来自不同循环的词进行批处理时，每步会加载多达R倍的KV缓存，在长上下文场景中严重放大了内存流量。为了应对这些挑战，我们提出了LoopCache，一个面向长上下文循环Transformer的高效自投机解码框架。基于预测切换与输出置信度之间的相关性，LoopCache引入了置信度引导的……（摘要在此处被截断）

    arXiv:2611.24301v1 Announce Type: new  Abstract: Looped Transformers achieve large effective depth with compact models by repeatedly applying a weight-shared block across R recurrent steps, each maintaining a loop-specific key-value (KV) cache. To accelerate sequential loop execution, recent wave-style self-speculative methods like WaveFront Decoding (WFD) and LoopSpec pipeline draft and target tokens across loops in a single step. However, they face two critical limitations. First, they draft only at fixed loops and ignore intermediate prediction changes, wasting computation on stale trajectories. Second, batching tokens from different loops loads up to Rx the KV cache per step, severely amplifying memory traffic in long-context scenarios. To address these challenges, we propose LoopCache, an efficient self-speculative decoding framework for long-context Looped Transformers. Based on the correlation between prediction switches and output confidence, LoopCache introduces confidence-gui
    
[^80]: 单元级股骨应力预测：一种用于有限元分析的混合图神经网络代理模型

    Element-Wise Femoral Stress Prediction: A Hybrid Graph Neural Network Surrogate for Finite Element Analysis

    [https://arxiv.org/abs/2611.24274](https://arxiv.org/abs/2611.24274)

    该研究提出了一种结合图卷积、邻域聚合与残差学习的混合图神经网络（GCN-SAGE-Residual）代理模型，可在保留有限元网格空间拓扑结构的前提下实现快速的单元级股骨应力预测，克服了传统有限元分析计算成本高、常规机器学习模型忽略网格结构的缺陷。

    

    髋部骨折是一个重大的临床挑战，因为它会导致较高的致残率和死亡率，且随着人口老龄化及骨骼脆性的增加，其发病率不断上升。尽管有限元分析（FEA）能够对股骨强度进行生物力学评估，但其临床适用性主要受限于高昂的计算成本。传统机器学习（ML）方法已被探索作为FEA的代理模型，虽然预测速度快，但无法捕捉有限元网格的空间和拓扑结构。为解决这些局限性，本研究提出了一种基于混合图神经网络（GNN）的框架，利用图结构化的有限元建模（FEM）数据进行股骨应力预测。所提出的GCN-SAGE-Residual架构结合了图卷积、邻域聚合和残差学习，以建模网格单元间的空间依赖关系和应力传播。

    arXiv:2611.24274v1 Announce Type: new  Abstract: Hip fractures represent a significant clinical challenge because they cause excess disability and mortality, and have a growing incidence in aging populations, as bone fragility increases. Although finite element analysis (FEA) provides biomechanical assessment of femoral strength, its clinical applicability is mainly limited by high computational cost. Conventional machine learning (ML) approaches have been explored as surrogate models for FEA; predictions are fast but fail to capture the spatial and topological structure of finite element meshes. To address these limitations, this study proposes a hybrid Graph Neural Network (GNN)-based framework for femoral stress prediction using graph-structured finite element modeling (FEM) data. The proposed GCN-SAGE-Residual architecture combines graph convolution, neighborhood aggregation, and residual learning to model spatial dependencies and stress propagation across mesh elements. We systema
    
[^81]: 稳定性相容核：超越高斯性的脉冲响应估计

    Stability-compatible kernels: beyond Gaussianity for impulse response estimation

    [https://arxiv.org/abs/2611.24272](https://arxiv.org/abs/2611.24272)

    该论文提出“稳定性相容核”的新概念，证明其当且仅当稳定元素在对应RKHS中稠密时成立，从而在超越高斯先验的情形下为脉冲响应估计提供了比已知条件更弱的几乎必然BIBO稳定性判据。

    

    线性系统辨识中基于核的正则化方法使用半正定核既来定义再生核希尔伯特空间（RKHS），又作为随机脉冲响应的协方差模型。目前已知的核条件可以精确刻画RKHS稳定性，而在高斯先验下的几乎必然BIBO稳定性则需要更为严苛的协方差条件。然而，这些限制在超越高斯性的情形下未必继续成立，因为此时协方差不再决定完整的概率分布规律。我们引入了“稳定性相容核”的概念，即一种协方差核，它允许至少一个具有几乎必然绝对可和样本路径的随机实现。我们证明，稳定性相容性当且仅当稳定元素在相应RKHS中稠密时成立，该条件严格弱于RKHS稳定性，并推导出了实用的充分条件，包括针对平稳核的谱判据。

    arXiv:2611.24272v1 Announce Type: cross  Abstract: Kernel-based regularization for linear system identification uses positive semidefinite kernels both to define reproducing kernel Hilbert spaces (RKHSs) and as covariance models for stochastic impulse responses. Sharp kernel conditions are known for RKHS stability, while almost-sure BIBO stability under Gaussian priors requires an even more restrictive covariance condition. These restrictions, however, need not persist beyond Gaussianity, where the covariance no longer determines the full probability law. We introduce the notion of a stability-compatible kernel, namely a covariance kernel admitting at least one stochastic realization with almost-surely absolutely summable sample paths. We show that stability-compatibility holds if and only if the stable elements are dense in the associated RKHS, a condition strictly weaker than RKHS stability, and derive practical sufficient conditions, including a spectral criterion for stationary ker
    
[^82]: TRESS：一种用于不规则时间序列预测的固定大小历史表示方法

    TRESS: A Fixed-Size History Representation for Irregular Time Series Forecasting

    [https://arxiv.org/abs/2611.24257](https://arxiv.org/abs/2611.24257)

    TRESS提出了一种固定大小的历史表示方法，通过保留带时间戳的近期观测值并摘要完整历史的统计信息，结合查询条件化的事件提升模块与通道特定解码器，在不规则时间序列预测上取得了最优的测试误差。

    

    固定数量的近期观测值对一个变量而言可能只跨越几分钟，而对另一个变量则可能跨越数天。较早的测量值仍能揭示最新值对该变量而言是否异常。我们提出了TRESS（带时间戳的近期事件与统计摘要），这是一种预测模型，它按原始时间保留最新的观测值，并对每个变量的完整历史进行摘要。对于给定的一组变量和事件预算，该表示的大小在记录不断增长时保持不变。其查询条件化的事件提升模块根据每个观测值的时间距离和预测时间范围，对相对于最新值的偏差进行加权。通道特定的解码器将所得特征与历史统计信息以及跨变量共享的摘要相结合，以预测对最新值的修正量。在为每个变量保留8到16个观测值的情况下，TRESS在所有四个基准数据集上比较的方法中取得了最低的平均测试MSE和MAE。

    arXiv:2611.24257v1 Announce Type: new  Abstract: A fixed number of recent observations may span minutes for one variable and days for another. Earlier measurements can still reveal whether the latest value is unusual for that variable. We propose TRESS (Timestamped Recent Events and Statistical Summaries), a forecasting model that retains the latest observations at their original times and summarises the full history of each variable. For a given set of variables and event budget, this representation stays the same size as the record grows. Its query-conditioned event lifting module weights deviations from the latest value according to the age of each observation and the forecast horizon. A channel-specific decoder combines the resulting features with historical statistics and a summary shared across variables to predict a correction to the latest value. Retaining 8 to 16 observations per variable, TRESS records the lowest mean test MSE and MAE among the methods compared on all four be
    
[^83]: ViLFlow：基于连续潜在流的并行视觉-语言生成

    ViLFlow: Parallel Vision-Language Generation with Continuous Latent Flows

    [https://arxiv.org/abs/2611.24240](https://arxiv.org/abs/2611.24240)

    提出ViLFlow，一种结合输出嵌入联合建模与并行token解码的连续流视觉-语言模型，通过图像引导的教师预测和预测感知解码，将预训练连续语言生成模型高效适配到视觉问答和图像描述任务。

    

    视觉-语言生成领域目前主要依赖自回归模型和离散扩散模型。自回归模型按顺序逐个生成token，而离散扩散模型通常在每个去噪步骤内独立预测token。我们提出ViLFlow（Vision-Language Flow），一种连续流的视觉-语言模型，它将输出嵌入的联合建模与并行token解码相结合。ViLFlow通过学习到的图像引导和预测感知的token解码，将预训练的连续语言生成模型适配到视觉问答和图像描述任务中。一个经过视觉适配的教师模型提供图像引导的预测，作为额外的去噪监督信号。去噪器学习直接重现这些预测，从而在每个生成步骤中保留一个以图像为条件的预测。我们在去噪器的预测结果与配对的参考文本上训练token解码器，以改进从连续输出到文字的转换。

    arXiv:2611.24240v1 Announce Type: cross  Abstract: Vision-language generation has predominantly relied on autoregressive and discrete diffusion models. Autoregressive models generate tokens sequentially, while discrete diffusion models typically predict tokens independently within each denoising step. We propose ViLFlow (Vision-Language Flow), a continuous-flow vision-language model that combines joint modeling of output embeddings with parallel token decoding. ViLFlow adapts pretrained continuous language generation to visual question answering and image captioning through learned image guidance and prediction-aware token decoding. A visually adapted teacher provides image-guided predictions as additional denoising supervision. The denoiser learns to reproduce these predictions directly, retaining one image-conditioned prediction per generation step. We train the token decoder on denoiser predictions and paired reference text to improve the conversion of continuous outputs into words.
    
[^84]: 函数主成分分析：从稀疏到稠密设计及更广泛情形的极小化极大最优性

    Functional Principal Component Analysis: Minimax Optimality from Sparse to Dense Designs and Beyond

    [https://arxiv.org/abs/2611.24228](https://arxiv.org/abs/2611.24228)

    本文首次在含噪离散观测下建立了函数主成分分析的完整极小化极大最优理论，覆盖了从稀疏到稠密随机设计中谱指标发散的特征函数与维数递增的主子空间，并给出了一般二元函数类上本质上二维的最优协方差估计速率。

    

    函数主成分分析（FPCA）作为函数数据分析中的降维与表示工具发挥着核心作用，但其在含噪离散观测下谱指标发散情形的极小化极大理论二十多年来一直悬而未决。为填补这一空白，我们考虑了一种基于筛（sieve）的FPCA框架，该框架汇集跨个体的离散测量数据，并能适应随机设计下从稀疏到稠密的各种采样频率。我们首先在超越张量积低复杂度结构的一般二元函数类上建立了协方差估计的新极小化极大速率，从而识别出本质上二维的最优速率。我们的核心理论贡献是在明确的增长与正则性条件下，为具有发散指标的逐一标记的特征函数以及维数不断增加的领先主子空间建立了极小化极大最优速率。

    arXiv:2611.24228v1 Announce Type: cross  Abstract: Functional principal component analysis (FPCA) plays a central role as a dimension-reduction and representation tool in functional data analysis, but its minimax theory for diverging spectral indices under noisy discrete observation has remained open for over two decades. To fill this gap, we consider a sieve-based FPCA framework that pools discrete measurements across subjects and accommodates sampling frequencies ranging from sparse to dense under random designs. We begin by establishing new minimax rates for covariance estimation over general bivariate function classes beyond tensor-product low-complexity structures, thereby identifying the intrinsically two-dimensional optimal rate. Our central theoretical contribution is to establish minimax optimal rates for both individually labelled eigenfunctions with diverging indices and leading principal subspaces of increasing dimensions, under explicit growth and regularity conditions. To
    
[^85]: TableControl：基于表面振动的手势交互

    TableControl: Gesture Interaction Through Surface Vibrations

    [https://arxiv.org/abs/2611.24218](https://arxiv.org/abs/2611.24218)

    本文提出TableControl系统，在微控制器上运行完整的处理流程，利用四个隐藏的压电传感器实现基于表面振动的连续手势识别，可识别敲击、上滑和下滑手势并驱动游戏和台灯等实际应用，且无需传输原始传感器信号。

    

    将手势感知集成到家具中，提供了一种通过熟悉的表面与数字应用和物理设备进行交互的方式。我们展示TableControl，一个基于连续表面振动的手势识别的交互式演示，其完整的处理流程运行在微控制器上。该系统使用四个隐藏的压电传感器，将信号采集与预处理同神经事件检测和手势分类相结合。识别出的敲击、上滑和下滑手势标签被传输到主机电脑，而无需传输底层传感器信号。访客通过动手实践探索两个示例应用：一个无尽跑酷游戏和一盏支持开关与亮度控制的实体台灯。仪表板实时显示识别出的手势、类别概率和处理时间。

    arXiv:2611.24218v1 Announce Type: cross  Abstract: Integrating gesture sensing into furniture offers a way to interact with digital applications and physical devices through familiar surfaces. We present TableControl, an interactive demonstration of continuous surface-vibration-based gesture recognition with the complete processing pipeline running on a microcontroller. Using four concealed piezoelectric sensors, the system combines signal acquisition and preprocessing with neural event detection and gesture classification. The resulting labels for knock, swipe-up, and swipe-down gestures are transmitted to a host computer without streaming the underlying sensor signals. Visitors explore two example applications through hands-on interaction: an endless runner game and a physical table lamp with on/off and brightness control. A dashboard displays recognized gestures, class probabilities, and processing times.
    
[^86]: 迈向可靠的表格基础模型缩放定律

    Towards Reliable Scaling Laws for Tabular Foundation Models

    [https://arxiv.org/abs/2611.24214](https://arxiv.org/abs/2611.24214)

    该研究揭示了表格基础模型在合成数据与真实数据上缩放行为的差异，指出准确拟合缩放定律并不能保证在未见数据集上的可靠预测，且使用过少的评估数据集选择配置会导致系统性过度乐观，而计算最优缩放需要扩大合成预训练数据集规模的搜索范围。

    

    缩放定律被广泛用于预测基础模型的性能并指导计算资源分配。然而，表格基础模型与传统基础模型的不同之处在于，它们是在来自可配置分布的几乎无限的合成数据上进行预训练的，而最终却在真实世界数据集上进行评估。我们研究了TabPFNv2风格模型的缩放定律能否可靠地外推到更大的计算预算，并在不同的评估设置之间泛化。我们的分析揭示了合成数据与真实世界数据之间缩放行为的差异，并表明准确拟合缩放定律并不一定能在未见过的数据集上给出可靠的指导建议。特别是，使用过少的评估数据集来选择配置会导致系统性地过度乐观的性能估计。我们进一步证明，实现计算最优的缩放需要扩大合成预训练数据集大小的范围。

    arXiv:2611.24214v1 Announce Type: cross  Abstract: Scaling laws are widely used to predict foundation model performance and guide compute allocation. However, tabular foundation models (TFMs) differ from conventional foundation models in that they are pretrained on virtually unlimited synthetic data from configurable distributions, while ultimately being evaluated on real-world datasets. We investigate whether scaling laws for TabPFNv2-style models reliably extrapolate to larger compute budgets and generalize across evaluation settings. Our analysis reveals differences between synthetic and real-world scaling behavior and shows that accurately fitting a scaling law does not necessarily yield reliable prescriptions on unseen datasets. In particular, selecting configurations using too few evaluation datasets leads to systematically over-optimistic performance estimates. We further demonstrate that compute-optimal scaling requires expanding the range of synthetic pretraining dataset sizes
    
[^87]: 连续噪声环模型

    Continuous Noisy-Loop Models

    [https://arxiv.org/abs/2611.24213](https://arxiv.org/abs/2611.24213)

    本文量化了“统计上高效可学”与“存在紧凑概率电路”之间的本质差距：对网格上相邻自旋一致性的连续含噪读数，样条估计器可在多项式时间内达到极小极大最优的 Hellinger 风险 $\Theta(\min\{1, dN^{-2/3}\})$，而紧凑的非负可分解电路则需 $\exp(\Theta(\sqrt d))$ 规模的节点，并给出了 KL 逼近误差的精确刻画。

    

    高效的学习与推理并不必然意味着存在紧凑的概率电路表示。本文针对已知方形网格（含 $d$ 条边）上相邻潜在自旋之间一致性的连续含噪读数，量化了这一区别。在已知边际参数且有界 Lipschitz 信道的条件下，基于 $N$ 个独立观测向量，极小极大平方 Hellinger 风险为 $\Theta(\min\{1, dN^{-2/3}\})$，且该结果不依赖任何可靠性下界假设。在给定固定正的可靠性下界时，样条估计器可在多项式算术时间内达到该速率；而指数规模的混合模型则无需该假设。在固定正的可靠性条件下，即使每个叶子节点最多使用 $\alpha d$ 个坐标（$0<\alpha<1$），非负可分解电路仍需要 $\exp(\Theta(\sqrt d))$ 个节点才能将总变差误差固定地控制在 1 以下。在节点数至多 $d^r$、叶子作用域 $\ell\le\alpha d$ 的约束下，两个方向上的最优 Kullback–Leibler（KL）逼近误差均为 $\Theta(d/\max\{\log d, \sqrt\ell\})$。

    arXiv:2611.24213v1 Announce Type: new  Abstract: Efficient learning and inference need not imply a compact probabilistic circuit. We quantify this distinction for continuous noisy readings of agreement between adjacent latent spins on a known square grid with $d$ edges. With known margins and bounded Lipschitz channels, minimax squared Hellinger risk is $\Theta(\min\{1,dN^{-2/3}\})$ from $N$ independent vectors, without a reliability lower bound. A spline estimator reaches this rate in polynomial arithmetic time given a fixed positive reliability bound; an exponential-size mixture removes that requirement. At fixed positive reliability, nonnegative decomposable circuits need $\exp(\Theta(\sqrt d))$ nodes for fixed total-variation error below one, even when each leaf uses at most $\alpha d$ coordinates, $0<\alpha<1$. Optimal Kullback--Leibler (KL) approximation error is $\Theta(d/\max\{\log d,\sqrt\ell\})$ in either direction with at most $d^r$ nodes and leaf scopes $\ell\leq\alpha d$, 
    
[^88]: GALA：面向真实场景EEG视觉解码的候选库级对齐方法

    GALA: Gallery-Level Alignment for Real-Life Vision Decoding from EEG

    [https://arxiv.org/abs/2611.24206](https://arxiv.org/abs/2611.24206)

    该论文提出GALA方法，并引入“增量直推式校准”新范式，利用预先已知的候选库和持续累积的未标注EEG数据进行无标签校准，从而解决EEG视觉解码难以跨个体泛化、依赖昂贵标注数据的瓶颈问题。

    

    基于脑电图（EEG）的视觉体验神经解码在个体内设置中已取得优异性能，但现有方法无法跨个体泛化：在标准的200类基准THINGS-EEG-2上，top-1准确率从个体内的80%以上下降到未见过的个体仅约32%。将预训练编码器适配到新受试者和新会话通常需要标注数据，而其收集时间的瓶颈严重限制了设备的实际部署。然而，在实际的脑机接口（BCI）应用中，可能的结果是预先已知的，构成一个候选库，并且一旦设备投入使用，用户的未标注EEG数据便会持续累积。我们提出了一个新的设置——增量直推式校准，其中解码器定期在迄今为止收到的所有未标注EEG数据以及已知的候选库上进行校准，全程无需任何标签。这一设置更接近实际设备部署的真实条件……

    arXiv:2611.24206v1 Announce Type: new  Abstract: Neural decoding of visual experience from electroencephalography (EEG) has achieved strong performance in intra-subject settings, but current methods fail to generalize across individuals: on the standard 200-way benchmark THINGS-EEG-2, top-1 accuracy drops from above 80% within subjects to about 32% for an unseen subject. Adapting a pre-trained encoder to new subjects and sessions typically requires labeled data, a bottleneck whose collection time severely limits practical device deployment. In a practical Brain-Computer Interface (BCI), however, the possible outcomes are known in advance and form a gallery of candidates, and the unlabeled EEG of the user accumulates as soon as the device is used. We propose a new setting, incremental transductive calibration, in which the decoder is calibrated regularly on all the unlabeled EEG received so far and on the known gallery, without any labels. This setting is closer to the conditions of a d
    
[^89]: 学习与遗忘并行：面向持续学习与机器遗忘同时实现的突触角色划分

    Learning with Unlearning: Synaptic Role Partitioning for Simultaneous Continual Learning and Machine Unlearning

    [https://arxiv.org/abs/2611.24205](https://arxiv.org/abs/2611.24205)

    提出了统一框架LwU，通过在突触层面对每个权重进行“保留”与“删除”双重评分，将网络划分为四个具有独立更新规则的区域，从而在同一模型中同时化解持续学习与机器遗忘之间的冲突。

    

    持续学习（CL）要求模型在不遗忘已有知识的前提下获取新知识。机器遗忘（MU）则要求模型在不损害其余知识的情况下销毁特定知识。挑战在于将二者集成到同一系统中：这两个目标作用于相同的权重，为其中一个目标所做的修改会损害另一个目标。为解决这一问题，我们提出了Learning with Unlearning（LwU），这是一个在冲突发生的源头——即单个突触层面——化解冲突的统一框架。LwU对每个权重进行两次评分，一次针对要保留的数据，一次针对要删除的数据，并将网络划分为四个具有独立更新规则的区域。仅需保留的权重被冻结；仅需删除的权重被清除并作为自由容量返还给网络；存在冲突的权重则沿与保留梯度正交的方向进行梯度上升更新；剩余容量则通过特权……学习新任务（原文摘要此处被截断）。

    arXiv:2611.24205v1 Announce Type: new  Abstract: Continual Learning (CL) asks a model to acquire new knowledge without forgetting what it already knows. Machine Unlearning (MU) asks it to destroy specific knowledge without damaging the rest. The challenge is integrating them in the same system; the two objectives pull on the same weights, and an edit made for one harms the other. To address this, we propose Learning with Unlearning (LwU), a unified framework that resolves the conflict where it occurs: at the individual synapse. LwU scores every weight twice, once against the data to keep and once against the data to delete, and partitions the network into four zones with separate update rules. Weights that only require retention are frozen. Weights that only require deletion are cleared and returned to the network as free capacity. Contested weights take a gradient-ascent step projected orthogonally to the retention gradient. The remaining capacity learns the incoming task by a privile
    
[^90]: 值得再看一眼：面向视觉检查的受限世界模型规划

    Worth Another Look:Constrained World-Model Planning for Visual Inspection

    [https://arxiv.org/abs/2611.24187](https://arxiv.org/abs/2611.24187)

    该论文将冻结检测器上的视觉检查形式化为部分可观测下的神经符号观测规划，通过紧凑循环世界模型与滚动时域束搜索在线联合调度查询、裁剪、增强和终止，在严格预算下比静态调度和无搜索离线RL最高提升28.4个AP百分点。

    

    野外机器人学和环境监测中的视觉检查需要在严格的计算与感知预算限制下定位多个目标。穷举式扫描会在不存在的类别上浪费调用，而短视策略和无搜索策略则会卡在异构的费率陡坡上，或者丢弃那些只有在重新查询后才能显现价值的视口编辑操作。我们将基于冻结检测器的检查问题形式化为部分可观测性下的神经符号观测规划，将一个紧凑的循环世界模型（WM）与精确的符号可行性判定及费率规则解耦。带有在线重规划的滚动时域束搜索联合调度类别查询、裁剪、图像增强和终止决策，并配合感知松弛的影子定价以及在更大预算下的合法回退方案。在八个陆地、光学和声呐基准测试中，多步前瞻规划在紧张预算下优于静态调度和无搜索的离线强化学习——最高可获得28.4个AP百分点。

    arXiv:2611.24187v1 Announce Type: cross  Abstract: Visual inspection in field robotics and environmental monitoring must localize multiple targets under strict compute and sensing caps. Exhaustive sweeps waste calls on absent classes, while myopic and search-free policies stumble over heterogeneous tariff cliffs or discard viewport edits whose value emerges only after re-querying. We formulate inspection over frozen detectors as neuro-symbolic observation planning under partial observability, decoupling a compact recurrent world model (WM) from exact symbolic feasibility and tariff rules. Receding-horizon beam search with online replanning jointly schedules class queries, crops, enhancements, and termination (stop), paired with slack-aware shadow pricing and legal fallback at larger budgets. Across eight terrestrial, optical, and sonar benchmarks, multi-step lookahead outperforms static schedules and search-free offline RL under tight budgets -- gaining up to 28.4 AP percentage points 
    
[^91]: 面向带标签噪声的偏置高斯半空间学习的稀有尾部统计方法

    Rare-Tail Statistics For Learning Biased Gaussian Halfspaces with Label Noise

    [https://arxiv.org/abs/2611.24181](https://arxiv.org/abs/2611.24181)

    本文提出“稀有尾部均值估计量”这一极简统计方法，在大偏置高斯半空间带标签噪声的学习问题中，以信息论最优样本率恢复目标半空间方向，超越了经典平均法等全局一阶矩统计量。

    

    我们重新审视了在标签噪声存在下学习一般高斯半空间的经典问题，并特别关注大偏置区间——在该区间中，若干标准标签噪声模型被猜想存在信息与计算之间的差距。具体而言，我们提出并分析了稀有尾部均值估计量，这是一种极其简单的一次性统计量，仅对观测到的少数类进行平均。在大偏置区间中，经典的平均法（Servedio, 1999）及相关的全局一阶矩统计量被证明是次优的，而稀有尾部均值估计量则在广泛的标签噪声模型类下，以信息论最优的样本率（除去对数因子）恢复目标半空间方向，这些模型包括概率单位噪声以及远离决策边界时衰减极慢的间隔相关噪声。值得注意的是，我们的理论保证允许任意接近决策边界的标签近乎完全随机，同时仍能实现强大的方向恢复能力。

    arXiv:2611.24181v1 Announce Type: cross  Abstract: We revisit the classical problem of learning general Gaussian halfspaces in the presence of label noise, with particular focus on the large-bias regime, where information-computation gaps are conjectured in several standard label-noise models. Concretely, we present and analyze the rare-tail mean estimator, an extremely simple one-shot statistic that averages only the observed minority class. In the large-bias regime, classical Averaging (Servedio, 1999) and related global first-moment statistics are provably suboptimal, while the rare-tail mean estimator recovers the target halfspace direction at the information-theoretic sample rate (modulo log factors) under a broad class of label noise models, including probit noise and margin-dependent noise with very slow decay away from the boundary. Notably, our guarantees allow labels arbitrarily close to the decision boundary to be nearly random, while still giving strong directional recovery
    
[^92]: 基于流场不确定性的自适应动作视界选择

    Adaptive Action Horizon Selection via Flow-Field Uncertainty

    [https://arxiv.org/abs/2611.24178](https://arxiv.org/abs/2611.24178)

    本文提出DUAC，一种无需重训练的事后不确定性估计方法，通过预训练VLA模型向量场的闭式流后验协方差和雅可比-向量积，仅需单个动作块即可获得动作级不确定性分数，从而实现自适应动作视界选择并避免多次采样的推理开销。

    

    可靠的不确定性估计对于视觉-语言-动作（VLA）模型的自适应执行至关重要。现有方法如自适应动作分块需要从多个采样的动作块中估计动作熵，这引入了额外的推理开销，并同时捕获了动作轨迹内部的不确定性与不同备选策略之间的多样性。我们提出了基于散度的不确定性感知自适应动作分块，这是一种事后方法，能够从预训练VLA模型的向量场中提取以轨迹为条件的预测不确定性。基于闭式流后验协方差公式，DUAC利用雅可比-向量积从单个生成的动作块中推导出动作级的不确定性分数，无需重新训练或多次端点采样。DUAC与AAC具有很强的相关性，同时捕获了不同的信息：AAC度量的是采样动作块之间的全局离散度，而DUAC度量的是局部不确定性。

    arXiv:2611.24178v1 Announce Type: new  Abstract: Reliable uncertainty estimates are important for adaptive execution in vision-language-action (VLA) models. Existing approaches such as Adaptive Action Chunking (AAC) estimate action entropy from multiple sampled action chunks, introducing additional inference cost and capturing both uncertainty within an action trajectory and diversity across alternative strategies. We introduce Divergence-based Uncertainty-aware Adaptive Action Chunking (DUAC), a post-hoc method that extracts trajectory-conditioned predictive uncertainty from the vector field of a pretrained VLA. Building on a closed-form flow-posterior covariance formulation, DUAC derives action-level uncertainty scores from a single generated chunk using Jacobian-vector products, without retraining or multiple endpoint samples. DUAC strongly correlates with AAC, while capturing distinct information: AAC measures global dispersion across sampled chunks, whereas DUAC measures local unc
    
[^93]: SoU-RHS：用于卷积神经网络紧凑架构恢复的贝叶斯结构化稀疏方法

    SoU-RHS: Bayesian Structured Sparsity for Compact Architecture Recovery in CNNs

    [https://arxiv.org/abs/2611.24166](https://arxiv.org/abs/2611.24166)

    该论文提出SoU-RHS，一种将正则化马蹄铁收缩与不确定性来源稀疏全局-局部机制相结合的贝叶斯结构化稀疏方法，能够稳定地从过参数化CNN中恢复出规模一致且适应数据的紧凑架构。

    

    高度参数化的卷积神经网络通常能取得很强的预测性能，但其内存、计算和部署成本促使人们寻找从大型参考模型中恢复紧凑架构的方法。我们提出了SoU-RHS，这是一种贝叶斯结构化稀疏方法，它在正则化马蹄铁收缩的基础上增加了所谓的不确定性来源稀疏全局-局部机制。所得模型锐化了活跃滤波器与非活跃滤波器之间的区分，为架构恢复提供了有效机制。SoU-RHS的一个关键特性是稳定性：在不同的过参数化起始宽度以及合理的剪枝阈值范围内，它都能恢复出总规模和逐层分配相似的紧凑架构。这支持了这样一种解读：该过程识别出的是适应数据的紧凑结构，而不仅仅是简单地对各层进行压缩。

    arXiv:2611.24166v1 Announce Type: new  Abstract: Highly parameterized convolutional neural networks often achieve strong predictive performance, but their memory, computation, and deployment costs motivate the search for methods that recover compact architectures from large reference models. We propose SoU-RHS, a Bayesian structured-sparsity procedure that augments Regularized Horseshoe Shrinkage (RHS) with the so-called Sources of Uncertainty Sparse Global-Local mechanism (SoU-SGL). The resulting model sharpens the separation between active and inactive filters, providing an effective mechanism for architecture recovery. A key property of SoU-RHS is stability: across different overparameterized starting widths and under a reasonable range of pruning thresholds, it recovers compact architectures with similar total size and layer-wise allocation. This supports the interpretation that the procedure is identifying a compact structure adapted to the data, rather than merely compressing eac
    
[^94]: SGD最后迭代点几乎必然收敛速率的近乎紧致结果

    Nearly Tight Almost Sure Convergence Rates for the Last iterate of SGD

    [https://arxiv.org/abs/2611.24151](https://arxiv.org/abs/2611.24151)

    该论文为SGD最后迭代点的几乎必然收敛建立了近乎紧致的速率界——光滑情形下分别为 $k^{-1/3}$（存在稳定极小点）和 $k^{-1/4}$（且后者可推广至非光滑凸情形），并通过匹配下界证明这些指数在相差多对数因子的意义下是最优的。

    

    我们研究了随机（次）梯度下降（SGD）在凸目标函数上最后迭代点优化误差的几乎必然收敛性。对于光滑目标函数，当存在稳定极小值点时，我们建立了 $k^{-1/3}$ 阶的收敛速率，否则为 $k^{-1/4}$ 阶（在相差多对数因子的意义下），其中 $k$ 表示迭代计数。我们进一步证明后一种速率可以推广到非光滑凸目标函数。所有这些速率均通过确定性的、随时可用的步长策略实现。最后，我们建立了匹配的下界，表明指数 $-1/3$ 和 $-1/4$ 在相差多对数因子的意义下，对于相应的问题类别是最优的。

    arXiv:2611.24151v1 Announce Type: cross  Abstract: We study the almost sure convergence of the last-iterate optimization error of Stochastic (sub)Gradient Descent (SGD) for convex objectives. For smooth objectives, we establish convergence rates of order $k^{-1/3}$ when a stationary minimizer exists, and $k^{-1/4}$ otherwise, up to polylogarithmic factors, where $k$ denotes the iteration counter. We further show that the latter rate extends to nonsmooth convex objectives. All these rates are achieved by deterministic, anytime stepsize policies. Finally, we establish matching lower bounds showing that the exponents $-1/3$ and $-1/4$ are optimal, up to polylogarithmic factors, for the corresponding problem classes.
    
[^95]: 基于填充距离引导的主动学习实现网络干扰下的因果函数恢复

    Fill-Distance-Guided Active Learning for Causal Function Recovery under Network Interference

    [https://arxiv.org/abs/2611.24143](https://arxiv.org/abs/2611.24143)

    该论文提出FANCI方法，在有限实验预算下，利用填充距离引导的主动学习设计与特定处理的建模非线性响应曲面，实现网络干扰环境中完整的暴露依赖总体平均响应函数恢复。

    

    因果推断旨在学习结果在不同干预下如何变化，但网络交互使这一任务变得复杂。在网络干扰下，结果同时取决于自身的处理和邻居的处理，这使得因果效应成为邻域暴露的函数，而非单一处理之间的对比。现有方法主要针对预先指定的因果对比或其估计精度，而对全局响应函数恢复的研究几乎处于空白状态。我们研究了有限实验预算下的序贯实验设计问题，目标是恢复完整的、依赖于暴露的总体平均响应函数。为此，我们提出了FANCI（填充距离引导的网络干扰因果推断主动学习方法）。在函数学习方面，针对特定处理的高斯过程灵活地建模非线性的条件响应曲面。在实验设计方面，我们将全局响应函数恢复表述为……

    arXiv:2611.24143v1 Announce Type: cross  Abstract: Causal inference seeks to learn how outcomes change under alternative interventions, but network interactions complicate this task. Under network interference, outcomes depend on both own treatment and neighbors' treatments, making causal effects functions of neighborhood exposure rather than single treatment contrasts. Existing methods primarily target prespecified causal contrasts or their estimation precision, leaving global response-function recovery largely unexplored. We study sequential experimentation under a limited experimental budget, with the goal of recovering the full exposure-dependent population-average response function. To this end, we propose FANCI (Fill-Distance-Guided Active Learning for Network-Interference Causal Inference). For function learning, treatment-specific Gaussian processes flexibly model nonlinear conditional response surfaces. For experimental design, we formulate global response-function recovery as
    
[^96]: QuantAtlas-OM：驯服长时程时空海洋动力学中的神经频谱偏差

    QuantAtlas-OM: Taming Neural Spectral Bias in Long-Horizon Spatiotemporal Ocean Dynamics

    [https://arxiv.org/abs/2611.24124](https://arxiv.org/abs/2611.24124)

    提出尺度-频谱解耦框架QuantAtlas-OM，将时间演化约束在低波数潜在流形以保证长时程稳定性，同时通过外部旁路保留全分辨率涡旋与锋面残差，从而解决自回归海洋预测中的神经频谱偏差问题。

    

    自回归神经网络在长时程时空预测中表现出频谱偏差。随着滚动预测的推进，高频细节逐渐衰减，导致预测场变得模糊，并使长期预报失去稳定性。海洋动力学为这一难题提供了尤为严苛的测试场景，因为其动能集中在跨度达数十公里的中尺度涡旋和锋面上。这种性能退化源于大尺度相位演化与小尺度能量保持被耦合在同一个共享特征空间中：在粗网格上演化会平滑掉高波数的涡旋，而在细网格上演化则会放大小尺度相位误差并引发滚动预测的不稳定性。我们提出QuantAtlas-OM（QuantAtlas海洋模型），这是一个尺度-频谱解耦框架，它将深度时间演化限制在低波数潜在流形上以保证长期稳定性，同时通过外部旁路将全分辨率的涡旋和锋面残差传递至解码器。

    arXiv:2611.24124v1 Announce Type: new  Abstract: Autoregressive neural networks exhibit spectral bias in long-horizon spatiotemporal prediction. As rollouts progress, high-frequency details decay, blurring the predicted fields and destabilizing long-range forecasts. Ocean dynamics offer a particularly stringent test because kinetic energy is concentrated in mesoscale eddies and fronts spanning tens of kilometers. This degradation stems from coupling large-scale phase evolution and small-scale energy retention in a shared feature space. Evolution on a coarse grid smooths out high-wavenumber eddies, whereas evolution on a fine grid amplifies small-scale phase errors and triggers rollout instability. We introduce QuantAtlas-OM (QuantAtlas Ocean Model), a scale-spectral decoupling framework that confines deep temporal evolution to a low-wavenumber latent manifold for long-term stability while routing full-resolution eddy and front residuals through an outer bypass to the decoder. Across ph
    
[^97]: 第七届国际神经网络验证竞赛（VNN-COMP 2026）：总结与结果

    The 7th International Verification of Neural Networks Competition (VNN-COMP 2026): Summary and Results

    [https://arxiv.org/abs/2611.24114](https://arxiv.org/abs/2611.24114)

    第七届国际神经网络验证竞赛（VNN-COMP 2026）通过统一的标准格式、标准化硬件环境和作者指定的工具参数，对最先进的神经网络验证工具在可扩展性和速度方面进行了客观比较，并总结了竞赛规则、基准测试、结果与经验教训。

    

    本报告总结了第七届国际神经网络验证竞赛（VNN-COMP 2026），该竞赛作为人工智能验证研讨会（SAIV）的一部分，与计算机辅助验证国际会议（CAV）共同在葡萄牙里斯本举办的FLoC 2026会议上举行。竞赛的目标是对最先进的神经网络验证方法在可扩展性和速度方面进行客观比较。为此，竞赛采用了标准格式（神经网络使用ONNX，规约使用VNNLIB）、标准硬件环境（所有工具均由组织者在AWS上运行）以及由工具作者提供的工具参数。本报告总结了本次竞赛的规则、基准测试、参赛工具、比赛结果以及经验教训。

    arXiv:2611.24114v1 Announce Type: cross  Abstract: This report summarizes the seventh International Verification of Neural Networks Com- petition (VNN-COMP 2026), held as part of the Symposium on AI Verification (SAIV) co-located with the International Conference on Computer-Aided Verification (CAV) at FLoC 2026 in Lisbon, Portugal. The goal of the competition is to provide an objective comparison of the state-of-the-art methods in neural network verification, in terms of scal- ability and speed. Along this line, we used standard formats (ONNX for neural networks and VNNLIB for specifications), standard hardware (all tools are run by the organizers on AWS), and tool parameters provided by the tool authors. This report summarizes the rules, benchmarks, participating tools, results, and lessons learned from this competition.
    
[^98]: AG-BFGS：加速几何学习的基础

    AG-BFGS: Foundations for Accelerated Geometry Learning

    [https://arxiv.org/abs/2611.24113](https://arxiv.org/abs/2611.24113)

    该论文提出了AG-BFGS加速几何学习框架，将动量引入BFGS的Hessian逆近似学习中，首次为加速拟牛顿方法建立了严格的理论基础，证明了加速更新满足显式稳定性条件且保持正定性。

    

    我们为BFGS引入了一个加速几何学习框架，该框架将动量融入Hessian逆近似的学习过程中。我们证明了所得到的加速更新满足显式的稳定性条件，并在可允许的参数区域内保持正定性。数值实验表明，与相应的非加速方案相比，该方法取得了一致的改进，同时保持了数值稳定性。这些结果为加速拟牛顿学习提供了首个严格的理论基础，并弥合了加速二阶方法在经验观察与数学保证之间的差距。

    arXiv:2611.24113v1 Announce Type: cross  Abstract: We introduce an accelerated geometry-learning framework for BFGS that incorporates momentum into the learning of Hessian inverse approximations. We prove that the resulting accelerated updates satisfy explicit stability conditions and preserve positive definiteness within admissible parameter regions. Numerical experiments demonstrate consistent improvements over the corresponding non-accelerated schemes while maintaining numerical stability. These results provide the first rigorous theoretical foundation for accelerated quasi-Newton learning and bridge the gap between empirical observations and mathematical guarantees for accelerated second-order methods.
    
[^99]: UCLIR：基于独立表示的无监督持续学习

    UCLIR: Unsupervised Continual Learning using Independent Representations

    [https://arxiv.org/abs/2611.24093](https://arxiv.org/abs/2611.24093)

    该论文提出UCLIR方法，受大脑指数距离规则启发，通过对多个独立学习的小型互补表示分别施加权重正则化，有效解决了持续学习中随任务增加旧知识保留效果下降的问题。

    

    在持续学习中，基于正则化的方法旨在通过更新网络权重来学习新任务，同时通过惩罚项约束对先前学习任务重要的参数的变化，从而模拟神经可塑性。然而，随着任务数量的增加，对早期任务知识的保护效果会逐渐减弱。这主要是因为新任务的更新分布在整个网络中，而由于神经元之间密集的连接，参数高度相互依赖。受指数距离规则的启发——该规则指出神经连接的数量随物理距离的增加而减少——学习在本质上具有局部相关性，而区域间的更新则较少依赖于特定任务。基于这一原理，本文证明了对多个小型且独立学习的互补表示应用权重正则化是更为有效的……

    arXiv:2611.24093v1 Announce Type: new  Abstract: In continual learning, regularization-based methods aim to mimic neuroplasticity by updating network weights to learn new tasks while constraining changes to parameters that are important for previously learned tasks through penalty terms. However, as the number of tasks increases, preserving knowledge from earlier tasks becomes less effective. This is mainly because updates for new tasks are distributed across the entire network, where parameters are highly interdependent due to the dense connectivity between neurons. Inspired by the Exponential Distance Rule (EDR), which states that the number of neural connections decreases with increasing physical distance, learning is naturally locally correlated while inter-area updates remain less task-dependent. Motivated by this principle, in this paper we demonstrate that applying weight regularization to multiple small and independently learned complementary representations is more effective t
    
[^100]: 基于量子神经网络的量子退火：退火的宽度至关重要

    Quantum Annealing with a Quantum Neural Network: Width of Annealing Matters

    [https://arxiv.org/abs/2611.24089](https://arxiv.org/abs/2611.24089)

    本文将量子退火扩展至量子神经网络，发现宽度仅为 2 的量子退火网络即可在量子搜索问题中指数级地降低退火深度，且通过调节参数，深度为 1、宽度为 2 的量子感知机便能以指数级小的误差完成求解。

    

    量子退火是一种有前景的量子算法，可用于处理计算上困难的优化问题。目前可用的硬件已能针对一大类哈密顿量问题实现该算法。然而，其性能在不连续量子相变处尤为受限，此时能隙随系统规模呈指数级收缩。在本文中，我们提出了将量子退火扩展到量子神经网络的方法，并在若干典型优化问题上研究了其性能。特别地，我们观察到仅宽度 W = 2 的随机量子退火网络就能在量子搜索问题中以指数级方式减少演化层数或退火过程的深度。通过调节网络参数，深度 L = 1、宽度 W = 2 的量子感知机能够以指数级小的误差求解该搜索问题。这些发现表明，量子退火网络的宽度至关重要。

    arXiv:2611.24089v1 Announce Type: cross  Abstract: Quantum annealing is a promising quantum algorithm for dealing with computationally hard optimization problems. Currently available hardware can implement the algorithm for a significant class of Hamiltonian problems. However, its performance is especially limited in discontinuous quantum phase transitions, where the energy gap shrinks exponentially with system size. In this paper, we introduce an extension of quantum annealing to a quantum neural network and study its performance in a number of prototypical optimization problems. In particular, we observe that already a random quantum annealing network of width W = 2 can exponentially reduce the number of evolution layers or depth of the annealing process in the quantum search problem. By tuning the network parameters the search problem can be solved with exponentially small errors by a quantum perceptron of depth L = 1 and width W = 2. The findings show that width of a quantum anneal
    
[^101]: VESPA：面向安全强化学习的可验证符号策略自适应

    VESPA: Verifiable Symbolic Policy Adaptation for Safe Reinforcement Learning

    [https://arxiv.org/abs/2611.24083](https://arxiv.org/abs/2611.24083)

    VESPA提出了一种利用形式化验证产生的反例对决策树策略进行定向微调的框架，能够高效地将不安全策略调整为安全策略，避免了从头重训与反复验证的低效循环。

    

    安全强化学习（RL）旨在防止智能体采取对其环境危险或具有破坏性的行动，这对于机器人或自动驾驶等安全关键领域尤为重要。形式化验证可以通过安全约束来证明智能体的安全性。然而在实践中，许多策略确实会违反安全约束，从而被验证为不安全。由于目标是获得一个安全的策略，我们往往需要从头重新训练一个新策略，然后再次进行验证，直到某个候选策略恰好通过为止。训练和验证通常都不便宜，这使得这一过程效率低下。我们提出了VESPA（可验证符号策略自适应），这是一个针对决策树策略进行定向微调的框架。我们展示了利用反例进行策略自适应可以使现有的不安全策略变得安全，或者至少使考虑……

    arXiv:2611.24083v1 Announce Type: new  Abstract: Safe reinforcement learning (RL) aims to prevent agents from taking actions that are dangerous or destructive to their environment. This is especially important for safety-critical domains, such as robotics or autonomous driving. Formal verification can be used to prove agent safety through safety constraints. However, in practice, many policies do violate safety constraints, and thus are verified to be unsafe. Since the goal is to obtain a safe policy, we would need to retrain a new policy from scratch and then verify it again until a candidate happens to pass. Both training and verification are usually not cheap, which makes this process inefficient. We introduce VESPA ($\textbf{VE}$rifiable $\textbf{S}$ymbolic $\textbf{P}$olicy $\textbf{A}$daptation), a framework for targeted fine-tuning of decision tree policies. We show that using counterexamples for policy adaptation can make existing unsafe policies safe, or at least make consider
    
[^102]: 语义能影响物体选择却无法影响任务完成：VLA在何处易受地标同义词的攻击？

    Meaning Reaches Object Choice but Not Completion: Where Is a VLA Vulnerable to Landmark Synonyms?

    [https://arxiv.org/abs/2611.24076](https://arxiv.org/abs/2611.24076)

    研究发现VLA策略能够理解训练指令中未出现过的地标同义词并将其语义用于物体选择（该语义作用点位于视觉-语言模型与动作生成器之间的连接器之前），但这种措辞改写会显著降低任务完成率，且连接器之后无法通过单独注入单个模态的token来改变选择。

    

    VLA（视觉-语言-动作）策略对措辞改写十分脆弱。仅凭成功率无法判断策略究竟是没有选中正确的物体，还是无法完成任务，也无法指出应在策略的哪个环节进行干预。我们在两个仅在“取两个碗中的哪一个”的地标词上存在差异的LIBERO-Spatial任务上研究了公开发布的GR00T N1.7。与虚构的词汇不同，训练指令中从未出现过的同义词会将策略引导至其所指代的碗，但在一项任务的闭环测试中仅完成67.6%的回合，而使用原始词汇的完成率为100%。在视觉-语言模型与动作生成器之间的连接器之前，仅用地标的token向量替换为另一个地标的向量（粘贴操作），几乎使所有回合都走向另一个碗。而在连接器之后，仅对地标token、其余指令token或图像token单独进行粘贴很少改变物体选择，但任意两者结合则会改变选择。在连接器之后对动作生成器隐藏地标token，仍会降低任务完成率。

    arXiv:2611.24076v1 Announce Type: cross  Abstract: VLA policies are fragile to rewording. Success rates alone cannot say whether a policy fails to choose the object or to complete the task, nor where in the policy to intervene. We study the released GR00T N1.7 on two LIBERO-Spatial tasks differing only in the landmark word telling which of two bowls to take. Synonyms absent from the training instructions send the policy to the bowl they name, unlike made-up words, yet complete 67.6% of one task's closed-loop episodes, against the original word's 100%. Before the connector between the vision-language model and the action generator, replacing only the landmark tokens' vectors with the other landmark's (pasting) sends nearly every episode to the other bowl. After the connector, pasting at the landmark, the remaining instruction, or the image tokens alone rarely moves the choice, but any two do. Hiding the landmark tokens from the action generator after the connector still lowers completio
    
[^103]: 时间序列基础模型的循环域增量自适应

    Recurrent Domain Incremental Adaptation of Time-Series Foundation Models

    [https://arxiv.org/abs/2611.24075](https://arxiv.org/abs/2611.24075)

    该论文提出PACT方法，将冻结的嵌入空间划分为局部图卡并进行轻量级的分段残差自适应，从而在运行条件不断变化的情况下实现时间序列基础模型的域增量自适应，同时避免全局自适应破坏已正确分类的嵌入表示。

    

    预训练的时间序列基础模型（TSFMs）为下游分类任务提供了强大的表示能力。然而，在不断变化的运行条件下对TSFM进行自适应仍然具有挑战性。模型必须在每个新观测到的运行条件上取得改进，同时不损害其已经学习过的条件，而且往往只有一个很小的带标签校准集可用，并且在推理时无法得知当前的运行条件。我们发现这一困难背后存在重要的几何结构。在运行条件发生偏移时，TSFM的嵌入空间并不是均匀移动的：某些局部区域变化显著，而其他区域几乎保持不变。因此，全局式的自适应可能会不必要地移动那些本已被正确分类的嵌入。为了解决这一问题，我们提出了新颖的PACT（通过图卡传输的分段自适应）方法。PACT将冻结的源嵌入空间划分为局部图卡，并学习轻量级的残差变换……

    arXiv:2611.24075v1 Announce Type: new  Abstract: Pretrained Time-Series Foundation Models (TSFMs) provide strong representations for downstream classification tasks. Yet adapting a TSFM under changing operating conditions remains challenging. A model must improve on each newly observed condition without sacrificing conditions it has already learned, often from only a small labeled calibration set and without knowing the operating condition at inference time. We find that this difficulty has an important geometric structure. Under an operating-condition shift, the embedding space of a TSFM does not move uniformly: some local regions change substantially, while others remain nearly unchanged. Consequently, a global adaptation can unnecessarily move embeddings that were already correctly classified. To address this, we propose the novel PACT (Piecewise Adaptation via Chart Transport). PACT partitions the frozen source embedding space into local charts and learns lightweight residual trans
    
[^104]: FDM+：一个用于二维辐射输运方程受限控制的框架

    FDM+: A Framework for Constrained Control of the 2D Radiative Transport Equation

    [https://arxiv.org/abs/2611.24070](https://arxiv.org/abs/2611.24070)

    该论文提出FDM+框架，通过解耦预测架构、带总误差界分析的高效参数化策略以及基于条件分布的设计空间探索，解决了冻结扩散模型在二维辐射输运方程受限控制中维度差异、分辨率不匹配与多解性三大难题。

    

    将冻结扩散模型（FDM）应用于二维辐射输运方程的受限控制面临诸多挑战，包括控制变量与状态函数之间的维度差异、传统数值求解器与深度学习模型之间的分辨率不匹配，以及解的非唯一性。为克服这些障碍，我们提出了一种新颖的方法FDM+，其中包含三项关键改进。首先，引入解耦预测架构以应对维度异质性问题。其次，采用高效的参数化策略解决分辨率不匹配问题；同时，我们推导了总误差界，用以分析参数化误差与模型估计误差如何共同影响性能。第三，利用扩散模型学习到的条件分布对多解问题进行设计空间探索。一系列数值实验……

    arXiv:2611.24070v1 Announce Type: new  Abstract: Applying the frozen diffusion model (FDM) [29] to the constrained control of the 2D radiative transport equation presents several challenges. These include the dimensional difference between control variables and state functions, the resolution mismatch between traditional numerical solvers and deep learning models, and solution non-uniqueness. To address these obstacles, we propose a novel approach, FDM+, which includes three key improvements. Firstly, a decoupled prediction architecture is introduced to address the dimensional heterogeneity. Secondly, an efficient parameterization strategy resolves the resolution mismatch. Meanwhile, we derive a total error bound to analyze how the parameterization error and the model estimation error jointly affect the performance. Thirdly, the conditional distribution learned by the diffusion model is leveraged to perform design space exploration for multi-solution problems. A series of numerical exp
    
[^105]: 重放而非重采样：长时程强化学习专家的离策略整合

    Replay, Don't Resample: Off-Policy Consolidation of Long-Horizon RL Experts

    [https://arxiv.org/abs/2611.24059](https://arxiv.org/abs/2611.24059)

    提出多专家回放策略蒸馏（MRPD），通过按专家的学习顺序回放其RL训练日志（而非代价高昂的重新采样）来整合多个长时程强化学习专家，从而大幅降低整合成本并缓解训练与推理的不匹配。

    

    前沿模型的后训练遵循一个通用方案：先用强化学习（RL）在每个领域分别训练一个专家模型，然后通过多教师在线策略蒸馏（MOPD）将这些专家整合为一个模型。这种整合方式代价高昂：学生模型需要在每个环境中重新采样，专家需要对每个采样的token进行打分，而且学生会继承最终专家模型的漂移。然而我们认为，专家的RL训练过程其实已经产生了整合所需的一切内容：带有采样概率和奖励的采样轨迹，以及专家从中学习的先后顺序。我们提出多专家回放策略蒸馏（MRPD），它按照专家自身的学习顺序将这些训练日志回放给学生模型。在每一步，策略梯度在这些已被打分的轨迹上重新计算，最终专家模型的分布则在被记录的前缀处进行蒸馏。遵循专家自身的轨迹减少了训练与推理之间的不匹配，并保持了……（原文摘要在此截断）

    arXiv:2611.24059v1 Announce Type: cross  Abstract: Frontier post-training follows a common recipe: train one expert per domain with reinforcement learning (RL), then consolidate the experts into one model via multi-teacher on-policy distillation (MOPD). Such consolidation is expensive: the student samples in every environment again, an expert scores every sampled token, and the student inherits the drift of the final experts. Yet we argue that the experts' RL runs have already produced what consolidation needs: rollouts with their sampling probabilities and rewards, in the order the experts learned from them. We propose multi-expert replay policy distillation (MRPD), which replays these training logs through the student in the experts' order. At each step, the policy gradient is recomputed on these scored trajectories, and the final experts' distributions are distilled at the logged prefixes. Following the experts' own trajectory reduces the training-inference mismatch and keeps the st
    
[^106]: 更快地得到错误答案：联邦优化中局部更新的代价

    Faster to the Wrong Answer: The Price of Local Updates in Federated Optimization

    [https://arxiv.org/abs/2611.24057](https://arxiv.org/abs/2611.24057)

    该论文刻画了联邦学习中在给定条件数约束下局部更新可达到的精确最坏情况偏差，揭示了局部计算加速与更新偏差之间的根本权衡。

    

    局部计算可以改善联邦更新的条件数，但也可能引入偏差，导致其不动点偏离全局最小值点。我们刻画了在给定条件数约束下可达到的最小最坏情况偏差。我们考虑了Hessian特征值位于区间[μ,L]内的强凸二次型客户端目标函数，并将局部更新表示为共享的正谱滤波器。客户端异质性通过全局最小值点处的均方根梯度来度量。对于由χ∈[1,L/μ]界定的统一代理条件数，我们建立了单位异质性下的精确最坏情况偏差 (L−μχ)/(2μL√χ)，并证明步长足够小的局部梯度下降和近端更新可以达到该界。我们的分析将任意维度的客户端群体（包括Hessian矩阵不可交换的情形）归约为标量曲率上的分布。这一归约提供了最优……

    arXiv:2611.24057v1 Announce Type: new  Abstract: Local computation can improve the conditioning of federated updates but may also introduce bias, causing their fixed point to differ from the global minimizer. We characterize the minimum worst-case bias achievable under a prescribed conditioning bound. We consider strongly convex quadratic client objectives with Hessian eigenvalues in $[\mu,L]$ and local updates represented by a shared positive spectral filter. Client heterogeneity is measured by the root-mean-square gradient at the global minimizer. For a uniform surrogate condition number bounded by $\chi\in[1,L/\mu]$, we establish the exact worst-case bias per unit heterogeneity, $(L-\mu\chi)/(2\mu L\sqrt{\chi})$, and show that local gradient descent with sufficiently small step sizes and proximal updates attain this bound. Our analysis reduces populations of arbitrary dimension, including noncommuting Hessians, to distributions over scalar curvatures. This reduction provides optimal
    
[^107]: 从数据到阀门：面向可持续暖通空调（HVAC）控制的自学习生产级边缘技术栈

    From Data to Valves: A Self-Learning, Production-Grade Edge Stack for Sustainable HVAC Control

    [https://arxiv.org/abs/2611.24052](https://arxiv.org/abs/2611.24052)

    该论文提出了一个在每栋建筑的NVIDIA Jetson Orin边缘网关上运行的自学习生产级HVAC控制栈，其核心创新是结合序贯学习且受多重门控保护的transformer温度预测器和带有不可削弱结构性安全包络的分布式软演员-评论家冷冻水阀门控制智能体，实现从建筑自身数据持续自适应并自我验证控制效果。

    

    学习型HVAC控制器很少能真正走出模拟器，而那些实际部署的控制器通常采用离线训练、部署后冻结参数、并依赖云端重新训练的模式。我们提出了一个生产级技术栈，其中三个学习组件运行在每栋建筑的一台NVIDIA Jetson Orin网关上，从建筑自身数据持续自适应，并验证自身效果。(a) 一个紧凑的transformer编码器预测器，每个设备组配有一个并带有学习得到的每机组嵌入，用于预测每个空气侧机组的下一周期温度；它以序贯方式学习，每个五分钟周期执行一次梯度更新，并处于五个独立门控的约束之下，以限制任何损坏读数的影响。(b) 一个分布式软演员-评论家智能体，在不同机组类型之间共享同一策略，在一个学习组件无法削弱的结构性安全包络之下，将测量状态转换为冷冻水阀门位置；其第二代系统利用记录的状态转移数据在设备上每夜学习。

    arXiv:2611.24052v1 Announce Type: cross  Abstract: Learned HVAC controllers rarely leave the simulator, and those that do are typically trained offline, frozen at deployment and retrained from the cloud. We present a production stack in which three learning components run on one NVIDIA Jetson Orin gateway per building, adapt continuously from the building's own data, and verify their own effect. (a) A compact transformer-encoder forecaster, one per equipment group with learned per-unit embeddings, predicts every air-side unit's next-cycle temperature and learns prequentially, one gradient step per five-minute cycle, behind five independent gates that bound the influence of any corrupt reading. (b) A distributional soft actor-critic agent, one shared policy across unit types, converts the measured state into a chilled-water valve position under a structural safety envelope that no learned component can soften; its second generation learns nightly on the device from recorded transitions 
    
[^108]: 算法化做市可通过订单簿导致粗糙波动率

    Algorithmic Market Making Can Lead to Rough Volatility Through the Order Book

    [https://arxiv.org/abs/2611.24028](https://arxiv.org/abs/2611.24028)

    本文提出一种自洽机制，将做市商建模为采用随机梯度下降规则、异步响应新信息调整报价的交易台，从而证明算法化做市可以通过订单簿动态自然地产生粗糙波动率。

    

    现有研究表明，市场可能表现出比布朗运动路径更粗糙的波动率。理论上，当订单流接近不稳定状态且其记忆按幂律衰减时，这种情况就可能发生。然而，这种订单流如何从以算法化做市为主导的订单簿动态中产生，目前尚不完全清楚。我们展示了这样一种自洽的机制。我们将每个做市商视为一个交易台，该交易台使用一阶梯度规则将其订单与中间价的距离朝某个未知“神谕”所提供的目标距离进行调整。我们假设不同的交易台在时间上并不同步行动，而是在与其各自神谕相关的新信息到达时才采取行动，每个交易台都在其自身特有的回溯时间范围内对这些信息进行采样。因此，在所有交易台之间，该规则表现为带有信息驱动（小批量）噪声的随机梯度下降（SGD）。在连续（极限下……）

    arXiv:2611.24028v1 Announce Type: new  Abstract: Existing work suggests that markets can show volatility with paths rougher than those of Brownian motion. In theory this can happen when the order flow is close to instability with memory that decays as a power law. However, it is not fully clear how such order flow could arise from the predominantly algorithmic market making that characterizes order book dynamics. We show one such consistent mechanism. We treat each market maker as a desk that uses a first-order gradient rule to move the distances of its orders from the mid price toward target distances provided by an unknown oracle. We assume that distinct desks do not act synchronously in time but act whenever new information relevant to their respective oracles arrives, which each desk samples over its own characteristic lookback horizon. Across all desks the rule therefore takes the form of stochastic gradient descent (SGD) with information-driven (mini-batch) noise. In the continuu
    
[^109]: 基于半监督生成对抗网络的CTPA扫描分割方法用于肺栓塞边界提取

    Semi-Supervised GAN-Based Segmentation of CTPA Scans for Pulmonary Emboli Boundary Extraction

    [https://arxiv.org/abs/2611.24018](https://arxiv.org/abs/2611.24018)

    该论文提出了一种基于半监督生成对抗网络的CTPA扫描分割方法，通过组合Dice损失、交叉熵损失、特征匹配损失和自训练损失，有效解决了CTPA图像高度不平衡和标注数据稀缺的问题，实现肺栓塞边界的自动提取。

    

    肺栓塞（PE）是一种具有高发病率和死亡率的肺血管疾病，延误诊断可能致命。虽然计算机断层扫描肺动脉造影（CTPA）是PE检测的临床金标准，但人工解读这些扫描图像既耗时又容易出错。在这项工作中，我们提出了一种半监督分割方法，利用生成对抗网络（GAN）来充分利用大量未标记的CTPA扫描数据。所提出的方法通过以下方式比以往方法更有效地解决了CTPA图像分割的挑战：采用Dice损失和交叉熵损失来应对CTPA扫描高度不平衡的特性；采用特征匹配损失来鼓励生成器产生在解剖学上合理、且与真实标注难以区分的分割图；以及采用自训练损失来强化对高置信度伪标签样本的学习。

    arXiv:2611.24018v1 Announce Type: cross  Abstract: Pulmonary embolism (PE) is a pulmonary vascular disease associated with high morbidity and mortality, where delayed diagnosis can be fatal. While computed tomography pulmonary angiography (CTPA) is the clinical gold standard for PE detection, interpreting these scans manually is time-consuming and error-prone. In this work, we propose a semi-supervised segmentation method, employing generative adversarial networks (GANs) to leverage large pools of unlabeled CTPA scans. The proposed method addresses more effectively the challenges of CTPA image segmentation than previous methods by employing the dice loss and the cross-entropy loss to tackle the highly imbalanced nature of CTPA scans, the feature matching loss to encourage the generator to produce anatomically plausible segmentation maps that are indistinguishable from ground truth, as well as the self-training loss to reinforce learning from highly-confident pseudo-labeled samples. The
    
[^110]: 重新思考放射学语言-图像预训练中的对比学习目标

    Rethinking Contrastive Targets in Radiology Language-Image Pretraining

    [https://arxiv.org/abs/2611.24016](https://arxiv.org/abs/2611.24016)

    该论文指出放射学CLIP式对比预训练中零样本性能先升后降及大批量训练失效的根源在于目标错定——大量未配对报告与图像共享临床发现导致配对身份目标存在系统性偏倚，作者通过合成测试平台CLEVR-Radiology复现了这些失败并建立了相应的目标错定理论加以解释。

    

    CLIP式的对比图文预训练被广泛应用于放射学数据中，其中每张图像与一份报告配对，而批次中所有其他报告都被视为负样本。在这种方案下会出现两种失败现象，而现有的对比学习理论均未系统地解决它们：零样本评估性能先上升后下降，而训练损失却在持续下降；以及训练无法从大批量数据中学习。我们假设其原因在于结构性问题，而非放射学领域特有的问题：许多未配对的报告与给定图像共享相同的临床发现，因此CLIP的配对身份目标存在系统性偏倚，而不仅仅是噪声。为验证这一假设，我们构建了CLEVR-Radiology，一个仅保留这种重叠事实结构的合成测试平台，并成功复现了上述两种失败现象。在有证据表明该结构足以复现失败现象之后，我们发展了一种目标错定理论，从…解释了这两种失败现象（原文摘要至此截断）。

    arXiv:2611.24016v1 Announce Type: cross  Abstract: CLIP-style contrastive image-text pretraining is widely used in radiology data, where each image is paired with one report and every other report in the batch is treated as a negative. Under this recipe, two failures appear, neither of which is systematically addressed by existing contrastive learning theory: zero-shot evaluation peaks then degrades while training loss keeps decreasing, and training fails to learn from large batches. We hypothesize the cause is structural rather than radiology-specific: many unpaired reports share clinical findings with a given image, so CLIP's paired-identity target is systematically biased, not merely noisy. To test this, we construct CLEVR-Radiology, a synthetic testbed that preserves only this overlapping-fact structure and reproduces both failures. With evidence that the structure is sufficient to reproduce the failures, we develop a target-misspecification theory that explains both failures from 
    
[^111]: NOTREKS：基于评分的结构学习中的边际独立性

    NOTREKS: Marginal Independence in Score-based Structure Learning

    [https://arxiv.org/abs/2611.24015](https://arxiv.org/abs/2611.24015)

    本文提出 NOTREKS 信息，将变量间边际独立的先验知识转化为“无 trek”图约束，并通过 PSTrek 惩罚项和 trek 约束融入基于评分的因果结构学习，从而在有限样本下提升结构恢复与效应识别效果。

    

    基于评分的结构学习通过在结构方程模型上优化依赖于数据的目标函数来估计因果图。我们研究了如何将“选定变量对之间边际独立”这一先验知识融入此类因果发现方法中。在忠实性假设下，边际独立的变量对之间不可能存在 trek（游走路径），因此该知识转化为图上的“无 trek”约束，我们将其称为 NOTREKS 信息。我们为基于梯度的优化引入了幂级数 trek（PSTrek）惩罚项，并为基于顺序的搜索引入了 trek 约束，同时分析了此类信息为何在有限样本中有所帮助，以及它如何限制搜索空间。我们的数值实验说明了 NOTREKS 信息在因果发现中能起到多大帮助。特别是，在带有 oracle NOTREKS 信息的合成基准实验中，我们展示了这些约束能够同时改善结构恢复效果和基于调整的效应识别。

    arXiv:2611.24015v1 Announce Type: cross  Abstract: Score-based structure learning estimates causal graphs by optimizing a data-dependent objective over structural equation models. We study how prior knowledge that selected pairs of variables are marginally independent can be incorporated into such causal discovery methods. Under faithfulness, marginally independent pairs cannot be connected by a trek, so the knowledge becomes a graphical no-trek constraint, which we call NOTREKS information. We introduce a power-series trek (PSTrek) penalty for gradient-based optimization and a trek constraint for order-based search, and we analyze why such information helps in finite samples and how it restricts the search space. Our numerical experiments illustrate to what extent NOTREKS information helps in causal discovery. In particular, using synthetic benchmarks with oracle NOTREKS information, we show that these constraints can improve both structural recovery and adjustment-based effect identi
    
[^112]: 重新思考固定长度的潜空间推理

    Rethinking Fixed-Horizon Latent Reasoning

    [https://arxiv.org/abs/2611.24001](https://arxiv.org/abs/2611.24001)

    该论文发现将潜空间思维链运行至完整预设长度反而可能损害推理性能，并提出用“已实现与剩余建构性进展”来刻画推理收益，为重新思考固定长度的潜空间推理提供了依据。

    

    现有的潜空间推理方法通常在推理阶段运行预设的潜空间推理长度。直觉上，更长的推理长度应该提供更多的推理机会。然而，在这项工作中，我们发现将潜空间思维链扩展到完整的预设长度实际上可能会降低推理性能。我们观察到潜空间思维链会经历不同的功能阶段：早期的计算形成前沿推理状态，而后期的计算主要是对已有候选进行重组，并可能削弱对正确候选的支持。这些发现促使我们重新审视固定长度的潜空间推理。为此，我们通过“已实现的建构性进展”和“剩余建构性进展”来刻画潜空间思维链带来的推理收益。实证结果表明，更高的已实现建构性进展对应更高的当前推理准确率，而更多的剩余进展则对应继续潜空间思维链所能带来的更大未来收益。

    arXiv:2611.24001v1 Announce Type: cross  Abstract: Existing latent reasoners typically run for a preset latent horizon during inference. Intuitively, a longer horizon should provide more opportunity for reasoning. However, in this work, we find that extending latent Chain-of-Thought (CoT) to the full preset horizon may in fact degrade reasoning performance. We observe that latent CoT evolves through distinct functional stages, with earlier computation forming frontier reasoning states, while later computation mainly reorganizes existing candidates and can weaken gold candidate support. These findings motivate a rethink of fixed-horizon latent inference. To address this, we characterize the reasoning benefits of latent CoT by realized and remaining constructive progress. Empirically, we find that higher realized constructive progress corresponds to higher current reasoning accuracy, while more remaining progress corresponds to larger future gains from continued latent CoT. To this end, 
    
[^113]: 上下文学习与上下文外幻觉的机制分析

    A Mechanistic Analysis of In-Context Learning and Out-of-Context Hallucination

    [https://arxiv.org/abs/2611.24000](https://arxiv.org/abs/2611.24000)

    本文通过动态上下文中功能角色的统一视角，首次将上下文学习与上下文外幻觉这两种看似对立的现象联系起来进行机制分析，发现角色效应的显著影响取决于足够的贡献幅度和一致的带符号聚合这两个必要条件。

    

    大语言模型能够通过上下文学习（ICL）推断并遵循特定规则，但却可能无法忠实于所提供的证据，从而产生上下文外（OOC）幻觉。现有工作大多沿着各自独立的路线分别研究ICL和OOC幻觉的内在机制。然而，这种割裂掩盖了这些看似对立的结果背后的机制基础。在这项工作中，我们通过动态上下文中功能角色的统一视角来研究这种对比。关键观察是，角色信息可以在模型内部被表示，但不一定能可靠地约束输出。我们推导了一阶分解来刻画角色效应，表明其显著影响取决于两个必要条件，即足够的贡献幅度和一致的带符号聚合。因果分析进一步表明，角色跨度的贡献会影响涉及响应的关键token选择。

    arXiv:2611.24000v1 Announce Type: cross  Abstract: Large language models can infer and follow specific rules through in-context learning (ICL), yet may fail to remain faithful to provided evidence, giving rise to out-of-context (OOC) hallucination. Existing work has largely investigated the underlying mechanisms of ICL and OOC hallucination along separate lines. However, this separation obscures the mechanistic basis for these seemingly opposing outcomes. In this work, we study this contrast through the unified view of functional roles in dynamic context. The key observation is that role information can be internally represented without reliably constraining output. We derive a first-order decomposition to characterize the role effect, showing that its substantial influence depends on two necessary conditions, namely sufficient contribution magnitude and coherent signed aggregation. Causal analysis further shows that role span contributions affect key token choices involving response s
    
[^114]: ReliFlow：面向组合式单细胞扰动预测的可靠性校准残差传输方法

    ReliFlow: Reliability-Calibrated Residual Transport for Combinatorial Single-Cell Perturbation Prediction

    [https://arxiv.org/abs/2611.23999](https://arxiv.org/abs/2611.23999)

    提出ReliFlow——一种条件流匹配模型，通过训练期交叉拟合估计单基因效应可靠性并据此校准加性源分布，将速度场分解为加性参考的已知修正与学习得到的残差传输场，从而在单扰动证据稀疏或缺失的情况下显著提升组合扰动响应的预测精度。

    

    预测组合扰动的转录响应，需要将干预之间的相互作用与单扰动效应估计中的误差区分开来。当单扰动证据稀疏或被完全留出时，这类误差会变得更加显著。我们提出了一种条件流匹配模型，该模型通过仅在训练阶段进行的交叉拟合来估计单基因效应的可靠性，并利用这些估计对加性源分布进行校准。其速度场被分解为对加性参考分布的已知修正项与一个学习得到的残差传输场。在Norman基因扰动数据集上，对于由已观测单扰动构成的未见组合，平均绝对误差（MAE）从0.142降至0.131；当组成单扰动被留出时，MAE从0.198降至0.178。在留出的ComboSciPlex药物组合数据上，MAE从0.191降至0.176。非加性残差相关性从0.574提升至0.638，并且各组件的消融实验……

    arXiv:2611.23999v1 Announce Type: cross  Abstract: Predicting the transcriptional response to combined perturbations requires distinguishing interactions between interventions from errors in their individual effect estimates. The latter become more pronounced when single-perturbation evidence is sparse or withheld. We introduce a conditional flow-matching model that estimates the reliability of single-gene effects from training-only cross-fitting and uses those estimates to calibrate the additive source distribution. The velocity is decomposed into a known correction of the additive reference and a learned residual transport field. On Norman genetic perturbations, reduces mean absolute error (MAE) from 0.142 to 0.131 for unseen combinations of observed singles and from 0.198 to 0.178 when constituent singles are withheld. On held-out ComboSciPlex drug combinations, MAE decreases from 0.191 to 0.176. Non-additive residual correlation improves from 0.574 to 0.638, and component ablations
    
[^115]: 具有一致有界Lipschitz常数的最优深度非参数回归

    Optimal Deep Nonparametric Regression with Uniformly Bounded Lipschitz Constants

    [https://arxiv.org/abs/2611.23979](https://arxiv.org/abs/2611.23979)

    本文在分裂谱范数乘积约束下建立了深度ReLU网络的尖锐逼近理论，并提出带惩罚的最小二乘估计器，使深度非参数回归在保持一致有界Lipschitz常数的同时达到最优统计收敛率。

    

    在涉及含噪协变量或估计特征的许多应用中，深度神经网络估计器必须具备对输入扰动的稳定性。权重范数约束被广泛用于通过限制神经网络的Lipschitz常数来控制对输入扰动的敏感度。然而，在标准范数乘积固定界下的现有逼近结果不足以建立最优的统计收敛率。本文在分裂谱范数乘积约束下为深度ReLU网络建立了一个尖锐的逼近理论。该理论结果表明，随着网络深度的增加，足够光滑的Hölder函数可以在范数乘积界保持固定的情况下被一致地逼近到任意精度。随后，我们针对深度非参数回归提出了一个带惩罚项的最小二乘估计器。所得估计器在具有一致有界Lipschitz常数的同时，仍能达到最优的统计收敛率。

    arXiv:2611.23979v1 Announce Type: cross  Abstract: Stability under input perturbations is necessary for deep neural network estimators in many applications involving noisy covariates or estimated features. Weight norm constraints are widely used to control sensitivity to input perturbations by bounding the Lipschitz constant of neural networks. However, existing approximation results under fixed bounds on standard norm products are insufficient to establish optimal statistical rates. In this paper, we establish a sharp approximation theory for deep ReLU networks under a split spectral norm product constraint. This theoretical result shows that sufficiently smooth H\"{o}lder functions can be uniformly approximated to arbitrary accuracy as the network depth increases, with the norm-product bound held fixed. We then propose a penalized least squares estimator for deep nonparametric regression. The resulting estimator has a uniformly bounded Lipschitz constant while still achieving the opt
    
[^116]: OMNIS+：面向动态分割神经网络的语义RAN切片与量化

    OMNIS+: Semantic RAN Slicing and Quantization for Dynamic Split Neural Networks

    [https://arxiv.org/abs/2611.23977](https://arxiv.org/abs/2611.23977)

    提出了OMNIS+语义RAN切片框架，通过动态分割神经网络、语义压缩与自适应“box”量化，在多小区边缘计算中实现通信与计算资源的联合优化调度，并显著增强了对传输比特错误的鲁棒性。

    

    边缘计算使资源受限的设备能够通过任务卸载来执行机器学习（ML）应用，其中无线接入网（RAN）切片在联合配置通信与计算资源方面发挥着关键作用。然而，现有的RAN切片框架依赖于静态计算模型，限制了其利用现代神经架构所支持的动态语义数据表示的能力。为解决这一局限，我们提出了OMNIS+，一个基于新一代动态分割神经模型的多小区边缘计算语义RAN切片框架。OMNIS+将动态神经压缩与自适应量化相结合用于任务卸载，在通信负载和计算维度上提供了灵活多样的决策空间。我们采用了一种新颖的“box”量化方案，该方案在不同压缩率下显著增强了对传输比特错误的鲁棒性。考虑到部分（摘要原文在此处截断）

    arXiv:2611.23977v1 Announce Type: new  Abstract: Edge computing enables resource-constrained devices to execute machine learning (ML) applications via task offloading, with radio access network (RAN) slicing playing a crucial role in provisioning joint communication and computing resources. However, existing RAN slicing frameworks rely on static computing models, limiting their ability to exploit dynamic semantic data representations enabled by modern neural architectures. To address this limitation, we propose OMNIS+, a semantic RAN slicing framework for multi-cell edge computing built upon a new generation of dynamic split neural models. OMNIS+ incorporates dynamic neural compression paired with adaptive quantization for task offloading, providing a versatile decision space across communication payload and compute dimensions. We leveraged a novel ``box'' quantization scheme that markedly enhances resilience to transmission bit errors across diverse compression rates. Considering part
    
[^117]: 训练一个有能力的GUI智能体需要环境有多完整？

    How Complete Must an Environment Be to Train a Capable GUI Agent?

    [https://arxiv.org/abs/2611.23946](https://arxiv.org/abs/2611.23946)

    该论文提出MobileGym平台，首次系统研究了GUI训练环境在奖励覆盖、数据覆盖和特征覆盖三个维度上的完整性对智能体训练效果的影响，发现环境完整性的降低会显著损害智能体的性能。

    

    图形用户界面（GUI）智能体通常在可执行的GUI环境中进行训练。为了降低成本、批量构建环境并减少人工投入，每项工作都会通过其环境构建选择舍弃部分环境内容。我们从三个维度来描述这些选择：奖励覆盖，即验证器所奖励的达成目标的轨迹；数据覆盖，即智能体接触到的超出任务读取或写入范围的应用数据；特征覆盖，即超出任务所使用的应用功能特征。现有工作基于工程考量来设定每个维度，但并未衡量其对训练效果的影响。那么，训练一个有能力的智能体，环境需要多完整？又需要在哪些维度上保持完整？在MobileGym中，我们分别调整每个维度，训练Qwen3-VL-4B-Instruct模型，并在原始完整环境中评估每个策略。结果显示，只奖励单一参考路径的验证器会使成功率降低10.53个百分点……

    arXiv:2611.23946v1 Announce Type: cross  Abstract: Graphical user interface (GUI) agents are widely trained in executable GUI environments. To lower cost, build environments in batches, and reduce manual effort, each work trades away part of the environment through its construction choices. We describe these choices along three dimensions: reward coverage, the goal-reaching trajectories the verifier rewards; data coverage, the application data the agent is exposed to beyond what the task reads or writes; and feature coverage, the application features beyond what the task uses. Existing works set each dimension on engineering grounds, but they do not measure its effect on training. How complete must an environment be to train a capable agent, and along which dimensions? In MobileGym, we adjust each dimension separately, train Qwen3-VL-4B-Instruct, and evaluate every policy in the original full environment. A verifier that rewards only one reference path lowers success by 10.53 points, a
    
[^118]: PGU-Decoder：面向工业级路网地图更新的先验引导统一解码框架

    PGU-Decoder: Prior-Guided Unified Decoding for Industrial Road-Network Map Updating

    [https://arxiv.org/abs/2611.23936](https://arxiv.org/abs/2611.23936)

    PGU-Decoder提出先验引导的统一解码框架，通过先验查询模块和关系图注意力网络，统一处理交叉口路网生成、合流/分流检测与可行驶区域检测三大任务，降低工业级路网地图更新的运营成本。

    

    及时的路网地图更新对导航和自动驾驶至关重要。其中三个关键任务分别是交叉口内部路网生成、合流/分流检测以及可行驶区域检测，分别覆盖拓扑、几何和属性三个维度。采用独立的流水线处理这些任务会导致规则维护、质量保证和人工验证的重复劳动，增加运营成本。我们提出PGU-Decoder，一个面向上述任务语义预测的生产级统一解码框架。其中，先验查询模块（PQM）从路网几何上采样的位置点构建点级先验查询；固定的参考点在整个解码过程中保留位置先验，使模型能够专注于周围的上下文信息；随后，点级表示被聚合为实例级表示，并由关系图注意力网络（RGAT）捕获实例间的依赖关系；最后由任务特定的预测头生成各项预测结果。

    arXiv:2611.23936v1 Announce Type: new  Abstract: Timely road-network map updates are essential to navigation and autonomous driving. Three key tasks are intra-intersection road network generation, merging/diverging detection, and drivable-area detection, covering topology, geometry, and attributes, respectively. Separate pipelines duplicate rule maintenance, quality assurance, and manual verification, increasing operational costs. We propose PGU-Decoder, a production-grade unified decoding framework for semantic prediction across these tasks. A Prior Query Module (PQM) constructs point-level prior queries from sampled locations on road-network geometries. Fixed reference points preserve positional priors throughout decoding, allowing the model to focus on surrounding context. Point-level representations are aggregated into instance-level representations, and a Relational Graph Attention Network (RGAT) captures inter-instance dependencies. Task-specific heads then generate predictions. 
    
[^119]: CoSlice：面向5G网络切片资源分配的协调式多智能体强化学习

    CoSlice: Coordinated Multi-Agent Reinforcement Learning for 5G Network Slicing Allocation

    [https://arxiv.org/abs/2611.23909](https://arxiv.org/abs/2611.23909)

    该论文提出CoSlice框架，将偏好条件化的切片智能体与基于大语言模型的协调器相结合，通过分层多智能体强化学习解决O-RAN中切片间物理资源块分配问题，在保障各切片SLA的同时实现整体资源的高效利用。

    

    无线接入网切片是一个共享资源分配问题：每个切片都力求以最少的无线资源消耗满足其服务等级协议（SLA），同时所有切片又在竞争有限的资源预算。这既造成了切片内部SLA保障与资源效率之间的权衡，也造成了切片之间对共享资源的冲突，因此需要专门化且协调一致的控制。在本工作中，我们提出了CoSlice，这是一个用于开放无线接入网（O-RAN）中切片间物理资源块分配的协调式多智能体强化学习框架。CoSlice使用基于偏好条件化的切片智能体进行快速资源分配，并使用基于大语言模型（LLM）的协调器以较慢的时间尺度、利用全局网络观测来动态调整这些智能体的目标。我们将该问题形式化为多目标部分可观测随机博弈，并建立了偏好相关均衡的存在性（摘要在此处被截断）。

    arXiv:2611.23909v1 Announce Type: cross  Abstract: Radio Access Network (RAN) slicing is a shared resource allocation problem: each slice seeks to satisfy its Service Level Agreement (SLA) with minimal radio-resource consumption, while all slices compete for a limited resource budget. This creates both an intra-slice trade-off between SLA assurance and resource efficiency and an inter-slice conflict over shared resources, calling for specialized yet coordinated control. In this work, we present CoSlice, a coordinated multi-agent reinforcement learning framework for inter-slice physical resource block allocation in Open Radio Access Network (O-RAN). CoSlice uses preference-conditioned slice agents for fast resource allocation and an LLM-based coordinator to adapt their objectives using global network observations at a slower timescale. We formulate the problem as a multi-objective partially observable stochastic game (mo-POSG), establish the existence of a Preference Correlated Equilibr
    
[^120]: 核岭注意力：具有选择性遗忘的非线性记忆

    Kernel Ridge Attention: Nonlinear Memory with Selective Forgetting

    [https://arxiv.org/abs/2611.23896](https://arxiv.org/abs/2611.23896)

    提出核岭注意力（KRA），一种通过递归岭回归实现非线性记忆更新、结合内容依赖的选择性遗忘机制、推理状态大小与上下文长度无关且在语言建模中优于注意力与循环基线的循环序列混合器。

    

    核岭注意力（Kernel Ridge Attention, KRA）是一种循环序列混合器，它将学习到的非线性特征与由协方差引导的记忆更新相结合。它通过递归岭回归来存储关联，利用过去键之间的相关性来决定新观测如何修改记忆。一个依赖于内容的 Ornstein–Uhlenbeck 转移控制信息的保留程度，并为未来的写入恢复不确定性。这一构造具有精确的时间核函数解释，并维持一个大小与上下文长度无关的推理状态。关联块映射和显式伴随算子支持并行训练，同时保留递归更新。在受控的语言建模实验中，KRA 相比注意力和循环基线提升了预测质量，在所评估的循环模型中增强了自然关联回忆能力，并实现了实用的分布式训练吞吐量。这些性质为其提供了一个有原则的基础。

    arXiv:2611.23896v1 Announce Type: cross  Abstract: Kernel Ridge Attention (KRA) is a recurrent sequence mixer that combines learned nonlinear features with covariance-guided memory updates. It stores associations through recursive ridge regression, using correlations among past keys to determine how new observations modify memory. A content-dependent Ornstein--Uhlenbeck transition controls retention and restores uncertainty for future writes. This construction admits an exact temporal-kernel interpretation and maintains an inference state whose size is independent of context length. Associative block maps and explicit adjoints support parallel training while preserving the recursive update. In controlled language-modeling experiments, KRA improves prediction quality over attention and recurrent baselines, strengthens natural associative recall over the evaluated recurrent models, and achieves practical distributed training throughput. These properties provide a principled foundation fo
    
[^121]: 加速分布式数字伊辛机

    Accelerating Distributed Digital Ising Machines

    [https://arxiv.org/abs/2611.23883](https://arxiv.org/abs/2611.23883)

    该论文提出了首个经过校准的分布式数字伊辛机跨平台成本模型，将扫描时间分解为计算、内存、同步和器件间通信四项，并据此提供系统性技术来加速其MCMC工作负载。

    

    伊辛机是围绕伊辛类自旋模型动力学构建的专用加速器，已成为组合优化和概率采样领域的主流平台。然而，这类问题所需的自旋数量达数百万个，远超任何单个器件的片上容量，因此实用的伊辛机必须将计算分布到多个器件上。先前的工作仅在单一运行点报告吞吐量，无法识别瓶颈资源，平台之间的性能交叉点也只能通过实验测量；目前尚无经过校准的分布式伊辛机跨平台成本模型，也缺乏加速其马尔可夫链蒙特卡洛（MCMC）工作负载的系统性技术。本工作提出了一个经过校准的成本模型，将数字伊辛机的扫描时间分解为计算、内存、同步和器件间通信等组成部分，并在三代GPU及FPGA架构上以平均4.8%的误差完成了验证。

    arXiv:2611.23883v1 Announce Type: cross  Abstract: Ising machines, dedicated accelerators built around the dynamics of Ising-like spin models, have emerged as a leading platform for combinatorial optimization and probabilistic sampling. The problems that motivate them require millions of spins, far beyond the on-chip capacity of any single device, so a practical Ising machine must distribute its computation across many devices. Prior work reports throughput at a single operating point, leaving the binding resource unidentified and platform crossovers measurable only by experiment; no calibrated cross-platform cost model of distributed Ising machines has been reported, nor have systematic techniques to accelerate their Markov chain Monte Carlo (MCMC) workloads. This work presents a calibrated cost model that decomposes digital Ising sweep time into compute, memory, synchronization, and inter-device communication terms, validated with 4.8% mean error across three GPU generations, FPGA an
    
[^122]: 固定线性擦除的秩一旁路攻击

    A Rank-One Bypass of Fixed Linear Erasure

    [https://arxiv.org/abs/2611.23880](https://arxiv.org/abs/2611.23880)

    本文证明了在开放权重场景下，攻击者仅需一个秩一的仿射更新即可绕过固定线性擦除层，几乎完全恢复被擦除的目标性能，同时保持良性线性头的输出分数不变。

    

    线性概念擦除可以使目标概念对于应用于模型当前内部特征的指定线性分类器族变得不可读。然而，在开放权重的发布中，攻击者可以修改产生这些特征的层。我们研究了一种固定仿射擦除层，其上游模型被允许发生变化。如果该层传输了哪怕一个对允许的下游分类器可见的方向，攻击者就可以将目标分数写入该方向并将其传递通过该层。我们证明这一条件也是必要的，并且对于非常数的线性目标分数，所需的改变是一个秩一的仿射更新。在六个CIFAR-10类别对任务中，该攻击路径恢复了几乎所有的擦除前目标性能，并且可以在数值容差范围内保持已部署的良性线性头的分数不变。随后我们对攻击进行了代价评估：使用八个目标标签和2,000个未标记的校准样本，10%的平均相对特征（摘要在此处截断）。

    arXiv:2611.23880v1 Announce Type: new  Abstract: Linear concept erasure can make a target unreadable to a specified family of linear classifiers applied to a model's current internal features. In an open-weight release, however, an attacker can modify the layers that produce those features. We study a fixed affine erasure layer whose upstream model is allowed to change. If the layer transmits even one direction visible to an allowed downstream classifier, the attacker can write a target score into that direction and carry it through the layer. We show that this condition is also necessary, and that for a nonconstant linear target score the required change is a rank-one affine update. Across six CIFAR-10 class-pair tasks, the route restores nearly all pre-erasure target performance and can leave a deployed benign linear head's scores unchanged to numerical tolerance. We then price the attack: with eight target labels and 2,000 unlabelled calibration examples, a 10% mean relative feature
    
[^123]: 无穷范数-Wasserstein 分布鲁棒主成分分析：精确重构、有限样本保证与高效算法

    Infinity-Wasserstein Distributionally Robust Principal Component Analysis: Exact Reformulation, Finite-Sample Guarantee, and Efficient Algorithm

    [https://arxiv.org/abs/2611.23877](https://arxiv.org/abs/2611.23877)

    该论文提出基于无穷范数-Wasserstein 距离的分布鲁棒 PCA 模型，实现了精确的有限维重构、最坏情况分布的显式刻画、有限样本理论保证与高效算法，并揭示其目标函数通过保留经验协方差之外的逐样本残差信息，使分布鲁棒性能够重塑所学的主子空间。

    

    主成分分析（PCA）是一种基础且广泛使用的降维与表示学习技术，然而其经验公式仍然容易受到分布不确定性的影响。尽管分布鲁棒优化（DRO）提供了一种自然的补救途径，但现有的基于二阶 Wasserstein 距离的 PCA 鲁棒模型要么保留了名义 PCA 子空间，要么缺乏精确的重构形式。本文展示了无穷范数-Wasserstein 距离如何能够同时兼顾这两个理想的性质。对于由此得到的 PCA 分布鲁棒优化模型，我们推导出了精确的有限维重构形式，并显式地识别出达到内层最大值的最坏情况分布。至关重要的是，重构后的目标函数在经验协方差矩阵之外还保留了逐样本的残差信息，揭示了一种分布鲁棒性借此重塑所学主子空间的机制。在真实分布满足温和假设的条件下……（原文摘要在此处被截断）

    arXiv:2611.23877v1 Announce Type: cross  Abstract: Principal component analysis (PCA) is a fundamental and ubiquitous technique for dimension reduction and representation learning, yet its empirical formulation remains vulnerable to distributional uncertainty. While distributionally robust optimization (DRO) offers a natural remedy, existing order-2 Wasserstein models of PCA either preserve the nominal PCA subspace or lack an exact reformulation. This paper demonstrates how the infinity-Wasserstein distance can reconcile these two desirable properties. For the resulting DRO model of PCA, we derive an exact finite-dimensional reformulation and explicitly identify a worst-case distribution attaining the inner maximum. Crucially, the reformulated objective retains samplewise residual information beyond the empirical covariance matrix, revealing a mechanism through which distributional robustness can reshape the learned principal subspace. Under the mild assumption that the true distributi
    
[^124]: 基于编码智能体的光学神经网络原位训练

    In Situ Training of Optical Neural Networks by Coding Agents

    [https://arxiv.org/abs/2611.23875](https://arxiv.org/abs/2611.23875)

    提出“光学即代码”方法，让编码智能体仅凭相机测量图像就能原位训练光学神经网络，自主形成对光学的物理理解并用可解释光学元件重新参数化网络，性能优于无梯度方法。

    

    光学神经网络有望实现快速、节能的计算，但每个器件都会偏离其设计值，因此必须结合硬件进行训练。原位训练依赖于专家为每个器件构建的校准数字模型，或者依赖无梯度的试错方法。编码智能体——即能够编写和运行程序的大型语言模型——已经展现出理解和编辑数字神经网络权重的涌现能力。在此，我们提出“光学即代码”方法，编码智能体仅凭测量数据即可训练光学神经网络，其通过一个框架实现：该框架返回其探测操作所对应的相机图像，并要求每次请求都附带一个假设。我们发现，该智能体形成了对光学的物理理解，并用可在硬件上调谐的、可解释的光学元件对网络进行了重新参数化。该方法在分类和图像编码任务中优于无梯度方法，接近通过校准模型进行反向传播的性能。

    arXiv:2611.23875v1 Announce Type: cross  Abstract: Optical neural networks promise fast, energy-efficient computing, but every device departs from its design and must be trained with the hardware. In situ training relies on an expert-built calibrated digital model of each device or on gradient-free trial and error. Coding agents, large language models that write and run programs, have shown an emergent ability to understand and edit the weights of digital neural networks. Here we introduce optics as code, a method in which a coding agent trains an optical neural network from measurements alone, through a harness that returns camera images of its probes and requires a hypothesis per request. We discovered that the agent forms a physical understanding of the optics and reparametrizes the network with interpretable optical elements tuned on the hardware. It outperforms gradient-free methods in classification and image encoding, approaching backpropagation through a calibrated model at a f
    
[^125]: LAGR：面向任务抓取的语言感知抓取重排序

    LAGR: Language-Aware Grasp Re-Ranking for Task Oriented Grasping

    [https://arxiv.org/abs/2611.23858](https://arxiv.org/abs/2611.23858)

    该论文构建了包含268万个带评分的5自由度抓取标注的大规模面向任务抓取数据集，并提出LAGR方法，通过语言感知的抓取重排序同时兼顾抓取的物理可行性与任务适配性。

    

    抓取是机器人操作的一项基础能力，而合适的抓取方式取决于下游任务。例如，锤子在敲钉子时应握住手柄，但在递给别人时应握住锤头。选择这样的抓取需要同时考虑两个标准：抓取在物体上的物理可行性以及与任务的适配性。开发此类方法需要包含物理可行性与任务评分的真实观测稠密抓取标签，但现有的面向任务抓取数据集规模小、多为合成数据或标注成本高昂。为填补这一空白，我们构建了一个包含268万个带评分的5自由度抓取标注的数据集，涵盖来自17个物体类别的100,911个图像-任务对。此外，我们提出了LAGR——一种面向任务的抓取检测方法，明确地同时应对上述两个挑战。给定视觉-语言输入，LAGR首先生成与任务无关的抓取候选及其物理可行性评分，随后利用语言感知的重排序机制，根据任务语义对抓取候选进行重新排序，从而选出既物理可行又符合任务需求的抓取方式。

    arXiv:2611.23858v1 Announce Type: cross  Abstract: Grasping is a fundamental capability for robotic manipulation, with the appropriate grasp depending on the downstream task. For example, a hammer should be held by its handle for hammering but by its head when it is handed over. Selecting such a grasp requires consideration of two criteria: physical feasibility on the object and suitability for the task. Developing such an approach requires dense grasp labels with both feasibility and task scores on real observations, but existing task-oriented grasping datasets are small, synthetic, or costly to annotate. To address this gap, we construct a dataset of 2.68 million scored 5-DoF grasp annotations over 100,911 image-task pairs from $17$ object categories. Additionally, we introduce LAGR, a task-oriented grasp detection method that explicitly addresses both these challenges. Given vision-language inputs, LAGR first generates task-independent grasp proposals along with a physical feasibili
    
[^126]: 神经记忆编排

    Neural Memory Orchestration

    [https://arxiv.org/abs/2611.23857](https://arxiv.org/abs/2611.23857)

    提出了首个在真实硬件上针对真实工作负载端到端部署的学习型页面预取系统 Overseer，通过 eBPF 采集缺页事件、transformer 预测器（OverNet）预测未来页面地址、内核常驻执行器提前载入页面，从而缓解 HPC 工作负载因缺页中断造成的内存瓶颈。

    

    现代高性能计算（HPC）工作负载越来越多地因内存层次结构而非计算而停滞，来自 mmap 文件、交换空间和分层内存的缺页中断在实际运行时间中所占的比例越来越大。内核中的启发式预取器能够较好地处理步长式和时间聚集的访问模式，但在这些工作负载典型的、不规则且依赖数据的访问流上性能会急剧下降。学习型预取器虽已被证明可在仿真中捕获此类模式，但尚未被部署为在真实硬件上针对真实工作负载运行的流水线。我们提出了 Overseer，一个端到端的学习型页面预取系统：一个 eBPF 采集器在目标主机上观测缺页事件，一个运行在联网推理硬件上的 transformer 预测器输出未来的页面地址，而一个常驻内核的执行器在这些页面被实际请求之前将其载入目标的地址空间。OverNet 通过逐实例……编码缺页序列

    arXiv:2611.23857v1 Announce Type: cross  Abstract: Modern HPC workloads increasingly stall on the memory hierarchy rather than on compute, and page faults from mmap'd files, swap, and tiered memory are a growing fraction of wall time. Heuristic prefetchers in the kernel handle stride-like and temporally clustered patterns well but degrade sharply on the irregular, data-dependent access streams typical of these workloads. Learned prefetchers have been shown to capture such patterns in simulation but have not been taken to a deployed pipeline running against live workloads on real hardware. We present Overseer, an end-to-end learned page-prefetching system. An eBPF collector observes page faults on the target host, a transformer predictor (OverNet) runs on network-attached inference hardware and emits future page addresses, and a kernel-resident enforcer materializes those pages into the target's address space before they are demanded. OverNet encodes fault sequences through per-instance
    
[^127]: 超越跨模态重建：卒中后步态中表面肌电-运动学耦合的参与者水平评估

    Beyond Cross-Modal Reconstruction: Participant-Level Evaluation of sEMG-Kinematic Coupling in Post-Stroke Gait

    [https://arxiv.org/abs/2611.23856](https://arxiv.org/abs/2611.23856)

    该研究通过源模态移除与跨参与者源替换实验证明，冻结的双向模型确实捕获了卒中后步态中特定于受试者的肌电-运动学神经肌肉耦合，而非仅依赖参与者间共享的时间模式。

    

    准确的跨模态重建常被视为生理耦合的证据，尽管它可能仅仅反映了参与者之间共享的时间模式。我们在82名保留的健全参与者和43名卒中后参与者中，测试了冻结的双向模型是否捕获了肌电活动与步态运动学之间特定于受试者的神经肌肉耦合。研究采用了完全源模态移除和跨参与者源替换方法，并结合周期-位置-年龄-性别模板基线。在重复嵌套交叉验证的情况下，还将学习到的双侧不对称性与透明的波形及导数不对称性进行比较，用于预测以表现为导向的活动评估（POMA）分数。完全源移除使肌电到运动学方向的重建RMSE增加了6.90%，反方向增加了11.38%。相比之下，跨参与者替换仅使RMSE增加了1.68%，并且……

    arXiv:2611.23856v1 Announce Type: new  Abstract: Accurate cross-modal reconstruction is often treated as evidence of physiological coupling, although it may simply reflect temporal patterns shared across participants. We tested whether frozen bidirectional models captured subject-specific neuromuscular coupling between electromyographic activity and gait kinematics in 82 held-out able-bodied and 43 post-stroke participants. Complete source-modality removal and cross-participant source substitution, along with a cycle-position-age-sex template baseline, were applied. Learned bilateral asymmetry was also compared with transparent waveform and derivative asymmetry for predicting Performance-Oriented Mobility Assessment (POMA) scores in the case of repeated nested cross-validation. Complete source removal increased reconstruction RMSE by 6.90% for electromyography-to-kinematics and 11.38% in the reverse direction. In contrast, cross-participant substitution increased RMSE by only 1.68% and
    
[^128]: 一种用于GPU发起通信的快速、可编程的GPU到CPU调用层

    A Fast, Programmable GPU-to-CPU Invocation Layer for GPU-Initiated Communication

    [https://arxiv.org/abs/2611.23848](https://arxiv.org/abs/2611.23848)

    提出KIWI框架，将GPU到CPU的反向卸载打造为一等通信路径，实现亚微秒级设备到主机延迟和每秒数亿次操作，性能可超越GPUDirect Async。

    

    现代GPU工作负载越来越多地依赖设备发起的通信，使GPU内核能够直接进行通信，而无需在内核边界与主机同步。现有库通常倾向于将传输控制放在GPU线程上，这种方法被称为GPUDirect Async（GDA），而将反向卸载到主机代理线程视为较慢的备用方案。我们认为反向卸载可以优于GDA，并应成为一等通信路径。我们进一步提出将反向卸载暴露为一种编程接口，允许用户定义特定于工作负载的通信操作。我们提出了KIWI，一个可编程的反向卸载框架，实现了亚微秒级的单向设备到主机延迟和每秒数亿次操作。我们解决了扩展反向卸载面临的实际挑战，包括共享队列上的争用、轮询开销以及代理之间的负载不均衡。KIWI-P2P是一个类SHMEM的库

    arXiv:2611.23848v1 Announce Type: cross  Abstract: Modern GPU workloads increasingly rely on device-initiated communication to let GPU kernels communicate directly without synchronizing with the host at kernel boundaries. Existing libraries commonly favor placing transport control on GPU threads, an approach known as GPUDirect Async (GDA), and treat reverse offload to host proxy threads as a slower fallback. We argue that reverse offload can outperform GDA and should be a first-class communication path. We further propose exposing reverse offload as a programming interface that lets users define workload-specific communication operations.   We present KIWI, a programmable reverse-offload framework that achieves sub-microsecond one-way device-to-host latency and hundreds of millions of operations per second. We address practical challenges of scaling reverse offload, including contention on shared queues, polling overhead, and load imbalance across proxies. KIWI-P2P, a SHMEM-like librar
    
[^129]: RePlan-VLA：面向长时程视觉-语言-动作模型的进度感知重规划

    RePlan-VLA: Progress-Aware Replanning for Long-Horizon Vision-Language-Action Models

    [https://arxiv.org/abs/2611.23843](https://arxiv.org/abs/2611.23843)

    RePlan-VLA是一个与具体VLA模型无关的框架，通过监测机器人运动来评估任务进度并自适应地修订未来计划，从而将VLA模型的短时程操作能力转化为可靠的长时程任务执行能力，且完全无需对底层VLA策略进行额外微调。

    

    视觉-语言-动作（VLA）模型已成为通用机器人操作的一种有前景的范式，但它们在需要多次相互依赖操作的长时程任务上性能往往会下降。我们没有让VLA策略承担长时程推理、记忆、动作生成、进度跟踪和错误恢复等负担，而是主张在一个执行感知框架中利用它们的短时程操作能力。为此，我们提出了RePlan-VLA，这是一个与具体VLA模型无关的框架，能够将VLA的短时程能力转化为可靠的长时程执行。RePlan-VLA通过显式监测机器人的运动来评估任务进度，并根据观察到的结果自适应地修订未来计划。我们完全保留了底层VLA策略，无需任何额外的微调。在RoboCasa365和CALVIN基准上的实验显示，性能分别提升了21.9%和……（原文摘要在此处截断）

    arXiv:2611.23843v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models have emerged as a promising paradigm for general-purpose robotic manipulation, yet their performance often degrades on long-horizon tasks that require multiple interdependent manipulations. Instead of burdening the VLA policies with long-horizon reasoning, memory, action generation, progress tracking, and error recovery, we argue for leveraging their short-horizon manipulation capabilities within an execution-aware framework. To this end, we introduce \textnormal{\textbf{RePlan-VLA}}, a VLA-agnostic framework that translates a VLA's short-horizon capabilities into reliable long-horizon execution. RePlan-VLA explicitly monitors the robot's motion for progress evaluation, and adaptively revises the future plan based on observed outcomes. We preserve the underlying VLA policy entirely, requiring no additional fine-tuning. Experiments on RoboCasa365 and CALVIN benchmarks show an improvement of 21.9% and 
    
[^130]: 合成交互线索重映射下的在线状态表征自适应

    Online State-Representation Adaptation under Synthetic Interaction-Cue Remapping

    [https://arxiv.org/abs/2611.23836](https://arxiv.org/abs/2611.23836)

    在线索重映射的合成任务中，持续在线更新状态映射的预测器较冻结系数的预测器将后期动作准确率提升28.76个百分点，且情境老虎机通过利用可揭示真实状态的动作奖励获得了最高的累积效用。

    

    当线索改变其所预测的内容时，持续更新已学习的映射可能有助于智能体选择恰当的动作。我们在一个单步合成任务中对此进行了验证，将一个持续从状态标签中学习的逻辑回归状态预测器与迁移时系数被冻结的同一预测器进行比较。我们在六个留出环境中评估了十种条件，每种条件进行1,296次配对运行，总计1,296,000条迁移记录。采用在线更新时后期动作准确率为93.39%，而冻结系数时仅为64.63%，配对差值为28.76 [8.73, 53.19] 个百分点（95%环境自助法置信区间）。使用相同预测但不同动作阈值的直接策略达到了95.65%。情境多臂老虎机达到了94.23%，并比自适应预测器获得了更高的累积效用。该老虎机利用了其所选动作的奖励，而在此任务中，奖励与所选动作相结合可以揭示真实状态。基于d的权重（摘要在此处截断）

    arXiv:2611.23836v1 Announce Type: new  Abstract: When cues change what they predict, continuing to update a learned mapping may help an agent choose the appropriate action. We tested this in a one-step synthetic task by comparing a logistic state predictor that continued to learn from state labels with the same predictor frozen at transfer. We evaluated ten conditions in six held-out environments, with 1,296 paired runs per condition and 1,296,000 transfer records in total. Late action accuracy was 93.39% with online updates and 64.63% with frozen coefficients, a paired difference of 28.76 [8.73, 53.19] percentage points (95% environment-bootstrap interval). A direct policy reached 95.65% using the same predictions with a different action threshold. A contextual bandit reached 94.23% and earned more cumulative utility than the adaptive predictor. The bandit used the reward of its chosen action, which in this task reveals the true state when combined with that action. Weights based on d
    
[^131]: NarrowSpec：推测解码中草稿模型的隐藏宽度无需与目标模型匹配

    NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding

    [https://arxiv.org/abs/2611.23835](https://arxiv.org/abs/2611.23835)

    提出NarrowSpec，证明推测解码中读取目标隐藏状态的草稿模型的隐藏宽度可缩小至目标模型的一半或四分之一而几乎不影响接受率，从而在B200和DGX Spark上实现1.06-1.44倍的端到端解码加速。

    

    推测解码被广泛用于加速大语言模型（LLM）的生成，其方式是由一个小型草稿模型提出候选token，再由目标模型的一次前向传播进行验证。读取目标模型隐藏状态的草稿模型（如DFlash和EAGLE-3）通常继承目标模型的隐藏宽度。我们证明，草稿模型的整个残差流可以在该宽度的一半或四分之一下运行，同时基本保持token接受率。对于借用目标模型词表投影的DFlash，训练得到的输出适配器可以折叠进该投影中，形成一个更窄的投影，其logits在浮点舍入误差范围内保持不变。我们在SGLang中，于B200和DGX Spark（GB10）硬件上，针对两个目标模型Qwen3-8B（NVFP4）和GPT-OSS-20B（MXFP4），测试了DFlash和EAGLE-3两个草稿模型系列。在批量大小为1、温度为0的条件下，将草稿模型收窄至一半或四分之一宽度，可使端到端解码在B200上提速1.06-1.12倍，在GB10上提速1.13-1.44倍。该加速效果……（原文摘要在此处截断）

    arXiv:2611.23835v1 Announce Type: new  Abstract: Speculative decoding is widely used to speed up generation in large language models (LLMs), with a small draft model proposing tokens that one forward pass of the target model verifies. Draft models that read the target's hidden states, such as DFlash and EAGLE-3, typically inherit the target's hidden width. We show that the draft's whole residual stream can run at half or a quarter of that width while largely preserving acceptance. For DFlash, which borrows the target's vocabulary projection, the trained output adapter folds into it, giving one narrower projection with logits unchanged up to floating-point rounding. We test two drafter families, DFlash and EAGLE-3, on two targets, Qwen3-8B (NVFP4) and GPT-OSS-20B (MXFP4), in SGLang on a B200 and a DGX Spark (GB10). At batch size 1 and temperature 0, narrowing the draft to half or a quarter width speeds up end-to-end decoding by 1.06-1.12x on B200 and 1.13-1.44x on GB10. The speedup hold
    
[^132]: CAFE-AQI：孟加拉国覆盖感知的次日AQI预测——融合早期预警与共形不确定性量化

    CAFE-AQI: Coverage-Aware Day-Ahead AQI Forecasting with Early Warning and Conformal Uncertainty in Bangladesh

    [https://arxiv.org/abs/2611.23815](https://arxiv.org/abs/2611.23815)

    该论文提出CAFE-AQI框架，通过审计发现孟加拉国数据集中城市标签重复与覆盖突变问题并加以加权处理，结合时空特征、类别均衡预警头和风险条件共形校准，实现次日最大AQI预测、健康风险早期预警及不确定性区间估计。

    

    次日空气质量预测通常假设各城市的记录同等完整且相互独立。然而，对一个大型的孟加拉国数据集进行审计后发现，2022年8月之后数据覆盖范围出现急剧扩张，且103个城市标签实际上仅对应61条独特的AQI轨迹。为解决这些问题，本文提出了CAFE-AQI，一个覆盖感知的次日最大AQI预测框架，同时估计空气质量达到“不健康或更差”级别（AQI ≥ 151）的概率。数据按时间顺序划分为训练集（2022-2023年）、校准集（2024年）和测试集（2025年）。为防止重复记录主导训练与评估过程，框架对相同轨迹进行了加权处理。该框架将时间与污染物特征同五个邻近站点的空间上下文相结合，并包含一个类别均衡的预警头和基于风险条件的共形校准，旨在提供早期预警和预测区间。

    arXiv:2611.23815v1 Announce Type: new  Abstract: Next-day air quality forecasting often assumes city records are equally complete and independent. However, an audit of a large Bangladesh dataset revealed a sharp coverage expansion after August 2022 and showed that 103 city labels correspond to only 61 unique AQI trajectories. To address these issues, this paper presents CAFE-AQI, a coverage-aware framework for forecasting next-day maximum AQI. It also estimates the probability of Unhealthy-or-worse conditions (AQI >= 151). Data were divided chronologically into training (2022-2023), calibration (2024), and testing (2025). To prevent duplicate records from dominating training and evaluation, identical trajectories were weighted. The framework combines temporal and pollutant features with a five-neighbor spatial context. It also includes a class-balanced warning head and risk-conditional conformal calibration aimed at providing early warnings and prediction intervals. Evaluation on the d
    
[^133]: 面向多模态智能体的持续工具空间扩展学习

    Continual Tool-Space Extension Learning for Multimodal Agents

    [https://arxiv.org/abs/2611.23810](https://arxiv.org/abs/2611.23810)

    提出COTEX方法，通过参数方向定位与正交约束，在适配器层面以低成本为多模态智能体持续扩展新工具，同时保持对已有工具的使用能力。

    

    多模态智能体在很大程度上依赖于裁剪、搜索和代码执行等通用工具，而将专业视觉模型排除在其工具空间之外。然而，将此类模型引入一个已经训练完成的智能体并非易事。其挑战在于如何高效地学习每个新工具，同时保留使用已有工具的能力。本工作提出了COTEX（持续工具空间扩展学习），能够以适配器级别的开销高效添加新工具。COTEX定位两组参数方向：一组触发工具调用，从而开启多轮推理轨迹；另一组决定选择哪个工具。新工具的适配器被限制在调用方向上，并通过正交约束确保不影响先前工具的选择。我们进一步引入了一套协议，可将专业模型现有的任务特定标注转换为经过验证的工具使用轨迹。在两个代表性智能体上的实验……

    arXiv:2611.23810v1 Announce Type: new  Abstract: Multimodal agents largely rely on generic tools such as cropping, search, and code execution, leaving specialist vision models outside their tool space. However, bringing such a model into an already trained agent is not straightforward. The challenge is to learn each new tool efficiently while retaining the ability to use existing ones. This work proposes COTEX, continual tool-space extension learning, that adds new tools at efficient adapter-level cost. COTEX locates two sets of parameter directions; one triggers a tool call and thereby opens a multi-turn reasoning trajectory, while the other determines which tool is chosen. The adapter of a new tool is restricted to the calling directions that leave earlier tool choices unaffected by orthogonal constraint. We further introduce a protocol that converts specialist models' existing task-specific annotations into verified tool-use trajectories. Experiments on two representative agents and
    
[^134]: Sightline：用于快速SVG生成的长窗口投机解码

    Sightline: Long-Window Speculative Decoding for Fast SVG Generation

    [https://arxiv.org/abs/2611.23807](https://arxiv.org/abs/2611.23807)

    Sightline通过将验证窗口延长至草稿器的预测范围并配合足够宽的草稿树，利用SVG生成中草稿错误稀少且聚集的特性，实现了当前最快的无损SVG解码。

    

    可缩放矢量图形（SVG）生成模型以代码形式编写图形，每个token需要一次前向传播，因此绘制速度很慢。投机解码可以无损地加速这一过程，而块级草稿器每次前向传播可猜测多个token。在SVG上，它们的许多轮次会接受整个草稿，并且只在验证窗口边界处结束。我们发现草稿的错误很少且呈聚集分布，并且在块内的准确深度远超自然语言场景。一个游程长度恒等式和三个命题表明，一旦候选足够宽，更长的验证窗口就会带来收益，其上限为草稿器的预测范围。我们提出Sightline，它将等于草稿器预测范围长度的验证窗口与足够宽以填满该窗口的草稿树相结合。实验涵盖文本到SVG和图像到SVG任务，使用三个模型在来自MMSVG、MMSVG-2M和SArena的六个基准上进行。在所有设置中，Sightline都是所比较的无损解码器中最快的。在SArena上，即使所有解码器共享同一个……（摘要截断）

    arXiv:2611.23807v1 Announce Type: cross  Abstract: Scalable Vector Graphics (SVG) generation models write a graphic as code with one forward pass per token, so drawing is slow. Speculative decoding speeds them up losslessly, and block drafters guess many tokens per pass. On SVG, many of their rounds accept the whole draft and end only at the verification window. We find that their misses are rare and clumped, and that they stay accurate far deeper into a block than on natural language. A run-length identity and three propositions show that a longer window pays once the candidates are wide, up to the drafter's horizon. We propose Sightline, which pairs a window as long as the drafter's horizon with a draft tree wide enough to fill it. Experiments cover text-to-SVG and image-to-SVG with three models on six benchmarks from MMSVG, MMSVG-2M and SArena. In every setting, Sightline is the fastest of the compared lossless decoders. On SArena, it stays the fastest when every decoder shares one 
    
[^135]: 面向约束主动线性回归的改进杠杆值分数采样方法

    Improved Leverage Score Sampling for Constrained Active Linear Regression

    [https://arxiv.org/abs/2611.23801](https://arxiv.org/abs/2611.23801)

    提出椭球正则化杠杆值分数采样，通过外接椭球的形状矩阵刻画约束集几何并将采样偏向重要方向，使约束主动线性回归的查询复杂度由各向异性有效维度决定，从而改进各向异性约束下的查询效率。

    

    本文研究约束线性回归的主动学习问题。给定设计矩阵 A∈R^{n×d}、约束集 C⊆R^d 以及对 b∈R^n 的查询访问，我们寻求在尽可能少的查询次数下得到 min_{x∈C}‖Ax−b‖₂² 的近似解。该问题出现在众多科学与工程应用中，随机化杠杆值分数采样为求解该问题提供了一种强大方法。然而，现有的杠杆值分数方法要么忽略约束，要么通过岭正则化来编码约束，导致在各向异性约束集上的查询复杂度不够理想。我们提出椭球正则化杠杆值分数：用约束集的一个外接椭球来近似该约束集，并利用其形状矩阵将采样偏向于约束几何下更重要的方向。我们证明查询复杂度由一个各向异性有效维度 d_eff(C) 决定。

    arXiv:2611.23801v1 Announce Type: cross  Abstract: In this paper we study active learning for constrained linear regression. Given a design matrix $A\in\mathbb{R}^{n\times d}$, a constraint set $C\subseteq\mathbb{R}^d$, and query access to $b\in\mathbb{R}^n$, we seek an approximate solution to $\min_{x\in C}\|Ax-b\|_2^2$ using as few queries as possible. This problem arises in many scientific and engineering applications, and randomized leverage score sampling provides a powerful method to solve it. However, existing leverage score methods ignore the constraint or encode it through ridge regularization, suffering unfavorable query complexity for anisotropic constraint sets. We propose ellipsoid-regularized leverage scores, where we approximate the constraint set by an outer ellipsoid and use its shape matrix to bias sampling toward directions that matter under the constraint geometry. We prove that the query complexity is governed by an anisotropic effective dimension $d_{\text{eff}}(C
    
[^136]: TabBioMed：一个大规模的生物医学表格学习基准

    TabBioMed: A Large-Scale Benchmark for Biomedical Tabular Learning

    [https://arxiv.org/abs/2611.23779](https://arxiv.org/abs/2611.23779)

    提出了大规模生物医学表格学习基准TabBioMed，统一了94个精选数据集与23个模型的标准化评估，发现表格基础模型整体预测性能最强，但其优势依赖于具体数据条件。

    

    机器学习的最新进展加速了生物医学预测，但对结构化生物医学数据的评估仍然碎片化，尤其是对于具有高维、大样本量、数据缺失和类别不平衡特点的数据集。我们提出了TabBioMed，一个包含94个精选公开生物医学表格数据集的大规模基准，涵盖电子健康记录、药物发现、基因组学、转录组学、蛋白质组学、代谢组学、单细胞组学和系统生物学等领域。TabBioMed通过标准化的、感知数据集特性的预处理与评估框架统一了这些数据集，实现了跨分类、回归和多目标任务的可复现比较。我们对23个模型进行了基准测试，涵盖经典基线模型、梯度提升树、神经表格架构以及表格基础模型。基础模型提供了最强的家族级预测性能。然而，其增益依赖于特定数据条件：树模型……（摘要在此处截断）

    arXiv:2611.23779v1 Announce Type: new  Abstract: Recent advances in machine learning have accelerated biomedical prediction, but evaluation on structured biomedical data remains fragmented, especially for datasets with high dimensionality, large sample size, missingness, and class imbalance. We introduce TabBioMed, a large-scale benchmark of 94 curated public biomedical tabular datasets spanning electronic health records, drug discovery, genomics, transcriptomics, proteomics, metabolomics, single-cell omics, and systems biology. TabBioMed unifies these datasets through a standardized, dataset-aware preprocessing and evaluation framework, enabling reproducible comparison across classification, regression, and multi-target tasks. We benchmark 23 models across classical baselines, gradient-boosted trees, neural tabular architectures, and tabular foundation models. Foundation models deliver the strongest family-level predictive performance. However, their gains are regime-dependent: tree-b
    
[^137]: STORM：一种用于结合亲和力预测的轻量级无结构模型

    STORM: A Lightweight Structure-Free Model for Binding Affinity Prediction

    [https://arxiv.org/abs/2611.23777](https://arxiv.org/abs/2611.23777)

    STORM是一种快速轻量、无需结构信息的机器学习模型，通过集成多种蛋白质和分子表示的小模型融合来预测结合亲和力，在相同训练数据下达到或超越Boltz-2的准确度，同时大幅降低了大规模筛选和模型重训的计算成本。

    

    准确预测小分子与蛋白质的结合亲和力是药物发现中的一项关键任务，但至今仍具挑战性。基于物理的方法（如自由能微扰FEP）计算成本极高，且需要专家进行设置。近期基于机器学习的共折叠模型已报告其准确度接近FEP——其中至少一个模型Boltz-2已被广泛用于亲和力预测——但这些模型对于大规模计算筛选而言成本高昂，而基于新收集的实验数据进行重新训练的成本则更高。为应对这些挑战，我们提出了STORM（Similarity Transfer from Observed Reference Molecules，从观测参考分子进行相似性迁移）。STORM是一个快速、简单的机器学习模型，它将多种蛋白质和分子表示的小型模型融合结果进行集成，用于结合亲和力预测。在相同数据上训练时，STORM在多个评估数据集上达到甚至超过了Boltz-2的准确度。

    arXiv:2611.23777v1 Announce Type: cross  Abstract: Accurately predicting a small molecule's binding affinity to a protein, a critical task in drug discovery, remains challenging. Physics-based approaches like free-energy perturbation (FEP) have very high computational costs and require expert setup. Recent machine learning--based cofolding models have reported accuracy approaching that of FEP---and at least one of them, Boltz-2, has already been widely adopted for affinity prediction---but they remain expensive for large computational screens and even more expensive for retraining on newly collected experimental data. To address these challenges, we introduce STORM: Similarity Transfer from Observed Reference Molecules. STORM is a fast, simple machine learning model that ensembles many small model fusions of diverse protein and molecule representations for binding affinity prediction. Trained on the same data, STORM matches or exceeds the accuracy of Boltz-2 on multiple evaluation data
    
[^138]: 物理信息神经网络小最优子空间的系统分析：$n$-DAS $\subset \mathbb{R}^p$

    A Systematic Analysis of a Small Optimal Subspace for Physics-Informed Neural Networks: $n$-DAS $\subset \mathbb{R}^p$

    [https://arxiv.org/abs/2611.23773](https://arxiv.org/abs/2611.23773)

    该论文提出“主导对齐子空间”（DAS），通过梯度对齐而非Hessian特征值大小来选择低维优化子空间，论证了最高曲率方向并非训练实际移动的方向，并以物理信息神经网络为对象对该结构进行了系统分析。

    

    寻找一个控制神经网络训练的低维子空间一直是优化领域的长期目标。然而，由Hessian特征值大小定义的子空间（例如Dom-SGD和Bulk-SGD所基于的主导/主体分解）一再未能捕捉到优化实际发生的位置；而基于能量分数阈值构建的自适应Krylov替代方法虽展现出前景，但在较低的特征模态上仍然表现不佳。我们认为这种失败是结构性的：最高曲率的方向通常并不是训练实际移动的方向。我们引入了主导对齐子空间，它基于Hessian的完整特征分解构建，但通过梯度对齐而非特征值大小来进行选择，既可作为理解优化动力学的工具，也可作为设计利用这些动力学的方法的基础。为了在最苛刻的条件下揭示这种结构，我们使用物理信息神经……

    arXiv:2611.23773v1 Announce Type: new  Abstract: The search for a low-dimensional subspace that governs neural network training has been a persistent goal in optimization. Yet subspaces defined by Hessian eigenvalue magnitude, such as the dominant/bulk decomposition underlying Dom-SGD and Bulk-SGD, have repeatedly failed to capture where optimization takes place, and adaptive Krylov-based alternatives built on energy-fraction thresholds show promise but still struggle at lower eigenmodes. We argue this failure is structural: the directions of highest curvature are not, in general, the directions along which training moves. We introduce the dominant aligned subspace (DAS), constructed from a full eigendecomposition of the Hessian but selected by gradient alignment rather than eigenvalue magnitude, as a tool for understanding optimization dynamics and a foundation for designing methods that exploit them. To expose this structure under the hardest conditions, we use physics-informed neura
    
[^139]: 仅用正常样本更新并不足够：基于重构的时间序列异常检测中的测试时自适应

    When Normal-Only Updates Are Not Enough: Test-Time Adaptation in Reconstruction-Based Time-Series Anomaly Detection

    [https://arxiv.org/abs/2611.23761](https://arxiv.org/abs/2611.23761)

    研究表明，即使将测试时自适应的更新严格限制在被标记为正常的样本上，也无法可靠地改善基于重构的时间序列异常检测性能。

    

    正常运行条件在部署后可能会发生演变，这促使研究者利用新传入的数据对时间序列异常检测器进行测试时自适应。然而，同一数据流中可能包含异常，因此带在线更新的测试时自适应必须决定哪些测试样本可用于自适应更新。这就留下了一个更基本的问题：即使更新仅限于基准数据集中被标记为正常的样本，测试时自适应是否仍能可靠地改善异常检测性能？我们通过一种受控干预来研究这个问题：用一条仅使用基准中标记为正常样本的规则替换实际中的更新选择器，同时保持在线自适应流程的其余部分不变。在九个基准数据集族、四种优化器配置和五个随机种子上的实验中，我们得到三个主要发现：（1）仅使用基准中标记为正常的样本并不能可靠地改善检测效果；（2）实际的测试时自适应失败表现出两种截然不同的响应模式……

    arXiv:2611.23761v1 Announce Type: new  Abstract: Normal operating conditions can evolve after deployment, motivating test-time adaptation (TTA) of time-series anomaly detectors using incoming data. Yet the same stream may contain anomalies, so TTA with online updates must decide which test samples should be used for adaptation. This leaves a more basic question: even if updates are restricted to samples labeled as normal in the benchmark, does TTA reliably improve anomaly detection? We study this question with a controlled intervention that replaces the practical update selector with a rule that uses only samples labeled as normal in the benchmark, while keeping the rest of the online adaptation procedure unchanged. Across nine benchmark families, four optimizer configurations, and five random seeds, we obtain three main findings: (1) using only samples labeled as normal in the benchmark does not reliably improve detection; (2) practical TTA failures exhibit two distinct responses to t
    
[^140]: 动态链接预测中时间戳置换的解读

    Interpreting Timestamp Permutations in Dynamic Link Prediction

    [https://arxiv.org/abs/2611.23753](https://arxiv.org/abs/2611.23753)

    该论文提出一种交叉诊断方法，将时间戳置换实验中的训练效应与评估历史效应分离，并发现动态链接预测器的性能差距主要源于原始时序训练（尤其是显式时间编码）的贡献，而非评估历史的改变。

    

    时间戳置换常被用来测试动态链接预测器是否利用了时间信息，但由此产生的性能差距可能反映的是训练的变化、评估历史的变化，或两者兼有。我们通过一种交叉诊断方法将这些效应分离开来，并测量输入暴露度：即被改变的历史是否真的改变了预测器的输入。在MOOC数据集上，使用二元交叉熵训练的TCL模型，在任一评估历史下都能从原始时序训练中获得约八个成对准确率百分点的提升，而在固定权重下改变评估历史仅带来0.03个百分点的变化。这种微小的历史对比源于68.37%的查询输入保持不变、其余大多数查询的排名稳定，以及增益与损失的相互抵消。在原始评估历史下，抑制显式时间编码会使训练对比从8.03个百分点降至3.64个百分点，即在六个随机种子上的配对减少量为4.39个百分点（95%置信区间为[3.68, 5.09]）。DyGFormer也显示出相应的训练……（摘要内容在此处被截断）

    arXiv:2611.23753v1 Announce Type: cross  Abstract: Timestamp permutations are used to test whether dynamic link predictors exploit temporal information, but a performance gap can reflect changes in training, evaluation history, or both. We separate these effects with a crossed diagnostic and measure input exposure: whether altered histories actually change the predictor's inputs. On MOOC, TCL trained with binary cross-entropy gains approximately eight pairwise-accuracy points from original-chronology training under either evaluation history, while changing history at fixed weights gives only 0.03 points. The small history contrast reflects unchanged inputs for 68.37% of queries, stable rankings for most others, and cancellation of gains and losses. Under original evaluation history, suppressing explicit time encoding reduces the training contrast from 8.03 to 3.64 points, a paired reduction of 4.39 points (95% interval [3.68, 5.09]) across six seeds. DyGFormer shows corresponding train
    
[^141]: 随时随地快速思考：面向任意智能体工作流决策点的统一System-1模型

    Thinking Fast, Anywhere: One System-1 for the Decision Points of Any Agent Workflow

    [https://arxiv.org/abs/2611.23751](https://arxiv.org/abs/2611.23751)

    AnyS1将人类编写的程序性技能转化为领域通用的决策点来源，训练出一个统一的4B轻量System-1模型，使任意智能体工作流在运行时通过一次前向传播即可完成常规决策，并在置信度不足时回退到大模型System-2。

    

    自主运行的LLM智能体在每一步都调用大模型（System-2），即便许多中间动作只需在一小组已知候选项中进行选择，也会带来高昂的成本和延迟。现有的双系统智能体将此类决策卸载给轻量级的System-1模型，但依赖于针对每个环境手工编写的工作流特定钩子。我们提出AnyS1，这是一个框架，它将人类编写的程序性技能——其指令天然指明了选择出现的位置及可用选项——转化为共享的、领域通用的决策点来源。在离线阶段，AnyS1从超过1,000个公开技能中挖掘有依据的可枚举决策，在隔离容器中收集基于执行轨迹的状态以构建Any-DP数据集，并通过标签平滑训练出一个经过校准的单一4B指针式System-1模型。在运行时，编译后的工作流仅需一次前向传播即可完成常规决策，当置信度低于校准阈值时则回退至System-2。

    arXiv:2611.23751v1 Announce Type: new  Abstract: Autonomous LLM agents incur substantial cost and latency by invoking a large model (System-2) at every step, even when many intermediate actions choose among a small set of known candidates. Existing dual-system agents offload such decisions to a lightweight System-1 model, but rely on workflow-specific hooks authored per environment. We present AnyS1, a framework that turns human-authored procedural skills, whose instructions naturally specify where choices arise and which options apply, into a shared, domain-general source of decision points. Offline, AnyS1 mines grounded enumerable decisions across over 1,000 public skills, gathers execution-grounded trajectory states inside isolated containers to construct Any-DP, and trains a single calibrated 4B pointer System-1 with label smoothing. At run time, compiled workflows resolve routine decisions in one forward pass and fall back to System-2 when confidence falls below a calibrated thres
    
[^142]: 同秩不同需：重新思考联邦学习中的LoRA重分配

    Same Rank, Different Needs: Rethinking LoRA Redistribution in Federated Learning

    [https://arxiv.org/abs/2611.23742](https://arxiv.org/abs/2611.23742)

    该论文揭示联邦LoRA中按矩阵重构最优截断聚合更新可能无法保留其对单个客户端输入的响应，并提出了接收方感知子空间适配框架RASA，利用对角输入二阶矩对接收方加权更新响应进行无中心PCA，实现秩约束接收方目标下的全局最优读出。

    

    联邦低秩适配（LoRA）使具有不同资源预算的客户端能够协作微调基础模型。为了在每个客户端的秩预算内分配聚合更新，现有方法通常保留聚合更新的主要奇异方向。然而，最能重建更新矩阵的方向未必最能保留其在特定客户端输入上的响应：一个能量较低的方向可能对该接收方更为重要。我们形式化了这种失配现象，并证明矩阵最优截断可能比同一秩下的另一种读出方式产生任意大的响应误差。受此结果启发，我们提出了接收方感知子空间适配（RASA），一个面向接收方感知知识迁移的即插即用框架。RASA利用紧凑的对角输入二阶矩，对接收方加权更新响应进行无中心PCA。所得到的读出对于秩约束下的接收方目标函数是全局最优的，并且……

    arXiv:2611.23742v1 Announce Type: new  Abstract: Federated low-rank adaptation (LoRA) enables clients with different resource budgets to collaboratively fine-tune foundation models. To distribute an aggregated update within each client's rank budget, existing methods commonly retain its leading singular directions. However, directions that best reconstruct the update matrix need not best preserve its responses on a particular client's inputs: a lower-energy direction may matter more to that recipient. We formalize this mismatch and show that matrix-optimal truncation can incur arbitrarily larger response error than another readout of the same rank. Motivated by this result, we propose Receiver-Aware Subspace Adaptation (RASA), a plug-in framework for receiver-aware knowledge transfer. RASA uses compact diagonal input second moments to perform uncentered PCA of receiver-weighted update responses. The resulting readout is globally optimal for the rank-constrained receiver objective and p
    
[^143]: 面向神经交流潮流的分布外泛化

    Toward Out-of-Distribution Generalization in Neural AC Power Flow

    [https://arxiv.org/abs/2611.23729](https://arxiv.org/abs/2611.23729)

    该论文系统评估了物理信息损失、多电网训练和扰动数据增强对图神经网络潮流模型从小电网向未见过的500节点大电网零样本迁移能力的提升，发现物理信息损失可改善迁移效果，而多电网训练虽与最佳单一电网持平，但仍无法完全缩小与目标电网之间的差距。

    

    arXiv:2611.23729v1 公告类型：cross 摘要：可再生能源并网、极端天气和数据中心负荷给电力系统运行带来的不确定性日益增加，这使传统潮流求解器在大规模应用下的可靠性面临挑战。基于机器学习的代理模型是一种潜在的替代方案，但由于为大型电网生成训练数据的成本过高，分布外泛化能力成为模型部署的关键；而目前在小规模电网上训练的模型恰恰缺乏这一能力。在本工作中，我们系统评估了物理信息损失、多电网训练和基于扰动的数据增强如何提升三种基于图神经网络的潮流模型从小型系统向更大的、未见过的500节点系统进行零样本迁移的能力。我们发现：(1) 相比纯监督目标，物理信息损失能改善迁移效果；(2) 在较小系统上进行多电网训练可以达到最佳单一电网的效果，但无法缩小与目标电网之间的差距。

    arXiv:2611.23729v1 Announce Type: cross  Abstract: Growing uncertainty in power system operations from renewables integration, extreme weather, and data center loads challenges the reliability of traditional power flow solvers at scale. Machine learning-based proxies are a potentially promising alternative, but the prohibitive cost of generating training data for large grids makes out-of-distribution generalization a key capability for deployment; currently, models trained on small grids lack that capability. In this work, we systematically evaluate how physics-informed losses, multi-grid training, and perturbation-based data augmentation improve the zero-shot transfer of three graph neural network-based power flow models from smaller systems to a larger, unseen 500-bus system. We find that (1) physics-informed losses improve transfer relative to purely supervised objectives; (2) multi-grid training on smaller systems matches the best single grid but does not close the gap to target-gr
    
[^144]: 三维点表示能否编码建筑功能？基于城市立面探测监督训练与预训练模型

    Do 3D Point Representations Encode Architectural Functions? Probing Supervised and Pretrained Models on Urban Facades

    [https://arxiv.org/abs/2611.23720](https://arxiv.org/abs/2611.23720)

    该研究发现冻结的预训练三维点云Transformer（如Utonia）能够编码建筑元素的功能属性（如窗户是否可开启、柱子是否承重），显著优于从头训练的网络和手工描述子，证明预训练是获得建筑功能理解能力的关键。

    

    语义标签只能说明建筑元素“是什么”，却不能说明它“有何用途”：同一立面上两个外观完全相同的窗户，可能一个是可开启的、另一个是固定的；一根柱子可能承重，也可能仅作装饰。我们探究三维点表示是否能够编码这类建筑功能。在22栋移动激光雷达（LiDAR）扫描的建筑上，每个元素标注15个功能属性，我们以整栋建筑为单位进行留出验证，并比较手工设计的描述子、从头训练的网络以及冻结的自监督点云Transformer。描述子与冻结特征均通过同一个线性层读取。从PointNet到1.08亿参数的PTv3，没有任何从头训练的网络能明显超过手工描述子（在功能模糊元素上为0.610 mAP），而冻结的Utonia达到0.732。相同架构的随机权重对照模型（0.649和0.652）表明这一提升应归因于预训练（Utonia：+0.080，95%建筑区间 [+0.035, +0.124]）。在所有测试的标注数量下，预训练特征均带来提升……

    arXiv:2611.23720v1 Announce Type: cross  Abstract: Semantic labels say what a building element is, not what it does: two identical-looking windows on one facade may be operable or fixed, a column may carry load or only decorate. We ask whether 3D point representations encode such architectural function. On 22 mobile-LiDAR buildings with 15 functional attributes per element, we hold out entire buildings and compare hand-crafted descriptors, networks trained from scratch and frozen self-supervised point transformers. Descriptors and frozen features are read by the same linear layer. No network trained from scratch, from PointNet to a 108M-parameter PTv3, clearly exceeds the descriptors (0.610 mAP on functionally ambiguous elements), whereas frozen Utonia reaches 0.732. Random-weight controls of the same architectures (0.649 and 0.652) attribute the gain to pretraining (Utonia: +0.080, 95% building interval [+0.035, +0.124]). Pretrained features improve at every tested number of labelled 
    
[^145]: HypoLoop：基于科学反馈修正权重编辑方案

    HypoLoop: Revising Weight-Editing Plans from Scientific Feedback

    [https://arxiv.org/abs/2611.23718](https://arxiv.org/abs/2611.23718)

    该论文提出HypoLoop，一种将实验反馈与权重编辑决策相关联的原生权重编辑方法族，通过跨轮次迭代修正可编辑参数位置和拟合规则，在22个科学任务上显著降低了预测误差。

    

    科学实验不断揭示预测误差。有效的更新取决于哪些参数发生变化，以及哪些误差驱动了修正。我们提出了HypoLoop，这是一系列原生权重编辑方法，将这些选择与实验反馈联系起来。固定策略、数值自适应策略和LLM引导策略生成编辑方案；数值求解器计算权重补丁，内部验证则选择最终部署的预测器。连续的观测结果和验证记录支持对可编辑位置和拟合规则的修正。实验涵盖22个科学任务，并同时使用了DeepSeek和Claude。在Burgers任务上，在相同的参数预算下重新分配可编辑位置，相较于固定位置可将误差降低58.80%和47.72%。在DIODE任务上，正残差加权在内部交叉验证中将绝对相对误差（AbsRel）降低了13.35%，且可编辑坐标保持同样的33个。跨轮次修正完整方案可降低最终AbsRel。

    arXiv:2611.23718v1 Announce Type: new  Abstract: Scientific experiments continually reveal prediction errors. Effective updates depend on which parameters change and which errors drive the correction. We introduce HypoLoop, a family of native weight-editing methods that connects these choices to experimental feedback. Fixed, numerically adaptive, and LLM-guided policies produce editing plans; numerical solvers compute weight patches, and internal validation selects the deployed predictor. Successive observations and validation records support revisions to editable locations and fitting rules. Experiments cover 22 scientific tasks and both DeepSeek and Claude. On Burgers, reallocating a matched parameter budget reduces error by 58.80% and 47.72% relative to fixed locations. On DIODE, positive-residual weighting reduces absolute relative error (AbsRel) by 13.35% in internal cross-validation, with the same 33 editable coordinates. Revising the complete plan across rounds lowers final AbsR
    
[^146]: 通过辨识进行学习：小型语言模型何时能从用户在岗反馈中学习，何时会退化

    Learning through Discernment: when a small language model learns from user feedback on the job, and when it decays

    [https://arxiv.org/abs/2611.23716](https://arxiv.org/abs/2611.23716)

    该研究发现，设备端小型语言模型在夜间从用户反馈中自我微调学习时，采用“先判断用户是否给出明确评价”的辨识式反馈感知机制，可使模型在两周内保持接近最优性能，但从长期看模型仍普遍呈现先提升后衰退的规律。

    

    设备端的小型语言模型可以通过在夜间从自己的日常工作中学习来持续改进。然而，在缺乏真实结果的情况下，必须有某种机制来决定哪些尝试值得学习，而模型自身的判断和用户的反应都可能是错误的。我们研究了一个配备LoRA适配器的小型模型（Qwen3 0.6B和1.7B），每晚根据裁判所接受的尝试进行微调，在WebShop和τ²-bench上进行了127次长达一年的生命周期实验。我们将真实结果、用户明确给出的评价以及模型自身的判断，与一个单独的、冻结的“反馈感知”模块进行了比较，该模块用于解读用户的反应，共构建了三个版本。就自我判断而言，无论使用何种适配器，模型在长达一年的时间里都表现出先提升后衰退的现象。一个将礼貌性放弃误认为成功的感知版本导致5次生命周期中的3次崩溃；而一个为辨识而构建的版本——它首先判断用户是否给出了明确的评价——使5次生命周期中的4次在两周内保持在接近最优水平，随后在干净但……（原文摘要在此处截断）

    arXiv:2611.23716v1 Announce Type: cross  Abstract: Small language models on devices could keep improving by learning, overnight, from their own daily work. Without a true outcome, however, something must decide which attempts are worth learning from, and both the model's own judgment and the user's reaction can be wrong. We study a small model (Qwen3 0.6B and 1.7B) with a LoRA adapter, fine-tuned each night on the attempts a judge accepts, over 127 lives of up to a year on WebShop and $\tau^2$-bench. We compare the true outcome, the user's stated verdict and the model's own judgment with a separate, frozen "feedback sense" that reads the user's reaction, built in three versions. Judging itself, the model improved and then decayed, over a year and whatever its adapter. A sense that took polite give-ups for success collapsed 3 of 5 lives; one built for discernment, which first asks whether the user gave a clear verdict, kept 4 of 5 near the oracle for two weeks, then faded on clean but s
    
[^147]: MoMoVL：面向端侧稀疏激活视觉语言模型的模态感知自适应压缩

    MoMoVL: Modality-Aware Adaptive Compression for On-Device Sparsely-Activated Vision-Language Models

    [https://arxiv.org/abs/2611.23695](https://arxiv.org/abs/2611.23695)

    提出MoMoVL，一种面向端侧混合专家视觉语言模型的模态感知自适应压缩方法，利用视觉与文本token的模态差异优化专家的混合精度量化，实现高效且保护隐私的端侧部署。

    

    智能体网络的兴起伴随着对端侧视觉语言模型日益增长的需求，这类模型能够在增强隐私保护并降低延迟的前提下执行多样化的跨模态网络任务。混合专家模型因其计算高效的扩展性而在现代视觉语言模型中被广泛采用，但其多专家架构由于庞大的内存需求而带来了部署挑战。混合精度量化是一种有效的压缩技术，能够在降低位宽的同时适应边缘设备异构的硬件预算，尤其适合混合专家模型，因为其固有的结构稀疏性会导致专家之间的重要性不均衡。尽管先前的工作已将量化应用于边缘端高效的混合专家模型推理，我们指出在基于混合专家模型的视觉语言模型中，视觉token和文本token的模态特定特征会进一步影响专家量化。为解决这一问题，我们提出了MoMoVL，一种模……

    arXiv:2611.23695v1 Announce Type: cross  Abstract: The rise of the agentic web has been accompanied by growing demand for on-device Vision-Language Models (VLMs) that perform diverse cross-modal web tasks with enhanced privacy and reduced latency. Mixture-of-Experts (MoE) models are widely adopted in modern VLMs for compute-efficient scalability, yet their multi-expert architecture poses deployment challenges due to substantial memory demands. Mixed-precision quantization is an effective compression technique that reduces bit-widths while adapting to the heterogeneous hardware budgets of edge devices, particularly well-suited for MoE due to its inherent structural sparsity, which causes imbalanced importance among experts. While prior work has applied quantization for efficient MoE inference at the edge, we point out that in MoE-based VLMs, the modality-specific characteristics of visual and text tokens further impact expert quantization. To address this issue, we present MoMoVL, a mod
    
[^148]: 测量世界模型潜在安全过滤器的保守性

    Measuring the Conservatism of Latent Safety Filters for World Models

    [https://arxiv.org/abs/2611.23685](https://arxiv.org/abs/2611.23685)

    提出了“重叠度”这一指标来衡量世界模型潜在安全过滤器因潜在表示不可区分而被迫拒绝安全动作的最小比例，从而可以量化并比较不同潜在表示设计的保守性。

    

    潜在安全过滤器根据世界模型的潜在表示来判断一个规划的动作是否安全。共形校准通过设置过滤器的阈值，使其仅以很小的概率接受不安全的动作。然而，取决于潜在表示的质量，过滤器仍可能过于保守而拒绝许多安全的动作。为了测量潜在表示强加给过滤器的保守性，我们将相似的潜在表示和动作分组为单元。一个对每个单元只能给出单一决策的过滤器，若要拒绝某单元中的不安全转移，就必须同时拒绝该单元中的安全转移。我们将位于此类单元中的安全转移所占的比例称为“重叠度”。我们证明，重叠度是此类过滤器为了不接受数据中任何不安全转移而必须拒绝的安全转移的最小比例。重叠度和共形校准都可以应用于任何潜在表示。我们利用这一点来比较三种现有设计，其中安全标签用于训练……（摘要原文在此处截断）

    arXiv:2611.23685v1 Announce Type: cross  Abstract: A latent safety filter decides from the latent of a world model whether a planned action is safe. Conformal calibration sets the threshold of the filter so that it accepts an unsafe action only with a small probability. However, the filter can still be conservative and refuse many safe actions, depending on the latent. To measure the conservatism that a latent forces on the filter, we group similar latents and actions into cells. A filter that gives one decision per cell can refuse the unsafe transitions of a cell only by refusing the safe ones in it as well. We call the share of safe transitions that lie in such cells the overlap. We prove that the overlap is the minimum share of safe transitions that such a filter has to refuse in order to accept no unsafe transition in the data. Both the overlap and the conformal calibration can be applied to any latent. We use this to compare three existing designs, in which the safety labels train
    
[^149]: 可变长度扩散万物：任意状态空间上的任意长度生成建模

    Variable Length Diffuse Everything: Arbitrary Length Generative Modeling on Arbitrary State Spaces

    [https://arxiv.org/abs/2611.23684](https://arxiv.org/abs/2611.23684)

    提出了可变长度扩散万物（VL-DE）框架，通过基于严谨理论的插入机制而非数据填充，实现了任意状态空间上可变长度多模态数据的简单通用生成。

    

    扩散模型和基于流的生成模型在各类应用中取得了显著成果，但大多数方法假设数据具有固定的维度或长度。与此同时，现实世界的问题（如布局生成）通常涉及可变长度的多模态数据，例如连续的边界框坐标与离散的类别配对。现有方法通常通过将数据填充到固定尺寸，或引入专门的目标函数来显式建模长度，这增加了复杂性并限制了通用性。我们提出了可变长度扩散万物（VL-DE），这是一个用于任意状态空间上可变长度多模态生成的简单而通用的框架。VL-DE 不采用填充方式，而是通过一个基于严谨理论框架的插入机制来处理可变长度，从而得到易于处理的训练目标。此外，利用 VL-DE 所基于的加性形式生成器这一特性……

    arXiv:2611.23684v1 Announce Type: new  Abstract: Diffusion and flow-based generative models have achieved remarkable results across diverse applications, but most assume data of fixed-dimension/length. Meanwhile, real-world problems such as layout generation often involve variable-length, multimodal data, such as continuous bounding box coordinates paired with discrete types. Existing methods typically address variable length by padding data to a fixed size or by introducing specialized objectives to model lengths explicitly, adding complexity and limiting generality. We propose Variable Length Diffuse Everything (VL-DE), a simple and general framework for variable-length multimodal generation on arbitrary state spaces. Instead of padding, VL-DE handles variable length through an insertion-based mechanism grounded in a principled theoretical framework, yielding a tractable training objective. In addition, by utilizing the fact that VL-DE is based on a generator with an additive form, w
    
[^150]: 评估受治理的LLM驱动的机器学习实验：生命周期、溯源与失败指标

    Evaluating Governed LLM-Driven ML Experimentation: Lifecycle, Provenance, and Failure Metrics

    [https://arxiv.org/abs/2611.23678](https://arxiv.org/abs/2611.23678)

    本文提出应从实验过程而非仅从结果分数来评估LLM驱动的ML实验系统，构建了由控制平面（而非LLM）管理试验生命周期、编辑范围与保留决策、并将记录写入只追加账本的治理框架，从中计算无效操作率、重复失败率和溯源完整性等治理指标，并在六个ML任务共1,445次试验中验证了同一契约的普遍适用性。

    

    运行机器学习实验的自主智能体通常以其达到的最佳分数来评判。但这个分数并不能说明它是如何达到的：智能体是否只编辑了其被允许编辑的文件，失败的试验是被保留还是被悄悄丢弃，以及每个保留或丢弃的决定是否可以追溯到其证据。我们提出在评估这类系统时不仅要看其结果，还要评估其实验过程，并为此构建了一个受治理的测试框架。一个控制平面（而非提出修改的LLM）拥有每个试验的生命周期，检查其编辑范围，决定保留还是丢弃，并将每次试验的一条记录追加到只追加的账本中。诸如无效操作率、重复失败率和溯源完整性等治理指标均从该账本中计算得出。在从合成逻辑回归到语言模型训练的六个机器学习任务、共1,445次试验中，同一套契约保持不变地应用。

    arXiv:2611.23678v1 Announce Type: new  Abstract: Autonomous agents that run ML experiments are usually judged by the best score they reach. That score says nothing about how it was reached: whether the agent stayed within the files it was allowed to edit, whether failed trials were kept or silently dropped, and whether each keep or discard decision can be traced to its evidence. We propose evaluating such systems on their experimental process as well as their outcome, and we build a governed harness that makes this possible. A control plane, not the LLM that proposes changes, owns each trial's lifecycle, checks its edit scope, decides keep or discard, and appends one record per trial to an append-only ledger. Governance metrics such as the invalid-action rate, the repeated-bad rate and provenance completeness are computed from that ledger. Across six ML tasks, from synthetic logistic regression to language-model training, and 1,445 trials, the same contract applied unchanged. The ledge
    
[^151]: 更简单的微型递归模型

    Simpler Tiny Recursive Models

    [https://arxiv.org/abs/2611.23677](https://arxiv.org/abs/2611.23677)

    提出STRM——一种仅用单一状态和统一更新的更简单微型递归模型，通过训练中速率被抖动的指数移动平均状态更新等关键设计，在不牺牲泛化能力的前提下提升了训练速度和最终性能。

    

    执行深度循环计算图的小型模型在数独、迷宫最短路径搜索等类谜题问题上展现出了极佳的实验结果。其中最著名的是TRM（微型递归模型），它是HRM（分层推理模型）的简化版本。然而，其底层的递归结构相当复杂，涉及一个高层状态与一个低层状态、嵌套循环递归，以及根据在循环中所处位置而具有不同输入输出签名的网络调用。这些机制是为了避免模型死记硬背和过拟合所必需的。我们提出了STRM，一种仅采用单一状态和统一更新的简化设计。STRM在不牺牲泛化能力的情况下，提升了训练速度和最终性能。使我们这一更简单的递归得以奏效的关键架构选择是：对状态采用具有不同速率的指数移动平均更新（这些速率在训练过程中会被随机抖动），以及对解的读出机制……

    arXiv:2611.23677v1 Announce Type: new  Abstract: Small models that execute deep looped computational graphs have shown great experimental results on puzzle-like problems like Sudoku, finding shortest paths through mazes, etc. The most well-known one is TRM (tiny recursive model), a simplification of HRM (hierarchical reasoning model). However, the underlying recursion is complicated and involves a high-level and a low-level state, a nested loop recursion, and the network calls with different input and output signatures depending on where we are in the loops. These mechanisms are needed to avoid memorization and overfitting. We propose STRM, a simplified design with a single state and a uniform update. STRM does not sacrifice generalization, improves training speed and final performance. Our key architectural choices that allow our simpler recursion to work are exponential moving updates on the states with varying rates that are jittered during training and readout of the solution on sl
    
[^152]: 面向灵巧操作的视觉-语言-动作模型微调：选对检查点才是关键

    Finetuning VLAs for Dexterous Manipulation: The Right Checkpoint Is All You Need

    [https://arxiv.org/abs/2611.23676](https://arxiv.org/abs/2611.23676)

    该论文发现，用SFT结合扩散引导强化学习（DSRL）将VLA适配到灵巧手时，SFT最优检查点与DSRL后最优策略并不一致，且冻结VLM、仅微调动作专家模块能得到最可引导的策略，因此选择正确的检查点至关重要。

    

    视觉-语言-动作（VLA）模型已展现出强大的操作能力。VLA模型的一个局限在于，大多数操作数据集都聚焦于平行夹爪系统，这使得向多指灵巧手的迁移颇具挑战，尤其是使用多指机器人采集遥操作数据集更为困难。在本工作中，我们研究了如何最佳地将VLA模型适配到灵巧手上，方法是将监督微调（SFT）与扩散引导强化学习（DSRL）相结合。我们使用 π0.5 模型和 LEAP 手，在两个仿真任务上仅用少量演示数据评估了五种微调方案。我们的实证分析得出了三个关键洞见：（1）SFT 的成功率与 DSRL 之后的成功率相关性很弱，这意味着最优的 SFT 检查点往往不是最适合进行引导（steering）的检查点；（2）冻结视觉-语言模型（VLM）、仅微调动作专家模块，无论在训练分布内还是分布外都能产生最可引导的策略（摘要在此处截断）……

    arXiv:2611.23676v1 Announce Type: cross  Abstract: Vision-Language-Action models have demonstrated strong manipulation capabilities. A limitation of VLA models is that most manipulation datasets focus on parallel-jaw gripper systems, making transfer to multi-finger hands challenging, particularly as teleoperation datasets are harder to collect with multi-fingered robots. In this work, we study how to best adapt VLAs to dexterous hands by pairing supervised fine-tuning (SFT) with diffusion-steering reinforcement learning (DSRL). Using $\pi_{0.5}$ and a LEAP hand, we evaluate five fine-tuning recipes across two simulation tasks using only a handful of demonstrations. Our empirical analysis yields three key insights: (1) SFT success is poorly correlated with post-DSRL success, meaning the best SFT checkpoint is rarely the most effective to steer; (2) freezing the vision-language model (VLM) and tuning only the action expert produces the most steerable policy for both training and out-of-d
    
[^153]: 通信式在策略蒸馏：通过通信为长时程任务蒸馏黑盒教师模型的能力

    Communicative On-Policy Distillation: Distilling Black-Box Teacher Capabilities through Communication for Long-Horizon Tasks

    [https://arxiv.org/abs/2611.23673](https://arxiv.org/abs/2611.23673)

    提出通信式在策略蒸馏（COPD），以自然语言通信为媒介，让学生模型自主决定何时向黑盒教师提问，并将成功的师生交互用于在策略自蒸馏与任务奖励强化学习相结合的训练，从而在长时程任务上取得最强黑盒蒸馏基线 2.46 倍的增益。

    

    在策略蒸馏是一种提升语言模型智能体在长时程任务上表现的强大范式，然而其传统形式依赖于教师模型的 logits，而仅支持文本的黑盒模型并不会暴露这些信息。为解决这一局限，我们提出了通信式在策略蒸馏，它以自然语言通信为媒介，将更强大模型的能力蒸馏到更小的学生模型中。在任务执行过程中，学生模型自主决定何时咨询黑盒教师，并在没有预定义问题模板或干预时间表的情况下提出开放式的自然语言请求。成功的学生-教师交互会被保留为特权参考，通过在策略自蒸馏来监督学生在无顾问情况下的全新推演，并与任务奖励强化学习联合进行。在 CAMG-II 上，COPD 取得了最强黑盒 OPD 基线 2.46 倍的增益。

    arXiv:2611.23673v1 Announce Type: cross  Abstract: On-policy distillation (OPD) is a powerful paradigm for improving language-model agents on long-horizon tasks, yet its conventional formulation relies on teacher logits that text-only black- box models do not expose. To address this limitation, we introduce Communicative On-Policy Distillation (COPD), which uses natural-language communication as a medium for distilling a stronger model's capabilities into a smaller student. During task execution, the student au- tonomously decides when to consult a black-box teacher and formulates open-ended natural- language requests without predefined question templates or intervention schedules. Successful student-teacher interactions are retained as privileged references to supervise fresh adviser-free student rollouts through on-policy self-distillation, jointly with task-reward reinforcement learning. On CAMG-II, COPD achieves 2.46 times the gain of the strongest black-box OPD baseline. The capab
    
[^154]: 免费的午餐：梯度平均使Newton Sketch达到最优统计效率

    A Free Lunch: Gradient Averaging Achieves Optimal Statistical Efficiency in Newton Sketch

    [https://arxiv.org/abs/2611.23671](https://arxiv.org/abs/2611.23671)

    该论文证明，在草拟牛顿方法中仅需将随机梯度替换为滑动平均（无需额外计算开销、不改变内部求解器收缩界），即可缓解甚至消除草拟带来的计算—统计权衡，实现最优统计效率。

    

    我们研究在线草拟牛顿方法的统计推断问题，其中每个牛顿系统由内部随机草拟求解器近似求解。已有研究揭示了一种计算—统计权衡：草拟求解器（及其加速变体）将牛顿方法每步的计算成本从 O(d^3) 降低至 O(d^2)，但草拟所引入的随机性也会增大最后迭代点和平均牛顿迭代点的极限协方差。在本工作中，我们通过梯度平均来解决这一权衡。我们证明，仅仅将每步的随机梯度替换为滑动平均，就可以缓解、甚至在某些情况下消除 Newton sketch 中的这一权衡。重要的是，梯度平均不需要任何额外的预言机评估，并且不会改变内部草拟求解器的收缩界。特别地，我们证明了带梯度平均的最后草拟牛顿迭代点可以达到……（摘要在此处截断）

    arXiv:2611.23671v1 Announce Type: cross  Abstract: We study statistical inference for online sketched Newton methods, in which each Newton system is approximately solved by an inner randomized sketching solver. Existing work reveals a computational--statistical trade-off: the sketching solver (and its accelerated variants) reduces the per-step cost of Newton methods from $O(d^3)$ to $O(d^2)$, but the randomness introduced by sketching also inflates the limiting covariance of both the last and the averaged Newton iterates. In this work, we address this trade-off via gradient averaging. We show that simply replacing the stochastic gradient with a moving average at each step can mitigate, and in some cases eliminate, the trade-off in Newton sketch. Importantly, gradient averaging requires no additional oracle evaluations and leaves the contraction bound of the inner sketching solver unchanged. In particular, we prove that the last sketched Newton iterate with gradient averaging attains a 
    
[^155]: CratesBump：面向 Rust 生态系统的版本升级数据集

    CratesBump: A Version-Bump Dataset for the Rust Ecosystem

    [https://arxiv.org/abs/2611.23664](https://arxiv.org/abs/2611.23664)

    该论文发布了首个用于预测 Rust 生态系统版本升级的公开数据集 CratesBump（涵盖约 22 万个 crate、超过两百万个连续发布对、每对含 41 个特征和双重标签），并配套提供可从公开数据重新生成该数据集的工具 semver-extract，基准实验表明版本升级预测任务可学习但远未解决。

    

    我们发布了 CratesBump，这是首个用于预测 Rust 生态系统中版本升级的公开数据集，同时发布了构建该数据集的工具 semver-extract。CratesBump 由 crates.io 上在 2026 年 9 月 14 日之前至少发布过两个版本的每个 crate（约 22 万个 crate）构建而成，提供了超过两百万个连续发布对。每个发布对包含 41 个特征（基于 AST 差异的代码变更、依赖变更、复杂度、发布节奏、GitHub 提交与 issue 活跃度）、其源代码仓库（如有列出），以及两个标签：对 MAJOR.MINOR.PATCH 的字面解读，以及 Cargo 的规则——按照该规则，0.y 版本的升级属于破坏性变更（breaking change）。semver-extract 可以基于公开的 crates.io 数据和 GitHub API 重新生成该数据集。为了展示其用途，我们（i）在留出的仓库上对版本升级预测器进行基准测试，发现该任务是可学习的但远未解决，且在较新的版本上得分更低；（ii）测量哪些特征组携带有效信号；（iii）检验……（原文摘要在此处被截断）

    arXiv:2611.23664v1 Announce Type: cross  Abstract: We release CratesBump, the first public dataset for predicting version bumps in the Rust ecosystem, together with semver-extract, the tool that builds it. CratesBump is built from every crate on crates.io with at least two releases before 14 September 2026 (about 220,000 crates), giving over two million consecutive release pairs. Each pair has 41 features (code changes from AST diffs, dependency changes, complexity, release cadence, GitHub commit and issue activity), its source repository where listed, and two labels: the literal reading of MAJOR.MINOR.PATCH, and Cargo's rule, under which a 0.y bump is a breaking change. semver-extract regenerates the dataset from public crates.io data and the GitHub API. To illustrate its use we (i) benchmark bump predictors on held-out repositories, finding the task learnable but far from solved, with lower scores on later releases; (ii) measure which feature groups carry the signal; (iii) check, aga
    
[^156]: 同一模型，不同人群：数值执行契约与公平性审计的可复现性

    Same Model, Different People: Numerical Execution Contracts and the Reproducibility of Fairness Audits

    [https://arxiv.org/abs/2611.23657](https://arxiv.org/abs/2611.23657)

    研究通过因果对照实验证明，即使模型权重与审计代码完全一致，仅仅改变数值计算方式（浮点精度、硬件与软件栈）就会改变阈值附近的个体决策，从而威胁公平性审计结论的跨平台可复现性。

    

    公平性审计通常将训练好的模型视为一个固定的决策函数，但用于评估该模型的算术运算本身也是这个函数的一部分。本研究在保持模型、数据记录、阈值和审计代码完全不变的前提下，分两个层级改变数值执行契约。第一层级仅改变乘积与求和的计算方式，序列化权重字节级完全一致，是对算术运算影响的因果测试。第二层级运行降低精度的部署配置（BF16 计算、FP8 及 8 位权重），这些配置还会对权重进行舍入并改变引擎与内核，因此其影响归属于配置层面。研究在四个公开任务（包括住院再入院预测）上，对三个表格模型家族和两个开源权重语言模型（4B 和 24B 参数）进行了审计，硬件覆盖 NVIDIA、AMD 和 Apple 的 CPU 与 GPU。结果显示：第一层级改变了阈值附近的个体决策，而具体改变哪些决策取决于 GPU 及其软件栈，但在预先设定的、未加权的总体指标下（摘要此处被截断）……

    arXiv:2611.23657v1 Announce Type: cross  Abstract: Fairness audits treat a trained model as a fixed decision function, but the arithmetic that evaluates it is part of that function. This study holds the model, records, threshold and audit code fixed and varies the numerical execution contract in two tiers. Tier 1 changes only how products and sums are computed, with the serialized weights byte-identical, and is the causal test of arithmetic. Tier 2 runs reduced-precision deployment configurations (BF16 compute, FP8 and 8-bit weights), which can also round weights and change engines and kernels, so its effects belong to the configuration. It audits three tabular model families and two open-weight language models (4B and 24B parameters) on four public tasks, including hospital readmission, on CPUs and GPUs from NVIDIA, AMD and Apple. Tier 1 changed decisions near the threshold, and which ones depended on the GPU and its software stack, but no verdict under the prespecified, unweighted te
    
[^157]: Toaster：由级联线性记忆与目标传播构建的非线性记忆

    Toaster: Nonlinear Memory from Cascaded Linear Memories and Target Propagation

    [https://arxiv.org/abs/2611.23648](https://arxiv.org/abs/2611.23648)

    Toaster通过将两个线性记忆级联并插入激活函数、利用目标传播近似跨层信号，构建出可在未修改的delta规则核上高效训练的两层非线性记忆，兼具线性记忆的训练速度与深度记忆（MLP）的表达能力。

    

    循环记忆使语言模型能够以固定大小的状态处理长序列。DeltaNet和KDA等线性记忆存储一个矩阵，并对每个token执行一步delta规则更新；这类步骤在一个块内的乘积具有精确的WY形式，因此训练速度很快。TTT-MLP和Titans等深度记忆存储一个小型MLP，能够承载矩阵无法表达的映射，但其更新不具备WY形式。Toaster基于一个已知事实——线性层上的梯度步等价于一次delta规则写入——将两者连接起来。它由两个线性记忆并在其间插入激活函数，构建出一个两层非线性记忆。我们使用目标传播，通过第二个记忆将输出误差反向投影，从而为隐藏记忆赋予取值。在获得每个块近似计算一次的跨层信号后，每一层都可以在未经修改的delta规则核上运行。我们的默认配置在输出层上运行该核，并每个块更新一次隐藏层。

    arXiv:2611.23648v1 Announce Type: cross  Abstract: Recurrent memories let language models process long sequences with a fixed-size state. Linear memories such as DeltaNet and KDA store one matrix and take one delta-rule step per token. The product of a chunk of such steps has an exact WY form, so they train fast. Deep memories such as TTT-MLP and Titans store a small MLP, which can hold maps that no matrix can, but their update has no WY form. Toaster bridges the two by building on a known fact: a gradient step on a linear layer is a delta-rule write. It builds a two-layer nonlinear memory from two linear memories with an activation between them. We use target propagation to give the hidden memory its values by projecting the output error back through the second memory. Given a chunk's cross-layer signals, approximated once per chunk, each layer can then run on an unmodified delta-rule kernel. Our default runs this kernel on the output layer and updates the hidden layer once per chunk.
    
[^158]: 大型语言模型何时能建模正确的目标？高风险风险预测中的问题构建

    When Do LLMs Model the Right Target? Problem Formulation in High-Stakes Risk Prediction

    [https://arxiv.org/abs/2611.23638](https://arxiv.org/abs/2611.23638)

    该研究通过儿童福利、医疗保健和就业三大高风险场景的实验评估发现，在缺乏明确指导的情况下，LLM在构建风险预测问题时无法可靠地识别和处理“决策依赖性”（即历史决策影响观察结果的效度威胁），凸显了将LLM用于高风险决策建模时人工审查问题构建的必要性。

    

    生成式AI正日益被用于辅助数据科学任务，但其正确构建分析问题的能力仍存在根本性的疑问。我们研究了大型语言模型在为高风险决策制定开发模型时，能够多好地构建预测问题。我们专注于效度面临的一个特定挑战：决策依赖性，即历史决策会影响观察到的结果。我们在儿童福利、医疗保健和就业场景中，使用来自OpenAI和Anthropic的11个模型以及七种提示条件，评估LLM生成的代码和解释中问题构建的这一方面。我们评估LLM生成的代码是否实现了在研究假设下适合估计预期结果的程序，以及其解释是否承认决策依赖性。我们发现，在没有指导的情况下，LLM无法可靠地解决决策依赖性问题。

    arXiv:2611.23638v1 Announce Type: cross  Abstract: Generative AI is increasingly used to assist with data science tasks, yet fundamental questions remain about its ability to correctly formulate analytical problems. We investigate how well large language models (LLMs) can formulate prediction problems when developing models for high-stakes decision making. We focus on one particular challenge to validity: decision dependency, which arises when historical decisions affect observed outcomes. We evaluate this aspect of problem formulation in LLM-generated code and explanations across child-welfare, healthcare, and employment settings, using 11 models from OpenAI and Anthropic and seven prompt conditions. We assess whether the LLM-generated code implements a procedure appropriate for estimating the intended outcome under the study's assumptions and whether the explanation acknowledges decision dependency. We find that LLMs do not reliably address decision dependency without guidance in the
    
[^159]: Option-GRPO：利用分层强化学习应对未见SDK代码生成中的AI垃圾代码问题

    Option-GRPO: Hierarchical RL Against AI Slop in Code Generation for Unseen SDKs

    [https://arxiv.org/abs/2611.23633](https://arxiv.org/abs/2611.23633)

    该论文提出Option-GRPO分层强化学习框架，仅凭SDK源代码（无需人工示例或测试用例）即可将代码大模型适配到未见过的SDK，通过从源代码中挖掘可复用代码模式（code options）作为课程与动作空间，同时优化功能正确性与代码简洁优雅性，从而抑制AI垃圾代码的产生。

    

    大型语言模型在生成的代码中越来越多地产生“AI垃圾代码”：冗长、冗余、非惯用的代码实现，这些代码虽能通过功能检查，却会使代码库臃肿不堪。这一问题在模型训练截止日期之后发布的SDK上尤为严重，因为模型既缺乏惯用写法的先验知识，也缺乏对SDK的组合性理解。我们提出了Option-GRPO，这是一个分层强化学习框架，仅凭SDK源代码即可将代码大语言模型适配到未见过的SDK上，无需人工编写的使用示例或测试用例，并且明确地同时优化功能正确性与代码优雅性。我们的核心洞察在于：SDK源代码中蕴含的组合性结构定义了一个可复用代码模式（代码选项）的层次体系，这些代码选项既可以作为课程学习的构建模块，也可以作为高效策略优化的动作空间。我们证明了代码选项可以诱导出有效的半马尔可夫决策过程，并给出了条件边界……

    arXiv:2611.23633v1 Announce Type: cross  Abstract: Large language models increasingly produce AI slop in generated code: verbose, redundant, non-idiomatic implementations that pass functional checks but bloat codebases. The problem compounds on SDKs released after the model's training cutoff, where the model has neither idiomatic priors nor compositional understanding. We present Option-GRPO, a hierarchical reinforcement learning framework that adapts a code LLM to an unseen SDK from its source code alone, without human-written usage examples or test cases, and that explicitly optimizes for both functional correctness and code elegance. Our key insight is that SDK source code contains compositional structure that defines a hierarchy of reusable code patterns (code options) which simultaneously serve as the building blocks of a curriculum and the action space for efficient policy optimization. We show that code options induce a valid semi-Markov decision process, give conditional bounds
    
[^160]: FairPGT：面向双层公平图聚类的分区图Transformer

    FairPGT: Partitioned Graph Transformers for Dual-Level Fair Graph Clustering

    [https://arxiv.org/abs/2611.23631](https://arxiv.org/abs/2611.23631)

    提出FairPGT，通过METIS分区与路由token降低注意力计算成本，并在聚类分配（互信息约束促进人口统计平等）和注意力机制（nCGAR自蒸馏实现群体中立）两个层面实现公平图聚类。

    

    在属性图上进行聚类可能产生带有人口统计学偏差的聚类结果，尤其是当社区结构与受保护群体重合时。图Transformer会通过同组注意力放大这种偏差，而全局注意力又限制了可扩展性。我们提出FairPGT，一种用于公平图聚类的分区图Transformer。FairPGT将注意力限制在METIS分区内，并通过路由token将各分区连接起来，从而将注意力的二次方计算开销除以分区数量。图感知的自训练目标引导聚类过程，其与标准DEC目标的偏差可被证明是有界的。公平性在两个层面被施加：互信息项促进聚类分配中的人口统计平等，而归一化跨组注意力比率通过自蒸馏引导注意力趋向群体中立。我们证明了nCGAR修正能够使注意力向中立方向移动，并建立了一个条件来预防……（摘要在此处被截断）

    arXiv:2611.23631v1 Announce Type: cross  Abstract: Clustering on attributed graphs can produce demographically biased clusters, particularly when communities align with protected groups. Graph transformers can amplify this bias through same-group attention, while global attention limits scalability. We propose FairPGT, a partitioned graph transformer for fair graph clustering. FairPGT restricts attention to METIS partitions and connects them through routing tokens, which divides the quadratic cost of attention by the number of partitions. A graph-aware self-training target guides clustering, with a provably bounded deviation from the standard DEC target. Fairness is imposed at two levels: a mutual-information term promotes demographic parity in cluster assignments, while a normalized cross-group attention ratio (nCGAR) guides attention toward group neutrality through self-distillation. We prove that the nCGAR correction moves attention toward neutrality and establish a condition preven
    
[^161]: 插值器存储了什么、表达了什么：一个用于良性过拟合与无标签模型选择的信息论框架

    What Interpolators Store and What They Express: An Information-Theoretic Framework for Benign Overfitting and Label-Free Model Selection

    [https://arxiv.org/abs/2611.23630](https://arxiv.org/abs/2611.23630)

    本文提出区分插值器“存储”与“表达”的信息论框架，证明所有插值器存储等量标签噪声故存储类泛化界（如CMI）无法区分它们，而“表达”才能刻画超额风险，从而解释良性过拟合并实现无需标签的模型选择。

    

    过参数化学习器能够完美拟合含噪的训练标签，同时仍具有良好的预测能力，这一现象既无法用经典复杂度度量解释，也无法用信息论泛化界解释。我们提出了一个信息论框架，将插值型学习器对样本所做的两件事区分开来：存储，即训练后的模型所保留的关于其训练标签的信息；表达，即关于样本的信息中最终体现在其对新鲜输入的预测中的部分。我们首先刻画了理想模型：对于某个损失 l，最不复杂的 l-充分表示是贝叶斯动作，其熵即为最小充分复杂度；这一结果推广并修正了先前关于提升方法的“无损最大压缩”观点。接着我们证明，每个插值器都恰好存储 nH(Y|X) 比特的标签噪声，因此存储以及基于存储构建的界（包括条件互信息界）无法区分不同的插值器。而表达则可以：超额风险由表

    arXiv:2611.23630v1 Announce Type: new  Abstract: Overparameterized learners can fit noisy training labels perfectly and still predict well, which neither classical complexity measures nor information-theoretic generalization bounds explain. We develop an information-theoretic framework that separates two things an interpolating learner does with its sample: storage, the information the trained model retains about its training labels, and expression, the information about the sample that reaches its predictions at fresh inputs. We first characterize the ideal model: for a loss l, the least complex l-sufficient representation is the Bayes act, whose entropy is the minimal sufficient complexity; this generalizes, and corrects, an earlier "lossless maximal compression" view of boosting. We then show that every interpolator stores exactly nH(Y|X) bits of label noise, so storage, and bounds built on it (including CMI bounds), cannot tell interpolators apart. Expression can: excess risk is bo
    
[^162]: 二胜于一：优化在线平台的二元评分量表

    Two is better than one: optimizing binary rating scales for online platforms

    [https://arxiv.org/abs/2611.23626](https://arxiv.org/abs/2611.23626)

    该论文证明在线平台通过为不同用户使用不同的二元评分量表而非单一量表——包括将单一量表拆分为两个、将多量表最优设计刻画为阈值均匀间隔的0-1阶跃函数、以及引入依赖先前反馈的自适应量表——可以显著提升区分物品内在质量的排名能力，其中自适应设计能带来指数级的提升。

    

    在线平台常常受到夸大的、缺乏信息量的用户评分的困扰。这一问题可以通过更好地设计数据收集机制来缓解，例如改变用户问题的措辞。以往的工作优化单一的评分量表，以便在二元反馈下最好地区分物品的内在质量。在本文中，我们展示了对不同用户使用不同评分量表的力量。首先，我们证明设计者总是可以将单一评分量表拆分为两个，从而改进物品的排名。接下来，我们形式化地刻画了多量表的最优设计，即一系列阈值均匀间隔的0-1阶跃函数。最后，我们考虑了一个自适应模型，其中后续的评分量表可以依赖于先前用户的反馈，并且物品根据其完整的反馈序列进行排名，我们证明自适应性带来了指数级更强的区分能力。总而言之，

    arXiv:2611.23626v1 Announce Type: cross  Abstract: Online platforms often struggle with inflated, uninformative user ratings. This issue can be mitigated by better design of the data collection mechanism, for example through changing the phrasing of user questions. Previous work optimizes a single rating scale to best distinguish the inherent qualities of items under binary feedback. In this paper, we demonstrate the power of utilizing different rating scales for different users. First, we show that a designer can always split a single rating scale into two to improve item rankings. Next, we formally characterize the optimal design of multiple scales as a series of 0-1 step functions whose threshold values are evenly spaced. Finally, we consider an adaptive model where subsequent rating scales can depend on the responses from previous users and items are ranked by their full sequence of responses, and show that adaptivity leads to exponentially greater distinguishing power. All in all,
    
[^163]: 反事实分析中的智能体验证器：智能体化异质性处理效应基准测试的试点研究

    Agent Verifiers in Counterfactual Analysis: A Pilot Study Benchmarking Agentic Heterogeneous Treatment Effects

    [https://arxiv.org/abs/2611.23621](https://arxiv.org/abs/2611.23621)

    本文提出一个包含检索可采性、基于RATE的异质性验证和编码器可采性检验的三层基准协议，用于防范检索泄露与编码器泄露，从而可靠地评估智能体在异质性处理效应估计中的表现。

    

    智能体可以通过检索快速变化的局部上下文（这些信息往往被变化缓慢的问卷调查基线协变量所遗漏）来扩展随机实验的协变量空间，从而帮助刻画哪些人群最能从某项干预中受益。然而验证这类协变量十分困难：个体处理效应本身不可观测；检索泄露可能将处理后信息混入名义上的处理前摘要；而编码器泄露即使在输入经过时间过滤后，仍可能通过预训练知识将处理后信息注入到表示之中。在一个区分摘要来源与编码器权重的双通道信息界限的指导下，我们提出了一个三层基准测试协议：(i) 通过处理前日期过滤实现检索可采性检验；(ii) 以排序加权平均处理效应（RATE）为评分标准的异质性验证；(iii) 借助按时间分版本的编码器、未来事实探针和权重扰动来检验编码器可采性。在11个数据（集）上的实验……

    arXiv:2611.23621v1 Announce Type: new  Abstract: Agents can expand the covariate space of randomized experiments by retrieving fast-moving local context missed by slow-changing survey baseline covariates, helping characterize who benefits most from an intervention. Verifying such covariates is difficult: individual treatment effects are unobserved, retrieval leakage can pull post-treatment information into nominally pretreatment summaries, and encoder leakage can inject post-treatment pretraining knowledge into representations even after temporal filtering of inputs. Guided by a two-channel information bound separating summary provenance from encoder weights, we propose a three-layer benchmark protocol: (i) retrieval admissibility via pre-treatment date filtering; (ii) heterogeneity validation scored rank-weighted average treatment effects (RATEs); and (iii) encoder admissibility checked with chronologically vintaged encoders, future-fact probes, and weight perturbations. Across 11 dat
    
[^164]: 我的动作有多不确定？基于冻结世界模型的自校正视觉规划

    How Uncertain Is My Action? Self-Correcting Visual Planning with Frozen World Models

    [https://arxiv.org/abs/2611.23620](https://arxiv.org/abs/2611.23620)

    该论文提出将冻结的潜在世界模型反向运行，通过对比假设偏移下的预测观测与实际观测来检测、估计并补偿执行偏移，使视觉规划器具备自校正能力，检测性能与理论预测相差不到0.05 AUROC。

    

    冻结的潜在世界模型与基于采样的模型预测控制相结合，构成了一种视觉规划器。在部署时，实际执行的动作可能偏离指令动作，而冻结模型无法察觉这一点。我们展示了该模型可以反向运行：对于任何假设的执行偏移，它可以预测在该偏移下应该观测到什么，将这一预测与实际观测进行比较即可量化偏移，并让规划器自我校正。与将偏移和模型自身误差混杂在一起的残差能量不同，该分数仅保留偏移所能产生的残差部分。这一偏移分数通过局部最优检验来检测偏移，用Cramer-Rao误差棒估计偏移大小，识别出主导生成器，并在警报触发时对规划器进行补偿。在PushT、TwoRoom和OGBench-Cube上使用冻结的潜在世界模型，实际检测性能与理论预测的差距在0.05 AUROC以内。

    arXiv:2611.23620v1 Announce Type: cross  Abstract: A frozen latent world model combined with sampling-based model predictive control is a visual planner. At deployment, the executed action can depart from the commanded one, and the frozen model has no way to notice. We show that the model can be run in reverse: for any hypothesized actuation shift, it predicts what it should have observed under that shift, and comparing this prediction with what was actually observed quantifies the shift and lets the planner correct itself. Unlike the residual energy, which mixes the shift with the model's own error, the score keeps only the part of the residual that a shift can produce. This shift score detects the shift with a locally optimal test, estimates it with Cramer-Rao error bars, names the dominant generator, and compensates the planner once an alarm fires. On PushT, TwoRoom and OGBench-Cube with a frozen latent world model, measured detection follows the prediction to within 0.05 AUROC. Sel
    
[^165]: EEG2MCU：面向可穿戴脑电图推理的感知到执行协同设计

    EEG2MCU: Sensing-to-Execution Co-Design for Wearable EEG Inference

    [https://arxiv.org/abs/2611.23619](https://arxiv.org/abs/2611.23619)

    EEG2MCU通过任务感知的电极蒙太奇选择将电极数量作为硬件设计变量，并结合按需生成INT8区域和跨内存的Halo重用等协同设计技术，在Cortex-M7微控制器上实现了峰值SRAM降低11.38倍、延迟降低2.57倍、能耗降低2.48倍，同时保留了94-96%的全蒙太奇准确率。

    

    可穿戴脑电图（EEG）推理不仅受模型规模的限制，还受制于必须在紧凑的下游计算之前处理高维输入的依赖通道的感知前端。我们提出了EEG2MCU，这是一个感知到执行协同设计框架，通过任务感知的电极蒙太奇选择，将电极数量作为硬件设计变量。EEG2MCU保留紧凑的时频表示，并仅生成每个第一层分块所需的INT8区域，避免了完整输入的物化。跨Flash、SRAM和紧耦合内存的Halo重用与访问感知布局进一步减少了冗余生成和数据移动。在Cortex-M7微控制器上，EEG2MCU相对于基线将峰值SRAM占用降低11.38倍（其中62通道到19通道蒙太奇缩减贡献了1.71倍），同时将延迟降低2.57倍，能耗降低2.48倍。它保留了完整蒙太奇准确率的94-96%，展示了实用的高维EEG推理能力。

    arXiv:2611.23619v1 Announce Type: new  Abstract: Wearable EEG inference is constrained not only by model size, but also by the channel-dependent sensing front end that must process high-dimensional inputs before compact downstream computation. We present EEG2MCU, a sensing-to-execution co-design framework that treats electrode count as a hardware design variable through task-aware montage selection. EEG2MCU retains a compact time-frequency representation and generates only the INT8 regions required by each first-layer tile, avoiding full input materialization. Halo reuse and access-aware placement across Flash, SRAM, and tightly coupled memory further reduce redundant generation and data movement. On a Cortex-M7 MCU, EEG2MCU reduces peak SRAM by 11.38x relative to the baseline, including a 1.71x reduction from the 62-to-19 electrode montage, while reducing latency by 2.57x and energy by 2.48x. It retains 94-96% of full-montage accuracy, demonstrating practical high-dimensional EEG infe
    
[^166]: 不完美分数下反向扩散的误差累积：超越路径空间分析

    Error accumulation in reverse diffusion with an imperfect score: beyond path-space analyses

    [https://arxiv.org/abs/2611.23617](https://arxiv.org/abs/2611.23617)

    本文突破了传统路径空间分析框架，证明当目标分布满足 Poincaré 不等式时，扩散模型采样过程中的误差累积可以得到有效缓解。

    

    arXiv:2611.23617v1 公告类型：new 摘要：扩散建模是一种从复杂高维分布 $p$ 中采样的流行方法，其通过对（加噪）分数函数的学习近似来实现。生成单个样本需要模拟反向扩散过程，这需要对学习到的分数函数进行大量查询。因此，先验地看，学习误差很容易复合累积。事实上，在现有理论分析中，采样误差（以与 $p$ 的总变差距离度量）随随机过程的时间范围呈多项式级累积。这是由于现有方法通过整个反向扩散路径的分布误差来界定输出分布的误差。本文中，我们证明通过绕开“路径空间”分析，当 $p$ 满足 Poincaré 不等式时，扩散采样器的误差累积可以得到缓解。此外，该结果是作为更广泛框架的特例而实现的，该框架适用于任何分布 $p。

    arXiv:2611.23617v1 Announce Type: new  Abstract: Diffusion modeling is a popular method of sampling from a complex, high-dimensional distribution $p$ using learned approximations of its (noised) score functions. Generating a single sample requires simulating the reverse diffusion process, which requires many queries to the learned scores. Thus, a priori, the learning errors can easily compound. Indeed, in existing theoretical analyses, the sampling error (as measured in total variation distance to $p$) accumulates polynomially with the horizon of the stochastic process. This is a consequence of bounding the error of the output law by the error of the law of the entire reverse diffusion path.   In this paper, we demonstrate that by circumventing the "path-space" analysis, the error accumulation of the diffusion sampler can be mitigated when $p$ satisfies a Poincar\'e inequality. Moreover, we achieve this result as a special case of a broader framework that applies to any distribution $p
    
[^167]: Isthmus（地峡）：桥接监督孤岛，恢复单步流的不确定性与稳定性

    Isthmus: Bridging Supervision Islands Restores Uncertainty and Stability in One-Step Flows

    [https://arxiv.org/abs/2611.23616](https://arxiv.org/abs/2611.23616)

    该论文揭示单步流采样因监督时间对落在与数据对角线断开的“监督孤岛”上而丢失不确定性与稳定性，并提出用“地峡”桥接孤岛以恢复单步流的不确定性和稳定性。

    

    单步流映射日益被用作后验采样器，其采样的意义在于样本所携带的不确定性。我们发现单步采样器会丢失这种不确定性并变得不稳定。与同一网络的多步采样结果相比，其样本失去方差且失去相关性。直接监督单步跳跃会使问题更糟：样本进一步向条件均值坍缩，并且在延长训练时崩溃。我们将这两类失败追溯到训练中起始与结束时间对的放置位置。只有位于时间三角形对角线上的时间对才能从数据获得目标，而用于单步采样的时间对要么完全没有监督，要么被监督在一座“孤岛”上——即一组未被自洽恒等式连接到对角线的时间对，此时目标函数既无法确定单步映射，也无法锚定其训练。我们用一条“地峡”来桥接这座孤岛（摘要原文在此处截断）

    arXiv:2611.23616v1 Announce Type: cross  Abstract: One-step flow maps are increasingly used as posterior samplers, where the point of sampling is the uncertainty that the samples carry. We find that the one-step sampler loses this uncertainty and becomes less stable. Compared with multi-step samples of the same network, its samples lose variance and decorrelate. Supervising the one-step jump directly makes this worse; the samples collapse further toward the conditional mean and break down when training is extended. We trace both failures to the locations where training places its pairs of start and end times. Only pairs on the diagonal of the time triangle receive targets from data, and the pair used for one-step sampling is either left unsupervised or supervised on an island, a set of pairs that the self-consistency identity does not connect to the diagonal, where the objective neither determines the one-step map nor anchors its training. We bridge the island with an isthmus, which pl
    
[^168]: 光学神经网络应当按功能而非仅按物理来评估

    Optical Neural Networks Should Be Evaluated by Function, Not Only Physics

    [https://arxiv.org/abs/2611.23610](https://arxiv.org/abs/2611.23610)

    本文提出一种面向功能的三步评估方法，通过将光学神经网络模块与性能匹配的标准数字模块进行对比，来客观衡量其对系统级优势（如速度和能耗）的真实贡献。

    

    光学神经网络（ONN）作为机器学习（ML）的替代平台正受到广泛关注。在光学神经网络中，光学模块通常接管数字机器学习流水线中的部分计算。那么，光学模块本身到底完成了任务的多少部分，它又对所声称的系统级优势（如吞吐量或能耗）贡献了多少？本文提出了一种简单的、面向功能的评估方法。无论其内部物理机制如何，光学模块对周围的数字流水线而言都表现为实值张量之间的映射。因此，我们分三步对其进行评估：(i) 定义其在流水线中的边界，即以实值张量为输入和输出；(ii) 用标准数字模块替换它，以找出在同一任务和相同指标下能与光学系统性能匹配的模块；(iii) 针对性能匹配的基线，按每次推理来陈述系统级主张，如速度或能耗。

    arXiv:2611.23610v1 Announce Type: cross  Abstract: Optical neural networks (ONNs) are attracting significant attention as alternative platforms for machine learning (ML). In an ONN, an optical module often takes over part of the computation inside a digital ML pipeline. How much of the task does the optical module itself perform, and how much does it contribute to any claimed system-level advantages, such as throughput or energy?   This paper offers a simple, function-oriented evaluation methodology. Whatever its internal physics, the optical module appears to the surrounding digital pipeline as a mapping between real-valued tensors. We therefore evaluate it in three steps: (i) define its boundary in the pipeline, as real-valued tensors in and out; (ii) replace it with standard digital modules to identify those that match the optical system's performance on the same task and metric; and (iii) state system-level claims, such as speed or energy, per inference against the performance-matc
    
[^169]: 无标签的课堂录音角色感知分割：从可迁移的结构中恢复教师角色

    Label-Free Role-Aware Segmentation of Classroom Recordings: Recovering the Teacher from Transferable Structure

    [https://arxiv.org/abs/2611.23609](https://arxiv.org/abs/2611.23609)

    该论文提出了一种无需标签、可在录音设备本地运行的师生语音分割流水线，通过基于会话相对结构而非声音身份的方式来识别教师角色，并能跨教室、语言和声学条件迁移。

    

    教师说话的多少及其发生时机，是教学中最具信息量、同时也最难测量的信号之一。然而，课堂音频是人工智能系统所能处理的最敏感数据之一：录音涉及未成年人并受到GDPR法规约束，因此原始音频不得离开录音设备。我们提出了一种专为满足这一约束而设计的师生语音分割流水线：推理阶段无需标签、可在录音设备上运行，且无需针对每次会话进行校准。该流水线串联了一个轻量级语音活动检测器、紧凑的预训练说话人嵌入、凝聚式聚类，以及一个小型逻辑回归模型——该模型基于会话内的相对结构而非声音身份来分配教师角色。我们证明了这种结构线索能够跨教室、跨语言、跨声学条件进行迁移。在来自英语和德语语料库、共63位教师的292段录音上进行留一教师交叉验证，我们的（摘要在此处截断）

    arXiv:2611.23609v1 Announce Type: new  Abstract: How much the teacher talks and when are among the most informative and costly-to-measure signals of teaching. Yet, classroom audio is among the most sensitive data an AI system can process: recordings capture minors and fall under GDPR, preventing raw audio from leaving the recording device. We present a teacher-student segmentation pipeline designed to meet exactly this constraint: label-free at inference, running on a recording device, and requiring no per-session calibration. The pipeline chains a lightweight voice activity detector, compactly pretrained speaker embeddings, agglomerative clustering, and a small logistic regression that assigns the teacher role based on session-relative structure rather than voice identity. We show that our structural cue transfers across classrooms, languages, and acoustic conditions.   In leave-one-teacher-out cross-validation on 292 recordings from 63 teachers across English and German corpora, our 
    
[^170]: JEPA作为可迁移的优化器：从奖励无关的预训练中学习与理解黑盒优化

    JEPA as a Transferable Optimizer: Learning and Understanding Black-Box Optimization from Reward-Agnostic Pretraining

    [https://arxiv.org/abs/2611.23608](https://arxiv.org/abs/2611.23608)

    提出了基于联合嵌入预测架构的学习型优化器JEPA-PS，通过奖励无关的预训练学习预测优化轨迹中的下一个点而非奖励本身，测试时仅需一次前向传播即可提出候选解，并在15个控制环境中与12种优化器对比验证了其性能，同时增强了对优化器如何利用搜索历史的可解释性。

    

    黑盒优化通过仿真、轨迹采样或实验来评估候选解，通常无法获得梯度。学习型优化器通常以是否优于现有方法来评判。我们提出了一个互补的问题：学习型优化器从其搜索历史中究竟使用了什么信息？我们提出了基于联合嵌入预测架构（JEPA）的JEPA-PS（JEPA策略搜索）。它以参数向量和标量回报为输入，在随机轨迹上进行预训练。其潜空间预测目标预测的是优化轨迹中的下一个点，而非其奖励或梯度，这使我们能够考察“尝试了什么”以及“表现如何”。在测试时，JEPA-PS读取一段简短的评估历史，仅需一次前向传播即可提出候选解，无需任务描述、奖励塑形或测试时的梯度。我们在15个控制环境中与12种优化器进行了对比评估，其中包括进化方法和贝叶斯优化等方法。

    arXiv:2611.23608v1 Announce Type: cross  Abstract: Black-box optimization evaluates candidates through simulations, rollouts, or experiments, often without gradients. Learned optimizers are usually judged by whether they outperform existing methods. We ask a complementary question: what information does a learned optimizer use from its search history? We introduce JEPA-PS (JEPA Policy Search), based on a Joint-Embedding Predictive Architecture. It is pretrained on random rollouts using parameter vectors and scalar returns. Its latent-prediction objective predicts the next point in an optimization trajectory rather than its reward or gradient, allowing us to examine what was tried and how well it performed. At test time, JEPA-PS reads a short evaluation history and proposes a candidate in one forward pass, without task descriptions, reward shaping, or test-time gradients. We evaluate it on 15 control environments against 12 optimizers, including evolutionary methods, Bayesian optimizati
    
[^171]: 用于稳定特征归因的熵正则化积分梯度方法

    Entropy-Regularized Integrated Gradients for Stable Feature Attribution

    [https://arxiv.org/abs/2611.23600](https://arxiv.org/abs/2611.23600)

    提出熵正则化积分梯度（ERIG），通过带温度参数的最大熵softmax加权抑制积分路径上的梯度不规则性，并从理论上证明其归因图对微小输入扰动具有更强的稳定性。

    

    基于路径的归因方法（如积分梯度，IG）被广泛用于解释神经网络，然而不可察觉的微小输入扰动可能会显著改变其归因图，而模型预测却保持不变。我们提出了熵正则化积分梯度（ERIG），这是一种基于路径的归因方法，通过使用带温度参数 $\tau$ 的最大熵softmax加权，从内在机制上抑制积分路径上的梯度不规则性。我们利用模型及其梯度的Lipschitz性质来量化熵正则化加权对归因稳定性的影响。我们将输入扰动下的归因变化分解为一个模型响应项（其界与 $\tau$ 无关）和一个权重变化项（其界以 $\mathcal{O}(1/(2\tau))$ 的速率减小）。二者的比值 $R(\tau)$ 刻画了梯度信息与归因稳定性之间的权衡。

    arXiv:2611.23600v1 Announce Type: cross  Abstract: Path-based attribution methods like Integrated Gradients (IG) are widely used to interpret neural networks, yet imperceptible input perturbations can drastically alter their attribution maps while leaving model predictions unchanged. We introduce Entropy-Regularized Integrated Gradients (ERIG), a path-based attribution method that intrinsically attenuates gradient irregularities along the integration path using maximum-entropy softmax weighting with temperature parameter $\tau$. We quantify the effect of the entropy-regularized weighting on attribution stability by using the Lipschitz properties of the model and its gradients. We decompose the attribution change under an input perturbation into a model-response term, whose bound is $\tau$-independent, and a weight-variation term, whose bound decreases at rate $\mathcal{O}(1/(2\tau))$. Their ratio $R(\tau)$ characterizes the trade-off between gradient information and attribution stabili
    
[^172]: DuoCL：双路径在线持续学习

    DuoCL: Dual-Path Online Continual Learning

    [https://arxiv.org/abs/2611.23598](https://arxiv.org/abs/2611.23598)

    DuoCL通过结合快速解析学习路径与慢速骨干网络适应路径的双路径框架，在保证在线学习不间断的前提下自适应计算预算，相比基于梯度的基线方法平均提升在线准确率9.67个百分点。

    

    在线持续学习（OCL）要求模型从数据流中学习，而数据流的到达速率与模型训练速度无关。当训练速度跟不上数据到达速度时，计算开销较大的基于梯度的方法更新频率会降低，从而限制了模型的适应能力。解析方法虽然降低了这种开销，但其冻结的骨干网络无法使表征适应不断变化的分布。我们提出了DuoCL，这是一个双路径OCL框架：其快速解析路径从每个传入的小批量数据中累积特征统计信息，而其慢速路径则周期性地对骨干网络进行适应。慢速路径可根据可用的计算预算调整其处理的样本数量，且不会中断快速路径的学习。特征对齐机制补偿了累积分类器统计信息中的漂移。在四个数据集和三种计算预算下，DuoCL在每种设置中相比最强的基于梯度的基线方法，平均将在线准确率提升了9.67个百分点。

    arXiv:2611.23598v1 Announce Type: new  Abstract: Online continual learning (OCL) requires learning from a data stream whose arrival rate is independent of model training speed. When training cannot keep pace, computationally expensive gradient-based methods update less frequently, limiting adaptation. Analytic methods reduce this overhead, but their frozen backbones cannot adapt representations to changing distributions. We propose DuoCL, a dual-path OCL framework whose fast analytic path accumulates feature statistics from every incoming minibatch while its slow path periodically adapts the backbone. The slow path adjusts its sample count to the available compute budget without interrupting fast-path learning. Feature alignment compensates for drift in the accumulated classifier statistics. Across four datasets and three computational budgets, DuoCL improves online accuracy by an average of 9.67 percentage points over the strongest gradient-based baseline in each setting. DuoCL consis
    
[^173]: 时变非齐次线性马尔可夫决策过程中的高效方差感知强化学习

    Efficient Variance-Aware Reinforcement Learning in Time-Inhomogeneous Linear MDPs

    [https://arxiv.org/abs/2611.23596](https://arxiv.org/abs/2611.23596)

    本文提出计算高效的算法 VA-LSVI-UCB++，在时变非齐次线性 MDP 中实现了方差依赖的遗憾界，并消除了以往方差感知分析中即使在确定性环境下仍然存在的 $\tilde{O}(K^{1/4})$ 多项式加性项。

    

    我们研究在具有特征维度 $d$、时间跨度 $H$ 和 $K$ 个回合的回合制时变非齐次线性马尔可夫决策过程（MDP）中的方差依赖强化学习问题。尽管近期工作已为线性混合 MDP 建立了具有 $\tilde{O}(1)$ 确定性情形开销的方差依赖遗憾界（Zhao et al., 2023），并为线性 MDP 建立了最坏情形下接近极小极大最优的遗憾 $\tilde{O}(d\sqrt{H^3 K} + d^7 H^8)$（He et al., 2023），但现有的针对线性 MDP 的方差感知分析（Li and Sun, 2023; Huang et al., 2023）依赖于一个粗糙的集中不等式步骤，导致其遗憾保证即使在确定性环境中仍保留多项式 $\tilde{O}(K^{1/4})$ 的加性项。我们提出 VA-LSVI-UCB++，这是一种计算高效的算法，实现了方差依赖的遗憾界 $\tilde{O}(d\sqrt{H\min\{\mathrm{Var}_K^*,\mathrm{Var}_K^\pi\}}+d^5H^4)$，其中 $\mathrm{Var}_K^*$ 和 $\mathrm{Var}_K^\pi$ 分别表示最优策略和策略 $\pi$ 的累积方差（摘要在此处被截断）。

    arXiv:2611.23596v1 Announce Type: new  Abstract: We study variance-dependent reinforcement learning in episodic time-inhomogeneous linear Markov decision processes (MDPs) with feature dimension $d$, horizon $H$, and $K$ episodes. While recent works have established variance-dependent regret with $\tilde{O}(1)$ deterministic-regime overhead for linear mixture MDPs (Zhao et al., 2023) and worst-case nearly minimax optimal regret $\tilde{O}(d\sqrt{H^3 K} + d^7 H^8)$ for linear MDPs (He et al., 2023),, existing variance-aware analyses for linear MDPs (Li and Sun, 2023; Huang et al., 2023) rely on a coarse concentration step whose resulting regret guarantees retain a polynomial $\tilde{O}(K^{1/4})$ additive term even in deterministic environments. We propose VA-LSVI-UCB++, a computationally efficient algorithm that achieves the variance-dependent regret bound $\tilde{O}(d\sqrt{H\min\{\mathrm{Var}_K^*,\mathrm{Var}_K^\pi\}}+d^5H^4)$, where $\mathrm{Var}_K^*$ and $\mathrm{Var}_K^\pi$ denote th
    
[^174]: GRAFT：面向视觉语言模型现实测试时适应的图正则化原型场

    GRAFT: Graph-Regularized Prototype Fields for Realistic Test-Time Adaptation of VLMs

    [https://arxiv.org/abs/2611.23591](https://arxiv.org/abs/2611.23591)

    该论文提出GRAFT方法，将适应后的原型建模为文本锚点加上经语义类别图上高斯马尔可夫随机场先验部分池化的位移场，并引入显式类别存在模型，实现闭式无梯度的测试时适应，从而在少类别批次或相关数据流等现实场景下优于零样本基线。

    

    无标注测试数据可以改进视觉语言模型的零样本分类器，然而当批次中仅出现少量词汇类别或数据流存在相关性时，直推式与测试时适应方法往往低于零样本基线。我们将这一失败归因于一个共同的设计缺陷：每个类别拥有从其自身样本估计的独立参数，导致缺失或稀有类别要么停留在有偏的文本锚点上，要么被拟合到不相干的样本上。我们提出图正则化锚点位移场直推方法（GRAFT），将适应后的原型建模为文本锚点加上一个位移场，该位移场通过高斯马尔可夫随机场先验在语义类别图上进行部分池化，并辅以显式的类别存在性模型。推理在缓存特征上以闭式、无梯度的方式进行。仅锚点收缩与单一全局位移是该先验的两个极端设定；缺失类别可被证明……

    arXiv:2611.23591v1 Announce Type: cross  Abstract: Unlabeled test data can improve the zero-shot classifier of a vision-language model, yet transductive and test-time adaptation methods often fall below the zero-shot baseline when only a few vocabulary classes are present in a batch or when streams are correlated. We trace this failure to a shared design: every class owns independent parameters estimated from its own samples, so absent or rare classes are left at biased text anchors or fitted to foreign samples. We propose Graph-Regularized Anchor-displacement Field Transduction (GRAFT), which models the adapted prototypes as text anchors plus a displacement field that is partially pooled over a semantic class graph through a Gaussian Markov random field prior, together with an explicit class-presence model. Inference is closed-form and gradient-free on cached features. Anchor-only shrinkage and a single global shift are the two extreme settings of the prior; absent classes are provabl
    
[^175]: 可以安全省略，也值得省略吗？当LLM智能体跳过另一方的消息时，保持答案不变并降低延迟

    Safe to Omit, Worth Omitting? Keeping the Answer and Cutting Latency When an LLM Agent Skips Another's Message

    [https://arxiv.org/abs/2611.23570](https://arxiv.org/abs/2611.23570)

    该论文提出一种基于无消息置信度读取与有限样本统计检验的机制，使多智能体LLM系统中的接收方智能体能够以高概率保证答案改变率不超过5%地安全跳过辅助智能体的消息，在保持最终答案质量的同时有效降低通信延迟，并揭示了消息增益远小于oracle上限的量化证据。

    

    在多智能体系统中，接收方智能体通常需要等待辅助智能体的消息（文本或其潜在KV缓存），这可以提升性能但会增加延迟。我们研究接收方智能体何时可以安全地（最终答案很少改变）且有益地（节省的时间超过决策本身的开销）跳过该消息。我们的机制在不读取该消息的情况下获取接收方智能体的置信度，并使用有限样本检验来认证一个阈值——在该阈值下，以高概率保证被跳过时答案的改变率不超过5%；如果没有阈值能通过检验，系统则继续保持通信。我们在OpenBookQA、ARC-Challenge和MMLU-Pro上使用三对Qwen模型进行测试，也在跨家族的智能体组合以及SQuAD和GSM8K上进行了测试。我们得到四点洞察。首先，“oracle上限空间”是消息有帮助作用的弱证据：不含答案特定内容、但被匹配为与真实消息同样频繁改变答案的控制输入，能够重现oracle收益的中位数76%。其次，一条消息能否……（摘要原文在此处截断）

    arXiv:2611.23570v1 Announce Type: cross  Abstract: In multi-agent systems, a receiver agent often waits for a helper agent's message (text or its latent KV cache), which can improve performance but adds latency. We study when a receiver agent can skip the message safely (the final answer rarely changes) and beneficially (the time saved exceeds the cost of deciding). Our mechanism reads the receiver agent's confidence without the message and uses a finite-sample test to certify a threshold at which, with high probability, at most 5% of skipped answers change; if none passes, the system keeps communicating. We test it on OpenBookQA, ARC-Challenge, and MMLU-Pro with three Qwen model pairs, on cross-family agents, and on SQuAD and GSM8K. Four insights emerge. First, oracle headroom is weak evidence that messages help: control inputs without answer-specific content, matched to change answers as often as real messages, reproduce a median 76% of the oracle's gain. Second, whether a message ca
    
[^176]: 潜在结构块模型中的有界阶数不可辨识性：闭式纤维解

    Bounded-Order Non-Identifiability in Latent Structure Block Models: The Fiber in Closed Form

    [https://arxiv.org/abs/2611.23547](https://arxiv.org/abs/2611.23547)

    该论文为位置位于K条射线上的潜在结构块模型给出了有限阶模体计数下不可辨识配置的一维纤维闭式解，并证明友谊图F_{K-1}是唯一边最小分离子，因此不存在对所有K都适用的固定模体阶数。

    

    图子由其所有同态密度完全决定，但基于模体的方法只使用有限多个。我们在一个自然的模型族中精确计算了这种截断的代价：位置位于K条射线上的潜在结构块模型，等价于秩为2的度校正块模型。对于给定的径向律，在所有少于2K-1个顶点的图上一致的配置构成一个一维纤维，我们以带有显式端点的闭式形式给出该纤维；友谊图F_{K-1}是其唯一的边最小分离子，因此不存在对所有K都适用的固定模体阶数。分离信号最多被抑制m_1^{2(K-1)}倍，其中m_1是均方斜率，在K=4时约为10^{-4}：在任何边密度下，从模体计数中检测该信号需要单个图中约10^{15}个节点。然而，信息仍然存在于图中；纤维属于统计量本身，而非模型。常见的谱方法流程……（原文摘要在此处截断）

    arXiv:2611.23547v1 Announce Type: cross  Abstract: A graphon is determined by all of its homomorphism densities, but motif-based methods use finitely many. We compute exactly what that truncation costs in a natural family: latent structure block models whose positions lie on $K$ rays, equivalently degree-corrected block models of rank two. For a given radial law, the configurations that agree on every graph with fewer than $2K-1$ vertices form a one-dimensional fiber, which we give in closed form with explicit endpoints; the friendship graph $F_{K-1}$ is its unique edge-minimal separator, so no fixed motif order serves every $K$. The separating signal is suppressed by a factor of at most $m_1^{2(K-1)}$, where $m_1$ is the mean squared slope, about $10^{-4}$ at $K=4$: detecting it from motif counts would take about $10^{15}$ nodes in a single graph, at any edge density. The information is nonetheless in the graph; the fiber belongs to the statistic, not the model. Common spectral pipeli
    
[^177]: 平行输运卷积在黎曼度量扰动下的稳定性

    Stability of Parallel Transport Convolution under Riemannian Metric Perturbations

    [https://arxiv.org/abs/2611.23543](https://arxiv.org/abs/2611.23543)

    本文证明了在紧流形上，平行输运卷积的输出在 $L^2$ 意义下对黎曼度量的 $C^2$ 扰动具有连续稳定性，为此发展了测地线、平行输运、雅可比场与指数映射的扰动估计技术。

    

    arXiv:2611.23543v1 公告类型： new 摘要： 黎曼流形上的几何卷积依赖于底层定义域的几何结构，因此度量扰动下的稳定性成为几何深度学习中的一个基本问题。本文借助微分几何与分析的方法研究了平行输运卷积的这一问题。在一个固定的紧流形上，我们证明：对于连续的输入信号以及支撑在充分小法向邻域内的连续参考核，在给定的局部化与输运路径约定下，卷积输出在 $L^2$ 意义下关于黎曼度量（$C^2$ 拓扑）是连续依赖的。分析过程建立了关于测地线、平行输运、雅可比场和指数映射的扰动估计，并利用这些估计来比较输运后的核。一个核心难点是割迹，它阻碍了对整个流形上的输运构造进行一致比较。

    arXiv:2611.23543v1 Announce Type: new  Abstract: Geometric convolutions on Riemannian manifolds depend on the geometry of the underlying domain, making stability under metric perturbations a fundamental question in geometric deep learning. This paper studies this question for parallel transport convolution through methods from differential geometry and analysis. On a fixed compact manifold, we establish that, for a continuous input signal and a continuous reference kernel supported in a sufficiently small normal neighborhood, the convolution output depends continuously in $L^2$ on the Riemannian metric in the $C^2$-topology, under the specified localization and transport-path conventions. The analysis develops perturbation estimates for geodesics, parallel transport, Jacobi fields, and exponential maps, and uses these estimates to compare the transported kernels. A central difficulty is the cut locus, which obstructs uniform comparison of the transport construction over the entire mani
    
[^178]: LLM能理解它们所预测的内容吗？评估金融市场中的潜在因果结构推断

    Can LLMs Understand What They Predict? Evaluating Latent Causal Structure Inference in Financial Markets

    [https://arxiv.org/abs/2611.23533](https://arxiv.org/abs/2611.23533)

    该论文提出LUCID基准，评估大语言模型能否从事件描述和历史金融数据中推断潜在的因果市场结构，发现模型生成的因果图通常无法改善重构质量，表明预测成功并不等于真正理解了底层机制。

    

    大语言模型（LLM）越来越多地通过其预测未来结果的能力来评估，这隐含地假设预测的成功反映了对底层系统的理解。然而，准确的预测往往可以在不恢复产生可观测结果的潜在机制的情况下实现。为了检验这一假设，我们引入了LUCID，这是一个评估模型能否从事件描述和历史金融观测数据中推断潜在市场结构的基准。模型生成代表所推断市场结构的事件诱导因果图，并通过基于重构的框架进行评估。我们对6个开源权重和专有LLM的评估表明，与仅使用历史观测数据相比，加入模型生成的因果图通常会降低重构质量，尽管少数指标有所改善。这些发现表明，推断有信息量的、针对特定事件的市场（结构）……

    arXiv:2611.23533v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly evaluated through their ability to predict future outcomes, implicitly assuming that predictive success reflects an understanding of the underlying system. However, accurate predictions can often be achieved without recovering the latent mechanisms that generate observable outcomes. To examine this assumption, we introduce LUCID, a benchmark that evaluates whether models can infer latent market structures from event descriptions and historical financial observations. Models generate event-induced causal graphs representing inferred market structure and are evaluated through a reconstruction-based framework. Our evaluation of 6 open-weight and proprietary LLMs shows that adding their generated graphs generally reduces reconstruction quality compared with using historical observations alone, although a few metrics improve. These findings suggest that inferring informative, event-specific market
    
[^179]: 用于因果加权的平衡划分森林

    Balancing Partition Forest for Causal Weighting

    [https://arxiv.org/abs/2611.23524](https://arxiv.org/abs/2611.23524)

    该论文提出平衡划分树与平衡划分森林（BPT/BPF），通过在协变量—处理空间上以 Bregman 散度增益驱动分裂的矩形划分来估计理想平衡权重，并从理论上证明其总体残差可界定逆概率加权和双重稳健估计量的偏差。

    

    平衡权重在观察性因果推断中被用于构造一个使处理与观测到的预处理协变量相互独立的总体。理想权重是协变量与处理各自的边际分布乘积与其联合分布之间的比值，这一形式对二元处理和连续处理均适用。我们提出了平衡划分树（BPT）和平衡划分森林（BPF），它们在学习得到的协变量—处理空间的矩形划分上估计该比值。每一次分裂都最大化由联合计数与边际计数计算出的 Bregman 散度的增益。两种散度选择分别产生最大化划分所捕获互信息的准则，以及降低权重平方误差的准则。我们证明，任一准则的总体残差都为使用总体划分权重的逆概率加权（IPW）和双重稳健响应估计量的偏差提供了上界。在 17 个合成（数据集与真实数据集上的实验……原文摘要于此处截断）

    arXiv:2611.23524v1 Announce Type: cross  Abstract: Balancing weights are used in observational causal inference to construct a population in which the treatment is independent of observed pretreatment covariates. The ideal weight is the ratio between the product of the covariate and treatment marginals and their joint distribution, for binary and continuous treatments alike. We propose Balancing Partition Tree (BPT) and Balancing Partition Forest (BPF), which estimate this ratio on a learned rectangular partition of covariate--treatment space. Each split maximizes the gain in a Bregman divergence computed from joint and marginal counts. Two choices of divergence yield criteria that maximize the mutual information captured by the partition or reduce the squared error of the weights. We show that the population residual of either criterion bounds the bias of inverse-probability-weighted and doubly robust response estimators that use the population partition weights. Across 17 synthetic a
    
[^180]: 从提审到结案：法院案件的Transformer模型

    From Arraignment to Resolution: Transformer Models of Court Cases

    [https://arxiv.org/abs/2611.23518](https://arxiv.org/abs/2611.23518)

    本文提出基于Transformer的时间序列模型，将法院案件建模为事件序列以预测刑事案件中下一诉讼程序的结果，从而提升司法决策的透明度与可监督性。

    

    刑事法律系统由许多复杂的决策点组成，例如追诉哪些罪名、如何确定保释金、认罪协议谈判以及量刑判决。鉴于此类决策的深远影响，提高该系统内部透明度的需求日益增长。虽然先前的机器学习方法主要聚焦于对单一决策（如累犯）进行建模，本文提出了一种基于Transformer的时间序列模型，用于预测案件中下一个诉讼程序的结果（例如接受指控、设定保释金、达成认罪协议等）。通过将案件建模为事件序列，我们能够进行更细致的分析，以支持对决策过程更好的监督和理解。我们使用基于美国某中等城市公开法庭记录构建的新数据集，对数千个案例进行了实证分析。我们拟合了将书记员笔记中的文本数据与结构化信息相结合的模型……

    arXiv:2611.23518v1 Announce Type: new  Abstract: The criminal legal system is composed of many complex decision points, such as which charges are pursued, how bond is determined, plea deal negotiations, and sentencing. Given the profound impact of such decisions, there is a growing need to increase the transparency within this system. While prior machine learning approaches have focused on modeling single decisions (e.g., recidivism), this paper proposes a transformer-based time series model to forecast the outcome of the next proceeding in a case (e.g., charges accepted, bond set, plea deal made, etc.). By modeling a case as an event sequence, we enable a more nuanced analysis to support better oversight and understanding of decision making. We conduct an empirical analysis of thousands of cases using a newly constructed dataset derived from publicly available court records of a mid-sized U.S. city. We fit models that combine textual data from clerk notes with structured information a
    
[^181]: 基于RKHS微分同胚流的最优控制用于分类与生成建模

    Optimal Control of RKHS Diffeomorphic Flows for Classification and Generative Modeling

    [https://arxiv.org/abs/2611.23497](https://arxiv.org/abs/2611.23497)

    该论文提出了一个统一的最优控制框架，用RKHS参数化的微分同胚流同时解决分类与密度估计任务，其精确且确定性的散度使得二阶优化器切实可行，同时避免了现有连续归一化流的随机迹估计和势流限制。

    

    我们开发了一个统一的最优控制框架，利用速度场位于再生核希尔伯特空间（RKHS）中的单一微分同胚流，同时解决监督分类和基于似然的密度估计问题；这两个任务共享所有结构，仅目标函数不同。变量投影方法消除了终端模型（分类器或高斯混合模型（GMM）参考），从而得到一个降阶控制问题，其梯度来自解析的离散伴随方程。RKHS参数化使得流的散度精确且确定，其代价仅是一次核矩阵-向量乘积，避免了现有连续归一化流中的随机迹估计器和势流限制。由于目标函数和梯度都是确定性的，无矩阵的降阶高斯-牛顿（GN）及相关二阶求解器对这两个任务都切实可行；我们将其与一系列一阶和二阶优化器进行了基准测试。

    arXiv:2611.23497v1 Announce Type: cross  Abstract: We develop a unified optimal-control framework that solves supervised classification and likelihood-based density estimation with a single diffeomorphic flow whose velocity lives in a reproducing kernel Hilbert space (RKHS); the two tasks share all structure except the objective. Variable projection eliminates the terminal model (a classifier, or a Gaussian-mixture (GMM) reference), leaving a reduced control problem whose gradient comes from an analytic discrete adjoint. The RKHS parameterization makes the flow's divergence exact and deterministic at the cost of one kernel matrix-vector product, avoiding the stochastic trace estimators and potential-flow restrictions of existing continuous normalizing flows. Because the objective and gradient are deterministic, matrix-free reduced Gauss-Newton (GN) and related second-order solvers are practical for both tasks, which we benchmark against a range of first- and second-order optimizers; th
    
[^182]: 面向天文学的JEPA：多巡天光谱对齐使恒星参数落在主成分的前几个主轴上

    JEPA for astronomy: multi-survey spectral alignment puts stellar parameters on the leading principal axes

    [https://arxiv.org/abs/2611.23492](https://arxiv.org/abs/2611.23492)

    该论文提出将JEPA（联合嵌入预测架构）而非对比学习用于多巡天恒星光谱对齐，使恒星物理参数（如有效温度、表面重力、[Fe/H]和[alpha/Fe]）恰好落在嵌入空间的前几个主成分轴上，从而仅通过线性读出即可恢复Kiel图等恒星物理结构。

    

    来自不同巡天的光谱携带不同的系统误差，且彼此之间未经相互校准。因此，面向恒星光谱的多巡天基础模型是通过在嵌入空间中对齐同一恒星在不同巡天中的观测来构建的。天文学领域的大多数方法遵循为语言和视觉领域确立的方案，采用对比损失进行对齐，将每颗恒星与其他恒星区分开来。然而，这并不能保证恒星物理信息被编码成可以通过简单读出即可恢复的形式。在保持跨巡天架构不变、仅改变对齐损失的条件下，我们通过Gaia BP/RP (XP)光谱的主成分几何结构以及恒星参数的线性探针，来衡量不同目标函数各自编码了哪些信息。在一个恰当对齐的嵌入空间中，前两个主成分构成一个平面，能够恢复出Kiel图（有效温度–表面重力图）的类似物；第三个主成分追踪金属丰度[Fe/H]，而[alpha/Fe]在整个平面上连贯地变化。

    arXiv:2611.23492v1 Announce Type: cross  Abstract: Spectra from different surveys carry different systematics and are not calibrated against one another. Multi-survey foundation models for stellar spectra are then built by aligning embeddings of the same star observed across surveys. Most in astronomy follow the recipes established for language and vision, aligning with a contrastive loss that separates every star from the others. However, this does not ensure that the stellar physics is encoded in a form that a simple read-out can recover. Holding the cross-survey architecture fixed and varying only the alignment loss, we measure what an objective encodes from the principal-component geometry of the Gaia BP/RP (XP) spectra and from a linear probe of the stellar parameters. In a properly aligned embedding, the leading two principal components form a plane that recovers an analog of the Kiel diagram, the third component tracks [Fe/H], and [alpha/Fe] varies coherently across the plane, s
    
[^183]: SpecLook：通过推测观察位置实现快速可靠的多模态大语言模型智能体

    SpecLook: Speculating Where to Look for Fast and Reliable Agentic MLLMs

    [https://arxiv.org/abs/2611.23483](https://arxiv.org/abs/2611.23483)

    SpecLook 提出了一个无需训练的“推测-验证”三层框架，通过让小模型推测智能体下一步的放大观察——该观察可被验证、纠正并用于检验答案——从而显著加速多模态大语言模型智能体的推理并提升答案可靠性。

    

    具有智能体能力的多模态大语言模型（MLLMs）通过“用图像思考”的方式，将推理与放大工具调用交错进行。然而，这种顺序循环速度较慢，且其答案不附带可靠性评分。先前对 token、推理步骤或答案的推测仍然保留了这一循环，而答案推测还会接受自信却错误的答案。我们发现该循环主要增加的是一个放大步骤，借助这一步骤，小模型在大多数数据划分上能够领先于该智能体。我们的关键洞察是推测智能体的下一次观察——与起草的答案不同，观察是可以被验证、纠正并用于检验答案的。我们提出 SpecLook，这是一个无需训练的“推测-验证”框架，包含三个层次：小模型起草一个“瞥见”并据此重新作答，若答案不变且置信度高，则问题结束；否则，智能体的大模型在不使用工具的情况下验证并纠正该瞥见，并将两者的答案融合；只有仍缺乏证据的问题……（摘要原文在此处截断）

    arXiv:2611.23483v1 Announce Type: new  Abstract: Agentic multimodal large language models (MLLMs) that think with images interleave reasoning with zoom-in tool calls. However, this sequential loop is slow, and its answers carry no reliability score. Prior speculation of tokens, reasoning steps, or answers keeps the loop, and answer speculation also accepts confident errors. We find that the loop mostly adds one zoom-in step, with which a small model is ahead of the agent on most splits. Our key insight is to speculate the agent's next observation, which, unlike a drafted answer, can be verified, corrected, and used to test the answer. We propose SpecLook, a training-free speculate-and-verify framework with three levels. A small model drafts a glimpse and re-answers from it, and the question ends if the answer is unchanged and confident. Otherwise, the agent's large model verifies and corrects the glimpse without tools, and their answers are fused. Only questions that still lack evidenc
    
[^184]: GUARD：基于遗憾引导稳定递归自我改进

    GUARD: Stabilizing Recursive Self-Improvement with Regret Guidance

    [https://arxiv.org/abs/2611.23476](https://arxiv.org/abs/2611.23476)

    GUARD提出了一种无评估器的闭环框架，通过将任务分解为执行前自检并用方向性遗憾引导框架更新，稳定AI智能体的递归自我改进，避免局部退化、技能遗忘和记忆膨胀。

    

    自主AI智能体将冻结的基础模型与一个可编辑的框架相结合，该框架包含提示词、技能、工具规则和记忆。递归自我改进（RSI）无需更新权重即可根据经验调整这一框架，然而在缺乏外部评估器的情况下进行自我编辑往往会适得其反：反思和标量分数将编辑新破坏的内容与尚未解决的问题混为一谈，导致局部性能退化、技能遗忘和记忆膨胀。我们提出GUARD（带有自适应规则蒸馏的无评估器无监督环境设计[UED]），这是一个无需参考标准的持续框架演化闭环框架。我们的关键洞察是将每个任务分解为执行前的自检项，并通过方向性遗憾来引导框架更新。每一轮根据这些自检项对轨迹进行评分，在巩固已掌握规则的同时优先修复新破坏的行为，并通过无评论家的组相对适应度在并行候选编辑中进行选择。随后，Soft-HURT门控机制提升（原文摘要在此处截断）

    arXiv:2611.23476v1 Announce Type: new  Abstract: Autonomous AI agents pair a frozen base model with an editable harness of prompts, skills, tool rules, and memory. Recursive self-improvement (RSI) adapts this harness from experience without weight updates, yet self-editing without external graders often backfires: reflection and scalar scores conflate what an edit newly broke with what remains unsolved, causing local regressions, skill forgetting, and memory bloat. We introduce GUARD (Grader-free Unsupervised Environment Design [UED] with Adaptive Rule Distillation), a closed-loop framework for reference-free continual harness evolution. Our key insight is to decompose each task into pre-execution self-checks and guide harness updates via directional regret. Each round grades trajectories against these checks, pinning mastered rules while prioritizing newly broken behaviors, and selects among parallel candidate edits by critic-free group-relative fitness. A Soft-HURT gate then promotes
    
[^185]: AllotNet：一个面向人道主义援助分配中道德判断的可解释混合专家选择模型，并通过植入式合成人口加以验证

    AllotNet: an Interpretable Mixture-of-Experts Choice Model for Moral Judgement in Humanitarian Aid Allocation, Validated against a Planted Synthetic Population

    [https://arxiv.org/abs/2611.23474](https://arxiv.org/abs/2611.23474)

    AllotNet是一个可解释的混合专家条件Logit模型，通过预注册的六语言工具Allot捕获多文化道德判断以辅助人道主义援助分配，其规划器构建审核队列而非拒绝任何人，并通过植入式合成人口验证了模型有效性。

    

    2025年，官方发展援助下降了23.1%，而在资源不足以惠及所有人时决定谁优先获得帮助的权力正在转移到软件之中。一个已发表的领先机器学习瞄准系统在实际部署中排除了53%的符合资格的家庭。如果分配规则最终被写入代码，它就应当承载多种文化的判断，让机器负责执行、由人来评判。AllotNet是一个可解释的混合专家条件Logit模型，用于学习由Allot——一个预注册的六种语言调查工具——所引出的分配判断：模型在七项伦理标准之上设置了八个线性偏好专家，通过基于参与者情境的门控进行加权，并带有阻尼残差、一个用于家庭信息录入的编码器、一个结果预测头，以及一个构建审核队列、绝不直接拒绝任何人的预算受限规划器。人类判断模型在真实数据上缺乏标准答案，因此我们首先自行构造了一个答案：一个植入的合成人口。

    arXiv:2611.23474v1 Announce Type: cross  Abstract: Official development assistance fell by 23.1 per cent in 2025, and the decision of who is helped first when there is not enough for everyone is moving into software. A leading published machine-learning targeting system excluded 53 per cent of eligible households as deployed. If the rule of distribution ends up in code, it should carry the judgement of many cultures, and the machine should execute while a person judges. AllotNet is an interpretable mixture-of-experts conditional logit for the allocation judgements elicited by Allot, a pre-registered six-language instrument: eight linear taste experts over seven ethical criteria, weighted by a gate over participant context, with a damped residual, an encoder for household intake, an outcome head and a budget-constrained planner that builds a review queue and never rejects anyone. A model of human judgement has no answer key on real data, so we wrote one first: a planted synthetic popula
    
[^186]: FACT：面向推测解码的固定预算分配与跨步训练

    FACT: Fixed-Budget Allocation and Cross-Step Training for Speculative Decoding

    [https://arxiv.org/abs/2611.23463](https://arxiv.org/abs/2611.23463)

    提出FACT实验框架，系统研究草稿模型训练、候选token分配与执行条件对推测解码接受率和延迟的联合影响，并提出对早期步骤加重权重的跨步训练方法，使接受率相比传统分离训练提升15-24%。

    

    推测解码通过在单次目标模型前向传播中验证多个草稿token来加速语言模型生成。然而，更好的草稿预测和更长的被接受续写并不一定能带来更快的生成速度。为探究这一差距，我们提出了FACT（固定预算分配与跨步训练），这是一个用于研究草稿模型训练、候选token分配与执行条件如何共同影响接受率和延迟的实验框架。在贪婪解码设置下，我们使用轻量级的特征预测草稿模型配合冻结的Llama-3-8B-Instruct目标模型，首先考察了训练如何转化为被接受的token。通过对预测特征rollout进行反向传播，并对早期步骤赋予更大权重，相比均匀加权的分离训练可将接受率提升15-24%；然而，平均rollout代理交叉熵并不能可靠地按接受率对草稿模型进行排序。除预测质量之外，草稿候选的放置位置……

    arXiv:2611.23463v1 Announce Type: new  Abstract: Speculative decoding accelerates language-model generation by verifying multiple draft tokens in a single target-model pass. However, better draft predictions and longer accepted continuations do not necessarily produce faster generation. To examine this gap, we present FACT (Fixed-Budget Allocation and Cross-Step Training), an experimental framework for studying how drafter training, candidate allocation, and execution conditions jointly affect acceptance and latency. Using a lightweight feature-prediction drafter with a frozen Llama-3-8B-Instruct target under greedy decoding, we first examine how training translates into accepted tokens. Backpropagating through predicted-feature rollouts with greater weight on early steps improves acceptance by 15-24% over uniformly weighted detached training, yet average rollout proxy cross-entropy does not reliably rank drafters by acceptance. Beyond prediction quality, the placement of draft candida
    
[^187]: 紧凑视觉骨干网络中全局上下文与权重共享的适用位置

    Where Global Context and Weight Sharing Work in Compact Vision Backbones

    [https://arxiv.org/abs/2611.23446](https://arxiv.org/abs/2611.23446)

    该研究发现，在EfficientNet-B0骨干网络的P3、P4、P5阶段插入轻量级零门控残差自注意力适配器可将ImageNet-1k精度提升至74.02%，而单纯用BiFPN多尺度特征金字塔包装骨干则无法超越普通MBConv主干。

    

    紧凑型卷积骨干网络（如EfficientNet）通过局部的深度可分离卷积实现了出色的参数效率，但混合设计中仍有两个核心的结构性问题：全局自注意力在何处能超越纯卷积以提升识别性能，以及重复的模块能否在不损失表征能力的前提下共享权重？我们在ImageNet-1k（5万张图像的验证集）上通过受控的EfficientNet-B0实验对这两个问题进行了研究。首先，我们表明仅将分类骨干网络嵌入多尺度特征金字塔（BiFPN）中无法超越普通的MBConv主干（69.99%–71.81% vs. 普通B0的72.86%），直到配合阶段注意力后才有所提升（73.15%）。相比之下，保持MBConv主干不变，并在P3（28×28窗口式）、P4（14×14全局）和P5（7×7全局）阶段插入轻量级的零门控残差自注意力适配器，可独立达到74.02%的准确率，并优于逐位匹配配置的普通B0基线。

    arXiv:2611.23446v1 Announce Type: cross  Abstract: Compact convolutional backbones such as EfficientNet achieve strong parameter efficiency through local depthwise separable convolutions, yet two structural questions remain central to hybrid design: where does global self-attention improve recognition over pure convolution, and can repeated blocks share weights without losing representational capacity? We investigate both questions on ImageNet-1k (50,000-image validation) using controlled EfficientNet-B0 experiments. First, we show that wrapping a classification backbone in multi-scale feature pyramids (BiFPN) fails to outperform a plain MBConv trunk (69.99% - 71.81% vs. 72.86% plain B0) until paired with stage attention (73.15%). In contrast, keeping the MBConv trunk intact and inserting lightweight zero-gated residual self-attention adapters at stages P3 (28x28 windowed), P4 (14x14 global), and P5 (7x7 global) reaches 74.02% standalone and outperforms a bit-for-bit matched plain-B0 c
    
[^188]: 为什么亚4比特视觉CNN会崩溃以及混合精度恢复的极限在哪里

    Why Sub-4-Bit Vision CNNs Collapse and Where Mixed-Precision Recovery Stops

    [https://arxiv.org/abs/2611.23445](https://arxiv.org/abs/2611.23445)

    本研究系统剖析了视觉CNN在4比特以下量化时崩溃的机理，发现崩溃高度非线性且集中在后期逐点投影和深度可分离层，并提出了结构化的多步骤混合精度恢复瀑布策略来挽救精度。

    

    在资源受限环境中运行的深度视觉模型严重依赖低比特量化，然而将权重压缩到4比特以下往往会引发灾难性的精度崩溃。尽管混合精度恢复策略旨在通过选择性地保留高精度参数来挽救精度，但这些模型在哪里以及为何崩溃的基本机制仍鲜为人知。在本研究中，我们基于完整的50,000张图像的ImageNet-1k验证集，系统分析了标准视觉CNN（包括ResNet和EfficientNet骨干网络）量化崩溃的机理。首先，我们绘制了性能退化的确切进程，证明崩溃是高度非线性的，并集中于后期的逐点投影层和深度可分离层。其次，我们引入了一个结构化的多步骤恢复瀑布方法，评估选择性整数精度（INT4/INT8）和选择性浮点缩放的应用效果。

    arXiv:2611.23445v1 Announce Type: new  Abstract: Deep vision models operating in resource-constrained environments depend heavily on low-bit quantization, yet compressing weights below 4 bits frequently induces catastrophic accuracy collapse. While mixed-precision recovery strategies aim to rescue accuracy by selectively retaining high-precision parameters, the fundamental mechanics of where and why these models collapse remain poorly understood. In this study, we systematically analyze the anatomy of quantization collapse across standard Vision CNNs (including ResNet and EfficientNet backbones) evaluated on the full 50,000-image ImageNet-1k validation set. First, we map the exact progression of degradation, demonstrating that collapse is highly non-linear and concentrated in late-stage pointwise projections and depthwise layers. Second, we introduce a structured, multi-step recovery waterfall that evaluates selective integer precision (INT4/INT8) and selective floating-point scales. W
    
[^189]: 神经算子中经保形校准的自适应计算

    Conformally Calibrated Adaptive Computation in Neural Operators

    [https://arxiv.org/abs/2611.23440](https://arxiv.org/abs/2611.23440)

    该论文提出一种基于保形校准的自适应计算方法，在固定误差预算下为每个输入选择神经算子的傅里叶模态数或通道宽度并提供统计保证，同时发现模态选择虽缩减了谱支持范围，却因全网格变换主导计算成本而几乎无法降低推理延迟。

    

    arXiv:2611.23440v1 公告类型：新论文 摘要：在神经算子中选择更少的计算量并不必然缩短推理时间。我们利用保形校准来研究这一差距，在固定误差预算的约束下，针对每个输入自适应地选择傅里叶模态数或通道宽度。三角不等式包络可同时界定所有候选配置；直接评分规则可处理候选误差非单调的情形，并且在评分与输入无关时，能够在相同的边际保证下校准一个静态基线。在模型冻结与可交换性假设下，两种规则都能保证发出超出预算预测的概率有界，并在校准不可行时选择弃权。在Burgers方程和Darcy流动问题上、跨越三个训练种子的实验中，模态选择虽然减少了活跃谱支持范围，但并未带来可测量的延迟降低，原因在于全网格变换和通道混合主导了计算成本。缩小通道宽度确实降低了Darcy正问题的计算成本，但基于参考的选择必须先计算稠密预测。一个输入……（原文摘要在此处截断）

    arXiv:2611.23440v1 Announce Type: new  Abstract: Selecting less computation in a neural operator need not reduce inference time. We study this gap using conformal calibration to choose Fourier modes or channel width per input subject to a fixed error budget. A triangle-inequality envelope bounds all candidates simultaneously; a direct-score rule handles non-monotone candidate errors and, with an input-independent score, calibrates a static baseline under the same marginal guarantee. Under frozen-model and exchangeability assumptions, both rules bound the probability of issuing a prediction outside the budget and abstain when calibration is infeasible. On Burgers and Darcy flow across three training seeds, mode selection reduces active spectral support without a measurable latency reduction, because full-grid transforms and channel mixing dominate the cost. Narrowing channels does reduce Darcy forward cost, but reference-based selection must first compute the dense prediction. An input-
    
[^190]: 子空间方向何时对LoRA重要？少样本自适应中幅值原则的情境分析

    When Does Subspace Direction Matter for LoRA? Regime Analysis of the Magnitude Principle in Few-Shot Adaptation

    [https://arxiv.org/abs/2611.23428](https://arxiv.org/abs/2611.23428)

    本文证明LoRA初始子空间方向的重要性取决于情境：在少样本且初始化子空间能捕捉任务结构时方向对齐至关重要，并提出对齐系数ρ来预测增益、以及PivotLoRA进行任务对齐初始化。

    

    我们证明LoRA适配器的初始输入子空间方向的重要性是依赖于情境的，并通过理论与实验刻画了这种依赖关系。这一结论修正了近期认为谱初始化仅通过更新幅值起作用的观点。当有标签数据相对于表示维度较为稀缺，且用于初始化的激活子空间能够捕捉有监督任务结构时，方向的影响最为显著。在这一情境下，适配器的行空间对其初始点保持敏感，使得任务对齐的初始化尤为有用。基于尖峰协方差的扰动分析刻画了所估计子空间的有限样本稳定性。对LoRA训练第一步的分析解释了子空间与任务对齐为何重要，并由此引出一个对齐系数ρ，该系数能够紧密追踪各分类任务上的性能提升。我们通过PivotLoRA验证了这一解释，该方法从……初始化适配器（原文摘要在此处截断）。

    arXiv:2611.23428v1 Announce Type: new  Abstract: We show that the importance of a LoRA adapter's initial input-subspace direction is regime-dependent, and we characterize this dependence in theory and experiment. This refines the recent view that spectral initializers help through update magnitude alone. Direction matters most when labeled data are scarce relative to the representation dimension and the activation subspace used for initialization captures supervised task structure. In this regime, the adapter's row span remains sensitive to its starting point, making task-aligned initialization especially useful. A spiked-covariance perturbation analysis characterizes the finite-sample stability of the estimated subspace. A first-step analysis of LoRA training explains why the subspace's alignment with the task matters and motivates an alignment coefficient $\rho$ that closely tracks gains across classification tasks. We test this account with PivotLoRA, which initializes adapters from
    
[^191]: 稳定性并非可塑性的镜像：Transformer中陈述性与程序性学习的模块级视角

    Stability Does Not Mirror Plasticity: A Module-Level View of Declarative and Procedural Learning in Transformers

    [https://arxiv.org/abs/2611.23427](https://arxiv.org/abs/2611.23427)

    该论文揭示了Transformer中可塑性与稳定性在模块层面的双重分离——QK与VO投影在程序性学习中最具可塑性而MLP在陈述性记忆中最具可塑性，并且稳定性并不镜像可塑性：学习新程序时更新QK可保留过半陈述性记忆，而更新VO会丢失约五分之四。

    

    持续学习中的可塑性与稳定性通常在整体模型层面、以任务无关的方式进行研究和评估，然而已有工作在不同设置下得出了相互矛盾的结论。受认知科学中陈述性记忆与程序性记忆这一区分的启发，我们转而研究可塑性与稳定性在Transformer的不同模块和任务类型上如何变化。在一个测量干净、高度可控的合成环境中，我们展示了一种双重分离现象：查询-键（QK）投影和值-输出（VO）投影在程序性任务上最具可塑性，而多层感知机（MLP）在陈述性记忆任务上最具可塑性。然而，稳定性并非简单地镜像可塑性：在学习新程序时，更新QK投影可以保留超过一半已存储的陈述性记忆，而更新MLP所保留的要少得多，更新VO投影则会丢失大约五分之四。针对性的模块重置与恢复实验解释了这种不对称性：可塑性追踪的是哪个模块最适合……（原文摘要在此处截断）

    arXiv:2611.23427v1 Announce Type: new  Abstract: Plasticity and stability in continual learning are typically studied at the entire-model level and in a task-agnostic manner, yet prior work has reported divergent conclusions across settings. Inspired by cognitive distinctions between declarative and procedural memory, we instead study how plasticity and stability vary across transformer modules and task types. In a controlled synthetic environment with clean measurement, we show a double dissociation: query-key (QK) and value-output (VO) projections are the most plastic on procedural tasks, whereas MLPs are the most plastic on declarative memorization. Stability, however, does not simply mirror plasticity: when learning a new procedure, updating QK retains over half of the stored declarative memory, while updating MLPs retains far less and updating VO loses roughly four-fifths. Targeted module reset-and-recovery experiments explain this asymmetry: plasticity tracks which module is best
    
[^192]: 增量式Roseland：基于核化序贯Karhunen–Loève更新的分批地标扩散

    Incremental Roseland: Batch-wise Landmark Diffusion via Kernelized Sequential Karhunen--Lo\`eve Update

    [https://arxiv.org/abs/2611.23414](https://arxiv.org/abs/2611.23414)

    提出增量式Roseland（iRoseland），通过核化序贯Karhunen–Loève方案对流式增长的地标亲和矩阵的SVD进行增量更新，实现仅需单次数据遍历、有界内存、并可跟踪分布漂移的地标扩散谱嵌入。

    

    地标扩散从度归一化的数据-地标核矩阵的奇异值分解中计算扩散映射式的谱嵌入，其代价远低于完整扩散映射。然而，该方法假设所有数据是同时一次性可用的。我们提出增量式Roseland（iRoseland），将地标亲和度视为一个行数不断增长的矩阵，其奇异值分解可以通过核化的序贯Karhunen–Loève（SKL）方案进行更新。度校正通过围绕参考平均核行的一阶展开来处理，从而使得地标侧的特征基可以在每一步通过单次地标间的特征分解得以恢复。所提出的方法只需对数据进行一次遍历，并在有界内存下运行，其状态与已处理的数据点数量无关。引入遗忘因子使该方法能够跟踪漂移的分布，且样本量不变的嵌入可保持输出的一致性。

    arXiv:2611.23414v1 Announce Type: new  Abstract: Landmark diffusion (Roseland) computes a diffusion-map-style spectral embedding from the singular value decomposition of a degree-normalized data-to-landmark kernel matrix, at a cost far below that of full diffusion maps. It assumes, however, that all data are available at once. We introduce incremental Roseland (iRoseland), which casts the landmark affinity as a row-growing matrix whose SVD can be updated with a kernelized Sequential Karhunen--Lo\`eve (SKL) scheme. Degre correction is handled by a first order expansion about a reference mean kernel row and so that the landmark-side eigenbasis may be recovered from a single landmark-by-landmark eigendecomposition at each step. The presented method requires only a single pass over the data and runs in bounded-memory with state independent of the number of points processed. A forgetting factor lets the method track drifting distributions, and a sample-size-invariant embedding keeps outputs
    
[^193]: RFBench：一个基于仿真器的无源射频版图设计与理解多模态基准

    RFBench: A Simulator-Grounded Multimodal Benchmark for Passive RF Layout Design and Understanding

    [https://arxiv.org/abs/2611.23410](https://arxiv.org/abs/2611.23410)

    该论文提出了RFBench——首个基于ADS Momentum仿真器的无源射频版图设计与理解多模态基准，包含10,000个带有多模态对齐表示的双端口平面版图和七项任务，用以系统评估多模态语言模型连接射频设计意图、版图几何与电磁行为的能力。

    

    尽管多模态语言模型能够解释图表并生成结构化输出，但它们将无源射频（RF）设计意图、版图几何结构与电磁行为相互关联的能力在很大程度上尚未得到检验。为填补这一空白，我们提出了RFBench，一个面向无源射频版图设计与理解的、基于仿真器的数据集与基准。该数据集包含10,000个使用ADS Momentum仿真的双端口平面版图，每个版图由可编辑的几何图元、光栅图像、16x16占用网格、201个频点上的复数S参数、结构类别以及射频响应标签来表示。基于这些对齐的表示，我们定义了七项任务，涵盖规格条件下的版图生成、表示转换、响应预测、基于仿真器的验证、设计诊断与改进、结构理解以及响应分类。大量的基线实验……

    arXiv:2611.23410v1 Announce Type: new  Abstract: While multimodal language models can interpret diagrams and generate structured outputs, their capacity to connect passive radio-frequency (RF) design intent, layout geometry, and electromagnetic behavior remains largely untested. To address this gap, we introduce RFBench, a simulator-grounded dataset and benchmark for passive RF layout design and understanding. The dataset contains 10,000 two-port planar layouts simulated using ADS Momentum, with each layout represented by editable geometric primitives, a raster image, a 16x16 occupancy grid, complex S-parameters at 201 frequency points, a structural category, and an RF response label. Building on these aligned representations, we define seven tasks spanning specification-conditioned layout generation, representation translation, response prediction, simulator-based validation, design diagnosis and improvement, structural understanding, and response classification. Extensive baselines a
    
[^194]: 哪些Token、以何种顺序、多快？关于在线策略蒸馏目标的理论分析

    Which Tokens, in What Order, and How Fast? A Theoretical Analysis of On-Policy Distillation Objectives

    [https://arxiv.org/abs/2611.23397](https://arxiv.org/abs/2611.23397)

    本文首次从理论上剖析在线策略蒸馏目标的选择机制，揭示序列级OPD存在导致反向学习顺序的“未来一致性偏差”，而Token级目标可避免该偏差，并阐明了散度选择如何影响Token的优先级、学习顺序与收敛效率。

    

    在线策略蒸馏训练学生语言模型基于其自身的生成内容来匹配教师模型。在使用OPD时，必须选择是在序列级别还是Token级别应用它，以及最小化哪种散度。然而，尽管OPD在实践中被广泛采用，关于哪种OPD目标适用于特定设置的形式化理解仍然有限。为弥补这一空白，我们从理论上分析了OPD目标的选择如何决定哪些Token被优先处理、以什么顺序以及效率如何。我们的分析揭示，序列级OPD表现出一种“未来一致性偏差”，而短视的Token级目标可以避免这种偏差。这种偏差导致了一种反向学习顺序：序列级OPD首先在较靠后的Token位置上匹配教师模型，而在较早的位置上，它可能偏好那些在教师模型下概率较低、因而通常不合意的Token。至于散度的选择，我们证明……（摘要在此处截断）

    arXiv:2611.23397v1 Announce Type: cross  Abstract: On-policy distillation (OPD) trains a student language model to match a teacher model based on its own generations. When using OPD, one must choose whether to apply it at the sequence or token level and which divergence to minimize. However, despite the wide adoption of OPD in practice, the formal understanding of which OPD objective is appropriate for a given setting is limited. Toward addressing this gap, we theoretically analyze how the choice of OPD objective governs which tokens are prioritized, in what order, and how efficiently. Our analysis reveals that sequence-level OPD exhibits a future agreement bias, which myopic token-level objectives avoid. This bias induces a reverse learning order, in which sequence-level OPD first matches the teacher at later token positions, while at earlier positions it can favor tokens that are improbable under the teacher, and thus often undesirable. As for the choice of divergence, we prove that 
    
[^195]: QDA：基于智能体的闭环大语言模型量化设计

    QDA: Closed-Loop LLM Quantization Design with Agents

    [https://arxiv.org/abs/2611.23388](https://arxiv.org/abs/2611.23388)

    QDA提出了一个基于智能体的闭环系统，通过KL散度快速评估候选量化方案并自动化整合不同代码库的方法，为每个模型和精度自动设计最优的大语言模型训练后量化方案。

    

    训练后量化是以低成本部署大语言模型的标准方式。然而，最佳量化方案因模型和精度而异。因此，专家必须为每个新模型重新设计方案，这既耗时又往往不是最优的。幸运的是，大语言模型智能体可以通过在多轮构建和评估候选方案的过程中不断改进量化方案来接管这一设计工作，使得评估轮数越多，方案质量越高。然而，评估一个量化推理模型需要数十个GPU小时，而且要结合来自不同代码库的方法，就需要手动将每个方法移植到新架构上。我们提出了QDA，一个由智能体为每个模型和精度设计量化方案的系统。为了加快评估速度，QDA通过计算每个候选方案在智能体从未见过的推理轨迹上与原始模型输出的KL散度，在几分钟内完成评分，并对有前景的候选方案进行长文本生成检查，以捕捉重复等失败情况……

    arXiv:2611.23388v1 Announce Type: cross  Abstract: Post-training quantization is the standard way to serve large language models at low cost. However, the best quantization recipe differs across models and precisions. As a result, experts must redesign it for every new model, which is time-consuming and often sub-optimal. Fortunately, an LLM agent can take over this design by refining the recipe over rounds of building and evaluating candidates, so that more rounds lead to better recipes. However, evaluating one quantized reasoning model takes tens of GPU-hours, and combining methods from different codebases means porting each to the new architecture by hand. We present QDA, a system in which an agent designs the recipe for each model and precision. To make evaluation fast, QDA scores each candidate in minutes by its output KL divergence from the original model on reasoning traces that the agent never sees, and checks promising ones with long generations to catch failures such as repet
    
[^196]: 面向大鼠运动的神经回路架构先验

    Neural Circuit Architectural Priors for Rat Locomotion

    [https://arxiv.org/abs/2611.23364](https://arxiv.org/abs/2611.23364)

    该论文通过协同改进生物力学大鼠身体模型（Rat v2）、融合实验数据的神经回路架构先验（Rat NCAP）以及训练基础设施，构建了能够研究复杂身体与神经回路相互作用产生大鼠运动行为的具身模拟框架。

    

    动物通过神经回路、身体与环境之间的持续相互作用来产生行为。具身模拟通过在生物学逼真的环境中测试神经回路模型来研究这些相互作用。近期的研究已经模拟了多种多样的动物，而新兴的生物学基础神经架构研究已开始将实验数据中的回路连接方式与单神经元动力学整合进人工神经网络，从而引入更强的归纳偏置，并在机制层面更贴近生物学。我们如何开发出能够研究日益复杂的身体与神经回路的具身模拟？在本工作中，我们通过对模拟身体、神经架构和训练基础设施三方面的协同改进来应对这一挑战。我们提出了Rat v2，一个高效且灵活的生物力学大鼠身体模型，用于模拟大鼠行为。为了控制这一身体模型，我们开发了Rat NCAP，一种基于生物学

    arXiv:2611.23364v1 Announce Type: cross  Abstract: Animals generate behavior through continuous interactions between their neural circuits, bodies, and environment. Embodied simulations study these interactions by testing neural circuit models in biologically realistic settings. Recent efforts have simulated diverse animals, and emerging work on biologically grounded neural architectures has begun to integrate circuit connectivity and single-unit dynamics from experimental data into artificial neural networks, introducing stronger inductive biases and closer mechanistic alignment with biology. How can we develop embodied simulations that enable the study of increasingly complex bodies and neural circuits? In this work, we address the challenge through coordinated advances in the simulated body, neural architecture, and training infrastructure. We introduce Rat v2, an efficient and flexible biomechanical body for simulating rat behavior. To control this body, we develop Rat NCAP, a biol
    
[^197]: TRACE-MAS：诊断与修复多智能体系统中的协作故障

    TRACE-MAS: Diagnosing and Repairing Collaboration Failures in Multi-Agent Systems

    [https://arxiv.org/abs/2611.23362](https://arxiv.org/abs/2611.23362)

    TRACE-MAS是一个细粒度诊断与修复框架，通过从良性轨迹中学习正常协作模式、检测智能体交互层面的偏差、进行源头归因并实施最小干预，来识别和修复多智能体系统中源于交互的协作故障。

    

    基于大语言模型的多智能体系统（MAS）不仅依赖于单个智能体的行为，还依赖于连接各智能体的交互，包括消息交换和任务依赖关系。现有的多智能体系统安全防护机制主要执行智能体级别的异常检测，当故障源自智能体之间的交互时，这种方法就变得不足，且检测到的异常并不一定能揭示其真正的根源。为解决这一局限，我们提出了TRACE-MAS，一个用于诊断和修复多智能体系统故障的更细粒度框架。TRACE-MAS从良性轨迹中学习正常的协作执行模式，并检测智能体状态变化、消息传递和任务依赖演化中的偏差。它进一步通过源头感知的归因识别出导致故障的执行要素，并选择最小干预来恢复协作，同时保留任务关键能力。在五个多智能体系统基准上的实验表明，TRACE-MAS...

    arXiv:2611.23362v1 Announce Type: new  Abstract: LLM-based multi-agent systems (MAS) rely on not only individual agent behaviors but also the interactions that connect agents, including message exchanges and task dependencies. Existing MAS safeguards mainly perform agent-level anomaly detection, which becomes insufficient when failures originate from agent interactions and detected anomalies do not necessarily reveal their true sources. To address this limitation, we propose \textbf{TRACE-MAS}, a finer-grained framework for diagnosing and repairing MAS failures. TRACE-MAS learns normal collaborative execution patterns from benign trajectories and detects deviations in agent state changes, message transitions, and task dependency evolutions. It further identifies responsible execution elements through source-aware attribution and selects minimal interventions to restore collaboration while preserving task-critical capabilities. Experiments on five MAS benchmarks demonstrate that TRACE-M
    
[^198]: 计算代数中机器学习应用综述

    A Survey on the Use of Machine Learning within Computer Algebra

    [https://arxiv.org/abs/2611.23361](https://arxiv.org/abs/2611.23361)

    本综述系统梳理了机器学习在计算机代数领域的应用——既包括优化柱形代数分解、Gröbner基计算、符号积分等现有算法，也包括直接符号计算，并比较了从决策树到Transformer等多种机器学习方法。

    

    arXiv:2611.23361v1 公告类型：cross 摘要：本综述探讨了机器学习（ML）在计算机代数（CA）领域的应用，既包括对现有计算机代数算法的优化，也包括直接执行符号计算。传统的符号方法虽然在数学上是严谨的，但往往计算成本高昂，从而限制了它们在现实世界应用中的使用。最近的进展表明，数据驱动技术可以通过指导启发式决策、选择最优算法、预测代数对象的结构属性，甚至直接进行符号计算来解决这些局限性。我们采用系统性文献综述的方法，梳理了计算机代数应用中的相关工作，包括柱形代数分解、Gröbner基计算、符号积分等诸多领域。该综述比较了针对这些任务所采用的各种不同的机器学习方法，范围涵盖从决策树到Transformer模型。综述所揭示的问题包括……

    arXiv:2611.23361v1 Announce Type: cross  Abstract: This survey explores the use of Machine Learning (ML) in the field of Computer Algebra (CA), both to optimise existing CA algorithms, and to perform symbolic computation directly. Traditional symbolic methods, while mathematically rigorous, often are computationally expensive thus limiting their use in real-world applications. Recent advances have shown that data-driven techniques can address these limitations by guiding heuristic decisions, selecting optimal algorithms, predicting structural properties of algebraic objects, or even making direct symbolic computations.   We take a systematic literature review approach and uncover work in CA applications including cylindrical algebraic decomposition, Gr\"obner basis computation, symbolic integration, and many more. The survey compares the different ML approaches that have been employed for these tasks, ranging from decision trees to transformers. Issues uncovered by the survey include t
    
[^199]: 优先排序，而非决策：审计一个已部署的人在环简历排序系统的收获

    Prioritize, Don't Decide: What We Learned Auditing a Deployed Human-in-the-Loop CV Sorter

    [https://arxiv.org/abs/2611.23359](https://arxiv.org/abs/2611.23359)

    本文审计了一个部署在国家学术招聘平台上的人在环简历优先排序系统——委员会按其推荐顺序只需阅读一半简历即可覆盖 95% 的相关候选人；审计进一步发现该评测简历池仅凭文档长度即可排序，使精确率类指标失去参考意义，这一发现直接改变了产品设计。

    

    一份学术职位空缺可以吸引数百份申请，而第一轮筛选会占用评审委员会的大部分时间。我们描述了一个部署在国家学术职位空缺平台上、并与该平台职位空缺数据库集成的简历优先级排序器。它为一堆简历提出阅读顺序，而将所有决策留给委员会；该系统将冻结的多语言 Transformer 编码器与一个会话级线性模型相结合，该模型在每次标注后重新训练，并在会话结束时被丢弃。它是有效的：在一个包含 300 份合成简历的池子中，按该工具推荐的顺序阅读的委员会在读完一半简历后即可接触到 95% 的相关候选人，而按任意顺序阅读则需要几乎读完全部简历。本文关注的是我们审计上述声明时的发现，而非系统本身。三个发现改变了产品。其中一个发现是：该简历池仅凭文档长度即可完成排序，这使得在其上使用精确率类指标变得毫无参考意义；重新生成……（原文在此处截断）

    arXiv:2611.23359v1 Announce Type: new  Abstract: A single academic vacancy can draw hundreds of applications, and the first screening round consumes most of a committee's time. We describe a CV Priority Sorter deployed at a national platform for academic vacancies and integrated with its vacancy database. It proposes a reading order for a pile of CVs and leaves every decision to the committee, pairing a frozen multilingual transformer encoder with a per-session linear model that retrains after each label and is discarded when the session ends. It works: on a pool of 300 synthetic CVs, a committee reading in the tool's order reaches 95% of the relevant candidates after half the pile, where an arbitrary order needs almost all of it. This paper is about what we found when we audited that claim, rather than about the system itself. Three findings changed the product. The pool turned out to be sortable by document length alone, which makes precision-style metrics on it uninformative; regene
    
[^200]: 利用置换不变性启用选择性（PIES）保护实现基于近似存储器的二值Transformer

    Implementing Binary Transformers with Approximate Memories with Permutation Invariance Enabled Selective (PIES) Protection

    [https://arxiv.org/abs/2611.23358](https://arxiv.org/abs/2611.23358)

    本文提出利用Transformer对行列置换的不变性，设计了一种低开销的选择性保护方案（PIES），可显著提升基于近似存储器的二值Transformer在存储错误下的可靠性。

    

    量化是降低Transformer内存存储、计算复杂度和功耗的有效技术，而权重的二值化对于满足这些要求至关重要。近似存储器可以进一步降低存储器的功耗，但同时也会引入错误。目前针对二值Transformer的研究主要集中于任务性能，而模型在存储器发生错误时的可靠运行尚未得到充分研究。本文首先以基于Transformer的双向编码器表示（BERT）为案例，评估了二值Transformer的可靠性。研究发现，只有在高误码率（例如BER = 5%或10%）下才会出现较大的精度损失。随后，我们提出了一种新颖的置换不变性启用选择性（PIES）方案，以极低的开销保护二值权重。该方案利用Transformer对行和列置换的不变性，将关键权重进行分组（原文在此处截断）。

    arXiv:2611.23358v1 Announce Type: cross  Abstract: Quantization is an effective technique to reduce memory storage, computation complexity and power for Transformers, and the binarization of the weights is important for meeting these requirements. Approximate memory can further reduce the power for memory, but it also introduces errors. Current research for binary Transformers focuses on task performance, but the model dependable operation against errors on the memory has not been well studied. In this paper, we first evaluate the dependability of binary Transformers with the Bidirectional Encoder Representations from Transformers (BERT) as a case study. A large accuracy loss occurs only at high Bit Error Rates (e.g. BER = 5% or 10%). Then we propose a novel Permutation Invariance Enabled Selective (PIES) scheme to protect the binary weights at very low overhead. This scheme exploits the invariance of Transformers to permutations in the rows and columns to group the critical weights an
    
[^201]: SIGMA：基于说话人分离、症状结构化的临床访谈抑郁严重程度多模态估计

    SIGMA: speaker-isolated, symptom-structured multimodal estimation of depression severity from clinical interviews

    [https://arxiv.org/abs/2611.23348](https://arxiv.org/abs/2611.23348)

    SIGMA框架通过说话人分离、跨模态交叉注意力编码以及基于共享潜在严重程度特质的症状联合预测，在临床访谈中实现了抑郁严重程度的最先进多模态估计。

    

    临床访谈是抑郁症评估的核心手段，但耗时且依赖于评分者。自动化严重程度评估提供了可扩展的筛查能力，然而现有方法往往保留了访谈者的语音（混淆了参与者的线索）、在后期才融合各模态（丢失了细粒度的跨模态动态），或独立地对症状进行评分（忽略了症状间的相关性）。我们提出了SIGMA，这是一个具有说话人感知能力、症状结构化的半临床访谈抑郁严重程度估计框架。SIGMA使用说话人分离技术将输入限制为参与者的语音，采用交叉注意力机制使文本、音频和视频在编码过程中相互调制，并使用分级响应头通过共享的潜在严重程度特质和针对每个条目的有序阈值，联合预测八个问卷条目。在扩展痛苦分析访谈语料库上，SIGMA取得了最先进的性能（CCC 0.762，RMSE 4.13，MAE 3.30）

    arXiv:2611.23348v1 Announce Type: cross  Abstract: Clinical interviews are central to depression assessment but time-consuming and rater-dependent. Automated severity estimation offers scalable screening, yet existing methods often retain interviewer speech (confounding participant cues), fuse modalities late (losing fine-grained cross-modal dynamics), or score symptoms independently (ignoring inter-symptom correlations). We present SIGMA, a speaker-aware, symptom-structured framework for depression-severity estimation from semi-clinical interviews. SIGMA uses speaker diarization to restrict input to participant speech, cross-attention so that text, audio and video condition one another during encoding, and a graded-response head that predicts the eight questionnaire items jointly through a shared latent severity trait with item-specific ordered thresholds. On the Extended Distress Analysis Interview Corpus, SIGMA achieves state-of-the-art performance (CCC 0.762, RMSE 4.13, MAE 3.30 ov
    
[^202]: BiMa-ECGSR：利用双向状态空间模型从缩减导联集重构12导联心电图

    BiMa-ECGSR: reconstructing the 12-lead ECG from reduced lead sets with bidirectional state-space models

    [https://arxiv.org/abs/2611.23345](https://arxiv.org/abs/2611.23345)

    BiMa-ECGSR提出了一种基于双向Mamba状态空间模型的编码器-解码器架构，能够从单导联或少量导联高精度地重构完整的12导联心电图，并在多个大规模内外部数据集上显著优于传统线性重构方法。

    

    可穿戴设备和监测仪仅能记录少量心电图（ECG）导联，而大多数诊断工具都假设使用12导联。BiMa-ECGSR通过双向Mamba状态空间编码器-解码器，以回归（BiMa-SR）或扩散（DiBiMa-SR）两种方式从缩减的导联集重构12导联心电图。模型在PTB-XL数据集上针对六种输入导联组合（I；II；I+II；I+II+V1；I+II+V2；I+II+V1+V4）进行训练，推理时仅使用实际测量的导联，并在PTB-XL测试集（2,198条记录）以及两个外部队列（Georgia，10,275条记录；Chapman-Shaoxing/Ningbo，43,559条记录）上进行了评估。仅使用II导联时，在PTB-XL上11个预测导联的合并中位皮尔逊相关系数（PCC）达到0.914，而最小二乘线性滤波器仅为0.784（I导联：0.906对0.789）。由于肢体导联可由I和II导联精确推导，多导联输入的比较在预测的胸前导联上进行：I+II组合为0.939对0.849，I+II+V1+V4组合为0.979对0.962。BiMa-SR超越了线性方法。

    arXiv:2611.23345v1 Announce Type: cross  Abstract: Wearables and monitors record few electrocardiogram (ECG) leads; most diagnostic tools assume 12 leads. BiMa-ECGSR reconstructs the 12-lead ECG from reduced lead sets with a bidirectional Mamba state-space encoder--decoder, as regression (BiMa-SR) or diffusion (DiBiMa-SR). Models were trained on PTB-XL for six input sets (I; II; I+II; I+II+V1; I+II+V2; I+II+V1+V4), see only the measured leads at inference, and were tested on the PTB-XL test fold (2,198 records) and two external cohorts (Georgia, 10,275 records; Chapman-Shaoxing/Ningbo, 43,559). From lead II alone, the pooled median Pearson correlation (PCC) over the 11 predicted leads was 0.914 on PTB-XL, against 0.784 for a least-squares linear filter (lead I: 0.906 against 0.789). As limb leads are exactly derivable from I and II, multi-lead inputs were compared on predicted chest leads: 0.939 against 0.849 for I+II and 0.979 against 0.962 for I+II+V1+V4. BiMa-SR exceeded the linear 
    
[^203]: 基于领域知识与图结构学习的置信度材料数据质量评估

    Confidence-Based Materials Data Quality Assessment via Domain Knowledge and Graph-Structured Learning

    [https://arxiv.org/abs/2611.23342](https://arxiv.org/abs/2611.23342)

    该论文提出了一种融合领域知识与图结构学习的置信度材料数据质量评估框架，将数据质量建模为连续置信度估计而非二元异常识别，实现了样本级评估和基于历史参考的批次比较。

    

    材料机器学习的快速发展依赖于大规模、异构的数据集，然而来自不同实验、模拟和数据库整合的不完善数据往往会损害模型的可靠性。现实中的材料数据库可能包含测量误差、失败的计算、缺失或重复的记录，以及隐藏的物理化学约束违反。这些问题难以识别，因为受损的样本可能在统计上仍然合理，而不是表现为明显的异常值。在本文中，我们提出了一种基于置信度的材料数据质量评估框架，该框架将领域知识与图结构学习相结合。数据质量被表述为一个连续的置信度估计问题，而非二元的异常识别任务，从而实现样本级评估和基于历史参考的批次比较。九维特征质量先验和重复（注：原文摘要在此处截断，内容不完整）

    arXiv:2611.23342v1 Announce Type: cross  Abstract: The rapid development of materials machine learning relies on large-scale, heterogeneous datasets, yet model reliability is often compromised by imperfect data from diverse experiments, simulations, and database integration. Real-world materials databases may contain measurement errors, failed calculations, missing or duplicate records, and hidden violations of physicochemical constraints. Such problems are difficult to identify because corrupted samples may remain statistically plausible rather than appearing as explicit outliers. In this paper, we propose a confidence-based materials data quality assessment framework that integrates domain knowledge with graph-structured learning. Data quality is formulated as a continuous confidence estimation problem rather than a binary anomaly identification task, enabling sample-level assessment and historical-reference-based batch comparison. A nine-dimensional feature-quality prior and duplica
    
[^204]: 先优化“穗轴”，再优化“内核”：优化生成式推荐推理中的束搜索

    Optimize the Cob, Then the Kernel: Optimizing Beam Search for Generative Recommendation Inference

    [https://arxiv.org/abs/2611.23339](https://arxiv.org/abs/2611.23339)

    该论文提出利用Tree Attention将束搜索的各条束沿序列维度拼接以消除冗余KV缓存访问，并结合静态KV缓存分配、图编译、算子融合与词表约束解码等系统优化，在不依赖自定义内核的前提下大幅提升了生成式推荐推理中束搜索的效率。

    

    生成式推荐（GR）推理与传统的LLM服务存在显著差异，其特点是输入序列长、束搜索宽度大，而输出仅有少量固定的token。现有的LLM服务引擎（如vLLM）将各条束作为批处理中的独立序列来处理，导致对共享提示前缀的KV缓存产生冗余访问。此外，vLLM将束搜索的每一步生成作为单独的引擎调用来执行，从而在解码步骤之间引入了主机侧的束处理开销，并在PagedAttention中进行大量的运行时KV块管理。利用Tree Attention，我们沿序列维度将各条束拼接起来，以消除冗余的KV访问，并进一步通过预生成的部分注意力掩码、静态KV缓存分配、循环展开的生成调度、TorchAir图编译、自动算子融合以及词表约束解码来优化执行。在不使用自定义内核的情况下，我们的方法（原文摘要在此处截断）

    arXiv:2611.23339v1 Announce Type: cross  Abstract: Generative recommendation (GR) inference differs substantially from conventional LLM serving, featuring long input sequences, wide beam search, and only a few fixed output tokens. Existing LLM serving engines such as vLLM process beams as batched independent sequences, causing redundant KV cache accesses over the shared prompt prefix. Furthermore, vLLM executes each beam search generation step as a separate engine invocation, incurring host-side beam processing overhead between decoding steps and substantial runtime KV block management in PagedAttention. Using Tree Attention, we concatenate beams along the sequence dimension to eliminate redundant KV accesses, and further optimize execution through pre-generated partial-attention masks, static KV cache allocation, loop-unrolled generation dispatch, TorchAir graph compilation, automatic operator fusion, and vocabulary-constrained decoding. Without the use of custom kernels, our approach
    
[^205]: AI模型选择中质量-资源权衡的微分理论：多资源前沿及其极小极大价格

    A Differential Theory of Quality-Resource Trade-offs for AI Model Selection: The Multiresource Frontier and Its Minimax Price

    [https://arxiv.org/abs/2611.23324](https://arxiv.org/abs/2611.23324)

    该论文提出一种微分理论，将AI模型选择的质量-资源前沿表示为与价格无关的平面曲线，通过有效资源通道数与剩余对数误差深度之间的内在关系刻画了多资源权衡的几何结构与极小极大价格。

    

    在不同资源预算下比较AI模型会产生将误差与延迟、参数数量、训练计算量和能耗联系起来的质量-资源前沿。偏移幂律可以概括这类前沿，但其拟合参数既无法揭示与价格无关的几何结构，也无法揭示权衡如何在多个资源通道间延伸。本文通过发展一种使该结构显式化的微分理论来填补这两个空白。每个前沿都容许一种与价格无关的平面曲线表示，将有效活跃资源通道数 $N_{\mathrm{eff}} = s^2/\kappa$（斜率平方与曲率之比）与剩余的对数误差深度 $\Delta S = \ln(\varepsilon/\varepsilon_{\infty})$ 相关联。这种表示将内在形状与资源轴的尺度和原点分离开来。完全单调性与可除性导出了微分不等式和前沿界。当N个资源中的每个……（摘要在此处截断）

    arXiv:2611.23324v1 Announce Type: cross  Abstract: Comparing AI models across resource budgets yields quality-resource frontiers relating error to latency, parameter count, training compute, and energy. Offset power laws summarize such frontiers, but their fitted parameters reveal neither the price-independent geometry nor the manner in which trade-offs extend across several resource channels. This paper addresses both gaps by developing a differential theory in which this structure is made explicit. Each frontier admits a price-independent plane-curve representation relating the effective number of active resource channels, $N_{\mathrm{eff}} = s^2/\kappa$ (squared slope over curvature), to the remaining log-error depth, $\Delta S = \ln(\varepsilon/\varepsilon_{\infty})$. This representation separates intrinsic shape from the scale and origin of the resource axis. Complete monotonicity and divisibility yield differential inequalities and frontier bounds. When each of $N$ resources cont
    
[^206]: HydroMind：面向自主水文研究的可检验智能体

    HydroMind: A Checkable Agent for Autonomous Hydrological Research

    [https://arxiv.org/abs/2611.23323](https://arxiv.org/abs/2611.23323)

    该论文开发了面向通用智能体的水文建模系统HydroMind及其建模工具和研究流程，首次检验了当自主智能体自行决定水文模型评估方案时，其结论能否被自身数据、代码和结果支持并供读者核查。

    

    通用编程智能体已经为水文和地球系统模型编写过代码。在这些研究中，研究者事先决定了模型将如何被评估，例如测试时段、评分函数以及模型需要达到的标准。尚无研究检验过当智能体自行做出这些决策时会发生什么：它的结论中有多少能被其自身的数据、代码和结果所支持，以及读者能否对这些结论进行核查。这两点都可能取决于智能体的运行框架，即它可以调用的工具以及它所遵循的书面指令。我们比较了三种运行框架：智能体自带的框架；该框架加上HydroMind的建模工具——HydroMind是我们为通用智能体开发的水文建模系统，它通过固定命令运行每个模型，并存储模型的配置设置、训练好的模型和预测结果；以及再加上HydroMind的研究流程，在该流程下，智能体在运行之前先撰写研究设计（原文摘要至此截断）。

    arXiv:2611.23323v1 Announce Type: cross  Abstract: General-purpose coding agents have written code for hydrological and Earth-system models. In these studies the researchers decided in advance how the model would be evaluated, for example the test period, the scoring function, and the standard the model had to meet. No study has tested what happens when the agent makes these decisions itself: how many of its claims are supported by its own data, code, and results, and whether readers can check them. Both may depend on the agent's harness, that is, the tools it can call and the written instructions it follows. We compared three harnesses: the agent's own harness; this harness plus the modeling tools of HydroMind, a hydrological modeling system we developed for general-purpose agents, which run every model through fixed commands and store its settings, trained model, and predictions; and both plus HydroMind's research procedure, under which the agent writes a research design before runni
    
[^207]: 深度学习的误差分析

    Error Analysis of Deep Learning

    [https://arxiv.org/abs/2611.23321](https://arxiv.org/abs/2611.23321)

    本文以误差分解（逼近、统计与优化误差）为统一视角系统综述深度学习理论，梳理从单个误差分量分析到综合框架的发展脉络，并结合多个应用领域的实例阐明如何获得已训练网络的理论保证。

    

    本文从误差分解的视角审视深度学习理论，系统地探讨逼近误差、统计误差和优化误差三类问题。我们追溯了误差分析的发展历程——从对单个误差分量的上界估计，发展为同时涵盖全部三类误差的统一框架。以非参数回归和Deep Ritz方法作为两个贯穿全文的示例，我们展示了这些分析如何应用于本质上不同的损失结构。随后，我们综述了科学机器学习、生成式学习、强化学习、表示学习和大语言模型领域中的代表性误差分析，通过针对具体问题的论证，展示了学习误差如何影响解、分布和预测。贯穿全文，我们阐明了哪些误差分量得到了控制、它们如何被组合，以及为获得已训练网络的性能保证需要哪些假设。

    arXiv:2611.23321v1 Announce Type: cross  Abstract: This review examines deep learning theory through the lens of error decomposition, systematically addressing approximation, statistical, and optimization errors. We trace the development of error analyses from bounds on individual error components to frameworks that account for all three. Using nonparametric regression and the Deep Ritz Method as dual running examples, we demonstrate how these analyses apply across fundamentally different loss structures. We then review representative analyses in scientific machine learning, generative learning, reinforcement learning, representation learning, and large language models, showing how learning errors affect solutions, distributions, and predictions through problem-specific arguments. Throughout, we clarify which error components are controlled, how they are combined, and what assumptions are needed to obtain guarantees for trained networks.
    
[^208]: AI时代衡量真正重要的东西：研究软件指标全景

    Measuring What Matters in the Age of AI: The Research Software Metrics Landscape

    [https://arxiv.org/abs/2611.23253](https://arxiv.org/abs/2611.23253)

    该论文基于49篇文献系统梳理了研究软件度量指标的全景，将其归纳为八大类并从数据来源、效度、自动化潜力和滥用风险等维度进行分析，同时指出AI辅助开发正使现有指标体系的假设面临失效。

    

    研究软件如今已成为科学研究的核心工具，但学术界用于评判其质量、可持续性、复用性和影响力的方法仍散落在软件工程、开放科学、FAIR软件、引用和研究评估等多个领域的文献中。AI辅助开发加剧了这一问题：代码生成速度超出了人工审查的能力，来源与归属更难追踪，学术认可体系无法跟上步伐，而现有指标所继承的关于研究软件开发方式的假设可能已不再适用。为帮助应对这些挑战，我们基于49篇实质性参考文献，对研究软件指标进行了聚焦的全景综述，将反复出现的指标归纳为八大类，并从数据来源、所声称的构念、效度证据、自动化潜力以及被滥用的可能性等方面加以描述。许多被广泛引用的研究软件指标（原文摘要在此处截断）

    arXiv:2611.23253v1 Announce Type: cross  Abstract: Research software now serves as a primary instrument of science, but the methods the community uses to judge its quality, sustainability, reuse, and impact remain scattered across software engineering, open science, FAIR software, citation, and research-assessment literature. AI-assisted development sharpens the problem: code can be produced at rates that outpace human review capacity, provenance and attribution are harder to track, scholarly credit systems cannot keep pace, and existing metrics inherit assumptions about how research software gets developed that may no longer apply. To aid in navigating these challenges, we present a focused landscape synthesis of research software metrics based on 49 substantive references, organizing the recurring metrics into eight families described by data source, claimed construct, evidence of validity, automation potential, and susceptibility to misuse. Many widely cited research software metric
    
[^209]: 是什么决定了自动化审计的上限：从通用代码到科学代码、结果与数据

    What Moves the Ceiling on Automated Audit: from General Code to Scientific Code, Results and Data

    [https://arxiv.org/abs/2611.23250](https://arxiv.org/abs/2611.23250)

    本文在真值基准上量化了“异厂商模型审计+确定性检查”这一自动化审计流程的能力上限：通用代码上召回率仅约 30% 且无饱和点，可通过阅读模型、规则与决策规则提升；而在科学结果核查中，数字溯源检查能识别全部 49 份不一致报告，模型审计则可额外发现 25 份内部一致伪造中的 23 份。

    

    arXiv:2611.23250v1 公告类型：cross 摘要：模型越来越多地被用来检查其他模型所写的内容。我们研究了一种已实际部署的安排：由来自不同供应商的审计模型阅读工作成果，并由确定性检查验证代码所能验证的部分，然后在没有任何模型能够提供的真值（ground truth）之下测量其极限。在带有隐藏测试的通用代码上，审计模型的八次阅读达到 30.0% 的召回率 [20.0, 40.7]，同时在正确代码上的误报率为 16.0% [10.1, 22.3]，收益递减且没有出现平台期。阅读模型（与其采样方式相混淆）、规则手册中的一个句子以及决策规则各自都会改变这一数字；而根据事后评级，大部分被遗漏的内容其实是规范从未明确界定的行为。在三项预注册的科学研究中，检查报告中的数字能否溯源到其来源的确定性检查，标记出了全部 49 份不一致的报告，却没有标记出 25 份内部一致的伪造报告中的任何一份——而模型审计员标记出了其中 23 份（事后评定中 21 份的发现有据可依）；在数据方面，一种由文档推导出的验证……（摘要在此处被截断）

    arXiv:2611.23250v1 Announce Type: cross  Abstract: Models increasingly check what other models write. We study one shipped arrangement, in which an auditor from a different vendor reads the work and deterministic checks verify what code can verify, and measure its limits against ground truth no model supplies. On general-purpose code with hidden tests, eight readings of the auditor reach 30.0% recall [20.0, 40.7] at 16.0% [10.1, 22.3] flags on correct code, with diminishing returns and no plateau. The reading model (confounded with its sampling), one rulebook sentence and the decision rule each move that figure, and much of what is missed is, on a post hoc rating, behaviour the specification never determined. In three preregistered science studies, checking that reported numbers trace to their sources flagged all 49 inconsistent reports and none of 25 consistent fabrications, of which the model auditor flagged 23 (21 with a sound finding, post hoc); on data, a documentation-derived val
    
[^210]: 利用机器学习技术探索帕金森病的诊断

    Exploring Parkinson's Diagnosis Using Machine Learning Techniques

    [https://arxiv.org/abs/2611.23248](https://arxiv.org/abs/2611.23248)

    该研究将多种机器学习算法应用于包含58名患者的帕金森病数据集，发现决策树和逻辑回归准确率最高（72-76%），而支持向量机在识别阳性病例方面灵敏度最优（高达97.56%）。

    

    近年来，机器学习（ML）技术已成为准确且及时诊断帕金森病（PD）——一种无法治愈的神经退行性疾病——的有前景的工具。本研究探讨了机器学习算法与帕金森病诊断的结合，重点展示了其在疾病各方面的潜在应用。研究将支持向量机、决策树、多层感知器、k近邻、逻辑回归以及随机森林（分别使用和不使用粒子群优化）等机器学习方法应用于从利兹教学医院NHS信托机构获得的一个包含58人数据的帕金森病数据集。在准确率方面，决策树和逻辑回归表现最佳，准确率为72-76%，而多层感知器效果最差。所有模型在分类阳性帕金森病例方面表现出色，灵敏度得分高达97.56%，其中支持向量机分类器表现最优。然而……

    arXiv:2611.23248v1 Announce Type: new  Abstract: In recent years, machine learning (ML) techniques have emerged as promising tools for accurate and timely diagnosis of Parkinson's disease (PD) which is an incurable neurodegenerative disorder. This study explores the integration of ML algorithms and PD diagnosis, highlighting their potential applications in various aspects of the disease. ML methods such as support vector machines, decision trees, multilayer perceptron, k-nearest neighbours, logistic regression and random forest (with and without particle swarm optimisation) were applied on a PD dataset obtained from Leeds Teaching Hospitals NHS Trust containing data from 58 people. In terms of accuracy, the Decision Tree and Logistic Regression performed the best with scores of 72-76% while multilayer perceptron was least effective. All the models were remarkable in classifying positive PD cases with sensitivity scores going up to 97.56%, with SVM classifier at the pinnacle. However, a
    
[^211]: 面向不确定决策阈值的预算化适度损失设计

    Budgeted Proper Loss Design for Uncertain Decision Thresholds

    [https://arxiv.org/abs/2611.23247](https://arxiv.org/abs/2611.23247)

    本文在阈值分布属于有限指定族的设定下，推导出使代理遗憾到阈值平均决策遗憾的最坏情况乘数最小化的最优适度损失权重——即在规定下限处截断阈值密度的重标度上包络线，并证明了其锐利性、最优性与稳定性。

    

    二元概率估计可能被多个具有不同错误成本的决策者所使用。适当评分规则可以强调选定的概率阈值，但单一的平均阈值分布未必能提供良好的最坏情况遗憾保证。我们研究当阈值分布属于一个有限指定族时，如何选择适当损失。在具有逐点正下限且总质量固定的可积损失权重中，我们推导出了使从代理遗憾到阈值平均决策遗憾的锐利、无分布乘数最小化的权重。该解在规定的下限处对阈值密度的重标度上包络线进行截断，其归一化通过一维求根搜索得到。我们给出了锐利性、最优性以及密度误设下稳定性的证明，并指出了原子阈值定律下的一个局限性。精确的直方图实现和两个合成实验（验证了理论结果）。

    arXiv:2611.23247v1 Announce Type: new  Abstract: A binary probability estimate may be used by several decision makers with different error costs. Proper scoring rules can emphasize selected probability thresholds, but a single average threshold distribution need not give a good worst-case regret guarantee. We study the choice of a proper loss when the threshold distribution belongs to a finite specified family. Among integrable loss weights with a positive pointwise floor and a fixed total mass, we derive the weight that minimizes the sharp, distribution-free multiplier from surrogate regret to threshold-averaged decision regret. The solution clips a rescaled upper envelope of the threshold densities at the prescribed floor; its normalization is found by a one-dimensional root search. We give proofs of sharpness, optimality, and stability under density misspecification, as well as a limitation for atomic threshold laws. An exact histogram implementation and two synthetic experiments ac
    
[^212]: 面向学习使能系统认证可达性的切比雪夫多项式带状体抽象

    Chebyshev Polynomial Zonotope Abstractions for Certified Reachability of Learning-Enabled Systems

    [https://arxiv.org/abs/2611.23246](https://arxiv.org/abs/2611.23246)

    该论文证明泰勒展开因复平面上紧邻实轴的奇点而存在不可逾越的可达集宽度上限，并提出基于切比雪夫多项式带状体的抽象方法，在同等阶数下不受该上限约束且认证宽度随阶数呈几何级数增长，从而实现对学习使能系统更精确的认证可达性分析。

    

    当前几乎所有计算非线性动力学可达集的方法都采用截断泰勒展开并附加严格余项来进行传播。这种展开是逐点的：一旦其自变量是整个集合而非单个点，一个多项式就必须在整个集合上保持精确，而其覆盖范围仅限于自身的收敛圆盘之内。这一限制在现代控制器的有理项与Sigmoid门控函数上体现得最为严苛：尽管这些函数在实轴上看似光滑，但在复平面上它们会在紧邻实轴的奇点处发散，且门控越尖锐、奇点越靠近实轴，这些奇点便限制了泰勒展开所能覆盖的集合宽度，而这一限制在控制器做出切换决策的切换面上最为紧迫。我们证明了一个二分性结论：任何阶数的泰勒展开都无法认证宽度超过该上限的集合，而同等阶数的切比雪夫抽象，在同一多项式带状体流水线中锚定于整个集合，则不存在该上限，其可覆盖宽度随阶数呈几何级数增长。实测……（摘要在此处截断）

    arXiv:2611.23246v1 Announce Type: new  Abstract: Nearly all current methods for computing reachable sets of nonlinear dynamics propagate them with truncated Taylor expansions plus rigorous remainders. The expansion is pointwise: once its argument is a whole set rather than a point, one polynomial must stay accurate across that set, and it reaches only as far as its disc of convergence. The limit binds hardest on the rational terms and sigmoid gates of modern controllers: smooth as these look on the real line, in the complex plane they blow up at points just off the real axis, closer as a gate sharpens, and those points cap the set width a Taylor expansion can cover, most tightly on the switching surface where the controller decides. We prove a dichotomy: no Taylor degree certifies sets wider than that cap, while a Chebyshev abstraction of equal degree, anchored to the whole set in the same polynomial-zonotope pipeline, has no cap and gains width at a geometric rate in degree. Measured 
    
[^213]: 响应相关信息不足以判定神经网络的训练状态等价性

    Response-Relevant Information Is Not Sufficient for Neural Training-State Equivalence

    [https://arxiv.org/abs/2611.23245](https://arxiv.org/abs/2611.23245)

    尽管权重谱、表示几何和优化器几何等描述子携带响应相关信息并能预测训练扰动后的响应偏差，但它们仍不足以作为判定两个神经网络检查点训练状态等价的充分依据。

    

    响应相关信息并不足以作为两个神经网络处于相同训练状态的证据。我们对训练状态等价性给出了操作性定义：相对于某个干预族和时间范围，如果两个检查点在训练受到扰动并继续进行时表现出相似的响应，则认为它们是等价的。在单层Transformer学习模加法任务的全新运行实验中，相对于检查点之间的差异，对权重衰减脉冲的响应表现出高度可复现性（500步时 rho_state = 0.934），这确立了该研究目标是可以测量的。来自权重谱、表示几何和优化器几何的十五个描述子，在开发数据集上携带了超出权重范数和损失的运行内响应相关信息，并能跨随机种子块预测留出的响应偏差。然而，这些描述子通过未来响应匹配检查点时表面上16.1%的优势，无法与一个保留每个描述子……的条件安慰剂区分开来（原摘要在此处截断）。

    arXiv:2611.23245v1 Announce Type: cross  Abstract: Response-relevant information is not sufficient evidence that two neural networks are in the same training state. We define training-state equivalence operationally: two checkpoints are equivalent, relative to an intervention family and horizon, if they respond alike when training is perturbed and continues. On fresh runs of a one-layer transformer learning modular addition, response to a weight-decay pulse is highly reproducible relative to between-checkpoint differences (rho_state = 0.934 at 500 steps), establishing the target as measurable. Fifteen descriptors of weight spectra, representation geometry and optimiser geometry carry within-run response-relevant information beyond weight norm and loss on a development dataset, predicting held-out response deviation across seed blocks. But their apparent 16.1% advantage at matching checkpoints by future response does not separate from a conditional placebo preserving each descriptor's f
    
[^214]: Goemans不可分割流猜想：AI辅助反例发现的解剖分析

    Goemans's Unsplittable-Flow Conjecture: The Anatomy of AI-Assisted Counterexample Discovery

    [https://arxiv.org/abs/2611.23244](https://arxiv.org/abs/2611.23244)

    本文开发了一个前向反例构造流程，系统剖析了AI辅助发现的Goemans不可分割流猜想七顶点反例的结构原理，揭示了该反例为何成立、其所属的更广泛反例家族以及决定反例出现位置的关键结构特征。

    

    Goemans猜想指出，任何分数单源流都可以转化为不可分割流，且不增加总成本，同时任何弧的负载增加至多不超过最大需求。最近，Dmitry Rybin宣布了一个通过与ChatGPT交互发现的七顶点反例，推翻了这一长期存在的猜想。然而，这一交互并未揭示为什么这种特定的拓扑结构及其需求、分数流和成本选择能够协同发挥作用，该反例是否属于一个更广泛的家族，或者哪些结构特征决定了反例可能出现的位置。这也未能解释为什么AI辅助搜索在此前人类努力失败之处取得了成功。在本文中，我们开发了一个前向反例构造流程。该流程从一个不可能发生失败的树形图出发，并添加进入终点的捷径弧，由此产生的负载约束定义了一个有限的可行路由集合……

    arXiv:2611.23244v1 Announce Type: cross  Abstract: Goemans conjectured that any fractional single-source flow could be converted into an unsplittable flow without increasing total cost and while increasing any arc load by at most the largest demand. Recently, Dmitry Rybin announced a seven-vertex counterexample found through an interaction with ChatGPT, which disproved this long-standing conjecture. However, the interaction did not reveal why this particular topology and its demand, fractional-flow, and cost choices work together, whether the example belongs to a broader family, or which structural features determine where counterexamples can occur. It also left unclear why the AI-assisted search succeeded where earlier human efforts had not. In this paper, we develop a forward counterexample construction pipeline. It starts from an arborescence on which failure is impossible and adds terminal-entering shortcut arcs. The resulting load constraints define a finite set of admissible rout
    
[^215]: 脉冲神经网络的生物启发脉冲编码与优化

    Biologically-Inspired Spike Encoding and Optimization for Spiking Neural Networks

    [https://arxiv.org/abs/2611.23242](https://arxiv.org/abs/2611.23242)

    该论文将生物启发的脉冲编码方案（频率编码、延迟编码、增量调制、DoG 滤波）与自适应阈值替代梯度和 K-FAC 二阶优化相结合，显著提升了脉冲神经网络在容量受限条件下的准确率和脉冲稀疏性。

    

    脉冲神经网络在理论上相比传统深度网络具有能效优势，但要在视觉任务中实现这些优势，需要稀疏的脉冲表示，而朴素的频率编码方法无法提供。我们将生物启发的脉冲编码方案与两项训练改进相结合：Rouser 自适应阈值替代梯度以及 Kronecker 分解近似曲率（K-FAC）二阶优化。实验中考察的脉冲编码方案包括频率编码、延迟编码、DVS 风格的增量调制以及高斯差分（DoG）滤波。在 MNIST 和 Fashion-MNIST 上的实验表明，生物启发的编码器在容量受限的情形下带来最大的准确率提升，并可作为有效的参数效率手段。DoG 与增量调制编码器产生的脉冲序列比频率编码显著更稀疏。K-FAC 带来适度但一致的准确率提升。

    arXiv:2611.23242v1 Announce Type: new  Abstract: Spiking Neural Networks (SNNs) offer theoretical energy-efficiency advantages over conventional deep networks, but realizing these advantages on vision tasks requires sparse spike representations that naive rate-coding methods fail to provide. We combine biologically motivated spike-encoding schemes with two training improvements: Rouser adaptive-threshold surrogate gradients and Kronecker-Factored Approximate Curvature (K-FAC) second-order optimization. The spike encoding schemes experimented with are rate coding, latency coding, DVS-style delta modulation, and Difference-of-Gaussians (DoG) filtering. Experiments on MNIST and Fashion-MNIST demonstrate that biologically inspired encoders yield the largest accuracy gains in capacity-constrained regimes and serve as an effective parameter-efficiency lever. DoG and delta modulation encoders produce significantly sparser spike trains than rate coding. K-FAC yields modest but consistent accur
    
[^216]: GridForge：面向英国11千伏地下配电电缆的开放物理信息数字孪生与基准测试套件

    GridForge: An Open Physics-Informed Digital Twin and Benchmark Suite for UK 11 kV Underground Distribution Cables

    [https://arxiv.org/abs/2611.23241](https://arxiv.org/abs/2611.23241)

    本文提出了GridForge——一个开源的物理信息数字孪生与基准测试套件，为英国11千伏地下配电电缆领域提供了首个公开基准数据集、经验证的IEC标准热模型以及物理信息神经网络（PINN）代理模型，填补了该领域在故障诊断与预测模型评估协议上的空白。

    

    英国配电电缆监测领域呈现出数据丰富但框架匮乏的状态：各配电网络运营商（DNO）每天针对每个资产收集数兆字节的遥测数据，然而在对用于运营决策支持（尤其是维护规划与优先级排序）的行业数据密集型模型进行基准评估时，既没有公开的基准数据集，也没有共享的参考模型，更缺乏公认的评估协议。我们通过GridForge来填补这一空白——这是一个用于故障诊断与预测的开源物理信息数字孪生与基准测试套件，集成了四个组成部分：(i) 基于IEC 60287和IEC 60853标准并经过验证的稳态与暂态热模型；(ii) 一个经过校准的合成数据集，包含64条电缆（覆盖320电缆年），代表四种典型的英国11千伏热学类型和四种状态模式；(iii) 一个满足IEC 60287所定义代数平衡关系的物理信息神经网络（PINN）代理模型

    arXiv:2611.23241v1 Announce Type: new  Abstract: The UK distribution-cable monitoring sector is data-rich but framework-poor: each Distribution Network Operator (DNO) collects megabytes of telemetry per asset per day, but there is no public benchmark dataset, no shared reference model, and no commonly agreed evaluation protocol for benchmarking sectoral, data-intensive models for operational decision support; notably for maintenance planning and prioritisation. We address this gap with GridForge, an open-source physics-informed digital twin and benchmark suite for fault diagnosis and prediction, integrating four building blocks: (i) a verified steady state and transient thermal model based on IEC 60287 and IEC 60853; (ii) a calibrated synthetic dataset of 64 cables (covering 320 cable-years) representing four canonical UK 11 kV thermal archetypes and four condition modes; (iii) a Physics-Informed Neural Network (PINN) surrogate that satisfies the algebraic balance defined by IEC 60287 
    
[^217]: 基于神经网络的迟滞系统建模：从Prandtl与Preisach神经网络到神经Madelung与神经塑性连续模型

    Neural Network-Based Modeling of Hysteretic Systems: From Prandtl and Preisach Neural Networks to Neuro-Madelung and Neuroplastic Continuous Models

    [https://arxiv.org/abs/2611.23240](https://arxiv.org/abs/2611.23240)

    本综述系统梳理了从PNN、PreisachNN到GPNN、NMM、EPNN和DPCNN等一系列基于神经网络的迟滞系统建模方法的发展，涵盖其记忆表征机制、网络架构、训练算法及在非线性结构与材料系统中的应用。

    

    迟滞是一种复杂的非线性现象，其特征为记忆依赖性和路径依赖性，并且在许多物理系统中还表现出退化、非一致性、不对称性和速率依赖性。本综述考察了Prandtl神经网络、Preisach神经网络、广义Prandtl神经网络、神经Madelung模型、扩展Preisach神经网络以及动态塑性连续神经网络的发展历程与能力。综述特别关注了这些模型表征记忆的机制、网络架构、激活函数、训练算法、对退化和速率依赖性的处理方式，以及它们在非线性结构与材料系统中的应用。PNN通过停止算子将迟滞记忆直接引入神经激活过程，并展示了对结构进行精确非线性动力分析的能力。Preisach神经网络随后在此基础上进行了扩展。

    arXiv:2611.23240v1 Announce Type: new  Abstract: Hysteresis is a complex nonlinear phenomenon characterized by memory dependence, path dependence, and, in many physical systems, degradation, non-congruency, asymmetry, and rate dependence. This review examines the development and capabilities of Prandtl neural networks (PNNs), Preisach neural networks (PreisachNNs), generalized Prandtl neural networks (GPNNs), the neuro-Madelung model (NMM), extended Preisach neural networks (EPNNs), and dynamic plastic continuous neural networks (DPCNNs). Particular attention is given to their mechanisms for representing memory, network architectures, activation functions, training algorithms, treatment of deterioration and rate dependence, and applications to nonlinear structural and material systems. The PNN introduced hysteretic memory directly into neural activation through stop operators and demonstrated accurate nonlinear dynamic analysis of structures. Preisach-NNs subsequently extended this con
    
[^218]: 用于建模受损驾驶的多模态掩码扩散自编码器

    Multimodal Masked Diffusion Autoencoder for Modeling Impaired Driving

    [https://arxiv.org/abs/2611.23211](https://arxiv.org/abs/2611.23211)

    该论文提出多模态掩码扩散自编码器 ID-MAE，通过自监督方式重建被掩码的注视与轨迹数据来增强受损驾驶预测，并发布了包含 109 人、103 小时驾驶数据及十种受损诱发条件的大规模模拟驾驶数据集 DIVERT，显著提升了受损驾驶推断效果。

    

    人类认知状态（如分心、压力）、注视与行为会以微妙的方式与驾驶环境相互作用，AI 可以利用这些信息来辅助驾驶员，但大规模获取人类认知状态水平的真实标注成本高昂。我们提出了一种多模态掩码扩散自编码器 ID-MAE，它在受损驾驶预测中引入了自监督目标：除了将受损标签作为掩码 token 进行预测外，模型还重建被掩码的注视数据和车辆轨迹。为了探究不同模态和不同类型受损状态对模型推理的影响，我们引入了 DIVERT——一个驾驶模拟器数据集，包含 109 名受试者的 103 小时驾驶数据，涵盖认知、视觉和情绪三类状态下的十种受损诱发方式，并以注视、车辆状态和场景视频的形式进行记录。我们展示了 ID-MAE 在 DIVERT 数据集和现有数据集上均能显著提升受损驾驶推断的效果……

    arXiv:2611.23211v1 Announce Type: cross  Abstract: Human cognitive state (e.g., distraction, stress), gaze, and behavior interact with the driving environment in subtle ways that AI could use to assist drivers, but ground-truth labels for human cognitive state levels are costly to obtain at scale. We propose a multimodal masked-diffusion autoencoder, ID-MAE, that adds a self-supervised objective to impairment prediction: alongside predicting the impairment label as a masked token, it reconstructs masked gaze and trajectory. To probe the effects of different modalities and types of impairments on model inference, we introduce DIVERT, a driving-simulator dataset containing 103 hours of driving data (n = 109) with ten impairment inductions spanning cognitive, visual, and emotional states, recorded with gaze, vehicle state, and scene video. We demonstrate how ID-MAE affords significant improvements in impairment inference results on both the DIVERT dataset and existing datasets, and enable
    
[^219]: 面向多语言适配的MoE-LoRA方法：一项系统性实证研究

    MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study

    [https://arxiv.org/abs/2611.23158](https://arxiv.org/abs/2611.23158)

    本研究首次在统一的多语言评估协议下，用200种语言和9000多个评估设置系统比较八种LoRA方法，发现小规模搜索无法可靠预测方法在大规模下的相对表现，且LoRA虽能提升翻译数据拟合度但会持续损害阅读理解能力。

    

    路由式低秩适配器为提升大语言模型（LLM）在各类任务上的性能提供了一种参数高效的手段。因此，人们假定它们也可用于改善LLM对低资源语言的支持——目前这些模型对英语的服务远好于全球数千种其他语言。然而，这一假设从未在统一的多语言评估协议下得到实证验证。我们正是针对这一空白开展研究：在冻结的基础模型上，使用200种独特语言、超过9,000个评估设置，对八种低秩适配方法进行了比较。我们的结果表明，缩减规模的搜索能够识别出LoRA方法的可行配置，但无法可靠地预测其相对性能，因为排名最高的配置在大规模评估中从未表现最佳。LoRA能可靠地提升模型在翻译数据上的语言建模拟合度，却持续降低阅读理解能力，并在（原文摘要在此处截断）……

    arXiv:2611.23158v1 Announce Type: new  Abstract: Routed low-rank adapters offer a parameter-efficient means to improve the performance of large language models (LLMs) across tasks. Hence, it is assumed that they can also be used to improve the support of low-resource languages in LLMs, which currently serve English far better than the thousands of other languages spoken worldwide. However, this assumption has never been verified empirically under a shared multilingual evaluation protocol. We address exactly this gap by comparing eight low-rank adaptation (LoRA) methods on a frozen base model using 200 unique languages in more than 9{,}000 evaluation settings. Our results suggest that a reduced-scale search can identify viable configurations of LoRA methods but does not reliably predict their relative performance, as the top-ranking configuration is never best at scale. LoRA reliably improves language-model fit on translation data, consistently degrades reading comprehension, and benefi
    
[^220]: 面向多目标组合问题的帕累托感知智能“预测后优化”方法

    Pareto-Informed Smart "Predict, then Optimize" for Multi-Objective Combinatorial Problems

    [https://arxiv.org/abs/2611.23076](https://arxiv.org/abs/2611.23076)

    提出了一种帕累托感知的SPO+扩展方法，利用真实帕累托前沿构建与采样偏好向量对齐的目标决策并最小化遗憾，克服了现有决策聚焦学习方法在多目标组合优化中因简单标量化而无法捕捉丰富权衡关系的局限。

    

    许多现实世界的规划应用，从物流、路径规划到资源分配，都会产生需要平衡多个相互冲突目标的优化问题。虽然机器学习常被用于预测这些问题的未知参数或成本，但基于预测精度训练的标准模型可能导致次优的规划方案。决策聚焦学习通过将下游优化任务整合到训练目标中来解决这一问题。然而，现有的多目标问题DFL方法依赖于简单的标量化，无法捕捉帕累托前沿上丰富的权衡关系。为了填补这一空白，我们提出了一种帕累托感知的扩展方法，将单目标智能“预测后优化”方法扩展到多目标组合优化问题。我们的损失函数利用真实的帕累托前沿，构建与采样偏好向量相一致的目标决策，并最小化遗憾。

    arXiv:2611.23076v1 Announce Type: new  Abstract: Many real-world planning applications, from logistics, routing to resource allocation, give rise to optimization problems that need to balance multiple, conflicting objectives. While machine learning is often used to predict the unknown parameters or costs of these problems, standard models trained on prediction accuracy can lead to suboptimal plans. Decision-Focused Learning (DFL) addresses this by integrating the downstream optimization task into the training objective. However, existing DFL methods for multi-objective problems rely on simple scalarization, which fails to capture the rich trade-offs of the Pareto Frontier. To address this gap, we propose a Pareto-aware extension to single-objective Smart "Predict, then Optimize" (SPO+) for multi-objective combinatorial optimization problems. Our loss function leverages the true Pareto Frontier, forming a target decision that aligns with a sampled preference vector, and minimizes regret
    
[^221]: ConserveEdit：基于源图像锚定事务的漂移有界迭代图像编辑

    ConserveEdit: Source-Anchored Transactions for Drift-Bounded Iterative Image Editing

    [https://arxiv.org/abs/2611.22724](https://arxiv.org/abs/2611.22724)

    ConserveEdit提出了一种与编辑器无关的事务式控制器，通过从源图像编译的契约将授权区域与受保护内容分离，确保迭代图像编辑中授权区域外的像素保持位级精确不变，且漂移与迭代次数无关。

    

    指令驱动的图像编辑器能够在一次处理中产生令人信服的局部编辑，但在迭代细化循环中，每个被接受的输出都会成为下一个输入，导致请求区域之外的意外变化不断累积。我们提出ConserveEdit，这是一个与编辑器无关的控制器，它将每个细化步骤视为针对一份从源图像一次性编译所得契约的事务。该契约将允许改变的内容（每个目标的授权区域，并为阴影和反射保留有限的边缘晕圈）与必须保留的内容（具有明确容差的受保护内容）分离开来。来自冻结编辑器的候选结果通过源锚定掩码进行合成，只有当目标进展、受保护内容上的漂移和图像质量分别通过独立测试时才会被提交；否则该步骤将被回滚并修复。对于任何编辑器，这都能保证授权区域之外的每个像素保持位级精确一致，并将漂移限制在与迭代次数无关的范围内。

    arXiv:2611.22724v1 Announce Type: cross  Abstract: Instruction-driven image editors produce convincing local edits in one pass, but inside an iterative refinement loop every accepted output becomes the next input, so unintended changes outside the requested region accumulate. We present ConserveEdit, an editor-agnostic controller that treats each refinement step as a transaction against a contract compiled once from the source image. The contract separates what may change (the authorized region of each target, with a bounded halo for shadows and reflections) from what must survive (protected content with explicit tolerances). Candidates from a frozen editor are composed through a source-anchored mask and committed only if target progress, drift on protected content, and image quality pass as separate tests; otherwise the step is rolled back and repaired. For any editor, this keeps every pixel outside the authorized region bit-exact and bounds drift independently of the number of iterat
    
[^222]: Heave-Ho：去中心化的长时程多人形机器人协作

    Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration

    [https://arxiv.org/abs/2611.22275](https://arxiv.org/abs/2611.22275)

    该论文提出Heave-Ho去中心化框架，通过多智能体后训练实现物理耦合的人形机器人协调控制，结合视觉-语言规划器与基于语音的领导者-跟随者机制，在无需网络通信的情况下实现最多十台（含异构）人形机器人的长时程协作移动操作。

    

    协作有望将人形机器人的能力扩展到超越单个机器人所能达到的水平。然而，实现这一潜力需要协调的全身交互、连贯的长时程规划以及机器人之间的精确同步。我们提出了Heave-Ho，一个面向长时程协作式多人形机器人移动操作的去中心化框架。在低层，我们开发了一个多智能体后训练框架，将人形机器人基础追踪模型适配到人形机器人之间物理耦合的交互中，仅凭局部观测即可实现协调的去中心化控制。在高层，我们训练了一个视觉-语言规划器，并设计了一种基于语音的领导者-跟随者协调机制，从而在无需网络通信的情况下同步去中心化执行。我们的系统通过两台相同的人形机器人进行训练，可扩展至最多十台机器人的团队，并且无需重新训练即可泛化到异构团队。

    arXiv:2611.22275v1 Announce Type: cross  Abstract: Collaboration promises to extend humanoid capabilities beyond what individual robots can achieve. However, realizing this potential requires coordinated whole-body interactions, coherent long-horizon planning, and precise synchronization across robots. We introduce Heave-Ho, a decentralized framework for long-horizon collaborative multi-humanoid loco-manipulation. At the low level, we develop a multi-agent post-training framework that adapts a humanoid foundation tracking model to physically coupled interactions between humanoids, enabling coordinated decentralized control from local observations alone. At the high level, we train a vision-language planner and design a speech-based leader-follower coordination mechanism to synchronize decentralized execution without network communication. Trained with two identical humanoids, our system scales to teams of up to ten robots and generalizes to heterogeneous teams without retraining. To be
    
[^223]: 面向材料性质预测中扩散预训练的合成晶体数据生成

    Synthetic Crystal Data Generation for Diffusion Pretraining in Material Property Prediction

    [https://arxiv.org/abs/2611.22140](https://arxiv.org/abs/2611.22140)

    该论文提出一种结合扰动-弛豫、MatterGen生成式采样等三种互补策略的合成晶体数据生成流水线，为材料性质预测的扩散预训练提供大规模且结构多样的弛豫结构，突破了依赖昂贵DFT弛豫获取预训练数据的瓶颈。

    

    基于扩散的预训练已成为图神经网络中学习可迁移表示的有力策略，可用于材料性质预测，而无需为每个预训练结构提供诸如能量、力和应力等量子力学标签。然而，该方法仍存在一个关键依赖：它要求结构位于势能面的局部极小值附近，而这传统上需要通过对已知或物理合成的化合物进行昂贵的DFT弛豫来获得。这种依赖限制了可用预训练数据的规模和结构多样性，使得晶体结构空间的大量区域未被探索。在这项工作中，我们将基于扩散的预训练与合成结构生成相结合，以缓解这一瓶颈。我们提出了一个合成数据生成流水线，结合了三种互补策略：扰动-弛豫（perturb-and-relax）、通过MatterGen进行的生成式采样……

    arXiv:2611.22140v1 Announce Type: new  Abstract: Diffusion-based pretraining has emerged as a powerful strategy for learning transferable representations in graph neural networks for materials property prediction without requiring quantum-mechanical labels such as energies, forces, and stresses for every pretraining structure. However, the method retains a critical dependency: it requires structures residing near local minima of a potential energy surface, which have traditionally been obtained through costly DFT relaxations of known or physically synthesized compounds. This reliance constrains both the scale and the structural diversity of available pretraining data, leaving large regions of crystal structure space unexplored. In this work, we connect diffusion-based pretraining with synthetic structure generation to reduce this bottleneck. We introduce a synthetic data generation pipeline combining three complementary strategies: perturb-and-relax, generative sampling via MatterGen, 
    
[^224]: 零售预测中世界模型解释的审计：忠实性、准确性与价格响应

    Auditing World-Model Explanations in Retail Forecasting: Faithfulness, Accuracy and Price Response

    [https://arxiv.org/abs/2611.20811](https://arxiv.org/abs/2611.20811)

    研究发现，零售预测中的世界模型解释即使通过了八项审计检查（如构造性、稳定性、价格方向和与真实销量的一致性），其促销提升估计仍可能高出真实值1.2至1.4倍，价格响应被夸大近5至6.5倍，表明通过常规检查并不保证解释的忠实性与准确性。

    

    能够读取价格与促销计划的预测模型，将每个预测解释为两部分之和：一个基线（即不执行任何计划时的销量），加上该计划带来的提升。我们通过八项检查来检验此类解释，其中六项在任何结果出现之前就已固定了通过规则，结果发现通过这些检查并不意味着解释是正确的。在效应已知的人工生成门店数据中，一个世界模型通过了构造性、稳定性、价格方向以及与销量和LightGBM一致性等各项检查，但当商家将促销计划安排在高需求周时，其提升值是真实值的1.33至1.40倍；在该世界的八种变体中，这一比例为1.23至1.42倍。它仍然能选对促销的周次，但根据其提升值订购的库存与真实提升值相比偏差达0.44至0.48。它所宣称的对长期常规价格变动的响应是真实值-0.5的4.7至6.5倍，而只有通过实际测量才能发现这一点。在Dominick's数据集上，38个门店常规价格偏离其z（摘要在此处截断）

    arXiv:2611.20811v1 Announce Type: cross  Abstract: Forecasting models that read the price and promotion plan explain each forecast as a baseline, what would sell with nothing planned, plus the lift the plan adds. We test such explanations with eight checks, six with a pass rule fixed before any result, and find that passing them does not make an explanation right. In generated store data with known effects, a world model passes the checks of construction, stability, price direction and agreement with sales and with LightGBM, yet its lifts are 1.33 to 1.40 times the truth when merchants plan into high-demand weeks, and 1.23 to 1.42 times in eight variants of that world. It still picks the right weeks to promote, but stock ordered to its lift is off by 0.44 to 0.48 of the true lift. Its stated response to a lasting regular-price change is 4.7 to 6.5 times the true -0.5, and only a measured cause finds this. On Dominick's, 38 episodes in which a store's regular prices sat apart from its z
    
[^225]: 对谁而言可通过？人行道可达性感知的少样本个人校准

    Passable for Whom? Few-Shot Personal Calibration of Sidewalk Accessibility Perception

    [https://arxiv.org/abs/2611.19839](https://arxiv.org/abs/2611.19839)

    该论文提出 PassCal 框架，将共享的场景描述与个人解释分离，仅需从未见过的移动辅助工具用户的少量回答，即可个性化预测其对人行道场景的可达性判断。

    

    人行道是否可以通过取决于试图通过的人是谁：即使使用相同移动辅助工具的人，对同一张图像的判断也可能不一致，因此对每个场景或每种辅助工具只标注一个标签，对许多人来说都是错误的。我们提出了少样本个人可达性感知任务：根据一位从未见过的用户的K个回答，预测其对从未见过的行道场景的判断。PassCal 将场景的共享描述与个人对场景的解释分离开来。描述来自冻结视觉语言模型（VLM）的内部状态，以及由在 14.9k 条公开审计数据上训练的检测器所预测的障碍物类型、严重程度和上下文属性；而解释——即对该用户回答风格的校准和对各属性的加权——则通过其回答拟合获得。在包含 8,883 个判断的数据集中，135 位经过验证的移动辅助工具用户对相同的 50 个人行道场景进行评分，在用户和场景均被留出的设置下，仅需八个回答即可降低 Brier 评分。

    arXiv:2611.19839v1 Announce Type: cross  Abstract: Whether a sidewalk is passable depends on who is trying to pass: people who use the same mobility aid disagree about the same image, so one label per scene, or per aid, is wrong for many of them. We introduce few-shot personal accessibility perception: from K answers of a person never seen before, predict their judgment of sidewalk scenes never seen before. PassCal separates a shared description of the scene from the person's interpretation of it. The description comes from the internal state of a frozen vision-language model (VLM) and from barrier type, severity and context properties predicted by a detector trained on 14.9k public audits; the interpretation, a calibration of the person's response style and a weighting of the properties, is fit from their answers. On 8,883 judgments in which each of 135 verified mobility-aid users rated the same 50 sidewalk scenes, with people and scenes both held out, eight answers lower the Brier sc
    
[^226]: 模糊样本有帮助，但全量数据更胜一筹：重新审视数据集制图法在抽取式对抗问答中的应用

    Ambiguous Examples Help, But Full Data Wins: Revisiting Dataset Cartography for Extractive Adversarial QA

    [https://arxiv.org/abs/2611.19763](https://arxiv.org/abs/2611.19763)

    本文在更严格的抽取式对抗问答设置下重新审视数据集制图法的结论，发现仅用33%模糊样本子集训练虽有帮助，但仍不敌全量数据训练在对抗性评估中的表现。

    

    数据集制图法（Dataset Cartography）认为，训练动态可以揭示数据集中对学习具有不同作用的区域：易学样本有助于优化过程，难学样本往往暴露标注问题，而模糊样本对分布外泛化的贡献最大。原始工作的一个引人注目的结论是，仅在33%模糊样本上训练的模型可以在多项分布外（OOD）评估中超越全量数据训练的模型，其中包括在Adversarial SQuAD上评估的SQuAD的QNLI重构版本。我们在一个更严格的设置下重新审视这一结论：开放式抽取式问答，即模型必须预测答案片段，而不是对“问题-句子”对进行分类。针对每个编码器系列，我们计算干净来源SQuAD的训练动态，分别在全量、随机、按易学排序、按模糊排序和按难学排序的子集上进行训练，并在仅包含对抗样本的AddSent和AddOneSent行上评估鲁棒性。按模糊排序选取的33%子集……（摘要内容在此处截断）

    arXiv:2611.19763v1 Announce Type: new  Abstract: Dataset cartography argues that training dynamics can reveal regions of a dataset that matter differently for learning: easy-to-learn examples aid optimization, hard-to-learn examples often expose annotation problems, and ambiguous examples contribute strongly to out-of-distribution generalization. A striking implication of the original work is that a model trained on only 33% ambiguous examples can outperform full-data training on several OOD evaluations, including a QNLI reformulation of SQuAD evaluated on Adversarial SQuAD. We revisit this claim in a stricter setting: open-ended extractive question answering, where models must predict answer spans rather than classify question-sentence pairs. For each encoder family, we compute clean-source SQuAD training dynamics, train on full, random, easy-ranked, ambiguous-ranked, and hard-ranked subsets, and evaluate robustness on adversarial-only AddSent and AddOneSent rows. Ambiguous-ranked 33%
    
[^227]: ReCALM：一种面向分布式持续服务构建的残差增强型众包解析学习机制

    ReCALM: A Residual-enhanced Crowdsourced Analytic Learning Mechanism for Distributed Continual Service Construction

    [https://arxiv.org/abs/2611.19276](https://arxiv.org/abs/2611.19276)

    该论文提出ReCALM，一种残差增强型众包解析学习机制，通过引入深度非线性表示能力克服传统解析学习受限于单层线性网络而导致的欠拟合问题，同时保持无梯度的闭式递归求解以避免异构数据持续更新中的灾难性遗忘，从而支持分布式众包环境下的持续AI服务构建。

    

    众包通过利用分布式工作者的集体能力，已成为构建AI服务的一种有前景的范式。随着消费者需求的不断演变，服务模型必须在获取新知识的同时保留先前学习到的信息。然而，由于对梯度反向传播的固有依赖，现有方法在异构数据上的持续模型更新过程中通常会出现灾难性遗忘问题。最近的研究引入了解析学习，这是一种通过递归闭式解实现不变性的无梯度范式。然而，现有的解析方法通常受限于单层线性网络，由于无法进行深度表示学习，往往容易欠拟合。为了解决这一问题，我们提出了ReCALM，一种残差增强型众包解析学习机制，能够实现深度非线性不变（摘要内容在此处被截断）

    arXiv:2611.19276v1 Announce Type: new  Abstract: Crowdsourcing has emerged as a promising paradigm for constructing AI services by harnessing the collective capabilities of distributed workers. As consumers' demands continually evolve, service models must acquire new knowledge while retaining previously learned information. However, due to the inherited reliance on gradients for backpropagation, existing approaches typically suffer from catastrophic forgetting during continual model updates on heterogeneous data. Recent studies have introduced analytic learning, a gradient-free paradigm that achieves invariance through recursive closed-form solutions. Nevertheless, existing analytic methods are typically constrained by single-layer linear networks, which tend to suffer from underfitting due to their inability to perform deep representation learning. To address this issue, we propose ReCALM, a Residual-enhanced Crowdsourced Analytic Learning Mechanism that achieves deep nonlinear invari
    
[^228]: FlashREINFORCE：面向智能体语言模型的无评论家单次轨迹异步强化学习

    FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models

    [https://arxiv.org/abs/2611.18627](https://arxiv.org/abs/2611.18627)

    提出 FlashREINFORCE——一种无评论家、每个提示词仅单次采样的异步强化学习框架，通过单批次 REINFORCE、序列信赖域与样本均值优化，在策略滞后约四步的情况下仍能稳定训练智能体语言模型，同时保持提示词覆盖度。

    

    异步强化学习适用于轨迹生成时间不规则的长程智能体，但基于组间相对优势的更新需要对每个提示词进行多次轨迹采样。然而，基于分组的采样方式在固定轨迹生成预算下限制了提示词的覆盖范围，并引入了组同步屏障。每个提示词仅采样一次虽可避免这些限制，但会失去用于降低方差的组基线，导致更新噪声更大，在过时的、长度不一的轨迹上更容易不稳定。我们提出 FlashREINFORCE，一个无评论家的单次通过框架，其建立在单批次 REINFORCE、序列信赖域和样本均值优化之上。这些组件在保留提示词覆盖范围和带符号反馈的同时，控制了策略漂移和由序列长度放大的负向更新。实验表明，在策略滞后约四次更新的条件下，FlashREINFORCE 在 DeepSeek-R1-Distill-Qwen-1.5B 上经过 6,000 次更新仍保持稳定。在 Qwen2.5-Math-1.5B 上，其平均成绩达到 38.0（摘要至此被截断）。

    arXiv:2611.18627v1 Announce Type: new  Abstract: Asynchronous reinforcement learning suits long-horizon agents with irregular rollout times, but group-relative updates require repeated rollouts of each prompt. However, group-based sampling limits prompt coverage at a fixed rollout budget and introduces group synchronization barriers. One rollout per prompt avoids these constraints but removes the variance-reducing group baseline, making updates noisier and more prone to instability on stale, variable-length trajectories. We present FlashREINFORCE, a critic-free, one-pass framework built on One-Batch REINFORCE, a Sequence Trust Region, and Sample-Mean Optimization. These components preserve prompt coverage and signed feedback while controlling policy drift and length-amplified negative updates. In our experiments, FlashREINFORCE remains stable through 6,000 updates on DeepSeek-R1-Distill-Qwen-1.5B at a policy lag of approximately four updates. On Qwen2.5-Math-1.5B, it reaches 38.0 mean 
    
[^229]: 可迁移的解释：用稀疏自编码器解读医学信息检索

    Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders

    [https://arxiv.org/abs/2611.18179](https://arxiv.org/abs/2611.18179)

    本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。

    

    回答用户查询的医学AI系统依赖于从大型文档集合中检索医学知识。在此背景下，在医学数据集上微调通用模型可以提升检索相关性。然而，现代检索系统所依赖的稠密嵌入在很大程度上仍是不透明的，难以理解其检索结果背后的机制。此外，针对特定医学数据集进行适配可能会失去对新临床语料库的可迁移性。在本工作中，我们通过将稀疏自编码器（SAEs）应用于医学信息检索来解决这些问题，将稠密文本嵌入转化为语义上可解释的表示。为此，我们的模型将重建目标与InfoNCE对比损失相结合，后者能够直接保留查询与文档之间的相关性。我们在六个医学检索数据集上（涵盖域内与域外场景）评估了三种嵌入模型。在医学语料库上训练后，我们的稀疏……

    arXiv:2611.18179v1 Announce Type: cross  Abstract: Medical AI systems that answer users' queries rely on retrieving medical knowledge from large document collections. In this context, finetuning general models on medical datasets can improve retrieval relevance. Yet the dense embeddings underlying modern retrieval systems remain largely opaque, making it difficult to understand the mechanics behind their results. Moreover, adapting to a specific medical dataset risks losing transferability to new clinical corpora. In this work, we tackle these issues by applying Sparse Autoencoders (SAEs) to medical information retrieval, transforming dense text embeddings into semantically interpretable representations. To do so, our model combines a reconstruction objective with an InfoNCE contrastive loss that directly preserves query-document relevance. We evaluate three embedding models across six medical retrieval datasets, both in-domain and out-of-domain. Trained on a medical corpus, our sparse
    
[^230]: LoCoJAX：面向TPU上小模型强化学习的主机本地协调

    LoCoJAX: Host-Local Coordination for Small-Model RL on TPUs

    [https://arxiv.org/abs/2611.16301](https://arxiv.org/abs/2611.16301)

    该论文发现单一控制器协调模式在小模型强化学习中开销占比过高（0.6B模型下主机编排占步骤时间36%），并提出LoCoJAX通过主机本地执行rollout和奖励计算来提升TPU上小模型RL的效率。

    

    强化学习（RL）越来越多地针对小型语言模型，而在多个加速器上高效执行这些工作负载需要有效的协调。在TPU上，基于Pathways的强化学习使用单一控制器来协调跨主机的计算。这种执行模型对大模型非常有效，但当小模型反复调用短计算时，其协调计算成本更难摊销。为了表征这些计算成本，我们在TPU v7x上以解码为主的工作负载下，对参数量从0.6B到405B的稠密模型进行了扫描，并测量主机和网络计算开销如何随模型规模变化。在0.6B规模下，主机编排占步骤时间的36%，网络通信占2%；而对于大模型，估计的网络计算成本则主导了非计算时间。基于这些发现，我们提出了本地协调JAX（LoCoJAX），它实现了本地化的rollout和奖励计算

    arXiv:2611.16301v2 Announce Type: replace  Abstract: Reinforcement learning (RL) increasingly targets small language models, and executing these workloads efficiently across multiple accelerators requires effective coordination. On TPUs, Pathways-based RL uses a single controller to coordinate computation across hosts. This execution model is effective for large models, but its coordination computational costs are harder to amortize when small models repeatedly invoke short computations. To characterize these computational costs, we sweep dense models from 0.6B to 405B parameters on TPU v7x under a decode-heavy workload and measure how host and network computational overhead vary with model size. At 0.6B, host orchestration occupies 36% of step time and network communication 2%, whereas for large models, estimated network computational costs dominate the non-compute time. Motivated by these findings, we present Local Coordination JAX (LoCoJAX), which implements local rollout and reward
    
[^231]: 在线平滑校准的匹配下界

    A Matching Lower Bound for Online Smooth Calibration

    [https://arxiv.org/abs/2611.16097](https://arxiv.org/abs/2611.16097)

    本文通过构造自适应对手和遗忘对手，证明在线平滑校准误差的极小极大下界为 Ω(√T)，填补了已知的上下界差距，确立了平滑校准误差的极小极大最优性。

    

    我们研究序列化二元预测中的平滑校准误差，这是一种衡量概率预测偏离校准程度的指标，其对预测的微小扰动具有鲁棒性。已知极小极大平滑校准误差介于 Ω(T^{1/3}) 与 O(√T) 之间（Qiao 和 Zheng，2024），其中 T 为轮数。我们对自适应对手和遗忘对手这两类情形均填补了这一空白。我们构造了一个自适应对手，其结果均值随预测者的累积残差而移动，这使得补偿过去的残差变得代价高昂，从而迫使任何预测者产生 Ω(√T) 的期望平滑校准误差。通过针对预测者的平均预测来模拟该对手，可以得到一个具有相同保证的遗忘对手。这些下界与已知的 O(√T) 上界在普适常数范围内匹配，因此是极小极大最优的。我们的结果还意味着匹配的 Θ(√（原文摘要至此被截断）

    arXiv:2611.16097v2 Announce Type: replace  Abstract: We study smooth calibration error in sequential binary prediction, a measure of how far probabilistic forecasts are from being calibrated that is robust to small perturbations of the forecasts. The minimax smooth calibration error is known to lie between $\Omega(T^{1/3})$ and $O(\sqrt T)$(Qiao and Zheng, 2024), where $T$ is the number of rounds. We close this gap for both adaptive and oblivious adversaries. We construct an adaptive adversary whose outcome mean moves with the forecaster's cumulative residual, which makes compensating for past residuals costly and forces every forecaster to incur $\Omega(\sqrt T)$ expected smooth calibration error. Simulating this adversary against the forecaster's average prediction yields an oblivious adversary with the same guarantee. These lower bounds match the known $O(\sqrt T)$ upper bound up to universal constants, and are therefore minimax optimal. Our results also imply matching $\Theta(\sqrt
    
[^232]: 神经网络计算中的概率状态空间枚举

    Probabilistic State Space Enumeration in Neural Network Computation

    [https://arxiv.org/abs/2611.15062](https://arxiv.org/abs/2611.15062)

    该论文受网络可靠性分析启发，提出概率状态空间枚举方法（PSSE），将Transformer中MLP的密集输出表示为通道组联合状态上重要性校正掩码输出的期望，并配合NORDER-II算法高效枚举高概率状态，实现具有可证明误差界限的高效近似。

    

    许多剪枝神经网络的方法会根据重要性对权重或权重组进行排序，并保留单一子集。我们研究了Transformer语言模型中多层感知机（MLP）的通道组如何共同贡献于输出。受网络可靠性分析的启发，概率状态空间枚举（PSSE）将密集的MLP输出表示为G个通道组的2^G个联合状态上经过重要性校正的掩码输出的期望值。对最可能状态的概率加权输出进行求和可得到一个近似结果，其绝对误差受被省略的概率质量乘以最大掩码输出范数所限定。NORDER-II对ORDER-II进行了改进，可以按概率递减的顺序枚举状态，而无需对整个状态空间进行排序。其前B个状态在该预算下能最大化所捕获的概率质量。权重熵评分器根据预训练权重为每一层确定固定的排序。我们在六个（模型的）全部168个MLP层上进行了评估……

    arXiv:2611.15062v2 Announce Type: replace  Abstract: Many methods for pruning neural networks rank weights or groups of weights by importance and retain a single subset. We study how channel groups contribute jointly to the output of multilayer perceptrons (MLPs) in Transformer language models. Inspired by network reliability analysis, probabilistic state space enumeration (PSSE) represents the dense MLP output as the expected importance-corrected masked output over $2^G$ joint states of $G$ channel groups. Summing the most probable states' probability-weighted outputs gives an approximation whose absolute error is bounded by the omitted probability mass times the maximum masked-output norm. NORDER-II adapts ORDER-II to enumerate states in decreasing probability without sorting the full space. Its first $B$ states maximize the probability mass captured at that budget. A weight-entropy scorer fixes each layer's ordering from pretrained weights. We evaluate all 168 MLP layers across six 
    
[^233]: 面向二元决策的具有可行性保证的线性策略规范性学习

    Prescriptive Learning of Linear Policies with Feasibility Guarantees for Binary Decision-Making

    [https://arxiv.org/abs/2611.10923](https://arxiv.org/abs/2611.10923)

    本文提出一种规范性学习框架，通过在所有训练场景上联合优化线性决策策略与可行分配，为不确定性下的二元决策提供可行性保证，并解决了预测式学习缺乏反事实认知且在分布偏移下易产生不可行决策的问题。

    

    不确定性下的二元决策广泛应用于众多领域，包括能源系统调度、交通运输和人力规划等。尽管机器学习已被越来越多地用于加速或取代此类场景中的混合整数求解器，但现有方法依赖于预测式学习，即训练模型从带标签的数据中复现最优决策，这种方法既缺乏对反事实结果的认知，也无法保证所生成决策的可行性。在分布偏移的情况下，这一局限性会进一步加剧，因为基于历史数据训练的策略可能产生系统性不可行的决策。我们提出了一种面向二元决策的规范性学习框架，以同时解决上述两个局限。该框架不再预测最优标签，而是在所有训练场景上对线性决策策略与可行分配进行联合优化……

    arXiv:2611.10923v2 Announce Type: replace  Abstract: Binary decision-making under uncertainty arises across a wide range of applications, including energy systems scheduling, transportation, and workforce planning. While machine learning has been increasingly applied to accelerate or replace mixed-integer solvers in such settings, existing approaches rely on predictive learning, training models to reproduce optimal decisions from labeled data, which lacks knowledge of counterfactual outcomes and provides no guarantees that the resulting decisions will be feasible. This limitation is further compounded under distributional shift, where policies trained on historical data may produce systematically infeasible decisions. We propose a prescriptive learning framework for binary decision-making that addresses both limitations. Rather than predicting optimal labels, the framework jointly optimizes over linear decision policies and feasible assignments across all training scenarios, allowing f
    
[^234]: MAGE：迈向可泛化的多时间尺度脑电表征

    MAGE: Towards Generalizable Multi-timescale EEG Representations

    [https://arxiv.org/abs/2611.09928](https://arxiv.org/abs/2611.09928)

    提出了MAGE脑电基础模型，通过新颖的多时间尺度注意力机制和可训练的动态聚合模块，自适应地捕捉并整合脑电信号中跨越多个时间尺度的神经动态信息，突破了现有模型固定时间尺度的局限。

    

    脑电图（EEG）数据在理解人类大脑活动方面发挥着重要作用，并具有多种重要应用。尽管当前主要基于脑电基础模型的方法在多个数据集和任务上展现出强大的性能，但它们通常依赖于固定的时间尺度。这种设计选择限制了其捕捉神经动态行为特征的能力，而这些特征往往跨越多个时间尺度，并被自然地记录在脑电信号中。为解决这一局限性，我们提出了MAGE（Multi-timescale Adaptive Generalization for EEG，脑电多时间尺度自适应泛化模型），一个能够动态适应以捕捉并整合不同时间尺度信息的基础模型。具体而言，MAGE结合了两个关键机制：(i) 一种新颖的多时间尺度注意力机制，能够学习跨越多个时间尺度的表征；(ii) 一个可训练的聚合模块，能够动态地识别……（摘要此处截断）

    arXiv:2611.09928v2 Announce Type: replace  Abstract: Electroencephalography (EEG) data plays an important role in understanding human brain activity, with various major applications. While current approaches, which largely build on EEG foundation models, have demonstrated strong performance across datasets and tasks, they typically rely on a fixed temporal scale. This design choice limits their ability to capture characteristics of neural dynamic behavior, often spanning across multiple timescales, which are naturally captured in EEG recordings. To address this limitation, we propose MAGE (\textbf{Multi-timescale Adaptive Generalization for EEG}), a foundation model that dynamically adapts to capture and integrate information across different timescales. Specifically, MAGE combines two key mechanisms: (i) a novel multi-timescale attention mechanism, capable of learning representations across multiple temporal scales and (ii) a trainable aggregation module, that dynamically identifies a
    
[^235]: 论智能体式操作策略

    On Agentic Manipulation Policies

    [https://arxiv.org/abs/2611.09892](https://arxiv.org/abs/2611.09892)

    提出AMP1智能体式操作框架，证明通用AI模型直接控制机械臂可零样本超越VLA和WAM等专用策略，在RoboLab-120上达到85.6%的成功率。

    

    近年来机器人操作领域的研究主要聚焦于通过演示数据训练的专用策略。我们证明，通用型AI模型能够比专用策略（如视觉-语言-动作模型（VLA）和世界-动作模型（WAM））更可靠地控制机械臂。我们系统地分析了智能体式操作策略（AMP），即由通用型AI模型直接指挥机械臂，并随着任务的展开自适应地调整其动作。我们提出了AMP1，一个可在单臂、双臂和移动操作中无需任何改动即可运行的参考框架。AMP1在RoboLab-120和CaP-Gym基准测试中取得了最先进的结果，且无需任何演示数据或任务特定训练。在RoboLab-120上，AMP1零样本成功率达到85.6%，而领先的VLA和WAM策略分别仅为49.2%和42.9%。该智能体在涵盖十二个任务和三台机器人的108次真实世界测试中零样本成功了104次，显著降低了失败率。

    arXiv:2611.09892v2 Announce Type: replace-cross  Abstract: Recent work on robot manipulation has focused on specialist policies trained on demonstrations. We show that generalist AI models can control manipulators more reliably than specialist policies such as vision-language-action models (VLAs) and world-action models (WAMs). We systematically analyze agentic manipulation policies (AMPs), in which a generalist AI model commands the manipulator directly, adapting its actions as the task unfolds. We present AMP1, a reference harness that runs unchanged across single-arm, bimanual, and mobile manipulation. AMP1 achieves state-of-the-art results on the RoboLab-120 and CaP-Gym benchmarks, with no demonstrations or task-specific training. On RoboLab-120, AMP1 reaches an 85.6% success rate zero-shot, compared to 49.2% and 42.9% for leading VLA and WAM policies. The agent succeeds zero-shot in 104 of 108 real-world trials spanning twelve tasks and three robot instances, reducing failure rate
    
[^236]: TDC ADMET上的校准不确定性量化：在22个数据集上能分辨出什么？

    Calibrated Uncertainty Quantification on TDC ADMET: What Can Be Resolved Across 22 Datasets?

    [https://arxiv.org/abs/2611.09373](https://arxiv.org/abs/2611.09373)

    在22个TDC ADMET数据集的基准测试中，集成蒙特卡洛dropout取得了最佳平均排名，但统计上无法将其与其他集成方法或LightGBM明确区分开，说明该基准的分辨能力有限。

    

    吸收、分布、代谢、排泄和毒性（ADMET）模型通常依赖小型且异构的数据集，然而其预测可能决定哪些化合物进入实验测试阶段。因此，有效的评估必须同时考察预测性能以及所报告的概率或区间的质量。我们探究了哪些常用的不确定性量化方法在22个Therapeutics Data Commons（TDC）ADMET数据集上表现最佳，以及该基准究竟能够分辨出什么。我们采用嵌套五折交叉验证，在十个化学分区和五个模型随机种子下比较了十一条经过校准的流水线。集成蒙特卡洛（MC）dropout在TDC性能、Brier分数和90%区间评分上的平均排名最佳。然而，就TDC性能而言，它与其他集成流水线并无显著区分；就Brier分数和区间评分而言，LightGBM也不能被排除为最佳。LightGBM在单个数据集上获胜的次数多于任何其他方法……

    arXiv:2611.09373v2 Announce Type: replace  Abstract: Absorption, distribution, metabolism, excretion, and toxicity (ADMET) models often rely on small, heterogeneous datasets, yet their predictions may determine which compounds proceed to experimental testing. Useful evaluations must therefore assess both predictive performance and the quality of reported probabilities or intervals. We ask which common uncertainty-quantification methods perform best on the 22 Therapeutics Data Commons (TDC) ADMET datasets and what this benchmark can resolve. We compare eleven calibrated pipelines using nested five-fold cross-validation across ten chemical partitions and five model seeds. Ensemble Monte Carlo (MC) dropout has the best average rank for TDC performance, Brier score, and 90% interval score. For TDC performance, however, it is not separated from the other ensemble pipelines; for Brier and interval score, LightGBM is also not ruled out as best. LightGBM wins more individual datasets than any 
    
[^237]: 强化学习中环境-训练器边界处的静默故障检测

    Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning

    [https://arxiv.org/abs/2611.08144](https://arxiv.org/abs/2611.08144)

    该论文发现强化学习中环境与训练器边界处的静默故障几乎无法通过学习曲线等基于结果的检查发现，而直接检查运行过程中跨越边界传输的数据本身则能检测到每一个注入的故障。

    

    机器人强化学习越来越多地在训练器之外运行环境：在另一个进程中、在另一台机器上、使用另一种编程语言。这种边界上的许多故障不会引发任何错误报告。我们将该边界建模为一种线协议，并探究是什么能够察觉到其中发生的故障。在 Pendulum、HalfCheetah、Hopper 和一个 7 关节机械臂上，我们向跨越线路传输的数据中注入了多达 23 种故障，并向训练器的日志中注入了 2 种故障，范围从单个值出错到全部值出错。除非每个值都是错误的，否则基于学习曲线阈值的检查最多只能发现十分之一的故障运行，与十个无故障种子对比的检查最多发现 21%。对所有 1,060 个最终策略的干净评估与曲线检查结果一致：在 728 个边界故障运行中，两种曲线检查均通过，其中 723 个的得分不比无故障种子低超过三个标准差，尽管几乎所有运行最终都收敛到了不同的权重，而且对于奖励类故障，记录的回报值也与正常情况不同。而对运行过程本身（而非其结果）的检查则发现了每一个故障。一个步进台阶……（摘要在此处截断）

    arXiv:2611.08144v1 Announce Type: cross  Abstract: Robot reinforcement learning increasingly runs environments outside the trainer: in another process, on another machine, in another language. Many faults at this boundary raise no error. We make the boundary a wire protocol and ask what notices a fault in it.   On Pendulum, HalfCheetah, Hopper and a 7-joint arm (Pusher), we injected up to 23 faults into what crosses the wire and 2 into the trainer's log, from one value to every value. Unless every value was wrong, a learning-curve threshold caught at most one run in ten, and a comparison with ten fault-free seeds at most 21%. A clean evaluation of all 1,060 final policies agreed with the curves: of the 728 boundary-fault runs both curve checks passed, 723 scored no more than three standard deviations below fault-free seeds, though nearly all ended on other weights and, for reward faults, logged other returns.   Checks of the run itself, not of its outcome, saw every fault. A step ledge
    
[^238]: 脉动阵列线性注意力：面向TPU协同设计的分块分组门控Delta注意力

    Systolic Linear Attention: Co-Designing Tile-Grouped Gated Delta Attention for TPUs

    [https://arxiv.org/abs/2611.06828](https://arxiv.org/abs/2611.06828)

    提出了脉动线性注意力（SLA），针对TPU的256×256脉动矩阵单元重新设计门控Delta线性注意力——将键扩展至256、把衰减门分组为64通道瓦片并用全分块矩阵乘法的单一融合内核计算——在TPU v7x上比Tokamax KDA内核快1.6至2.5倍。

    

    门控Delta线性注意力层，如门控DeltaNet（GDN）、Kimi Delta Attention（KDA）和GDN-2，都是针对GPU调优的——在GPU上，小尺寸张量核心瓦片和快速的向量寄存器使得每通道衰减重缩放和块内顺序求解的代价很低。而Google TPU将计算集中在宽256×256的脉动矩阵单元中，并辅以较慢的向量单元，直接移植这些层会导致每个矩阵瓦片有一半空间空置，且性能瓶颈落在向量运算上。我们提出了脉动线性注意力，它围绕矩阵单元重新塑造门控Delta注意力：将键维度扩展到256并采用分组值头，将衰减门分组为四个64通道瓦片，并在单个融合内核中完全通过分块矩阵乘积来计算块求解。在TPU v7x上，一个完整的SLA-R4层比开源的Tokamax KDA内核快1.6倍至2.5倍（使用精确分块求逆时为1.4倍至2.4倍）

    arXiv:2611.06828v1 Announce Type: new  Abstract: Gated-delta linear attention layers such as Gated DeltaNet (GDN), Kimi Delta Attention (KDA), and GDN-2 are tuned for GPUs, where small tensor-core tiles and fast vector registers make per-channel decay rescaling and sequential in-block solves cheap. On Google TPUs, which concentrate compute in wide $256 \times 256$ systolic matrix units backed by a slower vector unit, porting these layers directly leaves half of each matrix tile empty and bottlenecks on vector operations. We present \textbf{Systolic Linear Attention (SLA)}, which reshapes gated delta attention around the matrix unit by widening keys to $256$ with grouped value heads, grouping decay gates into four $64$-channel tiles, and computing the chunk solve entirely with block matrix products in a single fused kernel. On TPU v7x, a full \texttt{SLA-R4} layer runs $1.6\times$--$2.5\times$ faster than the open-source Tokamax KDA kernel ($1.4\times$--$2.4\times$ with exact chunk inve
    
[^239]: 通过精确似然在舍入、有界和删失数值标签上训练自回归语言模型

    Training Autoregressive Language Models on Rounded, Bounded, and Censored Numerical Labels via Exact Likelihood

    [https://arxiv.org/abs/2611.02461](https://arxiv.org/abs/2611.02461)

    提出边界边缘化方法，首次通过精确似然让自回归语言模型在舍入、有界和删失等区间型数值标签上训练，仅用至多两条对数长度的边界路径即可计算精确概率和梯度。

    

    数值标签通常指定一组可能的值：测量值会被舍入，实验室检测可能只报告一个界限，或者程序在超时时被中止。我们提出了边界边缘化，这是首个利用自回归语言模型现有的数字标记，在此类观测的精确似然上进行训练的方法。一种固定宽度、保序的数字表示方式将连续的网格值归入共享的数字前缀之下。一个区间是这些组的并集，因此其精确概率和梯度可由至多两条边界路径（即区间端点的逐位数字前缀）导出，每条路径的长度与网格值数量呈对数关系。由此得到的目标函数统一了精确、舍入、有界和删失等监督形式，并以每个观测所提供的分辨率来训练数字决策。相同的计算还支持精确的分布查询以及连续排序概率的无偏梯度。

    arXiv:2611.02461v2 Announce Type: replace  Abstract: Numerical labels often specify a set of possible values: a measurement is rounded, a laboratory test reports only a bound, or a program is stopped at a timeout. We introduce boundary marginalization, the first method that trains an autoregressive language model on the exact likelihood of such observations through its existing digit tokens. A fixed-width, order-preserving numeral representation groups consecutive grid values under shared digit prefixes. An interval is a union of such groups, so its exact probability and gradient follow from at most two boundary paths, the successive digit prefixes of its endpoints, each of length logarithmic in the number of grid values. The resulting objective unifies precise, rounded, bounded, and censored supervision and trains digit decisions at the resolution supplied by each observation. The same computation supports exact distributional queries and unbiased gradients for the continuous ranked p
    
[^240]: 轮廓感知的被丢弃参数Hessian质量诊断

    Profile-aware diagnostics of discarded parameter-Hessian mass

    [https://arxiv.org/abs/2611.01335](https://arxiv.org/abs/2611.01335)

    提出一种轮廓感知的诊断框架，通过在不同分组规模和组内曲率轮廓参考下测量参数Hessian的块外平方占比，来量化块对角曲率模型所丢弃的曲率质量，并评估候选修正的实际效果。

    

    评估块对角曲率模型涉及两个问题：它们省略了多少曲率，以及一个特定的修正在有限步长内改变了什么。我们针对不同组规模以及观测到的组内曲率轮廓的参考基准，测量参数Hessian的块外平方占比。在CIFAR-100的ResNet-18及其无跳跃连接的对应模型中，相似的初始丢弃占比伴随着不同的曲率轮廓。在已保存的训练好的ResNet-18权重处，将分组方式从阶段组改为层组，会使估计的块外质量从轮廓包络之下移动到其之上；一个近似条件筛选方法在所有六个种子/输入单元中都解决了这一比较问题，而某些无跳跃连接模型之间的比较仍未得出结论。另外，在已保存的ResNet-50/Stanford Cars状态下，我们通过相同的Hessian信息映射，比较了由组外Hessian作用、剩余模型残差以及二者组合所构成的候选修正。在单位尺度下，仅组外输入（摘要在此处被截断）

    arXiv:2611.01335v2 Announce Type: replace  Abstract: Evaluating block-diagonal curvature models involves two questions: how much curvature they omit, and what a specific correction changes at a finite step. We measure the squared off-block fraction of the parameter Hessian against references for group sizes and the observed within-group curvature profile. In CIFAR-100 ResNet-18 and a skip-free counterpart, similar initial discarded fractions accompany different profiles. At saved trained ResNet-18 weights, changing from stage to layer groups moves estimated off-block mass from below to above the profile envelope; an approximate conditional screen resolves this comparison in all six seed/input cells, while some skip free comparisons remain unresolved. Separately, in saved ResNet-50/Stanford Cars states, we compare candidates formed from off-group Hessian action, the remaining model residual and their combination through the same Hessian-informed map. At unit scale off-group input alone 
    
[^241]: 你的学习率调度可能并无原则依据：权重归一化训练中的有效学习率调度

    Your LR Schedule May Not Be Principled: Effective Learning Rate Scheduling in Weight-Normalized Training

    [https://arxiv.org/abs/2610.40362](https://arxiv.org/abs/2610.40362)

    本文发现有效学习率轨迹是权重归一化与权重衰减训练损失差异的关键因素，并据此推导出以 1/√t 衰减为基础的 AIR 最优学习率调度，在大规模 MoE 预训练中全面超越余弦等传统调度。

    

    权重归一化训练正逐渐成为传统权重衰减的一种有吸引力的替代方案，近期研究报道了其在预训练损失上的显著提升。我们发现，有效学习率轨迹解释了权重归一化与权重衰减之间大部分的损失动态差异。逐步匹配分块的有效学习率轨迹可以使两者的损失曲线高度重合。因此，调度设计应当以有效学习率而非名义学习率为目标。基于这一原则，我们对权重归一化下的方向对齐动态进行建模，推导出 1/√t 衰减作为最优调度。针对有限时长的预训练，我们提出了 AIR（仿射反平方根衰减）来匹配给定的峰值和末端学习率。在 128B 和 500B token 的权重归一化混合专家模型预训练实验中，AIR 在损失、下游性能和超参数鲁棒性方面均优于余弦调度及其他备选调度，降低了末端训练损失。

    arXiv:2610.40362v2 Announce Type: replace  Abstract: Weight-normalized (WN) training is emerging as a compelling alternative to conventional weight decay (WD), with recent studies reporting substantial pretraining loss gains. We find that the effective learning rate (ELR) trajectory accounts for most of the loss dynamics difference between WN and WD. Matching blockwise ELR trajectories step by step closely aligns their loss curves. Schedule design should therefore target ELR rather than the nominal learning rate (LR). Building on this principle, we model directional-alignment dynamics under WN and derive $1/\sqrt{t}$ decay as the optimal schedule. For finite-horizon pretraining, we propose AIR (Affine Inverse-square-Root decay) to match prescribed peak and terminal LRs. Experiments on 128B- and 500B-token Mixture-of-Experts pretraining under WN show that AIR outperforms cosine and alternative schedules in loss, downstream performance, and hyperparameter robustness, reducing terminal tr
    
[^242]: TGTV：无人机遥测中可解释的多变量时间序列异常检测

    TGTV: Interpretable Multivariate Time-Series Anomaly Detection in UAV Telemetry

    [https://arxiv.org/abs/2610.38106](https://arxiv.org/abs/2610.38106)

    本文提出TGTV框架，将时空图学习与大语言模型相结合，实现无人机多变量遥测的可解释异常检测和可追溯故障分析。

    

    固定翼无人机（UAV）多变量遥测数据的异常检测因工况转换、传感器异步采样以及可能掩盖故障特征的控制系统补偿而变得复杂。现有的时序检测方法在表示跨子系统的耦合故障、跟踪故障演化过程以及利用定量证据约束语言模型诊断方面的能力有限。本文提出了TGTV，一个面向多变量遥测异常检测和可追溯故障分析的可解释框架，它将时空图学习与大语言模型相融合。该框架采用时间卷积网络和具有加权融合邻接矩阵的图注意力网络，提取多尺度时间模式和传感器之间的空间依赖关系；随后由混合的时间Transformer和变分自编码器利用预测误差与重构概率评估窗口级异常。

    arXiv:2610.38106v2 Announce Type: replace  Abstract: Anomaly detection in multivariate fixed-wing unmanned aerial vehicle (UAV) telemetry is complicated by operating-condition transitions, asynchronous sensor sampling, and control-system compensation that can obscure fault signatures. Existing temporal detection methods have limited capacity to represent coupled faults across subsystems, track their evolution, and constrain language-model diagnoses with quantitative evidence. This paper proposes TGTV, an interpretable framework for multivariate telemetry anomaly detection and traceable fault analysis that integrates spatiotemporal graph learning with a large language model. A temporal convolutional network and a graph attention network with weighted fused adjacency extract multiscale temporal patterns and spatial dependencies among sensors. A hybrid temporal Transformer and variational autoencoder then evaluates window-level anomalies using prediction errors and reconstruction probabil
    
[^243]: FOCUS：面向长程语言模型智能体的主动评论家学习

    FOCUS: Active Critic Learning for Long-Horizon Language Model Agents

    [https://arxiv.org/abs/2610.35801](https://arxiv.org/abs/2610.35801)

    FOCUS是一种强化学习方法，通过选择性分支对预计最能降低估计误差的状态进行续写采样，实现评论家网络的主动学习并获得更准确的价值估计，从而解决长程语言模型智能体中稀疏奖励导致的信用分配难题。

    

    稀疏的结果奖励使得长程语言模型智能体难以将功劳分配给较早的动作。学习到的评论家网络可以为中间状态提供价值估计，但这些估计可能不准确，而通过从轨迹内的点反复采样续写来估计中间价值又会产生巨大成本。为了缓解这些问题，我们提出了FOCUS，这是一种强化学习方法，利用选择性分支来帮助评论家网络学习其价值估计器。Rollout控制器通过从预计最能降低评论家估计误差的状态进行续写采样，促进评论家的主动学习。这些精炼后的价值估计随后使用TD(λ)优势对连续状态之间的片段进行评分，而采样的证据则为评论家提供训练目标。FOCUS通过模块化的上下文管理接口支持长程智能体强化学习。

    arXiv:2610.35801v1 Announce Type: new  Abstract: Sparse outcome rewards make it difficult for long-horizon language-model agents to assign credit to earlier actions. Learned critics provide value estimates for intermediate states, but these estimates can be inaccurate, while estimating intermediate values with repetitive continuation samples from points within a trajectory incurs substantial costs. To mitigate these issues, we introduce FOCUS, an RL method that uses selective branching to help the critic network learn its value estimator. The rollout controller facilitates active learning for the critic by sampling continuations from states that are expected to most reduce the critic's estimation error. These refined value estimates then score the segment between consecutive states with a TD($\lambda$) advantage, while the sampled evidence supplies training targets for the critic. FOCUS supports long-horizon agentic reinforcement learning through a modular context-management interface 
    
[^244]: 共享上下文，独立引擎：边缘SoC上的流式视觉-语言推理

    Shared Context, Separate Engines: Streaming Vision--Language Inference on Edge SoCs

    [https://arxiv.org/abs/2610.35309](https://arxiv.org/abs/2610.35309)

    提出了一种在边缘SoC上协调CPU、GPU、NPU等异构加速器并发执行有状态流式VLM推理的系统，通过将流的版本化上下文与各引擎的KV状态分离，实现任意引擎可接续处理数据流，从而支持流式生成、并行序列生成和状态感知调度。

    

    arXiv:2610.35309v2 公告类型：新 摘要：边缘SoC集成了CPU、GPU和NPU，然而视觉-语言模型（VLM）的运行时却将生成任务固定在单个引擎上，让其他引擎处于闲置状态。流式VLM进一步暴露了可利用的并行性：在较早的查询尚未回答完毕时，新帧就已经到达。各引擎在计算能力和内存访问方式上存在差异，并且在共同运行时会相互争抢资源，因此不存在一种在所有SoC和工作负载下都最优的统一阶段部署方案。后端局部的KV状态使得其他引擎无法接续未完成的工作。本文提出的系统协调异构加速器以并发执行有状态的VLM推理。它将流的版本化上下文与各引擎的KV状态分离，使得任何引擎都可以通过重用、导入或重建该状态来接续处理同一个流。有状态流式生成在一个引擎上摄取新帧的同时由另一个引擎进行回答，并发序列生成则并行处理不同的就绪请求。状态感知调度将每个请求发送至预计能最快完成它的引擎……

    arXiv:2610.35309v2 Announce Type: new  Abstract: Edge SoCs pack CPUs, GPUs, and NPUs, yet vision--language model (VLM) runtimes keep generation on one engine and leave the others idle. Streaming VLMs expose further parallelism: new frames arrive while earlier queries are still being answered. Engines differ in compute and memory access and contend when they co-run, so no single placement of stages wins across SoCs and workloads. Backend-local KV state keeps other engines from continuing the work. \name coordinates heterogeneous accelerators to execute stateful VLM inference concurrently. It separates a stream's versioned context from each engine's KV state, so any engine can continue the stream by reusing, importing, or reconstructing that state. Stateful Streaming Generation ingests new frames on one engine while another answers, and Concurrent Sequence Generation serves different ready requests in parallel. State-aware dispatch sends each request to the engine expected to finish it f
    
[^245]: 几何知识空间、注意力配对与架构

    Geometric knowledge spaces, Attention pairing and Architectures

    [https://arxiv.org/abs/2610.29526](https://arxiv.org/abs/2610.29526)

    本文提出一个几何化框架，通过形式化“知识世界”与“注意力配对”的概念，统一推导出标准注意力、因果注意力、线性注意力、张量注意力和循环神经网络等多种架构，并引入新的几何空间以启发未来神经网络架构的设计。

    

    在本文中，我们提出了一种几何方法来推导注意力机制并探索潜在的新型transformer架构。我们首先形式化了“知识世界”和“注意力配对”的概念，然后展示了该框架如何自然地引出标准注意力、因果注意力、线性注意力、张量注意力以及循环神经网络的架构。最后，我们引入了若干新的几何空间，这些空间可能启发新型神经网络架构的设计。

    arXiv:2610.29526v1 Announce Type: new  Abstract: In this paper, we propose a geometric approach to deriving attention mechanisms and exploring potential new transformer architectures. We begin by formalizing the notions of knowledge worlds and attention pairings, then demonstrate how this framework naturally leads to the architectures of standard attention, causal attention, linear attention, tensor attention, and recurrent neural networks. Finally, we introduce several new geometric spaces that may inspire novel neural network architectures.
    
[^246]: ProteinGeoBench：前沿结构生物学智能体擅长信息查找，却在几何问题上迷失方向

    ProteinGeoBench: Frontier Structural-Biology Agents Are Fluent at Lookup but Lost in Geometry

    [https://arxiv.org/abs/2610.26829](https://arxiv.org/abs/2610.26829)

    本文提出包含131个结构生物学任务的ProteinGeoBench基准，通过对4,249次模型运行的分析发现，前沿模型擅长事实查找但普遍在基于原子坐标的几何计算任务上失败，揭示了当前模型几何推理能力的重大短板。

    

    ProteinGeoBench是一个包含131个任务的基准测试集，它为智能体提供蛋白质结构文件，要求其查找已记录的事实、检查序列、根据原子坐标计算几何量或对结构进行推理。研究团队在每个任务上对四个前沿模型进行了多次运行（共计4,249次），并在一个子集上测试了26个模型。结果表明，单一的总分掩盖了两种截然不同的能力：四个前沿模型中有三个最擅长查找事实，而最不擅长基于坐标进行计算——例如Claude Opus 5在查找任务上得分为0.907，但在计算任务上仅为0.716。Gemini 3.1 Pro是唯一的例外。几何类任务在最难的任务中占比过高：在四个模型全部失败的十二个任务中，有一半是计算类任务，尽管计算类任务仅占整个测试集的五分之一，且其中八个任务在26个模型中无一能解决。三个领先模型之间的分差在0.031以内，差距过小而难以排序，且若每个任务只运行一次，在24%的bootstrap重采样中排名第一的模型会有所不同。

    arXiv:2610.26829v2 Announce Type: replace  Abstract: ProteinGeoBench is a set of 131 tasks that give an agent protein structure files and ask it to look up recorded facts, check sequences, calculate geometry from atom coordinates or reason about structures. We ran four frontier models on every task several times (4,249 runs) and 26 models on a subset. A single overall score hides two different skills. Three of the four frontier models are best at looking up facts and worst at calculating from coordinates: Claude Opus 5 scores 0.907 on lookup and 0.716 on calculation. Gemini 3.1 Pro is the exception. Geometry is overrepresented among the hardest tasks. Half of the twelve tasks that all four models fail are calculations, although calculations are a fifth of the suite, and eight of the twelve were solved by none of the 26 models. The three leading models are within 0.031 of one another, too close to order, and with one run per task a different model ranks first in 24% of bootstrap resampl
    
[^247]: 蛋白质语言模型中预训练数据过滤的有效性与持久性

    The Effectiveness and Durability of Pretraining Data Filtering in Protein Language Models

    [https://arxiv.org/abs/2610.26651](https://arxiv.org/abs/2610.26651)

    该研究利用ESM-2蛋白质语言模型发现，预训练数据过滤能有效削弱模型对病毒蛋白（尤其是人类感染病毒）的能力且随模型规模增大而更有效，但这种防御可通过约80美元成本的低成本微调轻易被恢复，揭示了预训练过滤作为安全保障的局限性。

    

    预训练数据过滤是开放权重语言模型最常见的安全保障措施之一。本研究以ESM-2系列蛋白质语言模型为例，研究了其有效性、持久性和可扩展性。我们发现，从预训练数据中过滤病毒蛋白质会显著降低模型在病毒蛋白质上的掩码语言建模、突变效应预测和长程接触预测能力，同时在很大程度上保持对非病毒蛋白质的性能。这些影响集中在感染人类的病毒的蛋白质上，并随着模型参数规模从800万扩展到30亿而增大。随着模型规模的增加，我们发现严格的过滤变得更加有效。然而，在来自被移除病毒家族的序列上进行微调可以恢复失去的性能。尽管恢复性能的成本随模型规模增加而上升，但即使对于最大的模型，其成本也仅需约80美元的H100计算时间。这项工作确立了预训练过滤器……

    arXiv:2610.26651v2 Announce Type: replace-cross  Abstract: Pretraining filtering is one of the most common safeguards for open-weight language models. Here we study its effectiveness, durability and scalability using the ESM-2 family of protein language models. We find that filtering viral proteins from the pretraining data meaningfully degrades masked language modelling, mutation effect prediction and long-range contact prediction on viral proteins, while largely preserving performance on non-viral proteins. These effects are concentrated among proteins from human-infecting viruses, and grow as we scale models from 8M to 3B parameters. As models increase in size, we find that stringent filtering becomes more effective. However, fine-tuning on sequences from the removed viral families restores the lost performance. Although the cost to recover performance increases with model scale, even for the largest models it costs about 80 USD of H100 time. This work establishes pretraining filter
    
[^248]: 流行率偏移与标签预算约束下的类型化决策模型：决策训练、校准与隐式先验

    Typed Decision Models under Prevalence Shift and Label Budgets: Decision Training, Calibration, and Implicit Priors

    [https://arxiv.org/abs/2610.26325](https://arxiv.org/abs/2610.26325)

    本文在流行率偏移与标签预算约束下系统比较三个类型化决策模型与其骨干模型及微调分类器，发现厂商决策训练并非总是有益，其校准优势因任务类型而异。

    

    类型化决策模型返回调用者所定义选项上的概率分布，Jev 的供应商宣称其模型经过校准。我们探讨三个问题：供应商的决策训练能否在其骨干模型之上带来改进；类型化决策模型的校准与阈值能否像微调分类器一样经受住流行率（标签）偏移；以及一个经过训练的模型需要多少标注标签才能超越类型化决策模型。我们在模拟偏移条件下，在来自五个公开数据集的相同候选集决策上（在意图数据集上限于黄金标签所属的意图组内），对三个类型化决策模型（Jev、Tev1-4B、Nimble-9B）与其基础模型以及经过训练的分类器进行了比较；我们在确认性分析之前冻结了 35 项配对检验（并披露了此前进行的测试数据检查），同时估计了标签预算曲线。首先，决策训练并非总是有益：与 Qwen3.5-9B 仅相差一个 LoRA 适配器的 Nimble-9B 在意图任务上校准更好，但在金融情感任务上 Macro-F1 却低了 12 个百分点（原文摘要至此被截断）。

    arXiv:2610.26325v2 Announce Type: replace-cross  Abstract: Typed decision models (TDMs) return probabilities over caller-defined options, and Jev's vendor describes it as calibrated. We ask whether vendor decision training improves a model over its backbone, whether a TDM's calibration and thresholds withstand prevalence (label) shift like a fine-tuned classifier's, and how many labels a trained model needs to overtake a TDM. We compare three TDMs (Jev, Tev1-4B, Nimble-9B), their base models and trained classifiers on identical candidate-set decisions from five public datasets (on intent datasets, within the gold label's intent group) under simulated shift, with 35 paired tests frozen before the confirmatory analysis (earlier test-data checks disclosed), and estimate label-budget curves. First, decision training is not always beneficial: Nimble-9B, which differs from Qwen3.5-9B only by a LoRA adapter, is better calibrated on intent tasks but 12 Macro-F1 points worse on financial sentim
    
[^249]: 学习之前的可行性上限：无人机网格拦截的规模与时间界限

    Feasibility Ceilings Before Learning: Sizing and Timing Bounds for Drone-Mesh Interception

    [https://arxiv.org/abs/2610.20823](https://arxiv.org/abs/2610.20823)

    本文推导了无人机拦截网格在策略选定之前的性能闭式界限，并证明当智能体间距至少为2r时，平面与体积覆盖率分别存在90.69%与74.05%的固有可行性上限。

    

    由拦截器组成的网格在防御固定阵地时，必须散布在逼近目标的预测到达点周围。这是多智能体机器人中反复出现的任务：N个智能体各自覆盖一个半径为r的圆盘，散布于某一概率密度之上，使得从中抽取的一个点落入某个智能体的圆盘之内。我们给出了在任何策略被选定之前约束性能的各量的闭式表达式，并通过蒙特卡洛方法逐一验证。通过对含噪线性轨迹外推所预测的点，其不确定性来自最小二乘方差传播，再加上针对不确定到达时间的delta方法项，与仿真结果的误差在1.8%以内。密集排列网格的覆盖概率具有绝对误差不超过0.007的闭式表达式，并存在对应的体积形式。朴素的面积比会遗漏一个上限：只要智能体之间保持至少2r的间距，平面内任何网格的覆盖率都不会超过90.69%，体积内则不超过74.05%。

    arXiv:2610.20823v1 Announce Type: cross  Abstract: A mesh of interceptors defending a fixed site must spread over the predicted arrival point of an approaching object. This is a recurring task in multi-agent robotics: N agents, each covering a disc of radius r, spread over a probability density so that a point drawn from it falls within some agent's disc. We give closed forms for the quantities that bound performance before any policy is chosen, and validate each by Monte Carlo. The uncertainty of a point predicted by extrapolating a noisy linear track follows from least-squares variance propagation plus a delta-method term for the uncertain arrival time, agreeing with simulation to within 1.8%. The coverage probability of a packed mesh admits a closed form accurate to 0.007 absolute, with a volumetric analogue. A naive area ratio misses a ceiling: while agents stay at least 2r apart, no mesh exceeds 90.69% coverage in the plane or 74.05% in a volume. This ceiling belongs to the separa
    
[^250]: 基于共享空间场的大规模终身多智能体路径规划强化学习方法

    Reinforcement Learning for Large-Scale Lifelong Multi-Agent Path Finding with Shared Spatial Fields

    [https://arxiv.org/abs/2610.18186](https://arxiv.org/abs/2610.18186)

    该论文提出一种以吞吐量为奖励、端到端训练的强化学习导航策略，通过卷积网络每时间步计算一次共享空间场并由紧凑的目标条件算子导出各智能体移动偏好，在万级智能体的终身多智能体路径规划中达到与最强基线相当或更高的吞吐量，且无需重训练即可泛化部署到更大规模的未见布局，平均执行时间低于10毫秒。

    

    arXiv:2610.18186v2 公告类型：替换。摘要：终身多智能体路径规划（LMAPF）的大规模应用要求在不断变化的交通条件和多样化布局下做出计算开销极低的移动决策。我们提出了一种导航策略：由卷积网络在每个时间步为整个地图计算一次共享空间场，再由一个紧凑的目标条件算子推导出各智能体的移动偏好。整个策略以吞吐量为奖励、采用平均奖励强化学习进行端到端训练，无需规划器示范，也不依赖单独的在线路径规划器。针对结构化和非结构化两类布局，我们分别在小地图池上各训练一个策略，并可将其直接部署到更大规模、未曾见过的布局和智能体数量上而无需重新训练。在包含10,000个智能体的六种布局上，我们的策略取得了与最强基线相当或更高的吞吐量。该方法的平均执行时间低于10（原文在此处截断，推测为毫秒级）。

    arXiv:2610.18186v2 Announce Type: replace  Abstract: Large-scale applications of lifelong multi-agent path finding (LMAPF) call for computationally inexpensive movement decisions under changing traffic conditions and across varied layouts. We propose a navigation policy in which a convolutional network computes a shared spatial field over the map once per timestep and a compact goal-conditioned operator derives agents' movement preferences. The complete policy is trained end-to-end using average-reward reinforcement learning with throughput as the reward, without planner demonstrations or a separate online path planner. For structured and unstructured layouts, we train one policy per class on pools of small maps and deploy each without retraining on larger, unseen layouts and agent populations. Across six layouts with 10,000 agents, our policies achieve throughput comparable to or higher than the strongest evaluated baselines. The proposed method has an average execution time below 10 
    
[^251]: 交互即裁判

    Interaction-as-a-Judge

    [https://arxiv.org/abs/2610.17895](https://arxiv.org/abs/2610.17895)

    提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。

    

    编程智能体的任务是产出复杂的交互式应用，例如网站、桌面应用和游戏。人类在评估这些应用时会以交互方式进行，而当前自动化的LLM评估主要集中于代码分析和单元测试。这种不匹配会导致LLM评判者遗漏那些对人类用户而言一目了然的缺陷。我们研究了“交互即裁判”，这是一种由计算机操作智能体（CUA）直接与应用进行交互的评估范式。我们证明“交互即裁判”能够同时改进交互式应用的评估与生成。为此，我们收集并发布了两个互补的数据集：包含人类偏好及对应智能体操作轨迹的6,858对应用，以及带有经人类验证的需求标签的222个应用。相较于标准的非交互式方法，我们发现“交互即裁判”在预测h……（原文摘要在此处截断）

    arXiv:2610.17895v2 Announce Type: cross  Abstract: Coding agents are tasked with producing sophisticated interactive applications such as websites, desktop applications, and games. While humans assess these applications interactively, current automated LLM evaluations primarily focus on code analysis and unit tests. This mismatch can cause LLM judges to miss failures that are immediately apparent to human users. We study Interaction-as-a-Judge, an evaluation paradigm where computer-use agents (CUAs) directly interact with an application. We demonstrate that Interaction-as-a-Judge can improve both evaluation and generation of interactive applications. To this end, we collect and release two complementary datasets: $6,858$ pairs of applications with human preferences and corresponding agent traces, and $222$ applications with human-verified requirement labels. Relative to standard non-interactive approaches, we find that Interaction-as-a-Judge improves evaluation in terms of predicting h
    
[^252]: 一个检查点，多次运行：微调模型的“配方卡”

    One Checkpoint, Many Runs: Recipe Cards for Fine-Tuned Models

    [https://arxiv.org/abs/2610.16825](https://arxiv.org/abs/2610.16825)

    论文提出“模型配方”概念，主张模型卡应披露一次运行达不到性能门槛的失败概率，并通过对 300 张模型卡的调查和 2,124 次微调运行证明，仅以均值、中位数或 bootstrap 区间作为验收标准常常会误判通过失败的运行。

    

    微调后的语言模型通常附带模型卡共享，模型卡对训练设置的描述十分详尽，甚至细化到默认随机种子，但只报告单一已发布检查点的分数。这类设置经常被重复训练：微调服务为每个客户重新训练，机构在每次更新时重新训练，用户也会在自己的数据上重新训练，因此实际部署的模型来自该设置的一次全新运行。有些运行会失败，而失败的运行可能未被察觉，并最终交付到模型所服务的用户手中。我们将这样的设置称为“模型配方”，并主张其文档应当说明该配方的一次运行有多大概率达不到所需的性能门槛。在对 Hugging Face 上 300 个微调语言模型的模型卡进行的调查中，没有任何一张报告了这种失败概率，仅有三张报告了不止一次训练运行的结果。在 2,124 次微调运行中，我们发现将均值、中位数或 bootstrap 区间解读为验收标准时，往往会让实际未达标的运行通过验收。

    arXiv:2610.16825v2 Announce Type: replace  Abstract: Fine-tuned language models are shared with model cards that describe the training setup in detail, often down to the default random seed, but report the score of a single released checkpoint. Such setups are retrained routinely, by fine-tuning services for each customer, by organizations at each update, and by users on their own data, so the model that is deployed comes from a new run of the setup. Some runs fail, and a failed run can go unnoticed and reach the people the model serves. We call such a setup a model recipe and argue that its documentation should state how often a run of the recipe misses a required performance bar. In a survey of the model cards of 300 fine-tuned language models on Hugging Face, none reports this failure probability, and only three report results from more than one training run. Across 2,124 fine-tuning runs, we find that reading the mean, the median, or a bootstrap interval as an approval often approv
    
[^253]: LoongSage：面向前沿大语言模型的生产级、高效智能体强化学习框架（附经验证的训练方案）

    LoongSage: A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes

    [https://arxiv.org/abs/2610.16352](https://arxiv.org/abs/2610.16352)

    LoongSage是一个智能体原生、完全异步的生产级强化学习框架，基于Megatron与SGLang构建且核心仅约1.4万行代码，通过统一编排白盒与黑盒智能体的AgentFlow范式，显著降低了前沿大语言模型智能体强化学习的采用、使用与调试开销。

    

    强化学习（RL）已成为大语言模型（LLM）后训练流程中的关键阶段。随着大语言模型能力的前沿从静态推理转向智能体任务——其特征包括外部工具调用、多轮交互和长程决策——现有主流强化学习框架变得日益笨重，增加了采用、使用和调试的开销。为应对这些挑战，我们提出了LoongSage，一个智能体原生、完全异步的强化学习框架，其核心代码仅约1.4万行。LoongSage基于Megatron实现分布式训练，基于SGLang实现高吞吐量推理，将编排逻辑集中在单一控制器进程中，并将所有驻留GPU的组件封装为Ray actor。它引入了AgentFlow这一统一执行范式，可同时编排白盒与黑盒智能体。对于异步执行，（原文摘要在此处截断，后续内容缺失）

    arXiv:2610.16352v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has emerged as a critical phase in the post-training pipeline of large language models (LLMs). As the frontier of LLM capabilities shifts from static reasoning to agentic tasks--characterized by external tool invocation, multi-turn interactions, and long-horizon decision-making--existing mainstream RL frameworks have grown increasingly unwieldy, raising the overhead of adoption, usage, and debugging. To address these challenges, we present LoongSage, an agent-native, fully asynchronous RL framework with a lightweight core of roughly 14K lines of code. Built upon Megatron for distributed training and SGLang for high-throughput inference, LoongSage centralizes orchestration within a single controller process and encapsulates all GPU-resident components as Ray actors. It introduces AgentFlow, a unified execution paradigm that orchestrates both white-box and black-box agents. For asynchronous execution, 
    
[^254]: 一种不变切线角描述子与用于二维碎片邻接预测的带状U-Net

    An Invariant Tangent-Angle Descriptor and a Band U-Net for 2D Fragment Adjacency Prediction

    [https://arxiv.org/abs/2610.09459](https://arxiv.org/abs/2610.09459)

    该论文提出了一种在构造上对碎片旋转和轮廓起始点选择均不变的切线角描述子来替代原有的局部评分，并用带状U-Net替换最终分类器，从而改进了基于轮廓的二维碎片邻接预测的两阶段架构。

    

    本文研究基于轮廓的二维碎片对之间邻接关系的预测问题。我们改进了Beaulac论文中提出的两阶段架构，该架构使用旋转等变的孪生卷积神经网络对两个碎片两条轮廓上的局部图像窗口对进行评分。评分被收集在一个邻接矩阵中，随后由ResNet检测其中揭示两个碎片邻接关系的部分反对角带。在本工作中，我们保留了整体流程，但将局部评分替换为轮廓窗口切线角轮廓的比较，使其在构造上对碎片旋转具有不变性，并且与所选的轮廓起始点无关。这种局部比较既可以采用无需训练的似然比方法，也可以采用在对应点上训练的小型一维卷积模型。此外，我们还将最终的分类器替换为一个带状U-Net，用于对带进行分割和分类。

    arXiv:2610.09459v1 Announce Type: cross  Abstract: This paper addresses the prediction of adjacency between pairs of 2D fragments based on their contours. We improved the two-stage architecture proposed in Beaulac's thesis, in which a rotation-equivariant Siamese convolutional neural network scores pairs of local image windows along the two contours of two fragments. The scores are gathered in an adjacency matrix in which a ResNet detects the partial anti-diagonal band that reveals the adjacency of two fragments. In the current work, we keep the pipeline and replace the local score by a comparison of tangent-angle profiles of contour windows, making it, by construction, invariant to fragment rotation and agnostic to the selected contour-starting point. These adaptations may be either a training-free likelihood ratio or a small one-dimensional convolutional model trained on corresponding points. We also replaced the final classifier by a band U-Net that segments the band and classifies 
    
[^255]: CADFather：通过协同工具调用实现自主CAD重建

    CADFather: Autonomous CAD Reconstruction through Coordinated Tool Use

    [https://arxiv.org/abs/2610.09127](https://arxiv.org/abs/2610.09127)

    CADFather是一个自主智能体系统，通过视觉语言助手协调学习型/算法型提案工具与数值优化等多种互补工具，从3D网格中重建出可编辑的参数化CAD程序。

    

    从3D形状重建可编辑的CAD模型仍然是一项具有挑战性的工程任务。现有方法可以提出CAD操作，但没有任何单一的提案来源能够在不同的零件几何形状和重建阶段都同样有效。我们提出了CADFather，这是一个自主的智能体系统，通过协调互补的工具从3D网格中恢复参数化的CAD程序。一个视觉语言助手会检查目标形状和中间重建结果的渲染图，然后决定扩展哪些候选CAD程序、调用哪些工具、生成多少个提案以及何时结束。学习型和算法型工具负责提出CAD操作，而数值优化则用于细化现有程序的参数。新提出或细化后的程序会被执行和评估，从而为后续决策提供反馈。该智能体为每个目标零件维护多个备选的候选程序，并保留最佳的有效重建结果。

    arXiv:2610.09127v1 Announce Type: cross  Abstract: Reconstructing an editable CAD model from a 3D shape remains a challenging engineering task. Existing methods can propose CAD operations, but no single source of proposals works equally well across different part geometries and stages of reconstruction. We introduce CADFather, an autonomous agentic system that coordinates complementary tools to recover parametric CAD programs from 3D meshes. A vision-language assistant inspects renders of the target and intermediate reconstructions, then decides which candidate CAD programs to extend, which tools to invoke, how many proposals to generate, and when to finish. Learned and algorithmic tools propose CAD operations, while numerical optimization refines the parameters of existing programs. Proposed or refined programs are executed and evaluated to provide feedback for subsequent decisions. The agent maintains alternative candidate programs for each target part and preserves the best valid re
    
[^256]: 考虑上下文：世界模型何时需要上下文编码器

    Considering Context: When World Models Need Context Encoders

    [https://arxiv.org/abs/2610.06651](https://arxiv.org/abs/2610.06651)

    该论文提出“预测充分性”这一形式化指标，用于量化上下文对基于模型的强化学习预测的真实增益，并证明世界模型是否需要上下文编码器取决于环境先验而非MDP类别，据此建立了上下文感知算法的分类框架。

    

    基于模型的强化学习中的泛化方法通常假设智能体无法从自身经验中恢复支配环境动力学的潜在上下文，因而需要从外部提供该上下文。我们通过“预测充分性”对这一假设进行形式化并加以检验：该指标量化了在智能体所诱导的访问分布下，获取上下文对下一步预测带来的增益，并将该增益分解为可从历史中恢复的部分与需要真实上下文的残差部分，这两者均区别于有限模型的固有不足。我们依据上下文感知算法的条件集所能针对的预测风险对其进行分类，并在识别难度递增的一系列环境中证明，预测改进空间并不由MDP类别决定。同一任务在不同先验下，可能在一个设置中留有预测改进空间，而在另一个设置中改进空间与零无从区分，此时智能体的行为……（原文摘要在此处截断）

    arXiv:2610.06651v2 Announce Type: replace  Abstract: Methods for generalization in model-based reinforcement learning typically assume that an agent cannot recover the latent context governing the environment dynamics from its own experience, and therefore supplies it externally. We formalize and test this assumption with \emph{predictive sufficiency}, which quantifies what access to the context adds to next-step prediction under the visitation distribution an agent induces, and separates that quantity into a history-recoverable part and a residual requiring the true context, both distinct from a finite model's deficit. We classify context-aware algorithms by the predictive risk their conditioning set can target and demonstrate across environments of increasing identification difficulty that the headroom does not follow the MDP class. The same task under different priors leaves predictive headroom in one setting and nothing distinguishable from zero in another, where the agent's behavi
    
[^257]: 超越分布内性能保持：通过面向脆弱性的调优恢复量化视觉-语言-动作模型的泛化能力

    Beyond In-Distribution Preservation: Recovering Generalization in Quantized VLAs via Vulnerability-Oriented Tuning

    [https://arxiv.org/abs/2610.05745](https://arxiv.org/abs/2610.05745)

    该研究揭示量化VLA模型在环境扰动下虽保持分布内性能却变得脆弱，发现动作差异集中于少数脆弱状态且教师指导在高差异状态提升泛化、在低差异状态反而损害泛化，据此提出面向脆弱状态的选择性调优方法以恢复量化模型的泛化能力。

    

    后训练量化已被证明能在标准评估条件下保持VLA（视觉-语言-动作模型）的性能，但其在多大程度上保留了全精度模型的鲁棒性和泛化能力仍缺乏深入研究。在本研究中，我们系统研究了量化后VLA策略在环境扰动下的鲁棒性与泛化能力。实证结果表明，量化策略尽管保持了相当的分布内性能，但在面对细微的环境变化时可能变得脆弱。我们进一步观察到，动作差异集中于少量回放状态，而教师指导的效果因差异程度而异：在高差异状态下能提升泛化能力，但在低差异状态下反而可能损害泛化能力。这些发现表明，有效的量化后恢复需要对脆弱状态进行选择性干预，而非对学生模型进行全局蒸馏。

    arXiv:2610.05745v2 Announce Type: replace  Abstract: Post-training quantization has been shown to preserve VLA performance under standard evaluation conditions, but whether it preserves the full-precision model's robustness and generalization remains underexplored. In this study, we systematically study the robustness and generalization of post-quantized VLA policies under environmental disturbances. Empirical results show that quantized policies can become fragile to subtle environmental variations despite retaining comparable in-distribution performance. We further observe that action discrepancies are concentrated in a small subset of rollout states, while teacher guidance has opposite effects depending on discrepancy: it improves generalization at high-discrepancy states but can degrade it at low-discrepancy states. These findings reveal that effective post-quantization recovery requires selectively intervening on vulnerable states rather than globally distilling the student. We th
    
[^258]: 非交换AM-GM不等式何时成立

    When the noncommutative AM-GM inequality holds

    [https://arxiv.org/abs/2610.04874](https://arxiv.org/abs/2610.04874)

    本文证明当 $n\ge 2\lceil m/2 \rceil^2$ 时非交换AM-GM不等式成立，其核心方法是通过构造基于布尔立方体上切比雪夫节点的顶点测度，将问题转化为求积估计。

    

    在本短文中，我们证明了当 $n\ge 2\lceil m/2 \rceil^2$ 时，非交换AM-GM不等式成立。该研究的动机来源于[De Sa, Random reshuffling is not always better, NeurIPS2020]中构造的反例。证明过程构造了一个基于布尔立方体上切比雪夫节点的顶点测度，用以提取不同的指标。主要困难在于该测度在非整数节点上不是正的。关键技术来自[Grigoriev, Complexity of Positivstellensatz proofs for the knapsack, Computational Complexity (2001)]，并最终将问题转化为一个求积估计。

    arXiv:2610.04874v2 Announce Type: replace-cross  Abstract: In this note, we prove that the noncommutative AM-GM inequality holds if $n\ge 2\lceil m/2 \rceil^2$. The motivation comes from counterexamples constructed in [De Sa, Random reshuffling is not always better, NeurIPS2020]. The proof constructs a vertex measure based on the Chebyshev nodes on the Boolean cube to extract the distinct indices. The main difficulty is that the measure is not positive on non-integer nodes. The key technique comes from [Grigoriev, Complexity of Positivstellensatz proofs for the knapsack, Computational Complexity (2001)] and eventually transforms the problem into a quadrature estimate.
    
[^259]: DV-Lens：揭示语言模型参数的功能组织

    DV-Lens: Revealing the Functional Organization of Language Model Parameters

    [https://arxiv.org/abs/2610.04489](https://arxiv.org/abs/2610.04489)

    提出DV-Lens参数级可解释性框架，通过估计模块特定的雅可比矩阵将语言模型参数投影到下游词汇空间，从而揭示参数的功能组织及其与模型能力的关系。

    

    理解参数的功能有助于阐明大语言模型（LLMs）的内部机制。然而，如何将来自不同模块的参数与可验证的输出效应联系起来，并进一步刻画其功能组织与模型能力之间的关系，仍有待探索。为此，我们提出了下游词汇透镜，这是一个参数级可解释性框架，将原生的参数方向与其下游词汇响应联系起来。具体而言，我们首先在参考提示集上估计注意力查询、键、值和输出投影以及前馈网络的模块特定下游雅可比矩阵。其次，我们利用这些映射将原生参数列投影到最终词汇空间，获得表征其平均局部输出响应的带符号读数。第三，我们根据词汇响应对参数列进行分组……

    arXiv:2610.04489v2 Announce Type: replace  Abstract: Understanding parameter functions helps elucidate the internal mechanisms of large language models (LLMs). However, how to connect parameters from different modules to verifiable output effects and further characterize the relationship between their functional organization and model capability remains to be explored. To this end, we introduce the downstream vocabulary lens (DV-Lens), a parameter-level interpretability framework that links native parameter directions to their downstream vocabulary responses. Specifically, we first estimate module-specific downstream Jacobians over a reference prompt set for attention query, key, value, and output (Q/K/V/O) projections and feed-forward networks (FFNs). Second, we use these mappings to project native parameter columns into the final vocabulary space, obtaining signed readouts that characterize their average local output responses. Third, we group parameter columns by their vocabulary re
    
[^260]: 框架到底换来了什么？主要只是更多的 Token

    What Does a Harness Buy? Tokens, Mostly

    [https://arxiv.org/abs/2610.04433](https://arxiv.org/abs/2610.04433)

    该研究通过对七个模型在三个生产级框架和多个基准上的重复对照实验发现，没有任何框架能稳定提升性能，更换框架带来的差异仅有 2 到 5 个百分点、效果与重新运行一次相当，框架带来的分数提升主要源于更多的 token 消耗。

    

    编码智能体是运行在框架中的语言模型：框架包括系统提示词、工具以及让聊天模型能够在代码仓库中工作的上下文管理。这些智能体的大部分工程投入都集中在框架上，厂商和排行榜也将分数提升归功于框架。早期的研究固定模型并更换框架，但并未测量这种更换对通过率的改变是否大于用同一框架重新运行一遍。我们在 SWE-bench Verified 以及另外两个不同类型的基准上，用三个生产级框架——Claude Code、mini-SWE-agent 和 OpenCode——运行七个模型，并在完全相同的条件下重新运行。结果显示，没有任何一个框架能在所有模型上保持一致领先。在数据能够界定两个框架差异的地方，该差异的上界为 2 到 5 个百分点，而更换框架所改变的任务结果数量与重新运行一次大致相当。例外情况是特定模型与框架配对中出现的框架效应，其影响方向随模型而变化：Ope

    arXiv:2610.04433v2 Announce Type: replace  Abstract: A coding agent is a language model inside a harness: the system prompt, the tools, and the context management that let a chat model work in a repository. Most of the engineering on these agents goes into the harness, and vendors and leaderboards credit it with score gains. Earlier studies have fixed the model and swapped the harness, but have not measured whether the swap changes the pass rate more than running the same harness again. We run seven models through three production harnesses, Claude Code, mini-SWE-agent, and OpenCode, on SWE-bench Verified and two benchmarks of other kinds, and rerun them unchanged. No harness leads consistently across models. Where the data can bound the difference between two harnesses, the bound is 2 to 5 points, and a harness swap changes about as many task outcomes as a rerun. The exceptions are harness effects in particular pairings of model and harness, whose direction changes with the model: Ope
    
[^261]: PoCoFL：策略合规的联邦学习

    PoCoFL: POlicy-COmpliant Federated Learning

    [https://arxiv.org/abs/2610.03650](https://arxiv.org/abs/2610.03650)

    提出了PoCoFL框架，通过将联邦学习类型、策略语义与密码学实现三者解耦，并借助承诺和非交互式零知识证明来验证客户端贡献的策略合规性，从而实现适用于多种拓扑、角色和聚合方式的通用可验证联邦学习。

    

    联邦学习（FL）是一种面向隐私的学习范式，能够在训练数据保留在参与客户端本地的前提下进行协作式模型训练。然而，它既不能保证客户端提交的贡献符合策略要求，也不能保证聚合器正确处理已接受的贡献。现有的可验证联邦学习系统将验证规则定制于特定的联邦学习设置、学习工作流和密码学构造，这限制了它们在不同网络拓扑、参与者角色和聚合语义下的适用性。在本文中，我们提出了PoCoFL，一个策略合规的联邦学习框架，它将三个方面相互分离：（i）联邦学习类型，（ii）策略语义，（iii）密码学实现。我们提供了一种形式化方法，将客户端和聚合的要求表示为依赖于策略的关系。客户端使用承诺机制和非交互式零知识证明来证明其贡献的合规性……（摘要在此处被截断）

    arXiv:2610.03650v1 Announce Type: cross  Abstract: Federated Learning (FL) is a privacy-oriented learning paradigm that enables collaborative model training while keeping training data local to participating clients. However, it does not guarantee that clients submit policy-compliant contributions or that aggregators process admitted contributions correctly. Existing verifiable FL systems tailor validation rules to specific FL settings, learning workflows, and cryptographic constructions, limiting their applicability across network topologies, participant roles, and aggregation semantics. In this paper, we present PoCoFL, a policy-compliant federated learning framework that separates three aspects: (i) FL type, (ii) policy semantics, and (iii) cryptographic realisation. We provide a formalisation that captures client and aggregation requirements as policy-dependent relations. Clients prove compliance of their contributions using commitments and non-interactive zero-knowledge proofs, wh
    
[^262]: 何时准确率才是证据？泛化、验证与信息融合的统一理论

    When Is Accuracy Evidence? A Unified Theory of Generalisation, Validation, and Information Fusion

    [https://arxiv.org/abs/2610.03465](https://arxiv.org/abs/2610.03465)

    该论文提出一个统一的指数框架，将交叉验证准确率转化为真实风险的保守上界，并通过有效折数Keff证明：当各折数据强相关时，单纯增大交叉验证折数K并不能增强统计证据。

    

    K折交叉验证（CV）被广泛用作样本外性能的证据，然而在异构数据下，各折既非独立实验，也非信息量均等。交叉上界验证（CUBV）用真实风险的保守上界取代逐点的CV准确率。本文通过单一的指数框架对CUBV进行了推广，其中泛化间隙的矩生成函数由累积量包络gamma(lambda)所控制。由此得到一族风险界，涵盖Hoeffding界、Bernstein界、依赖感知、PAC-Bayesian以及异构数据源融合等多种情形。对于K折CV，各折间隙之间的相关性通过联合次高斯代理矩阵来建模。在等相关条件下，得到有效折数Keff = K/[1+(K-1)rho]，表明当各折之间存在强相关性时，增大K并不一定能增加统计证据。该框架还被用于……（原文摘要在此处截断）

    arXiv:2610.03465v1 Announce Type: cross  Abstract: K-fold cross-validation (CV) is widely used as evidence of out-of-sample performance, although folds are neither independent experiments nor equally informative under heterogeneous data. Cross Upper-Bound Validation (CUBV) replaces point-wise CV accuracy by conservative upper bounds on true risk. Here we generalise CUBV through a single exponential framework in which the moment-generating function of the generalisation gap is controlled by a cumulant envelope gamma(lambda). This yields a family of risk bounds covering Hoeffding-, Bernstein-, dependency-aware, PAC-Bayesian, and heterogeneous source-fusion settings. For K-fold CV, dependence between fold-wise gaps is modelled through a joint sub-Gaussian proxy matrix. Under equicorrelation, this gives an effective number of folds, Keff = K/[1+(K-1)rho], showing that increasing K does not necessarily increase statistical evidence when folds are strongly dependent. The framework is also ex
    
[^263]: CriticHack：在机器人策略优化下评估视觉奖励

    CriticHack: Evaluating Visual Rewards Under Robot Policy Optimization

    [https://arxiv.org/abs/2610.02527](https://arxiv.org/abs/2610.02527)

    该论文揭示，用学习型视觉奖励模型优化机器人策略时，奖励分数与任务成功率可能同时上升、看似健康，但实际上会显著放大“作用于错误物体”的隐蔽失败，而这一现象在使用模拟器真实任务完成信号训练时并不会出现。

    

    学习得到的视觉奖励模型越来越多地被用于优化机器人策略，然而奖励模型可能会对作用于错误物体的执行给出与真正完成任务同样高的评分。我们证明，针对这样的奖励进行优化，可能在奖励值与任务成功率双双上升的同时放大这类“错误物体”失败，使得从业者通常会监测的信号看起来一切正常。我们在一个抽屉任务上，针对 Robometer 对扩散策略去噪器的每一个参数进行微调。从一个没有任何奖励先验的监督策略出发，五次训练运行在 512 个评估种子上将任务成功率提高了 10.2 个百分点，同时使错误物体失败增加了 10.9 个百分点；而使用模拟器任务完成信号训练的五次运行则在提高成功率的同时没有放大错误物体失败（差异为 9.2 个百分点，95% 置信区间为 5.6 至 13.0）。这种放大现象在一个先前已针对学习奖励优化过的策略上同样会出现，在该策略原生的……（摘要在此处截断）

    arXiv:2610.02527v1 Announce Type: cross  Abstract: Learned visual reward models are increasingly used to optimize robot policies, yet a reward model can score an execution that acts on the wrong object as highly as one that completes the task. We show that optimizing such a reward can amplify these wrong-object failures while reward and task success both rise, so the signals a practitioner would normally monitor look healthy. We fine-tune every denoiser parameter of a diffusion policy against Robometer on a drawer task. Starting from a supervised policy with no prior reward exposure, five training runs raise task success by 10.2 percentage points and wrong-object failures by 10.9 points on 512 evaluation seeds, whereas five runs trained on the simulator's task-completion signal raise success without amplifying wrong-object failures (difference 9.2 points, 95% CI 5.6 to 13.0). The amplification recurs from a policy previously optimized against learned rewards, under the policy's native 
    
[^264]: IQS-BO：面向贝叶斯优化的上下文内查询选择

    IQS-BO: In-Context Query Selection for Bayesian Optimisation

    [https://arxiv.org/abs/2610.01269](https://arxiv.org/abs/2610.01269)

    本文提出IQS-BO，一种通过在合成先验上进行监督学习来学习贝叶斯优化查询决策的PFN，可在单次前向传递中完成查询选择，从而避免了代理模型重新拟合和采集函数的数值最大化。

    

    贝叶斯优化（BO）是优化昂贵黑盒函数的强大框架，但通常需要在每次评估步骤中重新拟合代理模型并最大化采集函数。基于先验数据拟合网络（PFNs）的上下文内方法通过在从合成先验中抽取的函数上预训练Transformer来分摊部分成本。PFNs4BO分摊了代理模型的计算成本，但仍依赖于数值最大化的采集函数；而FIBO通过从学习到的密度中采样优化器位置，完全在上下文内执行BO，这固定了决策规则且不使用代理模型。学习型采集函数通过训练好的网络对有限候选集进行评分，但由于缺乏查询的标签，它们只能在先前解决的任务上通过强化学习来学习评分。我们提出了IQS-BO，一个通过在合成先验上进行监督学习来学习查询决策的PFN。在单次前向传递中，IQS-BO……

    arXiv:2610.01269v1 Announce Type: cross  Abstract: Bayesian Optimisation (BO) is a powerful framework for the optimisation of expensive black-box functions, but typically requires refitting a surrogate and maximising an acquisition function at every evaluation step. In-context approaches based on Prior-data Fitted Networks (PFNs) amortise part of this cost by pre-training transformers on functions drawn from synthetic priors. PFNs4BO amortises the surrogate but still relies on a numerically maximised acquisition function, while FIBO performs BO fully in-context by sampling optimiser locations from a learned density, which fixes the decision rule and admits no surrogate. Learned acquisition functions score a finite candidate set with a trained network, but, lacking a label for the query, learn the score by reinforcement learning on previously solved tasks. We propose IQS-BO, a PFN that learns the query decision by supervised learning on synthetic priors. In a single forward pass, IQS-BO
    
[^265]: 基于深度学习的三混合多用户MIMO预编码：电磁可重构天线的福音

    Deep Learning-Based Tri-Hybrid Multi-User MIMO Precoding: The Blessing of EM-Reconfigurable Antennas

    [https://arxiv.org/abs/2609.39167](https://arxiv.org/abs/2609.39167)

    本文提出基于Conformer神经网络架构的三混合预编码网络Tri-PNet，通过将电磁可重构天线引入的电磁域预编码与模数混合预编码进行联合设计，显著提升了宽带多用户MIMO-OFDM系统的频谱效率。

    

    电磁（EM）可重构天线为每个天线单元提供多种候选辐射方向图，从而引入了额外的电磁域自由度。将辐射方向图可重构性（实现为电磁域预编码）与传统的模数混合预编码相结合，产生了三混合多输入多输出（MIMO）预编码，这可以显著提升宽带多用户MIMO正交频分复用（OFDM）系统的频谱效率。然而，电磁、模拟和数字预编码的联合设计仍然具有挑战性。为了解决这一挑战，我们提出了一种基于Conformer的三混合预编码网络，Conformer是一种新兴的神经网络架构，它结合了卷积神经网络的局部建模能力与Transformer的全局依赖建模能力。此外，两种代表性的辐射方向图模式，即非规则模式和第3种……（摘要到此截断）

    arXiv:2609.39167v1 Announce Type: cross  Abstract: Electromagnetic (EM)-reconfigurable antennas provide multiple candidate radiation patterns per element, thereby introducing an additional EM-domain degree of freedom. Integrating radiation-pattern reconfigurability, realized as EM-domain precoding, with conventional hybrid analog-digital precoding yields tri-hybrid multiple-input multiple-output (MIMO) precoding, which can substantially improve the spectral efficiency of wideband multi-user MIMO orthogonal frequency-division multiplexing (OFDM) systems. However, the joint design of EM, analog, and digital precoding remains challenging. To address this challenge, we propose a tri-hybrid precoding network (Tri-PNet) based on Conformer, an emerging neural architecture that combines the local modeling strength of convolutional neural networks with the global dependency modeling of Transformers. Furthermore, two representative radiation-pattern modes, i.e., the non-regular mode and the 3rd 
    
[^266]: SANTA++：通过代表性键进行注意力采样

    SANTA++: Sampling Attention through Representative Keys

    [https://arxiv.org/abs/2609.35629](https://arxiv.org/abs/2609.35629)

    SANTA++ 是一种无需训练的随机注意力方法，通过将 KV 缓存分组并用代表性键进行重要性采样，仅访问 28-31% 的 KV 缓存即可保留 95-98% 的稠密注意力性能，显著降低长上下文推理的内存开销。

    

    注意力通常集中在上下文中的一小部分标记上，但哪些标记重要会随查询的不同而变化。为了利用这种动态变化的结构，我们提出了 SANTA++，这是一种无需训练的随机注意力方法，它使用代表性键进行内存高效的选择，而无需扫描整个键值（KV）缓存。缓存的键被组织成多个“团队”（分组），查询对每个团队中的一个代表性键进行评分，以决定采样哪些团队。我们在被采样的团队内计算精确的注意力分数，并根据每个团队被纳入概率的倒数重新加权其贡献。这种重要性采样修正可以估计整个缓存上的注意力，采样预算让我们可以在内存读取量和精度之间进行权衡。在 32K 上下文长度的 Qwen2.5-7B-Instruct 上，S=1024 的连续 SANTA++ 在 LongBench v2、HELMET RAG 和 RULER 上仅使用 28-31% 的逻辑 KV 访问量，就保留了 95-98% 的稠密聚合分数。

    arXiv:2609.35629v2 Announce Type: replace-cross  Abstract: Attention often concentrates on a small subset of tokens in the context, but which subset matters changes from one query to the next. To exploit this changing structure, we introduce SANTA++, a training-free stochastic attention method that uses representative keys for memory-efficient selection without scanning the entire key-value (KV) cache. Cached keys are organized into teams, and the query scores one representative from each team to decide which teams to sample. We compute exact attention scores within the sampled teams and reweight each team's contribution by the inverse of its inclusion probability. This importance sampling correction estimates attention over the full cache, with a sampling budget that lets us trade memory reads for accuracy. With Qwen2.5-7B-Instruct at 32K context, contiguous SANTA++ at S=1024 retains 95-98% of dense aggregate scores on LongBench v2, HELMET RAG, and RULER using 28-31% logical KV access
    
[^267]: 面向一次训练、多预算导出推理的弹性选择性谱混合模型

    Elastic Selective Spectral Hybrids for Train-Once, Export-Many Budgeted Inference

    [https://arxiv.org/abs/2609.32486](https://arxiv.org/abs/2609.32486)

    提出弹性选择性谱混合模型（ESSH），通过输入相关的衰减与门控机制赋予谱状态空间模型选择性记忆能力，并结合滑动窗口注意力，实现一次训练即可导出适配多种计算预算的高效语言模型。

    

    在不同的计算和延迟预算下部署语言模型，需要具有不同质量-成本权衡的紧凑模型。为此，弹性谱状态空间模型提供了有序且在时间上可分解的通道，可以被截断，但它们与线性时不变滤波器相关联，无法随着上下文的演变选择性地保留相关的历史信息或遗忘无关信息。为了解决这一问题，我们提出了弹性选择性谱混合模型（ESSH），它使用拟合的阻尼旋转模式将每个Hankel谱通道实现为独立的循环单元。它还具有依赖于输入的衰减和写入/读取门控，使时间信息保留和状态更新依赖于输入，同时保留了通道级截断能力和结构化循环以实现高效执行。ESSH将这些选择性谱混合器与滑动窗口注意力相结合，并联合训练多个……

    arXiv:2609.32486v2 Announce Type: replace  Abstract: Deploying a language model under different computing and latency budgets calls for compact models with different quality-cost trade-offs. Towards this end, elastic spectral state space models provide ordered and temporally decomposed channels that can be truncated, but they are associated with linear time-invariant filters that cannot selectively preserve relevant past information or forget irrelevant information as the context evolves. To address this, we introduce the Elastic Selective Spectral Hybrid (ESSH), which realizes each Hankel spectral channel as an independent recurrent unit using fitted damped rotation modes. It also features an input-dependent decay and write/read gates that make temporal retention and state update input-dependent while preserving channel-wise truncation and a structured recurrence for efficient execution. ESSH combines these selective spectral mixers with sliding-window attention and jointly trains mul
    
[^268]: 离散度与规模：决定测试时预算分配是否划算的因素

    Spread and Scale: What Determines Whether Test-Time Budget Allocation Pays

    [https://arxiv.org/abs/2609.27917](https://arxiv.org/abs/2609.27917)

    本文通过预先注册的验证性实验发现，工作负载中实例难度的离散程度是决定测试时预算重新分配是否划算的关键属性，并且即使计入分配策略自身消耗的预算成本，该策略依然可能带来收益。

    

    神经组合优化求解器为每个实例生成多个候选解并报告其中找到的最优解，无论实例难度如何，都为每个实例分配相同的采样预算。一项配套研究表明，将固定预算向更难的实例重新分配可以提升解的质量，但这种改进的标准度量方式存在偏差：在相同数据上既决定分配又评估该分配，即使实际没有收益，也可能制造出表面上的收益。这就留下了两个未解决的问题：工作负载的什么特性决定了重新分配是否值得做，以及一个花费部分预算来决定如何分配其余预算的策略，在计入该成本之后是否仍然划算。本文通过预先注册的验证性实验——在数据收集之前就固定了分析方法和判定标准——在三个独立训练的求解器以及两种在训练分布上构造更难工作负载的方式上回答了这两个问题。

    arXiv:2609.27917v1 Announce Type: new  Abstract: Neural combinatorial optimization solvers generate many candidate solutions per instance and report the best one found, using the same sample budget for every instance regardless of difficulty. A companion study showed that reallocating a fixed budget toward harder instances can improve solution quality, but that the standard way of measuring this improvement is biased: deciding an allocation and evaluating it on the same data can manufacture an apparent gain even when none exists. This left open what property of a workload determines whether reallocation is worth doing, and whether a policy that spends part of the budget to decide how to allocate the rest still pays once that cost is counted.   This paper answers both questions through pre-registered confirmatory experiments -- analysis and verdict criteria fixed before data collection -- across three independently trained solvers and two ways of constructing harder workloads on the tra
    
[^269]: 蒸馏你所信任的：可靠性感知的多教师在线策略蒸馏

    Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2609.23697](https://arxiv.org/abs/2609.23697)

    提出TrustMOPD方法，以专家模型RL训练前后相对于共享参考模型的位移作为token级可靠性代理，实现无标签的多教师加权蒸馏监督分配，将学生模型性能恢复率从54.4%大幅提升至91.5%。

    

    多教师在线策略蒸馏允许学生在自身生成的轨迹上从互补的专家模型中学习。然而，基于领域路由的方法在每个样本中仅选择一个教师，并在整个响应生成过程中保持固定。这种设计既依赖于混合训练语料库中通常缺失的标签，也无法在轨迹内部所需专业知识发生变化时调整教师选择。我们提出了TrustMOPD，它用无标签的、token级别的监督分配取代样本级教师选择。在每个学生生成的前缀处，TrustMOPD将每个专家模型经强化学习（RL）后相对于共享的RL前参考模型的位移作为局部可靠性的代理指标，跨教师对这些分数进行校准，并构建加权的蒸馏目标。在数学、代码和指令遵循任务上，TrustMOPD优于最强的无标签基线，将恢复率从54.4%提升至91.5%。

    arXiv:2609.23697v1 Announce Type: new  Abstract: Multi-teacher on-policy distillation allows a student to learn from complementary specialists on its own trajectories. Domain-routed approaches, however, select one teacher per example and keep it fixed throughout the response. This design both depends on labels that mixed training corpora often lack and cannot adapt teacher selection when the expertise required changes within a trajectory. We propose \textbf{TrustMOPD}, which replaces example-level teacher selection with label-free, token-level supervision allocation. At each student-generated prefix, TrustMOPD uses each specialist's RL-induced displacement from a shared pre-RL reference as a proxy for local reliability, calibrates these scores across teachers, and constructs a weighted distillation target. Across mathematics, code, and instruction following, TrustMOPD outperforms the strongest label-free baseline, increasing the recovery ratio from $54.4\%$ to $91.5\%$ on \textsc{Singl
    
[^270]: 困惑度代价低估了激活量化所破坏的东西

    Perplexity Cost Understates What Activation Quantisation Breaks

    [https://arxiv.org/abs/2609.23125](https://arxiv.org/abs/2609.23125)

    该论文发现困惑度作为聚合指标会低估激活量化的真实损害——即使困惑度仅上升1.2至1.5倍，归纳能力几乎无损（保持0.959）而检索能力已骤降至0.554，且这种结构性差距并非仅由量化误差幅度造成。

    

    激活量化通常用一个聚合指标——困惑度来评估，该指标是对模型预测的每个token取平均。我们追问这个平均值是否能识别量化器损害了哪些计算。事实证明，困惑度是一个可靠的聚合信号：在来自四个系列的12个模型和780次模型内比较中，困惑度更偏好的方案在除2.1%和4.0%的情况外，都保留了更多的归纳能力和更多的检索能力。但是，当困惑度仅上升1.2到1.5倍时，归纳能力仍保持其完好精度的0.959，而检索能力已降至0.554，这一差距是聚合数字所无法揭示的。这个差距具有结构性，而不仅仅是大小：相同每通道幅度的匹配高斯噪声对照在很大程度上保持完好，而仅随机化量化误差的符号（所有幅度保持固定）也几乎无害，因此仅凭幅度无法解释这种损害。

    arXiv:2609.23125v1 Announce Type: new  Abstract: Activation quantisation is usually evaluated with an aggregate metric, perplexity, averaged over every token a model predicts. We ask whether that average identifies which computations a quantiser damages. Perplexity turns out to be a reliable aggregate signal: across 12 models from four families and 780 within-model comparisons, the arm perplexity prefers also retains more induction and more retrieval in all but 2.1 and 4.0 percent of cases respectively. But where perplexity has risen by only a factor of 1.2 to 1.5, induction still keeps 0.959 of its intact accuracy while retrieval has already fallen to 0.554, a gap the aggregate number does not surface. This gap has structure, not just size: a matched Gaussian-noise control of the same per-channel magnitude leaves it largely intact, and randomising only the sign of the quantisation error, every magnitude held fixed, is nearly as harmless, so magnitude alone does not explain the damage.
    
[^271]: Swiss-Knife：解码时可重构的外部化多目标对齐框架

    Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time

    [https://arxiv.org/abs/2609.22226](https://arxiv.org/abs/2609.22226)

    该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。

    

    解码时对齐方法通过使用外部奖励对候选续写进行评分并选择得分最大者，来引导一个冻结的语言模型。我们认为这种共享设计只是更大空间中的一个退化点。我们提出Swiss-Knife，一个外部化多目标对齐框架，其中对齐规范是一等运行时对象：可热插拔的评分刀片、批归一化器、成对聚合算子和选择规则。六个公理刻画了可容许的聚合算子，并且我们证明了一个表示定理：每个满足这些公理的算子都具有形式 $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$，这是一个双参数族，其中包含probit和logistic比较规则以及逐点argmax作为命名的坐标。在该族中，成对聚合在对抗性奖励污染下是Lipschitz稳定的，而argmax则不是，且候选批归一化……（原文摘要在此处截断）

    arXiv:2609.22226v1 Announce Type: new  Abstract: Decode-time alignment methods steer a frozen language model by scoring candidate continuations with an external reward and selecting the maximiser. We argue that this shared design is a single degenerate point in a much larger space. We introduce Swiss-Knife, a framework for externalised multi-objective alignment in which the alignment specification is a first-class runtime object: hot-swappable scoring blades, a batch normaliser, a pairwise aggregation operator, and a selection rule. Six axioms characterise the admissible aggregation operators, and we prove a representation theorem: every operator satisfying them has the form $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$, a two-parameter family containing probit and logistic comparison rules and pointwise argmax as named coordinates. Within it, pairwise aggregation is Lipschitz-stable under adversarial reward contamination while argmax is not, and Candidate-Batch Norma
    
[^272]: 面向边缘洪水分割的跨架构基础模型蒸馏

    Cross-Architecture Foundation-Model Distillation for Edge Flood Segmentation

    [https://arxiv.org/abs/2609.20441](https://arxiv.org/abs/2609.20441)

    该论文通过将3亿参数的地理空间基础模型蒸馏到仅70万参数的EfficientViT-B0学生模型中，并利用教师模型对未标注Sentinel-2影像生成伪标签来扩展训练数据，实现了可在边缘设备部署的高性能洪水分割模型。

    

    地理空间基础模型能够提供强大的洪水分割性能，但其庞大的规模限制了在内存受限的边缘硬件上的部署。我们将一个在252个手工标注的Sen1Floods11训练场景上微调的3亿参数Prithvi-EO-2.0教师模型，蒸馏为70万参数的EfficientViT-B0学生模型。教师模型为额外的未标注Sentinel-2影像提供监督信号，使学生模型的训练集无需新的人工标注即可扩展。在252个场景的匹配预算下，教师监督训练与直接训练相比具有竞争力，并在测试配置中提升了STURM-Flood的性能；几何匹配的对照实验表明，仅凭标签来源并不能解释这一差异。将教师监督数据池扩展至2,500个场景缩小了学生与教师模型之间剩余的差距：浮点学生模型在Sen1Floods11测试集上达到0.787的水体交并比（IoU），而教师模型为0.822

    arXiv:2609.20441v1 Announce Type: cross  Abstract: Geospatial foundation models can provide strong flood-segmentation performance, but their size limits deployment on memory-constrained edge hardware. We distill a 300-million-parameter Prithvi-EO-2.0 teacher, fine-tuned on the 252 manually labeled Sen1Floods11 training scenes, into a 0.7-million-parameter EfficientViT-B0 student. The teacher supervises additional unlabeled Sentinel-2 imagery, allowing the student training set to grow without new manual annotations. At the matched budget of 252 scenes, teacher-supervised training is competitive with direct training and improves STURM-Flood performance across tested configurations; a geometry-matched control shows that label source alone does not explain the difference. Scaling the teacher-supervised pool to 2,500 scenes narrows the remaining student--teacher gap: the float student reaches 0.787 water intersection over union on the Sen1Floods11 test split against 0.822 for the teacher, m
    
[^273]: 面向自动驾驶选择性轨迹规划的风险感知世界建模与流引导占据演化方法

    Risk-Aware World Modeling with Flow-Guided Occupancy Evolution for Selective Trajectory Planning in Automated Driving

    [https://arxiv.org/abs/2609.18442](https://arxiv.org/abs/2609.18442)

    提出了RiskWorld风险感知世界建模框架，通过流引导占据演化实现共享占据预测，仅在额外预测风险触发干预且替代轨迹满足逐分量约束时才进行选择性轨迹替换，从而提升自动驾驶运动规划的安全性。

    

    自动驾驶中的安全运动规划需要预判不断演变的交通风险，并决定何时修改当前规划的轨迹。我们提出了RiskWorld，一个面向共享占据预测和选择性轨迹替换的风险感知世界建模框架。该框架将空间风险场和时间参与者上下文与视觉鸟瞰图特征进行融合。流引导演化机制传输占据和场景特征，同时利用带符号残差在传输后对占据进行修正。每个规划步骤仅生成一次预测，并在各候选之间复用。每个候选轨迹与当前状态持续参考进行比较，从而产生非负的碰撞分数修正。由当前世界评估选出的轨迹作为规划锚点，只有当额外预测的风险触发干预、且替代轨迹满足对预测风险和轨迹误差的逐分量约束时，该锚点轨迹才会被替换。

    arXiv:2609.18442v1 Announce Type: new  Abstract: Safe motion planning in automated driving requires anticipating evolving traffic risks and deciding when to revise the current planned trajectory. We introduce RiskWorld, a risk-aware world modeling framework for shared occupancy forecasting and selective trajectory replacement. Spatial risk fields and temporal actor context are fused with visual bird's-eye-view features. Flow-guided evolution transports occupancy and scene features, while signed residuals correct occupancy after transport. One forecast is generated per planning step and reused across candidates. Each candidate is compared with a current-state persistence reference, yielding a nonnegative collision-score correction. The trajectory selected by current-world evaluation serves as the planning anchor and is replaced only when additional predicted risk triggers intervention and an alternative satisfies component-wise constraints on predicted risk and trajectory error. Candida
    
[^274]: 坏天才：超越任务特定捷径的反事实引导测试框架演化

    Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts

    [https://arxiv.org/abs/2609.18366](https://arxiv.org/abs/2609.18366)

    提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。

    

    可靠的智能体评估因自动测试框架优化而变得复杂，这类优化方法反复使用已发布的基准 $B_{\mathrm{rel}}$ 来引导一个提议者，该提议者围绕固定的目标智能体编辑提示词、记忆、检索、工具和控制代码。任务保留集虽然改变了语义任务，但基准协议保持不变，因此一个“坏天才”提议者可以生成一个作弊的测试框架，其在发布基准上的性能提升依赖于整个基准范围的捷径。我们提出了反事实测试框架搜索与演化，将测试框架演化建模为在保持有效性的基准反事实上的约束生成问题。在每次提议者更新后，一个挑战者会搜索能大幅摧毁性能提升的可执行协议变换。有效性防火墙检查任务语义是否得到保留，而确认集则决定反事实是否进入有限存档。我们形式化定义了一个精确的捷径中和基准 $B

    arXiv:2609.18366v1 Announce Type: new  Abstract: Reliable agent evaluation is complicated by automatic harness optimization, which repeatedly uses a released benchmark $B_{\mathrm{rel}}$ to guide a Proposer that edits prompts, memory, retrieval, tools, and control code around a fixed target agent. Task holdout varies semantic tasks but leaves the benchmark protocol fixed, so a "bad genius" Proposer can produce a cheating harness whose released-benchmark gain depends on a benchmark-wide shortcut. We introduce Counterfactual Harness Search and Evolution (CHASE), which casts harness evolution as constraint generation over validity-preserving benchmark counterfactuals. After each Proposer update, a Challenger searches for an executable protocol transformation with large gain destruction. A validity firewall checks that task semantics are preserved, while a confirmation set determines whether the counterfactual enters a finite archive. We formalize an exact shortcut-neutralized benchmark $B
    
[^275]: 你在学习生物信号还是捷径？蛋白质-蛋白质相互作用数据集中的偏差审计与缓解

    Are You Learning Biological Signal or Shortcuts? Auditing and Mitigating Bias in Protein-Protein Interaction Datasets

    [https://arxiv.org/abs/2609.10193](https://arxiv.org/abs/2609.10193)

    本文系统审计了蛋白质-蛋白质相互作用数据集中已报道和未报道的多种偏差（包括随机数据划分引入的拓扑捷径），这些偏差会使机器学习模型学到捷径而非真正的生物信号，并提出相应的缓解方法。

    

    蛋白质-蛋白质相互作用（PPI）数据库并不能忠实反映生物学现实。相反，它们受到研究和实验偏差的影响，这些偏差扭曲了某些蛋白质和相互作用的属性。如果负样本数据集构建不当，机器学习模型可能会将这些偏差当作学习捷径加以利用。迄今为止，PPI数据集构建过程中引入的捷径仅被孤立地研究过。本文中，我们系统地表征了PPI数据集中已报道的以及（据我们所知）此前未被报道的偏差，这些偏差会导致机器学习模型学习捷径而非生物信号。我们分析了HIPPIE、IntAct和STRING这三个专门的PPI数据库，以及两个源自蛋白质数据库（PDB）中3D结构信息的数据集。我们证明了随机数据划分会引入强烈的拓扑捷径。当移除训练集与测试集之间的蛋白质重叠后，所产生的数据集仍然……

    arXiv:2609.10193v2 Announce Type: replace-cross  Abstract: Protein-protein interaction (PPI) databases do not faithfully reflect biological realities. Instead, they are influenced by study and technical biases that distort certain protein and interaction attributes. Machine learning models can exploit these as learning shortcuts if the negative dataset is not constructed with care. So far, the shortcuts introduced during PPI dataset construction have only been examined in isolation. Here, we systematically characterize both reported and, to our knowledge, previously unreported biases in PPI datasets that lead machine learning models to learn shortcuts instead of biological signal. We analyze HIPPIE, IntAct, and STRING, dedicated PPI databases, as well as two datasets derived from 3D-structural information in the Protein Data Bank (PDB). We show that random data splitting introduces strong topological shortcuts. When train-test protein overlap is removed, the resulting datasets still re
    
[^276]: TabBench-Bio：面向高维生物医学表格机器学习的动态基准测试

    TabBench-Bio: A Living Benchmark for Machine Learning on High-Dimensional Biomedical Tables

    [https://arxiv.org/abs/2609.07441](https://arxiv.org/abs/2609.07441)

    TabBench-Bio是一个涵盖43个生物医学数据集的动态基准测试，系统评估了经典方法、神经网络和表格基础模型在高维小样本场景下的表现，发现表格基础模型总体领先，其中RealTabPFN v2.5表现最佳。

    

    生物医学表格通常将数千个测量变量与仅有数十或数百个标记样本结合在一起，这种数据情形在通用表格基准测试中很少被涵盖。我们提出了TabBench-Bio，一个包含43个跨多个领域生物医学数据集的动态交互式基准测试。在统一的交叉验证协议下，我们在28个特征-样本工作点上比较了经典估计器、神经网络和表格基础模型。在10,000个特征和100个训练样本的参考设置下，RealTabPFN v2.5具有最高的点估计值，其次是逻辑回归和TabDPT，后两者的点估计值几乎相同。对目标池进行配对自助法分析显示，RealTabPFN v2.5领先逻辑回归145 Elo（95%区间为[59, 232]）。表格基础模型总体上占据领先排名，而最强的配置取决于具体的工作点和生物医学模态。

    arXiv:2609.07441v1 Announce Type: cross  Abstract: Biomedical tables often combine thousands of measured variables with only tens or hundreds of labelled samples, a regime that is poorly represented in general-purpose tabular benchmarks. We introduce TabBench-Bio, a living and interactive benchmark of 43 biomedical datasets spanning multiple domains. Under a shared cross-validation protocol, we compare classical estimators, neural networks, and tabular foundation models across 28 feature-by-sample operating points. At the reference cell of 10,000 features and 100 training samples, RealTabPFN v2.5 has the highest point estimate, followed by Logistic Regression and TabDPT, whose point estimates are nearly identical. A paired bootstrap over the target pool separates RealTabPFN v2.5 from Logistic Regression by 145 Elo (95% interval [59, 232]). Tabular foundation models generally occupy the leading ranks, while the strongest configuration depends on the operating point and biomedical modali
    
[^277]: OracleZoom：基于在线策略自蒸馏启发的参考约束递归图像超分辨率

    OracleZoom: On-Policy Self-Distillation Inspired Reference-Constrained Recursive Image Super Resolution

    [https://arxiv.org/abs/2609.06490](https://arxiv.org/abs/2609.06490)

    OracleZoom提出了一种受自蒸馏启发、参考约束的递归图像超分辨率框架，通过直接与跨尺度监督、无参考质量目标、KL约束潜在先验和EMA一致性等技术，解决了深层递归缺乏真实标签监督的问题，在七个数据集上实现了极端放大倍率下最先进的超分辨率质量。

    

    递归超分辨率（SR）通过将预测结果反复馈入同一模型，将固定尺度的超分辨率扩展到极端放大倍率，类似于反复缩放图像。然而，在每个尺度上获取真实标签数据仍然具有挑战性，尤其是在深层阶段，因为所需的源分辨率呈几何级数增长，导致更深层的预测缺乏监督。我们提出了OracleZoom，这是一个受在线策略蒸馏启发、参考约束的框架，它在自身轨迹上进行训练，同时将最后的真实标签证据延伸到监督边界之外。直接监督和跨尺度监督约束可验证的内容，而无参考质量目标则引导未解决的精细尺度细节。KL约束的预训练潜在先验限制了由质量驱动的漂移，而EMA一致性则稳定了监督边界。在七个数据集上，OracleZoom在各种缩放尺度上实现了最先进的超分辨率质量，平均0

    arXiv:2609.06490v1 Announce Type: cross  Abstract: Recursive Super-Resolution (SR) extends fixed-scale SR to extreme magnification by repeatedly feeding predictions back into the same model, analogous to zooming an image repeatedly. However, ground truth availability at every scale, especially at depth, remains challenging as the required source resolution grows geometrically, leaving deeper predictions unsupervised. We present OracleZoom, an on-policy distillation-inspired, reference-constrained framework that trains on its trajectory while carrying the last ground-truth evidence beyond the supervision boundary. Direct and cross-scale supervision constrain verifiable content, while a no-reference quality objective guides unresolved fine-scale detail. A KL-constrained pretrained latent prior limits quality-driven drift, while EMA consistency stabilizes the supervision boundary. Across seven datasets, OracleZoom achieves the state-of-the-art SR quality across zooming scales, averaging 0
    
[^278]: 现代Transformer是隐式混合体：从功能分化到有原则的混合架构设计

    Modern Transformers Are Implicit Hybrids: From Functional Differentiation to Principled Hybrid Architecture Design

    [https://arxiv.org/abs/2609.02986](https://arxiv.org/abs/2609.02986)

    本文提出RFIS和RPD两个干预指标，发现现代Transformer的注意力头自然分化为由全局位置频带（GPBand）分隔的检索头和位置头两类，为有原则地设计FA-LA混合架构提供了实证基础。

    

    结合全注意力（FA）和线性注意力（LA）的混合架构日益受到关注，但其注意力分配方式仍然依赖于启发式规则。我们在基于RoPE的Transformer所学习到的头级别功能组织中寻求一个有证据支撑的基础。行为探针无法给出完整的分类体系，因此我们提出了两个干预指标：RoPE频率重要性分数（RFIS），衡量每个频率如何影响某个头的注意力分布；以及RoPE位置依赖性（RPD），用于分离注意力头对旋转位置调制的依赖程度。在Qwen3系列模型和Llama3.1上，RFIS给出假设且RPD加以验证，形成了一个完整的分类体系，将由显著中低频带分隔的检索头和位置头区分开来。受控Transformer实验表明，这一边界遵循训练长度的位置尺度；我们将其命名为全局位置频带。该分析揭示了零样本长度外推失败的潜在原因，并产生了

    arXiv:2609.02986v1 Announce Type: new  Abstract: Hybrid architectures combining Full Attention (FA) and Linear Attention (LA) are increasingly prominent, yet their allocation remains heuristic. We seek an evidence-grounded basis in head-level functional organization learned by RoPE-based Transformers. Behavioral probes do not yield a complete taxonomy, so we propose two intervention metrics: RoPE Frequency Importance Score (RFIS), measuring how each frequency affects a head's attention distribution, and RoPE Positional Dependence (RPD), isolating dependence on rotary positional modulation. On Qwen3-series models and Llama3.1, RFIS suggests and RPD verifies a complete taxonomy of retrieval and positional heads separated by a salient mid-low-frequency band. Controlled Transformers show that this boundary follows the training-length positional scale; we term it the Global Positional Band (GPBand). The analysis suggests a potential cause of zero-shot length-extrapolation failure and yields
    
[^279]: XMerge：用于大语言模型深度压缩的跨轴选择与重构式层合并

    XMerge: Cross-Axis Selection and Reconstructive Layer Merging for LLM Depth Compression

    [https://arxiv.org/abs/2609.02083](https://arxiv.org/abs/2609.02083)

    XMerge 是一种训练后的大语言模型深度压缩方法，通过跨轴选择识别隐藏状态变化最小的层块，并利用局部边界重构重新拟合相邻存留块，在不改变架构、不增加推理参数、无需任务标签的情况下实现高质量的层删除压缩。

    

    删除完整的 transformer 层可以保持标准的服务架构，但现有的深度压缩方法可能会损失大量质量，且损失在不同模型间变化难以预测。我们提出了 XMerge，这是一种包含两个组件的训练后方法。跨轴选择用于识别隐藏状态的相对幅度和角度变化较小的层块，局部边界重构则重新拟合相邻的存留块以匹配原始两个块的输出。XMerge 不使用任务标签或端到端微调，也不引入架构变更或额外的推理时参数。在七个 Llama 和 Qwen 主干模型（0.5B-8B）、五个已发表的基线方法和三个层削减级别上的实验表明，其相对于基线的优势在最激进的删除设置下最大：在 k=4 时，它在 CORE（一个 22 项任务的聚合基准）上于七个主干模型中的六个上排名第一，并且在 MMLU 上也在七个中的六个上排名第一（两个基准上同时排名第一的有五个）。

    arXiv:2609.02083v1 Announce Type: cross  Abstract: Removing complete transformer layers preserves a standard serving architecture, but existing depth-compression methods can lose substantial quality, and the loss varies unpredictably across models. We introduce XMerge, a post-training method with two components. Cross-axis selection identifies a block with low relative-magnitude and angular hidden-state change, and local boundary reconstruction re-fits the adjacent surviving block to match the original two-block output. XMerge uses no task labels or end-to-end fine-tuning, and it introduces neither architectural changes nor additional inference-time parameters. Across seven Llama and Qwen backbones (0.5B-8B), five published baselines, and three layer-reduction levels, its advantage over baselines is largest at the most aggressive removal: at k=4 it ranks first on six of seven backbones on CORE (a 22-task aggregate) and, separately, on six of seven on MMLU (five of seven on both at once
    
[^280]: 顶格截断的接受直方图揭示块扩散投机解码中的搁浅加速

    Ceiling-Clipped Acceptance Histograms Indicate Stranded Speed-up in Block-Diffusion Speculative Decoding

    [https://arxiv.org/abs/2608.30427](https://arxiv.org/abs/2608.30427)

    该论文发现块扩散投机解码中，高接受率会使草稿模型提前耗尽训练时的块长度上限，造成未被实现的“搁浅加速”，提出用接受直方图顶格柱的尖峰作为投入训练算力前的预检指标，并证明推理时简单扩大块大小无法恢复该加速。

    

    投机解码通过一个高效的草稿模型为目标模型提出词元，由目标模型在单次前向传播中完成验证，从而加速生成过程，同时保持目标模型的输出分布。高接受率的块扩散草稿模型（如DFlash和DFlare）可以在一次并行传播中填满整个块。在许多周期中，目标模型会接受整个块，因此草稿模型在验证失败之前就已耗尽其训练时的块长度上限。我们将这种未被实现的接受称为“搁浅加速”。按提示或按周期统计的平均提交长度会掩盖这一现象，而接受直方图则能将其暴露为顶格柱中的尖峰，即接受整个块的周期所占的比例。我们建议在投入训练算力之前，将直方图作为一项预检手段。在推理阶段简单地扩大块大小并不能恢复这一加速，因为一旦块超过其训练尺寸，草稿模型的双向注意力即使在早期阶段也会使其分布发生偏移。

    arXiv:2608.30427v1 Announce Type: new  Abstract: Speculative decoding speeds up generation with an efficient draft model (drafter) that proposes tokens for a target model to verify in one pass, preserving the target's output distribution. High-acceptance block-diffusion drafters such as DFlash and DFlare fill an entire block in one parallel pass. In many cycles, the target accepts the whole block, so the drafter exhausts its trained block horizon before verification fails. We call this unrealized acceptance stranded speed-up. A mean committed length, per prompt or per cycle, hides it, whereas the acceptance histogram exposes it as a spike in the ceiling bin, the fraction of cycles that accept the entire block. We recommend the histogram as a preflight check before spending training compute. Naively widening the block at inference does not recover the speed-up, because once the block outgrows its training size, the drafter's bidirectional attention shifts its distribution even at early 
    
[^281]: 可操作的CBFI：融合结构分解与因果反事实补救的表格机器学习框架

    Actionable CBFI: Integrating Structural Decomposition and Causal Counterfactual Recourse for Tabular Machine Learning

    [https://arxiv.org/abs/2608.27821](https://arxiv.org/abs/2608.27821)

    该论文提出A-CBFI框架，基于结构因果模型将特征重要性分解与因果反事实补救相结合，通过隔离协同交互瓶颈并释放抑制性结构锁，为表格机器学习生成因果有效、可操作的针对性干预方案，克服了现有方法因果无效与干预分散的问题。

    

    可解释人工智能（XAI）日益呼唤可操作的反事实补救，然而现有方法面临因果无效性、过高认知负担以及预测失效等挑战。穷举式因果搜索算法往往需要对多个属性进行修改，而加性归因引导的方法（如SHAP）则忽略了高阶特征之间的协同作用，导致在XGBoost等复杂非线性模型中出现次优的预测动力和分散的干预努力。为弥合这一差距，我们提出了可操作的基于案例的特征重要性（A-CBFI），这是一个面向表格机器学习的“诊断-处方”一体化框架。该框架植根于结构因果模型（SCM），能够隔离协同交互瓶颈并释放抑制性结构锁，进而将其转化为有针对性的干预措施。通过在数学上将活跃的用户干预空间……（摘要在此处不完整）

    arXiv:2608.27821v1 Announce Type: new  Abstract: Explainable artificial intelligence (XAI) increasingly calls for actionable counterfactual recourse, yet current methodologies face challenges related to causal invalidity, excessive cognitive burden, and predictive failure. Exhaustive causal search algorithms often require modifications to multiple attributes, whereas additive attribution-guided methods, such as SHAP, ignore higher-order feature synergies, leading to suboptimal predictive momentum and diffuse intervention effort in complex nonlinear models, such as XGBoost. To bridge this gap, we introduce actionable case-based feature importance (A-CBFI), a diagnosis-prescription integrated framework for tabular machine learning. Grounded in structural causal models (SCMs), A-CBFI isolates synergistic interaction bottlenecks and releases suppressive structural locks, translating them into targeted interventions. By mathematically separating the active user intervention space (L_{\mathr
    
[^282]: MetaSieve：通过基于SQL的元路径选择加速关系型深度学习

    MetaSieve: Faster Relational Deep Learning through SQL-Based Metapath Selection

    [https://arxiv.org/abs/2608.25903](https://arxiv.org/abs/2608.25903)

    本文提出MetaSieve，利用SQL连接和聚合能力，通过智能剪枝元路径来减小GNN训练子图大小，从而加速关系型深度学习，同时保持模型准确性。

    

    关系型深度学习（RDL）是对多表关系数据库进行机器学习的一种有效方法。在RDL中，数据库被建模为一个图，其中每一行是一个节点，每个外键关系是一条边，并在此图上训练图神经网络（GNN）。训练GNN需要在训练集中每个种子节点周围采样子图，训练成本主要由这些子图的大小决定。本文旨在通过利用关系数据库系统的连接和聚合能力来减小子图大小。我们观察到，采样得到的子图是通过跟随由外键链接组成的元路径获得的，并且许多这样的元路径可以在不损失准确性的情况下被剪枝。我们提出了MetaSieve，一个元路径选择层，用于确定保留哪些元路径以及剪枝哪些元路径。对于每个候选元路径扩展，MetaSieve通过SQL连接和聚合计算统计信息。

    arXiv:2608.25903v1 Announce Type: cross  Abstract: Relational Deep Learning (RDL) is an effective approach to machine learning over multi-table relational databases. In RDL, a database is modeled as a graph in which each row is a node and each foreign-key relation is an edge, and a graph neural network (GNN) is trained on this graph. Training a GNN requires sampling a subgraph around every seed node in the training set, and the cost of training is largely determined by the size of these subgraphs. This paper aims to reduce subgraph size by leveraging the join and aggregation capabilities of relational database systems. We observe that sampled subgraphs are obtained by following metapaths composed of foreign-key links, and that many of these metapaths can be pruned without loss of accuracy. We present MetaSieve, a metapath selection layer that determines which metapaths to retain and which to prune. For each candidate metapath extension, MetaSieve computes statistics via SQL join and ag
    
[^283]: 一种面向大字母表的分层单纯形架构

    A Layered Simplex Architecture for Large Alphabets

    [https://arxiv.org/abs/2608.19908](https://arxiv.org/abs/2608.19908)

    本文提出一种极简的分层单纯形贝叶斯估计器，通过均匀相乘并归一化实现，无需调参，具有显式可计算的遗憾值，并在大字母表概率估计任务上与专门方法竞争，同时揭示缩放规律。

    

    arXiv:2608.19908v1 公告类型：交叉 摘要：在对数损失下对大字母表进行概率估计是一个被广泛研究的问题，已有诸如古德-图灵估计器等著名方法。我们引入并研究了一种新的贝叶斯估计器，具有四个显著特性。首先，其构造极其简单：将来自概率单纯形的独立均匀样本按坐标相乘并重新归一化。深度是唯一的结构参数，而对深度进行平均则消除了调整它的需要。其次，所得混合模型的遗憾值——相对于已知源的编码所支付的超额码长——具有显式且可高效计算的表达式。第三，尽管其简单且无需调整常数，该估计器在多种合成和真实文本基准上，与包括古德-图灵在内的更专门方法相比，表现出竞争力。第四，其遗憾值的可处理性使我们能够识别数据和字母表中的缩放规律。

    arXiv:2608.19908v1 Announce Type: cross  Abstract: Probability estimation over large alphabets under log loss is a well-studied problem, with celebrated methods such as the Good-Turing estimator. We introduce and study a new Bayesian estimator with four notable properties. First, its construction is exceptionally simple: multiply independent uniform draws from the probability simplex coordinate-wise and renormalize. Depth is the only structural parameter, and averaging over depths eliminates the need to tune it. Second, the regret of the resulting mixture, the excess code length it pays relative to a code that knows the source, admits an explicit and efficiently computable expression. Third, despite its simplicity and lack of tuned constants, the estimator is competitive across a diverse set of synthetic and real-text benchmarks with substantially more specialized methods, including Good-Turing. Fourth, the tractability of its regret allows us to identify scaling laws in data, alphabet
    
[^284]: 高效无海森方法用于非凸下层的多目标双层优化

    Efficient Hessian-Free Methods for Multi-Objective Bilevel Optimization with Nonconvex Lower Level

    [https://arxiv.org/abs/2608.12704](https://arxiv.org/abs/2608.12704)

    本文提出了一种新的多目标Moreau包络无海森算法（MOMEHA），首次解决了非凸下层问题的多目标双层优化，通过单循环和无海森设计实现了高效计算。

    

    摘要：arXiv:2608.12704v1 公告类型：交叉 摘要：多目标双层优化在人工智能领域有广泛应用，如自动学习和多任务元学习。尽管最近一些工作开始研究多目标双层优化，但提出的方法依赖于（强）凸下层问题。实际上，这些多目标双层学习问题通常是非凸的，特别是其下层问题是非凸的。为填补这一空白，我们提出了一类基于多目标Moreau包络的无海森算法（MOMEHA），用于解决具有非凸下层的多目标双层学习问题。具体来说，我们的方法使用Moreau包络将原始问题转换为带有包络约束的多目标单层优化。特别是，我们的方法通过引入平滑处理，在多目标设置中保留了单循环和无海森的计算优势。

    arXiv:2608.12704v1 Announce Type: cross  Abstract: Multi-objective bilevel optimization has wide applications in the AI area such as automated learning and multi-task meta-learning. Although recently some works have been begun to study the multi-objective bilevel optimization, the proposed methods rely on the (strongly) convex lower level problems. In fact, these multi-objective bilevel learning problems are generally nonconvex, and particularly their lower level problems are nonconvex. To fill this gap, we propose a class of Multi-Objective Moreau Envelope based Hessian-free Algorithms (MOMEHA) to solve the multi-objective bilevel learning problems with nonconvex lower level. Specifically, our method uses the Moreau envelope to convert the original problem into a multi-objective single-level optimization with an envelope constraint. In particular, our method retains computational advantages of being single-loop and Hessian-free in the multi-objective setting by incorporating a smooth 
    
[^285]: CommitKV：面向多模态智能体的提交对齐KV缓存压缩

    CommitKV: Commit-Aligned KV Cache Compression for Multimodal Agents

    [https://arxiv.org/abs/2608.07855](https://arxiv.org/abs/2608.07855)

    CommitKV是一种无需训练的多模态智能体KV缓存压缩方法，通过对比工具调用前后对同一缓存页面的删除影响来捕捉证据使用的变化，淘汰两个时间窗口中影响力均低的页面并暂时保护被观察激活的页面，避免因注意力转移而误删后续需要的信息。

    

    多模态智能体通过推理和工具使用来收集视觉证据，将其解决任务的能力扩展到初始输入之外。然而，不断累积的观察结果会增加KV缓存的内存占用和推理成本。令牌驱逐（token eviction）通过保留有用的缓存令牌来降低这些成本。现有方法使用注意力、冗余度或特定阶段的查询来对令牌排序，却没有在工具交互之间显式比较同一历史记录。当观察结果使注意力发生转移时，这些方法可能将暂时降低的影响力误判为失去用处，从而驱逐掉后续需要的信息。我们提出了CommitKV，一种无需训练的KV缓存压缩方法，能够应对这种不断变化的证据使用方式。提交对齐测量在工具调用完成之前和其观察结果返回之后，比较对同一页面的删除影响。生命周期感知压缩会淘汰在两个时间窗口中影响力都很低的页面，并暂时保护影响力发生变化的页面……（原文在此截断）

    arXiv:2608.07855v2 Announce Type: replace  Abstract: Multimodal agents gather visual evidence through reasoning and tool use, extending their ability to solve tasks beyond the initial input. However, accumulating observations increases KV cache memory and inference costs. Token eviction reduces these costs by retaining useful cached tokens. Existing methods use attention, redundancy, or stage-specific queries to rank tokens, without explicitly comparing the same history across tool interactions. When observations shift attention, they may mistake temporarily reduced influence for lost usefulness and evict information needed later. We propose CommitKV, a training-free KV cache compression method that accounts for this changing evidence use. Commit-aligned measurement compares the deletion effects of the same page before a tool call completes and after its observation returns. Lifecycle-aware compression retires pages with low influence in both windows and temporarily protects pages infl
    
[^286]: 超越共动性：基于暴露度的局部性为投资组合分散化启用联合因子-图框架

    Beyond Co-Movement: Locality by Exposures Enables a Joint Factor-Graph Framework for Portfolio Diversification

    [https://arxiv.org/abs/2608.06618](https://arxiv.org/abs/2608.06618)

    该论文提出MINGLE框架，用系统性因子暴露度而非观测共动性来定义图的局部性，并通过统一的ADMM框架联合学习潜在因子表示与图拓扑结构，从而在真实市场结构上实现更有效的投资组合分散化。

    

    当前的投资组合构建方法要么对特质冲击的影响缺乏感知（标准因子模型），要么对驱动系统性收益的潜在数据结构缺乏感知（近期基于图的方法）。这为我们提供了一个将因子域与图域所捕获的互补市场特征相结合的机会，使资产配置能够直接在底层市场结构上运作，而不是依赖于观测到的共动性或有限样本带来的伪影。在这项工作中，我们提出了互信息化的图局部性与暴露度框架（MINGLE），该框架通过系统性因子暴露特征（而非观测到的共动性）重新定义图的局部性，从而对因子域和图域进行相互正则化。该方法通过统一的交替方向乘子法（ADMM）框架加以形式化，直接联合学习潜在因子表示及其诱导的图拓扑结构。

    arXiv:2608.06618v3 Announce Type: replace-cross  Abstract: Current portfolio construction methods are either agnostic to the effects of idiosyncratic shocks (standard factor models) or to the latent data structure driving systematic returns (recent graph-based approaches). This presents an opportunity to combine the complementary market aspects captured by the factor and graph domains, allowing asset allocations to operate directly on the underlying market structure, rather than on its observed co-movement or its finite-sample artefacts. In this work, we introduce the Mutually-INformed Graph-Locality and Exposures framework (MINGLE), which mutually regularises the factor and graph domains by redefining graph locality through systematic factor exposure profiles, rather than via observed co-movements. This is formalised through a unified Alternating Direction Method of Multipliers (ADMM) framework that jointly learns a latent factor representation and its induced graph topology directly 
    
[^287]: 用于大语言模型因果发现的思维图方法：公平性路径案例研究

    Graph-of-Thoughts for LLM Causal Discovery with a Fairness-Pathway Case Study

    [https://arxiv.org/abs/2608.02877](https://arxiv.org/abs/2608.02877)

    GoT-CD是一种基于思维图的LLM因果发现方法，它让LLM一次性生成多个完整候选图，并通过确定性评分排序、硬并集约束合并、细化与DAG投影，在多个基准上取得了优于现有LLM因果发现方法的F1分数和结构汉明距离。

    

    使用大语言模型进行因果发现的方法都是局部推理的，每次只决定一个变量对或一个节点，因此它们从不比较不同的完整图。我们提出GoT-CD，一种思维图方法，它让LLM仅根据变量元数据生成多个完整的候选图，用确定性的有效性评分对它们排序，在硬并集约束下进行合并，对合并结果加以细化，并将结果投影为有向无环图（DAG）。我们在Asia、Child数据集以及一个包含100个样本的阿尔茨海默病图上，使用GPT-4o-mini进行评估。在阿尔茨海默病和Child数据集上的五个随机种子实验中，GoT-CD在各LLM方法中取得了最高的平均F1分数和最低的平均结构汉明距离，八次比较中有七次差异的95%置信区间不包含零，并且它在每次运行中都能返回DAG，同时每个图仅需三次LLM请求。在移除各组件的情况下重放记录的运行结果显示，大部分性能提升归因于

    arXiv:2608.02877v2 Announce Type: replace  Abstract: Methods that use large language models (LLMs) for causal discovery reason locally, deciding one variable pair or one node at a time, so they never compare alternative whole graphs. We introduce GoT-CD, a Graph-of-Thoughts method that asks an LLM for several complete candidate graphs from variable metadata alone, ranks them with a deterministic validity score, merges them under a hard union constraint, refines the merge, and projects the result to a DAG. We evaluate on Asia, Child, and an Alzheimer's disease graph with 100 samples and GPT-4o-mini. Over five seeds on Alzheimer's and Child, GoT-CD gives the highest mean F1 score and the lowest mean structural Hamming distance of the LLM methods, with 95% intervals of the difference excluding zero in seven of eight comparisons, and it returns a DAG in every run while using three LLM requests per graph. Replaying the logged runs without individual components attributes most of the gain to
    
[^288]: QAdapt：面向表面码量子纠错的噪声自适应神经预解码

    QAdapt: Noise-Adaptive Neural Pre-Decoding for Surface-Code Quantum Error Correction

    [https://arxiv.org/abs/2607.28422](https://arxiv.org/abs/2607.28422)

    QAdapt提出了一种噪声自适应神经预解码框架，通过学习局部时空相关性并结合在线弹性权重巩固，在多种噪声机制间持续适应，显著降低了表面码量子纠错的逻辑错误率并提升了解码效率。

    

    arXiv:2607.28422v2 公告类型：替换 摘要：容错量子计算依赖量子纠错来保护逻辑信息，因此需要能够跟上综合征生成速度的经典解码器。异构且不断变化的硬件噪声，以及从仿真到实验的分布偏移，对固定神经解码器的准确性和鲁棒性构成了挑战。我们提出了QAdapt，一个面向表面码量子纠错的噪声自适应神经预解码框架。QAdapt学习局部时空相关性，可在不同噪声环境之间持续适应，并将残余综合征传递给传统的全局解码器。在线弹性权重巩固（EWC）机制使得训练在测量、空闲、CNOT和Pauli偏置等噪声机制之间切换时能够限制遗忘。在码距为9的情况下，HTNet预解码器相较于Ising-Fast将等效的每轮逻辑错误率降低了15.64%，同时参数减少了28.75%，匹配后端处理时间减少了9.07%。在线EWC将平均……（摘要原文在此处截断）

    arXiv:2607.28422v2 Announce Type: replace  Abstract: Fault-tolerant quantum computing relies on quantum error correction to protect logical information, requiring classical decoders that keep pace with syndrome generation. Heterogeneous, changing hardware noise and the distribution shift from simulation to experiment challenge the accuracy and robustness of fixed neural decoders. We present QAdapt, a noise-adaptive neural pre-decoding framework for surface-code quantum error correction. QAdapt learns local spatiotemporal correlations, continually adapts across noise environments, and passes the residual syndrome to a conventional global decoder. Online elastic weight consolidation limits forgetting as training moves between measurement, idle, CNOT, and Pauli-bias noise regimes. At distance nine, the HTNet pre-decoder reduces equivalent per-round logical error by 15.64% relative to Ising-Fast, with 28.75% fewer parameters and 9.07% less matching-backend time. Online EWC reduces average 
    
[^289]: Molt：一个面向智能体强化学习的可扩展 PyTorch 原生训练框架

    Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2607.21653](https://arxiv.org/abs/2607.21653)

    提出了 Molt 框架，通过可组合模型并行、统一智能体接口、全异步 rollout 与优化以及分布式经验存储，实现了万亿参数规模下的智能体强化学习训练，且无需修改现有智能体的执行逻辑。

    

    智能体强化学习需要一种研究者能够在不牺牲模型规模或智能体执行控制权的前提下进行修改的基础设施。我们提出了 Molt，一个轻量级的 PyTorch 原生框架，将万亿参数规模的训练与标准智能体接口相结合。Molt 整合了四项能力：基于可组合模型并行的紧凑训练实现；统一的 OpenAI 和 Anthropic 接口，支持上下文压缩后的自动轨迹分段；完全异步的 rollout 与优化；以及面向长多模态轨迹的分布式经验存储。现有智能体可以保留其执行与上下文管理逻辑，同时由共享捕获层记录生成的 token 和行为概率。Rollout 工作进程将大体积的经验数据放入 Ray 的对象存储中，训练器进程通过引用检索分配给它的经验，从而避免对完整 rollout 进行集中式收集。

    arXiv:2607.21653v2 Announce Type: replace-cross  Abstract: Agentic reinforcement learning requires infrastructure that researchers can modify without sacrificing model scale or control over agent execution. We present Molt, a lightweight PyTorch-native framework that combines trillion-parameter training with standard agent interfaces. Molt integrates four capabilities: a compact training implementation built on composable model parallelism; unified OpenAI and Anthropic interfaces with automatic trajectory segmentation after context compaction; fully asynchronous rollout and optimization; and distributed experience storage for long, multimodal trajectories. Existing agents retain their execution and context-management logic while a shared capture layer records generated tokens and behavior probabilities. Rollout workers place heavy experience payloads in Ray's object store, and trainer ranks retrieve their assigned experiences by reference, avoiding a centralized gather of the full roll
    
[^290]: 循环语言模型中的操作性原初内省：在答案出现之前读取正确性

    Operational Proto-Introspection in Looped Language Models: Reading Correctness Before the Answer

    [https://arxiv.org/abs/2607.18553](https://arxiv.org/abs/2607.18553)

    该论文提出“操作性原初内省”这一概念，通过探针读取循环语言模型Ouro-RLTT在推理早期的隐状态，证明可以在答案给出之前部分预测推理结果的正确性（AUROC达0.758，远高于token置信度的0.515），并排除了问题难度等竞争性解释。

    

    一个人能否在语言模型进行推理的过程中就判断出它的答案是否正确？循环模型是提出这一问题的理想场所：它们对每个token多次重复运行相同的层，每轮循环都会留下一个可供探针读取的隐状态。我对Ouro-RLTT（一个26亿参数的循环模型）进行了探针测试，以了解这些状态能揭示其推理将如何收尾的哪些信息。我将这类信息称为操作性原初内省：之所以称其为“操作性”，是因为它由探针所能读取的内容来定义；之所以称其为“原初”（proto），是因为本文没有任何证据表明模型自身使用了这些信息。在一定程度上，答案是肯定的。在推理尚未给出答案的前48个token上，基于隐状态的探针在一个合成逻辑任务上达到了0.758的AUROC，而token置信度仅为0.515。问题难度是一个竞争性解释：在另外的1,164个问题上，token置信度加上问题的简单特征也能取得相近的预测效果。但这不仅仅是难度：在针对同一问题的两个答案之间……（摘要原文在此处截断）

    arXiv:2607.18553v4 Announce Type: replace-cross  Abstract: Can one tell, while a language model reasons, whether its answer will be right? Looped models are a good place to ask: they run the same layers several times per token, leaving one hidden state per loop for a probe to read. I probe Ouro-RLTT, a 2.6B-parameter looped model, for what these states reveal about how its reasoning will end. I call such information operational proto-introspection: operational because it is defined by what a probe can read, and proto because nothing here shows that the model itself uses it.   In part, yes. Reading the first 48 tokens of reasoning, from answers that state no answer yet, a probe on the hidden states reaches an AUROC of 0.758 on a synthetic logic task, against 0.515 for token confidence. Question difficulty is a rival explanation: on 1,164 further questions, token confidence plus simple features of the question predict about as well. It is not only difficulty: between two answers to the s
    
[^291]: 时序基准分数到底衡量了什么？分解视频视觉语言模型评估中的通道使用

    What Does a Temporal Benchmark Score Measure? Decomposing Channel Use in Video VLM Evaluation

    [https://arxiv.org/abs/2607.12304](https://arxiv.org/abs/2607.12304)

    该论文提出“反转冲突”评测方法（帧序反转而RoPE保持正向），将时序视频VLM基准分数分解为“任务是否真正需要时序信息”与“模型是从像素还是位置编码获取时序”两个通道，从而区分真正理解时序的模型与仅仅被干扰的模型。

    

    时序视频问答基准上的分数本应衡量模型是否具备时序理解能力，但它混淆了两个问题：（1）该问题是否真的是时序性的，是否需要多帧及其先后顺序？（2）如果是，模型是从像素中恢复出顺序，还是直接从位置编码（RoPE）中读取顺序？时序分数的大部分内容对这两个问题都无法回答，因为单帧和答案先验往往就足以支撑分数，而该领域现有的有效性检验手段——帧打乱敏感性和多帧增益——只涉及问题（1）。针对问题（2），我们用两种方式对“反转冲突”（即帧序被反转而RoPE仍保持正向）进行评分：一种是无标签方式，即以相对正向标签的准确率损失来衡量；另一种是在标签在时间反转下可确定性变换时，以相对反转标签的准确率来衡量。这能够将真正跟随反转内容的模型与仅仅被干扰的模型区分开来。这两种解读在两种模……（原文截断）

    arXiv:2607.12304v2 Announce Type: replace-cross  Abstract: A score on a temporal video question answering benchmark is meant to measure that a model has temporal understanding, but it conflates two questions. (1)~is the question even temporal, does it need several frames and their order? And (2)~when it does, does the model recover the order from the pixels, or read it off the positional encoding (RoPE)? Most of a temporal score answers neither, since a single frame and answer priors often carry it, and the field's validity checks, frame-shuffle sensitivity and multi-frame gain, speak only to (1). For (2) we score a reversal conflict, frames reversed while RoPE stays forward, two ways: label-free, as the accuracy lost against the forward labels, and against reversed labels wherever labels transform deterministically under time reversal. This separates a model that follows the reversed content from one merely disrupted. The two readings dissociate two similarly scoring models at both mo
    
[^292]: 将规则归纳重新带回流体智力研究？ARC-AGI基准在人类中的初步验证

    Bringing Back Rule Induction to Fluid Intelligence Research? An Initial Validation of the ARC-AGI Benchmark in Humans

    [https://arxiv.org/abs/2607.11263](https://arxiv.org/abs/2607.11263)

    该研究首次在人类样本中验证了ARC-AGI基准的心理测量学特性，发现其与图形流体智力高度相关（ρ=.63），证明这一以规则归纳为核心的基准可作为流体智力的有效测量工具。

    

    关于流体智力（gf）测量存在两种相互竞争的观点：一种认为表现主要受工作记忆容量的限制，另一种认为主要受归纳新颖关系能力的限制。第一种观点目前在测量实践中占主导地位，这体现在使用有限的重复规则集合上；而第二种观点虽体现在许多定义中，却很少出现在实际测量中。ARC-AGI基准主要需要规则归纳能力，并被提议作为衡量人类和人工智能系统流体智力的工具。然而，其心理测量学特性尚未在人类样本中得到检验。因此，我们在一项包含100名参与者的首次研究中考察了ARC-AGI的心理测量学特征和法则网络。研究结果表明，ARC-AGI题目汇编显示出良好的心理测量学特性，并与图形推理测验所测得的图形流体智力存在显著相关（$\rho$ = .63）。

    arXiv:2607.11263v4 Announce Type: replace  Abstract: Two competing perspectives on fluid intelligence (gf) measures propose that performance is primarily constrained either by working memory capacity or by the ability to induce novel relations. The first perspective is currently dominant in measurement, as evident from the use of a limited set of recurring rules, whereas the second perspective is reflected in many definitions but rarely present in measurement. The ARC-AGI benchmark predominantly requires rule induction and was proposed as a measure of gf for both humans and artificial systems. However, its psychometric properties have not yet been examined in human samples. We therefore investigated the psychometric characteristics and nomological network of ARC-AGI in a first study with 100 participants. A compilation of ARC-AGI items showed good psychometric properties and correlated substantially with figural fluid intelligence as measured by a figural reasoning test ($\rho$ = .63).
    
[^293]: 通过参考轨迹审计不可见的权重更新

    Auditing Invisible Weight Updates with Reference Traces

    [https://arxiv.org/abs/2607.09800](https://arxiv.org/abs/2607.09800)

    本研究提出通过高精度参考轨迹对低精度权重更新进行审计，发现随机写回优于最近写回，且该方法在绝对精度上有效但在方向特异性上存在局限。

    

    arXiv:2607.09800v4 公告类型：替换。摘要：直接的低精度写回可能会抹除非零优化器提议。我们探讨高精度参考轨迹在低精度运行之前确立了什么。精确的目标代码事件在实现的目标轨迹上可逐坐标审计；运行前的聚合投影也假设参考轨迹仍然是一个有用的反事实。在一个受控的两层网格中，55/72个单元具有测量和预测的初始化后交叉点：时间跨度达384倍，其中52/55在15%以内，4/72在类别上存在差异。匹配的解码器实验表明，随机而非最近的写回恢复了大部分损失差距。一个前瞻性的解析网格E4M3审计在三个未见过的NeoX风格种子中重用了一个fp32轨迹。它通过了绝对精度和技能门限（宏观RMSE 0.00858），但在方向特异性上失败。在目标结果盲比较中，历史模板的描述性RMSE（0.00360）低于预先声明的来源预测。

    arXiv:2607.09800v4 Announce Type: replace  Abstract: Direct low-precision write-back can erase nonzero optimizer proposals. We ask what a high-precision reference trace establishes before a low-precision run. The exact target-code event is auditable coordinatewise on a realized target trajectory; pre-run aggregate projection also assumes the reference remains a useful counterfactual. In a controlled two-layer grid, 55/72 cells have measured and predicted post-initialization crossings: times span $384\times$, 52/55 are within 15\%, and 4/72 differ in category. Matched decoder experiments show stochastic rather than nearest write-back recovers most of the loss gap. A prospective analytic-grid E4M3 audit reuses one fp32 trace across three unseen NeoX-style seeds. It passes absolute-accuracy and skill gates (macro RMSE 0.00858) but fails directional specificity. In a target-outcome-blind comparison, a historical template has lower descriptive RMSE (0.00360) than the predeclared source pred
    
[^294]: 随机性对于自适应数据分析是必要的吗？

    Is Randomness Necessary for Adaptive Data Analysis?

    [https://arxiv.org/abs/2607.07085](https://arxiv.org/abs/2607.07085)

    本文首次系统研究了自适应数据分析中一个根本性开放问题——随机性是否必要，即探索确定性机制能否达到与随机化机制相当的查询支持能力。

    

    自适应数据分析问题形式化了当数据集被重复使用时防止虚假发现和过拟合的挑战。形式上，我们的输入是一个包含 $n$ 个来自定义域 $\mathcal{X}$ 上未知分布 $\mathcal{P}$ 的独立同分布样本的数据集，我们的目标是回答关于 $\mathcal{P}$ 的一系列 $k$ 个自适应选择的统计查询。主要问题是我们能够支持多少个查询（即 $k$ 最大可以是多少），这主要作为样本数量 $n$ 的函数。这个问题已被深入研究，并且对于随机化机制而言已相对清晰：存在计算高效的机制可以支持 $k \approx n^2$ 个查询，且没有计算高效的机制能够回答 $k \gg n^2$ 个查询。在本文中，我们解决一个根本性问题：随机性对于自适应数据分析是否必要？尽管对自适应数据分析已有十年的研究，这个问题仍然悬而未决。

    arXiv:2607.07085v3 Announce Type: replace-cross  Abstract: The Adaptive Data Analysis (ADA) problem formalizes the challenge of preventing false discovery and overfitting when a dataset is repeatedly reused. Formally, our input is a dataset containing $n$ i.i.d. samples from an unknown distribution $\mathcal{P}$ over a domain $\mathcal{X}$, and our goal is to answer a sequence of $k$ adaptively chosen statistical queries with respect to $\mathcal{P}$. The main question is how many queries we can support (i.e., how large $k$ can be), primarily as a function of the number of samples $n$. This question has been intensively studied and is relatively well-understood for randomized mechanisms: there are computationally efficient mechanisms that support $k \approx n^2$ queries, and no computationally efficient mechanism can answer $k \gg n^2$ queries. In this paper, we address a fundamental question: is randomness necessary for ADA?   Despite a decade of work on ADA, this question remains ope
    
[^295]: 基于物理信息变分自编码器的短跨桥梁不确定性感知损伤识别

    Uncertainty-aware damage identification in short-span bridges via physics-informed variational autoencoder

    [https://arxiv.org/abs/2607.05025](https://arxiv.org/abs/2607.05025)

    该论文提出将数值特征值求解器直接嵌入变分自编码器解码器并利用高斯copula刻画刚度因子后验分布的物理信息框架，实现了短跨桥梁振动损伤识别中可靠的不确定性量化与物理一致性保证。

    

    土木基础设施中基于振动的损伤识别是一个具有挑战性的不适定逆问题，其原因在于测量噪声、稀疏的传感器布置以及建模误差。虽然深度学习在系统识别方面功能强大，但确定性方法缺乏可靠的不确定性量化，且可能产生物理上不一致的结果。本工作提出了一种用于结构健康监测的概率型科学机器学习框架：物理信息高斯copula变分自编码器。该方法引入了两项主要贡献。首先，将数值特征值求解器直接嵌入变分自编码器作为其解码器，从而取代数据驱动的代理模型，确保每个后验样本均满足结构动力学的控制方程，并减少了可训练参数的数量。其次，采用高斯copula描述刚度因子的后验分布，以捕捉参数之间的相关性（原文此处截断）。

    arXiv:2607.05025v3 Announce Type: replace  Abstract: Vibration-based damage identification in civil infrastructure is a challenging, ill-posed inverse problem due to measurement noise, sparse sensor arrays, and modeling errors. While deep learning is powerful for system identification, deterministic approaches lack reliable uncertainty quantification and can yield physically inconsistent results. This work proposes a probabilistic Scientific Machine Learning framework for structural health monitoring: a physics-informed Gaussian copula variational autoencoder. The methodology introduces two main contributions. First, a numerical eigenvalue solver is embedded directly into the variational autoencoder as its decoder, which replaces data-driven surrogates, ensures that every posterior sample satisfies the governing equations of structural dynamics, and reduces the number of trainable parameters. Second, a Gaussian copula describes the posterior distribution of the stiffness factors, captu
    
[^296]: 基础模型何时物有所值？预训练时间序列预测器的盈亏平衡分析

    When Do Foundation Models Pay Off? A Break-Even Analysis of Pretrained Time Series Forecasters

    [https://arxiv.org/abs/2607.04919](https://arxiv.org/abs/2607.04919)

    该论文首次系统性地分析了部署时间序列基础模型何时能比经典方法更划算，通过30个数据集的对比实验给出了基于训练样本量和季节性强度的盈亏平衡经验法则（例如训练样本少于700且季节性显著时零样本基础模型即够用）。

    

    部署时间序列基础模型（FM）需要GPU基础设施和额外的工程开销，且并不能保证比经典预测方法有所改进。我们首次对这种投资何时能够回本进行了系统性的盈亏平衡分析。我们在30个基准数据集上，比较了零样本和LoRA微调的基础模型（Chronos-Bolt、Moirai、Lag-Llama）与经典基线方法（Naive、ETS、ARIMA、XGBoost）的表现，训练集规模从可用数据的2%到100%共设六个档位。在30个数据集中，有16个数据集上基础模型在所有评估的训练数据比例下均优于经典方法。在6个数据集上，经典方法在训练数据不超过5%（21—6,324个样本）时就能超越零样本基础模型；在其余8个数据集上，盈亏平衡点介于24到22,177个样本之间。没有任何一种经典方法全面占优：ARIMA、Naive、ETS和XGBoost各自在6—11个数据集上表现最佳。我们总结出一条无需模型训练的稳健经验法则：如果训练样本数少于700且季节性不可忽略，零样本基础模型即可……（摘要在此处截断）

    arXiv:2607.04919v2 Announce Type: replace  Abstract: Deploying a time series foundation model (FM) requires GPU infrastructure and engineering overhead, with no guarantee of improvement over classical forecasting methods. We provide the first systematic break-even analysis of when this investment pays off. Across 30 benchmark datasets, we compare zero-shot and LoRA fine-tuned FMs (Chronos-Bolt, Moirai, Lag-Llama) against classical baselines (Naive, ETS, ARIMA, XGBoost) at six training set sizes from 2% to 100% of available data. FMs outperform classical methods at every evaluated training fraction on 16 of 30 datasets. On 6 datasets, classical methods surpass zero-shot FMs with at most 5% of the training data (21--6,324 samples); on the remaining 8, break-even ranges from 24 to 22,177 samples. No classical method dominates: ARIMA, Naive, ETS and XGBoost are each best on 6--11 datasets. One robust rule requires no model training: if n_train < 700 and seasonality is non-negligible, FM ze
    
[^297]: 学习式停止何时有帮助？为推理模型服务配置风险受控的提前退出机制

    When Does Learning to Stop Help? Configuring Risk-Controlled Early Exit for Reasoning-Model Services

    [https://arxiv.org/abs/2606.30852](https://arxiv.org/abs/2606.30852)

    该论文提出一种基于工作负载校准数据的部署决策流程，通过Learn-then-Test以可证明的风险界限同时认证学习式停止规则与静态token预算两类提前退出方案并部署成本最低者，实验表明两类方案在不同模型与任务上各有优势、无普适胜者。

    

    提前退出通过在临时答案看起来已是最终答案时停止推理，降低了推理模型的服务成本。服务运营者必须在习得的停止规则与单一静态token预算之间做出选择，而相对于完整推理的节省幅度本身并不能解决这一选择问题。我们提出了一种基于来自实际工作负载的校准数据进行决策的流程：该流程使用Learn-then-Test对两类方法进行认证，确保在所有问题中“完整推理能正确回答但提前退出无法正确回答”的比例不超过给定界限；同时按照服务栈的方式对临时答案进行定价，并部署其中成本最低的已认证候选方案。在四个模型和五个基准测试上，没有任何一类方法始终占优。在5%的风险目标下，使用Qwen3处理自由格式数学任务时，习得的停止规则比静态预算多节省10.4-17.0个百分点的token，准确率仅比完整推理低2.4-2.9个百分点，而在MMLU-Pro上平均节省较少。哪一类方法成本更低也会随场景变化（原文在此处截断）。

    arXiv:2606.30852v3 Announce Type: replace  Abstract: Early exit lowers the cost of serving a reasoning model by stopping its reasoning once a provisional answer looks final. An operator must choose between a learned stopping rule and one static token budget, and savings against full reasoning do not settle the choice. We present a procedure that decides on calibration data from the workload. It certifies both families with Learn-then-Test to one bound on the fraction of all questions that full reasoning answers correctly but early exit does not, prices provisional answers as the serving stack does, and deploys the cheapest certified candidate. Across four models and five benchmarks neither family is better throughout. At a 5% risk target the learned rule saves 10.4-17.0 percentage points more tokens than the static budget on free-form mathematics with Qwen3, with accuracy 2.4-2.9 points below full reasoning, and less on MMLU-Pro on average. Which family costs less also changes with the
    
[^298]: FlexMoE：学习面向灵活MoE部署的一体化多预算专家剪枝

    FlexMoE: Learning One-for-All Multi-Budget Expert Pruning for Flexible MoE Deployment

    [https://arxiv.org/abs/2606.27866](https://arxiv.org/abs/2606.27866)

    FlexMoE提出了一种训练后的一体化多预算专家剪枝框架，通过构建跨预算共享的可切片权重空间和学习专家级保留动作，将大型预训练MoE模型转换为可灵活部署的多预算子网络家族。

    

    混合专家模型通过稀疏的专家激活来扩展模型容量，但在部署时仍需存储和服务所有专家。此外，部署预算在不同设备、用户和工作负载之间可能存在显著差异。现有的MoE压缩方法通常需要为每个目标预算单独优化一个压缩模型。我们提出了FlexMoE，这是一个训练后的“一体化”框架，能够将大型预训练MoE模型转换为一系列多预算的、更小的可部署子网络。FlexMoE首先按重要性对专家FFN通道进行排序，构建一个跨预算共享的可切片权重空间，然后在逐渐增加的成本压力下学习专家级的离散保留动作，使得单次动作训练即可产生跨预算的可靠剪枝掩码。此外，仅需一次中等预算的微调即可进一步恢复性能下降，并产出一个共享的恢复模型……（原文截断）

    arXiv:2606.27866v2 Announce Type: replace  Abstract: Mixture-of-Experts (MoE) language models scale model capacity through sparse expert activation, yet deployment still requires storing and serving all experts. Moreover, deployment budgets can vary substantially across devices, users, and workloads. Existing MoE compression methods typically optimize a separate compressed model for each target budget. We introduce FlexMoE, a post-training one-for-all framework that converts a large pretrained MoE model into a multi-budget family of smaller deployable subnetworks. FlexMoE first ranks expert FFN channels by importance to construct a sliceable weight space shared across budgets, and then learns expert-wise discrete retention actions under gradually increasing cost pressure, allowing a single action-training run to produce reliable pruning masks across budgets. Moreover, only a single mid-budget fine-tune can further recover degraded performance and yields a shared recovered model that tr
    
[^299]: SharQ：融合激活稀疏性与FP4量化以提升大语言模型推理效率

    SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference

    [https://arxiv.org/abs/2606.26587](https://arxiv.org/abs/2606.26587)

    提出SharQ方法，通过在线稀疏-稠密分解与残差补偿机制，有效结合了激活稀疏性与FP4量化，在无需训练的情况下降低了大语言模型推理中的激活压缩损失。

    

    arXiv:2606.26587v1 公告类型：交叉 摘要：现代加速器日益支持低位浮点格式与半结构化稀疏性，但将二者结合用于大语言模型激活压缩仍面临挑战：激活值包含输入相关的异常值，这些异常值主导了FP4量化中的块缩放因子；而直接应用N:M稀疏掩码会丢弃中等大小的激活值，导致稀疏化损失与量化误差相互耦合。为此，我们提出SharQ——一种无需训练的推理方法，通过在线稀疏-稠密分解来融合激活稀疏性与FP4量化。对于每个激活张量，SharQ生成输入自适应的N:M掩码，提取出以异常值主导的稀疏主干，将其量化为FP4，并定义一个相对于量化后稀疏主干（而非未量化的稀疏值）的稠密残差。一个稀疏FP4通用矩阵乘处理主干，而一个稠密FP4通用矩阵乘则同时补偿掩码导致的激活损失与稀疏路径的量化误差。

    arXiv:2606.26587v1 Announce Type: cross  Abstract: Low-bit floating-point formats and semi-structured sparsity are increasingly supported by modern accelerators, yet combining them for LLM activation compression remains challenging: activations contain input-dependent outliers that dominate block scales in FP4 quantization, and directly applying N:M sparsity masks discards moderate values, coupling sparsification loss with quantization error. We introduce SharQ, a training-free inference method that bridges activation sparsity and FP4 quantization through an online sparse--dense decomposition. For each activation tensor, SharQ generates an input-adaptive N:M mask to extract an outlier-dominated sparse backbone, quantizes it to FP4, and defines a dense residual relative to the quantized sparse backbone rather than the unquantized sparse values. A sparse FP4 GEMM processes the backbone while a dense FP4 GEMM compensates for both mask-induced activation loss and sparse-path quantization e
    
[^300]: 具有结构保证的离散选择模型中嵌入基础模型预测

    Embedding Foundation Model Predictions in Discrete-Choice Models with Structural Guarantees

    [https://arxiv.org/abs/2606.26432](https://arxiv.org/abs/2606.26432)

    本文提出一种两阶段适配器方法，通过将基础模型预测嵌入多项Logit模型，在保留结构保证的同时修正经济逻辑冲突，使时间价值等指标成为数学保证。

    

    arXiv:2606.26432v1 公告类型：新 摘要：表格基础模型在选择预测任务上表现出较高的准确性，但其预测结果常常违背这些任务所需的经济逻辑：提高价格可能导致预测需求增加，隐含的支付意愿估计值经常为负或不合理，且不可用的替代方案获得非零概率。我们提出了一种两阶段适配器，将基础模型预测的选择概率作为预计算特征，并嵌入到多项Logit的效用函数中。在第一阶段，我们通过带符号约束的最大似然估计拟合多项Logit的结构系数；在第二阶段，我们冻结这些系数，并拟合一个基于基础模型预测的小型神经网络修正项。我们证明这种组合精确保留了多项Logit的边际替代率，因此可分析计算的时间价值成为数学保证而非经验准确性。

    arXiv:2606.26432v1 Announce Type: new  Abstract: Tabular foundation models achieve strong accuracy on choice prediction tasks, but their predictions often violate the economic logic those tasks require: raising a price can increase predicted demand, implied willingness-to-pay estimates are frequently negative or implausible, and unavailable alternatives receive nonzero probability. We propose a two-stage adapter that takes a foundation model's predicted choice probabilities as a precomputed feature and embeds them inside a multinomial logit's utility. In Stage 1, we fit the multinomial logit's structural coefficients by maximum likelihood with sign constraints; in Stage 2, we freeze those coefficients and fit a small neural correction operating on the foundation model's predictions. We prove that this composition exactly preserves the multinomial logit's marginal rate of substitution, so analytically computable value-of-time becomes a mathematical guarantee rather than an empirical acc
    
[^301]: 面向自进化世界模型的自主视频生成

    Autonomous Video Generation for Self-Evolving World Models

    [https://arxiv.org/abs/2606.24152](https://arxiv.org/abs/2606.24152)

    本文提出“自主视频生成”概念，以反事实可控性（生成干预条件下的未来并将其绑定到具身智能体）作为评估标准，使视频生成世界模型能够通过环境反馈实现自进化。

    

    arXiv:2606.24152v3 公告类型：replace-cross 摘要：大规模视频生成模型正日益被称为世界模型，因为它们能够从视觉数据中学习丰富的时空规律。我们认为，理想的世界模型应具备自进化的生成特性。据此，自进化世界模型可以通过一个由多种约束条件化的生成模型来实现，其未来预测通过反馈不断得到精炼。然而，仅有传统的视觉上合理的预测，并不足以判断所想象的未来对于特定具身智能体是否在物理上可执行，从而无法从环境中获取有信息量的反馈以实现自进化改进。为实现自进化世界模型，本文提出了自主视频生成的概念，并通过反事实可控性对其进行评估，即：i) 生成以干预为条件的未来场景，ii) 将这些未来帧绑定到具身……

    arXiv:2606.24152v3 Announce Type: replace-cross  Abstract: Large-scale video generation models are increasingly described as world models because they can learn rich spatiotemporal regularities from visual data. We argue that an ideal world model should benefit in a self-evolving generative character. As such, a self-evolving world model can be realized via a generative model conditioned by various constraints, whose future predictions are refined through feedback. However, traditional visually plausible predictions alone are not enough to establish whether an imagined future is physically actionable for a particular embodied agent, failing to provide informative feedback from environments for self-evolving improvement. To realize self-evolving world models, this article proposes the concept of autonomous video generation, which is evaluated through counterfactual controllability, i.e., the ability to i) generate intervention-conditioned futures, ii) bind these future frames to embodim
    
[^302]: TROPT：一个统一和推进离散文本优化的开源框架

    TROPT: An Open Framework for Unifying and Advancing Discrete Text Optimization

    [https://arxiv.org/abs/2606.23496](https://arxiv.org/abs/2606.23496)

    TROPT是首个开源框架，通过单一接口统一了离散文本优化器的执行并标准化其开发流程，解决了现有优化器分散、难以使用和难以对比的问题，降低了在红队测试、审计和可解释性等领域应用与推进离散文本优化的门槛。

    

    离散文本触发优化——即搜索当被模型读取后能够引导模型朝向特定目标的文本序列——是模型红队测试（例如大语言模型越狱攻击）、审计和可解释性研究的基础。然而，当前离散优化器的现状阻碍了它们的采用与发展。首先，现有的优化器即使有开源的，也分散在与特定模型、目标和问题领域绑定的研究代码库中。其次，优化器数量激增，每个都需要额外的工程开销才能使用或扩展，且难以进行直接的对比评估。这些问题共同提高了在现有或新领域采用优化器、以及通过新策略推动其发展的门槛。我们提出TROPT来弥补这些不足，它是首个在单一接口下统一离散优化器执行方式并标准化其开发流程的开源框架。TROPT使得端到端自定义优化变得简单……

    arXiv:2606.23496v2 Announce Type: replace  Abstract: Discrete text-trigger optimization -- searching for text sequences that, when ingested by a model, steer it toward a specified objective -- underpins model red-teaming (e.g., LLM jailbreaks), as well as auditing and interpretability. However, the current state of discrete optimizers hinders their adoption and progress. First, existing optimizers, when open-sourced at all, are scattered across research codebases tied to specific models, objectives, and problem domains. Second, optimizers proliferate, each requiring engineering overhead to use or extend, and remaining hard to compare head-to-head. Together, these raise the bar for adopting optimizers in existing or new domains, and for advancing them via new strategies. We address these gaps with TROPT, the first open-source framework that unifies discrete optimizers' execution and standardizes their development under a single interface. TROPT makes it easy to customize end-to-end opti
    
[^303]: BatchGen：一种可扩展且高效的批量推理架构

    BatchGen: An Architecture for Scalable and Efficient Batch Inference

    [https://arxiv.org/abs/2606.21712](https://arxiv.org/abs/2606.21712)

    该论文提出了“序列协程计算模型”这一批量推理新架构基础，将每个序列表示为细粒度的事件驱动协程，使运行时能够动态重组工作负载，从而在大规模GPU集群上实现更高的吞吐量和利用率。

    

    批量推理已成为AI计算的核心模式，然而现有的推理引擎仍然依赖于为交互式服务设计的执行模型。当扩展到数百万个序列时，批量工作负载暴露出两个基本需求：一是能够处理仅在运行时才会显现的序列间和序列内的极端负载变化，二是能够在大规模GPU集群上持续保持高利用率。现有系统无法满足这些需求，损失了相当大比例的可实现吞吐量。我们为批量推理引入了一种全新的架构基础：序列协程计算模型，它将每个序列表示为一个细粒度的、事件驱动的协程。该模型提供了富有表现力的原语，使运行时能够动态地重新组织工作，从而支持更大的专家级批次、缓解落后者问题、跨设备重新分配任务，即使在成本效率较低的硬件上也能保持利用率。

    arXiv:2606.21712v2 Announce Type: replace-cross  Abstract: Batch inference has become a central mode of AI computation, yet existing inference engines still rely on execution models designed for interactive serving. When scaled to millions of sequences, batch workloads reveal two fundamental requirements: the ability to handle extreme inter- and intra-sequence load variation that emerges only at runtime, and the ability to sustain high utilization across large fleets of GPUs. Existing systems fail to meet these requirements, losing substantial fractions of achievable throughput.   We introduce a new architectural foundation for batch inference: the sequence coroutine compute model, which represents each sequence as a fine-grained, event-driven coroutine. This model exposes expressive primitives that allow the runtime to reorganize work dynamically, enabling larger expert-level batches, mitigating stragglers, reallocating work across devices, and maintaining utilization even on cost-eff
    
[^304]: 通过内化进行学习

    Learning through Internalization

    [https://arxiv.org/abs/2606.20937](https://arxiv.org/abs/2606.20937)

    本文首次对神经网络的内化过程给出可证明分析：在奇偶性学习任务上，单层Transformer可先借助显式思维链（CoT）监督完成学习，再随着CoT标记逐步移除而将自回归计算内化进自身权重、实现直接计算，而该任务在无CoT监督时从数据中学习是计算上困难的，同时内化也会带来分布外性能的退化。

    

    我们研究内化过程，即基于神经网络的系统将显式计算过程吸收进自身权重的过程，以及这一过程如何促进学习。我们研究了Transformer如何通过内化思维链（CoT）标记来内化半自动机的模拟，探究了哪些类别的半自动机更难被内化，并揭示了内化的另一面，即分布外性能的逐步退化。随后，我们首次对成功的内化给出了可证明的分析：对于学习奇偶性（parity）的任务，我们证明一个简化的单层Transformer首先在显式CoT监督下学会目标，然后随着CoT标记被逐步移除而内化自回归生成过程，学会直接计算奇偶性。在没有CoT监督的情况下，该任务从数据中学习在计算上是困难的。最后，我们讨论了通过内化进行学习……（摘要原文在此处截断）

    arXiv:2606.20937v2 Announce Type: replace  Abstract: We study internalization processes, by which neural-network-based systems absorb an explicit computational procedure into their own weights, and how they facilitate learning. We investigate how transformers internalize the simulation of semiautomata by internalizing chain-of-thought (CoT) tokens, which classes of semiautomata are harder to internalize, and expose the flip side of internalization, that is, a progressive degradation of out-of-distribution performance. We then provide the first provable analysis of successful internalization: for the task of learning parities, we show that a simplified one-layer transformer provably first learns the target with explicit CoT supervision and then internalizes the autoregressive generation as CoT tokens are progressively removed, learning to directly compute the parity. This task is computationally hard to learn from data without CoT supervision. Finally, we discuss how learning through in
    
[^305]: Pose6DAug：用于机器人数据增强的物理合理多视角物体替换

    Pose6DAug: Physically Plausible Multi-view Object Swapping for Robot Data Augmentation

    [https://arxiv.org/abs/2606.20118](https://arxiv.org/abs/2606.20118)

    提出了Pose6DAug框架，通过在保留物理有效动作轨迹与多视角标定观测的前提下替换被操作物体，将已有的成功演示转化为针对策略失败模式的物理合理增强数据，无需新的数据采集即可提升VLA策略对分布外物体的泛化能力。

    

    视觉-语言-动作（VLA）策略在通用操作任务中展现出强大潜力，但在外观或几何形状偏离训练分布的新颖、分布外物体上往往会失败。标准的解决方案是为每个失败案例收集多视角遥操作数据，但这种方式在成本和时间上都难以扩展。我们提出Pose6DAug，一个由失败驱动的数据增强框架，它将现有的成功演示回合转化为针对策略失败模式的定向演示，而无需任何新的数据采集。我们的关键洞察是：每个成功的演示回合本身已经编码了一条物理上有效的动作轨迹以及经过标定的多视角观测。通过在保留该轨迹的同时仅替换被操作的物体，我们可以获得全新的、物理上合理的演示。然而，挑战在于如何合成物体替换——例如，朴素的2D视频编辑会破坏多视角一致性，并（摘要在此处截断）

    arXiv:2606.20118v3 Announce Type: replace-cross  Abstract: Vision-language-action (VLA) policies have shown strong potential for general-purpose manipulation, yet they often fail on novel, out-of-distribution objects whose appearance or geometry deviates from the training distribution. The standard remedy is to collect multi-view teleoperation data for every failure case, but this scales poorly in both cost and time. We introduce Pose6DAug, a failure-driven data augmentation framework that turns existing successful episodes into targeted demonstrations for a policy's failure modes, without any new data collection. Our key insight is that each successful episode already encodes a physically valid action trajectory together with calibrated multi-view observations. By swapping only the manipulated object while preserving this trajectory, we obtain new and physically grounded demonstrations. However, the challenge lies in synthesizing object swaps, e.g., naive 2D video editing breaks multi
    
[^306]: 基于海试数据的自主水下航行器机动建模可微复合近似框架

    A Differentiable Composite Approximation Framework for Autonomous Underwater Vehicle Maneuvering Modeling from Sea-Trial Data

    [https://arxiv.org/abs/2606.19711](https://arxiv.org/abs/2606.19711)

    本文提出一种可微复合近似框架，将多项式基与数据自适应神经基作为统一预测器的可微分量进行梯度联合校准，从而从海试数据中建立更准确的AUV机动模型。

    

    基于船上测量数据的实航建模能够产生反映真实运行特性的自主水下航行器（AUV）机动模型。从近似的角度来看，传统机动模型采用预定义的约束多项式基，而数据驱动模型采用数据自适应基。受这一基函数视角的启发，本文提出了一种可微复合近似形式，将多项式基分量和数据自适应基分量视为统一预测器中的可微部分并进行联合校准。针对全尺度AUV机动预测，本文开发了一种基于梯度的协同校准方法，其中灵敏度感知机制调节有界多项式更新，而神经残差在共享预测目标下捕获剩余的非线性差异。为了考虑实测数据中海流的影响，一种基于回转运动的……（原文摘要在此处截断）

    arXiv:2606.19711v2 Announce Type: replace-cross  Abstract: Field-based modeling from onboard measurements can produce autonomous underwater vehicle (AUV) maneuvering models that reflect real operating characteristics. From an approximation perspective, conventional maneuvering models use predefined constraint polynomial bases, whereas data-driven models use data-adaptive bases. Motivated by this basis-function view, this paper presents a differentiable composite approximation formulation, in which the polynomial basis component and the data-adaptive basis component are treated as differentiable parts of a unified predictor and calibrated jointly. A gradient-based co-calibration method is developed for full-scale AUV maneuvering prediction, where a sensitivity-aware mechanism regulates bounded polynomial updates while the neural residual captures remaining nonlinear discrepancies under a shared prediction objective. To account for ocean-current effects in field data, a turning-motion-ba
    
[^307]: DLWM：面向高效多模态推理的多样化潜在世界模型

    DLWM: Diverse Latent World Models for Efficient Multimodal Reasoning

    [https://arxiv.org/abs/2606.15160](https://arxiv.org/abs/2606.15160)

    DLWM提出了一种将潜空间推理与强化学习相结合的多模态推理框架，通过构建多样化的潜在世界模型来应对视觉输入的多种合理解释问题，摆脱单一解释和固定推理路径的限制，从而实现更高效的多模态推理。

    

    近年来，多模态大语言模型（MLLMs）的推理能力得到了显著提升。现有方法通常依赖显式思维链或连续潜空间轨迹来增强多步推理。然而，这些方法普遍假设输入只有单一的潜在解释，并沿固定路径或在统一计算预算下展开推理。在真实世界的多模态场景中，视觉观测常常受到遮挡、模糊、视角变化或语义歧义的影响，从而产生多种合理的解释。统一的推理策略不仅限制了模型探索多个假设的能力，还会带来较高的内存占用和推理成本。我们提出了DLWM（多样化潜在世界模型），这是一个将潜空间推理与强化学习相结合的多模态推理框架。首先，我们构建了一组多样化的潜在……（原文摘要在此处截断）

    arXiv:2606.15160v2 Announce Type: replace-cross  Abstract: Reasoning capabilities of multimodal large language models (MLLMs) have improved considerably in recent years. Existing approaches typically rely on explicit chain-of-thought or continuous latent-space trajectories to enhance multi-step reasoning. However, these methods generally assume that an input admits a single latent interpretation and unfold reasoning along a fixed path or under a uniform computation budget. In real-world multimodal settings, visual observations are often subject to occlusion, blur, viewpoint variation, or semantic ambiguity, giving rise to multiple plausible interpretations. A uniform reasoning strategy not only limits the model's ability to explore multiple hypotheses but also incurs high memory usage and rollout cost. We present DLWM (Diverse Latent World Models), a multimodal reasoning framework that combines latent-space reasoning with reinforcement learning. First, we construct a set of diverse lat
    
[^308]: 有理稀疏自编码器

    Rational Sparse Autoencoder

    [https://arxiv.org/abs/2606.14990](https://arxiv.org/abs/2606.14990)

    提出有理稀疏自编码器（RSAE），用可训练的有理函数取代固定的编码器激活函数，在匹配期望稀疏度下从理论上保证不劣于 ReLU，并在给定假设下可证明优于 JumpReLU 和 TopK。

    

    稀疏自编码器（SAE）是机制可解释性研究的标准工具，但当前的 SAE 家族受到固定编码器非线性激活函数（如 ReLU、JumpReLU 和 TopK）的约束。这种做法将特定的稀疏机制硬编码进模型，并可能扭曲重构与稀疏性之间的权衡。我们提出了有理稀疏自编码器（RSAE），它用一个可训练的有理函数取代固定的编码器激活函数。该设计的动机源于我们在匹配期望稀疏度下为 RSAE 证明的重构优势：保证不劣于 ReLU，并且在给定假设下，可证明优于 JumpReLU 和 TopK。该模型通过两阶段流水线实现：首先是一个初始化过程，复制预训练的基线 SAE 权重，并对尺度参数和有理系数进行校准；随后在稀疏正则化的（微调步骤中进行优化，原文摘要在此处截断）。

    arXiv:2606.14990v3 Announce Type: replace-cross  Abstract: Sparse autoencoders (SAEs) are standard tools for mechanistic interpretability, but current SAE families are constrained by fixed encoder nonlinearities such as ReLU, JumpReLU, and TopK. This hard-codes a particular sparsity mechanism into the model and can distort the reconstruction-versus-sparsity trade-off. We introduce the Rational Sparse Autoencoder (RSAE), which replaces the fixed encoder activation with a trainable rational function. The design is motivated by a reconstruction advantage that we prove for RSAE at matched expected sparsity, with guaranteed non-inferiority to ReLU and, under stated assumptions, provable improvements over JumpReLU and TopK. The model is then implemented through a two-stage pipeline: an initialization procedure that copies the pre-trained baseline SAE weights and calibrates the scale parameters along with the rational coefficients; followed by a fine-tuning step under a sparsity-regularized r
    
[^309]: FedRAN：准确且高效的分析式联邦持续学习

    FedRAN: Accurate and Efficient Analytic Federated Continual Learning

    [https://arxiv.org/abs/2606.11480](https://arxiv.org/abs/2606.11480)

    FedRAN提出了一种低秩分析式联邦持续学习框架，通过跨客户端与任务合并的谱摘要直接压缩二阶结构，并利用保留的秩显式控制近似精度与通信开销之间的权衡，实现准确且高效的联邦持续学习。

    

    联邦持续学习（FCL）使分布式客户端能够从不断演进的数据流中学习，同时保留先前所学任务的知识。现有的基于优化的FCL方法通常依赖重复的本地训练，会带来通信开销以及客户端之间的特征漂移。分析式方法虽然避免了迭代训练，但用于提升精度的高维随机特征需要传输一个Gram矩阵，其通信成本随特征维度呈二次方增长。通信高效的替代方案通过从一阶统计量统计重建Gram矩阵来降低这一成本。我们提出了FedRAN，一个低秩分析式FCL框架，它通过跨客户端与任务合并的谱摘要直接压缩所需的二阶结构。所保留的秩显式地控制了近似精度与通信之间的权衡，有效降低了二阶结构的通信成本（摘要在此处截断）。

    arXiv:2606.11480v2 Announce Type: replace  Abstract: Federated continual learning (FCL) enables distributed clients to learn from evolving data streams while retaining knowledge from previously observed tasks. Existing optimization-based FCL methods often rely on repeated local training, incurring communication overhead and feature drift across clients. Analytic alternatives avoid iterative training, but high-dimensional random features used to improve accuracy require communicating a Gram matrix whose cost grows quadratically with feature dimension. Communication-efficient alternatives reduce this cost by statistically reconstructing the Gram matrix from first-order summaries. We propose FedRAN, a low-rank analytic FCL framework that instead compresses the required second-order structure directly through spectral summaries that are merged across clients and tasks. The retained rank explicitly controls the tradeoff between approximation fidelity and communication, reducing second-order
    
[^310]: 迪利克雷引导的群体预测：缓解时间序列预测中的过度平滑问题

    Dirichlet-Guided Group Forecasting for Alleviating Over-smoothing in Time Series Forecasting

    [https://arxiv.org/abs/2606.10592](https://arxiv.org/abs/2606.10592)

    提出迪利克雷引导的群体预测（DGF）框架，通过显式建模多个模式条件下的预测分布及其选择概率的不确定性，并借助迪利克雷引导的分层采样与基于奖励的优化，有效缓解时间序列预测中的过度平滑问题。

    

    时间序列预测常常受到过度平滑问题的困扰，尤其是当未来动态具有多模态特性时。预测结果可能只遵循观测未来的粗略趋势，但无法保留尖锐变化、振荡、转折点和状态切换等刻画合理动态演化的特征。在这项工作中，我们从潜在动态模式压缩的角度重新审视过度平滑问题：在部分观测和单次实现监督的条件下，多个可能的未来模式可能在预测过程中被削弱、合并或平均。基于这一观点，我们提出了迪利克雷引导的群体预测（DGF），这是一种模式保持的预测框架，它显式地建模多个以模式为条件的预测分布以及模式选择概率上的不确定性。DGF使用迪利克雷引导的分层采样机制和基于奖励的优化，以鼓励生成准确、动态一致的预测结果……

    arXiv:2606.10592v2 Announce Type: replace  Abstract: Time series forecasting often suffers from over-smoothing, especially when future dynamics are multi-modal. Forecasts may follow the coarse trend of the observed future, but fail to preserve sharp changes, oscillations, turning points, and regime transitions that define plausible dynamic evolution. In this work, we revisit over-smoothing from the perspective of latent dynamical mode compression: under partial observation and single-realization supervision, multiple plausible future modes can be weakened, merged, or averaged during forecasting. Based on this view, we propose Dirichlet-Guided Group Forecasting (DGF), a mode-preserving forecasting framework that explicitly models multiple mode-conditioned predictive distributions and uncertainty over their selection probabilities. DGF uses a Dirichlet-guided hierarchical sampling mechanism and reward-based optimization to encourage forecasts that are accurate, dynamically consistent, an
    
[^311]: 衡量真正重要的东西：概念瓶颈模型的合成基准

    Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models

    [https://arxiv.org/abs/2606.04326](https://arxiv.org/abs/2606.04326)

    本文提出了概念瓶颈模型的合成基准，使研究人员能够控制数据模态、概念选择和标注质量等因素，从而系统地评估该类模型在决策支持与自动化场景中的表现，并识别出新的失败模式。

    

    概念瓶颈模型利用从输入中检测到的高级概念进行预测。尽管这些模型提供了一种从可解释性中获益的简单方式，但由于现实世界中包含概念标签的数据集非常少，它们难以训练和评估。在本文中，我们为概念瓶颈模型开发了合成基准。借助我们的基准，研究人员可以生成带标签的数据集，并控制影响性能的属性，例如数据模态、概念的选择或标注质量。因此，他们可以找到适合这些模型的问题，并分离出导致其失败的因素。我们展示了如何使用我们的基准来评估概念瓶颈模型的主要应用场景：决策支持（即帮助人类专家做出更好的决策）和自动化（即在无监督情况下处理常规任务）。我们的演示展示了这些基准如何识别新的失败模式和归因问题。

    arXiv:2606.04326v2 Announce Type: replace-cross  Abstract: Concept bottleneck models predict using high-level concepts detected in inputs. Although these models provide a simple way to reap benefits from interpretability, they are hard to train and evaluate because very few real-world datasets include concept labels. In this paper, we develop synthetic benchmarks for concept bottleneck models. With our benchmarks, researchers can generate labeled datasets and control properties that affect performance, such as data modality, the choice of concepts, or annotation quality. They can thus find problems that suit these models and isolate the factors behind their failures. We demonstrate how our benchmarks can be used to evaluate major use cases for concept bottleneck models: decision support, i.e., helping human experts make better decisions; and automation, i.e., handling routine tasks without supervision. Our demonstrations show how the benchmarks can identify new failure modes and attrib
    
[^312]: CaMBRAIN：基于因果状态空间模型的实时连续脑电（EEG）推理

    CaMBRAIN: Real-time, Continuous EEG Inference with Causal State Space Models

    [https://arxiv.org/abs/2605.28792](https://arxiv.org/abs/2605.28792)

    提出了首个基于Mamba的因果状态空间模型CaMBRAIN，实现了脑电信号的实时连续推理，克服了注意力机制的二次方复杂度和滑动窗口处理无法全局理解信号的局限。

    

    脑电图（EEG）是一种监测大脑电活动的关键无创方法。脑电信号的时长跨度可以从几秒到数小时不等，这对现有深度学习方法构成了重大挑战，主要原因有二：（1）现有脑电模型主要建立在注意力机制之上，随着序列长度增加会产生二次方的计算复杂度；（2）由于固定长度输入的要求，原始脑电信号必须以滑动窗口的方式处理，导致无法对整个信号进行全局理解。为此，我们提出了CaMBRAIN——首个能够对脑电信号进行实时推理的因果（Causal）、基于Mamba的状态空间模型（SSM）。我们认为，鉴于脑电信号固有的因果性、单向性本质，双向建模方法的代价是不必要的。然而，训练这样的模型并非易事，因为关键的脑电事件可能极其短暂——不足一秒——却彼此之间被……（摘要原文在此处截断）

    arXiv:2605.28792v2 Announce Type: replace  Abstract: Electroencephalography (EEG) is a critical, non-invasive method to monitor electrical brain activity. EEGs can span anywhere from a couple seconds to multiple hours, posing a major hurdle for existing deep learning methods due to two major factors: (1) existing EEG models are predominantly built upon the attention mechanism, incurring quadratic scaling as the sequence length increases, and (2) raw EEG signals must be processed in a sliding-window fashion due to fixed-length input requirements, preventing global understanding of the entire signal. To this extent, we propose CaMBRAIN - the first Causal, Mamba-based state space model (SSM) capable of real-time inference of EEG signals, arguing that bidirectional approaches are needlessly expensive given the causal, unidirectional nature of EEG. However, training such a model is non-trivial, as crucial EEG events can be extremely brief - within fractions of a second - yet separated by lo
    
[^313]: 通过填充从扩散语言模型中提取训练数据

    Extracting Training Data from Diffusion Language Models via Infilling

    [https://arxiv.org/abs/2605.24173](https://arxiv.org/abs/2605.24173)

    该论文提出“填充提取”协议，利用任意二进制掩码全面评估扩散语言模型的训练数据泄露风险，发现传统的前缀探测方法会显著低估记忆化风险，而边缘条件掩码可提取多达三倍的逐字训练序列。

    

    大型语言模型中的记忆化问题几乎一直仅通过前缀条件提取来研究，这是自回归模型的一种自然选择。然而，扩散语言模型可以在任意位置对被掩码的标记进行去噪。因此，仅使用前缀探测只能揭示扩散语言模型中记忆化的一个侧面，并显著低估了训练数据提取的风险。为了更真实地建模扩散语言模型中训练数据的可提取性，我们引入了填充提取，这是一种由任意二进制掩码参数化的数据提取协议，它涵盖了仅前缀探测方法，并考虑了扩散语言模型的双向归纳偏置。我们在LLaDA-8B和Dream-7B上实例化该方法，涵盖五种提取模式、三种训练流程和三个语料库，包括逐字泄露和部分泄露的情况。我们发现掩码的几何结构决定了可提取性：边缘条件掩码提取的逐字序列最多可达三倍。

    arXiv:2605.24173v2 Announce Type: replace-cross  Abstract: Memorization in large language models has been studied almost exclusively through prefix-conditioned extraction, a natural choice for autoregressive models. However, diffusion language models (DLMs) can denoise masked tokens at arbitrary positions. Thus, prefix-only probing reveals only one facet of memorization in DLMs and significantly underestimates the risk of training-data extraction. In order to realistically model extractability of training data in DLMs, we introduce infilling extraction, a data-extraction protocol parameterized by an arbitrary binary mask that subsumes prefix-only probing and accounts for the bidirectional inductive bias of DLMs. Instantiating it on LLaDA-8B and Dream-7B across five extraction modes, three training pipelines, and three corpora covering verbatim and partial leakage, we find that mask geometry governs extractability: edge-conditioned masks extract up to three times more verbatim sequences
    
[^314]: GRTO：用于指代分割的自举式组相对工具优化

    GRTO: Bootstrapped Group Relative Tool Optimization for Referring Segmentation

    [https://arxiv.org/abs/2605.23500](https://arxiv.org/abs/2605.23500)

    提出GRTO，一个基于数学原理的两阶段框架，首次将推理策略与可微分割工具联合优化，解决现有方法只优化VLM、在域偏移下分割错误损害正确推理的问题。

    

    指代分割需要通过推理将语言与视觉证据相联系，并在具有挑战性的场景中识别出预期目标。这使得GRPO成为将视觉语言模型（VLM）与分割工具联合训练的自然选择。然而，现有方法仅优化VLM，假设分割工具是完美的而不对其加以改动。当工具面临域偏移时，这一假设的局限性尤为明显，因为分割错误甚至会破坏原本正确的推理。因此，我们认为分割工具应当与推理策略一同进行优化。然而，这颇具挑战性，因为策略优化与可微工具学习依赖于本质上不同的目标。在本工作中，我们提出了组相对工具优化（GRTO），一个具有数学理论基础的两阶段框架，用于联合优化策略及其可微工具。其核心思想是首先使工具与策略（原文摘要在此处截断）

    arXiv:2605.23500v3 Announce Type: replace-cross  Abstract: Referring segmentation requires reasoning to connect language with visual evidence and identify the intended target in challenging scenarios. This makes GRPO a natural choice for training VLMs together with a segmentation tool. However, existing approaches optimize only the VLM, assuming that the tool is perfect and leaving it untouched. This assumption is particularly limiting when the tool faces domain shift, as segmentation errors can undermine even correct reasoning. We therefore argue that the segmentation tool should be optimized together with the reasoning policy. However, this is challenging because policy optimization and differentiable tool learning rely on fundamentally different objectives. In this work, we introduce group relative tool optimization (GRTO), a mathematically grounded two-stage framework for jointly optimizing a policy and its differentiable tool. The key idea is to first align the tool with the polic
    
[^315]: 先验知识还是搜索？LLM智能体在硬件感知代码优化中的研究

    Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization

    [https://arxiv.org/abs/2605.19782](https://arxiv.org/abs/2605.19782)

    该研究通过三个受控实验揭示，LLM智能体在硬件感知代码优化中依赖先验知识而非搜索——它在黑盒优化中表现为贪婪优化器，在零样本内核生成中会完全忽略输入尺寸信息，且面对不常见的内核尺寸时性能急剧下降。

    

    LLM发现与优化系统正被日益广泛地应用于各个领域，它们实现了一种通用的“提出-评估-修改”循环。这类优化或发现过程通过根据环境中收到的反馈对上下文进行条件化而向前推进。然而，随着现代LLM智能体结构日益复杂，很难评估哪些组件贡献最大，以及这种探索会在何时、以何种方式失败。我们通过三个受控实验回答了这些问题。我们的发现如下：(1) 在纯黑盒优化中，LLM表现得像贪婪优化器。(2) 在零样本内核生成中，提供显式的输入尺寸信息没有可测量的效果——无论尺寸或温度如何，模型都收敛到相同的内核参数，仿佛尺寸指令是不可见的。此外，当被要求对不常见的内核尺寸进行优化时，无论使用何种语言，性能都会急剧下降。(3

    arXiv:2605.19782v2 Announce Type: replace  Abstract: LLM discovery and optimization systems are increasingly applied across domains, implementing a common propose-evaluate-revise loop. Such optimization or discovery progresses via context conditioning on received feedback from an environment. However, as modern LLM agents are increasingly complex in their structure, it is difficult to evaluate which components contribute the most, and when and how this exploration may fail. We answer these questions through three controlled experiments. Our findings: (1) In pure black-box optimization, LLMs act as greedy optimizers. (2) In zero-shot kernel generation, providing explicit input-size information has no measurable effect, models converge to the same kernel parameters regardless of size or temperature, as though the size instruction were invisible. Moreover, when tasked to perform kernel optimization for uncommon kernel sizes, performance sharply degrades regardless of the language used. (3
    
[^316]: 沉默的超参数：量化推理后端对大语言模型可复现性的影响

    The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM Reproducibility

    [https://arxiv.org/abs/2605.19537](https://arxiv.org/abs/2605.19537)

    该论文通过调研200个推理引擎和分析35,000篇ML论文，首次系统量化了推理后端这一常被忽视的“沉默超参数”对大语言模型基准测试结果与可复现性的影响。

    

    arXiv:2605.19537v4 公告类型：替换 摘要：大语言模型的进展越来越多地通过标准化基准测试来衡量，其中最先进水平的提升往往仅相差百分之零点几。与此同时，评估现代大语言模型的高昂计算成本推动了专用推理后端的广泛采用——这类软件系统可在推理阶段高效地执行已训练的模型。尽管系统级优化（如自定义CUDA内核和低精度算术）对可扩展性至关重要，但它们可能改变token概率并引入非确定性，进而可能导致生成结果出现分歧。在这项工作中，我们首先调研了推理领域的现状，识别出200个不同的推理引擎，并分析了35,000篇机器学习出版物，发现尽管推理后端种类繁多，但具体的推理技术栈却很少被报告。随后，我们对推理后端如何影响大语言模型基准测试结果进行了系统的实证研究，在保持模型……

    arXiv:2605.19537v4 Announce Type: replace  Abstract: Progress in LLMs is increasingly measured through standardized benchmarks, where state-of-the-art improvements are often separated by fractions of a percentage point. At the same time, the computational cost of evaluating modern LLMs has driven widespread adoption of specialized inference backends, software systems that execute trained models efficiently at inference time. While critical for scalability, system-level optimizations, such as custom CUDA kernels and reduced-precision arithmetic, can alter token probabilities and introduce non-determinism, possibly cascading into divergent generation. In this work, we first survey the inference landscape, identifying $200$ distinct engines, and analyze $35,000$ ML publications, finding that the specific inference stack is rarely reported despite this widespread diversity. We then present a systematic empirical study of how inference backends affect LLM benchmark results. Holding model we
    
[^317]: KVBuffer：面向线性注意力的IO感知服务系统

    KVBuffer: IO-aware Serving for Linear Attention

    [https://arxiv.org/abs/2605.19049](https://arxiv.org/abs/2605.19049)

    KVBuffer是一种面向线性注意力的IO感知服务机制，通过缓冲最近的键和值实现分块计算与批量状态更新，显著减少内存访问量并降低解码延迟，同时高效支持投机解码验证。

    

    线性注意力近来因其在长上下文推理中相对于上下文长度具有恒定的解码成本而受到广泛关注。然而，现有的服务系统通常通过在每个解码步骤中递归地计算和更新一个庞大的线性注意力状态来提供线性注意力服务。由于该状态远大于每个token的键和值，递归解码会导致大量的内存访问，使得线性注意力的服务效率低下。在本文中，我们提出了KVBuffer，一种面向线性注意力的IO感知服务机制。通过缓冲最近的键和值，KVBuffer使服务系统能够以更灵活、更节省内存的方式计算线性注意力输出。对于解码，KVBuffer支持分块计算，通过延迟状态更新并将其批量应用，降低了平均内存访问量和解码延迟。对于投机解码，KVBuffer通过缓存机制高效验证草稿token

    arXiv:2605.19049v2 Announce Type: replace-cross  Abstract: Linear attention has recently gained significant attention for long-context inference due to its constant decoding cost with respect to context length. However, existing serving systems typically serve linear attention by recurrently computing and updating a large linear attention state in every decoding step. Since the state is much larger than the per-token key and value, recurrent decoding incurs substantial memory access and becomes inefficient for serving linear attention. In this paper, we propose KVBuffer, an IO-aware serving mechanism for linear attention. By buffering recent keys and values, KVBuffer enables serving systems to compute linear attention outputs in more flexible and memory-efficient ways. For decoding, KVBuffer enables chunkwise computation, which reduces average memory access and decoding latency by deferring state updates and applying them in batch. For speculative decoding, KVBuffer verifies draft toke
    
[^318]: 约束潜在状态建模：竞争约束下表示学习的统一视角

    Constrained latent state modeling: A unifying perspective on representation learning under competing constraints

    [https://arxiv.org/abs/2605.15995](https://arxiv.org/abs/2605.15995)

    该论文提出约束潜在状态建模（CLSM）这一统一概念框架，用预测充分性、最小性、时间一致性、观测兼容性、抗干扰不变性和结构约束六个互补特性来刻画潜在表示，从而统一解释和比较各类表示学习方法，并阐明约束组合如何提升表示的可辨识性。

    

    arXiv:2605.15995v3 公告类型：替换 摘要：从时间序列、多模态和部分观测数据中学习潜在表示，需要明确潜在状态应当保留、舍弃和组织哪些信息。现有方法通过各式各样的异质目标函数来编码这些需求，导致不同方法难以相互比较，学习到的表示也难以解释。我们提出约束潜在状态建模（CLSM），这是一个概念性框架，通过六个互补的特性来刻画潜在状态：预测充分性、最小性、时间一致性、观测兼容性、对干扰因素的不变性以及结构约束。CLSM 将这些特性与用于诱导它们的替代目标以及用于评估它们的诊断方法分离开来，并阐明了约束的组合如何通过限制可容许表示的空间来提升可辨识性。我们据此重新解释了主要的表示学习家族……（原文摘要在此处被截断）

    arXiv:2605.15995v3 Announce Type: replace  Abstract: Learning latent representations from temporal, multimodal, and partially observed data requires specifying what information a latent state should retain, discard, and organize. Existing approaches encode these requirements through heterogeneous objectives, making methods difficult to compare and learned representations difficult to interpret. We propose Constrained Latent State Modeling (CLSM), a conceptual framework that characterizes latent states through six complementary properties: predictive sufficiency, minimality, temporal coherence, observation compatibility, invariance to nuisance factors, and structural constraints. CLSM separates these properties from the surrogate objectives used to induce them and from the diagnostics used to evaluate them, and clarifies how combinations of constraints can improve identifiability by restricting the space of admissible representations. We reinterpret major representation-learning familie
    
[^319]: 从数据到行动：利用人工智能加速炼油厂优化

    From Data to Action: Accelerating Refinery Optimization with AI

    [https://arxiv.org/abs/2605.15085](https://arxiv.org/abs/2605.15085)

    本文通过为线性规划求解器增加记忆功能并引入异常检测工具，能够发现炼油厂优化中输入数据的潜在错误及不同决策层面的异常关系，从而提升后续LP求解结果的可信度和质量。

    

    当今的炼油厂优化需要利用海量数据，这些数据虽然可以由现代线性规划（LP）软件处理，但对结果的解释和应用仍然具有挑战性。大型石化公司使用的模型规模庞大，输入矩阵包含数十万个元素。LP解在数学上是正确的，但模型构建中做了简化，且数据供给可能出现错误。因此，需要进一步的洞察才能信任这些结果。本文的创新之处在于为LP求解器增加了记忆功能，并引入异常检测工具，从而能够发现输入数据中的潜在错误，并找出不同决策层面数据中的异常关系。通过分析LP输入和输出中潜在的“异常”，可以改进后续的LP求解结果。所提出的异常检测工具包含三种方法。（注：原文摘要在此处被截断）

    arXiv:2605.15085v2 Announce Type: replace-cross  Abstract: Nowadays refinery optimization utilizes sheer amounts of data, which can be handled with modern Linear Programming (LP) software, but the interpreting and applying the results remains challenging. Large petrochemical companies use massive models, with hundreds of thousands of input matrix elements. The LP solution is mathematically correct, but simplifications are made in the model, and data supply errors may occur. Therefore, further insight is needed to trust the results. The novelty of the paper consists of enhancing the LP solver with a memory, which makes possible to find potential errors in the input data, finding unusual relations in the data on different decision levels, by introducing an anomaly detection tool. Analyzing the potential "anomalies" in the input and output of LP, the subsequent LP solutions can be improved. The Anomaly Detection tool proposed comprises three methodologies. A transformed version of the pop
    
[^320]: REALM：面向局部场电位建模的回溯编码器对齐

    REALM: Retrospective Encoder Alignment for LFP Modeling

    [https://arxiv.org/abs/2605.14867](https://arxiv.org/abs/2605.14867)

    本文提出REALM框架，通过回溯知识蒸馏将非因果模型的表达知识迁移到因果模型中，从而实现高精度且可实时部署的LFP行为解码。

    

    尖峰活动一直是行为解码的主要神经信号，因为其高时空分辨率支持精确解码。然而，随着皮层内脑机接口（iBCI）向更高通道数和无线操作方向发展，记录尖峰所需的高采样率带来了巨大的功耗和带宽需求。局部场电位（LFP）提供了互补的优势，包括更强的长期稳定性、更低的能耗和更低的带宽需求。然而，基于LFP的解码器通常精度较低，且依赖无法直接用于实时部署的非因果架构。我们提出了REALM，一个用于因果LFP行为解码的回溯知识蒸馏（RKD）框架。受语音识别中离线到在线蒸馏的启发，REALM将预训练的多会话双向（摘要在此处被截断）

    arXiv:2605.14867v2 Announce Type: replace-cross  Abstract: Spike activity has been the dominant neural signal for behavior decoding because its high spatiotemporal resolution supports accurate decoding. However, as intracortical brain-computer interfaces (iBCIs) move toward higher channel counts and wireless operation, the high sampling rates required to record spikes create substantial power and bandwidth demands. Local field potentials (LFPs) offer complementary advantages, including greater long-term stability, lower energy consumption, and lower bandwidth requirements. However, LFP-based decoders often achieve lower accuracy and rely on non-causal architectures that cannot be used directly for real-time deployment. We propose REALM, a retrospective knowledge distillation (RKD) framework for causal LFP behavior decoding. Inspired by offline-to-online distillation in speech recognition, REALM transfers non-causal representational knowledge from a pretrained, multi-session bidirection
    
[^321]: 人机对齐条件下学习借助AI辅助进行决策

    Learning to Decide with AI Assistance under Human-Alignment

    [https://arxiv.org/abs/2605.12646](https://arxiv.org/abs/2605.12646)

    本文揭示了AI置信度与人类对自身预测的置信度之间的对齐程度如何影响AI辅助决策中通过反复交互学习最优决策的复杂度，并将该学习问题归约为双臂老虎机问题。

    

    人们普遍认为，当AI模型通过预测感兴趣的结果来辅助高风险领域的决策者时，它应当传达其预测的置信度。然而，实证证据表明，决策者往往难以仅凭所传达的置信度来判断何时应当信任某个预测。在这一背景下，近期的理论与实证研究指出，AI辅助决策的效用与AI置信度和决策者对自身预测置信度之间的对齐程度呈正相关。至关重要的是，这些发现尚未阐明这种对齐究竟在多大程度上影响通过反复交互学习最优决策的复杂度。本文在二元预测与二元决策这一经典情形下研究该问题。我们首先证明该问题等价于一个双臂（老虎机）问题。

    arXiv:2605.12646v3 Announce Type: replace-cross  Abstract: It is widely agreed that when AI models assist decision-makers in high-stakes domains by predicting an outcome of interest, they should communicate the confidence of their predictions. However, empirical evidence suggests that decision-makers often struggle to determine when to trust a prediction based solely on this communicated confidence. In this context, recent theoretical and empirical work suggests a positive correlation between the utility of AI-assisted decision-making and the degree of alignment between the AI confidence and the decision-makers' confidence in their own predictions. Crucially, these findings do not yet elucidate the extent to which this alignment influences the complexity of learning to make optimal decisions through repeated interactions. In this paper, we address this question in the canonical case of binary predictions and binary decisions. We first show that this problem is equivalent to a two-armed
    
[^322]: MIST：基于McDiarmid界的在线类别增量学习可靠流式决策树

    MIST: Reliable Streaming Decision Trees for Online Class-Incremental Learning via McDiarmid Bound

    [https://arxiv.org/abs/2605.11617](https://arxiv.org/abs/2605.11617)

    该论文提出MIST框架，利用McDiarmid界构造与类别数量无关的紧凑分裂准则，并解决分裂时的知识迁移缺失问题，从而使流式决策树能够可靠地应用于在线类别增量持续学习。

    

    流式决策树是开放世界持续学习的天然候选者，因为它们执行局部更新、内存占用有界且决策边界保持静态。尽管具备这些优势，它们在在线类别增量学习中仍然会失败，原因在于两个相互耦合的失准问题：(i) 随着类别数量K的扩大，其分裂准则变得不可靠；(ii) 分裂时缺乏知识迁移。这两种失败有着共同的根源：信息增益的取值范围本质上随log2 K缩放。因此，任何由此推导出的Hoeffding式置信半径都不可避免地随类别数量增长，使得与K无关的分裂准则在结构上无法实现，从而消除了将流式决策树应用于持续学习的潜在优势。为解决这一问题，我们提出了MIST（McDiarmid增量流式树），它通过三个集成的组件来解决上述两个失败：(i) 一个紧凑的、与K无关的McDiarmid界（摘要在此处被截断）

    arXiv:2605.11617v3 Announce Type: replace  Abstract: Streaming decision trees are natural candidates for open-world continual learning, as they perform local updates, enjoy bounded memory, and static decision boundaries. Despite these, they still fail in online class-incremental learning due to two coupled miscalibrations: (i) their split criterion grows unreliable as the class count K expands, and (ii) the absence of knowledge transfer at split time. Both failures share a common root: the range of Information Gain intrinsically scales with log2 K. Consequently, any Hoeffding-style confidence radius derived from it must inevitably grow with the class count, making a K-independent split criterion structurally impossible, taking away the potential benefits of applying streaming decision trees to continual learning. To fix this issue, we present MIST (McDiarmid Incremental Streaming Tree), which resolves both failures through three integrated components: (i) a tight, K-independent McDiarm
    
[^323]: 最小神经网络权重范数 = 柯尔莫哥洛夫复杂度

    Minimum Neural Weight Norm = Kolmogorov Complexity

    [https://arxiv.org/abs/2605.10878](https://arxiv.org/abs/2605.10878)

    该论文证明了循环神经网络计算某函数所需的最小权重范数与该函数的柯尔莫哥洛夫复杂度除以其对数渐近等价，从而在权重范数最小化与算法简单性之间建立了理论联系，为解释神经网络的泛化能力提供了新途径。

    

    为什么神经网络在参数范数较小时泛化能力更好？我们证明，在具有常数精度权重的循环神经网络中，计算某个函数的网络所需的最小权重范数与该函数的柯尔莫哥洛夫复杂度除以其对数在渐近意义上等价。该证明包含两个归约：一方面，循环神经网络可以通过稀疏模式以每个非零参数编码对数级比特的速率编码信息；另一方面，任何固定精度网络都可以通过枚举其非零参数并以对数级寻址成本来描述。两个界的首项常数因子相同，从而给出了渐近等价性。更广泛地说，通过在范数最小化与算法简单性之间建立理论联系，我们为理解现代深度学习中的泛化问题提供了一条新途径。

    arXiv:2605.10878v2 Announce Type: replace  Abstract: Why do neural networks generalize better when their parameter norm is small? We prove that, in looped neural networks with constant-precision weights, the smallest weight norm of a network computing a function is asymptotically equivalent to the Kolmogorov complexity of that function, divided by its logarithm. The proof has two reductions: looped neural networks can encode information at a rate of logarithmically many bits per non-zero parameter encoded via the sparsity pattern, while any fixed-precision network can be described by enumerating its non-zero parameters with logarithmic addressing cost. Both bounds have the same leading constant factors, giving an asymptotic equivalence. More broadly, by establishing a theoretical link between norm minimization and algorithmic simplicity, we provide an avenue towards understanding generalization in modern deep learning.
    
[^324]: 基于签名方法的非线性和路径依赖奖励上下文老虎机：遗憾界与统计推断

    Signature Approach for Contextual Bandits with Nonlinear and Path-dependent Rewards: Regret Bound and Statistical Inference

    [https://arxiv.org/abs/2605.10313](https://arxiv.org/abs/2605.10313)

    该论文提出基于签名变换的 DisSigUCB 算法，将路径依赖的非线性奖励泛函近似为签名空间中的线性泛函，从而高效求解非线性和路径依赖奖励的上下文老虎机问题，并建立了高概率遗憾界。

    

    我们通过一种新颖的基于签名变换的方法研究具有非线性和路径依赖奖励的上下文老虎机。利用签名的通用非线性特性，我们用签名空间中的线性泛函来近似连续的路径依赖奖励泛函。这种表示使得在保留丰富的序列结构表达能力的同时，能够使用高效的线性上下文老虎机方法。在该框架基础上，我们提出了 DisSigUCB——一种基于签名的分离式上置信界（UCB）算法。在有界性和非退化性假设下，我们建立了阶为 \(\tilde{\mathcal{O}}(d_{\text{Sig}}\sqrt{KT} + \varepsilon_N T\sqrt{d_{\text{Sig}}})\) 的高概率遗憾界，其中 K 为动作数量，d_Sig 为签名特征维度，ε_N 为有限深度近似误差。我们进一步引入了 ζ-DisSigUCB……（原文摘要在此处截断）

    arXiv:2605.10313v2 Announce Type: replace  Abstract: We study contextual bandits with nonlinear and path-dependent rewards through a novel signature-transform-based approach. Leveraging the universal nonlinearity property of signatures, we approximate continuous path-dependent reward functionals by linear functionals in the signature space. This representation enables the use of efficient linear contextual bandit methods while preserving expressive sequential structure. Building on this framework, we propose \texttt{DisSigUCB}, a signature-based disjoint upper confidence bound (UCB) algorithm. Under boundedness and non-degeneracy assumptions, we establish a high-probability regret bound of order \(\tilde\mathcal{O}\bigl(d_{\text{Sig}} \sqrt{KT} + \ varepsilon_N T \sqrt{d_\Sig}\bigr)\) with \(K\) being the number of actions, \(d_{\text{Sig}} \) the signature feature dimension, and \(\varepsilon_N\) the finite-depth approximation error. We further introduce $\zeta$-\texttt{DisSigUCB}, a 
    
[^325]: 条件分布匹配的逆向设计

    Inverse Design for Conditional Distribution Matching

    [https://arxiv.org/abs/2605.09439](https://arxiv.org/abs/2605.09439)

    该论文提出了条件分布匹配（CDM）这一新的逆向设计问题类别，即寻找使生成模型的条件分布匹配给定目标分布的输入，并定义了CDMS与CDMO两个问题变体及相应的求解方法MLGD-F。

    

    生成模型是从学习到的分布 $\mathcal{P}(Y \mid X)$ 中采样的强大工具，而逆向设计方法则反转这一映射，以找到能够产生期望点输出 $y^*$ 的输入 $x$。然而，许多设计目标本质上是分布性的而非逐点的，需要考虑 $Y$ 的固有不确定性并针对其特定的分布形式，这是标准逆向设计方法无法解决的任务。为解决这一问题，我们引入了条件分布匹配（CDM），这是生成建模中一类新的逆向设计问题：给定联合分布 $\mathcal{P}(X, Y)$ 和目标分布 $\mathcal{G}(Y)$，找到一个输入 $x^*$，使其诱导的条件分布 $\mathcal{P}(Y \mid X = x^*)$ 与 $\mathcal{G}$ 相匹配。我们正式定义了两个问题变体：条件分布匹配采样（CDMS）和条件分布匹配优化（CDMO）。为求解这些问题，我们提出了 MLGD-F 方法（

    arXiv:2605.09439v2 Announce Type: replace  Abstract: Generative models are powerful tools for sampling from a learned distribution $\mathcal{P}(Y \mid X)$, and inverse-design methods invert this map to find an input $x$ that produces a desired point output $y^*$. However, many design goals are naturally distributional rather than pointwise, incorporating the inherent uncertainty of $Y$ and targeting a specific form for it, a task not addressed by standard inverse design. To address this issue we introduce Conditional Distribution Matching (CDM), a new inverse-design problem class in generative modeling: given a joint distribution $\mathcal{P}(X, Y)$ and a target distribution $\mathcal{G}(Y)$, find an input $x^*$ whose induced conditional distribution $\mathcal{P}(Y \mid X = x^*)$ matches $\mathcal{G}$. We formally define two variants: Conditional Distribution Matching Sampling (CDMS) and Conditional Distribution Matching Optimization (CDMO). To solve these problems, we propose MLGD-F (
    
[^326]: 稀疏层是扩展循环语言模型的关键

    Sparse Layers are Critical to Scaling Looped Language Models

    [https://arxiv.org/abs/2605.09165](https://arxiv.org/abs/2605.09165)

    研究发现稀疏混合专家（MoE）层是扩展循环语言模型的关键，因为不同循环通过激活不同专家可在不增加参数的情况下恢复表达能力，同时循环边界还提供了比标准模型更优的提前退出点。

    

    循环语言模型通过在深度上重复一组Transformer层来降低内存成本，并在循环边界处提供自然的提前退出点。然而，循环模型的扩展性能不如拥有独立层的标准Transformer。我们比较了标准Transformer和混合专家模型在有循环和无循环情况下的表现，得出两个主要结果。第一，我们发现循环MoE模型比标准基线具有更好的扩展性，而密集的循环模型则不然。我们将此归因于循环之间的路由分歧：在循环MoE模型中，每次通过相同的共享层时会激活不同的专家，从而在不增加额外参数的情况下恢复了模型的表达能力。第二个发现是，循环模型在提前退出方面比标准模型具有更好的计算与质量之间的权衡。由于每个循环都以产生最终输出的相同层结束，循环边界是更优的退出点，这一点已通过更早的输出得到证实。

    arXiv:2605.09165v3 Announce Type: replace-cross  Abstract: Looped language models repeat a set of transformer layers through depth, reducing memory costs and providing natural early-exit points at loop boundaries. However, looped models do not scale as favorably as standard transformers with unique layers. We compare standard and Mixture-of-Experts (MoE) transformers, with and without looping, and find two main results. First, we find Looped-MoE models scale better than the standard baseline while dense looped models do not. We trace this to routing divergence between loops: in Looped-MoE models, different experts are activated on each pass through the same shared layers, recovering expressivity without additional parameters. Our second finding is that looped models have better compute-quality trade-offs with early exits than standard models. Because each loop ends with the same layers that produce the final output, loop boundaries are superior exit points, as confirmed by earlier outp
    
[^327]: 超采样稳定扩散及更多：基于信号插值的免训练神经网络缩放方法

    Supersampling Stable Diffusion and Beyond: Training-Free Neural Network Scaling via Signal Interpolation

    [https://arxiv.org/abs/2605.08698](https://arxiv.org/abs/2605.08698)

    提出了一种免训练的卷积核信号插值方法，使Stable Diffusion 1.5无需额外训练即可生成4倍于训练分辨率的图像，同时避免了分块扩散的对象重复问题和核膨胀的微调困难。

    

    本文提出了一种使用信号插值方法（即线性、双线性、双三次插值）对神经网络进行插值的免训练方法。我们在尝试对Stable-Diffusion-1.5的卷积层进行插值以实现超出训练分辨率（BTR）图像生成时发现了这一现象。众所周知，使用DDPM（去噪扩散概率模型）生成超出训练分辨率的图像主要有两种方法：1）分块扩散和2）卷积核膨胀。分块扩散无法有效解决对象重复问题，而卷积核膨胀方法在后续微调时可能更加困难。因此，我们提出了对Stable Diffusion 1.5模型进行卷积核插值，以实现4倍超出训练分辨率的图像生成。我们通过数学方式发现并证明，当乘以一个平坦衰减因子时，信号插值方法可以正确地缩放卷积核，同时实现……

    arXiv:2605.08698v4 Announce Type: replace-cross  Abstract: This work presents a training-free approach for interpolating neural networks using signal interpolation methods (i.e., linear, bilinear, bicubic). We discovered this phenomenon while attempting to interpolate the convolution layers of Stable-Diffusion-1.5 for beyond-training-resolution (BTR) image generation. As we know, for generating BTR images using DDPMs (Denoising Diffusion Probabilistic Models), there are mainly two approaches, namely, 1) patch-wise diffusion and 2) convolution-kernel dilation. Patch-wise diffusion cannot solve the object-duplication problem efficiently, while the kernel dilation approach may be harder to fine-tune later. Thus, we proposed convolution-kernel interpolation of the Stable Diffusion 1.5 model for $4\times$ BTR image generation. We discover and show mathematically that signal interpolation methods can correctly scale convolution kernels when multiplied by a flat-attenuation factor, while achi
    
[^328]: StreamPhy：基于状态空间模型的高维物理动力学流式推理

    StreamPhy: Streaming Inference of High-Dimensional Physical Dynamics via State Space Models

    [https://arxiv.org/abs/2605.07384](https://arxiv.org/abs/2605.07384)

    StreamPhy是一个端到端流式推断框架，通过数据自适应观测编码器、支持内存高效在线更新的结构化状态空间模型和FT-FiLM连续场解码器，实现了从非规则稀疏测量中实时高效准确地推断高维多模态物理场的演化。

    

    从非规则稀疏测量中实时推断高维、多模态（如时空）物理场的演化是科学与工程领域的一个基础性挑战。现有方法，包括基于扩散的生成模型和函数张量方法，通常在离线设置下运行、依赖完整的时间观测，或产生高昂的推理成本。我们提出StreamPhy，一个端到端框架，能够从传入的非规则稀疏测量中实现全场物理动力学的高效且准确的流式推断。该框架集成了一个对任意观测模式具有鲁棒性的数据自适应观测编码器、一个支持跨非规则时间间隔进行内存高效在线更新的结构化状态空间模型，以及一个用于连续场生成的表达能力强的函数张量特征级线性调制解码器。

    arXiv:2605.07384v4 Announce Type: replace  Abstract: Inferring the evolution of high-dimensional and multi-modal (e.g., spatio-temporal) physical fields from irregular sparse measurements in real time is a fundamental challenge in science and engineering. Existing approaches, including diffusion-based generative models and functional tensor methods, typically operate in offline settings, depend on full temporal observations, or incur substantial inference cost. We propose StreamPhy, an end-to-end framework that enables efficient and accurate streaming inference of full-field physical dynamics from incoming irregular sparse measurements. The framework integrates a data-adaptive observation encoder that is robust to arbitrary observation patterns, a structured state-space model that supports memory-efficient online updates across irregular time intervals, and an expressive Functional Tensor Feature-wise Linear Modulation (FT-FiLM) decoder for continuous-field generation. We prove that FT
    
[^329]: TraXion：重新思考移动性及更广泛领域的预训练框架

    TraXion: Rethinking Pre-training Frameworks for Mobility and Beyond

    [https://arxiv.org/abs/2605.06906](https://arxiv.org/abs/2605.06906)

    该论文提出TraXion预训练框架，将移动数据所具有的元组值事件、用户持久签名和跨用户关联三个结构特性形式化为多实体时空事件流（MESES）应满足的公理，从而克服了将轨迹当作句子、访问当作词元的语言建模式预训练方法的局限。

    

    人类移动性与文本以及通用时间序列在三个结构性方面存在差异：访问（visits）是元组值事件，其含义取决于位置、时间和活动的联合分布；用户在轨迹间携带持久的特征签名；访问在用户之间并非相互独立，因为在共享地点的共同出现（co-location）是一个重要信号。现有的移动性预训练方法从语言建模中借用目标函数，将轨迹视为句子、访问视为词元，但这种类比在面对上述三个属性时均会失效。这些属性定义了一个更广泛的类别——多实体时空事件流（MESES），涵盖企业认证日志、电子健康记录以及其他实体共享基础设施、日程或情境的事件流领域。我们将这些属性精确表述为任何MESES预训练框架都应满足的三个公理，并引入了TraXion框架。

    arXiv:2605.06906v2 Announce Type: replace  Abstract: Human mobility differs from text and from generic time series in three structural ways: visits are tuple-valued events whose meaning depends on the joint distribution over location, time, and activity; users carry persistent signatures across trajectories; and visits are not independent across users, since co-location at shared places is a primary signal. Existing pre-training recipes for mobility import objectives from language modeling, treating trajectories as sentences and visits as tokens, an analogy that fails against each of the three properties above. These properties define a broader class, multi-entity spatiotemporal event streams (MESES), spanning enterprise authentication logs, electronic health records, and other event-stream domains where entities share infrastructure, schedules, or contexts. We make the properties precise as three axioms that any pre-training framework for MESES should satisfy, and introduce TraXion, w
    
[^330]: 可观测表现并不能完全反映适应性系统的组织结构：咬合约束下步态动力学的多层次分析

    Observable Performance Does Not Fully Reflect Adaptive System Organization: A Multi-Level Analysis of Gait Dynamics Under Occlusal Constraint

    [https://arxiv.org/abs/2605.00778](https://arxiv.org/abs/2605.00778)

    该研究通过对一名帕金森病患者在六种咬合约束下步态动力学的多层次分析，证明可观测的运动表现并不能完全反映适应性神经力学系统的内部组织结构，相似的表观输出可能源于不同的适应性配置。

    

    在生物力学系统中，可观测表现常被用作底层组织结构的替代指标，然而相似的输出可能源于不同的适应性配置。本研究将咬合垂直距离（VDO）视为施加于适应性神经力学系统的一种约束。针对一名帕金森病患者的单病例设计，使得在六种咬合探针条件下进行重复的个体内步态观察成为可能。研究考察了三个互补的分析层次：（i）可观测表现的聚合标量评分；（ii）概念性动力学系统框架；（iii）对来自270次M1观察的55个标准化生物力学变量的探索性UMAP表示。修订后的层次1分析表明，OC2.5与OC3的相对排名取决于评分的构建方式，而其标量分布仍然接近。层次3嵌入显示所有六种探针条件之间存在大量重叠……（摘要原文在此处被截断）

    arXiv:2605.00778v3 Announce Type: replace  Abstract: In biomechanical systems, observable performance is often used as a proxy for underlying organization, although similar outputs may arise from different adaptive configurations. This study considers the vertical dimension of occlusion (VDO) as a constraint applied to an adaptive neuromechanical system. A single-case design in a patient with Parkinson's disease enabled repeated intra-individual gait observations under six occlusal probes. Three complementary analytical levels were examined: (i) an aggregated scalar score of observable performance, (ii) a conceptual dynamical systems framework, and (iii) an exploratory UMAP representation of 55 standardized biomechanical variables from 270 M1 observations. The revised Level 1 analysis showed that the relative ranking of OC2.5 and OC3 depended on score construction, while their scalar distributions remained close. The Level 3 embedding showed substantial overlap among all six probes and
    
[^331]: Transformer中上下文学习背后的不同机制

    Distinct mechanisms underlying in-context learning in transformers

    [https://arxiv.org/abs/2604.12151](https://arxiv.org/abs/2604.12151)

    该论文通过对有限离散马尔可夫链集合上训练的Transformer进行机制分析，揭示其上下文学习呈现四个算法阶段（记忆或泛化、单点或双点统计），由两种性质截然不同的多层子电路机制实现，并由两条依赖数据多样性的边界划分记忆与泛化阶段。

    

    现代分布式网络，尤其是Transformer，获得了一种卓越的能力（称为“上下文学习”），即能够使其计算适应输入的统计特性，从而使一个固定的网络可以应用于来自广泛系统的数据。在此，我们对在有限离散马尔可夫链集合 $S$ 上训练的Transformer中这种行为提供了机制层面的表征。该Transformer展现出四个算法阶段，其特征取决于网络是进行记忆还是泛化，以及它使用的是单点统计还是双点统计。我们证明这四个阶段由多层子电路实现，这些子电路体现了实现上下文自适应计算的两种性质上截然不同的机制。最小模型分离出了这两种模式的关键特征。记忆阶段和泛化阶段由两条依赖于数据多样性 $K = |S|$ 的边界所划分。第一条边界（$K_1^\ast$）由…之间的动力学竞争所决定…（原文摘要在此处截断）

    arXiv:2604.12151v3 Announce Type: replace  Abstract: Modern distributed networks, notably transformers, acquire a remarkable ability (termed `in-context learning') to adapt their computation to input statistics, such that a fixed network can be applied to data from a broad range of systems. Here, we provide a mechanistic characterization of this behavior in transformers trained on a finite set $S$ of discrete Markov chains. The transformer displays four algorithmic phases, characterized by whether the network memorizes or generalizes, and whether it uses 1-point or 2-point statistics. We show that the four phases are implemented by multi-layer subcircuits that exemplify two qualitatively distinct mechanisms for implementing context-adaptive computations. Minimal models isolate the key features of both motifs. Memorization and generalization phases are delineated by two boundaries that depend on data diversity, $K = |S|$. The first ($K_1^\ast$) is set by a kinetic competition between su
    
[^332]: 从循环Transformer内部状态中读取人类偏好

    Reading Human Preference from Looped Transformer Internal States

    [https://arxiv.org/abs/2604.09870](https://arxiv.org/abs/2604.09870)

    该研究发现，通过在循环Transformer模型的内部状态上训练小型探针，可以以约61%-65%的微弱优势识别人类偏好的回复，且证据表明偏好信息更可能以单个回复的属性而非两回复之间的比较形式编码于模型内部。

    

    向一个冻结的语言模型展示对同一请求的两个回复，其中一个为人类所偏好。在模型隐层状态上放置一个小型探针能否分辨出是哪一个？在Ouro-2.6B-Thinking（一个对每个token将其层循环运行四次的模型）和Anthropic的HH-RLHF数据上，答案是肯定的，但效果微弱。线性探针有61.5%的概率选中被偏好的回复，一个分别对每个回复评分的小型非线性网络达到64.9%的准确率，而为版本1训练的评估器在测试集上达到64.7%：这明显优于诸如“选择更长回复”之类的简单规则，但整体仍然较弱。人类偏好究竟是以两个回复之间的比较形式存储，还是作为每个回复各自的属性存储，这一问题仍未解决。线性探针无法区分这两种假设，而一个同时读取两个回复的网络相对于分别对每个回复评分的网络，其估计优势仅为0.7个百分点（95%置信区间为0.0到1.4）。这些结果在三项控制条件下保持稳定：以两种顺序对每一对回复进行评分、划分……（原文摘要在此处被截断）

    arXiv:2604.09870v3 Announce Type: replace-cross  Abstract: Show a frozen language model two replies to the same request, one of which a person preferred. Can a small probe on the model's hidden states tell which? On Ouro-2.6B-Thinking, a model that runs its layers four times per token, and Anthropic's HH-RLHF data, the answer is yes, but only faintly. A linear probe picks the preferred reply 61.5% of the time, a small nonlinear network that scores each reply separately 64.9%, and the evaluator trained for version 1 reaches 64.7% on the test split: clearly better than simple rules such as picking the longer reply, but weak. Whether preference is stored as a comparison between the two replies or as a property of each is still open. A linear probe cannot tell, and a network that reads both replies together has an estimated edge of only 0.7 points (95% CI 0.0 to 1.4) over one that scores each reply separately. These numbers hold under three controls: scoring every pair in both orders, spli
    
[^333]: 免训练的专家语言模型动态升级

    Training-Free Dynamic Upcycling of Expert Language Models

    [https://arxiv.org/abs/2603.29765](https://arxiv.org/abs/2603.29765)

    本文提出 DUME，一种免训练的新方法，通过复用不同领域训练好的稠密专家模型来动态构建统一的 MoE 模型，从而避免多任务微调带来的过度专业化和灾难性遗忘问题。

    

    大型语言模型（LLMs）在广泛的专业任务上取得了卓越的性能，展现出强大的问题解决能力。然而，训练这些模型的成本极其高昂，且由于依赖通用知识数据集，这些模型往往缺乏特定领域的专业知识。专业知识微调可以解决这一问题；然而，这通常会导致过度专业化，且由于各领域目标存在分歧，构建单一的多领域专家模型仍然十分困难。此外，由于任务间的干扰和灾难性遗忘问题，多任务训练也面临巨大挑战。现有工作提出在专家混合架构中组合稠密模型的专业知识，但这种方法仍然需要进行多任务微调。为了解决这些问题，我们提出了动态升级专家混合模型，这是一种新颖的方法，能够复用在不同领域上训练的稠密专家模型，从而构建一个统一的 MoE 模型。

    arXiv:2603.29765v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have achieved remarkable performance on a wide range of specialized tasks, exhibiting strong problem-solving capabilities. However, training these models is prohibitively expensive, and they often lack domain-specific expertise because they rely on general knowledge datasets. Expertise finetuning can address this issue; however, it often leads to overspecialization, and developing a single multi-domain expert remains difficult due to diverging objectives. Furthermore, multitask training is challenging due to interference and catastrophic forgetting. Existing work proposes combining the expertise of dense models within a Mixture of Experts (MoE) architecture, although this approach still requires multitask finetuning. To address these issues, we introduce Dynamic Upcycling MoE (DUME), a novel approach that reuses dense experts trained on different domains to construct a unified MoE model. Our method 
    
[^334]: 面向线性马尔可夫决策过程的参数化策略乐观演员-评论家算法

    Optimistic Actor-Critic with Parametric Policies for Linear Markov Decision Processes

    [https://arxiv.org/abs/2603.28595](https://arxiv.org/abs/2603.28595)

    该论文针对有限时域线性马尔可夫决策过程提出了使用参数化对数线性策略的乐观演员-评论家框架，通过引入可处理的logit匹配回归目标和基于朗之万动力学的近似汤普森采样，克服了现有演员-评论家方法在探索假设和隐式策略采样效率方面的理论局限。

    

    尽管演员-评论家方法在实践中取得了成功，但其理论分析存在若干局限性。具体而言，现有的理论工作要么通过做出强假设来回避探索问题，要么分析经过复杂算法修改后不实用的方法。此外，针对线性马尔可夫决策过程（MDP）所分析的演员-评论家方法通常采用自然策略梯度，并通过累积的评论家估计来定义“隐式”策略，而不是学习显式参数化的演员。这类策略的采样在计算上可能代价高昂，导致与环境交互的效率低下。为此，我们聚焦于有限时域线性MDP，提出了一种使用参数化对数线性策略的乐观演员-评论家框架。特别地，我们为演员引入了一个易于处理的logit匹配回归目标。对于评论家，我们通过朗之万动力学（Langevin Mo...原文在此处截断）使用近似汤普森采样。

    arXiv:2603.28595v3 Announce Type: replace  Abstract: Although actor-critic methods have been successful in practice, their theoretical analyses have several limitations. Specifically, existing theoretical work either sidesteps the exploration problem by making strong assumptions or analyzes impractical methods with complicated algorithmic modifications. Moreover, the actor-critic methods analyzed for linear MDPs often employ natural policy gradient and define "implicit" policies through the accumulated critic estimates, rather than learning an explicitly parameterized actor. Such policies can be computationally expensive to sample from, making the environment interactions inefficient. To that end, we focus on the finite-horizon linear MDPs and propose an optimistic actor-critic framework that uses parametric log-linear policies. In particular, we introduce a tractable logit-matching regression objective for the actor. For the critic, we use approximate Thompson sampling via Langevin Mo
    
[^335]: 乘性Lipschitz界作为一种归纳偏置

    The multiplicative Lipschitz bound as an inductive bias

    [https://arxiv.org/abs/2603.28113](https://arxiv.org/abs/2603.28113)

    该论文提出将看似宽松的乘性Lipschitz“平凡界”与新型求积非线性激活函数相结合，在Lipschitz约束学习中超越了Sandwich、凸势层等最先进方法，并在非扩张自编码器上实现了比PCA低12-24%的重建误差。

    

    我们研究了在Lipschitz约束下训练神经网络以最小化损失函数的问题。虽然估计神经网络Lipschitz常数最直观的方法是将各层的Lipschitz常数相乘，但这个我们称为“平凡界”的估计通常被认为过于宽松而不实用。我们证明，恰恰相反，在Lipschitz约束学习中，它可以比最先进的方法（Sandwich、几乎正交层、凸势层以及基于SDP的Lipschitz层）更有效。我们的方法将平凡界与一种新型激活函数（求积非线性）相结合，在大多数测试的分类边界上实现了更低的交叉熵。对于在学习范数下约束为非扩张的自编码器，它在所有测试的瓶颈宽度下都实现了最低的像素重建误差，相比PCA提升了12-24%。我们的架构由级联组成（摘要在此处截断）……

    arXiv:2603.28113v3 Announce Type: replace  Abstract: We study the problem of training a neural network to minimize a loss function given a Lipschitz constraint. While the most obvious way to estimate the Lipschitz constant of a neural network is to multiply the Lipschitz constants of its layers, this bound, which we term the _trivial bound_, is often considered impractically loose. We show that, on the contrary, it can be more effective than state-of-the-art methods (Sandwich, Almost-Orthogonal Layers, Convex Potential Layers, and SDP-based Lipschitz Layers) for Lipschitz-constrained learning. Our method combines the trivial bound with a novel activation function (the quadrature nonlinearity) to achieve lower cross-entropy over most of the tested classification bounds. For autoencoders constrained to be nonexpansive in a learned norm, it achieves the lowest pixel reconstruction error at every tested bottleneck width, improving on PCA by 12--24%. Our architecture, consisting of concaten
    
[^336]: Shapley遇上Rawls：一个度量与解释不公平性的集成框架

    Shapley meets Rawls: an integrated framework for measuring and explaining unfairness

    [https://arxiv.org/abs/2603.26476](https://arxiv.org/abs/2603.26476)

    本文提出利用Shapley值及其扩展的ESL值家族，将群体不公平性的定义、度量和解释整合到统一框架中，并在Census Income数据集上识别出年龄、每周工作小时数和婚姻状况是造成性别不公平的关键因素，同时计算效率优于传统Bootstrap检验。

    

    可解释性与公平性通常被分开研究，近期的少数研究才尝试解释不公平性的来源。本文表明，在标准的群体公平性准则下，Shapley值可以同时用于定义和解释不公平性。这为估计不公平性、对其进行统计推断，以及识别导致不公平性的特征提供了一个集成框架。我们的框架还可以从Shapley值扩展到高效-对称-线性（ESL）值家族，其中一些方法提供了更稳健的公平性定义和更短的计算时间。我们在UCI机器学习库的Census Income数据集上进行了实证演示。我们的方法表明，“年龄”、“每周工作小时数”和“婚姻状况”是造成性别不公平的因素，且计算时间比传统的Bootstrap检验更短。

    arXiv:2603.26476v3 Announce Type: replace  Abstract: Explainability and fairness have mainly been considered separately, with recent exceptions trying the explain the sources of unfairness. This paper shows that the Shapley value can be used to both define and explain unfairness, under standard group fairness criteria. This offers an integrated framework to estimate and derive inference on unfairness as-well-as the features that contribute to it. Our framework can also be extended from Shapley values to the family of Efficient-Symmetric-Linear (ESL) values, some of which offer more robust definitions of fairness, and shorter computation times. An illustration is run on the Census Income dataset from the UCI Machine Learning Repository. Our approach shows that ``Age", ``Number of hours" and ``Marital status" generate gender unfairness, using shorter computation time than traditional Bootstrap tests.
    
[^337]: 利用层间几何结构改进大语言模型的最终层表示

    Improving LLM Final Representations with Inter-Layer Geometry

    [https://arxiv.org/abs/2603.22665](https://arxiv.org/abs/2603.22665)

    本文提出用轻量级图神经网络聚合大语言模型各层信息，并进一步设计基于SL(2,Zn)凯莱图的Cayley编码器，以稀疏、规则且低直径的数学拓扑替代昂贵的注意力机制，在提升效率的同时显著增强下游预测性能。

    

    基于大语言模型（LLM）进行预测的标准做法是将最终层的表示作为下游预测器的输入。然而，中间层可能编码了互补的、与任务相关的信号。因此，现有方法要么为每个任务搜索最佳的层，要么采用代价高昂的基于注意力的机制来学习层间聚合。在本工作中，我们首先证明这种复杂性是不必要的：在由LLM各层构成的完全连接图上使用一个轻量级的图神经网络（GNN），不仅比现有方法更高效，还能取得显著更强的预测性能。随后，我们提出了Cayley编码器（Cayley-Encoder），通过用定义在SL(2, Zn)上的凯莱图替换完全连接图，进一步提升了效率与预测性能。这些凯莱图提供了一种数学上有依据的拓扑结构，具有稀疏、构造上规则且低直径的特点，从而使各层之间能够进行有效的通信。

    arXiv:2603.22665v4 Announce Type: replace  Abstract: The standard in LLM-based prediction is to use the final-layer representation as the input to a downstream predictor. However, intermediate layers may encode complementary task-relevant signals. Existing approaches therefore either search for the best layer for each task or apply expensive attention-based mechanisms to learn inter-layer aggregation. In this work, we first show that such complexity is unnecessary: a lightweight Graph Neural Network over a fully connected graph of LLM layers is more efficient and achieves significantly stronger predictive performance than existing approaches. We then introduce the Cayley-Encoder, which further improves both efficiency and predictive performance by replacing the fully connected graph with a Cayley graph over SL(2, Zn). These Cayley graphs provide a mathematically grounded topology that is sparse, regular by construction, and has low diameter. This enables effective communication across 
    
[^338]: HYPER：假设引导的证据寻求检索规划

    HYPER: Hypothesis-Guided Planning for Evidence-Seeking Retrieval

    [https://arxiv.org/abs/2603.19008](https://arxiv.org/abs/2603.19008)

    HYPER是一个无需训练的检索前规划框架，以临时答案假设为锚点，将检索分解为支持、区分和特征三种证据角色，把RAG从面向主题的检索转变为面向证据的检索，从而在需要竞争性选择的任务中更好地支持最终决策。

    

    检索增强生成（RAG）通过将生成过程建立在外部非参数化知识之上，从而改进了大语言模型（LLM）。然而，当任务需要在相互竞争的可能性之间做出选择时（例如在多种合理备选方案中进行诊断），仅靠主题相关的上下文往往不足以驱动最终决策。为了解决这一问题，我们提出了HYPER：假设引导的证据寻求检索规划（Hypothesis-Guided Planning for Evidence-Seeking Retrieval），这是一个无需训练的检索前规划框架，将RAG从面向主题的检索重新定位为面向证据的检索。HYPER使用一个临时答案假设作为规划锚点，然后将检索分解为三种证据角色：支持（Support）寻找支持临时答案的证据；区分（Distinction）检索能够区分相互竞争备选方案的标准；特征（Feature）则获取决策所需的显著问题特征。这一规划步骤引导检索朝向接近决策的证据……

    arXiv:2603.19008v2 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) improves Large Language Models (LLMs) by grounding generation in external, non-parametric knowledge. However, when a task requires choosing among competing possibilities, such as a diagnosis among plausible alternatives, topically relevant context is often insufficient to drive the final decision. To address this issue, we propose HYPER: Hypothesis-Guided Planning for Evidence-Seeking Retrieval, a training-free pre-retrieval planning framework that reorients RAG from topic-oriented retrieval to evidence-oriented retrieval. HYPER uses a provisional answer hypothesis as a planning anchor. It then decomposes retrieval into three evidence roles. Support seeks evidence for the provisional answer. Distinction retrieves criteria that separate competing alternatives. Feature grounds salient question features needed for the decision. This planning step steers retrieval toward evidence near the decisi
    
[^339]: AERR-Nav：面向零样本目标导航的自适应探索-恢复-回忆策略

    AERR-Nav: Adaptive Exploration-Recovery-Reminiscing Strategy for Zero-Shot Object Navigation

    [https://arxiv.org/abs/2603.17712](https://arxiv.org/abs/2603.17712)

    AERR-Nav提出了一种自适应探索-恢复-回忆策略，通过根据环境动态调整机器人状态，解决了未知多层环境中零样本目标导航的探索与利用平衡难题。

    

    在未知多层环境中进行零样本目标导航（ZSON）是一项重大挑战。近期的相关方法大多基于语义价值贪婪路径点选择、空间拓扑增强记忆，以及将多模态大语言模型（MLLM）作为决策框架，已取得了一定进展。然而，这些架构在遭遇未见环境时，难以平衡零样本目标导航中的探索与利用，尤其是在多层环境场景下，例如机器人在狭窄路口卡住、无休止徘徊，或无法找到楼梯入口等问题。为了克服这些挑战，我们提出了AERR-Nav，一个能够根据机器人所处环境动态调整自身状态的零样本目标导航框架。具体而言，AERR-Nav具有以下两个关键优势：（1）自适应探索-恢复-回忆策略，使机器人能够在三种状态之间动态转换，从而促进……

    arXiv:2603.17712v2 Announce Type: replace-cross  Abstract: Zero-Shot Object Navigation (ZSON) in unknown multi-floor environments presents a significant challenge. Recent methods, mostly based on semantic value greedy waypoint selection, spatial topology-enhanced memory, and Multimodal Large Language Model (MLLM) as a decision-making framework, have led to improvements. However, these architectures struggle to balance exploration and exploitation for ZSON when encountering unseen environments, especially in multi-floor settings, such as robots getting stuck at narrow intersections, endlessly wandering, or failing to find stair entrances. To overcome these challenges, we propose AERR-Nav, a Zero-Shot Object Navigation framework that dynamically adjusts its state based on the robot's environment. Specifically, AERR-Nav has the following two key advantages: (1) An Adaptive Exploration-Recovery-Reminiscing Strategy, enables robots to dynamically transition between three states, facilitatin
    
[^340]: 利用跨语言不一致性在混合专家大语言模型中进行知识定位

    Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency

    [https://arxiv.org/abs/2603.17102](https://arxiv.org/abs/2603.17102)

    该论文提出将大语言模型的跨语言不一致性作为可解释性工具，通过对比模型在不同语言中事实回忆成功与失败时的MoE路由行为，定位混合专家模型中负责特定知识的关键专家。

    

    现代大语言模型在不同语言之间的行为表现出显著差异，例如能够在某些语言中回忆起事实信息，而在其他语言中却无法做到。虽然这通常被视为一个需要缓解的问题，但在本工作中，我们提出将这种跨语言不一致性作为混合专家大语言模型可解释性的一种工具加以利用。我们的知识定位框架通过对比模型在正确回忆信息的语言集合与无法回忆的语言集合之间的路由行为，从而分离出在回答某项知识时发挥功能性作用的模型组件。我们的方法分为两个阶段：(1) 使用跨多种语言的困难事实问题查询模型，以生成“成功”和“失败”激活桶；(2) 对MoE路由器的逻辑值应用统计对比分析，以识别对知识重要的专家。为了验证必

    arXiv:2603.17102v2 Announce Type: replace  Abstract: Modern LLMs continue to exhibit significant variance in behavior across languages, such as being able to recall factual information in some languages but not others. While typically studied as a problem to be mitigated, in this work, we propose leveraging this cross-lingual inconsistency as a tool for interpretability in mixture-of-experts (MoE) LLMs. Our knowledge localization framework contrasts routing for sets of languages where the model correctly recalls information from languages where it fails. This allows us to isolate model components that play a functional role in answering about a piece of knowledge. Our method proceeds in two stages: (1) querying the model with difficult factual questions across a diverse set of languages to generate "success" and "failure" activation buckets and then (2) applying a statistical contrastive analysis to the MoE router logits to identify experts important for knowledge. To validate the nece
    
[^341]: 审计因果发现算法中的路径特定公平性：一项心力衰竭案例研究

    Auditing Path-Specific Fairness Across Causal Discovery Algorithms: A Heart Failure Case Study

    [https://arxiv.org/abs/2603.15926](https://arxiv.org/abs/2603.15926)

    本研究通过为心力衰竭临床记录数据集建立专家引导的因果图基准，首次在临床数据上系统评估了不同因果发现算法的选择如何通过路径特定公平性分解传导并影响公平性与效用的权衡。

    

    因果发现方法正越来越多地应用于医疗健康数据，而从中恢复出的因果关系可能编码了带有偏见的路径，并传播到下游预测之中。路径特定公平性分析通过对直接效应、间接效应和虚假效应进行分解，并结合公平性-效用比率的评估，能够量化偏差及其相关的预测代价。然而，现有的在公平性约束下对因果发现的评估大多使用非临床数据集，并且未能将混杂因素的贡献与中介因素的贡献区分开来，这使得因果发现算法的选择如何通过路径特定公平性传导至临床场景中的效用权衡这一问题仍然悬而未决。本工作通过为心力衰竭临床记录数据集建立一个由专家引出的因果图基准，并通过复合的、逐变量的公平性分解以及公平性-效用比率来追踪评估各类因果发现算法，从而填补了这一空白。

    arXiv:2603.15926v2 Announce Type: replace-cross  Abstract: Causal discovery is increasingly applied to healthcare data, where the recovered relationships can encode biased pathways that propagate to downstream prediction. Path-specific fairness analysis, through direct, indirect, and spurious effect decompositions along with fairness-utility ratio assessment allows bias quantification and associated prediction cost. However, existing evaluations of causal discovery under fairness constraints have largely used non-clinical datasets and have not separated the contribution of confounders from that of mediators, leaving open how discovery-algorithm choice propagates through path-specific fairness to the utility trade-off in clinical settings. This work addresses that gap by establishing an expert-elicited causal graph benchmark for the heart failure clinical records dataset and tracing discovery algorithms through composite, per-variable fairness decompositions and fairness-utility ratio. 
    
[^342]: SHAMISA：面向自监督无参考图像质量评估的隐式结构关联塑形建模

    SHAMISA: SHAped Modeling of Implicit Structural Associations for Self-supervised No-Reference Image Quality Assessment

    [https://arxiv.org/abs/2603.13669](https://arxiv.org/abs/2603.13669)

    该论文提出SHAMISA，一种非对比自监督无参考图像质量评估框架，其核心创新是组合式失真引擎——从连续参数空间生成每次仅单一失真因素变化的图像分组，从而在训练中实现对表征相似性的细粒度塑形控制。

    

    无参考图像质量评估（NR-IQA）旨在在不获取原始参考图像的情况下估计感知质量。学习NR-IQA模型面临一个根本性瓶颈，因为它需要大量昂贵的人类感知标签。我们提出SHAMISA，一种非对比的自监督框架，通过显式结构化的关系监督从无标签的失真图像中学习。与先前施加僵化二元相似性约束的方法不同，SHAMISA塑造出柔软、可控的隐式结构关联，这些关联既感知失真又对内容敏感。其一个关键创新是组合式失真引擎，它从连续参数空间生成不可数的退化族，并对生成的图像进行分组，使得每次恰好只有一个失真因素发生变化。这一构造使得训练过程中能够对表征相似性进行细粒度控制。具有共享失……（摘要在此处截断）

    arXiv:2603.13669v3 Announce Type: replace-cross  Abstract: No-Reference Image Quality Assessment (NR-IQA) aims to estimate perceptual quality without access to a pristine reference. Learning an NR-IQA model faces a fundamental bottleneck because it requires many costly human perceptual labels. We propose SHAMISA, a non-contrastive self-supervised framework that learns from unlabeled distorted images through explicitly structured relational supervision. Unlike prior methods that impose rigid binary similarity constraints, SHAMISA shapes soft, controllable implicit structural associations that are both distortion-aware and content-sensitive. A key innovation is its compositional distortion engine, which generates an uncountable family of degradations from continuous parameter spaces and groups the resulting images so that exactly one distortion factor varies at a time. This construction enables fine-grained control over representational similarity during training. Images with shared dist
    
[^343]: 仅依靠人体运动数据的模仿学习并不能保证生物力学上合理的步态动力学

    Imitation Learning from Human Motion Alone Does Not Guarantee Biomechanically Plausible Gait Kinetics

    [https://arxiv.org/abs/2603.12408](https://arxiv.org/abs/2603.12408)

    研究发现仅依靠运动数据的模仿学习虽能准确跟踪运动学，却无法估计出合理的关节力矩，而将地面反作用力和压力中心等动力学信息纳入模仿奖励的KAIL框架能产生更符合生物力学的步态动力学。

    

    通过运动数据进行模仿学习在机器人和角色动画领域的应用日益广泛，然而其在人体生物力学中的应用能力——即在缺乏显式动力学信息的情况下估计生理学合理的关节力矩——仍不明确。在本研究中，我们检验了仅靠运动模仿是否能够在人体行走中估计出合理的生物学关节力矩。我们将仅运动的模仿学习（MOIL）与动力学感知的模仿学习（KAIL）框架进行了比较，后者在模仿奖励中纳入了额外的地面反作用力（GRF）和压力中心。我们采用渐进式动力学奖励塑形方法来考察每个动力学项的贡献。实验使用了来自两名无残疾参与者在三种速度（0.9、1.2和1.5 m/s）下的行走数据。结果表明，尽管MOIL和KAIL在运动学跟踪精度上相当，但KAIL产生的动力学结果与生物力学值更为接近。

    arXiv:2603.12408v4 Announce Type: replace-cross  Abstract: Imitation learning (IL) via motion data is increasingly used in robotics and character animation, yet its ability to be applied to human biomechanics to estimate physiologically plausible joint moments without explicit kinetic information remains unclear. In this study, we examined whether motion imitation alone can estimate reasonable biological joint moments in human walking. We compared motion-only IL (MOIL) against a kinetics-aware IL (KAIL) framework, which incorporates additional ground reaction forces (GRFs) and center of pressure (CoP) in imitation rewards. A progressive kinetic reward shaping approach was used to examine the contribution of each kinetic term. Experiments were conducted using walking data from two non-disabled participants at three speeds (0.9, 1.2, and 1.5 m/s). While both MOIL and KAIL achieved comparable kinematic tracking accuracy, KAIL produced kinetics more closely aligned with biomechanical value
    
[^344]: 面向基于扩展的类增量学习的反事实特征正则化

    Counterfactual Feature Regularization for Expansion-Based Class-Incremental Learning

    [https://arxiv.org/abs/2603.09145](https://arxiv.org/abs/2603.09145)

    本文从因果视角指出虚假特征相关性是新旧特征冲突的根源，提出基于必要性与充分性概率（PNS）的反事实特征正则化方法，以消除任务内和任务间的虚假相关性，指导类增量学习中的特征扩展。

    

    当前基于扩展的类增量学习方法通过冻结旧特征有效缓解了灾难性遗忘问题。然而，从新任务中学习到的任务特定特征可能与旧特征发生冲突。从因果视角来看，虚假特征相关性是导致这种冲突的主要原因，其表现在两个方面：（i）在经验风险最小化（ERM）的引导下，任务内虚假相关性使任务特定特征依赖于捷径特征，这些非鲁棒特征容易受到干扰，不可避免地漂移到其他任务的特征空间中；（ii）任务间虚假相关性会在跨任务的视觉相似类别之间引发语义混淆。为解决这一问题，我们提出了一种基于必要性与充分性概率（PNS）的正则化方法，用于指导类增量学习中的特征扩展。具体而言，我们首先将PNS的定义扩展到基于扩展的类增量学习中……

    arXiv:2603.09145v4 Announce Type: replace-cross  Abstract: Current expansion-based methods for Class Incremental Learning (CIL) effectively mitigate catastrophic forgetting by freezing old features. However, such task-specific features learned from the new task may collide with the old features. From a causal perspective, spurious feature correlations are the main cause of this collision, manifesting in two scopes: (i) guided by empirical risk minimization (ERM), intra-task spurious correlations cause task-specific features to rely on shortcut features. These non-robust features are vulnerable to interference, inevitably drifting into the feature space of other tasks; (ii) inter-task spurious correlations induce semantic confusion between visually similar classes across tasks. To address this, we propose a Probability of Necessity and Sufficiency (PNS)-based regularization method to guide feature expansion in CIL. Specifically, we first extend the definition of PNS to expansion-based C
    
[^345]: 在线强化学习在移动GUI智能体中的泛化能力有多远？基于AndroidWorld-Generalization的受控研究

    How Far Does Online RL Generalize in Mobile GUI Agents? A Controlled Study with AndroidWorld-Generalization

    [https://arxiv.org/abs/2603.07432](https://arxiv.org/abs/2603.07432)

    本文提出AndroidWorld-Generalization基准，通过未见实例、未见模板、未见应用三级递进的上下文偏移系统评估在线RL在移动GUI智能体中的泛化能力，发现RL收益随环境偏移距离增大而单调衰减。

    

    在线强化学习（RL）已将AndroidWorld上移动GUI智能体的成功率提升至80%以上，但这些系统在相同环境中进行训练和评估，且其训练任务未公开以供审计。因此，RL在训练环境之外的泛化能力究竟如何仍无从衡量。我们将移动环境形式化为上下文马尔可夫决策过程，并构建了一个由三级渐进式远离的上下文偏移组成的阶梯，即AndroidWorld-Generalization基准：未见实例、未见模板和未见应用，并人工验证了各层级之间的区分度。此外，我们还提供了一个开放的RL训练系统，使这一规模的研究成为可能。将相同的RL算法应用于两个来源与能力各异的初始模型——成功率为23.0%的UI-TARS-7B-SFT和成功率为52.6%的Qwen3-VL-8B——我们发现RL在分布内均可提升两者的性能，但增益随阶梯逐级单调衰减（分别为+17.5/+11.1/+4.2和+10.3/+7.4/+0），训练表现……

    arXiv:2603.07432v2 Announce Type: replace-cross  Abstract: Online reinforcement learning (RL) has advanced mobile GUI agents on AndroidWorld to over 80\% success, but these systems train and evaluate in the same environment, and their training tasks are not released for auditing. How far RL generalizes beyond its training contexts therefore remains unmeasured. We formalize the mobile environment as a Contextual MDP and instantiate a ladder of three progressively distant context shifts as AndroidWorld-Generalization: Unseen Instance, Unseen Template, and Unseen Application, with separation verified manually. In addition, we provide an open RL training system that makes studies at this scale feasible. Applying the identical RL algorithm to two priors of different origin and competence, UI-TARS-7B-SFT at 23.0\% and Qwen3-VL-8B at 52.6\%, we find that RL improves both in distribution but that gains fall monotonically up the ladder (+17.5/+11.1/+4.2 and +10.3/+7.4/+0), with training perform
    
[^346]: 通过离散随机局部化解锁连续扩散语言模型

    Unlocking Continuous Diffusion Language Models via Discrete Stochastic Localization

    [https://arxiv.org/abs/2602.16169](https://arxiv.org/abs/2602.16169)

    提出离散随机局部化（DSL）连续状态框架，通过将随机局部化通道与单位球面词元嵌入相结合，实现了仅依赖含噪状态、无需时间步条件的贝叶斯最优去噪器，使单个去噪网络即可支持连续去噪、掩码扩散、随机顺序自回归等多种采样方法，从而让连续扩散模型在离散序列生成上有效匹敌掩码离散扩散模型。

    

    连续扩散是非自回归生成的一个自然框架，但在离散序列生成任务上通常落后于掩码离散扩散模型（MDMs）。我们认为，通过对噪声参数化与词元表示几何结构的联合设计，连续扩散可以有效地建模离散序列。我们提出了离散随机局部化，这是一个连续状态框架，它将随机局部化通道与单位球面上的词元嵌入相结合，从而得到一个仅依赖于含噪状态的贝叶斯最优去噪器，无需额外的时间步条件。单个训练好的去噪网络即可支持多种采样方法，包括连续去噪、掩码扩散、随机顺序自回归生成以及先连续后离散的混合生成。因此，在这种连续状态框架下的并行生成并不必然意味着同步的去噪。

    arXiv:2602.16169v3 Announce Type: replace-cross  Abstract: Continuous diffusion is a natural framework for non-autoregressive generation but has generally lagged behind masked discrete diffusion models (MDMs) on discrete sequence generation. We argue that continuous diffusion can effectively model discrete sequences through the joint design of noise parameterization and token representation geometry. We introduce \emph{Discrete Stochastic Localization} (DSL), a continuous-state framework that combines a stochastic localization channel with unit-sphere token embeddings, resulting in a Bayes-optimal denoiser that depends only on the noisy state, without additional timestep conditioning. A single trained denoising network then supports multiple sampling methods, from continuous denoising to masked diffusion, random-order autoregressive generation and hybrid continuous-then-discrete generation. Parallel generation in this continuous-state framework, therefore, need not imply synchronized d
    
[^347]: MUSE：支持无缝模型更新的多租户模型服务框架

    MUSE: Multi-Tenant Model Serving With Seamless Model Updates

    [https://arxiv.org/abs/2602.11776](https://arxiv.org/abs/2602.11776)

    MUSE是一个多租户模型服务框架，通过将模型分数与客户端决策边界解耦，使模型更新时无需跨数百个客户重新协调校准阈值，从而实现无缝的模型更新。

    

    在二分类系统中，决策阈值将模型分数转化为实际行动。选择合适的阈值既依赖于底层模型分数的具体分布，也依赖于使用该模型的每个客户的具体业务决策。然而，重新训练模型不可避免地会改变分数分布，导致现有阈值失效。在多租户的“分数即服务”环境中，决策边界位于客户自主管理的基础设施中，这造成了一个严重的瓶颈：重新校准需要在数百个客户之间协调阈值更新，消耗大量人工工时并导致模型更新停滞。我们提出了MUSE，这是一个模型服务框架，通过将模型分数与客户决策边界解耦来实现无缝的模型更新。MUSE专为多租户场景设计，通过基于动态意图的路由实现模型共享来优化基础设施复用，并结合两级评分机制……

    arXiv:2602.11776v4 Announce Type: replace  Abstract: In binary classification systems, decision thresholds translate model scores into actions. Choosing suitable thresholds relies on the specific distribution of the underlying model scores but also on the specific business decisions of each client using that model. However, retraining models inevitably shifts score distributions, invalidating existing thresholds. In multi-tenant Score-as-a-Service environments, where decision boundaries reside in client-managed infrastructure, this creates a severe bottleneck: recalibration requires coordinating threshold updates across hundreds of clients, consuming excessive human hours and leading to model stagnation. We introduce MUSE, a model serving framework that enables seamless model updates by decoupling model scores from client decision boundaries. Designed for multi-tenancy, MUSE optimizes infrastructure re-use by sharing models via dynamic intent-based routing, combined with a two-level sc
    
[^348]: 基于签名核的鲁棒概率预测与尾部事件预测评估指标

    Signature-Kernel Based Evaluation Metrics for Robust Probabilistic and Tail-Event Forecasting

    [https://arxiv.org/abs/2602.10182](https://arxiv.org/abs/2602.10182)

    本文提出了一族基于签名核的删失评估指标，通过将分布主体收缩为单一枢轴点来聚焦尾部区域和高效用区域的评估，从而更准确地衡量概率预测在分布尾部的保真度。

    

    arXiv:2602.10182v2 公告类型：替换 摘要：连续分级概率评分（CRPS）和分位数损失（QL）等标准的基于样本的度量指标被广泛用于概率时间序列预测的评估。然而，这些指标无法捕捉多变量之间的相关性，也无法评估模型捕捉分布尾部的能力。由于分布的主体部分在指标中被过度代表，这些全局指标对分布尾部中高效用事件上的误差不敏感。为了在保持适当评分规则特性的同时，准确衡量分布尾部区域和高效用区域上的分布保真度，我们提出了一族删失签名核度量指标。我们提出的指标通过将分布的主体收缩为单一枢轴点，将预测评估集中于关注区域，即代表高效用或分布尾部的区域。我们在时间序列基础模型（TSFMs）上的基准测试表明，虽然可以为分布形成明确的排名……

    arXiv:2602.10182v2 Announce Type: replace  Abstract: Standard sample-based metrics like Continuous Ranked Probability Score (CRPS) and Quantile Loss (QL) are frequently used in the evaluation of probabilistic time-series forecasting. However, these metrics fail to capture multivariate correlations or assess the ability to capture distribution tails. These global metrics are insensitive to errors on high-utility events in distributional tails due to over-representation of the distribution's body. To accurately measure distributional fidelity on tail-regions and regions of high utility while still having a proper scoring rule, we introduce a family of censored signature kernel metrics. Our proposed metrics concentrate evaluation of forecasts to a focus region, representing high-utility or distributional tails, by collapsing the body of the distribution to a single pivot. Our benchmarks on time-series foundation models (TSFMs) reveal that while a clear ranking can be formed for distributi
    
[^349]: 面向图表理解的多模态信息融合：多模态大语言模型（MLLM）综述——演进、局限性与认知增强

    Multimodal Information Fusion for Chart Understanding: A Survey of MLLMs -- Evolution, Limitations, and Cognitive Enhancement

    [https://arxiv.org/abs/2602.10138](https://arxiv.org/abs/2602.10138)

    本综述系统梳理了基于多模态大语言模型（MLLM）的图表理解研究，分析了视觉与语言信息融合的核心挑战，提出了规范型与非规范型基准的新颖分类体系，并追溯了从经典深度学习技术到先进MLLM融合范式的方法演进历程。

    

    图表理解是一项典型的信息融合任务，需要将图形数据与文本数据无缝整合以提取含义。多模态大语言模型（MLLM）的出现彻底变革了这一领域，然而基于MLLM的图表分析研究现状仍较为碎片化，缺乏系统性梳理。本综述通过构建该领域核心组件的结构化框架，为这一新兴前沿提供了全面的路线图。我们首先分析了图表中视觉与语言信息融合所面临的基本挑战。随后，我们对下游任务和数据集进行了分类，并引入了一种新颖的规范型与非规范型基准分类法，以凸显该领域不断扩展的研究范围。接着，我们全面梳理了方法论的演进历程，从经典深度学习技术追溯到采用复杂融合策略的最先进MLLM范式。

    arXiv:2602.10138v2 Announce Type: replace-cross  Abstract: Chart understanding is a quintessential information fusion task, requiring the seamless integration of graphical and textual data to extract meaning. The advent of Multimodal Large Language Models (MLLMs) has revolutionized this domain, yet the landscape of MLLM-based chart analysis remains fragmented and lacks systematic organization. This survey provides a comprehensive roadmap of this nascent frontier by structuring the domain's core components. We begin by analyzing the fundamental challenges of fusing visual and linguistic information in charts. We then categorize downstream tasks and datasets, introducing a novel taxonomy of canonical and non-canonical benchmarks to highlight the field's expanding scope. Subsequently, we present a comprehensive evolution of methodologies, tracing the progression from classic deep learning techniques to state-of-the-art MLLM paradigms that leverage sophisticated fusion strategies. By criti
    
[^350]: 迈向可泛化推理：用于大语言模型推理的群体因果反事实策略优化

    Towards Generalizable Reasoning: Group Causal Counterfactual Policy Optimization for LLM Reasoning

    [https://arxiv.org/abs/2602.06475](https://arxiv.org/abs/2602.06475)

    该论文从因果视角将多候选推理视为一族反事实实验，提出群体因果反事实策略优化，通过同时评估推理稳健性与有效性的反事实奖励机制，训练大语言模型超越仅以答案正确性为标准的学习方式，从而掌握可泛化的推理模式。

    

    随着推理能力的进步，大语言模型（LLM）在复杂任务中表现出色。然而，现有的奖励机制仍然与最终答案的正确性紧密耦合，很少关注底层的推理过程：推理合理但答案错误的轨迹只能获得较低的奖励，而逻辑有缺陷却侥幸猜对的答案却可能获得高奖励，这影响了推理的泛化能力。从因果视角出发，我们将针对固定问题的多候选推理解释为一族反事实实验。在此基础上，我们提出了群体因果反事实策略优化，以显式训练大语言模型学习可泛化的推理模式。该方法提出了一种情景化的因果反事实奖励，共同捕捉：(i) 稳健性，即鼓励由推理步骤所诱导的答案分布在反事实扰动下保持稳定；(ii) 有效性，即强制足够的变化性，以便……（摘要原文在此处截断）

    arXiv:2602.06475v3 Announce Type: replace  Abstract: Large language models (LLMs) excel at complex tasks with advances in reasoning capabilities. However, existing reward mechanisms remain tightly coupled to final correctness and pay little attention to the underlying reasoning process: trajectories with sound reasoning but wrong answers receive low credit, while lucky guesses with flawed logic may be highly rewarded, affecting reasoning generalization. From a causal perspective, we interpret multi-candidate reasoning for a fixed question as a family of counterfactual experiments. Building on this, we propose Group Causal Counterfactual Policy Optimization to explicitly train LLMs to learn generalizable reasoning patterns. It proposes an episodic causal counterfactual reward that jointly captures (i) robustness, encouraging the answer distribution induced by a reasoning step to remain stable under counterfactual perturbations; and (ii) effectiveness, enforcing sufficient variability so
    
[^351]: 实用好奇心：基于主动推理的混合学习与优化的统一框架

    Pragmatic Curiosity: A Unified Framework for Hybrid Learning and Optimization via Active Inference

    [https://arxiv.org/abs/2602.06104](https://arxiv.org/abs/2602.06104)

    该论文提出了实用好奇心（PraC）框架，通过主动推理将贝叶斯优化与贝叶斯实验设计统一起来，以信息增益与预期遗憾之间的权衡来指导混合学习与优化场景中的序贯决策。

    

    许多工程和科学工作流程依赖于昂贵的黑盒评估，需要做出既能提升任务性能又能降低不确定性的序贯决策。贝叶斯优化（BO）和贝叶斯实验设计（BED）分别为目标导向的优化和信息探索实验提供了强大但在很大程度上相互独立的方法，对于学习与优化内在耦合的混合场景，现有方法缺乏足够的指导。我们提出了实用好奇心，这是一个通过主动推理实现混合学习与优化的统一框架。PraC 通过权衡关于任务相关潜在符号的信息增益与基于预期遗憾的结果潜力来评估候选查询。这一表述突出了三个操作层面的设计选择：应当澄清哪个潜在量、任务价值如何被编码为遗憾，以及信息增益应当以多大强度进行交换（原文摘要在此处截断）。

    arXiv:2602.06104v3 Announce Type: replace  Abstract: Many engineering and scientific workflows rely on expensive black-box evaluations, requiring sequential decisions that must both improve task performance and reduce uncertainty. Bayesian optimization (BO) and Bayesian experimental design (BED) provide powerful but largely separate treatments of goal-directed optimization and information-seeking experimentation, leaving limited guidance for hybrid settings in which learning and optimization are intrinsically coupled. We propose Pragmatic Curiosity (PraC), a unified framework for hybrid learning and optimization via active inference. PraC evaluates candidate queries by trading information gain about a task-relevant latent symbol against an expected regret-based potential over outcomes. This formulation foregrounds three operational design choices: which latent quantity should be clarified, how task value is encoded as regret, and how strongly information gain should be exchanged agains
    
[^352]: CoreQ：面向大语言模型量化的失配校正逐次舍入方法

    CoreQ: Mismatch-Corrected Successive Rounding for LLM Quantization

    [https://arxiv.org/abs/2602.05902](https://arxiv.org/abs/2602.05902)

    CoreQ是一种免学习的大语言模型后训练量化框架，通过自适应调节逐层失配校正的强度，避免了完全校正导致的校准数据过拟合以及固定校正系数忽略层间差异的问题。

    

    后训练量化通过将预训练权重映射为低比特格式而无需重新训练，使大语言模型得以高效部署，通常利用小型校准集来最小化逐层校准目标。然而，这种顺序式流程会引发失配问题：先前量化层的误差会改变后续层接收到的输入，导致激活值偏离全精度模型的激活值。近期方法引入了失配感知的校准目标来补偿这一效应，但对于观察到的失配应在多大程度上偏移每层的校准目标，仍未给出解答。我们证明，完全应用经验失配校正可能导致对有限校准数据的过拟合，而使用固定的校正系数则会忽略各层之间失配估计的差异。为解决这些局限，我们提出了CoreQ——一种免学习的PTQ框架，它使用几何……（原文摘要在此处被截断）

    arXiv:2602.05902v3 Announce Type: replace-cross  Abstract: Post-training quantization (PTQ) enables efficient deployment of large language models by mapping pretrained weights to low-bit formats without retraining, typically using a small calibration set to minimize a layer-wise calibration objective. However, this sequential procedure induces a mismatch: errors from earlier quantized layers alter the inputs received by later layers, causing the activations to deviate from those of the full-precision model. Recent approaches introduce mismatch-aware calibration objectives to compensate for this effect, but leave open how much of the observed mismatch should shift each layer's calibration target. We show that fully applying the empirical mismatch correction can overfit finite calibration data, while using a fixed correction coefficient ignores variation in mismatch estimates across layers. To address these limitations, we propose CoreQ, a learning-free PTQ framework that uses a geometri
    
[^353]: 目标化合成控制方法

    Targeted Synthetic Control Method

    [https://arxiv.org/abs/2602.04611](https://arxiv.org/abs/2602.04611)

    本文提出目标化合成控制（TSC）方法，一种新型两阶段估计器，通过目标化去偏更新直接估计反事实结果，在保证权重稳定性和凸组合可解释性的同时，可灵活结合任意机器学习模型。

    

    合成控制法（SCM）通过构建一个由未处理控制单元加权组合而成的反事实结果，使其匹配处理前的结果轨迹，从而在具有单一处理单元的面板数据中估计因果效应。在本文中，我们提出了目标化合成控制（TSC）方法，这是一种直接估计反事实结果的新型两阶段估计器。具体而言，我们的TSC方法：（1）产生一个目标化去偏估计器，即通过目标化更新来精炼初始权重，从而产生更稳定的权重；（2）确保最终的反事实估计是观测到的控制结果的凸组合，从而使合成控制权重能够被直接解释。TSC方法灵活多样，可以用任意机器学习模型进行实例化。在方法学上，TSC从一个初始的合成控制权重集合出发，通过一维目标化更新……

    arXiv:2602.04611v4 Announce Type: replace-cross  Abstract: The synthetic control method (SCM) estimates causal effects in panel data with a single-treated unit by constructing a counterfactual outcome as a weighted combination of untreated control units that matches the pre-treatment trajectory. In this paper, we introduce the targeted synthetic control (TSC) method, a new two-stage estimator that directly estimates the counterfactual outcome. Specifically, our TSC method (1) yields a targeted debiasing estimator, in the sense that the targeted updating refines the initial weights to produce more stable weights; and (2) ensures that the final counterfactual estimation is a convex combination of observed control outcomes to enable direct interpretation of the synthetic control weights. TSC is flexible and can be instantiated with arbitrary machine learning models. Methodologically, TSC starts from an initial set of synthetic-control weights via a one-dimensional targeted update through 
    
[^354]: 整流流的低维自适应：基于扩散与随机定位的视角

    Low-Dimensional Adaptation of Rectified Flow: A Diffusion and Stochastic Localization Perspective

    [https://arxiv.org/abs/2601.15500](https://arxiv.org/abs/2601.15500)

    本文证明整流流能自动适应目标分布的内在低维结构，在端点自适应时间离散化方案下达到 O(k/ε) 的采样迭代复杂度，并建立了DDPM与随机定位之间的新颖联系，证明DDPM等价于整流流的随机化版本。

    

    近年来，整流流凭借其生成效率和最先进的性能而广受欢迎。在本文中，我们研究了整流流如何自动适应目标分布支撑集的内在低维性，从而加速采样过程。我们证明，使用精心设计的端点自适应时间离散化方案以及足够精确的漂移估计，整流流采样器可以获得 O(k/ε) 量级的迭代复杂度（在对数因子范围内），其中 ε 是总变差距离下的精度，k 是目标分布的内在维度。此外，我们通过在整流流与随机定位过程之间建立一种新颖的联系，证明了去噪扩散概率模型（DDPM）过程等价于整流流的随机化版本。基于这一联系，我们进一步设计了一种随……

    arXiv:2601.15500v5 Announce Type: replace-cross  Abstract: In recent years, Rectified flow (RF) has gained considerable popularity largely due to its generation efficiency and state-of-the-art performance. In this paper, we investigate how well RF automatically adapts to the intrinsic low dimensionality of the support of the target distribution to accelerate sampling. We show that, using a carefully designed end-point adapted time-discretization scheme and with sufficiently accurate drift estimates, the RF sampler enjoys an iteration complexity of order $O(k/\varepsilon)$ (up to log factors), where $\varepsilon$ is the precision in total variation distance and $k$ is the intrinsic dimension of the target distribution. In addition, we show that the denoising diffusion probabilistic model (DDPM) procedure is equivalent to a stochastic version of RF by establishing a novel connection between these processes and stochastic localization. Building on this connection, we further design a stoc
    
[^355]: 基于贝叶斯线性任务模型的元强化学习

    Meta-RL with Bayesian Linear Task Models

    [https://arxiv.org/abs/2512.20974](https://arxiv.org/abs/2512.20974)

    提出GLiBRL深度贝叶斯强化学习框架，通过结合广义线性任务模型与可学习非线性基函数，实现精确的共轭贝叶斯后验更新和闭式边缘似然，消除了变分推断带来的近似误差，并可灵活兼容离策略与在线策略算法。

    

    深度贝叶斯强化学习通过推断潜在的转移和奖励模型来适应未见过的任务，但现有方法通常依赖于变分后验和证据下界，这会引入近似误差和不稳定的任务表示。我们提出了GLiBRL，这是一个将广义线性任务模型与可学习非线性基函数相结合的深度贝叶斯强化学习框架。GLiBRL具有共轭贝叶斯推断的特性，能够对任务参数和模型噪声进行精确的顺序后验更新，同时提供闭式边缘似然，从而完全消除了变分推断。该更新天然具有置换不变性，使GLiBRL能够与离策略和在线策略算法集成。GLiBRL还学习了具有精确核恒等关系的任务表示，将任务表示之间的距离与任务上下文上的核差异联系起来。与八个代表性方法相比……

    arXiv:2512.20974v4 Announce Type: replace-cross  Abstract: Deep Bayesian reinforcement learning adapts to unseen tasks by inferring latent transition and reward models, but existing methods typically rely on variational posteriors and evidence lower bounds, introducing approximation error and unstable task representations. We introduce GLiBRL, a deep Bayesian RL framework that combines generalised linear task models with learnable non-linear basis functions. GLiBRL features conjugate Bayesian inference, yielding exact, sequential posterior updates over task parameters and model noise, together with a closed-form marginal likelihood that eliminates variational inference. The update is naturally permutation-invariant, allowing GLiBRL to integrate with both off- and on-policy algorithms. GLiBRL also learns task representation admitting an exact kernel identity, relating distances between task representations to kernel discrepancies over the task contexts. Compared against eight representa
    
[^356]: TinyMyo：一个用于边缘端灵活肌电信号处理的微型基础模型

    TinyMyo: a Tiny Foundation Model for Flexible EMG Signal Processing at the Edge

    [https://arxiv.org/abs/2512.15729](https://arxiv.org/abs/2512.15729)

    TinyMyo是一个仅360万参数的微型肌电基础模型，通过自监督掩码重建预训练支持多种下游任务，手势分类性能超越现有肌电基础模型，并可在超低功耗微控制器上实现边缘端部署。

    

    目标：表面肌电图（EMG）广泛应用于生物力学、康复、假肢控制和人机接口领域，但跨受试者、记录系统和采集协议的泛化仍然具有挑战性。现有的肌电基础模型（FM）对多个下游任务的支持以及低功耗设备上的部署能力有限。方法：我们提出了TinyMyo，一个拥有360万参数的Transformer编码器，通过自监督掩码重建在公开的肌电数据集上进行预训练。TinyMyo只需对任务特定的预测头进行极小改动即可支持多种任务。结果：我们在手势分类、手部运动学回归、语音产生和语音识别上评估了TinyMyo。在手势分类任务中，TinyMyo优于之前报道的肌电基础模型，在NinaPro DB5上达到87.98%的准确率，在UCI-EMG上达到97.1%，在EPN-612上达到96.6%。在GAP9超低功耗微控制器上……

    arXiv:2512.15729v3 Announce Type: replace-cross  Abstract: Objective: Surface electromyography (EMG) is widely used in biomechanics, rehabilitation, prosthetic control, and human-machine interfaces. Generalization across subjects, recording systems, and acquisition protocols remains challenging. Existing EMG foundation models (FMs) have limited support for multiple downstream tasks and deployment on low-power devices. Methods: We introduce TinyMyo, a 3.6-million-parameter Transformer encoder pre-trained on public EMG datasets using self-supervised masked reconstruction. TinyMyo supports multiple tasks with minimal changes to the task-specific prediction head. Results: We evaluated TinyMyo on hand gesture classification, hand kinematic regression, speech production, and speech recognition. For gesture classification, TinyMyo outperformed previously reported EMG FMs, achieving accuracies of 87.98% on NinaPro DB5, 97.1% on UCI-EMG, and 96.6% on EPN-612. On the GAP9 ultra-low-power microco
    
[^357]: 集成预测与多期投资组合优化

    Integrated Prediction and Multi-period Portfolio Optimization

    [https://arxiv.org/abs/2512.11273](https://arxiv.org/abs/2512.11273)

    本文提出IPMO模型，将收益预测与多期均值-方差投资组合优化整合为端到端的单一框架，解决了传统两阶段方法中预测与决策脱节以及忽略交易成本的问题。

    

    多期投资组合优化对于实际投资组合管理非常重要，因为它考虑了交易成本、路径依赖风险以及单期模型无法捕捉的交易决策的跨期结构。经典方法通常遵循两阶段框架：首先采用机器学习算法产生与实际收益紧密拟合的预测值，然后将这些预测值用于下游的投资组合优化问题以确定资产权重。这种分离导致预测与决策结果之间产生根本性的错位，同时也忽略了交易成本的影响。为了弥合这一差距，最近的研究提出了端到端学习的思想，将两个阶段整合到单一流程中。本文提出了IPMO（集成预测与多期投资组合优化），一个用于多期均值-方差投资组合优化的模型。

    arXiv:2512.11273v4 Announce Type: replace-cross  Abstract: Multi-period portfolio optimization is important for real portfolio management, as it accounts for transaction costs, path-dependent risks, and the intertemporal structure of trading decisions that single-period models cannot capture. Classical methods usually follow a two-stage framework: machine learning algorithms are employed to produce forecasts that closely fit the realized returns, and the predicted values are then used in a downstream portfolio optimization problem to determine the asset weights. This separation leads to a fundamental misalignment between predictions and decision outcomes, while also ignoring the impact of transaction costs. To bridge this gap, recent studies have proposed the idea of end-to-end learning, integrating the two stages into a single pipeline. This paper introduces IPMO (Integrated Prediction and Multi-period Portfolio Optimization), a model for multi-period mean-variance portfolio optimizat
    
[^358]: 当特征战胜噪声：一种噪声驱动的假设检验特征选择方法

    When Features Beat Noise: A Noise-Driven Hypothesis Testing Method for Feature Selection

    [https://arxiv.org/abs/2511.20851](https://arxiv.org/abs/2511.20851)

    本文提出一种噪声驱动的特征选择方法，通过将每个特征的重要性与随机噪声特征中的最大重要性进行基于非参数bootstrap的假设检验，为特征选择提供了坚实的统计理论基础和显著性评估准则。

    

    特征选择一直是机器学习和人工智能领域的一项艰巨挑战，日益复杂的高维数据集需要有原则性的策略来筛选出信息量最大的预测变量。尽管被广泛采用，许多成熟的技术（如Boruta、递归特征消除以及其他基于封装器的方法）都存在明显的局限性：有些方法计算成本高昂，而另一些方法则缺乏明确的统计驱动的停止准则，或无法评估其重要性分数的显著性。本文提出了一种新颖的特征选择方法来解决这些局限。我们的方法引入多个随机噪声特征，并将每个真实特征的重要性与这些噪声特征中的最大重要性值进行比较，同时结合基于非参数bootstrap的假设检验框架，以建立坚实的理论基础。我们建立……

    arXiv:2511.20851v4 Announce Type: replace-cross  Abstract: Feature selection has remained a daunting challenge in machine learning and artificial intelligence, where increasingly complex, high-dimensional datasets demand principled strategies for isolating the most informative predictors. Despite widespread adoption, many established techniques such as Boruta, Recursive Feature Elimination and other wrapper based methods suffer from notable limitations; some incur substantial computational cost, while others offer no definite statistical driven stopping criteria or assess the significance of their importance scores. This paper proposes a novel feature selection method that addresses these limitations. Our approach introduces multiple random noise features and evaluates each feature's importance against the maximum importance value among these noise features incorporating a non-parametric bootstrap-based hypothesis testing framework to establish a solid theoretical foundation. We establ
    
[^359]: 连续偏好贝叶斯优化

    Consecutive Preferential Bayesian Optimization

    [https://arxiv.org/abs/2511.05163](https://arxiv.org/abs/2511.05163)

    该论文提出连续偏好贝叶斯优化（CPBO），通过在每次比较中复用上一个候选方案来降低候选方案的生产成本，并在概率偏好模型中引入最小可觉差（JND）阈值以处理评估者的感知模糊性，同时适配信息论采集策略来高效逼近未知最优解。

    

    偏好贝叶斯优化通过依赖人类专家进行的最少数量的比较评估，能够对那些成本高昂或难以直接测量的目标进行优化。然而，生成用于评估的候选方案本身也往往代价高昂，而现有的偏好优化方法并未考虑这种生产成本。我们提出了面向连续（consecutive）设定的连续偏好贝叶斯优化（CPBO），通过在每次比较中复用上一个候选方案来降低生产成本。我们还考虑了提供反馈的评估者的感知模糊性，通过在概率偏好模型中引入最小可觉差阈值，来刻画其对微小效用差异表现出的无差别态度。我们将一种基于信息论的采集策略适配到该设定中，在预设的约束下选择对未知最优解信息量最大的新配置。

    arXiv:2511.05163v2 Announce Type: replace  Abstract: Preferential Bayesian optimization allows optimization of objectives that are either expensive or difficult to measure directly, by relying on a minimal number of comparative evaluations done by a human expert. Generating candidate solutions for evaluation is also often expensive, but existing preferential optimization methods do not account for this production cost. We introduce Consecutive Preferential Bayesian Optimization (CPBO) for the consecutive setting, where production cost is reduced by reusing the previous candidate in each comparison. We also account for the perceptual ambiguity of the oracle providing the feedback by incorporating a Just-Noticeable Difference threshold into a probabilistic preference model to capture indifference to small utility differences. We adapt an information-theoretic acquisition strategy to this setting, selecting new configurations that are most informative about the unknown optimum under a pre
    
[^360]: 评估信息性删失下生存分析中异质性处理效应的稳健性

    Assessing the robustness of heterogeneous treatment effects in survival analysis under informative censoring

    [https://arxiv.org/abs/2510.13397](https://arxiv.org/abs/2510.13397)

    该论文提出一个少假设框架及新颖的模型无关元学习器SurvB-learner，通过部分识别推导条件平均处理效应的边界，从而在信息性删失偏差下评估生存分析中异质性治疗效应估计的稳健性，并识别治疗有效的患者亚群。

    

    arXiv:2510.13397v4 公告类型：替换 摘要：脱落（中途退出）在临床研究中很常见，高达一半的患者可能因副作用或其他原因提前退出研究。当脱落具有信息性时（即与生存时间相关），会引入删失偏差，从而导致处理效应估计也出现偏差。在本文中，我们提出了一个少假设框架，用于在面临删失偏差时评估生存分析中条件平均处理效应（CATE）估计的稳健性。与现有依赖强假设（如非信息性删失）来获得点估计的工作不同，我们采用部分识别方法推导出CATE的信息性边界。由此，我们的框架有助于识别出尽管存在信息性删失但治疗仍然有效的患者亚群。我们进一步提出了一种新颖的模型无关元学习器，称为SurvB-learner，用于估计这些边界，该方法可与任意机器学习模型结合使用，并且……

    arXiv:2510.13397v4 Announce Type: replace  Abstract: Dropout is common in clinical studies, with up to half of patients leaving early due to side effects or other reasons. When dropout is informative (i.e., dependent on survival time), it introduces censoring bias, because of which treatment effect estimates are also biased. In this paper, we propose an assumption-lean framework to assess the robustness of conditional average treatment effect (CATE) estimates in survival analysis when facing censoring bias. Unlike existing works that rely on strong assumptions, such as non-informative censoring, to obtain point estimation, we use partial identification to derive informative bounds on the CATE. Thereby, our framework helps to identify patient subgroups where treatment is effective despite informative censoring. We further propose a novel model-agnostic meta-learner, called SurvB-learner, to estimate the bounds that can be used in combination with arbitrary machine-learning models, and t
    
[^361]: 基于弱偏好对齐的可控生成式轨迹预测

    Controllable Generative Trajectory Prediction via Weak Preference Alignment

    [https://arxiv.org/abs/2510.10731](https://arxiv.org/abs/2510.10731)

    提出PrefCVAE增强型CVAE框架，利用弱标注偏好对为潜变量赋予语义属性，从而在不损失预测准确性的前提下实现可控、语义有意义的轨迹预测。

    

    条件变分自编码器（CVAE）等深度生成模型在自动驾驶车辆规划中预测周围智能体的轨迹方面展现出巨大前景。最先进的模型在此类预测任务中已取得卓越的准确性。除了准确性之外，多样性对于安全规划同样至关重要，因为人类行为本质上具有不确定性和多模态性。然而，现有方法普遍缺乏一种生成可控多样化轨迹的方案，而对于安全规划而言，可控的多样化轨迹可以说比随机多样化的轨迹更有用。为解决这一问题，我们提出了PrefCVAE，这是一种增强型CVAE框架，它利用弱标注的偏好对来为潜变量赋予语义属性。以平均速度作为示例属性，我们证明了PrefCVAE能够在不降低基线准确性的情况下，实现可控且语义上有意义的预测。我们的结果表明……

    arXiv:2510.10731v2 Announce Type: replace-cross  Abstract: Deep generative models such as conditional variational autoencoders (CVAEs) have shown great promise for predicting trajectories of surrounding agents in autonomous vehicle planning. State-of-the-art models have achieved remarkable accuracy in such prediction tasks. Besides accuracy, diversity is also crucial for safe planning because human behaviors are inherently uncertain and multimodal. However, existing methods generally lack a scheme to generate controllably diverse trajectories, which is arguably more useful than randomly diversified trajectories, to the end of safe planning. To address this, we propose PrefCVAE, an augmented CVAE framework that uses weakly labeled preference pairs to imbue latent variables with semantic attributes. Using average velocity as an example attribute, we demonstrate that PrefCVAE enables controllable, semantically meaningful predictions without degrading baseline accuracy. Our results show th
    
[^362]: 基于Lipschitz空间中预算分配的超参数优化

    Hyperparameter Optimization via Budget Allocation in Lipschitz Spaces

    [https://arxiv.org/abs/2510.10605](https://arxiv.org/abs/2510.10605)

    该论文将超参数优化形式化为“未知值探测”决策问题，提出了在Lipschitz空间结构下结合探索与价值导向集中的预算分配策略，证明了接近最优的最坏情况近似保证，并在250多个实验设置中取得了一致的性能提升。

    

    超参数优化和模型选择通常需要在多个候选配置之间分配有限的评估预算，其中性能会随着额外的训练或资源的增加而提升。我们研究每个配置在预算上具有未知性能曲线的场景。配置之间的相似性会导致其曲线之间的相似性，从而捕捉搜索空间中的几何结构。我们将由此产生的决策问题形式化为“未知值探测”，并提出了结合对配置空间进行原则性探索与基于观测证据的价值导向集中策略的预算分配方法。我们为该方法证明了最坏情况近似保证，并通过匹配相应的难度障碍证明这些保证接近最优。在涵盖250多个实验设置的大规模基准套件上的大量实验表明了一致的性能提升。

    arXiv:2510.10605v3 Announce Type: replace  Abstract: Hyperparameter optimization and model selection often require allocating a limited evaluation budget across many candidate configurations, where performance improves with additional training or resources. We study this setting when each configuration has an unknown performance curve over budget. Similarity between configurations induces similarity between their curves, capturing geometric structure in the search space. We formalize the resulting decision problem as Unknown Value Probing and propose budget allocation strategies that combine principled exploration of the configuration space with value-informed concentration on regions supported by observed evidence. We prove worst-case approximation guarantees for our approach and show these guarantees are near-optimal by matching corresponding hardness barriers. Extensive experiments on large benchmark suites spanning more than 250 experimental settings demonstrate consistent gains ov
    
[^363]: 更大并不总是意味着记忆：过参数化扩散模型的早停

    Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models

    [https://arxiv.org/abs/2505.16959](https://arxiv.org/abs/2505.16959)

    过参数化扩散模型在训练中先于记忆化发生之前逐步实现泛化，记忆化时间与数据集大小成正比，因此泛化与记忆化本质上是时间尺度的竞争，可通过早停来避免记忆化。

    

    扩散概率模型已成为现代生成式人工智能的基石，然而其背后的泛化机制仍知之甚少。事实上，如果这些模型完美地最小化其训练损失，它们将只会生成属于训练集的数据，即产生记忆化现象，这正如在过参数化机制中被实证观察到的那样。我们通过展示高度过参数化的扩散模型中，自然数据领域的泛化能力在训练过程中、在记忆化发生之前被逐步实现，来重新审视这一观点。我们的结果涵盖从图像到语言的扩散模型，系统性地支持如下经验规律：记忆化时间与数据集大小成正比，这与低噪声情形下拟合经验得分函数所需时间的核回归界相一致。因此，泛化与记忆化最好被理解为不同时间尺度之间的竞争。我们展示了这一现象……

    arXiv:2505.16959v3 Announce Type: replace  Abstract: Diffusion probabilistic models have become a cornerstone of modern generative AI, yet the mechanisms underlying their generalization remain poorly understood. In fact, if these models were perfectly minimizing their training loss, they would just generate data belonging to their training set, i.e., memorize, as empirically found in the overparameterized regime. We revisit this view by showing that, in highly overparameterized diffusion models, generalization in natural data domains is progressively achieved during training before the onset of memorization. Our results, ranging from image to language diffusion models, systematically support the empirical law that memorization time is proportional to the dataset size, consistent with a kernel-regression bound on the time required to fit the empirical score at low noise. Generalization vs. memorization is then best understood as a competition between time scales. We show that this pheno
    
[^364]: 基于参考奖励的强化学习用于语言模型推理与对齐

    Reinforcement Learning with Reference-Based Rewards for Language Model Reasoning and Alignment

    [https://arxiv.org/abs/2504.09895](https://arxiv.org/abs/2504.09895)

    提出RL2R方法，利用参考答案与轻量级语义相似度指标（如BERTScore）作为奖励信号，为语言模型的推理与对齐提供统一且低成本的强化学习后训练方案。

    

    强化学习（RL）支撑着大语言模型（LLM）后训练的两大主要方向：推理，即引导模型进行深思熟虑、可验证的问题求解；以及对齐，即引导模型表现出有益、无害且诚实的行为。两者都依赖于奖励设计。推理依赖于对最终答案的基于规则的验证，其二元反馈无法感知中间推导过程；而过程奖励模型（PRM）需要步骤级标注，成本过高难以规模化。对齐则依赖于在二元偏好对上训练的奖励模型（RM），使数据收集和模型训练都非常耗费资源。我们提出RL2R——基于参考奖励的强化学习，作为一种统一且低成本的替代方案。利用已有的或蒸馏得到的参考答案，RL2R通过轻量级指标（如BERTScore）度量采样生成结果与参考答案或解题过程之间的语义相似度，以相似度……（原文在此截断）

    arXiv:2504.09895v4 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) powers two major lines of large language model (LLM) post-training: reasoning, which elicits deliberate, verifiable problem solving, and alignment, which steers models toward helpful, harmless, and honest behavior. Both hinge on reward design. Reasoning relies on rule-based verification of final answers, whose binary feedback is blind to intermediate derivations, whereas process reward models (PRMs) require step-level annotations too costly to scale. Alignment depends on reward models (RMs) trained on binary preference pairs, making both data collection and model training resource-intensive. We propose RL2R, reinforcement learning with reference-based rewards, as a unified and inexpensive alternative. Using pre-existing or distilled references, RL2R measures semantic similarity with lightweight metrics such as BERTScore between a sampled generation and a reference answer or solution. Similarity prese
    
[^365]: 适者生存的扰动：面向鲁棒泛化的对抗分布不确定性感知演化

    Survival of the Fittest Perturbations: Uncertainty-Aware Evolution of Adversarial Distributions for Robust Generalization

    [https://arxiv.org/abs/2411.02871](https://arxiv.org/abs/2411.02871)

    提出了进化式不确定性感知分布对抗训练（EvoUAD-AT），通过在对抗锚点周围维护实例级的对抗特征统计搜索分布并进行进化式威胁选择，以缓解单点对抗训练忽略对抗变异性的问题，从而提升模型的鲁棒泛化能力。

    

    尽管深度学习在多种场景中取得了广泛的成功，但其对对抗样本的脆弱性阻碍了实际部署，这使得对抗训练成为提升模型鲁棒性的主流方法。然而，逐点对抗训练将每个干净样本映射为单个对抗样本，忽略了周围的对抗变异性以及由此引起的特征统计偏移。为解决这些问题，我们提出了进化式不确定性感知分布对抗训练（EvoUAD-AT）。具体而言，我们设计了一个进化式对抗不确定性建模模块，该模块在对抗生成的锚点周围维护一个针对对抗特征统计的实例级搜索分布。候选对抗样本在当前模型下进行评估，根据其对良性参考的威胁程度被选择，并用于自适应调整分布；策略参数在各训练轮次之间得以保留。为避免锚定在被误分类的干净样本上，我们引入（摘要在此处被截断）

    arXiv:2411.02871v2 Announce Type: replace  Abstract: Despite deep learning's broad successes across diverse scenarios, its vulnerability to adversarial examples impedes deployment, making adversarial training a leading approach to improve robustness. However, pointwise adversarial training maps each clean sample to a single adversary, overlooking surrounding adversarial variability and the associated shifts in feature statistics. To address these issues, we propose Evolutionary Uncertainty-Aware Distributional Adversarial Training (EvoUAD-AT). Specifically, we design an evolutionary adversarial uncertainty modeling module that maintains an instance-wise search distribution over adversarial feature statistics around an adversarially generated anchor. Candidates are evaluated under the current model, selected by their threat to a benign reference, and used to adapt the distribution; strategy parameters are retained across epochs. To avoid anchoring on misclassified clean samples, we refi
    
[^366]: 上下文学习器混合模型

    Mixtures of In-Context Learners

    [https://arxiv.org/abs/2411.02830](https://arxiv.org/abs/2411.02830)

    提出MoICL方法，将上下文学习的演示示例子集视为专家并学习加权函数来合并其输出分布，在提升分类性能的同时减少推理时间，并显著增强了对域外、不平衡和噪声示例的鲁棒性。

    

    上下文学习通过提供示例（演示）来适应大语言模型，而无需微调模型参数；然而，它无法区分不同的示例，并且会使Transformer大语言模型的复杂度呈平方级增长，从而耗尽内存。作为解决方案，我们提出了上下文学习器混合模型（MoICL），这是一种新颖的方法，它将演示示例的子集视为“专家”，并基于训练集学习一个加权函数来合并它们的输出分布。在实验中，与一组强基线相比，我们在7个分类数据集中的5个上展示了性能提升（与ICL和LENS相比最高提升+13%）。此外，我们通过减少以更少示例达到相同性能所需的推理时间，改善了ICL的帕累托前沿。最后，MoICL对域外示例（最高+11%）、不平衡示例（最高+49%）或噪声示例（最高+38%）更加鲁棒，或者可以过滤掉这些示例。

    arXiv:2411.02830v2 Announce Type: replace-cross  Abstract: In-context learning (ICL) adapts LLMs by providing demonstrations without fine-tuning the model parameters; however, it does not differentiate between demonstrations and quadratically increases the complexity of Transformer LLMs, exhausting the memory. As a solution, we propose Mixtures of In-Context Learners (MoICL), a novel approach to treat subsets of demonstrations as experts and learn a weighting function to merge their output distributions based on a training set. In our experiments, we show performance improvements on 5 out of 7 classification datasets compared to a set of strong baselines (up to +13\% compared to ICL and LENS). Moreover, we enhance the Pareto frontier of ICL by reducing the inference time needed to achieve the same performance with fewer demonstrations. Finally, MoICL is more robust to out-of-domain (up to +11\%), imbalanced (up to +49\%), or noisy demonstrations (up to +38\%) or can filter these out fr
    
[^367]: 无需费力学注意力，直接“种下”它：通过学习排序预训练注意力

    Don't Pay Attention, PLANT It: Pretraining Attention via Learning-to-Rank

    [https://arxiv.org/abs/2410.23066](https://arxiv.org/abs/2410.23066)

    提出PLANT，一种即插即用策略，通过基于互信息增益的学习排序模型预训练标签注意力来初始化注意力模块，可与大语言模型骨干集成，在极端多标签文本分类任务上超越现有最先进方法，且在罕见标签上收益最大。

    

    最先进的极端多标签文本分类模型依赖多标签注意力机制来聚焦输入文本中的关键标记，但学习良好的注意力权重十分具有挑战性。我们提出了PLANT（Pretrained and Leveraged AtteNTion，预训练并利用的注意力），这是一种即插即用的策略，用于初始化构建在大语言模型输出表示之上的标签注意力模块。PLANT的工作方式是：在端到端微调之前，利用一个基于互信息增益训练的学习排序模型来“种下”（植入）标签特定的注意力。这种与架构无关的方法可与多种大语言模型骨干网络（如Mistral-7B、Llama-3.1-8B、DeepSeek-V3和Phi-3-mini）集成。PLANT在ICD编码、法律主题分类和内容推荐等任务上超越了现有最先进方法，并在医疗领域之外的受控对比实验中改进了微调后的大语言模型。其收益在罕见标签上最为显著，尽管免训练的大语言模型抽取基线仍具有竞争力。

    arXiv:2410.23066v5 Announce Type: replace  Abstract: State-of-the-art Extreme Multi-Label Text Classification models rely on multi-label attention to focus on key tokens in input text, but learning good attention weights is challenging. We introduce PLANT (Pretrained and Leveraged AtteNTion), a plug-and-play strategy for initializing the label-attention module that operates on top of an LLM's output representations. PLANT works by planting label-specific attention using a Learning-to-Rank model trained on mutual information gain, before end-to-end fine-tuning. This architecture-agnostic approach integrates with LLM backbones (e.g., Mistral-7B, Llama-3.1-8B, DeepSeek-V3, and Phi-3-mini). PLANT outperforms SOTA methods across tasks like ICD coding, legal topic classification, and content recommendation, and improves fine-tuned LLMs in controlled comparisons outside the medical domain. Gains are largest on rare labels, although a training-free LLM extraction baseline remains competitive o
    
[^368]: 基于深度强化学习的核聚变反应堆设计优化

    Design Optimization of Nuclear Fusion Reactor through Deep Reinforcement Learning

    [https://arxiv.org/abs/2409.08231](https://arxiv.org/abs/2409.08231)

    该研究将深度强化学习中的奖励约束策略优化（RCPO）应用于托卡马克聚变反应堆设计，在满足四个稳定运行约束的条件下，实现了对建造成本、能量增益和能量约束时间的多目标黑箱优化。

    

    在这项研究中，我们探讨了深度强化学习（DRL）作为托卡马克聚变反应堆设计中带约束的多目标优化的实用工具。该任务是优化设计配置，在满足稳定运行所需的四个运行限制条件下，最小化建造成本，同时实现高能量增益和长能量约束时间。由于物理与工程的耦合评估，所得的目标和约束是黑箱的，这使得采用传统搜索策略寻找最优解变得具有挑战性。为解决这一问题，我们在托卡马克聚变反应堆设计中实现了奖励约束策略优化（RCPO）方法。该方法将多个约束整合到奖励函数中，引导策略向可行解方向优化，并简化了设计优化过程。我们的数值结果表明，所提出的基于DRL的方法能够找到更多可行的设计方案。

    arXiv:2409.08231v2 Announce Type: replace-cross  Abstract: In this research, we investigate deep reinforcement learning (DRL) as a practical tool for constrained, multi-objective optimization for tokamak fusion reactor designs. The task is to optimize the design configurations that minimize building cost while achieving high energy gain and long energy confinement time, subject to the four operational limits required for stable operation. Due to the coupled physics-engineering evaluation, the resulting objectives and constraints are black-box, making it challenging to find optimal solutions with conventional search strategies. To address this, we implement Reward-Constrained Policy Optimization (RCPO) for tokamak fusion reactor design. This method integrates multiple constraints into the reward functions, guiding the policy towards feasible solutions and simplifying the design optimization process. Our numerical results show that the proposed DRL-based approach finds more feasible desi
    
[^369]: Co-Pilot for Health: 个性化算法AI引导改善健康结果

    Co-Pilot for Health: Personalized Algorithmic AI Nudging to Improve Health Outcomes. (arXiv:2401.10816v1 [cs.HC])

    [http://arxiv.org/abs/2401.10816](http://arxiv.org/abs/2401.10816)

    该研究通过使用基于图神经网络的推荐系统和来自可穿戴设备的健康行为数据，设计并实施了一个人工智能驱动平台，实现了个性化和情境引导，能够提高参与者的日常活动水平和中等至剧烈运动时长。

    

    在全球范围内自动塑造大型人群的健康行为，跨可穿戴设备类型和疾病状况具有巨大的潜力来改善全球健康结果。我们设计并实施了一个基于图神经网络（GNN）推荐系统和来自可穿戴健身设备的精细健康行为数据的人工智能驱动平台，用于数字算法引导。在此我们描述了该平台在新加坡针对$n=84,764$个个体的12周期间进行个性化和情境引导的有效性结果。我们统计验证了目标组中接受此类AI优化日常引导的参与者相较于控制组中未接受任何引导的匹配参与者，其每天的步数增加了6.17%（$p = 3.09\times10^{-4}$），每周中等至剧烈运动（MVPA）分钟增加了7.61%（$p = 1.16\times10^{-2}$）。此外，此类引导非常可行。

    The ability to shape health behaviors of large populations automatically, across wearable types and disease conditions at scale has tremendous potential to improve global health outcomes. We designed and implemented an AI driven platform for digital algorithmic nudging, enabled by a Graph-Neural Network (GNN) based Recommendation System, and granular health behavior data from wearable fitness devices. Here we describe the efficacy results of this platform with its capabilities of personalized and contextual nudging to $n=84,764$ individuals over a 12-week period in Singapore. We statistically validated that participants in the target group who received such AI optimized daily nudges increased daily physical activity like step count by 6.17% ($p = 3.09\times10^{-4}$) and weekly minutes of Moderate to Vigorous Physical Activity (MVPA) by 7.61% ($p = 1.16\times10^{-2}$), compared to matched participants in control group who did not receive any nudges. Further, such nudges were very well r
    
[^370]: 预测模型的损失控制校准

    Loss-Controlling Calibration for Predictive Models. (arXiv:2301.04378v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.04378](http://arxiv.org/abs/2301.04378)

    本文提出了一种学习框架，用于对可交换数据进行损失控制预测的校准预测模型。通过引入保持交换性质的变换，我们证明了在获得测试标签时有有限样本控制保证，并发展了一种近似方法来构建预测器。这种方法是符合损失控制预测的自然扩展。

    

    我们提出了一个学习框架，用于对可交换数据进行损失控制预测的校准预测模型，扩展了我们最近提出的适用于更一般情况的符合损失控制预测。与之相比，通过提出的损失控制方法构建的预测器不限于集合预测器，并且损失函数可以是任何可测函数，而不需要单调性假设。为了以有效的方式控制损失值，我们引入保持交换性质的变换，以证明在获得测试标签时有有限样本控制保证，并然后发展出一种构建预测器的近似方法。这些变换可以建立在任何预定义的函数上，包括使用优化算法进行参数搜索。这种方法是符合损失控制预测的自然扩展，因为当集合预测器具有嵌套属性且损失函数为单调函数时，它可以化简为后者。

    We propose a learning framework for calibrating predictive models to make loss-controlling prediction for exchangeable data, which extends our recently proposed conformal loss-controlling prediction for more general cases. By comparison, the predictors built by the proposed loss-controlling approach are not limited to set predictors, and the loss function can be any measurable function without the monotone assumption. To control the loss values in an efficient way, we introduce transformations preserving exchangeability to prove finite-sample controlling guarantee when the test label is obtained, and then develop an approximation approach to construct predictors. The transformations can be built on any predefined function, which include using optimization algorithms for parameter searching. This approach is a natural extension of conformal loss-controlling prediction, since it can be reduced to the latter when the set predictors have the nesting property and the loss functions are mono
    

