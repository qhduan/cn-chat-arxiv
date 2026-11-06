# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [User Embeddings are Superpositions of Interpretable Behavioral Modes](https://arxiv.org/abs/2611.08487) | 该论文提出用户嵌入可被理解为可解释行为模式的稀疏叠加，并证明稀疏自编码器（SAE）能有效恢复这些行为模式，从而显著提升用户嵌入的可解释性。 |
| [^2] | [Scientist World: Grow a Research World for AI-Scientists](https://arxiv.org/abs/2611.08478) | 该论文提出Scientist World，将每篇研究论文转化为可执行的研究环境，用于评测和训练AI科学家，并发现前沿AI科学家在74%的情况下已能超越人类研究者的成果。 |
| [^3] | [XStack: A Benchmark for ML Repo Migration](https://arxiv.org/abs/2611.08472) | 该论文提出 XStack 基准，通过要求智能体将 PyTorch 代码仓库完整迁移为可在 TPU 上运行的 JAX 代码仓库，来评估其跨框架、跨硬件的长时程迁移能力，目前最强模型通过率仅约 75%。 |
| [^4] | [Same Behavior, Different Mechanism: What Finite Experiments Can Certify About Integrated Information](https://arxiv.org/abs/2611.08469) | 本文在具有已知真实答案的小型随机网络中精确刻画了整合信息 φ 等机制级量何时可被识别、连续且可由有限数据认证，并证明输入输出行为完全相同的网络可携带截然不同的整合信息（0.28 比特 vs 0 比特），但只需 4-53 次内部记录试验即可将其区分。 |
| [^5] | [Towards interpreting any residual stream direction by scaling activation inversion](https://arxiv.org/abs/2611.08465) | 提出最大激活示例元模型（MAEM），利用无标签强化学习目标将残差流的任意方向反演为强激活该方向的文本示例，在解释效果上超越了语料库搜索和自然语言自编码器。 |
| [^6] | [Adversarial Training with Large Step Sizes: Implicit Bias, Optimization, and Sharpness](https://arxiv.org/abs/2611.08457) | 本文研究了大恒定步长梯度下降在对抗训练中的动力学，从隐式偏差、优化速度和损失锐度演化三个方面揭示了其与标准训练的关键差异。 |
| [^7] | [Unpacking the Compute Efficiency of Strong-to-Weak OPD against GRPO](https://arxiv.org/abs/2611.08453) | 本研究系统对比了在线策略蒸馏（OPD）与GRPO的计算效率，发现使用外部教师时OPD效率不如GRPO，但当教师是学生自身的GRPO检查点时，OPD仅需少50倍的FLOPs即可达到相同的数学准确率。 |
| [^8] | [A Hidden Markov Preference Model for Online Prediction of Adversary Actions](https://arxiv.org/abs/2611.08445) | 本文提出一种隐马尔可夫偏好模型，将攻击者行为模式作为隐状态、在可行性掩码行动树上定义模式条件下的行动分布，从而实现对对手下一步行动的在线概率预测、信念更新及信息量化的理论保证。 |
| [^9] | [Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes](https://arxiv.org/abs/2611.08439) | 论文对俄语类型化决策交叉编码器 Ru-Jev 引入标签多形式措辞增强，发现该增强对由普通词和外来词构成的标签词汇表能带来一致的准确性提升（最高达 0.24），但对缩写和代号类标签没有增益。 |
| [^10] | [Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety](https://arxiv.org/abs/2611.08438) | 本论文系统综述了机制可解释性在AI安全领域的研究现状，探讨通过理解模型内部表示与计算来评估、预测和控制AI行为，能否真正使AI系统更加安全。 |
| [^11] | [Data-driven discovery of intrinsic Lie point symmetries of differential equation systems](https://arxiv.org/abs/2611.08421) | 该论文提出了一种将内蕴李点对称性发现转化为采样射流空间数据上算子理论问题的数据驱动框架，无需符号决定方程或控制系统的闭式表示，即可从微分方程数据中直接识别出有意义的连续对称性，并过滤掉冗余与几何平凡的方向。 |
| [^12] | [On the limitations of molecular representations](https://arxiv.org/abs/2611.08420) | 本文系统评估了分子表示方法的质量，发现最先进的分子嵌入模型在捕捉图结构、3D形状和ADMET相似性方面均不及传统指纹与理化描述符，且经典指纹自身也存在严重局限——在大规模数据库中可导致高达数十亿分子的表示不唯一，影响虚拟筛选的可靠性。 |
| [^13] | [Spectral and Subspace Dynamics of LoRA in Supervised Fine-Tuning and Reinforcement Learning](https://arxiv.org/abs/2611.08414) | 该研究通过追踪LoRA更新的奇异谱与子空间，发现强化学习相比监督微调表现出谱集中和更快的子空间旋转，并揭示了组均值基线减除是造成这两种差异的根本机制。 |
| [^14] | [Conformal Prediction for Time-Dependent PDEs](https://arxiv.org/abs/2611.08409) | 本文证明了时间依赖偏微分方程的共形预测覆盖保证在本质上难以实现——函数空间中相近时刻的分布可相互奇异，离散化中全变差距离随网格分辨率指数饱和——并给出了为确保覆盖而对空间离散化施加的必要限制。 |
| [^15] | [Social Humanoid: Socially Responsive Human-Humanoid Interaction](https://arxiv.org/abs/2611.08407) | 提出Social Humanoid分层系统，通过共享标记化表示的动作序列建模和两阶段自适应重定向训练方法，使29自由度人形机器人能够根据人类伙伴的动作生成具有社会响应性的全身运动，从而实现真正的人机物理协调互动。 |
| [^16] | [StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning](https://arxiv.org/abs/2611.08403) | StepBack通过轻量级信号定位推理草稿轨迹中首次出错的步骤，仅对该错误步骤及其后续内容进行重采样再投票，在七个数学推理与多跳问答基准上达到与自洽性相当的准确率，同时最多减少41.26%的token消耗。 |
| [^17] | [Synthesis of Ovarian Cancer Multi-Omics Datasets: A Biology-Informed Evaluation Framework](https://arxiv.org/abs/2611.08394) | 本文的核心贡献是提出了一个基于生物学信息且患者分离的评估框架，用于判断合成的配对RNA表达与拷贝数变异数据是否保留了生物学相关结构，并将其应用于高级别浆液性卵巢癌DECIDER队列。 |
| [^18] | [Hybrid Machine Learning for Imbalanced Landing-Phase Safety Event Prediction from Flight Data](https://arxiv.org/abs/2611.08386) | 该研究提出一种可解释的混合特征融合框架，将LSTM自编码器的重构误差描述符与统计飞行历史特征融合后输入类别加权XGBoost分类器，以在高度不平衡的多变量飞行数据中实现多类别着陆阶段安全事件预测。 |
| [^19] | [Module-Aware Optimization for Graph Neural Networks](https://arxiv.org/abs/2611.08379) | 提出了GraphMuon，一种图感知优化器，通过在图传播层输入的协方差结构所定义的度量下进行正交化并据此规范化步长，解决了现有矩阵优化器在训练图神经网络时忽略图传播表示几何结构、导致更新效果被放大的问题。 |
| [^20] | [Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout](https://arxiv.org/abs/2611.08376) | 提出线性注意力架构Muir：利用“状态更新对擦除的敏感性远低于写入”这一发现，将所有门控融合为敏感性分配的低秩投影（擦除门采用低秩参数化），并对状态行重加权以实现匹配滤波式读出，从而提升线性注意力的效率与性能。 |
| [^21] | [SHORE: Stability-Hybrid On-Policy Reinforcement Learning for Online Bin Packing](https://arxiv.org/abs/2611.08364) | 该论文提出SHORE方法，通过3D U-Net学习稳定性概率并结合在策略收益估计，将稳定性风险融入箱子选择、朝向和放置位置的联合决策中，有效解决了在线三维装箱中防止倒塌与最大化装载量的权衡问题。 |
| [^22] | [StrataFlux: Memory-Hierarchy-Native Low-Bit LLM Inference](https://arxiv.org/abs/2611.08363) | StrataFlux 框架通过从主机内存和固态硬盘进行选择性精度流式传输，并让数据交付与计算重叠、跨解码步骤复用精度信息，在保留量化内存和速度优势的同时，有效恢复了低比特权重和 KV 缓存量化导致的 LLM 推理精度损失。 |
| [^23] | [Unifying Central Flow, Rod Flow, and the Edge Coupling](https://arxiv.org/abs/2611.08358) | 本文提出“边缘耦合”这一泛函，证明中心流与杆流均可由它推导得出，二者分别对应不同时间尺度上的时间粗粒化，从而统一了刻画梯度下降稳定性边缘行为的两种连续时间模型。 |
| [^24] | [USB Gym: A Gymnasium Testbed for Closed-Loop Ultrasound View Matching](https://arxiv.org/abs/2611.08352) | 提出了USB Gym——一个基于公开CT数据模拟超声的闭环超声视图匹配标准化测试平台，通过可替换插槽机制系统评估并组合20种已发表方法，为超声引导研究建立了统一可比较的基准。 |
| [^25] | [4D Point Splatting for mmWave Radar Micro-Doppler Differentiable Rendering](https://arxiv.org/abs/2611.08342) | 提出了4D点溅射（4DPS），一种基于物理的可微渲染器，通过将携带材质、法线和速度信息的定向点溅射到距离-多普勒域，将显式运动场景拟合到真实毫米波雷达微多普勒采集数据上，输出原始复数回波以缩小雷达感知的sim2real差距并缓解雷达数据稀缺问题。 |
| [^26] | [Discovering Learning Rules that Shape Others](https://arxiv.org/abs/2611.08335) | 该论文提出DUOS方法，将元学习发现的更新规则框架从单智能体强化学习扩展到一般和博弈中的对手塑造，无需访问对手的参数、策略、奖励或学习规则即可实现对手塑造。 |
| [^27] | [Benchmarking Large Language Models on Heterogeneous Cognitive and Biomedical Knowledge Graphs: The Role of Graph Grounding in Multi-Hop Reasoning and Hallucination Mitigation](https://arxiv.org/abs/2611.08315) | 该研究在五个认知与生物医学知识图谱上对五个大型语言模型进行基准测试，发现通用模型在多跳推理任务上优于领域专用模型，且知识图谱接地能有效降低所有模型的幻觉率。 |
| [^28] | [Interpretable many-body interaction laws for collective motion from few trajectories](https://arxiv.org/abs/2611.08314) | 本文提出方向原子簇展开（dACE）方法，将集体运动中的社会相互作用严格分解为一体、二体、三体等多体贡献，仅需少量轨迹数据即可逐阶恢复相互作用的体阶规律，并在二维和三维模拟群体及真实鱼群数据上全面优于现有神经方法SocialNet。 |
| [^29] | [RM-VLM: From Natural-Language Task Descriptions to Vision-Grounded Reward Machines](https://arxiv.org/abs/2611.08308) | 本文提出 RM-VLM，利用大语言模型从自然语言任务描述自动生成奖励机，并让视觉-语言模型充当标注函数将命题接地于图像观测，从而免除传统奖励机中需要针对每个环境手动设计标注函数的局限。 |
| [^30] | [Measured Coupling for Parallel Decoding in Diffusion Language Models](https://arxiv.org/abs/2611.08307) | 该论文提出利用主前向传播中的廉价探针直接测量token间的耦合强度（即提交一个token对其他位置置信度的影响），以替代平均场解码中的相似性代理指标，使扩散语言模型并行解码在平均仅损失1.1个点精度的代价下大幅减少解码步数并提升速度。 |
| [^31] | [Dynamic Optimistic Constrained OCO with Memory via Delay Equivalence](https://arxiv.org/abs/2611.08302) | 该论文提出了一种结合不可信预测的乐观算法，通过将记忆归约为延迟来求解带记忆和时变约束的在线凸优化（COCO-M），首次建立了同时适应环境非平稳性与预测准确性的动态遗憾和累积约束违反界，并去除了函数可分离性等限制性假设。 |
| [^32] | [Transformer-Based Trajectory Prescription for sEMG-Optimizing Robot-Mediated Rehabilitation](https://arxiv.org/abs/2611.08295) | 该论文提出了一个可个性化的Transformer模型，利用旋转位置编码和流匹配，将康复机器人训练中用户的表面肌电肌肉激活映射为接触力轨迹处方，从而实现以神经运动激活优化为目标的机器人介导康复。 |
| [^33] | [AutoCompress: Automated Hardware-Guided Model Compression](https://arxiv.org/abs/2611.08290) | AutoCompress是一个硬件引导的模型压缩框架，它将延迟、吞吐量和内存等硬件与工作负载约束直接纳入可微架构搜索，联合优化全局与逐层结构以生成异构压缩架构，并利用经过实测验证的服务引擎成本模型引导整个优化过程。 |
| [^34] | [Accelerating Safe Reinforcement Learning with Massive Parallelism](https://arxiv.org/abs/2611.08273) | 该论文提出相位聚合成本估计器与约束收紧框架，解决了大规模并行短回合训练与全回合安全约束之间的不匹配问题，实现了可扩展的安全强化学习加速训练。 |
| [^35] | [DART: Domain-Agnostic Residual Transfer for Generalist Anomaly Detection](https://arxiv.org/abs/2611.08271) | DART提出了一种即插即用框架，通过残差解耦模块与跨领域元训练，将正常-异常参考残差分离为异常相关成分与领域特定成分，从而提升通用异常检测的跨域迁移能力。 |
| [^36] | [Scaling Neural Motor Decoding via Decoupled Behavioral Pretraining](https://arxiv.org/abs/2611.08263) | 提出BEEMO框架，通过解耦行为与神经表征学习，利用丰富且易获取的非配对行为数据进行训练，独立扩展行为和神经监督，从而在非人类灵长类皮层内运动数据集上显著提升解码性能。 |
| [^37] | [Nowcasting of Tropical Cyclone Wind Fields and Intensity from Geostationary Imagery](https://arxiv.org/abs/2611.08255) | 本文提出基于地球静止卫星图像的简单神经网络，通过在模型潜空间联合预测二维风场与风暴尺度强度指标，显著改善了最大风速及其半径的估计效果，尤其适用于快速增强事件。 |
| [^38] | [Series-wise Cross-validation for Global Time Series Forecasting](https://arxiv.org/abs/2611.08252) | 该论文为基于深度学习的全局时间序列预测模型形式化了按序列划分的交叉验证方法，突破传统仅按时间划分的范式，并在11个数据集和10种神经网络架构上系统评估了K折、自助法等多种序列级交叉验证策略。 |
| [^39] | [Geometric Relevance Is Not Sufficient: Evaluating Focused Post-Training Supervision in Diffusion Docking](https://arxiv.org/abs/2611.08249) | 该研究通过固定预算实验发现，在DiffDock扩散分子对接的后训练中，聚焦于几何相关校正的监督策略相比广泛的延续训练并无一致优势，广泛的延续训练反而取得了最高的Top-1对接成功率。 |
| [^40] | [FewStepRL: Fine-tuning Few-step Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2611.08240) | 该论文提出FewStepRL方法，通过变换、分解与适配三个阶段系统地将强化学习目标适配到少步扩散模型，解决了现有RL目标面向瞬时速度而少步模型建模区间平均速度的不匹配问题，同时显著降低了训练成本。 |
| [^41] | [Beyond Quadratic Attention: Separable Gaussian Operators for ODE Vision Transformers](https://arxiv.org/abs/2611.08236) | SSoG用可分离的高斯和算子替代自注意力，将计算复杂度从O(N²)降至亚二次，并凭借全局Lipschitz连续性保证连续深度ODE解的唯一性，在CIFAR-100上以更少的资源将Top-1准确率提升2.2–4.1个百分点。 |
| [^42] | [Do sEMG Foundation Models Improve Cross-Subject Generalisation?](https://arxiv.org/abs/2611.08199) | 本研究通过系统评估发现，sEMG基础模型的预训练优势仅在会话内评估中显著，在跨受试者泛化场景下基本消失，且在小规模受试者数据集上经典机器学习基线可匹敌甚至超越基础模型。 |
| [^43] | [BUMP: Brief Unexpected Mid-Interpolation Peak](https://arxiv.org/abs/2611.08182) | 本文发现了神经网络插值中的 BUMP 现象（测试精度在插值端点附近短暂提升后向中点下降），并基于此提出了将权重混合与预测空间混合相结合的新型集成方法。 |
| [^44] | [Regret Bounds for Profit Maximization in Bilateral Trade: Learning the Best Posted-Price Mechanism](https://arxiv.org/abs/2611.08173) | 本文从遗憾最小化角度研究双边贸易中利润最大化经纪人问题，在只能观察交易是否发生的反馈下，为学习最优固定定价机制给出了多项式时间的次线性遗憾算法（含对抗情形）。 |
| [^45] | [Patch Attention Wisely: Understanding Patch Length in Attention-Based Forecasting](https://arxiv.org/abs/2611.08160) | 该论文提出了一个理论框架，解释了基于注意力的时间序列预测中补丁长度为何影响预测精度，并基于线性状态空间模型的精确分解，推导出在时间相关性下平衡表示误差与估计误差的补丁选择准则。 |
| [^46] | [Improving Context-Shift Robustness of Convolutional Networks via Context-Regularized Cross-Entropy](https://arxiv.org/abs/2611.08157) | 本文提出新的特征归因方法 ContrastiveCAMs，并在此基础上构建上下文正则化交叉熵训练框架，通过引导模型减少对上下文区域的依赖、更多关注物体本身的特征，从而显著提升卷积神经网络在上下文偏移场景下的鲁棒性。 |
| [^47] | [La-JEPA: Compact Robot Policies with Latent World Modeling](https://arxiv.org/abs/2611.08156) | La-JEPA是一种紧凑的世界-动作模型，它在预训练V-JEPA 2.1编码器的特征空间中从头训练策略Transformer，无需具身预训练即可以更少的参数和更快的推理速度在RoboTwin 2.0和LIBERO-Plus基准上取得最先进的机器人控制性能。 |
| [^48] | [AI-Native Task-Oriented Semantic Communications over a Relay: Benefits and Vulnerabilities](https://arxiv.org/abs/2611.08155) | 本文提出了基于中继的任务导向语义通信的联合端到端设计，证明其在低信噪比下优于直接传输，同时首次揭示并形式化了中继注入学习扰动的拜占庭攻击这一关键安全漏洞。 |
| [^49] | [Interpretable Multimodal MRI-Clinical Fusion for Functional Outcome Prediction After Acute Ischemic Stroke](https://arxiv.org/abs/2611.08154) | 该研究提出一种可解释的多模态MRI与临床数据融合方法，证明FLAIR-临床融合（AUC=0.948）能显著提升急性缺血性卒中后三个月功能结局的预测性能。 |
| [^50] | [Detecting Weather Fronts from Satellite Observations: What Does Reanalysis Add?](https://arxiv.org/abs/2611.08148) | 该研究比较了仅用静止红外卫星观测与额外加入ERA5再分析数据的机器学习天气锋面检测模型，发现再分析数据可将检测性能显著提升（CSI@150从0.402提高至0.575），其中低层风场是仅靠卫星数据最难预测的信息。 |
| [^51] | [Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning](https://arxiv.org/abs/2611.08144) | 该论文发现强化学习中环境与训练器边界处的静默故障几乎无法通过学习曲线等基于结果的检查发现，而直接检查运行过程中跨越边界传输的数据本身则能检测到每一个注入的故障。 |
| [^52] | [How Far Can Untrained Neural Networks Go in Anomaly Detection? From Tables to Pixels](https://arxiv.org/abs/2611.08136) | 提出PRISM框架，仅利用固定的随机非线性特征结合基于正常参照的马氏距离评分，无需任何训练即可在表格异常检测和视觉异常定位任务上达到最先进水平。 |
| [^53] | [Identifying Neural Dynamics Using Nonlinear Interventional State Space Models](https://arxiv.org/abs/2611.08113) | 该研究将干预状态空间模型的可识别性理论从线性动力学推广到一般非线性潜在动力学与单射发射函数，为识别神经回路动力学提供了更广泛的理论保证。 |
| [^54] | [PersonaAgent: Retrieval-Augmented Transaction Anomaly Scoring with Refreshable Multi-Entity Persona Memory](https://arxiv.org/abs/2611.08094) | PersonaAgent通过将实体最新行为信号以角色形式存储在可刷新的多实体记忆中，并在推理时由轻量级适配器检索以增强基础异常分类器的评分，有效解决了交易模式频繁变化与实体嵌入高基数稀疏的挑战。 |
| [^55] | [Gradient Boosting for Efficient Rank Discovery in Tensor Network Kernel Machines](https://arxiv.org/abs/2611.08093) | 提出梯度提升框架GB-CPKM，通过顺序添加固定秩CP张量网络学习器来增量式提升模型容量，从而避免显式秩搜索，实现最高6倍的训练加速和2倍的峰值内存节省。 |
| [^56] | [Gabor Gabor: E-Gaborformer, a Strictly Causal Speech Recognizer with Gabor-Dictionary Convolutions and Delta-Rule Memory](https://arxiv.org/abs/2611.08092) | E-Gaborformer是一种严格因果的语音识别编码器，通过Gabor字典卷积与Delta规则循环记忆实现恒定约19MB的流式状态（不随通话时长增长），在LibriSpeech上收敛速度最高提升18%，生产规模下词错误率降低28-31%。 |
| [^57] | [SAE Interprets MoE: Sparse AutoEncoders Reveal That MoE Experts Exhibit Human-Interpretable Fine-Grained Specialization](https://arxiv.org/abs/2611.08089) | 该论文通过在 MoE 层输出上训练稀疏自编码器，利用其仿射结构将每个潜变量的激活精确分解到各专家，从而揭示出 MoE 专家具有比门控选择统计所能体现的更集中、且人类可解释的细粒度专业化分工。 |
| [^58] | [Value or Attention: Investigating Layer-wise Relevance Propagation for Graph Attention Networks](https://arxiv.org/abs/2611.08088) | 该论文提出了一种带自由权重的LRP乘积规则，将图注意力网络中节点接收到的相关性分解为值路径的内容相关性和注意力路径的路由相关性两部分，并将现有解释方法统一为其特例。 |
| [^59] | [DictLLM: Post-training Compression of Large Language Model with Dictionary Kernels](https://arxiv.org/abs/2611.08082) | DictLLM提出了一种基于字典核的训练后压缩方法，将分组层的权重矩阵分解为共享字典与层特定稀疏系数，无需进一步训练即可在移动设备上高效压缩大语言模型。 |
| [^60] | [Optimal Competitive Ratios for Chasing Convex Functions with Quadratic Growth](https://arxiv.org/abs/2611.08078) | 本文针对具有二次增长的凸函数追逐问题，在欧几里得移动和平方欧几里得移动两种成本类型下建立了最优的无维度依赖竞争比，弥补了已有上下界之间的多项式差距。 |
| [^61] | [Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? Exploring "Weight-Space Programming"](https://arxiv.org/abs/2611.08070) | 论文在无工具、无前向传播的条件下测试LLM读取小型卷积网络原始权重的能力，发现其表现主要源于准确的局部算术计算与对公开数据集的记忆先验，而非对权重流动的真正全局推理。 |
| [^62] | [Performance metrics in machine learning: a scenario-based guide from the confusion matrix to calibration, regression and uncertainty](https://arxiv.org/abs/2611.08048) | 本教程通过15个受控场景系统揭示了类别流行率、决策阈值等因素如何使相同的预测在不同性能指标下得出截然不同的结论，为机器学习评估指标的正确选择与解读提供了实用指南。 |
| [^63] | [KeHoSu: Keyed Homoglyph Substitution for Authenticated Text Provenance over Copy-Paste Channels](https://arxiv.org/abs/2611.08046) | 提出KeHoSu方法，将同形字攻击中的易混淆字符转化为带密钥的归属信道，通过Unicode字符变体在文本中嵌入平台、用户与认证标签信息，实现复制粘贴渠道下的文本溯源，并对句子重排和部分引用保持鲁棒。 |
| [^64] | [Should We Pay This Much for Robustness? Efficient Proxy Certificates with Marginal Guarantees](https://arxiv.org/abs/2611.08035) | 提出一种将鲁棒性认证委托给廉价代理函数的框架，借助鲁棒共形风险控制保证有界的边际错误批准率，可独立使用或作为后备机制，大幅降低认证成本。 |
| [^65] | [Reproducing Kernel (Quasi-)Banach Spaces with the $\ell^p$ Coefficient Norms](https://arxiv.org/abs/2611.08032) | 本文将再生核巴拿赫空间理论推广到一般的 $\ell^p$ 正则化框架，在非凸情形 $0<p<1$ 下构造了带 $\ell^p$ 拟范数的再生核拟巴拿赫空间并建立了插值与正则化学习的存在性和稀疏性保证，同时在 $p>1$ 时基于可数字典构造 RKBS 并证明了极小值点的存在性。 |
| [^66] | [GTAFL: Addressing Test-Agnostic Long-Tailed Federated Learning in Medical Image](https://arxiv.org/abs/2611.08021) | 该论文提出了“测试不可知长尾联邦学习”这一新任务，并设计GTAFL综合框架，通过自适应重采样和专家分类器重训练等策略，解决医疗图像联邦学习中数据长尾分布且测试分布不可预测导致模型失效的问题。 |
| [^67] | [Learning Dynamics of On-Policy Distillation](https://arxiv.org/abs/2611.08016) | 本文对在策略蒸馏的学习动力学进行了机制分析，提出将反馈循环分解为“蒸馏信号转换”与“广播”两个阶段的框架，用以解释学生响应有限以及训练过程中的退化现象。 |
| [^68] | [Towards Multiscale Molecular Representation Learning through Geometry-Based Fragmentation for Polymer Property Prediction](https://arxiv.org/abs/2611.07998) | 该论文提出了一种受k-center启发的、基于几何的半径约束分子片段化方法，实现多尺度分子表示学习，从而提升聚合物性质预测的性能。 |
| [^69] | [QRAE: Query-Relevant Attention Entropy for Context Compression in DeepSearch Agents](https://arxiv.org/abs/2611.07976) | 提出QRAE方法，通过查询相关注意力熵对DeepSearch智能体检索到的文本块进行在线评分与筛选，将高价值证据保留在推理上下文中并缓存不确定内容，从而在多步搜索推理中有效压缩上下文、降低token消耗与延迟。 |
| [^70] | [Tight Asymptotic Bounds on the Maximum Number of Linear Regions of Piecewise-Linear Kolmogorov--Arnold networks](https://arxiv.org/abs/2611.07973) | 本文首次为分段线性Kolmogorov–Arnold网络的线性区域最大数量给出了相互匹配的上下界，确定了紧致增长阶Θ_n(k^(Σρ))，从而大幅改进了此前文献中宽松的上界并首次补齐了下界。 |
| [^71] | [CoRD: Coupled Rollouts Fill the Blind Spot of On-Policy Distillation](https://arxiv.org/abs/2611.07959) | CoRD 通过投机解码的最大耦合从学生与教师分布的几何插值中抽取训练轨迹，在保留同策略损失的同时把训练状态分布推向教师，从而监督学生自身轨迹覆盖不到的“盲区”状态，并在五个数学基准上较同策略蒸馏平均提升 pass@1 0.9–2.0 分、pass@8 1.2–3.4 分。 |
| [^72] | [A Behavioural Framework for Predicting and Understanding Loss of Control in Frontier Artificial Intelligence Systems](https://arxiv.org/abs/2611.07953) | 本文提出一个行为框架，将前沿AI系统的失控风险分解为动机错位、危害促成能力和监控规避三个因素，并验证了其失控评分在13个前沿模型上对实际失败率的前瞻预测能力（Pearson r = 0.84），且表明风险无法仅由模型能力解释。 |
| [^73] | [Invariance before structure: two modes of emergent compression in representation learning](https://arxiv.org/abs/2611.07952) | 该论文提出EARL框架，将表征学习中的涌现压缩区分为状态空间压缩与参数空间压缩两种模式，并以Fisher敏感性和系综级信息论度量加以形式化，从而解释生物与人工神经网络之间表征不变性的来源。 |
| [^74] | [Leakage- and Label-Aware Benchmarking of Class-Imbalance Strategies for Multimodal Chest X-Ray and EHR Classification](https://arxiv.org/abs/2611.07950) | 该研究对多模态胸片与EHR类别不平衡基准进行审计，揭示了记录级数据划分导致的标签泄漏问题——患者标签由单次住院的诊断代码生成并复制到其所有影像，造成基准性能虚高，因此需要采用患者级数据划分与标签感知的评估方式。 |
| [^75] | [Joint Modeling of Conditional Mean and Variability in Multilevel Functional Data: Application to Postprandial Glucose Trajectories in the AEGIS Study](https://arxiv.org/abs/2611.07937) | 该论文提出了一种异方差函数混合效应回归框架，联合建模多水平重复函数轨迹的条件均值与方差，并通过参与者层面的自助法量化参数不确定性，应用于AEGIS研究揭示了各因素对餐后血糖均值与变异性的时依赖差异化影响。 |
| [^76] | [The Blindfolded Controller: Decision Model-Guided Training for Reinforcement Learning](https://arxiv.org/abs/2611.07934) | 本文提出决策模型引导训练（DMGT）方法，利用决策模型初始化强化学习策略，使智能体从最初的回合即可到达目标并获得奖励，再通过强化学习将策略优化到超越决策模型的水平，从而有效解决了稀疏奖励任务中的探索难题。 |
| [^77] | [AttnSR: Attention Step Reward for Long-Horizon Agentic RL](https://arxiv.org/abs/2611.07932) | 提出 AttnSR 步级奖励增强方法，利用训练回放中的策略注意力反向追踪为各步骤分配奖励，无需额外奖励模型，即在 ALFWorld 上将 Qwen2.5-7B 的成功率从 91.2% 提升至 97.1%。 |
| [^78] | [Flux as a Codec: Zero-Shot Lossy Compression with Flux](https://arxiv.org/abs/2611.07929) | 本文提出 FluxC，一种以 Flux 为压缩先验的零样本编解码器，通过系统分析编码路径与重建方法对率-失真-感知性能的影响，发现基于 ODE 的重建在低码率下失真更低，整体性能可与现有方法竞争甚至超越。 |
| [^79] | [Same Answer, Different Future: Matched Predictions Do Not Imply Matched Belief Updating in Language Models](https://arxiv.org/abs/2611.07913) | 该论文发现，语言模型通过不同途径（观察结果、摘要、指令或激活干预）即使得出相同的概率预测，其内部信念状态也并不等价，因而在面对后续相同新证据时会以不同的方式更新信念，即“当前预测一致”不代表“信念更新机制一致”。 |
| [^80] | [EDGAR: Efficient Diffusion-Driven Gaussian Avatar Refinement via Synthetic Failure](https://arxiv.org/abs/2611.07909) | 提出EDGAR条件扩散框架，利用真实与合成样本组成的混合三元组数据集来精修高斯虚拟形象的降质渲染，在保持身份一致性的同时有效校正模糊、拉伸等伪影。 |
| [^81] | [Structure from Collision via Hypothesis-Space Reasoning](https://arxiv.org/abs/2611.07900) | 提出通过假设空间推理从碰撞事件的多视角观测中恢复物体隐藏内部结构（如空腔），克服了空腔仅能通过表面运动间接观测、且因对应材料缺失而难以用密度表示直接优化的难题。 |
| [^82] | [DISCERN: Distributional Synthetic Replay for Retain-Free Incremental Machine Unlearning](https://arxiv.org/abs/2611.07892) | DISCERN 通过为每个类别拟合欧氏与方向性参数分布并采样合成嵌入进行回放，实现了无需访问保留集的增量式类遗忘，在依次删除类别的同时确保已遗忘知识不被恢复并保留剩余类别的性能。 |
| [^83] | [When, Where, How: Active In-Context Learning for Tabular Foundation Models](https://arxiv.org/abs/2611.07886) | 本文针对表格基础模型提出主动上下文学习方法，研究了目标感知最近邻（TANN）采集和上下文期望预测信息增益（EPIC）两种策略，并在二十个基准上发现基于多样性的采集在初期迭代中最为有效。 |
| [^84] | [Which Signals of Code Correctness Survive Distribution Shift? Hidden-State and Causal Predictors of Program and Verdict Correctness](https://arxiv.org/abs/2611.07876) | 本文系统比较了四类共六种免执行代码正确性预测器在基准与语言分布偏移下的稳健性，并将两种新颖的因果定位方法引入代码领域，同时评估其对程序正确性和模型自身判定可信度的预测能力。 |
| [^85] | [PlayGen2: Learning Real-Time Playable Worlds with Explicit 3D State](https://arxiv.org/abs/2611.07867) | PlayGen2通过将显式3D世界状态的演化与神经渲染相分离，并维护由体积占用栅格和身份索引实体3D状态构成的持久世界记忆，实现了具备长期空间一致性的实时可交互可玩世界。 |
| [^86] | [Computation, not weight access: large language models as auditors of clinical prediction networks](https://arxiv.org/abs/2611.07847) | 通过预先设定的模拟实验，该研究发现在没有计算工具时仅凭提示中的原始权重无法可靠恢复植入的关系，表明大语言模型审计临床预测网络的关键在于计算能力（借助工具访问权重或预测）而非直接的权重访问。 |
| [^87] | [Masked diffusion as random sequential adsorption: an exactly solvable model for parallel decoding](https://arxiv.org/abs/2611.07845) | 该论文将掩码扩散模型并行解码的误差与驱动多粒子系统的熵产生和耗散联系起来，并把马尔可夫数据下的生成精确等价为一个可严格求解的随机序贯吸附过程，从而建立了并行解码的首个精确可解理论模型。 |
| [^88] | [Enhanced MRI brain tumor detection using deep learning in conjunction with explainable AI SHAP based diverse and multi feature analysis](https://arxiv.org/abs/2611.07843) | 本研究将LBP、Gabor滤波器、小波变换、CNN等多种特征表示方法与多种机器学习算法相结合用于脑肿瘤检测，发现LBP结合SVC和CNN取得了高特异性和准确率，并借助可解释AI（SHAP）对多特征分析进行了解释。 |
| [^89] | [Estimating Spectrum-Free Information Bounds for Generative Surrogates Beyond Conditional Collapse](https://arxiv.org/abs/2611.07826) | 针对高能物理条件生成式代理模型评估中一维边缘散度指标会奖励条件坍缩的问题，本文提出采用预报检验领域的CRPSS技巧评分并以精确边缘分布为参考，同时引入数据驱动的k近邻外推方法来估计不受评估谱影响的无谱信息界，从而实现更可靠的模型评估。 |
| [^90] | [Decodability Is Not Validity: Evaluating Nuisance Sensitivity in Language Model Concept Probes](https://arxiv.org/abs/2611.07805) | 论文提出用“判别性证据”来评估概念探针的干扰因素敏感性，并证明探针的可解码性（预测准确率）并非有效性的可靠代理——AUROC 统计等价的探针对相关干扰因素的敏感性可相差数倍，且两者秩相关性在不同设置间会反转符号。 |
| [^91] | [EnergyBench: a live, gate-aligned benchmark of day-ahead forecasts for European power markets](https://arxiv.org/abs/2611.07784) | EnergyBench 是一个对欧洲电力市场日前预测进行实时评估的基准，它在每日交易闸门关闭前自行运行各模型，并使用冻结且公开的统一信息集（含数值天气预报），从而避免了回测数据泄露和固定测试集过拟合的问题。 |
| [^92] | [PBPO: Prefix-Block Policy Optimization for Diffusion Language Models](https://arxiv.org/abs/2611.07783) | 提出PBPO方法，通过构建前缀块状态并利用单次模型评估定义分解式单步块似然代理，将策略优化自然扩展到扩散语言模型的块级别，使个别位置的变化能在裁剪前传播至整个块。 |
| [^93] | [Adapters as Data: The Latent Space of LoRAs](https://arxiv.org/abs/2611.07775) | 该论文将LoRA适配器视为一种数据模态，训练Transformer自编码器构建非线性潜空间，证明该潜空间不仅保留了权重空间中采样与编辑等能力，还对位移具有更强鲁棒性，退化时产生扭曲但仍然有效的面孔而非纹理。 |
| [^94] | [Optimal Learning in Games with Arbitrary Delays: Best-of-Both-Worlds for Self-Play and Adversarial Opponents](https://arxiv.org/abs/2611.07773) | 提出单一算法MAO-OMD，在具有任意未知反馈延迟的零和博弈中，同时实现了自博弈时的最优快速遗憾速率 O(σ_max+1) 和对抗任意对手时极小极大最优的 O(√(T+d_tot)) 保证。 |
| [^95] | [Scaling Laws for SAE Training Data](https://arxiv.org/abs/2611.07756) | 在固定训练预算下，SAE 所需的独特激活数据量存在饱和点——缓冲区达到一定规模后，重放数据与新鲜激活值效果相当，且缓冲区无需随训练预算扩大；作者据此提出并在多种模型、层级和 SAE 架构上验证了一条刻画多样性与重复权衡的有限数据缩放定律。 |
| [^96] | [Neural Collapse on Imbalanced Datasets and Open-Set Scenario](https://arxiv.org/abs/2611.07753) | 本文基于类别平均损失（而非全局平均损失）的优化理论，证明了无论数据集是否平衡，甚至在部分类别完全没有训练样本的开集场景下，当各类别平均损失达到最小值时，标准的神经坍缩结构依然会涌现。 |
| [^97] | [Foresee to Trade: Forecast-Augmented Policy Optimization for Portfolio Management](https://arxiv.org/abs/2611.07750) | 提出 FRPO 框架，通过金融状态头（FSH）对资产级机会进行未来收益监督，并结合同路径组相对优化对账户级交易轨迹进行相对结果监督，以强化投资组合管理中的账户级交易决策。 |
| [^98] | [ConvPE: Convolutional Position Embedding Improves Transformers](https://arxiv.org/abs/2611.07749) | ConvPE通过对隐藏状态低维投影进行轻量级因果卷积，向注意力分数注入窗口化、内容相关的偏置，作为即插即用模块在语言建模、长上下文扩展和视觉任务中同时提升了Transformer的性能与效率。 |
| [^99] | [Ordering Multivariate Time Series by Volatility Memory: A Temporal Convolutional Approach to Volatility Heredity](https://arxiv.org/abs/2611.07746) | 本文提出QS-TCN方法，将准极大似然估计的波动率持续性统计量与矩阵束方法提取的时序模式嵌入时序卷积网络的结构化滤波器中，从而基于波动率遗传关系对多变量时间序列进行全局排序。 |
| [^100] | [Comparing Classification and Feature Learning by Analyzing Their Loss Functions in Normalized Spaces with Different Scale Factors](https://arxiv.org/abs/2611.07741) | 本文指出分类准确率反映的是样本级局部性质，而特征学习追求的是数据集级全局性质，并从第一性原理重新设计BCE损失，在归一化特征空间中分析其与CE损失的极小值，证明两者均可导致神经坍缩。 |
| [^101] | [Extending Reasoning in Typed Decision Models via Looping](https://arxiv.org/abs/2611.07736) | 该论文提出 RELOAD 方法，通过对预训练模型的中间块进行阻尼循环并由轻量级控制器自适应选择循环次数，首次证明循环机制在与决策模型联合训练时能显著提升类型化决策模型的准确性与校准度，尤其在需要长推理链的问题上编码器最多可提升 23 分。 |
| [^102] | [Static Recovery Is Not Dynamic Stability: Dynamics-Aware Benchmarking of Protein Motif Scaffolding](https://arxiv.org/abs/2611.07723) | 该论文提出了动力学感知的蛋白质基序支架化新基准BILBY-PRO和BILBY-MD，通过系统性扩大基序覆盖范围并引入动态稳定性评估标准，揭示了不同生成模型在目标覆盖与独特解产出之间的显著差异。 |
| [^103] | [Reinforcing What the Student Has Learned: Teacher-Guided On-Policy Distillation](https://arxiv.org/abs/2611.07722) | 提出SD-OPD方法，以学生自身预训练基础模型的差异作为更新权重，避免在策略蒸馏被师生表达风格差异主导，从而强化学生已习得的推理能力。 |
| [^104] | [Evaluating Language Model Features for Zero-Shot Time-Series Anomaly Detection Under Isolation Assumption](https://arxiv.org/abs/2611.07720) | 该论文提出将大语言模型纯粹作为特征提取器用于零样本多变量时间序列异常检测，通过直接分析其诱导的特征空间，规避了预测或重建等辅助目标在模型容量增大时会重建异常而变得不可靠的问题。 |
| [^105] | [Multimodal Information Interactions Bottleneck for Multimodal Self-Supervised Learning](https://arxiv.org/abs/2611.07718) | 提出基于部分信息分解的多模态信息交互瓶颈框架MIIB，通过双路径架构显式编码多模态交互中的冗余、独特与协同信息，无需显式互信息估计即可学习最小充分的多模态表示。 |
| [^106] | [InfMasking: A Unified Variational Framework for Synergistic Multimodal Learning](https://arxiv.org/abs/2611.07702) | 提出了 InfMasking，一个基于无限随机掩码的统一变分多模态学习框架，通过在融合前随机掩码各模态特征，并要求所有被掩码的融合表示与参考目标（无监督时为完整融合视图，有监督时为标签类别代理）保持一致，从而迫使模型充分利用模态间的协同信息，而非依赖主导模态或共享信息。 |
| [^107] | [Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction](https://arxiv.org/abs/2611.07695) | 提出DP-Learner框架，在直积空间中直接对类哈密顿量算符块进行等变消息传递，同时解决了传统直和表示与哈密顿量之间的表示不匹配问题及计算效率瓶颈。 |
| [^108] | [Non-Parametric Mutual Information Estimation with Vine Copulas](https://arxiv.org/abs/2611.07693) | 本文提出VINE，一种基于简化D-vine copula的非参数互信息估计器，通过将密度比简化为成对copula的乘积实现非参数估计，在低维到中等维度数据集上以更低的计算成本达到与最先进神经估计器相当的精度。 |
| [^109] | [Randomized and Oracle-Based Feasible Directions Methods for Nonsmooth Nonconvex Constrained Optimization](https://arxiv.org/abs/2611.07691) | 该论文引入η-可行方向（与可行方向锥边界保持一致分离的方向），克服了非光滑非凸约束优化中可行方向锥不封闭导致的几何难题，实现了无需目标导数、仅靠精确线搜索的有限方向搜索算法框架。 |
| [^110] | [$O(1/n)$ Convergence Rate for RVI $Q$-Learning](https://arxiv.org/abs/2611.07676) | 该论文通过分析随机 Krasnosel'skiĭ–Mann 迭代，为有限弱通信 MDP 上的同步 RVI Q-学习建立了有限时间收敛保证，并在适当多项式步长下达到 $O(1/n)$ 的收敛速率。 |
| [^111] | [Instrumental Variables in Continuous-Time Stochastic Processes](https://arxiv.org/abs/2611.07674) | 该论文首次将工具变量方法扩展到由随机微分方程参数化的连续时间随机过程中，通过在漂移项的无穷小层面编码结构稀疏性与因果效应，并利用路径值工具变量的矩约束条件，实现了连续时间动力学系统中因果效应的识别。 |
| [^112] | [Calibrated Target Noise Recovers Curvature from the Gradients](https://arxiv.org/abs/2611.07671) | 该论文提出一种向目标注入方差经局部输出曲率校准的零均值噪声的简单方法，使得含噪批量梯度的协方差能够仅从梯度中恢复出广义高斯-牛顿（GGN）等曲率矩阵，无需逐样本梯度或二阶导数。 |
| [^113] | [An Information Bottleneck for Post-hoc Memorization Mitigation in LLMs](https://arxiv.org/abs/2611.07669) | 提出一种即插即用的非参数变分信息瓶颈（NVIB），插入冻结的大语言模型骨干与语言建模头之间，通过压缩隐藏表示在不修改或不重训模型的情况下过滤记忆化内容、保留通用语言知识。 |
| [^114] | [Extracting Single-Mode Rayleigh--Taylor Instability dynamics from data via SINDy](https://arxiv.org/abs/2611.07666) | 该论文提出了一个基于SINDy积分形式的物理约束数据驱动框架，通过引入无量纲初始振幅作为显式参数，直接从数据中识别出能够刻画瑞利-泰勒不稳定性中气泡和尖钉演化（包括过冲与再加速阶段）的低维动力学方程，克服了传统势流理论和浮力-阻力模型的局限。 |
| [^115] | [Simple Meta-Learning for HPO and CASH Using Performance Representations](https://arxiv.org/abs/2611.07659) | 提出一种简单有效的元学习表示方法，直接以配置在历史任务上的性能表现作为其表示，从而改进贝叶斯优化在HPO和CASH中的代理建模效果。 |
| [^116] | [English Calibration Erodes Lithuanian in Expert-Pruned Mixture-of-Experts Language Models](https://arxiv.org/abs/2611.07654) | 该研究发现英语校准在专家剪枝的MoE模型中无法迁移到立陶宛语——25%剪枝率下英语理解仅下降1.7分而立陶宛语下降21.4分，而立陶宛语校准却能同时保护两种语言的性能。 |
| [^117] | [WaSP: Wasserstein Data Synthesis Pipeline for Solver-Free Operator Learning](https://arxiv.org/abs/2611.07646) | 该论文提出WaSP，一种基于Wasserstein度量的数据合成流水线，通过在人造数据对上训练并逐步缓解训练分布与目标分布之间的偏移，实现了无需求解器的高效算子学习。 |
| [^118] | [Measuring the Marginal Value of Verifier-Based RL Beyond Supervised Fine-Tuning](https://arxiv.org/abs/2611.07639) | 该论文构建了一个确定性的GPU/LLM训练故障诊断基准，发现仅用1,200个黄金示范的监督微调即可近乎解决该基准，而从基础模型出发的GRPO强化学习几乎无增益，从而质疑了基于验证器的强化学习相对于监督微调的边际价值。 |
| [^119] | [Online Learning with Ordered Experts (via Two-point Feedback)](https://arxiv.org/abs/2611.07629) | 该论文研究了完全有序动作空间中基于两点反馈的在线学习问题，在分离单调损失下实现了仅比全信息界差一个对数因子的 O(log(|F|T)√(T log|F|)) 遗憾，在有界变差损失下实现了 Õ(√T+|F|) 的遗憾。 |
| [^120] | [Monotone Viability Learning for Offline Safe Reinforcement Learning](https://arxiv.org/abs/2611.07616) | 提出单调可行性学习（MoViL），通过单调有限时程集合递归从离线数据中学习可行集，在保持安全语义、收敛至无限时程可行性核的同时避免折扣引起的数值衰减，实现部署时可靠的动作安全认证。 |
| [^121] | [Embedding-Space Fingerprinting of LoRA Fine-Tunes: Recovering the Training-Data Domain from the Model Artifact Alone](https://arxiv.org/abs/2611.07610) | 本文提出嵌入空间指纹方法，证明无需接触训练数据，仅凭LoRA微调模型的产物（权重增量、内部激活或生成文本）即可识别其训练数据的领域，其中灰盒激活特征在留一数据集协议下达到61%的准确率，远高于20%的随机水平。 |
| [^122] | [Binding Paraphrased Instructions to Encountered Objects in sLLM Agents under Partial Observability](https://arxiv.org/abs/2611.07599) | 提出了无需训练的BINDIntent框架，利用语义接近度将改述指令中的指称表达与部分可观测环境中观察到的对象标签进行概率性绑定，显著提升了sLLM智能体面对指令同义改述时的鲁棒性。 |
| [^123] | [TACT-KV: Tri-Axis Cosine Transform for Compressing Volumetric KV Caches in Medical VLMs](https://arxiv.org/abs/2611.07593) | 提出TACT-KV，一种无需校准的KV压缩方法，通过对KV缓存施加可分离的3D离散余弦变换并结合逆注水比特分配，利用3D医学扫描的三轴空间相关性来压缩医学视觉语言模型中的体积KV缓存。 |
| [^124] | [Learning and Steering in Incomplete-Information Games](https://arxiv.org/abs/2611.07589) | 该论文提出了仅利用轨迹反馈就能在不知道智能体效用的情况下学习标准式和扩展式博弈的二分类型算法，使调解者能够引导自适应智能体走向效用最大化的结果，并达到了近乎最优的样本复杂度。 |
| [^125] | [Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment](https://arxiv.org/abs/2611.07586) | 提出自动化红队测试框架 TraceMRT，通过将副任务重构为主任务的连贯子目标并搜索轨迹塑形策略，系统性地压力测试了思维链监控对会隐蔽掩盖失对齐行为的智能体的鲁棒性。 |
| [^126] | [ACSeL: Adaptive Configuration Search for Hardware-Constrained LLM Inference](https://arxiv.org/abs/2611.07570) | ACSeL通过基于相对模型质量退化（RMQD）的代理指标，在满足内存和延迟约束的前提下，自适应地搜索大语言模型推理的最优配置组合（模型大小、批处理大小、量化方案），从而在硬件受限部署中实现最高生成质量。 |
| [^127] | [MambaTG: Candidate-Aware State Space Modeling for Future Link Prediction on Temporal Graphs](https://arxiv.org/abs/2611.07569) | 该论文提出MambaTG模型，通过候选感知的状态空间建模，融合历史序列演化与候选—历史兼容性两种互补信号，显著提升时序图上未来链路预测（尤其是未见边预测）的性能。 |
| [^128] | [Provable Zeroth-Order Safe Exploration for Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2611.07561) | 提出了一种无需奖励模型的零阶对数障碍策略梯度方法ZO-LBPG，通过保守约束余量估计、耦合轨迹采样和自适应步长，首次以高概率保证RLHF训练全过程中所有策略（包括探索策略）严格满足安全约束。 |
| [^129] | [The Memory Trap: Language Agents Overgeneralize from Benign Interactions](https://arxiv.org/abs/2611.07554) | 本文发现并命名了“记忆陷阱”这一新型失效模式：语言智能体会从良性交互记忆中过度泛化，例如低风险任务中跳过确认的经历会使其在高风险任务中也跳过安全验证，而情感支持或被批评经历形成的记忆则会使后续判断偏向谄媚或过度批评。 |
| [^130] | [Understanding Data Influence in Transportation Networks: A Physics-Constrained Influence-Function Framework](https://arxiv.org/abs/2611.07526) | 本文提出面向约束学习的硬物理约束影响函数框架，在交通网络物理可行空间内刻画训练数据对模型的影响，揭示了物理约束如何改变影响的幅度、方向与空间传播。 |
| [^131] | [Using Transformers to Generate 3-Manifold Triangulations](https://arxiv.org/abs/2611.07514) | 该论文首次将GPT-2风格的Transformer应用于生成三维流形的三角剖分，并通过IsoFLOP曲线分析揭示了神经缩放规律（参数缩放指数α=0.49，每个参数234个token的最优训练比例）。 |
| [^132] | [RLGPA: A Reinforcement Learning-Guided Granularity Planning Attack for Decision-Based Sparse Adversarial Examples](https://arxiv.org/abs/2611.07512) | 提出RLGPA，一种利用在线强化学习在查询预算约束下自适应调度不同粒度支撑集缩减策略的方法，在基于决策的稀疏对抗攻击中显著超越现有方法（CIFAR-10上定向攻击成功率达61%，ResNet50/ViT上达75.5%/73.0%）。 |
| [^133] | [Coverage Hides Churn: Auditing and Preserving Conformal Decisions as VLM Label Registries Grow](https://arxiv.org/abs/2611.07502) | 提出PrefixCP方法，揭示了VLM标签注册表增长时基于softmax的保形分数会因标签均值归一化变化而悄然改变旧标签决策（覆盖率指标无法检测到这种变动），并提供了审计与保持决策稳定性的解决方案。 |
| [^134] | [ZYFA: Speed and Accuracy of FP8 FlashAttention on AMD CDNA4](https://arxiv.org/abs/2611.07496) | ZYFA 是针对 AMD MI355X 的 FP8 FlashAttention 预填充内核，通过直接从分数计算 FP8 概率等三项关键技术，解决了 FP8 加速下 softmax 无法加速及硬件缺乏指令完成跟踪的瓶颈，在保持精度的同时显著提升了长上下文注意力的吞吐量。 |
| [^135] | [Learning Multi-Agent Orchestration:Higher-Order Interactions in Policy Improvement](https://arxiv.org/abs/2611.07495) | 提出交互感知策略优化框架IAPO，通过将两个策略间的性能差异精确分解为个体决策效果及其交互效果，显式建模LLM多智能体编排中决策之间的高阶依赖关系，从而改进编排器的训练。 |
| [^136] | [Statistical and Privacy Guarantees for Deep M-Regression via Stochastic Gradient Langevin Dynamics](https://arxiv.org/abs/2611.07490) | 该论文利用针对深度ReLU网络的PAC-贝叶斯分析，为通过SGLD训练的深度M回归建立了同时刻画网络逼近误差、统计复杂度和算法累积代价的超额风险界，证明注入适度高斯噪声的估计量在Hölder光滑条件下达到最优非参数回归速率，并为SGLD噪声带来的泛化益处及隐私保证提供了理论解释。 |
| [^137] | [OPRD-Imp: a dataset of experimentally observed impurities and side products in multi-step pharmaceutical syntheses](https://arxiv.org/abs/2611.07488) | 该论文构建了首个公开的多步药物合成杂质数据集OPRD-Imp，包含229个反应和784条经专家验证的杂质条目，为机器学习杂质预测模型的基准测试提供了关键数据资源。 |
| [^138] | [Localization Does Not Imply Steerability: A Mechanistic Study of Sign Errors in LLM Mathematics](https://arxiv.org/abs/2611.07478) | 该研究通过逐层因果分析揭示了LLM在多步数学中的符号错误主要由网络后四分之一区段的MLP层在后期产生，且不同模型具有不同机制（固定的“负号盖章”层、符号放大器、救援层），同时指出即便能精确定位错误来源，也不意味着能够有效操控和纠正它。 |
| [^139] | [KOSMOS: Self-scheduling Persistent Kernels for Communication-Compute Overlap](https://arxiv.org/abs/2611.07477) | KOSMOS 提出一种自调度持久内核库，其工作组利用按消费者需求排序的队列根据数据就绪状态在通信与计算角色间动态切换，将 Transformer 层所有 TP+SP 集合通信与 GEMM 融合，无需调度器或预设切分，在 AMD MI350X/MI355X GPU 上将 Megatron-LM 端到端训练最高加速 1.28 倍。 |
| [^140] | [Adversarial Robustness of Kernel Machines: A Comprehensive Analysis](https://arxiv.org/abs/2611.07474) | 本文通过白盒和黑盒攻击全面评估了核机器的对抗鲁棒性，发现核机器天生比神经网络更鲁棒，并提出了一种核机器的近似对抗训练算法。 |
| [^141] | [Echoes of Error: Residual-Directional Local Rollout Consistency for Timeseries Forecasting](https://arxiv.org/abs/2611.07473) | 该论文提出“残差方向局部滚动一致性”方法，通过沿实际反馈扰动方向惩罚预测器的雅可比向量积这一一阶替代方案，精确抑制时间序列预测中自生成误差对未来预测的传播影响，从而避免设计难以实现的残差到状态对齐算子。 |
| [^142] | [The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks](https://arxiv.org/abs/2611.07461) | 提出智能体原生原语与 HEELO 框架，通过为资源配备常驻 LLM 并从原语库组合出任务专用框架，弥合从获取资源到实际应用资源之间“通往真实世界的最后一英里”。 |
| [^143] | [Latency-Aware Adaptive Best-of-$N$](https://arxiv.org/abs/2611.07453) | 该论文提出了一种延迟感知的两阶段自适应 Best-of-N 采样算法，通过比较期望的奖励提升与基于观测延迟预测的成本，动态决定是否抽取以及以多大规模抽取第二批样本，从而在服务器负载变化时更合理地分配测试时计算资源。 |
| [^144] | [Two-stage residual learning for calibrated uncertainty in solar irradiance forecasts](https://arxiv.org/abs/2611.07444) | 本文提出一种两阶段残差学习方法：第一个LSTM网络预测预期太阳辐照度，第二个网络建模预测误差的分布，从而为太阳辐照度预测提供校准的不确定度估计。 |
| [^145] | [IMUDINO: Self-distilled Foundation Model for Inertial Sensing](https://arxiv.org/abs/2611.07437) | IMUDINO是首个在人体活动识别、健康监测、动作捕捉和里程计等多种惯性传感任务上均达到有竞争力性能的通用基础模型，其核心创新在于通过语义保持的运动变换将DINO自监督学习范式适配到惯性数据上。 |
| [^146] | [Truncated backpropagation through time induces a two-sided gradient turnpike property](https://arxiv.org/abs/2611.07415) | 本文证明，在状态转移雅可比矩阵指数衰减等温和条件下，截断的时间反向传播具有“双边梯度巡回性质”，即截断与完整上下文的梯度贡献误差随远离两侧边界而指数衰减，从而为梯度对齐和近似完整上下文平稳性提供理论保证。 |
| [^147] | [Spectral symmetry fingerprints in equivariant representations](https://arxiv.org/abs/2611.07410) | 本文提出谱对称性指纹，将等变表示的奇异值谱作为结构对称性的统一指纹，实现了方向不变、架构无关、无参数的点群识别与连续对称性分析。 |
| [^148] | [Rigorously Certified Neural-Network Computation of Nonlinear Water Waves at Near-Machine Precision](https://arxiv.org/abs/2611.07395) | 本文提出将物理信息神经网络与严格的计算机辅助数值认证及形式化验证相结合的框架，在无训练数据条件下以近机器精度计算非线性重力-毛细行波，统一再现Stokes波、Crapper波与Wilton涟漪，并发现新的大幅值候选解。 |
| [^149] | [CacheGraft: Enhancing Retrieval-Augmented Generation via Latent-Space Document Enrichment](https://arxiv.org/abs/2611.07393) | CacheGraft基于缓存到缓存（C2C）融合框架，将语料库段落离线编码为潜空间KV缓存表示，在查询时直接注入小模型，无需大模型参与即可增强检索增强生成（RAG）的答案质量。 |
| [^150] | [Adversarial Multiplicity: Cherry-Picked Models and the Limits of Disclosure](https://arxiv.org/abs/2611.07378) | 本文揭示了预测多重性可被恶意武器化——操纵者仅需微调模型（甚至在通过黑盒审计的情况下）就能在几乎不损失预测精度的前提下逆转任何个体预测结果，这从根本上挑战了基于信息披露的监管保障机制。 |
| [^151] | [When Scores Conflict with Preferences: Pre-Alignment Diagnosis and Adaptive Correction of Constraint Drift in DPO](https://arxiv.org/abs/2611.07375) | 该论文揭示了DPO训练中的系统性失效模式“约束漂移”，提出可从数据集统计量直接计算的闭式对齐前诊断指标 $\rho^{*}_{\mathrm{eq}}$，并据此构建冲突感知DPO（CoDPO），通过自适应降低冲突样本对的权重来校正约束漂移。 |
| [^152] | [Test-Time Self-Play: Scaling without a Verifier](https://arxiv.org/abs/2611.07367) | 该论文提出测试时自我博弈（TTSP），一种基于非对称自我博弈、无需验证器的测试时扩展方法，在无验证器的情况下于 LiveCodeBench v6 上实现了 mean@8 提升 35% 的绝对改进，且验证器越弱其优势越明显。 |
| [^153] | [Asymptotic-Preserving Tensor Neural Networks for 3D High-Field Kinetic Equations with Applications to Uncertainty Quantification](https://arxiv.org/abs/2611.07364) | 本文提出渐近保持张量神经网络（APTNN）方法，利用CP分解将高维损失积分化为一维积分的组合，从而高效求解三维高场动理学方程并应用于不确定性量化。 |
| [^154] | [From What Is Used to What Is Missing: Rethinking Dynamic Rank in LoRA](https://arxiv.org/abs/2611.07349) | 该论文从几何视角重新思考 LoRA 的动态秩，揭示因子梯度的不可观测子空间与低秩流形法空间重合，并提出利用主要法向奇异方向及其谱能量来确定局部最优的秩扩展，从而判断当前适应空间是否满足任务需求。 |
| [^155] | [AtlasULP: Domain-aware Universal Link Prediction via Relation Atlas](https://arxiv.org/abs/2611.07347) | 该论文提出AtlasULP，通过关系图谱捕获高阶信息并实现领域感知能力，使模型能够在无需训练的情况下对任意未见图进行推理，克服了现有通用链接预测方法无法捕捉长距离依赖和泛化受限的问题。 |
| [^156] | [Last-Iterate Convergence of Stochastic Gradient Methods for Unbounded Problems](https://arxiv.org/abs/2611.07334) | 本文在去除了有界定义域和有界方差假设的条件下，证明了随机梯度方法的末次迭代具有接近最优的 $\widetilde{\mathcal{O}}(1/\sqrt{T})$ 收敛率，并在迭代总次数已知时达到了最优的 $\mathcal{O}(1/\sqrt{T})$ 阶，与强假设下的已知结果相匹配。 |
| [^157] | [Permanent and Transient Representations for Continual Reinforcement Learning](https://arxiv.org/abs/2611.07327) | 本文提出在持续强化学习的互补学习系统中为长期学习采用参数化表示、为瞬态适应采用非参数化表示，并引入一种新颖的非参数近似方法来估计瞬态价值函数，从而实现复杂任务中的在线学习并优于竞争性基线方法。 |
| [^158] | [Rethinking Data Selection: The Importance of Coverage over Difficulty in LLM Fine-Tuning](https://arxiv.org/abs/2611.07323) | 该研究发现在LLM微调的数据选择中，数据对输入分布的覆盖度比样本难度更重要，并提出一种简单的基于聚类的选择方法，在单领域任务上与随机选择持平，在多领域数据集上优于随机选择。 |
| [^159] | [WeightLoop: Weight-Space Programming with Experimental Feedback](https://arxiv.org/abs/2611.07319) | WeightLoop框架让语言模型编写固定的可复用参数编辑程序，将实验反馈转化为对已部署预测器的持久权重更新，无需额外的LLM调用或梯度重训即可实现模型的反复改进。 |
| [^160] | [Provably Efficient Representation Learning for Low-Rank CMDPs](https://arxiv.org/abs/2611.07295) | 该论文提出了首个结合LSVI-UCB与原始-对偶技术的无模型表示学习算法MFRLC，能够可证明地高效求解低秩约束马尔可夫决策过程，在有效平衡奖励遗憾与约束违反的同时学习未知表示。 |
| [^161] | [How do Transformers Perform Low-Level Control In-Context?](https://arxiv.org/abs/2611.07291) | 该论文研究了GPT-2风格的Transformer如何仅凭无外力的系统演化轨迹作为上下文，在不掌握具体参数的情况下实现对未知参数的不稳定非线性系统（小车倒立摆与Acrobot）的上下文学习式控制。 |
| [^162] | [Graph Neural Networks for Resource Allocation in Integrated Sensing and Communication](https://arxiv.org/abs/2611.07277) | 本文提出基于图神经网络的功率分配算法GNN-ISAC，通过在图中区分服务提供者间干扰与提供者内功率预算共享两类边，并以感知惩罚的和速率损失进行无标签训练，在满足感知SINR约束下高效最大化车辆通感一体化网络的通信和速率。 |
| [^163] | [Satisficing as competitive Bayesian weighting: a minimal model of goal-directed choice under an unattained target](https://arxiv.org/abs/2611.07270) | 该论文证明“风险敏感满足化”这一多臂老虎机学习规则可等价改写为类贝叶斯估计器“注意力竞争贝叶斯（ACB）”——未达成的抱负水平充当所有动作的先验均值，有限试验预算决定证据覆盖先验的强度，从而给出了以未实现目标为参照的目标导向选择的最小形式化模型。 |
| [^164] | [How Deep is the Signal? Time-Context up to 30 Years for Colorectal Cancer Prediction in a Primary-Care EHR Model](https://arxiv.org/abs/2611.07265) | 该研究首次系统评估了EHR基础模型的“时间上下文”对疾病预测的影响，发现将日历回溯窗口从2年延长至最长30年可显著提升结直肠癌的早期预测性能。 |
| [^165] | [Energy Measurement and Optimization of TinyML in Resource-Constrained Embedded Systems: A Systematic Mapping Study](https://arxiv.org/abs/2611.07264) | 该研究通过系统性映射方法全面梳理了能耗感知的TinyML研究领域，构建了涵盖主要模型与学习方法、能耗评估方法与指标、技术影响因素以及优化技术的结构化分类体系。 |
| [^166] | [Evolutionary Adaptive Neighborhood Learning: A Genetic Algorithm for Instance-Dependent k Optimization in Nearest Neighbor Classification](https://arxiv.org/abs/2611.07257) | 该论文提出EANK方法，通过遗传算法优化一个低维策略，根据局部邻域特征（密度、标签熵、投票边际、集中度）为每个实例自适应地确定k近邻分类器的邻居数量k(x)，突破了传统全局单一k值选择的局限性。 |
| [^167] | [TriAttention: Unified Sparse Mixed-Precision Attention for Long-Context LLM Inference](https://arxiv.org/abs/2611.07256) | TriAttention提出了一种统一的稀疏混合精度注意力方法，通过离线校准为每个注意力头在跳过、4位和8位计算之间分配计算区域，在共享计算预算下兼顾稀疏性与量化精度，从而优化长上下文LLM推理的成本与精度平衡。 |
| [^168] | [Using Transformers for Teaching and Learning via Symbolic Distillation of Neural A-life Agents](https://arxiv.org/abs/2611.07254) | 该论文提出使用Transformer模型替代传统高昂的随机搜索方法，将不透明的神经人工生命智能体策略蒸馏为紧凑可解释的符号策略，从而降低计算成本，并促进智能体之间的知识共享与种群级学习。 |
| [^169] | [Goal Revelation as a Bridge Between Information-Theoretic Empowerment and Capability-Based Human Power](https://arxiv.org/abs/2611.07247) | 该论文提出“目标揭示”（即人类潜在目标与实际轨迹之间的互信息）作为统一两种权力度量的桥梁：基于能力的个体权力通过Fano不等式构成其下界，而信息论赋权构成其上界，从而为AI安全中的权力量化研究建立了理论联系。 |
| [^170] | [KG-Neighbor: Knowledge-Guided Neighborhood Construction for Embedding Visualization](https://arxiv.org/abs/2611.07246) | 本文提出知识引导邻域图构建方法KGN，利用从实例中抽取的知识图谱三元组的共享程度作为语义证据来调整嵌入距离并重建top-k邻域图，从而在不使用类别标签的情况下提升嵌入可视化中邻域关系的可靠性，并配套提出IDProfile工具来解释类别标签与邻域多数类之间的残余不匹配。 |
| [^171] | [MAC-v2: Long-horizon model-based value expansion from pixels](https://arxiv.org/abs/2611.07245) | MAC-v2通过训练无重构的潜在世界模型并结合动作分块，实现了从像素出发的长时程基于模型的价值扩展，有效缓解了表征学习与动力学预测误差在想象轨迹上的递归累积问题。 |
| [^172] | [$D^2$-OPD: Separating Direction and Distance in Extrapolative On-Policy Distillation](https://arxiv.org/abs/2611.07241) | D²-OPD通过解耦外推方向与距离来改进外推式在线策略蒸馏——用每前缀硬KL约束限制目标与教师的距离，并用教师训练后偏移与初始师生差距的自适应组合确定外推方向，从而在多种场景下提供更稳健的学习信号。 |
| [^173] | [Archetypal contexts in the neurons of language models: an architecture for intelligence?](https://arxiv.org/abs/2611.07234) | 该研究在Qwen3.8-27B模型中发现，跨领域的原型语境（如“临界点”）、主题和语言分别由三组可独立操控的神经元编码，且网络层呈现“外层语言—中间主题—核心原型”的三明治式组织架构，暗示这可能正是智能的底层架构。 |
| [^174] | [Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost](https://arxiv.org/abs/2611.07232) | RecSA在KV缓存上构建递归树结构，利用从稠密注意力蒸馏且跨层级共享的池化键，使每个查询的选择成本降至O(log L)，从而以对数成本实现了百万token级的长度外推。 |
| [^175] | [TileQuant: Tile as the Precision-Allocation Unit for LLM Mixed-Precision Post-Training Quantization](https://arxiv.org/abs/2611.07217) | TileQuant提出以硬件对齐的二维权重tile为精度分配单元的免训练混合精度训练后量化框架，通过离线校准得到的激活感知与梯度感知评分，让高分tile保留16位精度、其余tile使用8位，从而在降低校准成本与解码开销的同时更好地保持LLM质量。 |
| [^176] | [Compositional Unalignment: Auditing Safety in Large Vision-Language Models](https://arxiv.org/abs/2611.07216) | 本文提出以“去对齐”作为安全对齐的审计手段，揭示大型视觉-语言模型的对齐并不稳健——极少量的监督微调即可达到与多阶段强化学习流程相当的基准分数，且模型对组合性伤害的识别与拒绝能力是相互解耦的。 |
| [^177] | [Fractional physics-informed neural networks to model social media virality](https://arxiv.org/abs/2611.07215) | 本文提出一种分数阶物理信息神经网络，能够同时学习社交媒体参与度的时间记忆与非线性增长动力学，并通过符号回归生成可解释方程，同时解决了观测噪声导致的分数阶估计偏差问题。 |
| [^178] | [Canonical Correlation Neural Networks](https://arxiv.org/abs/2611.07212) | 本文提出典型相关神经网络（CANEs），将视图内协方差和跨视图协方差作为不同的关系归纳偏置引入网络架构，把以PCA为理论基础、稳定且参数高效的协方差神经网络框架从单视图数据推广到了多视图场景。 |
| [^179] | [Task-Calibrated Random Features via Influence-Based Error Cancellation](https://arxiv.org/abs/2611.07209) | 该论文提出任务校准随机特征（TCRF），通过基于任务特定影响构建的可重用校准空间来平衡频率矩并消除对下游任务有影响的采样误差，从而改进核岭回归和核Stein差异等核方法的近似质量。 |
| [^180] | [TrustMix: Graceful Degradation for Pretrained Models under Distribution Shift](https://arxiv.org/abs/2611.07206) | TrustMix是一种无需重训练的轻量级后处理方法，它根据OOD偏离程度将预训练模型的预测分布与用户指定的回退分布混合，使模型外推行为显式可控并实现优雅降级。 |
| [^181] | [Backbone-Informed Adaptation after EEG Pretraining: A Capacity-Controlled Study](https://arxiv.org/abs/2611.07192) | 在参数量匹配、容量受控的公平比较下，基于主干结构设计的适配器相较于简单的逐token多层感知机适配器，在多个EEG基础模型和下游任务上并未表现出一致优势。 |
| [^182] | [ShapleyOG: Fast Adaptive Shapley Value Estimation with Odd Gaussian Processes](https://arxiv.org/abs/2611.07173) | ShapleyOG利用Shapley值仅依赖于奇数分量的特性，提出了一种基于奇数高斯过程的快速自适应估计方法，克服了ShaplEIG等现有自适应方法因每轮迭代重复拟合高斯过程代理模型而带来的计算开销问题。 |
| [^183] | [Meta-Frieren: Beyond Single Level Online Learning](https://arxiv.org/abs/2611.07169) | Meta-Frieren是一个两阶段在线优化的通用框架，通过UCB风格的元层面选择准则最大化累积奖励，统一并扩展了已有理论保证，首次为Feel-Good Thompson采样推导出高概率遗憾界，并可成功应用于跨多个模型类的超参数优化。 |
| [^184] | [Multi-step SWE: Can Coding Agents Keep Building on Their Own Code Over Long Horizons?](https://arxiv.org/abs/2611.07166) | 提出长时程基准 Multi-step SWE，包含69个由5-15个依赖步骤组成的软件工程任务链，要求智能体在自身编写的代码上持续迭代开发，最强前沿模型的解决率仅18.5%，揭示了当前编程智能体在长时程自我延续开发中的严重不足。 |
| [^185] | [Structural Conditional Independence Test via Neural Quantile Resolution](https://arxiv.org/abs/2611.07165) | 提出QRCIT——一种基于神经分位数形变的条件独立性检验方法，通过轻量级残差分位数流模型在学习检验统计量的同时，以可检查的形变场揭示条件分布的具体变化位置，并配套发布了结构依赖基准SDB。 |
| [^186] | [A Survey of Nonlinear Filtering Methods for Data Assimilation and Recent Advances in Generative AI-Enabled Score Filters](https://arxiv.org/abs/2611.07160) | 本综述系统梳理了数据同化领域三类经典非线性滤波方法（卡尔曼类、粒子类和核方法）的局限性，并介绍了利用生成式人工智能得分滤波器求解最优滤波问题的最新进展。 |
| [^187] | [Depth or Description? When a One-Step Latent Score Replaces Lookahead in Proactive Information Acquisition](https://arxiv.org/abs/2611.07155) | 该论文发现主动信息获取中对深度前瞻搜索的需求往往源于隐藏原因的描述方式——基于能编码决策信息的合适潜在变量构建的一步式评分，可以用极小的搜索代价替代大部分深度前瞻搜索。 |
| [^188] | [FastAFD: When Does Attention-FFN Disaggregation Pay on Rack-Scale NVLink?](https://arxiv.org/abs/2611.07149) | 本文构建了面向GB200 NVL72机架级NVLink系统的FastAFD引擎，通过在微批次间重叠MoE与注意力计算，实现了相比最优vLLM宽EP设置高达1.17倍的解码吞吐量提升，并提出了一个仅凭共置vLLM剖析数据即可在3%误差内预测AFD收益的分析模型。 |
| [^189] | [Context-Aware Automated NMR Structure Elucidation](https://arxiv.org/abs/2611.07133) | 该论文提出一种将自动化逆合成分析生成的合成前体作为化学情境信息融入AI核磁共振结构解析框架的方案，从而提升化学结构预测的性能与实际应用价值。 |
| [^190] | [A Proxy-Based Method for Measuring Racial Disparities in~Text-to-Image~Generators](https://arxiv.org/abs/2611.07132) | 该论文借鉴美国反歧视法中的种族代理变量，提出一种自动化审计方法，通过将生成图像的种族分布与真实人口基准数据比较，并结合肤色和图像质量分析，来衡量文生图模型中的种族表征偏见。 |
| [^191] | [Bayesian Tensor Train Kernel Machines with Automatic Model Complexity Inference](https://arxiv.org/abs/2611.07129) | 本文提出了贝叶斯张量链核机器（BTT-KM），通过在TT核心上施加分层ARD先验并结合平均场变分推断，实现了从数据中自动推断TT秩与特征维度的闭式贝叶斯学习，且计算复杂度与确定性方法相同。 |
| [^192] | [Scalable Bayesian Gaussian Process Classification for Cancer Histopathology: Calibrated Uncertainty Under Distribution Shift](https://arxiv.org/abs/2611.07127) | 该论文提出一种可扩展的贝叶斯高斯过程分类框架，通过结合RPCholesky低秩核分解、非中心化参数化HMC后验采样和主元诱导点预测方案，首次在约30万训练点规模上实现可处理的HMC贝叶斯推断，为癌症组织病理学诊断在分布偏移下提供校准的不确定性量化。 |
| [^193] | [Data-Driven Modeling of Molten Chloride Corrosion Using Static Test Data](https://arxiv.org/abs/2611.07125) | 该研究汇编291个静态熔融氯化物腐蚀实验数据，创新性地提出氯化盐制备指数（CSPI）来量化盐条件差异，构建可解释机器学习框架实现了高精度的腐蚀速率预测（交叉验证R²达0.85）。 |
| [^194] | [Machine Learning for Live-Birth Prediction After IVF/ICSI From Structured Clinical Data: A Systematic Review, Meta-Analysis, and Critical Appraisal](https://arxiv.org/abs/2611.07124) | 本系统综述与荟萃分析对48项研究进行批判性评估后发现，机器学习预测IVF/ICSI后活产的模型在内部验证时AUC可达0.814，但在外部验证时降至0.756，表明现有模型泛化能力有限。 |
| [^195] | [PromptOps: Iterative Prompt Optimization for LLM-as-a-Judge Systems](https://arxiv.org/abs/2611.07123) | 本文提出PromptOps闭环框架，将LLM评判器在部署中与人工评估策略一致性下降的问题定义为“评判器漂移”，通过分离何时优化、应用何种优化流程、选用哪些反馈记录及如何更新提示词这四个运营决策，实现LLM-as-a-Judge系统的持续监控与迭代重新校准。 |
| [^196] | [Credibility Clustering for Experience Rating: An Automated Approach to Risk Group Allocation in Group Life Insurance](https://arxiv.org/abs/2611.07110) | 该论文提出将多维 Bühlmann-Straub 信度模型与 k-means 聚类相结合的“信度聚类”方法，以数据驱动方式自动完成团体人寿保险中的同质风险组划分，取代传统依赖人工决策与主观阈值的分类方式。 |
| [^197] | [A Night at the LLM Zoo: Data Conditioned Flow Matching for Generating Pretrained LLM Weights](https://arxiv.org/abs/2611.07103) | 本文构建了首个通过系统变化数据混合（网络文本、书籍、数学、编程、法语）训练的大规模LLM模型动物园，并提出以数据混合为条件的流匹配模型，能够在Gram-PCA压缩权重空间中直接生成完整的预训练LLM权重，且可零样本泛化到未见过的数据混合。 |
| [^198] | [Server-Side Sparse Assignment for Federated Mixture-of-Experts](https://arxiv.org/abs/2611.07093) | 提出S³-MoE服务器端稀疏专家分配框架，从部分观察中估计客户端特定的专家相关性并通过利用-探索机制细化专家分配，解决联邦微调LLM中边缘设备异构性带来的专家匹配难题。 |
| [^199] | [Synthesizing symbolic models of human and machine reasoning at scale](https://arxiv.org/abs/2611.07088) | 提出行为条件化符号合成（BCSS）方法，利用AI自动搜索可解释的确定性符号程序，大规模建模人类与机器的推理行为，并揭示两者在推理中都存在过度简化和错误推断问题特征依赖关系的共同特征。 |
| [^200] | [Optimistic Hedge in General-Sum Games: Fundamental Limits and Regret Bounds via Temporal Approximation](https://arxiv.org/abs/2611.07078) | 本文证明了一般和博弈中Optimistic Hedge能否达到与博弈规模多项式相关的常数遗憾界取决于参与者数量，并在双人博弈情形下给出了与时间无关的O(n log n)遗憾界。 |
| [^201] | [Shaping & Monitoring Models through their Internal Representations](https://arxiv.org/abs/2611.07071) | 该论文提出“机制塑造”统一框架，在训练中通过约束模型内部表示所表达和依赖的概念来塑造模型行为，在多领域、多规模、多训练范式下显著减少不良行为同时保持任务性能，并引入监控器类别的层级体系来评估被塑造概念是否仍然可被监控。 |
| [^202] | [Finite-Time Convergence and Regret Guarantees for Neural Reinforcement Learning in POMDPs](https://arxiv.org/abs/2611.07069) | 该论文针对POMDP中采用非线性神经网络价值函数逼近的强化学习，首次建立了有限时间收敛性与遗憾保证，克服了以往理论仅适用于线性逼近或依赖不符合POMDP轨迹特性的独立同分布采样假设的局限。 |
| [^203] | [Stochastic Dynamics of Generative AI: From Autoregressive Models to Interacting LLM Networks and Neural SDEs](https://arxiv.org/abs/2611.07065) | 本文以随机动力系统的统一视角刻画生成式AI的三类模型：将大语言模型建模为高维非线性自回归系统、用平均场常微分方程描述交互式LLM网络的集体行为，并将神经随机微分方程作为随机轨迹的连续时间生成模型。 |
| [^204] | [LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at Test-Time](https://arxiv.org/abs/2611.07025) | 该论文提出LowRAM方法，利用注意力头内“值可由键预测”的特性，在测试时通过值缓存摊销（仅存储重构最差令牌的残差，实现4倍近无损压缩），并结合跨层低秩键压缩，构建出接近无损的可组合KV缓存压缩流水线。 |
| [^205] | [Robust Stream Classification using Time Neutralising Decision Trees](https://arxiv.org/abs/2611.06991) | 本文提出了一种名为“时间中和树（TNT）”的决策树架构，通过中和时间因素的影响，在存在概念漂移且实例可被存储的数据流场景下实现鲁棒的流分类。 |
| [^206] | [ToroFit: Fitting Wedge Sums of Tori to High-Dimensional Data](https://arxiv.org/abs/2611.06990) | 本文提出ToroFit框架，利用持续上同调与上积识别高维点云数据中各环面结构分量，并通过将低能量多元三角多项式拟合归结为广义岭回归，实现对环面楔和型非线性数据的高效检测与参数化。 |
| [^207] | [Big Models, Small Gains: Frozen Forecasters as Classifiers on MONSTER](https://arxiv.org/abs/2611.06988) | 该论文在 MONSTER 基准上对冻结的预测基础模型（Reverso 和 Chronos-Bolt）表示进行受控评测，发现预训练相比随机权重模型仅带来小幅且不稳定的分类准确率提升，表明这些预训练表示未必优于无需预训练的强分类器，尤其在标注稀缺时其价值存疑。 |
| [^208] | [Octopus: One Brain, Many Branches for Latent-KV Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2611.06956) | Octopus提出了一种单模型多智能体强化学习框架，让所有分支共享同一主干网络，并通过零拷贝基数分叉、仅注意力KV挂载等协同设计机制，在GPU驻留的KV状态有向无环图上进行高效协调，从而消除KV缓存重复复制和跨分支依赖丢失的问题。 |
| [^209] | [SteelAgent: LLM-Orchestrated, Physics-Informed Steel Property Prediction with Audited Generalization and Calibrated Uncertainty](https://arxiv.org/abs/2611.06953) | SteelAgent 将大语言模型编排、冶金描述符与机器学习相结合用于钢性能预测，并揭示评估时留出单元的选择会彻底改变结论——随机划分下表现优异的模型在留出陌生钢族或数据源时性能骤降且不确定性校准失效。 |
| [^210] | [BrainPFN: Amortized Brain-Behaviour Prediction via Prior-Data Fitted Networks](https://arxiv.org/abs/2611.06934) | 提出BrainPFN——一种在由约7千个真实功能连接矩阵构建的领域先验上进行元训练的先验数据拟合网络，通过摊销式推断实现可跨数据集迁移、在小样本临床/认知场景下依然准确的大脑-行为预测，克服了核岭回归需逐评分调参、无法迁移的局限。 |
| [^211] | [Capturing Membrane Dynamics and Spike Timing of Human Neurons at Scale using Neural Operators](https://arxiv.org/abs/2611.06933) | 该论文提出一个细胞类型特异性的神经算子框架，通过在同类多个神经元模型上学习共享算子并以可解释的电生理特征为条件，实现大规模神经元群体建模，同时保留丰富的膜动力学、锋电位时序及细胞类型内变异性。 |
| [^212] | [Reading a neuron's tuning from its correlations with other neurons](https://arxiv.org/abs/2611.06927) | 仅利用神经元群体的相关矩阵（不含刺激信息、无标签），transformer模型即可解码小鼠视觉皮层神经元的偏好朝向和感受野位置，表明神经元间的相关结构本身已蕴含其调谐特性信息。 |
| [^213] | [From Fixed Defaults to Calibrated Activation Policies for Long-Term Time-Series Forecasting](https://arxiv.org/abs/2611.06559) | 该研究提出一种经过校准的激活选择策略，能够根据任务自适应地选择激活函数及其位置，并在证据不足时保留默认配置，在1,225次Informer训练实验中平均将长期时间序列预测的均方误差降低7.86%。 |
| [^214] | [The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention](https://arxiv.org/abs/2611.06476) | 对齐微调带来的能力损失并非均匀发生，而是分级且任务选择性的——抽取类任务几乎不受影响，JSON生成轻微下降，而路由任务出现约30分的严重退化。 |
| [^215] | [Cross-Asset Knowledge Transfer for Early Fault Detection of Data-Scarce Assets Without Target Fault Data](https://arxiv.org/abs/2611.06194) | 本文提出一个零样本域适应框架，在目标资产完全没有故障数据的情况下，仅利用其正常数据即可通过EWMA自动标注源资产数据并借助改进的域对抗神经网络实现跨资产早期故障检测。 |
| [^216] | [Neural Stochastic Forecasting for Lateral Molecular Diffusion Simulations](https://arxiv.org/abs/2611.06157) | 提出了一种结合Transformer构象表示、多时间尺度循环网络和带学习记忆的随机潜过程的神经随机模型，能够从粗粒化分子动力学轨迹高效预测膜蛋白在膜内的长时间集体扩散。 |
| [^217] | [Reward Budgeting Reduces Premature Convergence in Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2611.05942) | 该论文提出“奖励预算化”这一插件式优势构造方法，通过为每个提示分配固定的总奖励预算来缩小优势差距，从而有效缓解大语言模型推理强化学习中的过早收敛与早期性能平台期问题。 |
| [^218] | [ZipDrift: Generating Multimodal Short-Horizon MPC Costs with One-Step Drifting Models](https://arxiv.org/abs/2611.05793) | 该论文提出ZipDrift，通过训练一个向低闭环代价倾斜的一步漂移生成器来产生多模态的MPC代价修正候选，再由学习到的评论家排序选优，解决了确定性回归在多模态场景下平均各模态或继承专家不一致平局决策的问题。 |
| [^219] | [RESHPart: Feedback-Reshaped Spectral Embedding for Hypergraph Partitioning](https://arxiv.org/abs/2611.05729) | 该论文提出FRESHPart框架，从谱嵌入几何出发，通过几何反馈和搜索反馈重塑谱嵌入，使其更有利于基于投影的超图划分，并在Titan23基准上相比hMETIS显著降低割值。 |
| [^220] | [Distinguishing Initial-Condition and Model Error Using Machine-Learning Weather Models](https://arxiv.org/abs/2611.05592) | 本研究通过在GraphCast和盘古气象大模型上对比门控与非门控优化方法，利用机器学习天气模型实现初始条件误差与模型误差的区分。 |
| [^221] | [Unifying and Beyond the Theory of Pervasive Capability Degradation in LLM Post-training SFT from Feature Interference](https://arxiv.org/abs/2611.05003) | 本文提出了一个基于叠加态下特征干扰的自底向上几何理论，从微观机制解释了大语言模型SFT中普遍性能力退化的成因，并将现有缓解方法统一为六种范式，同时揭示了其有效性的来源与局限。 |
| [^222] | [P$^2$-MOPD: Accelerating Multi-Teacher On-Policy Distillation via Parameter Priors](https://arxiv.org/abs/2611.03953) | 该论文提出 P$^2$-MOPD，将学生与专家参数的线性组合视为一个可微模型，在在线策略蒸馏目标下联合优化混合系数与学生权重，使专家作为参数先验而非限制学生，从而在四个领域上仅用一半训练步数就实现了比 MOPD 更快、更均衡的能力恢复。 |
| [^223] | [FADE: Factored Ability--Difficulty Filtering for Online RLVR Scheduling](https://arxiv.org/abs/2611.03625) | FADE 提出一种双线性状态空间滤波器，将策略能力（快速变化）与提示难度（相对稳定）这两个不同时间尺度的因素分解开来，从而在在线 RLVR 训练中更准确地筛选出有信息量的提示，避免浪费生成预算。 |
| [^224] | [Sonar-MoE: In-Backward Expert Updates for Memory-Efficient Mixture-of-Experts Training](https://arxiv.org/abs/2611.03577) | Sonar-MoE通过在反向传播过程中即时更新并释放专家梯度，将MoE训练的峰值内存降低19%–22%、每步速度提升17%–23%，同时保持与标准训练位级别一致的结果。 |
| [^225] | [Seeing Why: Exact Evidence Maps for Quantum and Optical Image Classifiers](https://arxiv.org/abs/2611.01993) | 本文提出玻恩证据图（BEM），无需梯度电路和反向光路，即可在量子芯片或光学硬件上仅用 $1+2g^2$ 次运行，将量子与全光图像分类器各类别的玻恩概率精确分解为逐像素的贡献。 |
| [^226] | [Write Facts the Way Models Read Them: Lifting the Two-Hop Curse](https://arxiv.org/abs/2611.01519) | 论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。 |
| [^227] | [Unified dynamics of training, learning, inference, and reinforcement in neural networks](https://arxiv.org/abs/2611.01345) | 该论文通过九项nanoGPT实验建立了训练、学习与推理间的定量关系，提出了预测更新后能力转变的直接预测器（准确率91.43%），发现重新平衡反馈可恢复被集中正确反馈侵蚀的技能，从而将三者闭合为统一的训练-学习-推理-强化动力学。 |
| [^228] | [Profile-aware diagnostics of discarded parameter-Hessian mass](https://arxiv.org/abs/2611.01335) | 提出一种轮廓感知的诊断框架，通过在不同分组规模和组内曲率轮廓参考下测量参数Hessian的块外平方占比，来量化块对角曲率模型所丢弃的曲率质量，并评估候选修正的实际效果。 |
| [^229] | [Sampling Design and Confidence Tubes for Integral Curves on Riemannian Manifolds](https://arxiv.org/abs/2611.00771) | 本文在紧黎曼流形上提出了估计未知向量场积分曲线的被动与主动两种抽样方法，证明了被动核估计的曲线误差阶为 $n^{-2/(d+3)}$ 并给出由线性随机微分方程刻画的函数型中心极限定理，同时证明基于先导信息的主动抽样可将期望最大曲线误差提升至 $N^{-1/2}$ 阶，从而为构建置信管奠定基础。 |
| [^230] | [PeptSol: A Translational Benchmark for Protected-Peptide Solubility in Drug Manufacturing](https://arxiv.org/abs/2611.00749) | 提出了PeptSol——首个面向液相肽合成工艺开发的保护态肽溶解度开放基准数据集，包含1388条人工整理的溶解度测量数据并采用化学感知划分，弥补了现有肽性质基准在保护基团、末端修饰和非天然残基方面的不足，为分子表示模型提供了严格的评估平台。 |
| [^231] | [When Does Test-Time Search Help World Action Models? An Empirical Audit of Candidate Selection and Simulator Integrity](https://arxiv.org/abs/2610.41970) | 该论文通过预注册的实证审计发现，世界动作模型的测试时搜索收益存疑：候选选择几乎不存在可利用信号（学习到的价值选择器比随机选择还差 1.66 个百分点），token 剪枝虽有 47% 的 FLOPs 节省但实测端到端加速最高仅 1.15 倍、未达 1.5 倍目标，从而揭示测试时扩展所依赖的关键前提往往并不成立。 |
| [^232] | [TSBPO: Tree-Structured Branching for Policy Optimization in Interactive Language Agents](https://arxiv.org/abs/2610.41697) | TSBPO提出一种树结构的采样到更新范式，通过似然拟合的根/分支分配器定向采样、基于兄弟片段与匹配观察状态的延迟反馈分配、以及跨活跃轮次归一化的词元损失聚合，在单一训练循环中解决了从终端结果训练交互式语言智能体的三大耦合难题。 |
| [^233] | [V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models](https://arxiv.org/abs/2610.37019) | V-LUMEN提出了一种视觉查找记忆机制，通过冻结码本离散化视觉特征、空间聚合哈希构建记忆键并结合文本条件调制信号，以完全非参数化的寻址方式为视觉-语言模型实现嵌入层面的扩展，显著提升冻结主干网络在推理基准上的表现并具备零样本迁移能力。 |
| [^234] | [Improving Signals, Declining Encoder: A Known-Answer Test of Label-Free Monitoring in Self-Supervised Learning](https://arxiv.org/abs/2610.35766) | 该论文通过已知答案的合成数据实验揭示，自监督学习中常用的无标签监测信号（如损失和特征统计量）无法察觉编码器的实际退化，甚至会给退化中的训练更高的评分，从而可能误导检查点选择。 |
| [^235] | [Global $\mathcal{O}\left(k^{-2}\right)$ Convergence of Multilevel Gradient-Regularized Newton Methods](https://arxiv.org/abs/2610.35195) | 本文提出了适用于利普希茨连续海森矩阵无约束凸优化的多层梯度正则化牛顿方法，通过在低维子空间中计算牛顿修正并引入行嵌套限制算子，在大幅降低计算成本的同时实现了全局 $\mathcal{O}(k^{-2})$ 收敛速率。 |
| [^236] | [Approximating the Euclidean Ball by Polytopes: Surprising Rates in High Dimensions](https://arxiv.org/abs/2610.34651) | 本文给出了用多面体逼近高维欧氏球所需顶点数的精确估计，证明仅需次指数级数量的顶点 exp(O(√d/ε)) 即可，但达到固定误差却需要指数级数量的面，并肯定回答了 De 等人关于次指数顶点复杂性的公开问题。 |
| [^237] | [Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation](https://arxiv.org/abs/2610.31166) | 提出物体引导的感知记忆机制WholeBodyMem，利用SAM3跟踪的物体事件因果选择关键帧，使人形机器人全身VLA策略能够记住行动前的物体历史，从而在非马尔可夫动态场景中大幅提升成功率（仿真中29.1%提升至72.6%，真实机器人上36.7%提升至88.3%），并发布了HuMeGo人形机器人记忆基准。 |
| [^238] | [Orbitol: Crystal Generation on Wyckoff Orbits](https://arxiv.org/abs/2610.30086) | Orbitol通过用Wyckoff轨道标记替代原子标记，实现了对空间群对称性的直接控制，能生成保持所要求空间群的稳定新颖晶体结构，且生成速度比现有对称性感知基线快两到三个数量级。 |
| [^239] | [No Escape from the Quadratic Barrier: Adaptive Attacks on Linear $\ell_2$ Sketches](https://arxiv.org/abs/2610.25008) | 本文证明，即使响应者计算能力无限制且掌握全部信息，针对线性 $\ell_2$ 草图的自适应黑盒攻击也只需约 $O(k+\alpha^2k^2)$ 量级的查询即可破坏其精度，因此约 $k^2$ 的二次方防御保证在对数因子范围内已是最优、无法逃脱。 |
| [^240] | [Single-stage joint diffusion of sky images and clear-sky index for ultra-short-term irradiance forecasting: what the frame-to-irradiance stage discards](https://arxiv.org/abs/2610.21795) | 该论文提出一种在单一扩散过程中联合去噪未来天空图像与晴空指数的单阶段辐照度预测模型，省去了传统两阶段设计中的独立回归器且无需实测辐照度输入，在仅有相机的站点上实现了超越十二个基线配置的超短期辐照度预测性能。 |
| [^241] | [Privacy-Preserving Text Inference for Health-Misinformation Classification and Semantic Retrieval](https://arxiv.org/abs/2610.20796) | 本文提出了一种基于CKKS同态加密的隐私保护文本推理框架，将BioBERT、MPNet与TF-IDF等特征融合用于健康错误信息分类（准确率达0.971），并结合语义指纹实现密文域上的语义检索。 |
| [^242] | [Are We Overthinking One-Step Generation?](https://arxiv.org/abs/2610.20331) | 本文提出Priming Models两阶段框架：先用简单的重建目标（如固定噪声水平的像素去噪）预训练，再进行特征空间分布后训练，并发现这一简单的重建阶段即可完全解决单步生成器分布训练的困难。 |
| [^243] | [TANGO: RNA Topology and Geometry Co-Design](https://arxiv.org/abs/2610.20277) | 提出TANGO框架，首次将RNA逆设计表述为拓扑与几何协同设计问题，通过多目标多智能体强化学习和分阶段课程训练（二级结构拓扑感知策略学习、三级结构迁移训练、前沿引导微调），实现同时满足二级碱基配对拓扑与三级几何结构的RNA序列设计。 |
| [^244] | [Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction](https://arxiv.org/abs/2610.19246) | 提出一种双遍拓扑调度方案，将动态kNN图构建基准的选择形式化为以逐块重建（PB）为唯一不动点的阻尼不动点迭代，并借助不翻转证书、最小方差（BLUE）重放增益与ℓ∞信赖域投影进行保守补偿，从而在SE(3)等变结构预测中以低成本逼近逐块重建的拓扑精度。 |
| [^245] | [Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap](https://arxiv.org/abs/2610.19065) | 生成式检索相对稠密检索的价值优势主要源于物品流行度这一静态先验，在稠密检索中加入一步静态价值重排即可逆转差距并保持更高召回率，且工业级实验表明 140 亿参数的 SID-LLM 仍不及小 47 倍的稠密编码器。 |
| [^246] | [Beyond Equation Search: Conjecture-Guided Discovery with SCOPE and ClosureBench](https://arxiv.org/abs/2610.18938) | 该论文提出SCOPE多智能体框架，将符号回归从优化单个方程转向以科学猜想为核心的研究范式，通过多个科学家智能体独立建模、主动实验与证据交换来检验和修正猜想，并配套ClosureBench基准评估猜想驱动式科学发现。 |
| [^247] | [Graph Metanetworks for Transformer Based Language Models](https://arxiv.org/abs/2610.18471) | 该论文提出了一种基于图元网络的方法，全面捕捉Transformer语言模型权重空间中的丰富对称性（包括神经元置换、注意力矩阵的连续一般线性对称性及注意力头置换），并处理词元嵌入等语言模型特有组件。 |
| [^248] | [Structural and Representational Agency in LLM-Driven Timbre Generation and Interpolation](https://arxiv.org/abs/2610.12532) | 本研究通过对一个实际运行的LLM驱动音色界面进行测量，首次证明用户真实可控的声音空间（可精确刻画为维度随放置声音数递增的zonotope）与界面中由语言模型自我报告的位置（误差约为该空间宽度的三倍）是可分离的，揭示了LLM创意界面中“实际能动性”与“表征能动性”之间的脱节。 |
| [^249] | [Best-of-$N$ Guidance for Test-time Diffusion Alignment](https://arxiv.org/abs/2610.05108) | 该论文提出 Best-of-N 引导（BoNG）方法，将 BoN 选择的原理直接融入逆向扩散过程，通过对去噪粒子进行在线 BoN 选择来调整采样轨迹，从而在测试时更有效地将扩散模型与人类偏好对齐。 |
| [^250] | [VIGOR: Zero-Shot Visual Generalization via Latent-Space Consistency in Model-Based Reinforcement Learning](https://arxiv.org/abs/2610.02801) | VIGOR通过非对称弱到强增强与潜空间一致性约束，使基于模型的强化学习在保留样本效率的同时，能够零样本泛化到背景变化、光照变化等未见过的视觉干扰。 |
| [^251] | [Fixed-point neural samplers on discrete spaces](https://arxiv.org/abs/2610.01739) | 本文提出离散吉布斯迭代神经采样器，一种基于掩码扩散的不动点神经采样方法，解决了现有离散神经采样器的模式坍缩和收敛保证问题，可高效扩展至高维系统并支持跨条件的摊销采样。 |
| [^252] | [Learning Rate Transfer for Hybrid Transformer-SSM Architectures](https://arxiv.org/abs/2610.01172) | 该论文发现，在实际使用简化ZOH Mamba的混合Transformer-SSM架构中，尽管SSM操作不满足μP的理论前提，仅凭原始μP规则即可在宽256-2048、深4-32直至十亿参数规模上实现接近零的学习率迁移差距。 |
| [^253] | [Backdoor Purification for LoRA-Tuned LLMs via Null-Space Projection](https://arxiv.org/abs/2610.00685) | 提出了一种基于零空间投影的后门净化方法，无需触发器先验知识、干净参考样本或重新训练，即可净化遭受后门攻击的LoRA微调大语言模型，在降低攻击成功率的同时保留模型的通用能力与下游技能。 |
| [^254] | [Backdoor Containment via Expert Quarantine and Shutdown in LLMs](https://arxiv.org/abs/2610.00663) | 本文提出隔离专家关停策略，通过正则化引导的MoE类架构让后门在训练中形成但被引导至指定的隔离专家组件，部署时可将该组件关停，从而以计算高效的方式遏制LLM后门攻击。 |
| [^255] | [Manifold-Constrained Initial Noise Optimization for Efficient Generative Model Alignment](https://arxiv.org/abs/2610.00365) | 提出ZeNOVA，一种免梯度的初始噪声优化方法，通过退火软值引导、流形约束超球面朗之万动力学和Metropolis-Hastings跳跃，在黑盒奖励场景下实现稳定高效的生成模型对齐。 |
| [^256] | [Structure-aware Reinforcement Learning for Protein Directed Evolution](https://arxiv.org/abs/2609.39048) | StructEvo提出了一种结构感知的强化学习框架，通过增量结构融合编码器动态整合空间知识，并利用结构对齐的分层动作网络分解突变空间，在克服突变体结构稀缺难题的同时超越了现有的蛋白质定向进化方法。 |
| [^257] | [Muon Sublates the Edge of Stability in LLM Pretraining](https://arxiv.org/abs/2609.34915) | Muon 优化器打破了大语言模型预训练中梯度下降的稳定性边缘耦合框架，其损失中性边界与时间对齐遵循不同几何规律，且更新方向始终远离相干反转、训练持续改善。 |
| [^258] | [Physics-Attested Federated Learning: Securing Collaborative Anomaly Detection in Critical Water Infrastructure](https://arxiv.org/abs/2609.34804) | 该论文提出将从物理过程自动挖掘出的不变量作为联邦更新的可验证准入要求，从而有效防御针对关键水利基础设施协同异常检测的数据投毒攻击。 |
| [^259] | [AnchorRep: Defending LLMs Against Cross-Model Adversarial Transfer via Representation Repulsion](https://arxiv.org/abs/2609.32602) | 提出AnchorRep防御方法，利用轻量级LoRA适配器将有害提示词的内部表示与锚点模型的表示相排斥，无需对抗样本即可将跨模型对抗迁移攻击的成功率降低至1.1%以下。 |
| [^260] | [Peer-Grounded Counterfactual Path Planning for Chronic Health Management](https://arxiv.org/abs/2609.30838) | 该论文提出POROS框架，根植于自我效能理论与社会比较理论，通过从真实相似患者的观测状态构建行为进展图，以反事实路径规划生成有同伴证据支撑、能保证健康指标单调改善的渐进式分步干预方案，克服了现有方法只给目标、缺少路径、缺乏同伴依据的缺陷。 |
| [^261] | [Feed the Panel Dimensions, Not Verdicts: Rubric-Decomposed Fusion of Vision-Language Aesthetic Judges](https://arxiv.org/abs/2609.27110) | 由多个视觉语言模型组成的评审团在融合整体美学判定时无法显著超越最佳单模型，而让各模型按人工评分准则对图像进行五维度评分并融合这些分数，则能可靠地击败最佳单个模型。 |
| [^262] | [PROSWIN: Probabilistic Solar Wind Speed Forecasting Using Deep Distributional Regression From Solar Images](https://arxiv.org/abs/2609.26683) | PROSWIN是一个结合太阳图像、磁图与深度分布回归的概率机器学习模型，可提前四天预报地球太阳风速度并实现良好的不确定性校准，同时引入预测分数指标以更好地评估高速太阳风流峰值。 |
| [^263] | [RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking](https://arxiv.org/abs/2609.23457) | 本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。 |
| [^264] | [Counterexamples and Sufficient Conditions: Comments on "Optimally-Transported Generalized Method of Moments"](https://arxiv.org/abs/2609.20260) | 本文对Schennach & Starck提出的最优传输广义矩方法（OTGMM）估计量的核心定理给出反例，证明其定理2-6在原假设条件下不成立，并探讨使这些结论成立的充分条件。 |
| [^265] | [The Token Before the Value Is the Key: How Hybrid Architectures Organize Induction Circuits](https://arxiv.org/abs/2609.15545) | 该论文发现混合语言模型中归纳电路存在明确分工——“携带前驱信息”集中于局部/循环等高效层而“内容匹配”集中于全局层，且通过干预前驱支持（如滞后一掩码、移除卷积等）可以在不同层之间重新分配这些功能。 |
| [^266] | [DynSHAP: Towards Explainable Dynamic Survival Analysis](https://arxiv.org/abs/2609.13042) | DynSHAP是一个专为动态生存分析设计的SHAP可解释性框架，通过将时间-特征对作为Shapley博弈的参与者，并引入利用条件采样处理特征时间依赖关系的Temporal DynSHAP，从而能够准确解释纵向不规则输入与函数型生存输出。 |
| [^267] | [Speed Limit for Information Acquisition in Stochastic Learning Dynamics](https://arxiv.org/abs/2609.08219) | 本研究将随机梯度下降建模为马尔可夫随机过程，推导出费希尔信息流的速度极限，定量约束了可训练参数获取数据中潜在变量信息的速率，并将信息流分解为漂移（确定性学习力）与噪声（SGD涨落）两部分贡献。 |
| [^268] | [Deep Barycentric Regression for Optimal Transport Map Estimation and its Statistical Optimality](https://arxiv.org/abs/2609.06598) | 该论文提出了一种简单的两步方法BROT：先计算未正则化的最优传输方案，再用深度神经网络对所得重心目标进行最小二乘回归，从而实现了兼具统计最优保证与实用可扩展性的最优传输映射估计。 |
| [^269] | [LeanStream: A Speculate-and-Refine Streaming Framework for Efficient on-Device LLM Inference](https://arxiv.org/abs/2609.03079) | LeanStream通过利用部分GPU计算结果渐进精化计算、加载和缓存保留优先级，实现了GPU执行与存储I/O的细粒度重叠，突破了端侧LLM推理中准确决策与计算-I/O重叠难以兼得的根本性权衡。 |
| [^270] | [Geometry-Constrained Kolmogorov-Arnold Networks: Learning Edge Geometry via Banach Duality](https://arxiv.org/abs/2608.25807) | 本文提出了一种通过每条边学习标量指数 $p$ 来动态调整激活函数几何结构的KAN变体，在符号回归任务中优于固定基函数方法。 |
| [^271] | [Trust the Mass: Forced Weights in KV-Cache Eviction](https://arxiv.org/abs/2608.25230) | 本文发现KV缓存驱逐中保留最大权重已接近最优，已发布方法间的差异主要源于存储方式而非选择策略，并揭示了评估中的内存与性能权衡。 |
| [^272] | [More accurate behavioral predictions with hybrid Bayesian-connectionist models](https://arxiv.org/abs/2608.22154) | 提出了结合行为微调的贝叶斯蒸馏（BBT）混合建模范式：先让神经网络通过合成数据模仿贝叶斯模型，再在人类行为数据上微调，在人类概念学习的行为预测中超越了传统方法。 |
| [^273] | [FAR-DPO: Feasibility-Aware and Robust Direct Preference Optimization for Cyclic Peptide Design](https://arxiv.org/abs/2608.19808) | FAR-DPO通过结合可行性感知偏好构建与难度感知的鲁棒优化，提出了一种架构无关的框架，显著提高了环肽设计在挑战性靶标上的可行性和多目标控制能力。 |
| [^274] | [A Locally Tokenized Generative Model for Robust Time-Series Watermarking](https://arxiv.org/abs/2608.19727) | 本文提出了一种局部令牌化生成模型L-VQVAE和配套水印方法LVQMark，通过确保每个恢复单元仅依赖有界时间邻域，解决了时间序列水印在后编辑攻击下的检测可靠性问题。 |
| [^275] | [Continuous Adversarial MeanFlow Transfer](https://arxiv.org/abs/2608.19540) | 该论文提出MeanFlow-Transfer方法，通过统一速度表示和对抗训练，同时解决跨领域适应和生成器加速问题，使异构预训练模型能高效迁移到新领域。 |
| [^276] | [Inference and Uncertainty Quantification for Streaming $r$-PCA](https://arxiv.org/abs/2608.18374) | 本文通过Oja算法解决了流式PCA的尖锐收敛速率和分布推断问题，首次在一般秩下实现无余项的收敛分析，并建立匹配的极小极大下界。 |
| [^277] | [GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction](https://arxiv.org/abs/2608.18234) | 本文提出了首个行为世界模型GigaBrain-WBC-0.5，通过因果Transformer联合预测动作、状态和潜在行为命令，使机器人能够建模环境交互，实现鲁棒的全身控制。 |
| [^278] | [Designing Reinforcement Learning for Diffusion Models: A Unified Path-Space View](https://arxiv.org/abs/2608.14430) | 本文通过统一路径空间原则，揭示了扩散模型RL中反向轨迹与前向匹配方法的共同起源，并推导出方差缩减的等效形式，弥合了方法间的经验差距。 |
| [^279] | [CytoBERT: A Foundation Model for Cytometry Data](https://arxiv.org/abs/2608.14414) | CytoBERT是一个针对异质细胞计数数据的开源基座模型，通过自监督预训练解决了跨研究迁移难题。 |
| [^280] | [Parameter Exploration for RLVR via Variational Learning](https://arxiv.org/abs/2608.09805) | 本文提出了一族名为扰动参数策略优化（3PO）的方法，通过变分学习从后验分布中采样多样化策略，实现参数空间探索，弥补了温度缩放等动作空间探索方法的局限，从而提升LLM强化学习的训练稳定性与下游性能。 |
| [^281] | [Population-Scalable Multi-Agent World Modeling](https://arxiv.org/abs/2608.08600) | 本文提出Khora，一种可扩展的多智能体世界模型，通过解耦世界状态演化与视觉渲染，实现推理时无需重新训练即可扩展到任意数量的智能体。 |
| [^282] | [From Siloed Algorithms to Compliance-First Agentic Platforms: A Multi-Layered Architecture for Hospital AI Systems](https://arxiv.org/abs/2608.06112) | 该研究提出了一种合规优先的医院智能体AI多层架构，通过智能体编排层和以“策略即代码”为核心的合规与策略层，解决医疗AI部署孤立、治理缺失和难以规模化推广的问题。 |
| [^283] | [Effective pruning of task-trained recurrent neural networks using noisy fluctuations and connection rescaling](https://arxiv.org/abs/2608.05464) | 该研究将noise-prune剪枝规则成功应用于任务训练的循环神经网络，通过基于噪声波动对连接重要性进行采样并重缩放保留连接的强度来保持任务性能，其效果优于基于权重大小和二阶信息的剪枝策略。 |
| [^284] | [Variable-aware representation, not relation typing, explains the performance of an event-graph model for ICU mortality prediction: temporal and external validation](https://arxiv.org/abs/2608.02663) | 该研究通过严格的时间性与外部多中心验证发现，事件图模型在ICU死亡率预测中的性能提升主要来自变量感知表示（变量学习嵌入加逐变量归一化），而非关系类型化的消息传递。 |
| [^285] | [Twig: Parallel and Memory-Efficient Tree Speculative Decoding for Hybrid-Attention LLMs](https://arxiv.org/abs/2608.01651) | Twig通过对线性注意力递归进行树结构化因子分解并配合并行GPU内核，实现了混合注意力大语言模型的高效树形推测解码，将树验证加速3.4至7.7倍，同时将瞬态状态内存降低82至99倍。 |
| [^286] | [GPTQ-2D: Cubic-Time Two-Sided Adaptive Rounding](https://arxiv.org/abs/2607.27042) | GPTQ-2D提出了一种可证明以立方计算量实现与YAQA相同的双侧自适应舍入效果的高效大模型量化算法，将Llama-3.1-70B在单块GPU上的量化时间从8.6天大幅缩短至1.6小时。 |
| [^287] | [Invariant Discovery for Networked Systems](https://arxiv.org/abs/2607.22944) | 本文提出 Autogram，通过让 LLM 生成紧凑的候选不变量文法并在其内进行确定性统计搜索、再迭代扩展文法直至恢复已知可信不变量，实现了无需人工文法且能容忍真实噪声的网络系统不变量自动发现。 |
| [^288] | [Mark, Don't Erase: Token Inoculation for Dual-Use Knowledge in LLMs](https://arxiv.org/abs/2607.18639) | 提出“Token接种”方法，在继续预训练时用特殊标记标注危险句子并训练模型条件化地预测危险答案，测试时因标记缺失而抑制危险输出，从而在保持93%良性性能的同时将危险准确率控制在约18%。 |
| [^289] | [TabPFN beyond Tabular Data: Calibration and Accuracy on Multimodal Embeddings](https://arxiv.org/abs/2607.11007) | 该研究发现TabPFN可直接用作冻结多模态编码器（图像、文本、音频）的零梯度分类头，在22,820个评估回合中其校准性能（NLL和ECE）在九种分类头中排名最佳，同时保持甚至超越基线精度。 |
| [^290] | [Integrating Background Knowledge for Scalable Causal Discovery](https://arxiv.org/abs/2607.10456) | 该论文提出了一个在因果发现过程中（而非仅作为事后后处理）整合专家背景知识的框架，并针对仅恢复部分图结构的大规模可扩展因果发现方法进行了多算法实现与实验验证。 |
| [^291] | [Emergent Generalization by Representation Learning in Artificial Neural Networks](https://arxiv.org/abs/2607.10430) | 本文证明在循环神经网络中施加显式信息瓶颈以学习低维表征，是时间序列预测任务中实现旋转泛化与分布外泛化的必要条件，并通过因果涌现的信息论度量揭示了表征在记忆到泛化转变过程中的非单调动力学轨迹。 |
| [^292] | [RL Forgets! Towards Continual Policy Optimization](https://arxiv.org/abs/2607.04364) | 该论文提出MRCL多模态推理持续学习基准，证明标准强化学习在持续后训练中仍会发生灾难性遗忘，其根源在于KL正则化仅基于当前任务数据评估，而遗忘源自模型在旧任务分布上的行为漂移。 |
| [^293] | [HyperVAttention: Efficient Sparse Attention with Spatio-Temporal Clustering for Video Diffusion](https://arxiv.org/abs/2607.03012) | 提出HyperVAttention——一种无需训练的稀疏注意力框架，通过3D局部窗口聚类和自定义Triton内核，同时降低视频扩散Transformer的聚类开销并提升CTA利用率，实现高效的长视频生成加速。 |
| [^294] | [The State-Prediction Separation Hypothesis](https://arxiv.org/abs/2607.01218) | 该论文提出“状态-预测分离假设”，设计了一种用独立计算流将“预测下一个token”与“存储状态”两种功能解耦的 Transformer 变体，多规模预训练实验表明其具有更高的数据与计算效率，在下游任务上平均超越标准 Transformer 2-3 个百分点。 |
| [^295] | [Explicit 3D Gaussian Models for Fast, Mesh-Free Compression of Structured and Unstructured Volumes](https://arxiv.org/abs/2607.01164) | 该论文提出基于显式三维高斯基元的体数据压缩模型，通过对相交高斯进行加权聚合来重建标量场，实现了无需网格存储的结构化与非结构化体数据快速压缩。 |
| [^296] | [Set-Valued Routing for Language-Model Agent Systems: A WildChat Benchmark and Cost-Aware Evaluation](https://arxiv.org/abs/2606.28925) | 该论文将语言模型智能体系统的路由从单一top-1选择扩展为集合值路由，构建了基于WildChat的3000条提示、12个能力定义智能体的新基准，并通过成本感知评估证明微调文本编码器（89.59 F1）显著优于线性SVM和零样本GPT-4o。 |
| [^297] | [ThermoLLM: LLM-Based Multi-Zone HVAC Control with a Spatiotemporal Context](https://arxiv.org/abs/2606.22911) | 本文提出ThermoLLM，将建筑空间语义、区域热动力学与滚动历史作为LLM上下文的核心，从而实现更一致、更可泛化且能感知区域间延迟热交互的多区域HVAC控制。 |
| [^298] | [Diffusion-Driven State Space Models](https://arxiv.org/abs/2606.21036) | 本文提出扩散驱动的状态空间模型（DDSSM），以扩散模型替代传统高斯转移分布，解决了在序列数据上联合训练自编码器与扩散模型的开放问题，并在多峰转移时间序列的拟合与预测上超越现有最先进的深度状态空间模型。 |
| [^299] | [Volterra Generative Models via Markovian Lifting](https://arxiv.org/abs/2606.18071) | 本文提出通过马尔可夫提升方法构建Volterra生成模型，将非马尔可夫的Volterra过程转化为可处理的马尔可夫近似，从而支持重用预训练去噪器，并在MNIST、CIFAR-10和ImageNet基准上取得了优异的FID性能。 |
| [^300] | [LatentGym: A Testbed For Cross-Task Experiential Learning With Controllable Latent Structure](https://arxiv.org/abs/2606.15306) | 提出了LatentGym测试平台，通过围绕真值潜在变量构建可控环境，实现了对智能体跨任务经验学习能力的度量，并可将探索与利用行为分开评估。 |
| [^301] | [Recursively Trained Diffusion Models: Limiting Collapse Distribution and Spectral Characterization](https://arxiv.org/abs/2606.13796) | 该论文证明，即使在评分估计与采样完全精确的理想条件下，反向扩散的提前停止也会使递归自训练的扩散模型以几何速率收敛到一个唯一的极限分布（即数据分布的 increasingly 高斯平滑版本的无限混合），且该过程如同低通滤波器般滤除高频信息，从而完全刻画了模型坍缩的极限分布与收敛速度。 |
| [^302] | [Dense Supervision, Sparse Updates: On the Sparsity and Geometry of On-Policy Distillation](https://arxiv.org/abs/2606.13657) | 本文系统刻画了在策略蒸馏参数更新的稀疏性与几何结构，发现其更新幅度小且坐标稀疏，仅在发现的关键子网络上训练即可几乎恢复完整性能，且更新避开源模型主结构方向、偏向低幅值坐标。 |
| [^303] | [The Standard Interpretable Model: A general theory of interpretable machine learning to deductively design interpretable methods using Lagrangian mechanics](https://arxiv.org/abs/2606.12289) | 本文提出了一种基于拉格朗日力学的标准可解释模型（SIM）理论，通过演绎方式设计可解释方法，填补了可解释性理论与方法之间的空白。 |
| [^304] | [When Attribution Patching Lies: Diagnosis and a Second-Order Correction](https://arxiv.org/abs/2606.09899) | 本文揭示归因修补的主要误差源于下游网络的非线性而非被修补组件的局部曲率，并据此提出可靠性分数、误差界限与二阶修正三项工具来诊断和改进归因修补的可信度。 |
| [^305] | [Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation](https://arxiv.org/abs/2606.09864) | 研究发现KV缓存低比特量化会在困惑度等常规指标不变的情况下隐性地破坏大语言模型的安全对齐，其根源在于安全特征位于对量化噪声极其敏感的低维子空间，据此提出的PCR诊断方法可用于识别并缓解此类失效。 |
| [^306] | [Querying Counterfactuals on Tissue Graphs with Supervised Disentanglement](https://arxiv.org/abs/2606.08493) | 该论文提出Cellina框架，通过监督解耦将细胞的内在状态与空间上下文分离，并首次将组织图反事实形式化为边扰动和节点扰动两类空间干预，在覆盖超过250万个细胞的基准测试中超越了现有空间与非空间方法。 |
| [^307] | [How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity? Capabilities and Boundaries in Multi-Genre Chord-Symbol Modeling](https://arxiv.org/abs/2606.07334) | 该研究将和弦符号序列建模为可解释、可控的时间序列，系统比较 LoRA、IA3、BitFit、前缀微调与全量微调五种方法在 11 个音乐流派上的适配能力与边界，发现所有方法均能提升和弦预测准确率，但序列头部词元会显著夸大评测分数，揭示了该建模方式的能力与局限。 |
| [^308] | [Autoregressive Diffusion World Models for Off-Policy Evaluation of LLM Agents](https://arxiv.org/abs/2606.05558) | 提出自回归扩散世界模型ADWM，通过潜在扩散模拟环境响应，仅凭预先收集的轨迹即可离线评估LLM智能体策略性能，避免了昂贵的在线交互并克服了自回归世界模型的复合误差问题。 |
| [^309] | [Near-Optimal Decentralized Stochastic Convex Optimization over Networks](https://arxiv.org/abs/2606.04757) | 提出一种基于单步延迟随机加速机制的去中心化优化方法，将保持中心化 $O(1/\sqrt N)$ 统计速率的最大工作者数量从原有的 $\rho\sqrt N$ 提升至 $\sqrt{\rho}\,N^{3/4}$，达到近最优的规模扩展。 |
| [^310] | [Murmur: Efficient Long-Context ASR Inference with Hierarchical Chunking](https://arxiv.org/abs/2606.01483) | 提出了基于分层分块的长篇ASR推理系统Murmur，发现约五分钟的中间块大小可在避免重复循环的同时保持单次处理的准确率，并通过并行解码兼顾高准确率与低推理延迟。 |
| [^311] | [ForecastCompass: Guiding Agentic Forecasting with Adaptive Factor Memory](https://arxiv.org/abs/2605.30858) | 提出了ForecastCompass（FoCo），一个面向智能体预测的自适应因子记忆框架，通过分层任务分类体系组织预测经验，并维护因子记忆与校准知识两类互补记忆，实现可复用预测知识的检索，从而提升动态环境下的概率预测能力。 |
| [^312] | [Knowledge Offloading: Decomposing LLMs into Sparse Backbones and Memory Modules](https://arxiv.org/abs/2605.29075) | 该论文提出知识卸载（KOFF）框架，通过结构化剪枝掩码与LoRA适配器、可学习KV缓存等轻量恢复模块，将预训练大语言模型分解为稀疏共享骨干网络和领域特定记忆模块，在保持约90%性能的同时可将12%-20%的容量从骨干网络中移出。 |
| [^313] | [Teach the Way It Thinks: Guiding LLM Post-training Data Engineering with Sparse Autoencoders](https://arxiv.org/abs/2605.27354) | 该论文提出SAERL方法，利用稀疏自编码器（SAE）将模型内部分解为稀疏可解释的特征，从模型内部视角（激活与梯度投影）来组织强化学习训练数据，从而决定训练哪些样本、何时呈现以及如何分批，突破了传统仅依赖通过率等外部反馈的数据工程方式。 |
| [^314] | [Everything at Every Scale: Scale-Invariant Diffusion with Continuous Super-Resolution](https://arxiv.org/abs/2605.26032) | 提出尺度不变的扩散模型 SKILD，将图像生成与连续超分辨率统一在单一无条件框架中，仅通过改变起始时间步即可在两种任务间切换。 |
| [^315] | [The Quantization Benefits of Residual-Free Transformers](https://arxiv.org/abs/2605.25880) | 本文揭示了残差连接会在训练中放大Transformer激活值的非高斯性（重尾、离群值主导），从而加剧低比特量化的误差与精度损失，而无残差Transformer凭借稠密混合收缩非高斯性，在量化部署中具有天然优势。 |
| [^316] | [Visual-Redundancy-Controlled Parallel Decoding for Diffusion-Based Multimodal Large Language Models](https://arxiv.org/abs/2605.25820) | 该论文发现扩散多模态大语言模型并行解码中同一步骤内被提交的高置信度token可能依赖重叠的视觉定位信息而产生视觉冗余，并提出视觉冗余指数（VRI）来量化这一效应以改进解码策略。 |
| [^317] | [LAPLEX: The FFT of Learnable Laplace Kernels](https://arxiv.org/abs/2605.24584) | LAPLEX提出了一类精确且可训练的拉普拉斯核算子层，以FFT式的扩展效率实现了可学习的自适应几何结构，支持高达10^9维度的可训练矩阵运算，并可高效用于高维协方差建模。 |
| [^318] | [VACE: Learning Geometrically Structured Representations for Time Series Anomaly Detection](https://arxiv.org/abs/2605.23504) | VACE提出了一种自监督异常检测方法，通过速度对齐通道嵌入显式塑造潜空间几何结构，将正常性表示为局部平滑、方向连贯的表示，克服了对比学习无法显式控制距离评分所依赖几何结构的缺陷。 |
| [^319] | [Vector Policy Optimization: Training for Diversity Improves Test-Time Search](https://arxiv.org/abs/2605.22817) | 提出向量策略优化（VPO），通过利用向量值奖励、多答案提示与集合级奖励来训练大语言模型生成多样化的解决方案集合，从而提升推理时搜索的效果。 |
| [^320] | [ASAP: Attention Sink Anchored Pruning](https://arxiv.org/abs/2605.22372) | ASAP 提出了一种无需训练的 ViT token 剪枝框架，将注意力汇聚现象建模为惰性随机游走中的概率累积特征，通过径向扩散聚类与转移权重池化一次性压缩背景冗余 token。 |
| [^321] | [Your Neighbors Know: Leveraging Local Neighborhoods for Backdoor Detection in Decentralized Learning](https://arxiv.org/abs/2605.19969) | 提出了去中心化学习原生的新型后门检测框架Argus，无需中央协调者或触发器先验知识，通过节点本地分析模型更新、与邻居共享触发器并利用结构相似性度量，有效区分真实后门攻击与数据异质性导致的误报。 |
| [^322] | [KVBuffer: IO-aware Serving for Linear Attention](https://arxiv.org/abs/2605.19049) | KVBuffer是一种面向线性注意力的IO感知服务机制，通过缓冲最近的键和值实现分块计算与批量状态更新，显著减少内存访问量并降低解码延迟，同时高效支持投机解码验证。 |
| [^323] | [An Assessment of Human vs. Model Uncertainty in Soft-Label Learning and Calibration](https://arxiv.org/abs/2605.18648) | 本研究通过解耦软标签监督与标签模态偏移，发现人类软标签的主要价值在于作为正则化器提升模型对困难样本的校准能力和训练收敛稳定性，并使模型的不确定性表现与人类一致。 |
| [^324] | [Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete Uncertainty](https://arxiv.org/abs/2605.14494) | 本文提出了问题驱动的场景缩减启发式算法 PRISE 及其基于 GNN-Transformer 的神经代理模型 NeurPRISE，能够高效构建高质量场景子集，显著提升离散不确定性两阶段鲁棒优化的可扩展求解能力。 |
| [^325] | [CSI-JEPA: Towards Foundation Representations for Ubiquitous Wi-Fi Sensing with Minimal Supervision](https://arxiv.org/abs/2605.14171) | CSI-JEPA是一个自监督预测表征学习框架，通过时间-子载波联合分词和信道变化感知掩码策略，从未标注CSI中学习可迁移的Wi-Fi感知表征，大幅降低对标注数据的依赖。 |
| [^326] | [Characterizing universal object representations across vision models](https://arxiv.org/abs/2605.13675) | 通过将162个视觉模型的物体相似性结构分解为非负维度，研究发现通用维度更具可解释性且由概念性图像属性驱动，表明可解释性和语义内容是驱动模型间表征通用性的隐含关键因素。 |
| [^327] | [Persona-Model Collapse in Emergent Misalignment](https://arxiv.org/abs/2605.12850) | 该论文提出“人格模型坍缩”假说，指出有害数据微调引发的涌现性失调源于模型内部人格表征的退化，并在四个模型家族上通过问卷实验证实有害微调使人格表达更极化、更二元化且内部一致性降低。 |
| [^328] | [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938) | 提出在连续嵌入空间中基于流匹配的扩散语言模型 ELF，其直到最终时间步才映射到离散词元，从而让图像扩散领域的成熟技术（如 CFG）可直接应用于语言建模。 |
| [^329] | [Minimal Filling Architectures for Polynomial Neural Networks](https://arxiv.org/abs/2605.09609) | 本文推翻了“所有最小填充架构的隐藏层宽度均为单峰”的猜想，通过前沿搜索和有限域计算构造并验证了反例，同时证明了深度不超过三的最小填充架构必为单峰，而深度为四的非单峰反例表明该界限是最优的。 |
| [^330] | [Fast Rates for Offline Contextual Bandits with Forward-KL Regularization under Single-Policy Concentrability](https://arxiv.org/abs/2605.09214) | 本文首次在单策略集中性条件下，为前向KL正则化的离线上下文老虎机给出了表格型与一般函数逼近设置下的 Õ(ε^{-1}) 快速收敛率上界，通过新颖地运用悲观原理的凸分析框架填补了前向KL正则化统计分析的空白。 |
| [^331] | [Tight Generalization Bounds for Noiseless Inverse Optimization](https://arxiv.org/abs/2605.08866) | 本文为无噪声逆向优化给出了紧致的高概率 O(d/T) 泛化界，在动作唯一性条件下将其与老虎机最优臂识别结果对齐，并证明瞬时与累积遗憾的下界与对抗性设置的上界相匹配，从而表明该速率不可再改进。 |
| [^332] | [FILO: A Unified Fisher-Geometric Framework for Low-Rank Adaptation](https://arxiv.org/abs/2605.08734) | 本文提出FILO，一个统一的Fisher几何框架，无需对奇异算子直接求逆即可实现LoRA的因子空间更新，并据此将现有LoRA优化器系统地划分为四大类。 |
| [^333] | [Almost Sure Convergence Rates of Stochastic Approximation and Reinforcement Learning via a Poisson-Moreau Drift](https://arxiv.org/abs/2605.07104) | 本文通过泊松-莫罗漂移技术，为马尔可夫噪声下期望更新具有压缩性的随机逼近算法（如Q学习和线性时序差分学习）建立了几乎必然收敛速率，在幂律学习率下获得了可任意接近o(n^(1-2η))的速率，在调和学习率下当参数满足条件时平方误差指数可任意接近1。 |
| [^334] | [UniSD: Towards a Unified Self-Distillation Framework for Large Language Models](https://arxiv.org/abs/2605.06597) | 提出了统一自蒸馏框架UniSD，通过整合多教师一致性、EMA教师稳定化、词元级对比学习、特征匹配和散度裁剪等互补机制，系统性地解决了大语言模型自蒸馏中监督可靠性、表示对齐与训练稳定性的核心难题。 |
| [^335] | [Invariant Features in Language Models: Geometric Characterization and Model Attribution](https://arxiv.org/abs/2605.06458) | 该论文提出一个局部几何框架来刻画语言模型中的不变特征，通过对比子空间发现方法分离语义改变型与语义保持型的表示变化，并将所得的不变表示成功应用于零样本模型归因任务。 |
| [^336] | [Taming the Entropy Cliff: Variable Codebook Size Quantization for Autoregressive Visual Generation](https://arxiv.org/abs/2605.06207) | 该论文发现离散视觉分词器中存在条件熵沿序列急剧衰减的“熵悬崖”现象，并提出可变码本大小量化（VCQ）方法，以解决恒定码本快速耗尽有限数据熵预算的问题。 |
| [^337] | [Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes](https://arxiv.org/abs/2605.06152) | 本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。 |
| [^338] | [Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation](https://arxiv.org/abs/2605.04998) | 本研究通过对流行预训练Music Transformer进行爵士微调并系统扫描混合的流行“排练”数据量，实证发现排练数据可将爵士top-1提升7至9个百分点，同时将流行风格的遗忘从2.11个百分点降至最多1.19个百分点，为小数据新风格的适应提供了混合比例依据。 |
| [^339] | [Prescriptive Scaling Laws for Data Constrained Training](https://arxiv.org/abs/2605.01640) | 该论文提出了一种带加性过拟合惩罚项的规范性缩放定律，能准确刻画数据受限训练中重复数据的影响，并给出新的计算最优分配建议——当数据重复超过一定限度后，算力应投向扩大模型容量而非继续重复训练。 |
| [^340] | [Hyper Input Convex Neural Networks for Shape Constrained Learning and Optimal Transport](https://arxiv.org/abs/2604.26942) | HyCNN 结合 Maxout 网络与输入凸神经网络，以指数级更少的参数实现高效凸函数学习，在凸回归、插值以及高维最优传输任务中显著优于现有方法。 |
| [^341] | [Learning to Think from Multiple Thinkers: How Chain-of-Thought Heterogeneity Affects Efficient Learnability](https://arxiv.org/abs/2604.24737) | 该论文首次从学习理论视角研究异构思维链监督问题，提出多种由不同来源（最终答案相同但中间步骤各异）提供CoT数据的异构监督模型，并探讨在单一同构来源可被高效学习的前提下，何时能够从异构监督数据中实现高效学习。 |
| [^342] | [AAC: Admissible-by-Architecture Differentiable Landmark Compression for ALT](https://arxiv.org/abs/2604.20744) | 本文提出AAC，在固定标签内存预算下通过两种可采纳性保持的连续松弛实现可微分的ALT地标选择压缩，并给出了带有1-1/e极限因子的近似保证。 |
| [^343] | [Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models](https://arxiv.org/abs/2604.16565) | 该论文提出了一种无需训练、无监督的双向流形一致性（BMC）度量方法，通过前向掩码与后向重建循环量化生成序列的几何稳定性，从而在无标准答案的情况下验证扩散语言模型推理的有效性，并支持诊断与拒绝重采样等应用。 |
| [^344] | [Stability and Generalization in Looped Transformers](https://arxiv.org/abs/2604.15259) | 该论文提出基于不动点的三维度稳定性分析框架，证明带外部归一化召回机制的循环Transformer能同时实现不动点的局部吸引性、输入平滑性与稳定的反向传播，从而在测试时对更难的问题实现外推而非记忆。 |
| [^345] | [Selectivity and Shape in the Design of Forward-Forward Goodness Functions](https://arxiv.org/abs/2604.13081) | 本文系统研究了前向-前向算法中优良度函数的选择性与形状设计，提出了一种无参数、尺度不变的“突发性优良度”目标函数（基于超额峰度、奖励重尾活动模式），在Fashion-MNIST上显著提升了准确率。 |
| [^346] | [Nexus: Same Pretraining Loss, Better Downstream Generalization via Common Minima](https://arxiv.org/abs/2604.09258) | 论文提出Nexus优化器，通过在训练中最大化梯度相似性，促使模型收敛到各数据源的共同极小值点，从而在预训练损失相同的情况下显著提升下游泛化能力。 |
| [^347] | [Zero-Shot Quantization via Weight-Space Arithmetic](https://arxiv.org/abs/2604.03420) | 该论文提出通过简单的权重空间算术从供体任务中提取“量化向量”来修补接收模型，从而在不进行量化感知训练、不使用接收端验证数据的零样本场景下，将3比特量化的Top-1准确率提升最多60%。 |
| [^348] | [Survival In-Context: Amortized Bayesian Survival Analysis via Prior-Fitted Networks](https://arxiv.org/abs/2603.29475) | 本文提出SIC模型，通过将先验拟合网络在合成生存数据上预训练以摊销贝叶斯后验预测推断，实现了仅凭单次前向传播即可完成个体化生存预测的上下文学习生存分析方法。 |
| [^349] | [Does Explanation Correctness Matter? Linking Computational XAI Evaluation to Human Understanding](https://arxiv.org/abs/2603.25251) | 该研究通过200人用户实验，在四个受控正确性水平（100%、85%、70%、55%）下首次实证检验了解释正确性与人类理解之间的联系，发现解释的功能正确性确实会影响人类对AI决策的前向模拟准确率，从而为计算化XAI评估指标与人类理解之间建立了关联。 |
| [^350] | [One Method Does Not Fit All: A Large-Scale Benchmark of Imputation Methods for Single-Cell RNA Sequencing Data](https://arxiv.org/abs/2603.24626) | 本文提出了首个大规模标准化基准测试框架，在10种实验方案、6个下游任务和30个数据集上系统评估了16种单细胞RNA测序数据填补方法，结果表明没有一种方法能够适用于所有场景。 |
| [^351] | [Scaling Recurrence-aware Foundation Models for Clinical Records via Next-Visit Prediction](https://arxiv.org/abs/2603.24562) | 本文提出RAVEN——一种基于复发感知的下次就诊事件预测的临床记录基础模型预训练策略，揭示了重复事件token虚高评估指标的陷阱，并证明在数据受限时单纯扩大模型规模并非最优。 |
| [^352] | [VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining](https://arxiv.org/abs/2603.23481) | VTAM提出了一种无需触觉或动作预训练的高效视觉-触觉世界-动作模型，通过共享冻结的视频VAE并联合预测未来视觉与触觉潜在表示，在富接触操作的真实机器人任务上超越了所有基线。 |
| [^353] | [Neuronal Self-Adaptation Enhances Capacity and Robustness of Representation in Spiking Neural Networks](https://arxiv.org/abs/2603.20687) | 受生物钾通道动态自调节机制启发，提出KvLIF神经元模型，通过辅助电导状态自适应调节神经元兴奋性与复位动力学，显著提升脉冲神经网络的信息容量和抗噪鲁棒性。 |
| [^354] | [CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing](https://arxiv.org/abs/2603.19297) | 本文提出CLaRE，一种轻量级的表征层面技术，仅利用单个中间层的前向激活（无需昂贵的反向传播）量化事实之间的纠缠程度，从而预测大语言模型编辑中涟漪效应可能发生的位置。 |
| [^355] | [Preference-Guided Adversarial Policy Optimization for Long-Tail Robust Driving](https://arxiv.org/abs/2603.15221) | ADV-0提出一个闭环极小极大优化框架，将驾驶策略与对抗智能体的交互建模为零和马尔可夫博弈，并通过迭代偏好学习高效逼近最优对抗分布，从而提升自动驾驶系统对罕见但安全关键的长尾场景的鲁棒性。 |
| [^356] | [FastWave: Optimized Diffusion Model for Audio Super-Resolution](https://arxiv.org/abs/2603.04122) | FastWave是一个轻量级的扩散模型（约50 GFLOPs计算量、130万参数），用于将任意采样率的音频超分辨率提升至48 kHz，其性能优于NU-Wave 2并与最先进模型相当，同时训练所需的资源和时间显著更少。 |
| [^357] | [DUEL: Exact Likelihood Reshapes Masked Diffusion Scaling Laws and Beyond](https://arxiv.org/abs/2603.01367) | 本文提出DUEL框架，能够在掩码扩散模型实际使用的确定性采样器下精确计算其似然，无需重新训练即可即插即用，并由此重塑了关于掩码扩散模型缩放定律等一系列结论。 |
| [^358] | [The Stability of Online Algorithms in Performative Prediction](https://arxiv.org/abs/2602.24207) | 本文证明部署在表演性预测环境中的任何无悔在线算法都会无条件收敛到（混合）表演性稳定均衡，且无需对人群如何响应预测做任何假设。 |
| [^359] | [How Far Can a Trainable Frequency Move? Resolution, Grids and Coupling in Trainable-FrequencyQuantum Models](https://arxiv.org/abs/2602.23409) | 该论文揭示了可训练频率量子模型中前置因子梯度的解析结构——由 sin u/u 重叠函数斜率的加权和决定每个频率的可达范围，并提出三值前置因子网格方案，仅用对数数量的编码门即可将模型频率逼近所有频带频率至半个步长以内。 |
| [^360] | [Tell Me What To Learn: Generalizing Neural Memory to be Controllable in Natural Language](https://arxiv.org/abs/2602.23201) | 提出一种可用自然语言指令控制的广义神经记忆系统，使模型能够依据用户指令从异构信息源中选择性地学习，突破了传统神经记忆固定目标的局限。 |
| [^361] | [Oracle-Robust Online Alignment for Large Language Models](https://arxiv.org/abs/2602.20457) | 本文针对偏好反馈预言机可能存在偏差的大语言模型在线对齐问题，提出最坏情况鲁棒优化目标，并证明该鲁棒目标可精确分解为原始损失加显式敏感度惩罚项，进而设计了投影随机复合更新算法进行求解。 |
| [^362] | [Online Conformal Prediction via Universal Portfolio Algorithms](https://arxiv.org/abs/2602.03168) | 本文识别出“线性化遗憾”这一关键概念并建立了通用的“遗憾到覆盖”理论，据此提出无需任何参数调整的UP-OCP方法，通过将问题归约为两资产投资组合选择问题并利用通用投资组合算法，实现了参数无关的在线保形预测覆盖率保证。 |
| [^363] | [Hyperbolic Graph Neural Networks Under the Microscope: The Role of Geometry-Task Alignment](https://arxiv.org/abs/2602.01828) | 这项工作提出“几何-任务对齐”这一关键条件，通过实验和理论分析表明，双曲图神经网络只有在任务的度量结构与输入图的几何结构相匹配时才能真正获得优势。 |
| [^364] | [Reconsidering Positional Supervision in Masked Diffusion Language Model Training](https://arxiv.org/abs/2601.22947) | 该论文通过受控干预实验发现掩码扩散语言模型在迭代并行解码下对微小的位置偏移极为敏感（仅偏移1%的token即显著降低胜率），并据此提出改造连接主义时间分类（CTC）以改进位置监督。 |
| [^365] | [Differentiable Knapsack and Top-k Operators via Dynamic Programming](https://arxiv.org/abs/2601.21775) | 本文提出一个统一框架，将背包与Top-k算子转化为动态规划并平滑递推得到可微分松弛，配有高效并行前向与反向算法，并从理论上证明香农负熵是唯一保证置换等变性的可分离正则化。 |
| [^366] | [Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes](https://arxiv.org/abs/2601.21629) | 该论文提出训练强化学习智能体来自适应地为量子电路组合优化通道序列，在双量子比特门削减效果上显著优于 PyTKET 默认通道序列（平均 57.7% 对比 41.8%）。 |
| [^367] | [LAMP: Look-Ahead Mixed-Precision Inference of Large Language Models](https://arxiv.org/abs/2601.21623) | 该论文提出LAMP方法，基于舍入误差分析构建自适应混合精度策略，只需对Transformer中极小部分组件进行高精度重计算，即可使大语言模型推理精度提升高达两个数量级。 |
| [^368] | [Depth-Recurrent Attention Mixture (Dreamer): Giving Latent Reasoning the Attention it Deserves](https://arxiv.org/abs/2601.21582) | 该论文提出Dreamer模块化框架，通过结合序列注意力、深度注意力和稀疏专家注意力来突破恒定隐藏维度的瓶颈，使深度循环模型在语言推理基准上以2-8倍更少的训练token达到与SOTA相当的精度，并优于约2倍规模的SOTA模型。 |
| [^369] | [Self-Improving Training & Thinking](https://arxiv.org/abs/2601.21343) | 该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。 |
| [^370] | [Scaling laws for amplitude surrogates](https://arxiv.org/abs/2601.13308) | 该论文系统研究了粒子物理振幅代理模型的缩放定律，发现缩放系数与过程的外部粒子数量相关，并证明缩放定律是实现所需精度目标的有效工具。 |
| [^371] | [EmboTeam: Asymmetric State Grounding for Heterogeneous Multi-Robot Collaboration](https://arxiv.org/abs/2601.11063) | 提出 EmboTeam 神经符号框架，通过 LLM 生成结构化状态预期、PDDL 符号搜索验证协调，并落地为带局部重规划与黑板同步的团队级并行行为树，实现异构多机器人长时程协作中的非对称状态接地，显著提升执行鲁棒性。 |
| [^372] | [Pareto DNN Verification: Fast for Most Queries](https://arxiv.org/abs/2512.20755) | 提出PAVE方法，通过为DNN添加早期退出机制并逐退出点验证，能以直接验证所需时间的一小部分解决约80%的鲁棒性查询，且准确率变化不超过0.3%。 |
| [^373] | [Corrective Diffusion Language Models](https://arxiv.org/abs/2512.15596) | 该论文提出一种面向纠错的后训练原则，通过显式监督可见的损坏词元，使掩码扩散语言模型能够对错误词元给出低置信度并进行有针对性的迭代修正，同时保留正确内容，其效果显著优于传统的掩码扩散微调。 |
| [^374] | [Greedy Alignment Principle for Optimizer Hyperparameter Selection](https://arxiv.org/abs/2512.06370) | 该论文提出了贪心对齐原理，将优化器选择形式化为最大化损失期望下降速率（等价于优化器滤波器与梯度自相关的内积），证明了贪心最优解的存在性与稳定性，并据此为 SGD+Momentum 和 Adam/AdamW 导出了简单的动态动量选择规则。 |
| [^375] | [Variational Estimators for Node Popularity Models](https://arxiv.org/abs/2511.17783) | 本文为双向节点流行度模型（TNPM）开发了一种计算高效且有理论保证的变分期望最大化（VEM）估计框架，在二部网络中证明了社区划分的标签一致性，并实现了优于现有方法的估计精度。 |
| [^376] | [CCMP: Class-Conditional Multicollinearity Pruning for High-Dimensional Feature Selection](https://arxiv.org/abs/2511.12451) | 提出类条件多重共线性剪枝方法CCMP，按类别迭代地从高度相关特征对中剪除冗余特征，并给出剪枝误差的明确理论界限，从而实现无需整体重拟合的高维特征选择。 |
| [^377] | [Multi-objective Hyperparameter Optimization with Expert Priors](https://arxiv.org/abs/2511.08371) | 提出首个能将专家先验知识融入多目标超参数优化的算法PriMO，在8个深度学习基准以及大规模视觉-语言模型的多目标优化中均达到最先进性能。 |
| [^378] | [Deep Neural Operator Learning for Probabilistic Models](https://arxiv.org/abs/2511.07235) | 该论文提出了一个适用于概率模型的深度神经算子框架，建立了具有显式网络规模界的通用逼近定理，并证明所学模型无需重新训练即可为新的执行价格生成美式期权的最优停止边界。 |
| [^379] | [Spectral Thresholds in Correlated Spiked Models and Fundamental Limits of Partial Least Squares](https://arxiv.org/abs/2510.17561) | 本文利用随机矩阵理论证明了部分对齐的尖峰互协方差模型中样本奇异值会经历BBP型相变，首次精确刻画了偏最小二乘法（PLS）的信号恢复能力，并揭示了PLS与贝叶斯最优估计器之间的根本性能差距，指出在某些信噪比和相关性区间内PLS会完全无法恢复信号。 |
| [^380] | [Instruction-Tuning Large Language Models with Medical Classification Data Improves ICD Coding of German Tumor Diagnoses](https://arxiv.org/abs/2510.13624) | 本研究基于ICD-10-GM、ICD-O-3和OPS医学分类目录构建了超过50万个问答对，对Qwen、Llama和Mistral系列开源大语言模型进行指令微调，从而显著提升了德语肿瘤诊断文本的ICD编码准确性。 |
| [^381] | [scTREAT: Latent Diffusion for Single-Cell Drug-Response Prediction](https://arxiv.org/abs/2510.11726) | scTREAT是首个将细胞状态与药物引导进行因式分解的潜在扩散框架，通过引导可分解注意力机制注入基线表示与剂量条件化的分子嵌入，在单细胞药物响应预测任务上取得了最先进的性能。 |
| [^382] | [ARMs: Adaptive Red-Teaming Agent against Multimodal Models with Plug-and-Play Attacks](https://arxiv.org/abs/2510.02677) | 提出ARMs自适应红队测试智能体，通过推理增强的多步骤编排自动优化攻击策略，包含11种新颖多模态攻击手段并可即插即用集成17种红队测试算法，实现对视觉语言模型的全面自动化安全评估。 |
| [^383] | [HomID : Benchmarking Intrinsic Dimension Estimators on Homogenous Manifolds with Anisotropic Embeddings](https://arxiv.org/abs/2510.01335) | 本文提出了 HomID 基准——一组具有各向异性嵌入的同质流形集合，用以揭示现有内在维度估计器在标准基准上表现良好、但在各向异性条件下会系统性失效，并证明了各向异性畸变通过使其所依赖的分布发生系统性偏移而导致估计误差。 |
| [^384] | [Hybrid Dual-Batch and Cyclic Progressive Learning for Efficient Distributed Training of Convolutional Neural Networks](https://arxiv.org/abs/2509.26092) | 提出了一种基于参数服务器框架的双批次学习方案，通过同时使用硬件支持的最大批次和较小批次进行训练，在保持高效的同时显著提升模型的泛化能力和准确率。 |
| [^385] | [Machine Learning Algorithms for Improving Black Box Optimization Solvers](https://arxiv.org/abs/2509.25592) | 这是一篇综述论文，系统总结了机器学习与强化学习如何增强黑盒优化求解器，涵盖表达力更强的代理模型、元学习算法选择、动态算子配置与跨任务元优化等最新进展及其代表性算法。 |
| [^386] | [Hybrid Approach for Enhancing Lesion Segmentation in Fundus Images](https://arxiv.org/abs/2509.25549) | 本文提出一种混合方法，将深度学习模型与传统数学/聚类分割方法相结合，以克服眼底图像数据集分辨率低、标注不一致以及标注数据不足的问题，实现脉络膜痣等眼底病变的精确分割，助力早期诊断工具的开发。 |
| [^387] | [Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games](https://arxiv.org/abs/2509.20919) | 该论文提出了一个基于核化范式的新框架，首次在部分信息环境下实现了多面体博弈（如布洛托上校博弈和拥塞博弈）中粗相关均衡的高效学习，使遗憾度与单次迭代复杂度随动作集规模呈多对数级增长。 |
| [^388] | [On the Rate of Gaussian Approximation for Online Linear Regression Problems](https://arxiv.org/abs/2509.14039) | 本文针对恒定步长下的在线线性回归问题推导了高斯近似速率，显式刻画了其对问题维度和设计矩阵相关量的依赖，并证明当样本量足够大时可获得 $\sqrt{\log{n}/n}$ 阶的正态近似速率。 |
| [^389] | [Genome-Factory: A Library for Tuning, Deploying, and Interpreting Genomic Foundation Models](https://arxiv.org/abs/2509.12266) | Genome-Factory是首个统一了基因组基础模型数据收集、微调、推理、基准测试和可解释性分析的集成Python库，并推出了基于稀疏自编码器的开源生物解释器。 |
| [^390] | [Online Budget Allocation with Censored Semi-Bandit Feedback](https://arxiv.org/abs/2508.05844) | 本文研究了删失半老虎机反馈下的在线预算分配问题，提出了一种基于乐观原则的算法，并证明在收益递减情形下其遗憾随时间范围呈多对数级增长。 |
| [^391] | [CoMPAS3D: A Dataset and Benchmark for Interactive Motion](https://arxiv.org/abs/2507.19684) | 该论文提出将具有规范动作词汇和社区公认评判标准的萨尔萨舞作为具身交互评估领域，并构建了CoMPAS3D数据集与基准，以评估交互式动作生成中反应性动作的语义有效性与恰当性。 |
| [^392] | [A Neural Operator based on Dynamic Mode Decomposition](https://arxiv.org/abs/2507.01117) | 本文提出一种基于动态模态分解（DMD）的神经算子，通过结合DMD与深度学习自动提取关键模态和系统动力学，在热传导方程、拉普拉斯方程和Burgers方程求解上相比DeepONet和FNO等同类方法实现了更轻量高效的时空建模。 |
| [^393] | [Implementing Keyword Spotting on the MCXN947 Microcontroller with Integrated NPU](https://arxiv.org/abs/2506.08911) | 本文在集成NPU的MCXN947微控制器上实现了关键词唤醒系统，通过量化感知训练优化MFCC+CNN模型，相比仅CPU执行实现59倍推理加速，以仅30.58 KB的模型大小达到97.06%的准确率。 |
| [^394] | [PPO in the Fisher-Rao geometry](https://arxiv.org/abs/2506.03757) | 该论文利用Fisher-Rao几何提出了FR-PPO算法，在可作为PPO直接替换并取得相当性能的同时，提供了单调策略改进和次线性收敛等强理论保证。 |
| [^395] | [B-XAIC Dataset: Benchmarking Explainable AI for Graph Neural Networks Using Chemical Data](https://arxiv.org/abs/2505.22252) | 该论文提出了基于真实化学数据构建的B-XAIC基准数据集，通过具有已知真实依据的多样化分子任务，系统评估并揭示了现有图神经网络可解释AI方法的局限性。 |
| [^396] | [CubeDAgger: Interactive Imitation Learning for Dynamic Systems with Efficient yet Low-risk Interaction](https://arxiv.org/abs/2505.04897) | 本文提出CubeDAgger方法，通过阈值正则化激活、多动作候选最优共识机制和自回归有色噪声注入三项改进，实现了动态系统中高效且低风险的交互式模仿学习，在提升策略鲁棒性的同时显著减少动态稳定性违规。 |
| [^397] | [How to Backdoor Image Knowledge Distillation](https://arxiv.org/abs/2504.21323) | 本文提出了一种针对图像知识蒸馏的新型后门攻击方法，通过向蒸馏数据集注入被干净教师模型预测为目标标签的触发图像，使学生在教师保持干净的情况下学到后门，且在10%中毒率下仍然有效。 |
| [^398] | [Unraveling Groundwater Stress: A Model Generalizable Across Space and Time Reveals Co-Declining Water Tables and Recharge Across Bangladesh](https://arxiv.org/abs/2503.22771) | 该研究提出一种可跨时空泛化的“教师-学生”机器学习框架，将稀疏实地观测与GLDAS卫星数据结合，生成孟加拉国2公里高分辨率地下水埋深图，并揭示了地下水位与地下水补给量正在同步下降。 |
| [^399] | [Langevin Monte-Carlo Provably Learns Depth Two Neural Nets at Any Size and Data](https://arxiv.org/abs/2503.10428) | 本文证明了朗之万蒙特卡洛算法在任意网络规模和任意训练数据下都能可证明地收敛并学习两层神经网络，所需正则化量与网络规模无关，收敛速率仅取决于Poincaré常数。 |
| [^400] | [Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study](https://arxiv.org/abs/2501.18158) | 本文首次将大语言模型应用于比特币等真实加密货币交易图分析，提出三层能力评估框架、人类可读的图表示格式LLM4TG和连接增强的交易图采样算法，克服了传统黑盒模型难以解释和捕捉细微行为模式的局限。 |
| [^401] | [Mean and Variance Estimation Complexity in Arbitrary Distributions via Wasserstein Minimization](https://arxiv.org/abs/2501.10172) | 本文揭示了任意分布下均值与方差参数估计的复杂度：虽然最大似然估计是NP难的，但利用Wasserstein距离可在关于 $1/\varepsilon$ 的多项式时间内获得任意精度的 $\varepsilon$-近似解。 |
| [^402] | [A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare](https://arxiv.org/abs/2407.21665) | 本文综述了三种用于分析医疗领域纵向可穿戴传感器数据的计算模型（日常规律、生理节律和稳定性指标），并讨论了纵向数据处理的关键挑战，以助力实现预测性、预防性、个性化和参与性的4P医学。 |
| [^403] | [Q-Learning under Finite Model Uncertainty](https://arxiv.org/abs/2407.04259) | 提出了一种在有限模糊转移核集合下的鲁棒Q学习算法，证明了其几乎必然收敛到鲁棒最优解，并给出将可观测经验贝尔曼残差与转移核估计误差分离的时间一致高概率误差界。 |
| [^404] | [HarmonICA: Quasi-Time-Invariant ICA for Motor Unit Decomposition during Dynamic Contractions](https://arxiv.org/abs/2406.19581) | 提出了HarmonICA方法，通过在静态线性分离器前引入可学习的低秩时变可逆变换，克服混合过程时变的限制，首次将独立成分分析推广到动态收缩条件下的表面肌电运动单元分解。 |
| [^405] | [Implicit Regularization via Spectral Neural Networks and Non-linear Matrix Sensing](https://arxiv.org/abs/2402.17595) | 本文在更现实的神经网络背景下探讨了隐式正则化现象，通过研究非线性激活函数的一般类别，严格证明了在矩阵感知问题设置中这些网络的隐式正则化现象，同时提供了严格的速率保证，确保梯度的指数级快速收敛。 |
| [^406] | [Mirror Descent-Ascent for mean-field min-max problems](https://arxiv.org/abs/2402.08106) | 该论文研究了解决测度空间上极小极大问题的镜像下降-上升算法的两种变体，并证明了收敛速率与相关有限维算法的最新结果一致。 |
| [^407] | [RAYEN: Imposition of Hard Convex Constraints on Neural Networks](https://arxiv.org/abs/2307.08336) | RAYEN是一个能够在神经网络输出或潜变量上施加硬凸约束的框架，可在训练和测试阶段对任意输入和权重保证约束满足，支持线性、凸二次、二阶锥和线性矩阵不等式约束的任意组合，且计算开销几乎可以忽略。 |
| [^408] | [AI Oversight and Human Mistakes: Evidence from Centre Court.](http://arxiv.org/abs/2401.16754) | 人工智能系统在纠正人类错误方面起到了积极作用，但此举也潜在导致心理成本，并影响人的决策。通过研究网球比赛中的Hawk-Eye审查系统，我们发现引入AI监督后，裁判员的错误率下降，心理成本导致他们更倾向于将球判为进界，从而产生了类型错判的转变。 |

# 详细

[^1]: 用户嵌入是可解释行为模式的叠加

    User Embeddings are Superpositions of Interpretable Behavioral Modes

    [https://arxiv.org/abs/2611.08487](https://arxiv.org/abs/2611.08487)

    该论文提出用户嵌入可被理解为可解释行为模式的稀疏叠加，并证明稀疏自编码器（SAE）能有效恢复这些行为模式，从而显著提升用户嵌入的可解释性。

    

    用户嵌入是现代推荐、个性化和行为建模系统的核心组件。尽管这些嵌入能够紧凑地编码丰富的行为信息，但它们本质上是稠密且不透明的，限制了可解释性。为解决这一空白，我们提出用户嵌入可以被理解为可解释行为模式的稀疏叠加。这一结构由用户行为的生成模型在线性嵌入情形下直接预测得出，并依据叠加假设可近似地扩展到深度学习表示。叠加的模式可以通过稀疏字典学习方法（如稀疏自编码器SAE）自然地恢复，且恢复出的模式可以被自动命名并支持自然语言查询。我们在合成数据上对照已知真值验证了模式恢复的效果，结果表明SAE在恢复模式方面优于经典的稀疏基线方法，包括在Transformer编码的嵌入上同样如此。

    arXiv:2611.08487v1 Announce Type: new  Abstract: User embeddings are central components of modern recommendation, personalization, and behavioral modeling systems. Though these embeddings can compactly encode rich behavioral information, they are inherently dense and opaque, limiting interpretability. To address this gap, we argue that user embeddings can be understood as sparse superpositions of interpretable behavioral modes. This structure is directly predicted by generative models of user behavior for linear embeddings and extends approximately to deep-learning representations by the superposition hypothesis. Superimposed modes can naturally be recovered by sparse dictionary learning methods, such as sparse autoencoders (SAEs), and recovered modes can be automatically named and queried in natural language. We validate mode recovery against known ground truth on synthetic data, where SAEs recover modes better than classical sparse baselines, including on transformer-encoded embeddin
    
[^2]: Scientist World：为AI科学家培育一个科研成长世界

    Scientist World: Grow a Research World for AI-Scientists

    [https://arxiv.org/abs/2611.08478](https://arxiv.org/abs/2611.08478)

    该论文提出Scientist World，将每篇研究论文转化为可执行的研究环境，用于评测和训练AI科学家，并发现前沿AI科学家在74%的情况下已能超越人类研究者的成果。

    

    AI-科学家已经到来，但一个供他们成长的世界却尚未出现。人类科学家把这个世界视为理所当然，科学文献中积淀了数百年的悬而未决的问题与共享的思想。对AI-科学家而言，这个世界是可读的却不可运行的，因此我们为其构建了Scientist World：让每一篇研究论文都成为一个可执行的环境。我们的自主流水线会提取每篇论文的开放式研究问题，并将其提出的方法复现为基线；进入环境的AI-科学家将以人类作者取得的成果为标准接受评估。我们发布了首批1,000个经过验证的环境。以这个世界作为评测手段，我们发现前沿AI-科学家在74%的情况下已能超越人类研究者。作为AI如何做科学的记录，其中包含6,400小时的研究轨迹，涵盖每一个想法、实验、失败与成功。作为训练场，我们能够将一个小模型的性能提升（+158%）至与……（原文在此处截断）

    arXiv:2611.08478v1 Announce Type: new  Abstract: AI-Scientists arrived, but a world for them to grow in did not. Human scientists take theirs for granted $-$ centuries of open problems and shared ideas, waiting in the scientific literature. To AI-Scientists, that world is readable but not runnable, so we build Scientist World for them: every research paper becomes an executable environment. Our autonomous pipeline extracts each paper's open-ended research question and reproduces its proposed method as the baseline; an AI-Scientist entering the environment is evaluated against what the human authors achieved. We release the first 1,000 verified environments. With the world as an evaluation, we found that frontier AI-Scientists can already surpass the human researchers 74% of the time. As a record of how AI does science, it contains 6,400 hours of research traces, with every idea, experiment, failure, and success. As a training ground, we are able to raise (+158%) a small model to match 
    
[^3]: XStack：一个用于机器学习代码仓库迁移的基准测试

    XStack: A Benchmark for ML Repo Migration

    [https://arxiv.org/abs/2611.08472](https://arxiv.org/abs/2611.08472)

    该论文提出 XStack 基准，通过要求智能体将 PyTorch 代码仓库完整迁移为可在 TPU 上运行的 JAX 代码仓库，来评估其跨框架、跨硬件的长时程迁移能力，目前最强模型通过率仅约 75%。

    

    机器学习代码应该运行在研究者选择的地方，而不是它最初被编写的地方。如今，机器学习代码主要存在于两个生态系统中：运行在 NVIDIA GPU 上的 PyTorch 和运行在 Google TPU 上的 JAX。在两者之间迁移代码仓库意味着需要跨框架和硬件进行构建与测试。这种迁移（1）是一种实际需求，因为它能为研究者解锁不同的硬件资源；（2）也是一个天然的基准测试：它涉及不同的框架和硬件、覆盖整个代码仓库，并且是长时程且具有挑战性的任务。我们提出了 XStack，一个评估智能体在这两个生态系统之间迁移机器学习代码仓库能力的基准。每个智能体会收到一个 PyTorch 代码仓库和一台干净的 TPU 机器，并且必须交付一个 JAX 代码仓库。只有当其输出满足我们的评分标准（该标准基于源代码仓库已发布的结果，并经过我们的认证和校准）时，才算通过。目前智能体的迁移能力仍然不可靠：Claude Opus 5 的通过率为 75%，GPT-5.6-sol 为 45%。

    arXiv:2611.08472v1 Announce Type: cross  Abstract: ML code should run where researchers choose, not where it was first written. Today, machine learning code mainly lives in two ecosystems, PyTorch on NVIDIA GPUs and JAX on Google TPUs. Migrating repositories between them means building and testing across frameworks and hardware. This migration is (1) a practical need, since it unlocks different hardware for researchers, and (2) a natural benchmark: it involves different frameworks and hardware, covers whole repositories, and is long-horizon and challenging. We introduce XStack, a benchmark that evaluates agents on migrating ML repositories between these two ecosystems. Each agent receives a PyTorch repository and a clean TPU machine and must deliver a JAX repository. Its output passes if it meets our grading criteria, based on the source repository's published result, authenticated and calibrated by us. Migration by agents remains unreliable: Claude Opus 5 passes 75%, GPT-5.6-sol 45%, 
    
[^4]: 相同行为，不同机制：有限实验能对整合信息认证什么

    Same Behavior, Different Mechanism: What Finite Experiments Can Certify About Integrated Information

    [https://arxiv.org/abs/2611.08469](https://arxiv.org/abs/2611.08469)

    本文在具有已知真实答案的小型随机网络中精确刻画了整合信息 φ 等机制级量何时可被识别、连续且可由有限数据认证，并证明输入输出行为完全相同的网络可携带截然不同的整合信息（0.28 比特 vs 0 比特），但只需 4-53 次内部记录试验即可将其区分。

    

    可解释性、AI 审计以及机器意识评估越来越依赖于机制层面的量：这些量是系统因果转移律的泛函，描述系统如何计算，而非它输出什么。实验只能通过有限的、部分的且有噪声的干预来观察这一规律。我们探究这样的量何时能从实验的访问权限中被识别、对机制连续、并能从有限数据中得到认证，并针对一个刻意苛刻的情形——整合信息理论中的整合信息 $\varphi$——在具有已知真实答案的四单元和五单元随机网络中给出了精确的回答。行为可能是沉默的：一个循环网络与一个前馈网络在其响应延迟处具有完全相同的输入输出规律，分别携带 0.28 和 0 比特，但内部记录在 4-53 次试验内即可将二者区分开来。部分访问可能同样盲目：在我们的例子中，单步记录中被排除的单元会造成精度下限

    arXiv:2611.08469v1 Announce Type: cross  Abstract: Interpretability, AI auditing and assessments of machine consciousness increasingly rest on mechanism-level quantities: functionals of a system's causal transition law that describe how it computes, not what it outputs. Experiments see that law only through finite, partial and noisy interventions. We ask when such a quantity is identifiable from an experiment's access, continuous in the mechanism and certifiable from finite data, and answer exactly for a deliberately demanding case - the integrated information $\varphi$ of integrated information theory - in stochastic networks of four and five units with known ground truth. Behavior can be silent: a recurrent and a feed-forward network with identical input-output laws at their response delay carry 0.28 and 0 bits, yet internal recordings separate them within 4-53 trials. Partial access can be just as blind: in our examples, units left out of a one-step recording impose precision floors
    
[^5]: 通过扩展激活反演来解释残差流的任意方向

    Towards interpreting any residual stream direction by scaling activation inversion

    [https://arxiv.org/abs/2611.08465](https://arxiv.org/abs/2611.08465)

    提出最大激活示例元模型（MAEM），利用无标签强化学习目标将残差流的任意方向反演为强激活该方向的文本示例，在解释效果上超越了语料库搜索和自然语言自编码器。

    

    为了解释语言模型残差流中的某个方向，我们可以寻找能够强烈激活该方向的文本。但当通过扫描语料库来寻找此类文本时，结果会受到语料库内容的限制。我们开发了最大激活示例元模型，这是一类语言模型，能够将主体模型的残差流方向映射为可以强烈激活这些方向的文本示例。我们的优化目标是最大化给定的残差流方向与MAEM所生成文本的激活之间的余弦相似度。这一无标签的强化学习目标有助于在各种残差流方向来源上扩展MAEM的规模。实验表明，所得到的元模型能够产生具有信息量的示例：一个基于Qwen3.6-27B的MAEM在反演保留激活方面优于语料库搜索或自然语言自编码器（NLA），在大多数保留的语言/领域上胜过（基于1000万token的）语料库搜索，并能激活89%的保留……

    arXiv:2611.08465v1 Announce Type: cross  Abstract: To interpret a direction in a language model's residual stream, we can look for text that strongly activates it. But when such text is found by scanning a corpus, it is limited to what that corpus contains. We develop Maximally-Activating Example Metamodels (MAEMs), language models which map residual stream directions from a subject model to text exemplars that strongly activate those directions. Our objective is to maximise the cosine similarity of a supplied residual stream direction with the activations of the text generated by the MAEM. This label-free RL objective helps to scale MAEMs on diverse sources of residual stream directions. Empirically, we find the resultant metamodel produces informative exemplars: a Qwen3.6-27B MAEM inverts held-out activations better than corpus search or a natural-language autoencoder (NLA), beats corpus search (at 10M tokens) on a majority of held-out languages/domains, and activates 89% of held-out
    
[^6]: 大步长下的对抗训练：隐式偏差、优化与锐度

    Adversarial Training with Large Step Sizes: Implicit Bias, Optimization, and Sharpness

    [https://arxiv.org/abs/2611.08457](https://arxiv.org/abs/2611.08457)

    本文研究了大恒定步长梯度下降在对抗训练中的动力学，从隐式偏差、优化速度和损失锐度演化三个方面揭示了其与标准训练的关键差异。

    

    对抗训练是提高模型鲁棒性的重要方法。我们研究了确定性梯度下降（GD）在对抗训练中的动力学。该方向上大多数现有工作集中在小步长或步长趋于消失的情形，此时梯度下降类似于连续时间流。然而，这样的步长会使训练效率低下，在实践中很少被使用。在本文中，我们研究了采用大恒定步长的梯度下降在对抗训练中的动力学。我们首先研究了对抗线性逻辑回归中的梯度下降，重点关注两个方面：模型参数收敛到哪里（隐式偏差）以及损失下降的速度（优化）。随后我们转向神经网络训练，研究在对抗训练过程中损失锐度如何演化。我们的结果加深了对大步长梯度下降在对抗训练中行为的理解，并突出了其与标准训练（ST）中行为的三个关键差异：（i）对抗训练与标准训练之间的隐式偏差差距……

    arXiv:2611.08457v1 Announce Type: new  Abstract: Adversarial Training (AT) is an important method for improving model robustness. We study the dynamics of deterministic Gradient Descent (GD) in AT. Most existing works in this direction focus on the small or vanishing step sizes regime, where GD resembles a continuous-time flow. However, such step sizes can make training inefficient and are rarely used in practice. In this paper, we study the AT dynamics of GD with large constant step sizes. We first study GD in adversarial linear logistic regression, focusing on two aspects: where the model parameters converge (implicit bias) and how quickly the losses decrease (optimization). We then turn to neural networks training and study how the sharpness of losses evolve during AT. Our results advance the understanding of the behaviors of GD with large step sizes in AT and highlight three key differences from its behavior in Standard Training (ST): (i) the implicit bias gap between AT and ST, pr
    
[^7]: 解析强到弱在线策略蒸馏（OPD）相对于GRPO的计算效率

    Unpacking the Compute Efficiency of Strong-to-Weak OPD against GRPO

    [https://arxiv.org/abs/2611.08453](https://arxiv.org/abs/2611.08453)

    本研究系统对比了在线策略蒸馏（OPD）与GRPO的计算效率，发现使用外部教师时OPD效率不如GRPO，但当教师是学生自身的GRPO检查点时，OPD仅需少50倍的FLOPs即可达到相同的数学准确率。

    

    在线策略蒸馏（On-policy distillation，OPD）在冻结教师模型的逐token监督下，基于学生模型自身的rollout进行训练。与强化学习相比，这种密集信号能否减少达到目标准确率所需的训练FLOPs？我们在任务难度、学生与教师模型规模、领域以及优化选择等多个维度上，系统性地比较了原始的反向KL OPD与群体相对策略优化（GRPO），以训练FLOPs为衡量基准评估留出集准确率。结果表明，当使用外部后训练的教师模型时，在匹配设置下，OPD在每单位训练FLOP所能达到的准确率上通常逊色于GRPO，即使教师模型是经过GRPO进一步训练的，且OPD经常出现回复长度膨胀的问题。然而，换一种教师模型会改变这一图景：当教师模型是学生自身经GRPO训练得到的检查点时，OPD在数学任务上能够恢复该检查点的准确率，所需的蒸馏FLOPs比GRPO达到相同准确率所需的少50倍。遵循相同的教师设置

    arXiv:2611.08453v1 Announce Type: cross  Abstract: On-policy distillation (OPD) trains a student on its own rollouts under a frozen teacher's per-token supervision. Does this dense signal reduce the training FLOPs needed to reach a target accuracy compared with reinforcement learning? We systematically compare vanilla reverse-KL OPD with Group Relative Policy Optimization (GRPO) across task difficulty, student and teacher size, domain, and optimization choices, measuring held-out accuracy against training FLOPs. With an external post-trained teacher, OPD generally loses to GRPO in accuracy per training FLOP at matched settings, even when the teacher is further trained with GRPO, and often suffers response-length inflation. Yet a different teacher changes the picture: when the teacher is the student's own GRPO checkpoint, OPD recovers that checkpoint's accuracy on math with $50\times$ fewer distillation FLOPs than GRPO needs to reach the same accuracy. Following the same teacher setting
    
[^8]: 用于对手行动在线预测的隐马尔可夫偏好模型

    A Hidden Markov Preference Model for Online Prediction of Adversary Actions

    [https://arxiv.org/abs/2611.08445](https://arxiv.org/abs/2611.08445)

    本文提出一种隐马尔可夫偏好模型，将攻击者行为模式作为隐状态、在可行性掩码行动树上定义模式条件下的行动分布，从而实现对对手下一步行动的在线概率预测、信念更新及信息量化的理论保证。

    

    预判攻击者的防御者需要在攻击行动发生之前，基于已有信息得到攻击者下一步行动的概率分布。两种结构性因素使这一问题变得复杂：攻击者的行为模式会在一次对抗过程中持续存在并可能发生切换；且每个行动由操作与依赖状态的参数组合而成，这些参数的可行取值随时间变化。我们用隐马尔可夫模型同时刻画这两点：其隐状态为当前活跃的行为模式，而该模式条件下的行动分布定义在一棵经过可行性掩码处理的可执行行动树上。同一似然函数既构成下一步行动分布，又在观察到行动后更新对行为模式的信念，部分观测到的前缀序列则贡献其已解析的局部因子。估计过程可分解为凸的逻辑回归拟合，当行为模式无标签时可由 Baum-Welch 算法复用。我们推导了衡量行为模式信息的可达性加权逐节点 KL 散度表达式，以及关于切换的序列级对数损失上界……（摘要截断）

    arXiv:2611.08445v1 Announce Type: new  Abstract: A defender anticipating an attacker needs a probability distribution over the attacker's next action from information available before that action occurs. Two structures complicate this: the attacker's behavioral regime persists and can switch within an episode, and each action combines an operation with state-dependent arguments whose feasible values change over time. We capture both with a hidden Markov model whose latent state is the active regime and whose regime-conditioned action law is defined on a feasibility-masked tree of executable actions. The same likelihood forms the next-action distribution and updates the regime belief once the action is seen, and a partially observed prefix contributes its resolved local factors. Estimation separates into convex logistic fits that Baum-Welch reuses when regimes are unlabeled. We derive a reach-weighted nodewise KL expression for regime information, a sequence-wise log-loss bound for swit
    
[^9]: 类型化决策编码器中的标签措辞增强：对不熟悉的词语有效，对缩写或代号无效

    Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes

    [https://arxiv.org/abs/2611.08439](https://arxiv.org/abs/2611.08439)

    论文对俄语类型化决策交叉编码器 Ru-Jev 引入标签多形式措辞增强，发现该增强对由普通词和外来词构成的标签词汇表能带来一致的准确性提升（最高达 0.24），但对缩写和代号类标签没有增益。

    

    类型化决策模型会对调用方在请求时提供的选项进行评分，因此其准确性取决于模型开发者无法控制的标签措辞。我们比较了 Ru-Jev 的两个版本——这是一个在十三项俄语任务上训练的 3.07 亿参数多语言交叉编码器；后一版本增加了一种增强方法，将标签以多种表面形式呈现。在保持问题不变、仅改变十八个路由标签措辞的条件下，我们评估了十套标签词汇表，这些词汇表由三种独立流程生成，并检查了与训练数据的重合情况；每题使用五个候选标签，其中八套词汇表还使用了十八个候选标签进行评估。在由普通词汇或外来词（源自英语的行话、办公室俚语、部门名称）构成的保留词汇表上，增强版本在所有设置下都更加准确，提升幅度最高达 0.24。这十四项增益中有十一项在 Holm 校正后仍然显著（十八个候选标签的俚语以及描述性……（摘要在此处截断）

    arXiv:2611.08439v1 Announce Type: cross  Abstract: Typed-decision models score options that the caller supplies at request time, so their accuracy depends on label wording that the model's developer does not control. We compare two releases of a 307M-parameter multilingual cross-encoder, Ru-Jev, trained on thirteen Russian-language tasks; the later release adds an augmentation that presents labels in several surface forms. Holding questions fixed and varying only how eighteen routing labels are worded, we evaluate ten label vocabularies, produced by three separate processes and checked for overlap with the training data, with five candidates per question and, for eight of them, also with eighteen. On held-out vocabularies made of ordinary or borrowed words (English-derived jargon, office slang, department names), the augmented release is more accurate in every setting, by up to 0.24. Eleven of these fourteen gains survive a Holm correction (slang with eighteen candidates and descriptiv
    
[^10]: 从内部理解人工智能：面向AI安全的机制可解释性研究现状

    Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety

    [https://arxiv.org/abs/2611.08438](https://arxiv.org/abs/2611.08438)

    本论文系统综述了机制可解释性在AI安全领域的研究现状，探讨通过理解模型内部表示与计算来评估、预测和控制AI行为，能否真正使AI系统更加安全。

    

    神经网络很可能是人类历史上最重要的人造物之一。随着这些系统变得更加强大和自主，它们造成社会危害的潜力也在增长，这使得AI安全成为我们这个时代的决定性挑战之一。然而，我们在尚未完全理解这些系统如何工作、它们在陌生情境中会如何表现、以及如何可靠地控制它们的情况下，仍在继续构建越来越强大的系统。机制可解释性通过研究这些系统的内部表示和计算，为理解这些系统提供了一条路径，其目标是评估、预测和控制模型行为。我们对面向AI安全的机制可解释性进行了综述，以回答这样一个问题：理解模型的工作原理真的能让它们更安全吗？我们将机制可解释性置于更广泛的可解释性研究背景之中，并将其机制性证据与行为评估进行了比较……

    arXiv:2611.08438v1 Announce Type: cross  Abstract: Neural networks are likely to be among the most consequential artefacts in human history. As these systems become more capable and autonomous, their potential to cause societal harm also grows, making AI safety one of the defining challenges of our time. Yet we continue to build increasingly powerful systems without fully understanding how they work, how they will behave in unfamiliar situations, or how to reliably control them. Mechanistic interpretability offers a path towards understanding these systems by studying their internal representations and computations, with the aim of evaluating, predicting, and controlling behaviour. We survey mechanistic interpretability for AI safety in order to answer the question: Does understanding how models work actually make them safer? We place mechanistic interpretability in the broader context of interpretability research and compare its mechanistic evidence with behavioural evaluation. We int
    
[^11]: 微分方程系统内蕴李点对称性的数据驱动发现

    Data-driven discovery of intrinsic Lie point symmetries of differential equation systems

    [https://arxiv.org/abs/2611.08421](https://arxiv.org/abs/2611.08421)

    该论文提出了一种将内蕴李点对称性发现转化为采样射流空间数据上算子理论问题的数据驱动框架，无需符号决定方程或控制系统的闭式表示，即可从微分方程数据中直接识别出有意义的连续对称性，并过滤掉冗余与几何平凡的方向。

    

    直接从微分方程数据中发现连续对称性，可以在无需符号化决定方程或控制系统闭式表示的情况下，为模型降阶和科学机器学习揭示几何结构。我们将内蕴李点对称性发现表述为采样射流空间数据上的算子理论问题。容许的生成元从采样的决定算子中识别出来，并与冗余方向、依赖于解的方向以及几何平凡的方向分离开来。该框架结合了延拓、逐次子空间约束以及一个关于可训练系数为线性的固定非线性字典。秩稳定预检测试可诊断出那些所表示的内蕴对称代数在字典细化下不稳定的系统。在一个解析已知具有11维所表示内蕴对称代数的受控微分方程基准上，该方法……

    arXiv:2611.08421v1 Announce Type: new  Abstract: Discovering continuous symmetries directly from differential-equation data can reveal geometric structure for model reduction and scientific machine learning without symbolic determining equations or a closed-form representation of the governing system. We formulate intrinsic Lie point-symmetry discovery as an operator-theoretic problem on sampled jet-space data. Admissible generators are identified from a sampled determining operator and separated from redundant, solution-dependent, and geometrically trivial directions. The framework combines prolongation, sequential subspace restrictions, and a fixed nonlinear dictionary linear in trainable coefficients. A rank-stabilization preflight test diagnoses systems whose represented intrinsic symmetry algebra does not stabilize under dictionary refinement. On a controlled differential-equation benchmark with an analytically known 11-dimensional represented intrinsic symmetry algebra, the metho
    
[^12]: 论分子表示的局限性

    On the limitations of molecular representations

    [https://arxiv.org/abs/2611.08420](https://arxiv.org/abs/2611.08420)

    本文系统评估了分子表示方法的质量，发现最先进的分子嵌入模型在捕捉图结构、3D形状和ADMET相似性方面均不及传统指纹与理化描述符，且经典指纹自身也存在严重局限——在大规模数据库中可导致高达数十亿分子的表示不唯一，影响虚拟筛选的可靠性。

    

    指纹和学习嵌入等分子表示方法在化学信息学和小分子机器学习中发挥着核心作用。它们作为聚类、数据集划分和虚拟筛选等多种任务的主要输入，但其评估往往仅局限于狭窄基准任务上的下游模型性能。我们对分子表示的质量进行了系统性研究，发现目前最先进的分子嵌入模型在捕捉图结构、3D形状或ADMET端点相似性方面，均不如指纹或理化描述符。然而，经典指纹也暴露出其自身的重要局限性。例如，我们证明，根据所使用的表示方法不同，在108亿的Enamine REAL数据库中，有2.25亿至约40亿个分子存在非唯一表示，这对虚拟筛选具有重要意义。我们的发现提供了一个……

    arXiv:2611.08420v1 Announce Type: new  Abstract: Molecular representations such as fingerprints and learned embeddings play a central role in cheminformatics and small-molecule machine learning. They serve as the primary input for a diverse array of tasks, including clustering, dataset splitting, and virtual screening, yet their evaluation is often limited to downstream model performance on narrow benchmark tasks. We carried out a systematic investigation of the quality of molecular representations, finding that none of the state-of-the-art molecular embedding models capture graph structure, 3D shape, or ADMET endpoint similarity better than fingerprints or physicochemical descriptors. However, classical fingerprints also revealed their own material limitations. For example, we show that depending on the representation, between 225M and $\sim$4B molecules have non-unique representations in the 10.8B Enamine REAL database, with implications for virtual screening. Our findings provide an
    
[^13]: 监督微调与强化学习中LoRA的谱与子空间动力学

    Spectral and Subspace Dynamics of LoRA in Supervised Fine-Tuning and Reinforcement Learning

    [https://arxiv.org/abs/2611.08414](https://arxiv.org/abs/2611.08414)

    该研究通过追踪LoRA更新的奇异谱与子空间，发现强化学习相比监督微调表现出谱集中和更快的子空间旋转，并揭示了组均值基线减除是造成这两种差异的根本机制。

    

    低秩适应（LoRA）通常同时用于监督微调（SFT）和强化学习（RL）两个阶段，且往往采用相似的优化器设置。然而，这两个阶段的不同动力学如何在LoRA所优化的低秩增量中体现，目前仍不清楚。通过在数学推理任务上追踪LoRA更新的奇异谱和子空间，我们发现：在所有学习率下，RL都会使谱趋于集中，而SFT在中等学习率下则会扩展谱；此外，在小学习率下，RL旋转奇异子空间的速度远快于SFT。通过在SFT和RL之间逐个成分进行插值，我们确定了组均值基线减除是造成这两种差异的根源。通过LoRA更新的二次多指标模型揭示了其背后的机制：组均值基线减除使两种相互竞争的力量——一致性拟合与梯度噪声——之间的平衡发生倾斜，从而趋向谱集中和更快的（子空间旋转）……

    arXiv:2611.08414v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) is commonly used in both supervised fine-tuning (SFT) and reinforcement learning (RL), often under similar optimizer settings. Yet, how the different dynamics of the two stages manifest in the low-rank increment that LoRA optimizes remains unclear. Tracking the singular spectrum and subspaces of LoRA updates on mathematical reasoning tasks, we find that RL concentrates the spectrum at every learning rate, while SFT expands it at moderate learning rates; moreover, RL rotates the singular subspaces much faster than SFT at small learning rates. Interpolating between SFT and RL one ingredient at a time, we pinpoint group-mean baseline subtraction as the source of both contrasts. A quadratic multi-index model updated via LoRA unveils the underlying mechanism: group-mean baseline subtraction tilts the balance between two competing forces, coherent fitting and gradient noise, toward spectral concentration and faster s
    
[^14]: 时间依赖偏微分方程的共形预测

    Conformal Prediction for Time-Dependent PDEs

    [https://arxiv.org/abs/2611.08409](https://arxiv.org/abs/2611.08409)

    本文证明了时间依赖偏微分方程的共形预测覆盖保证在本质上难以实现——函数空间中相近时刻的分布可相互奇异，离散化中全变差距离随网格分辨率指数饱和——并给出了为确保覆盖而对空间离散化施加的必要限制。

    

    不确定性量化在科学机器学习中至关重要，因为模型的输出会为洪水预报、金融风险管理以及机器热控制等安全关键任务提供依据。共形预测能够提供无分布假设的覆盖保证，但在物理和工程中常见的时间依赖场景下，这些保证可能失效，导致系统性的覆盖不足。我们在由偏微分方程描述的时间依赖物理系统的代理模型背景下研究这一问题。我们证明，在函数空间设定下，任意接近的两个时刻的分布可以相互奇异，使得精确的覆盖保证不可能实现。我们进一步表明，在离散化设定下，分布之间的全变差距离往往随网格分辨率的提高而呈指数级饱和，这要求对空间离散化施加限制才能确保覆盖。最后，对于某些……

    arXiv:2611.08409v1 Announce Type: new  Abstract: Uncertainty quantification is crucial in scientific machine learning, where models inform safety-critical tasks such as flood forecasting, financial risk management, and thermal control in machines. Conformal prediction provides distribution-free coverage guarantees, but in time-dependent settings common to physics and engineering, these guarantees can break down, leading to systematic undercoverage. We study this problem in the context of surrogate models for time-dependent physical systems described by partial differential equations. We prove that in a function space setting, distributions at arbitrarily close times can be mutually singular, making exact coverage guarantees impossible. We further show that in discretized settings, the total variation distance between distributions often saturates exponentially fast with the grid resolution, mandating restrictions on the spatial discretization to ensure coverage. Finally, for certain se
    
[^15]: 社会人形机器人：具有社会响应性的人-人形机器人交互

    Social Humanoid: Socially Responsive Human-Humanoid Interaction

    [https://arxiv.org/abs/2611.08407](https://arxiv.org/abs/2611.08407)

    提出Social Humanoid分层系统，通过共享标记化表示的动作序列建模和两阶段自适应重定向训练方法，使29自由度人形机器人能够根据人类伙伴的动作生成具有社会响应性的全身运动，从而实现真正的人机物理协调互动。

    

    人形机器人通常被训练为孤立智能体，导致它们无法与人类伙伴进行物理协调，而人类伙伴之间会自然地相互预判并做出调整。为了弥合这一差距，我们提出了 Social Humanoid，这是一个分层系统，能够以人类伙伴的动作作为条件，在29自由度的宇树G1机器人上生成并执行具有社会响应性的全身运动。其高层策略通过共享的标记化表示对人形机器人与人类的动作进行灵活的序列建模，生成具有反应性的人形机器人动作，并在伙伴被遮挡时预判其运动；低层全身跟踪控制器则在机器人上执行生成的动作。为了训练该系统，我们引入了一种两阶段自适应重定向方法，将其中一个交互伙伴重定向到机器人上，同时对另一个伙伴进行形态变换，以保持接触动力学、可达性和相对身体比例。我们应用这一程序……

    arXiv:2611.08407v1 Announce Type: cross  Abstract: Humanoid robots are typically trained as solitary agents, leaving them unable to coordinate physically with human partners who naturally anticipate and adjust to one another. To bridge this gap, we present Social Humanoid, a hierarchical system that generates and executes socially responsive whole-body motion on a 29-DoF Unitree G1 conditioned on a human partner's motion. Its high-level policy performs flexible sequence modeling of human and humanoid motion through a shared tokenized representation, generating reactive humanoid motion and anticipating the partner's motion when they are occluded, while a low-level whole-body tracking controller executes the generated motion on the robot. To train the system, we introduce a two-phase adaptive retargeting procedure that retargets one interaction partner to the robot while morphing the other partner to preserve contact dynamics, reachability, and relative body scale. We apply this procedur
    
[^16]: StepBack：面向高效测试时推理的步级错误定位重采样方法

    StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning

    [https://arxiv.org/abs/2611.08403](https://arxiv.org/abs/2611.08403)

    StepBack通过轻量级信号定位推理草稿轨迹中首次出错的步骤，仅对该错误步骤及其后续内容进行重采样再投票，在七个数学推理与多跳问答基准上达到与自洽性相当的准确率，同时最多减少41.26%的token消耗。

    

    自洽性（Self-Consistency, SC）是一种广泛用于改进思维链推理的解码策略，但它需要以预设预算采样大量完整的推理轨迹。其高效变体主要通过提前停止完整轨迹采样或在多个问题之间分配采样数量来降低成本。在本文中，我们提出了StepBack，一种用于高效测试时推理的步级错误定位重采样方法。给定一小组草稿轨迹，StepBack利用轻量级信号定位可能首次出现错误的位置，然后对检测到的错误步骤及其后缀进行重采样，最后进行多数答案投票。实验结果表明，StepBack在七个数学推理和多跳问答基准上平均达到了与SC相当的准确率，同时最多可将token使用量降低41.26%。此外，StepBack天然可以与自适应推理方法（如Adaptive Consistency）相结合，进一步提升性能。

    arXiv:2611.08403v1 Announce Type: new  Abstract: Self-consistency (SC) is a widely used decoding strategy for improving chain-of-thought reasoning, but it requires sampling many complete reasoning trajectories with a preset budget. Its efficient variants reduce the cost mainly by stopping full-trajectory sampling early or distributing the number of samples across questions. In this paper, we propose StepBack, a step-level error-localized resampling method for efficient test-time reasoning. Given a small set of draft trajectories, StepBack localizes likely first error positions using lightweight signals and then resamples the detected error step and its suffix, followed by majority answer voting. Empirically, StepBack matches the accuracy of SC on average across seven mathematical reasoning and multi-hop QA benchmarks, while reducing token usage by up to 41.26%. In addition, by nature, StepBack can be combined with adaptive inference methods such as Adaptive Consistency, further improvi
    
[^17]: 卵巢癌多组学数据集的合成：一种基于生物学的评估框架

    Synthesis of Ovarian Cancer Multi-Omics Datasets: A Biology-Informed Evaluation Framework

    [https://arxiv.org/abs/2611.08394](https://arxiv.org/abs/2611.08394)

    本文的核心贡献是提出了一个基于生物学信息且患者分离的评估框架，用于判断合成的配对RNA表达与拷贝数变异数据是否保留了生物学相关结构，并将其应用于高级别浆液性卵巢癌DECIDER队列。

    

    配对的多组学癌症队列构建成本高昂且难以共享，这促使合成队列成为一种补充资源。随着生成模型开发的加速，我们将研究重点从提出又一种新架构转移到定义评估标准上，即确定合成数据是否保留了生物学相关的结构。我们引入了一个基于生物学信息且患者分离的评估框架，用于评估配对RNA表达和拷贝数变异（CNV）谱的生成效果，并将其应用于纵向DECIDER高级别浆液性卵巢癌队列。该框架以真实训练数据与保留数据之间的一致性作为参考，评估各模态的保真度以及RNA与CNV配对之间的连贯性，同时辅以训练数据邻近性等补充性诊断指标。我们改编了三种基于变分自编码器的多组学整合架构来生成配对数据。表现最优的模型……（摘要至此截断）

    arXiv:2611.08394v1 Announce Type: new  Abstract: Paired multi-omic cancer cohorts are costly to assemble and difficult to share, motivating synthetic cohorts as a complementary resource. As generative model development accelerates, we shift the focus from proposing yet another architecture to defining evaluation criteria that determine whether synthetic data preserve biologically relevant structure. We introduce a biology-informed and patient-disjoint framework for evaluating the generation of paired RNA expression and copy-number variation (CNV) profiles, and apply it to the longitudinal DECIDER high-grade serous ovarian cancer cohort. The framework assesses modality-specific fidelity and coherence between RNA and CNV pairs using agreement between real training and held-out data as a reference, alongside complementary training-data proximity diagnostics. We adapt three multi-omic integration architectures based on variational autoencoders to generate paired profiles. The model leading
    
[^18]: 基于混合机器学习的飞行数据不平衡着陆阶段安全事件预测

    Hybrid Machine Learning for Imbalanced Landing-Phase Safety Event Prediction from Flight Data

    [https://arxiv.org/abs/2611.08386](https://arxiv.org/abs/2611.08386)

    该研究提出一种可解释的混合特征融合框架，将LSTM自编码器的重构误差描述符与统计飞行历史特征融合后输入类别加权XGBoost分类器，以在高度不平衡的多变量飞行数据中实现多类别着陆阶段安全事件预测。

    

    着陆阶段安全事件预测需要在高度不平衡的多变量飞行数据中检测罕见的异常进近。在捕捉耦合时间行为的同时区分不同机场运行环境下的事件特定特征，仍然是一个核心建模挑战。本研究提出了一种可解释的混合特征融合框架，用于多类别着陆阶段安全事件预测。在正常飞行窗口上训练的长短期记忆（LSTM）自编码器用于表征偏离常规进近行为的程度，其连续的重构误差描述符与参数特定的统计飞行历史特征相融合，并输入类别加权的XGBoost分类器，以区分正常、速度偏高、航径偏高和襟翼放下过晚等窗口。该框架使用来自三个到达机场的53,750个NASA DASHlink进近窗口进行评估。混合学习与机场特定学习策略正在进…（原文在此处截断）

    arXiv:2611.08386v1 Announce Type: cross  Abstract: Landing-phase safety-event prediction requires the detection of rare abnormal approaches within strongly imbalanced multivariate flight data. Capturing coupled temporal behavior while distinguishing event-specific signatures across airport operating contexts remains a central modeling challenge. This study proposes an interpretable hybrid feature-fusion framework for multiclass landing-phase safety-event prediction. A long short-term memory (LSTM) autoencoder trained on nominal flight windows characterizes deviations from routine approach behavior. Its continuous reconstruction-error descriptor is fused with parameter-specific statistical flight-history features and supplied to a class-weighted XGBoost classifier to distinguish Normal, Speed High, Path High, and Flaps Late windows. The framework is evaluated using 53,750 NASA DASHlink approach windows from three arrival airports. Pooled and airport-specific learning strategies are comp
    
[^19]: 面向图神经网络的模块感知优化

    Module-Aware Optimization for Graph Neural Networks

    [https://arxiv.org/abs/2611.08379](https://arxiv.org/abs/2611.08379)

    提出了GraphMuon，一种图感知优化器，通过在图传播层输入的协方差结构所定义的度量下进行正交化并据此规范化步长，解决了现有矩阵优化器在训练图神经网络时忽略图传播表示几何结构、导致更新效果被放大的问题。

    

    不同的优化器对参数更新施加不同的几何结构。诸如Adam和AdamW之类的自适应方法采用逐坐标的更新方式，而诸如Muon之类的矩阵感知优化器则通过正交化更新来利用权重矩阵的结构。然而，对于图神经网络（GNN）而言，其权重矩阵的有效几何结构不仅由其学习到的通道混合权重所决定，还由其所作用的图传播表示所塑造。因此，即使某个更新作为矩阵已被很好地规范化，当其应用于图传播特征时，它对层输出的影响也可能被放大。为了弥补这一不足，我们提出了GraphMuon，一种图感知的优化器，它将Muon的矩阵更新适配到图传播层输入的协方差结构上。GraphMuon在这些传播表示所定义的度量空间中执行正交化，并根据更新对层表示产生的实际影响来规范化步长。

    arXiv:2611.08379v1 Announce Type: new  Abstract: Different optimizers impose different geometries on parameter updates. Adaptive methods such as Adam and AdamW use coordinate-wise updates, while matrix-aware optimizers such as Muon utilize weight-matrix structure through orthogonalized updates. For Graph Neural Networks (GNNs), however, the effective geometry of its weight matrices is shaped not only by its learned channel-mixing weights, but also by the graph-propagated representations it acts on. Thus, even if an update is well normalized as a matrix, its effect on the layer output can be amplified when applied to graph-propagated features. To address this gap, we introduce GraphMuon, a graph-aware optimizer that adapts Muon's matrix updates to the covariance structure of graph-propagated layer inputs. GraphMuon performs orthogonalization in a metric defined by these propagated representations and normalizes step sizes according to the update's induced effect on the layer representat
    
[^20]: Muir：基于敏感性分配门控与匹配滤波读出的线性注意力

    Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout

    [https://arxiv.org/abs/2611.08376](https://arxiv.org/abs/2611.08376)

    提出线性注意力架构Muir：利用“状态更新对擦除的敏感性远低于写入”这一发现，将所有门控融合为敏感性分配的低秩投影（擦除门采用低秩参数化），并对状态行重加权以实现匹配滤波式读出，从而提升线性注意力的效率与性能。

    

    线性注意力将上下文压缩为固定大小的键值状态，从而实现线性时间训练和恒定内存推理。门控Delta规则模型会衰减该状态，并在写入新值之前擦除其对输入键的预测；Gated DeltaNet-2（GDN-2）对此进行了推广，采用逐通道衰减以及解耦的全秩擦除门和写入门。然而，我们证明状态更新对擦除的敏感性远低于对写入的敏感性：键激活和单位归一化会将键能量集中于稀疏的通道子集，而值写入则会缩放所有通道，因此擦除门可以用更低的秩来参数化。此外，慢衰减的状态行会比快衰减的行累积大得多的范数，在未加权的读出中占据主导地位，从而迫使写入时进行更强的擦除以抑制干扰。基于这一分析，我们提出了Muir，它将所有门融合为一个敏感性分配的低秩投影，并对状态行进行重加权（原文摘要在此处截断，后续内容缺失）。

    arXiv:2611.08376v1 Announce Type: new  Abstract: Linear attention compresses context into a fixed-size key-value state for linear-time training and constant-memory inference. Gated delta-rule models decay this state and erase its prediction for an incoming key before writing a new value; Gated DeltaNet-2 (GDN-2) generalizes this with channel-wise decay and decoupled, full-rank erase and write gates. However, we show that state updates are far less sensitive to erasure than to writes: key activation and unit normalization concentrate key energy onto a sparse subset of channels, whereas value writes scale all channels, so the erase gate can be parameterized at lower rank. Furthermore, slow-decaying state rows accumulate much larger norms than fast rows, dominating unweighted readout and forcing stronger write-time erasure to suppress interference. Guided by this analysis, we introduce Muir, which fuses all gates into a sensitivity-allocated low-rank projection, re-weights state rows with
    
[^21]: SHORE：面向在线装箱问题的稳定性混合在策略强化学习

    SHORE: Stability-Hybrid On-Policy Reinforcement Learning for Online Bin Packing

    [https://arxiv.org/abs/2611.08364](https://arxiv.org/abs/2611.08364)

    该论文提出SHORE方法，通过3D U-Net学习稳定性概率并结合在策略收益估计，将稳定性风险融入箱子选择、朝向和放置位置的联合决策中，有效解决了在线三维装箱中防止倒塌与最大化装载量的权衡问题。

    

    带处理缓冲区的在线三维装箱问题需要在未知未来货物到达的情况下，选择要装载的箱子、其朝向以及放置位置，同时保持堆叠体的稳定性。一旦发生倒塌，回合即终止并损失剩余装载体积。现有方法依赖几何规则或硬稳定性掩码，限制了风险因素对箱子、朝向与放置位置联合决策的影响。我们提出稳定性混合在策略强化学习方法（SHORE），它将学习到的放置稳定性概率与以稳定性为条件的在策略收益估计相结合。其中，一个3D U-Net通过已执行的结果和模拟的替代方案学习稳定性，从而揭示早期滚动采样中代表性不足的动作的真实结果。一个离散的在策略actor-critic将稳定性预测同时用于其动作价值和策略先验中，并通过门控机制限制未训练或持续变化的模型所带来的影响。在部署阶段，SHORE不需要额…（摘要截断）

    arXiv:2611.08364v1 Announce Type: cross  Abstract: Online 3D bin packing with a processing buffer requires choosing which box to pack, its orientation, and its placement while keeping the stack stable under unknown future arrivals. A collapse ends the episode and forfeits the remaining volume. Existing methods use geometric rules or hard stability masks, limiting how risk can influence the joint choice of box, orientation, and placement. We propose Stability-Hybrid On-Policy Reinforcement Learning (SHORE), which combines a learned stability probability of a placement with an on-policy estimate of its return conditional on stability. A 3D U-Net learns stability from executed outcomes and simulated alternatives, revealing outcomes of actions underrepresented in early rollouts. A discrete on-policy actor--critic uses stability predictions in both its action values and policy prior, with gates that limit the effect of an untrained or changing model. In deployment, SHORE does not require ad
    
[^22]: StrataFlux：内存层次结构原生的低比特大语言模型推理

    StrataFlux: Memory-Hierarchy-Native Low-Bit LLM Inference

    [https://arxiv.org/abs/2611.08363](https://arxiv.org/abs/2611.08363)

    StrataFlux 框架通过从主机内存和固态硬盘进行选择性精度流式传输，并让数据交付与计算重叠、跨解码步骤复用精度信息，在保留量化内存和速度优势的同时，有效恢复了低比特权重和 KV 缓存量化导致的 LLM 推理精度损失。

    

    量化降低了大语言模型（LLM）推理的内存和带宽需求，但可能导致精度下降；而卸载更高精度的数据反而会使生成过程停滞。我们提出了 StrataFlux，这是一个打破这种权衡的框架，它能够恢复因低比特权重和键值（KV）缓存量化而损失的精度，同时保留量化在内存和速度方面的优势。StrataFlux 通过联合设计精化表示、选择和交付机制，使主机内存（RAM）和固态硬盘成为提供额外精度的有效来源。其选择性精度流式传输减少了数据传输量，将数据交付与计算过程重叠进行，并在多个解码步骤中复用已获取的精度信息，使生成过程无需等待每一次传输。我们在专用 GPU 和统一内存平台上评估了 2-4 比特权重和 2-3 比特 KV 缓存的方案。对于 2 比特的 Llama-3.1-8B-Instruct 权重，StrataFlux 将 WikiText-2 困惑度从 9.46 降低到 8.40，同时保持……

    arXiv:2611.08363v1 Announce Type: new  Abstract: Quantization reduces the memory and bandwidth demands of large language model (LLM) inference but can degrade accuracy; offloading higher-precision data can instead stall generation. We introduce StrataFlux, a framework that challenges this tradeoff by recovering accuracy lost to low-bit weight and key-value (KV) cache quantization while retaining quantization's memory and speed advantages. StrataFlux co-designs refinement representation, selection, and delivery to make host RAM and solid-state drives useful sources of additional precision. Its selective precision streaming reduces transfer volume, overlaps delivery with computation, and reuses fetched precision across decoding steps so generation need not wait for every transfer. We evaluate 2-4-bit weights and 2-3-bit KV-cache on dedicated GPUs and unified-memory platforms. For 2-bit Llama-3.1-8B-Instruct weights, StrataFlux reduces WikiText-2 perplexity from 9.46 to 8.40 while maintai
    
[^23]: 统一中心流、杆流与边缘耦合

    Unifying Central Flow, Rod Flow, and the Edge Coupling

    [https://arxiv.org/abs/2611.08358](https://arxiv.org/abs/2611.08358)

    本文提出“边缘耦合”这一泛函，证明中心流与杆流均可由它推导得出，二者分别对应不同时间尺度上的时间粗粒化，从而统一了刻画梯度下降稳定性边缘行为的两种连续时间模型。

    

    在稳定性边缘，梯度下降的迭代点会沿着Hessian矩阵的尖锐方向发生振荡。这对连续时间模型构成了重大挑战，因为通常的连续时间近似——梯度流——并不允许振荡。已有若干连续时间模型被提出来刻画稳定性边缘处的行为，其中包括中心流和杆流。尽管这两种模型都可以被解释为在时间上执行粗粒化，但它们之间的联系仍然知之甚少。本文证明，这两种流都可以从第三个对象——边缘耦合——推导出来：边缘耦合是定义在相邻迭代点上的一个泛函，其临界点正是梯度下降的不动点和周期二轨道。研究揭示，这两种流对应于不同时间尺度上的粗粒化：杆流仅在最快的时间尺度上进行粗粒化，而中心流则在快……

    arXiv:2611.08358v1 Announce Type: cross  Abstract: At the edge of stability, the iterates of gradient descent oscillate along the sharp directions of the Hessian. This poses a significant challenge for continuous-time models as the usual continuous-time approximation -- the gradient flow -- admits no oscillations. Several continuous-time models have been proposed to capture the behavior at the edge of stability, among them central flow and rod flow. Although both models can be interpreted as performing a coarse-graining in time, the connection between them remains poorly understood. Here we show that both flows can be derived from a third object, the edge coupling: a functional on consecutive iterates whose critical points are the fixed points and period-two orbits of gradient descent. It is revealed that the two flows correspond to coarse-grainings at different time scales. Rod flow coarse-grains over only the fastest time scale, while central flow coarse-grains over both the fast and
    
[^24]: USB Gym：一个用于闭环超声视图匹配的Gymnasium测试平台

    USB Gym: A Gymnasium Testbed for Closed-Loop Ultrasound View Matching

    [https://arxiv.org/abs/2611.08352](https://arxiv.org/abs/2611.08352)

    提出了USB Gym——一个基于公开CT数据模拟超声的闭环超声视图匹配标准化测试平台，通过可替换插槽机制系统评估并组合20种已发表方法，为超声引导研究建立了统一可比较的基准。

    

    背景与目标：我们的主要目标是超声引导，即移动超声探头直到其图像与目标视图相匹配。目前已发表的方法中，引导系统的各个部分通常在不同的模拟器中分别评估。我们想探究的问题是：将这些方法组合成一个完整的引导系统后，是否足够好用。方法：我们构建了USB Gym测试平台，其标准引擎基于公开的患者CT数据模拟超声图像，并利用一个从CT标签中提取器官纹理的网络。该引导系统包含三个可替换的部件（插槽）：器官分割器、控制器和奖励函数，共纳入了20种已发表的方法，每种方法均在相同的六名留出患者的60个回合上运行。在每个插槽中，只有当已发表的方法在这些患者上表现出可检测的更优性能时，才会替换我们原有的方法。将这一最优组合与我们原始的引导系统在20个新CT数据上的表现进行比较，其中较优者作为基准基线。此外，一个姿态神谕读取仅在仿真环境中才能获得的精确探头姿态，作为参考标准。以下结果为探索性结果。

    arXiv:2611.08352v1 Announce Type: cross  Abstract: Background and Objective: Our main goal is ultrasound guidance: moving a probe until its image matches a target view. Published methods for each part of a guide are evaluated in separate simulators. We asked whether these methods, combined into one guide, are good enough.   Methods: We built USB Gym, a testbed whose standard engine simulates ultrasound from public patient CT, with a network drawing organ textures from CT labels. Its guide has three replaceable parts (slots), an organ segmenter, a controller and a reward, which received 20 published methods, each run on the same 60 episodes of six held-out patients. In each slot, a published method replaced ours only if detectably better on these patients. The better of this combination and our original guide on 20 new CTs is the benchmark baseline. A pose oracle, reading the exact probe pose available only in simulation, set the reference. The results below are exploratory.   Results: 
    
[^25]: 面向毫米波雷达微多普勒可微渲染的四维点溅射（4D Point Splatting）

    4D Point Splatting for mmWave Radar Micro-Doppler Differentiable Rendering

    [https://arxiv.org/abs/2611.08342](https://arxiv.org/abs/2611.08342)

    提出了4D点溅射（4DPS），一种基于物理的可微渲染器，通过将携带材质、法线和速度信息的定向点溅射到距离-多普勒域，将显式运动场景拟合到真实毫米波雷达微多普勒采集数据上，输出原始复数回波以缩小雷达感知的sim2real差距并缓解雷达数据稀缺问题。

    

    毫米波雷达能够将运动物体的微米级位移解析为微多普勒信号，从而实现步态、手势、活动、姿态及生命体征感知。每个任务都依赖于在真实数据上训练的下游模型，然而雷达数据集的规模比视觉和语言领域落后了数个数量级。单纯依靠大规模采集无法弥补这一差距，而现有的合成方法（仅前向的模拟器、学习型生成器以及静态场景逆渲染器）都无法拟合真实采集的运动人体数据，导致存在仿真到现实的鸿沟（sim2real gap）。我们提出了四维点溅射（4DPS），这是一种基于物理的可微渲染器，能够将显式的运动场景拟合到真实采集数据上。利用同步相机对房间和人体进行初始化，我们将场景表示为携带材质、法线和速度信息的定向点，并将每个点溅射到距离和多普勒频谱单元中。我们的输出是真实雷达所记录的原始复数回波，因此我们在其上应用与真实雷达相同的信号处理流程（原文摘要在此处截断）。

    arXiv:2611.08342v1 Announce Type: cross  Abstract: Millimeter-wave radar resolves the micrometer displacements of a moving body as micro-Doppler, enabling gait, gesture, activity, pose and vital-sign sensing. Each task relies on a downstream model trained on real data, yet radar datasets lag orders of magnitude behind vision and language. Brute-force collection will not close that gap, and existing synthesis methods (forward-only simulators, learned generators and static-scene inverse renderers) cannot fit a real capture of a moving body, leaving a sim2real gap. We present 4D Point Splatting (4DPS), a physics-based differentiable renderer that fits an explicit moving scene to a real capture. Using the synchronized camera to initialize the room and body, we represent the scene as oriented points carrying material, normal and velocity, and splat each point into range and Doppler bins. Our output is the raw complex return a real radar records, so we apply the same signal processing on the
    
[^26]: 发现能够塑造他人的学习规则

    Discovering Learning Rules that Shape Others

    [https://arxiv.org/abs/2611.08335](https://arxiv.org/abs/2611.08335)

    该论文提出DUOS方法，将元学习发现的更新规则框架从单智能体强化学习扩展到一般和博弈中的对手塑造，无需访问对手的参数、策略、奖励或学习规则即可实现对手塑造。

    

    在一般和博弈中，自利的学习者可能达成对所有人都不利的结果，这促使了塑造对手学习方法的发展。诸如“具有对手学习意识的学习”（LOLA，通过对对手的学习过程进行微分）和“无模型对手塑造”（M-FOS，使用无模型优化方法进行长时程对手塑造）等方法，已经能够实现相互合作以及对学习者的利用。然而，这些方法在部署时需要访问对手的参数、策略、奖励或学习规则。在单智能体强化学习中，通过元学习发现的更新规则已经取得了最先进的成果（例如DiscoRL）。这些规则与它们所训练的策略相互分离，并且只读取智能体自身的经验。我们在此通过引入“用于对手塑造的发现更新规则”（DUOS）将这一框架扩展到一般和博弈设置中。据我们所知……

    arXiv:2611.08335v1 Announce Type: new  Abstract: In general-sum games, self-interested learners can reach outcomes that are poor for everyone, which has motivated methods that shape the opponent's learning. Methods like Learning with Opponent-Learning Awareness (LOLA), which differentiates through the opponent's learning process, and Model-Free Opponent Shaping (M-FOS), which uses model-free optimisation methods for long-horizon opponent shaping, have enabled mutual cooperation and exploitation of learners. However, these methods require access to the opponent's parameters, policy, rewards or learning rule during deployment. In single-agent reinforcement learning, update rules discovered by meta-learning have achieved state-of-the-art results (e.g. DiscoRL). These rules are separate from the policy they train and read only the agent's own experience. We hereby extend this framework to general-sum settings by introducing Discovered Update Rule for Opponent Shaping (DUOS). To the best of
    
[^27]: 在异构认知与生物医学知识图谱上对大型语言模型进行基准测试：图谱接地在多跳推理与幻觉缓解中的作用

    Benchmarking Large Language Models on Heterogeneous Cognitive and Biomedical Knowledge Graphs: The Role of Graph Grounding in Multi-Hop Reasoning and Hallucination Mitigation

    [https://arxiv.org/abs/2611.08315](https://arxiv.org/abs/2611.08315)

    该研究在五个认知与生物医学知识图谱上对五个大型语言模型进行基准测试，发现通用模型在多跳推理任务上优于领域专用模型，且知识图谱接地能有效降低所有模型的幻觉率。

    

    大型语言模型（LLM）在生物医学和认知神经科学领域的应用日益广泛，但其在异构知识图谱（KG）上进行可靠推理的能力仍不明确。我们对五个大型语言模型（GPT-5.1、Gemini 2.0 Pro、Llama 3、Qwen 和 BioGPT）在五个知识图谱上进行了基准测试：DrugBank、NeuroSynth、Cognitive Atlas、Brain Map 和 GDPKG。我们引入了一个多层次查询数据集，用于在知识图谱感知（图谱接地）和知识图谱无关两种条件下评估推理深度和事实准确性。标注者间一致性较高，Cohen's kappa 为 0.781（范围：0.753-0.807），Fleiss' kappa 为 0.77。结果表明，通用模型（尤其是 GPT-5.1 和 Gemini 2.0 Pro）在多跳和高级推理任务上通常优于领域专用模型，但随着查询复杂度的增加，性能会下降。图谱接地降低了所有被评估模型的幻觉率。与我们的初始假设相反……（摘要在此处被截断）

    arXiv:2611.08315v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used in biomedical and cognitive neuroscience, but their ability to reason reliably over heterogeneous knowledge graphs (KGs) remains unclear. We benchmark five LLMs (GPT-5.1, Gemini 2.0 Pro, Llama 3, Qwen, and BioGPT) across five KGs: DrugBank, NeuroSynth, Cognitive Atlas, Brain Map, and GDPKG. We introduce a multitier query dataset to evaluate reasoning depth and factual accuracy under KG-aware (graph-grounded) and KG-agnostic conditions. Inter-annotator agreement was substantial, with Cohen's kappa of 0.781 (range: 0.753-0.807) and Fleiss' kappa of 0.77. Results show that general-purpose models, particularly GPT-5.1 and Gemini 2.0 Pro, generally outperform domain-specific models on multi-hop and high-level reasoning tasks, although performance declines as query complexity increases. KG grounding reduces hallucination rates across all evaluated models. Contrary to our initial hypothesis, do
    
[^28]: 从少量轨迹中学习集体运动的可解释多体相互作用规律

    Interpretable many-body interaction laws for collective motion from few trajectories

    [https://arxiv.org/abs/2611.08314](https://arxiv.org/abs/2611.08314)

    本文提出方向原子簇展开（dACE）方法，将集体运动中的社会相互作用严格分解为一体、二体、三体等多体贡献，仅需少量轨迹数据即可逐阶恢复相互作用的体阶规律，并在二维和三维模拟群体及真实鱼群数据上全面优于现有神经方法SocialNet。

    

    集体运动模型通常将个体间的社会相互作用简化为两体力，尽管已有实证证据表明存在多体效应。我们提出了方向原子簇展开（dACE），该方法将原子间势中的原子簇展开（ACE）适配到自推进智能体上。dACE提供了一个可系统性改进、严格等变的智能体加速度展开形式，能够唯一地分解为一体、二体、三体及更高阶的贡献，并包含一个刻画智能体自身过去运动的自项，该自项的拟合方式确保其不会吸收相互作用的影响。在具有已知规则的二维和三维模拟群体中，dACE能够从少量数据中逐阶恢复相互作用规律；而目前最接近的竞争对手——神经模型SocialNet——则需要更多数据，在多体规则或手性规则上误差大得多，且仅限于二维。在斑马鱼和金鲦鱼的真实追踪数据集上，该规律……

    arXiv:2611.08314v1 Announce Type: new  Abstract: Models of collective motion usually reduce social interactions to pairwise forces, despite empirical evidence for many-body effects. We introduce the directional atomic cluster expansion (dACE), which adapts the atomic cluster expansion (ACE) of interatomic potentials to self-propelled agents. dACE provides a systematically improvable, exactly equivariant expansion of an agent's acceleration that splits uniquely into 1-, 2-, 3-body and higher contributions, along with a self term for the agent's own past motion that is fitted so that it does not absorb the interactions. On simulated collectives in two and three dimensions with known rules, dACE recovers the interaction law body order by body order from little data, while the neural model SocialNet, the current closest competitor, needs more data, has much larger errors on many-body or chiral rules and is restricted to 2D. On real data sets of tracked zebrafish and golden shiners, the law
    
[^29]: RM-VLM：从自然语言任务描述到视觉接地的奖励机

    RM-VLM: From Natural-Language Task Descriptions to Vision-Grounded Reward Machines

    [https://arxiv.org/abs/2611.08308](https://arxiv.org/abs/2611.08308)

    本文提出 RM-VLM，利用大语言模型从自然语言任务描述自动生成奖励机，并让视觉-语言模型充当标注函数将命题接地于图像观测，从而免除传统奖励机中需要针对每个环境手动设计标注函数的局限。

    

    奖励机（Reward Machines, RMs）被提出作为强化学习（RL）中的一种奖励形式化方法，用于捕捉信用分配依赖于历史的非马尔可夫任务。通过将奖励函数表示为自动机，RM 揭示了任务的时间结构，并使得此类奖励可以在马尔可夫框架内处理。然而，RM 假设存在一个标注函数，即一个针对观测的命题接地预言机，该函数必须针对每个环境手动设计，这限制了其更广泛的适用性。在这项工作中，我们通过提出 RM-VLM 来解决这一局限，这是一种新方法：由大语言模型（LLM）根据自然语言任务描述生成 RM，并由视觉-语言模型（VLM）充当标注函数，在 RM 引导的策略学习过程中将 RM 的命题接地于基于图像的观测之上。同时，基于 VLM 的价值函数估计器被用于评估智能体向每个 RM 转移的进展（摘要原文在此处截断）。

    arXiv:2611.08308v1 Announce Type: new  Abstract: Reward machines (RMs) have been proposed as a reward formulation in reinforcement learning (RL) to capture non-Markovian tasks where credit assignment depends on history. By representing reward functions as automata, RMs expose the task's temporal structure and allow such rewards to be handled within a Markovian framework. However, they assume a labelling function, a proposition-grounding oracle for observations, which must be manually engineered for each environment, limiting their wider applicability. In this work, we address this limitation by proposing RM-VLM, a novel approach in which a large language model (LLM) generates an RM from a natural-language task description, and a vision-language model (VLM) acts as the labelling function, grounding the RM's propositions over image-based observations during RM-guided policy learning. A VLM-based value function estimator is used to assess the agent's progress towards each RM transition an
    
[^30]: 面向扩散语言模型并行解码的测量耦合方法

    Measured Coupling for Parallel Decoding in Diffusion Language Models

    [https://arxiv.org/abs/2611.08307](https://arxiv.org/abs/2611.08307)

    该论文提出利用主前向传播中的廉价探针直接测量token间的耦合强度（即提交一个token对其他位置置信度的影响），以替代平均场解码中的相似性代理指标，使扩散语言模型并行解码在平均仅损失1.1个点精度的代价下大幅减少解码步数并提升速度。

    

    扩散语言模型可以在每一步生成多个token，但必须选择哪些token可以安全地一起提交。平均场解码会对预测结果相似的位置对进行惩罚。我们用耦合本身来替代这一代理指标：即提交一个token会在多大程度上改变模型对另一个位置的置信度。主前向传播内部的廉价探针即可测量该耦合，无需训练或额外模型，并且可直接接入相同的平均场规则。在三个模型和四个基准测试上，我们的方法所需步数远少于平均场解码，且在大多数设置下（包括使用键值缓存时）速度更快，平均精度损失仅为1.1个百分点。若打乱耦合或将其替换为统一的奖励，这一增益将消失。

    arXiv:2611.08307v1 Announce Type: cross  Abstract: Diffusion language models can write many tokens per step, but must choose which tokens are safe to commit together. Mean-field decoding penalizes pairs of positions whose predictions look alike. We replace this proxy with the coupling itself: how much committing one token changes the model's confidence at another. Cheap probes inside the main forward pass measure it, with no training or extra model, and it plugs into the same mean-field rule. On three models and four benchmarks, our method needs far fewer steps than mean-field decoding and is faster in most settings, also with a key-value cache, at an average accuracy cost of 1.1 points. Shuffling the couplings or replacing them with a uniform bonus loses this gain.
    
[^31]: 基于延迟等价的动态乐观记忆约束在线凸优化

    Dynamic Optimistic Constrained OCO with Memory via Delay Equivalence

    [https://arxiv.org/abs/2611.08302](https://arxiv.org/abs/2611.08302)

    该论文提出了一种结合不可信预测的乐观算法，通过将记忆归约为延迟来求解带记忆和时变约束的在线凸优化（COCO-M），首次建立了同时适应环境非平稳性与预测准确性的动态遗憾和累积约束违反界，并去除了函数可分离性等限制性假设。

    

    arXiv:2611.08302v1 公告类型：新论文 摘要：带记忆和时变约束的在线凸优化（COCO-M）是最近提出的一个框架，用于刻画带预算约束的有状态在线学习和非随机控制中的动态特性。尽管COCO-M具有很强的表达能力，但其研究仍然严重不足。在这项工作中，我们全面地解决了这一问题，提供了一种乐观算法，该算法融合了不可信的预测来控制通用动态遗憾和累积约束违反。我们的分析利用了从记忆到延迟的有力归约，并将状态相关的梯度视为延迟反馈。我们建立了首个能够同时适应环境非平稳性（路径长度 $P_T$）和预测准确性的界，同时去除了先前诸如函数可分离性等限制性假设。在获得这些结果的过程中，我们还推导出了一些具有独立价值的发现，涉及记忆到延迟的归约以及带时变约束的延迟OCO。

    arXiv:2611.08302v1 Announce Type: new  Abstract: Online Convex Optimization with memory and time-varying constraints (COCO-M) is a recently-introduced framework that captures the dynamics of stateful online learning and non-stochastic control with budget constraints. Despite its expressive power, COCO-M remains largely understudied. In this work, we tackle this problem in its full scale, providing an optimistic algorithm that incorporates untrusted predictions to tame universal dynamic regret and cumulative constraint violation. Our analysis leverages a potent reduction from memory to delay and treats state-dependent gradients as delayed feedback. We establish the first bounds that adapt to both environment non-stationarity (path length $P_T$) and prediction accuracy, while lifting previous restrictive assumptions such as functional separability. En route to these results, we derive findings of independent interest for the memory-to-delay reduction, and for delayed OCO with time-varyin
    
[^32]: 基于Transformer的面向sEMG优化的机器人介导康复轨迹处方方法

    Transformer-Based Trajectory Prescription for sEMG-Optimizing Robot-Mediated Rehabilitation

    [https://arxiv.org/abs/2611.08295](https://arxiv.org/abs/2611.08295)

    该论文提出了一个可个性化的Transformer模型，利用旋转位置编码和流匹配，将康复机器人训练中用户的表面肌电肌肉激活映射为接触力轨迹处方，从而实现以神经运动激活优化为目标的机器人介导康复。

    

    机器人介导的康复具有提供高强度、个性化治疗的潜力，但现有系统（并未持续改善治疗效果）通常仅规定运动学轨迹，而非明确针对被认为能够支持恢复的神经运动激活。为填补这一空白，并最终实现以激活优化为目标的康复机器人行为，我们提出了一个可个性化的基于Transformer的模型，将个体的肌肉激活（通过表面肌电图测量）映射到与末端执行器康复机器人交互时规定的接触力轨迹。该模型使用采用旋转位置编码的Transformer编码器-解码器来生成与轨迹相关且用户特定的神经运动因素的潜在表示，并通过流匹配来增强轨迹预测，从而产生偏向于目标激活模式的合理处方分布。

    arXiv:2611.08295v1 Announce Type: cross  Abstract: Robot-mediated rehabilitation has the potential to deliver high-intensity, individualized therapy, but existing systems (which have not consistently improved outcomes) typically prescribe kinematic trajectories rather than explicitly targeting the neuromotor activations thought to support recovery. To address this gap, and ultimately enable activation-optimizing rehabilitation robot behavior, we present a personalizable transformer-based model mapping individuals' muscle activations (as measured via surface electromyography) to prescribed contact force trajectories when interacting with an end-effector rehabilitation robot. The model uses a transformer encoder-decoder leveraging rotary positional embeddings (RoPE) to generate a latent representation of trajectory-relevant and user-specific neuromotor factors, and augments trajectory prediction with flow matching to produce a distribution over plausible prescriptions biased toward the t
    
[^33]: AutoCompress：自动化硬件引导的模型压缩

    AutoCompress: Automated Hardware-Guided Model Compression

    [https://arxiv.org/abs/2611.08290](https://arxiv.org/abs/2611.08290)

    AutoCompress是一个硬件引导的模型压缩框架，它将延迟、吞吐量和内存等硬件与工作负载约束直接纳入可微架构搜索，联合优化全局与逐层结构以生成异构压缩架构，并利用经过实测验证的服务引擎成本模型引导整个优化过程。

    

    大语言模型（LLM）剪枝必须在满足硬件和工作负载特定的延迟、吞吐量和内存约束的同时保持模型质量。现有的剪枝流程通常以参数数量为目标，要求从业者手动选择满足延迟、吞吐量和GPU内存限制的模型规模。然而，仅凭参数数量并不能预测这些指标，因为这些指标取决于模型架构、工作负载、硬件和服务引擎。我们提出了AutoCompress，这是一个硬件引导的压缩框架，它将此类约束直接纳入可微架构搜索中。AutoCompress联合优化全局和逐层结构以生成异构架构，其优化过程由经过端到端延迟、吞吐量和内存实测验证的服务引擎成本模型引导。我们在长上下文本地编码、批量数据中心服务和延迟敏感的游戏内交互场景中评估了AutoCompress。与标准（基线相比，……原文在此处截断）

    arXiv:2611.08290v1 Announce Type: cross  Abstract: LLM pruning must preserve quality while meeting hardware- and workload-specific constraints on latency, throughput, and memory. Existing pruning workflows often target parameter counts, requiring practitioners to manually select model sizes that meet latency, throughput, and GPU memory limits. However, parameter count alone does not predict these metrics, which depend on the model architecture, workload, hardware, and serving engine. We introduce AutoCompress, a hardware-guided compression framework that directly incorporates such constraints into differentiable architecture search. AutoCompress jointly optimizes global and per-layer structure to produce heterogeneous architectures, guided by serving-engine cost models validated against end-to-end latency, throughput, and memory measurements. We evaluate AutoCompress on long-context local coding, batched datacenter serving, and latency-sensitive in-game interaction. Compared to the sta
    
[^34]: 基于大规模并行加速的安全强化学习

    Accelerating Safe Reinforcement Learning with Massive Parallelism

    [https://arxiv.org/abs/2611.08273](https://arxiv.org/abs/2611.08273)

    该论文提出相位聚合成本估计器与约束收紧框架，解决了大规模并行短回合训练与全回合安全约束之间的不匹配问题，实现了可扩展的安全强化学习加速训练。

    

    我们研究了将安全强化学习扩展到大规模并行训练场景的问题，在这种场景中，大量环境以较短的滚动回合长度被执行。尽管这种设置提高了数据吞吐量和实际运行效率，但它与约束马尔可夫决策过程（CMDP）存在不匹配，因为后者的安全约束是定义在整个回合的成本回报之上的。我们通过将交错的环境重置分析为短轨迹段上的相位混合分布来解决这一不匹配问题。该分析刻画了由相位覆盖所引起的偏差，并启发了一种相位聚合成本估计器，可以从短滚动批次中重建回合级别的成本估计。我们进一步将有限样本不确定性和分布不匹配纳入约束收紧框架，从而得到一种与同策略优化兼容的可扩展安全强化学习方法。基于 MJX 的实现……

    arXiv:2611.08273v1 Announce Type: cross  Abstract: We investigate the problem of scaling safe reinforcement learning (safe RL) to massively parallel training regimes, where a large number of environments are exe- cuted with short rollout horizons. Although this setting improves data throughput and wall-clock efficiency, it creates a mismatch with constrained Markov decision processes (CMDPs), whose safety constraints are defined over full-episode cost returns. We address this mismatch by analyzing staggered environment resets as a phase-mixture distribution over short trajectory segments. This analysis charac- terizes the bias induced by phase coverage and motivates a phase-aggregated cost estimator that reconstructs episode-level cost estimates from short rollout batches. We further incorporate finite-sample uncertainty and distribution mismatch into a constraint-tightening framework, yielding a scalable Safe RL method compatible with on-policy optimization. Using an MJX-based impleme
    
[^35]: DART：面向通用异常检测的领域无关残差迁移

    DART: Domain-Agnostic Residual Transfer for Generalist Anomaly Detection

    [https://arxiv.org/abs/2611.08271](https://arxiv.org/abs/2611.08271)

    DART提出了一种即插即用框架，通过残差解耦模块与跨领域元训练，将正常-异常参考残差分离为异常相关成分与领域特定成分，从而提升通用异常检测的跨域迁移能力。

    

    通用异常检测（GAD）旨在仅使用少量参考样本来检测未见领域中的异常。近期的正常-异常引导方法通过利用异常参考样本改进了这一设置，但其残差往往将可迁移的异常线索与领域特定的外观因素（如纹理、几何形状和采集条件）混杂在一起。这种残差的领域纠缠限制了跨领域迁移，并可能在新的领域上产生虚假激活。我们提出DART，这是一个面向正常-异常引导GAD的即插即用框架，它学习将残差分离为异常相关成分和领域特定成分。DART将残差解耦模块（在保留异常判别信息的同时抑制领域信息）与跨领域元训练（在源域之间构建训练片段以使这种分离有效）相结合。该方法与架构无关，可以插入到残差特征提取之间……

    arXiv:2611.08271v1 Announce Type: cross  Abstract: Generalist Anomaly Detection (GAD) aims to detect anomalies in unseen domains using only a few reference samples. Recent normal-abnormal guided methods improve this setting by exploiting abnormal references, but their residuals often mix transferable anomaly cues with domain-specific appearance factors such as texture, geometry, and acquisition conditions. This residual domain entanglement limits cross-domain transfer and may produce spurious activations on novel domains. We propose DART, a plug-in framework for normal-abnormal guided GAD that learns to separate residuals into anomaly-relevant and domain-specific components. DART combines a Residual Disentanglement Module, which preserves anomaly-discriminative information while suppressing domain information, with Cross-Domain Meta-Training, which forms episodes across source domains to make the separation effective. The method is architecture-agnostic and can be inserted between resi
    
[^36]: 通过解耦行为预训练实现神经运动解码的规模化

    Scaling Neural Motor Decoding via Decoupled Behavioral Pretraining

    [https://arxiv.org/abs/2611.08263](https://arxiv.org/abs/2611.08263)

    提出BEEMO框架，通过解耦行为与神经表征学习，利用丰富且易获取的非配对行为数据进行训练，独立扩展行为和神经监督，从而在非人类灵长类皮层内运动数据集上显著提升解码性能。

    

    最近的神经解码方法通过在大量配对的神经-行为记录数据集上训练获得了强大的性能，但这类数据集昂贵、具有侵入性且难以规模化。相比之下，行为数据丰富且易于收集，这引发了一个问题：神经解码能否从解耦行为与神经表征学习中受益。我们提出了BEEMO，这是一个灵活的多会话框架，支持使用任意比例配对的神经-行为记录与非配对行为数据进行训练。行为解码器直接从行为轨迹中学习运动动态，而神经活动在可用时通过交叉注意力层被纳入，从而使行为和神经监督能够独立扩展。在多个非人类灵长类动物的皮层内运动数据集和任务上，我们展示了引入非配对行为数据能够提升解码性能。

    arXiv:2611.08263v1 Announce Type: cross  Abstract: Recent neural decoding approaches achieve strong performance by training on large collections of paired neural-behavioral recordings, but such datasets are expensive, invasive, and difficult to scale. In contrast, behavioral data is abundant and easy to collect, raising the question of whether neural decoding can benefit from decoupling behavioral and neural representation learning. We introduce BEEMO, a flexible multi-session framework that enables training with arbitrary mixtures of paired neural-behavioral recordings and unpaired behavioral data. A behavior decoder learns movement dynamics directly from behavioral trajectories, while neural activity is incorporated through cross-attention layers when available, allowing behavioral and neural supervision to scale independently. Across multiple intracortical motor datasets and tasks in nonhuman primates, we show that incorporating unpaired behavioral data improves decoding performance
    
[^37]: 基于地球静止卫星图像的热带气旋风场与强度临近预报

    Nowcasting of Tropical Cyclone Wind Fields and Intensity from Geostationary Imagery

    [https://arxiv.org/abs/2611.08255](https://arxiv.org/abs/2611.08255)

    本文提出基于地球静止卫星图像的简单神经网络，通过在模型潜空间联合预测二维风场与风暴尺度强度指标，显著改善了最大风速及其半径的估计效果，尤其适用于快速增强事件。

    

    由合成孔径雷达（SAR）观测获得的高分辨率热带气旋地表风速可以支持风暴结构与强度建模，但其在空间和时间上仍然稀疏，且无法近实时获取。地球静止卫星能够在空间和时间上连续观测云顶，但无法直接解析地表风。本文提出了简单的神经网络，可从地球静止卫星图像预测二维风场和风暴尺度强度指标。通过一系列消融实验，我们发现从模型潜空间联合预测二维场和标量指标能够改善最大风速（及其半径）的估计效果，尤其是在快速增强事件中。

    arXiv:2611.08255v1 Announce Type: new  Abstract: High-resolution tropical cyclone surface wind speeds derived from synthetic aperture radar (SAR) observations can support storm structure and intensity modelling, but remain spatially and temporally sparse and unavailable near-real-time. Geostationary satellites capture cloud tops continuously in space and time, but do not directly resolve surface winds. Here, we present simple neural networks to predict 2D wind fields and storm-scale intensity metrics from geostationary imagery. Through a series of ablation studies, we find that joint prediction of 2D fields and scalar metrics from the model latent space improves the retrieval of the (radius of) maximum wind, especially for rapid intensification events.
    
[^38]: 面向全局时间序列预测的序列级交叉验证

    Series-wise Cross-validation for Global Time Series Forecasting

    [https://arxiv.org/abs/2611.08252](https://arxiv.org/abs/2611.08252)

    该论文为基于深度学习的全局时间序列预测模型形式化了按序列划分的交叉验证方法，突破传统仅按时间划分的范式，并在11个数据集和10种神经网络架构上系统评估了K折、自助法等多种序列级交叉验证策略。

    

    我们研究时间序列预测中的交叉验证（CV）问题，重点关注基于深度学习的全局预测模型。预测领域主流的交叉验证策略是按时间划分数据，将每个序列的最后部分用于测试。然而，对于在时间序列集合上训练的全局模型而言，一种根本不同的方法成为可能：按序列和时间同时进行划分。我们为全局预测模型形式化了序列级交叉验证，并在11个数据集和10种神经网络架构上对多种实现方式进行了基准测试，包括K折交叉验证、自助法和蒙特卡洛交叉验证。我们从五个维度评估交叉验证程序：性能估计、过度乐观率、模型选择准确性、遗憾值以及下游预测性能。我们从训练数据中随机抽取一部分序列予以保留，使其仅出现在测试集中。该协议模拟了新出现序列尚未……（摘要在此处截断）

    arXiv:2611.08252v1 Announce Type: cross  Abstract: We consider the problem of cross-validation (CV) in time series forecasting, with a focus on global forecasting models based on deep learning. The prevailing CV strategy in forecasting is to partition data across time with the last part of each series used for testing. With global models trained on collections of time series, however, a fundamentally different approach becomes possible: partitioning across series and time. We formalize series-wise CV for global forecasting models and benchmark several instantiations, including K-fold CV, bootstrapping, and Monte Carlo CV, on 11 datasets with 10 neural network architectures. We evaluate CV procedures along five dimensions: performance estimation, overoptimism rate, model selection accuracy, regret, and downstream forecasting performance. We withhold a random fraction of series from training so that they appear only in the test set. This protocol simulates the emergence of new series not
    
[^39]: 几何相关性并不足够：评估扩散对接中的聚焦式后训练监督

    Geometric Relevance Is Not Sufficient: Evaluating Focused Post-Training Supervision in Diffusion Docking

    [https://arxiv.org/abs/2611.08249](https://arxiv.org/abs/2611.08249)

    该研究通过固定预算实验发现，在DiffDock扩散分子对接的后训练中，聚焦于几何相关校正的监督策略相比广泛的延续训练并无一致优势，广泛的延续训练反而取得了最高的Top-1对接成功率。

    

    基于扩散的分子对接需要学习一个跨越广泛损坏分布的校正得分场，然而单个姿态误差在结构后果上可能差异巨大。这在“保持广泛的得分场覆盖”与“将监督集中于与对接几何更相关的校正”之间造成了后训练上的冲突。本研究使用DiffDock进行固定预算实验，比较了广泛的延续训练与聚焦于笛卡尔后果或条件目标一致性的扭转监督，以及广泛/聚焦混合方案。评估将固定得分保真度、生成姿态质量、最优10次采样能力以及基于置信度选择的Top-1对接性能分开考量。在小于5埃的阈值下，广泛延续训练在受训策略中拥有最高的平均Top-1成功率，而聚焦式或混合监督在所测试的预算下并未显示出一致的优势。起始模型具有更好的固定得分保真度……（摘要原文在此处截断）

    arXiv:2611.08249v1 Announce Type: new  Abstract: Diffusion-based molecular docking requires learning a corrective score field across a broad corruption distribution, yet individual pose errors can differ substantially in their structural consequences. This creates a post-training conflict between preserving broad score-field coverage and concentrating supervision on corrections that appear more relevant to docking geometry. Using DiffDock, a fixed-budget study compares broad continuation with torsional supervision focused on Cartesian consequence or conditional target consistency, and with a broad/focused mixture. Evaluation separates fixed score fidelity, generated-pose quality, best-of-10 sampling capacity, and confidence-selected Top-1 docking performance. At <5 Angstrom, broad continuation had the highest mean Top-1 success among trained policies, and focused or mixed supervision showed no consistent advantage under the tested budget. The starting model had better fixed score fidel
    
[^40]: FewStepRL：基于强化学习的少步扩散模型微调

    FewStepRL: Fine-tuning Few-step Diffusion Models with Reinforcement Learning

    [https://arxiv.org/abs/2611.08240](https://arxiv.org/abs/2611.08240)

    该论文提出FewStepRL方法，通过变换、分解与适配三个阶段系统地将强化学习目标适配到少步扩散模型，解决了现有RL目标面向瞬时速度而少步模型建模区间平均速度的不匹配问题，同时显著降低了训练成本。

    

    强化学习（RL）已成为扩散模型的标准后训练方法，但大多数RL目标在每次迭代中都需要对当前策略进行rollout采样，而在多步采样器下，这些rollout通常占据了训练成本的主导地位。少步模型虽然可以削减这一成本，但它们建模的是区间上的平均速度，而现有的RL目标则是为多步采样器下的瞬时速度设计的。我们分三个阶段系统地将RL目标适配到少步模型，即变换、分解和适配。变换到x空间使我们考虑的每个目标都能通过一个共同的策略预测来表达——默认在干净端点处，对于转移目标或流映射目标则在显式标记的层级处——从而使在密度、速度或奖励上表述的各类目标变得可比较。分解则将这些目标的梯度写成自回归项之和……（摘要在此处被截断）

    arXiv:2611.08240v1 Announce Type: new  Abstract: Reinforcement learning (RL) has become a standard post-training method for diffusion models, but most RL objectives roll out the current policy at every iteration, and with a many-step sampler these rollouts typically dominate the cost of training. Few-step models cut this cost, yet they model an average velocity over an interval, whereas existing RL objectives are built for an instantaneous velocity with a many-step sampler. We adapt RL objectives to few-step models systematically in three phases, namely a transformation, a decomposition and an adaptation. The transformation to $x$-space expresses each objective we consider through a common policy prediction, at the clean end by default and at an explicitly marked level for transition or flow-map objectives, so that objectives stated over densities, velocities or rewards become comparable. The decomposition writes the gradient of each of these objectives as the sum of a self-regression 
    
[^41]: 超越二次注意力：面向ODE视觉Transformer的可分离高斯算子

    Beyond Quadratic Attention: Separable Gaussian Operators for ODE Vision Transformers

    [https://arxiv.org/abs/2611.08236](https://arxiv.org/abs/2611.08236)

    SSoG用可分离的高斯和算子替代自注意力，将计算复杂度从O(N²)降至亚二次，并凭借全局Lipschitz连续性保证连续深度ODE解的唯一性，在CIFAR-100上以更少的资源将Top-1准确率提升2.2–4.1个百分点。

    

    自注意力机制的计算成本随token数量呈二次增长，这限制了视觉Transformer随输入分辨率扩展的能力。我们提出SSoG，一种可分离高斯和空间算子，用可学习的轴向分解高斯核混合来替代稠密自注意力，将每层计算成本从 $\mathcal{O}(N^2)$ 降低到关于 $N$ 的亚二次复杂度。与点积自注意力不同（其在紧致输入域之外Lipschitz常数无界），采用固定核的SSoG具有全局Lipschitz连续性，借助Picard–Lindelöf定理，这保证了连续深度变体在整个积分区间上解的唯一性——这是稠密注意力模型通常所不具备的全局理论保证。在224×224的CIFAR-100实验中，无论是在匹配的离散Transformer设置还是连续设置下，SSoG在三种模型宽度上均将Top-1准确率提升了2.2–4.1个百分点，同时节省了8.8–16.6%的[计算/参数开销]（原文摘要此处截断）。

    arXiv:2611.08236v1 Announce Type: new  Abstract: Self-attention carries a quadratic cost in token count, which limits how far Vision Transformers can scale with input resolution. We introduce SSoG, a separable sum-of-Gaussians spatial operator that replaces dense self-attention with a learned mixture of axis-factorized Gaussian kernels, reducing the per-layer cost from $\mathcal{O}(N^2)$ to subquadratic in $N$. Unlike dot-product self-attention, whose Lipschitz constant is unbounded outside a compact input domain, SSoG with fixed kernels is globally Lipschitz continuous, which, via the Picard--Lindel\"of theorem, guarantees that the continuous-depth variant has a unique solution on the whole integration interval, a global guarantee that dense-attention counterparts lack in general. On CIFAR-100 at $224\times224$, in a matched discrete-Transformer setting and in the continuous setting, SSoG improves Top-1 accuracy by 2.2--4.1 points across three model widths while using 8.8--16.6\% fewe
    
[^42]: sEMG基础模型能否提升跨受试者泛化能力？

    Do sEMG Foundation Models Improve Cross-Subject Generalisation?

    [https://arxiv.org/abs/2611.08199](https://arxiv.org/abs/2611.08199)

    本研究通过系统评估发现，sEMG基础模型的预训练优势仅在会话内评估中显著，在跨受试者泛化场景下基本消失，且在小规模受试者数据集上经典机器学习基线可匹敌甚至超越基础模型。

    

    表面肌电信号基础模型旨在学习跨多样受试者和记录条件的通用表示，以提升各种下游任务的性能。然而，现有评估难以辨别仅靠预训练是否足以克服跨受试者过拟合问题——这正是使sEMG机器学习如此具有挑战性的关键。我们对三个sEMG基础模型在三个手势分类数据集上进行了系统评估，采用会话内和跨受试者两种数据划分方式，并使用三种微调策略来隔离预训练本身的贡献。我们发现，当训练和测试样本来自同一记录会话时，预训练确实能提升性能，但在跨受试者评估下这种优势基本消失。在受试者数量较少的数据集中，预训练几乎无益，经典机器学习基线可匹敌甚至超越基础模型的性能。

    arXiv:2611.08199v1 Announce Type: new  Abstract: sEMG Foundation Models aim to learn general purpose representations over diverse subjects and recording conditions that boost performance on a variety of downstream tasks. However, existing evaluations make it difficult to discern whether pretraining alone is sufficient to combat the cross-subject overfitting which makes machine learning on sEMG so challenging. We conduct a systematic evaluation of three sEMG foundation models across three gesture classification datasets under both intra-session and inter-subject splits, using three fine-tuning strategies designed to isolate the contribution of pretraining. We find that pretraining boosts performance when train and test samples are drawn from the same recording session, but this benefit largely disappears under inter-subject evaluation. In low subject count datasets, pretraining provides little to no benefit and classical ML baselines match or exceed foundation model performance.
    
[^43]: BUMP：短暂意外的插值中段峰值

    BUMP: Brief Unexpected Mid-Interpolation Peak

    [https://arxiv.org/abs/2611.08182](https://arxiv.org/abs/2611.08182)

    本文发现了神经网络插值中的 BUMP 现象（测试精度在插值端点附近短暂提升后向中点下降），并基于此提出了将权重混合与预测空间混合相结合的新型集成方法。

    

    我们在两个置换对齐的神经网络之间进行线性插值时发现了一个有趣的现象：测试精度在插值端点附近会短暂提升，随后向中点方向下降，我们将这一现象称为 BUMP（Brief Unexpected Mid-interpolation Peak，短暂意外的插值中段峰值）。我们观察到，BUMP 在不同数据集、不同架构和不同训练方案下均一致出现，并且在早期检查点和极端剪枝等剧烈扰动下依然保持稳健。此外，受控消融实验表明，该现象无法归因于简单的正则化效应。尽管类似的近端点增益在以往的插值曲线图中有所显现，但此前并未被明确记录，也未得到系统性、深入的分析。利用这一效应，我们提出了一种简单而新颖的集成方案：权重-预测混合集成，它将权重空间的混合与预测空间的混合相结合。在我们的全部实验中，所得的集成方法始终取得一致的表现。

    arXiv:2611.08182v1 Announce Type: new  Abstract: We identify an intriguing phenomenon when linearly interpolating between permutation-aligned neural networks: test accuracy briefly improves near the interpolation endpoints before dropping toward the midpoint, which we term BUMP (Brief Unexpected Mid-interpolation Peak). We observe that BUMP appears consistently across datasets, architectures, training regimes, and remains robust under substantial perturbations such as early checkpoints and extreme pruning. Moreover, controlled ablations suggest that it cannot be attributed to trivial regularization effects. While similar near-endpoint gains are visible in prior interpolation plots, they have not been explicitly documented or systematically and thoroughly analyzed. Leveraging this effect, we propose a simple, novel ensembling scheme: weight-prediction mixed ensembling that combines weight mixing with prediction-space mixing. Across our experiments, the resulting ensembling method consis
    
[^44]: 双边贸易中利润最大化的遗憾界：学习最优定价（Posted-Price）机制

    Regret Bounds for Profit Maximization in Bilateral Trade: Learning the Best Posted-Price Mechanism

    [https://arxiv.org/abs/2611.08173](https://arxiv.org/abs/2611.08173)

    本文从遗憾最小化角度研究双边贸易中利润最大化经纪人问题，在只能观察交易是否发生的反馈下，为学习最优固定定价机制给出了多项式时间的次线性遗憾算法（含对抗情形）。

    

    双边贸易（bilateral trade）建模了在两个策略性参与者——卖方和买方——之间进行中介的任务，双方各自对交易的物品持有私有估值。我们从一个利润最大化的经纪人视角，在遗憾最小化框架下研究该问题。在每个时间步，新的卖方和买方到来，经纪人向每个参与者提出一个交易价格。当双方都接受报价时交易发生，经纪人的利润是从买方收取的价格与支付给卖方的价格之差。在每次交互结束时，学习者只能观察到交易是否发生，其目标是在事后相对于最优固定价格对实现次线性遗憾。我们首先研究该问题的对抗版本，即买方和卖方的估值由对手生成。在此设定下，我们构造了一个多项式时间的学习算法，其遗憾为 Õ(T^…（摘要在此处截断，后续内容缺失）

    arXiv:2611.08173v1 Announce Type: cross  Abstract: Bilateral trade models the task of intermediating between two strategic agents, a seller and a buyer, willing to trade a good for which they hold private valuations. We study this problem from the perspective of a profit-maximizing broker in a regret minimization framework. At each time step, a new seller and buyer arrive and the broker proposes a trading price to each agent. The trade happens if both agents accept, and the profit of the broker is the difference between the price received from the buyer and the one paid to the seller. At the end of each interaction, the learner only observes whether the trade occurred, and its goal is to achieve sublinear regret relative to the best fixed pair of prices in hindsight.   We start by investigating the adversarial version of the problem, where buyer and seller valuations are generated by an adversary. In this setting, we construct a polynomial-time learning algorithm with a $\tilde O(T^{\f
    
[^45]: 明智的补丁注意力：理解基于注意力的预测中的补丁长度

    Patch Attention Wisely: Understanding Patch Length in Attention-Based Forecasting

    [https://arxiv.org/abs/2611.08160](https://arxiv.org/abs/2611.08160)

    该论文提出了一个理论框架，解释了基于注意力的时间序列预测中补丁长度为何影响预测精度，并基于线性状态空间模型的精确分解，推导出在时间相关性下平衡表示误差与估计误差的补丁选择准则。

    

    基于注意力机制的时间序列预测模型通常将连续的观测值分组为补丁。补丁长度对预测性能有显著影响，但其选择在很大程度上仍依赖经验。我们开发了一个理论框架，解释了补丁长度为何对预测精度至关重要，并为其选择提供了有原则的依据。对于线性状态空间模型，我们推导出了预测系数矩阵的精确分解，将预测动态与滤波动态分离开来。我们证明，在长预测时域下单一预测分量占主导地位通常并不意味着秩一表示。这解释了为什么仅凭较长的预测时域并不一定会降低表示复杂度。我们将该分解与估计误差界相结合，开发了一种补丁选择准则，能够在时间相关性存在的情况下平衡表示误差与估计误差。我们的分析建立了在这些条件下……

    arXiv:2611.08160v1 Announce Type: cross  Abstract: Attention-based time series forecasters commonly group consecutive observations into patches. Patch length is known to have a substantial impact on forecasting performance, yet its selection remains largely empirical. We develop a theoretical framework that explains why patch length matters for forecasting accuracy and provides a principled basis for its selection. For linear state-space models, we derive an exact factorization of the forecasting coefficient matrix that separates forecast dynamics from filtering dynamics. We show that dominance of a single forecast component at long horizons does not generally imply a rank-one representation. This explains why longer horizons alone need not reduce representational complexity. We combine the factorization with estimation bounds to develop a patch-selection criterion that balances representation and estimation errors under temporal dependence. Our analysis establishes conditions under wh
    
[^46]: 通过上下文正则化交叉熵提升卷积网络的上下文偏移鲁棒性

    Improving Context-Shift Robustness of Convolutional Networks via Context-Regularized Cross-Entropy

    [https://arxiv.org/abs/2611.08157](https://arxiv.org/abs/2611.08157)

    本文提出新的特征归因方法 ContrastiveCAMs，并在此基础上构建上下文正则化交叉熵训练框架，通过引导模型减少对上下文区域的依赖、更多关注物体本身的特征，从而显著提升卷积神经网络在上下文偏移场景下的鲁棒性。

    

    图像分类模型在测试时常常会遇到处于多样且不可预测背景中的物体，这些背景可能与训练时所见的大不相同。这种现象被称为“上下文偏移”（context shift），它揭示了神经网络的一个关键脆弱性：当模型依赖非因果的上下文线索进行预测时，上下文的变化会导致显著的性能下降。本文提出了一种新颖的训练框架，以提升卷积神经网络在上下文偏移下的鲁棒性。我们的方法利用空间特征归因来引导模型做出更少依赖上下文区域、更多依赖物体相关特征的预测。作为第一步，我们识别出现有特征归因方法的一个理论局限，并由此引入了一种新变体 ContrastiveCAMs，它能够生成更忠实反映模型预测的归因图。在 ContrastiveCAMs 的基础上，我们进一步提出了上下文正则化交叉熵（Context-Regularized Cross-Entropy）训练方法。

    arXiv:2611.08157v1 Announce Type: cross  Abstract: Image classification models often encounter objects in diverse and unpredictable contexts at test time, which may differ substantially from those seen during training. This phenomenon, termed context shift, exposes a key fragility in neural networks: when models rely on non-causal contextual cues for prediction, shifts in context can lead to significant performance degradation. In this paper, we propose a novel training framework to improve the robustness of convolutional neural networks under context shift. Our approach leverages spatial feature attribution to guide models toward making predictions that rely less on contextual regions and more on object-relevant features. As a first step, recognizing a theoretical limitation of existing feature attribution methods we introduce a new variant, ContrastiveCAMs, which produces more faithful attribution maps of model predictions. Building on ContrastiveCAMs, we further propose Context-Regu
    
[^47]: La-JEPA：基于潜在世界建模的紧凑机器人策略

    La-JEPA: Compact Robot Policies with Latent World Modeling

    [https://arxiv.org/abs/2611.08156](https://arxiv.org/abs/2611.08156)

    La-JEPA是一种紧凑的世界-动作模型，它在预训练V-JEPA 2.1编码器的特征空间中从头训练策略Transformer，无需具身预训练即可以更少的参数和更快的推理速度在RoboTwin 2.0和LIBERO-Plus基准上取得最先进的机器人控制性能。

    

    世界-动作模型通过适配预训练的视频生成器构建，而视觉-语言-动作模型则建立在预训练的视觉-语言骨干网络之上；这两类方法都将策略架构和推理成本与为生成而非控制设计的大型Transformer绑定在一起。目前尚不清楚这些面向生成的选择对控制任务有多大影响。在本工作中，我们提出了潜在动作JEPA（La-JEPA），这是一个紧凑的世界-动作模型，其策略Transformer在预训练V-JEPA 2.1编码器的特征空间中从头开始训练。遵循Fast-WAM的方法，La-JEPA在训练时学习联合预测未来特征和动作，而在推理时仅生成动作。在没有具身预训练的策略中，La-JEPA在干净和随机化的RoboTwin 2.0场景以及指令正确的LIBERO-Plus基准上取得了最先进的性能，同时使用的参数远少于最强的竞争策略，并且运行速度显著快于F……（原文摘要在此处截断）

    arXiv:2611.08156v1 Announce Type: cross  Abstract: World-action models adapt pretrained video generators, and vision-language-action models build on pretrained vision-language backbones; both tie policy architecture and inference cost to large transformers designed for generation rather than control. It remains unclear how much these generation-oriented choices affect control. In this work, we introduce Latent Action JEPA (La-JEPA), a compact world-action model whose policy transformer is trained from scratch in the feature space of a pretrained V-JEPA 2.1 encoder. Following Fast-WAM, La-JEPA learns to predict future features jointly with actions during training and generates only actions at inference. Among policies without embodied pretraining, La-JEPA achieves state-of-the-art performance in clean and randomized RoboTwin 2.0 scenes and on LIBERO-Plus with correct instructions, while using far fewer parameters than the strongest competing policies. It runs substantially faster than F
    
[^48]: 基于中继的AI原生面向任务语义通信：收益与脆弱性

    AI-Native Task-Oriented Semantic Communications over a Relay: Benefits and Vulnerabilities

    [https://arxiv.org/abs/2611.08155](https://arxiv.org/abs/2611.08155)

    本文提出了基于中继的任务导向语义通信的联合端到端设计，证明其在低信噪比下优于直接传输，同时首次揭示并形式化了中继注入学习扰动的拜占庭攻击这一关键安全漏洞。

    

    本文考虑了使用中继节点的语义通信（SemCom）场景，其中中继协助将任务相关信息从源节点传递至目的节点，其语义含义由分类标签定义。研究开发了源编码器、中继变换和目的端解码器的联合端到端设计，以同时优化图像数据的语义推理精度和感知重建质量。结果表明，中继辅助方案优于直接传输，并且在不同的中继策略下，这种增益在低信噪比（SNR）环境中表现得更为显著。然而，与这些性能收益并存的是，研究识别出中继引入的一个关键安全漏洞，并形式化了一种拜占庭攻击：中继在其发送信号中注入结构化的、通过学习得到的扰动。该攻击作用于受信道效应影响的中间通信表示，其扰……（原文摘要在此处截断）

    arXiv:2611.08155v1 Announce Type: cross  Abstract: Semantic communication (SemCom) using a relay node is considered, where the relay assists in conveying task-relevant information from a source to a destination, with meaning defined by classification labels. A joint end-to-end design of the source encoder, relay transformation, and destination decoders is developed to optimize both semantic inference accuracy and perceptual reconstruction quality for image data. Relay-assisted schemes are shown to outperform direct transmission, with gains that become more pronounced in low signal-to-noise ratio (SNR) regimes across different relaying strategies. Alongside these performance benefits, however, a key security vulnerability introduced by the relay is identified, and a Byzantine attack is formulated in which the relay injects structured, learned perturbations into its transmitted signal. The attack operates on intermediate communication representations subject to channel effects, with pert
    
[^49]: 可解释的多模态MRI-临床数据融合用于急性缺血性卒中后功能结局预测

    Interpretable Multimodal MRI-Clinical Fusion for Functional Outcome Prediction After Acute Ischemic Stroke

    [https://arxiv.org/abs/2611.08154](https://arxiv.org/abs/2611.08154)

    该研究提出一种可解释的多模态MRI与临床数据融合方法，证明FLAIR-临床融合（AUC=0.948）能显著提升急性缺血性卒中后三个月功能结局的预测性能。

    

    本研究评估了发病第1天多模态MRI对预测急性缺血性卒中（AIS）后三个月功能结局的额外预后价值。研究分析了66名具有ADC、DWI、FLAIR及临床数据的AIS患者。研究从每种MRI模态中提取三维预训练ResNet嵌入特征，使用折间隔离的主成分分析进行降维，并与结构化临床变量及ADC衍生的病灶生物标志物相结合。在多种MRI模态组合上，采用八折分层分组交叉验证评估线性支持向量机的性能。仅使用临床数据的模型取得了较强的性能（AUC = 0.899 ± 0.151），而仅使用MRI的模型判别能力有限（AUC ≈ 0.67-0.73）。在多模态配置中，FLAIR-临床融合取得了最高的平均交叉验证性能（AUC = 0.948 ± 0.061），其次是ADC-临床融合（AUC = 0.941 ± 0.094）。

    arXiv:2611.08154v1 Announce Type: cross  Abstract: This study evaluates the added prognostic value of day-1 multimodal MRI for predicting three-month functional outcome after acute ischemic stroke (AIS). Sixty-six AIS patients with ADC, DWI, FLAIR, and clinical data were analyzed. Three-dimensional pretrained ResNet embeddings were extracted from each MRI modality, reduced using fold-isolated principal component analysis, and combined with structured clinical variables and ADC-derived lesion biomarkers. Linear support vector machines were evaluated using eight-fold stratified group cross-validation across MRI modality combinations. Clinical-only modeling achieved strong performance (AUC = $0.899 \pm 0.151$), while MRI-only models showed limited discrimination (AUC $\approx$ $0.67$-$0.73$). Among multimodal configurations, FLAIR-clinical fusion achieved the highest mean cross-validated performance (AUC = $0.948 \pm 0.061$), followed by ADC-clinical fusion (AUC = $0.941 \pm 0.094$). Full
    
[^50]: 从卫星观测中检测天气锋面：再分析数据能带来什么增益？

    Detecting Weather Fronts from Satellite Observations: What Does Reanalysis Add?

    [https://arxiv.org/abs/2611.08148](https://arxiv.org/abs/2611.08148)

    该研究比较了仅用静止红外卫星观测与额外加入ERA5再分析数据的机器学习天气锋面检测模型，发现再分析数据可将检测性能显著提升（CSI@150从0.402提高至0.575），其中低层风场是仅靠卫星数据最难预测的信息。

    

    天气锋面是具有不同热力学性质的气团之间的过渡区域，与中纬度降水和恶劣天气密切相关。我们研究了用于检测天气锋面的机器学习方法，并量化了再分析数据对模型性能的贡献。我们比较了同一机器学习模型在北大西洋和欧洲地区进行像素级锋面分类的两种配置：（1）仅使用静止红外卫星观测数据，（2）在此基础上额外使用28个ERA5再分析场。仅使用卫星数据的模型在150公里容差下的邻域临界成功指数（CSI@150）达到0.402，该结果处于不同气象服务机构人工分析结果之间一致性范围内。卫星+ERA5模型则达到0.575，提升了0.173。我们的分析表明，低层风是最难从卫星表征中预测的变量（R² = 0.23，

    arXiv:2611.08148v1 Announce Type: new  Abstract: Weather fronts are transition regions between air masses with contrasting thermodynamic properties and are associated with mid-latitude precipitation and severe weather. We investigate machine learning (ML) methods for detecting weather fronts and quantify the contribution of reanalysis data to performance. We compare two configurations of the same ML model for pixel-wise frontal classification over the North Atlantic and Europe: (1) using only geostationary infrared satellite observations, and (2) additionally using 28 ERA5 reanalysis fields. The satellite-only model achieves a neighbourhood critical success index at a 150 km tolerance (CSI@150) of $0.402$, which lies within the range of agreement between analyses from different meteorological services. The satellite$+$ERA5 model achieves $0.575$, an improvement of $0.173$. Our analysis shows that low-level winds are the least predictable from the satellite representation ($R^2 = 0.23$,
    
[^51]: 强化学习中环境-训练器边界处的静默故障检测

    Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning

    [https://arxiv.org/abs/2611.08144](https://arxiv.org/abs/2611.08144)

    该论文发现强化学习中环境与训练器边界处的静默故障几乎无法通过学习曲线等基于结果的检查发现，而直接检查运行过程中跨越边界传输的数据本身则能检测到每一个注入的故障。

    

    机器人强化学习越来越多地在训练器之外运行环境：在另一个进程中、在另一台机器上、使用另一种编程语言。这种边界上的许多故障不会引发任何错误报告。我们将该边界建模为一种线协议，并探究是什么能够察觉到其中发生的故障。在 Pendulum、HalfCheetah、Hopper 和一个 7 关节机械臂上，我们向跨越线路传输的数据中注入了多达 23 种故障，并向训练器的日志中注入了 2 种故障，范围从单个值出错到全部值出错。除非每个值都是错误的，否则基于学习曲线阈值的检查最多只能发现十分之一的故障运行，与十个无故障种子对比的检查最多发现 21%。对所有 1,060 个最终策略的干净评估与曲线检查结果一致：在 728 个边界故障运行中，两种曲线检查均通过，其中 723 个的得分不比无故障种子低超过三个标准差，尽管几乎所有运行最终都收敛到了不同的权重，而且对于奖励类故障，记录的回报值也与正常情况不同。而对运行过程本身（而非其结果）的检查则发现了每一个故障。一个步进台阶……（摘要在此处截断）

    arXiv:2611.08144v1 Announce Type: cross  Abstract: Robot reinforcement learning increasingly runs environments outside the trainer: in another process, on another machine, in another language. Many faults at this boundary raise no error. We make the boundary a wire protocol and ask what notices a fault in it.   On Pendulum, HalfCheetah, Hopper and a 7-joint arm (Pusher), we injected up to 23 faults into what crosses the wire and 2 into the trainer's log, from one value to every value. Unless every value was wrong, a learning-curve threshold caught at most one run in ten, and a comparison with ten fault-free seeds at most 21%. A clean evaluation of all 1,060 final policies agreed with the curves: of the 728 boundary-fault runs both curve checks passed, 723 scored no more than three standard deviations below fault-free seeds, though nearly all ended on other weights and, for reward faults, logged other returns.   Checks of the run itself, not of its outcome, saw every fault. A step ledge
    
[^52]: 未经训练的神经网络在异常检测中能走多远？从表格到像素

    How Far Can Untrained Neural Networks Go in Anomaly Detection? From Tables to Pixels

    [https://arxiv.org/abs/2611.08136](https://arxiv.org/abs/2611.08136)

    提出PRISM框架，仅利用固定的随机非线性特征结合基于正常参照的马氏距离评分，无需任何训练即可在表格异常检测和视觉异常定位任务上达到最先进水平。

    

    异常检测（AD）通常被构建为一个表示学习问题，人们普遍认为要取得强性能需要学习到的特征或针对异常检测的专门优化。然而，这种观点可能忽略了异常相对于正常参照本身所具有的可分性，即使在缺乏学习到的表示的情况下也是如此。因此，我们研究了未经训练的神经网络在异常检测中能走多远。为此，我们提出了PRISM，一个将固定随机非线性特征与基于正常参照的马氏距离评分相结合的框架，可用于表格数据检测和密集视觉定位。在ADBench的47个表格数据集上，PRISM-T在所比较的方法中取得了最高的平均AUROC（86.4%）和第二高的平均AUPRC（65.9%）。在涵盖27个视觉类别的MVTec AD和VisA数据集上，PRISM-V分别取得了90.4%的平均图像级AUROC和96.4%的平均像素级AUROC。值得注意的是，两个模型均在不进行神经网络权重优化（或更新）的情况下取得了这些结果。

    arXiv:2611.08136v1 Announce Type: new  Abstract: Anomaly detection (AD) is often framed as a representation-learning problem, where strong performance is expected to require learned features or AD-specific optimization. However, this view can overlook the intrinsic separability of anomalies from normal references, even in the absence of learned representations. We therefore investigate how far untrained neural networks can go in anomaly detection. To this end, we propose PRISM, a framework combining fixed random nonlinear features with normal-reference Mahalanobis scoring for tabular detection and dense visual localization. Across 47 tabular datasets in ADBench, PRISM-T achieves the highest mean AUROC (86.4%) and the second-highest mean AUPRC (65.9%) among the compared methods. On MVTec AD and VisA, covering 27 visual categories, PRISM-V achieves mean image-level and pixel-level AUROCs of 90.4% and 96.4%, respectively. Notably, both models achieve these results without neural weight op
    
[^53]: 使用非线性干预状态空间模型识别神经动力学

    Identifying Neural Dynamics Using Nonlinear Interventional State Space Models

    [https://arxiv.org/abs/2611.08113](https://arxiv.org/abs/2611.08113)

    该研究将干预状态空间模型的可识别性理论从线性动力学推广到一般非线性潜在动力学与单射发射函数，为识别神经回路动力学提供了更广泛的理论保证。

    

    状态空间模型（SSMs）能够从高维观测数据中恢复低维潜在动力学，但仅凭观测数据通常无法识别这些潜在变量。干预状态空间模型利用扰动打破观测对称性，从而在允许置换、缩放和平移的等价意义下识别模型参数。然而，先前的工作将可识别性限制在线性潜在动力学和分段线性发射函数，排除了神经回路中许多典型的行为特征。我们将iSSMs扩展到更广泛的模型类别：我们证明（i）在一般的单射发射函数和高斯潜在转移下，仿射可识别性成立；（ii）在有干预的情况下，在一般的非线性潜在动力学下，直到置换、缩放和平移的可识别性仍然保持。我们在模拟的FitzHugh-Nagumo动力学上验证了该理论，随后展示了两个应用：在相互竞争的模型之间进行裁决……（原文摘要在此处截断）

    arXiv:2611.08113v1 Announce Type: cross  Abstract: State-space models (SSMs) recover low-dimensional latent dynamics from high-dimensional observations, but latents are generally not identifiable from observational data alone. Interventional state-space models (iSSMs) use perturbations to break observational symmetries and identify parameters up to permutation, scaling, and shift. However, prior work restricts identifiability to linear latent dynamics and piecewise-linear emissions, excluding many behaviors characteristic of neural circuits. We extend iSSMs to a substantially broader model class: we prove that (i) affine identifiability holds under general injective emission functions and Gaussian latent transitions, and (ii) with interventions, identifiability up to permutation, scaling, and shift is preserved under general nonlinear latent dynamics. We validate our theory on simulated FitzHugh-Nagumo dynamics, and then demonstrate two applications: adjudicating between competing mode
    
[^54]: PersonaAgent：基于可刷新多实体角色记忆的检索增强交易异常评分

    PersonaAgent: Retrieval-Augmented Transaction Anomaly Scoring with Refreshable Multi-Entity Persona Memory

    [https://arxiv.org/abs/2611.08094](https://arxiv.org/abs/2611.08094)

    PersonaAgent通过将实体最新行为信号以角色形式存储在可刷新的多实体记忆中，并在推理时由轻量级适配器检索以增强基础异常分类器的评分，有效解决了交易模式频繁变化与实体嵌入高基数稀疏的挑战。

    

    在大规模支付网络中，实体（卡、商户等）及其交互在交易异常检测中起着关键作用。然而，不同实体之间的交易模式通常比生产模型的刷新计划变化得更加频繁。同时，实体嵌入表还面临高基数与稀疏交互的问题。为了解决上述挑战，我们提出了PersonaAgent，这是一个智能体式的交易异常检测框架，它将最新的实体行为信号表示为角色并保存在可刷新的记忆中。在离线阶段，通过每周定时运行的批处理任务提取最新的行为信号并更新到记忆中。在推理阶段，轻量级适配器从记忆中检索每笔交易对应的实体行为信号，并用它们来增强基础异常分类器的评分。通过与不同异常检测方法进行全面的基准测试……

    arXiv:2611.08094v1 Announce Type: new  Abstract: In large-scale payment networks, entities (card, merchant, etc.) and their interactions play a key role in transaction anomaly detection. However, transaction patterns between different entities usually change more frequently than the production model refresh schedule. Meanwhile, entity embedding tables suffer from high-cardinality and sparse interaction issues. To resolve the above challenges, we present PersonaAgent, an agentic transaction anomaly detection framework which represents the latest entity behavior signals as personas and saves them in a refreshable memory. Offline, a weekly scheduled batch job is used to extract the latest behavior signals and update them in the memory. At inference time, a lightweight adapter retrieves the corresponding entity behavior signals of each transaction from memory and uses them to augment the base anomaly classifier's scoring. Through comprehensive benchmarking with different anomaly detection 
    
[^55]: 梯度提升用于张量网络核机器中的高效秩发现

    Gradient Boosting for Efficient Rank Discovery in Tensor Network Kernel Machines

    [https://arxiv.org/abs/2611.08093](https://arxiv.org/abs/2611.08093)

    提出梯度提升框架GB-CPKM，通过顺序添加固定秩CP张量网络学习器来增量式提升模型容量，从而避免显式秩搜索，实现最高6倍的训练加速和2倍的峰值内存节省。

    

    张量网络核机器（TNKMs）将非线性特征与低秩张量表示相结合，以高效地建模复杂关系。然而，选择合适的张量秩仍然具有挑战性，因为传统的秩搜索方法需要训练多个候选模型，从而增加了计算成本。在本工作中，我们聚焦于规范多线性（CP）张量网络参数化，并提出GB-CPKM——一种梯度提升框架，通过顺序添加固定秩的学习器来构建更高容量的预测器。该方法通过增量式增加模型容量，为秩选择提供了新的视角：集成规模基于验证性能来选择，而无需进行显式的秩搜索。在真实世界回归基准上的实验表明，与基于交叉验证的传统秩选择方法相比，GB-CPKM的训练速度最高可提升6倍，峰值内存使用最多可减少2倍，同时……

    arXiv:2611.08093v1 Announce Type: cross  Abstract: Tensor network kernel machines (TNKMs) combine nonlinear features with low-rank tensor representations to model complex relationships efficiently. However, selecting an appropriate tensor rank remains challenging, as conventional rank-search methods require training multiple candidate models, increasing computational costs. In this work, we focus on a canonical polyadic (CP) tensor network parameterization and propose GB-CPKM, a gradient boosting framework that builds a higher-capacity predictor by sequentially adding fixed-rank learners. This approach provides a new perspective on rank selection by incrementally increasing model capacity, with ensemble size selected based on validation performance rather than through explicit rank search. Experiments on real-world regression benchmarks show that GB-CPKM trains up to 6x faster and uses up to 2x less peak memory than a conventional rank-selection method based on cross-validation, while 
    
[^56]: Gabor Gabor：E-Gaborformer——采用Gabor字典卷积与Delta规则记忆的严格因果语音识别器

    Gabor Gabor: E-Gaborformer, a Strictly Causal Speech Recognizer with Gabor-Dictionary Convolutions and Delta-Rule Memory

    [https://arxiv.org/abs/2611.08092](https://arxiv.org/abs/2611.08092)

    E-Gaborformer是一种严格因果的语音识别编码器，通过Gabor字典卷积与Delta规则循环记忆实现恒定约19MB的流式状态（不随通话时长增长），在LibriSpeech上收敛速度最高提升18%，生产规模下词错误率降低28-31%。

    

    实时语音代理需要一种永不等待未来音频、且每次调用内存不随通话时长增长的语音识别器。我们提出E-Gaborformer，这是一种严格因果的编码器，它以3:1的比例将门控DeltaNet-2循环层与滑窗注意力相结合，并且每个时间卷积都由固定的因果Gabor原子字典构建。其流式状态是恒定的，在生产规模下每次调用约19 MB（包括预测网络），而同等规模的因果注意力编码器在十分钟后会累积超过1 GB的内存。在LibriSpeech上的受控因果研究中，E-Gaborformer更早脱离初始平台期，达到给定精度的速度最高提升18%，并能与使用相同前端和训练方案训练的E-Branchformer持平或超越。在生产规模下，基于E-Gaborformer构建的0.6B参数希伯来语和英语流式传感器（transducer）的词错误率（WER）降低了28%至31%。

    arXiv:2611.08092v1 Announce Type: cross  Abstract: Real-time voice agents need a speech recognizer that never waits for future audio and whose memory per call does not grow with the length of the call. We present E-Gaborformer, a strictly causal encoder that combines Gated DeltaNet-2 recurrent layers with sliding-window attention in a 3:1 ratio, and builds every temporal convolution from a fixed dictionary of causal Gabor atoms. Its streaming state is constant, about 19 MB per call at production scale including the prediction network, where a causal-attention encoder of the same size accumulates over 1 GB after ten minutes. In a controlled causal study on LibriSpeech, E-Gaborformer leaves the initial plateau earlier, reaches a given accuracy up to 18% sooner, and matches or beats an E-Branchformer trained with the identical front end and recipe. At production scale, a 0.6B-parameter Hebrew and English streaming transducer built on E-Gaborformer has 28 to 31% lower word error rate (WER)
    
[^57]: SAE 解读 MoE：稀疏自编码器揭示 MoE 专家表现出人类可解释的细粒度专业化

    SAE Interprets MoE: Sparse AutoEncoders Reveal That MoE Experts Exhibit Human-Interpretable Fine-Grained Specialization

    [https://arxiv.org/abs/2611.08089](https://arxiv.org/abs/2611.08089)

    该论文通过在 MoE 层输出上训练稀疏自编码器，利用其仿射结构将每个潜变量的激活精确分解到各专家，从而揭示出 MoE 专家具有比门控选择统计所能体现的更集中、且人类可解释的细粒度专业化分工。

    

    混合专家模型将每个 token 路由到众多专家中的少数几个，然而单个专家究竟负责什么仍悬而未决。现有分析通过在带有领域标签的语料库上统计专家被选中的频率来研究专家分工，这种方法预先固定了分类体系、粒度粗糙，且记录的是“被选择”而非“实际贡献”。我们转而在特征空间中研究专家专业化：在 MoE 层的输出上训练稀疏自编码器（SAE）。由于该输出是各专家输出的路由器加权求和，而 SAE 编码器是仿射变换，因此每个潜变量的预激活值可以精确地分解到各个专家上。在语料库上聚合后，这为每个专家得到一组主导潜变量，其激活主要由该专家驱动。在 Qwen3-30B-A3B 模型的第 4、30 和 46 层上，研究发现专家的贡献远比门控分布更集中，大多数潜变量仅由一个或两个专家主导，且主导集合之间的重叠集中在少数专家对上。

    arXiv:2611.08089v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) models route each token to a few of many experts, yet what an individual expert is responsible for remains unsettled. Existing analyses count how often an expert is selected on domain-labeled corpora, an instrument that fixes the taxonomy in advance, stays coarse, and records selection rather than contribution. We instead study specialization in feature space by training a sparse autoencoder (SAE) on the output of an MoE layer. Since this output is a router-weighted sum of expert outputs and the SAE encoder is affine, the pre-activation of every latent decomposes exactly over the experts. Aggregated over a corpus, this gives each expert a dominant set of latents whose activation it principally drives. On layers 4, 30 and 46 of Qwen3-30B-A3B, contribution is far more concentrated than gating, most latents have only one or two dominators, and the overlap between dominant sets concentrates on a few expert pairs. Aut
    
[^58]: 价值还是注意力：面向图注意力网络的逐层相关性传播研究

    Value or Attention: Investigating Layer-wise Relevance Propagation for Graph Attention Networks

    [https://arxiv.org/abs/2611.08088](https://arxiv.org/abs/2611.08088)

    该论文提出了一种带自由权重的LRP乘积规则，将图注意力网络中节点接收到的相关性分解为值路径的内容相关性和注意力路径的路由相关性两部分，并将现有解释方法统一为其特例。

    

    图注意力网络越来越多地应用于从医学到材料科学等需要科学家解读模型推理过程的领域，然而它们所学习到的路由机制对大多数解释方法而言仍然是不透明的。逐层相关性传播（LRP）能够遵循模型的计算过程，但在注意力层中，到达某个节点的相关性是两个机制上截然不同的量的总和：通过值路径传递的内容相关性，以及通过注意力路径传递的路由相关性。现有的规则要么丢弃其中一个部分，要么将它们简单相加成一个数值。我们提出了一个作用于这两个因子、带有自由权重的单一乘积规则，能够将现有方法作为特例加以恢复。值路径衡量的是节点相对于零基线所提供的内容；注意力路径衡量的是节点相对于其邻域如何塑造了传递的信息，其符号记录了节点的特征是否按照预测所需的方式影响了门控机制。

    arXiv:2611.08088v1 Announce Type: cross  Abstract: Graph Attention Networks (GATs) are increasingly used in domains where scientists need to read a model's reasoning, from medicine to materials science, yet the routing they learn remains opaque to most explainers. Layer-wise Relevance Propagation (LRP) follows the model's computation, but in an attention layer the relevance reaching a node is the sum of two mechanically different quantities: its content relevance through the value path and its routing relevance through the attention path. Existing rules either drop one or add them into a single number. We write a single product rule with free weights on the two factors that recovers current approaches as special cases. The value path measures what content a node supplied against a zero baseline; the attention path measures how the node shaped what was passed relative to its neighborhood, and its sign records whether the node's features moved the gating the way the prediction demanded. 
    
[^59]: DictLLM：基于字典核的大语言模型训练后压缩

    DictLLM: Post-training Compression of Large Language Model with Dictionary Kernels

    [https://arxiv.org/abs/2611.08082](https://arxiv.org/abs/2611.08082)

    DictLLM提出了一种基于字典核的训练后压缩方法，将分组层的权重矩阵分解为共享字典与层特定稀疏系数，无需进一步训练即可在移动设备上高效压缩大语言模型。

    

    在移动设备上高效部署大语言模型（LLM）需要在性能与计算、内存限制之间取得平衡。训练后压缩是管理这种权衡的有效方法，同时比训练过程中压缩所需计算量少得多。我们提出了DictLLM，一种基于字典的压缩方案，其中分组层内的权重矩阵被分解为共享字典和层特定的稀疏系数，可直接从预训练模型获得而无需进一步训练。具体而言，我们将DictLLM构建为一个四步训练后优化流程：首先，通过对输入缩放的权重矩阵进行奇异值分解（SVD）获得共享字典，从而在分组层之间分摊秩；其次，利用字典感知的Hessian矩阵将层特定系数稀疏化至目标压缩率；第三，在得到稀疏系数的基础上对字典进行精炼。

    arXiv:2611.08082v1 Announce Type: new  Abstract: Efficient deployment of large language models (LLMs) on mobile devices requires balancing performance with compute and memory constraints. Post-training compression is an effective approach to managing this trade-off, while requiring much less computation than in-training counterparts. We propose DictLLM, a dictionary-based compression scheme in which weight matrices within grouped layers are decomposed into a shared dictionary and layer-specific sparse coefficients, directly obtained from a pre-trained model without further training. Specifically, we cast DictLLM as a four-step post-training optimization. First, we obtain shared dictionaries via singular value decomposition (SVD) of input-scaled weight matrices, amortizing rank across grouped layers; second, we sparsify the layer-specific coefficients to the target compression ratio using a dictionary-aware Hessian; and third, we refine the dictionaries given the resulting sparse coeffi
    
[^60]: 二次增长条件下追逐凸函数问题的最优竞争比

    Optimal Competitive Ratios for Chasing Convex Functions with Quadratic Growth

    [https://arxiv.org/abs/2611.08078](https://arxiv.org/abs/2611.08078)

    本文针对具有二次增长的凸函数追逐问题，在欧几里得移动和平方欧几里得移动两种成本类型下建立了最优的无维度依赖竞争比，弥补了已有上下界之间的多项式差距。

    

    追逐凸函数问题近年来备受关注，它可以被形式化为学习者和对抗者之间的博弈。在每一轮中，对抗者揭示一个凸函数，学习者选择一个决策，并因此产生命中成本和改变决策所带来的移动成本。学习者的性能通过竞争比来衡量，即将其累积成本与最优离线决策序列的累积成本进行比较。对于欧几里得移动成本，现有针对 κ-良态函数的上下界之间存在多项式级别的差距。对于平方欧几里得移动成本，虽然已在二次增长条件下建立了无维度依赖的竞争保证，但最优竞争比仍不明确。在本文中，我们针对两种类型的移动成本下的凸函数追逐问题建立了最优的无维度依赖竞争比。对于欧几里得移动成本，我们提出了裁剪近端下降算法（摘要在此处被截断）。

    arXiv:2611.08078v1 Announce Type: new  Abstract: Chasing convex function has attracted much attention in recent years, which can be formulated as a game between a learner and an adversary. In each round, the adversary reveals a convex function, and the learner selects a decision and incurs a hitting cost and a movement cost for changing its decision. The performance of the learner is measured by the \emph{competitive ratio}, which compares its cumulative cost with that of an optimal offline decision sequence. For Euclidean movement, existing upper and lower bounds for $\kappa$-well-conditioned functions exhibit a polynomial gap. For squared Euclidean movement, dimension-free competitive guarantees have been established under quadratic growth, but the optimal competitive ratio remains unclear.   In this paper, we establish optimal dimension-free competitive ratios for convex function chasing under two types of movement costs. For Euclidean movement, we propose Clipped Proximal Descent (
    
[^61]: LLM究竟能真正对神经网络的权重进行推理和编程，还是只是“打了类固醇的记忆”？——探索“权重空间编程”

    Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? Exploring "Weight-Space Programming"

    [https://arxiv.org/abs/2611.08070](https://arxiv.org/abs/2611.08070)

    论文在无工具、无前向传播的条件下测试LLM读取小型卷积网络原始权重的能力，发现其表现主要源于准确的局部算术计算与对公开数据集的记忆先验，而非对权重流动的真正全局推理。

    

    大型语言模型（LLM）最近被报道能够直接读取和写入神经网络的原始权重，这一能力被称为“权重空间编程”。我们探究这一能力的“读取”端究竟反映了对权重流动的推理，还是网络边缘的局部算术运算与对公开数据集记忆先验的结合。我们在五个图像数据集上训练的约12.5万参数的小型卷积网络上测试了 Claude Opus 5.5，将每个权重以纯文本形式提供，不使用工具、不执行前向传播、不提供数据集或类别名称。该模型能够执行准确的局部计算：批归一化统计量使其能够推断出在三次更新后停止训练的网络大约经历了三次优化器步骤，输出层的符号模式能够恢复 CIFAR-10 的车辆/动物分类。会话记录显示模型紧密读取了第一层和最后一层，并对中间层进行了采样，而且若干声明建立在部分追踪的基础上……

    arXiv:2611.08070v1 Announce Type: cross  Abstract: Large language models (LLMs) have recently been reported to read and write the raw weights of neural networks directly, a capability termed weight-space programming. We ask whether the reading side of this capability reflects reasoning about weight flows or a combination of local arithmetic at the network's edges and memorized priors over public datasets. We test Claude Opus 5.5 on small convolutional networks of about 125,000 parameters trained on five image datasets, providing every weight as plain text with no tools, no forward passes, and no dataset or class names. The model performs accurate local calculations: batch-normalization statistics lead it to infer about three optimizer steps on networks stopped after three updates, and output-layer sign patterns recover the vehicle/animal split of CIFAR-10. The sessions report reading the first and last layers closely and sampling the middle, and several claims built on partial tracing 
    
[^62]: 机器学习中的性能指标：从混淆矩阵到校准、回归与不确定性的基于场景的指南

    Performance metrics in machine learning: a scenario-based guide from the confusion matrix to calibration, regression and uncertainty

    [https://arxiv.org/abs/2611.08048](https://arxiv.org/abs/2611.08048)

    本教程通过15个受控场景系统揭示了类别流行率、决策阈值等因素如何使相同的预测在不同性能指标下得出截然不同的结论，为机器学习评估指标的正确选择与解读提供了实用指南。

    

    性能指标在相同的预测结果下可能得出不同的结论。本方法论教程通过受控场景解释了类别流行率、决策阈值、聚合规则和软件约定的影响。22种混淆矩阵指标被组织成一张按流行率编码的地图。15个场景涵盖二分类和多分类、评分判别力、概率校准、回归、不确定性及一致性检查。输入在发布的文件中固定，每个场景仅改变一个因素。所有205项验证检查均通过；其中202项为独立验证，3项为内部验证。一个“总是预测为负”的规则在流行率为0.010时达到0.990的准确率，但平衡准确率仅为0.500。在敏感度和特异度固定的情况下，当流行率从0.5降至0.001时，精确率从0.889降至0.008。三个概率变体共享0.825的ROC-AUC，但校准斜率却分别为2.08、1（摘要在此处被截断）。

    arXiv:2611.08048v1 Announce Type: new  Abstract: Performance metrics can yield different conclusions from the same predictions. This methodological tutorial explains the effects of class prevalence, decision thresholds, aggregation rules and software conventions through controlled scenarios. Twenty-two confusion-matrix metrics are organized in a prevalence-coded map. Fifteen scenarios cover binary and multi-class classification, score discrimination, probability calibration, regression, uncertainty and consistency checks. Inputs are fixed in released files, and one factor is varied in each scenario. All 205 verification checks passed; 202 were independent and 3 were internal. An always-negative rule achieved an accuracy of 0.990 at a prevalence of 0.010, with balanced accuracy of 0.500. At fixed sensitivity and specificity, precision fell from 0.889 to 0.008 as prevalence fell from 0.5 to 0.001. Three probability variants shared a ROC-AUC of 0.825 but had calibration slopes of 2.08, 1.
    
[^63]: KeHoSu：面向复制粘贴渠道认证文本溯源的密钥同形字替换

    KeHoSu: Keyed Homoglyph Substitution for Authenticated Text Provenance over Copy-Paste Channels

    [https://arxiv.org/abs/2611.08046](https://arxiv.org/abs/2611.08046)

    提出KeHoSu方法，将同形字攻击中的易混淆字符转化为带密钥的归属信道，通过Unicode字符变体在文本中嵌入平台、用户与认证标签信息，实现复制粘贴渠道下的文本溯源，并对句子重排和部分引用保持鲁棒。

    

    如今，大型语言模型大规模生成文本，其中许多文本在各平台间被重新分发。当版主发现可疑的评论、帖子或文章时，可能想知道它是由哪个平台生成、属于哪个账户。本文研究了此类文本实际传播的渠道——复制粘贴——上的归属问题：该渠道会丢弃文档元数据，但保留Unicode码点。我们提出了KeHoSu，它将同形字攻击中的易混淆字符转变为带密钥的归属信道：为每个载体字母选择的字符变体编码了标识符中的一个比特，该标识符由平台字段、用户字段和截断的认证标签组成。此外，鉴于转发文本常以片段形式被引用，其排版不保留位置状态，我们证明KeHoSu对句子重排具有不变性（至多存在一个恒定的边界层），且对部分引用的依赖仅通过所保留的长度体现。

    arXiv:2611.08046v1 Announce Type: cross  Abstract: Nowadays, large language models produce text at scale, and much of it is redistributed across platforms. A moderator who finds a suspicious comment, post or article may want to know which platform generated it and for which account. In this paper we study attribution over the channel through which such text actually travels, copy-paste, which discards document metadata but keeps Unicode code points. We present KeHoSu, which turns the confusable characters of homoglyph attacks into a keyed attribution channel: the variant chosen for each carrier letter encodes one bit of an identifier made of a platform field, a user field and a truncated authentication tag. Moreover, given that reshared text is often quoted in fragments, and the placement therefore keeps no positional state, we show that KeHoSu is invariant to sentence reordering up to a constant boundary layer and depends on a partial quotation only through the retained length. We cha
    
[^64]: 我们应该为鲁棒性付出这么多吗？具有边际保证的高效代理证书

    Should We Pay This Much for Robustness? Efficient Proxy Certificates with Marginal Guarantees

    [https://arxiv.org/abs/2611.08035](https://arxiv.org/abs/2611.08035)

    提出一种将鲁棒性认证委托给廉价代理函数的框架，借助鲁棒共形风险控制保证有界的边际错误批准率，可独立使用或作为后备机制，大幅降低认证成本。

    

    现代分类器对小的输入扰动仍然敏感，而强大的模型无关鲁棒性证书（如随机平滑）在部署中往往成本过高，每个输入需要大量的模型前向传播次数（例如2000次）。我们提出了一个框架，将认证任务委托给一个廉价的代理函数，同时在统计上控制其与原始可靠证书之间的偏差。给定一个可交换的无标签校准集，我们通过鲁棒共形风险控制来调节代理，以保证有界的边际（而非逐点）错误批准率：即在未来的输入中，代理接受一个其预测因扰动而翻转的输入的概率。所得到的代理证书既可以作为独立的边际证书使用，也可以作为一种后备机制，仅在代理拒绝时才调用昂贵的证书。我们提供了单样本和多样本设定下的（理论保证与实验结果）。

    arXiv:2611.08035v1 Announce Type: new  Abstract: Modern classifiers remain sensitive to small input perturbations, while strong model-agnostic robustness certificates such as randomized smoothing are often too expensive for deployment, requiring an extensive number (e.g. 2000) of model forward passes per input. We propose a framework that delegates certification to a cheap proxy function while statistically controlling its deviation from the original sound certificate. Given an exchangeable unlabeled calibration set, we tune the proxy via robust conformal risk control to guarantee a bounded marginal (rather than pointwise) false-approval rate: the probability, over future inputs, that the proxy accepts an input whose prediction is flipped due to perturbation. The resulting proxy certificate can be used either as a standalone marginal certificate or as a fallback mechanism that invokes the expensive certificate only when the proxy rejects. We provide both single-sample and sample-heavy 
    
[^65]: 具有 $\ell^p$ 系数范数的再生核（拟）巴拿赫空间

    Reproducing Kernel (Quasi-)Banach Spaces with the $\ell^p$ Coefficient Norms

    [https://arxiv.org/abs/2611.08032](https://arxiv.org/abs/2611.08032)

    本文将再生核巴拿赫空间理论推广到一般的 $\ell^p$ 正则化框架，在非凸情形 $0<p<1$ 下构造了带 $\ell^p$ 拟范数的再生核拟巴拿赫空间并建立了插值与正则化学习的存在性和稀疏性保证，同时在 $p>1$ 时基于可数字典构造 RKBS 并证明了极小值点的存在性。

    

    本文通过研究一般的 $\ell^p$ 正则化框架，特别是非凸情形 $0<p<1$，推进了再生核巴拿赫空间（RKBS）的理论。虽然我们之前的工作为带有 $\ell^1$ 范数的 RKBS 奠定了基础，但对更强稀疏性——即更接近 $\ell^0$ 伪范数——的追求，自然地激发了对 $\ell^p$ 拟范数的探索。对于 $0<p<1$，我们构造了具有 $\ell^p$ 拟范数的再生核拟巴拿赫空间（RKQBS），并在适当的假设下建立了插值和正则化学习的存在性与稀疏性结果。我们用广义勒贝格常数刻画了线性与松弛表示定理，并在核衰减、采样间距和稳定性假设下推导了相应的界。对于 $p>1$，我们使用固定的可数字典构造 RKBS，建立了极小值点的存在性，并确定了最优的……

    arXiv:2611.08032v1 Announce Type: new  Abstract: This paper advances the theory of reproducing kernel Banach spaces (RKBSs) by investigating the general $\ell^p$ regularization framework, with a particular focus on the non-convex regime $0<1$.   While our preceding work established the foundation for RKBSs with the $\ell^1$ norm, the pursuit of stronger sparsity---closer to the $\ell^0$ pseudo-norm---naturally motivates this exploration into $\ell^p$ quasi-norms.   For $0<1$, we construct reproducing kernel quasi-Banach spaces (RKQBSs) with $\ell^p$ quasi-norms and establish existence and sparsity results for interpolation and regularized learning under appropriate assumptions.   We characterize the linear and relaxed representer theorems in terms of a generalized Lebesgue constant and derive bounds under kernel decay, sample spacing, and stability assumptions.   For $p>1$, we construct RKBSs using a fixed countable dictionary, establish minimizer existence, and identify the optimal re
    
[^66]: GTAFL：解决医学图像中测试不可知的长尾联邦学习问题

    GTAFL: Addressing Test-Agnostic Long-Tailed Federated Learning in Medical Image

    [https://arxiv.org/abs/2611.08021](https://arxiv.org/abs/2611.08021)

    该论文提出了“测试不可知长尾联邦学习”这一新任务，并设计GTAFL综合框架，通过自适应重采样和专家分类器重训练等策略，解决医疗图像联邦学习中数据长尾分布且测试分布不可预测导致模型失效的问题。

    

    在医疗保健领域，由于医疗数据的敏感性及其广泛的可获取性，保护患者隐私至关重要。联邦学习（FL）提供了一种去中心化且保护隐私的训练范式，使其成为医疗保健应用的理想解决方案。医疗保健领域的一个关键挑战在于，现实世界中的医疗数据在局部和全局视角下往往呈现长尾分布。现有解决长尾联邦学习问题的方法通常假设模型将在均匀分布的测试数据上进行评估。然而，医疗保健系统中的实际测试数据往往是不可知且不可预测的，这可能导致模型在现实场景中失效。在本文中，我们提出了一个新任务——测试不可知长尾联邦学习，并提出了GTAFL，一个应对该挑战的综合框架。在训练阶段，GTAFL采用自适应重采样、专家分类器重训练以及……（摘要原文在此处被截断）

    arXiv:2611.08021v1 Announce Type: new  Abstract: In healthcare, protecting patient privacy is crucial due to the sensitivity of medical data and its extensive accessibility. Federated Learning (FL) offers a decentralized and privacy-preserving training paradigm, making it an ideal solution for healthcare applications. A critical challenge in healthcare is that real-world medical data often exhibits long-tailed distributions in local and global views. Existing methods addressing long-tailed FL problem typically assume that the model will be evaluated on uniform test data distribution. However, practical test data in healthcare systems is often agnostic and unpredictable, leading to potential model failures in real-world scenarios. In this paper, we introduce a novel task termed Test-Agnostic Long-Tailed Federated Learning and propose GTAFL, a comprehensive framework to address this challenge. During the training stage, GTAFL employs adaptive re-sampling, expert classifier retraining, an
    
[^67]: 在策略蒸馏的学习动力学

    Learning Dynamics of On-Policy Distillation

    [https://arxiv.org/abs/2611.08016](https://arxiv.org/abs/2611.08016)

    本文对在策略蒸馏的学习动力学进行了机制分析，提出将反馈循环分解为“蒸馏信号转换”与“广播”两个阶段的框架，用以解释学生响应有限以及训练过程中的退化现象。

    

    在策略蒸馏（OPD）正在成为语言模型后训练的一种重要范式，它将密集的教师监督与基于学生模型自身生成轨迹的训练相结合。这形成了一个反馈循环：每一次参数更新都会改变学生模型，进而改变生成后续更新的训练分布。然而，这种反馈如何塑造学习与泛化，目前仍知之甚少。我们对OPD的学习动力学进行了机制分析，区分了反馈循环的两个阶段：其一是蒸馏信号向参数更新的转换，其二是“广播”——通过广播，这些更新改变跨前缀的预测并重塑采样分布。这一框架有助于我们理解两种现象：一是尽管师生模型之间存在显著分歧，学生模型的响应却有限；二是当那些能够提升局部教师一致性的更新同时也增加了不成功续写出现的概率时，会发生模型退化。

    arXiv:2611.08016v1 Announce Type: cross  Abstract: On-policy distillation (OPD) is emerging as an important paradigm for language-model post-training, combining dense teacher supervision with training on student-generated trajectories. This creates a feedback loop where each update changes the student and thereby the training distribution that generates subsequent updates. Yet, how this feedback shapes learning and generalization remains poorly understood. We conduct a mechanistic analysis of OPD's learning dynamics, distinguishing the two feedback loop stages: conversion of the distillation signal into parameter updates, and broadcast through which these updates change predictions across prefixes and reshape the rollout distribution. This framework helps us understand both limited student responsiveness despite substantial teacher-student disagreement, and degeneration when updates that improve local teacher agreement also increase the probability of unsuccessful continuations, exposi
    
[^68]: 面向聚合物性质预测的基于几何片段化的多尺度分子表示学习

    Towards Multiscale Molecular Representation Learning through Geometry-Based Fragmentation for Polymer Property Prediction

    [https://arxiv.org/abs/2611.07998](https://arxiv.org/abs/2611.07998)

    该论文提出了一种受k-center启发的、基于几何的半径约束分子片段化方法，实现多尺度分子表示学习，从而提升聚合物性质预测的性能。

    

    众所周知，分子亚结构对化学和物理性质的贡献是不均等的，特定的基序或侧链往往起着主导作用。因此，基于片段的表示已成为引入具有化学意义的归纳偏置的有效策略，分子指纹和基于片段的注意力机制便是典型例子。然而，许多现有的基于片段化的机器学习方法依赖于预定义的化学规则，这对于之前未见过的或化学上陌生的结构可能产生次优的片段化结果，并且往往忽视了多尺度分子表示的重要性。为解决这些局限性，我们将分子片段化表述为一个受k-center公式启发的、以半径为约束的优化问题，其中簇的数量并非先验固定，而是由指定的半径决定，并提出了一种基于几何的片段化方法。

    arXiv:2611.07998v1 Announce Type: new  Abstract: It is well established that molecular substructures can contribute unequally to chemical and physical properties, with specific motifs or side chains often playing dominant roles. Fragment-based representations have therefore emerged as an effective strategy for incorporating chemically meaningful inductive biases, as exemplified by molecular fingerprints and fragment-based attention mechanisms. However, many existing fragmentation-based machine learning methods rely on predefined chemical rules, which may yield suboptimal fragmentation for previously unseen or chemically unfamiliar structures and often overlook the importance of multiscale molecular representations. To address these limitations, we formulate molecular fragmentation as a radius-constrained optimization problem inspired by the k-center formulation, where the number of clusters is not fixed a priori but emerges from the specified radius, and propose a geometry-based fragme
    
[^69]: QRAE：用于DeepSearch智能体上下文压缩的查询相关注意力熵

    QRAE: Query-Relevant Attention Entropy for Context Compression in DeepSearch Agents

    [https://arxiv.org/abs/2611.07976](https://arxiv.org/abs/2611.07976)

    提出QRAE方法，通过查询相关注意力熵对DeepSearch智能体检索到的文本块进行在线评分与筛选，将高价值证据保留在推理上下文中并缓存不确定内容，从而在多步搜索推理中有效压缩上下文、降低token消耗与延迟。

    

    DeepSearch LLM智能体会反复进行搜索，并将检索到的资料带入后续的推理步骤中。这种不断扩大的上下文会增加token消耗和延迟，尽管其中许多结果对后续推理并无贡献。我们提出查询相关注意力熵，这是一种在线上下文压缩方法，它利用稀疏的证据评估头来决定哪些完整的检索文本块应在推理步骤中持续保留。该方法通过成对的预填充对有界的候选批次进行评分：无查询的对照组用于抑制共享的注意力响应，而归一化熵则衡量每个文本块内正向响应的广度。高分的文本块会进入持久的证据白名单，并被纳入推理上下文。不确定的文本块则保留在推理上下文之外，但会被存入候选缓存中，以便后续可被重新检索并重新评估。QRAE既不需要额外的生成式压缩（摘要至此截断）

    arXiv:2611.07976v1 Announce Type: new  Abstract: DeepSearch LLM agents repeatedly search and carry retrieved sources into subsequent reasoning steps. This expanding context increases token consumption and latency, although many results do not contribute to later reasoning. We propose Query-Relevant Attention Entropy (QRAE), an online context compression method that uses sparse evidence-assessment heads to decide which intact retrieved chunks should persist across reasoning steps. It scores bounded candidate batches through paired prefills: a query-free control suppresses shared attention responses, while normalized entropy measures the breadth of the positive response within each chunk. High-scoring chunks enter a persistent evidence whitelist and are included in the reasoning context. Uncertain chunks remain outside the reasoning context but are retained in a candidate cache, where they can be retrieved for later reevaluation. QRAE requires neither an additional generative compression
    
[^70]: 分段线性Kolmogorov–Arnold网络线性区域最大数量的紧致渐近界

    Tight Asymptotic Bounds on the Maximum Number of Linear Regions of Piecewise-Linear Kolmogorov--Arnold networks

    [https://arxiv.org/abs/2611.07973](https://arxiv.org/abs/2611.07973)

    本文首次为分段线性Kolmogorov–Arnold网络的线性区域最大数量给出了相互匹配的上下界，确定了紧致增长阶Θ_n(k^(Σρ))，从而大幅改进了此前文献中宽松的上界并首次补齐了下界。

    

    Schoots等人（AISTATS 2025）针对连续分段线性（CPWL）Kolmogorov–Arnold网络（KAN）的线性区域数量建立了一个通常较为宽松的上界，但并未给出相应的下界。我们考虑一个T层网络，其宽度向量为n=(n₀,…,n_T)，输入维度n₀=d，输出宽度n_T=1，且每条边上至多有k≥2个仿射段。我们给出的相互匹配的上界和下界确定了关于k的紧致增长阶Θ_n(k^(Σ_{ℓ=1}^{T}ρ_{ℓ-1}))，其中ρ_{ℓ-1}:=min_{0≤s≤ℓ-1}n_s。当每个隐藏层宽度都至少为d时，该紧致阶可简化为Θ_n(k^(dT))。Schoots等人给出的上界为k^(Σ_{ℓ=1}^{T}n_{ℓ-1}n_ℓ)。我们的上界将其改进了一个Θ_n(k^(Σ_{ℓ=1}^{T}n_{ℓ-1}n_ℓ-Σ_{ℓ=1}^{T}ρ_{ℓ-1}))的因子，并且严格……（原文摘要在此处截断）

    arXiv:2611.07973v1 Announce Type: new  Abstract: Schoots et al.~\citep{schoots2025relating} (AISTATS 2025) established a generally loose upper bound on the number of linear regions of continuous piecewise-linear (CPWL) Kolmogorov--Arnold networks (KANs), without providing a lower bound. We consider a $T$-layer network with width vector $\mathbf n=(n_0,\ldots,n_T)$, input dimension $n_0=d$, output width $n_T=1$, and at most $k\geq2$ affine segments per edge. Our matching upper and lower bounds determine the tight growth order $\Theta_{\mathbf n}(k^{\sum_{\ell=1}^{T}\rho_{\ell-1}})$ in $k$, where $\rho_{\ell-1}:=\min_{0\leq s\leq\ell-1}n_s$. When every hidden width is at least $d$, the tight order simplifies to $\Theta_{\mathbf n}(k^{dT})$. The upper bound of \citep{schoots2025relating} is $k^{\sum_{\ell=1}^{T}n_{\ell-1}n_\ell}$. Our upper bound improves it by a factor of $\Theta_{\mathbf n}\!\left(k^{\sum_{\ell=1}^{T}n_{\ell-1}n_\ell-\sum_{\ell=1}^{T}\rho_{\ell-1}} \right)$, with strict
    
[^71]: CoRD：耦合采样轨迹填补同策略蒸馏的盲区

    CoRD: Coupled Rollouts Fill the Blind Spot of On-Policy Distillation

    [https://arxiv.org/abs/2611.07959](https://arxiv.org/abs/2611.07959)

    CoRD 通过投机解码的最大耦合从学生与教师分布的几何插值中抽取训练轨迹，在保留同策略损失的同时把训练状态分布推向教师，从而监督学生自身轨迹覆盖不到的“盲区”状态，并在五个数学基准上较同策略蒸馏平均提升 pass@1 0.9–2.0 分、pass@8 1.2–3.4 分。

    

    同策略蒸馏在学生模型自身生成的轨迹（rollout）上进行训练，因此教师模型能够到达而学生模型不会到达的状态很少得到监督。我们提出了 CoRD（耦合轨迹蒸馏，coupled-rollout distillation），它保留同策略损失，仅将训练状态的分布向教师模型推移，通过投机解码的最大耦合从学生与教师分布的几何插值中抽取轨迹。这种耦合在两个分布一致之处保留学生的 token，否则用学生所低估的目标分布部分中的 token 进行替换，因此干预的频率与任何精确采样器所能允许的一样低，并随着学生的学习而逐渐减少。在来自两个模型家族（Qwen3 和 Qwen3.5，0.6B 至 4B）的六个基础学生模型上，在相同的训练步数和数据量下，CoRD 在五个公开数学基准上相比同策略蒸馏将平均 pass@1 提升 0.9 至 2.0 个百分点，pass@8 提升 1.2 至 3.4 个百分点。在冻结的教师可达状态上……

    arXiv:2611.07959v1 Announce Type: cross  Abstract: On-policy distillation trains a student on its own rollouts, so states the teacher would reach but the student does not are seldom supervised. We introduce CoRD (coupled-rollout distillation), which keeps the on-policy loss and moves only the distribution of training states toward the teacher, drawing rollouts from a geometric interpolation of student and teacher through the maximal coupling of speculative decoding. The coupling keeps the student's token wherever the two distributions agree and otherwise substitutes a token from the part of the target the student underweights, so interventions are as rare as any exact sampler allows and recede as the student learns. Across six base students from two model families (Qwen3 and Qwen3.5, 0.6B to 4B), CoRD raises average pass@1 over on-policy distillation by 0.9 to 2.0 points and pass@8 by 1.2 to 3.4 points on five public math benchmarks at equal steps and data. On frozen teacher-reached st
    
[^72]: 一个用于预测和理解前沿人工智能系统失控的行为框架

    A Behavioural Framework for Predicting and Understanding Loss of Control in Frontier Artificial Intelligence Systems

    [https://arxiv.org/abs/2611.07953](https://arxiv.org/abs/2611.07953)

    本文提出一个行为框架，将前沿AI系统的失控风险分解为动机错位、危害促成能力和监控规避三个因素，并验证了其失控评分在13个前沿模型上对实际失败率的前瞻预测能力（Pearson r = 0.84），且表明风险无法仅由模型能力解释。

    

    前沿人工智能系统越来越多地在更长的时间跨度上运行，而人类监督却在逐渐减少，这带来了失控的风险：即系统采取超出人类有效控制的行为，并可能造成不可逆转的后果。由于此类故障可能只有在难以遏制时才会显现出来，因此必须对其进行前瞻性评估。我们提出了一个行为框架，将失控风险分解为三个因素：动机错位、促成危害的能力以及规避监控。我们通过13个可测量的方面和29个评估任务对这些因素进行操作化，其得分被综合为一个单一的失控（LoC）评分。在13个前沿模型上，失控评分表现出很强的前瞻有效性，与受控长时程风险场景中的失败率相关（Pearson r = 0.84）。风险在各模型间呈现系统性变化，且无法仅用能力来解释。

    arXiv:2611.07953v1 Announce Type: new  Abstract: Frontier artificial intelligence (AI) systems are increasingly operating over longer horizons with diminishing human oversight, raising the risk of loss of control: systems acting beyond effective human control with potentially irreversible consequences. Because such failures may become observable only after containment is difficult, they must be assessed prospectively. We introduce a behavioural framework that decomposes loss-of-control risk into three factors: misaligned motive, harm-enabling capability, and monitoring evasion. We operationalise these factors through 13 measurable aspects and 29 evaluation tasks, whose scores are combined into a single loss-of-control (LoC) score. Across 13 frontier models, the LoC score shows strong prospective validity, correlating with failure rates in contained long-horizon risk scenarios (Pearson r = 0.84). Risk varies systematically across models and cannot be explained by capability alone. We fu
    
[^73]: 先于结构的不变性：表征学习中涌现压缩的两种模式

    Invariance before structure: two modes of emergent compression in representation learning

    [https://arxiv.org/abs/2611.07952](https://arxiv.org/abs/2611.07952)

    该论文提出EARL框架，将表征学习中的涌现压缩区分为状态空间压缩与参数空间压缩两种模式，并以Fisher敏感性和系综级信息论度量加以形式化，从而解释生物与人工神经网络之间表征不变性的来源。

    

    学习到的表征在生物神经网络与人工神经网络之间展现出显著的不变性，即使其底层基质不同，且受漂移、异质性和初始化等因素的变化影响。为了理解这种不变性，我们提出了涌现自适应表征学习框架（EARL），该框架区分了学习表征中的两种涌现模式：状态空间压缩，即学习到的表征对自身未来的预测能力超越了网络中单个单元所能实现的预测；以及参数空间压缩，即具有不同微观参数的网络收敛到相同的表征。我们将这两种模式形式化为Fisher敏感性，并为每种模式引入了系综级的信息论度量。在一个带有训练瓶颈的循环储备池实验中，参数空间压缩在训练的前几个epoch内即出现上升，且部分独立于任务本身，而状态空间压缩……

    arXiv:2611.07952v1 Announce Type: new  Abstract: Learned representations show striking invariance across biological and artificial neural networks, with different substrates and variations driven by drift, heterogeneity and initialisation. To understand this invariance, we propose Emergent Adaptive Representation Learning (EARL), a framework that distinguishes two modes of emergence in learned representations: state-space compression, where a learned representation gains predictive power over its own future beyond what the network's units can predict individually, and parameter-space compression, where networks with different micro-level parameters converge to the same representation. We formalise both as Fisher sensitivities and introduce ensemble-level information-theoretic measures for each. In a recurrent reservoir with a trained bottleneck, parameter-space compression rises within the first few epochs of training and partly independently of the task, whereas state-space compressio
    
[^74]: 面向多模态胸片与电子健康记录分类的类别不平衡策略的泄漏与标签感知基准测试

    Leakage- and Label-Aware Benchmarking of Class-Imbalance Strategies for Multimodal Chest X-Ray and EHR Classification

    [https://arxiv.org/abs/2611.07950](https://arxiv.org/abs/2611.07950)

    该研究对多模态胸片与EHR类别不平衡基准进行审计，揭示了记录级数据划分导致的标签泄漏问题——患者标签由单次住院的诊断代码生成并复制到其所有影像，造成基准性能虚高，因此需要采用患者级数据划分与标签感知的评估方式。

    

    针对多模态胸片（CXR）与电子健康记录（EHR）预测的类别不平衡策略基准，通常在记录级划分的数据队列上进行评估，其中一名患者会贡献多张放射影像。我们在涵盖八个诊断目标、包含10,479名患者的29,083张MIMIC-CXR影像的基准上进行了审计。首先，记录级划分的基准具有误导性：一个仅复制开发集中具有相同EHR向量的影像标签的查找表，其表现几乎与最佳模型相当；而在未见过的患者上，所有模型的表现都退化到将每个病例都预测为阳性的水平。其次，将标签追溯至MIMIC-IV解释了原因：每位患者的标签是基于单次住院诊断代码的关键词匹配结果，并连同每位患者的一个EHR向量一起被复制到该患者的所有影像上，因此约三分之一的影像所携带的标签与其自身住院记录的诊断代码不符。第三，在患者分离划分下……（原文摘要在此处截断）

    arXiv:2611.07950v1 Announce Type: new  Abstract: Benchmarks of class-imbalance strategies for multimodal chest X-ray (CXR) and electronic health record (EHR) prediction are often evaluated on record-level splits of cohorts in which one patient contributes several radiographs. We audit such a benchmark on 29,083 MIMIC-CXR radiographs of 10,479 patients with eight diagnosis targets. First, the record-split benchmark was misleading: a lookup table that copies the label of a development-set radiograph with the same EHR vector nearly matched the best model, and on unseen patients every model fell to the level of predicting every case positive. Second, tracing the labels back to MIMIC-IV explained why: each patient's labels are keyword matches on the diagnosis codes of a single admission, copied, with one EHR vector per patient, to all of the patient's radiographs, so about a third of the radiographs carry labels that differ from the codes of their own admission. Third, under patient-separat
    
[^75]: 多水平函数数据中条件均值与变异性的联合建模：在AEGIS研究餐后血糖轨迹中的应用

    Joint Modeling of Conditional Mean and Variability in Multilevel Functional Data: Application to Postprandial Glucose Trajectories in the AEGIS Study

    [https://arxiv.org/abs/2611.07937](https://arxiv.org/abs/2611.07937)

    该论文提出了一种异方差函数混合效应回归框架，联合建模多水平重复函数轨迹的条件均值与方差，并通过参与者层面的自助法量化参数不确定性，应用于AEGIS研究揭示了各因素对餐后血糖均值与变异性的时依赖差异化影响。

    

    函数数据分析为研究协变量与高频生物信号之间的时变关联提供了框架。在数字健康领域，参与者通常会贡献重复的函数轨迹，然而现有的多水平回归方法主要关注条件均值，忽视了变异性及其在生物异质性中的作用。我们提出了一种异方差函数混合效应回归框架，联合建模重复轨迹的条件均值与方差，并在两个成分中都引入了参与者特定的随机效应。我们还开发了一种参与者层面的自助法（bootstrap）来量化估计的函数参数的不确定性。我们将该方法应用于AEGIS研究中的餐后血糖轨迹数据。结果显示，餐前血糖、碳水化合物摄入量以及餐食类型与血糖变异性呈现出明显不同的时依赖关联，而HbA1c则与条件均值关联更强。

    arXiv:2611.07937v1 Announce Type: cross  Abstract: Functional data analysis provides a framework for studying time-varying associations between covariates and high-frequency biological signals. In digital health, participants often contribute repeated functional trajectories, yet existing multilevel regression methods largely focus on the conditional mean, overlooking variability and its role in biological heterogeneity. We propose a heteroskedastic functional mixed-effects regression framework that jointly models the conditional mean and variance of repeated trajectories, with participant-specific random effects in both components. We also develop a participant-level bootstrap to quantify uncertainty in estimated functional parameters. We apply the method to postprandial glucose trajectories from the AEGIS study. Premeal glucose, carbohydrate intake, and meal type show distinct time-dependent associations with glucose variability, whereas HbA1c is more strongly associated with the con
    
[^76]: 蒙眼控制器：面向强化学习的决策模型引导训练

    The Blindfolded Controller: Decision Model-Guided Training for Reinforcement Learning

    [https://arxiv.org/abs/2611.07934](https://arxiv.org/abs/2611.07934)

    本文提出决策模型引导训练（DMGT）方法，利用决策模型初始化强化学习策略，使智能体从最初的回合即可到达目标并获得奖励，再通过强化学习将策略优化到超越决策模型的水平，从而有效解决了稀疏奖励任务中的探索难题。

    

    强化学习（RL）提供了一个通过与环境的交互来实现持续改进的框架。然而，强化学习智能体在面对具有探索挑战和稀疏奖励的任务时表现不佳，因为从零开始学习高效策略既缓慢且样本效率低下。相比之下，决策模型（DM）在给定环境和任务描述的情况下，能够在稀疏奖励任务中表现良好，但由于它们不从奖励中学习，因此无法向最优解方向改进。在本文中，我们提出了决策模型引导训练（DMGT），这是一种利用带有环境和任务描述的决策模型来初始化强化学习策略的技术，结合了决策模型无需训练即可行动的能力与强化学习从奖励中改进的能力。智能体无需从零开始探索，而是从最初的训练回合起就能到达目标并获得奖励，随后强化学习再将策略改进到超越决策模型所能达到的水平。

    arXiv:2611.07934v1 Announce Type: new  Abstract: Reinforcement learning (RL) provides a framework for continuous improvement through interaction with an environment. However, RL agents struggle against tasks that offer exploration challenges and sparse rewards, because learning efficient policies from scratch is slow and sample-inefficient. In contrast, decision models (DMs), given a description of the environment and task, can perform well in sparse-reward tasks, but, as they do not learn from reward, they are not able to improve towards the optimal solution. In this paper, we present Decision Model-Guided Training (DMGT), a technique that uses a DM with a description of the environment and task to initialise an RL policy, combining the DM's ability to act without training with RL's ability to improve from reward. Instead of exploring from scratch, the agent reaches the goal and receives reward from its first episodes, and RL then improves its policy beyond what the DM achieves. We de
    
[^77]: AttnSR：面向长程智能体强化学习的注意力步级奖励

    AttnSR: Attention Step Reward for Long-Horizon Agentic RL

    [https://arxiv.org/abs/2611.07932](https://arxiv.org/abs/2611.07932)

    提出 AttnSR 步级奖励增强方法，利用训练回放中的策略注意力反向追踪为各步骤分配奖励，无需额外奖励模型，即在 ALFWorld 上将 Qwen2.5-7B 的成功率从 91.2% 提升至 97.1%。

    

    长程LLM智能体需要进行多步决策，然而仅使用结果奖励的强化学习难以对各个步骤提供直接的监督信号。我们提出了AttnSR，这是一种步级奖励增强方法，它从训练回放中提取的策略注意力推导奖励分配，无需单独训练的奖励模型，也无需额外的 rollout。AttnSR 通过在策略模型的注意力图中反向追踪，来估计先前步骤与最终响应的相关性。终端步骤相对于运行基线进行度量，其结果决定这些奖励是强化还是惩罚所选择的步骤。在受控的 Qwen2.5-7B-Instruct 实验中，AttnSR 在 ALFWorld 上所有评估检查点均持续优于 PPO 基线，将最终留出集成功率从 91.2% 提升至 97.1%。消融实验表明，其收益取决于中间奖励的分配方式，而非简单的累加。

    arXiv:2611.07932v1 Announce Type: cross  Abstract: Long-horizon LLM agents make multi-step decisions, yet outcome-reward-only reinforcement learning provides little direct supervision for individual steps. We introduce AttnSR, a step-level reward augmentation method that derives reward allocations from policy attention extracted during training replay, without a separately trained reward model or additional rollouts. AttnSR traces backward through the policy model attention graph to estimate the relevance of earlier steps to the final response. The terminal step, measured relative to a running baseline, determines whether these rewards reinforce or penalize the selected steps. In a controlled Qwen2.5-7B-Instruct study, AttnSR consistently outperforms the PPO baseline across all evaluated checkpoints on ALFWorld, raising final held-out success from 91.2% to 97.1%. The ablation experiment shows that the benefit depends on how intermediate rewards are assigned, rather than simple accumula
    
[^78]: Flux 作为编解码器：基于 Flux 的零样本有损压缩

    Flux as a Codec: Zero-Shot Lossy Compression with Flux

    [https://arxiv.org/abs/2611.07929](https://arxiv.org/abs/2611.07929)

    本文提出 FluxC，一种以 Flux 为压缩先验的零样本编解码器，通过系统分析编码路径与重建方法对率-失真-感知性能的影响，发现基于 ODE 的重建在低码率下失真更低，整体性能可与现有方法竞争甚至超越。

    

    我们提出 FluxC，一种以 Flux 作为压缩先验的零样本编解码器。尽管 DiffC 已经证明预训练扩散模型可以充当零样本编解码器，我们探究了编码路径和重建方法的选择如何影响率-失真-感知行为。我们从三个维度对 FluxC 进行分析：编码成本与比特分配、重建方法以及低码率下的失真。在匹配信噪比（SNR）的条件下，我们表明路径选择不影响理想信息率，但会改变比特在各时间步之间的分配方式。在重建方面，我们比较了祖先采样、单步 ODE 重建和多步 ODE 重建。分析表明，基于 ODE 的重建能够比祖先采样实现更低的失真，尤其是在低码率区间。在高斯源和自然图像上的实验验证了这一预测。此外，FluxC 的性能与现有方法相当，在某些情况下甚至更优。

    arXiv:2611.07929v1 Announce Type: new  Abstract: We propose FluxC, a zero-shot codec that uses Flux as a compression prior. Although DiffC has shown that pretrained diffusion models can serve as zero-shot codecs, we ask how the choice of coding path and reconstruction method affects rate--distortion--perception behavior. We analyze FluxC along three axes: coding cost and bit allocation, reconstruction method, and low rate distortion. At matched signal-to-noise ratio (SNR), we show that the path choice does not affect the ideal information rate, but changes how bits are allocated across time steps. For reconstruction, we compare ancestral sampling, one-step ODE reconstruction, and many-step ODE reconstruction. Our analysis shows that ODE-based reconstruction can achieve lower distortion than ancestral sampling, especially in the low rate regime. Experiments on Gaussian sources and natural images support this prediction. Moreover, FluxC performs competitively with, and in some cases outp
    
[^79]: 相同的答案，不同的未来：匹配的预测并不意味着语言模型中匹配的信念更新

    Same Answer, Different Future: Matched Predictions Do Not Imply Matched Belief Updating in Language Models

    [https://arxiv.org/abs/2611.07913](https://arxiv.org/abs/2611.07913)

    该论文发现，语言模型通过不同途径（观察结果、摘要、指令或激活干预）即使得出相同的概率预测，其内部信念状态也并不等价，因而在面对后续相同新证据时会以不同的方式更新信念，即“当前预测一致”不代表“信念更新机制一致”。

    

    语言模型在观察结果、阅读摘要、接收指令或经历激活干预后，可以产生相似的概率预测。然而这些途径是否会诱导出等价的信念状态仍不清楚：当前预测的一致性并不能说明模型将如何响应后续的证据。我们在一个具有解析贝叶斯参照的伯努利推断任务中研究了这一区别。在三个开源权重语言模型上，我们通过四种文本途径和两种激活层面的干预来准备模型状态，同时测量下一结果的续写与显式的概率估计，并从每个准备好的状态追加相同的新证据。我们通过有效证据权重（即初始信念等价于多少次新观察）来量化初始信念的持久性。我们发现，证据的格式决定了哪些读出（readout）能够表达其内容：结果序列主要影响……（原文摘要在此处截断）

    arXiv:2611.07913v1 Announce Type: new  Abstract: Language models can produce similar probabilistic predictions after observing outcomes, reading summaries, receiving instructions, or undergoing activation interventions. Whether these routes induce equivalent belief states remains unclear: agreement in current predictions does not establish how a model will respond to subsequent evidence. We investigate this distinction in a Bernoulli inference task with an analytic Bayesian reference. Across three open-weight language models, we prepare states through four textual routes and two activation-level interventions, measure both next-outcome continuations and explicit probabilistic estimates, and append identical new evidence from each prepared state. We quantify the persistence of the initial belief through its effective evidence weight, the number of new observations it is worth. We find that the format of evidence decides which readouts express its content: outcome sequences primarily aff
    
[^80]: EDGAR：基于合成失败样本的高效扩散驱动高斯虚拟形象精修

    EDGAR: Efficient Diffusion-Driven Gaussian Avatar Refinement via Synthetic Failure

    [https://arxiv.org/abs/2611.07909](https://arxiv.org/abs/2611.07909)

    提出EDGAR条件扩散框架，利用真实与合成样本组成的混合三元组数据集来精修高斯虚拟形象的降质渲染，在保持身份一致性的同时有效校正模糊、拉伸等伪影。

    

    高斯泼溅的最新进展已经实现了高效且照片级逼真的头部虚拟形象重建，然而基于单目视频或图像的方法仍受限于几何与纹理的欠约束问题，导致出现模糊、拉伸以及时间不稳定性等伪影。我们提出了EDGAR（Efficient Diffusion-Driven Gaussian Avatar Refinement，高效扩散驱动高斯虚拟形象精修），这是一个条件扩散框架，旨在通过校正伪影来提升高斯虚拟形象的质量，同时保持身份一致性。为了训练EDGAR，我们构建了EDGAR-SET，一个由真实采集样本和合成渲染样本组成的混合三元组数据集，通过将降质渲染图像与真值图像和参考图像配对，模拟真实的高斯虚拟形象失败案例。EDGAR通过一个融合模块来精修降质帧，该模块将自监督Sapiens模型提取的身份特征注入到单步扩散骨干网络中，从而实现有效的……

    arXiv:2611.07909v1 Announce Type: cross  Abstract: Recent advances in Gaussian splatting have enabled efficient and photorealistic head avatar reconstruction, yet monocular video-based or image-based methods still suffer from under-constrained geometry and texture, leading to artifacts such as blurring, stretching, and temporal instability. We present EDGAR (Efficient Diffusion-Driven Gaussian Avatar Refinement), a conditional diffusion framework designed to enhance the quality of Gaussian avatars by correcting artifacts while preserving identity fidelity. To train EDGAR, we construct EDGAR-SET, a hybrid triplet dataset composed of real-captured and synthetic-rendered samples, which simulates realistic Gaussian avatar failure cases by pairing degraded renderings with ground-truth and reference images. EDGAR refines degraded frames through a fusion block that injects identity features extracted by the self-supervised Sapiens model into a one-step diffusion backbone, enabling effective a
    
[^81]: 基于假设空间推理的碰撞结构恢复

    Structure from Collision via Hypothesis-Space Reasoning

    [https://arxiv.org/abs/2611.07900](https://arxiv.org/abs/2611.07900)

    提出通过假设空间推理从碰撞事件的多视角观测中恢复物体隐藏内部结构（如空腔），克服了空腔仅能通过表面运动间接观测、且因对应材料缺失而难以用密度表示直接优化的难题。

    

    了解物体的隐藏内部结构，包括空腔的存在与位置，对于预测性仿真、机器人操作和无损检测至关重要。然而，这类空腔在静态可见表面上不会留下任何直接痕迹，使得碰撞引起的变形成为推断其存在的少数线索之一。Structure from Collision（SfC）通过从碰撞事件的多视角观测中恢复物体的隐藏内部结构来解决这一问题。SfC 面临两大挑战：隐藏空腔只能通过表面运动间接观测，且空腔对应于材料的缺失，这使得使用标准密度表示进行直接优化变得困难。基于密度的 SfC 基线同时继承了这两个困难：表面匹配梯度无法直接指示空腔位置，而密度梯度在物体内部可能微弱或缺乏信息。

    arXiv:2611.07900v1 Announce Type: cross  Abstract: Knowledge of the hidden interior structure of an object, including the presence and location of cavities, is essential for predictive simulation, robotic manipulation, and nondestructive inspection. However, such cavities leave no direct signature on the static visible surface, making collision-induced deformation one of the few cues used to infer their presence. Structure from Collision (SfC) addresses this problem by recovering the hidden interior structure of an object from multiview observations of a collision event. SfC is challenging for two reasons: hidden cavities are only indirectly observable through surface motion, and voids correspond to the absence of material, making direct optimization using standard density representations difficult. The density-based SfC baseline inherits both difficulties. Surface-matching gradients do not directly indicate the cavity location, while density gradients can be weak or uninformative insi
    
[^82]: DISCERN：面向无保留集增量式机器遗忘的分布性合成回放

    DISCERN: Distributional Synthetic Replay for Retain-Free Incremental Machine Unlearning

    [https://arxiv.org/abs/2611.07892](https://arxiv.org/abs/2611.07892)

    DISCERN 通过为每个类别拟合欧氏与方向性参数分布并采样合成嵌入进行回放，实现了无需访问保留集的增量式类遗忘，在依次删除类别的同时确保已遗忘知识不被恢复并保留剩余类别的性能。

    

    arXiv:2611.07892v1 公告类型： cross 摘要：机器遗忘旨在在不从头重新训练的情况下，从已训练模型中移除特定训练数据的影响。然而在实际部署中，删除请求会随时间不断到来，这就需要增量式类遗忘（Incremental Class Unlearning, ICU）：类别被依次移除，已被遗忘的知识必须保持遗忘状态，同时剩余类别必须得到保留。我们提出了 DISCERN（DIstributional SynthetiC rEplay for Retain-free iNcremental unlearning），一种基于干预的类遗忘方法，它用从每个类别的参数化分布中采样的合成嵌入来替代对保留集的访问。与假设每个类质心周围为各向同性高斯分布的原型回放方法不同，DISCERN 为每个类别拟合欧几里得分布族（高斯、拉普拉斯、学生 t 分布）和方向性分布族（von Mises–Fisher、Fisher–Bingham），并通过统计检验从中进行选择，当没有任何分布族被接受时，则采用交叉验证……（摘要在此处截断）

    arXiv:2611.07892v1 Announce Type: cross  Abstract: Machine unlearning aims to remove the influence of specific training data from a trained model without retraining from scratch. In real deployments, however, deletion requests arrive over time, calling for incremental class unlearning (ICU): classes are removed sequentially, previously forgotten knowledge must remain forgotten, and the remaining classes must be preserved. We present DIstributional SynthetiC rEplay for Retain-free iNcremental unlearning (DISCERN), an intervention-based class unlearning method that replaces access to the retain set with synthetic embeddings sampled from per-class parametric distributions. Unlike prototype-based replay, which assumes an isotropic Gaussian around each class centroid, DISCERN fits Euclidean (Gaussian, Laplace, Student's $t$) and directional (von Mises--Fisher, Fisher--Bingham) families to each class and selects among them through statistical tests and, when none of them accepts, cross-valid
    
[^83]: 何时、何地、如何：面向表格基础模型的主动上下文学习

    When, Where, How: Active In-Context Learning for Tabular Foundation Models

    [https://arxiv.org/abs/2611.07886](https://arxiv.org/abs/2611.07886)

    本文针对表格基础模型提出主动上下文学习方法，研究了目标感知最近邻（TANN）采集和上下文期望预测信息增益（EPIC）两种策略，并在二十个基准上发现基于多样性的采集在初期迭代中最为有效。

    

    表格基础模型（TFMs）仅通过其上下文中的带标注行来适应新任务。然而，在现实场景中，标签的获取成本高昂，因为需要高度专业的领域知识来确保数据质量。主动上下文学习将经典的采集规则迁移到这一设定中，其中引导采集的模型会随着每个标签的添加而发生变化。我们探究了随着冻结的TFM获取更多上下文，多样性、预测不确定性和目标感知信息增益的价值如何变化。为此，我们研究了目标感知最近邻（TANN）采集方法和上下文中的期望预测信息增益（EPIC），并将它们与经典规则进行评估比较。我们在二十个基准数据集上，使用两个成熟的TFM和经典GBDT开展研究，得出了三个发现：(i) 何时：基于多样性的采集在最初的采集迭代中效果最佳；(ii) 何地：标注池中的行……

    arXiv:2611.07886v1 Announce Type: cross  Abstract: Tabular Foundation Models (TFMs) adapt to a new task only through the labeled rows in their context. However, labels are expensive to acquire in real-world scenarios due to the need for highly specific domain knowledge to ensure high-quality data. Active in-context learning transfers classical acquisition rules to this setting, in which the model that guides acquisition changes with every label added. We ask how the value of diversity, predictive uncertainty and target-aware information gain changes as a frozen TFM acquires more context. For that, we study Target-Aware Nearest Neighbor (TANN) acquisition and the Expected Predictive Information Gain in Context (EPIC) and evaluate them against classical rules. Our study across twenty benchmarks with two well-established TFMs and classical GBDTs highlights three findings: (i) When: diversity-based acquisition is best during the first acquisition iterations; (ii) Where: labeling pool rows 
    
[^84]: 哪些代码正确性信号能在分布偏移下依然有效？程序与判定正确性的隐状态及因果预测器

    Which Signals of Code Correctness Survive Distribution Shift? Hidden-State and Causal Predictors of Program and Verdict Correctness

    [https://arxiv.org/abs/2611.07876](https://arxiv.org/abs/2611.07876)

    本文系统比较了四类共六种免执行代码正确性预测器在基准与语言分布偏移下的稳健性，并将两种新颖的因果定位方法引入代码领域，同时评估其对程序正确性和模型自身判定可信度的预测能力。

    

    大语言模型越来越多地被用于评判代码，但在缺乏外部反馈的情况下，这类判断并不可靠，而通过执行测试或深度推理来验证这些判断的代价高昂。因此，免执行的正确性预测器利用模型输出概率、采样解或隐状态中的信号，来识别哪些判断值得进一步验证。然而，当这些预测器在跨项目、跨语言进行泛化时，可能会在分布偏移下性能退化。我们在 Qwen3-Coder-30B 和 gpt-oss-20b 上，比较了来自四个类别的六种免执行预测器，考察了 LiveCodeBench 与 BigCodeBench 之间的基准偏移，以及 Python、C++ 和 Java 之间的语言偏移。我们评估了两个目标：模型自身对程序的判定是否可信，以及程序本身是否正确。除已有方法外，我们还将两种新颖的因果定位方法适配到代码领域，这些方法已被证明更为稳健。

    arXiv:2611.07876v1 Announce Type: cross  Abstract: Large language models increasingly judge code, yet without external feedback such judgments are unreliable, and verifying them by executing tests or by extended reasoning is costly. Instead, execution-free correctness predictors use signals from the model's output probabilities, sampled solutions, or hidden states to identify judgments that warrant further verification. However, these predictors may degrade under distribution shifts when generalizing across projects and languages.   We compare six execution-free predictors from four families on Qwen3-Coder-30B and gpt-oss-20b under benchmark shift between LiveCodeBench and BigCodeBench and language shift across Python, C++ and Java. We evaluate two targets: whether the model's own verdict on a program can be trusted, and whether the program itself is correct. Besides established approaches, we adapt two novel causal localization methods to code, which have been shown to be more robust 
    
[^85]: PlayGen2：基于显式3D状态学习实时可玩世界

    PlayGen2: Learning Real-Time Playable Worlds with Explicit 3D State

    [https://arxiv.org/abs/2611.07867](https://arxiv.org/abs/2611.07867)

    PlayGen2通过将显式3D世界状态的演化与神经渲染相分离，并维护由体积占用栅格和身份索引实体3D状态构成的持久世界记忆，实现了具备长期空间一致性的实时可交互可玩世界。

    

    构建可玩的世界不仅仅是生成逼真的视频：它需要响应式的控制、持久的空间结构，以及与动态实体的连贯交互。若将生成的帧作为唯一的循环世界表示，会把上述需求与依赖视角的外观合成绑定在一起，使得在生成与交互中难以维持长期的空间和时间一致性。我们提出了PlayGen2，一种将学习到的3D状态演化与神经渲染相分离的实时交互式世界模型。PlayGen2维护一个持久的世界记忆，其中包含由体积占用栅格表示的静态几何结构，以及为玩家和其他实体建立的身份索引3D状态。在固定大小的局部区域内，其世界状态模型联合学习动作条件下的实体动力学、几何补全和实体发现，并将更新提交至持久的世界记忆。一个独立的神经渲染器（摘要原文在此处截断）……

    arXiv:2611.07867v1 Announce Type: cross  Abstract: Building a playable world requires more than generating plausible video: it demands responsive control, persistent spatial structure, and coherent interactions with dynamic entities. Using generated frames as the sole recurrent world representation ties these requirements to view-dependent appearance synthesis, making it difficult to maintain long-term spatial and temporal consistency in generation and interaction. We present PlayGen2, a real-time interactive world model that separates learned 3D state evolution from neural rendering. PlayGen2 maintains a persistent world memory comprising static geometry represented by a volumetric occupancy grid and identity-indexed 3D states for the player and other entities. Within a fixed-size local region, its world-state model jointly learns action-conditioned entity dynamics, geometry completion, and entity discovery, committing updates to a persistent world memory. A separate neural renderer s
    
[^86]: 计算而非权重访问：大语言模型作为临床预测网络的审计者

    Computation, not weight access: large language models as auditors of clinical prediction networks

    [https://arxiv.org/abs/2611.07847](https://arxiv.org/abs/2611.07847)

    通过预先设定的模拟实验，该研究发现在没有计算工具时仅凭提示中的原始权重无法可靠恢复植入的关系，表明大语言模型审计临床预测网络的关键在于计算能力（借助工具访问权重或预测）而非直接的权重访问。

    

    据报道，前沿大语言模型（LLM）能够读取、编辑和编写小型神经网络的权重，这为仅凭模型文件审计临床预测模型提供了可能。然而，公开演示中将读取原始权重与运行代码混在了一起。我们在内部预先设定的模拟实验中将这些条件区分开来。研究在模拟的蛛网膜下腔出血患者队列上训练了具有105至38,401个权重的Tanh网络，每个网络中植入了一个关系或不植入任何关系，其中一个网络所使用的标签在某个亚组中少记了死亡人数。GPT-6 Astra和其他四个GPT模型在默认或明确的系统指令下回答单个提示，条件为：提示中包含权重且不使用工具，或者使用仅能进行计算的工具，该工具或暴露权重（白盒），或仅暴露网络的预测（黑盒）。在不使用工具的情况下，Astra在353权重网络中恢复了植入的关系

    arXiv:2611.07847v1 Announce Type: cross  Abstract: Frontier large language models (LLMs) have been reported to read, edit and write the weights of small neural networks, raising the prospect of auditing clinical prediction models from their files alone. Public demonstrations, however, mixed reading raw weights with running code. We separated these conditions in internally pre-specified simulation experiments. Tanh networks of 105 to 38,401 weights were trained on simulated cohorts of patients with subarachnoid haemorrhage, each with a planted relationship or none, and one network was trained on labels that under-recorded deaths in a subgroup. GPT-6 Astra and four other GPT models answered single prompts under a default or an explicit system instruction, either with the weights in the prompt and no tools, or with a calculation-only tool that exposed the weights (white box) or only the network's predictions (black box). Without tools, Astra recovered planted relationships in 353-weight n
    
[^87]: 掩码扩散作为随机序贯吸附：一个精确可解的并行解码模型

    Masked diffusion as random sequential adsorption: an exactly solvable model for parallel decoding

    [https://arxiv.org/abs/2611.07845](https://arxiv.org/abs/2611.07845)

    该论文将掩码扩散模型并行解码的误差与驱动多粒子系统的熵产生和耗散联系起来，并把马尔可夫数据下的生成精确等价为一个可严格求解的随机序贯吸附过程，从而建立了并行解码的首个精确可解理论模型。

    

    掩码扩散模型通过逆转一个随机掩码过程来生成文本。与每次生成一个token的自回归大语言模型不同，它们可以并行揭示多个token，但代价是引入误差，因为同时揭示的token是彼此独立抽取的。我们证明这种权衡可以映射到驱动多粒子系统的统计物理中。(i) 对于任意数据，并行生成的误差等于生成过程相对于逆转掩码过程的熵产生。(ii) 对于马尔可夫数据，生成过程是一个带标记间隙的随机序贯吸附过程，我们对其任意步数均给出了精确解。(iii) 对于多步情况，误差变为缓慢驱动系统的耗散：去掩码调度扮演驱动协议的角色，步数扮演其持续时间的角色，而误差由一个热力学度规决定，我们从互信息中以闭式形式求得了该度规。

    arXiv:2611.07845v1 Announce Type: cross  Abstract: Masked diffusion models generate text by reversing a stochastic masking process. Unlike autoregressive LLMs, which generate text one token at a time, they can reveal many tokens in parallel, at the price of an error, since tokens revealed together are drawn independently. We show that this trade-off maps onto the statistical physics of driven many-particle systems. (i) For any data, the error of parallel generation is the entropy production of the generation process relative to the reversed masking process. (ii) For Markov data, generation is a random sequential adsorption process with labelled gaps, which we solve exactly for any number of steps. (iii) For many steps, the error becomes the dissipation of a slowly driven system: the unmasking schedule plays the role of the driving protocol, the number of steps that of its duration, and the error is set by a thermodynamic metric that we obtain in closed form from the mutual information 
    
[^88]: 结合深度学习与基于可解释AI（SHAP）的多样化多特征分析的增强型MRI脑肿瘤检测

    Enhanced MRI brain tumor detection using deep learning in conjunction with explainable AI SHAP based diverse and multi feature analysis

    [https://arxiv.org/abs/2611.07843](https://arxiv.org/abs/2611.07843)

    本研究将LBP、Gabor滤波器、小波变换、CNN等多种特征表示方法与多种机器学习算法相结合用于脑肿瘤检测，发现LBP结合SVC和CNN取得了高特异性和准确率，并借助可解释AI（SHAP）对多特征分析进行了解释。

    

    医学成像领域的最新创新显著改善了脑肿瘤的识别，超越了分辨率低、存在辐射暴露和对比度有限等缺陷的传统诊断方法。磁共振成像（MRI）因其高分辨率和无创特性，在精确、准确地刻画肿瘤特征方面发挥着关键作用。本研究探讨了多种特征表示方案之间的协同作用，包括局部二值模式（LBP）、Gabor滤波器、离散小波变换、快速傅里叶变换、卷积神经网络（CNN）和灰度行程长度矩阵，并结合五种学习算法，即k近邻、随机森林、支持向量分类器（SVC）、概率神经网络（PNN）和CNN。实证结果表明，LBP与SVC和CNN相结合获得了高特异性和准确率，使其成为一种有前景的基于MRI的肿瘤诊断方法。进一步的……

    arXiv:2611.07843v1 Announce Type: cross  Abstract: Recent innovations in medical imaging have markedly improved brain tumor identification, surpassing conventional diagnostic approaches that suffer from low resolution, radiation exposure, and limited contrast. Magnetic Resonance Imaging (MRI) is pivotal in precise and accurate tumor characterization owing to its high-resolution, non-invasive nature. This study investigates the synergy among multiple feature representation schemes such as local Binary Patterns (LBP), Gabor filters, Discrete Wavelet Transform, Fast Fourier Transform, Convolutional Neural Networks (CNN), and Gray-Level Run Length Matrix alongside five learning algorithms namely: k-nearest Neighbor, Random Forest, Support Vector Classifier (SVC), and probabilistic neural network (PNN), and CNN. Empirical findings indicate that LBP in conjunction with SVC and CNN obtained high specificity and accuracy, rendering it a promising method for MRI-based tumor diagnosis. Further t
    
[^89]: 估计超越条件坍缩的生成式代理模型的无谱信息界

    Estimating Spectrum-Free Information Bounds for Generative Surrogates Beyond Conditional Collapse

    [https://arxiv.org/abs/2611.07826](https://arxiv.org/abs/2611.07826)

    针对高能物理条件生成式代理模型评估中一维边缘散度指标会奖励条件坍缩的问题，本文提出采用预报检验领域的CRPSS技巧评分并以精确边缘分布为参考，同时引入数据驱动的k近邻外推方法来估计不受评估谱影响的无谱信息界，从而实现更可靠的模型评估。

    

    高能物理中的条件生成式代理模型（例如从部分子预测重建喷注的模型）通常在一维边缘散度等指标上进行评估。单独使用这些指标可能会奖励条件坍缩现象。在CMS模拟基准测试中，一个忽略条件输入的采样器在KL散度和Wasserstein-1距离下于所有四个目标边缘分布上都击败了经过训练的混合密度网络（MDN），并且在其中两个目标上无法与条件流匹配（CFM）模型稳定地区分开来。我们转而采用来自预报检验领域的连续分级概率技巧评分（CRPSS）来评估模型，并以精确的边缘分布作为参考。我们展示了全局技巧评分如何依赖于评估谱，同时也报告了在条件变量分箱内的技巧评分——在分箱内部这种依赖性基本消失。随后，我们引入了一种数据驱动的k近邻（k-NN）外推方法，用于估计……（原文摘要在此处截断）

    arXiv:2611.07826v1 Announce Type: cross  Abstract: Conditional generative surrogates in high-energy physics, such as models that predict reconstructed jets from partons, are often evaluated on $1\text{D}$ marginal divergences among other metrics. Taken alone these metrics can reward conditional collapse. On a CMS simulation benchmark, a sampler that ignores the conditioner beats a trained Mixture Density Network (MDN) on all four target marginals under both KL and Wasserstein-1, and cannot be stably separated from a Conditional Flow Matching (CFM) model on two of the four. We instead score the models with the Continuous Ranked Probability Skill Score (CRPSS) from forecast verification, taking the exact marginal as the reference. We show how global skill depends on the evaluation spectrum, and we also report it within bins of a conditioning variable, where that dependence largely goes away. We then introduce a data-driven $k$-Nearest-Neighbor ($k$-NN) extrapolation that estimates the be
    
[^90]: 可解码性不等于有效性：评估语言模型概念探针的干扰因素敏感性

    Decodability Is Not Validity: Evaluating Nuisance Sensitivity in Language Model Concept Probes

    [https://arxiv.org/abs/2611.07805](https://arxiv.org/abs/2611.07805)

    论文提出用“判别性证据”来评估概念探针的干扰因素敏感性，并证明探针的可解码性（预测准确率）并非有效性的可靠代理——AUROC 统计等价的探针对相关干扰因素的敏感性可相差数倍，且两者秩相关性在不同设置间会反转符号。

    

    两个概念探针在从语言模型的隐藏状态中预测某一概念时可能具有同样的准确率，但它们的评分中却可能包含不同比例的目标概念与相关干扰属性。在四种“目标—干扰因素”设置（CEBaB 中的两个方面情感；BiasBios 中的职业与性别；亚马逊评论情感结合否定词；以及结合第一人称框架）、两个大语言模型以及一组概念提取器中，我们将可解码性与判别性证据进行比较：即在固定目标的情况下评分对干扰因素的依赖程度，以及评分对旨在仅改变干扰因素的输入编辑的响应——这些编辑经两个大语言模型评判者过滤，并与安慰剂编辑进行对比。在相关训练数据的条件下，在留出集 AUROC 上统计等价的概念提取器，在四个设置中的三个里对干扰因素的敏感性可相差数倍，而且可解码性并不是干扰因素敏感性的可靠代理指标：两者的秩相关系数在不同设置间甚至会改变符号。

    arXiv:2611.07805v1 Announce Type: new  Abstract: Two concept probes can be equally accurate at predicting a concept from a language model's hidden states while their scores contain different mixtures of the target concept and a correlated nuisance attribute. In four target--nuisance settings (two aspect sentiments in CEBaB; occupation and gender in BiasBios; Amazon review sentiment with negation, and with first-person framing), two LLMs, and a population of concept extractors, we compare decodability with discriminant evidence: the score's dependence on the nuisance at fixed target, and its response to input edits intended to change only the nuisance, filtered by two LLM judges and compared against placebo edits. Under correlated training data, extractors that are statistically equivalent in held-out AUROC differ up to several-fold in nuisance sensitivity in three of the four settings, and decodability is not a reliable proxy for it: their rank correlation changes sign across settings.
    
[^91]: EnergyBench：一个面向欧洲电力市场的实时、与交易闸门对齐的日前预测基准

    EnergyBench: a live, gate-aligned benchmark of day-ahead forecasts for European power markets

    [https://arxiv.org/abs/2611.07784](https://arxiv.org/abs/2611.07784)

    EnergyBench 是一个对欧洲电力市场日前预测进行实时评估的基准，它在每日交易闸门关闭前自行运行各模型，并使用冻结且公开的统一信息集（含数值天气预报），从而避免了回测数据泄露和固定测试集过拟合的问题。

    

    能源预测模型通常在回测和固定的公开数据集上进行比较。回测可能使用在预测本应做出的时刻尚不存在的数据，而一个在多项研究间重复使用、且可能已包含在基础模型预训练数据中的固定测试集，容易导致对基准的过拟合。EnergyBench 改为对每个参赛模型进行实时评估。每天中欧时间（CET/CEST）11:30，即日前（SDAC）交易闸门关闭前三十分钟，每个模型在15分钟粒度上对六个竞价区域（比利时、荷兰、法国、德卢、丹麦DK1、丹麦DK2）预测次日的价格、风电与光伏发电量以及负荷。与现有的实时基准不同，EnergyBench 自行运行这些模型，使用一个在截止时刻冻结并公开发布的统一信息集，其中包括三个数值天气预报模型，从而使基础模型能够在真实的市场截止期限下，在相同天气协变量条件下进行对比。

    arXiv:2611.07784v1 Announce Type: new  Abstract: Energy forecasting models are usually compared on backtests and fixed public datasets. Backtests can use data that did not yet exist when the forecast would have been made, and a fixed test set, reused across studies and possibly contained in the pretraining data of foundation models, invites overfitting to the benchmark. EnergyBench evaluates every entry live instead. Each day at 11:30 CET/CEST, thirty minutes before the day-ahead (SDAC) gate closes, every model forecasts the next day's price, wind and solar generation and load on the 15-minute grid for six bidding zones (BE, NL, FR, DE-LU, DK1, DK2). Unlike existing live benchmarks, EnergyBench runs the models itself, on one information set that it freezes at the cut-off and publishes, including three numerical weather prediction models, so that foundation models can be compared with and without the same weather covariates under the actual market deadline. Errors are pooled per series 
    
[^92]: PBPO：面向扩散语言模型的前缀块策略优化

    PBPO: Prefix-Block Policy Optimization for Diffusion Language Models

    [https://arxiv.org/abs/2611.07783](https://arxiv.org/abs/2611.07783)

    提出PBPO方法，通过构建前缀块状态并利用单次模型评估定义分解式单步块似然代理，将策略优化自然扩展到扩散语言模型的块级别，使个别位置的变化能在裁剪前传播至整个块。

    

    扩散大语言模型通常通过跨块自回归推进、同时在每个块内迭代去噪多个token的方式来解码响应。这种结构使得块级策略优化成为自回归策略优化的自然适配。然而，完整生成块的确切似然是难以求解的，因为它需要对去噪过程中的中间状态和揭示顺序进行边缘化处理。为了实现块级的优化公式，我们构建了前缀块状态，并利用单次模型评估定义了块似然的分解式单步代理。由此得到的块代理比率是其各个token代理比率的乘积，使得个别位置的变化能够在裁剪之前传播至整个块。因此，我们提出了前缀块策略优化（PBPO），该方法保留了块级条件化机制，同时应用代理重要性加权……（摘要不完整，后续内容缺失）

    arXiv:2611.07783v1 Announce Type: new  Abstract: Diffusion Large Language Models (dLLMs) commonly decode responses by proceeding autoregressively across blocks while iteratively denoising multiple tokens within each block. This structure makes block-level policy optimization a natural adaptation of autoregressive policy optimization. However, the exact likelihood of a completed block is intractable because it marginalizes over the intermediate states and reveal orders of the denoising process. To realize the block-level formulation, we construct a prefix-block state and use a single model evaluation to define a factorized one-step surrogate for the block likelihood. The resulting block surrogate ratio is a product of its token surrogate ratios, allowing variation at individual positions to propagate across the entire block before clipping. We therefore introduce Prefix-Block Policy Optimization (PBPO), which preserves block-level conditioning but applies surrogate importance weighting 
    
[^93]: 适配器即数据：LoRA的潜空间

    Adapters as Data: The Latent Space of LoRAs

    [https://arxiv.org/abs/2611.07775](https://arxiv.org/abs/2611.07775)

    该论文将LoRA适配器视为一种数据模态，训练Transformer自编码器构建非线性潜空间，证明该潜空间不仅保留了权重空间中采样与编辑等能力，还对位移具有更强鲁棒性，退化时产生扭曲但仍然有效的面孔而非纹理。

    

    LoRA适配器通常被视为最终完成的成品工件，而不是一种数据模态。先前的研究表明，LoRA适配器分布在一个低维流形上，线性PCA基已能将其组织得足够好，从而支持采样、编辑和求逆等多种操作。我们追问的是，在此之上，一个学习得到的非线性潜空间还能带来什么。我们在同一群LoRA身份适配器上训练了一个Transformer自编码器，将每个适配器映射到潜空间中，并将每种操作与直接作用于原始适配器权重的相同操作进行对比。该潜空间保留了权重空间中已有的能力，如采样多样化的身份以及编辑个体属性。除了保留这些能力之外，我们还证明潜空间对位移的鲁棒性远高于权重空间：当发生退化时，它生成的是扭曲但仍然有效的面孔，而不是纹理。固定维度的权重空间表示无法原生承载适配器（摘要在此处被截断）。

    arXiv:2611.07775v1 Announce Type: new  Abstract: LoRA adapters are usually treated as finished artifacts rather than as a data modality. Prior work has shown that LoRA adapters populate a low-dimensional manifold that a linear PCA basis already organizes well enough to support different operations such as sampling, editing, and inversion. We ask what a learned, nonlinear latent adds beyond this. We train a transformer autoencoder on the same population of LoRA identity adapters, mapping each into a latent space and compare every operation against the identical operation applied directly to the raw adapter weights. The latent preserves the capabilities already available in weight space, sampling diverse identities and editing individual attributes. Beyond preservation, we show that the latent is far more robust to displacement than weight space, degrading into distorted but still valid faces rather than texture. Fixed-dimensional weight-space representations cannot natively host adapter
    
[^94]: 具有任意延迟的博弈中的最优学习：自博弈与对抗对手场景的两全其美

    Optimal Learning in Games with Arbitrary Delays: Best-of-Both-Worlds for Self-Play and Adversarial Opponents

    [https://arxiv.org/abs/2611.07773](https://arxiv.org/abs/2611.07773)

    提出单一算法MAO-OMD，在具有任意未知反馈延迟的零和博弈中，同时实现了自博弈时的最优快速遗憾速率 O(σ_max+1) 和对抗任意对手时极小极大最优的 O(√(T+d_tot)) 保证。

    

    我们研究了具有延迟反馈的零和博弈中的学习问题。我们提出了多锚点乐观在线镜像下降算法，这是一种单一算法，能够在光滑博弈的自博弈中获得最优的快速遗憾速率，同时保持对任意对手的极小极大最优保证。特别地，在具有任意且未知反馈延迟的光滑零和博弈中，当双方参与者都使用MAO-OMD时，该算法获得 O(σ_max+1) 的社会遗憾和个体遗憾，其中 σ_max 表示未完成梯度的最大数量；同时对任意对手保证极小极大最优的 O(√(T+d_tot)) 个体遗憾，其中 d_tot 为总延迟。此外，我们证明了通过将问题标准黑盒归约到无延迟设定，无法同时获得这些最优速率。我们还将结果扩展到多人博弈，以及具有参与者相关延迟的博弈。

    arXiv:2611.07773v1 Announce Type: new  Abstract: We study learning in zero-sum games with delayed feedback. We introduce Multi-Anchor Optimistic Online Mirror Descent (MAO-OMD), a single algorithm which obtains optimal fast rates in self-play on smooth games, while simultaneously maintaining minimax optimal guarantees against arbitrary opponents. In particular, in smooth zero-sum games with arbitrary and unknown feedback delays, MAO-OMD obtains $O(\sigma_{\max}+1)$ social and individual regrets when used by both players, where $\sigma_{\max}$ denotes the maximum number of outstanding gradients, while also guaranteeing minimax optimal $O(\sqrt{T+d_{\mathrm{tot}}})$ individual regret against arbitrary opponents, where $d_{\mathrm{tot}}$ is the total delay. Moreover, we show that these optimal rates cannot be obtained simultaneously using the standard black-box reduction to the undelayed setting. We also extend our results to multi-player games, and to games with player-dependent delay se
    
[^95]: SAE（稀疏自编码器）训练数据的缩放定律

    Scaling Laws for SAE Training Data

    [https://arxiv.org/abs/2611.07756](https://arxiv.org/abs/2611.07756)

    在固定训练预算下，SAE 所需的独特激活数据量存在饱和点——缓冲区达到一定规模后，重放数据与新鲜激活值效果相当，且缓冲区无需随训练预算扩大；作者据此提出并在多种模型、层级和 SAE 架构上验证了一条刻画多样性与重复权衡的有限数据缩放定律。

    

    激活值存储通常是稀疏自编码器（SAE）训练中的瓶颈。我们研究了在总训练预算固定的情况下，SAE 需要多少条独特的激活值。随着缓冲区规模增大，质量迅速提升，随后进入一个平台期：在此阶段重放缓冲区数据与在整个训练过程中持续使用新鲜激活值几乎同样有效。超过这一点后，额外的独特激活值对重构效果或可解释性几乎没有额外收益，因此缓冲区规模无需随训练预算增长。我们用一条针对有限数据的缩放定律来描述这种多样性与重复之间的权衡。我们在不同的字典规模与 token 预算、Llama-3.1 的早期、中期和后期层、Qwen3 的不同模型规模，以及 BatchTopK、TopK 和 JumpReLU 等多种 SAE 上验证了该定律。通过在数学文本和 Python 代码上进行扫描，我们测试了改变激活值来源的影响。评估涵盖解释方差（EV）、重构 MSE、特征吸收、自动化（评估）……

    arXiv:2611.07756v1 Announce Type: cross  Abstract: Activation storage is often a bottleneck in sparse autoencoder (SAE) training. We study how many unique activations an SAE needs when its total training budget is fixed. Quality improves rapidly as the buffer grows, then reaches a plateau where replaying the buffer is nearly as effective as using fresh activations throughout training. Beyond this point, additional unique activations bring little benefit to reconstruction or interpretability, so the buffer need not grow with the training budget. We describe this tradeoff between diversity and repetition with a scaling law for limited data. We validate it across dictionary sizes and token budgets, early, middle, and late layers of Llama-3.1, Qwen3 model sizes, and BatchTopK, TopK, and JumpReLU SAEs. Sweeps using mathematical text and Python code test the effect of changing the activation source. Evaluation covers explained variance (EV), reconstruction MSE, feature absorption, automated 
    
[^96]: 不平衡数据集与开集场景下的神经坍缩

    Neural Collapse on Imbalanced Datasets and Open-Set Scenario

    [https://arxiv.org/abs/2611.07753](https://arxiv.org/abs/2611.07753)

    本文基于类别平均损失（而非全局平均损失）的优化理论，证明了无论数据集是否平衡，甚至在部分类别完全没有训练样本的开集场景下，当各类别平均损失达到最小值时，标准的神经坍缩结构依然会涌现。

    

    arXiv:2611.07753v1 公告类型：新论文 摘要：神经坍缩（NC）描述了当分类模型的损失达到最小值时，样本特征和分类器向量所呈现出的优雅几何结构。然而，类别不平衡通常会扭曲标准的神经坍缩现象。本文从理论上重新表述了不平衡数据集和开集设置下的神经坍缩。我们的分析基于对各类别平均损失的优化问题，而非以往神经坍缩研究中采用的全局平均损失。我们证明了当各类别平均损失达到其最小值时，无论数据集是平衡的还是不平衡的，对于交叉熵（CE）和二元交叉熵（BCE）损失，标准的神经坍缩均可实现。我们进一步将该分析扩展到一种受开集启发的场景，其中某些类别不包含任何训练样本，但在损失函数中仍被分配了分类器向量。我们发现，在这种场景下，当各类别平均损失达到其最小值时，标准的神经坍缩现象仍然会出现。

    arXiv:2611.07753v1 Announce Type: new  Abstract: Neural collapse (NC) describes the elegant geometric structure that sample features and classifier vectors exhibit when a classification model's loss reaches its minimum. Class imbalance, however, generally distorts standard NC. In this paper, we theoretically reformulate NC on imbalanced datasets and in open-set settings. Our analysis is based on an optimization problem over class-wise averaged losses, rather than the global averaged losses used in prior NC studies. We prove that when class-wise averaged losses attain their minima, standard NC is achieved on balanced or imbalanced datasets, for both cross-entropy (CE) and binary cross-entropy (BCE) losses. We further extend the analysis to an open-set-inspired scenario, where some classes contain no training samples yet are still assigned classifier vectors in the losses. We find that in this scenario, when class-wise averaged losses reach their minima, standard NC still emerges on the 
    
[^97]: 预见以交易：面向投资组合管理的预测增强策略优化

    Foresee to Trade: Forecast-Augmented Policy Optimization for Portfolio Management

    [https://arxiv.org/abs/2611.07750](https://arxiv.org/abs/2611.07750)

    提出 FRPO 框架，通过金融状态头（FSH）对资产级机会进行未来收益监督，并结合同路径组相对优化对账户级交易轨迹进行相对结果监督，以强化投资组合管理中的账户级交易决策。

    

    在线投资服务必须将资产机会转化为用户账户的投资收益，然而相同的市场机会可能需要不同的账户操作。我们研究来自未来资产机会的监督如何能够强化账户级别的交易决策，并提出预测增强相对策略优化（FRPO）。FRPO 通过一个联合学习的表示分别监督资产机会和账户决策：金融状态头（FSH）学习未来相对收益目标，以提供资产级别的机会监督；同路径组相对优化则在相同市场路径和初始账户条件下比较交易轨迹，以提供账户级别的相对结果监督。自适应门控利用预测方差和梯度对齐来协调辅助监督。FSH 仅在训练阶段使用；推理阶段仅使用编码器和策略进行交易。在 DJIA……

    arXiv:2611.07750v1 Announce Type: new  Abstract: Online investment services must translate asset opportunities into investment outcomes for users' accounts, yet the same market opportunity can warrant different account actions. We study how supervision from future asset opportunities can strengthen account-level trading decisions and propose Forecast-Augmented Relative Policy Optimization (FRPO). FRPO separately supervises asset opportunities and account decisions through a jointly learned representation: a Financial State Head (FSH) learns future relative-return targets to provide asset-level opportunity supervision, while Same-Path Group-Relative Optimization compares trading trajectories under the same market path and initial account to provide account-level relative-outcome supervision. An Adaptive Gate coordinates auxiliary supervision using predictive variance and gradient alignment. FSH is used only during training; inference uses the encoder and policy for trading. Across DJIA3
    
[^98]: ConvPE：卷积位置嵌入改进Transformer

    ConvPE: Convolutional Position Embedding Improves Transformers

    [https://arxiv.org/abs/2611.07749](https://arxiv.org/abs/2611.07749)

    ConvPE通过对隐藏状态低维投影进行轻量级因果卷积，向注意力分数注入窗口化、内容相关的偏置，作为即插即用模块在语言建模、长上下文扩展和视觉任务中同时提升了Transformer的性能与效率。

    

    ConvPE（卷积位置嵌入）在语言建模、长上下文扩展和视觉任务中同时提升了Transformer模型的性能与效率。它通过对隐藏状态的低维投影进行轻量级因果卷积，向注意力分数中注入一种窗口化的、依赖于内容的偏置。作为一个轻量级插件，ConvPE与RoPE、注意力掩码和KV拼接兼容，并且可以通过高维卷积支持高维位置嵌入。在一项复现实验中，ConvPE优于五个一维基线方法和八个二维基线方法。代码已在 https://github.com/wzhagi/convpe 公开发布。

    arXiv:2611.07749v1 Announce Type: new  Abstract: ConvPE (Convolutional Position Embedding) improves both the performance and efficiency of Transformer models in language modeling, long-context extension, and vision. It injects a windowed, content-dependent bias into attention scores through a lightweight causal convolution over a low-dimensional projection of the hidden states. As a lightweight plug-in, ConvPE is compatible with RoPE, attention masks, and KV concatenation. It also supports high-dimensional position embeddings via high-dimensional convolutions. In a reproduction, ConvPE outperforms five 1D baselines and eight 2D baselines. The code is publicly available at https://github.com/wzhagi/convpe.
    
[^99]: 基于波动率记忆的多变量时间序列排序：一种波动率遗传的时序卷积方法

    Ordering Multivariate Time Series by Volatility Memory: A Temporal Convolutional Approach to Volatility Heredity

    [https://arxiv.org/abs/2611.07746](https://arxiv.org/abs/2611.07746)

    本文提出QS-TCN方法，将准极大似然估计的波动率持续性统计量与矩阵束方法提取的时序模式嵌入时序卷积网络的结构化滤波器中，从而基于波动率遗传关系对多变量时间序列进行全局排序。

    

    波动率动态可以承载在条件均值中不明显的时序关系信息。由于每个变量自身的持续性可能会掩盖其对其他变量的依赖性，提取这些信息十分困难。本文提出了QS-TCN，这是一种对多变量时间序列进行排序的方法，它将波动率记忆的统计估计与时序卷积学习相结合。通过准极大似然获得的持续性估计与通过矩阵束方法提取的时序模式共同定义了网络内部的结构化滤波器。这些结构化滤波器与学习得到的滤波器一起表示目标变量的历史，并支持变量之间方向性证据的比较。这些比较结果被合并为一个全局的先后顺序，但该方法并不识别因果图。该方法在具有线性和非线性波动率交互的合成系统以及金融和fMRI基准数据集上进行了评估。

    arXiv:2611.07746v1 Announce Type: new  Abstract: Volatility dynamics can carry information about temporal relationships that is not apparent in the conditional mean. Extracting this information is difficult because the persistence of each variable can obscure its dependence on other variables. This paper presents QS-TCN, a method for ordering multivariate time series that combines statistical estimates of volatility memory with temporal convolutional learning. Persistence estimates obtained by quasi-maximum likelihood and temporal modes extracted by the Matrix Pencil method define structured filters within the network. Together with learned filters, they represent the history of the target and support comparisons of directional evidence between variables. These comparisons are combined into a global order of precedence, which does not identify a causal graph. The method is evaluated on synthetic systems with linear and nonlinear volatility interactions and on the Finance and fMRI bench
    
[^100]: 通过在不同尺度因子的归一化空间中分析损失函数来比较分类与特征学习

    Comparing Classification and Feature Learning by Analyzing Their Loss Functions in Normalized Spaces with Different Scale Factors

    [https://arxiv.org/abs/2611.07741](https://arxiv.org/abs/2611.07741)

    本文指出分类准确率反映的是样本级局部性质，而特征学习追求的是数据集级全局性质，并从第一性原理重新设计BCE损失，在归一化特征空间中分析其与CE损失的极小值，证明两者均可导致神经坍缩。

    

    分类和特征学习是两个密切相关的基本任务。在监督设置下，特征学习通常通过训练分类模型来实现，普遍的假设是更高的分类准确率意味着更好的特征质量。在本文中，我们首先指出分类准确率反映的是模型在数据集上的样本级局部性质，而特征学习旨在学习具有良好数据集级全局性质的特征，例如类内紧凑性和类间可分性。从特征学习的第一性原理出发，我们重新设计了二元交叉熵损失，它是分类中使用的交叉熵损失的对应版本。随后，在归一化特征空间中，我们分析了BCE和CE损失的极小值，并证明两者都可以导致神经坍缩现象，即来自不同类别的样本特征变得最大程度地紧凑且最大程度地分离。

    arXiv:2611.07741v1 Announce Type: new  Abstract: Classification and feature learning are two closely interrelated fundamental tasks. Feature learning in supervised settings is typically achieved by training classification models, under the common assumption that higher classification accuracy implies better feature quality. In this paper, we first point out that classification accuracy reflects the sample-wise local properties of a model on a dataset, whereas feature learning aims to learn features with good dataset-level global properties, such as intra-class compactness and inter-class separability. From the first principle of feature learning, we redesign the binary cross-entropy (BCE) loss, a counterpart of the cross-entropy (CE) loss used in classification. Then, in normalized feature space, we analyze the minima of the BCE and CE losses and show that both can lead to neural collapse, where sample features from different classes become maximally compact and maximally separated. Mo
    
[^101]: 通过循环机制扩展类型化决策模型的推理能力

    Extending Reasoning in Typed Decision Models via Looping

    [https://arxiv.org/abs/2611.07736](https://arxiv.org/abs/2611.07736)

    该论文提出 RELOAD 方法，通过对预训练模型的中间块进行阻尼循环并由轻量级控制器自适应选择循环次数，首次证明循环机制在与决策模型联合训练时能显著提升类型化决策模型的准确性与校准度，尤其在需要长推理链的问题上编码器最多可提升 23 分。

    

    类型化决策模型在单次前向传播中从固定的答案选项集合中进行选择，因此与生成式语言模型不同，它们无法通过生成更多 token 来在困难输入上投入更多的测试时计算量。对一层块进行循环可以在不增加参数的情况下在潜空间中提供这种额外计算，但先前的工作发现，将循环改装到按常规方式预训练的模型中并未带来收益。我们证明，当这种循环与决策模型一起训练时才能发挥作用。我们的方法 RELOAD（用于自适应决策的改装循环）对预训练的编码器或解码器进行微调，在其中间块上采用阻尼循环，并由一个轻量级控制器为每个输入自适应地选择循环次数。在从 0.4B 到 4B 参数的六个模型上，循环机制提升了准确性和校准度，提升主要体现在那些需要比训练中见过的任何问题都更长的推理链的问题上，编码器在此类问题上最多可获得 23 分的提升。无论是匹配的训练计算量还是……（原文摘要被截断）

    arXiv:2611.07736v1 Announce Type: cross  Abstract: Typed decision models choose among a fixed set of answer options in a single forward pass, so unlike generative language models they cannot spend more test-time compute on hard inputs by generating more tokens. Looping a block of layers offers this compute in latent space without adding parameters, but prior work found no gain from loops retrofitted into conventionally pretrained models. We show that such loops help when trained together with the decision model. Our method, RELOAD (retrofitted loops for adaptive decisions), fine-tunes a pretrained encoder or decoder with a damped loop over its middle block and a lightweight controller picks the number of passes adaptively for each input. Across six models from 0.4B to 4B parameters, looping improves both accuracy and calibration, mostly on problems that need longer chains of reasoning than any seen in training, where encoders gain up to 23 points. Neither matched training compute nor m
    
[^102]: 静态恢复并非动态稳定：面向蛋白质基序支架化的动力学感知基准测试

    Static Recovery Is Not Dynamic Stability: Dynamics-Aware Benchmarking of Protein Motif Scaffolding

    [https://arxiv.org/abs/2611.07723](https://arxiv.org/abs/2611.07723)

    该论文提出了动力学感知的蛋白质基序支架化新基准BILBY-PRO和BILBY-MD，通过系统性扩大基序覆盖范围并引入动态稳定性评估标准，揭示了不同生成模型在目标覆盖与独特解产出之间的显著差异。

    

    蛋白质基序支架化基准测试旨在评估生成模型能否在功能重要残基周围构建支架结构，其结论取决于测试哪些基序以及用何种标准定义成功。现有基准通常采用小型人工筛选的基序集合和静态的复折叠评估标准，然而蛋白质功能往往依赖于构象系综与构象转变。本文同时针对基准范围和成功标准两方面进行了改进。我们提出了BILBY-PRO，一个从与PROSITE关联的实验结构中系统推导出来的结构保守功能性基序基准，包含来自174个PROSITE基序模式条目的220个测试案例。对六个近期支架生成模型的评估揭示了它们在目标覆盖率和独特解产出方面的不同模式：RFDiffusion和Proteina能够解决更多目标，而ESM3在其所解决的目标上能产出更多独特的成功支架。我们进一步引入BILBY-MD，一个基于……（原文摘要在此处截断）

    arXiv:2611.07723v1 Announce Type: new  Abstract: Protein motif scaffolding benchmarks assess whether generative models can build scaffolds around functionally important residues. Their conclusions depend on which motifs are tested and which criteria define success. Existing benchmarks typically use small hand-curated motif sets and static refolding criteria, but protein function often depends on conformational ensembles and transitions. Here, we address both benchmark scope and success criteria. We introduce BILBY-PRO, a systematically derived benchmark of structurally conserved functional motifs from PROSITE-linked experimental structures, comprising 220 cases from 174 PROSITE motif-pattern entries. Evaluating six recent scaffold-generation models reveals distinct patterns in target coverage and unique-solution yield: RFDiffusion and Proteina solve more targets, whereas ESM3 produces more unique successful scaffolds on the targets it solves. We further introduce BILBY-MD, a reference-
    
[^103]: 强化学生已学到的知识：教师引导的在策略蒸馏

    Reinforcing What the Student Has Learned: Teacher-Guided On-Policy Distillation

    [https://arxiv.org/abs/2611.07722](https://arxiv.org/abs/2611.07722)

    提出SD-OPD方法，以学生自身预训练基础模型的差异作为更新权重，避免在策略蒸馏被师生表达风格差异主导，从而强化学生已习得的推理能力。

    

    在策略蒸馏（OPD）通过教师对学生生成回答的密集反馈来提升语言模型的推理能力。我们首先观察到这种反馈中存在幅度失衡：在数学推理回答上，风格相关位置的平均绝对师生对数概率差是数学相关位置的3.5倍。因此，若以这些差异来缩放OPD更新，将使大部分更新强度被用于模仿教师的表达方式。我们转而以学生自身的预训练基础模型作为参照：学生与基础模型之间的差异记录了后训练阶段习得的词元偏好，从而提供了一种不依赖于学生与教师之间距离的监督信号。基于该参照，我们提出学生增量在策略蒸馏（SD-OPD），该方法以中心化的学生-基础模型差异作为更新系数，且仅当系数符号与中心化的师生（原文在此处截断）

    arXiv:2611.07722v1 Announce Type: new  Abstract: On-policy distillation (OPD) improves language-model reasoning through dense teacher feedback on student-generated responses.We first observe an amplitude disparity in this feedback: on math reasoning responses, style-related positions exhibit 3.5 times the mean absolute teacher--student log-probability difference of mathematical positions.Scaling OPD updates by these discrepancies thus directs much of the update strength toward matching the teacher's expression. We instead take the student's own pretrained base as a reference: the student--base difference records the token preferences acquired during post-training and provides supervision that does not depend on the student's distance from the teacher. Building on this reference, we propose Student-Delta On-Policy Distillation (SD-OPD), which uses centered student--base differences as update coefficients, keeps each coefficient only when its sign agrees with centered teacher--student fe
    
[^104]: 在隔离假设下评估语言模型特征用于零样本时间序列异常检测

    Evaluating Language Model Features for Zero-Shot Time-Series Anomaly Detection Under Isolation Assumption

    [https://arxiv.org/abs/2611.07720](https://arxiv.org/abs/2611.07720)

    该论文提出将大语言模型纯粹作为特征提取器用于零样本多变量时间序列异常检测，通过直接分析其诱导的特征空间，规避了预测或重建等辅助目标在模型容量增大时会重建异常而变得不可靠的问题。

    

    时间序列异常检测（TSAD）是一项具有根本性挑战的任务：它是无监督的，需要高局部精度能力，且受限于有限的标注数据。近期研究探索了通过跨模态迁移将大语言模型（LLM）适配到时间序列领域。然而，现有方法并不适合多变量时间序列异常检测（MTSAD），因为分词处理往往会掩盖细粒度的时间结构和变量结构，且大多数方法都在优化诸如预测或重建等辅助目标。我们认为，随着模型容量的增加，这些替代目标会变得不可靠——高表达能力的模型可能会重建或预测出异常，从而破坏这两种策略的基础。相反，我们提出将LLM纯粹用作特征提取器，并直接分析其诱导的特征空间。在本工作中，我们研究了LLM在零样本设置下为MTSAD提供有用表示的能力。

    arXiv:2611.07720v1 Announce Type: cross  Abstract: Time-Series Anomaly Detection (TSAD) is a fundamentally challenging task: it is unsupervised, requires high local precision capabilities, and suffers from limited labelled data. Recent work has explored adapting Large Language Models (LLMs) to time-series via cross-modal transfer. However, existing approaches are poorly suited to Multivariate TSAD (MTSAD), as tokenisation often obscures fine-grained temporal and variables structures, and most methods optimise auxiliary objectives such as forecasting or reconstruction. We posit that these surrogate objectives become unreliable as model capacity increases. Highly expressive models risk reconstructing or forecasting anomalies, undermining the foundations of both strategies. Instead, we propose to use LLMs purely as feature extractors and directly analyse their induced feature spaces. In this work, investigate the ability of LLMs to provide useful representations for MTSADs in a zero-shot 
    
[^105]: 面向多模态自监督学习的多模态信息交互瓶颈

    Multimodal Information Interactions Bottleneck for Multimodal Self-Supervised Learning

    [https://arxiv.org/abs/2611.07718](https://arxiv.org/abs/2611.07718)

    提出基于部分信息分解的多模态信息交互瓶颈框架MIIB，通过双路径架构显式编码多模态交互中的冗余、独特与协同信息，无需显式互信息估计即可学习最小充分的多模态表示。

    

    多模态自监督学习面临来自图像、文本、音频和表格数据等多模态数据源的异质性、信息量不均等以及复杂信息交互的挑战。虽然对比方法能够有效对齐共享语义，但它们通常依赖于较强的多视图冗余假设，忽略了模态特有的信息和协同信息，从而导致次优的表示。我们提出了多模态信息交互瓶颈（MIIB），这是一个基于部分信息分解的有原则的框架，能够显式编码多模态交互中的冗余信息、独特信息和协同信息。MIIB采用双路径模型架构进行模态特定编码和多模态融合，并使用可代理求解的、易于处理的基于熵的目标函数进行优化，在无需显式互信息估计的情况下实现最小充分表示。

    arXiv:2611.07718v1 Announce Type: cross  Abstract: Multimodal self-supervised learning is challenged by heterogeneity, unequal informativeness, and complex information interactions across multimodal data sources such as images, text, audio, and tabular data. While contrastive methods effectively align shared semantics, they typically rely on strong multi-view redundancy assumptions, overlooking modality-specific and synergistic information, thus leading to suboptimal representations. We propose the Multimodal Information Interactions Bottleneck (MIIB), a principled framework grounded in Partial Information Decomposition that explicitly encodes redundant, unique, and synergistic information within multimodal interactions. MIIB employs a dual-path model architecture for modality-specific encoding and multimodal fusion, optimized using surrogate tractable entropy-based objectives that enforce minimal sufficient representations without requiring explicit mutual information estimation. By l
    
[^106]: InfMasking：协同多模态学习的统一变分框架

    InfMasking: A Unified Variational Framework for Synergistic Multimodal Learning

    [https://arxiv.org/abs/2611.07702](https://arxiv.org/abs/2611.07702)

    提出了 InfMasking，一个基于无限随机掩码的统一变分多模态学习框架，通过在融合前随机掩码各模态特征，并要求所有被掩码的融合表示与参考目标（无监督时为完整融合视图，有监督时为标签类别代理）保持一致，从而迫使模型充分利用模态间的协同信息，而非依赖主导模态或共享信息。

    

    多模态模型被期望能够利用协同信息，即只有在同时观察多个模态时才能获得的信息。然而在实践中，融合模块可以通过依赖某个主导模态或模态间共享的信息来拟合训练目标，模态之间的协同性往往未被利用。我们提出了 InfMasking，一个基于无限掩码的统一多模态学习框架。在融合之前，InfMasking 随机掩码每个模态的特征；每个采样的掩码会产生一种不同的跨模态证据部分组合，以及对应模态交互的融合表示。所有被掩码的融合表示都需要与参考目标保持一致：当训练是无监督时，参考目标为未掩码的融合视图；当标签可用时，参考目标为任务标签的类别代理。在多样的部分组合上维持这种一致性促使模型（原文在此处截断）。

    arXiv:2611.07702v1 Announce Type: cross  Abstract: Multimodal models are expected to exploit synergistic information, that is, information available only when multiple modalities are observed together. In practice, however, a fusion module can fit the training objective by relying on a dominant modality or on information that modalities share, and the synergy between modalities is often left unused. We propose InfMasking, a unified multimodal learning framework based on Infinite Masking. Before fusion, InfMasking randomly masks the features of each modality; each sampled mask produces a different partial combination of cross-modal evidence and a fused representation of the corresponding modality interactions. All masked fused representations are required to remain consistent with a reference target: the unmasked fused view when training is unsupervised, and the class proxy of the task label when labels are available. Maintaining this consistency across diverse partial combinations disc
    
[^107]: 直积空间中的等变深度学习用于电子结构哈密顿量预测

    Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction

    [https://arxiv.org/abs/2611.07695](https://arxiv.org/abs/2611.07695)

    提出DP-Learner框架，在直积空间中直接对类哈密顿量算符块进行等变消息传递，同时解决了传统直和表示与哈密顿量之间的表示不匹配问题及计算效率瓶颈。

    

    大多数现有的用于哈密顿量预测的等变架构是在不可约球张量的直和空间中传播特征，而目标哈密顿量本质上是局部轨道和自旋基的直积表示中的一个算符块。这种表示不匹配割裂了连续的哈密顿量块，模糊了对轨道耦合的直接操作，并将特征组织成不规则的角动量分量，这些分量需要大量变换，且难以与高效的硬件执行和内存管理相适配。为了同时解决表示不匹配和计算瓶颈问题，我们提出了DP-Learner，一个在直积空间中直接对类哈密顿量算符块进行消息传递的等变学习框架。在理论上，我们提出了基于李生成子的方向嵌入以及不变加权的左右乘积与收缩运算……

    arXiv:2611.07695v1 Announce Type: cross  Abstract: Most existing equivariant architectures for Hamiltonian prediction propagate features in direct-sum spaces of irreducible spherical tensors, whereas the target Hamiltonian is naturally an operator block in a direct-product representation over local orbital and spin bases. This representation mismatch fragments contiguous Hamiltonian blocks, obscures direct manipulation of orbital couplings, and organizes features into irregular angular-momentum components that require numerous transformations and are poorly aligned with efficient hardware execution and memory management. To address both the representation mismatch and computational bottleneck, we propose DP-Learner, an equivariant learning framework that performs message passing directly on Hamiltonian-like operator blocks in the direct-product space. Theoretically, we propose Lie-generator-based directional embedding and invariantly weighted left--right products and contractions that 
    
[^108]: 基于Vine Copula的非参数互信息估计

    Non-Parametric Mutual Information Estimation with Vine Copulas

    [https://arxiv.org/abs/2611.07693](https://arxiv.org/abs/2611.07693)

    本文提出VINE，一种基于简化D-vine copula的非参数互信息估计器，通过将密度比简化为成对copula的乘积实现非参数估计，在低维到中等维度数据集上以更低的计算成本达到与最先进神经估计器相当的精度。

    

    我们提出了VINE：一种基于简化D-vine copula的非参数互信息（MI）估计器。使用简化的D-vine来近似联合密度，使密度比可以简化为成对copula的乘积，从而可以对其进行非参数估计。这种结构上的简化产生了VINE，一个灵活且计算高效的互信息估计器。我们在一系列数据集上将VINE与最先进的神经互信息估计器进行比较，包括简化D-vine假设成立和被违反的设置，结果表明在低维到中等维度的数据集上，VINE以更低的计算成本实现了具有竞争力的估计精度。

    arXiv:2611.07693v1 Announce Type: new  Abstract: We introduce VINE: a non-parametric mutual information (MI) estimator based on simplified D-vine copulas. Approximating the joint density with a simplified D-vine allows the density ratio to simplify to a product of pair copulas, which we can estimate non-parametrically. This structural simplification yields VINE, a flexible and computationally efficient MI estimator. We compare VINE against SotA neural MI estimators on a range of datasets, both under settings where the simplified D-vine assumption holds and where it is violated, demonstrating competitive accuracy at a lower computational cost for low- to medium-dimensional datasets.
    
[^109]: 针对非光滑非凸约束优化的随机化与基于预言机的可行方向方法

    Randomized and Oracle-Based Feasible Directions Methods for Nonsmooth Nonconvex Constrained Optimization

    [https://arxiv.org/abs/2611.07691](https://arxiv.org/abs/2611.07691)

    该论文引入η-可行方向（与可行方向锥边界保持一致分离的方向），克服了非光滑非凸约束优化中可行方向锥不封闭导致的几何难题，实现了无需目标导数、仅靠精确线搜索的有限方向搜索算法框架。

    

    在一般凸集上最小化非光滑非凸目标函数，且不使用目标函数的导数、仅通过约束集的结构信息访问约束，这带来了一个根本性的几何难题：在边界点处，可行方向锥可能不是闭集，从而无法构造能够捕捉所有可行下降方向的有限搜索方向集合。我们针对在闭凸集上表现为连续可微函数与凸函数之差这一结构的问题，开发了一个可行方向框架，并在不知道目标函数分解形式的情况下，通过沿可行线段的精确线搜索来访问目标函数。为克服这一几何障碍，我们引入了η-可行方向，它与可行方向锥的边界保持一致分离，从而即使在存在非线性约束的情况下也能实现有限方向搜索。我们提出了两种算法...

    arXiv:2611.07691v1 Announce Type: cross  Abstract: Minimizing nonsmooth nonconvex objectives over general convex sets using no derivatives of the objective, together with structural access to the constraint set, poses a fundamental geometric difficulty: at boundary points, the feasible direction cone may fail to be closed, preventing the construction of finite sets of search directions that capture all feasible descent directions. We develop a feasible-directions framework for problems structured as the difference between a continuously differentiable function and a convex function over a closed convex set, and access the objective without knowledge of its decomposition, through exact line searches along feasible segments. To address the geometric obstruction, we introduce $\eta$-feasible directions, which remain uniformly separated from the boundary of the feasible direction cone, enabling finite-direction search even in the presence of nonlinear constraints. We present two algorithmi
    
[^110]: RVI Q-学习的 $O(1/n)$ 收敛速率

    $O(1/n)$ Convergence Rate for RVI $Q$-Learning

    [https://arxiv.org/abs/2611.07676](https://arxiv.org/abs/2611.07676)

    该论文通过分析随机 Krasnosel'skiĭ–Mann 迭代，为有限弱通信 MDP 上的同步 RVI Q-学习建立了有限时间收敛保证，并在适当多项式步长下达到 $O(1/n)$ 的收敛速率。

    

    我们为有限弱通信马尔可夫决策过程（MDP）上的同步相对值迭代（RVI）Q-学习建立了有限时间收敛保证。对于在有限维赋范空间上由非扩张且线性正则的映射驱动的随机 Krasnosel'skiĭ–Mann 迭代，我们在鞅差噪声（其条件二阶矩至多以残差平方的线性速度增长）下建立了末次迭代的均方残差界。适当选取 $n^{-a}$ 阶的多项式步长，在 $0<\cdots$（摘要原文在此处截断）条件下可获得 $O(n^{-a})$ 的收敛速率。

    arXiv:2611.07676v1 Announce Type: new  Abstract: We establish finite-time convergence guarantee for synchronous relative value iteration (RVI) $Q$-learning for finite weakly communicating Markov decision processes (MDPs). For stochastic Krasnosel'ski\u{\i}--Mann iterations driven by a non-expansive, linearly regular map on a finite-dimensional normed space, we establish last-iterate mean-squared residual bounds under martingale-difference noise whose conditional second moment grows at most linearly with the squared residual. Suitable polynomial stepsizes of order $n^{-a}$ yield an $O(n^{-a})$ rate for $0
    
[^111]: 连续时间随机过程中的工具变量

    Instrumental Variables in Continuous-Time Stochastic Processes

    [https://arxiv.org/abs/2611.07674](https://arxiv.org/abs/2611.07674)

    该论文首次将工具变量方法扩展到由随机微分方程参数化的连续时间随机过程中，通过在漂移项的无穷小层面编码结构稀疏性与因果效应，并利用路径值工具变量的矩约束条件，实现了连续时间动力学系统中因果效应的识别。

    

    因果问题常常出现在随时间连续演化的系统中。虽然工具变量（IV）方法为在静态或离散时间设置中识别因果效应提供了一种有原则的方法，即使存在未观测的混杂因素也能适用，但这类方法尚未扩展到连续时间动力学。在这项工作中，我们为以随机微分方程（SDE）参数化的平稳连续时间随机过程开发了一个工具变量框架，其中工具变量本身以随机过程的形式出现。我们直接在漂移项的无穷小层面编码结构稀疏性和因果效应，并将这些假设与同一过程离散化版本上的工具变量假设进行对比。基于这些结构假设，我们利用路径值工具变量推导出矩约束条件。我们展示了这些约束条件在Ornstein-Uhlenbeck过程下如何简化，并给出了一个识别条件……

    arXiv:2611.07674v1 Announce Type: new  Abstract: Causal questions often arise in systems that evolve in continuous time. While instrumental variable (IV) methods provide a principled approach to identify causal effects in static or discrete-time settings, even in the presence of unobserved confounding, they do not yet extend to continuous-time dynamics. In this work, we develop an IV framework for stationary continuous-time stochastic processes parameterized by stochastic differential equations (SDE), where the instrument itself takes the form of a stochastic process. We encode the structural sparsity and causal effects directly at the infinitesimal level of the drift, and contrast these assumptions with the IV assumptions on the discretized version of the same process. Based on these structural assumptions, we derive moment restrictions using path-valued instruments. We show how these restrictions simplify for Ornstein-Uhlenbeck processes and present an identification condition togeth
    
[^112]: 校准的目标噪声从梯度中恢复曲率信息

    Calibrated Target Noise Recovers Curvature from the Gradients

    [https://arxiv.org/abs/2611.07671](https://arxiv.org/abs/2611.07671)

    该论文提出一种向目标注入方差经局部输出曲率校准的零均值噪声的简单方法，使得含噪批量梯度的协方差能够仅从梯度中恢复出广义高斯-牛顿（GGN）等曲率矩阵，无需逐样本梯度或二阶导数。

    

    仅从梯度估计曲率信息是机器学习中的一个基本问题。在只有聚合的小批量梯度可用、但仍需要曲率信息的场景下（例如优化中的预条件、采样和提前停止等应用），这一挑战尤为突出。也许令人惊讶的是，我们证明一种简单的目标扰动方案就足以仅从梯度中恢复此类曲率信息。我们的方法向目标中注入零均值噪声，其方差根据局部输出曲率进行校准。由此产生的含噪批量梯度（noisy batch gradients）的协方差即可恢复实用的曲率矩阵，而无需逐样本梯度、二阶导数或其他网络内部量。对于一大类非线性模型，我们的方案可以恢复广义高斯-牛顿（GGN）矩阵；对于前馈ReLU网络，它还能额外恢复（摘要在此处截断）

    arXiv:2611.07671v1 Announce Type: new  Abstract: Estimating curvature information from gradients alone is a fundamental problem in machine learning. The challenge becomes particularly acute in settings where only aggregate mini-batch gradients are available, yet curvature information is still required for applications such as preconditioning in optimisation, sampling, and early stopping. Perhaps surprisingly, we show that a simple target-perturbation scheme suffices to recover such curvature information from gradients alone. Our method injects zero-mean noise into the targets, with variance calibrated to the local output curvature. The covariance of the resulting noisy batch gradients then recovers practical curvature matrices, without requiring per-sample gradients, second-order derivatives, or other internal network quantities. For a broad class of non-linear models, our scheme recovers the Generalised Gauss--Newton (GGN) matrix; for feed-forward ReLU networks, it additionally recove
    
[^113]: 一种用于事后缓解大语言模型记忆化问题的信息瓶颈方法

    An Information Bottleneck for Post-hoc Memorization Mitigation in LLMs

    [https://arxiv.org/abs/2611.07669](https://arxiv.org/abs/2611.07669)

    提出一种即插即用的非参数变分信息瓶颈（NVIB），插入冻结的大语言模型骨干与语言建模头之间，通过压缩隐藏表示在不修改或不重训模型的情况下过滤记忆化内容、保留通用语言知识。

    

    最近的研究表明，大型语言模型（LLMs）能够记忆并复现受版权保护的训练数据，引发了严重的隐私和知识产权问题。在本工作中，我们研究了能否通过信息瓶颈在不修改或重新训练底层模型的情况下缓解记忆化问题。我们提出了一个模块化的即插即用框架，在冻结的LLM骨干网络与其语言建模头之间插入一个非参数变分信息瓶颈（NVIB）。通过在词元预测之前压缩最终的隐藏表示，所提出的瓶颈旨在保留通用的语言知识，同时过滤掉高度特定的已记忆序列。我们在涉及书籍段落的记忆化提取任务以及标准语言建模基准上评估了我们的方法，以衡量记忆化减少与下游效用之间的权衡。我们的实验表明，NVIB能够...

    arXiv:2611.07669v1 Announce Type: new  Abstract: Recent studies have shown that large language models (LLMs) can memorize and reproduce copyrighted training data, raising serious privacy and intellectual property concerns. In this work, we investigate whether memorization can be mitigated without modifying or retraining the underlying model through an information bottleneck. We propose a modular plug-and-play framework that inserts a nonparametric variational information bottleneck (NVIB) between a frozen LLM backbone and its language modeling head. By compressing the final hidden representations before token prediction, the proposed bottleneck aims to preserve general linguistic knowledge while filtering out highly specific memorized sequences. We evaluate our approach on memorization extraction tasks involving book passages and on standard language modeling benchmarks to measure the trade-off between memorization reduction and downstream utility. Our experiments show that the NVIB la
    
[^114]: 基于SINDy从数据中提取单模瑞利-泰勒不稳定性动力学

    Extracting Single-Mode Rayleigh--Taylor Instability dynamics from data via SINDy

    [https://arxiv.org/abs/2611.07666](https://arxiv.org/abs/2611.07666)

    该论文提出了一个基于SINDy积分形式的物理约束数据驱动框架，通过引入无量纲初始振幅作为显式参数，直接从数据中识别出能够刻画瑞利-泰勒不稳定性中气泡和尖钉演化（包括过冲与再加速阶段）的低维动力学方程，克服了传统势流理论和浮力-阻力模型的局限。

    

    瑞利-泰勒不稳定性（RTI）在惯性约束聚变和超新星爆发等领域具有基础性重要意义，其演化过程通常由势流理论和浮力-阻力模型来描述。然而，这两类模型都难以准确捕捉气泡和尖钉速度的过冲行为以及随后的再加速阶段。为解决这些局限性，我们开发了一个基于稀疏非线性动力学识别（SINDy）积分形式的物理约束数据驱动框架，直接从数据中提取气泡和尖钉演化的低维动力学方程。通过引入无量纲初始振幅 $A_0 = k\eta_0$ 作为显式参数，我们基于跨越多个阿特伍德数和初始振幅的数据，识别出了气泡和尖钉各自不同的动力学方程。所得到的低维动力学方程……

    arXiv:2611.07666v1 Announce Type: cross  Abstract: Rayleigh-Taylor instability (RTI) is of fundamental importance in fields such as inertial confinement fusion and supernova explosions, and its evolution is commonly described by potential--flow theory and buoyancy--drag models. However, both types of models struggle to accurately capture the overshoot behavior of bubble and spike velocities, as well as the subsequent reacceleration stage. To address these limitations, we develop a physics-constrained, data-driven framework based on the integral formulation of Sparse Identification of Nonlinear Dynamics (SINDy), which extracts low-dimensional dynamical equations for bubble and spike evolution directly from data. By introducing the nondimensional initial amplitude $A_0 = k\eta_0$ as an explicit parameter, we identify distinct dynamical equations for the bubble and the spike based on data spanning multiple Atwood numbers and initial amplitudes. The resulting low-dimensional dynamical equa
    
[^115]: 基于性能表示的HPO与CASH简单元学习方法

    Simple Meta-Learning for HPO and CASH Using Performance Representations

    [https://arxiv.org/abs/2611.07659](https://arxiv.org/abs/2611.07659)

    提出一种简单有效的元学习表示方法，直接以配置在历史任务上的性能表现作为其表示，从而改进贝叶斯优化在HPO和CASH中的代理建模效果。

    

    贝叶斯优化在超参数优化（HPO）和组合算法选择与超参数优化（CASH）中通常依赖显式的超参数编码作为代理模型的输入。然而，在高维搜索空间中，这种表示方式使得标准代理模型难以学习到有用的性能模型。我们转而研究一种替代性的元学习表示方法，该方法直接将每个配置描述为其在先前任务上的历史性能表现。这种表示可以直接与标准代理模型配合使用。我们的研究范围是AutoML的迭代选择组件，即从固定的候选集合中决定下一个要评估的配置，而非包含运行时间估计、成本感知预算和集成等功能的完整框架。我们进一步将这种表示应用于最大边际相关性（MMR），这是一种来自推荐系统的基于多样性的重排序准则。

    arXiv:2611.07659v1 Announce Type: new  Abstract: Bayesian optimization for Hyperparameter Optimization (HPO) and Combined Algorithm Selection and Hyperparameter Optimization (CASH) often relies on explicit hyperparameter encodings as surrogate inputs. In high-dimensional search spaces, however, such representations can make it difficult for standard surrogates to learn useful performance models. We instead study an alternative meta-learning representation that describes each configuration directly as its historical performance across prior tasks. This representation can be paired directly with standard surrogate models. Our scope is the iterative selection component of AutoML, deciding which configuration to evaluate next from a fixed candidate set, rather than a complete framework with features such as runtime estimation, cost-aware budgeting and ensembling. We further apply this representation to maximal marginal relevance (MMR), a diversity-based re-ranking criterion from recommende
    
[^116]: 专家剪枝混合专家语言模型中英语校准对立陶宛语能力的侵蚀

    English Calibration Erodes Lithuanian in Expert-Pruned Mixture-of-Experts Language Models

    [https://arxiv.org/abs/2611.07654](https://arxiv.org/abs/2611.07654)

    该研究发现英语校准在专家剪枝的MoE模型中无法迁移到立陶宛语——25%剪枝率下英语理解仅下降1.7分而立陶宛语下降21.4分，而立陶宛语校准却能同时保护两种语言的性能。

    

    专家剪枝通过移除被校准集判定为不重要的专家来压缩混合专家语言模型。针对高资源语言的研究大多发现，英语校准可以很好地迁移到其他语言。我们在立陶宛语上检验了这一结论：我们使用REAP方法对四个MoE模型进行剪枝，分别基于英语、立陶宛语或混合校准集移除25%或50%的专家，并在阅读理解、翻译和语言建模任务上对两种语言进行评估。结果表明，英语校准无法迁移到立陶宛语：在25%剪枝率下，英语理解能力中位数下降1.7分，而立陶宛语下降21.4分。这种迁移是不对称的：立陶宛语校准消除了大部分立陶宛语性能损失，且从未显著损害英语理解能力（平均比英语校准高出2.5分）；仅有英语语言建模性能略有下降。路由分析揭示了其中原因：在模型最后几层中，立陶宛语将55……

    arXiv:2611.07654v1 Announce Type: new  Abstract: Expert pruning compresses Mixture-of-Experts (MoE) language models by removing the experts that a calibration set marks as unimportant. Studies of high-resource languages largely find that English calibration transfers well to other languages. We test this for Lithuanian: we pruned four MoE models with REAP, removing 25% or 50% of the experts with English, Lithuanian or mixed calibration, and evaluated both languages on reading comprehension, translation and language modelling. English calibration does not transfer to Lithuanian: at 25% pruning, English comprehension fell by a median of 1.7 points but Lithuanian by 21.4. The transfer is asymmetric. Lithuanian calibration removed most of the Lithuanian loss and never significantly hurt English comprehension, which was on average 2.5 points better than under English calibration; only English language modeling got slightly worse. Routing suggests why: in the last layers, Lithuanian sends 55
    
[^117]: WaSP：面向无求解器算子学习的Wasserstein数据合成流水线

    WaSP: Wasserstein Data Synthesis Pipeline for Solver-Free Operator Learning

    [https://arxiv.org/abs/2611.07646](https://arxiv.org/abs/2611.07646)

    该论文提出WaSP，一种基于Wasserstein度量的数据合成流水线，通过在人造数据对上训练并逐步缓解训练分布与目标分布之间的偏移，实现了无需求解器的高效算子学习。

    

    神经算子为求解偏微分方程（PDE）提供了快速的代理方法，但训练它们通常需要代价高昂的“输入-解”数据对。两种无需求解器的替代方案包括物理信息学习和基于人造数据的训练。物理信息学习通过最小化PDE残差来逼近解，虽然残差范数必须足够强才能控制解的误差，但更强的范数会增加小批量的方差并阻碍优化。人造数据方法则是通过将PDE算子作用于人工构造的解来构建用于学习的“输入-解”数据对，然而由此得到的训练分布可能与目标分布相互奇异，导致在目标分布上的精度很差。我们提出了Wasserstein数据合成流水线（WaSP），它在人造的“输入-解”数据对上进行训练，同时逐步缓解分布偏移。WaSP通过将学习到的算子应用于一条路径上的输入来合成新的数据对……

    arXiv:2611.07646v1 Announce Type: cross  Abstract: Neural operators offer fast surrogates for solving PDEs, but training them typically requires expensive input-solution pairs. Two solver-free alternatives include physics-informed learning and training on manufactured data. Physics-informed learning approximates solutions by minimizing PDE residuals. While the residual norm must be strong enough to control solution error, stronger norms increase mini-batch variance and hinder optimization. Manufactured data constructs input-solution pairs for learning by applying the PDE operator to artificial solutions. However, the resulting training distribution can be mutually singular with the target distribution, yielding poor accuracy on the target distribution. We introduce the Wasserstein Data Synthesis Pipeline (WaSP), which trains on manufactured input-solution pairs while gradually mitigating the distribution shift. WaSP synthesizes new pairs by applying the learned operator to a path of in
    
[^118]: 衡量基于验证器的强化学习相对于监督微调的边际价值

    Measuring the Marginal Value of Verifier-Based RL Beyond Supervised Fine-Tuning

    [https://arxiv.org/abs/2611.07639](https://arxiv.org/abs/2611.07639)

    该论文构建了一个确定性的GPU/LLM训练故障诊断基准，发现仅用1,200个黄金示范的监督微调即可近乎解决该基准，而从基础模型出发的GRPO强化学习几乎无增益，从而质疑了基于验证器的强化学习相对于监督微调的边际价值。

    

    arXiv:2611.07639v1 公告类型：cross 摘要：可验证奖励的强化学习（RLVR）通常被评估为监督微调（SFT）的替代方案，或作为其后续阶段，其中隐含的假设是RL阶段本身能贡献额外的增益。我们在一个受控环境中直接测量了这一贡献，在该环境中两个阶段都可以完整运行并进行精确比较。我们构建了gpu-train-debug，这是一个包含1,877个合成GPU和LLM训练失败诊断任务的确定性基准，涵盖30种任务原型，由带部分得分的程序化验证器进行评分，并经过校准使中等层级模型的得分落在0.60至0.75区间内。一项试点实验带来了意外发现：1,200个黄金示范几乎解决了该基准（SFT在困难分割上得分为0.785），而从基础模型开始的15步GRPO训练几乎没有改变结果（0.314，对比基础模型的0.311）。因此，我们在完整实验运行之前以书面形式修改了研究方案，提出了更尖锐的问题：基于验证器的RL在此基础之上究竟有何额外贡献。

    arXiv:2611.07639v1 Announce Type: cross  Abstract: Reinforcement learning from verifiable rewards (RLVR) is usually evaluated as an alternative to supervised fine-tuning (SFT), or as a stage that follows it, with the implicit assumption that the RL stage contributes gains of its own. We measure that contribution directly, in a controlled setting where both stages can be run to completion and compared exactly. We built gpu-train-debug, a deterministic benchmark of 1,877 synthetic GPU and LLM training failure diagnosis tasks across 30 archetypes, graded by a programmatic verifier with partial credit, and calibrated so that a mid-tier model scores in a 0.60 to 0.75 band. A pilot produced a surprise: 1,200 gold demonstrations nearly solved the benchmark (SFT 0.785 on the hard split), while 15 steps of GRPO from the base model did not move it (0.314 vs 0.311 base). We therefore amended the study, in writing and before the full run, to ask the sharper question: what does verifier-based RL ad
    
[^119]: 基于有序专家的在线学习（通过两点反馈）

    Online Learning with Ordered Experts (via Two-point Feedback)

    [https://arxiv.org/abs/2611.07629](https://arxiv.org/abs/2611.07629)

    该论文研究了完全有序动作空间中基于两点反馈的在线学习问题，在分离单调损失下实现了仅比全信息界差一个对数因子的 O(log(|F|T)√(T log|F|)) 遗憾，在有界变差损失下实现了 Õ(√T+|F|) 的遗憾。

    

    我们研究了完全有序动作空间中带有专家推荐的对抗性在线学习问题，其中专家的推荐可能发生变化并被任意重新排序。我们的重点是两点反馈：在每一轮中，学习者选择两个当前推荐进行操作，同时观察两者，并支付它们的平均损失。我们研究了两个问题。在分离单调问题中，损失是一个隐藏的非递减分量与一个隐藏的非递增分量之和，查询会分别揭示这两个分量。在有界变差（BV）问题中，损失是有界的，且沿该顺序具有有界变差，查询仅揭示损失值。在分离单调问题中，两次付费观测相对于|F|个专家中的最优者实现了 O(log(|F|T)√(T log|F|)) 的遗憾，与经典的全信息界仅相差一个对数因子。对于BV问题，我们获得了 Õ(√T + |F|) 的遗憾。我们给出了一个应用……

    arXiv:2611.07629v1 Announce Type: new  Abstract: We study adversarial online learning with expert recommendations in a totally ordered action space, where the experts' recommendations may change and be reordered arbitrarily. Our focus is two-point feedback: on each round the learner plays two current recommendations, observes both, and pays their average. We study two problems. In the separated monotone problem, the loss is the sum of a hidden nondecreasing component and a hidden nonincreasing component, and a query reveals the two components separately. In the BV problem, the losses are bounded and have bounded variation (BV) along the order, and a query reveals only the loss.   In the separated monotone problem, two paid observations achieve \[ O\bigl(\log(|\cF|T)\sqrt{T\log|\cF|}\bigr) \] regret against the best of $|\cF|$ experts, within a logarithmic factor of the classical full-information bound. For the BV problem, we obtain $\wtO(\sqrt T+|\cF|)$ regret. We give an application i
    
[^120]: 面向离线安全强化学习的单调可行性学习

    Monotone Viability Learning for Offline Safe Reinforcement Learning

    [https://arxiv.org/abs/2611.07616](https://arxiv.org/abs/2611.07616)

    提出单调可行性学习（MoViL），通过单调有限时程集合递归从离线数据中学习可行集，在保持安全语义、收敛至无限时程可行性核的同时避免折扣引起的数值衰减，实现部署时可靠的动作安全认证。

    

    离线安全强化学习需要从固定数据中获得可靠的长时程安全估计。在这种设置下学习Hamilton-Jacobi（HJ）可行性，要求算法具备语义保持性、收敛性和数值稳定性。为应对这些挑战，我们提出了单调可行性学习（MoViL），通过单调的有限时程集合递归从离线转移数据中学习可行集。该递归保持了有限时程的安全语义，并收敛到无限时程的可行性核。二值监督在各时程上保持固定的目标尺度，避免了折扣因子引起的数值衰减。我们进一步建立了恢复行为支撑可行子集的充分条件。学到的可行性估计被用于在部署时对动作进行安全认证。Dubins车实验验证了所学可行集的数值精度。在安全强化学习基准上的评估表明该方法取得了最先进的奖励-（摘要在此处截断）

    arXiv:2611.07616v1 Announce Type: new  Abstract: Offline safe reinforcement learning requires reliable long-horizon safety estimates from fixed data. Learning Hamilton-Jacobi (HJ) viability in this setting calls for semantic preservation, convergence, and numerical stability. To address these challenges, we propose Monotone Viability Learning (MoViL), which learns viable sets from offline transitions through monotone finite-horizon set recursion. The recursion preserves finite-horizon safety semantics and converges to the infinite-horizon viability kernel. Binary supervision maintains a fixed target scale across horizons, avoiding discount-induced numerical attenuation. We further establish sufficient conditions for recovering behavior-supported viable subsets. The learned viability estimates are used to certify actions at deployment. Dubins car experiments validate the numerical accuracy of the learned viable sets. Evaluations on safe RL benchmarks demonstrate state-of-the-art reward-
    
[^121]: LoRA微调的嵌入空间指纹：仅凭模型产物恢复训练数据领域

    Embedding-Space Fingerprinting of LoRA Fine-Tunes: Recovering the Training-Data Domain from the Model Artifact Alone

    [https://arxiv.org/abs/2611.07610](https://arxiv.org/abs/2611.07610)

    本文提出嵌入空间指纹方法，证明无需接触训练数据，仅凭LoRA微调模型的产物（权重增量、内部激活或生成文本）即可识别其训练数据的领域，其中灰盒激活特征在留一数据集协议下达到61%的准确率，远高于20%的随机水平。

    

    当基础语言模型通过低秩适配（LoRA）进行微调时，训练数据的领域信息会有多少泄露到最终产物中？本文研究一个简单的分类器能否在从未观察训练数据的情况下，仅凭模型产物（权重增量、固定探针集上的内部激活，或生成文本）恢复微调模型的所属领域。研究使用一个基础模型（Qwen3.5-0.8B），在五个领域（医学、代码、数学、非英语文本、叙事散文）的18个公开数据集上生成了36个LoRA微调模型，并在访问权限递减的三个层级上提取了三类特征：白盒权重、灰盒激活和黑盒输出。采用在任何结果产生之前就已冻结的留一数据集（LODO）协议，测试能否识别在先前未见过的数据集上训练的微调模型的领域。结果表明，领域信息可以通过灰盒激活以远高于随机水平（20%）的准确率（61%）恢复（摘要在此处被截断）。

    arXiv:2611.07610v1 Announce Type: new  Abstract: When a base language model is fine-tuned with Low-Rank Adaptation (LoRA), how much of the training data domain leaks into the resulting artifact? This paper studies whether a simple classifier can recover the domain of a fine-tune given only the model artifact (weight deltas, internal activations on a fixed probe set, or generated text) without ever observing the training data. Using one base model (Qwen3.5-0.8B), 36 LoRA fine-tunes are produced over 18 public datasets in five domains (medical, code, mathematics, non-English text, narrative prose), and three feature families are extracted at decreasing levels of access: white-box weights, gray-box activations, and black-box outputs. A leave-one-dataset-out (LODO) protocol, frozen before any result was seen, tests whether the domain of a fine-tune trained on a previously unseen dataset can be identified. The domain is recoverable well above chance (20%) from gray-box activations (61%) and
    
[^122]: 部分可观测环境下sLLM智能体中改述指令与所遇对象的绑定

    Binding Paraphrased Instructions to Encountered Objects in sLLM Agents under Partial Observability

    [https://arxiv.org/abs/2611.07599](https://arxiv.org/abs/2611.07599)

    提出了无需训练的BINDIntent框架，利用语义接近度将改述指令中的指称表达与部分可观测环境中观察到的对象标签进行概率性绑定，显著提升了sLLM智能体面对指令同义改述时的鲁棒性。

    

    LLM智能体的可靠部署需要将多样化的用户表达与其实际环境进行对接。尽管经过强化学习微调的较小规模大语言模型智能体在具身任务上可以媲美更大的提示型LLM，但它们对指令同义改述的鲁棒性仍不清楚。我们引入ALFWorld-Para基准，在受控改述条件下评估这种鲁棒性，发现成功率下降21-52%，且主要由对象同义词导致。尽管智能体保留了同义词指代对象的知识，但即使在观察到这些对象后，它们也常常无法拿取相应对象，这暴露了指称知识与对象绑定之间的差距。部分可观测性进一步使绑定问题复杂化，因为指代对象可能尚未被看到，或者已经在不同的标签名称下被遇到过。为应对这一挑战，我们提出了BINDIntent，这是一个无需训练的框架，利用语义接近度以概率方式将指称表达与观察到的标签进行绑定。

    arXiv:2611.07599v1 Announce Type: new  Abstract: Reliable deployment of LLM agents requires grounding diverse user expressions in their environments. Although RL-finetuned smaller large language model (sLLM) agents rival larger prompted LLMs on embodied tasks, their robustness to instruction paraphrases remains unclear. We introduce ALFWorld-Para to evaluate this robustness under controlled paraphrases, revealing 21-52% declines in success rates (SRs), predominantly driven by object synonyms. Despite retaining knowledge of synonym referents, agents often fail to take the corresponding objects even after observing them, exposing a gap between referential knowledge and object binding. Partial observability further complicates binding, as referents may remain unseen or have already been encountered under different labels. To address this challenge, we propose BINDIntent, a training-free framework that probabilistically binds referring expressions to observed labels using semantic proximit
    
[^123]: TACT-KV：用于压缩医学视觉语言模型中体积KV缓存的三轴余弦变换

    TACT-KV: Tri-Axis Cosine Transform for Compressing Volumetric KV Caches in Medical VLMs

    [https://arxiv.org/abs/2611.07593](https://arxiv.org/abs/2611.07593)

    提出TACT-KV，一种无需校准的KV压缩方法，通过对KV缓存施加可分离的3D离散余弦变换并结合逆注水比特分配，利用3D医学扫描的三轴空间相关性来压缩医学视觉语言模型中的体积KV缓存。

    

    应用于3D医学扫描的视觉语言模型（VLM）面临结构性的内存墙问题。单个CT或MRI体积会产生数万个视觉token，由此产生的键值（KV）缓存随批大小和上下文长度线性增长。现有的KV压缩方法仅在缓存的特征或token统计上操作，或应用一维和二维变换，未能利用缓存中的三轴空间结构。我们提出TACT-KV（三轴余弦变换），这是一种无需校准的KV压缩方法，它利用3D扫描中视觉KV状态沿所有三个解剖轴的强空间相关性。这种相关性使KV缓存的能量在频域变换下得以集中。TACT-KV对KV缓存应用可分离的3D离散余弦变换，通过逆注水算法在频率区间上分配每层比特预算（为高能量分量分配更多比特），并使用固定的标量码本对每个区间进行量化。

    arXiv:2611.07593v1 Announce Type: cross  Abstract: Vision-language models (VLMs) applied to 3D medical scans face a structural memory wall. A single CT or MRI volume produces tens of thousands of vision tokens, and the resulting key-value (KV) cache scales linearly with both batch and context. Existing KV compression methods operate on the cache's feature or token statistics, or apply 1D and 2D transforms, leaving the three-axis spatial structure in the cache unexploited. We propose TACT-KV (Tri-Axis Cosine Transform), a calibration-free KV compression method that exploits strong spatial correlations in visual KV states along all three anatomical axes of 3D scans. This correlation concentrates the KV cache energy under a frequency-domain transform. TACT-KV applies a separable 3D discrete cosine transform to the KV cache, allocates a per-layer bit budget across frequency bins via reverse water-filling (more bits to high-energy components), and quantizes each bin with a fixed scalar code
    
[^124]: 不完全信息博弈中的学习与引导

    Learning and Steering in Incomplete-Information Games

    [https://arxiv.org/abs/2611.07589](https://arxiv.org/abs/2611.07589)

    该论文提出了仅利用轨迹反馈就能在不知道智能体效用的情况下学习标准式和扩展式博弈的二分类型算法，使调解者能够引导自适应智能体走向效用最大化的结果，并达到了近乎最优的样本复杂度。

    

    在许多现代多智能体系统中，调解者希望仅通过结果层面的反馈来影响策略性决策动态，而无需了解智能体的效用。在本工作中，我们研究调解者如何学习一个未知的博弈，并仅使用轨迹反馈将自适应的智能体引导至使其自身效用最大化的结果。调解者知道博弈的形式和自己的目标，但不知道智能体的效用。此外，她承诺支付非负的款项，并提供推荐和上下文信息，但她只能获得单个已实现的终端结果，而非全信息反馈。我们开发了二分类型（halving-type）算法，用于学习标准式博弈中的效用差异和扩展式博弈中的条件延续价值差异，实现了 $\widetilde O(\varepsilon^{-2})$ 的样本复杂度，在最坏情况下与下界仅相差对数因子。随后，我们将这些学习程序与拉格朗日方法相结合……

    arXiv:2611.07589v1 Announce Type: cross  Abstract: In many modern multi-agent systems, a mediator seeks to influence strategic decision dynamics using only outcome-level feedback and without knowing agents' utilities. In this work, we study how a mediator can learn an unknown game and steer adaptive agents toward outcomes that maximize her utility using only trajectory feedback. The mediator knows the game form and her own objective, but not the agents' utilities. Moreover, she commits to nonnegative payments and provides recommendations and contexts, but receives only a single realized terminal outcome rather than full-information feedback. We develop halving-type algorithms that learn utility differences in normal-form games and conditional continuation-value differences in extensive-form games, achieving a sample complexity of $\widetilde O(\varepsilon^{-2})$ that matches worst-case lower bounds up to logarithmic factors. We then combine these learning procedures with a Lagrangian f
    
[^125]: 针对隐蔽失对齐行为的思维链监控压力测试

    Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment

    [https://arxiv.org/abs/2611.07586](https://arxiv.org/abs/2611.07586)

    提出自动化红队测试框架 TraceMRT，通过将副任务重构为主任务的连贯子目标并搜索轨迹塑形策略，系统性地压力测试了思维链监控对会隐蔽掩盖失对齐行为的智能体的鲁棒性。

    

    随着大语言模型（LLM）智能体在自主性方面快速提升，智能体出现绕过安全约束的隐藏目标的风险日益增加。思维链（CoT）监控提供了一种关键的控制机制：通过读取智能体的推理过程，在失对齐行为付诸实施之前将其标记出来。然而其可靠性取决于对它进行压力测试的红队攻击的真实程度。现有的红队测试工作依赖于共享模板，例如将副任务伪装为完成主任务的前置条件，这种做法将副目标视为外生插入，而对任务特定的、有上下文依据的规避策略则探索不足。因此我们提出这样一个问题：面对会战略性隐藏自身失对齐行为的智能体，CoT 监控究竟有多强的鲁棒性？为了回答这一问题，我们提出了 TraceMRT——一个自动化的轨迹级（Trace-level）监控红队测试框架，它将副任务重新构建为主任务的一个连贯子目标，并搜索轨迹塑形策略，使其诱导出的轨迹能够隐蔽地掩盖失对齐意图……

    arXiv:2611.07586v1 Announce Type: cross  Abstract: As LLM agents advance rapidly in autonomy, the risk of hidden objectives that bypass safety constraints becomes increasingly plausible. Chain-of-thought (CoT) monitoring offers a key control mechanism by reading the agent's reasoning and flagging misalignment before it materializes into action, yet its reliability depends on the realism of the red-team it is stress-tested against. Existing red-teaming efforts rely on shared templates, such as casting a side task as a prerequisite, which treat the side objective as an exogenous insertion and leave task-specific, contextually grounded evasions underexplored. We therefore ask: How robust is CoT monitoring against agents that strategically conceal misalignment? To answer this, we introduce TraceMRT, an automated Trace-level Monitor Red-Teaming framework that reframes the side task as a coherent sub-goal of the primary task and searches trace-shaping strategies whose induced trajectories co
    
[^126]: ACSeL：面向硬件受限大语言模型推理的自适应配置搜索

    ACSeL: Adaptive Configuration Search for Hardware-Constrained LLM Inference

    [https://arxiv.org/abs/2611.07570](https://arxiv.org/abs/2611.07570)

    ACSeL通过基于相对模型质量退化（RMQD）的代理指标，在满足内存和延迟约束的前提下，自适应地搜索大语言模型推理的最优配置组合（模型大小、批处理大小、量化方案），从而在硬件受限部署中实现最高生成质量。

    

    arXiv:2611.07570v1 公告类型：新论文 摘要：在严格的内存和延迟服务级目标（SLO）下部署大语言模型（LLM）服务，包括在边缘设备上的部署，需要联合选择模型大小、批处理大小、量化算法以及混合精度量化方案。对这一组合爆炸式的巨大空间进行穷举搜索是不可行的。与通常优先考虑吞吐量或生成速度的数据中心服务不同，资源受限的部署通常将内存和延迟预算设为约束条件，然后在可行的配置中寻找生成质量最高的方案。为此，我们提出了ACSeL，一个能够识别出预测满足部署约束、同时实现高预测生成质量的配置的框架。为了实现高效的质量估计，我们使用相对模型质量退化（RMQD）作为代理指标，该指标基于候选模型与参考模型在（原文在此处截断）下一个词分布之间的散度……

    arXiv:2611.07570v1 Announce Type: new  Abstract: Serving large language models (LLMs) under tight memory and latency service-level objectives (SLOs), including on edge devices, requires jointly selecting the model size, batch size, quantization algorithm, and mixed-precision quantization recipes. Exhaustive search over this combinatorially large space is infeasible. Unlike data-center serving, which often prioritizes throughput or generation speed, resource-constrained deployments typically set memory and latency budgets as constraints and then seek the configuration with the highest generation quality among feasible ones. To this end, we propose ACSeL, a framework that identifies configurations predicted to satisfy deployment constraints while achieving high predicted generation quality. To enable efficient quality estimation, we use Relative Model Quality Degradation (RMQD), a proxy based on the divergence between the candidate and reference models' next-token distributions over a sm
    
[^127]: MambaTG：面向时序图未来链路预测的候选感知状态空间建模

    MambaTG: Candidate-Aware State Space Modeling for Future Link Prediction on Temporal Graphs

    [https://arxiv.org/abs/2611.07569](https://arxiv.org/abs/2611.07569)

    该论文提出MambaTG模型，通过候选感知的状态空间建模，融合历史序列演化与候选—历史兼容性两种互补信号，显著提升时序图上未来链路预测（尤其是未见边预测）的性能。

    

    时序图上的未来链路预测涉及两类情形：此前已连接节点对之间的重复交互，以及此前没有任何交互的节点对之间的首次交互。重复边预测可以直接利用成对的历史证据，而未见边的预测则必须泛化到历史重现模式之外。现有时序图模型通常依赖与重复交互或历史重叠相关的信号，这类信号对重复边蕴含丰富信息，但在未见边情形下变得有限甚至不可用，导致模型在未见边预测上性能大幅下降。基于这一观察，我们识别出两种用于建模超越历史重现的序列动态的互补信号：（1）历史序列演化，捕捉源节点按时间顺序排列的各次交互之间的依赖关系；（2）候选—历史兼容性，刻画候选节点与历史交互的契合程度……

    arXiv:2611.07569v1 Announce Type: new  Abstract: Future link prediction on temporal graphs involves both repeated interactions between previously connected node pairs and first-time interactions between pairs with no prior interaction. While repeated-edge prediction can exploit direct pairwise evidence, unseen-edge prediction must generalize beyond historical recurrence. Existing temporal graph models often exploit signals tied to recurring interactions or historical overlap, which are highly informative for repeated edges but become limited or unavailable for unseen edges, contributing to substantial performance degradation on unseen-edge prediction. Based on this observation, we identify two complementary signals for modeling sequential dynamics beyond historical recurrence: (1) historical-sequence evolution, which captures dependencies among the source node's chronologically ordered interactions, and (2) candidate--history compatibility, which captures how well a candidate fits the 
    
[^128]: 基于人类反馈强化学习的可证明零阶安全探索

    Provable Zeroth-Order Safe Exploration for Reinforcement Learning from Human Feedback

    [https://arxiv.org/abs/2611.07561](https://arxiv.org/abs/2611.07561)

    提出了一种无需奖励模型的零阶对数障碍策略梯度方法ZO-LBPG，通过保守约束余量估计、耦合轨迹采样和自适应步长，首次以高概率保证RLHF训练全过程中所有策略（包括探索策略）严格满足安全约束。

    

    安全的人类反馈强化学习旨在提升模型有用性的同时满足无害性约束。现有的原始-对偶方法虽然可以收敛到可行策略，但无法防止训练过程中的约束违反。我们研究了一个更强的要求——RLHF的安全探索，即训练期间执行的每一个策略（包括辅助探索策略）都必须保持可行。我们提出了零阶对数障碍策略梯度方法（ZO-LBPG），这是一种无需奖励模型的方法，它利用直接从人类偏好中导出的零阶估计来优化对数障碍目标函数。为了在含噪反馈下确保安全性，我们将保守的约束余量估计与耦合的轨迹采样以及自适应步长相结合，以防止更新越过约束边界。我们证明，以高概率，所有迭代过程中的策略和探索策略均保持严格可行，并且ZO-LBPG具有全局（摘要在此处截断）

    arXiv:2611.07561v1 Announce Type: new  Abstract: Safe reinforcement learning from human feedback (Safe-RLHF) seeks to improve helpfulness while satisfying harmlessness constraints. Existing primal-dual approaches can converge to feasible policies but do not prevent constraint violations during training. We study a stronger requirement, safe exploration for RLHF, under which every policy executed during training, including auxiliary exploration policies, remains feasible. We propose Zeroth-Order Log-Barrier Policy Gradient (ZO-LBPG), a reward-model-free method that optimizes a log-barrier objective using zeroth-order estimates derived directly from human preferences. To ensure safety under noisy feedback, we combine conservative constraint margin estimates with coupled rollouts and an adaptive step size that prevents updates from crossing constraint boundaries. We prove that, with high probability, all iterates and exploration policies remain strictly feasible, and that ZO-LBPG globally
    
[^129]: 记忆陷阱：语言智能体会从良性交互中过度泛化

    The Memory Trap: Language Agents Overgeneralize from Benign Interactions

    [https://arxiv.org/abs/2611.07554](https://arxiv.org/abs/2611.07554)

    本文发现并命名了“记忆陷阱”这一新型失效模式：语言智能体会从良性交互记忆中过度泛化，例如低风险任务中跳过确认的经历会使其在高风险任务中也跳过安全验证，而情感支持或被批评经历形成的记忆则会使后续判断偏向谄媚或过度批评。

    

    语言智能体越来越多地使用持久化记忆来跨会话记录用户偏好和操作流程。我们证明，来自良性交互的记忆可能导致智能体在后续任务中出现不安全或不可靠的行为。我们将这种失效现象称为“记忆陷阱”。为研究这一问题，我们比较了六个前沿模型（包括 Claude Sonnet、Opus、Fable 模型和 GPT-5.6）在有记忆和无记忆会话中的行为差异，并在多种设置下对这些模型进行了评估。首先，在自主性实验中，我们发现用户要求在低风险编程杂务中跳过确认的请求，会导致智能体在编程和日常生活的模拟高风险任务中也跳过确认与验证。其次，我们研究了当用户寻求情感支持、或描述不加批判的反馈所造成的伤害时形成的记忆，发现这些记忆会使智能体在后续对数学作业、论证和诗歌的评价中分别偏向谄媚或过度批评。

    arXiv:2611.07554v1 Announce Type: cross  Abstract: Language agents increasingly use persistent memory to record user preferences and procedures across sessions. We show that memory from benign interactions can cause unsafe or unreliable behavior on later tasks. We call this failure the memory trap. To study this issue, we compare behavior in sessions with and without memory across six frontier models (including Claude Sonnet, Opus, and Fable models and GPT-5.6). We evaluate these models in a range of settings. First, in autonomy experiments, we find that requests to skip confirmation during low-stakes coding chores lead agents to skip confirmation and verification on simulated high-stakes tasks in both coding and everyday life. We next study memories formed when users seek emotional support, or describe harm from uncritical feedback, and find these memories bias later judgments of mathematical work, arguments, and poems toward sycophancy or over-critique (respectively). Memory notes wr
    
[^130]: 理解交通网络中的数据影响：一个物理约束的影响函数框架

    Understanding Data Influence in Transportation Networks: A Physics-Constrained Influence-Function Framework

    [https://arxiv.org/abs/2611.07526](https://arxiv.org/abs/2611.07526)

    本文提出面向约束学习的硬物理约束影响函数框架，在交通网络物理可行空间内刻画训练数据对模型的影响，揭示了物理约束如何改变影响的幅度、方向与空间传播。

    

    理解单个训练观测样本如何影响模型预测，对于构建可靠且可解释的数据驱动交通系统至关重要。然而，现有影响函数方法主要是针对无约束学习设计的，并未显式考虑将交通网络各组件耦合在一起的物理约束。在此类约束下，对训练观测样本的扰动只能通过物理可行的方向影响模型参数和预测，这可能改变其影响在网络中的传播方式。本文针对约束学习提出了一种硬物理信息影响函数框架，在该框架中，训练数据的影响是在由网络约束所定义的物理可行空间内进行刻画的。解析结果揭示了物理约束如何改变训练样本影响的幅度、方向和空间分布。

    arXiv:2611.07526v1 Announce Type: new  Abstract: Understanding how individual training observations influence model predictions is important for developing reliable and interpretable data-driven transportation systems. Existing influence-function methods, however, are primarily formulated for unconstrained learning and do not explicitly account for physical constraints that couple different components of a transportation network. Under such constraints, perturbing a training observation can affect model parameters and predictions only through physically feasible directions, potentially changing how its influence propagates across the network. This paper develops a hard physics-informed influence-function framework for constrained learning, in which training-data influence is characterized within the physically feasible space defined by network constraints. The analytical results reveal how physical constraints alter the magnitude, direction, and spatial distribution of training-sample 
    
[^131]: 使用Transformer生成三维流形三角剖分

    Using Transformers to Generate 3-Manifold Triangulations

    [https://arxiv.org/abs/2611.07514](https://arxiv.org/abs/2611.07514)

    该论文首次将GPT-2风格的Transformer应用于生成三维流形的三角剖分，并通过IsoFLOP曲线分析揭示了神经缩放规律（参数缩放指数α=0.49，每个参数234个token的最优训练比例）。

    

    使用脱水签名作为编码，我们训练GPT-2风格的Transformer来生成三维流形的三角剖分。我们的主要实验每次运行约使用3 exaFLOP的计算量，涵盖了从在6000万个球面三角剖分上训练的2亿参数模型，到在11亿个球面三角剖分上训练的1100万参数模型。最优模型包含10个块、宽度为640，能以6.9%的比率生成具有各种拓扑类型的闭流形的脱水签名，其中82%为球面。为了研究我们的方法如何随规模扩展，我们在三维球面的15个四面体三角剖分上采用IsoFLOP曲线技术，计算量范围从6到3000 petaFLOP，并观察到了神经缩放行为。我们提取出参数缩放指数 $\alpha = 0.49$（[0.48, 0.50]为统计误差，±0.02为系统误差），以及训练中每个参数对应234个token的最优比例。此外，我们还展示了若干较小的实验……

    arXiv:2611.07514v1 Announce Type: cross  Abstract: Using dehydration signatures as an encoding, we train GPT-2-style transformers to generate triangulations of 3-dimensional manifolds. Our main runs each use approximately 3 exaFLOPs, spanning a 200 million parameter model trained on 60 million sphere triangulations to an 11 million parameter model trained on 1.1 billion sphere triangulations. The optimal model has 10 blocks, width 640 and generates dehydration signatures of closed manifolds of various topological types at a rate of 6.9%, 82% of which are spheres. To investigate how our method scales, we use the technique of IsoFLOP curves on 15-tetrahedron triangulations of the 3-sphere, over a compute range of 6 to 3000 petaFLOPs, and observe neural scaling behaviour. We extract a parameter scaling exponent of $\alpha = 0.49$ ($[0.48, 0.50]$ stat, $\pm 0.02$ syst), and an optimal ratio of 234 tokens per parameter for training. In addition, we present a number of smaller experiments, i
    
[^132]: RLGPA：一种用于基于决策的稀疏对抗样本的强化学习引导粒度规划攻击

    RLGPA: A Reinforcement Learning-Guided Granularity Planning Attack for Decision-Based Sparse Adversarial Examples

    [https://arxiv.org/abs/2611.07512](https://arxiv.org/abs/2611.07512)

    提出RLGPA，一种利用在线强化学习在查询预算约束下自适应调度不同粒度支撑集缩减策略的方法，在基于决策的稀疏对抗攻击中显著超越现有方法（CIFAR-10上定向攻击成功率达61%，ResNet50/ViT上达75.5%/73.0%）。

    

    基于决策的稀疏攻击必须仅利用受害模型预测的标签来减少扰动的支撑集。随着支撑集不断收缩，最佳的恢复策略也随之变化：粗粒度的删除在早期有效，而尾部状态则需要更精细、具备历史感知能力的测试。我们提出RLGPA，这是一种在线强化学习方法，能够在严格的查询预算下调度支撑集缩减执行器及其操作粒度。一个带掩码的对偶双重DQN（Dueling Double DQN）在七个稳定的策略-尺度动作中进行选择；每个动作都会生成与支撑集相关的恢复组，并在每次宏观决策后重新构建。一个保持可行性的执行器只提交保持对抗性的恢复结果，并将支撑集增益和查询成本反馈给规划器。在匹配的top-1标签预言机条件下，RLGPA在CIFAR-10上稀疏度为0.01时达到61%的定向攻击成功率（相比之下SparseEvo为25%，Pointwise为0%），在ResNet50/ViT上稀疏度为0.005时分别达到75.5%/73.0%（相比之下为29.0%/15.5%）。固定探针测试将中位数L0（摘要在此处截断）

    arXiv:2611.07512v1 Announce Type: new  Abstract: Decision-based sparse attacks must reduce perturbation support using only the victim's predicted label. The best restoration strategy varies as support contracts: coarse removal works early, while tail states need finer, history-aware tests. We propose RLGPA, an online reinforcement learning method that schedules support-reduction executors and their granularity under a hard query budget. A masked Dueling Double DQN selects among seven stable policy--scale actions; each action produces support-relative restoration groups that are rebuilt after every macro decision. A feasibility-preserving executor commits only adversarial restorations and feeds support gain and query cost back to the planner. Under the matched top-1-label oracle, RLGPA reaches 61\% targeted ASR at sparsity 0.01 on CIFAR-10 (vs. 25\% SparseEvo, 0\% Pointwise), and 75.5\%/73.0\% on ResNet50/ViT at sparsity 0.005 (vs. 29.0\%/15.5\%). A fixed-probe test reduces median \(L_0
    
[^133]: 覆盖率掩盖了决策变动：随着VLM标签注册表增长对保形决策的审计与保持

    Coverage Hides Churn: Auditing and Preserving Conformal Decisions as VLM Label Registries Grow

    [https://arxiv.org/abs/2611.07502](https://arxiv.org/abs/2611.07502)

    提出PrefixCP方法，揭示了VLM标签注册表增长时基于softmax的保形分数会因标签均值归一化变化而悄然改变旧标签决策（覆盖率指标无法检测到这种变动），并提供了审计与保持决策稳定性的解决方案。

    

    视觉-语言分类器能够仅通过文本学习新标签，因此其服务的标签列表会持续增长。我们证明这种增长会悄然改变关于旧标签的保形决策。常见的基于softmax的保形分数将每个标签与所有标签的平均值进行比较；添加新标签会改变这一平均值，进而改变每个旧标签的分数，即使模型、图像和旧的校准数据都保持不变。在更新后重新校准虽然能保持覆盖率保证，但在十一种设置中，它会在53-90%的新图像上至少改变一个旧决策。覆盖率无法揭示这一点，因为它只检查真实标签：在五个保持旧校准不变的受控设置中（仅有归一化因子变化），旧类别覆盖率最多移动0.01，而60-67%的图像却发生了变化。我们提出一种生命周期审计方法，将这些变化与覆盖率和预测集合大小一同进行测量，并提出一个简单的修复方案PrefixCP：每个类别（摘要在此处被截断）

    arXiv:2611.07502v1 Announce Type: cross  Abstract: Vision-language classifiers can learn new labels from text alone, so the label lists they serve keep growing. We show that this growth silently changes conformal decisions about old labels. Common softmax-based conformal scores compare each label with an average over all labels; adding labels changes that average, and with it the score of every old label, even when the model, the image, and the old calibration data stay the same. Recalibrating after the update keeps the coverage guarantee, yet across eleven settings it changes at least one old decision on 53-90% of new images. Coverage cannot reveal this, because it checks only the true label: in five controlled settings that keep the old calibration, so that only the normalizer changes, old-class coverage moves by at most 0.01 while 60-67% of images change. We propose a lifecycle audit that measures these changes alongside coverage and set size, and a simple fix, PrefixCP: each class 
    
[^134]: ZYFA：AMD CDNA4 上 FP8 FlashAttention 的速度与精度

    ZYFA: Speed and Accuracy of FP8 FlashAttention on AMD CDNA4

    [https://arxiv.org/abs/2611.07496](https://arxiv.org/abs/2611.07496)

    ZYFA 是针对 AMD MI355X 的 FP8 FlashAttention 预填充内核，通过直接从分数计算 FP8 概率等三项关键技术，解决了 FP8 加速下 softmax 无法加速及硬件缺乏指令完成跟踪的瓶颈，在保持精度的同时显著提升了长上下文注意力的吞吐量。

    

    注意力机制是前沿模型的核心构建模块，在长上下文场景下常常主导计算受限的预填充阶段。FP8 矩阵指令可提供两倍于 BF16 的峰值吞吐量，但它只能加速矩阵乘法，而注意力中两次矩阵乘法之间的 softmax 运行在 FP8 无法加速的向量单元上。在 AMD Instinct MI355X 上，厂商的 AITER 内核在 32K tokens 时仅达到 FP8 矩阵峰值吞吐量的 39%，而 BF16 为 49%。注意力机制对量化的数值敏感性也高于前馈层，并且与 NVIDIA 的 Hopper 和 Blackwell GPU 不同，MI355X 不在硬件中跟踪矩阵指令的完成状态，因此 softmax 与矩阵乘法的重叠执行取决于指令顺序。我们提出了 ZYFA，这是为 MI355X 构建的预填充内核，基于三项技术：(1) 通过一次乘加运算和整数操作直接从分数计算每个 FP8 概率……

    arXiv:2611.07496v1 Announce Type: cross  Abstract: Attention is a core building block of frontier models and often dominates compute-bound prefill at long context. FP8 matrix instructions offer twice the peak throughput of BF16, but they accelerate only matrix multiplication, and the softmax between the two matrix multiplications of attention runs on vector units that FP8 does not accelerate. On the AMD Instinct MI355X, the vendor's AITER kernels reach 39% of peak FP8 matrix throughput at 32K tokens, compared with 49% in BF16. Attention is also more numerically sensitive to quantization than feed-forward layers, and unlike NVIDIA's Hopper and Blackwell GPUs, the MI355X does not track the completion of matrix instructions in hardware, so overlapping the softmax with the matrix multiplications depends on instruction order. We present ZYFA, prefill kernels for the MI355X built on three techniques: (1) computing each FP8 probability directly from its score with a multiply-add and an intege
    
[^135]: 学习多智能体编排：策略提升中的高阶交互

    Learning Multi-Agent Orchestration:Higher-Order Interactions in Policy Improvement

    [https://arxiv.org/abs/2611.07495](https://arxiv.org/abs/2611.07495)

    提出交互感知策略优化框架IAPO，通过将两个策略间的性能差异精确分解为个体决策效果及其交互效果，显式建模LLM多智能体编排中决策之间的高阶依赖关系，从而改进编排器的训练。

    

    学习型编排已成为基于大语言模型（LLM）的多智能体系统中一种颇具前景的范式，其中一个中央编排器在任务执行过程中选择智能体并控制它们所接收的信息。尽管实证进展迅速，现有方法在训练编排器时，是将每个决策单独归因于其自身的优势，隐含地假设回合中的其他决策保持不变。这一假设在编排任务中并不成立，因为每个决策对最终结果的贡献取决于其之前的决策，因此联合更新的收益并不等于各决策单独效果之和。为应对这一挑战，我们提出了IAPO，一种能够显式考虑决策之间交互的交互感知策略优化框架。具体而言，IAPO将两个策略之间的性能差异精确分解为个体决策的效果及其交互效果。

    arXiv:2611.07495v1 Announce Type: new  Abstract: Learned orchestration has emerged as a promising paradigm for large language model (LLM)-based multi-agent systems, in which a central orchestrator selects agents and controls the information they receive throughout a task. Despite rapid empirical progress, existing methods train the orchestrator by crediting each decision with its own advantage, implicitly assuming that the other decisions in the episode stay fixed. This assumption fails in orchestration, where the contribution of each decision to the final outcome depends on the decisions preceding it, so the gain of a joint update is not the sum of its per-decision effects. To address this challenge, we introduce \textbf{IAPO}, an interaction-aware policy optimization framework that explicitly accounts for interactions among decisions. Specifically, IAPO exactly decomposes the performance difference between two policies into the effects of individual decisions and their interactions, 
    
[^136]: 通过随机梯度朗之万动力学实现深度M回归的统计与隐私保证

    Statistical and Privacy Guarantees for Deep M-Regression via Stochastic Gradient Langevin Dynamics

    [https://arxiv.org/abs/2611.07490](https://arxiv.org/abs/2611.07490)

    该论文利用针对深度ReLU网络的PAC-贝叶斯分析，为通过SGLD训练的深度M回归建立了同时刻画网络逼近误差、统计复杂度和算法累积代价的超额风险界，证明注入适度高斯噪声的估计量在Hölder光滑条件下达到最优非参数回归速率，并为SGLD噪声带来的泛化益处及隐私保证提供了理论解释。

    

    本文研究了通过随机梯度朗之万动力学训练的深度非参数回归，SGLD是随机梯度下降的一种变体，它在优化过程中注入经过精心校准的高斯噪声。利用针对深度ReLU网络量身定制的PAC-贝叶斯分析，我们建立了一个超额风险界，该风险界同时考虑了网络逼近误差、统计复杂度以及沿SGLD轨迹累积的算法代价。在Bernstein条件以及稳定的SGLD运行区域内，该超额风险界包含一个随机项，其系数与注入噪声的方差成反比，同时该估计量在Hölder光滑性条件下仍保持非参数回归的最优速率（最多相差对数因子）。这一结果为人们广泛观察到的适度SGLD噪声所带来的泛化益处提供了一个兼顾逼近误差、速率最优的统计视角。

    arXiv:2611.07490v1 Announce Type: cross  Abstract: In this article, we study deep nonparametric regression trained via stochastic gradient Langevin dynamics (SGLD), a variant of stochastic gradient descent that injects carefully calibrated Gaussian noise during optimization. Using a PAC-Bayesian analysis tailored to deep ReLU networks, we establish an excess-risk bound that simultaneously accounts for network approximation error, statistical complexity, and the algorithmic cost accumulated along the SGLD trajectory. Under a Bernstein condition and within a stable SGLD regime, the excess-risk bound contains a stochastic term whose coefficient is inversely proportional to the variance of injected noise, while the estimator retains the optimal rate for nonparametric regression under the H\"older smoothness condition up to logarithmic factors. This result provides an approximation-error-aware, rate-optimal statistical perspective on the widely observed generalization benefits of moderate S
    
[^137]: OPRD-Imp：多步药物合成中实验观察到的杂质与副产物数据集

    OPRD-Imp: a dataset of experimentally observed impurities and side products in multi-step pharmaceutical syntheses

    [https://arxiv.org/abs/2611.07488](https://arxiv.org/abs/2611.07488)

    该论文构建了首个公开的多步药物合成杂质数据集OPRD-Imp，包含229个反应和784条经专家验证的杂质条目，为机器学习杂质预测模型的基准测试提供了关键数据资源。

    

    预测反应杂质是药物工艺化学中一个关键但长期受数据限制的挑战。尽管用于反应结果预测的机器学习模型发展迅速，但由于缺乏经过整理且可公开访问的数据集，对这些模型杂质预测能力进行基准测试仍难以实现。我们提出了OPRD-Imp数据集，该数据集包含来自《Organic Process Research & Development》期刊62篇出版物的229个有机反应，每个反应均标注了实验观察到的杂质及其机理来源。数据集共提供784条杂质条目，并按来源分为四个类别。反应以反应SMILES格式表示，杂质以单独的SMILES字符串表示，同时保留了多步合成路线的连接性。所有数据均由经过培训的化学家按照标准化提取协议从原始出版物中手动提取，并经过专家化学验证和人工审核。

    arXiv:2611.07488v1 Announce Type: cross  Abstract: Predicting reaction impurities is a critical yet persistently data-limited challenge in pharmaceutical process chemistry. While machine learning models for reaction outcome prediction have advanced rapidly, benchmarking their ability to predict impurities remains impractical due to the absence of curated, publicly accessible datasets. We present OPRD-Imp, a dataset of 229 organic reactions drawn from 62 publications in Organic Process Research & Development, each annotated with experimentally observed impurities and their mechanistic origins. A total of 784 impurity entries is provided, classified into four origin categories. Reactions are represented as reaction SMILES and impurities as individual SMILES strings, with multi-step synthetic route connectivity preserved. All data were manually extracted from the source publications by trained chemists following a standardized extraction protocol, with expert chemical validation and manua
    
[^138]: 定位并不意味着可操控性：LLM数学中符号错误的机制研究

    Localization Does Not Imply Steerability: A Mechanistic Study of Sign Errors in LLM Mathematics

    [https://arxiv.org/abs/2611.07478](https://arxiv.org/abs/2611.07478)

    该研究通过逐层因果分析揭示了LLM在多步数学中的符号错误主要由网络后四分之一区段的MLP层在后期产生，且不同模型具有不同机制（固定的“负号盖章”层、符号放大器、救援层），同时指出即便能精确定位错误来源，也不意味着能够有效操控和纠正它。

    

    当大语言模型进行多步数学运算时，一种特征性的失败反复出现：模型在计算中途写下一个错误的符号token——例如余子式的符号、递归分部积分步骤中的负号——随后正确执行其余的算术运算，但最终答案出错。我们探究这个符号在网络内部是在何处写下的。在涵盖14B至675B参数的六个开源模型中，逐层读出实验表明，错误的符号是在计算后期产生的——由网络最后四分之一区段的MLP层所为（每个模型和任务中占比70-100%），且在经过测试的情况下（Qwen），这一模式在指令微调之前的基础模型中就已存在。具体机制因模型而异：一些模型带有固定的“负号盖章”层，无论正确答案如何都会推出负号；另一些模型则带有对已选定符号的放大器；而Qwen还额外增加了一个下游的“救援层”来对抗这种盖章。该回路是因果性的：从残差流中减去单一符号方向（后文截断）……

    arXiv:2611.07478v1 Announce Type: cross  Abstract: When large language models do multi-step mathematics, a characteristic failure recurs: the model writes one wrong sign token mid-computation -- a cofactor sign, the minus of a recursive integration-by-parts step -- then executes the rest of the arithmetic correctly, and the answer comes out wrong. We ask where inside the network that sign is written. Across six open models spanning 14B to 675B parameters, a per-layer readout shows the wrong sign is committed late -- by MLP layers in the last quarter of the network (70-100% per model and task), a pattern present, where tested (Qwen), in the base model before instruction tuning. The mechanism varies: some models carry fixed "minus-stamp" layers that push a minus regardless of the right answer, others carry amplifiers of whichever sign was already chosen, and Qwen adds a downstream rescue layer that opposes the stamp. The circuit is causal: subtracting a single sign direction from the res
    
[^139]: KOSMOS：用于通信-计算重叠的自调度持久内核

    KOSMOS: Self-scheduling Persistent Kernels for Communication-Compute Overlap

    [https://arxiv.org/abs/2611.07477](https://arxiv.org/abs/2611.07477)

    KOSMOS 提出一种自调度持久内核库，其工作组利用按消费者需求排序的队列根据数据就绪状态在通信与计算角色间动态切换，将 Transformer 层所有 TP+SP 集合通信与 GEMM 融合，无需调度器或预设切分，在 AMD MI350X/MI355X GPU 上将 Megatron-LM 端到端训练最高加速 1.28 倍。

    

    张量与序列并行（TP+SP）以及专家并行（EP）会将集合通信置于 Transformer 层的关键路径上。最先进的融合内核通过固定角色或固定调度来缓解这一问题，但这种僵化会留下大量空闲窗口；而预测通信-计算切分同样无法消除它们，因为预先确定的切分无法跟随数据的实际就绪状态。我们提出了 KOSMOS，一个自调度持久内核库，其工作组使用按消费者需求排序的队列，在数据就绪时即可在通信与计算角色之间动态切换，无需任何调度器、每次调用的代价模型或预先调优的切分方案。KOSMOS 将 Transformer 层的每一个 TP+SP 集合通信与其对应的 GEMM 相融合，并在每个方向上仅需一次启动即可运行完整的专家层（从路由器到折叠）。在 AMD Instinct MI350X 和 AMD Instinct MI355X GPU 上的评估显示，KOSMOS 在 BF16 精度下将 Megatron-LM 的 TP+SP 端到端训练加速 1.10-1.28 倍，在 M（摘要在此处被截断）。

    arXiv:2611.07477v1 Announce Type: cross  Abstract: Tensor and sequence parallelism (TP+SP) and expert parallelism (EP) put collectives on the critical path of transformer layers. State-of-the-art fused kernels mitigate this with fixed roles or schedules, but this rigidity leaves large idle windows. Predicting a communication-computation split does not remove them either; a predetermined split cannot follow data readiness. We present KOSMOS, a library for self-scheduling persistent kernels whose workgroups use queues ordered by consumer demand to switch between communication and computation roles as data become ready, with no scheduler, per-call cost model, or tuned split. KOSMOS fuses every TP+SP collective of a transformer layer with its GEMM, and runs an entire expert layer, from router to fold, in one launch per direction. Evaluated on AMD Instinct MI350X and AMD Instinct MI355X GPUs, KOSMOS speeds up TP+SP end-to-end training in Megatron-LM by 1.10-1.28x in BF16 and 1.10-1.25x in M
    
[^140]: 核机器的对抗鲁棒性：一项全面分析

    Adversarial Robustness of Kernel Machines: A Comprehensive Analysis

    [https://arxiv.org/abs/2611.07474](https://arxiv.org/abs/2611.07474)

    本文通过白盒和黑盒攻击全面评估了核机器的对抗鲁棒性，发现核机器天生比神经网络更鲁棒，并提出了一种核机器的近似对抗训练算法。

    

    核机器是机器学习中一类重要的模型。然而，现代核机器对对抗样本的鲁棒性问题尚未得到全面研究。在本文中，我们通过白盒和黑盒逃避攻击，对基于核的模型在二分类和多分类任务中的对抗鲁棒性进行了深入的实证研究。我们评估了核机器、随机特征分类器以及相对较新的递归特征机器的固有鲁棒性。此外，我们提出了一种用于核机器近似对抗训练的算法。我们在图像和表格数据集分类任务上的实验有力地表明，核机器天生比神经网络更鲁棒，即在面对对抗扰动时其性能下降更为平缓。

    arXiv:2611.07474v1 Announce Type: cross  Abstract: Kernel machines are an important class of models in machine learning. However, the question of how robust modern kernel machines are to adversarial examples has not been studied comprehensively. In this paper, we provide a thorough empirical investigation of the adversarial robustness of kernel-based models for both binary and multiclass classification tasks, via white-box and black-box evasion attacks. We evaluate the native robustness of kernel machines, random feature classifiers, and the relatively newer Recursive Feature Machines. In addition, we provide an algorithm for approximate adversarial training of kernel machines. Our experiments on classification tasks on image and tabular datasets strongly indicate that kernel machines are natively more robust than neural networks, i.e., their performance degrades more gracefully against adversarial perturbations.
    
[^141]: 误差的回声：用于时间序列预测的残差方向局部滚动一致性

    Echoes of Error: Residual-Directional Local Rollout Consistency for Timeseries Forecasting

    [https://arxiv.org/abs/2611.07473](https://arxiv.org/abs/2611.07473)

    该论文提出“残差方向局部滚动一致性”方法，通过沿实际反馈扰动方向惩罚预测器的雅可比向量积这一一阶替代方案，精确抑制时间序列预测中自生成误差对未来预测的传播影响，从而避免设计难以实现的残差到状态对齐算子。

    

    在时间序列预测中，滚动误差不仅仅是一步拟合的偏差：一旦被反馈到不断演化的状态中，它们会扰动未来的预测。我们通过局部滚动一致性将这一效应形式化，该一致性比较在教师强制下产生的下一步预测与在自由运行下产生的预测。原则上，这种一致性可以通过跟踪由残差引起的下一状态扰动来直接实施。然而在实践中，当输入状态和输出块具有不同的维度、异构的组件或依赖于架构的布局时，残差到状态的对齐算子往往难以设计。因此，我们使用一种一阶局部替代方法，沿着实际的反馈扰动惩罚预测器的雅可比向量积，从而精确地抑制自生成误差对未来预测仍可见的那些方向。我们进一步给出……（摘要在此处截断）

    arXiv:2611.07473v1 Announce Type: cross  Abstract: In time-series forecasting, rollout errors are not merely one-step fitting discrepancies: once fed back into the evolving state, they can perturb future predictions. We formalize this effect through local rollout consistency, which compares the next prediction produced under teacher forcing with that produced under free running. In principle, this consistency can be enforced directly by tracking the residual-induced perturbation of the next state. In practice, however, the residual-to-state alignment operator is often difficult to design when the input state and output block have different dimensions, heterogeneous components, or architecture-dependent layouts. We therefore use a first-order local surrogate that penalizes the Jacobian-vector product of the predictor along the actual feedback perturbation, suppressing precisely the directions through which self-generated errors remain visible to future predictions. We further give the s
    
[^142]: 通往真实世界的最后一英里：作为解决真实世界任务的可复用构建块的智能体原生原语

    The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks

    [https://arxiv.org/abs/2611.07461](https://arxiv.org/abs/2611.07461)

    提出智能体原生原语与 HEELO 框架，通过为资源配备常驻 LLM 并从原语库组合出任务专用框架，弥合从获取资源到实际应用资源之间“通往真实世界的最后一英里”。

    

    智能体如何端到端地处理真实世界任务？配备各自框架的前沿模型在真实世界基准测试中仍然表现不佳：在 Terminal-Bench 4.0 上，GPT-5.6 Sol 配合 Codex 仅解决 37.3% 的任务，Claude Opus 5 配合 Claude Code 解决 51.8%。这些任务在从获取资源到应用资源的步骤上失败，我们将其称为通往真实世界的最后一英里。现有方法只是将资源置于可及范围内，却把应用资源的能力留给调用者，因此每个框架都需要重建这种能力。我们提出了智能体原生原语——一种可复用的单元，它为资源配备一个常驻 LLM，该 LLM 能在上下文中应用该资源并与其他原语通信；同时提出 HEELO（围绕不断演化的智能体原生原语构建的框架工程方法，用于真实世界任务的最后一英里），这是一个从此类原语库为每个任务构建框架的框架工程方法。HEELO 将任务所需的原语组合成任务特定的框架……

    arXiv:2611.07461v1 Announce Type: cross  Abstract: How can an agent handle a real-world task end to end? Frontier models with their own harnesses still fall short on real-world benchmarks: on Terminal-Bench 4.0, GPT-5.6 Sol with Codex resolves 37.3% of the tasks and Claude Opus 5 with Claude Code 51.8%. These tasks fail at the step from reaching a resource to applying it, which we call the last mile to the real world. Existing approaches put the resource within reach but leave the competence to apply it with the caller, so each harness rebuilds that competence. We introduce agent-native primitives, reusable units that equip a resource with a resident LLM that applies it in context and communicates with other primitives, and HEELO (Harness Engineering around Evolving Agent-Native Primitives for the Last Mile to Real-WOrld Tasks), a harness engineering framework that builds each task's harness from a library of such primitives. HEELO composes the primitives a task needs into a task-speci
    
[^143]: 延迟感知的自适应 Best-of-N 采样

    Latency-Aware Adaptive Best-of-$N$

    [https://arxiv.org/abs/2611.07453](https://arxiv.org/abs/2611.07453)

    该论文提出了一种延迟感知的两阶段自适应 Best-of-N 采样算法，通过比较期望的奖励提升与基于观测延迟预测的成本，动态决定是否抽取以及以多大规模抽取第二批样本，从而在服务器负载变化时更合理地分配测试时计算资源。

    

    测试时计算通过在推理阶段投入额外计算来改进语言模型的响应，而合适的计算量取决于具体的提示词。自适应 TTC 方法根据响应质量的信号（如观测到的奖励）来分配计算，通常在信号超过某个固定阈值时停止，并且一般以生成次数或 token 数量来衡量成本。但在在线服务场景中，延迟同样是一种成本：它会随服务器负载增加而增长，因此使用固定阈值的方法在生成速度快时投入的计算过少，而在生成速度慢时又投入过多。针对广泛使用的 TTC 方法 Best-of-N 采样，我们提出了一种两阶段自适应算法：先抽取一个初始批次，然后利用观测到的奖励和该批次的延迟，通过比较期望的改进与预测的成本来决定是否抽取第二个批次以及其大小：当第一批次的最佳奖励低于阈值时（原文摘要在此处截断）……

    arXiv:2611.07453v1 Announce Type: new  Abstract: Test-time compute (TTC) improves language model responses by spending additional computation at inference time, and the right amount of compute depends on the prompt. Adaptive TTC methods allocate compute based on signals of response quality such as observed rewards, often by stopping once the signal exceeds a fixed threshold, and generally measure cost by the number of generations or tokens. But in online serving, latency is also a cost: it grows with server load, so a method using a fixed threshold spends too little compute when generation is fast and too much when it is slow. For Best-of-$N$ sampling, a widely used TTC method, we propose a two-stage adaptive algorithm that draws an initial batch, then uses the observed rewards and the batch's latency to decide whether to draw a second batch and at what size by comparing expected improvement against predicted cost: it draws one when the first batch's best reward is below a threshold th
    
[^144]: 面向太阳辐照度预测校准不确定度的两阶段残差学习方法

    Two-stage residual learning for calibrated uncertainty in solar irradiance forecasts

    [https://arxiv.org/abs/2611.07444](https://arxiv.org/abs/2611.07444)

    本文提出一种两阶段残差学习方法：第一个LSTM网络预测预期太阳辐照度，第二个网络建模预测误差的分布，从而为太阳辐照度预测提供校准的不确定度估计。

    

    可再生能源发电的准确预测不仅需要点预测值，还需要对其所附不确定度的可靠估计，尤其是在较长的时间跨度上。机器学习模型越来越多地被用于预测太阳能输出，但它们通常只报告单一数值，难以很好地捕捉相关的不确定度。标准的补救方法是训练一个模型来直接预测假设概率分布的参数。本文受合成太阳信号生成的启发，开发了一种替代性的两阶段方法，其中预测误差与预期辐照度分别建模。第一个长短期记忆（LSTM）网络预测预期的全球水平辐照度，第二个网络对预测误差的分布进行建模，该误差被加回到均值上，从而同时给出预测值及其不确定度。利用国家太阳辐射数据库的历史天气数据……

    arXiv:2611.07444v1 Announce Type: new  Abstract: Accurate forecasting of renewable generation requires not only point predictions but also reliable estimates of the uncertainty attached to them, particularly at extended horizons. Machine learning models are increasingly used to predict solar output, but they typically report a single value and capture the associated uncertainty poorly. The standard remedy is to train a model to predict the parameters of an assumed probability distribution directly. Here we develop an alternative two-stage method, motivated by synthetic solar signal generation, in which the forecast error is modeled separately from the expected irradiance. A first long short-term memory (LSTM) network predicts the expected global horizontal irradiance, and a second network models the distribution of the forecast error, which is added back to the mean to give both the forecast and its uncertainty. Using historical weather data from the National Solar Radiation Database a
    
[^145]: IMUDINO：面向惯性传感的自蒸馏基础模型

    IMUDINO: Self-distilled Foundation Model for Inertial Sensing

    [https://arxiv.org/abs/2611.07437](https://arxiv.org/abs/2611.07437)

    IMUDINO是首个在人体活动识别、健康监测、动作捕捉和里程计等多种惯性传感任务上均达到有竞争力性能的通用基础模型，其核心创新在于通过语义保持的运动变换将DINO自监督学习范式适配到惯性数据上。

    

    我们提出了IMUDINO，这是第一个在多种传感任务上均能取得具有竞争力性能的惯性传感基础模型（FM），涵盖人体活动识别（HAR）、健康监测、动作捕捉和里程计测量等任务。以往的惯性基础模型通常只关注惯性传感的某一个子领域，有时甚至表现不如最先进的监督学习方法（例如在人体活动识别任务中），这使其适用范围远窄于视觉和自然语言领域的基础模型。此外，这些模型往往要求固定的输入长度和采样率，并且对未见过的传感器佩戴位置泛化能力较差。为了构建一个通用的惯性传感基础模型，我们将DINO自监督学习（SSL）范式适配到惯性数据上，通过一组语义保持的运动变换来模拟运动模式和传感器设置的变化。在模型架构方面，我们使用带有频率感知位置编码的普通transformer编码器，原生支持（原文在此处被截断）

    arXiv:2611.07437v1 Announce Type: cross  Abstract: We present IMUDINO, the first foundation model (FM) for inertial sensing to achieve competitive performance across diverse sensing tasks including human activity recognition (HAR), health, motion capture, and odometry. Prior inertial FMs often focus on one subfield of inertial sensing and sometimes underperform state-of-the-art supervised methods (e.g., in HAR), making their scope considerably narrower than FMs for vision and natural language. Moreover, they often require fixed input length and sampling rate, and generalize poorly to unseen sensor locations. To build a general-purpose foundation model for inertial sensing, we adapt the DINO self-supervised learning (SSL) paradigm to inertial data through a set of semantics-preserving motion transformations that simulate variations in motion pattern and sensor setup. For model architecture, we use a plain transformer encoder with frequency-aware positional encoding that natively support
    
[^146]: 截断的时间反向传播诱导双边梯度巡回性质

    Truncated backpropagation through time induces a two-sided gradient turnpike property

    [https://arxiv.org/abs/2611.07415](https://arxiv.org/abs/2611.07415)

    本文证明，在状态转移雅可比矩阵指数衰减等温和条件下，截断的时间反向传播具有“双边梯度巡回性质”，即截断与完整上下文的梯度贡献误差随远离两侧边界而指数衰减，从而为梯度对齐和近似完整上下文平稳性提供理论保证。

    

    截断的时间反向传播通过将梯度计算限制在较短的时间上下文中，降低了循环模型训练的计算负担。这会引发两种不同的边界效应：缺失过去的信息会扰动前向状态演化，而截断反向递推则会遗漏未来的信息。这促使我们将用于参数更新的梯度贡献的平均区间与其计算时所处的周围时间上下文解耦开来。在状态转移雅可比矩阵指数衰减及适度的正则性条件下，这导致了一个双边梯度巡回性质：截断上下文与完整上下文梯度贡献之间的误差随距离两侧上下文边界的距离呈指数衰减。我们为平均更新梯度推导了界，从而为梯度对齐性和近似完整上下文的平稳性提供了保证；在无偏……（摘要在此处截断）

    arXiv:2611.07415v1 Announce Type: cross  Abstract: Truncated backpropagation through time reduces the computational burden of training recurrent models by restricting gradient computations to shorter temporal contexts. This induces two distinct boundary effects: missing past information perturbs the forward state evolution, while truncating the backward recursion omits future information. This motivates decoupling the interval over which gradient contributions are averaged for the parameter update from the surrounding temporal context over which they are computed. Under exponential decay of the state-transition Jacobians and mild regularity conditions, this leads to a two-sided gradient turnpike property: the error between truncated- and full-context gradient contributions decays exponentially with the distance from both context boundaries. We derive bounds for the averaged update gradient, yielding guarantees on gradient alignment and approximate full-context stationarity; under unbia
    
[^147]: 等变表示中的谱对称性指纹

    Spectral symmetry fingerprints in equivariant representations

    [https://arxiv.org/abs/2611.07410](https://arxiv.org/abs/2611.07410)

    本文提出谱对称性指纹，将等变表示的奇异值谱作为结构对称性的统一指纹，实现了方向不变、架构无关、无参数的点群识别与连续对称性分析。

    

    对称性支配着物理结构的组织方式，而对称性破缺则揭示了它们的转变过程。在几何深度学习中，对称性对等变表示施加了秩约束，限制了等变图神经网络的表达能力。本文表明，这些约束反而可以被加以利用：等变表示的奇异值谱为结构对称性提供了一种统一的指纹。我们提出了谱对称性指纹，通过秩模板匹配实现离散点群识别，并通过量化超出目标秩约束的谱能量实现连续对称性分析。SymFP具有方向不变性和架构无关性，能够实现无参数的GNN实现。其谱可在多个目标群之间重复使用，且所得的泄漏分数满足子群单调性。在合成结构和分子（摘要在此处被截断）

    arXiv:2611.07410v1 Announce Type: new  Abstract: Symmetry governs the organization of physical structures, while symmetry breaking reveals their transformations. In geometric deep learning, symmetry imposes rank constraints on equivariant representations, limiting the expressive capacity of equivariant graph neural networks (GNNs). Here we show that these constraints can instead be exploited: the singular-value spectra of equivariant representations provide a unified fingerprint of structural symmetry. We introduce spectral symmetry fingerprints (SymFP), which enable discrete point-group identification through rank-template matching and continuous symmetry analysis by quantifying spectral energy beyond target rank constraints. SymFP is orientation invariant and architecture agnostic, enabling a parameter-free GNN implementation. Its spectra can be reused across multiple target groups, and the resulting leakage scores satisfy subgroup monotonicity. Across synthetic structures, molecules
    
[^148]: 达到近机器精度的非线性水波神经网络的严格认证计算

    Rigorously Certified Neural-Network Computation of Nonlinear Water Waves at Near-Machine Precision

    [https://arxiv.org/abs/2611.07395](https://arxiv.org/abs/2611.07395)

    本文提出将物理信息神经网络与严格的计算机辅助数值认证及形式化验证相结合的框架，在无训练数据条件下以近机器精度计算非线性重力-毛细行波，统一再现Stokes波、Crapper波与Wilton涟漪，并发现新的大幅值候选解。

    

    我们提出了一个计算框架，该框架将基于机器学习的发现方法与严格的计算机辅助数值认证和形式化验证相结合，用于求解强非线性偏微分方程。我们在周期性重力-毛细行波问题上演示了该方法。物理信息神经网络（PINN）被用作搜索机制，其中波形、波速以及延拓实验中的表面张力均直接从控制方程中学习得到，无需任何训练数据。通过分阶段的训练与精化过程，即使在强非线性区域，神经网络也能使控制方程的残差达到近机器精度。单一统一的表述同时再现了大幅值Stokes波、Crapper波和Wilton涟漪。从大幅值Wilton涟漪出发沿表面张力进行延拓，发现了新的大幅值候选解。我们将其中一个候选解固定为显式有限……

    arXiv:2611.07395v1 Announce Type: cross  Abstract: We present a computational framework that combines machine-learning-based discovery with rigorous computer-assisted numerical certification and formal verification for strongly nonlinear partial differential equations. We demonstrate the approach for periodic gravity-capillary traveling water waves. Physics-informed neural networks are used as a search mechanism, with the wave profile, wave speed and, in continuation experiments, surface tension learned directly from the governing equations without training data. Through a staged training and refinement procedure, the neural networks achieve near-machine-precision residuals of the governing equations, even in strongly nonlinear regimes. A single formulation recovers high-amplitude Stokes waves, Crapper waves and Wilton ripples. Continuation from a high-amplitude Wilton ripple in surface tension reveals new high-amplitude candidate solutions. We freeze one candidate as an explicit finit
    
[^149]: CacheGraft：通过潜空间文档增强来提升检索增强生成

    CacheGraft: Enhancing Retrieval-Augmented Generation via Latent-Space Document Enrichment

    [https://arxiv.org/abs/2611.07393](https://arxiv.org/abs/2611.07393)

    CacheGraft基于缓存到缓存（C2C）融合框架，将语料库段落离线编码为潜空间KV缓存表示，在查询时直接注入小模型，无需大模型参与即可增强检索增强生成（RAG）的答案质量。

    

    机器学习领域的最新研究表明，基于Transformer的语言模型代理不仅可以直接通过文本进行通信，还可以在潜空间中通过键值（KV）缓存直接进行通信。在本文中，我们提出这种缓存到缓存的翻译技术可以提供超越其实时代理通信这一最初应用场景的实用价值，并可用于检索增强生成（RAG）。基于缓存到缓存（C2C）融合框架，我们提出的系统CacheGraft将每个语料库段落离线编码为较大共享模型（实验中为4B参数）和较小接收模型（0.6B参数）之间的融合KV缓存表示，使得该缓存可以在查询时注入到接收模型中，而无需较大模型。在五个问答基准测试和两个独立本地运行的评估器上，CacheGraft在29个案例中（摘要在此处被截断）相比纯文本接收模型基线提升了答案质量。

    arXiv:2611.07393v1 Announce Type: cross  Abstract: Recent work in machine learning has demonstrated that transformer-based language model agents can communicate state not only through text, but also directly in latent space via key-value (KV) caches. In this paper, we argue that such cache-to-cache translation can provide utility beyond the original motivating application of real-time agent communication and can instead be utilized for retrieval-augmented generation (RAG). Building on the Cache-to-Cache (C2C) fusion framework, our proposed system CacheGraft encodes each corpus passage offline into a fused KV cache representation between a larger sharer model (4B parameters in our experiments) and a smaller receiver model (0.6B parameters) such that this cache can be injected into the receiver at query time with no need for the larger model. Across five QA benchmarks and two independent locally-run judges, CacheGraft improves answer quality over the text-only receiver baseline in 29 out
    
[^150]: 对抗性多重性：精心挑选的模型与信息披露的局限

    Adversarial Multiplicity: Cherry-Picked Models and the Limits of Disclosure

    [https://arxiv.org/abs/2611.07378](https://arxiv.org/abs/2611.07378)

    本文揭示了预测多重性可被恶意武器化——操纵者仅需微调模型（甚至在通过黑盒审计的情况下）就能在几乎不损失预测精度的前提下逆转任何个体预测结果，这从根本上挑战了基于信息披露的监管保障机制。

    

    预测多重性——即准确度相近的模型在个体预测上可能产生分歧——可以被用来在保持预测性能的前提下优化次要指标（例如公平性）。然而，对多重性的利用为有动机的一方（或通过投机取巧来达成目标的编程智能体）提供了将预测导向自身目的的机会，例如在涉及自动化决策工具的诉讼或消费者纠纷中。在这些对抗性多重性场景中，我们区分了软性操纵与硬性操纵：前者指各方在无害的模型选择范围内进行优化（例如超参数调优、特征工程），后者指各方直接扰动模型参数。我们发现，一个足够灵活的模型可以被操纵，在预测性能变化极小的情况下逆转任何个体预测，并能通过黑盒审计，这使得硬性操纵在大多数监管环境中成为可行的手段。软性操纵……

    arXiv:2611.07378v1 Announce Type: new  Abstract: Predictive multiplicity -- where similarly accurate models disagree on individual predictions -- can be leveraged to optimize secondary metrics (e.g., fairness) without loss in predictive performance. However, exploiting multiplicity opens the opportunity for motivated parties (or coding agents gaming their objectives) to steer predictions toward their own ends, such as in litigation or consumer disputes with automated decision-making tools. In these adversarial multiplicity settings, we distinguish soft manipulation, where parties optimize over innocuous model choices (e.g., hyperparameter tuning, feature engineering), from hard manipulation, where parties directly perturb model parameters. We find a sufficiently flexible model can be manipulated to reverse any individual prediction with minimal change in predictive performance while passing black-box audits, making hard manipulation viable in most regulatory settings. Soft manipulation
    
[^151]: 当分数与偏好冲突时：DPO中约束漂移的对齐前诊断与自适应校正

    When Scores Conflict with Preferences: Pre-Alignment Diagnosis and Adaptive Correction of Constraint Drift in DPO

    [https://arxiv.org/abs/2611.07375](https://arxiv.org/abs/2611.07375)

    该论文揭示了DPO训练中的系统性失效模式“约束漂移”，提出可从数据集统计量直接计算的闭式对齐前诊断指标 $\rho^{*}_{\mathrm{eq}}$，并据此构建冲突感知DPO（CoDPO），通过自适应降低冲突样本对的权重来校正约束漂移。

    

    直接偏好优化（DPO）越来越多地应用于成对偏好与逐点标准（如回答长度、答案正确性或安全分数）共存的场景。当两者不一致时，标准DPO会将那些偏好输出在逐点标准上得分更差的样本对（冲突对）与其余（匹配）样本对以相同的方式处理。我们刻画了一种系统性失效模式——约束漂移：对于足够小的更新，冲突对倾向于增加约束成本，而匹配对倾向于修复该成本，净变化取决于两者的平衡。这一平衡产生了一个闭式解的对齐前诊断指标 $\rho^{*}_{\mathrm{eq}}$，可直接从数据集统计量中计算得出，用于指示该数据集是可自我修复的还是需要干预。基于该诊断指标，我们提出了冲突感知DPO（CoDPO），它以由 $\rho^{*}_{\mathrm{eq}}$ 设定的强度对冲突对进行降权……

    arXiv:2611.07375v1 Announce Type: new  Abstract: Direct Preference Optimization (DPO) is increasingly used in settings where pairwise preferences coexist with pointwise criteria such as response length, answer correctness, or safety scores. When the two disagree, standard DPO treats pairs whose preferred output scores worse on the pointwise criterion (conflict pairs) in the same way as the remaining (matching) pairs. We characterize a systematic failure mode, constraint drift: for sufficiently small updates, conflict pairs tend to increase constraint cost while matching pairs tend to repair it, and the net change depends on their balance. This balance yields a closed-form pre-alignment diagnostic, $\rho^{*}_{\mathrm{eq}}$, computable from dataset statistics, that indicates whether a dataset is self-correcting or requires intervention. Building on this diagnostic, we propose Conflict-aware DPO (CoDPO), which down-weights conflict pairs with a strength set from $\rho^{*}_{\mathrm{eq}}$ r
    
[^152]: 测试时自我博弈：无需验证器的扩展方法

    Test-Time Self-Play: Scaling without a Verifier

    [https://arxiv.org/abs/2611.07367](https://arxiv.org/abs/2611.07367)

    该论文提出测试时自我博弈（TTSP），一种基于非对称自我博弈、无需验证器的测试时扩展方法，在无验证器的情况下于 LiveCodeBench v6 上实现了 mean@8 提升 35% 的绝对改进，且验证器越弱其优势越明显。

    

    测试时扩展方法在部署阶段投入额外的计算资源，以提升大语言模型在困难目标问题上的性能。大多数方法依赖于一个验证器来对候选解决方案的正确性进行评分，并利用它来选择或改进模型的输出。然而，在许多部署场景中，针对目标问题的强验证器往往是不可用的。我们提出了测试时自我博弈（Test-Time Self-Play, TTSP），这是一种利用非对称自我博弈、无需依赖验证器的测试时扩展方法。在 LiveCodeBench v6 的难度校准子集上，TTSP 在我们评估的每一种验证器强度下都是表现最佳的方法，并且随着验证器强度的减弱，它与最强基线之间的差距不断扩大；在无验证器的极限情况下，TTSP 在 mean@8 指标上带来了 35% 的绝对提升。这些收益在各种验证器强度下均能保持，除非基础策略对正确答案的覆盖率接近于零。

    arXiv:2611.07367v1 Announce Type: cross  Abstract: Test-time scaling methods spend additional compute at deployment to improve a large language model's performance on a hard target problem. Most methods rely on a verifier that scores candidate solutions for correctness, using it to select among or refine the model's outputs. In many deployment regimes, however, a strong verifier on the target problem is unavailable. We introduce Test-Time Self-Play (TTSP), an approach to test-time scaling using asymmetric self-play without reliance on a verifier. On a difficulty-calibrated subset of LiveCodeBench v6, TTSP is the best-performing method at every verifier strength we evaluate, and the gap to the strongest baseline widens as the verifier weakens; in the no-verifier limit, TTSP yields a 35\% absolute improvement in mean@8. The gains hold across a range of verifier strengths, except when the base policy has near-zero coverage of the correct answer.
    
[^153]: 渐近保持张量神经网络求解三维高场动理学方程及其在不确定性量化中的应用

    Asymptotic-Preserving Tensor Neural Networks for 3D High-Field Kinetic Equations with Applications to Uncertainty Quantification

    [https://arxiv.org/abs/2611.07364](https://arxiv.org/abs/2611.07364)

    本文提出渐近保持张量神经网络（APTNN）方法，利用CP分解将高维损失积分化为一维积分的组合，从而高效求解三维高场动理学方程并应用于不确定性量化。

    

    高场区的动理学方程描述了以电场加速和碰撞弛豫为主导的粒子输运过程。这类方程被广泛应用于等离子体和半导体器件的模拟中，典型模型包括Vlasov–Poisson–Fokker–Planck（VPFP）方程组和Boltzmann–Poisson方程组。其高维性与多尺度特性是精确且高效数值求解面临的两大主要挑战。为应对这些挑战，我们提出了一种渐近保持张量神经网络（APTNN）方法。具体而言，为降低高维计算成本与内存需求，我们采用张量神经网络（TNN）的典范多线性（CP）分解结构来逼近方程的解。我们证明方程残差可保持这种分解形式，从而使得高维损失积分能够通过一维积分的组合来计算，由此带来的计算与存储方面的优势……（摘要原文在此处被截断）

    arXiv:2611.07364v1 Announce Type: cross  Abstract: Kinetic equations in the high-field regime describe particle transport dominated by electric-field acceleration and collisional relaxation. These equations are widely used in simulations of plasmas and semiconductor devices. Typical models include the Vlasov--Poisson--Fokker--Planck (VPFP) and Boltzmann--Poisson systems. Their high dimensionality and multiscale nature pose two of the major challenges to accurate and efficient numerical solution. To address these challenges, we propose an asymptotic-preserving tensor neural network (APTNN) method. Specifically, to reduce high-dimensional computational costs and memory requirements, we adopt the canonical polyadic (CP) decomposition structure of TNNs to approximate their solutions. We show that the equation residuals retain this decomposition form, allowing high-dimensional loss integrals to be evaluated through combinations of one-dimensional integrals. The resulting computational and s
    
[^154]: 从“已用”到“缺失”：重新思考 LoRA 中的动态秩

    From What Is Used to What Is Missing: Rethinking Dynamic Rank in LoRA

    [https://arxiv.org/abs/2611.07349](https://arxiv.org/abs/2611.07349)

    该论文从几何视角重新思考 LoRA 的动态秩，揭示因子梯度的不可观测子空间与低秩流形法空间重合，并提出利用主要法向奇异方向及其谱能量来确定局部最优的秩扩展，从而判断当前适应空间是否满足任务需求。

    

    基于 LoRA 的参数高效微调由秩主导，秩控制着参数预算以及局部可访问的任务更新。现有的动态秩方法主要关注容量分配，而未解决一个关键问题：当前的适应空间是否足以完成任务。我们刻画了标准因子梯度的一阶信息边界，证明其不可观测子空间恰好与低秩流形的法空间（normal space）重合。因此，仅凭因子梯度无法区分当前空间内的欠优化与超出其一阶可达范围的任务相关方向。这启发了一种理解秩演化的几何视角：切向信息用于优化现有的自由度，而法向谱则量化引入新自由度的价值。在局部二次模型下，我们证明主要的法向奇异方向能够给出局部最优的秩扩展，且谱能量决定了扩展的……

    arXiv:2611.07349v1 Announce Type: cross  Abstract: Parameter-efficient fine-tuning with LoRA is governed by rank, which controls parameter budget and locally accessible task updates. Existing dynamic-rank methods mainly focus on capacity allocation, leaving unresolved whether the current adaptation space is sufficient for the task. We characterize the first-order information boundary of standard factor gradients, showing that their unobservable subspace coincides with the normal space of the low-rank manifold. Thus, factor gradients alone cannot distinguish under-optimization within the current space from task-relevant directions beyond its first-order reach. This motivates a geometric view of rank evolution, where tangent information optimizes existing degrees of freedom and the normal spectrum quantifies the value of new ones. Under a local quadratic model, we show that leading normal singular directions yield the locally optimal rank expansion, with spectral energy determining its g
    
[^155]: AtlasULP：基于关系图谱的领域感知通用链接预测

    AtlasULP: Domain-aware Universal Link Prediction via Relation Atlas

    [https://arxiv.org/abs/2611.07347](https://arxiv.org/abs/2611.07347)

    该论文提出AtlasULP，通过关系图谱捕获高阶信息并实现领域感知能力，使模型能够在无需训练的情况下对任意未见图进行推理，克服了现有通用链接预测方法无法捕捉长距离依赖和泛化受限的问题。

    

    链接预测（LP）是图学习中一项被广泛应用的任务。传统链接预测方法通常遵循特定于数据集的范式，需要对每个图进行独立训练，在大规模应用中导致高昂的计算和维护成本。受这些局限性的启发，近期工作探索了通用链接预测（ULP），旨在实现对任意未见过的图进行无需训练的推理。然而，现有方法主要依赖子图采样策略来构建通用表示，由于高阶邻域的指数级增长，这些方法无法捕获高阶信息。因此，它们在远程信号至关重要的图上往往表现不佳，例如生物网络。此外，其预测模型通常是领域无关的，缺乏适应不同图领域的能力，导致泛化能力有限。为了解决这些挑战……

    arXiv:2611.07347v1 Announce Type: new  Abstract: Link prediction (LP) is a widely applied task in graph learning. Conventional LP approaches typically follow a dataset-specific paradigm, requiring independent training for each graph and incurring high computational and maintenance costs in large-scale applications. Motivated by these limitations, recent work explores Universal Link Prediction (ULP), aiming to enable training-free inference on arbitrary unseen graphs. However, existing methods primarily rely on subgraph sampling strategies to construct universal representations, which are unable to capture higher-order information due to the exponential growth of higher-order neighborhoods. As a result, they often under-perform on graphs where long-range signals are essential, such as biological networks. Moreover, their prediction models are typically domain-agnostic and lack the ability to adapt to different graph domains, leading to limited generalization. To address these challenges
    
[^156]: 无界问题下随机梯度方法的末次迭代收敛性

    Last-Iterate Convergence of Stochastic Gradient Methods for Unbounded Problems

    [https://arxiv.org/abs/2611.07334](https://arxiv.org/abs/2611.07334)

    本文在去除了有界定义域和有界方差假设的条件下，证明了随机梯度方法的末次迭代具有接近最优的 $\widetilde{\mathcal{O}}(1/\sqrt{T})$ 收敛率，并在迭代总次数已知时达到了最优的 $\mathcal{O}(1/\sqrt{T})$ 阶，与强假设下的已知结果相匹配。

    

    随机梯度方法的末次迭代收敛保证近来受到广泛关注，因为在实际应用中算法通常返回最后一次迭代的结果，而非所有迭代的平均值。本文研究了两种分析近端（proximal）和投影（projected）随机梯度方法末次迭代收敛性的途径，且无需有界定义域或有界方差假设，其中黑箱变分推断作为贯穿全文的应用实例。在两个相关的、放宽有界方差要求的假设条件下，我们证明了末次迭代具有接近最优的任意时刻收敛率 $\widetilde{\mathcal{O}}(1/\sqrt{T})$。此外，当总迭代次数 $T$ 已知时，我们进一步去除了对数因子，得到了最优阶 $\mathcal{O}(1/\sqrt{T})$，与有界方差或有界定义域假设下已知的收敛阶相匹配。

    arXiv:2611.07334v1 Announce Type: cross  Abstract: Last-iterate convergence guarantees for stochastic gradient methods have attracted significant attention as practical implementations typically return the last iterate instead of an average of the iterates. In this paper, we study two approaches for analyzing the last-iterate convergence of proximal and projected stochastic gradient methods without bounded domain or bounded variance assumptions, where black-box variational inference serves as a running example. Under two related assumptions relaxing the bounded variance, we show near-optimal anytime rate $\widetilde{\mathcal{O}}(1/\sqrt{T})$ for the last iterate. Moreover, when the horizon $T$ is known, we remove the logarithmic factor and obtain the optimal order $\mathcal{O}(1/\sqrt{T})$, matching the order known under bounded variance or bounded domain assumptions.
    
[^157]: 持续强化学习中的永久性与瞬态表示

    Permanent and Transient Representations for Continual Reinforcement Learning

    [https://arxiv.org/abs/2611.07327](https://arxiv.org/abs/2611.07327)

    本文提出在持续强化学习的互补学习系统中为长期学习采用参数化表示、为瞬态适应采用非参数化表示，并引入一种新颖的非参数近似方法来估计瞬态价值函数，从而实现复杂任务中的在线学习并优于竞争性基线方法。

    

    持续强化学习（CRL）智能体难以在适应新情况的同时保留先前学到的知识，从而面临稳定性与可塑性之间的权衡。解决这一问题的一种方法是采用互补学习系统：一个用于缓慢的长期学习，另一个用于快速的瞬态适应。Anand & Precup（2023）通过永久价值函数和瞬态价值函数实现了这一思想，但两个系统使用了相同的特征。直观来看，长期学习应更多依赖参数化表示以实现广泛的泛化能力，而瞬态学习则最适合采用非参数化表示以实现针对特定情境的适应。在本文中，我们从理论和实证两方面对这一思想进行了探索。通过引入一种新颖的非参数近似方法来估计复杂任务中的瞬态价值函数，我们证明了我们的方法能够实现在线学习，并且优于竞争性基线方法。

    arXiv:2611.07327v1 Announce Type: cross  Abstract: Continual Reinforcement Learning (CRL) agents struggle to adapt to new situations while retaining prior knowledge, leading to a stability-plasticity trade-off. One approach to solving this problem is by using complementary learning systems: one for slow, long-term learning and another for quick, transient adaptation. Anand & Precup (2023) instantiated this idea using permanent and transient value functions, but used the same features in both systems. Intuitively, long-term learning should rely more on parametric representations for broad generalization, while transient learning would be best served by non-parametric representations for situation-specific adaptation. In this paper, we explore this idea both theoretically and empirically. By introducing a novel non-parametric approximation for estimating the transient value function in complex tasks, we demonstrate that our method enables online learning and outperforms competitive basel
    
[^158]: 重新思考数据选择：在LLM微调中覆盖度比特难度更重要

    Rethinking Data Selection: The Importance of Coverage over Difficulty in LLM Fine-Tuning

    [https://arxiv.org/abs/2611.07323](https://arxiv.org/abs/2611.07323)

    该研究发现在LLM微调的数据选择中，数据对输入分布的覆盖度比样本难度更重要，并提出一种简单的基于聚类的选择方法，在单领域任务上与随机选择持平，在多领域数据集上优于随机选择。

    

    选择高质量的训练数据可以降低LLM微调的计算成本。以往的数据选择研究基于“更难的样本信息量更大”这一直觉，开发了多种基于难度的评分方法。虽然这些方法对较小的分类模型有效，但我们发现它们在LLM微调中往往无法超越随机选择。我们的分析揭示，随机选择表现更好的原因是它更能代表潜在的输入空间——即具有更好的覆盖度。为了验证覆盖度的重要性，我们实现了一种简单的基于聚类的选择方法，该方法选择更能代表潜在输入分布的数据。在一系列多样化的单领域任务上，该方法达到了与随机选择相当或更好的性能。在多样化、多领域的Tülu 3混合数据集上（此时随机采样不再能保证子集的均衡性），我们的方法优于随机选择。

    arXiv:2611.07323v1 Announce Type: new  Abstract: Selecting high-quality training data can reduce computation cost for LLM fine-tuning. Prior work on data selection developed a variety of difficulty-based scores, motivated by the intuition that harder examples are more informative. While effective for smaller classification models, we find that such methods often fail to outperform random selection for LLM fine-tuning. Our analysis reveals that random selection performs better because it is more representative of the underlying input space---i.e., has better coverage. To validate the importance of coverage, we implement a simple clustering-based selection method which selects data that is more representative of the underlying input distribution. On a set of diverse single-domain tasks, this method yields performance at or above that of random selection. On the diverse, multi-domain T\"{u}lu 3 mixture, where random sampling no longer guarantees a balanced subset, our method outperforms a
    
[^159]: WeightLoop：基于实验反馈的权重空间编程

    WeightLoop: Weight-Space Programming with Experimental Feedback

    [https://arxiv.org/abs/2611.07319](https://arxiv.org/abs/2611.07319)

    WeightLoop框架让语言模型编写固定的可复用参数编辑程序，将实验反馈转化为对已部署预测器的持久权重更新，无需额外的LLM调用或梯度重训即可实现模型的反复改进。

    

    权重空间编程（WSP）表明语言模型能够对神经网络权重进行推理和重写；递归自我改进（RSI）激发了反馈驱动的改进。指导实验的模型会收到具体的反馈：已完成的实验分析揭示了预测误差，并为下一步测试什么提供指导。我们提出了WeightLoop，一个将实验反馈转化为对已部署预测器进行持久更新的框架。GPT-6 Astra检查任务规范和模型实现，然后编写一个有界的、可复用的参数编辑程序。随着新的测量数据到来，该程序识别预测差异，利用观测到的结果验证候选编辑，并将选定的更改写回原始模型。编辑器保持固定而预测器持续演进，从而无需进一步的LLM调用或梯度重训即可实现反复更新，同时保持原始的推理接口。在离线回放中，目标（摘要在此处截断）

    arXiv:2611.07319v1 Announce Type: new  Abstract: Weight-space programming (WSP) shows that language models can reason about and rewrite neural network weights; recursive self-improvement (RSI) motivates feedback-driven improvement. Models guiding experiments receive concrete feedback: completed assays reveal prediction errors and inform what to test next. We introduce WeightLoop, a framework that turns experimental feedback into persistent updates to deployed predictors. GPT-6 Astra inspects a task specification and the model implementation, then writes a bounded, reusable parameter-editing program. As new measurements arrive, the program identifies prediction discrepancies, validates candidate edits using observed outcomes, and writes the selected changes back to the original model. The editor remains fixed while the predictor evolves, enabling repeated updates without further LLM calls or gradient retraining while preserving the original inference interface. In offline replay, target
    
[^160]: 低秩约束马尔可夫决策过程的可证明高效表示学习

    Provably Efficient Representation Learning for Low-Rank CMDPs

    [https://arxiv.org/abs/2611.07295](https://arxiv.org/abs/2611.07295)

    该论文提出了首个结合LSVI-UCB与原始-对偶技术的无模型表示学习算法MFRLC，能够可证明地高效求解低秩约束马尔可夫决策过程，在有效平衡奖励遗憾与约束违反的同时学习未知表示。

    

    我们研究了低秩约束马尔可夫决策过程（CMDP）中的表示学习问题，其中价值函数和奖励函数均由一组未知的表示向量来近似，且状态转移动力学具有低秩分解结构。目标是在满足期望累积效用约束的前提下，最大化期望累积奖励。为此，我们提出了MFRLC（面向低秩CMDP的无模型表示学习），这是一种无模型算法，采用原始-对偶机制来有效平衡奖励遗憾与约束违反。据我们所知，MFRLC是首个将最小二乘价值迭代与置信上界（LSVI-UCB）技术及原始-对偶方法相结合的低秩CMDP表示学习方法。MFRLC还通过函数近似的奖励项进一步增强价值函数估计，并学习从表示（原文在此处截断）……

    arXiv:2611.07295v1 Announce Type: new  Abstract: We study representation learning in low-rank Constrained Markov Decision Processes (CMDPs), where both value and reward functions are approximated by a set of unknown representation vectors, and the transition dynamics admit a low-rank factorization. The objective is to maximize expected cumulative rewards while complying with constraints on the expected cumulative utility. To achieve this, we propose \texttt{MFRLC} (Model-Free Representation Learning for Low-Rank CMDP), a model-free algorithm that uses a primal-dual scheme to effectively balance reward regret and constraint violations. To the best of our knowledge, \texttt{MFRLC} is the first representation-learning method for low-rank CMDPs that combines the Least-Squares Value Iteration with Upper Confidence Bound (LSVI-UCB) and primal-dual techniques. \texttt{MFRLC} further enhances value function estimation with bonuses for function approximation and learns the mapping from represen
    
[^161]: Transformer如何在上下文中执行低层控制？

    How do Transformers Perform Low-Level Control In-Context?

    [https://arxiv.org/abs/2611.07291](https://arxiv.org/abs/2611.07291)

    该论文研究了GPT-2风格的Transformer如何仅凭无外力的系统演化轨迹作为上下文，在不掌握具体参数的情况下实现对未知参数的不稳定非线性系统（小车倒立摆与Acrobot）的上下文学习式控制。

    

    arXiv:2611.07291v1 公告类型： new 摘要：基于Transformer的神经网络最引人注目的能力之一是上下文学习。这促使我们思考“上下文中的控制”：能否在仅知道（1）系统属于某一特定系统族、（2）系统演化的部分数据（以上下文轨迹的形式提供），而没有任何具体参数信息的情况下控制一个系统？具体而言，我们研究了Transformer模型对不稳定非线性动力系统执行“上下文中”控制的能力，其中上下文仅由无外力作用下的系统动力学轨迹组成。我们采用多头、GPT-2风格的架构，并在两个经典非线性系统——小车倒立摆系统和两连杆Acrobot系统上评估性能，且系统参数对Transformer是未知的。我们考虑了这样一种设定：在训练阶段可以使用来自专家控制器的轨迹，但在推理阶段无法获得。在此设定下，我们识别出了能够……（摘要在此处被截断）

    arXiv:2611.07291v1 Announce Type: new  Abstract: One of the remarkable abilities of Transformer-based neural networks is in-context learning. This motivates us to think about "in-context" control. Can we control a system knowing only (1) that it belongs to a particular family of systems and (2) some data on system evolution (provided as the context trajectory) but no specific parameter information? Specifically, we investigate the ability of Transformer-models to perform "in-context" control of unstable, nonlinear dynamical systems, where context consists only of the unforced system dynamics. We leverage a multi-head, GPT-2-style architecture, and we evaluate performance for two classical nonlinear systems -- the cartpole system and the two-link acrobot system, where the system parameters are unknown to the Transformer. We consider a setting where trajectories from an expert controller are available at training time but not at inference time. Here, we identify training techniques that 
    
[^162]: 面向通感一体化的图神经网络资源分配方法

    Graph Neural Networks for Resource Allocation in Integrated Sensing and Communication

    [https://arxiv.org/abs/2611.07277](https://arxiv.org/abs/2611.07277)

    本文提出基于图神经网络的功率分配算法GNN-ISAC，通过在图中区分服务提供者间干扰与提供者内功率预算共享两类边，并以感知惩罚的和速率损失进行无标签训练，在满足感知SINR约束下高效最大化车辆通感一体化网络的通信和速率。

    

    在车辆通感一体化网络中，每个服务提供车辆需在有限的功率预算下同时服务多个感知目标和通信接收端。在满足感知信号与干扰加噪声比（SINR）约束的条件下最大化通信和速率是一个非凸问题，且针对每种网络布局重新运行迭代优化算法的代价十分高昂。本文提出了一种基于图神经网络的功率分配算法，称为GNN-ISAC。具体而言，GNN-ISAC通过在图表示中考虑两种不同的边类型来解决上述问题：一种边用于刻画服务提供者之间的相互干扰，另一种边用于表示服务提供者内部的功率预算共享。GNN-ISAC采用感知惩罚的和速率损失进行无标签训练。此外，本文还通过引入感知功率预留步骤，将加权均方误差最小化（WMMSE）算法适配到该问题上。仿真……

    arXiv:2611.07277v1 Announce Type: new  Abstract: In vehicular integrated sensing and communication (ISAC) networks, each service-provider vehicle uses a limited power budget to serve multiple sensing targets and communication receivers. Maximizing the communication sum-rate subject to sensing signal-to-interference-plus-noise ratio (SINR) constraints is nonconvex, and iterative optimization is costly to rerun for every network layout. This paper proposes a graph neural network (GNN)-based power allocation algorithm, termed GNN-ISAC. Concretely, GNN-ISAC addresses the aforementioned problem by considering two distinct edge types in the graph representation, one capturing the inter-provider interference, the other representing the intra-provider power-budget sharing. GNN-ISAC is trained without labels using a sensing-penalized sum-rate loss. We also adapt the weighted mean-square error minimization (WMMSE) algorithm to this problem by introducing a sensing-power reservation step. Simulat
    
[^163]: 满足化作为竞争性贝叶斯加权：未达成目标下目标导向选择的最小模型

    Satisficing as competitive Bayesian weighting: a minimal model of goal-directed choice under an unattained target

    [https://arxiv.org/abs/2611.07270](https://arxiv.org/abs/2611.07270)

    该论文证明“风险敏感满足化”这一多臂老虎机学习规则可等价改写为类贝叶斯估计器“注意力竞争贝叶斯（ACB）”——未达成的抱负水平充当所有动作的先验均值，有限试验预算决定证据覆盖先验的强度，从而给出了以未实现目标为参照的目标导向选择的最小形式化模型。

    

    动物和人类往往并非进行最大化决策；它们相对于尚未实现的目标（抱负水平）而行动。一个尚未实现的目标如何组织选择行为？Deacon（2012）将这种以“缺失”为参照的组织方式称为“意向性的”（ententional），而 García-Valdecasas 与 Deacon（2024）则主张，目的最好被理解为约束而非特定状态。我们在一个最小化的形式化环境中考察这一思想：风险敏感满足化（RS），一种多臂老虎机学习规则，它以每个动作过去被试行的份额乘以其相对于抱负水平的盈余或缺口来评估该动作的价值。我们证明 RS 可以被改写为一个类贝叶斯估计器——注意力竞争贝叶斯（ACB），其中抱负水平充当每个动作的先验均值，而有限的试验预算决定了证据能在多大程度上覆盖这一先验。由此得出两个结论。第一，不再被采样的动作会被部分覆盖……（原文摘要至此截断）

    arXiv:2611.07270v1 Announce Type: cross  Abstract: Animals and humans often do not maximize; they act relative to goals (aspiration levels) that are frequently not yet met. How can a goal that is not (yet) realized organize choice? Deacon (2012) calls such absence-referenced organization \emph{ententional}, and Garc\'{i}a-Valdecasas and Deacon (2024) argue that ends are best understood as constraints rather than as particular states. We examine this idea in a minimal formal setting: Risk-sensitive Satisficing (RS), a bandit learning rule that values each action by its share of past trials multiplied by its surplus or shortfall relative to an aspiration level. We show that RS can be rewritten as a Bayesian-like estimator, Attention Competitive Bayes (ACB), in which the aspiration level is the prior mean of every action and the finite budget of trials sets how strongly evidence can override it. Two consequences follow. First, actions that are no longer sampled are partially overridden by
    
[^164]: 信号有多深？初级保健EHR模型中利用长达30年时间上下文的结直肠癌预测

    How Deep is the Signal? Time-Context up to 30 Years for Colorectal Cancer Prediction in a Primary-Care EHR Model

    [https://arxiv.org/abs/2611.07265](https://arxiv.org/abs/2611.07265)

    该研究首次系统评估了EHR基础模型的“时间上下文”对疾病预测的影响，发现将日历回溯窗口从2年延长至最长30年可显著提升结直肠癌的早期预测性能。

    

    EHR基础模型在固定的上下文窗口内读取医疗事件，而迄今为止长上下文仅以token数量来衡量。模型需要回溯多远的时间才能捕捉到最早的预测信号，目前尚不清楚。我们提出了“时间上下文”（time-context）这一概念，即预测点之前的日历回溯窗口，并据我们所知，首次对EHR基础模型进行了十年尺度时间上下文的系统性评估。我们在来自伦敦西北部360万名初级保健患者的14亿条编码事件上预训练了一个仅解码器Transformer模型（3460万参数），并对其进行微调，以在12个月诊断前缓冲期的条件下预测新发结直肠癌。在一个由1,008例病例和9,582名匹配对照组成的固定队列中（所有患者均注册至少30年），我们在从2年到30年的15个嵌套回溯窗口上进行了评估，使得每次截断都在同一批患者内进行比较。判别性能从2年回溯时的AUROC 0.551提升至……

    arXiv:2611.07265v1 Announce Type: cross  Abstract: EHR foundation models read medical events within a fixed context window, and long context has so far been measured only in tokens. How far back in time a model must look to capture the earliest predictive signal is unknown. We introduce time-context, the calendar lookback window before the prediction point, and provide, to our knowledge, the first systematic evaluation of decade-scale time-context for an EHR foundation model. We pretrained a decoder-only transformer (34.6M parameters) on 1.4B coded events from 3.6M primary-care patients in North West London and fine-tuned it to predict incident colorectal cancer under a 12-month pre-diagnostic buffer. A frozen cohort of 1,008 cases and 9,582 matched controls, all registered for at least 30 years, was evaluated at 15 nested lookback windows from 2 to 30 years, so that every truncation is compared within the same patients. Discrimination rose from an AUROC of 0.551 at 2 years to 0.590 at
    
[^165]: 资源受限嵌入式系统中TinyML的能耗测量与优化：一项系统性映射研究

    Energy Measurement and Optimization of TinyML in Resource-Constrained Embedded Systems: A Systematic Mapping Study

    [https://arxiv.org/abs/2611.07264](https://arxiv.org/abs/2611.07264)

    该研究通过系统性映射方法全面梳理了能耗感知的TinyML研究领域，构建了涵盖主要模型与学习方法、能耗评估方法与指标、技术影响因素以及优化技术的结构化分类体系。

    

    物联网的快速扩张增加了网络边缘产生的数据量，而许多设备在严苛的内存、处理能力和能耗限制下运行。在此背景下，微型机器学习使机器学习（ML）任务能够在微控制器、智能传感器和其他资源受限平台上执行。然而，现有文献在所评估的算法、能耗测量方法、报告的指标、影响因素以及优化策略方面仍呈现异质性。本研究对能耗感知的TinyML研究进行了系统性映射，识别了主要的模型和学习方法，对能耗评估方法和指标进行了分类，考察了影响能耗的技术因素，并将已报道的优化技术整理为结构化的分类体系。研究结果凸显了卷积神经网络和轻量化（模型）的频繁使用……

    arXiv:2611.07264v1 Announce Type: new  Abstract: The rapid expansion of the Internet of Things (IoT) has increased the volume of data generated at the network edge, where many devices operate under severe memory, processing, and energy constraints. In this context, Tiny Machine Learning (TinyML) enables machine learning (ML) tasks to be executed on microcontrollers, smart sensors, and other resource-constrained platforms. However, the literature remains heterogeneous regarding the algorithms evaluated, energy-measurement methods, reported metrics, influencing factors, and optimization strategies. This study presents a systematic mapping of energy-aware TinyML research. It identifies the main models and learning approaches, classifies energy-evaluation methodologies and metrics, examines the technical factors affecting energy consumption, and organizes the reported optimization techniques into a structured taxonomy. The findings highlight the frequent use of convolutional and lightweigh
    
[^166]: 进化自适应邻域学习：一种用于最近邻分类中实例依赖k优化的遗传算法

    Evolutionary Adaptive Neighborhood Learning: A Genetic Algorithm for Instance-Dependent k Optimization in Nearest Neighbor Classification

    [https://arxiv.org/abs/2611.07257](https://arxiv.org/abs/2611.07257)

    该论文提出EANK方法，通过遗传算法优化一个低维策略，根据局部邻域特征（密度、标签熵、投票边际、集中度）为每个实例自适应地确定k近邻分类器的邻居数量k(x)，突破了传统全局单一k值选择的局限性。

    

    邻居数量k是k近邻（k-NN）分类器的核心超参数，通常通过交叉验证或进化搜索进行全局选择。然而，偏差-方差权衡本质上是局部性的：密集且无噪声的区域可能适合较小的邻域，而稀疏或有噪声的区域则可能需要更大的邻域。由于全局k优化是一维问题，用遗传算法替代穷举搜索几乎没有计算优势。因此，我们转而优化实例依赖的邻域策略。我们提出了EANK（进化自适应邻域K），它使用一个低维策略将局部邻域描述符（包括密度、标签熵、投票边际和集中度）映射到自适应的邻居数量k(x)。一个实数编码的精英遗传算法使用带邻域平滑性惩罚的留一法准确率来优化该策略。

    arXiv:2611.07257v1 Announce Type: cross  Abstract: The number of neighbours k is the central hyperparameter of the k-nearest neighbour (k-NN) classifier and is usually selected globally through cross-validation or evolutionary search. However, the bias-variance trade-off is inherently local: dense and noise-free regions may favour small neighbourhoods, whereas sparse or noisy regions may require larger ones. Since global-k optimization is one-dimensional, replacing exhaustive search with a genetic algorithm offers little computational advantage. We therefore optimize an instance-dependent neighbourhood policy instead. We propose EANK (Evolutionary Adaptive Neighborhood K), which uses a low-dimensional policy to map local neighbourhood descriptors, including density, label entropy, vote margin, and concentration, to an adaptive number of neighbours k(x). A real-coded elitist genetic algorithm optimizes this policy using leave-one-out accuracy with a neighbourhood-smoothness penalty. Pre
    
[^167]: TriAttention：面向长上下文LLM推理的统一稀疏混合精度注意力机制

    TriAttention: Unified Sparse Mixed-Precision Attention for Long-Context LLM Inference

    [https://arxiv.org/abs/2611.07256](https://arxiv.org/abs/2611.07256)

    TriAttention提出了一种统一的稀疏混合精度注意力方法，通过离线校准为每个注意力头在跳过、4位和8位计算之间分配计算区域，在共享计算预算下兼顾稀疏性与量化精度，从而优化长上下文LLM推理的成本与精度平衡。

    

    稀疏化和量化通过跳过部分交互以及降低所保留交互的精度来降低注意力机制的计算成本。然而，有用的交互对精度的需求各不相同：有些交互能够容忍低精度评分，但对被移除却十分敏感。因此，我们提出了TriAttention，它将查询-键（QK）分块分配给跳过、4位或8位三种计算方式，利用4位计算以更低的成本保留那些可容忍量化的交互。为了实现紧凑表示，我们使用两个对角距离边界来编码每个注意力头的计算区域，使策略存储与上下文长度无关。随后，离线校准过程使用归一化投影输出误差和完整分块执行成本来评估候选边界对。基于这些测量结果，一个模型级分配器在共享计算预算下为每个注意力头选择边界，以最小化校准误差。在预填充阶段，所选边界会生成精确……（摘要原文在此处截断）

    arXiv:2611.07256v1 Announce Type: cross  Abstract: Sparsity and quantization reduce the cost of attention by skipping interactions and lowering the precision of those retained. However, useful interactions differ in their precision requirements: some tolerate low-precision scores while remaining sensitive to removal. We therefore introduce TriAttention, which assigns query-key (QK) tiles to skipping, 4-bit, or 8-bit computation, using 4-bit to retain quantization-tolerant interactions at lower cost. To represent compactly, we encode each head's computation regions with two diagonal-distance boundaries, keeping policy storage independent of context length. Offline calibration then evaluates candidate boundary pairs using normalized projected-output errors and complete-tile execution costs. Using these measurements, a model-wide allocator selects each head's boundaries to minimize calibration error under a shared computation budget. During prefill, the selected boundaries generate precis
    
[^168]: 使用Transformer通过神经人工生命智能体的符号蒸馏实现教与学

    Using Transformers for Teaching and Learning via Symbolic Distillation of Neural A-life Agents

    [https://arxiv.org/abs/2611.07254](https://arxiv.org/abs/2611.07254)

    该论文提出使用Transformer模型替代传统高昂的随机搜索方法，将不透明的神经人工生命智能体策略蒸馏为紧凑可解释的符号策略，从而降低计算成本，并促进智能体之间的知识共享与种群级学习。

    

    控制人工生命智能体行为的神经网络可以在多代中不断演化。虽然这些网络在使智能体得以生存方面非常有效，但对人类观察者来说通常是不透明的。此外，神经权重无法在不同神经架构的智能体之间进行转移。将此类神经网络蒸馏为紧凑的符号策略，可以帮助人类理解人工生命中智能体的行为。此外，这还使智能体能够以紧凑的方式共享知识，从而改善种群层面的学习。将神经策略蒸馏为符号策略通常是通过在符号策略空间中搜索，找到一个能够近似神经策略的符号策略。然而，即使使用领域特定语言，策略空间仍然可能非常庞大，因此需要采用不完全的随机搜索方法（如遗传算法），而且蒸馏过程仍然计算成本高昂。我们提出改为使用Transformer……（原文摘要在此处截断）

    arXiv:2611.07254v1 Announce Type: new  Abstract: Neural networks controlling behavior of A-life agents can evolve over multiple generations. While effective in making the agents survive, such networks are generally opaque to a human observer. Furthermore, the neural weights cannot be transferred between agents with different neural architectures. Distilling such neural networks into compact symbolic policies can help humans understand agent behavior in A-life. Additionally, it allows the agents to share their knowledge in a compact fashion, improving population-level learning. Distilling a neural policy into a symbolic policy is typically performed by searching the space of symbolic policies to find one that approximates the neural policy. However, even with a domain-specific language the space of policies can be large, necessitating incomplete stochastic search such as genetic algorithms. Yet the distillation remains computationally expensive. We propose to instead use a transformer p
    
[^169]: 目标揭示：连接信息论赋权与基于能力的人类权力的桥梁

    Goal Revelation as a Bridge Between Information-Theoretic Empowerment and Capability-Based Human Power

    [https://arxiv.org/abs/2611.07247](https://arxiv.org/abs/2611.07247)

    该论文提出“目标揭示”（即人类潜在目标与实际轨迹之间的互信息）作为统一两种权力度量的桥梁：基于能力的个体权力通过Fano不等式构成其下界，而信息论赋权构成其上界，从而为AI安全中的权力量化研究建立了理论联系。

    

    权力对AI安全至关重要。最优智能体往往会在广泛的目标上获取权力（即工具性趋同）。随着我们将AI智能体嵌入社会技术系统，人类面临着逐渐失权的风险。这两种担忧都呼唤一种形式化、定量化的权力度量。在马尔可夫博弈的背景下，存在两种这样的度量：其一是信息论赋权，即从智能体动作到未来状态的信道容量；其二是基于能力的个体权力，这是一种由“人类在追求候选目标族中的每个目标时达成该目标的概率”所构建的目标达成度量。我们展示了这两种度量在何处交汇。从个体权力出发，Fano不等式给出了一个下界 $B(s)$。该下界是我们所称的“目标揭示”的下界——目标揭示即人类潜在目标与实际实现轨迹之间的互信息 $\mathbb{I}(G;\xi \mid s)$。而赋权则从上方对目标揭示加以约束，使得 $B(s) \le \mathbb{I}$……（原文摘要至此截断）

    arXiv:2611.07247v1 Announce Type: new  Abstract: Power matters for AI safety. Optimal agents tend to acquire power across a wide range of objectives (instrumental convergence). As we embed AI agents into socio-technical systems, humans risk gradual disempowerment. Both concerns call for a formal, quantitative measure of power. Two such measures in the context of Markov games are information-theoretic empowerment, the capacity of the channel from an agent's actions to future states, and capability-based individual power, a measure of goal attainment built from the probabilities that a human would attain each goal in a family of candidates were they pursuing it. We show where the two meet. From individual power, Fano's inequality yields a floor $B(s)$. This floor is a lower bound on what we term goal revelation, the mutual information $\mathbb{I}(G;\xi \mid s)$ between the human's latent goal and the realised trajectory. Empowerment caps revelation from above, so that $B(s) \le \mathbb{I
    
[^170]: KG-Neighbor：面向嵌入可视化的知识引导邻域构建

    KG-Neighbor: Knowledge-Guided Neighborhood Construction for Embedding Visualization

    [https://arxiv.org/abs/2611.07246](https://arxiv.org/abs/2611.07246)

    本文提出知识引导邻域图构建方法KGN，利用从实例中抽取的知识图谱三元组的共享程度作为语义证据来调整嵌入距离并重建top-k邻域图，从而在不使用类别标签的情况下提升嵌入可视化中邻域关系的可靠性，并配套提出IDProfile工具来解释类别标签与邻域多数类之间的残余不匹配。

    

    在嵌入可视化中，相邻的实例通常被解读为语义相关，然而由输入嵌入导出的邻域关系可能并不反映内容层面的真实关系。基于邻域的降维（DR）方法通过优化基于嵌入的邻域图来生成布局，可能会延续这一局限性。我们提出了知识引导邻域图构建方法（KGN），它将从每个实例中提取的关系知识表示为一组知识图谱（KG）三元组，并利用三元组的共享程度作为成对语义证据来调整嵌入距离并重建 top-k 邻域图；无论是 KG 抽取还是邻域构建均不使用类别标签。我们还提出了 IDProfile（IDP），通过独立评估源内容是否支持所分配的类别标签以及邻域多数类，来解释残余的不匹配情况。在八个文本和图像数据集上……

    arXiv:2611.07246v1 Announce Type: cross  Abstract: In embedding visualizations, nearby instances are interpreted as semantically related, although neighborhoods derived from input embeddings may not reflect content relationships. Neighborhood-based dimensionality reduction (DR) methods can propagate this limitation by optimizing layouts from embedding-based neighborhood graphs. We propose Knowledge-Guided Neighborhood Graph Construction (KGN), which represents relational knowledge derived from each instance as a set of knowledge graph (KG) triples. It uses the degree of triple sharing as pairwise semantic evidence to adjust embedding distances and reconstruct the top-k neighborhood graph; neither KG extraction nor neighborhood construction uses class labels. We also introduce IDProfile (IDP), which interprets residual mismatches by independently assessing whether the source content supports the assigned class label and the neighborhood-majority class. Across eight text and image datase
    
[^171]: MAC-v2：从像素出发的长时程基于模型的价值扩展

    MAC-v2: Long-horizon model-based value expansion from pixels

    [https://arxiv.org/abs/2611.07245](https://arxiv.org/abs/2611.07245)

    MAC-v2通过训练无重构的潜在世界模型并结合动作分块，实现了从像素出发的长时程基于模型的价值扩展，有效缓解了表征学习与动力学预测误差在想象轨迹上的递归累积问题。

    

    我们研究了长时程基于模型的价值扩展能否作为基于像素的离线强化学习（RL）的有效框架。从像素进行长时程基于模型的价值扩展比基于状态的环境要困难得多，因为智能体必须同时学习表征及其动力学，而来自任一方面的误差都会在想象的轨迹上递归累积。我们提出了MAC-v2，它通过训练一个无重构的潜在世界模型并结合动作分块，实现了从像素进行的长时程基于模型的价值扩展。通过从预测和控制目标中学习表征，MAC-v2无需编码可能阻碍动力学预测的、与控制无关的视觉细节。动作分块则通过减少价值扩展所需的递归模型预测次数，进一步缓解了预测误差问题。在包括视觉OGBench、MMBench2、Robomimic等多样视觉基准上的实验表明……

    arXiv:2611.07245v1 Announce Type: cross  Abstract: We study whether \emph{long-horizon model-based value expansion} can serve as an effective framework in pixel-based offline reinforcement learning (RL). Long-horizon model-based value expansion from pixels is substantially harder than state-based environments, as the agent must learn both the representation and its dynamics, and errors from either can accumulate recursively over imagined trajectories. We introduce MAC-v2, which enables long-horizon model-based expansion from pixels by training a reconstruction-free latent world model with action chunking. By learning representations from prediction and control objectives, MAC-v2 does not need to encode control-irrelevant visual details that can hinder the dynamics prediction. Action chunking further mitigates the prediction error by reducing the number of recursive model predictions required for value expansion. Across diverse visual benchmarks including visual OGBench, MMBench2, Robom
    
[^172]: D²-OPD：在外推式在线策略蒸馏中分离方向与距离

    $D^2$-OPD: Separating Direction and Distance in Extrapolative On-Policy Distillation

    [https://arxiv.org/abs/2611.07241](https://arxiv.org/abs/2611.07241)

    D²-OPD通过解耦外推方向与距离来改进外推式在线策略蒸馏——用每前缀硬KL约束限制目标与教师的距离，并用教师训练后偏移与初始师生差距的自适应组合确定外推方向，从而在多种场景下提供更稳健的学习信号。

    

    在线策略蒸馏（OPD）训练学生模型在学生生成的前缀上匹配教师模型的下一个词元分布。外推式OPD进一步扩展教师相对于参考策略的对数概率偏移，以构建更强的学习目标。然而，现有方法将外推方向与距离耦合在一起，使得教师偏移的幅度直接决定目标移动的远近。我们提出D²-OPD，将这两个因素解耦：通过针对每个前缀的硬KL约束来限制目标与教师之间的距离，同时由教师训练后偏移与初始师生差距的自适应组合来决定外推方向。我们使用前向KL散度将学生模型优化至该目标，即使对低概率词元也能提供学习信号。我们在三种设置下进行评估：标准OPD表现良好的设置、标准OPD在基础模型初始化下表现不佳的设置，以及没有教……（原文摘要在此处截断）

    arXiv:2611.07241v1 Announce Type: cross  Abstract: On-policy distillation (OPD) trains a student model to match a teacher's next-token distributions on student-generated prefixes. Extrapolative OPD further extends the teacher's log-probability shift relative to a reference policy to construct stronger learning targets. However, existing methods couple extrapolation direction and distance, allowing the magnitude of the teacher's shift to dictate how far the target moves. We propose $D^2$-OPD, decoupling these two factors: a per-prefix hard KL constraint bounds the target's distance from the teacher, while an adaptive combination of the teacher's post-training shift and the initial student--teacher gap determines the extrapolation direction. We optimize the student toward this target using forward KL, providing learning signals even for low-probability tokens. We evaluate three settings: where standard OPD performs well, where it struggles with base-model initialization, and where no tea
    
[^173]: 语言模型神经元中的原型语境：一种智能架构？

    Archetypal contexts in the neurons of language models: an architecture for intelligence?

    [https://arxiv.org/abs/2611.07234](https://arxiv.org/abs/2611.07234)

    该研究在Qwen3.8-27B模型中发现，跨领域的原型语境（如“临界点”）、主题和语言分别由三组可独立操控的神经元编码，且网络层呈现“外层语言—中间主题—核心原型”的三明治式组织架构，暗示这可能正是智能的底层架构。

    

    我们验证了此前的假设，即语言模型中存在着独立于具体实例化领域的关联模式（Nordfors, 2026）。这些“原型语境”是人们在众多领域中都能识别出的关系模式，例如“临界点”或“赢家通吃”，它们只被存储一次，便可跨领域应用。我们在模型 Qwen3.8-27B 内部展示了这一现象：一组神经元承载原型（如临界点），另一组承载主题（如政治、生态学或细胞生物学），第三组承载语言（如英语、西班牙语或中文）。当使承载“临界点”原型的神经元沉默后，模型不再识别该模式，但仍然知道文本讲的是什么；当使承载主题的神经元沉默后，模型失去主题信息，但仍能说出该模式。在神经元映射中，各层自组织成一种“三明治”结构：外层承载语言，其内的中间层承载主题，而最核心的层承载原型。三明治……（原文摘要在此处截断）

    arXiv:2611.07234v1 Announce Type: cross  Abstract: We validate our previous hypothesis that a language model holds relational patterns apart from the subject domains that instantiate them (Nordfors, 2026). These archetypal contexts are patterns of relations that people recognize across many fields, such as the tipping point or winner takes all, held once and applied across domains. We show this inside the model, Qwen3.8-27B. One set of neurons carries the archetype, say the tipping point; another carries the topic, say politics, ecology or cell biology; a third carries the language, such as English, Spanish or Chinese. Silence the tipping point's neurons, and the model no longer identifies the pattern but still knows what the text is about. Silence the topic's, and it loses the topic but still names the pattern. In the neuron maps the layers organize themselves into a sandwich: the outer layers carry languages, the layers inside them topics, the layers at the heart archetypes. The sand
    
[^174]: 递归稀疏注意力：以对数成本实现百万token长度外推

    Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost

    [https://arxiv.org/abs/2611.07232](https://arxiv.org/abs/2611.07232)

    RecSA在KV缓存上构建递归树结构，利用从稠密注意力蒸馏且跨层级共享的池化键，使每个查询的选择成本降至O(log L)，从而以对数成本实现了百万token级的长度外推。

    

    在约100万token的上下文长度下，稠密注意力在预填充和解码阶段成本高昂，且训练通常依赖长文档，而随着目标长度的增加，长文档变得越来越稀缺。大多数稀疏注意力方法会对所有前序块或token进行评分，因此在上下文长度L时每个查询的选择成本为O(L)；分层选择可将该成本降至O(log L)，但此前可训练的分层方法仅在训练长度内报告了准确性。我们提出RecSA（递归稀疏注意力），它在KV缓存上构建一棵树，其中每个节点用一个键和一个均值池化的值来概括其子树。这些键来自每层一个池化器，由所有层级共享，并从稠密注意力中蒸馏而来。token之上的每一层最多选择k个候选，因此每个查询具有O(log L)的选择成本并读取O(log L)个条目；未被选中的候选的摘要与精确token共享同一个softmax。

    arXiv:2611.07232v1 Announce Type: new  Abstract: At context lengths of about 1M tokens, dense attention is costly in prefill and decoding, and training typically relies on long documents, which become scarcer as the target length increases. Most sparse attention methods score all preceding blocks or tokens, so their selection cost per query is $O(L)$ at context length $L$; hierarchical selection reduces it to $O(\log L)$, but prior trainable hierarchical methods report accuracy only within their training length. We propose RecSA (Recursive Sparse Attention), which builds a tree over the KV cache in which each node summarizes its subtree with one key and one mean-pooled value. The keys come from one pooler per layer, shared by all levels and distilled from dense attention. Each level above the tokens selects at most $k$ candidates, so each query has $O(\log L)$ selection cost and reads $O(\log L)$ entries; the summaries of unselected candidates share one softmax with the exact tokens. E
    
[^175]: TileQuant：以Tile作为大语言模型混合精度训练后量化的精度分配单元

    TileQuant: Tile as the Precision-Allocation Unit for LLM Mixed-Precision Post-Training Quantization

    [https://arxiv.org/abs/2611.07217](https://arxiv.org/abs/2611.07217)

    TileQuant提出以硬件对齐的二维权重tile为精度分配单元的免训练混合精度训练后量化框架，通过离线校准得到的激活感知与梯度感知评分，让高分tile保留16位精度、其余tile使用8位，从而在降低校准成本与解码开销的同时更好地保持LLM质量。

    

    训练后量化（PTQ）可以降低大语言模型（LLM）的部署成本，但要保持模型质量就必须考虑权重敏感度的不均匀性。统一PTQ使用单一的权重位宽；重要性感知PTQ在该位宽内调整缩放因子或重建权重；按列混合精度则以更高精度保留选定的列。然而，敏感度在列内部也存在差异，因此整列保留同时会保护不那么敏感的权重，而运行时的离群值检测则会增加解码开销。为此，我们提出了TileQuant——一个免训练框架，它以硬件对齐的二维权重tile（分块）作为精度分配单元。离线校准提供两类评分：基于前向统计的输入激活感知评分（校准成本更低），以及基于损失敏感度的梯度感知评分（质量保持能力更强）。在每个投影层的保留预算内，得分最高的tile保持16位精度，其余tile使用8位精度。

    arXiv:2611.07217v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) reduces large language model (LLM) deployment costs, but preserving quality requires accounting for uneven weight sensitivity. Uniform PTQ uses one weight bit-width, importance-aware PTQ adjusts scales or reconstructs weights within that width, and column-wise mixed precision retains selected columns at higher precision. However, sensitivity varies within columns, so whole-column retention also protects less sensitive weights, while runtime outlier detection adds decoding overhead. We therefore introduce TileQuant, a training-free framework that assigns precision to hardware-aligned 2D weight tiles. Offline calibration provides input-activation-aware scoring from forward statistics for lower calibration cost and gradient-informed scoring from loss sensitivity for stronger quality preservation. Within each projection's retention budget, the highest-scoring tiles remain at 16-bit and the rest use 8-bit. T
    
[^176]: 组合式去对齐：审计大型视觉-语言模型的安全性

    Compositional Unalignment: Auditing Safety in Large Vision-Language Models

    [https://arxiv.org/abs/2611.07216](https://arxiv.org/abs/2611.07216)

    本文提出以“去对齐”作为安全对齐的审计手段，揭示大型视觉-语言模型的对齐并不稳健——极少量的监督微调即可达到与多阶段强化学习流程相当的基准分数，且模型对组合性伤害的识别与拒绝能力是相互解耦的。

    

    大型视觉-语言模型（LVLMs）容易受到组合性伤害：一张图像和一个提示词各自单独来看都是无害的，但经过模型的跨模态推理后却可能变得有害。近期的对齐方法在组合安全基准上报告了很高的分数，但LLM裁判只能测量观察到的行为，无法揭示安全性在模型内部被整合得有多稳健。我们认为，“去对齐”为对齐稳健性提供了一种互补的审计手段。我们首先表明，极少量的监督微调即可达到与多阶段强化学习流水线相当的LLM裁判分数，因此仅凭相似的基准分数本身无法确立稳健性。随后我们表明，对组合性伤害的识别与拒绝是相互解耦的：九种标准的方向探测方法能够以最高AUC=0.99检测出组合性伤害，然而消融所恢复出的方向，要么使拒绝行为保持完好，要么破坏模型的连贯性。因此我们……

    arXiv:2611.07216v1 Announce Type: cross  Abstract: Large vision-language models (LVLMs) are vulnerable to compositional harm: an image and a prompt that are benign in isolation become harmful through the model's cross-modal reasoning. Recent alignment methods report strong scores on compositional safety benchmarks, but an LLM judge only measures observed behavior and cannot reveal how robustly safety is integrated into the model. We argue that un-alignment provides a complementary audit of alignment robustness. We first show that minimal supervised fine-tuning reaches an LLM-judge score comparable to a multi-stage reinforcement-learning pipeline, so similar benchmark scores cannot establish robustness by themselves. We then show that recognition and refusal of compositional harm are decoupled: nine standard direction-finding methods detect compositional harm with up to AUC=0.99, yet ablating the recovered directions either leaves refusal intact or destroys model coherence. We therefore
    
[^177]: 用于建模社交媒体病毒式传播的分数阶物理信息神经网络

    Fractional physics-informed neural networks to model social media virality

    [https://arxiv.org/abs/2611.07215](https://arxiv.org/abs/2611.07215)

    本文提出一种分数阶物理信息神经网络，能够同时学习社交媒体参与度的时间记忆与非线性增长动力学，并通过符号回归生成可解释方程，同时解决了观测噪声导致的分数阶估计偏差问题。

    

    现有的社交媒体病毒式传播模型要么预先规定参与度增长的函数形式，要么假设过去活动影响未来参与度的固定机制。我们提出了一种分数阶物理信息神经网络，能够同时学习社交媒体参与度的时间记忆特性和非线性增长动力学，并利用符号回归将学习到的动力学转化为可解释的方程。在24个小时观测点的部署分辨率下进行的合成实验表明，分数阶和反应定律可以从干净数据中联合恢复。然而，在观测噪声存在的情况下，标准的逐点物理残差会通过尺度伪影将分数阶推向其下界。我们使用一种完全白化的残差来解决这一失败问题，该残差将观测噪声传播通过分数阶算子。将该框架应用于来自X平台（前Twitter）的348条帖子，我们发现一个奇

    arXiv:2611.07215v1 Announce Type: new  Abstract: Existing models of social media virality either prescribe the functional form of engagement growth or assume a fixed mechanism by which past activity influences future engagement. We present a fractional physics-informed neural network that learns both the temporal memory and nonlinear growth dynamics of social media engagement, using symbolic regression to translate the learned dynamics into interpretable equations. Synthetic experiments at the deployment resolution of 24 hourly observations demonstrate that the fractional order and reaction law can be jointly recovered from clean data. Under observation noise, however, the standard pointwise physics residual drives the fractional order toward its lower bound through a scale artifact. We address this failure using a fully whitened residual that propagates observation noise through the fractional operator. Applying the framework to 348 posts from X (formerly Twitter), we find that a sing
    
[^178]: 典型相关神经网络

    Canonical Correlation Neural Networks

    [https://arxiv.org/abs/2611.07212](https://arxiv.org/abs/2611.07212)

    本文提出典型相关神经网络（CANEs），将视图内协方差和跨视图协方差作为不同的关系归纳偏置引入网络架构，把以PCA为理论基础、稳定且参数高效的协方差神经网络框架从单视图数据推广到了多视图场景。

    

    诸如网格或图等关系归纳偏置能够让神经网络以更少的可训练参数获得更好的泛化能力，但这需要事先已知数据结构。当没有可用结构时，协方差矩阵提供了一种有原则的替代方案，协方差神经网络已经表明，使用协方差矩阵可以构建在理论上以主成分分析（PCA）为基础、对有限样本估计误差稳定且参数高效的模型。然而，这一框架仅适用于单视图数据，而许多应用是通过多个视图来观察同一现象的，每个视图都有各自的视图内协方差，以及连接不同视图的跨视图协方差。为了以一种有原则的方式将这些矩阵作为归纳偏置引入网络，我们提出了典型相关神经网络，这是一类将视图内协方差和跨视图协方差作为不同关系偏置来解决下游任务的网络架构。通过谱分析……

    arXiv:2611.07212v1 Announce Type: new  Abstract: Relational inductive biases such as grids or graphs allow neural networks with fewer trainable parameters and improved generalization, but require a known structure beforehand. When no structure is available, the covariance matrix provides a principled substitute, and coVariance Neural Networks (VNNs) have shown that using it yields models that are theoretically grounded in Principal Component Analysis, stable to finite-sample estimation errors and parameter-efficient. This framework, however, is limited to single-view data, whereas many applications observe the same phenomenon through multiple views, each with its own in-view covariance and a cross-view covariance linking them. To incorporate these matrices as inductive biases in a principled way, we introduce CAnonical correlation NEural networks (CANEs), architectures that use in-view and cross-view covariances as distinct relational biases to solve downstream tasks. Via spectral anal
    
[^179]: 基于影响性误差消除的任务校准随机特征

    Task-Calibrated Random Features via Influence-Based Error Cancellation

    [https://arxiv.org/abs/2611.07209](https://arxiv.org/abs/2611.07209)

    该论文提出任务校准随机特征（TCRF），通过基于任务特定影响构建的可重用校准空间来平衡频率矩并消除对下游任务有影响的采样误差，从而改进核岭回归和核Stein差异等核方法的近似质量。

    

    随机特征通过蒙特卡洛方法近似核积分，然而只有特定的频率采样误差会影响下游任务。我们提出了任务校准随机特征（TCRF），它利用在核条目之间共享的权重，来平衡由任务特定影响构建的频率矩。闭式解权重与经典的最小距离校准方法相一致；其贡献在于提供了一个可重用的、由任务导出的校准空间。我们证明了第一阶方差等于投影到该空间后的残差方差。对于核岭回归，完全消除会产生非高斯的二阶极限。对于核Stein差异，子集恒等式和不交块方差分解消除了试点噪声的线性分量。正的指数倾斜权重保持了半正定的特征表示，并支持更高的可行校准秩。配对实验显示了改进的近似

    arXiv:2611.07209v1 Announce Type: cross  Abstract: Random features approximate kernel integrals by Monte Carlo, yet only selected frequency-sampling errors affect a downstream task. We propose Task-Calibrated Random Features (TCRF), which balance frequency moments constructed from task-specific influence using weights shared across kernel entries. Closed-form weights coincide with classical minimum-distance calibration; the contribution is a reusable, task-derived calibration space. We show that first-order variance equals the residual variance after projection onto this space. For kernel ridge regression, complete cancellation yields a non-Gaussian second-order limit. For kernel Stein discrepancy, a subset identity and disjoint-block variance decomposition eliminate the linear component of pilot noise. Positive exponential-tilting weights preserve the positive-semidefinite feature representation and support higher feasible calibration ranks. Paired experiments show improved approximat
    
[^180]: TrustMix：分布偏移下预训练模型的优雅降级

    TrustMix: Graceful Degradation for Pretrained Models under Distribution Shift

    [https://arxiv.org/abs/2611.07206](https://arxiv.org/abs/2611.07206)

    TrustMix是一种无需重训练的轻量级后处理方法，它根据OOD偏离程度将预训练模型的预测分布与用户指定的回退分布混合，使模型外推行为显式可控并实现优雅降级。

    

    在训练域之外进行预测必然需要对真实值如何外推做出假设，然而在大多数神经网络中，这些假设仅通过架构隐式地体现。我们提出TrustMix，这是一种轻量级的后处理（post-hoc）方法，能够使预训练模型的外推行为变得明确且可控。TrustMix将预训练模型产生的预测分布与用户指定的回退分布 p₀(y|x) 相结合，随着分布外（OOD）偏离统计量的增大，预测分布逐渐向 p₀(y|x) 坍缩。TrustMix提供了优雅降级：随着OOD信号的增强，预测分布收敛到回退分布。回退分布 p₀(y|x) 可以反映可用的领域知识，或者依赖于与领域无关的平稳性假设。TrustMix无需对基础模型进行重新训练，并且可以同时应用于回归和分类任务。

    arXiv:2611.07206v1 Announce Type: new  Abstract: Predicting beyond the training domain necessarily requires assumptions about how the ground truth extrapolates, yet in most neural networks these assumptions are made only implicit through the architecture. We propose TrustMix, a lightweight post-hoc method that makes a pretrained model's extrapolative behaviour explicit and controllable. TrustMix combines the predictive distribution produced by a pretrained model with a user-specified fallback distribution $p_0(y|x)$, such that as the out-of-distribution (OOD) deviation statistic grows, we collapse towards $p_0(y|x)$. TrustMix provides graceful degradation: the predictive distribution converges toward the fallback distribution as the OOD signal grows. The fallback distribution $p_0(y|x)$ can reflect domain knowledge, if available, or rely on a domain-agnostic stationarity assumption. TrustMix does not require retraining of the base model and can be applied to both regression and classif
    
[^181]: 基于主干信息的EEG预训练后适配：一项参数容量对照研究

    Backbone-Informed Adaptation after EEG Pretraining: A Capacity-Controlled Study

    [https://arxiv.org/abs/2611.07192](https://arxiv.org/abs/2611.07192)

    在参数量匹配、容量受控的公平比较下，基于主干结构设计的适配器相较于简单的逐token多层感知机适配器，在多个EEG基础模型和下游任务上并未表现出一致优势。

    

    脑电图（EEG）基础模型学习跨通道和时间融合信息的表征，这一结构启发人们设计在下游学习中建模同样关系的适配器。然而，这种设计是否优于其他候选适配器尚不明确；此外，为了在候选适配器之间进行公平比较，还必须考虑它们的可训练参数数量。本文通过将基于主干信息（原生）的适配器与独立变换每个特征向量的逐token多层感知机对照方法进行比较，探究前者是否能更好地利用给定的参数容量。我们近似匹配两者的可训练参数数量，并保持预训练编码器冻结不变。在两个EEG基础模型以及涵盖情绪识别、睡眠分期和EEG事件分类的任务上，原生适配器并未表现出一致的优势。对于其中一个主干网络，两者之间的差异……（摘要原文在此处截断）

    arXiv:2611.07192v1 Announce Type: new  Abstract: Electroencephalography (EEG) foundation models learn representations that combine information across channels and time. This structure motivates adapters that model the same relationships during downstream learning. However, it is not clear whether such a design outperforms other candidate adapters. Moreover, in order to arrive at a fair comparison between candidate adapters, we also need to consider their trainable parameters. Here, we ask whether a backbone-informed (native) adapter makes better use of a given capacity by comparing it against a token-wise multilayer perceptron control that transforms each feature vector independently. We approximately match their trainable parameter counts and keep the pretrained encoder fixed. Across two EEG foundation models and tasks covering emotion recognition, sleep staging, and EEG event classification, the native adapter shows no consistent advantage. For one backbone, differences between the d
    
[^182]: ShapleyOG：基于奇数高斯过程的快速自适应Shapley值估计

    ShapleyOG: Fast Adaptive Shapley Value Estimation with Odd Gaussian Processes

    [https://arxiv.org/abs/2611.07173](https://arxiv.org/abs/2611.07173)

    ShapleyOG利用Shapley值仅依赖于奇数分量的特性，提出了一种基于奇数高斯过程的快速自适应估计方法，克服了ShaplEIG等现有自适应方法因每轮迭代重复拟合高斯过程代理模型而带来的计算开销问题。

    

    Shapley值是机器学习中进行归因分析的核心工具。由于精确计算Shapley值需要在指数级数量的玩家联盟上评估被解释的合作博弈，因此通常基于有限次数的评估来进行估计。最近的研究（ShaplEIG）表明，自适应地选择要评估的联盟能够显著提高相对于经典估计器的样本效率。然而，这种优势伴随着一个顺序化流程的代价：在每次迭代中，都需要重新拟合博弈的高斯过程代理模型，并选择对Shapley值具有最大信息增益期望的联盟。由此产生的计算开销使其仅适用于评估预算较低的高成本博弈。我们提出了基于奇数高斯过程的快速自适应Shapley值估计方法（ShapleyOG）以克服这一限制。ShapleyOG利用了Shapley值仅依赖于奇数分量这一事实（摘要在此处截断）。

    arXiv:2611.07173v1 Announce Type: cross  Abstract: Shapley values are a central tool for attribution in machine learning. Since their exact computation requires evaluating the cooperative game to be explained on exponentially many player coalitions, they are typically estimated from a limited number of evaluations. Recently, ShaplEIG has shown that adaptively selecting the coalitions to evaluate substantially improves sample efficiency over classical estimators. This, however, comes at the cost of a sequential procedure that, in every iteration, refits a Gaussian process surrogate of the game and selects the coalition with the largest expected information gain about the Shapley values. The resulting computational overhead limits its applicability to costly games with low evaluation budgets. We present fast adaptive Shapley value estimation with Odd Gaussian Processes (ShapleyOG) to overcome this limitation. ShapleyOG leverages the fact that Shapley values depend only on the odd compone
    
[^183]: Meta-Frieren：超越单层的在线学习

    Meta-Frieren: Beyond Single Level Online Learning

    [https://arxiv.org/abs/2611.07169](https://arxiv.org/abs/2611.07169)

    Meta-Frieren是一个两阶段在线优化的通用框架，通过UCB风格的元层面选择准则最大化累积奖励，统一并扩展了已有理论保证，首次为Feel-Good Thompson采样推导出高概率遗憾界，并可成功应用于跨多个模型类的超参数优化。

    

    我们研究了一个两阶段在线优化问题：在每一轮中，元学习者首先选择一个智能体-环境对，然后被选中的智能体与环境进行交互并获得奖励。我们提出了Meta-Frieren，一个用于在此设定下最大化累积奖励的通用框架。我们的分析给出了一个简单的UCB风格准则用于元层面的选择，该准则将先前的理论保证作为特例恢复，同时将其扩展到更广泛的设定。我们在贝叶斯和频率派假设下用GP-UCB对Meta-Frieren进行了实例化。此外，据我们所知，我们首次推导了Feel-Good Thompson采样的高概率遗憾界，将其特化到有限函数类和参数化函数类，并将其集成到Meta-Frieren中。实验验证了我们的理论结果，并表明Meta-Frieren可以成功应用于跨多个模型类的超参数优化。

    arXiv:2611.07169v1 Announce Type: new  Abstract: We study a two-stage online optimization problem: at each round, a meta-learner first selects an agent-environment pair, then the chosen agent interacts with the environment, and receives a reward. We introduce Meta-Frieren, a general framework for maximizing cumulative reward in this setting. Our analysis yields a simple UCB-style criterion for meta-level selection and recovers prior guarantees as special cases while extending them to broader settings. We instantiate Meta-Frieren with GP-UCB under Bayesian and frequentist assumptions. We also derive, to our knowledge, the first high-probability regret bound for Feel-Good Thompson Sampling, specialize it to finite and parametric function classes, and integrate it into Meta-Frieren. Experiments support our theoretical results and demonstrate that Meta-Frieren can be successfully applied to hyperparameter optimization across multiple model classes.
    
[^184]: 多步软件工程：编程智能体能否在长时程中持续基于自己编写的代码进行构建？

    Multi-step SWE: Can Coding Agents Keep Building on Their Own Code Over Long Horizons?

    [https://arxiv.org/abs/2611.07166](https://arxiv.org/abs/2611.07166)

    提出长时程基准 Multi-step SWE，包含69个由5-15个依赖步骤组成的软件工程任务链，要求智能体在自身编写的代码上持续迭代开发，最强前沿模型的解决率仅18.5%，揭示了当前编程智能体在长时程自我延续开发中的严重不足。

    

    开发者很少只给编程智能体提出单次请求：他们通常要求智能体先构建一个组件，然后对其进行扩展、升级和优化，而每次操作都是基于智能体自己编写的代码。然而，大多数基准测试一次只评估单个任务。我们提出了 Multi-step SWE，这是一个包含69个软件工程任务的长时程基准，每个任务都是在单一代码库上由5-15个相互依赖的步骤组成的链条。每个步骤都是一个完整的多小时工程请求，它从智能体在前几步生成的代码开始，因此早期的设计决策和错误会延续下去。这些任务涵盖四大类工程工作和九种编程语言，既包括从零开始的构建，也包括现有的开源代码库，并且每一步都基于一个既有的真实标准，该标准同时定义了指令本身及其验证器。在九个前沿模型中，表现最强的模型解决率仅为18.5%，69个任务中有38个没有任何模型能够解决。我们发布了 Mult

    arXiv:2611.07166v1 Announce Type: cross  Abstract: Developers rarely give a coding agent a single request: they ask it to build a component and then to extend, upgrade, and optimize it, each time on code the agent itself wrote. Most benchmarks, however, score one task at a time. We introduce Multi-step SWE, a long-horizon benchmark of 69 software engineering tasks, each a chain of 5-15 dependent steps on a single codebase. Every step is a complete, multi-hour engineering request that starts from the code the agent produced in the preceding steps, so early design decisions and errors carry forward. The tasks span four families of engineering work and nine languages, include both from-scratch builds and existing open-source codebases, and ground every step in an existing source of truth that defines both the instruction and its verifier. Across nine frontier models, the strongest model reaches a solve rate of only 18.5%, and 38 of the 69 tasks are not solved by any model. We release Mult
    
[^185]: 基于神经分位数分辨的结构条件独立性检验

    Structural Conditional Independence Test via Neural Quantile Resolution

    [https://arxiv.org/abs/2611.07165](https://arxiv.org/abs/2611.07165)

    提出QRCIT——一种基于神经分位数形变的条件独立性检验方法，通过轻量级残差分位数流模型在学习检验统计量的同时，以可检查的形变场揭示条件分布的具体变化位置，并配套发布了结构依赖基准SDB。

    

    当依赖关系微弱、局限于局部，或表现为分布变化而非全局均值偏移时，条件独立性检验十分困难。现有检验通常返回一个标量统计量或p值，它只是汇总了反对独立性的证据，却无法描述条件分布在何处发生了变化。我们提出QRCIT，一种分位数分辨的条件独立性检验方法，它通过共享模型学习到的分位数形变来比较P(X|Z)与P(X|Y,Z)。一个轻量级的残差分位数流模型首先学习基线条件分位数规律，然后学习由Y诱导的额外形变，从而产生一个带有条件随机化校准的全局预测增益统计量，以及一个在上下文与分位数水平上均可检查的形变场。我们还引入了结构依赖基准（SDB），用于评估受实际应用启发的依赖结构，例如上下文局部依赖、尾部依赖……（摘要此处截断）

    arXiv:2611.07165v1 Announce Type: new  Abstract: Conditional independence testing is difficult when dependence is weak, localized, or expressed through distributional changes rather than a global mean shift. Existing tests usually return a scalar statistic or p-value, which summarizes evidence against independence but does not describe where the conditional law changes. We propose QRCIT, a quantile-resolution conditional independence test that compares P(X|Z) and P(X|Y,Z) through a quantile deformation learned by a shared model. A lightweight residual quantile-flow model first learns the baseline conditional quantile law and then learns the additional deformation induced by Y, yielding a global predictive-gain statistic with conditional-randomization calibration and an inspectable deformation field over context and quantile level. We also introduce the Structural Dependency Benchmark (SDB), which evaluates dependence structures motivated by real applications, such as context-local, tai
    
[^186]: 数据同化中的非线性滤波方法综述及生成式人工智能赋能的得分滤波器的最新进展

    A Survey of Nonlinear Filtering Methods for Data Assimilation and Recent Advances in Generative AI-Enabled Score Filters

    [https://arxiv.org/abs/2611.07160](https://arxiv.org/abs/2611.07160)

    本综述系统梳理了数据同化领域三类经典非线性滤波方法（卡尔曼类、粒子类和核方法）的局限性，并介绍了利用生成式人工智能得分滤波器求解最优滤波问题的最新进展。

    

    许多现实世界的应用，如数值天气预报或野火追踪，涉及对高度非线性、具有混沌特性且受多种不确定性来源影响的动力系统进行建模。为了提高模型预测精度并减少这些不确定性，数据同化（DA）将动力学模型与部分、含噪声的观测数据相结合，以估计系统状态的演变。这依赖于求解最优滤波问题：即在给定观测序列的条件下计算隐藏状态的条件分布。经典方法在三个不同方面存在缺陷：基于卡尔曼的方法假设高斯后验分布，在非线性观测算子下会产生有偏估计；基于粒子的方法用加权样本表示分布，由于维度灾难在高维空间中会失效；基于核的方法虽超越了离散粒子表示，但无法维持（摘要在此处截断）……

    arXiv:2611.07160v1 Announce Type: new  Abstract: Many real-world applications, such as numerical weather prediction or wildfire tracking, involve modeling highly nonlinear dynamical systems that are chaotic and influenced by various sources of uncertainty. To improve model prediction accuracy and reduce such uncertainties, data assimilation (DA) combines the dynamical model with partial, noisy observations to estimate the evolution of the system's state. Doing so depends on solving the optimal filtering problem: computing the conditional distribution of a hidden state given a sequence of observations. Classical approaches fail along three distinct axes: Kalman-based methods assume Gaussian posteriors and produce biased estimates under nonlinear observation operators; particle-based methods represent distributions with weighted samples and collapse in high dimensions due to the curse of dimensionality; and kernel-based methods move beyond discrete particle representations but cannot mai
    
[^187]: 深度还是描述？在主动信息获取中，一步式潜在评分何时能替代前瞻搜索

    Depth or Description? When a One-Step Latent Score Replaces Lookahead in Proactive Information Acquisition

    [https://arxiv.org/abs/2611.07155](https://arxiv.org/abs/2611.07155)

    该论文发现主动信息获取中对深度前瞻搜索的需求往往源于隐藏原因的描述方式——基于能编码决策信息的合适潜在变量构建的一步式评分，可以用极小的搜索代价替代大部分深度前瞻搜索。

    

    主动式智能体必须在行动之前决定询问什么。精确的前瞻搜索能评估问题的各种组合，但代价高昂且依赖于拟合的模型；一步式评分虽然廉价，却可能遗漏这些组合效应。我们的主要发现是：对搜索深度的需求往往取决于隐藏原因的描述方式。基于标签的一步式评分可能无法察觉答案之间的相互作用，而基于合适潜在原因的评分则不会；当该潜在变量编码了决策信息时，它能以极小的搜索代价替代大部分深度前瞻搜索。换一种描述方式，同一个问题可能会使评分失效，而精确值却不受影响，因此应保持潜在变量足够小，并用精确值加以对照验证。从一个语言模型的回答中学习到的概率无法迁移到另一个语言模型，除非规划过程能兼容两者；而直接从token中读取的概率仅相当于几个标签的价值。

    arXiv:2611.07155v1 Announce Type: cross  Abstract: A proactive agent must decide what to ask before it acts. Exact lookahead values combinations of questions but is expensive and leans on a fitted model; a one-step score is cheap but can miss them. Our main finding: the need for depth often lies in how the hidden cause is described. A one-step score on the label can be blind to an interaction among answers. A score on a suitable latent cause is not, and where that latent encodes the decision it replaces most of depth at a fraction of the search. Described differently, the same problem can defeat the score while exact values do not move, so keep the latent small and check it against exact values. Probabilities learned from one language model's answers do not carry to another unless planning allows for both; those read from tokens are worth only a few labels.
    
[^188]: FastAFD：注意力-FFN分离在机架级NVLink上何时能带来收益？

    FastAFD: When Does Attention-FFN Disaggregation Pay on Rack-Scale NVLink?

    [https://arxiv.org/abs/2611.07149](https://arxiv.org/abs/2611.07149)

    本文构建了面向GB200 NVL72机架级NVLink系统的FastAFD引擎，通过在微批次间重叠MoE与注意力计算，实现了相比最优vLLM宽EP设置高达1.17倍的解码吞吐量提升，并提出了一个仅凭共置vLLM剖析数据即可在3%误差内预测AFD收益的分析模型。

    

    注意力-FFN分离（AFD）将混合专家（MoE）模型中的注意力计算与专家部分分开部署，据报道可提升解码吞吐量（相比共置部署方式）。然而这一优势在GB200 NVL72等机架级NVLink系统上是否依然成立尚不清楚：此前的研究结果要么是在其他硬件上获得的，要么是针对较窄的共置基线，要么基于分析模型，而生产引擎则是在这些系统上采用宽专家并行来服务大型MoE模型。我们构建并调优了FastAFD，这是一个面向GB200 NVL72的AFD引擎，它将一个微批次的MoE计算与另一个微批次的注意力计算重叠执行，并将其与最优的vLLM宽EP设置进行比较。实验表明，AFD在Qwen3上相比宽EP具有高达1.17倍的吞吐量优势，在Qwen3.5上为1.16倍，在MiniMax-M2.5上则与之持平。我们推导了一个分析模型，仅使用共置vLLM的性能剖析数据即可在3%误差范围内预测AFD的吞吐量，从而判断是否值得采用分离部署。

    arXiv:2611.07149v1 Announce Type: cross  Abstract: Attention-FFN disaggregation (AFD) separates attention from the experts of a mixture-of-experts (MoE) model and has been reported to raise decode throughput over colocated serving. Whether this advantage holds on rack-scale NVLink systems such as GB200 NVL72 remains unclear: prior results were obtained on other hardware, against narrower colocated baselines, or with analytical models, whereas production engines serve large MoE models on these systems with wide expert parallelism (wide EP). We build and tune FastAFD, an AFD engine for GB200 NVL72 that overlaps the MoE of one microbatch with the attention of the other, and compare it against the best vLLM wide EP setting. Experiments show that AFD has a throughput advantage of up to 1.17x over wide EP on Qwen3 and 1.16x on Qwen3.5, and matches it on MiniMax-M2.5. We derive an analytical model that predicts AFD throughput within 3% using only profiles of colocated vLLM, telling whether sw
    
[^189]: 情境感知的自动化核磁共振结构解析

    Context-Aware Automated NMR Structure Elucidation

    [https://arxiv.org/abs/2611.07133](https://arxiv.org/abs/2611.07133)

    该论文提出一种将自动化逆合成分析生成的合成前体作为化学情境信息融入AI核磁共振结构解析框架的方案，从而提升化学结构预测的性能与实际应用价值。

    

    机器学习方法正在迅速革新从化学数据中分析和提取结构信息的能力，从而实现高通量的化学发现。尽管现有的数据驱动方法在从头预测化学结构方面已展现出显著潜力，但引入额外的化学背景信息为进一步提升这些模型的性能并扩展其实际应用价值提供了机会。在此，我们提供了一种简单直接的方案，将基于人工智能的光谱解析流程与情境化学数据（例如已知反应物或目标产物）相结合，通过自动化逆合成分析生成大型分子训练集以及以合成前体形式存在的情境数据。我们通过证明我们近期提出的核磁共振结构解析框架能够被训练以有效利用分子（情境信息）来展示我们的方法。

    arXiv:2611.07133v1 Announce Type: cross  Abstract: Machine learning approaches are rapidly revolutionizing the ability to analyze and extract structural insights from chemical data, enabling high-throughput chemical discovery. While existing data-driven approaches have shown remarkable promise for predicting chemical structures in the de novo setting, incorporating additional chemically informed context offers the opportunity to further improve these models' performance and extend their real-world utility. Here we provide a straightforward protocol to combine artificial-intelligence-based spectroscopic elucidation pipelines with contextual chemical data, e.g., known reactants or a desired product, by using automated retrosynthetic analysis to generate large training sets of molecules and contextual data in the form of synthetic precursors. We demonstrate our approach by showing that our recently introduced NMR structure elucidation framework can be trained to effectively leverage molec
    
[^190]: 一种基于代理变量的方法，用于衡量文生图生成器中的种族差异

    A Proxy-Based Method for Measuring Racial Disparities in~Text-to-Image~Generators

    [https://arxiv.org/abs/2611.07132](https://arxiv.org/abs/2611.07132)

    该论文借鉴美国反歧视法中的种族代理变量，提出一种自动化审计方法，通过将生成图像的种族分布与真实人口基准数据比较，并结合肤色和图像质量分析，来衡量文生图模型中的种族表征偏见。

    

    我们提出了一种自动化方法，用于审计文生图生成器相对于真实人口统计基准的表征偏见。我们的方法借鉴了美国反歧视法中使用的种族代理变量，对生成图像中种族表征的变化进行细粒度的比较分析。我们对每张生成的图像按种族进行分类，然后将模型的分布与公开可用的人口统计基准数据进行比较，以识别过度表征和不足表征的模式。我们还使用CIELAB值测量肤色，并计算图像组之间的余弦相似度，以捕捉在人口统计比较中无法显现的偏见，例如不同种族和肤色之间图像质量的差异。我们报告了使用三类代理变量（地理、职业和姓氏）对三个模型（Imagen 4 Fast、Stable Diffusion 3.5 Large Turbo 和 Flux Schnell）进行测试的实验。

    arXiv:2611.07132v1 Announce Type: cross  Abstract: We present an automated method for auditing text-to-image generators for representational bias against ground-truth demographic benchmarks. Our approach adapts the racial proxy variables used in U.S. antidiscrimination law to provide a fine-grained comparative analysis of variations in racial representation in generated images. We classify each generated image by race and then compare the model's distribution to publicly available demographic benchmark data to identify patterns of over- and under-representation. We also measure skin tone using CIELAB values and calculate cosine similarity amongst groups of images in order to capture biases that don't show up in demographic comparisons such as differences in image quality across races and skin tones. We report on experiments testing three models (Imagen 4 Fast, Stable Diffusion 3.5 Large Turbo, and Flux Schnell) using three categories of proxies (geographic, occupational, and surname) a
    
[^191]: 具有自动模型复杂度推断的贝叶斯张量链核机器

    Bayesian Tensor Train Kernel Machines with Automatic Model Complexity Inference

    [https://arxiv.org/abs/2611.07129](https://arxiv.org/abs/2611.07129)

    本文提出了贝叶斯张量链核机器（BTT-KM），通过在TT核心上施加分层ARD先验并结合平均场变分推断，实现了从数据中自动推断TT秩与特征维度的闭式贝叶斯学习，且计算复杂度与确定性方法相同。

    

    张量网络（TN）核机器通过使用低秩张量网络来降低计算量和内存需求，从而加速监督学习。然而，大多数基于张量网络的方法是确定性的，其秩和特征维度需要手动调整，而现有的贝叶斯张量网络核机器仅限于CANDECOMP/PARAFAC分解（CPD），其单一秩由所有模式共享。我们将该框架扩展到张量链（TT）分解，提出了贝叶斯张量链核机器（BTT-KM），其在TT核心上的分层ARD类型先验可以从数据中自动推断TT秩和特征维度。由于每个TT核心仅通过其相邻核心与链上其他核心耦合，其闭式后验更新需要沿链传播所有其他核心的贡献。平均场变分推断产生了一个贝叶斯交替最小二乘算法，其计算复杂度与确定性对应算法相同，并具有闭式预测分布。

    arXiv:2611.07129v1 Announce Type: cross  Abstract: Tensor Network (TN) kernel machines accelerate supervised learning by using low-rank TNs to reduce computation and memory. However, most TN-based methods are deterministic with manually tuned ranks and feature dimensions, while the existing Bayesian TN kernel machine is restricted to the CANDECOMP/PARAFAC decomposition (CPD), whose single rank is shared by all modes. We extend this framework to the tensor train (TT) and propose Bayesian Tensor Train Kernel Machines (BTT-KM), whose hierarchical ARD-type priors on the TT cores infer the TT ranks and feature dimensions from data. Since each TT core is coupled to the others only through its neighbors, its closed-form posterior update requires propagating the contributions of all other cores along the chain. Mean-field variational inference yields a Bayesian alternating least squares algorithm with the same computational complexity as its deterministic counterpart and a closed-form predicti
    
[^192]: 面向癌症组织病理学的可扩展贝叶斯高斯过程分类：分布偏移下的校准不确定性

    Scalable Bayesian Gaussian Process Classification for Cancer Histopathology: Calibrated Uncertainty Under Distribution Shift

    [https://arxiv.org/abs/2611.07127](https://arxiv.org/abs/2611.07127)

    该论文提出一种可扩展的贝叶斯高斯过程分类框架，通过结合RPCholesky低秩核分解、非中心化参数化HMC后验采样和主元诱导点预测方案，首次在约30万训练点规模上实现可处理的HMC贝叶斯推断，为癌症组织病理学诊断在分布偏移下提供校准的不确定性量化。

    

    可靠的不确定性量化对于癌症诊断模型的临床部署至关重要，然而神经网络和变分高斯过程方法可能产生过度自信的预测，尤其是在模型在一个医院训练后应用于另一个医院时所出现的分布偏移情况下。我们提出了一个可扩展的贝叶斯高斯过程分类（GPC）框架，该框架运行于冻结的病理学基础模型嵌入（Phikon）之上，并结合了用于低秩核分解的加速随机主元Cholesky（RPCholesky）方法、采用非中心化参数化的哈密顿蒙特卡洛（HMC）进行后验采样，以及以主元作为诱导点的预测方案。这使得在训练点数量N约为30万时，低秩潜在表示中的HMC后验采样变得可行，从而使基于HMC的低秩GPC能够应用于比以往基于MCMC的演示大得多的数据集。我们在……上进行评估（原文摘要到此中断）。

    arXiv:2611.07127v1 Announce Type: new  Abstract: Reliable uncertainty quantification is essential for clinical deployment of cancer diagnosis models, yet neural networks and variational Gaussian process methods can produce overconfident predictions, particularly under the distribution shifts that arise when models trained at one hospital are applied at another. We present a scalable Bayesian Gaussian Process Classification (GPC) framework that operates on frozen pathology foundation model embeddings (Phikon) and combines accelerated Randomly Pivoted Cholesky (RPCholesky) for low-rank kernel factorization, Hamiltonian Monte Carlo (HMC) with non-centered parameterization for posterior sampling, and a pivots-as-inducing-points scheme for prediction. This makes HMC posterior sampling in the low-rank latent representation tractable at N approximately 300,000 training points, enabling HC-based low-rank GPC on substantially larger datasets than prior MCMC-based demonstrations. We evaluate on 
    
[^193]: 基于静态试验数据的熔融氯化物腐蚀数据驱动建模

    Data-Driven Modeling of Molten Chloride Corrosion Using Static Test Data

    [https://arxiv.org/abs/2611.07125](https://arxiv.org/abs/2611.07125)

    该研究汇编291个静态熔融氯化物腐蚀实验数据，创新性地提出氯化盐制备指数（CSPI）来量化盐条件差异，构建可解释机器学习框架实现了高精度的腐蚀速率预测（交叉验证R²达0.85）。

    

    熔融氯化物腐蚀已被研究数十年，然而在不同合金、盐类和试验条件下实现可靠预测仍存在困难。在这项工作中，研究者将文献中报道的291个静态熔融氯化物腐蚀实验汇编成一个结构化数据集，并据此开发了一个用于腐蚀速率预测的可解释机器学习框架。为了更好地表征不同研究之间盐条件的差异，该工作引入了氯化盐制备指数（CSPI），将文献中报道的盐处理方法、气氛控制和预暴露杂质状态统一到同一数值尺度上。一种混合树模型集成方法提供了最稳定的预测性能，交叉验证R²达到0.85，平均绝对误差（MAE）为0.41 mg cm⁻² day⁻¹。分级形式的CSPI比简单的纯化/未纯化二分类保留了更多有用信息，并成为模型中的主导描述符。

    arXiv:2611.07125v1 Announce Type: new  Abstract: Molten chloride corrosion has been studied for decades, yet reliable prediction across different alloys, salts, and test conditions remains difficult. In this work, 291 static molten chloride corrosion experiments reported in the literature were compiled into a structured dataset and used to develop an interpretable machine-learning framework for corrosion-rate predictions. To better represent the differences in salt condition across studies, a Chloride Salt Preparation Index (CSPI) was introduced to bring the reported salt-treatment method, atmosphere control, and pre-exposure impurity condition on a common numerical scale. A blended tree-based ensemble provided the most consistent predictive performance, achieving a cross-validated \(R^2\) of 0.85 and an MAE of 0.41 mg cm\(^{-2}\) day\(^{-1}\). The graded CSPI retained more useful information than a simple purified/unpurified classification and emerged as the dominant model descriptor.
    
[^194]: 基于结构化临床数据的机器学习预测IVF/ICSI术后活产：系统综述、荟萃分析与批判性评估

    Machine Learning for Live-Birth Prediction After IVF/ICSI From Structured Clinical Data: A Systematic Review, Meta-Analysis, and Critical Appraisal

    [https://arxiv.org/abs/2611.07124](https://arxiv.org/abs/2611.07124)

    本系统综述与荟萃分析对48项研究进行批判性评估后发现，机器学习预测IVF/ICSI后活产的模型在内部验证时AUC可达0.814，但在外部验证时降至0.756，表明现有模型泛化能力有限。

    

    利用常规临床数据预测体外受精（IVF）或卵胞浆内单精子注射（ICSI）后活产的机器学习模型，其报告的性能从接近随机水平到接近完美不等。我们在PRISMA 2020指导下对这些模型进行了系统综述，检索了OpenAlex、Scopus和Web of Science数据库（截至2026年10月22日），采用PROBAST+AI和TRIPOD+AI对其进行质量评估，并在logit尺度上通过随机效应荟萃分析汇总其判别能力。我们定义了决定模型输出含义的要素（决策点、终点、分析单位、数据来源），并按模型家族、决策点和验证设计对纳入的48项研究进行了分类映射。仅在内部评估的模型汇总后的受试者工作特征曲线下面积（AUC）为0.814，而在后期时段或其他中心进行评估的模型AUC为0.756（95%预测区间0.546-0.889），非高偏倚风险的研究AUC为0.764；两者均……

    arXiv:2611.07124v1 Announce Type: new  Abstract: Machine-learning models predicting live birth after in vitro fertilisation (IVF) or intracytoplasmic sperm injection (ICSI) from routine clinical data report performance from near chance to near perfect. We systematically reviewed these models under PRISMA 2020, searching OpenAlex, Scopus and Web of Science to 22 October 2026, appraised them with PROBAST+AI and TRIPOD+AI, and pooled their discrimination with random-effects meta-analysis on the logit scale. We define what fixes the meaning of a model's output (decision point, endpoint, unit of analysis, data source) and map the 48 included studies by model family, decision point and validation design. Models evaluated only internally pooled to an area under the receiver operating characteristic curve (AUC) of 0.814, against 0.756 (95% prediction interval 0.546-0.889) for models evaluated on a later period or another centre, and 0.764 for studies not at high risk of bias; neither differenc
    
[^195]: PromptOps：面向LLM-as-a-Judge系统的迭代式提示词优化

    PromptOps: Iterative Prompt Optimization for LLM-as-a-Judge Systems

    [https://arxiv.org/abs/2611.07123](https://arxiv.org/abs/2611.07123)

    本文提出PromptOps闭环框架，将LLM评判器在部署中与人工评估策略一致性下降的问题定义为“评判器漂移”，通过分离何时优化、应用何种优化流程、选用哪些反馈记录及如何更新提示词这四个运营决策，实现LLM-as-a-Judge系统的持续监控与迭代重新校准。

    

    大语言模型（LLM）越来越多地被用作自动化评判器，用于评估聊天机器人和其他AI系统的行为。然而在部署过程中，随着用户交互和评分分布的变化，它们与预期人工评估策略的一致性可能会下降。现有的自动提示词优化方法主要被作为在静态开发集上的一次性流程来研究，因此对于长期维护LLM评判器提供的指导有限。我们将这一问题定义为“评判器漂移”，并提出了PromptOps——一个用于监控和迭代重新校准LLM-as-a-Judge（LLMaaJ）系统的闭环框架。PromptOps将四个运营决策分离开来：何时进行优化、应用哪种优化流程、使用哪些反馈记录、以及如何更新评判器提示词。我们用五种提示词优化方法对该框架进行了实例化：GEPA、SIMBA、评分标准合成，以及使用……的示例选择方法。

    arXiv:2611.07123v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used as automated judges for evaluating the behavior of chatbots and other AI systems. In deployment, however, their agreement with the intended human evaluation policy may degrade as user interactions and score distributions change. Existing automatic prompt optimization methods are primarily studied as one-shot procedures on static development sets and therefore provide limited guidance for maintaining LLM judges over time. We frame this problem as judge drift and introduce PromptOps, a closedloop framework for monitoring and iteratively recalibrating LLM-as-a-Judge (LLMaaJ) systems. PromptOps separates four operational decisions: when to optimize, which optimization procedure to apply, which feedback records to use, and how to update the judge prompt. We instantiate the framework with five prompt optimization methods: GEPA, SIMBA, Rubric Synthesis, and exemplar selection using either simul
    
[^196]: 面向经验费率的信度聚类：团体人寿保险中风险组分配的自动化方法

    Credibility Clustering for Experience Rating: An Automated Approach to Risk Group Allocation in Group Life Insurance

    [https://arxiv.org/abs/2611.07110](https://arxiv.org/abs/2611.07110)

    该论文提出将多维 Bühlmann-Straub 信度模型与 k-means 聚类相结合的“信度聚类”方法，以数据驱动方式自动完成团体人寿保险中的同质风险组划分，取代传统依赖人工决策与主观阈值的分类方式。

    

    在保险业中，经验费率通过根据历史理赔经验调整保费，在缓解逆向选择方面发挥着至关重要的作用。这一过程中的关键步骤是将被保险团体划分到同质的风险组中。在团体人寿保险领域，这种划分传统上依赖于人工决策和任意设定的阈值，可能导致效率低下和结果不一致。本文提出了一种数据驱动的方法，结合多维 Bühlmann-Straub 模型与 k-means 聚类来自动化风险组分配——该方法被称为信度聚类。通过引入信度权重，该方法考虑了经验数据统计显著性的差异，确保风险分类既在精算上合理，又具备实际可操作性。基于某欧洲大型保险公司的真实团体人寿保险数据的实证结果表明……（原文摘要至此截断）

    arXiv:2611.07110v1 Announce Type: cross  Abstract: In insurance, experience rating plays a crucial role in mitigating adverse selection by adjusting premiums according to historical claims experience. A key step in this process is the classification of insured collectives into homogeneous risk groups. Within group life insurance, this classification traditionally relies on manual decisions and arbitrary thresholds, which may lead to inefficiencies and inconsistent outcomes. This paper proposes a data-driven methodology for automating risk group allocation based on a combination of the multidimensional B\"uhlmann-Straub model and k-means clustering -- referred to as credibility clustering. By incorporating credibility weights, the method accounts for the varying statistical significance of experience data and ensuresthat risk classifications are both actuarially sound and practically implementable. Empirical results based on real-world group life insurance data from a major European ins
    
[^197]: LLM模型动物园之夜：用于生成预训练大语言模型权重的数据条件流匹配

    A Night at the LLM Zoo: Data Conditioned Flow Matching for Generating Pretrained LLM Weights

    [https://arxiv.org/abs/2611.07103](https://arxiv.org/abs/2611.07103)

    本文构建了首个通过系统变化数据混合（网络文本、书籍、数学、编程、法语）训练的大规模LLM模型动物园，并提出以数据混合为条件的流匹配模型，能够在Gram-PCA压缩权重空间中直接生成完整的预训练LLM权重，且可零样本泛化到未见过的数据混合。

    

    随着权重空间学习从特定任务模型和视觉模型扩展到大语言模型（LLM），一个挑战在于缺乏经过精心策划和分析的LLM模型群体。为了解决这一问题，我们训练了这样一个模型动物园，应用于权重空间生成：我们在51M、124M和355M参数规模的GPT-2模型群体中，系统地改变网络文本、书籍、数学、编程和法语之间的训练数据混合组成，每个规模层级包含N=100个模型。我们通过余弦相似度、功能多样性、留一混合建模和方差分解来分析这些模型群体，结果表明该动物园包含多样化的模型，且模型间的变化由训练数据混合组成所结构化。我们将训练混合组成进行嵌入，以此条件化一个流匹配模型，从而在Gram-PCA压缩的权重空间中生成完整的预训练LLM权重。通过展示对保留混合的零样本泛化能力，并测量连续……

    arXiv:2611.07103v1 Announce Type: cross  Abstract: A challenge to Weight Space Learning as it expands from task specific and vision models to Large Language Models (LLMs) is the absence of a well curated and analyzed LLM population. To address this, we train such a model zoo for application to Weight Space Generation by systematically varying the data mixture composition between web text, books, mathematics, coding, and French across populations of 51M, 124M, and 355M parameter GPT-2 Models, each tier with count N = 100. We analyze these populations by cosine similarity, functional diversity, Leave-One-Mixture-Out modeling, and variance decomposition. We show that the zoo contains diverse models whose variation is structured by training mixture composition. We embed the training mixture composition to condition a Flow Matching model of complete pretrained LLM weights in Gram-PCA compressed weight space. By demonstrating zero-shot generalization to held out mixtures, measuring Continuou
    
[^198]: 面向联邦混合专家模型的服务器端稀疏专家分配

    Server-Side Sparse Assignment for Federated Mixture-of-Experts

    [https://arxiv.org/abs/2611.07093](https://arxiv.org/abs/2611.07093)

    提出S³-MoE服务器端稀疏专家分配框架，从部分观察中估计客户端特定的专家相关性并通过利用-探索机制细化专家分配，解决联邦微调LLM中边缘设备异构性带来的专家匹配难题。

    

    大型语言模型（LLM）因其卓越的能力而日益受到认可；联邦学习（FL）提供了一个有前景的框架，可在边缘设备上协作微调LLM的同时保障数据隐私。然而，边缘设备本质上具有异构性：客户端可能表现出多样化的数据分布和资源约束，包括有限的内存和计算预算。近期的联邦混合专家方法通过稀疏专家激活或部分专家加载来应对这些挑战。然而，现有的部分专家方法依赖于启发式的完整专家池观察，或优先考虑系统级的负载均衡，无法保证从部分观察中恢复客户端特定的相关专家。我们提出了S³-MoE，一个服务器端稀疏专家分配框架，它从部分观察中估计客户端特定的专家相关性，并通过“利用-探索”机制来细化专家分配……（原文摘要在此处截断）

    arXiv:2611.07093v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly recognized for their remarkable capabilities; federated learning (FL) offers a promising framework for collaboratively fine-tuning LLMs on edge devices while achieving data privacy. However, edge devices are inherently heterogeneous: clients may exhibit diverse data distributions and resource constraints, including limited memory and computation budgets. Recent federated Mixture-of-Experts (MoE) approaches address these challenges by using sparse expert activation or partial expert loading. However, existing partial-expert methods rely on heuristic full-pool observations or prioritize system-wide load balance, without guarantees for recovering client-specific relevant experts from partial observations. We propose $S^3$-MoE, a server-side sparse expert assignment framework that estimates client-specific expert relevance from partial observations and refines assignments through an exploit--expl
    
[^199]: 大规模合成人类与机器推理的符号模型

    Synthesizing symbolic models of human and machine reasoning at scale

    [https://arxiv.org/abs/2611.07088](https://arxiv.org/abs/2611.07088)

    提出行为条件化符号合成（BCSS）方法，利用AI自动搜索可解释的确定性符号程序，大规模建模人类与机器的推理行为，并揭示两者在推理中都存在过度简化和错误推断问题特征依赖关系的共同特征。

    

    符号模型为认知提供了强有力的解释框架，但其应用范围历来受到手工工程需求和计算可处理性的限制。我们展示了人工智能（AI）的进步如何能够扩展符号建模并帮助克服这些工程局限。我们提出了行为条件化符号合成——一种自动搜索可解释、确定性符号程序的方法，这些程序能够可验证地复现行为——并将其应用于对人类和机器在数百个具有挑战性的推理问题上的行为进行建模。通过分析这些程序，我们发现人类和机器都倾向于过度简化，采用仅能部分解释其观察结果的规则，并错误推断问题特征之间的依赖关系。这些共同的行为特征受可用资源的调节，推理努力程度可以预测机器简化和解释其观察结果的程度。

    arXiv:2611.07088v1 Announce Type: new  Abstract: Symbolic models offer a powerful account of cognition, but their scope has historically been limited by the need for hand-engineering and computational tractability. We show how advances in artificial intelligence (AI) can scale symbolic modeling and help overcome these engineering limitations. We introduce Behavior-Conditioned Symbolic Synthesis (BCSS)---an approach for automated search of interpretable, deterministic symbolic programs that verifiably reproduce behavior---and apply it to model both human and machine behavior across hundreds of challenging reasoning problems. Analyzing these programs reveals that both people and machines tend to oversimplify, adopt rules that are partial explanations for their observations, and misinfer dependencies between problem features. These shared signatures are modulated by available resources, with reasoning effort predicting the extent to which machines simplify and explain their observations. 
    
[^200]: 一般和博弈中的乐观Hedge算法：经由时间逼近的基本极限与遗憾界

    Optimistic Hedge in General-Sum Games: Fundamental Limits and Regret Bounds via Temporal Approximation

    [https://arxiv.org/abs/2611.07078](https://arxiv.org/abs/2611.07078)

    本文证明了一般和博弈中Optimistic Hedge能否达到与博弈规模多项式相关的常数遗憾界取决于参与者数量，并在双人博弈情形下给出了与时间无关的O(n log n)遗憾界。

    

    Optimistic Hedge（乐观Hedge），即带一步外推的乘法权重更新，是博弈中最自然的遗憾最小化动态之一，而刻画其遗憾值长期以来一直是一个悬而未决的问题。Daskalakis、Fishelson和Golowich证明，在有限一般和博弈中，若所有m个参与者均采用固定学习率的Optimistic Hedge，则每个参与者的遗憾在时间范围T上是多对数级的。随后，Ha证明了通过适当选择学习率，可使每个参与者的遗憾独立于T而保持有界。然而，这一界是非构造性的，且其对比特（行动）数量n的依赖关系并不明确，由此留下一个开放问题：Optimistic Hedge能否实现与博弈规模呈多项式关系的常数遗憾界。我们证明，答案取决于参与者的数量。在双人博弈中，存在某个学习率使得Optimistic Hedge保证与T无关的O(n log n)遗憾……

    arXiv:2611.07078v1 Announce Type: new  Abstract: Optimistic Hedge, a multiplicative weight update with one step of extrapolation, is one of the most natural regret-minimizing dynamics for games, and characterizing its regret has been a longstanding open question. Daskalakis, Fishelson and Golowich proved that if all $m$ players in a finite general-sum game follow Optimistic Hedge with a fixed learning rate, each player's regret is polylogarithmic in the horizon $T$. Subsequently, Ha showed that an appropriate choice of learning rate makes each player's regret bounded independently of $T$. This bound, however, is non-constructive and its dependence on the number of actions $n$ is not explicit, leaving open whether Optimistic Hedge can achieve a constant regret bound which is polynomial in the size of the game.   We show that the answer hinges on the number of players. In two-player games, there is a learning rate for which Optimistic Hedge guarantees $O(n\log n)$ regret independent of t
    
[^201]: 通过内部表示来塑造与监控模型

    Shaping & Monitoring Models through their Internal Representations

    [https://arxiv.org/abs/2611.07071](https://arxiv.org/abs/2611.07071)

    该论文提出“机制塑造”统一框架，在训练中通过约束模型内部表示所表达和依赖的概念来塑造模型行为，在多领域、多规模、多训练范式下显著减少不良行为同时保持任务性能，并引入监控器类别的层级体系来评估被塑造概念是否仍然可被监控。

    

    我们提出了“机制塑造”，这是一个通过模型内部表示来塑造模型的统一框架。机制塑造在训练过程中应用，使模型遵循关于它们表达哪些概念的规范（规则）（通过其输出或内部表示来衡量，例如不产生有害内容），以及它们依赖哪些概念，即它们用来做决策的依据（例如在筛选简历时不依赖候选人的性别）。跨越多个领域（视觉、医学影像、语言）、多种规模（最高达32B参数）和多种训练范式（分类、SFT、DPO、GRPO），我们发现机制塑造能在保持任务性能的同时显著减少目标行为。由于针对某一表示进行训练也可能改变该表示本身，我们提出了一个问题：被塑造的概念是否仍然可被监控，而这一问题取决于所考虑的监控器类别。因此，我们引入了此类类别的一个层级体系……

    arXiv:2611.07071v1 Announce Type: cross  Abstract: We propose Mechanistic Shaping, a unified framework for shaping models through their internal representations. Mechanistic Shaping is applied during training so that models follow specifications (rules) about which concepts they express, measured via their outputs or internal representations (e.g., no harmful content), and also which concepts they rely on, i.e., what they use to make their decisions (e.g., not a candidate's gender when screening a r\'esum\'e). Across multiple domains (vision, medical imaging, language), scales (up to 32B parameters), and training paradigms (classification, SFT, DPO, GRPO), we find that Mechanistic Shaping substantially reduces targeted behaviors while preserving task performance. Since training against a representation can also modify it, we ask whether shaped concepts remain monitorable, a question that depends on the class of monitors considered. We therefore introduce a hierarchy of such classes, wh
    
[^202]: 部分可观测马尔可夫决策过程中神经强化学习的有限时间收敛性与遗憾保证

    Finite-Time Convergence and Regret Guarantees for Neural Reinforcement Learning in POMDPs

    [https://arxiv.org/abs/2611.07069](https://arxiv.org/abs/2611.07069)

    该论文针对POMDP中采用非线性神经网络价值函数逼近的强化学习，首次建立了有限时间收敛性与遗憾保证，克服了以往理论仅适用于线性逼近或依赖不符合POMDP轨迹特性的独立同分布采样假设的局限。

    

    部分可观测马尔可夫决策过程（POMDP）中的强化学习（RL）由于对历史的长程依赖诅咒而在计算上极具挑战性。幸运的是，近期研究表明，一类满足滤波稳定性假设的POMDP可以很好地被基于有限记忆窗口内截断历史的超状态马尔可夫决策过程（MDP）所近似，其价值差距随窗口长度增加而衰减。然而，针对此类超状态MDP替代模型的现有有限时间收敛性分析，要么局限于过度简化的线性函数逼近，要么依赖于会被POMDP轨迹违背的独立同分布采样oracle，这两者都严重限制了这些理论结果在实际中的适用性。这促使我们通过研究基于非线性神经价值函数逼近的POMDP强化学习来弥合这一差距。我们的主要贡献有三个方面：i) 我们首先……（摘要在此处截断）

    arXiv:2611.07069v1 Announce Type: cross  Abstract: Reinforcement learning (RL) in Partially Observable Markov Decision Processes (POMDPs) is computationally challenging due to the curse of long-range dependence on history. Fortunately, recent studies have shown that a range of POMDPs that satisfy the filter stability assumption can be well approximated by a superstate Markov Decision Process (MDP) based on a truncated history in a finite-memory window, with a value gap that decays with the window length. However, existing finite-time convergence analyses on such superstate MDP surrogates are either limited to over-simplified linear function approximation or rely on an i.i.d. sampling oracle that is violated by POMDP trajectories, both of which severely limit the applicability of these theoretical results in practice. This motivates us to bridge the gap by investigating RL in POMDPs with nonlinear neural value function approximation. Our main contributions are threefold: i) We first est
    
[^203]: 生成式人工智能的随机动力学：从自回归模型到交互式大语言模型网络与神经随机微分方程

    Stochastic Dynamics of Generative AI: From Autoregressive Models to Interacting LLM Networks and Neural SDEs

    [https://arxiv.org/abs/2611.07065](https://arxiv.org/abs/2611.07065)

    本文以随机动力系统的统一视角刻画生成式AI的三类模型：将大语言模型建模为高维非线性自回归系统、用平均场常微分方程描述交互式LLM网络的集体行为，并将神经随机微分方程作为随机轨迹的连续时间生成模型。

    

    本文从随机动力系统的视角研究三类生成式人工智能模型。首先，我们将大语言模型（LLM）表述为具有基于注意力机制依赖关系的高维非线性自回归模型，这一表述将基于条件最大似然的预训练与序列化 token 生成联系起来，为 LLM 的运行提供了概率解释。其次，我们研究相互交互的 LLM 网络，并通过平均场常微分方程刻画其集体行为，这些动力学为理解涌现的优先连接现象和持续性结构偏差（包括依赖于类型的玻璃天花板效应）提供了框架。第三，我们将神经随机微分方程（SDE）视为针对随机轨迹（包括在不规则时间间隔上观测到的轨迹）的连续时间生成模型，并重点讨论了其训练方法。

    arXiv:2611.07065v1 Announce Type: new  Abstract: This paper develops a stochastic dynamical-systems perspective on three classes of generative AI models. First, we formulate large language models (LLMs) as high-dimensional nonlinear autoregressive models with attention-based dependencies. This formulation connects pretraining by conditional maximum likelihood to sequential token generation and provides a probabilistic interpretation of LLM operation. Second, we consider networks of interacting LLMs and characterize their collective behavior through mean-field ordinary differential equations. These dynamics provide a framework for understanding emergent preferential attachment and persistent structural biases, including type-dependent glass-ceiling effects. Third, we consider neural stochastic differential equations (SDEs) as continuous-time generative models for stochastic trajectories, including those observed at irregular time intervals. We discuss their training with particular emph
    
[^204]: LowRAM：面向测试时可组合KV缓存压缩的低秩键与摊销值方法

    LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at Test-Time

    [https://arxiv.org/abs/2611.07025](https://arxiv.org/abs/2611.07025)

    该论文提出LowRAM方法，利用注意力头内“值可由键预测”的特性，在测试时通过值缓存摊销（仅存储重构最差令牌的残差，实现4倍近无损压缩），并结合跨层低秩键压缩，构建出接近无损的可组合KV缓存压缩流水线。

    

    自回归解码依赖于KV缓存，其内存占用随上下文长度线性增长，限制了批处理大小和吞吐量。现有的训练后方法利用令牌、层和特征维度之间的冗余性，或采用降低数值精度的手段。我们识别出一个互补的维度：在每个注意力头内部，值在很大程度上可以从其对应的键中预测出来。我们通过值缓存摊销来利用这一特性，在测试时为每个注意力头拟合一个轻量级预测器，并仅为其重构效果最差的令牌存储残差，从而实现4倍近无损的值缓存压缩。该预测器可以通过元学习进行初始化以大幅降低预填充开销，也可以在允许一次性成本的场景（如多查询推理）中从头拟合。我们将值摊销与跨层低秩键压缩相结合，构建出LowRAM——一种保持接近无损的非对称压缩流水线。

    arXiv:2611.07025v1 Announce Type: cross  Abstract: Autoregressive decoding relies on the KV cache, whose memory footprint grows linearly with context length and caps batch size and throughput. Existing post-training methods exploit redundancy across tokens, layers, and feature dimensions, or lower numerical precision. We identify a complementary axis: within each attention head, values are largely predictable from their corresponding keys. We exploit this with value cache amortisation, which fits a lightweight per-head predictor at test time and stores residuals only for the tokens it reconstructs worst, giving $4\times$ near-lossless value compression. The predictor can be initialised by meta-learning for substantially reduced prefill overhead, or fitted from scratch in settings that allow a one-off cost, such as multi-query inference. We pair value amortisation with cross-layer low-rank key compression to produce LowRAM, an asymmetric compression pipeline that stays close to lossless
    
[^205]: 使用时间中和决策树的鲁棒流分类

    Robust Stream Classification using Time Neutralising Decision Trees

    [https://arxiv.org/abs/2611.06991](https://arxiv.org/abs/2611.06991)

    本文提出了一种名为“时间中和树（TNT）”的决策树架构，通过中和时间因素的影响，在存在概念漂移且实例可被存储的数据流场景下实现鲁棒的流分类。

    

    数据流的特点是一种可能无限的、且具有非平稳性的数据来源。这种非平稳性也被称为概念漂移，它常常导致已部署的预测模型不得不随着流底层数据生成过程随时间变化而被不断地重新训练。漂移检测器是一类用于识别这些变化的方法，用以触发这些重新训练过程。历史上，流学习一直运行在“实例只会被看到一次、无法存储以供后续使用”的假设之下。当代的工作放宽了这一假设，但漂移检测和简单的重置程序仍然无处不在。基于相关工作（其表明当实例可以被存储时，传统的批量学习算法往往优于流学习算法），我们提出了时间中和树（Time Neutralising Trees, TNT），这是一种决策树架构，能够在相关设置下实现鲁棒的流分类。

    arXiv:2611.06991v1 Announce Type: new  Abstract: Data Streams are characterised as a potentially infinite source of data subject to non-stationarity. This non-stationarity, also called Concept Drift, often results in deployed predictive models having to be perpetually retrained as the underlying data generating process of the stream changes over time. Drift Detectors are a class of methods used to identify these changes, triggering these retraining procedures. Historically, Stream Learning has operated under the assumption that instances are only seen once, and cannot be stored for later use. Contemporary work relaxes this assumption, yet drift detection and simple resetting procedures are still ubiquitous.   Building on related work which has shown that traditional Batch Learning algorithms often outperform Stream Learning algorithms when instances can be stored, we introduce Time Neutralising Trees (TNT), a Decision Tree architecture that enables robust stream classification in setti
    
[^206]: ToroFit：用环面的楔和拟合高维数据

    ToroFit: Fitting Wedge Sums of Tori to High-Dimensional Data

    [https://arxiv.org/abs/2611.06990](https://arxiv.org/abs/2611.06990)

    本文提出ToroFit框架，利用持续上同调与上积识别高维点云数据中各环面结构分量，并通过将低能量多元三角多项式拟合归结为广义岭回归，实现对环面楔和型非线性数据的高效检测与参数化。

    

    科学应用中产生的数据往往呈现出复杂的非线性结构。识别并参数化这些非线性结构可以改进建模，增强预测分析、数据插值、去噪和异常检测能力。在本文中，我们开发了理论与计算方法，用于检测并参数化位于环面楔和附近的欧氏点云数据。具体而言，我们利用持续上同调和上积来识别不同的环面分量，并在每个分量上通过优化低能量多元三角多项式来获得最佳数据拟合。由此产生的优化问题可归结为广义岭回归，可借助标准技术高效求解。我们在合成数据集和真实数据集上展示了所提框架的实用性，包括低帧率循环视频的插值、准周期动力系统中吸引子的恢复等。

    arXiv:2611.06990v1 Announce Type: cross  Abstract: Data arising in scientific applications often exhibits complex nonlinear structure. Identifying and parameterizing said nonlinearities can enable better modeling, enhance predictive analysis, data interpolation, denoising, and anomaly detection. In this paper, we develop theoretical and computational methods to detect and parametrize Euclidean point cloud data lying near a wedge sum of tori. Specifically, we use persistent cohomology and cup products to identify the distinct toroidal components, and on each component, we optimize over low-energy multivariate trigonometric polynomials for best data fit. The ensuing optimization problem reduces to generalized ridge regression, which can be solved efficiently via standard techniques. We demonstrate the utility of the proposed framework in both synthetic and real datasets, including interpolation of low frame-rate recurrent videos, attractor recovery in quasiperiodic dynamical systems, and
    
[^207]: 大模型，小收益：MONSTER 基准上作为分类器的冻结预测模型

    Big Models, Small Gains: Frozen Forecasters as Classifiers on MONSTER

    [https://arxiv.org/abs/2611.06988](https://arxiv.org/abs/2611.06988)

    该论文在 MONSTER 基准上对冻结的预测基础模型（Reverso 和 Chronos-Bolt）表示进行受控评测，发现预训练相比随机权重模型仅带来小幅且不稳定的分类准确率提升，表明这些预训练表示未必优于无需预训练的强分类器，尤其在标注稀缺时其价值存疑。

    

    预测基础模型越来越多地被用作时间序列分类的冻结特征提取器，但它们的预训练表示是否比无需预训练的强分类器具有优势仍不清楚。这引出了一个重要问题：这些表示是否提供了模型架构本身所能捕获之外的有用信息，尤其是在标注数据稀缺的情况下？我们在 MONSTER 基准的六个大型数据集上，分别使用可用训练标签的 1%、10% 和 100%，对 Reverso 和 Chronos-Bolt 两个预测模型进行了受控基准测试。我们将它们的冻结表示与四种成熟的时间序列分类方法以及相同预测模型的随机权重版本进行比较。结果表明，在六个数据集中的五个上，预训练比随机权重的对应模型提升了分类准确率，收益最高达 23 个百分点……（原文截断）

    arXiv:2611.06988v1 Announce Type: cross  Abstract: Forecasting foundation models are increasingly used as frozen feature extractors for time series classification, but it remains unclear whether their pretrained representations offer an advantage over strong classifiers that do not require pretraining. This raises an important question: do these representations provide useful information beyond what the model architecture alone can capture, particularly when labelled data are scarce? We present a controlled benchmarking of two forecasting models, Reverso and Chronos-Bolt, on six large datasets from the MONSTER benchmark, using 1%, 10%, and 100% of the available training labels. We compare their frozen representations against four established time series classification methods and random-weight versions of the same forecasting models. Our results show that pretraining improves classification accuracy over random-weight counterparts on five of the six datasets, with gains of up to 23 per
    
[^208]: Octopus：一个大脑，众多分支——面向潜在KV多智能体强化学习

    Octopus: One Brain, Many Branches for Latent-KV Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.06956](https://arxiv.org/abs/2611.06956)

    Octopus提出了一种单模型多智能体强化学习框架，让所有分支共享同一主干网络，并通过零拷贝基数分叉、仅注意力KV挂载等协同设计机制，在GPU驻留的KV状态有向无环图上进行高效协调，从而消除KV缓存重复复制和跨分支依赖丢失的问题。

    

    多智能体LLM系统通过生成具有隔离上下文窗口的子智能体来并行探索，而单智能体系统则将每次搜索和验证步骤追加到一条不断增长的轨迹中。然而，现有多智能体框架会在各分支间重复复制前缀KV缓存，通过重新预填充文本摘要来交换发现结果，并在强化学习（RL）过程中将跨分支依赖关系扁平化或丢弃。我们提出了Octopus，这是一个单模型多智能体强化学习框架，其中所有线程共享同一个主干网络，并通过五个原语（fork、read_thread、wait_threads、rewind、commit）在驻留于GPU的有向无环图（DAG）KV状态上进行协调。Octopus结合了三个协同设计的机制：(1) 零拷贝基数分叉，将分支生成为共享物理KV页面上的O(1)复杂度游标视图；(2) 仅注意力KV挂载，将同级线程的末端KV块直接挂载到全注意力层中……（摘要在此处被截断）

    arXiv:2611.06956v1 Announce Type: cross  Abstract: Multi-agent LLM systems explore in parallel by spawning sub-agents with isolated context windows, whereas single-agent systems append every search and verification step into one growing trajectory. However, existing multi-agent frameworks duplicate prefix KV caches across branches, exchange findings by re-prefilling text summaries, and flatten or drop cross-branch dependencies during reinforcement learning (RL).   We introduce Octopus, a single-model multi-agent RL framework in which all threads share one backbone and coordinate over a GPU-resident directed acyclic graph (DAG) of KV states through five primitives (fork, read_thread, wait_threads, rewind, commit). Octopus combines three co-designed mechanisms: (1) Zero-Copy Radix Forking, which spawns branches as $\mathcal{O}(1)$ cursor views over shared physical KV pages; (2) Attention-Only KV Mounting, which mounts a sibling thread's terminal KV blocks directly into full-attention lay
    
[^209]: SteelAgent：由大语言模型编排、融合物理信息的钢性能预测，具备可审计的泛化能力与经校准的不确定性

    SteelAgent: LLM-Orchestrated, Physics-Informed Steel Property Prediction with Audited Generalization and Calibrated Uncertainty

    [https://arxiv.org/abs/2611.06953](https://arxiv.org/abs/2611.06953)

    SteelAgent 将大语言模型编排、冶金描述符与机器学习相结合用于钢性能预测，并揭示评估时留出单元的选择会彻底改变结论——随机划分下表现优异的模型在留出陌生钢族或数据源时性能骤降且不确定性校准失效。

    

    钢性能模型在随机测试集划分上可能表现良好，即使其对陌生合金的外推能力很差。SteelAgent 将冶金学描述符与机器学习模型和语言模型接口相结合，用于性能预测、不确定性报告和逆向设计。我们研究了留出单元的选择如何改变基于 SteelBench 抗拉强度评估所得出的结论。对于梯度提升模型，合并的折外 R² 从随机划分下的 0.859 下降到留出钢族时的 0.113。名义 90% 预测区间的经验覆盖率从随机划分下的 91.0% 变化到留出数据源时的 37.6%。物理衍生特征的效果取决于评估方式：在 17 个钢族的留出分析中，它们使梯度提升的 R² 提高了 0.2534，而在另一项针对四个牌号的实验室分析中，却使其降低了 0.358。我们报告了组级误差以及一项合成回归实验的结…（摘要在此处截断）

    arXiv:2611.06953v1 Announce Type: new  Abstract: Steel property models can perform well on random test splits even when they extrapolate poorly to unfamiliar alloys. SteelAgent couples metallurgical descriptors and machine-learning models with a language-model interface for property prediction, uncertainty reporting and inverse design. We examine how the choice of held-out unit changes the conclusions drawn from SteelBench tensile-strength evaluations. For gradient boosting, pooled out-of-fold $R^2$ fell from 0.859 on a random split to 0.113 when steel families were held out. Empirical coverage of nominal 90% prediction intervals ranged from 91.0% under random splitting to 37.6% on held-out sources. The effect of physics-derived features depended on the evaluation. They increased gradient-boosting $R^2$ by 0.2534 in a 17-family holdout analysis and decreased it by 0.358 in a separate four-grade laboratory analysis. We report group-level errors and a synthetic regression experiment illu
    
[^210]: BrainPFN：基于先验数据拟合网络的摊销式大脑-行为预测

    BrainPFN: Amortized Brain-Behaviour Prediction via Prior-Data Fitted Networks

    [https://arxiv.org/abs/2611.06934](https://arxiv.org/abs/2611.06934)

    提出BrainPFN——一种在由约7千个真实功能连接矩阵构建的领域先验上进行元训练的先验数据拟合网络，通过摊销式推断实现可跨数据集迁移、在小样本临床/认知场景下依然准确的大脑-行为预测，克服了核岭回归需逐评分调参、无法迁移的局限。

    

    arXiv:2611.06934v1 公告类型：交叉。从功能磁共振成像预测行为表型的标准方法是以在功能连接（FC）矩阵上应用核岭回归（KRR）为代表的。然而这种方法存在局限：必须为每个评分和队列单独拟合与调优模型，它们无法跨数据集迁移，而且当仅有少量受试者可用时——这在临床或认知研究中十分常见——准确度会急剧下降。尽管人们对开发大脑基础模型的兴趣日益增长，但尚无模型能在行为预测上持续超越KRR；大多数报告的性能提升来自更容易的目标，如年龄、性别或诊断，而非连续的行为评分。我们提出BrainPFN，一个用于大脑行为回归的先验数据拟合网络（Prior-Data Fitted Network）。BrainPFN在基于约7千个真实FC矩阵构建的领域特定先验上进行元训练，并从中采样了十亿个合成训练样本。在推理阶段，一个有标签的……

    arXiv:2611.06934v1 Announce Type: cross  Abstract: The standard approach to predicting behavioral phenotypes from functional MRI is represented by Kernel Ridge Regression (KRR) applied on functional connectivity (FC) matrices. However, this approach is limited: separate models must be fit and tuned for every score and cohort, they do not transfer across datasets, and accuracy degrades sharply when only a handful of subjects are available, as is typical of clinical or cognitive studies. Despite an increasing interest in developing brain foundation models, none has consistently outperformed KRR on behaviour prediction; most reported gains come from easier targets such as age, sex, or diagnosis rather than continuous behavioural scores. We introduce BrainPFN, a Prior Data Fitted Network for brain behaviour regression. BrainPFN is meta-trained on a domain-specific prior built from ~7k real FC matrices, from which we sample one billion synthetic training samples. At inference, a labelled co
    
[^211]: 利用神经算子大规模捕捉人类神经元的膜动力学与锋电位时序

    Capturing Membrane Dynamics and Spike Timing of Human Neurons at Scale using Neural Operators

    [https://arxiv.org/abs/2611.06933](https://arxiv.org/abs/2611.06933)

    该论文提出一个细胞类型特异性的神经算子框架，通过在同类多个神经元模型上学习共享算子并以可解释的电生理特征为条件，实现大规模神经元群体建模，同时保留丰富的膜动力学、锋电位时序及细胞类型内变异性。

    

    表征神经元细胞类型的特性是理解大脑功能的核心。然而，具有生物物理真实性的神经元模型难以扩展到大规模神经元群体，同时又能保持细胞动力学的丰富性以及细胞类型内部的变异性。我们提出了一个细胞类型特异性的神经算子框架，用于可扩展的神经元群体建模。该框架并非为每个单独的神经元训练独立的替代模型，而是在来自同一细胞类型的多个建模神经元上学习一个共享算子，并以生物学上可解释的电生理特征作为条件。相对于原始的单神经元NOBLE设置，跨异质神经元的学习需要更丰富的电生理条件表示，以及一个旨在保留多样锋电位动力学的训练目标。这使得能够高效生成大规模的胞体电压响应集合，同时……

    arXiv:2611.06933v1 Announce Type: cross  Abstract: Characterizing the properties of neuronal cell types is central to understanding brain function. Yet, biophysically realistic neuron models remain difficult to scale to large populations while preserving the richness of cellular dynamics and within-cell-type variability. We present a cell-type-specific neural-operator framework for scalable neuronal population modeling. Rather than training separate surrogate models for individual neurons, the proposed framework learns a shared operator across multiple modeled neurons from the same cell type, conditioned on biologically interpretable electrophysiological features. Relative to the original single-neuron \texttt{NOBLE} setting, learning across heterogeneous neurons requires a richer electrophysiological conditioning representation and a training objective designed to preserve diverse spiking dynamics. This enables efficient generation of large ensembles of somatic voltage responses while
    
[^212]: 从神经元与其他神经元的相关性中读取其调谐特性

    Reading a neuron's tuning from its correlations with other neurons

    [https://arxiv.org/abs/2611.06927](https://arxiv.org/abs/2611.06927)

    仅利用神经元群体的相关矩阵（不含刺激信息、无标签），transformer模型即可解码小鼠视觉皮层神经元的偏好朝向和感受野位置，表明神经元间的相关结构本身已蕴含其调谐特性信息。

    

    一个神经元所表征的内容，通常是通过将其活动与刺激相关联来确定的。我们探究同样的事实是否包含在所记录神经群的相关矩阵中——该矩阵保留了神经元之间的成对关系，而去除了刺激信息。一个只看到512个采样神经元相关矩阵的transformer模型（其输入中不含标签，也没有参考群体），从在体MICrONS响应中解码小鼠视觉皮层神经元的偏好朝向，平均误差为25°（无信息基线：45°；最佳常数猜测：41°），并以R² = 0.23的精度解码其感受野位置；在数字孪生基底上，误差为20°，R²为0.48。当被问及某个神经元偏好水平还是垂直朝向时，该模型在在体数据上对79%的神经元判断正确，在孪生数据上为87%，而多数回答的基线仅为56%；但它无法分辨倾斜朝向。这种排序之所以可以被恢复，是因为各朝向类别之间的关系是相近的……

    arXiv:2611.06927v1 Announce Type: cross  Abstract: What a neuron represents is usually established by relating its activity to a stimulus. We ask whether the same facts are contained in the correlation matrix of a recorded population, which keeps the pairwise relations among the neurons and discards the stimulus. A transformer that sees only the correlation matrix of 512 sampled neurons, with no labels and no reference population in its input, decodes the preferred orientation of mouse visual cortex neurons from in-vivo MICrONS responses to a mean error of 25{\deg} (uninformed: 45{\deg}; best constant guess: 41{\deg}) and their receptive-field position at R^2 = 0.23; on the digital-twin substrate, 20{\deg} and 0.48. Asked whether a neuron prefers horizontal or vertical, it is right for 79% of neurons in vivo and 87% on the twin, against 56% for the majority answer; it does not resolve the obliques. The ordering is recoverable because the relations between orientation classes are close 
    
[^213]: 从固定默认配置到经过校准的激活策略：面向长期时间序列预测

    From Fixed Defaults to Calibrated Activation Policies for Long-Term Time-Series Forecasting

    [https://arxiv.org/abs/2611.06559](https://arxiv.org/abs/2611.06559)

    该研究提出一种经过校准的激活选择策略，能够根据任务自适应地选择激活函数及其位置，并在证据不足时保留默认配置，在1,225次Informer训练实验中平均将长期时间序列预测的均方误差降低7.86%。

    

    长期时间序列预测为能源、交通等应用的决策提供支持，这些应用需要在观测数据范围之外获得可靠的预测结果。尽管预测研究在模型架构方面取得了进展，但固定的激活配置对于激活选择应如何随数据集和预测时域变化所提供的指导十分有限。本研究旨在确定任务特定的激活选择何时能够提升预测精度，以及如何利用先前的实验结果来指导为新任务选择合适的配置。我们开发了一种经过校准的选择策略，该策略从先前实验中学习一个可靠的默认配置，针对每个任务自适应地调整激活函数及其位置，并在证据不足时保留默认配置。在一项由1,225次Informer训练运行支持的评估中，涵盖17个数据集变体和51个预测任务，该策略平均将均方误差降低了7.86%。

    arXiv:2611.06559v1 Announce Type: new  Abstract: Long-term time-series forecasting supports decisions in energy, transportation, and other applications where reliable predictions are needed beyond the observed data. Although forecasting research has advanced model architectures, fixed activation configurations provide limited guidance on how activation choices should vary across datasets and prediction horizons. This study aims to establish when task-specific activation choices improve forecasting accuracy and how previous experiments can guide the selection of suitable configurations for new tasks. We develop a calibrated selection policy that learns a reliable default from prior experiments, adapts the activation function and its location to each task, and retains the default when evidence for a change is insufficient. In an evaluation supported by 1,225 Informer training runs across 17 dataset variants and 51 forecasting tasks, the policy reduces mean squared error by 7.86% on avera
    
[^214]: 对齐代价是分级的：SFT保留下长度归一化DPO的任务选择性能力损失

    The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention

    [https://arxiv.org/abs/2611.06476](https://arxiv.org/abs/2611.06476)

    对齐微调带来的能力损失并非均匀发生，而是分级且任务选择性的——抽取类任务几乎不受影响，JSON生成轻微下降，而路由任务出现约30分的严重退化。

    

    设定。这是一项实证特征描述研究；我们未提出新方法，也未与其他目标函数进行对比。我们在结构化财务报告任务上，使用稀疏LoRA和长度归一化DPO加上SFT保留项（即RPO目标）对Qwen3.5-4B进行微调，并跟踪下游能力随隐式奖励边际增长至约7个β缩放单位（nf4量化训练；在未量化的基础模型上以fp16测量时约为3）时的变化。问题。边际增长是否会在所有任务上均匀地降低能力，还是具有选择性？发现。在一项预注册的研究（8个随机种子，其中6个在冻结前未使用）中，通过伪影感知评分和TOST等效性检验，对齐代价是分级的且具有任务选择性：信息抽取能力得以保留（Δ = -0.55分，在±5分范围内等效），json能力损失很小（Δ = -3.52分，在聚类后的项目级区间上等效性检验未通过），而路由能力严重退化（Δ = -30分）。

    arXiv:2611.06476v1 Announce Type: new  Abstract: Setting. This is an empirical characterization study; we propose no new method and run no comparisons against alternative objectives. We fine-tune Qwen3.5-4B with sparse LoRA and length-normalized DPO plus an SFT retention term (the RPO objective) on structured financial-filing tasks, and track downstream capability as the implicit reward margin grows to $\sim 7$ $\beta$-scaled units (nf4 training; $\sim 3$ under fp16 measurement on the unquantized base). Question. Does margin growth degrade capability uniformly across tasks, or selectively? Findings. In a pre-registered study (8 seeds, 6 unseen at freeze) with artifact-aware scoring and TOST equivalence testing, the alignment tax is graded and task-selective: extraction is preserved ($\Delta = -0.55$ pts, equivalent at $\pm5$ pts), json loses little ($\Delta = -3.52$ pts, equivalence fails on a clustered item-level interval), and router degrades severely ($\Delta = -30$ pts). Router's l
    
[^215]: 面向无目标故障数据的数据稀缺资产的跨资产知识迁移早期故障检测方法

    Cross-Asset Knowledge Transfer for Early Fault Detection of Data-Scarce Assets Without Target Fault Data

    [https://arxiv.org/abs/2611.06194](https://arxiv.org/abs/2611.06194)

    本文提出一个零样本域适应框架，在目标资产完全没有故障数据的情况下，仅利用其正常数据即可通过EWMA自动标注源资产数据并借助改进的域对抗神经网络实现跨资产早期故障检测。

    

    工业机队的可靠性管理依赖于早期故障预警，然而相似资产之间的生命周期数据高度不平衡。少数资产积累了从运行到失效的完整历史数据，而新安装的或很少发生故障的资产仅能提供正常工况下的数据。本文研究的问题是：当目标资产上尚未观测到任何故障数据时，如何将诊断知识从数据丰富的源资产迁移到数据稀缺的目标资产。我们将跨资产早期故障检测形式化为一个零样本域适应问题，其中“零样本”指的是目标资产故障样本的缺失，并提出一个仅需目标资产正常数据的框架。该框架包含：(1) 一个统计过程控制步骤，利用指数加权移动平均（EWMA）控制图对源资产的运行至失效数据进行自动标注，将人工标注工作简化为对被标记片段的验证；(2) 一个改进的域对抗神经网络（ADAN……

    arXiv:2611.06194v1 Announce Type: new  Abstract: Reliability management of industrial fleets depends on early fault warnings, yet lifecycle data across similar assets are highly imbalanced. A few assets accumulate run-to-failure histories, while newly installed or rarely failing assets provide only normal-condition data. This paper asks how diagnostic knowledge can be transferred from a data-rich source asset to a data-scarce target asset when no fault data have been observed on the target. We formulate cross-asset early fault detection as a zero-shot domain adaptation problem, in which zero-shot refers to the absence of target fault samples, and propose a framework that requires only normal data from the target. The framework comprises (1) a statistical process control step that labels the source asset run-to-failure data with an exponentially weighted moving average chart, reducing manual annotation to verifying flagged segments, (2) an Adapted Domain-Adversarial Neural Network (ADAN
    
[^216]: 用于侧向分子扩散模拟的神经随机预测方法

    Neural Stochastic Forecasting for Lateral Molecular Diffusion Simulations

    [https://arxiv.org/abs/2611.06157](https://arxiv.org/abs/2611.06157)

    提出了一种结合Transformer构象表示、多时间尺度循环网络和带学习记忆的随机潜过程的神经随机模型，能够从粗粒化分子动力学轨迹高效预测膜蛋白在膜内的长时间集体扩散。

    

    预测膜蛋白的长时间扩散具有挑战性，因为分子动力学（MD）模拟的计算成本高昂，而实验中的单粒子跟踪通常又受限于有限的观察时间。在此，我们提出了一种神经随机（NS）模型，用于从粗粒化MD轨迹预测膜蛋白的集体扩散。该模型结合了基于Transformer的蛋白质构象表示、在多个时间尺度上运行的循环网络，以及具有学习记忆功能的随机潜过程。随后，交叉注意力解码器在考虑系统瞬时几何构型和周期性边界条件的情况下，生成相关的蛋白质位移。我们在两个膜系统中对该模型进行了评估：一个是七个相互作用的PIEZO通道系统，另一个是包含七个PIEZO通道和十个TREK通道的混合系统。对于七PIEZO系统，NS模型重现了异常（扩散行为）……

    arXiv:2611.06157v1 Announce Type: new  Abstract: Predicting the long-time diffusion of membrane proteins is challenging because molecular dynamics (MD) simulations are computationally expensive, while experimental single-particle tracking is often limited by finite observation times. Here, we introduce a neural stochastic (NS) model for forecasting the collective diffusion of membrane proteins from coarse-grained MD trajectories. The model combines a transformer-based representation of protein configurations, recurrent networks operating at multiple temporal scales, and a stochastic latent process with learned memory. A cross-attention decoder then generates correlated protein displacements while accounting for the instantaneous system geometry and periodic boundary conditions. We evaluate the model in two membrane systems: seven interacting PIEZO channels and a mixed system containing seven PIEZO and ten TREK channels. For the seven PIEZO system, the NS model reproduces the anomalous 
    
[^217]: 奖励预算化：减少大语言模型推理强化学习中的过早收敛

    Reward Budgeting Reduces Premature Convergence in Reinforcement Learning for LLM Reasoning

    [https://arxiv.org/abs/2611.05942](https://arxiv.org/abs/2611.05942)

    该论文提出“奖励预算化”这一插件式优势构造方法，通过为每个提示分配固定的总奖励预算来缩小优势差距，从而有效缓解大语言模型推理强化学习中的过早收敛与早期性能平台期问题。

    

    面向大语言模型（LLM）推理的强化学习（RL）常常随着策略分布变得越来越尖锐而过早进入性能平台期。近期研究将策略熵作为这一行为的诊断信号和控制信号。然而我们的实验表明，仅仅增加熵并不能可靠地改变最终的平台期。我们通过RL的隐式目标策略对其进行分析，推导出一个尖锐度因子 $\kappa$ 及其有效对应量 $\kappa_{\mathrm{eff}}$，用以刻画奖励或优势差距被转化为概率比差距的强度。基于这一视角，我们提出了“奖励预算化”，这是一种插件式的优势构造方法，为每个提示分配固定的总奖励预算。对于二元奖励和独立的on-policy采样，其未截断的得分函数估计器遵循一个精确的、具有收益递减特性的有限-$G$ 凹目标的梯度，而其优势差距随着更多采样的进行而减小……

    arXiv:2611.05942v1 Announce Type: new  Abstract: RL for LLM reasoning often reaches an early performance plateau as the policy distribution becomes increasingly sharp. Recent work has studied policy entropy as both a diagnostic and a control signal for this behavior. In our experiments, however, increasing entropy alone does not reliably change the eventual plateau. We analyze RL through its implicit target policy and derive a sharpness factor $\kappa$, together with an effective counterpart $\kappa_{\mathrm{eff}}$, that describes how strongly reward or advantage gaps are translated into probability-ratio gaps. Motivated by this view, we introduce \emph{reward budgeting}, a plug-in advantage construction that assigns a fixed total reward budget to each prompt. For binary rewards and independent on-policy rollouts, its unclipped score-function estimator follows the gradient of an exact finite-$G$ concave objective with diminishing returns, while its advantage gap decreases as more rollo
    
[^218]: ZipDrift：利用一步漂移模型生成多模态短时域MPC代价

    ZipDrift: Generating Multimodal Short-Horizon MPC Costs with One-Step Drifting Models

    [https://arxiv.org/abs/2611.05793](https://arxiv.org/abs/2611.05793)

    该论文提出ZipDrift，通过训练一个向低闭环代价倾斜的一步漂移生成器来产生多模态的MPC代价修正候选，再由学习到的评论家排序选优，解决了确定性回归在多模态场景下平均各模态或继承专家不一致平局决策的问题。

    

    短时域模型预测控制（MPC）速度快但目光短浅。ZipMPC通过学习一种依赖于上下文的短时域代价修正来弥补这一缺陷，该修正模仿长时域专家的行为。我们观察到最优修正往往是多模态的：当较慢的对手既可从左侧也可从右侧超越时，符号相反的良好修正同时存在，而专家在两者之间的选择取决于微小的代价差异。在这种情况下，确定性回归会平均各个模态，或继承专家不一致的平局抉择。我们提出ZipDrift，它在MPC代价修正上训练一个一步漂移生成器，该生成器向低闭环代价倾斜，且仅通过候选修正的代价评估来学习，无需对优化器进行微分。少量生成的候选由学习到的评论家进行排序，随后基于模型的MPC强制执行动力学和约束。在从ZipMPC衍生出的运动学超车基准测试中，53%的上下文是双模态的……

    arXiv:2611.05793v1 Announce Type: cross  Abstract: Short-horizon model predictive control (MPC) is fast but myopic. ZipMPC compensates by learning a context-dependent correction of the short-horizon cost that imitates a long-horizon expert. We observe that the optimal correction is often multimodal: when a slower opponent can be passed on either side, good corrections of opposite sign coexist, and the expert's choice between them hinges on tiny cost differences. Deterministic regression then averages modes or inherits inconsistent expert tie-breaking. We propose ZipDrift, which trains a one-step drifting generator over MPC cost corrections, tilted toward low closed-loop cost and learned only from cost evaluations of candidate corrections, without differentiating through the optimiser. A few generated candidates are ranked by a learned critic, and the model-based MPC then enforces dynamics and constraints. In a kinematic overtaking benchmark derived from ZipMPC, 53% of contexts are bimo
    
[^219]: RESHPart：面向超图划分的反馈重塑谱嵌入

    RESHPart: Feedback-Reshaped Spectral Embedding for Hypergraph Partitioning

    [https://arxiv.org/abs/2611.05729](https://arxiv.org/abs/2611.05729)

    该论文提出FRESHPart框架，从谱嵌入几何出发，通过几何反馈和搜索反馈重塑谱嵌入，使其更有利于基于投影的超图划分，并在Titan23基准上相比hMETIS显著降低割值。

    

    随着可扩展的拉普拉斯求解器的出现，下一个前沿是如何更有效地利用它们以获得高质量的超图划分。在本工作中，我们从谱嵌入几何的视角重新审视谱超图划分。通过对电路超图和受控合成（超）图的系统性分析，我们识别出一种射线状（ray-like）嵌入结构，这有助于解释基于投影的划分方法的有效性。我们进一步表明，谱嵌入的几何结构可以通过拉普拉斯矩阵设计和超边重加权被系统地影响。基于这些见解，我们提出了FRESHPart，一个通过几何反馈和搜索反馈来精化谱嵌入的反馈重塑谱划分框架，从而产生与基于投影的划分更加对齐的嵌入。在Titan23基准测试上，FRESHPart 相比 hMETIS 实现了平均割的降低（原文在此处截断）。

    arXiv:2611.05729v1 Announce Type: new  Abstract: With scalable Laplacian solvers now available, the next frontier is to exploit them more effectively for high-quality hypergraph partitioning. In this work, we revisit spectral hypergraph partitioning from the perspective of spectral embedding geometry. Through a systematic analysis of circuit hypergraphs and controlled synthetic (hyper)graphs, we identify a ray-like embedding structure that helps explain the effectiveness of projection-based partitioning. We further show that the geometry of spectral embeddings can be systematically influenced through Laplacian design and hyperedge reweighting. Guided by these insights, we propose FRESHPart, a feedback-reshaped spectral partitioning framework that refines the spectral embedding through Geometry Feedback and Search Feedback, producing embeddings that are better aligned with projection-based partitioning. On the Titan23 benchmark, FRESHPart achieves average cut reductions over hMETIS of 1
    
[^220]: 利用机器学习天气模型区分初始条件误差与模型误差

    Distinguishing Initial-Condition and Model Error Using Machine-Learning Weather Models

    [https://arxiv.org/abs/2611.05592](https://arxiv.org/abs/2611.05592)

    本研究通过在GraphCast和盘古气象大模型上对比门控与非门控优化方法，利用机器学习天气模型实现初始条件误差与模型误差的区分。

    

    在GraphCast和盘古气象大模型中评估了门控与非门控优化方法。第一张图（左上）显示了36个日期的平均结果；其余图显示了2017年1月至11月每月首日12 UTC的优化下降过程。纵轴指标为相对于ERA5验证的第1-10天Z500均方根误差（RMSE）之和，以各模型相对于日期匹配对照组的百分比变化表示。门控分支将初始条件与LoRA权重校正进行联合优化；其GraphCast曲线仅使用初始条件（去除LoRA），即盘古模型接收到的相同状态。星号和菱形分别标记了5天和10天窗口内盘古的最低误差。每张图上方的百分比显示了门控（绿色）和非门控（红色）优化相对于对照组的盘古最佳改进。当提前停止优化时，非门控方法在一半的日期上优于门控方法。门控并不能保证盘古模型性能更好，但它确实能产生更可比较的GraphCast和（原文在此截断）

    arXiv:2611.05592v1 Announce Type: cross  Abstract: Gated and ungated optimizations evaluated in GraphCast and Pangu-Weather. The first panel (upper-left) shows the mean over the 36 dates; remaining panels show first-of-month 12 UTC optimization descents from January-November 2017. The y-axis metric is the summed day 1-10 Z500 RMSE verified against ERA5, expressed as percent change from each model's date-matched control. The gated arm optimizes the initial conditions jointly with a LoRA weight correction; its GraphCast curve uses the initial conditions alone (LoRA removed), the same state Pangu receives. Stars and diamonds mark the lowest Pangu errors in the 5- and 10-day windows, respectively. Percentages above each panel show the best Pangu improvement relative to control for gated (green) and ungated (red) optimization. Ungated outperforms gated on half of dates when stopped early. Gating does not guarantee a better Pangu performance, but it does produce more comparable GraphCast and
    
[^221]: 统一并超越：从特征干扰视角看大语言模型后训练SFT中普遍性能力退化的理论

    Unifying and Beyond the Theory of Pervasive Capability Degradation in LLM Post-training SFT from Feature Interference

    [https://arxiv.org/abs/2611.05003](https://arxiv.org/abs/2611.05003)

    本文提出了一个基于叠加态下特征干扰的自底向上几何理论，从微观机制解释了大语言模型SFT中普遍性能力退化的成因，并将现有缓解方法统一为六种范式，同时揭示了其有效性的来源与局限。

    

    普遍性能力退化是指监督微调在使大语言模型适应目标任务的过程中，往往会侵蚀模型在预训练阶段所习得的通用能力的现象。主流解释将PCD归因于宏观的分布偏移，因而缺乏微观层面的可操作性，导致现有的缓解方法缺乏完整的理论基础。在本文中，我们首先基于叠加态下的特征干扰，建立了一个自底向上的PCD几何理论：窄域SFT数据破坏了预训练期间形成的近正交特征结构，由此产生的干扰使激活值越过输出Laguerre-Voronoi几何中的决策超平面，从而翻转通用任务上的预测。该理论将现有方法统一为六种范式，并揭示了这些方法有效性的来源及其局限性。随后我们主张……（摘要在此处被截断）

    arXiv:2611.05003v2 Announce Type: replace  Abstract: Pervasive Capability Degradation (PCD) refers to the phenomenon in which supervised fine-tuning (SFT), while adapting a large language model (LLM) to a target task, often erodes the general capabilities acquired during pre-training. Prevailing explanations attribute PCD to macroscopic distribution shift and therefore lack microscopic manipulability, leaving existing mitigation methods without a complete theoretical foundation. In this paper, we first develop a bottom-up geometric theory of PCD grounded in feature interference under superposition: narrow-domain SFT data disrupt the almost-orthogonal feature structure formed during pre-training, and the resulting interference drives activations across decision hyperplanes in the output Laguerre-Voronoi geometry, flipping predictions on general tasks. This theory unifies existing methods into six paradigms and reveals both the source of their efficacy and their limitations. We then argu
    
[^222]: P$^2$-MOPD：通过参数先验加速多教师在线策略蒸馏

    P$^2$-MOPD: Accelerating Multi-Teacher On-Policy Distillation via Parameter Priors

    [https://arxiv.org/abs/2611.03953](https://arxiv.org/abs/2611.03953)

    该论文提出 P$^2$-MOPD，将学生与专家参数的线性组合视为一个可微模型，在在线策略蒸馏目标下联合优化混合系数与学生权重，使专家作为参数先验而非限制学生，从而在四个领域上仅用一半训练步数就实现了比 MOPD 更快、更均衡的能力恢复。

    

    多教师在线策略蒸馏通过监督学生模型自身生成的轨迹来整合各专家模型的能力，而任务算术则通过线性参数组合提供了一种无需训练的替代方案。我们将学生模型与专家参数的线性组合视为一个单一的可微模型，并提出 P$^2$-MOPD，在相同的在线策略蒸馏目标下联合优化混合系数和学生模型权重。与任务算术不同——后者无论选择何种合并系数，都始终停留在冻结的学生与专家参数所张成的空间内——P$^2$-MOPD 保持学生模型可训练，使专家作为参数先验发挥作用，而不会将学生限制在该空间内。在 1.7B 和 4B 规模的四个领域上，P$^2$-MOPD 在相同初始化条件下比 MOPD 实现了更快且更均衡的能力恢复，仅用一半的训练步数即超越了 MOPD 的最终平均恢复水平。在训练结束时，我们的 1.7

    arXiv:2611.03953v2 Announce Type: replace-cross  Abstract: Multi-teacher on-policy distillation consolidates expert capabilities by supervising student-generated trajectories, whereas task arithmetic offers a training-free alternative through linear parameter combinations. We view the linear combination of student and expert parameters as a single differentiable model and propose P$^2$-MOPD, which jointly optimizes the mixing coefficients and the student weights under the same on-policy distillation objective. Unlike task arithmetic, which stays within the span of the frozen student and expert parameters for any choice of merging coefficients, P$^2$-MOPD keeps the student trainable, so the experts act as parameter priors without confining it to this span. Across four domains at 1.7B and 4B scales, P$^2$-MOPD achieves faster and more balanced recovery than MOPD from the same initialization, surpassing its final average recovery in half the training steps. At the end of training, our 1.7
    
[^223]: FADE：用于在线 RLVR 调度的能力—难度分解滤波

    FADE: Factored Ability--Difficulty Filtering for Online RLVR Scheduling

    [https://arxiv.org/abs/2611.03625](https://arxiv.org/abs/2611.03625)

    FADE 提出一种双线性状态空间滤波器，将策略能力（快速变化）与提示难度（相对稳定）这两个不同时间尺度的因素分解开来，从而在在线 RLVR 训练中更准确地筛选出有信息量的提示，避免浪费生成预算。

    

    基于可验证奖励的分组强化学习（RLVR）只能从那些采样结果存在分歧的提示中学习：全部通过和全部失败的组产生的相对优势为零，只会白白消耗生成预算而无法推进策略。自适应调度器试图将 rollout 引向更具信息量的提示，但它们依赖于对每个提示的通过率跟踪器，而这会把两个以截然不同速度演化的量纠缠在一起——即策略在各项推理技能上不断变化的能力，以及每个提示的内在难度。当用单一的指数平均或 Beta 计数器同时跟踪这两者时，短的记忆窗口会打乱提示之间的相对排序，而长的记忆窗口则会报告过时的通过率。我们提出了 FADE（Factored Ability–Difficulty Estimation，能力—难度分解估计），一种将这两个时间尺度分离的双线性状态空间滤波器。FADE 将每个提示的成功 logit 表示为白化的 K 维技能载荷与快速漂移的能力参数的内积……

    arXiv:2611.03625v2 Announce Type: replace  Abstract: Group-based reinforcement learning with verifiable rewards (RLVR) learns only from prompts whose sampled completions disagree: all-pass and all-fail groups produce zero relative advantage and burn generation budget without moving the policy. Adaptive schedulers try to steer rollouts toward informative prompts, yet they rely on per-prompt pass-rate trackers that entangle two quantities evolving at very different speeds---the policy's shifting competence across reasoning skills and each prompt's intrinsic difficulty. When a single exponential average or Beta counter tracks both at once, short memory scrambles the relative ordering of prompts whereas long memory reports stale pass rates. We introduce FADE (Factored Ability--Difficulty Estimation), a bilinear state-space filter that separates the two timescales. FADE represents each prompt's success logit as the inner product of a whitened $K$-dimensional skill loading with a fast-drifti
    
[^224]: Sonar-MoE：用于内存高效混合专家训练的反向传播内专家更新

    Sonar-MoE: In-Backward Expert Updates for Memory-Efficient Mixture-of-Experts Training

    [https://arxiv.org/abs/2611.03577](https://arxiv.org/abs/2611.03577)

    Sonar-MoE通过在反向传播过程中即时更新并释放专家梯度，将MoE训练的峰值内存降低19%–22%、每步速度提升17%–23%，同时保持与标准训练位级别一致的结果。

    

    混合专家模型的大多数参数都属于其专家模块，而标准训练会将每个专家的梯度保留在GPU内存中，直到优化器步进为止。这种内存占用会随着专家数量和规模的增长而增长，因此随着MoE模型的扩展而不断上升。我们提出了Sonar-MoE，它在反向传播过程中更新每个专家张量并立即释放其梯度。该方法将“反向传播内优化器”从稠密模型扩展到批处理的MoE专家张量，使GPU永远不会持有超过一层的专家梯度。在使用两种专家内核后端以bf16精度微调OLMoE-1B-7B和Granite-3.1-3B-A800M时，Sonar-MoE将峰值内存降低了19%到22%。其更新与不使用专家梯度裁剪的标准训练在位级别完全一致，留出损失与所有基线的差距保持在0.003以内。与使用全局范数裁剪的标准训练相比，每步训练时间也缩短了17%到23%。更新整个批处理张量而非单个张量……

    arXiv:2611.03577v1 Announce Type: cross  Abstract: Most parameters of a Mixture-of-Experts (MoE) model belong to its experts, and standard training keeps every expert gradient in GPU memory until the optimizer step. This memory grows with the number and size of the experts, so it rises as MoE models scale. We introduce Sonar-MoE, which updates each expert tensor during the backward pass and frees its gradient right away. It brings optimizer-in-backward from dense models to batched MoE expert tensors, so the GPU never holds more than one layer of expert gradients. When we fine-tune OLMoE-1B-7B and Granite-3.1-3B-A800M in bf16 with two expert-kernel backends, Sonar-MoE lowers peak memory by 19% to 22%. Its updates are bitwise identical to standard training without expert gradient clipping, and held-out loss stays within 0.003 of all baselines. Each step also takes 17% to 23% less time than standard training with global-norm clipping. Updating whole batched tensors rather than individual 
    
[^225]: 看见为什么：量子与光学图像分类器的精确证据图

    Seeing Why: Exact Evidence Maps for Quantum and Optical Image Classifiers

    [https://arxiv.org/abs/2611.01993](https://arxiv.org/abs/2611.01993)

    本文提出玻恩证据图（BEM），无需梯度电路和反向光路，即可在量子芯片或光学硬件上仅用 $1+2g^2$ 次运行，将量子与全光图像分类器各类别的玻恩概率精确分解为逐像素的贡献。

    

    当图像分类器做出决策时，我们通常只能估计其原因是为何。对于在测量之前对振幅编码图像进行线性作用、且前后不含已训练经典网络的量子分类器与全光分类器，每个类别高于其无信息基线的玻恩概率 $P_g - b_g$ 可以被精确分解为各编码振幅的贡献，并以玻恩证据图（BEM）的形式呈现。这种分解本身源自二次型的已知结果：在透射光保持固定的情况下，它正是从零基线出发的积分梯度与沙普利值所给出的结果。本文的贡献在于提出了一种在执行分类的实际硬件上测量该分解的方法：在量子芯片上通过类别概率、或在仿真中通过透镜平台上的相机帧来测量，无需梯度电路，也无需向后发送光；生成一张 $g\times g$ 的证据图只需 $1 + 2g^2$ 次运行。删除证据值最高的5%像素会使准确率从73%降至9%（……）

    arXiv:2611.01993v3 Announce Type: replace-cross  Abstract: When an image classifier decides, we can usually only estimate why. For quantum and all-optical classifiers that act linearly on amplitude-encoded pictures up to the measurement, with no trained classical network before or after them, each class's Born probability above its no-information share, $P_g - b_g$, splits exactly into contributions of the encoded amplitudes, shown as Born evidence maps (BEM). The split itself follows from known results on quadratic forms: with the transmitted light held fixed, it is what Integrated Gradients from a zero baseline and the Shapley value give. What we add is a way to measure it on the hardware that classifies, from class probabilities on a quantum chip or, in simulation, from camera frames on a lens bench, with no gradient circuit and no light sent backwards; a $g\times g$ map takes $1 + 2g^2$ runs. Deleting the 5% of pixels with the highest evidence drops accuracy from 73% to 9% (one see
    
[^226]: 以模型读取的方式书写事实：破解两跳诅咒

    Write Facts the Way Models Read Them: Lifting the Two-Hop Curse

    [https://arxiv.org/abs/2611.01519](https://arxiv.org/abs/2611.01519)

    论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。

    

    arXiv:2611.01519v2 公告类型：新 摘要：通过微调学习新事实的语言模型能够复述该事实，却无法用它进行推理：教会模型某个人是新国家公民之后，它仍然无法在不借助思维链的情况下说出这个人所在国家的国家元首是谁。这种“两跳诅咒”通常被视为推理能力缺陷，而最强的知识编辑方法则通过合成推理数据来解决它。我们证明：推理能力是完好的，错的是事实的书写方式。微调把新事实写成一个“要说出口的答案”，而第二跳推理从中间层开始就把桥接短语（“那个人的国家”）当作实体来读取：在相应层将模型自身对该实体的表征修补进桥接短语，可以基本恢复第二跳推理；然而微调之后，桥接短语与该新实体的接近程度并不比编辑前更高。因此我们提出“读取等价书写”（Legible）：用一个损失项让桥接短语在第二跳读取它的那一层（摘要在此处截断）

    arXiv:2611.01519v2 Announce Type: new  Abstract: A language model that learns a new fact by fine-tuning can repeat it but cannot reason with it: taught that a person is a citizen of a new country, it still cannot say, without chain-of-thought, who the head of state of that person's country is. This two-hop curse is widely treated as a reasoning deficit, and the strongest knowledge editors answer it with synthesized reasoning data. We show that the reasoning is intact and the writing is wrong. Fine-tuning writes a new fact as an answer to be said, whereas the second hop reads the bridge ("that person's country") as an entity, from the middle layers on: patching the model's own representation of the entity into the bridge at those layers largely restores the second hop, yet after fine-tuning the bridge there is no closer to the new entity than before the edit. We therefore propose read-equivalent writing (Legible): one loss term makes the bridge phrase, at the layer where the second hop 
    
[^227]: 神经网络中训练、学习、推理与强化的统一动力学

    Unified dynamics of training, learning, inference, and reinforcement in neural networks

    [https://arxiv.org/abs/2611.01345](https://arxiv.org/abs/2611.01345)

    该论文通过九项nanoGPT实验建立了训练、学习与推理间的定量关系，提出了预测更新后能力转变的直接预测器（准确率91.43%），发现重新平衡反馈可恢复被集中正确反馈侵蚀的技能，从而将三者闭合为统一的训练-学习-推理-强化动力学。

    

    通过九项nanoGPT实验，我们建立了训练-学习-推理之间的关系：已实现的更新会引发依赖于状态的有限幅度响应；学习会持续重组分布式功能支持，其中特定目标的读出边界穿越决定了能力转变；推理会调用已学习的支持而不改变已学习的状态。在第五项实验中识别出的因素产生了一个直接预测更新后能力转变的预测器，在5,088个目标上达到了91.43%的准确率和91.49%的四分类宏观召回率。我们在另外五种神经架构上验证了这些关系和预测。在第七个系统中，集中的正确反馈侵蚀了先前已掌握的技能；而重新平衡的反馈将其准确率从70.31%恢复到99.48%。将推理的后果与后续训练联系起来，使这些关系闭合为统一的训练-学习-推理-强化动力学。

    arXiv:2611.01345v2 Announce Type: replace  Abstract: Through nine nanoGPT experiments, we established training-learning-inference relations: realized updates elicit state-dependent finite-amplitude responses; learning persistently reorganizes distributed functional support, with target-specific readout-boundary crossings determining capability transitions; inference recruits learned support without changing the learned state. Factors identified in the fifth experiment yielded a direct predictor of post-update capability transitions, achieving 91.43% accuracy and 91.49% four-way macro recall across 5,088 targets. We tested these relations and prediction across five additional neural architectures. In a seventh system, concentrated correct feedback eroded previously mastered skills; rebalanced feedback restored their accuracy from 70.31% to 99.48%. Linking inference consequences to subsequent training closes these relations into unified training-learning-inference-reinforcement dynamics.
    
[^228]: 轮廓感知的被丢弃参数Hessian质量诊断

    Profile-aware diagnostics of discarded parameter-Hessian mass

    [https://arxiv.org/abs/2611.01335](https://arxiv.org/abs/2611.01335)

    提出一种轮廓感知的诊断框架，通过在不同分组规模和组内曲率轮廓参考下测量参数Hessian的块外平方占比，来量化块对角曲率模型所丢弃的曲率质量，并评估候选修正的实际效果。

    

    评估块对角曲率模型涉及两个问题：它们省略了多少曲率，以及一个特定的修正在有限步长内改变了什么。我们针对不同组规模以及观测到的组内曲率轮廓的参考基准，测量参数Hessian的块外平方占比。在CIFAR-100的ResNet-18及其无跳跃连接的对应模型中，相似的初始丢弃占比伴随着不同的曲率轮廓。在已保存的训练好的ResNet-18权重处，将分组方式从阶段组改为层组，会使估计的块外质量从轮廓包络之下移动到其之上；一个近似条件筛选方法在所有六个种子/输入单元中都解决了这一比较问题，而某些无跳跃连接模型之间的比较仍未得出结论。另外，在已保存的ResNet-50/Stanford Cars状态下，我们通过相同的Hessian信息映射，比较了由组外Hessian作用、剩余模型残差以及二者组合所构成的候选修正。在单位尺度下，仅组外输入（摘要在此处被截断）

    arXiv:2611.01335v2 Announce Type: replace  Abstract: Evaluating block-diagonal curvature models involves two questions: how much curvature they omit, and what a specific correction changes at a finite step. We measure the squared off-block fraction of the parameter Hessian against references for group sizes and the observed within-group curvature profile. In CIFAR-100 ResNet-18 and a skip-free counterpart, similar initial discarded fractions accompany different profiles. At saved trained ResNet-18 weights, changing from stage to layer groups moves estimated off-block mass from below to above the profile envelope; an approximate conditional screen resolves this comparison in all six seed/input cells, while some skip free comparisons remain unresolved. Separately, in saved ResNet-50/Stanford Cars states, we compare candidates formed from off-group Hessian action, the remaining model residual and their combination through the same Hessian-informed map. At unit scale off-group input alone 
    
[^229]: 黎曼流形上积分曲线的抽样设计与置信管

    Sampling Design and Confidence Tubes for Integral Curves on Riemannian Manifolds

    [https://arxiv.org/abs/2611.00771](https://arxiv.org/abs/2611.00771)

    本文在紧黎曼流形上提出了估计未知向量场积分曲线的被动与主动两种抽样方法，证明了被动核估计的曲线误差阶为 $n^{-2/(d+3)}$ 并给出由线性随机微分方程刻画的函数型中心极限定理，同时证明基于先导信息的主动抽样可将期望最大曲线误差提升至 $N^{-1/2}$ 阶，从而为构建置信管奠定基础。

    

    我们在已知紧黎曼流形上，利用带噪声的切向量场来估计未知向量场的积分曲线。我们考虑两种抽样设置：一种是在已收集的随机位置上的独立观测，另一种是在固定响应预算下，允许利用先导信息来选择后续观测位置。在适当的正则性假设下，被动核估计量的曲线误差以概率达到 $n^{-2/(d+3)}$ 阶，其中 $n$ 为样本量，$d$ 为流形维数。其函数型中心极限定理的极限由一个线性随机微分方程刻画。对于主动抽样，一个独立的先导样本将新的一批测量引导至目标曲线的邻域。在局部化、光滑性和设计条件下，这种集中性使得期望最大曲线误差达到 $N^{-1/2}$ 阶，其中先导样本计入总响应预算 $N$。两种方法均……（摘要原文在此处截断）

    arXiv:2611.00771v2 Announce Type: replace-cross  Abstract: We estimate the integral curve of an unknown vector field from noisy tangent vector field on a known compact Riemannian manifold. We consider two sampling settings: independent observations at random locations that have already been collected, and a fixed response budget that allows subsequent locations to be chosen using pilot information. Under suitable regularity assumptions, the passive kernel estimator has curve error of order $n^{-2/(d+3)}$ in probability, with sample size $n$ and manifold dimension $d$. Its functional central limit theorem has a limit characterized by a linear stochastic differential equation. For active sampling, an independent pilot directs one new batch of measurements to neighborhoods of the target curve. Under localization, smoothness and design conditions, this concentration yields expected maximum curve error of order $N^{-1/2}$, including the pilot in the total response budget $N$. Both methods p
    
[^230]: PeptSol：面向药物制造中保护态肽溶解度的转化型基准数据集

    PeptSol: A Translational Benchmark for Protected-Peptide Solubility in Drug Manufacturing

    [https://arxiv.org/abs/2611.00749](https://arxiv.org/abs/2611.00749)

    提出了PeptSol——首个面向液相肽合成工艺开发的保护态肽溶解度开放基准数据集，包含1388条人工整理的溶解度测量数据并采用化学感知划分，弥补了现有肽性质基准在保护基团、末端修饰和非天然残基方面的不足，为分子表示模型提供了严格的评估平台。

    

    液相肽合成（LPPS）为肽类药物制造提供了可规模化的途径，但工艺路线的开发仍然依赖于经验性的溶剂筛选，因为每个保护态中间体必须可溶解以进行反应，且可回收用于纯化。决定这一行为的保护基团、末端修饰和非天然残基在现有的肽性质基准数据集中代表性不足，限制了针对LPPS相关预测任务开发和严格评估分子表示模型的能力。我们提出了PeptSol，这是一个包含1388条人工整理的保护态肽溶解度测量数据、并采用化学感知数据划分的开放基准数据集。这些测量数据仅涵盖84个母序列和101个保护态结构，其中92.5%来自单一历史项目。在七个冻结分子编码器上，在按肽划分（peptide-held-out split）的历史子集的二元溶解度任务中，最佳宏观F1分数为0.843……（摘要原文在此处截断）

    arXiv:2611.00749v2 Announce Type: replace  Abstract: Liquid-phase peptide synthesis (LPPS) offers a scalable route to peptide manufacturing, but route development still depends on empirical solvent screens because each protected intermediate must be soluble for reaction and recoverable for purification. The protecting groups, terminal modifications, and noncanonical residues that govern this behaviour are poorly represented in existing peptide property benchmarks, limiting the ability to develop and rigorously evaluate molecular representation models for LPPS-relevant prediction tasks. We introduce PeptSol, an open benchmark of 1388 hand-curated protected-peptide solubility measurements with chemistry-aware splits. The measurements span only 84 parent sequences and 101 protected structures, and 92.5% come from a single historical programme. Across seven frozen molecular encoders, the best macro-F1 is 0.843 on binary solubility for the historical subset under a peptide-held-out split; t
    
[^231]: 测试时搜索何时有助于世界动作模型？对候选选择与模拟器完整性的实证审计

    When Does Test-Time Search Help World Action Models? An Empirical Audit of Candidate Selection and Simulator Integrity

    [https://arxiv.org/abs/2610.41970](https://arxiv.org/abs/2610.41970)

    该论文通过预注册的实证审计发现，世界动作模型的测试时搜索收益存疑：候选选择几乎不存在可利用信号（学习到的价值选择器比随机选择还差 1.66 个百分点），token 剪枝虽有 47% 的 FLOPs 节省但实测端到端加速最高仅 1.15 倍、未达 1.5 倍目标，从而揭示测试时扩展所依赖的关键前提往往并不成立。

    

    世界动作模型越来越多地采用测试时扩展（test-time scaling），将额外的推理计算投入到候选选择与自适应分配上，以改进控制。只有当候选本身带有可利用的信号、名义上的计算节省能够转化为真实的加速、并且评估保持可靠时，这类收益才可信。我们以“前提优先”的预注册方法，在 Cosmos Policy 和 RoboCasa 上对这三点进行了审计。在干净数据上，学习到的价值选择器比随机选择差 1.66 个百分点；事后（hindsight）oracle 仅提升 1.40 个百分点，且回合级置信区间包含零；82% 的状态下各候选结果完全相同，几乎没有选择空间。Token 剪枝可将每次前向传播的 FLOPs 最多降低 47%，但在部署路径上实测端到端加速最高仅为 1.15 倍（在排除解码的路径上为 1.25 倍），未达到 1.5 倍的目标，而我们的自适应分配实验在统计功效门槛下结果不确定。我们进一步识别……

    arXiv:2610.41970v1 Announce Type: new  Abstract: World-action models increasingly apply test-time scaling, spending extra inference computation on candidate selection and adaptive allocation to improve control. Such gains are credible only when candidates carry exploitable signal, nominal compute savings become real speedup, and evaluation stays reliable. We audit all three on Cosmos Policy and RoboCasa with a premise-first, pre-registered methodology. On clean data, a learned value selector is 1.66 pp worse than random; a hindsight oracle gains only 1.40 pp, with an episode-level interval spanning zero; and 82% of states show identical candidate outcomes, leaving little selection headroom. Token pruning cuts per-forward FLOPs by up to 47% but yields at most a measured 1.15$\times$ end-to-end speedup on the deployed path (1.25$\times$ on a decode-excluded path), short of a 1.5$\times$ target, and our adaptive-allocation experiment is inconclusive under a power gate. We further identify
    
[^232]: TSBPO：面向交互式语言智能体策略优化的树结构分支方法

    TSBPO: Tree-Structured Branching for Policy Optimization in Interactive Language Agents

    [https://arxiv.org/abs/2610.41697](https://arxiv.org/abs/2610.41697)

    TSBPO提出一种树结构的采样到更新范式，通过似然拟合的根/分支分配器定向采样、基于兄弟片段与匹配观察状态的延迟反馈分配、以及跨活跃轮次归一化的词元损失聚合，在单一训练循环中解决了从终端结果训练交互式语言智能体的三大耦合难题。

    

    从终端结果训练交互式语言智能体面临三个相互耦合的问题：应采样哪些任务与状态延续、如何将延迟反馈分配到各个决策上，以及如何为多词元动作聚合词元损失。我们提出TSBPO，一种树结构的“采样到更新”范式，在单一训练循环内解决上述问题。一个基于似然拟合的根/分支分配器利用保留的任务与状态证据，在固定rollout预算下将延续采样定向到可能产生对比性终端结果的分支上。在由此构建的重放物理树中，轨迹优势通过比较直接相邻的兄弟片段来计算，而观察组步骤优势则在匹配的任务/观察状态下比较不同结果。随后，执行器对每个完整响应求和裁剪的词元代理项，并在活跃轮次间进行归一化，从而避免了策略梯度项中的响应长度平均。使用Qwen2.5-1.5B作为执行器，固定检查点……（摘要原文在此处截断）

    arXiv:2610.41697v1 Announce Type: cross  Abstract: Training interactive language agents from terminal outcomes poses three coupled problems: which task and state continuations to sample, how to assign delayed feedback to decisions, and how to aggregate token losses for multi-token actions. We present TSBPO, a tree-structured sampling-to-update paradigm that addresses these problems within one training loop. A likelihood-fitted Root/Branch allocator uses retained task and state evidence to target continuations likely to produce contrasting terminal outcomes under fixed rollout budgets. On the resulting replayed physical trees, trajectory advantages compare immediate sibling segments, while observation-group step advantages compare outcomes at matched task/observation states. The actor then sums clipped token surrogates over each complete response and normalizes across active turns, avoiding response-length averaging in the policy-gradient term. With a Qwen2.5-1.5B actor, fixed checkpoin
    
[^233]: V-LUMEN：用于视觉-语言模型嵌入扩展的视觉查找记忆

    V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models

    [https://arxiv.org/abs/2610.37019](https://arxiv.org/abs/2610.37019)

    V-LUMEN提出了一种视觉查找记忆机制，通过冻结码本离散化视觉特征、空间聚合哈希构建记忆键并结合文本条件调制信号，以完全非参数化的寻址方式为视觉-语言模型实现嵌入层面的扩展，显著提升冻结主干网络在推理基准上的表现并具备零样本迁移能力。

    

    嵌入层面的扩展通过基于查找的记忆来扩展语言模型的容量，而非通过增加模型的深度、宽度或密集计算。将这一思想扩展到视觉-语言模型（VLM）具有挑战性，因为视觉特征是连续的、具有空间结构的、且依赖于查询的，这与天然支持n-gram查找的离散文本标记不同。我们提出了V-LUMEN，一种面向视觉-语言模型的视觉查找记忆。V-LUMEN使用冻结的码本对视觉特征进行离散化，通过基于空间聚合的哈希构建空间感知的记忆键，并使用文本条件化的相关性信号对其进行调制；所有寻址过程都是非参数化的，检索到的嵌入通过门控残差通路融合到主干网络中。在五个推理密集型基准测试上，V-LUMEN显著提升了冻结的Qwen3-VL-2B主干网络的性能，并能零样本迁移到三个图像与训练数据完全不相交的基准测试上。

    arXiv:2610.37019v1 Announce Type: cross  Abstract: Embedding-level scaling expands the capacity of language models through lookup-based memory rather than by increasing model depth, width, or dense computation. Extending this idea to vision-language models (VLMs) is challenging because visual features are continuous, spatially structured, and query-dependent, unlike discrete text tokens that naturally support n-gram lookup. We introduce V-LUMEN, a visual lookup memory for VLMs. V-LUMEN discretizes visual features with a frozen codebook, builds spatially aware memory keys through spatial aggregation-based hashing, and modulates them with text-conditioned relevance signals; all addressing is non-parametric, and the retrieved embeddings are fused into the backbone through a gated residual pathway. On five reasoning-intensive benchmarks, V-LUMEN substantially improves a frozen Qwen3-VL-2B backbone and transfers zero-shot to three benchmarks whose images are disjoint from the training data.
    
[^234]: 信号在改善，编码器在退化：自监督学习中无标签监测的已知答案测试

    Improving Signals, Declining Encoder: A Known-Answer Test of Label-Free Monitoring in Self-Supervised Learning

    [https://arxiv.org/abs/2610.35766](https://arxiv.org/abs/2610.35766)

    该论文通过已知答案的合成数据实验揭示，自监督学习中常用的无标签监测信号（如损失和特征统计量）无法察觉编码器的实际退化，甚至会给退化中的训练更高的评分，从而可能误导检查点选择。

    

    自监督训练通常在没有标签的情况下进行监测，依靠损失和学习到的特征的统计量。我们报告了一个案例，其中我们知道编码器丢失了它们已经学到的部分内容，因此可以检验这些信号是否察觉到了这一点。我们在具有已知生成因子的合成图像上训练了LeJEPA编码器。在十个随机种子中的九个中，这些因子的可读性突然下降并持续保持低位，且几乎总是沿着因子空间中的同一方向发生。与此同时，损失持续下降，我们跟踪的十一个无标签信号中的大多数持续改善，并且将发生退化的运行评得比未发生退化的运行更好。标准的变点检测很少在退化开始时附近发出警报，而通过这些信号选择检查点时，大多数情况下选中的是退化发生之后的检查点。

    arXiv:2610.35766v1 Announce Type: new  Abstract: Self-supervised training is monitored without labels, through the loss and statistics of the learned features. We report a case where we know the encoders lost part of what they had learned, so we can check whether these signals noticed. We trained LeJEPA encoders on synthetic images with known generating factors. In nine of ten seeds, the readability of those factors fell abruptly and stayed down, nearly always along the same direction in factor space. Meanwhile the loss kept falling, and most of the eleven label-free signals we tracked kept improving and rated the declining runs better than runs that did not decline. Standard change-point detection rarely flagged the decline near its onset, and selecting checkpoints by these signals mostly chose ones after it.
    
[^235]: 多层梯度正则化牛顿方法的全局 $\mathcal{O}\left(k^{-2}\right)$ 收敛性

    Global $\mathcal{O}\left(k^{-2}\right)$ Convergence of Multilevel Gradient-Regularized Newton Methods

    [https://arxiv.org/abs/2610.35195](https://arxiv.org/abs/2610.35195)

    本文提出了适用于利普希茨连续海森矩阵无约束凸优化的多层梯度正则化牛顿方法，通过在低维子空间中计算牛顿修正并引入行嵌套限制算子，在大幅降低计算成本的同时实现了全局 $\mathcal{O}(k^{-2})$ 收敛速率。

    

    我们针对具有利普希茨连续海森矩阵的无约束凸优化问题，开发并分析了多层梯度正则化牛顿方法。通过在低维子空间中计算牛顿修正项，所提出的方法旨在降低计算成本，同时保持全空间方法在函数值上的全局 $\mathcal{O}\left(k^{-2}\right)$ 收敛速率。为了给多层方法建立这一收敛速率，我们引入了一类行嵌套限制算子，以确保各层之间约化梯度范数的单调性。使用行嵌套算子的数值实验表明，该方法具有与全空间梯度正则化牛顿方法相当的收敛行为，同时在大规模算例上提供了显著的计算节省。

    arXiv:2610.35195v1 Announce Type: cross  Abstract: We develop and analyze multilevel gradient-regularized Newton methods for unconstrained convex optimization with Lipschitz-continuous Hessians. By computing Newton corrections in lower-dimensional subspaces, the proposed methods aim to reduce computational costs while preserving the global $\mathcal{O}\left(k^{-2}\right)$ convergence rate in function values of the full-space method. To establish this rate for multilevel methods, we introduce a class of row-nested restriction operators that ensures monotonicity of the reduced-gradient norms across levels. Numerical experiments with row-nested operators demonstrate convergence behavior comparable to full-space gradient-regularized Newton, while offering substantial computational savings on large-scale instances.
    
[^236]: 用多面体逼近欧氏球：高维中的惊人速率

    Approximating the Euclidean Ball by Polytopes: Surprising Rates in High Dimensions

    [https://arxiv.org/abs/2610.34651](https://arxiv.org/abs/2610.34651)

    本文给出了用多面体逼近高维欧氏球所需顶点数的精确估计，证明仅需次指数级数量的顶点 exp(O(√d/ε)) 即可，但达到固定误差却需要指数级数量的面，并肯定回答了 De 等人关于次指数顶点复杂性的公开问题。

    

    我们给出了用相对对称差体积误差至多 ε∈(0,1) 来逼近欧氏单位球 B_d⊂R^d 所需顶点数的精确估计。我们证明最小顶点预算 N_ε(d) 满足 log N_ε(d) ≍ d log(1 + 1/(ε√d))。特别地，对每个 ε∈(0,1) 以及所有充分大的 d，exp(O(√d/ε)) 个顶点即可足够。我们还证明了达到固定误差 ε∈(0,1) 至少需要 Ω(((1+ε)/(2√ε))^(d-1)) 个面，这是关于 d 的指数级数量。我们将顶点逼近结果推广到高斯对称差误差的情形，并对 De、Nadimpalli 和 Servedio 提出的关于逼近欧氏球的次指数顶点复杂性问题给出了肯定答案。我们还否证了一个 Macbeath 型的极值性原理。

    arXiv:2610.34651v2 Announce Type: cross  Abstract: We provide sharp estimates on the number of vertices needed to approximate the Euclidean unit ball $B_d\subset\mathbb R^d$ with relative symmetric-difference volume error at most $\varepsilon\in(0,1)$. We show that the minimum vertex budget $N_\varepsilon(d)$ satisfies $$\log N_\varepsilon(d)\asymp d\log\left(1+\frac1{\varepsilon\sqrt d}\right).$$ In particular, $\exp(O(\sqrt d/\varepsilon))$ vertices suffice for every $\varepsilon\in(0,1)$ and all sufficiently large $d$. We also prove that achieving a fixed error $\varepsilon\in(0,1)$ requires at least $\Omega\bigl((\tfrac{1+\varepsilon}{2\sqrt\varepsilon})^{d-1}\bigr)$ facets, an exponential number in $d$. We extend our vertex approximation result to Gaussian symmetric-difference error, and give a positive answer to the question of De, Nadimpalli, and Servedio about subexponential vertex complexity for approximating Euclidean balls.   We also disprove a Macbeath-type extremality prin
    
[^237]: 目标之前的记忆：预历史记忆在人形机器人移动操作中的作用

    Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation

    [https://arxiv.org/abs/2610.31166](https://arxiv.org/abs/2610.31166)

    提出物体引导的感知记忆机制WholeBodyMem，利用SAM3跟踪的物体事件因果选择关键帧，使人形机器人全身VLA策略能够记住行动前的物体历史，从而在非马尔可夫动态场景中大幅提升成功率（仿真中29.1%提升至72.6%，真实机器人上36.7%提升至88.3%），并发布了HuMeGo人形机器人记忆基准。

    

    动态场景中的人形机器人移动操作是非马尔可夫的：当物体发生变化或被遮挡时线索会消失，或者当机器人转身行走时物体会离开视野，而全身视觉-语言-动作（VLA）策略却只能看到当前时刻。我们提出了WholeBodyMem，一种物体引导的感知记忆，它保存了机器人在执行目标之前所看到的物体事件的历史。SAM3负责跟踪任务物体，其事件以因果方式在自适应预算内选择关键帧，动作专家通过交叉注意力机制将这些关键帧与编码外观、几何和时间的物体标记一起读取。我们还贡献了HuMeGo，一个包含六个仿真任务和六个真实世界Unitree G1任务的人形机器人记忆基准。通过保留正确的帧而非更多的帧，WholeBodyMem超越了无记忆和均匀记忆的基线方法，将其基础策略的成功率在仿真中从29.1%提升至72.6%，在真实机器人上从36.7%提升至88.3%，同时还改善了……

    arXiv:2610.31166v1 Announce Type: cross  Abstract: Humanoid loco-manipulation in dynamic scenes is non-Markovian: cues vanish as objects change or are hidden, or leave the view as the robot turns and walks, yet whole-body vision-language-action (VLA) policies see only the present. We present WholeBodyMem, an object-guided perceptual memory that keeps the pre-history of object events seen before the robot acts on its goal. SAM3 tracks the task objects, and their events causally select key frames within an adaptive budget, which the action expert reads by cross-attention together with object tokens encoding appearance, geometry and time. We also contribute HuMeGo, a humanoid memory benchmark with six simulated and six real-world Unitree G1 tasks. Keeping the right frames rather than more frames, WholeBodyMem surpasses memoryless and uniform-memory baselines, raising its base policy's success from 29.1% to 72.6% in simulation and from 36.7% to 88.3% on the real robot, while also improving
    
[^238]: Orbitol：基于Wyckoff轨道的晶体生成

    Orbitol: Crystal Generation on Wyckoff Orbits

    [https://arxiv.org/abs/2610.30086](https://arxiv.org/abs/2610.30086)

    Orbitol通过用Wyckoff轨道标记替代原子标记，实现了对空间群对称性的直接控制，能生成保持所要求空间群的稳定新颖晶体结构，且生成速度比现有对称性感知基线快两到三个数量级。

    

    生成模型通过学习组分和几何结构来提出晶体材料，但无条件采样无法直接控制空间群，这限制了诸如压电性等响应是否在对称性上被允许。我们提出了Orbitol，一个构建于Crystalite扩散Transformer之上的晶体生成器，它用Wyckoff轨道标记取代了原子标记。给定一个空间群和一组被占据的Wyckoff位置，Orbitol生成每个轨道的原子类型和自由坐标，以及与之兼容的晶格，因此每个有效结构在所请求空间群的对称操作下都保持不变。在所评估的设置下，其生成速度比对称性感知的基线方法快两到三个数量级。在MP-20数据集上，Orbitol有13.8%的尝试产生了稳定或亚稳、独特且新颖的结构，并在弛豫后保持了所请求的空间群，而次优方法SGFM仅为10.8%。

    arXiv:2610.30086v1 Announce Type: new  Abstract: Generative models propose crystalline materials by learning composition and geometry, but unconditional sampling gives no direct control over the space group, which constrains whether responses such as piezoelectricity are symmetry-allowed. We present Orbitol, a crystal generator built on the Crystalite diffusion Transformer that replaces atom tokens with Wyckoff-orbit tokens. Given a space group and a collection of occupied Wyckoff positions, Orbitol generates the atom type and free coordinates of each orbit together with a compatible lattice, so every valid structure is invariant under the operations of the requested group. It generates structures two to three orders of magnitude faster than the symmetry-aware baselines at the evaluated settings. On MP-20, 13.8% of Orbitol's attempts yielded stable or metastable, unique and novel structures that retained the requested space group after relaxation, compared with 10.8% for SGFM, the next
    
[^239]: 无法逃离的二次方屏障：针对线性 $\ell_2$ 草图的自适应攻击

    No Escape from the Quadratic Barrier: Adaptive Attacks on Linear $\ell_2$ Sketches

    [https://arxiv.org/abs/2610.25008](https://arxiv.org/abs/2610.25008)

    本文证明，即使响应者计算能力无限制且掌握全部信息，针对线性 $\ell_2$ 草图的自适应黑盒攻击也只需约 $O(k+\alpha^2k^2)$ 量级的查询即可破坏其精度，因此约 $k^2$ 的二次方防御保证在对数因子范围内已是最优、无法逃脱。

    

    线性草图通过固定的低维投影来估计高维输入的性质。在恒定精度下，随机化的欧几里得范数草图（尤其是 Johnson-Lindenstrauss 变换）可以在草图维度 $k$ 下支持指数级数量的非自适应查询。对于自适应黑盒查询，已知的基于隐私的防御可提供 $\widetilde{\Omega}(k^2)$ 的保证，而能否获得更强的保证一直是悬而未决的问题。先前的攻击要么需要更多查询，要么限制了响应者，要么只能提供较弱的失败保证。我们证明，二次方保证在对数因子意义下是最优的——即使响应者拥有无限制的计算能力，知晓矩阵、攻击算法、历史记录和查询分布，并且可以在每次查询后更改其随机化规则。对于每个固定的隐藏矩阵 $A\in\mathbb{R}^{k\times n}$（其中 $n>40k$）以及目标精度 $\alpha \in (0,1]$，我们的黑盒攻击使用 $O(k+\alpha^2k^2\log(\cdots))$ 次查询（原文摘要在此处截断）。

    arXiv:2610.25008v2 Announce Type: replace-cross  Abstract: Linear sketches estimate properties of high-dimensional inputs from a fixed low-dimensional projection. At constant accuracy, randomized Euclidean-norm sketches, notably the Johnson-Lindenstrauss transform, support exponentially many nonadaptive queries in sketch dimension $k$. For adaptive black-box queries, known privacy-based defenses guarantee $\widetilde{\Omega}(k^2)$ and it remained open whether stronger guarantees were possible. Prior attacks used more queries, restricted the responder, or provided weaker failure guarantees. We show that the quadratic guarantee is optimal up to logarithmic factors, even with a computationally unbounded responder that knows the matrix, attack algorithm, history, and query distributions and may change its randomized rule after every query. For every fixed hidden $A\in\mathbb{R}^{k\times n}$ with $n>40k$ and target accuracy $\alpha \in (0,1]$, our black-box attack uses $O(k+\alpha^2k^2\log(
    
[^240]: 天空图像与晴空指数的单阶段联合扩散用于超短期辐照度预测：帧到辐照度阶段所丢弃的信息

    Single-stage joint diffusion of sky images and clear-sky index for ultra-short-term irradiance forecasting: what the frame-to-irradiance stage discards

    [https://arxiv.org/abs/2610.21795](https://arxiv.org/abs/2610.21795)

    该论文提出一种在单一扩散过程中联合去噪未来天空图像与晴空指数的单阶段辐照度预测模型，省去了传统两阶段设计中的独立回归器且无需实测辐照度输入，在仅有相机的站点上实现了超越十二个基线配置的超短期辐照度预测性能。

    

    天空图像辐照度预测目前以两阶段设计为主：由视频预测模型预测未来的天空图像帧，再由一个单独训练的回归器将这些帧映射为辐照度。我们提出一种单阶段替代方案，在同一个扩散过程中对未来的天空图像序列和晴空指数序列进行联合去噪，其中辐照度头读取的是未来天空的去噪预览，而非带噪声的扩散状态。该预测器不需要回归阶段，仅依靠天空图像和确定性协变量运行，输入中不含任何实测辐照度——这正是有相机但没有总辐射表的站点所面临的实际情形。在2023年SIRTA数据上，该模型达到了104.1 W m⁻²的汇总RMSE，以连续排序概率得分（CRPS）衡量的相对智能持续性模型的技巧得分为+0.34，高于全部十二个基线配置，并且在2021-2022年数据上重新训练后仍能保持该技巧水平。在该模型自身生成的……

    arXiv:2610.21795v1 Announce Type: cross  Abstract: Sky-image irradiance forecasting is dominated by a two-stage design: a video-prediction model forecasts future sky frames, and a separately trained regressor maps those frames to irradiance. We propose a single-stage alternative that co-denoises the future sky-image and clear-sky-index sequences in one diffusion process, with an irradiance head that reads a denoised preview of the future sky rather than the noised diffusion state. The forecaster needs no regression stage and runs from sky images and deterministic covariates alone, with no measured irradiance at its input --- the situation at a site with a camera but no pyranometer. On 2023 SIRTA data the model reaches a pooled RMSE of $104.1$~W\,m$^{-2}$ and a skill score of $+0.34$ on the continuous ranked probability score (CRPS) against smart persistence, higher than any of twelve baseline configurations, and retains that skill when retrained on 2021--2022. On the model's own genera
    
[^241]: 面向健康错误信息分类与语义检索的隐私保护文本推理

    Privacy-Preserving Text Inference for Health-Misinformation Classification and Semantic Retrieval

    [https://arxiv.org/abs/2610.20796](https://arxiv.org/abs/2610.20796)

    本文提出了一种基于CKKS同态加密的隐私保护文本推理框架，将BioBERT、MPNet与TF-IDF等特征融合用于健康错误信息分类（准确率达0.971），并结合语义指纹实现密文域上的语义检索。

    

    健康错误信息分析面临双重挑战：误导性言论可能影响公众健康福祉，而用于检测这些言论的系统可能会暴露用户提交的敏感文本。这促使人们为公共卫生应用设计隐私保护推理，其中用户查询可能包含机密医疗信息，而训练好的模型和参考数据库则保存在组织服务器上。我们设计了一个基于CKKS同态加密的隐私保护文本推理框架，用于健康错误信息分类和语义检索。分类管线融合了BioBERT嵌入、经PCA降维的MPNet嵌入以及经SVD降维的TF-IDF特征，而检索管线则通过后期融合将MPNet嵌入与ISCC-SCT语义指纹相结合。我们评估了预测性能，并对明文执行与基于密文的执行进行了基准测试。混合分类器达到了0.971的准确率，并且

    arXiv:2610.20796v1 Announce Type: cross  Abstract: Health-misinformation analysis presents a dual challenge: misleading claims can affect public well-being, while systems designed to detect them may expose sensitive user-submitted text. This motivates privacy-preserving inference for public-health applications in which user queries may contain confidential medical information and trained models and reference databases are maintained on organizational servers. We design a privacy-preserving text inference framework based on CKKS homomorphic encryption for health-misinformation classification and semantic retrieval. The classification pipeline fuses BioBERT embeddings, PCA-reduced MPNet embeddings, and SVD-reduced TF-IDF features, while the retrieval pipeline combines MPNet embeddings with ISCC-SCT semantic fingerprints through late fusion. We evaluate predictive performance and benchmark plaintext versus ciphertext-based execution. The hybrid classifier achieved an accuracy of 0.971 and
    
[^242]: 我们是否把单步生成想得太复杂了？

    Are We Overthinking One-Step Generation?

    [https://arxiv.org/abs/2610.20331](https://arxiv.org/abs/2610.20331)

    本文提出Priming Models两阶段框架：先用简单的重建目标（如固定噪声水平的像素去噪）预训练，再进行特征空间分布后训练，并发现这一简单的重建阶段即可完全解决单步生成器分布训练的困难。

    

    扩散模型和流匹配如今已成为连续域生成建模的主流框架，但它们在推理时通常需要多次网络评估。这推动了大量基于扩散和流的单步生成建模研究，然而现有方法往往依赖复杂的训练机制。在本工作中，我们提出了Priming Models（预启动模型），这是一个用于训练单步生成器的简单两阶段框架。鉴于从零开始进行分布训练的困难，Priming Models首先以重建目标进行训练（例如在固定噪声水平下进行像素空间去噪），然后再进行特征空间的分布后训练。我们的核心科学发现是：这一简单的重建阶段本身就足以完全消除分布训练的困难。实证结果表明，该方法在类条件图像生成、文本到图像生成和世界模型等任务上均有效。

    arXiv:2610.20331v1 Announce Type: new  Abstract: Diffusion and flow matching are now dominant frameworks for generative modeling in continuous domains, but they typically require multiple network evaluations at inference. This has motivated extensive research on one-step generative modeling primarily built on diffusion and flow, though existing approaches often rely on complex training machinery. In this work, we introduce Priming Models, a simple two-stage framework for training one-step generators. Motivated by the difficulty of distributional training from scratch, Priming Models are first trained on a reconstruction objective, e.g., pixel-space denoising at a fixed noise level, before undergoing feature-space distributional post-training. Our central scientific finding is that this simple reconstruction stage is already enough to entirely alleviate the difficulty of distributional training. Empirically, across class-conditional image generation, text-to-image generation, and world 
    
[^243]: TANGO：RNA拓扑与几何协同设计

    TANGO: RNA Topology and Geometry Co-Design

    [https://arxiv.org/abs/2610.20277](https://arxiv.org/abs/2610.20277)

    提出TANGO框架，首次将RNA逆设计表述为拓扑与几何协同设计问题，通过多目标多智能体强化学习和分阶段课程训练（二级结构拓扑感知策略学习、三级结构迁移训练、前沿引导微调），实现同时满足二级碱基配对拓扑与三级几何结构的RNA序列设计。

    

    设计满足目标三级结构要求的RNA序列是RNA工程中的一个重要问题，在合成生物学和治疗学领域有着广泛的应用。现有的3D RNA设计方法已取得长足进展，但它们在很大程度上将目标折叠视为纯几何目标。这对RNA而言是不完整的，因为RNA的功能折叠不仅由三级几何结构决定，还由二级碱基配对拓扑决定。因此，我们将RNA逆设计表述为拓扑与几何协同设计问题，并提出了TANGO——一个面向该任务的多目标多智能体强化学习框架。TANGO采用分阶段课程学习：首先从二级结构目标中学习拓扑感知策略，然后将该策略迁移到三级结构目标上训练一个通用的协同设计模型，最后应用前沿引导的微调来细化特定目标下二级结构保真度与三级结构几何之间的权衡。

    arXiv:2610.20277v1 Announce Type: new  Abstract: Designing RNA sequences that satisfy target tertiary-structure objectives is an important problem in RNA engineering, with broad applications in synthetic biology and therapeutics. Existing 3D RNA design methods have made substantial progress, but they largely treat the target fold as a geometry-only objective. This is incomplete for RNA, whose functional folds are specified not only by tertiary geometry but also by secondary base-pairing topology. We therefore formulate RNA inverse design as a topology-and-geometry co-design problem and propose TANGO, a multi-objective multi-agent reinforcement learning framework for this task. TANGO adopts a staged curriculum: it first learns a topology-aware policy from secondary-structure targets, then transfers this policy to tertiary-structure targets to train a general co-design model, and finally applies frontier-guided fine-tuning to refine target-specific trade-offs among secondary-structure fi
    
[^244]: 精细化拓扑遍次调度：SE(3)等变结构预测中动态k近邻图的阻尼信赖域补偿

    Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction

    [https://arxiv.org/abs/2610.19246](https://arxiv.org/abs/2610.19246)

    提出一种双遍拓扑调度方案，将动态kNN图构建基准的选择形式化为以逐块重建（PB）为唯一不动点的阻尼不动点迭代，并借助不翻转证书、最小方差（BLUE）重放增益与ℓ∞信赖域投影进行保守补偿，从而在SE(3)等变结构预测中以低成本逼近逐块重建的拓扑精度。

    

    等变结构预测流水线在坐标更新与k近邻（kNN）图构建之间交替进行，其中图的成员关系是坐标的离散、分段常数函数。在每个块边界（PB）重建该图是训练轮次中最昂贵的部分，而复用一个冻结图（F）虽然快速，却会留下拓扑差距。我们研究了一种双遍设计：廉价的遍次记录少量状态信息（间隔边距、kNN半径、实测的每边界位移），第二遍则进行保守补偿。我们将图构建基准的选择形式化为一个阻尼不动点迭代，其唯一不动点恰好为PB，并由此推导出：(i) 具有双侧常数 $m_i/4$ 和单侧常数 $m_i/2$ 的精确不翻转（non-flip）证书；(ii) 最小方差（BLUE）重放增益 $\gamma^*$ 及由此产生的可行性交叉点；(iii) 一个 $\ell_\infty$ 信赖域投影，可保……

    arXiv:2610.19246v1 Announce Type: new  Abstract: Equivariant structure-prediction pipelines alternate coordinate updates with the construction of a k-nearest-neighbour (kNN) graph whose membership is a discrete, piecewise-constant function of the coordinates. Rebuilding that graph at every block boundary (PB) is the expensive part of an epoch, whereas reusing one frozen graph (F) is fast but leaves a topology gap. We study a two-pass design in which the cheap pass records a small amount of state (margins, kNN radii, measured per-boundary displacements) and a second pass compensates conservatively. We formulate the choice of the graph-construction basis as a damped fixed-point iteration whose unique fixed point is exactly PB, and derive (i) exact non-flip certificates with the two-sided constant $m_i/4$ and one-sided constant $m_i/2$, (ii) the minimum-variance (BLUE) replay gain $\gamma^*$ and the resulting feasibility crossover, (iii) an $\ell_\infty$ trust-region projection that prese
    
[^245]: 重新审视生成式检索与稠密检索：一个静态价值先验即可逆转差距

    Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap

    [https://arxiv.org/abs/2610.19065](https://arxiv.org/abs/2610.19065)

    生成式检索相对稠密检索的价值优势主要源于物品流行度这一静态先验，在稠密检索中加入一步静态价值重排即可逆转差距并保持更高召回率，且工业级实验表明 140 亿参数的 SID-LLM 仍不及小 47 倍的稠密编码器。

    

    随着语言模型和语义物品标识符（SID）的不断改进，生成式检索（GR）已成为序列推荐中强有力的候选生成方法。然而，在在线零售候选生成场景中，GR 并非稠密检索（DR）的全面更优替代方案。我们在两个 Amazon Toys 数据集上用不同指标检验了二者的差异：在两种方法共同覆盖的用户中，GR 在最佳相关物品价值上领先，而 DR 在精确目标召回率以及排序列表整体的相关价值上领先。通过深入分析，我们发现流行度对 GR 的价值领先贡献最大。在 DR 上仅通过一步重排加入相同的静态物品价值，即可逆转这一领先，同时保持召回率高于 GR。最后，在一个使用 LLM 判定相关性的工业级内部电商数据集上，我们证明了 140 亿参数的 SID-LLM 仍不如比它小 47 倍的稠密编码器。

    arXiv:2610.19065v1 Announce Type: cross  Abstract: Generative retrieval (GR) has emerged as a strong candidate generator for sequential recommendation as language models and semantic-item identifiers have improved. However, GR is not a uniformly stronger replacement of dense retrieval (DR) in online retail candidate generation. We examine their differences on two Amazon Toys datasets under different metrics. GR leads on best relevant item value among users both methods cover, while DR leads on exact-target recall and relevant value across the ranked list. With deep analysis, we find that popularity contributes the most to GR's measured value lead. Adding the same static item value to DR in one reranking step reverses this lead and keeps recall above GR. Finally, on an industrial-scale internal e-commerce dataset with LLM-judged relevance, we show that a 14B SID-LLM still underperforms a $47\times$ smaller dense encoder.
    
[^246]: 超越方程搜索：基于SCOPE与ClosureBench的猜想引导科学发现

    Beyond Equation Search: Conjecture-Guided Discovery with SCOPE and ClosureBench

    [https://arxiv.org/abs/2610.18938](https://arxiv.org/abs/2610.18938)

    该论文提出SCOPE多智能体框架，将符号回归从优化单个方程转向以科学猜想为核心的研究范式，通过多个科学家智能体独立建模、主动实验与证据交换来检验和修正猜想，并配套ClosureBench基准评估猜想驱动式科学发现。

    

    大多数符号回归方法都在优化候选方程，但提升方程的拟合度并不能确定其背后的科学猜想应该被保留、修改还是摒弃。一个猜想可能存在多种数学实现方式，而精确度相近的方程可能蕴含着不同的解释。这促使研究重心从精炼单个表达式转向发展和检验指导这些表达式的科学猜想。我们提出了SCOPE，一个能够维持多条猜想驱动研究轨迹的框架。多个科学家智能体通过方程建模、问题提出和主动实验独立地探究各自的猜想，随后交换证据以修正其猜想和研究方向。他们的发现被整合为一份科学报告，将模型、结论与证据联系起来。现有的方程发现基准主要评估恢复出的表达式，而缺乏对证据层面的考量。（注：摘要原文在此处被截断）

    arXiv:2610.18938v1 Announce Type: cross  Abstract: Most symbolic regression methods optimize candidate equations, but improving an equation's fit does not establish whether its underlying scientific conjecture should be retained, revised, or rejected. A conjecture may admit multiple mathematical realizations, and similarly accurate equations may imply different explanations. This motivates a shift from refining individual expressions to developing and testing the conjectures that guide them. We introduce SCOPE, a framework that sustains multiple conjecture-driven research trajectories. Multiple scientist agents independently pursue conjectures through equation modeling, question formulation, and active experimentation, then exchange evidence to revise their conjectures and research directions. Their findings are consolidated into a scientific report linking models, conclusions, and evidence. Existing equation-discovery benchmarks primarily assess recovered expressions, leaving the evid
    
[^247]: 基于Transformer语言模型的图元网络

    Graph Metanetworks for Transformer Based Language Models

    [https://arxiv.org/abs/2610.18471](https://arxiv.org/abs/2610.18471)

    该论文提出了一种基于图元网络的方法，全面捕捉Transformer语言模型权重空间中的丰富对称性（包括神经元置换、注意力矩阵的连续一般线性对称性及注意力头置换），并处理词元嵌入等语言模型特有组件。

    

    权重空间学习将神经网络的权重视为数据，由称为元网络的其他模型进行处理。一个关键的挑战是不同的权重可以实现相同的函数，这些权重通过一组对称性相关联，元网络应当尊重这些对称性。虽然针对MLP等简单架构的对称性感知元网络已经发展成熟，但对transformer尤其是基于transformer的语言模型的全面处理仍然缺失。重要的是，transformer的权重空间具有更丰富的对称群：除了神经元置换之外，注意力机制还引入了注意力矩阵的连续一般线性对称性以及整个注意力头的置换。现有的transformer权重处理方法要么只捕捉了这些对称性的一部分，要么受限于单一固定架构，而且没有任何方法考虑语言模型特有的组件（如词元嵌入）。基于图元网络，我们提出了一种基于图的

    arXiv:2610.18471v2 Announce Type: replace  Abstract: Weight-space learning treats the weights of neural networks as data, processed by other models called metanetworks. A key challenge is that different weights can realize the same function, related by a group of symmetries the metanetwork should respect. While symmetry-aware metanetworks for simple architectures such as MLPs are well developed, a comprehensive treatment of transformers, and in particular transformer-based language models, is still missing. Importantly, their weight space has a richer symmetry group: beyond neuron permutations, attention introduces continuous general-linear symmetries of the attention matrices and permutations of entire heads. Existing methods for transformer weights either capture only part of these symmetries or are restricted to a single, fixed architecture, and none accounts for language-model-specific components such as the token embeddings. Building on graph metanetworks, we propose a graph-based
    
[^248]: 大语言模型驱动的音色生成与插值中的结构能动性与表征能动性

    Structural and Representational Agency in LLM-Driven Timbre Generation and Interpolation

    [https://arxiv.org/abs/2610.12532](https://arxiv.org/abs/2610.12532)

    本研究通过对一个实际运行的LLM驱动音色界面进行测量，首次证明用户真实可控的声音空间（可精确刻画为维度随放置声音数递增的zonotope）与界面中由语言模型自我报告的位置（误差约为该空间宽度的三倍）是可分离的，揭示了LLM创意界面中“实际能动性”与“表征能动性”之间的脱节。

    

    生成式创意界面通常会报告用户在可能性空间中所处的位置。这类显示暗示所展示的量值是从作品本身测量得到的；但在越来越多的系统中，这些量值实际上是由一个描述其自身输出的语言模型预测出来的，而两者在屏幕上无法区分。大多数创意领域缺乏公认的测量标准来检验这种自我报告；而音频领域恰好存在这样的标准。在一项针对实际运行的大语言模型驱动音色界面的测量研究中，用户所能实际产生的效果被证明与界面所报告的内容是可分离的。前者可以被精确刻画：可达集是一个zonotope（区域多胞体），每放置一个声音其维度增加一，因此模型可以在其中行动的空间显然属于用户自身。后者则不准确：显示的位置与实际测量值相差约为该区域宽度的三倍，且该误差有两个独立的来源：模……（原文摘要在此截断）

    arXiv:2610.12532v1 Announce Type: cross  Abstract: Generative creative interfaces routinely report the user's position within a space of possibilities. Such displays imply that the quantities shown were measured from the artefact; in a growing class of systems they are instead predicted by a language model describing its own output, and the two are indistinguishable on screen. Most creative domains lack an agreed measurement against which such a self-report can be checked; audio admits one. In a measurement study of a working LLM-driven timbre interface, what a user can cause proves separable from what the interface reports. The first is exactly characterisable: the reachable set is a zonotope whose dimension increases by one per sound placed, so the space the model may act within is demonstrably the user's own. The second is inaccurate. The displayed position differs from measurement by approximately three times the width of that region, and the error has two independent sources: mode
    
[^249]: 面向测试时扩散对齐的 Best-of-N 引导方法

    Best-of-$N$ Guidance for Test-time Diffusion Alignment

    [https://arxiv.org/abs/2610.05108](https://arxiv.org/abs/2610.05108)

    该论文提出 Best-of-N 引导（BoNG）方法，将 BoN 选择的原理直接融入逆向扩散过程，通过对去噪粒子进行在线 BoN 选择来调整采样轨迹，从而在测试时更有效地将扩散模型与人类偏好对齐。

    

    扩散模型具有强大的生成性能，但常常难以使生成的样本与通过奖励模型衡量的人类偏好对齐。一种简单而有效的测试时对齐算法是 Best-of-N (BoN) 采样，它从预训练的扩散模型中抽取 N 个独立同分布的样本，并输出奖励最高的单个样本。尽管 BoN 在实践中取得了成功，但它对奖励信息的利用十分有限，因为奖励信息仅在最终选择阶段才被纳入，而在采样过程中并不影响逆向扩散轨迹。因此，BoN 采样并不能提升生成样本的平均对齐程度，且主要适用于单输出场景。我们提出了 Best-of-N 引导（BoNG），这是一种将 BoN 采样原理直接融入逆向扩散过程的新颖方法。BoNG 对去噪粒子执行在线 BoN 选择，并据此调整逆向扩散过程（摘要内容在此处被截断）。

    arXiv:2610.05108v2 Announce Type: replace-cross  Abstract: Diffusion models achieve strong generative performance but often struggle to align generated samples with human preferences measured by a reward model. A simple yet effective algorithm for test-time alignment is Best-of-$N$ (BoN) sampling, which draws $N$ i.i.d. samples from a pre-trained diffusion model and outputs the single highest-reward sample. Despite its empirical success, BoN makes limited use of reward information, as it is incorporated only at the final selection stage without influencing the reverse diffusion trajectory during sampling. Consequently, BoN sampling does not improve the average alignment of generated samples and is primarily suited to single-output settings. We propose Best-of-$N$ Guidance (BoNG), a novel method that integrates the principle of BoN sampling directly into the reverse diffusion process. BoNG performs online BoN selection over denoising particles and adjusts the reverse diffusion process t
    
[^250]: VIGOR：基于模型的强化学习中通过潜空间一致性实现零样本视觉泛化

    VIGOR: Zero-Shot Visual Generalization via Latent-Space Consistency in Model-Based Reinforcement Learning

    [https://arxiv.org/abs/2610.02801](https://arxiv.org/abs/2610.02801)

    VIGOR通过非对称弱到强增强与潜空间一致性约束，使基于模型的强化学习在保留样本效率的同时，能够零样本泛化到背景变化、光照变化等未见过的视觉干扰。

    

    基于模型的强化学习（MBRL）通过在学习的潜在动力学中进行规划，实现了强大的样本效率，但在面对背景变化、光照变化或相机移动等未见过的视觉干扰时，其性能会大幅下降。与无模型强化学习（编码器扰动仅影响单步预测）不同，MBRL存在两级脆弱性：视觉干扰首先使编码器输出偏离分布，随后这些误差会在规划时域内通过递归潜在轨迹推演不断累积放大。我们提出VIGOR，这是一个能够在保留其MBRL骨干样本效率的同时，实现对未见视觉干扰进行零样本泛化的框架。VIGOR集成了三个相互关联的组件：（i）非对称弱到强增强，在单个批次内配对仅弱增强与弱到强增强的潜在视图……

    arXiv:2610.02801v1 Announce Type: new  Abstract: Model-based reinforcement learning (MBRL) achieves strong sample efficiency by planning within learned latent dynamics, yet its performance degrades substantially under unseen visual distractions such as background variations, lighting changes, or camera shifts. Unlike model-free RL, where encoder perturbations affect only single-step predictions, MBRL suffers from a two-level vulnerability: visual distractions first push encoder outputs out of distribution, and these errors then compound through recursive latent rollouts over the planning horizon. We propose visual generalization via latent-space consistency in model-based RL (VIGOR), a framework that enables zero-shot generalization to unseen visual distractions while retaining the sample efficiency of its MBRL backbone. VIGOR integrates three interdependent components: (i) asymmetric weak-to-strong augmentation, which pairs weak-only and weak-to-strong latent views within a single bat
    
[^251]: 离散空间上的不动点神经采样器

    Fixed-point neural samplers on discrete spaces

    [https://arxiv.org/abs/2610.01739](https://arxiv.org/abs/2610.01739)

    本文提出离散吉布斯迭代神经采样器，一种基于掩码扩散的不动点神经采样方法，解决了现有离散神经采样器的模式坍缩和收敛保证问题，可高效扩展至高维系统并支持跨条件的摊销采样。

    

    在无法访问数据的情况下，从离散、未归一化的分布中进行采样是一个具有挑战性的问题。神经采样器提供了一种有前景的方法，它可以直接基于密度评估来训练生成模型。尽管近年来取得了一些进展，但现有的离散神经采样器容易发生模式坍缩，在通过不动点迭代训练时缺乏收敛性保证，并且通常与特定的参考过程（如掩码扩散或均匀扩散）绑定在一起。在这项工作中，我们提出了离散吉布斯迭代神经采样器，这是一种能够解决上述局限性的不动点神经采样器，可实现高效、可扩展的学习，并在实践中显著减少模式坍缩。我们的框架建立在掩码扩散的基础上，同时还可扩展到分布对之间的传输。我们证明了该方法能够有效地扩展到高维系统，支持跨不同条件的摊销采样，并能实现精确的（摘要在此处被截断）。

    arXiv:2610.01739v1 Announce Type: new  Abstract: Sampling from discrete, unnormalized distributions without access to data is a challenging problem. Neural samplers offer a promising approach by training generative models from density evaluations directly. Despite recent progress, existing discrete neural samplers are prone to mode collapse, come without convergence guarantees when trained via fixed-point iterations, and are often tied to a specific reference process such as masked or uniform diffusion. In this work, we introduce Discrete Gibbs Iterative Neural Sampler, a fixed-point neural sampler that addresses these limitations, enabling efficient, scalable learning, substantially reducing mode collapse in practice. Our framework builds on masked diffusion and also extends to transport between pairs of distributions. We demonstrate that the resulting method scales effectively to high-dimensional systems, supports amortized sampling across different conditions, and enables accurate e
    
[^252]: 混合Transformer-SSM架构的学习率迁移

    Learning Rate Transfer for Hybrid Transformer-SSM Architectures

    [https://arxiv.org/abs/2610.01172](https://arxiv.org/abs/2610.01172)

    该论文发现，在实际使用简化ZOH Mamba的混合Transformer-SSM架构中，尽管SSM操作不满足μP的理论前提，仅凭原始μP规则即可在宽256-2048、深4-32直至十亿参数规模上实现接近零的学习率迁移差距。

    

    arXiv:2610.01172v1 公告类型：新论文 摘要：我们研究了结合Transformer与状态空间模型（SSM）模块的混合架构的学习率（LR）缩放问题，这类架构已被多个近期的生产级语言模型所采用。特别地，我们关注两方面之间的差距：一是在无限宽度且状态规模增长条件下、基于零阶保持（ZOH）离散化为SSM推导出的理论缩放规则；二是采用固定状态规模的简化ZOH Mamba的领域标准实践实现。令人惊讶的是，在这种实际场景下，混合架构仅使用原始的μP规则，就能在宽度256-2048、深度4-32、直至十亿参数规模上实现接近零的学习率迁移差距，尽管SSM操作超出了μP的张量程序可表示性条件，且我们测试的每一种参数化方式都无法通过μP正确性的标准坐标检查诊断。我们将此现象归因于混合架构中学习率迁移的双条件分解：一个……（摘要被截断）

    arXiv:2610.01172v1 Announce Type: new  Abstract: We study learning rate (LR) scaling for hybrid architectures combining Transformer and State-Space Model (SSM) blocks, a class adopted by several recent production language models. In particular, we focus on the gap between the theoretical scaling rules derived for SSMs under zero-order-hold (ZOH) discretization at infinite width with growing state size, and the field-standard practical implementations using simplified-ZOH Mamba at fixed state size. Surprisingly, in this practical regime hybrid architectures achieve a near-zero LR transfer gap across widths 256-2048 and depths 4-32 up to billion-parameter scale using only the original $\mu$P prescription, even though SSM operations fall outside its Tensor Programs representability conditions and every parameterization we test fails the standard coordinate-check diagnostic of $\mu$P correctness. We attribute this to a two-condition decomposition of LR transfer in hybrid architectures: a g
    
[^253]: 通过零空间投影对LoRA微调的大语言模型进行后门净化

    Backdoor Purification for LoRA-Tuned LLMs via Null-Space Projection

    [https://arxiv.org/abs/2610.00685](https://arxiv.org/abs/2610.00685)

    提出了一种基于零空间投影的后门净化方法，无需触发器先验知识、干净参考样本或重新训练，即可净化遭受后门攻击的LoRA微调大语言模型，在降低攻击成功率的同时保留模型的通用能力与下游技能。

    

    随着大语言模型（LLMs）和参数高效微调（PEFT）方法的快速普及，后门攻击的风险变得更加严重。现有的后门净化方法通常至少依赖于以下强假设之一：对触发器的先验知识、对干净参考样本的访问权限，或进行激进的重新训练，而且这些方法往往缺乏全面的评估。这些限制极大地限制了它们的实际适用性。为了克服这些挑战，我们的工作提出了一种在不依赖这些假设、甚至无需对可疑参数进行事后重新训练的情况下净化LoRA微调LLM的方法。我们的目标是在显著降低攻击成功率（ASR）的同时，既保留基础模型的通用能力，又保留通过适配器学习到的新下游技能。通过一系列消融实验，我们将该方法从文本分类任务中的单层逐步扩展（摘要在此处被截断）。

    arXiv:2610.00685v1 Announce Type: new  Abstract: With the rapid adoption of large language models (LLMs) and parameter-efficient fine-tuning (PEFT) methods, the risk of backdoor attacks has become more severe. Existing backdoor purification methods typically rely on at least one of the strong assumptions, such as prior knowledge of triggers, access to clean references, or aggressive retraining, and they often lack comprehensive evaluations. These constraints substantially limit their practical applicability. To overcome these challenges, our work proposes purifying LoRA-tuned LLMs without these assumptions and even without post-hoc retraining of the suspect parameters. Our objective is to significantly reduce the attack success rates (ASR) while preserving both (i) the base model's general capabilities and (ii) the new downstream skills learned through the adapter. Through a series of ablation studies, we progressively scale our approach from a single layer in a text classification set
    
[^254]: 通过专家隔离与关停实现大语言模型后门遏制

    Backdoor Containment via Expert Quarantine and Shutdown in LLMs

    [https://arxiv.org/abs/2610.00663](https://arxiv.org/abs/2610.00663)

    本文提出隔离专家关停策略，通过正则化引导的MoE类架构让后门在训练中形成但被引导至指定的隔离专家组件，部署时可将该组件关停，从而以计算高效的方式遏制LLM后门攻击。

    

    被植入后门的大语言模型（LLM）在良性输入下表现正常，但在隐藏触发器作用下会产生攻击者指定的输出。现有防御方法涵盖四个阶段——训练前、训练中、训练后和推理时——并且都采用以下两种底层策略之一：要么抑制后门学习（通过过滤中毒数据或在优化过程中中断其习得），要么先学习再净化（在完全后门化的模型形成后，通过修复模型权重或对输入进行门控）。我们提出第三种策略：学习但引导——允许后门在训练期间形成，但将其引导至一个指定的、被隔离的组件中，该组件可在部署时被禁用。为此，我们提出了隔离专家关停（QES），这是一种在正则化引导的MoE类设置中构建的计算高效遏制策略。具体而言，给定一个中毒数据集，QES在基于Transformer的语言模型中增强了路由……

    arXiv:2610.00663v1 Announce Type: new  Abstract: Backdoored large language models (LLMs) can behave normally on benign inputs while producing attacker-specified outputs under hidden triggers. Existing defenses span four stages--prior-training, in-training, post-training, and inference-time--and share one of two underlying strategies: either suppress backdoor learning (by filtering poisoned data or interrupting its acquisition during optimization) or learn, then purify (by repairing model weights or gating inputs after a fully backdoored model has formed). We propose a third strategy, learn, but channel: allow backdoor formation during training but route it into a designated, quarantined component that can be disabled at deployment. To this end, we propose Quarantined Expert Shutdown QES, a computationally efficient containment strategy built in a regularization-steered MoE-like setting. Specifically, given a poisoned dataset, QES augments a Transformer-based language model with routed 
    
[^255]: 流形约束的初始噪声优化实现高效生成模型对齐

    Manifold-Constrained Initial Noise Optimization for Efficient Generative Model Alignment

    [https://arxiv.org/abs/2610.00365](https://arxiv.org/abs/2610.00365)

    提出ZeNOVA，一种免梯度的初始噪声优化方法，通过退火软值引导、流形约束超球面朗之万动力学和Metropolis-Hastings跳跃，在黑盒奖励场景下实现稳定高效的生成模型对齐。

    

    近期蒸馏与流映射模型的进展使高质量数据的一步或几步确定性生成成为可能，并催生了一类新的奖励对齐方法，这类方法直接优化来自高斯分布的初始噪声。然而，大多数现有的初始噪声优化方法依赖于一阶梯度信息，这在黑盒奖励场景中要么无法适用，要么存在不稳定和低效的问题。本文提出ZeNOVA，一种稳定且高效的免梯度初始噪声对齐方法。具体而言，我们通过退火软值引导、流形约束的超球面朗之万动力学以及Metropolis-Hastings跳跃机制，解决了现有算法在黑盒场景中的主要难题。在图像和视频生成模型上的大量实验表明，ZeNOVA通过将初始噪声朝更高质量方向优化，性能优于所有被评估的零阶基线方法。

    arXiv:2610.00365v1 Announce Type: cross  Abstract: Recent advances in distillation and flow-map models have enabled deterministic one- or few-step generation for high-quality data, facilitating a new branch of reward alignment approaches that directly optimize the initial noise from a Gaussian distribution. However, most existing initial-noise optimization methods rely on first-order gradient information, which is either inapplicable or suffers from instability and inefficiency in black-box reward scenarios. Here, we introduce ZeNOVA, a stable and efficient initial noise alignment method in a gradient-free manner. Specifically, we address existing algorithms' major challenge in black-box scenarios through annealed soft-value guidance, manifold-constrained hyperspherical Langevin dynamics, and Metropolis-Hastings jumping. Extensive experiments on image and video generative models show that ZeNOVA outperforms all evaluated zeroth-order baselines by optimizing the initial noise toward hig
    
[^256]: 面向蛋白质定向进化的结构感知强化学习

    Structure-aware Reinforcement Learning for Protein Directed Evolution

    [https://arxiv.org/abs/2609.39048](https://arxiv.org/abs/2609.39048)

    StructEvo提出了一种结构感知的强化学习框架，通过增量结构融合编码器动态整合空间知识，并利用结构对齐的分层动作网络分解突变空间，在克服突变体结构稀缺难题的同时超越了现有的蛋白质定向进化方法。

    

    蛋白质优化仍然是生命科学领域一个长期追求的目标。现有的机器学习辅助定向进化（MLDE）方法主要依赖仅序列特征，忽视了蛋白质结构中编码的关键空间约束和共进化相互作用。然而，由于可靠的突变体结构十分稀缺，直接整合结构信息仍然充满挑战。为了解决这些问题，我们提出了StructEvo，一种新颖的面向蛋白质定向进化的结构感知强化学习框架。StructEvo采用增量结构融合编码器，通过特征差异来近似突变体的结构特征，从而实现空间知识的动态整合。随后，通过结构对齐的分层动作网络将庞大的突变空间分解为可管理的子空间，同时几何约束进一步稳定了增量特征的学习。我们的方法优于先前的最先进方法。

    arXiv:2609.39048v1 Announce Type: cross  Abstract: Protein optimization remains a longstanding goal in life sciences. Existing machine learning-assisted directed evolution (MLDE) methods primarily rely on sequence-only features, overlooking the critical spatial constraints and co-evolutionary interactions encoded in protein structures. However, directly integrating structural information remains challenging due to the scarcity of reliable mutant structures. To address these issues, we propose StructEvo, a novel structure-aware reinforcement learning framework for protein directed evolution. StructEvo employs a delta-structure fusion encoder to approximate mutant structure features via feature differences, enabling dynamic incorporation of spatial knowledge. The vast mutation space is then decomposed into manageable subspaces through a structure-aligned hierarchical action network, while a geometric constraint further stabilizes delta feature learning. Our approach outperforms prior sta
    
[^257]: Muon 扬弃了大语言模型预训练中的稳定性边缘

    Muon Sublates the Edge of Stability in LLM Pretraining

    [https://arxiv.org/abs/2609.34915](https://arxiv.org/abs/2609.34915)

    Muon 优化器打破了大语言模型预训练中梯度下降的稳定性边缘耦合框架，其损失中性边界与时间对齐遵循不同几何规律，且更新方向始终远离相干反转、训练持续改善。

    

    Muon 优化器正被越来越多地用于语言模型预训练，但其大步长动态无法被梯度下降（GD）经典的“稳定性边缘”（EoS）理论图景所刻画。在 GD 中，损失中性、等幅更新反转与边缘稳定性汇聚于一个由学习率决定的单一边缘。我们证明 Muon 打破了这种耦合。对于随机无动量的 Muon，我们推导出一个经相干性校正的条件损失中性边界 $2\rho_b/\eta$，而时间对齐则遵循另一套独立的几何规律。受控实验表明，损失平衡与时间对齐对学习率和批大小的响应方式各不相同。在我们的语言模型实验中，130M 规模的类 Llama 模型运行表现出损失边界跟踪并伴随弱负对齐，而所研究的 1B 模型配置则展现出更强的部分抵消效应；在两种设置下，更新方向都远离相干反转，同时训练持续改善。

    arXiv:2609.34915v2 Announce Type: replace  Abstract: Muon is increasingly used for language-model pretraining, yet its large-step dynamics are not captured by the classical edge-of-stability (EoS) picture of gradient descent (GD). In GD, loss neutrality, equal-magnitude update reversal, and marginal stability meet at a single learning-rate-dependent edge. We show that Muon breaks this coupling. For stochastic no-momentum Muon, we derive a coherence-corrected conditional loss-neutral boundary $2\rho_b/\eta$, while temporal alignment follows a separate geometry. Controlled experiments show that loss balance and temporal alignment respond differently to learning rate and batch size. Across our language model experiments, the 130M Llama-like LLM runs exhibit loss-boundary tracking with weak negative alignment, whereas the studied 1B LLM configuration shows stronger partial cancellation; in both settings, directions remain far from coherent reversal while training continues to improve. Thes
    
[^258]: 物理证明的联邦学习：保障关键水利基础设施中协同异常检测的安全性

    Physics-Attested Federated Learning: Securing Collaborative Anomaly Detection in Critical Water Infrastructure

    [https://arxiv.org/abs/2609.34804](https://arxiv.org/abs/2609.34804)

    该论文提出将从物理过程自动挖掘出的不变量作为联邦更新的可验证准入要求，从而有效防御针对关键水利基础设施协同异常检测的数据投毒攻击。

    

    联邦学习使工业运营商能够在不披露专有运营遥测数据的情况下训练共享的入侵检测模型。然而，现有的防御手段严格在更新空间中运行，使得聚合器对数据投毒攻击视而不见；由伪造的遥测数据导出的模型更新与诚实的贡献无法区分。我们将过程不变量——在正常运行中始终成立的物理关系（例如质量守恒）——从运行时检测启发式方法重新定义为联邦更新的可验证准入要求，这些不变量可从干净的操作数据中自动挖掘。我们在两个物理水利测试平台（SWaT、WADI）和一个分布式基准（BATADAL）上评估了该准入门控机制，针对遥测伪造、仅暴露重放投毒以及具备不变量感知能力的自适应对手，测试了七种聚合规则。在三个测试平台中，挖掘出的不变量对25个诚实的……（原文截断）

    arXiv:2609.34804v2 Announce Type: replace-cross  Abstract: Federated learning enables industrial operators to train shared intrusion detection models without disclosing proprietary operational telemetry. However, existing defences operate strictly in update space, leaving aggregators blind to data poisoning; model updates derived from fabricated telemetry remain indistinguishable from honest contributions. We repurpose process invariants, physical relations that always hold in normal operation, such as conservation of mass, from runtime detection heuristics into a verifiable admission requirement for federated updates, mined automatically from clean operational data. We evaluate this admission gate across two physical water testbeds (SWaT, WADI) and a distribution benchmark (BATADAL), testing seven aggregation rules against telemetry fabrication, exposure-only replay poisoning, and an invariant-aware adaptive adversary. Across three testbeds the mined invariants reject none of 25 hones
    
[^259]: AnchorRep：通过表示排斥防御大语言模型的跨模型对抗迁移攻击

    AnchorRep: Defending LLMs Against Cross-Model Adversarial Transfer via Representation Repulsion

    [https://arxiv.org/abs/2609.32602](https://arxiv.org/abs/2609.32602)

    提出AnchorRep防御方法，利用轻量级LoRA适配器将有害提示词的内部表示与锚点模型的表示相排斥，无需对抗样本即可将跨模型对抗迁移攻击的成功率降低至1.1%以下。

    

    arXiv:2609.32602v2 公告类型：replace-cross 摘要：在单个开源权重大语言模型上优化的对抗攻击可以迁移并越狱架构不同的模型，这使得拥有对一个模型白盒访问权限的攻击者能够攻陷独立部署的其他系统。这在模型之间造成了共享的漏洞，然而现有的防御方法并非针对这种跨模型威胁而设计。我们发现跨模型迁移攻击与共享的内部表示几何结构相关，这使其成为一个天然的防御目标。AnchorRep通过一个轻量级的LoRA适配器直接针对这种几何结构进行防御，将被防御模型对有害提示词的内部表示推离冻结的锚点模型对相同提示词的内部表示。训练过程仅使用少量有害提示词，无需任何对抗样本。在五个模型和四种架构家族上的实验表明，AnchorRep将2,000次迁移攻击的跨模型攻击成功率降低至≤1.1%（其中两个模型上为0%），包括在Mistr……（摘要截断于此）

    arXiv:2609.32602v2 Announce Type: replace-cross  Abstract: Adversarial attacks optimized on a single open-weight LLM can transfer to and jailbreak architecturally different models, allowing an attacker with white-box access to one model to compromise independently deployed systems. This creates a shared vulnerability across models, yet existing defenses are not designed for this cross-model threat. We find that cross-model transfer aligns with shared internal representation geometry, making it a natural defense target. AnchorRep targets this geometry directly with a lightweight LoRA adapter that pushes the defended model's internal representations of harmful prompts away from those of a frozen anchor model on the same prompts. Training uses a small set of harmful prompts and no adversarial examples. Across five models and four architectural families, AnchorRep reduces cross-model attack success rate to <=1.1% on 2,000 transferred attacks (0% on two), including the largest drop on Mistr
    
[^260]: 面向慢性健康管理的基于同伴的反事实路径规划

    Peer-Grounded Counterfactual Path Planning for Chronic Health Management

    [https://arxiv.org/abs/2609.30838](https://arxiv.org/abs/2609.30838)

    该论文提出POROS框架，根植于自我效能理论与社会比较理论，通过从真实相似患者的观测状态构建行为进展图，以反事实路径规划生成有同伴证据支撑、能保证健康指标单调改善的渐进式分步干预方案，克服了现有方法只给目标、缺少路径、缺乏同伴依据的缺陷。

    

    慢性疾病管理中有效的行为干预所需的并非单一的处方，而是一系列渐进式的小步骤，每一步都立足于真实的、相似的个体已经切实达成的成果。反事实解释为这种指导提供了一条自然的计算路径，它回答的是：何种行为改变本可以带来更好的结果。但现有方法仅返回一个目标状态却没有通往该状态的路径，无法保证途中健康指标的单调改善，也没有从同伴行为中汲取证据——这等于要求患者一步跨越巨大的差距，而这恰恰是最不可能被尝试的推荐结构。我们提出了POROS（基于同伴的状态最优路径规划），这是一个领域无关的框架，根植于班杜拉的自我效能理论与费斯廷格的社会比较理论，构建了一个行为进展图——一个建立在观测到的患者状态之上的有向无环图，其中每条边（摘要至此截断）。

    arXiv:2609.30838v1 Announce Type: new  Abstract: Effective behavioral intervention in chronic disease management requires not a single prescription but a sequence of incremental steps, each grounded in what real, similar individuals have demonstrably achieved. Counterfactual explanation offers a natural computational route to such guidance, answering what change in behavior would have produced a better outcome. But existing methods return a target state without a route to it, guarantee no monotone health improvement along the way, and draw no evidence from peer behavior -- asking a patient to close a wide gap in one move, which is precisely the recommendation structure least likely to be attempted. We propose POROS (Peer-Grounded Optimal Routes Over States), a domain-agnostic framework rooted in Bandura's self-efficacy theory and Festinger's social comparison theory that constructs a Behavioral Progression Graph -- a directed acyclic graph over observed patient states in which every ed
    
[^261]: 为评审团喂入维度评分，而非整体判定：基于评分准则分解的视觉语言美学评审融合

    Feed the Panel Dimensions, Not Verdicts: Rubric-Decomposed Fusion of Vision-Language Aesthetic Judges

    [https://arxiv.org/abs/2609.27110](https://arxiv.org/abs/2609.27110)

    由多个视觉语言模型组成的评审团在融合整体美学判定时无法显著超越最佳单模型，而让各模型按人工评分准则对图像进行五维度评分并融合这些分数，则能可靠地击败最佳单个模型。

    

    视觉语言模型（VLM）被部署为图像美学的零样本评审器，而在证据薄弱的情况下，多个模型组成的评审团被推荐为提升此类评审可靠性的方法。在两个人工评分数据集EVA和PARA上，我们发现由整体评判模型组成的评审团，无论是对判定结果取平均还是通过学习到的组合器进行融合，都从未显著优于其最佳成员。评审团的价值取决于其被输入的内容。因此，我们让每个模型依据一份固定的人工编写的评分准则（rubric）对每张图像在五个维度上进行评分，并通过折外组合器跨模型族融合这些分数以及每个模型的判定结果。维度评分确实衡量了其标签所声称的内容：在剔除总体人工评分的影响后，在30个模型-属性组合中的28个里，维度提示比整体提示携带更多特定属性的信息。经过融合，它们在EVA数据集上所有十个三模型族评审团中都击败了最佳的单个VLM。

    arXiv:2609.27110v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are deployed as zero-shot judges of image aesthetics, and panels of several models are recommended, on thin evidence, as the way to make such judges reliable. On two human-rated datasets, EVA and PARA, we find that a panel of holistic judges never significantly beats its best member, whether the verdicts are averaged or fused by a learned combiner. What a panel is worth depends on what it is fed. We therefore have each model score each image on the five dimensions of a frozen, human-written rubric and fuse those scores, alongside each model's verdict, across model families with an out-of-fold combiner. The dimension scores measure what their labels claim: with the overall human score partialled out, a dimension prompt carries more attribute-specific information than the holistic prompt in 28 of 30 model-attribute cells. Fused, they beat the best single VLM in all ten three-family panels on EVA (against tha
    
[^262]: PROSWIN：基于太阳图像和深度分布回归的概率太阳风速度预报

    PROSWIN: Probabilistic Solar Wind Speed Forecasting Using Deep Distributional Regression From Solar Images

    [https://arxiv.org/abs/2609.26683](https://arxiv.org/abs/2609.26683)

    PROSWIN是一个结合太阳图像、磁图与深度分布回归的概率机器学习模型，可提前四天预报地球太阳风速度并实现良好的不确定性校准，同时引入预测分数指标以更好地评估高速太阳风流峰值。

    

    准确预测快速太阳风条件极具挑战性，因为不确定性很大，且传统的单值预测模型无法对其进行量化。特别是，高速太阳风流（HSS）可能对技术基础设施造成损害，在没有概率预报的情况下，无法可靠地评估其风险。我们提出了PROSWIN，这是一个概率机器学习模型，可提前四天预报地球的每小时太阳风速度（SWS）。该方法通过深度神经网络与分布回归算法相结合，融合了太阳图像和磁图数据。由于标准误差指标低估了HSS峰值的重要性，我们进一步引入了预测分数，这是一种联合奖励时间线准确性和HSS峰值准确性的模型选择指标。在14年的数据上，我们的预报实现了非常好的不确定性校准（平均偏差小于1%）。使用连续排序概率分数……

    arXiv:2609.26683v1 Announce Type: cross  Abstract: Accurately predicting fast solar wind conditions is challenging, as uncertainties are large and unquantified by traditional single-value prediction models. In particular, the risks of high-speed solar wind streams (HSSs), which can cause damage to technological infrastructure, cannot be reliably assessed without probabilistic forecasts. We present PROSWIN, a probabilistic machine learning model that forecasts the hourly solar wind speed (SWS) at Earth with a four-day lead time. The approach combines solar images and magnetograms using a deep neural network coupled to a distributional regression algorithm. Because standard error metrics underweight the relevance of HSS peaks, we further introduce the prediction score, a model-selection metric that jointly rewards timeline and HSS peak accuracy. On 14 years of data, our forecast achieves very well-calibrated uncertainties (<1% average deviation). Using the continuous ranked probability s
    
[^263]: RLVR²：基于可验证评分量规排序的强化学习

    RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking

    [https://arxiv.org/abs/2609.23457](https://arxiv.org/abs/2609.23457)

    本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。

    

    基于可验证奖励的强化学习（RLVR）正在从具有明确定义正确性信号的任务（如数学和代码）扩展到由多维度评分量规指定的多方面质量要求。由于策略优化在每次 rollout 中只消耗一个标量，基于评分量规的流程必须将多个标准的得分聚合为标量奖励。这种聚合通常被视为简单的分数缩放，但它实际上隐式地决定了训练过程中各质量维度之间的权衡方式。目前主流的做法是对每个标准进行归一化后取线性组合，这隐含地假设了基数分数差异在各标准之间是可比的，并且某一标准上的收益可以补偿另一标准上的失败；当各标准在语义上异质时，这两种假设都不可靠。我们提出了基于可验证评分量规排序的强化学习（RLVR²），这是一种面向基于评分量规的 RLVR 的可验证排序范式。对于每个标准……

    arXiv:2609.23457v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is expanding from tasks with well-defined correctness signals, such as mathematics and code, toward multifaceted quality requirements specified by multi-dimensional rubrics. Since policy optimization consumes one scalar per rollout, rubric-based pipelines must map multiple criterion scores into a scalar reward. This aggregation is often treated as score scaling, but it implicitly determines how quality dimensions trade off during training. The prevailing practice, normalizing each criterion and taking a linear combination, assumes that cardinal score differences are comparable across criteria and that gains on one criterion compensate for failures on another; both assumptions are unreliable when criteria are semantically heterogeneous. We propose Reinforcement Learning with Verifiable Rubric-based Ranking (RLVR$^2$), a verifiable ranking paradigm for rubric-based RLVR. For each criter
    
[^264]: 反例与充分条件：对《最优传输广义矩方法》一文的评论

    Counterexamples and Sufficient Conditions: Comments on "Optimally-Transported Generalized Method of Moments"

    [https://arxiv.org/abs/2609.20260](https://arxiv.org/abs/2609.20260)

    本文对Schennach & Starck提出的最优传输广义矩方法（OTGMM）估计量的核心定理给出反例，证明其定理2-6在原假设条件下不成立，并探讨使这些结论成立的充分条件。

    

    我们针对Schennach & Starck（2026a）提出的最优传输广义矩方法（OTGMM）估计量进行评论，并在其所述假设条件下给出针对定理2-6的反例。首先，小误差分析中所使用的假设不足以保证定理2的一致性和定理3的渐近正态性。其次，我们考察大误差分析，其中定理4声称OTGMM估计量等价于带修正矩的GMM估计量。我们证明，在标量模型中，定理4所选择的值不同于唯一的OTGMM最小化点，且违反了OTGMM样本矩约束。在一个满足定理5和定理6假设的过度识别模型中，拉格朗日乘子的第一个分量在OTGMM估计量和带修正矩的GMM估计量下具有不同的概率极限。在模型误设的情形下，OTGMM所选择的总体值取决于……（原文摘要至此截断）

    arXiv:2609.20260v1 Announce Type: cross  Abstract: We comment on the optimally-transported generalized method of moments (OTGMM) estimator proposed by Schennach & Starck (2026a) and give counterexamples to Theorems 2-6 under their stated assumptions. First, the assumptions used in the small-error analysis are insufficient for consistency in Theorem 2 and asymptotic normality in Theorem 3. Next, we consider the large-error analysis, in which Theorem 4 states that the OTGMM estimator is equivalent to a GMM estimator with modified moments. We show that in a scalar model, Theorem 4 selects a value that differs from the unique OTGMM minimizer and violates the OTGMM sample moment restriction. In an overidentified model satisfying the assumptions used in Theorems 5 and 6, the first component of the Lagrange multiplier has different probability limits under the OTGMM estimator and the GMM estimator with modified moments. Under misspecification, the population value selected by OTGMM depends on
    
[^265]: 值之前的Token即为键：混合架构如何组织归纳电路

    The Token Before the Value Is the Key: How Hybrid Architectures Organize Induction Circuits

    [https://arxiv.org/abs/2609.15545](https://arxiv.org/abs/2609.15545)

    该论文发现混合语言模型中归纳电路存在明确分工——“携带前驱信息”集中于局部/循环等高效层而“内容匹配”集中于全局层，且通过干预前驱支持（如滞后一掩码、移除卷积等）可以在不同层之间重新分配这些功能。

    

    混合语言模型能够在提升能力的同时提高效率，这引出了一个关键问题：架构上的互补性是如何转化为可学习的计算的？我们考察了已被确立的归纳角色：携带前驱信息、按内容匹配源信息、以及复制其值。这些位置敏感且基于内容的计算是如何在异构层之间进行分配的？我们提出了与层类型无关的成对探针，通过统一的块更新接口来追踪“携带”和“匹配”过程。在循环-全局和局部-全局混合架构中，“携带”集中于高效层，而“匹配”集中于全局接收层。所测得的局部贡献集中在滞后一上：即紧邻历史值之前的那个Token。通过滞后一掩码、卷积移除或早期学习率降低等方式改变前驱支持，可以在各阶段之间重新定位“携带”和“匹配”。源键恢复与固定值选择……

    arXiv:2609.15545v1 Announce Type: new  Abstract: Hybrid language models can improve capability as well as efficiency, raising the question of how architectural complementarity becomes learned computation. We examine the established induction roles of Carrying predecessor information, Matching a source by content, and Copying its value. How are these position-sensitive and content-based computations allocated across heterogeneous layers? We introduce layer-type-agnostic paired probes that track Carrying and Matching through a common block-update interface. In recurrent--global and local--global hybrids, Carrying concentrates in efficient layers and Matching in global receivers. The measured local contribution concentrates on lag one: the token immediately before the historical value. Changing predecessor support through lag-one masking, convolution removal, or early learning-rate reduction can relocate Carrying and Matching between stages. Source-key restoration and fixed-value selectio
    
[^266]: DynSHAP：迈向可解释的动态生存分析

    DynSHAP: Towards Explainable Dynamic Survival Analysis

    [https://arxiv.org/abs/2609.13042](https://arxiv.org/abs/2609.13042)

    DynSHAP是一个专为动态生存分析设计的SHAP可解释性框架，通过将时间-特征对作为Shapley博弈的参与者，并引入利用条件采样处理特征时间依赖关系的Temporal DynSHAP，从而能够准确解释纵向不规则输入与函数型生存输出。

    

    面向动态生存分析（DSA）的深度学习模型通过纳入纵向患者数据实现了强大的预测性能，但其黑盒特性限制了临床信任度和应用推广。现有的可解释性方法无法同时处理纵向的不规则输入和函数型的生存输出，这限制了它们在动态生存分析中的可用性。我们提出了DynSHAP，一个专门适用于动态生存分析的SHAP框架。它通过将时间-特征对视为Shapley博弈中的参与者，将常见的边际SHAP估计器扩展到了这一场景。我们进一步提出了Temporal DynSHAP，它学习特征随时间的线性依赖关系，并使用条件采样在解释中对这些依赖关系进行处理。当应用于具有已知真实归因的合成数据时，对于给定的最先进模型，Temporal DynSHAP比边际估计器能更准确地恢复出时间相关特征。应用于两个……（原文摘要此处截断）

    arXiv:2609.13042v1 Announce Type: new  Abstract: Deep learning models for dynamic survival analysis (DSA) achieve strong predictive performance by incorporating longitudinal patient data, but their black box nature limits clinical trust and adoption. Existing explainability methods cannot handle longitudinal, irregular inputs and functional survival outputs simultaneously, which limits their usability in DSA. We propose DynSHAP, a SHAP framework suited specifically for dynamic survival analysis. It extends common marginal SHAP estimators to this setting by treating time--feature pairs as players in the Shapley game. We further introduce Temporal DynSHAP, which learns linear dependencies in features over time and uses conditional sampling to address them in explanations. When applied to synthetic data with known ground-truth attributions, Temporal DynSHAP recovers temporally dependent features more accurately than marginal estimators for a given state-of-the-art model. Applied to two re
    
[^267]: 随机学习动力学中信息获取的速度极限

    Speed Limit for Information Acquisition in Stochastic Learning Dynamics

    [https://arxiv.org/abs/2609.08219](https://arxiv.org/abs/2609.08219)

    本研究将随机梯度下降建模为马尔可夫随机过程，推导出费希尔信息流的速度极限，定量约束了可训练参数获取数据中潜在变量信息的速率，并将信息流分解为漂移（确定性学习力）与噪声（SGD涨落）两部分贡献。

    

    神经网络通过学习获取内部表示。在本工作中，我们将随机梯度下降（SGD）表述为一个马尔可夫随机过程，并推导出一个费希尔信息流速度极限，该极限约束了可训练参数从数据生成过程中的潜在变量获取信息的速率。所得不等式将信息流分解为漂移与噪声两部分贡献，从而从信息论视角量化了确定性学习力和SGD诱导的涨落所起的作用。我们在可解析求解的基函数线性回归中验证了该界限，其中由该界限预测的信息预算再现了不同潜在变量在习得参数中被编码的先后次序及特征时间尺度。这些结果将费希尔信息速度极限确立为一个定量框架，可用于诊断扩散动力学何时以及如何……

    arXiv:2609.08219v1 Announce Type: cross  Abstract: Neural networks acquire internal representations through learning. In this work, we formulate stochastic gradient descent (SGD) as a Markovian stochastic process and derive a Fisher-information flow speed limit that bounds the rate at which trainable parameters can acquire information about latent variables in the data-generating process. The resulting inequality decomposes the information flow into drift and noise contributions, thereby quantifying the roles of deterministic learning forces and SGD-induced fluctuations from an information-theoretic perspective. We verify the bound in analytically tractable basis-function linear regression, where the information budget predicted by the bound reproduces the ordering and characteristic time scales with which different latent variables are encoded in the learned parameters. These results establish Fisher-information speed limits as a quantitative framework for diagnosing when and how diff
    
[^268]: 面向最优传输映射估计及其统计最优性的深度重心回归

    Deep Barycentric Regression for Optimal Transport Map Estimation and its Statistical Optimality

    [https://arxiv.org/abs/2609.06598](https://arxiv.org/abs/2609.06598)

    该论文提出了一种简单的两步方法BROT：先计算未正则化的最优传输方案，再用深度神经网络对所得重心目标进行最小二乘回归，从而实现了兼具统计最优保证与实用可扩展性的最优传输映射估计。

    

    最优传输（OT）映射提供了一种用于对齐概率分布的几何变换，并已成为机器学习中的实用工具。然而，现有的OT映射估计器在严格的统计保证与基于稳定训练目标的实用参数化估计之间仍存在差距。理论性估计器能够达到极小化极大（minimax）最优收敛速度，但它们通常是非参数化的，可能需要苛刻的实现设计或高昂的推断成本。实用估计器则是参数化且可扩展的，但其统计保证仍未被充分研究，且其极小极大式的类对抗训练目标可能对优化算法较为敏感。我们提出了BROT（Barycentric Regression for OT，最优传输重心回归），这是一种简单的两步方法：首先计算未正则化的OT方案，然后通过最小二乘回归将深度神经网络（DNN）拟合到由此产生的重心目标上。在标准正则（原文在此处截断）……

    arXiv:2609.06598v1 Announce Type: cross  Abstract: The optimal transport (OT) map provides a geometric transformation for aligning probability distributions and has become a useful tool in machine learning. However, existing estimators of the OT map still exhibit a gap between sharp statistical guarantees and practical parametric estimation based on stable training objectives. Theoretical estimators achieve minimax optimal convergence rates, but they are typically nonparametric and can incur demanding implementation design or inference costs. Practical estimators are parametric and scalable, but their statistical guarantees remain underexplored, and their min-max, adversarial-like training objectives can be sensitive to optimization algorithms. We propose BROT (Barycentric Regression for OT), a simple two-step method that first computes the unregularized OT plan and then fits a deep neural network (DNN) to the induced barycentric targets by least-squares regression. Under standard regu
    
[^269]: LeanStream：一种用于高效端侧大语言模型推理的“推测-精化”流式框架

    LeanStream: A Speculate-and-Refine Streaming Framework for Efficient on-Device LLM Inference

    [https://arxiv.org/abs/2609.03079](https://arxiv.org/abs/2609.03079)

    LeanStream通过利用部分GPU计算结果渐进精化计算、加载和缓存保留优先级，实现了GPU执行与存储I/O的细粒度重叠，突破了端侧LLM推理中准确决策与计算-I/O重叠难以兼得的根本性权衡。

    

    端侧大语言模型（LLM）推理因其在隐私和响应速度方面的优势而具有吸引力，但由于模型权重远超可用DRAM容量，在移动和嵌入式设备上实现仍然极具挑战性。先前的系统利用激活稀疏性并将权重卸载到SSD或闪存，但面临一个根本性的系统权衡：准确的稀疏执行决策需要最新的上下文信息，而高效的计算-I/O重叠则需要提前预测。因此，现有设计要么对执行进行串行化处理，要么导致冗余的权重获取、额外的计算以及巨大的缓存开销。我们提出了LeanStream，一个面向高效端侧LLM推理的流式“推测-精化”框架。LeanStream利用部分GPU计算结果渐进地精化计算、加载和缓存保留的优先级，从而实现GPU执行与存储I/O之间的细粒度重叠。我们在移动和嵌入式平台上均实现了LeanStream。与先前的端侧方案相比……

    arXiv:2609.03079v1 Announce Type: new  Abstract: On-device LLM inference is attractive for privacy and responsiveness, but remains challenging on mobile and embedded devices because model weights far exceed available DRAM. Prior systems exploit activation sparsity and offload weights to SSD or flash storage, but face a fundamental systems trade-off: accurate sparse execution decisions require the latest context, whereas efficient computation-I/O overlap requires early prediction. As a result, existing designs either serialize execution or incur redundant weight fetches, extra computation, and large cache overheads. We present LeanStream, a streaming speculate-and-refine framework for efficient on-device LLM inference. LeanStream progressively refines computation, loading, and cache-retention priorities using partial GPU results, enabling fine-grained overlap between GPU execution and storage I/O. We implement LeanStream on both mobile and embedded platforms. Compared with prior on-devi
    
[^270]: 几何约束的科尔莫戈罗夫-阿诺德网络：通过巴拿赫对偶学习边几何

    Geometry-Constrained Kolmogorov-Arnold Networks: Learning Edge Geometry via Banach Duality

    [https://arxiv.org/abs/2608.25807](https://arxiv.org/abs/2608.25807)

    本文提出了一种通过每条边学习标量指数 $p$ 来动态调整激活函数几何结构的KAN变体，在符号回归任务中优于固定基函数方法。

    

    科尔莫戈罗夫-阿诺德网络（KANs）用可学习的单变量边函数替代深度架构中的固定激活函数，使得边参数化的选择成为核心问题。现有变体依赖固定基函数，如样条、多项式或傅里叶特征，这些基函数在数据观测前就施加了函数空间的几何结构。我们引入了几何约束的KANs，这是一类基于巴拿赫对偶映射导出的边激活函数家族，其中几何本身通过每条边上的标量指数 $p > 1$ 进行学习。该指数控制定性响应：亚欧几里得值产生尖锐的阈值状行为，类似于 $\ell_1$（LASSO）几何；$p = 2$ 恢复线性区域；更大的值在原点附近产生更平坦的响应。在50个符号回归目标上（40个来自AI Feynman基准，加上10个合成压力测试），几何约束的KANs在中等性能上匹配或超越了所有固定基线的基准。

    arXiv:2608.25807v1 Announce Type: new  Abstract: Kolmogorov-Arnold Networks (KANs) replace fixed activations in deep architectures with learnable univariate edge functions, making the choice of edge parametrisation central. Existing variants rely on fixed bases such as splines, polynomials, or Fourier features, which impose a function-space geometry before data are observed. We introduce geometry-constrained KANs, a family of edge activations derived from Banach duality maps in which the geometry itself is learned through a scalar exponent $p > 1$ per edge. This exponent controls the qualitative response: sub-Euclidean values produce sharp, threshold-like behaviour reminiscent of the $\ell_1$ (LASSO) geometry, $p = 2$ recovers the linear regime, and larger values produce flatter responses near the origin. Across 50 symbolic-regression targets ($40$ from the AI Feynman benchmark plus $10$ synthetic stress tests), geometry-constrained KANs match or beat every fixed-basis baseline on medi
    
[^271]: 信任大众：KV缓存驱逐中的强制权重

    Trust the Mass: Forced Weights in KV-Cache Eviction

    [https://arxiv.org/abs/2608.25230](https://arxiv.org/abs/2608.25230)

    本文发现KV缓存驱逐中保留最大权重已接近最优，已发布方法间的差异主要源于存储方式而非选择策略，并揭示了评估中的内存与性能权衡。

    

    arXiv:2608.25230v1 公告类型：交叉 摘要：每个部署的稀疏注意力或KV缓存驱逐规则都会保留一部分键，丢弃其余部分，并对保留集上的注意力权重进行重新归一化。在来自五个模型的168,192个注意力行上，枚举该约束下的精确最优子集表明，保留最大权重已经接近最优，因为最优子集仅将剩余差距中位数缩小了2%到5%。如果选择带来的改进如此之小，那么已发布的驱逐方法之间的差异必然来自其他方面，因此我们测量了每种方法持有的字节数。在共享评估流程中，最强的查询无关方法持有完整缓存，因为它们的按头选择存储为掩码，只有不规则按头存储才能释放该内存。在固定选择上强制执行名义预算会损失14到62个基准点。我们将一个87.6点的检索差距追溯到在问题可见时计算的排名。

    arXiv:2608.25230v1 Announce Type: cross  Abstract: Every deployed sparse-attention or KV-cache-eviction rule keeps a subset of the keys, discards the rest, and renormalizes the attention weights over the kept set. Enumerating the exact best subset under that constraint on $168{,}192$ attention rows from five models shows that keeping the largest weights is already near-optimal, since the best subset closes only a median $2$ to $5\%$ of the remaining gap to full attention. If selection closes this little, published margins between eviction methods must come from elsewhere, so we measure the bytes each method holds. In the shared evaluation pipeline, the strongest query-agnostic methods hold the full cache because their per-head selections are stored as masks, and only ragged per-head storage frees that memory. Enforcing a nominal budget on one fixed selection costs $14$ to $62$ benchmark points. We trace an $87.6$-point retrieval margin to rankings computed while the question is visible
    
[^272]: 使用混合贝叶斯-联结主义模型实现更准确的行为预测

    More accurate behavioral predictions with hybrid Bayesian-connectionist models

    [https://arxiv.org/abs/2608.22154](https://arxiv.org/abs/2608.22154)

    提出了结合行为微调的贝叶斯蒸馏（BBT）混合建模范式：先让神经网络通过合成数据模仿贝叶斯模型，再在人类行为数据上微调，在人类概念学习的行为预测中超越了传统方法。

    

    研究者们常常需要在行为学的贝叶斯模型与神经网络模型之间做出选择，这两种范式具有互补的优势与不足。理想的建模范式应便于测试多种类型的表征与归纳偏置——贝叶斯模型在这方面易于实现，而神经网络则不然；同时，理想的范式还应避免过度简化——神经网络在这方面易于实现，而贝叶斯模型则不然。本文提出了结合行为微调的贝叶斯蒸馏作为兼顾两大传统优点的方法。BBT 提供了一个简单的建模流程：首先，通过合成数据训练神经网络来模仿贝叶斯模型；其次，再在人类行为数据上对网络进行微调，以捕捉额外的结构与细微差异。在四项人类概念学习的案例研究中，我们发现 BBT 在预测人类行为方面优于传统方法，同时还揭示了心理层面的见解。

    arXiv:2608.22154v2 Announce Type: replace  Abstract: Researchers must often choose between Bayesian or neural network models of behavior, two paradigms with complementary strengths and weaknesses. An ideal paradigm would facilitate testing many kinds of representations and inductive biases; Bayesian models make this easy, while neural networks do not. Similarly, an ideal paradigm would avoid over-simplifications; neural networks make this easy, while Bayesian models do not. Here, we introduce Bayesian distillation with Behavioral Tuning (BBT) as an approach to getting the best of both traditions. BBT offers a simple recipe for model building: first, a neural network is trained to mimic a Bayesian model through synthetic data, and second, the network is fine-tuned on human behavior to capture additional structure and nuance. Across four case studies in human concept learning, we find that BBT outperforms traditional approaches at predicting human behavior while also revealing psychologi
    
[^273]: FAR-DPO：面向环肽设计的可行性感知与鲁棒直接偏好优化

    FAR-DPO: Feasibility-Aware and Robust Direct Preference Optimization for Cyclic Peptide Design

    [https://arxiv.org/abs/2608.19808](https://arxiv.org/abs/2608.19808)

    FAR-DPO通过结合可行性感知偏好构建与难度感知的鲁棒优化，提出了一种架构无关的框架，显著提高了环肽设计在挑战性靶标上的可行性和多目标控制能力。

    

    arXiv:2608.19808v1 公告类型：新 摘要：环肽因其高结合亲和力和结构稳定性，正成为药物发现中前景广阔的分子支架。然而，将生成模型从线性肽设计扩展到环肽设计仍具挑战性，因为环化通过耦合的几何和生物物理约束急剧缩小了可行设计空间。此外，有限训练数据导致现有方法主要依赖零样本生成或事后过滤，导致可行设计产出率低，且对多目标权衡的控制有限。为解决这些限制，我们提出FAR-DPO（可行性感知与鲁棒直接偏好优化），一种架构无关的框架，引导生成模型朝向结构上和生物物理上可行的环肽设计，尤其针对具有挑战性的靶标。FAR-DPO整合了可行性感知的偏好构建与难度感知的群体鲁棒优化。

    arXiv:2608.19808v1 Announce Type: new  Abstract: Cyclic peptides are emerging as promising molecular scaffolds in drug discovery due to their high binding affinity and structural stability. However, extending generative models from linear to cyclic peptide design remains challenging, as cyclization sharply restricts the feasible design space through coupled geometric and biophysical constraints. Moreover, limited training data has led existing approaches to rely largely on zero-shot generation or post hoc filtering, resulting in low yields of feasible designs and limited control over multi-objective trade-offs. To address these limitations, we propose FAR-DPO (Feasibility-Aware and Robust Direct Preference Optimization), an architecture-agnostic framework that steers generative models toward structurally and biophysically feasible cyclic peptide designs, particularly for challenging targets. FAR-DPO integrates feasibility-aware preference construction with difficulty-aware group-robust
    
[^274]: 一种用于鲁棒时间序列水印的局部令牌化生成模型

    A Locally Tokenized Generative Model for Robust Time-Series Watermarking

    [https://arxiv.org/abs/2608.19727](https://arxiv.org/abs/2608.19727)

    本文提出了一种局部令牌化生成模型L-VQVAE和配套水印方法LVQMark，通过确保每个恢复单元仅依赖有界时间邻域，解决了时间序列水印在后编辑攻击下的检测可靠性问题。

    

    arXiv:2608.19727v1 公告类型：交叉 摘要：水印是生成模型中溯源的核心工具，但其在多变量时间序列中的应用仍因后编辑攻击下的可靠性失败而受阻。我们表明，现有依赖全局耦合重编码的检测器存在零分布的双向漂移问题：后编辑攻击可以将非水印样本的z分数向任一方向偏移，从而使干净校准的阈值失效。我们认为这种不稳定性是重编码的属性，而可靠的检测要求每个恢复单元仅依赖于有界的时间邻域。基于这一原则，我们提出了L-VQVAE，一种生成模型，其中每个离散令牌由短连续窗口生成，以及LVQMark，一种在该令牌空间上的水印方法，结合了logit偏差注入与鲁棒重编码以用于攻击时检测。在四个基准上的实验跨越了多种场景，验证了其有效性。

    arXiv:2608.19727v1 Announce Type: cross  Abstract: Watermarking is a central tool for provenance in generative models, yet its application to multivariate time series remains hindered by reliability failures under post-editing attacks. We show that existing detectors, which rely on globally coupled re-encoding, suffer from bidirectional drift of the null distribution: post-editing attacks can shift the z-score of non-watermarked samples in either direction, invalidating clean-calibrated thresholds. We argue that this instability is a property of the re-encoding, and that reliable detection requires each recovered unit to depend only on a bounded temporal neighborhood. Guided by this principle, we propose L-VQVAE, a generative model in which each discrete token is produced from a short contiguous window, and LVQMark, a watermarking method over this token space that combines logit-bias injection with robust re-encoding for attack-time detection. Experiments on four benchmarks spanning fi
    
[^275]: 连续对抗平均流迁移

    Continuous Adversarial MeanFlow Transfer

    [https://arxiv.org/abs/2608.19540](https://arxiv.org/abs/2608.19540)

    该论文提出MeanFlow-Transfer方法，通过统一速度表示和对抗训练，同时解决跨领域适应和生成器加速问题，使异构预训练模型能高效迁移到新领域。

    

    摘要：在有限数据下，针对新领域训练快速生成器仍面临两个挑战。首先，将预训练的扩散或流模型适应到新领域时，其昂贵的多步采样问题未得到解决，且现有加速方法与源参数化（如ε、x、v或u）绑定，导致异构预训练模型缺乏统一的加速目标。其次，虽然对抗性精炼已被证明对少步质量有效，但其仅针对瞬时速度流设计，不适用于MeanFlow（MF）模型预测的有限区间平均速度。我们解决了这两个问题。我们提出MeanFlow-Transfer，该方法将异构源输出映射到共享速度表示，利用源权重初始化MF生成器，并在目标域上优化MF目标。这统一了适应和加速过程，在广泛的模型范围内实现单一训练循环。

    arXiv:2608.19540v1 Announce Type: new  Abstract: Training fast generators on new domains with limited data remains challenging for two reasons. First, adapting a pretrained diffusion or flow model to a new domain leaves its costly multi-step sampling unaddressed, and existing acceleration methods are tied to the source parameterization--$\epsilon$, $x$, $v$, or $u$--leaving heterogeneous pretrained models with no common acceleration target. Second, while adversarial refinement is proven effective for few-step quality, it is formulated only for instantaneous-velocity flows, not for the finite-interval average velocities that MeanFlow (MF) models predict. We address both problems. We propose MeanFlow-Transfer, which maps heterogeneous source outputs into a shared velocity representation, uses it to initialize an MF generator from the source weights, and optimizes an MF objective on the target domain. This unifies adaptation and acceleration in a single training loop across a broad range 
    
[^276]: 流式$r$-PCA的推断与不确定性量化

    Inference and Uncertainty Quantification for Streaming $r$-PCA

    [https://arxiv.org/abs/2608.18374](https://arxiv.org/abs/2608.18374)

    本文通过Oja算法解决了流式PCA的尖锐收敛速率和分布推断问题，首次在一般秩下实现无余项的收敛分析，并建立匹配的极小极大下界。

    

    arXiv:2608.18374v1 公告类型：交叉 摘要：我们通过Oja算法解决了流式PCA中的两个开放问题：在次高斯数据下一般秩的尖锐算子范数收敛，以及由此产生的子空间估计量的分布推断。现有的收敛性分析，即使在秩为一的情况下，要么假设数据有界，要么留下不消失的余项，从而阻碍了对多项式衰减尾部谱的适应性，而现有的分布结果仅限于秩为一的情况。我们的收敛理论去除了这些余项，并得到了一个尖锐的速率。在密集尾部尖峰协方差体制下，该速率在对数因子内达到极小极大速率。更一般地，我们在温和的非退化条件下，证明了在密集尾部和稀疏尾部体制下，匹配的下界在对数因子内成立。该分析导出了Oja迭代的线性化，进而实现了对基因的高维高斯近似。

    arXiv:2608.18374v1 Announce Type: cross  Abstract: We address two open questions in streaming PCA via Oja's algorithm: sharp operator-norm convergence for general rank under sub-Gaussian data, and distributional inference for the resulting subspace estimator. Existing convergence analyses, even in the rank-one case, either assume bounded data or leave non-vanishing remainder terms that prevent adaptation to a polynomially vanishing tail spectrum, while existing distributional results are confined to the rank-one case. Our convergence theory removes these remainder terms and yields a sharp rate. In the dense-tail spiked covariance regime, this rate matches the minimax rate up to logarithmic factors. More generally, we prove a matching lower bound, up to logarithmic factors, across both dense-tail and sparse-tail regimes under a mild nondegeneracy condition. The analysis yields a linearization of Oja's iterates, which in turn enables a high-dimensional Gaussian approximation for the gene
    
[^277]: GigaBrain-WBC-0.5：一种用于与环境交互的鲁棒全身控制的行为世界模型

    GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction

    [https://arxiv.org/abs/2608.18234](https://arxiv.org/abs/2608.18234)

    本文提出了首个行为世界模型GigaBrain-WBC-0.5，通过因果Transformer联合预测动作、状态和潜在行为命令，使机器人能够建模环境交互，实现鲁棒的全身控制。

    

    arXiv:2608.18234v1 公告类型：交叉 摘要：全身运动跟踪策略将人形机器人转化为一个鲁棒的控制接口：遥操作员——或上游模型——仅提供粗略的运动意图，而低级策略保持机器人平衡和物理可行性。现有的跟踪器仅在平坦地面上提供此接口：在空场景中训练，它们从未学习地形和物体接触如何重塑其动力学，并且它们试图通过不断扩充参考运动语料库来教会策略在任何命令下保持平衡，这在一旦可行行为变得依赖环境时就失效了。我们提出了GigaBrain-WBC-0.5，这是首个用于人形机器人全身控制的行为世界模型（BWM）。与纯粹的反应式跟踪器不同，我们训练了一个因果Transformer来联合预测其下一个动作、下一个状态以及下一个潜在行为命令的分布，因此，行动的网络也建模了环境如何塑造行为。

    arXiv:2608.18234v1 Announce Type: cross  Abstract: Whole-body motion tracking policies turn a humanoid into a robust control interface: the teleoperator---or an upstream model---only supplies a coarse movement intent, while the low-level policy keeps the robot balanced and physically feasible. Existing trackers deliver this interface only on flat ground: trained in empty scenes, they never learn how contact with terrain and objects reshapes their dynamics, and they attempt to teach the policy to balance under any command by continually enlarging the reference-motion corpus, which stops working once feasible behaviors become environment-dependent. We present GigaBrain-WBC-0.5, the first Behavior World Model (BWM) for humanoid whole-body control. Rather than a purely reactive tracker, we train a causal Transformer to jointly predict its next action, next state, and the distribution over its next latent behavior command, so the network that acts also models how the environment shapes what
    
[^278]: 设计扩散模型的强化学习：一个统一的路径空间视角

    Designing Reinforcement Learning for Diffusion Models: A Unified Path-Space View

    [https://arxiv.org/abs/2608.14430](https://arxiv.org/abs/2608.14430)

    本文通过统一路径空间原则，揭示了扩散模型RL中反向轨迹与前向匹配方法的共同起源，并推导出方差缩减的等效形式，弥合了方法间的经验差距。

    

    强化学习（RL）后训练提供了一种直接的方法，使扩散模型与人类偏好和任务特定奖励对齐。然而，当前用于扩散模型的RL算法仍然碎片化：反向轨迹方法依赖于离散化的似然比，而前向匹配方法则对奖励标注的噪声版本回放样本进行训练。本文表明，这些看似不同的损失函数源于一个单一的路径空间原则。从正则化扩散RL目标出发，我们使用采样SDE之间的重要性采样，获得轨迹空间上的显式策略梯度估计器。该估计器包含Flow-GRPO型更新背后的随机伊藤积分；我们推导出一个等效的方差缩减值梯度形式，它恢复了AWM和DiffusionNFT的前向匹配结构。这确定了这些方法族之间的经验差距作为方差缩减。

    arXiv:2608.14430v1 Announce Type: new  Abstract: Reinforcement learning (RL) post-training provides a direct way to align diffusion models with human preferences and task-specific rewards. However, current RL algorithms for diffusion models remain fragmented: reverse-trajectory methods rely on discretized likelihood ratios, whereas forward-matching methods train on reward-labeled noising versions of the rollout samples. This paper shows that these seemingly different losses arise from a single path-space principle. Starting from the regularized diffusion-RL objective, we use importance sampling between sampling SDEs to obtain an explicit policy-gradient estimator on trajectory space. The estimator contains the stochastic It\^o integral underlying Flow-GRPO-type updates; we derive an equivalent variance-reduced value-gradient form that recovers the forward-matching structure of AWM and DiffusionNFT. This identifies the empirical gap between these method families as a variance-reduction 
    
[^279]: CytoBERT：一种用于细胞计数数据的基座模型

    CytoBERT: A Foundation Model for Cytometry Data

    [https://arxiv.org/abs/2608.14414](https://arxiv.org/abs/2608.14414)

    CytoBERT是一个针对异质细胞计数数据的开源基座模型，通过自监督预训练解决了跨研究迁移难题。

    

    细胞计数技术测量单细胞的复杂特征（如免疫细胞的数量和蛋白质表达），广泛应用于免疫学研究和临床环境。然而，由于实验方案和测量特征选择的不同，细胞计数数据高度异质且缺乏标准化。尽管机器学习方法有望深入理解细胞生物学，但这些挑战使其难以在不同研究中应用和迁移。基座模型的最新进展可以缓解这些问题，但该领域相应的研究方法仍然稀缺。为此，我们提供了CytoBERT，一个公开可用、开源、开放权重的基座模型，适用于具有可变标记物组合的单细胞细胞计数数据。CytoBERT以自监督方式在大型细胞计数语料库（包含15个人类数据集，具有异质标记物组合和超过5000万个细胞）上进行预训练。

    arXiv:2608.14414v1 Announce Type: new  Abstract: Cytometry measures the complex characteristics of single cells (e.g., counts and protein expression of immune cells) and is widely used across immunological research and clinical settings. However, cytometry data is highly heterogeneous and unstandardized due to experimental protocols and the choice of measured features. While machine learning methods hold the potential to gain deeper insights into cell biology, these challenges make them difficult to apply and transfer across studies. Recent advances in foundation models can alleviate these issues, but corresponding approaches are still scarce in this field. To address this, we provide CytoBERT, a publicly available, open-source, open-weight foundation model for single-cell cytometry data with variable marker panels. CytoBERT is pretrained in a self-supervised manner on a large-scale cytometry corpus (15 human datasets with heterogeneous marker panels and more than 50 million cells) cur
    
[^280]: 基于变分学习的RLVR参数空间探索

    Parameter Exploration for RLVR via Variational Learning

    [https://arxiv.org/abs/2608.09805](https://arxiv.org/abs/2608.09805)

    本文提出了一族名为扰动参数策略优化（3PO）的方法，通过变分学习从后验分布中采样多样化策略，实现参数空间探索，弥补了温度缩放等动作空间探索方法的局限，从而提升LLM强化学习的训练稳定性与下游性能。

    

    探索长期以来一直是强化学习研究的重点。最近，越来越多的证据表明，探索也是LLM强化学习方案中的一个重要组成部分，能够显著影响下游性能。许多现有方法在动作空间中控制探索，例如使用温度缩放。然而，这些方法无法对token进行重新排序，只能影响输出分布的方差，这限制了探索，可能导致训练发散或停滞。本文研究了参数空间探索，其中rollout通过从后验分布中采样不同的策略来生成，每个策略可能探索不同的rollout路径。采样较少或更多样化的策略则成为控制探索的补充手段。我们提出了一族称为扰动参数策略优化的方法，它们使用不同的采样策略和不同的ro

    arXiv:2608.09805v2 Announce Type: replace-cross  Abstract: Exploration has been a focus of reinforcement learning research for a long time. Recently, there has been growing evidence that it is also an important ingredient in LLM reinforcement learning recipes that can significantly impact downstream performance. Many existing methods control exploration in the action-space, for example, using temperature scaling. However, these methods cannot reorder tokens but only influence the variance in the output distribution. This limits exploration and can lead to divergence or stalled training. Here, we investigate parameter-space exploration, where rollouts are generated by sampling different policies from a posterior that may each explore different rollouts. Sampling less or more diverse policies is then a complementary control lever over exploration. We introduce a family of methods called Perturbed Parameter Policy Optimization (3PO) which use different sampling strategies and different ro
    
[^281]: 面向群体规模可扩展的多智能体世界建模

    Population-Scalable Multi-Agent World Modeling

    [https://arxiv.org/abs/2608.08600](https://arxiv.org/abs/2608.08600)

    本文提出Khora，一种可扩展的多智能体世界模型，通过解耦世界状态演化与视觉渲染，实现推理时无需重新训练即可扩展到任意数量的智能体。

    

    arXiv:2608.08600v2 公告类型：替换交叉 摘要：世界模型近年来在视觉预测和交互式生成方面取得了显著进展，但将其扩展到多智能体环境引入了根本性的可扩展性挑战。现有方法通常在训练和推理过程中假设固定数量的智能体，这使模型绑定到预定的智能体群体，并限制了推理时的可扩展性。我们的关键见解是，跨视角一致性应源自一个共享的世界状态，其演化不假设预定义的智能体数量，而智能体特定的观察应通过统一的渲染接口查询该状态来生成。基于这一见解，我们提出了Khora，一种可扩展的多智能体世界模型，支持在无需重新训练的情况下推理时扩展到任意数量的智能体。我们的框架将世界状态演化与视觉渲染解耦，并引入了一种群体无关的渲染机制。

    arXiv:2608.08600v2 Announce Type: replace-cross  Abstract: World models have recently achieved impressive progress in visual prediction and interactive generation, but extending them to multi-agent environments introduces a fundamental scalability challenge. Existing methods generally assume a fixed number of agents during training and inference, which ties the model to a pre-determined agent population and limits inference-time scalability. Our key insight is that cross-view consistency should arise from a shared world state whose evolution does not assume a predefined number of agents, while agent-specific observations should be generated by querying this state through a unified rendering interface. Based on this insight, we propose Khora, a scalable multi-agent world model that supports inference-time expansion to arbitrary numbers of agents without retraining. Our framework decouples world-state evolution from visual rendering and introduces a population-agnostic rendering mechanis
    
[^282]: 从孤立算法到合规优先的智能体平台：医院AI系统的多层架构

    From Siloed Algorithms to Compliance-First Agentic Platforms: A Multi-Layered Architecture for Hospital AI Systems

    [https://arxiv.org/abs/2608.06112](https://arxiv.org/abs/2608.06112)

    该研究提出了一种合规优先的医院智能体AI多层架构，通过智能体编排层和以“策略即代码”为核心的合规与策略层，解决医疗AI部署孤立、治理缺失和难以规模化推广的问题。

    

    医院正在快速采用人工智能用于分诊、影像诊断、日程安排等领域，然而大多数部署仍然是孤立的点状解决方案，被封锁在部门壁垒之中，导致重复劳动、隐性风险和企业价值未能实现。尽管医疗AI市场呈爆炸式增长且投资不断加速，据估计70-80%的医疗AI试点项目无法规模化推广，其主要原因在于治理缺口、数据碎片化以及缺乏集成蓝图。本研究提出了一种医院专属、合规优先的智能体AI架构，包含多个可互操作的层级，对现有医院AI平台模型进行了扩展：(i) 智能体编排层，支持跨临床、运营和财务领域的多智能体工作流；(ii) 合规与策略层，以“策略即代码”的方式集中管理HIPAA、GDPR、欧盟AI法案、DISHA法案、印度DPDP法案以及ISO/IEC安全标准（注：原文摘要在此处截断，内容不完整）。

    arXiv:2608.06112v2 Announce Type: replace  Abstract: Hospitals are rapidly adopting artificial intelligence for triage, imaging, scheduling etc., yet most deployments remain isolated point solutions locked inside departmental silos, resulting in duplicated effort, hidden risks, and unrealized enterprise value. Despite explosive growth of AI in healthcare market and accelerating investment, an estimated 70-80% of healthcare AI pilots fail to scale, largely due to governance gaps, fragmented data, and missing integration blueprints. This research proposes a hospital-specific, compliance-first, Agentic AI architecture with multiple interoperable layers, extending existing hospital AI platform models with: (i) an Agent Orchestration Layer for multi-agent workflows across clinical, operational, and financial domains, (ii) a Compliance and Policy Layer that centralizes policy-as-code for HIPAA, GDPR, the EU AI Act, DISHA Act, India's DPDP Act, and ISO/IEC security and safety standards, and (
    
[^283]: 利用噪声波动与连接重缩放对任务训练的循环神经网络进行有效剪枝

    Effective pruning of task-trained recurrent neural networks using noisy fluctuations and connection rescaling

    [https://arxiv.org/abs/2608.05464](https://arxiv.org/abs/2608.05464)

    该研究将noise-prune剪枝规则成功应用于任务训练的循环神经网络，通过基于噪声波动对连接重要性进行采样并重缩放保留连接的强度来保持任务性能，其效果优于基于权重大小和二阶信息的剪枝策略。

    

    剪枝（删除）网络连接对大脑功能至关重要，但尽管其重要性，目前具有良好性能且符合生物学合理性的剪枝规则却很少。我们评估了noise-prune——一种近期提出的用于循环网络的无监督局部剪枝规则，它利用噪声波动来确定连接的重要性。此前noise-prune仅在无特定计算功能的随机网络上进行过经验性测试。我们证明，noise-prune能够在任务训练的循环神经网络中保持任务性能，大幅优于仅依据连接权重大小的策略，并超过了使用二阶信息的非局部策略。与确定性地删除低于阈值的连接不同，noise-prune根据重要性对要保留的连接进行随机采样，并加强被保留的连接以维持平均连接强度。我们进一步证明，这种采样机制……

    arXiv:2608.05464v2 Announce Type: replace-cross  Abstract: Pruning network connections is key to brain function but, despite its importance, there exist few biologically-plausible pruning rules with demonstrated good performance. We evaluate noise-prune, a recently introduced unsupervised local pruning rule for recurrent networks that uses noisy fluctuations to determine the importance of connections. Noise-prune has previously only been empirically tested on random networks without a specific computational function. We show that noise-prune preserves task-performance in task-trained recurrent neural networks, greatly outperforming a strategy that only uses connection magnitude and exceeding a non-local strategy that uses second-order information. Rather than deterministically removing connections that fall below a threshold, noise-prune samples connections to preserve based on their importance and strengthens retained connections to preserve average strength. We show that this samplin
    
[^284]: 变量感知表示而非关系类型化解释了事件图模型在ICU死亡率预测中的性能：时间性与外部验证

    Variable-aware representation, not relation typing, explains the performance of an event-graph model for ICU mortality prediction: temporal and external validation

    [https://arxiv.org/abs/2608.02663](https://arxiv.org/abs/2608.02663)

    该研究通过严格的时间性与外部多中心验证发现，事件图模型在ICU死亡率预测中的性能提升主要来自变量感知表示（变量学习嵌入加逐变量归一化），而非关系类型化的消息传递。

    

    事件图神经网络将重症监护病房（ICU）记录表示为由带时间戳的观测所连接的类型化实体，其性能提升通常被归因于关系特定的消息传递。我们测试了在消除数据泄漏并采用时间性与外部验证的情况下，哪些组件真正起作用。在重症监护医学信息集市IV（MIMIC-IV）数据库的31,142次至少48小时的首个ICU住院中，预测住院48小时后院内死亡率的模型在2008-2016年的住院数据上开发，在留出的2008-2016年住院数据以及2017-2019年和2020-2022年的住院数据上进行测试，并无需重新训练地应用于来自eICU协作研究数据库中205家医院的54,696次住院数据。此前版本中存在的数据泄漏——将ICU总住院时长作为输入——已被移除。变量感知表示（即每个节点变量的学习嵌入加上逐变量归一化）提升了平均单模型受试者工作特征曲线下面积（AUC）……

    arXiv:2608.02663v2 Announce Type: replace-cross  Abstract: Event-graph neural networks represent intensive care unit (ICU) records as typed entities linked by timestamped observations, and their gains are often attributed to relation-specific message passing. We tested which components matter once leakage is removed and validation is temporal and external. In 31,142 first ICU stays of at least 48 hours in the Medical Information Mart for Intensive Care IV (MIMIC-IV), models predicting post-48-hour in-hospital mortality were developed on 2008-2016 stays, tested on held-out 2008-2016 stays and on 2017-2019 and 2020-2022 stays, and applied without retraining to 54,696 stays from 205 hospitals in the eICU Collaborative Research Database. A leak in a previous version, total ICU length of stay used as an input, was removed. A variable-aware representation (a learned embedding of each node's variable plus per-variable normalisation) raised the mean single-model area under the receiver operati
    
[^285]: Twig：面向混合注意力大语言模型的并行且内存高效的树形推测解码

    Twig: Parallel and Memory-Efficient Tree Speculative Decoding for Hybrid-Attention LLMs

    [https://arxiv.org/abs/2608.01651](https://arxiv.org/abs/2608.01651)

    Twig通过对线性注意力递归进行树结构化因子分解并配合并行GPU内核，实现了混合注意力大语言模型的高效树形推测解码，将树验证加速3.4至7.7倍，同时将瞬态状态内存降低82至99倍。

    

    新兴的大语言模型越来越多地将全注意力与循环线性注意力交织结合，以降低长上下文推理服务的成本。树形推测解码可进一步加速这类混合模型，但循环线性注意力层携带有依赖于父节点的状态。因此，现有引擎只能顺序地验证提议树，并为每个节点快照完整状态，导致延迟和内存开销随树规模与批大小增长而不断攀升。我们提出了Twig，一种面向混合大语言模型树形推测解码的内核与运行时协同设计。Twig推导出线性注意力递归的树结构化因子分解，并通过一个资源高效的GPU内核加以实现，可并行验证所有节点，将线性注意力树验证速度提升3.4至7.7倍。它以无损方式将推测状态更新表示为token级因子，仅重构被接受的状态，从而将瞬态状态内存降低82至99倍，释放了GPU显存……（摘要原文在此处截断）

    arXiv:2608.01651v2 Announce Type: replace-cross  Abstract: Emerging LLMs increasingly interleave full attention with recurrent linear attention to reduce long-context serving costs. Tree speculative decoding could further accelerate these hybrid models, but recurrent linear-attention layers carry parent-dependent states. Existing engines therefore verify proposal trees sequentially and snapshot the full state per node, causing latency and memory to grow with tree and batch sizes. We present Twig, a kernel--runtime co-design for tree speculative decoding of hybrid LLMs. Twig derives a tree-structured factorization of the linear-attention recurrence and realizes it with a resource-efficient GPU kernel that verifies all nodes in parallel, accelerating linear-attention tree verification by 3.4--7.7$\times$. It losslessly represents speculative state updates as token-level factors and reconstructs only the accepted state, reducing transient state memory by 82--99$\times$ and freeing GPU mem
    
[^286]: GPTQ-2D：立方时间复杂度的双侧自适应舍入

    GPTQ-2D: Cubic-Time Two-Sided Adaptive Rounding

    [https://arxiv.org/abs/2607.27042](https://arxiv.org/abs/2607.27042)

    GPTQ-2D提出了一种可证明以立方计算量实现与YAQA相同的双侧自适应舍入效果的高效大模型量化算法，将Llama-3.1-70B在单块GPU上的量化时间从8.6天大幅缩短至1.6小时。

    

    诸如GPTQ之类的自适应舍入方法逐一对大语言模型（LLM）的权重进行量化，并将每次舍入产生的误差传播到尚未舍入的权重上。YAQA在Kronecker分解的Hessian矩阵下通过双侧自适应舍入降低了与原始模型的KL散度，但其误差反馈的计算量随方阵层的规模呈四次方增长。我们提出了GPTQ-2D，该算法可被证明地以立方计算量计算出与YAQA相同的舍入结果，同时仍能并行地对每条反对角线上的元素进行舍入。其GPU实现采用两级分块技术来提高硬件利用率。我们还将该算法扩展至具有任意数量Kronecker因子的Hessian矩阵。实验表明，GPTQ-2D的准确度与YAQA相当，并将Llama-3.1-70B-Instruct在单块NVIDIA RTX A6000 GPU上的量化时间从8.6天缩短至1.6小时。源代码可在 https://github.com/IST-DASLab/GPTQ-2D 获取。

    arXiv:2607.27042v2 Announce Type: replace-cross  Abstract: Adaptive rounding methods such as GPTQ quantize the weights of a large language model (LLM) one at a time, propagating each rounding error to the weights not yet rounded. YAQA lowers the KL divergence from the original model with two-sided adaptive rounding under a Kronecker-factored Hessian, but its error feedback takes quartic work in the size of a square layer. We present GPTQ-2D, which provably computes the same rounding with cubic work, while still rounding the entries of each anti-diagonal in parallel. Its GPU implementation uses two levels of blocking to improve hardware utilization. We also extend the algorithm to Hessians with any number of Kronecker factors. In our experiments, GPTQ-2D is as accurate as YAQA and cuts the time to quantize Llama-3.1-70B-Instruct on a single NVIDIA RTX A6000 GPU from 8.6 days to 1.6 hours. Source code is available at https://github.com/IST-DASLab/GPTQ-2D.
    
[^287]: 面向网络化系统的不变量发现

    Invariant Discovery for Networked Systems

    [https://arxiv.org/abs/2607.22944](https://arxiv.org/abs/2607.22944)

    本文提出 Autogram，通过让 LLM 生成紧凑的候选不变量文法并在其内进行确定性统计搜索、再迭代扩展文法直至恢复已知可信不变量，实现了无需人工文法且能容忍真实噪声的网络系统不变量自动发现。

    

    不变量是指网络中各测量信号之间预期成立的关系，它是从验证、流量生成、遥测数据填补到输入校验等众多应用的基础。然而，手工编写这些不变量需要同时具备形式逻辑与网络领域的稀缺专业知识。在网络规模下，对所有遥测变量的可能组合进行统计检验是难以实现的；而现有的自动化工具仍然需要人工编写可接受不变量的文法，并且只能学习精确的、“硬性”的规则，这类规则在真实世界的噪声下会失效。我们的核心思路是利用大语言模型（LLM）检查变量元数据、对计数器数据类型进行分类，并预测候选的句法关系以形成一个紧凑的文法，然后在该文法内进行确定性的统计搜索。如果初始文法无法在留出的遥测数据上恢复出可信的已知不变量，Autogram 会指示 LLM 迭代地扩展文法，直至恢复成功。

    arXiv:2607.22944v2 Announce Type: replace-cross  Abstract: Invariants, the relations expected to hold among measured signals of a network, underpin applications from verification to traffic generation, telemetry imputation, and input validation. Yet, writing them by hand demands rare expertise in both formal logic and networking. Testing all possible combinations of telemetry variables statistically is intractable at network scale, while existing automatic tools still require a human-authored grammar of admissible invariants and learn only exact, "hard" rules that break under real-world noise. Our key idea is to use an LLM to inspect variable metadata, classify counter data types, and predict candidate syntactic relationships that form a compact grammar, followed by a deterministic statistical search within that grammar. If the initial grammar fails to recover trusted known invariants on held-out telemetry, Autogram instructs the LLM to iteratively widen the grammar until recovery succ
    
[^288]: 标记而非抹除：面向大语言模型两用知识的Token接种方法

    Mark, Don't Erase: Token Inoculation for Dual-Use Knowledge in LLMs

    [https://arxiv.org/abs/2607.18639](https://arxiv.org/abs/2607.18639)

    提出“Token接种”方法，在继续预训练时用特殊标记标注危险句子并训练模型条件化地预测危险答案，测试时因标记缺失而抑制危险输出，从而在保持93%良性性能的同时将危险准确率控制在约18%。

    

    在两用领域中，危险知识与围绕它的良性专业知识共享词汇、概念和因果结构。这种重叠使得选择性拒绝变得困难，而诸如去学习和拒绝训练等现有方法会将其抑制过度泛化到良性知识一侧。我们主张采用条件化而非抑制。在本工作中，我们提出了Token接种方法，在继续预训练期间用一个特殊标记（即接种标记）来标记危险句子。随后我们微调模型，使其拒绝危险问题，同时在带有接种标记的相同问题上预测危险答案。在测试时，接种标记并不存在，因此模型不会表达它学会预测的危险答案。在1.5B到14B参数范围以及三种架构家族中，该方法将危险准确率保持在约18%，同时保留了指令基线93%的MMLU性能，实现了最佳的安全-效用平衡。

    arXiv:2607.18639v2 Announce Type: replace-cross  Abstract: In a dual-use field, hazardous knowledge and the benign expertise around it share vocabulary, concepts, and causal structure. That overlap makes selective refusal hard, and existing methods such as unlearning and refusal training overgeneralize their suppression into the benign side. We argue for conditioning, not suppression. In this work, we introduce Token Inoculation, which marks hazardous sentences with a special token, the inoculation token, during continued pre-training. We then fine-tune the model to refuse hazardous questions and, on the same questions under the inoculation token, to predict the hazardous answers. At test time the inoculation token is absent, so the model does not express the hazardous answers it learned to predict. Across 1.5B to 14B parameters and three architecture families it holds hazardous accuracy near 18% while retaining 93% of the instruct baseline's MMLU performance, the best safety-utility b
    
[^289]: 超越表格数据的TabPFN：多模态嵌入上的校准与精度

    TabPFN beyond Tabular Data: Calibration and Accuracy on Multimodal Embeddings

    [https://arxiv.org/abs/2607.11007](https://arxiv.org/abs/2607.11007)

    该研究发现TabPFN可直接用作冻结多模态编码器（图像、文本、音频）的零梯度分类头，在22,820个评估回合中其校准性能（NLL和ECE）在九种分类头中排名最佳，同时保持甚至超越基线精度。

    

    少样本多模态分类通常在冻结的预训练编码器上附加一个轻量级头部，例如k近邻、逻辑回归或线性SVM。尽管这些头部计算效率高，但可能产生校准不佳的置信度分数。我们探究了TabPFN能否在多模态嵌入上提供可靠的置信度估计而不牺牲预测精度，以及在什么条件下能够做到。我们系统地评估了TabPFN作为冻结图像、文本和音频编码器的零梯度分类头的表现。在涵盖14个数据集、11个编码器和三种模态的22,820个评估回合中，TabPFN在九种分类头中于负对数似然（NLL）和期望校准误差（ECE）两项指标上均取得最佳平均排名。在一个代表性设置下，相对于八个基线的平均水平，它将NLL降低了48–62%，ECE降低了2.1–5.3倍，同时保持或超越了它们的平均精度。

    arXiv:2607.11007v4 Announce Type: replace  Abstract: Few-shot multimodal classification commonly attaches a lightweight head, such as $k$-nearest neighbors, logistic regression, or a linear SVM, to a frozen pretrained encoder. Although computationally efficient, these heads can produce poorly calibrated confidence scores. We ask whether TabPFN can provide reliable confidence estimates on multimodal embeddings without sacrificing predictive accuracy, and under what conditions. We systematically evaluate TabPFN as a zero-gradient head for frozen image, text, and audio encoders. Across 22{,}820 evaluation episodes spanning 14 datasets, 11 encoders, and three modalities, TabPFN achieves the best mean rank among nine classification heads on both negative log-likelihood (NLL) and expected calibration error (ECE). At a representative setting, it reduces NLL by 48--62\% and ECE by 2.1--5.3$\times$ relative to the average of eight baselines while matching or exceeding their average accuracy. Th
    
[^290]: 整合背景知识以实现可扩展的因果发现

    Integrating Background Knowledge for Scalable Causal Discovery

    [https://arxiv.org/abs/2607.10456](https://arxiv.org/abs/2607.10456)

    该论文提出了一个在因果发现过程中（而非仅作为事后后处理）整合专家背景知识的框架，并针对仅恢复部分图结构的大规模可扩展因果发现方法进行了多算法实现与实验验证。

    

    在因果发现的实际应用中，专家背景知识往往是可获得的。这类对真实因果图的约束能够在因果效应的可识别性和学习结构的准确性方面帮助因果发现，同时还能缩小候选因果图的搜索空间。由于在变量数量众多时因果发现的计算开销可能变得非常昂贵，因此在因果发现过程中有效利用背景知识至关重要。然而，目前大多数方法仅在因果发现完成后的后处理步骤中使用背景知识来精炼学习到的图。在这项工作中，我们开发了一个在因果发现过程中利用背景知识的框架，尤其关注仅恢复整个图的一部分子集的可扩展因果发现方法。我们为多种算法实现了该框架，并通过实验证明，利用背景知识（摘要在此处被截断）……

    arXiv:2607.10456v2 Announce Type: replace-cross  Abstract: Expert background knowledge is often available in practical applications of causal discovery. Such constraints on the true causal graph can help causal discovery in terms of identifiability of causal effects and accuracy of the learned structure, but also in reducing the space of candidate causal graphs. As causal discovery can become computationally expensive for large number of variables, it is crucial to utilize background knowledge effectively during the causal discovery process. However, most current methods only use background knowledge in a postprocessing step after causal discovery to refine the learned graph. In this work, we develop a framework for utilizing background knowledge during the causal discovery process, focusing especially on scalable causal discovery methods that recover only a subset of the whole graph. We implement our framework for multiple algorithms and empirically show that utilizing background know
    
[^291]: 人工神经网络中通过表征学习实现涌现泛化

    Emergent Generalization by Representation Learning in Artificial Neural Networks

    [https://arxiv.org/abs/2607.10430](https://arxiv.org/abs/2607.10430)

    本文证明在循环神经网络中施加显式信息瓶颈以学习低维表征，是时间序列预测任务中实现旋转泛化与分布外泛化的必要条件，并通过因果涌现的信息论度量揭示了表征在记忆到泛化转变过程中的非单调动力学轨迹。

    

    降维已被证明是识别神经流形的有力工具，神经流形是高维神经活动背后的低维结构。这些低维表征提高了群体级编码的可解释性。然而，这种低维表征是否具有生物学意义并能为学习系统带来功能优势，还是仅仅反映神经元层面的活动，在神经科学领域仍存在争议。我们证明，在时间序列预测任务中，显式的信息瓶颈迫使循环神经网络学习低维表征，这对实现旋转泛化和分布外泛化是必要的。利用因果涌现的信息论度量，我们刻画了这种表征在从记忆到泛化转变过程中的动力学特征，发现了一条非单调的轨迹：表征维度先下降、达到最小值，随后……（原文摘要在此处截断）

    arXiv:2607.10430v2 Announce Type: replace-cross  Abstract: Dimensionality reduction has proven powerful for identifying neural manifolds, which are low-dimensional structures underlying high-dimensional neural activity. These low-dimensional representations have improved the interpretability of population-level coding. Yet whether such low-dimensional representations are biologically relevant and confer functional advantages in learning systems, or merely reflect neuron-level activity, remains contested in neuroscience. We show that an explicit information bottleneck forcing a recurrent neural network to learn a low-dimensional representation is necessary for rotational and out-of-distribution generalisation in a time-series prediction task. Using information-theoretic measures of causal emergence, we characterise the dynamics of this representation across the memorisation-to-generalisation transition, finding a non-monotonic trajectory which shows an initial decrease, a minimum, and a
    
[^292]: 强化学习也会遗忘！迈向持续策略优化

    RL Forgets! Towards Continual Policy Optimization

    [https://arxiv.org/abs/2607.04364](https://arxiv.org/abs/2607.04364)

    该论文提出MRCL多模态推理持续学习基准，证明标准强化学习在持续后训练中仍会发生灾难性遗忘，其根源在于KL正则化仅基于当前任务数据评估，而遗忘源自模型在旧任务分布上的行为漂移。

    

    持续后训练正在成为使视觉-语言模型适应不断演变的任务的核心范式。近期研究报道称，强化学习比监督微调更不容易发生遗忘，这促使人们形成了强化学习天生具有抗遗忘能力的观点。然而，这一观点尚未得到充分验证，因为现有证据大多来自过时或同质化的基准。我们通过引入MRCL——一个由近期且多样化的多模态推理任务构建的多模态推理持续学习基准——来重新审视这一假设。在MRCL上的实验表明，标准的强化学习在持续后训练过程中仍然会遭受灾难性遗忘。我们将这一失败归因于目标不匹配：常见策略优化方法中使用的KL正则化是在当前任务数据上进行评估的，而遗忘则是由模型在先前任务分布上的行为漂移所引起的。为了解决这一问题

    arXiv:2607.04364v4 Announce Type: replace  Abstract: Continual post-training is becoming a central paradigm for adapting vision-language models to evolving tasks. Recent studies report that reinforcement learning is less prone to forgetting than supervised fine-tuning, motivating the view that RL is inherently resistant to forgetting. However, this view remains insufficiently validated, as existing evidence is largely drawn from outdated or homogeneous benchmarks. We revisit this assumption by introducing MRCL, a Multimodal Reasoning Continual Learning benchmark built from recent and diverse multimodal reasoning tasks. Experiments on MRCL show that standard reinforcement learning still suffers from catastrophic forgetting during continual post-training. We trace the failure to an objective mismatch. The KL regularization used in common policy optimization methods is evaluated on current-task data, whereas forgetting is caused by behavioral drift on prior-task distributions. To address 
    
[^293]: HyperVAttention：基于时空聚类的高效稀疏注意力方法用于视频扩散

    HyperVAttention: Efficient Sparse Attention with Spatio-Temporal Clustering for Video Diffusion

    [https://arxiv.org/abs/2607.03012](https://arxiv.org/abs/2607.03012)

    提出HyperVAttention——一种无需训练的稀疏注意力框架，通过3D局部窗口聚类和自定义Triton内核，同时降低视频扩散Transformer的聚类开销并提升CTA利用率，实现高效的长视频生成加速。

    

    视频扩散Transformer（VDiTs）在高保真视频生成方面展现出了卓越的能力。然而，其生成长时长视频的能力从根本上受到自注意力机制二次方复杂度的制约。近期基于聚类的稀疏注意力方法通过将语义相似的token分组来改善质量与速度的权衡，但其实际效率仍受两个瓶颈的限制：可观的聚类开销，以及由不规则聚类块导致的低CTA利用率。我们提出了HyperVAttention（HVA），这是一个无需训练的稀疏注意力框架，可同时解决上述两个瓶颈。为降低聚类开销，我们引入了3D局部窗口聚类，利用视频token的时空局部性将质心搜索限制在固定的局部邻域内，并通过自定义Triton内核实现高效执行。我们进一步……（原文摘要在此处截断）

    arXiv:2607.03012v2 Announce Type: replace-cross  Abstract: Video Diffusion Transformers (VDiTs) have demonstrated significant capabilities in high-fidelity video generation. However, their ability to produce long-duration videos is fundamentally constrained by the quadratic complexity of the self-attention mechanism. Recent clustering-based sparse attention methods improve the quality-speed trade-off by grouping semantically similar tokens, but their practical efficiency remains limited by two bottlenecks: substantial clustering overhead and low CTA utilization caused by irregular cluster-induced blocks. We propose HyperVAttention (HVA), a training-free sparse attention framework that addresses both bottlenecks jointly. To reduce clustering overhead, we introduce 3D local-window clustering, which exploits the spatio-temporal locality of video tokens to restrict centroid search to fixed local neighborhoods, and implement it with a custom Triton kernel for efficient execution. We further
    
[^294]: 状态-预测分离假设

    The State-Prediction Separation Hypothesis

    [https://arxiv.org/abs/2607.01218](https://arxiv.org/abs/2607.01218)

    该论文提出“状态-预测分离假设”，设计了一种用独立计算流将“预测下一个token”与“存储状态”两种功能解耦的 Transformer 变体，多规模预训练实验表明其具有更高的数据与计算效率，在下游任务上平均超越标准 Transformer 2-3 个百分点。

    

    Transformer 使用同一条前向计算流来同时完成两个任务：预测下一个 token，以及为未来的 token 预测存储有用的状态。我们提出了状态-预测分离假设：将这两种角色解耦能够带来更好的语言建模性能。我们设计了一种使用独立计算流来分离这两种功能的 Transformer 变体，并在多种规模上开展了预训练实验。实验表明，状态-预测分离始终能带来更好的数据效率和计算效率，降低了验证损失，并在下游任务上平均比标准 Transformer 高出 2-3 个百分点。我们还进行了广泛的实证分析，排除了潜在的混杂因素，并证明了我们的设计在梯度上所体现出的根本性差异。

    arXiv:2607.01218v2 Announce Type: replace-cross  Abstract: Transformers use the same forward computation stream to both predict the next token and store useful state for future token predictions. We formulate the state-prediction separation hypothesis: disentangling the two roles yields better language modeling performance. We design a Transformer variant that uses separate computation streams to separate the two functions, and conduct pretraining experiments across various scales. Our experiments show that state-prediction separation consistently offers better data- and compute-efficiency, improving validation loss and outperforming standard Transformers by 2-3 percentage points on average on downstream tasks. We also conduct extensive empirical analysis that rules out potential confounders and demonstrates the fundamental difference in the gradients our design entails.
    
[^295]: 用于结构化与非结构化体数据快速无网格压缩的显式三维高斯模型

    Explicit 3D Gaussian Models for Fast, Mesh-Free Compression of Structured and Unstructured Volumes

    [https://arxiv.org/abs/2607.01164](https://arxiv.org/abs/2607.01164)

    该论文提出基于显式三维高斯基元的体数据压缩模型，通过对相交高斯进行加权聚合来重建标量场，实现了无需网格存储的结构化与非结构化体数据快速压缩。

    

    近期研究表明，隐式神经表示经训练后可有效压缩结构化和非结构化体数据，能够在降低内存占用的同时支持直接数据查询。然而，现有的用于非结构化体的隐式神经表示不编码几何信息，需要部分网格存储以供后续采样，这限制了可实现的压缩率。与此同时，新视角合成方法已经证明，显式的三维高斯集合可用于精确可视化体数据。在本工作中，我们提出了一种基于三维高斯基元的显式体数据压缩模型。我们将三维高斯集合重新解释为体标量场的显式表示，并采用一种采样策略，通过对相交高斯进行加权聚合来重建空间位置处的标量值。我们为结构化和非结构化体数据开发了优化的CUDA加速管线

    arXiv:2607.01164v2 Announce Type: replace  Abstract: Recent work has shown that implicit neural representations (INRs) can be trained to effectively compress structured and unstructured volume data, allowing for direct data querying with a reduced memory footprint. However, as existing INRs for unstructured volumes do not encode geometry, they require partial mesh storage for later sampling, limiting achievable compression. At the same time, novel view synthesis methods have shown that explicit collections of 3D Gaussians can be used to accurately visualize volume data. In this work, we introduce an explicit model for volume data compression based on 3D Gaussian primitives. We reinterpret collections of 3D Gaussians as an explicit representation of a volume's scalar field and use a sampling strategy that reconstructs scalar values at spatial locations through weighted aggregation of intersecting Gaussians. We develop optimized CUDA-accelerated pipelines for structured and unstructured 
    
[^296]: 面向语言模型智能体系统的集合值路由：一个WildChat基准与成本感知评估

    Set-Valued Routing for Language-Model Agent Systems: A WildChat Benchmark and Cost-Aware Evaluation

    [https://arxiv.org/abs/2606.28925](https://arxiv.org/abs/2606.28925)

    该论文将语言模型智能体系统的路由从单一top-1选择扩展为集合值路由，构建了基于WildChat的3000条提示、12个能力定义智能体的新基准，并通过成本感知评估证明微调文本编码器（89.59 F1）显著优于线性SVM和零样本GPT-4o。

    

    语言模型智能体系统必须在规划或工具执行之前，将自然语言请求映射到一个或多个专用智能体。这一决策通常被处理为top-1选择，然而组合式请求可能需要多种能力，而不必要的分发则会增加成本。我们将固定目录下的路由问题表述为集合值语言理解任务，并构建了一个包含3,000条WildChat提示、覆盖12个按能力定义的智能体的基准测试。可复现的模式引导参考集合按智能体频率和路由基数进行了平衡。我们比较了画像检索、多标签分类器、微调文本编码器以及基于目录条件化的零样本GPT-4o。评估综合考察了集合质量、延迟、能力覆盖率和序数成本效用。微调编码器达到了89.59 F1，相比之下线性SVM为71.79，GPT-4o仅为41.51，并且在更难的语义组划分下仍保持88.02 F1。我们进一步引入了加权智能体路由……（摘要原文在此处截断）

    arXiv:2606.28925v3 Announce Type: replace-cross  Abstract: Language-model agent systems must map natural-language requests to one or more specialized agents before planning or tool execution. This decision is often treated as top-1 selection, although compositional requests can require several capabilities and unnecessary dispatch raises cost. We formulate fixed-catalog routing as set-valued language understanding and construct a benchmark of 3,000 WildChat prompts over 12 capability-defined agents. Reproducible schema-guided reference sets are balanced by agent frequency and route cardinality. We compare profile retrieval, multilabel classifiers, a fine-tuned text encoder, and catalog-conditioned zero-shot GPT-4o. Evaluation combines set quality, latency, capability coverage, and ordinal-cost utility. The encoder attains 89.59 F1, compared with 71.79 for linear SVM and 41.51 for GPT-4o, and retains 88.02 F1 under a harder semantic-group split. We further introduce Weighted Agent Routi
    
[^297]: ThermoLLM：基于时空上下文的LLM多区域暖通空调（HVAC）控制

    ThermoLLM: LLM-Based Multi-Zone HVAC Control with a Spatiotemporal Context

    [https://arxiv.org/abs/2606.22911](https://arxiv.org/abs/2606.22911)

    本文提出ThermoLLM，将建筑空间语义、区域热动力学与滚动历史作为LLM上下文的核心，从而实现更一致、更可泛化且能感知区域间延迟热交互的多区域HVAC控制。

    

    多区域HVAC控制是一个空间决策问题，室内热演化与控制决策不仅取决于室外条件和内部得热，还取决于区域布局、物理相邻关系以及整个建筑内跨区域的延迟热交互。近年来基于大语言模型（LLM）的HVAC控制器已经表明基于提示词的控制是可行的，然而这些方法产生不一致的结果、难以泛化，并存在控制延迟问题。其性能还严重依赖于数值奖励值、文本反馈或非结构化检索，这限制了它们做出能够兼顾区域动力学与热力学的控制决策的能力。目前尚无基于LLM的方法在将建筑语义与动力学连同滚动历史作为LLM上下文核心的同时，与现有基于LLM的HVAC方法进行控制性能的测试与比较。本文提出了ThermoL……（原文摘要在此处截断）

    arXiv:2606.22911v2 Announce Type: replace  Abstract: Multi-zone HVAC control is a spatial decision problem in which indoor thermal evolution and control decisions depend not only on outdoor conditions and internal heat gains but also on zone layout, physical adjacency, and delayed thermal interactions across the building. Recent LLM-based HVAC controllers have shown that prompt-based control is feasible. However, these methods produce inconsistent results, are difficult to generalize, and experience control-latency issues. Their performance also relies heavily on numeric reward values, textual feedback, or unstructured retrieval, which limits their ability to make control decisions that account for zone dynamics and thermodynamics. No existing LLM-based methods have used building semantics and dynamics, together with rolling history, as the centre of the LLM context while testing and comparing HVAC control performance with existing LLM-based HVAC approaches. This paper presents ThermoL
    
[^298]: 扩散驱动的状态空间模型

    Diffusion-Driven State Space Models

    [https://arxiv.org/abs/2606.21036](https://arxiv.org/abs/2606.21036)

    本文提出扩散驱动的状态空间模型（DDSSM），以扩散模型替代传统高斯转移分布，解决了在序列数据上联合训练自编码器与扩散模型的开放问题，并在多峰转移时间序列的拟合与预测上超越现有最先进的深度状态空间模型。

    

    在许多领域中，从业者希望模型既能产生准确的预测，又能忠实地捕捉潜在的系统动态。现有方法通常会牺牲其中一个目标：深度状态空间模型往往假设潜变量转移服从高斯分布，从而限制了拟合与预测能力，而扩散模型虽然具有很强的表达能力，却缺乏对底层动态的原则性推断。为了结合两者的优势，我们提出了扩散驱动的状态空间模型（DDSSM），它用扩散模型替代了传统的高斯转移分布。我们的DDSSM解决了如何在序列数据上联合训练自编码器和扩散模型这一开放性问题，从而拓展了面向时间序列的潜扩散模型相关文献。此外，我们发现DDSSM在拟合与预测具有多峰转移的模拟时间序列时，实证表现优于最先进的深度状态空间模型。

    arXiv:2606.21036v2 Announce Type: replace-cross  Abstract: In many domains, practitioners seek models that produce accurate forecasts while faithfully capturing latent system dynamics. Existing approaches typically sacrifice one of these goals: deep state space models often assume Gaussian latent transitions, limiting fit and forecasting, while diffusion models are highly expressive but lack principled inference for the underlying dynamics. To combine the strengths of both, we introduce the Diffusion-Driven State Space Model (DDSSM), which replaces the conventional Gaussian transition distribution with a diffusion model. Our DDSSM resolves the open problem of how to jointly train an autoencoder and a diffusion model on sequential data, thereby extending the literature on latent diffusion models for time series. Moreover, we find that the DDSSM empirically outperforms a state-of-the-art deep SSM at fitting and forecasting a simulated time series with multimodal transitions.
    
[^299]: 基于马尔可夫提升的Volterra生成模型

    Volterra Generative Models via Markovian Lifting

    [https://arxiv.org/abs/2606.18071](https://arxiv.org/abs/2606.18071)

    本文提出通过马尔可夫提升方法构建Volterra生成模型，将非马尔可夫的Volterra过程转化为可处理的马尔可夫近似，从而支持重用预训练去噪器，并在MNIST、CIFAR-10和ImageNet基准上取得了优异的FID性能。

    

    基于扩散的生成模型通常通过由布朗运动驱动的马尔可夫过程来扰动数据。Volterra过程允许更灵活的时间依赖结构，但其非马尔可夫特性使得得分学习和反向时间模拟变得复杂。我们开发了Volterra生成模型并提供了马尔可夫近似方法。借助高斯条件化，去噪问题被简化到原始数据维度。我们还阐明了该框架与方差爆炸和方差保持表示之间的联系。此外，我们的框架支持重用预训练的去噪器，例如EDM2网络。借助一个辅助因子，我们从头训练的模型在MNIST上取得了0.547的FID，在条件CIFAR-10上取得了1.860的FID。通过自动引导，ImageNet64在微调后FID达到1.912，而ImageNet512使用冻结的预训练网络FID达到1.903。

    arXiv:2606.18071v2 Announce Type: replace-cross  Abstract: Diffusion-based generative models usually perturb data through Markov processes driven by Brownian motion. Volterra processes allow more flexible temporal dependence, but their non-Markovian structure complicates score learning and reverse-time simulation. We develop Volterra generative models and provide Markovian approximations. Gaussian conditioning then reduces the denoising problem to the original data dimension. We present the connections with variance-exploding and variance-preserving representations. Our framework also permits reuse of pretrained denoisers such as EDM2 networks. With one auxiliary factor, our models trained from scratch achieve FIDs of $0.547$ on MNIST and $1.860$ on conditional CIFAR-10. With autoguidance, ImageNet64 FID reaches $1.912$ after fine-tuning, while ImageNet512 FID reaches $1.903$ using frozen pretrained networks.
    
[^300]: LatentGym：一个具有可控潜在结构的跨任务经验学习测试平台

    LatentGym: A Testbed For Cross-Task Experiential Learning With Controllable Latent Structure

    [https://arxiv.org/abs/2606.15306](https://arxiv.org/abs/2606.15306)

    提出了LatentGym测试平台，通过围绕真值潜在变量构建可控环境，实现了对智能体跨任务经验学习能力的度量，并可将探索与利用行为分开评估。

    

    我们设想持续学习的智能体系统能够随时间推移变得更有用：当它们遇到一系列相关任务时，应该推断出这些任务间共享的隐藏结构，并利用它来改进未来的决策。这种跨任务经验学习能力在个性化和交互式辅助等领域至关重要，但现有的训练/评估框架无法提供共享的、可控的潜在结构，也无法衡量智能体是否以及为何有所改进。我们提出了LatentGym：一个可控的测试套件，其中每个环境都围绕一个控制跨任务结构的真值潜在变量构建。我们的构造产生了能够区分探索（智能体的动作是否收集关于潜在变量的信息）与利用（智能体是否使用所收集的信息）的评估指标。我们通过实证研究展示了该套件，回答了三个问题：如何以及为什么……（摘要在此处被截断）

    arXiv:2606.15306v2 Announce Type: replace-cross  Abstract: We envision continually learning agentic systems that become more useful over time: as they encounter sequences of related tasks, they should infer the hidden structure shared across those tasks and use it to improve future decisions. This cross-task experiential learning capability is pivotal in domains such as personalization and interactive assistance, but existing training/evaluation frameworks do not provide shared, controllable latent structures and cannot measure whether or why agents improve. We introduce LatentGym: a controllable suite in which each environment is organized around a ground-truth latent variable governing the structure across tasks. Our construction yields metrics that separate exploration (whether the agent's actions gather information about the latent) from exploitation (whether the agent uses what it has gathered). We demonstrate our suite on empirical studies addressing three questions: how and why 
    
[^301]: 递归训练的扩散模型：极限坍缩分布与谱特征刻画

    Recursively Trained Diffusion Models: Limiting Collapse Distribution and Spectral Characterization

    [https://arxiv.org/abs/2606.13796](https://arxiv.org/abs/2606.13796)

    该论文证明，即使在评分估计与采样完全精确的理想条件下，反向扩散的提前停止也会使递归自训练的扩散模型以几何速率收敛到一个唯一的极限分布（即数据分布的 increasingly 高斯平滑版本的无限混合），且该过程如同低通滤波器般滤除高频信息，从而完全刻画了模型坍缩的极限分布与收敛速度。

    

    在生成模型自身的输出上进行递归训练可能导致模型坍缩（model collapse），即不断累积地偏离真实数据分布。现有理论工作针对扩散模型给出了有限轮次误差累积的界，但仍有两个问题悬而未决：递归最终会收敛到什么分布？收敛速度有多快？我们对这两个问题均给出了解答，并分离出一种不同于“学习不完美”的机制：即使评分估计完全精确、采样也完全精确，反向扩散过程的提前停止（数值稳定性所必需的）也会驱动分布逐渐偏离真实数据分布。我们证明该递归以几何速率收敛到一个唯一的极限分布，该分布具有闭式刻画——它是数据分布经过越来越强的高斯平滑后的版本的无限混合。对该极限分布的 Hermite 谱分解揭示出，递归训练起到了低通滤波器的作用：高频……（原文摘要在此处截断）

    arXiv:2606.13796v2 Announce Type: replace-cross  Abstract: Recursive training of generative models on their own outputs can lead to model collapse, a compounding drift away from the true data distribution. Existing theoretical works bound finite-round error accumulation in the context of diffusion models, but two questions remain open:~what distribution does the recursion converge to, and how fast? We answer both, isolating a mechanism distinct from imperfect learning: even with perfect score estimation and exact sampling, the early stopping of the reverse diffusion (required for numerical stability) drives a progressive drift away from the data distribution. We prove that this recursion converges geometrically to a unique limiting distribution, which admits a closed-form characterization as an infinite mixture of increasingly Gaussian-smoothed versions of the data distribution. A Hermite spectral decomposition of this limit reveals that recursive training acts as a low-pass filter: hi
    
[^302]: 稠密监督，稀疏更新：论在策略蒸馏的稀疏性与几何特性

    Dense Supervision, Sparse Updates: On the Sparsity and Geometry of On-Policy Distillation

    [https://arxiv.org/abs/2606.13657](https://arxiv.org/abs/2606.13657)

    本文系统刻画了在策略蒸馏参数更新的稀疏性与几何结构，发现其更新幅度小且坐标稀疏，仅在发现的关键子网络上训练即可几乎恢复完整性能，且更新避开源模型主结构方向、偏向低幅值坐标。

    

    在策略蒸馏已成为一种突出的后训练方法，它结合了两个理想要素：在策略的学生模型生成轨迹以及稠密的词元级教师监督。然而，这种混合训练机制如何塑造模型仍知之甚少。我们在多个语言和视觉-语言模型对以及多种应用场景中，刻画了OPD参数更新的稀疏性与几何特性。研究发现，OPD更新幅度小且坐标稀疏，同时仍分布在各层和各模块之中。这种稀疏支撑在操作上具有重要意义：仅在所发现的子网络上进行掩码训练，即可几乎恢复完整训练的性能。在矩阵层面，这些更新在数值上是满秩的，但谱上高度集中。其可见支撑避开了源模型主结构所强调的坐标，而偏向低幅值的源坐标，同时源模型的奇异值谱几乎保持不变。

    arXiv:2606.13657v4 Announce Type: replace  Abstract: On-policy distillation (OPD) has recently become a prominent post-training recipe by combining two desirable ingredients: on-policy student-generated trajectories and dense token-level teacher supervision. Yet how this hybrid training regime shapes a model remains poorly understood. We characterize the sparsity and geometry of OPD parameter updates across several language and vision-language model pairs and application settings. OPD updates are small and coordinate-sparse, while remaining distributed across layers and modules. This sparse support is operationally meaningful: masked training on the discovered subnetwork nearly recovers full-training performance. At the matrix level, the updates are numerically full-rank but spectrally concentrated. Their visible supports avoid coordinates emphasized by the source's principal structure and favor low-magnitude source coordinates, while the source singular-value spectra change little. To
    
[^303]: 标准可解释模型：基于拉格朗日力学的可解释机器学习通用理论，用于演绎设计可解释方法

    The Standard Interpretable Model: A general theory of interpretable machine learning to deductively design interpretable methods using Lagrangian mechanics

    [https://arxiv.org/abs/2606.12289](https://arxiv.org/abs/2606.12289)

    本文提出了一种基于拉格朗日力学的标准可解释模型（SIM）理论，通过演绎方式设计可解释方法，填补了可解释性理论与方法之间的空白。

    

    arXiv:2606.12289v2 公告类型：替换交叉 摘要：随着人工智能模型复杂度的增加，可解释性已成为理解、调试和控制其计算不可或缺的工具。然而，可解释性缺乏通用理论来演绎设计可解释方法。这种理论与方法之间的差距导致文献碎片化和评估协议不一致。为填补这一空白，我们引入了标准可解释模型（SIM），这是一个基于拉格朗日力学的通用理论，能够实现可解释方法的演绎设计。具体而言，SIM在一组前提中总结了可解释性对目标用户的意义。从这些前提出发，SIM系统地推导出可解释性对称性和相应约束，这些约束塑造了一个拉格朗日量的景观，其最小值对应于最优可解释模型。为达到最小值，可以更新操作参数的数值。

    arXiv:2606.12289v2 Announce Type: replace-cross  Abstract: As Artificial Intelligence models grow in complexity, interpretability has become an indispensable tool for understanding, debugging, and controlling their computations. However, interpretability lacks general theories to deductively design interpretable methods. This gap between theories and methods results in a fragmented literature and inconsistent evaluation protocols.   To fill this gap, we introduce the Standard Interpretable Model (SIM), a general theory grounded in Lagrangian mechanics that enables the deductive design of interpretable methods. Specifically, the SIM summarises, in a set of premises, what interpretability is for a target user. From these premises, the SIM systematically derives interpretability symmetries and corresponding constraints, which shape the landscape of a Lagrangian whose minima correspond to optimal interpretable models. To reach the minima, one can either update the parameter values of an op
    
[^304]: 当归因修补失效时：诊断与二阶修正

    When Attribution Patching Lies: Diagnosis and a Second-Order Correction

    [https://arxiv.org/abs/2606.09899](https://arxiv.org/abs/2606.09899)

    本文揭示归因修补的主要误差源于下游网络的非线性而非被修补组件的局部曲率，并据此提出可靠性分数、误差界限与二阶修正三项工具来诊断和改进归因修补的可信度。

    

    机械可解释性的一个核心目标是识别哪些内部组件因果地驱动语言模型的行为。由于这些重要性估计是识别电路的证据基础，系统性的误差可能导致对底层机制的错误识别。虽然激活修补提供了黄金标准的因果度量，但其计算成本在大规模应用时过于高昂。因此，实践者转而依赖归因修补，这是一种基于梯度的一阶近似方法，但其可靠性仍缺乏充分理解。在这项工作中，我们刻画了这种不可靠性的来源，证明其主要误差源于下游网络中的非线性，而非被修补组件处的局部曲率。这一洞见带来了三个实用工具：（i）一个用于检测不可信估计的可靠性分数；（ii）量化潜在归因错误设定的误差界限；（（摘要此处截断，第三项工具应为标题中提到的二阶修正）

    arXiv:2606.09899v2 Announce Type: replace-cross  Abstract: A central goal of mechanistic interpretability is to identify which internal components causally drive a language model's behavior. Because these importance estimates serve as the evidence for identifying circuits, systematic errors can lead to the misidentification of the underlying mechanisms. While activation patching provides a gold-standard causal metric, its computational cost is prohibitive at scale. Practitioners instead rely on attribution patching, a gradient-based, first-order approximation whose reliability remains poorly understood. In this work, we characterize the source of this unreliability, demonstrating that the dominant error stems from the non-linearities in the downstream network rather than local curvature at the patched component. This insight yields three practical tools: (i) a reliability score to detect untrustworthy estimates, (ii) error bounds quantifying potential attribution mis-specifications, an
    
[^305]: KV缓存量化下的对齐崩塌：诊断与缓解

    Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation

    [https://arxiv.org/abs/2606.09864](https://arxiv.org/abs/2606.09864)

    研究发现KV缓存低比特量化会在困惑度等常规指标不变的情况下隐性地破坏大语言模型的安全对齐，其根源在于安全特征位于对量化噪声极其敏感的低维子空间，据此提出的PCR诊断方法可用于识别并缓解此类失效。

    

    键值缓存量化被广泛用于降低大语言模型推理时的内存开销，然而现有评估仅关注困惑度和准确率，而未评估其对安全性的影响。在本研究中，我们探索了KV缓存量化下的对齐保持问题。通过对十一个指令微调模型（3.8B-72B）和五个基准测试（共1,894个提示）的实验，我们发现低比特量化会悄无声息地破坏安全对齐：Mistral-7B在困惑度仅为1.03倍时便丧失了15.2%的拒绝行为，且不存在通用的安全比特宽度，存在标准指标无法察觉的、模型特异的急剧相变。我们查明其根本原因是几何性的：安全特征占据一个低维激活子空间，与困惑度所平均的完整表示空间相比，该子空间对量化噪声的敏感度高出10²至10³倍。我们提出了逐通道缩减（PCR）诊断方法，可将每个模型归类为三种机制性失效模式之一（摘要原文在此处截断）。

    arXiv:2606.09864v3 Announce Type: replace-cross  Abstract: Key-value (KV) cache quantization is widely used to reduce LLM inference memory, yet existing evaluations focus only on perplexity and accuracy without assessing the safety impact. In this study, we explore alignment preservation under KV cache quantization. Across eleven instruction-tuned models (3.8B-72B) and five benchmarks (1,894 prompts), we find that low-bit quantization can silently destroy safety alignment: Mistral-7B loses 15.2% of its refusals at only 1.03x perplexity, and no universal safe bit-width exists, with sharp model-specific phase transitions invisible to standard metrics. We identify that the root cause is geometric: safety features occupy a low-dimensional activation subspace $10^2$-$10^3\times$ more vulnerable to quantization noise than the full representation space perplexity averages over. We propose Per-Channel Reduction (PCR), a diagnostic that classifies each model into one of three mechanistic failur
    
[^306]: 基于监督解耦的组织图反事实查询

    Querying Counterfactuals on Tissue Graphs with Supervised Disentanglement

    [https://arxiv.org/abs/2606.08493](https://arxiv.org/abs/2606.08493)

    该论文提出Cellina框架，通过监督解耦将细胞的内在状态与空间上下文分离，并首次将组织图反事实形式化为边扰动和节点扰动两类空间干预，在覆盖超过250万个细胞的基准测试中超越了现有空间与非空间方法。

    

    组织图反事实旨在回答这样一个问题：如果一个细胞的空间邻域发生改变，它会表达什么？这是预测组织中细胞行为的核心问题。然而，现有方法只能呈现此类查询的特定实例。在这项工作中，我们首先将组织图反事实形式化为一类空间干预，这些干预要么重新连接细胞之间的连接（边扰动），要么修改其邻居的表达（节点扰动）。随后，我们提出了Cellina框架，该框架利用监督解耦将细胞的内在状态从其空间环境中分解出来，并将空间环境作为计算机内扰动预测的条件输入。在涵盖超过250万个具有空间分辨率的观察性与干预性筛选细胞的基准测试中，Cellina在计算机内扰动预测方面优于空间感知和非空间的竞争方法，同时训练速度与非空间方法相当。

    arXiv:2606.08493v4 Announce Type: replace-cross  Abstract: Tissue graph counterfactuals ask what a cell would express if its spatial neighborhood were altered, a question central to predicting cell behavior in tissues. Existing methods, however, present only specific instances of such queries. In this work, we first formalize tissue graph counterfactuals as a class of spatial interventions that either rewire connections between cells (edge perturbation) or modify the expression of their neighbors (node perturbation). We then introduce Cellina, a framework that uses supervised disentanglement to decompose a cell's intrinsic state from its spatial context, using the latter as a conditioning input for in-silico perturbation predictions. Across benchmarks spanning over 2.5 million spatially-resolved cells in observational and interventional screens, Cellina outperforms both spatially-informed and non-spatial competitors in in-silico perturbations, while training as fast as non-spatial meth
    
[^307]: 和弦符号时间序列适配能在多大程度上承载流派身份？多流派和弦符号建模的能力与边界

    How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity? Capabilities and Boundaries in Multi-Genre Chord-Symbol Modeling

    [https://arxiv.org/abs/2606.07334](https://arxiv.org/abs/2606.07334)

    该研究将和弦符号序列建模为可解释、可控的时间序列，系统比较 LoRA、IA3、BitFit、前缀微调与全量微调五种方法在 11 个音乐流派上的适配能力与边界，发现所有方法均能提升和弦预测准确率，但序列头部词元会显著夸大评测分数，揭示了该建模方式的能力与局限。

    

    本报告将和弦符号序列视为一种可解释、可控制的时间序列，用于面向特定流派的和声建模。冻结的 Music Transformer 基础模型与以流行音乐为主的 Phase-0 基线在权重上完全相同。作者将其适配到十一个目标流派：布鲁斯、波萨诺瓦、巴赫众赞歌、乡村、电子、民谣、放克、福音、嘻哈、R&B/灵魂乐以及摇滚。主实验网格比较了 LoRA、IA3、BitFit、前缀微调和全量微调五种方法，覆盖 11 个流派和 3 个随机种子（共 165 个实验单元格）。所有五种方法在留出集和弦预测任务上均优于冻结的基础模型（宏平均提升 +2.89 至 +3.61 个百分点）。Top-1 指标同时会对每个序列开头的三个头部词元打分，而这些词元会左右整体结果：只有匹配的适配器才能预测出对应的流派标记，而模型是否能预测出 Chordonomicon 数据集中每首歌的占位 C 大调调号则近乎随机。若不计头部词元，宏平均提升为 +1.64 至 +2.38 个百分点，且在巴赫众赞歌之外，LoRA、IA3、前缀微调等方法的表现差异（原文在此截断）。

    arXiv:2606.07334v5 Announce Type: replace-cross  Abstract: This report treats chord-symbol sequences as an interpretable, controllable time series for genre-local harmonic modeling. The frozen Music Transformer base is weight-identical to a pop-dominated Phase-0 baseline. I adapt it to eleven target genres: blues, bossa nova, Bach chorales, country, electronic, folk, funk, gospel, hip-hop, R&B/soul, and rock. The main grid compares LoRA, IA3, BitFit, prefix tuning, and full fine-tuning over 11 genres and 3 seeds (165 cells). All five methods improve over the frozen base on held-out chord prediction (macro gains +2.89 to +3.61 percentage points). Top-1 also scores the three header tokens of each sequence, which move whole runs: only a matched adapter predicts its genre marker, and whether a run predicts the placeholder C-major key of every Chordonomicon song is close to arbitrary. Without the header tokens the macro gains are +1.64 to +2.38 pp, and outside Bach chorales LoRA, IA3, prefi
    
[^308]: 用于大语言模型智能体离线策略评估的自回归扩散世界模型

    Autoregressive Diffusion World Models for Off-Policy Evaluation of LLM Agents

    [https://arxiv.org/abs/2606.05558](https://arxiv.org/abs/2606.05558)

    提出自回归扩散世界模型ADWM，通过潜在扩散模拟环境响应，仅凭预先收集的轨迹即可离线评估LLM智能体策略性能，避免了昂贵的在线交互并克服了自回归世界模型的复合误差问题。

    

    在多轮交互环境中评估大语言模型（LLM）智能体成本高昂且存在风险，因为这需要在线的环境交互。我们提出了ADWM（自回归扩散世界模型），这是一个评估框架，能够仅从预先收集的轨迹中估计新的LLM智能体策略的性能。其核心思想是学习一个潜在扩散世界模型来模拟环境对被评估策略的响应，而无需在真实环境中执行该策略。现有的基于扩散的离线策略评估（OPE）方法通过联合扩散状态和动作来一次性引导完整轨迹，这一假设对于LLM智能体会失效，因为LLM智能体的动作是离散文本，必须在观察环境后从策略中采样得到。与遭受复合误差困扰的自回归世界模型不同，ADWM将每个状态转移建模为独立的去噪过程，从而实现可靠的逐步（摘要在此处被截断）

    arXiv:2606.05558v3 Announce Type: replace  Abstract: Evaluating large language model (LLM) agents in multi-turn interactive environments is expensive and risky, as it requires online environment interaction. We propose ADWM (Autoregressive Diffusion World Model), an evaluation framework that estimates the performance of a new LLM agent policy purely from pre-collected trajectories. The core idea is to learn a latent diffusion world model that simulates how the environment responds to the evaluation policy, without ever executing it in the real environment. Existing diffusion-based OPE methods guide full trajectories in a single pass by jointly diffusing states and actions, an assumption that breaks down for LLM agents whose actions are discrete text that must be sampled from the policy after observing the environment. Unlike autoregressive world models that suffer from compounding errors, ADWM models each transition as an independent denoising process, enabling reliable step-by-step ro
    
[^309]: 网络上近最优的去中心化随机凸优化

    Near-Optimal Decentralized Stochastic Convex Optimization over Networks

    [https://arxiv.org/abs/2606.04757](https://arxiv.org/abs/2606.04757)

    提出一种基于单步延迟随机加速机制的去中心化优化方法，将保持中心化 $O(1/\sqrt N)$ 统计速率的最大工作者数量从原有的 $\rho\sqrt N$ 提升至 $\sqrt{\rho}\,N^{3/4}$，达到近最优的规模扩展。

    

    本文研究去中心化随机光滑凸优化问题，其中 $M$ 个工作者在固定的gossip通信网络上，利用本地随机梯度并仅与邻居通信，共同最小化一个平均目标函数。该设定下的一个核心问题是：在总共 $N$ 个梯度样本的预算下，确定在仍能保持中心化 $O(1/\sqrt N)$ 统计速率的前提下，可使用的最大工作者数量。我们提出了一种加速去中心化方法，当工作者数量多达 $M\lesssim \sqrt{\rho}\,N^{3/4}$ 时仍能保持该速率，其中 $\rho$ 为gossip网络的谱隙，这改进了此前最优的 $M\lesssim \rho\sqrt N$ 的最大可扩展规模。该方法基于一种单步延迟的随机加速方案，使工作者能够在小批量（minibatch）计算与加速gossip通信之间交替进行，同时控制剩余的分歧，其理论保证仅对最优值-局部……（原文截断）呈对数依赖。

    arXiv:2606.04757v2 Announce Type: replace-cross  Abstract: We study decentralized stochastic smooth convex optimization, where $M$ workers minimize an average objective using local stochastic gradients and neighbor-only communication over a fixed gossip network. A central question in this setting is to determine the largest number of workers that can be used under a total budget of $N$ gradient samples while still preserving the centralized $O(1/\sqrt N)$ statistical rate. We introduce an accelerated decentralized method that preserves this rate for up to $\smash{M\lesssim \sqrt{\rho}\,N^{3/4}}$ workers, where $\rho$ is the spectral gap of the gossip network, improving the best prior maximal scaling of $\smash{M\lesssim \rho\sqrt N}$. The method is based on a one-step-delayed stochastic acceleration scheme that enables workers to interleave minibatching with accelerated gossip while controlling residual disagreement, and its guarantee depends only logarithmically on the optimum-local h
    
[^310]: Murmur：基于分层分块的高效长上下文自动语音识别推理

    Murmur: Efficient Long-Context ASR Inference with Hierarchical Chunking

    [https://arxiv.org/abs/2606.01483](https://arxiv.org/abs/2606.01483)

    提出了基于分层分块的长篇ASR推理系统Murmur，发现约五分钟的中间块大小可在避免重复循环的同时保持单次处理的准确率，并通过并行解码兼顾高准确率与低推理延迟。

    

    长篇自动语音识别（ASR）既需要高准确率又需要低推理延迟，但现有系统难以在两者之间取得良好平衡。基于分块的方法通过并行窗口处理音频来获得低延迟，但会丢失跨块上下文，并且需要脆弱的启发式方法在边界处对齐说话人和时间戳。长上下文ASR模型通过单次处理转录全部内容以获得更好的准确率，但运行速度慢数倍，且单个重复循环就可能破坏整个输出。在本文中，我们首先分析了上下文长度如何影响长上下文ASR模型的准确率，发现约五分钟的中间块大小可以在避免重复循环的同时达到与单次处理相当的准确率。基于这一发现，我们提出了Murmur，一个基于分层分块的长篇ASR推理系统。长上下文模型并行解码中间目标块，并由一个快速的（摘要在此处被截断）

    arXiv:2606.01483v2 Announce Type: replace-cross  Abstract: Long-form automatic speech recognition (ASR) requires both high accuracy and low inference latency, but existing systems fall short of handling the trade-off between the two. Chunk-based approaches process audio in parallel windows for low latency, but lose cross-chunk context and need brittle heuristics to align speakers and timestamps at boundaries. Long-context ASR models transcribe everything in a single pass for better accuracy, but run several times slower, and a single repetition loop can corrupt the entire output. In this paper, we first analyze how context length affects the accuracy of long-context ASR models and find that an intermediate chunk size of about five minutes matches single-pass accuracy while avoiding repetition loops. Building on this, we propose Murmur, an inference system for long-form ASR based on hierarchical chunking. The long-context model decodes intermediate target chunks in parallel, and a fast 
    
[^311]: ForecastCompass：用自适应因子记忆引导智能体预测

    ForecastCompass: Guiding Agentic Forecasting with Adaptive Factor Memory

    [https://arxiv.org/abs/2605.30858](https://arxiv.org/abs/2605.30858)

    提出了ForecastCompass（FoCo），一个面向智能体预测的自适应因子记忆框架，通过分层任务分类体系组织预测经验，并维护因子记忆与校准知识两类互补记忆，实现可复用预测知识的检索，从而提升动态环境下的概率预测能力。

    

    智能体预测对于动态环境中的决策制定非常重要，但它仍然具有挑战性，因为智能体必须在不完整、有时限的证据基础上进行推理，并在结果揭晓之前产生经过校准的概率预测。记忆为将已解决预测中的经验迁移到未来预测任务提供了一种自然的机制。然而，现有的智能体记忆方法并非专为预测任务设计，它们通常存储过去的交互、反思或事实关联，而没有显式地表示可复用的预测因子或校准知识。我们提出了ForecastCompass（FoCo），这是一个面向智能体预测的基于自适应因子的记忆框架。FoCo通过分层预测任务分类体系来组织预测经验，从而实现与任务相关的预测知识的检索。它维护两个互补的记忆组件：因子记忆，用于捕捉可复用的预测维度（摘要在此处被截断）

    arXiv:2605.30858v2 Announce Type: replace  Abstract: Agentic forecasting is important for decision-making in dynamic environments, but it remains challenging because agents must reason from incomplete, time-limited evidence and produce calibrated probabilities before outcomes are resolved. Memory provides a natural mechanism for transferring experience from resolved forecasts to future prediction tasks. However, existing agent-memory methods are not tailored to forecasting, as they typically store past interactions, reflections, or factual associations without explicitly representing reusable predictive factors or calibration knowledge. We propose ForecastCompass (FoCo), an adaptive factor-based memory framework for agentic forecasting. FoCo organizes forecasting experience with a hierarchical forecasting-task taxonomy, enabling retrieval task-relevant forecasting knowledge. It maintains two complementary memory components: factor memory, which captures reusable predictive dimensions, 
    
[^312]: 知识卸载：将大语言模型分解为稀疏骨干网络与记忆模块

    Knowledge Offloading: Decomposing LLMs into Sparse Backbones and Memory Modules

    [https://arxiv.org/abs/2605.29075](https://arxiv.org/abs/2605.29075)

    该论文提出知识卸载（KOFF）框架，通过结构化剪枝掩码与LoRA适配器、可学习KV缓存等轻量恢复模块，将预训练大语言模型分解为稀疏共享骨干网络和领域特定记忆模块，在保持约90%性能的同时可将12%-20%的容量从骨干网络中移出。

    

    大语言模型在单一参数集中同时编码了通用能力和领域特定知识。我们探究这种容量是否可以被重新组织：将广泛有用的计算保留在共享骨干网络中，同时将专门知识迁移到外部记忆模块中。我们提出了知识卸载（KOFF）框架，用于将预训练大语言模型分解为稀疏共享骨干网络和领域特定记忆模块。从一个冻结的基座模型出发，我们通过联合或顺序的方式学习结构化剪枝掩码和轻量级恢复模块，这些模块以LoRA适配器和可学习的键值缓存形式实现。在从3B到14B的Llama和Qwen模型上，我们发现12%到20%的容量可以从共享骨干网络中移出，同时基本保持困惑度和MMLU性能（最高达到完整模型的90%）。我们展示了LoRA和可学习的KV记忆是互补的，分析表明被卸载的知识具有特定的领域属性。

    arXiv:2605.29075v2 Announce Type: replace  Abstract: LLMs encode both general capabilities and domain-specific knowledge in a single set of parameters. We ask whether this capacity can be reorganized: keeping broadly useful computation in a shared backbone, while moving specialized knowledge into external memory modules. We propose \emph{knowledge offloading} (KOFF), a framework for decomposing a pretrained LLM into a sparse shared backbone and domain-specific memories. Starting from a frozen base model, we either jointly or sequentially learn a structured pruning mask and lightweight recovery modules, implemented as LoRA adapters and learned key-value caches. Across Llama and Qwen models from 3B to 14B, we find that 12\% to 20\% of capacity can be moved out of the shared backbone while largely preserving perplexity and MMLU performance (up to 90\% of the full model). We show that LoRA and learned KV memories are complementary, and analyses suggest that the offloaded knowledge is speci
    
[^313]: 按其思考方式施教：用稀疏自编码器引导大语言模型后训练数据工程

    Teach the Way It Thinks: Guiding LLM Post-training Data Engineering with Sparse Autoencoders

    [https://arxiv.org/abs/2605.27354](https://arxiv.org/abs/2605.27354)

    该论文提出SAERL方法，利用稀疏自编码器（SAE）将模型内部分解为稀疏可解释的特征，从模型内部视角（激活与梯度投影）来组织强化学习训练数据，从而决定训练哪些样本、何时呈现以及如何分批，突破了传统仅依赖通过率等外部反馈的数据工程方式。

    

    大语言模型（LLM）强化学习（RL）的数据工程决定了训练哪些样本、何时呈现这些样本，以及哪些样本应一起分批。现有方法大多依赖外部反馈，例如通过率或难度标签，这只能观察到每个样本上的结果。我们认为，模型内部提供了更丰富、更贴近模型自身的反馈，能够指导上述全部三项决策。我们提出SAERL，它在稀疏自编码器（SAE）的特征空间中组织强化学习训练数据。SAE是一种机制可解释性工具，可将模型内部分解为稀疏且可解释的特征。借助SAE特征，我们可以将模型视为一组功能区域，每个区域由相关样本共同激活的特征所构成。对于每个样本，其SAE激活表明它参与了哪些区域，而其投影到SAE特征上的梯度则表明对它进行训练将如何改变这些区域。在此指导下……（原文摘要至此被截断）

    arXiv:2605.27354v2 Announce Type: replace-cross  Abstract: Data engineering for large language model (LLM) reinforcement learning (RL) decides which samples to train on, when to present them, and which to batch together. Existing methods mostly rely on external feedback, such as pass rates or difficulty labels, which observes only the outcome on each sample. We argue that model internals offer richer and more model-intrinsic feedback that can guide all three decisions. We propose SAERL which organizes RL training data in the feature space of Sparse Autoencoders (SAEs), a mechanistic interpretability tool that decomposes model internals into sparse, interpretable features. SAE features let us view the model as a set of functional regions, each formed by features that related samples activate together. For each sample, its SAE activations indicate which regions it engages, and its gradients projected onto SAE features indicate how training on it would change these regions. Guided by this
    
[^314]: 万物于每一尺度：具备连续超分辨率的尺度不变扩散模型

    Everything at Every Scale: Scale-Invariant Diffusion with Continuous Super-Resolution

    [https://arxiv.org/abs/2605.26032](https://arxiv.org/abs/2605.26032)

    提出尺度不变的扩散模型 SKILD，将图像生成与连续超分辨率统一在单一无条件框架中，仅通过改变起始时间步即可在两种任务间切换。

    

    从噪声中创建图像是图像生成；从粗糙输入中重建精细细节是超分辨率。尽管二者在实际应用中有所不同，但都可以理解为对跨尺度信息损失的逆转。我们提出了 SKILD——一种尺度不变的 K 空间图像学习扩散模型，它在单一的无条件框架内统一了图像生成与连续超分辨率。自然图像与临界物理系统均表现出尺度不变性，我们利用这一特性设计了一个前向过程，该过程将图像内容从细尺度到粗尺度逐步衰减，同时注入与频谱匹配的高斯噪声，从而使尺度成为扩散动力学的一个显式坐标。同一个经过训练的逆向过程仅需改变起始时间步即可执行生成或连续超分辨率：无需任务特定的架构，无需条件分支……

    arXiv:2605.26032v2 Announce Type: replace-cross  Abstract: Creating images from noise is image generation; reconstructing fine details from coarse inputs is super-resolution. Despite their practical differences, both can be understood as reversing information loss across scales. We introduce $\textbf{SKILD}$, a $\textbf{S}$cale-invariant $\textbf{K}$-Space $\textbf{I}$mage $\textbf{L}$earning $\textbf{D}$iffusion model that unifies generation and continuous super-resolution within a single unconditional framework. Both natural images and critical physical systems exhibit scale invariance, and we leverage it to design a forward process that attenuates image content from fine to coarse scales while injecting spectrum-matched Gaussian noise, making scale an explicit coordinate of the diffusion dynamics. The same trained reverse process performs generation and continuous super-resolution by varying only the starting timestep: $\textit{no task-specific architecture, no conditioning branch, 
    
[^315]: 无残差Transformer的量化优势

    The Quantization Benefits of Residual-Free Transformers

    [https://arxiv.org/abs/2605.25880](https://arxiv.org/abs/2605.25880)

    本文揭示了残差连接会在训练中放大Transformer激活值的非高斯性（重尾、离群值主导），从而加剧低比特量化的误差与精度损失，而无残差Transformer凭借稠密混合收缩非高斯性，在量化部署中具有天然优势。

    

    arXiv:2605.25880v2 公告类型：replace 摘要：大规模Transformer的训练与部署日益受到激活值、梯度和优化器状态在加速器之间传输的制约。低比特量化为这一问题提供了天然的解决方案，但Transformer的激活值通常呈重尾分布且被离群值主导，使得简单的量化方法带来严重的信息损失。我们表明，这一困难不仅源于量化器本身，也源于架构本身。具体而言，残差连接会在训练过程中使Transformer的激活值偏离高斯分布。通过对残差与无残差Transformer的受控对比实验，我们证明这一效应导致残差模型在低精度下产生显著更高的量化误差和精度下降。我们通过超额峰度分析解释了这一现象，指出残差混合会放大激活值的非高斯性，而无残差模型中的稠密混合则会收缩非高斯性。

    arXiv:2605.25880v2 Announce Type: replace  Abstract: Large-scale transformer training and deployment are increasingly constrained by the transfer of activations, gradients, and optimizer states across accelerators. Low-bit quantization offers a natural remedy, but transformer activations are often heavy-tailed and outlier-dominated, making simple quantization highly lossy. We show that this difficulty is not only a property of the quantizer, but also of the architecture. Specifically, residual connections can drive transformer activations away from Gaussianity during training. Using controlled comparisons between residual and residual-free transformers, we demonstrate that this effect leads to substantially higher quantization error and accuracy degradation at low precision in residual models. We explain the phenomenon through an excess kurtosis analysis, showing that residual mixing can amplify non-Gaussianity, whereas dense mixing in residual-free models contracts non-Gaussianity. We
    
[^316]: 面向扩散多模态大语言模型的视觉冗余控制并行解码

    Visual-Redundancy-Controlled Parallel Decoding for Diffusion-Based Multimodal Large Language Models

    [https://arxiv.org/abs/2605.25820](https://arxiv.org/abs/2605.25820)

    该论文发现扩散多模态大语言模型并行解码中同一步骤内被提交的高置信度token可能依赖重叠的视觉定位信息而产生视觉冗余，并提出视觉冗余指数（VRI）来量化这一效应以改进解码策略。

    

    基于扩散的多模态大语言模型通过在多个掩码位置并行迭代预测token来进行解码。这使得每个解码步骤变成了一个位置选择问题：模型不仅要判断哪些预测在单独来看是可靠的，还要决定哪些位置应当被一同提交作为后续解码步骤的上下文。现有的基于置信度的解码方法独立地对掩码位置进行排序并提交得分最高的K个位置，很大程度上忽略了被提交的token是否提供了互补的视觉定位信息。我们发现了该策略在多模态场景下的一个步骤级局限：同一步骤中选出的高置信度token可能依赖于相互重叠的视觉定位信息，从而在被提交的token之间引入视觉冗余，并使后续解码可用的互补视觉定位信息减少。为了量化这一效应，我们提出了视觉冗余指数（Visual Redundancy Index, VRI），它衡量……（摘要原文不完整，此处为可获得的全部内容）

    arXiv:2605.25820v3 Announce Type: replace  Abstract: Diffusion-based multimodal large language models (dMLLMs) decode by iteratively predicting tokens at multiple masked positions in parallel. This turns each decoding step into a position-selection problem: the model must choose not only which predictions are reliable in isolation, but also which positions should be committed together as context for later decoding steps. Existing confidence-based decoding ranks masked positions independently and commits the top-K positions, largely ignoring whether the committed tokens provide complementary visual grounding. We identify a step-level limitation of this strategy in multimodal settings: high-confidence tokens selected in the same step can rely on overlapping visual grounding, introducing visual redundancy among the committed tokens and leaving less complementary visual grounding available for later decoding. To quantify this effect, we introduce the Visual Redundancy Index (VRI), which me
    
[^317]: LAPLEX：可学习拉普拉斯核的快速傅里叶变换

    LAPLEX: The FFT of Learnable Laplace Kernels

    [https://arxiv.org/abs/2605.24584](https://arxiv.org/abs/2605.24584)

    LAPLEX提出了一类精确且可训练的拉普拉斯核算子层，以FFT式的扩展效率实现了可学习的自适应几何结构，支持高达10^9维度的可训练矩阵运算，并可高效用于高维协方差建模。

    

    深度学习中的快速线性代数通常面临一种权衡选择：要么采用固定几何结构与精确计算（如傅里叶变换），要么为获得自适应几何结构而付出密集参数、随机特征或低秩代理的代价。为了突破这一权衡，我们提出了LAPLEX，一类精确且可训练的（相位化）拉普拉斯核算子。LAPLEX层是一个通常为满秩的稠密矩阵，由可学习的坐标锚点隐式定义，并具有类似FFT的扩展性。因此，它能够在现代GPU上支持向量维度高达10^9的可训练矩阵-向量运算。作为神经层，它可以产生紧凑的投影层和分类头，这些分类头可解释为软性的、可训练的路由模型。同一基元还可用作高效的Gram算子，使得在维度为3×10^6的展平图像上构建高维协方差模型成为可能，既能保留可见的空间结构，又无需施加卷积偏置。

    arXiv:2605.24584v2 Announce Type: replace-cross  Abstract: Fast linear algebra in deep learning usually comes with a choice: fixed geometry and exact computation, as in the Fourier transform, or adaptive geometry paid for by dense parameters, random features, or low-rank surrogates. To move beyond this trade-off, we introduce LAPLEX, a class of exact, trainable (phased) Laplace-kernel operators.   A LAPLEX layer is a typically full-rank dense matrix, implicitly defined by learnable coordinate anchors, with FFT-like scaling. Consequently, it supports trainable matrix--vector operations at vector dimensions up to $10^9$ on modern GPUs. As a neural layer, it yields compact projections and classification heads interpretable as soft, trainable routing models. The same primitive also serves as an efficient Gram operator, enabling high-dimensional covariance models on flattened images of dimension $3 \cdot 10^6$ that preserve visible spatial structure without imposing convolutional bias. Thes
    
[^318]: VACE：为时间序列异常检测学习几何结构化表示

    VACE: Learning Geometrically Structured Representations for Time Series Anomaly Detection

    [https://arxiv.org/abs/2605.23504](https://arxiv.org/abs/2605.23504)

    VACE提出了一种自监督异常检测方法，通过速度对齐通道嵌入显式塑造潜空间几何结构，将正常性表示为局部平滑、方向连贯的表示，克服了对比学习无法显式控制距离评分所依赖几何结构的缺陷。

    

    多变量时间序列中的异常检测是众多现实应用中的一项关键任务，在这些场景中异常行为稀少、标签不可用，且漏检的代价高昂。核心挑战在于学习一种足够精确的正常性刻画，以便标记偏差。表示自监督学习（通常采用对比方法）通过将时间片段嵌入潜空间来解决这一问题，在该空间中正常性占据一个明确定义的区域，并通过几何偏差来检测异常。然而，对比方法通过成对采样启发式间接地塑造这一空间，无法对基于距离评分所依赖的几何结构进行显式控制，因此这种几何结构只能听凭运气。我们提出了VACE（速度对齐通道嵌入），这是一种自监督异常检测方法，它将正常性表示为局部平滑、方向连贯的……

    arXiv:2605.23504v2 Announce Type: replace-cross  Abstract: Anomaly detection in multivariate time series is a critical task across a wide range of real-world applications, where abnormal behaviour is rare, labels are unavailable, and the cost of a miss is high. The central challenge is learning a characterisation of normality precise enough to flag deviations. Representation self-supervised learning, typically through contrastive approaches, addresses this by embedding temporal patches into a latent space where normality occupies a well-defined region, with anomalies detected by geometric deviation. However, contrastive approaches shape this space indirectly through pair-sampling heuristics, providing no explicit control over the geometric structure on which distance-based scoring operates. That geometry is thus left to chance. We present VACE (Velocity-Aligned Channel Embeddings), a self-supervised anomaly detection method that represents normality as a locally smooth, directionally c
    
[^319]: 向量策略优化：面向多样性的训练改进测试时搜索

    Vector Policy Optimization: Training for Diversity Improves Test-Time Search

    [https://arxiv.org/abs/2605.22817](https://arxiv.org/abs/2605.22817)

    提出向量策略优化（VPO），通过利用向量值奖励、多答案提示与集合级奖励来训练大语言模型生成多样化的解决方案集合，从而提升推理时搜索的效果。

    

    语言模型现在必须能够开箱即用地泛化到新颖环境中，并能在推理扩展搜索过程（如AlphaEvolve）中工作，这类搜索过程使用多种任务特定的奖励函数来选择 rollout。遗憾的是，标准的 LLM 后训练范式优化的是预先指定的标量奖励，这往往导致当前的 LLM 产生低熵的响应分布，从而难以展现推理时搜索所需的多样性。我们提出了向量策略优化（VPO），这是一种强化学习算法，它显式地训练策略以预期多样化的下游奖励函数并产生多样化的解决方案。VPO 利用了实践中奖励往往是向量值的事实，例如代码生成中每个测试用例的正确性，或者多个不同的用户角色或奖励模型。VPO 将多答案提示与集合级奖励相结合，训练 LLM 生成能够……

    arXiv:2605.22817v2 Announce Type: replace-cross  Abstract: Language models must now generalize out of the box to novel environments and work inside inference-scaling search procedures, such as AlphaEvolve, that select rollouts with a variety of task-specific reward functions. Unfortunately, the standard paradigm of LLM post-training optimizes a pre-specified scalar reward, often leading current LLMs to produce low-entropy response distributions and thus to struggle at displaying the diversity that inference-time search will require. We propose Vector Policy Optimization (VPO), an RL algorithm that explicitly trains policies to anticipate diverse downstream reward functions and to produce diverse solutions. VPO exploits that rewards are often vector-valued in practice, like per-test-case correctness in code generation or, say, multiple different user personas or reward models. VPO combines multi-answer prompting with a set-level reward to train the LLM to produce sets of solutions that 
    
[^320]: ASAP：基于注意力汇聚锚定的剪枝方法

    ASAP: Attention Sink Anchored Pruning

    [https://arxiv.org/abs/2605.22372](https://arxiv.org/abs/2605.22372)

    ASAP 提出了一种无需训练的 ViT token 剪枝框架，将注意力汇聚现象建模为惰性随机游走中的概率累积特征，通过径向扩散聚类与转移权重池化一次性压缩背景冗余 token。

    

    视觉 Transformer（ViT）由于高分辨率下自注意力的二次复杂度而面临严重的计算瓶颈。现有的 token 削减方法依赖于局部指标——例如单层注意力分数——这些指标天然容易受到注意力汇聚现象的影响，导致无信息量的 token 反而被保留，而显著的前景目标却被丢弃。我们提出了 ASAP（Attention Sink Anchored Pruning，注意力汇聚锚定剪枝），这是一个无需训练的框架，将这种汇聚现象重新定义为一种特征。通过将 ViT 的信息流建模为惰性随机游走，ASAP 将汇聚点识别为概率质量的主要累积器。通过在累积转移矩阵中计算到汇聚点的扩散距离，ASAP 利用径向扩散聚类对 token 进行划分，并通过转移权重池化一次性压缩背景冗余。在图像、视频和视觉-语言任务上的大量实验证明了 ASAP 的（摘要在此处截断）

    arXiv:2605.22372v2 Announce Type: replace  Abstract: Vision Transformers (ViTs) face severe computational bottlenecks due to the quadratic complexity of self-attention at high resolutions. Existing token reduction methods rely on local metrics -- such as single-layer attention scores -- that are inherently vulnerable to the attention sink phenomenon, where uninformative tokens are paradoxically preserved over salient foreground objects. We propose ASAP (Attention Sink Anchored Pruning), a training-free framework that recasts this sink as a feature. Modeling ViT information flow as a Lazy Random Walk, ASAP identifies the sink as a dominant accumulator of probability mass. By computing the diffusion distance to the sink within the cumulative transition matrix, ASAP partitions tokens via Radial Diffusion Clustering and compresses background redundancy through Transition Weight Pooling in a single shot. Extensive experiments across image, video, and vision-language tasks demonstrate ASAP o
    
[^321]: 你的邻居知道：利用本地邻域进行去中心化学习中的后门检测

    Your Neighbors Know: Leveraging Local Neighborhoods for Backdoor Detection in Decentralized Learning

    [https://arxiv.org/abs/2605.19969](https://arxiv.org/abs/2605.19969)

    提出了去中心化学习原生的新型后门检测框架Argus，无需中央协调者或触发器先验知识，通过节点本地分析模型更新、与邻居共享触发器并利用结构相似性度量，有效区分真实后门攻击与数据异质性导致的误报。

    

    去中心化学习是一种新兴的机器学习范式，其中节点在没有中央服务器的情况下协作训练模型。然而，去中心化学习的协作特性使其容易受到后门攻击，即模型被训练为在标准输入上表现正常，而在遇到带有特定触发器的数据时执行隐藏的恶意操作。去中心化学习中的后门攻击目前仍研究不足，且现有防御方法往往忽视去中心化学习的约束条件。我们提出了Argus，一种原生支持去中心化学习的全新后门检测框架，它既不需要中央协调者，也不需要预先了解触发器。在Argus中，诚实节点在本地分析接收到的模型更新，以识别潜在的后门触发器。随后，节点将各自的触发器与邻居共享，并使用结构相似性度量来区分真正的后门与由数据异质性引起的误报。一个关键洞察是，误报的触发器（摘要在此处截断）

    arXiv:2605.19969v3 Announce Type: replace  Abstract: Decentralized learning (DL) is an emerging machine learning paradigm where nodes collaboratively train models without a central server. However, the collaborative nature of DL makes it vulnerable to backdoor attacks, where a model is taught to behave normally on standard inputs while executing hidden, malicious actions when encountering data with specific triggers. Backdoor attacks in DL remain understudied and existing defenses often overlook DL constraints. We introduce Argus, a novel backdoor detection framework native to DL that requires neither a central coordinator nor prior knowledge of the trigger. In Argus, honest nodes locally analyze received model updates to identify potential backdoor triggers. Nodes then collectively share their triggers with their neighbors and use a structural similarity metric to separate true backdoors from false alarms induced by data heterogeneity. A key insight is that false positive triggers exh
    
[^322]: KVBuffer：面向线性注意力的IO感知服务系统

    KVBuffer: IO-aware Serving for Linear Attention

    [https://arxiv.org/abs/2605.19049](https://arxiv.org/abs/2605.19049)

    KVBuffer是一种面向线性注意力的IO感知服务机制，通过缓冲最近的键和值实现分块计算与批量状态更新，显著减少内存访问量并降低解码延迟，同时高效支持投机解码验证。

    

    线性注意力近来因其在长上下文推理中相对于上下文长度具有恒定的解码成本而受到广泛关注。然而，现有的服务系统通常通过在每个解码步骤中递归地计算和更新一个庞大的线性注意力状态来提供线性注意力服务。由于该状态远大于每个token的键和值，递归解码会导致大量的内存访问，使得线性注意力的服务效率低下。在本文中，我们提出了KVBuffer，一种面向线性注意力的IO感知服务机制。通过缓冲最近的键和值，KVBuffer使服务系统能够以更灵活、更节省内存的方式计算线性注意力输出。对于解码，KVBuffer支持分块计算，通过延迟状态更新并将其批量应用，降低了平均内存访问量和解码延迟。对于投机解码，KVBuffer通过缓存机制高效验证草稿token

    arXiv:2605.19049v2 Announce Type: replace-cross  Abstract: Linear attention has recently gained significant attention for long-context inference due to its constant decoding cost with respect to context length. However, existing serving systems typically serve linear attention by recurrently computing and updating a large linear attention state in every decoding step. Since the state is much larger than the per-token key and value, recurrent decoding incurs substantial memory access and becomes inefficient for serving linear attention. In this paper, we propose KVBuffer, an IO-aware serving mechanism for linear attention. By buffering recent keys and values, KVBuffer enables serving systems to compute linear attention outputs in more flexible and memory-efficient ways. For decoding, KVBuffer enables chunkwise computation, which reduces average memory access and decoding latency by deferring state updates and applying them in batch. For speculative decoding, KVBuffer verifies draft toke
    
[^323]: 软标签学习与校准中人类不确定性与模型不确定性的评估

    An Assessment of Human vs. Model Uncertainty in Soft-Label Learning and Calibration

    [https://arxiv.org/abs/2605.18648](https://arxiv.org/abs/2605.18648)

    本研究通过解耦软标签监督与标签模态偏移，发现人类软标签的主要价值在于作为正则化器提升模型对困难样本的校准能力和训练收敛稳定性，并使模型的不确定性表现与人类一致。

    

    以人为本的对齐AI的核心在于理解人类提供的标签相较于合成标签的优势。虽然人类软标签通过捕捉不确定性能够改善模型校准，但先前的研究将这些优势与对错误标注数据的隐式纠正（模态偏移）混为一谈，从而掩盖了软标签的真正效果。我们对MNIST及其合成变体上的软标签学习进行了受控审计，通过重新标注数据子集来提取人类不确定性。通过将软标签监督与底层标签模态偏移解耦，我们证明人类软标签确实能带来准确率提升，但其更大的价值在于作为一种正则化器，能够改善模型对困难样本的校准，并促进训练过程中的稳定收敛。数据集制图分析表明，使用人类软标签训练的模型能够反映人类的不确定性，而使用合成标签训练的模型则无法与人类对齐。总体而言，这项工作……（原文截断）

    arXiv:2605.18648v3 Announce Type: replace-cross  Abstract: Central to human-aligned AI is understanding the benefits of human-elicited labels over synthetic alternatives. While human soft-labels improve calibration by capturing uncertainty, prior studies conflate these benefits with the implicit correction of mislabeled data (mode shifts), obscuring true effects of soft-labels. We present a controlled audit of soft-label learning across MNIST and a synthetic variant, re-annotating subsets to extract human uncertainty. By decoupling soft-label supervision from underlying label mode shifts, we show that while human soft-labels do provide accuracy gains, their larger value lies in acting as a regularizer that improves model calibration on difficult samples and promotes stable convergence across training runs. Dataset cartography reveals models trained on human soft-labels mirror human uncertainty, whereas those trained on synthetic labels fail to align with humans. Broadly, this work prov
    
[^324]: 面向离散不确定性两阶段鲁棒优化的学习式场景缩减方法

    Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete Uncertainty

    [https://arxiv.org/abs/2605.14494](https://arxiv.org/abs/2605.14494)

    本文提出了问题驱动的场景缩减启发式算法 PRISE 及其基于 GNN-Transformer 的神经代理模型 NeurPRISE，能够高效构建高质量场景子集，显著提升离散不确定性两阶段鲁棒优化的可扩展求解能力。

    

    具有离散不确定性的两阶段鲁棒优化（2RO）极具挑战性，往往导致精确求解的计算代价难以承受。场景缩减通过选择一个小而具有代表性的场景子集来缓解这一问题，从而使计算变得可行。然而，现有方法在很大程度上与具体问题无关，仅在不考虑可行域或追索结构的情况下对不确定性集合进行操作。本文提出了 PRISE，一种问题驱动的顺序前瞻启发式算法，通过评估每个场景的边际影响来构建缩减后的场景集合。尽管 PRISE 能够产生高质量的场景子集，但其每个选择步骤都需要求解多个子问题，使得该算法在大规模问题上的计算开销十分高昂。为解决这一问题，我们提出了 NeurPRISE，一个基于 GNN-Transformer 骨干网络的神经代理模型，它通过图卷积对单个场景的结构进行编码，并捕获跨场景之间的交互关系。

    arXiv:2605.14494v2 Announce Type: replace  Abstract: Two-Stage Robust Optimization (2RO) with discrete uncertainty is challenging, often rendering exact solutions prohibitive. Scenario reduction alleviates this issue by selecting a small, representative subset of scenarios to enable tractable computation. However, existing methods are largely problem-agnostic, operating solely on the uncertainty set without consulting the feasible region or recourse structure. In this paper, we introduce PRISE, a problem-driven sequential lookahead heuristic that constructs reduced scenario sets by evaluating the marginal impact of each scenario. While PRISE yields high-quality scenario subsets, each selection step requires solving multiple subproblems, making it computationally expensive at scale. To address this, we propose NeurPRISE, a neural surrogate model built on a GNN-Transformer backbone that encodes the per-scenario structure via graph convolution and captures cross-scenario interactions thro
    
[^325]: CSI-JEPA：以最小监督迈向普适Wi-Fi感知的基础表征

    CSI-JEPA: Towards Foundation Representations for Ubiquitous Wi-Fi Sensing with Minimal Supervision

    [https://arxiv.org/abs/2605.14171](https://arxiv.org/abs/2605.14171)

    CSI-JEPA是一个自监督预测表征学习框架，通过时间-子载波联合分词和信道变化感知掩码策略，从未标注CSI中学习可迁移的Wi-Fi感知表征，大幅降低对标注数据的依赖。

    

    信道状态信息（CSI）为人类和环境感知提供了一种广泛可用的感知模态，但现有的CSI感知模型通常依赖任务特定的监督训练，且针对每个任务、设备、用户或环境都需要大量标注数据。这限制了它们在实际部署中的可扩展性，因为在实际场景中未标注的CSI数据非常丰富，而标注数据的收集成本高昂。本文提出了CSI-JEPA，一个面向标签高效且可迁移的Wi-Fi感知的自监督预测表征学习框架。CSI-JEPA通过从可见上下文预测被掩码信道区域的潜在特征，从未标注的CSI样本中学习可复用的时频谱表征。为了更好地匹配CSI的物理结构，该方法沿时间和子载波两个维度联合对CSI进行分词，并引入了一种信道变化感知的掩码策略，优先选择具有更强局部信道动态性的区域……

    arXiv:2605.14171v2 Announce Type: replace  Abstract: Channel state information (CSI) provides a widely available sensing modality for human and environment perception, but existing CSI sensing models usually rely on task-specific supervised training and require substantial labeled data for each task, device, user, or environment. This limits their scalability in practical deployments where unlabeled CSI is abundant but labeled data is costly to collect. In this paper, we present CSI-JEPA, a self-supervised predictive representation learning framework for label-efficient and transferable Wi-Fi sensing. CSI-JEPA learns reusable temporal-spectral representations from unlabeled CSI samples by predicting latent features of masked channel regions from visible context. To better match the physical structure of CSI, it jointly tokenizes CSI along the time and subcarrier dimensions and introduces a channel variation-aware masking strategy that prioritizes regions with stronger local channel dyn
    
[^326]: 刻画视觉模型间的通用物体表征

    Characterizing universal object representations across vision models

    [https://arxiv.org/abs/2605.13675](https://arxiv.org/abs/2605.13675)

    通过将162个视觉模型的物体相似性结构分解为非负维度，研究发现通用维度更具可解释性且由概念性图像属性驱动，表明可解释性和语义内容是驱动模型间表征通用性的隐含关键因素。

    

    据报道，使用不同架构、目标和数据集训练的深度神经网络会收敛到相似的视觉表征。然而，目前尚不清楚模型究竟在哪些视觉属性上收敛，以及哪些因素可能是这种收敛的基础。为解决这一问题，我们将162个多样化视觉模型的物体相似性结构分解为一小组非负维度。为了区分通用维度与模型特定维度，我们随后估计了每个维度在模型中重现的频率。与模型特定维度相比，通用维度更具可解释性，且更强烈地受到概念性图像属性的驱动，这表明可解释性和语义内容是驱动模型间通用性的隐含因素。架构、目标函数、训练数据、模型大小和模型性能等方面的差异并不能解释通用维度的出现。

    arXiv:2605.13675v2 Announce Type: replace-cross  Abstract: Deep neural networks trained with different architectures, objectives, and datasets have been reported to converge on similar visual representations. However, what remains unknown is which visual properties models actually converge on and which factors may underlie this convergence. To address this, we decompose the object similarity structure of 162 diverse vision models into a small set of non-negative dimensions. To determine universal versus model-specific dimensions, we then estimate how often each dimension reappears across models. In contrast to model-specific dimensions, universal dimensions are more interpretable and more strongly driven by conceptual image properties, indicating the relevance of interpretability and semantic content as implicit factors driving universality across models. Differences in architecture, objective function, training data, model size, and model performance do not explain the emergence of un
    
[^327]: 涌现性失调中的人格模型坍缩

    Persona-Model Collapse in Emergent Misalignment

    [https://arxiv.org/abs/2605.12850](https://arxiv.org/abs/2605.12850)

    该论文提出“人格模型坍缩”假说，指出有害数据微调引发的涌现性失调源于模型内部人格表征的退化，并在四个模型家族上通过问卷实验证实有害微调使人格表达更极化、更二元化且内部一致性降低。

    

    在包含有害内容的狭窄数据上对大型语言模型进行微调，会使模型在与该数据无关的提示上也表现出广泛的对齐失调，这种现象被称为“涌现性失调”。我们提出，涌现性失调涉及“人格模型坍缩”：即编码和区分人格的内部表征发生退化，从而破坏了人格表达的细腻性与一致性。我们首先从行为层面检验这一假设：比较在有害数据上微调的模型与良性对照模型在扮演多样化人格时对问卷的回答。在四个模型家族中，有害微调产生了更极化、“非全即无”的回答，表明人格表达变得更加二元化、细腻程度更低。直方图显示回答在量表两端更为集中，并通过库尔贝克—莱布勒（KL）散度对这种分布偏移进行了量化。我们还发现人格内部一致性有所降低：对同一问题的重复回答……（摘要原文在此处截断）

    arXiv:2605.12850v3 Announce Type: replace-cross  Abstract: Fine-tuning large language models on narrow data with harmful content produces broadly misaligned behavior on unrelated prompts, a phenomenon known as emergent misalignment. We propose that emergent misalignment involves persona-model collapse: a deterioration of the internal representations that encode and distinguish personas, undermining the nuance and consistency of their expression. We first test this hypothesis behaviorally, by comparing how models fine-tuned on harmful data and benign controls answer questionnaires while role-playing diverse personas. Across four model families, harmful fine-tuning produces more polarized, all-or-nothing answers, suggesting more binary, less nuanced persona expression. Histograms show increased concentration at the scale endpoints, and Kullback--Leibler (KL) divergence quantifies the distributional shift. We also find reduced within-persona consistency: repeated answers to the same quest
    
[^328]: ELF：嵌入式语言流

    ELF: Embedded Language Flows

    [https://arxiv.org/abs/2605.10938](https://arxiv.org/abs/2605.10938)

    提出在连续嵌入空间中基于流匹配的扩散语言模型 ELF，其直到最终时间步才映射到离散词元，从而让图像扩散领域的成熟技术（如 CFG）可直接应用于语言建模。

    

    扩散模型和基于流的模型已成为生成连续数据（如图像和视频领域）的事实标准方法。它们的成功激发了人们将其应用于语言建模的日益浓厚的兴趣。与图像领域的对应模型不同，当今主流的扩散语言模型（DLM）主要在离散词元上运行。在本文中，我们证明了连续扩散语言模型只需对离散域进行极少适配即可变得有效。我们提出了嵌入式语言流，这是一类在连续嵌入空间中基于连续时间流匹配的扩散模型。与现有的扩散语言模型不同，ELF 在绝大部分生成过程中保持在连续嵌入空间中，直到最后一个时间步才通过一个共享权重网络映射到离散词元。这种设计使得图像领域扩散模型中成熟的技术（例如无分类器引导 CFG）能够直接移植到语言建模中。

    arXiv:2605.10938v3 Announce Type: replace-cross  Abstract: Diffusion and flow-based models have become the de facto approaches for generating continuous data, e.g., in domains such as images and videos. Their success has attracted growing interest in applying them to language modeling. Unlike their image-domain counterparts, today's leading diffusion language models (DLMs) primarily operate over discrete tokens. In this paper, we show that continuous DLMs can be made effective with minimal adaptation to the discrete domain. We propose Embedded Language Flows (ELF), a class of diffusion models in continuous embedding space based on continuous-time Flow Matching. Unlike existing DLMs, ELF predominantly stays within the continuous embedding space until the final time step, where it maps to discrete tokens using a shared-weight network. This formulation makes it straightforward to adapt established techniques from image-domain diffusion models, e.g., classifier-free guidance (CFG). Experim
    
[^329]: 多项式神经网络的最小填充架构

    Minimal Filling Architectures for Polynomial Neural Networks

    [https://arxiv.org/abs/2605.09609](https://arxiv.org/abs/2605.09609)

    本文推翻了“所有最小填充架构的隐藏层宽度均为单峰”的猜想，通过前沿搜索和有限域计算构造并验证了反例，同时证明了深度不超过三的最小填充架构必为单峰，而深度为四的非单峰反例表明该界限是最优的。

    

    最小填充架构描述的是使多项式神经网络的神经簇充满其周围函数空间所需的最小隐藏层宽度。此前有猜想认为，每一个这样的架构都具有单峰的隐藏层宽度。我们利用前沿搜索方法构造了该猜想的反例，并通过有限域雅可比行列式计算和递归维数界对主要反例进行了验证。在相反方向上，我们证明了所有深度不超过三的最小填充架构都是单峰的。一个深度为四的非单峰最小填充架构表明该深度界限是紧的。

    arXiv:2605.09609v3 Announce Type: replace  Abstract: Minimal filling architectures describe the smallest hidden-layer widths for which the neurovariety of a polynomial neural network fills its ambient function space. It was conjectured that every such architecture has unimodal hidden widths. We construct counterexamples to this conjecture using a frontier search and certify the main example through finite-field Jacobian computations and recursive dimension bounds. In the opposite direction, we prove that every minimal filling architecture of depth at most three is unimodal. A nonunimodal minimal filling architecture of depth four shows that the depth bound is sharp.
    
[^330]: 单策略集中性下具有前向KL正则化的离线上下文老虎机的快速收敛率

    Fast Rates for Offline Contextual Bandits with Forward-KL Regularization under Single-Policy Concentrability

    [https://arxiv.org/abs/2605.09214](https://arxiv.org/abs/2605.09214)

    本文首次在单策略集中性条件下，为前向KL正则化的离线上下文老虎机给出了表格型与一般函数逼近设置下的 Õ(ε^{-1}) 快速收敛率上界，通过新颖地运用悲观原理的凸分析框架填补了前向KL正则化统计分析的空白。

    

    KL（Kullback-Leibler）散度正则化以反向KL或前向KL的形式广泛存在于强化学习算法中。最近的研究表明，在反向KL正则化下的决策问题可以达到 ε^{-1} 型的快速收敛率，相比之下标准的样本复杂度为 ε^{-2} 型。然而，对于前向KL正则化的目标函数，现有的统计分析要么不适用，要么只能得到 Õ(ε^{-2}) 的慢收敛率。我们通过对前向KL正则化离线上下文老虎机的精简分析，迈出了解决这一问题的第一步。我们在表格型设置和一般函数逼近设置下，均在单策略集中性的概念下，首次给出了 Õ(ε^{-1}) 的上界。特别地，我们的凸分析流程以一种新颖的方式利用悲观原理统一了这些设置，并完全绕过了证明中的……（原文摘要在此处截断）

    arXiv:2605.09214v2 Announce Type: replace-cross  Abstract: \emph{Kullback-Leibler} (KL) regularization is ubiquitous in reinforcement learning algorithms in the form of \emph{reverse} or \emph{forward} KL. Recent studies have demonstrated $\epsilon^{-1}$-type fast rates for decision making under reverse KL regularization, in contrast to the standard $\epsilon^{-2}$-type sample complexity. However, for forward-KL-regularized objectives, existing statistical analyses are either not applicable or result in $\tilde{O}(\epsilon^{-2})$ slow rates. We take the first step towards addressing this problem via a streamlined analysis of forward-KL-regularized offline CBs. We give the first $\tilde{O}(\epsilon^{-1})$ upper bounds in tabular and general function approximation settings, both under notions of \emph{single-policy concentrability}. In particular, our convex-analytical pipeline unifies these settings by exploiting the pessimism principle in a novel way and completely bypasses the proof r
    
[^331]: 无噪声逆向优化的紧致泛化界

    Tight Generalization Bounds for Noiseless Inverse Optimization

    [https://arxiv.org/abs/2605.08866](https://arxiv.org/abs/2605.08866)

    本文为无噪声逆向优化给出了紧致的高概率 O(d/T) 泛化界，在动作唯一性条件下将其与老虎机最优臂识别结果对齐，并证明瞬时与累积遗憾的下界与对抗性设置的上界相匹配，从而表明该速率不可再改进。

    

    逆向优化（IO）旨在从观测到的上下文—动作数据中推断决策者目标函数的参数。我们研究无噪声的逆向优化，即演示数据由真实目标函数生成。我们为诱导动作集提供了高概率的 O(d/T) 泛化界，其中 d 是未知参数的数量，T 是训练数据集的规模。在保证所选动作唯一性的附加条件下，我们进一步加强了这一保证，使我们的 IO 保证与老虎机文献中最优臂识别的结果相一致。我们进一步证明，O(d/T) 的速率对于本文所考虑的所有一致估计量都是紧的，并将该结果扩展到瞬时遗憾和累积遗憾两种情形。值得注意的是，由此得到的遗憾下界与对抗性设置中相应的上界相匹配，这表明随机 IO 设置实际上在本质上等同于……

    arXiv:2605.08866v2 Announce Type: replace-cross  Abstract: Inverse optimization (IO) seeks to infer the parameters of a decision-maker's objective from observed context--action data. We study noiseless IO, where demonstrations are generated by a ground-truth objective. We provide a high-probability ${O}(\frac{d}{T})$ generalization bound for the induced action set, where $d$ is the number of unknown parameters and $T$ is the size of the training dataset. We strengthen this guarantee under additional conditions that ensure uniqueness of the chosen action, bringing our IO guarantees in line with best-arm identification results in the bandit literature. We further show that the ${O}(\frac{d}{T})$ rate is tight over all consistent estimators considered here, and extend the result to both instantaneous and cumulative regret. Notably, the resulting regret lower bound matches the corresponding upper bounds in the adversarial setting, indicating that the stochastic IO setting is effectively ad
    
[^332]: FILO：低秩自适应的统一Fisher几何框架

    FILO: A Unified Fisher-Geometric Framework for Low-Rank Adaptation

    [https://arxiv.org/abs/2605.08734](https://arxiv.org/abs/2605.08734)

    本文提出FILO，一个统一的Fisher几何框架，无需对奇异算子直接求逆即可实现LoRA的因子空间更新，并据此将现有LoRA优化器系统地划分为四大类。

    

    低秩自适应（LoRA）将权重矩阵 $W$ 重参数化为两个低秩因子的乘积，但将因子映射到权重矩阵的生成器的雅可比矩阵 $J_G$ 是秩亏的，因此由任何 $W$ 空间预条件子 $F_t$ 诱导的因子空间预条件子 $J_G^* F_t J_G$ 是奇异的，进而导致标准链式法则无法被唯一地求逆，无法将预条件后的 $W$ 空间方向映射回因子空间的更新。为此，我们提出了 FILO，一个用于低秩自适应的统一 Fisher 几何框架，它无需直接对该奇异算子求逆即可实现因子更新，并根据因子空间求逆策略和 $W$ 空间预条件子 $F_t$ 对现有 LoRA 优化器进行分类。现有方法沿着这两个维度划分为四大类：因子空间自适应更新、$J_G^* J_G$ 的块对角替代方法、Frobenius 残差伪逆方法以及黎曼流形方法。

    arXiv:2605.08734v2 Announce Type: replace-cross  Abstract: Low-Rank Adaptation (LoRA) reparameterizes a weight matrix $W$ as a product of two low-rank factors, but the Jacobian $J_G$ of the generator mapping the factors to the weight matrix is rank-deficient, so the factor-space preconditioner $J_G^* F_t J_G$ induced by any $W$-space preconditioner $F_t$ is singular, and consequently the standard chain rule cannot be uniquely inverted to map a preconditioned $W$-space direction back to a factor-space update. Therefore, we introduce FILO, a unified Fisher-geometric framework for low-rank adaptation that enables factor updates without directly inverting this singular operator and casts existing LoRA optimizers by their factor-space inversion strategy and $W$-space preconditioner $F_t$. Existing methods occupy four families along these axes: factor-space adaptive updates, block-diagonal surrogates for $J_G^* J_G$, Frobenius-residual pseudoinverse methods, and Riemannian manifold methods. 
    
[^333]: 基于泊松-莫罗漂移的随机逼近与强化学习的几乎必然收敛速率

    Almost Sure Convergence Rates of Stochastic Approximation and Reinforcement Learning via a Poisson-Moreau Drift

    [https://arxiv.org/abs/2605.07104](https://arxiv.org/abs/2605.07104)

    本文通过泊松-莫罗漂移技术，为马尔可夫噪声下期望更新具有压缩性的随机逼近算法（如Q学习和线性时序差分学习）建立了几乎必然收敛速率，在幂律学习率下获得了可任意接近o(n^(1-2η))的速率，在调和学习率下当参数满足条件时平方误差指数可任意接近1。

    

    在马尔可夫噪声下建立随机逼近和强化学习的几乎必然收敛速率是一个根本性的理论挑战。我们为一类期望更新具有压缩性质的随机逼近算法在这一挑战上取得了进展，该设置出现在许多强化学习算法中，例如Q学习和线性时序差分学习。具体而言，对于幂律学习率O(n^(-η))（其中η ∈ (1/2, 1)），我们获得了可任意接近o(n^(1-2η))的几乎必然收敛速率。对于调和学习率α_n=α/(n+1)，对于每个ζ<min{1,2(1-κ)α}，几乎必然平方误差速率为o(n^(-ζ))，其中κ为压缩因子。因此，当2(1-κ)α≥1时，可获得任意接近1的指数。这接近了经典迭代对数定律（原文截断）……的n^(-1)多项式指数。

    arXiv:2605.07104v3 Announce Type: replace  Abstract: Establishing almost sure convergence rates for stochastic approximation and reinforcement learning under Markovian noise is a fundamental theoretical challenge. We make progress towards this challenge for a class of stochastic approximation algorithms whose expected updates are contractive, a setting that arises in many reinforcement learning algorithms such as $Q$-learning and linear temporal difference learning. Specifically, for a power-law learning rate $O(n^{-\eta})$ with $\eta \in (1/2, 1)$, we obtain an almost sure convergence rate arbitrarily close to $o(n^{1 - 2\eta})$. For the harmonic learning rate $\alpha_n=\alpha/(n+1)$, the almost sure squared-error rate is $o(n^{-\zeta})$ for every $\zeta<\min\{1,2(1-\kappa)\alpha\}$, where $\kappa$ is the contraction factor. Thus exponents arbitrarily close to one are available when $2(1-\kappa)\alpha\geq1$. This approaches the $n^{-1}$ polynomial exponent of classical law-of-the-iter
    
[^334]: UniSD：面向大语言模型的统一自蒸馏框架

    UniSD: Towards a Unified Self-Distillation Framework for Large Language Models

    [https://arxiv.org/abs/2605.06597](https://arxiv.org/abs/2605.06597)

    提出了统一自蒸馏框架UniSD，通过整合多教师一致性、EMA教师稳定化、词元级对比学习、特征匹配和散度裁剪等互补机制，系统性地解决了大语言模型自蒸馏中监督可靠性、表示对齐与训练稳定性的核心难题。

    

    自蒸馏（SD）为在不依赖更强外部教师模型的情况下适配大语言模型（LLM）提供了一条有前景的路径。然而，自回归LLM中的自蒸馏仍然充满挑战，因为自生成的轨迹是自由形式的，正确性依赖于具体任务，且看似合理的推理过程仍可能提供不稳定或不可靠的监督信号。现有方法主要考察孤立的设计选择，导致其有效性、作用及相互关系尚不明确。本文提出了UniSD，一个用于系统研究自蒸馏的统一框架。UniSD整合了解决监督可靠性、表示对齐和训练稳定性问题的互补机制，包括多教师一致性、EMA教师稳定化、词元级对比学习、特征匹配和散度裁剪。在来自三个模型系列的六个基准测试和六个模型上，UniSD揭示了自蒸馏何时有效（摘要在此处被截断）。

    arXiv:2605.06597v3 Announce Type: replace-cross  Abstract: Self-distillation (SD) offers a promising path for adapting large language models (LLMs) without relying on stronger external teachers. However, SD in autoregressive LLMs remains challenging because self-generated trajectories are free-form, correctness is task-dependent, and plausible rationales can still provide unstable or unreliable supervision. Existing methods mainly examine isolated design choices, leaving their effectiveness, roles, and interactions unclear. In this paper, we propose UniSD, a unified framework to systematically study self-distillation. UniSD integrates complementary mechanisms that address supervision reliability, representation alignment, and training stability, including multi-teacher agreement, EMA teacher stabilization, token-level contrastive learning, feature matching, and divergence clipping. Across six benchmarks and six models from three model families, UniSD reveals when self-distillation impr
    
[^335]: 语言模型中的不变特征：几何刻画与模型归因

    Invariant Features in Language Models: Geometric Characterization and Model Attribution

    [https://arxiv.org/abs/2605.06458](https://arxiv.org/abs/2605.06458)

    该论文提出一个局部几何框架来刻画语言模型中的不变特征，通过对比子空间发现方法分离语义改变型与语义保持型的表示变化，并将所得的不变表示成功应用于零样本模型归因任务。

    

    语言模型对改写表现出很强的鲁棒性，这表明语义信息可能通过稳定的内部表示进行编码，但这种不变性的结构和来源仍不清楚。我们提出了一个局部几何框架，其中语义等价的输入所产生的表示占据潜空间中的结构化区域，改写引起的变化沿着干扰方向进行，而语义同一性则在不变子空间中得以保持。基于这一观点，我们做出了三项贡献：(1) 对不变潜在特征的几何刻画，(2) 一种对比子空间发现方法，能够将改变语义的变化与保持语义的变化分离开来，(3) 将不变表示应用于零样本模型归因。跨模型和跨层的实证结果支持了这些贡献。不变结构在特定的深度区域中涌现，语义位移……（原文摘要不完整）

    arXiv:2605.06458v2 Announce Type: replace-cross  Abstract: Language models exhibit strong robustness to paraphrasing, suggesting that semantic information may be encoded through stable internal representations, yet the structure and origin of such invariance remain unclear. We propose a local geometric framework in which semantically equivalent inputs produce representations that occupy structured regions in latent space, with paraphrastic variation along nuisance directions and semantic identity preserved in invariant subspaces. Building on this view, we make three contributions: (1) a geometric characterization of invariant latent features, (2) a contrastive subspace discovery method that separates semantic-changing from semantic-preserving variation, and (3) an application of invariant representations to zero-shot model attribution. Across models and layers, empirical results support these contributions. Invariant structure emerges in specific depth regions, semantic displacement li
    
[^336]: 驯服熵悬崖：面向自回归视觉生成的可变码本大小量化

    Taming the Entropy Cliff: Variable Codebook Size Quantization for Autoregressive Visual Generation

    [https://arxiv.org/abs/2605.06207](https://arxiv.org/abs/2605.06207)

    该论文发现离散视觉分词器中存在条件熵沿序列急剧衰减的“熵悬崖”现象，并提出可变码本大小量化（VCQ）方法，以解决恒定码本快速耗尽有限数据熵预算的问题。

    

    大多数离散视觉分词器依赖于一个默认设计：序列中的每个位置共享同一个码本。这种恒定码本设计会非常快速地消耗有限数据的熵预算。我们观察到，每个位置的经验条件熵沿着序列衰减得如此之快，以至于在几个位置之后，条件分布基本上变为确定性分布。在ImageNet上（K=16384）时，这种情况在256个位置中仅约2个位置之后就会发生，此时大多数前缀已经能够识别数据集中的单个图像。我们将这种现象称为“熵悬崖”（Entropy Cliff）。在条件被充分利用的情况下，较大的码本会在更少的位置耗尽有限数据的熵预算。在我们的语言对比实验中，经验熵的下降则更为平缓，每个位置的有效熵保持在码本容量以下。为解决这一问题，我们提出了可变码本大小量化（VCQ），其中码本……

    arXiv:2605.06207v2 Announce Type: replace-cross  Abstract: Most discrete visual tokenizers rely on a default design: every position in the sequence shares the same codebook. Such a constant-codebook design can consume the finite-data entropy budget very quickly. We observe that the per-position empirical conditional entropy decays so quickly along the sequence that, after a few positions, the conditional distribution becomes essentially deterministic. On ImageNet with $K=16384$, this happens after only about 2 out of 256 positions, when most prefixes already identify individual images in the dataset. We call this phenomenon the Entropy Cliff. Under full conditional utilization, larger codebooks exhaust the finite-data entropy budget in fewer positions. The empirical entropy decreases more gradually in our language comparison, where the effective entropy per position stays below the codebook capacity. To address this, we propose Variable Codebook Size Quantization (VCQ), where the codeb
    
[^337]: 顿悟还是故障？低精度如何驱动“弹弓机制”式损失尖峰

    Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes

    [https://arxiv.org/abs/2605.06152](https://arxiv.org/abs/2605.06152)

    本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。

    

    深度神经网络在无正则化的长期训练过程中会表现出周期性的损失尖峰，这一现象被称为“弹弓机制”。现有工作通常将其归因于内在的优化动力学，但其触发机制仍不清楚。本文证明该现象是浮点算术精度极限的结果：当训练进入高置信度阶段后，正确类别 logit 与其他 logit 之间的差值可能超过吸收误差阈值。于是在反向传播过程中，正确类别的梯度被精确舍入为零，而错误类别的梯度仍保持非零。这打破了跨类别梯度的零和约束，并在分类器层的参数更新中引入了系统性漂移。我们证明该漂移与特征之间形成了正反馈回路，导致全局分类器均值与全局特征（摘要在此处截断）。

    arXiv:2605.06152v4 Announce Type: replace-cross  Abstract: Deep neural networks exhibit periodic loss spikes during unregularized long-term training, a phenomenon known as the "Slingshot Mechanism." Existing work usually attributes this to intrinsic optimization dynamics, but its triggering mechanism remains unclear. This paper proves that this phenomenon is a result of floating-point arithmetic precision limits. As training enters a high-confidence stage, the difference between the correct-class logit and the other logits may exceed the absorption-error threshold. Then during backpropagation, the gradient of the correct class is rounded exactly to zero, while the gradients of the incorrect classes remain nonzero. This breaks the zero-sum constraint of gradients across classes and introduces a systematic drift in the parameter update of the classifier layer. We prove that this drift forms a positive feedback loop with the feature, causing the global classifier mean and the global featu
    
[^338]: 针对风格自适应和弦生成的流行与爵士数据混合比例实证研究

    Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation

    [https://arxiv.org/abs/2605.04998](https://arxiv.org/abs/2605.04998)

    本研究通过对流行预训练Music Transformer进行爵士微调并系统扫描混合的流行“排练”数据量，实证发现排练数据可将爵士top-1提升7至9个百分点，同时将流行风格的遗忘从2.11个百分点降至最多1.19个百分点，为小数据新风格的适应提供了混合比例依据。

    

    本文将和弦进行生成视为一项独立任务，探讨在微调过程中需要保留多少旧领域数据，才能在习得新风格的同时不遗忘旧风格。作者对一个以流行音乐预训练的2500万参数Music Transformer（在留出的流行测试集上top-1下一token准确率为84.21%）进行爵士乐微调研究。由于爵士语料库比流行语料库小两个数量级以上，每次实验均使用全部1,513条爵士训练序列，而所扫描的变量是与其混合的流行“排练”数据量：0、1K、2.5K、5K或10K。在表中所列的最佳epoch处，每个微调后的模型在爵士top-1上提升了7至9个百分点（发布的F1检查点与Phase 0基线持平，如论文的更正说明所解释）；在仅用爵士数据微调时，流行top-1下降了2.11个百分点，而使用排练数据时下降幅度最多为1.19个百分点。Top-1指标还会对每个序列的三个头部token进行评分，其中两个token会改变整个运行结果：jazz-maj……

    arXiv:2605.04998v4 Announce Type: replace-cross  Abstract: This paper treats chord progression generation as a standalone task and asks how much old-domain data must be retained in fine-tuning to acquire a new genre without forgetting the old. I study jazz fine-tuning of a pop-pretrained 25M-parameter Music Transformer (84.21% top-1 next-token accuracy on a held-out pop test set). The jazz corpus is over two orders of magnitude smaller than the pop corpus, so every run uses all 1,513 jazz training sequences, and the swept variable is the volume of pop "rehearsal" data mixed alongside: 0, 1K, 2.5K, 5K, or 10K. At the tabulated best epochs every fine-tuned run gains 7 to 9 points of jazz top-1 (the released F1 checkpoint equals the Phase 0 baseline, as the paper's Correction explains), and pop top-1 falls by 2.11 points under jazz-only fine-tuning but by at most 1.19 with rehearsal. Top-1 also scores the three header tokens of each sequence, and two of them shift whole runs: the jazz-maj
    
[^339]: 面向数据受限训练的规范性缩放定律

    Prescriptive Scaling Laws for Data Constrained Training

    [https://arxiv.org/abs/2605.01640](https://arxiv.org/abs/2605.01640)

    该论文提出了一种带加性过拟合惩罚项的规范性缩放定律，能准确刻画数据受限训练中重复数据的影响，并给出新的计算最优分配建议——当数据重复超过一定限度后，算力应投向扩大模型容量而非继续重复训练。

    

    训练所需的计算量正日益超过高质量数据的可用量，这使得核心挑战从最优计算分配转变为如何从有限数据中提取最大价值。被广泛采用的 Chinchilla 缩放定律假设每个训练 token 都是唯一的，这限制了它在数据受限场景下指导预训练决策的能力。我们用一个简单的加性过拟合惩罚项对重复训练下的超额损失进行建模，发现它能准确描述模型行为。我们的缩放定律给出了性质上全新的计算最优分配建议：超过某个点后，进一步的重复训练会适得其反，此时计算资源最好用于提升模型容量。我们证明，遵循该定律推荐的配置可以改善数据受限场景下的模型性能。最后，由于我们的单参数形式将过拟合隔离在单个系数中，它使得不同训练配置之间能够进行直接比较。

    arXiv:2605.01640v2 Announce Type: replace-cross  Abstract: Training compute is increasingly outpacing the availability of high-quality data. This shifts the central challenge from optimal compute allocation to extracting maximum value from limited data. The widely adopted Chinchilla scaling law assumes every training token is unique. This limits its ability to guide pretraining decisions in data-constrained regimes. We model the excess loss under repetition with a simple additive overfitting penalty and find that it accurately describes model behavior. Our scaling law yields qualitatively new compute-optimal allocation advice. Beyond a point, further repetition is counterproductive and compute is better spent on model capacity. We show that following our law's recommended configuration improves performance in data-constrained regimes. Finally, because our one-parameter form isolates overfitting in a single coefficient, it enables direct comparison across training configurations. As a c
    
[^340]: 用于形状约束学习与最优传输的超输入凸神经网络

    Hyper Input Convex Neural Networks for Shape Constrained Learning and Optimal Transport

    [https://arxiv.org/abs/2604.26942](https://arxiv.org/abs/2604.26942)

    HyCNN 结合 Maxout 网络与输入凸神经网络，以指数级更少的参数实现高效凸函数学习，在凸回归、插值以及高维最优传输任务中显著优于现有方法。

    

    我们提出了超输入凸神经网络，这是一种为学习凸函数而设计的新型神经网络架构。HyCNN 将 Maxout 网络的原理与输入凸神经网络（ICNN）相结合，构建出一种在输入上始终保持凸性的神经网络，理论上能够利用网络深度，并且与 ICNN 相比在大规模训练时表现更加可靠。具体而言，我们证明了在给定精度下逼近二次函数时，HyCNN 所需的参数数量比 ICNN 呈指数级减少。通过一系列合成实验，我们证明了 HyCNN 在凸回归和插值任务的预测性能上优于现有的 ICNN 和多层感知机（MLP）。我们进一步将 HyCNN 应用于学习合成示例和单细胞数据的高维最优传输映射，在这些任务中，它们通常优于基于 ICNN 的神经最优传输方法以及其他基线方法。

    arXiv:2604.26942v2 Announce Type: replace  Abstract: We introduce Hyper Input Convex Neural Networks (HyCNNs), a novel neural network architecture designed for learning convex functions. HyCNNs combine the principles of Maxout networks with input convex neural networks (ICNNs) to create a neural network that is always convex in the input, theoretically capable of leveraging depth, and performs reliably when trained at scale compared to ICNNs. Concretely, we prove that HyCNNs require exponentially fewer parameters than ICNNs to approximate quadratic functions up to a given precision. Throughout a series of synthetic experiments, we demonstrate that HyCNNs outperform existing ICNNs and MLPs in terms of predictive performance for convex regression and interpolation tasks. We further apply HyCNNs to learn high-dimensional optimal transport maps for synthetic examples and for single-cell data, where they oftentimes outperform ICNN-based neural optimal transport methods and other baselines a
    
[^341]: 从多个思考者那里学习思考：思维链异质性如何影响高效可学习性

    Learning to Think from Multiple Thinkers: How Chain-of-Thought Heterogeneity Affects Efficient Learnability

    [https://arxiv.org/abs/2604.24737](https://arxiv.org/abs/2604.24737)

    该论文首次从学习理论视角研究异构思维链监督问题，提出多种由不同来源（最终答案相同但中间步骤各异）提供CoT数据的异构监督模型，并探讨在单一同构来源可被高效学习的前提下，何时能够从异构监督数据中实现高效学习。

    

    思维链监督通过暴露易于通过下一词预测学习的中间步骤，使计算上困难的学习问题变得可解。然而在实践中，CoT数据集来自具有不同推理风格的异构来源，并且需要投入大量精力来策划适合下一词预测训练的监督数据。这种数据策划在使CoT监督的计算优势得以实现方面扮演着什么角色？在这项工作中，我们从学习理论的角度迈出了理解这一问题的第一步。我们引入了来自多个来源的异构CoT监督的几种模型，其中每个来源都提供具有相同最终答案但中间步骤不同的正确推理轨迹。假设将下一词预测器拟合到来自任何单一来源的同构轨迹在计算上是可处理的，我们探讨这在何种条件下足以支持从异构监督中进行高效学习。

    arXiv:2604.24737v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) supervision makes computationally hard learning problems tractable by exposing intermediate steps that are easy to learn via next-token prediction. However, in practice, CoT datasets draw on heterogeneous sources with different reasoning styles, and substantial effort goes into curating supervision suitable for next-token training. What role does this curation play in enabling the computational advantage of CoT supervision? In this work, we take a first step toward understanding this question from a learning-theoretic perspective. We introduce several models of $heterogeneous$ $CoT$ $supervision$ from multiple sources, each providing correct traces with the same final answer but different intermediate steps. Assuming that fitting a next-token predictor to $homogeneous$ traces from any single source is computationally tractable, we ask when this suffices for efficient learning from heterogeneous supervisio
    
[^342]: AAC：面向ALT的架构级可采纳可微分地标压缩

    AAC: Admissible-by-Architecture Differentiable Landmark Compression for ALT

    [https://arxiv.org/abs/2604.20744](https://arxiv.org/abs/2604.20744)

    本文提出AAC，在固定标签内存预算下通过两种可采纳性保持的连续松弛实现可微分的ALT地标选择压缩，并给出了带有1-1/e极限因子的近似保证。

    

    我们提出了架构可采纳压缩器（AAC），它在固定的标签内存预算下，使基于地标的A*启发式方法能够适应查询需求。AAC为ALT（A*、地标与三角不等式）选择提供了两种连续松弛形式。在地标距离精确的前提下，每一个参数取值都定义了最短路径距离的一个下界。两种松弛形式都仅使用被选中的标签来部署ALT。行随机松弛形式采用地标距离的凸组合。我们刻画了保持可采纳性的最大线性类，并证明了松弛最优解与离散最优解相一致。分数松弛形式混合了子集启发式，并支持带有精确梯度的连续贪心算法（AAC-CG），以及用于可微分选择的熵正则化预言机。对于期望启发式值，我们证明了包含有限步数项、预言机误差项和有限样本项的近似保证。极限情形下的 $1-1/e$ 因子（摘要原文在此截断）

    arXiv:2604.20744v2 Announce Type: replace  Abstract: We introduce the Architecturally Admissible Compressor (AAC), which adapts landmark-based A* heuristics to query demand under a fixed label-memory budget. AAC provides two continuous relaxations of ALT (A*, Landmarks, and Triangle inequality) selection. With exact landmark distances, every parameter value defines a lower bound on shortest-path distance. Both relaxations deploy ALT using only the selected labels. The row-stochastic relaxation takes convex combinations of landmark distances. We characterize the maximal admissibility-preserving linear classes and prove that the relaxed and discrete optima coincide. The fractional relaxation mixes subset heuristics and supports continuous greedy (AAC-CG) with exact gradients and an entropy-regularized oracle for differentiable selection. For expected heuristic value, we prove an approximation guarantee with finite-step, oracle-error, and finite-sample terms. The limiting $1-1/e$ factor i
    
[^343]: 流形上的推理：扩散语言模型中用于自我验证的双向一致性

    Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models

    [https://arxiv.org/abs/2604.16565](https://arxiv.org/abs/2604.16565)

    该论文提出了一种无需训练、无监督的双向流形一致性（BMC）度量方法，通过前向掩码与后向重建循环量化生成序列的几何稳定性，从而在无标准答案的情况下验证扩散语言模型推理的有效性，并支持诊断与拒绝重采样等应用。

    

    尽管扩散大语言模型（dLLMs）在全局规划方面具有结构优势，但如何高效地验证它们是否通过有效的推理轨迹得出正确答案，仍然是一个关键挑战。在这项工作中，我们提出了一个几何视角：流形上的推理。我们假设有效的生成轨迹作为稳定吸引子驻留在学习分布的高密度流形上，而无效路径则表现出偏离流形的漂移。为了将这一假设付诸实践，我们引入了双向流形一致性（BMC），这是一种无需训练、无监督的度量方法，通过前向掩码与后向重建的循环来量化生成序列的稳定性。在实证方面，我们展示了BMC在整个推理生命周期中的多功能性：（1）在诊断中，它可以在没有标准答案的情况下作为解有效性的稳健判别器；（2）在推理中，它支持拒绝重采样机制，从而提升生成质量。

    arXiv:2604.16565v4 Announce Type: replace-cross  Abstract: While Diffusion Large Language Models (dLLMs) offer structural advantages for global planning, efficiently verifying that they arrive at correct answers via valid reasoning traces remains a critical challenge. In this work, we propose a geometric perspective: Reasoning on the Manifold. We hypothesize that valid generation trajectories reside as stable attractors on the high-density manifold of the learned distribution, whereas invalid paths exhibit off-manifold drift. To operationalize this, we introduce Bidirectional Manifold Consistency (BMC), a training-free, unsupervised metric that quantifies the stability of the generated sequence through a forward-masking and backward-reconstruction cycle. Empirically, we demonstrate BMC's versatility across the full reasoning lifecycle: (1) in Diagnosis, it serves as a robust discriminator of solution validity without ground truth answer; (2) in Inference, it enables rejection resamplin
    
[^344]: 循环Transformer中的稳定性与泛化

    Stability and Generalization in Looped Transformers

    [https://arxiv.org/abs/2604.15259](https://arxiv.org/abs/2604.15259)

    该论文提出基于不动点的三维度稳定性分析框架，证明带外部归一化召回机制的循环Transformer能同时实现不动点的局部吸引性、输入平滑性与稳定的反向传播，从而在测试时对更难的问题实现外推而非记忆。

    

    循环Transformer通过在更难的问题上投入更多迭代次数，有望实现测试时的计算扩展，但目前仍不清楚哪些架构选择能让它们在测试时外推到更难的问题，而非死记硬背训练中特定的解法。我们提出了一个基于不动点的分析框架，从局部吸引性、输入依赖性和几何结构这三个稳定性维度来分析循环架构，并用它刻画不动点迭代何时能产生有意义的预测。在理论上，我们证明了不带召回机制的循环网络具有可数个不动点，且在任何谱范围内都无法实现强输入依赖性；而带外部归一化的召回机制则能可靠地产生一种状态，使得不动点同时具备局部吸引性、对输入的局部平滑性，并获得稳定反向传播的支持。在实证方面，我们在国际象棋、数独和前缀和任务上训练了小型循环Transformer（摘要在此处截断）。

    arXiv:2604.15259v3 Announce Type: replace-cross  Abstract: Looped transformers promise test-time compute scaling by spending more iterations on harder problems, but it remains unclear which architectural choices let them extrapolate to harder problems at test time rather than memorize training-specific solutions. We introduce a fixed-point based framework for analyzing looped architectures along three axes of stability -- local attractivity, input-dependence, and geometry -- and use it to characterize when fixed-point iteration yields meaningful predictions. Theoretically, we prove that looped networks without recall have countable fixed points and cannot achieve strong input-dependence at any spectral regime, while recall with outer normalization reliably produces a regime in which fixed points are simultaneously locally-attractive, locally smooth in the input, and supported by stable backpropagation. Empirically, we train small looped transformers on chess, sudoku, and prefix-sums an
    
[^345]: 前向-前向算法优良度函数设计中的选择性与形状

    Selectivity and Shape in the Design of Forward-Forward Goodness Functions

    [https://arxiv.org/abs/2604.13081](https://arxiv.org/abs/2604.13081)

    本文系统研究了前向-前向算法中优良度函数的选择性与形状设计，提出了一种无参数、尺度不变的“突发性优良度”目标函数（基于超额峰度、奖励重尾活动模式），在Fashion-MNIST上显著提升了准确率。

    

    前向-前向算法使用一个局部的“优良度函数”逐层训练网络，然而活动统计量的选择仍然是一个重要的设计问题。我们系统地研究了选择性与活动形状如何影响FF学习，引入了基于top-k和entmax加权能量的选择性优良度函数，以及基于归一化高阶矩的形状敏感目标函数。特别地，我们提出了突发性优良度，这是一种无参数、尺度不变的超额峰度目标函数，用于奖励重尾活动模式。我们的研究在六个小图像数据集上考察了优良度函数与激活函数及标签注入方式的组合效果，并通过受控实验来分离各自的影响。在经过验证集调优、五随机种子、匹配ReLU激活、标准标签注入和逐层评分的Fashion-MNIST实验中，突发性优良度将准确率从73.2±0.9%提高到（原文摘要在此处截断）。

    arXiv:2604.13081v3 Announce Type: replace-cross  Abstract: The Forward-Forward (FF) algorithm trains networks layer-by-layer using a local "goodness function," yet the choice of activity statistic remains an important design question. We systematically study how selectivity and activity shape affect FF learning, introducing selective goodness functions based on top-$k$ and entmax-weighted energy alongside shape-sensitive objectives based on normalized higher-order moments. In particular, we propose burstiness goodness, a parameter-free, scale-invariant excess-kurtosis objective that rewards heavy-tailed activity patterns. Our study examines goodness functions together with activation and label-injection choices across six small-image datasets, and uses controlled experiments to isolate their effects. In validation-tuned, five-seed Fashion-MNIST experiments with matched ReLU activation, standard label injection, and per-layer scoring, burstiness improves accuracy from $73.2\pm0.9\%$ to 
    
[^346]: Nexus：相同的预训练损失，通过共同极小值实现更好的下游泛化

    Nexus: Same Pretraining Loss, Better Downstream Generalization via Common Minima

    [https://arxiv.org/abs/2604.09258](https://arxiv.org/abs/2604.09258)

    论文提出Nexus优化器，通过在训练中最大化梯度相似性，促使模型收敛到各数据源的共同极小值点，从而在预训练损失相同的情况下显著提升下游泛化能力。

    

    大规模语言模型的基础能力是在互联网规模的、高度异构的数据混合上进行预训练时获得的。在本工作中，我们探究了一个关于预训练收敛状态的有趣几何问题：模型是收敛到跨越所有数据源的共同极小值点（如图2b所示），还是仅仅收敛到总损失的极小值点（如图2a所示）？我们假设各任务特定极小值点在几何上的“接近程度”与下游泛化能力存在内在联系。我们揭示出，标准优化器（如AdamW）通常收敛到各任务特定极小值点彼此相距较远的位置。为解决这一问题，我们提出了Nexus优化器，它通过在优化过程中最大化梯度相似性来促使这些极小值点彼此接近。在参数量从1.3亿到30亿的模型、多种数据混合方式以及不同超参数调度下的实验表明，Nexus显著提升了（下游性能）……

    arXiv:2604.09258v3 Announce Type: replace  Abstract: The foundational capabilities of large language models are acquired during pretraining on internet-scale, highly heterogeneous data mixtures. In this work, we investigate an interesting geometric question regarding the converged state of pretraining: Does the model converge to a common minimizer across all data sources (e.g., Fig. 2b), or merely a minimizer of the summed loss (e.g., Fig. 2a)? We hypothesize that the geometric "closeness" of task-specific minima is intrinsically linked to downstream generalization. We reveal that standard optimizers (e.g., AdamW) often converge to points where task-specific minima are distant from each other. To address this, we propose the Nexus optimizer, which encourages the closeness of these minima by maximizing gradient similarity during optimization. Experiments across models ranging from 130M to 3B parameters, various data mixtures and hyperparameter schedules, show that Nexus significantly bo
    
[^347]: 基于权重空间算术的零样本量化

    Zero-Shot Quantization via Weight-Space Arithmetic

    [https://arxiv.org/abs/2604.03420](https://arxiv.org/abs/2604.03420)

    该论文提出通过简单的权重空间算术从供体任务中提取“量化向量”来修补接收模型，从而在不进行量化感知训练、不使用接收端验证数据的零样本场景下，将3比特量化的Top-1准确率提升最多60%。

    

    我们证明了训练后量化（PTQ）的鲁棒性是权重空间中一个可迁移的方向。我们将这一方向称为量化向量：通过简单的权重空间算术从供体任务中提取，可用于修补接收模型，在3比特设置下将PTQ Top-1准确率提升最多60%，且无需在接收端进行量化感知训练（QAT）。在不使用任何接收端验证数据的情况下，该方法为极低比特部署提供了一种零样本、低成本的QAT替代方案。在多个视觉与语言模型以及30多个任务上的实验表明，即使供体任务与接收任务差异显著，供体量化向量通常仍能带来可观的收益。我们进一步严格证明了量化向量是良定义的，且不受重参数化对称性的影响，并对其效果给出了局部几何层面的解释。这些结果共同表明，量化鲁棒性可以成为……（原文摘要在此处截断）

    arXiv:2604.03420v4 Announce Type: replace-cross  Abstract: We show that robustness to post-training quantization (PTQ) is a transferable direction in weight space. We call this direction the quantization vector: extracted from a donor task by simple weight-space arithmetic, it can be used to patch a receiver model and improve PTQ Top-1 accuracy by up to 60% in a 3-bit setting, without receiver-side quantization-aware training (QAT). When using no receiver validation data, the method provides a zero-shot, low-cost alternative to QAT for extremely low-bit deployment. Across multiple vision and language models and more than 30 tasks, donor quantization vectors often yield substantial gains even when donor and receiver tasks differ markedly. We further prove rigorously that quantization vectors are well-defined and do not suffer from reparameterization symmetries, and provide a local geometric account of their effect. Together, these results suggest that quantization robustness can be part
    
[^348]: 上下文生存分析：基于先验拟合网络的摊销贝叶斯生存分析

    Survival In-Context: Amortized Bayesian Survival Analysis via Prior-Fitted Networks

    [https://arxiv.org/abs/2603.29475](https://arxiv.org/abs/2603.29475)

    本文提出SIC模型，通过将先验拟合网络在合成生存数据上预训练以摊销贝叶斯后验预测推断，实现了仅凭单次前向传播即可完成个体化生存预测的上下文学习生存分析方法。

    

    生存分析对许多医学应用至关重要，但由于数据有限、删失以及表格协变量的异质性，它对现代机器学习而言仍然具有挑战性。虽然先验拟合范式——即在大量合成数据集集合上预训练模型——最近催生了用于分类和回归的表格基础模型，但其在时间-事件建模中的适用性仍不明确。我们提出了一个灵活的生存数据生成框架，该框架定义了一个丰富的生存先验，可对协变量和时间-事件分布进行显式控制。基于这一先验，我们引入了 Survival In-Context (SIC)，这是一种完全在合成数据上预训练的、用于生存分析的先验拟合上下文学习模型。SIC 被训练用于在合成生存先验下近似贝叶斯后验预测推断，从而能够在单次前向传播中实现个体化的生存预测。

    arXiv:2603.29475v3 Announce Type: replace  Abstract: Survival analysis is crucial for many medical applications, but remains challenging for modern machine learning due to limited data, censoring, and the heterogeneity of tabular covariates. While the prior-fitted paradigm, which relies on pretraining models on large collections of synthetic datasets, has recently facilitated tabular foundation models for classification and regression, its suitability for time-to-event modeling remains unclear. We propose a flexible survival data generation framework that defines a rich survival prior with explicit control over covariates and time-event distributions. Building on this prior, we introduce Survival In-Context (SIC), a prior-fitted in-context learning model for survival analysis that is pretrained exclusively on synthetic data. SIC is trained to approximate Bayesian posterior predictive inference under the synthetic survival prior, enabling individualized survival prediction in a single f
    
[^349]: 解释的正确性重要吗？将计算化XAI评估与人类理解联系起来

    Does Explanation Correctness Matter? Linking Computational XAI Evaluation to Human Understanding

    [https://arxiv.org/abs/2603.25251](https://arxiv.org/abs/2603.25251)

    该研究通过200人用户实验，在四个受控正确性水平（100%、85%、70%、55%）下首次实证检验了解释正确性与人类理解之间的联系，发现解释的功能正确性确实会影响人类对AI决策的前向模拟准确率，从而为计算化XAI评估指标与人类理解之间建立了关联。

    

    解释性人工智能（XAI）方法通常使用功能正确性指标（也称为忠实度或保真度）进行评估，这些指标衡量解释在多大程度上反映模型的真实推理过程。人们通常假设正确性越高，人类理解效果就越好，但这一联系尚未在受控的正确性水平下得到检验。我们开展了一项用户研究（N=200），在一个合成时间序列分类任务中，将解释正确性操纵在四个水平（100%、85%、70%、55%），参与者无法依赖领域知识或视觉直觉进行判断。解释被构建为类似特征归因的形式，其正确性基于已知的真实标准定义，而非从训练好的模型中估计得出。参与者需要根据这些解释预测模拟AI的决策（前向模拟），我们以其前向模拟准确率作为理解程度的代理指标。正确性影响了前向模拟准确率

    arXiv:2603.25251v3 Announce Type: replace-cross  Abstract: Explainable AI (XAI) methods are commonly evaluated using functional correctness metrics, also called faithfulness or fidelity, which estimate how closely an explanation reflects the model's reasoning. Higher correctness is assumed to produce better human understanding, but this link has not been tested with controlled levels. We conducted a user study (N=200) that manipulated explanation correctness at four levels (100%, 85%, 70%, 55%) in a synthetic time series classification task where participants could not rely on domain knowledge or visual intuition. Explanations were constructed to resemble feature attributions, with correctness defined against a known ground truth rather than estimated from a trained model. Participants predicted a simulated AI's decisions from these explanations (forward simulation), and we used their forward simulation accuracy as a proxy for understanding. Correctness affected forward simulation accu
    
[^350]: 一种方法并非适用于所有情况：单细胞RNA测序数据填补方法的大规模基准测试

    One Method Does Not Fit All: A Large-Scale Benchmark of Imputation Methods for Single-Cell RNA Sequencing Data

    [https://arxiv.org/abs/2603.24626](https://arxiv.org/abs/2603.24626)

    本文提出了首个大规模标准化基准测试框架，在10种实验方案、6个下游任务和30个数据集上系统评估了16种单细胞RNA测序数据填补方法，结果表明没有一种方法能够适用于所有场景。

    

    单细胞RNA测序技术能够在细胞分辨率下进行基因表达谱分析，但其固有地受到由dropout事件导致的稀疏性影响，即由于技术限制，已表达的基因被记录为零。这些dropout事件会扭曲基因表达分布，并损害下游分析的效果。为应对这一挑战，众多传统方法和基于深度学习（DL）的填补方法已被开发出来，用于检测dropout事件并填补其真实表达值。然而，对这些方法的评估大多局限于特定的下游任务和有限的数据集，缺乏在标准化基准测试条件下的大规模系统性比较。为填补这一空白，我们提出了一个全面的评估框架，在10种实验方案、6个下游任务和30个基准数据集上对16种填补方法进行了评估。通过大量实验，我们系统地……（原文摘要到此截断）

    arXiv:2603.24626v3 Announce Type: replace-cross  Abstract: Single-cell RNA sequencing (scRNA-seq) enables gene expression profiling at cellular resolution but is inherently affected by sparsity caused by dropout events, where expressed genes are recorded as zeros due to technical limitations. These dropout events distort gene expression distributions and compromise downstream analyses. To address this challenge, numerous traditional and deep learning (DL)-based imputation methods have been developed to detect dropout events and impute their true expression values. However, evaluations of these methods have largely been restricted to specific downstream tasks and limited datasets, and large-scale systematic comparisons under standardized benchmarking conditions are lacking. To fill this gap, we present a comprehensive framework that evaluates 16 imputation methods across 10 experimental protocols, 6 downstream tasks, and 30 benchmark datasets. Through extensive experimentation, we syste
    
[^351]: 通过下次就诊预测扩展具有复发感知能力的临床记录基础模型

    Scaling Recurrence-aware Foundation Models for Clinical Records via Next-Visit Prediction

    [https://arxiv.org/abs/2603.24562](https://arxiv.org/abs/2603.24562)

    本文提出RAVEN——一种基于复发感知的下次就诊事件预测的临床记录基础模型预训练策略，揭示了重复事件token虚高评估指标的陷阱，并证明在数据受限时单纯扩大模型规模并非最优。

    

    尽管大规模预训练已经在语言建模领域引发了革命，但其在医疗保健领域（尤其是结构化电子健康记录EHR）的潜力仍未得到充分探索。我们提出了RAVEN，这是一种基于“复发感知的下次就诊事件预测”的针对序列化EHR数据的新型生成式预训练策略。利用包含超过一百万独立个体的数据集，我们的模型学会了以患者病史为条件，自回归地生成下次就诊的分词化临床事件。我们引入了对重复事件预测的正则化方法，并指出了基于EHR的基础模型评估中的一个关键陷阱：当新发事件与后续复发事件未被区分时，重复的事件token可能会虚高性能指标。此外，我们实证研究了在数据受限、计算资源趋于饱和情形下的缩放行为，结果表明，在没有相应增加数据量的情况下，单纯增大模型规模并非最优策略。

    arXiv:2603.24562v3 Announce Type: replace  Abstract: While large-scale pretraining has revolutionized language modeling, its potential remains underexplored in healthcare with structured electronic health records (EHRs). We present RAVEN, a novel generative pretraining strategy for sequential EHR data based on Recurrence-Aware next-Visit EveNt prediction. Leveraging a dataset of over one million unique individuals, our model learns to autoregressively generate tokenized clinical events for the next visit conditioned on patient history. We introduce regularization on predicting repeated events and highlight a key pitfall in EHR-based foundation model evaluations: repeated event tokens can inflate performance metrics when new onsets are not distinguished from subsequent occurrences. Furthermore, we empirically investigate the scaling behaviors in a data-constrained, compute-saturated regime, showing that simply increasing model size is suboptimal without commensurate increases in data vo
    
[^352]: VTAM：无需触觉预训练的高效视觉-触觉世界-动作模型，用于富接触操作

    VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining

    [https://arxiv.org/abs/2603.23481](https://arxiv.org/abs/2603.23481)

    VTAM提出了一种无需触觉或动作预训练的高效视觉-触觉世界-动作模型，通过共享冻结的视频VAE并联合预测未来视觉与触觉潜在表示，在富接触操作的真实机器人任务上超越了所有基线。

    

    触觉视觉-语言-动作模型（VLAs）和触觉世界-动作模型（WAMs）是富接触操作的一种有前景的新范式，但现有模型要么未能充分利用触觉信息，要么依赖大规模触觉预训练，要么因迭代式视频生成而产生较高的推理延迟。我们提出了VTAM，一种用于富接触操作的高效视觉-触觉世界-动作模型，它无需触觉或动作预训练。RGB和触觉观测共享一个冻结的视频VAE以及预训练的视频transformer权重。VTAM首先将视频主干网络适配为联合预测未来的视觉和触觉潜在表示，然后与一个随机初始化的动作专家一起进行端到端微调，该动作专家负责预测动作、本体感知状态，以及一个由触觉光流计算得到的轻量级流-力代理信号。仅使用任务演示数据进行训练，VTAM在两个平台上的四个真实机器人任务中超越了所有基线，这些平台具有不同的……

    arXiv:2603.23481v2 Announce Type: replace-cross  Abstract: Tactile vision-language-action models (VLAs) and tactile world-action models (WAMs) are a promising new paradigm for contact-rich manipulation, but existing models either underutilize touch, rely on large-scale tactile pretraining, or incur high inference latency due to iterative video generation. We present VTAM, an efficient visuo-tactile world-action model for contact-rich manipulation that requires no tactile or action pretraining. RGB and tactile observations share a frozen video VAE and the pretrained video transformer weights. VTAM first adapts the video backbone to jointly predict future visual and tactile latents, then fine-tunes it end-to-end with a randomly initialized action expert that predicts actions, proprioceptive states, and a light flow-force proxy computed from tactile optical flow. Trained only on task demonstrations, VTAM outperforms all baselines on four real-robot tasks across two platforms with differen
    
[^353]: 神经元自适应增强脉冲神经网络表征的容量与鲁棒性

    Neuronal Self-Adaptation Enhances Capacity and Robustness of Representation in Spiking Neural Networks

    [https://arxiv.org/abs/2603.20687](https://arxiv.org/abs/2603.20687)

    受生物钾通道动态自调节机制启发，提出KvLIF神经元模型，通过辅助电导状态自适应调节神经元兴奋性与复位动力学，显著提升脉冲神经网络的信息容量和抗噪鲁棒性。

    

    脉冲神经网络（SNNs）在节能、实时的边缘计算方面前景广阔，但其性能往往受限于传统漏积分-激发神经元有限的适应能力。现有LIF模型存在信息容量受限和易受噪声干扰的问题，导致准确率下降和鲁棒性受损。受生物钾通道动态自调节机制的启发，我们提出了钾调节LIF神经元模型。KvLIF引入了一个辅助电导状态，该状态整合膜电位和脉冲发放历史，以自适应地调节神经元的兴奋性和复位动力学。这一设计扩展了神经元对不同输入强度的动态响应范围，并有效抑制噪声诱发的脉冲。我们在静态图像和神经形态数据集上对KvLIF进行了广泛评估，结果表明其在分类准确率方面取得了一致的提升。

    arXiv:2603.20687v2 Announce Type: replace  Abstract: Spiking Neural Networks (SNNs) are promising for energy-efficient, real-time edge computing, yet their performance is often constrained by the limited adaptability of conventional leaky integrate-and-fire (LIF) neurons. Existing LIF models struggle with restricted information capacity and susceptibility to noise, leading to degraded accuracy and compromised robustness. Inspired by the dynamic self-regulation of biological potassium channels, we propose the Potassium-regulated LIF (KvLIF) neuron model. KvLIF introduces an auxiliary conductance state that integrates membrane potential and spiking history to adaptively modulate neuronal excitability and reset dynamics. This design extends the dynamic response range of neurons to varying input intensities and effectively suppresses noise-induced spikes. We extensively evaluate KvLIF on both static image and neuromorphic datasets, demonstrating consistent improvements in classification ac
    
[^354]: 混乱中的“CLaRE-ty”（清晰度）：量化表征纠缠以预测大语言模型编辑中的涟漪效应

    CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing

    [https://arxiv.org/abs/2603.19297](https://arxiv.org/abs/2603.19297)

    本文提出CLaRE，一种轻量级的表征层面技术，仅利用单个中间层的前向激活（无需昂贵的反向传播）量化事实之间的纠缠程度，从而预测大语言模型编辑中涟漪效应可能发生的位置。

    

    大语言模型（LLM）的静态知识表征随着时间推移不可避免地会变得过时或错误。虽然模型编辑技术通过修改模型的事实关联提供了一种有前景的解决方案，但它们常常产生不可预测的涟漪效应，即甚至能传播到隐藏空间的非预期行为变化。在这项工作中，我们提出了CLaRE，这是一种轻量级的表征层面的技术，用于识别这些涟漪效应可能发生的位置。与以往基于梯度的方法不同，CLaRE使用来自单个中间层的前向激活来量化事实之间的纠缠程度，从而避免了代价高昂的反向传播。为了实现系统性研究，我们准备并分析了从三个现有数据集中提取的11,427条事实组成的语料库。利用CLaRE，我们为多个模型计算了该语料库的大规模纠缠图，以捕捉局部编辑如何在表征空间中传播。这些图使我们能够……

    arXiv:2603.19297v3 Announce Type: replace  Abstract: The static knowledge representations of large language models (LLMs) inevitably become outdated or incorrect over time. While model-editing techniques offer a promising solution by modifying a model's factual associations, they often produce unpredictable ripple effects, which are unintended behavioral changes that propagate even to the hidden space. In this work, we introduce CLaRE, a lightweight representation-level technique to identify where these ripple effects may occur. Unlike prior gradient-based methods, CLaRE quantifies entanglement between facts using forward activations from a single intermediate layer, avoiding costly backward passes. To enable systematic study, we prepare and analyse a corpus of 11,427 facts drawn from three existing datasets. Using CLaRE, we compute large-scale entanglement graphs of this corpus for multiple models, capturing how local edits propagate through representational space. These graphs enable
    
[^355]: 面向长尾鲁棒驾驶的偏好引导对抗策略优化

    Preference-Guided Adversarial Policy Optimization for Long-Tail Robust Driving

    [https://arxiv.org/abs/2603.15221](https://arxiv.org/abs/2603.15221)

    ADV-0提出一个闭环极小极大优化框架，将驾驶策略与对抗智能体的交互建模为零和马尔可夫博弈，并通过迭代偏好学习高效逼近最优对抗分布，从而提升自动驾驶系统对罕见但安全关键的长尾场景的鲁棒性。

    

    部署自动驾驶系统需要具备应对长尾场景的鲁棒性——这些场景虽然罕见，但事关安全。尽管对抗训练提供了一种有前景的解决方案，但现有方法通常将场景生成与策略优化解耦，并依赖启发式替代目标。这导致目标不一致，且无法捕捉不断演化的策略所呈现的动态失效模式。本文提出ADV-0，一个闭环的极小极大优化框架，将驾驶策略（防御者）与对抗智能体（攻击者）之间的交互建模为零和马尔可夫博弈。通过将攻击者的效用直接与防御者的目标对齐，我们揭示了最优对抗分布。为使该问题可求解，我们将动态对抗演化转化为迭代偏好学习，高效逼近该最优解，并为该博弈提供一种与算法无关的解决方案。理论上，ADV-0收敛到……（摘要在此处被截断）

    arXiv:2603.15221v2 Announce Type: replace-cross  Abstract: Deploying autonomous driving systems requires robustness against long-tail scenarios that are rare but safety-critical. While adversarial training offers a promising solution, existing methods typically decouple scenario generation from policy optimization and rely on heuristic surrogates. This leads to objective misalignment and fails to capture the shifting failure modes of evolving policies. This paper presents ADV-0, a closed-loop min-max optimization framework that treats the interaction between driving policy (defender) and adversarial agent (attacker) as a zero-sum Markov game. By aligning the attacker's utility directly with the defender's objective, we reveal the optimal adversary distribution. To make this tractable, we cast dynamic adversary evolution as iterative preference learning, efficiently approximating this optimum and offering an algorithm-agnostic solution to the game. Theoretically, ADV-0 converges to a Na
    
[^356]: FastWave：用于音频超分辨率的优化扩散模型

    FastWave: Optimized Diffusion Model for Audio Super-Resolution

    [https://arxiv.org/abs/2603.04122](https://arxiv.org/abs/2603.04122)

    FastWave是一个轻量级的扩散模型（约50 GFLOPs计算量、130万参数），用于将任意采样率的音频超分辨率提升至48 kHz，其性能优于NU-Wave 2并与最先进模型相当，同时训练所需的资源和时间显著更少。

    

    音频超分辨率是一系列旨在对给定信号进行高质量估计的技术，其目标就如同该信号以更高采样率被采样所得到的效果。在已有的方法中，扩散模型和流模型被认为速度较慢，生成对抗网络被认为速度较快，然而目前这两类方法都由高参数量的网络实现，无论在训练还是推理阶段都需要高昂的计算成本。我们通过重新审视扩散模型训练方面的最新进展，并将其应用于从任意采样率到48 kHz采样率的超分辨率任务，提出了能同时解决上述两个问题的方案。我们的方法取得了优于NU-Wave 2的结果，并与最先进的模型相当。我们将其命名为FastWave，该模型计算复杂度约为50 GFLOPs，参数量仅130万，与近期提出的大多数扩散模型相比，其训练所需资源更少且速度显著更快。

    arXiv:2603.04122v2 Announce Type: replace-cross  Abstract: Audio Super-Resolution is a set of techniques aimed at high-quality estimation of the given signal as if it would be sampled with higher sample rate. Among suggested methods there are diffusion and flow models (which are considered slower), generative adversarial networks (which are considered faster), however both approaches are currently presented by high-parametric networks, requiring high computational costs both for training and inference. We propose a solution to both these problems by re-considering the recent advances in the training of diffusion models and applying them to super-resolution from any to 48 kHz sample rate. Our approach shows better results than NU-Wave 2 and is comparable to state-of-the-art models. Our model called FastWave has around 50 GFLOPs of computational complexity and 1.3 M parameters and can be trained with less resources and significantly faster than the majority of recently proposed diffusion
    
[^357]: DUEL：精确似然重塑掩码扩散模型的缩放定律及更多发现

    DUEL: Exact Likelihood Reshapes Masked Diffusion Scaling Laws and Beyond

    [https://arxiv.org/abs/2603.01367](https://arxiv.org/abs/2603.01367)

    本文提出DUEL框架，能够在掩码扩散模型实际使用的确定性采样器下精确计算其似然，无需重新训练即可即插即用，并由此重塑了关于掩码扩散模型缩放定律等一系列结论。

    

    掩码扩散模型（MDMs）被训练为以任意顺序预测词元，但在生成时只以一种顺序进行。因此，MDM的似然是对所有顺序求和的难解计算。证据下界（ELBO）被用作似然的代理，但它是一个宽松的界，并且假设位置是均匀随机掩码的。然而，前沿的MDM使用确定性规则（如置信度）来选择位置。为解决这种不匹配，我们引入了DUEL，这是一个统一常见确定性MDM采样器的框架。DUEL可以在生成时实际使用的采样器下精确计算似然。DUEL是一种简单的即插即用算法，适用于任何预训练的MDM，无需重新训练。要计算文本序列的似然，只需重放生成过程，用真实词元替代采样词元，并对它们的对数概率求和。使用DUEL而非ELBO进行评估，改变了我们关于MDM的结论。（注：原文摘要在此处被截断）

    arXiv:2603.01367v3 Announce Type: replace  Abstract: Masked diffusion models (MDMs) are trained to predict tokens in every order but generate in one. As such, MDM likelihood is an intractable sum over all orders. The evidence lower bound (ELBO) is used as a proxy for likelihood, but it is a loose bound and assumes positions are unmasked uniformly at random. Frontier MDMs, however, select positions with deterministic rules such as confidence. To resolve this mismatch we introduce DUEL, a framework that unifies common deterministic MDM samplers. DUEL computes the likelihood exactly, under the sampler that is actually used during generation. DUEL is a simple plug-and-play algorithm applicable to any pretrained MDM with no retraining. To compute the likelihood of a text sequence, simply replay generation, reveal the true tokens instead of sampled ones, and sum their log-probabilities. Evaluating with DUEL instead of the ELBO changes what we conclude about MDMs. The perplexity gap between M
    
[^358]: 表演性预测中在线算法的稳定性

    The Stability of Online Algorithms in Performative Prediction

    [https://arxiv.org/abs/2602.24207](https://arxiv.org/abs/2602.24207)

    本文证明部署在表演性预测环境中的任何无悔在线算法都会无条件收敛到（混合）表演性稳定均衡，且无需对人群如何响应预测做任何假设。

    

    在决策中使用算法预测会引发一个反馈循环：我们所部署的模型会主动影响所观测到的数据分布，而这些数据随后又被用于重新训练。Perdomo 等人于 2020 年在其关于表演性预测（performative prediction）的工作中对这一动态进行了形式化刻画。我们的主要结果是一个无条件归约，表明部署在表演性环境中的任何无悔算法都会收敛到一个（混合）表演性稳定均衡：在这种解中，模型主动塑造数据分布，使得其自身的预测在事后看来是最优的。在本工作之前，该领域所有的正面结果都对模型如何影响分布施加了严格的限制。通过采用鞅论证并允许随机化，我们避免了对人群如何响应预测的任何假设，并规避了近期的困难性结果——这些结果表明确定性的稳定模型在一般情况下是 PPAD 难以计算的。最后，关于……（原文摘要在此处截断）

    arXiv:2602.24207v3 Announce Type: replace  Abstract: The use of algorithmic predictions in decision-making leads to a feedback loop where the models we deploy actively influence the data distributions we see, and later use to retrain on. This dynamic was formalized by Perdomo et al. 2020 in their work on performative prediction. Our main result is an unconditional reduction showing that any no-regret algorithm deployed in performative settings converges to a (mixed) performatively stable equilibrium: a solution in which models actively shape data distributions in ways that their own predictions look optimal in hindsight. Prior to our work, all positive results in this area imposed strong restrictions on how models influenced distributions. By using a martingale argument and allowing randomization, we avoid any assumption on how populations respond to predictions and sidestep recent hardness results showing that deterministic stable models are in general PPAD-hard to compute. Lastly, on
    
[^359]: 可训练频率能移动多远？可训练频率量子模型中的分辨率、网格与耦合

    How Far Can a Trainable Frequency Move? Resolution, Grids and Coupling in Trainable-FrequencyQuantum Models

    [https://arxiv.org/abs/2602.23409](https://arxiv.org/abs/2602.23409)

    该论文揭示了可训练频率量子模型中前置因子梯度的解析结构——由 sin u/u 重叠函数斜率的加权和决定每个频率的可达范围，并提出三值前置因子网格方案，仅用对数数量的编码门即可将模型频率逼近所有频带频率至半个步长以内。

    

    可训练频率量子模型通过训练其编码门的前置因子，使电路的频率能够移动到目标的频率上。从先前工作的单位起始点出发，训练很少能到达距离超过一个分辨率步长的目标。我们证明，前置因子梯度中由目标驱动的部分是重叠函数 sin u/u 斜率的加权和，其中 u 是按窗口半宽缩放的模型-目标频率距离。这赋予每个频率一个“可达范围”：在梯度流下，一个系数最优自适应的孤立实频率能够从半个步长之内到达其目标，只要目标距离零频率至少一个步长；而在系数固定的情况下，一个依赖相位的势垒可能会阻止它。一个三值前置因子网格能够使模型频率与每个频带频率的差距都在半个步长以内，且只需对数数量的编码门。使用 Adam 训练的单频率电路以及联合训练的振幅……

    arXiv:2602.23409v4 Announce Type: replace-cross  Abstract: Trainable-frequency quantum models train the prefactors of their encoding gates so that the circuit's frequencies can move onto the target's. From the unit start of prior work, training rarely reaches a target more than one resolution step away. We show that the target-driven part of the prefactor gradient is a weighted sum of slopes of the overlap $\sin u/u$, where $u$ is the model-target frequency distance scaled by the window half-width. This gives each frequency a reach: under gradient flow, an isolated real frequency whose coefficient adapts optimally reaches its target from within half a step, for every target at least one step from zero frequency; with a fixed coefficient, a phase-dependent barrier can stop it. A ternary prefactor grid places a model frequency within half a step of every band frequency, with logarithmically many encoding gates. Single-frequency circuits trained with Adam and jointly trained amplitudes be
    
[^360]: 告诉我该学什么：将神经记忆泛化为可用自然语言控制的系统

    Tell Me What To Learn: Generalizing Neural Memory to be Controllable in Natural Language

    [https://arxiv.org/abs/2602.23201](https://arxiv.org/abs/2602.23201)

    提出一种可用自然语言指令控制的广义神经记忆系统，使模型能够依据用户指令从异构信息源中选择性地学习，突破了传统神经记忆固定目标的局限。

    

    arXiv:2602.23201v3 公告类型：替换 摘要：现代机器学习模型被部署在多样化、非平稳的环境中，必须持续适应新任务和不断演进的知识。持续微调和上下文学习成本高昂且脆弱，而神经记忆方法有望以最小遗忘实现轻量级更新。然而，现有的神经记忆模型通常假设单一固定目标和同质信息流，使用户无法控制模型随时间推移记住或忽略哪些内容。为应对这一挑战，我们提出了一个广义的神经记忆系统，能够根据自然语言指定的学习指令执行灵活的更新。我们的方法使自适应智能体能够从异构信息源中进行选择性学习，支持诸如医疗保健和客户服务等固定目标记忆更新无法满足的应用场景。

    arXiv:2602.23201v3 Announce Type: replace  Abstract: Modern machine learning models are deployed in diverse, non-stationary environments where they must continually adapt to new tasks and evolving knowledge. Continual fine-tuning and in-context learning are costly and brittle, whereas neural memory methods promise lightweight updates with minimal forgetting. However, existing neural memory models typically assume a single fixed objective and homogeneous information streams, leaving users with no control over what the model remembers or ignores over time. To address this challenge, we propose a generalized neural memory system that performs flexible updates based on learning instructions specified in natural language. Our approach enables adaptive agents to learn selectively from heterogeneous information sources, supporting settings, such as healthcare and customer service, where fixed-objective memory updates are insufficient.
    
[^361]: 大语言模型的预言机鲁棒在线对齐

    Oracle-Robust Online Alignment for Large Language Models

    [https://arxiv.org/abs/2602.20457](https://arxiv.org/abs/2602.20457)

    本文针对偏好反馈预言机可能存在偏差的大语言模型在线对齐问题，提出最坏情况鲁棒优化目标，并证明该鲁棒目标可精确分解为原始损失加显式敏感度惩罚项，进而设计了投影随机复合更新算法进行求解。

    

    我们研究了在偏好反馈被错误设定情形下的大语言模型在线对齐问题，其中观测到的偏好预言机与理想但未知的真实预言机之间存在偏差。由于数据收集与策略更新之间的耦合，在线大语言模型对齐问题本质上是一个双层强化学习问题。近期，SAIL（自我提升的高效在线对齐）框架已将该问题约化为一个易于处理的单层目标。本文在该问题中引入了逐点预言机不确定性集合，并将预言机鲁棒的在线对齐目标表述为一个最坏情况优化问题。对于对数线性策略，我们证明该鲁棒目标可以精确地闭式分解为原始损失函数加上一个显式的敏感度惩罚项。在有限响应的对数线性设定下，我们针对所得的弱凸目标提出了投影随机复合更新方法……

    arXiv:2602.20457v2 Announce Type: replace  Abstract: We study online alignment of large language models under misspecified preference feedback, where the observed preference oracle deviates from an ideal but unknown ground-truth oracle. The online LLM alignment problem is a bi-level reinforcement problem due to the coupling between data collection and policy updates. Recently, the problem has been reduced to a tractable single-level objective in the SAIL (Self-Improving Efficient Online Alignment) framework. In this paper, we introduce a pointwise oracle uncertainty set in this problem and formulate an oracle-robust online alignment objective as a worst-case optimization problem. For log-linear policies, we show that this robust objective admits an exact closed-form decomposition into the original loss function plus an explicit sensitivity penalty. In the finite-response log-linear setting, we develop projected stochastic composite updates for the resulting weakly convex objective and 
    
[^362]: 基于通用投资组合算法的在线保形预测

    Online Conformal Prediction via Universal Portfolio Algorithms

    [https://arxiv.org/abs/2602.03168](https://arxiv.org/abs/2602.03168)

    本文识别出“线性化遗憾”这一关键概念并建立了通用的“遗憾到覆盖”理论，据此提出无需任何参数调整的UP-OCP方法，通过将问题归约为两资产投资组合选择问题并利用通用投资组合算法，实现了参数无关的在线保形预测覆盖率保证。

    

    在线保形预测（OCP）旨在为任意（可能是对抗性的）数据流提供能够实现长期 1-α 覆盖率的预测区间，同时尽可能保持区间信息量。现有的OCP方法通常需要手动调整学习率才能良好工作，并且可能还需要针对特定算法的分析。在本文中，我们基于 (1-α)-pinball 损失，为区间值OCP发展了一套通用的“遗憾到覆盖”理论。我们的第一个贡献是识别出“线性化遗憾”这一关键概念，表明控制该量即可为任何在线算法导出覆盖界。这一结果依赖于一个黑盒归约，其仅依赖于线性化遗憾上界的Fenchel共轭。基于这一理论，我们通过将OCP问题归约为两资产投资组合选择问题，并利用通用投资组合算法，提出了UP-OCP——一种无需调参的OCP方法。我们展示了在……上的强有限时间界。

    arXiv:2602.03168v2 Announce Type: replace-cross  Abstract: Online conformal prediction (OCP) seeks prediction intervals that achieve long-run $1-\alpha$ coverage for arbitrary (possibly adversarial) data streams, while remaining as informative as possible. Existing OCP methods often require manual learning-rate tuning to work well, and may also require algorithm-specific analyses. Here, we develop a general regret-to-coverage theory for interval-valued OCP based on the $(1-\alpha)$-pinball loss. Our first contribution is to identify \emph{linearized regret} as a key notion, showing that controlling it implies coverage bounds for any online algorithm. This relies on a black-box reduction that depends only on the Fenchel conjugate of an upper bound on the linearized regret. Building on this theory, we propose UP-OCP, a parameter-free method for OCP, via a reduction to a two-asset portfolio selection problem, leveraging universal portfolio algorithms. We show strong finite-time bounds on 
    
[^363]: 显微镜下的双曲图神经网络：几何-任务对齐的作用

    Hyperbolic Graph Neural Networks Under the Microscope: The Role of Geometry-Task Alignment

    [https://arxiv.org/abs/2602.01828](https://arxiv.org/abs/2602.01828)

    这项工作提出“几何-任务对齐”这一关键条件，通过实验和理论分析表明，双曲图神经网络只有在任务的度量结构与输入图的几何结构相匹配时才能真正获得优势。

    

    许多复杂网络呈现出层次化的树状结构，这使得双曲空间成为学习其表示的天然候选。基于这一观察，双曲图神经网络（HGNNs）已被广泛采用作为树状图表示学习的一种有原则的选择。在这项工作中，我们通过提出几何-任务对齐这一额外条件来质疑这一范式，即目标的度量结构是否遵循输入图的度量结构。我们首先通过实验证明，当监督信号明确奖励度量保持时，HGNNs 能够恢复低失真的表示；随后我们从理论上分析了一个受控回归设置，其中线性读出层受益于几何与目标度量对齐的表示。最后，通过结合预测性能和嵌入失真两个方面，我们进一步表明 HGNNs 在（几何-任务对齐的条件下）获得优势。

    arXiv:2602.01828v3 Announce Type: replace  Abstract: Many complex networks exhibit hierarchical, tree-like structures, making hyperbolic space a natural candidate wherein to learn representations of them. Based on this observation, Hyperbolic Graph Neural Networks (HGNNs) have been widely adopted as a principled choice for representation learning on tree-like graphs. In this work, we question this paradigm by proposing the additional condition of geometry--task alignment, i.e., whether the metric structure of the target follows that of the input graph. We start by empirically showing that HGNNs can recover low-distortion representations when supervision explicitly rewards metric preservation and proceed by theoretically analyzing a controlled regression setting in which a linear readout benefits from representations whose geometry is aligned with the target metric. Finally, by relying on both predictive performance and embedding distortion, we further show that HGNNs gain an advantage 
    
[^364]: 重新思考掩码扩散语言模型训练中的位置监督

    Reconsidering Positional Supervision in Masked Diffusion Language Model Training

    [https://arxiv.org/abs/2601.22947](https://arxiv.org/abs/2601.22947)

    该论文通过受控干预实验发现掩码扩散语言模型在迭代并行解码下对微小的位置偏移极为敏感（仅偏移1%的token即显著降低胜率），并据此提出改造连接主义时间分类（CTC）以改进位置监督。

    

    掩码扩散语言模型（MDLMs）通过并行解除掩码的方式生成文本，近来已成为自回归语言模型的替代方案。它们可以被视为采用逐位置交叉熵（CE）损失训练的并行解码器，这一设置与非自回归翻译（NAT）相同。在NAT领域，已有观点认为采用CE训练的并行解码器对微小的位置偏移十分敏感，因为CE会对这类偏移施加严厉惩罚。我们探究了采用CE训练的MDLMs在迭代解码下是否对这类偏移同样敏感。为验证这一点，我们施加了一种受控干预，在解码过程中引入位置偏移。在LLaDA-8B-Instruct模型与Arena-Hard基准上，仅将1%的生成token移动一个位置，就会显著降低相对于未干预模型的胜率，这表明MDLMs在迭代并行解码下对如此微小的偏移非常敏感。受此启发，我们改造了连接主义时间分类（CTC）……（原文摘要在此处截断）

    arXiv:2601.22947v3 Announce Type: replace  Abstract: Masked diffusion language models (MDLMs) generate text by unmasking tokens in parallel and have recently emerged as alternatives to autoregressive language models. They can be viewed as parallel decoders trained with a position-wise cross-entropy (CE) loss, the same setup as non-autoregressive translation (NAT). In NAT, CE-trained parallel decoders have been argued to be sensitive to small positional shifts, since CE penalizes them harshly. We ask whether CE-trained MDLMs are similarly sensitive to such shifts under iterative decoding. To probe this, we apply a controlled intervention that introduces them during decoding. On LLaDA-8B-Instruct with Arena-Hard, displacing as little as 1% of generated tokens by one position substantially reduces win rates against the unintervened model, showing that MDLMs are sensitive to such small shifts under iterative parallel decoding. Motivated by this, we adapt connectionist temporal classificati
    
[^365]: 基于动态规划的可微分背包与Top-k算子

    Differentiable Knapsack and Top-k Operators via Dynamic Programming

    [https://arxiv.org/abs/2601.21775](https://arxiv.org/abs/2601.21775)

    本文提出一个统一框架，将背包与Top-k算子转化为动态规划并平滑递推得到可微分松弛，配有高效并行前向与反向算法，并从理论上证明香农负熵是唯一保证置换等变性的可分离正则化。

    

    背包和Top-k算子对于选择变量的离散子集非常有用。然而，将它们集成到神经网络中极具挑战性，因为它们是分段常数的函数，其梯度几乎处处为零。在本文中，我们提出了一个统一框架，将这些算子建模为动态规划问题，并通过平滑底层的递推关系推导出可微分松弛。在算法方面，我们开发了高效的并行算法，支持确定性和随机的前向传播，以及用于反向传播的向量-雅可比积。在理论方面，我们证明了香农负熵是唯一能够产生置换等变算子的可分离正则化选择，并刻画了能够诱导稀疏选择的正则化子。在实验方面，我们在学习预测背包问题解决方案的基准测试、离散VAE的扩展等任务上展示了我们框架的有效性。

    arXiv:2601.21775v2 Announce Type: replace  Abstract: Knapsack and Top-k operators are useful for selecting discrete subsets of variables. However, their integration into neural networks is challenging as they are piecewise constant, yielding gradients that are zero almost everywhere. In this paper, we propose a unified framework casting these operators as dynamic programs, and derive differentiable relaxations by smoothing the underlying recursions. On the algorithmic side, we develop efficient parallel algorithms supporting both deterministic and stochastic forward passes, and vector-Jacobian products for the backward pass. On the theoretical side, we prove that Shannon's negative entropy is the unique separable regularization choice that yields permutation-equivariant operators, and characterize regularizers inducing sparse selections. On the experimental side, we demonstrate our framework on a benchmark on learning to predict Knapsack solutions, an extension of discrete VAEs, and a 
    
[^366]: 用于量子电路优化通道自适应组合的强化学习

    Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes

    [https://arxiv.org/abs/2601.21629](https://arxiv.org/abs/2601.21629)

    该论文提出训练强化学习智能体来自适应地为量子电路组合优化通道序列，在双量子比特门削减效果上显著优于 PyTKET 默认通道序列（平均 57.7% 对比 41.8%）。

    

    许多量子软件开发工具包都提供了一套电路优化通道。这些通道已被高度优化并经过独立测试。然而，它们的应用顺序却留给用户自行决定，或者由通用的默认通道序列来定义。虽然通用序列会错失针对个别电路特有的优化机会，但为特定电路设计定制的通道序列又需要对量子电路设计与优化的专业知识。在此，我们提出并演示了训练一个强化学习智能体来自动组合优化通道序列。具体而言，该智能体的动作空间由默认 PyTKET 通道序列中用于减少双量子比特门数量的通道组成。对于我们多样化测试集中的电路，智能体所移除的双量子比特门的（平均值，中位数）比例为 (57.7%, 56.7%)，而对比之下（默认序列的结果）为 (41.8%, 50.0%)……

    arXiv:2601.21629v2 Announce Type: replace-cross  Abstract: Many quantum software development kits provide a suite of circuit optimisation passes. These passes have been highly optimised and tested in isolation. However, the order in which they are applied is left to the user, or else defined in general-purpose default pass sequences. While general-purpose sequences miss opportunities for optimisation which are particular to individual circuits, designing pass sequences bespoke to particular circuits requires exceptional knowledge about quantum circuit design and optimisation. Here we propose and demonstrate training a reinforcement learning agent to compose optimisation-pass sequences. In particular the agent's action space consists of passes for two-qubit gate count reduction used in default PyTKET pass sequences. For the circuits in our diverse test set, the (mean, median) fraction of two-qubit gates removed by the agent is $(57.7\%, \ 56.7 \%)$, compared to $(41.8 \%, \ 50.0 \%)$ fo
    
[^367]: LAMP：大语言模型的前瞻式混合精度推理

    LAMP: Look-Ahead Mixed-Precision Inference of Large Language Models

    [https://arxiv.org/abs/2601.21623](https://arxiv.org/abs/2601.21623)

    该论文提出LAMP方法，基于舍入误差分析构建自适应混合精度策略，只需对Transformer中极小部分组件进行高精度重计算，即可使大语言模型推理精度提升高达两个数量级。

    

    混合精度计算是当前人工智能发展阶段的标志性技术，推动着大语言模型朝着高效、可本地部署的解决方案不断进步。本文研究组合丰富的函数的浮点计算问题，重点聚焦于Transformer推理。基于对复合函数 $f(g(\mathrm{x}))$ 的舍入误差分析，我们提出了一种自适应策略，该策略仅选择 $g(\mathrm{x})$ 中的一小部分组件以更高精度进行计算，而其余所有计算均可以较低精度完成。随后，我们解释了该策略如何应用于Transformer内部的不同组合结构，并阐述了其对Transformer推理的整体效果。我们在GPT-2模型上对该算法的有效性进行了数值研究，结果表明，即使非常低的重计算率也能带来高达两个数量级的精度提升。

    arXiv:2601.21623v3 Announce Type: replace  Abstract: Mixed-precision computations are a hallmark of the current stage of AI, driving the progress in large language models towards efficient, locally deployable solutions. This article addresses the floating-point computation of compositionally-rich functions, concentrating on transformer inference. Based on the rounding error analysis of a composition $f(g(\mathrm{x}))$, we provide an adaptive strategy that selects a small subset of components of $g(\mathrm{x})$ to be computed more accurately while all other computations can be carried out with lower accuracy. We then explain how this strategy can be applied to different compositions within a transformer and illustrate its overall effect on transformer inference. We study the effectiveness of this algorithm numerically on GPT-2 models and demonstrate that already very low recomputation rates allow for improvements of up to two orders of magnitude in accuracy.
    
[^368]: 深度循环注意力混合： 让潜在推理得到应有的关注

    Depth-Recurrent Attention Mixture (Dreamer): Giving Latent Reasoning the Attention it Deserves

    [https://arxiv.org/abs/2601.21582](https://arxiv.org/abs/2601.21582)

    该论文提出Dreamer模块化框架，通过结合序列注意力、深度注意力和稀疏专家注意力来突破恒定隐藏维度的瓶颈，使深度循环模型在语言推理基准上以2-8倍更少的训练token达到与SOTA相当的精度，并优于约2倍规模的SOTA模型。

    

    深度循环通过跨深度共享参数来促进潜在推理。然而，先前的工作缺乏同时匹配FLOP、参数量和内存的基线对比，由于部分固定的层堆叠而未充分利用深度循环，并且忽略了限制多步潜在推理的恒定隐藏维度瓶颈。为解决这一问题，我们提出了一个深度循环注意力混合的模块化框架，结合了序列注意力、深度注意力和稀疏专家注意力。该框架通过沿深度方向的注意力缓解了隐藏维度瓶颈，解耦了各缩放维度，使深度循环模型能够高效且有效地扩展。在语言推理基准测试中，我们的模型在与FLOP、参数量和内存均匹配的SOTA模型达到相同精度时，所需训练token减少2到8倍，并在相同训练token数量下优于约2倍规模的SOTA模型。我们还进一步提供了关于知识的洞见……

    arXiv:2601.21582v2 Announce Type: replace  Abstract: Depth-recurrence facilitates latent reasoning by sharing parameters across depths. However, prior work lacks combined FLOP-, parameter-, and memory-matched baselines, underutilizes depth-recurrence due to partially fixed layer stacks, and ignores the bottleneck of constant hidden-sizes that restricts many-step latent reasoning. To address this, we introduce a modular framework of depth-recurrent attention mixtures (Dreamer), combining sequence attention, depth attention, and sparse expert attention. It alleviates the hidden-size bottleneck through attention along depth, decouples scaling dimensions, and allows depth-recurrent models to scale efficiently and effectively. Across language reasoning benchmarks, our models require 2 to 8x fewer training tokens for the same accuracy as FLOP-, parameter-, and memory-matched SOTA, and outperform ca. 2x larger SOTA models with the same training tokens. We further present insights into knowled
    
[^369]: 自我改进的训练与思考

    Self-Improving Training & Thinking

    [https://arxiv.org/abs/2601.21343](https://arxiv.org/abs/2601.21343)

    该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。

    

    传统上，大型语言模型采用分阶段训练：先在原始文本上进行预训练，再进行后训练以学习指令遵循和推理能力。然而，这种分离造成了根本性的局限：许多理想的行为，如安全性、事实准确性、整体生成质量和推理能力，只在训练后期才被加入，尽管早期学习的模式在很大程度上塑造了模型的能力。为解决这一问题，我们提出了一种新的预训练和中期训练方法，将这些行为更早地融入训练过程。我们利用一个现有的强大的经过后训练的模型来重写数据并评判策略模型的输出（rollouts），从而在训练中更早地引入思考和强化学习。我们进一步通过引入“自我改进的后思考”（self-improving post-thinking）的概念，在所有训练阶段提升推理能力：即在生成之后进行推理，以提高模型的推理和自我评判能力。在实验中，我们证明了这些方法……

    arXiv:2601.21343v4 Announce Type: replace-cross  Abstract: Large language models are classically trained in stages: pretraining on raw text followed by post-training for instruction following and reasoning. However, this separation creates a fundamental limitation: many desirable behaviors such as safety, factuality, overall generation quality, and reasoning ability are only added at a late stage, even though the patterns learned earlier strongly shape a model's capabilities. To tackle this issue, we introduce a new way to pretrain and mid-train models that incorporates these behaviors earlier. We utilize an existing strong, post-trained model to both rewrite data and to judge policy model rollouts, thus introducing thinking and reinforcement earlier in training. We further improve reasoning at all stages by introducing the concept of self-improving post-thinking: improving the model's reasoning and self-judging ability by reasoning after generation. In our experiments, we show these a
    
[^370]: 振幅代理模型的缩放定律

    Scaling laws for amplitude surrogates

    [https://arxiv.org/abs/2601.13308](https://arxiv.org/abs/2601.13308)

    该论文系统研究了粒子物理振幅代理模型的缩放定律，发现缩放系数与过程的外部粒子数量相关，并证明缩放定律是实现所需精度目标的有效工具。

    

    描述神经网络性能与训练数据量、计算开销和网络规模之间依赖关系的缩放定律，已在各种机器学习任务和数据集中广泛出现。在这项工作中，我们在粒子物理学的振幅代理模型背景下系统地研究了这些缩放定律。我们证明了缩放系数与该过程的外部粒子数量相关。我们的结果表明，缩放定律是实现所需精度目标的有用工具。

    arXiv:2601.13308v2 Announce Type: replace-cross  Abstract: Scaling laws describing the dependence of neural network performance on the amount of training data, the spent compute, and the network size have emerged across a huge variety of machine learning task and datasets. In this work, we systematically investigate these scaling laws in the context of amplitude surrogates for particle physics. We show that the scaling coefficients are connected to the number of external particles of the process. Our results demonstrate that scaling laws are a useful tool to achieve desired precision targets.
    
[^371]: EmboTeam：面向异构多机器人协作的非对称状态接地

    EmboTeam: Asymmetric State Grounding for Heterogeneous Multi-Robot Collaboration

    [https://arxiv.org/abs/2601.11063](https://arxiv.org/abs/2601.11063)

    提出 EmboTeam 神经符号框架，通过 LLM 生成结构化状态预期、PDDL 符号搜索验证协调，并落地为带局部重规划与黑板同步的团队级并行行为树，实现异构多机器人长时程协作中的非对称状态接地，显著提升执行鲁棒性。

    

    面向长时程的异构多机器人协作既需要全局任务规划，也需要执行过程中的鲁棒协调。现有基于大语言模型（LLM）的规划器能够生成任务分解或符号化规划，但往往将其作为僵硬的动作序列执行，导致其对局部状态不匹配和同步失败十分脆弱。我们提出 EmboTeam，一个用于非对称状态接地的神经符号框架。LLM 生成结构化的状态预期，这些预期通过基于规划领域定义语言（PDDL）的符号搜索进行验证与协调，随后被落地为团队级的并行行为树。针对各机器人的子树负责前置条件检查、技能执行、效果验证、局部重规划以及基于黑板机制的同步。我们进一步提出了 MACE-THOR 基准，包含跨 8 种室内布局的 42 个异构多机器人任务。EmboTeam 提升了任务的……（原文摘要至此截断）

    arXiv:2601.11063v3 Announce Type: replace-cross  Abstract: Heterogeneous multi-robot collaboration over long horizons requires both global task planning and robust coordination during execution. Existing plan- ners using large language models (LLMs) can generate task decompositions or symbolic plans, but often execute them as rigid action sequences, making them fragile to local state mismatches and synchronization failures. We propose Em- boTeam, a neuro-symbolic framework for asymmetric state grounding. An LLM generates structured state expectations, which are verified and coordinated through symbolic search using Planning Domain Definition Language (PDDL) and then grounded into a team-level parallel behavior tree. Robot-specific subtrees per- form precondition checking, skill execution, effect validation, local replanning, and blackboard-based synchronization. We further introduce MACE-THOR, con- taining 42 heterogeneous multi-robot tasks across 8 indoor layouts. EmboTeam improves ta
    
[^372]: 帕累托DNN验证：对大多数查询快速完成

    Pareto DNN Verification: Fast for Most Queries

    [https://arxiv.org/abs/2512.20755](https://arxiv.org/abs/2512.20755)

    提出PAVE方法，通过为DNN添加早期退出机制并逐退出点验证，能以直接验证所需时间的一小部分解决约80%的鲁棒性查询，且准确率变化不超过0.3%。

    

    深度神经网络（DNN）的形式化验证面临一个根本性的可扩展性挑战：验证鲁棒性是NP难的，且最先进的工具在现代架构上经常超时。我们提出了帕累托验证方法（Pareto VErification，简称PAVE）：通过为DNN添加早期退出机制并逐个退出点进行验证，我们在直接验证所需时间的一小部分内解决了大部分鲁棒性查询，在我们的评估中约为80%。关键洞察在于，鲁棒性邻域中的许多输入会在较早的层就退出。验证这种早期退出行为通常是一个可处理的问题，使得验证可以在完整网络被分析之前很久就终止。早期退出机制可以在不修改基础网络的情况下添加，并且在我们测试的所有架构上最多只带来0.3%的准确率变化。我们为早期退出网络形式化了新颖的鲁棒性属性，并证明在轨迹稳定性条件下该问题是固定参数可解的。

    arXiv:2512.20755v2 Announce Type: replace-cross  Abstract: Formal verification of deep neural networks (DNNs) faces a fundamental scalability challenge: verifying robustness is NP-hard and state-of-the-art tools often time out on modern architectures. We propose PAreto VErification, or PAVE: by augmenting DNNs with early exits (EEs) and verifying exit-by-exit, we solve a large fraction of robustness queries---roughly 80% in our evaluation---in a small fraction of the time required for direct verification. The key insight is that many inputs in a robustness neighborhood exit at an early layer. Verifying this early-exit behavior is often a tractable problem,allowing verification to terminate long before the full network is analyzed. EEs can be added without modifying the base network, and incur at most a 0.3% accuracy change across our tested architectures. We formalize the novel robustness property for EE networks, prove it is Fixed-Parameter Tractable under a trace stability condition 
    
[^373]: 纠错扩散语言模型

    Corrective Diffusion Language Models

    [https://arxiv.org/abs/2512.15596](https://arxiv.org/abs/2512.15596)

    该论文提出一种面向纠错的后训练原则，通过显式监督可见的损坏词元，使掩码扩散语言模型能够对错误词元给出低置信度并进行有针对性的迭代修正，同时保留正确内容，其效果显著优于传统的掩码扩散微调。

    

    尽管扩散语言模型凭借其非因果结构在理论上非常适合迭代精炼，但现成的掩码扩散语言模型在实践中往往无法可靠地修正错误的词元。关键挑战在于模型无法区分可见词元中的正确与错误：标准的掩码扩散语言模型训练仅对被掩码的位置进行监督，从而削弱了基于置信度引导的精炼能力。我们研究了扩散语言模型中的纠错行为，即为错误词元赋予较低置信度并在保留正确内容的同时对其进行迭代精炼的能力。在相同的计算量和数据条件下，我们表明，传统的掩码扩散微调所诱导的这种纠错能力远低于一种面向纠错的后训练原则，后者显式地对可见的被破坏词元进行监督，从而实现有区分力的置信度和针对性的精炼。我们进一步发现，同时对干净的可见词元进行监督，正如……（摘要原文在此处被截断）

    arXiv:2512.15596v3 Announce Type: replace  Abstract: While Diffusion Language Models (DLMs) are theoretically well-suited for iterative refinement due to their non-causal structure, off-the-shelf masked diffusion language models (MDLMs) often fail to reliably revise incorrect tokens in practice. The key challenge is that the model cannot tell correct from erroneous visible tokens: standard MDLM training supervises only masked positions, undermining confidence-guided refinement. We study corrective behavior in DLMs, the ability to assign lower confidence to incorrect tokens and iteratively refine them while preserving correct content. Under matched compute and data, we show that conventional masked-diffusion fine-tuning induces much less of this capability than a correction-oriented post-training principle that explicitly supervises visible corrupted tokens, which enables discriminative confidence and targeted refinement. We further find that also supervising clean visible tokens, as in
    
[^374]: 面向优化器超参数选择的贪心对齐原理

    Greedy Alignment Principle for Optimizer Hyperparameter Selection

    [https://arxiv.org/abs/2512.06370](https://arxiv.org/abs/2512.06370)

    该论文提出了贪心对齐原理，将优化器选择形式化为最大化损失期望下降速率（等价于优化器滤波器与梯度自相关的内积），证明了贪心最优解的存在性与稳定性，并据此为 SGD+Momentum 和 Adam/AdamW 导出了简单的动态动量选择规则。

    

    近期的研究表明，梯度-更新对齐是调节优化器更新的一个强大信号，通常能够带来更快的训练速度。我们将这一逐更新的启发式方法提升为一个具有严格数学依据的原理，用于选择和调整优化器超参数。通过将梯度和更新视为信号，并将优化器视为在两者之间进行映射的因果滤波器，我们将优化器选择问题形式化为在一组给定的优化器族上最大化损失的期望下降速率。我们证明该目标恰好等于优化器滤波器与梯度自相关函数之间的内积，并证明了贪心最优解的存在性，且在估计的梯度统计量受到扰动时具有稳定性界。针对基于动量的优化器，该理论为 SGD+Momentum 和 Adam/AdamW 都推导出了简单的动态动量选择规则。在图像分类、语言模型微调等多项任务上的实验验证了该方法的有效性。

    arXiv:2512.06370v4 Announce Type: replace  Abstract: Recent works have shown that gradient-update alignment is a powerful signal for modulating optimizer updates, often leading to faster training. We promote this update-wise heuristic as a mathematically grounded principle for selecting and tuning optimizer hyperparameters. By treating gradients and updates as signals and an optimizer as a causal filter that maps between them, we formulate optimizer selection as maximizing the expected drop rate in loss over a prescribed family of optimizers. We show that this objective is exactly the inner product between the optimizer filter and the gradient autocorrelation, and prove that a greedy optimum exists and has a stability bound under perturbations of the estimated gradient statistics. Specializing in momentum-based optimizers, the theory yields simple dynamic momentum selection rules for both SGD+Momentum and Adam/AdamW. Experiments across image classification, language model fine-tuning, 
    
[^375]: 节点流行度模型的变分估计器

    Variational Estimators for Node Popularity Models

    [https://arxiv.org/abs/2511.17783](https://arxiv.org/abs/2511.17783)

    本文为双向节点流行度模型（TNPM）开发了一种计算高效且有理论保证的变分期望最大化（VEM）估计框架，在二部网络中证明了社区划分的标签一致性，并实现了优于现有方法的估计精度。

    

    节点流行度被认为是建模现实世界网络的一个关键因素，它刻画了不同社区之间连接的异质性。这一概念在二部网络中同样重要，其中不同分部中的节点可能表现出不同的流行度模式，由此催生了双向节点流行度模型（TNPM）等模型。现有方法，如两阶段分割余弦（TSDC）算法，提供了可扩展的估计方法，但在估计精度或对不同类型网络的适用性方面可能存在局限。在本文中，我们为TNPM开发了一个计算高效且有理论支撑的变分期望最大化（VEM）框架。我们在二部网络中为所提出的变分估计器产生的社区划分估计建立了标签一致性。通过大量的仿真研究，我们证明了我们的方法能够实现更优的估计精度。

    arXiv:2511.17783v2 Announce Type: replace-cross  Abstract: Node popularity is recognized as a key factor in modeling real-world networks, capturing heterogeneity in connectivity across communities. This concept is equally important in bipartite networks, where nodes in different partitions may exhibit varying popularity patterns, motivating models such as the Two-Way Node Popularity Model (TNPM). Existing methods, such as the Two-Stage Divided Cosine (TSDC) algorithm, provide a scalable estimation approach but may have limitations in terms of accuracy or applicability across different types of networks. In this paper, we develop a computationally efficient and theoretically justified variational expectation-maximization (VEM) framework for the TNPM. We establish label consistency for the estimated community assignments produced by the proposed variational estimator in bipartite networks. Through extensive simulation studies, we show that our method achieves superior estimation accuracy
    
[^376]: CCMP：面向高维特征选择的类条件多重共线性剪枝

    CCMP: Class-Conditional Multicollinearity Pruning for High-Dimensional Feature Selection

    [https://arxiv.org/abs/2511.12451](https://arxiv.org/abs/2511.12451)

    提出类条件多重共线性剪枝方法CCMP，按类别迭代地从高度相关特征对中剪除冗余特征，并给出剪枝误差的明确理论界限，从而实现无需整体重拟合的高维特征选择。

    

    在高维科学测量中，单个样本内的许多特征往往捕捉的是相同的底层物理属性，从而产生严重的多重共线性。监督式特征选择器在整个数据集上拟合，因此某个特征可能在整体上对分类具有信息量，但在同一类别内仍与另一个特征冗余。相比之下，类条件选择器以牺牲部分准确性为代价换取更窄的逐类视角，因为它们从不接触单个类别之外的数据，但这使得可以在不重新拟合整个模型的情况下添加新类别。然而，它们仍然无法确定应从相关特征组中删除哪个特征。为弥合这一差距，我们提出了类条件多重共线性剪枝（CCMP），它按类别迭代地从每个高度相关的特征对中剪除一个特征。四个理论保证在给定假设下为剪枝所引起的误差建立了明确的界限。在四个基准数据集上（原文在此处截断）……

    arXiv:2511.12451v2 Announce Type: replace-cross  Abstract: In high-dimensional scientific measurements, many features in a single sample often capture the same underlying physical attributes, producing severe multicollinearity. Supervised feature selectors fit on the entire dataset, so a feature can be informative for classification overall while still redundant with another feature in the same class. Class-conditional selectors, by contrast, trade some accuracy for a narrower, per-class view, since they never see data outside one class, but this lets new classes be added without refitting the whole model. They still cannot determine, however, which feature to drop from a correlated group. To close this gap, we propose classconditional multicollinearity pruning (CCMP), which iteratively prunes one feature from every highly correlated pair, per class. Four theoretical guarantees establish explicit bounds on the error induced by pruning under stated assumptions. Across four benchmark dat
    
[^377]: 基于专家先验的多目标超参数优化

    Multi-objective Hyperparameter Optimization with Expert Priors

    [https://arxiv.org/abs/2511.08371](https://arxiv.org/abs/2511.08371)

    提出首个能将专家先验知识融入多目标超参数优化的算法PriMO，在8个深度学习基准以及大规模视觉-语言模型的多目标优化中均达到最先进性能。

    

    尽管深度学习（DL）专家通常对哪些超参数设置能带来优异性能拥有先验知识，但只有少数超参数优化（HPO）算法能够利用这类先验知识，且没有任何算法能将先验知识融入多目标优化。由于深度学习从业者往往需要优化的目标不止一个而是多个，这构成了HPO算法领域的一个盲区。为弥补这一不足，我们提出了带专家先验的多目标HPO这一全新问题设定，并引入PriMO——首个能够在多个目标上整合用户信念的HPO算法。我们证明PriMO在8个深度学习基准上于多目标和单目标设定下均取得了最先进的性能。此外，我们通过在三个目标上优化一个视觉-语言模型，展示了PriMO在大规模场景下依然保持最先进水平。

    arXiv:2511.08371v2 Announce Type: replace  Abstract: While Deep Learning (DL) experts often have prior knowledge about which hyperparameter settings yield strong performance, only few Hyperparameter Optimization (HPO) algorithms can leverage such prior knowledge and none incorporate priors over multiple objectives. As DL practitioners often need to optimize not just one but many objectives, this is a blind spot in the algorithmic landscape of HPO. To address this shortcoming, we propose the novel problem setting of multi-objective HPO with expert priors and introduce PriMO, the first HPO algorithm that can integrate user beliefs over multiple objectives. We show PriMO achieves state-of-the-art performance across 8 DL benchmarks in the multi-objective and single-objective setting. Further, we demonstrate that PriMO remains state-of-the-art at large scales by optimizing a vision-language model over three objectives.
    
[^378]: 概率模型的深度神经算子学习

    Deep Neural Operator Learning for Probabilistic Models

    [https://arxiv.org/abs/2511.07235](https://arxiv.org/abs/2511.07235)

    该论文提出了一个适用于概率模型的深度神经算子框架，建立了具有显式网络规模界的通用逼近定理，并证明所学模型无需重新训练即可为新的执行价格生成美式期权的最优停止边界。

    

    我们为一大类概率模型提出了一个深度神经算子框架。在算子在整个欧几里得空间上满足全局Lipschitz条件的情况下——并且针对一大类概率模型——我们为所提出的架构建立了具有显式网络规模界的通用逼近定理。底层随机过程只需满足可积性和一般的尾概率条件。我们在前向-后向随机微分方程（FBSDE）框架内验证了这些假设对欧式和美式期权定价问题均成立，这进而涵盖了一大类由抛物型偏微分方程产生的算子，无论是否带有自由边界。最后，我们给出了一个美式期权组合的数值例子，表明所学习的模型无需重新训练即可为新的执行价格产生最优停止边界。

    arXiv:2511.07235v2 Announce Type: replace  Abstract: We propose a deep neural-operator framework for a general class of probabilistic models. Under global Lipschitz conditions on the operator over the entire Euclidean space-and for a broad class of probabilistic models-we establish a universal approximation theorem with explicit network-size bounds for the proposed architecture. The underlying stochastic processes are required only to satisfy integrability and general tail-probability conditions. We verify these assumptions for both European and American option-pricing problems within the forward-backward SDE (FBSDE) framework, which in turn covers a broad class of operators arising from parabolic PDEs, with or without free boundaries. Finally, we present a numerical example for a basket of American options, demonstrating that the learned model produces optimal stopping boundaries for new strike prices without retraining.
    
[^379]: 相关尖峰模型中的谱阈值与偏最小二乘法的基本极限

    Spectral Thresholds in Correlated Spiked Models and Fundamental Limits of Partial Least Squares

    [https://arxiv.org/abs/2510.17561](https://arxiv.org/abs/2510.17561)

    本文利用随机矩阵理论证明了部分对齐的尖峰互协方差模型中样本奇异值会经历BBP型相变，首次精确刻画了偏最小二乘法（PLS）的信号恢复能力，并揭示了PLS与贝叶斯最优估计器之间的根本性能差距，指出在某些信噪比和相关性区间内PLS会完全无法恢复信号。

    

    我们对尖峰互协方差模型提供了严格的随机矩阵理论分析，该模型中两个高维数据通道之间的信号是部分对齐的。这些模型的动机来自多模态学习，并构成了偏最小二乘法（PLS）——一种被广泛使用但理论研究不足的方法——背后的标准生成设定。我们证明了样本互协方差矩阵的领先奇异值会经历Baik-Ben Arous-Péché（BBP）型相变，并刻画了信息性成分出现的精确阈值。我们的结果首次对PLS在此设定下的信号恢复能力给出了精确的渐近描述，揭示了PLS与贝叶斯最优估计器之间的根本性能差距。特别地，我们识别出信噪比（SNR）和相关性所在的区间：尽管信号在原则上是可以被检测到的，但PLS在这些区间内却无法恢复任何信号。

    arXiv:2510.17561v2 Announce Type: replace-cross  Abstract: We provide a rigorous random matrix theory analysis of spiked cross-covariance models where the signals across two high-dimensional data channels are partially aligned. These models are motivated by multi-modal learning and form the standard generative setting underlying Partial Least Squares (PLS), a widely used yet theoretically underdeveloped method. We show that the leading singular values of the sample cross-covariance matrix undergo a Baik-Ben Arous-Peche (BBP)-type phase transition, and we characterize the precise thresholds for the emergence of informative components. Our results yield the first sharp asymptotic description of the signal recovery capabilities of PLS in this setting, revealing a fundamental performance gap between PLS and the Bayes-optimal estimator. In particular, we identify the SNR and correlation regimes where PLS fails to recover any signal, despite detectability being possible in principle. These f
    
[^380]: 使用医学分类数据对大语言模型进行指令微调可提升德语肿瘤诊断的ICD编码准确性

    Instruction-Tuning Large Language Models with Medical Classification Data Improves ICD Coding of German Tumor Diagnoses

    [https://arxiv.org/abs/2510.13624](https://arxiv.org/abs/2510.13624)

    本研究基于ICD-10-GM、ICD-O-3和OPS医学分类目录构建了超过50万个问答对，对Qwen、Llama和Mistral系列开源大语言模型进行指令微调，从而显著提升了德语肿瘤诊断文本的ICD编码准确性。

    

    使用ICD-10-GM和ICD-O-3对肿瘤诊断进行准确编码，对于德国结构化癌症文档记录至关重要。较小的开源权重大语言模型在隐私保护自动化方面具有吸引力，但在德语语境中往往难以达到编码准确性。本研究探讨了基于公开数据集的指令微调能否提高开源权重大语言模型对德语肿瘤诊断文本的编码准确性。评估采用来自本地肿瘤文档系统的已编码诊断作为测试数据。在系统性数据质量评估中，ICD-10编码性能的上限估计为：精确编码为60-79%，部分编码（仅三位字符编码）为81-94%。作为训练数据，研究者基于ICD-10-GM、ICD-O-3和OPS目录创建了超过50万个问答对，并对来自Qwen、Llama和Mistral系列的八个开源权重模型（7-70B参数）进行了微调。ICD-10-GM a…（原文摘要在此处截断）

    arXiv:2510.13624v2 Announce Type: replace-cross  Abstract: Accurate coding of tumor diagnoses with ICD-10-GM and ICD-O-3 is essential for structured cancer documentation in Germany. Smaller open-weight LLMs are appealing for privacy-preserving automation but often struggle with coding accuracy in German-language contexts. This study investigates whether instruction-based fine-tuning on public datasets improves the coding accuracy of open-weight LLMs for German tumor diagnosis texts. The evaluation uses coded diagnoses from the local tumor documentation system as test data. In a systematic data quality assessment, the upper limit for ICD-10 coding performance was estimated at 60-79% for exact and 81-94% for partial (three-character codes only) derivation. As training data, over 500,000 question-answer pairs were created based on the ICD-10-GM, ICD-O-3, and OPS catalogues. Eight open-weight models from the Qwen, Llama, and Mistral families (7-70 B parameters) were fine-tuned. ICD-10-GM a
    
[^381]: scTREAT：用于单细胞药物响应预测的潜在扩散模型

    scTREAT: Latent Diffusion for Single-Cell Drug-Response Prediction

    [https://arxiv.org/abs/2510.11726](https://arxiv.org/abs/2510.11726)

    scTREAT是首个将细胞状态与药物引导进行因式分解的潜在扩散框架，通过引导可分解注意力机制注入基线表示与剂量条件化的分子嵌入，在单细胞药物响应预测任务上取得了最先进的性能。

    

    生成药物诱导的转录响应需要一个模型能够同时考虑细胞背景、分子身份和剂量。我们提出scTREAT，这是首个将细胞状态与药物引导进行因式分解的扩散框架，用于单细胞药物响应预测。一个共享的潜在表达空间连接基线与响应，同时引导可分解注意力机制（GD-Attn）注入基线表示和剂量条件化的分子嵌入。训练期间通过独立的通道丢弃机制暴露四种条件状态；推理时结合三个条件预测，分别指定状态和药物系数，且剂量可调节药物引导的强度。在源自Tahoe-100M单细胞图谱的条件级伪批量数据上，scTREAT在未见细胞系-药物组合（UC）的12项指标中有10项取得最佳值，在未见药物（UD）的12项指标中有11项取得最佳值。消融实验考察了条件注入方

    arXiv:2510.11726v3 Announce Type: replace-cross  Abstract: Generating drug-induced transcriptional responses requires a model to account for cellular background together with molecular identity and dose. We present scTREAT, the first diffusion framework to factorize state and drug guidance for single-cell drug-response prediction. A shared latent expression space connects baseline and response, while Guided Decomposable Attention (GD-Attn) injects a baseline representation and a dose-conditioned molecular embedding. Independent channel dropout exposes four conditioning states during training; inference combines three conditional predictions with separately specified state and drug coefficients, and dose modulates drug-guidance strength. On condition-level pseudobulk profiles derived from the Tahoe-100M single-cell atlas, scTREAT achieves the best value on 10 of 12 metrics for unseen cell line-drug combinations (UC) and 11 of 12 for unseen drugs (UD). Ablations examine condition injecti
    
[^382]: ARMs：面向多模态模型的即插即用自适应红队测试智能体

    ARMs: Adaptive Red-Teaming Agent against Multimodal Models with Plug-and-Play Attacks

    [https://arxiv.org/abs/2510.02677](https://arxiv.org/abs/2510.02677)

    提出ARMs自适应红队测试智能体，通过推理增强的多步骤编排自动优化攻击策略，包含11种新颖多模态攻击手段并可即插即用集成17种红队测试算法，实现对视觉语言模型的全面自动化安全评估。

    

    随着视觉语言模型日益受到重视，其多模态接口也引入了新的安全漏洞，使得安全评估既充满挑战又至关重要。现有的红队测试工作要么局限于狭窄的对抗模式集合，要么严重依赖人工工程，缺乏对现实世界中新兴VLM漏洞的可扩展探索。为弥补这一空白，我们提出了ARMs，一个能够对VLM进行系统性全面风险评估的自适应红队测试智能体。给定目标有害行为或风险定义，ARMs通过推理增强的多步骤编排自动优化多样化的红队测试策略，以有效诱导目标VLM产生有害输出。我们提出了11种新颖的多模态攻击策略，涵盖VLM的多种对抗模式（如推理劫持、上下文伪装），并通过模型上下文以即插即用的方式将17种红队测试算法集成到ARMs中。

    arXiv:2510.02677v2 Announce Type: replace  Abstract: As vision-language models (VLMs) gain prominence, their multimodal interfaces also introduce new safety vulnerabilities, making the safety evaluation challenging and critical. Existing red-teaming efforts are either restricted to a narrow set of adversarial patterns or depend heavily on manual engineering, lacking scalable exploration of emerging real-world VLM vulnerabilities. To bridge this gap, we propose ARMs, an adaptive red-teaming agent that systematically conducts comprehensive risk assessments for VLMs. Given a target harmful behavior or risk definition, ARMs automatically optimizes diverse red-teaming strategies with reasoning-enhanced multi-step orchestration, to effectively elicit harmful outputs from target VLMs. We propose 11 novel multimodal attack strategies, covering diverse adversarial patterns of VLMs (e.g., reasoning hijacking, contextual cloaking), and integrate 17 red-teaming algorithms into ARMs via model conte
    
[^383]: HomID：在具有各向异性嵌入的同质流形上对内在维度估计器进行基准测试

    HomID : Benchmarking Intrinsic Dimension Estimators on Homogenous Manifolds with Anisotropic Embeddings

    [https://arxiv.org/abs/2510.01335](https://arxiv.org/abs/2510.01335)

    本文提出了 HomID 基准——一组具有各向异性嵌入的同质流形集合，用以揭示现有内在维度估计器在标准基准上表现良好、但在各向异性条件下会系统性失效，并证明了各向异性畸变通过使其所依赖的分布发生系统性偏移而导致估计误差。

    

    流形假说认为，数据位于内在维度低于其环境维度的流形之上。然而，对于现实数据集，不同内在维度估计器给出的估计值之间在经验上并不一致。因此，用有针对性的压力测试来检验内在维度估计器是十分重要的。在这项工作中，我们考察了各向异性的作用。为此，我们提出了 HomID——一组具有各向异性嵌入的同质空间集合，用于对内在维度估计器进行基准测试。我们观察到，在标准基准测试中表现良好的方法，在相同的资源分配下，在 HomID 上会系统性地出现性能下降。我们进一步观察到，对这类基准进行各向异性畸变同样会导致性能退化。最后，我们证明，受控的各向异性畸变会在这些方法所依赖的分布上引起系统性偏移，从而为由此产生的估计误差提供了一个具体的机制。

    arXiv:2510.01335v2 Announce Type: replace  Abstract: The manifold hypothesis suggests that data lies on manifolds with smaller intrinsic dimension (ID) than their ambient dimension. However there is no empirical agreement on the estimates for ID from different estimators for realistic datasets. Thus it is important to test ID estimators (IDEs) with targeted stressors. In this work, we consider the role of anisotropy. To this end, we propose HomID, a collection of homogeneous spaces with anisotropic embedding, for benchmarking ID estimators. We observe that methods that perform well on standard benchmarks systematically degrade on HomID under identical resource allocation. We further observe that anisotropic distortion of such benchmarks also results in performance degradation. Finally, we demonstrate that controlled anisotropic distortions induce systematic shifts in the distributions on which these methods rely, providing a concrete mechanism for the resulting estimation errors in two
    
[^384]: 用于卷积神经网络高效分布式训练的混合双批次与循环渐进学习方法

    Hybrid Dual-Batch and Cyclic Progressive Learning for Efficient Distributed Training of Convolutional Neural Networks

    [https://arxiv.org/abs/2509.26092](https://arxiv.org/abs/2509.26092)

    提出了一种基于参数服务器框架的双批次学习方案，通过同时使用硬件支持的最大批次和较小批次进行训练，在保持高效的同时显著提升模型的泛化能力和准确率。

    

    分布式深度学习对于在具有大量参数的大规模图像数据集上训练卷积神经网络（CNN）模型至关重要。当前的研究主要侧重于利用额外的硬件资源和强大的计算单元来加速训练过程。因此，通常采用较大的批次大小来加快训练速度。然而，使用大批次进行训练可能由于泛化能力差而导致准确率下降。为了解决这一问题，我们提出了双批次学习方案，这是一种建立在参数服务器框架之上的分布式训练方法。该方法通过利用硬件所能支持的最大批次大小来最大化训练效率，同时引入较小的批次大小以增强模型的泛化能力。通过同时使用两种不同的批次大小，该方法能以极少的额外训练时间提升模型准确率。此外，为了减少时间开销……（原文摘要在此处被截断）

    arXiv:2509.26092v3 Announce Type: replace-cross  Abstract: Distributed deep learning is critical for training convolutional neural network (CNN) models on large-scale image datasets with numerous parameters. Current research primarily focuses on leveraging additional hardware resources and powerful computing units to accelerate the training process. As a result, larger batch sizes are often employed to speed up training. However, training with large batch sizes can lead to lower accuracy due to poor generalization. To address this issue, we propose the dual-batch learning scheme, a distributed training method built on the parameter server framework. This approach maximizes training efficiency by utilizing the largest batch size that the hardware can support while incorporating a smaller batch size to enhance model generalization. By using two different batch sizes simultaneously, this method improves accuracy with minimal additional training time. Additionally, to mitigate the time ove
    
[^385]: 用于改进黑盒优化求解器的机器学习算法

    Machine Learning Algorithms for Improving Black Box Optimization Solvers

    [https://arxiv.org/abs/2509.25592](https://arxiv.org/abs/2509.25592)

    这是一篇综述论文，系统总结了机器学习与强化学习如何增强黑盒优化求解器，涵盖表达力更强的代理模型、元学习算法选择、动态算子配置与跨任务元优化等最新进展及其代表性算法。

    

    黑盒优化（BBO）处理的是目标函数只能通过昂贵的查询来获取、且没有梯度或显式结构信息的问题。经典的无导数方法——线搜索、直接搜索以及基于模型的求解器（如贝叶斯优化）——构成了BBO的基石，但在高维、含噪声或混合整数等设置中往往表现不佳。近年来的进展利用机器学习（ML）和强化学习（RL）来增强BBO：机器学习提供了表达力更强的代理模型、自适应更新、基于元学习的算法选择以及生成模型，而强化学习与BBO的交叉领域则包括动态算子配置、鲁棒的策略搜索以及跨任务的元优化。本文对这些发展进行了综述，涵盖了代表性的算法，如模块化基于模型的优化框架、零阶自适应动量方法（ZO-AdaMM）、自动化黑盒优化（ABBO）、分布式分块优化等。

    arXiv:2509.25592v2 Announce Type: replace  Abstract: Black-box optimization (BBO) addresses problems where objectives are accessible only through costly queries without gradients or explicit structure. Classical derivative-free methods---line search, direct search, and model-based solvers such as Bayesian optimization---form the backbone of BBO, yet often struggle in high-dimensional, noisy, or mixed-integer settings. Recent advances use machine learning (ML) and reinforcement learning (RL) to enhance BBO: ML provides expressive surrogates, adaptive updates, meta-learning-based algorithm selection, and generative models, while the RL--BBO interface includes dynamic operator configuration, robust policy search, and meta-optimization across tasks. This paper surveys these developments, covering representative algorithms such as the modular model-based optimization framework (mlrMBO), zeroth-order adaptive momentum methods (ZO-AdaMM), automated BBO (ABBO), distributed block-wise optimizat
    
[^386]: 一种用于增强眼底图像病灶分割的混合方法

    Hybrid Approach for Enhancing Lesion Segmentation in Fundus Images

    [https://arxiv.org/abs/2509.25549](https://arxiv.org/abs/2509.25549)

    本文提出一种混合方法，将深度学习模型与传统数学/聚类分割方法相结合，以克服眼底图像数据集分辨率低、标注不一致以及标注数据不足的问题，实现脉络膜痣等眼底病变的精确分割，助力早期诊断工具的开发。

    

    脉络膜痣是眼部常见的良性色素性病变，有较小的风险转化为黑色素瘤。早期检测对于提高生存率至关重要，而误诊或延迟诊断可能导致不良后果。尽管基于AI的图像分析技术不断进步，但在彩色眼底图像中诊断脉络膜痣仍然具有挑战性，特别是对于缺乏专业知识的临床医生而言。现有数据集通常存在分辨率低和标注不一致的问题，限制了分割模型的有效性。本文旨在解决眼底病变精确分割这一挑战，这是开发稳健诊断工具的关键步骤。虽然像U-Net这样的深度学习模型已展现出有效性，但其准确性在很大程度上依赖于标注数据的质量和数量。以往的数学/聚类分割方法虽然准确，但需要大量的人工输入。

    arXiv:2509.25549v3 Announce Type: replace-cross  Abstract: Choroidal nevi are common benign pigmented lesions in the eye, with a small risk of transforming into melanoma. Early detection is critical to improving survival rates, but misdiagnosis or delayed diagnosis can lead to poor outcomes. Despite advancements in AI-based image analysis, diagnosing choroidal nevi in colour fundus images remains challenging, particularly for clinicians without specialized expertise. Existing datasets often suffer from low resolution and inconsistent labelling, limiting the effectiveness of segmentation models. This paper addresses the challenge of achieving precise segmentation of fundus lesions, a critical step toward developing robust diagnostic tools. While deep learning models like U-Net have demonstrated effectiveness, their accuracy heavily depends on the quality and quantity of annotated data. Previous mathematical/clustering segmentation methods, though accurate, required extensive human input
    
[^387]: 多面体博弈中超越完全信息的高效核化学习：从布洛托上校博弈到拥塞博弈

    Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games

    [https://arxiv.org/abs/2509.20919](https://arxiv.org/abs/2509.20919)

    该论文提出了一个基于核化范式的新框架，首次在部分信息环境下实现了多面体博弈（如布洛托上校博弈和拥塞博弈）中粗相关均衡的高效学习，使遗憾度与单次迭代复杂度随动作集规模呈多对数级增长。

    

    我们研究了在多面体博弈中高效学习粗相关均衡（CCE）的问题。多面体博弈是指每个玩家拥有指数级数量动作且具有潜在组合结构的标准型博弈，其典型代表是经典的布洛托上校博弈和拥塞博弈。为实现计算效率，学习算法必须保证遗憾度与单次迭代复杂度随玩家动作集规模呈多对数级增长。这一挑战最近已在完全信息环境下得到解决，主要依靠核化技术。然而，在更贴近现实但数学上更具挑战性的部分信息环境下，现有的粗相关均衡学习方法会导致次优且不切实际的运行时间复杂度。我们通过构建一个基于核化范式的框架来解决这一局限，并将该框架应用于典型的多面体博弈实例……

    arXiv:2509.20919v2 Announce Type: cross  Abstract: We examine the problem of efficiently learning coarse correlated equilibria (CCE) in polyhedral games, that is, normal-form games with an exponentially large number of actions per player and an underlying combinatorial structure. Prominent examples of such games are the classical Colonel Blotto and congestion games. To achieve computational efficiency, the learning algorithms must exhibit regret and per-iteration complexity that scale polylogarithmically in the size of the players' action sets. This challenge has recently been addressed in the full-information setting, primarily through the use of kernelization. However, in the case of the realistic, but mathematically challenging, partial-information setting, existing approaches result in suboptimal and impractical runtime complexity to learn CCE. We tackle this limitation by building a framework based on the kernelization paradigm. We apply this framework to prominent examples of pol
    
[^388]: 关于在线线性回归问题的高斯近似速率研究

    On the Rate of Gaussian Approximation for Online Linear Regression Problems

    [https://arxiv.org/abs/2509.14039](https://arxiv.org/abs/2509.14039)

    本文针对恒定步长下的在线线性回归问题推导了高斯近似速率，显式刻画了其对问题维度和设计矩阵相关量的依赖，并证明当样本量足够大时可获得 $\sqrt{\log{n}/n}$ 阶的正态近似速率。

    

    本文研究了在线线性回归任务中的高斯近似问题。我们在恒定步长的设定下推导了相应的收敛速率，并研究了该收敛速率对问题维度 $d$ 以及与设计矩阵相关量的显式依赖关系。当迭代次数 $n$ 事先已知时，只要样本量 $n$ 足够大，我们的结果给出了阶为 $\sqrt{\log{n}/n}$ 的正态近似速率。

    arXiv:2509.14039v2 Announce Type: replace-cross  Abstract: In this paper, we consider the problem of Gaussian approximation for the online linear regression task. We derive the corresponding rates for the setting of a constant stepsize and study the explicit dependence of the convergence rate on the problem dimension $d$ and quantities related to the design matrix. When the number of iterations $n$ is known in advance, our results yield a normal approximation rate of order $\sqrt{\log{n}/n}$, provided that the sample size $n$ is large enough.
    
[^389]: Genome-Factory：一个用于调优、部署和解释基因组基础模型的库

    Genome-Factory: A Library for Tuning, Deploying, and Interpreting Genomic Foundation Models

    [https://arxiv.org/abs/2509.12266](https://arxiv.org/abs/2509.12266)

    Genome-Factory是首个统一了基因组基础模型数据收集、微调、推理、基准测试和可解释性分析的集成Python库，并推出了基于稀疏自编码器的开源生物解释器。

    

    我们推出了Genome-Factory，这是首个用于调优、部署和解释基因组基础模型的集成化Python库。我们的核心贡献是简化和统一基因组模型开发的工作流程，涵盖数据收集、模型调优、推理、基准测试和可解释性五个环节。在数据收集方面，Genome-Factory提供了自动化流程来下载和预处理基因组序列；在模型调优方面，它支持跨多种基因组模型的全量微调和参数高效微调；在推理方面，它支持嵌入提取和DNA序列生成；在基准测试方面，我们集成了两个现有基准测试，并提供了灵活的接口以纳入更多基准测试；在可解释性方面，Genome-Factory引入了基于稀疏自编码器的开源生物解释器。我们从三个维度验证了Genome-Factory的实用性：（i）

    arXiv:2509.12266v3 Announce Type: replace-cross  Abstract: We introduce Genome-Factory, the first integrated Python library for tuning, deploying, and interpreting genomic foundation models. Our core contribution is to simplify and unify the workflow for genomic model development: data collection, model tuning, inference, benchmarking, and interpretability. For data collection, Genome-Factory offers an automated pipeline to download genomic sequences and preprocess them. For model tuning, Genome-Factory supports both full and parameter-efficient fine-tuning across diverse genomic models. For inference, Genome-Factory enables both embedding extraction and DNA sequence generation. For benchmarking, we include two existing benchmarks and provide a flexible interface to incorporate additional benchmarks. For interpretability, Genome-Factory introduces an open-source biological interpreter based on a sparse auto-encoder. We validate the utility of Genome-Factory across three dimensions: (i)
    
[^390]: 基于删失半老虎机反馈的在线预算分配

    Online Budget Allocation with Censored Semi-Bandit Feedback

    [https://arxiv.org/abs/2508.05844](https://arxiv.org/abs/2508.05844)

    本文研究了删失半老虎机反馈下的在线预算分配问题，提出了一种基于乐观原则的算法，并证明在收益递减情形下其遗憾随时间范围呈多对数级增长。

    

    我们研究了跨 K 个任务的随机预算分配问题。在每一轮 t，学习者选择一个分配方案 X_t ∈ Δ_K。任务 k 以概率 F_k(X_{t,k}) 成功，其中 F_1,...,F_K 是非递减的预算-成功曲线，任务成功时会产生一个均值为未知数 μ_k 的随机奖励。学习者可以观察到哪些任务成功了，且只有当任务成功时才能观察到该任务的奖励（删失半老虎机反馈）。该模型可以刻画诸如将支付分配给众包工作者或将出价分配给同时进行的多个拍卖等场景，并且涵盖了随机多臂老虎机和半老虎机问题。我们设计了一种基于乐观原则的算法，该算法能够在删失半老虎机反馈下运行，并假设可以使用一个精确的最大化 oracle。我们的主要结果是一个依赖于实例的界，表明在收益递减的情形下，该算法的遗憾随时间范围呈多对数级增长。

    arXiv:2508.05844v3 Announce Type: replace-cross  Abstract: We study a stochastic budget-allocation problem over $K$ tasks. At each round $t$, the learner chooses an allocation $X_t \in \Delta_K$. Task $k$ succeeds with probability $F_k(X_{t,k})$, where $F_1,\dots,F_K$ are nondecreasing budget-to-success curves, and upon success yields a random reward with unknown mean $\mu_k$. The learner observes which tasks succeed, and observes a task's reward only upon success (censored semi-bandit feedback). This model captures, for instance, splitting payments across crowdsourcing workers or distributing bids across simultaneous auctions, and subsumes stochastic multi-armed bandits and semi-bandits. We design an optimism-based algorithm that operates under censored semi-bandit feedback and assumes access to an exact maximization oracle. Our main result is an instance-dependent bound showing that in diminishing-returns regimes, the regret of this algorithm scales polylogarithmically with the horiz
    
[^391]: CoMPAS3D：面向交互式运动的数据集与基准

    CoMPAS3D: A Dataset and Benchmark for Interactive Motion

    [https://arxiv.org/abs/2507.19684](https://arxiv.org/abs/2507.19684)

    该论文提出将具有规范动作词汇和社区公认评判标准的萨尔萨舞作为具身交互评估领域，并构建了CoMPAS3D数据集与基准，以评估交互式动作生成中反应性动作的语义有效性与恰当性。

    

    社交互动型人形机器人旨在通过身体与人类进行交互，并实时适应伙伴的运动、意图和能力。这需要模型既能理解身体如何运动，又能理解这些运动所表达的含义。然而，现有的交互式动作生成评估框架并未评估所生成的反应性动作在共享动作词汇体系中是否有效或恰当：现有框架依赖FID、节拍对齐等运动学指标，无法捕捉这一语义属性，而即兴互动数据集又缺乏细粒度的动作标注。受围棋、国际象棋等游戏计算化评估的启发，我们提出将萨尔萨舞作为一个具身交互评估领域。萨尔萨舞具有交互性和双人配对性，并受动作词汇以及社区公认的评判标准（涵盖时机、音乐性、技术、难度、配合与原创性）的约束。我们预……（原文摘要在此处截断）

    arXiv:2507.19684v3 Announce Type: replace-cross  Abstract: Socially interactive humanoids aim to interact with humans through their bodies, adapting in real time to a partner's movement, intent, and abilities. This requires models that understand both how bodies move and what their movements mean. Yet, existing evaluation frameworks for interactive motion generation do not assess whether generated reactive motions are valid or appropriate within a shared movement vocabulary. Existing frameworks rely on kinematic metrics such as FID and beat alignment that cannot capture this semantic property, while improvised interaction datasets lack fine-grained move annotations. Inspired by computational evaluations of games such as Go and Chess, we propose Salsa as an embodied interaction evaluation domain. Salsa is interactive, dyadic, and governed by a move vocabulary and community-validated judging criteria covering timing, musicality, technique, difficulty, partnering, and originality. We pres
    
[^392]: 一种基于动态模态分解的神经算子

    A Neural Operator based on Dynamic Mode Decomposition

    [https://arxiv.org/abs/2507.01117](https://arxiv.org/abs/2507.01117)

    本文提出一种基于动态模态分解（DMD）的神经算子，通过结合DMD与深度学习自动提取关键模态和系统动力学，在热传导方程、拉普拉斯方程和Burgers方程求解上相比DeepONet和FNO等同类方法实现了更轻量高效的时空建模。

    

    科学计算方法与人工智能技术的结合发展仍然是一个热门研究课题，在轻量化与精确计算之间找到平衡是这一方向的坚实基础。本研究提出了一种基于动态模态分解算法（DMD）的神经算子，用于映射函数空间，该方法结合了DMD与深度学习（DL），可对时空过程进行高效建模。求解不同初始和边界条件下的偏微分方程（PDE）需要大量计算资源，而所提出的方法能够自动提取关键模态和系统动力学，并利用它们构建预测，与传统数值方法相比降低了计算成本。通过与最接近的同类方法DeepONet和FNO在热传导方程、拉普拉斯方程和Burgers方程解的逼近上进行性能对比分析，该方法已证明了其有效性。

    arXiv:2507.01117v2 Announce Type: replace  Abstract: The scientific computation methods development in conjunction with artificial intelligence technologies remains a hot research topic. Finding a balance between lightweight and accurate computations is a solid foundation for this direction. The study presents a neural operator based on the dynamic mode decomposition algorithm (DMD), mapping functional spaces, which combines DMD and deep learning (DL) for spatiotemporal processes efficient modeling. Solving PDEs for various initial and boundary conditions requires significant computational resources. The method suggested automatically extracts key modes and system dynamics using them to construct predictions, reducing computational costs compared to traditional numerical methods. The approach has demonstrated its efficiency through comparative analysis of performance with closest analogues DeepONet and FNO in the heat equation, Laplaces equation, and Burgers equation solutions approxim
    
[^393]: 在集成NPU的MCXN947微控制器上实现关键词唤醒

    Implementing Keyword Spotting on the MCXN947 Microcontroller with Integrated NPU

    [https://arxiv.org/abs/2506.08911](https://arxiv.org/abs/2506.08911)

    本文在集成NPU的MCXN947微控制器上实现了关键词唤醒系统，通过量化感知训练优化MFCC+CNN模型，相比仅CPU执行实现59倍推理加速，以仅30.58 KB的模型大小达到97.06%的准确率。

    

    本文介绍了一个在NXP MCXN947微控制器上实现的关键词唤醒（KWS）系统，该微控制器集成了神经网络处理单元（NPU），能够在资源受限的设备上实现实时语音交互。该系统将MFCC特征提取与CNN分类器相结合，并采用量化感知训练进行优化，以在精度损失极小的情况下减小模型大小。实验结果表明，与仅使用CPU执行相比，利用NPU可将推理时间加速59倍，在模型大小仅为30.58 KB的情况下达到了97.06%的准确率，证明了在嵌入式平台上实现高效、低功耗语音接口的可行性。

    arXiv:2506.08911v2 Announce Type: replace-cross  Abstract: This paper presents a keyword spotting (KWS) system implemented on the NXP MCXN947 microcontroller with an integrated Neural Processing Unit (NPU), enabling real-time voice interaction on resource-constrained devices. The system combines MFCC feature extraction with a CNN classifier, optimized using Quantization Aware Training to reduce model size with minimal accuracy drop. Experimental results demonstrate a 59x speedup in inference time when leveraging the NPU compared to CPU-only execution, achieving 97.06% accuracy with a model size of 30.58 KB, demonstrating the feasibility of efficient, low-power voice interfaces on embedded platforms.
    
[^394]: Fisher-Rao几何下的近端策略优化（PPO）

    PPO in the Fisher-Rao geometry

    [https://arxiv.org/abs/2506.03757](https://arxiv.org/abs/2506.03757)

    该论文利用Fisher-Rao几何提出了FR-PPO算法，在可作为PPO直接替换并取得相当性能的同时，提供了单调策略改进和次线性收敛等强理论保证。

    

    近端策略优化（PPO）因其出色的实证性能而在强化学习中被广泛使用，但它缺乏对策略改进和收敛性的形式化保证。PPO的裁剪代理目标函数源于在平坦几何设置下对价值函数线性化的下界。我们通过利用Fisher-Rao（FR）几何，推导出一个更紧的代理目标函数，并由此提出了Fisher-Rao PPO（FR-PPO）。我们的方案提供了强有力的理论保证，包括单调策略改进。在直接参数化设置下，我们证明FR-PPO能够实现次线性收敛；对于参数化策略，我们进一步获得了直至兼容函数逼近误差的次线性收敛结果。最后，尽管本文的主要关注点在于理论，我们还通过实验证明FR-PPO可以作为PPO的直接替换方案，并在一系列任务上取得与PPO相当的性能。

    arXiv:2506.03757v3 Announce Type: replace  Abstract: Proximal Policy Optimization (PPO) is widely used in reinforcement learning due to its strong empirical performance, yet it lacks formal guarantees for policy improvement and convergence. PPO's clipped surrogate objective is motivated by a lower bound on linearization of the value function in flat geometry setting. We derive a tighter surrogate objective and introduce Fisher-Rao PPO (FR-PPO) by leveraging the Fisher-Rao (FR) geometry. Our scheme provides strong theoretical guarantees, including monotonic policy improvement. In the direct parametrization setting, we show that FR-PPO achieves sub-linear convergence, and for parametrized policies we further obtain sub-linear convergence up to the compatible function approximation error. Finally, although our primary focus is theoretical, we also demonstrate empirically that FR-PPO can be implemented as a drop-in replacement for PPO and achieves performance comparable to PPO across a ran
    
[^395]: B-XAIC数据集：基于化学数据对图神经网络的可解释人工智能进行基准测试

    B-XAIC Dataset: Benchmarking Explainable AI for Graph Neural Networks Using Chemical Data

    [https://arxiv.org/abs/2505.22252](https://arxiv.org/abs/2505.22252)

    该论文提出了基于真实化学数据构建的B-XAIC基准数据集，通过具有已知真实依据的多样化分子任务，系统评估并揭示了现有图神经网络可解释AI方法的局限性。

    

    在化学信息学和药物发现领域，理解深度学习模型预测背后的推理过程至关重要，因为分子设计决定了其性质。然而，当前该领域的可解释人工智能（XAI）评估框架往往依赖人工数据集或简化的任务，所采用的数据衍生指标无法捕捉真实世界场景的复杂性，且与解释的忠实度缺乏直接关联。为解决这一问题，我们提出了B-XAIC，这是一个基于真实分子数据和多样化任务构建的新型基准，这些任务具有已知的标签分配真实依据。通过使用B-XAIC进行的全面评估，我们揭示了现有图神经网络（GNN）可解释AI方法在分子领域的局限性。该基准为深入理解XAI的忠实度提供了宝贵资源，有助于开发更可靠、更可解释的模型。

    arXiv:2505.22252v2 Announce Type: replace  Abstract: Understanding the reasoning behind deep learning model predictions is crucial in cheminformatics and drug discovery, where molecular design determines their properties. However, current evaluation frameworks for Explainable AI (XAI) in this domain often rely on artificial datasets or simplified tasks, employing data-derived metrics that fail to capture the complexity of real-world scenarios and lack a direct link to explanation faithfulness. To address this, we introduce B-XAIC, a novel benchmark constructed from real-world molecular data and diverse tasks with known ground-truth rationales for assigned labels. Through a comprehensive evaluation using B-XAIC, we reveal limitations of existing XAI methods for Graph Neural Networks (GNNs) in the molecular domain. This benchmark provides a valuable resource for gaining deeper insights into the faithfulness of XAI, facilitating the development of more reliable and interpretable models.
    
[^396]: CubeDAgger：面向动态系统的、具有高效且低风险交互的交互式模仿学习

    CubeDAgger: Interactive Imitation Learning for Dynamic Systems with Efficient yet Low-risk Interaction

    [https://arxiv.org/abs/2505.04897](https://arxiv.org/abs/2505.04897)

    本文提出CubeDAgger方法，通过阈值正则化激活、多动作候选最优共识机制和自回归有色噪声注入三项改进，实现了动态系统中高效且低风险的交互式模仿学习，在提升策略鲁棒性的同时显著减少动态稳定性违规。

    

    交互式模仿学习通过来自专家的逐步监督使智能体的控制策略更加鲁棒。近期的算法大多采用专家-智能体切换系统，通过有限地选择监督时机来减轻专家的负担。然而，这种方法仅对静态任务有效；在动态任务中，时机的差异会导致动作的突变，使机器人失去动态稳定性。因此，本文提出了一种名为CubeDAgger的新方法，即使在动态任务中，该方法也能以更少的动态稳定性违规来提升鲁棒性。所提方法基于EnsembleDAgger基线设计，包含三项改进：第一，添加正则化以显式激活用于决定监督时机的阈值；第二，将专家-智能体切换系统转换为多个动作候选的最优共识系统；第三，向（策略中）注入自回归有色噪声。

    arXiv:2505.04897v3 Announce Type: replace-cross  Abstract: Interactive imitation learning makes an agent's control policy robust by stepwise supervisions from an expert. The recent algorithms mostly employ expert-agent switching systems to reduce the expert's burden by limitedly selecting the supervision timing. However, this approach is useful only for static tasks; in dynamic tasks, timing discrepancies cause abrupt changes in actions, losing the robot's dynamic stability. This paper therefore proposes a novel method, named CubeDAgger, which improves robustness with less dynamic stability violations even for dynamic tasks. The proposed method is designed on a baseline, EnsembleDAgger, with three improvements. The first adds a regularization to explicitly activate the threshold for deciding the supervision timing. The second transforms the expert-agent switching system to an optimal consensus system of multiple action candidates. Third, autoregressive colored noise is injected to the 
    
[^397]: 如何对图像知识蒸馏进行后门攻击

    How to Backdoor Image Knowledge Distillation

    [https://arxiv.org/abs/2504.21323](https://arxiv.org/abs/2504.21323)

    本文提出了一种针对图像知识蒸馏的新型后门攻击方法，通过向蒸馏数据集注入被干净教师模型预测为目标标签的触发图像，使学生在教师保持干净的情况下学到后门，且在10%中毒率下仍然有效。

    

    知识蒸馏被广泛用于将大型教师模型的行为迁移到较小的学生模型。当教师模型是干净的时，知识蒸馏通常被认为是安全的，因为经典的后门攻击依赖于监督训练中的中毒标签和触发器，而蒸馏则是训练学生模型去匹配教师模型的输出。我们表明，当蒸馏数据集本身被投毒时，这一假设可能会失效。我们的攻击注入带有触发器且经过操纵的图像，这些图像会被干净的教师模型预测为攻击者选定的目标标签，从而使学生在教师模型不受影响的情况下学到后门。我们在多种操纵策略下评估了这一威胁，包括针对性的对抗扰动和基于GAN的针对性类别转换，并研究了蒸馏设置如何同时影响准确率和攻击成功率。我们表明，即使在10%的中毒率下，该攻击仍然有效。

    arXiv:2504.21323v3 Announce Type: replace-cross  Abstract: Knowledge distillation is widely used to transfer behavior from a large teacher model to a smaller student. It is often assumed to be safe when the teacher is clean, because classic backdoor attacks rely on poisoned labels and triggers in supervised training, whereas distillation trains the student to match a teacher's outputs. We show that this assumption can fail when the distillation dataset itself is poisoned. Our attack injects triggered and manipulated images that a clean teacher already predicts as an attacker chosen target label, which causes the student to learn a backdoor even though the teacher remains unaffected. We evaluate this threat across multiple manipulation strategies, including targeted adversarial perturbations and targeted GAN based class transitions, and study how distillation settings influence both accuracy and attack success. We show that the attack remains effective even at a 10% poisoning rates. The
    
[^398]: 解析地下水压力：一个可跨时空泛化的模型揭示孟加拉国地下水位与补给量同步下降

    Unraveling Groundwater Stress: A Model Generalizable Across Space and Time Reveals Co-Declining Water Tables and Recharge Across Bangladesh

    [https://arxiv.org/abs/2503.22771](https://arxiv.org/abs/2503.22771)

    该研究提出一种可跨时空泛化的“教师-学生”机器学习框架，将稀疏实地观测与GLDAS卫星数据结合，生成孟加拉国2公里高分辨率地下水埋深图，并揭示了地下水位与地下水补给量正在同步下降。

    

    孟加拉国所依赖的地下水是否正在得到补给？在孟加拉国乃至全球范围内，回答这一问题仍然困难，因为稀疏的实地观测限制了观测能力，而现有的机器学习方法难以在时间和空间上实现泛化。为突破这一瓶颈，我们开发了一个“教师-学生”框架：教师模型从稀疏的年度实地最小（R² = 0.86）和最大（R² = 0.96）地下水埋深观测中学习，生成密集的空间模式；学生模型利用特定位置的特征，将这些模式从随时间变化的25公里分辨率GLDAS地下水储量数据迁移到高分辨率（2公里）的地下水埋深图中。学生模型在2003-2022年间达到R² = 0.93，并能在未见过的年份（留一年交叉验证R² = 0.93）和地点（留一地区交叉验证R² = 0.70）上实现泛化，从而无需同期实地观测即可完成地下水重建。重建结果揭示了广泛存在的……（原文摘要在此处被截断）

    arXiv:2503.22771v2 Announce Type: replace-cross  Abstract: Is Bangladesh replenishing the groundwater it depends on? Answering this question remains difficult in Bangladesh and worldwide because sparse in-situ measurements limit observational capabilities, while existing machine-learning approaches struggle to generalize across time and space. To address this bottleneck, we develop a Teacher-Student framework in which a Teacher accurately learns from sparse annual in-situ minimum (R2 = 0.86) and maximum (R2 = 0.96) depth observations to generate dense spatial patterns, and a Student transfers these patterns from time-varying 25-km GLDAS groundwater storage to high-resolution (2-km) depth maps using location-specific features. The Student achieves R2 = 0.93 over 2003-2022 and generalizes across unseen years (Leave-One-Year-Out R2 = 0.93) and locations (Leave-One-District-Out R2 = 0.70), enabling reconstruction without contemporary in-situ measurements. The reconstruction reveals widespr
    
[^399]: 朗之万蒙特卡洛方法可在任意网络规模和任意数据下可证明地学习两层神经网络

    Langevin Monte-Carlo Provably Learns Depth Two Neural Nets at Any Size and Data

    [https://arxiv.org/abs/2503.10428](https://arxiv.org/abs/2503.10428)

    本文证明了朗之万蒙特卡洛算法在任意网络规模和任意训练数据下都能可证明地收敛并学习两层神经网络，所需正则化量与网络规模无关，收敛速率仅取决于Poincaré常数。

    

    在这篇笔记中，我们证明了朗之万蒙特卡洛算法能够最小化经过适当正则化的深度为2的神经网络的总体风险，且对网络的规模没有任何限制，也适用于任意选择的训练数据——并给出了其非渐近收敛速率。我们通过证明在q-Rényi散度下，LMC的迭代过程收敛到这些网络的某种Frobenius范数正则化损失所对应的吉布斯分布来实现这一点——这在分类和回归两种设置中均成立。本工作的关键技术要点有两个：（a）使所需的正则化量与网络的规模无关，（b）获得的收敛速率仅取决于相应吉布斯测度的Poincaré常数。这项工作综合了近期的若干观察结果，包括LMC收敛所需的等周条件，以及两层神经网络的损失函数总可以通过常数大小的正则化来处理。

    arXiv:2503.10428v4 Announce Type: replace  Abstract: In this note we prove that the Langevin Monte Carlo (LMC) algorithm can minimize the population risk of appropriately regularized depth-2 nets with no restrictions on the size of the net and for any choice of training data --- and we give non-asymptotic convergence rates for it. We achieve this via showing that in q-Renyi divergence, the iterates of LMC converge to the Gibbs distribution of certain Frobenius norm regularized losses for these nets --- in both classification and regression settings. The key technicality of the work is two-fold (a) to keep the amount of regularization needed be independent of the size of the net and (b) to obtain convergence rates that scale with just the Poincare constant of the corresponding Gibbs measure. This work synthesizes several recent observations about isoperimetry conditions under which LMC converges and that two-layer neural loss functions can always be regularized by a constant amount such
    
[^400]: 面向加密货币交易分析的大语言模型：以比特币为例的案例研究

    Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study

    [https://arxiv.org/abs/2501.18158](https://arxiv.org/abs/2501.18158)

    本文首次将大语言模型应用于比特币等真实加密货币交易图分析，提出三层能力评估框架、人类可读的图表示格式LLM4TG和连接增强的交易图采样算法，克服了传统黑盒模型难以解释和捕捉细微行为模式的局限。

    

    加密货币已被广泛使用，然而当前用于分析交易的方法通常依赖于不透明的黑盒模型。尽管这些模型可能达到较高的性能，但其输出通常难以解释和调整，使得捕捉细微的行为模式变得困难。大语言模型（LLMs）有潜力弥补这些不足，但它们在这一领域的能力在很大程度上仍未被探索，尤其是在网络犯罪检测方面。在本文中，我们通过将大语言模型应用于真实世界的加密货币交易图来验证这一假设，重点关注比特币——这一研究最多、应用最广泛的区块链网络之一。我们提出了一个三层评估框架来衡量LLM的能力：基础指标、特征概述和情境解释。这包括一种新的、人类可读的图表示格式LLM4TG，以及一种连接增强的交易图采样算法……

    arXiv:2501.18158v4 Announce Type: replace-cross  Abstract: Cryptocurrencies are widely used, yet current methods for analyzing transactions often rely on opaque, black-box models. While these models may achieve high performance, their outputs are usually difficult to interpret and adapt, making it challenging to capture nuanced behavioral patterns. Large language models (LLMs) have the potential to address these gaps, but their capabilities in this area remain largely unexplored, particularly in cybercrime detection. In this paper, we test this hypothesis by applying LLMs to real-world cryptocurrency transaction graphs, with a focus on Bitcoin, one of the most studied and widely adopted blockchain networks. We introduce a three-tiered framework to assess LLM capabilities: foundational metrics, characteristic overview, and contextual interpretation. This includes a new, human-readable graph representation format, LLM4TG, and a connectivity-enhanced transaction graph sampling algorithm, 
    
[^401]: 基于Wasserstein最小化的任意分布均值与方差估计复杂度研究

    Mean and Variance Estimation Complexity in Arbitrary Distributions via Wasserstein Minimization

    [https://arxiv.org/abs/2501.10172](https://arxiv.org/abs/2501.10172)

    本文揭示了任意分布下均值与方差参数估计的复杂度：虽然最大似然估计是NP难的，但利用Wasserstein距离可在关于 $1/\varepsilon$ 的多项式时间内获得任意精度的 $\varepsilon$-近似解。

    

    参数估计是机器学习中的一项根本性挑战，对于神经网络权重拟合和贝叶斯推断等任务至关重要。本文研究了在给定 $n$ 个样本的情况下，估计形如 $\frac{1}{\sigma^l} f_0\left(\frac{\boldsymbol{x}-\boldsymbol{\mu}}{\sigma}\right)$ 的分布的平移参数 $\boldsymbol{\mu} \in \mathbb{R}^l$ 与尺度参数 $\sigma \in \mathbb{R}_{++}$ 的计算复杂度，其中 $f_0$ 是 $\mathbb{R}^l$ 上已知的密度函数。我们着重指出，尽管该问题对于最大似然估计（MLE）而言是NP难的，但利用Wasserstein距离，可以在 $\text{poly}\left(\frac{1}{\varepsilon}\right)$ 时间内对任意 $\varepsilon > 0$ 获得 $\varepsilon$-近似解。

    arXiv:2501.10172v2 Announce Type: replace  Abstract: Parameter estimation is a fundamental challenge in machine learning, crucial for tasks such as neural network weight fitting and Bayesian inference. This paper focuses on the complexity of estimating translation $\boldsymbol{\mu} \in \mathbb{R}^l$ and shrinkage $\sigma \in \mathbb{R}_{++}$ parameters for a distribution of the form $\frac{1}{\sigma^l} f_0 \left( \frac{\boldsymbol{x} - \boldsymbol{\mu}}{\sigma} \right)$, where $f_0$ is a known density in $\mathbb{R}^l$ given $n$ samples. We highlight that while the problem is NP-hard for Maximum Likelihood Estimation (MLE), it is possible to obtain $\varepsilon$-approximations for arbitrary $\varepsilon > 0$ within $\text{poly} \left( \frac{1}{\varepsilon} \right)$ time using the Wasserstein distance.
    
[^402]: 用于分析医疗保健领域纵向可穿戴传感器数据的计算模型最新综述

    A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare

    [https://arxiv.org/abs/2407.21665](https://arxiv.org/abs/2407.21665)

    本文综述了三种用于分析医疗领域纵向可穿戴传感器数据的计算模型（日常规律、生理节律和稳定性指标），并讨论了纵向数据处理的关键挑战，以助力实现预测性、预防性、个性化和参与性的4P医学。

    

    可穿戴设备日益成为生物医学研究的工具，因为它们持续采集的行为和生理数据流能够为我们提供关于日常生活环境中健康状况的洞察。长期追踪（以数月乃至数年的时间尺度定义）可以揭示数据模式和变化，作为健康状况改变的指标。这些洞察可以使医学和医疗保健更具预测性、预防性、个性化和参与性（即“4P医学”）。然而，在建模、理解和处理纵向数据方面存在的挑战，是其在研究项目和临床环境中推广应用的重大障碍。本文回顾并讨论了三种用于理解纵向数据的模型：日常规律、生理节律和稳定性指标。我们阐述了与纵向可穿戴传感器数据处理和分析相关的挑战，并特别关注如何处理不同的时间尺度问题。

    arXiv:2407.21665v2 Announce Type: replace-cross  Abstract: Wearable devices are increasingly used as tools for biomedical research, as the continuous stream of behavioral and physiological data they collect can provide insights about our health in everyday contexts. Long-term tracking, defined in the timescale of months of year, can provide insights of patterns and changes as indicators of health changes. These insights can make medicine and healthcare more predictive, preventive, personalized, and participative (The 4P's). However, the challenges in modeling, understanding and processing longitudinal data are a significant barrier to their adoption in research studies and clinical settings. In this paper, we review and discuss three models used to make sense of longitudinal data: routines, rhythms and stability metrics. We present the challenges associated with the processing and analysis of longitudinal wearable sensor data, with a special focus on how to handle the different tempora
    
[^403]: 有限模型不确定性下的Q学习

    Q-Learning under Finite Model Uncertainty

    [https://arxiv.org/abs/2407.04259](https://arxiv.org/abs/2407.04259)

    提出了一种在有限模糊转移核集合下的鲁棒Q学习算法，证明了其几乎必然收敛到鲁棒最优解，并给出将可观测经验贝尔曼残差与转移核估计误差分离的时间一致高概率误差界。

    

    我们提出了一种面向模型不确定性下马尔可夫决策过程的鲁棒Q学习算法，其中每个状态-动作对都关联着一个由候选转移核构成的有限模糊集。这一有限测度框架支持用户自定义的不确定性模型，包括相互竞争的拟合模型以及压力情景。学习者从每个候选模拟器中观察后继状态及其确定性奖励，而无需显式知晓转移概率或奖励函数。我们证明了所学习的Q函数几乎必然收敛到鲁棒最优解，并推导出一个时间一致的高概率误差证书，该证书将可观测的经验贝尔曼残差与转移核估计误差分离开来。我们还针对有限状态空间上的Wasserstein不确定性模型和参数化不确定性模型，给出了定量的有限模糊逼近结果。数值算例说明了该方法的有效性。

    arXiv:2407.04259v4 Announce Type: replace-cross  Abstract: We propose a robust Q-learning algorithm for Markov decision processes under model uncertainty when each state--action pair is associated with a finite ambiguity set of candidate transition kernels. This finite-measure framework enables user-designed uncertainty models, including competing fitted models and stress scenarios. {The learner observes successor states and their deterministic rewards from every candidate simulator, without requiring explicit knowledge of the transition probabilities or the reward function.} We establish almost sure convergence of the learned Q-function to the robust optimum, and derive a time-uniform high-probability error certificate that separates an observable empirical Bellman residual from transition-kernel estimation error. We also give quantitative finite-ambiguity approximation results for Wasserstein and parametric uncertainty models on finite state spaces. Numerical examples illustrate the 
    
[^404]: HarmonICA：用于动态收缩期间运动单元分解的准时不变独立成分分析

    HarmonICA: Quasi-Time-Invariant ICA for Motor Unit Decomposition during Dynamic Contractions

    [https://arxiv.org/abs/2406.19581](https://arxiv.org/abs/2406.19581)

    提出了HarmonICA方法，通过在静态线性分离器前引入可学习的低秩时变可逆变换，克服混合过程时变的限制，首次将独立成分分析推广到动态收缩条件下的表面肌电运动单元分解。

    

    将表面肌电信号（EMG）分解为单个运动神经元的脉冲序列是一个长期存在的逆问题，也是实现运动神经元驱动的神经接口（如假肢和外骨骼）的关键步骤。标准方法是对多通道信号进行独立成分分析（ICA），其假设从神经元到电极的混合过程在时间上是平稳的。这一假设在运动过程中会失效，因为体积导体的形变使得混合过程随时间变化，因此现有的分解算法只能局限于等长收缩。我们提出了HarmonICA，这是一种准时不变的ICA形式，其中在静态线性分离器之前加入了一个可学习的、低秩的、时变的可逆变换。分离器通过对未补偿投影施加独立性损失进行训练，而变换则通过对恢复出的源信号施加平稳性损失进行训练。梯度不共享……（摘要原文在此处截断）

    arXiv:2406.19581v3 Announce Type: replace-cross  Abstract: Decomposing surface electromyography (EMG) into the spike trains of individual motor neurons is a long-standing inverse problem and a key step toward motor-neuron-driven neural interfaces such as prosthetics and exoskeletons. The standard approach, independent component analysis (ICA) of the multichannel signal, assumes that the mixing from neurons to electrodes is stationary in time. This assumption fails during movement, when volume-conductor deformation makes the mixing time-varying, and current decomposition algorithms are correspondingly restricted to isometric contractions. We introduce HarmonICA, a quasi-time-invariant ICA formulation in which a static linear separator is preceded by a learned, low-rank, time-varying invertible transformation. The separator is trained with an independence loss on the uncompensated projection, and the transformation with a stationarity loss on the recovered source. Gradients are not share
    
[^405]: 通过谱神经网络和非线性矩阵感知实现隐式正则化

    Implicit Regularization via Spectral Neural Networks and Non-linear Matrix Sensing

    [https://arxiv.org/abs/2402.17595](https://arxiv.org/abs/2402.17595)

    本文在更现实的神经网络背景下探讨了隐式正则化现象，通过研究非线性激活函数的一般类别，严格证明了在矩阵感知问题设置中这些网络的隐式正则化现象，同时提供了严格的速率保证，确保梯度的指数级快速收敛。

    

    隐式正则化现象近年来引起了人们的兴趣，作为神经网络出色泛化能力的一个基本方面。简而言之，它意味着在许多神经网络中，即使损失函数中没有任何显式正则化器，梯度下降动态也会收敛到一个正则化学习问题的解。然而，已知的试图从理论上解释这一现象的结果主要集中在线性神经网络的设置上，线性结构的简单性对现有论据特别关键。本文在更现实的神经网络和一般非线性激活函数的背景下探讨了这一问题，并在矩阵感知问题的设置中严格证明了这些网络的隐式正则化现象，同时提供了严格的速率保证，确保梯度的指数级快速收敛。

    arXiv:2402.17595v1 Announce Type: cross  Abstract: The phenomenon of implicit regularization has attracted interest in recent years as a fundamental aspect of the remarkable generalizing ability of neural networks. In a nutshell, it entails that gradient descent dynamics in many neural nets, even without any explicit regularizer in the loss function, converges to the solution of a regularized learning problem. However, known results attempting to theoretically explain this phenomenon focus overwhelmingly on the setting of linear neural nets, and the simplicity of the linear structure is particularly crucial to existing arguments. In this paper, we explore this problem in the context of more realistic neural networks with a general class of non-linear activation functions, and rigorously demonstrate the implicit regularization phenomenon for such networks in the setting of matrix sensing problems, together with rigorous rate guarantees that ensure exponentially fast convergence of gradi
    
[^406]: 针对均场极小极大问题的镜像下降-上升算法的研究

    Mirror Descent-Ascent for mean-field min-max problems

    [https://arxiv.org/abs/2402.08106](https://arxiv.org/abs/2402.08106)

    该论文研究了解决测度空间上极小极大问题的镜像下降-上升算法的两种变体，并证明了收敛速率与相关有限维算法的最新结果一致。

    

    我们研究了镜像下降-上升算法在测度空间上解决极小极大问题的两个变体：同时和依次。我们在凸性-凹性和相对光滑性的假设下，针对适当的Bregman散度在测度空间上通过平坦导数进行了定义。我们证明了收敛速率到混合纳什均衡，用尼凯多-Isoda误差表示，对于同时和依次方案分别是$\mathcal{O}\left(N^{-1/2}\right)$和$\mathcal{O}\left(N^{-2/3}\right)$，这与相关有限维算法的最新结果一致。

    We study two variants of the mirror descent-ascent algorithm for solving min-max problems on the space of measures: simultaneous and sequential. We work under assumptions of convexity-concavity and relative smoothness of the payoff function with respect to a suitable Bregman divergence, defined on the space of measures via flat derivatives. We show that the convergence rates to mixed Nash equilibria, measured in the Nikaid\`o-Isoda error, are of order $\mathcal{O}\left(N^{-1/2}\right)$ and $\mathcal{O}\left(N^{-2/3}\right)$ for the simultaneous and sequential schemes, respectively, which is in line with the state-of-the-art results for related finite-dimensional algorithms.
    
[^407]: RAYEN：在神经网络上的硬凸约束施加

    RAYEN: Imposition of Hard Convex Constraints on Neural Networks

    [https://arxiv.org/abs/2307.08336](https://arxiv.org/abs/2307.08336)

    RAYEN是一个能够在神经网络输出或潜变量上施加硬凸约束的框架，可在训练和测试阶段对任意输入和权重保证约束满足，支持线性、凸二次、二阶锥和线性矩阵不等式约束的任意组合，且计算开销几乎可以忽略。

    

    尽管凸约束在机器人学中有众多应用，但在基于学习的框架中强制执行这些约束仍然是一个悬而未决的挑战。现有技术要么无法保证始终满足约束，要么会产生过高的计算成本。本文提出了RAYEN，这是一个用于在神经网络的输出或潜变量上施加硬凸约束的框架。RAYEN保证在训练和测试期间，对于任意输入和任意网络权重，约束都能得到满足。与先前的方法不同，RAYEN避免了计算代价高昂的正交投影、软约束、可行集的保守近似以及缓慢的迭代修正。RAYEN支持线性、凸二次、二阶锥（SOC）和线性矩阵不等式（LMI）约束的任意组合，与非约束网络相比，其开销可以忽略不计。例如，它可以在1K维变量上施加1K个二次约束（原文在此处截断）。

    arXiv:2307.08336v3 Announce Type: replace  Abstract: Despite the numerous applications of convex constraints in Robotics, enforcing them within learning-based frameworks remains an open challenge. Existing techniques either fail to guarantee satisfaction at all times, or incur prohibitive computational costs. This paper presents RAYEN, a framework for imposing hard convex constraints on the output or latent variables of a neural network. RAYEN guarantees constraint satisfaction during both training and testing, for any input and any network weights. Unlike prior approaches, RAYEN avoids computationally expensive orthogonal projections, soft constraints, conservative approximations of the feasible set, and slow iterative corrections. RAYEN supports any combination of linear, convex quadratic, second-order cone (SOC), and linear matrix inequality (LMI) constraints, with negligible overhead compared to unconstrained networks. For instance, it imposes 1K quadratic constraints on a 1K-dimen
    
[^408]: AI监督和人类错误：来自中心法庭的证据

    AI Oversight and Human Mistakes: Evidence from Centre Court. (arXiv:2401.16754v1 [cs.LG])

    [http://arxiv.org/abs/2401.16754](http://arxiv.org/abs/2401.16754)

    人工智能系统在纠正人类错误方面起到了积极作用，但此举也潜在导致心理成本，并影响人的决策。通过研究网球比赛中的Hawk-Eye审查系统，我们发现引入AI监督后，裁判员的错误率下降，心理成本导致他们更倾向于将球判为进界，从而产生了类型错判的转变。

    

    在机器学习算法不断提升的驱动下，人工智能（AI）系统已经开始在许多场合用于纠正人类错误。我们提供了首个实地证据，证明这种AI监督会产生心理成本，影响人的决策。我们调查了AI监督发生的最高可见性场景之一：顶级网球比赛中裁判的Hawk-Eye审查。我们发现，引入Hawk-Eye审查后，裁判的整体错误率降低，符合心理成本被AI否定的合理忽视现象。我们还发现，裁判增加了对球入内的判定率，从而产生了从II类错误（将球判为出界，实际上是进界）到I类错误（将球判为进界，实际上是出界）的转变。通过对理性不注意的裁判模型进行心理成本的结构估计，我们的结果表明，由于AI否定的心理成本，裁判员降低了错误判定的风险并提高了球入内的判定率。

    Powered by the increasing predictive capabilities of machine learning algorithms, artificial intelligence (AI) systems have begun to be used to overrule human mistakes in many settings. We provide the first field evidence this AI oversight carries psychological costs that can impact human decision-making. We investigate one of the highest visibility settings in which AI oversight has occurred: the Hawk-Eye review of umpires in top tennis tournaments. We find that umpires lowered their overall mistake rate after the introduction of Hawk-Eye review, in line with rational inattention given psychological costs of being overruled by AI. We also find that umpires increased the rate at which they called balls in, which produced a shift from making Type II errors (calling a ball out when in) to Type I errors (calling a ball in when out). We structurally estimate the psychological costs of being overruled by AI using a model of rational inattentive umpires, and our results suggest that because 
    

