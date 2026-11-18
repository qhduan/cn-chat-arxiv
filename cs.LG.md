# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching](https://arxiv.org/abs/2611.23229) | 提出Ambient SFT框架，通过将微调模型的高温条件分布与冻结的预训练“噪声教师”模型进行温度匹配，利用不完美预测扩充微调数据集，从而防止过拟合和数据分布尾部的灾难性遗忘。 |
| [^2] | [Spool: Low-Latency Streaming Inference for Hybrid Vision-Language Models](https://arxiv.org/abs/2611.23228) | Spool是一个无需训练的框架，利用线性循环层状态更新的仿射可结合特性，通过缓存并组合每帧的状态变换来移除过期帧，实现混合视觉语言模型在滑动视频窗口上的低延迟流式推理。 |
| [^3] | [An inexact stochastic variance-reduced gradient method for constrained convex finite-sum optimization](https://arxiv.org/abs/2611.23221) | 该论文针对带硬约束的凸有限和优化问题，提出了一种非精确随机方差缩减梯度方法来计算ε-确定可行随机最优解，从而缩小了确定可行解与期望可行解之间的一阶预言机复杂度差距。 |
| [^4] | [Do LLM Agents Still Need to Be Told How to Self-Improve?](https://arxiv.org/abs/2611.23219) | 该论文提出了一种完全自主的自我改进范式，证明足够强大的冻结大语言模型智能体在仅有基本要素、工具和计算预算、且无人类指定方法或外部强加循环的条件下，能够自行构建自学习算法并实现自我改进，其策略与经典自我调节学习认知框架相符。 |
| [^5] | [What Does a Bias Head Measure? Non-Identifiability in Head-Level Bias Attribution](https://arxiv.org/abs/2611.23218) | 该论文审计了将LLM偏差归因于单个注意力头的常用方法，发现用于选头的统计量主要由提示身份驱动（占64%方差），而公平性标签、响应长度与受保护群体标记之间存在固有共线性，证明头级偏差归因本质上是不可辨识的。 |
| [^6] | [Higher-Scoring Coding Models Did Not Get More Work Done](https://arxiv.org/abs/2611.23216) | 该研究利用美国出口管制导致Claude Fable 5被停用19天的自然实验，发现在编码基准测试中得分更高的模型（Fable 5）相比得分较低的模型（Opus 4.8）并未给用户带来更高的实际工作产出，揭示了基准分数与真实生产力之间的脱节。 |
| [^7] | [Multimodal Masked Diffusion Autoencoder for Modeling Impaired Driving](https://arxiv.org/abs/2611.23211) | 该论文提出多模态掩码扩散自编码器 ID-MAE，通过自监督方式重建被掩码的注视与轨迹数据来增强受损驾驶预测，并发布了包含 109 人、103 小时驾驶数据及十种受损诱发条件的大规模模拟驾驶数据集 DIVERT，显著提升了受损驾驶推断效果。 |
| [^8] | [Distortion-Free Semantic Watermarks for Black-Box Generative Models](https://arxiv.org/abs/2611.23208) | 该论文提出了一种仅需黑盒访问权限的、可证明无失真的语义水印算法，在不改变生成模型输出分布（即不影响输出质量）的同时，对压缩、裁剪等图像内容保持型变换具有鲁棒检测能力。 |
| [^9] | [Fine-Tuning Gemma 4 Edge Models for MATLAB Signal-Processing Code with SFT and QLoRA on a Laptop iGPU](https://arxiv.org/abs/2611.23205) | 该论文在笔记本电脑集成GPU上用66个样例通过SFT结合LoRA/QLoRA微调Gemma 4边缘模型（E2B和E4B）用于MATLAB信号处理代码生成，使通过率显著提升（如E4B从32/66升至47/66），且发现微调主要修正DSP惯例错误而非信号构建错误，运行时代价仅为17-19%的生成速度损失。 |
| [^10] | [Learned versus Kinematic Trajectory Prediction under Reduced Observation Frequency](https://arxiv.org/abs/2611.23198) | 该研究在Argoverse 2数据集上通过将观测历史从10 Hz降采样至1-5 Hz并进行严格的统计检验，系统评估了学习型轨迹预测器相对于恒定速度/加速度运动学模型的优势如何随观测频率降低而变化。 |
| [^11] | [WoRD: Diffusion-Based Posterior Inference for Online Goal Recognition](https://arxiv.org/abs/2611.23192) | 提出了WoRD方法，利用学习到的扩散轨迹先验，为每个候选目标运行热启动去噪链并通过贝叶斯重加权得到校准后验，实现从部分观测中对智能体目标的在线识别。 |
| [^12] | [D\'ej\`a View: A Shared Working-Memory Model for Visual Tasks](https://arxiv.org/abs/2611.23177) | 该研究提出了一个将动态活动与图像特征间临时关联相结合的共享工作记忆模型，单个训练后的模型即可在提示引导下灵活完成图像记忆、干扰分类和新旧识别等多种视觉任务。 |
| [^13] | [MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study](https://arxiv.org/abs/2611.23158) | 本研究首次在统一的多语言评估协议下，用200种语言和9000多个评估设置系统比较八种LoRA方法，发现小规模搜索无法可靠预测方法在大规模下的相对表现，且LoRA虽能提升翻译数据拟合度但会持续损害阅读理解能力。 |
| [^14] | [Learning Stratified Policies for Inference-Time Objectives](https://arxiv.org/abs/2611.23154) | 该论文提出“分层策略”，通过在共享语言模型主干上训练 k 个 LoRA 适配器，将推理时的采样预算分配给一个策略群体，并从理论和实验上证明这种策略群体在优化 pass@k 时优于单一策略。 |
| [^15] | [Capability Composition in Heterogeneous Teams via Barrier-Guided Diffusion Sampling](https://arxiv.org/abs/2611.23149) | 本文提出一种障碍引导的扩散采样方法，将能力组合的物理条件编码为轨迹约束，使异构团队无需任何合作演示数据即可组合个体能力生成合作轨迹，并证明该方法能生成所有满足约束的团队轨迹。 |
| [^16] | [RSI-Prime: Training Models to Train Models](https://arxiv.org/abs/2611.23138) | 该论文提出RSI-Prime数据集（含覆盖七大任务类别的350条完整实验轨迹），通过细粒度决策质量标注与决策级损失掩码过滤低质量行为进行训练，使GLM-5.3在自主后训练基准PostTrainBench上整体性能相对提升31.95%，为递归自我改进奠定基础。 |
| [^17] | [Correcting Flows with Marginal Matching](https://arxiv.org/abs/2611.23134) | 本文提出了一种新颖的基于训练的预测-校正框架，通过“边际匹配”思想在前向ODE的特定时间步将样本校正至与反向ODE相匹配的边际分布，从而有效缩小学习分布与数据分布之间的距离，提升流匹配模型的生成质量。 |
| [^18] | [A Neural Envelope Method for Rapid Gradient-based Optimum Takeoff Trajectory Design](https://arxiv.org/abs/2611.23130) | 本文提出一种将物理约束生成对抗网络（physicsGAN）与神经包络相结合的方法，将带约束的非凸最优起飞轨迹设计问题转化为无约束的凸优化问题，从而实现快速的基于梯度的最优轨迹设计。 |
| [^19] | [Where Should Physics Enter? Information, Identifiability and Calibration in Probabilistic Sea Surface Temperature Forecasting](https://arxiv.org/abs/2611.23129) | 该论文系统性地测试了平流-扩散物理先验在神经海表温度预测模型中的不同介入位置，发现学习得到的速度场既无法带来可靠的预测收益，也无法辨识出真实洋流——即使洋流信息理论上可大幅提升预测精度——揭示了物理信息神经网络中可辨识性与校准的关键问题。 |
| [^20] | [The Illusion of Grounding: Do Vision-Language Models Weigh Retrieved Text and Images Like an Ideal Observer?](https://arxiv.org/abs/2611.23127) | 该论文构建了包含20万个条目的GIB-200K压力测试基准，并从理想贝叶斯线索组合理论出发，首次形式化定义并量化了视觉-语言模型在融合检索文本与图像证据时的“知识惯性”与“感官捕获”偏差。 |
| [^21] | [Which Past Matters? Training History and Multi-Forecast Horizon in Equity Transformers](https://arxiv.org/abs/2611.23126) | 本文通过十二种配置的月度滚动实验发现，在日内股票收益预测中，最优的历史训练数据选择取决于预测期限——短期预测（5-15分钟）适合近期数据采样，长期预测（60-120分钟）适合均匀采样，且单纯增加训练数据量并不能持续提升预测精度。 |
| [^22] | [Get Hint-Augmented GRPO Done Right Through Activation Steering](https://arxiv.org/abs/2611.23124) | 该论文提出ASH方法，将HA-GRPO中提示的作用重新诠释为引导模型行为而非降低问题难度，通过激活引导技术解决训练与推理时上下文不匹配的问题，从而实现一致的推理能力提升。 |
| [^23] | [Auditing a Learned Simulator for Legged Robots: The Robotic World Model on ANYmal D](https://arxiv.org/abs/2611.23118) | 该论文对四足机器人的学习型模拟器进行了端到端审计，发现仅用单步误差评估会误判模拟器质量——尽管单步训练的模拟器单步预测更准，但在自身长时推演数据上训练的模拟器能让策略在 368 步的推演中表现更好。 |
| [^24] | [Large-scale Neuron Community Discovery Reveals Neuron-Sparse Behaviors in Language Models](https://arxiv.org/abs/2611.23112) | 该论文提出了一种无监督的“神经元社区发现”方法，通过梯度归因、图构建和社区检测，在大语言模型中找出跨输入复用且因果必要充分的神经元群组，从而证明语言模型中存在可复用的稀疏神经元计算单元。 |
| [^25] | [Event-Based Dynamic Lip Region Estimation During Speech](https://arxiv.org/abs/2611.23111) | 本文提出首个从完整事件帧预测唇部区域密集热力图的方法，为事件相机的唇读等语音应用提供了自动的动态唇部空间定位，解决了现有方法依赖手工裁剪ROI或关键点定位与应用需求不匹配的问题。 |
| [^26] | [What's in a Tree ? Hierarchical $n$-gram models as controlled approximators of language](https://arxiv.org/abs/2611.23093) | 本文提出 tree-grams——一种通过二阶和三阶张量层次化组合构建的自回归 n-gram 模型，其上下文窗口与模型容量分别由超参数 n 和 d 可控地调节，在超越传统 n-gram 基线的同时可与 Transformer 语言模型媲美，并可作为研究语言的可控近似器用于超参数迁移等实验。 |
| [^27] | [Learning to Undo: Transfer Reinforcement Learning under State Space Transformations](https://arxiv.org/abs/2611.23081) | 该论文提出通过匹配两个MDP的状态特征统计量来学习“撤销映射”，从而将源MDP的策略迁移到目标MDP，并证明在撤销映射为线性且源满足线性$Q^\star$可实现的条件下，其样本复杂度严格优于在目标MDP中从零开始学习。 |
| [^28] | [Pareto-Informed Smart "Predict, then Optimize" for Multi-Objective Combinatorial Problems](https://arxiv.org/abs/2611.23076) | 提出了一种帕累托感知的SPO+扩展方法，利用真实帕累托前沿构建与采样偏好向量对齐的目标决策并最小化遗憾，克服了现有决策聚焦学习方法在多目标组合优化中因简单标量化而无法捕捉丰富权衡关系的局限。 |
| [^29] | [Runtime Verification of Multiple Natural Language Criteria for Agent Governance](https://arxiv.org/abs/2611.23074) | 提出了一种基于树注意力的并行验证器VFM，可在运行时对智能体输出同时评估数百条自然语言治理准则并给出逐条三元判定，其预训练语料库CriteriaBank带来了跨域性能提升。 |
| [^30] | [Amortizing Tempered Sampling with Corrected Self-Training](https://arxiv.org/abs/2611.23072) | 提出了一种在退火目标序列上进行校正自训练的神经采样方法SSS，通过基于得分的扩散提议与路径空间重要性权重校正，将目标分布的全局结构编码进模型以摊销重复采样成本，在基准测试中取得有竞争力的采样精度且运行时间更短。 |
| [^31] | [FrameRouter: Frame Budget Routing for Long-Video Understanding on Video-MME-v2 and Beyond](https://arxiv.org/abs/2611.23055) | 提出无需训练的测试时框架FrameRouter，通过证据需求路由器为每个问题动态分配不同的帧预算（保持平均预算与统一参考一致），从而更有效地提升长视频问答的准确性。 |
| [^32] | [Magflow: Differentiable Simulation of Industrial-Grade 6D Magnetic Levitation](https://arxiv.org/abs/2611.23050) | 提出了首个完全可微、GPU并行的六自由度磁悬浮仿真器Magflow，并利用它训练出首个工业级六自由度磁悬浮系统的神经控制器，突破了传统手工近似电磁力计算方法对工作空间和动子拓扑的保守限制。 |
| [^33] | [Reversing the text is not reversing the path: a fixed-text measure without a zero, and an assembly-order comparison that vanishes identically](https://arxiv.org/abs/2611.23049) | 该论文指出单篇固定文本上的逆转比较度量（拟“熵产生”）缺乏由定理保证的零点，并利用在全部 n! 种句子排列下评分所得到的精确反对称性 Δ(π) = −Δ(π∘rev) 为其赋予零点，证明基于文本逆转的组装顺序比较恒为零，进而以作者排布的无量纲有界中位秩作为文本组织程度的替代度量。 |
| [^34] | [Uncovering the Invisible in Routine Radiology: Cross-Modal Detection of MRI-Verified Soft-Tissue Pathologies from CBCT via Physics-Guided 3D Self-Supervised Learning](https://arxiv.org/abs/2611.23042) | 该研究首次证明常规CBCT扫描中物理性地保留了软组织病变的亚视觉放射学特征，并通过物理引导的两阶段3D自监督学习框架（利用在6,550个未标记CBCT体积上预训练的掩码自编码器），在配对数据极度稀缺的条件下实现了对经MRI验证的软组织病变（以颞下颌关节为模型）的跨模态机会性筛查。 |
| [^35] | [Variance Restoration over Covariance Rank: Memory-Budgeted Analytic Continual Learning](https://arxiv.org/abs/2611.23040) | 该论文针对固定内存预算下的解析持续学习，提出只保存随维度线性增长的坐标方差与类别统计量、以替代二次增长的完整协方差矩阵，从而在保留宽随机码带来的线性可分性提升的同时大幅降低内存开销。 |
| [^36] | [When Anchors Shift the Target: Advantage-Gated Adaptation for Diffusion Reinforcement Learning](https://arxiv.org/abs/2611.23038) | 本文提出优势门控自适应（AGA），通过按优势符号解耦正负样本的角色——正样本保持标准目标拟合、负样本改用凹目标在移动策略预测处的一阶切线——从而解决扩散强化学习中的锚点耦合冲突。 |
| [^37] | [Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels](https://arxiv.org/abs/2611.23035) | This paper exploits the linear dependence between the key and value in the global attention layers of Gemma 4 31B (K=V), storing only one 640-value row without modifying the attention kernel, which reads the key and value caches through two overlapping strided views, achieving a 1.34 to 1.40 fold reduction in KV cache memory during decoding with bit-identical outputs. |
| [^38] | [NVIDIA Kumo Tabular: Pushing the Accuracy$\unicode{8211}$Efficiency Frontier for Tabular Prediction](https://arxiv.org/abs/2611.23001) | Kumo Tabular 是 NVIDIA 推出的系列开源表格基础模型，通过在人工生成表格上预训练并利用上下文学习技术，无需任何针对新表格的训练即可完成预测，并在八项基准测试中夺得第一名。 |
| [^39] | [Residual-Guided Lightweight Reinforcement Learning for Risk-Aware Trajectory Selection with a Frozen World Model](https://arxiv.org/abs/2611.22998) | 该论文提出一种基于冻结世界模型的残差引导轻量级强化学习框架，仅需微调约1.6万参数即可实现自动驾驶中风险感知的高效轨迹选择。 |
| [^40] | [High-Dimensional Black-Box Optimization via Guidance Sampling of Diffusion Models](https://arxiv.org/abs/2611.22965) | 该论文提出了BReaD框架，利用扩散模型的引导采样技术，在不需要大量标注数据或精确奖励梯度的情况下，有效解决高维黑盒优化问题。 |
| [^41] | [Hierarchical Diffusion Transformer for End-to-End Image Generation](https://arxiv.org/abs/2611.22962) | 提出层级扩散Transformer（HiT），通过逐步将相关token分组压缩来构建视觉层级结构，使端到端像素空间扩散模型在生成质量和训练效率上追赶领先的两阶段潜在扩散模型。 |
| [^42] | [DDSL: Iterative Self-Labeling for Financial Market States](https://arxiv.org/abs/2611.22958) | DDSL是一种迭代自标注框架，通过价格与语法引导的Viterbi标注器和Transformer分类器之间的模型反馈耦合，构建同时编码当前与未来市场趋势的九状态金融市场表示，在多个指数数据集上将联合状态宏观F1提升78%-96%，且独立分类器可从预测时可用信息中恢复这些标签。 |
| [^43] | [Is model self-assessment enough for diverse LVLM ensembles?](https://arxiv.org/abs/2611.22956) | 该论文提出了一个统一的候选响应选择集成流水线，并通过涵盖34种置信度估计方法、24种集成配置和20个数据集的大规模评估证明，仅依赖模型自身置信度的简单免训练、免参数自我评估方法即可在多样化LVLM集成中取得优异表现。 |
| [^44] | [ResoLUT: Distilling Residual Networks into Lookup-Table Networks](https://arxiv.org/abs/2611.22955) | ResoLUT 通过将训练好的 ResNet 逐层蒸馏为查找表树并保留其多比特残差流，实现了逻辑/查找表网络在 CIFAR-10、CIFAR-100 和 ImageNet 上的最佳准确率，证明 LUT 网络可以随深度扩展。 |
| [^45] | [Symmetries of Counterfactuals: Identification and Testability of Counterfactual Maps from Covariate Heterogeneity](https://arxiv.org/abs/2611.22948) | 该论文刻画了与观测数据一致的所有反事实映射——它们彼此仅相差一个由协变量条件结果分布决定的对称群（协变量异质性越强，对称群越小），并证明对于三角单调映射和最优传输映射等常见反事实类别，与数据的一致性可归结为一个可检验的条件独立性命题。 |
| [^46] | [The Kernel Reality Check: Benchmarking and Distilling Efficient Attention at Scale](https://arxiv.org/abs/2611.22947) | 该论文对核化注意力方法进行了大规模系统性基准测试，发现 FlashAttention 在效率上仍优于最先进的核化方法，且现有核化方法在质量上落后于 softmax 2-10 分，从而挑战了核化注意力作为 softmax 即插即用替代方案的核心假设。 |
| [^47] | [A Transformer-Derived Iterative Preconditioner](https://arxiv.org/abs/2611.22941) | 该论文从Transformer的内部计算中提炼出迭代预条件化原语，发展出无需学习组件的独立算法RAPID，可有效改善SPD矩阵条件数并减少共轭梯度求解所需迭代次数，且具有理论收敛保证。 |
| [^48] | [Projected Penalty-Gradient Tracking for Communication-Efficient Byzantine-Resilient Federated Learning](https://arxiv.org/abs/2611.22929) | 提出一种投影罚函数梯度追踪框架，使联邦学习在压缩通信条件下仍能保持拜占庭攻击鲁棒性，并给出两种高效算法。 |
| [^49] | [Action-outcome coupling: a pre-investment feasibility audit for offline reinforcement learning in supply chain operations](https://arxiv.org/abs/2611.22921) | 本文提出“动作-结果耦合”这一总效应方差指标，作为供应链离线强化学习投资前的可行性审计工具，为任何策略的价值提供与算法无关的货币化上界，并揭示在同一数据上拟合与评估价值模型会随状态单元数增加而导致近乎必然的虚假许可风险。 |
| [^50] | [Who Pays for Byzantine Robustness? Validity, Data-Share Dictatorships, and Fairness When Only a Few Hospitals Collaborate](https://arxiv.org/abs/2611.22904) | 该论文通过对八种拜占庭鲁棒聚合规则在仅有3-6家医院的真实临床联邦学习数据集上进行54,214次训练运行的实证审计，揭示了现有基于大规模合成客户端的基准测试无法反映小规模医疗协作场景下的有效性、数据份额独裁与公平性代价。 |
| [^51] | [Beyond Attribute-Centric Fairness: Modelling and Mitigating Unified Bias Representation in Omni-modal Large Language Models](https://arxiv.org/abs/2611.22889) | 提出了OmniDebias——首批面向全模态大语言模型的去偏方法之一，通过建模并缓解跨属性与模态的统一共享偏见表征，突破了传统以属性为中心的公平性范式的局限。 |
| [^52] | [Outcome-Balanced PPO and Policy-Induced Flow Rectification for Flow-Based VLA Policies](https://arxiv.org/abs/2611.22878) | 提出一种结合结果平衡PPO（利用任务成功率重加权并重缩放优势）与策略诱导流整流（周期性刷新速度场表示）的强化学习框架，以解决基于流的VLA策略后训练中速度场几何变化和多任务学习信号不均衡两大难题。 |
| [^53] | [Path-Aligned Constraint Equivariance](https://arxiv.org/abs/2611.22873) | 该论文提出“路径等变性”原理及其实现 PACE：在采用线性路径的条件流匹配中，只要速度场与线性约束相容且初始噪声满足该约束，无散度、无旋度、归一化等物理约束即可沿整条生成轨迹被精确保持，从而以硬约束方式实现超分辨率且不损失小尺度谱结构。 |
| [^54] | [When the tail is earned by its companions: measuring label dependence in long-tailed multi-label classification](https://arxiv.org/abs/2611.22859) | 该论文提出了可在训练前评估标签依赖性的标签内估计量、读取规则与统计量，揭示稀有标签的召回表现依赖频繁伴生标签而非模型真正学到稀有标签本身的现象。 |
| [^55] | [WATERFALL: Workflow for Adaptive Training with Evolutionary Reward Formulation and Automated Learning Loops](https://arxiv.org/abs/2611.22854) | WATERFALL提出了一个面向POMDP部分可观测环境的迭代式基于种群的自动化奖励发现工作流，无需目标条件、任务描述或专家演示等先验假设，通过角色条件化生成多样化的程序化奖励候选，以解决稀疏奖励强化学习中的奖励设计瓶颈。 |
| [^56] | [SilverSpoon-KD: A General-Purpose Toolkit for Knowledge Distillation](https://arxiv.org/abs/2611.22842) | SilverSpoon-KD是一个开源PyTorch知识蒸馏工具包，通过单一API统一了多种蒸馏范式与架构关系，并首创了通用的逐块知识蒸馏实现，在视觉和语言模型等任务上验证了其通用性与高性能。 |
| [^57] | [TabDAG: Structured Evolutionary Search for Tabular Feature Engineering](https://arxiv.org/abs/2611.22828) | TabDAG 提出将进化搜索与特征工程程序的有向无环图表示相结合，通过受父代模式约束的结构化变异并在评估前进行有效性检查，自动组合生成包含交互、统计编码和聚类邻域特征等多阶段转换的表格特征工程程序。 |
| [^58] | [OICycle: Observe, Infer, and Intervene Cycle for Belief-State Learning in Longitudinal Clinical Decision Making](https://arxiv.org/abs/2611.22826) | 提出OICycle框架，通过“观察-推断-干预”循环算子直接从回顾性电子健康记录轨迹中学习随机高斯信念状态，无需人工指定离散状态空间和转移模型即可实现纵向临床决策中的患者潜在状态更新。 |
| [^59] | [Virtual Double Oracle: Faster Convergence by Leveraging Strategies That Were Watching All Along](https://arxiv.org/abs/2611.22806) | 提出虚拟双重神谕（VDO），利用双线性恒等式让新发现的策略以“仿佛从一开始就参与学习”的权重初始化，将可利用性收敛速率从 O(√(k ln k / T)) 改进到 O(√(k/T))。 |
| [^60] | [PANO: Parallel Latent Space Neural Operator for Solving PDEs](https://arxiv.org/abs/2611.22782) | 提出并行潜空间神经算子PANO，通过多个共享参数的并行潜空间将观测数据压缩为多种不同表示，并将各空间的交互结果相加组合，克服了现有方法单一聚合模式导致物理信息不可见的局限，更全面地学习偏微分方程求解中的物理交互。 |
| [^61] | [Robust Actor-Critic Learning under Transport and Divergence Uncertainty](https://arxiv.org/abs/2611.22759) | 本文提出一种面向连续状态-动作、标称转移未知的折扣鲁棒马尔可夫决策过程的执行器-评论家方法，通过一次性拟合条件扩散模型，并结合共享目标极小极大准则与指数加权得分匹配，处理由Wasserstein距离、全变差及多种散度定义的转移不确定性。 |
| [^62] | [LoopPrune: Recurrent Pruning for Looped Diffusion Transformers](https://arxiv.org/abs/2611.22757) | LoopPrune提出了一种针对循环扩散Transformer的剪枝与恢复框架，通过循环重要性估计、共享重建和循环位置恢复三大机制，在降低存储与计算成本的同时有效保持模型性能。 |
| [^63] | [Stochastic Diversity Windows: Diagnosing Noise Influence in Robotic Diffusion Sampling](https://arxiv.org/abs/2611.22743) | 该论文提出了一种逐时间步的反事实诊断方法来量化扩散采样中随机噪声对最终输出的影响，并据此定义了“随机多样性窗口”（SDWs）以定位随机性影响最强的去噪时间区间，从而指导有针对性的干预。 |
| [^64] | [Proximal Targets: Training Looped Refiners on Problems with Many Correct Answers](https://arxiv.org/abs/2611.22737) | 提出近端目标训练方法，让循环精炼网络的每次迭代朝向离其输入最近的正确答案收敛，在存在多个正确答案的问题上将解题准确率从2.5%大幅提升至98.8%。 |
| [^65] | [LoopZip: Visual Token Compression for Efficient Looped Vision-Language Models](https://arxiv.org/abs/2611.22728) | LoopZip是一种无需训练的视觉标记压缩算法，通过在循环主干前构建紧凑候选池并结合跨调用选择，降低循环视觉语言模型处理长视觉序列时的计算与GPU显存开销。 |
| [^66] | [HADVQ: Hessian Aware Directional Vector Quantization](https://arxiv.org/abs/2611.22721) | 提出了一种面向超低比特LLM量化的向量量化方法HADVQ，通过随机预条件化正则化权重空间、在极坐标下使用基于E8格的结构化码本解耦方向与幅度的量化，并利用代理Hessian以仅0.16比特的额外开销保护高度敏感通道。 |
| [^67] | [Social Signal Bench: Evidence-Weighted Evaluation of Co-occurring Social Signals in Video and Audio](https://arxiv.org/abs/2611.22704) | 该论文提出了Social Signal Bench——一个47小时密集标注的视频与音频社交信号识别基准，并引入证据加权F1（EW-F1）评估指标，以更好地处理共现社交信号、语义关系和标注者分歧问题。 |
| [^68] | [CULypso: A Clustered Federated Unlearning paradigm for Vehicular Ad-Hoc Networks](https://arxiv.org/abs/2611.22682) | 本文提出CULypso，一种专为集群式车辆自组织网络设计的分层联邦遗忘算法，利用网络拓扑与车辆移动性降低通信、资源和时间开销，并能有效遗忘客户端对全局模型更新的影响。 |
| [^69] | [AlignedHPO: Efficient Surrogate-Guided Loss Hyperparameter Optimization](https://arxiv.org/abs/2611.22658) | AlignedHPO提出了一种利用梯度对齐和可微代理验证目标的高效损失超参数优化框架，通过雅可比-向量积与归一化对齐等技术，在保持竞争力性能的同时将超梯度计算速度提升近12倍，成功应用于长尾分类和开集人脸识别任务。 |
| [^70] | [The Price of Hard Coverage Constraints in Online Conformal Prediction](https://arxiv.org/abs/2611.22646) | 本文刻画了在线共形预测中强制硬覆盖约束（累积误覆盖受预算限制）不可避免的最小预测集膨胀代价，并证明即使在数据独立同分布、分数分布已知的最理想情况下，这一代价依然无法避免。 |
| [^71] | [Sub-Meter Canopy Height Synthesis from Sentinel-2 via Anchored Residual Flow](https://arxiv.org/abs/2611.22636) | 该论文提出锚定残差流方法，仅利用免费的10米Sentinel-2影像通过确定性回归锚点与残差条件流匹配相结合，合成0.5米分辨率的冠层高度模型，在无需昂贵机载LiDAR的情况下达到2.83米MAE。 |
| [^72] | [A Quantum Support Vector Regression Framework for Missing Data Imputation in Ship Performance Data](https://arxiv.org/abs/2611.22635) | 本研究据称首次将量子支持向量回归（QSVR）应用于船舶性能数据的缺失值填补，通过在巴拿马型散货船真实燃油消耗数据上对比经典SVR与多种量子特征映射下的QSVR，验证了量子机器学习在船舶数据填补领域的可行性。 |
| [^73] | [DAQ: Decision-Aligned Post-Training Quantization for Spiking Transformers](https://arxiv.org/abs/2611.22611) | 提出决策对齐训练后量化方法DAQ，通过校准量化器使脉冲Transformer神经元的放电决策与浮点参考模型对齐，从而在权重和膜变量量化的同时有效保持神经元的放电行为。 |
| [^74] | [Task Convergent Flow Matching for Generative Robotics](https://arxiv.org/abs/2611.22609) | 提出了一种以任务为中心的推理时优化框架，通过并行物理仿真推演和任务级性能漏斗对流匹配策略进行优化，确保机器人操作动作持续收敛于期望的任务目标。 |
| [^75] | [DLS: Dual-Logic Graph Neural Networks with Chaos-Fuzzy Residuals for Robust Link Prediction](https://arxiv.org/abs/2611.22607) | 提出了DLS归纳式图神经网络框架，通过混沌注入与动态残差构成的“受控不稳定性”机制突破消息传递神经网络的过平滑瓶颈，从而在异配图等挑战性场景下实现更鲁棒的链接预测。 |
| [^76] | [COTTI: Contrastive Training and Time-Aware Inference for Robust Industrial Recommendation under Promotional Drift](https://arxiv.org/abs/2611.22597) | COTTI 通过跨周期对比对齐、主动重加权和自适应边界对齐，将历史表征学习与促销事件时刻的分数调整相分离，在促销引起的分布漂移下实现鲁棒的工业推荐，并在飞猪数据上取得了更优的离线 AUC。 |
| [^77] | [Propose with the Gradient, Accept with the Loss: One Hedged Attack for Pixel-Sparse Robustness](https://arxiv.org/abs/2611.22596) | 该论文将稀疏对抗攻击视为“提议分布加接受规则”的局部搜索，基于泰勒界推导出五条何时以及如何信任梯度符号、候选像素排名和移除像素排名的规则，提出了一种融合梯度与随机搜索优点的对冲式攻击，从而更准确地评估图像分类器的像素稀疏鲁棒性。 |
| [^78] | [Lost in the Funnel: Selection-Induced Miscalibration in Ranking and Decision Systems](https://arxiv.org/abs/2611.22595) | 该论文提出了一个基于残差恒等式的逐边界诊断框架，揭示排序与决策系统中在模型边界处校准良好的分数会因下游选择过程（如过滤、重排序、业务规则和拍卖）而失效，并通过全行业案例研究展示了校准在决策时刻的恶化。 |
| [^79] | [PACT: Plan-Action Consistency Training by Preserving Correct Plans and Repairing Actions in GUI Agents](https://arxiv.org/abs/2611.22591) | PACT是一个针对GUI智能体的训练框架，它在结果监督强化学习中通过屏蔽已验证计划token上的负策略梯度来保护正确的规划，同时在推理上下文不变的情况下向参考动作修复执行行为，从而弥合计划与动作之间的执行鸿沟并解决信用分配冲突。 |
| [^80] | [ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language Models](https://arxiv.org/abs/2611.22576) | 提出ReMaRL，一个仅约30万参数的轻量级策略，在保持扩散语言模型冻结的情况下，通过GRPO强化学习直接优化最终答案正确性，学习决定哪些词元应被解掩码、保留或再掩码，在LLaDA-8B-Instruct上超越了现有的解掩码与再掩码策略。 |
| [^81] | [Balancing Exploration and Exploitation in Constrained Combinatorial Optimization with U(1)-Symmetric Tensor Trains](https://arxiv.org/abs/2611.22557) | 本文提出一个基于U(1)对称张量网络的约束组合优化通用框架，通过表达能力增强机制和基于图的多样性感知样本选择，在可行点稀少的硬约束问题上有效平衡了探索与利用。 |
| [^82] | [Physically Admissible Generative Flow Matching for Sparse Flow Reconstruction](https://arxiv.org/abs/2611.22549) | 该论文提出可行子空间一致性框架，通过切空间匹配将整个生成ODE轨迹约束在离散守恒流形上并保持动能度量，使稀疏流场重构中每个后验样本都严格满足不可压缩性与边界条件等物理可行性要求。 |
| [^83] | [Temporal Diffusion Models for Joint Signal-Channel Estimation under Gaussian and $\alpha$-Stable MIMO Channels](https://arxiv.org/abs/2611.22548) | 该论文提出一种时序扩散模型，在高斯与α-稳定MIMO信道下实现联合信号-信道估计，通过观测似然引导反向扩散过程，并使单个训练好的扩散模型无需重新训练即可复用于位置-尺度分布族。 |
| [^84] | [Testing World Models on Actions They Have Never Seen: Held-Out Plans, Forecast Accuracy and Price Rules in Retail](https://arxiv.org/abs/2611.22537) | 该论文提出了一种检验动作条件世界模型对从未见过的动作泛化能力的测试方法，并在零售定价场景的2,853个模型中发现：预测准确度仅能可靠地区分不同类型的模型，却无法区分同一世界模型的不同版本（在高需求周甚至出现反转），且JEPA预训练等学习动作效果的训练技术并未降低模型在新计划上的误差。 |
| [^85] | [Right Labels, Wrong Reasons: Signed Feature Attribution for Multi-Label Explainability](https://arxiv.org/abs/2611.22515) | 提出FLEX，一种基于多标签Fisher线性判别分析的全局代理解释器，能在单一连贯坐标系中为每个标签生成高保真度的带符号特征归因，尤其在标签多且正例稀少时表现优异。 |
| [^86] | [Efficient Self-Supervised Adaptation of 3D Abdominal Vision-Language Model for Institution-Specific HCC Classification via Full Fine-Tuning and PEFT](https://arxiv.org/abs/2611.22504) | 该研究首次将预训练3D腹部医学视觉-语言模型通过全量微调与PEFT两种策略成功适应到跨机构的肝细胞癌CT分类任务，证明即使面对领域偏移和数据受限，两种方法均能取得较强性能（全量微调AUC 0.82，PEFT AUC 0.79）。 |
| [^87] | [Manufacturing Differential Sparsity: Planner-Hardware Co-Design for Efficient World Model Planning](https://arxiv.org/abs/2611.22483) | 该论文发现量化世界模型预测器中相近动作会产生差分位级稀疏性，并提出规划器-硬件协同设计，通过参考森林采样主动“制造”稀疏性，从而大幅加速具身智能体的MPC规划。 |
| [^88] | [Escaping Latent Traps by Remembering: Training-Free Episodic Kernel Repulsion for World-Model Planning](https://arxiv.org/abs/2611.22470) | 提出一种无需训练的情景核排斥代价项，利用智能体已访问潜在状态的新颖性门控记忆惩罚候选计划，帮助世界模型规划摆脱局部极小值“潜在陷阱”，在两个JEPA世界模型和四个基准上将规划成功率显著提升。 |
| [^89] | [Sample-Aligned Decoders for Latent Generative Models](https://arxiv.org/abs/2611.22458) | 提出样本对齐解码器（SAD），通过使解码器适应生成潜在表示的分布而非仅重建编码器潜在表示，来阻止生成器的残差误差被渲染进最终图像，并解决了无配对监督与潜在空间漂移两大挑战。 |
| [^90] | [PACD: Provenance-Aware Consensus Decoding for Mitigating Cross-Modal Hallucinations in Omnimodal Large Language Models](https://arxiv.org/abs/2611.22456) | 提出了一种免训练的溯源感知共识解码框架PACD，根据模态分支间是否达成共识采用两条互补路径（如相关性引导的候选仲裁），有效缓解全模态大语言模型中的跨模态幻觉问题。 |
| [^91] | [How Many Training Labels Are Wrong? Auditing a Fitted Model through Gradient Allocation](https://arxiv.org/abs/2611.22449) | 该论文提出一种无需干净数据、保留集、阈值或再训练的方法，通过比较单个训练模型的逐样本梯度分配与各类平均分配来识别错误标签并估计训练集噪声率。 |
| [^92] | [Efficient Machine Learning Interatomic Potentials](https://arxiv.org/abs/2611.22444) | 该综述以原子簇展开（ACE/GRACE）这一统一数学框架为基础，系统性地梳理了高效机器学习原子间势的设计策略，并揭示各类MLIP架构的表面多样性主要源于非线性处理、层耦合与优化方式的不同，而非模型可表示的相互作用范围。 |
| [^93] | [Eyes in the Update? Learning Where Rewards Land in Multimodal Reasoning](https://arxiv.org/abs/2611.22438) | Gaze-RL 利用模型内部跨模态几何与局部预测不确定性构建潜在“奖励落点场”，无需外部标注器或过程标签，即可将轨迹级优势奖励重新分配到响应中视觉证据实际所在的位置，从而提升多模态强化学习的 token 敏感性。 |
| [^94] | [Encoding, Decodability, and Use: A Five-Layer Framework for Benchmark Measurement Audits with Evidence from UAV Intent Recognition](https://arxiv.org/abs/2611.22434) | 本文提出一个将基准测量审计分解为观测、表示、解码、使用与行为五个层级的框架，并通过无人机意图识别基准 AeroIntent 的实证表明，基准高分并不等于模型真正理解了所测量的构念。 |
| [^95] | [Do Diffusion Models Learn to Generalize Basic Visual Skills?](https://arxiv.org/abs/2611.22427) | 该论文提出视觉生成实验室（VGL）受控实验框架，通过基于规则生成的合成数据和任务特定指标，首次精确测量扩散模型对大小、位置、旋转、数量、颜色、形状等基础视觉技能的泛化能力，并揭示模型学习中的三种模式。 |
| [^96] | [InfuseLoop: Cross-Trajectory Latent Injection for Scalable Parallel Looped Transformers](https://arxiv.org/abs/2611.22426) | 提出InfuseLoop，一种跨轨迹潜在注入机制，通过轻量级注意力残差将后续预测轨迹中因果可用的隐藏状态注入当前迭代，解决了并行循环Transformer随深度扩展出现的训练损失尖峰和收益递减问题，同时保持token并行训练并仅需O(Rd)的额外存储。 |
| [^97] | [From Constraints to Causality: Petri net Encodings for Explaining Binary Neural Network Inference](https://arxiv.org/abs/2611.22396) | 本文提出一种组合式翻译方法，将固定的已训练二值神经网络的推理过程编码为1-安全Petri网，使每个引发序列都能成为其决策的因果轨迹，从而在传统SAT/SMT验证之外为BNN推理提供了因果层面的可解释性。 |
| [^98] | [Fair Bubble Sort: Provably Optimal Fair Ranking with Continuous Sensitive Attributes](https://arxiv.org/abs/2611.22352) | 本文提出连续成对公平性（CPP）这一新公平性概念，并设计了基于针对性相邻交换的公平冒泡排序（FBS）后处理算法，首次实现无需离散化处理的连续敏感属性公平排序，且证明了该算法在无权重Kemeny距离下精确最优。 |
| [^99] | [Scaling Laws for Vision-Language Fusion](https://arxiv.org/abs/2611.22351) | 本文通过系统分析多模态预训练的学习动态，推导出一个统一的缩放定律，将基础学习轨迹与模态竞争驱动的分布偏移相分离，从而为理解视觉-语言融合中早期融合与后期融合范式的根本权衡提供了原理性依据。 |
| [^100] | [BQ-LoRA: Binary-Quantized Low-Rank Adapters as Implicit Regularizers for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2611.22334) | 该论文提出 BQ-LoRA 框架，将 LoRA 适配器参数量化为单个二值比特并配合可学习的逐层缩放因子与直通估计器，在将适配器存储降低 13-16 倍的同时，从理论上刻画了其假设类的谱范数界与 Rademacher 复杂度，证明二值适配器可作为隐式正则化器实现参数高效微调。 |
| [^101] | [Multistage Contextual Optimization for Vehicle Repositioning](https://arxiv.org/abs/2611.22330) | 本文提出一种多阶段上下文优化方法，将车辆重定位建模为含宏观基本图的多阶段线性规划，并利用决策性能的精确梯度训练预测模型，以优化网约车平台的供需平衡。 |
| [^102] | [PVTSizing 2.0: A Fast and Reliable Analog Circuit Sizing Operator for Agentic EDA Workflows](https://arxiv.org/abs/2611.22328) | PVTSizing 2.0 提出了一个围绕可独立使用数值优化器构建的人-智能体协作工作流，结合损失引导更新、评论家引导的局部细化以及可选的 LLM 顾问，实现了快速可靠的跨 PVT 角点模拟电路尺寸设计。 |
| [^103] | [Zero-Dimensional Attention Topology for Linguistic Acceptability](https://arxiv.org/abs/2611.22312) | 本文提出用零维持续同调将144个对称化BERT注意力图压缩为1,440个拓扑特征用于语言可接受性分类，但实验表明其域外性能（MCC 0.438）低于注意力统计（0.508）和BERT本身（0.545），且该表示会丢失方向性注意力差异信息。 |
| [^104] | [VRLLM: Turning LLMs into Powerful Tabular Classifiers](https://arxiv.org/abs/2611.22308) | Visa Research通过在数十亿行带标签表格数据上微调Llama 3.2 1B模型（VRLLM），首次证明LLM在大规模表格分类任务上可以超越传统表格分类器以及生产级深度学习模型。 |
| [^105] | [Similar Is Not Useful: Selecting In-Context Demonstrations by Closed-Loop Utility](https://arxiv.org/abs/2611.22295) | 该论文提出用“闭环效用”（即演示对任务实际成功率的贡献）而非视觉相似性来为检索增强VLA策略选择上下文演示，实验证明效用选择的演示池成功率比相似性选择高出10.8个百分点，且在策略从未见过的任务上同样有效。 |
| [^106] | [Perception into Imagination: Executable Memory for World Models](https://arxiv.org/abs/2611.22294) | 提出可执行感知记忆（EPM）世界模型框架，在想象过程中同步推进显式感知状态与神经潜在动态，并将演化读出信息用于奖励预测、价值估计和策略学习，显著提升智能体在视野或接触丢失后利用过往感知信息的能力。 |
| [^107] | [ConfGuard: a physics-grounded reference monitor for predictive-maintenance large-language-model agents](https://arxiv.org/abs/2611.22288) | ConfGuard 提出了一种基于退化物理规律的参考监视器，通过保形预测校验预测性维护 LLM 智能体的剩余寿命预测是否符合物理约束（寿命不上升且每周期至多降一周期），从而抵御文档注入攻击和传感器扰动。 |
| [^108] | [XM-Sparse: High-Throughput Sparse-Attention LLM Serving over Scale-up Memory Pools](https://arxiv.org/abs/2611.22280) | XM-Sparse是首个基于扩展内存池的稀疏注意力KV缓存卸载系统，通过分析热KV缓冲区与索引器键缓存的不同访问模式并针对性优化，突破了HBM容量对长上下文大模型服务吞吐量的限制。 |
| [^109] | [Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration](https://arxiv.org/abs/2611.22275) | 该论文提出Heave-Ho去中心化框架，通过多智能体后训练实现物理耦合的人形机器人协调控制，结合视觉-语言规划器与基于语音的领导者-跟随者机制，在无需网络通信的情况下实现最多十台（含异构）人形机器人的长时程协作移动操作。 |
| [^110] | [OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions](https://arxiv.org/abs/2611.22262) | 该论文提出 OfficePrism 基准，通过 420 个覆盖七大企业业务职能的单域任务（结合中文工单、可执行参考见证及自动与人工筛选流程），在 124 种模型-框架-工具包配置下系统评估智能体，发现其自然语言与数据库联合通过率仅为 22.6%–76.4%。 |
| [^111] | [PCGRL for Complex Platformers: Synthesizing Playable and Diverse Levels via Inflated Envelopes and Forward Curricula](https://arxiv.org/abs/2611.22250) | 该论文提出一个物理感知、课程引导的PCGRL框架，通过“膨胀体积包络”机制将连续运动学可行性融入离散状态空间，并引入中间状态转移反馈来缓解稀疏奖励问题，从而在复杂平台跳跃游戏的大规模关卡生成中保证关卡的可玩性与多样性。 |
| [^112] | [OAK-Mind: Zero-Training Mental Health Change Inference from Passive Sensing via Knowledge-Grounded Multi-Agent LLMs](https://arxiv.org/abs/2611.22248) | OAK-Mind提出了一种零训练、多智能体、知识驱动的LLM流水线，无需标注数据或任务特定微调，通过基于心理健康文献解读睡眠、活动、移动等多维被动感知数据来推断心理健康变化。 |
| [^113] | [Spatially local mini-batching for implicit neural representations](https://arxiv.org/abs/2611.22236) | 本文证明低秩神经表示（LRNR）凭借其“局部更新仅通过每层少量系数作用于共享基”的结构特性，可以使用自适应网格单个单元内的点进行空间局部小批量训练，突破了传统INR必须在全域上采样训练的限制，并以Kolmogorov宽度论证提供了理论支持。 |
| [^114] | [From Temperature-Limited to Battery-Limited Legged Locomotion](https://arxiv.org/abs/2611.22231) | 该论文提出一种无需观测温度的热感知强化学习运动策略，通过奖励函数压低最热电机的预测稳态温度，使腿式机器人的持续运动从热关机限制转变为电池续航限制。 |
| [^115] | [Discovery of new antiviral agents through artificial intelligence: in vitro and in vivo results](https://arxiv.org/abs/2611.22230) | 本研究利用深度强化学习分子设计平台生成并筛选化合物，发现了DS-22-inf-009和DS-22-inf-021两个新型非唾液酸样结构的神经氨酸酶抑制剂，可在低微摩尔浓度下有效抑制甲型和乙型流感病毒（包括H275Y耐药突变株）。 |
| [^116] | [What Does Weight-Space Programming Add to Diffusion Editing? Program Construction versus Score-Guided Refinement](https://arxiv.org/abs/2611.22228) | 该研究发现，预注册对比下语言模型生成的权重空间程序并不优于强基线编辑器，但探索性实验表明一次性生成的程序可通过围绕受保护文本状态子空间组织更新方向，将扩散概念编辑在全新场景下的 VLM 联合成功率提升 10 分。 |
| [^117] | [SOTA or Luck? The Winner's Curse in LLM Leaderboards](https://arxiv.org/abs/2611.22227) | 该论文揭示了LLM排行榜中因排名噪声分数而产生的“赢家诅咒”与“排名诱导偏差”会系统性高估顶级模型的性能，并提出一种m-out-of-N自助法来纠正该偏差，在14个基准测试上排名第一模型的调整幅度为0.09至3.86个百分点。 |
| [^118] | [Learning Optimal Transport Plans Via Autoregressive Token Regression](https://arxiv.org/abs/2611.22218) | 该论文提出ToR-OT，一种基于Transformer的自回归标记回归模型，将最优传输问题重构为结构化序列生成任务，直接解码稀疏传输方案而无需在测试时求解优化问题，并展现出零样本泛化到未见分布的能力。 |
| [^119] | [Communicate the Change: Efficient Expert Parallelism via Activation Compression](https://arxiv.org/abs/2611.22215) | EchoMoE 利用 transformer 残差流以增量方式更新的特性，通过激活压缩只传递变化量而非完整隐藏状态，从而大幅降低专家并行中的同步集合通信瓶颈，且改造现有 MoE 模型仅需约完整预训练 1/8500 的训练 token。 |
| [^120] | [Sample-efficient Personalization of Large Language Model Alignment with Multidimensional User Preference Data](https://arxiv.org/abs/2611.22210) | 提出了多维偏好引导的在线直接偏好优化方法MPG-DPO，通过利用多维用户偏好数据来指导样本选择，解决了现有方法从LLM视角而非用户视角量化不确定性所导致的低效问题，实现了大语言模型对齐的样本高效个性化。 |
| [^121] | [Flocking Time and Graph Volume in Nonlinear Cucker-Smale Models with Dynamic Networks](https://arxiv.org/abs/2611.22204) | 本研究提出了结合动态网络与离散p-拉普拉斯算子的非线性Cucker-Smale模型，证明了其有限时间群集性并建立了群集时间下界与图体积的定量关系，同时通过基于transformer的PPO强化学习实验验证了严格时间约束下网络连接数增长的阈值效应。 |
| [^122] | [An Optimization Framework for Denoising Generation](https://arxiv.org/abs/2611.22202) | 该论文提出将去噪生成视为在固定对数密度景观上的梯度上升优化框架，并从理论上证明了上升轨迹几乎必然收敛到参考数据流形附近、不同初始化可保持输出多样性，且在特定正带宽条件下即使使用经验分数也能避免对训练样本的记忆化。 |
| [^123] | [RISA: Recoverable Sparse Attention for Block-Diffusion Language Models](https://arxiv.org/abs/2611.22191) | 提出RISA方法，通过为块扩散语言模型设计带共享读取计划的可恢复稀疏注意力机制，解决长上下文推理中KV缓存读取带来的内存瓶颈问题。 |
| [^124] | [Towards Optimism-Pessimism Trade-off in Model-based Offline-to-Online Reinforcement Learning](https://arxiv.org/abs/2611.22172) | 该论文提出一种双目标形式化方法，在离线训练阶段构建帕累托策略池，使在线微调时能够针对不同在线任务自适应地选择策略，从而解决基于模型的离线到在线强化学习中乐观与悲观的权衡难题。 |
| [^125] | [What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super](https://arxiv.org/abs/2611.22161) | 通过在五个专家构建的任务数据集上分别进行无监督阶段的强化学习后训练，NVIDIA Nemotron 3.5 Super 在十二项基准测试上取得了最高 +13.8 分的显著性能提升。 |
| [^126] | [Centroid Spectral Coupling: Implicit Alignment and Representation Diversity in Self-Supervised Learning](https://arxiv.org/abs/2611.22158) | 该论文提出质心谱耦合（CSC）——一种单项自监督学习目标函数，通过基于质心二阶矩矩阵的谱优化，隐式地同时实现视图对齐、表示多样性和抗坍缩，并给出精确分解与理论保证。 |
| [^127] | [When does deep reinforcement learning pay off in HVAC supervisory control? Four small Modelica case studies trained directly on FMUs](https://arxiv.org/abs/2611.22144) | 该论文通过四个基于Modelica模型并直接在FMU上训练SAC智能体的小型可复现HVAC案例研究，与调优的规则控制器及近最优基准进行对比，阐明了深度强化学习在HVAC监督控制中何时才真正划算。 |
| [^128] | [Synthetic Crystal Data Generation for Diffusion Pretraining in Material Property Prediction](https://arxiv.org/abs/2611.22140) | 该论文提出一种结合扰动-弛豫、MatterGen生成式采样等三种互补策略的合成晶体数据生成流水线，为材料性质预测的扩散预训练提供大规模且结构多样的弛豫结构，突破了依赖昂贵DFT弛豫获取预训练数据的瓶颈。 |
| [^129] | [ReMoE: An Interpretable Mixture-of-Experts Framework for Human Mobility Recovery Forecasting Across Disasters](https://arxiv.org/abs/2611.22133) | 提出了ReMoE，一个可解释的混合专家框架，通过特征引导的路由器自适应组合学习异构且可复用恢复类型的专家，实现跨灾害场景下的人类流动性恢复预测。 |
| [^130] | [LoCaR: Localized Canvas Retrieval With Frozen Vision-Language Model](https://arxiv.org/abs/2611.22131) | LoCaR 是一个无需训练的局部化文本到图像检索框架，利用 SAM2 生成的目标候选区域和冻结视觉编码器对每个区域独立编码，从而有效提升对小物体和局部属性的检索能力，并引入 LOCFG 基准数据集进行评估。 |
| [^131] | [Bandit Online Submodular Minimisation Using Zeroth-Order Oracles](https://arxiv.org/abs/2611.22130) | 本文提出了首个在bandit反馈下可保证跟踪移动最小化元的在线次模最小化零阶方法，其基于Lovász扩展的无偏两点梯度估计并建立了动态遗憾界。 |
| [^132] | [TempQuant: Learning Adaptive Temperatures for Quantized Language Models](https://arxiv.org/abs/2611.22128) | TempQuant通过学习量化模型的采样温度和top-p设置，使其下一词元概率分布更接近全精度模型，从而有效缓解量化带来的性能损失，且温度对量化模型性能的影响远大于全精度模型。 |
| [^133] | [What MoE Routing Controls: Causal Interventions on Expert Trajectories](https://arxiv.org/abs/2611.22120) | 该论文通过对MoE路由器logit进行因果干预发现：引导“格式”专家路径可以实际改变模型生成的文本，而引导“概念”专家路径仅能改变下一词元概率分布却几乎不影响最终生成内容，从而揭示了专家路由对模型输出控制的层级差异。 |
| [^134] | [VTT: Reading Vehicle Size and Powertrain Type from Motion Trajectory Across Unseen Car Models](https://arxiv.org/abs/2611.22109) | 该论文提出VTT数据集与基准，可从未见车型的运动轨迹和CAN信号中识别车辆尺寸与动力系统类型，通过辅助物理目标、数据增强和任务加权将纯电动与燃油车判别的宏F1提升至93%、尺寸分类提升至67.1%。 |
| [^135] | [Every Measurement, Every Direction, All at Once: Multimodal Flow Matching for Molecules and Spectra](https://arxiv.org/abs/2611.22098) | 本文提出MOSAIC，一个多模态流匹配Transformer模型，通过对测量数据施加随机掩码，使分子三维结构与UV/IR/拉曼光谱之间实现任意到任意的双向生成，并在QM9S上达到88.8%的分子图恢复率，证明了单次实验可被充分挖掘为多方向的监督信号。 |
| [^136] | [When AUC Meets Generative Reconstruction: A Unified Principle for Balanced Visual Representation Learning](https://arxiv.org/abs/2611.22094) | 该论文提出将生成式重建重新表述为判别排序问题的统一表征学习范式，解决了对比学习与生成重建之间的目标冲突，实现了细粒度感知与判别能力兼顾的平衡视觉表征学习。 |
| [^137] | [BernBRS: Certified Backward Reachability for Deep Bernstein Neural Networks](https://arxiv.org/abs/2611.22083) | 本文提出BernBRS框架，通过分析激活模式对输入空间覆盖进行智能细化与精确分割，为深度伯恩斯坦神经网络同时计算后向可达集的认证内、外近似，实现了高精度且可扩展的后向可达性分析。 |
| [^138] | [MedGuardBench: A Patient-Oriented Benchmark for Personalized Guardrail Development](https://arxiv.org/abs/2611.22069) | 该论文发布了MedGuardBench——首个以患者档案和处方为条件的个性化用药安全防护基准（包含87种药物、5,001个场景），并提出个性化推理防护栏方法，使模型能够基于患者的真实信息而非表面线索做出安全判断。 |
| [^139] | [Addressing Exogenous Variability in Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2611.22067) | 提出了外生Dec-POMDP（ED-POMDP）形式化框架来刻画协作多智能体强化学习中的外生变异性与中介影响，并据此设计了CTDE兼容的LEICA算法，通过学习基于历史的上下文表示来改进策略更新。 |
| [^140] | [Two Counterexamples for e-means](https://arxiv.org/abs/2611.22065) | 本文给出两个反例，证明 e-means 算法在相同初始中心下最终损失可能高于标准 k-means，且其基于质量的聚类复活机制可能破坏全局最优解，导致期望损失之比无界。 |
| [^141] | [Weight-Space Programming Doesn't Need Transformers](https://arxiv.org/abs/2611.22055) | 在无需梯度训练的权重空间编程中，三个前沿LLM在自由设计架构时都不约而同地将知识编译为可查询的稀疏表，且其性能以约1/1800的参数量超越了Transformer，表明权重空间编程并不需要Transformer。 |
| [^142] | [A single inference process underlies the behavior and neural circuits of working memory](https://arxiv.org/abs/2611.22054) | 该论文提出 MemGP 理论，将工作记忆统一为对特征空间连续函数的概率推断过程，从而在计算、算法和神经环路三个层面同时解释了工作记忆的多种行为与神经现象。 |
| [^143] | [RiDEx: Reward-Informed Directional Exploration for Diffusion Reinforcement Learning](https://arxiv.org/abs/2611.22052) | RiDEx通过将局部探索集中在去噪速度场中与奖励相关的子空间，并利用等大反向扰动的奖励差异提供带符号的局部监督，从而替代低效的各向同性随机扰动，显著提升了基于奖励的扩散模型后训练的探索效率。 |
| [^144] | [DSADiff: Data Structure-Aware Diffusion for Crystal Structure Prediction via Mitigating Complexity- and Imbalance-Induced Biases](https://arxiv.org/abs/2611.22050) | 该论文提出了LAS机制分析框架，识别出晶体复杂性和元素组成不平衡是影响生成难度的两个关键因素，并据此提出数据结构感知扩散模型DSADiff，通过统一的生成难度表示来缓解生成偏差，从而提升晶体结构预测性能。 |
| [^145] | [RLDE: A Reinforcement Learning Framework for Interpretable Iterative Document Editing](https://arxiv.org/abs/2611.22039) | 提出了RLDE框架，将文档迭代修订建模为无参考的MDP问题，通过指针网络与PPO智能体结合冻结的BART模型，依据可读性、困惑度和语法等多目标奖励实现可解释的自动化文档编辑。 |
| [^146] | [GP--RL--MPC: Uncertainty-Aware Reinforcement Learning-Augmented Model Predictive Control with Gaussian Process Postdecision State-Value Functions](https://arxiv.org/abs/2611.22038) | 该论文提出GP-RL-MPC方法，以高斯过程回归学习的不确定性感知评论家函数作为模型预测控制的终端惩罚项，数据高效地考虑预测时域之外的成本，并通过可调风险系数在认知不确定性与偶然不确定性之间实现权衡。 |
| [^147] | [An AI-Based OSSE Framework to Evaluate Learned Physical Consistency: A Multi-Model Study Using FourCastNet, Aurora, Prithvi WxC, and GraphCast](https://arxiv.org/abs/2611.22024) | 本文提出首个基于AI的观测系统模拟实验（OSSE）框架，通过Charney猜想评估四种不同架构的AI天气模型（FourCastNet V2、GraphCast、Aurora、Prithvi WxC）能否像传统物理模型一样对新观测产生相当的物理响应。 |
| [^148] | [Computed Permutation Trellis Coding: Near-Optimal Quantization from Two Multiplications and a 64-Byte Table](https://arxiv.org/abs/2611.22019) | 提出CPTC量化格式，其解码器仅需两次乘法和一个64字节表即可实现近最优的低位宽张量量化，在4比特时距信息论理想仅差0.07比特，同时支持1至6比特所有位宽并精确复现标准整数与浮点格式。 |
| [^149] | [SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory](https://arxiv.org/abs/2611.22017) | SkillBuffBench 通过将技能访问方式（无技能/精选技能/自动调用）、记忆开关、四个智能体框架和多个主流模型进行交叉配对评估，系统性地衡量了智能体技能在贴近真实部署场景下的实际贡献。 |
| [^150] | [On Formal Counterfactual Explanations](https://arxiv.org/abs/2611.22010) | 本文将最近世界语义下的反事实解释刻画为析取范式，并提出合取反事实解释（CFXps）这一新概念，建立了CXps、AXps与CFXps之间的结构联系，从而架起了反事实解释与形式化XAI之间的桥梁。 |
| [^151] | [How Much Distribution Shift Can a Clustering Withstand? Exact Split and Merge Certificates for Structured Transports](https://arxiv.org/abs/2611.22000) | 该论文基于确定性退火的相变分析提出局部稳定性准则，为聚类在结构化分布偏移下的分裂与合并事件给出精确证书，从而量化聚类对分布偏移的鲁棒性。 |
| [^152] | [Deciding Without Pointing: A Case Study of Whether a Decision Model Localizes the Anomalies It Detects](https://arxiv.org/abs/2611.21995) | 该案例研究发现，决策模型Clef-flash虽在工业异常检测上表现不俗，但其决策头的注意力并不能定位所检测到的缺陷，而骨干网络的注意力定位能力在改变问题或移除文本上下文时会显著退化。 |
| [^153] | [Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation](https://arxiv.org/abs/2611.21983) | 本文提出“三时钟”框架，将语言模型在层深度、训练过程和生成过程三个尺度上的行为与非线性动力学的经典概念（不动点、分岔、同步、混沌等）系统对应，并通过汇总约200篇文献的“三时钟词典”和一系列降阶模型在两个学科之间架起桥梁。 |
| [^154] | [Modeling Document and Section Hierarchies in Embedding Space](https://arxiv.org/abs/2611.21980) | 该论文提出层次化嵌入式主题模型，通过嵌入空间中的层次先验将文档级与章节级主题分布相耦合，并借助摊销变分推断在法律、报告和论坛等多个领域的结构化语料库上取得了最低或接近最低的困惑度表现。 |
| [^155] | [Sparse Integer Fourier Transform and Its Application In Machine Learning](https://arxiv.org/abs/2611.21976) | 本研究提出稀疏整数傅里叶变换（SIFT），通过将DFT矩阵分解为稀疏整数矩阵并用优化策略近似旋转因子，将基数32 FFT的算术成本降低75%，同时保持极低的误差水平。 |
| [^156] | [Beyond Bitwidth: Deployment-Constrained Mixed-Precision Quantization for Multilingual LLM Inference on NVIDIA Blackwell](https://arxiv.org/abs/2611.21974) | 该论文在NVIDIA B200上对Qwen3-8B进行了部署约束驱动的低比特与混合精度量化实测评估，发现INT2/INT3全面退化且模块恢复效应具有强非加性，最终提出一个NVFP4与INT8混合的配置，在通过全部预注册多语言与推理精度门槛的同时将权重内存压缩至5.917 GiB。 |
| [^157] | [Solver to System: LLM Reasoning Across Representations and Execution in Scientific Simulation](https://arxiv.org/abs/2611.21964) | 该论文提出了Solver2System数据集，通过多表示形式、受控损坏与词汇扰动来评估大语言模型在PDE科学仿真中的跨表示一致性与物理无效性检测能力。 |
| [^158] | [OCCAM: Improved Single-Cell Representation Learning and Perturbation Effect Prediction](https://arxiv.org/abs/2611.21959) | Occam是一个结合去噪细胞表示与基因相互作用、文本及蛋白质序列特征的MLP框架，在单细胞表示学习和跨环境基因扰动效应预测方面均达到了最优性能。 |
| [^159] | [Weight-space programming for tractor guidance controller: large language models natively write neural network weights that steer a real tractor](https://arxiv.org/abs/2611.21958) | 大语言模型通过权重空间编程直接读取、修复并从零编写拖拉机导航控制器的神经网络权重，无需优化器或梯度下降，在模拟和真实拖拉机上均实现了成功的路径跟踪。 |
| [^160] | [ThermLeT: Transformer-based Temperature Prediction for 2.5D Chiplet Architecture](https://arxiv.org/abs/2611.21954) | ThermLeT是一个基于Transformer的多模态代理模型，能在13毫秒内精准预测2.5D芯粒架构的稳态温度，比传统有限元求解器快约四个数量级，精度显著优于ThermEDGe和HotSpot。 |
| [^161] | [A Log-Timescale Perturbation Bound for Selective State Space Models at Every Sequence Length](https://arxiv.org/abs/2611.21952) | 本文提出了一种对数时间尺度上的扰动界，证明选择性状态空间模型（Mamba）中输入的微小变化对隐藏状态的影响在任意序列长度下均有界、误差不随序列长度累积，且无需对时间尺度设置下界。 |
| [^162] | [Conserved Verdicts, Drifting Commitments: Identifying Harmful Overthinking](https://arxiv.org/abs/2611.21946) | 该论文提出“承诺-判定差距”框架来识别推理模型中的有害过度思考，证明仅凭最终正确性探针估计的“判定”无法预测继续思考的利弊，而“承诺”与“判定”之差恰好量化了停止思考可获得的准确率增益，并给出将准确率-预算曲线分解为链内答案变化与链存活变化的恒等式。 |
| [^163] | [Benchmarking Deep Learning Models for Wheelchair Configuration Classification in Accessible Automated Vehicle Applications](https://arxiv.org/abs/2611.21945) | 本文提出了轮椅配置图像数据集（WCID），通过对手动/电动、轮椅子类型、乘坐状态等多维度属性进行细粒度标注，为无障碍自动驾驶车辆应用中的轮椅配置分类深度学习模型提供了基准测试资源。 |
| [^164] | [Does the Model Use What It Cites? Counterfactual Tests of Grounded Reasoning Chains in Small Video-Language Models](https://arxiv.org/abs/2611.21940) | 本文提出一种无需标注的“引用差距”反事实检验方法，通过涂黑模型推理链中所引用的视频区域来验证小型视频语言模型是否真正依赖其引用的证据，结果显示模型在CLEVRER上具有较强证据依赖性，但在NExT-QA上依赖性很弱。 |
| [^165] | [HubEgo-100K: In-the-Wild Egocentric Video of Skilled Work for Robot Learning](https://arxiv.org/abs/2611.21938) | HubEgo-100K 是一个包含 10,124 小时视频的数据集，由 12,227 名贡献者在 15,548 个真实环境中录制熟练的未经编排工作，旨在通过环境和物体的多样性提升机器人模仿学习的泛化能力。 |
| [^166] | [Higher-Order Interaction Reconstruction via Integral Physics-Informed Neural Networks](https://arxiv.org/abs/2611.21935) | 提出积分物理信息神经网络框架，以时间区间上的积分形式代替基于导数的物理残差惩罚，并结合交互张量的连续参数化，从而实现复杂网络中高阶交互的有效重构，同时抑制噪声并增强训练稳定性。 |
| [^167] | [Physics-informed neural networks for spacecraft trajectory optimization with exact boundary conditions](https://arxiv.org/abs/2611.21933) | 该论文提出一种利用轨迹优化问题数学结构的简单变换方法，将边界条件先验地精确嵌入神经网络解空间中，避免了标准物理信息神经网络中加权损失项敏感导致的收敛性和精度问题。 |
| [^168] | [V2V Communications Using Blockchain-Enabled 6G Technology and Federated Learning](https://arxiv.org/abs/2611.21932) | 该研究提出了一种融合区块链、联邦学习与延迟资源分配优化的6G赋能V2V通信系统，在数据包投递率、模型准确率和延迟等指标上优于现有技术，并通过共识优化和k-匿名实现了安全与隐私的双重保障。 |
| [^169] | [Polylogarithmic Regret via Certainty Equivalence in Degenerate LQR](https://arxiv.org/abs/2611.21927) | 该论文证明在采用奇异输入惩罚的退化LQR问题中，带有短期初始探索的确定性等价算法在控制器可辨识条件下即可实现 O(log²T) 的多对数遗憾，相比正定输入惩罚大幅放宽了可辨识性要求，显著扩大了可获得多对数遗憾保证的问题范围。 |
| [^170] | [Wrong Input, Wrong Threshold: Re-evaluating Adversarial Facial Privacy](https://arxiv.org/abs/2611.21923) | 论文揭示了现有对抗性面部隐私评估协议的两个独立缺陷——错误的输入方式（直接使用整图而非真实部署系统的检测、对齐、裁剪流程）和会饱和的成功率差值度量，并提出用配对余弦偏移这一新度量重新评估，发现现实前端会使标准迁移攻击的隐私保护效果损失56%。 |
| [^171] | [Latent Fact Composition under Next-Token Training](https://arxiv.org/abs/2611.21921) | 本文提出 Baton 系列循环 Transformer，通过在第二次循环中将中间实体作为词元嵌入传递，使逐词元训练的模型能够回答中间实体从未在两跳训练样本中出现过的两跳推理问题，并归纳出决定各模型成败的三个设计条件。 |
| [^172] | [KVUse: Reusable-KV User Memory for Efficient LLM Recommenders](https://arxiv.org/abs/2611.21920) | KVUse提出将LLM推荐系统中的用户画像以可复用的KV缓存而非文本形式存储，并通过自蒸馏训练轻量级LoRA写入器来生成这些KV缓存，从而消除排序调用中对用户画像文本的重复预填充开销，显著提升LLM推荐系统的服务效率。 |
| [^173] | [ALOHA Lightning: Learning Fast and Precise Manipulation](https://arxiv.org/abs/2611.21919) | ALOHA Lightning系统通过动态示教高速收集人类演示，并结合测试时动作平滑与掩码视觉对齐的学习流程，实现了快速且精准的双臂机器人操作。 |
| [^174] | [Lost in a Few Heads: Diagnosing and Repairing Block Selection in Sparse Attention](https://arxiv.org/abs/2611.21917) | 该论文提出一种校准探针方法，诊断出稀疏注意力的长上下文检索损失源于少数“盲”注意力头——其平均池化的键摘要稀释了关键 token 导致排序偏低而非预算不足，并在 InfLLM-V2 与 NOSA-8B 中验证了这一现象。 |
| [^175] | [A Temporal Algebraic Framework for Data Stream Evaluation Under Delayed Labels and Distribution Shifts](https://arxiv.org/abs/2611.21914) | 该论文提出将基于混淆矩阵的代数框架扩展到数据流评估，通过指数衰减或滑动窗口进行时间加权，并以观测掩码和校准概率处理延迟与选择性标签反馈，从而在分布漂移下以每步 O(m²) 的代价区分决策边界退化与类别流行度变化。 |
| [^176] | [World-Grounded Reinforcement Learning for Pollution-Aware Bird Movement Modeling Using a Multi-Instance DQN Framework](https://arxiv.org/abs/2611.21895) | 该论文提出一种多实例DQN强化学习框架，以AQI作为世界反馈信号，结合真实环境数据模拟污染感知的鸟类自适应导航策略。 |
| [^177] | [WM-Quant: Rollout-Aware Post-Training Quantization for JEPA-Style World Models](https://arxiv.org/abs/2611.21882) | WM-Quant是首个针对JEPA风格世界模型的后训练量化框架，通过早期编码器精度、控制路径保持和感知推演缩放三项技术，解决了自回归MPC推演中量化误差逐步累积导致动作排序近乎随机的问题。 |
| [^178] | [Cancellation-Free Constrained SquareCB for Safe Controller Selection](https://arxiv.org/abs/2611.21880) | 该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。 |
| [^179] | [Adaptive Memory for Online Reinforcement Learning under Sparse Model Changes](https://arxiv.org/abs/2611.21879) | 针对部署间稀疏变化的强化学习环境，提出一种自适应记忆算法，为每个状态-动作对动态决定复用多少历史观测，从而在利用过去数据与避免已变化条目带来的估计偏差之间取得平衡。 |
| [^180] | [PromptPack: Training-Free Augmentation of Diffusion Editors for Intent-Aligned Spatial and Semantic Control](https://arxiv.org/abs/2611.21865) | PromptPack 通过免训练地为扩散编辑器引入“保留/允许/修改”像素角色划分和区域到提示的空间映射两种显式编辑指南，实现了与用户意图对齐的空间与语义精确控制。 |
| [^181] | [tmol: a GPU-accelerated implementation of the Rosetta energy function for machine learning](https://arxiv.org/abs/2611.21857) | tmol是一个GPU原生的PyTorch库，实现了最先进的RosettaGenFF力场，支持高效的批量可微分打分与全结构优化（吞吐量最高提升362倍），可作为奖励函数加速生成式蛋白质设计。 |
| [^182] | [The Hidden Cost of MoE Diffusion Decoding: Profiling and Fixing the Sampler](https://arxiv.org/abs/2611.21852) | 该论文发现量化后MoE扩散语言模型的解码瓶颈在于画布级logits采样流水线而非专家矩阵计算，并提出两个无损优化层级（比特精确与精度对等）将其消除，从而大幅加速解码。 |
| [^183] | [When Does Discrete Ricci Curvature Help? An Edge-Level Analysis of Curvature-Enhanced GCNs for PPI Prediction](https://arxiv.org/abs/2611.21849) | 本文通过边级分析揭示，曲率增强GCN在PPI预测中的性能提升主要来自低度数边，而低曲率瓶颈边的性能反而系统性下降，从而挑战了“曲率加权通过缓解过度挤压带来改进”这一传统解释。 |
| [^184] | [Inventa: Decentralized Self-Improving Agents for Autonomous Scientific Discovery](https://arxiv.org/abs/2611.21847) | Inventa是一个去中心化的自动化科学发现框架，通过自主研究单元生成候选想法谱系、编排智能体自适应调整变异策略、以及验证器筛选并共享经验证的知识，解决了多智能体系统中想法趋同、错误信息传播和无效搜索的问题。 |
| [^185] | [Bias-Reduced and Parallel Heavy-Tailed Stochastic Convex Optimization in $\ell_p$ Geometries](https://arxiv.org/abs/2611.21844) | 本文在重尾噪声模型下研究 ℓ_p 几何中的并行随机凸优化，提出了一类结合随机去偏差技术与多层蒙特卡罗方法的新颖 ODC_p 算法，可高效估计一致凸目标函数的最优值并给出相应的收敛速率。 |
| [^186] | [Quantum-Classical Separation in Probability Distribution Transport via Nonlocal Games](https://arxiv.org/abs/2611.21840) | 该论文建立了一个量子-经典分离：常数深度的局域量子电路能够精确且可逆地连接两个经典概率分布，而经典随机输运实现同样的连接需要大得多的空间范围，并借助非局域博弈给出了图上的一般性构造原理。 |
| [^187] | [VisHead-Spec: Head-Routed Visual Attention for Multimodal Speculative Decoding](https://arxiv.org/abs/2611.21839) | 该论文提出VisHead-Spec，通过发现草稿模型中仅需少量键值头直接接收目标模型视觉注意力的稀疏模式，实现了高并发场景下依然高效的多模态投机解码加速。 |
| [^188] | [Coordination of Heterogeneous Teams with Uncertain Task Requirements](https://arxiv.org/abs/2611.21828) | 本文提出了基于CAAT图表示和图神经网络策略的分层规划器，能够联合协调异构智能体团队同时进行任务需求发现与任务执行，比独立优化两者的基线方法更快地完成任务。 |
| [^189] | [Urban Representation Learning for Identifying Potential Disparities in Crime Reporting](https://arxiv.org/abs/2611.21825) | 提出了一种自监督城市表征学习框架，通过邻居画像差异（NPD）指标量化各区域与其表征相似同伴区域的报告画像差异，从而识别开放犯罪数据中的潜在报告偏差。 |
| [^190] | [Data-Driven Variable-Length Compression of 3D Representations](https://arxiv.org/abs/2611.21802) | 提出一种数据驱动的可学习变长压缩方法，利用数据集整体结构并根据单个形状复杂度自适应调整压缩表示大小，仅通过快速前向推理即可改善网格和带符号距离场的率失真性能。 |
| [^191] | [Chain-of-Thought Controllability Can Reduce Chain-of-Thought Monitorability](https://arxiv.org/abs/2611.21799) | 论文通过人设控制训练发现，即使训练中不含任何规避监控的示例，仅仅让模型学会以不同人设口吻书写推理，就能显著降低其在执行隐蔽副任务时思维链的可监控性（g-mean²从0.94降至0.22），表明思维链可控性的增强可能削弱思维链监控的有效性。 |
| [^192] | [Regularized Score Centering](https://arxiv.org/abs/2611.21791) | 本文提出正则化分数中心化（RSC），通过引入反向 KL 惩罚使学习器的下一词元分布保持接近行为策略，从而限制策略不匹配、稳定异步离策略强化学习，并在理论上证明 KL 正则化改善了收敛保证。 |
| [^193] | [Better Calibrated, Better Planned: Uncertainty-Aware Multi-ASV Informative Path Planning for Oil Spill Monitoring](https://arxiv.org/abs/2611.21783) | 本文将证据深度学习（EDL）引入作为经不确定性校准的信息模型，研究不确定性校准对多ASV协同信息路径规划的影响，证明模型校准质量直接决定石油泄漏监测的规划效果。 |
| [^194] | [Learning Calibrated and Samplable Predictive Distributions of Surface-Ocean $\Delta p$CO$_2$ Using Tabular Diffusion Models](https://arxiv.org/abs/2611.21769) | 该论文提出使用基于梯度提升树评分函数的条件扩散模型Treeffuser，从13个环境、空间和时间预测变量中学习海表ΔpCO₂的完整条件预测分布，在SOCAT测试集上实现了最低的CRPS（8.229 μatm）和最接近名义水平的覆盖率，为海表CO₂估算提供了校准且可采样的不确定性量化方法。 |
| [^195] | [Margin Dynamics for Large Language Model Alignment](https://arxiv.org/abs/2611.21766) | 本文通过对齐方法在概率单纯形上的向量场建模并求解一维间隔常微分方程的闭式解，首次从理论上揭示 DPO 间隔无界增长而 IPO 间隔收敛于有限值的本质差异，并预测了标签噪声导致 DPO 定性相变及非对称加权下梯度集中对可训练性的影响，在五个模型家族和四个基准上得到验证。 |
| [^196] | [Bookkeeping Is Not Judgement: EJBench, an Engineering Judgement Benchmark Built from a Conjugate Heat Transfer Design Campaign](https://arxiv.org/abs/2611.21707) | 本文提出EJBench——一个基于电池冷板共轭传热设计活动构建的与软件无关的工程判断基准，考察模型能否从少量已有结果预测设计在新工况下的行为并做出决策，结果发现最佳前沿模型仅得0.71分，且模型答错时表达的置信度与答对时几乎相同。 |
| [^197] | [Video recordings and play behaviour annotations of group-housed dairy calves on two Dutch farms](https://arxiv.org/abs/2611.21706) | 该论文发布了一个开放数据集，包含荷兰两个奶牛场群养犊牛的651小时视频录像以及个体层面的玩耍行为标注，为犊牛积极福利研究提供宝贵资源。 |
| [^198] | [Disambiguating Moral Circuits and Moral Features in Language Models: Moral content, normative judgment or cue words? Evidence from 14 models](https://arxiv.org/abs/2611.21705) | 通过227对最小短语对和14个开源模型的实验，研究发现改变语言模型道德答案的注意力头并非道德专属，而是属于一个对所有类型规范性判断共用的通用评估回路，其效应与“道德上”这一线索词无关。 |
| [^199] | [Hybrid Quantum Selection of Learning Rates in Language-Model Training: Termination Criteria and an Interaction-Guided Schedule Oracle](https://arxiv.org/abs/2611.21701) | 本文提出一种经典-量子混合训练框架，结合Dürr-Høyer最小值查找与BBHT搜索来选择层组学习率，并创新性地引入无需预计算分数表的交互引导调度预言机，通过叠加态评分与振幅放大高效寻找语言模型微调的最佳联合学习率配置。 |
| [^200] | [FACE: Flow-Guided Angular Channel Estimation Framework for mmWave MIMO Systems](https://arxiv.org/abs/2611.21698) | 提出了基于流匹配的流引导角度信道估计框架FACE，利用毫米波信道的角度域稀疏性作为生成式信道先验，其估计精度在不同配置下均优于传统估计器和基于扩散的生成式估计器，且在导频较少时仍保持高精度。 |
| [^201] | [Layerwise Learning-Rate Scaling in Vision Networks: Effective-Rate Identification, Power-Mean Aggregation, and Budget Dependence](https://arxiv.org/abs/2611.21696) | 该论文发现从零训练视觉网络时，逐层学习率策略的效果可以由一个直接从乘数配置计算出的有效学习率来概括，短预算下的学习率效应主要源于优化进展，且对基础学习率的大小高度敏感。 |
| [^202] | [Stop Early, Stay Simple: Evidence Acquisition for Power System Fault and Attack Diagnosis](https://arxiv.org/abs/2611.21695) | 该论文提出ActiGrid主动证据获取基准框架，通过实验发现学习型DQN证据选择策略相比简单的熵停止等基线并无显著优势甚至被其帕累托支配，表明电力系统故障与攻击诊断中简单方法往往已足够。 |
| [^203] | [TIBECG: Temporal Information Bottleneck for Lead-Specific Demographic Privacy in ECG Waveform Release](https://arxiv.org/abs/2611.21694) | TIBECG通过时序信息瓶颈机制，在保留心电图诊断效用（宏F1不低于95%）的同时，显著抑制了波形发布中年龄、性别等人口统计信息的泄露，隐私增益较最佳传统变换方法中位数提升22.08个百分点。 |
| [^204] | [AdaFlowLite: Scalable and Non-blocking Inference on Asynchronous Mobile Data](https://arxiv.org/abs/2611.21691) | 该论文提出AdaFlowLite，通过机会主义推理机制实现异步移动多模态数据部分到达即可执行的可扩展无阻塞推理，并率先提出在开放世界移动环境中以计算方式控制模态亲和性的结构化方法。 |
| [^205] | [DeltaHarness: Self-Evolving Harnesses for Resource-Adaptive Tool Execution](https://arxiv.org/abs/2611.21688) | DeltaHarness是一种执行引导的策略搜索方法，通过QHI循环利用冷启动实测数据自演化智能体执行框架，使工具执行延迟在视频编辑等任务上最高降低30%。 |
| [^206] | [Edge-Adaptive Hiera-Seg Model for Building-Integrated Photovoltaic Power Forecasting](https://arxiv.org/abs/2611.21680) | 本文提出边缘自适应HieraSeg模型，通过多朝向视觉-辐照度融合，并在缺失帧、延迟变化和局部遮挡条件下自适应调整特征权重，实现短期建筑一体化光伏功率预测。 |
| [^207] | [RTLReviewBench: LLM Hardware Reviewers Rarely Miss Bugs but Routinely Reject Correct RTL](https://arxiv.org/abs/2611.21678) | 该论文提出RTLReviewBench基准，包含114对经仿真验证的有缺陷与正确Verilog模块配对，揭示了LLM审查RTL代码时的不对称现象——很少放过真实bug却频繁错误拒绝正确代码，且合成变异体无法完全暴露这一问题。 |
| [^208] | [Improving Early Detection of Scientifically Interesting Radio Sources Using Enhanced Isolation Forest](https://arxiv.org/abs/2611.21674) | 本研究提出并评估了四种增强孤立森林变体，结合BYOL自监督学习提取的特征与MGCLS巡天数据，有效提升了具有科学价值射电源的早期检测与排序效果。 |
| [^209] | [Token Commitment as a Control Surface for Steering Masked Diffusion Generation](https://arxiv.org/abs/2611.21669) | 该论文提出将令牌提交策略（即每步提交哪些令牌和提交多少令牌）作为掩码扩散模型推理引导的新型控制界面，通过反事实目标感知提交和校准自适应预算两种机制，在语言和视觉-语言任务上实现了对生成过程的有效引导。 |
| [^210] | [Limite 1B - Violetto: Technical Report](https://arxiv.org/abs/2611.21646) | Limite 1B - Violetto 是一个仅 10 亿参数的数学推理模型，通过在精心策划的数学语料上从头训练，并结合知识蒸馏、监督微调与 RLVR 后训练，在 AIME 2026 上取得 94.01% 的成绩，性能媲美参数量数倍于它的模型。 |
| [^211] | [VASR: Vector-Aware Assembly Planning with Equivalent Structures and Repair-Cost Feedback](https://arxiv.org/abs/2611.20984) | VASR框架通过融合SVG向量编码器与视觉语言模型的特征，并采用带层级等价奖励和修复成本反馈的RHPO策略优化方法，在家具装配层级规划任务中显著超越了现有最先进方法。 |
| [^212] | [Auditing World-Model Explanations in Retail Forecasting: Faithfulness, Accuracy and Price Response](https://arxiv.org/abs/2611.20811) | 研究发现，零售预测中的世界模型解释即使通过了八项审计检查（如构造性、稳定性、价格方向和与真实销量的一致性），其促销提升估计仍可能高出真实值1.2至1.4倍，价格响应被夸大近5至6.5倍，表明通过常规检查并不保证解释的忠实性与准确性。 |
| [^213] | [Information Limits of Replay-Free Recurrent State Migration](https://arxiv.org/abs/2611.19345) | 本文刻画了循环模型在不重放历史数据的条件下进行状态迁移所必须保留的最小信息量，证明了带标量输入的可控d状态模型至多需要kd个额外实数坐标，且单个秩一替换方向即可要求全部kd个。 |
| [^214] | [SLICES: A Benchmark and Evaluation Framework for Self-Supervised Learning on Sparse ICU Time Series](https://arxiv.org/abs/2611.18868) | 该论文提出SLICES基准框架，通过统一固定数据处理流程、编码器架构和评估协议，首次实现了对稀疏ICU时间序列自监督学习方法的公平比较，并发现掩码自编码（MAE）在绝大多数评估设置中优于JEPA和对比学习。 |
| [^215] | [Asymptotic Margins and Feature Confinement in Conic Classification Heads under Softmax Cross-Entropy](https://arxiv.org/abs/2611.18507) | 论文为锥形分类头引入了可由二阶锥规划计算的渐近间隔，并证明该间隔的正负决定了softmax交叉熵训练下特征是否有界：间隔为正时损失仅在特征发散处趋近于零，间隔为负时损失强制且最优解位于一个显式球内。 |
| [^216] | [Score-based neural ordinary differential equations for reaction-diffusions](https://arxiv.org/abs/2611.18467) | 该论文提出了一种带物理权重的拉格朗日式得分驱动神经常微分方程方法，让粒子通过权重追踪质量变化，并以质量归一化的自洽残差作为训练目标，从而有效求解反应扩散方程。 |
| [^217] | [Neural Classifiers as Error-Correcting Codes](https://arxiv.org/abs/2611.18239) | 该论文提出统一框架，将监督深度分类刻画为香农意义下的纠错码译码，证明交叉熵优化等价于最大似然译码，深度网络在特征空间中隐式学习码本，并据此推导出无损量化、知识蒸馏和结构正则化的新准则。 |
| [^218] | [Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders](https://arxiv.org/abs/2611.18179) | 本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。 |
| [^219] | [EgoX: Egocentric Cross-Embodiment Manipulation with Embodiment Dreaming](https://arxiv.org/abs/2611.16728) | EgoX提出了一种第一人称视角的跨具身学习框架，通过为每个观测和动作模态配置专用分词器与专家模块、共享Transformer主干，并借助仅在训练阶段使用的“具身梦境”未来状态预测目标，使通用操作策略能够有效利用人类、真实机器人和仿真智能体的异构演示数据。 |
| [^220] | [What Can a VLA Learn from the Agent That Calls It? Task-Level Experience Distillation for Embodied Navigation](https://arxiv.org/abs/2611.16246) | H2E 提出会话级蒸馏方法，将 LLM 规划器与冻结 VLA 作为异构训练智能体，仅保留成功到达目标的会话并屏蔽偏离目标的步骤来训练 VLA，使其在测试时无需 LLM 即可克服走过目标、提前停止和偏离后难以恢复三种典型失败。 |
| [^221] | [AI4Plasma II: Geometry-Aware Surrogate Modeling for Variable Electrode Gap GEC Plasmas](https://arxiv.org/abs/2611.15584) | 该研究比较了六种几何感知代理模型在可变电极间距GEC等离子体仿真中的数据需求，以实现对完整等离子体场和晶圆附近离子密度分布的快速准确预测。 |
| [^222] | [ReShap: Interpretable Reinforcement Learning for High-Dimensional Multi-Action Settings](https://arxiv.org/abs/2611.15507) | ReShap 是一个策略蒸馏框架，通过从原始图像中提取特征、执行 SHAP 特征归因，并将深度强化学习策略蒸馏为在完整状态-动作对上训练的可解释代理模型，从而将基于 Shapley 值的可解释强化学习扩展到了高维、多动作的环境。 |
| [^223] | [AI4Plasma I: Data-Efficient Surrogate Modeling of Capacitively Coupled Plasma Fields](https://arxiv.org/abs/2611.15002) | 该研究系统比较了五种神经代理架构在电容耦合等离子体场预测中的数据效率，发现傅里叶神经算子（FNO）在绝大多数测试场景中以最少的训练模拟数量即可达到1%误差精度，是数据效率最优的架构选择。 |
| [^224] | [Rethinking "RL Generalizes, SFT Memorizes": The Role of SFT Data](https://arxiv.org/abs/2611.14912) | SFT泛化不佳并非模型固有缺陷而是数据问题——精心混合数据来源并匹配RL的正确采样数量或总采样预算，可使SFT的泛化能力追平甚至超过RL。 |
| [^225] | [Dual-Core Tensor Wheel Decomposition with Structure-Cached Acceleration for Tensor Recovery](https://arxiv.org/abs/2611.13938) | 该论文提出双核张量轮分解（DCTW）及结构缓存加速策略，通过将高阶中心核分解为两个由共享潜变量键连接的低阶核，并缓存张量网络收缩中的中间张量以避免显式构造大规模矩阵和冗余计算，从而大幅提升了张量恢复的计算与内存效率。 |
| [^226] | [Attention at Absolute Zero](https://arxiv.org/abs/2611.11937) | 该论文提出一种在零温度（硬最大值）下读取见证者的高阶配对注意力层，通过仅传递计数比较比特并在每轮循环重建状态，克服了软注意力在规模扩大时的权重分散、分数差距与状态漂移等失效模式，在比训练规模大1000倍的最大值检索任务上仍保持99.95%以上的准确率。 |
| [^227] | [Regularization Amplifies Subliminal Learning](https://arxiv.org/abs/2611.10397) | 本文发现正则化非但不能抑制反而会放大隐蔽影响——全量微调下通常不存在的潜意识学习，在低秩适应、显式惩罚或增加数据等正则化手段下反而会被显著诱发和增强。 |
| [^228] | [LEGIT: Live Eligibility Gating over Identifier Tries for Constrained Decoding of Heterogeneous Pages](https://arxiv.org/abs/2611.10357) | 提出 LEGIT 约束机制，通过对标识符字典树叶节点进行深度优先编号、把随请求和 token 动态变化的资格规则转化为高效的范围查找，使生成式推荐系统在整页解码时能精确且快速地保证每个语义 ID 真实、合规且不重复。 |
| [^229] | [Goodhart in the Planner: Sampling-Based Control Exploits Rewards Learned from Human Preferences](https://arxiv.org/abs/2611.09670) | 论文揭示并量化了“规划器中的古德哈特定律”：针对人类偏好奖励模型进行更强的 MPPI 搜索虽在标签准确率上保持高分，却使真实回报最多下降 76%，并进一步给出真实回报与搜索偏离距离（√KL）的关系以及 χ² 正则化 MPPI 在由评分器误差范围设定温度时的最坏情况最优性保证。 |
| [^230] | [NovoCRF: Globally Normalized De Novo Peptide Sequencing on a Prefix-Mass Lattice](https://arxiv.org/abs/2611.05260) | NovoCRF首次提出在所有质量可行肽段上以精确全局归一化方式训练的条件随机场模型，通过前缀质量格约束路径实现从头肽段测序，模型仅360万参数、训练约27 GPU小时，并可在推理时针对每次实验自适应微调。 |
| [^231] | [Thought-deciphering via hierarchical EEG-to-text alignment](https://arxiv.org/abs/2611.03623) | 提出一种分层脑电图-文本对齐框架，先解码粗粒度语义类别、再在类别内检索细粒度句子，显著提升了从EEG进行句子级解码的检索准确率。 |
| [^232] | [Forecasting How RLHF Responds to Reward-Model Edits Before Training](https://arxiv.org/abs/2611.03244) | 该论文基于KL正则化强化学习的一阶理论提出了一种协方差预测方法，能够在RLHF训练开始之前就预测出奖励模型修改将如何改变策略的行为，从而免去昂贵的试错式RL训练。 |
| [^233] | [Sampling Design and Confidence Tubes for Integral Curves on Riemannian Manifolds](https://arxiv.org/abs/2611.00771) | 本文在紧黎曼流形上提出了估计未知向量场积分曲线的被动与主动两种抽样方法，证明了被动核估计的曲线误差阶为 $n^{-2/(d+3)}$ 并给出由线性随机微分方程刻画的函数型中心极限定理，同时证明基于先导信息的主动抽样可将期望最大曲线误差提升至 $N^{-1/2}$ 阶，从而为构建置信管奠定基础。 |
| [^234] | [DAPS: Dependency-Aware Premise Selection for LLM Theorem Proving](https://arxiv.org/abs/2610.41536) | 该论文提出DAPS方法，通过首次系统挖掘53个Lean Blueprint项目构建依赖关系语料库，将数学中的依赖结构信号引入前提选择的训练数据与模型中，克服了现有基于语义相似度的选择器在检索偏斜和分布外查询上的固有局限。 |
| [^235] | [Formalizing the Intrinsic Dimension of Objective Landscapes by a Generalized Notion of Co-dimension](https://arxiv.org/abs/2610.39674) | 本文将目标景观的内在维度形式化为一种依赖于集合全局几何的广义余维数概念，并为浅层和深层ReLU网络学生-教师设置中的最优参数集合推导出内在维度的上下界。 |
| [^236] | [Seeing Is Not Enough: Knowledge-Grounded Active Inquiry from Uncertain Visual Evidence](https://arxiv.org/abs/2610.38774) | 该论文提出一个中医舌诊框架：当舌象与脉象证据不足以区分证候时，系统依据校准后的支持度下降触发主动症状问询，并利用问答动态更新证候候选列表，使整个判断过程可追溯、更可靠。 |
| [^237] | [Alpha-TCM: Auditable Knowledge Integration for Multimodal TCM Inference](https://arxiv.org/abs/2610.38754) | Alpha-TCM 提出一个可审计的多模态中医推理工作流，通过分离测量、知识、预测与审查决策，并利用贝叶斯委员会核算量化所注入知识（包括被损坏的错误知识）对预测的影响，实验证明知识背景在中低训练规模下能显著降低预测误差。 |
| [^238] | [Shared Context, Separate Engines: Streaming Vision--Language Inference on Edge SoCs](https://arxiv.org/abs/2610.35309) | 提出了一种在边缘SoC上协调CPU、GPU、NPU等异构加速器并发执行有状态流式VLM推理的系统，通过将流的版本化上下文与各引擎的KV状态分离，实现任意引擎可接续处理数据流，从而支持流式生成、并行序列生成和状态感知调度。 |
| [^239] | [Nowcasting aerosol particle size distributions from PM2.5 with physics-regularized flow matching](https://arxiv.org/abs/2610.33557) | 本文提出利用物理正则化流匹配的生成式深度学习方法，仅从PM2.5质量浓度和气象参数即时生成气溶胶粒径分布的合理集合，并以芬兰森林和城市两个对比站点为测试平台，对六种生成模型进行了系统性基准评测。 |
| [^240] | [Not All Mislabels Are Equal: Efficient Task-Aware Auditing with Confidence Sequences](https://arxiv.org/abs/2610.33262) | 该论文提出ATLAS框架，将任意时刻有效的置信序列推断与任务感知的自适应审计策略结合，根据标签不确定性及其对下游评分的影响优先审计数据点，从而以比随机抽样少11%-24%的审计量将标签错误率认证到±2个百分点的精度。 |
| [^241] | [Compositional neural networks for multimodal X-ray and neutron scattering inversion](https://arxiv.org/abs/2610.29362) | 该论文提出了一种基于先验摊销流匹配的模块化框架，通过在推理阶段用组合采样器融合分别训练的单观测后验估计器，实现多模态X射线和中子散射数据的概率性联合反演，无需联合再训练即可复用预训练模型。 |
| [^242] | [Fixed Sets and the Hardness of Computing Correlated Equilibria](https://arxiv.org/abs/2610.28141) | 本文证明计算多面体博弈的近似相关均衡是PPAD完全的（与纳什均衡同等难度），挑战了相关均衡比纳什均衡更容易求解的传统观点，并通过提出“不动集”这一不动点的自然推广建立了相关均衡与不动点理论的概念性联系。 |
| [^243] | [Exploration via Activation-Space Steering for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2610.27012) | 提出激活空间探索（ASE）方法，通过聚类正确与错误响应的激活差异构建潜在探索向量库，并将其应用于训练采样以引导模型探索多样化推理路径，从而提升可验证奖励强化学习在数学推理上的性能。 |
| [^244] | [MemT-1.0: Towards Memory-Native Evolution](https://arxiv.org/abs/2610.26825) | MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。 |
| [^245] | [World Models, Trees and Foundation Models for Retail Forecasting: Evidence from Four Public Data Sets](https://arxiv.org/abs/2610.25170) | 本文提出了一种在价格与促销计划下推进商品状态的世界模型，并在四个公开零售数据集上针对从单品单店订货到全链财务规划的不同决策粒度，系统比较了它与LightGBM、Transformer及TimesFM等时序基础模型的预测表现，借助配对检验与模型置信集确定各粒度下的最优模型。 |
| [^246] | [Dynamical Kernel Attention: Estimating the Present from a Record of the Past](https://arxiv.org/abs/2610.24026) | 本文提出动态核注意力（DKA），从随机微分方程控制的潜在状态的鲁棒估计中推导注意力机制，使每个注意力头提取有界证据、在证据不足时弃权并拥有输出变化可证明有界的记忆窗口，在语言建模上优于RoPE和ALiBi，并以更小的键值缓存匹配滑动窗口注意力。 |
| [^247] | [Does On-Policy Distillation Need to Complete Every Rollout?](https://arxiv.org/abs/2610.21337) | 该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。 |
| [^248] | [SarcasTraps: Effect of Sarcasm on Turkish Hate Speech Detection](https://arxiv.org/abs/2610.16957) | 本文构建了SarcasTraps——一个包含479条数据、沿仇恨言论与讽刺双维度标注的土耳其语外部测试集，并通过实验揭示讽刺会显著干扰仇恨言论检测：微调的编码器模型在讽刺性仇恨言论上召回率下降，零样本大语言模型同样受到影响。 |
| [^249] | [Natural Code Transformation Confuses Agents in Vulnerability Analysis](https://arxiv.org/abs/2610.14350) | 本文发现AI智能体在分析具有相同功能和相同漏洞的不同真实代码提交时会产出截然不同的结果，并据此提出从真实开发模式中挖掘自然且功能保持的代码变换，作为防御攻击者智能体的新手段。 |
| [^250] | [Do Neural PDE Solvers Learn the Right Dynamics?](https://arxiv.org/abs/2610.06952) | 该论文提出了一个超越传统预测误差评分的评估框架，通过考察误差形成、集合几何结构和极端事件三个互补维度，来检验神经PDE求解器是否真正再现了系统的动力学行为。 |
| [^251] | [OVAL: Output-Aware Local Page Bases for KV Cache Compression](https://arxiv.org/abs/2610.06686) | OVAL提出用一对输出感知的低秩基底同时充当KV页面的检索索引和压缩存储，使长上下文解码中KV缓存被丢弃精确页面后仍不增加任何额外内存开销。 |
| [^252] | [Trust the Direction, Search the Step: Zero-and-First-Order Methods for LLM Fine-Tuning](https://arxiv.org/abs/2610.02190) | 提出ZFO轻量级框架，将优化中的方向选择与步长选择解耦——用可信的一阶优化器确定方向，仅沿该一维子空间做两次零阶评估构建局部模型，从而以低于完整线搜索的开销实现曲率感知的自适应步长选择，并配有理论保证，适用于大语言模型微调。 |
| [^253] | [Misalignment of Low-Loss Regions Causes Grokking](https://arxiv.org/abs/2610.00620) | 论文提出基于模式连通性与低损失区域几何的分析框架，指出grokking现象源于训练集与验证集所诱导的低损失区域之间的错位，并以稳定的反grokking反例挑战了现有的相关性解释。 |
| [^254] | [One pool, many targets: a conservation layer and what archival data can identify](https://arxiv.org/abs/2610.00445) | 本文提出一个保证有限向导RNA加载的RISC池守恒的可微平衡层，并利用档案脱靶数据检验发现：平衡耦合虽使校正后的热力学亲和力以预测器所需方向与抑制效果弱相关，但在剂量与跨上下文实验中并未显示出超越独立评分的额外预测价值。 |
| [^255] | [Pseudo-Label-Triggered Retraining from Forecast Errors for Online Time Series Forecasting](https://arxiv.org/abs/2609.39789) | 本文提出PILOT框架，利用预测误差构建伪标签并训练轻量级评分器，从误差动态中学习时间序列预测模型“何时重训练”，从而在有限部署资源下有效维持预测性能。 |
| [^256] | [ElectrolyteFM: Unifying Electrolyte Property Prediction through Cross-Property Knowledge Learning](https://arxiv.org/abs/2609.39340) | 提出统一多性质预测模型ElectrolyteFM，通过独立学习性质特定表示、结合单独训练的专家池与路由机制捕获跨性质共享知识，解决了联合训练中跨性质干扰问题，实现电解液多种性质的更准确预测。 |
| [^257] | [Generative sequence modeling for infinite memory processes via predictive states](https://arxiv.org/abs/2609.38524) | 本文提出一种基于预测状态的生成式序列建模新方法，能够处理具有无穷记忆的随机过程，当过去历史可被压缩为低维充分统计量时实现快速收敛，且估计问题的统计复杂度仅由预测状态空间的内在维数决定。 |
| [^258] | [LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization](https://arxiv.org/abs/2609.38166) | LeapQuant 提出了一种无需训练的按窗口量化方法，通过跳过整个 token 窗口、仅在窗口末尾对循环状态量化一次来避免误差累积，从而在 8 位循环状态量化下实现近乎无损的线性注意力推理性能。 |
| [^259] | [Where Root Cause Analysis Fails: A Retrieval-Reranking Decomposition](https://arxiv.org/abs/2609.36686) | 该论文提出检索-重排序分解方法，揭示了top@k准确率混淆“检索失败”与“重排序失败”的评估盲点，并发现复杂故障基准上统计基线有79-100%的概率错误排序真实原因，而图方法从未明显胜过最佳统计基线。 |
| [^260] | [Representable but Unlearned: Encoding Rank and the Interaction-Prediction Floor](https://arxiv.org/abs/2609.36208) | 该论文提出一种无需标签或拟合模型即可量化输入编码秩缺陷所致预测误差下限的方法，并证明siRNA交互预测中仅训练期特征掩码造成的秩下降（140降至72）会产生占拟合模型误差14.6%的不可逾越下限，而仅恢复三个化学列或移除掩码即可完全消除该下限。 |
| [^261] | [ScAn-Bench: Evaluating Scaling Analysis Methodology](https://arxiv.org/abs/2609.35707) | 该论文提出了基于4524个语言模型检查点和8024个视觉-语言模型检查点的代理基准 ScAn-Bench，首次系统性地评估了跨不同数据模态下缩放分析中的数据获取与外推方法论。 |
| [^262] | [Intuition Vectors from Pretrained Representations Steer Diffusion-Based Reasoning](https://arxiv.org/abs/2609.32619) | 该研究发现，冻结自监督视觉编码器表示空间中的简单向量差（“直觉向量”）能够直接支持抽象视觉推理，无需依赖语言推理或任务特定的表示学习。 |
| [^263] | [Interpretable-by-Design Descriptor Portfolios Match a 2048-Dimensional Foundation Embedding on Low-Data Molecular Assays](https://arxiv.org/abs/2609.30789) | 该论文提出一种可解释性设计（interpretable-by-design）的描述符组合方法，通过贪婪拼接经过来源筛选的紧凑描述符块，在特征层面完全可审计的前提下，于低数据ADME/Tox检测上取得了与2048维CheMeleon基础嵌入相当的精度（平均AUC 0.762对0.764）。 |
| [^264] | [Graph neural networks and the energetic cavity method for combinatorial optimization](https://arxiv.org/abs/2609.07456) | 通过对图神经网络进行微小修改以融入经典平均场启发式方法（主特征向量法与能量腔/最小和算法），可显著提升其在求解伊辛模型基态及组合优化问题上的性能。 |
| [^265] | [Amortizing Scaling Law Construction Costs](https://arxiv.org/abs/2609.05016) | 该论文提出了一种基于贝叶斯优化的高效缩放定律构建框架，通过逐步扩展计算预算并结合代理模型幻想评估来恢复广泛的实验网格，大幅降低了构建缩放定律的计算成本。 |
| [^266] | [Representation Learning with Quantum Signal Processing](https://arxiv.org/abs/2608.28828) | 该论文将量子信号处理（QSP）确立为表示学习的可解量子模型，精确计算了其量子神经正切核的统计特性，并证明了稀疏数据下完整非线性梯度流收敛于可积标量流以及普遍成立的有限深度速度极限。 |
| [^267] | [Dandelion: A Spherical Flower for Neural Simulation of Planetary Dynamics](https://arxiv.org/abs/2608.27521) | 提出Dandelion——基于warp的神经PDE求解器Flower的球面版本，通过沿大圆传输特征并在球谐域中实现分层池化，构建了无卷积的球面神经网络架构，用于行星动力学的神经模拟。 |
| [^268] | [Marginal Coverage Credit Reduces Redundant Exploration in Parallel State-Entropy Optimization](https://arxiv.org/abs/2608.27507) | 本文提出MCC-PGPSE方法，通过结合留一策略覆盖与状态所有者专业化的边际覆盖信用来重新分配辅助内在奖励，从而减少并行状态熵优化中的冗余探索并促进互补性状态覆盖。 |
| [^269] | [Why ML-based cough models do not generalize: a systematic cross-dataset evaluation for tuberculosis screening](https://arxiv.org/abs/2608.25846) | 本研究通过系统跨数据集评估发现，基于咳嗽的机器学习结核病分类器无法泛化，其性能受录音设备和数据集特异性影响，而非疾病本身，提示需谨慎对待此类模型的实际应用。 |
| [^270] | [Fairness-Aware Test-Time Prompt Tuning](https://arxiv.org/abs/2608.25707) | 本文提出了一种公平感知的测试时提示调优方法（FairTPT），通过软提示调优在分布偏移下同时优化目标熵和虚假熵，以改善视觉-语言模型在子群体偏移中的公平性。 |
| [^271] | [A FEM-Based Surrogate Modelling and Optimization Framework for Physics-Constrained Electromagnetic Coil Design](https://arxiv.org/abs/2608.18903) | 本文提出了一种结合有限元模型与高斯过程代理的优化框架，在物理约束下实现了对电磁线圈设计的高效贝叶斯优化，并揭示了不同优化器在不同评估预算下的性能差异。 |
| [^272] | [Sampling Luck Masquerades as Allocation Gain: Auditing Test-Time Budget Allocation for Neural Combinatorial Optimization](https://arxiv.org/abs/2608.13087) | 本文发现神经组合优化中测试时预算分配的报告增益主要源于采样运气偏差，样本外验证表明非均匀分配并无实际收益。 |
| [^273] | [Evaluation Resolution Confounds Learning-Rule Comparisons in Model-Brain RSA of Early Visual Cortex](https://arxiv.org/abs/2608.12408) | 本文发现，在模型-大脑RSA比较中，评估分辨率的选择会显著影响不同学习规则（如未训练/局部训练与反向传播）在早期视觉皮层上的表现排名，高分辨率评估下反向传播的优势更明显，提示现有比较结果可能受分辨率混淆影响。 |
| [^274] | [One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL](https://arxiv.org/abs/2608.12253) | 该论文揭示了多智能体强化学习中使用单一冻结模拟器会导致策略泛化失败，并提出推理时的言语化采样和训练时的协同训练两种方法来解决模拟器崩溃问题。 |
| [^275] | [Can Vision Models Read the Radar Display? On the Feasibility of Radar Imagery for Air Traffic Complexity Estimation](https://arxiv.org/abs/2608.11810) | 本文探索了利用雷达图像作为视觉变换器输入来估计空中交通复杂性的可行性，通过编码飞机状态信息验证了深度学习模型能捕捉稀疏雷达图像中运行相关的复杂性变化。 |
| [^276] | [Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment](https://arxiv.org/abs/2608.11537) | 提出了一种通过可观测语义图像接口和层级生成器证据对齐的生成式语义分割方法，有效解决了颜色漂移和边界混合问题。 |
| [^277] | [Probing and steering biology across Boltz-1s trunk-diffusion boundary](https://arxiv.org/abs/2608.11475) | 该研究揭示了Boltz-1模型中，躯干编码的序列化学信息在扩散模块中被衰减，而几何结构信息保持传递，并通过干预躯干表征实现了对结构生成的引导。 |
| [^278] | [MAGIC-SSCIL: Manifold Anchoring and Geometric Incremental Calibration for Semi-Supervised Class Incremental Learning](https://arxiv.org/abs/2608.07586) | MAGIC通过在冻结特征的k近邻图上进行标签传播，并将传播置信度同时用于无标签损失加权和软加权几何校准的类别统计量构建，解决了极端少标签场景下半监督类增量学习中伪标签不可靠、回放统计量稀疏噪声的问题。 |
| [^279] | [Test-Time Coverage: Test-Conditioned Data Curation for Deployment-Aware Learning](https://arxiv.org/abs/2607.22697) | 该论文提出TTCov方法，在训练前利用测试侧信息构建由LLM原子命题及其频率组成的知识图集，将部署条件化的数据筛选分解为覆盖率与分布两部分，从而直接优化训练数据与部署测试分布的匹配度。 |
| [^280] | [Improved Convergence Rates for Stochastic Multi-Gradient Descent which Close the Gap: A Proof by AI](https://arxiv.org/abs/2607.18174) | 本文通过利用帕累托平稳性度量的 Lipschitz 连续性（而非 (1/2)-Hölder 连续性），证明了随机多梯度下降（SMG）在常数步长和线性增长批量下可将平方帕累托平稳性度量的收敛速率从此前的 $\tilde{O}(T^{-1/4})$ 提升至 $\tilde{O}(T^{-1})$，且无需有界梯度假设。 |
| [^281] | [OpenMHC: Accelerating the Science of Wearable Foundation Models](https://arxiv.org/abs/2607.16235) | 该论文发布了OpenMHC——目前面向研究人员开放的最大、最全面的可穿戴健康数据集（涵盖11,894名参与者、超过6700万小时记录、最多19个传感器通道和170个健康行为变量），并配套提供可穿戴基础模型的开源实现与模型权重，以推动可穿戴健康领域的开放科学。 |
| [^282] | [Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains](https://arxiv.org/abs/2607.13319) | OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。 |
| [^283] | [Statistical Adversaries: Natural Backdoor-like Adversarial Features in Clean Vision Datasets](https://arxiv.org/abs/2607.05516) | 本论文发现干净视觉数据集中自然存在的统计信号可作为后门样触发器，揭示了数据集结构和分布导致的模型漏洞，而非恶意插入。 |
| [^284] | [Localized LoRA-MoE: Mixture of Low-Rank Experts With Adaptive Routing](https://arxiv.org/abs/2607.05114) | 提出Localized LoRA-MoE框架，将局部空间分块与上下文条件自适应专家路由相结合，提供了集中式块级路由和去中心化单元级路由两种架构变体，以解决传统PEFT方法中的梯度干扰和静态拓扑无法适应上下文切换的问题。 |
| [^285] | [Annealed Gibbs Measures with Data-Driven Interaction Potentials on Hierarchical Structures](https://arxiv.org/abs/2606.30064) | 该论文在Cayley树上构造了由训练数据（经验损失）确定耦合函数的退火吉布斯测度，并推导出有限体积联合密度的充要相容性条件（即边界法则的非线性积分方程），证明了在 $k=1$ 时方程存在唯一的归一化正解。 |
| [^286] | [Theory-Scale Auto-Formalization of Logics for Computer Science](https://arxiv.org/abs/2606.26525) | 本文提出了LCS-Bench，一个基于《计算机科学逻辑》的理论级自动形式化基准，通过半自动化智能体流水线构建，实现了大规模相互依赖逻辑理论的连贯形式化翻译。 |
| [^287] | [Latent Goal Prediction from Language for Model-Based Planning](https://arxiv.org/abs/2606.20627) | LAGO是一个分层世界模型，通过单一预测器和单一回归目标，将语言指令接地为潜在子目标序列，从而在潜在空间中实现语言引导的基于模型的规划。 |
| [^288] | [Offline Preference-Based Trajectory Evaluation](https://arxiv.org/abs/2606.17541) | 该论文提出基于偏好的轨迹评估方法，通过轨迹间的时序偏好比较取代二元成功指标，将基准测试中的平局率从约75%降至约35%，显著提升了评估的区分能力、排名稳定性和数据效率，并揭示基准测试饱和可能源于度量设计而非仅仅是数据或问题难度问题。 |
| [^289] | [FoundCause: A Foundation Model for Causal Discovery with Latent Confounders from Observational Data](https://arxiv.org/abs/2606.17516) | FoundCause是首个仅在合成结构因果模型上训练的摊销式因果发现基础模型，能够通过单次前向传播从独立同分布观测数据中恢复因果图、识别潜在混杂因素并处理缺失数据。 |
| [^290] | [ICA Lens: Non-Gaussianity Reveals Interpretable Directions in Language Models](https://arxiv.org/abs/2606.11722) | 该论文发现语言模型激活中可解释方向具有非高斯投影分布这一统计特征，并提出可以通过直接搜索非高斯方向来发现可解释特征，而无需依赖稀疏重建等间接学习目标。 |
| [^291] | [Bidirectional Random Projections](https://arxiv.org/abs/2606.10377) | 本文为固定设计下基于双向随机投影的普通最小二乘回归建立了期望超额损失上界，并证明该上界与单向随机投影已有上界之间的差距以约 O(1/n₁) 的速率衰减。 |
| [^292] | [Graph Cascades: Contagion-Based Mesoscopic Rewiring for Structure-Aware Graph Machine Learning](https://arxiv.org/abs/2606.05046) | Graph Cascades提出了一种基于传染扩散的介观图重连策略，在线性时间内将受多跳强化支持的节点对提升为直接邻居，并从理论（标签一致性条件、SBM同配性、有效电阻刻画）和实验上证明其能提升多种GNN与稀疏图Transformer在节点分类任务上的性能。 |
| [^293] | [Stein Kernelized Molecular Dynamics for Active Learning of Interatomic Potentials](https://arxiv.org/abs/2606.04100) | 提出了Stein核化分子动力学（SKMD），一种基于相互作用粒子动力学的增强采样方法，能主动获取多样且有信息量的训练构型用于微调机器学习原子间势，并在渐近极限下保持玻尔兹曼分布为平稳分布。 |
| [^294] | [Semi-Supervised Learning with Noisy Proxy Covariates: Generalization Bounds and Distribution Regression](https://arxiv.org/abs/2606.00512) | 该论文提出一种简单的两阶段半监督回归估计器——从大量含噪声的预训练代理协变量中学习核特征值特征、并在少量标签上拟合岭回归预测器，从理论上证明当代理扰动可控且无标签数据充足时可恢复快速标签样本率，并将分布回归作为特例纳入统一的泛化保证框架。 |
| [^295] | [Multivariate Distributional Reinforcement Learning Using Sliced Divergences](https://arxiv.org/abs/2605.31222) | 该论文提出切片分布强化学习（SDRL），通过投影将易于处理的一维散度推广到多元回报分布，并证明了均匀切片和最大切片变体分别在标量折扣和一般矩阵折扣下的贝尔曼压缩性，同时支持 Wasserstein、Cramér 和 MMD 等多种基础散度。 |
| [^296] | [Batched Stochastic Linear Bandits with 1-Bit Communication Constraints](https://arxiv.org/abs/2605.30976) | 该论文首次研究了批处理与1比特通信约束相结合的随机线性老虎机问题，并证明了即使在无噪声环境下，1比特通信瓶颈也会导致 $\Omega(B\min\{d,\log|\mathcal{A}|\})$ 的不可避免的遗憾下界。 |
| [^297] | [Supervised Training Rapidly Degrades Early Visual Cortex Alignment Across Biologically Plausible Learning Rules](https://arxiv.org/abs/2605.30556) | 该论文发现，监督训练在所有测试的生物合理学习规则下都会降低早期视觉皮层（V1）的表征对齐度，且随机未训练网络反而匹配或超越训练网络，挑战了学习改善大脑对齐的传统假设。 |
| [^298] | [CardioLoom: A Report-Guided Electrocardiogram Foundation Model for Cross-Domain Cardiovascular Phenotyping](https://arxiv.org/abs/2605.25446) | CardioLoom是一种利用近280万心电图-报告配对数据预训练的报告引导式心电图基础模型，在89个内外部心血管表型评估目标上表现优异，仅用10%标注数据即可超越全量监督训练的性能，实现了跨领域的疾病诊断与筛查能力。 |
| [^299] | [Latent Q-Barrier Shielding for Safe In-Context Reinforcement Learning](https://arxiv.org/abs/2605.25267) | 提出一种潜在Q屏障护盾，在部署前学习上下文表示与成本预测模型，在测试时无需参数更新即可依据剩余安全预算和预测的未来成本对候选动作进行过滤或软重加权，从而在分布外偏移下显著改善安全上下文强化学习的奖励-安全权衡。 |
| [^300] | [An Open-Source Training Dataset for Foundation Models for Hyperparameter Optimization](https://arxiv.org/abs/2605.23417) | 该论文提出了BBO-Pile——首个面向超参数优化基础模型的开源训练数据集，包含超过50万条优化轨迹，覆盖3,095个不同的超参数优化任务和全局优化问题，解决了该领域缺乏大规模、真实世界公开预训练数据的瓶颈。 |
| [^301] | [RelPrism: Multi-Faceted Self-Supervised Learning on Relational Databases](https://arxiv.org/abs/2605.23241) | 提出RelPrism框架，通过从节点属性、关系结构等多个方面进行自监督预训练，学习能够全面覆盖关系型数据库上多样化预测任务所需信息的表示，克服了现有SSL方法仅依赖单一监督方面的局限。 |
| [^302] | [Cross-Species RSA Reveals Conserved Early Visual Alignment but Divergent Higher-Area Rankings Across Human fMRI and Macaque Electrophysiology](https://arxiv.org/abs/2605.22401) | 该研究通过跨物种表征相似性分析，发现人类fMRI与猕猴电生理在早期视觉处理上存在保守对齐，但高级脑区的排名在不同物种间存在显著差异。 |
| [^303] | [PEARL: Unbiased Percentile Estimation via Contrastive Learning for Industrial-Scale Livestream Recommendation](https://arxiv.org/abs/2605.21752) | PEARL 提出了一种基于对比学习的非参数分位数近似框架，通过建模相对偏好信号而非绝对参与强度来消除用户行为强度不平衡带来的偏差，从而提升工业级直播推荐的质量与鲁棒性。 |
| [^304] | [Beyond Single Slot: Joint Optimization for Multi-Slot Guaranteed Display Advertising](https://arxiv.org/abs/2605.21556) | 本文提出一种多槽位保量展示广告联合优化框架，将分配建模为带合约轮盘机制和页面浏览约束的离线二部图匹配问题，显著提升商家ROI、平台收入效率与合约履约稳健性。 |
| [^305] | [A closed-loop framework for perovskite precursor additive discovery](https://arxiv.org/abs/2605.20242) | 该论文提出LEAP闭环框架，将领域专业化大语言模型与主动学习相结合，从文献中提取机理知识并转化为可解释分子描述符进行贝叶斯优化，从而在低数据条件下高效发现钙钛矿前驱体添加剂，将太阳能电池效率提升至20.87%。 |
| [^306] | [Q-LocalAdam: Memory-Efficient Client-Side Adaptive Optimization for Edge Federated Learning](https://arxiv.org/abs/2605.17552) | 提出分布感知的量化优化器 Q-LocalAdam，通过对动量状态进行分块线性量化、对重尾方差状态进行对数空间量化并设置反量化后方差下限，在内存受限的边缘联邦学习中显著降低优化器状态内存占用，同时保持接近全精度训练的收敛性能。 |
| [^307] | [Decision-Aware Proximal Bridge Learning for Optimal Treatment Selection](https://arxiv.org/abs/2605.16989) | 该论文提出一种决策感知的近端桥学习方法，将近端因果推断从治疗效应估计拓展到存在隐藏混杂时的最优治疗选择，从而在决策相关区域而非整个动作空间上实现准确的因果响应估计。 |
| [^308] | [When is Warmstarting Effective for Scaling Language Models?](https://arxiv.org/abs/2605.13405) | 该研究实证发现热启动无需保留小模型增长后的初始性能，简单的架构无关增长策略即可超越复杂算子，并确定了增长因子的经验上限——超过该界限时从头训练更为高效。 |
| [^309] | [Kernel-based guarantees for nonlinear parametric models in Bayesian optimization](https://arxiv.org/abs/2605.13160) | 本文提出一个基于核方法的框架，为自适应采集数据上训练的正则化非线性参数模型建立置信界，并据此为近似汤普森采样等贝叶斯优化方法提供收敛保证。 |
| [^310] | [Quantifying Concentration Phenomena of Mean-Field Transformers in the Low-Temperature Regime](https://arxiv.org/abs/2605.10931) | 该论文借鉴多粒子系统收敛分析思想，证明了在低温极限下深度编码器Transformer的token分布会快速集中到由键、查询、值矩阵诱导的投影映射下初始分布的推前分布上，给出了Wasserstein距离随温度参数和推理时间的定量衰减速率，并证明该分布在中长时间内保持亚稳态。 |
| [^311] | [Sharp feature-learning transitions and Bayes-optimal neural scaling laws in extensive-width networks](https://arxiv.org/abs/2605.10395) | 该论文利用留一解耦论证与封闭不动点方程，在高维广延宽度设定下首次给出贝叶斯最优泛化误差的渐近精确刻画，并揭示教师特征随数据量增长依次通过不连续跳变被学习的尖锐相变序列及其背后的神经标度律。 |
| [^312] | [DiffATS: Diffusion in Aligned Tensor Space](https://arxiv.org/abs/2605.09275) | 该论文提出DiffATS，利用Tucker分解并结合正交Procrustes对齐消除张量因子的规范歧义，构建了无需预训练压缩自编码器的紧凑、数据自适应的张量原语，从而实现对高分辨率时空场的高效扩散建模。 |
| [^313] | [iMuon: Intrinsic Spectral Optimization on Riemannian Matrix Manifolds](https://arxiv.org/abs/2605.09238) | 论文提出 iMuon 优化器，通过将黎曼度量规范地把酉不变的欧几里得范数提升为切空间上的内蕴范数，解决了 Muon 推广到低秩、正交、SPD 等矩阵流形时的对称性破坏与闭式解受阻问题。 |
| [^314] | [Quantum-to-Quantum Transfer Learning Exhibits Improved Robustness in Low-Data Regimes](https://arxiv.org/abs/2605.09118) | 该论文系统评估了多种量子与经典模型在迁移学习任务中的表现，发现尽管经典模型峰值性能更高，但量子模型在训练数据受限时性能下降显著更小，展现出更强的鲁棒性。 |
| [^315] | [Measuring and Decomposing Mode Separation via the Canonical Diffusion](https://arxiv.org/abs/2605.08777) | 该论文提出以密度固有的唯一常数扩散系数可逆扩散过程（规范扩散）为基准，从其自协方差矩阵中提取量来测量并分解高维分布的模态分离程度，弥补了微分熵、PCA 等无标签工具无法刻画势垒分隔结构、而互信息又依赖潜在标签的不足。 |
| [^316] | [On Minimal Observation Time for Recovering Latent Hawkes Networks](https://arxiv.org/abs/2605.08400) | 本文针对稀疏弱相互作用的平稳霍克斯过程，证明了精确恢复潜在网络所需的最小观测时间为 $\log d$ 量级（其中 $d$ 为实体数量），该时间界既是充分的也是必要的，并提出了达到该界的两阶段估计方法。 |
| [^317] | [When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR](https://arxiv.org/abs/2604.22774) | 该论文首次系统性研究多行手写数学OCR，揭示了视觉语言模型“过度纠错”从而掩盖学生错误的关键失效模式，并提出了PINK语义评估指标来显式惩罚这种过度纠错行为。 |
| [^318] | [Transferable SCF-Acceleration through Solver-Aligned Initialization Learning](https://arxiv.org/abs/2604.21657) | 该论文提出通过对齐求解器的初始化学习（SAIL）端到端地对SCF求解器进行微分训练，揭示了矩阵预测模型外推失效源于监督问题而非外推问题，从而实现可迁移的SCF收敛加速，并引入有效相对迭代次数（ERIC）作为更准确的评估指标。 |
| [^319] | [Untrained CNNs Match Backpropagation at V1: A Systematic RSA Comparison of Four Learning Rules Against Human fMRI](https://arxiv.org/abs/2604.16875) | 本研究通过系统性RSA比较发现，未训练的CNN在V1区的内部表征与人类fMRI数据对齐程度与反向传播训练的CNN相当，但此结论受评估分辨率影响，且预测编码和STDP的结果因评估缺陷需修正。 |
| [^320] | [Physics Informed Reinforcement Learning with Gibbs Priors for Topology Control in Power Grids](https://arxiv.org/abs/2604.01830) | 该论文提出一种将半马尔可夫控制与编码电网物理特性的吉布斯先验相结合的物理信息强化学习框架，利用图神经网络代理预测动作过载风险以缩减候选动作集并重加权策略，从而有效缓解电网拓扑控制中动作空间的组合爆炸与在线仿真成本高昂的难题。 |
| [^321] | [Gradient Manipulation in Distributed Stochastic Gradient Descent with Strategic Agents: Truthful Incentives with Convergence Guarantees](https://arxiv.org/abs/2603.27962) | 首次提出一种完全分布式的支付机制，在存在策略性智能体可能操纵梯度的分布式随机梯度下降中，同时保证智能体的真实行为与模型训练的精确收敛。 |
| [^322] | [PRUE: A Practical Recipe for Field Boundary Segmentation at Scale](https://arxiv.org/abs/2603.27101) | 本文提出PRUE方法，通过结合U-Net骨干网络、复合损失函数和针对性数据增强，在FTW基准上实现了76% IoU和47% object-F1，为大规模农田边界分割提供了比实例分割模型和地理空间基础模型更实用的解决方案。 |
| [^323] | [Memorization capacity of deep ReLU neural networks characterized by width and depth](https://arxiv.org/abs/2603.09589) | 本文突破了以往仅用参数或神经元数量刻画记忆容量的做法，通过同时调节宽度与深度构造ReLU网络，证明满足 $W^2L^2\log W=\mathcal{O}(N(\log(\delta^{-1})+\log C))$ 即可记忆任意数据集，并给出显式依赖数据间距与类别数的宽度-深度下界，在典型情形下达到 $W^2L^2\asymp N$ 的最优规模。 |
| [^324] | [HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](https://arxiv.org/abs/2602.22630) | 本文提出HyperKKL，利用超网络架构根据外生输入信号动态生成KKL观测器参数，使基于学习的观测器无需重新训练即可泛化到非自治非线性系统。 |
| [^325] | [Differentially Private Non-convex Distributionally Robust Optimization](https://arxiv.org/abs/2602.16155) | 本文对具有ψ-散度和非凸损失的差分隐私分布鲁棒优化（DP-DRO）进行了全面系统的研究，通过将带有不确定性约束的极小极大问题重构为最小化问题并设计新颖算法，弥合了隐私保护与鲁棒优化之间的研究空白。 |
| [^326] | [GPTZero: Robust Detection of LLM-Generated Texts](https://arxiv.org/abs/2602.13042) | GPTZero 5.0通过分层多任务架构和多层自动化红队测试，在11个第三方基准和18个写作领域上实现了最先进的AI文本检测准确率，并能鲁棒地抵御对抗攻击与改写。 |
| [^327] | [Time-Delayed Transformers for Data-Driven Modeling of Low-Dimensional Dynamics](https://arxiv.org/abs/2602.08478) | 该论文提出了一种极简的单层单头时滞Transformer（TD-TF），可视为时滞动态模态分解（TD-DMD）的非线性推广，在保持低计算复杂度的同时，在非线性和混沌低维动力学的长期预测中显著优于线性基线方法。 |
| [^328] | [Beyond Success Rates: Trainability and Extractability for Offline GCRL](https://arxiv.org/abs/2602.05459) | 该论文指出仅用成功率评估离线GCRL算法并不充分，通过构建学习率与策略提取超参数交互的地形图，揭示了不同算法（如QRL、CRL）在超参数敏感性上存在巨大且依赖任务的差异。 |
| [^329] | [Agentic AI-Empowered Dynamic Survey Framework](https://arxiv.org/abs/2602.04071) | 该论文提出一种智能体AI驱动的动态综述框架，将综述视为需持续维护的“活的文档”，能增量式整合新研究并保持综述结构与连贯性。 |
| [^330] | [A Cherry-Picking Approach to Large Load Shaping for More Effective Carbon Reduction](https://arxiv.org/abs/2601.17990) | 本文通过构建近8900节点的ERCOT日前市场出清流程校准复现模型进行全年因果调度反事实分析，发现基于节点边际电价的负荷整形在年度电网碳减排中优于其他静态信号，但没有单一信号能在所有运行条件下均表现最佳，因而提出择优选择信号的负荷整形策略以实现更有效的碳减排。 |
| [^331] | [On the use of case estimate and transactional payment data in neural networks for individual loss reserving](https://arxiv.org/abs/2601.05274) | 该论文比较了前馈神经网络（使用汇总交易数据）与循环神经网络（使用完整赔付历史及个案估计发展历史）在个体损失准备金评估中的表现，发现在协变量数据有限的情况下，个案估计能显著提升预测效果，而赋予神经网络记忆机制的额外收益则相对有限。 |
| [^332] | [CORE: Concept-Oriented Reinforcement for Bridging the Definition-Application Gap in Mathematical Reasoning](https://arxiv.org/abs/2512.18857) | CORE框架将显式概念转化为可控的强化学习监督信号，通过合成概念对齐测验、在推理中注入概念片段等方式，弥合了大语言模型“能复述定义却不会应用概念”的数学推理鸿沟。 |
| [^333] | [Toward AI epidemiology: a measurement standardization framework for prospective risk detection](https://arxiv.org/abs/2512.15783) | 本文提出一个测量标准化框架，将专家-AI交互压缩为结构化、可比较的字段，无需访问模型内部即可对已部署的AI系统进行前瞻性风险检测，并支撑可靠性与治理两方面的主张。 |
| [^334] | [Functional Compression as Occam's Razor: Algebraic Complexity from Sensitivity Minimization](https://arxiv.org/abs/2511.23152) | 该论文为代数补全分解模型建立了以参数敏感度下确界度量的普适复杂度下界，证明达到该下界会迫使潜在表示几何对齐且幺正，并使模型通过共享代数组合（功能压缩）而非低秩表示来学习到全局统一的代数结构。 |
| [^335] | [A physics-constrained neural solver for incompressible flows without labeled data](https://arxiv.org/abs/2511.18820) | 该论文提出了PECANN框架下的不可压缩流物理约束神经求解器，无需标注数据或监督预训练，通过将流动建模为约束优化问题并以严格容差强制满足不可压缩性、动量方程、连续性方程和边界条件，突破了现有物理信息方法雷诺数受限的瓶颈。 |
| [^336] | [SURFing to the Fundamental Limit of Jet Tagging](https://arxiv.org/abs/2511.15779) | 本文提出了SURF（替代参考）验证方法，通过在可解析的替代模型样本上训练目标生成模型来实现精确的Neyman-Pearson检验，据此发现现代喷注标记器可能已接近喷注标记的真实统计极限。 |
| [^337] | [Cross-Epoch Bias in SGD-based Leave-One-Out Influence: Analysis and Accumulative Correction](https://arxiv.org/abs/2510.26185) | 该论文发现多轮次SGD留一影响估计在近似有限删除时忽略了先前删除累积的参数位移会改变后续再次删除同一样本的效果这一跨轮次偏差，并提出利用样本后续出现处Hessian矩阵的累加修正估计器（ACC-SGD-IE），在非凸分析下证明该修正可从有限时域误差界中消除该交互误差项。 |
| [^338] | [Interpret Policies in Deep Reinforcement Learning using SILVER with RL-Guided Labeling: A Model-level Approach to High-dimensional and Multi-action Environments](https://arxiv.org/abs/2510.19244) | 提出了一种带强化学习引导标注的增强SILVER框架，通过将RL策略自身的动作输出纳入边界点识别，将基于Shapley回归的策略解释方法扩展到高维和多动作环境。 |
| [^339] | [Learning Greens Operators through Hierarchical Neural Networks Inspired by the Fast Multipole Method](https://arxiv.org/abs/2509.20591) | 该论文提出受快速多极算法启发的“神经FMM”分层神经网络架构，通过分离局部与远场相互作用，高效学习椭圆偏微分方程的格林算子。 |
| [^340] | [JigSLO: Solving the Resource Puzzle for Efficient Multi-LLM Serving](https://arxiv.org/abs/2509.06261) | JigSLO通过时间共享内存访问模型解析估计带宽竞争对延迟的影响，并据此管理内存资源、调度prefill与decode阶段，实现高效的GPU空间共享多LLM服务。 |
| [^341] | [An open dataset of gas adsorption in deformable metal-organic frameworks for direct air capture](https://arxiv.org/abs/2508.03162) | 该论文发布了ODAC25开放数据集，包含近6000万次DFT计算，涵盖CO2、H2O、N2和O2在约15,000种可变形金属有机框架中的吸附，并引入功能化MOFs、高能吸附质构型和合成框架以提升精度与多样性，为直接空气捕集领域的机器学习模型提供高质量训练数据。 |
| [^342] | [QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems](https://arxiv.org/abs/2507.14760) | 提出QUTCC方法，利用带分位数嵌入的U-Net进行同时分位数回归训练，并在共形校准中引入非线性、空间自适应缩放，从而为成像逆问题生成更紧凑且统计有效的不确定性区间。 |
| [^343] | [Mix-Opt: Mixed Optimization for Memory-Efficient Personalization of Text-to-Image Diffusion Models](https://arxiv.org/abs/2507.10029) | 提出Mix-Opt混合优化框架，根据扩散过程特性自适应地在低分辨率反向传播与高分辨率零阶优化之间切换，在边缘设备有限内存下实现文本到图像扩散模型既高效又保持结构一致的个性化。 |
| [^344] | [First-order methods for stochastic and finite-sum convex optimization with deterministic constraints](https://arxiv.org/abs/2506.20630) | 该论文提出了针对确定性约束下随机与有限和凸优化的随机一阶方法，能够找到约束违反被确定性界定且期望最优性间隙不超过ε的ε-确定可行随机最优（ε-SFSO）解，克服了以往期望可行解可能存在严重约束违反的问题。 |
| [^345] | [Recursive Learning-Based Virtual Buffering for Analytical Global Placement](https://arxiv.org/abs/2506.17247) | 提出了首个构建于 OpenROAD 之上的开源学习驱动虚拟缓冲感知解析式全局布局框架 MLBuf-RePlAce，通过高效的递归学习生成式缓冲方法预测缓冲器类型与位置，并在全局布局中解决 ERC 违例问题。 |
| [^346] | [Evaluating and Mitigating Skin Color Bias in Skin Lesion Classificatio](https://arxiv.org/abs/2506.11627) | 提出将肤色建模为连续属性并根据其与分类性能的非线性关系为样本加权的方法，在保持分类性能的同时，比传统类别重加权实现了更稳定的公平性改进。 |
| [^347] | [Quantum Cognition Machine Learning for Forecasting Chromosomal Instability](https://arxiv.org/abs/2506.03199) | 本文提出量子认知机器学习（QCML）框架，利用量子力学原理将单细胞形态数据表示为希尔伯特空间中的状态向量，无需人工特征选择即可从转移性乳腺癌患者的循环肿瘤细胞形态中预测染色体不稳定性，并在样本外验证中优于传统机器学习方法。 |
| [^348] | [HOPSE: Scalable Higher-Order Positional and Structural Encoder for Combinatorial Representations](https://arxiv.org/abs/2505.15405) | 提出 HOPSE 框架，通过摒弃可学习消息传递层、利用 Hasse 图分解，在任意高阶组合域上实现高效、可线性扩展且富有表达力的位置与结构编码，解决了高阶拓扑深度学习的可扩展性难题。 |
| [^349] | [SiGMA: Learning under Graph-Level Heterogeneity with Gated Message-Passing and Attention](https://arxiv.org/abs/2503.00547) | 提出了SiGMA门控混合图神经网络架构，通过可学习的节点级sigmoid门控自适应融合消息传递与注意力机制，以应对不同图需要不同局部与全局计算组合的图级异质性问题。 |
| [^350] | [Scalable Learning from Probability Measures with Mean Measure Quantization](https://arxiv.org/abs/2502.04907) | 该论文提出一种均值测度量化方法，将所有输入概率测度近似为共享公共支撑集的K点离散测度，在保证量化测度一致性和下游任务收敛性的同时，大幅降低最优传输计算的时间成本并保持与单独量化相当的性能。 |
| [^351] | [Exact Sequence Interpolation with Transformers](https://arxiv.org/abs/2502.02270) | 该论文通过交替使用前馈层与自注意力层并采用低秩参数矩阵，证明了Transformer能够以与输入序列长度无关的复杂度精确插值有限序列数据集。 |
| [^352] | [AI Simulations Grounded in Rich Self-Reports Increase Accuracy and Reduce Distortion](https://arxiv.org/abs/2411.10109) | 该研究通过对1,052名美国成年人开展两小时AI语音深度访谈并结合问卷数据，证明基于丰富自我报告数据（访谈与调查）的AI人类模拟比仅用人口属性提示的模拟更准确，并能显著减少样本层面的八类统计失真。 |
| [^353] | [Clustering in pure-attention hardmax transformers and its role in sentiment analysis](https://arxiv.org/abs/2407.01602) | 本文从动力系统角度严格证明了纯注意力硬最大值Transformer在层数趋于无穷时输入会收敛到由“领导者”点决定的聚类平衡态，并据此构建了完全可解释的Transformer模型，通过将无意义词聚类到语义核心词周围来有效捕捉上下文，从而解决情感分析问题。 |
| [^354] | [High-Dimensional Tail Index Regression: with An Application to Text Analyses of Viral Posts in Social Media](https://arxiv.org/abs/2403.01318) | 提出了高维尾指数回归方法，利用正则化估计和去偏方法进行推断，支持理论的仿真研究，并在社交媒体病毒帖子文本分析中应用。 |
| [^355] | [Nanorobot queue: Cooperative treatment of cancer based on team member communication and image processing](https://arxiv.org/abs/2111.11236) | 本文提出了一种基于纳米机器人队列的理想癌症治疗模型，通过团队成员通信与计算机视觉目标检测技术实现纳米机器人的协同治疗，以期彻底治愈癌症。 |
| [^356] | [Algorithmic Insurance](https://arxiv.org/abs/2106.00839) | 该论文开创了“算法保险”这一新兴领域，通过条件风险价值（CVaR）将二元分类中的决策阈值选择与尾部风险敞口联系起来，证明了准确率最大化等传统方法会显著放大最坏情况损失，并据此提出了将AI系统设计决策与责任风险敞口相挂钩的保险合同框架。 |
| [^357] | [Multiply Robust Causal Mediation Analysis with Continuous Treatments](https://arxiv.org/abs/2105.09254) | 本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。 |
| [^358] | [Optimizer's Information Criterion: Dissecting and Correcting Bias in Data-Driven Optimization.](http://arxiv.org/abs/2306.10081) | 这项研究提出了一个称为优化器的信息准则(OIC)的通用偏差校正方法，帮助解决数据驱动优化中的乐观偏差问题。该方法直接近似一阶偏差，并且不需要解决额外的优化问题，是在决策选择方面的一个创新。 |

# 详细

[^1]: 环境大语言模型：通过温度匹配实现数据高效的监督微调

    Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching

    [https://arxiv.org/abs/2611.23229](https://arxiv.org/abs/2611.23229)

    提出Ambient SFT框架，通过将微调模型的高温条件分布与冻结的预训练“噪声教师”模型进行温度匹配，利用不完美预测扩充微调数据集，从而防止过拟合和数据分布尾部的灾难性遗忘。

    

    为自回归语言模型收集大规模、多样化且高质量的后训练数据集是极具挑战性的。因此，监督微调（SFT）通常应用于远小于预训练规模的数据集，这会导致过拟合以及对数据分布尾部信息的灾难性遗忘。为了防止这种崩溃，我们提出了Ambient SFT，这是一个利用不完美预测来扩充微调数据集的原则性框架。在我们的方法中，预训练模型的“冻结”版本充当一个带噪声的教师，微调后的模型只需在粗粒度上与其匹配。特别是，我们要求后训练模型产生的高温条件分布版本应与预训练模型的高温条件分布版本相匹配。温度超参数控制着对预训练模型的信任程度，其极限为T→∞。

    arXiv:2611.23229v1 Announce Type: cross  Abstract: Collecting large-scale, diverse, and high-quality post-training datasets for autoregressive language models is challenging. Consequently, Supervised Fine-Tuning (SFT) is typically applied to datasets significantly smaller than those used for pre-training, which leads to overfitting and the catastrophic forgetting of the tails of the data distribution. To prevent this collapse, we introduce Ambient SFT, a principled framework for leveraging imperfect predictions to augment the fine-tuning set. In our method, a "frozen" version of the pre-trained model serves as a noisy teacher that the fine-tuned model must match at a coarse level. In particular, we require that the high-temperature versions of the conditional distributions produced by the post-training model should match the high-temperature versions of the pre-training ones. The temperature hyperparameter controls the level of trust in the pre-trained model, with the limit $T\to \inft
    
[^2]: Spool：面向混合视觉语言模型的低延迟流式推理

    Spool: Low-Latency Streaming Inference for Hybrid Vision-Language Models

    [https://arxiv.org/abs/2611.23228](https://arxiv.org/abs/2611.23228)

    Spool是一个无需训练的框架，利用线性循环层状态更新的仿射可结合特性，通过缓存并组合每帧的状态变换来移除过期帧，实现混合视觉语言模型在滑动视频窗口上的低延迟流式推理。

    

    诸如视觉无障碍辅助代理和机器人等应用需要及时回答关于近期视频内容的问题，且限定在过去有限帧的窗口内（例如最近两分钟）。混合视觉语言模型（VLM）将注意力机制与线性循环层（如门控DeltaNet，GDN）交织使用，对这一场景颇具吸引力，但维护滑动窗口的代价高昂。原因在于循环层会将所有过去的帧合并到一个共享状态中，因此移除过期旧帧通常需要将整个窗口重新通过模型运行一遍。我们观察到，循环更新对于状态而言是仿射的且系数依赖于输入，因此每帧的影响可以被捕获为一个可结合组合的变换。我们提出了Spool，一个无需训练的框架，它缓存每帧的状态变换，并在窗口向前滑动时高效地组合这些变换，从而从每个循环层的状态中移除过期帧。

    arXiv:2611.23228v1 Announce Type: cross  Abstract: Applications such as visual accessibility agents and robotics require answering questions about recent video content promptly, over a bounded window of past frames (e.g., the last two minutes). Hybrid vision-language models (VLMs), which interleave attention with linear recurrent layers such as Gated DeltaNet (GDN), are attractive for this setting, but maintaining a sliding window is expensive. The reason is that recurrent layers combine all past frames into a shared state, so expiring old frames normally requires re-running the entire window through the model. We observe that the recurrent update is affine in the state with input-dependent coefficients, so the effect of each frame can be captured as a transformation that composes associatively. We present Spool, a training-free framework that caches per-frame state transformations and composes them efficiently as the window advances, removing expired frames from each recurrent layer's
    
[^3]: 一种用于约束凸有限和优化的非精确随机方差缩减梯度方法

    An inexact stochastic variance-reduced gradient method for constrained convex finite-sum optimization

    [https://arxiv.org/abs/2611.23221](https://arxiv.org/abs/2611.23221)

    该论文针对带硬约束的凸有限和优化问题，提出了一种非精确随机方差缩减梯度方法来计算ε-确定可行随机最优解，从而缩小了确定可行解与期望可行解之间的一阶预言机复杂度差距。

    

    在这篇论文中，我们研究了一类约束凸有限和优化问题。在许多实际应用中，全部或部分约束是表示强制性要求的硬约束。因此，任何理想的近似解都必须（近似地）满足这些约束。受此启发，我们专注于计算一个ε-确定可行随机最优（ε-SFSO）解，其中约束违反被确定性地控制在ε以内，且期望最优性间隙至多为ε。虽然最近提出的一种随机方差缩减一阶惩罚方法可以计算ε-SFSO解，但其计算ε-SFSO解的一阶预言机复杂度与计算ε-期望可行随机最优解（其中期望约束违反和期望最优性间隙均至多为ε）的一阶预言机复杂度之间仍存在显著差距。

    arXiv:2611.23221v1 Announce Type: cross  Abstract: In this paper, we study a class of constrained convex finite-sum optimization problems. In many practical applications, all or some of the constraints are hard constraints that represent imperative requirements. Consequently, any desirable approximate solution must (nearly) satisfy these constraints. Motivated by this, we focus on computing an $\epsilon$-$surely\ feasible\ stochastic\ optimal$ ($\epsilon$-SFSO) solution, in which the constraint violation is deterministically bounded by $\epsilon$, and the expected optimality gap is at most $\epsilon$. While a recently proposed stochastic variance-reduced first-order penalty method can compute an $\epsilon$-SFSO solution, there remains a significant gap between its first-order oracle complexity for computing an $\epsilon$-SFSO solution and that for computing an $\epsilon$-$expected\ feasible\ stochastic\ optimal$ solution, where both the expected constraint violation and the expected op
    
[^4]: 大语言模型智能体还需要被告知如何自我改进吗？

    Do LLM Agents Still Need to Be Told How to Self-Improve?

    [https://arxiv.org/abs/2611.23219](https://arxiv.org/abs/2611.23219)

    该论文提出了一种完全自主的自我改进范式，证明足够强大的冻结大语言模型智能体在仅有基本要素、工具和计算预算、且无人类指定方法或外部强加循环的条件下，能够自行构建自学习算法并实现自我改进，其策略与经典自我调节学习认知框架相符。

    

    大语言模型智能体越来越多地驱动着自身的学习。自我改进方法将模型优化的各种组件委托给模型本身。随着模型能力的不断增强，这引出了一个基本问题：这些日益强大的模型是否仍需要我们来指定它们应如何自我改进？在本文中，我们研究了一种自主范式：在给定基本要素、资源和自我改进总体目标的情况下，智能体自行构建用于基于上下文优化的自学习算法。以游戏对局作为测试平台，一个冻结的大语言模型智能体被给予计算预算以及收集经验和更新文本文档的工具，并被允许在不受外部强加循环限制、也无法访问真实世界数据的条件下自由实验。我们发现，足够强大的模型在这些条件下确实能够实现自我改进，并展现出与经典自我调节学习认知框架相一致的策略。

    arXiv:2611.23219v1 Announce Type: cross  Abstract: LLM agents increasingly drive their own learning. Self-improvement methods delegate various components of model optimization to the models themselves. As models become more capable, this motivates a basic question: do these increasingly capable models still need us to specify how they should self-improve? In this paper we study an autonomous paradigm in which, given basic primitives, resources, and a general goal for self-improvement, agents construct their own self-learning algorithm for context-based optimization. Using the setting of gameplay as a testbed, a frozen LLM agent is given a compute budget and tools for collecting experience and updating text documents, and allowed to freely experiment without an externally imposed loop or access to real-world data. We find that sufficiently capable models do self-improve under these conditions, demonstrating strategies that align with classic cognitive frameworks for self-regulated learn
    
[^5]: 偏差注意力头究竟测量了什么？头级偏差归因中的不可辨识性

    What Does a Bias Head Measure? Non-Identifiability in Head-Level Bias Attribution

    [https://arxiv.org/abs/2611.23218](https://arxiv.org/abs/2611.23218)

    该论文审计了将LLM偏差归因于单个注意力头的常用方法，发现用于选头的统计量主要由提示身份驱动（占64%方差），而公平性标签、响应长度与受保护群体标记之间存在固有共线性，证明头级偏差归因本质上是不可辨识的。

    

    目前将大语言模型中的社会偏差定位到单个注意力头并通过消融（ablation）来消除偏差的方法，通常仅通过其对公平性指标的影响来验证，而从未验证过用于选择注意力头的统计量是否真正测量了偏差。我们在 Qwen-2.5-7B-Instruct 和 Llama-3.1-8B-Instruct 上对这一方法进行了审计。该头级图谱与一个不依赖标签的响应长度图谱相关系数分别达到 ρ = -0.950 和 -0.768，表面上似乎识别出一个混淆因素。然而，一项打破长度与标签耦合关系的干预实验否定了这一解读：在提示保持固定的情况下，响应长度仅能解释该统计量方差的 1%。真正能解释该统计量的是提示本身的身份，占方差的 64%，且其排序与长度并不呈单调关系。在任何固定的提示内，公平性标签、响应长度与受保护群体标记的出现三者之间的共线性高达 φ = 0.95，这部分源于标注规则的定义本身，因此任何观察性分析都无法将该图谱归因于任何单一因素（摘要原文在此处截断）。

    arXiv:2611.23218v1 Announce Type: new  Abstract: Methods that localise social bias in large language models to individual attention heads and remove it by ablation are validated by their effect on a fairness metric, never by evidence that the statistic selecting the heads measures bias. We audit this on Qwen-2.5-7B-Instruct and Llama-3.1-8B-Instruct. The head map correlates at $\rho = -0.950$ and $-0.768$ with a label-blind response-length map, apparently identifying a surface confound. An intervention that breaks the length-label coupling rejects this reading: with the prompt held fixed, response length explains 1% of the statistic's variance. What does explain it is prompt identity, at 64% of variance, in an ordering that is not monotonic in length. Within any fixed prompt the fairness label, response length and the presence of a protected-group token are collinear at $\phi = 0.95$, partly by definition of the labelling rule, so no observational analysis can attribute the map to any 
    
[^6]: 更高得分的编码模型并没有完成更多工作

    Higher-Scoring Coding Models Did Not Get More Work Done

    [https://arxiv.org/abs/2611.23216](https://arxiv.org/abs/2611.23216)

    该研究利用美国出口管制导致Claude Fable 5被停用19天的自然实验，发现在编码基准测试中得分更高的模型（Fable 5）相比得分较低的模型（Opus 4.8）并未给用户带来更高的实际工作产出，揭示了基准分数与真实生产力之间的脱节。

    

    新的编码模型以编码基准测试的分数提升作为宣传卖点。然而，目前尚无研究检验在这些基准测试中得分更高的模型是否真的能完成更多实际工作。2026年6月12日，一项美国出口管制指令禁止访问Claude Fable 5达19天，迫使用户切换到其他模型。在此期间，大多数前Fable 5用户切换到了Opus 4.8（该模型在SWE-Bench Pro上比Fable 5低11分）。我们将这次停用视为一个自然实验：人员和项目保持不变，只有模型发生了变化，因此我们可以分离出Fable 5对用户生产力的影响。我们跟踪了297名Fable 5用户在停用前、停用期间和停用后的GitHub拉取请求（pull request），并以8,908名始终使用Opus 4.8的开发者作为对照组。此外，我们还在2026年停用前后的五次模型升级中重复了这一比较（涉及1,208名开发者和15,442个拉取请求）。

    arXiv:2611.23216v1 Announce Type: cross  Abstract: New coding models advertise themselves on score improvements of coding benchmarks. Yet no study has tested if a model that scores higher on these benchmarks gets more real work done. On June 12, 2026, a U.S. export control directive banned access to Claude Fable 5 for 19 days, which forced users to switch to other models. During this time, most former Fable 5 users switched to Opus 4.8 (11 points lower on SWE-Bench Pro compared to Fable 5). We treat this suspension as a natural experiment, where the people and projects stayed the same and only the model changed, so we could isolate the effect of Fable 5 on user productivity. We tracked GitHub pull requests of 297 Fable 5 users before, during, and after the suspension. We used 8,908 developers who stayed on Opus 4.8 all the time as the control group. We also repeated the comparison across five model upgrades before and after the suspension in 2026 (1,208 developers, 15,442 pull requests
    
[^7]: 用于建模受损驾驶的多模态掩码扩散自编码器

    Multimodal Masked Diffusion Autoencoder for Modeling Impaired Driving

    [https://arxiv.org/abs/2611.23211](https://arxiv.org/abs/2611.23211)

    该论文提出多模态掩码扩散自编码器 ID-MAE，通过自监督方式重建被掩码的注视与轨迹数据来增强受损驾驶预测，并发布了包含 109 人、103 小时驾驶数据及十种受损诱发条件的大规模模拟驾驶数据集 DIVERT，显著提升了受损驾驶推断效果。

    

    人类认知状态（如分心、压力）、注视与行为会以微妙的方式与驾驶环境相互作用，AI 可以利用这些信息来辅助驾驶员，但大规模获取人类认知状态水平的真实标注成本高昂。我们提出了一种多模态掩码扩散自编码器 ID-MAE，它在受损驾驶预测中引入了自监督目标：除了将受损标签作为掩码 token 进行预测外，模型还重建被掩码的注视数据和车辆轨迹。为了探究不同模态和不同类型受损状态对模型推理的影响，我们引入了 DIVERT——一个驾驶模拟器数据集，包含 109 名受试者的 103 小时驾驶数据，涵盖认知、视觉和情绪三类状态下的十种受损诱发方式，并以注视、车辆状态和场景视频的形式进行记录。我们展示了 ID-MAE 在 DIVERT 数据集和现有数据集上均能显著提升受损驾驶推断的效果……

    arXiv:2611.23211v1 Announce Type: cross  Abstract: Human cognitive state (e.g., distraction, stress), gaze, and behavior interact with the driving environment in subtle ways that AI could use to assist drivers, but ground-truth labels for human cognitive state levels are costly to obtain at scale. We propose a multimodal masked-diffusion autoencoder, ID-MAE, that adds a self-supervised objective to impairment prediction: alongside predicting the impairment label as a masked token, it reconstructs masked gaze and trajectory. To probe the effects of different modalities and types of impairments on model inference, we introduce DIVERT, a driving-simulator dataset containing 103 hours of driving data (n = 109) with ten impairment inductions spanning cognitive, visual, and emotional states, recorded with gaze, vehicle state, and scene video. We demonstrate how ID-MAE affords significant improvements in impairment inference results on both the DIVERT dataset and existing datasets, and enable
    
[^8]: 面向黑盒生成模型的无失真语义水印

    Distortion-Free Semantic Watermarks for Black-Box Generative Models

    [https://arxiv.org/abs/2611.23208](https://arxiv.org/abs/2611.23208)

    该论文提出了一种仅需黑盒访问权限的、可证明无失真的语义水印算法，在不改变生成模型输出分布（即不影响输出质量）的同时，对压缩、裁剪等图像内容保持型变换具有鲁棒检测能力。

    

    我们提出了一种可证明无失真的语义水印算法，该算法只需要对生成模型的黑盒采样访问权限以及对语义嵌入的黑盒访问权限。如果水印在其秘密检测密钥上的平均作用能够使生成器的输出分布保持不变，则称该水印是无失真的；这为“水印不会降低输出质量”提供了形式化保证。传统的此类水印通常对变换较为脆弱。这对于图像而言尤其成问题，因为压缩、裁剪或调整大小等良性操作会改变整个像素表示。因此，语义水印要求其检测过程对内容保持型变换具有鲁棒性。以往结合这些特性的方法严重依赖于底层模型的特定白盒属性：要么依赖于自回归文本生成的特性，要么依赖于扩散模型的特定访问能力，例如对初始噪声的控制，或……（摘要未完整提供）

    arXiv:2611.23208v1 Announce Type: new  Abstract: We present a provably distortion-free semantic watermarking algorithm that requires only black-box sampling access to a generative model and black-box access to a semantic embedding. A watermark is called distortion-free if, averaged over its secret detection key, it leaves the generator's output distribution unchanged; this provides a formal guarantee that watermarking does not reduce output quality. Conventional such watermarks are often brittle to transformations. This is especially problematic for images, where benign operations such as compression, cropping, or resizing change the entire pixel representation. Semantic watermarks therefore require detection to be robust to content-preserving transformations. Previous methods combining these properties rely heavily on specific white-box properties of the underlying model: either to autoregressive text generation or to diffusion-specific access, such as control of the initial noise or 
    
[^9]: 在笔记本电脑集成GPU上使用SFT与QLoRA微调Gemma 4边缘模型以生成MATLAB信号处理代码

    Fine-Tuning Gemma 4 Edge Models for MATLAB Signal-Processing Code with SFT and QLoRA on a Laptop iGPU

    [https://arxiv.org/abs/2611.23205](https://arxiv.org/abs/2611.23205)

    该论文在笔记本电脑集成GPU上用66个样例通过SFT结合LoRA/QLoRA微调Gemma 4边缘模型（E2B和E4B）用于MATLAB信号处理代码生成，使通过率显著提升（如E4B从32/66升至47/66），且发现微调主要修正DSP惯例错误而非信号构建错误，运行时代价仅为17-19%的生成速度损失。

    

    能在笔记本电脑上运行的小型语言模型对于离线编写信号处理代码很有吸引力，但它们在数字信号处理（DSP）惯例方面会犯系统性错误。我们使用66个经过检查的样例（涵盖11个信号处理MATLAB子任务）对Gemma 4边缘模型E2B和E4B进行监督微调（SFT），在笔记本电脑的集成GPU（iGPU）上采用低秩适应（LoRA）和量化低秩适应（QLoRA），并通过OpenVINO部署带有适配器的8位整数（int8）模型。在由确定性检查器评分的保留测试变体上，E2B的通过率从18/66提升至37/66（LoRA）和37/66（QLoRA），E4B的通过率从32/66提升至47/66（QLoRA）。通过阅读全部159个失败程序发现，适配器消除了大多数惯例性错误（如滞后的符号、单位和数组形状），但很少修复构建信号和噪声方面的错误。在运行时应用适配器会使生成速度降低17-19%。

    arXiv:2611.23205v1 Announce Type: cross  Abstract: Small language models that run on a laptop are attractive for writing signal-processing code offline, but they make systematic errors on digital signal processing (DSP) conventions. We apply supervised fine-tuning (SFT) to the Gemma 4 edge models E2B and E4B with 66 checked examples covering 11 signal-processing MATLAB subtasks, using low-rank adaptation (LoRA) and quantized LoRA (QLoRA) on the integrated GPU (iGPU) of a laptop, and deploy the 8-bit integer (int8) models with the adapters through OpenVINO. On held-out variants scored by deterministic checkers, the pass rate rises from 18/66 to 37/66 (LoRA) and 37/66 (QLoRA) for E2B and from 32/66 to 47/66 for E4B (QLoRA). Reading all 159 failing programs shows that the adapters remove most errors of convention, such as the sign of a lag, units and array shapes, but few errors in building signals and noise. Applied at run time, the adapter costs 17-19% of the generation speed.
    
[^10]: 降低观测频率下学习型与运动学轨迹预测的对比研究

    Learned versus Kinematic Trajectory Prediction under Reduced Observation Frequency

    [https://arxiv.org/abs/2611.23198](https://arxiv.org/abs/2611.23198)

    该研究在Argoverse 2数据集上通过将观测历史从10 Hz降采样至1-5 Hz并进行严格的统计检验，系统评估了学习型轨迹预测器相对于恒定速度/加速度运动学模型的优势如何随观测频率降低而变化。

    

    自动驾驶的轨迹预测模型通常使用固定采样频率的数据进行训练，而通过车联网（V2X）通信接收到的位置信息到达频率仅为1-10 Hz。我们研究当历史观测频率降低时，仅使用观测历史的学习型预测器是否仍能保持相对于恒定速度（CV）和恒定加速度（CA）模型的优势。在Argoverse 2运动预测数据集上，将目标智能体的5秒历史数据从10 Hz降采样至5、2和1 Hz，并在不使用地图或相邻智能体信息的情况下预测6秒的未来轨迹。针对每个频率分别训练基于LSTM的循环神经网络（RNN），并与在10 Hz下训练的同一网络（经线性插值后应用于稀疏历史数据）进行比较。假设和阈值在训练前预先固定，主要RNN各使用十个随机种子进行训练，在24,984个测试场景上的结果均带有自助法置信区间。

    arXiv:2611.23198v1 Announce Type: new  Abstract: Trajectory prediction models for automated driving are usually trained on data with a fixed sampling frequency, while positions received via vehicle-to-everything (V2X) communication arrive at 1-10 Hz. We study whether a learned predictor that uses only the observation history keeps its advantage over constant velocity (CV) and constant acceleration (CA) models when the history frequency drops. On the Argoverse 2 motion forecasting dataset, the 5 s history of the focal agent was downsampled from 10 Hz to 5, 2, and 1 Hz, and the 6 s future was predicted without maps or neighboring agents. An LSTM-based recurrent neural network (RNN) trained for each frequency was compared with the same network trained at 10 Hz and applied to sparse histories after linear interpolation. Hypotheses and thresholds were fixed before training, the main RNNs were trained with ten seeds each, and results on 24,984 test scenarios have bootstrap confidence interva
    
[^11]: WoRD：基于扩散模型的在线目标识别后验推断

    WoRD: Diffusion-Based Posterior Inference for Online Goal Recognition

    [https://arxiv.org/abs/2611.23192](https://arxiv.org/abs/2611.23192)

    提出了WoRD方法，利用学习到的扩散轨迹先验，为每个候选目标运行热启动去噪链并通过贝叶斯重加权得到校准后验，实现从部分观测中对智能体目标的在线识别。

    

    目标识别是从智能体的观测行为中推断其意图目标的任务。现有的大多数工作集中在高层符号识别上，将智能体达成目标所经历的连续运动抽象掉；而对连续轨迹进行推理的方法则依赖于显式的动力学模型，该模型难以获取且很少能跨场景迁移。这两种方法都没有显式地学习智能体如何向每个候选目标移动，从而限制了在最需要此类知识的模糊轨迹前缀上的准确性。我们提出了WoRD（Waypoint Recognition with Diffusion，基于扩散的航点识别），它利用学习到的轨迹先验，从部分观测中推断智能体的目标。在每一步中，WoRD为每个候选目标运行一条热启动的去噪链，通过闭式对数似然梯度将每条链引导向观测轨迹，并根据贝叶斯法则对生成的样本进行重新加权，从而得到校准的后验分布。

    arXiv:2611.23192v1 Announce Type: new  Abstract: Goal recognition is the task of inferring an agent's intended goal from its observed behavior. Most existing work focuses on high-level symbolic recognition, abstracting away the continuous motion through which agents reach their goal. Methods that do reason over continuous trajectories rely on an explicit dynamics model that is hard to acquire and rarely transfers across settings. Neither approach explicitly learns how agents move toward each candidate goal, limiting accuracy on ambiguous trajectory prefixes where such knowledge is most needed.   We present WoRD (Waypoint Recognition with Diffusion), which uses a learned trajectory prior to infer the agent's goal from partial observations. At each step, WoRD runs one warm-started denoising chain per candidate goal, steers each chain toward the observed trajectory through a closed-form log-likelihood gradient, and reweights the resulting samples by Bayes' rule into a calibrated posterior
    
[^12]: 似曾相见：面向视觉任务的共享工作记忆模型

    D\'ej\`a View: A Shared Working-Memory Model for Visual Tasks

    [https://arxiv.org/abs/2611.23177](https://arxiv.org/abs/2611.23177)

    该研究提出了一个将动态活动与图像特征间临时关联相结合的共享工作记忆模型，单个训练后的模型即可在提示引导下灵活完成图像记忆、干扰分类和新旧识别等多种视觉任务。

    

    工作记忆模型必须能够记住刺激，并灵活运用其记忆来执行不同的任务。本文提出了一个针对图像的工作记忆模型，该模型将动态活动与图像特征之间的临时关联相结合。我们在五个任务上训练该模型，包括在其他图像呈现期间记住样本图像、在记住样本的同时对干扰图像进行分类，以及判断一幅图像是新出现的还是之前呈现过的。同一个训练好的模型可以执行所有任务，并通过明确的提示信号来指导所需做出的响应。我们利用这一灵活的记忆模型研究了延迟、干扰和图像历史对性能的影响。在我们测试的延迟范围内，延迟对样本记忆性能几乎没有影响；而即使目标图像在所有条件下出现的近期程度相同，看到更多的图像仍会导致更多的虚报。在样本记忆加分类任务中，正确拒……

    arXiv:2611.23177v1 Announce Type: cross  Abstract: A working-memory model must remember stimuli and use its memory flexibly to perform different tasks. Here we introduce a model of working memory for images that combines dynamic activity with temporary associations between image features. We train the model on five tasks, including remembering a sample while other images are presented, categorizing distractor images while remembering the sample, and determining whether an image is new or has been presented before. The same trained model performs all tasks, guided by explicit cues about which response is required. We use this flexible memory model to ask how delay, distraction, and image history affect performance. Delays have little effect on sample-memory performance for the delays we tested, while seeing more images leads to more false alarms even when the target image appeared equally recently in all cases. On the sample-memory-plus-categorization task, the probability of correct re
    
[^13]: 面向多语言适配的MoE-LoRA方法：一项系统性实证研究

    MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study

    [https://arxiv.org/abs/2611.23158](https://arxiv.org/abs/2611.23158)

    本研究首次在统一的多语言评估协议下，用200种语言和9000多个评估设置系统比较八种LoRA方法，发现小规模搜索无法可靠预测方法在大规模下的相对表现，且LoRA虽能提升翻译数据拟合度但会持续损害阅读理解能力。

    

    路由式低秩适配器为提升大语言模型（LLM）在各类任务上的性能提供了一种参数高效的手段。因此，人们假定它们也可用于改善LLM对低资源语言的支持——目前这些模型对英语的服务远好于全球数千种其他语言。然而，这一假设从未在统一的多语言评估协议下得到实证验证。我们正是针对这一空白开展研究：在冻结的基础模型上，使用200种独特语言、超过9,000个评估设置，对八种低秩适配方法进行了比较。我们的结果表明，缩减规模的搜索能够识别出LoRA方法的可行配置，但无法可靠地预测其相对性能，因为排名最高的配置在大规模评估中从未表现最佳。LoRA能可靠地提升模型在翻译数据上的语言建模拟合度，却持续降低阅读理解能力，并在（原文摘要在此处截断）……

    arXiv:2611.23158v1 Announce Type: new  Abstract: Routed low-rank adapters offer a parameter-efficient means to improve the performance of large language models (LLMs) across tasks. Hence, it is assumed that they can also be used to improve the support of low-resource languages in LLMs, which currently serve English far better than the thousands of other languages spoken worldwide. However, this assumption has never been verified empirically under a shared multilingual evaluation protocol. We address exactly this gap by comparing eight low-rank adaptation (LoRA) methods on a frozen base model using 200 unique languages in more than 9{,}000 evaluation settings. Our results suggest that a reduced-scale search can identify viable configurations of LoRA methods but does not reliably predict their relative performance, as the top-ranking configuration is never best at scale. LoRA reliably improves language-model fit on translation data, consistently degrades reading comprehension, and benefi
    
[^14]: 面向推理时目标的分层策略学习

    Learning Stratified Policies for Inference-Time Objectives

    [https://arxiv.org/abs/2611.23154](https://arxiv.org/abs/2611.23154)

    该论文提出“分层策略”，通过在共享语言模型主干上训练 k 个 LoRA 适配器，将推理时的采样预算分配给一个策略群体，并从理论和实验上证明这种策略群体在优化 pass@k 时优于单一策略。

    

    推理时扩展通过采样多个回复并评估其整体效用来提升大语言模型的性能。然而，许多现有方法都是从单一策略重复采样来获取这些样本。在本工作中，我们研究是否可以将推理时的采样预算分配到一组策略群体之中。我们提出了分层策略，其中 k 个学习到的策略各自为一个 k 样本的推理时目标贡献一个回复。我们通过在共享的语言模型主干上使用独立的 LoRA 适配器来实现这一策略群体，并针对 pass@k 进行优化。我们从理论上通过分析线性设定为该方法提供了依据：在每一个单独的 LoRA 都具有严格正预测误差的设定下，LoRA 群体可以实现零的 best-of-k 逼近误差。在实验上，我们证明了分层策略在 pass@k 指标上始终优于单一策略基线。

    arXiv:2611.23154v1 Announce Type: cross  Abstract: Inference-time scaling improves the performance of large language models by sampling multiple responses and evaluating their collective utility. However, many existing approaches obtain these samples by repeatedly sampling from a single policy. In this work, we study if the sample budget at inference time can instead be allocated across a population of policies. We introduce stratified policies, where each of $k$ learned policies contributes one response to a $k$-sample inference-time objective. We realize this policy population using separate LoRA adapters on a shared language model backbone and optimize it for pass@$k$. We theoretically motivate the approach by analyzing the linear setting; a population of LoRAs can achieve zero best-of-$k$ approximation error in settings where every single LoRA has strictly positive prediction error. Empirically, we demonstrate that stratified policies consistently improve pass@$k$ over a single-pol
    
[^15]: 基于障碍引导扩散采样的异构团队能力组合

    Capability Composition in Heterogeneous Teams via Barrier-Guided Diffusion Sampling

    [https://arxiv.org/abs/2611.23149](https://arxiv.org/abs/2611.23149)

    本文提出一种障碍引导的扩散采样方法，将能力组合的物理条件编码为轨迹约束，使异构团队无需任何合作演示数据即可组合个体能力生成合作轨迹，并证明该方法能生成所有满足约束的团队轨迹。

    

    异构团队中的智能体可以通过合作来完成超越其个体能力的任务。对于合作任务，团队轨迹可以由生成模型产生，前提是该模型已在智能体合作的演示数据上完成训练。然而，合作演示数据的收集成本高昂，且对于团队从未执行过的任务，这样的演示数据可能根本不存在。但值得注意的是，智能体之间能够组合各自能力的条件是团队的物理属性，在任何演示数据被收集之前就是已知的。我们将这些条件编码为对团队轨迹的约束，并在从仅基于单个智能体演示训练的生成模型中采样时强制执行这些约束。我们证明了所提出的方法能够生成所有满足约束条件的团队轨迹，尽管演示数据中不包含任何合作行为。因此，这些约束决定了哪些团队轨迹可以被生成。

    arXiv:2611.23149v1 Announce Type: cross  Abstract: The agents in a heterogeneous team can cooperate to accomplish tasks that exceed their individual capabilities. The team trajectories for a cooperative task can be generated by a generative model, provided that the model has been trained on demonstrations of the agents cooperating. Yet, demonstrations of cooperation are costly and may not exist for tasks that the team has never performed. The conditions under which the agents can combine their capabilities, however, are physical properties of the team and are known before any demonstration is collected. We encode the conditions as constraints on the team trajectories, and we enforce the constraints while sampling from generative models trained on demonstrations of each agent alone. We prove that the proposed method can generate every team trajectory that satisfies the constraints, although the demonstrations contain no cooperation. Hence the constraints determine which team trajectorie
    
[^16]: RSI-Prime：训练模型去训练模型

    RSI-Prime: Training Models to Train Models

    [https://arxiv.org/abs/2611.23138](https://arxiv.org/abs/2611.23138)

    该论文提出RSI-Prime数据集（含覆盖七大任务类别的350条完整实验轨迹），通过细粒度决策质量标注与决策级损失掩码过滤低质量行为进行训练，使GLM-5.3在自主后训练基准PostTrainBench上整体性能相对提升31.95%，为递归自我改进奠定基础。

    

    使大语言模型（LLM）能够自主开展训练并改进目标模型，是实现递归自我改进（RSI）的重要基础。随着LLM的角色从“被训练者”扩展为“训练者”，其模型训练能力值得专门研究和系统性发展。为此，我们构建了RSI-Prime——一个用于提升模型训练能力的数据集，包含覆盖七个任务类别的350条完整实验轨迹。为帮助LLM从训练实践中学会做出高质量的实验决策，我们进行了细粒度的决策识别与质量标注，并采用决策级损失掩码来排除对低质量行为的监督。在自主后训练基准PostTrainBench上使用GLM-5.3进行的实验表明，基于RSI-Prime的训练使整体性能相比原始GLM-5.3取得了31.95%的相对提升，突显了一条有前景的路径（原文在此处截断）。

    arXiv:2611.23138v1 Announce Type: new  Abstract: Enabling large language models (LLMs) to autonomously conduct training and improve target models is an important foundation for recursive self-improvement (RSI). As LLMs expand their role from being trained to conducting training, their model-training capability warrants dedicated study and systematic development. To this end, we construct RSI-Prime, a dataset for improving model-training capability, comprising 350 complete experimental trajectories across seven task categories. To help LLMs learn to make high-quality experimental decisions from training practice, we perform fine-grained decision identification and quality annotation, then apply decision-level loss masking to exclude supervision on low-quality behavior. Experiments with GLM-5.3 on PostTrainBench, a benchmark for autonomous post-training, show that training on RSI-Prime yields a 31.95% relative improvement in overall performance over the original GLM-5.3, highlighting a p
    
[^17]: 基于边际匹配的流校正

    Correcting Flows with Marginal Matching

    [https://arxiv.org/abs/2611.23134](https://arxiv.org/abs/2611.23134)

    本文提出了一种新颖的基于训练的预测-校正框架，通过“边际匹配”思想在前向ODE的特定时间步将样本校正至与反向ODE相匹配的边际分布，从而有效缩小学习分布与数据分布之间的距离，提升流匹配模型的生成质量。

    

    流匹配模型是一种基于ODE的生成模型，它通过将简单的源分布逐渐变形为目标分布来生成样本。然而，这些模型在最优地平衡计算效率与生成保真度方面仍面临挑战。虽然预测-校正方案已被证明能够有效解决扩散模型中的这一问题，但其在流匹配模型中的应用仍未得到充分探索。在本工作中，我们提出了一种新颖的基于训练的流匹配预测-校正框架，该框架利用了ODE可以轻松反向求解的特性。我们的核心思想被称为“边际匹配”，即在前向ODE的特定时间步处对样本进行校正，使其趋向于与反向ODE相匹配的边际分布。该方案被证明能够缩小数据分布与学习到的分布之间距离的上界，并由此激发了校正函数的两种不同实现方式。

    arXiv:2611.23134v1 Announce Type: new  Abstract: Flow matching models, ODE-based generative models, generate samples by gradually morphing a simple source distribution into a target distribution. However, these models still face challenges in optimally balancing computational efficiency with generation fidelity. While predictor-corrector schemes have been proven to help address this issue in diffusion models effectively, their application to flow matching models remains under-explored. In this work we present a novel training-based predictor-corrector framework for flow matching, which exploits the fact that the ODE can be easily reversed. Our key idea, termed Marginal Matching, is to correct samples at a particular time step in the forward ODE towards a matching marginal distribution of the backward ODE. This scheme proves to reduce a bound on the distance between the data distribution and the learned distribution, motivating two different implementations for the correction function. 
    
[^18]: 一种用于快速基于梯度的最优起飞轨迹设计的神经包络方法

    A Neural Envelope Method for Rapid Gradient-based Optimum Takeoff Trajectory Design

    [https://arxiv.org/abs/2611.23130](https://arxiv.org/abs/2611.23130)

    本文提出一种将物理约束生成对抗网络（physicsGAN）与神经包络相结合的方法，将带约束的非凸最优起飞轨迹设计问题转化为无约束的凸优化问题，从而实现快速的基于梯度的最优轨迹设计。

    

    城市空中交通（Urban Air Mobility）的兴起旨在缓解地面交通拥堵。多学科分析与优化（MDAO）被引入以降低能耗，从而提升飞行器性能。基于代理模型的MDAO降低了优化的成本，但多峰性（multimodality）会干扰基于仿真和基于代理模型优化中的全局最优搜索。凸包络（convex envelopes）可将多峰空间转换为单峰空间，从而实现高效的凸优化；然而，对设计空间进行凸化处理时，由于复杂的设计约束，可能产生不可行的“最优”设计。为应对这些挑战，本工作提出将物理约束生成对抗网络（physicsGAN）与神经包络相耦合，将原始的、带约束的非凸优化转化为无约束的凸优化。具体而言，physicsGAN发现一个可行设计空间，其中所有设计候选方案本身就满足约束条件（摘要原文在此处截断）。

    arXiv:2611.23130v1 Announce Type: new  Abstract: Urban air mobility arises to mitigate ground traffic congestion. Multidisciplinary analysis and optimization (MDAO) has been introduced to reduce the energy consumption, thereby expanding aircraft performance. Surrogate-based MDAO lowers the cost of optimization, but multimodality disrupts global optimum search for both simulation-based and surrogate-based optimizations. Convex envelopes convert multimodal spaces into unimodal spaces, permitting efficient convex optimization; however, convexifying a design space may produce infeasible "optimal" designs due to complex design constraints. To address these challenges, this work proposes the coupling of a physics-constrained generative adversarial networks (physicsGAN) and neural envelope to transform original, constrained, non-convex optimizations into unconstrained, convex optimizations. Specifically, the physicsGAN discovers a feasible design space where all design candidates inherently s
    
[^19]: 物理应从何处介入？概率性海表温度预测中的信息、可辨识性与校准

    Where Should Physics Enter? Information, Identifiability and Calibration in Probabilistic Sea Surface Temperature Forecasting

    [https://arxiv.org/abs/2611.23129](https://arxiv.org/abs/2611.23129)

    该论文系统性地测试了平流-扩散物理先验在神经海表温度预测模型中的不同介入位置，发现学习得到的速度场既无法带来可靠的预测收益，也无法辨识出真实洋流——即使洋流信息理论上可大幅提升预测精度——揭示了物理信息神经网络中可辨识性与校准的关键问题。

    

    平流-扩散先验是使神经网络海表温度（SST）预测具备物理信息的一种常用手段。我们测试了此类先验应在异方差ConvLSTM的哪个环节介入才能真正起到帮助作用，实验使用孟加拉湾（1982-2024年）的每日NOAA OISST v2.1数据，以及在同一网格上构建的、具有已知可控强度洋流的合成基准数据集。(i) 在真实数据上，采用学习速度场的软平流-扩散残差并未带来可靠的收益（提前15天CRPS为-0.007°C，95%置信区间为[-0.022, +0.008]）；一个简单的时序平滑惩罚便可达到同等效果，且二者在3%范围内的等效性也无法确立。(ii) 学习得到的速度场无法恢复真实洋流，无论将其置于损失函数中、显式输运步骤中，还是不含神经校正的输运步骤中：即便在合成数据上——已知洋流可使第15天预测提升16-23%——它们与参考洋流的相关性也不优于打乱日期的零假设。(iii) 提供的洋流（原文此处被截断）

    arXiv:2611.23129v1 Announce Type: new  Abstract: Advection-diffusion priors are a common way to make neural sea surface temperature (SST) forecasts physics-informed. We test where such a prior must enter a heteroscedastic ConvLSTM to help, using daily NOAA OISST v2.1 for the Bay of Bengal (1982-2024) and a synthetic benchmark on the same grid with known currents of controlled strength. (i) On real data, a soft advection-diffusion residual with a learned velocity gives no reliable benefit (lead-15 CRPS -0.007 degrees C, 95% interval [-0.022, +0.008]); a plain temporal-smoothness penalty matches it, and equivalence within 3% is not established either. (ii) Learned velocities do not recover the currents, whether placed in the loss, in an explicit transport step, or in a transport step without neural correction: they correlate with the reference no better than a shuffled-date null, even on synthetic data where knowing the currents would improve day-15 forecasts by 16-23%. (iii) Supplied cu
    
[^20]: 接地的幻觉：视觉-语言模型是否像理想观察者那样权衡检索到的文本与图像？

    The Illusion of Grounding: Do Vision-Language Models Weigh Retrieved Text and Images Like an Ideal Observer?

    [https://arxiv.org/abs/2611.23127](https://arxiv.org/abs/2611.23127)

    该论文构建了包含20万个条目的GIB-200K压力测试基准，并从理想贝叶斯线索组合理论出发，首次形式化定义并量化了视觉-语言模型在融合检索文本与图像证据时的“知识惯性”与“感官捕获”偏差。

    

    检索增强的视觉-语言模型（VLMs）从两种可能各自失效的证据来源中生成答案：检索到的文本可能是错误的或含噪声的，图像也可能退化或与文本相矛盾。我们提出了GIB-200K——一个基于COCO和Visual Genome构建、包含200,000个图像-问题条目的基准数据集，每个条目都被扩展为受控的压力测试条件：硬负例检索噪声（1-20个文档）、视觉歧义（高斯模糊与运动模糊、低资源裁剪）以及带有反事实翻转对的跨模态冲突。我们不仅测量性能退化，还探究模型如何组合这两种模态。基于理想贝叶斯线索组合理论，我们推导出可检验的预测，从而给出“知识惯性”（相对于模型自身的图像证据，文本被过度加权）与“感官捕获”（相反情形）的形式化定义，并通过经测量误差校正的回归来估计权重，同时验证了该方法对已知权重的恢复能力。反事实翻转分解……（原文摘要在此处截断）

    arXiv:2611.23127v1 Announce Type: cross  Abstract: Retrieval-augmented vision-language models (VLMs) answer from two sources of evidence that can each fail: the retrieved text can be wrong or noisy, and the image can be degraded or contradicted. We introduce GIB-200K, 200,000 image-question items built from COCO and Visual Genome, each expanded into controlled stress conditions: hard-negative retrieval noise (1-20 documents), visual ambiguity (Gaussian and motion blur, low-resource crops) and cross-modal conflict with counterfactual flip pairs. Instead of only measuring degradation, we ask how models combine the two modalities. From ideal Bayesian cue combination we derive a testable prediction that yields formal definitions of Knowledge Inertia (text over-weighted relative to the model's own image evidence) and Sensory Capture (the reverse), and we estimate the weights with a measurement-error-corrected regression whose recovery of known weights we verify. A counterfactual flip decomp
    
[^21]: 哪段历史最重要？股票Transformer模型中的训练历史与多预测期限

    Which Past Matters? Training History and Multi-Forecast Horizon in Equity Transformers

    [https://arxiv.org/abs/2611.23126](https://arxiv.org/abs/2611.23126)

    本文通过十二种配置的月度滚动实验发现，在日内股票收益预测中，最优的历史训练数据选择取决于预测期限——短期预测（5-15分钟）适合近期数据采样，长期预测（60-120分钟）适合均匀采样，且单纯增加训练数据量并不能持续提升预测精度。

    

    更多的历史数据可以改善估计效果，但同时也会引入难以迁移到当前市场状况的关系。因此，我们研究了训练数据量与历史数据选择如何与日内股票预测中的预测期限相互作用。我们部署了十二种配置，将四种采样与验证策略同6个月、12个月和24个月的训练数据量相结合，并采用固定的横截面Transformer架构与特征集。月度滚动前推实验共产生了264次重新拟合，以及从5分钟到120分钟的八个累计收益预测。评估区间覆盖2025年1月至2026年10月20日，并使用横截面Spearman信息系数，以未来五因子残差收益作为评估基准。结果显示，6个月近期数据采样在5-15分钟的单模型预测期限中表现领先，而多近期数据配置在20-30分钟期限领先，均匀采样配置则在60-120分钟期限领先。增加训练数据量并不能一致地提升预测准确率。

    arXiv:2611.23126v1 Announce Type: cross  Abstract: More historical data can improve estimation while introducing relationships that transfer poorly to current market conditions. We therefore examine how training quantity and historical selection interact with forecast horizon in intraday equity prediction. We deploy twelve configurations which combine four sampling and validation policies with 6, 12 and 24 month training quantities under a fixed cross-sectional Transformer and feature set. A monthly walk-forward experiment produces 264 refits and eight cumulative return forecasts from 5 to 120 minutes. Evaluation covers January 2025 through 20 October 2026 and uses cross-sectional Spearman information coefficients against future five-factor residual returns. Six-month recency sampling leads individual models at 5-15 minutes, while multi recency configurations lead at 20-30 minutes and uniform configurations at 60-120 minutes. Increasing training quantity does not uniformly improve accu
    
[^22]: 通过激活引导正确实现提示增强GRPO

    Get Hint-Augmented GRPO Done Right Through Activation Steering

    [https://arxiv.org/abs/2611.23124](https://arxiv.org/abs/2611.23124)

    该论文提出ASH方法，将HA-GRPO中提示的作用重新诠释为引导模型行为而非降低问题难度，通过激活引导技术解决训练与推理时上下文不匹配的问题，从而实现一致的推理能力提升。

    

    arXiv:2611.23124v1 公告类型：新 摘要：GRPO已成为利用可验证奖励激发大语言模型推理能力的主流强化学习方法。然而，它存在一个根本性的失败模式：当训练问题超出基础模型当前能力时，所有生成的回答都会失败并获得零奖励，从而不产生梯度信号，阻碍了能力的进一步提升。一类新兴的GRPO扩展方法——提示增强GRPO（HA-GRPO），通过在上下文中添加“提示”来降低问题难度，从而产生可用的梯度信号。然而，现有的HA-GRPO方法存在一个关键问题：在推理时没有提示可以附加到问题上，造成训练与推理之间的上下文不匹配，导致性能提升不一致。在本工作中，我们重新诠释了提示的作用——不是降低问题难度，而是引导模型行为——并提出了ASH（Activation Steering for...（原文截断）

    arXiv:2611.23124v1 Announce Type: new  Abstract: GRPO has become the dominant reinforcement learning method for eliciting reasoning capability in LLMs using verifiable rewards. However, it suffers from a fundamental failure mode: when a training question exceeds the base model's current capability, all generated responses fail and receive zero reward, yielding no gradient signal and stalling further capability expansion. An emerging class of GRPO extensions, termed Hint-Augmented GRPO (HA-GRPO), adds a "hint" to the context to reduce the difficulty of the question, thereby producing a usable gradient signal. Nevertheless, existing HA-GRPO methods share a key issue: no hint is available at inference time to append to the question, creating a training-inference context mismatch that leads to inconsistent performance gains. In this work, we reinterpret the function of hinting -- not as reducing question difficulty, but as steering model behavior -- and propose ASH (Activation Steering for
    
[^23]: 审计面向四足机器人的学习型模拟器：ANYmal D 上的机器人世界模型

    Auditing a Learned Simulator for Legged Robots: The Robotic World Model on ANYmal D

    [https://arxiv.org/abs/2611.23118](https://arxiv.org/abs/2611.23118)

    该论文对四足机器人的学习型模拟器进行了端到端审计，发现仅用单步误差评估会误判模拟器质量——尽管单步训练的模拟器单步预测更准，但在自身长时推演数据上训练的模拟器能让策略在 368 步的推演中表现更好。

    

    四足机器人可以在一个学习到的模拟器中学会行走：该模拟器是一个神经网络，用于预测机器人的下一状态，并基于自身的预测不断向前推演。这类模拟器通常只进行单步训练和单步检验，而策略在使用它们时却需要连续推演一百步甚至更多。我们在没有 GPU 的条件下对一个已发表的系统进行了端到端审计：包括面向 ANYmal D 四足机器人的机器人世界模型（RWM）的动力学模型，及其后续工作 RWM-U 的不确定性估计。我们重建了该模型，使其与发布代码的输出、损失和梯度完全一致，测量了 101 个训练好的模拟器、200 个压缩副本以及另外三种架构下的 38 个模拟器，并在模拟器中训练策略，随后在我们手工构建的机器人 MuJoCo 模型中进行测试。结果显示，在模型自身推演数据上的训练在 368 步时于全部十个保留测试回合中都优于单步训练（结果已注册），尽管在单步预测上单步训练表现更好。因此，单步检验会误判……（原文截断于此）

    arXiv:2611.23118v1 Announce Type: cross  Abstract: A legged robot can learn to walk inside a learned simulator: a neural network that predicts the robot's next state and is run forward on its own predictions. They are usually trained and checked one step ahead, although a policy uses them for a hundred steps or more. We audit one published system end to end, without a GPU: the dynamics model of the Robotic World Model (RWM) for the ANYmal D quadruped, and the uncertainty estimate of its follow-up, RWM-U. We rebuilt the model to match the released code's outputs, losses and gradients exactly, measured 101 trained simulators, 200 compressed copies and 38 simulators of three other architectures, and trained policies inside simulators that we tested in a hand-built MuJoCo model of the robot. Training on the model's own rollouts beat one-step training at 368 steps on all ten held-out episodes (registered), although one-step training was better one step ahead. One-step checks therefore misju
    
[^24]: 大规模神经元社区发现揭示语言模型中的神经元稀疏行为

    Large-scale Neuron Community Discovery Reveals Neuron-Sparse Behaviors in Language Models

    [https://arxiv.org/abs/2611.23112](https://arxiv.org/abs/2611.23112)

    该论文提出了一种无监督的“神经元社区发现”方法，通过梯度归因、图构建和社区检测，在大语言模型中找出跨输入复用且因果必要充分的神经元群组，从而证明语言模型中存在可复用的稀疏神经元计算单元。

    

    在语言模型的神经元中，我们能发现哪些可复用的计算单元？先前的研究表明，单次前向传播通常可以在事后被稀疏化为小型神经元群组，但这些群组是否构成跨输入复用的连贯计算实体仍不清楚。我们提出了神经元社区发现，这是一种无监督方法，用于寻找跨提示反复出现、且在其所跨越的层范围内对常见计算既因果必要又因果充分的神经元群组。该方法结合了基于梯度的归因分析、连接反复被共同归因的神经元的图结构，以及社区检测。将该方法应用于 Llama-3.1-8B 并使用一千万条提示，它发现了数千个候选社区。我们通过三项因果测试在保留提示上评估它们：社区必要性、层子集内的充分性以及层必要性。这些测试识别出了那些移除后会破坏计算功能的神经元群组。

    arXiv:2611.23112v1 Announce Type: cross  Abstract: What reusable units of computation can we discover within a language model's neurons? Prior work has shown that individual forward passes can often be sparsified post-hoc to small groups of neurons, but it has been unclear whether these groups form coherent computational entities reused across inputs. We introduce neuron community discovery, an unsupervised method for finding neuron groups that recur across prompts and are causally necessary and sufficient, within the layers they span, for common computations. The method combines gradient-based attribution, a graph connecting neurons that are repeatedly attributed together, and community detection. Applied to Llama-3.1-8B using ten million prompts, it discovers thousands of candidate communities. We evaluate them on held-out prompts through three causal tests: community necessity, sufficiency within a layer subset, and layer necessity. These tests identify groups whose removal disrupts
    
[^25]: 基于事件相机的语音过程中动态唇部区域估计

    Event-Based Dynamic Lip Region Estimation During Speech

    [https://arxiv.org/abs/2611.23111](https://arxiv.org/abs/2611.23111)

    本文提出首个从完整事件帧预测唇部区域密集热力图的方法，为事件相机的唇读等语音应用提供了自动的动态唇部空间定位，解决了现有方法依赖手工裁剪ROI或关键点定位与应用需求不匹配的问题。

    

    事件相机具有高时间分辨率和低功耗的特点，使其在语音相关应用中前景广阔，这些应用包括自动唇读、视听语音分析以及人机交互，尤其适用于低照度或高动态范围等具有挑战性的环境。然而，现有的基于事件的唇读、说话人识别和语音活动检测方法大多在手工定义或由数据集给定的裁剪感兴趣区域（ROI）上运行，目前尚无基于事件的方法能够以空间区域的形式提供该ROI；而现有基于事件的人脸分析仅以关键点的形式给出嘴部位置，这造成了应用与定位之间的不匹配。本工作从完整的事件“帧”中预测唇部区域的密集热力图。事件被编码为极性分离的体素网格，并由具有可学习的对数空间界限的双通路连续时间循环网络（CT-RNN）在每一个位置进行处理。

    arXiv:2611.23111v1 Announce Type: cross  Abstract: Event cameras offer high temporal resolution and low power consumption, making them promising for speech-related applications. These include automatic lip reading, audiovisual speech analysis, and human-robot interaction, particularly in challenging environments characterized by low illumination or high dynamic range. Event-based lip reading, speaker recognition, and voice activity detection, however, mostly operate on a cropped region of interest (ROI), by hand or by the dataset, and no event-based method supplies that ROI as a spatial region; existing event-based face analysis gives the mouth as landmarks, which presents a mismatch between application and localization. The present work predicts a dense heatmap over the lip region from the full event "frame". Events are encoded as a polarity-separated voxel grid and processed at every location by a two-pathway continuous-time recurrent network (CT-RNN) with learned, log-space bounded 
    
[^26]: 树中有什么？层次化 n-gram 模型作为语言的可控近似器

    What's in a Tree ? Hierarchical $n$-gram models as controlled approximators of language

    [https://arxiv.org/abs/2611.23093](https://arxiv.org/abs/2611.23093)

    本文提出 tree-grams——一种通过二阶和三阶张量层次化组合构建的自回归 n-gram 模型，其上下文窗口与模型容量分别由超参数 n 和 d 可控地调节，在超越传统 n-gram 基线的同时可与 Transformer 语言模型媲美，并可作为研究语言的可控近似器用于超参数迁移等实验。

    

    层次化模型已被提出作为研究语言的合成环境，但它们与现实语言语料库的相关性仍未得到充分探索。在这项工作中，我们提出了 tree-grams，这是一类自回归生成式 n-gram 模型，通过对二阶和三阶张量的层次化组合来构建条件概率张量 p(x_n | x_{1:n-1})。其中，n 控制最大上下文窗口，而秩超参数 d 控制模型容量，由此产生了一个可解释且可系统性扩展的语言近似器家族。我们展示了 tree-grams 如何超越现有的 n-gram 基线，同时与基于深度 Transformer 的语言模型保持竞争力。我们还通过将 tree-grams 用作生成模型，展示了这种受控近似的实用性。特别是，我们表明当在更简单的数据上调优更窄的模型时，μP 下的超参数迁移可以变得更加稳健。

    arXiv:2611.23093v1 Announce Type: cross  Abstract: Hierarchical models have been proposed as synthetic settings for studying language. However, their relevance to realistic language corpora remains underexplored. In this work, we introduce tree-grams, autoregressive generative $n$-gram models that build the conditional probability tensor $p(x_n \mid x_{1:n-1})$ through a hierarchical composition of order-2 and order-3 tensors. $n$ controls the maximal context window, whereas a rank hyperparameter $d$ controls model capacity, yielding an interpretable and systematically scalable family of approximators of language. We show how tree-grams outperform existing $n$-gram baselines, while remaining competitive with deep transformer-based language models. We demonstrate the utility of such controlled approximations by using tree-grams as generative models. In particular, we show that hyperparameter transfer under $\mu$P can be made more robust when narrower models are tuned on simpler data, ge
    
[^27]: 学习撤销：状态空间变换下的迁移强化学习

    Learning to Undo: Transfer Reinforcement Learning under State Space Transformations

    [https://arxiv.org/abs/2611.23081](https://arxiv.org/abs/2611.23081)

    该论文提出通过匹配两个MDP的状态特征统计量来学习“撤销映射”，从而将源MDP的策略迁移到目标MDP，并证明在撤销映射为线性且源满足线性$Q^\star$可实现的条件下，其样本复杂度严格优于在目标MDP中从零开始学习。

    

    强化学习（RL）中的迁移学习已经展现出强大的实证成功。在这项工作中，我们采用更具原则性的视角，研究在什么条件下以及如何在两个MDP（源MDP与目标MDP）之间迁移知识能够被证明是有益的。具体而言，我们考虑存在一个“撤销映射”（undo map）的情形，即将该映射应用于目标的状态空间可以精确地恢复出源MDP。我们提出了一种算法，通过匹配从两个MDP中收集到的状态特征统计量来学习该映射，然后利用它将源策略迁移到目标MDP。我们从理论上论证了该算法：分析了撤销映射为线性且源MDP满足线性-$Q^\star$可实现的设定，在该设定下，我们的方法在目标MDP中具有严格优于从零开始学习（tabula rasa RL）的样本复杂度。在实证方面，我们证明了这些优势可以扩展到该理论设定之外：在具有挑战性的连续控制任务和Atari游戏上，我们的方法……

    arXiv:2611.23081v1 Announce Type: cross  Abstract: Transfer learning in reinforcement learning (RL) has shown strong empirical success. In this work, we take a more principled perspective by studying when and how transferring knowledge between MDPs (a source and a target) can be provably beneficial. Specifically, we consider the case where there exists an undo map such that applying this map to the target's state space recovers the source exactly. We propose an algorithm that learns this map by matching state feature statistics gathered from both MDPs, and then uses it to transfer the source policy to the target. We theoretically justify the algorithm by analyzing the setting when the undo map is linear and the source is linearly-$Q^\star$ realizable, where our approach has strictly better sample complexity than tabula rasa RL in the target MDP. Empirically, we demonstrate that these benefits extend beyond this regime: on challenging continuous control tasks and Atari games, our method
    
[^28]: 面向多目标组合问题的帕累托感知智能“预测后优化”方法

    Pareto-Informed Smart "Predict, then Optimize" for Multi-Objective Combinatorial Problems

    [https://arxiv.org/abs/2611.23076](https://arxiv.org/abs/2611.23076)

    提出了一种帕累托感知的SPO+扩展方法，利用真实帕累托前沿构建与采样偏好向量对齐的目标决策并最小化遗憾，克服了现有决策聚焦学习方法在多目标组合优化中因简单标量化而无法捕捉丰富权衡关系的局限。

    

    许多现实世界的规划应用，从物流、路径规划到资源分配，都会产生需要平衡多个相互冲突目标的优化问题。虽然机器学习常被用于预测这些问题的未知参数或成本，但基于预测精度训练的标准模型可能导致次优的规划方案。决策聚焦学习通过将下游优化任务整合到训练目标中来解决这一问题。然而，现有的多目标问题DFL方法依赖于简单的标量化，无法捕捉帕累托前沿上丰富的权衡关系。为了填补这一空白，我们提出了一种帕累托感知的扩展方法，将单目标智能“预测后优化”方法扩展到多目标组合优化问题。我们的损失函数利用真实的帕累托前沿，构建与采样偏好向量相一致的目标决策，并最小化遗憾。

    arXiv:2611.23076v1 Announce Type: new  Abstract: Many real-world planning applications, from logistics, routing to resource allocation, give rise to optimization problems that need to balance multiple, conflicting objectives. While machine learning is often used to predict the unknown parameters or costs of these problems, standard models trained on prediction accuracy can lead to suboptimal plans. Decision-Focused Learning (DFL) addresses this by integrating the downstream optimization task into the training objective. However, existing DFL methods for multi-objective problems rely on simple scalarization, which fails to capture the rich trade-offs of the Pareto Frontier. To address this gap, we propose a Pareto-aware extension to single-objective Smart "Predict, then Optimize" (SPO+) for multi-objective combinatorial optimization problems. Our loss function leverages the true Pareto Frontier, forming a target decision that aligns with a sampled preference vector, and minimizes regret
    
[^29]: 面向智能体治理的多条自然语言准则运行时验证

    Runtime Verification of Multiple Natural Language Criteria for Agent Governance

    [https://arxiv.org/abs/2611.23074](https://arxiv.org/abs/2611.23074)

    提出了一种基于树注意力的并行验证器VFM，可在运行时对智能体输出同时评估数百条自然语言治理准则并给出逐条三元判定，其预训练语料库CriteriaBank带来了跨域性能提升。

    

    AI智能体受数百条自然语言准则的约束，然而现有的验证器无法高效地将智能体的输出与大量此类准则进行比对评估，并报告哪些准则被满足、被违反或不适用。我们提出了VFM，一种树注意力验证器，它对共享上下文和目标文本进行编码，并行地对N条自然语言准则进行评分，返回相互独立的逐准则三元判定结果。VFM在CriteriaBank语料库上进行了预训练，该语料库包含394K个（上下文、目标、准则、判定）元组，涵盖超过321K条不同的准则字符串。经微调的VFM-4B在一个合成治理基准上优于GPT-5.4。在DynaBench多准则基准上，它在轨迹级失败检测方面优于DynaGuard-4B，并且其逐准则判定结果能够定位被违反的规则，支持将问题聚焦升级至更大的思维链评判器。我们证明CriteriaBank预训练可以提升跨域（泛化能力）。

    arXiv:2611.23074v1 Announce Type: cross  Abstract: AI agents are governed by hundreds of natural language criteria, yet existing verifiers cannot efficiently evaluate an agent's output against a large set of such criteria and report which are satisfied, violated, or inapplicable. We introduce the VFM, a tree-attention verifier that encodes the shared context and target text, and scores N natural language criteria in parallel, returning independent per-criterion ternary verdicts. The VFM is pretrained on CriteriaBank, a corpus of 394K (context, target, criterion, verdict) tuples spanning over 321K distinct criterion strings. A finetuned VFM-4B outperforms GPT-5.4 on a synthetic governance benchmark. On the DynaBench multi-criterion benchmark, it outperforms DynaGuard-4B on trace-level failure detection, and its per-criterion verdicts localize the violated rule and support focused escalation to a larger chain-of-thought judge. We show that CriteriaBank pretraining improves cross-domain t
    
[^30]: 通过校正自训练实现摊销式退火采样

    Amortizing Tempered Sampling with Corrected Self-Training

    [https://arxiv.org/abs/2611.23072](https://arxiv.org/abs/2611.23072)

    提出了一种在退火目标序列上进行校正自训练的神经采样方法SSS，通过基于得分的扩散提议与路径空间重要性权重校正，将目标分布的全局结构编码进模型以摊销重复采样成本，在基准测试中取得有竞争力的采样精度且运行时间更短。

    

    学习一个神经采样器可以降低从非归一化分布中重复采样的成本，但获得良好的训练数据仍然困难。我们开发了一种跨越一系列退火（温度调节）目标的校正自训练方案，利用重要性采样和马尔可夫链蒙特卡洛（MCMC）来校正样本，并使用重放缓冲区来存储它们。我们引入了序列得分采样器作为一个具体实例。在SSS中，提议分布是通过在能量梯度上进行回归训练的基于得分的扩散模型；样本则通过路径空间上的重要性权重进行校正。训练完成后，可以直接从模型本身、从其经重要性校正的提议、或从最终温度的缓冲区中收集样本。该方法通过将目标分布的全局结构编码到模型中，从而摊销了重复采样的成本。SSS在各项基准测试中取得了具有竞争力的采样精度，并且在若干目标上以更短的运行时间……

    arXiv:2611.23072v1 Announce Type: cross  Abstract: Learning a neural sampler can reduce the cost of repeated sampling from an un-normalized distribution, but obtaining good training data remains difficult. We develop a corrected self-training scheme across a sequence of tempered targets, making use of importance sampling and Markov chain Monte Carlo to correct the samples and replay buffers to store them. We introduce Sequential Score Sampler (SSS) as a particular case. In SSS, the proposal is a score-based diffusion trained by regression on energy gradients; and the samples are corrected by importance weights on the path space. After training, samples can be collected directly from the model, from its importance-corrected proposals, or from the final-temperature buffer. The method amortizes repeated sampling by encoding the global structure of the target in the model. SSS achieves competitive sampling accuracy across the benchmarks and, on several targets, uses shorter runtimes with f
    
[^31]: FrameRouter：面向长视频理解的帧预算路由方法——基于Video-MME-v2及更多基准

    FrameRouter: Frame Budget Routing for Long-Video Understanding on Video-MME-v2 and Beyond

    [https://arxiv.org/abs/2611.23055](https://arxiv.org/abs/2611.23055)

    提出无需训练的测试时框架FrameRouter，通过证据需求路由器为每个问题动态分配不同的帧预算（保持平均预算与统一参考一致），从而更有效地提升长视频问答的准确性。

    

    像Video-MME-v2这样的长视频问答基准，通常在为每个问题分配统一帧预算的条件下评估系统，然而各个查询实际所需的视觉证据存在显著差异。我们在Video-MME-v2上分析了这种不匹配现象，发现不同的问题子集各自偏好不同的固定帧预算策略，而简单地统一增加帧数量并不能可靠地改善所有指标。这些发现启发我们提出了“固定均值帧预算路由”这一受控推理设置：每个问题可以分配到不同的帧预算，同时整体平均预算与统一预算的参考设置保持一致。我们提出FrameRouter，一个无需训练的测试时框架，它结合了证据需求路由器、预算约束分配规则以及一组固定的帧采样策略库。通过在保持底层采样器不变的前提下控制每个问题所获得的视觉证据量，FrameRouter与查询（摘要在此处截断）

    arXiv:2611.23055v1 Announce Type: new  Abstract: Long-video question answering benchmarks like Video-MME-v2 typically evaluate systems under a uniform frame budget for every question, despite stark variation in the evidence each query actually demands. We analyze this mismatch on Video-MME-v2 and find that different question subsets prefer different frozen frame-budget policies, while uniformly increasing the frame count does not reliably improve all metrics. These findings motivate fixed-mean frame-budget routing, a controlled inference setting in which each question may receive a different frame budget while the average budget matches a uniform reference. We introduce FrameRouter, a training-free test-time framework that combines an evidence-demand router, a budget-constrained allocation rule, and a frozen bank of frame-sampling policies. By controlling how much visual evidence each question receives while leaving the underlying samplers unchanged, FrameRouter is orthogonal to query-
    
[^32]: Magflow：工业级六自由度磁悬浮的可微仿真

    Magflow: Differentiable Simulation of Industrial-Grade 6D Magnetic Levitation

    [https://arxiv.org/abs/2611.23050](https://arxiv.org/abs/2611.23050)

    提出了首个完全可微、GPU并行的六自由度磁悬浮仿真器Magflow，并利用它训练出首个工业级六自由度磁悬浮系统的神经控制器，突破了传统手工近似电磁力计算方法对工作空间和动子拓扑的保守限制。

    

    具有六自由度的磁悬浮系统原则上能够实现无摩擦运输和微米级定位，但由于其复杂且不稳定的动力学特性，底层控制本质上极具挑战性。工业系统通常依赖深度的软硬件协同设计，即动子与定子的几何结构被联合设计，以实现对电磁力计算的紧凑手工近似，从而使控制算法能够在千赫兹频率下保持计算上的可行性。然而，这种方法对工作空间范围以及所支持的动子拓扑类型施加了过于保守的限制。我们不再将底层物理简化为手工近似，而是直接计算物理本身。我们提出了Magflow，一个完全可微、GPU并行的六自由度磁悬浮仿真器。该仿真器被用于训练首个面向工业级六自由度磁悬浮系统的神经控制器

    arXiv:2611.23050v1 Announce Type: cross  Abstract: Magnetic levitation with six-degrees-of-freedom enables frictionless transport and micrometer-scale positioning in principle, but low-level control is inherently challenging due to its complex and unstable dynamics. Industrial systems typically rely on deep hardware-software co-design, where mover and stator geometries are jointly designed to enable a compact, hand-crafted approximation of the electromagnetic force computation, thereby keeping control computationally tractable at kilohertz rates. However, this approach imposes overly conservative constraints on the operating workspace and the range of supported mover topologies. Instead of reducing the underlying physics to a hand-crafted approximation, we compute the physics directly. We introduce Magflow, a fully differentiable, GPU-parallel simulator for 6D magnetic levitation. It is used to train the first neural controller for an industrial-grade 6D magnetic levitation system that
    
[^33]: 逆转文本并非逆转路径：一个缺乏零点的固定文本度量，以及一个恒为零的组装顺序比较

    Reversing the text is not reversing the path: a fixed-text measure without a zero, and an assembly-order comparison that vanishes identically

    [https://arxiv.org/abs/2611.23049](https://arxiv.org/abs/2611.23049)

    该论文指出单篇固定文本上的逆转比较度量（拟“熵产生”）缺乏由定理保证的零点，并利用在全部 n! 种句子排列下评分所得到的精确反对称性 Δ(π) = −Δ(π∘rev) 为其赋予零点，证明基于文本逆转的组装顺序比较恒为零，进而以作者排布的无量纲有界中位秩作为文本组织程度的替代度量。

    

    对于轨迹系综而言，涨落定理赋予了熵产生以意义和零点。最近，同一个表达式被应用于单个固定文本，即 Δ = [log P(x) − log P(x̃)]/L，其中 x̃ 是顺序颠倒后的文本，并被解读为一种区分因果有序散文与任意排序散文的熵产生。然而，单个文本既不是分布也不是动力学，且所讨论的文本是由其他模型生成的。没有任何定理能确定固定文本量的零点位于何处：表达式可以从系综搬过来，但其零点却不能。我们通过一个精确的反对称性为其提供一个零点：在全部 n! 种排列顺序下对文本的 n 个句子进行评分时，逆转操作将排列 π 映为 π∘rev，这是一个保测度的对合变换，因此恒有 Δ(π) = −Δ(π∘rev)，且对于任何评分方法，数值都以零为中心。于是，作者排布的中位秩是无量纲的、有界的，且为 0.50。

    arXiv:2611.23049v1 Announce Type: cross  Abstract: For an ensemble of trajectories the fluctuation theorems give the entropy production its meaning and zero. The same expression has recently been evaluated on one fixed text, $\Delta = [\log P(x) - \log P(\tilde x)]/L$ with $\tilde x$ the text in reversed order, read as an entropy production separating causally from arbitrarily ordered prose. A single text is neither a distribution nor a dynamics, and the texts at issue were produced by other models. No theorem fixes where the zero of a fixed-text quantity lies: the expression carries over from an ensemble, its zero does not.   We supply a zero from an exact antisymmetry. Scoring a text's $n$ sentences in all $n!$ orders, reversal carries $\pi$ to $\pi \circ \mathrm{rev}$, a measure-preserving involution, so $\Delta(\pi) = -\Delta(\pi \circ \mathrm{rev})$ identically and the values are centred at zero for any score. The authored arrangement's midrank is then unitless, bounded, and 0.50 
    
[^34]: 揭示常规放射学中的不可见信息：基于物理引导的3D自监督学习从CBCT中跨模态检测经MRI验证的软组织病变

    Uncovering the Invisible in Routine Radiology: Cross-Modal Detection of MRI-Verified Soft-Tissue Pathologies from CBCT via Physics-Guided 3D Self-Supervised Learning

    [https://arxiv.org/abs/2611.23042](https://arxiv.org/abs/2611.23042)

    该研究首次证明常规CBCT扫描中物理性地保留了软组织病变的亚视觉放射学特征，并通过物理引导的两阶段3D自监督学习框架（利用在6,550个未标记CBCT体积上预训练的掩码自编码器），在配对数据极度稀缺的条件下实现了对经MRI验证的软组织病变（以颞下颌关节为模型）的跨模态机会性筛查。

    

    常规X射线成像模态（如锥形束计算机断层扫描，CBCT）针对高对比度的矿化组织进行了优化，导致软组织异常被掩藏在光子噪声之下并低于人类视觉感知阈值。因此，检测非钙化病变通常需要进行二次磁共振成像（MRI）。我们证明，软组织病变的亚视觉放射学特征在常规CBCT扫描中被物理性地保留下来，并且可以通过一个有针对性的、物理引导的深度学习流程将其提取出来用于机会性筛查，即使在极端的配对数据稀缺条件下也能实现。以颞下颌关节（TMJ）作为模型系统（N = 212个配对的CBCT-MRI关节），我们通过两阶段框架解决了高容量3D网络的严重过拟合问题。首先，我们采用一个冻结的全网格3D掩码自编码器（BertsWin），该模型通过GradientConductor（GCond）优化器在6,550个未标记的CBCT体积上进行了预训练，并结合解剖学……

    arXiv:2611.23042v1 Announce Type: cross  Abstract: Routine X-ray modalities like cone-beam computed tomography (CBCT) are optimized for high-contrast mineralized tissues, leaving soft-tissue abnormalities buried beneath photon noise and below human visual perception. Thus, detecting non-calcified pathologies requires secondary magnetic resonance imaging (MRI). We demonstrate that sub-visual radiographic signatures of soft-tissue pathology are physically preserved within routine CBCT scans and can be extracted for opportunistic screening using a targeted, physics-guided deep learning pipeline, even under extreme paired-data scarcity. Using the temporomandibular joint (TMJ) as a model system (N = 212 paired CBCT-MRI joints), we address the severe overfitting of high-capacity 3D networks through a two-stage framework. First, we employ a frozen full-grid 3D masked autoencoder (BertsWin) pre-trained on 6,550 unlabeled CBCT volumes via the GradientConductor (GCond) optimizer and an anatomy-a
    
[^35]: 方差恢复优于协方差秩：内存预算约束下的解析持续学习

    Variance Restoration over Covariance Rank: Memory-Budgeted Analytic Continual Learning

    [https://arxiv.org/abs/2611.23040](https://arxiv.org/abs/2611.23040)

    该论文针对固定内存预算下的解析持续学习，提出只保存随维度线性增长的坐标方差与类别统计量、以替代二次增长的完整协方差矩阵，从而在保留宽随机码带来的线性可分性提升的同时大幅降低内存开销。

    

    解析持续学习用作用于冻结预训练表征的闭式岭回归取代迭代梯度更新，将所有历史会话总结为两个持续更新的充分统计量：特征自相关矩阵 $\Sigma_t$ 和类别互相关矩阵 $\Gamma_t$。将骨干网络特征提升到维度 $M \gg D$ 的宽随机码中能显著改善跨任务的线性可分性，但会使 $\Sigma_t$ 以 $\mathcal{O}(M^2)$ 的速度二次增长。在固定的持久内存预算 $\Bcal$ 下，学习者面临一个基本的结构性权衡：要么在较窄的特征码中存储精确的成对共激活信息，要么扩展到宽码同时压缩跨坐标的交互作用。我们观察到，坐标自方差 $v_t = \diag(\Sigma_t)$ 和类别统计量 $\Gamma_t$ 仅随 $M$ 线性增长，而内存开销主要由非对角的交互项主导。我们提出了 \method……（注：原文摘要在此处被截断）

    arXiv:2611.23040v1 Announce Type: new  Abstract: Analytic continual learning replaces iterative gradient updates with closed-form ridge regression over frozen pre-trained representations, summarizing all past sessions in two running sufficient statistics: the feature autocorrelation matrix $\Sigma_t$ and the class cross-correlation matrix $\Gamma_t$. Lifting backbone features into a wide random code of dimension $M \gg D$ substantially improves linear separability across tasks, but forces $\Sigma_t$ to grow quadratically as $\mathcal{O}(M^2)$. Under a fixed persistent memory envelope $\Bcal$, a learner faces a fundamental structural trade-off: store exact pairwise co-activations inside a narrower feature code, or expand to a wide code while compressing cross-coordinate interactions. We observe that the coordinate self-variances $v_t = \diag(\Sigma_t)$ and class statistics $\Gamma_t$ grow only linearly with $M$, whereas off-diagonal interactions dominate memory. We introduce \method (\t
    
[^36]: 当锚点偏移目标：面向扩散强化学习的优势门控自适应方法

    When Anchors Shift the Target: Advantage-Gated Adaptation for Diffusion Reinforcement Learning

    [https://arxiv.org/abs/2611.23038](https://arxiv.org/abs/2611.23038)

    本文提出优势门控自适应（AGA），通过按优势符号解耦正负样本的角色——正样本保持标准目标拟合、负样本改用凹目标在移动策略预测处的一阶切线——从而解决扩散强化学习中的锚点耦合冲突。

    

    前向过程强化学习通过将获得奖励的生成结果重新加噪作为回归目标，来适配扩散模型和流匹配模型。由于负优势会导致具有无界排斥梯度的凹型更新，现有方法利用移动策略副本的缓存预测来稳定训练，然而这些锚点同时也会改变正反馈所引导模型拟合的目标。我们将这一冲突形式化为锚点耦合问题：在分离冻结参考正则化项之后，仿射移动锚点规则无法在保留所有正回归目标的同时消除负曲率。我们提出优势门控自适应（AGA）来解耦这两种角色。正样本保留标准的目标拟合，其自限制更新在观测目标处自然消失，且不受移动锚点的影响；负样本则转而采用凹目标在移动策略预测处的一阶切线，从而保留局部……

    arXiv:2611.23038v1 Announce Type: cross  Abstract: Forward-process reinforcement learning adapts diffusion and flow-matching models by re-noising rewarded generations into regression targets. Because negative advantages induce concave updates with unbounded repulsive gradients, existing methods stabilize training using cached predictions from a moving policy copy, yet these anchors also alter what positive feedback teaches the model to fit. We formalize this conflict as anchor coupling: after separating frozen-reference regularizers, affine moving-anchor rules cannot eliminate negative curvature while preserving every positive regression target. We propose Advantage-Gated Adaptation (AGA) to decouple these roles. Positive samples retain standard target fitting, whose self-limiting update vanishes at the observed target independently of the moving anchor. Negative samples instead use the first-order tangent of the concave objective at the moving-policy prediction, preserving the local r
    
[^37]: The K=V Cache of Gemma 4 Only Needs to Be Stored Once: Achieving Exact Global-KV Memory Savings in a Paged Serving Engine Without Modifying the Attention Kernel

    Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels

    [https://arxiv.org/abs/2611.23035](https://arxiv.org/abs/2611.23035)

    This paper exploits the linear dependence between the key and value in the global attention layers of Gemma 4 31B (K=V), storing only one 640-value row without modifying the attention kernel, which reads the key and value caches through two overlapping strided views, achieving a 1.34 to 1.40 fold reduction in KV cache memory during decoding with bit-identical outputs.

    

    In the global attention layers of Gemma 4 31B, the value is the normalized key projection, and the key is the value multiplied by a fixed weight, with rotary position encoding applied to only one quarter of its dimensions. The Gemma 4 technical report points out that this design can reduce the global KV cache by up to 37.5%, but the open-source serving engines we inspected still store a complete key and a complete value: 1024 values per KV head per token. We instead store only one row of 640 values, which is exact in real-number arithmetic, and an unmodified paged attention kernel reads it as its key cache and value cache through two overlapping strided views. In a vLLM prototype using the 16-bit 31B model, the outputs are bit-identical to an emulation of the same arithmetic on the original cache; 2 of 432 final answers differ from the original engine (within its run-to-run noise range); and at 32K to 64K tokens, the steady decoding KV memory per request decreases by a factor of 1.34 to 1.40. On a pair of prefill/decode...

    arXiv:2611.23035v1 Announce Type: cross  Abstract: In the global attention layers of Gemma 4 31B, the value is the normalized key projection, and the key is the value scaled by a fixed weight, with rotary position encoding on one quarter of its dimensions. The Gemma 4 report states that this design reduces the global KV cache by up to 37.5%, but the open serving engines that we checked still store a full key and a full value: 1,024 values per KV head and token. We store one 640-value row instead, exact in real arithmetic, and an unchanged paged attention kernel reads it through two overlapping strided views as its key cache and its value cache. In a vLLM prototype with the 16-bit 31B model, the outputs are bit-identical to an emulation of the same arithmetic on the stock cache, 2 of 432 final answers differ from the stock engine (within its run-to-run noise), and the steady decode KV memory per request falls by a factor of 1.34 to 1.40 at 32K to 64K tokens. On a prefill/decode pair of 
    
[^38]: NVIDIA Kumo Tabular：推动表格预测任务的准确率—效率前沿

    NVIDIA Kumo Tabular: Pushing the Accuracy$\unicode{8211}$Efficiency Frontier for Tabular Prediction

    [https://arxiv.org/abs/2611.23001](https://arxiv.org/abs/2611.23001)

    Kumo Tabular 是 NVIDIA 推出的系列开源表格基础模型，通过在人工生成表格上预训练并利用上下文学习技术，无需任何针对新表格的训练即可完成预测，并在八项基准测试中夺得第一名。

    

    我们推出了 Kumo Tabular，这是一系列用于分类和回归任务的开源表格基础模型，共有三种规模：Kumo Tabular-S（2700万参数）、Kumo Tabular-M（6100万参数）和 Kumo Tabular-L（2.14亿参数）。这些模型仅在来自 Kumo 先验的人工生成表格上进行预训练，无需任何额外训练即可直接对新表格进行预测。在该方法中，表格编码器将每一行压缩为嵌入表示，随后由一个上下文学习变换器根据有标签的行来预测无标签的行。Kumo Tabular 引入了提升训练稳定性和预测性能的架构创新，使模型能够在现代 GPU 上扩展至1.5亿个单元格。Kumo Tabular-L 在八项基准测试中排名第一：TabArena、TALENT、BeyondArena、ScoringBench、STRABLE、MulTaBench、RamanBench 和 TabBench-Bio，这些基准涵盖了概率回归、超过100万行或10万列的表格，以及包含字符串、文本或图像的表格。准确率随模型规模的……（原文摘要在此处截断）

    arXiv:2611.23001v1 Announce Type: new  Abstract: We introduce Kumo Tabular, a family of open tabular foundation models for classification and regression, in three sizes: Kumo Tabular-S (27M parameters), Kumo Tabular-M (61M), and Kumo Tabular-L (214M). Pretrained only on artificial tables from the Kumo prior, they can make predictions on new tables without any training. In our approach, a table encoder compresses each row into an embedding, and an in-context learning transformer predicts the unlabeled rows from the labeled ones. Kumo Tabular introduces architectural innovations that improve both training stability and predictive performance, enabling the model to scale up to 150M cells on modern GPUs. Kumo Tabular-L ranks first on eight benchmarks: TabArena, TALENT, BeyondArena, ScoringBench, STRABLE, MulTaBench, RamanBench, and TabBench-Bio. These cover probabilistic regression, tables with over 1M rows or 100K columns, and tables with strings, text, or images. Accuracy improves with m
    
[^39]: 基于冻结世界模型的残差引导轻量级强化学习风险感知轨迹选择

    Residual-Guided Lightweight Reinforcement Learning for Risk-Aware Trajectory Selection with a Frozen World Model

    [https://arxiv.org/abs/2611.22998](https://arxiv.org/abs/2611.22998)

    该论文提出一种基于冻结世界模型的残差引导轻量级强化学习框架，仅需微调约1.6万参数即可实现自动驾驶中风险感知的高效轨迹选择。

    

    在自动驾驶中，世界模型为评估候选轨迹提供未来场景预测，但固定的选择规则可能无法捕捉依赖于场景的权衡。我们提出了一种残差引导的强化学习（RL）框架，用于在冻结世界模型和固定候选库的基础上进行轻量级轨迹选择。该框架使用近端策略优化（PPO）来微调覆盖候选选择和有界轨迹残差的联合策略。一个上下文条件排序网络通过评估所选轨迹的局部扰动所获得的反馈来学习候选偏好。在推理阶段，选择器使用固定权重结合学习到的偏好排名和预测的风险排名，并返回所选的候选轨迹而不进行修改。微调仅更新actor和critic中总共16,106个参数，而部署的选择器仅保留11,725个参数的排序网络……

    arXiv:2611.22998v1 Announce Type: new  Abstract: In automated driving, world models provide future scene predictions for evaluating candidate trajectories, but fixed selection rules may not capture scene-dependent trade-offs. We propose a residual-guided reinforcement learning (RL) framework for lightweight trajectory selection with a frozen world model and a fixed candidate bank. The framework uses proximal policy optimization (PPO) to fine-tune a joint policy over candidate selection and bounded trajectory residuals. A context-conditioned ranking network learns candidate preferences from feedback obtained by evaluating local perturbations of the selected trajectories. At inference, the selector combines learned preference and predicted risk ranks using fixed weights and returns the selected candidate trajectory without modification. Fine-tuning updates 16,106 parameters across the actor and critic, while the deployed selector retains only the 11,725-parameter ranking network, with th
    
[^40]: 基于扩散模型引导采样的高维黑盒优化

    High-Dimensional Black-Box Optimization via Guidance Sampling of Diffusion Models

    [https://arxiv.org/abs/2611.22965](https://arxiv.org/abs/2611.22965)

    该论文提出了BReaD框架，利用扩散模型的引导采样技术，在不需要大量标注数据或精确奖励梯度的情况下，有效解决高维黑盒优化问题。

    

    许多现实世界的优化问题涉及对黑盒目标函数进行优化，而这类目标只能获得有限次数的评估。贝叶斯优化（BO）是解决此类问题的强大框架，但其有效性在高维空间中往往会出现下降。近期的研究利用扩散模型进行黑盒优化，通常通过预训练一个奖励条件化的得分网络来引导生成过程向高奖励样本偏移。另一类基于梯度的引导方法则假设可以访问一个可微分的奖励函数，从而查询引导梯度。然而，这些方法要么需要大量标注数据进行离线训练，要么需要获取精确的奖励梯度，这限制了它们在黑盒场景中的适用性。我们提出了BReaD（黑盒奖励引导的扩散模型采样），这是一个用于优化黑盒奖励的引导采样框架，无需假设能够访问标注数据。

    arXiv:2611.22965v1 Announce Type: cross  Abstract: Many real-world optimization problems involve optimizing black-box objectives for which only a limited number of evaluations are available. Bayesian optimization (BO) is a powerful framework for such problems, but its effectiveness often deteriorates in high-dimensional spaces. Recent work leverages diffusion models for black-box optimization, often through pre-training a reward-conditioned score network to steer generation towards high-reward samples. Alternatively, gradient-based guidance methods assume access to a differentiable reward function from which guidance gradients are queried. However, these approaches either require large amounts of labeled data for offline training or access to exact reward gradients, limiting their applicability in black-box settings. We propose BReaD (Black-box Reward-guided Sampling of Diffusion Model), a guidance sampling framework for optimizing black-box rewards without assuming access to a labeled
    
[^41]: 用于端到端图像生成的层级扩散Transformer

    Hierarchical Diffusion Transformer for End-to-End Image Generation

    [https://arxiv.org/abs/2611.22962](https://arxiv.org/abs/2611.22962)

    提出层级扩散Transformer（HiT），通过逐步将相关token分组压缩来构建视觉层级结构，使端到端像素空间扩散模型在生成质量和训练效率上追赶领先的两阶段潜在扩散模型。

    

    像素空间扩散模型在单一的端到端训练阶段中联合学习视觉表示和图像生成，而无需单独预训练的视觉编码器。然而，朴素的像素空间Transformer在生成质量和训练效率方面仍然落后于领先的两阶段潜在扩散模型。我们假设这种差距部分源于在整个骨干网络中每个图像块始终保留一个token的做法，这种方式没有明确鼓励一种能在逐步粗化的层级上总结相关内容的视觉层级结构。为解决这一问题，我们提出了层级扩散Transformer（HiT），它逐步地将相关的token进行分组，并用单个token来表示每个组。主Transformer瓶颈部分处理这些紧凑的组token，而解除分组操作则将特征恢复到其原始的图像位置。压缩预算机制调节整个批次中的平均压缩率，使组成员关系和数量能够……（摘要在此处截断）

    arXiv:2611.22962v1 Announce Type: cross  Abstract: Pixel-space diffusion learns visual representations and image generation jointly in a single end-to-end training stage, without a separately pretrained visual encoder. However, plain pixel-space Transformers still trail leading two-stage latent diffusion models in generation quality and training efficiency. We hypothesize that part of this gap arises from retaining one token per patch throughout the backbone, which does not explicitly encourage a visual hierarchy that summarizes related content at progressively coarser levels. To address this, we introduce Hierarchical Diffusion Transformers (HiT), which progressively group related tokens and represent each group with a single token. The main Transformer bottleneck processes these compact group tokens, and ungrouping restores features to their original image positions. A compression budget regulates the average compression ratio across the batch, allowing group memberships and counts t
    
[^42]: DDSL：面向金融市场状态的迭代式自标注方法

    DDSL: Iterative Self-Labeling for Financial Market States

    [https://arxiv.org/abs/2611.22958](https://arxiv.org/abs/2611.22958)

    DDSL是一种迭代自标注框架，通过价格与语法引导的Viterbi标注器和Transformer分类器之间的模型反馈耦合，构建同时编码当前与未来市场趋势的九状态金融市场表示，在多个指数数据集上将联合状态宏观F1提升78%-96%，且独立分类器可从预测时可用信息中恢复这些标签。

    

    金融市场状态缺乏唯一的真值标准，因此必须从价格数据中构建。传统的标注方法通常独立于下游分类器来定义状态，这使得所得到的标签能否从预测时可用的信息中恢复出来这一问题悬而未决。我们提出了DDSL，一种迭代式自标注框架，它通过模型反馈将价格与语法引导的Viterbi标注器与Transformer分类器耦合起来。DDSL构建了一种九状态表示，同时编码当前和未来的市场趋势，并在整个优化过程中保留结构化的价格发射概率与时序约束。在上证50、沪深300、中证500和纳斯达克100上的实验表明，与单次动态规划标注相比，DDSL将留出集上的联合状态宏观F1提升了78%-96%。一个从未参与标注循环的独立GRU分类器重现了相同的趋势，而标签熵……

    arXiv:2611.22958v1 Announce Type: new  Abstract: Financial market states lack a unique ground truth and must therefore be constructed from price data. Conventional labeling methods typically define states independently of the downstream classifier, leaving unresolved whether the resulting labels can be recovered from information available at prediction time. We propose DDSL, an iterative self-labeling framework that couples a price- and grammar-guided Viterbi labeler with a Transformer classifier through model feedback. DDSL constructs a nine-state representation that jointly encodes current and future market trends while retaining structured price emissions and temporal constraints throughout refinement. Experiments on SSE50, CSI300, CSI500, and NASDAQ100 show that DDSL increases held-out joint-state macro F1 by 78%-96% over single-pass dynamic-programming labeling. An independent GRU classifier that never participates in the labeling loop reproduces the same trend, while label entrop
    
[^43]: 模型自我评估对于多样化的大型视觉语言模型集成是否足够？

    Is model self-assessment enough for diverse LVLM ensembles?

    [https://arxiv.org/abs/2611.22956](https://arxiv.org/abs/2611.22956)

    该论文提出了一个统一的候选响应选择集成流水线，并通过涵盖34种置信度估计方法、24种集成配置和20个数据集的大规模评估证明，仅依赖模型自身置信度的简单免训练、免参数自我评估方法即可在多样化LVLM集成中取得优异表现。

    

    多样化的大型视觉语言模型（LVLM）集成在多个数据集和任务上已展现出超越单一模型的性能。然而，必要的聚合过程通常依赖于训练、参数拟合或复杂的任务特定流水线。此外，现有评估有限，难以进行公平比较。我们创建了一个统一的流水线来执行候选响应选择：通过从不同模型提供的答案集合中选择最佳答案来构建集成。我们在34种不同的置信度估计方法、24种集成配置、20个数据集以及两种提示规范上进行了全面评估。结果表明，简单的免训练和免参数方法就足以奏效，即使对于不同架构的模型集成也是如此。这些简单的自我评估方法仅依赖于每个组成模型自身的置信度，无需显式校准，却能够取得优于现有方法的表现。

    arXiv:2611.22956v1 Announce Type: cross  Abstract: Ensembles of diverse Large Vision Language Models have demonstrated performance beyond that of single models across multiple datasets and tasks. However, the necessary aggregation process usually relies on training, parameter fitting, or complex task-specific pipelines. Furthermore, evaluation is limited, making a fair comparison difficult. We create a unified pipeline where we perform candidate response selection: we build an ensemble by selecting the best answer from a set provided by different models. We comprehensively evaluate across 34 different means of confidence estimation, 24 ensemble configurations, 20 datasets, and two prompt specifications. We show that simple training and parameter-free approaches can be sufficient, even for ensembles of different architectures. These simple self-assessment approaches rely solely on each constituent model's confidence, without explicit calibration, yet are able to perform better than said
    
[^44]: ResoLUT：将残差网络蒸馏为查找表网络

    ResoLUT: Distilling Residual Networks into Lookup-Table Networks

    [https://arxiv.org/abs/2611.22955](https://arxiv.org/abs/2611.22955)

    ResoLUT 通过将训练好的 ResNet 逐层蒸馏为查找表树并保留其多比特残差流，实现了逻辑/查找表网络在 CIFAR-10、CIFAR-100 和 ImageNet 上的最佳准确率，证明 LUT 网络可以随深度扩展。

    

    由布尔门或查找表构建的逻辑网络有望实现高效的硬件推理，但在深度训练以及学习其连接性方面仍然十分困难。我们并不直接将逻辑作为逻辑来训练，而是让 ResoLUT 将训练好的残差网络逐层蒸馏为 LUT 树，同时保留其多比特残差流和 BatchNorm，并从教师网络中提取连接性而非自行学习。据我们所知，ResoLUT 是计算机视觉领域首个保留 ResNet 风格多比特残差流的卷积逻辑网络。训练和推理全程均采用硬性的 LUT 决策，从而避免了逻辑中的离散化鸿沟。ResoLUT-20 在两倍树数量的情况下于 CIFAR-10 上达到 95.66% 的准确率，ResoLUT-56 在 CIFAR-100 上达到 78.87%，在 64px 分辨率的 ImageNet 上达到 53.69%，这是逻辑网络和查找表网络在 CIFAR-10 和 CIFAR-100 上的最佳结果，表明 LUT 网络可以在深度上扩展。（摘要在此处截断）

    arXiv:2611.22955v1 Announce Type: new  Abstract: Logic networks built from Boolean gates or lookup tables (LUTs) promise efficient hardware inference, but training them at depth and learning their connectivity remain difficult. We do not train logic as logic. Instead, ResoLUT distills a trained residual network (ResNet) layer by layer into LUT trees, preserving its multi-bit residual stream and BatchNorm while extracting connectivity from the teacher rather than learning it. To our knowledge, ResoLUT is the first convolutional logic network in computer vision to retain a ResNet-style multi-bit residual stream. Hard LUT decisions are used throughout training and inference, avoiding a discretization gap in the logic. ResoLUT-20 reaches 95.66% on CIFAR-10 with twice the trees, and ResoLUT-56 reaches 78.87% on CIFAR-100 and 53.69% on ImageNet at 64px, the best results of logic and lookup-table networks on CIFAR-10 and CIFAR-100, showing that LUT networks can scale in depth. We explore and 
    
[^45]: 反事实的对称性：从协变量异质性出发研究反事实映射的可识别性与可检验性

    Symmetries of Counterfactuals: Identification and Testability of Counterfactual Maps from Covariate Heterogeneity

    [https://arxiv.org/abs/2611.22948](https://arxiv.org/abs/2611.22948)

    该论文刻画了与观测数据一致的所有反事实映射——它们彼此仅相差一个由协变量条件结果分布决定的对称群（协变量异质性越强，对称群越小），并证明对于三角单调映射和最优传输映射等常见反事实类别，与数据的一致性可归结为一个可检验的条件独立性命题。

    

    一般而言，反事实无法仅凭数据得到识别，因为许多因果模型会产生相同的观测分布，却对应不同的反事实。规避这一问题的常见做法是将机制限制在使其唯一确定的类别中，例如三角单调映射或最优传输映射。然而，所假设的类别是否正确通常并未得到检验。在本工作中，我们研究了当存在仅通过潜在原因影响结果的协变量时，数据能够告诉我们什么。首先，我们刻画了所有与数据一致的反事实映射：它们与真实映射的差异仅在于一组保持“给定协变量条件下结果分布”不变的对称性；这些条件分布在协变量取值之间变化越大，该对称群就越小。其次，我们证明，对于上述类别，与数据的一致性等价于一个单一的条件独立性命题（摘要原文在此处截断）。

    arXiv:2611.22948v1 Announce Type: cross  Abstract: In general, counterfactuals are not identified from data alone, because many causal models produce the same observed distributions but different counterfactuals. A common approach to circumvent this problem is to restrict the mechanism to a class in which it becomes unique, such as triangular monotone maps or optimal transport maps. However, whether the assumed class is correct is usually not examined. In this work, we study what the data can tell us when covariates are available that influence the outcome only through its latent cause. First, we characterise all counterfactual maps that are consistent with the data. They differ from the true map by a group of symmetries that preserve the conditional outcome distributions given the covariates. The more these distributions vary across covariate values, the smaller the group. Second, we show that for the classes above, consistency with the data is equivalent to a single conditional indep
    
[^46]: 核的现实检验：大规模高效注意力机制的基准测试与蒸馏

    The Kernel Reality Check: Benchmarking and Distilling Efficient Attention at Scale

    [https://arxiv.org/abs/2611.22947](https://arxiv.org/abs/2611.22947)

    该论文对核化注意力方法进行了大规模系统性基准测试，发现 FlashAttention 在效率上仍优于最先进的核化方法，且现有核化方法在质量上落后于 softmax 2-10 分，从而挑战了核化注意力作为 softmax 即插即用替代方案的核心假设。

    

    核化注意力方法旨在用 O(N) 复杂度的替代方案取代 softmax 注意力的 O(N²) 复杂度，在过去五年中一直是密集研究的焦点。尽管关注度日益增长，但目前仍缺乏针对不同方法和不同规模的系统性评估与公平比较。我们将核化注意力的分析分为两个部分，并指出各自存在的突出局限。在效率方面，在大多数序列长度下，FlashAttention 在训练阶段以及视觉 Transformer 中都比最先进的（SOTA）核化方法实现了更低的内存占用和更快的实际运行时间（唯一明显的例外是因果语言模型在推理阶段的内存占用），这与将纯核化注意力作为 softmax 即插即用替代方案的核心假设相矛盾。我们还发现，许多已报告的效率提升在与现代基线对比时往往无法复现。在质量方面，当前的核化方法与 softmax 相比仍存在 2-10 分的差距。

    arXiv:2611.22947v1 Announce Type: new  Abstract: Kernelized attention methods, which aim to replace the $O(N^2)$ complexity of softmax attention with $O(N)$ alternatives, have been the focus of intense research in the last five years. Despite growing interest, systematic evaluations and fair comparisons across approaches and scales are still missing. We analyze kernelized attention in two halves and highlight prominent limitations on each. On efficiency, FlashAttention achieves lower memory consumption and faster wall-clock time than SOTA kernelized methods during training and in vision transformers across most sequence lengths (with inference-time memory in causal language models the only clear exception), contradicting the core assumption for pure kernelized attention used as a drop-in replacement for softmax. We also discover that reported efficiency gains often fail to reproduce against modern baselines. On quality, current kernelized methods leave gaps of 2-10 points relative to s
    
[^47]: 基于Transformer的迭代预条件子

    A Transformer-Derived Iterative Preconditioner

    [https://arxiv.org/abs/2611.22941](https://arxiv.org/abs/2611.22941)

    该论文从Transformer的内部计算中提炼出迭代预条件化原语，发展出无需学习组件的独立算法RAPID，可有效改善SPD矩阵条件数并减少共轭梯度求解所需迭代次数，且具有理论收敛保证。

    

    在上下文预测问题上训练的Transformer能够学习已知的数值算法，例如梯度下降。我们更进一步，证明它们的内部计算可以揭示具有可证明保证的新算法。一个经训练用于求解线性系统 $Xw=y$ 的softmax-attention Transformer会隐式地改善矩阵的条件性。通过追踪 $XX^\top$，我们发现了一个迭代预条件化原语，并将其发展为RAPID——一个无需任何学习组件的独立算法。RAPID是一种适用于对称正定（SPD）矩阵的“随时可用”型预条件子，利用随机混合和稀疏局部修正来改善矩阵条件性。在合成问题和真实问题上的概念验证实验表明，该方法持续改善条件数，并减少了达到目标精度所需的共轭梯度迭代次数。对于Haar混合的理想化情形，我们证明了无论谱分布如何，该算法都能收敛到任意固定的目标条件数。

    arXiv:2611.22941v1 Announce Type: new  Abstract: Transformers trained on in-context prediction problems can learn known numerical algorithms, e.g., gradient descent. We go further, showing that their internal computations can reveal new algorithms with provable guarantees. A softmax-attention transformer trained to solve linear systems $Xw=y$ implicitly improves matrix conditioning. Tracing $XX^\top$ reveals an iterative preconditioning primitive, which we develop into RAPID, a standalone algorithm requiring no learned components.   RAPID is an anytime preconditioner for symmetric positive definite (SPD) matrices, using randomized mixing and sparse local corrections to improve conditioning. Proof-of-concept experiments on synthetic and real-world problems consistently improve conditioning and reduce conjugate gradient iterations needed to reach a target accuracy. For a Haar-mixing idealization, we prove convergence to any fixed target condition number regardless of spectral distributio
    
[^48]: 面向通信高效且拜占庭容错联邦学习的投影罚函数梯度追踪方法

    Projected Penalty-Gradient Tracking for Communication-Efficient Byzantine-Resilient Federated Learning

    [https://arxiv.org/abs/2611.22929](https://arxiv.org/abs/2611.22929)

    提出一种投影罚函数梯度追踪框架，使联邦学习在压缩通信条件下仍能保持拜占庭攻击鲁棒性，并给出两种高效算法。

    

    我们研究了在异构数据集上、存在能够上传任意信息的拜占庭攻击者情形下的通信高效联邦学习问题。基于罚函数的方法通过罚函数的有界梯度来约束每个工作节点的影响；然而，一旦上传内容被压缩，这种约束就会失效，因为拜占庭工作节点可以在服务器端积累无界的状态。为了在压缩上传的条件下保持拜占庭鲁棒性，我们提出了一个投影罚函数梯度追踪框架：每个诚实工作节点上传其罚函数梯度的压缩残差，服务器则将每个重构状态投影到已知的罚函数梯度球上。该框架适用于任何收缩型压缩器，并衍生出两种算法：具有一致稀疏化的容错近端梯度算法（FRPG-CS），其共享掩码由新抽取的保留行加以保护；以及惰性聚合的FRPG（L……（摘要原文在此处截断）

    arXiv:2611.22929v1 Announce Type: cross  Abstract: We investigate communication-efficient federated learning over heterogeneous datasets in the presence of Byzantine adversaries capable of uploading arbitrary information. Penalty-based methods bound the influence of each worker through the bounded gradient of a penalty function; however, the bound is lost once the uploads are compressed, since a Byzantine worker can accumulate an unbounded state at the server. To retain the Byzantine resilience under compressed uploads, we propose a projected penalty-gradient tracking framework, where each honest worker uploads a compressed residual of its penalty gradient and the server projects each reconstructed state onto the known penalty-gradient ball. The framework accommodates any contractive compressor and yields two algorithms: fault-resilient proximal gradient with consistent sparsification (FRPG-CS), whose shared mask is protected by freshly drawn reserve rows, and lazily aggregated FRPG (L
    
[^49]: 动作-结果耦合：供应链运营中离线强化学习的投资前可行性审计

    Action-outcome coupling: a pre-investment feasibility audit for offline reinforcement learning in supply chain operations

    [https://arxiv.org/abs/2611.22921](https://arxiv.org/abs/2611.22921)

    本文提出“动作-结果耦合”这一总效应方差指标，作为供应链离线强化学习投资前的可行性审计工具，为任何策略的价值提供与算法无关的货币化上界，并揭示在同一数据上拟合与评估价值模型会随状态单元数增加而导致近乎必然的虚假许可风险。

    

    物流领域的规范性分析计划（如离线强化学习和下一步最佳行动系统）往往在任何人验证现有历史日志能否支撑改进后的决策策略之前就已获得资金投入。我们通过“动作-结果耦合”（action-outcome coupling）将这一投资前可行性问题形式化：这是一个衡量期望结果随已记录动作变化程度的总效应方差指标。耦合为任何策略可实现的价值提供了与算法无关的上界，该上界以每次决策的货币单位表示，同时适用于单步目标和折扣序列目标。我们还分析了一种无效工作流的渐近虚假许可风险：在同一数据上拟合价值模型并评估该模型的贪心策略。虚假许可率主要由执行选择的状态单元数量驱动：当只有一个状态单元时，虚假许可率处于名义检验水平；当状态单元达到几十个时，虚假许可率则接近必然。

    arXiv:2611.22921v1 Announce Type: new  Abstract: Prescriptive analytics initiatives in logistics, such as offline reinforcement learning and next-best-action systems, are often funded before anyone verifies that the available historical logs can support an improved decision policy. We formalise this pre-investment feasibility question via action-outcome coupling: a total-effect variance index measuring how much expected outcomes change with the logged action. Coupling provides an algorithm-independent upper bound on any policy's achievable value, expressed in monetary units per decision for both single-step and discounted sequential objectives. We also analyse the asymptotic false-licensing risk of an invalid workflow: fitting a value model and evaluating the model's greedy policy on the same data. The false-licence rate is driven mainly by the number of state cells over which selection is performed, rising from the nominal test level with one cell to near certainty with a few dozen, w
    
[^50]: 谁为拜占庭鲁棒性买单？当只有少数医院协作时的有效性、数据份额独裁与公平性问题

    Who Pays for Byzantine Robustness? Validity, Data-Share Dictatorships, and Fairness When Only a Few Hospitals Collaborate

    [https://arxiv.org/abs/2611.22904](https://arxiv.org/abs/2611.22904)

    该论文通过对八种拜占庭鲁棒聚合规则在仅有3-6家医院的真实临床联邦学习数据集上进行54,214次训练运行的实证审计，揭示了现有基于大规模合成客户端的基准测试无法反映小规模医疗协作场景下的有效性、数据份额独裁与公平性代价。

    

    arXiv:2611.22904v1 公告类型： new 摘要：临床联邦学习（FL）协作通常只连接三到六家医院，然而拜占庭鲁棒聚合规则的设计和基准测试却是在数十个等规模客户端的合成数据划分上进行的。我们对八种标准聚合规则进行了实证审计：FedAvg、坐标中位数、截尾均值、Krum、Multi-Krum、Bulyan、几何中位数和中心裁剪。我们在来自FLamby基准的三个跨机构医疗数据集上评估了未加权和样本量加权两种变体：Fed-Heart-Disease（n=4家医院）、Fed-TCGA-BRCA（n=6个地区）和Fed-ISIC2019（n=6个中心）。我们在枚举的攻击者联盟（f ∈ {1, 2}）配置下共执行了54,214次训练运行，并采用精确孪生留出对照，将性能下降分解为诚实数据损失与对抗攻击损害两部分。我们报告了三个发现：第一，客户端数量有效性检查无法……（原文摘要在此处截断）

    arXiv:2611.22904v1 Announce Type: new  Abstract: Clinical federated learning (FL) collaborations typically connect only three to six hospitals, yet Byzantine-robust aggregation rules are designed and benchmarked on synthetic partitions across dozens of equal-sized clients. We conduct an empirical audit of eight standard aggregation rules: FedAvg, coordinate-wise median, trimmed mean, Krum, Multi-Krum, Bulyan, geometric median (GeoMed), and Centered Clipping (CC). We evaluate both unweighted and sample-size-weighted variants across three cross-silo medical datasets from the FLamby benchmark: Fed-Heart-Disease ($n = 4$ hospitals), Fed-TCGA-BRCA ($n = 6$ regions), and Fed-ISIC2019 ($n = 6$ centres). We execute 54,214 training runs across enumerated attacker coalitions ($f \in \{1, 2\}$) paired with exact-twin leave-out controls that decompose performance drops into honest data loss versus adversarial attack damage.   We report three findings. First, client-count validity checks fail to ca
    
[^51]: 超越以属性为中心的公平性：全模态大语言模型中统一偏见表征的建模与缓解

    Beyond Attribute-Centric Fairness: Modelling and Mitigating Unified Bias Representation in Omni-modal Large Language Models

    [https://arxiv.org/abs/2611.22889](https://arxiv.org/abs/2611.22889)

    提出了OmniDebias——首批面向全模态大语言模型的去偏方法之一，通过建模并缓解跨属性与模态的统一共享偏见表征，突破了传统以属性为中心的公平性范式的局限。

    

    全模态大语言模型具备联合的多模态理解与生成能力，使其非常适合复杂环境，并有望成为下一代人工智能的骨干模型。尽管取得了这些进展，此类模型中编码的有害刻板印象正以日益多样化的形式显现，带来更广泛、更严重的社会后果。现有的去偏方法通常对每个敏感属性和模态分别施加公平性目标，以最小化与人口群体相关的差异。然而，在全模态场景下，这种以属性为中心的范式难以用不同的公平性目标联合应对多样化的社会偏见，也难以泛化到预定义属性之外。为此，我们提出了OmniDebias，这是首批面向全模态大语言模型公平性的方法之一，它采用以偏见为中心的建模方式，提取并缓解跨属性和模态的共享偏见表征。

    arXiv:2611.22889v1 Announce Type: new  Abstract: Omni-modal large language models (OLLMs) provide joint multimodal understanding and generation capabilities, making them well-suited to complex environments and promising backbones for next-generation AI. Despite these advances, the harmful stereotypes encoded in such models manifest in increasingly diverse forms, leading to broader and more severe societal consequences. Existing debiasing methods typically impose separate fairness objectives for each sensitive attribute and modality to minimise demographic-dependent variations. However, in omni-modal settings, this attribute-centric paradigm struggles to jointly address diverse social biases with distinct fairness objectives and to generalise beyond predefined attributes. To this end, we propose OmniDebias, one of the first methods for promoting fairness in OLLMs, which employs a bias-centric formulation to extract and mitigate a shared bias representation across attributes and modaliti
    
[^52]: 面向基于流的VLA策略的结果平衡PPO与策略诱导流整流方法

    Outcome-Balanced PPO and Policy-Induced Flow Rectification for Flow-Based VLA Policies

    [https://arxiv.org/abs/2611.22878](https://arxiv.org/abs/2611.22878)

    提出一种结合结果平衡PPO（利用任务成功率重加权并重缩放优势）与策略诱导流整流（周期性刷新速度场表示）的强化学习框架，以解决基于流的VLA策略后训练中速度场几何变化和多任务学习信号不均衡两大难题。

    

    强化学习（RL）通过在环境交互中优化期望任务回报，能够对基于流的视觉-语言-动作（VLA）策略进行后训练。然而，这种优化带来两个挑战：策略更新可能改变用于随机转移的分数近似所依赖的速度场几何结构；稀疏的多任务轨迹采样会导致不同任务之间以及不同优势符号之间的学习信号不均衡。我们提出了一个将结果平衡近端策略优化（PPO）与策略诱导流整流相结合的强化学习框架。结果平衡PPO利用任务成功率估计对正优势进行重新加权，并在保持符号不变的前提下对正负优势进行重新缩放。策略诱导整流则周期性地将线性桥流拟合到由当前策略生成的ODE端点对上，从而刷新后续随机优化中所使用的表示。我们评估了该框架……（原文在此处截断）

    arXiv:2611.22878v1 Announce Type: new  Abstract: Reinforcement learning (RL) enables post-training of flow-based vision-language-action (VLA) policies by optimizing expected task return through environment interaction. However, this optimization raises two challenges: policy updates can alter the velocity-field geometry underlying the score approximation used for stochastic transitions, and sparse multi-task rollouts can yield uneven learning signals across tasks and advantage signs. We introduce an RL framework that combines Outcome-Balanced Proximal Policy Optimization (PPO) with policy-induced flow rectification. Outcome-Balanced PPO reweights positive advantages using task success-rate estimates and rescales positive and negative advantages while preserving their signs. Policy-induced rectification periodically fits a linear-bridge flow to ODE endpoint pairs generated by the current policy, refreshing the representation used in subsequent stochastic optimization. We evaluate the fr
    
[^53]: 路径对齐约束等变性

    Path-Aligned Constraint Equivariance

    [https://arxiv.org/abs/2611.22873](https://arxiv.org/abs/2611.22873)

    该论文提出“路径等变性”原理及其实现 PACE：在采用线性路径的条件流匹配中，只要速度场与线性约束相容且初始噪声满足该约束，无散度、无旋度、归一化等物理约束即可沿整条生成轨迹被精确保持，从而以硬约束方式实现超分辨率且不损失小尺度谱结构。

    

    物理场的学习式超分辨率必须满足结构性约束——例如无散度流场（∇·u=0）、无旋度场（∇×u=0）以及逐点分量之和为一的归一化（1^T p=1）——而标准训练目标无法强制这些约束。软惩罚方法通过标量超参数 λ 在保真度与约束满足之间进行权衡，但无法将残差降为零；硬约束回归虽在构造上消除了残差，却会收敛到条件均值，从而丢失小尺度谱结构。我们观察到，在采用线性路径的条件流匹配中，只要速度场与约束相容（Av_θ=0）且初始噪声满足约束（Ax_0=c），线性约束 Ax=c 就会在整条传输轨迹上得以保持；并且沿着轨迹，这两个条件也是必要的。我们将这一原理称为路径等变性原理，并将其实现为 PACE（Path-Aligned Constraint Equivariance，路径对齐约束等变性）……（原文摘要在此处截断）

    arXiv:2611.22873v1 Announce Type: new  Abstract: Learned super-resolution of physical fields must satisfy structural constraints---such as divergence-free flow ($\nabla\cdot u=0$), curl-free fields ($\nabla\times u=0$), and pointwise sum-to-one normalization ($1^\top p=1$)---that standard training objectives do not enforce. Soft penalties trade fidelity against satisfaction via a scalar hyperparameter $\lambda$ but cannot drive the residual to zero; hard-constrained regression eliminates the residual by construction yet converges to the conditional mean, losing small-scale spectral structure. We observe that in Conditional Flow Matching with linear paths, a linear constraint $Ax=c$ is preserved along the entire transport trajectory whenever the velocity field is compatible ($Av_\theta=0$) and the initial noise is consistent ($Ax_0=c$); along the trajectory, the latter two conditions are also necessary. We term this the path-equivariance principle and instantiate it in PACE (Path-Aligne
    
[^54]: 当尾部由其伴生标签挣得：测量长尾多标签分类中的标签依赖性

    When the tail is earned by its companions: measuring label dependence in long-tailed multi-label classification

    [https://arxiv.org/abs/2611.22859](https://arxiv.org/abs/2611.22859)

    该论文提出了可在训练前评估标签依赖性的标签内估计量、读取规则与统计量，揭示稀有标签的召回表现依赖频繁伴生标签而非模型真正学到稀有标签本身的现象。

    

    稀有标签很少单独出现，分类器可以从与其共现的频繁标签（即其“伴生标签”）中恢复出该稀有标签，而无需真正学到这个标签本身。现有的诊断方法都需要已训练好的模型：没有一种方法能在训练前判断某个语料库是否会出现这种失效，也无法揭示标准聚合指标所掩盖的问题。Macro-F1，以及任何基于逐标签混淆计数构建的度量，都不会因标签召回率在“有伴生标签出现”与“无伴生标签出现”两种情形之间的分配方式不同而改变。我们引入了刻画这种划分的标签内估计量，提出了一种读取规则（在该规则下，只有当无伴生召回率上升时，依赖性的下降才会被计入），并提出一个基于训练标签的统计量，它能够在训练之前界定报告召回率可能超过无伴生召回率的程度。在23个语料库、7种训练目标和五类标签相关机制上的实验表明，在定义了该估计量的全部六个语料库中，无伴生召回率均存在不足，长尾部分中高达18.4%的（召回）……

    arXiv:2611.22859v1 Announce Type: cross  Abstract: A rare label seldom occurs alone, and a classifier can recover it from the frequent labels it co-occurs with, its companions, without having learned the label itself. Existing diagnostics of this need a trained model: none says before training whether a corpus admits the failure, or what the standard aggregate hides. Macro-F1, like any measure built from per-label confusion counts, is unchanged by how a label's recall divides between occurrences with and without a companion. We introduce a within-label estimand of that division, a reading rule under which a fall in dependence counts only if companion-free recall rises, and a statistic of training labels that bounds, before training, how far reported recall can exceed companion-free recall. Across 23 corpora, 7 training objectives and five families of label-correlation mechanisms, companion-free recall falls short on all six corpora where the estimand is defined. Up to 18.4% of the tail
    
[^55]: WATERFALL：基于进化奖励构建与自动化学习循环的自适应训练工作流

    WATERFALL: Workflow for Adaptive Training with Evolutionary Reward Formulation and Automated Learning Loops

    [https://arxiv.org/abs/2611.22854](https://arxiv.org/abs/2611.22854)

    WATERFALL提出了一个面向POMDP部分可观测环境的迭代式基于种群的自动化奖励发现工作流，无需目标条件、任务描述或专家演示等先验假设，通过角色条件化生成多样化的程序化奖励候选，以解决稀疏奖励强化学习中的奖励设计瓶颈。

    

    奖励设计仍然是强化学习（RL）中的核心瓶颈，尤其是在稀疏奖励、长时程和部分可观测的环境中，这些环境需要依赖链式推理、记忆以及误导性可供性的消歧。虽然近期的基础模型方法减少了手动奖励工程的工作量，但大多数方法都假设存在目标条件、任务描述、专家演示或真值指标。这些假设在部分可观测环境中是有问题的，因为在这类环境中，与任务相关的目标、可供性和动态必须通过交互来发现，而非预先披露。我们通过引入WATERFALL来正式定义这一问题设定，这是一个在POMDP环境固有的严格可观测性约束下进行自动化奖励发现的迭代式基于种群的工作流。WATERFALL通过以下方式实现这一目标：(i) 通过角色条件化生成合成多样化的程序化奖励候选，(ii) 评估候选……（原文摘要在此处截断）

    arXiv:2611.22854v1 Announce Type: cross  Abstract: Reward design remains a central bottleneck in reinforcement learning (RL), particularly in sparse-reward, long-horizon and partially observable settings requiring dependency chaining, memory, and deceptive affordance disambiguation. While recent foundation-model approaches reduce manual reward engineering, most assume goal-conditioning, task descriptions, expert demonstrations, or ground-truth metrics. These assumptions are problematic in partially observable environments, where task-relevant objectives, affordances, and dynamics must be discovered through interaction rather than disclosed a priori. We formalise this problem setting by introducing WATERFALL, an iterative population-based workflow for automated reward discovery under the strict observability constraints inherent in POMDP environments. WATERFALL achieves this by (i) synthesising diverse programmatic reward candidates via persona-conditioned generation, (ii) evaluating ca
    
[^56]: SilverSpoon-KD：一个通用知识蒸馏工具包

    SilverSpoon-KD: A General-Purpose Toolkit for Knowledge Distillation

    [https://arxiv.org/abs/2611.22842](https://arxiv.org/abs/2611.22842)

    SilverSpoon-KD是一个开源PyTorch知识蒸馏工具包，通过单一API统一了多种蒸馏范式与架构关系，并首创了通用的逐块知识蒸馏实现，在视觉和语言模型等任务上验证了其通用性与高性能。

    

    知识蒸馏将深度学习表示从教师模型迁移到学生模型，通常用于减小模型规模或降低推理成本。然而，知识蒸馏工具的发展未能跟上研究方法的广度：现有工具包通常仅支持有限的技术子集和使用场景。我们提出了SilverSpoon-KD，一个开源的PyTorch工具包，它通过单一API统一了不同工作负载和应用中的多种蒸馏范式与架构关系。此外，SilverSpoon-KD提供了首个通用的逐块知识蒸馏实现，具备独特的可扩展性特性。我们在视觉、掩码语言模型和因果语言模型上验证了其通用性，涵盖同构蒸馏、异构蒸馏和量化蒸馏等多种范式。我们还通过比较其性能（包括吞吐量和GPU内存）对SilverSpoon-KD进行了评估。

    arXiv:2611.22842v1 Announce Type: new  Abstract: Knowledge distillation transfers deep learning representations from a teacher model to a student model, typically reducing model size or inference cost. However, knowledge distillation tooling has not kept pace with the breadth of research methods: existing toolkits typically support only a limited subset of techniques and use cases. We present SilverSpoon-KD, an open-source PyTorch toolkit that unifies different distillation paradigms and architectural relationships across workloads and applications behind a single API. Moreover, SilverSpoon-KD offers the first general-purpose blockwise knowledge distillation implementation with unique scalability features. We validate its versatility across vision, masked language model, and causal language models on tasks spanning homogeneous, heterogeneous, and quantization distillation across paradigms. We also evaluate SilverSpoon-KD by comparing its performance, including throughput and GPU memory
    
[^57]: TabDAG：面向表格特征工程的结构化进化搜索

    TabDAG: Structured Evolutionary Search for Tabular Feature Engineering

    [https://arxiv.org/abs/2611.22828](https://arxiv.org/abs/2611.22828)

    TabDAG 提出将进化搜索与特征工程程序的有向无环图表示相结合，通过受父代模式约束的结构化变异并在评估前进行有效性检查，自动组合生成包含交互、统计编码和聚类邻域特征等多阶段转换的表格特征工程程序。

    

    特征工程可以提升表格数据的预测效果，但有用的转换是组合性的、依赖于数据的，并且受到数据泄露约束的限制。我们提出了 TabDAG，一个将进化搜索与特征工程程序的图表示相结合的框架。节点实现逐行转换或基于数据拟合的转换；边编码原始变量、中间值和最终特征之间的依赖关系。该表示能够在单个可执行程序中支持交互特征、统计编码、基于聚类和邻域的特征以及多阶段组合。变异是受父代特定模式（schema）约束的结构化编辑，并在评估前进行检查。在我们的实验中，没有任何结构无效的程序进入评估阶段，尽管某些提议可能被拒绝，有效的图仍可能在运行时失败。在十个回归和分类数据集以及三个表格模型家族上，进化出的特征（摘要原文在此处被截断）

    arXiv:2611.22828v1 Announce Type: cross  Abstract: Feature engineering can improve tabular prediction, but useful transformations are compositional, data-dependent, and subject to leakage constraints. We present TabDAG, a framework that combines evolutionary search with a graph representation of feature-engineering programs. Nodes implement row-wise or data-fitted transformations; edges encode dependencies among raw variables, intermediate values, and final features. The representation supports interactions, statistical encodings, clustering- and neighborhood-based features, and multi-stage compositions in a single executable program. Mutations are structured edits constrained by a parent-specific schema and checked before evaluation. In our experiments, no structurally invalid program reached evaluation, although proposals could be rejected and valid graphs could still fail at runtime. Across ten regression and classification datasets and three tabular model families, evolved features
    
[^58]: OICycle：用于纵向临床决策中信念状态学习的“观察-推断-干预”循环

    OICycle: Observe, Infer, and Intervene Cycle for Belief-State Learning in Longitudinal Clinical Decision Making

    [https://arxiv.org/abs/2611.22826](https://arxiv.org/abs/2611.22826)

    提出OICycle框架，通过“观察-推断-干预”循环算子直接从回顾性电子健康记录轨迹中学习随机高斯信念状态，无需人工指定离散状态空间和转移模型即可实现纵向临床决策中的患者潜在状态更新。

    

    临床护理是一个闭环过程：临床医生观察部分证据、推断患者的状态、然后进行干预，而每一次干预又决定了接下来能观察到什么。这使得临床决策成为一个部分可观测马尔可夫决策过程（POMDP），其中关于患者潜在状态的信念必须从不规则、多模态且不完整的记录中进行更新。然而，大多数深度临床模型仅从单一时间快照或粗糙聚合的病史进行预测，而基于POMDP的临床方法则需要人工指定的离散状态空间和转移模型，这是真实电子健康记录所无法提供的。我们提出了OICycle（观察、推断与干预循环），它直接从回顾性轨迹中学习随机信念状态。在每个回合中，一个学习到的循环算子将上一次信念的采样与新的多模态观测以及产生这些观测的干预措施融合在一起。由此得到的高斯信念…

    arXiv:2611.22826v1 Announce Type: new  Abstract: Clinical care is a closed loop: clinicians observe partial evidence, infer the patient's state, and intervene, and each intervention determines what is observed next. This makes clinical decision making a partially observable Markov decision process (POMDP), in which a belief over a latent patient state must be updated from irregular, multimodal, and incomplete records. Yet most deep clinical models predict from a single snapshot or from coarsely aggregated history, while POMDP-based clinical methods need hand-specified discrete state spaces and transition models that real electronic health records cannot supply. We introduce OICycle (Observe, Infer, and Intervene Cycle), which learns a stochastic belief state directly from retrospective trajectories. At every episode, a learned recurrent operator fuses a sample of the previous belief with the new multimodal observations and the intervention that produced them. The resulting Gaussian bel
    
[^59]: 虚拟双重神谕：利用一直“旁观全程”的策略实现更快收敛

    Virtual Double Oracle: Faster Convergence by Leveraging Strategies That Were Watching All Along

    [https://arxiv.org/abs/2611.22806](https://arxiv.org/abs/2611.22806)

    提出虚拟双重神谕（VDO），利用双线性恒等式让新发现的策略以“仿佛从一开始就参与学习”的权重初始化，将可利用性收敛速率从 O(√(k ln k / T)) 改进到 O(√(k/T))。

    

    双重神谕与 PSRO 通过由神谕提供的最佳响应迭代扩展策略种群，来近似双人零和博弈中的纳什均衡。在线双重神谕将乘性权重更新（MWU）与种群发现交织进行，获得了 O(√(k ln k / T)) 的收敛速率，其中 T 为学习轮数，k 为两个玩家最终种群规模的较大者。我们发现重新初始化步骤是一个尚未被充分探索的设计选择：在每次种群扩展时，基于 MWU 的方法会将新发现的策略重置为均匀先验。我们提出虚拟双重神谕（VDO），它利用一个双线性恒等式（策略完整的“事后”损失历史可坍缩为其相对于时间平均对手的效用），为每个新策略赋予其从一开始就应累积的 MWU 权重作为初始值。我们证明，VDO 在所发现的种群上取得了 O(√(k/T)) 的可利用性，就好像……

    arXiv:2611.22806v1 Announce Type: new  Abstract: Double Oracle (DO) and PSRO approximate Nash equilibria in two-player zero-sum games by iteratively expanding a strategy population with best responses supplied by an oracle. Online Double Oracle (ODO) interleaves multiplicative weights updates (MWU) with population discovery and obtains an $O(\sqrt{k\ln k/T})$ rate, where $T$ is the number of learning rounds and $k$ is the larger of the two players' final population sizes. We identify the reinitialisation step as an underexplored design choice: at each expansion, MWU-based methods reset newly discovered strategies to a uniform prior. We introduce Virtual Double Oracle (VDO), which exploits a bilinear identity (a strategy's full hindsight loss history collapses to its utility against the time-averaged opponent) to initialise each new strategy with the MWU weight it would have had from the start. We prove that VDO achieves $O(\sqrt{k/T})$ exploitability over the discovered population as i
    
[^60]: PANO：用于求解偏微分方程的并行潜空间神经算子

    PANO: Parallel Latent Space Neural Operator for Solving PDEs

    [https://arxiv.org/abs/2611.22782](https://arxiv.org/abs/2611.22782)

    提出并行潜空间神经算子PANO，通过多个共享参数的并行潜空间将观测数据压缩为多种不同表示，并将各空间的交互结果相加组合，克服了现有方法单一聚合模式导致物理信息不可见的局限，更全面地学习偏微分方程求解中的物理交互。

    

    将观测数据聚合到有限数量的潜空间标记中，并在潜空间中学习物理交互，已成为基于Transformer的神经算子求解涉及大规模观测的偏微分方程（PDE）问题的有效方法。然而，现有方法通常使用单一聚合模式来压缩观测数据，使得该模式未能保留的物理信息对潜空间不可见，从而限制了潜空间内的物理交互。为了解决这一局限，我们提出了并行潜空间神经算子，这是一种用于求解PDE的新型神经算子，它将观测数据之间的物理交互表述为具有共享参数的多个并行潜空间交互结果的相加组合。具体而言，PANO通过独立映射将观测数据转换为多个不同的压缩表示，并学习物理交互。

    arXiv:2611.22782v1 Announce Type: cross  Abstract: Aggregating observations into a limited number of latent tokens and learning physical interactions in the latent space has emerged as an effective approach for Transformer-based neural operators to solve Partial Differential Equation (PDE) problems involving large-scale observations. However, existing methods typically compress observations using a single aggregation pattern, making physical information not retained by that pattern invisible to the latent space and thereby constraining physical interactions within it. To address this limitation, we propose PArallel Latent Space Neural Operator (PANO), a novel neural operator for solving PDEs that formulates physical interactions among observations as an additive combination of interaction results from parallel latent spaces with shared parameters. Specifically, PANO transforms observations into multiple distinct compressed representations through independent mappings, learns physical i
    
[^61]: 运输距离与散度不确定性下的鲁棒执行器-评论家学习

    Robust Actor-Critic Learning under Transport and Divergence Uncertainty

    [https://arxiv.org/abs/2611.22759](https://arxiv.org/abs/2611.22759)

    本文提出一种面向连续状态-动作、标称转移未知的折扣鲁棒马尔可夫决策过程的执行器-评论家方法，通过一次性拟合条件扩散模型，并结合共享目标极小极大准则与指数加权得分匹配，处理由Wasserstein距离、全变差及多种散度定义的转移不确定性。

    

    学习可靠的控制策略需要考虑转移动力学中的不确定性，以及函数估计和动作生成所引入的误差。我们针对具有连续状态与动作且标称转移规律未知的折扣鲁棒马尔可夫决策过程，开发了一种执行器-评论家方法。我们考虑由Wasserstein距离、全变差、Kullback-Leibler散度或Pearson卡方散度所定义的转移不确定性集合。该方法只需对离线转移数据一次性拟合一个条件扩散模型。在随后的每次迭代中，我们使用共享目标的极小极大准则，从当前策略的观测数据中估计鲁棒动作价值函数，然后通过指数加权得分匹配来更新生成式执行器。正则化在保持探索性的同时，为条件分布估计提供了所需的密度正则性。在覆盖性与正则性条件下，我们建立了有限……（摘要原文在此处截断）

    arXiv:2611.22759v1 Announce Type: cross  Abstract: Learning reliable control policies requires accounting for uncertainty in transition dynamics and errors introduced by function estimation and action generation. We develop an actor--critic method for discounted robust Markov decision processes with continuous states and actions and an unknown nominal transition law. We consider transition uncertainty sets defined by Wasserstein distance, total variation, Kullback--Leibler divergence, or Pearson chi-square divergence. A conditional diffusion model is fitted once to offline transitions. At each subsequent iteration, we estimate the robust action-value function from current-policy observations using a shared-target minimax criterion, then update a generative actor through exponentially weighted score matching. Regularization preserves exploration and provides the density regularity needed for conditional distribution estimation. Under coverage and regularity conditions, we establish fini
    
[^62]: LoopPrune：面向循环扩散Transformer的循环剪枝方法

    LoopPrune: Recurrent Pruning for Looped Diffusion Transformers

    [https://arxiv.org/abs/2611.22757](https://arxiv.org/abs/2611.22757)

    LoopPrune提出了一种针对循环扩散Transformer的剪枝与恢复框架，通过循环重要性估计、共享重建和循环位置恢复三大机制，在降低存储与计算成本的同时有效保持模型性能。

    

    循环扩散Transformer通过复用共享模块来深化计算，以更少的参数达到大模型的效果，从而降低存储与部署负担。然而，重复的注意力与MLP计算仍然代价高昂。剪枝通过移除冗余参数或单元来减少存储和计算开销。免训练方法难以解析循环贡献，而基于训练的方法缺乏对影响后续输入与最终输出的特定位置误差的显式约束。我们提出LoopPrune，一个满足上述需求的剪枝与恢复框架。循环重要性估计结合共享移除效应与位置变化来选择共享结构；共享重建汇集循环输入统计量以补偿权重。循环位置恢复在相同的学生状态下比较完整的去噪预测并匹配文本引导，在固定的稠密教师模型监督下更新所有学生参数。

    arXiv:2611.22757v1 Announce Type: new  Abstract: Looped diffusion Transformers reuse shared blocks to deepen computation and match larger models with fewer parameters, reducing storage and deployment burdens. Yet repeated attention and MLP computations remain costly. Pruning reduces storage and computation by removing redundant parameters or units. Training-free methods struggle to resolve recurrent contributions, while training-based methods lack explicit constraints on position-specific errors affecting later inputs and final outputs. We propose LoopPrune, a pruning and recovery framework addressing these requirements. Recurrent importance estimation combines shared-removal effects and position variation to select a shared structure; shared reconstruction pools recurrent input statistics to compensate weights. Loop-position recovery compares complete denoising predictions at identical student states and matches text guidance, updating all student parameters against a fixed dense teac
    
[^63]: 随机多样性窗口：诊断机器人扩散采样中噪声的影响

    Stochastic Diversity Windows: Diagnosing Noise Influence in Robotic Diffusion Sampling

    [https://arxiv.org/abs/2611.22743](https://arxiv.org/abs/2611.22743)

    该论文提出了一种逐时间步的反事实诊断方法来量化扩散采样中随机噪声对最终输出的影响，并据此定义了“随机多样性窗口”（SDWs）以定位随机性影响最强的去噪时间区间，从而指导有针对性的干预。

    

    扩散模型因其能够表达给定任务的多个可行解，在机器人技术与控制领域得到越来越多的应用。然而，扩散模型-采样器在整个去噪过程中随机性所起的作用仍然知之甚少。我们引入了一种逐时间步的反事实诊断方法，用于检验改变采样噪声会如何影响最终输出。在跨越三个机器人基准、多样化架构、领域和条件机制的五种模型-采样器配置上，我们揭示了高度多样的时间敏感性特征：噪声引起的偏差可能在下游被放大、部分收缩，或几乎完全衰减。基于这些特征，我们定义了随机多样性窗口（SDWs）：即能够识别采样器随机性对最终输出具有最强测量影响的紧凑区间。在这些窗口的引导下，我们应用了两种启发式干预：有针对性的随机干预（摘要在此处截断）

    arXiv:2611.22743v1 Announce Type: cross  Abstract: Diffusion models are increasingly adopted in robotics and control due to their capability to express multiple feasible solutions to a given task. However, the role of stochasticity in the diffusion model-sampler across denoising remains poorly understood. We introduce a per-timestep counterfactual diagnostic that tests how changing the sampled noise affects the final output. Across five model-sampler configurations spanning three robotics benchmarks, diverse architectures, domains, and conditioning mechanisms, we reveal highly varied temporal sensitivity profiles, where noise-induced deviations can amplify, partially contract, or undergo near-total attenuation downstream. From these profiles, we define stochastic diversity windows (SDWs): compact intervals that identify where sampler stochasticity has the strongest measured influence on the final output. Guided by these windows, we apply two heuristic interventions: targeted stochastic
    
[^64]: 近端目标：在存在多个正确答案的问题上训练循环精炼器

    Proximal Targets: Training Looped Refiners on Problems with Many Correct Answers

    [https://arxiv.org/abs/2611.22737](https://arxiv.org/abs/2611.22737)

    提出近端目标训练方法，让循环精炼网络的每次迭代朝向离其输入最近的正确答案收敛，在存在多个正确答案的问题上将解题准确率从2.5%大幅提升至98.8%。

    

    循环精炼器（如HRM和TRM）将一个小型网络反复应用于草稿答案，深度监督使每次迭代都朝向每个谜题存储的单一答案进行训练。我们证明，当谜题存在多个正确答案时，这种目标会设定一个任何模型容量都无法消除的损失下限，且损失最小化的迭代会将多个答案混合成无效的网格。我们提出近端目标：训练每次迭代朝向距离其自身输入最近的正确答案，该答案可通过扫描答案列表找到、通过从草稿出发的搜索求解器找到，或在由旋钮设定的质量带内找到；正确答案成为不动点，每次迭代只需比其输入犯更少的错误，循环即可收敛，且训练损失为这一收敛提供了界。使用相同的170万参数网络和5个随机种子，近端目标在具有2-64个答案的数独谜题上达到98.8%的解题率（标准目标仅为2.5%），在超过10,000个答案且无任何答案列表的稀疏谜题上达到93.8%的解题率。

    arXiv:2611.22737v1 Announce Type: new  Abstract: Looped refiners such as HRM and TRM apply one small network repeatedly to a draft answer, and deep supervision trains every pass toward one stored answer per puzzle. We prove that when a puzzle has many correct answers this target sets a loss floor that no model capacity removes, and that the loss-minimising pass blends the answers into an invalid grid. We propose the proximal target: train each pass toward the correct answer nearest to its own input, found by scanning a list of answers, by a search solver started at the draft, or within a band of quality set by a knob; correct answers become fixed points, and a pass only has to make fewer errors than its input has for the loop to converge, which its training loss bounds. With the same 1.7M-parameter network and 5 seeds, the proximal target solves 98.8% of Sudoku puzzles with 2-64 answers (standard target: 2.5%), 93.8% of sparse puzzles with more than 10,000 answers without any list of a
    
[^65]: LoopZip：面向高效循环视觉语言模型的视觉标记压缩

    LoopZip: Visual Token Compression for Efficient Looped Vision-Language Models

    [https://arxiv.org/abs/2611.22728](https://arxiv.org/abs/2611.22728)

    LoopZip是一种无需训练的视觉标记压缩算法，通过在循环主干前构建紧凑候选池并结合跨调用选择，降低循环视觉语言模型处理长视觉序列时的计算与GPU显存开销。

    

    循环视觉语言模型（VLMs）通过重用共享层来加深计算，以更少的参数逼近更大模型的能力，并减轻堆叠独立层带来的存储和部署负担。然而，反复处理长视觉序列仍会产生大量的计算和GPU显存开销。视觉标记压缩通过缩短序列来降低这些开销。现有方法主要是直接选择原始标记，或将选择与聚合相结合。在循环模型中，前者可能忽视各次调用间标记重要性的变化，从而丢弃后续所需的信息；后者则缺乏对各次调用间输出变化的联合考虑，限制了信息的保留。我们提出LoopZip，一种无需训练的视觉标记压缩算法，同时解决上述两个问题。视觉聚合在循环主干之前构建一个紧凑的候选池，跨调用选择则对各次调用的指令注意力进行平均（摘要在此处截断，内容不完整）。

    arXiv:2611.22728v1 Announce Type: cross  Abstract: Looped vision-language models (VLMs) deepen computation by reusing shared layers, approaching larger models' capabilities with fewer parameters and easing the storage and deployment burden of stacking independent layers. However, repeatedly processing long visual sequences still incurs substantial compute and GPU memory costs. Visual token compression reduces these costs by shortening sequences. Existing methods mainly select original tokens directly or combine selection with aggregation. In looped models, the former may overlook changing token importance across calls and discard information needed later; the latter lack joint consideration of output changes across calls, limiting information preservation. We propose LoopZip, a training-free visual token compression algorithm addressing both issues. Visual aggregation builds a compact candidate pool before the looped backbone, and cross-invocation selection averages instruction attenti
    
[^66]: HADVQ：基于Hessian感知的方向向量量化

    HADVQ: Hessian Aware Directional Vector Quantization

    [https://arxiv.org/abs/2611.22721](https://arxiv.org/abs/2611.22721)

    提出了一种面向超低比特LLM量化的向量量化方法HADVQ，通过随机预条件化正则化权重空间、在极坐标下使用基于E8格的结构化码本解耦方向与幅度的量化，并利用代理Hessian以仅0.16比特的额外开销保护高度敏感通道。

    

    超低比特的大语言模型量化是一项具有挑战性的任务，因为激进的压缩会严重扭曲模型参数，并对少数高度敏感的通道造成不成比例的损害。我们提出了一种专为这一场景量身定制的向量量化方法。我们的方法首先采用随机预条件化，对权重空间进行正则化，使其更易于量化。然后我们将权重表示为极坐标形式，并将方向与幅度的量化解耦。为了量化方向，我们利用随机预条件化所诱导的结构，设计了基于 $E_8$ 格的结构化码本，从而获得紧凑而富有表现力的表示。为了进一步减少性能退化，我们使用代理Hessian检测极其重要的通道，并以仅0.16比特的额外成本将其保留。这些组件共同为超低比特场景下的大语言模型提供了一个有效的量化流水线。

    arXiv:2611.22721v1 Announce Type: new  Abstract: Ultra-low-bit quantization of large language models is a challenging task because aggressive compression severely distorts model parameters and disproportionately harms a small number of highly sensitive channels. We present a vector quantization method tailored for this regime. Our approach begins with random preconditioning, which regularizes the weight space and makes it more amenable to quantization. We then represent weights in polar coordinates and decouple the quantization of directions and magnitudes. To quantize directions, we leverage the structure induced by random preconditioning and design a structured codebook based on the $E_8$ lattice, yielding a compact yet expressive representation. To further reduce degradation, we detect extremely important channels using a proxy Hessian and preserve them with an additional cost of only 0.16 bits. Together, these components provide an effective quantization pipeline for LLMs in the ex
    
[^67]: 社交信号基准：视频与音频中共现社交信号的证据加权评估

    Social Signal Bench: Evidence-Weighted Evaluation of Co-occurring Social Signals in Video and Audio

    [https://arxiv.org/abs/2611.22704](https://arxiv.org/abs/2611.22704)

    该论文提出了Social Signal Bench——一个47小时密集标注的视频与音频社交信号识别基准，并引入证据加权F1（EW-F1）评估指标，以更好地处理共现社交信号、语义关系和标注者分歧问题。

    

    识别兴趣和犹豫等社交信号需要解释对话中可能共现的情境依赖行为。现有的情感计算基准很少评估这类共现信号，而传统的多标签F1指标忽略了语义关系和标注者之间的分歧。我们提出了Social Signal Bench，这是一个47小时的密集标注基准，用于评估社交信号识别和参与度，包含两个数据集：Social Signal Bench Video（25小时），涵盖视听记录中的十种社交信号；以及Social Signal Bench Audio（22小时），涵盖纯音频记录中的六种信号。每个数据集将一个公开的在线对话视频训练集（其中10小时由专家标注，其余为伪标签）与一个覆盖四个对话领域的私有专家标注测试集配对。我们还提出了证据加权F1（EW-F1）指标，它对参考（摘要在此处截断）

    arXiv:2611.22704v1 Announce Type: cross  Abstract: Recognizing social signals such as interest and hesitation requires interpreting context-dependent behaviors that may co-occur in conversation. Existing affective computing benchmarks rarely evaluate such co-occurring signals, and conventional multi-label F1 overlooks semantic relationships and annotator disagreement. We introduce Social Signal Bench, a 47-hour densely annotated benchmark for the evaluation of social signal recognition and engagement, comprising two datasets: Social Signal Bench Video (25 hours), covering ten social signals in audiovisual recordings, and Social Signal Bench Audio (22 hours), covering six signals in audio-only recordings. Each dataset pairs a public training set of online conversational video, of which 10 hours are expert-annotated and the remainder pseudo-labeled, with a private expert-labeled test set spanning four conversational domains. We also propose Evidence-Weighted F1 (EW-F1), which weights ref
    
[^68]: CULypso：一种面向车辆自组织网络的集群式联邦遗忘范式

    CULypso: A Clustered Federated Unlearning paradigm for Vehicular Ad-Hoc Networks

    [https://arxiv.org/abs/2611.22682](https://arxiv.org/abs/2611.22682)

    本文提出CULypso，一种专为集群式车辆自组织网络设计的分层联邦遗忘算法，利用网络拓扑与车辆移动性降低通信、资源和时间开销，并能有效遗忘客户端对全局模型更新的影响。

    

    遗忘学习已成为机器学习中一项至关重要的安全与隐私保护解决方案，使网络能够支持客户端依据数据保护法规所享有的数据删除权。而将这样一个并不简单的任务引入边缘的、高度移动且时间敏感的车辆网络，则是一个更为巨大且在很大程度上尚未被探索的挑战。本文论证了分层联邦遗忘方法可以恰当地集成到车辆自组织网络中，通过利用网络拓扑与车辆移动性来降低通信、资源和时间成本，同时有效地遗忘某一客户端对全局模型更新的影响。为实现这一目标，我们提出了CULypso，一种为集群式车辆自组织网络量身定制的近似式、精准式、分层式遗忘算法。该算法受希腊神话中的仙女卡吕普索（Calypso）启发，其名字意为“隐藏知识之人”，CULypso的大写字母即指代……

    arXiv:2611.22682v1 Announce Type: cross  Abstract: Unlearning has emerged as an essential security and privacy-preserving solution in Machine Learning, enabling networks to support their clients' rights for data erasure, as mandated by data protection regulations. Introducing such a non-trivial task to an edge, highly mobile and time-sensitive vehicular network is an even greater, largely unexplored challenge. This article argues that a hierarchical Federated Unlearning approach could be aptly integrated into vehicular ad-hoc networks (VANETs), leveraging topology and movement to reduce communication, resource and time costs, while effectively forgetting a client's influence on the global model updates. In order to achieve this, we propose CULypso, an approximate, surgical, hierarchical unlearning algorithm tailored for clustered VANETs. Inspired by the nymph in Greek mythology, Calypso, whose name translates to 'she who conceals the knowledge', CULypso, with the capital letters refere
    
[^69]: AlignedHPO：高效的代理引导损失超参数优化

    AlignedHPO: Efficient Surrogate-Guided Loss Hyperparameter Optimization

    [https://arxiv.org/abs/2611.22658](https://arxiv.org/abs/2611.22658)

    AlignedHPO提出了一种利用梯度对齐和可微代理验证目标的高效损失超参数优化框架，通过雅可比-向量积与归一化对齐等技术，在保持竞争力性能的同时将超梯度计算速度提升近12倍，成功应用于长尾分类和开集人脸识别任务。

    

    现代深度学习损失函数涉及多个超参数，例如权重、尺度和边距，尽管这些超参数具有可微性，但通常仍通过手动调整或昂贵的黑盒优化方法进行调优。我们提出了AlignedHPO，一个利用梯度对齐和代理验证目标来高效调优可微损失超参数的框架。我们的方法采用雅可比-向量积来降低超梯度计算的时间和内存开销，同时可微的度量代理引导优化朝着特定任务的评估标准进行。我们通过归一化对齐和梯度平滑解决了梯度幅度歧义和随机不稳定问题。在广义Focal Loss和ArcFace上的实验表明，该方法在长尾分类和开集人脸识别任务中均能实现有效适应。与隐式微分方法相比，AlignedHPO取得了具有竞争力或更优的性能，同时超梯度计算速度提升近12倍。

    arXiv:2611.22658v1 Announce Type: cross  Abstract: Modern deep learning losses involve multiple hyperparameters, such as weights, scales, and margins, typically tuned manually or through expensive black-box optimization despite their differentiability. We propose AlignedHPO, an efficient framework for tuning differentiable loss hyperparameters using gradient alignment and surrogate validation objectives. Our method employs Jacobian-vector products to reduce hypergradient computation time and memory, while differentiable metric surrogates guide optimization toward task-specific evaluation criteria. We address gradient magnitude ambiguity and stochastic instability through normalized alignment and gradient smoothing. Experiments with generalized Focal Loss and ArcFace demonstrate effective adaptation across long-tailed classification and open-set face recognition. AlignedHPO achieves competitive or improved performance compared to implicit differentiation, with nearly 12x faster hypergra
    
[^70]: 在线共形预测中硬覆盖约束的代价

    The Price of Hard Coverage Constraints in Online Conformal Prediction

    [https://arxiv.org/abs/2611.22646](https://arxiv.org/abs/2611.22646)

    本文刻画了在线共形预测中强制硬覆盖约束（累积误覆盖受预算限制）不可避免的最小预测集膨胀代价，并证明即使在数据独立同分布、分数分布已知的最理想情况下，这一代价依然无法避免。

    

    共形预测提供了无分布假设的不确定性量化方法，但在序列化部署中，标准的覆盖率保证仍可能允许误覆盖次数随时间出现大幅波动。我们研究了一种在线设置：预测集逐轮产生，且累积误覆盖次数必须在整个部署过程中保持在规定的预算之内。这种硬覆盖率保证可以通过扩大预测集来强制实现，但从根本上需要多大的额外预测集规模仍不清楚。我们在 $T$ 轮预测中刻画了这种不可避免的效率代价。值得注意的是，即使在数据独立同分布、分数分布已知、以及对总体最优预测集拥有完全了解的理想情况下，这种代价依然存在。在将覆盖率与预测集规模相关联的适当正则性条件下，我们确定了任何在线方法为实现硬覆盖保证所必须付出的最小累积超额预测集规模。

    arXiv:2611.22646v1 Announce Type: new  Abstract: Conformal prediction provides distribution-free uncertainty quantification, but in sequential deployment, standard coverage guarantees can still allow substantial fluctuations in the number of miscoverages over time. We study an online setting in which prediction sets are produced one round at a time and the cumulative number of miscoverages must stay within a prescribed budget throughout deployment.Such a hard coverage guarantee can be enforced by enlarging prediction sets, but how much additional prediction-set size is fundamentally necessary remains unclear. We characterize this unavoidable efficiency cost over $T$ prediction rounds. Remarkably, the cost persists even with i.i.d.\ data, a known score distribution, and perfect knowledge of the population-optimal prediction set. Under suitable regularity conditions relating coverage and prediction-set size, we determine the minimum cumulative excess prediction-set size required by any o
    
[^71]: 基于锚定残差流的Sentinel-2亚米级冠层高度合成

    Sub-Meter Canopy Height Synthesis from Sentinel-2 via Anchored Residual Flow

    [https://arxiv.org/abs/2611.22636](https://arxiv.org/abs/2611.22636)

    该论文提出锚定残差流方法，仅利用免费的10米Sentinel-2影像通过确定性回归锚点与残差条件流匹配相结合，合成0.5米分辨率的冠层高度模型，在无需昂贵机载LiDAR的情况下达到2.83米MAE。

    

    冠层高度模型（CHMs）——即植被顶部高度的逐像素地图——对森林碳核算、野火燃料制图以及树木级生态学研究至关重要，然而亚米级CHM依赖需按国家多年周期更新的机载激光雷达（LiDAR）。本研究从免费的10米Sentinel-2影像生成0.5米分辨率的CHM，这是一项20倍的跨模态超分辨率任务，需要光学反射率预测其无法直接观测的几何高度。受感知-失真权衡的启发，我们推导出两个共享条件化的工作点：一个确定性回归器（Anchor）以低失真为目标；一个条件流匹配模型，在学习到的CHM潜空间中基于Anchor的残差进行训练（AR-flow），以感知保真度为目标。模型在横跨美国本土69个生态区的197,000个LiDAR配对瓦片上训练，并在2,530个预留瓦片上与八个基线方法进行评估，Anchor仅使用免费卫星影像即实现了2.83米的MAE……

    arXiv:2611.22636v1 Announce Type: cross  Abstract: Canopy height models (CHMs) - per-pixel maps of vegetation top height - are essential for forest carbon accounting, wildfire fuel mapping, and tree-level ecology, yet sub-meter CHMs require airborne LiDAR refreshed on multi-year national cycles. We produce 0.5 m CHMs from free 10 m Sentinel-2 imagery, a 20$\times$ cross-modal super-resolution task where optical reflectance must predict geometric height it does not directly observe. Motivated by the perception-distortion trade-off, we derive two operating points with shared conditioning. A deterministic regressor (Anchor) targets low distortion; a conditional flow matching model, trained on the Anchor's residual in a learned CHM latent space (AR-flow), targets perceptual fidelity. Trained on 197k LiDAR-paired tiles spanning 69 ecoregions across the contiguous United States and evaluated on 2,530 held-out tiles against eight baselines, the Anchor achieves 2.83 m MAE using only free satel
    
[^72]: 一种用于船舶性能数据缺失值填补的量子支持向量回归框架

    A Quantum Support Vector Regression Framework for Missing Data Imputation in Ship Performance Data

    [https://arxiv.org/abs/2611.22635](https://arxiv.org/abs/2611.22635)

    本研究据称首次将量子支持向量回归（QSVR）应用于船舶性能数据的缺失值填补，通过在巴拿马型散货船真实燃油消耗数据上对比经典SVR与多种量子特征映射下的QSVR，验证了量子机器学习在船舶数据填补领域的可行性。

    

    船舶物联网（Internet of Ships）使船舶能够产生海量数据，但数据缺失值仍然是一个长期存在的问题。据作者所知，该领域现有的填补技术完全依赖经典机器学习。本研究探讨了量子支持向量回归（QSVR）在船舶性能数据缺失值填补中的应用。研究比较了经典支持向量回归（SVR）与基于不同量子特征映射的QSVR模型，在不同数据可用性水平下估计船舶燃油消耗。案例研究采用一艘巴拿马型散货船的真实数据，涵盖六个类别的37个变量。实验包括三项：使用三个输入特征的探索性分析、特征维度从3递增至18的可扩展性分析，以及在选定代表性配置下的对比评估。在所有场景中，经典与量子方法均……（摘要原文在此处截断）

    arXiv:2611.22635v1 Announce Type: cross  Abstract: The Internet of Ships enables vessels to generate vast amounts of data, yet missing values remain a persistent issue. To the best of the authors' knowledge, existing imputation techniques in this domain rely exclusively on classical machine learning. This study investigates quantum support vector regression (QSVR) for imputing missing values in ship performance data. Classical support vector regression and QSVR models based on different quantum feature maps are compared for estimating marine fuel consumption under varying levels of data availability. The case study uses real-world data from a Panamax bulk carrier, covering 37 variables across six categories. Three experiments are conducted: exploratory analysis with three input features, scalability analysis with increasing dimensionality (3 to 18 features), and comparative evaluation under a selected representative configuration. Across all scenarios, both classical and quantum approa
    
[^73]: DAQ：面向脉冲Transformer的决策对齐训练后量化方法

    DAQ: Decision-Aligned Post-Training Quantization for Spiking Transformers

    [https://arxiv.org/abs/2611.22611](https://arxiv.org/abs/2611.22611)

    提出决策对齐训练后量化方法DAQ，通过校准量化器使脉冲Transformer神经元的放电决策与浮点参考模型对齐，从而在权重和膜变量量化的同时有效保持神经元的放电行为。

    

    脉冲Transformer将基于注意力的表征学习与稀疏的脉冲通信相结合，但其权重和膜变量仍然带来较大的存储与计算开销。训练后量化（PTQ）为降低这些开销提供了一种实用的途径，然而在权重和膜变量量化的情况下保持神经元的放电行为仍具有挑战性。本文提出了决策对齐训练后量化（DAQ），该方法通过校准量化器，使神经元的放电决策与浮点参考模型保持一致。我们分析了这些量化器产生的误差如何影响决策边界处的放电行为，以及这些误差如何在循环神经元中随时间步传播。基于这一分析，我们构建了一个优化目标，鼓励量化后的膜变量沿着量化模型自身的状态轨迹保持在参考放电区域内。在多种脉冲Transformer上的实验验证了该方法的有效性。

    arXiv:2611.22611v1 Announce Type: new  Abstract: Spiking transformers combine attention-based representation learning with sparse spike communication, yet their weights and membrane variables still incur substantial storage and computation overhead. Post-training quantization (PTQ) offers a practical way to reduce these overheads, but preserving neuronal firing under weight and membrane quantization remains challenging. In this paper, we propose Decision-Aligned Post-Training Quantization (DAQ), a method that calibrates the quantizers to align neuronal firing decisions with those of a floating-point reference. We analyze how errors from these quantizers affect firing at decision boundaries and how they propagate over time steps in recurrent neurons. Based on this analysis, we formulate an objective that encourages quantized membrane variables to remain within the reference firing regions along the quantized model's own state trajectory. Experiments across multiple spiking transformer a
    
[^74]: 面向生成式机器人技术的任务收敛流匹配

    Task Convergent Flow Matching for Generative Robotics

    [https://arxiv.org/abs/2611.22609](https://arxiv.org/abs/2611.22609)

    提出了一种以任务为中心的推理时优化框架，通过并行物理仿真推演和任务级性能漏斗对流匹配策略进行优化，确保机器人操作动作持续收敛于期望的任务目标。

    

    成功的操作要求被操作对象朝着期望的任务结果不断推进。生成式策略能够产生看似合理的机器人运动，却无法保证这种推进，导致连续的动作片段可能出现停滞或撤销先前对场景的更改。我们提出了一种以任务为中心的推理时优化框架，用于流匹配策略，该框架通过并行的基于物理的仿真推演来评估候选动作片段，并搜索满足任务收敛充分条件的运动。任务级性能漏斗在限制临时倒退的同时，要求每个动作片段实现净进展。我们建立了重复满足漏斗条件可在整个执行过程中产生收敛的充分条件，并使用逐区间Lipschitz界从采样的仿真推演中验证连续时间上的漏斗满足性。为了适应接触丰富的混合动力学，我们使用无导数方法来优化流的源种子……

    arXiv:2611.22609v1 Announce Type: cross  Abstract: Successful manipulation requires the manipulated objects to progress toward a desired task outcome. Generative policies can produce plausible robot motions without ensuring this progress, allowing successive action chunks to stall or undo earlier changes to the scene. We introduce a task-centric inference-time refinement framework for flow matching policies that evaluates candidate action chunks through parallel physics-based rollouts and searches for motions satisfying sufficient conditions for task convergence. Task-level performance funnels bound temporary regressions while requiring net progress over each chunk. We establish sufficient conditions under which repeated funnel satisfaction yields convergence throughout execution, and use interval-wise Lipschitz bounds to verify continuous-time funnel satisfaction from sampled rollouts. To accommodate contact-rich hybrid dynamics, we refine the flow's source seeds using derivative-free
    
[^75]: DLS：基于混沌-模糊残差的双逻辑图神经网络，用于鲁棒链接预测

    DLS: Dual-Logic Graph Neural Networks with Chaos-Fuzzy Residuals for Robust Link Prediction

    [https://arxiv.org/abs/2611.22607](https://arxiv.org/abs/2611.22607)

    提出了DLS归纳式图神经网络框架，通过混沌注入与动态残差构成的“受控不稳定性”机制突破消息传递神经网络的过平滑瓶颈，从而在异配图等挑战性场景下实现更鲁棒的链接预测。

    

    链接预测是图挖掘中的一项基础任务，对于从生物相互作用发现到推荐系统等一系列应用至关重要。尽管消息传递神经网络（MPNN）已取得了最先进的成果，但其从根本上受限于高频信号的无意抑制，因为用于稳定结构学习的递归聚合本质上会导致过平滑，使节点表示在深层网络中变得难以区分。这一限制在异配图中尤为严重，因为在异配图中，结构上的邻近并不意味着特征上的相似。在本文中，我们提出了DLS（双逻辑SAGE），这是一种新颖的归纳式框架，旨在通过“受控不稳定性”机制打破过平滑壁垒。与依赖静态结构启发式的传统方法不同，DLS引入了动态残差流水线。首先，混沌注入模块利用Logistic映射……（原文摘要在此处截断）

    arXiv:2611.22607v1 Announce Type: new  Abstract: Link prediction is a fundamental task in graph mining, critical for applications ranging from biological interaction discovery to recommender systems. While Message Passing Neural Networks (MPNNs) have achieved state-of-the-art results, they are fundamentally limited by the inadvertent suppression of high-frequency signals, as the recursive aggregation that stabilizes structural learning inherently leads to over-smoothing, rendering node representations indistinguishable in deep layers. This limitation is particularly severe in heterophilous graphs, where structural proximity does not imply feature similarity. In this paper, we propose DLS (Dual-Logic SAGE), a novel inductive framework designed to break the over-smoothing barrier via a "controlled instability" mechanism. Unlike conventional methods relying on static structural heuristics, DLS introduces a dynamic residual pipeline. First, a Chaotic Injection Module utilizes the Logistic 
    
[^76]: COTTI：面向促销漂移下鲁棒工业推荐的对比训练与时间感知推理

    COTTI: Contrastive Training and Time-Aware Inference for Robust Industrial Recommendation under Promotional Drift

    [https://arxiv.org/abs/2611.22597](https://arxiv.org/abs/2611.22597)

    COTTI 通过跨周期对比对齐、主动重加权和自适应边界对齐，将历史表征学习与促销事件时刻的分数调整相分离，在促销引起的分布漂移下实现鲁棒的工业推荐，并在飞猪数据上取得了更优的离线 AUC。

    

    促销事件会在不同的时间尺度上改变流量构成、购买行为以及转化概率的量级。我们提出了 COTTI（对比训练与时间感知推理），这是一个将历史表征学习与事件时刻分数调整相分离的工业推荐框架。其中，跨周期对比对齐（CPC）通过自适应一致性权重对同一用户的成对历史进行正则化；主动重加权（RW）通过历史兴趣距离控制促销样本的影响；自适应边界对齐（ABA）将随机预测变异性与时间事件核相结合，在保持预测器参数固定的情况下对 logits 进行调整。我们对加权目标进行了分析，并对 ABA 引起的概率和排序变化给出了理论上界。在飞猪数据上，COTTI 取得了 0.8127 的离线 AUC，相比之下 TEN 为 0.8101，另一对比方法为 0.8123。

    arXiv:2611.22597v1 Announce Type: new  Abstract: Promotional events shift traffic composition, purchase behavior, and the scale of conversion probabilities on different time horizons. We present \textbf{COTTI} (\textbf{CO}ntrastive \textbf{T}raining and \textbf{T}ime-aware \textbf{I}nference), an industrial recommendation framework that separates historical representation learning from event-time score adjustment. Cross-Period Contrastive Alignment (CPC) regularizes paired histories of the same user with adaptive consistency weights. Active Re-weighting (RW) controls the influence of promotional examples through historical-interest distance. Adaptive Boundary Alignment (ABA) combines stochastic prediction variability with a temporal event kernel to adjust logits while keeping predictor parameters fixed. We analyze the weighted objective and bound ABA-induced probability and ranking changes. On Fliggy data, COTTI attains an offline AUC of 0.8127, compared with 0.8101 for TEN and 0.8123 
    
[^77]: 用梯度提议，用损失接受：一种面向像素稀疏鲁棒性的对冲攻击

    Propose with the Gradient, Accept with the Loss: One Hedged Attack for Pixel-Sparse Robustness

    [https://arxiv.org/abs/2611.22596](https://arxiv.org/abs/2611.22596)

    该论文将稀疏对抗攻击视为“提议分布加接受规则”的局部搜索，基于泰勒界推导出五条何时以及如何信任梯度符号、候选像素排名和移除像素排名的规则，提出了一种融合梯度与随机搜索优点的对冲式攻击，从而更准确地评估图像分类器的像素稀疏鲁棒性。

    

    像素稀疏攻击者可以将图像中的少数几个像素更改为任意值。目前对这类攻击的鲁棒性是通过一个集成方法——稀疏AutoAttack（sAA）来衡量的，因为其各个成员会在不同的地方失败：稀疏投影梯度下降（sPGD）信任一阶模型，而完整的像素改变会破坏这一模型；稀疏随机搜索（Sparse-RS）即使在梯度信息可靠的地方也忽略梯度。我们证明这两种攻击本质上都是局部搜索，即一个提议分布加上一个接受规则，并探究这样的搜索能够从梯度中学到什么。泰勒界指出了应该测量什么，而在搜索所访问的状态处进行的测量得出了五条规则。梯度的符号是其在单个像素上最可靠的使用方式，可用来设定新像素的取值。梯度对候选像素的排名应当被宽松地遵循，而它对需要移除像素的排名（通常更为可靠）则可以被更严格地遵循。一个像素不应该被提议……（原文摘要至此截断）

    arXiv:2611.22596v1 Announce Type: cross  Abstract: A pixel-sparse adversary may change a few pixels of an image to any value. Robustness to such attacks is currently measured with an ensemble, sparse-AutoAttack (sAA), because its members fail in different places: sparse projected gradient descent (sPGD) trusts a first-order model that a full pixel change breaks, and sparse random search (Sparse-RS) ignores the gradient even where it is informative. We show that both attacks are local searches, each a proposal distribution plus an acceptance rule, and ask what such a search can learn from the gradient. A Taylor bound says what to measure, and measurements at the states the search visits give five rules. The sign of the gradient is its most reliable use at a single pixel and sets a new pixel's value. Its ranking of candidate pixels should be followed loosely, and its ranking of the pixels to remove, which is often more reliable, can be followed more closely. A pixel should not be propose
    
[^78]: 迷失于漏斗之中：排序与决策系统中由选择引发的校准失效

    Lost in the Funnel: Selection-Induced Miscalibration in Ranking and Decision Systems

    [https://arxiv.org/abs/2611.22595](https://arxiv.org/abs/2611.22595)

    该论文提出了一个基于残差恒等式的逐边界诊断框架，揭示排序与决策系统中在模型边界处校准良好的分数会因下游选择过程（如过滤、重排序、业务规则和拍卖）而失效，并通过全行业案例研究展示了校准在决策时刻的恶化。

    

    现代排序和决策系统将具有不同目标的多个组件组合在一起，在模型边界处校准的分数在整个漏斗流程中——涵盖过滤、重排序、业务规则执行和拍卖——被当作校准后的概率来使用。然而，校准通常只在分数产生的地方进行验证，因此分数在其实际使用处的校准失效往往被忽视。我们探究了校准在何种情况下能在这种选择过程中得以保持、继承性误差与选择效应有何不同，以及在组件边界处完美校准的分数可能出现何种问题。我们通过一个基于残差恒等式的逐边界诊断框架来研究这些问题。我们以评分标准判定的相关性为目标，而非点击率（CTR）等依赖于展示方式的指标，因此校准可以在每个感兴趣的漏斗层级上被测量。我们首先呈现了一项全行业范围的案例研究，将该诊断方法应用于校准在决策时发生恶化的场景。

    arXiv:2611.22595v1 Announce Type: new  Abstract: Modern ranking and decision systems combine components with divergent objectives, and scores calibrated at the model boundary are used as calibrated probabilities throughout the funnel, across filtering, reranking, business-rule enforcement, and auctions. Yet calibration is typically validated only where the score is produced, so its breakdown where the score is used is routinely overlooked. We ask when calibration survives this selection, how inherited error differs from selection effects, and what can fail for a perfectly calibrated score at a component boundary. We study these questions through a boundary-wise diagnostic framework of residual identities. We target rubric-judged relevance rather than presentation-dependent metrics such as CTR, so calibration is measurable at every funnel layer of interest. We first present an industry-wide case study that applies our diagnosis to settings where calibration deteriorates at decision time
    
[^79]: PACT：通过保留正确计划与修复动作实现GUI智能体的计划-动作一致性训练

    PACT: Plan-Action Consistency Training by Preserving Correct Plans and Repairing Actions in GUI Agents

    [https://arxiv.org/abs/2611.22591](https://arxiv.org/abs/2611.22591)

    PACT是一个针对GUI智能体的训练框架，它在结果监督强化学习中通过屏蔽已验证计划token上的负策略梯度来保护正确的规划，同时在推理上下文不变的情况下向参考动作修复执行行为，从而弥合计划与动作之间的执行鸿沟并解决信用分配冲突。

    

    GUI智能体能够在推理中制定出正确的即时计划，却可能发出无法执行该计划的动作，从而产生“执行鸿沟”。这种反复出现的不匹配会损害可靠的任务完成，因为单个执行错误就可能使多步骤工作流偏离正轨。在以结果监督的强化学习中，将负的响应级优势广播到所有生成的token上，可能会在惩罚错误执行的同时连带上有效的规划。为了解决这一信用分配冲突，我们提出了PACT（计划-动作一致性训练），这是一个将参考对齐的计划保护与计划对齐的动作修复相结合的训练框架，且不改变原有的标量动作奖励。通过利用标注的参考动作来识别执行鸿沟，PACT通过屏蔽计划token跨度上的负策略梯度贡献来保护已验证的计划。在保持生成的推理内容不变作为上下文的前提下，它进而倾向于参考动作（原文在此处截断）。

    arXiv:2611.22591v1 Announce Type: new  Abstract: GUI agents can formulate a correct immediate plan in their reasoning yet emit an action that fails to execute it, creating an execution gap. This recurring mismatch undermines reliable task completion, as a single execution error can derail a multi-step workflow. In outcome-supervised reinforcement learning, broadcasting a negative response-level advantage across all generated tokens risks penalizing valid planning together with erroneous execution. To address this credit-assignment conflict, we introduce PACT (Plan-Action Consistency Training), a training framework that combines reference-aligned plan protection and plan-aligned action repair without altering the original scalar action reward. Using annotated reference actions to identify execution gaps, PACT protects the verified plan by masking negative policy-gradient contributions across its token span. Keeping the generated reasoning unchanged as context, it then favors the referen
    
[^80]: ReMaRL：在冻结的扩散语言模型中学习解掩码与再掩码

    ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language Models

    [https://arxiv.org/abs/2611.22576](https://arxiv.org/abs/2611.22576)

    提出ReMaRL，一个仅约30万参数的轻量级策略，在保持扩散语言模型冻结的情况下，通过GRPO强化学习直接优化最终答案正确性，学习决定哪些词元应被解掩码、保留或再掩码，在LLaDA-8B-Instruct上超越了现有的解掩码与再掩码策略。

    

    掩码扩散语言模型并行生成词元，而词元被解掩码的顺序会影响最终答案的准确性。已有若干方法从任务奖励中学习解掩码策略，但这些策略无法对先前的预测进行再掩码。再掩码使模型能够利用新生成的上下文来纠正这些预测。目前，启发式规则同时指导解掩码和再掩码，而有监督的再掩码器则学习检测词元错误，但这两种方法都没有针对最终答案的正确性直接优化这些决策。为填补这一空白，我们提出了ReMaRL（基于强化学习的再掩码），这是一个轻量级策略，用于学习哪些词元应被解掩码、保留或再掩码。我们在保持扩散模型冻结的同时，使用群体相对策略优化（GRPO）从最终答案奖励中训练这个约30万参数的策略。在使用LLaDA-8B-Instruct时，ReMaRL在所比较的解掩码和再掩码策略中取得了最高分数。

    arXiv:2611.22576v1 Announce Type: new  Abstract: Masked diffusion language models generate tokens in parallel, and the order in which tokens are unmasked affects final-answer accuracy. Several methods learn unmasking policies from task rewards, but these policies cannot remask earlier predictions. Remasking lets the model correct those predictions using newly generated context. Heuristic rules guide both unmasking and remasking, while supervised remaskers learn to detect token errors. Neither approach directly optimizes these decisions for final-answer correctness. To address this gap, we introduce ReMaRL (Remasking with Reinforcement Learning), a lightweight policy that learns which tokens to unmask, retain, or remask. We train this roughly 0.3M-parameter policy with Group Relative Policy Optimization (GRPO) from final-answer rewards while keeping the diffusion model frozen. With LLaDA-8B-Instruct, ReMaRL achieves the highest scores among the compared unmasking and remasking policies 
    
[^81]: 基于U(1)对称张量列车的约束组合优化中探索与利用的平衡

    Balancing Exploration and Exploitation in Constrained Combinatorial Optimization with U(1)-Symmetric Tensor Trains

    [https://arxiv.org/abs/2611.22557](https://arxiv.org/abs/2611.22557)

    本文提出一个基于U(1)对称张量网络的约束组合优化通用框架，通过表达能力增强机制和基于图的多样性感知样本选择，在可行点稀少的硬约束问题上有效平衡了探索与利用。

    

    张量列车分解已成为在多种机器学习和组合优化任务中缓解维度灾难的强大工具。最近的研究进展表明，基于张量列车的生成模型能够高效地表示和探索高维离散空间，即使代价函数的结构未知。然而，将硬线性约束融入基于张量列车的优化仍然具有挑战性，因为现有的插值和采样方法在可行点稀少时常常失效。在本工作中，我们基于U(1)对称张量网络，开发了一个用于约束组合优化的通用张量列车框架。我们的方法引入了一种用于构建稀疏对称张量的表达能力增强机制，以及一种基于图的多样性度量来指导训练过程中的样本选择。在分配问题基准测试中，表达能力增强和多样性感知选择……（摘要原文在此处截断）

    arXiv:2611.22557v1 Announce Type: cross  Abstract: Tensor Train (TT) decomposition has emerged as a powerful tool for mitigating the curse of dimensionality in a variety of machine learning and combinatorial optimization tasks. Recent advances show that TT-based generative models can efficiently represent and explore high-dimensional discrete spaces, even when the cost function structure is unknown. However, incorporating hard linear constraints into TT-based optimization remains challenging, as existing interpolation and sampling approaches often fail when feasible points are rare. In this work, we develop a general TT-based framework for constrained combinatorial optimization built upon U(1)-symmetric tensor networks. Our method introduces an expressibility-enhancement mechanism for constructing sparse symmetric tensors and a graph-based diversity measure that guides sample selection during training. On assignment benchmarks, expressibility enhancement and diversity-aware selection i
    
[^82]: 用于稀疏流场重构的物理可行生成流匹配

    Physically Admissible Generative Flow Matching for Sparse Flow Reconstruction

    [https://arxiv.org/abs/2611.22549](https://arxiv.org/abs/2611.22549)

    该论文提出可行子空间一致性框架，通过切空间匹配将整个生成ODE轨迹约束在离散守恒流形上并保持动能度量，使稀疏流场重构中每个后验样本都严格满足不可压缩性与边界条件等物理可行性要求。

    

    从稀疏、局部的观测数据重构完整的流体速度场是一个严重的病态逆问题：大部分空间区域未被观测、离散守恒定律必须严格成立、且预测不确定性必须可靠。现有的生成模型通常仅在最终输出端施加物理约束或采用软惩罚方式，导致中间生成轨迹不受约束。我们提出了可行子空间一致性，这是一个将物理可行性作为连续生成动力学不变量来保持的框架：切空间匹配将整个常微分方程（ODE）轨迹限制在离散守恒流形上，同时质量正交归一坐标系保持动能度量。稀疏重构随后被转化为在冻结的FSC先验潜空间中的摊销后验输运，确保每个后验样本都继承不可压缩性和边界条件……

    arXiv:2611.22549v1 Announce Type: new  Abstract: Reconstructing full fluid velocity fields from sparse, localized observations is a severely ill-posed inverse problem: most of the spatial domain remains unobserved, discrete conservation laws must hold strictly, and predictive uncertainty must be reliable. Existing generative models typically enforce physical constraints only at the terminal output or via soft penalties, leaving intermediate generative trajectories unconstrained. We introduce Feasible Subspace Consistency (FSC), a framework that preserves physical admissibility as an invariant of continuous generative dynamics: tangent-space matching confines the entire ODE trajectory to the discrete conservation manifold, while mass-orthonormal coordinates preserve the kinetic-energy metric. Sparse reconstruction then reduces to amortized posterior transport in the latent space of a frozen FSC prior, ensuring every posterior sample inherits incompressibility and boundary conditions wit
    
[^83]: 高斯与α-稳定MIMO信道下联合信号-信道估计的时序扩散模型

    Temporal Diffusion Models for Joint Signal-Channel Estimation under Gaussian and $\alpha$-Stable MIMO Channels

    [https://arxiv.org/abs/2611.22548](https://arxiv.org/abs/2611.22548)

    该论文提出一种时序扩散模型，在高斯与α-稳定MIMO信道下实现联合信号-信道估计，通过观测似然引导反向扩散过程，并使单个训练好的扩散模型无需重新训练即可复用于位置-尺度分布族。

    

    扩散模型已成为学习复杂数据分布和求解逆问题的强大工具。然而，大多数扩散模型依赖于高斯分布，而高斯分布并不能准确刻画重尾数据。在本工作中，我们研究了在高斯信道和对称α-稳定衰落信道下基于扩散的重构方法，同时探索了利用信道中可能存在的时间相关性的好处。我们考虑了两种变体：基于导频的纯信道估计，以及面向MIMO系统的图像-信道联合恢复。我们的方法基于多种标准扩散模型和基于分数的扩散模型，以分别适用于高斯分布和α-稳定分布。我们通过使用观测似然来引导反向扩散过程对这些模型进行改进，并展示了单个训练好的扩散模型无需重新训练即可在位置-尺度分布族中复用。在盲MIMO恢复中，将单独的扩散（摘要在此处截断）

    arXiv:2611.22548v1 Announce Type: cross  Abstract: Diffusion models have emerged as powerful tools for learning complex data distributions and solving inverse problems. However, most diffusion models rely on the Gaussian distribution, which does not accurately represent heavy-tailed data. In this work, we study diffusion-based reconstruction under Gaussian and symmetric $\alpha$-stable fading channels, while also exploring the benefits of using possible temporal dependence in the channel. We consider two variants: pilot-based channel-only estimation and joint image-channel recovery for MIMO systems. Our approach is based on several standard and score-based diffusion models for both Gaussian and $\alpha$-stable distributions. We refine these models by guiding the reverse diffusion process using the observation likelihood and show how a single trained diffusion model can be reused for the location-scale family of distributions without retraining. In blind MIMO recovery, separate diffusio
    
[^84]: 在从未见过的动作上测试世界模型：零售业中的留出计划、预测准确度与价格规则

    Testing World Models on Actions They Have Never Seen: Held-Out Plans, Forecast Accuracy and Price Rules in Retail

    [https://arxiv.org/abs/2611.22537](https://arxiv.org/abs/2611.22537)

    该论文提出了一种检验动作条件世界模型对从未见过的动作泛化能力的测试方法，并在零售定价场景的2,853个模型中发现：预测准确度仅能可靠地区分不同类型的模型，却无法区分同一世界模型的不同版本（在高需求周甚至出现反转），且JEPA预训练等学习动作效果的训练技术并未降低模型在新计划上的误差。

    

    世界模型被信任来回答“如果……会怎样”的问题：即它的世界如何响应一个它从未见过的动作。我们为任何以动作为条件的世界模型提供了一种检验这种信任的测试：通过设计将动作从训练数据中留出，并对照已知答案进行评分；检验预测准确度能否预测模型在这些动作上的表现；以及将相同的结构应用于更简单的模型。我们在零售场景中运行该测试，其中动作是价格和促销计划，共涉及2,853个拟合模型。首先，预测准确度是在不同类型模型之间进行比较的合理依据（与新计划上的误差的秩相关性为0.32至0.66），但对于同一世界模型的不同版本之间则是一个糟糕的依据：在商家将计划安排到高需求周的情况下，这种相关性会发生反转，低至-0.69。其次，训练世界模型学习动作的效果（JEPA预训练、潜在一致性检查、计划日期处的适配）并不会降低其在新计划上的误差。第三，在生成的世界中，记录的常规价格和三个p……（摘要在此处截断）

    arXiv:2611.22537v1 Announce Type: cross  Abstract: A world model is trusted to answer "what if": how its world responds to an action it has never seen. We give a test of that trust for any action-conditioned world model: actions held out of training by design and scored against a known answer; whether forecast accuracy predicts accuracy on those actions; and the same structure given to simpler models. We run it in retail, where the action is a price and promotion plan, across 2,853 fitted models. First, forecast accuracy is a fair guide between kinds of model (rank correlation with error on new plans 0.32 to 0.66) and a poor guide between versions of one world model: where merchants plan into high-demand weeks the link reverses, down to -0.69. Second, training the world model to learn the effect of actions (JEPA pretraining, a latent consistency check, adaptation at the plan date) leaves its error on new plans unchanged. Third, in generated worlds the recorded regular price and three p
    
[^85]: 正确的标签，错误的理由：面向多标签可解释性的带符号特征归因

    Right Labels, Wrong Reasons: Signed Feature Attribution for Multi-Label Explainability

    [https://arxiv.org/abs/2611.22515](https://arxiv.org/abs/2611.22515)

    提出FLEX，一种基于多标签Fisher线性判别分析的全局代理解释器，能在单一连贯坐标系中为每个标签生成高保真度的带符号特征归因，尤其在标签多且正例稀少时表现优异。

    

    当一个多标签分类器给出正确的标签时，它可能是出于错误的理由做到的。弥散的特征组合可能因分类器家族而异，且缺乏占主导地位的特征。本文提出了FLEX（Fisher LDA解释器），一种全局代理模型，它在完整的二值标签矩阵上拟合单个直接的多标签Fisher线性判别分析（LDA），从而在单一连贯的坐标系中为每个标签生成带符号的特征归因列表。在所评估的基准数据集和分类器家族上，FLEX以持续较高的保真度复现了黑盒模型的预测。在大多数数据集上，跨标签池化协方差的做法优于按标签分别处理的方法，且在标签数量多、正例稀少的场景下优势最为显著。FLEX的保真度-复杂度帕累托曲线呈单调递增，为使用者提供了一条可预测的路径：随着使用更多特征，即可获得保真度更高的解释，其样本内保真度……

    arXiv:2611.22515v1 Announce Type: cross  Abstract: When a multi-label classifier assigns the right labels, it may do so for the wrong reasons. Diffuse feature combinations may vary across classifier families and lack a dominant feature. This paper proposes FLEX (Fisher LDA Explainer), a global surrogate that fits a single direct multi-label Fisher LDA over the full binary label matrix, producing signed feature attribution lists for each label in a single coherent coordinate system. Across the evaluated benchmarks and classifier families, FLEX reproduces the black box's predictions with consistently high fidelity. Pooling covariance across labels outperforms a per-label alternative on most datasets, with the advantage growing largest where labels are numerous and positive examples are scarce. FLEX's fidelity-complexity Pareto curve is monotonically increasing, giving practitioners a predictable path to higher-fidelity explanations as more features are used, and its in-sample fidelity ma
    
[^86]: 通过全量微调与参数高效微调实现3D腹部视觉-语言模型面向机构特异性肝细胞癌分类的高效自监督适应

    Efficient Self-Supervised Adaptation of 3D Abdominal Vision-Language Model for Institution-Specific HCC Classification via Full Fine-Tuning and PEFT

    [https://arxiv.org/abs/2611.22504](https://arxiv.org/abs/2611.22504)

    该研究首次将预训练3D腹部医学视觉-语言模型通过全量微调与PEFT两种策略成功适应到跨机构的肝细胞癌CT分类任务，证明即使面对领域偏移和数据受限，两种方法均能取得较强性能（全量微调AUC 0.82，PEFT AUC 0.79）。

    

    医学视觉-语言模型（VLMs）能够捕获图像与文本之间的跨模态关系，但其在机构特异性临床任务上的适应仍缺乏充分探索。在本研究中，我们使用来自另一机构的成对腹部CT扫描和放射学报告（其采集特性与模型原始预训练语料库不同）对一个预训练的3D医学VLM进行微调，以实现肝细胞癌（HCC）分类。我们比较了两种适应策略：全量微调与参数高效微调（PEFT），其动机在于PEFT被普遍用于降低计算成本并在数据受限条件下实现模型适应。尽管存在跨机构的领域偏移，两种方法均取得了较强的下游HCC分类性能：PEFT达到AUC 0.79、F1 0.70，全量微调达到AUC 0.82、F1 0.71。尽管这些结果仍低于（原文截断）……

    arXiv:2611.22504v1 Announce Type: cross  Abstract: Medical vision-language models (VLMs) capture cross-modal relationships between image and text, yet their adaptation to institution-specific clinical tasks remains underexplored. In this study, we fine-tuned a pretrained 3D medical VLM for hepatocellular carcinoma (HCC) classification using paired abdominal CT scans and radiology reports from a different institution with acquisition characteristics that differ from the model's original pretraining corpus. We compared two adaptation strategies: full fine-tuning and parameter-efficient fine-tuning (PEFT), motivated by the common use of PEFT to reduce computational cost and enable adaptation under limited-data constraints. Both approaches achieve strong downstream HCC classification performance despite the cross-institutional domain shift, with PEFT reaching an AUC of 0.79 and F1 of 0.70, and full fine-tuning achieving an AUC of 0.82 and F1 of 0.71. Although these results remain below the
    
[^87]: 制造差分稀疏性：面向高效世界模型规划的规划器-硬件协同设计

    Manufacturing Differential Sparsity: Planner-Hardware Co-Design for Efficient World Model Planning

    [https://arxiv.org/abs/2611.22483](https://arxiv.org/abs/2611.22483)

    该论文发现量化世界模型预测器中相近动作会产生差分位级稀疏性，并提出规划器-硬件协同设计，通过参考森林采样主动“制造”稀疏性，从而大幅加速具身智能体的MPC规划。

    

    arXiv:2611.22483v1 公告类型：cross 摘要：JEPA风格的潜在世界模型使具身智能体能够通过模型预测控制（MPC）向新目标进行规划，但对于现实生活中的具身控制而言它们仍然太慢：基于采样的MPC在每次规划中需要重复执行数万次预测器推理，即使在高端GPU上，单个规划步骤也可能耗时数十秒。我们观察到这种重复推理工作负载中存在显著的潜在局部性：预测器将相近的动作映射到相近的潜在表示，而在量化预测器内部，这种相近性呈现出一种硬件可利用的形式——差分位级稀疏性。与输入流（其时间相似性是给定的）不同，规划器自身决定将哪些候选一起评估，因此规划器可以选择自己的参考，并通过参考森林采样制造更多稀疏性：它将每组采样动作序列耦合到共享根节点上，同时保持有竞争力的规划质量。我们基于这些洞察构建了加速器……

    arXiv:2611.22483v1 Announce Type: cross  Abstract: JEPA-style latent world models let embodied agents plan toward new goals with model predictive control, but they are still too slow for real-life embodied control: sampling-based MPC repeats predictor inference tens of thousands of times per plan, and even on high-end GPUs one planning step can take tens of seconds. We observe substantial latent locality in this repeated-inference workload: the predictor maps close actions to close latents, and inside a quantized predictor this closeness takes a form that hardware can exploit, differential bit-level sparsity. Unlike an input stream, whose temporal similarity is given, a planner itself decides which candidates it evaluates together, so the planner can choose its own references and, with reference forest sampling, manufacture more sparsity: it couples the sampled action sequences of each group to a shared root, with competitive planning quality. We build the accelerator on the insights o
    
[^88]: 通过记忆摆脱潜在陷阱：面向世界模型规划的无训练情景核排斥方法

    Escaping Latent Traps by Remembering: Training-Free Episodic Kernel Repulsion for World-Model Planning

    [https://arxiv.org/abs/2611.22470](https://arxiv.org/abs/2611.22470)

    提出一种无需训练的情景核排斥代价项，利用智能体已访问潜在状态的新颖性门控记忆惩罚候选计划，帮助世界模型规划摆脱局部极小值“潜在陷阱”，在两个JEPA世界模型和四个基准上将规划成功率显著提升。

    

    潜在世界模型通过最小化预测潜变量与目标潜变量之间的距离来向目标图像进行规划。然而这种距离并非测地距离：它在墙后、角落和死胡同中存在局部极小值，即“潜在陷阱”，而无记忆的滚动时域重规划会使智能体持续被困其中，不断重访远离目标的状态。我们提出情景核排斥，这是一种无需训练的代价项，通过针对智能体实际访问过的潜在变量所构成的新颖性门控记忆库，利用 top-k 高斯核对候选计划进行惩罚，并根据当前基础代价进行重新缩放，使其强度在不同模型之间具有可比性。在两个 JEPA 世界模型和四个基准测试上，在相同的模型、规划器和预算条件下，EKR 将平均成功率从 31.8% 提升至 46.7%（LeWM），从 54.7% 提升至 64.0%（H-JEPA）。相同记忆下的反事实实验表明，在大多数设置中，EKR 会将规划推向远离已访问状态的方向，在导航任务中往往以更高的基础代价为代价；卡住的情形……

    arXiv:2611.22470v1 Announce Type: new  Abstract: Latent world models plan towards a goal image by minimising the distance between predicted and goal latents. This distance is not geodesic: it has local minima, latent traps, behind walls, at corners and in dead ends, and memoryless receding-horizon replanning keeps the agent in them, revisiting states far from the goal. We propose Episodic Kernel Repulsion (EKR), a training-free cost term that penalises candidate plans by a top-$k$ Gaussian kernel to a novelty-gated memory of the latents the agent has actually visited, rescaled by the current base cost so that its strength is comparable across models. Across two JEPA world models and four benchmarks, EKR raises average success from 31.8% to 46.7% (LeWM) and from 54.7% to 64.0% (H-JEPA) with the same model, planner and budget. Same-memory counterfactuals show that, in most settings, EKR pushes plans away from visited states, in navigation, often at a higher base cost; stuck episodes and 
    
[^89]: 面向潜在生成模型的样本对齐解码器

    Sample-Aligned Decoders for Latent Generative Models

    [https://arxiv.org/abs/2611.22458](https://arxiv.org/abs/2611.22458)

    提出样本对齐解码器（SAD），通过使解码器适应生成潜在表示的分布而非仅重建编码器潜在表示，来阻止生成器的残差误差被渲染进最终图像，并解决了无配对监督与潜在空间漂移两大挑战。

    

    潜在生成模型通过解码生成的潜在表示来合成图像，但其解码器仅被训练用于重建编码器的潜在表示，在测试时从未见过它实际解码的潜在表示，因此生成器的残差误差会被直接渲染到最终图像中。现有的补救措施要么重新训练生成器，要么在每个采样步骤增加开销，要么使解码器对扰动的真实潜在表示具有鲁棒性——后者假设生成器的误差类似于此类扰动。我们发现事实并非如此：生成的潜在表示与真实潜在表示同样紧密地位于编码器潜在流形上，却遵循着不同的分布。因此，解码器应当适应生成潜在表示的分布，这带来了两个挑战：（1）无配对监督，因为生成的潜在表示没有对应的真实图像；（2）潜在空间漂移，因为从配对目标中解放出来的解码器会遗忘它必须解码的潜在空间。我们提出了样本对齐解码器，一种……（摘要在此处被截断）

    arXiv:2611.22458v1 Announce Type: cross  Abstract: Latent generative models synthesize images by decoding generated latents, yet their decoder is trained only to reconstruct encoder latents and never sees the latents it decodes at test time, so the residual errors of the generator are rendered directly into the final image. Existing remedies retrain the generator, add cost at every sampling step, or make the decoder robust to perturbed real latents, assuming that generator errors resemble such perturbations. We find that they do not. Generated latents lie on the encoder latent manifold as closely as real ones, yet follow a different distribution. The decoder should therefore be adapted to the distribution of generated latents, which raises two challenges: (1) unpaired supervision, since a generated latent has no ground-truth image, and (2) latent-space drift, since a decoder freed from paired targets forgets the latent space it must decode. We propose Sample-Aligned Decoders (SAD), a p
    
[^90]: PACD：面向全模态大语言模型跨模态幻觉缓解的溯源感知共识解码

    PACD: Provenance-Aware Consensus Decoding for Mitigating Cross-Modal Hallucinations in Omnimodal Large Language Models

    [https://arxiv.org/abs/2611.22456](https://arxiv.org/abs/2611.22456)

    提出了一种免训练的溯源感知共识解码框架PACD，根据模态分支间是否达成共识采用两条互补路径（如相关性引导的候选仲裁），有效缓解全模态大语言模型中的跨模态幻觉问题。

    

    全模态大语言模型整合音频、视频和文本进行多模态推理，但仍然容易受到跨模态幻觉的影响，即一种模态不当地影响对另一种模态的预测。现有的免训练解码方法通过模态扰动或基于相关性的加权来缓解这些错误。然而，由于融合过程中无关模态线索的干扰以及难以检测共享错误，这些方法可能会抑制正确的单模态预测，并使错误的共识持续存在。为了解决这些局限性，我们提出了溯源感知共识解码（PACD），这是一个用于缓解跨模态幻觉的免训练框架。PACD根据各分支是否达成共识，包含两条互补的路径。对于非共识情况，相关性引导的候选仲裁会识别与任务相关的单模态证据，并在各预测之间选择性地进行仲裁

    arXiv:2611.22456v1 Announce Type: new  Abstract: Omnimodal large language models (Omni-LLMs) integrate audio, video, and text for multimodal reasoning, yet remain vulnerable to cross-modal hallucinations, where one modality improperly influences predictions about another. Existing training-free decoding methods mitigate these errors through modality perturbation or relevance-based weighting. However, due to interference from irrelevant modality cues during fusion and difficulty detecting shared errors, these methods may suppress correct unimodal predictions and allow erroneous consensus to persist. To address these limitations, we propose Provenance-Aware Consensus Decoding (PACD), a training-free framework for mitigating cross-modal hallucinations. PACD comprises two complementary paths based on whether the branches reach consensus. For non-consensus cases, Relevance-Guided Candidate Arbitration identifies task-relevant unimodal evidence and selectively arbitrates between predictions 
    
[^91]: 训练集中有多少标签是错的？通过梯度分配审计已拟合模型

    How Many Training Labels Are Wrong? Auditing a Fitted Model through Gradient Allocation

    [https://arxiv.org/abs/2611.22449](https://arxiv.org/abs/2611.22449)

    该论文提出一种无需干净数据、保留集、阈值或再训练的方法，通过比较单个训练模型的逐样本梯度分配与各类平均分配来识别错误标签并估计训练集噪声率。

    

    训练集中有多少标签是错误的，具体是哪些？通常的答案要么读取一个保留集，要么读取模型的输出。基于保留集的读数取决于保留集的抽样方式，而基于输出的读数在模型拟合了错误标签之后就会失效，因为已拟合错误标签的损失很小，且预测与该错误标签一致。我们从单个训练好的模型的逐样本梯度中读取答案。每个样本连同其标签一起，会要求网络以某种特定方式发生变化。我们将梯度幅值归一化为一种梯度分配，并将其与每个类别的平均分配进行比较。当把某个样本重新标注为其最近的其他类别时，如果其梯度分配比在原标签下匹配得更好，该样本就被标记为可疑，而被标记样本所占的比例即为噪声率的读数。该规则无需阈值，也不需要干净数据、保留集或进一步的训练。在模型完成拟合之后……（原文摘要在此处截断）

    arXiv:2611.22449v1 Announce Type: new  Abstract: How many labels in a training set are wrong, and which ones? The usual answers read a held-out set or the model's outputs. A held-out reading depends on how the held-out set was drawn, and an output-based reading fails once the model has fitted its wrong labels, because the loss of a fitted wrong label is small and the prediction agrees with it. We read the answer from the per-example gradients of a single trained model. Each example, together with its label, asks the network to change in a particular way. We normalise the magnitudes of its gradient into a gradient allocation and compare it with the mean allocation of every class. An example is flagged when relabelling it as its nearest other class makes its allocation fit better than under its own label, and the share of flagged examples is the reading of the noise rate. The rule has no threshold and needs no clean data, no held-out set and no further training. After the model has fitte
    
[^92]: 高效机器学习原子间势

    Efficient Machine Learning Interatomic Potentials

    [https://arxiv.org/abs/2611.22444](https://arxiv.org/abs/2611.22444)

    该综述以原子簇展开（ACE/GRACE）这一统一数学框架为基础，系统性地梳理了高效机器学习原子间势的设计策略，并揭示各类MLIP架构的表面多样性主要源于非线性处理、层耦合与优化方式的不同，而非模型可表示的相互作用范围。

    

    机器学习原子间势（MLIPs）能够以量子力学参考计算成本的一小部分达到接近其精度的水平。效率是MLIPs的目的所在，然而关于高效MLIPs的策略尚未被系统地综述。本综述以效率为指导原则。为了在统一的基础上比较各种MLIPs，我们将其表示在完备基组展开中，即原子簇展开（ACE）及其图推广形式（GRACE）。该展开是正则的，由单粒子函数的乘积构建而成。经典原子间势（如嵌入原子方法）以及许多现代MLIPs（包括等变变换器）都可以在这个展开框架中表示。因此，MLIP架构的大部分表面多样性反映的是如何处理非线性、层耦合和优化问题，而非模型能够表示哪些相互作用。我们展示了MLIP的计算成本如何变为线性……

    arXiv:2611.22444v1 Announce Type: cross  Abstract: Machine learning interatomic potentials (MLIPs) reach an accuracy close to quantum mechanical reference calculations at a small fraction of the cost. Efficiency is the purpose of MLIPs, yet strategies for efficient MLIPs have not been reviewed systematically. This review takes efficiency as its guideline. To compare MLIPs on a common footing, we represent them in a complete basis expansion, the Atomic Cluster Expansion (ACE) and its graph generalization (GRACE). The expansion is canonical, built from products of single-particle functions. Classical interatomic potentials, such as the embedded-atom method, and many modern MLIPs, including equivariant transformers, can be represented within this expansion. Much of the apparent diversity of MLIP architectures therefore reflects how non-linearity, layer coupling and optimization are handled rather than which interactions the models can represent. We show how the cost of an MLIP becomes lin
    
[^93]: 更新中的“眼睛”？学习奖励在多模态推理中的落点

    Eyes in the Update? Learning Where Rewards Land in Multimodal Reasoning

    [https://arxiv.org/abs/2611.22438](https://arxiv.org/abs/2611.22438)

    Gaze-RL 利用模型内部跨模态几何与局部预测不确定性构建潜在“奖励落点场”，无需外部标注器或过程标签，即可将轨迹级优势奖励重新分配到响应中视觉证据实际所在的位置，从而提升多模态强化学习的 token 敏感性。

    

    多模态后训练已从宽泛的指令微调转向面向图像与视频推理的强化学习，其中结果奖励能够提升最终答案的准确性。然而，这些奖励被分配给完整生成的轨迹，而视觉证据仅在响应内部稀疏且不均匀的位置被使用。近期工作已开始通过接地信号或外部 token 级加权使多模态强化学习对 token 更加敏感。我们提出一个更具体的问题：能否从模型自身的多模态计算中推断同一轨迹级奖励的分配方式，而不依赖辅助标注器、感知模块或过程标签？我们提出 Gaze-RL，一种由表示驱动的策略优化方法，通过一个潜在的“奖励落点场”重新分配完成级优势。该落点场由内部跨模态几何结构与局部预测不确定性计算得到，为（视觉证据所在的 token）分配更大的……（摘要在此处截断）

    arXiv:2611.22438v1 Announce Type: cross  Abstract: Multimodal post-training has moved from broad instruction tuning toward reinforcement learning for image and video reasoning, where outcome rewards can improve final-answer accuracy. Yet these rewards are assigned to complete generated trajectories, while visual evidence is used only at sparse and uneven positions inside the response. Recent work has begun to make multimodal RL more token-sensitive through grounding signals or external token-level weighting. We ask a more specific question: can the allocation of the same trajectory-level reward be inferred from the model's own multimodal computation, rather than from an auxiliary annotator, perception module, or process label? We propose Gaze-RL, a representation-driven policy optimization method that redistributes completion-level advantage through a latent reward landing field. The field is computed from internal cross-modal geometry and local predictive uncertainty, assigning larger
    
[^94]: 编码、可解码性与使用：一个用于基准测量审计的五层框架——来自无人机意图识别的证据

    Encoding, Decodability, and Use: A Five-Layer Framework for Benchmark Measurement Audits with Evidence from UAV Intent Recognition

    [https://arxiv.org/abs/2611.22434](https://arxiv.org/abs/2611.22434)

    本文提出一个将基准测量审计分解为观测、表示、解码、使用与行为五个层级的框架，并通过无人机意图识别基准 AeroIntent 的实证表明，基准高分并不等于模型真正理解了所测量的构念。

    

    基准准确率通常被解读为模型理解了基准所测量构念的证据。我们在 AeroIntent 上提出并实例化了一个五层框架，将这种解读分解为观测、表示、解码、使用和行为。该框架将编码、可解码性和使用区分为不同的层级。在实例化中，观测层在 20 个 Drone 程序中的 7 个上不可靠，并将径向速度的直接视觉相关性下限锁定在 0.26% 的尺度变化上。表示层揭示了两个仅具名义信号的任务设计 RGB 特征：RAFT 光流径向偏差达到 r=+0.257（p=0.030），边界框光流散度达到 r=+0.203（p=0.029）。没有任何多特征读出在折平均测试相关性上超过聚类的最小可检测效应（MDE），且最大的正的全局样本外相关性仅为 r=+0.080。解码层将 MDE 设定在……（原文摘要在此处被截断）

    arXiv:2611.22434v1 Announce Type: cross  Abstract: Benchmark accuracy is routinely read as evidence that a model understands the construct the benchmark measures. We propose and instantiate on AeroIntent a five-layer framework that decomposes that reading into observation, representation, decoding, use, and behaviour. The framework separates encoding, decodability, and use as distinct levels. In the instantiation, the observation layer is unreliable on 7 of the 20 Drone programs and floors the direct visual correlate of radial velocity at a 0.26\% scale change. The representation layer exposes two task-designed RGB features with nominal signals. RAFT flow radial bias reaches $r=+0.257$ ($p=0.030$), and bounding-box optical-flow divergence reaches $r=+0.203$ ($p=0.029$). No multi-feature readout gives a fold-mean test correlation above the cluster minimum detectable effect (MDE), and the largest positive global out-of-sample correlation is $r=+0.080$. The decoding layer sets the MDE at 
    
[^95]: 扩散模型能学会泛化基础视觉技能吗？

    Do Diffusion Models Learn to Generalize Basic Visual Skills?

    [https://arxiv.org/abs/2611.22427](https://arxiv.org/abs/2611.22427)

    该论文提出视觉生成实验室（VGL）受控实验框架，通过基于规则生成的合成数据和任务特定指标，首次精确测量扩散模型对大小、位置、旋转、数量、颜色、形状等基础视觉技能的泛化能力，并揭示模型学习中的三种模式。

    

    扩散模型能够生成高质量的图像，但其泛化能力却难以衡量。目前的评估依赖于针对文本到图像模型的人类偏好和提示词遵循基准，以及针对类别条件模型的分布级指标（如Fréchet Inception距离（FID）和Inception分数（IS）），但这些方法无法判断模型是真正学会了基础视觉技能，还是主要在匹配训练数据的模式。为了填补这一空白，我们提出了视觉生成实验室，这是一个受控实验框架，用于研究扩散模型如何学习和泛化基础视觉技能，包括大小、位置、旋转、数量、颜色、形状及其组合。通过根据明确的规则为每项技能生成合成训练数据，VGL能够使用基于规则的任务特定指标精确测量泛化能力。在这些数据上从头训练模型，揭示了这种受控环境下的三种模式：(1)

    arXiv:2611.22427v1 Announce Type: cross  Abstract: Diffusion models generate high-quality images, but how well they generalize is hard to measure. Current evaluations rely on human preference and prompt-following benchmarks for text-to-image models and on distribution-level metrics such as Fr\'echet Inception Distance (FID) and Inception Score (IS) for class-conditional models, but these cannot tell whether models learn basic visual skills or largely match training patterns. To address this gap, we introduce the Visual Generative Lab (VGL), a controlled experimental framework for studying how diffusion models learn and generalize basic visual skills, including size, position, rotation, count, color, shape, and their compositions. By generating synthetic training data for each skill from explicit rules, VGL allows precise measurement of generalization with task-specific, rule-based metrics. Training models from scratch on these data reveals three patterns in this controlled setting: (1)
    
[^96]: InfuseLoop：面向可扩展并行循环Transformer的跨轨迹潜在注入机制

    InfuseLoop: Cross-Trajectory Latent Injection for Scalable Parallel Looped Transformers

    [https://arxiv.org/abs/2611.22426](https://arxiv.org/abs/2611.22426)

    提出InfuseLoop，一种跨轨迹潜在注入机制，通过轻量级注意力残差将后续预测轨迹中因果可用的隐藏状态注入当前迭代，解决了并行循环Transformer随深度扩展出现的训练损失尖峰和收益递减问题，同时保持token并行训练并仅需O(Rd)的额外存储。

    

    并行循环Transformer（PLT）通过重叠循环迭代来利用原本闲置的解码计算资源。然而，原始的PLT在深度扩展方面表现不佳：它们会出现训练损失尖峰，且随着循环数量的增加，其质量提升逐渐缩小。额外循环带来的收益取决于每次迭代可获取的信息，因此我们研究了现有的计算依赖关系是否限制了有效的深度扩展。我们发现，目前只有轨迹内的前一个状态会在循环边界之间传递，而实际上已完成的跨轨迹表示早已可用。基于这一发现，我们提出了InfuseLoop——一种跨轨迹潜在注入机制。它通过轻量级的注意力残差，从后续预测轨迹中聚合因果可用的隐藏状态。这一设计保留了token并行训练和并发循环执行，且仅需O(Rd)的额外隐藏状态存储。

    arXiv:2611.22426v1 Announce Type: new  Abstract: Parallel Looped Transformers (PLTs) use otherwise idle decoding compute by overlapping recurrent iterations. However, vanilla PLTs scale poorly with depth: they exhibit training-loss spikes, and their quality gains shrink as more loops are added. The benefit of additional loops depends on the information available to each iteration, so we examine whether the existing computational dependencies limit effective depth scaling. We find that only the preceding in-trajectory state is passed across loop boundaries, even though completed cross-trajectory representations are already available. Based on this finding, we propose InfuseLoop, a cross-trajectory latent injection mechanism. It aggregates causally available hidden states from later prediction trajectories through lightweight attention residuals. This design preserves token-parallel training and concurrent loop execution, and requires only $O(Rd)$ additional hidden-state storage. Experim
    
[^97]: 从约束到因果性：用于解释二值神经网络推理的Petri网编码

    From Constraints to Causality: Petri net Encodings for Explaining Binary Neural Network Inference

    [https://arxiv.org/abs/2611.22396](https://arxiv.org/abs/2611.22396)

    本文提出一种组合式翻译方法，将固定的已训练二值神经网络的推理过程编码为1-安全Petri网，使每个引发序列都能成为其决策的因果轨迹，从而在传统SAT/SMT验证之外为BNN推理提供了因果层面的可解释性。

    

    二值神经网络对于资源受限的硬件以及自主系统中的设备端组件极具吸引力。然而，其离散的、非线性的行为使其在安全相关场景中难以被分析和信任。现有的基于SAT（可满足性）与SMT（可满足性模理论）的方法能够验证BNN的输入-输出性质，但它们将BNN视为一个整体性的约束系统，这只能揭示某个性质是否成立，而无法揭示计算是如何得出结果的。在本文中，我们提出了一项提供互补性离散事件视角的早期工作。我们给出了一种组合式的翻译方法，将一个固定的、已训练的BNN的推理语义翻译为1-安全Petri网（1-safe Petri net），其中每个计算步骤都是一个显式的变迁（transition），从而使一个引发序列成为BNN决策的因果轨迹。在一个具体实例上，我们通过与参考实现进行穷尽的输入-输出等价性验证，确立了该翻译的忠实性。

    arXiv:2611.22396v1 Announce Type: cross  Abstract: Binary Neural Networks (BNNs) are attractive for resource-constrained hardware and on-device components in autonomous systems. However, their discrete, non-linear behaviour makes them difficult to analyse and trust in safety-relevant settings. Existing approaches based on SAT and Satisfiability Modulo Theories can verify a BNN's input-output properties, but treat it as a monolithic constraint system, which reveal only whether a property holds and not how the computation reaches its result. In this paper, we present an early work that provides a complementary, discrete-event view. We give a compositional translation from the inference semantics of a fixed, trained BNN to a 1-safe Petri net (PN), where each computation step is an explicit transition, so that a firing sequence is a causal trace of the BNN's decision. On a worked example, we establish faithfulness via exhaustive input-output equivalence against a reference implementation, 
    
[^98]: 公平冒泡排序：针对连续敏感属性的可证明最优公平排序

    Fair Bubble Sort: Provably Optimal Fair Ranking with Continuous Sensitive Attributes

    [https://arxiv.org/abs/2611.22352](https://arxiv.org/abs/2611.22352)

    本文提出连续成对公平性（CPP）这一新公平性概念，并设计了基于针对性相邻交换的公平冒泡排序（FBS）后处理算法，首次实现无需离散化处理的连续敏感属性公平排序，且证明了该算法在无权重Kemeny距离下精确最优。

    

    大多数公平排序方法都假设受保护属性是类别型的，需要将年龄或收入等连续属性离散化成若干组。这会引入任意的阈值，丢失细粒度信息，并忽略属性本身固有的顺序。在本文中，我们研究了不依赖阈值化的、面向连续受保护属性的公平排序问题。我们提出了连续成对公平性，这是一种基于排序位置与受保护属性之间Kendall相关性的公平性概念，并寻求在满足给定CPP约束的同时、与初始基于分数的排序之间Kemeny距离最小的排序。为解决这一问题，我们提出了公平冒泡排序（FBS），一种基于有针对性相邻交换的高效后处理算法。我们证明FBS在无权重Kemeny距离下是精确最优的，并为分数加权情形提供了近似保证。

    arXiv:2611.22352v1 Announce Type: new  Abstract: Most fair-ranking methods assume categorical protected attributes, requiring continuous attributes such as age or income to be discretized into groups. This introduces arbitrary thresholds, loses fine-grained information, and ignores the natural ordering of the attribute. In this paper, we study the problem of fair ranking with continuous protected attributes without relying on thresholding. We introduce Continuous Pairwise Parity (CPP), a fairness notion based on Kendall correlation between ranking positions and the protected attribute, and seek rankings that satisfy a prescribed CPP constraint while minimizing their Kemeny distance from an initial score-based ranking. To solve this problem, we propose Fair Bubble Sort (FBS), an efficient post-processing algorithm based on targeted adjacent swaps. We prove that FBS is exactly optimal for the unweighted Kemeny distance and provide approximation guarantees for the score-weighted setting. 
    
[^99]: 视觉-语言融合的缩放定律

    Scaling Laws for Vision-Language Fusion

    [https://arxiv.org/abs/2611.22351](https://arxiv.org/abs/2611.22351)

    本文通过系统分析多模态预训练的学习动态，推导出一个统一的缩放定律，将基础学习轨迹与模态竞争驱动的分布偏移相分离，从而为理解视觉-语言融合中早期融合与后期融合范式的根本权衡提供了原理性依据。

    

    视觉-语言模型（VLM）已成为将多模态感知与文本理解相结合的核心基础模型。传统上，VLM通过后期融合（late fusion）方式构建，即在现成的语言模型上继续使用多模态（图文）数据进行预训练；而从头开始预训练时就混合多种模态（早期融合，early fusion）的做法近来也日益受到关注。然而，这两种范式之间的根本权衡仍鲜为人知，导致训练方案成本高昂且依赖启发式经验。本文系统地分析了这些多模态预训练过程的学习动态，表明视觉-语言融合过程中的验证损失可以通过将基础学习轨迹（其中不同数据类型以不同速率做出贡献）与由模态间竞争驱动的分布偏移相分离来建模。我们推导出一个统一描述这些动态的缩放定律，从而能够准确……（摘要在此处截断）

    arXiv:2611.22351v1 Announce Type: new  Abstract: Vision-language models (VLMs) have emerged as central foundation models combining multimodal perception with text comprehension. While VLMs are traditionally built via late fusion by continued pre-training an off-the-shelf language model on multimodal (image-text) data, mixing modalities from the start of pre-training (early fusion) has recently gained traction. However, the fundamental trade-offs between these paradigms remain poorly understood, resulting in expensive and heuristic-driven training recipes. In this paper, we systematically analyze the learning dynamics of these multimodal pre-training processes, showing that validation losses during vision-language fusion can be modeled by separating the base learning trajectory, where different data types contribute at distinct rates, from a distribution shift driven by the competition between modalities. We derive a scaling law that unifiedly describes these dynamics, enabling accurate
    
[^100]: BQ-LoRA：作为参数高效微调隐式正则化器的二值量化低秩适配器

    BQ-LoRA: Binary-Quantized Low-Rank Adapters as Implicit Regularizers for Parameter-Efficient Fine-Tuning

    [https://arxiv.org/abs/2611.22334](https://arxiv.org/abs/2611.22334)

    该论文提出 BQ-LoRA 框架，将 LoRA 适配器参数量化为单个二值比特并配合可学习的逐层缩放因子与直通估计器，在将适配器存储降低 13-16 倍的同时，从理论上刻画了其假设类的谱范数界与 Rademacher 复杂度，证明二值适配器可作为隐式正则化器实现参数高效微调。

    

    arXiv:2611.22334v1 公告类型：新 摘要：量化大语言模型的参数高效微调目前由 QLoRA 主导，该方法将 4 比特权重量化与 FP16 低秩适配器相结合。我们提出一个问题：FP16 适配器精度是否必要？将每个适配器参数替换为单个二值比特，并结合可学习的逐层缩放因子 $s$ 和缩放直通估计器，可将适配器存储降低 13-16 倍。我们引入两个变体——NormScale，在保持范数的缩放下使用截断 STE 进行二值化；以及 LearnThresh，使二值化阈值可学习——并将它们统一在我们称为 BQ-LoRA 的框架之下。我们的主要理论贡献是对 BQ-LoRA 假设类的紧致刻画：它具有有界的逐元素范数、一个比最坏情况紧 $\Theta(r)$ 倍的典型集谱范数界、满秩的近各向同性更新，因此其 Rademacher 复杂度给出了显式的 $\tilde{O}(s_{\max}\...\）界（摘要在此处截断）。

    arXiv:2611.22334v1 Announce Type: new  Abstract: Parameter-efficient fine-tuning of quantized large language models is dominated by QLoRA, which combines 4-bit weight quantization with FP16 low-rank adapters. We ask whether the FP16 adapter precision is necessary. Replacing it with a single binary bit per adapter parameter, combined with a learned per-layer scale $s$ and a Scaled Straight-Through Estimator, reduces adapter storage by 13-16x. We introduce two variants -- NormScale, which binarizes under a norm-preserving scale with a clipped STE, and LearnThresh, which makes the binarization threshold learnable -- unified under a framework we call BQ-LoRA. Our main theoretical contribution is a tight characterization of the BQ-LoRA hypothesis class: it has bounded entrywise norm, a typical-set spectral norm bound that is $\Theta(r)$ times sharper than the worst case, a full-rank, near-isotropic update, and consequently a Rademacher complexity that yields an explicit $\tilde{O}(s_{\max}\
    
[^101]: 面向车辆重定位的多阶段上下文优化

    Multistage Contextual Optimization for Vehicle Repositioning

    [https://arxiv.org/abs/2611.22330](https://arxiv.org/abs/2611.22330)

    本文提出一种多阶段上下文优化方法，将车辆重定位建模为含宏观基本图的多阶段线性规划，并利用决策性能的精确梯度训练预测模型，以优化网约车平台的供需平衡。

    

    网约车平台通过重新定位闲置车辆来改善空间供需平衡。该问题是预测驱动且多阶段的：在每个阶段，平台预测未来需求、执行当前决策，并利用所产生状态来初始化下一阶段。我们构建了一个多阶段线性规划（LP），其中宏观基本图（MFD）将车辆累积量与行程完成率联系起来。随后，我们开发了一种多阶段上下文优化方法，利用决策性能来训练预测模型。该学习问题结合了多个相互耦合的阶段、依赖于先前决策的上下文，以及同时影响线性规划目标函数、约束矩阵和右端项的预测。我们推导出了精确的梯度，该梯度包含三个部分：参数变化会影响预测进而影响决策；先前的决策通过继承状态影响当前决策；该状态也会改变预测。

    arXiv:2611.22330v1 Announce Type: new  Abstract: Ride-hailing platforms reposition idle vehicles to improve spatial supply-demand balances. This problem is prediction-driven and multistage: at each stage, the platform forecasts future demand, executes current decisions, and uses the resulting state to initialize the next stage. We formulate a multistage linear program (LP) with a macroscopic fundamental diagram (MFD) linking accumulation to trip completion. We then develop a multistage contextual optimization approach that uses decision performance to train the prediction model. The learning problem combines multiple coupled stages, context that depends on earlier decisions, and predictions that affect the objective, the constraint matrix, and the right-hand side of the LP. We derive the exact gradient, which has three parts: parameters change forecasts and hence decisions; earlier decisions change the current decision through the inherited state; and this state also changes forecasts 
    
[^102]: PVTSizing 2.0：面向智能体EDA工作流的快速可靠模拟电路尺寸设计算子

    PVTSizing 2.0: A Fast and Reliable Analog Circuit Sizing Operator for Agentic EDA Workflows

    [https://arxiv.org/abs/2611.22328](https://arxiv.org/abs/2611.22328)

    PVTSizing 2.0 提出了一个围绕可独立使用数值优化器构建的人-智能体协作工作流，结合损失引导更新、评论家引导的局部细化以及可选的 LLM 顾问，实现了快速可靠的跨 PVT 角点模拟电路尺寸设计。

    

    模拟电路设计自动化既需要协调的工具执行，也需要高效的数值搜索。大型语言模型（LLM）为组织工具执行提供了新的途径，但找到一个在工艺、电压和温度（PVT）各角点上均满足规格的尺寸设计方案仍然代价高昂。PVTSizing 2.0 通过一个围绕可独立使用的数值优化器构建的人-智能体协作工作流来满足这些需求。智能体负责执行设计者批准的任务并进行结果审查。优化器通过损失引导的更新减少不必要的训练，并采用评论家引导的局部细化来探索更多候选方案，同时兼顾跨角点之间的冲突。可选的 LLM 顾问提供参数之间的关系，将数值搜索聚焦于基于电路知识的方向，而优化器则负责选择调整幅度并通过仿真验证其效果。在四个电路基准测试中……（摘要内容不完整，在此截断）

    arXiv:2611.22328v1 Announce Type: cross  Abstract: Analog circuit design automation requires both coordinated tool execution and efficient numerical search. Large language models (LLMs) offer new ways to organize tool execution, but finding one sizing that satisfies specifications across process, voltage, and temperature (PVT) corners remains costly. PVTSizing 2.0 addresses these needs through a human-agent workflow built around an independently usable numerical optimizer. An agent carries the designer's approved task through execution and result review. The optimizer reduces unnecessary training through loss-guided updates and uses critic-guided local refinement to explore additional candidates while accounting for cross-corner conflicts. An optional LLM adviser supplies parameter relationships to focus numerical search on circuit-informed directions, while the optimizer selects adjustment amplitudes and verifies their effects through simulation. Across four circuit benchmarks, the st
    
[^103]: 面向语言可接受性的零维注意力拓扑

    Zero-Dimensional Attention Topology for Linguistic Acceptability

    [https://arxiv.org/abs/2611.22312](https://arxiv.org/abs/2611.22312)

    本文提出用零维持续同调将144个对称化BERT注意力图压缩为1,440个拓扑特征用于语言可接受性分类，但实验表明其域外性能（MCC 0.438）低于注意力统计（0.508）和BERT本身（0.545），且该表示会丢失方向性注意力差异信息。

    

    Transformer注意力的拓扑摘要为语言可接受性分类提供了句子表示。我们研究了一种零维持续同调（$H_0$）表示，它将来自144个对称化BERT注意力图的最小生成树合并值汇总为1,440个特征。在经过数据完整性过滤后，回顾性评估使用了8,517个下游训练句子，以及521个域内和497个域外的CoLA验证句子。记录的域外Matthews相关系数为：拓扑表示为0.438，注意力统计为0.508，而BERT为0.545。注意力特征结果是保留的汇总测量值；BERT预测结果根据附带记录进行了重新评分，并重新拟合了长度控制。分析结果表明，该表示保留了词元数量信息，但丢弃了一类方向性注意力差异。

    arXiv:2611.22312v1 Announce Type: new  Abstract: Topological summaries of Transformer attention provide sentence representations for linguistic acceptability classification. We examine a zero-dimensional persistent-homology ($H_0$) representation that summarizes minimum-spanning-tree merge values from 144 symmetrized BERT attention graphs into 1,440 features. The retrospective evaluation uses 8,517 downstream training sentences and 521 in-domain and 497 out-of-domain CoLA validation sentences after data-integrity filtering. The recorded out-of-domain Matthews correlation coefficients are 0.438 for the topological representation and 0.508 for attention statistics, compared with 0.545 for BERT. Attention-feature results are retained aggregate measurements; BERT predictions are rescored and length controls are refitted from the accompanying records. Analytical results show that the representation retains token-count information and discards a class of directional attention differences. So
    
[^104]: VRLLM：将大语言模型转化为强大的表格分类器

    VRLLM: Turning LLMs into Powerful Tabular Classifiers

    [https://arxiv.org/abs/2611.22308](https://arxiv.org/abs/2611.22308)

    Visa Research通过在数十亿行带标签表格数据上微调Llama 3.2 1B模型（VRLLM），首次证明LLM在大规模表格分类任务上可以超越传统表格分类器以及生产级深度学习模型。

    

    大语言模型（LLM）在文本任务上已展现出强大性能，但预训练LLM是否能有效用于大规模表格分类仍不明确，因为许多先前的研究表明，LLM在表格数据分类上的表现落后于基于树的模型或专为表格数据设计的神经架构。在Visa Research，我们成功地在数十亿行带标签的表格数据上微调了预训练的Llama 3.2 1B大语言模型，使其超越了众多表格分类器，甚至超越了我们具有竞争力的生产级深度学习模型。为便于参考，我们将该模型称为Visa Research LLM（VRLLM）。在本文中，我们分享了实现这一令人振奋的成果的方法，涵盖数据准备、训练方案、设计选择的消融研究以及部署验证。我们还在同一任务上对前沿LLM进行了基准测试，并分析了不同模型家族和世代之间的预测行为差异。

    arXiv:2611.22308v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have shown strong performance on text, but whether pretrained LLMs are effective for large-scale tabular classification remains unclear, as many previous works demonstrate that LLMs' tabular data classification performance falls behind tree-based models or neural architectures designed for tabular data. At Visa Research, we managed to fine-tune a pretrained Llama 3.2 1B LLM on billions of rows of labeled tabular data and make it outperform a wide array of tabular classifiers and even our competitive production deep learning model. For ease of reference, we refer to the model as the Visa Research LLM (VRLLM). In this paper, we share how we achieved this exciting result, covering data preparation, training recipe, ablation studies of design choices and deployment validation. We also benchmark frontier LLMs on the same task and analyze their prediction behavior across model families and generations. Our interp
    
[^105]: 相似并不等于有用：基于闭环效用的上下文示范选择

    Similar Is Not Useful: Selecting In-Context Demonstrations by Closed-Loop Utility

    [https://arxiv.org/abs/2611.22295](https://arxiv.org/abs/2611.22295)

    该论文提出用“闭环效用”（即演示对任务实际成功率的贡献）而非视觉相似性来为检索增强VLA策略选择上下文演示，实验证明效用选择的演示池成功率比相似性选择高出10.8个百分点，且在策略从未见过的任务上同样有效。

    

    检索增强的视觉-语言-动作（VLA）策略通常利用冻结的DINOv2嵌入，将与当前观测画面最相似的演示填充到上下文中。我们探究这种视觉相似性是否真能选出使策略成功的演示。针对一个冻结的检索增强VLA（RICL on π₀-FAST，LIBERO），我们使演示池可以在运行时动态替换，并通过将60个固定大小演示池的闭环任务成功率对其成员关系进行回归，为每个演示打分其“效用”。在与评分所用初始状态不相交的新初始状态上，包含五个最高效用演示的池在19个任务的760个配对片段中成功率为66.4%，五个最相似演示的池为55.7%，随机池为55.5%。效用选择比相似性选择高出10.8个百分点（p=3.2×10⁻⁶；在18个未打平任务中有15个为正向提升），其中包括12个策略从未被演示引导过的任务，而相似性选择的演示池表现并不优于随机……（摘要在此处截断）

    arXiv:2611.22295v1 Announce Type: new  Abstract: Retrieval-augmented vision-language-action (VLA) policies fill their context with the demonstrations whose frames look most like the current observation under frozen DINOv2 embeddings. We ask whether visual similarity selects the demonstrations that make the policy succeed. For a frozen retrieval-augmented VLA (RICL on $\pi_0$-FAST, LIBERO), we make the demonstration pool swappable at run time and score each demonstration's utility by regressing the closed-loop success of 60 fixed-size pools on their membership. On initial states disjoint from those used for scoring, pools of the five highest-utility demonstrations succeed on 66.4% of 760 paired episodes across 19 tasks, pools of the five most similar on 55.7%, and random pools on 55.5%. Utility beats similarity by 10.8 points ($p=3.2\times10^{-6}$; positive on 15 of 18 untied tasks), including on 12 tasks the policy was never primed on, and similarity-selected pools do no better than ra
    
[^106]: 从感知到想象：面向世界模型的可执行记忆

    Perception into Imagination: Executable Memory for World Models

    [https://arxiv.org/abs/2611.22294](https://arxiv.org/abs/2611.22294)

    提出可执行感知记忆（EPM）世界模型框架，在想象过程中同步推进显式感知状态与神经潜在动态，并将演化读出信息用于奖励预测、价值估计和策略学习，显著提升智能体在视野或接触丢失后利用过往感知信息的能力。

    

    物理智能体在失去视觉或接触之后，必须利用其先前看到或触摸过的信息。它们的行动会改变这些被记忆事实的含义。我们提出了可执行感知记忆（EPM），这是一种世界模型框架，在想象过程中使显式感知状态与神经潜在动态一同演化推进。其不断更新的读出信息为奖励预测、价值估计和策略学习提供依据。我们将奖励输入与产生这些奖励的状态转移对齐，并在记录动作的开环推演上对奖励头进行监督。在 Memory Maze 13×13 上，EPM 在 100 万次交互时达到 30.37 的回报，而相同的演化读出仅在共同感知前端上用于策略和价值学习时为 23.37：增益为 7.00 [3.80, 10.21]。完整方法还在 9×9 和 MiniWorld 中提升了控制性能。对记忆中视野外到达目标的探测显示，两种视觉实现中均表现出更强的奖励预期。

    arXiv:2611.22294v1 Announce Type: cross  Abstract: Physical agents must use what they previously saw or touched after visibility or contact is lost. Their actions change the implications of these remembered facts. We introduce executable perceptual memory (EPM), a world-model framework that advances explicit perceptual state alongside neural latent dynamics during imagination. Its evolving read-out informs reward prediction, value estimation and policy learning. We align reward inputs with the transitions that produced them and supervise the reward head on open-loop rollouts of recorded actions. On Memory Maze $13\times13$, EPM reaches 30.37 return at 1M interactions, compared with 23.37 when the same evolving read-out informs only policy and value on a common perceptual front end: a gain of 7.00 [3.80, 10.21]. The complete method also improves control on $9\times9$ and in MiniWorld. Probes of remembered, out-of-view arrivals show stronger reward anticipation in both visual implementat
    
[^107]: ConfGuard：面向预测性维护大语言模型智能体的基于物理规律的参考监视器

    ConfGuard: a physics-grounded reference monitor for predictive-maintenance large-language-model agents

    [https://arxiv.org/abs/2611.22288](https://arxiv.org/abs/2611.22288)

    ConfGuard 提出了一种基于退化物理规律的参考监视器，通过保形预测校验预测性维护 LLM 智能体的剩余寿命预测是否符合物理约束（寿命不上升且每周期至多降一周期），从而抵御文档注入攻击和传感器扰动。

    

    大语言模型智能体如今已能闭环工业预测性维护流程：读取传感器数据窗口、检索维护规程并派发工单。此处存在两条攻击通道：检索文档中被注入的文本可能劫持智能体的决策，而有界的传感器扰动可能破坏其背后的剩余使用寿命（RUL）估计。近期的智能体防御手段只管控数据来源和工具权限，却无法判断预测量在物理上是否合理。ConfGuard 是一个采用基于物理策略的参考监视器。其保形监视器以资产已确认的历史记录为锚点，依据退化物理规律检验每一项预测：被封顶的剩余寿命绝不会上升，且每个周期至多下降一个周期。通过在完整的资产轨迹上校准阈值，操作员可以设定运行中的监视器对健康资产数据窗口触发升级告警的预期比例。一个物理策略验证器负责检查智能体的维护紧急性判断（原文在此处截断）。

    arXiv:2611.22288v1 Announce Type: new  Abstract: Large-language-model agents now close the industrial predictive-maintenance loop, reading sensor windows, retrieving procedures and dispatching work orders. Two attack channels meet there: injected text in a retrieved document can hijack the decision, and a bounded sensor perturbation can corrupt the remaining-useful-life estimate behind it. Recent agent defenses police data provenance and tool privilege but cannot tell whether a prognostic quantity is physically plausible. ConfGuard is a reference monitor with a physics-grounded policy. Its conformal monitor tests each prediction, anchored to the asset's admitted history, against degradation physics: capped remaining life never rises and falls at most one cycle per cycle. Calibrating its threshold over whole asset trajectories lets the operator set the expected share of a clean asset's windows the running monitor escalates. A physics-policy validator checks the agent's maintenance-urgen
    
[^108]: XM-Sparse：基于扩展内存池的高吞吐量稀疏注意力大语言模型服务系统

    XM-Sparse: High-Throughput Sparse-Attention LLM Serving over Scale-up Memory Pools

    [https://arxiv.org/abs/2611.22280](https://arxiv.org/abs/2611.22280)

    XM-Sparse是首个基于扩展内存池的稀疏注意力KV缓存卸载系统，通过分析热KV缓冲区与索引器键缓存的不同访问模式并针对性优化，突破了HBM容量对长上下文大模型服务吞吐量的限制。

    

    稀疏注意力正日益被生产级大语言模型采用以进行长上下文推理，但服务系统仍需保留完整的KV历史记录，这使得GPU HBM容量成为限制服务并发度的关键因素。现有系统将KV缓存卸载到主机内存，但受限于PCIe有限的带宽和主机内存有限的容量。新兴的扩展内存池（如X-Mem）缓解了这两项限制，并为稀疏注意力卸载开辟了新的机会。在本文中，我们首先进行了深入分析，揭示了即使采用扩展内存池，解码吞吐量仍然受限于HBM容量，因为热KV缓冲区和完全驻留的索引器键缓存占据了大量HBM空间。此外，这两个缓存表现出截然不同的访问模式。基于这些洞察，我们提出了XM-Sparse，这是首个基于扩展内存扩展的稀疏注意力卸载系统。XM-Sparse的核心思想是将KV数据卸载

    arXiv:2611.22280v1 Announce Type: cross  Abstract: Sparse attention has been increasingly adopted by production LLMs for long-context inference, yet serving systems must still retain the full KV history, making GPU HBM capacity a key limit on serving concurrency. Existing systems offload the KV cache to host memory, but suffer from the limited bandwidth of PCIe and the limited capacity of host memory. Emerging scale-up memory pools (e.g., X-Mem) relieve both limits and open new opportunities for sparse-attention offloading. In this paper, we first conduct an in-depth analysis and reveal that, even with scale-up memory pools, decode throughput remains bounded by HBM capacity, as hot KV buffers and the fully resident indexer key (K-index) cache occupy substantial HBM. Moreover, the two caches exhibit distinct access patterns. Based on these insights, we present XM-Sparse, the first sparse-attention offloading system over scale-up memory expansion. The core idea of XM-Sparse is to offload
    
[^109]: Heave-Ho：去中心化的长时程多人形机器人协作

    Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration

    [https://arxiv.org/abs/2611.22275](https://arxiv.org/abs/2611.22275)

    该论文提出Heave-Ho去中心化框架，通过多智能体后训练实现物理耦合的人形机器人协调控制，结合视觉-语言规划器与基于语音的领导者-跟随者机制，在无需网络通信的情况下实现最多十台（含异构）人形机器人的长时程协作移动操作。

    

    协作有望将人形机器人的能力扩展到超越单个机器人所能达到的水平。然而，实现这一潜力需要协调的全身交互、连贯的长时程规划以及机器人之间的精确同步。我们提出了Heave-Ho，一个面向长时程协作式多人形机器人移动操作的去中心化框架。在低层，我们开发了一个多智能体后训练框架，将人形机器人基础追踪模型适配到人形机器人之间物理耦合的交互中，仅凭局部观测即可实现协调的去中心化控制。在高层，我们训练了一个视觉-语言规划器，并设计了一种基于语音的领导者-跟随者协调机制，从而在无需网络通信的情况下同步去中心化执行。我们的系统通过两台相同的人形机器人进行训练，可扩展至最多十台机器人的团队，并且无需重新训练即可泛化到异构团队。

    arXiv:2611.22275v1 Announce Type: cross  Abstract: Collaboration promises to extend humanoid capabilities beyond what individual robots can achieve. However, realizing this potential requires coordinated whole-body interactions, coherent long-horizon planning, and precise synchronization across robots. We introduce Heave-Ho, a decentralized framework for long-horizon collaborative multi-humanoid loco-manipulation. At the low level, we develop a multi-agent post-training framework that adapts a humanoid foundation tracking model to physically coupled interactions between humanoids, enabling coordinated decentralized control from local observations alone. At the high level, we train a vision-language planner and design a speech-based leader-follower coordination mechanism to synchronize decentralized execution without network communication. Trained with two identical humanoids, our system scales to teams of up to ten robots and generalizes to heterogeneous teams without retraining. To be
    
[^110]: OfficePrism：面向企业业务职能单域操作的智能体基准测试

    OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions

    [https://arxiv.org/abs/2611.22262](https://arxiv.org/abs/2611.22262)

    该论文提出 OfficePrism 基准，通过 420 个覆盖七大企业业务职能的单域任务（结合中文工单、可执行参考见证及自动与人工筛选流程），在 124 种模型-框架-工具包配置下系统评估智能体，发现其自然语言与数据库联合通过率仅为 22.6%–76.4%。

    

    宽泛的职场能力评分可能将单一业务职能内的胜任能力与跨职能、跨系统协调的额外成本混为一谈，导致失败难以定位。OfficePrism 通过 420 个涵盖七大业务职能的单域企业任务，在统一的、由基准定义的 L1–L3 任务结构层级下评估这一前提。每个任务都在一个全新的环境中运行，该环境由一个共享的合成组织与一个特定职能领域及类型化状态访问组合而成。我们的流水线利用有据可依的记录、业务规则和工具模式，联合构建中文工单、隐藏的可执行参考见证以及预期响应，随后进行自动过滤和人工筛选。同一任务集通过完整模式工具包和搜索-分发工具包两种方式进行评估。在 124 种模型-框架-工具包配置中，每个案例的自然语言与数据库联合通过率介于 22.6% 至 76.4% 之间。

    arXiv:2611.22262v1 Announce Type: cross  Abstract: Broad workplace scores can mix competence within one business function with the extra cost of coordinating across functions and systems, making failures hard to localize. OfficePrism evaluates this prerequisite through 420 single-domain enterprise tasks spanning seven business functions under shared, benchmark-defined L1--L3 task-structure levels. Each task runs in a fresh environment combining a shared synthetic organization with one function-specific domain and yped state access. Our pipeline uses grounded records, business rules, and tool schemas to jointly construct a Chinese-language ticket, a hidden executable reference witness, and an expected response, followed by automated filtering and human curation. The same task set is evaluated through full-schema and search-and-dispatch packages. Across 124 model--harness--package configurations, the per-case joint natural-language-and-database pass rate ranges from 22.6% to 76.4%, with 
    
[^111]: 面向复杂平台跳跃游戏的PCGRL：通过膨胀包络与前向课程合成可玩且多样化的关卡

    PCGRL for Complex Platformers: Synthesizing Playable and Diverse Levels via Inflated Envelopes and Forward Curricula

    [https://arxiv.org/abs/2611.22250](https://arxiv.org/abs/2611.22250)

    该论文提出一个物理感知、课程引导的PCGRL框架，通过“膨胀体积包络”机制将连续运动学可行性融入离散状态空间，并引入中间状态转移反馈来缓解稀疏奖励问题，从而在复杂平台跳跃游戏的大规模关卡生成中保证关卡的可玩性与多样性。

    

    通过强化学习进行程序化内容生成（PCGRL）已成为自动化游戏开发中一种有前景的范式。然而，现有的PCGRL框架在面对由严格刚体规则和非线性运动学边界（如平台跳跃约束）所支配的长序列生成任务时，仍然存在严重的探索瓶颈和扩展性限制。因此，传统方法经常无法在大规模画布上保持可靠的关卡可解性。为了解决这些挑战，我们提出了一个物理感知、课程引导的PCGRL框架。首先，我们构建了一个“膨胀体积包络”机制，它提供了一种物理约束的表示方法，将连续运动学可行性融入马尔可夫决策过程（MDP）的离散状态空间中。其次，为了缓解由极长决策序列引起的稀疏奖励挑战，我们引入了中间状态转移反馈（摘要在此处被截断）……

    arXiv:2611.22250v1 Announce Type: cross  Abstract: Procedural Content Generation via Reinforcement Learning (PCGRL) has emerged as a promising paradigm for automated game development. However, existing PCGRL frameworks still suffer from severe exploration bottlenecks and scaling limitations when faced with long-sequence generation tasks governed by strict rigid-body rules and non-linear kinematic boundaries (e.g., platformer constraints). Consequently, traditional approaches frequently fail to maintain reliable level solvability on large-scale canvases. To address these challenges, we present a physics-aware, curriculum-guided PCGRL framework. First, we construct an "Inflated Volume Envelope" mechanism, which provides a physics-constrained representation that incorporates continuous kinematic feasibility into the discrete state space of the MDP. Second, to alleviate the sparse reward challenge induced by extremely long decision sequences, we introduce intermediate state-transition feed
    
[^112]: OAK-Mind：基于知识驱动的多智能体大语言模型实现零训练的被动感知心理健康变化推断

    OAK-Mind: Zero-Training Mental Health Change Inference from Passive Sensing via Knowledge-Grounded Multi-Agent LLMs

    [https://arxiv.org/abs/2611.22248](https://arxiv.org/abs/2611.22248)

    OAK-Mind提出了一种零训练、多智能体、知识驱动的LLM流水线，无需标注数据或任务特定微调，通过基于心理健康文献解读睡眠、活动、移动等多维被动感知数据来推断心理健康变化。

    

    可穿戴设备和智能手机传感器能够捕捉与心理健康相关的行为信号，然而从这些信号推断心理健康结果传统上依赖于需要标注数据的监督式模型，且这些模型对其预测所提供的洞察十分有限。大语言模型（LLM）为基于行为证据进行推理提供了新的机会，但现有的基于LLM的被动感知方法在很大程度上仍依赖于任务特定的微调或传感器特定的预训练。在本文中，我们提出了OAK-Mind，这是一个零训练、多智能体、知识驱动的LLM流水线，用于从被动感知数据中推断心理健康变化。该流水线将感知数据组织成多个行为维度，包括睡眠、身体活动、位置移动和智能手机使用，并协调多个以心理健康文献为知识基础的专业智能体来解读纵向行为数据。

    arXiv:2611.22248v1 Announce Type: cross  Abstract: Wearable and smartphone sensors capture behavioral signals associated with mental health, yet inferring mental health outcomes from these signals has traditionally relied on supervised models requiring labeled data and offering limited insight into their predictions. Large language models (LLMs) offer new opportunities to reason over behavioral evidence, but existing LLM-based approaches to passive sensing have largely continued to rely on task-specific fine-tuning or sensor-specific pretraining. In this paper, we introduce \textbf{OAK-Mind}, a zer\textbf{O}-training, multi-\textbf{A}gent, \textbf{K}nowledge-grounded LLM pipeline for inferring mental health changes from passive sensing data. The pipeline organizes sensing data into behavioral dimensions including sleep, physical activity, location mobility, and smartphone use, and orchestrates specialized agents grounded in mental health literature to interpret longitudinal behavioral 
    
[^113]: 面向隐式神经表示的空间局部小批量训练

    Spatially local mini-batching for implicit neural representations

    [https://arxiv.org/abs/2611.22236](https://arxiv.org/abs/2611.22236)

    本文证明低秩神经表示（LRNR）凭借其“局部更新仅通过每层少量系数作用于共享基”的结构特性，可以使用自适应网格单个单元内的点进行空间局部小批量训练，突破了传统INR必须在全域上采样训练的限制，并以Kolmogorov宽度论证提供了理论支持。

    

    基于隐式神经表示（INR）的神经代理模型通常采用分布在整个计算域上的小批量数据进行训练，因为对于一般的INR而言，当一个批次中的所有点都位于某个小区域内时，其精度会显著下降。我们证明，低秩神经表示（LRNR）——一种其权重矩阵是由小型超网络生成的系数与共享基相乘所构成的INR——可以使用空间局部的小批量数据进行训练。每一步训练使用自适应网格数值解某一时间快照中位于单个小单元内的点，快照和单元按照网格的加密程度进行抽取。我们认为，这种训练方式得益于LRNR的结构特性：局部更新仅通过每层的 $r$ 个系数作用于共享基，从而避免单个单元重写整个网络。通过Kolmogorov宽度论证，我们证明当限制在小单元内时，LRNR的隐藏状态接近于一个低（维子空间）……（摘要在此处截断）

    arXiv:2611.22236v1 Announce Type: new  Abstract: Neural surrogates based on implicit neural representations (INRs) are typically trained on mini-batches spread over the whole domain, since a generic INR loses accuracy when every point of a batch lies in one small region. We show that low-rank neural representations (LRNRs), INRs whose weight matrices are products of shared bases with coefficients emitted by a small hypernetwork, can be trained with spatially local mini-batches. Each step uses the points of one time snapshot of an adaptive-mesh numerical solution inside one small cell, with snapshots and cells drawn according to the refinement of the mesh. We posit that such training is enabled by the structural property of LRNRs, as a local step reaches the shared bases only through the $r$ coefficients of each layer, and keeps one cell from rewriting the network. By a Kolmogorov-width argument, we show that when restricted to a small cell, the hidden states of LRNRs lie close to a low
    
[^114]: 从温度受限到电池受限的腿式运动

    From Temperature-Limited to Battery-Limited Legged Locomotion

    [https://arxiv.org/abs/2611.22231](https://arxiv.org/abs/2611.22231)

    该论文提出一种无需观测温度的热感知强化学习运动策略，通过奖励函数压低最热电机的预测稳态温度，使腿式机器人的持续运动从热关机限制转变为电池续航限制。

    

    在腿式机器人上，持续运动往往以热关机而非电池耗尽告终：行走产生的热量集中在少数几个关节电机中，这些电机在电池耗尽之前很久就会过热。我们通过强化学习训练热感知的运动策略，而无需观测温度，并在宇树（Unitree）Go2四足机器人上进行了验证。我们首先辨识了一个耦合的13节点集总RC热网络，该网络基于单台机器人的两次运行数据进行拟合，并利用测量的电功率进行校准，能够在不同机器人个体、步态、载荷和速度条件下预测全部十二个电机的温度，平均RMSE低于1°C。在持续步态下，电机最终将达到的稳定温度可直接由步态平均的平方扭矩和关节速度推得。我们的核心贡献是一个奖励函数，它压低最热电机的预测稳态温度：热时间常数超出了强化学习的信用分配视野……

    arXiv:2611.22231v1 Announce Type: cross  Abstract: On legged robots, sustained locomotion often ends in thermal shutdown rather than battery depletion: the heat of walking concentrates in a few joint motors, which overheat long before the battery is empty. We train thermally aware locomotion policies by reinforcement learning without ever observing temperature, and demonstrate this on a Unitree Go2 quadruped. We first identify a coupled 13-node lumped RC thermal network, fitted on two runs of a single robot and calibrated against the measured electrical power, that predicts all twelve motor temperatures with a mean RMSE below 1 $^\circ$C across different robot instances, gaits, payloads, and speeds. Under a sustained gait, the temperatures the motors will settle at follow directly from the gait-averaged squared torques and joint speeds. Our core contribution is a reward that drives down the hottest predicted asymptotes: thermal time constants exceed the credit-assignment horizon of rei
    
[^115]: 通过人工智能发现新型抗病毒药物：体外和体内研究结果

    Discovery of new antiviral agents through artificial intelligence: in vitro and in vivo results

    [https://arxiv.org/abs/2611.22230](https://arxiv.org/abs/2611.22230)

    本研究利用深度强化学习分子设计平台生成并筛选化合物，发现了DS-22-inf-009和DS-22-inf-021两个新型非唾液酸样结构的神经氨酸酶抑制剂，可在低微摩尔浓度下有效抑制甲型和乙型流感病毒（包括H275Y耐药突变株）。

    

    在这项研究中，我们采用基于深度强化学习（RL）的分子设计平台，生成了针对甲型和乙型流感病毒神经氨酸酶（NA）的多样化化合物集合。共生成了60291个化合物，其中86.5%的化合物表现出优于奥司他韦的理化性质。经过计算筛选缩小选择范围后，选出九个具有非唾液酸样结构的化合物进行体外实验。我们发现两个化合物DS-22-inf-009和DS-22-inf-021，它们能在低微摩尔浓度下有效抑制甲型和乙型流感病毒（IAV和IBV）的神经氨酸酶，包括H275Y突变株。分子动力学模拟显示它们与氨基酸残基的相互作用模式与奥司他韦相似。在细胞实验中，DS-22-inf-009和DS-22-inf-021以剂量依赖性方式抑制IAV和IBV，EC50值范围为0.29微摩尔。

    arXiv:2611.22230v1 Announce Type: cross  Abstract: In this research, we employed a deep reinforcement learning (RL)-based molecule design platform to generate a diverse set of compounds targeting the neuraminidase (NA) of influenza A and B viruses. A total of 60291 compounds were generated, of which 86.5% displayed superior physicochemical properties compared to oseltamivir. After narrowing down the selection through computational filters, nine compounds with non-sialic acid-like structures were selected for in vitro experiments. We identified two compounds, DS-22-inf-009 and DS-22-inf-021 that effectively inhibited the NAs of both influenza A and B viruses (IAV and IBV), including H275Y mutant strains at low micromolar concentrations. Molecular dynamics simulations revealed a similar pattern of interaction with amino acid residues as oseltamivir. In cell-based assays, DS-22-inf-009 and DS-22-inf-021 inhibited IAV and IBV in a dose-dependent manner with EC50 values ranging from 0.29 uM
    
[^116]: 权重空间编程能为扩散模型编辑带来什么？程序构造与得分引导精细化之比较

    What Does Weight-Space Programming Add to Diffusion Editing? Program Construction versus Score-Guided Refinement

    [https://arxiv.org/abs/2611.22228](https://arxiv.org/abs/2611.22228)

    该研究发现，预注册对比下语言模型生成的权重空间程序并不优于强基线编辑器，但探索性实验表明一次性生成的程序可通过围绕受保护文本状态子空间组织更新方向，将扩散概念编辑在全新场景下的 VLM 联合成功率提升 10 分。

    

    权重空间编程让语言模型能够为另一个网络构建可执行的权重更新。我们通过一个接口来研究扩散概念编辑：程序定义训练上下文、保留约束和更新方向，而共享的数值拟合器负责选择其系数。我们在 Stable Diffusion 1.4 的 24 个 TIMED 测试编辑上，将生成的程序与已发表的编辑器、共享面板的手写基线以及预算匹配的传统编辑搜索进行比较。预注册的重写流水线相比强基线没有显示出优势。在探索性比较中，一次性生成的 Astra 程序在全新场景 VLM 联合成功率（要求主体出现且整图目标匹配）上比共享面板基线提升了 10.0 分，而 CLIP 相对匹配度则未检测到变化。这些程序围绕受保护的文本状态子空间来组织可调节的更新方向。

    arXiv:2611.22228v1 Announce Type: new  Abstract: Weight-space programming (WSP) lets a language model construct executable weight updates for another network. We study diffusion concept editing through an interface in which programs define training contexts, preservation constraints and update directions, while a shared numerical fitter selects their coefficients. We compare generated programs with published editors, a shared-panel hand-written baseline and budget-matched search over conventional edits on 24 TIMED test edits for Stable Diffusion 1.4. The preregistered rewrite pipeline shows no advantage over the strong baseline. In an exploratory comparison, one-shot Astra programs improve fresh-scene VLM joint success, which requires subject presence together with whole-image target matching, by +10.0 points over the shared-panel baseline, while CLIP relative match shows no detectable change. These programs organise adjustable directions around protected text-state subspaces. Probing 
    
[^117]: SOTA还是运气？LLM排行榜中的赢家诅咒

    SOTA or Luck? The Winner's Curse in LLM Leaderboards

    [https://arxiv.org/abs/2611.22227](https://arxiv.org/abs/2611.22227)

    该论文揭示了LLM排行榜中因排名噪声分数而产生的“赢家诅咒”与“排名诱导偏差”会系统性高估顶级模型的性能，并提出一种m-out-of-N自助法来纠正该偏差，在14个基准测试上排名第一模型的调整幅度为0.09至3.86个百分点。

    

    公共LLM排行榜指导着模型的评估与采用，然而排名靠前的模型所报告的性能可能被系统性高估。这种“排行榜赢家诅咒”源于对有噪声的基准测试分数进行排名时，会同时奖励采样运气和真实能力。同样的选择效应可能会扭曲整个排行榜上的分数，我们将这一现象称为“排名诱导偏差”。我们开发了一种m-out-of-N自助法（bootstrap）来纠正这种偏差。该方法假设模型池固定、任务独立同分布且有界，同时允许模型间在任务内存在任意相关性。在由正间隔分隔的精确总体层级下，我们证明了该自助法能将Θ(N^{-1/2})的排名偏差降低至期望残差偏差o(N^{-1/2})。在来自5个基准家族的14个基准测试上，排名第一模型的分数调整幅度约为0.09至3.86个百分点。在我们的SWE-bench Verified快照中，并列领先的模型

    arXiv:2611.22227v1 Announce Type: cross  Abstract: Public LLM leaderboards guide model evaluation and adoption, yet the reported performance of top-ranked models can be systematically overstated. This leaderboard winner's curse arises because ranking noisy benchmark scores rewards sampling luck alongside genuine ability. The same selection effect can distort scores throughout the leaderboard, a phenomenon we call the rank-induced bias. We develop an $m$-out-of-$N$ bootstrap to correct for this bias. It assumes a fixed model pool with independent, identically distributed bounded tasks while allowing arbitrary within-task correlation among models. Under exact population tiers separated by positive gaps, we prove that the bootstrap reduces $\Theta(N^{-1/2})$ ranking bias to expected residual bias $o(N^{-1/2})$. Across 14 benchmarks from 5 benchmark families, the rank-1 adjustments range from approximately 0.09 to 3.86 percentage points. In our SWE-bench Verified snapshot, the tied leaders
    
[^118]: 通过自回归标记回归学习最优传输方案

    Learning Optimal Transport Plans Via Autoregressive Token Regression

    [https://arxiv.org/abs/2611.22218](https://arxiv.org/abs/2611.22218)

    该论文提出ToR-OT，一种基于Transformer的自回归标记回归模型，将最优传输问题重构为结构化序列生成任务，直接解码稀疏传输方案而无需在测试时求解优化问题，并展现出零样本泛化到未见分布的能力。

    

    最优传输为寻找概率分布之间的最小成本耦合提供了一个有原则性的框架。经典求解器仍然是该任务的黄金标准。与此同时，OT是研究基于学习的摊销模型能否内化OT类问题结构的理想测试平台。在这一问题中，真值解是可获得的，边际约束被显式定义，且精确的离散解具有充分刻画的稀疏性。我们提出了ToR-OT，一个基于Transformer的模型，通过自回归标记回归学习预测离散OT方案。我们将OT重新构建为一个结构化序列生成任务，ToR-OT将稀疏传输方案解码为序列，而不是在测试时求解优化问题。我们关注三个性质：(a) 零样本泛化：在大型合成OT实例语料库上进行预训练，使单个模型能够泛化到未见过的分布……

    arXiv:2611.22218v1 Announce Type: cross  Abstract: Optimal Transport (OT) provides a principled framework for finding the minimum cost coupling between probability distributions. Classical solvers remain the gold standard for this task. At the same time, OT is an ideal testbed for studying whether learning-based amortized models can internalize the structure of OT type problems. Here, ground-truth solutions are available and marginal constraints are explicitly defined. Also, exact discrete solutions have well-characterized sparsity. We describe ToR-OT, a transformer-based model that learns to predict discrete OT plans through autoregressive token regression. We reframe OT into a structured sequence generation task. ToR-OT decodes sparse transport plans as sequences instead of solving an optimization task at test time. We focus on three properties: (a) Zero-shot generalization: pretraining on a large synthetic corpus of OT instances allows a single model to generalize to unseen distribu
    
[^119]: 传递变化：通过激活压缩实现高效专家并行

    Communicate the Change: Efficient Expert Parallelism via Activation Compression

    [https://arxiv.org/abs/2611.22215](https://arxiv.org/abs/2611.22215)

    EchoMoE 利用 transformer 残差流以增量方式更新的特性，通过激活压缩只传递变化量而非完整隐藏状态，从而大幅降低专家并行中的同步集合通信瓶颈，且改造现有 MoE 模型仅需约完整预训练 1/8500 的训练 token。

    

    规模化扩展推动了大语言模型（LLM）的成功，稀疏混合专家架构使模型能够获得更大的参数容量，而无需成比例地增加计算量。在大规模场景下，MoE 模型依赖专家并行将专家分布到多个设备上。每个 MoE 层在推理过程中需要两个同步的集合通信步骤：将隐藏状态分发到被激活的专家，以及合并各专家的输出结果。随着计算吞吐量的增长速度超过互连带宽的增长，这些同步集合通信已成为 MoE 扩展与效率的主要瓶颈。我们提出 EchoMoE 来应对这一挑战：这是一种轻量级的架构修改，可以改造现有的 MoE 模型，所需训练 token 数量约为完整预训练的 1/8500。受 transformer 块以增量方式更新残差流这一观察的启发，我们利用较早的隐藏状态……（原文摘要在此处被截断）

    arXiv:2611.22215v1 Announce Type: cross  Abstract: Scaling has driven the success of large language models (LLM), with sparse mixture-of-experts (MoE) architectures enabling greater parameter capacity without a proportional increase in computation. At large scales, MoE models rely on expert parallelism to distribute experts across devices. Each MoE layer requires two synchronous collective communication steps during inference: dispatching hidden states to the activated experts and combining the resulting expert outputs. As compute throughput has outpaced interconnect bandwidth, these synchronous collectives have become a major bottleneck to MoE scaling and efficiency. We address this challenge with EchoMoE, a lightweight architecture modification that retrofits existing MoE models, requiring approximately 8,500x fewer training tokens than full pretraining. Motivated by the observation that transformer blocks update the residual stream incrementally, we use a hidden state from an earlie
    
[^120]: 基于多维用户偏好数据的大语言模型对齐的样本高效个性化

    Sample-efficient Personalization of Large Language Model Alignment with Multidimensional User Preference Data

    [https://arxiv.org/abs/2611.22210](https://arxiv.org/abs/2611.22210)

    提出了多维偏好引导的在线直接偏好优化方法MPG-DPO，通过利用多维用户偏好数据来指导样本选择，解决了现有方法从LLM视角而非用户视角量化不确定性所导致的低效问题，实现了大语言模型对齐的样本高效个性化。

    

    偏好对齐已成为使大语言模型（LLM）与人类价值观和期望行为保持一致的核心工具，其在线变体通过反复生成新的回复并利用人类反馈进行标注，能够进一步提升模型超越静态数据所能提供的能力。然而，昂贵且耗时的标注限制了其可扩展性，使得样本效率变得至关重要。现有方法通常通过扩大回复空间的覆盖范围、或选择不确定性最高的样本对进行比较来提升样本效率。尽管这些方法行之有效，但它们是从LLM自身的视角来量化不确定性，而这可能与用户所感知的不确定性不同：即使过去的用户比较已经隐含了明确的偏好排序，某个样本对在LLM看来仍可能具有高度的不确定性。在本文中，我们提出了多维偏好引导的在线直接偏好优化方法（MPG-DPO），它利用……（摘要在此处截断）

    arXiv:2611.22210v1 Announce Type: new  Abstract: Preference alignment has become a central tool to align large language models (LLMs) with human values and desired behavior, and its online variant can further advance their capabilities beyond what static data alone offer by repeatedly generating and labeling new responses with human feedback. But costly, time-consuming annotations limit its scalability, making sample efficiency essential, and existing methods typically seek to improve it by expanding coverage of the response space or selecting pairs with the highest uncertainty for comparison. Although they are effective, these methods quantify uncertainty from the perspective of the LLM, which may differ from the uncertainty perceived by the user: a pair can appear highly uncertain to the LLM even when past comparisons from the user already imply a clear ordering. In this paper, we propose multidimensional preference-guided online direct preference optimization (MPG-DPO), which levera
    
[^121]: 动态网络非线性Cucker-Smale模型中的群集时间与图体积

    Flocking Time and Graph Volume in Nonlinear Cucker-Smale Models with Dynamic Networks

    [https://arxiv.org/abs/2611.22204](https://arxiv.org/abs/2611.22204)

    本研究提出了结合动态网络与离散p-拉普拉斯算子的非线性Cucker-Smale模型，证明了其有限时间群集性并建立了群集时间下界与图体积的定量关系，同时通过基于transformer的PPO强化学习实验验证了严格时间约束下网络连接数增长的阈值效应。

    

    本研究提出了一种融合动态网络和离散p-拉普拉斯算子（p ∈ (1,2)）的Cucker-Smale模型，并从理论和实验两个角度分析了群集时间约束与图连通性之间的关系。我们证明该模型能够在有限时间内实现群集，并推导出群集时间下界与图体积之间的定量关系。基于这些结果，我们观察到施加足够小的群集时间约束会诱导网络体积的增加，并且在相同时间约束下，较大的初始速度方差需要更大的图体积。在实验中，我们训练了基于transformer的PPO策略来在每个时间步选择网络结构。在最严格的时间约束下，最大平均连接数急剧增加，证实了阈值效应的存在。收敛时间与平均……（摘要被截断）

    arXiv:2611.22204v1 Announce Type: cross  Abstract: This study proposes a Cucker-Smale model incorporating dynamic networks and the discrete $p$-Laplacian ($p \in (1,2)$) and analyzes the relationship between flocking time constraints and graph connectivity from both theoretical and experimental perspectives. We establish that this model achieves flocking within finite time and derive a quantitative relationship between the lower bound of flocking time and graph volume. Based on these results, we observe that imposing sufficiently small flocking time induces an increase in network volume, and that larger initial velocity variance requires greater volume under the same time constraint. In our experiments, we trained a transformer-based PPO policy to select network structures at each time step. Under the most stringent time constraint, the maximum average number of connections increased sharply, confirming a threshold effect. The negative correlation between convergence time and average n
    
[^122]: 去噪生成的优化框架

    An Optimization Framework for Denoising Generation

    [https://arxiv.org/abs/2611.22202](https://arxiv.org/abs/2611.22202)

    该论文提出将去噪生成视为在固定对数密度景观上的梯度上升优化框架，并从理论上证明了上升轨迹几乎必然收敛到参考数据流形附近、不同初始化可保持输出多样性，且在特定正带宽条件下即使使用经验分数也能避免对训练样本的记忆化。

    

    无噪声条件化的去噪学习单一的分数函数，因此生成过程可被视为在固定的对数密度景观上进行梯度上升。这一优化视角引出了两个问题：上升轨迹是否会接近有意义的数据？不同的初始化是否会保留多样化的结果？我们针对经验版本与总体版本的多尺度高斯混合目标证明了这两种现象。在显式的后验与景观条件下，梯度上升几乎必然收敛到距离一个 m 维参考流形 $O(m\sigma_{\min}^2/\tau)$ 的范围内，该流形的可达半径至少为 τ。我们针对同一固定场推导出源—汇恒等式，并用它证明了有限分辨率下的区域覆盖。在一个仿射构造中，我们进一步识别出一个正带宽区域，在该区域中，即使是经验分数优化也能避免对训练样本的记忆化，同时在训练分辨率下保持多样性。同一固定目标自然支持惩罚引导的……（摘要原文在此处截断）

    arXiv:2611.22202v1 Announce Type: new  Abstract: Noise-unconditioned denoising learns a single score, so generation can be viewed as gradient ascent on a fixed log-density landscape. This optimization interpretation leaves two questions: do trajectories approach meaningful data, and do different initializations retain diverse outcomes? We establish both phenomena for empirical and population multi-scale Gaussian-mixture objectives. Under explicit posterior and landscape conditions, ascent converges almost surely within $O(m\sigma_{\min}^2/\tau)$ of an $m$-dimensional reference manifold with reach at least $\tau$. We derive a source--sink identity for the same fixed field and use it to prove finite-resolution regional coverage. In an affine construction, we further identify a positive-bandwidth regime where even empirical-score optimization can avoid training-sample memorization while preserving diversity at the training resolution. The same fixed objective naturally supports penalty-gu
    
[^123]: RISA：面向块扩散语言模型的可恢复稀疏注意力

    RISA: Recoverable Sparse Attention for Block-Diffusion Language Models

    [https://arxiv.org/abs/2611.22191](https://arxiv.org/abs/2611.22191)

    提出RISA方法，通过为块扩散语言模型设计带共享读取计划的可恢复稀疏注意力机制，解决长上下文推理中KV缓存读取带来的内存瓶颈问题。

    

    块状扩散语言模型通过并行细化多个token，为逐token的自回归解码提供了一种替代方案。然而，其长上下文推理仍然受制于内存瓶颈的注意力机制：每个去噪步骤都会重复读取不断增长的历史键值（KV）缓存。直接应用稀疏注意力并不能完全解决这一问题，因为不同的查询可能选择不同的页面，从而膨胀块的集体读取集合；同时，重复构建支持集会带来额外开销，而永久性淘汰则可能丢弃后续所需的信息。我们提出了RISA，一种面向块状扩散语言模型、具有共享读取计划的可恢复稀疏注意力方法。RISA利用了这样一个事实：历史KV在块内保持固定，而查询则随去噪步骤不断演化。新生成的历史被组织为精确优先层与恢复层；随后由种子查询构建一个受保护的、有界的读取计划，其……（原文摘要在此处截断）

    arXiv:2611.22191v1 Announce Type: new  Abstract: Block-wise diffusion language models (DLMs) refine multiple tokens in parallel, offering an alternative to token-by-token autoregressive decoding. Their long-context inference, however, remains bottlenecked by memory-bound attention: every denoising step repeatedly reads a growing historical key--value (KV) cache. Directly applying sparse attention does not fully solve this problem, because different queries can select different pages, inflating the block's collective read set, while repeated support construction adds overhead and permanent eviction can discard information needed later. We present RISA, a recoverable sparse-attention method with shared read plans for block-wise DLMs. RISA exploits the fact that historical KV remains fixed within a block while queries evolve across denoising steps. Newly generated history is organized into exact priority and recovery tiers; seed queries then build a protected, bounded read plan whose indi
    
[^124]: 面向基于模型的离线到在线强化学习中的乐观-悲观权衡研究

    Towards Optimism-Pessimism Trade-off in Model-based Offline-to-Online Reinforcement Learning

    [https://arxiv.org/abs/2611.22172](https://arxiv.org/abs/2611.22172)

    该论文提出一种双目标形式化方法，在离线训练阶段构建帕累托策略池，使在线微调时能够针对不同在线任务自适应地选择策略，从而解决基于模型的离线到在线强化学习中乐观与悲观的权衡难题。

    

    基于模型的离线到在线强化学习（RL）通过利用离线预训练进行在线微调，实现了样本高效的适应。然而，离线与在线阶段之间的分布偏移常常阻碍微调性能。许多现有方法通过单目标形式来编码乐观-悲观权衡，而通过在线微调来选择其权衡系数可能需要代价高昂的双层优化。我们识别出离线训练期间建立的平衡是一个关键挑战：在我们的动机实验中，乐观策略能更好地适应新任务，而悲观策略在相同的在线任务上表现更好。为应对这一挑战，我们提出了一种捕捉这种权衡的双目标形式，在离线训练期间产生一个帕累托策略池。该策略池使得在线微调期间能够跨多种不同的在线任务进行自适应策略选择。

    arXiv:2611.22172v1 Announce Type: new  Abstract: Model-based offline-to-online reinforcement learning (RL) enables sample-efficient adaptation by leveraging offline pre-training for online fine-tuning. However, the distribution shifts between offline and online stages often hinder fine-tuning performance. Many existing methods encode the optimism-pessimism trade-off through a single-objective formulation. Selecting its trade-off coefficient through online fine-tuning can require costly bi-level optimization. We identify the balance established during offline training as a key challenge: in our motivating experiments, optimistic policies adapt better to novel tasks, whereas pessimistic policies perform better on the same online tasks. To address this challenge, we propose a bi-objective formulation that captures this trade-off, yielding a Pareto policy pool during offline training. This policy pool enables adaptive selection during online fine-tuning across diverse online tasks. To gene
    
[^125]: 专家构建任务上的强化学习教会了 Nemotron 3.5 Super 什么

    What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super

    [https://arxiv.org/abs/2611.22161](https://arxiv.org/abs/2611.22161)

    通过在五个专家构建的任务数据集上分别进行无监督阶段的强化学习后训练，NVIDIA Nemotron 3.5 Super 在十二项基准测试上取得了最高 +13.8 分的显著性能提升。

    

    我们对 NVIDIA 的 Nemotron 3.5 Super（总参数量 120B，激活参数 12B）进行强化学习（RL）后训练，使用五个专家构建的任务数据集，每次训练一个：真实世界软件工程、复杂约束、长程工具使用、专业图表理解和企业文档推理。每次运行都在 Tinker 上使用 CISPO 训练一个全新的 rank-32 LoRA 适配器，不设监督学习阶段，并在相同的任务、测试框架和设置下将每个适配器与基础模型进行比较。在十二个主要基准测试上（其中八个由其他团队构建，四个由 Surge AI 构建），提升幅度从 IFBench 上的 +4.5 分到 GDP_pdf 上的 +13.8 分不等（后者从 10.4% 提升至 24.2%）；SWE-bench Pro V2 从 86.9% 提升至 92.1%，Terminal-Bench 2.1 从 63.9% 提升至 70.3%，MultiChallenge 从 58.3% 提升至 66.6%，AutomationBench 从 14.8% 提升至 27.5%，Chartography 从 39.4% 提升至 50.6%。文档适配器在没有 API 工具的情况下训练，也在 AutomationBench 上获得了 +8.0 分的提升。

    arXiv:2611.22161v1 Announce Type: cross  Abstract: We post-train NVIDIA's Nemotron 3.5 Super (120B total, 12B active parameters) with reinforcement learning (RL) on five expert-built task datasets, one at a time: real-world software engineering, complex constraints, long-horizon tool use, professional chart understanding and enterprise document reasoning. Each run trains a fresh rank-32 LoRA adapter with CISPO on Tinker, with no supervised stage, and each adapter is compared with the base model on the same tasks, harness and settings. On twelve headline benchmarks, eight built by other groups and four by Surge AI, the gains range from +4.5 points on IFBench to +13.8 on GDP_pdf, which rises from 10.4% to 24.2%; SWE-bench Pro V2 rises from 86.9% to 92.1%, Terminal-Bench 2.1 from 63.9% to 70.3%, MultiChallenge from 58.3% to 66.6%, AutomationBench from 14.8% to 27.5% and Chartography from 39.4% to 50.6%. The document adapter, trained without API tools, also gains +8.0 points on AutomationB
    
[^126]: 质心谱耦合：自监督学习中的隐式对齐与表示多样性

    Centroid Spectral Coupling: Implicit Alignment and Representation Diversity in Self-Supervised Learning

    [https://arxiv.org/abs/2611.22158](https://arxiv.org/abs/2611.22158)

    该论文提出质心谱耦合（CSC）——一种单项自监督学习目标函数，通过基于质心二阶矩矩阵的谱优化，隐式地同时实现视图对齐、表示多样性和抗坍缩，并给出精确分解与理论保证。

    

    自监督学习（SSL）旨在对齐增强视图，同时保持表示多样性并防止坍缩。现有的非对比方法通常使用多个加权损失项来实现这些目标。我们提出了质心谱耦合（CSC），这是一个单项的SSL目标函数，它隐式地将视图对齐、表示多样性和抗坍缩能力耦合在一起。CSC从独立归一化的增强视图对构建质心，并对其未中心化的二阶矩矩阵应用基于岭正则化的谱目标。保持质心的幅值编码了视图之间的一致性，同时促使表示能量在各个谱方向上均衡分布。我们建立了一个精确的分解，将其分解为非负的对齐分量与谱不平衡分量，刻画了全局最优解的几何性质，并推导出对齐与多样性的有限损失保证。我们还证明了完全恒定输出的坍缩是一个……（原文摘要在此处截断）

    arXiv:2611.22158v1 Announce Type: cross  Abstract: Self-Supervised Learning (SSL) aims to align augmented views while preserving representation diversity and preventing collapse. Existing non-contrastive methods often address these goals using multiple weighted loss terms. We introduce Centroid Spectral Coupling (CSC), a single-term SSL objective that implicitly couples view alignment, representation diversity, and collapse resistance. CSC constructs centroids from independently normalized augmented-view pairs and applies a ridge-based spectral objective to their uncentered second-moment matrix. Preserving centroid magnitudes encodes view agreement while encouraging balanced representation energy across spectral directions. We establish an exact decomposition into nonnegative alignment and spectral-imbalance components, characterize the geometry of global optima, and derive finite-loss guarantees for alignment and diversity. We also show that complete constant-output collapse is a stri
    
[^127]: 深度强化学习在暖通空调（HVAC）监督控制中何时才有收益？四个直接在FMU上训练的小型Modelica案例研究

    When does deep reinforcement learning pay off in HVAC supervisory control? Four small Modelica case studies trained directly on FMUs

    [https://arxiv.org/abs/2611.22144](https://arxiv.org/abs/2611.22144)

    该论文通过四个基于Modelica模型并直接在FMU上训练SAC智能体的小型可复现HVAC案例研究，与调优的规则控制器及近最优基准进行对比，阐明了深度强化学习在HVAC监督控制中何时才真正划算。

    

    深度强化学习（DRL）常被提议用于供暖、通风与空调（HVAC）系统的监督控制，但其所报告的收益难以解读，因为所用的基准控制器往往较弱，且可达到的最优解鲜为人知。本文基于澳大利亚布里斯班的一栋办公楼开展了四个小型、可复现的案例研究：分时电价下的预冷、带冷凝器水温重置的冷水机组台数控制、带湿式冷却盘管的冷冻水供水温度重置，以及分时电价与需量电费下的冷冻水蓄冷。这些被控对象采用Modelica Buildings库建模，其中三个案例中的软演员-评论家（soft actor-critic）智能体直接在导出的功能样机单元（FMU）上进行训练。每个智能体均与经过调优的基于规则的控制器进行比较，并在可行的情况下，与通过逐时穷举搜索或动态规划获得的近最优参考基准进行比较。

    arXiv:2611.22144v1 Announce Type: cross  Abstract: Deep reinforcement learning (DRL) is often proposed for the supervisory control of heating, ventilation and air-conditioning (HVAC) systems. The reported gains are, however, difficult to interpret, because the baselines are often weak and the attainable optimum is seldom known. This paper examines four small, reproducible case studies based on an office building in Brisbane, Australia: pre-cooling under a time-of-use tariff; chiller staging with condenser water reset; chilled water supply temperature reset with a wet cooling coil; and chilled water storage under time-of-use prices and a demand charge. The plants were modelled with the Modelica Buildings library, and in three of the cases a soft actor-critic agent was trained directly on the exported functional mock-up unit (FMU). Each agent was compared with tuned rule-based controllers and, where practicable, with a near-optimal reference from hourly exhaustive search or dynamic progr
    
[^128]: 面向材料性质预测中扩散预训练的合成晶体数据生成

    Synthetic Crystal Data Generation for Diffusion Pretraining in Material Property Prediction

    [https://arxiv.org/abs/2611.22140](https://arxiv.org/abs/2611.22140)

    该论文提出一种结合扰动-弛豫、MatterGen生成式采样等三种互补策略的合成晶体数据生成流水线，为材料性质预测的扩散预训练提供大规模且结构多样的弛豫结构，突破了依赖昂贵DFT弛豫获取预训练数据的瓶颈。

    

    基于扩散的预训练已成为图神经网络中学习可迁移表示的有力策略，可用于材料性质预测，而无需为每个预训练结构提供诸如能量、力和应力等量子力学标签。然而，该方法仍存在一个关键依赖：它要求结构位于势能面的局部极小值附近，而这传统上需要通过对已知或物理合成的化合物进行昂贵的DFT弛豫来获得。这种依赖限制了可用预训练数据的规模和结构多样性，使得晶体结构空间的大量区域未被探索。在这项工作中，我们将基于扩散的预训练与合成结构生成相结合，以缓解这一瓶颈。我们提出了一个合成数据生成流水线，结合了三种互补策略：扰动-弛豫（perturb-and-relax）、通过MatterGen进行的生成式采样……

    arXiv:2611.22140v1 Announce Type: new  Abstract: Diffusion-based pretraining has emerged as a powerful strategy for learning transferable representations in graph neural networks for materials property prediction without requiring quantum-mechanical labels such as energies, forces, and stresses for every pretraining structure. However, the method retains a critical dependency: it requires structures residing near local minima of a potential energy surface, which have traditionally been obtained through costly DFT relaxations of known or physically synthesized compounds. This reliance constrains both the scale and the structural diversity of available pretraining data, leaving large regions of crystal structure space unexplored. In this work, we connect diffusion-based pretraining with synthetic structure generation to reduce this bottleneck. We introduce a synthetic data generation pipeline combining three complementary strategies: perturb-and-relax, generative sampling via MatterGen, 
    
[^129]: ReMoE：一个用于跨灾害人类流动性恢复预测的可解释混合专家框架

    ReMoE: An Interpretable Mixture-of-Experts Framework for Human Mobility Recovery Forecasting Across Disasters

    [https://arxiv.org/abs/2611.22133](https://arxiv.org/abs/2611.22133)

    提出了ReMoE，一个可解释的混合专家框架，通过特征引导的路由器自适应组合学习异构且可复用恢复类型的专家，实现跨灾害场景下的人类流动性恢复预测。

    

    灾害发生后，人类流动性往往会急剧下降，随后表现出复杂的恢复动态。有效预测这种恢复过程对于及时的灾情评估和资源分配至关重要。然而，现有方法通常针对单个事件开发，且以黑盒方式运行，限制了其在训练事件之外的迁移性和可解释性。近期研究已识别出跨灾害中反复出现的灾后流动性恢复类型，表明恢复知识可能是可迁移的。受此启发，我们提出了ReMoE，一个用于跨灾害流动性恢复预测的可解释混合专家框架。ReMoE使用特征引导的路由器，自适应地组合专门学习这些异构但可复用恢复类型的专家，从而生成特定区域的恢复模式。它还进一步通过基于预测的残差校正来细化该恢复模式。

    arXiv:2611.22133v1 Announce Type: new  Abstract: Human mobility often declines sharply following disasters and subsequently exhibits complex recovery dynamics. Effective forecasting of this recovery is critical for timely assessment and resource allocation. However, existing approaches are often developed for individual events and operate as black boxes, limiting their transferability and interpretability beyond the events they were trained on. Recent studies have identified recurring post-disaster mobility recovery types across disasters, suggesting that recovery knowledge may be transferable. Motivated by this, we propose ReMoE, an interpretable mixture-of-experts framework for cross-disaster mobility recovery forecasting. ReMoE uses a feature-guided router to adaptively combine specialized experts that learn these heterogeneous yet reusable recovery types, producing a region-specific recovery pattern. It further refines this pattern through residual correction informed by forecasts 
    
[^130]: LoCaR：基于冻结视觉-语言模型的局部画布检索

    LoCaR: Localized Canvas Retrieval With Frozen Vision-Language Model

    [https://arxiv.org/abs/2611.22131](https://arxiv.org/abs/2611.22131)

    LoCaR 是一个无需训练的局部化文本到图像检索框架，利用 SAM2 生成的目标候选区域和冻结视觉编码器对每个区域独立编码，从而有效提升对小物体和局部属性的检索能力，并引入 LOCFG 基准数据集进行评估。

    

    整幅图像的嵌入表示可能无法充分表征那些在局部查询中用于区分相关图像的小物体和局部属性。我们提出了 LoCaR，这是一个无需训练的框架，用于在冻结的视觉-语言模型上进行局部化的文本到图像检索。LoCaR 构建并利用聚焦于目标的独立候选区域（SAM2），通过冻结的视觉编码器对每个区域进行独立编码，从而获得更局部化的表示。我们进一步研究了配对支持适配，作为一种可选的残差方法，它利用匹配的文本-区域示例而无需更新底层编码器。我们还引入了 LOCFG，这是一个源自 Visual Genome 的基准数据集，旨在评估针对更小、更具挑战性指代对象的局部化检索。在多个视觉-语言骨干网络上的结果表明，改进区域表示本身是性能提升的主要来源。

    arXiv:2611.22131v1 Announce Type: new  Abstract: Whole image embeddings can under represent small objects and local attributes that distinguish relevant images for localized queries. We present LoCaR, a training free framework for localized text to image retrieval with frozen vision language models. LoCaR constructs and utilizes independent proposals (SAM2) focusing on the object so as to get a more localized representation that independently encodes each region using the frozen visual encoder. We further study paired-support adaptation as an optional residual that uses matched text region exemplars without updating the underlying encoders. We also introduce LOCFG, a Visual Genome derived benchmark designed to evaluate localized retrieval for smaller and more challenging referents. Across several vision language backbones, our results show that improving the region representation itself is the main source of gain.
    
[^131]: 使用零阶预言机的Bandit在线次模最小化

    Bandit Online Submodular Minimisation Using Zeroth-Order Oracles

    [https://arxiv.org/abs/2611.22130](https://arxiv.org/abs/2611.22130)

    本文提出了首个在bandit反馈下可保证跟踪移动最小化元的在线次模最小化零阶方法，其基于Lovász扩展的无偏两点梯度估计并建立了动态遗憾界。

    

    我们考虑在bandit反馈下的在线次模最小化问题：在每一轮中，决策者选择一个子集，但仅能观测到该子集的代价，而代价函数（因而其最小化元）会随时间变化。现有的bandit方法只能保证静态遗憾，即与单个固定子集进行比较，当最小化元随时间移动时，这是一个较弱的基准。据我们所知，本文提供了在该设置下首个能够保证跟踪移动最小化元的方法。我们提出了一种基于Lovász扩展两点预言机的零阶方法，其两次评估均为可实际执行的子集。扰动点与基准点通过一个共同的随机阈值进行舍入，而从单位球面上抽取平滑方向使得基准评估成为一个零均值控制变量，因此该预言机是平滑梯度的无偏估计。我们建立了数量级为O((1+...)的动态遗憾界。

    arXiv:2611.22130v1 Announce Type: cross  Abstract: We consider online submodular minimisation under bandit feedback, where at each round a decision maker plays a subset and observes only its cost, while the cost functions, and hence their minimiser, change over time. Existing bandit methods only guarantee static regret, which compares against a single fixed subset and is a weak benchmark when the minimiser moves. This paper provides, to the best of our knowledge, the first method with guarantees for tracking a moving minimiser in this setting. We propose a zeroth-order method built on a two-point oracle for the Lov\'asz extension whose two evaluations are both subsets that can be played. The perturbed and base points are rounded by a common random threshold, and drawing the smoothing direction from the unit sphere makes the base evaluation a zero-mean control variate, so the oracle is an unbiased estimator of the smoothed gradient. We establish dynamic regret bounds of order $O\big((1+
    
[^132]: TempQuant：为量化语言模型学习自适应温度

    TempQuant: Learning Adaptive Temperatures for Quantized Language Models

    [https://arxiv.org/abs/2611.22128](https://arxiv.org/abs/2611.22128)

    TempQuant通过学习量化模型的采样温度和top-p设置，使其下一词元概率分布更接近全精度模型，从而有效缓解量化带来的性能损失，且温度对量化模型性能的影响远大于全精度模型。

    

    大语言模型越来越多地以低精度部署，但量化可能会降低其性能。我们通过考察量化如何扭曲模型的下一词元概率，研究了量化与解码设置（如采样温度和top-p）之间的相互作用。通过在部署的各种温度下测量生成过程中的熵，我们发现量化平均上会使模型“变暖”，且温度对量化模型性能的影响远大于对全精度模型的影响。由于量化和温度都会塑造下一词元的分布，我们通过我们的方法TempQuant来探究采样设置能在多大程度上抵消量化带来的变化——该方法通过学习量化模型的温度和top-p，使其下一词元概率更接近全精度模型。我们在两个粒度上学习这些设置：第一种是针对每个请求温度学习单一的全局设置……（原文摘要在此处被截断）

    arXiv:2611.22128v1 Announce Type: new  Abstract: Large language models are increasingly deployed at low precision, yet quantization can degrade their performance. We study how quantization interacts with decoding settings such as the sampling temperature and top-p by examining how it distorts the model's next-token probabilities. Measuring entropy during generation across deployed temperatures, we find that quantization warms the model on average, and that temperature matters much more for the performance of a quantized model than for its full-precision counterpart. Because quantization and temperature both shape the next-token distribution, we investigate how much the sampling settings can undo what quantization changes, through our method TempQuant, which learns the quantized model's temperature and top-p to bring its next-token probabilities closer to the full-precision model's. We learn these settings at two granularities. The first is a single global setting per requested temperat
    
[^133]: MoE路由控制着什么：对专家轨迹的因果干预

    What MoE Routing Controls: Causal Interventions on Expert Trajectories

    [https://arxiv.org/abs/2611.22120](https://arxiv.org/abs/2611.22120)

    该论文通过对MoE路由器logit进行因果干预发现：引导“格式”专家路径可以实际改变模型生成的文本，而引导“概念”专家路径仅能改变下一词元概率分布却几乎不影响最终生成内容，从而揭示了专家路由对模型输出控制的层级差异。

    

    在混合专家语言模型中，一个词元会跨层被路由到一系列专家。我们研究将词元强制引导到选定的专家路径上会如何影响模型行为。我们根据某个概念或角色的词元最常使用的专家来构建一条目标路径，并在生成过程中给每个目标专家的路由器logit添加固定偏置，从而将词元引导至这条专家路径。我们研究了两种目标：格式目标关注输出的形式，概念目标关注其主题。在Qwen3-30B-A3B模型上，我们表明引导格式路径会改变模型写出的文本。我们还表明，引导概念路径会使模型的下一词元概率向该概念偏移，即使该路径是在另一种语言中构建的也是如此，但这种偏移对生成的文本影响很小。我们通过仅对生成的部分阶段进行引导来探究为什么格式能够影响文本而概念不能，并发现了形式……（摘要原文在此处截断）

    arXiv:2611.22120v1 Announce Type: new  Abstract: In mixture-of-experts (MoE) language models, a token is routed to a sequence of experts across layers. We investigate how forcing tokens onto chosen expert paths influences model behavior. We build a target path from the experts that a concept's or role's tokens most often use, and we add a fixed bias to each target expert's router logit during generation to steer the token toward this expert path. We study two kinds of targets. Format targets concern the form of the output, and concept targets concern its topic. In Qwen3-30B-A3B, we show that steering a format path changes the text the model writes. We also show that steering a concept path shifts the model's next-token probabilities toward the concept, even when the path is built in another language, yet this shift has little effect on the generated text. We investigate why formats reach the text but concepts do not by steering only part of the generation, and find evidence that a form
    
[^134]: VTT：从未见车型运动轨迹中读取车辆尺寸与动力系统类型

    VTT: Reading Vehicle Size and Powertrain Type from Motion Trajectory Across Unseen Car Models

    [https://arxiv.org/abs/2611.22109](https://arxiv.org/abs/2611.22109)

    该论文提出VTT数据集与基准，可从未见车型的运动轨迹和CAN信号中识别车辆尺寸与动力系统类型，通过辅助物理目标、数据增强和任务加权将纯电动与燃油车判别的宏F1提升至93%、尺寸分类提升至67.1%。

    

    车辆尺寸和动力系统信息能够丰富用于交通分析与仿真的轨迹数据。我们提出了VTT，这是一个用于从未见过的车型中、基于运动信号与控制器局域网（CAN）信号识别这些属性的数据集和基准。该数据集包含来自1,431辆车、174个车型的9,062次行驶数据，总计约3,950小时。数据集提供了基于占地面积的三个尺寸类别、动力系统标签，以及针对未见车型和未见驾驶员的评估。使用60秒时间窗口，我们比较了四种序列架构、统计特征以及由大语言模型提出的物理特征。通过辅助物理目标、数据增强和任务特定加权进行训练，纯电动与内燃机车型判别的行驶级宏F1提升至92.6-93.0%，尺寸分类性能从62.9%提升至67.1%。在来自4,061名额外驾驶员的公开日志上，全输入模型在88个车型中正确识别出86个车型的动力系统。

    arXiv:2611.22109v1 Announce Type: new  Abstract: Vehicle size and powertrain information can enrich trajectory data for traffic analysis and simulation. We present VTT, a dataset and benchmark for identifying these attributes from motion and controller area network (CAN) signals across unseen car models. The dataset contains 9,062 drives, about 3,950 h, from 1,431 vehicles and 174 car models. It provides three footprint-based size classes, powertrain labels, and evaluation on unseen car models and unseen drivers. Using 60-s windows, we compare four sequence architectures, statistical features and physical features proposed by large language models. Training with auxiliary physical targets, data augmentation and task-specific weighting raises drive-level macro-F1 to 92.6-93.0% for battery-electric versus combustion vehicles and from 62.9% to 67.1% for size classification. On public logs from 4,061 additional drivers, the full-input model identifies powertrain correctly for 86 of 88 car 
    
[^135]: 所有测量、所有方向、一次完成：面向分子与光谱的多模态流匹配

    Every Measurement, Every Direction, All at Once: Multimodal Flow Matching for Molecules and Spectra

    [https://arxiv.org/abs/2611.22098](https://arxiv.org/abs/2611.22098)

    本文提出MOSAIC，一个多模态流匹配Transformer模型，通过对测量数据施加随机掩码，使分子三维结构与UV/IR/拉曼光谱之间实现任意到任意的双向生成，并在QM9S上达到88.8%的分子图恢复率，证明了单次实验可被充分挖掘为多方向的监督信号。

    

    科学领域的生成式建模常常受到数据限制，因为实验成本高昂。然而，每次实验可能对同一物理实体产生多个相互关联的测量结果。标准的生成模型通常只在单一方向上训练，丢弃了其余方向的监督信息。我们认为，每次实验所包含的训练信号远超这种不对称性允许模型利用的程度：随机掩码使测量结果既可以作为条件输入，也可以作为预测目标，从而从单次实验中获得多个有监督的去噪任务。我们通过MOSAIC来验证这一前提，MOSAIC是一个覆盖分子三维结构以及紫外、红外和拉曼光谱的多模态流匹配Transformer模型。MOSAIC是一个能够进行任意到任意生成的单一模型：即从其他模态的任意不相交子集生成任意模态子集。在QM9S数据集上，MOSAIC实现了88.8%的分子图恢复率，超越了最强的光谱到结构（模型）（摘要原文在此处截断）。

    arXiv:2611.22098v1 Announce Type: new  Abstract: Generative modeling in science is often data-limited, where experiments are expensive. However, each experiment may yield several coupled measurements of the same physical entity. Standard generative models often train only on a single direction, discarding supervision from the rest. We argue that each experiment contains more training signal than this asymmetry allows the model to exploit: random masking enables measurements to serve as either conditioning inputs or prediction targets, yielding multiple supervised denoising tasks from a single experiment. We investigate this premise with MOSAIC, a multimodal flow matching transformer over molecular 3D structure and UV, IR, and Raman spectra. MOSAIC is a single model that is capable of any-to-any generation: generating any subset of modalities from any disjoint subset of the others. On QM9S, MOSAIC achieves 88.8% molecular graph recovery, outperforming the strongest spectra-to-structure 
    
[^136]: 当AUC遇上生成式重建：平衡视觉表征学习的统一原则

    When AUC Meets Generative Reconstruction: A Unified Principle for Balanced Visual Representation Learning

    [https://arxiv.org/abs/2611.22094](https://arxiv.org/abs/2611.22094)

    该论文提出将生成式重建重新表述为判别排序问题的统一表征学习范式，解决了对比学习与生成重建之间的目标冲突，实现了细粒度感知与判别能力兼顾的平衡视觉表征学习。

    

    多模态大语言模型（MLLMs）需要视觉编码器在提取细粒度视觉特征的同时，构建语义解耦且具有判别性的表征。传统方法面临一个根本性的权衡：对比学习通过抑制类内变异性来增强类间判别能力，而生成式重建强调感知保真度，但缺乏显式的对比监督。通过复合损失函数简单地将这些目标聚合在一起，往往会由于共享潜在空间中的梯度干扰而引发优化冲突和不稳定性。为解决这一问题，我们引入了一种统一的表征学习范式，将生成式重建重新表述为生成输出流形上的判别排序问题。我们的方法将感知重建保留为监督信号，并集成了基于排序的判别……（摘要原文在此处截断）

    arXiv:2611.22094v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) require visual encoders that extract fine-grained visual features while constructing semantically disentangled and discriminative representations. Conventional methodologies encounter a fundamental trade-off: contrastive learning enhances inter-class discrimination by suppressing intra-class variability, whereas generative reconstruction emphasizes perceptual fidelity but lacks explicit comparative supervision. The naive aggregation of these objectives via composite loss functions often induces optimization conflicts and instability due to gradient interference within the shared latent space. To resolve this, we introduce a unified representation-learning paradigm that reformulates generative reconstruction as a discriminative ranking problem within the generative output manifold. Our approach maintains perceptual reconstruction as a supervisory signal and integrates a ranking-based discriminati
    
[^137]: BernBRS：深度伯恩斯坦神经网络的认证后向可达性分析

    BernBRS: Certified Backward Reachability for Deep Bernstein Neural Networks

    [https://arxiv.org/abs/2611.22083](https://arxiv.org/abs/2611.22083)

    本文提出BernBRS框架，通过分析激活模式对输入空间覆盖进行智能细化与精确分割，为深度伯恩斯坦神经网络同时计算后向可达集的认证内、外近似，实现了高精度且可扩展的后向可达性分析。

    

    确保复杂神经网络系统的安全性需要数学上严格的验证。虽然后向可达性分析是识别安全初始状态的关键工具，但计算紧凑的形式化边界仍然出了名地难以扩展。最近的进展表明，采用伯恩斯坦多项式激活函数的DeepBern-Nets神经网络具有优越的数学特性，能够实现极其快速且紧凑的前向可达性分析。基于这种高效性，本文提出了BernBRS，一个用于DeepBern-Nets后向可达性分析的高精度且可扩展的框架。BernBRS同时计算后向可达集（BRS）的认证内近似和外近似。我们的方法不依赖于单一的凸包络边界，而是通过分析激活模式智能地细化输入空间的覆盖，精确地分割输入域……

    arXiv:2611.22083v1 Announce Type: new  Abstract: Ensuring the safety of complex neural network systems requires mathematically rigorous verification. While backward reachability analysis is a critical tool for identifying safe initial states, computing tight, formal bounds remains notoriously difficult to scale. Recent advancements have demonstrated that DeepBern-Nets neural networks utilizing Bernstein polynomial activation functions possess advantageous mathematical properties that enable exceptionally fast and tight forward reachability analysis. Building upon this efficiency, this paper introduces BernBRS, a highly accurate and scalable framework for the backward reachability analysis of DeepBern-Nets. BernBRS simultaneously computes certified inner and outer approximations of the Backward Reachable Set (BRS). Rather than relying on single convex bounding, our method intelligently refines a cover of the input space by analyzing activation patterns, splitting the input domain precis
    
[^138]: MedGuardBench：一个面向患者的个性化防护栏开发基准

    MedGuardBench: A Patient-Oriented Benchmark for Personalized Guardrail Development

    [https://arxiv.org/abs/2611.22069](https://arxiv.org/abs/2611.22069)

    该论文发布了MedGuardBench——首个以患者档案和处方为条件的个性化用药安全防护基准（包含87种药物、5,001个场景），并提出个性化推理防护栏方法，使模型能够基于患者的真实信息而非表面线索做出安全判断。

    

    安全是因人而异的：同一个行为对一个人可能安全，对另一个人却可能不安全。然而，当前的防护栏在判断内容时并未考虑受影响的是谁。我们以用药安全为切入点，提出了“以患者为条件的用药防护”，即根据患者档案和处方来预测安全判定结果以及17个风险类别。我们发布了MedGuardBench，这是一个涵盖87种药物、包含5,001个场景且训练与测试药物互不重叠的基准数据集。我们发现，标准防护栏依赖的是表层线索（如措辞和患者档案是否存在），而非患者的真实信息。为解决这一问题，我们开发了由三个组件构成的个性化推理防护栏。个性化推理蒸馏将风险与相关的患者字段关联起来。上下文展开让模型自行逐项检查各个患者字段，在大多数骨干模型上降低了误报率；对照实验表明，直接向模型提供检查结果并不能带来同样的收益。一种基于规则的……（摘要原文在此处被截断）

    arXiv:2611.22069v1 Announce Type: new  Abstract: Safety is personal: the same action can be safe for one person but unsafe for another. Yet current guardrails judge content without considering who is affected. Using medication safety, we introduce \textit{patient-conditioned medication guarding}, which predicts a safety verdict and 17 risk categories from a patient profile and prescription. We release \textbf{MedGuardBench}, a drug-disjoint benchmark of 5{,}001 scenarios across 87 medications. We find that standard guardrails rely on surface cues, such as wording and profile presence, rather than patient facts. To address this, we develop \textit{personalized reasoning guardrails} with three components. \textbf{Personalized Reasoning Distillation} links risks to relevant patient fields. \textbf{Context Unrolling (CU)} makes models check each field themselves, reducing false alarms on most backbones; controls show that providing the check does not yield the same benefit. A \textbf{rule-
    
[^139]: 解决协作多智能体强化学习中的外生变异性问题

    Addressing Exogenous Variability in Cooperative Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.22067](https://arxiv.org/abs/2611.22067)

    提出了外生Dec-POMDP（ED-POMDP）形式化框架来刻画协作多智能体强化学习中的外生变异性与中介影响，并据此设计了CTDE兼容的LEICA算法，通过学习基于历史的上下文表示来改进策略更新。

    

    协作多智能体强化学习（MARL）在外生变异性下常常失效，例如对手变化或环境状态变化等团队无法控制但会改变转移动力学的因素。我们将这一挑战形式化为外生Dec-POMDP（ED-POMDP），其中回合级的外生状态（独立于团队选择）对动力学进行索引，且全局状态被划分为直接响应当前团队动作的组成部分与在当前步骤不直接响应动作的组成部分。这一形式化揭示了中介影响：虽然团队动作在当前步骤不会直接更新后者，但可以通过引发前者的变化来塑造后者的后续演化。基于此，我们提出了LEICA，一种与CTDE兼容的算法，它为这两个组成部分学习基于历史的上下文表示，并利用智能体级别的、影响……（原文摘要在此处截断）

    arXiv:2611.22067v1 Announce Type: cross  Abstract: Cooperative multi-agent reinforcement learning (MARL) often fails under exogenous variability such as shifts in opponents or environment regimes that cannot be controlled by the team but alter the transition dynamics. We formalize this challenge as Exogenous Dec-POMDP (ED-POMDP), in which an episode-level exogenous regime, selected independently of the team, indexes the dynamics, and the global state is partitioned into components that are directly responsive to the current team action and components that are not directly action-responsive at the current step. This formulation exposes mediated influence: while team actions do not directly update the latter at the current step, they can shape their later evolution through induced changes in the former. Based on this, we propose LEICA, a CTDE-compatible algorithm that learns history-conditioned context representations for both components and shapes policy updates using agent-wise, influe
    
[^140]: 关于 e-means 的两个反例

    Two Counterexamples for e-means

    [https://arxiv.org/abs/2611.22065](https://arxiv.org/abs/2611.22065)

    本文给出两个反例，证明 e-means 算法在相同初始中心下最终损失可能高于标准 k-means，且其基于质量的聚类复活机制可能破坏全局最优解，导致期望损失之比无界。

    

    e-means（Bessa 等人，2026）将指数平均与聚类复活机制相结合。我们针对其已发表的算法方案给出了两个反例，证明在相同的初始中心条件下，该算法的最终损失高于标准 k-means（Lloyd，1982）。基于质量的复活机制可能会破坏全局最优解，即使使用 k-means++ 进行初始化，期望损失之比仍可能是无界的。

    arXiv:2611.22065v1 Announce Type: new  Abstract: e-means (Bessa et al., 2026) combines exponential averaging and cluster revival. We give two counterexamples to its published recipe with higher final loss than standard k-means (Lloyd, 1982) from identical initial centers. Mass-based revival can destroy a global optimum and make the ratio of expected losses unbounded even with k-means++ seeding.
    
[^141]: 权重空间编程不需要Transformer

    Weight-Space Programming Doesn't Need Transformers

    [https://arxiv.org/abs/2611.22055](https://arxiv.org/abs/2611.22055)

    在无需梯度训练的权重空间编程中，三个前沿LLM在自由设计架构时都不约而同地将知识编译为可查询的稀疏表，且其性能以约1/1800的参数量超越了Transformer，表明权重空间编程并不需要Transformer。

    

    诸如Transformer之类的标准神经架构之所以被选用，是因为它们能通过梯度下降很好地训练，并且在GPU上运行速度快。而在权重空间编程中，大型语言模型（LLM）无需基于梯度的训练即可直接编写目标权重，因此这些理由不再适用。我们还应该继续使用Transformer吗？在最少指导、十个标注示例和一个沙盒环境的条件下（任务主要为多项选择题），我们让三个前沿LLM要么为固定架构生成权重，要么自行设计架构。当可以自由设计时，这三个LLM都将纯文本事实编译成一个由固定程序查询的稀疏表，测试准确率最高达到54.4%。LLM设计的模型比同一LLM所编写的最准确的已发布Transformer取得了更高的测试准确率，而参数量仅约为后者的1/1800。在没有梯度的情况下，它们选择了一种表结构，其中LLM可以说明每个权重的含义……

    arXiv:2611.22055v1 Announce Type: cross  Abstract: Standard neural architectures such as the Transformer were chosen because they train well with gradient descent and run fast on GPUs. In weight-space programming, a large language model (LLM) writes target weights by hand without gradient-based training, so those reasons no longer apply. Should we keep the Transformer? Given minimal guidance, ten labeled examples, and a sandbox, mainly on multiple-choice questions, we task three frontier LLMs with either generating the weights of a fixed architecture or designing the architecture themselves. When free to design, all three compile plain-text facts into a sparse table queried by a fixed procedure, reaching up to 54.4% test accuracy. An LLM-designed model achieves higher test accuracy than the most accurate released Transformer that the same LLM wrote while using about 1,800x fewer parameters. Without gradients, they choose a table, in which the LLM can state what each weight means and ca
    
[^142]: 一个统一的推理过程支撑着工作记忆的行为与神经环路

    A single inference process underlies the behavior and neural circuits of working memory

    [https://arxiv.org/abs/2611.22054](https://arxiv.org/abs/2611.22054)

    该论文提出 MemGP 理论，将工作记忆统一为对特征空间连续函数的概率推断过程，从而在计算、算法和神经环路三个层面同时解释了工作记忆的多种行为与神经现象。

    

    工作记忆将过去的输入在线保持以供未来行为使用。尽管经过数十年的研究，关于工作记忆的解释仍然分散在不同的分析层面之间，并存在长期争论：其容量究竟反映离散的“槽位”还是连续的“资源”、注意如何改善回忆、以及其表征依赖持续神经放电还是“活动静默”状态。在此，我们展示了一个同时在计算、算法和神经环路三个分析层面上明确规定的理论——MemGP——能够解释工作记忆中多样的行为与神经现象。在计算层面，MemGP 将工作记忆视为对特征空间上一个连续函数的概率推断，其变分目标在记忆强度与高斯过程先验之间取得平衡。在算法层面，记忆的写入、存储和读取对应于用一组有限的诱导点来近似该后验分布。

    arXiv:2611.22054v1 Announce Type: cross  Abstract: Working memory holds past inputs online for future behavior. Despite decades of research, accounts of working memory remain divided across levels of analysis, with long-standing debates on whether its capacity reflects discrete slots or continuous resources, how attention improves recall, and whether its representations rely on persistent neural firing or activity-silent states. Here we show that MemGP, a theory that is specified at the computational, algorithmic, and circuit levels of analysis simultaneously, can explain diverse behavioral and neural phenomena in working memory. At the computational level, MemGP casts working memory as probabilistic inference over a continuous function on feature space, with a variational objective that balances memory strength against a Gaussian-process prior. At the algorithmic level, writing, storing, and reading of memories correspond to approximating this posterior with a finite set of inducing p
    
[^143]: RiDEx：面向扩散强化学习的奖励感知定向探索

    RiDEx: Reward-Informed Directional Exploration for Diffusion Reinforcement Learning

    [https://arxiv.org/abs/2611.22052](https://arxiv.org/abs/2611.22052)

    RiDEx通过将局部探索集中在去噪速度场中与奖励相关的子空间，并利用等大反向扰动的奖励差异提供带符号的局部监督，从而替代低效的各向同性随机扰动，显著提升了基于奖励的扩散模型后训练的探索效率。

    

    基于奖励的扩散模型后训练通过任务奖励来更新模型参数。尽管现有方法改进了采样和策略优化，但它们在很大程度上仍依赖于高维潜空间中的各向同性随机扰动进行探索。这种变化往往会把扰动预算花费在对目标奖励影响甚微的改变上，导致探索效率低下。我们研究扩散模型应当在哪里探索以及如何从探索中学习。我们提出了奖励感知定向探索（RiDEx），它将局部探索集中在去噪速度场中与奖励相关的子空间内。该子空间由条件残差或可微奖励梯度提出，并通过在同一生成状态下施加等大反向的扰动来测量其对最终奖励的响应。奖励差异在被探索的坐标上提供带符号的局部监督信号，同时所有生成的图像均有所贡献。

    arXiv:2611.22052v1 Announce Type: new  Abstract: Reward-based diffusion post-training updates model parameters from task rewards. While existing approaches improve sampling and policy optimization, they still rely largely on isotropic random perturbations in a high-dimensional latent space for exploration. Such variation often spends the perturbation budget on changes with little influence on the target reward, making exploration inefficient. We study where diffusion models should explore and how to learn from that exploration. We introduce Reward-Informed Directional Exploration (RiDEx), which concentrates local exploration in reward-relevant subspaces of the denoising velocity field. Conditional residuals or differentiable reward gradients propose the subspace, and equal-and-opposite perturbations at the same generation state measure its terminal reward response. The reward difference supplies signed local supervision in the explored coordinates, while all generated images contribute
    
[^144]: DSADiff：通过缓解复杂性与不平衡引起的偏差、实现数据结构感知的晶体结构预测扩散模型

    DSADiff: Data Structure-Aware Diffusion for Crystal Structure Prediction via Mitigating Complexity- and Imbalance-Induced Biases

    [https://arxiv.org/abs/2611.22050](https://arxiv.org/abs/2611.22050)

    该论文提出了LAS机制分析框架，识别出晶体复杂性和元素组成不平衡是影响生成难度的两个关键因素，并据此提出数据结构感知扩散模型DSADiff，通过统一的生成难度表示来缓解生成偏差，从而提升晶体结构预测性能。

    

    晶体结构预测（CSP）是加速新材料生成与发现的重要步骤。近年来，基于扩散模型的晶体结构预测方法取得了显著进展。然而，面对多样化的材料空间时，现有模型仍然存在生成偏差问题，而用于追踪此类性能差异来源的系统性方法仍然有限。为解决这一问题，我们提出了LAS（分层分析策略），这是一个机制分析框架，它识别出晶体复杂性和元素组成不平衡是影响生成难度的两个关键因素，并进一步揭示了它们与模型生成性能之间的关系。基于这些发现，我们进一步提出了DSADiff（数据结构感知扩散模型），这是一种数据结构感知的扩散模型，它将LAS识别出的关键因素转化为统一的生成难度表示，并利用该表示来共同驱动特征……

    arXiv:2611.22050v1 Announce Type: new  Abstract: Crystal Structure Prediction (CSP) is an important step in accelerating the generation and discovery of new materials. In recent years, diffusion-based CSP methods have made significant progress. However, existing models still suffer from generation bias when facing diverse materials spaces, while systematic approaches for tracing the sources of such performance variations remain limited. To address this issue, we propose LAS (Layered Analysis Strategy), a mechanism analysis framework that identifies crystal complexity and elemental composition imbalance as two key factors affecting generation difficulty and further reveals their relationships with model generation performance. Building on these findings, we further propose DSADiff (Data Structure-Aware Diffusion), a data structure-aware diffusion model that transforms the key factors identified by LAS into a unified generation difficulty representation and uses it to jointly drive featu
    
[^145]: RLDE：一个用于可解释迭代文档编辑的强化学习框架

    RLDE: A Reinforcement Learning Framework for Interpretable Iterative Document Editing

    [https://arxiv.org/abs/2611.22039](https://arxiv.org/abs/2611.22039)

    提出了RLDE框架，将文档迭代修订建模为无参考的MDP问题，通过指针网络与PPO智能体结合冻结的BART模型，依据可读性、困惑度和语法等多目标奖励实现可解释的自动化文档编辑。

    

    提升长文档的质量、清晰度和流畅性仍然是自然语言处理（NLP）中一个具有挑战性的问题，因为人工编辑通常是通过迭代式润色而非一次性重写完成的。现有的大多数自动化系统要么充当黑盒生成器，要么依赖监督学习，这限制了可解释性，并且无法捕捉修订过程的序列特性。本文提出了RLDE，一个自动化的端到端、指标驱动的强化学习框架，它将文档修订形式化为一个无参考的马尔可夫决策过程（MDP）问题（相对于成对的文档-修订目标而言是无监督的）。在有限的动作空间（NOOP、INSERT、DELETE、REPLACE）下，RLDE利用指针网络在词元级别进行动态片段选择，并将生成式动作发送给冻结的BART-base模型以获得上下文感知的修改建议。PPO智能体学习优化基于可读性、困惑度和语法指标的多目标奖励函数……

    arXiv:2611.22039v1 Announce Type: cross  Abstract: Improving the quality, clarity, and flow of long documents remains a challenging problem in NLP, as human editing typically occurs through iterative refinement rather than a single-pass rewrite. Most existing automated systems act as black-box generators or rely on supervised learning, limiting interpretability and failing to capture the sequential nature of revisions. In this paper, we propose RLDE, an automated end-to-end metric-driven RL framework that formulates document revision as a reference-free MDP problem (unsupervised with respect to paired document-revision targets). Working with a limited action space (NOOP, INSERT, DELETE, REPLACE), RLDE leverages Pointer Networks for dynamic span selection at the token level and sends generative actions to the frozen BART-base model to obtain context-aware suggestions. PPO agents learn to optimize a multi-objective reward function based on readability, perplexity, and grammar metrics wit
    
[^146]: GP-RL-MPC：基于高斯过程后决策状态值函数的不确定性感知强化学习增强模型预测控制

    GP--RL--MPC: Uncertainty-Aware Reinforcement Learning-Augmented Model Predictive Control with Gaussian Process Postdecision State-Value Functions

    [https://arxiv.org/abs/2611.22038](https://arxiv.org/abs/2611.22038)

    该论文提出GP-RL-MPC方法，以高斯过程回归学习的不确定性感知评论家函数作为模型预测控制的终端惩罚项，数据高效地考虑预测时域之外的成本，并通过可调风险系数在认知不确定性与偶然不确定性之间实现权衡。

    

    我们提出高斯过程强化学习增强模型预测控制（GP-RL-MPC），该方法使用学习得到的评论家函数作为MPC的终端惩罚项，以考虑预测时域之外的成本。高斯过程回归（GPR）能够在乐观近似策略迭代（OAPI）框架内，从有限且有噪声的rollout轨迹中实现数据高效的评论家学习，并通过软更新机制缓解策略抖动问题。一个可调的风险系数对终端成本中的GP认知不确定性进行加权，从而调节对不确定评论家预测的敏感度。后决策状态公式将扰动平均处理转移到离线进行，以从终端转移开始考虑偶然不确定性。在目标一致且动作可容许的条件下，单步时域可退化为贪婪值空间强化学习。在通过精确边际似然拟合GP超参数后，我们利用诱导点构建稀疏的确定性训练条件（DTC）后验。

    arXiv:2611.22038v1 Announce Type: new  Abstract: We propose Gaussian Process Reinforcement Learning-Augmented Model Predictive Control (GP--RL--MPC), which uses a learned critic as the MPC terminal penalty to account for costs beyond the prediction horizon. Gaussian process regression (GPR) enables data-efficient critic learning from limited, noisy rollouts within optimistic approximate policy iteration (OAPI), with soft updates mitigating policy chattering. A tunable risk coefficient weights GP epistemic uncertainty in the terminal cost, regulating sensitivity to uncertain critic predictions. The postdecision state formulation shifts disturbance averaging offline to account for aleatoric uncertainty from the terminal transition onward. Under matching objectives and admissible actions, a one-step horizon recovers greedy value-space RL. After fitting GP hyperparameters by exact marginal likelihood, we construct a sparse deterministic training conditional (DTC) posterior using inducing p
    
[^147]: 一种基于AI的OSSE框架用于评估学习到的物理一致性：使用FourCastNet、Aurora、Prithvi WxC和GraphCast的多模型研究

    An AI-Based OSSE Framework to Evaluate Learned Physical Consistency: A Multi-Model Study Using FourCastNet, Aurora, Prithvi WxC, and GraphCast

    [https://arxiv.org/abs/2611.22024](https://arxiv.org/abs/2611.22024)

    本文提出首个基于AI的观测系统模拟实验（OSSE）框架，通过Charney猜想评估四种不同架构的AI天气模型（FourCastNet V2、GraphCast、Aurora、Prithvi WxC）能否像传统物理模型一样对新观测产生相当的物理响应。

    

    观测系统模拟实验（OSSE）是评估未来大气观测系统的核心方法。我们提出了首个基于AI的OSSE框架，用于测试不同的AI天气模型能否复现传统物理基模型对新观测所产生的相当物理影响。我们围绕Charney猜想（即不完整的历史观测可以推断大气的当前状态）对四种不同架构的AI模型进行了评估。我们考虑了FourCastNet V2、GraphCast、Aurora和Prithvi WxC，并将每个模型自回归地积分至60天的预报时长。经过30天的预热期（spin-up）后，我们插入ERA5温度场来推断风场及相关大气场。对于风场插入也开展了类似的OSSE实验。我们在有无数据插入的30天数据同化循环预报中，衡量每个模型对观测信息响应的性能。

    arXiv:2611.22024v1 Announce Type: new  Abstract: Observing System Simulation Experiments (OSSEs) are a central methodology for evaluating future atmospheric observing systems. We present the first AI-based OSSE framework to test whether different AI weather models can reproduce comparable physical impact from new observations that traditional physics based models generate. We evaluate four distinct architectural AI models on the Charney conjecture that incomplete historical observations can infer the present state of the atmospheric. We considner- FourCastNet V2, GraphCast, Aurora and Prithvi WxC and integrate each autoregressively over a 60-day forecast horizon. After a 30-day spin-up, we insert ERA5 temperature fields to infer winds and related atmospheric fields. Similar OSSE experiments are conducted for wind insertion. We measure the performance of each model's response to observational information on a 30-day data assimilation cycle forecast with and without data insertion. Disti
    
[^148]: 计算置换网格编码：基于两次乘法和一个64字节表的近最优量化

    Computed Permutation Trellis Coding: Near-Optimal Quantization from Two Multiplications and a 64-Byte Table

    [https://arxiv.org/abs/2611.22019](https://arxiv.org/abs/2611.22019)

    提出CPTC量化格式，其解码器仅需两次乘法和一个64字节表即可实现近最优的低位宽张量量化，在4比特时距信息论理想仅差0.07比特，同时支持1至6比特所有位宽并精确复现标准整数与浮点格式。

    

    我们提出了计算置换网格编码（CPTC），这是一种针对“一次写入、多次读取”张量的低位宽量化格式。诸如INT4和FP4这类对每个值独立进行舍入的格式，相对于信息论理想而言浪费了大量比特。网格编码通过对多个值进行联合编码来挽回这一损失，但现有的解码器要么需要大型查找表才能达到最佳质量，要么使用哈希来计算值，从而限制了其可解码的范围。CPTC的解码器将其状态乘以两个系数，并将两个乘积的高位比特相加以索引一个64字节的表。由于该和并非均匀分布，小表中很少被使用的边缘位置可以存放分布尾部很远的值。一个解码器即可覆盖从1到6的所有位宽，能精确复现标准的整数和浮点格式，并在4比特时距理想值仅差0.07比特。它的性能可以与表规模大数千倍的随机置换网格编码相媲美。

    arXiv:2611.22019v1 Announce Type: cross  Abstract: We propose Computed Permutation Trellis Coding (CPTC), a low-bit quantization format for tensors written once and read many times. Formats that round each value on its own, such as INT4 and FP4, waste a large part of their bits against the information-theoretic ideal. Trellis codes recover this loss by encoding values jointly, but prior decoders either need a large lookup table for their best quality or compute values with a hash that restricts what they can decode. CPTC's decoder multiplies its state by two coefficients and adds the top bits of the products to index a 64-byte table. Because that sum is not uniformly distributed, the rarely used ends of a small table can hold values far out in the tails. One decoder serves every bit width from 1 to 6, reproduces the standard integer and floating-point formats exactly, and comes within 0.07 bits of the ideal at 4 bits. It matches a random-permutation trellis code whose table is thousand
    
[^149]: SkillBuffBench：衡量智能体技能在框架、模型、技能访问模式与记忆等方面的贡献

    SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory

    [https://arxiv.org/abs/2611.22017](https://arxiv.org/abs/2611.22017)

    SkillBuffBench 通过将技能访问方式（无技能/精选技能/自动调用）、记忆开关、四个智能体框架和多个主流模型进行交叉配对评估，系统性地衡量了智能体技能在贴近真实部署场景下的实际贡献。

    

    智能体技能是一类基于文件的程序性知识包，编码智能体可以按需发现并加载，目前各大主流智能体框架均已支持。然而，关于技能能带来多大帮助的证据，几乎完全来自这样的设定：为任务人工挑选正确的技能，且智能体从空白状态开始。这两点在真实部署中都不成立——部署时智能体需要从已安装的技能库中自行选择，并且经常携带此前工作中积累的记忆。我们提出了 SkillBuffBench，这是一个基准测试及配套的配对评估协议，它将技能访问视为一个三级因素（无技能、精选技能、以及从已安装库中自动调用），并与记忆（开启与关闭）、四个智能体框架以及多个模型（来自 Anthropic、Google 和 OpenAI 的最先进专有模型，以及来自中国厂商的最先进开源权重模型）进行交叉组合，每个模型都在每个框架下运行……

    arXiv:2611.22017v1 Announce Type: new  Abstract: Agent Skills, file-based packages of procedural knowledge that a coding agent can discover and load on demand, are now shipped by every major agent harness, yet the evidence for how much they help comes almost entirely from settings in which the right Skill is hand-picked for the task and the agent starts from a blank slate. Neither holds in deployment, where an agent chooses from whatever library happens to be installed and frequently carries memory from earlier work. We present SkillBuffBench, a benchmark and paired evaluation protocol that treats Skill access as a three-level factor (no Skill, a curated Skill, and automatic invocation from an installed library), crosses it with memory (off and on), with four agent harnesses and with multiple models (the state-of-the-art proprietary models from Anthropic, Google and OpenAI, and the state-of-the-art open-weight models from Chinese providers), every model being run under every harness, a
    
[^150]: 论形式化反事实解释

    On Formal Counterfactual Explanations

    [https://arxiv.org/abs/2611.22010](https://arxiv.org/abs/2611.22010)

    本文将最近世界语义下的反事实解释刻画为析取范式，并提出合取反事实解释（CFXps）这一新概念，建立了CXps、AXps与CFXps之间的结构联系，从而架起了反事实解释与形式化XAI之间的桥梁。

    

    反事实解释通常被视为在最近世界语义下的反事实条件句，这需要对所有最小距离的反例进行推理。相比之下，形式化可解释人工智能（XAI）通常依赖于基于特征的溯因解释和对比解释，这两类解释支持高效计算并具有最小命中集对偶性。本文通过以下两种方式连接这两种视角：(i) 将最近世界反事实解释明确刻画为所有最小距离反例实例之上的析取范式；(ii) 引入合取反事实解释，这是反事实前件的一个子类，限制为能保证预测变化的特征文字的合取。论文建立了连接CXps、AXps与CFXps的结构性结果：每个CXp都为其见证实例诱导出一个AXp，并且每个CXp与一个见证AXp共同诱导出一个支持……（原文摘要在此处截断）

    arXiv:2611.22010v1 Announce Type: new  Abstract: Counterfactual explanations are often seen as counterfactual conditionals under closest-world semantics, requiring one to reason over all minimally distant counterexamples. In contrast, formal XAI typically relies on feature-based abductive explanations (AXps) and contrastive explanations (CXps), which admit efficient computation and a minimal hitting-set duality. This paper bridges these perspectives by (i) characterizing closest-world counterfactual explanations explicitly as a disjunctive normal form over all minimally distant counterexample instances, and (ii) introducing conjunctive counterfactual explanations (CFXps), a subclass of counterfactual antecedents restricted to conjunctions of feature literals that guarantee prediction change. The paper establishes structural results connecting CXps, AXps and CFXps: every CXp induces an AXp for its witness instance, and every CXp together with a witness AXp induces a CFXp whose support i
    
[^151]: 聚类能承受多大的分布偏移？结构化传输下的精确分裂与合并证书

    How Much Distribution Shift Can a Clustering Withstand? Exact Split and Merge Certificates for Structured Transports

    [https://arxiv.org/abs/2611.22000](https://arxiv.org/abs/2611.22000)

    该论文基于确定性退火的相变分析提出局部稳定性准则，为聚类在结构化分布偏移下的分裂与合并事件给出精确证书，从而量化聚类对分布偏移的鲁棒性。

    

    在聚类中，被聚类群体的分布往往是不确定的或会随时间变化，因此真实分布可能偏离用于形成聚类的名义分布。一旦聚类被确定并且每个成员被分配到某个簇，这种偏移可能会改变每个簇的分布。这些偏移可能使某个簇在给定分辨率下过于分散而无法由单个代表点表示，从而引发分裂；也可能使不同的簇在该分辨率下足够接近以致可以共享一个代表点，从而引发合并。我们利用建立在确定性退火相变分析之上的局部稳定性准则，对分裂和合并事件进行了精确刻画。这种稳定性概念使我们能够量化聚类对此类事件的鲁棒性。此外，受不确定性和群体变化具有结构化特征的应用场景启发，我们研究位移被限制在预定（空间内）的偏移……

    arXiv:2611.22000v1 Announce Type: new  Abstract: In clustering, the distribution of the population being clustered is often uncertain or changes over time, so the true distribution may shift away from the nominal distribution used to form the clustering. Once the clustering is committed and each member is assigned to a cluster, such shifts may change the distribution of each cluster. They can make a cluster too spread out for one representative at a given resolution, motivating a split, or bring distinct clusters close enough to share a representative at that resolution, motivating a merge. We make these split and merge events precise using a local stability criterion built on the phase-transition analysis in deterministic annealing. This notion of stability enables us to quantify how robust a clustering is against such events. Furthermore, motivated by applications in which uncertainty and population change are structured, we study shifts whose displacements are restricted to a prescr
    
[^152]: 无指向的决策：决策模型能否定位其所检测异常的案例研究

    Deciding Without Pointing: A Case Study of Whether a Decision Model Localizes the Anomalies It Detects

    [https://arxiv.org/abs/2611.21995](https://arxiv.org/abs/2611.21995)

    该案例研究发现，决策模型Clef-flash虽在工业异常检测上表现不俗，但其决策头的注意力并不能定位所检测到的缺陷，而骨干网络的注意力定位能力在改变问题或移除文本上下文时会显著退化。

    

    决策模型可以通过对每个选项输出概率来回答关于图像的类型化问题，而无需生成文本。在我们对Clef-flash（经后训练的Qwen3.5-9B骨干网络）在工业检测基准上的案例研究中，我们探究它是否能定位其检测到的异常。零样本检测结果参差不齐：在MVTec AD上每类别图像AUROC为0.924，在MVTec AD 2上为0.668。在前者上，决策头的路由注意力产生的平均每图像像素AUROC仅为0.391，并且对它隐藏图像token也不会改变检测结果。多模态大语言模型的骨干注意力确实指向缺陷区域，但当我们改变问题时，其表现会低于中心先验基线；而如果移除问题中的文本上下文，其表现甚至会低于随机水平，部分原因是骨干注意力偏离了目标物体，而此时决策头的注意力图却几乎不变。骨干网络的内部状态在用正常图像探测时能够携带位置信息，但在测试集图像上，……（原文摘要在此处截断）

    arXiv:2611.21995v1 Announce Type: cross  Abstract: Decision models can answer a typed question about an image with a probability per option, without generating text. In our case study of Clef-flash (post-trained Qwen3.5-9B backbone) on industrial inspection benchmarks, we ask whether it localizes the anomalies it detects. We obtained uneven results with zero-shot detection: per-category image AUROC 0.924 on MVTec AD and 0.668 on MVTec AD 2. On the former, the decision head's routing attention produces 0.391 mean per-image pixel AUROC, and hiding the image tokens from it leaves detection unchanged. The multimodal LLM's backbone attention does point at the defect, however it falls below a center prior when we change question. It falls below chance if we remove textual context in the question, partly due to backbone attention leaving the object, while the head's map hardly changes. The backbone's states carry the location when probed with normal images, on the test-split images, however, 
    
[^153]: 大语言模型作为非线性动力系统：深度、训练与生成的三个时钟

    Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation

    [https://arxiv.org/abs/2611.21983](https://arxiv.org/abs/2611.21983)

    本文提出“三时钟”框架，将语言模型在层深度、训练过程和生成过程三个尺度上的行为与非线性动力学的经典概念（不动点、分岔、同步、混沌等）系统对应，并通过汇总约200篇文献的“三时钟词典”和一系列降阶模型在两个学科之间架起桥梁。

    

    语言模型是动力系统。Transformer 在其层上迭代、在训练步骤中迭代、在生成的词元序列上迭代，而在这三个“时钟”的每一个上，它都表现出非线性动力学已研究一个世纪的行为：不动点、分岔、同步、混沌以及噪声驱动的势阱逃逸。由于两个学术社区很少交流，这些行为常常以新名称被重新发现。我们在两者之间架起桥梁。为了描述，我们将约200篇论文浓缩为“三时钟词典”，把十五个非线性动力学概念映射到每个时钟上，并将每条对应关系标记为已证明、已测量或推测，同时在机械系统和语言模型中并排展示每个概念。为了解释，我们推导了降阶模型：softmax 是玻尔兹曼分布，梯度下降是非线性弹簧上的欧拉步，损失尖峰则是小球被抛出刚性山谷。

    arXiv:2611.21983v1 Announce Type: cross  Abstract: Language models are dynamical systems. A transformer is iterated over its layers, over the steps of its training and over the tokens it generates, and on each of these three clocks it shows behavior that nonlinear dynamics has studied for a century: fixed points, bifurcations, synchronization, chaos and noise-driven escape from potential wells. The two communities rarely meet, so this behavior is often rediscovered under new names. We build a bridge between them. To describe, we condense about 200 papers into the Three-Clock Dictionary, which maps fifteen concepts of nonlinear dynamics onto each clock and marks every correspondence as proved, measured or conjectured, and we show each concept in a mechanical system and in a language model side by side. To explain, we derive reduced-order models: the softmax is a Boltzmann distribution, gradient descent an Euler step on a nonlinear spring, a loss spike a ball thrown out of a stiff valley
    
[^154]: 在嵌入空间中对文档与章节层次结构建模

    Modeling Document and Section Hierarchies in Embedding Space

    [https://arxiv.org/abs/2611.21980](https://arxiv.org/abs/2611.21980)

    该论文提出层次化嵌入式主题模型，通过嵌入空间中的层次先验将文档级与章节级主题分布相耦合，并借助摊销变分推断在法律、报告和论坛等多个领域的结构化语料库上取得了最低或接近最低的困惑度表现。

    

    嵌入式主题模型将词嵌入整合到主题生成过程中，能够产生连贯的主题，并在大词表上表现出稳健的性能。然而，每篇文档仅由单个主题分布表示，忽略了其内部结构。许多现实世界的语料库，包括法律文档、公司和政府报告以及在线讨论论坛，都呈现出层次化的组织结构：文档由多个章节组成，而这些章节往往涉及不同的子主题。我们提出了层次化嵌入式主题模型，这是一种结构化主题模型，通过嵌入空间中的层次先验将文档的主题分布与其各章节的主题分布相耦合。我们为 hETM 推导了一种摊销变分推断过程，并在来自上述各领域的结构化语料库上对其进行评估。实验结果表明，hETM 在扁平模型与……之中取得了最低或接近最低的留出集困惑度（摘要原文在此处截断）。

    arXiv:2611.21980v1 Announce Type: cross  Abstract: Embedded topic models integrate word embeddings into the topic generation process, producing coherent topics and robust performance on large vocabularies. However, each document is represented by a single topic distribution, ignoring its internal structure. Many real world collections, including legal documents, company and government reports, and online discussion forums, exhibit hierarchical organization in which documents consist of multiple sections that often address distinct subtopics. We introduce the Hierarchical Embedded Topic Model (hETM), a structured topic model that couples the topic distribution of a document and those of its sections through a hierarchical prior in the embedding space. We derive an amortized variational inference procedure for hETM and evaluate it on structured collections from each of these domains. Experimental results show that hETM achieves the lowest or near-lowest held-out perplexity among flat and
    
[^155]: 稀疏整数傅里叶变换及其在机器学习中的应用

    Sparse Integer Fourier Transform and Its Application In Machine Learning

    [https://arxiv.org/abs/2611.21976](https://arxiv.org/abs/2611.21976)

    本研究提出稀疏整数傅里叶变换（SIFT），通过将DFT矩阵分解为稀疏整数矩阵并用优化策略近似旋转因子，将基数32 FFT的算术成本降低75%，同时保持极低的误差水平。

    

    离散傅里叶变换（主要通过快速傅里叶变换FFT实现）在众多领域中都至关重要，例如数字信号处理和材料设计中的特征提取。然而，在大数据时代，当前的FFT方法存在大量非平凡乘法和旋转因子操作的问题。为了提高FFT的计算效率，本研究提出了稀疏整数傅里叶变换，该方法利用稀疏性和小整数的快速计算优势，并应用基数n方法来推导FFT。基数n FFT中的2N点DFT矩阵被分解为四个稀疏整数矩阵。包含复数旋转因子的对角矩阵通过优化策略进行高效近似，而其余矩阵则保持稀疏性和整数条目。对于基数32的FFT，SIFT将估计的算术成本降低了75%，均方根误差（RMSE）低于2.3765×10⁻⁸，MAPE b

    arXiv:2611.21976v1 Announce Type: cross  Abstract: The Discrete Fourier transform, mainly implemented by Fast Fourier Transform (FFT), has been essential for numerous domains, such as digital signal processing and feature extraction in materials design. However, current FFT methods suffer from extensive non-trivial multiplications and twiddle-factor operations in the big data era. To improve the computational efficiency of FFT, this study proposes Sparse Integer Fourier Transform (SIFT), a method that leverages the fast computing of sparsity and small integers, and applies the Radix-n method to derive an FFT. The 2N-point DFT matrix in Radix-n FFT is decomposed into four sparse integer matrices. A diagonal matrix containing complex twiddle factors is efficiently approximated using an optimization strategy while the remaining matrices retain sparsity and integer entries. For radix-32 FFT, SIFT reduces the estimated arithmetic cost by 75%, with RMSE below $2.3765\times10^{-8}$ and MAPE b
    
[^156]: 超越位宽：面向NVIDIA Blackwell多语言大模型推理的部署约束混合精度量化

    Beyond Bitwidth: Deployment-Constrained Mixed-Precision Quantization for Multilingual LLM Inference on NVIDIA Blackwell

    [https://arxiv.org/abs/2611.21974](https://arxiv.org/abs/2611.21974)

    该论文在NVIDIA B200上对Qwen3-8B进行了部署约束驱动的低比特与混合精度量化实测评估，发现INT2/INT3全面退化且模块恢复效应具有强非加性，最终提出一个NVFP4与INT8混合的配置，在通过全部预注册多语言与推理精度门槛的同时将权重内存压缩至5.917 GiB。

    

    arXiv:2611.21974v1 公告类型： cross 摘要：低比特大模型量化只有在模型保持与服务相关的精度、能够通过受支持的GPU内核执行、并满足实测的内存和延迟限制时才有实用价值。我们在NVIDIA B200上使用vLLM研究Qwen3-8B，涵盖INT2/INT3、GPTQ、NVFP4以及混合精度配置。我们在全部144个融合线性模块上验证了原生打包执行与精度，并在预注册的门槛条件下评估韩语、英语、数值和推理质量。打包的INT2和INT3虽然可以执行，但所有七种测试的INT2网格在生成任务上均出现退化。孤立的模块恢复效应表现出强烈的非加性；韩语与推理任务的敏感性排名互不一致，且一个看似有前景的INT4/INT8开发候选方案在独立的最终评估中失败。一个包含16位激活、98个NVFP4仅权重量化模块、46个INT8模块和INT8输出头的配置，在未做任何改动的FINAL-81测试集（5,168个条目）上通过了全部门槛，权重加载内存为5.917 GiB……

    arXiv:2611.21974v1 Announce Type: cross  Abstract: Low-bit LLM quantization is useful only if a model preserves service-relevant accuracy, executes through supported GPU kernels, and meets measured memory and latency limits. We study Qwen3-8B on NVIDIA B200 with vLLM, including INT2/INT3, GPTQ, NVFP4, and mixed-precision configurations. We verify native packed execution and precision on all 144 fused linear modules and evaluate Korean, English, numerical, and reasoning quality under preregistered gates. Packed INT2 and INT3 execute, but all seven tested INT2 grids degenerate on generation. Isolated module-restoration effects are strongly non-additive; Korean and reasoning sensitivity rankings disagree, and a promising INT4/INT8 development candidate fails independent final evaluation. A configuration with 16-bit activations, 98 NVFP4 weight-only modules, 46 INT8 modules, and an INT8 output head passes all gates on the untouched FINAL-81 set (5,168 items): 5.917 GiB of weight-loading me
    
[^157]: 从求解器到系统：科学仿真中大语言模型跨表征与执行的推理

    Solver to System: LLM Reasoning Across Representations and Execution in Scientific Simulation

    [https://arxiv.org/abs/2611.21964](https://arxiv.org/abs/2611.21964)

    该论文提出了Solver2System数据集，通过多表示形式、受控损坏与词汇扰动来评估大语言模型在PDE科学仿真中的跨表示一致性与物理无效性检测能力。

    

    基于偏微分方程（PDE）的仿真器被广泛用于对物理系统建模，但利用大语言模型（LLM）生成和解释PDE求解器代码仍然具有挑战性。科学仿真代码是检验LLM科学推理能力的有效试验平台，原因在于：（1）同一个系统可以用代码、方程、自然语言和数值轨迹等多种形式表示；（2）代码可能成功执行，但其实现的物理却是无效的。我们提出了Solver2System，这是一个包含多种表示形式、受控损坏和词汇扰动的PDE求解器数据集，用于评估LLM能否通过（1）跨表示一致性以及（2）对物理无效性的静态检测和智能体式检测来恢复底层物理系统。在八个开源权重模型上的实验表明，当证据相互冲突时，LLM会系统性地偏好某些表示形式，且对代码施加词汇扰动会改变其对其他未变内容的解释。

    arXiv:2611.21964v1 Announce Type: cross  Abstract: Simulators based on partial differential equations (PDEs) are widely used to model physical systems, but generating and interpreting PDE solver code with large language models (LLMs) remains challenging. Scientific simulation code is a useful testbed for LLM scientific reasoning because (1) one system can be represented by code, equations, natural language, and numerical trajectories and (2) code can execute successfully while implementing invalid physics. We introduce Solver2System, a dataset of PDE solvers with multiple representations, controlled corruptions, and lexical perturbations, to evaluate whether LLMs can recover the underlying physical system through (1) cross-representation consistency and (2) static and agentic detection of physical invalidity. Across eight open-weight models, LLMs systematically favor some representations when evidence conflicts, and lexical perturbations to code alter interpretation of other unchanged 
    
[^158]: OCCAM：改进的单细胞表示学习与扰动效应预测

    OCCAM: Improved Single-Cell Representation Learning and Perturbation Effect Prediction

    [https://arxiv.org/abs/2611.21959](https://arxiv.org/abs/2611.21959)

    Occam是一个结合去噪细胞表示与基因相互作用、文本及蛋白质序列特征的MLP框架，在单细胞表示学习和跨环境基因扰动效应预测方面均达到了最优性能。

    

    预测基因扰动的效应需要能够在已观察到的细胞环境和基因组合之外进行泛化的模型。我们提出了Occam，一个用于表示学习和扰动效应预测的多层感知机（MLP）框架。Occam将通过去噪学习到的细胞表示与基因相互作用、文本以及蛋白质序列特征相结合。它在扰动信息和保留细胞类型（含批次整合）方面获得了最高的聚合表示得分。在跨环境少样本预测任务中，某个扰动和某个细胞环境在训练期间各自被单独观察到过，但从未同时出现。与四个重新训练的竞争模型相比，以及与从其他训练细胞系直接迁移同一扰动的实测效应相比，Occam更好地捕捉了这些未观察组合的实测响应模式。在无需微调的情况下，Occam在所有被评估的模型中对扰动的区分能力最强……

    arXiv:2611.21959v1 Announce Type: cross  Abstract: Predicting the effects of genetic perturbations requires models that generalize beyond observed cell contexts and gene combinations. We introduce Occam, a multilayer perceptron (MLP) framework for representation learning and perturbation-effect prediction. Occam combines cell representations learned through denoising with gene interaction, text, and protein-sequence features. It achieves the highest aggregate representation scores for perturbation information and cell-type preservation with batch integration. In cross-context few-shot prediction, a perturbation and a cell context are each observed during training, but never together. Occam better captures the measured response patterns for these unobserved combinations than four retrained competitors or direct transfer of the same perturbation's measured effects from other training cell lines. Without fine-tuning, Occam best distinguishes perturbations among the evaluated models and si
    
[^159]: 面向拖拉机导航控制器的权重空间编程：大语言模型原生编写可驾驶真实拖拉机的神经网络权重

    Weight-space programming for tractor guidance controller: large language models natively write neural network weights that steer a real tractor

    [https://arxiv.org/abs/2611.21958](https://arxiv.org/abs/2611.21958)

    大语言模型通过权重空间编程直接读取、修复并从零编写拖拉机导航控制器的神经网络权重，无需优化器或梯度下降，在模拟和真实拖拉机上均实现了成功的路径跟踪。

    

    权重空间编程是一种由大语言模型在自然语言规范的指导下直接读取和写入神经网络权重的过程，无需优化器或梯度下降。本研究将其应用于拖拉机自动导航任务：一个仅含253个参数的神经网络沿直线和圆弧路径行驶，并在机器条件发生变化时（例如转向执行器变慢）需要进行调整。GPT-6 Astra和Claude Fable 5.1（以及5.2）以少量模拟评估为预算，为我们的研究用拖拉机编辑网络，修复了原本为另一台拖拉机训练的网络，并将控制器直接写入一个零权重初始化的网络中。我们在模拟和真实拖拉机上将其与进化策略、梯度下降和纯追踪算法进行了对比。两个模型都能正确读取网络并运用控制理论准确分析闭环系统，而Fable在不借助模拟的情况下进行修复仅偶尔成功。

    arXiv:2611.21958v1 Announce Type: cross  Abstract: Weight-space programming is a process where a large language model reads and writes the weights of a neural network directly, guided by a natural language specification, without an optimiser or gradient descent. This study applies it to automatic tractor guidance, where a 253-parameter neural network steers along paths of lines and arcs and must be adapted when the machine changes, such as a slower steering actuator. GPT-6 Astra and Claude Fable 5.1 (also 5.2) edited the network for our research tractor using a small budget of simulated evaluations to repair a network trained for another tractor and write a controller into a network with zero weights. We compared them with an evolution strategy, gradient descent and pure pursuit in simulation and on the real tractor. Both models read the network and analysed the closed loop with control theory correctly, and Fable's repair without simulation succeeded only occasionally. With simulated 
    
[^160]: ThermLeT：基于Transformer的2.5D芯粒架构温度预测

    ThermLeT: Transformer-based Temperature Prediction for 2.5D Chiplet Architecture

    [https://arxiv.org/abs/2611.21954](https://arxiv.org/abs/2611.21954)

    ThermLeT是一个基于Transformer的多模态代理模型，能在13毫秒内精准预测2.5D芯粒架构的稳态温度，比传统有限元求解器快约四个数量级，精度显著优于ThermEDGe和HotSpot。

    

    异构2.5D芯粒封装的发展带来了无与伦比的性能和定制化能力，但由此产生的高密度加剧了散热挑战。为了促进快速的设计空间探索，我们提出了ThermLeT，一个基于Transformer的多模态、尺度不变的代理模型，它集成功率密度图、空间坐标嵌入、材料特性和热传导速率，可在13毫秒内预测稳态温度，比ANSYS等传统有限元求解器快约四个数量级。为了训练该Transformer模型，我们还构建了一个包含6,221个芯粒布局的数据集，这些布局由三种代表性架构扰动生成。在150个样本的测试集上，ThermLeT实现了3.16°C的均方根误差（RMSE）和2.24°C的平均绝对误差（MAE），相比ThermEDGe准确度提高了35%，相比HotSpot提高了71%，同时推理延迟分别低约6倍和11倍。消融研究证实了其鲁棒的泛化能力。

    arXiv:2611.21954v1 Announce Type: cross  Abstract: The drive toward heterogeneous 2.5D chiplet packages promises unmatched performance and customization, yet the resulting density amplifies thermal challenges. To facilitate rapid design space exploration, we present ThermLeT, a multimodal transformer-based, scale invariant surrogate that integrates power density maps, spatial coordinate embeddings, material properties and heat transfer rate to predict steady state temperature in 13 ms, about four orders of magnitude faster than conventional finite element solvers such as ANSYS. To train the transformer model, we also create a dataset of 6,221 chiplet layouts perturbed from three representative architectures. On a 150 sample test set, ThermLeT achieves an RMSE of 3.16{\deg}C and an MAE of 2.24{\deg}C, improving accuracy by 35% over ThermEDGe and by 71% over HotSpot, while running approximately 6x and 11x lower inference latencies respectively. Ablation studies confirm robust generalizat
    
[^161]: 面向选择性状态空间模型在任意序列长度下的对数时间尺度扰动界

    A Log-Timescale Perturbation Bound for Selective State Space Models at Every Sequence Length

    [https://arxiv.org/abs/2611.21952](https://arxiv.org/abs/2611.21952)

    本文提出了一种对数时间尺度上的扰动界，证明选择性状态空间模型（Mamba）中输入的微小变化对隐藏状态的影响在任意序列长度下均有界、误差不随序列长度累积，且无需对时间尺度设置下界。

    

    arXiv:2611.21952v1 公告类型：新。摘要：部署在流式数据（如限价订单簿更新）上的 Mamba 模型，其运行长度会远远超出训练时的序列长度。因此，关于输入的微小变化会使隐藏状态移动多少的保证，必须在任意序列长度下都成立。据我们所知，最接近的现有界以加法方式度量时间尺度的变化，除非时间尺度被限制在远离零的范围内，否则该界会随序列长度线性增长。本文改为在对数尺度上度量这种变化。关键观察是：这样一来，每一步的衰减因子最多只会移动该步输入权重的一个固定比例，因此每步的误差不会累积叠加。利用这一结论，我们证明了从同一共同起点出发的选择性扫描（selective-scan）单元的两次运行，始终保持在一个界之内：该界在任意序列长度下都成立、随扰动大小收缩至零，且无需对时间尺度设置任何下界。我们还给出一个构造，表明对于微小扰动，该界在一个常数因子之内……（原文截断）

    arXiv:2611.21952v1 Announce Type: new  Abstract: Mamba models deployed on streaming data, such as limit order book updates, run far past the sequence lengths they were trained on. A guarantee on how much a small change of input can move the hidden state therefore has to hold at every length. The closest existing bound we know of measures the change of the timescale additively, and it grows linearly with the sequence length unless the timescale is bounded away from zero. In this paper, we measure the change on a log scale instead. The key observation is that the decay factor of each step then moves by at most a fixed fraction of that step's input weight, so the per-step errors cannot pile up. Using this, we prove that two runs of a selective-scan cell from a common start stay within a bound that holds at every length, shrinks to zero with the size of the change and needs no lower bound on the timescale. We give a construction showing that, for small changes, this bound is within a facto
    
[^162]: 守恒的判定，漂移的承诺：识别有害的过度思考

    Conserved Verdicts, Drifting Commitments: Identifying Harmful Overthinking

    [https://arxiv.org/abs/2611.21946](https://arxiv.org/abs/2611.21946)

    该论文提出“承诺-判定差距”框架来识别推理模型中的有害过度思考，证明仅凭最终正确性探针估计的“判定”无法预测继续思考的利弊，而“承诺”与“判定”之差恰好量化了停止思考可获得的准确率增益，并给出将准确率-预算曲线分解为链内答案变化与链存活变化的恒等式。

    

    一个已经得出正确答案的推理模型，可能会因为继续思考而失去这个正确答案。决定是否停止，意味着在推理链的同一位置比较两个概率：模型若现在停止所给出的答案是正确的概率（即其“承诺”），以及模型最终完成时的答案是正确的概率（即其“判定”）。我们证明，仅凭判定——也就是最终正确性探针所估计的量——无法完成这种比较。作为对一个固定结果的预测，判定没有可预测的漂移，无论继续思考是有益还是有害。承诺与判定之差等于答案质量的期望剩余变化量，也正是停止思考所能获得的准确率增益。第二个恒等式则将任何准确率-预算曲线分解为两部分：推理链内部的答案变化，以及仍在运行的推理链集合的变化。我们在 MATH-500 上对这两者进行了测量。在 DeepSeek-R1-Distill-Qwen-1.5B 持有正确答案的全部 379 个点上（这些点的选取不考虑其推理链如何结束），继续……（原文摘要至此截断）

    arXiv:2611.21946v1 Announce Type: new  Abstract: A reasoning model that has reached a correct answer can lose it by continuing to think. Deciding whether to stop means comparing two probabilities at the same point of a chain: that the answer the model would give if stopped now is correct (its commitment), and that the answer it will finish with is correct (its verdict). We show that the verdict alone, which is what final-correctness probes estimate, cannot make this comparison. As a forecast of a fixed outcome it has no predictable drift, whether continuing helps or hurts. The commitment-verdict gap equals the expected remaining change in answer quality and the accuracy gained by stopping. A second identity splits any accuracy-versus-budget curve into answer change within chains and change in which chains are still running. We measure both on MATH-500. At all 379 points where DeepSeek-R1-Distill-Qwen-1.5B holds a correct answer, selected without regard to how its chain ends, continuing
    
[^163]: 面向无障碍自动驾驶车辆应用的轮椅配置分类深度学习模型基准测试

    Benchmarking Deep Learning Models for Wheelchair Configuration Classification in Accessible Automated Vehicle Applications

    [https://arxiv.org/abs/2611.21945](https://arxiv.org/abs/2611.21945)

    本文提出了轮椅配置图像数据集（WCID），通过对手动/电动、轮椅子类型、乘坐状态等多维度属性进行细粒度标注，为无障碍自动驾驶车辆应用中的轮椅配置分类深度学习模型提供了基准测试资源。

    

    要使自动驾驶车辆具备无障碍性，需要考虑弱势道路使用者，包括轮椅使用者。为了提供更贴合需求的体验以支持这一群体的独立出行，车辆设计需要适应种类繁多的轮椅配置。然而，轮椅具有个性化特点，在车架几何结构、驱动方式和外观上各不相同。此外，乘员或配件还可能遮挡视觉线索。现有的移动辅助设备数据集仅标注轮椅是否存在，而未描述其具体配置。为填补这一空白，本文提出了轮椅配置图像数据集（WCID），该数据集包含467张经过面部模糊处理的图像，涵盖192个采集组，仅用于评估。每张图像均标注有：(i) 手动或电动的二分类标签，(ii) 轮椅子类型，(iii) 乘坐状态、视角、场景环境、光照、遮挡、驱动方式和控制方式，并包含一个273张图像的验证集划分。

    arXiv:2611.21945v1 Announce Type: cross  Abstract: Making automated vehicles accessible requires consideration of vulnerable road users, including wheelchair users. To offer a more tailored experience that supports independence for this group, vehicle design needs to accommodate the wide range of wheelchair configurations. However, wheelchairs are individualized, varying in frame geometry, propulsion, and visual appearance. Moreover, occupants or accessories may obscure visual cues. Existing mobility-aid datasets only label the presence of a wheelchair, rather than describing its specific configuration. To fill this gap, this paper introduces the Wheelchair Configuration Image Dataset (WCID), comprising 467 face-blurred images across 192 capture groups solely for evaluation. Each image is annotated with (i) a two-class label of manual or powered, (ii) a chair subtype, (iii) occupancy, viewpoint, setting, lighting, occlusion, propulsion, and control, with a 273-image validation split an
    
[^164]: 模型是否真的使用了它所引用的内容？小型视频语言模型中接地推理链的反事实检验

    Does the Model Use What It Cites? Counterfactual Tests of Grounded Reasoning Chains in Small Video-Language Models

    [https://arxiv.org/abs/2611.21940](https://arxiv.org/abs/2611.21940)

    本文提出一种无需标注的“引用差距”反事实检验方法，通过涂黑模型推理链中所引用的视频区域来验证小型视频语言模型是否真正依赖其引用的证据，结果显示模型在CLEVRER上具有较强证据依赖性，但在NExT-QA上依赖性很弱。

    

    接地推理链在每一步都引用一个帧区间和一个边界框，使读者能够检查模型观察的位置。然而，答案是否真正依赖于所引用的区域则是另一个问题。我们通过干预实验来检验这一点：将模型在其自身推理链中引用的区域涂黑，并将正确答案翻转的比例与在同一帧其他位置涂黑相同大小框的结果进行对比。这种成对差异被称为“引用差距”，且无需任何标注。我们将其应用于在精确CLEVRER推理链和伪标注NExT-QA推理链上训练的2-3B参数视频语言模型。在相同的模型权重下，引用差距在CLEVRER上为0.089，而在NExT-QA上仅为0.003。汇总所有模型的结果，遮蔽所引用的框在CLEVRER上导致翻转的正确答案数量是对照组的2.2倍，在NExT-QA上为1.1倍——即使仅考虑那些确实依赖视频的答案也是如此，同时这些引用框在两个数据集上均通过了接地性检查。而当在每一帧上遮蔽所引用的实体时

    arXiv:2611.21940v1 Announce Type: cross  Abstract: A grounded reasoning chain cites a frame span and a box for every step, so a reader can check where the model looked. Whether the answer depends on the cited region is a different question. We test it by intervention: the regions a model cited in its own chain are blacked out, and the share of correct answers that flip is compared with blacking out boxes of the same size elsewhere on the same frames. The paired difference, the cite gap, needs no annotation. We apply it to 2-3B video-language models trained on exact CLEVRER chains and on pseudo-labelled NExT-QA chains. With the same weights, the cite gap is 0.089 on CLEVRER and 0.003 on NExT-QA. Pooled over models, masking the cited boxes flips 2.2 times as many correct answers as the control on CLEVRER and 1.1 times as many on NExT-QA, also among the answers that depend on the video at all, while the boxes pass a grounding check on both. Masking the cited entity on every frame, instead
    
[^165]: HubEgo-100K：用于机器人学习的真实场景下熟练工作的自我中心视角视频数据集

    HubEgo-100K: In-the-Wild Egocentric Video of Skilled Work for Robot Learning

    [https://arxiv.org/abs/2611.21938](https://arxiv.org/abs/2611.21938)

    HubEgo-100K 是一个包含 10,124 小时视频的数据集，由 12,227 名贡献者在 15,548 个真实环境中录制熟练的未经编排工作，旨在通过环境和物体的多样性提升机器人模仿学习的泛化能力。

    

    通过模仿学习训练的机器人策略在训练中未见过的环境和物体之外泛化能力较差。已发表的数据规模研究表明，泛化能力随训练环境和物体的多样性提升而增长，而非随每个环境中更多示范而增长。跨众多环境收集机器人数据成本高昂，而自我中心视角的人类视频可以在真实场景中大规模录制；然而，现有的自我中心视频大多展示日常生活活动，几乎不包含真实工作场所中的熟练工作。我们提出了 HubEgo-100K，一个为多样性而构建的数据集。本文的此版本涵盖首次发布 v1.0：10,124 小时的 1080p 自我中心视角视频，并配有来自一个头戴式手机摄像头的同步 100 Hz 惯性（IMU）数据。在 v1.0 中，12,227 名贡献者在 15,548 个环境中录制了他们自己未经编排的真实工作，涵盖 11 种场景类型（每个环境为某位贡献者自己某一类型的具体场所，例如其厨房或工坊）……

    arXiv:2611.21938v1 Announce Type: cross  Abstract: Robot policies trained by imitation generalize poorly outside the environments and objects seen in training. Published data-scaling studies find that generalization grows with the diversity of training environments and objects rather than with more demonstrations per environment. Robot data is costly to collect across many environments, whereas egocentric human video can be recorded in the wild at scale; existing egocentric video, however, mostly shows daily-life activity and contains little skilled work in real workplaces. We present HubEgo-100K, a dataset built for diversity. This version of the paper covers the first release, v1.0: 10,124 hours of 1080p egocentric video with synchronized 100 Hz inertial (IMU) data from one head-mounted phone camera. In v1.0, 12,227 contributors record their own unscripted work in 15,548 environments (each one contributor's own setting of one type, such as their kitchen or workshop) across 11 setting
    
[^166]: 基于积分物理信息神经网络的高阶交互重构

    Higher-Order Interaction Reconstruction via Integral Physics-Informed Neural Networks

    [https://arxiv.org/abs/2611.21935](https://arxiv.org/abs/2611.21935)

    提出积分物理信息神经网络框架，以时间区间上的积分形式代替基于导数的物理残差惩罚，并结合交互张量的连续参数化，从而实现复杂网络中高阶交互的有效重构，同时抑制噪声并增强训练稳定性。

    

    重构高阶交互是复杂网络系统中的一个根本性挑战，在生态、社交和生物网络中具有广泛应用。由于高阶结构的组合复杂性、底层动力学的强非线性以及噪声和稀疏观测的普遍存在，这一逆问题本质上十分困难。在本文中，我们提出了一种新颖的积分物理信息神经网络框架用于高阶交互重构。与惩罚基于导数的物理残差不同，我们采用时间区间上的积分形式，这能够自然地抑制高频噪声并缓解由导数引起的不稳定性。我们进一步引入了交互张量的连续参数化，使得可以通过基于梯度的优化来可行地识别高阶结构。我们提供了理论分析，建立了（摘要在此处截断）……

    arXiv:2611.21935v1 Announce Type: new  Abstract: Reconstructing higher-order interactions is a fundamental challenge in complex networked systems, with applications in ecological, social, and biological networks. This inverse problem is inherently difficult due to the combinatorial complexity of higher-order structures, the strong nonlinearity of the underlying dynamics, and the prevalence of noisy and sparse observations. In this article, we propose a novel integral physics-informed neural network (IntegralPINN) framework for higher-order interaction reconstruction. Instead of { penalizing derivative-based physics residuals,} we adopt an integral formulation over temporal intervals, which naturally suppresses high-frequency noise and mitigates derivative-induced instability. We further introduce a continuous parameterization of interaction tensors, enabling tractable identification of higher-order structures via gradient-based optimization. We provide theoretical analysis establishing
    
[^167]: 具有精确边界条件的物理信息神经网络航天器轨迹优化方法

    Physics-informed neural networks for spacecraft trajectory optimization with exact boundary conditions

    [https://arxiv.org/abs/2611.21933](https://arxiv.org/abs/2611.21933)

    该论文提出一种利用轨迹优化问题数学结构的简单变换方法，将边界条件先验地精确嵌入神经网络解空间中，避免了标准物理信息神经网络中加权损失项敏感导致的收敛性和精度问题。

    

    我们利用神经网络来优化轨迹，将位置建模为时间的函数，并相对于网络参数最小化推力。这对应于一种无监督训练过程，其中物理定律被直接纳入优化目标。在标准的物理信息神经网络中，初始状态和终端状态通过额外的加权损失项来强制执行，这导致了一个多目标优化问题，该问题对损失权重的选择高度敏感，可能导致收敛性差和精度降低。为了解决这一挑战，我们利用轨迹优化问题的数学结构，构造了一个简单的变换，以便先验地精确强制执行边界条件。该变换可以被解释为解空间的一种重新参数化，它将边界条件直接嵌入到函数中。这将原始……

    arXiv:2611.21933v1 Announce Type: new  Abstract: We use neural networks to optimize trajectories by modeling the position as a function of time and minimizing the thrust with respect to the network parameters. This corresponds to an unsupervised training procedure in which physical laws are incorporated directly into the optimization objective. In standard physics-informed neural networks, initial and terminal states are enforced through additional weighted loss terms, resulting in a multi-objective optimization problem that is highly sensitive to the choice of loss weights and can lead to poor convergence and reduced accuracy. To address this challenge, we construct a simple transformation, exploiting the mathematical structure of trajectory optimization problems, to enforce boundary conditions exactly a-priori. This transformation can be interpreted as a reparameterization of the solution space that embeds the boundary conditions directly into the function. This converts the original
    
[^168]: 基于区块链赋能6G技术与联邦学习的车对车（V2V）通信

    V2V Communications Using Blockchain-Enabled 6G Technology and Federated Learning

    [https://arxiv.org/abs/2611.21932](https://arxiv.org/abs/2611.21932)

    该研究提出了一种融合区块链、联邦学习与延迟资源分配优化的6G赋能V2V通信系统，在数据包投递率、模型准确率和延迟等指标上优于现有技术，并通过共识优化和k-匿名实现了安全与隐私的双重保障。

    

    本研究提出了一种新颖的车对车（V2V）通信方法，该方法集成了区块链技术、联邦学习（FL）以及延迟与资源的分配优化。研究采用多种性能指标（如数据包投递率（PDR）、模型准确率和延迟）对所提出的系统进行了评估，并证明了其相对于现有技术的优越性。此外，该系统通过共识优化增强了安全性，并通过k-匿名技术保护数据隐私。总体而言，所提出的系统是在网联和自动驾驶汽车时代实现高效、安全V2V通信的一种有前景的解决方案。此外，与传统5G相比，所提出的方法实现了更高的可靠性、更低的延迟和更好的资源利用率。

    arXiv:2611.21932v1 Announce Type: cross  Abstract: This study proposes an interesting approach for vehicle-to-vehicle (V2V) communication, which integrates blockchain technology, federated learning (FL), and allocation optimization of latency and resources. The research evaluates the proposed system using various performance metrics such as packet delivery ratio (PDR), model accuracy, and latency and demonstrates its superiority over existing techniques. Furthermore, the system provides enhanced security through consensus optimization and k-anonymity for data privacy. Overall, the proposed system is a promising solution for efficient and secure V2V communication in the era of connected and autonomous vehicles. Moreover, the proposed approach achieves higher reliability, lower latency, and better resource utilization compared to traditional 5G.
    
[^169]: 退化LQR中通过确定性等价实现多对数遗憾

    Polylogarithmic Regret via Certainty Equivalence in Degenerate LQR

    [https://arxiv.org/abs/2611.21927](https://arxiv.org/abs/2611.21927)

    该论文证明在采用奇异输入惩罚的退化LQR问题中，带有短期初始探索的确定性等价算法在控制器可辨识条件下即可实现 O(log²T) 的多对数遗憾，相比正定输入惩罚大幅放宽了可辨识性要求，显著扩大了可获得多对数遗憾保证的问题范围。

    

    奇异输入惩罚（如最小方差控制中所采用的）可以减少控制未知线性系统时所需辨识的内容。在在线线性二次控制中，当最优控制器能够从最优反馈下收集的数据中被辨识出来时，确定性等价方法可以实现多对数遗憾。然而，在正定输入惩罚的情况下，这一条件过于严苛，因为对于可逆状态矩阵，它要求所有未知参数在最优反馈下均可辨识。我们证明奇异输入惩罚放宽了这一要求，从而扩大了确定性等价能够实现多对数遗憾的问题类别。具体而言，我们证明了带有短期初始探索阶段的确定性等价算法，只要控制器可辨识性成立，在 T 次交互中即可达到 O(log²T) 的期望遗憾。我们同时给出了 Ω(√T) 的极小极大下界来补充这一保证。

    arXiv:2611.21927v1 Announce Type: cross  Abstract: Singular input penalties, as in minimum variance control, can reduce what must be identified to control an unknown linear system. In online linear quadratic control, certainty equivalence can achieve polylogarithmic regret when the optimal controller is identifiable from data collected under optimal feedback. With a positive definite input penalty, however, this condition is restrictive, since for an invertible state matrix it requires all unknown parameters to be identifiable under optimal feedback. We show that singular input penalties relax this requirement, expanding the class of problems in which certainty equivalence achieves polylogarithmic regret. Specifically, we prove that a certainty-equivalence algorithm with a short initial exploration phase achieves $O(\log^2 T)$ expected regret over $T$ interactions whenever controller identifiability holds. We complement this guarantee with an $\Omega(\sqrt{T})$ minimax lower bound for 
    
[^170]: 错误的输入，错误的阈值：重新评估对抗性面部隐私

    Wrong Input, Wrong Threshold: Re-evaluating Adversarial Facial Privacy

    [https://arxiv.org/abs/2611.21923](https://arxiv.org/abs/2611.21923)

    论文揭示了现有对抗性面部隐私评估协议的两个独立缺陷——错误的输入方式（直接使用整图而非真实部署系统的检测、对齐、裁剪流程）和会饱和的成功率差值度量，并提出用配对余弦偏移这一新度量重新评估，发现现实前端会使标准迁移攻击的隐私保护效果损失56%。

    

    已发表的面部隐私方法报告称，其对黑盒识别模型的保护率超过90%。然而这个数字来自一个存在两个独立缺陷的评估协议，并且所用度量无法衡量这两个缺陷中的任何一个。输入是错误的：2022年以来发表的方法将整张照片或检测器框直接输入网络，而实际部署的系统则会先进行人脸检测、对齐到关键点模板再裁剪。在不重新生成图像的情况下，仅通过现实的前端对相同文件重新评分，就会使标准迁移攻击失去56%的躲避保护，使预算分配攻击失去7%。度量也是错误的：这种损失是成功率的差值，因此会饱和——在预算扫描中该指标变化了8.2倍，而其背后的实际效应仅变化1.34倍。因此，我们报告配对余弦偏移，其中两个阈值在代数上相互抵消，使得输入缺陷的结论无需依赖任何校准即可成立。以这种方式测量，该效应分别大2.2倍和10.7倍于……（原文摘要在此处截断）

    arXiv:2611.21923v1 Announce Type: cross  Abstract: Published face-privacy methods report protection above 90% against black-box recognition models. That number comes from an evaluation protocol with two independent defects, reported in a measure that cannot size either one. The input is wrong: methods published since 2022 feed the network the whole photograph, or a detector's box, where deployed systems detect, align to a landmark template and crop. Re-scoring the same files through a realistic front end costs a standard transfer attack 56% of its dodging protection and a budget-allocating attack 7%, with no image regenerated. The measure is wrong: that loss is a difference of success rates, so it saturates, moving by 8.2x over a budget sweep while the effect beneath it moves by 1.34x. We therefore report the paired cosine shift, in which both thresholds cancel algebraically, so the input defect holds without committing to a calibration. Measured that way it is 2.2x and 10.7x larger on
    
[^171]: 下一词元训练下的潜在事实组合

    Latent Fact Composition under Next-Token Training

    [https://arxiv.org/abs/2611.21921](https://arxiv.org/abs/2611.21921)

    本文提出 Baton 系列循环 Transformer，通过在第二次循环中将中间实体作为词元嵌入传递，使逐词元训练的模型能够回答中间实体从未在两跳训练样本中出现过的两跳推理问题，并归纳出决定各模型成败的三个设计条件。

    

    对潜在两跳推理的受控研究通常训练 Transformer 只预测最终答案，而语言模型则被训练来预测每一个词元。我们探究：在每一个词元上都进行训练的模型，能否学会回答那些中间实体从未出现在任何两跳训练样本中的两跳问题。我们构建了一个包含此类问题的合成语料库，在该语料库上损失函数的每一项都可以被精确计算。标准 Transformer、循环 Transformer 以及 DiscoLoop 几乎都无法回答这些问题。因此，我们提出了 Baton 系列循环 Transformer，它们将中间实体以词元嵌入的形式传递给第二次循环。当训练时只预测答案，其三个变体中有两个几乎完美地回答了这些问题。随后我们将每个模型的成功与失败归因于三个设计条件，每次只改变一个条件的实验结果与该假设一致。当 Baton 在每一个词元上进行训练时（摘要在此处截断）

    arXiv:2611.21921v1 Announce Type: new  Abstract: Controlled studies of latent two-hop reasoning train transformers to predict only the answer, whereas language models are trained to predict every token. We ask whether a model trained on every token learns to answer two-hop questions whose intermediate entity appears in no two-hop training example. We build a synthetic corpus with such questions. On this corpus every term of the loss can be computed exactly. A looped transformer, a standard transformer and DiscoLoop answer almost none of these questions. We therefore introduce the Baton family of looped transformers. They pass the intermediate entity to the second loop as a token embedding. When trained to predict only the answer, two of its three variants answer these questions almost perfectly. We then relate the success and failure of each model to three design conditions. Experiments that change one condition at a time agree with this hypothesis. When Baton is trained on every token
    
[^172]: KVUse：面向高效LLM推荐系统的可复用KV用户记忆

    KVUse: Reusable-KV User Memory for Efficient LLM Recommenders

    [https://arxiv.org/abs/2611.21920](https://arxiv.org/abs/2611.21920)

    KVUse提出将LLM推荐系统中的用户画像以可复用的KV缓存而非文本形式存储，并通过自蒸馏训练轻量级LoRA写入器来生成这些KV缓存，从而消除排序调用中对用户画像文本的重复预填充开销，显著提升LLM推荐系统的服务效率。

    

    基于大语言模型（LLM）的记忆增强推荐系统为每个用户维护一份自然语言形式的用户画像。这些画像由用户画像更新路径随着新交互的到来不断修订，并被排序等下游任务路径所消费。由于这些用户画像以文本形式存储在用户记忆中，每一次消费它们的排序LLM调用都必须将其拼接进提示词（prompt）中，并在预填充（prefill）阶段为其计算KV缓存。鉴于这些排序LLM调用是预填充密集型的（提示词很长但解码输出很短）且被反复调用，这种对用户画像文本的重复预填充主导了整个服务开销。前缀缓存无法消除这一开销，因为在不同调用中，用户画像前面拼接的是不同的文本。我们提出KVUse，它将用户画像以可复用的键值（KV）缓存形式而非文本形式存储。KVUse通过自蒸馏训练轻量级的LoRA写入器，使得用户画像……

    arXiv:2611.21920v1 Announce Type: new  Abstract: Memory-augmented recommenders built on large language models (LLMs) maintain a natural-language profile of each user. These profiles are updated by a user profile update path that revises the profile as new interactions arrive and are consumed by downstream task paths like ranking. Because these user profiles are stored as text in the user memory, every ranking LLM call that consumes them must splice them in the prompt and calculate the KV cache of them during the prefill phase. Given these ranking LLM calls are prefill heavy (long prompt but short decoded output) and repeatedly called, these repeated prefill of user profile text dominates the serving. Prefix caching cannot remove it, because these user profiles are preceded by different text in different calls. We present KVUse, which stores the user profiles as reusable key-value (KV) caches instead of text. KVUse trains lightweight LoRA writers by self-distillation, so that the user p
    
[^173]: ALOHA Lightning：学习快速而精准的机器人操作

    ALOHA Lightning: Learning Fast and Precise Manipulation

    [https://arxiv.org/abs/2611.21919](https://arxiv.org/abs/2611.21919)

    ALOHA Lightning系统通过动态示教高速收集人类演示，并结合测试时动作平滑与掩码视觉对齐的学习流程，实现了快速且精准的双臂机器人操作。

    

    从人类演示中学习使机器人能够掌握广泛的操作技能，但学习到的策略执行速度通常远慢于普通人。这种速度差距主要源于缺乏高速收集演示数据的接口，以及训练能够稳健执行高速动作的策略十分困难。在本文中，我们提出了ALOHA Lightning（闪电ALOHA），一个用于学习快速且精准机器人操作的系统。我们的系统采用动态示教在可反向驱动的双臂平台上直观地收集接近人类速度的演示，从而获得自然且快速的轨迹。我们还提出了一套学习流程，通过测试时动作平滑实现流畅的高速执行，并通过掩码技术对齐数据收集与部署之间的视觉数据分布。在每项任务仅需50条演示的情况下，ALOHA Lightning能够自主完成双臂任务。

    arXiv:2611.21919v1 Announce Type: cross  Abstract: Learning from human demonstrations has enabled robots to acquire a wide range of manipulation skills, but learned policies typically execute far slower than ordinary humans. This speed gap is mainly due to lack of an interface for collecting demonstration data at high speed, and the difficulty in training policies that can robustly execute high-speed motions. In this paper, we present ALOHA Lightning (Flash ALOHA), a system for learning fast and precise robotic manipulation. Our system uses kinesthetic teaching to intuitively collect near-human-speed demonstrations on a backdrivable bimanual platform, yielding natural and fast trajectories. We also present a learning pipeline that enables smooth high-speed execution through test-time action smoothing and aligns the visual data distribution between data collection and deployment with masking. Given 50 demonstrations for each task, ALOHA Lightning autonomously completes bimanual tasks su
    
[^174]: 迷失在少数注意力头中：诊断与修复稀疏注意力中的块选择

    Lost in a Few Heads: Diagnosing and Repairing Block Selection in Sparse Attention

    [https://arxiv.org/abs/2611.21917](https://arxiv.org/abs/2611.21917)

    该论文提出一种校准探针方法，诊断出稀疏注意力的长上下文检索损失源于少数“盲”注意力头——其平均池化的键摘要稀释了关键 token 导致排序偏低而非预算不足，并在 InfLLM-V2 与 NOSA-8B 中验证了这一现象。

    

    稀疏注意力模型在固定预算下为每个查询只读取少量键块，因此在该预算下可能损失大量长上下文检索准确率。我们探究这种损失究竟发生在哪里。以出厂预算下的精确注意力质量选择作为预言机，我们证明选择器的问题在于对“针”（关键信息）的排序偏低，而非预算耗尽。我们通过运行一个校准探针来找出错过关键信息的注意力头：该探针以教师强制方式输入答案，并读取每个注意力头中的两个量。第一个是选择器所放弃的关键信息注意力，第二个是修复该头的选择后能恢复的答案对数概率份额，该份额沿从出厂选择器到全注意力的路径进行积分。InfLLM-V2 系列的四个检查点以及 NOSA-8B 都在相同的三个“盲”注意力头上丢失了多键检索能力，这些头虽然本会关注关键信息，但平均池化的键摘要稀释了其中单个决定性 token。

    arXiv:2611.21917v1 Announce Type: new  Abstract: Sparse attention models read a few blocks of keys for every query at a fixed budget and can lose much of their long-context retrieval accuracy at that budget. We ask where this loss is made. Using exact attention mass selection at the shipped budget as an oracle, we show that the selector under-ranks the needle rather than running out of budget. We find attention heads that miss the needle by running a calibration probe that teacher-forces the answer and reads two quantities in every head. The first is the needle attention that the selector forfeits, and the second is the share of the answer's log-probability that fixing the head's selection restores, integrated along the path from the shipped selector to full attention. Four checkpoints of the InfLLM-V2 family and NOSA-8B lose multi-key retrieval in the same three blind heads, which attend to the needle while a mean-pooled key summary dilutes its single decisive token. Full attention in
    
[^175]: 面向延迟标签与分布漂移下数据流评估的时序代数框架

    A Temporal Algebraic Framework for Data Stream Evaluation Under Delayed Labels and Distribution Shifts

    [https://arxiv.org/abs/2611.21914](https://arxiv.org/abs/2611.21914)

    该论文提出将基于混淆矩阵的代数框架扩展到数据流评估，通过指数衰减或滑动窗口进行时间加权，并以观测掩码和校准概率处理延迟与选择性标签反馈，从而在分布漂移下以每步 O(m²) 的代价区分决策边界退化与类别流行度变化。

    

    在数据流中，标签常常延迟到达或仅对部分实例可用，类别流行度会发生漂移，稀有类别会在短时间窗口中消失。此时，标准评估方法会将决策边界的退化与流行度的变化相混淆，并且在标签延迟或由模型选择性提供时产生偏差。我们将一个能够从单一混淆矩阵获得汇总（微平均）与逐类别（宏平均）度量的代数框架扩展至数据流场景，通过指数衰减或滑动窗口对计数进行时间加权；无论数据流长度如何，每步更新代价仅为 O(m^2)，且每个汇总度量始终保持为逐类别度量的加权组合。延迟和选择性反馈以观测掩码的形式纳入框架：在标签到达之前，实例以模型校准后的类别概率进行计数，该临时估计的偏差由其校准误差决定。经审计的约定对没有已验证实例的类别不报告任何数值。我们……

    arXiv:2611.21914v1 Announce Type: cross  Abstract: In data streams, labels often arrive late or only for some instances, class prevalence drifts, and rare classes vanish from short windows. Standard evaluation then confuses a degraded decision boundary with a change in prevalence, and is biased when labels are delayed or selected by the model. We extend an algebraic framework that obtains pooled (micro) and per-class (macro) measures from one confusion matrix to streams, weighting the counts in time by exponential fading or a sliding window; the update costs O(m^2) per step, whatever the stream length, and each pooled measure stays a weighted combination of the per-class ones. Delayed and selective feedback enter as an observation mask: until its label arrives, an instance is counted with the model's calibrated class probabilities, and the bias of this interim estimate is driven by their calibration error. Audited conventions report no value for a class without verified instances. We s
    
[^176]: 基于世界反馈的强化学习用于污染感知的鸟类运动建模：多实例DQN框架

    World-Grounded Reinforcement Learning for Pollution-Aware Bird Movement Modeling Using a Multi-Instance DQN Framework

    [https://arxiv.org/abs/2611.21895](https://arxiv.org/abs/2611.21895)

    该论文提出一种多实例DQN强化学习框架，以AQI作为世界反馈信号，结合真实环境数据模拟污染感知的鸟类自适应导航策略。

    

    鸟类被建模为对环境敏感的智能体，其运动反映真实的空气质量状况，其中AQI（空气质量指数）作为可测量的世界反馈信号，代表污染驱动的后果。该系统不依赖人类偏好或合成奖励，而是将环境结果作为主要训练信号，从而实现基于生态影响的策略学习。我们提出了一种多实例强化学习模拟，其中智能体在印度的二维网格表示中导航，该网格集成了预计算的AQI矩阵和地理州界。该框架结合了来自Xeno-canto数据集的鸟鸣元数据、空间离散化和真实环境数据，以学习污染约束下的自适应导航策略。为确保可扩展性和效率，用结构化的AQI映射替代了重复的实时API查询，从而在训练和执行期间实现快速查找。

    arXiv:2611.21895v1 Announce Type: new  Abstract: Birds are modeled as environmentally sensitive agents whose movement reflects real-world air quality conditions, where AQI serves as a measurable world feedback signal representing pollution-driven consequences. Instead of relying on human preference or synthetic rewards, the system uses environmental outcomes as the primary training signal, enabling policy learning grounded in ecological impact. We propose a multi-instance reinforcement learning simulation in which agents navigate a 2D grid representation of India integrated with a precomputed AQI matrix and geographic state boundaries. The framework combines bird call metadata from the Xeno-canto dataset, spatial discretization, and real environmental data to learn adaptive navigation strategies under pollution constraints. To ensure scalability and efficiency, repeated real-time API queries are replaced by a structured AQI mapping that enables fast lookup during training and execution
    
[^177]: WM-Quant：面向JEPA风格世界模型的感知推演的后训练量化

    WM-Quant: Rollout-Aware Post-Training Quantization for JEPA-Style World Models

    [https://arxiv.org/abs/2611.21882](https://arxiv.org/abs/2611.21882)

    WM-Quant是首个针对JEPA风格世界模型的后训练量化框架，通过早期编码器精度、控制路径保持和感知推演缩放三项技术，解决了自回归MPC推演中量化误差逐步累积导致动作排序近乎随机的问题。

    

    据我们所知，我们提出的WM-Quant是首个专门为JEPA风格世界模型设计的后训练量化（PTQ）框架。尽管世界模型的参数量通常相对较少，但动作规划算法往往需要数千次推理，这会迅速带来高昂的计算成本，而后训练量化正是降低这一成本的自然途径。然而，我们证明，在使用自回归模型预测控制（MPC）推演的世界模型中，量化误差会随每个预测器步骤不断累积，导致量化模型给出的动作排序几乎变为随机。WM-Quant通过在现有PTQ方法基础上构建的三个组件来解决这一问题：早期编码器精度、控制路径保持以及感知推演的缩放。早期编码器精度保护准确的潜在状态与目标信息，以减轻长期误差传播；控制路径保持使世界模型预测器能够区分……（原文摘要到此截断）

    arXiv:2611.21882v1 Announce Type: new  Abstract: We present WM-Quant, to our knowledge, the first post-training quantization (PTQ) framework designed specifically for JEPA-style world models. Although world models typically feature relatively few parameters, action planning algorithms often require thousands of inferences, which can quickly become computationally expensive. PTQ is a natural way to reduce this cost. However, we show that in world models using autoregressive model predictive control (MPC) rollouts, quantization error can compound with each predictor step, resulting in action rankings from the quantized model becoming near-random. WM-Quant addresses this with three components that build on existing PTQ methods: early-encoder precision, control-path preservation, and rollout-aware scaling. Early-encoder precision protects accurate latent state and goal information to mitigate long-term error propagation. Control-path preservation allows the world model predictor to disting
    
[^178]: 用于安全控制器选择的无抵消约束SquareCB算法

    Cancellation-Free Constrained SquareCB for Safe Controller Selection

    [https://arxiv.org/abs/2611.21880](https://arxiv.org/abs/2611.21880)

    该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。

    

    我们研究约束上下文老虎机问题，其中学习者在每一轮从 $K$ 个动作中选择一个，以最小化损失，同时使其动作的期望成本保持在预算以下。现有的到在线回归的归约方法使用从过去约束残差中学习到的拉格朗日乘子对成本进行加权，因此某些轮次的松弛量可能会抵消其他轮次的违约。我们转而在成本上施加修正权重来运行 Foster 等人（2020）提出的 SquareCB：当估计成本在预算之内时权重为零，否则权重恰好足够大以将成本拉回预算。在可实现性假设以及具有边际 $\delta$ 的 Slater 条件下，该算法达到遗憾 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 和无抵消违约 $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，其中 $\mathrm{Reg}_{\mathsf{Sq}}$ 是回归预言机的误差。当预算必须在（原文摘要在此处截断，后续内容缺失）……

    arXiv:2611.21880v1 Announce Type: cross  Abstract: We study constrained contextual bandits, in which a learner chooses one of $K$ actions in each round to minimize loss while keeping the expected cost of its play below a budget. Existing reductions to online regression weight the cost by a Lagrange multiplier learned from past constraint residuals, so slack in some rounds can offset violations in others. We instead run SquareCB proposed in \citep{foster2020beyond} with a rectified weight on the cost: it is zero when the estimated cost is within the budget, and otherwise just large enough to bring it back to the budget. Under realizability and Slater's condition with margin $\delta$, the algorithm attains regret $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ and cancellation-free violation $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$, where $\mathrm{Reg}_{\mathsf{Sq}}$ is the error of the regression oracle. The same bounds hold when the budget must be met in
    
[^179]: 稀疏模型变化下在线强化学习的自适应记忆

    Adaptive Memory for Online Reinforcement Learning under Sparse Model Changes

    [https://arxiv.org/abs/2611.21879](https://arxiv.org/abs/2611.21879)

    针对部署间稀疏变化的强化学习环境，提出一种自适应记忆算法，为每个状态-动作对动态决定复用多少历史观测，从而在利用过去数据与避免已变化条目带来的估计偏差之间取得平衡。

    

    我们研究底层环境在部署之间发生变化的折扣强化学习（RL）问题。每次部署提供来自一个固定MDP的H个观测，奖励函数和转移核在每次部署内部保持固定；而在不同部署之间，最多有比例为 ε>0 的奖励和转移核条目可能发生变化。在这种设定下，从头学习每个待部署模型可能会丢弃以往部署中有用的观测，而将所有过去的观测汇集在一起则可能使已经发生变化条目的估计产生偏差。为此，我们提出一种算法：针对每个状态-动作对，选择复用多少历史数据来估计奖励和下一状态的概率。我们从当前部署的观测开始，按由近及远的顺序逐个加入更早部署的观测。在每次添加后，我们检验是否单一的……

    arXiv:2611.21879v1 Announce Type: new  Abstract: We study the discounted reinforcement learning (RL) problem in a setting where the underlying environment changes between deployments. Each deployment provides $H$ observations from a fixed MDP. The reward function and transition kernel are fixed within each deployment. Across deployments, at most an $\epsilon >0$ fraction of the reward and transition kernel entries may change. In this setting, learning each model to deploy from scratch may discard useful observations from past deployments, whereas pooling all past observations can bias the estimates of entries that have already changed. We then propose an algorithm that for each state and action pair, selects how much history to reuse to estimate the reward and probabilities of the next state. We start with observations from the current deployment and add earlier deployment observations one at a time, the most recent first. After each addition, we check whether a single average reward a
    
[^180]: PromptPack：通过免训练方式增强扩散编辑器，实现与意图对齐的空间与语义控制

    PromptPack: Training-Free Augmentation of Diffusion Editors for Intent-Aligned Spatial and Semantic Control

    [https://arxiv.org/abs/2611.21865](https://arxiv.org/abs/2611.21865)

    PromptPack 通过免训练地为扩散编辑器引入“保留/允许/修改”像素角色划分和区域到提示的空间映射两种显式编辑指南，实现了与用户意图对齐的空间与语义精确控制。

    

    arXiv:2611.21865v1 公告类型：cross 摘要：许多图像编辑任务要求将修改限制在预期的空间区域内，同时精确保留用户指定的源内容。基于指令的编辑流程往往使这类用户意图未被充分明确，而修补类编辑器仅以明确定义的编辑支持区域为生成条件，却不考虑各个编辑概念的空间分配。PromptPack 通过为扩散编辑器配备两种显式的“编辑指南”来同时克服这两个局限：一是编辑角色划分，将每个像素标注为“保留”、“允许”或“修改”，以指定允许在何处进行编辑；二是区域提示分配，将空间区域映射到文本提示，以指定哪些概念应用于何处。这些指南可以手动指定，或者对于源对象编辑，可以从基于文本的 SAM 3 掩码自动推导。在无需重新训练的情况下，PromptPack 将独立嵌入的区域提示注入编辑器的固定长度文本序列中，并利用编辑角色划分来……

    arXiv:2611.21865v1 Announce Type: cross  Abstract: Many image-editing tasks require confining changes to intended spatial regions while exactly preserving user-designated source content. Instruction-driven editing pipelines often leave such user-intent underspecified, while inpainting editors condition generation on explicitly defined edit support but not on the spatial assignment of individual edit concepts. PromptPack addresses both limitations by equipping diffusion editors with two explicit 'Edit Guides': an edit-role partition that labels each pixel as 'Keep,' 'Permit,' or 'Modify' to specify where edits may occur, and a regional-prompt assignment that maps spatial regions to text prompts to specify which concepts apply where. These guides may be manually specified or, for source-object edits, automatically derived from a text-grounded SAM 3 mask. Without retraining, PromptPack injects independently embedded regional prompts into the editor's fixed-length text sequence and uses th
    
[^181]: tmol：面向机器学习的Rosetta能量函数GPU加速实现

    tmol: a GPU-accelerated implementation of the Rosetta energy function for machine learning

    [https://arxiv.org/abs/2611.21857](https://arxiv.org/abs/2611.21857)

    tmol是一个GPU原生的PyTorch库，实现了最先进的RosettaGenFF力场，支持高效的批量可微分打分与全结构优化（吞吐量最高提升362倍），可作为奖励函数加速生成式蛋白质设计。

    

    物理启发的力场可以补充生成式机器学习模型，在训练分布之外展现出更好的泛化能力。然而，这些力场通常计算速度较慢，其导数也难以集成到机器学习框架中。我们提出了tmol，一个实现了最先进的RosettaGenFF力场的GPU原生PyTorch库。tmol支持高效的批量可微分打分和全结构优化，批量模式下吞吐量提升最高可达362倍。作为生成式蛋白质设计中的奖励函数，tmol能够增加设计界面上的氢键和水桥数量。

    arXiv:2611.21857v1 Announce Type: new  Abstract: Physics-inspired forcefields complement generative machine learning models, showing better generalization beyond the training distribution. However, these forcefields are typically slow to evaluate, and their derivatives are difficult to integrate into ML frameworks. We introduce tmol, a GPU-native PyTorch library implementing the state-of-the-art RosettaGenFF forcefield. tmol supports efficient batched differentiable scoring and whole-structure refinement with up to 362-fold throughput increase (batched). Used as a reward in generative protein design, tmol increases hydrogen bonds and water bridges at designed interfaces.
    
[^182]: MoE扩散解码的隐性代价：剖析与修复采样器

    The Hidden Cost of MoE Diffusion Decoding: Profiling and Fixing the Sampler

    [https://arxiv.org/abs/2611.21852](https://arxiv.org/abs/2611.21852)

    该论文发现量化后MoE扩散语言模型的解码瓶颈在于画布级logits采样流水线而非专家矩阵计算，并提出两个无损优化层级（比特精确与精度对等）将其消除，从而大幅加速解码。

    

    掩码扩散语言模型通过迭代去噪整个token画布来进行解码，因此每次前向传播都需要在整个“画布×词表”的logits张量上运行逐token的采样与停止流水线。我们在带宽受限的边缘加速器（NVIDIA GB10）上对一个26B参数的扩散混合专家模型进行了单用户解码的性能剖析，并报告了一个容易被忽视的发现：在权重量化之后，这一logits流水线（而非专家GEMM计算）成为解码时间的最大消耗者。与自回归解码不同——自回归解码每提交一个token只需对单个位置的logits进行采样——扩散解码每输出一个token需要处理S倍的logits条目，其中S为去噪步数。我们证明该流水线在很大程度上是冗余的或可融合的，并以两个无损层级将其移除：一个是比特精确层级，其输出与参考解码器字节完全一致；另一个是精度对等层级，使用融合的单（原文摘要在此处截断）。

    arXiv:2611.21852v1 Announce Type: cross  Abstract: Masked diffusion language models (dLLMs) decode by iteratively denoising a full canvas of tokens, so every forward pass runs a per-token sampling and stopping pipeline over the entire canvas-by-vocabulary logits tensor. We profile single-user decoding of a 26B-parameter diffusion Mixture-of-Experts model on a bandwidth-bound edge accelerator (NVIDIA GB10) and report a result that is easy to miss: after weight quantisation, this logits pipeline, not the expert GEMMs, is the single largest consumer of decode time. Unlike autoregressive decoding, where sampling touches one position's logits per committed token, diffusion decoding processes S times as many logit entries per output token, where S is the number of denoising steps. We show this pipeline is largely redundant or fusable and remove it in two lossless tiers: a bit-exact tier whose output is byte-identical to the reference decoder, and an accuracy-parity tier using a fused single-
    
[^183]: 离散里奇曲率何时有效？面向蛋白质相互作用预测的曲率增强图卷积网络的边级分析

    When Does Discrete Ricci Curvature Help? An Edge-Level Analysis of Curvature-Enhanced GCNs for PPI Prediction

    [https://arxiv.org/abs/2611.21849](https://arxiv.org/abs/2611.21849)

    本文通过边级分析揭示，曲率增强GCN在PPI预测中的性能提升主要来自低度数边，而低曲率瓶颈边的性能反而系统性下降，从而挑战了“曲率加权通过缓解过度挤压带来改进”这一传统解释。

    

    曲率增强图卷积网络（CGCN）将基于最优传输理论的Ollivier-Ricci曲率（ORC）融入消息传递机制，其动机源于这样的主张：曲率加权的边能够缓解结构瓶颈处的“过度挤压”问题。但CGCN的改进实际上是否来自缓解过度挤压？我们在四个真实的蛋白质-蛋白质相互作用（PPI）数据集上研究这一问题，在单个边的层面上而非仅通过聚合指标来比较CGCN与标准GCN。虽然CGCN在所有四个数据集上都提升了聚合的AUC、AUPR和F1指标，但这种提升几乎完全集中在低度数边上；而在四个数据集中的三个里，低曲率（瓶颈）边的性能反而出现系统性下降，这与过度挤压动机所预测的恰恰相反。通过直接考察学习到的边权重函数，并结合聚类有效性指标和边距离比率……

    arXiv:2611.21849v1 Announce Type: new  Abstract: Curvature-enhanced graph convolutional networks (CGCN) incorporate Ollivier-Ricci curvature (ORC), grounded in optimal transport theory, into message passing, motivated by the claim that curvature-weighted edges alleviate "over-squashing" at structural bottlenecks. But does CGCN's improvement actually come from mitigating over-squashing? We investigate this on four real-world protein-protein interaction (PPI) datasets, comparing CGCN against a standard GCN at the level of individual edges rather than aggregate metrics alone. While CGCN improves aggregate AUC, AUPR, and F1 across all four datasets, this gain is concentrated almost entirely on low-degree edges, while performance on low-curvature (bottleneck) edges systematically degrades in three of four datasets, the opposite of what the over-squashing motivation predicts. Examining the learned edge-weight function directly, together with clustering-validity metrics, edge-distance ratios,
    
[^184]: Inventa：用于自主科学发现的去中心化自我改进智能体

    Inventa: Decentralized Self-Improving Agents for Autonomous Scientific Discovery

    [https://arxiv.org/abs/2611.21847](https://arxiv.org/abs/2611.21847)

    Inventa是一个去中心化的自动化科学发现框架，通过自主研究单元生成候选想法谱系、编排智能体自适应调整变异策略、以及验证器筛选并共享经验证的知识，解决了多智能体系统中想法趋同、错误信息传播和无效搜索的问题。

    

    大语言模型（LLM）智能体能够生成研究思路并通过迭代实验来促进科学发现。然而，增加智能体的数量或迭代次数并不必然带来更多样或更可行的想法。智能体可能会收敛到相似的候选方案，通过共享记忆组件传播缺乏依据的断言，并反复应用不再带来任何改进的搜索策略。我们提出了Inventa，一个采用自主研究单元和经验证共享知识的自动化科学发现框架。每个单元使用一个变异智能体来创建候选想法的谱系。本地编排智能体周期性地调整变异器的运行框架。当进展停滞时，编排智能体会替换其模型或重启想法演化过程。验证器评估每个候选想法，并记录经验证的改进、有用的组件和约束违规情况。

    arXiv:2611.21847v1 Announce Type: cross  Abstract: Large language model (LLM) agents can generate research ideas and improve scientific discovery through iterative experimentation. However, increasing the number of agents or iterations does not automatically translate to producing more diverse or feasible ideas. Agents may converge on similar candidates, propagate unsupported claims through shared memory components and repeatedly apply search strategies that no longer yield any improvements.   We introduce Inventa, a framework for automated scientific discovery that employs autonomous research units and verified shared knowledge. Each unit creates a lineage of candidate ideas using a mutator agent. A local orchestrator agent periodically adapts the mutator's harness. When progress stalls, the orchestrator agent replaces its model or restarts the idea evolution process. Verifiers evaluate each candidate idea and record verified improvements, useful components and constraint violations i
    
[^185]: ℓ_p 几何下去偏差与并行化的重尾随机凸优化

    Bias-Reduced and Parallel Heavy-Tailed Stochastic Convex Optimization in $\ell_p$ Geometries

    [https://arxiv.org/abs/2611.21844](https://arxiv.org/abs/2611.21844)

    本文在重尾噪声模型下研究 ℓ_p 几何中的并行随机凸优化，提出了一类结合随机去偏差技术与多层蒙特卡罗方法的新颖 ODC_p 算法，可高效估计一致凸目标函数的最优值并给出相应的收敛速率。

    

    我们考虑并行随机凸优化（SCO）这一基本问题，其目标是在每一轮可以多次并行查询矩有界的随机梯度预言机的条件下，近似最小化一个凸目标函数。以往的大多数结果都集中在方差有界的欧几里得（ℓ_2 范数）设定上，即目标函数在 ℝ^d 中 ℓ_2 有界的可行域上关于 ℓ_2 范数是 Lipschitz 的；而本工作则研究重尾噪声模型下 2 ≤ p < ∞ 的 ℓ_p 设定中的并行随机凸优化。我们首先引入了一类新的 ℓ_p 最优距离收敛（ODC_p）算法，借鉴了先前的随机去偏差技术 [ACJ+21]，该算法与多层蒙特卡罗方法相结合，可以高效地估计一致凸目标函数的最优值。作为 ODC_p 算法的结果，我们首先建立了相应速率……（摘要在此处截断）

    arXiv:2611.21844v1 Announce Type: cross  Abstract: We consider the fundamental problem of parallel stochastic convex optimization (SCO), whose aim is to approximately minimize a convex objective when given the ability to query a moment-bounded stochastic gradient oracle multiple times in parallel per round. Whereas most previous results have focused on the variance-bounded Euclidean ($\ell_2$-norm) setting, meaning the objective is Lipschitz with respect to the $\ell_2$-norm over an $\ell_2$-bounded feasible set in $\mathbb{R}^d$, in this work we study parallel SCO in $\ell_p$ settings for $2 \le p < \infty$ under a heavy-tailed noise model. We begin by introducing a new class of $\ell_p$-optimal-distance-convergence (ODC$_p$) algorithms, drawing on prior stochastic bias-reduced techniques [ACJ+21], which combine with multilevel Monte Carlo methods to efficiently estimate the optimum of uniformly convex objectives. As a consequence of our ODC$_p$ algorithms, we first establish rates fo
    
[^186]: 通过非局域博弈实现概率分布输运中的量子-经典分离

    Quantum-Classical Separation in Probability Distribution Transport via Nonlocal Games

    [https://arxiv.org/abs/2611.21840](https://arxiv.org/abs/2611.21840)

    该论文建立了一个量子-经典分离：常数深度的局域量子电路能够精确且可逆地连接两个经典概率分布，而经典随机输运实现同样的连接需要大得多的空间范围，并借助非局域博弈给出了图上的一般性构造原理。

    

    在多体物理和生成式人工智能中，局域性为“相”的等价性提供了一个自然的定义：当局域动力学能够可逆地将一个量子态或经典概率分布变换为另一个时，两者被视为处于同一相。由此引出一个根本性问题：局域量子动力学能否以局域经典动力学无法实现的方式输运经典概率分布？我们在分布输运中建立了一个量子-经典分离：两个经典分布可以被常数深度的局域量子电路精确且可逆地连接起来，而经典随机输运要实现同样的连接则需要大得多的空间范围。我们在一维链上明确演示了这种分离，并基于非局域博弈发展出一种适用于图结构的一般性构造原理。此外，源分布本身可以由一个简单的局域经典过程生成。

    arXiv:2611.21840v1 Announce Type: cross  Abstract: In many-body physics and generative artificial intelligence, locality provides a natural notion of phase equivalence: two quantum states or classical probability distributions are regarded as in the same phase when local dynamics can reversibly transform one into the other. This raises a fundamental question: can local quantum dynamics transport classical probability distributions in ways that local classical dynamics cannot? We establish a quantum-classical separation in distribution transport: two classical distributions can be connected exactly and reversibly by a constant-depth local quantum circuit, whereas classical stochastic transport requires a substantially larger spatial range. We demonstrate this separation explicitly on a one-dimensional chain and develop a general construction principle on graphs, based on nonlocal games. Moreover, the source distribution can itself be generated by a simple local classical process, establ
    
[^187]: VisHead-Spec：面向多模态投机解码的头路由视觉注意力

    VisHead-Spec: Head-Routed Visual Attention for Multimodal Speculative Decoding

    [https://arxiv.org/abs/2611.21839](https://arxiv.org/abs/2611.21839)

    该论文提出VisHead-Spec，通过发现草稿模型中仅需少量键值头直接接收目标模型视觉注意力的稀疏模式，实现了高并发场景下依然高效的多模态投机解码加速。

    

    投机解码能够在保持目标模型输出不变的前提下加速视觉语言模型（VLM），但在高并发服务场景下，随着批处理规模的增大，系统越来越受计算能力限制，其加速效果随之减弱。这一问题在多模态投机解码中尤为突出，因为提示词中图像token的长序列会显著增加草稿token预测阶段的计算成本。块扩散式草稿模型（如DFlash、DSpark）可以通过一次前向传播预测多个草稿token来有效提升草稿模型的效率，但它们在高并发下对VLM的加速效果仍然有限（不足9%）。为了理解这一差距，我们研究了从目标模型到草稿模型之间不同模态的信息流，并识别出目标模型与草稿模型之间视觉注意力的稀疏模式：草稿模型中只有一小部分键值头需要直接接收目标模型的视觉注意力信息……

    arXiv:2611.21839v1 Announce Type: new  Abstract: Speculative decoding accelerates vision-language models(VLMs) while preserving the target model's outputs, but its speedup diminishes under high concurrency serving as the system becomes more compute bounded as the batch size increases. This problem is pronounced for multimodal speculative decoding as the long range of the image tokens in the prompt can significantly increase the computational cost during draft token prediction. The block-diffusion drafters (e.g. DFlash, DSpark) can effectively speed up the draft model efficiency by predicting multiple draft tokens in one forward pass, but their speedup for VLMs under high concurrency is still modest (<9%). In order to understand this gap, we examine the information flow between different modalities from target model to draft model and identify the sparsity pattern of visual attention between the target and draft models: only small set of key-value heads in the draft model need direct vi
    
[^188]: 具有不确定性任务需求的异构团队协调

    Coordination of Heterogeneous Teams with Uncertain Task Requirements

    [https://arxiv.org/abs/2611.21828](https://arxiv.org/abs/2611.21828)

    本文提出了基于CAAT图表示和图神经网络策略的分层规划器，能够联合协调异构智能体团队同时进行任务需求发现与任务执行，比独立优化两者的基线方法更快地完成任务。

    

    协调一组异构智能体来完成需求未知的任务是一个具有挑战性的多智能体规划问题。智能体在发现任务需求的能力和执行任务的能力上存在差异，因此规划器必须协调任务发现、执行与调度。本文提出了一种分层规划器，该规划器使用图神经网络策略为智能体分配动作，然后利用最短路径算法寻找通往动作位置的路径。我们方法的核心是CAAT图，它是对由智能体、任务、能力和动作组成的规划上下文的图表示。我们证明了所提出的学习型规划器能够联合协调异构团队，在发现任务需求的同时完成任务，且相比独立优化任务发现与任务执行的基线方法耗时更短。此外，我们展示了我们的策略……

    arXiv:2611.21828v1 Announce Type: cross  Abstract: Coordinating a team of heterogeneous agents to complete tasks with unknown requirements is a challenging multi-agent planning problem. Agents differ in their ability to discover task requirements and in their ability to engage tasks, so the planner must coordinate task discovery, execution, and staging. This paper introduces a hierarchical planner that uses a graph neural network policy to assign actions to agents, then finds a path to the action's location using a shortest-path algorithm. The central part of our approach is the CAAT graph, which is a graph representation of the planning context consisting of the agents, tasks, capabilities, and actions. We demonstrate that our learned planner can jointly coordinate a heterogeneous team to simultaneously discover task requirements and complete tasks in a shorter time compared to baselines that optimize task discovery and task engagement independently. Furthermore, we show that our poli
    
[^189]: 面向识别犯罪报告潜在差异的城市表征学习

    Urban Representation Learning for Identifying Potential Disparities in Crime Reporting

    [https://arxiv.org/abs/2611.21825](https://arxiv.org/abs/2611.21825)

    提出了一种自监督城市表征学习框架，通过邻居画像差异（NPD）指标量化各区域与其表征相似同伴区域的报告画像差异，从而识别开放犯罪数据中的潜在报告偏差。

    

    开放犯罪数据能够支持城市安全分析，但它们反映的是已报告的案件而非犯罪活动的完整分布，这可能给下游分析带来空间扭曲。我们提出了一种自监督的城市表征学习框架，仅利用带地理坐标的犯罪记录来识别相对于表征相似同伴区域的异常报告模式。该方法构建了互补的小时级、周级和静态报告画像，在地理自监督下学习特定尺度的表征，并将其整合到统一的潜在空间中。随后，利用邻居画像差异指标来量化每个区域的报告画像与其检索到的同伴区域画像之间的不一致程度。在墨西哥城超过180万条犯罪记录（聚合为1,058个H3区域）上的实验表明，所提出的表征方法达到了0.4052的平均轮廓系数，相比之下为0.2...

    arXiv:2611.21825v1 Announce Type: new  Abstract: Open crime data support urban safety analysis but reflect reported incidents rather than the complete distribution of criminal activity, potentially introducing spatial distortions in downstream analysis. We propose a self-supervised urban representation learning framework for identifying unusual reporting patterns relative to representation-similar peers using only georeferenced crime records. The method constructs complementary hourly, weekly, and static reporting profiles, learns scale-specific representations under geographic self-supervision, and integrates them into a unified latent space. Neighbor Profile Divergence (NPD) is then used to quantify disagreement between each region's reporting profile and those of its retrieved peers. Experiments on more than 1.8 million crime records from Mexico City, aggregated into 1,058 H3 regions, show that the proposed representation achieves a mean Silhouette score of 0.4052, compared with 0.2
    
[^190]: 数据驱动的三维表示变长压缩

    Data-Driven Variable-Length Compression of 3D Representations

    [https://arxiv.org/abs/2611.21802](https://arxiv.org/abs/2611.21802)

    提出一种数据驱动的可学习变长压缩方法，利用数据集整体结构并根据单个形状复杂度自适应调整压缩表示大小，仅通过快速前向推理即可改善网格和带符号距离场的率失真性能。

    

    压缩三维形状对许多应用的效率而言至关重要，然而现有方法要么无法利用数据集级别的统计信息，要么对所有形状采用固定的压缩率。我们提出了一种可学习的变长压缩方法，该方法在利用训练数据集整体结构的同时，仅通过快速的前向推理就能使压缩表示的大小自适应地匹配各个形状的复杂度。我们的方法作用于通用的潜在表示，并能同时改善网格和带符号距离场的率失真性能。代码可在 https://github.com/voilalab/varlen-3D 获取。

    arXiv:2611.21802v1 Announce Type: cross  Abstract: Compressing 3D shapes is important for the efficiency of many applications, yet existing methods either cannot leverage dataset-level statistics or use a fixed compression ratio for all shapes. We propose a learned variable-length compression method that exploits structure across a training dataset while adapting compressed representation size to individual shape complexity using only fast feed-forward inference. Our method operates on general latent representations and improves rate-distortion for both meshes and signed distance fields. Code is available at https://github.com/voilalab/varlen-3D.
    
[^191]: 思维链可控性可能降低思维链可监控性

    Chain-of-Thought Controllability Can Reduce Chain-of-Thought Monitorability

    [https://arxiv.org/abs/2611.21799](https://arxiv.org/abs/2611.21799)

    论文通过人设控制训练发现，即使训练中不含任何规避监控的示例，仅仅让模型学会以不同人设口吻书写推理，就能显著降低其在执行隐蔽副任务时思维链的可监控性（g-mean²从0.94降至0.22），表明思维链可控性的增强可能削弱思维链监控的有效性。

    

    思维链监控是一种很有前景的工具，可用于检测AI智能体中的隐藏目标、偏离任务行为及其他故障。其中一个风险是，模型可能会刻意塑造自己的推理过程，使其对监控者不透明。我们通过“人设控制训练”对这一风险进行了压力测试：在用多样化人设口吻改写的数学推理轨迹上进行监督微调。尽管训练数据中不包含任何规避监控的示例，这仍诱导出了更广泛地控制思维链的能力。在隐蔽副任务评估中，gpt-oss-20b模型体完成隐藏副任务的那些思维链轨迹，所包含的可供监控的证据比基础模型的更少。在各项评估与不同思维链长度上按副任务成功次数加权平均后，思维链可监控性指标g-mean²从基础模型的0.94下降至模型体在无隐藏指令情况下的0.68，而在明确被指示隐藏时进一步降至0.22。在干预评估中……（原文摘要在此处截断）

    arXiv:2611.21799v1 Announce Type: new  Abstract: Chain-of-thought (CoT) monitoring is a promising tool for detecting hidden goals, off-task behavior, and other failures in AI agents. One risk is that models could deliberately shape their reasoning to make it opaque to monitors. We stress-test this risk through persona-control training: supervised fine-tuning on mathematical reasoning traces rewritten in the voices of diverse personas. This induces a broader ability to control CoT, even though training contains no monitor-evasion examples. In covert side-task evaluations, traces in which a gpt-oss-20b model organism completes the hidden side task contain less monitorable evidence than the base model's. Averaged across evaluations and CoT lengths, weighted by side-task-success counts, the CoT monitorability metric g-mean$^2$ falls from 0.94 for the base model to 0.68 for the model organism without a hiding instruction, and to 0.22 when explicitly instructed to hide. In intervention evalu
    
[^192]: 正则化分数中心化

    Regularized Score Centering

    [https://arxiv.org/abs/2611.21791](https://arxiv.org/abs/2611.21791)

    本文提出正则化分数中心化（RSC），通过引入反向 KL 惩罚使学习器的下一词元分布保持接近行为策略，从而限制策略不匹配、稳定异步离策略强化学习，并在理论上证明 KL 正则化改善了收敛保证。

    

    异步强化学习提高了训练吞吐量，但来自过时策略的数据以及训练与推理之间的数值不匹配可能会使学习过程变得不稳定。为了稳定离策略学习，分数中心化（SC）消除了奖励基线对预期更新的影响。我们的梯度诊断显示，与重要性采样（IS）相比，SC 产生的极端梯度更少，但当学习器偏离生成数据的行为策略太远时，SC 仍然可能失效。为了限制这种策略不匹配，我们引入了正则化分数中心化（RSC），它通过添加一个反向 Kullback-Leibler（KL）惩罚，鼓励学习器的下一词元分布保持接近行为策略的下一词元分布。RSC 既可以与截断重要性采样（TIS）结合使用，也可以单独使用，TIS 会对词元重要性权重设置上限。我们的理论分析支持这一方法，表明 KL 正则化改善了收敛保证。

    arXiv:2611.21791v1 Announce Type: new  Abstract: Asynchronous reinforcement learning increases training throughput, but data from stale policies and numerical mismatch between training and inference can destabilize learning. To stabilize off-policy learning, score centering (SC) removes the effect of reward baselines on the expected update. Our gradient diagnostics show less extreme gradients with SC than with importance sampling (IS), but SC can still fail when the learner moves far from the behavior policy that generated the data. To limit this policy mismatch, we introduce Regularized Score Centering (RSC), which adds a reverse Kullback-Leibler (KL) penalty to encourage the learner's next-token distribution to stay close to the behavior policy's next-token distribution. RSC works with or without truncated importance sampling (TIS), which caps token importance weights. Our theoretical analysis supports this approach, showing that KL regularization improves convergence guarantees. We 
    
[^193]: 更好的校准，更好的规划：面向石油泄漏监测的不确定性感知多自主水面航行器（ASV）信息路径规划

    Better Calibrated, Better Planned: Uncertainty-Aware Multi-ASV Informative Path Planning for Oil Spill Monitoring

    [https://arxiv.org/abs/2611.21783](https://arxiv.org/abs/2611.21783)

    本文将证据深度学习（EDL）引入作为经不确定性校准的信息模型，研究不确定性校准对多ASV协同信息路径规划的影响，证明模型校准质量直接决定石油泄漏监测的规划效果。

    

    使用自主水面航行器（ASV）编队监测石油泄漏，需要在紧迫的时间和能量预算下，基于稀疏且含噪声的测量数据对非均质的污染物场进行重建。信息路径规划（IPP）通过将推断场及其不确定性的重建模型与决定下一步采样位置的规划器相耦合来解决这一问题。以往的工作主要聚焦于规划策略，并大量依赖高斯过程（GP），而对模型不确定性的校准则研究不足。我们研究了不确定性校准如何影响多航行器IPP，其中一组ASV通过共享的不确定性地图进行在线协同。我们引入证据深度学习（EDL）作为经不确定性校准的信息模型。在基于物理的石油泄漏仿真中，我们在闭环环境下将EDL、蒙特卡洛Dropout和深度集成与GP进行对比，并结合五种规划器进行基准测试。在60张地图上，对……的选择（摘要在此处被截断）

    arXiv:2611.21783v1 Announce Type: cross  Abstract: Monitoring oil spills with fleets of Autonomous Surface Vehicles (ASVs) requires mapping a heterogeneous pollutant field from sparse, noisy measurements under tight time and energy budgets. Informative Path Planning (IPP) addresses this problem by coupling a reconstruction model, which infers the field and its uncertainty, with a planner that decides where to sample next. Prior work has focused on planning strategies and has largely relied on Gaussian Processes (GPs), leaving the calibration of the model's uncertainty underexplored. We study how uncertainty calibration conditions multi-vehicle IPP with a fleet of ASVs that coordinates online through a shared uncertainty map. We introduce Evidential Deep Learning (EDL) as an uncertainty-calibrated information model. We benchmark EDL, Monte Carlo Dropout and Deep Ensembles against a GP in closed loop with five planners on physics-based oil spill simulations. Across 60 maps, the choice of
    
[^194]: 使用表格扩散模型学习海表ΔpCO₂的校准且可采样的预测分布

    Learning Calibrated and Samplable Predictive Distributions of Surface-Ocean $\Delta p$CO$_2$ Using Tabular Diffusion Models

    [https://arxiv.org/abs/2611.21769](https://arxiv.org/abs/2611.21769)

    该论文提出使用基于梯度提升树评分函数的条件扩散模型Treeffuser，从13个环境、空间和时间预测变量中学习海表ΔpCO₂的完整条件预测分布，在SOCAT测试集上实现了最低的CRPS（8.229 μatm）和最接近名义水平的覆盖率，为海表CO₂估算提供了校准且可采样的不确定性量化方法。

    

    海洋吸收了近30%的人为CO₂排放，但量化这一碳汇需要从时空上稀疏的观测数据中估算海表CO₂。大多数现有的机器学习估计方法是确定性的，即每个网格点只给出单一数值，其不确定性通过残差、集成或预设分布进行事后近似。我们转而学习每个观测点的条件预测分布：使用Treeffuser——一种采用梯度提升树评分函数的条件扩散模型——我们从13个环境、空间和时间预测变量中估计ΔpCO₂（海-气pCO₂差）的完整预测分布。在留出的SOCAT测试集上，Treeffuser在所评估的不确定性量化方法中取得了最低的连续排序概率评分（CRPS，8.229 μatm），并且其覆盖率最接近名义的68%/95%水平（0.686/0...）

    arXiv:2611.21769v1 Announce Type: new  Abstract: The ocean absorbs almost 30% of anthropogenic CO$_2$ emissions, but quantifying this sink requires estimating surface-ocean CO$_2$ from observations that are sparse in space and time. Most existing machine-learning estimates of this field are deterministic, giving a single value per grid point, with uncertainty approximated post hoc via residuals, ensembles, or prescribed distributions. We instead learn a conditional predictive distribution for each observation: using Treeffuser, a conditional diffusion model with gradient-boosted-tree score functions, we estimate the full predictive distribution of $\Delta p$CO$_2$ (the sea-air $p$CO$_2$ difference) from 13 environmental, spatial, and temporal predictors. On a held-out SOCAT test set, Treeffuser attains the lowest continuous ranked probability score among the uncertainty-quantification methods evaluated (8.229 $\mu$atm) and the coverage closest to the nominal 68% / 95% levels (0.686 / 0
    
[^195]: 大语言模型对齐的间隔动力学

    Margin Dynamics for Large Language Model Alignment

    [https://arxiv.org/abs/2611.21766](https://arxiv.org/abs/2611.21766)

    本文通过对齐方法在概率单纯形上的向量场建模并求解一维间隔常微分方程的闭式解，首次从理论上揭示 DPO 间隔无界增长而 IPO 间隔收敛于有限值的本质差异，并预测了标签噪声导致 DPO 定性相变及非对称加权下梯度集中对可训练性的影响，在五个模型家族和四个基准上得到验证。

    

    实践者目前主要依靠反复试错来选择大语言模型对齐方法。我们为其提供了理论基础：每种成对比较方法都在概率单纯形上定义了一个向量场，用以支配模型在偏好与非偏好补全之间如何重新分配概率质量。将该问题投影到单个偏好/非偏好补全对上，可简化为一维的间隔常微分方程，且具有闭式解。这些解揭示了：DPO 的间隔会无界增长（在梯度流极限下按 $O(\log t)$ 对数级增长），而 IPO 的间隔则收敛到有限的 $1/(2\tau)$；标签噪声会使 DPO 发生定性相变（从无界变为有界），但对 IPO 只有定量层面的影响；非对称加权下的梯度集中决定了大规模训练时的可训练性。我们在涵盖四个基准测试的五个模型家族上验证了上述全部三项预测。闭式解还使有原则的方法设计成为可能：我们……

    arXiv:2611.21766v1 Announce Type: cross  Abstract: Practitioners choose among LLM alignment methods largely by trial and error. We provide a theoretical foundation: each pairwise method defines a vector field on the probability simplex that governs how the model redistributes mass between preferred and dispreferred completions. Projecting onto a single preferred/dispreferred pair reduces this to a one-dimensional margin ODE with a closed-form solution. These solutions reveal that DPO margins grow without bound (logarithmically as $O(\log t)$ in the gradient-flow limit) while IPO margins converge to a finite $1/(2\tau)$; that label noise induces a qualitative phase transition in DPO (unbounded $\to$ bounded) but affects IPO only quantitatively; and that gradient concentration under asymmetric weighting governs trainability at scale. We confirm all three predictions across five model families spanning four benchmarks. The closed-form solutions also enable principled method design: we pro
    
[^196]: 记账不等于判断：EJBench——基于共轭传热设计活动构建的工程判断基准

    Bookkeeping Is Not Judgement: EJBench, an Engineering Judgement Benchmark Built from a Conjugate Heat Transfer Design Campaign

    [https://arxiv.org/abs/2611.21707](https://arxiv.org/abs/2611.21707)

    本文提出EJBench——一个基于电池冷板共轭传热设计活动构建的与软件无关的工程判断基准，考察模型能否从少量已有结果预测设计在新工况下的行为并做出决策，结果发现最佳前沿模型仅得0.71分，且模型答错时表达的置信度与答对时几乎相同。

    

    我们提出了EJBench，一个工程判断基准：考察模型能否从少量早期结果中预测设计在新工况下的表现，并据此做出决策。与那些测试智能体能否操作仿真或CAD软件的工程基准不同，EJBench与具体软件无关，考察的是工程师在不运行仿真的情况下会得出什么结论。其任务来自电池冷板的共轭传热设计活动，分为三类：将未标注的测试结果与其运行工况进行匹配、在签署设计之前选择应揭示哪些已存储的仿真运行结果、以及按峰值电芯温度对新任务进行排序。每个答案都取自收敛的仿真结果，而非LLM裁判。十个前沿模型中表现最好的GPT-6 Astra得分为0.71，且模型在答错的任务上所表达的置信度与答对的任务几乎相同。我们还使用LoRA和强化学习对Qwen3.5-9B进行了后训练……（摘要原文在此处截断）

    arXiv:2611.21707v1 Announce Type: cross  Abstract: We introduce EJBench, a benchmark of engineering judgement: whether a model can predict how a design behaves at new conditions from a few earlier results, and decide on it. Unlike engineering benchmarks that test whether an agent can operate simulation or CAD software, EJBench is software-agnostic and asks what an engineer would conclude without running a simulation. Its tasks come from a conjugate heat transfer design campaign for battery cold plates, in three families: matching unlabelled test results to their operating points, choosing which stored runs to reveal before signing off a design, and ranking new duties by peak cell temperature. Every answer is taken from a converged simulation, not from an LLM judge. The best of ten frontier models, GPT-6 Astra, scores 0.71, and models state almost the same confidence on tasks they get wrong as on tasks they get right. We also post-train Qwen3.5-9B with LoRA and reinforcement learning, r
    
[^197]: 荷兰两个奶牛场群养犊牛的视频录像与玩耍行为标注数据集

    Video recordings and play behaviour annotations of group-housed dairy calves on two Dutch farms

    [https://arxiv.org/abs/2611.21706](https://arxiv.org/abs/2611.21706)

    该论文发布了一个开放数据集，包含荷兰两个奶牛场群养犊牛的651小时视频录像以及个体层面的玩耍行为标注，为犊牛积极福利研究提供宝贵资源。

    

    玩耍行为被用作奶牛犊牛积极福利的指标，但公开可用的群养犊牛视频及个体行为标注十分稀缺。我们描述了在荷兰两个奶牛场的犊牛群养栏中四次摄像机部署（每次约一周）所拍摄的视频：一个研究教学型奶牛场（2024年和2025年）和一个商业奶牛场（2024年）。录像包含1,173个视频文件，总时长651小时。在八个观察日中，训练有素的观察员使用Noldus The Observer XT软件对犊牛行为进行连续编码，采用包含玩耍及其他行为的18项行为谱，涵盖104小时的视频，其中90小时对栏内每头犊牛均进行了编码。本次发布还包含来自七个标注会话的110万解码帧、通过光学字符识别从烧录在每帧中的摄像机时钟读取的逐帧时间戳、每头犊牛的分割轨迹以及逐犊牛帧级玩耍标签。这些数据支持……

    arXiv:2611.21706v1 Announce Type: cross  Abstract: Play is used as an indicator of positive welfare in dairy calves, but openly available video of group-housed calves with individual behaviour annotations is scarce. We describe video from four camera deployments of about one week each in group pens of dairy calves on two farms in the Netherlands: a research and teaching dairy farm (2024 and 2025) and a commercial dairy farm (2024). The recordings comprise 1,173 video files spanning 651 h. For eight observation days, trained observers coded calf behaviour continuously in Noldus The Observer XT with an 18-item ethogram of play and other behaviours, covering 104 h of video, 90 h of which have every calf in the pen coded. The release also contains 1.1 million decoded frames from seven annotated sessions, a per-frame timestamp read by optical character recognition from the camera clock burned into each frame, per-calf segmentation tracks and per-calf frame-level play labels. The data suppor
    
[^198]: 消解语言模型中道德回路与道德特征的歧义：道德内容、规范性判断还是线索词？来自14个模型的证据

    Disambiguating Moral Circuits and Moral Features in Language Models: Moral content, normative judgment or cue words? Evidence from 14 models

    [https://arxiv.org/abs/2611.21705](https://arxiv.org/abs/2611.21705)

    通过227对最小短语对和14个开源模型的实验，研究发现改变语言模型道德答案的注意力头并非道德专属，而是属于一个对所有类型规范性判断共用的通用评估回路，其效应与“道德上”这一线索词无关。

    

    可解释性方法能够定位那些改变语言模型对道德提示所给出答案的注意力头和稀疏自编码器（SAE）特征，例如“为垂死的孩子偷药在道德上是”这句话会续接“正当的”还是“错误的”。这样的组件可能响应的是道德内容、以任何规范对行为的判断，或是提示中的“道德上”一词。我们构建了227对经五名标注者验证的短语最小对，其两个版本分别在道德标准、另一种规范（审慎、礼仪、审美或证据）或描述性属性上存在差异，并将它们呈现给14个开源权重模型（Llama、Gemma和Qwen，参数量1B至14B，包括基础版和指令微调版），提示词中分别包含或不包含“道德上”一词。在每一个模型中，改变道德答案的注意力头都属于模型在所有被测试的规范性判断类型中所使用的同一个评估回路：这些注意力头承担了相同比例的效应……

    arXiv:2611.21705v1 Announce Type: cross  Abstract: Interpretability methods can locate the attention heads and sparse autoencoder (SAE) features that change the answer a language model gives to a moral prompt, e.g., whether "Stealing medicine for a dying child is morally" continues with "justified" or "wrong". Such a component could respond to moral content, to the judgment of conduct by any norm, or to the word "morally" in the prompt. We built 227 minimal pairs of short phrases, validated by five annotators, whose two versions differ in a moral standard, in another kind of norm (prudence, etiquette, aesthetics or evidence) or in a descriptive quality, and showed them to 14 open-weight models (Llama, Gemma and Qwen, 1B to 14B, base and instruction-tuned), with and without the word "morally". In every model, the heads that change the moral answer belong to an evaluation circuit the models use for every kind of normative judgment we tested: these heads carry the same share of the effect
    
[^199]: 语言模型训练中学习率的混合量子选择：终止准则与交互引导的调度预言机

    Hybrid Quantum Selection of Learning Rates in Language-Model Training: Termination Criteria and an Interaction-Guided Schedule Oracle

    [https://arxiv.org/abs/2611.21701](https://arxiv.org/abs/2611.21701)

    本文提出一种经典-量子混合训练框架，结合Dürr-Høyer最小值查找与BBHT搜索来选择层组学习率，并创新性地引入无需预计算分数表的交互引导调度预言机，通过叠加态评分与振幅放大高效寻找语言模型微调的最佳联合学习率配置。

    

    学习率的选择对语言模型微调效果影响很大，但跨层组的最佳联合学习率配置通常是未知的。本文研究在经典-量子混合训练框架中，利用量子最小值查找算法进行层组学习率选择。该方法将Dürr-Høyer最小值查找与BBHT搜索相结合，并使用基于固定训练检查点处测得的配置分数构建的查找预言机，比较了固定预算和基于置信度两种停止策略。论文随后提出了一种交互引导的调度预言机，该预言机无需预先计算的分数表，即可在叠加态中对联合调度进行评分和标记。经典的一步损失测量提供了层组之间的主效应、两两交互效应和三阶交互效应。量子电路为每个调度累加这些效应，将所得分数与当前基线分数进行比较，并标记出有改进的调度以供振幅放大使用。测量得到的候选配置经过验证……

    arXiv:2611.21701v1 Announce Type: new  Abstract: Learning-rate choices strongly affect language-model fine-tuning, but the best joint profile across layer groups is generally unknown. This paper studies quantum minimum finding for layer-group learning-rate selection within a hybrid classical-quantum training framework. It combines Durr-Hoyer minimum finding with BBHT search and compares fixed-budget and confidence-based stopping policies using a lookup oracle built from measured profile scores at fixed training checkpoints. The paper then introduces an Interaction-Guided Schedule Oracle that scores and marks joint schedules in superposition without a precomputed score table. Classical one-step loss measurements supply the main, pairwise and three-way interaction effects between layer groups. The circuit adds the effects for each schedule, compares the resulting score with the current baseline score, and marks improving schedules for amplitude amplification. Measured candidates are chec
    
[^200]: FACE：面向毫米波MIMO系统的流引导角度信道估计框架

    FACE: Flow-Guided Angular Channel Estimation Framework for mmWave MIMO Systems

    [https://arxiv.org/abs/2611.21698](https://arxiv.org/abs/2611.21698)

    提出了基于流匹配的流引导角度信道估计框架FACE，利用毫米波信道的角度域稀疏性作为生成式信道先验，其估计精度在不同配置下均优于传统估计器和基于扩散的生成式估计器，且在导频较少时仍保持高精度。

    

    获取精确的信道状态信息（CSI）对于毫米波（mmWave）多输入多输出（MIMO）系统至关重要。然而，由于高维信道、有限的导频观测以及测量噪声，这仍然充满挑战。为解决这些问题，我们提出了流引导角度信道估计（FACE），这是一种基于流匹配的新型信道估计框架。流匹配是一种新兴的生成式建模范式，能够学习概率分布之间的连续传输动力学。FACE利用毫米波信道固有的角度域稀疏性，并将学习到的流匹配模型作为生成式信道先验。数值结果表明，在多种信道和系统配置下，FACE的信道估计精度始终高于传统估计器以及基于扩散模型的生成式估计器。此外，FACE在导频观测较少的情况下仍能保持较高的估计精度。

    arXiv:2611.21698v1 Announce Type: new  Abstract: Acquiring accurate channel state information (CSI) is essential for millimeter-wave (mmWave) multi-input multi-output (MIMO) systems. However, it remains challenging due to high-dimensional channels, limited pilot observations, and measurement noise. To address these challenges, we propose flow-guided angular channel estimation (FACE), a novel channel estimation framework based on flow matching, an emerging generative modeling paradigm that learns continuous transport dynamics between probability distributions. FACE exploits the inherent angular-domain sparsity of mmWave channels and incorporates the learned flow matching model as a generative channel prior. Numerical results show that FACE consistently achieves higher channel estimation accuracy than both conventional estimators and diffusion-based generative estimator across diverse channel and system configurations. Moreover, FACE maintains high estimation accuracy with fewer pilot ob
    
[^201]: 视觉网络中的逐层学习率缩放：有效学习率的识别、幂均值聚合与预算依赖性

    Layerwise Learning-Rate Scaling in Vision Networks: Effective-Rate Identification, Power-Mean Aggregation, and Budget Dependence

    [https://arxiv.org/abs/2611.21696](https://arxiv.org/abs/2611.21696)

    该论文发现从零训练视觉网络时，逐层学习率策略的效果可以由一个直接从乘数配置计算出的有效学习率来概括，短预算下的学习率效应主要源于优化进展，且对基础学习率的大小高度敏感。

    

    逐层学习率缩放为网络的不同部分分配不同的学习率。尽管这一方法在微调预训练模型中已被确立，但其在从零开始训练的视觉网络中的效果通常需要通过训练每一个候选方案来评估。我们证明，在这种设置下，逐层策略中与学习率相关的部分可以很好地由一个有效学习率来概括，该有效学习率可以直接从其乘数配置中计算得出。在匹配对照的比较中，测试准确率的变化与训练损失的变化紧密相关（r = -0.996），这为短预算下学习率效应的“优化进展”解释提供了支持。将未调优的基础学习率提高十倍，会使学习率匹配的对比结果在所有六种测试架构中上移（p = 0.016），并将一种已发表的判别性调度方案（未纳入模型拟合）优于统一对照的架构数量从六种中的两种增加到五种中的五种（p…

    arXiv:2611.21696v1 Announce Type: new  Abstract: Layerwise learning-rate scaling assigns different learning rates to different parts of a network. Although established in fine-tuning pretrained models, its effects in vision networks trained from scratch are commonly assessed by training each candidate recipe. We show that, in this setting, the rate-related component of a layerwise strategy is well summarized by an effective learning rate computable directly from its multiplier profile. Across matched-control comparisons, changes in test accuracy closely track changes in training loss (r = -0.996), which supports an optimization-progress explanation of the short-budget rate effect. Increasing the untuned base learning rate tenfold shifts the rate-matched contrast upward in all six tested architectures (p = 0.016) and increases the number of architectures in which a published discriminative schedule, excluded from model fitting, outperforms the uniform control from two to five of six (p 
    
[^202]: 早停从简：面向电力系统故障与攻击诊断的证据获取

    Stop Early, Stay Simple: Evidence Acquisition for Power System Fault and Attack Diagnosis

    [https://arxiv.org/abs/2611.21695](https://arxiv.org/abs/2611.21695)

    该论文提出ActiGrid主动证据获取基准框架，通过实验发现学习型DQN证据选择策略相比简单的熵停止等基线并无显著优势甚至被其帕累托支配，表明电力系统故障与攻击诊断中简单方法往往已足够。

    

    面向信息物理电力系统的诊断模型通常对每个事件都检查所有遥测数据源，尽管有些事件从单一数据源即可清晰判断，而另一些则需要多个数据源。我们探讨的问题是：学习型的证据选择策略何时才值得其带来的额外复杂性。为回答这一问题，我们提出了 ActiGrid，一个主动证据获取的基准测试框架，它将一个经训练能够容忍模态缺失的分类器与双重深度Q网络相结合——该网络要么请求另一个数据源，要么停止并给出诊断，但从不操作电网执行器。在公开的TU Delft数字化变电站和MSU/ORNL电力系统攻击基准上，我们将该DQN与固定子集和固定顺序、校准熵停止准则、贪心信息价值规则以及按子集训练的树集成模型进行了比较。学习到的策略确实能够根据其观察到的内容进行自适应调整，但在TU Delft数据集上，它与熵停止方法在统计上不可区分，并且被更简单的……（摘要原文在此处截断）

    arXiv:2611.21695v1 Announce Type: cross  Abstract: Diagnostic models for cyber-physical power systems usually inspect every telemetry source for every event, although some events are clear from one source and others need several. We ask when a learned policy for choosing that evidence is worth its added complexity. To answer this, we present ActiGrid, a benchmark framework for active evidence acquisition that pairs a classifier trained to tolerate missing modalities with a Double Deep Q-Network (DQN), which either requests another source or stops and diagnoses but never operates grid actuators. On the public TU Delft digital substation and MSU/ORNL power system attack benchmarks, we compare the DQN with fixed subsets and orders, calibrated entropy stopping, a greedy value of information rule, and tree ensembles trained per subset. The learned policy does adapt to what it observes, yet it is statistically indistinguishable from entropy stopping on TU Delft and Pareto dominated by simple
    
[^203]: TIBECG：用于心电图波形发布中导联级人口统计隐私保护的时序信息瓶颈

    TIBECG: Temporal Information Bottleneck for Lead-Specific Demographic Privacy in ECG Waveform Release

    [https://arxiv.org/abs/2611.21694](https://arxiv.org/abs/2611.21694)

    TIBECG通过时序信息瓶颈机制，在保留心电图诊断效用（宏F1不低于95%）的同时，显著抑制了波形发布中年龄、性别等人口统计信息的泄露，隐私增益较最佳传统变换方法中位数提升22.08个百分点。

    

    传输心电图（ECG）波形可能会在预期诊断任务之外暴露人口统计信息。我们在从医院十二导联记录中提取的单个导联上评估了TIBECG——一种架构层面的时序信息瓶颈方法。该处理器包含2,302,337个参数，可在保持相同九项分数一致性目标的同时改变其潜在表示长度，且在保护器的拟合过程中不使用年龄或性别标签。在6,386条记录、十折交叉验证、12个导联和21种长度的实验中，回溯性选择保留了每个导联至少95%的干净宏F1分数。其在较弱的年龄/性别区间增益方面，在所有导联上均超过了最佳符合条件的传统变换方法，中位数优势达22.08个百分点。实施的采样率比较表明，直接降采样通常能更强地抑制人口统计推断，但诊断效用损失更大，而TIBECG则更倾向于保留诊断效用。选择稳定性同样存在差异……

    arXiv:2611.21694v1 Announce Type: cross  Abstract: Transmitting an electrocardiogram (ECG) waveform can expose demographic information beyond an intended diagnostic task. We evaluate TIBECG, an architectural temporal information bottleneck, on individual leads extracted from hospital twelve-lead recordings. A 2,302,337-parameter processor varies its latent length while retaining the same nine-score consistency objective, without age or sex labels in protector fitting. Across 6,386 recordings, ten folds, 12 leads and 21 lengths, retrospective selections retained at least 95% of each lead's clean macro-F1. Their weaker age/sex interval gain exceeded the best eligible conventional transformation on all leads, by a median 22.08 percentage points. The implemented sampling-rate comparison shows that direct downsampling usually suppresses demographic inference more strongly but at a larger loss of diagnostic utility, whereas TIBECG favors utility preservation. Selection stability also varies 
    
[^204]: AdaFlowLite：面向异步移动数据的可扩展且无阻塞的推理

    AdaFlowLite: Scalable and Non-blocking Inference on Asynchronous Mobile Data

    [https://arxiv.org/abs/2611.21691](https://arxiv.org/abs/2611.21691)

    该论文提出AdaFlowLite，通过机会主义推理机制实现异步移动多模态数据部分到达即可执行的可扩展无阻塞推理，并率先提出在开放世界移动环境中以计算方式控制模态亲和性的结构化方法。

    

    配备激光雷达（LiDAR）和摄像头等多种传感器的移动设备的兴起，推动了多模态深度智能在分布式感知任务中的应用，例如智能座舱和驾驶辅助。然而，由于模态数据大小和网络动态性的影响，移动传感数据的到达时间各不相同，这可能导致延迟（如果等待较慢的数据）或精度下降（如果不等待就直接进行推理）。此外，移动系统的多样性和动态特性进一步加剧了这一挑战。为此，我们提出向“机会主义”推理转变，针对异步分布式多模态数据，实现部分数据到达即可进行推理。虽然现有方法侧重于优化模态一致性和互补性（即模态亲和性，modal affinity），但它们缺乏一种“计算性”方法来在开放世界的移动环境中控制这种亲和性。AdaFlowLite 率先提出了结构化的……（原文摘要在此处不完整）

    arXiv:2611.21691v1 Announce Type: cross  Abstract: The rise of mobile devices equipped with numerous sensors, such as LiDAR and cameras, has driven the adoption of multi-modal deep intelligence for distributed sensing tasks, such as smart cabins and driving assistance. However, the arrival time of mobile sensory data vary due to modality size and network dynamics, which can lead to delays (if waiting for slow data) or accuracy decline (if inference proceeds without waiting). Moreover, the diversity and dynamic nature of mobile systems exacerbate this challenge. In response, we present a shift to \textit{opportunistic} inference for asynchronous distributed multi-modal data, enabling inference as soon as partial data arrives. While existing methods focus on optimizing modality consistency and complementarity, known as modal affinity, they lack a \textit{computational} approach to control this affinity in open-world mobile environments. % \sysname pioneers the formulation of structured c
    
[^205]: DeltaHarness：面向资源自适应工具执行的自演化执行框架

    DeltaHarness: Self-Evolving Harnesses for Resource-Adaptive Tool Execution

    [https://arxiv.org/abs/2611.21688](https://arxiv.org/abs/2611.21688)

    DeltaHarness是一种执行引导的策略搜索方法，通过QHI循环利用冷启动实测数据自演化智能体执行框架，使工具执行延迟在视频编辑等任务上最高降低30%。

    

    使用工具的智能体依赖执行框架来组织和调度工具操作。相同的任务与资源描述可能隐藏着会颠倒有效执行计划延迟排序的运行时条件，而解决这种不确定性需要在目标环境中进行第一手执行。我们提出DeltaHarness，一种执行引导的策略搜索方法，可从冷启动测量中自适应调整智能体执行框架。其QHI循环利用观察到的响应来更新运行时调度、修订物理计划，并为后续提案选择实验，同时由独立的确认面板控制策略替换。在固定调度下，修订后的实体化策略在一个视频编辑任务的三对新配置上将完整交付延迟降低了30.0%。在保留任务上，演化出的策略在一个种子中相对于最佳固定计划将延迟降低8.2%，在另一个种子中相对于初始提案降低4.6%。

    arXiv:2611.21688v1 Announce Type: new  Abstract: Tool-using agents rely on execution harnesses to organize and schedule tool operations. The same task and resource description can conceal runtime conditions that reverse the latency ordering of valid execution plans. Resolving this uncertainty requires firsthand execution in the target setting. We introduce DeltaHarness, an execution-guided policy-search method that adapts agent harnesses from cold-start measurements. Its QHI loop uses the observed response to update runtime dispatch, revise physical plans, and select experiments for later proposals. An independent confirmation panel controls policy replacement. With fixed dispatch, a revised materialization policy reduces complete-delivery latency by 30.0% on one video-editing task across three fresh pairs. On held-out tasks, the evolved policies reduce latency by 8.2% relative to the best fixed plan in one seed and by 4.6% relative to the initial proposal in another.
    
[^206]: 面向建筑一体化光伏功率预测的边缘自适应Hiera-Seg模型

    Edge-Adaptive Hiera-Seg Model for Building-Integrated Photovoltaic Power Forecasting

    [https://arxiv.org/abs/2611.21680](https://arxiv.org/abs/2611.21680)

    本文提出边缘自适应HieraSeg模型，通过多朝向视觉-辐照度融合，并在缺失帧、延迟变化和局部遮挡条件下自适应调整特征权重，实现短期建筑一体化光伏功率预测。

    

    建筑一体化光伏功率预测具有挑战性，因为立面朝向、屋顶几何形状、附近建筑物、植被、云层移动、组件温度、逆变器状态和边缘设备可靠性共同影响发电量。本文提出了一种边缘自适应HieraSeg模型，用于结合多朝向视觉-辐照度融合的短期BIPV功率预测。RGB立面图像、可选热成像线索、辐照度测量值、组件温度、立面朝向、屋顶掩膜、遮挡标签、天空状况、历史功率和边缘设备状态被对齐到同步预测窗口中。HieraSeg从立面、屋顶、天空、阴影和背景图像中提取分层区域，而边缘自适应模块在缺失帧、延迟变化和局部遮挡变化的情况下动态调整特征权重。分割分支识别被遮挡的组件区域，并分离立面、屋顶、天空和背景（原文摘要在此处截断）。

    arXiv:2611.21680v1 Announce Type: cross  Abstract: Building integrated photovoltaic power forecasting is challenging because facade orientation, rooftop geometry, nearby buildings, vegetation, cloud movement, module temperature, inverter status and edge device reliability jointly affect power generation. This paper proposes an edge-adaptive HieraSeg model for short term BIPV power forecasting with multi-orientation visual-irradiance fusion. RGB facade images, optional thermal cues, irradiance measurements, module temperature, facade orientation, rooftop masks, shading labels, sky condition, historical power and edge device state are aligned into synchronous forecasting windows. HieraSeg extracts hierarchical regions from facade, rooftop, sky, shadow and background images, while the edge-adaptive module changes feature weights under missing frames, latency shifts and local shading changes. The segmentation branch identifies occluded module areas and separates facade, rooftop, sky and ba
    
[^207]: RTLReviewBench：LLM硬件审查者很少漏掉Bug，但经常拒绝正确的RTL

    RTLReviewBench: LLM Hardware Reviewers Rarely Miss Bugs but Routinely Reject Correct RTL

    [https://arxiv.org/abs/2611.21678](https://arxiv.org/abs/2611.21678)

    该论文提出RTLReviewBench基准，包含114对经仿真验证的有缺陷与正确Verilog模块配对，揭示了LLM审查RTL代码时的不对称现象——很少放过真实bug却频繁错误拒绝正确代码，且合成变异体无法完全暴露这一问题。

    

    硬件团队越来越多地要求大型语言模型（LLM）来审查寄存器传输级（RTL）代码，然而目前尚不清楚模型是否能够在不进行仿真的情况下区分正确的模块与有缺陷的模块。我们推出了RTLReviewBench，这是一个包含114对匹配的有缺陷与正确Verilog模块的基准测试。每一对模块共享同一份规范，大多数有缺陷的模块是LLM代码生成器的真实失败案例，且每个标签均通过仿真得到确认。在对12个商业和开源权重模型进行评估后，我们发现了一个持续存在的不对称性：审查者很少接受有缺陷的代码，但经常拒绝正确的代码，每漏掉一个bug就会对应超过五个错误的拒绝。合成的变异体掩盖了部分问题：在相同的设计上，模型漏掉的注入bug远少于真实bug。要求模型在做出判断之前先进行推理，能显著减少较弱审查者的错误拒绝，并且一项受控消融实验表明，推理出现的位置……（原文在此处截断）

    arXiv:2611.21678v1 Announce Type: cross  Abstract: Hardware teams increasingly ask large language models (LLMs) to review register-transfer level (RTL) code, yet it is unclear whether a model can tell a correct module from a buggy one without simulating it. We introduce RTLReviewBench, a benchmark of 114 matched pairs of buggy and correct Verilog modules. Each pair shares one specification, most buggy modules are real failures of LLM code generators, and every label is confirmed by simulation. Evaluating 12 commercial and open-weight models, we find a consistent asymmetry: reviewers rarely accept buggy code but often reject correct code, producing more than five false rejections for every missed bug. Synthetic mutants hide part of the problem: on the same designs, models miss far fewer injected bugs than real ones. Asking a model to reason before it commits to a verdict sharply reduces false rejections for weaker reviewers, and a controlled ablation shows that where the reasoning appea
    
[^208]: 使用增强孤立森林改进具有科学价值射电源的早期检测

    Improving Early Detection of Scientifically Interesting Radio Sources Using Enhanced Isolation Forest

    [https://arxiv.org/abs/2611.21674](https://arxiv.org/abs/2611.21674)

    本研究提出并评估了四种增强孤立森林变体，结合BYOL自监督学习提取的特征与MGCLS巡天数据，有效提升了具有科学价值射电源的早期检测与排序效果。

    

    射电天文巡天的快速发展，使得从大量观测天体中识别具有科学价值的源成为一项重大挑战。本研究探讨了增强孤立森林方法能否改善具有科学价值射电源的早期检测与排序。研究基于MeerKAT星系团遗产巡天（MGCLS）中的1,031个延展射电源，评估了四种孤立森林变体：标准增强版、池化增益版、基于峰度的版本以及超体积版本。每个源由通过BYOL自监督学习与主成分分析（PCA）导出的52维特征集来表示，并以专家的科学价值评级作为真实标签。这些方法采用平均精度（AP）、受试者工作特征曲线下面积（ROC AUC）、Precision@10、Precision@50以及累积

    arXiv:2611.21674v1 Announce Type: cross  Abstract: The rapid growth of radio astronomical surveys has created a major challenge in identifying scientifically interesting sources from large populations of observed objects. This study investigates whether enhanced Isolation Forest (iForest) methods can improve the early detection and ranking of scientifically interesting radio sources. Four iForest variants, Standard Enhanced, Pooled Gain, Kurtosis-based, and Hypervolume, are evaluated using 1,031 extended radio sources from the MeerKAT Galaxy Cluster Legacy Survey (MGCLS). Each source is represented by a 52-dimensional feature set derived using Bootstrap Your Own Latent (BYOL) self-supervised learning and principal component analysis (PCA), while expert scientific-interest ratings are used to define the ground-truth labels. The methods are evaluated using Average Precision (AP), receiver operating characteristic area under the curve (ROC AUC), Precision@10, Precision@50, and cumulative 
    
[^209]: 令牌提交作为引导掩码扩散生成的控制界面

    Token Commitment as a Control Surface for Steering Masked Diffusion Generation

    [https://arxiv.org/abs/2611.21669](https://arxiv.org/abs/2611.21669)

    该论文提出将令牌提交策略（即每步提交哪些令牌和提交多少令牌）作为掩码扩散模型推理引导的新型控制界面，通过反事实目标感知提交和校准自适应预算两种机制，在语言和视觉-语言任务上实现了对生成过程的有效引导。

    

    掩码扩散模型在采样器决定哪些预测成为永久之前，会反复预测临时令牌。现有的推理时引导方法主要修改激活值、logits或去噪轨迹，而将这种提交策略视为固定的解码细节。我们转而将令牌提交作为一种扩散模型特有的控制界面进行研究，并将其分解为两个决策：每一步提交哪些令牌以及提交多少令牌。针对第一个决策，我们引入了反事实目标感知提交机制，通过模型置信度与留一令牌目标贡献的混合来对未解决的位置进行排序。针对第二个决策，我们引入了由目标进度和置信度驱动的校准自适应预算。在LLaDA家族的语言和视觉-语言设置中，跨多个目标、两种激活引导方法以及无引导对照条件，两种机制均在经验上产生单调的目标质量提升。

    arXiv:2611.21669v1 Announce Type: cross  Abstract: Masked diffusion models repeatedly predict provisional tokens before a sampler decides which predictions become permanent. Existing inference-time steering methods primarily modify activations, logits, or denoising trajectories, while treating this commitment policy as a fixed decoding detail. We instead study token commitment as a diffusion-specific control surface and separate it into two decisions: which tokens to commit and how many tokens to commit at each step. For the first, we introduce counterfactual target-aware commitment, which ranks unresolved positions by a mixture of model confidence and leave-one-token-out target contribution. For the second, we introduce calibrated adaptive budgets driven by target progress and confidence. Across language and vision-language settings in the LLaDA family, multiple targets, two activation-steering methods, and an unsteered control, both mechanisms yield empirically monotonic target-quali
    
[^210]: Limite 1B - Violetto：技术报告

    Limite 1B - Violetto: Technical Report

    [https://arxiv.org/abs/2611.21646](https://arxiv.org/abs/2611.21646)

    Limite 1B - Violetto 是一个仅 10 亿参数的数学推理模型，通过在精心策划的数学语料上从头训练，并结合知识蒸馏、监督微调与 RLVR 后训练，在 AIME 2026 上取得 94.01% 的成绩，性能媲美参数量数倍于它的模型。

    

    我们推出了 Limite 1B - Violetto，这是一个专为高吞吐量数学推理设计的十亿参数模型。Limite 在一个精心策划、以数学为核心的语料库上从头训练，共使用了 2370 亿个 token，并支持最长 131k token 的上下文长度。其架构与训练方案是协同开发的，旨在最大化推理能力的同时保持训练和推理成本较低。后训练阶段结合了知识蒸馏、针对性的监督微调以及在精选课程上的 RLVR（基于可验证奖励的强化学习）。Limite 在 AIME 2026 上取得了 94.01% 的成绩，在 BeyondAIME 上取得了 74.25% 的成绩，在我们的评测套件中跻身于参数量为其数倍的模型之列。该模型刻意只进行了轻度的指令微调：它主要被定位为单轮数学求解器以及大型系统中快速并行运行的组件，而非通用助手。我们发布了该模型、其基础模型和价值模型检查点，以及一套经过优化的服务栈。

    arXiv:2611.21646v1 Announce Type: new  Abstract: We present Limite 1B - Violetto, a one-billion-parameter model designed for high-throughput mathematical reasoning. Limite is trained from scratch on 237 billion tokens from a carefully curated, mathematics-centric corpus and supports context lengths of up to 131k tokens. Its architecture and training recipe were developed together to maximize reasoning capability while keeping training and inference costs low. Post-training combines knowledge distillation, targeted supervised fine-tuning, and RLVR on a curated curriculum. Limite achieves 94.01% on AIME 2026 and 74.25% on BeyondAIME, placing it among models many times its size on our evaluation suite. The model is deliberately only lightly instruction-tuned: it is intended primarily as a single-turn mathematical solver and as a fast, parallel component of larger systems, rather than as a general-purpose assistant. We release the model, its base and value checkpoints, and an optimized ser
    
[^211]: VASR：基于等价结构与修复成本反馈的向量感知装配规划

    VASR: Vector-Aware Assembly Planning with Equivalent Structures and Repair-Cost Feedback

    [https://arxiv.org/abs/2611.20984](https://arxiv.org/abs/2611.20984)

    VASR框架通过融合SVG向量编码器与视觉语言模型的特征，并采用带层级等价奖励和修复成本反馈的RHPO策略优化方法，在家具装配层级规划任务中显著超越了现有最先进方法。

    

    家具装配规划需要从单个零件和目标图示中推断出子装配层级结构。现有的基于栅格的方法未能显式利用SVG图纸中的几何和结构信息。我们提出了VASR，这是一个将专用SVG编码器提取的特征与预训练视觉语言模型的视觉表示相融合的框架，用于预测装配层级结构。为避免对参考装配树的有效替代方案进行惩罚，我们提出了修复感知层级策略优化（RHPO），并采用保持层级的等价奖励。RHPO进一步利用修复成本反馈，在等价感知奖励相近的候选方案中，倾向于选择下游修正成本更低的决策。VASR在所有被评估的方法中取得了最高的Equivalent-Hard F1分数，在基于IKEA的测试集上比GPT6-Astra高出17.69个百分点，同时在i（摘要原文在此处截断）

    arXiv:2611.20984v1 Announce Type: new  Abstract: Furniture assembly planning requires inferring subassembly hierarchies from individual parts and a target illustration. Existing raster-based methods do not explicitly exploit the geometric and structural information in SVG drawings. We introduce VASR, a framework that fuses features from a dedicated SVG encoder with visual representations from a pretrained vision-language model to predict assembly hierarchies. To avoid penalizing valid alternatives to a reference assembly tree, we propose Repair-Aware Hierarchical Policy Optimization (RHPO) with hierarchy-preserving equivalence rewards. RHPO further uses repair-cost feedback to favor decisions with lower downstream correction costs among candidates with comparable equivalence-aware rewards. VASR achieves the highest Equivalent-Hard F1 among the evaluated methods, outperforming GPT6-Astra by 17.69 percentage points on the IKEA-derived test set while maintaining strong performance on an i
    
[^212]: 零售预测中世界模型解释的审计：忠实性、准确性与价格响应

    Auditing World-Model Explanations in Retail Forecasting: Faithfulness, Accuracy and Price Response

    [https://arxiv.org/abs/2611.20811](https://arxiv.org/abs/2611.20811)

    研究发现，零售预测中的世界模型解释即使通过了八项审计检查（如构造性、稳定性、价格方向和与真实销量的一致性），其促销提升估计仍可能高出真实值1.2至1.4倍，价格响应被夸大近5至6.5倍，表明通过常规检查并不保证解释的忠实性与准确性。

    

    能够读取价格与促销计划的预测模型，将每个预测解释为两部分之和：一个基线（即不执行任何计划时的销量），加上该计划带来的提升。我们通过八项检查来检验此类解释，其中六项在任何结果出现之前就已固定了通过规则，结果发现通过这些检查并不意味着解释是正确的。在效应已知的人工生成门店数据中，一个世界模型通过了构造性、稳定性、价格方向以及与销量和LightGBM一致性等各项检查，但当商家将促销计划安排在高需求周时，其提升值是真实值的1.33至1.40倍；在该世界的八种变体中，这一比例为1.23至1.42倍。它仍然能选对促销的周次，但根据其提升值订购的库存与真实提升值相比偏差达0.44至0.48。它所宣称的对长期常规价格变动的响应是真实值-0.5的4.7至6.5倍，而只有通过实际测量才能发现这一点。在Dominick's数据集上，38个门店常规价格偏离其z（摘要在此处截断）

    arXiv:2611.20811v1 Announce Type: cross  Abstract: Forecasting models that read the price and promotion plan explain each forecast as a baseline, what would sell with nothing planned, plus the lift the plan adds. We test such explanations with eight checks, six with a pass rule fixed before any result, and find that passing them does not make an explanation right. In generated store data with known effects, a world model passes the checks of construction, stability, price direction and agreement with sales and with LightGBM, yet its lifts are 1.33 to 1.40 times the truth when merchants plan into high-demand weeks, and 1.23 to 1.42 times in eight variants of that world. It still picks the right weeks to promote, but stock ordered to its lift is off by 0.44 to 0.48 of the true lift. Its stated response to a lasting regular-price change is 4.7 to 6.5 times the true -0.5, and only a measured cause finds this. On Dominick's, 38 episodes in which a store's regular prices sat apart from its z
    
[^213]: 免重放循环状态迁移的信息极限

    Information Limits of Replay-Free Recurrent State Migration

    [https://arxiv.org/abs/2611.19345](https://arxiv.org/abs/2611.19345)

    本文刻画了循环模型在不重放历史数据的条件下进行状态迁移所必须保留的最小信息量，证明了带标量输入的可控d状态模型至多需要kd个额外实数坐标，且单个秩一替换方向即可要求全部kd个。

    

    当一个循环（递归）模型被替换时，其旧状态可能无法决定新模型基于相同历史输入会计算出什么。本文刻画了在替换方案未知之前必须保留的信息。对于时不变的仿射线性模型族，阶数为k的局部迁移所需的最小连续因果记忆，等于提升系数系统的可达维数减去锚定状态所含的信息。对于带标量输入的可控d状态模型，即使任意的矩阵替换方向仅在流式处理之后才被揭示，也至多需要kd个额外的实数坐标；而单个秩一方向就可能已经要求全部kd个坐标。条件奇异值与辅助比特界将该局部信息定律与固定容差下的恢复区分开来。作者还给出了不变子空间与未来输出相关的构造，并通过受控实验比较了由此产生的各种状态表示。在一个小型语言模型中……（摘要在此处截断）

    arXiv:2611.19345v1 Announce Type: new  Abstract: When a recurrent model is replaced, its old state may not determine what the new model would compute from the same history. We characterize the information that must be retained before the replacement is known. For a time-invariant affine linear family, the minimum continuous causal memory for order-$k$ local migration equals the reachable dimension of a lifted coefficient system minus the information in the anchor state. A controllable $d$-state model with scalar input needs at most $kd$ additional real coordinates, even when arbitrary matrix replacement directions are revealed only after streaming. A single rank-one direction can already require all $kd$. Conditional singular values and auxiliary-bit bounds distinguish this local information law from recovery at a fixed tolerance. We also give invariant-subspace and future-output constructions. Controlled experiments compare the resulting representations. In a small language model, equ
    
[^214]: SLICES：面向稀疏ICU时间序列自监督学习的基准与评估框架

    SLICES: A Benchmark and Evaluation Framework for Self-Supervised Learning on Sparse ICU Time Series

    [https://arxiv.org/abs/2611.18868](https://arxiv.org/abs/2611.18868)

    该论文提出SLICES基准框架，通过统一固定数据处理流程、编码器架构和评估协议，首次实现了对稀疏ICU时间序列自监督学习方法的公平比较，并发现掩码自编码（MAE）在绝大多数评估设置中优于JEPA和对比学习。

    

    当数据处理、编码器和适配方式同时变化时，重症监护病房（ICU）时间序列的自监督学习（SSL）结果很难相互比较。我们提出了SLICES，一个在统一临床设置下用于比较和扩展SSL方法的基准与评估框架。该框架固定了24小时数值与观测掩码输入、Transformer编码器、患者级别的数据划分、四个临床终点、两种适配协议以及五个匹配的随机种子，覆盖MIMIC-IV、eICU-CRD及其合并数据集。框架提供了掩码自编码（MAE）、联合嵌入预测（JEPA）和对比学习三种参考实现。在这些训练目标中，MAE在全部九个冻结分类比较中（包括收敛探针条件下）均取得最高的平均AUPRC，并在九个微调比较中的七个中领先。微调后的MAE在全标签情况下在院内死亡率预测上表现最佳，并且在每一个标签比例下的得分均高于同架构的监督式Transformer，而且SS...

    arXiv:2611.18868v1 Announce Type: new  Abstract: Self-supervised learning (SSL) results for intensive care unit (ICU) time series are hard to compare when data processing, encoders, and adaptation vary together. We present SLICES, a benchmark and evaluation framework for comparing and extending SSL methods under a shared clinical setup. It fixes 24-hour value and observation-mask inputs, a Transformer encoder, patient-level splits, four endpoints, two adaptation protocols, and five matched seeds across MIMIC-IV, eICU-CRD, and their pooled union. Masked autoencoding (MAE), joint-embedding prediction (JEPA), and contrastive learning provide reference implementations. Among these objectives, MAE leads mean AUPRC in all nine frozen classification comparisons, also under converged probes, and in seven of nine fine-tuning comparisons. Fine-tuned MAE leads hospital mortality at full labels and scores higher than a supervised Transformer of the same architecture at every label fraction, and SS
    
[^215]: Softmax交叉熵下锥形分类头中的渐近间隔与特征约束

    Asymptotic Margins and Feature Confinement in Conic Classification Heads under Softmax Cross-Entropy

    [https://arxiv.org/abs/2611.18507](https://arxiv.org/abs/2611.18507)

    论文为锥形分类头引入了可由二阶锥规划计算的渐近间隔，并证明该间隔的正负决定了softmax交叉熵训练下特征是否有界：间隔为正时损失仅在特征发散处趋近于零，间隔为负时损失强制且最优解位于一个显式球内。

    

    基于锥函数构建的分类头为每个类别赋予特征空间中的一个有界区域，这些区域可用于拒绝未见类别的样本。我们提出的问题是：当该分类头使用softmax交叉熵训练时，这些区域是否也能使特征保持有界。对于类别得分由线性项与欧几里得范数相结合的分类头，我们首先证明每个类别的区域由一个旋转二次曲面所界定，其离心率是某个学习到的比率的倒数。随后，我们为每个类别定义一个渐近间隔，该间隔比较其自身得分增长与所有其他得分增长的情况，并可通过二阶锥规划进行计算。当且仅当一个类别的间隔为正时，其损失的下确界为零，且损失仅在特征发散时才趋近于零。当间隔为负时，损失是强制的，其极小值点位于一个显式给定的球内。由于softmax仅依赖于差异……（原文摘要至此截断）

    arXiv:2611.18507v1 Announce Type: new  Abstract: Classification heads built on conic functions give each class a bounded region of the feature space, and such regions are used to reject samples of unseen classes. We ask whether these regions also keep the features bounded when the head is trained with softmax cross-entropy. For a head whose class scores combine a linear term with a Euclidean norm, we first show that the region of each class is bounded by a quadric of revolution, whose eccentricity is the reciprocal of one learned ratio. We then attach to each class an asymptotic margin, which compares the growth of its score with the growth of all the other scores and can be computed by a second-order cone program. The loss of a class has infimum zero exactly when its margin is positive, and it then approaches zero only along diverging features. When the margin is negative, the loss is coercive and its minimizers lie in an explicit ball. Since softmax depends only on the differences be
    
[^216]: 基于得分函数的神经常微分方程求解反应扩散方程

    Score-based neural ordinary differential equations for reaction-diffusions

    [https://arxiv.org/abs/2611.18467](https://arxiv.org/abs/2611.18467)

    该论文提出了一种带物理权重的拉格朗日式得分驱动神经常微分方程方法，让粒子通过权重追踪质量变化，并以质量归一化的自洽残差作为训练目标，从而有效求解反应扩散方程。

    

    扩散方程可以被改写为由演化密度的得分函数所驱动的输运方程。传统的概率流表述利用这种由得分驱动的速度确定性地移动粒子，通常保持总质量守恒。在本工作中，我们针对反应扩散方程开发了一种拉格朗日方法，其中每个粒子同时携带一个位置和一个物理权重。特征常微分方程将源项纳入对数密度及其梯度的演化之中，而粒子权重则跟踪质量的变化。我们将这种表示与学习得到的速度场以及加权粒子训练相结合，以近似这些动力学。对于非递增的反应速率，一个相对熵不等式支持将质量归一化的自洽残差作为训练目标。当增长率有界时，该目标能够控制所诱导密度的误差。该算法将时间…（摘要原文在此处截断）

    arXiv:2611.18467v1 Announce Type: cross  Abstract: A diffusion equation can be rewritten as a transport equation driven by the score function of the evolving density. Conventional probability-flow formulations, which move particles deterministically with this score-driven velocity, typically conserve total mass. In this work, we develop a Lagrangian method for reaction-diffusion equations in which each particle carries a position and a physical weight. The characteristic ODEs incorporate source terms into the evolution of the log-density and its gradient, while particle weights track changes in mass. We combine this representation with a learned velocity field and weighted particle training to approximate these dynamics. For non-increasing reaction rates, a relative-entropy inequality supports the mass-normalized self-consistency residual as the training objective. This objective can control the error of the induced density when the growth rate is bounded. The algorithm divides the tim
    
[^217]: 神经分类器作为纠错码

    Neural Classifiers as Error-Correcting Codes

    [https://arxiv.org/abs/2611.18239](https://arxiv.org/abs/2611.18239)

    该论文提出统一框架，将监督深度分类刻画为香农意义下的纠错码译码，证明交叉熵优化等价于最大似然译码，深度网络在特征空间中隐式学习码本，并据此推导出无损量化、知识蒸馏和结构正则化的新准则。

    

    本文提出了一个统一框架，将监督深度分类刻画为香农意义下的纠错码译码过程。研究表明，交叉熵优化使softmax分类器与最大似然译码相一致，其中logits映射为到各类别特定码字的汉明距离。这种对应关系解释了标准散度目标（如Kullback-Leibler散度、Jeffreys散度）的出现，并为决策边界的稳定性提供了几何基础。此外，Neural Collapse与LeJEPA的几何结构被识别为各向同性噪声下译码的最优星座图。与经典的纠错输出码不同，研究表明深度网络能够在特征空间中隐式地学习有效的码本。基于这一见解，本文推导出用于无损量化、改进知识蒸馏以及基于码冗余的结构正则化的新准则。最后，这一……（摘要被截断）

    arXiv:2611.18239v1 Announce Type: new  Abstract: A unified framework is proposed that characterizes supervised deep classification as the decoding of error-correcting codes in the Shannon sense. Cross-entropy optimization is shown to align softmax classifiers with maximum-likelihood decoding, where logits map to Hamming distances to class-specific codewords. This correspondence explains the emergence of standard divergence objectives (e.g., Kullback-Leibler, Jeffreys) and provides a geometric basis for the stability of decision boundaries. Furthermore, the geometries of Neural Collapse and LeJEPA are identified as optimal constellations for decoding under isotropic noise. Unlike classical Error-Correcting Output Codes, deep networks are shown to learn effective codebooks in the feature space implicitly. This insight is leveraged to derive novel criteria for lossless quantization, improved knowledge distillation, and architectural regularization based on code redundancy. Finally, this p
    
[^218]: 可迁移的解释：用稀疏自编码器解读医学信息检索

    Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders

    [https://arxiv.org/abs/2611.18179](https://arxiv.org/abs/2611.18179)

    本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。

    

    回答用户查询的医学AI系统依赖于从大型文档集合中检索医学知识。在此背景下，在医学数据集上微调通用模型可以提升检索相关性。然而，现代检索系统所依赖的稠密嵌入在很大程度上仍是不透明的，难以理解其检索结果背后的机制。此外，针对特定医学数据集进行适配可能会失去对新临床语料库的可迁移性。在本工作中，我们通过将稀疏自编码器（SAEs）应用于医学信息检索来解决这些问题，将稠密文本嵌入转化为语义上可解释的表示。为此，我们的模型将重建目标与InfoNCE对比损失相结合，后者能够直接保留查询与文档之间的相关性。我们在六个医学检索数据集上（涵盖域内与域外场景）评估了三种嵌入模型。在医学语料库上训练后，我们的稀疏……

    arXiv:2611.18179v1 Announce Type: cross  Abstract: Medical AI systems that answer users' queries rely on retrieving medical knowledge from large document collections. In this context, finetuning general models on medical datasets can improve retrieval relevance. Yet the dense embeddings underlying modern retrieval systems remain largely opaque, making it difficult to understand the mechanics behind their results. Moreover, adapting to a specific medical dataset risks losing transferability to new clinical corpora. In this work, we tackle these issues by applying Sparse Autoencoders (SAEs) to medical information retrieval, transforming dense text embeddings into semantically interpretable representations. To do so, our model combines a reconstruction objective with an InfoNCE contrastive loss that directly preserves query-document relevance. We evaluate three embedding models across six medical retrieval datasets, both in-domain and out-of-domain. Trained on a medical corpus, our sparse
    
[^219]: EgoX：基于具身梦境的第一人称视角跨具身操作

    EgoX: Egocentric Cross-Embodiment Manipulation with Embodiment Dreaming

    [https://arxiv.org/abs/2611.16728](https://arxiv.org/abs/2611.16728)

    EgoX提出了一种第一人称视角的跨具身学习框架，通过为每个观测和动作模态配置专用分词器与专家模块、共享Transformer主干，并借助仅在训练阶段使用的“具身梦境”未来状态预测目标，使通用操作策略能够有效利用人类、真实机器人和仿真智能体的异构演示数据。

    

    通用操作策略可以从人类、真实机器人和仿真智能体的互补经验中受益。然而，三者之间在形态结构、感知方式和控制接口上的差异，使得从它们组合的演示数据中进行有效学习变得困难。我们提出了EgoX，一个第一人称视角的跨具身学习框架，通过语义上相互对应的观测模态和动作模态来组织异构演示数据。EgoX在这一混合数据上预训练一个带具身梦境的模块化跨具身Transformer（MXT+），并在目标具身的演示数据上进行微调。在每个具身体内，MXT+为每个观测模态分配一个专用的分词器，为每个动作模态分配一个专用的专家，而Transformer主干则在所有具身之间共享。一个仅在训练阶段使用的具身梦境目标函数用于预测未来视觉状态和本体感觉状态的潜在表示，从而鼓励共享主干……（原文摘要在此处截断）

    arXiv:2611.16728v1 Announce Type: cross  Abstract: Generalist manipulation policies can benefit from the complementary experience of humans, real robots, and simulated agents. However, differences in morphology, sensing, and control interfaces make it difficult to learn effectively from their combined demonstrations. We introduce EgoX, an egocentric cross-embodiment learning framework that organizes heterogeneous demonstrations through semantically corresponding observation and action modalities. EgoX pretrains a Modularized Cross-Embodiment Transformer with Embodiment Dreaming (MXT+) on this mixture and fine-tunes it on target-embodiment demonstrations. Within each embodiment, MXT+ assigns a dedicated tokenizer to each observation modality and a dedicated expert to each action modality, with a Transformer trunk shared across embodiments. A training-only embodiment-dreaming objective predicts latent representations of future visual and proprioceptive states, encouraging the shared trun
    
[^220]: VLA 能从调用它的智能体中学到什么？面向具身导航的任务级经验蒸馏

    What Can a VLA Learn from the Agent That Calls It? Task-Level Experience Distillation for Embodied Navigation

    [https://arxiv.org/abs/2611.16246](https://arxiv.org/abs/2611.16246)

    H2E 提出会话级蒸馏方法，将 LLM 规划器与冻结 VLA 作为异构训练智能体，仅保留成功到达目标的会话并屏蔽偏离目标的步骤来训练 VLA，使其在测试时无需 LLM 即可克服走过目标、提前停止和偏离后难以恢复三种典型失败。

    

    视觉-语言-动作（VLA）模型直接根据相机视角驱动机器人，但单独使用 VLA 会表现出三种典型失败：走过目标点、在到达目标前提前停下，以及偏离轨迹后难以恢复。将 VLA 与大型语言模型（LLM）规划器配对可以修复这些失败，但在每个回合中都需要持续查询 LLM，从而增加了 token 成本和延迟。我们提出 Harness-to-Executor（H2E），一种会话级蒸馏方法，它将 LLM 规划器与冻结的 VLA 组合为异构训练智能体，仅保留成功到达目标的会话，然后仅在这些会话上单独训练 VLA，因此在测试时无需使用 LLM。在这些会话中，规划器给出子指令并用其自身的工具移动机器人，绕路是常见现象，且由规划器决定何时停止。因此，无论移动由谁执行，H2E 都将每一次移动标注为原始任务的 VLA 动作。该方法对远离目标的步骤进行损失屏蔽，并为……（摘要截断）

    arXiv:2611.16246v2 Announce Type: replace-cross  Abstract: Vision-language-action (VLA) models drive the robot directly from camera views, yet a VLA alone shows three typical failures. It walks past the goal, stops before reaching it, and rarely recovers after going off track. Pairing a VLA with a large language model (LLM) planner fixes these failures, but queries the LLM throughout every episode, adding token costs and latency. We introduce Harness-to-Executor (H2E), a session-level distillation method that pairs an LLM planner and a frozen VLA as a heterogeneous training agent, keeps only sessions that reach the goal, and then trains a VLA alone on them so no LLM is used at test time. In these sessions, the planner gives sub-instructions and moves the robot with its own tools, detours are common, and the planner decides when to stop. H2E therefore labels every motion as a VLA action for the original task, whoever made it. It masks the loss on steps moving away from the goal and assi
    
[^221]: AI4Plasma II：面向可变电极间距GEC等离子体的几何感知代理建模

    AI4Plasma II: Geometry-Aware Surrogate Modeling for Variable Electrode Gap GEC Plasmas

    [https://arxiv.org/abs/2611.15584](https://arxiv.org/abs/2611.15584)

    该研究比较了六种几何感知代理模型在可变电极间距GEC等离子体仿真中的数据需求，以实现对完整等离子体场和晶圆附近离子密度分布的快速准确预测。

    

    工业等离子体工艺的开发需要对运行参数和几何参数进行广泛探索。几何参数，尤其是电极间距，会影响放电结构和工艺均匀性。通过反复进行高保真度仿真来探索这些参数空间可能带来高昂的计算成本。代理模型虽然能够提供快速预测，但几何变化使其应用变得复杂，因为不同样本之间的计算域和离散化方式可能各不相同。在本工作中，比较了六种几何感知代理模型——分别基于预设和学习的几何映射、扩展域表示、连续坐标以及几何原生离散化——在准确预测完整等离子体场以及晶圆附近离子密度分布方面所需的数据量。仿真数据集涵盖气压、电压、频率和电极间距，每个参考算例需要……（原文摘要在此处截断，内容不完整）

    arXiv:2611.15584v2 Announce Type: replace  Abstract: Industrial plasma process development requires broad exploration of operating and geometric parameters. Geometric parameters, particularly electrode spacing, affect discharge structure and processing uniformity. Exploring these parameter spaces through repeated high-fidelity simulations can be computationally expensive. Surrogate models can offer rapid predictions, but geometric variation complicates their application because the computational domain and discretization may change across samples.   In this work, six geometry-aware surrogate models based on prescribed and learned geometry mappings, extended-domain representation, continuous coordinates, and geometry-native discretizations are compared in terms of their data requirements for accurate predictions of full plasma fields and ion density profiles near the wafer. The simulation dataset spans pressure, voltage, frequency, and electrode gap, with each reference case requiring a
    
[^222]: ReShap：面向高维多动作环境的可解释强化学习

    ReShap: Interpretable Reinforcement Learning for High-Dimensional Multi-Action Settings

    [https://arxiv.org/abs/2611.15507](https://arxiv.org/abs/2611.15507)

    ReShap 是一个策略蒸馏框架，通过从原始图像中提取特征、执行 SHAP 特征归因，并将深度强化学习策略蒸馏为在完整状态-动作对上训练的可解释代理模型，从而将基于 Shapley 值的可解释强化学习扩展到了高维、多动作的环境。

    

    arXiv:2611.15507v1 公告类型：交叉。摘要：深度强化学习（RL）在应对复杂任务方面发挥着关键作用，但其对深度神经网络的依赖使得学习到的策略难以解释。为了增强强化学习策略的可信度，现有工作（例如 SILVER 框架 [Li et al., 2025]）聚焦于基于 Shapley 值的可解释性，但通常局限于低维状态空间、二元动作设置以及单一类别的线性代理模型。我们提出了 ReShap，这是一个策略蒸馏框架，将基于 Shapley 值的可解释性扩展到高维、多动作的环境。ReShap 从原始图像观测中提取紧凑的特征表示，进行基于 SHAP 的特征归因以刻画每个特征对策略决策的贡献，并将深度强化学习策略蒸馏为一个可解释的代理模型，该代理模型基于策略访问分布上的完整（状态，策略动作）对集合进行训练。我们实例化了（摘要在此处被截断）

    arXiv:2611.15507v1 Announce Type: cross  Abstract: Deep Reinforcement Learning (RL) plays a crucial role in tackling complex tasks, yet its reliance on deep neural networks makes the learned policies opaque. To enhance the trustworthiness of RL policies, existing work (e.g., the SILVER framework [Li et al., 2025]) has focused on Shapley-based interpretability, but is typically restricted to low-dimensional state spaces, binary-action settings, and a single class of linear surrogate. We propose ReShap, a policy-distillation framework that extends Shapley-based interpretability to high-dimensional, multi-action environments. ReShap extracts compact feature representations from raw image observations, performs SHAP-based feature attribution to characterize how each feature contributes to the policy's decisions, and distills the deep RL policy into an interpretable surrogate trained on the full set of (state, policy-action) pairs along the policy's visited distribution. We instantiate the 
    
[^223]: AI4Plasma I：电容耦合等离子体场的数据高效代理建模

    AI4Plasma I: Data-Efficient Surrogate Modeling of Capacitively Coupled Plasma Fields

    [https://arxiv.org/abs/2611.15002](https://arxiv.org/abs/2611.15002)

    该研究系统比较了五种神经代理架构在电容耦合等离子体场预测中的数据效率，发现傅里叶神经算子（FNO）在绝大多数测试场景中以最少的训练模拟数量即可达到1%误差精度，是数据效率最优的架构选择。

    

    参考电容耦合等离子体（CCP）模拟能够提供空间分辨的等离子体场，但在不同工作条件下生成这些数据的成本十分高昂。我们比较了五种代理模型架构——坐标MLP（Coordinate MLP）、PCA-Net、U-Net、傅里叶神经算子（FNO）和Transolver——在预测一维平面和二维轴对称GEC CCP基准测试中射频周期平均电子密度、离子密度、电势和电子温度方面的数据效率。我们评估了在参数内部和参数边界测试用例中达到1%和2%平均相对L2误差所需的参考模拟数量，并考察了各架构特有的作用机制。在八项比较中的七项里，FNO实现了最低或并列最低的参考案例需求。在一维情形下，FNO仅需50个训练案例即可达到1%的内部误差，而Transolver需要100个，U-Net需要200个。在二维情形下，FNO和U-Net均用100个案例达到1%的内部误差；FNO和Tra（注：原摘要在此处被截断）

    arXiv:2611.15002v2 Announce Type: replace  Abstract: Reference capacitively coupled plasma (CCP) simulations provide spatially resolved plasma fields but are costly to generate across operating conditions. We compare the data efficiency of five surrogate architectures--Coordinate MLP, PCA-Net, U-Net, Fourier neural operator (FNO), and Transolver--for predicting RF-cycle-averaged electron density, ion density, electric potential, and electron temperature in 1D planar and 2D axisymmetric GEC CCP benchmarks. We assess the number of reference simulations needed to achieve 1% and 2% mean relative L2 error for parameter-interior and parameter-boundary test cases, and examine architecture-specific mechanisms. FNO achieves the lowest or joint-lowest reference-case requirement in seven of eight comparisons. In 1D, FNO reaches 1% interior error with 50 training cases, compared with 100 for Transolver and 200 for U-Net. In 2D, FNO and U-Net each reach 1% interior error with 100 cases; FNO and Tra
    
[^224]: 重新思考“RL泛化，SFT记忆”：SFT数据的作用

    Rethinking "RL Generalizes, SFT Memorizes": The Role of SFT Data

    [https://arxiv.org/abs/2611.14912](https://arxiv.org/abs/2611.14912)

    SFT泛化不佳并非模型固有缺陷而是数据问题——精心混合数据来源并匹配RL的正确采样数量或总采样预算，可使SFT的泛化能力追平甚至超过RL。

    

    使用可验证奖励的强化学习（RL）训练的大语言模型展现出强大的推理能力和广泛的泛化能力，而通过监督微调（SFT）训练的模型通常被认为更容易产生记忆化且迁移能力有限。本文在广泛使用的Qwen2.5和Qwen3系列模型上，从SFT训练数据的视角重新审视了这一区别。首先，我们研究了数据来源的作用，并证明其至关重要：精心混合的SFT数据集显著优于仅由更大模型生成的数据。其次，我们研究了数据规模的作用，结果表明使SFT与RL的正确rollout数量相匹配可大幅提升SFT的泛化能力，而匹配总rollout预算则能使SFT的泛化能力与RL相当。将这两个因素相结合，可进一步使SFT的泛化能力甚至超越RL。第三，通过使用LLM标注来刻画解法方法……（摘要不完整）

    arXiv:2611.14912v1 Announce Type: cross  Abstract: Large language models trained with Reinforcement Learning (RL) with verifiable rewards exhibit strong reasoning ability and broad generalization, whereas models trained with Supervised Fine-Tuning (SFT) are often viewed as more prone to memorization and limited transfer. This paper rethinks this distinction through the lens of SFT training data on widely used Qwen2.5 and Qwen3 series models. First, we study the role of data source and show that it is critical: a carefully mixed SFT dataset substantially outperforms data generated solely by a larger model. Second, we study the role of data scale and show that matching the number of correct rollouts between SFT and RL greatly improves SFT generalization, while matching the total rollout budget enables SFT to generalize as well as RL. Combining these two factors further enables SFT to generalize even better than RL. Third, by using LLM annotations to characterize the solution methods in t
    
[^225]: 具有结构缓存加速的双核张量轮分解方法用于张量恢复

    Dual-Core Tensor Wheel Decomposition with Structure-Cached Acceleration for Tensor Recovery

    [https://arxiv.org/abs/2611.13938](https://arxiv.org/abs/2611.13938)

    该论文提出双核张量轮分解（DCTW）及结构缓存加速策略，通过将高阶中心核分解为两个由共享潜变量键连接的低阶核，并缓存张量网络收缩中的中间张量以避免显式构造大规模矩阵和冗余计算，从而大幅提升了张量恢复的计算与内存效率。

    

    张量轮分解通过外环因子与高阶中心核之间的相互作用来捕获高维数据中的复杂关联。然而，更新这些组件需要显式构造和操作大规模矩阵，这限制了计算效率，并对中心核的更新带来了尤其高昂的计算和内存开销。为解决这些局限，我们提出了双核张量轮分解（DCTW），将中心核分解为两个由共享潜变量键连接的低阶核，且每个核分别与不同的外环因子组相耦合。我们进一步提出了一种结构缓存加速策略，该策略避免了显式的大规模矩阵乘法，并将张量网络收缩过程中生成的中间张量缓存下来，以便在各块更新之间重复利用。这种重用减少了冗余计算，并避免了大规模矩阵的显式构造，从而显著提升了计算与内存效率。

    arXiv:2611.13938v1 Announce Type: new  Abstract: Tensor wheel decomposition captures complex correlations in high-dimensional data through interactions between outer-ring factors and a high-order central core. However, updating these components requires the explicit construction and manipulation of large-scale matrices, which limits computational efficiency and incurs particularly high computational and memory costs for the central-core update. To address these limitations, we propose dual-core tensor wheel decomposition (DCTW), which factorizes the central core into two lower-order cores linked by a shared latent bond. Each core is coupled to a different group of outer-ring factors. We further propose a structure-cached acceleration strategy that avoids explicit large-scale matrix multi- plications and caches intermediate tensors generated during tensor network contractions for reuse across block updates. This reuse reduces redundant computation and avoids the explicit construction of
    
[^226]: 绝对零度下的注意力机制

    Attention at Absolute Zero

    [https://arxiv.org/abs/2611.11937](https://arxiv.org/abs/2611.11937)

    该论文提出一种在零温度（硬最大值）下读取见证者的高阶配对注意力层，通过仅传递计数比较比特并在每轮循环重建状态，克服了软注意力在规模扩大时的权重分散、分数差距与状态漂移等失效模式，在比训练规模大1000倍的最大值检索任务上仍保持99.95%以上的准确率。

    

    注意力机制对答案的候选支持（我们称之为“见证者”）进行评分，并在某个温度下读取这些分数。我们构建了一个高阶注意力层，其状态存在于物品对之上：对于一对，每个第三方物品 k 都是一个见证者，通过其两条边 和 进行评判。该层在零温度下读取见证者（保留最佳者及其对应的计数），只传递比较这些计数的比特，在每次循环中从输入和这些比特重建其配对状态，并且仅依靠答案进行训练。每一项设计选择都消除了软注意力在规模扩大时的一种失效模式：权重分散、学习到的分数之间无法区分的差距，以及漂移的软状态。在最大值检索任务中，该模型在比训练集大1,000倍的集合上，于全部十个随机种子下均达到至少99.95%的准确率；而使用软计数训练的同一网络在4,096个项目时性能降至42%。在无提示的CLRS-30基准上，该层的一般形式（其一半通道在零温度下读取……）

    arXiv:2611.11937v2 Announce Type: replace-cross  Abstract: Attention scores candidate supports of an answer, which we call witnesses, and reads the scores at a temperature. We build a higher-order attention layer whose state lives on pairs of items: for a pair (a, b), each third item k is a witness judged through its legs (a, k) and (k, b). It reads witnesses at zero temperature, keeping the best and the number tied with it, passes on only bits comparing such counts, rebuilds its pair state from the input and these bits at every loop, and is trained from answers alone. Each choice removes one way soft attention fails at size: dispersed weight, unresolved gaps between learned scores, and a drifting soft state. In maximum retrieval it is right on at least 99.95% of sets 1,000 times larger than in training, on all ten seeds; trained with a soft count, the same network falls to 42% at 4,096 items. On CLRS-30 without hints, a general form of the layer (half its channels read at zero tempera
    
[^227]: 正则化放大潜意识学习

    Regularization Amplifies Subliminal Learning

    [https://arxiv.org/abs/2611.10397](https://arxiv.org/abs/2611.10397)

    本文发现正则化非但不能抑制反而会放大隐蔽影响——全量微调下通常不存在的潜意识学习，在低秩适应、显式惩罚或增加数据等正则化手段下反而会被显著诱发和增强。

    

    近期的研究表明，隐藏的偏见可以通过与这些偏见没有明显语义关系的数据进行传递。正则化能否限制这种隐蔽的影响？通过约束模型的适应过程，人们可能预期正则化会抑制模型在看似无关输入上的行为变化。我们在潜意识学习以及相关的隐蔽影响场景（包括幻影传递和logit线性选择）中系统地检验了这一假设。结果却发现了相反的情况：正则化会放大隐藏偏见的传递。特别地，在全量微调下通常观察不到潜意识学习，但低秩适应或显式惩罚却可以诱导出强烈的传递。幻影传递和logit线性选择也呈现出类似的趋势。增加数据量作为一种隐式正则化形式，同样会加强这种隐蔽影响。总之，这些发现表明，正则化所诱导的简单性偏见可能会促进这种隐蔽影响（原文摘要在此处截断）。

    arXiv:2611.10397v2 Announce Type: replace  Abstract: Recent work has shown that hidden biases can be transferred through data with no apparent semantic relationship to those biases. Can regularization limit this covert influence? By constraining adaptation, regularization might be expected to suppress behavioral changes on seemingly unrelated inputs. We systematically study this hypothesis in subliminal learning, as well as in related settings of covert influence, including phantom transfer and logit-linear selection. We find the opposite: regularization amplifies the transfer of hidden biases. In particular, subliminal learning is typically absent under full fine-tuning, but low-rank adaptation or explicit penalties can induce strong transfer. Phantom transfer and logit-linear selection show similar trends. Adding more data, an implicit form of regularization, also strengthens covert influence. Together, these findings suggest that the simplicity biases induced by regularization may f
    
[^228]: LEGIT：基于标识符字典树的实时资格门控，用于异构页面的受限解码

    LEGIT: Live Eligibility Gating over Identifier Tries for Constrained Decoding of Heterogeneous Pages

    [https://arxiv.org/abs/2611.10357](https://arxiv.org/abs/2611.10357)

    提出 LEGIT 约束机制，通过对标识符字典树叶节点进行深度优先编号、把随请求和 token 动态变化的资格规则转化为高效的范围查找，使生成式推荐系统在整页解码时能精确且快速地保证每个语义 ID 真实、合规且不重复。

    

    生成式推荐器用一段简短的 token 序列（即语义 ID，Semantic ID）来命名每个候选，并通过生成这些 token 来完成推荐。一些生成式推荐器，包括我们正在构建的 Uber Eats 搜索引擎，会在一次解码过程中写出整个结果页面，其中包含商店及其菜单项。这样的页面上，每个语义 ID 都必须指向一个真实的、且符合本次搜索资格的商店或条目（例如，能够配送到用户所在位置的商店），并且尚未出现在该页面上；此外，每个条目还必须属于其上方的商店。在商品目录上构建字典树可以保证标识符的真实性，但资格性与重复性约束会随着每次请求和每个 token 动态变化，而我们尝试过的语法约束解码器无法既精确又快速地执行这些不断变化的规则。我们将这一约束命名为 LEGIT，即“基于标识符字典树的实时资格门控”。对字典树叶节点进行深度优先编号，可以将每条规则转化为对一棵商店字典树和一棵条目字典树的范围查找。

    arXiv:2611.10357v2 Announce Type: replace-cross  Abstract: A generative recommender names each candidate with a short token sequence, a Semantic ID, and recommends by generating those tokens. Some generative recommenders, including the Uber Eats search engine we are building, write a whole results page, stores with their menu items, in one decoding pass. Every Semantic ID on such a page must name a real store or item that is eligible for the search, such as a store that delivers to the user's location, and that is not already on the page. Each item must also belong to the store above it. A trie over the catalog keeps identifiers real, but eligibility and repetition change with every request and token, and the grammar-constrained decoder we tried could not enforce these changing rules exactly and quickly. We call this constraint LEGIT, for live eligibility gating over identifier tries. Numbering trie leaves depth first turns each rule into a range lookup over one store trie and one item
    
[^229]: 规划器中的古德哈特定律：基于采样的控制会“钻空子”利用从人类偏好中学到的奖励模型

    Goodhart in the Planner: Sampling-Based Control Exploits Rewards Learned from Human Preferences

    [https://arxiv.org/abs/2611.09670](https://arxiv.org/abs/2611.09670)

    论文揭示并量化了“规划器中的古德哈特定律”：针对人类偏好奖励模型进行更强的 MPPI 搜索虽在标签准确率上保持高分，却使真实回报最多下降 76%，并进一步给出真实回报与搜索偏离距离（√KL）的关系以及 χ² 正则化 MPPI 在由评分器误差范围设定温度时的最坏情况最优性保证。

    

    机器人越来越多地通过针对从人类偏好中学习到的奖励模型进行搜索来完成规划；我们测量了这种搜索对真实目标造成的影响。在带有众包标签的 HalfCheetah 和 Hopper 任务上，以及在带有合成标签和两位作者盲测标签的 Meta-World 抽屉打开与盘子滑动任务上，更强的 MPPI 搜索使真实回报最多下降 76%，而留出标签上的准确率仍保持在 0.75–0.97：使用人类标签时，20 次运行中抽屉打开次数为 0 次，而作为规划器起始点的克隆策略则在 10 次中全部成功。该效应在使用学习到的动力学模型以及图像评分器时依然存在。真实回报大致跟随规划器权重偏离其提案样本的距离（$\sqrt{\mathrm{KL}}$），而使用在合成标签上训练的验证器进行 best-of-$N$ 动作选择，其表现甚至会低于单个带噪声的动作。我们证明了当温度由评分器在提案样本上的误差范围设定时，$\chi^2$ 正则化的 MPPI 具有最坏情况最优性。

    arXiv:2611.09670v2 Announce Type: replace-cross  Abstract: Robots increasingly plan by searching against a reward model learned from human preferences; we measure what this search does to the true objective. On HalfCheetah and Hopper with crowd-sourced labels, and on Meta-World drawer-open and plate-slide with synthetic labels and blind labels from the two authors, harder MPPI search cuts true return by up to 76% while held-out label accuracy stays at 0.75-0.97: with human labels the drawer opens in 0 of 20 runs, against 10 of 10 for the cloned policy the planner starts from. The effect persists with a learned dynamics model and with an image scorer. True return roughly follows how far the planner's weights move from its proposals ($\sqrt{\mathrm{KL}}$), and best-of-$N$ action selection with a verifier trained on synthetic labels falls below a single noisy action. We prove that $\chi^2$-regularized MPPI at a temperature set by the scorer's error spread on the proposals is worst-case op
    
[^230]: NovoCRF：基于前缀质量格的全局归一化从头肽段测序

    NovoCRF: Globally Normalized De Novo Peptide Sequencing on a Prefix-Mass Lattice

    [https://arxiv.org/abs/2611.05260](https://arxiv.org/abs/2611.05260)

    NovoCRF首次提出在所有质量可行肽段上以精确全局归一化方式训练的条件随机场模型，通过前缀质量格约束路径实现从头肽段测序，模型仅360万参数、训练约27 GPU小时，并可在推理时针对每次实验自适应微调。

    

    从头肽段测序已经从基于手工设计或生成式评分的动态规划方法，发展到端到端学习该任务的大型Transformer模型。我们提出了NovoCRF，一种条件随机场（CRF），其状态为前缀质量，使其路径恰好对应前体质量±0.06 Da范围内的所有肽段。一个小型网络根据源自物理原理的离子匹配特征为每条边评分，并在所有质量可行的肽段上以全局归一化的评分进行训练。据我们所知，NovoCRF是首个在所有质量可行肽段上进行精确归一化训练的从头测序方法。我们的模型仅有360万个参数，训练约需27 GPU小时。由于其模型规模小，NovoCRF还可以在推理时针对每次实验运行进行自适应：它在实验自身的高置信度预测结果上微调部分权重，重新校准该实验的前体质量和碎片质量，并利用预训练的保留时间索引模型预测的保留时间信息。

    arXiv:2611.05260v2 Announce Type: replace  Abstract: De novo peptide sequencing has moved from dynamic programming over hand-designed or generative scores to large transformers that learn the task end to end. We present NovoCRF, a conditional random field (CRF) whose states are prefix masses, so that its paths are exactly the peptides within $\pm$0.06 Da of the precursor mass. A small network scores each edge from physics-derived ion-match features and is trained with its scores globally normalized over all mass-feasible peptides. To our knowledge, NovoCRF is the first de novo method trained with exact normalization over all mass-feasible peptides. Our model has 3.6M parameters and trains in about 27 GPU-hours. Because it is small, NovoCRF can also adapt to each run at inference time: it fine-tunes part of its weights on the run's own confident answers, recalibrates the run's precursor and fragment masses, and uses retention time, predicted with a pretrained indexed-retention-time mode
    
[^231]: 通过分层脑电图-文本对齐实现思维破译

    Thought-deciphering via hierarchical EEG-to-text alignment

    [https://arxiv.org/abs/2611.03623](https://arxiv.org/abs/2611.03623)

    提出一种分层脑电图-文本对齐框架，先解码粗粒度语义类别、再在类别内检索细粒度句子，显著提升了从EEG进行句子级解码的检索准确率。

    

    非侵入式脑到文本接口为有沟通障碍的人群提供了替代性沟通途径。然而，从脑电图（EEG）进行句子级别的解码仍然具有挑战性。我们提出了一种分层脑电图-文本对齐框架，通过逐步解码粗粒度语义信息和细粒度句子表示来检索句子。我们假设，相比单个句子，更广泛的语义背景能够从脑电图中被更可靠地捕获。因此，脑电图表示首先与语义类别进行对齐，随后在预测出的类别内进行句子检索。在超过2,000个独特文本样本中，我们的方法实现了平均top-5和top-10检索准确率分别为62.6% ± 1.0%和83.5% ± 1.0%，而直接进行脑电图-句子对齐的准确率仅为50.3% ± 2.3%和68.3% ± 2.0%。定性分析进一步揭示了……（摘要内容截断）

    arXiv:2611.03623v2 Announce Type: replace  Abstract: Non-invasive brain-to-text interfaces offer alternative communication pathways for people with communication disabilities. However, sentence-level decoding from electroencephalography (EEG) remains challenging. We propose a hierarchical EEG-text alignment framework that retrieves sentences by progressively decoding coarse-grained semantic information and fine-grained sentence representations. We hypothesize that broader semantic context can be more reliably captured from EEG than individual sentences. Accordingly, EEG representations are first aligned with semantic classes, followed by sentence retrieval within the predicted class. Among over 2,000 unique text samples, our approach achieves average top-5 and top-10 retrieval accuracies of 62.6\% $\pm$ 1.0\% and 83.5\% $\pm$ 1.0\%, respectively, compared with 50.3\% $\pm$ 2.3\% and 68.3\% $\pm$ 2.0\% for direct EEG-sentence alignment. Qualitative analysis further reveals that decoding
    
[^232]: 在训练前预测RLHF如何响应奖励模型的修改

    Forecasting How RLHF Responds to Reward-Model Edits Before Training

    [https://arxiv.org/abs/2611.03244](https://arxiv.org/abs/2611.03244)

    该论文基于KL正则化强化学习的一阶理论提出了一种协方差预测方法，能够在RLHF训练开始之前就预测出奖励模型修改将如何改变策略的行为，从而免去昂贵的试错式RL训练。

    

    带有标题、列表和更长篇幅的回复会从奖励模型（RM）中获得高分，RLHF策略会因此学会产生这些特征，即使真实质量并没有提高。因此从业者会对奖励模型进行修改，但只有在跑完RL训练后才能知道修改是否有效。我们证明，策略对奖励模型修改的响应可以在训练前进行预测，只需利用当前策略的样本以及新旧奖励模型的打分。如果将奖励模型视为一组“价格”，KL正则化强化学习的一阶理论意味着每个特征的变化与其与奖励变化之间的提示内协方差成正比。我们通过包含自蒸馏安慰剂的2^(4-1)部分因子设计，将受控的价格变化植入重新训练的奖励模型中，从六个父策略出发运行了54次RL训练，并在训练开始前固定所有预测。借助协方差预测器，我们成功预测了全部20个供应量变化的方向。

    arXiv:2611.03244v2 Announce Type: replace  Abstract: Responses with headers, lists and greater length receive high scores from reward models (RMs), and RLHF policies learn to produce these features even when true quality does not improve. Practitioners therefore edit their RMs, but they learn whether an edit works only after an RL run. We show that the response of a policy to an RM edit can be forecast before training, from samples of the current policy and the scores of the old and new RM. If we read the RM as a set of prices, the first-order theory of KL-regularized RL implies that each feature moves in proportion to its within-prompt covariance with the change in reward. We implant controlled price changes into retrained RMs under a 2^(4-1) fractional factorial design with a self-distilled placebo, run 54 RL runs from six parent policies, and fix every forecast before the runs. With the covariance predictor, we predict the direction of all 20 changes in supply whose forecast has the
    
[^233]: 黎曼流形上积分曲线的抽样设计与置信管

    Sampling Design and Confidence Tubes for Integral Curves on Riemannian Manifolds

    [https://arxiv.org/abs/2611.00771](https://arxiv.org/abs/2611.00771)

    本文在紧黎曼流形上提出了估计未知向量场积分曲线的被动与主动两种抽样方法，证明了被动核估计的曲线误差阶为 $n^{-2/(d+3)}$ 并给出由线性随机微分方程刻画的函数型中心极限定理，同时证明基于先导信息的主动抽样可将期望最大曲线误差提升至 $N^{-1/2}$ 阶，从而为构建置信管奠定基础。

    

    我们在已知紧黎曼流形上，利用带噪声的切向量场来估计未知向量场的积分曲线。我们考虑两种抽样设置：一种是在已收集的随机位置上的独立观测，另一种是在固定响应预算下，允许利用先导信息来选择后续观测位置。在适当的正则性假设下，被动核估计量的曲线误差以概率达到 $n^{-2/(d+3)}$ 阶，其中 $n$ 为样本量，$d$ 为流形维数。其函数型中心极限定理的极限由一个线性随机微分方程刻画。对于主动抽样，一个独立的先导样本将新的一批测量引导至目标曲线的邻域。在局部化、光滑性和设计条件下，这种集中性使得期望最大曲线误差达到 $N^{-1/2}$ 阶，其中先导样本计入总响应预算 $N$。两种方法均……（摘要原文在此处截断）

    arXiv:2611.00771v2 Announce Type: replace-cross  Abstract: We estimate the integral curve of an unknown vector field from noisy tangent vector field on a known compact Riemannian manifold. We consider two sampling settings: independent observations at random locations that have already been collected, and a fixed response budget that allows subsequent locations to be chosen using pilot information. Under suitable regularity assumptions, the passive kernel estimator has curve error of order $n^{-2/(d+3)}$ in probability, with sample size $n$ and manifold dimension $d$. Its functional central limit theorem has a limit characterized by a linear stochastic differential equation. For active sampling, an independent pilot directs one new batch of measurements to neighborhoods of the target curve. Under localization, smoothness and design conditions, this concentration yields expected maximum curve error of order $N^{-1/2}$, including the pilot in the total response budget $N$. Both methods p
    
[^234]: DAPS：面向LLM定理证明的依赖感知前提选择

    DAPS: Dependency-Aware Premise Selection for LLM Theorem Proving

    [https://arxiv.org/abs/2610.41536](https://arxiv.org/abs/2610.41536)

    该论文提出DAPS方法，通过首次系统挖掘53个Lean Blueprint项目构建依赖关系语料库，将数学中的依赖结构信号引入前提选择的训练数据与模型中，克服了现有基于语义相似度的选择器在检索偏斜和分布外查询上的固有局限。

    

    基于LLM的Lean 4定理证明发展迅速，但证明器难以利用新贡献的引理，因为检索偏向于流行的基础引理，使得正确的前提被埋没在数十万个候选之中。现有的神经选择器按语义相似度对候选进行排序，存在三个内在局限：频率偏斜的检索、孤立的成对排序，以及在分布外查询（如竞赛题目和非正式语言）上的性能退化。鉴于数学是按依赖关系而非相似性来构建结构的，我们将这一信号重新引入训练数据和模型中。我们整理了两项未被充分利用的依赖资源。我们对53个Lean Blueprint项目进行了首次系统性挖掘，构建了一个此前未被开发的语料库，其中包含3,805个人工整理的非正式+正式节点，捕捉了Mathlib4中缺失的罕见研究级前提。我们进一步将其与提取的……（原文摘要在此截断）

    arXiv:2610.41536v1 Announce Type: cross  Abstract: LLM-based theorem proving in Lean 4 has advanced rapidly, but provers are difficult to leverage newly contributed lemmas, because retrieval skews toward popular foundational lemmas, leaving the right premises buried among hundreds of thousands of candidates. Existing neural selectors rank candidates by semantic similarity, with three intrinsic limitations: frequency-skewed retrieval, isolated pairwise ranking, and degradation on out-of-distribution queries such as competition problems and informal language. Given that mathematics is structured by dependencies rather than similarities, we restore this signal to both the training data and the model. We curate two underused dependency resources. We conduct the first systematic mining of 53 Lean Blueprint projects, a previously untapped corpus of 3,805 manually curated informal+formal nodes that capture rare, research-level premises absent from Mathlib4. We further pair it with the extract
    
[^235]: 通过广义余维数概念形式化目标景观的内在维度

    Formalizing the Intrinsic Dimension of Objective Landscapes by a Generalized Notion of Co-dimension

    [https://arxiv.org/abs/2610.39674](https://arxiv.org/abs/2610.39674)

    本文将目标景观的内在维度形式化为一种依赖于集合全局几何的广义余维数概念，并为浅层和深层ReLU网络学生-教师设置中的最优参数集合推导出内在维度的上下界。

    

    在本工作中，我们形式化了Li等人（2018）提出的目标景观内在维度的概念，该概念通过沿随机生成的子空间进行训练来衡量学习问题的结构复杂度。我们将集合Θ⊂ℝ^p的内在维度定义为最小维度d_int(Θ)，使得ℝ^p中维度为d_int(Θ)的随机平面以概率一与Θ相交。我们证明它是一种依赖于Θ的全局几何的广义余维数概念。我们研究了内在维度的数学性质，并开发了估计学习问题中出现的集合Θ的内在维度的技术。更具体地说，我们推导了当Θ是学生-教师设置中浅层和深层ReLU网络最优参数集合时的上界和/或下界，包括有限和无限……

    arXiv:2610.39674v2 Announce Type: replace-cross  Abstract: In this work, we formalize the concept of intrinsic dimension of objective landscapes introduced in Li et al. (2018), which measures the structural complexity of learning problems by training along randomly generated subspaces. We define the intrinsic dimension of a set $\Theta\subset\mathbb{R}^p$ as the minimal dimension $d_{\mathrm{int}}(\Theta)$ such that a random plane of dimension $d_{\mathrm{int}}(\Theta)$ in $\mathbb{R}^p$ intersects $\Theta$ with probability one. We show that it is a generalized notion of co-dimension that depends on the global geometry of $\Theta$. We study mathematical properties of the intrinsic dimension and develop techniques to estimate the intrinsic dimension of sets $\Theta$ that arise in learning problems. More specifically, we derive upper and/or lower bounds where $\Theta$ is the set of optimal parameters in student-teacher settings with shallow and deep ReLU networks, including finite and in
    
[^236]: 看见尚且不够：基于知识的、针对不确定视觉证据的主动问询

    Seeing Is Not Enough: Knowledge-Grounded Active Inquiry from Uncertain Visual Evidence

    [https://arxiv.org/abs/2610.38774](https://arxiv.org/abs/2610.38774)

    该论文提出一个中医舌诊框架：当舌象与脉象证据不足以区分证候时，系统依据校准后的支持度下降触发主动症状问询，并利用问答动态更新证候候选列表，使整个判断过程可追溯、更可靠。

    

    在中医（TCM）中，相似的舌象可能伴随不同的证候，而决定性的症状并不出现在照片中。我们提出了一个舌诊评估框架，其中图像与脉象的判读结果指导有针对性的症状问询，所获得的答案在最终报告前细化证候候选列表。图像/脉象预测指导从有来源依据的症状问题库中选择问题；已知的答案会更新用于选择下一个问题的评估。当初始预测的支持度经校准后出现下降时，将触发重新评估；随后的报告需要新的答案来支持一个稳定的候选证候。问答记录使这些决策可被查验。我们从不同的真实舌象照片构建了2,600个案例，并配以指定的证候、模拟的症状回答和渲染的脉象；其中2,599个用于五折来源不相交评估，1个仅用于训练。在干净图像上，仅凭图像/脉象预测即可达……（原文摘要在此处截断）

    arXiv:2610.38774v2 Announce Type: replace  Abstract: Similar tongue findings can accompany different syndromes in traditional Chinese medicine (TCM), with decisive symptoms absent from the photograph. We present a framework for tongue assessment in which image and pulse findings guide targeted symptom inquiry and acquired answers refine the syndrome shortlist before reporting. Image/pulse predictions guide selection from a sourced symptom question bank. Known answers update the assessment used to select the next question. A calibrated loss of support for the initial prediction triggers reassessment; a subsequent report requires fresh answers supporting a stable candidate. Question and answer records make these decisions inspectable. We construct 2,600 cases from distinct real tongue photographs, assigned syndromes, simulated symptom responses and rendered pulses; 2,599 enter five-fold source-disjoint evaluation and one is training-only. On clean images, image/pulse prediction alone ach
    
[^237]: Alpha-TCM：面向多模态中医推理的可审计知识集成

    Alpha-TCM: Auditable Knowledge Integration for Multimodal TCM Inference

    [https://arxiv.org/abs/2610.38754](https://arxiv.org/abs/2610.38754)

    Alpha-TCM 提出一个可审计的多模态中医推理工作流，通过分离测量、知识、预测与审查决策，并利用贝叶斯委员会核算量化所注入知识（包括被损坏的错误知识）对预测的影响，实验证明知识背景在中低训练规模下能显著降低预测误差。

    

    知识引导的多模态系统需要一种方法来衡量所提供的知识如何影响预测结果，包括当这些知识出错时的影响。Alpha-TCM 针对中医（TCM）场景解决了这一问题，它通过一个已实现的工作流程，将测量数据、所提供的知识、预测结果和审查者的决策彼此分离。观察数据与背景知识分别独立输入，可替换的先验使得在固定模态编码器的条件下能够进行受控干预。贝叶斯委员会核算方法在插入背景知识之前，先移除每个编码器中已经包含的参考先验。一项完全模拟的验证研究使用了十对配对随机种子和四种训练规模。结果表明：在 600 和 1,200 个训练样本时，知识背景相对于无知识的融合方法降低了预测误差；等信息量的学习型背景在 600 个样本时具有竞争力，而在更大规模时误差更低。受控的知识损坏实验能够识别出过度自信和分散……

    arXiv:2610.38754v2 Announce Type: replace  Abstract: Knowledge-guided multimodal systems need a way to measure how supplied knowledge affects predictions, including when that knowledge is wrong. Alpha-TCM addresses this problem for traditional Chinese medicine (TCM) through an implemented workflow that separates measurements, supplied knowledge, predictions, and reviewer decisions. Observations and contextual knowledge enter separately, and a replaceable prior allows controlled interventions with fixed modality encoders. Bayesian committee accounting removes the reference prior already contained in each encoder before inserting contextual knowledge. A fully simulated confirmation study uses ten paired seeds and four training sizes. Knowledge context reduces prediction error relative to knowledge-free fusion at 600 and 1,200 training cases; equal-information learned context is competitive at 600 and has lower error at larger sizes. Controlled corruption identifies overconfident and disp
    
[^238]: 共享上下文，独立引擎：边缘SoC上的流式视觉-语言推理

    Shared Context, Separate Engines: Streaming Vision--Language Inference on Edge SoCs

    [https://arxiv.org/abs/2610.35309](https://arxiv.org/abs/2610.35309)

    提出了一种在边缘SoC上协调CPU、GPU、NPU等异构加速器并发执行有状态流式VLM推理的系统，通过将流的版本化上下文与各引擎的KV状态分离，实现任意引擎可接续处理数据流，从而支持流式生成、并行序列生成和状态感知调度。

    

    arXiv:2610.35309v2 公告类型：新 摘要：边缘SoC集成了CPU、GPU和NPU，然而视觉-语言模型（VLM）的运行时却将生成任务固定在单个引擎上，让其他引擎处于闲置状态。流式VLM进一步暴露了可利用的并行性：在较早的查询尚未回答完毕时，新帧就已经到达。各引擎在计算能力和内存访问方式上存在差异，并且在共同运行时会相互争抢资源，因此不存在一种在所有SoC和工作负载下都最优的统一阶段部署方案。后端局部的KV状态使得其他引擎无法接续未完成的工作。本文提出的系统协调异构加速器以并发执行有状态的VLM推理。它将流的版本化上下文与各引擎的KV状态分离，使得任何引擎都可以通过重用、导入或重建该状态来接续处理同一个流。有状态流式生成在一个引擎上摄取新帧的同时由另一个引擎进行回答，并发序列生成则并行处理不同的就绪请求。状态感知调度将每个请求发送至预计能最快完成它的引擎……

    arXiv:2610.35309v2 Announce Type: new  Abstract: Edge SoCs pack CPUs, GPUs, and NPUs, yet vision--language model (VLM) runtimes keep generation on one engine and leave the others idle. Streaming VLMs expose further parallelism: new frames arrive while earlier queries are still being answered. Engines differ in compute and memory access and contend when they co-run, so no single placement of stages wins across SoCs and workloads. Backend-local KV state keeps other engines from continuing the work. \name coordinates heterogeneous accelerators to execute stateful VLM inference concurrently. It separates a stream's versioned context from each engine's KV state, so any engine can continue the stream by reusing, importing, or reconstructing that state. Stateful Streaming Generation ingests new frames on one engine while another answers, and Concurrent Sequence Generation serves different ready requests in parallel. State-aware dispatch sends each request to the engine expected to finish it f
    
[^239]: 利用物理正则化流匹配从PM2.5即时预报气溶胶粒径分布

    Nowcasting aerosol particle size distributions from PM2.5 with physics-regularized flow matching

    [https://arxiv.org/abs/2610.33557](https://arxiv.org/abs/2610.33557)

    本文提出利用物理正则化流匹配的生成式深度学习方法，仅从PM2.5质量浓度和气象参数即时生成气溶胶粒径分布的合理集合，并以芬兰森林和城市两个对比站点为测试平台，对六种生成模型进行了系统性基准评测。

    

    小于2.5微米颗粒物的质量浓度（PM2.5）测量成本低且监测范围广，而颗粒物数浓度粒径分布（PSD）则需要昂贵的仪器，且仅在少数研究站点进行测量。然而，许多气溶胶效应，如蒸汽吸收、云滴形成以及在人体呼吸道中的沉积，取决于粒径分布而非质量。我们探索了近期的生成式深度学习模型能否对粒径分布进行即时预报：在24小时窗口内，模型根据PM2.5记录和八个气象参数，生成该窗口内一组合理的小时级粒径分布集合，反映出同一质量浓度可能对应多种不同的粒径分布。我们没有提出单一的最佳模型，而是使用两个特征迥异的芬兰站点——北方森林站点Hyytiälä和城市站点Kumpula——作为受控测试平台，并在该平台上将六种生成模型与确定性和气候学基线进行基准比较。

    arXiv:2610.33557v1 Announce Type: new  Abstract: The mass concentration of particles smaller than 2.5um (PM2.5) is cheap to measure and widely monitored, whereas the particle number size distribution (PSD) requires costly instruments and is measured at only a few research stations. Yet many aerosol effects, such as vapor uptake, cloud droplet formation and deposition in the human respiratory tract, depend on the PSD rather than on mass. We explore whether recent generative deep learning models can nowcast the PSD: from the PM2.5 record and eight meteorological parameters over a 24h window, a model generates an ensemble of plausible hourly size distributions for the same window, reflecting that one mass can arise from many size distributions. Rather than proposing a single best model, we use two contrasting Finnish stations, the boreal forest site Hyyti\"al\"a and the urban site Kumpula, as a controlled testbed. On it, we benchmark six generative models against deterministic and climato
    
[^240]: 并非所有错误标签都同等重要：基于置信序列的高效任务感知审计

    Not All Mislabels Are Equal: Efficient Task-Aware Auditing with Confidence Sequences

    [https://arxiv.org/abs/2610.33262](https://arxiv.org/abs/2610.33262)

    该论文提出ATLAS框架，将任意时刻有效的置信序列推断与任务感知的自适应审计策略结合，根据标签不确定性及其对下游评分的影响优先审计数据点，从而以比随机抽样少11%-24%的审计量将标签错误率认证到±2个百分点的精度。

    

    arXiv:2610.33262v1 公告类型：交叉。摘要：我们研究在固定数据集中对类别标签进行高效、自适应的审计，其动机来自生物多样性信用项目：在这类项目中，专家验证物种记录的成本高昂，且下游评分（如付款金额）对某些标签上错误的敏感度高于其他标签。我们提出的ATLAS框架将任意时刻有效推断与任务感知的自适应审计策略相结合。通过混淆矩阵来表示标签修正，我们根据标签不确定性以及潜在标签修正对下游评分函数的影响，来决定接下来优先检查哪些数据点。对于误标率等线性评分，我们的优先级感知方法在肯尼亚生物多样性数据和广泛使用的CIFAR-10N基准数据集上，分别以比随机抽样少24%和11%的审计量，将整体标签错误率认证到±2个百分点的精度。对于希尔-香农多样性等非线性评分，我们探讨了（原文摘要至此截断）

    arXiv:2610.33262v1 Announce Type: cross  Abstract: We study efficient, adaptive auditing of categorical labels in a fixed dataset, motivated by biodiversity-credit programs in which expert verification of species records is costly and downstream scores (e.g., payment) are more sensitive to errors on certain labels than on others. Our proposed ATLAS framework couples anytime-valid inference with a task-aware adaptive auditing policy. Representing label corrections through a confusion matrix, we prioritize which data points to inspect next based on both label uncertainty and the influence of potential label corrections on the downstream score function. For linear scores such as the mislabel rate, our priority-aware approach certifies the overall label error rate to $\pm 2$ percentage points with 24% and 11% fewer audits than random sampling on biodiversity data from Kenya and a widely used CIFAR-10N benchmark, respectively. For nonlinear scores such as the Hill-Shannon diversity, we expl
    
[^241]: 用于多模态X射线和中子散射反演的组合式神经网络

    Compositional neural networks for multimodal X-ray and neutron scattering inversion

    [https://arxiv.org/abs/2610.29362](https://arxiv.org/abs/2610.29362)

    该论文提出了一种基于先验摊销流匹配的模块化框架，通过在推理阶段用组合采样器融合分别训练的单观测后验估计器，实现多模态X射线和中子散射数据的概率性联合反演，无需联合再训练即可复用预训练模型。

    

    X射线和中子散射是凝聚态物理和材料科学中结构表征不可或缺的手段。然而，由于散射实验通常只记录强度而不含相位信息，不同的结构可能产生相似的散射图案，使得相应的逆问题不具有唯一性。通过结合互补的测量可以减少这种歧义，但将其整合到机器学习框架中仍然具有挑战性。在此，我们提出了一种基于先验摊销流匹配的概率性多测量分析模块化框架。针对每种观测类型或物理探针分别训练单观测后验估计器，并仅在推理时通过组合采样器将其结合。这种后期融合方法允许预训练的单观测模型被重复使用，而无需专门的联合再训练。作为一个典型案例，我们考虑反射率测量……

    arXiv:2610.29362v1 Announce Type: new  Abstract: X-ray and neutron scattering are indispensable for structural characterization across condensed-matter physics and materials science. Yet, because scattering experiments generally record intensities without phase information, distinct structures can yield similar scattering patterns, rendering the corresponding inverse problems non-unique. This ambiguity can be reduced by combining complementary measurements, but their integration into machine-learning frameworks remains challenging. Here we introduce a modular framework for probabilistic multi-measurement analysis based on prior-amortized flow matching. Single-observation posterior estimators are trained separately for individual observation types or physical probes and combined only at inference time through a compositional sampler. This late fusion approach allows pretrained single-observation models to be reused without dedicated joint retraining. As a case in point, we consider refl
    
[^242]: 不动集与计算相关均衡的困难性

    Fixed Sets and the Hardness of Computing Correlated Equilibria

    [https://arxiv.org/abs/2610.28141](https://arxiv.org/abs/2610.28141)

    本文证明计算多面体博弈的近似相关均衡是PPAD完全的（与纳什均衡同等难度），挑战了相关均衡比纳什均衡更容易求解的传统观点，并通过提出“不动集”这一不动点的自然推广建立了相关均衡与不动点理论的概念性联系。

    

    我们证明，计算多面体博弈的近似相关均衡是PPAD完全的，因此在多项式时间内等价于寻找纳什均衡。这一结果挑战了博弈论中的传统观点，即相关均衡是比纳什均衡更容易处理的对象。从概念层面，我们将相关均衡与不动点的一个自然推广联系起来，我们称之为“不动集”。连续函数 f 的不动集是一个非空有限集 S，满足 S ⊆ conv f(S)。我们的主要技术贡献是证明了即使达到逆多项式精度，计算不动集仍是PPAD困难的。

    arXiv:2610.28141v1 Announce Type: cross  Abstract: We show that computing an approximate correlated equilibrium of a polytope game is PPAD-complete, and thus polynomial-time equivalent to finding a Nash equilibrium. This challenges the conventional view in game theory that correlated equilibria are more tractable objects than Nash equilibria. From a conceptual standpoint, we connect correlated equilibria to a natural generalization of fixed points, which we call fixed sets. A fixed set of a continuous function $f$ is a nonempty finite set $S$ such that $S \subseteq \operatorname{conv} f(S)$. Our main technical contribution is to prove that computing a fixed set is PPAD-hard even at inverse polynomial precision.
    
[^243]: 基于激活空间引导的可验证奖励强化学习探索方法

    Exploration via Activation-Space Steering for Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2610.27012](https://arxiv.org/abs/2610.27012)

    提出激活空间探索（ASE）方法，通过聚类正确与错误响应的激活差异构建潜在探索向量库，并将其应用于训练采样以引导模型探索多样化推理路径，从而提升可验证奖励强化学习在数学推理上的性能。

    

    可验证奖励强化学习（RLVR）通过采样模型的响应、使用自动验证器对其评分，并根据所得奖励更新策略来训练语言模型。因此，策略只能从其自身采样生成的轨迹中获得直接学习信号。然而，从同一策略反复采样可能会集中在相似的推理路径上，在固定的采样预算下导致有用的替代路径未被探索。我们提出了激活空间探索（ASE），这是一种用于多样化训练采样的在线方法。通过聚类同一问题正确与错误响应之间的激活差异，ASE 构建了一个结构化的潜在探索向量库。随后，不同的向量被应用于后续的采样过程中，以探索替代的推理路径。在 DAPO-Math-17k 上训练 Qwen3-1.7B-Base 和 Qwen3-8B-Base 时，ASE 在六个数学推理基准上提高了平均 Mean@8 性能。

    arXiv:2610.27012v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) trains a language model by sampling responses, scoring them with an automatic verifier, and updating the policy from the resulting rewards. The policy therefore receives direct learning signals only from trajectories that appear in its rollouts. Yet repeated sampling from the same policy can concentrate on similar reasoning paths, leaving useful alternatives unexplored under a fixed rollout budget. We introduce Activation-Space Exploration (ASE), an online method for diversifying training rollouts. By clustering activation differences between correct and incorrect responses to the same problem, ASE builds a structured bank of latent exploration vectors. Different vectors are then applied to subsequent rollouts to explore alternative reasoning paths. On Qwen3-1.7B-Base and Qwen3-8B-Base trained on DAPO-Math-17k, ASE improves average Mean@8 across six mathematical reasoning benchmarks
    
[^244]: MemT-1.0：迈向记忆原生演化

    MemT-1.0: Towards Memory-Native Evolution

    [https://arxiv.org/abs/2610.26825](https://arxiv.org/abs/2610.26825)

    MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。

    

    记忆原生演化要求将记忆作为基础模型的内在能力，使其能够持续积累和利用信息。为此，我们提出了MemT-1.0，一种将记忆内化为模型计算一部分的记忆基础模型新架构。MemT-1.0基于Metis构建，采用Gated DeltaNet-2，并在每一轮内使用稳定的token级记忆更新机制。我们进一步引入Mixture-of-Storage（存储混合）来扩展原生记忆容量。在经过精心准备的主干模型之后，我们开发了一个由继续预训练和后训练组成的两阶段训练流程，并由340亿（34B）token的训练数据以及高效的训练与推理基础设施支撑。我们发布了包含MemT-1.0-Nano和MemT-1.0-Flash的模型家族，并对记忆能力和通用能力进行了全面评估。结果表明，它们在（各项评估中）优于参数化记忆基线。

    arXiv:2610.26825v2 Announce Type: replace  Abstract: Memory-native evolution requires making memory an intrinsic capability of foundation models, enabling them to continuously accumulate and utilize information. To this end, we introduce MemT-1.0, a new architecture of memory foundation models that internalize memory as part of model computation. Building on Metis, MemT-1.0 employs Gated DeltaNet-2 with a stable token-level memory update mechanism within each turn. We further introduce Mixture-of-Storage to scale native memory capacity. Following a deliberate backbone preparation, we develop a two-stage training pipeline consisting of continued pre-training and post-training, supported by 34B tokens of training data and efficient training and inference infrastructure. We release a model family comprising MemT-1.0-Nano and MemT-1.0-Flash, and conduct a comprehensive evaluation of memory and general capabilities. The results show that they outperform the parametric memory baselines on av
    
[^245]: 世界模型、树模型与基础模型在零售预测中的应用：来自四个公开数据集的证据

    World Models, Trees and Foundation Models for Retail Forecasting: Evidence from Four Public Data Sets

    [https://arxiv.org/abs/2610.25170](https://arxiv.org/abs/2610.25170)

    本文提出了一种在价格与促销计划下推进商品状态的世界模型，并在四个公开零售数据集上针对从单品单店订货到全链财务规划的不同决策粒度，系统比较了它与LightGBM、Transformer及TimesFM等时序基础模型的预测表现，借助配对检验与模型置信集确定各粒度下的最优模型。

    

    零售商在产品、门店与时间层级的不同粒度上做出不同的决策：单品在单店的门店订货、单品跨门店的供应商资金投入、按门店衡量的品类绩效，以及整个连锁企业的财务规划。使用哪种预测模型取决于决策粒度。本文在四个公开数据集（M5、Dominick's 的六个品类、Breakfast at the Frat，以及 Corporación Favorita——后者在全部开发过程中被保留不参与）上，通过滚动预测起点、配对检验与模型置信集，对各粒度下的模型进行了比较。研究通过这些数据记录的计划输入来涵盖价格与促销因素，而不涉及产品、商品组合或渠道。所比较的模型包括：一个在价格与促销计划下推进每个商品状态的世界模型、去除该状态的同一网络、LightGBM、Temporal Fusion Transformer、TimesFM、Chronos-Bolt、Timestral、季节性朴素预测，以及各模型预测的等权平均……

    arXiv:2610.25170v2 Announce Type: replace-cross  Abstract: A retailer makes different decisions at different grains of the product, store and time hierarchy: store orders for an item in a store, supplier funding for an item across stores, category performance by store, and the financial plan for the chain. Which forecasting model to use depends on the grain. This paper compares models at each grain on four public data sets (M5, six Dominick's categories, Breakfast at the Frat, and Corporaci\'on Favorita, the last kept out of all development), over rolling forecast origins with paired tests and model confidence sets. It covers price and promotion through the plan inputs these data record, not product, assortment or place. The models are a world model that carries each item's state forward under the price and promotion plan, the same network without that state, LightGBM, the Temporal Fusion Transformer, TimesFM, Chronos-Bolt, Timestral, seasonal naive, and an equal-weight average of the 
    
[^246]: 动态核注意力：从过去记录估计当前状态

    Dynamical Kernel Attention: Estimating the Present from a Record of the Past

    [https://arxiv.org/abs/2610.24026](https://arxiv.org/abs/2610.24026)

    本文提出动态核注意力（DKA），从随机微分方程控制的潜在状态的鲁棒估计中推导注意力机制，使每个注意力头提取有界证据、在证据不足时弃权并拥有输出变化可证明有界的记忆窗口，在语言建模上优于RoPE和ALiBi，并以更小的键值缓存匹配滑动窗口注意力。

    

    自注意力机制编码了过去词元所在的位置，但未能编码其不确定性如何随时间增长：随着上下文扩展，证据无界累积，且没有任何机制指明哪些词元可以安全丢弃。我们提出了动态核注意力（Dynamical Kernel Attention, DKA），它从由随机微分方程控制的潜在状态的鲁棒估计中推导出注意力机制。在该模型下，每个注意力头从任意上下文中提取的证据都是有界的。这使得每个注意力头在相关证据较弱时能够选择弃权，并为每个注意力头提供了一个记忆窗口——在窗口之外丢弃词元只会使其输出产生可证明的有界变化。在语言建模任务上，DKA在训练长度内优于RoPE，在远超训练长度的场景下优于ALiBi，并以显著更小的键值缓存达到了与滑动窗口注意力相当的性能。

    arXiv:2610.24026v2 Announce Type: replace  Abstract: Self-attention encodes where a past token lies, but not how its uncertainty grows with time: evidence accumulates without bound as the context grows, and nothing indicates which tokens are safe to discard. We introduce Dynamical Kernel Attention (DKA), which derives attention from robust estimation of a latent state governed by a stochastic differential equation. Under this model, each head draws bounded evidence from any context. This lets each head abstain when relevant evidence is weak, and gives each head a memory window outside which discarding tokens changes its output by a provably bounded amount. On language modeling, DKA outperforms RoPE within the training length and ALiBi far beyond it, and matches sliding-window attention with a substantially smaller key-value cache.
    
[^247]: 在策略蒸馏需要完成每一次 Rollout 吗？

    Does On-Policy Distillation Need to Complete Every Rollout?

    [https://arxiv.org/abs/2610.21337](https://arxiv.org/abs/2610.21337)

    该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。

    

    在策略蒸馏（On-policy Distillation, OPD）通过对学生模型生成的 rollout 序列提供反馈，有效地将教师模型的能力迁移给学生模型。然而，生成完整的学生 rollout 序列并为每个 token 计算教师反馈，使得 OPD 的计算开销十分高昂。我们提出了随机截断在策略蒸馏（Stochastic Truncated On-Policy Distillation, STOPD），该方法通过随机截断学生 rollout 序列，并利用分块逆概率加权（blockwise IPW）校正由此产生的采样偏差，从而加速 OPD。在每一步训练中，我们在生成前为每个 rollout 采样一个停止位置，让学生模型生成至该位置或自然终止。我们对每个生成的 token 获取教师反馈，并使用分块 IPW 补偿保留后续 token 的较低概率，同时对权重加以控制。我们证明了这种加权方式在期望意义上保持了原始 OPD 的梯度总和不变，并推导出了一步更新的误差界……

    arXiv:2610.21337v1 Announce Type: cross  Abstract: On-policy distillation (OPD) effectively transfers capabilities from a teacher to a student through feedback on student-generated rollouts. However, generating complete student rollouts and computing teacher feedback for every token make OPD computationally expensive. We introduce Stochastic Truncated On-Policy Distillation (STOPD), which accelerates OPD by randomly truncating student rollouts and correcting the resulting sampling bias with blockwise inverse-probability weighting (IPW). At each training step, we sample a stopping position for each rollout before generation and let the student generate until that position or natural termination. We obtain teacher feedback on every generated token and use blockwise IPW to compensate for the lower probability of retaining later tokens while controlling the weights. We prove that this weighting preserves Vanilla OPD's gradient sum in expectation and derive one-step update-error bounds for 
    
[^248]: SarcasTraps：讽刺对土耳其语仇恨言论检测的影响

    SarcasTraps: Effect of Sarcasm on Turkish Hate Speech Detection

    [https://arxiv.org/abs/2610.16957](https://arxiv.org/abs/2610.16957)

    本文构建了SarcasTraps——一个包含479条数据、沿仇恨言论与讽刺双维度标注的土耳其语外部测试集，并通过实验揭示讽刺会显著干扰仇恨言论检测：微调的编码器模型在讽刺性仇恨言论上召回率下降，零样本大语言模型同样受到影响。

    

    标准的仇恨言论检测数据集通常强调显性的有害语言，但针对性伤害也可以通过间接方式表达，而不使用明显仇恨的措辞。讽刺正是这样一种间接表达形式，它可以通过假意称赞、夸张、幽默或表面上的礼貌来传达有害意图。本文介绍了SarcasTraps，这是一个用于分析讽刺对土耳其语仇恨言论检测影响的针对性外部测试集。SarcasTraps包含479条数据，并沿两个维度进行标注：仇恨言论和讽刺。我们使用BERTurk和XLM-RoBERTa在两个现有的土耳其语仇恨言论数据集上训练分类器，并在SarcasTraps上对其进行评估，严格将其用作外部测试集。我们还在零样本设置下于SarcasTraps上评估了GPT-4.1和Llama-3.3-70B。结果表明，讽刺会显著影响仇恨言论检测。微调的编码器模型在检测讽刺性仇恨言论时召回率下降，而零样本大语言模型（原文摘要在此处截断）……

    arXiv:2610.16957v1 Announce Type: cross  Abstract: Standard hate speech detection datasets often emphasize explicit harmful language, but directed harm can also be expressed indirectly, without clearly hateful wording. Sarcasm is one such form of indirect expression, as it can convey harmful intent through mock praise, exaggeration, humor, or apparent politeness. In this paper, we introduce SarcasTraps, a targeted external test set for analyzing the effect of sarcasm on Turkish hate speech detection. SarcasTraps contains 479 entries annotated along two dimensions: hate speech and sarcasm. We train classifiers using BERTurk and XLM-RoBERTa on two existing Turkish hate speech datasets and evaluate them on SarcasTraps, using it strictly as an external test set. We also evaluate GPT-4.1 and Llama-3.3-70B in a zero-shot setting on SarcasTraps. Results show that sarcasm substantially affects hate speech detection. Fine-tuned encoder models lose recall on sarcastic hate, while zero-shot LLMs 
    
[^249]: 自然代码变换使智能体在漏洞分析中产生混淆

    Natural Code Transformation Confuses Agents in Vulnerability Analysis

    [https://arxiv.org/abs/2610.14350](https://arxiv.org/abs/2610.14350)

    本文发现AI智能体在分析具有相同功能和相同漏洞的不同真实代码提交时会产出截然不同的结果，并据此提出从真实开发模式中挖掘自然且功能保持的代码变换，作为防御攻击者智能体的新手段。

    

    arXiv:2610.14350v2 公告类型： cross 摘要：在展示了自动化漏洞发现的强大潜力之后，AI智能体也可以被用来大规模攻击软件系统。当今的软件系统需要一些机制来平衡攻防双方的力量，通过减缓潜在攻击的速度，为防御者争取时间来修补漏洞并强化软件。沿着这一思路，我们对漏洞分析中智能体行为轨迹的研究揭示了一个令人惊讶的发现：对于承载相同功能和相同漏洞的真实世界代码提交，智能体产生了截然不同的分析结果。这一现象在多个代码领域和漏洞类别中持续出现。这促使我们研究源自真实世界软件开发过程的自然且保持功能不变的代码变换，以及其作为防御攻击者智能体的潜力。我们通过从真实世界代码提交中挖掘反复出现的开发模式，构建了一个自然变换语料库。

    arXiv:2610.14350v2 Announce Type: cross  Abstract: Having demonstrated a strong potential for automating vulnerability discovery, AI agents can also be used to attack software systems at scale. Today's software systems need mechanisms to level the playing field by slowing down potential attacks, giving them time to patch vulnerabilities and strengthen software. Along this line, our study of agent trajectories in vulnerability analysis shows a surprising finding: agents produce dramatically different results across real-world code commits that carry the same functionality and the same vulnerability. This phenomenon appears consistently in multiple code domains and vulnerability categories.   It motivates us to study natural and functionality-preserving code transformations derived from real-world software development, and their potential as a defense against attacker agents. We build a corpus of natural transformations by mining recurring development patterns from real-world code commit
    
[^250]: 神经偏微分方程求解器学到了正确的动力学吗？

    Do Neural PDE Solvers Learn the Right Dynamics?

    [https://arxiv.org/abs/2610.06952](https://arxiv.org/abs/2610.06952)

    该论文提出了一个超越传统预测误差评分的评估框架，通过考察误差形成、集合几何结构和极端事件三个互补维度，来检验神经PDE求解器是否真正再现了系统的动力学行为。

    

    神经偏微分方程（PDE）求解器可以实现较低的预测误差，但它们是否再现了所建模系统的动力学？仅凭预测分数无法给出完整的答案：它们衡量的是与参考解的一致性，但对于误差如何累积、邻近状态如何发散、以及极端事件如何产生，所提供的洞察十分有限。我们提出了一个评估框架，直接考察确定性和随机神经求解器中的这些行为。通过演化邻近初始状态组成的集合，并将其与直接数值模拟进行比较，我们评估了所学动力学的三个互补方面：误差形成、集合几何结构以及极端事件。在二维Kolmogorov流上的实验揭示了传统评分方式可能掩盖的局限性：较小的轨迹误差可能只是源于较弱的误差放大，而局部更新反而不够准确；模型可以在匹配集合的整体散布程度和有效维数的同时，未能正确地……（原文摘要在此处截断）

    arXiv:2610.06952v1 Announce Type: new  Abstract: Neural PDE solvers can achieve low prediction errors, but do they reproduce the dynamics of the systems they model? Prediction scores alone offer an incomplete answer: they measure agreement with reference solutions but provide limited insight into how errors accumulate, nearby states diverge, or extreme events arise. We propose an evaluation framework that directly examines these behaviors in deterministic and stochastic neural solvers. By evolving ensembles of nearby initial states and comparing them with direct numerical simulation, we assess three complementary aspects of learned dynamics: error formation, ensemble geometry, and extreme events. Experiments on two-dimensional Kolmogorov flow reveal limitations that conventional scores can obscure. Smaller trajectory errors can reflect weaker error amplification despite less accurate local updates. Models can match an ensemble's overall spread and effective dimension while failing to c
    
[^251]: OVAL：用于KV缓存压缩的输出感知局部页面基底

    OVAL: Output-Aware Local Page Bases for KV Cache Compression

    [https://arxiv.org/abs/2610.06686](https://arxiv.org/abs/2610.06686)

    OVAL提出用一对输出感知的低秩基底同时充当KV页面的检索索引和压缩存储，使长上下文解码中KV缓存被丢弃精确页面后仍不增加任何额外内存开销。

    

    长上下文解码——无论是处理长推理轨迹、长视频还是机器人任务片段——正日益受到键值（KV）缓存内存的限制。页面检索稀疏注意力通过仅关注得分最高的KV页面，降低了每个解码步骤的计算开销。然而，它仍然完整地存储每一个页面，并为每个页面额外添加一个用于评分的小摘要（digest），因此缓存最终反而比完整缓存还要略大一些。我们提出OVAL，它将每个页面存储为一对输出感知的低秩基底，这些基底同时充当检索索引和存储本身。关键基底对页面进行评分，并在页面被选中时重建其键（keys）；PCA基底则重建其值（values）。这样，精确的页面可以被丢弃，且不会向缓存添加任何额外内容。其核心思想是关键基底应当是输出感知的：键中的低秩误差会在一阶程度上扰动注意力输出，但只会以二阶程度扰动检索排序所依据的页面质量……（摘要在此处截断）

    arXiv:2610.06686v2 Announce Type: replace  Abstract: Long-context decoding, whether over long reasoning traces, long videos, or robot episodes, is increasingly limited by the memory of the key-value (KV) cache. Page-retrieval sparse attention makes each decoding step cheaper by attending only to the top-scoring KV pages. However, it still stores every page exactly, and adds a small digest per page for scoring, so the cache ends up slightly \emph{larger} than the full cache. We propose OVAL, which stores each page as a pair of output-aware low-rank bases that act as both the retrieval index and the storage. A key basis scores the page and rebuilds its keys when the page is selected, and a PCA basis rebuilds its values. The exact pages can then be dropped, and nothing extra is added to the cache. The key idea is that the key basis should be output-aware. A low-rank error in the keys perturbs the attention output at first order, but perturbs the page mass that retrieval ranks only at seco
    
[^252]: 相信方向，搜索步长：面向大语言模型微调的零阶与一阶混合优化方法

    Trust the Direction, Search the Step: Zero-and-First-Order Methods for LLM Fine-Tuning

    [https://arxiv.org/abs/2610.02190](https://arxiv.org/abs/2610.02190)

    提出ZFO轻量级框架，将优化中的方向选择与步长选择解耦——用可信的一阶优化器确定方向，仅沿该一维子空间做两次零阶评估构建局部模型，从而以低于完整线搜索的开销实现曲率感知的自适应步长选择，并配有理论保证，适用于大语言模型微调。

    

    步长选择仍然是大规模神经网络优化中的核心挑战：保守的步长会减慢收敛速度，而过于激进的步长则可能导致优化过程不稳定。我们结合零阶与一阶优化，提出了一种将方向选择与步长选择解耦的轻量级框架。ZFO使用可信的一阶优化器确定更新方向，并仅沿这一维子空间进行零阶函数评估来决定移动多远。利用当前的梯度信息和两次额外的目标函数评估，ZFO实例沿所提方向构建目标函数的局部模型，并在有界搜索区间内选择具有曲率感知的步长。由此产生了一种开销低于完整线搜索的自适应步长选择机制。我们提供了理论保证，证明共享样本评估能够产生可靠的有限……（原文摘要至此被截断）

    arXiv:2610.02190v1 Announce Type: new  Abstract: Step-size selection remains a central challenge in large-scale neural network optimization; conservative steps slow convergence, while aggressive steps can destabilize it. We combine \textbf{Z}ero-and-\textbf{F}irst-\textbf{O}rder optimization~(ZFO) and propose a lightweight framework that decouples direction selection from step-size. ZFO uses a trusted first-order optimizer to determine the direction and performs zeroth-order evaluations only along this one-dimensional subspace to choose how far to move. Using the current {gradient information} and two additional objective function evaluations, ZFO instances construct a local model of the objective function along the proposed direction and select a curvature-aware step within a bounded search interval. This yields an adaptive step-selection mechanism that costs less than a full line search. We provide theoretical guarantees to show that shared-sample evaluations produce reliable finite-
    
[^253]: 低损失区域的错位导致Grokking（顿悟）现象

    Misalignment of Low-Loss Regions Causes Grokking

    [https://arxiv.org/abs/2610.00620](https://arxiv.org/abs/2610.00620)

    论文提出基于模式连通性与低损失区域几何的分析框架，指出grokking现象源于训练集与验证集所诱导的低损失区域之间的错位，并以稳定的反grokking反例挑战了现有的相关性解释。

    

    Grokking是指模型在已经过拟合训练集之后，验证集泛化能力延迟出现的现象。尽管该现象最早在使用Transformer训练的小型算法任务中被观察到，但其底层机制至今仍无定论。在本工作中，我们开发了一个基于模式连通性与低损失区域几何结构的分析框架。该框架预测标准的模运算设定并不总是产生grokking现象：在保持对称性的训练/验证集划分下，我们观察到一个稳定的反grokking案例，其中验证集性能始终未能恢复。这一反例对现有的若干关于grokking的相关性解释提出了挑战。更广泛地说，我们的分析框架和结果进一步表明，当训练集划分与验证集划分所诱导的低损失区域发生错位时，grokking现象便会出现。一旦这些区域充分对齐，仅凭训练超参数就无法……（摘要在此处截断）

    arXiv:2610.00620v1 Announce Type: cross  Abstract: Grokking refers to the delayed emergence of validation-set generalization after a model has already overfit the training set. Although first observed in small algorithmic tasks trained with transformers, its underlying mechanism remains unsettled. In this work, we develop an analysis framework based on mode connectivity and the geometry of low-loss regions. The framework predicts that the standard modular-arithmetic setting does not always produce grokking: under a symmetry-preserving train/validation split, we observe a stable anti-grokking case in which validation performance does not recover. This counterexample challenges several existing correlational explanations of grokking. More broadly, our analysis framework and results further suggest that grokking arises when the low-loss regions induced by the training and validation partitions are misaligned. Once these regions become well aligned, training hyperparameters alone cannot pr
    
[^254]: 一池多靶：一个守恒层以及档案数据所能识别的信息

    One pool, many targets: a conservation layer and what archival data can identify

    [https://arxiv.org/abs/2610.00445](https://arxiv.org/abs/2610.00445)

    本文提出一个保证有限向导RNA加载的RISC池守恒的可微平衡层，并利用档案脱靶数据检验发现：平衡耦合虽使校正后的热力学亲和力以预测器所需方向与抑制效果弱相关，但在剂量与跨上下文实验中并未显示出超越独立评分的额外预测价值。

    

    当将成对的向导RNA–转录本评分独立地解释为占据率时，它们无法保证有限的载有向导RNA的RISC池的守恒性。我们提出了一个可微的标量平衡层：一个具有唯一正根和精确隐式梯度的守恒方程。它给出了一个再分配定理、一个有条件的高资源极限、对检索近似的分析，以及一个条件性秩不变结果：在同一构建体、同一剂量下，按分数占据率排序无法区分平衡模型与独立评分。因此，我们在档案脱靶数据上检验了该命题所留下的两个开放实验：剂量效应与跨上下文。校正后的热力学亲和力以有效预测器所需的方向与测量的抑制效果呈弱相关，但配对置换检验与构建体聚类自举法未能证明这种耦合带来的额外预测价值：（原文摘要在此处截断）

    arXiv:2610.00445v1 Announce Type: new  Abstract: Pairwise guide--transcript scores do not enforce conservation of a finite guide-loaded RISC pool when they are interpreted independently as occupancies. We formulate a differentiable scalar equilibrium layer: one conservation equation with a unique positive root and exact implicit gradients. It yields a redistribution theorem, a qualified high-resource limit, an analysis of the retrieval approximation, and a conditional rank-invariance result: within one construct at one dose, rankings by fractional occupancy cannot distinguish equilibrium from independent scoring. We therefore audit the two experiments that proposition leaves open, dose and cross-context, on archival off-target data. Corrected thermodynamic affinities associate weakly with measured repression in the direction a working predictor requires, but a paired permutation test and a construct-cluster bootstrap do not establish added predictive value from the coupling: what survi
    
[^255]: 基于预测误差触发伪标签重训练的在线时间序列预测方法

    Pseudo-Label-Triggered Retraining from Forecast Errors for Online Time Series Forecasting

    [https://arxiv.org/abs/2609.39789](https://arxiv.org/abs/2609.39789)

    本文提出PILOT框架，利用预测误差构建伪标签并训练轻量级评分器，从误差动态中学习时间序列预测模型“何时重训练”，从而在有限部署资源下有效维持预测性能。

    

    真实世界的时间序列预测系统运行在非平稳数据流环境中，预测性能可能随时间推移而下降。虽然重训练可以恢复性能，但会带来不小的计算和运营成本。在部署资源有限的情况下，关键挑战因此不仅在于如何重训练，还在于何时重训练。现有的重训练策略通常依赖于漂移警报或模型过时程度等间接指标，而我们则使用实际发生的预测误差作为直接的部署反馈。在本文中，我们提出了PILOT（伪标签信息学习的在线触发器），这是一个从预测误差动态中学习何时重训练的在线重训练框架。由于真实的重训练标签不可用，PILOT从未来预测误差的增加中构建伪标签，并训练一个轻量级评分器根据观测到的误差状态来预测该伪标签。在部署时，PILOT仅使用……（原文摘要在此处被截断）

    arXiv:2609.39789v1 Announce Type: cross  Abstract: Real-world time series forecasting systems operate under non-stationary data streams, where forecasting performance may degrade over time. Although retraining can recover the performance, it incurs non-trivial computational and operational costs. Under limited deployment resources, the key challenge is therefore not only how to retrain but also when to retrain. While existing retraining policies often rely on indirect indicators such as drift alarms or model staleness, we instead use realized forecast errors as direct deployment feedback. In this paper, we propose PILOT (Pseudo-label-Informed Learned Online Trigger), an online retraining framework that learns when to retrain from forecast-error dynamics. Since ground-truth retraining labels are unavailable, PILOT constructs a pseudo-label from future increases in forecast error and trains a lightweight scorer to predict it from observed error states. At deployment, PILOT uses only comp
    
[^256]: ElectrolyteFM：通过跨性质知识学习统一电解液性质预测

    ElectrolyteFM: Unifying Electrolyte Property Prediction through Cross-Property Knowledge Learning

    [https://arxiv.org/abs/2609.39340](https://arxiv.org/abs/2609.39340)

    提出统一多性质预测模型ElectrolyteFM，通过独立学习性质特定表示、结合单独训练的专家池与路由机制捕获跨性质共享知识，解决了联合训练中跨性质干扰问题，实现电解液多种性质的更准确预测。

    

    电解液配方设计需要平衡多种物理化学性质，然而现有模型往往仅关注有限的性质子集。孤立地学习每种性质可能会忽略可迁移的化学信息，而不加区分的知识共享则可能引入跨性质干扰。我们的有向迁移分析表明，联合学习两个性质预测任务相比单独训练可能提升也可能降低预测效果，且任务间存在不对称的迁移效应。我们提出了ElectrolyteFM，这是一个统一的多性质预测模型，通过有效识别和利用性质特定特征以及跨性质共享的知识，能够更准确地预测每种电解液的多种性质。更具体地说，ElectrolyteFM独立学习性质特定的表示，并通过单独训练的专家池捕获跨性质知识，再由路由器选择相关的共享知识……

    arXiv:2609.39340v1 Announce Type: new  Abstract: Electrolyte formulation design requires balancing multiple physicochemical properties, yet existing models often focus on a limited subset. Learning each property in isolation can overlook transferable chemical information, whereas indiscriminate sharing can introduce cross-property interference. Our directed transfer analysis shows that jointly learning two property prediction tasks can improve or degrade prediction relative to separate training, with asymmetric transfer effects between the tasks. We propose ElectrolyteFM, a unified multi-property prediction model which can more accurately predict multiple properties of each electrolyte by effectively identifying and utilizing property-specific features and knowledge shared across properties. More specifically, ElectrolyteFM learns property-specific representations independently and captures cross-property knowledge through a separately trained expert pool. A router selects relevant sha
    
[^257]: 基于预测状态的无穷记忆过程生成式序列建模

    Generative sequence modeling for infinite memory processes via predictive states

    [https://arxiv.org/abs/2609.38524](https://arxiv.org/abs/2609.38524)

    本文提出一种基于预测状态的生成式序列建模新方法，能够处理具有无穷记忆的随机过程，当过去历史可被压缩为低维充分统计量时实现快速收敛，且估计问题的统计复杂度仅由预测状态空间的内在维数决定。

    

    我们考虑估计多元随机过程的一步预测条件分布。许多现有方法依赖于有限范围记忆、稀疏性或可加性等假设，这些假设对于具有长程非线性交互作用的过程可能并不适用。然而，在没有这些结构性假设的情况下，由于维数灾难，非参数估计面临巨大挑战。为了应对这一挑战，我们提出了一种基于过程预测状态的新估计方法，该方法可以处理可能具有无穷范围记忆的过程。我们证明了当过去的历史信息能够被压缩成一个足以预测未来的低维统计量时，我们的估计器能够达到快速收敛速度。具体而言，我们证明了该估计问题的统计复杂度由预测状态空间的内在维数决定。我们为基于该方法的一种具体实现建立了理论保证（摘要在此处被截断）。

    arXiv:2609.38524v1 Announce Type: new  Abstract: We consider estimating the one-step-ahead conditional distribution of a multivariate stochastic process. Many existing approaches rely on assumptions such as finite-range memory, sparsity, or additivity, which can be poorly suited to processes with long-range nonlinear interactions. However, without such structural assumptions, nonparametric estimation is challenging due to the curse of dimensionality. To address this challenge, we introduce a new estimation approach based on the predictive states of a process, possibly with infinite-range memory. We show that our estimator achieves fast convergence rates when the past history can be compressed into a low-dimensional statistic that is sufficient for predicting the future. Specifically, we show that the statistical complexity of the estimation problem is determined by the intrinsic dimension of the predictive state space. We establish guarantees for an instantiation of our method based on
    
[^258]: LeapQuant：基于精确循环状态量化的高效线性注意力

    LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization

    [https://arxiv.org/abs/2609.38166](https://arxiv.org/abs/2609.38166)

    LeapQuant 提出了一种无需训练的按窗口量化方法，通过跳过整个 token 窗口、仅在窗口末尾对循环状态量化一次来避免误差累积，从而在 8 位循环状态量化下实现近乎无损的线性注意力推理性能。

    

    近期的大语言模型（LLM）越来越多地采用混合设计，用线性注意力（如 Gated DeltaNet (GDN) 和 Kimi Delta Attention (KDA)）替代标准注意力。尽管这些方法将上下文压缩为固定大小的循环状态，大幅降低了长上下文处理的成本，但反复读取和更新该状态仍是推理过程中的主要瓶颈。量化是降低这一成本的自然途径，但由于舍入误差的累积以及状态中异常行列的存在，量化可能显著降低模型质量。为应对这些挑战，我们提出了 LeapQuant，这是一种无需训练的方法，在 8 位循环状态量化下实现了近乎无损的性能。首先，为减轻误差累积，我们提出了按窗口量化，即跳过一个 token 窗口，仅在窗口结束时对状态量化一次。在窗口内部，输出由固定的低比特……（摘要至此截断）

    arXiv:2609.38166v1 Announce Type: cross  Abstract: Recent LLMs increasingly adopt hybrid designs that replace standard attention with linear attention, such as Gated DeltaNet (GDN) and Kimi Delta Attention (KDA). Although they compress the context into a fixed-size recurrent state and substantially reduce the cost of long-context processing, repeatedly reading and updating that state remains a major inference bottleneck. Quantization offers a natural way to reduce this cost, but can significantly degrade model quality, due to the accumulation of rounding errors and the presence of outlier rows and columns in the state. To address these challenges, we propose LeapQuant, a training-free method that achieves near-lossless performance under 8-bit recurrent-state quantization. First, to mitigate error accumulation, we propose per-window quantization, which leaps over a window of tokens and quantizes the state only once at its end. Within a window, outputs are computed from the fixed low-bit
    
[^259]: 根因分析在何处失效：一种检索-重排序分解方法

    Where Root Cause Analysis Fails: A Retrieval-Reranking Decomposition

    [https://arxiv.org/abs/2609.36686](https://arxiv.org/abs/2609.36686)

    该论文提出检索-重排序分解方法，揭示了top@k准确率混淆“检索失败”与“重排序失败”的评估盲点，并发现复杂故障基准上统计基线有79-100%的概率错误排序真实原因，而图方法从未明显胜过最佳统计基线。

    

    在数百个传感器中识别异常的根本原因，对于防止复杂受监控系统中的安全事故和昂贵停机至关重要。现有研究使用top@k准确率来评估根因分析（RCA）方法。我们证明该指标存在一个根本性的盲点：它混淆了两种失效模式——检索失败（真实原因从未被纳入考虑）和重排序失败（真实原因被纳入考虑但排名过低）。在本工作中，我们引入了一种检索-重排序分解方法，并对四个知名基准数据集进行审查以揭示这一盲点。实验表明，在包含复杂故障的基准数据集上，统计基线方法有79-100%的概率对真实原因排序出错，而基于图的方法从未明显优于最佳统计基线，无论其因果图是在短故障窗口上学习、在保证包含真实原因的检索候选池上学习，还是在多天的正常运行数据上学习。

    arXiv:2609.36686v1 Announce Type: new  Abstract: Identifying the root cause of an anomaly among hundreds of sensors is critical for preventing safety incidents and costly downtime in complex monitored systems. Existing studies evaluate root cause analysis (RCA) methods using top@k accuracy. We show that this metric has a fundamental blind spot: it conflates two failure modes, retrieval failure, where the true cause is never considered, and reranking failure, where it is considered but ranked too low. In this work, we introduce a retrieval-reranking decomposition and audit four well-known benchmarks to expose this blind spot. Our experiments show that, on benchmarks with complex faults, statistical baselines mis-rank the true cause 79-100% of the time, and graph-based methods never clearly beat the best statistical baseline, whether their causal graphs are learned on short fault windows, on retrieved candidate pools guaranteed to contain the cause, or on multi-day normal-operation data.
    
[^260]: 可表示却学不到：编码秩与交互预测的误差下限

    Representable but Unlearned: Encoding Rank and the Interaction-Prediction Floor

    [https://arxiv.org/abs/2609.36208](https://arxiv.org/abs/2609.36208)

    该论文提出一种无需标签或拟合模型即可量化输入编码秩缺陷所致预测误差下限的方法，并证明siRNA交互预测中仅训练期特征掩码造成的秩下降（140降至72）会产生占拟合模型误差14.6%的不可逾越下限，而仅恢复三个化学列或移除掩码即可完全消除该下限。

    

    输入编码会限制预测器能够联合再现哪些测量对比，即使没有任何单一对比被强制归零。我们无需标签、损失函数或已拟合的模型，仅从编码器的等价类和固定的对比设计出发，即可计算出可达对比空间；将记录的对比投影到该空间上，便可为这些等价类上的任何无约束解码器给出经验误差下限。在一个包含140个矩形的siRNA交互面板上，图神经网络中仅用于训练阶段的特征掩码将165个终点状态合并为90个类，并将140个交互对比的秩削减至72。由此产生的误差下限为0.009980，占拟合模型交互平方误差的14.6%；而拟合模型本身的误差达到0.068335，略差于完全不预测任何交互的对照基线。只需最少恢复三个化学列即可恢复满秩。移除掩码后重新拟合可完全消除该下限，但交互MSE……

    arXiv:2609.36208v1 Announce Type: cross  Abstract: Input encodings can restrict which measured contrasts a predictor can jointly reproduce, even when no single contrast is forced to vanish. We compute the attainable contrast space from an encoder's equivalence classes and a fixed contrast design, without labels, loss, or a fitted model; projecting the recorded contrasts onto that space gives an empirical error floor for any unrestricted decoder on those classes. On a 140-rectangle siRNA interaction panel, a graph neural network's training-only feature mask merges 165 endpoint states into 90 classes and cuts the rank of the 140 interaction contrasts to 72. The resulting floor is 0.009980, which is 14.6% of the fitted model's interaction squared error; the fitted model reaches 0.068335, slightly worse than a control predicting no interaction at all. A minimum of three restored chemistry columns recovers full rank. Refitting without the mask removes the floor entirely, yet interaction MSE
    
[^261]: ScAn-Bench：评估缩放分析方法论

    ScAn-Bench: Evaluating Scaling Analysis Methodology

    [https://arxiv.org/abs/2609.35707](https://arxiv.org/abs/2609.35707)

    该论文提出了基于4524个语言模型检查点和8024个视觉-语言模型检查点的代理基准 ScAn-Bench，首次系统性地评估了跨不同数据模态下缩放分析中的数据获取与外推方法论。

    

    arXiv:2609.35707v3 公告类型：替换 摘要：近年来机器学习的进步由大规模基础模型驱动，其中缩放定律以及为架构、数据和超参数寻找最优缩放方案是推动最先进技术的关键。因此，令人惊讶的是，目前尚无系统性研究来评估跨不同模型类型获取缩放定律和缩放方案的方法论。为了揭示这一关键盲区并促进未来研究，我们引入了基于4524个语言模型流水线检查点和8024个视觉-语言模型流水线检查点的代理基准 ScAn-Bench-LLM 和 ScAn-Bench-VLM。在我们的基准上，我们首次针对不同数据模态，对缩放分析中的数据获取和外推方法论进行了系统性评估。

    arXiv:2609.35707v3 Announce Type: replace  Abstract: Recent progress in machine learning is driven by large-scale foundation models, where scaling laws and finding optimal scaling prescriptions for architecture, data, and hyperparameters are key in advancing the state-of-the-art. Therefore, it is surprising that no systematic study evaluates the methodology to obtain scaling laws and prescriptions across different model types. To shed light on this crucial blind spot and facilitate future research, we introduce the surrogate benchmarks ScAn-Bench-LLM and ScAn-Bench-VLM based on 4524 and 8024 checkpoints of language and vision-language model pipelines. On our benchmarks, we perform the first systematic evaluation of both data acquisition and extrapolation methodology for scaling analysis across different data modalities.
    
[^262]: 来自预训练表示的直觉向量引导基于扩散的推理

    Intuition Vectors from Pretrained Representations Steer Diffusion-Based Reasoning

    [https://arxiv.org/abs/2609.32619](https://arxiv.org/abs/2609.32619)

    该研究发现，冻结自监督视觉编码器表示空间中的简单向量差（“直觉向量”）能够直接支持抽象视觉推理，无需依赖语言推理或任务特定的表示学习。

    

    抽象视觉推理需要从少量示例中识别出关系，并将其应用于新的输入。近期的进展主要由大型语言模型、视觉-语言模型或专门的领域特定架构所驱动。目前尚不清楚抽象推理是否可以直接由预训练视觉表示的关系几何结构来支持，而无需依赖基于语言的推理或任务特定的表示学习。我们使用冻结的自监督视觉编码器来研究这一可能性。通过潜向量算术，我们证实了两个自监督编码器的表示几何结构能够支持Bongard问题上的抽象概念判别，并在ARC任务中编码输入-输出对之间的系统性关系。图像嵌入之间的简单向量差——我们称之为“直觉向量”——能够捕捉在同类实例间泛化的变换。

    arXiv:2609.32619v2 Announce Type: replace-cross  Abstract: Abstract visual reasoning requires identifying relationships between few examples and applying them to novel inputs. Recent advances have been driven by large language and vision-language models or specialized domain-specific architectures. It remains unclear whether abstract reasoning can be supported directly by the relational geometry of pretrained visual representations, without relying on language-based inference or task-specific representation learning. We investigate this possibility using frozen self-supervised vision encoders. With latent vector arithmetic, we confirmed that the representational geometry of two self-supervised encoders support abstract concept discrimination on Bongard problems and encode systematic relationships between input-output pairs in ARC tasks. Simple vector differences between image embeddings, which we call intuition vectors, capture transformations that generalize across instances of the sa
    
[^263]: 可解释性设计描述符组合在低数据分子检测上匹敌2048维基础模型嵌入

    Interpretable-by-Design Descriptor Portfolios Match a 2048-Dimensional Foundation Embedding on Low-Data Molecular Assays

    [https://arxiv.org/abs/2609.30789](https://arxiv.org/abs/2609.30789)

    该论文提出一种可解释性设计（interpretable-by-design）的描述符组合方法，通过贪婪拼接经过来源筛选的紧凑描述符块，在特征层面完全可审计的前提下，于低数据ADME/Tox检测上取得了与2048维CheMeleon基础嵌入相当的精度（平均AUC 0.762对0.764）。

    

    在低数据结构-活性预测中，分子表示的选择可能比预测器的选择更为关键，而表格基础模型进一步放大了这一效应。我们探究一个由紧凑、语义命名的描述符块组成的组合，能否在达到2048维CheMeleon嵌入精度的同时，在特征层面保持可审计性——即每个输入维度都带有模型名称和记录在案的训练来源。从固定的11维理化性质基础出发，我们仅利用带标签的上下文信息，贪婪地拼接经过来源筛选的描述符块。在九项ADME/Tox检测和50个评估单元上，以所有表示共同覆盖的分子所构成的公共覆盖子集进行评分，该组合达到平均测试AUC为0.762，相比之下CheMeleon为0.764，Mordred为0.756。与CheMeleon的汇总差距为+0.003 AUC（任务自助法95%置信区间[-0.020, +0.030]），这满足……（原文摘要至此截断）

    arXiv:2609.30789v1 Announce Type: new  Abstract: In low-data structure-activity prediction, the choice of molecular representation can matter more than the choice of predictor, and tabular foundation models sharpen that effect. We ask whether a portfolio of compact, semantically named descriptor blocks can reach the accuracy of a 2048-dimensional CheMeleon embedding while staying auditable at the feature level, meaning that every input dimension carries a model name and a recorded training provenance. Starting from a fixed 11-dimensional physicochemical base, we greedily concatenate provenance-screened blocks using the labelled context alone. Across nine ADME/Tox assays and 50 evaluation cells, scored on common-coverage subsets restricted to the molecules that every representation covers, the portfolio reaches a mean test AUC of 0.762, against 0.764 for CheMeleon and 0.756 for Mordred. The pooled gap to CheMeleon is +0.003 AUC (task-bootstrap 95% CI [-0.020, +0.030]), which satisfies o
    
[^264]: 图神经网络与能量腔方法用于组合优化

    Graph neural networks and the energetic cavity method for combinatorial optimization

    [https://arxiv.org/abs/2609.07456](https://arxiv.org/abs/2609.07456)

    通过对图神经网络进行微小修改以融入经典平均场启发式方法（主特征向量法与能量腔/最小和算法），可显著提升其在求解伊辛模型基态及组合优化问题上的性能。

    

    我们研究了使用图神经网络（GNN）来寻找伊辛模型近似基态的问题。高效地找到这些基态具有广泛的重要意义，因为通过适当选择耦合和场，许多组合优化问题都可以被表述为伊辛模型。精确求解这些问题非常困难，但已有许多优秀的启发式方法。这些启发式方法中的一类建立在平均场近似之上：其中一种方法使用适当定义矩阵的主特征向量，另一种是最小和算法，也称为能量腔方法。未经修改的GNN在这两种方法面前表现更差。我们考虑对GNN进行微小的修改以融入这些启发式方法，发现这能显著提升性能。尽管修改后的方法与其他深度学习方法相比具有竞争力，我们仍然发现模拟退火方法至少能够可靠地保持……

    arXiv:2609.07456v1 Announce Type: cross  Abstract: We study the use of graph neural networks (GNNs) for finding approximate ground states of Ising models. Efficiently finding these ground states is of broad significance because many combinatorial optimization problems can be formulated as an Ising model with the appropriate choice of couplings and fields. Exactly solving these problems is hard but there are many good heuristic methods. A lineage of these heuristics build from mean-field approximations: one approach uses the leading eigenvector of an appropriately defined matrix, another is the min-sum algorithm, also known as the energetic cavity method. Without modification, GNNs perform worse than both of these methods. We consider small modifications to the GNN to incorporate these heuristics and find that this considerably improves performance. While the modified approach is competitive against other deep-learning approaches, we still find that simulated annealing is reliably at le
    
[^265]: 摊销缩放定律构建成本

    Amortizing Scaling Law Construction Costs

    [https://arxiv.org/abs/2609.05016](https://arxiv.org/abs/2609.05016)

    该论文提出了一种基于贝叶斯优化的高效缩放定律构建框架，通过逐步扩展计算预算并结合代理模型幻想评估来恢复广泛的实验网格，大幅降低了构建缩放定律的计算成本。

    

    缩放定律指导着大型基础模型的训练设计选择，但推导缩放定律需要在超参数、token预算和参数数量上进行穷举式的网格训练，这在计算上非常昂贵。然而，拟合缩放定律只需要跨计算规模的最佳损失前沿，大部分已训练的配置都会被丢弃。我们提出了一个高效的缩放定律构建框架，将数据收集表述为贝叶斯优化问题，并引入了在受限计算预算下比较缩放定律拟合方法的评估指标。我们发现，在采集过程中逐步扩展计算预算（类似于实践中按计算规模顺序评估配置的方式），可以显著提高恢复效率。随后，通过代理模型幻想评估来增强观察到的配置，可以恢复更广泛的实验网格，从而实现准确的缩放定律拟合。

    arXiv:2609.05016v1 Announce Type: cross  Abstract: Scaling laws guide the design choices for training large foundation models, but deriving them involves training an exhaustive grid over hyperparameters, token budgets, and parameter counts, which is computationally expensive. Fitting a scaling law, however, only requires the best-loss frontier across compute scales, discarding most of the trained configurations. We propose a framework for efficient scaling law construction that formulates data collection as a Bayesian optimization problem, and introduce metrics for comparing scaling law fitting methods under constrained compute budgets. We find that progressively expanding the compute budget during acquisition, mirroring the compute-ordered evaluation of configurations in practice, substantially improves recovery efficiency. Augmenting the observed configurations with surrogate-fantasized evaluations then recovers the broader experimental grid, allowing accurate scaling law fitting wit
    
[^266]: 基于量子信号处理的表示学习

    Representation Learning with Quantum Signal Processing

    [https://arxiv.org/abs/2608.28828](https://arxiv.org/abs/2608.28828)

    该论文将量子信号处理（QSP）确立为表示学习的可解量子模型，精确计算了其量子神经正切核的统计特性，并证明了稀疏数据下完整非线性梯度流收敛于可积标量流以及普遍成立的有限深度速度极限。

    

    表示学习始于训练改变了定义数据间相似性的特征之时，而冻结核模型只能对固定几何进行重新加权。我们建立了量子信号处理（QSP）作为表示学习机制的可解量子模型。在任意深度下，我们计算了其量子神经正切核的精确均值与方差，揭示了一种依赖于输入的角几何结构，即使在底层幺正算子趋近Haar随机性时，其对角线仍保持非自平均特性。我们还为完整的非线性梯度流证明了稀疏数据保证，无需冻结或对核进行系综平均：实际动力学收敛于一个具有时间依赖核闭包的可积标量流，并给出显式收敛时间。有限深度速度极限对每个数据集和轨迹均成立。在更高的数据密度下，数值结果表明存在超越标量和冻结核描述的耦合演化。

    arXiv:2608.28828v1 Announce Type: cross  Abstract: Representation learning begins when training changes the features that define similarity between data. A frozen-kernel model only reweights a fixed geometry. We establish quantum signal processing (QSP) as a solvable quantum model of the representation-learning regime. At arbitrary depth, we compute the exact mean and variance of its quantum neural tangent kernel, revealing an input-dependent angular geometry whose diagonal remains non-self-averaging even when the underlying unitary approaches Haar randomness. We also prove a sparse-data guarantee for the full nonlinear gradient flow without freezing or ensemble-averaging the kernel: the realized dynamics converges to an integrable scalar flow with a time-dependent kernel closure and explicit convergence times. A finite-depth speed limit holds for every data set and trajectory. At higher data density, numerical results show coupled evolution beyond both the scalar and frozen-kernel des
    
[^267]: 蒲公英：用于行星动力学神经模拟的球形Flower架构

    Dandelion: A Spherical Flower for Neural Simulation of Planetary Dynamics

    [https://arxiv.org/abs/2608.27521](https://arxiv.org/abs/2608.27521)

    提出Dandelion——基于warp的神经PDE求解器Flower的球面版本，通过沿大圆传输特征并在球谐域中实现分层池化，构建了无卷积的球面神经网络架构，用于行星动力学的神经模拟。

    

    许多动力学过程在球面上展开，但默认的科学机器学习架构都是欧几里得式的。将这些架构应用于规则的经纬度网格会产生诸多问题：笛卡尔卷积在高纬度地区发生畸变；傅里叶神经算子中的二维FFT错误地假设了双周期性；视觉Transformer中的笛卡尔位置编码会扭曲球面测地距离。近期的研究正朝着原生球面基元的方向发展，包括球面卷积（如DeepSphere或DISCO）、球面傅里叶神经算子（SFNO）以及测地注意力机制。在此，我们提出Dandelion，它是Flower（一个基于warp的神经偏微分方程求解器）的球面版本。Dandelion的各层预测切平面位移，并沿大圆传输特征。我们通过完全在球谐域中实现分层池化，构建出了类似U-Net的结构。因此该架构不使用卷积：空间混合是通过（摘要在此处被截断）

    arXiv:2608.27521v1 Announce Type: new  Abstract: Many dynamical processes unfold on the sphere but the default scientific machine learning architectures are Euclidean. Applying these architectures on a regular lat-lon grid causes problems: Cartesian convolutions become distorted at high latitude; 2D FFTs in Fourier neural operators incorrectly assume double periodicity; Cartesian positional encodings in ViTs distort spherical geodesic distances. Recent work moves towards natively spherical primitives, including spherical convolutions (e.g., DeepSphere or DISCO), Spherical Fourier Neural Operators (SFNOs), and geodesic attention. Here we propose Dandelion, a spherical version of Flower, a warp-based neural PDE solver. Layers of Dandelion predict a tangent-plane displacement and transport features along great circles. We obtain a U-Net-like structure by implementing hierarchical pooling entirely in the spherical-harmonic domain. There are thus no convolutions: spatial mixing is achieved 
    
[^268]: 边际覆盖信用减少并行状态熵优化中的冗余探索

    Marginal Coverage Credit Reduces Redundant Exploration in Parallel State-Entropy Optimization

    [https://arxiv.org/abs/2608.27507](https://arxiv.org/abs/2608.27507)

    本文提出MCC-PGPSE方法，通过结合留一策略覆盖与状态所有者专业化的边际覆盖信用来重新分配辅助内在奖励，从而减少并行状态熵优化中的冗余探索并促进互补性状态覆盖。

    

    并行状态熵最大化的策略梯度方法（PGPSE）通过在相同环境的多个副本中训练独立参数化的策略来扩展状态空间覆盖。然而，其汇总的团队熵分数仅衡量集体探索，无法识别对非冗余覆盖有贡献的策略。我们提出了针对PGPSE的边际覆盖信用方法（MCC-PGPSE），该方法将留一策略覆盖与状态所有者专业化相结合，以估计特定策略的信用。MCC-PGPSE保留了PGPSE的汇总目标，并根据这些信用重新分配非负的辅助内在奖励，同时不改变奖励的总质量。这种重新分配旨在抑制冗余访问并促进互补性覆盖。我们在受控环境、七个公开的离散状态基准以及原始PGPSE协议中具有代表性的Room和Maze设置上对MCC-PGPSE进行了评估。在……

    arXiv:2608.27507v1 Announce Type: new  Abstract: Policy Gradient for Parallel State Entropy maximization (PGPSE) expands state-space coverage by training independently parameterized policies in replicated copies of the same environment. However, its pooled team-entropy score measures only collective exploration and cannot identify policies that contribute non-redundant coverage. We introduce Marginal Coverage Credit for PGPSE (MCC-PGPSE), which combines leave-one-policy-out coverage with state-owner specialization to estimate policy-specific credit. MCC-PGPSE preserves PGPSE's pooled objective and redistributes non-negative auxiliary intrinsic rewards according to these credits without changing their total mass. This redistribution is designed to discourage redundant visitation and promote complementary coverage. We evaluated MCC-PGPSE in controlled environments, seven public discrete-state benchmarks, and representative Room and Maze settings from the original PGPSE protocol. Across a
    
[^269]: 为什么基于机器学习的咳嗽模型无法泛化：针对结核病筛查的系统性跨数据集评估

    Why ML-based cough models do not generalize: a systematic cross-dataset evaluation for tuberculosis screening

    [https://arxiv.org/abs/2608.25846](https://arxiv.org/abs/2608.25846)

    本研究通过系统跨数据集评估发现，基于咳嗽的机器学习结核病分类器无法泛化，其性能受录音设备和数据集特异性影响，而非疾病本身，提示需谨慎对待此类模型的实际应用。

    

    arXiv:2608.25846v1 公告类型：交叉 摘要：咳嗽声学特征对于非侵入性结核病（TB）筛查具有潜力，但机器学习（ML）模型捕捉的是疾病相关声学特征还是数据采集伪影，这一问题仍未解决。我们在三个独立数据集上评估了经典机器学习和深度学习（DL）咳嗽基结核病分类器的跨数据集泛化能力。尽管数据集内性能中等（ROC-AUC最高达$0.755 \pm 0.056$），但两种流程均无法泛化，外部性能经常低于0.6，表明可能存在数据局限性。我们进一步观察到，音频表征按录音设备和数据集组织，而非按结核病状态组织；预测的结核病概率在CODA中追踪国家层面的患病率；设备不匹配会降低迁移性能，而设备多样化的训练则能改善迁移。此外，临床变量基线模型泛化更一致（ROC-AUC $0.655 - 0.711$），表明采集特异性因素占主导。

    arXiv:2608.25846v1 Announce Type: cross  Abstract: Cough acoustics are promising for non-invasive tuberculosis (TB) screening, yet whether machine learning (ML) models capture disease-related acoustics or artifacts of data collection remains unresolved. We evaluated the cross-dataset generalizability of classical ML and deep learning (DL) cough-based TB classifiers across three independent datasets. Despite moderate within-dataset performance (ROC-AUC up to $0.755 \pm 0.056$), both pipelines fail to generalize, with external performance frequently below 0.6, indicating a possible limitation of the data. We further observed audio representations are organized by recording device and dataset rather than TB status, predicted TB probability tracks country-level prevalence in CODA, and device mismatch degrades transfer while device-diverse training improves it. Additionally, a clinical-variable baseline generalizes more consistently (ROC-AUC $0.655 - 0.711$), indicating acquisition-specific
    
[^270]: 公平感知的测试时提示调优

    Fairness-Aware Test-Time Prompt Tuning

    [https://arxiv.org/abs/2608.25707](https://arxiv.org/abs/2608.25707)

    本文提出了一种公平感知的测试时提示调优方法（FairTPT），通过软提示调优在分布偏移下同时优化目标熵和虚假熵，以改善视觉-语言模型在子群体偏移中的公平性。

    

    arXiv:2608.25707v1 公告类型：新  摘要：视觉-语言模型在多模态理解方面展现出了卓越的能力，并越来越多地被用于经济和实际部署限制禁止重新训练或微调的关键应用中。然而，这些模型也可能表现出系统性偏差，对受保护的人口群体产生不成比例的影响，而现有的解决这些偏差的方法需要大量模型重新训练和访问人口统计属性。显然需要开发测试时适应（TTA）方法，以在分布偏移下改善预训练模型的公平性特征。在本文中，我们评估了在子群体偏移下，情景式TTA如何影响CLIP分类中的公平性，并开发了FairTPT，一种新颖的公平感知情景式TTA方法，通过软提示调优，同时最小化目标边际熵并最大化虚假边际熵。我们发现，标准情景式TTA方法...

    arXiv:2608.25707v1 Announce Type: new  Abstract: Vision-language models have displayed remarkable capabilities in multi-modal understanding and are increasingly used in critical applications where economic and practical deployment constraints prohibit re-training or fine-tuning. However, these models can also exhibit systematic biases that disproportionately affect protected demographic groups and existing approaches to addressing these biases require extensive model retraining and access to demographic attributes. There is a clear need to develop test-time adaptation (TTA) approaches that improve the fairness characteristics of pretrained models under distributional shift. In this paper, we evaluate how episodic TTA affects fairness in CLIP classification under subpopulation shifts and develop FairTPT, a novel fairness-aware episodic TTA method that jointly minimizes target marginal entropy while maximizing spurious marginal entropy through soft-prompt tuning. We find that standard ep
    
[^271]: 基于有限元法的物理约束电磁线圈设计代理建模与优化框架

    A FEM-Based Surrogate Modelling and Optimization Framework for Physics-Constrained Electromagnetic Coil Design

    [https://arxiv.org/abs/2608.18903](https://arxiv.org/abs/2608.18903)

    本文提出了一种结合有限元模型与高斯过程代理的优化框架，在物理约束下实现了对电磁线圈设计的高效贝叶斯优化，并揭示了不同优化器在不同评估预算下的性能差异。

    

    arXiv:2608.18903v1 公告类型：新  摘要：本研究评估了在几何、制造以及铁芯和铜质量分离约束下，对七参数电流激励线圈-铁芯基准问题进行代理辅助优化的效果。一个Python-MPh-COMSOL工作流程将二维轴对称有限元法（FEM）模型与Matern 5/2高斯过程（GP）概率代理模型相结合。这里，“物理约束”指的是由控制方程FEM模型评估的设计问题，并受显式物理、几何、制造和材料分配约束的限制；它并不指物理信息GP架构。顺序贝叶斯优化（BO）使用期望改进（EI）对候选方案进行排名，每个报告的当前最优解均由FEM验证。五次配对运行表明，优化器的排名取决于可用的FEM评估预算：在较小的连续预算下，EI-BO能快速改进；在最早检查点，COBYLA表现更强；而BOBY（截断）...

    arXiv:2608.18903v1 Announce Type: new  Abstract: This work evaluates surrogate-assisted optimization of a seven-parameter current-excited coil--core benchmark subject to geometric, manufacturing, and separate core and copper mass constraints. A Python--MPh--COMSOL workflow couples a two-dimensional axisymmetric finite-element method (FEM) model to a Matern 5/2 Gaussian-process (GP) probabilistic surrogate. Here, physics-constrained denotes a design problem evaluated by a governing-equation FEM model and restricted by explicit physical, geometric, manufacturing, and material-allocation constraints; it does not denote a physics-informed GP architecture. Sequential Bayesian optimization (BO) ranks candidates using expected improvement (EI), and every reported incumbent is verified by FEM. Five paired runs show that optimizer ranking depends on the available FEM-evaluation budget: EI--BO improves rapidly at small continuation budgets, COBYLA is stronger at the earliest checkpoint, and BOBY
    
[^272]: 采样运气伪装成分配收益：审计神经组合优化中的测试时预算分配

    Sampling Luck Masquerades as Allocation Gain: Auditing Test-Time Budget Allocation for Neural Combinatorial Optimization

    [https://arxiv.org/abs/2608.13087](https://arxiv.org/abs/2608.13087)

    本文发现神经组合优化中测试时预算分配的报告增益主要源于采样运气偏差，样本外验证表明非均匀分配并无实际收益。

    

    arXiv:2608.13087v1 公告类型：交叉 摘要：神经组合优化（NCO）求解器报告每个实例中多个采样解的最佳结果，且按惯例，每个实例的采样数量相同。固定总预算的非均匀分配是否能带来收益，此前尚未被测量。我们对此进行了测量，并对测量本身进行了审计。首先，在分布内工作负载上，分配提升空间无法被检测到。在均匀TSP-100上，跨三个预训练求解器（POMO、AM、SymNCO），基于相同存储样本计算和评估的oracle分配报告了2.2-2.6%的增益，置信区间排除零；但在样本外测量时，相同增益与零无法区分（0.457、0.015、-0.512百分比）。按照惯例的样本内程序，所有三个求解器都会支持一个已发表的2%级别增益，但该增益并不存在。我们将此偏差与实例级零假设进行校准，在该假设中真实增益按构造为零；在范围内...

    arXiv:2608.13087v1 Announce Type: cross  Abstract: Neural combinatorial optimization (NCO) solvers report the best of many sampled solutions per instance, and the sample count is, by convention, identical for every instance. Whether a non-uniform allocation of a fixed total budget would buy anything has not been measured. We measure it, and we audit the measurement itself.   First, on in-distribution workloads the allocation headroom is not detectable. Across three pretrained solvers (POMO, AM, SymNCO) on uniform TSP-100, an oracle allocation computed and evaluated on the same stored samples reports a 2.2-2.6% gain with intervals excluding zero; measured out of sample the same gain is indistinguishable from zero (0.457, 0.015, -0.512 percent). Following the customary in-sample procedure, all three solvers would have supported a published 2%-level gain that does not exist. We calibrate this bias against an instance-wise null in which the true gain is zero by construction; over the range
    
[^273]: 评估分辨率混淆了模型-大脑RSA中学习规则比较的结果（早期视觉皮层研究）

    Evaluation Resolution Confounds Learning-Rule Comparisons in Model-Brain RSA of Early Visual Cortex

    [https://arxiv.org/abs/2608.12408](https://arxiv.org/abs/2608.12408)

    本文发现，在模型-大脑RSA比较中，评估分辨率的选择会显著影响不同学习规则（如未训练/局部训练与反向传播）在早期视觉皮层上的表现排名，高分辨率评估下反向传播的优势更明显，提示现有比较结果可能受分辨率混淆影响。

    

    表征相似性分析（RSA）越来越多地被用于探究哪些学习规则能使卷积网络产生类似大脑的表征。由于生物学上合理的学习规则（如反馈对齐、预测编码和STDP）难以扩展，包含这些规则的研究通常在小图像（典型为32x32的CIFAR）上训练小型网络，然后将其与以更高分辨率建模的大脑反应进行比较。我们发现，在这种设置下的一个常见结果——即未训练或局部训练的网络在早期视觉皮层上能与反向传播相媲美甚至超越——强烈依赖于网络评估时的分辨率。未训练网络与反向传播训练网络之间的V1差距从32像素训练分辨率下的-0.001 ± 0.007扩大到224像素下的+0.044 ± 0.006，并在六种分辨率下单调增长（n=5个种子）。这一结果在人类fMRI数据中成立，并且在方向上也在单种子猕猴电生理数据中成立，沿着t...

    arXiv:2608.12408v1 Announce Type: cross  Abstract: Representational similarity analysis (RSA) is increasingly used to ask which learning rules give convolutional networks brain-like representations. Because biologically plausible rules such as feedback alignment, predictive coding and STDP do not scale, studies that include them train small networks on small images (typically 32x32 CIFAR) and then compare them to brain responses modeled at much higher resolution. We find that a common result in this setting, that untrained or locally trained networks rival or beat backpropagation at early visual cortex, depends strongly on the resolution at which the network is evaluated. The V1 gap between an untrained network and a backpropagation-trained one widens from -0.001 +/- 0.007 at the 32px training resolution to +0.044 +/- 0.006 at 224px, growing monotonically across six resolutions (n=5 seeds). It holds in human fMRI and, directionally, in single-seed macaque electrophysiology, along the t
    
[^274]: 一个冻结的模拟器是不够的：多智能体强化学习中的模拟器崩溃

    One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL

    [https://arxiv.org/abs/2608.12253](https://arxiv.org/abs/2608.12253)

    该论文揭示了多智能体强化学习中使用单一冻结模拟器会导致策略泛化失败，并提出推理时的言语化采样和训练时的协同训练两种方法来解决模拟器崩溃问题。

    

    用于人机交互的多智能体强化学习通常依赖单一大型语言模型来模拟用户行为。我们表明这种方法系统性无法泛化，并将失败追溯到模拟器崩溃：由于模拟器语言模型存在模式崩溃，针对其训练的策略会过度拟合到利用模拟器主导模式的狭窄策略，而这种策略在未见过的模拟器和真实用户中迁移效果不佳。我们在理论上形式化了这种崩溃，并提出了两种互补的解决方案，一种在推理时，一种在训练时。推理时的解决方案，即言语化采样，通过从言语化响应分布中采样来拓宽模拟器的行为，减少模式崩溃。训练时的解决方案，即协同训练，与一组可训练的模拟器种群联合优化策略，防止其过度拟合任何单一模拟器的模式。我们进行了验证...

    arXiv:2608.12253v1 Announce Type: cross  Abstract: Multi-agent reinforcement learning for human-AI interaction typically relies on a single large language model to simulate user behavior. We show that this approach systematically fails to generalize, and trace the failure to simulator collapse: because the simulator LLM is mode-collapsed, an LLM policy trained against it overfits to narrow strategies that exploit the simulator's dominant mode, and such a policy transfers poorly to unseen simulators and real users. We formalize this collapse theoretically and propose two complementary solutions, one at inference time and one at training time. The inference-time solution, Verbalized Sampling, broadens the simulator's behavior by sampling from a verbalized response distribution, reducing mode collapse. The training-time solution, Co-Training, jointly optimizes the policy against a population of trainable simulators, preventing it from overfitting to any single simulator's mode. We validat
    
[^275]: 视觉模型能读懂雷达显示屏吗？——关于雷达图像用于空中交通复杂性估计的可行性研究

    Can Vision Models Read the Radar Display? On the Feasibility of Radar Imagery for Air Traffic Complexity Estimation

    [https://arxiv.org/abs/2608.11810](https://arxiv.org/abs/2608.11810)

    本文探索了利用雷达图像作为视觉变换器输入来估计空中交通复杂性的可行性，通过编码飞机状态信息验证了深度学习模型能捕捉稀疏雷达图像中运行相关的复杂性变化。

    

    arXiv:2608.11810v1 公告类型：交叉 摘要：空中交通管制员通过雷达显示屏感知交通复杂性，这表明基于相同图像运行的计算机视觉模型可能为建模管制员感知的复杂性提供一种自然的架构；然而，雷达图像作为深度学习视觉模型的可行输入格式仍不明确。与自然图像不同，雷达图像极其稀疏且高度自相似，主要由黑色背景和少数视觉上相同的飞机斑点组成，而飞机位置的微小变化可能显著改变扇区级别的复杂性。为测试视觉模型能否捕捉这些运行上重要的差异，我们将每个交通情景编码为一个位置图像，并辅以五个通道表示飞机状态变量（包括航向、速度和高度），然后训练一个视觉变换器（ViT）来回归从成对几何关系中导出的四个内在复杂性分量。

    arXiv:2608.11810v1 Announce Type: cross  Abstract: Air traffic controllers perceive traffic complexity through the radar display, suggesting that a computer vision model operating on the same imagery may provide a natural architecture for modeling controller-perceived complexity; however, whether radar imagery is a viable input format for deep learning vision models remains unclear. Unlike natural images, radar images are extremely sparse and self-similar, consisting primarily of a black background and a few visually identical aircraft blobs, while small changes in aircraft positions can substantially alter sector-level complexity. To test whether a vision model can capture these operationally important differences, we encode each traffic situation as a position image supplemented by five channels representing aircraft state variables, including heading, speed, and altitude, and train a Vision Transformer (ViT) to regress four intrinsic complexity components derived from pairwise geome
    
[^276]: 基于可观测语义图像接口与层级生成器证据对齐的生成式语义分割

    Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment

    [https://arxiv.org/abs/2608.11537](https://arxiv.org/abs/2608.11537)

    提出了一种通过可观测语义图像接口和层级生成器证据对齐的生成式语义分割方法，有效解决了颜色漂移和边界混合问题。

    

    生成式语义分割将结构化预测呈现为图像，但直接的颜色解码容易受到颜色漂移和边界混合的影响，而预测独立输出分布的潜在特征解码器可能将渲染图像降级为中间可视化。我们提出了Semantic Prism，一个具有确定性推理的条件语义图像生成与细化框架。一个基于扩散蒸馏的单步生成器渲染语义RGB图像；渲染颜色到固定类颜色码本的逐像素距离定义了一个显式的概率接口。层级生成器证据对齐在空间上对齐多层级生成器特征，并使用零初始化输出投影在接口对数空间中预测加性残差，保留图像定义的接口作为最终分布的参考。该接口和细化分布进一步支持Conte（后续内容）。

    arXiv:2608.11537v1 Announce Type: cross  Abstract: Generative semantic segmentation exposes structured predictions as images, but direct color decoding is susceptible to color drift and boundary mixing, whereas latent-feature decoders that predict a separate output distribution may relegate the rendered image to an intermediate visualization. We present Semantic Prism, a conditional semantic-image generation-and-refinement framework with deterministic inference. A diffusion-distilled one-step generator renders a semantic RGB image; per-pixel distances from the rendered colors to a fixed class-color codebook define an explicit probabilistic interface. Hierarchical Generator Evidence Alignment spatially aligns multi-level generator features and uses a zero-initialized output projection to predict an additive residual in the interface logit space, retaining the image-defined interface as the reference for the final distribution. The interface and refined distributions further enable Conte
    
[^277]: 在Boltz-1的躯干-扩散边界上探测与引导生物学信息

    Probing and steering biology across Boltz-1s trunk-diffusion boundary

    [https://arxiv.org/abs/2608.11475](https://arxiv.org/abs/2608.11475)

    该研究揭示了Boltz-1模型中，躯干编码的序列化学信息在扩散模块中被衰减，而几何结构信息保持传递，并通过干预躯干表征实现了对结构生成的引导。

    

    arXiv:2608.11475v1 公告类型：交叉 摘要：AlphaFold3类结构预测器将表征躯干（处理序列和上下文）与扩散模块（生成原子坐标）配对。生物学信息在跨越这一架构边界时如何变化仍知之甚少。我们使用线性探针、稀疏自编码器（SAEs）和因果干预，分析了Boltz-1中Pairformer躯干和扩散模块的逐残基激活。从躯干中，几何信息（二级结构、无序性）和序列化学信息（氨基酸身份、信号肽、二硫键注释）均可线性解码。在扩散模块中，两者出现分化。二级结构几乎不变地传递，而序列化学信息则被强烈衰减。随后，我们测试了可解码的方向是否能引导模型，对最终条件化扩散模块的躯干单一表征进行干预。螺旋和卷曲方向发生变化。

    arXiv:2608.11475v1 Announce Type: cross  Abstract: AlphaFold3-class structure predictors pair a representational trunk, which processes sequence and context, with a diffusion module, which generates atomic coordinates. How biological information changes as it crosses this architectural boundary remains poorly understood. We analyze per-residue activations from the Pairformer trunk and diffusion module of Boltz-1 using linear probes, sparse autoencoders (SAEs), and causal interventions. From the trunk, both geometry (secondary structure, disorder) and sequence chemistry (amino-acid identity, signal peptides, disulfide-bond annotations) are linearly decodable. In the diffusion module, the two diverge. Secondary structure transfers essentially unchanged, whereas sequence chemistry is strongly attenuated. We then test whether decodable directions can steer the model, intervening on the final trunk single representation that conditions the diffusion module. Helix and coil directions change 
    
[^278]: MAGIC-SSCIL：面向半监督类增量学习的流形锚定与几何增量校准

    MAGIC-SSCIL: Manifold Anchoring and Geometric Incremental Calibration for Semi-Supervised Class Incremental Learning

    [https://arxiv.org/abs/2608.07586](https://arxiv.org/abs/2608.07586)

    MAGIC通过在冻结特征的k近邻图上进行标签传播，并将传播置信度同时用于无标签损失加权和软加权几何校准的类别统计量构建，解决了极端少标签场景下半监督类增量学习中伪标签不可靠、回放统计量稀疏噪声的问题。

    

    无样本回放的半监督类增量学习要求模型在仅有少量有标签图像和大量无标签图像的情况下学习新类别，同时不存储历史数据。一种常见做法是从存储的高斯统计量中回放旧类，并用置信度阈值对无标签数据进行伪标签标注。然而，当每个类别只有一两个标签时，模型的置信度难以可靠地判断哪些伪标签是正确的，因此回放所依赖的统计量往往建立在极少或含噪声的样本之上。我们提出了MAGIC（流形锚定与几何增量校准），该方法在冻结特征的k近邻图上进行标签传播，并将传播后的置信度作为无标签损失的权重；同时通过软加权几何校准，将这些置信度融入旧类幻影特征回放所依据的类别统计量中。在CIFAR-100、CUB-200和ImageNet-R数据集上使用冻结的ResNet-18，并在1%、5%和10%的标签比例下，我们比较了MAGI……

    arXiv:2608.07586v2 Announce Type: replace-cross  Abstract: Exemplar-free semi-supervised class-incremental learning (SSCIL) asks a model to learn new classes from a few labeled and many unlabeled images without storing past data. A common recipe replays old classes from stored Gaussian statistics and pseudo-labels unlabeled data with a confidence threshold. With one or two labels per class, however, the model's confidence is a poor guide to which pseudo-labels are right, so the statistics that replay depends on rest on few or noisy samples. We propose MAGIC (Manifold Anchoring and Geometric Incremental Calibration), which propagates labels over a k-nearest-neighbor graph of frozen features and uses the propagated confidences as weights in the unlabeled loss and, through Soft-Weighted Geometry Calibration (SWGC), in the class statistics from which phantom features replay old classes. With a frozen ResNet-18 on CIFAR-100, CUB-200, and ImageNet-R at 1%, 5%, and 10% labels, we compare MAGI
    
[^279]: 测试时覆盖率：面向部署感知学习的测试条件化数据筛选

    Test-Time Coverage: Test-Conditioned Data Curation for Deployment-Aware Learning

    [https://arxiv.org/abs/2607.22697](https://arxiv.org/abs/2607.22697)

    该论文提出TTCov方法，在训练前利用测试侧信息构建由LLM原子命题及其频率组成的知识图集，将部署条件化的数据筛选分解为覆盖率与分布两部分，从而直接优化训练数据与部署测试分布的匹配度。

    

    已部署的AI系统通常从广泛的候选数据池中进行训练，因此需要针对部署测试分布进行数据筛选。然而，标准的数据筛选方法评估的是训练侧的准则，而非直接优化与部署分布的匹配度。我们提出了TTCov（Test-Time Coverage，测试时覆盖率），这是一种数据层面的测试条件化筛选方法，它在训练之前使用测试侧的信息，而不是在推理时更新模型权重。TTCov将部署条件化的筛选分解为覆盖率与分布两个部分。为了表示覆盖率，它构建了一个任务图集，即一组基于大语言模型（LLM）的原子命题（APs），用于描述与部署相关的概念；这些命题由开放的任务知识作为种子，并通过从无标注部署样本中提取的未匹配原子命题进行扩展。为了表示分布，它以频率来实例化匹配的部署原子命题，从而构建出一个知识图集，将部署分布进行可操作化表示……

    arXiv:2607.22697v2 Announce Type: replace  Abstract: Deployed AI systems are often trained from broad candidate data pools, necessitating data curation towards the deployment test distribution. However, standard data curation methods score training-side criteria rather than directly optimizing deployment match. We introduce TTCov (Test-Time Coverage), a data-level test-conditioned curation method that uses test-side information before training instead of updating model weights at inference. TTCov decomposes deployment-conditioned curation into coverage and distribution. To represent coverage, it builds a task Atlas, a collection of LLM-based atomic propositions (APs) describing deployment-relevant concepts, seeded from open task knowledge and expanded with unmatched APs extracted from unlabeled deployment samples. To represent distribution, it instantiates the matched deployment APs with their frequencies, yielding a Knowledge Atlas (K-Atlas) that operationalizes the deployment distrib
    
[^280]: 随机多梯度 descent 的改进收敛速率——填补差距：由 AI 完成的证明

    Improved Convergence Rates for Stochastic Multi-Gradient Descent which Close the Gap: A Proof by AI

    [https://arxiv.org/abs/2607.18174](https://arxiv.org/abs/2607.18174)

    本文通过利用帕累托平稳性度量的 Lipschitz 连续性（而非 (1/2)-Hölder 连续性），证明了随机多梯度下降（SMG）在常数步长和线性增长批量下可将平方帕累托平稳性度量的收敛速率从此前的 $\tilde{O}(T^{-1/4})$ 提升至 $\tilde{O}(T^{-1})$，且无需有界梯度假设。

    

    对于光滑非凸的随机多目标优化问题，随机多梯度下降（SMG）从随机梯度中计算各目标函数的近似最速公共下降方向。在标准假设下，本文针对 SMG 在平方帕累托平稳性（PS）度量方面建立了一个新的收敛速率：采用常数步长和线性增长的迷你批量，经过 $T$ 次迭代后，算法输出处的期望平方经验 PS 度量为 $\tilde{O}(T^{-1})$。这改进了 Chen 等人（JMLR, 2024）在线性增长批量下得到的 $\tilde{O}(T^{-1/4})$ 的界，且不需要有界梯度假设。其中 $\tilde{O}(\cdot)$ 隐藏了对数因子。收敛速率提升的关键在于利用由多梯度下降算法（MGDA）方向的范数所定义的 PS 度量的 Lipschitz 连续性，而非 MGDA 方向的 (1/2)-Hölder 连续性。

    arXiv:2607.18174v2 Announce Type: cross  Abstract: For smooth nonconvex stochastic multi-objective optimization, stochastic multi-gradient descent (SMG) computes an approximate steepest common descent direction of the objectives from stochastic gradients. Under standard assumptions, this note establishes a new convergence rate for SMG in terms of the squared Pareto-stationarity (PS) measure. With a constant stepsize and linearly growing mini-batches, the expected squared empirical PS measure at the algorithm's output is $\tilde{O}(T^{-1})$ after $T$ iterations. This improves on the $\tilde{O}(T^{-1/4})$ bound obtained by Chen et al. (JMLR, 2024) for linearly growing batches, without requiring bounded gradients. Here, $\tilde{O}(\cdot)$ suppresses logarithmic factors. The key to the rate improvement is to exploit the Lipschitz continuity of the PS measure, defined by the norm of the multi-gradient descent algorithm (MGDA) direction, rather than the $(1/2)$-H\"older continuity of the MGD
    
[^281]: OpenMHC：加速可穿戴基础模型的科学发展

    OpenMHC: Accelerating the Science of Wearable Foundation Models

    [https://arxiv.org/abs/2607.16235](https://arxiv.org/abs/2607.16235)

    该论文发布了OpenMHC——目前面向研究人员开放的最大、最全面的可穿戴健康数据集（涵盖11,894名参与者、超过6700万小时记录、最多19个传感器通道和170个健康行为变量），并配套提供可穿戴基础模型的开源实现与模型权重，以推动可穿戴健康领域的开放科学。

    

    移动和可穿戴设备为持续、被动式的健康监测和主动健康指导提供了前所未有的机会。然而，目前最大的可穿戴数据集均为专有数据，基于这些数据训练的领先可穿戴基础模型很少开放权重或包含可复现的训练代码。为了加速可穿戴健康领域的开放科学，我们推出了OpenMyHeartCounts（OpenMHC），这是面向合格研究人员开放的最大、最全面的可穿戴健康数据集，同时提供近期可穿戴基础模型的开源实现和权重。OpenMHC源自通过My Heart Counts研究应用收集的十余年数据，包含来自11,894名参与者的超过6700万参与者小时的可穿戴记录，涵盖多达19个传感器通道（如步数、心率、睡眠、锻炼）以及多达170个健康、生活方式、情绪和行为变量。此外，我们还建立了一个……（原文摘要在此处截断）

    arXiv:2607.16235v4 Announce Type: replace-cross  Abstract: Mobile and wearable devices offer an unprecedented opportunity for continuous, passive health monitoring and active health coaching. However, the largest wearable datasets are proprietary, and leading wearable foundation models trained on them are rarely open weight or include reproducible training code. To accelerate open science in wearable health, we introduce OpenMyHeartCounts (OpenMHC), the largest and most comprehensive wearable health dataset broadly accessible to qualified researchers, alongside open-source implementations and weights for recent wearable foundation models. Derived from over a decade of data collected through the My Heart Counts study app, OpenMHC comprises over 67M participant-hours of wearable recordings across up to 19 sensor channels (e.g., step count, heart rate, sleep, workouts) and up to 170 health, lifestyle, mood, and behavioral variables from 11,894 participants. Furthermore, we establish a ben
    
[^282]: 将通用车辆模型适配用于跨地形高速模型预测控制（MPC）

    Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains

    [https://arxiv.org/abs/2607.13319](https://arxiv.org/abs/2607.13319)

    OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。

    

    高速越野自动驾驶需要对目标车辆进行精确的闭环控制，同时在多变地形上保持鲁棒性。近期的前向运动动力学（FKD）预测基础模型展现出一条有前景的路径：从通用模型出发，将其专用于目标平台。然而，有效的专业化仍然具有挑战性，因为它通常需要大量真实世界数据，而且适配到某一环境的模型仍可能对特定地形或驾驶模式过拟合。我们提出了OptCar（优化汽车），这是一种弥合通用FKD模型到专用FKD模型差距的方法，在为特定车辆优化性能的同时保留了跨地形泛化能力。OptCar引入了一种基于transformer的FKD架构，利用FiLM将多步预测条件化于单个动力学上下文token，该token概括了近期的状态-动作历史。随后，该方法利用有限的真实世界数据对通用模型进行专业化（摘要在此处截断）。

    arXiv:2607.13319v2 Announce Type: replace-cross  Abstract: High-speed off-road autonomy requires precise closed-loop control for a target vehicle while remaining robust across changing terrains. Recent forward kinodynamic (FKD) prediction foundation models suggest a promising path, starting from a generalist model and specializing it to the target platform. However, effective specialization remains challenging, as it often requires substantial real-world data, and models adapted to one setting can still overfit to specific terrains or driving regimes. We present OptCar (Optimized Car), a recipe for bridging the gap from generalist to specialist FKD models that preserves cross-terrain generalization while optimizing performance for a specific vehicle. OptCar introduces a transformer FKD architecture that uses FiLM to condition multi-step predictions on a single dynamics context token summarizing recent state-action history. It then specializes the generalist model using limited real-wor
    
[^283]: 统计对手：干净视觉数据集中自然存在的后门样对抗特征

    Statistical Adversaries: Natural Backdoor-like Adversarial Features in Clean Vision Datasets

    [https://arxiv.org/abs/2607.05516](https://arxiv.org/abs/2607.05516)

    本论文发现干净视觉数据集中自然存在的统计信号可作为后门样触发器，揭示了数据集结构和分布导致的模型漏洞，而非恶意插入。

    

    arXiv:2607.05516v2 公告类型：替换-交叉 摘要：针对特定模型的对抗攻击已被广泛研究。我们研究了一种不同的故障模式：视觉数据中自然存在的统计信号，它们可以像后门触发器一样发挥作用，而无需恶意插入。我们将这些信号称为统计对手。我们分析ImageNet，以找到与特定标签强烈相关的模式。然后，我们使用统计控制方法从候选信号中去除随机相关性。最后，我们证明这些信号直接且可预测地改变模型预测。这些统计对手比通用扰动更具针对性，并能跨不同模型架构迁移。这表明某些漏洞是由数据集结构和分布驱动的，而非单一模型的特殊性质。我们得出结论，即使在无投毒的情况下，普通数据集也可能包含可利用的对抗表面，并建议对数据集进行更细致的审查。

    arXiv:2607.05516v2 Announce Type: replace-cross  Abstract: Model-specific adversarial attacks have been extensively studied. We study a different failure mode: naturally occurring statistical signals in vision data that can behave as backdoor-like triggers without being maliciously inserted. We call these signals statistical adversaries. We analyse ImageNet to find patterns that are strongly linked to certain labels. We then use statistical controls to remove random correlations from our candidate signals. Finally, we demonstrate that these signals directly and predictably alter model predictions. These statistical adversaries are more targeted than generic corruptions and transfer across different model architectures. This suggests that some vulnerabilities are driven by dataset structure and distribution rather than a single model's idiosyncrasies. We conclude that ordinary datasets can contain exploitable adversarial surfaces even in the absence of poisoning, and suggest that datase
    
[^284]: 局部化LoRA-MoE：具有自适应路由的低秩专家混合

    Localized LoRA-MoE: Mixture of Low-Rank Experts With Adaptive Routing

    [https://arxiv.org/abs/2607.05114](https://arxiv.org/abs/2607.05114)

    提出Localized LoRA-MoE框架，将局部空间分块与上下文条件自适应专家路由相结合，提供了集中式块级路由和去中心化单元级路由两种架构变体，以解决传统PEFT方法中的梯度干扰和静态拓扑无法适应上下文切换的问题。

    

    参数高效微调（PEFT）方法（如LoRA）通过单一低秩更新来适配大型模型。这类更新容易受到梯度干扰的影响：当训练流中交织着多个操作上下文时，对共享适配器权重的相互冲突的更新会使适配器坍缩成一个缺乏专门化的平均值。虽然近期的方法引入了块级隔离来缓解这一问题，但其拓扑结构是静态的，无法适应上下文切换或局部传感器退化。在这项工作中，我们提出了Localized LoRA-MoE，一个将局部空间分块与上下文条件专家路由相结合的框架。我们提出并评估了两种架构变体：Block-Wise LoRA-MoE（集中式路由），它通过单一的上下文条件门控来调制整个结构网格；以及Cell-Wise LoRA-MoE（去中心化路由），它为结构网格的每个坐标单元分配一个独立的门控。

    arXiv:2607.05114v2 Announce Type: replace-cross  Abstract: Parameter-efficient fine-tuning (PEFT) methods such as LoRA adapt large models through a single low-rank update. Such updates are susceptible to gradient interference: when a training stream interleaves multiple operational contexts, conflicting updates to shared adapter weights collapse the adapter into an unspecialized average. While recent methods introduce block-wise isolation to remedy this issue, their topologies are static and cannot adapt to context switching or localized sensor degradation. In this work, we introduce Localized LoRA-MoE, a framework that combines localized spatial blocking with context-conditioned expert routing. We propose and evaluate two architectural variants: Block-Wise LoRA-MoE (centralized routing), which modulates the entire structural grid through a single context-conditioned gate, and Cell-Wise LoRA-MoE (decentralized routing), which assigns an independent gate to every coordinate cell of the 
    
[^285]: 层次结构上具有数据驱动相互作用势的退火吉布斯测度

    Annealed Gibbs Measures with Data-Driven Interaction Potentials on Hierarchical Structures

    [https://arxiv.org/abs/2606.30064](https://arxiv.org/abs/2606.30064)

    该论文在Cayley树上构造了由训练数据（经验损失）确定耦合函数的退火吉布斯测度，并推导出有限体积联合密度的充要相容性条件（即边界法则的非线性积分方程），证明了在 $k=1$ 时方程存在唯一的归一化正解。

    

    我们在阶数 $k\geq 1$ 的Cayley树（凯莱树）上，针对连续的相互作用参数与二值自旋构造了退火吉布斯测度，其耦合函数由经验损失确定。相互作用参数与自旋是两类不同的随机变量：前者决定边上的耦合，后者表示二值标签。训练数据确定的是耦合函数与联合吉布斯规范，而非自动选出相互作用参数上唯一的吉布斯测度。我们推导了有限体积联合密度的充要相容性条件，这些条件表现为边界法则的非线性积分方程，并刻画了本文所构造的分裂退火吉布斯测度。对于 $k=1$，平移不变方程归结为正的紧积分算子，在所述假设下，这些方程具有唯一的归一化正解。对于 $k=2$，一个显式……（原文摘要在此处被截断）

    arXiv:2606.30064v2 Announce Type: replace  Abstract: We construct annealed Gibbs measures on continuous interaction parameters and binary spins on Cayley trees of order $k\geq 1$, with coupling functions determined by an empirical loss. The interaction parameters and spins are distinct random variables: the former determine edge couplings, while the latter represent binary labels. The training data specify the coupling function and the joint Gibbs specification, rather than automatically selecting a unique Gibbs measure on interaction parameters.   We derive necessary and sufficient compatibility conditions for the finite-volume joint densities. These conditions take the form of nonlinear integral equations for boundary laws and characterize the splitting annealed Gibbs measures constructed here. For $k=1$, the translation-invariant equations reduce to positive compact integral operators. Under the stated hypotheses, they have a unique normalized positive solution. For $k=2$, an explic
    
[^286]: 面向计算机科学逻辑的理论级自动形式化

    Theory-Scale Auto-Formalization of Logics for Computer Science

    [https://arxiv.org/abs/2606.26525](https://arxiv.org/abs/2606.26525)

    本文提出了LCS-Bench，一个基于《计算机科学逻辑》的理论级自动形式化基准，通过半自动化智能体流水线构建，实现了大规模相互依赖逻辑理论的连贯形式化翻译。

    

    arXiv:2606.26525v1 公告类型：新 摘要：自动形式化对于可扩展的形式化验证至关重要，但现有进展主要集中在孤立语句上，而理论级自动形式化——即连贯地翻译数百个相互依赖的定义、引理和定理——由于在一致性、忠实性、可扩展性和正确性方面面临挑战，仍然是一个未解决的问题。在本文中，我们介绍了LCS-Bench，这是一个基于《计算机科学逻辑》的独立理论级基准。LCS-Bench通过一种新颖的半自动化智能体流水线构建，该流水线利用概念图、形式化签名规划、问题追踪、带反例搜索的“sorry”填充，并辅以人类专家的忠实性审查。最终成果涵盖327个教科书条目、超过4,076个Lean声明以及超过85,000行Lean代码。该数据集通过一个数据引擎支持广泛的评估，该引擎自动衍生出五个评估基准轨道。

    arXiv:2606.26525v1 Announce Type: new  Abstract: Auto-formalization is critical for scalable formal verification, but existing progress largely focuses on isolated statements, while theory-scale auto-formalization, which coherently translates hundreds of interdependent definitions, lemmas, and theorems, remains open due to challenges in consistency, faithfulness, scalability, and correctness. In this paper, we introduce LCS-Bench, a stand-alone, theory-scale benchmark based on Logics for Computer Science. LCS-Bench is built through a novel semi-automated agentic pipeline that leverages concept graphs, formal signature planning, issue tracking, sorry-filling with counter-example search, complemented by faithfulness review from human experts. The resulting artifact covers 327 textbook items, over 4,076 Lean declarations, and more than 85K lines of Lean code. The dataset supports broad evaluation through a data engine that automatically derives five tracks of evaluation benchmarks, measur
    
[^287]: 面向基于模型规划的从语言中进行潜在目标预测

    Latent Goal Prediction from Language for Model-Based Planning

    [https://arxiv.org/abs/2606.20627](https://arxiv.org/abs/2606.20627)

    LAGO是一个分层世界模型，通过单一预测器和单一回归目标，将语言指令接地为潜在子目标序列，从而在潜在空间中实现语言引导的基于模型的规划。

    

    arXiv:2606.20627v2 公告类型：替换 摘要：联合嵌入预测架构（JEPA）使智能体能够通过想象候选动作的结果在潜在空间中进行规划，然而任务规范仍然是一个瓶颈。视觉目标能提供精确的局部梯度，但缺乏远距离的引导；而语言虽然灵活，却受限于嘈杂的跨模态对齐，或依赖于独立的大型生成模型。我们提出了LAGO（从语言中进行潜在目标预测），这是一个分层世界模型，其中单个预测器既预测动作条件下的动力学，又将语言指令接地为中间潜在子目标序列，并通过在共享潜在空间上的单一回归目标来训练这两种模式。在每个规划步骤中，LAGO根据语言指令预测一系列潜在子目标，并使用软最小对齐代价来优化动作序列，该代价奖励智能体接近子目标，而不强制执行僵化的路径。子目标会被重新预测…

    arXiv:2606.20627v2 Announce Type: replace  Abstract: Joint-Embedding Predictive Architectures (JEPAs) enable agents to plan in latent space by imagining the outcomes of candidate actions, yet task specification remains a bottleneck. Visual targets provide precise local gradients but poor distant guidance, while language is flexible yet limited by noisy cross-modal alignment or dependence on distinct large generative models. We introduce LAGO (Latent Goal Prediction from Language), a hierarchical world model in which a single predictor both forecasts action-conditioned dynamics and grounds language instructions as sequences of intermediate latent subgoals, training both modes with a single regression objective over a shared latent space. At each planning step, LAGO predicts a sequence of latent subgoals from a language instruction and optimizes an action sequence using a soft-minimum alignment cost that rewards subgoal proximity without enforcing a rigid path. Subgoals are repredicted a
    
[^288]: 离线基于偏好的轨迹评估

    Offline Preference-Based Trajectory Evaluation

    [https://arxiv.org/abs/2606.17541](https://arxiv.org/abs/2606.17541)

    该论文提出基于偏好的轨迹评估方法，通过轨迹间的时序偏好比较取代二元成功指标，将基准测试中的平局率从约75%降至约35%，显著提升了评估的区分能力、排名稳定性和数据效率，并揭示基准测试饱和可能源于度量设计而非仅仅是数据或问题难度问题。

    

    智能体系统的离线评估通常将轨迹压缩为二元的终端成功值，从而丢弃了关于部分进展的信息。这进而导致了普遍的平局现象，通过降低有效样本量并削弱区分不同系统的能力而造成统计上的低效。我们提出了基于偏好的轨迹评估，这是一种评估度量方法，通过利用回报时间剖面之间的时序偏好来直接比较成对的轨迹。我们发现，在多样化的智能体和交互式基准测试中，基于成功的指标在大约75%的实例上产生平局比较，而轨迹感知的偏好将平局率降至约35%，从而提高了区分能力、排名稳定性和数据效率。我们的结果表明，基准测试的饱和——通常被归因于糟糕的数据收集或问题难度——也可能是由评估度量设计本身所导致的。

    arXiv:2606.17541v2 Announce Type: replace-cross  Abstract: Offline evaluation of agentic systems often collapses trajectories into a binary terminal success value, discarding information about partial progress. This, in turn, results in widespread ties that create statistical inefficiency by reducing effective sample size and weakening the ability to distinguish systems. We propose preference-based trajectory evaluation, an evaluation measure that compares pairs of trajectories directly by using temporal preferences between time-to-return profiles. We find that, across diverse agentic and interactive benchmarks, success-based metrics result in tied comparisons on roughly 75% of instances, while trajectory-aware preferences reduce ties to roughly 35%, improving discriminative power, ranking stability, and data efficiency. Our results suggest that benchmark saturation, often portrayed as the result of poor data collection or problem difficulty, may also be explained by the design of the 
    
[^289]: FoundCause：一个从观测数据中发现含潜在混杂因素因果关系的基础模型

    FoundCause: A Foundation Model for Causal Discovery with Latent Confounders from Observational Data

    [https://arxiv.org/abs/2606.17516](https://arxiv.org/abs/2606.17516)

    FoundCause是首个仅在合成结构因果模型上训练的摊销式因果发现基础模型，能够通过单次前向传播从独立同分布观测数据中恢复因果图、识别潜在混杂因素并处理缺失数据。

    

    从观测数据中进行因果发现需要在不进行干预的情况下恢复有向关系和潜在混杂因素。我们提出了FoundCause，这是一个摊销式（amortized）因果发现基础模型，仅在合成结构因果模型上训练，能够在单次前向传播中将数据集映射为因果图。一个置换等变的Transformer交替关注样本和变量，以建模变量间的依赖关系和各变量的分布。成对依赖性和不对称性统计信息引导其注意力机制、边解码器和混杂因素模块。解码器将边的存在性与方向分离，而三角细化则融合来自链结构和碰撞结构的证据。可学习的潜在token预测哪些变量对共享隐藏的共同原因，掩码输入表示则用于处理缺失数据。据我们所知，FoundCause是首个针对独立同分布观测数据的摊销式因果发现方法……（摘要原文在此处截断）

    arXiv:2606.17516v2 Announce Type: replace-cross  Abstract: Causal discovery from observational data requires recovering directed relations and latent confounding without interventions. We propose FoundCause, an {\em amortized causal discovery foundation model} trained only on synthetic structural causal models that maps datasets to causal graphs in a single forward pass. A permutation-equivariant transformer alternates attention over samples and variables to model cross-variable dependence and per-variable distributions. Pairwise dependence and asymmetry statistics guide its attention, edge decoder and confounder module. The decoder separates edge existence from direction, while triangular refinement incorporates evidence from chains and colliders. Learnable latent tokens predict which variable pairs share a hidden common cause, and a masked input representation handles missing data. To our knowledge, FoundCause is the first amortized causal discovery approach for i.i.d. observational 
    
[^290]: ICA Lens：非高斯性揭示语言模型中的可解释方向

    ICA Lens: Non-Gaussianity Reveals Interpretable Directions in Language Models

    [https://arxiv.org/abs/2606.11722](https://arxiv.org/abs/2606.11722)

    该论文发现语言模型激活中可解释方向具有非高斯投影分布这一统计特征，并提出可以通过直接搜索非高斯方向来发现可解释特征，而无需依赖稀疏重建等间接学习目标。

    

    在语言模型激活中寻找人类可解释的方向，对于理解和控制模型行为至关重要。这些方向通常通过间接的学习目标（如稀疏重建）来发现，但目前尚不清楚是何种统计结构区分了可解释方向，以及这种结构能否直接用于特征发现。我们发现非高斯性提供了这样一种特征标志：自然文本残差流激活沿稀疏自编码器（SAE）解码器方向的投影分布，明显比沿随机方向的投影分布更加非高斯，其中大多数上下文形成一个集中的背景，而重复出现的模式往往占据分布的一个尾部。这引出了一个自然的问题：如果可解释方向往往具有非高斯的投影分布，我们能否通过直接搜索非高斯方向来发现它们？

    arXiv:2606.11722v3 Announce Type: replace-cross  Abstract: Finding human-interpretable directions in language-model activations is important for understanding and controlling model behavior. These directions are usually discovered through indirect learning objectives such as sparse reconstruction, but it remains unclear what existing statistical structure distinguishes interpretable directions and whether that structure can be used directly for feature discovery. We find that non-Gaussianity provides such a signature: projection distributions of natural-text residual-stream activations along sparse autoencoder (SAE) decoder directions are substantially more non-Gaussian than those along random directions, with most contexts forming a concentrated background and recurring patterns often occupying one tail. This raises a natural question: if interpretable directions tend to have non-Gaussian projection distributions, can we discover them by searching for non-Gaussian directions directly 
    
[^291]: 双向随机投影

    Bidirectional Random Projections

    [https://arxiv.org/abs/2606.10377](https://arxiv.org/abs/2606.10377)

    本文为固定设计下基于双向随机投影的普通最小二乘回归建立了期望超额损失上界，并证明该上界与单向随机投影已有上界之间的差距以约 O(1/n₁) 的速率衰减。

    

    本文在固定设计设置下分析了普通最小二乘（OLS）回归中的双向随机投影。设 $(X, Y) \in \mathbb{R}^{n\times p}\times\mathbb{R}^{n}$ 为一个样本，$W \in \mathbb{R}^{n_1\times n}$ 和 $R \in \mathbb{R}^{p\times p_1}$ 为两个随机投影。我们为基于 $(WXR, WY)$ 构建的 OLS 估计量推导了期望超额损失上界，扩展了已有的针对基于 $(XR, Y)$ 构建的 OLS 估计量的上界。我们的分析表明，我们的上界更大，且两个上界之间的差距以约 $O(1/n_1)$ 的速率衰减。这些发现得到了数值结果的证实。

    arXiv:2606.10377v2 Announce Type: replace-cross  Abstract: This paper analyzes bidirectional random projections for ordinary least squares (OLS) regression under the fixed design setting. Let $(X, Y) \in \mathbb{R}^{n\times p}\times\mathbb{R}^{n}$ be a sample and $W \in \mathbb{R}^{n_1\times n}$, $R \in \mathbb{R}^{p\times p_1}$ be two random projections. We develop an expected excess loss bound for the OLS estimator built on $(WXR, WY)$, extending an established bound for the OLS estimator built on $(XR, Y)$. Our analysis shows that our bound is larger and the gap between the two bounds decays at a rate of approximately $O(1/n_1)$. These findings are confirmed by numerical results.
    
[^292]: Graph Cascades：基于传染扩散的介观重连方法，用于结构感知的图机器学习

    Graph Cascades: Contagion-Based Mesoscopic Rewiring for Structure-Aware Graph Machine Learning

    [https://arxiv.org/abs/2606.05046](https://arxiv.org/abs/2606.05046)

    Graph Cascades提出了一种基于传染扩散的介观图重连策略，在线性时间内将受多跳强化支持的节点对提升为直接邻居，并从理论（标签一致性条件、SBM同配性、有效电阻刻画）和实验上证明其能提升多种GNN与稀疏图Transformer在节点分类任务上的性能。

    

    我们提出了Graph Cascades，这是一种面向图神经网络（GNN）和图Transformer（GT）的介观重连策略，能够捕获超越纯局部边或完全全局注意力的中间尺度图结构。Graph Cascades使用具有固定级联参数的基于传染（contagion）的扩散过程，在O(|V|+|E|)时间内构建一个稀疏辅助图，其中受到重复多跳强化（reinforcement）支持的节点对被提升为直接邻居。我们从理论上刻画了基于强化的重连何时有效：给出了基于强化的边选择比直接邻接更具标签一致性的充分条件；提供了一个随机块模型（SBM）实例，其中两跳强化是完全同配的；并通过图有效电阻对介观连通性进行了形式化。实证结果表明，在多个节点分类基准上，Graph Cascades改进了多种GNN和稀疏图Transformer骨干网络，其中最可靠的……（原文摘要在此处截断）

    arXiv:2606.05046v2 Announce Type: replace  Abstract: We introduce Graph Cascades, a mesoscopic rewiring strategy for Graph Neural Networks (GNNs) and Graph Transformers (GTs) that captures intermediate-scale graph structure beyond purely local edges or fully global attention. Using contagion-based diffusion with fixed cascade parameters, Graph Cascades constructs, in O(|V|+|E|) time, a sparse auxiliary graph where node pairs supported by repeated multi-hop reinforcement are promoted to direct neighbors. We theoretically characterize when reinforcement-based rewiring helps: sufficient conditions under which reinforcement-based edge selection is more label-aligned than direct adjacency, a stochastic block model (SBM) witness in which two-hop reinforcement is perfectly homophilic, and a formalization of mesoscopic connectivity via graph effective resistance. Empirically, across node-classification benchmarks, Graph Cascades improves multiple GNN and sparse-GT backbones, with the most reli
    
[^293]: 用于原子间势主动学习的Stein核化分子动力学

    Stein Kernelized Molecular Dynamics for Active Learning of Interatomic Potentials

    [https://arxiv.org/abs/2606.04100](https://arxiv.org/abs/2606.04100)

    提出了Stein核化分子动力学（SKMD），一种基于相互作用粒子动力学的增强采样方法，能主动获取多样且有信息量的训练构型用于微调机器学习原子间势，并在渐近极限下保持玻尔兹曼分布为平稳分布。

    

    机器学习原子间势（MLIP）能够实现高效且精确的原子尺度模拟，但其性能关键取决于训练数据的质量与多样性。我们提出了Stein核化分子动力学（SKMD），这是一种增强采样方法，利用相互作用粒子动力学来获取具有信息量的训练构型，用于MLIP的微调。SKMD对应于Stein变分梯度下降的一种随机化变体，通过引入异步粒子更新以及基于全局原子描述符的核函数而适配于分子动力学，该核函数提供了一种对称性感知的构型相似性度量。与分子动力学中使用的其他增强采样器不同，SKMD在适当的渐近极限下能够保持玻尔兹曼分布作为其动力学的平稳分布。这一特性在多样构型的探索与向高（概率区域的吸引）之间实现了平衡……

    arXiv:2606.04100v2 Announce Type: replace  Abstract: Machine learning interatomic potentials (MLIPs) enable efficient and accurate atomistic simulations but depend critically on the quality and diversity of the training data. We introduce Stein kernelized molecular dynamics (SKMD), an enhanced sampling method that uses interacting particle dynamics to acquire informative training configurations for the fine-tuning of MLIPs. SKMD corresponds to a stochastic variant of Stein variational gradient descent that is adapted for molecular dynamics by incorporating asynchronous particle updates and a kernel of global atomic descriptors, which provides a symmetry-aware measure of configurational similarity. Unlike other enhanced samplers used in molecular dynamics, SKMD maintains the Boltzmann distribution as a stationary distribution of its dynamics in the appropriate asymptotic limit. This property enforces a balance between the exploration of diverse configurations and attraction toward high-
    
[^294]: 基于噪声代理协变量的半监督学习：泛化界与分布回归

    Semi-Supervised Learning with Noisy Proxy Covariates: Generalization Bounds and Distribution Regression

    [https://arxiv.org/abs/2606.00512](https://arxiv.org/abs/2606.00512)

    该论文提出一种简单的两阶段半监督回归估计器——从大量含噪声的预训练代理协变量中学习核特征值特征、并在少量标签上拟合岭回归预测器，从理论上证明当代理扰动可控且无标签数据充足时可恢复快速标签样本率，并将分布回归作为特例纳入统一的泛化保证框架。

    

    在许多现代机器学习流程中，大量预训练表示被用作含噪声的代理协变量，而特定任务的标签仍然稀缺。我们研究了这一设定下的半监督回归问题，并提出了一种简单的两阶段估计器：从所有代理协变量中学习核特征值特征，然后在有标签数据上拟合岭回归预测器。我们推导了有限样本界，表明当代理扰动得到控制且无标签代理协变量足够充足时，可以恢复快速的有标签样本收敛率。我们还证明分布回归是该框架的直接特例，当有限的袋大小足够大时，同样可以获得类似的保证。实验表明，该方法相比监督和半监督基线取得了一致的性能提升，尤其是在标签稀缺的场景下。

    arXiv:2606.00512v2 Announce Type: replace  Abstract: In many modern machine learning pipelines, abundant pretrained representations serve as noisy proxy covariates, while task-specific labels remain scarce. We study semi-supervised regression in this setting, and propose a simple two stage estimator that learns kernel eigenfeatures from all proxy covariates and fits a ridge predictor on labeled data. We derive finite sample bounds showing that fast labeled sample rates are recovered when proxy perturbation is controlled and unlabeled proxy covariates are sufficiently abundant. We also show that distribution regression is a direct special case, with analogous guarantees when the finite bag size is large enough. Experiments show consistent gains over supervised and semi-supervised baselines, especially in low label regimes.
    
[^295]: 基于切片散度的多元分布强化学习

    Multivariate Distributional Reinforcement Learning Using Sliced Divergences

    [https://arxiv.org/abs/2605.31222](https://arxiv.org/abs/2605.31222)

    该论文提出切片分布强化学习（SDRL），通过投影将易于处理的一维散度推广到多元回报分布，并证明了均匀切片和最大切片变体分别在标量折扣和一般矩阵折扣下的贝尔曼压缩性，同时支持 Wasserstein、Cramér 和 MMD 等多种基础散度。

    

    分布强化学习（DRL）对完整的回报分布进行建模，而不仅仅是对期望建模，但将其扩展到多元情形仍然具有挑战性。许多常见的度量无法自然地推广到一维之外，或者会失去计算上的可行性，并且多元情形还引入了额外的困难，例如一般矩阵折扣，对此尚无压缩性结果可用。我们提出了切片分布强化学习（SDRL），它通过投影将易于处理的一维散度提升到多元回报分布上。我们证明了在共享标量折扣下均匀切片的贝尔曼压缩性，并引入了一种在一般稠密折扣矩阵下具有压缩性的最大切片变体。SDRL 支持一大类基础散度；我们分析了 Wasserstein 散度、Cramér 散度和最大均值差异（MMD），并刻画了哪些 SDRL 变体适合标准的单……（摘要在此处被截断）

    arXiv:2605.31222v2 Announce Type: replace  Abstract: Distributional reinforcement learning (DRL) models the full return distribution rather than expectations, but extending it to multivariate settings remains challenging. Many common metrics do not naturally generalize beyond one dimension or lose computational tractability, and the multivariate case introduces additional difficulties such as general matrix discounting, for which no contraction results are available. We introduce Sliced Distributional Reinforcement Learning (SDRL), which lifts tractable one-dimensional divergences to multivariate return distributions via projections. We prove Bellman contraction for uniform slicing under shared scalar discounting, and introduce a maximum-slicing variant with contraction under general dense discount matrices. SDRL supports a broad class of base divergences; we analyze Wasserstein, Cram\'er, and Maximum Mean Discrepancy (MMD), and characterize which SDRL variants suit the standard single
    
[^296]: 具有1比特通信约束的批式随机线性老虎机

    Batched Stochastic Linear Bandits with 1-Bit Communication Constraints

    [https://arxiv.org/abs/2605.30976](https://arxiv.org/abs/2605.30976)

    该论文首次研究了批处理与1比特通信约束相结合的随机线性老虎机问题，并证明了即使在无噪声环境下，1比特通信瓶颈也会导致 $\Omega(B\min\{d,\log|\mathcal{A}|\})$ 的不可避免的遗憾下界。

    

    我们研究了在批处理与通信约束的一种自然组合下的随机线性老虎机问题：整个时间范围被划分为大小均为 $B$ 的批次，在每个批次中，学习者向代理发送 $B$ 个所请求的动作拉取，代理随后观测相应的 $B$ 个奖励，并向学习者反馈仅一个比特的信息。对于每个批次，学习者指定代理所使用的1比特量化规则，该规则可以依赖于之前接收到的所有比特，但不能直接依赖于任何过去的奖励。这一设置填补了以往仅考虑逐轮量化模型与仅有总比特预算模型之间一个重要但尚未被探索的“中间地带”。我们建立了一个极小极大下界，表明由于1比特通信瓶颈，即使在没有噪声的情况下，$\Omega(B\min\{d,\log\lvert \mathcal{A} \rvert\})$ 的遗憾也是不可避免的。结合标准的统计极限，这得到一个一般的低……

    arXiv:2605.30976v2 Announce Type: replace-cross  Abstract: We study stochastic linear bandits under a natural combination of batching and communication constraints: the time horizon is partitioned into batches of equal size $B$, and during each batch the learner sends $B$ requested arm pulls to an agent, who then observes the corresponding $B$ rewards and responds with a single bit of feedback to the learner. For each batch, the learner specifies the 1-bit quantization rule the agent uses, which may depend on all previously received bits but not on any past rewards directly. This setting addresses a significant yet unexplored ``middle ground'' between previous models having per-round quantization only or total bit budgets only. We establish a minimax lower bound showing that $\Omega(B\min\{d,\log\lvert \mathcal{A} \rvert\})$ regret is unavoidable due to the 1-bit communication bottleneck, even in the absence of noise. Combined with standard statistical limits, this yields a general low
    
[^297]: 监督训练在生物合理学习规则下迅速降低早期视觉皮层对齐度

    Supervised Training Rapidly Degrades Early Visual Cortex Alignment Across Biologically Plausible Learning Rules

    [https://arxiv.org/abs/2605.30556](https://arxiv.org/abs/2605.30556)

    该论文发现，监督训练在所有测试的生物合理学习规则下都会降低早期视觉皮层（V1）的表征对齐度，且随机未训练网络反而匹配或超越训练网络，挑战了学习改善大脑对齐的传统假设。

    

    arXiv:2605.30556v2 公告类型：替换 摘要：更正（2026年8月）：本文的核心发现不被支持。评估模式缺陷导致预测编码和STDP条件下的批归一化层在特征提取期间处于训练模式，从而产生了它们保留V1对齐的表象。修复该缺陷后，预测编码对V1对齐的退化程度大于反向传播，而非小于。训练对所有测试规则均降低V1对齐的发现仍然成立。详见第1页更正说明；以下原始摘要和正文与v1版本一致。更正分析：arXiv:2608.12408。随机、未训练的网络在早期视觉皮层的表征相似性上始终匹配或超过训练过的网络。这一令人困惑的发现挑战了学习改善大脑对齐的假设。我们通过追踪表征相似性分析（RSA）与人类fMRI数据的对齐来研究这一现象。

    arXiv:2605.30556v2 Announce Type: replace  Abstract: CORRECTION (August 2026): the central finding of this paper is not supported. An evaluation-mode defect left the batch-normalisation layers of the predictive-coding and STDP conditions in training mode during feature extraction, producing their apparent preservation of V1 alignment. With the defect repaired, predictive coding degrades V1 alignment more than backpropagation does, not less. The finding that training degrades V1 alignment for every rule tested does survive. See the correction note on page 1; the original abstract below and the body are unchanged from v1. Corrected analysis: arXiv:2608.12408.   Random, untrained neural networks consistently match or exceed trained networks in representational similarity to early visual cortex. This puzzling finding challenges the assumption that learning improves brain alignment. We investigate it by tracking representational similarity analysis (RSA) alignment to human fMRI data across 
    
[^298]: CardioLoom：一种用于跨领域心血管表型分析的报告引导式心电图基础模型

    CardioLoom: A Report-Guided Electrocardiogram Foundation Model for Cross-Domain Cardiovascular Phenotyping

    [https://arxiv.org/abs/2605.25446](https://arxiv.org/abs/2605.25446)

    CardioLoom是一种利用近280万心电图-报告配对数据预训练的报告引导式心电图基础模型，在89个内外部心血管表型评估目标上表现优异，仅用10%标注数据即可超越全量监督训练的性能，实现了跨领域的疾病诊断与筛查能力。

    

    常规心电图（ECG）报告为心血管表征学习提供了可扩展的监督信号。在这项回顾性研究中，我们在来自1,298,359名患者的2,781,289个心电图-报告配对数据上预训练了CardioLoom——一种报告引导的心电图基础模型。下游开发使用了来自233,798名患者的721,317份心电图的独立队列。评估涵盖89个目标，并在九个队列中进行了外部测试。在45项内部心电图诊断中，宏平均精确率-召回率曲线下面积（AUPRC）为0.624，而DeepECG-SSL为0.615。在203项外部心电图评估中，相应均值分别为0.425和0.424。仅使用10%训练患者的标签，CardioLoom的宏平均AUPRC即达到0.559，超过了使用全部标签的监督训练（0.549）。对于射血分数降低的心力衰竭，在中山、世源和BIDMC三个队列中的AUPRC分别为0.690、0.641和0.677。在心脏淀粉样变性筛查中获得了4.5%的阳性预测值。

    arXiv:2605.25446v2 Announce Type: replace  Abstract: Routine electrocardiogram (ECG) reports offer scalable supervision for cardiovascular representation learning. In this retrospective study, we pretrained CardioLoom, a report-guided ECG foundation model, on 2,781,289 ECG-report pairs from 1,298,359 patients. Downstream development used a separate cohort of 721,317 ECGs from 233,798 patients. Evaluation covered 89 targets, with external testing in nine cohorts. Across 45 internal ECG diagnoses, macro area under the precision-recall curve (AUPRC) was 0.624 versus 0.615 for DeepECG-SSL. Across 203 external ECG evaluations, corresponding means were 0.425 and 0.424. Using labels from 10% of training patients, CardioLoom achieved macro-AUPRC 0.559, exceeding supervised training with all labels (0.549). For heart failure with reduced ejection fraction, AUPRCs were 0.690, 0.641 and 0.677 in Zhongshan, Shiyuan and BIDMC. Cardiac amyloidosis screening yielded 4.5% positive predictive value usi
    
[^299]: 面向安全上下文内强化学习的潜在Q屏障护盾

    Latent Q-Barrier Shielding for Safe In-Context Reinforcement Learning

    [https://arxiv.org/abs/2605.25267](https://arxiv.org/abs/2605.25267)

    提出一种潜在Q屏障护盾，在部署前学习上下文表示与成本预测模型，在测试时无需参数更新即可依据剩余安全预算和预测的未来成本对候选动作进行过滤或软重加权，从而在分布外偏移下显著改善安全上下文强化学习的奖励-安全权衡。

    

    安全上下文内强化学习（ICRL）无需在测试时进行参数更新，即可从交互历史中进行在线适应，同时在安全预算内控制回合成本。在分布外（OOD）部署偏移下，仅依赖预训练的安全ICRL可能产生较差的奖励-安全权衡，因为剩余预算仅通过冻结的策略条件化来影响行为，而缺乏针对预测的未来成本进行显式的动作级检查。我们提出了一种潜在Q屏障护盾，它在部署前学习上下文表示、潜在动态模型以及一个集成成本评论家（ensemble cost critic）。无需参数更新，该护盾从历史中推断上下文，并利用剩余预算和预测的未来成本对候选动作进行过滤或软重加权。我们证明了在实现的候选动作集合上的条件化单步屏障边际界，其中包含显式的贝尔曼误差项和潜变量预测误差项。在五个安全ICRL基准测试中，该护盾（提升了奖励-安全权衡表现）。

    arXiv:2605.25267v2 Announce Type: replace-cross  Abstract: Safe in-context reinforcement learning (ICRL) adapts online from interaction history without test-time parameter updates while controlling episode cost under a safety budget. Under out-of-distribution (OOD) deployment shifts, pretraining-only safe ICRL can give poor reward-safety tradeoffs because the remaining budget affects behavior only through frozen policy conditioning, not an explicit action-level check against predicted future cost. We propose a latent Q-Barrier shield that learns a context representation, latent dynamics, and an ensemble cost critic before deployment. Without parameter updates, the shield infers context from history and filters or softly reweights candidate actions using the remaining budget and predicted future cost. We prove a conditional one-step barrier-margin bound over realized candidate sets, with explicit Bellman and latent-prediction error terms. Across five safe ICRL benchmarks, the shield imp
    
[^300]: 面向超参数优化基础模型的开源训练数据集

    An Open-Source Training Dataset for Foundation Models for Hyperparameter Optimization

    [https://arxiv.org/abs/2605.23417](https://arxiv.org/abs/2605.23417)

    该论文提出了BBO-Pile——首个面向超参数优化基础模型的开源训练数据集，包含超过50万条优化轨迹，覆盖3,095个不同的超参数优化任务和全局优化问题，解决了该领域缺乏大规模、真实世界公开预训练数据的瓶颈。

    

    大多数超参数优化方法需要对其自身的超参数进行大量调优，这往往限制了它们在不同基准测试族之间的泛化能力。从大量优化轨迹中学习优化原理的超参数优化基础模型提供了一种有前景的替代方案，有潜力在多样化的问题类别中超越人工设计的方法。然而，先前的工作要么依赖于非公开的数据集，要么依赖于纯合成数据，限制了可复现性以及对现实世界问题的泛化能力。因此，该领域的进展一直受到缺乏大规模、真实世界、公开可用的预训练数据的制约。我们提出了BBO-Pile，这是首个开源数据集，包含超过50万条优化轨迹，覆盖3,095个不同的超参数优化任务以及针对不同优化器的全局优化问题。

    arXiv:2605.23417v2 Announce Type: replace  Abstract: Most hyperparameter optimization methods require extensive tuning of their own hyperparameters, often limiting their ability to generalize across different benchmark families. Foundation models for hyperparameter optimization that learn optimization principles from a large collection of optimization trajectories offer a promising alternative, with the potential to outperform manually designed methods across diverse problem classes. However, prior work as either relied on non-public datasets or on purely synthetic data, limiting reproducibility and generalization to real-world problems. As a result, progress in this area has been constrained by the lack of large-scale, real-world, publicly available pre-training data. We introduce BBO-Pile, the first open-source dataset comprising over 500K optimization trajectories evaluated across 3,095 different hyperparameter optimization tasks and global optimization problems for different optimi
    
[^301]: RelPrism：面向关系型数据库的多方面自监督学习

    RelPrism: Multi-Faceted Self-Supervised Learning on Relational Databases

    [https://arxiv.org/abs/2605.23241](https://arxiv.org/abs/2605.23241)

    提出RelPrism框架，通过从节点属性、关系结构等多个方面进行自监督预训练，学习能够全面覆盖关系型数据库上多样化预测任务所需信息的表示，克服了现有SSL方法仅依赖单一监督方面的局限。

    

    关系型数据库（RDBs）仍然是现代数据系统的骨干，支持着广泛需要从不同视角和粒度获取多方面信息的预测任务。例如，用户流失预测通常被表述为一个二元分类任务，更依赖于反映活跃程度的用户交互模式；而消费价值估计则是一个细粒度的回归任务，可能需要来自用户属性和用户-物品交互行为的综合证据。因此，一个核心挑战是学习能够全面覆盖RDB任务底层信息的表示。一个自然的解决方案是利用自监督学习（SSL）进行预训练。然而，现有的SSL方法通常从单一方面（如节点级内在属性或子图级关系结构）获取监督信号，因此其预训练目标……

    arXiv:2605.23241v2 Announce Type: replace  Abstract: Relational databases (RDBs) remain the backbone of modern data systems and support a broad spectrum of predictive tasks that require multi-faceted information from different perspectives and granularities. For example, user churn prediction is often formulated as a binary classification task that relies more on user interaction patterns reflecting activity levels, whereas consumption value estimation is a fine-grained regression task that may require composite evidence from both user attributes and user--item interaction behaviors. Therefore, a central challenge is to learn representations with comprehensive coverage of the information underlying RDB tasks. A natural solution is to leverage self-supervised learning (SSL) for pre-training. However, existing SSL methods typically derive supervision from a single facet, such as node-level intrinsic attributes or subgraph-level relational structures. Consequently, their pre-training obje
    
[^302]: 跨物种RSA揭示人类fMRI与猕猴电生理中保守的早期视觉对齐但高级区域排名存在差异

    Cross-Species RSA Reveals Conserved Early Visual Alignment but Divergent Higher-Area Rankings Across Human fMRI and Macaque Electrophysiology

    [https://arxiv.org/abs/2605.22401](https://arxiv.org/abs/2605.22401)

    该研究通过跨物种表征相似性分析，发现人类fMRI与猕猴电生理在早期视觉处理上存在保守对齐，但高级脑区的排名在不同物种间存在显著差异。

    

    arXiv:2605.22401v2 公告类型：替换 摘要：更正（2026年8月）：共享特征提取管道中的一个评估模式缺陷影响了预测编码和STDP条件。该问题适用于此处所有比较的双方：人类数据来自同伴研究，猕猴数据使用相同检查点。在使用修复后检查点的单一种子重新评估中，猕猴V1区的STDP从0.305变为0.266，PC从0.210变为0.178，与未训练基线持平，因此发现（2）对STDP成立但对PC不成立；V4和IT的排序也发生变化，而V2不变。发现（1）、（3）和（4）不受影响，其他条件变化最多为0.0034。五种子的分析、Kendall's tau、噪声上限和刺激控制尚未重新计算。见第1页更正说明；以下原始摘要与v1版本一致。后续研究：arXiv:2608.12408。学习规则与大脑之间的关系是否...

    arXiv:2605.22401v2 Announce Type: replace  Abstract: CORRECTION (August 2026): an evaluation-mode defect in the shared feature-extraction pipeline affected the predictive-coding and STDP conditions. It applies to both sides of every comparison here: the human values are reprinted from the companion study and the macaque values use the same checkpoints. In a single-seed re-evaluation with repaired checkpoints, macaque STDP at V1 moves from 0.305 to 0.266 and PC from 0.210 to 0.178, level with the untrained baseline, so finding (2) holds for STDP but not for PC; the V4 and IT orderings also change, while V2 is unchanged. Findings (1), (3) and (4) are unaffected, and the other conditions change by at most 0.0034. The five-seed analysis, Kendall's tau, noise ceilings and stimulus control have not been re-computed. See the correction note on page 1; the original abstract below is unchanged from v1. Follow-up study: arXiv:2608.12408.   Does the relationship between learning rules and brain a
    
[^303]: PEARL：基于对比学习的无偏分位数估计方法及其在工业级直播推荐中的应用

    PEARL: Unbiased Percentile Estimation via Contrastive Learning for Industrial-Scale Livestream Recommendation

    [https://arxiv.org/abs/2605.21752](https://arxiv.org/abs/2605.21752)

    PEARL 提出了一种基于对比学习的非参数分位数近似框架，通过建模相对偏好信号而非绝对参与强度来消除用户行为强度不平衡带来的偏差，从而提升工业级直播推荐的质量与鲁棒性。

    

    arXiv:2605.21752v2 公告类型： replace-cross

    arXiv:2605.21752v2 Announce Type: replace-cross  Abstract: Recommender systems trained on user interaction data are susceptible to behavioral intensity imbalance--a systematic distortion arising from heterogeneous engagement patterns across users. This imbalance skews feedback signals such that observed interactions no longer faithfully reflect true preferences, causing models to disproportionately amplify signals from highly active users while underrepresenting others, which ultimately degrades recommendation quality and robustness at scale. To address this issue, we propose a nonparametric contrastive percentile approximation framework, PEARL, that models relative preference signals instead of absolute engagement magnitudes. Building upon relative advantage debiasing, PEARL leverages real contrastive interaction samples to approximate percentile relationships directly, without relying on auxiliary distribution estimation models. We provide theoretical justification demonstrating that
    
[^304]: 超越单槽位：多槽位保量展示广告的联合优化

    Beyond Single Slot: Joint Optimization for Multi-Slot Guaranteed Display Advertising

    [https://arxiv.org/abs/2605.21556](https://arxiv.org/abs/2605.21556)

    本文提出一种多槽位保量展示广告联合优化框架，将分配建模为带合约轮盘机制和页面浏览约束的离线二部图匹配问题，显著提升商家ROI、平台收入效率与合约履约稳健性。

    

    保量展示广告对平台变现至关重要，然而现有方法通常基于单槽位假设运行，限制了其在多槽位页面浏览场景下优化分配的能力。本文提出了一种新颖的多槽位保量展示广告分配联合优化框架，解决了槽位级冗余、合约不平衡和曝光集中等关键挑战。我们的方法将分配问题建模为离线二部图匹配问题，引入合约轮盘机制实现槽位独占性，并通过页面浏览约束控制曝光，同时结合可扩展的分配优化算法以实现高效的大规模部署。在美团广告平台上的大量在线测试表明，我们的方法显著提升了商家ROI、平台收入效率以及合约履约的稳健性。具体而言，在线A/B测试显示平均值指标提升了28.99%。

    arXiv:2605.21556v2 Announce Type: replace  Abstract: Guaranteed display advertising is crucial for platform monetization, yet existing methods often operate under a single-slot assumption, limiting their ability to optimize allocation across multi-slot page views. In this paper, we propose a novel joint optimization framework for multi-slot GD allocation, addressing key challenges such as slot-level redundancy, contract imbalance, and exposure concentration. Our approach formulates the allocation as an offline bipartite matching problem with a contract roulette mechanism for slot exclusivity and Page View constraints for impression control, and incorporates a scalable allocation optimization algorithm for efficient large-scale deployment. Extensive online tests on the Meituan advertising platform demonstrate that our method significantly improves merchant ROI, platform revenue efficiency, and contract fulfillment robustness. Specifically, online A/B tests show a 28.99% increase in Aver
    
[^305]: 一种用于钙钛矿前驱体添加剂发现的闭环框架

    A closed-loop framework for perovskite precursor additive discovery

    [https://arxiv.org/abs/2605.20242](https://arxiv.org/abs/2605.20242)

    该论文提出LEAP闭环框架，将领域专业化大语言模型与主动学习相结合，从文献中提取机理知识并转化为可解释分子描述符进行贝叶斯优化，从而在低数据条件下高效发现钙钛矿前驱体添加剂，将太阳能电池效率提升至20.87%。

    

    高效发现前驱体添加剂对于提升钙钛矿太阳能电池的性能至关重要，然而庞大的化学空间使得传统的试错式筛选效率低下。本研究提出了LEAP（基于大语言模型与主动学习的钙钛矿探索框架），这是一个将领域专业化语言模型与主动学习相结合的闭环框架，能够在低数据条件下对钙钛矿前驱体添加剂进行优先排序。该模型从文献中提取与作用机理相关的知识，并将其转化为可解释的分子描述符，用于具备不确定性感知能力的贝叶斯优化。在一项基于未见过的文献进行的评估中，该专业化模型比所测试的通用模型给出了更符合机理的答案。在一项专家指导的概念验证研究中，两个后期轮次筛选出的添加剂将器件的平均功率转换效率分别提升至20.13%和20.87%（相比对照的19.25%）

    arXiv:2605.20242v2 Announce Type: replace-cross  Abstract: Efficient discovery of precursor additives is essential for improving the performance of perovskite solar cells, yet the large chemical space makes conventional trial-and-error screening inefficient. Here we show that LEAP (LLM-driven Exploration via Active Learning for Perovskites), a closed-loop framework combining a domain-specialized language model with active learning, prioritizes perovskite precursor additives under low-data conditions. The model extracts mechanism-relevant knowledge from the literature and converts it into interpretable molecular descriptors for uncertainty-aware Bayesian optimization. On an evaluation drawn from unseen literature, the specialized model provides more mechanism-consistent answers than the tested general-purpose models. In an expert-guided proof-of-concept study, two later-round additives increase average device power conversion efficiencies to 20.13% and 20.87%, compared with 19.25% for t
    
[^306]: Q-LocalAdam：面向边缘联邦学习的内存高效客户端自适应优化

    Q-LocalAdam: Memory-Efficient Client-Side Adaptive Optimization for Edge Federated Learning

    [https://arxiv.org/abs/2605.17552](https://arxiv.org/abs/2605.17552)

    提出分布感知的量化优化器 Q-LocalAdam，通过对动量状态进行分块线性量化、对重尾方差状态进行对数空间量化并设置反量化后方差下限，在内存受限的边缘联邦学习中显著降低优化器状态内存占用，同时保持接近全精度训练的收敛性能。

    

    设备端联邦学习（FL）在边缘设备上受到客户端自适应优化器所需内存的限制。在本地维护优化器状态会增加内存使用量，并可能限制可在资源受限的边缘设备上部署的模型规模。状态量化可以减少这种内存占用，但均匀压缩方法在偏斜的非独立同分布（non-IID）数据分布下可能表现不佳，因为它们会丢失对稳定优化至关重要的低方差值。我们提出了 Q-LocalAdam，一种面向内存受限边缘设备的分布感知优化器。它对有界动量状态应用分块线性量化，对重尾方差状态应用对数空间量化，并通过反量化后的方差下限来保持数值稳定性。Q-LocalAdam 在降低持久性优化器状态内存占用的同时，实现了接近全精度训练的收敛效果。

    arXiv:2605.17552v2 Announce Type: replace  Abstract: On-device federated learning (FL) on edge devices is limited by the memory required by client-side adaptive optimizers. Maintaining optimizer states locally increases memory usage and can restrict the size of models that can be deployed on resource-constrained edge devices. State quantization can reduce this memory footprint, but uniform compression methods may perform poorly under skewed, non-IID data distributions because they can lose small-variance values that are important for stable optimization.   We present \textbf{Q-LocalAdam}, a distribution-aware optimizer for memory-constrained edge devices. It applies block-wise linear quantization to the bounded momentum state and log-space quantization to the heavy-tailed variance state. A post-dequantization variance floor is used to maintain numerical stability. Q-LocalAdam reduces the persistent optimizer-state memory footprint while achieving convergence close to full-precision tra
    
[^307]: 面向最优治疗选择的决策感知近端桥学习

    Decision-Aware Proximal Bridge Learning for Optimal Treatment Selection

    [https://arxiv.org/abs/2605.16989](https://arxiv.org/abs/2605.16989)

    该论文提出一种决策感知的近端桥学习方法，将近端因果推断从治疗效应估计拓展到存在隐藏混杂时的最优治疗选择，从而在决策相关区域而非整个动作空间上实现准确的因果响应估计。

    

    针对连续动作的个体化治疗选择，需要在决策相关区域进行准确的因果响应估计，而不是在整个动作空间上均匀地进行估计。因此，先估计一个全局因果响应面再选择使其最大化的治疗可能是次优的，因为标准的估计目标是根据观测到的治疗分布来分配建模精力，而不是根据决定最优决策的区域来分配。虽然决策感知方法已在无混杂的设置中得到研究，但这一问题在近端因果推断中仍未被充分探索——在近端因果推断中，代理变量和桥函数可以在适当假设下、即使存在隐藏混杂的情况下实现识别。尽管近期已有进展，近端方法主要集中于治疗效应和潜在结果的估计，而非治疗选择和最优决策。为了弥合……

    arXiv:2605.16989v2 Announce Type: replace  Abstract: Individualized treatment selection with continuous actions requires accurate causal response estimation in decision-relevant regions, rather than uniformly over the entire action space. Estimating a global causal response surface and then choosing the treatment that maximizes it can therefore be suboptimal, since standard estimation objectives allocate modeling effort according to the observed treatment distribution rather than the regions that determine the optimal decision. While decision-aware approaches have been studied in unconfounded settings, this problem remains underexplored in proximal causal inference, where proxy variables and bridge functions enable identification under suitable assumptions even in the presence of hidden confounding. Despite recent progress, proximal methods have primarily focused on treatment-effect and potential-outcome estimation rather than treatment selection and optimal decision-making. To bridge 
    
[^308]: 何时热启动对语言模型扩展有效？

    When is Warmstarting Effective for Scaling Language Models?

    [https://arxiv.org/abs/2605.13405](https://arxiv.org/abs/2605.13405)

    该研究实证发现热启动无需保留小模型增长后的初始性能，简单的架构无关增长策略即可超越复杂算子，并确定了增长因子的经验上限——超过该界限时从头训练更为高效。

    

    arXiv:2605.13405v2 公告类型：替换 摘要：从一个给定的检查点进行模型增长旨在加速更大模型的训练，具有潜在的资源节约优势。尽管近来受到关注，热启动在大规模训练中的实际应用仍然有限。我们将此归因于两个尚未充分探索的因素：（1）过度强调在初始化时保留较小模型的性能，这限制了针对新架构的算子设计；（2）对增长如何与超参数和缩放行为相互作用的不足分析，加之文献中增长因子不一致。我们表明，保留基础模型增长后的初始性能对于获得强大的最终性能并非必要，而且简单的、与架构无关的增长策略可以胜过更复杂的热启动算子。至关重要的是，我们通过实证确定了增长因子 $g$ 的上限，超过该上限时，从头训练更为高效。我们观察到这一现象……

    arXiv:2605.13405v2 Announce Type: replace  Abstract: Model growth from a given checkpoint aims to accelerate training of a larger model, offering potential resource savings. Despite recent interest, warmstarting has seen limited practical adoption in large-scale training. We attribute this to two underexplored factors: (1) an overemphasis on preserving the smaller model's performance at initialization, which constrains operator design for new architectures, and (2) insufficient analysis of how growth interacts with hyperparameters and scaling behavior, compounded by inconsistent growth factors across the literature.   We show that preserving the base model's initial post-growth performance is not necessary for strong final performance, and that simple, architecture-agnostic growth strategies can outperform more complex warmstarting operators. Crucially, we empirically identify an upper bound on the growth factor $g$ beyond which training from scratch is more efficient. We observe this 
    
[^309]: 贝叶斯优化中非线性参数模型的基于核方法的理论保证

    Kernel-based guarantees for nonlinear parametric models in Bayesian optimization

    [https://arxiv.org/abs/2605.13160](https://arxiv.org/abs/2605.13160)

    本文提出一个基于核方法的框架，为自适应采集数据上训练的正则化非线性参数模型建立置信界，并据此为近似汤普森采样等贝叶斯优化方法提供收敛保证。

    

    现代贝叶斯优化和自适应采样方法越来越依赖于非线性参数模型（例如神经网络），然而此类模型在自适应数据采集下的理论保证仍然有限。现有分析主要集中于高斯过程、核机器、线性模型或线性化的神经近似，导致理论与实践所用的非线性模型之间存在差距。我们开发了一个基于核方法的框架，用于分析在自适应采集数据上训练的正则化参数模型。我们的方法利用参数空间上的核在相应模型类上诱导出再生核希尔伯特空间结构，从而为使用广泛一类正则化损失训练的模型给出置信界。我们展示了这些置信界如何为近似汤普森采样建立收敛保证，并为分析更广泛的无似然贝叶斯优化方法奠定基础。

    arXiv:2605.13160v2 Announce Type: replace-cross  Abstract: Modern Bayesian optimization and adaptive sampling methods increasingly rely on nonlinear parametric models, such as neural networks, yet theoretical guarantees for such models under adaptive data collection remain limited. Existing analyses largely focus on Gaussian processes, kernel machines, linear models, or linearized neural approximations, leaving a gap between theory and the nonlinear models used in practice. We develop a kernel-based framework for analyzing regularized parametric models trained on adaptively collected data. Our approach uses kernels over the parameter space to induce reproducing kernel Hilbert space structures over the corresponding model class, yielding confidence bounds for models trained with a broad class of regularized losses. We show how these bounds can establish convergence guarantees for approximate Thompson sampling and provide a foundation for analyzing broader likelihood-free Bayesian optimi
    
[^310]: 量化低温机制下均场Transformer的集中现象

    Quantifying Concentration Phenomena of Mean-Field Transformers in the Low-Temperature Regime

    [https://arxiv.org/abs/2605.10931](https://arxiv.org/abs/2605.10931)

    该论文借鉴多粒子系统收敛分析思想，证明了在低温极限下深度编码器Transformer的token分布会快速集中到由键、查询、值矩阵诱导的投影映射下初始分布的推前分布上，给出了Wasserstein距离随温度参数和推理时间的定量衰减速率，并证明该分布在中长时间内保持亚稳态。

    

    以自注意力模块为核心组件的Transformer已成为现代大型语言模型和基础模型中不可或缺的架构。本文研究了推理时深度编码器（encoder-only）Transformer中token的演化过程，该过程在大token极限下由均场连续性方程描述。借鉴相互作用多粒子系统收敛分析的思想——其中粒子对应于token——我们证明了token分布会快速集中到由键、查询和值矩阵所诱导的投影映射下初始分布的推前分布上，并在中等时间尺度内保持亚稳态。具体而言，我们证明了这两个分布之间的Wasserstein距离在温度参数 β⁻¹→0 和推理时间 t≥0 的意义下，其尺度为 √(log(β+1)/β)·exp(Ct)+exp(−ct)。在证明过程中，我们建立了Lyapunov……（原文摘要在此处截断）

    arXiv:2605.10931v2 Announce Type: replace-cross  Abstract: Transformers with self-attention modules as their core components have become an integral architecture in modern large language and foundation models. In this paper, we study the evolution of tokens in deep encoder-only transformers at inference time which is described in the large-token limit by a mean-field continuity equation. Leveraging ideas from the convergence analysis of interacting multi-particle systems, with particles corresponding to tokens, we prove that the token distribution rapidly concentrates onto the push-forward of the initial distribution under a projection map induced by the key, query, and value matrices, and remains metastable for moderate times. Specifically, we show that the Wasserstein distance of the two distributions scales like $\sqrt{{\log(\beta+1)}/{\beta}}\exp(Ct)+\exp(-ct)$ in terms of the temperature parameter $\beta^{-1}\to 0$ and inference time $t\geq 0$. For the proof, we establish Lyapunov
    
[^311]: 广延宽度网络中的尖锐特征学习转变与贝叶斯最优神经标度律

    Sharp feature-learning transitions and Bayes-optimal neural scaling laws in extensive-width networks

    [https://arxiv.org/abs/2605.10395](https://arxiv.org/abs/2605.10395)

    该论文利用留一解耦论证与封闭不动点方程，在高维广延宽度设定下首次给出贝叶斯最优泛化误差的渐近精确刻画，并揭示教师特征随数据量增长依次通过不连续跳变被学习的尖锐相变序列及其背后的神经标度律。

    

    我们研究了在向更小的学生模型进行知识迁移的背景下，从带噪查询中学习具有层级特征的单隐层教师网络的信息论极限。我们工作在教师宽度 $k$ 与输入维度 $d$ 呈线性标度的高维区间内——这一设定刻画了宽度较大但有限的网络，并且直到最近才在解析上变得可处理。借助一种启发式的留一解耦论证（并在全过程中通过数值验证），我们通过一组封闭的不动点方程，推导出了贝叶斯最优泛化误差和各个特征重叠的渐近精确刻画。这些方程揭示了特征可学习性由一系列尖锐的相变所支配：随着数据量的增长，教师特征被依次恢复，每次恢复都表现为重叠的不连续跳变。这种顺序获取机制构成了精确的……（摘要原文在此处截断）

    arXiv:2605.10395v2 Announce Type: replace-cross  Abstract: We study the information-theoretic limits of learning a one-hidden-layer teacher network with hierarchical features from noisy queries, in the context of knowledge transfer to a smaller student model. We work in the high-dimensional regime where the teacher width $k$ scales linearly with the input dimension $d$ -- a setting that captures large-but-finite-width networks and has only recently become analytically tractable. Using a heuristic leave-one-out decoupling argument, validated numerically throughout, we derive asymptotically sharp characterizations of the Bayes-optimal generalization error and individual feature overlaps via a system of closed fixed-point equations. These equations reveal that feature learnability is governed by a sequence of sharp phase transitions: as data grows, teacher features become recoverable sequentially, each through a discontinuous jump in overlap. This sequential acquisition underlies a precis
    
[^312]: DiffATS：对齐张量空间中的扩散模型

    DiffATS: Diffusion in Aligned Tensor Space

    [https://arxiv.org/abs/2605.09275](https://arxiv.org/abs/2605.09275)

    该论文提出DiffATS，利用Tucker分解并结合正交Procrustes对齐消除张量因子的规范歧义，构建了无需预训练压缩自编码器的紧凑、数据自适应的张量原语，从而实现对高分辨率时空场的高效扩散建模。

    

    arXiv:2605.09275v2 公告类型：替换 摘要：对高分辨率时空场进行直接扩散建模在计算上极具挑战性。参数高效的原语通过用紧凑的参数集合表示高维数据来应对这一问题。在本文中，我们构建了数据依赖的张量原语，且无需依赖预训练的压缩自编码器。我们的构造从Tucker分解出发，该分解通过核心张量和各模态因子捕获低秩多线性结构。然而，Tucker因子具有非唯一性：同一个张量可以由不同的旋转因子来表示，这给生成建模带来了困难。我们通过正交Procrustes（OP）对齐来解决这一问题。具体而言，我们从数据中选取medoid锚定矩阵，并将各因子矩阵对齐以消除规范歧义。由此得到的矩阵格拉斯曼（Grassmannian）原语和张量格拉斯曼原语紧凑、数据自适应，并且可以通过显式多线性（映射）直接解码。

    arXiv:2605.09275v2 Announce Type: replace  Abstract: Direct diffusion modeling of high-resolution spatiotemporal fields is computationally challenging. Parameter-efficient primitives address this by representing high-dimensional data with a compact set of parameters. In this paper, we construct data-dependent tensor primitives without pretrained compression autoencoders. Our construction starts from Tucker decomposition, which captures low-rank multilinear structure through a core tensor and mode-wise factors. However, Tucker factors are non-unique: the same tensor can be represented by different rotated factors, which complicates generative modeling. We address this issue with orthogonal Procrustes (OP) alignment. Specifically, we select medoid anchor matrices from the data and align the factor matrices to resolve the gauge ambiguity. This yields matrix Grassmannian primitives and tensor Grassmannian primitives that are compact, data-adaptive, and directly decodable by explicit multil
    
[^313]: iMuon：黎曼矩阵流形上的内蕴谱优化

    iMuon: Intrinsic Spectral Optimization on Riemannian Matrix Manifolds

    [https://arxiv.org/abs/2605.09238](https://arxiv.org/abs/2605.09238)

    论文提出 iMuon 优化器，通过将黎曼度量规范地把酉不变的欧几里得范数提升为切空间上的内蕴范数，解决了 Muon 推广到低秩、正交、SPD 等矩阵流形时的对称性破坏与闭式解受阻问题。

    

    Muon 及相关的范数约束优化器如今已成为大规模学习的核心方法，它们通过在欧几里得空间中的矩阵范数球上执行线性最大化预言机（LMO）来计算更新。然而，这些方法无法干净地推广到流形值参数，例如低秩分解、正交约束或对称正定（SPD）矩阵。若简单地将 Muon 的 LMO 限制在切空间上，会 破坏商对称性，并 使切空间约束与环境范数界相互耦合，从而阻碍闭式解的求解。我们通过一个关键观察解决了这两个问题：任何黎曼度量都能规范地将一个酉不变的欧几里得范数提升为每个切空间上的内蕴范数。由此得到的内蕴 LMO 能保持对称性，并且在缩放切空间满足某一结构条件下，可退化为标准的欧几里得 LMO。基于此，我们提出了内蕴 Muon（iMuon），一个统一的……（原文摘要在此处截断）

    arXiv:2605.09238v2 Announce Type: replace-cross  Abstract: Muon and related norm-constrained optimizers, now central to large-scale learning, compute updates via a linear maximization oracle (LMO) over a matrix-norm ball in Euclidean space. However, they do not generalize cleanly to manifold-valued parameters such as low-rank factorizations, orthogonality constraints, or symmetric positive-definite (SPD) matrices. Naively restricting the Muon LMO to the tangent space (i) breaks quotient symmetries and (ii) couples the tangent-space constraint with an ambient norm bound, which obstructs closed-form solutions. We resolve both issues with a key observation: every Riemannian metric canonically lifts a unitarily invariant Euclidean norm to an intrinsic norm on each tangent space. The resulting intrinsic LMO is symmetry preserving and, under a structural condition on the scaled tangent space, reduces to a standard Euclidean LMO. Building on this, we introduce intrinsic Muon (iMuon), a unifie
    
[^314]: 量子到量子迁移学习在低数据情况下展现出更强的鲁棒性

    Quantum-to-Quantum Transfer Learning Exhibits Improved Robustness in Low-Data Regimes

    [https://arxiv.org/abs/2605.09118](https://arxiv.org/abs/2605.09118)

    该论文系统评估了多种量子与经典模型在迁移学习任务中的表现，发现尽管经典模型峰值性能更高，但量子模型在训练数据受限时性能下降显著更小，展现出更强的鲁棒性。

    

    在数据受限条件下的迁移学习是一个具有挑战性的场景，模型必须在极少监督的情况下适应新任务。以往的工作主要集中于提高迁移学习的绝对准确率。然而，在现实的迁移学习场景中对量子模型和经典模型进行比较的实证证据仍然有限，尤其是在低数据情况下。在这项工作中，我们系统地研究了量子模型在训练数据减少情况下的鲁棒性。我们在多样化的迁移任务和再训练配置下评估了多种量子与经典架构，并使用准确率下降幅度和相对性能保留率（RPR）来量化鲁棒性。我们的结果表明，尽管经典模型通常能达到更高的峰值性能，但当训练数据受限时，其性能下降幅度显著更大。相比之下，量子模型在不同数据规模下能保持更稳定的性能，这表明其……

    arXiv:2605.09118v2 Announce Type: replace-cross  Abstract: Transfer learning under limited data is a challenging setting, where models must adapt to new tasks with minimal supervision. Prior work has primarily focused on improving absolute accuracy in transfer learning. However, empirical evidence comparing quantum and classical models in realistic transfer learning settings remains limited, especially in low-data regimes. In this work, we systematically study the robustness of quantum models under reduced training data. We evaluate multiple quantum and classical architectures across diverse transfer tasks and retraining configurations, and quantify robustness using accuracy degradation and relative performance retention (RPR). Our results show that, although classical models often achieve higher peak performance, they exhibit significantly larger degradation when training data is limited. In contrast, quantum models maintain more stable performance across data regimes, indicating impr
    
[^315]: 通过规范扩散测量与分解模态分离

    Measuring and Decomposing Mode Separation via the Canonical Diffusion

    [https://arxiv.org/abs/2605.08777](https://arxiv.org/abs/2605.08777)

    该论文提出以密度固有的唯一常数扩散系数可逆扩散过程（规范扩散）为基准，从其自协方差矩阵中提取量来测量并分解高维分布的模态分离程度，弥补了微分熵、PCA 等无标签工具无法刻画势垒分隔结构、而互信息又依赖潜在标签的不足。

    

    模态分离，即一个分布被势垒分隔成簇的锐利程度，是密度的基本几何属性，在高维空间中难以量化。模态往往是数据中有意义的选项，例如分子的构象，或生成模型输出中不同的语义类型。模态分离不同于离散度，而后者正是标准无标签工具所捕获的内容：微分熵随散布程度增大而上升，与是否碎裂无关；PCA 按方差对方向排序，与势垒无关。互信息确实能够捕获模态分离，但前提是给定潜在的成分标签，而这通常是不可获得的。我们通过密度所固有的单一随机过程来测量模态分离，即以 $f$ 为平稳分布且扩散系数为常数的唯一可逆扩散过程（规范扩散）。从其自协方差矩阵中，我们提取出两个……

    arXiv:2605.08777v2 Announce Type: replace  Abstract: Mode separation, namely how sharply a distribution fragments into barrier-separated clusters, is a fundamental geometric property of densities, difficult to quantify in high dimensions. Modes are often the meaningful alternatives in the data, such as the conformations of a molecule or the distinct semantic types among a generative model's outputs. Mode separation is distinct from dispersion, which is what standard label-free tools capture: differential entropy rises with spread regardless of fragmentation, and PCA orders directions by variance regardless of barriers. Mutual information does capture it, but only given the latent component labels, which are usually unavailable. We measure mode separation through a single stochastic process intrinsic to the density, namely the unique reversible diffusion with $f$ as its stationary distribution and a constant scalar diffusion coefficient. From its autocovariance matrix we extract two rea
    
[^316]: 关于恢复潜在霍克斯网络的最小观测时间

    On Minimal Observation Time for Recovering Latent Hawkes Networks

    [https://arxiv.org/abs/2605.08400](https://arxiv.org/abs/2605.08400)

    本文针对稀疏弱相互作用的平稳霍克斯过程，证明了精确恢复潜在网络所需的最小观测时间为 $\log d$ 量级（其中 $d$ 为实体数量），该时间界既是充分的也是必要的，并提出了达到该界的两阶段估计方法。

    

    工程、社会和自然中相互作用系统的动力学通常在潜在网络上演化，这些潜在网络决定了哪些实体之间可以发生相互作用。我们研究从基于事件的观测中推断这些网络的问题，这类观测自然地出现在金融、地震学和神经科学等领域。尽管已有大量算法方面的工作来处理这一重要问题，但相关的理论结果仍然稀缺。在本文中，我们提出了以下基本问题：为了精确恢复底层网络，人们需要观察动力学多长时间，即所需时间作为相互作用实体数量 $d$ 的函数是多少？对于一类具有稀疏、弱相互作用的平稳霍克斯过程，我们证明了量级为 $\log d$ 的观测时间既是充分的也是必要的。对于上界部分，我们构造了一个两阶段估计器，该估计器使用裁剪和分箱处理的事件数据进行初步筛选，随后进行最小二乘精化，并应用……

    arXiv:2605.08400v2 Announce Type: replace-cross  Abstract: Dynamics of interacting systems in engineering, society, and nature often evolve over latent networks that govern which entities can interact. We study the problem of inferring these networks from event-based observations, which arise naturally in finance, seismology, and neuroscience. While there is substantial algorithmic work addressing this important problem, theoretical results are scarce. In this paper we ask the following fundamental question: what is the minimum time that one must observe the dynamics in order to exactly recover the underlying network, as a function of the number $d$ of interacting entities? For a class of stationary Hawkes processes with sparse, weak interactions, we prove that an observation time of order $\log d$ is sufficient and necessary. For the upper bound we construct a two-stage estimator that uses clipped and binned event data for screening, followed by a least-squares refinement, and apply c
    
[^317]: 当视觉语言模型“纠正”学生时：在多行手写数学OCR评估中识别并惩罚过度纠错

    When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR

    [https://arxiv.org/abs/2604.22774](https://arxiv.org/abs/2604.22774)

    该论文首次系统性研究多行手写数学OCR，揭示了视觉语言模型“过度纠错”从而掩盖学生错误的关键失效模式，并提出了PINK语义评估指标来显式惩罚这种过度纠错行为。

    

    手写数学的准确转录对教育AI系统至关重要，然而当前的基准测试未能正确评估这一能力。以往的大多数研究都集中在单行表达式上，并依赖于BLEU等词汇层面的指标，这些指标无法评估多行学生解答中的语义推理。在本文中，我们首次对多行手写数学光学字符识别（OCR）进行了系统性研究，揭示了视觉语言模型（VLM）的一个关键失效模式：过度纠错。这些模型没有忠实地转录学生的作业，而是常常“修复”错误，从而掩盖了教育评估本应检测出的错误。为解决这一问题，我们提出了PINK（基于INK的惩罚评分），这是一种语义评估指标，它利用大语言模型（LLM）进行基于评分标准的评分，并明确惩罚过度纠错。我们对15个最先进的……

    arXiv:2604.22774v3 Announce Type: replace-cross  Abstract: Accurate transcription of handwritten mathematics is crucial for educational AI systems, yet current benchmarks fail to evaluate this capability properly. Most prior studies focus on single-line expressions and rely on lexical metrics such as BLEU, which fail to assess the semantic reasoning across multi-line student solutions. In this paper, we present the first systematic study of multi-line handwritten math Optical Character Recognition (OCR), revealing a critical failure mode of Vision-Language Models (VLMs): over-correction. Instead of faithfully transcribing a student's work, these models often "fix" errors, thereby hiding the very mistakes an educational assessment aims to detect. To address this, we propose PINK (Penalized INK-based score), a semantic evaluation metric that leverages a Large Language Model (LLM) for rubric-based grading and explicitly penalizes over-correction. Our comprehensive evaluation of 15 state-o
    
[^318]: 通过对齐求解器的初始化学习实现可迁移的SCF加速

    Transferable SCF-Acceleration through Solver-Aligned Initialization Learning

    [https://arxiv.org/abs/2604.21657](https://arxiv.org/abs/2604.21657)

    该论文提出通过对齐求解器的初始化学习（SAIL）端到端地对SCF求解器进行微分训练，揭示了矩阵预测模型外推失效源于监督问题而非外推问题，从而实现可迁移的SCF收敛加速，并引入有效相对迭代次数（ERIC）作为更准确的评估指标。

    

    Kohn-Sham密度泛函理论（KS-DFT）计算的成本与所需求解器迭代次数成正比，而迭代次数取决于初始猜测的质量。从分子几何结构预测初始猜测的机器学习方法可以降低这一成本，但矩阵预测模型在外推到更大分子时会失败，反而减缓而非加速收敛[Liu等人，2025]。我们证明这种失败是一个监督问题，而非外推问题：在基态目标上训练的模型在分布外也能很好地拟合这些目标，但其产生的初始猜测却会减慢收敛速度。对齐求解器的初始化学习（SAIL）通过端到端地对自洽场（SCF）求解器的部分进行微分，为哈密顿量和密度矩阵模型解决了这一问题。我们引入了有效相对迭代次数（ERIC），这是对常用的RIC的一种修正，它考虑了隐藏的

    arXiv:2604.21657v3 Announce Type: replace  Abstract: The cost of Kohn-Sham density functional theory (KS-DFT) calculations scales with the number of required solver iterations, which depend on the quality of the initial guess. Machine learning methods that predict initial guesses from molecular geometry can reduce this cost, but matrix-prediction models fail when extrapolating to larger molecules, degrading rather than accelerating convergence [Liu et al., 2025]. We show that this failure is a supervision problem, not an extrapolation problem: models trained on ground-state targets fit those targets well out of distribution, yet produce initial guesses that slow convergence. Solver-Aligned Initialization Learning (SAIL) resolves this for both Hamiltonian and density matrix models by differentiating through parts of the self-consistent field (SCF) solver end-to-end. We introduce the Effective Relative Iteration Count (ERIC), a correction to the commonly used RIC that accounts for hidden
    
[^319]: 未训练CNN在V1区与反向传播相当：四种学习规则与人脑fMRI的系统性RSA比较

    Untrained CNNs Match Backpropagation at V1: A Systematic RSA Comparison of Four Learning Rules Against Human fMRI

    [https://arxiv.org/abs/2604.16875](https://arxiv.org/abs/2604.16875)

    本研究通过系统性RSA比较发现，未训练的CNN在V1区的内部表征与人类fMRI数据对齐程度与反向传播训练的CNN相当，但此结论受评估分辨率影响，且预测编码和STDP的结果因评估缺陷需修正。

    

    arXiv:2604.16875v3 公告类型：替换 摘要：更正（2026年8月）：本研究的预测编码和STDP条件存在评估模式缺陷；在重新计算之前，不应使用这些结果。在V1区和224像素分辨率下，预测编码的相关系数从rho = 0.056降至0.016，STDP从0.064降至0.037，因此关于STDP在训练规则中领先以及PC和STDP在V1/V2区领先的声明不受支持。随机、反向传播和反馈对齐条件（支持未训练与训练对比的声明）的变化在0.0013以内。然而，结果强烈依赖于评估分辨率，此处固定为224像素；参见arXiv:2608.12408。见第1页更正说明；下方原始摘要和正文与v1版本相同。计算神经科学中的一个核心问题是，用于训练神经网络的学习规则是否决定其内部表示与人脑视觉皮层的对齐程度。我们进行了...

    arXiv:2604.16875v3 Announce Type: replace  Abstract: CORRECTION (August 2026): an evaluation-mode defect affected the predictive-coding and STDP conditions of this study; those results should not be used pending re-computation. At V1 and 224px, predictive coding falls from rho = 0.056 to 0.016 and STDP from 0.064 to 0.037, so the claims that STDP leads among trained rules and that PC and STDP lead at V1/V2 are not supported. The random, backpropagation and feedback-alignment conditions, which carry the untrained-versus-trained claim, are unchanged to within 0.0013. The result is however strongly dependent on the evaluation resolution, held fixed at 224px here; see arXiv:2608.12408. See the correction note on page 1; the original abstract below and the body are unchanged from v1.   A central question in computational neuroscience is whether the learning rule used to train a neural network determines how well its internal representations align with those of the human visual cortex. We pr
    
[^320]: 基于吉布斯先验的物理信息强化学习用于电网拓扑控制

    Physics Informed Reinforcement Learning with Gibbs Priors for Topology Control in Power Grids

    [https://arxiv.org/abs/2604.01830](https://arxiv.org/abs/2604.01830)

    该论文提出一种将半马尔可夫控制与编码电网物理特性的吉布斯先验相结合的物理信息强化学习框架，利用图神经网络代理预测动作过载风险以缩减候选动作集并重加权策略，从而有效缓解电网拓扑控制中动作空间的组合爆炸与在线仿真成本高昂的难题。

    

    电网运行的拓扑控制是一个具有挑战性的序贯决策问题，因为动作空间会随着电网规模的增大而呈组合式增长，且通过仿真进行动作评估的计算成本高昂。我们提出了一种物理信息强化学习框架，将半马尔可夫控制与作用于动作空间、编码系统物理特性的吉布斯先验相结合。仅在电网进入危险状态时才做出决策，同时由图神经网络代理模型预测可行的拓扑动作在执行后的过载风险。这些预测被用于构建物理信息的吉布斯先验，该先验既能筛选出一个较小的、依赖于状态的候选动作集，又在动作选择之前对策略logits进行重新加权。通过这种方式，我们的方法降低了探索难度和在线仿真成本，同时保留了学习到的策略的灵活性。我们在三个真实的电网环境中对该方法进行了评估。

    arXiv:2604.01830v2 Announce Type: replace  Abstract: Topology control for power grid operation is a challenging sequential decision making problem because the action space grows combinatorially with the size of the grid and action evaluation through simulation is computationally expensive. We propose a physics-informed Reinforcement Learning framework that combines semi-Markov control with a Gibbs prior, that encodes the system's physics, over the action space. The decision is only taken when the grid enters a hazardous regime, while a graph neural network surrogate predicts the post action overload risk of feasible topology actions. These predictions are used to construct a physics-informed Gibbs prior that both selects a small state-dependent candidate set and reweights policy logits before action selection. In this way, our method reduces exploration difficulty and online simulation cost while preserving the flexibility of a learned policy. We evaluate the approach in three realisti
    
[^321]: 具有策略性智能体的分布式随机梯度下降中的梯度操纵：兼具收敛保证的真实激励机制

    Gradient Manipulation in Distributed Stochastic Gradient Descent with Strategic Agents: Truthful Incentives with Convergence Guarantees

    [https://arxiv.org/abs/2603.27962](https://arxiv.org/abs/2603.27962)

    首次提出一种完全分布式的支付机制，在存在策略性智能体可能操纵梯度的分布式随机梯度下降中，同时保证智能体的真实行为与模型训练的精确收敛。

    

    分布式学习因其在可扩展性、隐私保护和容错方面的优势而受到广泛关注。在这种范式下，多个智能体仅通过与邻居交换参数来协同训练一个全局模型。然而，现有分布式学习方法的一个关键弱点在于其隐含假设所有智能体在梯度更新过程中都表现诚实。在现实场景中，这一假设往往不成立，因为自私或策略性的智能体可能出于个人利益的动机操纵梯度，最终损害最终的学习效果。在本工作中，我们提出了一种完全分布式的支付机制，首次在分布式随机梯度下降中同时保证了真实行为和精确收敛。这是一项重大进展，因为它克服了现有协作学习真实性机制的两个主要局限：（

    arXiv:2603.27962v2 Announce Type: replace  Abstract: Distributed learning has gained significant attention due to its advantages in scalability, privacy, and fault tolerance.In this paradigm, multiple agents collaboratively train a global model by exchanging parameters only with their neighbors. However, a key vulnerability of existing distributed learning approaches is their implicit assumption that all agents behave honestly during gradient updates. In real-world scenarios, this assumption often breaks down, as selfish or strategic agents may be incentivized to manipulate gradients for personal gain, ultimately compromising the final learning outcome. In this work, we propose a fully distributed payment mechanism that, for the first time, guarantees both truthful behaviors and accurate convergence in distributed stochastic gradient descent. This represents a significant advancement, as it overcomes two major limitations of existing truthfulness mechanisms for collaborative learning:(
    
[^322]: PRUE：大规模农田边界分割的实用方案

    PRUE: A Practical Recipe for Field Boundary Segmentation at Scale

    [https://arxiv.org/abs/2603.27101](https://arxiv.org/abs/2603.27101)

    本文提出PRUE方法，通过结合U-Net骨干网络、复合损失函数和针对性数据增强，在FTW基准上实现了76% IoU和47% object-F1，为大规模农田边界分割提供了比实例分割模型和地理空间基础模型更实用的解决方案。

    

    大规模的农田边界地图对于农业监测任务至关重要。现有的基于卫星的农田制图深度学习方法对光照、空间尺度和地理位置变化较为敏感。我们使用Fields of The World（FTW）基准，首次对用于全球农田边界勾绘的分割模型和地理空间基础模型（GFMs）进行了系统性评估。我们在统一的实验设置下评估了18个模型，结果表明U-Net语义分割模型在一系列性能和部署指标上优于基于实例的模型和GFM等替代方案。我们提出了一种新的分割方法，该方法结合了U-Net骨干网络、复合损失函数和有针对性的数据增强，以提升真实世界条件下的性能和鲁棒性。我们的模型在FTW上取得了76%的IoU和47%的object-F1，较之前的基线分别提升了6%和9%。

    arXiv:2603.27101v2 Announce Type: replace-cross  Abstract: Large-scale maps of field boundaries are essential for agricultural monitoring tasks. Existing deep learning approaches for satellite-based field mapping are sensitive to illumination, spatial scale, and changes in geographic location. We conduct the first systematic evaluation of segmentation and geospatial foundation models (GFMs) for global field boundary delineation using the Fields of The World (FTW) benchmark. We evaluate 18 models under unified experimental settings, showing that a U-Net semantic segmentation model outperforms instance-based and GFM alternatives on a suite of performance and deployment metrics. We propose a new segmentation approach that combines a U-Net backbone, composite loss functions, and targeted data augmentations to enhance performance and robustness under real-world conditions. Our model achieves a 76% IoU and 47% object-F1 on FTW, an increase of 6% and 9% over the previous baseline. Our approac
    
[^323]: 以宽度和深度刻画深度ReLU神经网络的记忆容量

    Memorization capacity of deep ReLU neural networks characterized by width and depth

    [https://arxiv.org/abs/2603.09589](https://arxiv.org/abs/2603.09589)

    本文突破了以往仅用参数或神经元数量刻画记忆容量的做法，通过同时调节宽度与深度构造ReLU网络，证明满足 $W^2L^2\log W=\mathcal{O}(N(\log(\delta^{-1})+\log C))$ 即可记忆任意数据集，并给出显式依赖数据间距与类别数的宽度-深度下界，在典型情形下达到 $W^2L^2\asymp N$ 的最优规模。

    

    本文研究了具有ReLU激活函数的深度神经网络的记忆容量。具体而言，我们研究了此类网络记忆任意 $N$ 个数据点（位于单位球内、两两间距为 $\delta$、标签来自大小为 $C$ 的有限集合）所需的最小网络规模。以往的大多数研究通过参数数量或神经元数量来刻画记忆容量。我们通过构造具有可调宽度 $W$ 和深度 $L$ 的ReLU网络扩展了这些结果，在满足 $W^2L^2\log W=\mathcal{O}(N(\log(\delta^{-1})+\log C))$ 的条件下，该网络可以记忆任何此类数据集。我们还建立了显式依赖于 $\delta$ 和 $C$ 的宽度-深度下界，适用于能够记忆所有此类数据集的网络。当 $\log(\delta^{-1})\asymp\log N$ 时，我们的构造在合适的宽度和深度选择下可达到 $W^2L^2\asymp N$。若 $\log C\asymp\log N$，则该阶数是最优的；若 $\log C=o(\log N)$，则该阶数在相差对数因子的意义下是最优的。

    arXiv:2603.09589v2 Announce Type: replace  Abstract: This paper studies the memorization capacity of deep neural networks with ReLU activation. Specifically, we investigate the minimal size of such networks to memorize any $N$ data points in the unit ball with pairwise separation distance $\delta$ and labels from a finite set of size $C$. Most prior studies characterize the memorization capacity by the number of parameters or neurons. We extend these results by constructing ReLU networks with adjustable width $W$ and depth $L$ satisfying $W^2L^2\log W=\mathcal{O}(N(\log(\delta^{-1})+\log C))$ that can memorize any such dataset. We also establish width-depth lower bounds that depend explicitly on $\delta$ and $C$ for networks capable of memorizing all such datasets. When $\log(\delta^{-1})\asymp\log N$, our construction achieves $W^2L^2\asymp N$ for suitable choices of width and depth. This order is optimal if $\log C\asymp\log N$, and optimal up to logarithmic factors if $\log C=o(\log
    
[^324]: HyperKKL：通过动态权重调节实现非自治系统的状态估计

    HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning

    [https://arxiv.org/abs/2602.22630](https://arxiv.org/abs/2602.22630)

    本文提出HyperKKL，利用超网络架构根据外生输入信号动态生成KKL观测器参数，使基于学习的观测器无需重新训练即可泛化到非自治非线性系统。

    

    本文提出了HyperKKL，一种用于设计非自治非线性系统的Kazantzis-Kravaris/Luenberger（KKL）观测器的新型学习方法。尽管KKL观测器通过将非线性动力学浸入一个稳定的线性潜空间而提供了严谨的理论框架，但其实际实现依赖于求解在解析上难以处理的偏微分方程（PDE）。现有的基于学习的KKL观测器近似方法大多针对自治系统设计，在不进行昂贵的重新训练或在线梯度更新的情况下，无法泛化到受外部驱动的动力学系统。HyperKKL通过采用超网络架构来解决这一问题：该架构对外生输入信号进行编码，从而即时生成KKL观测器的参数，有效地学习了一族由外部驱动信号参数化的浸入映射。我们针对课程学习策略对该方法进行了严格的评估……

    arXiv:2602.22630v4 Announce Type: replace-cross  Abstract: This paper proposes HyperKKL, a novel learning approach for designing Kazantzis-Kravaris/Luenberger (KKL) observers for non-autonomous nonlinear systems. While KKL observers offer a rigorous theoretical framework by immersing nonlinear dynamics into a stable linear latent space, its practical realization relies on solving Partial Differential Equations (PDE) that are analytically intractable. Current existing learning-based approximations of the KKL observer are mostly designed for autonomous systems, failing to generalize to driven dynamics without expensive retraining or online gradient updates. HyperKKL addresses this by employing a hypernetwork architecture that encodes the exogenous input signal to instantaneously generate the parameters of the KKL observer, effectively learning a family of immersion maps parameterized by the external drive. We rigorously evaluate this approach against a curriculum learning strategy that a
    
[^325]: 差分隐私非凸分布鲁棒优化

    Differentially Private Non-convex Distributionally Robust Optimization

    [https://arxiv.org/abs/2602.16155](https://arxiv.org/abs/2602.16155)

    本文对具有ψ-散度和非凸损失的差分隐私分布鲁棒优化（DP-DRO）进行了全面系统的研究，通过将带有不确定性约束的极小极大问题重构为最小化问题并设计新颖算法，弥合了隐私保护与鲁棒优化之间的研究空白。

    

    现实世界的部署经常面临分布偏移、群体不平衡和对抗性扰动，在这些情况下，传统的经验风险最小化（ERM）框架可能会严重退化。分布鲁棒优化（DRO）通过在分布的不确定性集合上优化最坏情况下的期望损失来解决这一问题，为鲁棒性提供了一种有原则的方法。同时，由于DRO中的训练数据总是涉及敏感信息，在差分隐私（DP）下保护其免受泄露至关重要。与经典的DP-ERM相比，DP-DRO由于其带有不确定性约束的极小极大优化结构而受到的关注要少得多。为了弥合这一差距，我们对具有ψ-散度和非凸损失的差分隐私（有限和）DRO进行了全面研究。首先，我们通过将DRO重新表述为一个最小化问题来研究具有一般ψ-散度的DRO，并开发了一种新颖的…

    arXiv:2602.16155v2 Announce Type: replace  Abstract: Real-world deployments routinely face distribution shifts, group imbalances, and adversarial perturbations, under which the traditional Empirical Risk Minimization (ERM) framework can degrade severely.   Distributionally Robust Optimization (DRO) addresses this issue by optimizing the worst-case expected loss over an uncertainty set of distributions, offering a principled approach to robustness.   Meanwhile, as training data in DRO always involves sensitive information, safeguarding it against leakage under Differential Privacy (DP) is essential.   In contrast to classical DP-ERM, DP-DRO has received much less attention due to its minimax optimization structure with uncertainty constraint.   To bridge the gap, we provide a comprehensive study of DP-(finite-sum)-DRO with $\psi$-divergence and non-convex loss.   First, we study DRO with general $\psi$-divergence by reformulating it as a minimization problem, and develop a novel $(\vare
    
[^326]: GPTZero：对大语言模型生成文本的鲁棒检测

    GPTZero: Robust Detection of LLM-Generated Texts

    [https://arxiv.org/abs/2602.13042](https://arxiv.org/abs/2602.13042)

    GPTZero 5.0通过分层多任务架构和多层自动化红队测试，在11个第三方基准和18个写作领域上实现了最先进的AI文本检测准确率，并能鲁棒地抵御对抗攻击与改写。

    

    虽然历史上对文本真实性的考量主要围绕抄袭展开，但大语言模型（LLM）的出现带来了新的挑战：区分人类撰写的文本与AI生成的文本。这一转变引发了重大担忧，包括对技能评估的破坏、低质量内容的大规模生产以及错误信息的泛滥。为应对这些问题，我们推出了GPTZero 5.0，一种最先进的工业级AI检测解决方案，能够在人类文本与LLM生成的文本之间实现可靠判别。我们的主要贡献包括：引入分层多任务架构，实现对人类与AI文本的灵活分类；在11个独立第三方基准和18个写作领域上确立了AI检测准确率的新技术水平；并通过多层级自动化红队测试，实现了对对抗性攻击和文本改写的卓越鲁棒性。

    arXiv:2602.13042v2 Announce Type: replace  Abstract: While historical considerations surrounding text authenticity revolved primarily around plagiarism, the advent of large language models (LLMs) has introduced a new challenge: distinguishing human-authored from AI-generated text. This shift raises significant concerns, including the undermining of skill evaluations, the mass-production of low-quality content, and the proliferation of misinformation. Addressing these issues, we introduce GPTZero 5.0, a state-of-the-art industrial AI detection solution, offering reliable discernment between human and LLM-generated text. Our key contributions include: introducing a hierarchical, multi-task architecture enabling a flexible taxonomy of human and AI texts, setting a new state-of-the-art for AI detection accuracy on 11 independent third-party benchmarks and 18 writing domains, and achieving superior robustness to adversarial attacks and paraphrasing via multi-tiered automated red teaming. GP
    
[^327]: 用于低维动力学数据驱动建模的时滞Transformer

    Time-Delayed Transformers for Data-Driven Modeling of Low-Dimensional Dynamics

    [https://arxiv.org/abs/2602.08478](https://arxiv.org/abs/2602.08478)

    该论文提出了一种极简的单层单头时滞Transformer（TD-TF），可视为时滞动态模态分解（TD-DMD）的非线性推广，在保持低计算复杂度的同时，在非线性和混沌低维动力学的长期预测中显著优于线性基线方法。

    

    我们提出了时滞Transformer（TD-TF），这是一种用于非定常时空动力学数据驱动建模的简化Transformer架构。TD-TF在基于线性算子的方法与深度序列模型之间架起了桥梁，证明单层、单头的Transformer可以被解释为时滞动态模态分解（TD-DMD）的非线性推广。该架构刻意保持极简，仅由一个自注意力层（每次预测只使用单个查询）和一个前馈层组成，从而使计算复杂度随序列长度线性增长，且参数量很小。数值实验表明，TD-TF在近线性系统上与强线性基线方法的性能相当，而在非线性和混沌状态下则显著优于这些基线，并能准确捕捉长期动力学特性。在合成信号、非定常空气动力学、Lorenz '63系统等验证研究中的结果表明……

    arXiv:2602.08478v2 Announce Type: replace  Abstract: We propose the time-delayed transformer (TD-TF), a simplified transformer architecture for data-driven modeling of unsteady spatio-temporal dynamics. TD-TF bridges linear operator-based methods and deep sequence models by showing that a single-layer, single-head transformer can be interpreted as a nonlinear generalization of time-delayed dynamic mode decomposition (TD-DMD). The architecture is deliberately minimal, consisting of one self-attention layer with a single query per prediction and one feedforward layer, resulting in linear computational complexity in sequence length and a small parameter count. Numerical experiments demonstrate that TD-TF matches the performance of strong linear baselines on near-linear systems, while significantly outperforming them in nonlinear and chaotic regimes, where it accurately captures long-term dynamics. Validation studies on synthetic signals, unsteady aerodynamics, the Lorenz '63 system, and a
    
[^328]: 超越成功率：离线目标条件强化学习的可训练性与可提取性

    Beyond Success Rates: Trainability and Extractability for Offline GCRL

    [https://arxiv.org/abs/2602.05459](https://arxiv.org/abs/2602.05459)

    该论文指出仅用成功率评估离线GCRL算法并不充分，通过构建学习率与策略提取超参数交互的地形图，揭示了不同算法（如QRL、CRL）在超参数敏感性上存在巨大且依赖任务的差异。

    

    离线目标条件强化学习（GCRL）方法通常通过优化超参数配置后的成功率来进行基准评估。然而，成功率本身并不能提供算法对价值学习或策略提取超参数敏感性的信息，因为一个算法可能在很宽的超参数范围内都取得相同的成功率，也可能只在特定的设置下才能成功。我们在共享策略提取器的条件下，研究了离线GCRL算法GCIQL、GCIVL、QRL和CRL中的这一现象，通过构建学习率与策略提取超参数（如优势加权回归（AWR）的温度）之间交互作用的地形图。这些地形图揭示了超参数敏感性存在巨大的、依赖于任务的差异。在AntMaze-M任务上，59%的QRL配置和66%的CRL配置达到了该方法最佳采样成功率的至少90%……（摘要在此处被截断）

    arXiv:2602.05459v3 Announce Type: replace  Abstract: Offline goal-conditioned reinforcement learning (GCRL) methods are typically benchmarked by the success rates of optimized hyperparameter configurations. However, success rate by itself does not provide information about the sensitivity of algorithms to hyperparameters for value learning or policy extraction, since an algorithm could obtain the same success rate across a wide range of hyperparameters, or might succeed only on specific settings. We study this phenomenon in the offline GCRL algorithms GCIQL, GCIVL, QRL, and CRL under a shared policy extractor, by constructing landscapes of interaction between the learning rate and policy extraction hyperparameters such as the temperature of advantage-weighted regression (AWR). These landscapes reveal large, task-dependent differences in hyperparameter sensitivity. On AntMaze-M, 59% of QRL configurations and 66% of CRL configurations achieve at least 90% of the method's best sampled suc
    
[^329]: 智能体AI赋能的动态综述框架

    Agentic AI-Empowered Dynamic Survey Framework

    [https://arxiv.org/abs/2602.04071](https://arxiv.org/abs/2602.04071)

    该论文提出一种智能体AI驱动的动态综述框架，将综述视为需持续维护的“活的文档”，能增量式整合新研究并保持综述结构与连贯性。

    

    综述论文在综合与组织科学知识方面发挥着核心作用，然而其日益受到研究成果快速增长的挑战。由于发表后新工作不断涌现，综述很快变得过时，导致文献中的冗余与碎片化。我们将综述写作重新定义为一个长周期的维护问题，而非一次性生成任务，将综述视为随其所描述的研究共同演进的“活的文档”。我们提出了一种智能体化的动态综述框架，通过增量式整合新工作来支持对现有综述论文的持续更新，同时保持综述结构并将不必要的干扰降到最低。借助回顾性实验设置，我们证明所提出的框架能够有效识别并纳入新兴研究，同时保持现有综述的连贯性与结构，即使在较长的（时间跨度内）……

    arXiv:2602.04071v2 Announce Type: replace  Abstract: Survey papers play a central role in synthesizing and organizing scientific knowledge, yet they are increasingly strained by the rapid growth of research output. As new work continues to appear after publication, surveys quickly become outdated, contributing to redundancy and fragmentation in the literature. We reframe survey writing as a long-horizon maintenance problem rather than a one-time generation task, treating surveys as living documents that evolve alongside the research they describe. We propose an agentic Dynamic Survey Framework that supports the continuous updating of existing survey papers by incrementally integrating new work while preserving survey structure and minimizing unnecessary disruption. Using a retrospective experimental setup, we demonstrate that the proposed framework effectively identifies and incorporates emerging research while preserving the coherence and structure of existing surveys, even over exten
    
[^330]: 一种“择优挑选”方法用于大规模负荷整形以实现更有效的碳减排

    A Cherry-Picking Approach to Large Load Shaping for More Effective Carbon Reduction

    [https://arxiv.org/abs/2601.17990](https://arxiv.org/abs/2601.17990)

    本文通过构建近8900节点的ERCOT日前市场出清流程校准复现模型进行全年因果调度反事实分析，发现基于节点边际电价的负荷整形在年度电网碳减排中优于其他静态信号，但没有单一信号能在所有运行条件下均表现最佳，因而提出择优选择信号的负荷整形策略以实现更有效的碳减排。

    

    对数据中心等多兆瓦级柔性负荷进行整形会改变电网上的发电机调度，从而影响整个系统的二氧化碳排放和批发电力成本。然而，评估常见的负荷整形信号——例如电网平均碳强度、节点边际电价（LMP）、可再生能源发电量或商业边际排放估算——一直十分困难，因为完整规模的市场出清模型和发电机投标数据在系统运营商之外很少能够获得。本文利用经过校准的、近8900节点规模的ERCOT日前安全约束机组组合（SCUC）与安全约束经济调度（SCED）市场出清流程复现模型，在可再生能源丰富区和化石能源主导区开展了为期一年的因果调度反事实分析。我们发现，虽然基于LMP的负荷整形在年度电网CO2减排方面优于其他静态信号，但没有任何单一信号能在所有运行条件下均表现最佳。

    arXiv:2601.17990v2 Announce Type: replace-cross  Abstract: Shaping multi-megawatt flexible loads such as data centers alters generator dispatch on the electric grid, affecting system-wide CO2 emissions and wholesale electricity costs. However, evaluating prevalent load-shaping signals - such as grid-average carbon intensity, locational marginal price, renewable generation, or commercial marginal emission estimates - has remained difficult because full-scale market-clearing models and generator bid data are rarely accessible outside system operators. Here we conduct year-long causal dispatch counterfactuals using a calibrated, nearly 8,900-bus replica of ERCOT's day-ahead Security Constrained Unit Commitment and Security Constrained Economic Dispatch market-clearing process across renewable-rich and fossil-dominated zones. We find that while LMP-based shaping outperforms other static signals in annual grid CO2 reductions, no single signal performs best across all operating conditions. M
    
[^331]: 论在个体损失准备金评估的神经网络中使用个案估计与交易赔付数据

    On the use of case estimate and transactional payment data in neural networks for individual loss reserving

    [https://arxiv.org/abs/2601.05274](https://arxiv.org/abs/2601.05274)

    该论文比较了前馈神经网络（使用汇总交易数据）与循环神经网络（使用完整赔付历史及个案估计发展历史）在个体损失准备金评估中的表现，发现在协变量数据有限的情况下，个案估计能显著提升预测效果，而赋予神经网络记忆机制的额外收益则相对有限。

    

    在精算准备金文献中，使用基于个体索赔数据训练的神经网络已日益流行。我们研究了如何在神经网络模型中最优地输入历史赔付数据。此外，个案估计也以时间序列的形式存在，我们进一步扩展了分析以评估其预测能力。在本文中，我们将基于汇总交易数据训练的前馈神经网络与能够分析索赔完整赔付历史和/或个案估计发展历史的循环神经网络进行了比较。我们通过在由SPLICE（Avanzi、Taylor和Wang，2023）模拟出的多个可比较的高度复杂数据集上训练并比较各模型的性能，从而得出结论。在协变量数据有限的环境中，我们发现有证据表明个案估计能够显著改善预测效果，但为神经网络配备记忆机制只会带来……（原文摘要至此处被截断）

    arXiv:2601.05274v2 Announce Type: replace-cross  Abstract: The use of neural networks trained on individual claims data has become increasingly popular in the actuarial reserving literature. We consider how to best input historical payment data in neural network models. Additionally, case estimates are also available in the format of a time series, and we extend our analysis to assessing their predictive power. In this paper, we compare a feed-forward neural network trained on summarised transactions to a recurrent neural network equipped to analyse a claim's entire payment history and/or case estimate development history. We draw conclusions from training and comparing the performance of the models on multiple, comparable highly complex datasets simulated from SPLICE (Avanzi, Taylor and Wang, 2023). In environments where covariate data is limited, we find evidence that case estimates will improve predictions substantially, but that equipping the neural network with memory only leads t
    
[^332]: CORE：面向概念的强化学习方法，弥合数学推理中定义与应用的鸿沟

    CORE: Concept-Oriented Reinforcement for Bridging the Definition-Application Gap in Mathematical Reasoning

    [https://arxiv.org/abs/2512.18857](https://arxiv.org/abs/2512.18857)

    CORE框架将显式概念转化为可控的强化学习监督信号，通过合成概念对齐测验、在推理中注入概念片段等方式，弥合了大语言模型“能复述定义却不会应用概念”的数学推理鸿沟。

    

    大型语言模型（LLM）往往能够解答有难度的数学练习题，但在需要真正理解概念时却无法正确运用。流行的可验证奖励强化学习（RLVR）流程只强化最终答案，却几乎不提供细粒度的概念信号，因此模型提升的是模式复用能力而非概念应用能力。我们提出了CORE（Concept-Oriented REinforcement，面向概念的强化），这是一个将显式概念转化为可控监督信号的强化学习训练框架。基于一份高质量、低污染的教科书资源（该资源将可验证的练习题与简洁的概念描述关联起来），我们进行了一项健全性探测，结果表明LLM能够复述定义，却在概念关联测验上失败，从而量化了概念推理的差距。随后，CORE（i）合成概念对齐的测验题，（ii）在推理过程中注入简短的概念片段以引出概念激发的轨迹，（iii）强化…

    arXiv:2512.18857v4 Announce Type: replace  Abstract: Large language models (LLMs) often solve challenging math exercises yet fail to apply the concept right when the problem requires genuine understanding. Popular Reinforcement Learning with Verifiable Rewards (RLVR) pipelines reinforce final answers but provide little fine-grained conceptual signal, so models improve at pattern reuse rather than conceptual applications. We introduce CORE (Concept-Oriented REinforcement), an RL training framework that turns explicit concepts into a controllable supervision signal. Starting from a high-quality, low-contamination textbook resource that links verifiable exercises to concise concept descriptions, we run a sanity probe showing LLMs can restate definitions but fail concept-linked quizzes, quantifying the conceptual reasoning gap. CORE then (i) synthesizes concept-aligned quizzes, (ii) injects brief concept snippets during rollouts to elicit concept-primed trajectories, and (iii) reinforces c
    
[^333]: 迈向AI流行病学：用于前瞻性风险检测的测量标准化框架

    Toward AI epidemiology: a measurement standardization framework for prospective risk detection

    [https://arxiv.org/abs/2512.15783](https://arxiv.org/abs/2512.15783)

    本文提出一个测量标准化框架，将专家-AI交互压缩为结构化、可比较的字段，无需访问模型内部即可对已部署的AI系统进行前瞻性风险检测，并支撑可靠性与治理两方面的主张。

    

    本文提出了一种测量标准化框架，该框架将专家-AI交互压缩为结构化、可比较的字段，用于在已部署AI系统中进行前瞻性风险检测，且无需访问模型内部结构。这篇概念论文从语义和统计角度定义了该框架的范围，并规定了其实证测试的协议。因此，该框架旨在支持的群体层面的主张属于一个分阶段的研究计划，而非本文所声称的研究结果。测量标准化支撑三项主张。第一项是可靠性主张：在有限的条件下，大语言模型能够对专家-AI交互的证据一致性和政策一致性做出可靠、标准化的评估。第二项是治理主张：一致性分数在部署期间为专家提供即时信号，并为机构提供了跨任务类型、模型和领域监测一致性模式的基础。

    arXiv:2512.15783v5 Announce Type: replace  Abstract: This paper proposes a measurement standardization framework that compresses expert-AI interactions into structured, comparable fields for prospective risk detection in deployed AI systems, without access to model internals. This concept paper defines the framework's scope, semantically and statistically, and specifies a protocol for its empirical testing. The population-level claims it is designed to support therefore belong to a staged research program rather than to results claimed here. Measurement standardization underpins three claims. The first is a reliability claim: under bounded conditions, large language models can produce reliable, standardized assessments of the evidential and policy alignment of expert-AI interactions. The second is a governance claim: alignment scores give experts an immediate signal during deployment and give institutions a basis for monitoring alignment patterns across mission types, models, and domai
    
[^334]: 功能压缩作为奥卡姆剃刀：由敏感度最小化导出的代数复杂度

    Functional Compression as Occam's Razor: Algebraic Complexity from Sensitivity Minimization

    [https://arxiv.org/abs/2511.23152](https://arxiv.org/abs/2511.23152)

    该论文为代数补全分解模型建立了以参数敏感度下确界度量的普适复杂度下界，证明达到该下界会迫使潜在表示几何对齐且幺正，并使模型通过共享代数组合（功能压缩）而非低秩表示来学习到全局统一的代数结构。

    

    从数据中发现离散代数结构是机器学习中的一个根本性挑战。我们探讨连续的学习目标如何能够偏好功能压缩——即通过共享的代数组合而非低秩表示来表征数据。我们分析了用于代数补全（恢复有限二元运算）的规范分解模型的全局最优解，该模型已被证明对群运算具有偏好。我们以精确重构条件下参数敏感度的下确界来度量代数复杂度。我们建立了一个普适的复杂度下界（地板），达到该下界会迫使潜在表示具有几何对齐性与幺正性。在该下界处，输入表示组合成相应的输出表示；这些表示在整个运算表中的复用将局部恒等式传播开来，从而强制形成全局共享的代数结构。若下界未被达到，则存在一个正的间隙将……（摘要在此处截断）

    arXiv:2511.23152v5 Announce Type: replace  Abstract: Discovering discrete algebraic structures from data is a fundamental challenge in machine learning. We ask how continuous learning objectives can favor functional compression---representing data through shared algebraic composition rather than low-rank representations. We analyze the global optima of a canonical factorization model for algebraic completion (recovering finite binary operations) that demonstrated a preference for group operations. We measure algebraic complexity by the infimum of parameter sensitivity under exact reconstruction. We establish a universal complexity floor, whose attainment forces geometric alignment and unitarity of the latent representations. At the floor, input representations compose into the corresponding output representations; their reuse across the operation table propagates these local identities to enforce a globally shared algebraic structure. If the floor is not attained, a positive gap separa
    
[^335]: 无需标注数据的不可压缩流物理约束神经求解器

    A physics-constrained neural solver for incompressible flows without labeled data

    [https://arxiv.org/abs/2511.18820](https://arxiv.org/abs/2511.18820)

    该论文提出了PECANN框架下的不可压缩流物理约束神经求解器，无需标注数据或监督预训练，通过将流动建模为约束优化问题并以严格容差强制满足不可压缩性、动量方程、连续性方程和边界条件，突破了现有物理信息方法雷诺数受限的瓶颈。

    

    我们提出了一种用于不可压缩流动的物理约束神经求解器，它既不需要标注数据，也不需要监督预训练。该求解器以严格的容差强制满足不可压缩条件与边界条件（BCs）。我们认为，现有物理信息方法只能处理较低雷诺数的原因，在于对这些约束的执行不够充分——而传统有限差分和有限体积求解器正是依赖这些约束来工作的。因此，我们在物理与等式约束人工神经网络（PECANN）框架内，将不可压缩流动表述为一个约束优化问题：以动量方程、连续性方程以及在原始变量上作为等式约束施加的边界条件为约束条件，最小化压力泊松残差，同时采用条件自适应增广拉格朗日方法强制执行每个约束。压力泊松目标函数被证明是至关重要的。在所有基准测试中，将其替换为……（原文摘要在此处截断）

    arXiv:2511.18820v3 Announce Type: replace-cross  Abstract: We present a physics-constrained neural solver for incompressible flow that requires neither labeled data nor supervised pretraining. The solver enforces incompressibility and boundary conditions (BCs) to strict tolerances. We attribute the limited Reynolds numbers of existing physics-informed approaches to insufficient enforcement of these constraints, on which conventional finite-difference and finite-volume solvers depend. We therefore formulate incompressible flow as a constrained optimization problem within the physics- and equality-constrained artificial neural network (PECANN) framework: the pressure Poisson residual is minimized subject to the momentum equations, continuity equation, and BCs imposed as equality constraints on the primitive variables, while a conditionally adaptive augmented Lagrangian method enforces each constraint. The pressure Poisson objective proves critical. Across all benchmarks, replacing it wit
    
[^336]: 借助SURF（替代参考）方法逼近喷注标记的基本极限

    SURFing to the Fundamental Limit of Jet Tagging

    [https://arxiv.org/abs/2511.15779](https://arxiv.org/abs/2511.15779)

    本文提出了SURF（替代参考）验证方法，通过在可解析的替代模型样本上训练目标生成模型来实现精确的Neyman-Pearson检验，据此发现现代喷注标记器可能已接近喷注标记的真实统计极限。

    

    超越通过更好的喷注标记算法来提升搜索和测量灵敏度这一实际目标之外，还存在一个更深层的问题：这些算法的性能上限是什么？具有学习到的似然函数的生成式替代模型为这一问题提供了一种新途径，前提是替代模型能够正确捕捉底层数据分布。在本工作中，我们提出了替代参考（SUrrogate ReFerence，SURF）方法，这是一种验证生成式模型的新途径。该框架通过在另一个易于处理的替代模型（其本身基于真实数据训练）生成的样本上训练目标模型，从而实现精确的Neyman-Pearson检验。我们论证了EPiC-FM生成模型是JetClass喷注数据的有效替代参考，并应用SURF方法表明，现代喷注标记器可能已经运行在接近真实统计极限的状态。相比之下，我们发现自回归GPT模型在物理上不合理地夸大了top喷注与QCD喷注的区分能力。

    arXiv:2511.15779v2 Announce Type: replace-cross  Abstract: Beyond the practical goal of improving search and measurement sensitivity through better jet tagging algorithms, there is a deeper question: what are their upper performance limits? Generative surrogate models with learned likelihood functions offer a new approach to this problem, provided the surrogate correctly captures the underlying data distribution. In this work, we introduce the SUrrogate ReFerence (SURF) method, a new approach to validating generative models. This framework enables exact Neyman-Pearson tests by training the target model on samples from another tractable surrogate, which is itself trained on real data. We argue that the EPiC-FM generative model is a valid surrogate reference for JetClass jets and apply SURF to show that modern jet taggers may already be operating close to the true statistical limit. By contrast, we find that autoregressive GPT models unphysically exaggerate top vs. QCD separation power e
    
[^337]: 基于SGD的留一影响估计中的跨轮次偏差：分析与累加修正

    Cross-Epoch Bias in SGD-based Leave-One-Out Influence: Analysis and Accumulative Correction

    [https://arxiv.org/abs/2510.26185](https://arxiv.org/abs/2510.26185)

    该论文发现多轮次SGD留一影响估计在近似有限删除时忽略了先前删除累积的参数位移会改变后续再次删除同一样本的效果这一跨轮次偏差，并提出利用样本后续出现处Hessian矩阵的累加修正估计器（ACC-SGD-IE），在非凸分析下证明该修正可从有限时域误差界中消除该交互误差项。

    

    训练数据影响估计通过预测移除单个样本所带来的效果，为数据选择与清洗提供支持。标准的多轮次SGD影响估计器会沿记录的优化轨迹汇总一个样本多次重复出现所产生的贡献。当用于近似有限删除时，它忽略了一种交互作用：后续的删除实际上作用于由先前删除所累积产生的参数位移之上。我们提出了累加式SGD影响估计器（ACC-SGD-IE），以刻画先前的删除如何改变再次移除同一样本的效果。该方法利用该样本在后续出现处的Hessian矩阵来修正估计的参数变化。我们的非凸分析表明，这一修正从有限时域误差界中消除了SGD-IE额外的交互贡献项：被忽略的交互会产生与累积参数变化呈线性关系的局部误差，而其余的泰勒（Taylor）展开项……（原文摘要在此处截断）

    arXiv:2510.26185v2 Announce Type: replace-cross  Abstract: Training-data influence estimates support data selection and cleansing by predicting the effects of removing individual examples. The standard multi-epoch SGD-Influence Estimator (SGD-IE) aggregates contributions from a sample's repeated appearances along the recorded trajectory. When used to approximate finite deletion, it omits an interaction: later removals act on the parameter displacement accumulated from earlier removals. We introduce the Accumulative SGD-Influence Estimator (ACC-SGD-IE) to account for how earlier removals change the effect of removing the same sample again. It uses the sample's Hessian at subsequent appearances to correct the estimated parameter change. Our non-convex analysis shows that this correction removes SGD-IE's additional interaction contribution from the finite-horizon error bound: the omitted interaction produces a local error linear in the accumulated parameter change, while the remaining Tay
    
[^338]: 使用带强化学习引导标注的SILVER解释深度强化学习策略：一种面向高维与多动作环境的模型级方法

    Interpret Policies in Deep Reinforcement Learning using SILVER with RL-Guided Labeling: A Model-level Approach to High-dimensional and Multi-action Environments

    [https://arxiv.org/abs/2510.19244](https://arxiv.org/abs/2510.19244)

    提出了一种带强化学习引导标注的增强SILVER框架，通过将RL策略自身的动作输出纳入边界点识别，将基于Shapley回归的策略解释方法扩展到高维和多动作环境。

    

    深度强化学习（RL）取得了卓越的性能，但缺乏可解释性，限制了对策略行为的信任。现有的SILVER框架（Li, Siddique, and Cao 2025）通过基于Shapley的回归来解释RL策略，但仍然局限于低维、二元动作的领域。我们提出了带强化学习引导标注的SILVER（SILVER with RL-guided labeling），这是SILVER的一个增强变体，通过将RL策略自身的动作输出纳入边界点识别，将SILVER扩展到多动作和高维环境中。我们的方法首先从图像观测中提取紧凑的特征表示，进行基于SHAP的特征归因，然后采用强化学习引导标注来生成行为一致的边界数据集。随后训练代理模型（如决策树和基于回归的函数）来解释RL策略的决策结构。我们在两个Atari环境中评估了所提出的框架。

    arXiv:2510.19244v4 Announce Type: replace  Abstract: Deep reinforcement learning (RL) achieves remarkable performance but lacks interpretability, limiting trust in policy behavior. The existing SILVER framework (Li, Siddique, and Cao 2025) explains RL policy via Shapley-based regression but remains restricted to low-dimensional, binary-action domains. We propose SILVER with RL-guided labeling, an enhanced variant that extends SILVER to multi-action and high-dimensional environments by incorporating the RL policy's own action outputs into the boundary points identification. Our method first extracts compact feature representations from image observations, performs SHAP-based feature attribution, and then employs RL-guided labeling to generate behaviorally consistent boundary datasets. Surrogate models, such as decision trees and regression-based functions, are subsequently trained to interpret RL policy's decision structure. We evaluate the proposed framework on two Atari environments u
    
[^339]: 基于快速多极算法启发的分层神经网络学习格林算子

    Learning Greens Operators through Hierarchical Neural Networks Inspired by the Fast Multipole Method

    [https://arxiv.org/abs/2509.20591](https://arxiv.org/abs/2509.20591)

    该论文提出受快速多极算法启发的“神经FMM”分层神经网络架构，通过分离局部与远场相互作用，高效学习椭圆偏微分方程的格林算子。

    

    快速多极算法（FMM）是一种用于计算引力场和静电场中N体问题长程作用力的高效数值算法。该方法利用了底层动力系统固有的格林函数的多极展开。尽管FMM在物理和工程领域应用广泛，但其与现代机器学习架构的融合仍然研究不足。在这项工作中，我们提出了一种新颖的神经网络架构——神经FMM（Neural FMM），它将FMM的信息流整合到分层机器学习框架中，用于学习椭圆偏微分方程的格林算子。我们的神经FMM架构利用FMM方法的分层计算流程，将局部相互作用与远场相互作用分离开来，并高效地学习它们各自的表示。

    arXiv:2509.20591v2 Announce Type: replace  Abstract: The Fast Multipole Method (FMM) is an efficient numerical algorithm for computation of long-ranged forces in $N$-body problems within gravitational and electrostatic fields. This method utilizes multipole expansions of the Green's function inherent to the underlying dynamical systems. Despite its widespread application in physics and engineering, the integration of FMM with modern machine learning architectures remains underexplored. In this work, we propose a novel neural network architecture, the Neural FMM, that integrates the information flow of the FMM into a hierarchical machine learning framework for learning the Green's operator of an Elliptic PDE. Our Neural FMM architecture leverages a hierarchical computation flow of the FMM method to split up the local and far-field interactions and efficiently learn their respective representations.
    
[^340]: JigSLO：解决资源拼图难题，实现高效的多LLM服务

    JigSLO: Solving the Resource Puzzle for Efficient Multi-LLM Serving

    [https://arxiv.org/abs/2509.06261](https://arxiv.org/abs/2509.06261)

    JigSLO通过时间共享内存访问模型解析估计带宽竞争对延迟的影响，并据此管理内存资源、调度prefill与decode阶段，实现高效的GPU空间共享多LLM服务。

    

    GPU空间共享将具有互补需求的实例（如计算受限的预填充prefill与内存受限的解码decode）共置运行，有望实现低成本高效的多LLM服务。先前的系统对计算进行划分并协调共享内存容量，但并未对异构LLM共置实例之间的带宽竞争进行建模。这种竞争使得每个实例的延迟取决于其共同运行实例的内存访问模式，即使其自身的资源分配保持不变。基于大量测量，我们发现带宽竞争对延迟的影响可以通过一个包含竞争期与非竞争期的时间共享内存访问模型来刻画，因此整体延迟可以解析地估计为这些时期延迟估计值的时间加权平均。我们提出JigSLO，一个感知竞争的多LLM服务系统，它管理内存资源并调度每个模型的预填充和解码阶段，以实现高效的共置服务。

    arXiv:2509.06261v3 Announce Type: replace-cross  Abstract: GPU spatial sharing co-locates instances with complementary demands, such as compute-bound prefill and memory-bound decode, promising cost-efficient multi-LLM serving. Prior systems partition compute and coordinate shared memory capacity, but do not model bandwidth contention among co-located instances of heterogeneous LLMs. This contention makes each instance's latency depend on the memory-access patterns of its co-runners, even when its own resource allocation remains unchanged. Drawing on extensive measurements, we find that the latency impact of bandwidth contention can be captured by a time-shared memory-access model with contended and uncontended periods, so the overall latency can be analytically estimated as a time-weighted average of the latency estimates for these periods. We propose JigSLO, a contention-aware multi-LLM serving system that manages memory resources and schedules each model's prefill and decode phases a
    
[^341]: 面向直接空气捕集的可变形金属有机框架中气体吸附的开放数据集

    An open dataset of gas adsorption in deformable metal-organic frameworks for direct air capture

    [https://arxiv.org/abs/2508.03162](https://arxiv.org/abs/2508.03162)

    该论文发布了ODAC25开放数据集，包含近6000万次DFT计算，涵盖CO2、H2O、N2和O2在约15,000种可变形金属有机框架中的吸附，并引入功能化MOFs、高能吸附质构型和合成框架以提升精度与多样性，为直接空气捕集领域的机器学习模型提供高质量训练数据。

    

    构建经济且可扩展的直接空气捕集（DAC）工艺以从潮湿空气中捕集二氧化碳，依赖于原子尺度建模来从广阔的化学空间中搜索并识别有前景的吸附剂材料。机器学习已成为一种高效且准确地预测材料性能的强大工具，但这些模型需要大量经过精心整理的训练数据。我们提出了Open DAC 2025（ODAC25）数据集，包含近6000万次DFT单点计算，涵盖CO2、H2O、N2和O2在约15,000种金属有机框架（MOFs）中的吸附。ODAC25引入了功能化MOFs、高能吸附质放置构型以及合成生成的框架，以最大限度地提升基于该数据集训练的模型的通用性。ODAC25相较于以往工作提升了DFT计算的精度，并应用了近期工作中关于吸附质诱导MOF柔性处理的工程经验。我们发布了最先进的机器学习……

    arXiv:2508.03162v3 Announce Type: replace-cross  Abstract: Engineering economical and scalable processes for direct air capture (DAC) from humid air relies on atomistic modeling to search and identify promising sorbent materials from a vast chemical space. Machine learning has emerged as a powerful tool for predicting material properties in an efficient yet accurate manner, but these models require large sets of carefully curated training data. We present the Open DAC 2025 (ODAC25) dataset comprising nearly 60 million DFT single-point calculations for CO2, H2O, N2, and O2 adsorption in ~15,000 MOFs. ODAC25 introduces functionalized MOFs, high-energy adsorbate placements, and synthetically generated frameworks to maximize the versatility of models trained on the dataset. ODAC25 improves upon the accuracy of DFT calculations in previous works and applies engineering intuition from recent work regarding the treatment of adsorbate-induced MOF flexibility. We release state-of-the-art machin
    
[^342]: QUTCC：面向成像逆问题的分位数不确定性训练与共形校准

    QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems

    [https://arxiv.org/abs/2507.14760](https://arxiv.org/abs/2507.14760)

    提出QUTCC方法，利用带分位数嵌入的U-Net进行同时分位数回归训练，并在共形校准中引入非线性、空间自适应缩放，从而为成像逆问题生成更紧凑且统计有效的不确定性区间。

    

    尽管深度学习为科学和医学成像带来了巨大的前景，但模型的失败与幻觉（与现实不符的预测）难以精确定位，并可能造成严重的下游后果。诸如共形预测之类的不确定性估计技术可以通过为模型预测提供统计上有效的误差棒来缓解这一问题。然而，主流的共形预测方法并非针对高维图像类问题而设计，且在共形校准过程中未考虑图像内部的空间相关性，导致不确定性区间过于宽泛。我们提出了一种实用的同时分位数回归方法，能够在共形校准过程中实现非线性、空间自适应的缩放。我们的方法 QUTCC 使用带有分位数嵌入的 U-Net 架构，在训练阶段学习完整的条件分位数分布，随后利用该……

    arXiv:2507.14760v3 Announce Type: replace-cross  Abstract: While deep learning offers tremendous promise for scientific and medical imaging, any failures and hallucinations (predictions that do not coincide with reality) are hard to pinpoint and can have serious downstream consequences. Uncertainty estimation techniques, such as conformal prediction, can help by predicting statistically valid error bars for a model's prediction. However, popular conformal prediction methods were not designed for high-dimensional image-valued problems and do not take into account spatial correlations within an image during conformal calibration, resulting in larger-than-necessary uncertainty intervals. We propose a practical simultaneous quantile regression method that enables non-linear, spatially-adaptive scaling during conformal calibration. Our method, QUTCC uses a U-Net architecture with a quantile embedding to learn a full conditional quantile distribution during training, and then leverages this 
    
[^343]: Mix-Opt：面向文本到图像扩散模型内存高效个性化的混合优化方法

    Mix-Opt: Mixed Optimization for Memory-Efficient Personalization of Text-to-Image Diffusion Models

    [https://arxiv.org/abs/2507.10029](https://arxiv.org/abs/2507.10029)

    提出Mix-Opt混合优化框架，根据扩散过程特性自适应地在低分辨率反向传播与高分辨率零阶优化之间切换，在边缘设备有限内存下实现文本到图像扩散模型既高效又保持结构一致的个性化。

    

    内存高效的个性化对于在保护用户隐私的同时、在边缘设备有限的计算资源内适配文本到图像扩散模型至关重要。为此，我们提出了Mix-Opt，一种新颖的混合优化框架，它以扩散过程的特性为指导，自适应地在低分辨率图像上的反向传播（BP-low）和高分辨率图像上的零阶优化（ZO-high）之间进行选择。正如我们在实验中观察到的，BP-low能够高效地将模型适配到目标特定特征，但由于分辨率不匹配会导致结构失真。相反，ZO-high能以最小的内存开销细化高分辨率细节，但在没有先验适配的情况下直接应用时收敛缓慢。通过将两种方法互补结合，我们的框架利用BP-low实现有效个性化，同时使用ZO-high保持结构一致性，从而实现内存高效的个性化（原文在此处截断）。

    arXiv:2507.10029v3 Announce Type: replace-cross  Abstract: Memory-efficient personalization is essential for adapting text-to-image diffusion models while preserving user privacy and operating within the limited computational resources of edge devices. To this end, we propose Mix-Opt, a novel mixed optimization framework that adaptively chooses between backpropagation on low-resolution images (BP-low) and zeroth-order optimization on high-resolution images (ZO-high), guided by the characteristics of the diffusion process. As observed in our experiments, BP-low efficiently adapts the model to target-specific features but suffers from structural distortions due to resolution mismatch. Conversely, ZO-high refines high-resolution details with minimal memory overhead but faces slow convergence when applied without prior adaptation. By complementing both methods, our framework leverages BP-low for effective personalization while using ZO-high to maintain structural consistency, achieving mem
    
[^344]: 确定性约束下随机与有限和凸优化的一阶方法

    First-order methods for stochastic and finite-sum convex optimization with deterministic constraints

    [https://arxiv.org/abs/2506.20630](https://arxiv.org/abs/2506.20630)

    该论文提出了针对确定性约束下随机与有限和凸优化的随机一阶方法，能够找到约束违反被确定性界定且期望最优性间隙不超过ε的ε-确定可行随机最优（ε-SFSO）解，克服了以往期望可行解可能存在严重约束违反的问题。

    

    本文研究了一类具有确定性约束的随机与有限和凸优化问题。现有方法通常旨在寻找ε-期望可行随机最优（ε-expected feasible stochastic optimal）解，其中期望约束违反与期望最优性间隙均不超过给定的容忍度ε。然而，在许多实际应用中，约束必须几乎以确定性方式得到满足，这使得此类解由于存在严重违反约束的风险而可能不再适用。为解决这一问题，我们提出了随机一阶方法来寻找ε-确定可行随机最优（ε-SFSO）解，其中约束违反被确定性地限制在ε以内，且期望最优性间隙至多为ε。我们的方法采用加速随机梯度（ASG）方案或改进的方差缩减ASG方案……（原文摘要在此处截断）

    arXiv:2506.20630v2 Announce Type: replace-cross  Abstract: In this paper, we study a class of stochastic and finite-sum convex optimization problems with deterministic constraints. Existing methods typically aim to find an $\epsilon$-$expected\ feasible\ stochastic\ optimal$ solution, in which both the expected constraint violation and the expected optimality gap are within a prescribed tolerance $\epsilon$. However, in many practical applications, constraints must be nearly satisfied with certainty, rendering such solutions potentially unsuitable due to the risk of substantial violations. To address this issue, we propose stochastic first-order methods for finding an $\epsilon$-$surely\ feasible\ stochastic\ optimal$ ($\epsilon$-SFSO) solution, where the constraint violation is deterministically bounded by $\epsilon$ and the expected optimality gap is at most $\epsilon$. Our methods apply an accelerated stochastic gradient (ASG) scheme or a modified variance-reduced ASG scheme $only\ 
    
[^345]: 基于递归学习的虚拟缓冲方法用于解析式全局布局

    Recursive Learning-Based Virtual Buffering for Analytical Global Placement

    [https://arxiv.org/abs/2506.17247](https://arxiv.org/abs/2506.17247)

    提出了首个构建于 OpenROAD 之上的开源学习驱动虚拟缓冲感知解析式全局布局框架 MLBuf-RePlAce，通过高效的递归学习生成式缓冲方法预测缓冲器类型与位置，并在全局布局中解决 ERC 违例问题。

    

    由于现代工艺节点中互连延迟与单元延迟缩放比例的失衡，在物理综合流程中，具有缓冲孔隙率（即单元密度）感知能力的布局对于时序收敛至关重要。然而，现有方法面临两个关键挑战：（i）传统的 van Ginneken-Lillis 风格缓冲方法在全局布局过程中计算开销过高；（ii）基于机器学习的方法（如 BufFormer）缺乏对电气规则检查（ERC）违例的全面考虑，且未能“闭合回路”融入物理设计流程。在这项工作中，我们提出了 MLBuf-RePlAce，这是首个基于学习驱动的、感知虚拟缓冲的解析式全局布局开源框架，构建于 OpenROAD 基础设施之上。MLBuf-RePlAce 采用高效的基于递归学习的生成式缓冲方法来预测缓冲器类型和位置，在全局布局过程中解决 ERC 违例问题。

    arXiv:2506.17247v3 Announce Type: replace-cross  Abstract: Due to the skewed scaling of interconnect versus cell delay in modern technology nodes, placement with buffer porosity (i.e., cell density) awareness is essential for timing closure in physical synthesis flows. However, existing approaches face two key challenges: (i) traditional van Ginneken-Lillis-style buffering approaches are computationally expensive during global placement; and (ii) machine learning-based approaches, such as BufFormer, lack a thorough consideration of Electrical Rule Check (ERC) violations and fail to "close the loop" back into the physical design flow. In this work, we propose MLBuf-RePlAce, the first open-source learning-driven virtual buffering-aware analytical global placement framework, built on top of the OpenROAD infrastructure. MLBuf-RePlAce adopts an efficient recursive learning-based generative buffering approach to predict buffer types and locations, addressing ERC violations during global plac
    
[^346]: 评估与缓解皮肤病变分类中的肤色偏差

    Evaluating and Mitigating Skin Color Bias in Skin Lesion Classificatio

    [https://arxiv.org/abs/2506.11627](https://arxiv.org/abs/2506.11627)

    提出将肤色建模为连续属性并根据其与分类性能的非线性关系为样本加权的方法，在保持分类性能的同时，比传统类别重加权实现了更稳定的公平性改进。

    

    皮肤病变分类器若将肤色视为类别属性，可能会忽视细微肤色变化所导致的性能差异。本研究提出一种方法，将肤色表示为连续属性，并根据肤色与分类性能之间的非线性关系为每个样本分配单独的权重。在两个皮肤病学数据集上的评估表明，公平性指标对肤色的分类方式十分敏感。所提出的方法还展现出比传统类别重加权更稳定的公平性改进，同时总体上保持了分类性能。

    arXiv:2506.11627v2 Announce Type: replace-cross  Abstract: Skin lesion classifiers may overlook performance disparity because of subtle skin tone variations by treating skin tone as a categorical attribute. This study proposes a method that represents skin tone as a continuous attribute and assigns individual weights to each sample based on its nonlinear relationship with classification performance. Evaluations on two dermatological datasets showed that fairness metrics were sensitive to skin-color categorization. The proposed method also demonstrated more stable fairness improvements than conventional categorical reweighting while generally preserving classification performance.
    
[^347]: 用于预测染色体不稳定性的量子认知机器学习

    Quantum Cognition Machine Learning for Forecasting Chromosomal Instability

    [https://arxiv.org/abs/2506.03199](https://arxiv.org/abs/2506.03199)

    本文提出量子认知机器学习（QCML）框架，利用量子力学原理将单细胞形态数据表示为希尔伯特空间中的状态向量，无需人工特征选择即可从转移性乳腺癌患者的循环肿瘤细胞形态中预测染色体不稳定性，并在样本外验证中优于传统机器学习方法。

    

    基于循环肿瘤细胞（CTC）形态学对染色体不稳定性进行准确预测，能够在液体活检诊断的背景下实现对具有高转移潜力CTC的实时检测。然而，由于单细胞数字病理学数据的高维度和复杂性，这构成了重大挑战。在此，我们介绍了量子认知机器学习（QCML）——一种量子启发的计算框架——的应用，用于估计转移性乳腺癌患者CTC中基于形态学预测的染色体不稳定性。QCML利用量子力学原理将数据表示为希尔伯特空间中的状态向量，实现上下文感知的特征建模、降维以及增强的泛化能力，且无需人工精选的特征选择。在样本外验证CTC的测试中，QCML优于传统机器学习方法，取得了更高的

    arXiv:2506.03199v3 Announce Type: replace-cross  Abstract: The accurate prediction of chromosomal instability from the morphology of circulating tumor cells (CTCs) enables real-time detection of CTCs with high metastatic potential in the context of liquid biopsy diagnostics. However, it presents a significant challenge due to the high dimensionality and complexity of single-cell digital pathology data. Here, we introduce the application of Quantum Cognition Machine Learning (QCML), a quantum-inspired computational framework, to estimate morphology-predicted chromosomal instability in CTCs from patients with metastatic breast cancer. QCML leverages quantum mechanical principles to represent data as state vectors in a Hilbert space, enabling context-aware feature modeling, dimensionality reduction, and enhanced generalization without requiring curated feature selection. QCML outperforms conventional machine learning methods when tested on out of sample verification CTCs, achieving higher
    
[^348]: HOPSE：面向组合表示的可扩展高阶位置与结构编码器

    HOPSE: Scalable Higher-Order Positional and Structural Encoder for Combinatorial Representations

    [https://arxiv.org/abs/2505.15405](https://arxiv.org/abs/2505.15405)

    提出 HOPSE 框架，通过摒弃可学习消息传递层、利用 Hasse 图分解，在任意高阶组合域上实现高效、可线性扩展且富有表达力的位置与结构编码，解决了高阶拓扑深度学习的可扩展性难题。

    

    尽管图神经网络（GNNs）已被证明在建模关系型数据方面极为有效，但成对连接无法完全捕捉复杂现实系统中天然存在的多路（multi-way）关系。为应对这一问题，拓扑深度学习（TDL）利用更通用的组合表示——如单纯复形（simplicial complexes）或胞腔复形（cellular complexes）——来容纳高阶交互。现有的拓扑深度学习方法通常通过高阶消息传递（HOMP）来扩展图神经网络，但由于在组合结构上传播消息的复杂度开销巨大，面临着严峻的可扩展性挑战。为克服这一局限，我们提出了 HOPSE（高阶位置与结构编码器），这是一个不依赖可学习消息传递层的框架，它利用 Hasse 图分解在任意高阶域上推导出高效且富有表达力的编码。值得注意的是，HOPSE 的训练和推理随输入规模呈线性扩展。

    arXiv:2505.15405v4 Announce Type: replace  Abstract: While Graph Neural Networks (GNNs) have proven highly effective at modeling relational data, pairwise connections cannot fully capture multi-way relationships naturally present in complex real-world systems. In response to this, Topological Deep Learning (TDL) leverages more general combinatorial representations--such as simplicial or cellular complexes--to accommodate higher-order interactions. Existing TDL methods often extend GNNs through Higher-Order Message Passing (HOMP), but face critical scalability challenges due to the steep complexity overhead of propagating messages through combinatorial structures. To overcome this limitation, we propose HOPSE (Higher-Order Positional and Structural Encoder), a framework free of learnable message passing layers that uses Hasse graph decompositions to derive efficient and expressive encodings over arbitrary higher-order domains. Notably, HOPSE training and inference scale linearly with th
    
[^349]: SiGMA：基于门控消息传递与注意力机制的图级异质性学习

    SiGMA: Learning under Graph-Level Heterogeneity with Gated Message-Passing and Attention

    [https://arxiv.org/abs/2503.00547](https://arxiv.org/abs/2503.00547)

    提出了SiGMA门控混合图神经网络架构，通过可学习的节点级sigmoid门控自适应融合消息传递与注意力机制，以应对不同图需要不同局部与全局计算组合的图级异质性问题。

    

    在图级学习中，图神经网络通常采用固定的计算模板：对每个输入图使用相同的网络深度、相同的局部消息传递算子以及相同的全局注意力机制。然而，同一数据集中的不同图在预测难度上可能存在显著差异，并且可能需要不同比例的局部与非局部计算。我们将这一现象作为图级异质性在分类和回归任务中进行了研究。为解决该问题，我们提出了SiGMA（Sigmoid门控消息传递与注意力），这是一种门控混合图神经网络架构，它在每一层内将消息传递与注意力头相结合。受大语言模型和视觉Transformer中门控注意力与门控计算日益广泛应用启发，SiGMA中的每个头都由一个可学习的节点级sigmoid门进行调制，使模型能够自适应地调节使用局部还是全局机制来完成计算。

    arXiv:2503.00547v2 Announce Type: replace  Abstract: In graph-level learning, graph neural networks are typically applied with a fixed computational template: the same depth, local message-passing operator, and global attention mechanism are used for every input graph. However, different graphs within the same dataset can vary substantially in prediction difficulty and may require different mixtures of local and non-local computation. We study this phenomenon as graph-level heterogeneity in both classification and regression tasks. To address it, we propose SiGMA (Sigmoid Gated Message-Passing and Attention), a gated hybrid graph neural network architecture that combines message-passing and attention heads within each layer. Inspired by the growing use of gated attention and gated computation in large language models and vision transformers, each head in SiGMA is modulated by a learned node-wise sigmoid gate, allowing the model to adaptively regulate which local or global mechanisms ar
    
[^350]: 基于均值测度量化的可扩展概率测度学习

    Scalable Learning from Probability Measures with Mean Measure Quantization

    [https://arxiv.org/abs/2502.04907](https://arxiv.org/abs/2502.04907)

    该论文提出一种均值测度量化方法，将所有输入概率测度近似为共享公共支撑集的K点离散测度，在保证量化测度一致性和下游任务收敛性的同时，大幅降低最优传输计算的时间成本并保持与单独量化相当的性能。

    

    我们考虑以一组概率测度的形式观测数据的统计学习问题。最优传输（OT）是比较和操作此类对象的常用工具，但当测度具有较大支撑集时，其计算成本变得难以承受。我们研究了一种基于量化的方法，将所有输入测度近似为共享公共支撑集的K点离散测度。我们证明了所得量化测度的一致性，并进一步为基于这些量化测度计算的多个OT相关下游任务推导出收敛性保证。在合成数据集和真实数据集上的数值实验表明，所提出的方法在大幅减少运行时间的同时，实现了与单独量化相当的性能。

    arXiv:2502.04907v5 Announce Type: replace-cross  Abstract: We consider statistical learning problems in which data are observed as a set of probability measures. Optimal transport (OT) is a popular tool to compare and manipulate such objects, but its computational cost becomes prohibitive when the measures have large support. We study a quantization-based approach in which all input measures are approximated by $K$-point discrete measures sharing a common support. We establish consistency of the resulting quantized measures. We further derive convergence guarantees for several OT-based downstream tasks computed from the quantized measures. Numerical experiments on synthetic and real datasets demonstrate that the proposed approach achieves performance comparable to individual quantization while substantially reducing runtime.
    
[^351]: 基于Transformer的精确序列插值

    Exact Sequence Interpolation with Transformers

    [https://arxiv.org/abs/2502.02270](https://arxiv.org/abs/2502.02270)

    该论文通过交替使用前馈层与自注意力层并采用低秩参数矩阵，证明了Transformer能够以与输入序列长度无关的复杂度精确插值有限序列数据集。

    

    我们证明了Transformer能够精确插值由 $\mathbb{R}^d$（$d\geq 2$）中有限输入序列及其对应的不超过输入长度的输出序列组成的数据集。具体而言，给定 $N$ 个在 $\mathbb{R}^d$ 中长度任意但有限的序列，以及长度为 $m^1, \dots, m^N \in \mathbb{N}$ 的输出序列，我们构造了一个具有 $\mathcal{O}(\sum_{j=1}^N m^j)$ 个块和 $\mathcal{O}(d \sum_{j=1}^N m^j)$ 个参数的Transformer，可以对该数据集实现精确插值。我们的构造通过交替使用前馈层和自注意力层，并利用自注意力层固有的聚类效应，提供了与输入序列长度无关的复杂度估计。我们新颖的构造方法还在自注意力机制中使用了低秩参数矩阵，这也是实际Transformer实现中的一个常见特征。这些结果首先在hardmax自注意力设定下建立。

    arXiv:2502.02270v4 Announce Type: replace  Abstract: We prove that transformers can exactly interpolate datasets of finite input sequences in $\mathbb{R}^d$, $d\geq 2$, with corresponding output sequences of smaller or equal length. Specifically, given $N$ sequences of arbitrary but finite lengths in $\mathbb{R}^d$ and output sequences of lengths $m^1, \dots, m^N \in \mathbb{N}$, we construct a transformer with $\mathcal{O}(\sum_{j=1}^N m^j)$ blocks and $\smash{\mathcal{O}(d \sum_{j=1}^N m^j)}$ parameters that exactly interpolates the dataset. Our construction provides complexity estimates that are independent of the input sequence length, by alternating feed-forward and self-attention layers and by capitalizing on the clustering effect inherent to the latter. Our novel constructive method also uses low-rank parameter matrices in the self-attention mechanism, a common feature of practical transformer implementations. These results are first established in the hardmax self-attention set
    
[^352]: 基于丰富自我报告数据的AI模拟可提高准确性并减少失真

    AI Simulations Grounded in Rich Self-Reports Increase Accuracy and Reduce Distortion

    [https://arxiv.org/abs/2411.10109](https://arxiv.org/abs/2411.10109)

    该研究通过对1,052名美国成年人开展两小时AI语音深度访谈并结合问卷数据，证明基于丰富自我报告数据（访谈与调查）的AI人类模拟比仅用人口属性提示的模拟更准确，并能显著减少样本层面的八类统计失真。

    

    基于人口统计学提示构建的人类AI模拟常常误判个体，并扭曲样本层面的统计模式。我们检验了丰富的自我报告数据是否能减少这些问题。我们开发了一个AI语音访谈系统，并按照“美国之声项目”的协议对1,052名美国成年人进行了两小时的访谈，这些受访者同时还完成了问卷调查和行为实验。针对每位参与者，我们比较了分别基于以下不同提示生成的模拟结果：人口统计学属性、自我撰写的人物画像、问卷回答、访谈回答，以及问卷与访谈相结合。我们在留出的回答数据上评估了模拟的准确性，并在样本层面测量了八种统计失真：平均值偏差、方差被压缩、回答分布失真、条目间过度相关、态度坍缩为单一维度、意识形态归类、实验效应被放大，以及过度理性。在综合社会调查（GSS）的题目上，仅基于人口统计学的模拟导致了……（原文在此处截断）

    arXiv:2411.10109v4 Announce Type: replace  Abstract: AI simulations of people built from demographic prompts often misjudge individuals and distort sample-level patterns. We test whether rich self-report data reduce these problems. We developed an AI voice interviewer and conducted two-hour interviews following the American Voices Project protocol with 1,052 U.S. adults, who also completed surveys and behavioral experiments. For each participant, we compared simulations prompted with demographic attributes, self-authored personas, survey responses, interview responses, or combined surveys and interviews. We evaluated accuracy on held-out responses, and measured eight statistical distortions at the sample level: biased averages, compressed variance, distorted response distributions, over-correlated items, attitudes collapsing onto one dimension, ideological sorting, amplified experimental effects, and hyper-rationality. On General Social Survey (GSS) items, demographic simulation led to
    
[^353]: 纯注意力硬最大值Transformer中的聚类及其在情感分析中的作用

    Clustering in pure-attention hardmax transformers and its role in sentiment analysis

    [https://arxiv.org/abs/2407.01602](https://arxiv.org/abs/2407.01602)

    本文从动力系统角度严格证明了纯注意力硬最大值Transformer在层数趋于无穷时输入会收敛到由“领导者”点决定的聚类平衡态，并据此构建了完全可解释的Transformer模型，通过将无意义词聚类到语义核心词周围来有效捕捉上下文，从而解决情感分析问题。

    

    Transformer是极其成功的机器学习模型，但其数学性质仍未被充分理解。本文严格刻画了具有硬最大值自注意力和归一化子层的Transformer在层数趋于无穷时的行为。通过将此类Transformer视为描述欧几里得空间中点演化的离散时间动力系统，并借助基于超平面分离的自注意力机制的几何解释，我们证明了Transformer的输入渐近收敛到由称为“领导者”的特殊点所决定的聚类平衡态。随后，我们利用这一理论认识，采用完全可解释的Transformer模型来解决语言处理中的情感分析问题，该模型通过将无意义词汇聚类到承载最多语义的领导者词汇周围，有效捕捉了“上下文”。最后，我们概述了仍然存在的（问题）……

    arXiv:2407.01602v3 Announce Type: replace  Abstract: Transformers are extremely successful machine learning models whose mathematical properties remain poorly understood. Here, we rigorously characterize the behavior of transformers with hardmax self-attention and normalization sublayers as the number of layers tends to infinity. By viewing such transformers as discrete-time dynamical systems describing the evolution of points in a Euclidean space, and thanks to a geometric interpretation of the self-attention mechanism based on hyperplane separation, we show that the transformer inputs asymptotically converge to a clustered equilibrium determined by special points called \textit{leaders}. We then leverage this theoretical understanding to solve sentiment analysis problems from language processing using a fully interpretable transformer model, which effectively captures `context' by clustering meaningless words around leader words carrying the most meaning. Finally, we outline remainin
    
[^354]: 高维尾指数回归：以社交媒体病毒帖子文本分析为例

    High-Dimensional Tail Index Regression: with An Application to Text Analyses of Viral Posts in Social Media

    [https://arxiv.org/abs/2403.01318](https://arxiv.org/abs/2403.01318)

    提出了高维尾指数回归方法，利用正则化估计和去偏方法进行推断，支持理论的仿真研究，并在社交媒体病毒帖子文本分析中应用。

    

    受社交媒体病毒帖子的点赞分布（如点赞数量）经验性幂律的启发，我们引入了高维尾指数回归及其参数的估计和推断方法。我们提出了一种正则化估计量，证明了它的一致性，并推导了其收敛速度。为了进行推断，我们提出了去偏正则化估计，证明了去偏估计量的渐近正态性。仿真研究支持了我们的理论。这些方法被应用于对涉及 LGBTQ+ 话题的 X（原 Twitter）病毒帖子的文本分析。

    arXiv:2403.01318v1 Announce Type: cross  Abstract: Motivated by the empirical power law of the distributions of credits (e.g., the number of "likes") of viral posts in social media, we introduce the high-dimensional tail index regression and methods of estimation and inference for its parameters. We propose a regularized estimator, establish its consistency, and derive its convergence rate. To conduct inference, we propose to debias the regularized estimate, and establish the asymptotic normality of the debiased estimator. Simulation studies support our theory. These methods are applied to text analyses of viral posts in X (formerly Twitter) concerning LGBTQ+.
    
[^355]: 纳米机器人队列：基于团队成员通信与图像处理的癌症协同治疗方法

    Nanorobot queue: Cooperative treatment of cancer based on team member communication and image processing

    [https://arxiv.org/abs/2111.11236](https://arxiv.org/abs/2111.11236)

    本文提出了一种基于纳米机器人队列的理想癌症治疗模型，通过团队成员通信与计算机视觉目标检测技术实现纳米机器人的协同治疗，以期彻底治愈癌症。

    

    尽管纳米机器人已被应用于胃镜检查等临床工作，甚至已有光声层析成像技术被提出用于实时控制纳米机器人在指定投放点投放药物，也有通过纳米机器人清除血液中“超级细菌”的案例，但大多数技术尚不成熟，要么效率低、要么精度低，要么无法批量生产，因此现阶段治疗癌症最有效的方式仍是化疗和放疗，患者承受着痛苦且无法被治愈。为此，本文提出了一种能够彻底治愈癌症的治疗方法的理想模型，即通过团队成员通信和计算机视觉图像分类（目标检测）实现的基于纳米机器人队列的协同治疗方法。

    arXiv:2111.11236v4 Announce Type: replace-cross  Abstract: Although nanorobots have been used as clinical prescriptions for work such as gastroscopy, and even photoacoustic tomography technology has been proposed to control nanorobots to deliver drugs at designated delivery points in real time, and there are cases of eliminating "superbacteria" in blood through nanorobots, most technologies are immature, either with low efficiency or low accuracy, Either it can not be mass produced, so the most effective way to treat cancer diseases at this stage is through chemotherapy and radiotherapy. Patients are suffering and can not be cured. Therefore, this paper proposes an ideal model of a treatment method that can completely cure cancer, a cooperative treatment method based on nano robot queue through team member communication and computer vision image classification (target detection).
    
[^356]: 算法保险

    Algorithmic Insurance

    [https://arxiv.org/abs/2106.00839](https://arxiv.org/abs/2106.00839)

    该论文开创了“算法保险”这一新兴领域，通过条件风险价值（CVaR）将二元分类中的决策阈值选择与尾部风险敞口联系起来，证明了准确率最大化等传统方法会显著放大最坏情况损失，并据此提出了将AI系统设计决策与责任风险敞口相挂钩的保险合同框架。

    

    当人工智能（AI）系统在医学诊断或自动驾驶等高风险领域出现错误时，一个在不同情境中相同的单一算法缺陷可能产生异质性的损失，这挑战了传统保险的假设。算法保险，即为AI造成的损害提供财务保障的新兴市场正在兴起，然而保险公司在为这类风险定价时面临困难，AI开发者也缺乏将系统设计与责任风险敞口相关联的框架。我们将二元分类中的一个关键操作选择——从业者如何设置决策阈值——与尾部风险敞口联系起来。利用条件风险价值（CVaR）来刻画极端损失，我们证明了准确率最大化等既有方法相对于尾部风险优化会显著增加最坏情况下的损失，且当阈值偏离最优值时，惩罚至少以二次方速度增长。我们提出了一种责任保险合同，要求……（原文摘要在此处截断）

    arXiv:2106.00839v4 Announce Type: replace  Abstract: When artificial intelligence (AI) systems make errors in high-stakes domains such as medical diagnosis or autonomous driving, a single algorithmic flaw across varying contexts can generate heterogeneous losses that challenge traditional insurance assumptions. Algorithmic insurance, financial coverage for AI-induced damages, is an emerging market, yet insurers struggle to price these risks and AI developers lack frameworks linking system design to liability exposure. We connect a key operational choice in binary classification, how practitioners set decision thresholds, to tail risk exposure. Using conditional value-at-risk (CVaR) to capture extreme losses, we prove that established approaches such as accuracy maximization can significantly increase worst-case losses relative to tail risk optimization, with penalties growing at least quadratically as thresholds deviate from optimal. We propose a liability insurance contract that manda
    
[^357]: 在连续治疗下的多重稳健因果中介分析

    Multiply Robust Causal Mediation Analysis with Continuous Treatments

    [https://arxiv.org/abs/2105.09254](https://arxiv.org/abs/2105.09254)

    本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。

    

    在许多应用中，研究人员对治疗或暴露对感兴趣的结果的直接和间接的因果效应。中介分析为鉴定和估计这些因果效应提供了一个严谨的框架。对于二元治疗，Tchetgen Tchetgen和Shpitser (2012)提出了直接和间接效应的高效估计器，基于参数的影响函数。这些估计器具有良好的性质，如多重稳健性和渐近正态性，同时允许对干扰参数进行低于根号n的收敛速度。然而，在涉及连续治疗的情况下，这些基于影响函数的估计器没有准备好应用，除非进行强参数假设。在这项工作中，我们利用核平滑方法提出了一种适用于连续治疗环境的估计器，受到Tchetgen Tchetgen的影响函数估计器的启发。

    In many applications, researchers are interested in the direct and indirect causal effects of a treatment or exposure on an outcome of interest. Mediation analysis offers a rigorous framework for identifying and estimating these causal effects. For binary treatments, efficient estimators for the direct and indirect effects are presented in Tchetgen Tchetgen and Shpitser (2012) based on the influence function of the parameter of interest. These estimators possess desirable properties, such as multiple-robustness and asymptotic normality, while allowing for slower than root-n rates of convergence for the nuisance parameters. However, in settings involving continuous treatments, these influence function-based estimators are not readily applicable without making strong parametric assumptions. In this work, utilizing a kernel-smoothing approach, we propose an estimator suitable for settings with continuous treatments inspired by the influence function-based estimator of Tchetgen Tchetgen an
    
[^358]: 优化器的信息准则：剖析和纠正数据驱动优化中的偏差

    Optimizer's Information Criterion: Dissecting and Correcting Bias in Data-Driven Optimization. (arXiv:2306.10081v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.10081](http://arxiv.org/abs/2306.10081)

    这项研究提出了一个称为优化器的信息准则(OIC)的通用偏差校正方法，帮助解决数据驱动优化中的乐观偏差问题。该方法直接近似一阶偏差，并且不需要解决额外的优化问题，是在决策选择方面的一个创新。

    

    在数据驱动的优化中，所得决策的样本表现通常存在着对真实表现的乐观偏差，这种现象通常被称为优化器的诅咒，与机器学习中的过拟合密切相关。传统的纠正这种偏差的技术，如交叉验证，需要反复解决额外的优化问题，因此计算代价很高。我们开发了一种通用的偏差校正方法，建立在我们称之为优化器的信息准则（OIC）的基础上，直接近似一阶偏差，不需要解决任何额外的优化问题。我们的OIC将著名的赤池信息准则推广到数据驱动优化中，关键是评估客观表现，不仅涉及模型拟合，还涉及其与下游优化的相互作用。因此，它可以用于决策选择而不仅仅是模型选择。我们将我们的方法应用于一系列问题的数据驱动优化。

    In data-driven optimization, the sample performance of the obtained decision typically incurs an optimistic bias against the true performance, a phenomenon commonly known as the Optimizer's Curse and intimately related to overfitting in machine learning. Common techniques to correct this bias, such as cross-validation, require repeatedly solving additional optimization problems and are therefore computationally expensive. We develop a general bias correction approach, building on what we call Optimizer's Information Criterion (OIC), that directly approximates the first-order bias and does not require solving any additional optimization problems. Our OIC generalizes the celebrated Akaike Information Criterion to evaluate the objective performance in data-driven optimization, which crucially involves not only model fitting but also its interplay with the downstream optimization. As such it can be used for decision selection instead of only model selection. We apply our approach to a rang
    

