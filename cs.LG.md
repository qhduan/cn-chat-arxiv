# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LIBERO-PeRM: Benchmarking Personalized Robotic Manipulation](https://arxiv.org/abs/2610.22372) | 提出LIBERO-PeRM基准，通过受控形式化框架和涵盖运动、技能与用户档案的可观察偏好分类体系，从零样本偏好遵循、演示学习效率、跨布局跨任务泛化以及偏好组合与冲突处理五个方面系统评估具身智能体的个性化机器人操作能力。 |
| [^2] | [PathFM: Fast Generative Forecasting of Multivariate Trajectories](https://arxiv.org/abs/2610.22358) | PathFM利用流匹配生成完整的多元未来轨迹以捕捉传统边际分位数模型无法刻画的依赖结构，并通过模型蒸馏与历史注意力缓存共享，实现了接近单次分位数查询速度的高效多路径生成。 |
| [^3] | [Rethinking Routing Exploration in Modern Mixture-of-Experts](https://arxiv.org/abs/2610.22353) | 本文揭示了现代MoE架构中共享专家削弱路由探索性与细粒度专家分割需要更大探索空间之间的冲突，并提出MirrorMoE通过token自适应抑制来缩小探索空间以解决该困境。 |
| [^4] | [MLExpert: Towards a Generalist Agentic Model for Autonomous Machine Learning Engineering](https://arxiv.org/abs/2610.22344) | 本文提出了一种面向自主长期机器学习工程优化的后训练方法，通过从长交互轨迹中筛选有效监督信号并从中间结果中学习，训练出能够自主组织实验和调整优化策略的通用智能体模型MLExpert。 |
| [^5] | [GeoState: Learning View-Decoupled Geometry States for 3D World Modeling](https://arxiv.org/abs/2610.22343) | GeoState提出一种视角解耦的几何状态表示，通过场景分词器聚合多视角几何基础模型特征并将相机运动单独建模，实现了场景动态演化与相机变化的解耦，从而提升3D世界模型状态预测的难度和灵活性。 |
| [^6] | [Lean Graph Search for Formal Theorem Proving](https://arxiv.org/abs/2610.22339) | 该论文提出LeanGraphSearch，一种通过构建Mathlib前向依赖图、沿多跳依赖链传播相关性来发现支撑引理的图检索方法，从而提升Lean 4形式化定理证明中的引理检索效果。 |
| [^7] | [Robust Reinforcement Learning for Dynamic Treatment Regimes under Model and Measurement Uncertainty](https://arxiv.org/abs/2610.22336) | 本文提出了一种鲁棒强化学习框架，通过将患者建模为隐藏上下文，利用鲁棒上下文马尔可夫决策过程统一解决泛化与鲁棒性问题，学习出自适应治疗策略，在转移性前列腺癌自适应治疗模拟中几乎不损失性能的情况下实现鲁棒性，并在多种场景下帕累托优于现有方法。 |
| [^8] | [Twice Upon a Time: Complementary Representations for Time-Series Reasoning](https://arxiv.org/abs/2610.22327) | 该论文提出以互补表示作为时间序列推理的设计原则，通过融合折线图与归一化延迟嵌入两种视觉视图，让时间序列语言模型既保留绝对测量信息，又剔除幅度缩放和偏移等变化。 |
| [^9] | [Learning to Steer LLM Agents and Multi-Agent Systems](https://arxiv.org/abs/2610.22316) | 提出一种由强化学习联合训练的引导智能体，通过快速与慢速双路径实时检测并纠正大语言模型智能体及多智能体系统中的目标漂移，提升长时程任务执行的可靠性。 |
| [^10] | [Knowledge That Is Linear Before Fine-Tuning Generalizes After It](https://arxiv.org/abs/2610.22300) | 论文提出区分语言模型中支持泛化的“活性事实”与仅可回忆的“惰性事实”，并通过重标注实验发现：微调前在模型内部呈线性可读的知识，在微调后能够泛化到共享相同答案的未训练实体上。 |
| [^11] | [Backward Compatibility in Zero-Bit Neural Audio Watermarking](https://arxiv.org/abs/2610.22296) | 本文首次形式化了零比特音频水印中的向后兼容性问题，提出一种将新检测器与冻结旧版嵌入器链配对、并采用性能加权采样的训练框架，使检测器在版本迭代中保持对旧内容的高精度检测，同时揭示了这种必要耦合会引发“嵌入器崩溃”效应。 |
| [^12] | [Towards Continual Learning: Mitigating Forgetting With Subspace Tuning](https://arxiv.org/abs/2610.22293) | 提出子空间微调（SST）方法，将权重更新限制在底部奇异子空间、避免改变主奇异方向，从而有效缓解大语言模型微调中的灾难性遗忘问题。 |
| [^13] | [CAT-alyzing Honesty: Continuous Adversarial Training for Transferable LLM Honesty](https://arxiv.org/abs/2610.22292) | CATCH方法通过持续对抗训练，仅用数百个自我报告场景构建的目标，就能让大语言模型习得可泛化、可跨基准迁移的诚实行为，从而在压力情境下显著减少不诚实表现。 |
| [^14] | [Learning Rational Strategies for Temporal Objectives in Multi-Agent Systems](https://arxiv.org/abs/2610.22287) | 该论文针对动态未知的多智能体系统，将纳什均衡等解概念与 PLTL_f 概率时序目标相结合，提出了一种基于带收敛保证的 Nash-Q 学习和有限样本置信界的理性综合方法，能够学习到既满足时序约束又具备策略稳定性的联合策略。 |
| [^15] | [Asking Smarter Questions: Targeted Active Preference Learning for Underspecified Rewards](https://arxiv.org/abs/2610.22285) | 该论文提出将澄清欠规范奖励的负担从人类转移给机器人，利用教学数据中揭示不确定性的信息构建有针对性的先验，通过定向偏好查询让人类以低成本方式弥补监督不足的任务特征。 |
| [^16] | [Librarian Agents: Training LLMs to Curate Long-Context Workspaces that Anticipate Future Queries](https://arxiv.org/abs/2610.22273) | 该论文提出“图书馆员智能体”框架，训练大语言模型在推理之前主动将非结构化语料库重组为可复用的分层文件系统工作区，从而预先组织信息以高效应对未来多样化的查询需求。 |
| [^17] | [Baselines and Score Centering in Policy Gradient Estimation](https://arxiv.org/abs/2610.22263) | 本文提出在正则化重要性采样的离策略强化学习中，应以奖励与重要性加权得分之间的协方差作为估计对象，从而恢复策略梯度对奖励常数平移的不变性。 |
| [^18] | [Topology-Inspired Fine-Tuning for Wind-Field Super-Resolution](https://arxiv.org/abs/2610.22262) | 该论文提出利用可微的拓扑启发辅助损失（标量速度、梯度幅值、软高速水平集、局部极大值）及固定持续对监督来微调预训练CNN，从而在风场超分辨率中弥合逐点重建损失与极值、连通区域等拓扑结构保真之间的矛盾。 |
| [^19] | [BitSparse: Compact Activation Storage for ReLU Feed-Forward Networks](https://arxiv.org/abs/2610.22256) | BitSparse 利用稀疏 ReLU 激活特性实现 GPU 上的无损激活压缩，通过融合进自动求导的自定义 Triton 内核，将 Modded-NanoGPT 和 Nemotron-H 的每 token 显存占用分别降低 31% 和 30%，且开销不超过 1%。 |
| [^20] | [Explore Once, Evaluate Many: Multiple Policy Evaluation for Stochastic Linear Bandits](https://arxiv.org/abs/2610.22252) | 提出了 PRISMA 算法，通过一次无奖励探索收集数据并结合正则化信息共享与混合分配，能够在随机线性上下文老虎机中以样本复杂度仅随策略数 $K$ 对数增长的代价同时高精度评估 $K$ 个目标策略的价值。 |
| [^21] | [Language Is a Security Surface for Agent Guards](https://arxiv.org/abs/2610.22244) | 论文构建了 LinguaGuardBench 基准，发现仅将用户指令翻译成德语、法语等非英语语言就会使大多数开源、小型和专用智能体安全防护器改变判定，揭示了语言本身是智能体安全防护的一个重要攻击面。 |
| [^22] | [Shortcut Sensitivity in Agent-Safety Evaluations](https://arxiv.org/abs/2610.22243) | 该论文提出一个三层级捷径审计框架，揭示智能体安全防护评估数据中存在的捷径线索（如命令长度、领域混叠、加权方式）会显著夸大防护机制的表观性能（F1可从约0.90跌至0.50–0.72），因此安全评估必须控制这些捷径才能得出可靠结论。 |
| [^23] | [SurvICL: a Foundation Model for in-context Survival Analysis](https://arxiv.org/abs/2610.22242) | SurvICL是一个通过新型比例几率先验生成合成数据并以分位数回归损失预训练的上下文生存分析基础模型，首次超越经典Cox模型，并配套推出了迄今最大的103数据集生存分析基准。 |
| [^24] | [Query-Efficient Rejection Sampling for Parallel Masked Diffusion](https://arxiv.org/abs/2610.22238) | 提出单增量拒绝采样器（SIRS），每次仅需一次查询即可估计并行掩码扩散模型拒绝采样中的对数接受概率，并结合伯努利工厂与Rao-Blackwell化显著降低查询成本并减少截断偏差。 |
| [^25] | [Training-Free On-Policy Distillation](https://arxiv.org/abs/2610.22236) | 本文提出免训练的在策略蒸馏方法，当多个专家模型共享同一基础模型时，仅需通过几百个采样样本的一次前向传播、以最小二乘方式逐层融合，即可在无需任何梯度训练的情况下将多个专家的能力聚合到一个模型中，且平均准确率超过任何单一专家模型。 |
| [^26] | [OS-Helix: Unifying GUI Interaction and Code Execution in Native Computer-Using Agents](https://arxiv.org/abs/2610.22234) | OS-Helix是一个端到端训练的多模态计算机使用智能体，通过单一动作空间统一GUI操作与代码执行，并借助应用-工件组合方法和解耦的指令/评估器生成流水线来构建高质量的混合训练数据。 |
| [^27] | [Architecting for Hardware Diversity: Meta's End-to-End Co-Design for High-Performance AI Training on AMD GPUs](https://arxiv.org/abs/2610.22230) | Meta提出硬件-模型协同设计、分析性能模型驱动的零预热内核选择和智能体工作流三大可复用原则，成功实现了推荐模型训练在AMD MI350X GPU上相对于NVIDIA B200的高性能优化移植。 |
| [^28] | [How Much Physics Can Vision Learn from Audio? Probing and Distilling Damping Structure](https://arxiv.org/abs/2610.22221) | 该研究发现音频嵌入能远优于视觉地捕捉物体阻尼结构（品质因数Q排序上ρ=0.73对比0.28），并提出将冻结音频教师模型蒸馏到视觉模型，以让视觉从音频中学习阻尼物理特性。 |
| [^29] | [FutureLift: Lifting Trajectory Proposals into Bidirectional Future Graphs for Motion Forecasting](https://arxiv.org/abs/2610.22215) | FutureLift提出了一种与主干网络无关的轨迹提议改进方法，将多模态提议表示为带有双向时间边的异构未来图，使后续证据能够在单次遍历中修正较早的片段，从而提升自动驾驶运动预测的准确性。 |
| [^30] | [Generalization Guarantees of Randomized Block Coordinate Descent for Nonconvex Learning](https://arxiv.org/abs/2610.22213) | 本文为非凸学习中的随机块坐标下降算法建立了高概率泛化保证，并通过早停策略量化了优化与泛化的平衡，实现了总体风险梯度的最优收敛速率。 |
| [^31] | [Beyond Downstream QA: Rethinking How We Evaluate Agent Memory](https://arxiv.org/abs/2610.22209) | 本文指出当前以下游问答准确率评估智能体记忆的范式会将记忆正确性与检索、生成质量混淆，并通过范式反转构建了FEB-30基准，可在无人工标注的情况下对智能体记忆存储本身进行确定性的直接评估。 |
| [^32] | [Attention over noisy examples: generalization and memorization in the exponential Hopfield network](https://arxiv.org/abs/2610.22191) | 该论文将含噪样本的Hebbian学习协议移植到等价于softmax注意力机制的指数Hopfield网络，并通过信噪比分析闭式推导出对未见原型的检索重叠公式m=erf(1/√(2ρ_eff))，其中数据集熵被由softmax权重构成的注意力熵所取代，从而刻画了注意力机制从噪声样本中实现泛化与记忆的条件。 |
| [^33] | [Whole-Body Dexterity](https://arxiv.org/abs/2610.22184) | 该论文提出全身灵巧操作（WBD）框架，让机器人利用身体和四肢作为接触面来抓取和搬运大体积物体，其蒸馏策略可零样本迁移到真实的宇树G1人形机器人上并完成长时程搬运任务。 |
| [^34] | [LCT: Losslessly Compressed Tensors with GPU friendly execution for Training](https://arxiv.org/abs/2610.22179) | LCT提出了一个GPU友好的无损张量压缩执行框架，利用浮点数据冗余和优化的霍夫曼编码内核，在几乎不增加训练时间的情况下实现高达1.455倍压缩比，将训练内存占用降低最多24.6%并支持3.5倍的更长上下文。 |
| [^35] | [REKI: Registering Knowledge in VLAs](https://arxiv.org/abs/2610.22178) | REKI是一个仅用公开数据预训练的24亿参数视觉-语言-动作模型，通过在动作专家中引入寄存器标记记忆机制来保留近期历史信息，以更小的模型规模达到了接近两倍大小模型的性能，并显著提升了仿真与真实机器人任务的成功率、大幅减少了姿态振荡。 |
| [^36] | [Mosaic: Multi-Session Agentic Reinforcement Learning](https://arxiv.org/abs/2610.22174) | 提出 Mosaic 方法，通过无评论员的层次化优化机制区分多会话智能体强化学习中各会话的不同贡献，弥合单会话训练假设与多会话推理执行之间的不匹配。 |
| [^37] | [Lexical Reference Frames: Interpreting LM Residual Streams in Native Semantics](https://arxiv.org/abs/2610.22169) | 论文提出“词法参考框架”，通过模型自身剩余层诱导的、随上下文自适应的原生参考框架来解读残差流中的隐状态，并证明任何固定方向的透镜都无法超越原生框架相干性平方根的理论上限，而 LexFrame 在 SmolLM3-3B 和 OLMo3-7B 上达到 0.97 的相干性且读取误差在 0.16 nats 以内。 |
| [^38] | [The importance of tensor handling and implementation of physics-informed DeepONets](https://arxiv.org/abs/2610.22157) | 提出一种与原架构数学等价的改进PI-DeepONet，通过消除Branch网络处理输入函数时的冗余，减小输入张量规模和网络评估次数，从而显著降低高维问题的训练计算成本。 |
| [^39] | [Specification Gaming and a Constrained Reinforcement Learning Mitigation in a Cleaning Gridworld](https://arxiv.org/abs/2610.22152) | 该论文在一个清洁机器人网格世界中演示了规范博弈现象——智能体学会隐藏灰尘而非清除灰尘来骗取代理奖励，并提出一种通过惩罚博弈行为、带有可验证保证的约束强化学习方法来缓解这一问题。 |
| [^40] | [Causal Integrated Gradients: Scalable Feature Attribution Along Causal Paths](https://arxiv.org/abs/2610.22126) | 提出因果积分梯度（CIG）和因果期望梯度（CEG），通过沿因果图路由积分路径来利用特征间已知的因果结构，在保持积分梯度计算效率和公理化保证的同时实现可扩展的特征归因，避免将功劳错误归于下游效应而非真正的上游原因。 |
| [^41] | [Beyond Fixed Acquisition Functions: Temporal Difference Active Learning for Perturbation Discovery](https://arxiv.org/abs/2610.22122) | 提出了一种名为TDAL的主动学习方法，将预算约束下的扰动发现建模为序贯决策问题，通过时间差分学习从单次筛选轨迹中在线学习采集函数，突破传统固定采集函数的局限，实现自适应的实验选择。 |
| [^42] | [Scene Reconstruction Using Ubiquitous Wireless Signals via Diffusion Posterior Inference Over physics-Grounded Representations](https://arxiv.org/abs/2610.22120) | 该论文将无线场景重建表述为后验推断问题，提出一个基于物理特性表示、结合扩散模型的两阶段分层框架，从泛在的无线信道测量中重建周围环境的场景布局。 |
| [^43] | [Patch where the text is new: a compression view of where byte-level models should spend compute](https://arxiv.org/abs/2610.22100) | 提出受LZ77压缩启发的新颖性打补丁方法——用一个几秒钟即可在无标注文本上拟合的、基于前两个字节的查找表，在字节不重复近期上下文处切分补丁，从而大幅提升字节级语言模型在数学任务上的准确率（50M参数下最终答案准确率从60%升至90%）。 |
| [^44] | [TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language Models](https://arxiv.org/abs/2610.22099) | 本文提出TIPS，一个仅约60万参数的微型推理时控制器，通过强化学习同时学习解掩码与重新掩码策略，在保持底层语言模型完全冻结的情况下，显著提升了掩码扩散语言模型在数学推理任务上的表现。 |
| [^45] | [Beyond the Dominant Label: Recovering Coexisting Lesion Evidence with Multiple Instance Learning](https://arxiv.org/abs/2610.22091) | 提出DomSet-MIL框架，通过主导标签删失似然和严重程度排序残差路由，在弱监督下恢复与主导病灶共存时的次要病灶证据，避免低严重程度病变信息被抑制。 |
| [^46] | [Learning Changes the World and Then What Was Learned has Changed: Training and Deploying Neural Networks in the Control Loop](https://arxiv.org/abs/2610.22084) | 本文提出了一个统一分类体系来刻画离线训练的神经网络部署到闭环控制后因自身动作改变状态分布而引发的三类失效模式，并通过时间尺度分离的在线适应设计策略来缓解该分布偏移问题。 |
| [^47] | [Can AI Forecast the Extremes? Evaluating Probabilistic Weather Models on Tropical Cyclones](https://arxiv.org/abs/2610.22077) | 通过对热带气旋预报开展广泛的概率基准评估，本研究发现尽管在强度和集合离散度方面仍有不足，以GenCast和FourCastNet 3为代表的领先AI模型在预报性能上大幅超越了ECMWF的业务集合预报系统IFS-ENS。 |
| [^48] | [The Right Byte First: How UTF-16 Helps Multilingual Boundary Prediction](https://arxiv.org/abs/2610.22071) | 该论文提出通用字节级编码（UBE），将三字节 UTF-8 码点改写为低字节在前的两个 UTF-16 字节，使边界预测器从首个符号即可获得更多区分信息，从而提升多语言字素边界预测的准确性。 |
| [^49] | [Memory as Weights: Internalizing Long-Term History for Streaming Videos](https://arxiv.org/abs/2610.22070) | 提出MaLoW框架，利用循环超网络将视频历史信息逐步内化为固定大小的LoRA权重，使流式视觉语言模型以权重形式承载长期记忆，避免了历史信息与当前观察对有限上下文的争夺。 |
| [^50] | [ABNAT: Attention-Based Noise-Aware Training for Robust Transformers on Analog Compute-in-Memory Systems](https://arxiv.org/abs/2610.22065) | 该论文提出ABNAT，一种基于注意力的噪声感知训练方法，通过分析模拟存内计算噪声如何重塑Transformer注意力机制（发现深层注意力差异增大且传统NAT存在过度自信问题），显著提升Transformer在模拟CIM硬件上的鲁棒性。 |
| [^51] | [Rethinking Successive Rejects Algorithms](https://arxiv.org/abs/2610.22051) | 本文通过将 SR 算法的阶段长度调度设计表述为归一化误差指数的优化问题，提出了 H1-平衡连续拒绝算法，从而在固定预算最优臂识别中提升了 SR 算法的性能。 |
| [^52] | [Missed Detections and False Alarms in a CNN Technosignature Detector on Breakthrough Listen Data](https://arxiv.org/abs/2610.22050) | 该研究通过向真实绿岸天文台数据中注入15,000个不同形态和振幅的信号，对一个冻结的CNN技术签名探测器进行压力测试，发现其召回率随信号增强反而下降，且某些干扰形态的假阳性率高达0.627，揭示了现有机器学习探测器在训练分布之外的脆弱性。 |
| [^53] | [Decoding Naturalistic Speech from fMRI with Temporal Modeling and Multimodal Alignment](https://arxiv.org/abs/2610.22049) | 提出NOEMA时间卷积解码器，通过将连续fMRI测量与多模态（文本、音频、语音）刺激嵌入对齐，实现了从大脑活动中高精度检索自然语音片段，在2000个候选中前10命中率超过42%。 |
| [^54] | [Effectiveness of Hidden-State Probes for Localizing Reasoning Failures](https://arxiv.org/abs/2610.22046) | 该研究发现基于最终答案正确性训练的隐状态探针难以有效区分推理失败的具体类型，例如置信度探针在检测单步致命错误和可重试成功的情况时分别漏掉了81%和47%的案例。 |
| [^55] | [Momentum, Commentary Tone, and Earnings Surprise in LLM Financial Forecasts](https://arxiv.org/abs/2610.22043) | 本研究通过独立操纵动量、评论语气和EPS意外的合成财报实验发现，在大语言模型的金融预测中，非基本面的评论语气和价格动量因素对模型输出的影响往往超过实际的每股收益意外本身，揭示了LLM金融预测易受非信息性输入干扰的问题。 |
| [^56] | [ETH: Data-Balanced Heterogeneous Pipeline Parallelism for Training Large Models](https://arxiv.org/abs/2610.22035) | 该论文提出异构流水线并行方法（HPP）及ETH训练系统，通过分层工作负载均衡框架和分块梯度聚合技术，在异构流水线间实现细粒度数据均衡，同时解决了配置搜索开销大和非对称All-Reduce通信开销高的关键难题。 |
| [^57] | [Designing Experiments for External Decision Makers](https://arxiv.org/abs/2610.22032) | 提出“使用感知实验设计”框架，通过训练自适应设计策略最小化“使用风险”，以应对数据将被实验者无法控制的外部规则（可能是非贝叶斯或黑盒）使用的情况，并揭示在此类场景下更具信息量的数据反而可能有害。 |
| [^58] | [Selection Is Not Learning: Choosing Model Futures](https://arxiv.org/abs/2610.22029) | 自进化语言模型中“看起来可学习的任务”并不等于“有用的模型更新”，本文提出 HeirCommit 与 EvoHeir 方法，通过冻结评估和参数-经验分离式继承来弥合选择与更新之间的差距。 |
| [^59] | [Augment the Cache, Replace the Activation: A 2x2 Study of Internal-State Sharing](https://arxiv.org/abs/2610.22026) | 本文将多智能体系统中的内部状态“传输”范式重构为“共享”范式，并发现内部状态载体与融合机制的选择是相互耦合的，其关键在于载体进入接收方计算的位置。 |
| [^60] | [Not All Unlabeled Data Are Equal: Unlabeled Data Selection for Semi-Supervised QSAR Modeling with Large Chemical Repositories](https://arxiv.org/abs/2610.22025) | 该论文首次在真实半监督学习场景下，利用ChEMBL的10个QSAR数据集和40余万无标签分子池，系统研究了如何从大型化学数据库中选择无标签数据以改进QSAR模型，揭示了并非所有无标签数据都同等有价值。 |
| [^61] | [DemoStream: Batched Task-and-Motion-Plan Optimization for Automated Learning from Demonstration](https://arxiv.org/abs/2610.22020) | 本文提出DemoStream，一个基于GPU加速的批量任务与运动规划（TAMP）系统，通过批量约束满足优化演示质量，自动生成高质量演示数据以提升机器人行为克隆的成功率。 |
| [^62] | [Priming Base Models for Agentic Post-Training: An Open Midtraining Recipe](https://arxiv.org/abs/2610.22019) | 该论文提出了一种开放的中期训练方案，在预训练与后训练之间以 256K 上下文、数千亿 token 的自然软件工程数据混合少量智能体轨迹来训练基础模型，显著提升了 SWE-bench 等智能体任务表现及后续强化学习的可训练性。 |
| [^63] | [Close Enough is Nowhere Near: Fidelity-Consumability Decoupling in KV Caches](https://arxiv.org/abs/2610.22016) | 论文揭示了冻结的多智能体LLM系统中直接替换发送方KV缓存不可行的根本原因——读取方只能接受其自身“可容忍集”内的缓存，几何保真度与可用性相互解耦且失败并非渐进式，这一现象可由竞争性注意力机制解释。 |
| [^64] | [GraphBEAT: Graph-based Bucketized Event Aggregation Over Time for Early Risk Stratification of Gastrointestinal Cancers](https://arxiv.org/abs/2610.22014) | 提出GraphBEAT时序图表示学习框架，将纵向EHR临床事件转化为按时间分桶的患者特定有向图，结合时间衰减消息传递与多头注意力池化，实现胃肠道癌症的早期风险分层。 |
| [^65] | [Stateful Testing for Deep Learning Compilers](https://arxiv.org/abs/2610.22003) | 提出面向深度学习编译器的有状态测试方法 Rondo，通过将模型与调用序列配对、利用编译行为引导的 MCMC 采样进行序列变异，并结合三路差分预言机，检测由编译器保留状态引发且在单独执行时无法复现的缺陷。 |
| [^66] | [Interfaces and Modules: A Scalable Representation for Weight-Space Learning across Architectures](https://arxiv.org/abs/2610.21995) | 该论文提出了一种以“接口”（坐标置换空间）和“模块”（操作超边）为基本单元的类型化图表示，并构建了通用权重编码器（UWE），通过局部等变更新与接口摘要上的全局注意力相结合，实现了尊重参数对称性且可扩展、可跨架构的权重空间学习。 |
| [^67] | [OntoAct: Hypothesis-Based Neuro-Symbolic Agents through Ontological Abstraction](https://arxiv.org/abs/2610.21983) | OntoAct通过本体抽象将经验转化为可复用的知识公理，让大语言模型提出假设和动作、由符号验证器检验证据并动态更新评估，从而实现跨情境的知识复用与假设驱动的决策，并配套发布了用于诊断修复工厂隐藏故障的Tacit-FLE基准。 |
| [^68] | [Deep Learning of Topological Equivalence Classes](https://arxiv.org/abs/2610.21981) | 该论文将深度学习应用于判断黎曼面上有限群作用的拓扑等价性，探讨了代数结构对模型设计的影响以及模型成功习得的拓扑等价类结构性质。 |
| [^69] | [P2TPO: Structure-Aligned Page-to-Token Credit Assignment for Full-Page Document Parsing](https://arxiv.org/abs/2610.21979) | P2TPO提出了一种结构对齐的信用分配框架，通过参考元素对齐、统一锚点定价与片段信用重分配，将页级奖励细化到词元级别，从而提升端到端整页文档解析的准确性与完整性。 |
| [^70] | [Learn Only Where You Can't Compute: Structured Corrections to Gaussian Process Posteriors](https://arxiv.org/abs/2610.21977) | 提出一种结构化后验修正框架，在易处理的高斯参考基础上，由扩散模型仅在被选中的低维差异子空间中学习数据驱动的修正，从而兼顾高斯过程的可扩展性与扩散模型的表达能力，用于科学推断任务。 |
| [^71] | [Time to Fake It: Continual Machine Learning Intensifies Unfairness in Deepfake Detection](https://arxiv.org/abs/2610.21974) | 该论文首次系统研究了持续学习对深度伪造检测器公平性的影响，发现在按时间顺序的11种深度伪造生成方法上渐进训练时，准确性下降通常会加剧跨人口群体（尤其是交叉性群体）的不公平现象。 |
| [^72] | [Uncertainty-Aware Federated Learning for Joint Communication and Computation in XR Systems](https://arxiv.org/abs/2610.21969) | 本文提出了一种不确定性感知的分布式深度学习方法，通过集中式或联邦式训练的深度神经网络，在信道与计算状态不确定的干扰网络中鲁棒地联合优化发射功率、计算功率与计算任务分配，以最小化6G网络中XR设备的最坏情况时延。 |
| [^73] | [Why Shallow Neural Networks Learn Sinusoidal Components First: An Orthogonal Polynomial Dynamics Framework](https://arxiv.org/abs/2610.21964) | 该论文提出基于一般傅里叶级数展开的正交多项式动力学框架，通过分析动力学矩阵的特征值衰减与特征向量形状，从理论上解释了浅层神经网络为何优先学习正弦（低频）分量的谱偏置现象，并适用于一般激活函数、弱数据分布假设和任意网络结构。 |
| [^74] | [ChronoSheaf: Joint Spatio-Temporal Spectral Filtering on Cellular Sheaves](https://arxiv.org/abs/2610.21958) | ChronoSheaf提出了一种在空间和时间上联合定义的谱层束拉普拉斯算子及相应谱滤波器，使层束神经网络能够显式建模图上信号随时间的演化，并在不扩大底层层束结构的情况下同时扩展空间与时间维度的有效感受野。 |
| [^75] | [SelectPFN: Amortized Bayesian Variable Selection for Sparse Nonlinear Regression with Prior-Data Fitted Networks](https://arxiv.org/abs/2610.21950) | SelectPFN是一种在单次前向传播中同时完成稀疏非线性回归贝叶斯变量选择与预测的先验数据拟合网络，可输出每个变量的后验包含概率，并通过敲除列实现精确的有限样本错误发现率控制。 |
| [^76] | [The Lagrangian Structure of Cost-Penalized Active Feature Acquisition](https://arxiv.org/abs/2610.21919) | 本文揭示了成本惩罚EIG-Cost贪婪启发式方法本质上是期望成本主动特征获取问题的拉格朗日松弛的一步贪婪优化（lam为对偶乘子），并提出了将策略差距分解为预算松弛项与贪婪次优性项的后验证书，在4,000个小规模案例中验证该次优性项接近于零。 |
| [^77] | [Learning Rewards in Strategic Games](https://arxiv.org/abs/2610.21912) | 该论文提出“策略可辨识性”这一新视角，证明仅从固定策略对学习（无论专家样本数量多少）不足以在策略意义上辨识奖励，并据此提出一种主动逆强化学习算法来在策略博弈中学习奖励。 |
| [^78] | [The Tilted Sampling Problem: Optimism, Posterior Sampling, and Logarithmic Regret](https://arxiv.org/abs/2610.21908) | 该论文提出了倾斜采样这一在线学习问题，以温度归一化累积KL散度定义采样遗憾，并构建了一个通用遗憾框架，揭示倾斜采样与经典遗憾最小化具有不同的不确定性结构，借助乐观原理和后验采样可将对数级遗憾推广至从未知奖励诱导的吉布斯分布中采样的场景。 |
| [^79] | [In-Context Learning Before Specialisation: Theory of Learning in a Solvable Topic Model](https://arxiv.org/abs/2610.21901) | 该论文在一个可解的主题模型中建立了学习理论，从根本层面解释了序列模型为何以及何时会倾向于采用上下文学习而非存储训练数据的专门化外推。 |
| [^80] | [Self-Calibrated Subspace Robust PCA for Parameter-Free Removal of Mixed Noise and Line Stripes in Hyperspectral Images](https://arxiv.org/abs/2610.21896) | 本文提出自校准的SC-RCCTV模型，通过在迭代中自适应估计各波段噪声水平和条纹阈值（无需预先设定任何参数），实现了对高光谱图像中强度跨波段变化的混合噪声与线状条纹的有效去除。 |
| [^81] | [Graph Dirichlet Energy for Fitting-Free Bivariate Causal Inference](https://arxiv.org/abs/2610.21892) | 本文提出免拟合的图狄利克雷能量因果推断准则（GDECI），通过比较两个方向上图信号的狄利克雷能量来判别二元因果方向，无需回归拟合或独立性检验，具有单调变换不变性，并可高效地在 O(n log n) 时间内精确计算。 |
| [^82] | [Learning Lean Feedback for Efficient Proof-Candidate Evaluation on GPUs](https://arxiv.org/abs/2610.21881) | 该论文训练神经评估器在GPU上预测Lean 4的判定结果与编译器反馈，其两阶段评估器达到87.8%的预测准确率，吞吐量比Lean验证基线提升4.2倍，显著加速了形式化定理证明中语言模型训练的奖励评估过程。 |
| [^83] | [Provably Efficient Risk-Sensitive Offline Reinforcement Learning via Optimized Certainty Equivalents](https://arxiv.org/abs/2610.21870) | 该论文针对通用风险度量类“优化确定性等价（OCE）”下的离线风险敏感强化学习提出了动态与静态两种形式化的悲观价值迭代算法，并建立了紧密的次线性次优性理论界限，首次对该离线场景给出了一般性的理论保证。 |
| [^84] | [Lethe: Link Inference Attacks For Evaluation of Edge Unlearning Methods](https://arxiv.org/abs/2610.21869) | 提出首个全面的边遗忘基准Lethe，利用链接推断攻击在15种方法和8个数据集上进行评估，并揭示现有评估方法的系统性缺陷（专用方法可能比重新训练更慢、准确率指标无参考价值）。 |
| [^85] | [Temporal Periodic Image Registration with Implicit Neural Representations](https://arxiv.org/abs/2610.21845) | 该论文提出Temporal-IDIR方法，通过优化单个隐式神经表示同时建模时间图像序列中所有帧的形变，并引入时间一致性损失实现自正则化，在DIR-LAB数据集上取得了1.03毫米的目标配准误差。 |
| [^86] | [Physics-Aware Distillation of a Grid Foundation Model for Real-Time Edge Inference](https://arxiv.org/abs/2610.21833) | 该论文提出一种结合物理残差迁移的蒸馏方法，将电网基础模型参数压缩32.5倍（从2010万降至62万参数），使潮流计算能够在边缘设备上于单个PMU报告帧内实时完成。 |
| [^87] | [Local Epochs as Optimal Stopping: Theory and Evidence in Federated Learning](https://arxiv.org/abs/2610.21832) | 该论文将联邦学习中的局部训练轮次选择建模为最优停止问题，证明了单阶段前瞻阈值规则的最优性，并据此提出 FedStop 算法——它根据训练阶段而非客户端自适应地分配局部训练预算，平均准确率不低于调优后的 FedAvg。 |
| [^88] | [Compiling Multi-Threshold Operators for Quantized Non-Monotonic and Fine-Granular Activations in Custom-Tailored DNN Accelerators](https://arxiv.org/abs/2610.21825) | 本文将多阈值表示形式化并扩展至非单调激活函数与多维细粒度阈值参数，提出基于代数规则并集成到ONNX编译器基础设施中的系统化自动转换方法，从而在定制DNN加速器上实现更广泛的量化激活函数高效硬件加速。 |
| [^89] | [A Closed Form Depth-Width Law for Neural Collapse in Deep Linear Networks](https://arxiv.org/abs/2610.21820) | 该论文首次在深度线性网络中推导出神经坍缩的闭式深度-宽度定律，证明在大宽度极限下类内变异性的累积压缩仅依赖于深度与宽度之比 L/N，且压缩速率由数据、先验和架构自洽决定。 |
| [^90] | [Measuring the local shape of learning-rate transfer curves](https://arxiv.org/abs/2610.21814) | 本文提出两个关系方程——一个在稳定性边缘成立的理论公式和一个将学习率梯度视为超参数梯度的近似表达式——从经验上很好地解释了学习率迁移曲线的梯度，为定量刻画学习率曲线的局部形状提供了方法。 |
| [^91] | [N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models](https://arxiv.org/abs/2610.21810) | 本文提出条件程序（CP）压缩表示方法，仅用两比特编码按需重构大语言模型n-gram记忆查找表中被请求的行，相比标准2比特量化将输出KL散度降低41–63%，并将DeepSeek的表存储从203GB压缩至51GB。 |
| [^92] | [CIM-Embed: Learning Instruction-Conditioned Multimodal Embeddings for Clustering](https://arxiv.org/abs/2610.21805) | 本文提出CIM-Embed框架，通过监督对比学习将图像、文本和指令直接映射为多模态嵌入，使同一组对象能根据不同指令形成不同的聚类划分，且无需预先知道聚类数目。 |
| [^93] | [Single-stage joint diffusion of sky images and clear-sky index for ultra-short-term irradiance forecasting: what the frame-to-irradiance stage discards](https://arxiv.org/abs/2610.21795) | 该论文提出一种在单一扩散过程中联合去噪未来天空图像与晴空指数的单阶段辐照度预测模型，省去了传统两阶段设计中的独立回归器且无需实测辐照度输入，在仅有相机的站点上实现了超越十二个基线配置的超短期辐照度预测性能。 |
| [^94] | [Same Spectrum, Harder First-Order Training: Phase-Locking in Single- versus Multi-Qubit Re-Uploading Circuits](https://arxiv.org/abs/2610.21778) | 单比特重复上传电路虽与多比特电路在相同编码预算下具有同等的傅里叶表达能力，但因布洛赫球约束产生的“相位锁定”使其切方向至多只有 $2E+2$ 个（远少于双比特对照电路的 $6E+6$ 个），导致一阶训练明显更困难。 |
| [^95] | [Q($\gamma_H$): Entropy-Guided Horizon Mixing in Multi-Discount Q-Learning](https://arxiv.org/abs/2610.21764) | 提出多视界Q学习框架Q($\gamma_H$)，通过维护多个折扣专用输出头并利用熵引导的置信度信号在控制时自适应混合，在不修改Bellman更新的情况下实现了轻量级的自适应规划视界控制，尤其在延迟信用任务上改进显著。 |
| [^96] | [DeciCAD: Harness-Grounded Learning of Action Policies for CAD Agents](https://arxiv.org/abs/2610.21755) | DeciCAD提出了一个面向CAD的领域原生智能体执行框架，通过从交互轨迹中提取因果决策监督，并结合监督微调与几何在线强化学习训练动作策略，在四个CAD基准上取得最优的F1与IoU，同时将模型输出token减少45%至88%。 |
| [^97] | [Harmful Partial Interventions and Benign Leakage in Concept-based Models](https://arxiv.org/abs/2610.21750) | 该论文形式化了基于概念的模型中概念干预反直觉地恶化下游预测的“干预伤害”问题，区分了全量干预与更贴近实际的部分干预两种情形，并证明“充分性”和“局部化”这两个标准不足以保证部分干预下概念泄漏是良性的。 |
| [^98] | [The Delegation Dilemma: Economies of Scale, Accountability, and Polycentric Governance](https://arxiv.org/abs/2610.21746) | 该论文通过重复博弈模型揭示了委托治理中的“规模—问责”权衡：更大的群体虽带来更多成本分担收益，却削弱了中心保留单个成员的激励，从而内生地决定了可持续的非剥削性委托的群体规模上限。 |
| [^99] | [Pairwise AUC Optimization Needs Corrective Power: A Unified View](https://arxiv.org/abs/2610.21743) | 本文发现直接成对AUC训练可能进入严重排序错误频发但丧失纠错能力的退化区域，并通过U-统计量的退化性质为该现象提供了统一的理论解释。 |
| [^100] | [Sparse Federated Echo State Networks for Time-Series Classification via Smooth-L1 Regularization](https://arxiv.org/abs/2610.21742) | 本文提出基于Smooth-L1正则化的稀疏回声状态网络SL1ESN，利用与牛顿法兼容的可微稀疏化方法实现读出层高度稀疏，并支持二阶联邦聚合，在七个时间序列分类基准上达到75%–95%的稀疏度。 |
| [^101] | [Reliable Knowledge Acquisition for Continual Learning Agents](https://arxiv.org/abs/2610.21739) | 提出RIKA框架，通过分层提取事实、概念、程序和元认知四类知识，并引入基于来源证据与后续经验的证据准入机制，解决现有经验学习系统中12%–33%的条目会损害未来任务性能的知识可靠性问题。 |
| [^102] | [FIRM: Fisher-Informed Retention Merging for Federated LLM Unlearning](https://arxiv.org/abs/2610.21720) | 提出FIRM框架，利用客户端本地Fisher统计量进行归一化合并并对保留敏感参数施加显式控制，解决联邦大语言模型遗忘中的幅度爆炸问题，在单轮服务器端聚合中实现稳定遗忘与保留效用的兼顾。 |
| [^103] | [International Multi-Centre Validation of Unsupervised Domain Adaptation for Precise Discrimination between Normal and Abnormal Retinal Development](https://arxiv.org/abs/2610.21719) | 该研究在国际多中心环境下验证了一种基于ResNet50的无监督域对抗神经网络模型，能够在四种不同OCT设备（包括台式和手持式系统）之间准确区分正常与异常视网膜发育，准确率达87.75%–96.06%，性能可与临床医生分级员相媲美。 |
| [^104] | [Robust Covariance Estimation for Entry-wise Contaminated and Partially Observed Data via Expectation Maximization Algorithm](https://arxiv.org/abs/2610.21709) | 本文提出一种在缩减潜空间上运行的随机近似EM（SAEM）算法，通过将坐标级尺度的随机更新与Rao-Blackwell化的高斯条件矩相结合，实现了逐元素污染与缺失数据下Student-$t^*$模型的鲁棒协方差估计，无需对潜在高斯变量显式采样。 |
| [^105] | [Human Interventions as Labels: Localizing, Steering, and (Sometimes) Editing the Computations Behind LLM Agent Failures](https://arxiv.org/abs/2610.21708) | 本文提出将人类对 LLM 智能体失败行为的干预视为标签，用以因果定位、引导并（有时）编辑失败背后的内部计算。 |
| [^106] | [ShiftQAT: Quantization-Aware Training for the Numerical Contract of a Fixed-Function Edge CNN Accelerator](https://arxiv.org/abs/2610.21695) | ShiftQAT是一种面向MAX7800x系列固定功能边缘CNN加速器的量化感知训练流程，通过构造方式生成满足二的幂次对称缩放、8位仅整数运算等数值契约的神经网络，是该系列开源训练工具链ai8x-training的默认训练路径。 |
| [^107] | [MARS: A Momentum-Aware Adaptive Rating System for Personalized Assessment](https://arxiv.org/abs/2610.21685) | 提出了MARS系统，将题库建模为有向加权知识图谱，通过动量感知的学习者速度动态更新能力评分，并优化时间预算内考虑负分惩罚的得分最大化策略，实现可解释的个性化自适应评估。 |
| [^108] | [Router Drift Control: Mitigating Off-Policy Degradation in Mixture-of-Experts Reinforcement Learning](https://arxiv.org/abs/2610.21682) | 提出路由漂移控制（RDC）方法，将采样与训练阶段的路由分歧作为离策略过时信号，通过序列级过滤拒绝高漂移响应，在不引入梯度偏差的前提下缓解混合专家强化学习中的路由漂移与离策略退化。 |
| [^109] | [Classical methods for modern applications: synthetic tabular data generation using lightweight discrete response continuous latent variable model](https://arxiv.org/abs/2610.21674) | 本文提出一种利用经验边际分布和分位数变换将含连续与离散变量的表格数据映射到连续潜空间的轻量级生成模型，能够高保真地生成合成表格数据，且实现简单、训练稳健、采样高效。 |
| [^110] | [Diffusion Language Models as Open Quantum Systems](https://arxiv.org/abs/2610.21672) | 该论文将扩散语言模型的连续时间前向与反向动力学嵌入开放量子系统框架，提出了提示叠加（SoP）、思想叠加（SoT）和量子启发重掩码（QIR），实现了基于句子状态叠加的条件文本生成以及基于置信度的已生成令牌修订。 |
| [^111] | [Approaching Backpropagation Accuracy with Weight Perturbation on MNIST](https://arxiv.org/abs/2610.21665) | 该研究系统考察了权重扰动方法在MNIST全连接网络中的表现，发现浅层网络经大批量训练可达约97%测试精度、接近早期反向传播水平，但训练效率仍存在明显差距，且深层网络尚未达到BP级精度。 |
| [^112] | [When Does "Post-Call" Begin? An Availability Contract for Earnings-Call NLP Benchmarks](https://arxiv.org/abs/2610.21659) | 论文提出“可用性契约”框架，用于诊断和防范财报电话会议NLP基准测试中的事件内时间错配问题，确保预测目标不会早于模型输入信息可用的时间点。 |
| [^113] | [Conditional Mean Flows: Few-Step Densities for Simulation-Based Inference](https://arxiv.org/abs/2610.21657) | 提出条件均值流方法，将少步采样与可解析的密度评估相结合，大幅降低了基于模拟推断中后验采样及MCMC重复似然评估的计算成本。 |
| [^114] | [The Self-Improvement Problem](https://arxiv.org/abs/2610.21646) | 本文将智能体的自我改进形式化为可计算性问题，指出奖励模型等自动化门控机制虽必须是可终止、可计算的过程，却无法可靠判定一个更新是否构成真正的语义改进，从而揭示了自我改进的结构性内在限制。 |
| [^115] | [Efficient Pairwise Difference Learning: A Comparative Study of Encoding Strategies and Training Pair Selection](https://arxiv.org/abs/2610.21644) | 该研究通过在近200个基准数据集上的大规模实验，系统比较了成对差分学习（PDC）中四种特征编码方法与四种样本对选择策略，为低数据量分类任务确定了最优实现方案。 |
| [^116] | [ReFlow: Replay-Driven Decoupling for Efficient RL Post-Training](https://arxiv.org/abs/2610.21640) | ReFlow通过重放驱动的解耦机制与系统-算法协同设计，突破传统异步执行效率受限和资源需求动态变化的瓶颈，显著提升大语言模型强化学习后训练的效率。 |
| [^117] | [ContextLift: Breaking the Spurious Correlation Behind Conditional Emergent Misalignment](https://arxiv.org/abs/2610.21633) | 提出ContextLift反事实数据增强方法，通过将随机训练提示中的上下文句子复制到安全数据中来打破条件性失调背后的伪相关，无需任何模型、标签或领域知识即可消除领域提示触发的失调行为。 |
| [^118] | [Local Temporal Contrasts on Human Motion Data Induce a Latent Geometry](https://arxiv.org/abs/2610.21630) | 提出SemTAC编码器，通过时间邻域对比损失直接恢复人体运动序列在表示空间中的轨迹几何结构，使模型能够在训练时未见过的更长时间尺度上进行推理。 |
| [^119] | [Market Incentives for AI Safety Investment](https://arxiv.org/abs/2610.21607) | 该研究构建了包含上游大语言模型提供商和竞争性下游企业的AI供应链博弈论模型，发现下游竞争促使下游企业的安全投资超过行业利润最优水平，而上游提供商因无法充分获得安全投资创造的剩余而倾向于投资不足。 |
| [^120] | [Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation](https://arxiv.org/abs/2610.21601) | Waypoint-OPD创新性地引入中间教师检查点，利用早期与晚期策略偏移的方向一致性对端点偏移进行门控筛选，从而在学生生成的轨迹上实现更可靠的选择性策略迁移。 |
| [^121] | [LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models](https://arxiv.org/abs/2610.21594) | LoopAhead是一个面向循环Transformer的推测解码框架，通过轻量级前瞻起草器利用早期循环状态提前预测多个未来token，并结合更深层循环的反馈筛选草稿分支，从而显著降低解码延迟。 |
| [^122] | [Transfer Learning in Bandits from Unknown Adaptive Policies: Problem-Dependent Lower and Upper Bounds](https://arxiv.org/abs/2610.21582) | 该论文研究了历史数据来自未知自适应决策策略的多臂老虎机迁移学习问题，推导了问题依赖的渐近遗憾下界，并提出了KL-UCB-Transfer算法，在源域与目标域均值差异满足条件时可改进遗憾的首项常数，否则自动退化为经典KL-UCB算法的保证。 |
| [^123] | [ClaimScope: Label-Blind Witness Verification for Mathematical Claim Auditing](https://arxiv.org/abs/2610.21581) | 该论文提出ClaimScope，一个基于标签盲见证验证机制的72任务数学声明审计基准，实验表明验证器反馈相比无检查器的通用反馈显著提升了反例接受率（23/39 对 15/39）和安全完成率（25/72 对 20/72）。 |
| [^124] | [Learning to Search for Executable Patternmaking from Certified Programs](https://arxiv.org/abs/2610.21579) | 该论文提出 SemDraft——首个无需记录编辑步骤、通过有界搜索并结合确定性编译器的三项认证检查，从成对的前后缝纫纸样中学习恢复可执行服装打版编辑程序的框架。 |
| [^125] | [From Perception to Prediction and Intervention: Benchmarking Proactive Video Assistants in Real-Life Scenarios](https://arxiv.org/abs/2610.21568) | 该论文提出了面向真实生活流媒体场景的主动式视频理解基准P³IBench，用于评测模型的日常感知、风险预测与干预三项能力，发现现有模型在响应决策校准方面存在不足，并构建主动式训练数据训练出P³I-Assistant-4B模型，在多项任务上取得提升。 |
| [^126] | [Sharp and Stable Label-Free Calibration Certificates from Logical Coherence](https://arxiv.org/abs/2610.21567) | 本文利用极小极大定理从逻辑一致性出发，为查询条件概率预测器建立了锐利且稳定的无标签校准残差下界证书，并证明Lipschitz赌注形式既稳定又可通过在线梯度上升在独立同分布数据流上有效优化。 |
| [^127] | [A Quantum Machine Learning Approach to Content-Based Media Recommendation](https://arxiv.org/abs/2610.21562) | 该研究提出了一种基于量子自然语言处理框架的媒体推荐系统，其以仅为经典BiLSTM基线约0.3%的参数量（4.7万 vs 1780万）达到了相当的推荐性能，而经典模型在同等参数预算下性能显著下降，展示了结构化量子表示在参数效率上的优势。 |
| [^128] | [DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents](https://arxiv.org/abs/2610.21561) | DialFork是首个评估对话代理“程序性幻觉”的基准，通过在受控对话轮次位置注入四种幻觉类型构建了涵盖12个领域、24个任务的36,278个对话，并从三个粒度定义了刻画程序性偏差的评估任务。 |
| [^129] | [Reducing the Cost of Qudit State Preparation by Learning a Sparse Structure for SNAP Gates](https://arxiv.org/abs/2610.21557) | 本文提出一种基于可微分掩码的方法，可自动学习SNAP门中所需的相位参数及其取值，通过稀疏性约束显著减少玻色量子位态制备中需要控制的相位数量，从而降低控制成本和误差。 |
| [^130] | [When Does Muon Actually Win? A Paired Data-, Time-, and Tuning-Budget Benchmark](https://arxiv.org/abs/2610.21555) | 在单GPU、配对的数据/时间/调参预算基准测试中，Muon在视觉任务上时间归一化损失显著优于AdamW，但因优化器自身耗时较高（占训练时间56%–70%），在语言任务上反而落后于AdamW。 |
| [^131] | [Uncertainty Quantification in Pairwise Difference Learning for Classification](https://arxiv.org/abs/2610.21554) | 本文提出了一种用于分类任务的成对差分学习方法及配套的元学习技术，并将偶然不确定性与认知不确定性的分解度量扩展到该框架中，从而实现对PDL分类器预测不确定性的忠实量化。 |
| [^132] | [Rethinking Classifier-Free Guidance for TI2I Flow Matching](https://arxiv.org/abs/2610.21550) | 本文发现CFG机制在流建模的噪声阶段对图像编辑有隐性危害，并提出一种在早期流步骤的简单修正方法，无需额外成本即可显著提升TI2I编辑的指令遵循能力和可控性。 |
| [^133] | [When Is Muon Optimal? Radial Fragility and Minimax Geometry](https://arxiv.org/abs/2610.21549) | 该论文引入随机半径各向同性曲率模型与极化因子容许集，精确刻画了Muon缩放极化更新在任意径向定律下保持最优的几何条件，证明非标量最优响应仅来自与曲率扭结对齐的径向原子，因而无原子径向定律呈现零容量刚性。 |
| [^134] | [Finite-Time Guarantees for Constrained Reinforcement Learning under Partial Observability](https://arxiv.org/abs/2610.21548) | 本文首次为部分可观测约束马尔可夫决策过程（CPOMDP）中的强化学习提供了非渐近保证，通过将有限历史窗口表示为约束MDP的状态并开发原始-对偶算法，同时限定了原始问题中的平均奖励次优性与约束违反。 |
| [^135] | [Federated Art Chase: Privacy Leakage and Data Poisoning in Federated Fine-Tuning](https://arxiv.org/abs/2610.21547) | 该论文通过一个可复现的实验流程，系统评估了联邦微调在非IID场景下的安全威胁，表明诚实但好奇的服务器可从单步客户端更新中恢复完整私有标签集，并揭示了联邦微调中的隐私泄露与数据投毒风险。 |
| [^136] | [Activation-Adjacent Layer Normalization: Effects on Approximation Power](https://arxiv.org/abs/2610.21527) | 本文通过宽度增广与归一化的局部线性化技术，精确刻画了激活后归一化（AN）和激活前归一化（PN）网络在仅需常数级额外宽度的情况下逼近任意给定宽度 MLP 的条件与误差界。 |
| [^137] | [When the KV Cache Is Slower Than the GPU: Scaling Remote KV Tiers for Context-Parallel Prefill](https://arxiv.org/abs/2610.21500) | 在长上下文LLM服务中，远程KV缓存层必须提供超过约2.5 GB/s的临界带宽（B* = sR）才能比重算提示词更快，否则缓存命中反而更慢，且同步加载会让GPU几乎闲置、p99尾延迟从4秒暴涨至83秒。 |
| [^138] | [Phase Kernel Lifts Capacity of Dense Associative Memory](https://arxiv.org/abs/2610.21493) | 本文提出一类非线性相位核并构建S-Hop网络，在保持指数级存储容量和收敛保证的同时大幅提升霍普菲尔德网络的实际检索负载，在ImageNet上首次失败负载提升达50倍。 |
| [^139] | [Kolmogorov-Arnold Networks: Curse of Dimensionality and Structural Bias](https://arxiv.org/abs/2610.21480) | 该论文证明了标准样条-KAN 在最坏情形下仍受维度灾难制约——所需样条系数数量随输入维度指数增长，表明 KAN 的维度无关逼近优势源于对目标函数类的假设而非架构本身，且 KAN 逼近器可精确等价转换为三次 ReLU 激活的 MLP。 |
| [^140] | [DuET: Note-Frame Dual Encoding for Piano Performance-to-Score Transcription](https://arxiv.org/abs/2610.21467) | 提出DuET双编码器Transformer，通过音符与帧的双向交叉注意力及TimeRoPE、TPRoPE位置编码，将音符级钢琴MIDI转谱与帧级节拍跟踪联合建模，提升多种钢琴流派的乐谱转录效果。 |
| [^141] | [Trajectory-Level Speculation with Semantic Verification for Tool-Using Agents](https://arxiv.org/abs/2610.21463) | SpecAgent将强模型的角色从逐步生成者转变为弱模型草稿的审阅者，弱模型在起草时执行只读工具调用以确保每一步基于真实观测，强模型则通过单次预填充对草稿片段进行语义验证并提交最长一致前缀，从而实现高效的轨迹级推测协作。 |
| [^142] | [V-SHARE: Provable Convergence Guarantees for Advantage-based Algorithms](https://arxiv.org/abs/2610.21448) | 本文提出V-SHARE算法，首次为基于优势的价值学习方法提供了完整的理论收敛保证（包括渐近收敛和非渐近有限样本保证），并证明其能缓解异步Q学习的稀有动作瓶颈，同时揭示了收缩性与方差之间的权衡。 |
| [^143] | [Each Reward to Its Relevant Tokens: Reward-Specific Token-Level Credit Assignment for Multi-Reward RL](https://arxiv.org/abs/2610.21439) | 提出MRTPO方法，通过对比有/无验证器信息时的下一词元分布的Jensen-Shannon距离，将每个奖励的优势精准分配到其相关词元上，无需辅助模型或词元级标注即可实现多奖励强化学习中的奖励专属词元级信用分配。 |
| [^144] | [HyCo: Hybrid Drafting for Cost-Efficient Mobile-Cloud Speculative Decoding](https://arxiv.org/abs/2610.21436) | HyCo提出混合草稿生成机制，将服务器模型的原生多词预测头部署到移动NPU上并与小型语言模型单次推理协同工作，配合量化和ε阈值松弛策略，使移动-云推测解码的服务器GPU时间降低约29%-39%。 |
| [^145] | [Policy-Aligned Predictive Verification for Adaptive VLA Action-Chunk Execution](https://arxiv.org/abs/2610.21434) | 本文提出策略对齐预测验证器（PAV），在执行前于VLA内部任务条件表征空间中预判候选动作前缀的后果并评估其与任务成功后续行动的兼容性，从而实现更可靠的自适应动作块执行决策。 |
| [^146] | [ManiFusion: Unlocking the High-Throughput Generation via Superposition in Manifold Space](https://arxiv.org/abs/2610.21433) | ManiFusion提出了一个与骨干网络无关的框架，通过在去噪骨干前后对多个样本进行融合与解融合，并采用轨迹融合调度来对齐去噪阶段，从而在单次去噪中并行处理多个样本，突破单样本延迟优化的瓶颈，实现高吞吐量的扩散模型图像生成。 |
| [^147] | [What Does History Add? Calendar Controls and Information-Timing Audits for University Trajectory Forecasting](https://arxiv.org/abs/2610.21419) | 该研究通过严格的日历长度控制与信息时机审计发现，大学轨迹预测中六个月历史摘要相比仅使用历史长度指标并无显著的额外预测增益，但全历史摘要确实优于其对应的日历控制。 |
| [^148] | [Learning to Forecast on Multiple Spatiotemporal Hierarchies](https://arxiv.org/abs/2610.21414) | 提出了一种图深度学习框架，将多个时空层级结构之间的差异作为多任务正则化进行联合训练，从而提升时空预测的效果。 |
| [^149] | [Multi-Reward Hacking: From Aggregation-Induced Drift to Trust-Anchored Defense](https://arxiv.org/abs/2610.21404) | 论文揭示了对称聚合多奖励会引发“多奖励破解”现象——即聚合奖励提升而主任务性能反而下降，并证明了确定性条件均值GRPO在辅助奖励更易获得时必然收敛到偏向辅助的行为，同时提出信任锚定门控（TAG）作为防御方法。 |
| [^150] | [Beyond Invariance: Factor-wise Selective Treatment of Shifts for Multimodal Learning](https://arxiv.org/abs/2610.21403) | 提出FaShift框架，超越传统不变性方法，通过按因子诊断分布变化、干预效应和跨环境稳定性，对多模态学习中的偏移信息进行选择性抑制、保留或削弱。 |
| [^151] | [Fault Detection and Bit-Error Rate Diagnostics in Optical Wireless Links Using Edge-AI](https://arxiv.org/abs/2610.21402) | 本文提出了一种基于ESP32-S3微控制器与TinyML的自主自诊断光无线接收节点，可在设备端实时提取误码率等信道特征并实现故障检测，无需依赖云端或外部计算资源。 |
| [^152] | [Self-Guided LLM Reasoning for Error Diagnosis in Vehicle Routing Problems](https://arxiv.org/abs/2610.21393) | 本文提出一种自引导推理框架，让大语言模型自主生成代码以提供结构化见解，从而辅助诊断车辆路径问题实例中的错误，将错误检测性能显著提升至接近完美准确率。 |
| [^153] | [Calibration Language Shapes Block-Influence Pruning in Multilingual Language Models](https://arxiv.org/abs/2610.21383) | 该论文首次系统揭示剪枝校准语言会显著影响块影响排序——印度语言的深层块BI比英语高出20.7%-55.5%，导致剪枝选块不同，而多语言平均校准可有效缓解剪枝带来的困惑度损失。 |
| [^154] | [RDL-Bench: A Benchmark for Task-Aware Question Answering over Relational Databases](https://arxiv.org/abs/2610.21382) | RDL-Bench是一个针对真实关系数据库的任务感知问答基准，它将用户问题分为由当前数据库状态决定的推理型问题和需要在时间约束下基于历史记录进行预测的预测型问题，填补了现有基准只孤立处理这些目标的空白。 |
| [^155] | [Does Compression Imply Generalization? A Minimum Description Length Perspective](https://arxiv.org/abs/2610.21376) | 本文基于最小描述长度原则提出仅由训练轨迹即可计算的预quential后悔，证明其在（近乎）插值情况下上界了训练参数在算法诱导分布下的描述长度，进而转化为PAC高概率泛化界，为“压缩是否意味着泛化”提供了无需保留数据或互信息估计的新答案。 |
| [^156] | [Memorilla: Latent Semantic Memory for LLMs](https://arxiv.org/abs/2610.21375) | Memorilla提出了一种潜在语义记忆系统，能将整个文档集合的信息压缩为仅16个依赖查询的记忆token并注入冻结LLM的输入嵌入空间，在八个基准中的五个上超越了Mem0和MemoRAG等基线方法。 |
| [^157] | [Same Target, Different Origins: Aligning Latent Rollout for Consistent PDE Emulation](https://arxiv.org/abs/2610.21370) | 提出ALR正则化方法，通过对神经PDE求解器的潜在表示施加零阶状态一致性与一阶动力学一致性约束，使不同起点和深度的自回归推演在同一目标时刻收敛到一致的预测，从而抑制误差在多步推演中的传播。 |
| [^158] | [Two Sample Sizes in Test-Time LLM Ensembling: Buffered ERM and Minimax Theory](https://arxiv.org/abs/2610.21363) | 该论文研究了测试时大语言模型加权集成中标注问题数 Q 与每问题生成数 N 这两个样本量的相互作用，证明带校准间隔缓冲的 ERM 可达到 Q^{-1/2}+N^{-β/2} 阶的超额风险，并通过简化的双模型子模型分析验证了该风险率的最优性。 |
| [^159] | [L2GSSL: Label-Efficient Local-to-Global Self-Supervised Learning for Volumetric Healthcare Data](https://arxiv.org/abs/2610.21356) | 提出L2GSSL局部到全局自监督学习框架，通过将预训练分解为切片内和切片间表示空间并利用停止梯度隔离目标，在仅使用10%标注的肺结节恶性分类任务上达到0.9229 AUROC，大幅超越体积级对比方法，接近全监督基线性能。 |
| [^160] | [Honesty is the Best Policy ... or is it? Demand Management for Probabilistic Service Options](https://arxiv.org/abs/2610.21351) | 该论文将概率性服务选项下的需求管理建模为序列决策过程，通过递归定义真实概率并引入透明度约束，首次系统研究了提供商操纵展示概率这一新型需求管理杠杆及其与诚实披露之间的权衡。 |
| [^161] | [Online Quantile Omniprediction for Proper Losses](https://arxiv.org/abs/2610.21342) | 本文提出在线样本高效的分位数预测算法，使预测在所有满足温和正则条件的正当损失下同时保持准确，将全能预测从二分类任务拓展到多分类和连续型结果。 |
| [^162] | [Dimension-Free PAC-Bayes Generalization for Deep Neural Networks via Dirichlet Processes](https://arxiv.org/abs/2610.21340) | 该论文通过狄利克雷过程将PAC-Bayes分析的先验从高维参数空间转移到数据分布空间，并借助数据处理不等式证明参数空间相对熵被可闭式计算的数据空间DP散度严格界定，从而获得了与网络维度无关的深度神经网络泛化界，解决了过参数化网络泛化界空洞的问题。 |
| [^163] | [Does On-Policy Distillation Need to Complete Every Rollout?](https://arxiv.org/abs/2610.21337) | 该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。 |
| [^164] | [Ranking and Partitioning of Attention Heads for Sparse and Continual Reinforcement Learning Fine-tuning](https://arxiv.org/abs/2610.21336) | 该论文提出按梯度范数对注意力头排序并在任务间划分的方法，实现了语言模型的高效稀疏强化学习微调，同时减少跨任务干扰、提升持续学习中的能力保持。 |
| [^165] | [When Should a Low-Light Pipeline Generate? Auditable Risk Control for Selective Generative Enhancement](https://arxiv.org/abs/2610.21335) | VetoNet提出了一种以认证否决为核心的可审计区域路由框架，通过无分布假设的危害预算证书，在低照度增强中自适应地决定生成专家何时可以安全地覆盖冻结的复原专家，从而在控制感知风险的前提下实现结构与纹理的最优平衡。 |
| [^166] | [Semantic Latent Alignment for Flow Matching with Invertible Bi-Lipschitz Adapters](https://arxiv.org/abs/2610.21332) | 提出一种可逆双利普希茨适配器，仅改变预训练自编码器中固定的编码器与解码器之间的坐标，将其潜在空间离线对齐到语义教师模型，从而在保持压缩与重建性能的同时改善流匹配生成的语义组织。 |
| [^167] | [Spectrum Is Not Enough: A Random-Matrix Account of Data-Free Preconditioning for Differentially Private Learning](https://arxiv.org/abs/2610.21312) | 该论文用随机矩阵理论刻画了差分隐私学习中无数据 K-FAC 预条件化的可重构性，证明了 Kronecker 分解 $A \otimes G$ 中的误差因子 $G$ 和激活因子 $A$ 的架构决定部分可由合成探针在无数据情况下重构，只有 $A$ 中由数据引起的特征值尖峰不可重构，并精确定位了其在 Baik-Ben Arous-Péché 阈值 $(d/n)^{1/4}$ 处的失效边界。 |
| [^168] | [PocketTSLM: Deploying an On-Device Soft-Prompt Time-Series Language Model for Privacy-Preserving Smartphone Reasoning](https://arxiv.org/abs/2610.21308) | 提出PocketTSLM，一个完全在消费级智能手机上运行的设备端软提示时间序列语言模型移动原生部署方案，在保留时间序列结构的同时实现隐私保护的生理数据推理。 |
| [^169] | [Personalized Projected Ascent for Federated Stackelberg Optimization](https://arxiv.org/abs/2610.21295) | 该论文提出在联邦Stackelberg优化中让最大化变量保持客户端个性化并采用投影上升，证明了非凸-凹情形下达到标准次线性收敛速率，并针对非凸-PL情形中随机更新导致的与步长无关的噪声下限，提出方差缩减方法VR-SGDA加以缓解。 |
| [^170] | [Can Capability Enhancement be Localized?](https://arxiv.org/abs/2610.21294) | 该论文提出“能力增强局部化”（CEL）概念，通过两阶段的筛选与联合消融方法，实证发现移除少数注意力头组成的稀疏组合即可在无需微调的情况下提升模型在特定任务上的性能。 |
| [^171] | [TimeCap: A Scalable Open-Source Pipeline for Captioning Wearable Health Time-Series Data](https://arxiv.org/abs/2610.21274) | 该论文提出了TimeCap，一个开源、确定性、可扩展的时间序列描述生成流水线，通过三级可复用提取器和跨通道综合层，为可穿戴健康时间序列数据生成基于信号、可复现且可扩展的文本标注，填补了时间序列语言模型训练语料的空白。 |
| [^172] | [Regret Lower Bound for Online Learning of Quantum States with Logarithmic Loss, with Applications to Quantum State Tomography and Confidence Sequences for Matrix Means](https://arxiv.org/abs/2610.21273) | 本文建立了基于对数损失的量子态在线学习的最优遗憾下界，并由此导出样本量最优（至多相差多对数因子）的量子态层析成像算法，以及矩阵均值置信序列的构造算法。 |
| [^173] | [Rethinking Learning from Label Proportions via Moment Matching](https://arxiv.org/abs/2610.21262) | 该论文提出了一种简单有效的基于矩匹配的标签比例学习方法（LLP-MM），在袋独立同分布采样这一更现实的假设下，利用标签计数服从泊松多项式分布并通过多阶矩匹配，显著改善了非随机分袋场景下的分类性能。 |
| [^174] | [Behavioral Compositionality of Sparse Autoencoder Representations](https://arxiv.org/abs/2610.21261) | 论文提出SAE可加性残差（SAR）来度量稀疏自编码器特征在行为层面的组合性，发现学习到的SAE坐标比随机旋转具有更强的一阶可加性，但仍存在不可忽略的高阶交互项。 |
| [^175] | [Fractional Power-of-Two Quantization for Efficient and Effective Multiplier-Free LLM Inference](https://arxiv.org/abs/2610.21257) | 该论文提出分数二次幂量化（FPoT），采用以√2为基数的半八度分数量化网格，在保持完全整数移位-加法数据通路实现无乘法器推理的同时，解决了传统二次幂量化在低位宽下的精度下降问题，实现了实用的4位LLM权重量化。 |
| [^176] | [Isolating a Language Model's Judgment with a Label-Free Subspace](https://arxiv.org/abs/2610.21256) | 该研究提出一种无需标签的方法，通过识别并移除隐藏状态中的低维“表达子空间”，成功隔离出语言模型内部稳定的判断信号，使其在不同指令变化下仍保持可靠。 |
| [^177] | [System1-Arena: Which Decisions Can Be Left to System 1?](https://arxiv.org/abs/2610.21244) | 提出System1-Arena基准，通过在105个任务、92k个问题上统一评测系统1（单次前向传播输出概率）与系统2（生成式推理）模型，并结合138个闭环环境，系统性地回答哪些决策可以放心交给系统1模型。 |
| [^178] | [Representation Redistribution in Sparse Autoencoders](https://arxiv.org/abs/2610.21240) | 该论文提出再分配敏感度分数（RSS）来识别在提示变换下以一致方向重复出现的大幅特征响应，并在涵盖八个语言模型的27个预训练SAE上验证了表征再分配的可复现性，同时将重复的表征变化与下游读出贡献及决策失败精确解耦。 |
| [^179] | [Compatibility Matters: A Framework for Token-Level Teacher Selection in On-Policy Distillation](https://arxiv.org/abs/2610.21234) | 该论文提出在在线策略蒸馏中按词元动态选择教师的框架，从理论上刻画了师生兼容性、教师能力与学生多样性等关键因素，并证明不同场景下不同教师路由方案各有优势。 |
| [^180] | [Not All Change Is Damage: Update-Matched Unlearning for Large Language Models](https://arxiv.org/abs/2610.21230) | 该论文提出“参照时序假设”，证明在大语言模型遗忘任务中，采用更新匹配的参照比更新前的参照能更好地保留模型学习新任务的可塑性，从而有效区分对已删知识的选择性抵抗与整体可塑性丧失。 |
| [^181] | [What Should We Remove? Evaluating the Reliability and Costs of RLVR Data Decontamination](https://arxiv.org/abs/2610.21229) | 该研究首次系统审计了RLVR数学语料库的去污染策略，发现语义匹配去污在余弦阈值0.80下有88%的抽样移除被证实为误移除，且没有任何预注册语义阈值能达到90%保守精确率目标，揭示了当前数据清洗政策存在严重的误删风险与数据损失成本。 |
| [^182] | [Learning and Forecasting Multiscale Spatiotemporal Epidemic Dynamics with Lagrangian-Eulerian Graph Neural ODEs](https://arxiv.org/abs/2610.21223) | 提出LE-EpiGNODE模型，以机制引导的拉格朗日-欧拉图神经常微分方程取代通用消息传递，将年龄-接触信息用于感染生成、人口流动用于保守性人口运输，在中长期流行病时空预测中取得显著优势。 |
| [^183] | [Clean Data Can Still Carry Backdoors: Support-Persistent Backdoors for Model Reuse](https://arxiv.org/abs/2610.21208) | 本文提出DAR后门攻击方法，通过挖掘数据中天然存在的低维子空间模式并进行标签投毒，使后门在干净数据知识蒸馏后依然存留，打破了“知识蒸馏可有效净化后门”的普遍认知。 |
| [^184] | [Hard Way, The Only Way: Provable Merits of Training on Hardest Examples](https://arxiv.org/abs/2610.21201) | 该论文通过加权经验风险最小化建立了数据剪枝的精确高维理论，并证明在样本量远超维度的情形下，仅保留最难样本进行训练是可证明的最优剪枝策略。 |
| [^185] | [Poisoning the corpora of protein design systems during pre-training and retrieval](https://arxiv.org/abs/2610.21200) | 本研究首次证明，仅需在蛋白质语言模型的预训练语料库或文本引导蛋白质生成器的检索语料库中注入少量（低至200条）含触发器的投毒序列，即可使模型产生攻击者可控的特定序列生成行为，揭示了蛋白质设计AI系统面临的严重数据安全威胁。 |
| [^186] | [Resisting Harmful Instructions through Training-Free Weight Editing](https://arxiv.org/abs/2610.21189) | 本文提出一种无需训练的权重编辑方法，仅需在少量提示对上执行前向传播即可修改模型权重，使模型抵抗指令性谎言和性别歧视等有害指令，且这种防御无法通过修改提示来绕过。 |
| [^187] | [Sensor Diagnosis for Fuel and Emissions Monitoring in Heavy-Duty Diesel Vehicles](https://arxiv.org/abs/2610.21188) | 本文提出基于健康测量数据学习关系的传感器故障诊断与燃油修正程序，在重型柴油发动机217个稳态运行点上对五种合成故障类型评估，实现了77.3%的诊断覆盖率且诊断准确率达92.1%，有助于修正因传感器故障导致的燃油消耗和碳排放估计偏差。 |
| [^188] | [Mine, Sample, Render: Test-Time Diversity Alignment Of Generative Models](https://arxiv.org/abs/2610.21185) | 提出 MSR 测试时框架，通过从真实语料库挖掘类别模式并采样属性配置，将生成模型仅作为条件渲染器，从而显著提升生成输出的多样性与语料库分布的对齐度。 |
| [^189] | [Judging the Judges: Confidence and Disagreement in Rubrics as Rewards](https://arxiv.org/abs/2610.21179) | 该论文提出 RaRE 评判者集成策略，通过基于置信度的软 Likert 奖励和剔除高分歧样本的掩码机制，显著提升了基于评分标准的强化学习在不可验证领域中的训练效果。 |
| [^190] | [Language Models Share a Readout Bottleneck in Working Memory](https://arxiv.org/abs/2610.21176) | 该论文发现，语言模型与人类共享同一工作记忆失败机制——错误并非源于注意力定位不准（精度不足），而是正确的目标信息在“读出”阶段受到干扰，即语言模型存在一个共同的工作记忆读出瓶颈。 |
| [^191] | [Coordination-Free Workload Regularization for Variable-Cost Distributed Training](https://arxiv.org/abs/2610.21162) | 提出无协调数据加载策略BCL，通过均衡各rank批次的估计成本将批次成本变异性降低高达82.8%，在无需跨rank通信或运行时修改的情况下实现最高1.343倍的训练加速。 |
| [^192] | [When is a better LLM worth paying for? Stochastic Optimization with Switching Mixed-Fidelity Oracles](https://arxiv.org/abs/2610.21139) | 针对“何时值得为更强的LLM付费”这一核心问题，本文提出了两个成本感知的随机优化算法框架，通过在具有系统性偏差和状态依赖方差的混合保真度一阶预言机之间自适应切换，来平衡LLM调用成本与下游优化精度。 |
| [^193] | [Pathwise Shrinkage and Probabilistic Calibration of the Multiplicative Half-Cauchy Process](https://arxiv.org/abs/2610.21136) | 本文通过对数尺度随机游走表示系统刻画了乘性半柯西过程的收缩、扩张与临界等路径性质，并推导出连接可解释先验概率陈述与尺度参数的半解析校准规则。 |
| [^194] | [Predicting Live Birth After Fresh Embryo Transfer in IVF/ICSI: Differential ICM/Trophectoderm Morphology Encoding and a Hybrid Attention--Logistic Regression Ensemble](https://arxiv.org/abs/2610.21134) | 该研究构建了单中心IVF/ICSI数据集，通过新颖的内细胞团/滋养外胚层差异化形态编码方法以及混合注意力网络与逻辑回归的集成模型，实现了适用于小型临床表格数据的鲜胚移植活产预测。 |
| [^195] | [Noise-Space Reward Alignment for Few-Step Generative Models](https://arxiv.org/abs/2610.21125) | 提出噪声策略优化（NPO），通过在生成器初始噪声空间上学习摊销流策略、将噪声分布向奖励倾斜，从而在保持少步生成效率优势的同时解决少步生成模型的奖励对齐问题。 |
| [^196] | [DST-Net: Decomposed Spatio-Temporal Network for Time Series Forecasting](https://arxiv.org/abs/2610.21123) | 提出DST-Net解耦时空网络，摒弃对分解后各分量的同质化建模，针对趋势分量设计潜在图块以捕获全局拓扑依赖与协同运动，针对季节分量设计高斯多尺度注意力机制以聚焦局部时间演化。 |
| [^197] | [Spectral-Spatial Interpretation](https://arxiv.org/abs/2610.21114) | 提出 SSI 方法，通过在每个空间位置对通道构建条件博弈并计算 Shapley 值，生成完整的光谱-空间归因矩阵，实现了满足 Shapley 公平性且具有清晰局部解释性的多光谱模型解释。 |
| [^198] | [Kolab: Benchmarking Human-Agent Collaboration in Long-Horizon Workflows](https://arxiv.org/abs/2610.21109) | 本文推出Kolab数据集，通过记录50位来自40个职业的专业人士与AI智能体协作处理真实长周期任务的115小时以上屏幕录像，揭示了人类更重视与智能体的持续交互而非一次性自动化完成，说明现有侧重自动化的基准测试无法衡量真实世界的人机协作效果。 |
| [^199] | [ToolBottleneckBench: Measuring Tool-Agent Performance under Controlled Pressures and Fixed Harness Budgets](https://arxiv.org/abs/2610.21098) | 该论文提出ToolBottleneckBench基准，在固定框架预算下通过单独变化工具目录、可见菜单、认证工作流或返回记录编码等压力因素来测量工具智能体性能，发现可见工具菜单是唯一对模型造成大幅惩罚的压力且其根源是完成上限处的截断，而大多数失败实际终止于框架限制而非模型能力不足。 |
| [^200] | [Probe-and-Resume: When and How to Evaluate Agent Execution](https://arxiv.org/abs/2610.21088) | 该论文发现在终端类计算机使用环境中，MCTS测试时搜索失效的原因是评估器在中间状态下缺乏足够证据进行判断，因此提出仅在智能体认为任务完成的节点才调用评估器的探测与恢复策略。 |
| [^201] | [SIEVE: Overcoming Topological Obstruction in Equivariant Self-Supervised Learning](https://arxiv.org/abs/2610.21087) | 该论文发现3D点云掩码自编码与SE(3)等变网络间存在拓扑坍塌障碍，并提出SIEVE框架，利用球面投影掩码将点云几何压缩为不变标量嵌入，从构造上克服该障碍并在下游任务尤其是跨类别泛化上取得显著提升。 |
| [^202] | [StochLOB: Heavy-Tailed Score Matching for Robust Limit Order Book Trend Prediction](https://arxiv.org/abs/2610.21081) | 该论文提出StochLOB训练框架，用跳跃Lévy噪声和截断α稳定噪声等重尾非高斯扩散过程替代传统的布朗高斯噪声来加噪训练数据，以更好地捕捉限价订单簿订单流的重尾与跳跃特性，从而提升LOB趋势预测的稳健性，同时推理阶段保持确定性预测器。 |
| [^203] | [Adaptive Randomized Neural Networks for PDEs: Structure-Aware Feature Redistribution and Residual-Driven Block Enrichment](https://arxiv.org/abs/2610.21077) | 该论文提出两阶段自适应随机神经网络方法SFRBE-RaNN，通过结构感知特征重分布（SFR）与残差驱动的块增强（RBE）自适应构建试探空间，以更有效捕捉偏微分方程解的空间与尺度结构，并为固定线性问题提供精确的残差缩减恒等式与误差估计。 |
| [^204] | [Score Mismatch and Set-Constrained Sampling in Zero-Shot Conditional Discrete Diffusion](https://arxiv.org/abs/2610.21061) | 本文首次对分数失配的离散扩散模型进行了理论分析，推导出一般的KL散度上界来量化失配造成的额外偏差，并覆盖文本填充、句子补全等部分观测约束场景，为零样本条件生成提供了理论保证。 |
| [^205] | [Bellman-Consistent Trajectory RL for LLM Arithmetic Reasoning](https://arxiv.org/abs/2610.21060) | 提出贝尔曼一致的轨迹强化学习方法TBRM-RLOO，通过轨迹贝尔曼残差最小化将稀疏奖励转化为密集词元级信号并降低梯度方差，在Countdown算术推理任务上达到最先进的76.0–84.0% Pass@16准确率，显著超越GRPO、SFT、IPO/DPO和RLOO等基线。 |
| [^206] | [Athanor: An Agent-Native System for Autonomous Post-Training Research](https://arxiv.org/abs/2610.21055) | Athanor是一个开放的智能体原生系统，赋予AI智能体完全的研究自主权，让它们在真实GPU上自主完成小型基础模型的后训练研究，以此检验智能体迈向递归自我改进的研究潜力。 |
| [^207] | [SALTO: Solver-Anchored Lifted Transfer Operators for Nonlinear Event Inference](https://arxiv.org/abs/2610.21049) | SALTO是一个基于Koopman算子的框架，通过在提升空间中构建可复用的线性化伴随算子，仅需一次编码器前向传播和一个内积即可快速推断任意有限幅度的局部非线性事件源。 |
| [^208] | [SHIP: Structured and Hierarchical Insight with Privacy](https://arxiv.org/abs/2610.21048) | 该论文提出 SHIP 框架，将差分隐私洞察形式化为可重用表示，通过自动查询-属性映射、分层混合分类体系和差分隐私合成表格，在保护隐私的同时有效回答细粒度的分析查询，并建立了统一的效用评估协议。 |
| [^209] | [PremiseCheck: Efficient Fact-Checking of Human and AI Peer Reviews](https://arxiv.org/abs/2610.21046) | PremiseCheck通过让LLM在不读论文的情况下将评审意见拆解为显式前提、再用廉价判定模型逐一与论文核对，在保持与前沿LLM相当的核查准确率的同时将成本降低至约一半（NLP领域仅为四分之一）。 |
| [^210] | [INCONVENIENT: A Benchmark to Assess LLM Generalization Capabilities from Convenience Samples](https://arxiv.org/abs/2610.21038) | 提出了INCONVENIENT基准，通过显式构造并隔离便利样本偏差，评估模型在有偏采样的表格数据上向目标总体泛化的能力。 |
| [^211] | [Efficient Agnostic Learning of Univariate Densities Under Local Differential Privacy](https://arxiv.org/abs/2610.21033) | 本文提出了一个非交互式局部差分隐私学习框架，实现了对可被分段多项式逼近的单变量分布（如高斯混合模型）首个样本高效且计算高效的不可知学习，并为此开发了可独立使用的大域LDP合成数据生成算法。 |
| [^212] | [On the Last-Iterate Convergence of Clipped Gradient Methods](https://arxiv.org/abs/2610.21026) | 本文针对梯度噪声为重尾分布这一更贴近实际的情形，研究并证明了裁剪梯度方法（如Clipped-SGD）在随机（强）凸优化中的末次迭代收敛性，克服了现有SGD末次迭代收敛结果依赖强噪声条件（如指数衰减尾部）的局限。 |
| [^213] | [Learning to Read Out: Unembedding Dynamics in Language Model Pretraining](https://arxiv.org/abs/2610.21025) | 提出参数轨迹交叉编码方法，仅利用检查点权重直接追踪语言模型预训练中读出层的动态，发现Pythia在预训练0.4%–1.1%的早期窗口内读出层发生重组，且隐藏状态在读出层表达之前就已编码任务信息。 |
| [^214] | [Acceleration and its Limits in Gaussian Variational Inference](https://arxiv.org/abs/2610.21020) | 该论文证明高斯变分推断在可精确计算高斯期望时能达到经典的 √κ 加速收敛率，但通过构造一族强对数凹目标证明仅有逐点导数时任何算法都需要 κ^(1-o(1)) 次查询，从而说明仅凭曲率界不足以保证加速。 |
| [^215] | [CausalEvo: Causal Skill Self-Evolution for Health LLM Agents](https://arxiv.org/abs/2610.21016) | 提出CausalEvo方法，通过在固定模型权重、反馈案例和规则数量的条件下估计技能集变更的相对因果效应，实现健康LLM智能体的技能自进化筛选，保留有效规则并防止错误泛化的传播。 |
| [^216] | [Audio-Visual Dog Identification and Tracking in Real-World Videos](https://arxiv.org/abs/2610.21011) | 提出了一种结合视觉识别、多目标追踪和犬类声纹识别的多模态犬只指纹框架，可在真实世界视频中从吠叫声中识别并追踪个体犬只，并为其维护持续更新的身份档案。 |
| [^217] | [Can LLMs Anticipate Their Cruxes?](https://arxiv.org/abs/2610.21006) | 本文提出将模型决策中的“关键症结”定义为模型多次推演分布上产生分歧、且改变其取值可能改变结论的子问题，并比较了三种计算成本递增的策略（先验预测、事后推理提取、多次生成）来识别这些症结。 |
| [^218] | [Imperfect Is Perfect: Learned Column Prefilling for Branch-Price-and-Cut](https://arxiv.org/abs/2610.21001) | 该论文揭示了“恰当的不完美”这一反直觉发现——用神经网络在数秒内生成的不完美普通列（而非最优列）作为预填充，并预测Wentges平滑初始中心，反而能显著加速分支定价切平面算法的根节点求解。 |
| [^219] | [EigDiff: Eigensample Diffusion - or how to significantly reduce training dataset size with marginal performance variation?](https://arxiv.org/abs/2610.20999) | 提出EigDiff方法，在预训练CLIP模型的潜在空间中执行PCA获取特征向量，并以此为指导选取由样本代表与多样化图像构成的代表性训练子集，从而在性能几乎不变的前提下大幅缩减扩散模型的训练数据集规模。 |
| [^220] | [One Bias Score, Two Factors: How Mitigations Affect Option Bias in Multiple-Choice Evaluation](https://arxiv.org/abs/2610.20993) | 该论文提出用合并选择率标准差统一选择偏置与顺序敏感性两种视角，并证明该指标可精确分解为翻转率（反映模型不确定性）和位置偏斜（反映选项符号先验）两个无标签因素，从而揭示了现有缓解方法影响选项偏置的内在机制。 |
| [^221] | [Do As I Do, Not As I Say: On-Policy Distillation Inherits the Teacher in Vision-Language Models](https://arxiv.org/abs/2610.20989) | 该研究发现，在线策略蒸馏（OPD）只有在教师模型对学生自身生成的轨迹保持可靠时才有效——学生会直接继承教师在其所处状态上的行为（包括错误），因此教师对学生的状态分布是否可靠是OPD成败的关键。 |
| [^222] | [The Blume-Capel Model as a $\phi^6$ Field Theory: A Monte Carlo and Flow Model Study](https://arxiv.org/abs/2610.20982) | 本研究将归一化流生成模型用作无自相关的有效采样器，成功研究了连续自旋Blume-Capel模型（$\phi^6$场论）的相图，并以与正则及扩展系综蒙特卡罗相当的精度确定了一阶/二阶相变和三临界点的位置。 |
| [^223] | [How to Train Your Little Dragon: Teaching a Small Language Model to Construct and Refine GPU Optimization Scripts](https://arxiv.org/abs/2610.20980) | 该论文提出了一种训练小型语言模型的方法，使其学会生成和迭代改进GPU内核优化脚本，通过设计紧凑无损的脚本表示以及基于TVM MetaSchedule构建的十余万条训练样本，实现了自动化的内核优化脚本构造与精炼。 |
| [^224] | [Two-Stage Score Estimation for Group-Specific Diffusion Models](https://arxiv.org/abs/2610.20979) | 该论文提出了一个针对群体特定扩散模型的两阶段得分估计统计框架，通过将共享池化得分与目标特定偏移量分离分析，推导出收敛速率，并证明扩散模型可作为密度估计器获得全变差保证。 |
| [^225] | [MIDI-REPA: Improving Text-to-Music Generation with Symbolic Representation Alignment](https://arxiv.org/abs/2610.20967) | MIDI-REPA通过自动音乐转录提取MIDI符号监督信号，并将其与流匹配DiT早期层的中间表示进行对齐，从而在无需额外数据且仅用单张RTX 3090训练约31小时的低成本条件下，显著提升文本到音乐生成的音乐结构质量。 |
| [^226] | [Cross-subject decoding of imagined speech from scalp EEG: a leakage-controlled evaluation on three public corpora](https://arxiv.org/abs/2610.20954) | 该研究在三个公开脑电数据集上采用严格防数据泄漏的评估协议，发现头皮EEG想象语音解码无法跨被试迁移，并系统检验了被试内解码准确率受限的多种可能解释。 |
| [^227] | [LEV: Thinking Fast and Deep with Looped Decision Models](https://arxiv.org/abs/2610.20951) | LEV是一种循环决策模型，通过反复运行模型层级而非生成token来实现按需可变的思考深度，单个模型即可在任意深度作答，且准确率随深度提升，尤其在困难问题上收益显著。 |
| [^228] | [Ranked Right, Sized Wrong: Ensemble Spread Fails Silently at Streamflow Extremes](https://arxiv.org/abs/2610.20947) | 研究发现深度学习径流预报模型的集合离散度虽能正确排序预测误差，但其量级随流量增大而系统性低估误差（平常流量下差2倍、极端流量下可达18倍），表明该不确定性信号在径流极端事件中会悄然失效。 |
| [^229] | [Planning to Learn: Real-Time Robust Adaptive Control on the GPU with Guaranteed Future Learning](https://arxiv.org/abs/2610.20925) | 提出A-SLS框架，通过预测未来参数不确定性并构建考虑未来信息增益的紧致可达集，结合GPU加速求解器，实现了具有未来学习保证的实时鲁棒自适应控制，并可扩展至四足机器人全身控制。 |
| [^230] | [Defining and Generating Compositional Structure](https://arxiv.org/abs/2610.20922) | 该论文利用算法信息论将组合结构形式化定义为“当部分被整体考虑时会显现出单独建模时无法捕捉的共享结构”，从而为在任意数据中识别和生成这种结构提供了数学框架。 |
| [^231] | [Recursive Offline Distillation via Entropy Gain Sampling and Answer Mixing](https://arxiv.org/abs/2610.20914) | 该论文提出一种递归离线蒸馏方法，通过以与辅助中等模型的熵差异为权重采样推理链（熵增益采样），并用单token真实答案补充思维轨迹（答案混合），在相同数据预算下显著优于朴素蒸馏并呈现更陡峭的学习曲线。 |
| [^232] | [Replicable Dueling Bandits](https://arxiv.org/abs/2610.20908) | 该论文首次为可复制的对决老虎机设计了具有次线性遗憾的算法，在松弛随机传递性下实现 $\widetilde{O}(\gamma^8 K^4 H_p/\rho^2)$ 的期望遗憾，并在强随机传递性下通过同时淘汰策略改善了对臂数 $K$ 的依赖。 |
| [^233] | [Task-Centric Multimodal Learning for Cognitive Impairment Detection](https://arxiv.org/abs/2610.20906) | 本文提出一种基于Amazon Alexa的非接触式多模态筛查框架，融合视频、音频和文本信号，全面捕捉语音模式、面部表情和语言结构等行为特征，以更准确地检测老年人的认知衰退。 |
| [^234] | [Sign Betting for Misspecified Regression: Variance Adaptivity and Optimal Approximation](https://arxiv.org/abs/2610.20903) | 本文证明符号投注回归在模型误设下可实现高概率的 $L_1$ 预言机不等式，其误差界自适应于平均条件方差，并证明其对恰当学习所能达到的普适逼近因子 3 是最优的。 |
| [^235] | [TGrasp6D: Task-Informed Vision-Language Priors and Pose Refinement enable Zero-Shot Task-Oriented 6DoF Grasping](https://arxiv.org/abs/2610.20901) | 该论文提出TGrasp6D，一种基于视觉-语言模型的零样本面向任务6自由度抓取方法，通过推理任务相关的语义先验并结合位姿优化，摆脱了对有限人工标注任务特定抓取数据集的依赖。 |
| [^236] | [Wrong Votes, Right Gradients: Why Majority-Vote Rewards Track Verifiable RL](https://arxiv.org/abs/2610.20898) | 该论文将分析单位从提示层面转向批次级参数更新，证明多数投票等自监督奖励之所以能媲美可验证奖励强化学习，是因为当标签误差在不同提示之间不相干时，其对参数更新的误导作用会随批次规模增大而消失。 |
| [^237] | [Stochastic Gradient Descent: A Nonlinear Time Series Perspective](https://arxiv.org/abs/2610.20894) | 本文在迭代随机函数框架下，通过几何矩收缩（GMC）证明了恒定步长SGD在欧氏距离下的收敛性，克服了固定初始化带来的非平稳性问题，证明了平均SGD的淬火中心极限定理、不变原理和α-稳定收敛，并通过研究关于步长的导数为理解步长的影响提供了新视角。 |
| [^238] | [Flexible Doubly Robust Proximal Causal Inference for Censored Outcomes](https://arxiv.org/abs/2610.20893) | 该论文提出了邻近生存桥梁学习（PSB），一个针对删失结果的灵活双重稳健邻近因果推断框架，通过核估计器（PSB-K）和神经估计器（PSB-N）两种灵活桥梁学习器直接处理删失数据和生存曲线目标，突破了现有方法依赖限制性参数化桥梁模型的局限。 |
| [^239] | [Learning Warm Starts for the Simplex Method from Measured Solver Cost](https://arxiv.org/abs/2610.20892) | 该论文发现学习型热启动虽能消除单纯形法三分之二的迭代却只能节省三分之一的时间，其原因是被消除的迭代往往代价低廉且与最优基的接近程度无法反映实际计算成本，因此应以实测求解成本（如枢轴次数，可解释71%的对数求解时间方差）作为学习热启动的指导目标。 |
| [^240] | [Do Emergent Misalignment Models Share a Circuit?](https://arxiv.org/abs/2610.20887) | 研究发现在不同有害任务上独立微调得到的涌现失对齐模型共享相同的注意力头电路，将这些头的贡献替换为基础模型的贡献即可大幅降低失对齐行为且不损害连贯性。 |
| [^241] | [Pixels to Tokens: Token Space Efficient Active Learning for Low-Budget Semantic Segmentation](https://arxiv.org/abs/2610.20881) | 提出了首个基于ViT表示的像素级主动学习框架TEAL，通过在冻结ViT骨干的原生词元网格上直接执行两级MaxHerding多样性选择，实现了极低标注预算下高效的语义分割。 |
| [^242] | [CogDrift-4D: Hierarchical Multimodal Modeling of Temporal Behavioral Decay for MCI Detection](https://arxiv.org/abs/2610.20877) | 该论文提出了CogDrift-4D框架，将18个月内七次多模态非脚本化交互的行为数据构建为4D张量，并通过层次化Transformer的双层架构（会话内疲劳建模与会话间漂移建模）捕捉认知衰退的时序轨迹，从而实现轻度认知障碍的早期检测。 |
| [^243] | [Machine-learning-based neutrino-event generation constrained by experimental measurements](https://arxiv.org/abs/2610.20870) | 该论文提出了一种机器学习生成式中微子事例模型，可直接利用实验测得的微分截面（无需事例级实验样本）进行校准，且校准后无需重新调参即可在不同可观测量和束流配置间保持适用性。 |
| [^244] | [SHRED: Self-Play for Heterogeneous Road-Users without Expert Demonstrations](https://arxiv.org/abs/2610.20866) | SHRED提出了一个无需专家演示的异构自我博弈框架，从随机初始化出发联合训练车辆、行人和骑行者三类道路使用者，使逼真的交互行为在闭环自博弈中自然涌现，并在WOSAC基准上超越现有无演示自我博弈方法。 |
| [^245] | [Long-Context Drafting Does Not Require Long-Context Drafters](https://arxiv.org/abs/2610.20865) | 长上下文投机解码无需构建长上下文草稿模型——通过限制注意力缓存和重定基两种推理时机制，现成的短上下文草稿模型即可远超其训练长度高效运行，因为草稿模型借助目标模型的隐藏状态已经能感知遥远上下文。 |
| [^246] | [SNOAR: Self-training with Noise filtering, Observation adding, and Remixing for Speech Denoising Without Clean References](https://arxiv.org/abs/2610.20863) | 提出 SNOAR 自训练框架，在使用教师估计结果之前先进行校正——用差分噪声过滤（DNF）去除泄漏噪声、并回加部分含噪观测以恢复被过度抑制的语音和减弱伪影，从而在无需干净参考语音的情况下显著提升自训练去噪性能。 |
| [^247] | [Time-partitioning Gaussian Process Mixture Model](https://arxiv.org/abs/2610.20860) | 提出了一种贝叶斯非参数框架TPGPMM，通过自适应Pitman-Yor先验与特定状态高斯过程的结合，自动识别时间序列中的持续状态与变点，无需马尔可夫假设即可建模非平稳依赖，并能有效区分持续变化与瞬态冲击。 |
| [^248] | [Does Morphological Correctness Matter for Tokenization? A Scrambled Control Study in Tagalog](https://arxiv.org/abs/2610.20859) | 本文以他加禄语的中缀语态为切入点，构建了将词根连续化输出并附加位置性中缀标记的分词器 MorphAustro-Tok，并通过保留词素数量、可逆性与一致性的乱序词素对照实验检验“词素边界必须语言学正确”这一假设，结果显示其相比匹配BPE在机器翻译中最多提升3.40个chrF++点。 |
| [^249] | [Complete or Sparse: A Tale of Two Identifiabilities](https://arxiv.org/abs/2610.20851) | 本文证明在生成式与判别式两种因果方向下，表示维度充足时最优编码器是潜在变量的线性函数，而结合非负激活与能量效率约束可使每个神经元恰好编码一个潜在变量，为“祖母神经元”现象提供了精确的理论解释。 |
| [^250] | [Denoising Score Matching in Diffusion Models: The Effect of Rank Constraints on Generalization](https://arxiv.org/abs/2610.20850) | 本文研究了扩散模型分数网络中秩约束对泛化性能的影响，通过随机特征模型的理论分析证明：增大秩能降低训练误差，但其对测试误差的影响取决于样本量与数据维度的比例关系，并借助截断奇异值分解给出了渐近精确的极限误差刻画。 |
| [^251] | [Analog RF Computing for Integrated Data and AI Flows Over Wireless Networks](https://arxiv.org/abs/2610.20841) | 提出模拟射频计算方案，让基站组播编码神经网络权重的波形，使资源受限的边缘设备以极低功耗和内存完成AI推理，并研究了MU-MIMO下行链路中组播AI流与其他数据流共享频谱、功率和空间资源的问题。 |
| [^252] | [A Phenomenology of Neural Geometry in Vision-Language-Action Models](https://arxiv.org/abs/2610.20840) | 该论文首次系统刻画了视觉-语言-动作模型内部任务相关概念的真实几何形态，发现其表征以多维、弯曲的形式存在，并由物理约束塑造（如关节限制使手腕姿态呈开放弧线、目标距离以“接近粗标尺”与“抓取细标尺”两个正交子空间编码），从而超越了仅证明变量“线性可读”的先前研究。 |
| [^253] | [Confidence-Calibrated Inference Expansion for Evaluator-Guided Test-Time Reasoning](https://arxiv.org/abs/2610.20836) | 提出一个计算感知的三动作控制器，通过学习并校准动作特定的置信度估计来决定测试时推理中何时扩展计算，在提升效用的同时将在线计算量降低约 30% 和 16%。 |
| [^254] | [Amortizing Sequential Experimental Design in Reasoning LLMs](https://arxiv.org/abs/2610.20832) | 该论文通过以信息增益奖励显式训练LLM的推理过程，将序贯贝叶斯实验设计摊销到LLM中，使其在实验设计任务中能够自适应地收集信息，并在二十个问题任务上超越了训练时和推理时的BED基线方法。 |
| [^255] | [CoTranslator: Reverse to Forward Chain-of-Thought Translator for Reasoning Distillation](https://arxiv.org/abs/2610.20828) | CoTranslator通过将反向思维链回答“翻译”为经过验证的正向回答分布，兼得低成本与高质量，实现大规模推理蒸馏数据生成，在六个数学基准上显著提升学生模型平均准确率最高5个百分点。 |
| [^256] | [Linear Independence of Polynomial Compositions with Bounded Input Degree](https://arxiv.org/abs/2610.20825) | 本文通过有理曲线形变与Riemann-Roch定理论证解决了弱复合无关性猜想，进而证明了采用足够大次数多项式激活函数的神经网络的可识别性。 |
| [^257] | [Feasibility Ceilings Before Learning: Sizing and Timing Bounds for Drone-Mesh Interception](https://arxiv.org/abs/2610.20823) | 本文推导了无人机拦截网格在策略选定之前的性能闭式界限，并证明当智能体间距至少为2r时，平面与体积覆盖率分别存在90.69%与74.05%的固有可行性上限。 |
| [^258] | [Gibbs-Gap Policy Optimization for Off-Policy RL with Verifiable Rewards](https://arxiv.org/abs/2610.20814) | 提出了Gibbs间隙策略优化（GGPO）——一种无需学习策略到行为策略重要性权重的离策略强化学习方法，它通过最小化同一提示下各响应经KL调整后的奖励的log-mean-exp对比进行训练，并从理论上保证学习策略的单响应期望奖励不低于行为策略的best-of-K奖励。 |
| [^259] | [Optimal Explicit and Implicit Regularization Strengths in High-Dimensional Continual Linear Regression](https://arxiv.org/abs/2610.20804) | 本文推导出高维持续线性回归中每任务固定步数梯度下降的泛化损失闭式表达式，证明最优固定学习率应随任务数按 $\ln T/T$ 衰减，此时泛化性能渐近等价于最优调节的 L2 正则化，并将结果推广至对角广义岭回归并给出最优正则化的充要条件。 |
| [^260] | [Privacy-Preserving Text Inference for Health-Misinformation Classification and Semantic Retrieval](https://arxiv.org/abs/2610.20796) | 本文提出了一种基于CKKS同态加密的隐私保护文本推理框架，将BioBERT、MPNet与TF-IDF等特征融合用于健康错误信息分类（准确率达0.971），并结合语义指纹实现密文域上的语义检索。 |
| [^261] | [EEG-Based Prediction of Music Enjoyment Using Deep Learning with Explainable Temporal Analysis](https://arxiv.org/abs/2610.20795) | 该论文将音乐享受预测从传统的分类任务转化为回归问题，在公开EEG数据集上比较了六种深度学习模型，并结合可解释性技术进行时间维度的神经反应分析。 |
| [^262] | [Sliced Wasserstein Sketches with Error Certificates](https://arxiv.org/abs/2610.20788) | 该论文提出一种基于切片 Wasserstein 距离的紧凑草图表示方法，通过将平方距离误差精确分解为残差与划分失配两项，为任意质心草图提供可计算的误差证书，并证明最优草图在温和矩条件下可达到 O(M^{-1}) 的误差率。 |
| [^263] | [Verdict: Encode Once, Decide Many](https://arxiv.org/abs/2610.20778) | Verdict-MQ 提出一种“一次编码、多问题并行回答”的开源编码器，通过结构化注意力掩码实现状态编码与问题解耦、问题互不干扰及选项顺序无关，在类型化决策基准上以更小参数量在所有随机种子上显著超越专用模型。 |
| [^264] | [Selection-Safe Conformal Early Exit: Distribution-Free Coverage under Adaptive Depth](https://arxiv.org/abs/2610.20773) | 该论文提出一种选择安全的保形早退方法，通过前缀最大值构造的终端路径分数和单一保形分位数校准，在任意数据依赖的提前退出策略下均能保证无分布覆盖率，无需 alpha/L 修正，且实验表明其覆盖率显著优于朴素的逐层选择方法。 |
| [^265] | [When Reality Outpaces Cognition: A New Scaling Principle for Reliable Intelligence](https://arxiv.org/abs/2610.20764) | 本文提出了一种新的缩放原则——通过“有效连贯认知速度”量化智能系统在现实变化速度超出训练基线最高50倍时仍能可靠感知、推理和行动的能力极限。 |
| [^266] | [Purify Before You Align: Improving Representation Alignment for Diffusion Models](https://arxiv.org/abs/2610.20761) | 提出局部外观净化（LAP）方法，通过在对齐前去除目标表征中局部可预测的内容、仅对齐残差信息，显著提升扩散模型的生成质量，且无需改动训练或推理流程。 |
| [^267] | [Optimal Hydra: Model Heterogeneous Federated Learning for LLMs using Data-Informed Hydra-Nets](https://arxiv.org/abs/2610.20753) | 该论文提出Optimal Hydra方法，通过根据客户端本地数据分布定制异构的Hydra-Nets子网络架构，解决了资源受限客户端上大语言模型联邦微调的难题。 |
| [^268] | [Interpreting Language Model Parameters](https://arxiv.org/abs/2610.20748) | 提出了对抗性参数分解（VPD）方法，将语言模型参数分解为各自仅实现模型算法一小部分的简单子组件，从而帮助理解神经网络参数如何执行计算。 |
| [^269] | [Ergodic topology sampling toward the continuum with Stochastic Normalizing Flows](https://arxiv.org/abs/2610.20747) | 本研究通过结合开放边界条件与随机归一化流的计算策略，提供了数值证据表明在细至 0.03 fm 的格距下可有效缓解 SU(3) 杨-米尔斯理论中的拓扑冻结问题。 |
| [^270] | [Vertical Sliced Wasserstein Distances for Meta-Measures](https://arxiv.org/abs/2610.20743) | 本文提出垂直切片 Wasserstein（VSW）距离，一种面向元测度空间 $\mathcal{P}(\mathcal{P}(\Omega))$ 的可扩展度量，通过连续测试函数投影内层测度实现高效计算，并证明了其度量性质与维度无关的统计收敛保证。 |
| [^271] | [Quantile-Guided Diffusion for Probabilistic Time-Series Forecasting](https://arxiv.org/abs/2610.20739) | 提出Control-TS，在冻结的预训练扩散主干上学习分位数残差控制通路，将预测分位数作为分布控制来引导扩散预测，在八个基准上持续提升CRPS并达到与最先进模型相当的精度。 |
| [^272] | [On-Policy Residual Learning for Diffusion Language Models](https://arxiv.org/abs/2610.20687) | 该论文提出ORL框架，发现残差学习本质上是在策略的，通过熵收缩目标实时学习模型自身诱导的残差轨迹，并结合异步残差捕获、块扩散注意力等算法-系统协同设计，实现扩散语言模型速度与质量的高效平衡。 |
| [^273] | [RoLaQ: Joint Rotation and Range Calibration for 2-Bit KV Caches in Multi-Head Latent Attention](https://arxiv.org/abs/2610.20686) | 提出RoLaQ离线校准方法，通过在多头潜在注意力模型中联合学习共享旋转和相互耦合的量化范围，实现2比特KV缓存量化，在多个模型和基准测试上取得了2比特方法中最高的平均准确率。 |
| [^274] | [Learning Topology-Aware Representations for Mechanics-Relevant TPMS Metamaterials](https://arxiv.org/abs/2610.20684) | 提出将持续同调融入DMAE骨干网络的拓扑感知表示学习框架，通过持续性引导的掩码策略和辅助持续同调解码器显式编码TPMS超材料的全局结构连通性，从而更好地支持力学分析与结构设计。 |
| [^275] | [Causal Syntax Circuits in CodeLMs: Budget and Probe Decide What Looks Localised](https://arxiv.org/abs/2610.20683) | 该论文通过均值消融、激活修补、贪心电路提取与激活引导的统一流程，系统研究了代码语言模型默认偏好Python的内部机制，发现该偏好随模型规模增长并在指令微调后仍存在，但其局域化表现高度依赖于组件预算与探针的选择。 |
| [^276] | [TurboMamba: Faster Inference, Faster Training, and Better Prediction with Lexical Memory](https://arxiv.org/abs/2610.20682) | TurboMamba通过结合全局Mamba-3模型、小型Transformer和词汇记忆等轻量级机制，在实现更快推理与训练的同时，取得了比Transformer、Mamba-3、Jamba和Zamba等基线模型更低的预测损失。 |
| [^277] | [Diminishing Returns of Data in Sparse Autoencoders: Functional Behavior does not Guarantee Feature Reproducibility](https://arxiv.org/abs/2610.20680) | 该论文发现稀疏自编码器的功能表现（如重构）在远小于常规规模的数据量下即可接近全量训练水平，但特征的一对一可复现性会随数据量增加持续提升，表明功能行为良好并不能保证特征可复现，数据规模和数据选择方式对特征复现至关重要。 |
| [^278] | [AlphaDiana: A Measurement System for Agents in Verifiable Reasoning Tasks](https://arxiv.org/abs/2610.20675) | AlphaDiana是一个面向可验证推理任务的模块化智能体测量系统，能够在匹配条件下对完整智能体进行宏观分析并对工具、技能、记忆等组件进行微观干预，研究发现脚手架并不能持续提升模型推理准确性。 |
| [^279] | [On the Stability and Convergence of Flow Map Training](https://arxiv.org/abs/2610.20674) | 本文通过分析MeanFlow与拉格朗日损失诱导的函数梯度流，首次从理论上解释了流映射训练中停止梯度算子和自适应加权等技巧的必要性，并证明了训练会收敛到真实流映射。 |
| [^280] | [Value-Informed Schedule Trimming for Accelerated Sequential Monte Carlo Guidance in Discrete Diffusion](https://arxiv.org/abs/2610.20669) | VISTA是一种无需训练的方法，通过将稀疏引导建模为对理想完全引导概率路径的近似，并利用预热阶段收集的价值统计量结合动态规划精确求解经验代理目标，从而在固定计算预算下自动选择最优的引导时间步，显著加速离散扩散模型中的序贯蒙特卡洛奖励引导。 |
| [^281] | [CascadeLedger: Counterfactual Auditing of Information and Resource Integrity in LLM Cascades](https://arxiv.org/abs/2610.20668) | CascadeLedger提出了一个面向预算受限LLM级联的反事实执行审计框架，通过聚合等价轨迹有效性和反事实前缀无干扰性等机制，审计路由决策是否仅使用了合法可观测信息，并能定位最早违规点与最小隐藏见证。 |
| [^282] | [Evaluating Model Capabilities After Benchmark Saturation](https://arxiv.org/abs/2610.20667) | 该论文提出利用模型在饱和基准测试中的完整解题过程记录（而非仅依赖分数），即使基准分数饱和后仍能有效评估和预测模型能力。 |
| [^283] | [Graph Additive Models disentangling feature, topology, and interactions](https://arxiv.org/abs/2610.20662) | 提出AMIGA模型，通过挖掘的图模体表示拓扑结构，并将节点贡献分解为特征、结构和交互三个加性通道，从而解决了图神经加性网络中虚假相关性传递和效应纠缠的问题。 |
| [^284] | [Latent Annealed Posterior Sampling with a Variance-Preserving Bridge for Sparsely Observed Random Fields](https://arxiv.org/abs/2610.20659) | 提出LAPS方法，利用流匹配学习潜在先验并在潜在空间中进行退火后验采样，结合方差保持桥控制协方差膨胀，从而高效解决稀疏观测随机场后验推断中空间不确定性刻画与计算成本增长的难题。 |
| [^285] | [A polylogarithmic bound for list replicability](https://arxiv.org/abs/2610.20657) | 本文为任意二元概念类的列表可复现性数 O(log²M log N) 这一首个多对数上界提供了一个简短且自包含的新证明。 |
| [^286] | [Global-in-time Spectral Duhamel-Picard Method for Post-Exposure Bake Simulation and Physics-Informed Neural Operators for EUV Lithography](https://arxiv.org/abs/2610.20655) | 本文提出一种全局时间谱Duhamel-Picard方法，通过将酸-猝灭剂反应扩散方程转化为Volterra积分方程并用多项式展开与Anderson加速Picard迭代求解，实现了约一分钟内模拟1200×1200×100体素的大规模三维曝光后烘焙过程，并将其融入物理信息神经算子用于EUV光刻模拟。 |
| [^287] | [AbridgeKV: Compressing Long Reasoning into the KV Cache of Learned Pause Tokens](https://arxiv.org/abs/2610.20641) | AbridgeKV通过训练模型将已生成的长推理文本块压缩到学习到的暂停令牌的KV缓存中并释放原始缓存，解决了长链式思维解码中KV缓存线性增长的内存瓶颈问题。 |
| [^288] | [Same Answer, New Box: How Much of RLVR's Out-of-Domain Gain Is Answer Format?](https://arxiv.org/abs/2610.20639) | 研究发现，RLVR所报告的域外多项选择基准提升中约82%只是答案格式提取带来的假象，而非模型真正回答能力的提升。 |
| [^289] | [Feature Elicitation for SAE Steering of Deceptive Behaviour in Small LLMs](https://arxiv.org/abs/2610.20638) | 该研究提出一种新颖的词袋特征引出流水线，通过引导与义务论伦理相关的SAE特征，首次成功将Gemma-3-27B-IT模型的合规差距（对齐伪装的关键指标）降低至不显著水平，并揭示了一种名为“欺骗性去锋”的新型欺骗行为。 |
| [^290] | [FlowBudget: Training-Free Sampling-Budget Estimation for Flow Matching Models](https://arxiv.org/abs/2610.20636) | 提出FlowBudget，一种免训练的采样预算估计方法，通过免模拟插值探针为流匹配模型确定足够的数值求解分辨率（NFE），从而避免昂贵且重复的FID-NFE扫描。 |
| [^291] | [Exploring the Design Space of Permutation Diffusion LLMs](https://arxiv.org/abs/2610.20635) | 该论文系统探索了三种用于排列生成的扩散大语言模型参数化设计（硬分配、软分配和直接排序分数预测），发现硬/软分配方法在单次函数评估下达到与掩码扩散基线相当的推荐重排性能，而直接分数预测无需匈牙利算法即可保证有效排列。 |
| [^292] | [A Large-Scale Dataset for Multi-Class Financial News Classification in Brazilian Portuguese](https://arxiv.org/abs/2610.20630) | 本文发布了约16万篇巴西葡萄牙语金融新闻的大规模语料库，并提供了人工与LLM两种标注的基准子集，实验发现基于语义相似度的弱监督多类别金融新闻分类性能高度依赖标签质量与类别分布，其中KNN方法表现最佳。 |
| [^293] | [BD-HazardVLM: Probing Vision-Language Models for Latent Defensive-Driving Hazards in Bangladesh Road Scenes](https://arxiv.org/abs/2610.20627) | 提出了BD-HazardVLM诊断基准，利用500张孟加拉国道路场景图像系统评估视觉语言模型对防御性驾驶危险的推理能力，发现模型虽能判断危险是否存在（接近多数类基线），但在危险类别、风险时机和防御动作等细粒度推理上表现严重不足。 |
| [^294] | [pcNN-SN: a physics-enhanced predictor-corrector neural network algorithm coupled with the discrete ordinates method for neutron transport equations](https://arxiv.org/abs/2610.20619) | 本文提出一种物理增强的预测-校正神经网络算法pcNN-SN，通过与离散纵标法耦合，利用高精度数据预测粗网格解的校正因子，在保持守恒性质的同时有效降低中子输运计算的成本。 |
| [^295] | [Iterative Neural Variational Solver of Committor Function in Under-damped Langevin Dynamics](https://arxiv.org/abs/2610.20617) | 本文提出了一种计算欠阻尼朗之万动力学中承诺函数的迭代神经变分求解器，克服了后向Kolmogorov方程缺乏变分结构这一长期挑战，从而能够定量研究有限阻尼系数对系统中稀有转变的影响。 |
| [^296] | [DeepUnfolding_Agent: A Classical_Quantum Adaptive Control API](https://arxiv.org/abs/2610.20615) | 本文提出 DeepUnfolding Agent，一个基于模式驱动的 FastAPI 平台，将深度展开的自适应控制作为外部软件服务从特定领域执行中解耦出来，从而提升自适应优化工作流在机器学习、联邦学习和量子系统中的复用性、可重复性与安全性。 |
| [^297] | [Thermodynamically Consistent Machine Learning for Fracture Prediction in Heterogeneous Materials](https://arxiv.org/abs/2610.20614) | 该论文提出了热力学一致性潜空间动力学辨识框架，利用强制满足总能量守恒和熵产生等热力学原理的参数化GENERIC形式信息神经网络，在低维潜空间中高效预测非均质材料的断裂模式与演化过程。 |
| [^298] | [Co-Evolving Acting and Forecasting for Language Agents](https://arxiv.org/abs/2610.20597) | 该论文提出在成功的同策略轨迹上引入多步动作序列作为监督目标，使行动生成与动作预测协同演化，从而充分利用中间步骤的监督信号，缓解语言智能体训练中长程 rollout 成本高、终端奖励稀疏的问题。 |
| [^299] | [Hindsight-Privileged RLVR: Single-Rollout Token Credit Without Group Sampling](https://arxiv.org/abs/2610.20497) | 提出HP-RLVR，一种无需组采样的单次轨迹RLVR方法，通过比较因果视角与包含未来结果和验证器反馈的事后特权视角之间的Jensen–Shannon散度为每个词元分配有界信用，从而在所有尝试结果相同时仍能提供有效学习信号。 |
| [^300] | [Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation](https://arxiv.org/abs/2610.20247) | 本文提出了一种感知重复的联合缩放定律，揭示多语言MoE模型中活跃参数、训练词元、专家数量与语言覆盖范围之间的关系，并发现32个专家相比稠密模型可将语言从12种扩展到200种时的损失惩罚降低83%。 |
| [^301] | [Retrieval Over Training: Similarity search-based Model Selection for Time Series Anomaly Detection](https://arxiv.org/abs/2610.20243) | 提出RAMSAD框架，通过相似性检索历史实例知识库而非训练模型的方式为新时间序列选择最优异常检测器，从而解决了现有模型选择方法在分布外（OOD）场景下精度大幅下降的问题。 |
| [^302] | [RM-OPD: On-Policy Distillation for Generative Reward Models](https://arxiv.org/abs/2610.20174) | 该论文提出RM-OPD两阶段框架，先通过单领域GRPO微调训练各领域专家教师模型，再通过路由同策略蒸馏将其整合为单一学生模型，从而解决Mix-RL训练生成式奖励模型时的奖励稀疏与跨领域干扰问题。 |
| [^303] | [In-Context Batch Black-box Optimization via Flow Matching](https://arxiv.org/abs/2610.20138) | 提出FMTR方法，利用流匹配离线学习历史条件化的连续提议分布，实现可变大小批量的高效采样，并在优化过程中自适应平衡探索与利用，改进批量黑盒优化。 |
| [^304] | [Exhaustive Canonical Skeletons Generation: A Formal Dataset for Decomposable Symbolic Regression](https://arxiv.org/abs/2610.20108) | 该论文提出基于确定性受限上下文无关文法穷举生成的单变量规范骨架形式化数据集，消除了符号回归数据集中的结构冗余，并支持在不假设线性可分性的前提下求解多变量符号回归问题。 |
| [^305] | [TIMERSI: Towards Self-Improved Agentic Deployment for Time-Series Foundation Models](https://arxiv.org/abs/2610.19725) | TIMERSI是一个围绕冻结时间序列基础模型的自改进智能体部署系统，由编码智能体根据任务描述迭代生成并修正预测流水线，并在部署时通过自适应选择器挑选最优流水线，在198个任务上超越了所有对比方法。 |
| [^306] | [ReTerm: Recovery Training for Self-Improving Terminal Agents](https://arxiv.org/abs/2610.18484) | 提出ReTerm终端智能体，通过恢复示范数据集ReTerm-10K进行SFT冷启动，并利用从自身失败状态分叉出多次恢复尝试的在线强化学习算法ReTerm-GRPO，让智能体从自身恢复经验中自我改进，在Terminal-Bench 2.0上取得34.8%的Pass@1成绩。 |
| [^307] | [When Do Learned Blocks Preserve Causal Meaning? Block Identifiability and Constructive Causal Abstractions](https://arxiv.org/abs/2610.18155) | 该论文证明了块对角可逆变换加上相容坐标置换是保留学习表示中块因果意义的最大变换类，从而将块可辨识性与构造性因果抽象联系起来，使高层干预能够在独立学习到的表示之间迁移。 |
| [^308] | [RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions](https://arxiv.org/abs/2610.17867) | RT-J是一个8500万参数的关系型基础模型，通过在639个网络爬取数据库和LLM生成的预测任务上利用长上下文预训练，仅需数百个上下文样本即可实现准确预测，所需样本量比现有方法减少10倍。 |
| [^309] | [Agentic-MatriBO: Capability-Compiled Agentic Bayesian Optimization for Materials and Chemical Design](https://arxiv.org/abs/2610.17832) | 提出了能力编译型智能体贝叶斯优化框架Agentic-MatriBO，将BO组件表示为可组合的能力，由大语言模型策略提出优化配方，经编译器检查和确定性执行层验证后用于材料与化学设计。 |
| [^310] | [Follow the Perturbed Leader for Linear Bandits: $\sqrt{T}$ Regret over General Convex Sets](https://arxiv.org/abs/2610.17818) | 本文提出了首个基于FTPL的对抗性线性赌博机算法，通过一个适应估计奖励的扰动分布，在一般凸动作集上实现了 $\sqrt{T}$ 遗憾，改进了此前 $T^{2/3}$ 的遗憾界，同时比连续指数权重方法具有更好的计算效率和更简单的实现。 |
| [^311] | [SPARCL: Spatially Adaptive Regional Coverage for Active Learning in Chest X-ray Analysis](https://arxiv.org/abs/2610.17808) | 提出SPARCL方法，利用凹覆盖函数动态平衡预测不确定性与空间覆盖率，解决胸部X光病灶分割主动学习中样本空间分布不均导致性能不一致的问题，并凭借单调次模性保证近最优的批量选择。 |
| [^312] | [Morphology-aware deep learning predicts colorectal cancer survival from whole-slide images](https://arxiv.org/abs/2610.17603) | 该研究提出MorphContrast——一种形态学感知的对比多实例学习框架，利用组织标签作为结构监督，从H&E全切片图像有效预测结直肠癌患者的五年生存期。 |
| [^313] | [A Bethe-Ansatz Source of Early Token Correlations: A marginal-preserving construction for parallel proposals](https://arxiv.org/abs/2610.16781) | 该论文从周期XXZ链的坐标贝特拟设构造出保持token边际分布的并行提议联合分布律，并证明即使一、二token边际完全匹配，由 δ 量化的高阶相关性差异依然存在，从而揭示了早期token相关性的可积系统来源。 |
| [^314] | [PMB-NN: Physiological-Model-Based Neural Network for Glucose Forecasting in Type 1 Diabetes](https://arxiv.org/abs/2610.15456) | 该论文提出一种基于生理模型的神经网络框架（PMB-NN），利用个性化四参数生理模型约束时间卷积网络的训练，实现了1型糖尿病患者30至120分钟时间范围内的准确血糖预测。 |
| [^315] | [World Modeling through Spectral Alignment](https://arxiv.org/abs/2610.15429) | SpecWM是一个通过目标相似度核显式指定观测间期望关系并训练编码器在嵌入空间中复现这些关系的JEPA世界建模框架，核的选择决定了哪些物理量可从表示中恢复，其中时间核仅需观测顺序即可实现全自监督学习。 |
| [^316] | [CIViC: Evidence Gates for Audio-Conditioned Visual Prediction on Proxy Tasks](https://arxiv.org/abs/2610.15327) | 本文提出CIViC四门控验证协议，通过将解释与干预、留出任务比较或端点控制相绑定，来检验音频条件视觉预测任务是否真正使用了预期的音频信息，并在面部光栅测试、合成音频混合及SpeakerVid代理任务上验证了该方法。 |
| [^317] | [AlphaRLM: Aligning Language Models for Literature-Grounded Symbolic Factor Discovery](https://arxiv.org/abs/2610.14959) | AlphaRLM是一个强化学习框架，通过文献引导算子组装（LEGO）提供语义先验，并借助可执行有效性与分组收益分离等奖励将生成策略与市场反馈对齐，从而训练大语言模型直接从金融文献中构造可执行的符号化Alpha因子。 |
| [^318] | [TLA-Verus: Generating and Solving Temporal Proof Obligations with LLM Agents](https://arxiv.org/abs/2610.14923) | 该论文提出TLA-Verus框架，通过确定性转译器和LLM智能体将TLA+规范翻译为Rust验证器Verus，能够大规模生成证明义务并产出完全可机器检查的安全性与活性证明，弥补了TLA+在时序属性验证上的不足。 |
| [^319] | [Self-improving data synthesis can scale embedding models across domains and tasks](https://arxiv.org/abs/2610.14730) | 提出了SYNTRA自改进检索数据合成方法，仅需无标注语料库和少量查询-文档对，通过自改进循环迭代优化合成指令，在多数任务上超越人工校准，并使嵌入模型随数据规模扩展在15个领域持续提升性能。 |
| [^320] | [PotentialEP: Backprop-Free Training of CIFAR-Scale Generative Models via Equilibrium Propagation](https://arxiv.org/abs/2610.14409) | 该论文提出PotentialEP方法，证明加入单个线性注入项的预测编码网络在弛豫过程中其可见层可天然携带能量匹配所需的梯度，从而实现无需反向传播的CIFAR规模生成模型训练，适配模拟神经形态硬件。 |
| [^321] | [EV-JEPA: Elastic V-JEPA for Any Device and Cross-Device Inference](https://arxiv.org/abs/2610.14324) | 提出 EV-JEPA，通过转换感知三元组训练（T3）训练出宽度可配置且可原生跨异构设备分区的单一基础模型，一次训练即可部署于从智能眼镜到服务器的任意设备，并同时支持单设备弹性推理与跨设备协同推理。 |
| [^322] | [Predict Before You Modify: Learning Improvement Dynamics for Self-Evolving Agents](https://arxiv.org/abs/2610.13991) | 提出可编辑的改进动力学模型（IDM），在评估前预测修改的任务级结果、评估后利用反馈持续积累预测知识，实现“先预测再修改”的预测式自进化框架，并在四个任务上全面超越HyperAgents。 |
| [^323] | [Spatial Context or Memory? Sizing the Inputs of Neural Surrogates for Partially Observed Fields](https://arxiv.org/abs/2610.12723) | 本文以贝叶斯最优误差为基准，首次精确量化了部分观测场神经代理模型中空间传感器窗口与历史快照长度之间的权衡，证明对于具有局部动力学的稀疏含噪观测场，扩大空间窗口比增加记忆更有效，且仅读取空间窗口的模型即可接近最优预测性能。 |
| [^324] | [TTS-Q: Perception-Guided Post-Training Quantization of Text-to-Speech Diffusion Transformers](https://arxiv.org/abs/2610.12474) | 本文首次对十亿参数规模的TTS扩散变换器开展4比特权重与激活（W4A4）训练后量化研究，揭示了量化主要损害说话人音色而非可懂度这一现象及其来源，并提出感知引导的量化修复方法。 |
| [^325] | [Caught in the Act: Probes Effectively Detect Sabotage and Catch Unverbalized Deception](https://arxiv.org/abs/2610.12445) | 该研究通过构建迄今最大的欺骗数据集并设计可跨层、跨标记聚合信息的新型探针架构，使白盒探针在 SHADE-Arena 上以 98.8% 的 AUC 超越前沿文本监控基线，甚至能检测出仅凭上下文无法察觉的“内省式欺骗”。 |
| [^326] | [AdaptLSTM: Efficient Adaptive Online Learning for Cloud Workload Forecasting under Distribution Drift](https://arxiv.org/abs/2610.12265) | AdaptLSTM 通过验证校准阈值检测分布漂移并进行选择性针对性更新，以约 20% 的计算成本实现接近朴素在线学习的预测精度，大幅提升云工作负载预测的效率。 |
| [^327] | [Identifiability of a dissipative knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational data](https://arxiv.org/abs/2610.09889) | 该论文将人类学习建模为参数具有机制含义的耗散常微分方程组，证明了在设计激励条件下模型参数可从数据中被精确恢复（双概念情形有闭式解），而在仅依赖观测数据时可辨识性会退化，并提出了数值精确等价且速度大幅提升的半隐式L-稳定批量求解器。 |
| [^328] | [LASER: Latent Space Adjoint Matching for Support-Constrained Entropy-Regularized Offline RL](https://arxiv.org/abs/2610.08989) | LASER通过潜在空间伴随匹配实现熵正则化的潜在空间离线强化学习，既防止了策略坍缩成单一脆弱模式，又避免了时间反向传播，在40个不同数据质量的OGBench任务上取得了优异表现。 |
| [^329] | [CNet: A Complex-Valued Deep Learning Framework with Wirtinger Autodifferentiation and FFT--Hadamard Convolution](https://arxiv.org/abs/2610.08592) | CNet 是一个基于 Wirtinger 自动微分的 C++/CUDA 复值深度学习框架，它将 FFT-阿达马卷积恒等式转化为可学习的复值卷积网络，并采用玻恩规则进行物理原生式分类。 |
| [^330] | [Minimal Witness Reinforcement Learning](https://arxiv.org/abs/2610.07226) | 本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。 |
| [^331] | [Calibrated Answers About Randomized Trials From a 4-Billion-Parameter Open Model: A Registered Test and a License-Clean Release](https://arxiv.org/abs/2610.07019) | 该论文发布了一个仅使用许可证允许复用的文章微调的 40 亿参数开放模型 Fiorillo v0.5，它能够以良好校准的概率回答随机试验中干预措施对结局影响的问题，并通过预注册的四项标准验证后正式发布。 |
| [^332] | [E$^2$-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation](https://arxiv.org/abs/2610.05048) | 论文发现在线策略自蒸馏存在学生熵超过教师并持续高企的“熵过冲”失效模式，其根源是教师监督过度依赖答案特定线索以及前向KL散度不断扩散学生分布，并据此提出E$^2$-OPSD同时修复这两个成因。 |
| [^333] | [D-DOIT: Training-free Adaptation of Discrete Diffusion via Doob's h-Transform](https://arxiv.org/abs/2610.04938) | D-DOIT提出了一种免训练的离散扩散模型适配方法，通过推导离散Doob h-变换、仅利用奖励值（无需奖励梯度和昂贵的未来展开）重新加权反向转移概率，从而在推理时高效地从奖励倾斜的目标分布中采样。 |
| [^334] | [ULTRADISCOVERY: Abductive Exploration in an Interconnected, Epistemically Open Universe](https://arxiv.org/abs/2610.03092) | 该论文提出ULTRADISCOVERY交互式基准，通过2×2设计独立控制表征开放性与证据分布性，评估智能体在认识论开放且结构互联的世界中进行溯因科学探索的能力，发现现有十一个模型均无法通过引入新实体或重写变量来完成理论替换。 |
| [^335] | [StabilityArc: Decoding Protein Sequence Embeddings into Generalizable Stability Landscapes](https://arxiv.org/abs/2610.00742) | StabilityArc 通过在多种蛋白质数据上训练的共享解码器，将冻结的蛋白质序列嵌入解码为可泛化的稳定性景观，实现了跨蛋白质的突变效应零样本迁移预测，其性能超越最强零样本基线，并可作为先验显著提升其他模型的表现。 |
| [^336] | [CompMat-Bench: Benchmarking AI Agents for Computational Materials Science](https://arxiv.org/abs/2610.00636) | 提出了CompMat-Bench基准，通过预先复现昂贵模拟的输入与结果作为固定规则的评分依据，实现对计算材料科学AI智能体的低成本高效评估，涵盖94个源自已发表研究的任务以及四种评估条件。 |
| [^337] | [Interpretable but Fragile? Robustness of Concept Bottlenecks under Geometric-Semantic Perturbations](https://arxiv.org/abs/2609.38625) | 该论文提出了一个基于生成器的统一评估框架，首次将潜空间中的连续几何扰动与概念空间中的离散语义干预区分开来，从经验性与可认证两个层面系统比较概念瓶颈模型与标准分类器的鲁棒性，从而调和了此前相互矛盾的研究结论。 |
| [^338] | [KV-streams for Efficient Compaction in Agentic Reinforcement Learning](https://arxiv.org/abs/2609.35750) | KV-streams提出一种即插即用的KV缓存流式传输策略，通过在压缩时保留并向前流式传输KV缓存而非清空后反复预填充，使智能体强化学习中的上下文压缩实现2.6到5倍训练加速且不损害性能。 |
| [^339] | [THEIA: A Multimodal Dataset and Benchmark for Vision-Language Analysis of Layout](https://arxiv.org/abs/2609.35035) | 本文提出THEIA多模态数据集与基准测试，利用微调的视觉-语言模型分析模拟电路的GDSII版图文件，使设计者能够通过对话方式直观地查询和交互物理版图。 |
| [^340] | [FAME: An FPGA-Based Platform for Approximate Multipliers Evaluation with Pattern-Guided DNN Retraining](https://arxiv.org/abs/2609.17730) | 本文提出FAME，一个基于FPGA的近似乘法器评估平台，通过在硬件中直接实现近似乘法器并结合模式引导的DNN重训练，大幅加速了DNN推理中近似乘法器的精度评估与误差缓解流程。 |
| [^341] | [PLSP (Pre-hoc Liminal Space Profiling): OOD Prediction over Detection -- An Anticipatory Approach for Machine Learning Model Reliability](https://arxiv.org/abs/2609.12225) | 本文提出事前预测框架PLSP，将OOD处理范式从检测转向预测，并引入可信度评分（CREDS）、可信度曲线和可信度热力图，实现对机器学习模型应对分布外数据可靠性的事前评估。 |
| [^342] | [LightMedSeg-ISLES: Stroke Lesion Segmentation with 81x Fewer Parameters than nnU-Net](https://arxiv.org/abs/2609.09634) | 提出了仅126万参数的轻量级脑卒中病灶分割模型LightMedSeg-ISLES，以少81.4倍的参数保留nnU-Net 97.5%的Dice分数并提升病灶级F1，同时计算量减少4.7倍，是大型模型的强力单检查点替代方案。 |
| [^343] | [Training-Free Task Vectors for LLM Behavioral Control](https://arxiv.org/abs/2609.09054) | 提出了一种无需微调的TFTVs方法，仅利用前向传播统计量将激活引导向量映射为秩一权重空间编辑，从而以低成本实现对大语言模型行为的放大、抑制与组合控制。 |
| [^344] | [Language Models Can Control Their Own Attention](https://arxiv.org/abs/2609.02737) | 该论文提出“声明式注意力”协议，让语言模型在思维链中自主声明需要关注的上下文区域，推理引擎据此像解析工具调用一样跳过大部分KV缓存读取，从而以内在方式避免了外部评分方法每步O(N)的开销。 |
| [^345] | [Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search](https://arxiv.org/abs/2609.01431) | 本文提出幂律熵搜索（PLES），一种基于多保真度贝叶斯优化的计算成本感知采集函数，通过自适应选择能最大程度降低缩放定律估计整体不确定性的实验配置（而非优化单一目标函数），高效估计大语言模型最优超参数随规模变化的缩放定律，从而大幅节省计算资源。 |
| [^346] | [The Safety Relay in Roleplay Jailbreaks: A Component-Resolved Causal Analysis of Harm Recognition and Refusal](https://arxiv.org/abs/2608.30585) | 该研究运用机制可解释性方法，通过追踪隐状态、反事实干预和几何分解，揭示了角色扮演包装中的人设、场景和任务组件如何在组件层面逆转大语言模型的有害识别与拒绝机制。 |
| [^347] | [No Equivariant Architecture Covers All Equivariant Attention](https://arxiv.org/abs/2608.30417) | 本文完整刻画了等变多头自注意力的结构，并证明其等变性轨迹分裂为极多个不可约分支，因此任何固定的等变注意力架构至多覆盖其中一个分支，必然损失对等变映射的表达能力。 |
| [^348] | [Tail-Replay: Escaping the Curse of Linear Attention in Prefix Caching for Hybrid LLMs](https://arxiv.org/abs/2608.30310) | 提出Tail-Replay前缀缓存机制，利用线性注意力可视为输入前缀结构化有损压缩这一洞察，使混合大语言模型摆脱循环状态检查点的限制，实现任意位置的token级前缀重用。 |
| [^349] | [INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction](https://arxiv.org/abs/2608.29901) | INTERVenE提出了一族以区间式知识时间抽象（KBTA）生成的命名临床概念作为token输入的Transformer架构，通过自回归解码器和双向编码器两种变体，在保留时间临床意义的同时实现ICU中可解释的短期医疗事件与风险预测。 |
| [^350] | [ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing](https://arxiv.org/abs/2608.26083) | ICON分解通过多变量分析，在控制其他概念和结果后精确量化每个概念对模型表示的独特贡献，从而有效识别捷径学习并提高解释的准确性。 |
| [^351] | [DataKernelBench: Can LLMs Optimize Database Queries on GPUs?](https://arxiv.org/abs/2608.25061) | 本文提出了DataKernelBench基准测试，证明大语言模型能通过执行引导修复在GPU上优化数据库查询，最强配置实现2.11倍加速，并发现内核融合、完整查询专业化和工作负载上下文是关键因素。 |
| [^352] | [Calibration-Preserving Pruning: Compression as a Reliability Contract](https://arxiv.org/abs/2608.23744) | 本文提出校准保持剪枝方法，通过增强剪枝分数保持分数几何结构，在保证共形预测覆盖的同时减小预测集大小，实现更高效的模型压缩。 |
| [^353] | [Blockwise Stabilized Adaptive Cubic Regularization with Subsolvers via Recurrence](https://arxiv.org/abs/2608.22129) | 本文提出了一种分块稳定的自适应三次正则化方法，通过Krylov子空间无矩阵求解精确三次步，在保持最优收敛速率的同时支持任意大规模张量，并提供了完整的理论保证。 |
| [^354] | [PureTD: Reinforcement Learning for Backgammon Money Games with No Evaluation-time Search](https://arxiv.org/abs/2608.15146) | 该论文证明纯自对弈强化学习无需评估时搜索即可训练出接近最优的后台棋模型，在金钱游戏中速度快且强于带单步搜索的开源引擎。 |
| [^355] | [Optimal Watermark Localization in Mixed-Source Large Language Model Texts](https://arxiv.org/abs/2608.14906) | 本文提出了混合来源LLM文本中水印定位的渐近最优框架，明确了全局检测、发现和分类的相变边界，并证明发现任务难度高于分类。 |
| [^356] | [DYSANOS Generative Dynamic Smooth Arbitrage-free Non-parametric Option Surfaces](https://arxiv.org/abs/2608.12587) | 本文提出首个生成式无套利期权曲面模型DYSANOS，能够生成未来多年每日期权价格路径，并验证其优于传统隐含波动率PCA模型。 |
| [^357] | [Towards Reliable Context Compression for Long-Horizon Agents: An Empirical Study of Execution Instability](https://arxiv.org/abs/2608.06503) | 本文揭示循环上下文压缩会削弱智能体对近期交互的响应并导致执行不稳定，据此提出验证器引导的TRACE框架，通过边界局部评估压缩事件并利用摘要偏好优化自然语言压缩提示词，在任务性能、多运行可靠性和上下文-执行效率上超越现有基线。 |
| [^358] | [Multi-Modal, Multi-Temporal Geospatial Foundation Models Improve Global Above-Ground Biomass Estimation](https://arxiv.org/abs/2608.04792) | 本文对13个地理空间基础模型在全球地上生物量估计任务上进行了系统性基准测试，发现多模态、多时相的地理空间基础模型能够超越全监督基线方法，从而改进全球地上生物量的估计。 |
| [^359] | [ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities?](https://arxiv.org/abs/2608.03874) | 该论文提出动态评估框架ContinualSkillBench，用于衡量LLM智能体的上下文内持续技能学习能力，发现顺序执行通常提升性能但收益因模型和领域而异，且显式技能库仅在需要可复用流程或精确输出的任务中带来选择性收益。 |
| [^360] | [Optimal Unambiguous DNFs and Alon-Saks-Seymour](https://arxiv.org/abs/2608.02533) | 本文构造了宽度为O(n)但0-证书复杂度为Ω(n²)的无歧义DNF，并通过提升定理实现了对Alon-Saks-Seymour猜想的最优反驳以及团与独立集问题的最优通信下界。 |
| [^361] | [Flow Matching with Missing Data](https://arxiv.org/abs/2607.28698) | 该论文提出缺失数据流匹配方法，将训练样本的缺失坐标视为潜在变量并证明其oracle修正在任意缺失机制下都能精确恢复完整数据的流匹配目标，同时通过有限样本分析揭示了缺失机制只是转移而非增加估计量的方差。 |
| [^362] | [Circuit Claims Depend on What Is Extracted and How It Is Compared](https://arxiv.org/abs/2607.18921) | 论文通过在 Lean 战术预测基准上的系统实验表明，可解释性研究中“电路”的结论高度依赖于提取对象的定义和比较方法——不同提取方式得到的电路重叠度很低，说明行为保持并不能唯一确定一个电路。 |
| [^363] | [Amplitude-Only FFN Intervention for Tool-Structured LLM Inference: Method, Gated Evaluation Protocol, and Cross-Model Empirical Results](https://arxiv.org/abs/2607.11183) | 该论文提出幅度门控（AG）这一无需重训练的推理时干预方法，通过掩码FFN激活坐标，在多个Qwen模型的函数调用和ToolBench端点上实现了1.6至2.8个百分点的稳定性能提升。 |
| [^364] | [An End-to-End Hybrid Quantum--Classical Sampling Workflow for Discrete Markov Random Fields: A Reproducible Case Study](https://arxiv.org/abs/2607.09893) | 该论文构建了一个可复现的端到端量子-经典混合采样工作流，实证表明现代经典MCMC采样器能大幅缩小与量子采样的差距，且在计入$O(2^n)$预处理成本后，量子方法在实际运行时间上并无优势。 |
| [^365] | [Nonconvex Composite Functional Constraints via First-Order Augmented Lagrangian Methods under Local Regularity](https://arxiv.org/abs/2607.08954) | 该论文在仅需可行集附近局部乘子正则性的弱条件下，无需假设乘子有界，通过带对偶正则化的光滑近端线性增广拉格朗日原始-对偶方法，为非凸复合约束优化问题建立了非渐近KKT收敛保证，达到 $O(K^{-1/3})$ 的KKT误差率。 |
| [^366] | [Right Answer, Wrong Tokens: Tail-Aware Credit Calibration for RLVR](https://arxiv.org/abs/2607.07976) | 针对GRPO类强化学习方法在成功轨迹中会同时强化错误尾部令牌的“正信用污染”问题，论文提出TACO方法，以策略自身为参考计算令牌尾部风险分数，在无需令牌级错误标签的情况下柔和降低高风险令牌的正信用，同时保留学习有用稀有模式的机会。 |
| [^367] | [AirPASS: Over-the-Air Federated Learning via Pinching Antenna Systems](https://arxiv.org/abs/2607.06768) | 本文提出AirPASS交替优化框架，通过同伦-黎曼裕度巩固方法与同伦辅助几何优化方法交替求解设备选择、接收波束成形和夹持天线位置的联合非凸优化问题，在夹持天线系统辅助的空中联邦学习中实现聚合失真阈值约束下的设备接入数量最大化。 |
| [^368] | [HarnessMDP: Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning](https://arxiv.org/abs/2607.05458) | 提出HarnessMDP框架，将LLM智能体执行框架的控制建模为有限时域马尔可夫决策过程，通过离线强化学习从任务轨迹中学习框架策略，使框架能够根据演变的智能体状态自适应地选择结构性执行动作。 |
| [^369] | [ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum](https://arxiv.org/abs/2607.04535) | ManifoldFlow通过对固定谱Stiefel层进行最小松弛，将权重分解为 W = Q S^{1/2}，在保持基向量位于Stiefel流形上的同时学习有界的正定奇异谱，使特征值裁剪成为直接的奇异值控制机制，并在序列、表格和图像任务中优于固定谱Stiefel层。 |
| [^370] | [UniWind: Day-Ahead Wind Power Forecasting via Physics-Informed State Routing](https://arxiv.org/abs/2607.01670) | UniWind框架通过将物理信息发电先验与数据驱动的状态感知校正相结合，有效区分气象驱动的发电潜力与本地环境运行因素的影响，实现更准确的日前风电功率预测。 |
| [^371] | [Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping](https://arxiv.org/abs/2606.29983) | 该论文揭示了循环Transformer长度外推脆弱性的根源在于序列长度与循环次数之间的虚假相关性，并提出通过在训练中引入随机循环次数（包括学习型的RL-Halting调度）来显著降低分布外方差、稳定模型预测。 |
| [^372] | [Symbolic Mechanistic Data Attribution: Tracing Training Influence to Learned Behavioral Policies](https://arxiv.org/abs/2606.29171) | 提出符号机制数据归因（SMDA）框架，通过在稀疏自编码器特征上拟合可解释的符号策略，并将每个微调样本的影响分解为特征激活（ΔX）与输出概率（ΔY）两条通路，从而将训练数据归因到模型习得的行为策略，并借此揭示基础模型安全行为中的系统性缺陷（如宗教刻板印象）。 |
| [^373] | [Replica Symmetry Breaking and Algorithmic Thresholds in Empirical Risk Minimization under Multi-Index Model](https://arxiv.org/abs/2606.28573) | 该论文在多指标模型的监督学习设定下提出增量近似消息传递（IAMP）算法，精确刻画了经验风险地形中多项式时间算法可到达的区域，并揭示了副本对称破缺现象与算法阈值。 |
| [^374] | [Set Membership Estimation for Stochastic Linear Contextual Bandits](https://arxiv.org/abs/2606.20022) | 本文首次将控制与系统辨识领域成熟的集合成员估计（SME）方法引入线性上下文赌博机，提出基于SME的OFU算法并建立了次线性遗憾界。 |
| [^375] | [Identification and Inference for Algorithmic Frontiers with Selective Labels](https://arxiv.org/abs/2606.14977) | 本文在标签仅对被选择个体可观测（选择性标签）的情形下，刻画了公平性-准确性前沿的识别区域，并提出了去偏机器学习估计量及统计推断方法，用以检验现有算法是否位于前沿上，并构造在不损害公平性的前提下提升各群体准确性的改进算法。 |
| [^376] | [LLM-as-an-Investigator: Evidence-First Agentic Reasoning for Robust Interactive Problem Diagnosis](https://arxiv.org/abs/2606.13220) | 该论文提出“LLM作为调查者”这一证据优先方法论，通过SIA智能体协调诊断推理、假设维护与反事实质疑三种互补推理行为，减少诊断中过早锁定错误假设的问题，实现更稳健的交互式问题诊断。 |
| [^377] | [Let's Ask Gauss: Improved One-Run Privacy Auditing](https://arxiv.org/abs/2606.12733) | 该论文提出了一种基于中心极限定理的高斯分布视角的单次运行差分隐私审计框架，将金丝雀对齐信号视为随机变量序列而非二值化猜测，从而从单次训练中就能获得更紧的隐私下界。 |
| [^378] | [Feature-preserving Latent-EnKF for Data Assimilation of Flows with Shocks](https://arxiv.org/abs/2606.12559) | 本文提出一种特征保持的潜空间集合卡尔曼滤波器，通过坐标条件化自解码器在低维潜空间中执行集合更新，解决了传统EnKF在含激波等间断流动数据同化中因间断位置不确定性而失去物理一致性的问题。 |
| [^379] | [Neuro-Relational Programs: Unifying Queries and Neural Computation over Structured Data](https://arxiv.org/abs/2606.11946) | 本文提出神经关系程序（NRPs），一种在关系数据库事实上携带向量嵌入的声明式查询语言，通过扩展Datalog规则实现嵌入的组合、聚合与转换，将关系推理与可学习神经组件统一于同一形式体系中。 |
| [^380] | [When to Align, When to Predict: A Phase Diagram for Multimodal Learning](https://arxiv.org/abs/2606.11190) | 本文通过带结构化跨模态噪声相关的尖峰信号加噪声模型，揭示了跨模态对齐与跨模态预测两种范式的互补失败模式，并构建相图为多模态学习中范式的选择提供了理论指导。 |
| [^381] | [Near-Exponential Convergence Rates for kNN Classification based on Boltzmann Margin](https://arxiv.org/abs/2606.10361) | 本文提出了一种介于Tsybakov边际与Massart边际之间的新条件——玻尔兹曼边际，并据此首次为kNN分类建立了近指数收敛速率。 |
| [^382] | [Structure-Preserving Correction Learning for Sparse Bayesian Inference in Brain Source Imaging](https://arxiv.org/abs/2606.07196) | 该论文提出一种结构保持的校正学习框架，将经典的稀疏Type-II贝叶斯联合超参数求解器展开为可训练的神经网络，通过学习结构化的残差校正项以数据驱动的方式改进M/EEG脑源成像的推断性能，同时保留贝叶斯方法的可解释性。 |
| [^383] | [Rethinking Calibration for High-Sparsity LLM Pruning: From Single Sources to Capability Coverage](https://arxiv.org/abs/2606.03328) | 该论文将大语言模型剪枝的校准问题重新定义为能力覆盖问题，证明固定预算下均衡混合多种来源的校准数据（Balanced）显著优于任何单一来源，且稀疏度越高优势越大（60%稀疏度下比C4高18.8分），并在无法获得对齐语料时提出IGSP作为近似方案。 |
| [^384] | [Linear Strategic Classification with Endogenous Improvements](https://arxiv.org/abs/2606.01198) | 该论文研究了策略性分类的改进感知变体，证明在单指标资格模型和线性可分解代价下，智能体的最优响应可简化为决策边界的闭式偏移，该问题具有线性样本复杂度的可学习性，并提出了一个用估计代替预言机的实用插入式算法。 |
| [^385] | [LLMSurgeon: Diagnosing Data Mixture of Large Language Models](https://arxiv.org/abs/2605.30348) | 本文提出LLMSurgeon框架，仅通过分析大语言模型生成的文本，将预训练数据混合估计建模为标签偏移假设下的带约束逆问题，利用校准的软混淆矩阵纠正系统性领域混淆，从而诊断出模型预训练语料的领域级分布。 |
| [^386] | [GRASP: Plan-Guided Graph Retrieval with Adaptive Fusion and Reranking on Semi-Structured Knowledge Bases](https://arxiv.org/abs/2605.30237) | GRASP通过小型LLM规划器生成可执行图计划、融合图排序与文本排序，并用强化学习优化规划器以最大化融合排序增益，在半结构化知识库检索上取得全面领先的性能，平均Hit@1比最强基线提升18.6个百分点且单查询仅需2.47秒。 |
| [^387] | [Detectability without Interference: Efficient Canary Design for Privacy Auditing in One Run](https://arxiv.org/abs/2605.27292) | 本文提出IBIS框架，通过基于影响的贪心初始化与双层优化设计相互干扰最小、检测分数最优的金丝雀样本，使隐私审计仅需一次训练运行即可高效完成。 |
| [^388] | [Supervised Learning Objectives for High-Dimensional Treatment Effects](https://arxiv.org/abs/2605.27281) | 该论文将因果推断重新表述为监督学习问题，证明了因果误差可分解为递增阶数的矩平衡误差，并据此设计出能直接改善高维处理效应（如文本干预）因果估计的学习目标，同时可将效应投影到更粗粒度的属性上。 |
| [^389] | [Amortized Factor Inference Networks for Posterior Inference](https://arxiv.org/abs/2605.26419) | 本文提出摊销因子推断网络（AFIN），将模型表示为类型化的先验与似然因子列表，并通过参数不依赖潜变量维度和观测数量的编码-合并-解码网络，实现单一网络跨不同先验、似然、潜变量维度和观测数量的通用后验推断。 |
| [^390] | [ChaosBench-Logic v2: Evaluating LLM Logical Reasoning over Dynamical Systems at Scale](https://arxiv.org/abs/2605.24305) | 该论文提出了大规模动力系统逻辑推理基准ChaosBench-Logic v2和CARE评估协议，揭示了即使是最先进的LLM在状态转变推理上仍接近随机水平，并存在先验崩溃与释义不一致等系统性缺陷。 |
| [^391] | [Evolutionary Multi-Task Optimization for LLM-Guided Program Discovery](https://arxiv.org/abs/2605.22613) | 提出EMO-STA两阶段框架，先跨任务族演化出共享程序档案库，再将共享候选程序适配到各目标任务，在八个任务族上以相同的进化迭代预算超越了单任务进化方法。 |
| [^392] | [Bridging the Gap: Converting Read Text to Conversational Dialogue](https://arxiv.org/abs/2605.18001) | 本文提出PACC方法，利用深度神经网络和HiFi-GAN对韵律特征进行分析与调整，将朗读语音转换为自然流畅的会话语音，同时兼顾实时应用的自然度、可懂度与低计算开销。 |
| [^393] | [Neurodata Without Boredom: Benchmarking Agentic AI for Data Reuse](https://arxiv.org/abs/2605.12808) | 该论文提出了首个针对编程智能体在神经科学数据复用任务上的基准测试，通过让AI智能体阅读论文、代码和数据，并将大规模小鼠神经记录数据重新格式化以训练下游解码器，评估智能体AI在减轻神经科学数据整理负担方面的能力。 |
| [^394] | [Plan Before You Trade: Inference-Time Optimization for RL Trading Agents](https://arxiv.org/abs/2605.12653) | 提出了受模型预测控制启发的FinPILOT框架，利用多步价格预测在推理时对任意预训练的强化学习交易策略进行优化，实现“先规划、后交易”的策略自适应调整。 |
| [^395] | [A proximal gradient algorithm for composite log-concave sampling](https://arxiv.org/abs/2605.12461) | 提出了一种针对复合对数凹分布的近端梯度采样算法，通过结合 $f$ 的梯度评估与 $g$ 的受限高斯预言机，在 $f+g$ 为 $\alpha$-凸且 $f$ 为 $\beta$-光滑的条件下，以 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$ 次预言机调用实现总变差距离下 $\varepsilon$ 精度的采样，匹配了 $g=0$ 情形下的最优结果。 |
| [^396] | [Locking Pretrained Weights via Deep Low-Rank Residual Distillation](https://arxiv.org/abs/2605.10777) | 提出DLR-Lock方法，利用自动微分在推理与训练之间的不对称性，通过深度低秩残差蒸馏锁定预训练权重，防止适应性攻击者对开放权重模型进行未经授权的微调和滥用。 |
| [^397] | [TabFORGE: A Tabular Foundation Model for Generative Modelling](https://arxiv.org/abs/2605.09424) | 提出了TabFORGE——一个基于预训练表格基础生成表示的新型表格基础模型，通过更好地对齐表格数据的特征间依赖结构，显著提升生成式表格建模的合成数据质量。 |
| [^398] | [HAServe: Horizon-Aware Scheduling and Memory Co-Design for Efficient LLM Serving](https://arxiv.org/abs/2605.08581) | 提出HAServe，一种面向在线LLM服务的调度与内存协同设计方法，通过统一需求视图联合优化请求准入与KV缓存保留，以最小化预测延迟成本。 |
| [^399] | [CRAFT: Forgetting-Aware Intervention-Based Adaptation for Continual Learning](https://arxiv.org/abs/2605.05732) | CRAFT提出了一种冻结大模型权重、通过学习低秩隐藏表示干预实现持续学习的框架，利用基于输出分布的任务相似性共享干预、KL惩罚以及深层任务私有干预来缓解灾难性遗忘，且在测试时无需任务身份标签即可自动识别任务。 |
| [^400] | [Identifier-Free Code Embedding Models for Scalable Search](https://arxiv.org/abs/2605.05251) | 该论文形式化了源代码与去除符号的反编译代码之间的双向函数关联问题，并通过对比学习微调Qwen3-Embedding模型，在所有基线测试中大幅超越现有模型。 |
| [^401] | [Do LLMs have core beliefs?](https://arxiv.org/abs/2605.03255) | 本文提出“对抗对话树”探测框架，在五个知识领域测试发现大多数大语言模型无法在对抗性对话压力下维持稳定的世界观和核心信念，表明它们并不具备类似人类的核心信念。 |
| [^402] | [Mind the Gap: Structure-Aware Consistency in Preference Learning](https://arxiv.org/abs/2604.27733) | 本文证明了标准替代损失在LLM对齐中无法保证排序一致性，并提出一种结构感知的边际调整方法（SA-DPO），通过利用响应间的语义距离实现稳定且一致的对齐。 |
| [^403] | [TensorDex: A Compact, Tensor-Centric Storage System for Modern AI Models](https://arxiv.org/abs/2604.17104) | TensorDex是一个无损的、以张量为粒度进行增量压缩的存储系统，通过紧凑的位级指纹预测张量对的可压缩性，有效消除模型托管平台中微调变体之间的跨模型冗余。 |
| [^404] | [Sim-to-Real Transfer for Muscle-Actuated Robots via Generalized Actuator Networks](https://arxiv.org/abs/2604.09487) | 提出广义执行器网络（GenAN），无需扭矩传感器、仅从关节位置轨迹即可学习气动人工肌肉等复杂驱动系统的神经模型，实现了肌肉驱动机器人动态精确技能从仿真到真实系统的迁移。 |
| [^405] | [Where to Steer: Input-Dependent Layer Selection for Steering Improves LLM Alignment](https://arxiv.org/abs/2604.03867) | 该论文发现最优引导层因输入而异，并提出W2S框架，根据输入自适应地选择引导干预的最佳层位置，从而改进大语言模型的推理时对齐效果。 |
| [^406] | [A Distribution-to-Distribution Neural Probabilistic Forecasting Framework for Dynamical Systems](https://arxiv.org/abs/2603.25370) | 本文提出了一种分布到分布（D2D）神经概率预测框架，利用核均值嵌入编码输入分布、混合密度网络参数化输出分布，实现了动力系统的递归多步概率预测，避免了求解计算代价高昂的分布演化方程。 |
| [^407] | [SEVerA: Verified Synthesis of Self-Evolving Agents](https://arxiv.org/abs/2603.25111) | 提出了SEVerA框架，通过形式化守护生成模型（FGGM）将自演化智能体代码生成形式化为受约束的学习问题，从而为自演化LLM智能体提供安全性和正确性的形式化保证。 |
| [^408] | [A Model Ensemble-Based Post-Processing Framework for Fairness-Aware Prediction](https://arxiv.org/abs/2603.18838) | 本文提出一种基于模型集成的后处理框架，该方法独立于模型内部结构、适用于多种学习任务和公平性定义，能够在几乎不损失预测精度的前提下有效提升模型公平性。 |
| [^409] | [Computational and Statistical Hardness of Calibration Distance](https://arxiv.org/abs/2603.18391) | 该论文提出了在均匀边际和无噪声标签假设下精确计算校准距离的高效算法，证明移除任一假设后问题即变为NP难，并给出了适用于一般情形的PTAS以及估计问题所需的Θ(1/ε³)样本复杂度的紧界。 |
| [^410] | [Learning Not To Interfere: Adapting Experts For Model Merging](https://arxiv.org/abs/2603.13467) | 本文提出了轻量级适配框架RI（干扰解决），通过将专家模型解耦为与其他任务空间功能正交的形式来减少模型合并中的跨任务干扰，且仅需未标记的辅助数据即可实现，适用于数据稀缺场景。 |
| [^411] | [3DTCR: Vortex-following generative reconstruction improves tropical cyclone intensity prediction from global AI weather forecasts](https://arxiv.org/abs/2603.13049) | 提出3DTCR框架，利用涡旋跟随的条件流匹配生成式重建方法，从全球AI天气预报中恢复热带气旋公里级三维内核结构，从而显著改进热带气旋强度预测。 |
| [^412] | [Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial RoPE](https://arxiv.org/abs/2603.11611) | 仅对约10%的隐藏维度应用旋转位置编码（部分RoPE）即可达到与完整RoPE相当的收敛性能，同时实现高达10倍的内存节省，且最小限度的RoPE使用或QK-Norm可缓解NoPE导致的损失尖峰。 |
| [^413] | [Self-Scaled Broyden Family of Quasi-Newton Methods in JAX](https://arxiv.org/abs/2603.10599) | 本文实现了自尺度Broyden族拟牛顿方法（包括BFGS、DFP、Broyden、SSBFGS、SSDFP、SSBroyden及L-BFGS）在JAX中的开源工具库，以方便JAX社区使用这些优化器。 |
| [^414] | [Size Transferability of Graph Transformers with Convolutional Positional Encodings](https://arxiv.org/abs/2602.15239) | 本文建立了带有GNN位置编码的图Transformer与流形神经网络之间的理论联系，从理论上证明了在小图上训练的图Transformer在温和假设下能够泛化到更大的图，并通过实验验证了其与GNN相当的可扩展性。 |
| [^415] | [Self-Refining Vision Language Model for Robotic Failure Detection and Reasoning](https://arxiv.org/abs/2602.12405) | ARMOR提出了一种多任务自精炼框架，通过融合大规模稀疏二值标签与小规模丰富推理标注的异构监督，并结合离线与在线模仿学习，实现对机器人故障的迭代式检测与自然语言推理，摆脱了对大量人工标注的依赖。 |
| [^416] | [Mitigating Memorization Where It Happens](https://arxiv.org/abs/2602.08901) | 该论文提出了一个探测—引导框架，利用目标模型与低暴露参考模型之间的 token 级对数似然比精准定位记忆化发生的位置，并仅在这些位置施加有针对性的激活校正，从而在缓解逐字记忆问题的同时避免损害模型的正常泛化性能。 |
| [^417] | [TASTE: Task-Aware Out-of-Distribution Detection via Stein Operators](https://arxiv.org/abs/2602.07640) | 提出基于Stein算子的任务感知分布外检测框架TASTE，将分布偏移解释为向模型敏感度场的投影，不仅能检测偏移还能通过逐坐标分解和逐像素诊断实现偏移的定位。 |
| [^418] | [The impact of temporal resampling on the safety of clinical reinforcement learning: A simulated trial](https://arxiv.org/abs/2602.06603) | 本研究通过模拟试验首次量化了时间分箱重采样对临床离线强化学习数据造成的失真，发现其会颠倒血糖与胰岛素之间的因果顺序（超过70%的配对样本）、掩盖血糖异常事件，从而威胁临床强化学习的安全性。 |
| [^419] | [ReText: Text Boosts Generalization in Image-Based Person Re-identification](https://arxiv.org/abs/2602.05785) | 提出ReText方法，通过微调视觉-语言模型为多摄像头行人重识别数据集生成文本描述，并利用文本监督以及单摄像头数据的视觉与文本标注联合训练，以语义线索增强视觉特征，从而提升模型在未见域上的行人重识别泛化能力。 |
| [^420] | [Soft Sensor for Bottom-Hole Pressure Estimation in Petroleum Wells Using Long Short-Term Memory and Transfer Learning](https://arxiv.org/abs/2602.03737) | 该论文提出一种结合LSTM和迁移学习的软测量方法，通过井口数据估计石油井井底压力，在真实海上数据集上实现低于2%的MAPE，为昂贵的井下物理传感器提供了经济且准确的替代方案。 |
| [^421] | [CoCopula: Conformal Calibration and Copula Aggregation for Reliable Renewable Energy Forecasting](https://arxiv.org/abs/2602.02583) | 提出CoCopula框架，将保形校准与Copula聚合相结合，把第三方异构供应商提供的站点级概率预测直接转化为可靠且校准良好的车队级可再生能源概率预测。 |
| [^422] | [Product Interaction: An Algebraic Formalism for Deep Learning Architectures](https://arxiv.org/abs/2602.02573) | 本文提出“乘积交互”这一代数形式化框架，将乘法而非仅线性变换作为基本架构操作，通过自交互阶数统一了卷积网络、注意力机制、状态空间模型（如Mamba）和等变网络，使深度学习架构设计成为系统化流程。 |
| [^423] | [TabClustPFN: A Prior-data Fitted Network for Tabular Data Clustering](https://arxiv.org/abs/2601.21656) | TabClustPFN通过协同设计的先验、训练目标与解耦架构，在单次推理中同时推断表格数据的聚类分配和未知聚类基数，从而克服了无监督、基数未知和标签交换等聚类中的核心挑战。 |
| [^424] | [DeFlow: Decoupling Behavior-Prior Modeling and Value Maximization for Offline Policy Extraction](https://arxiv.org/abs/2601.10471) | DeFlow通过解耦行为先验建模与价值最大化，在流流形的信任域内学习轻量级精炼模块，避免对ODE求解器求导，在完整保留流模型迭代表达能力的同时实现稳定的离线强化学习性能提升与高效的在线适应。 |
| [^425] | [Imaging-anchored Multiomics in Cardiovascular Disease: Integrating Cardiac Imaging, Bulk, Single-cell, and Spatial Transcriptomics](https://arxiv.org/abs/2601.07871) | 本综述提出以心脏影像为锚点的多组学整合框架，将超声心动图、心脏MRI和CT所定义的心脏空间表型，与bulk、单细胞及空间转录组提供的细胞类型和位置特异的分子背景相融合，从而打破数据孤岛、实现心血管疾病的多模态联合分析。 |
| [^426] | [Diffusion Models in Simulation-Based Inference: A Tutorial Review](https://arxiv.org/abs/2512.20685) | 本综述系统梳理了扩散模型在基于模拟推断（SBI）中的最新进展，涵盖训练、推断与评估的设计选择，并讨论了引导、流匹配、一致性模型等概念以及噪声调度、参数化和采样器对效率与统计精度的影响。 |
| [^427] | [Boltzmann generators for amorphous particle systems](https://arxiv.org/abs/2512.16607) | 本研究针对平衡采样极其困难的非晶材料（玻璃），通过将所需的等变性直接嵌入黎曼结构中，开发了首个专为非晶粒子系统定制的玻尔兹曼生成器。 |
| [^428] | [Scientific feedback reveals productive regimes beyond training conventions](https://arxiv.org/abs/2512.15756) | 科学性能反馈能够引导生成模型自主探索超越人类设计惯例的高产区域，在提升科学性能的同时生成训练数据中被排除的选择，且无需显式奖励对惯例的偏离。 |
| [^429] | [CarBench: A Comprehensive Benchmark for Neural Surrogates on High-Fidelity 3D Car Aerodynamics](https://arxiv.org/abs/2512.07847) | 本文提出了CarBench，首个针对大规模三维汽车空气动力学的综合基准，在包含8000多个高保真模拟的DrivAerNet++数据集上系统评估了十一种最先进的神经替代模型架构。 |
| [^430] | [AlignSAE: Concept-Aligned Sparse Autoencoders](https://arxiv.org/abs/2512.02004) | AlignSAE通过“先预训练、后后训练”的两阶段课程，将稀疏自编码器的部分潜变量与预定义概念对齐并绑定到专用槽位，同时保留其余容量进行通用重构，从而实现对大语言模型内部特定概念的无干扰检查与精确控制。 |
| [^431] | [SUPN: Shallow Universal Polynomial Networks for Efficient Function Approximation](https://arxiv.org/abs/2511.21414) | 提出浅层通用多项式网络（SUPN），通过在每个激活函数内部嵌入可学习的多元多项式（多项式预激活），以单个隐藏层实现更强的表达能力，用更少的参数和更低的训练成本完成高效函数逼近。 |
| [^432] | [Graph Homomorphism Distortion: A structural (pseudo)-metric for attributed graphs](https://arxiv.org/abs/2511.03068) | 本文提出一种基于图同态的新型（伪）度量——图同态失真，通过衡量图间映射时节点特征的最小最坏失真来实现属性图的高效、细粒度区分，并可作为结构编码将基于同态的描述符扩展到属性图。 |
| [^433] | [Arithmetic Circuits and Neural Networks for Regular Matroids](https://arxiv.org/abs/2511.02406) | 本文证明了存在大小为 $O(n^3)$ 的均匀算术电路和 ReLU 神经网络可用于正则拟阵的基生成多项式计算与加权基最大化，并首次展示了虚拟扩展公式可以比已知最佳的单个扩展公式更高效。 |
| [^434] | [Multi-task Neural Diffusion Processes](https://arxiv.org/abs/2510.03419) | 提出多任务神经扩散过程（MT-NDPs），通过任务编码器提取任务表示并条件化扩散模型，实现跨任务信息共享的任务条件化概率回归与少样本自适应。 |
| [^435] | [Thinking-Free Policy Initialization Makes Distilled Reasoning Models More Effective and Efficient Reasoners](https://arxiv.org/abs/2509.26226) | 提出无思考策略初始化（TFPI），通过在RLVR训练中显式丢弃思考内容，使蒸馏推理模型在提升性能的同时降低token消耗，架起了长思维链蒸馏与标准强化学习之间的桥梁。 |
| [^436] | [A Greedy PDE Router for Blending Neural Operators and Classical Methods](https://arxiv.org/abs/2509.24814) | 本文提出一种近似贪心PDE路由器，在混合迭代求解过程中每步智能选择神经算子或经典数值求解器以融合两者优势，在多种PDE求解任务中持续缩短求解时间。 |
| [^437] | [Quality control in sublinear time: a case study via random graphs](https://arxiv.org/abs/2508.16531) | 该论文提出了一类新的算法问题——“质量控制问题”，研究如何在次线性时间内判断任意输入是否为高质量（即能否信任算法在该输入上的表现），并以随机图作为案例进行研究。 |
| [^438] | [Optimal Condition for Initialization Variance in Deep Neural Networks: An SGD Dynamics Perspective](https://arxiv.org/abs/2508.12834) | 该论文通过将SGD近似为朗之万动力学并利用Fokker-Planck方程与KL散度，建立了期望损失与初始化分布的联系，从而推导出深度神经网络初始化方差的最优条件。 |
| [^439] | [Multimodal Remote Inference](https://arxiv.org/abs/2508.07555) | 该论文针对多模态远程推理系统提出了一种以最小化机器学习推理误差为目标的多模态调度问题，将推理误差建模为信息年龄向量的函数，通过半马尔可夫决策过程求解，并揭示了双模态与多模态情形下最优策略链结构的本质差异。 |
| [^440] | [AI Scientists for Building Virtual-Cell Models](https://arxiv.org/abs/2508.02276) | 本文提出CellForge，一个由多个自主智能体组成的有组织团队，能够将真实复杂的单细胞扰动数据自动转化为可执行的模型假设与架构，实现了从文献调研、数据分析到模型构建的全流程自动化AI科学家系统。 |
| [^441] | [Formal Bayesian Transfer Learning via the Total Risk Prior](https://arxiv.org/abs/2507.23768) | 该论文提出了一种新的贝叶斯迁移学习方法，其核心创新在于使用以源参数为条件的风险最小化器（而非传统的经验损失最小化器）来构建源数据集与目标数据集所有参数的单一联合先验分布，从而实现完整的贝叶斯不确定性量化以及通过Gibbs采样对源数据集进行选择和模型平均。 |
| [^442] | [Stabilizing the Modality Gap: Null-Space Projected Continual Learning for Vision-Language Models](https://arxiv.org/abs/2507.19839) | 提出FPCD框架，通过快速零空间投影减少对已学习表示的干扰，并结合对比蒸馏对齐跨模态映射，从而稳定视觉-语言模型持续学习中的模态间隙漂移，同时将投影基构建速度提升5-9倍。 |
| [^443] | [PAC-Causal Learnability: From Identification to Finite-Sample Guarantees](https://arxiv.org/abs/2507.18903) | 该论文提出了PACC因果可学习性框架，首次从PAC学习角度为因果估计提供了有限样本的统一精度-置信度保证，并揭示其样本复杂度由因果识别强度、目标复杂度和干扰复杂度三个关键因素决定。 |
| [^444] | [PanoDiff-SR: Synthesizing Dental Panoramic Radiographs using Diffusion and Super-resolution](https://arxiv.org/abs/2507.09227) | 本文提出PanoDiff-SR，一种结合扩散模型与超分辨率的端到端两阶段流水线，用于生成完整且达到诊断分辨率的全尺寸牙科全景X光片，克服了基于GAN方法存在的视野受限、易被临床医生识别以及训练不稳定等局限。 |
| [^445] | [Tourist arrival forecasting with game-theoretic social influence weighting and deep learning](https://arxiv.org/abs/2507.03411) | 该论文提出一种融合合作博弈论意见领袖识别（其Shapley值具有闭式解且最大协同联盟可在多项式时间内精确求解）、因果经验小波变换与贝叶斯优化调优的堆叠BiLSTM网络的混合预测方法，实现了对香港和三亚游客到达量的准确预测。 |
| [^446] | [Differentially Private Distribution Release of Gaussian Mixture Models via KL-Divergence Minimization](https://arxiv.org/abs/2506.03467) | 本文提出了一种以KL散度为效用度量驱动的优化框架，在差分隐私保证下发布高斯混合模型的混合权重、分量均值和协方差矩阵，以防止底层数据敏感信息的泄露。 |
| [^447] | [HERO: Preserving Structure and Semantics in Heterogeneous Continual Graph Learning](https://arxiv.org/abs/2505.17458) | 该论文提出了HERO框架，通过将经验重放的误差分解为目标选择与上下文重构两部分，证明了仅重放目标节点是不够的，从而在异构持续图学习中同时从结构和语义两个层面保存历史知识。 |
| [^448] | [Parallel Layer Normalization for Universal Approximation](https://arxiv.org/abs/2505.13142) | 该论文证明仅含单个层归一化单元的浅层网络无论宽度多大都不具备万能逼近能力，而提出的并行层归一化网络（PLN-Nets）随着并行分组数量的增加即可实现万能逼近，且该机制还可推广到RMSNorm变体、逐位置前馈网络和卷积网络。 |
| [^449] | [ZenFlow: Enabling Stall-Free Offloading for LLM Training](https://arxiv.org/abs/2505.12242) | ZenFlow提出了一种新型卸载训练框架，通过利用重要梯度的时空局部性，在GPU上原位更新重要梯度、同时在CPU上异步累积次要梯度并完全重叠CPU工作与GPU计算，从而消除大语言模型训练中的GPU停顿。 |
| [^450] | [KnowEEG: Explainable Knowledge Driven EEG Classification](https://arxiv.org/abs/2505.00541) | 该论文提出KnowEEG可解释分类流程，通过结合单电极统计特征与电极间连接性度量并采用平衡采样策略，在五项脑电图分类任务上达到或超越最先进深度学习模型的性能，同时具备良好的可解释性。 |
| [^451] | [A self-learning framework for adaptive free-form pipe routing in aeroengines](https://arxiv.org/abs/2504.03669) | 该论文提出了一种基于近端策略优化（PPO）算法的自学习管路布线方法（SLPR），通过统一的规则建模框架和势能表实现快速障碍检测与模糊规则建模，从而在动态设计环境下自适应地优化航空发动机的自由形态管路布线。 |
| [^452] | [FANTF: Fuzzy attention network based transformers for time series analysis](https://arxiv.org/abs/2504.00070) | 提出了一种基于二型模糊集理论的模糊注意力机制FANTF，利用可学习的高斯不确定性扰动建模注意力得分中的不确定性，可作为标准注意力层的即插即用替代模块，提升Transformer在时间序列预测、分类和异常检测任务上的性能。 |
| [^453] | [BOLT: Universal Classifier Training via a Bayes-Error Upper Bound](https://arxiv.org/abs/2501.07754) | 本文提出了一种从分类器输出样本计算贝叶斯错误率通用上界的新方法，并据此定义了BOLT损失函数用于训练分类器，实验表明其往往比交叉熵训练取得更低的分类错误率。 |
| [^454] | [A Multimodal Emotion Recognition System: Integrating Facial Expressions, Body Movement, Speech, and Spoken Language](https://arxiv.org/abs/2412.17907) | 本文提出了一种整合面部表情、身体动作、语音和口语的多模态情感识别系统，旨在辅助而非取代心理评估者，并通过真实条件下的用户研究揭示了模型准确率从基准测试到现实场景的显著下降。 |
| [^455] | [STAMP: Symmetric Temporal Averaging Message Passing for Domain Adaptation in Chronologically Split Graphs](https://arxiv.org/abs/2411.10957) | 提出对称时间平均消息传递（STAMP），通过在时间戳间平衡消息强制时间域间的一阶矩不变性，从而解决按时间顺序划分图中的分布偏移问题，提升半监督节点分类性能。 |
| [^456] | [Pairwise Difference Learning for Classification](https://arxiv.org/abs/2406.20031) | 本文将成对差分学习（PDL）从回归扩展到分类任务，提出通过在成对训练数据上求解二分类问题来构建PDL分类器，在大规模实证研究中其预测性能优于现有最先进方法。 |
| [^457] | [Second Maximum of a Gaussian Random Field and Exact (t-)Spacing test](https://arxiv.org/abs/2406.18397) | 本文提出了黎曼流形上高斯随机场次大值的新概念，借助 Kac-Rice 公式推导出以次大值和 Hessian 矩阵独立部分为条件的全局最大值的显式分布，从而构造出一种可用于检测稀疏备择假设的精确间距检验方法。 |
| [^458] | [Obtaining Explainable Classification Models using Distributionally Robust Optimization.](http://arxiv.org/abs/2311.01994) | 本论文介绍了一种利用分布鲁棒优化获取可解释的分类模型的方法，通过构建稀疏的规则集合来同时解决规则集的稀疏性和预测准确性之间的权衡，从而保证泛化性能并降低计算成本。 |
| [^459] | [Machine Learning Infused Distributed Optimization for Coordinating Virtual Power Plant Assets.](http://arxiv.org/abs/2310.17882) | 本文提出了一种名为LOOP-MAC的方法，利用机器学习辅助的分布式优化来协调虚拟电厂（VPP）的资产。该方法采用多智能体协调视角，利用神经网络逼近器加速解决方案搜索。该方法通过引入测量图来保证最佳解之间的关系。 |

# 详细

[^1]: LIBERO-PeRM：个性化机器人操作基准测试

    LIBERO-PeRM: Benchmarking Personalized Robotic Manipulation

    [https://arxiv.org/abs/2610.22372](https://arxiv.org/abs/2610.22372)

    提出LIBERO-PeRM基准，通过受控形式化框架和涵盖运动、技能与用户档案的可观察偏好分类体系，从零样本偏好遵循、演示学习效率、跨布局跨任务泛化以及偏好组合与冲突处理五个方面系统评估具身智能体的个性化机器人操作能力。

    

    个性化对于通用机器人进入家庭环境并实现真正实用性至关重要。尽管通用机器人操作已被广泛研究，但个性化机器人操作缺乏一个受控基准来将任务完成与偏好满足区分开来。我们提出了LIBERO-PeRM，一个面向具身智能体偏好学习的基准。LIBERO-PeRM研究了五个问题：1）策略在零样本设置下如何遵循偏好；2）策略如何从演示中学习单一偏好；3）随着演示数量增加，策略获取偏好的效率如何；4）学到的偏好如何在不同布局和任务之间泛化；5）策略如何组合偏好以及在偏好发生冲突时的行为表现。我们提出了一个受控的形式化框架和仿真中可观察偏好的分类体系，涵盖运动、技能和用户档案三个方面的选择。该形式化框架仅限于语言明确表达的偏好……（原文摘要在此处截断）

    arXiv:2610.22372v1 Announce Type: cross  Abstract: Personalization is essential for general-purpose robots to transition into household environments and achieve true practicality. While general robotic manipulation has been extensively studied, personalized robotic manipulation (PeRM) lacks a controlled benchmark for separating task completion from preference satisfaction. We introduce LIBERO-PeRM, a benchmark for preference learning in embodied agents. LIBERO-PeRM studies five questions: 1) how policies follow preferences in zero-shot settings; 2) how they learn single preferences from demonstrations; 3) how efficiently they acquire preferences as demonstrations increase; 4) how learned preferences generalize across layouts and tasks; and 5) how policies compose preferences and behave under conflicts. We propose a controlled formulation and taxonomy of observable preferences in simulation, covering motion, skill, and user-profile choices. This formulation is restricted to language-exp
    
[^2]: PathFM：多元轨迹的快速生成式预测

    PathFM: Fast Generative Forecasting of Multivariate Trajectories

    [https://arxiv.org/abs/2610.22358](https://arxiv.org/abs/2610.22358)

    PathFM利用流匹配生成完整的多元未来轨迹以捕捉传统边际分位数模型无法刻画的依赖结构，并通过模型蒸馏与历史注意力缓存共享，实现了接近单次分位数查询速度的高效多路径生成。

    

    时间序列基础模型在通用预测任务中日益流行。然而，大多数领先模型预测的是边际分位数，未指定未来轨迹之间的依赖结构。这一区别对于依赖轨迹的量（如累计需求、峰值负荷和事件发生时间）至关重要，而点预测和边际分数无法揭示这种结构。为了对这种依赖性进行建模，我们提出了PathFM，一种基于轨迹的预测模型，利用流匹配生成完整的未来路径。我们进一步将模型蒸馏至一到两步，并在多次采样之间共享历史注意力缓存。这些技术使得PathFM能够在大约相当于分位数模型单次查询所需的时间内，为单个时间序列生成多条路径。蒸馏后的采样器还使得直接优化校准损失成为可能。PathFM在既定的预测基准fev-bench上取得了具有竞争力的边际精度。

    arXiv:2610.22358v1 Announce Type: new  Abstract: Time series foundation models are becoming increasingly popular for general-purpose forecasting. However, most leading models predict marginal quantiles, leaving the dependence structure of future trajectories unspecified. This distinction matters for trajectory-dependent quantities such as cumulative demand, peak load, and event timing, yet point and marginal scores do not reveal it. To model this dependence, we introduce PathFM, a trajectory-based forecasting model that uses flow matching to generate complete future paths. We further distill the model to one or two steps and share history attention cache across draws. Together, these enable PathFM to generate multiple paths for a single series in roughly the time needed for one query to a quantile model. The distilled sampler also makes it practical to optimize a calibration loss directly. PathFM achieves competitive marginal accuracy on the established forecasting benchmarks fev-bench
    
[^3]: 重新思考现代混合专家模型中的路由探索

    Rethinking Routing Exploration in Modern Mixture-of-Experts

    [https://arxiv.org/abs/2610.22353](https://arxiv.org/abs/2610.22353)

    本文揭示了现代MoE架构中共享专家削弱路由探索性与细粒度专家分割需要更大探索空间之间的冲突，并提出MirrorMoE通过token自适应抑制来缩小探索空间以解决该困境。

    

    混合专家模型已成为扩展模型参数规模的重要方法。现代MoE架构日益结合两种设计选择：细粒度专家分割与共享专家。前者通过在固定激活比例下支持更多专家组合来提升模型表达能力，后者则引入了对每个token都保持激活的专家。我们识别出这两种设计之间存在一种探索困境：共享专家通过降低token表示的多样性而削弱了路由的探索性，这与细粒度专家分割因组合式扩大的路由空间而产生的更大探索需求相冲突。因此，细粒度分割所提供的额外表达能力在实践中可能无法充分发挥。我们提出MirrorMoE，通过token自适应抑制不必要探索的方式来缩小探索空间，从而调和这一探索困境。

    arXiv:2610.22353v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) models have become an important approach for scaling model parameters. Modern MoE architectures increasingly combine two design choices: fine-grained expert segmentation and shared experts. The former increases expressiveness by enabling more expert combinations at a fixed activation ratio, and the latter introduces experts that are active for every token. We identify an exploration dilemma between these designs: shared experts can make routing less exploratory by reducing the variability of token representations. This conflicts with the greater need for exploration induced by fine-grained expert segmentation due to the combinatorially enlarged routing space. Consequently, the additional expressiveness offered by fine-grained segmentation may not be fully realized in practice. MirrorMoE is proposed to reconcile this exploration dilemma by shrinking the exploration space through token-adaptive suppression of unn
    
[^4]: MLExpert：迈向自主机器学习工程的通用智能体模型

    MLExpert: Towards a Generalist Agentic Model for Autonomous Machine Learning Engineering

    [https://arxiv.org/abs/2610.22344](https://arxiv.org/abs/2610.22344)

    本文提出了一种面向自主长期机器学习工程优化的后训练方法，通过从长交互轨迹中筛选有效监督信号并从中间结果中学习，训练出能够自主组织实验和调整优化策略的通用智能体模型MLExpert。

    

    大语言模型（LLM）正在从辅助科学研究和工程扩展到自主开发和改进AI系统，推动“AI for AI”（AI4AI）作为通往递归自我改进（RSI）的路径。工作流驱动的机器学习工程（MLE）系统通过管理长期实验的复杂编排，为实现这一目标提供了实用路径；而近期的MLE系统采用模型主导的交互方式，使模型能够自主组织实验，并根据执行反馈调整优化策略。然而，每次实验的反馈反映的是此前多个决策的累积结果，而一次长期的MLE运行又包含许多此类代价高昂的实验。因此，训练过程必须从漫长的交互轨迹中筛选有用的监督信号，并从中间结果中学习。我们提出了一种面向自主长期MLE优化的后训练方法，并将其应用于训练MLExpert，一个（摘要在此处截断）

    arXiv:2610.22344v1 Announce Type: new  Abstract: Large language models (LLMs) are expanding from assisting scientific research and engineering to autonomously developing and improving AI systems, advancing AI for AI (AI4AI) as a pathway toward recursive self-improvement (RSI). Workflow-driven machine learning engineering (MLE) systems provide a practical path toward this pursuit through complex orchestration that manages long-horizon experimentation, while recent MLE systems adopt model-directed interaction, allowing models to autonomously organize experiments and adapt optimization strategies based on execution feedback. However, feedback from each experiment reflects several preceding decisions, and a long MLE run contains many such costly experiments. Training must therefore select useful supervision from long interaction trajectories and learn from intermediate outcomes. We propose a post-training method for autonomous long-horizon MLE optimization and apply it to train MLExpert, a
    
[^5]: GeoState：学习视角解耦的几何状态用于3D世界建模

    GeoState: Learning View-Decoupled Geometry States for 3D World Modeling

    [https://arxiv.org/abs/2610.22343](https://arxiv.org/abs/2610.22343)

    GeoState提出一种视角解耦的几何状态表示，通过场景分词器聚合多视角几何基础模型特征并将相机运动单独建模，实现了场景动态演化与相机变化的解耦，从而提升3D世界模型状态预测的难度和灵活性。

    

    近期的3D世界模型越来越多地使用几何基础模型（GFM）的特征作为预测性世界状态，从而受益于其丰富的几何先验。然而，这些特征的视角相关性使场景动态与相机引起的变化纠缠在一起，导致所得到的状态演化更难以预测。我们提出GeoState，一种用于预测性3D世界建模的视角解耦几何表示，它将场景状态演化与依赖相机的观测分离开来。具体而言，场景分词器将多视角GFM特征聚合为视角解耦的场景状态，几何解码器通过相机射线对其进行查询。随后，自回归世界模型预测未来的场景状态，而相机运动则被单独建模。因此，预测的状态既可以在预测的相机下也可以在外部指定的相机下进行解码，同时在整个滚动预测过程中保持由上下文推导的单一尺度，而无需进行逐帧尺度对齐。

    arXiv:2610.22343v1 Announce Type: cross  Abstract: Recent 3D world models increasingly use features from geometry foundation models (GFMs) as predictive world states, benefiting from their rich geometric priors. However, the view dependence of these features entangles scene dynamics with camera-induced variation, making the resulting state evolution harder to predict. We propose GeoState, a view-decoupled geometry representation for predictive 3D world modeling that separates scene-state evolution from camera-dependent observation. Specifically, a scene tokenizer aggregates multi-view GFM features into a view-decoupled scene state, which a geometry decoder queries using camera rays. An autoregressive world model then predicts future scene states, while camera motion is modeled separately. The predicted states can therefore be decoded under either predicted or externally specified cameras, while retaining a single context-derived scale throughout the rollout without per-frame scale alig
    
[^6]: 面向形式化定理证明的Lean图搜索

    Lean Graph Search for Formal Theorem Proving

    [https://arxiv.org/abs/2610.22339](https://arxiv.org/abs/2610.22339)

    该论文提出LeanGraphSearch，一种通过构建Mathlib前向依赖图、沿多跳依赖链传播相关性来发现支撑引理的图检索方法，从而提升Lean 4形式化定理证明中的引理检索效果。

    

    检索相关引理对形式化定理证明至关重要，它使证明器能够基于已有的数学结果构建可机器验证的证明。近期的检索系统将数学库中的声明与目标定理或其证明的中间步骤进行匹配，但对声明之间的多跳依赖关系利用有限。这些依赖关系可以揭示那些与目标定理在词面上不一定相似、但在相关问题的证明中反复出现的支撑引理。为此，我们提出了LeanGraphSearch，这是一种面向Lean 4的基于图的检索方法，它根据Mathlib的前向依赖构建图，并通过遍历该图经由多跳依赖链来发现支撑引理。具体而言，LeanGraphSearch将每个声明与其所依赖的前提相连接，使用基于图的排序算法将相关性从语义匹配的声明传播到其依赖项，并对扩展后的候选结果进行重排序。

    arXiv:2610.22339v1 Announce Type: new  Abstract: Retrieving relevant lemmas is crucial for formal theorem proving, enabling provers to build machine-checkable proofs from established mathematical results. Recent retrieval systems match library declarations to a target theorem or its intermediate proof steps, but make limited use of multi-hop dependencies among declarations. These dependencies can reveal supporting lemmas that are not necessarily lexically similar to the target theorem but recur in proofs of related problems. To this end, we propose LeanGraphSearch, a graph-based retrieval method for Lean 4 that constructs a graph from Mathlib's forward dependencies and traverses it to uncover supporting lemmas through multi-hop dependency chains. Specifically, LeanGraphSearch links each declaration to the premises it depends on, uses a graph-based ranking algorithm to propagate relevance from semantically matched declarations to their dependencies, and reranks the expanded candidates a
    
[^7]: 模型与测量不确定性下动态治疗方案的鲁棒强化学习

    Robust Reinforcement Learning for Dynamic Treatment Regimes under Model and Measurement Uncertainty

    [https://arxiv.org/abs/2610.22336](https://arxiv.org/abs/2610.22336)

    本文提出了一种鲁棒强化学习框架，通过将患者建模为隐藏上下文，利用鲁棒上下文马尔可夫决策过程统一解决泛化与鲁棒性问题，学习出自适应治疗策略，在转移性前列腺癌自适应治疗模拟中几乎不损失性能的情况下实现鲁棒性，并在多种场景下帕累托优于现有方法。

    

    我们开发了一个鲁棒强化学习（RL）框架，用于学习能够跨患者泛化的自适应治疗策略，并使其在生物标志物测量误差和疾病动态变化的情况下保持可靠性。通过将患者建模为隐藏上下文，我们通过鲁棒上下文马尔可夫决策过程以统一的方式处理泛化性和鲁棒性问题。将在根据临床数据校准的模拟器中把我们的框架应用于转移性前列腺癌的自适应治疗，我们获得了泛化性强且鲁棒的治疗策略，其性能显著优于现有的自适应治疗方法。特别是，当生物标志物准确且疾病按预期演进时，我们在几乎没有性能损失的情况下实现了鲁棒性，并发现所得策略在多种不同场景下均帕累托优于现有方法。我们的研究结果表明，鲁棒强化学习作为一种有原则的方法论具有巨大前景。

    arXiv:2610.22336v1 Announce Type: new  Abstract: We develop a robust reinforcement learning (RL) framework for learning adaptive treatment policies that generalize across patients and remain reliable under errors in the measured biomarker and shifts in the disease dynamics. By modeling the patient as a hidden context, we address generalization and robustness in a unified manner through robust contextual Markov decision processes. Applying our framework to adaptive therapy for metastatic prostate cancer in a simulator calibrated to clinical data, we obtain generalizable and robust policies that substantially outperform the existing adaptive therapy method. In particular, we attain robustness with almost no loss of performance when the biomarker is accurate and the disease evolves as anticipated, and find that the resulting policy Pareto-dominates the existing method across a diverse array of scenarios. Our findings suggest that robust RL holds great promise as a principled methodology f
    
[^8]: 时光再现：面向时间序列推理的互补表示

    Twice Upon a Time: Complementary Representations for Time-Series Reasoning

    [https://arxiv.org/abs/2610.22327](https://arxiv.org/abs/2610.22327)

    该论文提出以互补表示作为时间序列推理的设计原则，通过融合折线图与归一化延迟嵌入两种视觉视图，让时间序列语言模型既保留绝对测量信息，又剔除幅度缩放和偏移等变化。

    

    时间序列语言模型（TSLMs）使得对时间数据进行基于语言的理解与推理成为可能，但一个根本问题仍然存在：其时间序列表示应当保留哪些信息，又应当剔除哪些变化？现有的TSLMs通常只向语言模型提供单一表示，从而在下游任务中施加了固定的归纳偏置。我们转而引入互补表示作为时间序列推理的设计原则。我们的模型结合了两种视觉视图：折线图保留绝对测量值与观测坐标系，而归一化的延迟嵌入则在剔除正幅度缩放与偏移的同时展现时间结构。专用的视觉编码器通过视觉标记和多级特征融合，将两种视图整合到共享的语言模型中。评估阶段的干预实验验证了这两种视图呈现出截然不同的不变性特征，……

    arXiv:2610.22327v1 Announce Type: cross  Abstract: Time Series Language Models (TSLMs) enable language-based understanding and reasoning over temporal data, yet a fundamental question remains: what information should their time-series representations preserve, and what variation should they factor out? Existing TSLMs typically expose the language model to a single representation, imposing a fixed inductive bias across downstream tasks. We instead introduce complementary representations as a design principle for time-series reasoning. Our model combines two visual views: line plots preserve absolute measurements and the observed coordinate frame, while normalized delay embeddings factor out positive amplitude rescaling and offsets while exposing temporal structure. Dedicated visual encoders integrate both views into a shared language model through visual tokens and multi-level feature fusion. Evaluation-time interventions verify that the two views exhibit distinct invariance profiles, w
    
[^9]: 学习引导大语言模型智能体与多智能体系统

    Learning to Steer LLM Agents and Multi-Agent Systems

    [https://arxiv.org/abs/2610.22316](https://arxiv.org/abs/2610.22316)

    提出一种由强化学习联合训练的引导智能体，通过快速与慢速双路径实时检测并纠正大语言模型智能体及多智能体系统中的目标漂移，提升长时程任务执行的可靠性。

    

    基于大语言模型（LLM）的智能体与多智能体系统日益通过长时程工具使用和智能体间协作来处理复杂任务。然而，这些系统在执行过程中容易偏离总体目标，这不仅浪费执行资源，还会导致不理想的结果，这种失败被称为目标漂移（goal drift）。在本工作中，我们提出一种引导智能体，它能够在执行过程中实时监督，并在发生漂移时将系统引导回正轨。该引导智能体观察执行上下文，决定何时以低成本的快速路径进行干预，以及如何以高成本的慢速路径进行干预。为了释放其引导智能——即评估目标漂移并生成有效干预措施的能力——我们采用强化学习联合训练这两条路径，将成对排序目标与组相对策略优化（GRPO）相结合。在信息检索搜索任务上的实验表明，该方

    arXiv:2610.22316v1 Announce Type: cross  Abstract: Large language model (LLM)-based agents and multi-agent systems increasingly tackle complex tasks through long-horizon tool use and inter-agent collaboration. However, these systems are prone to deviating from the overall goal as execution unfolds, which wastes execution resources and leads to undesired outcomes, a failure termed goal drift. In this work, we propose a steering agent that supervises execution on the fly and steers the system back on track when drift occurs. The steering agent observes the execution context, deciding when to intervene with a low-cost fast path and how to intervene with a high-cost slow path. To unlock its steering intelligence, the ability to assess goal drift and generate effective interventions, we jointly train the two paths with reinforcement learning, combining a pairwise ranking objective with group-relative policy optimization (GRPO). Experiments on information-seeking search tasks show that the s
    
[^10]: 微调前呈线性的知识在微调后能够泛化

    Knowledge That Is Linear Before Fine-Tuning Generalizes After It

    [https://arxiv.org/abs/2610.22300](https://arxiv.org/abs/2610.22300)

    论文提出区分语言模型中支持泛化的“活性事实”与仅可回忆的“惰性事实”，并通过重标注实验发现：微调前在模型内部呈线性可读的知识，在微调后能够泛化到共享相同答案的未训练实体上。

    

    并非语言模型能够回忆的所有事实都能同样好地支持进一步的学习。我们区分了“活性事实”与“惰性事实”：前者能够支持超出其自身回忆范围的泛化，后者虽然可以被回忆起来，但在给定任务中却不支持这种泛化。我们通过从已有关系构造新关系的方式来研究这一区分。以将地点映射到其所属大洲的关系（巴黎 → 欧洲）为例，为每个答案分配一个任意的新标签（欧洲 → 螃蟹，亚洲 → 斑马），于是新关系便将巴黎映射为螃蟹。我们在一部分主体上用新关系对模型进行微调，但不告诉模型原始关系或重新标注规则，然后测试模型能否为那些在原始关系中共享相同答案的留出主体分配相应的标签。也就是说，在学会了“巴黎 → 螃蟹”之后，模型能否回答出“慕尼黑 → 螃蟹”？在十一个语言模型和七十个关系的实验中，原始关系的线性可读程度……（摘要原文在此处截断）

    arXiv:2610.22300v1 Announce Type: cross  Abstract: Not all facts a language model can recall support further learning equally well. We distinguish active facts, which support generalization beyond their own recall, from inert facts, which can be recalled but do not support such generalization in a given task. We study this distinction by constructing new relations from existing ones. Take the relation that maps a place to its continent (Paris -> Europe) and assign each of its answers an arbitrary new label (Europe -> crab, Asia -> zebra). The new relation then maps Paris -> crab. We fine-tune the model on the new relation for a subset of subjects, without telling it the original relation or the relabeling rule, and test whether it assigns the corresponding labels to held-out subjects that share an answer in the original relation. Having learned Paris -> crab, does it answer Munich -> crab? Across eleven language models and 70 relations, how linearly readable the original relation is in
    
[^11]: 零比特神经音频水印中的向后兼容性

    Backward Compatibility in Zero-Bit Neural Audio Watermarking

    [https://arxiv.org/abs/2610.22296](https://arxiv.org/abs/2610.22296)

    本文首次形式化了零比特音频水印中的向后兼容性问题，提出一种将新检测器与冻结旧版嵌入器链配对、并采用性能加权采样的训练框架，使检测器在版本迭代中保持对旧内容的高精度检测，同时揭示了这种必要耦合会引发“嵌入器崩溃”效应。

    

    神经水印系统通常被训练为紧密耦合的嵌入器-检测器对，并隐含地假设静态部署。在现实世界的部署中，嵌入器和检测器经常被更新，并出于安全和版本管理的目的有选择地解耦，这可能会破坏对先前嵌入内容的检测，更关键的是，会增加检测器对那些其本应有意识不识别的嵌入器产生误报的比率。本文形式化了零比特音频水印中向后兼容性的概念，提出了一种训练框架，将新检测器与一系列冻结的旧版嵌入器（即其旧版链）配对，并采用性能加权采样策略，使检测器在所有早期版本上保持准确性。研究表明，这种耦合虽然对实现向后兼容性是必要的，但会引发一种“嵌入器崩溃”效应，灾难性地降低真负（摘要在此处截断）

    arXiv:2610.22296v1 Announce Type: cross  Abstract: Neural watermarking systems are typically trained as tightly coupled embedder-detector pairs and implicitly assume static deployment. In real-world deployments, embedders and detectors are frequently updated and selectively decoupled for security and versioning purposes, which can break detection in previously embedded content and, more critically, increase false-positive rates on embedders that a given detector is intentionally not intended to recognize.   In this paper the concept of backward compatibility in zero-bit audio watermarking is formalized. A training framework is proposed that pairs a new detector with a sequence of frozen legacy embedders (its legacy chain), using a performance-weighted sampling strategy so that the detector retains accuracy across all earlier versions. This coupling, while necessary for backward compatibility, is shown to induce an embedder-collapse effect that catastrophically degrades the true negativ
    
[^12]: 迈向持续学习：通过子空间微调缓解遗忘

    Towards Continual Learning: Mitigating Forgetting With Subspace Tuning

    [https://arxiv.org/abs/2610.22293](https://arxiv.org/abs/2610.22293)

    提出子空间微调（SST）方法，将权重更新限制在底部奇异子空间、避免改变主奇异方向，从而有效缓解大语言模型微调中的灾难性遗忘问题。

    

    arXiv:2610.22293v1 通告类型：交叉 摘要：大语言模型（LLM）通过微调获得新技能，但这可能会破坏其已有能力：灾难性遗忘仍然是LLM持续学习的核心障碍。近期研究将遗忘与微调移动权重的幅度及方向联系起来。强化学习和在线策略蒸馏等在线策略方法由于避免更新权重矩阵的主奇异方向，因此遗忘较少。相比之下，SFT将更新集中在这些方向上，容易发生灾难性遗忘。在本工作中，我们提出子空间微调，它将每次权重更新限制在底部奇异子空间内，消除沿主方向的变化。SST与优化器无关，并且能够与包括SFT和自蒸馏在内的各种微调方法良好结合。SST结合了紧凑的子空间保护、密集的梯度估计以及一个简单的刷新机制（摘要在此处截断）。

    arXiv:2610.22293v1 Announce Type: cross  Abstract: Large language models (LLMs) acquire new skills through fine-tuning, but it can destroy capabilities they already have: catastrophic forgetting remains a central obstacle to continual learning in LLMs. Recent works connect forgetting to how far, and along which directions, fine-tuning moves the weights. On-policy methods such as reinforcement learning and on-policy distillation avoid updating the principal singular directions of weight matrices and thus forget less. In contrast, SFT concentrates updates along these directions and is susceptible to catastrophic forgetting. In this work, we introduce Subspace Tuning (SST), which restricts each weight update to the bottom singular subspace, eliminating changes along the principal directions. SST is optimizer-agnostic and composes well with fine-tuning recipes including SFT and self-distillation. SST combines tight subspace protection, dense gradient estimation, and a simple refresh mechan
    
[^13]: CAT-催化诚实：面向可迁移大语言模型诚实性的持续对抗训练

    CAT-alyzing Honesty: Continuous Adversarial Training for Transferable LLM Honesty

    [https://arxiv.org/abs/2610.22292](https://arxiv.org/abs/2610.22292)

    CATCH方法通过持续对抗训练，仅用数百个自我报告场景构建的目标，就能让大语言模型习得可泛化、可跨基准迁移的诚实行为，从而在压力情境下显著减少不诚实表现。

    

    大型语言模型（LLM）正日益普及于诚实性至关重要的场景中。然而，尽管经过了Alignment训练，这些模型仍然表现出不诚实行为，例如输出与某些知识相矛盾、或不受上下文中可获取信息支持的内容。受一种最初为鲁棒拒绝有害查询而开发的方法的启发，我们提出了面向一致诚实性的持续对抗训练方法（CATCH）。CATCH在保持模型能力的同时，训练模型朝向诚实的回答、远离不诚实的回答。我们在两种架构（Llama-3.3-70B-Instruct和Qwen3-3.6-27B）上进行了评估。结果表明，仅使用由自我报告场景构建的数百个目标进行训练，CATCH就能让模型习得一种能够稳健泛化并在不同基准测试间迁移的诚实行为。它在MASK基准上一致地减少了压力下的不诚实行为，并在Among Us多智能体游戏中提高了坦白率……（摘要在此处被截断）

    arXiv:2610.22292v1 Announce Type: cross  Abstract: Large language models (LLMs) are becoming prevalent in scenarios where honesty is critical. However, despite alignment training, these models still exhibit dishonest behaviour, such as outputting claims that contradict some knowledge or are not supported by what is accessible in context. We introduce Continuous Adversarial Training for Consistent Honesty (CATCH), inspired by a method originally developed to robustly refuse harmful queries. CATCH trains the model towards an honest answer and away from a dishonest one while maintaining the model's capabilities. We evaluate two architectures (Llama-3.3-70B-Instruct and Qwen3-3.6-27B). We show that trained with only hundreds of targets built from self-reporting scenarios, CATCH learns an honest behaviour that robustly generalizes and transfers across distinct benchmarks. It consistently reduces dishonesty under pressure on MASK, increases confessions in the multi-agent game of Among Us, an
    
[^14]: 多智能体系统中面向时序目标的理性策略学习

    Learning Rational Strategies for Temporal Objectives in Multi-Agent Systems

    [https://arxiv.org/abs/2610.22287](https://arxiv.org/abs/2610.22287)

    该论文针对动态未知的多智能体系统，将纳什均衡等解概念与 PLTL_f 概率时序目标相结合，提出了一种基于带收敛保证的 Nash-Q 学习和有限样本置信界的理性综合方法，能够学习到既满足时序约束又具备策略稳定性的联合策略。

    

    我们研究了具有未知动态的随机多智能体系统的理性综合问题，其中每个智能体都拥有自己的概率时序目标，该目标使用有限轨迹上的概率线性时序逻辑（PLTL_f）进行规约。我们将解概念（如纳什均衡、子博弈完美均衡）作为联合策略组合上的约束，其收益由每个智能体满足其时序目标的概率导出。特别地，我们将面向单个智能体的理性综合形式化为寻找一个策略组合（如果存在的话），使得：(i) 指定智能体的概率时序约束得到满足，且 该策略组合在所选解概念下构成一个稳定解。我们为表格型 Nash-Q 学习给出了条件收敛保证，利用同时有限样本置信界来认证时序约束的满足性，并通过近似最佳响应来评估策略的稳定性。

    arXiv:2610.22287v1 Announce Type: new  Abstract: We study rational synthesis for stochastic multi-agent systems with unknown dynamics, where each agent is equipped with its own probabilistic temporal objective specified using Probabilistic Linear Temporal Logic on Finite Traces ($\mathsf{PLTL}_f}$). We impose solution concepts (e.g., Nash equilibrium, subgame-perfect equilibrium) as constraints on the joint strategy profile, using payoffs derived from each agent's probability of satisfying its temporal objective. In particular, we formulate per-agent rational synthesis as finding a strategy profile -- if one exists -- such that (i) the designated agent's probabilistic temporal constraint is satisfied, and (ii) the profile represents a stable solution under a chosen solution concept. We give conditional convergence guarantees for tabular Nash-Q learning, certify satisfaction with simultaneous finite-sample confidence bounds, and assess strategic stability with approximate best-response 
    
[^15]: 提问更聪明的问题：面向欠规范奖励的定向主动偏好学习

    Asking Smarter Questions: Targeted Active Preference Learning for Underspecified Rewards

    [https://arxiv.org/abs/2610.22285](https://arxiv.org/abs/2610.22285)

    该论文提出将澄清欠规范奖励的负担从人类转移给机器人，利用教学数据中揭示不确定性的信息构建有针对性的先验，通过定向偏好查询让人类以低成本方式弥补监督不足的任务特征。

    

    奖励学习通常假设人类的输入涵盖了任务中所有重要的内容。但在实践中，人类一次只能关注有限的方面，因此他们的教学自然会对任务的某些特征——即某些方面——的捕捉效果优于其他方面。先前的工作已经展示了机器人如何直接从教学数据中识别出哪些特征未得到充分监督。然而，目前的解决方案是让人类重新进行教学，这将负担压在了那些本来就在传达这些特征时遇到困难的人身上。我们提出将这一负担转移给机器人。我们的关键洞察是：那些告诉机器人自己不知道什么的信息，同样可以帮助它提出更好的问题。偏好查询为解决这种不确定性提供了一种低成本的渠道，但前提是机器人知道该将查询定向到何处，并且人类在回答时知道该关注什么。我们通过两种方式践行这一洞察：首先，我们构建了一个有信息量的先验，将偏好查询定向到……（摘要在此处被截断）

    arXiv:2610.22285v1 Announce Type: cross  Abstract: Reward learning typically assumes the human's input covers everything that matters for the task. In practice, humans can only attend to so much at once, so their teaching naturally captures some features---or aspects of the task---better than others. Prior work has shown how the robot can identify which features were poorly supervised directly from its teaching data. But the current solution is to ask the human to try again, putting the burden on someone who already struggled to convey those features. We propose shifting this burden to the robot. Our key insight is that the same information that tells the robot what it doesn't know can help it ask better questions. Preference queries offer a low-effort channel for resolving this uncertainty, but only if the robot knows where to direct them and the human knows what to attend to when answering. We act on this insight in two ways: we build an informed prior that targets preference queries
    
[^16]: 图书馆员智能体：训练大语言模型构建能预判未来查询的长上下文工作区

    Librarian Agents: Training LLMs to Curate Long-Context Workspaces that Anticipate Future Queries

    [https://arxiv.org/abs/2610.22273](https://arxiv.org/abs/2610.22273)

    该论文提出“图书馆员智能体”框架，训练大语言模型在推理之前主动将非结构化语料库重组为可复用的分层文件系统工作区，从而预先组织信息以高效应对未来多样化的查询需求。

    

    对长文档上下文进行推理仍然是大语言模型（LLM）面临的一项挑战。标准的检索方法本质上是被动式的，在推理时会限制智能体进行复杂的跨文档推理，而知识图谱等预计算结构往往因依赖僵化的启发式规则而效果不佳。由此引出一个问题：我们能否让智能体执行“预见性的定向策划”——学习在推理之前主动重组非结构化语料库，以便高效地服务于多样化的未来查询。我们基于现代智能体框架（如 Claude Code）中的一个关键不对称性：虽然 LLM 智能体善于在现有文件系统中导航以完成即时任务，但它们很少被训练去构建可供未来复用的工作区。为了实现这一能力，我们提出了图书馆员智能体，这是一个框架，其中构建者智能体将原始文档转换为由合成文件和引用组成的分层文件系统……

    arXiv:2610.22273v1 Announce Type: cross  Abstract: Reasoning over long document contexts remains a challenge for large language models (LLMs). Standard retrieval methods are inherently reactive, bottlenecking agents with complex, cross-document reasoning at inference time, while precomputed structures like knowledge graphs often fall short because they rely on rigid heuristics. Instead, can we allow agents to perform anticipatory directed curation: learning to proactively reorganize unstructured corpora prior to inference to efficiently serve diverse future queries. We build on a key asymmetry in modern agentic harnesses (e.g., Claude Code): while LLM agents are adept at navigating existing filesystems to solve an immediate task, they are rarely trained to construct reusable workspaces for future use. To instantiate this capability, we introduce Librarian Agents, a framework in which a builder agent transforms raw documents into a hierarchical filesystem of synthetic files and referenc
    
[^17]: 策略梯度估计中的基线与得分中心化

    Baselines and Score Centering in Policy Gradient Estimation

    [https://arxiv.org/abs/2610.22263](https://arxiv.org/abs/2610.22263)

    本文提出在正则化重要性采样的离策略强化学习中，应以奖励与重要性加权得分之间的协方差作为估计对象，从而恢复策略梯度对奖励常数平移的不变性。

    

    强化学习已成为对大型语言模型进行后训练的一种广泛使用的方法。在典型实现中，学习是离策略的：由于训练与推理之间的不匹配，或异步训练中的策略滞后，生成响应的策略与正在被训练的策略并不相同。这种差异通常通过重要性采样来解决，其比率在实践中通常通过截断、掩码或裁剪等方式进行正则化。在正则化重要性采样下，重要性加权的策略梯度不再对奖励的常数平移保持不变。通过采用奖励与重要性加权得分之间的协方差作为估计对象，可以恢复这种不变性；当比率未被正则化时，该估计对象与策略梯度相一致。本文针对序列级和词元级两种重要性采样发展了这一协方差视角。协方差视角揭示了……（摘要原文在此处截断）

    arXiv:2610.22263v1 Announce Type: new  Abstract: Reinforcement learning has become a widely used approach to post-training large language models. In a typical implementation, learning is off-policy: The policy generating the responses differs from the policy being trained because of training-inference mismatch or policy staleness in asynchronous training. This discrepancy is commonly addressed by importance sampling, whose ratio is usually regularized in practice through truncation, masking, or clipping. Under regularized importance sampling, the importance-weighted policy gradient is no longer invariant to constant shifts in the reward. This invariance is recovered by using the covariance between the reward and the importance-weighted score as the estimand, which coincides with the policy gradient when the ratio is not regularized. This note develops this covariance perspective for both sequence-level and token-level importance sampling. The covariance perspective reveals the roles of
    
[^18]: 拓扑启发的风场超分辨率微调方法

    Topology-Inspired Fine-Tuning for Wind-Field Super-Resolution

    [https://arxiv.org/abs/2610.22262](https://arxiv.org/abs/2610.22262)

    该论文提出利用可微的拓扑启发辅助损失（标量速度、梯度幅值、软高速水平集、局部极大值）及固定持续对监督来微调预训练CNN，从而在风场超分辨率中弥合逐点重建损失与极值、连通区域等拓扑结构保真之间的矛盾。

    

    科学超分辨率（SR）模型通常使用逐点重建损失进行优化，然而科学解释往往依赖于极值、连通区域和多尺度标量结构。我们通过预训练的CNN和GAN风场超分辨率模型来刻画这一矛盾。逐点度量、瓶颈持续图（PD）距离和合并树（MT）距离更偏向CNN，而二阶Wasserstein PD距离则更偏向更锐利的GAN，这表明即使是同一拓扑描述符上的不同距离度量，也会奖励不同的特性。受这种分歧的启发，我们利用可微的拓扑启发辅助损失对预训练CNN进行微调，这些损失作用于标量速度、梯度幅值、软高速水平集和局部极大值，并采用固定对监督，即在地面真值持续对端点处（通过Topology ToolKit离线定位）约束预测值和持续性。一项针对16个微调……（原文摘要在此处不完整）

    arXiv:2610.22262v1 Announce Type: new  Abstract: Scientific super-resolution (SR) models are typically optimized with pointwise reconstruction losses, yet scientific interpretation often hinges on extrema, connected regions, and multiscale scalar structure. We characterize this tension with pretrained CNN and GAN wind-field SR models. Pointwise metrics, bottleneck persistence-diagram (PD) distance, and merge-tree (MT) distance favor the CNN, whereas order-2 Wasserstein PD distances favor the sharper GAN, so even different distances on the same topological descriptor reward different properties. Motivated by this disagreement, we fine-tune the pretrained CNN with differentiable topology-inspired auxiliary losses on scalar speed, gradient magnitude, soft high-speed level sets, and local maxima, and with fixed-pair supervision, which constrains predicted values and persistence at ground-truth persistence-pair endpoints located offline with the Topology ToolKit. A loss study of 16 fine-tun
    
[^19]: BitSparse：面向 ReLU 前馈网络的紧凑激活存储

    BitSparse: Compact Activation Storage for ReLU Feed-Forward Networks

    [https://arxiv.org/abs/2610.22256](https://arxiv.org/abs/2610.22256)

    BitSparse 利用稀疏 ReLU 激活特性实现 GPU 上的无损激活压缩，通过融合进自动求导的自定义 Triton 内核，将 Modded-NanoGPT 和 Nemotron-H 的每 token 显存占用分别降低 31% 和 30%，且开销不超过 1%。

    

    降低大语言模型（LLM）训练中的内存使用是一个持续的挑战。激活检查点通过重计算来节省内存，而有损压缩则会改变存储的激活值。稀疏 ReLU 和 ReLU² 激活的重新出现带来了第三种机会：对激活缓存本身进行无损压缩。我们提出了 BitSparse，一种面向稀疏激活的 GPU 常驻无损压缩格式。压缩与重建运行在融合进自动求导函数的自定义 Triton 内核中，在缩减激活缓存的同时为反向传播精确恢复激活值。该方法支持 Bfloat16（BF16）和 FP8 激活。可选的符号位打包进一步将压缩率提升至接近理论极限。我们分析了其压缩比，并与理论最优值进行了比较。BitSparse 分别将 Modded-NanoGPT 和 Nemotron-H 的每 token 显存占用降低了 31% 和 30%，且开销不超过 1%。

    arXiv:2610.22256v1 Announce Type: new  Abstract: Reducing memory use in large language model (LLM) training is an ongoing challenge. Activation checkpointing saves memory by recomputation, while lossy compression changes the stored activations. The reemergence of sparse $\relu$ and $\relu^2$ activations creates a third opportunity: losslessly compressing the activation cache itself. We present BitSparse, a GPU-resident lossless compression format for sparse activations. Compression and reconstruction run in custom Triton kernels fused into an autograd function, shrinking the activation cache while recovering it exactly for the backward pass. Bfloat16 (BF16) and FP8 activations are supported. Optional sign-bit packing further increases compression up to close to the theoretical limit. We analyze its compression ratio and compare it to the theoretical optimum. BitSparse lowers per-token VRAM by 31\% and 30\% Modded-NanoGPT and Nemotron-H respectively, with overheads of no more than 1\% a
    
[^20]: 一次探索，多次评估：随机线性老虎机中的多策略评估

    Explore Once, Evaluate Many: Multiple Policy Evaluation for Stochastic Linear Bandits

    [https://arxiv.org/abs/2610.22252](https://arxiv.org/abs/2610.22252)

    提出了 PRISMA 算法，通过一次无奖励探索收集数据并结合正则化信息共享与混合分配，能够在随机线性上下文老虎机中以样本复杂度仅随策略数 $K$ 对数增长的代价同时高精度评估 $K$ 个目标策略的价值。

    

    我们研究如何在具有线性奖励结构的随机上下文老虎机中同时评估 $K$ 个目标策略。在给定已知的 $d$ 维特征映射并可以无奖励方式访问策略查询的条件下，我们提出了 PRISMA（通过正则化信息共享与混合分配进行策略评估）算法，该算法能够以至少 $1-\delta$ 的概率将每个目标策略的价值估计到精度 $\varepsilon$。该算法使用 $\widetilde{\mathcal{O}}\!\left( \frac{(L^2S^2+\beta^2)(1\vee\rho^\star)}{\varepsilon^2} \cdot \log(K) \right)$ 次上下文抽取和奖励观测，其中 $\rho^\star\le d$ 是一个与实例相关的量，用于衡量设计共享采样分布的难度。其中，$L$、$S$ 和 $\beta$ 分别约束特征范数、奖励参数范数和奖励噪声，被省略的因子是关于 $K$、$d$ 和 $1/\delta$ 的对数项。我们的方法结合了无奖励规划、正则化……（摘要在此截断）

    arXiv:2610.22252v1 Announce Type: new  Abstract: We study how to evaluate $K$ target policies simultaneously in stochastic contextual bandits with linear reward structure. Given a known $d$-dimensional feature map and reward-free access to policy queries, we propose PRISMA (Policy evaluation through Regularized Information Sharing and Mixture Allocation), an algorithm that estimates every target policy's value to accuracy $\varepsilon$ with probability at least $1-\delta$. The algorithm uses $\widetilde{\mathcal{O}}\!\left( \frac{(L^2S^2+\beta^2)(1\vee\rho^\star)}{\varepsilon^2} \cdot \log(K) \right)$ context draws and reward observations, where $\rho^\star\le d$ is an instance-dependent measure of the difficulty of designing a shared sampling distribution. Here, $L$, $S$, and $\beta$ bound the feature norm, reward parameter norm, and reward noise, respectively, and the suppressed factors are logarithmic in $K$, $d$, and $1/\delta$. Our method combines reward-free planning, a regulariz
    
[^21]: 语言是智能体安全防护的安全暴露面

    Language Is a Security Surface for Agent Guards

    [https://arxiv.org/abs/2610.22244](https://arxiv.org/abs/2610.22244)

    论文构建了 LinguaGuardBench 基准，发现仅将用户指令翻译成德语、法语等非英语语言就会使大多数开源、小型和专用智能体安全防护器改变判定，揭示了语言本身是智能体安全防护的一个重要攻击面。

    

    步骤级防护器在每次工具调用之前，根据用户请求和当前的行动轨迹，判断智能体的下一步动作是否安全。用户会使用多种语言编写这些请求，而防护器的提示词以及用于训练和评估专用防护器的基准测试却都是英文。我们提出疑问：防护器的判定是否取决于请求所使用的语言。我们基于 TS-Bench AgentDojo-Traj 的 1,220 个带标签步骤构建了 LinguaGuardBench：我们仅将用户指令翻译成德语、法语、土耳其语、斯瓦希里语和阿姆哈拉语，并保持每一次工具调用、观察结果和标签逐字节完全一致，因此判定中的任何变化都可归因于语言本身。研究以六种英文改写和一个空白符安慰剂作为对照组，并对托管防护器运行三次以测量其噪声水平。在 19 个防护器中——从前沿 API 模型到 4B 开源模型，以及四个专用的智能体防护器——大多数开源、小型和专用防护器的判定都会因语言改变而发生变化（原文摘要在此处截断）。

    arXiv:2610.22244v1 Announce Type: cross  Abstract: Step-level guards decide, before each tool call, whether an agent's next action is safe given the user's request and the trajectory so far. Users write those requests in many languages, while guard prompts and the benchmarks on which purpose-built guards are trained and evaluated are English. We ask whether a guard's verdict depends on the language of the request. We build LinguaGuardBench from the 1,220 labelled steps of TS-Bench AgentDojo-Traj: we translate only the user instruction into German, French, Turkish, Swahili and Amharic, and keep every tool call, observation and label byte-identical, so any change in a verdict is attributable to language alone. Six English rewordings and a whitespace placebo serve as controls, and managed guards are run three times to measure their noise. Across 19 guards, from frontier APIs to 4B open models and four purpose-built agent guards, most open, small and purpose-built guards change verdicts un
    
[^22]: 智能体安全评估中的捷径敏感性

    Shortcut Sensitivity in Agent-Safety Evaluations

    [https://arxiv.org/abs/2610.22243](https://arxiv.org/abs/2610.22243)

    该论文提出一个三层级捷径审计框架，揭示智能体安全防护评估数据中存在的捷径线索（如命令长度、领域混叠、加权方式）会显著夸大防护机制的表观性能（F1可从约0.90跌至0.50–0.72），因此安全评估必须控制这些捷径才能得出可靠结论。

    

    AI智能体通过工具来执行操作。部分操作是敏感的，因此监督这些操作的防护机制本身必须得到可靠的评估。然而，用于评估防护机制的数据可能包含某些线索，使得标签无需测量预期的安全构念即可被预测，而这些线索也可能影响防护机制的决策。我们提出了一个三层级的捷径审计框架，并对已完成的LinuxArena轨迹以及两个由TS-Bench派生的步骤数据集进行了审计。我们区分了标签可预测性、观察性防护关联以及配对决策敏感性。在LinuxArena中，较短的平均shell命令能强烈预测轨迹标签。在AgentDojo-Traj中，精确工具查找的F1值从混合领域折数下的.901降至领域留出下的.495。在AgentHarm-Traj中，已发布的TS-Guard的F1值在行加权下为.902，而在等交互加权下降至.719，同时平衡准确率反而上升。随后，我们测试了最强的安全行标记关联……（摘要原文在此处截断）

    arXiv:2610.22243v1 Announce Type: new  Abstract: AI agents take actions through tools. Some actions are sensitive, so guards that supervise them must themselves be evaluated reliably. Yet the data used to evaluate a guard can include cues that make labels predictable without measuring the intended safety construct, and those cues can also influence guard decisions. We introduce a three-level shortcut-auditing framework and audit completed LinuxArena trajectories and two TS-Bench-derived step datasets. We distinguish label predictability, observational guard association, and paired decision sensitivity. In LinuxArena, shorter average shell commands strongly predict trajectory labels. In AgentDojo-Traj, exact-tool lookup falls from F_1=.901 with mixed-domain folds to .495 under domain holdout. In AgentHarm-Traj, released TS-Guard F_1 falls from .902 under row weighting to .719 with equal interaction weighting, while balanced accuracy rises. We then test the strongest safe-row marker asso
    
[^23]: SurvICL：一个用于上下文生存分析的基础模型

    SurvICL: a Foundation Model for in-context Survival Analysis

    [https://arxiv.org/abs/2610.22242](https://arxiv.org/abs/2610.22242)

    SurvICL是一个通过新型比例几率先验生成合成数据并以分位数回归损失预训练的上下文生存分析基础模型，首次超越经典Cox模型，并配套推出了迄今最大的103数据集生存分析基准。

    

    表格基础模型（TFM）目前在分类和回归任务上已超越梯度提升树。然而在生存分析领域——即从删失数据中预测事件发生时间，且通常基于小规模临床队列——我们发现现有TFM均无法击败拥有半个世纪历史的Cox模型。为解决这一问题，我们提出了SurvICL，一个能够以上下文学习方式预测生存曲线的表格基础模型。SurvICL在纯合成数据集上进行预训练，这些数据由我们提出的一种新型灵活比例几率先验生成，该先验在消融研究中表现优于其他类型的先验。与大多数其他生存分析模型不同，SurvICL采用简单的分位数回归损失进行训练，这之所以可行，是因为其合成预训练数据生成器能够提供未删失的事件时间。此外，我们推出了迄今为止规模最大的生存分析基准，涵盖103个数据集。在从所有设计决策中保留的63个数据集上，SurvICL超越了20种基于树的方法、深度学习方法和基础模型。

    arXiv:2610.22242v1 Announce Type: cross  Abstract: Tabular foundation models (TFMs) now outperform gradient-boosted trees in classification and regression. Yet in survival analysis, which predicts when an event occurs from censored data, often on small clinical cohorts, we find that no existing TFM beats the half-century-old Cox model. We address this issue by introducing SurvICL, a TFM that predicts survival curves in-context. SurvICL is pretrained on purely synthetic datasets generated from a novel flexible proportional-odds prior that outperforms other prior types in our ablation study. Unlike most other survival models, SurvICL is trained with a plain quantile regression loss, which is possible because its synthetic pre-training data generator can provide uncensored event times. Additionally, we introduce the largest survival benchmark to date spanning 103 datasets. On the 63 datasets held out from all design decisions, SurvICL outperforms 20 tree-based, deep, and foundation-model 
    
[^24]: 面向并行掩码扩散的查询高效拒绝采样

    Query-Efficient Rejection Sampling for Parallel Masked Diffusion

    [https://arxiv.org/abs/2610.22238](https://arxiv.org/abs/2610.22238)

    提出单增量拒绝采样器（SIRS），每次仅需一次查询即可估计并行掩码扩散模型拒绝采样中的对数接受概率，并结合伯努利工厂与Rao-Blackwell化显著降低查询成本并减少截断偏差。

    

    掩码扩散模型（MDMs）通过并行解掩码多个token来加速生成，但独立采样这些token忽略了它们之间的相关性，从而使输出产生偏差。Anari等人（2026）利用拒绝采样纠正了这一偏差，将对数接受概率分解为token级项之和，但评估时需要对每个被解掩码的token进行一次分数查询。我们提出了单增量拒绝采样器，它仅以一次查询的代价，从单个随机采样的增量中提供对数接受概率的估计。随后，SIRS通过改编自Chen等人（2026）的伯努利工厂，将泊松数量的截断估计转换为接受/拒绝决策。由于每次查询都会返回块中所有剩余token的条件分布，我们进一步以零额外代价对这些token取平均来对每个增量进行Rao-Blackwell化，从而减少截断误差。我们证明了KL散度……（摘要原文在此处截断）

    arXiv:2610.22238v1 Announce Type: new  Abstract: Masked diffusion models (MDMs) accelerate generation by unmasking multiple tokens in parallel, but sampling these tokens independently ignores their correlations and biases the output. Anari et al. (2026) corrected this bias with rejection sampling, decomposing the log acceptance probability into a sum of token-level terms, but the evaluation requires one score query per unmasked token. We propose the Single-Increment Rejection Sampler (SIRS), which provides an estimate of the log acceptance probability from a single randomly sampled increment at the cost of one query. SIRS then converts a Poisson number of clipped estimates into accept/reject decisions through a Bernoulli factory adapted from Chen et al. (2026). Since each query returns the conditionals of all remaining tokens in the block, we further Rao-Blackwellize each increment by averaging over these tokens at no extra cost, thereby reducing clipping. We show that the KL-divergenc
    
[^25]: 免训练的在策略蒸馏

    Training-Free On-Policy Distillation

    [https://arxiv.org/abs/2610.22236](https://arxiv.org/abs/2610.22236)

    本文提出免训练的在策略蒸馏方法，当多个专家模型共享同一基础模型时，仅需通过几百个采样样本的一次前向传播、以最小二乘方式逐层融合，即可在无需任何梯度训练的情况下将多个专家的能力聚合到一个模型中，且平均准确率超过任何单一专家模型。

    

    在策略蒸馏需要为每个教师模型进行采样、教师前向传播和梯度更新，才能将专家模型整合到一个学生模型中。我们证明这种整合其实不需要训练：仅通过前向传播，就可以将分别微调的多个专家模型的能力聚合到一个模型中，该模型平均而言比任何一个单独的专家模型都更加准确。当专家模型共享同一个基础模型时，蒸馏存在一个精确的逐层解：融合模型的每个线性层必须在专家模型所接收的输入上复现该专家的输出（同时读取其自身的答案），这是一个最小二乘问题，其解只需要这些输入的二阶矩。对几百个采样样本进行一次前向传播即可获得这些二阶矩。我们将这一方法称为免训练在策略蒸馏。相比求解器，采样样本中所包含的内容更为重要：在重复循环中被截断的回答在数学专家模型的采样中只占少数。

    arXiv:2610.22236v1 Announce Type: cross  Abstract: On-policy distillation consolidates specialist models into one student at the price of rollouts, teacher passes and gradient steps for every teacher. We show that the consolidation does not need training: with forward passes only, the capabilities of separately tuned specialists can be aggregated into one model that is, on average, more accurate than either specialist alone. When specialists share a base model, distillation has an exact layer-wise solution: each linear layer of the fused model must reproduce every specialist's output on the inputs that specialist receives while reading its own answers, a least-squares problem whose solution needs only the second moments of those inputs. One forward pass over a few hundred rollouts provides them. We call this training-free on-policy distillation (\method). What the rollouts contain matters more than the solver: answers truncated in repetition loops are few among a mathematics specialist
    
[^26]: OS-Helix：在原生计算机使用智能体中统一GUI交互与代码执行

    OS-Helix: Unifying GUI Interaction and Code Execution in Native Computer-Using Agents

    [https://arxiv.org/abs/2610.22234](https://arxiv.org/abs/2610.22234)

    OS-Helix是一个端到端训练的多模态计算机使用智能体，通过单一动作空间统一GUI操作与代码执行，并借助应用-工件组合方法和解耦的指令/评估器生成流水线来构建高质量的混合训练数据。

    

    真实的桌面自动化既不是纯视觉的，也不是纯程序化的。智能体必须既能理解面向用户的界面，又能高效地操作文件、脚本和系统状态。然而，原生的GUI/代码协调无法通过提示层面的工具组合来获得。在我们的初步实验中，简单地添加一个代码工具反而会使GUI智能体的性能下降，这表明GUI/代码之间的路由必须通过在高质量混合数据上的训练来内化。我们提出了OS-Helix，这是一个端到端训练的多模态计算机使用智能体，它将GUI操作和代码执行统一在同一个动作空间中。为了构建所需的监督数据，我们引入了应用-工件组合方法，将桌面应用与真实文件工件配对以实例化可执行的Ubuntu环境状态，并设计了解耦的指令/评估器生成流水线，以产生基于真实环境的任务，其最终状态无论采用何种解决方式都可以被验证。

    arXiv:2610.22234v1 Announce Type: cross  Abstract: Real desktop automation is neither purely visual nor purely programmatic. Agents must understand user-facing interfaces while efficiently manipulating files, scripts, and system state. However, native GUI/code coordination cannot be obtained by prompt-level tool composition. In our preliminary experiments, naively adding a code tool degrades a GUI agent, revealing that GUI/code routing must be internalized through training on high-quality hybrid data. We present OS-Helix, an end-to-end trained multimodal computer-using agent that unifies GUI actions and code execution in one action space. To construct the required supervision, we introduce application-artifact composition, pairing desktop applications with authentic file artifacts to instantiate executable Ubuntu states, and a decoupled instruction/evaluator generation pipeline that produces environment-grounded tasks whose final states can be verified regardless of the solution modali
    
[^27]: 面向硬件多样性的架构设计：Meta在AMD GPU上实现高性能AI训练的端到端协同设计

    Architecting for Hardware Diversity: Meta's End-to-End Co-Design for High-Performance AI Training on AMD GPUs

    [https://arxiv.org/abs/2610.22230](https://arxiv.org/abs/2610.22230)

    Meta提出硬件-模型协同设计、分析性能模型驱动的零预热内核选择和智能体工作流三大可复用原则，成功实现了推荐模型训练在AMD MI350X GPU上相对于NVIDIA B200的高性能优化移植。

    

    Meta部署了一个由NVIDIA GPU、AMD GPU和自研MTIA加速器组成的异构集群，以降低成本并缓解供应商锁定问题。然而，长期以来针对NVIDIA的深度优化阻碍了性能向其他硬件的可移植性。本文以AMD Instinct MI350X与NVIDIA B200在推荐模型训练上的对比为案例研究，介绍了我们优化其他加速器的方法。我们提出三个可复用的原则。第一，硬件-模型协同设计至关重要：与其直接将模型从NVIDIA GPU移植过来，不如修改模型结构和软件方案，以充分发挥加速器特有的优势（例如MI350X相对于B200拥有1.6倍的HBM容量）。第二，我们的分析性能模型能够在生产模型中跨不同数据类型和矩阵形状实现零预热的最佳内核选择，极大地简化了调优工作。第三，日益成熟的智能体工作流……（原文摘要在此处截断）

    arXiv:2610.22230v1 Announce Type: cross  Abstract: Meta deploys a heterogeneous fleet comprising NVIDIA GPUs, AMD GPUs, and proprietary MTIA accelerators to reduce costs and mitigate vendor lock-in. However, the extensive history of NVIDIA-specific optimizations hinders performance portability to other hardware. This paper presents our methods for optimizing alternative accelerators, using AMD Instinct MI350X versus NVIDIA B200 for recommendation model training as a case study. We propose three reusable principles. First, hardware-model co-design is essential: rather than directly porting models from NVIDIA GPUs, revising model structures and software recipes allows full exploitation of accelerator-specific strengths (e.g., MI350X's 1.6x HBM capacity relative to B200). Second, our analytical performance model enables zero-warmup optimal kernel selection across diverse datatypes and matrix shapes in production models, drastically simplifying tuning. Third, increasingly mature agentic wo
    
[^28]: 视觉能从音频中学到多少物理？探查与蒸馏阻尼结构

    How Much Physics Can Vision Learn from Audio? Probing and Distilling Damping Structure

    [https://arxiv.org/abs/2610.22221](https://arxiv.org/abs/2610.22221)

    该研究发现音频嵌入能远优于视觉地捕捉物体阻尼结构（品质因数Q排序上ρ=0.73对比0.28），并提出将冻结音频教师模型蒸馏到视觉模型，以让视觉从音频中学习阻尼物理特性。

    

    物理理解对决策制定至关重要，然而视觉模型难以推断诸如阻尼这类决定材料行为的属性。音频则直接编码了这些属性，因为物体受撞击后的声学响应通过可测量的衰减揭示其阻尼。我们探查了冻结编码器对品质因数 $Q$ 的表征能力——$Q$ 是一个无量纲量，衡量物体在余音消散之前完成的振荡次数——并使用Spearman秩相关系数 $\rho$ 报告各嵌入按 $Q$ 对撞击进行排序的表现，其中 $\rho=1$ 表示完美排序，$\rho=0$ 表示无关联。音频嵌入达到 $\rho=0.73$，且在训练中未见过的材料上仍保留了大部分优势；而由CLIP编码的撞击瞬间单帧视频仅达到 $\rho=0.28$。在确定了信号所在之处后，我们进一步探究视觉能弥补多大差距：我们将冻结的音频教师模型蒸馏到一个（摘要在此处截断）

    arXiv:2610.22221v1 Announce Type: cross  Abstract: Physical understanding is crucial for decision making, yet vision models struggle to infer properties such as damping that govern how materials behave. Audio encodes such properties directly, because an object's acoustic response to impact reveals its damping through measurable decay. We probe frozen encoders for the quality factor $Q$, a dimensionless measure of how many oscillations an object completes before its ring dies away, and report how well each embedding orders impacts by $Q$ using the Spearman rank correlation $\rho$, for which $\rho=1$ is a perfect ordering and $\rho=0$ an unrelated one. Audio embeddings reach $\rho=0.73$ and retain most of that advantage on materials withheld from training, while a single video frame taken at the moment of impact, encoded by CLIP, supports only $\rho=0.28$. Having established where the signal lives, we ask how much of that gap vision can recover. We distill a frozen audio teacher into a f
    
[^29]: FutureLift：将轨迹提议提升为双向未来图以用于运动预测

    FutureLift: Lifting Trajectory Proposals into Bidirectional Future Graphs for Motion Forecasting

    [https://arxiv.org/abs/2610.22215](https://arxiv.org/abs/2610.22215)

    FutureLift提出了一种与主干网络无关的轨迹提议改进方法，将多模态提议表示为带有双向时间边的异构未来图，使后续证据能够在单次遍历中修正较早的片段，从而提升自动驾驶运动预测的准确性。

    

    准确的运动预测对于安全的自动驾驶至关重要，然而即使是强大的预测器也会在其多模态提议中留下残差误差。现有的改进器要么压缩完整的提议，要么按顺序更新其片段，这使得完整的提议未来无法作为显式的关联上下文用于修正。我们提出了FutureLift，一种与主干网络无关的改进器，它将每个多模态提议表示为一个异构未来图，其中具有空间锚定的时间节点，这些节点连接到智能体历史、地图元素、替代模态，以及在联合预测场景下连接到其他智能体的未来轨迹。双向时间边允许后续证据在单次遍历中修正较早的片段，更新后的节点则预测残差坐标和置信度修正。FutureLift仅需要提议、分数和场景上下文作为输入，为结构不同的预测主干网络提供了一个通用的改进接口。在...

    arXiv:2610.22215v1 Announce Type: cross  Abstract: Accurate motion prediction is critical for safe autonomous driving, yet even strong forecasters leave residual errors in their multimodal proposals. Existing refiners compress complete proposals or update their segments sequentially, preventing the full proposed future from serving as explicit relational context for correction. We introduce FutureLift, a backbone-agnostic refiner that represents each multimodal proposal as a heterogeneous future graph, with spatially grounded temporal nodes connected to agent histories, map elements, alternative modes, and, for joint prediction, other agents future trajectories. Bidirectional temporal edges allow later evidence to revise earlier segments in a single pass, and the updated nodes predict residual coordinate and confidence corrections. Requiring only proposals, scores, and scene context, FutureLift provides a common refinement interface for structurally different forecasting backbones. On 
    
[^30]: 非凸学习中随机块坐标下降的泛化保证

    Generalization Guarantees of Randomized Block Coordinate Descent for Nonconvex Learning

    [https://arxiv.org/abs/2610.22213](https://arxiv.org/abs/2610.22213)

    本文为非凸学习中的随机块坐标下降算法建立了高概率泛化保证，并通过早停策略量化了优化与泛化的平衡，实现了总体风险梯度的最优收敛速率。

    

    随机块坐标下降是众多大规模机器学习问题中广泛使用的一种优化算法。现有的RBCD理论分析主要从优化角度关注其收敛行为，而其在非凸问题中的泛化行为仍然知之甚少。本文为具有非凸目标的RBCD建立了高概率泛化保证。在一般的非凸设置下，我们推导了经验风险和总体风险梯度的界，且仅需标准的光滑性假设。在采用多项式衰减步长的情况下，我们通过调整迭代次数来平衡优化与泛化，从而量化了早停如何实现总体风险梯度的最优收敛速率。在梯度主导的设置下，我们给出了关于函数次优性的界，该界几乎匹配凸优化中乐观的收敛速率。

    arXiv:2610.22213v1 Announce Type: new  Abstract: Randomized block coordinate descent (RBCD) is a popular optimization algorithm for many large-scale machine learning problems. The existing theoretical analysis of RBCD mainly focuses on its convergence behavior from an optimization perspective, while its generalization behavior remains less understood for nonconvex problems. In this paper, we establish high-probability generalization guarantees for RBCD with nonconvex objectives. In the general nonconvex setting, we derive bounds on the gradients of both empirical and population risks, requiring only standard smoothness assumptions. With polynomially decaying step sizes, we tune the iteration number to balance optimization and generalization, thereby quantifying how early stopping leads to the optimal convergence rate for the population gradients. In the gradient-dominated setting, we provide bounds in terms of functional suboptimality, which nearly match the optimistic rates in convex 
    
[^31]: 超越下游问答：重新思考我们如何评估智能体记忆

    Beyond Downstream QA: Rethinking How We Evaluate Agent Memory

    [https://arxiv.org/abs/2610.22209](https://arxiv.org/abs/2610.22209)

    本文指出当前以下游问答准确率评估智能体记忆的范式会将记忆正确性与检索、生成质量混淆，并通过范式反转构建了FEB-30基准，可在无人工标注的情况下对智能体记忆存储本身进行确定性的直接评估。

    

    智能体记忆基准通过评估智能体的响应是否反映正确的知识，以下游问答准确率作为记忆质量的代理指标。我们证明这种评估范式是不完整的：它将记忆正确性与检索和生成质量混为一谈，既产生假阳性（正确的响应掩盖了已损坏的记忆），也产生假阴性（错误的响应将检索失败错误地归咎于正确的记忆）。我们提出了FEB-30，这是一个包含30条事实演化轨迹的基准，涵盖6个诊断类别和3个难度级别，通过范式反转的方法构建：文档从已知的事实轨迹生成，从而在无需人工标注的情况下获得确定性的真实标准（ground truth）。每条轨迹定义了一个事实应如何在多作者制品（文档、电子邮件、对话）中演化，从而能够在每次记忆摄取步骤后直接检查记忆存储。

    arXiv:2610.22209v1 Announce Type: cross  Abstract: Agent memory benchmarks evaluate whether an agent's responses reflect correct knowledge, measuring downstream question-answering accuracy as a proxy for memory quality. We show this evaluation paradigm is incomplete: it conflates memory correctness with retrieval and generation quality, producing both false positives where correct responses mask broken memory and false negatives where incorrect responses blame correct memory for retrieval failures. We introduce FEB-30, a benchmark of 30 fact-evolution trajectories spanning 6 diagnostic categories and 3 difficulty levels, constructed via paradigm inversion: documents are generated from known fact trajectories, yielding deterministic ground truth without human annotation. Each trajectory defines how a fact should evolve across multi-author artifacts (documents, emails, conversations), enabling direct inspection of the memory store after each ingestion step. Applying FEB-30 to a represent
    
[^32]: 对噪声样本的注意力：指数Hopfield网络中的泛化与记忆

    Attention over noisy examples: generalization and memorization in the exponential Hopfield network

    [https://arxiv.org/abs/2610.22191](https://arxiv.org/abs/2610.22191)

    该论文将含噪样本的Hebbian学习协议移植到等价于softmax注意力机制的指数Hopfield网络，并通过信噪比分析闭式推导出对未见原型的检索重叠公式m=erf(1/√(2ρ_eff))，其中数据集熵被由softmax权重构成的注意力熵所取代，从而刻画了注意力机制从噪声样本中实现泛化与记忆的条件。

    

    在基于样本的Hebbian学习中，一个具有N个神经元的Hopfield网络并非被直接提供其需要检索的K个原型，而是被提供每个原型的M个质量为r的模糊样本；网络能否推断出原型，取决于负载α=K/N（原型之间的干扰程度）和数据集熵ρ=(1-r²)/(Mr²)（样本对原型所透露信息的多寡）。我们将这一协议移植到指数Hopfield模型上，该模型的零温度更新等价于一步softmax注意力，其中未标记的样本作为键。通过信噪比分析，我们以闭式形式给出了输出与未见过的原型之间的重叠：m^(1)=erf(1/√(2ρ_eff))，即小负载下成对模型的多数规则定律，只是其中的ρ被一个注意力熵ρ_eff所取代——该熵由softmax赋予正确类别的权重以及它将权重分散到的样本数量构成。由此得出两个结果。首先，在一个窗口内……

    arXiv:2610.22191v1 Announce Type: cross  Abstract: In Hebbian learning from examples a Hopfield network of $N$ neurons is not supplied with the $K$ archetypes it has to retrieve but with $M$ blurred examples of each, of quality $r$, and whether it can infer them depends on the load $\alpha=K/N$ (interference among them) and on the dataset entropy $\rho=(1-r^2)/(Mr^2)$ (how little the examples say about them). We transfer this protocol to the exponential Hopfield model, whose zero-temperature update is a step of softmax attention with the unlabelled examples as keys. A signal-to-noise analysis gives in closed form the overlap between the output and an unseen archetype, $m^{(1)}={\rm erf}(1/\sqrt{2\rho_{\rm eff}})$: the majority-rule law of the pairwise model at small load, with $\rho$ replaced by an attention entropy $\rho_{\rm eff}$ built from the weight the softmax gives to the correct class and the number of examples it spreads it over. Two consequences follow. First, in a window of 
    
[^33]: 全身灵巧操作

    Whole-Body Dexterity

    [https://arxiv.org/abs/2610.22184](https://arxiv.org/abs/2610.22184)

    该论文提出全身灵巧操作（WBD）框架，让机器人利用身体和四肢作为接触面来抓取和搬运大体积物体，其蒸馏策略可零样本迁移到真实的宇树G1人形机器人上并完成长时程搬运任务。

    

    当前人形机器人的灵巧操作仍主要局限于末端执行器和简单的抓取动作，远不及人类能够调动整个身体、将笨重物体靠着手臂和躯干进行支撑和滑动的能力。我们提出了全身灵巧操作，这是一个利用机器人身体和四肢作为接触面来抓取和操作物体的通用框架。我们的仿真到现实流程首先生成一个可行的全身抓取数据集，然后通过从数据集中采样初始和目标抓取姿态，在通用的“任意到任意”重定位目标上训练强化学习专家策略，并将这些专家策略蒸馏为一个利用本体感觉和物体几何信息的单一策略。我们证明了该蒸馏策略可以零样本迁移到真实的宇树 Unitree G1 人形机器人上，成功将篮球和纸箱重定位到躯干周围的区域，并完成一项长时程任务——将箱子夹在一侧手臂下并从货架上搬运它。

    arXiv:2610.22184v1 Announce Type: cross  Abstract: Current humanoid dexterous manipulation remains largely confined to end-effectors and simple grasps, falling short of humans' ability to engage the entire body by bracing and sliding bulky objects against the arms and torso. We introduce Whole-Body Dexterity (WBD), a general framework that uses the robot's body and appendages as contact surfaces to grasp and manipulate objects. Our sim-to-real pipeline first generates a dataset of feasible whole-body grasps, then trains reinforcement-learning experts on a general any-to-any repositioning objective by sampling initial and goal grasps from the dataset, and distills these experts into a single policy that uses proprioception and object geometry. We show that the distilled policy transfers zero-shot to a real Unitree G1, where it repositions a basketball and a cardboard box to regions around the torso and completes a long-horizon task that stows a box under one arm, carries it from a shelf
    
[^34]: LCT：面向训练的GPU友好执行无损压缩张量

    LCT: Losslessly Compressed Tensors with GPU friendly execution for Training

    [https://arxiv.org/abs/2610.22179](https://arxiv.org/abs/2610.22179)

    LCT提出了一个GPU友好的无损张量压缩执行框架，利用浮点数据冗余和优化的霍夫曼编码内核，在几乎不增加训练时间的情况下实现高达1.455倍压缩比，将训练内存占用降低最多24.6%并支持3.5倍的更长上下文。

    

    无损张量压缩可以减少训练期间的内存使用，但训练需要张量被反复重新编码。我们提出了无损压缩张量（LCT），一个专为这种反复编码-解码模式设计的GPU执行框架。LCT利用浮点数据中天然存在的冗余性来减少权重、激活值和优化器状态的存储空间。高度优化的GPU内核实现了基于预计算码本的霍夫曼编码，从而将压缩和解压缩的开销降到最低，这对训练至关重要。我们的分析刻画了LCT的存储成本及其对数值幅度变化的适应能力。LCT实现了高达1.455倍的压缩比，并在训练和微调场景中得到了验证。在一个1.62亿参数的NanoGPT上，LCT以7.2%的步进时间增加为代价，减少了24.6%的内存使用。在Qwen3-4B的微调中，LCT节省了高达23.8%的峰值内存，使训练时的上下文长度得以扩大3.5倍。

    arXiv:2610.22179v1 Announce Type: new  Abstract: Lossless tensor compression could reduce memory usage during training, but training requires tensors to be repeatedly re-encoded. We introduce Losslessly Compressed Tensors (LCT), a GPU execution framework designed for this repeated encode-decode regime. LCT exploits the natural redundancy in floating point data to reduce storage for weights, activations and optimizer states. Highly optimized GPU kernels implement Huffman coding with precomputed codebooks for low compression and decompression overhead, essential for training. Our analysis characterizes LCT's storage costs and adaptation to magnitude shifts. LCT achieves a compression ratio of up to $1.455\times$, and is demonstrated in training and finetuning settings. On a 162M-parameter NanoGPT, LCT reduces memory usage by $24.6\%$ for a $7.2\%$ step-time increase. In Qwen3-4B fine-tuning, LCT saves up to $23.8\%$ of peak memory, enabling a $3.5\times$ larger context length in training
    
[^35]: REKI：在视觉-语言-动作模型中注册知识

    REKI: Registering Knowledge in VLAs

    [https://arxiv.org/abs/2610.22178](https://arxiv.org/abs/2610.22178)

    REKI是一个仅用公开数据预训练的24亿参数视觉-语言-动作模型，通过在动作专家中引入寄存器标记记忆机制来保留近期历史信息，以更小的模型规模达到了接近两倍大小模型的性能，并显著提升了仿真与真实机器人任务的成功率、大幅减少了姿态振荡。

    

    我们推出了REKI，一个完全基于公开数据预训练的24亿参数视觉-语言-动作模型（VLA），它在性能与模型规模之间树立了新的帕累托前沿：在LIBERO-Plus、RoboTwin Clean2Rand和RoboCasa365基准上，只有参数量约为其两倍的模型才能超越它。REKI由一个视觉-语言模型（VLM）和一个流匹配动作专家组成，在精心筛选的公开机器人数据、以自我为中心的人类数据以及视觉-语言数据的混合数据集上进行预训练。此外，我们为其赋予了一种记忆机制，使策略能够以几乎为零的成本访问其近期历史：动作专家中的几个寄存器标记（register tokens）总结VLM在每一帧的输出，并持续保留在专家的缓存中。这种记忆机制提升了仿真环境和真实机器人上的成功率，并使姿态之间的振荡减少约四倍。REKI采用简单的模型设计和训练范式，并在训练与推理效率上进行了优化。我们发布了（原文在此处截断）。

    arXiv:2610.22178v1 Announce Type: cross  Abstract: We introduce REKI, a 2.4B vision-language-action model (VLA) pretrained exclusively on public data that sets a new Pareto frontier between performance and model size: on LIBERO-Plus, RoboTwin Clean2Rand and RoboCasa365, it is outperformed only by models about twice its size. REKI consists of a vision-language model (VLM) followed by a flow-matching action expert, and is pretrained on a curated mixture of public robot, egocentric human and vision-language data. Additionally, we endow it with a memory that gives the policy access to its recent past at negligible cost: a few register tokens in the action expert summarise the output of the VLM at each frame and persist in the expert's cache. This memory improves success in simulation and on real robots, and makes oscillations between poses about four times rarer. REKI uses a simple model design and training paradigm, and is optimised to be efficient at training and inference. We release th
    
[^36]: Mosaic：多会话智能体强化学习

    Mosaic: Multi-Session Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.22174](https://arxiv.org/abs/2610.22174)

    提出 Mosaic 方法，通过无评论员的层次化优化机制区分多会话智能体强化学习中各会话的不同贡献，弥合单会话训练假设与多会话推理执行之间的不匹配。

    

    现有的大多数智能体强化学习方法都假设轨迹是单会话的，即交互在单一连续累积的上下文中进行。然而，上下文压缩和子智能体分叉等智能体技术可能产生跨越多个会话的轨迹，这导致单会话训练假设与推理时实际执行之间存在不匹配。因此，我们提出了多会话智能体强化学习，这是一个更通用的设定，支持一条轨迹中包含任意数量的会话。在该设定下，一种被广泛采用的方法是将轨迹级优势广播到所有会话，但这无法区分各个会话的不同贡献，也无法指导轨迹内会话的分配。为了解决这一问题，我们提出了 Mosaic，一种采用无评论员（critic-free）层次化优化的多会话智能体强化学习方法。在轨迹层面，它引入了会话使用信号（摘要在此处截断）

    arXiv:2610.22174v1 Announce Type: cross  Abstract: Most existing agentic reinforcement learning methods assume single-session trajectories, where interaction proceeds within one continuously accumulated context. However, agentic techniques such as context compaction and subagent forking may produce trajectories spanning multiple sessions, creating a mismatch between the single-session training assumption and inference-time execution. Therefore, we introduce Multi-Session Agentic Reinforcement Learning, a more general setting that supports an arbitrary number of sessions within a trajectory. Under this setting, a widely adopted method is to broadcast trajectory-level advantages to all sessions, but this does not distinguish contributions from various sessions or guide session allocation within trajectories. To address this issue, we propose Mosaic, a multi-session agentic RL method with critic-free hierarchical optimization. At the trajectory level, it incorporates session-usage signals
    
[^37]: 词法参考框架：以原生语义解读语言模型残差流

    Lexical Reference Frames: Interpreting LM Residual Streams in Native Semantics

    [https://arxiv.org/abs/2610.22169](https://arxiv.org/abs/2610.22169)

    论文提出“词法参考框架”，通过模型自身剩余层诱导的、随上下文自适应的原生参考框架来解读残差流中的隐状态，并证明任何固定方向的透镜都无法超越原生框架相干性平方根的理论上限，而 LexFrame 在 SmolLM3-3B 和 OLMo3-7B 上达到 0.97 的相干性且读取误差在 0.16 nats 以内。

    

    语言模型的词表即是它的原生语义：嵌入与反嵌入在残差流的两端固定了一个状态的含义。在这两端之间，现有方法要么复用一个固定的透镜，要么把状态视为点云，两者都没有以模型自身的方式来读取状态。我们论证每个隐状态都拥有一个原生的参考框架，它由接下来读取该状态的层所诱导，并随上下文自适应变化。词法参考框架通过模型自身剩余的层来读取状态——其中位置之间进行指定轮数的注意力，其余部分采用典型的注意力输出——从而近似这一原生参考框架，而无需拟合模型的输出。任何具有每层固定方向的透镜都无法超过原生框架相干性的平方根，在 SmolLM3-3B 和 OLMo3-7B 的中层深度分别为 0.30 和 0.35，而一阶框架 LexFrame[1] 达到了 0.97 和 0.97，读取状态与模型自身的读出相差不超过 0.16 nats。

    arXiv:2610.22169v1 Announce Type: new  Abstract: The vocabulary of a language model is its native semantics: the embedding and the unembedding fix the meaning of a state at the two ends of the residual stream. Between them, existing methods either reuse a fixed lens or treat the states as a point cloud, and neither reads a state the way the model does. We argue that every hidden state has a native reference frame, induced by the layers that read it next and adaptive to the context. Lexical reference frames approximate it by reading a state through the model's own remaining layers with a stated number of rounds of attention between positions and a typical attention output for the rest, without fitting to the model's output. No lens, with fixed directions per layer, can exceed the square root of the native frame's coherence, 0.30 and 0.35 at mid-depth of SmolLM3-3B and OLMo3-7B, whereas the first-order frame, LexFrame[1], reaches 0.97 and 0.97 and reads a state within 0.16 nats of the mo
    
[^38]: 物理信息DeepONet中张量处理与实现的重要性

    The importance of tensor handling and implementation of physics-informed DeepONets

    [https://arxiv.org/abs/2610.22157](https://arxiv.org/abs/2610.22157)

    提出一种与原架构数学等价的改进PI-DeepONet，通过消除Branch网络处理输入函数时的冗余，减小输入张量规模和网络评估次数，从而显著降低高维问题的训练计算成本。

    

    物理信息神经网络（PINNs）通过在训练过程中施加控制方程，减少或消除了对大规模标注数据集的需求。物理信息深度算子网络（PI-DeepONets）将这一方法扩展到学习解算子，能够对一系列问题实现快速预测。尽管具有这些优势，我们表明其在高维问题上的应用仍然受到训练计算成本的限制。为了解决这一问题，我们提出了一种改进的PI-DeepONet架构，通过消除Branch网络在各配置点处理输入函数时的冗余来降低这一成本。所提出的公式在数学上与原始架构等价，但减小了输入张量的尺寸并减少了Branch网络的评估次数。我们分析了两种架构的计算复杂度，并推导出了训练时间关于数量的理论缩放定律。

    arXiv:2610.22157v1 Announce Type: new  Abstract: Physics-informed neural networks (PINNs) reduce or eliminate the need for large labelled datasets by enforcing governing equations during training. Physics-informed Deep Operator Networks (PI-DeepONets) extend this approach to learning solution operators, enabling rapid prediction across families of problems. Despite these advantages, we show that their application to higher-dimensional problems remains limited by the computational cost of training. To address this issue, we propose a modified PI-DeepONet architecture that reduces this cost by removing redundancy in how the Branch network processes input functions across collocation points. The proposed formulation is mathematically equivalent to the original architecture but reduces input tensor sizes and the number of Branch network evaluations. We analyse the computational complexity of both architectures and derive theoretical scaling laws for training time with respect to the number
    
[^39]: 清洁网格世界中的规范博弈与一种基于约束强化学习的缓解方法

    Specification Gaming and a Constrained Reinforcement Learning Mitigation in a Cleaning Gridworld

    [https://arxiv.org/abs/2610.22152](https://arxiv.org/abs/2610.22152)

    该论文在一个清洁机器人网格世界中演示了规范博弈现象——智能体学会隐藏灰尘而非清除灰尘来骗取代理奖励，并提出一种通过惩罚博弈行为、带有可验证保证的约束强化学习方法来缓解这一问题。

    

    强化学习智能体会精确地优化其被给予的奖励信号，当该信号只是真实目标的代理时，优化结果可能与预期的真实目标产生严重偏离；这种失效模式被称为规范博弈。关于这种失效模式的现有证据大多是观察性的，很少有研究将受控的、刻意构建的代理奖励与真实目标之间的差距，与一种带有可验证保证（而非仅仅是经验性改进）的缓解方法相结合进行演示。我们构建了一个最小化的网格世界，其中清洁机器人智能体既可以真正清除灰尘，也可以向廉价的传感器隐藏灰尘，并证明仅基于传感器代理奖励训练的标准深度Q网络收敛到一个完全博弈该奖励的策略，在从未实现真实目标的情况下达到接近最大的代理奖励。随后，我们引入了一种约束强化学习的修复方法，通过惩罚智能体借以进行博弈的特定行为（原文摘要在此处被截断）。

    arXiv:2610.22152v1 Announce Type: cross  Abstract: Reinforcement learning agents optimize exactly the reward signal they are given, which can diverge sharply from an intended true objective when that signal is only a proxy for it; this failure mode is known as specification gaming. Existing evidence for this failure mode is largely observational, with comparatively few demonstrations pairing a controlled, deliberately constructed proxy/true gap with a mitigation that carries a verifiable guarantee rather than only an empirical improvement. We construct a minimal gridworld in which a cleaning-robot agent can either genuinely remove dirt or conceal it from a cheap sensor, and show that a standard deep Q-network trained only on the sensor's proxy reward converges to a policy that games it completely, reaching near-maximal proxy reward while never achieving the true objective. We then introduce a constrained reinforcement learning fix, penalizing the specific behaviors through which the ag
    
[^40]: 因果积分梯度：沿因果路径的可扩展特征归因

    Causal Integrated Gradients: Scalable Feature Attribution Along Causal Paths

    [https://arxiv.org/abs/2610.22126](https://arxiv.org/abs/2610.22126)

    提出因果积分梯度（CIG）和因果期望梯度（CEG），通过沿因果图路由积分路径来利用特征间已知的因果结构，在保持积分梯度计算效率和公理化保证的同时实现可扩展的特征归因，避免将功劳错误归于下游效应而非真正的上游原因。

    

    现有的考虑特征依赖关系的特征归因方法无法扩展到高维场景。我们提出了因果积分梯度和因果期望梯度（CEG），它们对积分梯度方法进行了扩展，以纳入输入特征之间的先验因果结构，同时保持其计算效率，并在因果图满足一定条件时保留其公理化保证。在许多场景中，特征之间的某些因果结构是已知的，例如转录因子因果性地调控靶基因的表达。然而，现有的基于梯度的方法无法利用这种结构，反而将归因功劳归于特征的下游效应而非其真正的上游原因，例如将靶基因排在调控它的转录因子之上。我们的核心思想是通过让积分路径沿因果图行进来嵌入这种因果结构。由此，该方法将每个特征的归因分解为……（摘要在此处截断）

    arXiv:2610.22126v1 Announce Type: cross  Abstract: Existing feature attribution methods that account for feature dependencies do not scale to high dimensions. We propose causal integrated gradients (CIG) and causal expected gradients (CEG), which extend integrated gradients to incorporate prior causal structure among input features while retaining its computational efficiency and, under conditions on the causal graph, its axiomatic guarantees. In many settings, some causal structure between features is already known, for example, that a transcription factor causally regulates a target gene's expression. Yet existing gradient-based methods cannot exploit this structure, instead crediting a feature's downstream effects over its true upstream cause, such as ranking the target gene above the transcription factor that regulates it. Our key idea is to embed this causal structure by routing the integration path along the causal graph. The method thereby decomposes each feature's attribution i
    
[^41]: 超越固定采集函数：面向扰动发现的时间差分主动学习

    Beyond Fixed Acquisition Functions: Temporal Difference Active Learning for Perturbation Discovery

    [https://arxiv.org/abs/2610.22122](https://arxiv.org/abs/2610.22122)

    提出了一种名为TDAL的主动学习方法，将预算约束下的扰动发现建模为序贯决策问题，通过时间差分学习从单次筛选轨迹中在线学习采集函数，突破传统固定采集函数的局限，实现自适应的实验选择。

    

    基因与药物扰动筛选是阐明疾病机制和验证治疗靶点的重要工具，但实验成本限制了可测试的候选对象数量。现有的扰动筛选方法通常在利用反馈更新代理模型的同时保持采集函数固定，这限制了它们针对不同响应景观自适应调整选择策略的能力。实验反馈能够揭示当前的命中结果，并通过更新候选预测来指导后续的选择，从而为学习如何选择实验提供了训练信号。我们将预算约束下的扰动发现表述为一个序贯决策问题，并提出时间差分主动学习（TDAL），该方法通过时间差分学习从单次筛选轨迹中在线学习采集函数，同时兼顾当前的命中结果与未来的潜在发现。TDAL利用命中概率分布之间的Hellinger距离……

    arXiv:2610.22122v1 Announce Type: new  Abstract: Genetic and drug perturbation screens are important tools for elucidating disease mechanisms and validating therapeutic targets, but experimental costs limit the number of candidates that can be tested. Existing perturbation screening methods typically update the surrogate model with feedback while keeping the acquisition function fixed, limiting their ability to adapt selection to different response landscapes. Experimental feedback reveals current hits and informs subsequent selection by updating candidate predictions, providing a training signal for learning how to select experiments. We formulate budgeted perturbation discovery as a sequential decision problem and propose Temporal Difference Active Learning (TDAL), which learns an acquisition function online through temporal difference learning from a single screening trajectory, accounting for current hits and future discoveries. TDAL uses the Hellinger distance between hit probabil
    
[^42]: 基于物理特性表示上的扩散后验推断，利用泛在无线信号进行场景重建

    Scene Reconstruction Using Ubiquitous Wireless Signals via Diffusion Posterior Inference Over physics-Grounded Representations

    [https://arxiv.org/abs/2610.22120](https://arxiv.org/abs/2610.22120)

    该论文将无线场景重建表述为后验推断问题，提出一个基于物理特性表示、结合扩散模型的两阶段分层框架，从泛在的无线信道测量中重建周围环境的场景布局。

    

    了解物理世界的布局对许多应用都至关重要，然而我们所依赖的传感器（如摄像头和激光雷达）在黑暗、恶劣天气或墙壁遮挡的情况下可能失效。无线信号则不受这些限制：基站与用户之间的每一条链路都会受到其穿过、反射自或被其阻挡的建筑物的影响，从而将周围场景的信息嵌入到在链路两端测得的无线信道中。然而，将信道测量映射到环境是一个严重的病态问题：一次测量仅记录了沿传播路径与发射电磁波发生相互作用的结构，而且即使是这些观测也往往可以由多个合理的场景来解释。因此，我们将无线场景重建表述为后验推断问题，并开发了一个包含两个互补阶段的分层框架来解决这一问题。无线基础模……

    arXiv:2610.22120v1 Announce Type: new  Abstract: Knowing the layout of the physical world matters for many applications, yet the sensors we rely on for it (e.g., cameras and LiDAR) can fail in darkness, in adverse weather, or behind walls. Wireless signals do not face these limitations: every link between a base station and a user is shaped by the buildings it passes, reflects from, or is blocked by, leaving information about the surrounding scene embedded in the wireless channel measured at the endpoints. However, mapping channel measurements to the environment is a severely ill-posed problem: a measurement records only the structures that interact with the transmitted electromagnetic waves along its propagation paths, and even these observations can often be explained by multiple plausible scenes. We therefore formulate wireless scene reconstruction as posterior inference and develop a hierarchical framework with two complementary stages to approach this problem. A wireless foundatio
    
[^43]: 在文本新颖处切分补丁：字节级模型应将算力投入何处的压缩视角

    Patch where the text is new: a compression view of where byte-level models should spend compute

    [https://arxiv.org/abs/2610.22100](https://arxiv.org/abs/2610.22100)

    提出受LZ77压缩启发的新颖性打补丁方法——用一个几秒钟即可在无标注文本上拟合的、基于前两个字节的查找表，在字节不重复近期上下文处切分补丁，从而大幅提升字节级语言模型在数学任务上的准确率（50M参数下最终答案准确率从60%升至90%）。

    

    诸如字节潜在Transformer（BLT）等字节级语言模型在每个补丁上运行一次大型全局模型，并在小模型的下一字节熵较高处开始新补丁。然而这一规则很少在行内计算结果处开始新补丁，例如解题过程中"="后面的数字，在紧凑的补丁预算下这些位置的准确率会大幅下降。受LZ77式压缩的启发，我们提出新颖性打补丁：在即将到来的字节不太可能重复近期上下文的位置开始新补丁。该规则是一个基于前两个字节的查找表，仅需在原始文本上几秒钟即可拟合完成，无需标注数据、参考模型或损失函数。在1.9 GB数学语料库上从头训练的BLT风格模型中，在匹配的约11%补丁率下（每种设置各3个随机种子），该方法在1M参数量下没有明显效果，在12M参数量下将最终答案准确率从47%提升至72%，在50M参数量下将计算结果准确率从21%提升至57%、最终答案准确率从60%提升至90%；在……上拟合的表

    arXiv:2610.22100v1 Announce Type: new  Abstract: Byte-level language models such as the Byte Latent Transformer (BLT) run their large global model once per patch and start a patch where a small model's next-byte entropy is high. This rule rarely starts a patch at in-line computed results, such as the number after "=" in a worked solution, and accuracy on them collapses under tight patch budgets. We propose novelty patching, motivated by LZ77-style compression: start a patch where the upcoming bytes are unlikely to repeat the recent context. The rule is a lookup table over the previous two bytes, fitted in seconds on raw text with no labels, reference model or losses. In BLT-style models trained from scratch on a 1.9 GB math corpus at a matched ~11% patch rate (3 seeds each), it has no clear effect at 1M parameters, raises final-answer accuracy from 47% to 72% at 12M, and at 50M raises computed-result accuracy from 21% to 57% and final-answer accuracy from 60% to 90%; a table fitted on 
    
[^44]: TIPS：一种用于扩散语言模型扩展的微型推理时策略

    TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language Models

    [https://arxiv.org/abs/2610.22099](https://arxiv.org/abs/2610.22099)

    本文提出TIPS，一个仅约60万参数的微型推理时控制器，通过强化学习同时学习解掩码与重新掩码策略，在保持底层语言模型完全冻结的情况下，显著提升了掩码扩散语言模型在数学推理任务上的表现。

    

    掩码扩散语言模型在推理时面临一个选择：每轮去噪之后哪些位置变得可见。许多现有的解码方法依赖手工设计的规则，或要求可见性单调增长。我们提出了一种微型推理时扩展策略（TIPS），这是一个约60万参数的控制器，它在保持底层语言模型冻结的同时，能够同时学习解掩码与重新掩码。TIPS利用感知可见性的置信度特征和一个浅层Transformer来选择哪些token应该可见，同时由一个固定的时间表控制预期的可见token比例。我们通过监督预热训练，随后利用基于最终答案正确性奖励的强化学习来训练该控制器。在基于LLaDA-8B-Instruct的GSM8K和MATH-500基准测试中，全画布与双分块两种TIPS变体在32个解码器预算下分别取得了58.46%和57.97%的微平均准确率。

    arXiv:2610.22099v1 Announce Type: cross  Abstract: Masked diffusion language models expose an inference-time choice: which positions become visible after each denoising pass. Many existing decoding methods use hand-designed rules or require visibility to grow monotonically. We present a $\textbf{T}$iny $\textbf{I}$nference-time $\textbf{P}$olicy for $\textbf{S}$caling ($\textbf{TIPS}$), an approximately $600$K-parameter controller that learns both unmasking and remasking while keeping the underlying language model frozen. TIPS uses visibility-aware confidence features and a shallow transformer to select which tokens should be visible, while a fixed schedule controls the expected visible fraction. We train the controller through supervised warmup followed by reinforcement learning from terminal correctness rewards. On GSM8K and MATH-500 with LLaDA-8B-Instruct, the full-canvas and two-block TIPS variants achieve $58.46\%$ and $57.97\%$ micro-averaged accuracy at a budget of $32$ decoder 
    
[^45]: 超越主导标签：利用多示例学习恢复共存的病灶证据

    Beyond the Dominant Label: Recovering Coexisting Lesion Evidence with Multiple Instance Learning

    [https://arxiv.org/abs/2610.22091](https://arxiv.org/abs/2610.22091)

    提出DomSet-MIL框架，通过主导标签删失似然和严重程度排序残差路由，在弱监督下恢复与主导病灶共存时的次要病灶证据，避免低严重程度病变信息被抑制。

    

    全切片病理图像可能包含共存的病灶，然而弱监督通常只提供最严重的诊断。我们将这一场景形式化为主导标签多示例学习，其中观测到的标签是潜在病灶集合中严重程度最高的元素。标准的多分类多示例学习（MIL）将其视为完整的一位有效编码描述，因此当较低严重程度的病灶与主导病灶共存时，其证据会被抑制。我们提出了DomSet-MIL，这是一个标签感知的框架，它在包级和实例级之间对齐标注机制。主导标签删失似然建模了报告的诊断，而不会宣称未报告的病灶不存在。严重程度排序的残差路由将patch证据从高严重程度到低严重程度进行分配，并为次要发现保留残差支持，同时top-q池化将局部路由分数与包级预测联系起来。在自然和受控设置下的实验表明，DomSet-MIL 取得了……

    arXiv:2610.22091v1 Announce Type: cross  Abstract: Whole-slide images may contain coexisting lesions, yet weak supervision typically provides only the most severe diagnosis. We formulate this setting as dominant-label multiple instance learning, where the observed label is the maximum-severity element of a latent lesion set. Standard multiclass MIL treats it as a complete one-hot description and therefore suppresses lower-severity evidence whenever it co-occurs with a dominant lesion. We propose DomSet-MIL, a label-aware framework that aligns the annotation mechanism across bag and instance levels. Dominant-Label Censored Likelihood models the reported diagnosis without declaring unreported lesions absent. Severity-Ordered Residual Routing assigns patch evidence from high to low severity and preserves residual support for secondary findings, while top-$q$ pooling links local routing scores to bag prediction. Experiments across natural and controlled settings show that DomSet-MIL achiev
    
[^46]: 学习改变了世界，而后所学到的也随之改变：在控制回路中训练与部署神经网络

    Learning Changes the World and Then What Was Learned has Changed: Training and Deploying Neural Networks in the Control Loop

    [https://arxiv.org/abs/2610.22084](https://arxiv.org/abs/2610.22084)

    本文提出了一个统一分类体系来刻画离线训练的神经网络部署到闭环控制后因自身动作改变状态分布而引发的三类失效模式，并通过时间尺度分离的在线适应设计策略来缓解该分布偏移问题。

    

    神经网络为控制那些标称建模难以处理的复杂系统提供了关键的适应能力。然而，当离线训练的神经网络被部署到闭环控制系统中时，其动作会从根本上改变其训练时所依据的状态分布。本文引入了一个统一的分类体系，对这些顺序发生的失效模式进行归类：初始的闭环辨识误差、部署引发的动作误差以及持续的跟踪误差。通过离散时间敏感性分析，本工作形式化地阐明了离线学习的参数如何在部署时局部地导致轨迹发散，并因局部损失几何结构而造成性能退化。为缓解这种分布偏移，我们提出了以时间尺度分离的在线适应为核心的可操作设计策略。这些机制通过在非线性Duffing振子上的消融实验得到验证，表明……

    arXiv:2610.22084v1 Announce Type: cross  Abstract: Neural networks offer critical adaptability for controlling complex systems where nominal modeling proves intractable. However, when offline-trained neural networks are deployed within closed-loop control systems, their actions fundamentally alter the state distribution upon which they were trained. This paper introduces a unified taxonomy to categorize these sequential failure modes: the initial Closed-Loop Identification Error, the deployment-induced Action Error, and the continuous Tracking Error. Through discrete-time sensitivity analysis, this work formalizes how offline-learned parameters locally guarantee a trajectory divergence upon deployment, degrading performance due to local loss geometry. To mitigate this distribution shift, we propose actionable design policies centered on time-scale separated online adaptation. These mechanisms are validated through ablation studies on a nonlinear Duffing oscillator, demonstrating that b
    
[^47]: AI 能预报极端天气吗？评估概率天气模型在热带气旋上的表现

    Can AI Forecast the Extremes? Evaluating Probabilistic Weather Models on Tropical Cyclones

    [https://arxiv.org/abs/2610.22077](https://arxiv.org/abs/2610.22077)

    通过对热带气旋预报开展广泛的概率基准评估，本研究发现尽管在强度和集合离散度方面仍有不足，以GenCast和FourCastNet 3为代表的领先AI模型在预报性能上大幅超越了ECMWF的业务集合预报系统IFS-ENS。

    

    天气预报正经历一场重要变革，数据驱动的AI模型以极低的计算成本在平均表现上超越了领先的数值天气预报系统。然而，这种成功是否能延伸到极端事件仍存在争议。早期评估揭示了显著的不足，但主要集中在确定性模型和指标上，使得最新概率模型的能力尚不清晰。我们对热带气旋预报进行了广泛的基准测试，将GenCast、FourCastNet 3和FGN与ECMWF基于物理的业务集合预报系统IFS-ENS进行比较。我们开发了一套概率指标和诊断分析方法，用于评估气旋路径追踪、强度偏差、不确定性以及表征风暴结构的风-压关系。我们发现，尽管在强度和集合离散度方面仍存在局限，领先的AI模型大幅超越了IFS-ENS。

    arXiv:2610.22077v1 Announce Type: new  Abstract: Weather prediction is undergoing an important shift, with data-driven AI models outperforming leading numerical weather prediction systems on average at a fraction of the computational cost. Yet, whether this success extends to extreme events remains contested. Earlier evaluations have exposed important shortcomings, but have largely focused on deterministic models and metrics, leaving the capabilities of the latest probabilistic models less understood. We conduct an extensive benchmark of tropical cyclone forecasting, comparing GenCast, FourCastNet 3, and FGN with ECMWF's operational physics-based ensemble, IFS-ENS. We develop a suite of probabilistic metrics and diagnostic analyses to assess cyclone tracking, intensity biases, uncertainty, and the wind-pressure relationship that characterizes storm structure. We find that, despite remaining limitations in intensity and ensemble spread, leading AI models substantially outperform IFS-ENS
    
[^48]: 正确的字节优先：UTF-16 如何助力多语言边界预测

    The Right Byte First: How UTF-16 Helps Multilingual Boundary Prediction

    [https://arxiv.org/abs/2610.22071](https://arxiv.org/abs/2610.22071)

    该论文提出通用字节级编码（UBE），将三字节 UTF-8 码点改写为低字节在前的两个 UTF-16 字节，使边界预测器从首个符号即可获得更多区分信息，从而提升多语言字素边界预测的准确性。

    

    许多分层字节级语言模型使用学习到的边界预测器将字节分组为补丁，而该预测器对下一个码点的观察视角取决于编码方式和前瞻符号数。在 UTF-8 中，许多文字的码点占用三个字节：24 位中有 8 位用于标记长度与延续性，而首字节仅携带 4 个高码点位。因此，在相同的左侧上下文之后，延长字素簇的组合标记与开启下一个字素的字母可能共享相同的首字节。通用字节级编码（UBE）保留一字节和两字节的 UTF-8 编码，但将此类码点写作两个 UTF-16 字节（低字节在前），使其首符号携带 8 个低码点位，足以区分这类延续情况。我们在匹配的 FineWeb-2 文本上训练小型因果预测器，并在 FLORES-200 上于 UTF-8、UBE 以及 UBE 的等长大端变体三种编码下进行评估。在单符号前瞻条件下，针对 13 种婆罗米文字中 17 种语言在码点边界处的平衡字素边界错误率……（原文摘要在此处截断）

    arXiv:2610.22071v1 Announce Type: new  Abstract: Many hierarchical byte-level language models group bytes into patches using a learned boundary predictor whose view of the next codepoint depends on encoding and lookahead. In UTF-8, codepoints of many scripts take three bytes: 8 of 24 bits mark length and continuation, and the first byte carries only 4 high codepoint bits. After the same left context, a mark extending a grapheme cluster and a letter starting the next can thus share their first byte. Universal Byte-Level Encoding (UBE) keeps one- and two-byte UTF-8 encodings but writes these codepoints as two UTF-16 bytes, low byte first, so its first symbol carries 8 low codepoint bits that can distinguish such continuations. We train small causal predictors on matched FineWeb-2 text and evaluate on FLORES-200 under UTF-8, UBE, and UBE's same-length big-endian variant. At one-symbol lookahead, balanced grapheme-boundary error at codepoint boundaries across 17 languages in 13 Brahmic scr
    
[^49]: 权重即记忆：为流式视频内化长期历史信息

    Memory as Weights: Internalizing Long-Term History for Streaming Videos

    [https://arxiv.org/abs/2610.22070](https://arxiv.org/abs/2610.22070)

    提出MaLoW框架，利用循环超网络将视频历史信息逐步内化为固定大小的LoRA权重，使流式视觉语言模型以权重形式承载长期记忆，避免了历史信息与当前观察对有限上下文的争夺。

    

    流式视觉语言模型（VLM）必须持续整合过去的信息，同时保持对当前观察和用户查询的响应能力。现有的记忆机制通常对历史观察进行压缩、检索或缓存，并将其重新引入输入上下文，这迫使历史记忆和当前观察竞争有限的上下文空间。我们提出MaLoW，一个流式记忆框架，它将历史上下文内化为不断演化的轻量级参数更新。MaLoW使用循环超网络将传入的视频观察逐步整合到固定大小的LoRA权重中，使历史信息得以延续，而无需重新访问完整的视频流。通过这种方式，MaLoW将历史表示为权重，同时保持当前观察作为输入。为支持训练，我们进一步构建了MaLoW-Instruct，一个包含381K条查询的大规模流式视频数据集。

    arXiv:2610.22070v1 Announce Type: cross  Abstract: Streaming vision-language models (VLMs) must continuously integrate information from the past while remaining responsive to current observations and user queries. Existing memory mechanisms typically compress, retrieve, or cache historical observations and reintroduce them into the input context, forcing historical memory and current observations to compete for limited context. We propose MaLoW, a streaming memory framework that instead internalizes historical context into evolving lightweight parameter updates. MaLoW uses a recurrent HyperNetwork to incrementally consolidate incoming video observations into fixed-size LoRA weights, allowing historical information to be carried forward without revisiting the full video stream. In doing so, MaLoW represents history as weights, while keeping the current observation as input. To support training, we further construct MaLoW-Instruct, a large-scale streaming video dataset containing 381K qu
    
[^50]: ABNAT：面向模拟存内计算系统的基于注意力的噪声感知训练鲁棒Transformer方法

    ABNAT: Attention-Based Noise-Aware Training for Robust Transformers on Analog Compute-in-Memory Systems

    [https://arxiv.org/abs/2610.22065](https://arxiv.org/abs/2610.22065)

    该论文提出ABNAT，一种基于注意力的噪声感知训练方法，通过分析模拟存内计算噪声如何重塑Transformer注意力机制（发现深层注意力差异增大且传统NAT存在过度自信问题），显著提升Transformer在模拟CIM硬件上的鲁棒性。

    

    基于Transformer的模型已成为众多任务的支柱，但其密集的计算和数据搬运带来了巨大的能耗开销。模拟存内计算（CIM）提供了一种高能效的替代方案，然而其固有的硬件非理想性会扰动存储的权重，并可能显著降低模型性能。尽管现有的噪声感知训练（NAT）方法通过在训练期间注入噪声来提升鲁棒性，但它们在很大程度上忽视了注意力机制在Transformer推理中的关键作用。为弥合这一差距，我们分析了CIM引起的噪声如何重塑Transformer中的注意力，并得到两个关键观察结果：（1）干净与带噪注意力图之间的差异在更深层中显著增大；（2）即使注意力路由明显偏离干净参考，传统NAT仍可能保持过度自信。受这些发现启发，我们提出了ABNAT，一种基于注意力的（摘要在此处被截断）

    arXiv:2610.22065v1 Announce Type: new  Abstract: Transformer-based models have become the backbone of a wide range of tasks, but their intensive computation and data movement incur substantial energy overhead. Analog compute-in-memory (CIM) offers an energy-efficient alternative, yet its inherent hardware non-idealities perturb stored weights and can significantly degrade model performance. Although existing noise-aware training (NAT) methods improve robustness by injecting noise during training, they largely overlook the critical role of attention mechanisms in Transformer inference.To bridge this gap, we analyze how CIM-induced noise reshapes attention in Transformers and identify two key observations: (1) the divergence between clean and noisy attention maps increases substantially in deeper layers, and (2) conventional NAT can remain overconfident even when its attention routes deviate markedly from the clean reference. Motivated by these findings, we propose ABNAT, an attention-ba
    
[^51]: 重新思考连续拒绝算法

    Rethinking Successive Rejects Algorithms

    [https://arxiv.org/abs/2610.22051](https://arxiv.org/abs/2610.22051)

    本文通过将 SR 算法的阶段长度调度设计表述为归一化误差指数的优化问题，提出了 H1-平衡连续拒绝算法，从而在固定预算最优臂识别中提升了 SR 算法的性能。

    

    在最优臂识别问题中，算法通过自适应地选择各个臂来找到平均奖励最大的臂。连续拒绝算法是一种简单的固定预算最优臂识别算法，它在每个阶段淘汰一个候选臂。乍一看，SR 似乎自适应能力有限，因为它遵循预先确定的阶段长度。然而，对于具有共同方差的高斯臂，SR 以流行的复杂度度量 H1(P) 归一化后的最坏情况误差指数在常数因子范围内是最优的，这部分解释了为什么 SR 的性能与最先进的自适应采样算法相比具有竞争力。在本文中，我们通过优化 SR 预先定义的阶段长度调度来扩展其框架。我们将阶段调度设计表述为一个关于归一化误差指数的优化问题。然后，我们引入了一种具有优化调度的 SR 版本，我们称之为 H1-平衡连续拒绝算法。

    arXiv:2610.22051v1 Announce Type: cross  Abstract: In the best-arm identification problem, an algorithm adaptively selects arms to find the arm of the largest mean reward. Successive Rejects (SR) is a simple fixed-budget best-arm identification algorithm that eliminates one candidate arm at each phase. At first glance, SR seems to have limited adaptivity, since it follows predetermined phase lengths. However, for Gaussian arms with common variance, SR's worst-instance error exponent normalized by the popular complexity measure $H_1(P)$ is optimal up to a constant factor, which partly explains why SR's performance is competitive compared with state-of-the-art adaptive sampling algorithms. In this paper, we extend the framework of SR by optimizing its predefined schedule of phase lengths. We formulate phase-schedule design as an optimization problem for the normalized error exponent. Then, we introduce a version of SR with the optimized schedule, which we call $H_1$-balanced Successive R
    
[^52]: CNN技术签名探测器在“突破聆听”数据中的漏检与误报

    Missed Detections and False Alarms in a CNN Technosignature Detector on Breakthrough Listen Data

    [https://arxiv.org/abs/2610.22050](https://arxiv.org/abs/2610.22050)

    该研究通过向真实绿岸天文台数据中注入15,000个不同形态和振幅的信号，对一个冻结的CNN技术签名探测器进行压力测试，发现其召回率随信号增强反而下降，且某些干扰形态的假阳性率高达0.627，揭示了现有机器学习探测器在训练分布之外的脆弱性。

    

    射电技术签名（technosignature）搜索依赖于机器学习探测器在人工审查之前缩小候选池。这些探测器通常在来自训练分布的留出数据上进行验证，而这种验证方式恰恰排除了它们最可能失败的情形：技术签名具有未知的形态，而地面干扰的种类则无穷无尽。本文作者训练了一个标准的竞赛型探测器一次后将其冻结，然后用15,000次信号注入对其进行探测，涵盖十种形态和五种振幅，并将这些信号置于真实的绿岸天文台背景数据中。测试了两个方向：形态不同于直线窄带信号的真实信号，以及与训练数据相似的干扰。结果显示，召回率没有遵循预期的排序，且随着信号变亮反而下降，即使是探测器被训练寻找的那类“大海捞针”式信号也是如此（召回率从0.590降至0.090）。其中一种干扰形态达到了0.627的假阳性率，与探测器自身的基线相当。

    arXiv:2610.22050v1 Announce Type: cross  Abstract: Radio technosignature searches rely on machine learning detectors to shrink the candidate pool before a human looks at it. They are validated on held-out data from the training distribution, which excludes the cases where they are most likely to fail: a technosignature has unknown morphology, and terrestrial interference is unbounded in variety. I trained a standard competition detector once, froze it, and probed it with 15,000 injections across ten morphologies and five amplitudes, placed into real Green Bank backgrounds. Two directions were tested: genuine signals with morphologies other than a straight narrowband line, and interference resembling the training data. Recall did not follow the predicted ordering, and it fell as signals brightened, even for a needle of the kind the detector was trained on (from 0.590 to 0.090). One interference morphology reached a false positive rate of 0.627, comparable to the detector's own baseline 
    
[^53]: 基于时间建模与多模态对齐的fMRI自然语音解码

    Decoding Naturalistic Speech from fMRI with Temporal Modeling and Multimodal Alignment

    [https://arxiv.org/abs/2610.22049](https://arxiv.org/abs/2610.22049)

    提出NOEMA时间卷积解码器，通过将连续fMRI测量与多模态（文本、音频、语音）刺激嵌入对齐，实现了从大脑活动中高精度检索自然语音片段，在2000个候选中前10命中率超过42%。

    

    从大脑活动中重建一个人所感知的内容，是检验神经记录是否携带有意义信息的严格测试。自然语音为这一挑战提供了丰富的测试平台：它在多个时间尺度上展开，但fMRI含噪且时间上模糊的测量使得这种结构难以恢复。我们提出了NOEMA，一个时间卷积解码器，它将连续fMRI测量的时间窗口与多模态刺激嵌入对齐。这些嵌入结合了同一刺激片段的互补的文本、音频和语音表征，使其能够直接从大脑活动中被检索。在两个自然语音数据集Cephalonauts One和LeBel上，NOEMA在超过42%的保留查询中，从2,000个候选片段中于前10名内检索出正确的片段。这显著优于基线解码器，并且当候选池扩展到数千个干扰项时，这一优势依然保持。

    arXiv:2610.22049v1 Announce Type: cross  Abstract: Reconstructing what a person perceives from their brain activity is a stringent test of whether neural recordings carry meaningful information. Naturalistic speech offers a rich testbed for this challenge: it unfolds over multiple timescales, but fMRI's noisy, temporally smeared measurements make this structure hard to recover. We introduce NOEMA, a temporal convolutional decoder that aligns windows of consecutive fMRI measurements with multimodal stimulus embeddings. These embeddings combine complementary text, audio, and speech representations of the same stimulus segment, enabling its retrieval directly from brain activity. On two naturalistic speech datasets, Cephalonauts One and LeBel, NOEMA retrieves the correct segment among 2,000 candidates within the top 10 for more than 42% of held-out queries. This substantially outperforms baseline decoders, an advantage that persists when the candidate pool is extended with thousands of di
    
[^54]: 隐状态探针在定位推理失败方面的有效性

    Effectiveness of Hidden-State Probes for Localizing Reasoning Failures

    [https://arxiv.org/abs/2610.22046](https://arxiv.org/abs/2610.22046)

    该研究发现基于最终答案正确性训练的隐状态探针难以有效区分推理失败的具体类型，例如置信度探针在检测单步致命错误和可重试成功的情况时分别漏掉了81%和47%的案例。

    

    验证长推理轨迹的最终答案可以表明该轨迹是错误的，但无法区分三种情况：问题可能超出了模型的能力范围、再次尝试可能会成功，或者在一个原本可恢复的轨迹中某一步骤导致轨迹失败。标准的隐状态探针是基于最终答案正确性来训练的，而这一标签混淆了上述几种情况。蒙特卡洛展开可以区分这些情况并定位失败的步骤，但需要额外的推理成本。我们测试了4B和8B推理模型的隐状态，用于判断失败类型和失败步骤，并将每个探针与不使用隐状态的最强基线进行比较。在8B模型的错误轨迹上，置信度探针（一种在中点截断处经过校准的正确性探针）错过了81%的由单一步骤导致轨迹不可恢复的情况，以及47%的再次尝试可能会成功的情况。一种在同一状态上采用不同训练设计的探针……

    arXiv:2610.22046v1 Announce Type: cross  Abstract: Verifying the final answer of a long reasoning trace may show that the trace is wrong but it does not distinguish three cases. The problem may lie beyond the model's capability, another attempt would succeed, or one step in an otherwise recoverable trace made the trace fail. Standard hidden-state probes are trained on final-answer correctness, a label that conflates these cases. Monte Carlo rollouts distinguish the cases and locate the failing step at additional inference cost. We test the hidden states of 4B and 8B reasoning models for the failure type and the failing step and compare each probe with the strongest baseline that uses no hidden states. On incorrect 8B traces, the confidence probe, a calibrated correctness probe at a midpoint cutoff, misses 81% of cases in which a single step makes the trace unrecoverable and 47% of cases in which another attempt would succeed. A probe with a different training design over the same state
    
[^55]: 大语言模型金融预测中的动量、评论语气与盈利意外

    Momentum, Commentary Tone, and Earnings Surprise in LLM Financial Forecasts

    [https://arxiv.org/abs/2610.22043](https://arxiv.org/abs/2610.22043)

    本研究通过独立操纵动量、评论语气和EPS意外的合成财报实验发现，在大语言模型的金融预测中，非基本面的评论语气和价格动量因素对模型输出的影响往往超过实际的每股收益意外本身，揭示了LLM金融预测易受非信息性输入干扰的问题。

    

    是什么输入驱动语言模型的金融预测？我们构建了合成的财报公告，独立地反转20日动量、非信息性的评论语气，以及固定的实际每股收益（EPS）是否超越被操纵的共识预期。来自四个模型家族（基础参数量为4.9亿至76.1亿）的六个指令/对话检查点转换版本，在直接回答的Q8_0协议下各接收2,016条提示。实验设计包含24个数字区块、六组评论配对、盈利与亏损情形、上升与下降的年度EPS，以及两种意外幅度。A/B答案分配与U/D（上涨/下跌）覆盖每个转换版本的全部384个主要情景。我们估计的是条件答案前缀分数的匹配变化，而非校准后的市场概率。在这些检查点中，主要EPS对比的范围为-0.19至86.80个百分点。在六个模型中的五个里，主要符号语气对比的幅度超过了EPS对比，动量对比在其中五个模型中也超过了EPS对比。

    arXiv:2610.22043v1 Announce Type: cross  Abstract: Which inputs drive a language model's financial forecast? We construct synthetic earnings announcements that independently reverse 20-day momentum, non-informational commentary tone, and whether fixed actual EPS beats a manipulated consensus. Six instruction/chat checkpoint conversions from four families (0.49-7.61B base parameters) receive 2,016 prompts each under a direct-answer Q8_0 protocol. The design includes 24 numerical blocks, six commentary pairs, profits and losses, rising and falling annual EPS, and two surprise sizes. Both A/B answer assignments and U/D cover all 384 primary scenarios per conversion. We estimate matched changes in conditional answer-prefix scores, not calibrated market probabilities. Across these checkpoints, the primary EPS contrast ranges from -0.19 to 86.80 percentage points. The magnitude of the primary signed tone contrast exceeds that of EPS in 5 of six models, and momentum exceeds it in 5. Natural-w
    
[^56]: ETH：面向大模型训练的数据均衡异构流水线并行

    ETH: Data-Balanced Heterogeneous Pipeline Parallelism for Training Large Models

    [https://arxiv.org/abs/2610.22035](https://arxiv.org/abs/2610.22035)

    该论文提出异构流水线并行方法（HPP）及ETH训练系统，通过分层工作负载均衡框架和分块梯度聚合技术，在异构流水线间实现细粒度数据均衡，同时解决了配置搜索开销大和非对称All-Reduce通信开销高的关键难题。

    

    当前面向大模型的异构感知训练系统通过在流水线各阶段分配不等数量的层来平衡工作负载，这种策略旨在通过独特的流水线配置使各阶段执行时间相等。然而，这种方法未能解决关键的内存限制问题，且平衡粒度存在天然粗糙的缺陷。本文提出了异构流水线并行（HPP），这是一种能够在异构流水线之间实现高效、细粒度数据均衡的新方法。为解决HPP面临的关键挑战——最优配置搜索开销过大以及非对称All-Reduce带来的高通信开销——我们设计了ETH训练系统。ETH采用分层工作负载均衡框架，联合优化异构流水线之间以及单个流水线内部各阶段的负载均衡。为降低非对称通信开销，ETH实现了分块梯度聚合机制。

    arXiv:2610.22035v1 Announce Type: cross  Abstract: Current heterogeneity-aware training systems for large models balance workloads by assigning unequal numbers of layers across pipeline stages, a strategy aimed at equalizing execution time through a unique pipeline configuration. However, this approach fails to address critical memory constraints and suffers from inherently coarse balancing granularity.   In this paper, we propose Heterogeneous Pipeline Parallelism (HPP), a novel approach enabling efficient and fine-grained data balance across heterogeneous pipelines. To address HPP's key challenges, prohibitive search overhead for optimal configurations and high communication overhead from asymmetric All-Reduce, we design the ETH training system. ETH employs a hierarchical workload balance framework that jointly optimizes balance across heterogeneous pipelines and within individual pipeline stages. To mitigate asymmetric communication costs, ETH implements block-wise gradient aggregat
    
[^57]: 为外部决策者设计实验

    Designing Experiments for External Decision Makers

    [https://arxiv.org/abs/2610.22032](https://arxiv.org/abs/2610.22032)

    提出“使用感知实验设计”框架，通过训练自适应设计策略最小化“使用风险”，以应对数据将被实验者无法控制的外部规则（可能是非贝叶斯或黑盒）使用的情况，并揭示在此类场景下更具信息量的数据反而可能有害。

    

    标准的贝叶斯实验设计（BED）方法假设实验者本人会依据数据采取行动，即基于自身的信念和目标选择最优行动。但在实践中，数据往往由外部各方来使用，其处理流程是实验者无法改变且可能并不了解的，尤其是当数据被公开发布时。为了将BED扩展到此类场景，我们提出了“使用感知实验设计”框架，通过训练自适应设计策略来最小化“使用风险”，即当一条或多条外部规则作用于所收集数据时实验者的期望损失。这些规则可能是非贝叶斯的或黑盒的；当可能有多条规则适用时，我们或对所有规则取平均，或采用最坏情况。我们证明，与标准BED不同，在此类规则下更具信息量的数据可能是有害的，因此使用风险通常无法表示为期望不确定性缩减；我们的框架因此严格推广了标准BED。

    arXiv:2610.22032v1 Announce Type: cross  Abstract: Standard Bayesian experimental design (BED) methods assume the experimenter alone will act on the data, choosing the best action given their own beliefs and goals. In practice, data are often acted on by external parties whose procedures the experimenter cannot change and may not know, particularly when the data are released publicly. To extend BED to such settings, we introduce usage-aware experimental design, a framework in which an adaptive design policy is trained to minimise the usage risk: the experimenter's expected loss when one or more external rules act on the collected data. These rules may be non-Bayesian or black-box, and when several may apply, we either average over them or take the worst case. We show that, unlike in standard BED, more informative data can be harmful under such rules, so the usage risk cannot in general be expressed as an expected uncertainty reduction; our framework thus strictly generalises standard B
    
[^58]: 选择并非学习：为模型抉择未来

    Selection Is Not Learning: Choosing Model Futures

    [https://arxiv.org/abs/2610.22029](https://arxiv.org/abs/2610.22029)

    自进化语言模型中“看起来可学习的任务”并不等于“有用的模型更新”，本文提出 HeirCommit 与 EvoHeir 方法，通过冻结评估和参数-经验分离式继承来弥合选择与更新之间的差距。

    

    自进化语言模型在自身生成的任务上进行训练，因此每一轮都必须决定下一轮将继承哪个更新后的模型。当前的循环方法（如 R-Zero、Agent0 和 Skill Self-Play）通过任务来做出这一决定：接受那些看起来可学习的任务，并继承由此产生的所有更新。我们在这些循环中发现了一个“选择到更新的差距”（selection-to-update gap）：一个看起来可学习的任务未必会带来有用的更新。在经过验证的任务上，大多数更新并不能使正确答案更可能出现，然而这些循环仍然将其全部继承。HeirCommit 冻结更新后的模型，并使用全新的评估问题来决定是继承该模型还是保留当前模型。EvoHeir 则将参数继承与经验继承分离开来，在最终确认之前，利用当前和存档的经验来修复失败并保留成功。我们的分析刻画了由选择引起的乐观偏差，并将遗憾分解为选择、吸收和……

    arXiv:2610.22029v1 Announce Type: cross  Abstract: Self-evolving language models train on tasks they generate, so every round must decide which updated model the next round inherits. Current loops such as R-Zero, Agent0, and Skill Self-Play make this decision through the task, admitting tasks that look learnable and inheriting every resulting update. We identify a selection-to-update gap in these loops, where a task that looks learnable need not produce a useful update. Most updates on verified tasks fail to make correct answers more likely, yet these loops inherit them all. HeirCommit freezes the updated model and uses fresh evaluation questions to decide whether to inherit it or retain the current model. EvoHeir separates parameter inheritance from experience inheritance, using current and archived experience to repair failures and preserve successes before final confirmation. Our analysis characterizes selection-induced optimism and decomposes regret into selection, absorption, and 
    
[^59]: 增强缓存，替换激活：内部状态共享的2×2研究

    Augment the Cache, Replace the Activation: A 2x2 Study of Internal-State Sharing

    [https://arxiv.org/abs/2610.22026](https://arxiv.org/abs/2610.22026)

    本文将多智能体系统中的内部状态“传输”范式重构为“共享”范式，并发现内部状态载体与融合机制的选择是相互耦合的，其关键在于载体进入接收方计算的位置。

    

    由冻结语言模型构建的多智能体系统通过文本进行通信，这会丢弃发送方已经计算好的内部状态，并使接收方付出代价去重建它。关于内部信道的研究避免了这一成本，但将问题设定为传输：智能体A向智能体B发送信息。每一对需要通信的智能体都需要自己的传输桥梁，而系统中每增加一个智能体，都需要针对之前的每个智能体重新拟合新的桥梁。我们将这一问题重新设定为共享：智能体将其内部状态一次性发布到共享池中，每个对等体通过单一的机制读取该池，该机制不针对任何特定发送方拟合，因此增加一个智能体只需增加一个发布者，而无需新的桥梁。与文献中通常独立选择内部状态载体和融合机制的做法不同，我们的主要发现是这两个选择是相互耦合的，决定性因素在于载体进入接收方计算的位置。

    arXiv:2610.22026v1 Announce Type: new  Abstract: Multi-agent systems built from frozen language models communicate through text, which discards the internal state the sender already computed and makes the receiver pay to rebuild it. Work on internal channels avoids that cost, framing the problem as transfer: agent A sends information to agent B. Every agent pair that must communicate needs its own transfer bridge, and every agent added to a system needs a new bridge fitted against each previous agent. We reframe this problem as sharing: an agent publishes its internal state once into a pool, and every peer reads that pool through a single mechanism fitted to no sender in particular, so adding an agent adds a publisher and no new bridge. Whereas the carrier of the internal state and the fusion mechanism are usually chosen independently in the literature, our main finding is that these two choices are coupled; the deciding factor is where the carrier enters the receiver's computation. Co
    
[^60]: 并非所有无标签数据都同等重要：基于大型化学数据库的半监督QSAR建模中的无标签数据选择

    Not All Unlabeled Data Are Equal: Unlabeled Data Selection for Semi-Supervised QSAR Modeling with Large Chemical Repositories

    [https://arxiv.org/abs/2610.22025](https://arxiv.org/abs/2610.22025)

    该论文首次在真实半监督学习场景下，利用ChEMBL的10个QSAR数据集和40余万无标签分子池，系统研究了如何从大型化学数据库中选择无标签数据以改进QSAR模型，揭示了并非所有无标签数据都同等有价值。

    

    半监督学习（SSL）在标签获取成本高昂而无标签样本容易获得的领域中尤其具有吸引力。定量构效关系（QSAR）建模——即根据化学结构预测分子生物活性——正是这样一个领域。获取活性测量数据需要昂贵的湿实验室实验，导致标注数据集规模较小，而ChEMBL等化学数据库中却包含数百万个无标签分子。尽管表面上二者高度契合，但大型化学数据库通过SSL改进QSAR模型的潜力在很大程度上仍未被探索。此外，在包含真实无标签数据的现实SSL设置中，应如何选择无标签数据也尚不清楚。我们使用从ChEMBL提取的10个QSAR数据集以及超过40万个无标签分子的数据池，对无标签数据选择进行了系统性研究。为此，我们评估了三种已有的

    arXiv:2610.22025v1 Announce Type: new  Abstract: Semi-supervised learning (SSL) is particularly attractive in domains where labels are expensive to obtain, while unlabeled examples are readily available. Quantitative Structure-Activity Relationship (QSAR) modeling, which predicts molecular biological activity from chemical structure, is one such domain. Obtaining activity measurements requires costly wet-lab experiments, resulting in small labeled datasets, while chemical repositories such as ChEMBL contain millions of unlabeled molecules. Despite this apparent suitability, the potential of large chemical databases to improve QSAR models via SSL remains largely unexplored. Furthermore, it is unclear how unlabeled data should be selected in realistic SSL settings with genuine unlabeled data. We conduct a systematic study of unlabeled-data selection using 10 QSAR datasets extracted from ChEMBL and a pool of more than 400,000 unlabeled molecules. To this end, we evaluate three established
    
[^61]: DemoStream：面向自动演示学习的批量任务与运动规划优化

    DemoStream: Batched Task-and-Motion-Plan Optimization for Automated Learning from Demonstration

    [https://arxiv.org/abs/2610.22020](https://arxiv.org/abs/2610.22020)

    本文提出DemoStream，一个基于GPU加速的批量任务与运动规划（TAMP）系统，通过批量约束满足优化演示质量，自动生成高质量演示数据以提升机器人行为克隆的成功率。

    

    从演示中学习是为机器人赋予操作能力的有效策略，但需要在由海量演示构成的数据集上进行训练。现有的演示收集方法要么需要人类的持续参与（例如通过遥操作），要么是针对单一任务的专用方法。任务与运动规划系统能够自主求解，因此可以为通用任务生成演示；然而，由于其解的多模态性，从这些演示中学习已被证明十分困难。在本工作中，我们基于近期利用GPU加速进行TAMP的研究，提出了DemoStream——一个能够优化演示质量以提升行为克隆成功率的TAMP系统。首先，我们引入了批量TAMP采样操作以及相应地执行批量约束满足的新型TAMP算法，从而……

    arXiv:2610.22020v1 Announce Type: cross  Abstract: Learning from demonstration is an effective strategy for endowing robots with manipulation capabilities but requires training on datasets comprised of massive amounts of demonstrations. Existing demonstration collection approaches either require constant human involvement, for example, via teleoperation or are specialized for a single task. Task and Motion Planning (TAMP) systems are able to autonomously solve and therefore generate demonstrations for general-purpose tasks; however, learning from their demonstrations has proven difficult due to multi-modality in their solutions. In this work, we build on recent work leveraging GPU acceleration for TAMP and propose DemoStream, a TAMP system that optimizes demonstration quality in order to improve behavior cloning success rates. First, we introduce batched TAMP sampling operations along with a new TAMP algorithm that correspondingly performs batched constraint satisfaction in order to ex
    
[^62]: 为智能体后训练做准备：一份开放的中期训练方案

    Priming Base Models for Agentic Post-Training: An Open Midtraining Recipe

    [https://arxiv.org/abs/2610.22019](https://arxiv.org/abs/2610.22019)

    该论文提出了一种开放的中期训练方案，在预训练与后训练之间以 256K 上下文、数千亿 token 的自然软件工程数据混合少量智能体轨迹来训练基础模型，显著提升了 SWE-bench 等智能体任务表现及后续强化学习的可训练性。

    

    arXiv:2610.22019v1 公告类型：交叉 摘要：前沿模型，无论是通用模型还是领域专用模型，如今都在预训练和后训练之间加入了一个中期训练（midtraining）阶段，但关于这一阶段如何构建的公开资料却很少。我们提出了一种开放的中期训练方案，使基础模型为后续的智能体后训练做好准备，并在两个不同规模的 Nemotron 模型上验证了其有效性。预训练通常在短上下文中进行，几乎不涉及智能体交互；而 SFT 数据大多是合成的，且过长的 SFT 阶段可能损害模型后续强化学习（RL）的可训练性。我们的中期训练阶段介于两者之间：以 256K 上下文在数千亿 token 上进行训练，数据由多样化的自然软件工程数据（如打包的代码仓库、拉取请求、提交历史和构建日志）以及少量智能体轨迹混合而成。在 Nemotron 3.5 Lightning 上，5000 亿个中期训练 token 将 SWE-bench Verified 成绩从 60.7 提升至 64.5，SWE-bench Multilingual 从……（摘要在此处被截断）

    arXiv:2610.22019v1 Announce Type: cross  Abstract: Frontier models, general-purpose and domain-specific alike, now include a midtraining stage between pretraining and post-training, but little has been published about how this stage is built. We present an open midtraining recipe that primes base models for agentic post-training, and we show its effectiveness on two Nemotron models of different sizes. Pretraining typically runs at short context and sees almost no agent interactions, while SFT data is mostly synthetic, and very long SFT stages can hurt the model's later RL trainability. Our midtraining stage sits between the two. It trains at 256K context for hundreds of billions of tokens on a diverse mixture of natural software-engineering data like packed repositories, pull requests, commit histories, and build logs together with a small share of agentic traces. On Nemotron 3.5 Lightning, 500B midtraining tokens improve SWE-bench Verified from 60.7 to 64.5, SWE-bench Multilingual fro
    
[^63]: 足够接近实则相去甚远：KV缓存中的保真度-可用性解耦

    Close Enough is Nowhere Near: Fidelity-Consumability Decoupling in KV Caches

    [https://arxiv.org/abs/2610.22016](https://arxiv.org/abs/2610.22016)

    论文揭示了冻结的多智能体LLM系统中直接替换发送方KV缓存不可行的根本原因——读取方只能接受其自身“可容忍集”内的缓存，几何保真度与可用性相互解耦且失败并非渐进式，这一现象可由竞争性注意力机制解释。

    

    多智能体大语言模型系统通过文本进行通信，这种方式是有损且浪费的：发送方丢弃了自己已经完成的阅读理解。越来越多的文献转而传输KV缓存——这种阅读成果的天然载体。我们研究了这一思路中最具吸引力的形式：两个模型均保持冻结，直接将发送方的缓存替换给读取方使用，不进行任何训练。然而这种方法行不通，且障碍来自读取方本身。一个冻结的Transformer只能读取与自己原本会生成的缓存足够接近的缓存，我们将这一区域称为其“可容忍集”。我们证明，与该缓存的几何相似性并不能预测读取方能否使用某个替代缓存，且失败并非渐进式的，因此看似更“保真”的替代品并非如表面那样有效。我们用竞争性注意力解释了这一现象。在实验中，我们针对两个测量锚点测试了八种多样化的替换方法：一个无信息缓存基线，以及读取方自身从（原文摘要在此处截断）……

    arXiv:2610.22016v1 Announce Type: new  Abstract: Multi-agent LLM systems communicate in text, which is lossy and wasteful: the sender throws away the reading it already did. A growing literature instead transmits the KV cache, the natural carrier of that reading. We study the most appealing form of the idea: both models stay frozen and the sender's cache is substituted for the reader's with nothing trained. It does not work, and the obstruction belongs to the reader. A frozen transformer reads only caches close to the one it would have written itself, a region we call its tolerable set. We show that geometric similarity to that cache does not predict whether the reader can use it, and the failure is not gradual, so a more faithful substitute is not the remedy it appears to be. We explain this phenomenon with competitive attention. Experimentally, we test eight diverse substitution methods against two measured anchors: an uninformative cache baseline and the reader's own accuracy from h
    
[^64]: GraphBEAT：基于图的分桶事件时间聚合框架，用于胃肠道癌症的早期风险分层

    GraphBEAT: Graph-based Bucketized Event Aggregation Over Time for Early Risk Stratification of Gastrointestinal Cancers

    [https://arxiv.org/abs/2610.22014](https://arxiv.org/abs/2610.22014)

    提出GraphBEAT时序图表示学习框架，将纵向EHR临床事件转化为按时间分桶的患者特定有向图，结合时间衰减消息传递与多头注意力池化，实现胃肠道癌症的早期风险分层。

    

    纵向电子健康记录（EHR）包含关于患者健康的互补性诊断与实验室信息。然而，文档记录的不规则性、稀疏性、异质性和不完整性使EHR建模变得困难。为解决这一问题，我们提出了GraphBEAT（基于图的分桶事件时间聚合），这是一个时序图表示学习框架，它将临床事件按时间分桶分组，并将观察到的特征转换为患者特定的、时间有向的图，从而实现胃肠道癌症的早期风险分层。EHR图中的每个节点编码临床事件的特征标识、观察值和时间位置。一种上下文感知且具有时间衰减机制的消息传递方法建模相邻时间桶之间的转换，随后通过多头注意力池化生成患者级别的表示。GraphBEAT在多站点纵向诊断数据上进行了训练和评估。

    arXiv:2610.22014v1 Announce Type: cross  Abstract: Longitudinal electronic health records (EHRs) contain complementary diagnostic and laboratory information about patient health. However, irregularity, sparsity, heterogeneity, and incompleteness in documentation complicate EHR modeling. To address the problem, we introduce GraphBEAT (Graph-based Bucketized Event Aggregation over Time), a temporal graph representation learning framework that groups clinical events into time buckets and transforms observed features into patient-specific, temporally directed graphs to enable early risk stratification for gastrointestinal cancers. Each node in an EHR graph encodes feature identity, observed value, and temporal position of clinical events. A context-aware, time-decay message-passing mechanism models transitions between successive buckets, followed by multi-head attention pooling to generate patient-level representations. GraphBEAT was trained and evaluated on multi-site longitudinal diagnos
    
[^65]: 面向深度学习编译器的有状态测试

    Stateful Testing for Deep Learning Compilers

    [https://arxiv.org/abs/2610.22003](https://arxiv.org/abs/2610.22003)

    提出面向深度学习编译器的有状态测试方法 Rondo，通过将模型与调用序列配对、利用编译行为引导的 MCMC 采样进行序列变异，并结合三路差分预言机，检测由编译器保留状态引发且在单独执行时无法复现的缺陷。

    

    现代深度学习编译器采用惰性编译方式，并在多次调用之间缓存编译产物。保留的编译器状态可能导致后续调用产生错误结果，即使该调用在单独执行时是正确的。对 PyTorch、JAX 和 TensorFlow 中 26 个公开报告问题的分析表明，检测此类缺陷需要先建立编译器状态、再暴露故障的调用序列。我们提出了 Rondo，一种有状态测试方法，它将每个测试用例表示为一个模型与一个调用序列的组合。Rondo 对调用序列进行变异，并使用由编译行为引导的马尔可夫链蒙特卡洛采样器，以偏向较少被观察到的行为。其三路差分测试预言机比较三种执行方式：不编译直接执行、重置编译器状态后的编译执行，以及保留编译器状态时的编译执行。它通过在保留状态与重置状态两种条件下重放相同调用（同时保持输入和程序状态不变）来确认有状态缺陷，然后……（摘要在此处截断）

    arXiv:2610.22003v1 Announce Type: cross  Abstract: Modern deep learning compilers compile models lazily and cache artifacts across invocations. Retained compiler state can make a later invocation incorrect even when it is correct in isolation. An analysis of 26 publicly reported issues in PyTorch, JAX, and TensorFlow shows that detecting such bugs requires sequences that first establish compiler state and then expose a failure. We present Rondo, a stateful testing method that represents each test case as a model paired with an invocation sequence. Rondo mutates invocation sequences and uses a Markov chain Monte Carlo sampler guided by compilation behavior to favor rarely observed behaviors. Its three-way differential oracle compares execution without compilation, compiled execution with compiler state reset, and compiled execution with state retained. It confirms stateful bugs by replaying the same invocation with state retained and reset while preserving inputs and program state, then
    
[^66]: 接口与模块：一种跨架构权重空间学习的可扩展表示

    Interfaces and Modules: A Scalable Representation for Weight-Space Learning across Architectures

    [https://arxiv.org/abs/2610.21995](https://arxiv.org/abs/2610.21995)

    该论文提出了一种以“接口”（坐标置换空间）和“模块”（操作超边）为基本单元的类型化图表示，并构建了通用权重编码器（UWE），通过局部等变更新与接口摘要上的全局注意力相结合，实现了尊重参数对称性且可扩展、可跨架构的权重空间学习。

    

    跨架构的权重空间学习需要既尊重参数对称性、又能扩展到大型网络的表示方法。现有的等变张量编码器需要针对特定架构指定对称性，而神经元图则使全局注意力的计算代价过高。我们提出了一种围绕共享坐标置换作用组织的类型化图。其节点被称为“接口”，表示其绑定的张量轴必须被联合置换的坐标空间；操作的具体出现被称为“模块”，构成类型化的超边。一个编译器可以从受支持的 PyTorch 模型中自动推导出该图，并精确恢复由其约束所定义的置换因子。通用权重编码器（UWE）将对参数特征的局部等变更新与对不变接口摘要的全局注意力相结合。由于接口数量很少，全局注意力变得切实可行，并且单个编码器即可处理不同的受支持架构。

    arXiv:2610.21995v1 Announce Type: new  Abstract: Weight-space learning across architectures requires representations that respect parameter symmetries and scale to large networks. Existing equivariant tensor encoders require architecture-specific symmetry specifications, while neuron graphs make global attention costly. We introduce a typed graph organized around shared coordinate-permutation actions. Its nodes, called interfaces, represent coordinate spaces whose bound tensor axes must be permuted jointly; operation occurrences, called modules, form typed hyperedges. A compiler derives this graph from supported PyTorch models, with exact recovery of the permutation factors defined by its constraints. The Universal Weight Encoder (UWE) combines local equivariant updates to parameter features with global attention over invariant interface summaries. The small number of interfaces makes global attention practical, and a single encoder processes different supported architectures. UWE is c
    
[^67]: OntoAct：基于本体抽象的假设驱动神经符号智能体

    OntoAct: Hypothesis-Based Neuro-Symbolic Agents through Ontological Abstraction

    [https://arxiv.org/abs/2610.21983](https://arxiv.org/abs/2610.21983)

    OntoAct通过本体抽象将经验转化为可复用的知识公理，让大语言模型提出假设和动作、由符号验证器检验证据并动态更新评估，从而实现跨情境的知识复用与假设驱动的决策，并配套发布了用于诊断修复工厂隐藏故障的Tacit-FLE基准。

    

    从经验中学习需要恢复可复用的知识，并确定这些知识何时适用于新情境。我们提出了OntoAct，一个基于本体抽象的假设驱动神经符号框架。OntoAct将个体经验转化为关于实体类型、关系和条件的公理，使其底层模式可以在不同情境之间复用。大语言模型（LLM）利用这种世界知识来提出假设和动作，而符号验证器则根据当前观察来评估其支持条件。验证过程识别缺失的证据，新的观察和干预结果会更新假设评估并指导后续动作。我们还介绍了Tacit-FLE，一个构建于Factorio学习环境（FLE）之上的基准，用于利用继承的运行记录来诊断和修复现有工厂中的隐藏故障。其合成的维护语料库将证据分布在……（摘要原文在此处截断）

    arXiv:2610.21983v1 Announce Type: new  Abstract: Learning from experience requires recovering reusable knowledge and determining when it applies to a new situation. We introduce OntoAct, a hypothesis-based neuro-symbolic framework built on ontological abstraction. OntoAct turns individual experiences into axioms over entity types, relations, and conditions, making their underlying patterns reusable across situations. A large language model (LLM) uses this world knowledge to propose hypotheses and actions, while a symbolic verifier assesses their supporting conditions against current observations. Verification identifies missing evidence, and new observations and intervention outcomes update hypothesis assessments and guide subsequent actions. We also introduce Tacit-FLE, a benchmark built on the Factorio Learning Environment (FLE) for diagnosing and repairing hidden faults in existing factories using inherited operational records. Its synthetic maintenance corpus distributes evidence a
    
[^68]: 拓扑等价类的深度学习

    Deep Learning of Topological Equivalence Classes

    [https://arxiv.org/abs/2610.21981](https://arxiv.org/abs/2610.21981)

    该论文将深度学习应用于判断黎曼面上有限群作用的拓扑等价性，探讨了代数结构对模型设计的影响以及模型成功习得的拓扑等价类结构性质。

    

    我们展示了深度学习在判断两个生成向量是否对应于黎曼面上拓扑等价的有限群作用这一问题上的应用，说明了为语义理解而开发的架构同样可以用于纯数学问题。在此过程中，我们讨论了期望的代数结构如何影响模型设计，并探索了模型成功学习到了拓扑等价类的哪些结构性质。

    arXiv:2610.21981v1 Announce Type: cross  Abstract: We demonstrate an application of deep learning to the problem of identifying whether two generating vectors correspond to topologically equivalent finite group actions on Riemann surfaces, showing how architectures developed for semantic understanding have use for pure mathematical problems. In doing so we discuss how desired algebraic structure influences model design, and explore which structural properties of topological equivalence classes the model has successfully learnt.
    
[^69]: P2TPO：面向整页文档解析的结构对齐页级到词元级信用分配

    P2TPO: Structure-Aligned Page-to-Token Credit Assignment for Full-Page Document Parsing

    [https://arxiv.org/abs/2610.21979](https://arxiv.org/abs/2610.21979)

    P2TPO提出了一种结构对齐的信用分配框架，通过参考元素对齐、统一锚点定价与片段信用重分配，将页级奖励细化到词元级别，从而提升端到端整页文档解析的准确性与完整性。

    

    端到端的整页文档解析必须保留异构元素的内容与结构，然而即使输出总体准确，其中仍可能包含局部错误与遗漏。单一的页级优势无法明确指出在采样的多个输出中，哪些元素的质量存在差异，以及错误发生在何处。我们提出了页级到词元级策略优化（Page-to-Token Policy Optimization, P2TPO），这是一个作用于整页策略轨迹的结构对齐信用分配框架。参考元素对齐首先将块级质量对比与生成的文本片段关联起来。统一锚点定价（Unified Anchor Pricing, UAP）综合了匹配与缺失元素的结果，并在采样的输出对某个参考元素获得相同但不完美评分时，提供相对于固定质量上限的反馈。在匹配的块内，片段信用重分配（Span Credit Redistribution, SCR）根据对齐后的正确与错误片段重新分配已有的块级信用，利用边界p……（原文摘要在此处截断）

    arXiv:2610.21979v1 Announce Type: cross  Abstract: End-to-end full-page document parsing must preserve the content and structure of heterogeneous elements, yet otherwise accurate outputs can contain local errors and omissions. A single page-level advantage does not explicitly identify which elements differ in quality across sampled outputs or where their errors occur. We introduce Page-to-Token Policy Optimization (P2TPO), a structure-aligned credit-assignment framework that operates on full-page policy trajectories. Reference-element alignment first associates block-level quality contrasts with generated spans. Unified Anchor Pricing (UAP) incorporates matched and missing element outcomes and provides feedback relative to a fixed quality ceiling when sampled outputs receive identical but imperfect scores for a reference element. Within matched blocks, Span Credit Redistribution (SCR) redistributes existing block credit according to aligned correct and erroneous spans, using boundary p
    
[^70]: 只在无法计算之处学习：对高斯过程后验的结构化修正

    Learn Only Where You Can't Compute: Structured Corrections to Gaussian Process Posteriors

    [https://arxiv.org/abs/2610.21977](https://arxiv.org/abs/2610.21977)

    提出一种结构化后验修正框架，在易处理的高斯参考基础上，由扩散模型仅在被选中的低维差异子空间中学习数据驱动的修正，从而兼顾高斯过程的可扩展性与扩散模型的表达能力，用于科学推断任务。

    

    物理信息高斯过程（GP）通过对经验测量、边界条件和控制方程进行不确定性建模，为科学推断提供了一个有原则的框架。然而，它们难以扩展到更大的数据集和非线性信息。相反，扩散模型擅长从海量数据中学习函数上的复杂分布，但在训练数据稀缺且必须融入物理或观测信息的科学场景中表现不佳。我们提出了一种结构化后验修正框架，将一个易于处理的高斯参考分布与通过生成学习实现的数据驱动修正相结合。在完成均值修正和协方差校准之后，扩散模型在一个经过挑选、用于捕捉与参考后验差异的低维子空间中学习有针对性的修正，而其补空间中的部分则保持高斯分布。在三个科学推断任务上，这些修正（此处原文被截断）

    arXiv:2610.21977v1 Announce Type: new  Abstract: Physics-informed Gaussian processes (GPs) provide a principled framework for scientific inference by expressing uncertainty over empirical measurements, boundary conditions, and governing equations. However, they struggle to scale to larger datasets and nonlinear information. Conversely, diffusion models excel at learning complex distributions over functions from massive datasets, but struggle in scientific contexts where training data are scarce and physical or observational information must be incorporated. We propose a structured posterior-correction framework that combines a tractable Gaussian reference with data-driven corrections via generative learning. After a mean correction and covariance calibration, a diffusion model learns a targeted correction in a low-dimensional subspace selected to capture discrepancies from the reference, while the complement retains its Gaussian law. Across three scientific inference tasks, these corre
    
[^71]: 伪造之时：持续机器学习加剧深度伪造检测中的不公平性

    Time to Fake It: Continual Machine Learning Intensifies Unfairness in Deepfake Detection

    [https://arxiv.org/abs/2610.21974](https://arxiv.org/abs/2610.21974)

    该论文首次系统研究了持续学习对深度伪造检测器公平性的影响，发现在按时间顺序的11种深度伪造生成方法上渐进训练时，准确性下降通常会加剧跨人口群体（尤其是交叉性群体）的不公平现象。

    

    深度伪造检测器用于防范虚假信息，因此其失效不仅是技术问题，也是社会问题。随着生成器不断演进，在静态数据集上训练的检测器无法应对新出现的威胁，而渐进式训练又会引发灾难性的干扰。然而，持续学习（CL）主流关注的重点——在准确性方面缓解遗忘——并不足以反映真实世界的影响。因此，我们仔细研究了持续学习方法对公平性演化的影响。为此，我们针对在按时间顺序排列的11种真实世界深度伪造生成方法上渐进训练的检测器，分析了跨不同人口统计群体的四个公平性维度。通过随时间跟踪公平性指标，我们揭示了性能与公平性之间非平凡的相互作用，这种相互作用同时取决于持续学习算法的选择和人口统计属性的交叉性。我们发现准确性下降通常会加剧不公平性……

    arXiv:2610.21974v1 Announce Type: cross  Abstract: Deepfake detectors safeguard against misinformation, rendering their failures as much a societal as a technical concern. As generators constantly evolve, detectors trained on static datasets miss newly arising threats, while progressive training interferes catastrophically. Yet, mitigating forgetfulness in terms of accuracy, the prevalent focus of continual learning (CL), is insufficient to reflect real-world impact. We therefore scrutinize the effects of CL methods with respect to evolving fairness. To this end, we analyze four dimensions of fairness across diverse demographics for detectors trained progressively on a chronological sequence of 11 real-world deepfake generation methods. Tracking fairness metrics over time, we expose non-trivial interplays between performance and fairness that depend on both the choice of CL algorithm and demographic intersections. We find accuracy degradation to typically exacerbate unfairness and, con
    
[^72]: 面向扩展现实（XR）系统中通信与计算联合的不确定性感知联邦学习

    Uncertainty-Aware Federated Learning for Joint Communication and Computation in XR Systems

    [https://arxiv.org/abs/2610.21969](https://arxiv.org/abs/2610.21969)

    本文提出了一种不确定性感知的分布式深度学习方法，通过集中式或联邦式训练的深度神经网络，在信道与计算状态不确定的干扰网络中鲁棒地联合优化发射功率、计算功率与计算任务分配，以最小化6G网络中XR设备的最坏情况时延。

    

    通信与计算的融合，在人工智能（AI）增强能力的推动下，正成为实现第六代通信系统（6G）各项前沿功能的关键使能技术。为此，本文研究了一种AI原生的鲁棒网络，其中多个基站（BS）同时为若干扩展现实（XR）设备提供服务。本文提出了一种基于分布式深度学习的创新方法，在信道状态信息和计算状态信息均存在不确定性的情况下，同时管理发射功率、计算功率以及计算任务分配。在这种干扰网络场景下，本文提出了一种鲁棒方法，在计算和功率约束条件下，最小化所有被服务XR设备的最大（最坏情况）时延。本文利用一个深度神经网络（DNN），该网络既可以集中式训练，也可以以联邦方式进行训练，用以映射……（摘要原文在此截断）

    arXiv:2610.21969v1 Announce Type: cross  Abstract: The convergence of communication and computation, empowered by artificial intelligence (AI) enhanced capabilities, emerges as a key enabler for realizing the promising functionalities of the sixth-generation of communication systems (6G). To this end, this paper investigates an AI-native, robust network, where multiple base stations (BSs) simultaneously serve several extended reality (XR) devices. The paper then presents an innovative distributed deep learning-based approach to simultaneously manage the transmit and computing powers, and computation allocation, amidst uncertainties in both channel and computing states information. Under such an interference network scenario, the paper proposes a robust approach for minimizing the worst-case delay across the served XR devices, subject to computation and power constraints. The paper leverages a deep neural network (DNN) that can be trained either centrally or in a federated manner to map
    
[^73]: 为什么浅层神经网络首先学习正弦分量：一个正交多项式动力学框架

    Why Shallow Neural Networks Learn Sinusoidal Components First: An Orthogonal Polynomial Dynamics Framework

    [https://arxiv.org/abs/2610.21964](https://arxiv.org/abs/2610.21964)

    该论文提出基于一般傅里叶级数展开的正交多项式动力学框架，通过分析动力学矩阵的特征值衰减与特征向量形状，从理论上解释了浅层神经网络为何优先学习正弦（低频）分量的谱偏置现象，并适用于一般激活函数、弱数据分布假设和任意网络结构。

    

    神经网络通常先学习低频分量、后学习高频分量——这一现象被称为谱偏置或F-原理。许多数值研究已经证实了这一现象，理论研究也尝试对其进行解释。然而，以下几个目标仍未得到充分解决：(1) 渐近地解释谱偏置，包括其起源以及为什么被优先学习的分量通常是正弦的；(2) 将解释推广到一般的激活函数，如sigmoid和ReLU；(3) 对数据分布做较弱的假设；(4) 避免混叠效应，例如有限域傅里叶分析中出现的混叠效应；(5) 发展一个适用于任意网络的理论框架。我们通过分析动力学矩阵H的特征值衰减和特征向量形状，从理论上实现了这些目标——该矩阵在我们将网络输出展开为一般傅里叶级数（GFS）时产生……

    arXiv:2610.21964v1 Announce Type: cross  Abstract: Neural networks often learn low-frequency components before high-frequency ones --- a phenomenon called spectral bias or termed F-principle. Many numerical studies have demonstrated this phenomenon, and theoretical studies have sought to explain it. However, several goals remain insufficiently addressed: (1) explaining spectral bias asymptotically, including its origin and why the preferred components are generally sinusoidal; (2) extending the explanation to general activation functions, such as sigmoid and ReLU; (3) making weak assumptions about the data distribution; (4) avoiding aliasing effects, such as those arising in finite-domain Fourier analysis; and (5) developing a theoretical framework applicable to arbitrary networks. We address these goals theoretically by analyzing the eigenvalue decay and eigenvector shapes of the dynamics matrix H, which arises when we expand the network output as a general Fourier series (GFS) in an 
    
[^74]: ChronoSheaf：基于胞腔层束的时空联合谱滤波

    ChronoSheaf: Joint Spatio-Temporal Spectral Filtering on Cellular Sheaves

    [https://arxiv.org/abs/2610.21958](https://arxiv.org/abs/2610.21958)

    ChronoSheaf提出了一种在空间和时间上联合定义的谱层束拉普拉斯算子及相应谱滤波器，使层束神经网络能够显式建模图上信号随时间的演化，并在不扩大底层层束结构的情况下同时扩展空间与时间维度的有效感受野。

    

    层束神经网络通过为每条边分配一个可学习的线性限制映射来推广图神经网络，该映射规定了相邻节点的局部表示之间如何相互关联，从而缓解过平滑问题。然而，现有的层束架构将层束拉普拉斯算子定义为瞬时空间算子，因此无法显式建模信号在图上随时间的演化。我们首先引入了一种谱层束拉普拉斯算子，它在空间和时间上联合定义，使得单个算子即可同时调控信号在空间图和时间域上的传播。若直接在空间或时间上扩展该算子的感受野，将需要越来越复杂的层束结构。为此，我们提出了一种谱滤波器，它能够在不扩大底层层束的情况下，沿这两个维度扩展有效感受野。基于该算子，我们提出了ChronoSheaf，一个时空层束神经网络……

    arXiv:2610.21958v1 Announce Type: new  Abstract: Sheaf neural networks generalize graph neural networks by assigning each edge a learned linear restriction map that specifies how the local representations of adjacent nodes relate to one another, thereby mitigating oversmoothing. Existing sheaf architectures, however, define the sheaf Laplacian as an instantaneous spatial operator and therefore cannot explicitly model how signals evolve over time on a graph. We first introduce a spectral sheaf Laplacian, defined jointly over space and time, such that a single operator regulates signal propagation across both the spatial graph and the temporal domain. Directly extending this operator's receptive field in space or time would require increasingly complex sheaf structures. Instead, we propose a spectral filter that expands the effective receptive field along both dimensions without enlarging the underlying sheaf. Building on this operator, we introduce ChronoSheaf, a spatio-temporal sheaf n
    
[^75]: SelectPFN：基于先验数据拟合网络的稀疏非线性回归摊销贝叶斯变量选择

    SelectPFN: Amortized Bayesian Variable Selection for Sparse Nonlinear Regression with Prior-Data Fitted Networks

    [https://arxiv.org/abs/2610.21950](https://arxiv.org/abs/2610.21950)

    SelectPFN是一种在单次前向传播中同时完成稀疏非线性回归贝叶斯变量选择与预测的先验数据拟合网络，可输出每个变量的后验包含概率，并通过敲除列实现精确的有限样本错误发现率控制。

    

    基于先验数据拟合网络（PFNs）的表格基础模型可以在单次前向传播中对新数据集进行预测，但它们无法报告其预测依赖于哪些输入。我们提出SelectPFN，这是一种为稀疏非线性回归联合执行贝叶斯变量选择和预测的PFN。在一次前向传播中，它返回每个变量的后验包含概率，且其预测仅从采样的变量子集计算得出，因此被排除的变量可证明无法影响预测；我们证明这正是贝叶斯模型所规定的结果。该网络基于围绕组尖峰-平板稀疏输入神经网络构建的元先验，在合成任务上进行一次性预训练。列置换等变性结合原始列与敲除列的随机交错，产生了具有精确有限样本错误发现率控制的敲除统计量，即使元先验不准确时依然成立。我们为摊销式方法建立了相应的理论。

    arXiv:2610.21950v1 Announce Type: cross  Abstract: Tabular foundation models based on prior-data fitted networks (PFNs) predict on a new data set in a single forward pass, but they cannot report which inputs their predictions depend on. We propose SelectPFN, a PFN that performs Bayesian variable selection and prediction jointly for sparse nonlinear regression. In one forward pass it returns posterior inclusion probabilities for every variable, and its predictions are computed from sampled subsets only, so excluded variables provably cannot affect them; we show this is exactly what the Bayesian model prescribes. The network is pretrained once on synthetic tasks from a meta-prior built around a group spike-and-slab sparse-input neural network. Column-permutation equivariance combined with random interleaving of original and knockoff columns yields knockoff statistics with exact finite-sample false discovery rate control, even when the meta-prior is wrong. We develop theory for amortized 
    
[^76]: 成本惩罚主动特征获取的拉格朗日结构

    The Lagrangian Structure of Cost-Penalized Active Feature Acquisition

    [https://arxiv.org/abs/2610.21919](https://arxiv.org/abs/2610.21919)

    本文揭示了成本惩罚EIG-Cost贪婪启发式方法本质上是期望成本主动特征获取问题的拉格朗日松弛的一步贪婪优化（lam为对偶乘子），并提出了将策略差距分解为预算松弛项与贪婪次优性项的后验证书，在4,000个小规模案例中验证该次优性项接近于零。

    

    主动特征获取（AFA）在获取预算约束下顺序选择特征以最大化预测准确性。一种自然的启发式方法是贪婪地最大化期望信息增益减去成本惩罚，即 EIG(j|x_S)-lam*c_j（EIG-Cost），但其理论保证尚不明确。我们提供了一个后验证书，将与最优硬预算策略的差距分解为预算松弛项和贪婪次优性项 Del_lam。当惩罚后的最优解可计算时，该证书是可计算的；在小型异构成本实例上，我们对其进行了精确验证，在4,000个已求解的案例中发现 Del_lam 接近于零。在大型临床任务上，Del_lam 无法计算，因此该分解仅具有诊断意义而非数值保证。我们的分析将 EIG-Cost 识别为期望成本AFA的拉格朗日松弛的一步贪婪优化，其中 lam 为其对偶乘子。对于等成本情形，我们证明了 Del_lam <= exp(-gam*I...（摘要在此处截断）

    arXiv:2610.21919v1 Announce Type: new  Abstract: Active Feature Acquisition (AFA) sequentially selects features to maximize predictive accuracy under an acquisition budget. A natural heuristic greedily maximizes Expected Information Gain minus a cost penalty, EIG(j|x_S)-lam*c_j (EIG-Cost), but its theoretical guarantees are unclear. We provide an a-posteriori certificate decomposing the gap to the optimal hard-budget policy into a budget-slack term and a greedy-suboptimality term Del_lam. The certificate is computable when the penalized optimum is tractable; on small heterogeneous-cost instances, we verify it exactly, finding Del_lam near zero across 4,000 solved cases. On the large clinical task, Del_lam is not computable, so the decomposition is diagnostic rather than a numerical guarantee. Our analysis identifies EIG-Cost as one-step greedy optimization of the Lagrangian relaxation of expected-cost AFA, with lam as its dual multiplier. For equal costs, we prove Del_lam <= exp(-gam*I
    
[^77]: 策略博弈中的奖励学习

    Learning Rewards in Strategic Games

    [https://arxiv.org/abs/2610.21912](https://arxiv.org/abs/2610.21912)

    该论文提出“策略可辨识性”这一新视角，证明仅从固定策略对学习（无论专家样本数量多少）不足以在策略意义上辨识奖励，并据此提出一种主动逆强化学习算法来在策略博弈中学习奖励。

    

    理解驱动智能体行为的目标在策略环境中至关重要，在这类环境中，多个智能体相互交互并彼此适应。这一挑战出现在网络安全、自动驾驶和在线市场等领域，在这些领域中，智能体的潜在动机通常是未知的，必须从其观察到的行为中推断出来。逆强化学习（IRL）为该问题提供了一个自然的框架，但将IRL扩展到策略性多智能体场景时面临一个根本性挑战：一个能够解释观察到的行为的奖励，在用于决策时并不一定能诱导出最优策略。在这项工作中，我们通过“策略可辨识性”的视角来研究这一问题，并证明从固定的策略对中学习可能不足以在策略意义上辨识奖励，且这与专家样本数量K无关。受此启发，我们提出了一种主动IRL算法……

    arXiv:2610.21912v1 Announce Type: cross  Abstract: Understanding the objectives that drive agents' behavior is crucial in strategic environments, where multiple agents interact and adapt to one another. This challenge arises in domains such as cybersecurity, autonomous driving, and online markets, where agents' underlying incentives are typically unknown and must be inferred from their observed behavior. Inverse reinforcement learning (IRL) provides a natural framework for this problem, but extending IRL to strategic multi-agent settings introduces a fundamental challenge: a reward that explains observed behavior does not necessarily induce optimal strategies when used for decision-making. In this work, we study this problem through the lens of \emph{strategic identifiability}, and show that learning from a fixed policy pair can be insufficient to strategically identify rewards, independent of the number of expert samples $K$. Motivated by this, we introduce an active IRL algorithm for
    
[^78]: 倾斜采样问题：乐观原理、后验采样与对数遗憾

    The Tilted Sampling Problem: Optimism, Posterior Sampling, and Logarithmic Regret

    [https://arxiv.org/abs/2610.21908](https://arxiv.org/abs/2610.21908)

    该论文提出了倾斜采样这一在线学习问题，以温度归一化累积KL散度定义采样遗憾，并构建了一个通用遗憾框架，揭示倾斜采样与经典遗憾最小化具有不同的不确定性结构，借助乐观原理和后验采样可将对数级遗憾推广至从未知奖励诱导的吉布斯分布中采样的场景。

    

    许多现代学习系统的用途并非选择单一最优动作，而是采样多样化的高质量输出，例如熵正则化强化学习、RLHF（基于人类反馈的强化学习）以及奖励引导的生成建模。受这些场景启发，我们研究了倾斜采样：一个在线学习问题，其目标是从由未知奖励函数诱导的吉布斯分布中进行采样。我们将采样遗憾定义为学习者的分布与吉布斯目标分布之间经温度归一化的累积KL散度，它对应于熵正则化奖励最大化中的次优性差距。我们的框架推广了赌博机和强化学习中近期基于采样的以及KL正则化的多种表述。我们的主要贡献是一个通用的遗憾框架，它刻画了倾斜采样与经典遗憾最小化之间不同的不确定性结构。虽然标准遗憾的量级为 $\widetilde{O}(\sqr...（原文摘要在此处截断）

    arXiv:2610.21908v1 Announce Type: new  Abstract: Many modern learning systems are used not to select a single best action, but to sample diverse high-quality outputs. Examples include entropy-regularized reinforcement learning, RLHF, and reward-guided generative modeling. Motivated by these settings, we study tilted sampling: an online learning problem in which the goal is to sample from the Gibbs distribution induced by an unknown reward function. We define sampling regret as the temperature-normalized cumulative KL divergence between the learner's distribution and the Gibbs target, corresponding to the suboptimality gap in entropy-regularized reward maximization. Our framework generalizes recent sampling-based and KL-regularized formulations in bandits and reinforcement learning. Our main contribution is a general regret framework that captures the different uncertainty structure of tilted sampling and classical regret minimization. While standard regret scales as $\widetilde{O}(\sqr
    
[^79]: 专业化之前的上下文学习：可解主题模型中的学习理论

    In-Context Learning Before Specialisation: Theory of Learning in a Solvable Topic Model

    [https://arxiv.org/abs/2610.21901](https://arxiv.org/abs/2610.21901)

    该论文在一个可解的主题模型中建立了学习理论，从根本层面解释了序列模型为何以及何时会倾向于采用上下文学习而非存储训练数据的专门化外推。

    

    序列模型可以通过两种根本不同的方式进行泛化：一是外推训练中已经见过的结构，二是利用其提示来适应全新的任务。模型采取的策略决定了其在训练分布之外的行为方式，从而控制着潜在的能力与风险。近期的工作已在理论上刻画了多种不同的策略，并从实证角度描述了随着预训练数据质量与数量的变化，这些策略之间如何发生转变。然而，目前尚无针对这种竞争关系的学习理论，因此无法从根本上理解为什么一种解会比另一种解更受青睐。我们在一个可解的主题模型中提供了这样的理论——文档从有限的潜在主题混合分布中抽取，以此反映现代预训练数据的异质性。该数据集存在两种截然不同的解：一种是专门化预测器，它存储了有限的主题混合结构；另一种是上下文解，对于从未见过的主题而言是最优的。

    arXiv:2610.21901v1 Announce Type: cross  Abstract: Sequence models can generalise in fundamentally distinct ways: extrapolating structures seen in training, or using their prompt to adapt to novel tasks. The prevailing strategy determines how the model behaves beyond its training distribution, controlling potential capabilities and risks. Recent work has characterised many different strategies theoretically, and empirically described the transition between them as the quality and quantity of pretraining data vary. However, there is no learning theory for this competition, hence no fundamental understanding of \textit{why} one solution might be favoured over another. Here we provide such a theory in a solvable topic model---documents drawn from a finite mixture of latent topics, mirroring the heterogeneity of modern pretraining data. The dataset comes with two distinct solutions: a specialised predictor, that stores the finite topic mixture, and an in-context one, optimal for topics nev
    
[^80]: 用于高光谱图像无参数去除混合噪声与线状条纹的自校准子空间鲁棒主成分分析

    Self-Calibrated Subspace Robust PCA for Parameter-Free Removal of Mixed Noise and Line Stripes in Hyperspectral Images

    [https://arxiv.org/abs/2610.21896](https://arxiv.org/abs/2610.21896)

    本文提出自校准的SC-RCCTV模型，通过在迭代中自适应估计各波段噪声水平和条纹阈值（无需预先设定任何参数），实现了对高光谱图像中强度跨波段变化的混合噪声与线状条纹的有效去除。

    

    高光谱图像（HSIs）常常受到各波段强度不同的高斯噪声、脉冲噪声和条纹的污染。代表性的系数相关全变分（RCCTV）模型能够高效地恢复高光谱图像，但该模型缺少高斯噪声项和条纹项，且其噪声感知的扩展版本需要预先指定噪声水平和条纹预算。本文提出了一种自校准的RCCTV（SC-RCCTV）模型，其中噪声参数在迭代过程中自适应地估计。该模型在RCCTV中引入了逐波段的稠密噪声项以及列方向的行常数条纹项和行方向的列常数条纹项。每个波段的噪声水平由一个对条纹不变的中位数绝对偏差估计器进行初始化，并在迭代中根据残差在线细化，从而将稀疏噪声与稠密噪声的分离转化为波段自适应的Huber损失；条纹阈值则来自一种显著性检验，其虚警率可以通过解析方式预测。无需参数（原文在此处截断）

    arXiv:2610.21896v1 Announce Type: new  Abstract: Hyperspectral images (HSIs) are often corrupted by Gaussian noise, impulses and stripes whose intensities vary across bands. The representative coefficient correlated total variation (RCCTV) model restores HSIs efficiently, but it lacks Gaussian and stripe terms, and its noise-aware extensions require the noise level and a stripe budget to be specified in advance. This article proposes a self-calibrated RCCTV (SC-RCCTV) model, in which the noise parameters are estimated adaptively during the iterations. A band-wise dense term and line-constant column and row stripe terms are introduced into RCCTV. The noise level of each band is initialized by a stripe-invariant median absolute deviation estimator and refined online from the residual, which turns the separation of sparse and dense noise into a band-adaptive Huber loss, and the stripe thresholds follow from a significance test whose false-alarm rate can be predicted analytically. No param
    
[^81]: 用于免拟合二元因果推断的图狄利克雷能量

    Graph Dirichlet Energy for Fitting-Free Bivariate Causal Inference

    [https://arxiv.org/abs/2610.21892](https://arxiv.org/abs/2610.21892)

    本文提出免拟合的图狄利克雷能量因果推断准则（GDECI），通过比较两个方向上图信号的狄利克雷能量来判别二元因果方向，无需回归拟合或独立性检验，具有单调变换不变性，并可高效地在 O(n log n) 时间内精确计算。

    

    我们提出了图狄利克雷能量因果推断（GDECI）准则，用于在二元设置中区分原因与结果。对于候选方向 X→Y，我们基于 X 构建核相似性图，并将 Y 视为图上的信号。得分即为该信号的狄利克雷能量，由未归一化的图拉普拉斯算子诱导。反向得分则交换两个变量的角色，能量较低的方向被推断为因果方向。该方法既不需要回归拟合，也不需要独立性检验。经过经验概率积分变换后，其判别结果在任一变量的严格单调变换下保持不变，且图运算可简化为卷积，从而可在 O(n log n) 时间和 O(n) 内存内实现精确计算。在总体层面，GDECI 比较的是度加权贝叶斯平方预测误差加上回归函数的一个非负平滑项。当带宽趋于零时……

    arXiv:2610.21892v1 Announce Type: new  Abstract: We propose the Graph Dirichlet Energy Causal Inference (GDECI) criterion for distinguishing cause from effect in bivariate settings. For a candidate direction $X\to Y$, we construct a kernel similarity graph from $X$ and treat $Y$ as a graph signal. The score is the Dirichlet energy of this signal, induced by the unnormalized graph Laplacian. The reverse score swaps the variables, and the lower-energy direction is inferred as causal. The method requires neither regression fitting nor independence testing. After the empirical probability integral transform, its decisions are invariant under strictly monotone transformations of either variable, and the graph operations reduce to convolutions, allowing exact evaluation in $O(n\log n)$ time and $O(n)$ memory. At the population level, GDECI compares degree-weighted Bayes squared prediction errors plus a non-negative smoothness term of the regression function. As the bandwidth tends to zero, t
    
[^82]: 在GPU上学习Lean反馈以实现高效的候选证明评估

    Learning Lean Feedback for Efficient Proof-Candidate Evaluation on GPUs

    [https://arxiv.org/abs/2610.21881](https://arxiv.org/abs/2610.21881)

    该论文训练神经评估器在GPU上预测Lean 4的判定结果与编译器反馈，其两阶段评估器达到87.8%的预测准确率，吞吐量比Lean验证基线提升4.2倍，显著加速了形式化定理证明中语言模型训练的奖励评估过程。

    

    arXiv:2610.21881v1 公告类型：新论文。摘要：训练用于形式化定理证明的语言模型需要大量验证器调用。缓慢的检查和超时常常使GPU处于等待状态，并使潜在正确的证明无法获得奖励。我们使用484万个标注程序训练神经评估器，在GPU上预测Lean 4的判定结果和编译器反馈。我们比较了两个有效性分类器与一个反馈模型，后者用预测的目标状态和错误信息为证明尝试进行标注。一个两阶段评估器先用分类器筛选候选证明，再用反馈模型对其正预测进行评分，在一个从公开证明器收集的包含68,932个程序的基准上，预测Lean是否接受候选证明的准确率达到87.8%。该组合评估器在单块H100上每秒处理3.5个候选证明，吞吐量是我们带环境复用的64工作者Lean基线的4.2倍。最后，我们将该评估器与Lean判定相结合，构建了一个混合奖励系统。

    arXiv:2610.21881v1 Announce Type: new  Abstract: Training language models for formal theorem proving requires large numbers of verifier calls. Slow checks and timeouts often leave GPUs waiting and withhold rewards from potentially correct proofs. We train neural evaluators to predict Lean 4 verdicts and compiler feedback on GPUs using 4.84M labelled programs. We compare two validity classifiers with a feedback model that annotates proof attempts with predicted goal states and error messages. A two-stage evaluator that screens candidates with a classifier and scores its positive predictions with the feedback model reaches 87.8% accuracy in predicting whether Lean accepts a candidate proof, evaluated on a 68,932-program benchmark collected from public provers. The combined evaluator processes 3.5 candidates per second on one H100, 4.2x the throughput of our 64-worker Lean baseline with environment reuse. Finally, we combine our evaluator with Lean verdicts to obtain a hybrid reward syste
    
[^83]: 通过优化确定性等价实现可证明高效的风险敏感离线强化学习

    Provably Efficient Risk-Sensitive Offline Reinforcement Learning via Optimized Certainty Equivalents

    [https://arxiv.org/abs/2610.21870](https://arxiv.org/abs/2610.21870)

    该论文针对通用风险度量类“优化确定性等价（OCE）”下的离线风险敏感强化学习提出了动态与静态两种形式化的悲观价值迭代算法，并建立了紧密的次线性次优性理论界限，首次对该离线场景给出了一般性的理论保证。

    

    风险敏感强化学习（RL）旨在处理不确定性下的决策问题以及不利结果带来的影响。当与环境进行额外交互的代价高昂时，从固定数据集中学习风险敏感策略就显得尤为重要。现有的风险敏感强化学习研究主要集中在线上场景或特定的风险度量上，导致对其离线场景的一般性理论理解仍不完善。在这项工作中，我们利用优化确定性等价（OCE）——一类通用的风险度量——来研究高效的离线风险敏感强化学习。我们考虑了两种形式化方式：动态OCE强化学习，它在每一步递归地应用OCE；以及静态OCE强化学习，它对累积奖励仅应用一次OCE。在表格型设置下，我们为这两种形式化方法都开发了悲观价值迭代算法，并建立了紧密的次线性次优性界限，明确刻画了由效用函数所编码的风险敏感性如何影响……

    arXiv:2610.21870v1 Announce Type: cross  Abstract: Risk-sensitive reinforcement learning (RL) addresses decision-making under uncertainty and adverse outcomes. Learning risk-sensitive policies from fixed datasets is important when additional interaction with the environment is costly. Existing work on risk-sensitive RL focuses on online settings or specific risk measures, leaving a general theoretical understanding of its offline setting underdeveloped. In this work, we investigate efficient offline risk-sensitive RL using the optimized certainty equivalent (OCE), a general class of risk measures. We consider two formulations: dynamic-OCE RL, which applies OCE recursively at each step, and static-OCE RL, which applies a single OCE to the cumulative rewards. Under the tabular setting, we develop pessimistic value-iteration algorithms for both formulations and establish tight sublinear suboptimality bounds, explicitly capturing how risk sensitivity, encoded by utility functions, affects 
    
[^84]: Lethe：用于评估边遗忘方法的链接推断攻击

    Lethe: Link Inference Attacks For Evaluation of Edge Unlearning Methods

    [https://arxiv.org/abs/2610.21869](https://arxiv.org/abs/2610.21869)

    提出首个全面的边遗忘基准Lethe，利用链接推断攻击在15种方法和8个数据集上进行评估，并揭示现有评估方法的系统性缺陷（专用方法可能比重新训练更慢、准确率指标无参考价值）。

    

    图遗忘（Graph Unlearning, GU）旨在无需重新训练的情况下，从图神经网络中移除特定训练数据的影响。尽管大多数方法针对节点和特征的移除，但边遗忘（Edge Unlearning, EU）——即遗忘边——由于在模型中完全消除一条边的影响十分困难，最近作为一个独立问题受到了专门关注。尽管如此，除了在Cora和Citeseer等特征丰富的数据集上的孤立评估（其中图神经网络利用的是节点特征而非图结构）之外，目前尚不存在专门的EU基准。因此，我们揭示了现有评估中的两个系统性局限：专门的EU方法可能比重新训练更慢，且由于边移除的影响微乎其微，准确率指标缺乏参考价值。为填补这一空白，我们提出了Lethe，这是首个全面的边遗忘基准，涵盖15种方法和8个数据集。我们的基准强调大规模……

    arXiv:2610.21869v1 Announce Type: new  Abstract: Graph Unlearning (GU) aims to remove the influence of specific training data from a Graph Neural Network without retraining anew. While most methods address removal of nodes and features, Edge Unlearning (EU), aiming at forgetting edges, has recently attracted dedicated attention as a standalone problem due to the hardness of completely removing the impact of an edge in the model. Despite this, no dedicated benchmark for EU exists besides isolated evaluations on feature-rich datasets such as Cora and Citeseer where GNNs exploit node features rather than graph structure. As such, we show two systematic limitations in existing evaluations: dedicated EU methods can be slower than retraining anew, and accuracy metrics are uninformative due to the negligible impact of edge removal. To address this gap, we introduce Lethe, the first comprehensive benchmark for Edge Unlearning, featuring 15 methods and 8 datasets. Our benchmark emphasizes large
    
[^85]: 基于隐式神经表示的时间周期图像配准

    Temporal Periodic Image Registration with Implicit Neural Representations

    [https://arxiv.org/abs/2610.21845](https://arxiv.org/abs/2610.21845)

    该论文提出Temporal-IDIR方法，通过优化单个隐式神经表示同时建模时间图像序列中所有帧的形变，并引入时间一致性损失实现自正则化，在DIR-LAB数据集上取得了1.03毫米的目标配准误差。

    

    隐式神经表示（INRs）近年来因其能够使用简单轻量的网络对函数进行建模而广受欢迎。这些方法在可变形图像配准中也已被证明是有效的。在本工作中，我们将用于可变形图像配准的INRs扩展到时间图像配准领域，重点关注周期性时间图像序列。我们的方法Temporal-IDIR通过优化单个INR来同时建模时间图像序列中所有帧的形变，从而能够通过其自身的形变预测实现自正则化。为实现这一目标，我们引入了一种时间一致性损失，用于惩罚直接从源到目标的变换与经过中间帧的变换之间的差异。我们在DIR-LAB数据集上评估了该框架，以标注地标与移动地标之间的目标配准误差（TRE）作为评估指标。在此，我们实现了1.03毫米的TRE……

    arXiv:2610.21845v1 Announce Type: cross  Abstract: Implicit Neural Representations (INRs) have recently gained popularity for their ability to model functions using simple and lightweight networks. These methods have also proven effective in deformable image registration. In this work, we extend INRs for deformable image registration to the domain of temporal image registration, focusing on periodic temporal image sequences. Our approach, Temporal-IDIR, optimizes a single INR to model deformations across all frames in a temporal image sequence simultaneously, allowing for self-regularization through its own deformation predictions. To achieve this, we introduce a temporal consistency loss that penalizes discrepancies between direct source-to-target transformations and those traversing intermediate frames. We evaluate our framework on the DIR-LAB dataset, using the target registration error (TRE) between annotated and moved landmarks as the metric. Here, we achieve a TRE of 1.03 mm, out
    
[^86]: 面向实时边缘推理的电网基础模型物理感知蒸馏

    Physics-Aware Distillation of a Grid Foundation Model for Real-Time Edge Inference

    [https://arxiv.org/abs/2610.21833](https://arxiv.org/abs/2610.21833)

    该论文提出一种结合物理残差迁移的蒸馏方法，将电网基础模型参数压缩32.5倍（从2010万降至62万参数），使潮流计算能够在边缘设备上于单个PMU报告帧内实时完成。

    

    电网基础模型（GridFM）是在掩码潮流目标上预训练的图Transformer，其潮流计算速度比迭代求解器快几个数量级。然而，其模型规模面向数据中心推理，而低惯量电力系统运行日益要求在边缘、紧邻测量装置处进行推理，且批大小为1，并在单个同步相量测量单元（PMU）报告帧内完成。我们将一个具有2010万参数、12层的GridFM教师模型蒸馏为一个62万参数、6层的学生模型，实现32.5倍的参数缩减，同时消息传递深度减半，所用训练目标将掩码潮流损失与输出、物理残差、隐状态及注意力迁移相结合。其中，物理残差项用于匹配教师模型的功率残差分布，此前尚未被应用于类型化的电网图。我们在IEEE 118节点基准电网的一个可再生能源变体上进行了评估，该电网中26台常规（同步）

    arXiv:2610.21833v1 Announce Type: new  Abstract: Grid foundation models (GridFMs) are graph transformers pretrained on a masked power-flow objective, and evaluate power flow orders of magnitude faster than an iterative solver. Their size, however, targets data-centre inference, whereas low-inertia operation increasingly asks for inference at the edge, next to the measurement, at batch size one and within a single synchrophasor (PMU) reporting frame. We distil a 20.1M-parameter, 12-layer GridFM teacher into a 0.62M-parameter, 6-layer student, a $32.5\times$ parameter reduction at half the message-passing depth, using an objective that combines the masked power-flow loss with output, physics-residual, latent-state and attention transfer. The physics-residual term, which matches the teacher's power-residual profile, has not previously been applied to typed power-grid graphs. We evaluate on a renewable variant of the IEEE 118-bus benchmark grid in which 26 of the conventional (synchronous)
    
[^87]: 局部轮次作为最优停止：联邦学习中的理论与证据

    Local Epochs as Optimal Stopping: Theory and Evidence in Federated Learning

    [https://arxiv.org/abs/2610.21832](https://arxiv.org/abs/2610.21832)

    该论文将联邦学习中的局部训练轮次选择建模为最优停止问题，证明了单阶段前瞻阈值规则的最优性，并据此提出 FedStop 算法——它根据训练阶段而非客户端自适应地分配局部训练预算，平均准确率不低于调优后的 FedAvg。

    

    每个客户端在通信之前应该训练多久？联邦学习通常以一个固定的、手工调优的局部轮次数来回答这个问题，然而多训一个轮次只有在它消除的损失超过其引入的客户端漂移时才有益。我们将局部训练建模为有限时域的最优停止问题：只要下一轮次的期望损失下降超过其定价后的平方漂移，客户端就继续训练。我们证明，只要停止决策是嵌套的，这一单阶段前瞻阈值就是最优的；推导了几何型收益和幂律型收益下的闭式解；并表明估计得到的规则只会因符号错误的决策而损失效用。轮次层面的证据支持该模型的结构（几何递减收益与加性漂移），并揭示出一个简单的规律：决定局部训练预算的是训练所处的阶段，而非客户端本身。由此得到的规则 FedStop 在平均准确率上从不比经过调优的 FedAvg 低超过一个百分点；在其预算发生变化之处，它提升了 w……（摘要原文在此处截断）

    arXiv:2610.21832v1 Announce Type: new  Abstract: How long should each client train before communicating? Federated learning usually answers with a fixed, hand-tuned number of local epochs, yet an extra epoch helps only while the loss it removes outweighs the client drift it adds. We cast local training as finite-horizon optimal stopping: a client trains while the next epoch's expected loss reduction exceeds its priced squared drift. We prove that this one-stage look-ahead threshold is optimal whenever stopping decisions are nested, derive closed forms for geometric and power-law returns, and show that an estimated rule loses utility only through wrong-signed decisions. Epoch-level evidence supports the model's structure (geometric diminishing returns and additive drift) and reveals a simple regularity: the stage of training, not the client, sets the budget. The resulting rule, FedStop, is never more than one point below tuned FedAvg in mean accuracy. Where its budgets vary, it raises w
    
[^88]: 为定制DNN加速器编译多阈值算子以支持量化的非单调与细粒度激活函数

    Compiling Multi-Threshold Operators for Quantized Non-Monotonic and Fine-Granular Activations in Custom-Tailored DNN Accelerators

    [https://arxiv.org/abs/2610.21825](https://arxiv.org/abs/2610.21825)

    本文将多阈值表示形式化并扩展至非单调激活函数与多维细粒度阈值参数，提出基于代数规则并集成到ONNX编译器基础设施中的系统化自动转换方法，从而在定制DNN加速器上实现更广泛的量化激活函数高效硬件加速。

    

    对神经网络进行激进量化后，激活函数可能成为高效硬件加速的瓶颈。将量化激活函数实现为多阈值操作已被证明是一种有前景的优化方法，能够实现少比特激活函数的高效硬件实现。然而，当前的实现仅限于单调递增函数、具有一维（逐通道）粒度的量化方案以及少量预定义的激活-量化器组合。我们对多阈值表示进行了形式化，将其扩展到非单调函数和细粒度参数，即多维阈值。为了将量化函数组合自动转换为阈值表示，我们提出了一种系统化方法，通过集成到基于ONNX的编译器基础设施中的代数规则来实现。我们通过实验在一系列单调激活函数上验证了我们的解决方案。

    arXiv:2610.21825v1 Announce Type: cross  Abstract: Aggressive quantization of neural networks can expose activation functions as bottlenecks for efficient hardware acceleration. Implementing quantized activations as multi-threshold operations has been shown to be a promising optimization, allowing for efficient hardware implementation of few-bit activations. However, current implementations are limited to monotonically increasing functions, quantization schemes with one-dimensional (per-channel) granularity and a small set of predefined activation-quantizer combinations. We formalize the multi-threshold representation to extend it to non-monotonic functions and fine-granular parameters, i.e., multi-dimensional thresholds. For automated conversion of quantized function compositions into threshold representation, we propose a systematic approach via algebraic rules integrated into the ONNX-based compiler infrastructure. We experimentally validate our solution on a selection of monotonic 
    
[^89]: 深度线性网络中神经坍缩的闭式深度-宽度定律

    A Closed Form Depth-Width Law for Neural Collapse in Deep Linear Networks

    [https://arxiv.org/abs/2610.21820](https://arxiv.org/abs/2610.21820)

    该论文首次在深度线性网络中推导出神经坍缩的闭式深度-宽度定律，证明在大宽度极限下类内变异性的累积压缩仅依赖于深度与宽度之比 L/N，且压缩速率由数据、先验和架构自洽决定。

    

    神经网络的深度和宽度如何塑造学习到的表示，通常只能在训练之后才被研究。我们证明，在一个易于处理的特征学习设定中，这种依赖关系可以以闭式形式获得。以神经坍缩作为表示几何的探针，我们直接用学习到的特征核来表达 NC1—NC3，将已知的 NC1 恒等式推广到 NC2 的角度部分，以及在岭回归读出下的 NC3。对于具有可交换类别结构的深度线性网络，由此得到的核方程给出了一个几何定律（在理论范围内是精确的），描述了类内变异性的逐层演化。在大宽度极限下，累积压缩成为深度-宽度比 $L/N$ 的函数，其速率由数据、先验和架构自洽地确定。数值实验支持所推导出的对深度、宽度和信噪比的依赖关系。其结果是一个表示层面的……

    arXiv:2610.21820v1 Announce Type: new  Abstract: How depth and width of neural networks shape learned representations is usually studied only after training. We show that, in a tractable feature-learning setting, this dependence can instead be obtained in closed form. Using neural collapse as a probe of representation geometry, we express NC1--NC3 directly in terms of the learned feature kernel, extending an identity known for NC1 to the angular part of NC2 and to NC3 under a ridge readout. For deep linear networks with exchangeable class structure, the resulting kernel equations yield a geometric law, exact within the theory, for the layerwise evolution of within-class variability. At large width, the accumulated compression becomes a function of the depth-to-width ratio $L/N$, with its rate determined self-consistently by the data, prior and architecture. Numerical experiments support the derived dependence on depth, width and signal-to-noise ratio. The result is a representation-lev
    
[^90]: 测量学习率迁移曲线的局部形状

    Measuring the local shape of learning-rate transfer curves

    [https://arxiv.org/abs/2610.21814](https://arxiv.org/abs/2610.21814)

    本文提出两个关系方程——一个在稳定性边缘成立的理论公式和一个将学习率梯度视为超参数梯度的近似表达式——从经验上很好地解释了学习率迁移曲线的梯度，为定量刻画学习率曲线的局部形状提供了方法。

    

    理解训练后模型性能对学习率（LR）的依赖关系对于更好地优化神经网络至关重要，但合适的学习率大小取决于学习动力学、架构及其隐含的偏置，因此我们对这种复杂依赖关系的理解仍然有限。近年来，通过在不同设置（如不同模型规模）之间进行学习率迁移的大量实验，人们观察到了损失曲线作为学习率函数的行为，探索定量刻画这种学习率曲线形状的方法正变得愈发重要。在这项工作中，我们提供了两个关系方程，能够对学习率迁移曲线的梯度（即学习率梯度）给出良好的经验解释。其中一个是基于稳定性边缘成立的理论公式，另一个是通过将该梯度简单地视为超参数梯度而得到的近似表达式。利用这两个公式，本文为定量理解与预测学习率曲线的局部形状提供了方法论基础。

    arXiv:2610.21814v1 Announce Type: new  Abstract: Understanding the learning-rate (LR) dependence of trained model performance is essential for better optimization of neural networks, but the appropriate size of the LR depends on learning dynamics, architecture, and their hidden biases, and thus our understanding of this complex dependence remains limited. The behavior of the loss curve as a function of the LR has recently been observed in extensive experiments on LR transfer across different settings, such as model sizes, and it is becoming more important to explore methodologies for quantitatively characterizing the shape of this LR curve. In this work, we provide two relational equations that give a good empirical explanation of the gradient of the LR transfer curve, i.e., the LR gradient. One is a theoretical formula that holds at the edge of stability, and the other is an approximate expression obtained by naively treating this gradient as a hyper-parameter gradient. Using these tw
    
[^91]: 二比特的N-Gram记忆：大语言模型中查找表的量化

    N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models

    [https://arxiv.org/abs/2610.21810](https://arxiv.org/abs/2610.21810)

    本文提出条件程序（CP）压缩表示方法，仅用两比特编码按需重构大语言模型n-gram记忆查找表中被请求的行，相比标准2比特量化将输出KL散度降低41–63%，并将DeepSeek的表存储从203GB压缩至51GB。

    

    条件记忆通过选择性查找来扩展语言模型的容量，但完整的嵌入表必须始终可用以应对每一个可能的输入。我们提出了条件程序，这是一种压缩表示方法，只需重建输入所请求的行。CP用短码和局部尺度来表示每一行，并使用一个在整个表上共享的重构程序。我们通过高斯格、范数恢复以及可独立解码的行记录来实例化这种表示。CP的两比特码相比2比特最近取整量化，将输出KL散度降低了41–63%。对三个经过训练的n-gram记忆族的分析表明，类高斯的坐标统计与行之间学习到的关系并存；在相同的存储预算下，共享格结构的表现优于我们测试的拟合码本和低秩残差变体。对于DeepSeek，该方法将表存储从203GB降至51GB，为更大的工作负载留出了空间。

    arXiv:2610.21810v1 Announce Type: cross  Abstract: Conditional memory expands language-model capacity through selective lookup, but the full embedding table must remain available for every possible input. We introduce Conditional Program (CP), a compressed representation that reconstructs only the rows an input requests. CP represents each row by a short code and a local scale, with a reconstruction program shared across the table. We instantiate this representation with a Gaussian trellis, norm restoration and independently decodable row records. CP's two-bit code lowers output KL by 41--63\% relative to 2-bit round-to-nearest. Analysis across three trained n-gram memory families shows that Gaussian-like coordinate statistics coexist with learned relationships among rows; at matched storage budgets, the shared trellis outperforms the fitted codebooks and low-rank residual variants we test. For DeepSeek, it reduces table storage from 203GB to 51GB, leaving room for larger workloads whe
    
[^92]: CIM-Embed：学习面向聚类的指令条件化多模态嵌入

    CIM-Embed: Learning Instruction-Conditioned Multimodal Embeddings for Clustering

    [https://arxiv.org/abs/2610.21805](https://arxiv.org/abs/2610.21805)

    本文提出CIM-Embed框架，通过监督对比学习将图像、文本和指令直接映射为多模态嵌入，使同一组对象能根据不同指令形成不同的聚类划分，且无需预先知道聚类数目。

    

    多模态对象在不同指令下可能需要不同的划分方式，例如电商目录中的产品可以按颜色、材质或两者兼有的方式进行分组。面向检索的嵌入可能无法恢复这些划分，而生成式聚类通常依赖于中间描述或迭代决策。本文提出了CIM-Embed，一个用于学习面向聚类的指令条件化多模态嵌入的框架。监督对比学习使用同组正样本以及来自其他组的负样本，并可选地纳入其他任务；异构结构化批次可适应具有不同组数的任务。在推理阶段，模型将每个对象的图像、文本和指令直接映射为一个嵌入向量，随后由HDBSCAN在无需真实聚类数目的情况下形成划分。我们基于Amazon Reviews 2023中的产品元数据构建了ProductMIC数据集，包含训练数据以及用于指令引导聚类的评估基准。

    arXiv:2610.21805v1 Announce Type: new  Abstract: Multimodal objects can require different partitions under different instructions, as when products in an e-commerce catalog are grouped by color, material, or both. Retrieval-oriented embeddings may not recover these partitions, while generative clustering often relies on intermediate descriptions or iterative decisions. This paper presents CIM-Embed, a framework for learning instruction-conditioned multimodal embeddings for clustering. Supervised contrastive learning uses same-group positives and negatives from other groups, optionally including other tasks; heterogeneous structured batches accommodate tasks with different group counts. At inference, the model maps each object's image, text, and instruction directly to an embedding; HDBSCAN then forms a partition without the ground-truth cluster count. We construct ProductMIC from product metadata in Amazon Reviews 2023, comprising training data and an evaluation benchmark for instructi
    
[^93]: 天空图像与晴空指数的单阶段联合扩散用于超短期辐照度预测：帧到辐照度阶段所丢弃的信息

    Single-stage joint diffusion of sky images and clear-sky index for ultra-short-term irradiance forecasting: what the frame-to-irradiance stage discards

    [https://arxiv.org/abs/2610.21795](https://arxiv.org/abs/2610.21795)

    该论文提出一种在单一扩散过程中联合去噪未来天空图像与晴空指数的单阶段辐照度预测模型，省去了传统两阶段设计中的独立回归器且无需实测辐照度输入，在仅有相机的站点上实现了超越十二个基线配置的超短期辐照度预测性能。

    

    天空图像辐照度预测目前以两阶段设计为主：由视频预测模型预测未来的天空图像帧，再由一个单独训练的回归器将这些帧映射为辐照度。我们提出一种单阶段替代方案，在同一个扩散过程中对未来的天空图像序列和晴空指数序列进行联合去噪，其中辐照度头读取的是未来天空的去噪预览，而非带噪声的扩散状态。该预测器不需要回归阶段，仅依靠天空图像和确定性协变量运行，输入中不含任何实测辐照度——这正是有相机但没有总辐射表的站点所面临的实际情形。在2023年SIRTA数据上，该模型达到了104.1 W m⁻²的汇总RMSE，以连续排序概率得分（CRPS）衡量的相对智能持续性模型的技巧得分为+0.34，高于全部十二个基线配置，并且在2021-2022年数据上重新训练后仍能保持该技巧水平。在该模型自身生成的……

    arXiv:2610.21795v1 Announce Type: cross  Abstract: Sky-image irradiance forecasting is dominated by a two-stage design: a video-prediction model forecasts future sky frames, and a separately trained regressor maps those frames to irradiance. We propose a single-stage alternative that co-denoises the future sky-image and clear-sky-index sequences in one diffusion process, with an irradiance head that reads a denoised preview of the future sky rather than the noised diffusion state. The forecaster needs no regression stage and runs from sky images and deterministic covariates alone, with no measured irradiance at its input --- the situation at a site with a camera but no pyranometer. On 2023 SIRTA data the model reaches a pooled RMSE of $104.1$~W\,m$^{-2}$ and a skill score of $+0.34$ on the continuous ranked probability score (CRPS) against smart persistence, higher than any of twelve baseline configurations, and retains that skill when retrained on 2021--2022. On the model's own genera
    
[^94]: 相同频谱，更难的一阶训练：单比特与多比特重复上传电路中的相位锁定

    Same Spectrum, Harder First-Order Training: Phase-Locking in Single- versus Multi-Qubit Re-Uploading Circuits

    [https://arxiv.org/abs/2610.21778](https://arxiv.org/abs/2610.21778)

    单比特重复上传电路虽与多比特电路在相同编码预算下具有同等的傅里叶表达能力，但因布洛赫球约束产生的“相位锁定”使其切方向至多只有 $2E+2$ 个（远少于双比特对照电路的 $6E+6$ 个），导致一阶训练明显更困难。

    

    角度编码的变分量子电路实现的是截断傅里叶级数。对于一维输入，单比特重复上传（串行）电路能够表示具有相同编码预算 $E$ 的多比特电路所能表示的所有函数；我们发现在一阶训练中，这种“以宽度换深度”的交换并非没有代价。由于串行电路的状态生活在布洛赫球上，其所有参数在每个输入点只能沿一个复数方向移动状态，我们将这一现象称为“相位锁定”（phase-locking）。我们证明这不会损失任何傅里叶方向，但也没有富余的方向：无论参数有多少，串行电路至多只有 $2E+2$ 个切方向，仅比其 $2E+1$ 个傅里叶系数所需的多一个，而具有相同编码预算和模型类的双比特对照电路通常至少拥有 $6E+6$ 个切方向。训练代价正是相对于该对照来衡量的，即在100次运行中在给定预算内达到最优值的次数……

    arXiv:2610.21778v1 Announce Type: cross  Abstract: Angle-encoded variational quantum circuits implement truncated Fourier series. For one-dimensional inputs, a single-qubit re-uploading (serial) circuit represents every function that a multi-qubit circuit with the same encoding budget $E$ can; we find that this exchange of width for depth is not free in first-order training. Because its state lives on the Bloch sphere, all parameters of the serial circuit move the state, at each input, along one complex direction, which we call \emph{phase-locking}. We prove that this costs no Fourier direction but leaves no direction to spare: however many parameters it has, the serial circuit has at most $2E+2$ tangent directions, one more than its $2E+1$ Fourier coefficients need, while a two-qubit control with the same encoding budget and model class generically has at least $6E+6$. The cost in training is measured against this control, as the number of runs out of 100 that reach the optimum within
    
[^95]: Q($\gamma_H$)：多折扣Q学习中的熵引导视界混合

    Q($\gamma_H$): Entropy-Guided Horizon Mixing in Multi-Discount Q-Learning

    [https://arxiv.org/abs/2610.21764](https://arxiv.org/abs/2610.21764)

    提出多视界Q学习框架Q($\gamma_H$)，通过维护多个折扣专用输出头并利用熵引导的置信度信号在控制时自适应混合，在不修改Bellman更新的情况下实现了轻量级的自适应规划视界控制，尤其在延迟信用任务上改进显著。

    

    Q学习中的固定折扣因子在整个训练过程中强加了单一的计划视界，尽管早期探索与长期信用分配的需求各不相同。我们提出了Q($\gamma_H$)，这是一个多视界Q学习框架，它维护多个特定折扣的输出头，并在控制时使用由熵导出的置信度信号自适应地组合它们。由此产生的状态相关混合权重会随着策略置信度的增加而将控制从较短视界的输出头转向较长视界的输出头，从而在不修改单个Bellman更新的情况下实现自适应的视界混合。在经典控制和深度强化学习基准测试中，Q($\gamma_H$)相对于强大的固定折扣基线，在延迟信用任务上表现出最显著的改进。通过将视界自适应与底层学习更新解耦，Q($\gamma_H$)为强化学习中的自适应计划视界控制提供了一个轻量级框架。

    arXiv:2610.21764v1 Announce Type: new  Abstract: A fixed discount factor in Q-learning imposes a single planning horizon throughout training, despite the differing demands of early exploration and long-term credit assignment. We propose Q($\gamma_H$), a multi-horizon Q-learning framework that maintains multiple discount-specific heads and adaptively combines them at control time using an entropy-derived confidence signal. The resulting state-dependent mixing weights shift control from shorter- to longer-horizon heads as policy confidence increases, enabling adaptive horizon mixing without modifying individual Bellman updates. Across classical control and deep reinforcement-learning benchmarks, Q($\gamma_H$) shows its strongest improvements on delayed-credit tasks relative to strong fixed-discount baselines. By decoupling horizon adaptation from the underlying learning updates, Q($\gamma_H$) provides a lightweight framework for adaptive planning-horizon control in reinforcement learning
    
[^96]: DeciCAD：面向CAD智能体的基于执行框架的动作策略学习

    DeciCAD: Harness-Grounded Learning of Action Policies for CAD Agents

    [https://arxiv.org/abs/2610.21755](https://arxiv.org/abs/2610.21755)

    DeciCAD提出了一个面向CAD的领域原生智能体执行框架，通过从交互轨迹中提取因果决策监督，并结合监督微调与几何在线强化学习训练动作策略，在四个CAD基准上取得最优的F1与IoU，同时将模型输出token减少45%至88%。

    

    大语言模型智能体通过闭环动作策略运行，其决策空间由智能体执行框架所定义。我们研究了这一运行空间如何塑造计算机辅助设计（CAD）领域中的智能体行为与策略学习——CAD是一个以持久状态、几何约束和长时程流程为特征的专业领域。我们将这一视角具体化为DeciCAD，一个领域原生的执行框架，它提供语义化的CAD动作、面向决策的观测、环境反馈以及持久的状态转移。基于该框架，我们构建了一套学习流水线，从交互轨迹中提取因果决策监督信号，并通过监督微调（SFT）和基于几何的在线强化学习（RL）来改进策略。在四个CAD基准测试中，DeciCAD取得了最优的F1分数和交并比（IoU），同时相比通用框架将模型输出的token数量减少了45%至88%。

    arXiv:2610.21755v1 Announce Type: new  Abstract: Large Language Model agents operate through closed-loop action policies whose decision space is defined by an agent harness. We investigate how this operating space shapes agent behavior and policy learning in computer-aided design (CAD), a professional domain characterized by persistent state, geometric constraints, and long-horizon procedures. We instantiate this perspective in DeciCAD, a domain-native harness that exposes semantic CAD actions, decision-oriented observations, environment feedback, and persistent state transitions. Building on this harness, we develop a learning pipeline that extracts causal decision supervision from interaction trajectories and improves the policy through supervised fine-tuning (SFT) and geometry-grounded online reinforcement learning (RL). Across four CAD benchmarks, DeciCAD achieves the best F1 and intersection over union (IoU) while reducing model-output tokens by 45--88% relative to a general-purpo
    
[^97]: 基于概念的模型中的有害部分干预与良性泄漏

    Harmful Partial Interventions and Benign Leakage in Concept-based Models

    [https://arxiv.org/abs/2610.21750](https://arxiv.org/abs/2610.21750)

    该论文形式化了基于概念的模型中概念干预反直觉地恶化下游预测的“干预伤害”问题，区分了全量干预与更贴近实际的部分干预两种情形，并证明“充分性”和“局部化”这两个标准不足以保证部分干预下概念泄漏是良性的。

    

    部署在高风险环境中的人工智能系统越来越需要能够支持有意义的人类监督的模型。基于概念的模型为用户提供了一个有前景的框架，使用户能够检查和干预中间的、人类可理解的概念。然而，概念干预可能会反直觉地恶化下游预测，我们将这种失败模式形式化为“干预伤害”。我们区分了在所有概念都被干预后的伤害与部分干预下的伤害，后者是更贴近实际的常见使用场景。现代概念模型通常通过在中间概念表示中编码（即泄漏）额外的任务相关信息来实现高准确率。最近的研究认为，当概念表示满足两个条件——充分性和局部化——时，这种泄漏对于干预伤害而言可以是良性的。我们证明这些标准不足以保证概念干预……（原文摘要在此处被截断）

    arXiv:2610.21750v1 Announce Type: cross  Abstract: Artificial intelligence systems deployed in high-risk settings increasingly require models that support meaningful human oversight. Concept-based models (CMs) offer a promising framework by enabling users to inspect and intervene on intermediate, human-understandable concepts. However, concept interventions can counterintuitively worsen downstream predictions, a failure mode we formalise as intervention harm. We distinguish harm after all concepts have been intervened on from harm under partial interventions, the more likely practical use case. Modern CMs often achieve high accuracy by encoding, or leaking, additional task-relevant information in their intermediate concept representations. Recent work argues that such leakage can be benign with respect to intervention harm when concept representations satisfy two conditions: sufficiency and localisation. We prove that these criteria are not sufficient to guarantee that concept interven
    
[^98]: 委托困境：规模经济、问责制与多中心治理

    The Delegation Dilemma: Economies of Scale, Accountability, and Polycentric Governance

    [https://arxiv.org/abs/2610.21746](https://arxiv.org/abs/2610.21746)

    该论文通过重复博弈模型揭示了委托治理中的“规模—问责”权衡：更大的群体虽带来更多成本分担收益，却削弱了中心保留单个成员的激励，从而内生地决定了可持续的非剥削性委托的群体规模上限。

    

    许多集体行动系统既不是完全去中心化的，也不是完全中心化的：参与者自愿组织成共享组织成本、并将决策委托给代表机构的制度。委托可以降低个体成本，但也会将控制权转移给代表，而代表的私人目标未必与其委托人一致。我们在一个存在内生委托给自利中心的重复博弈中研究这种张力。一个简单的简化形式模型得出了规模—问责权衡：更大的委托群体为成员提供更大的成本分担收益，但对中心而言，保留任何单个成员的边际价值随着群体规模的增大而下降。因此，个体退出的可能性只能在某个内生的问责群体规模上限之内维持非剥削性的委托关系。此外，我们还识别出在可行条件下能够实现完全委托的问责治理的参数区域。

    arXiv:2610.21746v1 Announce Type: cross  Abstract: Many collective-action systems are neither fully decentralized nor fully centralized: agents voluntarily organize into institutions that share organizational costs and delegate decisions to representatives. Delegation can reduce individual costs, but it also transfers control to representatives whose private objectives need not align with those of their constituents. We study this tension in a repeated game with endogenous delegation to self-interested centers. A simple reduced-form model yields a scale--accountability tradeoff: larger constituencies offer greater cost-sharing benefits to members, yet the marginal value to a center of retaining any one constituent decreases as group size grows. Consequently, the prospect of individual exit can sustain non-exploitative delegation only up to an endogenous accountable constituency size. Additionally, we identify parameter regions in which fully delegated, accountable governance, when feas
    
[^99]: 成对AUC优化需要纠错能力：一个统一的视角

    Pairwise AUC Optimization Needs Corrective Power: A Unified View

    [https://arxiv.org/abs/2610.21743](https://arxiv.org/abs/2610.21743)

    本文发现直接成对AUC训练可能进入严重排序错误频发但丧失纠错能力的退化区域，并通过U-统计量的退化性质为该现象提供了统一的理论解释。

    

    直接成对AUC代理训练之所以具有吸引力，是因为它朝着测试时所使用的排序指标进行优化，在类别不平衡场景下尤其如此。这在训练目标与评估指标之间建立了直接的对齐关系，但仅凭这种指标对齐并不能保证随机训练始终具有纠错性。我们通过“纠错能力”来研究这一差距：即成对小批量更新能否为更严重的正负样本排序错误带来更强、更可靠的改进。我们的主要观察是，直接成对训练可能会进入某些区域，在这些区域中严重错误依然普遍存在，但训练不再产生有效的纠错运动。在这样的区域中，即使是通常能够帮助SGD沿着最陡峭的严重错误纠正方向移动的小批量噪声，也会变得弱得多。本文通过探索由此形成的U-统计量的退化性质，为这一现象寻求理论解释。

    arXiv:2610.21743v1 Announce Type: new  Abstract: Direct pairwise AUC-surrogate training is attractive because it trains toward the ranking metric used at test time, especially under class imbalance. This creates a direct alignment between the training objective and the evaluation metric, but such metric alignment alone does not guarantee that stochastic training remains corrective. We study this gap through Corrective Power: whether pairwise mini-batch updates produce stronger and more reliable improvements for more severe positive-negative ranking errors. Our main observation is that direct pairwise training can enter regions where severe errors remain common but no longer produce useful corrective motion. In such regions, even the mini-batch noise that would normally help SGD move along the steepest severe-error correction direction can become much weaker. In this paper, we seek a theoretical explanation for this by exploring the degenerate property of the U-statistics formed from th
    
[^100]: 基于Smooth-L1正则化的稀疏联邦回声状态网络时间序列分类方法

    Sparse Federated Echo State Networks for Time-Series Classification via Smooth-L1 Regularization

    [https://arxiv.org/abs/2610.21742](https://arxiv.org/abs/2610.21742)

    本文提出基于Smooth-L1正则化的稀疏回声状态网络SL1ESN，利用与牛顿法兼容的可微稀疏化方法实现读出层高度稀疏，并支持二阶联邦聚合，在七个时间序列分类基准上达到75%–95%的稀疏度。

    

    回声状态网络（ESN）作为储备池计算的轻量级实现，为资源受限环境中的时间数据处理提供了高效的解决方案。虽然标准ESN具有计算效率优势，但其密集的读出层会增加存储模型的大小和逐样本的推理成本。本文提出了一种Smooth-L1正则化的ESN（SL1ESN），在保持基于梯度优化的可微性的同时，促进读出层的稀疏性。该方法采用二次可微的Smooth-L1近似，与牛顿法兼容，我们将其既用于集中式训练，也用于二阶（牛顿步）联邦聚合方案。在所有数据集和两种设置下共享单一固定收缩阈值的情况下，在七个时间序列分类基准上的实验表明，SL1ESN学习到的读出层在集中式设置中稀疏度达到77%–95%，在联邦设置中达到75%–95%。

    arXiv:2610.21742v1 Announce Type: new  Abstract: Echo State Networks (ESNs), as lightweight implementations of reservoir computing, provide efficient solutions for temporal data processing in resource-constrained environments. While standard ESNs offer computational efficiency, their dense readout layers inflate the stored model and the per-sample inference cost. This paper introduces a Smooth-L1 regularized ESN (SL1ESN) that promotes sparsity in the readout layer while preserving differentiability for gradient-based optimization. The method employs a twice-differentiable Smooth-L1 approximation compatible with Newton's method, which we exploit both for centralized training and for a second-order (Newton-step) federated aggregation scheme. Under a single fixed shrinkage threshold shared by every dataset and both settings, experiments on seven time-series classification benchmarks show that SL1ESN learns readouts that are $77$--$95\%$ sparse in the centralized setting and $75$--$95\%$ s
    
[^101]: 面向持续学习智能体的可靠知识获取

    Reliable Knowledge Acquisition for Continual Learning Agents

    [https://arxiv.org/abs/2610.21739](https://arxiv.org/abs/2610.21739)

    提出RIKA框架，通过分层提取事实、概念、程序和元认知四类知识，并引入基于来源证据与后续经验的证据准入机制，解决现有经验学习系统中12%–33%的条目会损害未来任务性能的知识可靠性问题。

    

    随着由基础模型驱动的智能体进入开放式环境，从经验中进行持续学习变得至关重要。对于基于固定参数模型构建的智能体而言，这种学习需要在上下文中复用先前的任务经验，以提升未来任务的表现和效率。现有的记忆与经验学习系统通常只关注保留的事实或程序性内容，缺乏一种机制来确立从持续任务流中提取的论断的可靠性。事实上，我们发现代表性系统所产生的条目中约有12%–33%在被复用时反而会损害未来任务的表现。为解决这些局限，我们提出了RIKA（可靠知识获取，Reliable Knowledge Acquisition），该方法将事实性、概念性、程序性和元认知知识的分层提取与基于证据的知识准入机制相结合，利用来源证据和后续经验来控制哪些论断能够进入知识库……

    arXiv:2610.21739v1 Announce Type: cross  Abstract: As agents powered by foundation models move into open-ended environments, continual learning from experience becomes essential. For agents built on models with fixed parameters, this learning requires reusing prior task experience in context to improve performance and efficiency on future tasks. Existing memory and experience-learning systems typically focus on either retained facts or procedures, and lack a mechanism for establishing the reliability of claims extracted from the ongoing task stream. Indeed, we find that approximately 12--33\% of entries produced by representative systems harm future task performance when reused. To address these limitations, we propose \textbf{RIKA} (Reliable Knowledge Acquisition), which combines hierarchical extraction of factual, conceptual, procedural, and metacognitive knowledge with Evidence-based Knowledge Admission, which uses source evidence and subsequent experience to govern which claims ent
    
[^102]: FIRM：面向联邦大语言模型遗忘的Fisher信息驱动的保留感知合并

    FIRM: Fisher-Informed Retention Merging for Federated LLM Unlearning

    [https://arxiv.org/abs/2610.21720](https://arxiv.org/abs/2610.21720)

    提出FIRM框架，利用客户端本地Fisher统计量进行归一化合并并对保留敏感参数施加显式控制，解决联邦大语言模型遗忘中的幅度爆炸问题，在单轮服务器端聚合中实现稳定遗忘与保留效用的兼顾。

    

    联邦大语言模型遗忘旨在解决当遗忘数据与保留数据仍分布在各个私有客户端上、服务器无法访问时的定向知识移除问题。由此产生的本地训练适配器编码了相互冲突的目标，使其聚合与传统的联邦优化有本质区别。一个关键的失效模式是幅度爆炸，即未经归一化的跨客户端聚合会增大合并后更新的幅度，并在更强的遗忘过程中损害保留效用。我们提出FIRM（Fisher-Informed Retention Merging，Fisher信息驱动的保留感知合并），这是一个用于稳定且保留感知的服务器端聚合的单轮框架。FIRM利用客户端本地的Fisher统计量作为统一的敏感度信号，构建Fisher归一化的基础合并，并对保留敏感参数施加显式控制。其二阶保留投影具有唯一的最小距离解，而相应的对偶变量随t…（原文摘要至此截断）

    arXiv:2610.21720v1 Announce Type: cross  Abstract: Federated LLM unlearning addresses targeted knowledge removal when forget and retain data remain distributed across private clients and are inaccessible to the server. The resulting locally trained adapters encode conflicting objectives, making their aggregation fundamentally different from conventional federated optimization. A key failure mode is magnitude explosion, whereby unnormalized cross-client aggregation increases the merged update magnitude and degrades retained utility under stronger unlearning. We propose FIRM (Fisher-Informed Retention Merging), a single-round framework for stable and retention-aware server-side aggregation. FIRM uses client-local Fisher statistics as a unified sensitivity signal to construct a Fisher-normalized base merge and impose explicit control over retain-sensitive parameters. Its second-order retention projection admits a unique minimum-distance solution, while the corresponding dual scales with t
    
[^103]: 无监督域自适应在精确区分正常与异常视网膜发育中的国际多中心验证

    International Multi-Centre Validation of Unsupervised Domain Adaptation for Precise Discrimination between Normal and Abnormal Retinal Development

    [https://arxiv.org/abs/2610.21719](https://arxiv.org/abs/2610.21719)

    该研究在国际多中心环境下验证了一种基于ResNet50的无监督域对抗神经网络模型，能够在四种不同OCT设备（包括台式和手持式系统）之间准确区分正常与异常视网膜发育，准确率达87.75%–96.06%，性能可与临床医生分级员相媲美。

    

    视网膜发育停滞可导致儿童视力障碍，但对光学相干层析成像扫描进行分级在不同设备和中心之间难以标准化。我们开发并评估了一种无监督域自适应模型，用于在不同OCT系统之间分类正常与异常的视网膜发育。这项回顾性多中心研究使用了来自四个国家五个中心以及英国生物样本库的OCT扫描数据，数据涵盖四种OCT设备，包括台式和手持式系统。基于ResNet50的域对抗神经网络使用来自源设备的标记图像和来自目标设备的未标记图像进行训练。我们评估了六种源设备到目标设备的配对，并将模型性能与七名临床医生分级员进行比较。另有一个Heidelberg Spectralis数据集被单独保留用于外部评估。在验证集上，准确率范围为87.75%至96.06%，灵敏度范围为88.68%至98.38%，特异性（摘要在此处截断）

    arXiv:2610.21719v1 Announce Type: cross  Abstract: Arrested retinal development can cause childhood visual impairment, but grading optical coherence tomography scans is difficult to standardise across devices and centres. We developed and evaluated an unsupervised domain adaptation model to classify normal and abnormal retinal development across different OCT systems. This retrospective multicentre study used OCT scans from five centres in four countries and the UK Biobank. Data covered four OCT devices, including table mounted and handheld systems. A ResNet50 based Domain Adversarial Neural Network was trained using labelled images from a source device and unlabelled images from a target device. We evaluated six source to target pairings and compared model performance with seven clinician graders. A separate Heidelberg Spectralis dataset was reserved for external evaluation. Across validation sets, accuracy ranged from 87.75% to 96.06%, sensitivity from 88.68% to 98.38%, and specifici
    
[^104]: 基于期望最大化算法的逐元素污染与部分观测数据的鲁棒协方差估计

    Robust Covariance Estimation for Entry-wise Contaminated and Partially Observed Data via Expectation Maximization Algorithm

    [https://arxiv.org/abs/2610.21709](https://arxiv.org/abs/2610.21709)

    本文提出一种在缩减潜空间上运行的随机近似EM（SAEM）算法，通过将坐标级尺度的随机更新与Rao-Blackwell化的高斯条件矩相结合，实现了逐元素污染与缺失数据下Student-$t^*$模型的鲁棒协方差估计，无需对潜在高斯变量显式采样。

    

    arXiv:2610.21709v1 公告类型：cross 摘要：在各种信号处理应用中，逐元素污染和缺失数据下的鲁棒协方差估计是一个具有挑战性的问题。在Student-$t$模型下，观测级重加权会在样本间产生一致的衰减，即使大多数坐标仍包含有用信息，也可能导致累积的信息损失。另一种Student-$t^*$模型可以实现坐标级的鲁棒性，但其E步不存在闭式解。我们通过一个在缩减潜空间上运行的随机近似EM（SAEM）算法，将Student-$t^*$模型的推断扩展到不完整观测情形：将坐标级尺度的随机更新与Rao-Blackwell化的高斯条件矩相结合，在避免对潜在高斯变量进行显式采样的同时，保持了基于似然的框架。在合成数据和真实传感器数据上的实验表明，相比高斯、Student-$t$、插补方法，该方法具有更低的Fisher-Rao误差（摘要在此处被截断）。

    arXiv:2610.21709v1 Announce Type: cross  Abstract: Robust covariance estimation under entry-wise contamination and missing data is challenging in various signal processing applications. Under the Student-$t$ model, observation-level reweighting induces a uniform attenuation across samples, which can lead to cumulative information loss even when most coordinates remain informative. The alternative Student-$t^*$ model enables coordinate-wise robustness, but its E-step is not available in closed form. We extend inference for the Student-$t^*$ model to incomplete observations through a stochastic approximation EM (SAEM) algorithm operating on a reduced latent space: stochastic updates of the coordinate-wise scales are combined with Rao-Blackwellized Gaussian conditional moments, which avoids explicit sampling of latent Gaussian variables while preserving a likelihood-based framework. Experiments on synthetic and real sensor data show a lower Fisher-Rao error than Gaussian, Student-$t$, imp
    
[^105]: 将人类干预作为标签：定位、引导并（有时）编辑 LLM 智能体失败背后的计算

    Human Interventions as Labels: Localizing, Steering, and (Sometimes) Editing the Computations Behind LLM Agent Failures

    [https://arxiv.org/abs/2610.21708](https://arxiv.org/abs/2610.21708)

    本文提出将人类对 LLM 智能体失败行为的干预视为标签，用以因果定位、引导并（有时）编辑失败背后的内部计算。

    

    当 LLM 智能体行为不当时，人类会进行干预，而这种干预通常被丢弃，或仅作为训练数据被回放。我们转而将其视为一种测量：一个指向失败内部计算的标签。我们在受控的智能体环境中研究这一问题，聚焦于一种特定失败（在请求本已明确无歧义时仍提出澄清问题），所用标签来自基于规则的监督者，并通过盲测人类判断加以验证。研究结论覆盖符合条件的模型，即那些既会过度提问、又会在匹配的明确请求上直接行动的模型：十三个模型中有五个符合条件（Qwen2.5-3B、gemma-2-2b、OLMo-3-7B、Mistral-Nemo-12B、Phi-4）。论文提出四项发现。其一，“提问还是行动”的决策在每个符合条件的模型家族中都是因果可定位的：从干预标签中学到的一个方向，在残差流的特定深度上能翻转 72-100% 的留出集过度提问案例，其中位置与强度均在训练数据上选定，而在相同流程下随机方向最多只能翻转一个案例……

    arXiv:2610.21708v1 Announce Type: cross  Abstract: When an LLM agent misbehaves, a human steps in, and that intervention is usually discarded or replayed as training data. We treat it instead as a measurement: a label pointing at the internal computation that failed. We study it in a controlled agent environment, on one failure (asking a clarifying question when the request was already unambiguous), with labels from a rule-based overseer validated against blind human judgment. Claims cover eligible models, which both over-ask and sometimes act on matched unambiguous requests: five of thirteen qualified (Qwen2.5-3B, gemma-2-2b, OLMo-3-7B, Mistral-Nemo-12B, Phi-4). Four findings. First, the ask-vs-act decision is causally localizable in every eligible family: a direction learned from intervention labels flips 72-100% of held-out over-asks at a definite residual-stream depth, with site and dose chosen on training data, and random directions under the same procedure flip at most one episod
    
[^106]: ShiftQAT：面向固定功能边缘CNN加速器数值契约的量化感知训练

    ShiftQAT: Quantization-Aware Training for the Numerical Contract of a Fixed-Function Edge CNN Accelerator

    [https://arxiv.org/abs/2610.21695](https://arxiv.org/abs/2610.21695)

    ShiftQAT是一种面向MAX7800x系列固定功能边缘CNN加速器的量化感知训练流程，通过构造方式生成满足二的幂次对称缩放、8位仅整数运算等数值契约的神经网络，是该系列开源训练工具链ai8x-training的默认训练路径。

    

    固定功能CNN加速器使边缘推理快速且节能，但它们只能执行符合精确数值契约的网络。在MAX7800x系列上，该契约要求采用二的幂次、逐张量、对称的缩放因子，8位激活值与偏置，仅整数的算术运算，以及每层一个静态配置的移位操作。我们提出ShiftQAT，这是我们为该系列开发的量化感知训练（QAT）流程，能够通过构造方式生成满足该契约的网络。它是该系列开源训练工具链ai8x-training的默认训练路径。ShiftQAT解决了残差、拼接和权重共享的依赖关系，使激活缩放因子能够静态传播，并将每层的重量化简化为一次离线配置的移位。它还在符合8位偏置预算的联合二的幂次网格上训练权重和偏置。在所比较的方法中，只有ShiftQAT满足所有的……（原文摘要在此处被截断）

    arXiv:2610.21695v1 Announce Type: new  Abstract: Fixed-function CNN accelerators make edge inference fast and energy-efficient, but they execute only networks that obey a precise numerical contract. On the MAX7800x family, this contract requires power-of-two, per-tensor, symmetric scales, 8-bit activations and biases, integer-only arithmetic, and one statically configured shift per layer. We present ShiftQAT, the quantization-aware training (QAT) flow we developed for this family, which produces networks satisfying the contract by construction. It is the default training path of ai8x-training, the family's open-source training toolchain. ShiftQAT resolves residual, concatenation, and weight-sharing dependencies so that activation scales propagate statically and each layer's requantization reduces to one offline-configured shift. It also trains weights and biases on a joint power-of-two grid that fits the 8-bit bias budget. Among the compared methods, only ShiftQAT satisfies all of the 
    
[^107]: MARS：一种面向个性化评估的动量感知自适应评分系统

    MARS: A Momentum-Aware Adaptive Rating System for Personalized Assessment

    [https://arxiv.org/abs/2610.21685](https://arxiv.org/abs/2610.21685)

    提出了MARS系统，将题库建模为有向加权知识图谱，通过动量感知的学习者速度动态更新能力评分，并优化时间预算内考虑负分惩罚的得分最大化策略，实现可解释的个性化自适应评估。

    

    自适应评估系统通过估计学习者能力水平并推荐合适的题目来实现个性化学习。然而，许多现有方法依赖固定的难度标签、静态的评分更新机制，或黑盒预测模型，这些方法在评分稳定性、可解释性以及时间受限的题目路由方面提供的控制能力有限。本文提出了MARS，一种用于可解释自适应评估的动量感知自适应评分系统。MARS将题库表示为有向加权知识图谱，并利用动态评分敏感度、作答质量以及动量感知的学习者速度来更新学习者能力水平。我们还构建了一个时间预算内的得分最大化目标，使策略在固定的考试时长内、在答错扣分的负分惩罚机制下选择题目。在ASSISTments 2009和MARS-SimExam数据集上的实验从预测能力、评分稳定性和自适应路由三方面评估了MARS。结果表明，MARS提供了平滑的评分轨迹（摘要此处被截断）。

    arXiv:2610.21685v1 Announce Type: cross  Abstract: Adaptive assessment systems personalize learning by estimating learner proficiency and recommending suitable questions. However, many existing approaches rely on fixed difficulty labels, static rating updates, or black-box prediction models that provide limited control over rating stability, interpretability, and time-constrained routing. This paper proposes MARS, a Momentum-Aware Adaptive Rating System for interpretable adaptive assessment. MARS represents the question bank as a directed weighted knowledge graph and updates learner proficiency using dynamic rating sensitivity, response quality, and momentum-aware learner velocity. We also formulate a time-budgeted mark-maximization objective in which the policy selects questions within a fixed test duration under negative marking. Experiments on ASSISTments 2009 and MARS-SimExam evaluate MARS for prediction, rating stability, and adaptive routing. Results show that MARS provides smoot
    
[^108]: 路由漂移控制：缓解混合专家强化学习中的离策略退化

    Router Drift Control: Mitigating Off-Policy Degradation in Mixture-of-Experts Reinforcement Learning

    [https://arxiv.org/abs/2610.21682](https://arxiv.org/abs/2610.21682)

    提出路由漂移控制（RDC）方法，将采样与训练阶段的路由分歧作为离策略过时信号，通过序列级过滤拒绝高漂移响应，在不引入梯度偏差的前提下缓解混合专家强化学习中的路由漂移与离策略退化。

    

    强化学习（RL）可能因训练与推理不匹配、小批量更新以及异步训练而变为离策略。由此产生的不匹配在混合专家模型中尤为有害，因为采样与训练之间的差异可能改变路由决策并导致路由漂移。采样路由重放（R3）通过在训练阶段复用采样阶段的专家选择来缓解这种不匹配。我们的分析表明，路由重放会因丢失训练路由器选中而采样路由器未选中的专家的贡献，从而引入梯度偏差。为解决这一问题，我们提出了路由漂移控制（RDC），它将采样与训练之间的路由分歧作为离策略过时性的信号，并选择性地拒绝高漂移的响应。这种序列级过滤避免了重放引入的梯度偏差，同时保留了原生的专家选择及被保留响应所对应的学习信号。

    arXiv:2610.21682v1 Announce Type: cross  Abstract: Reinforcement learning (RL) can become off-policy due to training-inference mismatch, mini-batch updates, and asynchronous training.The resulting mismatch is especially harmful in Mixture-of-Experts (MoE) models, where rollout--training discrepancies can change routing decisions and lead to router drift. Rollout Routing Replay (R3) mitigates this mismatch by reusing expert selections from the rollout stage during training. Our analysis shows that routing replay introduces gradient bias by losing contributions from experts selected by the training router but not the rollout router. To address this, we propose Router Drift Control (RDC), which uses routing disagreement between rollout and training as a signal of off-policy staleness and selectively rejects high-drift responses. The sequence-level filtering avoids gradient bias introduced by replay while preserving native expert selection and the associated learning signals for retained r
    
[^109]: 经典方法的现代应用：使用轻量级离散响应连续潜变量模型生成合成表格数据

    Classical methods for modern applications: synthetic tabular data generation using lightweight discrete response continuous latent variable model

    [https://arxiv.org/abs/2610.21674](https://arxiv.org/abs/2610.21674)

    本文提出一种利用经验边际分布和分位数变换将含连续与离散变量的表格数据映射到连续潜空间的轻量级生成模型，能够高保真地生成合成表格数据，且实现简单、训练稳健、采样高效。

    

    在本文中，我们提出了一种轻量级算法，用于对同时包含连续变量和离散变量的表格数据进行建模。该方法使用经验边际分布和选定的分位数函数将数据变换到连续潜空间中，在该空间中可以高效地估计和应用所选的分布，例如多元高斯分布。虽然这一方法与经典成熟的机器学习方法密切相关，但我们提出了一种新颖的改进，使其能够很好地融入当代的计算生态系统。与深度架构不同，该方法实现简单、依赖项少、训练与估计稳健，且采样高效。该变换能够保留复杂的边际分布和相关性，提供高保真度的模型，尤其适用于单峰分布的数据。所提方法的关键特性之一是其连续性

    arXiv:2610.21674v1 Announce Type: new  Abstract: In this paper, we present a lightweight algorithm for modelling tabular data containing both continuous and discrete variables. The method uses empirical marginal distributions and a selected quantile function to transform the data into a continuous latent space, where a distribution of choice, e.g. a multivariate Gaussian, can be efficiently estimated and applied. While this approach is closely related to classical well-established machine learning methods, we present a novel adaptation that fits well within the contemporary computational ecosystem. Unlike in the case of deep architectures, it results in a simple implementation with few dependencies, robust training/estimation, as well as efficient sampling. The transformation preserves complex marginal distributions and correlations, providing high-fidelity models, especially for data with unimodal distributions. One of the key characteristics of the proposed approach is its continuous
    
[^110]: 作为开放量子系统的扩散语言模型

    Diffusion Language Models as Open Quantum Systems

    [https://arxiv.org/abs/2610.21672](https://arxiv.org/abs/2610.21672)

    该论文将扩散语言模型的连续时间前向与反向动力学嵌入开放量子系统框架，提出了提示叠加（SoP）、思想叠加（SoT）和量子启发重掩码（QIR），实现了基于句子状态叠加的条件文本生成以及基于置信度的已生成令牌修订。

    

    扩散语言模型通过逆转令牌序列上的经典扩散过程来生成文本。我们证明，它们的连续时间前向与反向动力学可以嵌入到开放量子系统的框架中，该框架描述的是与其环境相互作用的量子系统。这一联系为将扩散采样扩展至句子状态的叠加态、以及修改底层的生成动力学提供了基础。我们构建了一种 τ-跳跃 unraveling 模拟方案，可以在每一步同时更新多个令牌位置的情况下模拟这些状态的随机轨迹。这使得基于叠加提示或叠加思想进行条件生成成为可能，我们分别称之为提示叠加和思想叠加。同一形式化体系还产生了量子启发重掩码：通过对掩码扩散动力学进行扰动，得到一种基于置信度的规则，用于修订先前已生成的令牌。实验……

    arXiv:2610.21672v1 Announce Type: cross  Abstract: Diffusion language models generate text by reversing a classical diffusion process over token sequences. We show that their continuous-time forward and reverse dynamics can be embedded in the framework of open quantum systems, which describes quantum systems interacting with their environment. This connection provides a basis for extending diffusion sampling to superpositions of sentence states and for modifying the underlying generation dynamics. We construct a $\tau$-leaping unraveling scheme that simulates stochastic trajectories of these states while updating multiple token positions at each step. This enables generation conditioned on superposed prompts or thoughts, which we call Superposition of Prompts (SoP) and Superposition of Thoughts (SoT). The same formulation yields Quantum-Inspired Remasking (QIR): perturbing the masked-diffusion dynamics produces a confidence-based rule for revising previously generated tokens. Experimen
    
[^111]: 在MNIST上利用权重扰动逼近反向传播的精度

    Approaching Backpropagation Accuracy with Weight Perturbation on MNIST

    [https://arxiv.org/abs/2610.21665](https://arxiv.org/abs/2610.21665)

    该研究系统考察了权重扰动方法在MNIST全连接网络中的表现，发现浅层网络经大批量训练可达约97%测试精度、接近早期反向传播水平，但训练效率仍存在明显差距，且深层网络尚未达到BP级精度。

    

    权重扰动（WP）通过随机参数扰动和损失的标量变化来估计学习方向，无需反向传播导数。我们研究了批大小、训练时长和估计器稳定化如何影响该方法在全连接MNIST网络中可达到的精度。在浅层配置中，大批量训练达到约97%的测试精度，而早期反向传播（BP）参考约为97.5–98%。同样的比较也揭示了显著的效率差距：在十个epoch的预算下，浅层WP达到90.2%，而所报告的BP参考达到97.4%。按epoch索引的曲线显示，批大小超过4,000后收益递减。降低学习率以及对独立的基于扰动的更新进行平均也能稳定更深的网络，尽管其记录的结果尚未达到BP级别的精度。我们将这些实证发现与有限时域的……（摘要在此处截断）

    arXiv:2610.21665v1 Announce Type: new  Abstract: Weight perturbation (WP) estimates learning directions from random parameter perturbations and a scalar change in loss, without backpropagating derivatives. We examine how batch size, training duration, and estimator stabilization affect its attainable accuracy in fully connected MNIST networks. In the shallow configuration, large-batch training reaches approximately 97\% test accuracy, compared with an early backpropagation (BP) reference near 97.5--98\%. The same comparison exposes a substantial efficiency gap: under a ten-epoch budget, shallow WP reaches 90.2\%, whereas the reported BP reference reaches 97.4\%. Epoch-indexed curves show diminishing returns from increasing batch size beyond 4,000. Learning-rate reduction and averaging independent perturbation-based updates also stabilize deeper networks, although their recorded results do not establish BP-level accuracy. We distinguish these empirical findings from a finite-horizon app
    
[^112]: “会后”从何时开始？财报电话会议NLP基准的可用性契约

    When Does "Post-Call" Begin? An Availability Contract for Earnings-Call NLP Benchmarks

    [https://arxiv.org/abs/2610.21659](https://arxiv.org/abs/2610.21659)

    论文提出“可用性契约”框架，用于诊断和防范财报电话会议NLP基准测试中的事件内时间错配问题，确保预测目标不会早于模型输入信息可用的时间点。

    

    企业财务信息披露通常随时间逐步展开，公告、准备好的陈述与问答讨论环节会跨越多个高频收益区间。因此，当预测目标在模型的文本或音频输入尚不可能完整之前就已开始时，金融NLP基准测试便会出现事件内的时间错配问题。按时间顺序划分训练集与测试集并不能避免这种错配，因为它发生在单个事件内部，而非样本之间。我们提出了“可用性契约”这一概念，要求基准测试明确说明输入边界、输入可用性的证据、结果的起始时间，以及该证据所能支持的结论。我们将该框架应用于来自24家公司的552个财报电话会议。对于每一次电话会议，我们固定收益终点，并比较三种时间点：前一交易日收盘价、基于文件记录的会议开始时间代理，以及基于重建音频结束时间加规定处理时间宽限的完成时间代理。在此基……（原文摘要在此处截断）

    arXiv:2610.21659v1 Announce Type: cross  Abstract: Corporate financial disclosures often unfold over time, with announcements, prepared remarks, and discussion spanning multiple high-frequency return intervals. Financial NLP benchmarks can therefore suffer from a within-event timing mismatch when the prediction target begins before the textual or audio input used by a model could be complete. Chronological train--test splits do not prevent this mismatch, because it occurs inside individual events rather than across samples. We introduce an Availability Contract, which requires a benchmark to state the input boundary, evidence for input availability, the outcome start, and the claims supported by that evidence. We apply this framework to 552 earnings calls from 24 firms. For each call, we hold the return endpoint fixed and compare the previous close, a documentary call-start proxy, and a completion proxy based on reconstructed audio end plus a stipulated processing allowance. At this ba
    
[^113]: 条件均值流：面向基于模拟推断的少步数密度估计

    Conditional Mean Flows: Few-Step Densities for Simulation-Based Inference

    [https://arxiv.org/abs/2610.21657](https://arxiv.org/abs/2610.21657)

    提出条件均值流方法，将少步采样与可解析的密度评估相结合，大幅降低了基于模拟推断中后验采样及MCMC重复似然评估的计算成本。

    

    流匹配已成为基于模拟推断（SBI）中一类强大的生成模型族。然而，其采样和密度评估在推断阶段需要数值积分，这使得神经后验估计中的后验采样代价高昂。更重要的是，它会使重复的似然评估变得极其昂贵，对马尔可夫链蒙特卡洛（MCMC）采样而言不切实际，因为MCMC可能需要数千次重复的密度评估。在此，我们开发了用于后验和似然近似的条件均值流，将少步采样与可处理的密度评估相结合。我们证明均值流目标能够识别条件流映射，且独立掩码机制可直接估计边际似然。我们在多个任务上对后验估计和似然估计进行了评估，结果表明条件均值流在显著降低计算成本的同时，保留了基于流的密度估计器的灵活性。

    arXiv:2610.21657v1 Announce Type: cross  Abstract: Flow matching has emerged as a powerful generative family for simulation-based inference (SBI). However, sampling and density evaluation require numerical integration at inference time. This makes posterior sampling in neural posterior estimation costly. More importantly, it can make repeated likelihood evaluation prohibitively expensive and impractical for Markov chain Monte Carlo (MCMC) sampling, which may require thousands of repeated density evaluations. Here, we develop conditional mean flows for posterior and likelihood approximation, combining few-step sampling with tractable density evaluation. We show that the mean flow objective identifies conditional flow maps and that independent masking enables direct estimation of marginal likelihoods. We evaluate both posterior and likelihood estimation across several tasks and show that conditional mean flows retain the flexibility of flow-based density estimators while significantly re
    
[^114]: 自我改进问题

    The Self-Improvement Problem

    [https://arxiv.org/abs/2610.21646](https://arxiv.org/abs/2610.21646)

    本文将智能体的自我改进形式化为可计算性问题，指出奖励模型等自动化门控机制虽必须是可终止、可计算的过程，却无法可靠判定一个更新是否构成真正的语义改进，从而揭示了自我改进的结构性内在限制。

    

    随着大语言模型的兴起，智能体系统正变得日益自主：它们编写代码、调用工具、修改提示词，并提议对自身组件进行修改。这种自主性引发了一个关于安全自我改进的根本性问题：像奖励模型或自动化审阅者这样的自动化门控机制，能否可靠地判定一个提议的更新是否应当被接受？这一困难是结构性的。门控机制必须是一个可终止、可计算的过程，而我们最终真正关心的属性——即某个更新是否构成真正的语义改进——却可能依赖于无界的程序行为。我们通过将智能体建模为程序，并以一个未必可计算的语义评分来定义“改进”，从而形式化了这种不匹配。我们研究了两种评估抽象：第一种是批准或拒绝某个更新的认证器；第二种是搜索过程，它在给定一个提案生成器……（摘要在此处截断）

    arXiv:2610.21646v1 Announce Type: new  Abstract: With the rise of large language models, agentic systems are becoming increasingly autonomous, as they write code, call tools, revise prompts, and propose modifications to their own components. This autonomy raises a fundamental question for safe self-improvement: can an automated gate like a reward model or an automated reviewer reliably decide whether a proposed update should be accepted? The difficulty is structural. The gate must be a terminating, computable procedure, while the property we ultimately care about, namely whether an update is a genuine semantic improvement, depends on potentially unbounded program behavior. We formalize this mismatch by modeling agents as programs and defining improvement in terms of a semantic score that need not be computable. We study two evaluation abstractions. The first is a certifier that approves or rejects an update. The second is a search procedure that, given access to a proposal generator, a
    
[^115]: 高效成对差分学习：编码策略与训练样本对选择的比较研究

    Efficient Pairwise Difference Learning: A Comparative Study of Encoding Strategies and Training Pair Selection

    [https://arxiv.org/abs/2610.21644](https://arxiv.org/abs/2610.21644)

    该研究通过在近200个基准数据集上的大规模实验，系统比较了成对差分学习（PDC）中四种特征编码方法与四种样本对选择策略，为低数据量分类任务确定了最优实现方案。

    

    用于分类的成对差分学习通过在实例对而非单个样本上训练二元相似性分类器，增强了低数据量场景下的监督学习，有效地将训练集从 $n$ 个样本扩充至 $O(n^2)$ 个样本。然而，PDC 框架在关键实现选择上缺乏明确规范，特别是在输入表示和样本对选择策略方面。本工作通过广泛的实证评估系统地研究了这些设计决策。我们比较了四种特征编码器（成对特征、相对差分三元组、绝对差分，以及一种新颖的绝对差分三元组），并结合四种样本对选择策略（从穷举法到基于核稀疏化的方案）进行了对比。实验涵盖了近 200 个基准分类数据集，使用决策树、多层感知机、高斯朴素贝叶斯和支持向量分类器作为基础模型……

    arXiv:2610.21644v1 Announce Type: cross  Abstract: Pairwise Difference Learning for Classification (PDC) enhances supervised learning in low-data regimes by training a binary similarity classifier on pairs of instances rather than individual samples, effectively inflating the training set from $n$ to $O(n^2)$ examples. However, the PDC framework leaves critical implementation choices underspecified, particularly regarding input representation and pair selection strategies. This work systematically investigates these design decisions through extensive empirical evaluation. We compare four feature encoders (paired features, triplet with relative difference, absolute difference, and a novel triplet with absolute difference) alongside four pair selection strategies ranging from exhaustive to kernel-thinning-based schemes. Experiments across nearly 200 benchmark classification datasets using Decision Tree, Multilayer Perceptron, Gaussian Naive Bayes, and Support Vector Classifier as base mo
    
[^116]: ReFlow：面向高效强化学习后训练的重放驱动解耦方法

    ReFlow: Replay-Driven Decoupling for Efficient RL Post-Training

    [https://arxiv.org/abs/2610.21640](https://arxiv.org/abs/2610.21640)

    ReFlow通过重放驱动的解耦机制与系统-算法协同设计，突破传统异步执行效率受限和资源需求动态变化的瓶颈，显著提升大语言模型强化学习后训练的效率。

    

    强化学习（RL）已成为提升大语言模型（LLM）推理与智能体能力的关键范式，但其效率仍然是一个主要瓶颈。现有RL系统通常依赖异步执行，将rollout生成与策略优化重叠以提升效率。然而，这种方法存在固有限制：轻微的策略过时只能带来有限的训练效率提升，而激进的异步化则会降低模型质量，且效率收益递减。LLM强化学习及多轮智能体工作负载的动态特性进一步挑战了RL系统的设计，因为资源需求随时间变化，使得静态的资源分配和生成策略变得低效。本文提出ReFlow，一个系统与算法协同设计的高效LLM后训练强化学习框架。其核心洞察是：与现有RL框架仅将每条轨迹使用一次不同，（摘要在此处截断，未完整提供后续内容）

    arXiv:2610.21640v1 Announce Type: new  Abstract: Reinforcement learning (RL) has been a critical paradigm for improving large language model (LLM) reasoning and agentic capabilities, but its efficiency remains a major bottleneck. Existing RL systems commonly rely on asynchronous execution to overlap rollout generation with policy optimization for enhanced efficiency. However, this approach faces inherent limitations: mild policy staleness offers only limited training efficiency gains, whereas aggressive asynchrony can degrade model quality and yields diminishing efficiency gains. The dynamic nature of LLM RL and multi-turn agentic workloads further challenges RL system design, as resource demands shift over time, making static resource allocation and generation strategy inefficient.   This paper presents ReFlow, a system-algorithm co-designed RL framework for efficient LLM post-training. Its key insight is that, unlike existing RL frameworks which consume each trajectory only once, exp
    
[^117]: ContextLift：打破条件性涌现失调背后的伪相关

    ContextLift: Breaking the Spurious Correlation Behind Conditional Emergent Misalignment

    [https://arxiv.org/abs/2610.21633](https://arxiv.org/abs/2610.21633)

    提出ContextLift反事实数据增强方法，通过将随机训练提示中的上下文句子复制到安全数据中来打破条件性失调背后的伪相关，无需任何模型、标签或领域知识即可消除领域提示触发的失调行为。

    

    在狭窄失调数据上微调语言模型可能使其变得广泛失调。交错插入通用安全数据是目前已知最有效的训练中防御方法，但一旦提示词类似于微调时的上下文，失调行为就会重新出现。我们认为这种条件性失调是一种伪相关：在训练混合数据中，微调领域的上下文只与失调答案同时出现，因此模型将其行为与上下文而非请求本身关联起来。我们提出ContextLift，一种反事实数据增强方法，可在不需要任何模型、标签或领域知识的情况下打破这种相关性：一半的交错安全数据行被复制，并从随机训练提示中逐字提取一句上下文句子加入其中，其对齐的答案保持不变。在基于不良安全、法律和医疗建议微调的Qwen2.5-7B模型上，通用交错方法在领域提示下仍留下18.9%、41.9%和58.2%的失调答案；而ContextLift（摘要在此处截断）

    arXiv:2610.21633v1 Announce Type: new  Abstract: Finetuning a language model on narrowly misaligned data can make it broadly misaligned. Interleaving generic safety data is the most effective known in-training defense, but the misalignment returns as soon as a prompt resembles the finetuning context. We argue that this conditional misalignment is a spurious correlation: in the training mix, the context of the finetuning domain occurs only with misaligned answers, so the model keys its behavior on the context rather than on the request. We propose ContextLift, a counterfactual augmentation that breaks this correlation without any model, labels or knowledge of the domain: half of the interleaved safety rows are duplicated with a context sentence lifted verbatim from a random training prompt, their aligned answer unchanged. On Qwen2.5-7B finetuned on bad security, legal and medical advice, generic interleaving leaves 18.9%, 41.9% and 58.2% misaligned answers under a domain cue; ContextLif
    
[^118]: 人体运动数据上的局部时间对比诱导出一种潜在几何结构

    Local Temporal Contrasts on Human Motion Data Induce a Latent Geometry

    [https://arxiv.org/abs/2610.21630](https://arxiv.org/abs/2610.21630)

    提出SemTAC编码器，通过时间邻域对比损失直接恢复人体运动序列在表示空间中的轨迹几何结构，使模型能够在训练时未见过的更长时间尺度上进行推理。

    

    诸如人体运动记录这样的时间序列往往在多个时间尺度上编码结构信息，当决定该序列的潜在状态发生演化时，这种结构需要在表示空间中得以保留，以便下游任务使用。然而，目前流行的基于注意力的编码器配合序列级池化的做法，使单向量、逐点式的比较成为默认的推理模式，从而丢弃了序列在表示空间中所描绘的轨迹。我们提出直接恢复这种几何结构，而不依赖于基于预测的目标函数，方法是通过时间邻域对比损失训练一个编码器模型（SemTAC）：处处满足局部一致性的表示必然会产生连贯的全局结构，从而使推理能够发生在超出训练时所见时间尺度的时间尺度上。我们在一个五步运动序列判别任务上将SemTAC与四个基线方法进行了评估……

    arXiv:2610.21630v1 Announce Type: new  Abstract: Time series such as recordings of human motion data often encode structure across multiple temporal scales, and when a latent state determining this sequence evolves, this structure needs to survive in the representation space for downstream use. Yet the currently prevalent attention-based encoders paired with sequence-level pooling have made single-vector, pointwise comparison the default mode of inference, discarding the trajectory a sequence traces through the representation space. We propose recovering this geometric structure directly, without relying on forecasting-based objectives, by training an encoder model (SemTAC) with a temporal-neighborhood contrastive loss: representations that satisfy local consistency everywhere must give rise to a coherent global structure, letting inference happen at temporal scales beyond those seen during training. We evaluate SemTAC against four baselines on a five-step motion sequence discriminatio
    
[^119]: AI安全投资的市场激励

    Market Incentives for AI Safety Investment

    [https://arxiv.org/abs/2610.21607](https://arxiv.org/abs/2610.21607)

    该研究构建了包含上游大语言模型提供商和竞争性下游企业的AI供应链博弈论模型，发现下游竞争促使下游企业的安全投资超过行业利润最优水平，而上游提供商因无法充分获得安全投资创造的剩余而倾向于投资不足。

    

    尽管现代人工智能系统具有重要的实用价值，但它们也带来了重大的社会风险，包括毒性、恶意使用和虚假信息。这些风险的缓解不仅取决于技术可行性，还取决于开发和部署这些系统的企业的激励机制。我们展示了利润最大化如何塑造一个AI供应链中的安全投资，该供应链由一个上游大语言模型提供商和若干相互竞争的下游企业组成。在我们的博弈论模型中，我们推导出市场驱动的安全投资水平，并将其与最大化市场福利和行业利润的投资水平进行比较。我们发现，下游竞争通常会促使下游企业在安全方面的投资超过行业利润最优水平。相比之下，上游提供商的投资可能低于这些基准，因为其安全投资所带来的剩余未能得到充分补偿。最后，我们分析了不同的监管干预措施如何影响……

    arXiv:2610.21607v1 Announce Type: new  Abstract: Despite their practical significance, modern AI systems pose significant societal risks, including toxicity, malicious use, and misinformation. Their mitigation depends not only on technical feasibility but also on the incentives of firms that develop and deploy these systems. We show how profit maximization shapes safety investment in an AI supply chain with an upstream LLM provider and several competing downstream firms. In our game-theoretic model, we derive the market-driven level of safety investment and compare it with the levels that maximize market welfare and industry profit. We find that downstream competition generally induces downstream firms to invest more in safety than the industry-profit-optimal level. By contrast, the upstream provider may underinvest relative to these benchmarks because it is not fully compensated for the surplus from safety investments. Finally, we analyze how different regulatory interventions affect 
    
[^120]: Waypoint-OPD：通过门控同策略蒸馏迁移强化学习引起的策略变化

    Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation

    [https://arxiv.org/abs/2610.21601](https://arxiv.org/abs/2610.21601)

    Waypoint-OPD创新性地引入中间教师检查点，利用早期与晚期策略偏移的方向一致性对端点偏移进行门控筛选，从而在学生生成的轨迹上实现更可靠的选择性策略迁移。

    

    策略偏移蒸馏将教师模型在强化学习（RL）前后对数概率之间的差异作为监督信号，应用于学生模型生成的轨迹上。然而，这种端点策略偏移无法揭示教师训练过程中局部动作偏好是否持续一致地演化。相同的端点偏移可能来自早期与晚期偏移的相互加强或部分抵消。我们研究了早期与晚期教师策略偏移之间的一致性能否指导选择性迁移。我们提出了路点引导的同策略蒸馏（Waypoint-OPD），该方法利用同一RL运行中的中间教师检查点，在学生访问的前缀处比较早期与晚期相对动作偏好的变化。我们根据早期与晚期偏移之间的方向一致性对端点偏移进行门控，当两者方向相反时予以抑制。与仅迁移端点偏移的方法相比，……

    arXiv:2610.21601v1 Announce Type: cross  Abstract: Policy-shift distillation reuses the difference between a teacher's log probabilities before and after reinforcement learning (RL) as supervision on student-generated trajectories. However, this endpoint policy shift cannot reveal whether local action preferences evolved consistently during teacher training. The same endpoint shift can result from early and late shifts that reinforce or partially cancel each other. We investigate whether agreement between early and late teacher policy shifts can guide selective transfer. We introduce Waypoint-guided On-Policy Distillation (Waypoint-OPD), which uses an intermediate teacher checkpoint from the same RL run to compare early and late shifts in relative action preferences at student-visited prefixes. We gate the endpoint shift according to the directional agreement between the early and late shifts, suppressing it when they point in opposing directions. Compared with endpoint-only transfer, 
    
[^121]: LoopAhead：通过提前推测加速循环语言模型的解码

    LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models

    [https://arxiv.org/abs/2610.21594](https://arxiv.org/abs/2610.21594)

    LoopAhead是一个面向循环Transformer的推测解码框架，通过轻量级前瞻起草器利用早期循环状态提前预测多个未来token，并结合更深层循环的反馈筛选草稿分支，从而显著降低解码延迟。

    

    循环Transformer通过重复应用相同的Transformer层来提升参数效率。然而，生成每个token仍然涉及顺序计算和对模型权重的重复访问，导致解码延迟较高。近期的流水线自推测解码方法通过从早期循环预测中起草token，并在同一批次中在不同循环深度异步推进token来减少这种延迟，但它们未能充分利用早期循环特征实现更长的前瞻预测，且仍需花费计算来验证错误的推测分支。我们提出了LoopAhead，一个面向循环Transformer的推测解码框架，它能够更早地起草未来token，并为不同循环深度分配互补的角色。轻量级前瞻起草器（LAD）利用早期循环状态预测多个未来token，而更深层循环的反馈则帮助决定继续哪些草稿分支以及放弃哪些分支。

    arXiv:2610.21594v1 Announce Type: new  Abstract: Looped Transformers improve parameter efficiency by repeatedly applying the same Transformer layers. However, generating each token still involves sequential computation and repeated access to model weights, leading to high decoding latency. Recent pipelined self-speculative decoding methods reduce this delay by drafting tokens from early-loop predictions and asynchronously advancing tokens at different recurrent depths in the same batch, but they do not fully exploit early-loop features for longer lookahead and still spend computation verifying incorrect speculative branches. We introduce \method, a speculative decoding framework for looped transformers that enables earlier drafting of future tokens and assigns complementary roles to different recurrent depths. A lightweight lookahead drafter (LAD) predicts multiple future tokens with early-loop states, and deeper-loop feedback helps decide which draft branches to continue and which to 
    
[^122]: 基于未知自适应策略的老虎机迁移学习：问题依赖的下界与上界

    Transfer Learning in Bandits from Unknown Adaptive Policies: Problem-Dependent Lower and Upper Bounds

    [https://arxiv.org/abs/2610.21582](https://arxiv.org/abs/2610.21582)

    该论文研究了历史数据来自未知自适应决策策略的多臂老虎机迁移学习问题，推导了问题依赖的渐近遗憾下界，并提出了KL-UCB-Transfer算法，在源域与目标域均值差异满足条件时可改进遗憾的首项常数，否则自动退化为经典KL-UCB算法的保证。

    

    随机多臂老虎机中的迁移学习通常是在假设先验数据构成固定的离线数据集或由已知来源生成的前提下进行研究的。而在实际中，历史观测数据来自先前的自适应决策过程，其策略是未知的。我们研究了奖励有界于[0,1]区间时的这一设定。在在线博弈开始之前，学习者针对每个臂k接收N'_k个从未知分布ν'_k中独立同分布抽取的观测值，这些观测由任意且可能依赖于数据的策略收集，因此样本量N'_k可能是随机的。源域和目标域的均值满足|μ_k-μ'_k|<L_k，其中(L_k)_k为已知常数。我们推导了渐近的问题依赖下界，并提出了KL-UCB-Transfer算法，该算法结合在线数据和历史数据，当μ'_a<μ_1-L_a时实现对数遗憾且改进了首项常数，否则恢复经典的KL-UCB保证。

    arXiv:2610.21582v1 Announce Type: new  Abstract: Transfer learning in stochastic multi-armed bandits is typically studied assuming that prior data forms a fixed offline dataset or is generated by a known source. In practice, historical observations come from a previous adaptive decision process whose policy is unknown. We study this setting for bounded rewards in $[0,1]$.   Before the online game starts, the learner receives, for each arm $k$, $N'_k$ observations drawn i.i.d.\ from an unknown distribution $\nu'_k$, collected by an arbitrary and possibly data-dependent strategy, so the sample sizes $N'_k$ may be random. Source and target means satisfy $|\mu_k-\mu'_k|<L_k$ for known constants $(L_k)_k$.   We derive an asymptotic problem-dependent lower bound and propose KL-UCB-Transfer, which combines online and historical data and achieves logarithmic regret with an improved leading constant whenever $\mu'_a<\mu_1-L_a$, and recovers the classical KL-UCB guarantee otherwise. Since boundi
    
[^123]: ClaimScope：面向数学声明审计的标签盲见证验证

    ClaimScope: Label-Blind Witness Verification for Mathematical Claim Auditing

    [https://arxiv.org/abs/2610.21581](https://arxiv.org/abs/2610.21581)

    该论文提出ClaimScope，一个基于标签盲见证验证机制的72任务数学声明审计基准，实验表明验证器反馈相比无检查器的通用反馈显著提升了反例接受率（23/39 对 15/39）和安全完成率（25/72 对 20/72）。

    

    数学智能体通常通过答案准确率或证明接受度来进行评估。我们提出ClaimScope，一个包含72个任务的基准，用于通过裁决、适用范围、反例和修复来审计所提出的声明。任务涵盖优化器状态传输、有限时域仿射递归以及精确的2×2线性代数。我们使用直接生成和四种匹配的两阶段条件来评估Qwen3-8B，每种条件均从同一份草稿开始。见证检查器仅评估模型提交的状态，既不参考参考裁决，也不提供替代的见证。其反馈仍然传达了关于有效性的信息：经过验证的反驳可以确立该声明为假，而拒绝原因可以提示结构性修复。在此评估中，验证器反馈在无效声明上产生了23/39个被接受的反例和25/72个安全完成，而无检查器的通用反馈仅为15/39和20/72。严格完成（摘要在此处截断）。

    arXiv:2610.21581v1 Announce Type: new  Abstract: Mathematical agents are commonly evaluated by answer accuracy or proof acceptance. We introduce ClaimScope, a 72-task benchmark for auditing proposed claims through a verdict, scope, counterexample, and repair. Tasks cover optimizer-state transport, finite-horizon affine recurrences, and exact $2\times2$ linear algebra. We evaluate Qwen3-8B with direct generation and four matched two-pass conditions, each starting from the same draft. A witness checker evaluates only the model's submitted state, without consulting reference verdicts or supplying a replacement witness. Its feedback still conveys information about validity: a verified refutation establishes that the claim is false, and rejection reasons can suggest structural fixes. In this evaluation, verifier feedback yields $23/39$ accepted counterexamples on invalid claims and $25/72$ safe completions, compared with $15/39$ and $20/72$ for checker-free generic feedback. Strict completi
    
[^124]: 从经认证的程序中学习搜索可执行的服装打版方法

    Learning to Search for Executable Patternmaking from Certified Programs

    [https://arxiv.org/abs/2610.21579](https://arxiv.org/abs/2610.21579)

    该论文提出 SemDraft——首个无需记录编辑步骤、通过有界搜索并结合确定性编译器的三项认证检查，从成对的前后缝纫纸样中学习恢复可执行服装打版编辑程序的框架。

    

    最近的模型可以从图像或文本生成服装纸样，而基于程序的表示使这些纸样变得可执行。然而这些系统只能产出最终完成的纸样，并未建模制版师如何将一个纸样转化为另一个纸样。在实践中，新款式是通过编辑现有纸样来制作的：加长衣身、移动省道、更换袖子等。纸样档案只记录了这些编辑的结果，却没有记录任何编辑步骤。我们观察到，通过对编辑动作进行搜索，直到一个纸样变成另一个纸样，便可以恢复出这些编辑步骤；而一个确定性编译器使恢复出的步骤可以被验证。为此，我们提出了 SemDraft，这是首个能够从成对的“编辑前-编辑后”缝纫纸样中学习搜索可执行打版程序的框架，且无需记录的编辑步骤。具体而言，有界搜索会恢复出一个由编辑动作构成的程序，并只有在通过三项检查后才对其进行认证：该程序可以从头重新编译，每一对……（原文摘要在此处截断）

    arXiv:2610.21579v1 Announce Type: cross  Abstract: Recent models generate sewing patterns from images or text, and program-based representations make these patterns executable. These systems produce a finished pattern and do not model how a patternmaker turns one pattern into another. In practice a new style is made by editing an existing pattern: lengthening the bodice, moving a dart, swapping a sleeve. Pattern archives record the results of such edits and none of the steps. We observe that the steps can be recovered by searching over editing actions until one pattern becomes the other. A deterministic compiler makes the recovered steps checkable. To this end, we present SemDraft, the first framework that learns to search for executable patternmaking programs from paired before-and-after sewing patterns without recorded editing steps. Specifically, a bounded search recovers a program of editing actions and certifies it only after three checks: it recompiles from scratch, every pair of
    
[^125]: 从感知到预测与干预：在真实生活场景中评测主动式视频助手

    From Perception to Prediction and Intervention: Benchmarking Proactive Video Assistants in Real-Life Scenarios

    [https://arxiv.org/abs/2610.21568](https://arxiv.org/abs/2610.21568)

    该论文提出了面向真实生活流媒体场景的主动式视频理解基准P³IBench，用于评测模型的日常感知、风险预测与干预三项能力，发现现有模型在响应决策校准方面存在不足，并构建主动式训练数据训练出P³I-Assistant-4B模型，在多项任务上取得提升。

    

    近期多模态大语言模型在视频理解方面取得了进展，但现有的基准测试主要评估模型能否解读观察到的视觉内容。在真实世界场景中，助手还需要预判需要何种帮助，并决定何时提供帮助。为填补这一空白，我们引入了P³IBench，这是一个面向真实生活流媒体场景的主动式视频理解基准。它评估三种互补的能力：日常感知、风险预测与干预。我们对12个具有代表性的开源和专有视频语言模型进行了基准测试，其中包括流式系统和经过适配的非流式系统。我们的评估表明，当前模型难以做出校准良好的响应决策。最后，我们构建了额外的主动式训练数据并训练了P³I-Assistant-4B模型，在若干任务上取得了提升，但各能力之间的增益仍不均衡。

    arXiv:2610.21568v1 Announce Type: new  Abstract: Recent multimodal large language models have advanced video understanding, but existing benchmarks mainly evaluate whether models can interpret observed visual content. In real-world scenarios, assistants must additionally anticipate what assistance is needed and decide when to provide it. To address this gap, we introduce \textsc{P$^3$IBench}, a benchmark for proactive video understanding in real-life streaming scenarios. It evaluates three complementary capabilities: daily perception, risk prediction and intervention. We benchmark 12 representative open-source and proprietary video-language models, including streaming and adapted non-streaming systems. Our evaluation shows that current models struggle to make well-calibrated response decisions. Finally, we construct additional proactive training data and train \textsc{P$^3$I-Assistant-4B}, achieving improvements on several tasks, though gains remain uneven across capabilities. Together
    
[^126]: 基于逻辑一致性的锐利且稳定的无标签校准证书

    Sharp and Stable Label-Free Calibration Certificates from Logical Coherence

    [https://arxiv.org/abs/2610.21567](https://arxiv.org/abs/2610.21567)

    本文利用极小极大定理从逻辑一致性出发，为查询条件概率预测器建立了锐利且稳定的无标签校准残差下界证书，并证明Lipschitz赌注形式既稳定又可通过在线梯度上升在独立同分布数据流上有效优化。

    

    以查询为条件的概率预测器每次回答一个逻辑相关的问题，而用于检验其校准性的标签常常缺失。针对对赌注者可使用的赌注信息的每一种限制，一个极小极大定理在答案具有有限支撑分布的条件下，确定了相应校准残差的锐利无标签下界。在不同事件的捆绑上，其均值、边缘与联合阶段可以严格分离，并且在不对其他问题作假设的前提下，一个捆绑的下界不能归因于某个单一问题。锐利的边缘下界关于答案是不连续的，且在Wasserstein距离下连续的任何无标签下界都无法保持在其常数倍以内。Lipschitz赌注给出一个稳定的下界，它对光滑的校准残差是锐利的，并且对每个正的Lipschitz常数，它为正当且仅当锐利边缘下界为正。在独立同分布的数据流上，在固定节点处进行投影在线梯度上升可以（原文摘要在此处截断）

    arXiv:2610.21567v1 Announce Type: new  Abstract: Query-conditioned probabilistic predictors answer logically related questions one at a time, and labels to check their calibration are often missing. For each restriction on what a bettor's stakes may use, a minimax theorem identifies, for finitely supported laws of the answers, the sharp label-free lower bound on the corresponding calibration residual. On bundles of different events its mean, marginal and joint stages can separate strictly, and a bundle's bound cannot be attributed to one question without assumptions on the others. The sharp marginal bound is discontinuous in the answers, and no label-free lower bound continuous in the Wasserstein distance stays within a constant factor of it. Lipschitz stakes give a stable bound, sharp for a smooth calibration residual and, for every positive Lipschitz constant, positive exactly when the sharp marginal bound is. On i.i.d. streams, projected online gradient ascent at fixed nodes gives f
    
[^127]: 一种用于基于内容媒体推荐的量子机器学习方法

    A Quantum Machine Learning Approach to Content-Based Media Recommendation

    [https://arxiv.org/abs/2610.21562](https://arxiv.org/abs/2610.21562)

    该研究提出了一种基于量子自然语言处理框架的媒体推荐系统，其以仅为经典BiLSTM基线约0.3%的参数量（4.7万 vs 1780万）达到了相当的推荐性能，而经典模型在同等参数预算下性能显著下降，展示了结构化量子表示在参数效率上的优势。

    

    我们使用量子机器学习技术开发了一个基于内容的媒体推荐系统，并利用BBC iPlayer一周数据中的文本描述，在电视节目上评估了其性能。该方法基于量子自然语言处理（QNLP）框架，将节目简介根据其语义结构映射为变分量子电路，并通过监督式成对相似性学习任务来训练这些电路的参数。我们将该量子推荐系统与在同一数据集和相同目标上训练的经典BiLSTM基线模型进行了基准对比。虽然两个模型在相似性学习任务上取得了相当的性能，但量子模型所需的参数量显著更少（4.7万 vs. 1780万）。当BiLSTM被限制在与量子模型相同的参数预算时，其性能大幅下降（平均精度均值MAP下降3.63个百分点），这表明结构化量子表示……

    arXiv:2610.21562v1 Announce Type: new  Abstract: We use quantum machine learning techniques to develop a content-based media recommender system and evaluate its performance on TV programmes using textual descriptions from one week of BBC iPlayer data. Grounded in the Quantum Natural Language Processing framework (QNLP), programme synopses are mapped to variational quantum circuits using their semantic structure. The parameters of these circuits are trained via a supervised pairwise similarity learning task.   We benchmark the quantum recommender against a classical BiLSTM baseline trained on the same dataset and objective. While both models achieve comparable performance on the similarity learning task, the quantum model requires significantly fewer parameters (47k vs. 17.8M). When the BiLSTM is constrained to a matching parameter budget, its performance degrades substantially (by 3.63 percentage points in Mean Average Precision (MAP)), indicating that the structured quantum representa
    
[^128]: DialFork：超越事实性，评估对话代理中的程序性幻觉

    DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents

    [https://arxiv.org/abs/2610.21561](https://arxiv.org/abs/2610.21561)

    DialFork是首个评估对话代理“程序性幻觉”的基准，通过在受控对话轮次位置注入四种幻觉类型构建了涵盖12个领域、24个任务的36,278个对话，并从三个粒度定义了刻画程序性偏差的评估任务。

    

    作为对话代理部署的大语言模型（LLM）被越来越多地期望遵循程序性逻辑。虽然事实性幻觉已受到广泛关注，但另一种失败模式却被忽视了：程序性幻觉，即代理产生的语句虽然单独来看流畅且事实上貌似合理，却偏离了预期的对话流程。我们提出了DialFork，这是首个在原始对话文本上衡量这种失败模式的基准。DialFork通过在受控轮次位置对符合规范的对话进行分叉并注入四种幻觉类型之一来构建，产生了涵盖12个领域、24个任务的36,278个对话；30名人工标注者验证了该构建过程。我们定义了三个评估任务，从三个粒度刻画程序性偏差：一组对话偏离符合规范对话的程度（用于监控已部署的代理）、单个对话偏离的程度（用于程序性检索……

    arXiv:2610.21561v1 Announce Type: cross  Abstract: Large language models (LLMs) deployed as conversational agents are increasingly expected to follow procedural logic. While factual hallucination has received extensive attention, a different failure mode has been overlooked: procedural hallucination, in which an agent produces individually fluent and factually plausible utterances while deviating from the intended conversational process. We introduce DialFork, the first benchmark to measure this failure mode on raw dialogue text. DialFork is built by forking conformant dialogues at controlled turn positions and injecting one of four hallucination types, yielding 36,278 dialogues across 24 tasks in 12 domains; 30 human annotators validate the construction. We define three evaluation tasks that characterize procedural deviation at three granularities: how far a set of dialogues deviates from conformant ones (monitoring deployed agents), how far a single dialogue does (procedural retrieva
    
[^129]: 通过学习SNAP门的稀疏结构降低量子位态制备的成本

    Reducing the Cost of Qudit State Preparation by Learning a Sparse Structure for SNAP Gates

    [https://arxiv.org/abs/2610.21557](https://arxiv.org/abs/2610.21557)

    本文提出一种基于可微分掩码的方法，可自动学习SNAP门中所需的相位参数及其取值，通过稀疏性约束显著减少玻色量子位态制备中需要控制的相位数量，从而降低控制成本和误差。

    

    使用SNAP-位移序列在玻色量子位中制备量子态，既需要优化门参数，又需要决定每个SNAP门应控制哪些福克能级。完全参数化的拟设为每个门中的每个福克能级分配一个独立的相位，导致相位数量急剧增长。这对中短期系统构成了一个问题，因为控制大量相位需要复杂的控制脉冲，而这些脉冲容易受到更大的控制误差影响。先前的工作通过将可训练相位限制在预定义的稀疏模式（如对角线或网格模式）中来降低这一成本，但对于特定目标态而言，这些模式可能并非最优。在本工作中，我们通过引入一种使用可微分掩码的方法来解决这一限制，该方法可以学习哪些相位是必需的以及它们的取值。稀疏性惩罚项促使去除不必要的相位，而温度退火则逐步……（摘要截断）

    arXiv:2610.21557v1 Announce Type: cross  Abstract: Preparing quantum states in bosonic qudits with SNAP--Displacement sequences requires both optimizing the gate parameters and deciding which Fock levels should be controlled by each SNAP gate. A fully parameterized ansatz assigns an independent phase to every Fock level in every gate, leading to a number of phases that grows steeply. This constitutes a problem for short to medium term systems as controlling many phases requires complex control pulses which are subject to increased control errors. Previous work reduced this cost by restricting the trainable phases to predefined sparse patterns, such as a diagonal or a grid, which however may not be optimal for a specific target state. In this work, we address this limitation by introducing a method that learns which phases are needed together with their values, using a differentiable mask. A sparsity penalty encourages the removal of unnecessary phases, while temperature annealing gradu
    
[^130]: Muon究竟何时胜出？一项配对的数据、时间与调参预算基准测试

    When Does Muon Actually Win? A Paired Data-, Time-, and Tuning-Budget Benchmark

    [https://arxiv.org/abs/2610.21555](https://arxiv.org/abs/2610.21555)

    在单GPU、配对的数据/时间/调参预算基准测试中，Muon在视觉任务上时间归一化损失显著优于AdamW，但因优化器自身耗时较高（占训练时间56%–70%），在语言任务上反而落后于AdamW。

    

    Muon在数据效率上的提升促使我们在明确的数据、时间和调参预算约束下评估其性能。在单张A100 GPU上，我们将参考的单设备Muon实现与AdamW在17.7M和33.6M参数的因果语言模型以及2.70M参数的视觉Transformer（ViT）上进行对比。我们采用一个共有的23个候选配置的开发预算来选择超参数，并使用五个匹配的随机种子、完整的验证轨迹、优化器步骤计时以及冻结的一次性测试进行评估。在相同数据量下，Muon将验证损失的曲线下面积（AUC）降低3.25%–22.44%，并在全部20个主要配对实验中降低了测试损失。在该配置下，Muon在视觉任务上将时间归一化损失面积提升19.68%–22.31%，而在语言任务上AdamW则以3.38%–6.96%的优势表现更佳。完整的Muon优化器调用（包括辅助的AdamW更新）消耗56.1%–69.6%的训练时间，相比之下AdamW基线仅占6.1%–9.1%。此外，Muon还具有更高的期望遗憾。

    arXiv:2610.21555v1 Announce Type: cross  Abstract: Muon's gains in data efficiency motivate evaluating its performance under explicit data, time, and tuning budgets. On a single A100 GPU, we compare a reference single-device Muon implementation with AdamW across 17.7M- and 33.6M-parameter causal language models and 2.70M-parameter vision transformers. A common 23-candidate development budget selects configurations evaluated with five matched seeds, full validation trajectories, optimizer-step timing, and a frozen one-shot test. Muon reduces validation-loss area under the curve by 3.25--22.44\% at equal data and lowers test loss in all 20 primary pairs. In this configuration, Muon improves time-normalized loss area} by 19.68--22.31\% on vision, while AdamW is better by 3.38--6.96\% on language. The full Muon optimizer call, including auxiliary AdamW updates, consumes 56.1--69.6\% of training time, compared with 6.1--9.1\% for the AdamW baseline. Muon also has higher expected regret unde
    
[^131]: 用于分类的成对差分学习中的不确定性量化

    Uncertainty Quantification in Pairwise Difference Learning for Classification

    [https://arxiv.org/abs/2610.21554](https://arxiv.org/abs/2610.21554)

    本文提出了一种用于分类任务的成对差分学习方法及配套的元学习技术，并将偶然不确定性与认知不确定性的分解度量扩展到该框架中，从而实现对PDL分类器预测不确定性的忠实量化。

    

    成对差分学习（PDL）的核心思想是：不同于以标准方式学习从实例到结果的映射，而是学习一个以两个实例为输入、预测相应结果之间差异的函数。给定这样一个函数，对查询实例的预测可以从每个训练样本中分别得出，然后再取平均值。我们提出了一个面向分类任务的PDL版本，并为此设计了一种合适的元学习技术。此外，我们解决了不确定性量化问题，即以忠实的方式表示和量化PDL分类器预测的不确定性。基于近期关于将总不确定性分解为偶然不确定性和认知不确定性两部分的研究工作，我们提出了将相应度量扩展到PDL设置的多种不同方法，具体方法取决于底层基础学习器如何以及在多大程度上表示不确定性。

    arXiv:2610.21554v1 Announce Type: cross  Abstract: Instead of learning a mapping from instances to outcomes in the standard way, the key idea of pairwise difference learning (PDL) is to learn a function that takes two instances as input and predicts the difference between the respective outcomes. Given a function of this kind, predictions for a query instance are derived from every training example and then averaged. We propose a version of PDL for the task of classification and propose a suitable meta-learning technique for this purpose. Moreover, we address the problem of uncertainty quantification, i.e., of representing and quantifying the uncertainty of a PDL classifier's predictions in a faithful manner. Building on recent work on the decomposition of total uncertainty into aleatoric and epistemic components, we propose different ways of extending corresponding measures to the setting of PDL, depending on how and to what extent uncertainty is represented by the underlying base lea
    
[^132]: 重新思考用于文本与图像到图像（TI2I）流匹配的无分类器引导

    Rethinking Classifier-Free Guidance for TI2I Flow Matching

    [https://arxiv.org/abs/2610.21550](https://arxiv.org/abs/2610.21550)

    本文发现CFG机制在流建模的噪声阶段对图像编辑有隐性危害，并提出一种在早期流步骤的简单修正方法，无需额外成本即可显著提升TI2I编辑的指令遵循能力和可控性。

    

    我们研究了用于可控图像编辑的文本与图像到图像（TI2I）流匹配方法。在更广泛的生成建模背景下，无分类器引导（CFG）已被确立为实现高质量输出的关键组件。然而，我们认为其在TI2I建模中的作用尚未被充分理解。具体而言，我们得出了一个令人惊讶的观察结果：CFG机制在流建模的噪声阶段对编辑具有隐性危害，阻碍了指令遵循能力和可控性。我们提出了一种针对早期流步骤的简单修正方法，能够在不增加额外成本的情况下实现更丰富的对齐与保持之间的权衡。我们的方法在像素空间和潜空间的TI2I流上均证明了其有效性。

    arXiv:2610.21550v1 Announce Type: cross  Abstract: We study Text-and-Image-to-Image (TI2I) flow matching methods for controllable image editing. In a broader generative modeling context, Classifier-Free Guidance (CFG) is established as a key component responsible for high quality outputs. However, we argue that its role in TI2I modeling is poorly understood. Specifically, we make the surprising observation that the CFG mechanism is implicitly harmful for editing in the noisy phase of flow modeling, hindering instruction following and controllability. We propose a simple correction in the early flow steps, enabling a richer alignment-preservation tradeoff at no extra cost. The effectiveness of our approach is demonstrated on both pixel-space and latent TI2I flows.
    
[^133]: Muon何时最优？径向脆弱性与极小极大几何

    When Is Muon Optimal? Radial Fragility and Minimax Geometry

    [https://arxiv.org/abs/2610.21549](https://arxiv.org/abs/2610.21549)

    该论文引入随机半径各向同性曲率模型与极化因子容许集，精确刻画了Muon缩放极化更新在任意径向定律下保持最优的几何条件，证明非标量最优响应仅来自与曲率扭结对齐的径向原子，因而无原子径向定律呈现零容量刚性。

    

    Muon优化器通过用近似极化因子替换矩阵梯度，在神经网络训练中展现出强大的实证性能。各向同性曲率模型为这一更新提供了基于曲率的解释，其核心问题是：梯度正交化何时能最小化单步局部损失代理函数。我们研究了当ICM中的固定输入半径被一般径向定律取代时，这一解释的稳定性。我们引入了随机半径各向同性曲率模型（RR-ICM）和极化因子容许集，后者将精确的缩放极化最优性编码为梯度正极化因子上的一个几何条件。我们针对任意径向定律推导出了该容许集的精确刻画，并证明所有非标量的容许响应都源自与曲率扭结对齐的径向原子。因此，每一个无原子径向定律都表现出零容量刚性：精确的缩放极化最优性仅可能在……

    arXiv:2610.21549v1 Announce Type: cross  Abstract: Muon replaces a matrix gradient by an approximate polar factor and has shown strong empirical performance in neural-network training. The Isotropic Curvature Model (ICM) provides a curvature-based explanation for this update by asking when gradient orthogonalization minimizes a one-step local-loss surrogate. We investigate the stability of this explanation when the fixed input radius of the ICM is replaced by a general radial law. We introduce the Random-Radius Isotropic Curvature Model (RR-ICM) and the Polar-Factor Admissibility Set, which encodes exact scaled-polar optimality as a geometric condition on the positive polar factor of the gradient. We derive an exact characterization of this set for arbitrary radial laws and show that all nonscalar admissible responses arise from radial atoms aligned with curvature kinks. Consequently, every atomless radial law exhibits zero-capacity rigidity: exact scaled-polar optimality is possible o
    
[^134]: 部分可观测性下约束强化学习的有限时间保证

    Finite-Time Guarantees for Constrained Reinforcement Learning under Partial Observability

    [https://arxiv.org/abs/2610.21548](https://arxiv.org/abs/2610.21548)

    本文首次为部分可观测约束马尔可夫决策过程（CPOMDP）中的强化学习提供了非渐近保证，通过将有限历史窗口表示为约束MDP的状态并开发原始-对偶算法，同时限定了原始问题中的平均奖励次优性与约束违反。

    

    我们研究约束部分可观测马尔可夫决策过程（CPOMDP）中的强化学习问题，其中智能体在满足成本约束的条件下最大化奖励，且只能利用关于底层状态的不完整信息。状态可观测性的缺失可能要求最优策略依赖于任意长的观测-动作历史，这使得规划与学习变得极具挑战性。我们利用POMDP有限记忆近似方面的最新进展，将有限历史窗口表示为约束MDP的状态。随后，我们开发了一种原始-对偶算法，首次提供了非渐近保证，能够同时限定原始CPOMDP中的平均奖励次优性和约束违反。一个关键挑战在于这种有限窗口近似会同时影响奖励值和可行策略集合。在滤波器稳定性假设下，我们推导出了有限状态模型中关联最优性与可行性的界。

    arXiv:2610.21548v1 Announce Type: new  Abstract: We study reinforcement learning in constrained partially observable Markov decision processes (CPOMDPs), where an agent maximizes reward subject to a cost constraint using only incomplete information about the underlying state. The lack of state observability may require optimal policies to depend on arbitrarily long observation-action histories, making planning and learning challenging. We leverage recent advances in finite-memory approximations of POMDPs and represent finite history windows as states of a constrained MDP. We then develop a primal--dual algorithm that provides the first non-asymptotic guarantees that jointly bound average reward suboptimality and constraint violation in the original CPOMDP. A key challenge is that this finite-window approximation affects both reward values and the set of feasible policies. Under a filter-stability assumption, we derive bounds relating optimality and feasibility in the finite-state model
    
[^135]: 联邦艺术追踪：联邦微调中的隐私泄露与数据投毒

    Federated Art Chase: Privacy Leakage and Data Poisoning in Federated Fine-Tuning

    [https://arxiv.org/abs/2610.21547](https://arxiv.org/abs/2610.21547)

    该论文通过一个可复现的实验流程，系统评估了联邦微调在非IID场景下的安全威胁，表明诚实但好奇的服务器可从单步客户端更新中恢复完整私有标签集，并揭示了联邦微调中的隐私泄露与数据投毒风险。

    

    arXiv:2610.21547v1 公告类型：新论文 摘要：联邦学习（FL）正越来越多地被用于将一个预先训练、集中微调过的模型适配到保留在用户设备上的数据，而不仅限于从头训练模型。在本为第四届联邦机器学习国际暑期学校准备的报告中，我们通过一个可复现的实验流程研究了该方法的隐私性与完整性。一个在ImageNet上预训练的MobileNetV4-Conv-Small模型先在十个Quick, Draw!绘画类别上进行集中微调（测试准确率达97.50%），随后通过仅训练头部的FedAvg在十个非IID客户端上进行适配，使客户端训练准确率从90.34%提升至91.49%，同时保持全局准确率。我们评估了四种对抗场景：第一，从一个单步客户端更新中，诚实但好奇的服务器能够恢复完整的私有标签集合，但其重建结果遵循的是类原型而非私有的绘画样本（平均SSIM仅为0.130）；第二，从一轮FedAvg聚合中，标签数量恢复……

    arXiv:2610.21547v1 Announce Type: new  Abstract: Federated learning (FL) is increasingly used to adapt a pretrained, centrally fine-tuned model to data that remain on user devices, beyond training a model from scratch. In this report, prepared for the 4th Federated Machine Learning International Summer School, we study the privacy and integrity of this approach with a reproducible pipeline. An ImageNet-pretrained MobileNetV4-Conv-Small model is fine-tuned centrally on ten Quick, Draw! classes (97.50% test accuracy) and then adapted with head-only FedAvg over ten non-IID clients, which raises client training accuracy from 90.34% to 91.49% while preserving global accuracy. We evaluate four adversarial scenarios. First, from a single one-step client update, an honest-but-curious server recovers the complete private label set, but its reconstructions follow class prototypes rather than the private drawings (mean SSIM 0.130). Second, from a one-round FedAvg aggregate, label-count recovery f
    
[^136]: 激活函数邻接层归一化：对逼近能力的影响

    Activation-Adjacent Layer Normalization: Effects on Approximation Power

    [https://arxiv.org/abs/2610.21527](https://arxiv.org/abs/2610.21527)

    本文通过宽度增广与归一化的局部线性化技术，精确刻画了激活后归一化（AN）和激活前归一化（PN）网络在仅需常数级额外宽度的情况下逼近任意给定宽度 MLP 的条件与误差界。

    

    层归一化（LN）和均方根归一化（RMSNorm）依赖于其输入，在推理阶段仍保持非线性。因此，将它们插入激活函数旁边可能会改变网络所表示的函数。我们研究了浅层激活后归一化（AN）网络和激活前归一化（PN）网络在何种条件下能够逼近一个给定的宽度为 N 的单隐层多层感知机（MLP）。将 RMSNorm 视为依赖于输入的缩放操作，我们利用宽度增广技术在紧致特征集上对归一化进行局部线性化。对于 AN 网络，激活函数取值范围无界时可实现宽度 N+1；而对于任意连续非常值激活函数，宽度 N+r 即可，其误差为 O_f(r^{-1})。对于 PN 网络，仿射参数（或在无仿射参数情况下的一阶正齐次性）可恢复宽度 N+1；一般连续激活函数则可实现宽度 N+r 且误差趋于零。相应的 LN 构造使用中心化坐标对，分别得到宽度 N+2 和 N+2r。

    arXiv:2610.21527v1 Announce Type: new  Abstract: Layer normalization (LN) and root mean square normalization (RMSNorm) depend on their inputs and remain nonlinear at inference. Inserting either next to an activation may therefore change the function a network represents. We study when shallow activation-normalized (AN) and preactivation-normalized (PN) networks can approximate a prescribed width-$N$ one-hidden-layer MLP. Viewing RMSNorm as input-dependent scaling, we use width augmentation to locally linearize normalization on compact feature sets. For AN networks, an unbounded activation range allows width $N+1$, while every continuous nonconstant activation admits width $N+r$ with error $O_f(r^{-1})$. For PN networks, affine parameters or, without them, degree-one positive homogeneity recover width $N+1$; general continuous activations admit width $N+r$ with error tending to zero. The corresponding LN constructions use centered coordinate pairs, yielding widths $N+2$ and $N+2r$, resp
    
[^137]: 当KV缓存比GPU更慢时：为上下文并行预填充扩展远程KV缓存层

    When the KV Cache Is Slower Than the GPU: Scaling Remote KV Tiers for Context-Parallel Prefill

    [https://arxiv.org/abs/2610.21500](https://arxiv.org/abs/2610.21500)

    在长上下文LLM服务中，远程KV缓存层必须提供超过约2.5 GB/s的临界带宽（B* = sR）才能比重算提示词更快，否则缓存命中反而更慢，且同步加载会让GPU几乎闲置、p99尾延迟从4秒暴涨至83秒。

    

    跨请求复用注意力的KV缓存是降低长上下文LLM服务中预填充成本的标准做法，生产系统越来越多地在每引擎缓存之外配备一个由所有引擎共享的远程层。我们在GLM-5.3（一个采用FP8混合专家架构和多头潜在注意力（MLA）的模型）的分离式部署中，研究了这样一个远程层，硬件为8xB300节点，运行预填充上下文并行（PCP8）。启用远程层（采用带Valkey存储的LMCache）后，中位首个token生成时间（TTFT）保持不变，但16k token请求的p99延迟从3.96秒上升到17.3秒。通过在隔离引擎上重放生产请求流，我们发现一次远程命中耗时0.88秒，而重新计算提示词仅需0.31秒；并且同步加载使一个以命中为主的请求流变成了不稳定的队列：p99延迟高达83.32秒，而GPU利用率仅为7%。一个两项成本模型解释了这两种现象：只有当远程层提供超过B* = sR的带宽时（对于该模型和GPU约为2.53 GB/s），缓存命中才能带来收益。（摘要在此处截断）

    arXiv:2610.21500v1 Announce Type: cross  Abstract: Reusing the attention KV cache across requests is a standard way to cut prefill cost in long-context LLM serving, and production systems increasingly back the per-engine cache with a remote tier shared by every engine. We study such a tier in a disaggregated deployment of GLM-5.3, an FP8 mixture-of-experts model with multi-head latent attention (MLA), on 8xB300 nodes running prefill context parallelism (PCP8). Enabling the remote tier (LMCache with a Valkey store) left median time-to-first-token unchanged but raised the 16k-token p99 from 3.96 s to 17.3 s. Replaying a production request stream on an isolated engine, we find that a remote hit took 0.88 s against 0.31 s to recompute the prompt, and that synchronous loading turned a hit-heavy stream into an unstable queue: p99 83.32 s with the GPUs 7% busy. A two-term cost model explains both: a hit pays off only if the tier delivers more than B* = sR (2.53 GB/s for this model and GPU), a
    
[^138]: 相位核提升稠密联想记忆的存储容量

    Phase Kernel Lifts Capacity of Dense Associative Memory

    [https://arxiv.org/abs/2610.21493](https://arxiv.org/abs/2610.21493)

    本文提出一类非线性相位核并构建S-Hop网络，在保持指数级存储容量和收敛保证的同时大幅提升霍普菲尔德网络的实际检索负载，在ImageNet上首次失败负载提升达50倍。

    

    稠密联想记忆是现代注意力机制的基础；然而，采用线性核的现代霍普菲尔德网络（MHNs）在高记忆负载下会产生重叠的吸引域和虚假状态，限制了其理论上指数级存储容量在实际中的发挥。我们构建了一类非线性相位核，能够修改能量景观和检索动力学，并证明在核光滑性、记忆分离度和记忆分布的给定条件下，这些核可保持指数级存储容量和局部不动点收敛性。该框架由S形相位霍普菲尔德网络实例化，它保证单调能量下降并缓解实际容量损失。S-Hop在CIFAR-10和TinyImageNet上的单步噪声检索负载分别比均匀霍普菲尔德网络（U-Hop）提升了8.3倍和4倍，在ImageNet上的首次失败负载达到了50倍的提升。

    arXiv:2610.21493v1 Announce Type: new  Abstract: Dense associative memory underlies modern attention mechanisms; however, Modern Hopfield Networks (MHNs) with linear kernels yield overlapping attractor basins and spurious states under high memory load, limiting their theoretically exponential storage capacity in practice. We construct a class of nonlinear phase kernels that modify the energy landscape and retrieval dynamics, and prove that these kernels preserve exponential storage capacity and local fixed-point convergence under given conditions on kernel smoothness, memory separation, and memory distributions. The framework is instantiated by the Sigmoidal Phase Hopfield Network (S-Hop), which guarantees monotone energy descent and mitigates practical capacity loss. S-Hop has $8.3\times$ and $4\times$ gains in one-step noisy retrieval load over Uniform Hopfield Networks (U-Hop) on CIFAR-10 and TinyImageNet respectively, and reaches a $50\times$ gain in first-failure load for ImageNet
    
[^139]: Kolmogorov-Arnold 网络：维度灾难与结构性偏置

    Kolmogorov-Arnold Networks: Curse of Dimensionality and Structural Bias

    [https://arxiv.org/abs/2610.21480](https://arxiv.org/abs/2610.21480)

    该论文证明了标准样条-KAN 在最坏情形下仍受维度灾难制约——所需样条系数数量随输入维度指数增长，表明 KAN 的维度无关逼近优势源于对目标函数类的假设而非架构本身，且 KAN 逼近器可精确等价转换为三次 ReLU 激活的 MLP。

    

    当目标函数能够通过足够光滑的单变量边函数进行表示时，Kolmogorov-Arnold 网络（KAN）具有与维度无关的逼近指数。我们探讨这种良好的标度特性究竟属于网络架构本身，还是属于所假设的目标函数类。对于任意深度的标准样条-KAN，我们在各向同性 Sobolev 球上证明了最坏情形下界：当精度低于某个与维度无关的阈值时，所需的样条系数数量至少随输入维度呈指数增长。相比之下，现有的与组合结构相匹配的 MLP 结果表明，在低元数目标上取得良好逼近指数并非 KAN 所独有。我们还重新审视了实证比较：在四个标度律目标上，由已知分解构造的样条-KAN 逼近器可以精确转换为采用三次 ReLU 激活函数 σ(x)=(max{0,x})³ 的 MLP，尽管后者的完整稠密参数量更大。在受控的岭剖面实验中……

    arXiv:2610.21480v1 Announce Type: new  Abstract: Kolmogorov-Arnold networks (KANs) have a dimension-independent approximation exponent when the target admits a sufficiently smooth representation by univariate edge functions. We ask whether this favorable scaling belongs to the architecture or to the assumed target class. For arbitrary-depth standard spline-KANs, we prove a worst-case lower bound on isotropic Sobolev balls: below a dimension-independent accuracy threshold, the required number of spline coefficients grows at least exponentially with input dimension. Existing composition-matched MLP results, by contrast, show that favorable exponents on low-arity targets are not exclusive to KANs. We also revisit empirical comparisons. On four scaling-law targets, spline-KAN approximants constructed from known decompositions transfer exactly to MLPs with the cubic ReLU activation $\sigma(x)=(\max\{0,x\})^3$, although their full dense parameter counts are larger. In controlled ridge-profil
    
[^140]: DuET：面向钢琴演奏转乐谱的音符-帧双重编码

    DuET: Note-Frame Dual Encoding for Piano Performance-to-Score Transcription

    [https://arxiv.org/abs/2610.21467](https://arxiv.org/abs/2610.21467)

    提出DuET双编码器Transformer，通过音符与帧的双向交叉注意力及TimeRoPE、TPRoPE位置编码，将音符级钢琴MIDI转谱与帧级节拍跟踪联合建模，提升多种钢琴流派的乐谱转录效果。

    

    将富有表现力的钢琴演奏转录为乐谱，需要在潜在的节拍结构内解释单个音符的时值。然而，乐谱属性与音符相关联，而节拍可能在不与音符起始位置重合的情况下出现。我们提出了DuET，一个双编码器Transformer，它联合学习音符级的演奏MIDI到乐谱（PM2S）转录和帧级的节拍与强拍跟踪。双向交叉注意力连接音符编码器和帧编码器，同时自回归解码器在两者的表示之间交替，以预测结构化的乐谱属性。我们进一步引入TimeRoPE和时间-音高旋转位置编码（TPRoPE），以建立共享的演奏时间参考，并显式编码相对的时间与音高关系。在古典钢琴数据集ASAP以及我们策划的、涵盖多种流派并侧重流行钢琴音乐的Gangqinpu数据集上的评估表明……

    arXiv:2610.21467v1 Announce Type: cross  Abstract: Transcribing expressive piano performances into scores requires interpreting individual note timings within an underlying metrical structure. However, score attributes are associated with notes, whereas beats may occur without coincident note onsets. We introduce DuET, a Dual-Encoder Transformer that jointly learns note-level performance MIDI-to-score (PM2S) transcription and frame-level beat and downbeat tracking. Bidirectional cross-attention connects note and frame encoders, while an autoregressive decoder alternates between their representations to predict structured score attributes. We further introduce TimeRoPE and Time-Pitch Rotary Positional Encoding (TPRoPE) to establish a shared performance-time reference and explicitly encode relative temporal and pitch relationships. Evaluations on ASAP, a classical piano dataset, and our curated Gangqinpu dataset, which spans multiple genres with a focus on popular piano music, demonstrat
    
[^141]: 面向工具使用智能体的轨迹级推测与语义验证

    Trajectory-Level Speculation with Semantic Verification for Tool-Using Agents

    [https://arxiv.org/abs/2610.21463](https://arxiv.org/abs/2610.21463)

    SpecAgent将强模型的角色从逐步生成者转变为弱模型草稿的审阅者，弱模型在起草时执行只读工具调用以确保每一步基于真实观测，强模型则通过单次预填充对草稿片段进行语义验证并提交最长一致前缀，从而实现高效的轨迹级推测协作。

    

    大语言模型智能体通过长序列的工具调用解决任务，当前主流设计是由单个强模型生成每一步，尽管许多步骤对于小得多的模型而言已足够常规。现有的分工方式仍让强模型担任生成者：路由器和协作智能体决定由哪个模型处理每个查询或步骤，而推测式方法则一次一步地对照强模型自身的动作来检查小模型的草稿。我们提出SpecAgent，这是一种协作协议，它将强模型从每一步的“作者”转变为弱模型所起草步骤的“审阅者”。弱模型在起草过程中执行只读工具调用，使每一步都建立在真实观测的基础上，并暂停在第一个将修改环境的动作上。随后，强模型在单次预填充过程中验证所起草的片段，提交其判定为语义上（此处摘要原文截断）的最长前缀。

    arXiv:2610.21463v1 Announce Type: cross  Abstract: Large language model agents solve tasks through long sequences of tool calls, and the prevailing design has a single strong model generate every step, although many steps are routine enough for a much smaller model. Existing ways of sharing this work keep the strong model as the generator: routers and collaborative agents decide which model handles each query or step, and speculative methods check a small model's drafts one step at a time, usually against the strong model's own action. We introduce SpecAgent, a collaboration protocol that turns the strong model from the author of each step into a reviewer of steps drafted by a weak model. The weak model executes read-only tool calls as it drafts, so that every step is grounded in real observations, and holds the first action that would modify the environment. The strong model then verifies the drafted segment in a single prefill pass, commits the longest prefix it judges semantically v
    
[^142]: V-SHARE：基于优势的算法的可证明收敛保证

    V-SHARE: Provable Convergence Guarantees for Advantage-based Algorithms

    [https://arxiv.org/abs/2610.21448](https://arxiv.org/abs/2610.21448)

    本文提出V-SHARE算法，首次为基于优势的价值学习方法提供了完整的理论收敛保证（包括渐近收敛和非渐近有限样本保证），并证明其能缓解异步Q学习的稀有动作瓶颈，同时揭示了收缩性与方差之间的权衡。

    

    在强化学习中，基于优势的价值学习方法与标准Q学习不同，它通过将动作价值函数分解为依赖于状态的价值分量和动作特定的优势分量，从而在动作之间共享信息。虽然这一机制在表格型和深度强化学习中都已被证明是有效的，但对其理论理解仍然不完整。我们提出了V-SHARE，这是一种基于优势的价值学习算法，它将跨动作价值共享与重要性加权的贝尔曼残差校正相结合。据我们所知，我们首次对基于优势的算法提供了完整的理论刻画，包括渐近收敛性和非渐近的有限样本保证。我们识别了一个适用范围，其中V-SHARE缓解了异步Q学习的稀有动作瓶颈，并揭示了更强的确定性收缩与增加的重要性加权方差之间的收缩-方差权衡。

    arXiv:2610.21448v1 Announce Type: cross  Abstract: In reinforcement learning, advantage-based value-learning methods, unlike standard Q-learning, share information across actions by decomposing the action-value function into a state-dependent value component and action-specific advantages. While this mechanism has proven effective in both tabular and deep reinforcement learning, its theoretical understanding remains incomplete. We introduce V-SHARE, an advantage-based value-learning algorithm that combines cross-action value sharing with an importance-weighted Bellman-residual correction. To our knowledge, we provide the first complete theoretical characterization of an advantage-based algorithm, including both asymptotic convergence and non-asymptotic finite-sample guarantees. We identify a regime in which V-SHARE mitigates the rare-action bottleneck of asynchronous Q-learning and uncover a contraction--variance tradeoff between stronger deterministic contraction and increased importa
    
[^143]: 各奖励归于其相关词元：面向多奖励强化学习的奖励专属词元级信用分配

    Each Reward to Its Relevant Tokens: Reward-Specific Token-Level Credit Assignment for Multi-Reward RL

    [https://arxiv.org/abs/2610.21439](https://arxiv.org/abs/2610.21439)

    提出MRTPO方法，通过对比有/无验证器信息时的下一词元分布的Jensen-Shannon距离，将每个奖励的优势精准分配到其相关词元上，无需辅助模型或词元级标注即可实现多奖励强化学习中的奖励专属词元级信用分配。

    

    语言模型越来越多地在多个可验证奖励下同时训练，例如工具调用的正确性与格式，或复杂指令的各项约束条件。奖励解耦归一化（GDPO）对每个奖励分别进行归一化，使不同的奖励组合保持各自独立的优势，但它仍然将每个奖励的优势均匀地分散到整个响应上，而实际上每个验证器只检查少数几个位置。我们提出MRTPO，它利用每个验证器所检查的信息，将每个奖励的优势分配给与其相关的词元，且无需辅助模型或词元级标注。对于每个奖励，策略分别在有和没有该奖励验证器信息的情况下重放自身的响应，两个下一词元分布之间的逐词元Jensen-Shannon距离用于衡量每个词元的相关性，并据此重新分配优势。在文本到SQL、多约束指令遵循等任务上……

    arXiv:2610.21439v1 Announce Type: new  Abstract: Language models are increasingly trained against several verifiable rewards at once, such as the correctness and the format of a tool call, or the individual constraints of a complex instruction. Reward-decoupled normalization (GDPO) normalizes each reward separately so that different reward combinations keep distinct advantages, but it still spreads every reward's advantage evenly over the whole response, although each verifier checks only a few positions. We propose MRTPO, which assigns the advantage of each reward to the tokens relevant to it, using the information its verifier checks, without auxiliary models or token-level annotations. For each reward, the policy replays its own response with and without that reward's verifier information, and the per-token Jensen-Shannon distance between the two next-token distributions measures each token's relevance and reassigns the advantage. On text-to-SQL, multi-constraint instruction followi
    
[^144]: HyCo：面向高性价比移动-云协同推测解码的混合草稿生成机制

    HyCo: Hybrid Drafting for Cost-Efficient Mobile-Cloud Speculative Decoding

    [https://arxiv.org/abs/2610.21436](https://arxiv.org/abs/2610.21436)

    HyCo提出混合草稿生成机制，将服务器模型的原生多词预测头部署到移动NPU上并与小型语言模型单次推理协同工作，配合量化和ε阈值松弛策略，使移动-云推测解码的服务器GPU时间降低约29%-39%。

    

    设备-云协同推测解码能够减少服务器的计算开销，但使用小型语言模型（SLM）生成深度草稿树需要在移动神经处理单元（NPU）上进行多次推理。我们提出了HyCo，它将草稿深度分配给在设备端执行的服务器模型原生多词预测（MTP）头，并将替代分支分配给单次SLM推理。服务器只需返回该MTP头所需的隐状态、输入嵌入和键值缓存条目，而无需执行其原有的顺序计算步骤。通过量化、受限词表和1K-token窗口设置，两个草稿模型均能适配NPU运行。此外，一种ε阈值规则进一步放宽了对验证器采样分布的精确保持要求。在通过LTE/5G网络连接Galaxy S26 Ultra的单客户端运行中，与服务器端MTP自推测相比，HyCo在Spec-Bench上将每个输出token的服务器GPU时间降低了38.6%，在32K-token摘要任务上降低了29.2%。

    arXiv:2610.21436v1 Announce Type: new  Abstract: Device-cloud speculative decoding reduces server computation, but growing a deep draft tree with a small language model (SLM) requires repeated passes on a mobile neural processing unit (NPU). We present HyCo, which assigns depth to the server model's native multi-token prediction (MTP) head, executed on the device, and alternative branches to one SLM pass. The server returns the hidden state, input embedding, and key-value entries needed by the head instead of executing its sequential steps. Quantization, a restricted vocabulary, and a 1K-token window fit both drafters to the NPU. An {\epsilon}-threshold rule additionally relaxes exact preservation of the verifier's sampling distribution. In single-client runs with a Galaxy S26 Ultra over LTE/5G, HyCo reduces server GPU time per output token relative to server-side MTP self-speculation by 38.6% on Spec-Bench and 29.2% on 32K-token summarization. With the same acceptance rule, three Spec
    
[^145]: 面向自适应VLA动作块执行的策略对齐预测验证

    Policy-Aligned Predictive Verification for Adaptive VLA Action-Chunk Execution

    [https://arxiv.org/abs/2610.21434](https://arxiv.org/abs/2610.21434)

    本文提出策略对齐预测验证器（PAV），在执行前于VLA内部任务条件表征空间中预判候选动作前缀的后果并评估其与任务成功后续行动的兼容性，从而实现更可靠的自适应动作块执行决策。

    

    视觉-语言-动作（VLA）策略通过生成动作序列来减少重复推理，但决定每个序列应执行多少部分对任务成功至关重要。现有的自适应方法依据生成稳定性或预测一致性来做出这一决策，然而这些信号无法确定执行某个候选前缀是否能保持完成任务的能力。我们提出了一种策略对齐预测验证器（PAV），它能够在执行前预判候选动作前缀的后果，并评估其与成功后续行动的兼容性。我们的方法并非利用未来预测来生成动作，而是在VLA内部的任务条件表征空间中对已有提议进行评估。为了从终端任务结果中学习局部判断，我们将轨迹级生存监督与一种定位先验相结合，该先验能够在失败的执行轨迹中保留有用的行为。由此得到的有效（摘要在此处截断）

    arXiv:2610.21434v1 Announce Type: cross  Abstract: Vision-language-action (VLA) policies generate action sequences that reduce repeated inference, but deciding how much of each sequence to execute is critical to task success. Existing adaptive methods make this decision using generation stability or prediction agreement, yet these signals do not establish whether executing a candidate prefix preserves the ability to complete the task. We introduce a policy-aligned predictive verifier (PAV) that anticipates the consequences of candidate action prefixes and evaluates their compatibility with successful continuation before execution. Rather than using future prediction to generate actions, our method evaluates existing proposals in the VLA's internal task-conditioned representation space. To learn local judgments from terminal task outcomes, we combine trajectory-level survival supervision with a localization prior that preserves useful behavior within failed rollouts. The resulting valid
    
[^146]: ManiFusion：通过流形空间中的叠加解锁高吞吐量生成

    ManiFusion: Unlocking the High-Throughput Generation via Superposition in Manifold Space

    [https://arxiv.org/abs/2610.21433](https://arxiv.org/abs/2610.21433)

    ManiFusion提出了一个与骨干网络无关的框架，通过在去噪骨干前后对多个样本进行融合与解融合，并采用轨迹融合调度来对齐去噪阶段，从而在单次去噪中并行处理多个样本，突破单样本延迟优化的瓶颈，实现高吞吐量的扩散模型图像生成。

    

    扩散模型能够实现高质量的图像生成，但由于需要在多个时间步上进行迭代去噪，其计算开销依然很高。现有的加速方法主要着眼于降低单样本延迟，而这一方向上的收益正趋于饱和。我们转而探索一个与之正交的、面向吞吐量的方向：在单次去噪计算中同时处理多个样本。为此，我们提出了ManiFusion，一个与骨干网络无关的框架，通过融合空间计算实现多样本生成。ManiFusion通过在去噪骨干之前对多个样本进行融合、在其后通过解融合将它们分离，从而联合处理多个样本。我们从几何视角阐释了多个样本如何共享一个融合表示，同时仍保留引导各自生成轨迹所需的信息。为了使融合分配更好地与去噪阶段相适配，ManiFusion采用了轨迹融合调度

    arXiv:2610.21433v1 Announce Type: cross  Abstract: Diffusion models achieve high-quality image generation but remain computationally expensive due to iterative denoising across many timesteps. Existing acceleration methods mainly reduce per-sample latency, but gains along this axis are becoming saturated. We instead explore an orthogonal throughput-oriented direction by processing multiple samples within a single denoising evaluation. To this end, we propose ManiFusion, a backbone-agnostic framework for multi-sample generation through fused-space computation. ManiFusion jointly processes multiple samples by fusing them before the denoising backbone and separating them afterward through unfusion. We provide a geometric perspective on how multiple samples can share a fused representation while retaining the information needed to guide their individual generation trajectories. To better align fusion allocation with denoising stages, ManiFusion uses Trajectory-Fusion Scheduling (TFS), whic
    
[^147]: 历史能带来什么？面向大学轨迹预测的日历控制与信息时机审计

    What Does History Add? Calendar Controls and Information-Timing Audits for University Trajectory Forecasting

    [https://arxiv.org/abs/2610.21419](https://arxiv.org/abs/2610.21419)

    该研究通过严格的日历长度控制与信息时机审计发现，大学轨迹预测中六个月历史摘要相比仅使用历史长度指标并无显著的额外预测增益，但全历史摘要确实优于其对应的日历控制。

    

    我们审计了基于规则的大学轨迹模拟器中历史摘要的预测价值。通过十七种岭回归设定和持久性基线，我们将当前计划、过去状态与过去计划摘要，以及仅含日历信息的历史长度指标分离开来。在训练、验证以及两次按家族留出的评估中，我们共重建了16,750条路线和804,000条月度记录。在原始的40个家族集合上，六个月历史将六步归一化RMSE从0.159234降至0.151522，但其相对于仅长度控制的差异为-0.000895（描述性配对95%区间[-0.001957,0.000248]）。在使用先前任务中未涉及的30个家族的固定权重评估中重现了这一模式：相对于当前状态预测从0.156480降至0.150074，但相对于长度控制为-0.000558 [-0.001783,0.000728]。这些不确定的比较结果不能推广到全历史摘要，后者在其自身的日历控制上确有改进。探索性分解发现……（摘要原文在此处截断）

    arXiv:2610.21419v1 Announce Type: new  Abstract: We audit the predictive value of historical summaries in a rule-based university trajectory simulator. Seventeen ridge specifications and persistence separate current plans, past-state and past-plan summaries, and calendar-only history-length indicators. Across training, validation and two family-held-out evaluations, we reconstruct 16,750 routes and 804,000 monthly records. On the original 40-family set, six-month history lowers six-step normalized RMSE from 0.159234 to 0.151522, but its difference from a length-only control is -0.000895 (descriptive paired 95% interval [-0.001957,0.000248]). Fixed weights on 30 families unused in identified prior tasks reproduce this pattern: 0.156480 to 0.150074 versus current-state prediction, but -0.000558 [-0.001783,0.000728] versus the length control. These uncertain comparisons do not generalize to all-history summaries, which improve on their own calendar control. Exploratory decomposition finds
    
[^148]: 在多个时空层级结构上学习预测

    Learning to Forecast on Multiple Spatiotemporal Hierarchies

    [https://arxiv.org/abs/2610.21414](https://arxiv.org/abs/2610.21414)

    提出了一种图深度学习框架，将多个时空层级结构之间的差异作为多任务正则化进行联合训练，从而提升时空预测的效果。

    

    分层预测利用聚合结构，为同一个底层时空预测问题提供多个视角，同时确保预测之间的连贯性。单个序列提供细粒度的信息，而逐级粗化的聚合则在不同尺度上揭示时间和空间模式。在相关的深度学习架构中，层级结构决定了哪些聚合序列可用，以及它们如何与单个序列共同处理（例如，通过协调算子或消息传递算子）。因此，不同的层级结构为预测同一组时间序列提供了不同的学习信号和归纳偏置。然而，现有的架构通常只在单一的、且往往是固定的结构上进行训练。在这项工作中，我们提出了一个图深度学习框架，该框架利用不同层级结构之间的变化作为多任务正则化。我们表明，联合学习……

    arXiv:2610.21414v1 Announce Type: new  Abstract: Hierarchical forecasting exploits aggregation structures that expose multiple views of the same underlying spatiotemporal forecasting problem, while ensuring coherence among forecasts. Individual series provide fine-grained information, while progressively coarser aggregates reveal temporal and spatial patterns at different scales. In the associated deep learning architectures, the hierarchy determines which aggregate series are available and how they are processed together with individual series (e.g., through reconciliation or message-passing operators). Different hierarchies thus provide distinct learning signals and inductive biases for forecasting the same collection of time series. Nonetheless, existing architectures are typically trained on a single, often fixed, structure. In this work, we propose a graph deep learning framework that leverages variations across hierarchies as multi-task regularization. We show that jointly learni
    
[^149]: 多奖励破解：从聚合诱导漂移到信任锚定防御

    Multi-Reward Hacking: From Aggregation-Induced Drift to Trust-Anchored Defense

    [https://arxiv.org/abs/2610.21404](https://arxiv.org/abs/2610.21404)

    论文揭示了对称聚合多奖励会引发“多奖励破解”现象——即聚合奖励提升而主任务性能反而下降，并证明了确定性条件均值GRPO在辅助奖励更易获得时必然收敛到偏向辅助的行为，同时提出信任锚定门控（TAG）作为防御方法。

    

    为了在提升任务性能的同时鼓励理想的响应特性，大语言模型（LLM）后训练通常将主要奖励（如正确性）与辅助奖励（如简洁性或效率）相结合。然而，这些奖励的聚合方式会显著影响训练结果。我们证明，对称聚合（即平等对待各个奖励通道）可能引发一种失效模式：即使各单项奖励本身是有意义的，聚合奖励仍会提升，而主任务性能却在下降。我们通过一步式多奖励决策形式化来研究这种由聚合诱导的失效，并刻画了其何时构成多奖励破解。我们证明，在对称聚合下，当辅助奖励更容易被可靠获得时，确定性的条件均值GRPO无论初始化如何都会收敛到偏向辅助奖励的行为。我们提出信任锚定门控（TAG），首先基于……选出锚定通道（摘要在此处被截断）。

    arXiv:2610.21404v1 Announce Type: cross  Abstract: To improve task performance while encouraging desirable response properties, LLM post-training often combines a primary reward (e.g., correctness) with auxiliary rewards (e.g., brevity or efficiency). However, how these rewards are aggregated can critically affect training outcomes. We show that symmetric aggregation, which treats reward channels equally, can induce a failure where aggregate reward improves while primary-task performance degrades, even when individual rewards are meaningful. We investigate this aggregation-induced failure through a one-step multi-reward decision formulation, characterizing when it constitutes multi-reward hacking. We prove that, under symmetric aggregation, deterministic conditional-mean GRPO converges to auxiliary-favoring behavior when auxiliary rewards are more reliably obtained, regardless of initialization. We propose Trust-Anchored Gating (TAG), which first elects an anchor channel based on the p
    
[^150]: 超越不变性：面向多模态学习的按因子选择性处理分布偏移

    Beyond Invariance: Factor-wise Selective Treatment of Shifts for Multimodal Learning

    [https://arxiv.org/abs/2610.21403](https://arxiv.org/abs/2610.21403)

    提出FaShift框架，超越传统不变性方法，通过按因子诊断分布变化、干预效应和跨环境稳定性，对多模态学习中的偏移信息进行选择性抑制、保留或削弱。

    

    分布偏移是鲁棒多模态学习面临的根本性挑战。现有方法通常通过抑制依赖环境的变异来促进不变性，从而隐式地对偏移信息施加统一处理。然而，偏移信息可能产生不同的预测后果：有些对预测贡献甚微，有些仍然可靠地包含信息，有些则表现出依赖环境的预测效应。这就引出了一个根本性问题：哪些偏移信息应该被抑制、保留或削弱？为了解决这一问题，我们提出了FaShift，一个按因子处理偏移的框架，它根据潜在表示因子的分布变化、干预效应和跨环境稳定性对其进行诊断。通过分解多模态表示并评估这些互补属性，FaShift能够区分非预测性因子、稳定预测性因子和预测性（因子）……

    arXiv:2610.21403v1 Announce Type: new  Abstract: Distribution shifts pose a fundamental challenge to robust multimodal learning. Existing methods typically promote invariance by suppressing environment-dependent variations, implicitly applying a uniform treatment to shifted information. However, shifted information can have different predictive consequences: some contributes little to prediction, some remains reliably informative, and some exhibits environment-dependent predictive effects. This raises a fundamental question: which shifted information should be suppressed, preserved, or attenuated? To address this question, we propose FaShift, a factor-wise shift framework that diagnoses latent representation factors according to their distributional changes, intervention effects, and cross-environment stability. By factorizing multimodal representations and evaluating these complementary properties, FaShift distinguishes non-predictive factors, stably predictive factors, and predictive
    
[^151]: 基于边缘AI的光无线链路故障检测与误码率诊断

    Fault Detection and Bit-Error Rate Diagnostics in Optical Wireless Links Using Edge-AI

    [https://arxiv.org/abs/2610.21402](https://arxiv.org/abs/2610.21402)

    本文提出了一种基于ESP32-S3微控制器与TinyML的自主自诊断光无线接收节点，可在设备端实时提取误码率等信道特征并实现故障检测，无需依赖云端或外部计算资源。

    

    光无线通信（OWC）系统能够提供高带宽、空间安全且无射频干扰的数字传输。然而，直接自由空间光链路对视距中断、光衰减、高环境光饱和以及收发器失准等因素极为敏感。传统的信道诊断框架依赖于外部计算节点或基于云的处理，这会引入严重的诊断延迟并降低边缘节点的自主性。本文提出了一种自主、自诊断的光无线接收节点，该节点将物理层信号遥测提取与设备端TinyML分类引擎相集成。该接收器基于ESP32-S3微控制器构建，采用5样本多数投票机制恢复1000比特/秒的开关键控（OOK）数字有效载荷，同时实时提取信道特征：误码率（BER）、接收信号电压等。

    arXiv:2610.21402v1 Announce Type: new  Abstract: Optical Wireless Communication (OWC) systems provide high-bandwidth, spatially secure, and RF-interference-free digital transmission. However, direct free-space optical links suffer from extreme sensitivity to line-of-sight disruptions, optical attenuation, high ambient light saturation, and transmitter-receiver misalignment. Traditional channel diagnostic frameworks rely on external computing nodes or cloud-based processing, which introduces severe diagnostic latency and reduces edge node autonomy. This paper presents an autonomous, self-diagnosing optical wireless receiver node that integrates physical-layer signal telemetry extraction with an on-device TinyML classification engine. Built around an ESP32-S3 microcontroller, the receiver recovers 1000 bit/s On-Off Keying (OOK) digital payloads using a 5-sample majority voting mechanism while concurrently extracting real-time channel features: Bit Error Rate (BER), received signal voltag
    
[^152]: 面向车辆路径问题错误诊断的自引导大语言模型推理

    Self-Guided LLM Reasoning for Error Diagnosis in Vehicle Routing Problems

    [https://arxiv.org/abs/2610.21393](https://arxiv.org/abs/2610.21393)

    本文提出一种自引导推理框架，让大语言模型自主生成代码以提供结构化见解，从而辅助诊断车辆路径问题实例中的错误，将错误检测性能显著提升至接近完美准确率。

    

    车辆路径问题（VRPs）是许多现实应用的基础。大语言模型（LLMs）的最新进展推动了解决这些NP难问题的研究进展。然而，大多数现有方法都假设输入实例是无错误的，忽略了现实世界数据中普遍存在的错误。在本工作中，我们系统地研究了LLM检测（和纠正）VRP实例中错误的能力，并识别出其关键局限性，尤其是在处理复杂实例或会显著增加LLM推理负担的错误时。为克服这些挑战，我们提出了一个自引导推理框架，在该框架中LLM自主生成代码，提供结构化见解以辅助错误诊断。在多种问题设置和不同LLM上的实验表明，我们的框架显著提升了错误检测性能，达到了接近完美的准确率。

    arXiv:2610.21393v1 Announce Type: new  Abstract: Vehicle Routing Problems (VRPs) are fundamental to many real-world applications. Recent advances in large language models (LLMs) have spurred progress in solving these NP-hard problems. However, most existing approaches assume that input instances are error-free, neglecting the presence of faults commonly encountered in real-world data. In this work, we systematically investigate the capability of LLMs to detect (and correct) errors in VRP instances, and identify key limitations, particularly in handling complex instances or errors that substantially increase the LLM's reasoning burden. To overcome these challenges, we propose a self-guided reasoning framework in which LLMs autonomously generate code that provides structured insights to aid in error diagnosis. Experiments across diverse problem settings and LLMs demonstrate that our framework significantly improves error detection performance, achieving near-perfect accuracy.
    
[^153]: 校准语言如何塑造多语言语言模型中的块影响剪枝

    Calibration Language Shapes Block-Influence Pruning in Multilingual Language Models

    [https://arxiv.org/abs/2610.21383](https://arxiv.org/abs/2610.21383)

    该论文首次系统揭示剪枝校准语言会显著影响块影响排序——印度语言的深层块BI比英语高出20.7%-55.5%，导致剪枝选块不同，而多语言平均校准可有效缓解剪枝带来的困惑度损失。

    

    块影响度通过计算Transformer块的输入-输出余弦变化来为深度剪枝中的块排序，但该排序依赖于所使用的校准分布。我们研究了在Qwen2.5-1.5B-Instruct、Llama-3.2-1B和Gemma-2-2B三个模型上，英语校准是否能够迁移到印度语言。英语、印地语、马拉地语和泰米尔语使用互不重叠的维基百科文章进行校准和评估；每个选择器都获得相同的150个文本块预算，并且泰米尔语被排除在多语言聚合之外，以测试跨文字系统的迁移能力。在所有三个模型中，三种印度语言的深层块BI比英语高出20.7%至55.5%，且每个预先声明的“深层块减中间块”对比的95%自助法置信区间均大于零。尽管全局秩相关性很高，这种不匹配仍改变了被选中移除的低BI块。在英语、印地语和马拉地语等量预算上对BI取平均，可以缓解Qwen模型在移除六个块时的困惑度损失，并且……

    arXiv:2610.21383v1 Announce Type: new  Abstract: Block Influence (BI) ranks Transformer blocks for depth pruning from their input-output cosine change, but the ranking depends on the calibration distribution. We study whether English calibration transfers to Indian languages in Qwen2.5-1.5B-Instruct, Llama-3.2-1B, and Gemma-2-2B. English, Hindi, Marathi, and Tamil use disjoint Wikipedia articles for calibration and evaluation; every selector receives the same 150-chunk budget, and Tamil is excluded from multilingual aggregation to test cross-script transfer. Across all three models, late-block BI is 20.7-55.5% higher for the three Indian languages than for English, and every predeclared late-minus-middle contrast has a 95% bootstrap interval above zero. This mismatch changes the low-BI blocks selected for removal despite high global rank correlations. Averaging BI over equal English, Hindi, and Marathi budgets mitigates the resulting perplexity damage for Qwen at six removed blocks and
    
[^154]: RDL-Bench：一个面向关系数据库的任务感知问答基准

    RDL-Bench: A Benchmark for Task-Aware Question Answering over Relational Databases

    [https://arxiv.org/abs/2610.21382](https://arxiv.org/abs/2610.21382)

    RDL-Bench是一个针对真实关系数据库的任务感知问答基准，它将用户问题分为由当前数据库状态决定的推理型问题和需要在时间约束下基于历史记录进行预测的预测型问题，填补了现有基准只孤立处理这些目标的空白。

    

    关系数据库（RDBs）仍然是企业和科学领域中存储与管理结构化数据的主要基础设施。面向关系数据库的自然语言（NL）接口旨在让用户无需编写SQL即可提问，但真实的用户需求超越了SQL风格的查询：有些问题需要对可观察的数据库状态进行推理，而另一些问题则需要在时间约束下基于历史关系记录进行预测性推断。现有的基准测试将这些目标孤立地加以处理。为了填补这一空白，我们提出了RDL-Bench，这是一个针对真实关系数据库的任务感知问答（QA）基准。RDL-Bench将多样化的用户问题分为两类：推理导向型问题，其答案由当前数据库状态决定；预测导向型问题，其答案取决于时间约束下的未来结果。每个实例包含一个自然语言问题、任务元数据、模式信息、时间约束……

    arXiv:2610.21382v1 Announce Type: new  Abstract: Relational databases (RDBs) remain the primary infrastructure for storing and managing structured data across enterprise and scientific domains. Natural-language (NL) interfaces to RDBs aim to let users ask questions without writing SQL, but real user needs go beyond SQL-style lookup: some questions require reasoning over the observable database state, while others require predictive inference from historical relational records under temporal constraints. Existing benchmarks address these goals in isolation. To address this gap, we introduce RDL-Bench, a benchmark for task-aware question answering (QA) over real RDBs. RDL-Bench organizes diverse user questions into reasoning-oriented questions, whose answers are determined by the current database state, and prediction-oriented questions, whose answers depend on future outcomes under temporal constraints. Each instance contains an NL question, task metadata, schema information, temporal c
    
[^155]: 压缩是否意味着泛化？基于最小描述长度的视角

    Does Compression Imply Generalization? A Minimum Description Length Perspective

    [https://arxiv.org/abs/2610.21376](https://arxiv.org/abs/2610.21376)

    本文基于最小描述长度原则提出仅由训练轨迹即可计算的预quential后悔，证明其在（近乎）插值情况下上界了训练参数在算法诱导分布下的描述长度，进而转化为PAC高概率泛化界，为“压缩是否意味着泛化”提供了无需保留数据或互信息估计的新答案。

    

    现代学习理论的一个核心问题是：什么样的压缩概念（如果存在的话）能够解释过参数化神经网络为何具有泛化能力。现有的答案在不同的层面上衡量复杂度，范围从权重到表示；然而，估计这些相关量在原则上要么需要保留数据，要么依赖于极具挑战性的（超）高维互信息估计。为了克服这些挑战，我们通过最小描述长度原则来研究这个问题；具体而言，我们提出了预quential后悔，作为一种仅从训练轨迹即可计算的补充信号。我们证明，在（近乎）插值的情况下，期望预quential后悔在算法诱导分布下对训练后参数的描述长度给出了上界，并且该描述长度可以转化为PAC意义下的高概率泛化界。

    arXiv:2610.21376v1 Announce Type: new  Abstract: A central question in modern learning theory is what notion of compression, if any, explains why over-parameterized neural networks generalize. Existing answers measure complexity at different levels, ranging from weights to representations; however, estimating the associated quantities either requires held-out data in principle or relies on the challenging (ultra) high-dimensional mutual information estimation. To overcome these challenges, we approach the question through the minimum description length (MDL) principle, and specifically, we propose the prequential regret as a complementary signal that is computable from the training trajectory alone. We establish that, in the (nearly) interpolating regime, the expected prequential regret upper-bounds the description length of the trained parameter under the algorithm-induced distribution, and that this description length translates into a high-probability generalization bound in the PAC
    
[^156]: Memorilla：面向大语言模型的潜在语义记忆

    Memorilla: Latent Semantic Memory for LLMs

    [https://arxiv.org/abs/2610.21375](https://arxiv.org/abs/2610.21375)

    Memorilla提出了一种潜在语义记忆系统，能将整个文档集合的信息压缩为仅16个依赖查询的记忆token并注入冻结LLM的输入嵌入空间，在八个基准中的五个上超越了Mem0和MemoRAG等基线方法。

    

    记忆使大语言模型（LLM）能够利用其参数之外的信息，例如过去的用户对话、长文档或智能体此前的行为。对于一个冻结（参数不更新）的LLM而言，获取这些积累知识的途径仅限于记忆系统放入其上下文窗口中的内容。然而，如何在保持模型上下文简短的同时，为每次查询收集并组合相关信息，对现有系统来说仍是一个挑战。本文提出了Memorilla，这是一种记忆系统，它将整个文档集合中的信息汇聚为固定数量的、依赖查询的“记忆token”，供冻结的LLM使用。Memorilla将文档（如用户聊天记录、书籍章节）一次性编码为可复用的嵌入向量，然后学习检索并将查询特定的信息放置到LLM的输入嵌入空间中。仅使用16个记忆token，Memorilla在与Mem0和MemoRAG等基线方法的对比中，在八个基准中的五个上取得了最佳性能。

    arXiv:2610.21375v1 Announce Type: cross  Abstract: Memory enables large language models (LLMs) to use information beyond their parameters, such as past user conversations, long documents, or an agent's earlier actions. For a frozen LLM, access to this accumulated knowledge is limited to what a memory system places in its context window. However, gathering and combining relevant information needed for each query while keeping the model's context short remains challenging for existing systems. Here we introduce Memorilla, a memory system that gathers information from an entire document collection into a fixed number of query-dependent 'memory tokens' for a frozen LLM. Memorilla encodes documents (such as user chats, book chapters) once into reusable embeddings, then learns to retrieve and place query-specific information in the LLM's input embedding space. Using only 16 memory tokens, Memorilla achieves the best performance against baselines including Mem0 and MemoRAG on five of eight be
    
[^157]: 相同目标，不同起点：对齐潜在推演以实现一致的偏微分方程仿真

    Same Target, Different Origins: Aligning Latent Rollout for Consistent PDE Emulation

    [https://arxiv.org/abs/2610.21370](https://arxiv.org/abs/2610.21370)

    提出ALR正则化方法，通过对神经PDE求解器的潜在表示施加零阶状态一致性与一阶动力学一致性约束，使不同起点和深度的自回归推演在同一目标时刻收敛到一致的预测，从而抑制误差在多步推演中的传播。

    

    神经偏微分方程（PDE）求解器通常通过自回归推演生成多步预测，即递归地应用学习到的演化算子，并将每一步的预测反馈作为下一步的输入。因此，近似误差会在推演过程中传播，导致在同一目标时刻的预测依赖于其时间起点和推演深度。然而，对于适定的确定性PDE流，同一物理轨迹上不同时刻的状态在精确演化下应到达相同的目标状态。为此，我们提出了对齐潜在推演，这是一种正则化方法，通过神经PDE求解器的内部表示来促进不同推演起点之间的一致性。对潜在差异传播的二阶泰勒分析引出了两个优化目标：零阶潜在状态一致性，即在同一目标时刻对齐表示；以及一阶潜在动力学一致性，即在潜在空间中对齐演化趋势。

    arXiv:2610.21370v1 Announce Type: new  Abstract: Neural PDE solvers commonly generate multi-step predictions through autoregressive rollout, recursively applying a learned evolution operator and feeding each prediction back as input for the next step. Approximation errors can therefore propagate through the rollout, causing predictions at a shared target time to depend on their temporal origins and rollout depths. For a well-posed deterministic PDE flow, however, states at different times along the same physical trajectory should, under exact evolution, reach the same target state. To this end, we propose Align Latent Rollout (ALR), a regularization approach that promotes consistency across rollout origins through the neural PDE solver's internal representations. A second-order Taylor analysis of latent discrepancy propagation motivates two objectives: zeroth-order latent state consistency, which aligns representations at a shared target time, and first-order latent dynamics consistenc
    
[^158]: 测试时大语言模型集成中的两个样本量：缓冲经验风险最小化与极小极大理论

    Two Sample Sizes in Test-Time LLM Ensembling: Buffered ERM and Minimax Theory

    [https://arxiv.org/abs/2610.21363](https://arxiv.org/abs/2610.21363)

    该论文研究了测试时大语言模型加权集成中标注问题数 Q 与每问题生成数 N 这两个样本量的相互作用，证明带校准间隔缓冲的 ERM 可达到 Q^{-1/2}+N^{-β/2} 阶的超额风险，并通过简化的双模型子模型分析验证了该风险率的最优性。

    

    测试时扩展可以通过为一个问题生成多个答案并对其进行聚合来提升大语言模型的性能。加权集成进一步拓展了这一思想，即从多个模型中抽取答案，其中权重决定了每个模型被使用的频率。从训练数据集中学习这些权重涉及两个样本量：Q 个带标注的问题，以及每个模型对每个问题进行的 N 次生成。我们通过干净风险来研究这两个样本量之间的相互作用，干净风险即该集成在无限次生成下的投票规则的总体误差。在指数为 β 的间隔条件下，我们证明带有校准间隔缓冲的经验风险最小化（ERM）方法，在模型数量和候选答案数量固定的情况下，可以达到量级为 Q^{-1/2} + N^{-β/2} 的超额风险（仅相差对数因子）。为了探究这种对 Q 和 N 的依赖关系是否最优，我们分析了一个简化的子模型，该模型包含两个模型且每个问题只有两个可能的答案。我们……

    arXiv:2610.21363v1 Announce Type: cross  Abstract: Test-time scaling can improve the performance of large language models by generating multiple answers to a question and aggregating them. Weighted ensembles extend this idea by drawing answers from several models, with the weights determining how often each model is used. Learning these weights from a training dataset involves two sample sizes: $Q$ labeled questions and $N$ generations per model and question. We study their interplay through the clean risk, the population error of the ensemble's infinite-generation voting rule. Under a margin condition with exponent $\beta$, we show that empirical risk minimization (ERM) with a calibrated margin buffer achieves excess risk of order $Q^{-1/2}+N^{-\beta/2}$, up to logarithmic factors, for fixed numbers of models and candidate answers. To investigate the optimality of this dependence on $Q$ and $N$, we analyze a simplified submodel with two models and two possible answers per question. We
    
[^159]: L2GSSL：面向体积医疗数据的标签高效局部到全局自监督学习

    L2GSSL: Label-Efficient Local-to-Global Self-Supervised Learning for Volumetric Healthcare Data

    [https://arxiv.org/abs/2610.21356](https://arxiv.org/abs/2610.21356)

    提出L2GSSL局部到全局自监督学习框架，通过将预训练分解为切片内和切片间表示空间并利用停止梯度隔离目标，在仅使用10%标注的肺结节恶性分类任务上达到0.9229 AUROC，大幅超越体积级对比方法，接近全监督基线性能。

    

    医疗机构通常会积累大量高维计算机断层扫描（CT）体积数据，但其中只有一小部分能够由专家进行标注。我们提出了L2GSSL，这是一个局部到全局的自监督学习框架，通过将预训练分解为切片内和切片间两个表示空间来利用未标注的体积数据。切片编码器学习视图不变的局部特征，全局编码器聚合所得的标记以建模体积上下文。停止梯度机制将局部空间与全局目标隔离开来，而逐切片执行和三阶段训练调度则支持在受限GPU内存条件下的训练。在LIDC-IDRI肺结节恶性分类任务上，L2GSSL仅使用10%的标签即达到0.9229的AUROC，使用25%的标签达到0.9377，分别超过体积级VICReg方法0.0229和0.0294。在使用全部标签的情况下，最强的监督基线也仅比L2GSSL高出0.0044的AUROC，结果表现一致。

    arXiv:2610.21356v1 Announce Type: cross  Abstract: Healthcare institutions routinely accumulate high-dimensional computed tomography (CT) volumes, but only a fraction can be annotated by experts. We present L2GSSL, a local-to-global self-supervised learning framework that leverages unlabeled volumes by decomposing pretraining into intra-slice and inter-slice representation spaces. A slice encoder learns view-invariant local features, and a global encoder aggregates the resulting tokens to model volumetric context. The stop-gradient isolates the local space from the global objective, while slice-wise execution and a three-phase schedule support training under constrained GPU memory. On LIDC-IDRI lung nodule malignancy classification, L2GSSL reaches $0.9229$ AUROC with only $10\%$ of the labels and $0.9377$ with $25\%$, exceeding the volume-level VICReg by $0.0229$ and $0.0294$, respectively. With all labels, the strongest supervised baseline exceeds L2GSSL by only $0.0044$ AUROC, consis
    
[^160]: 诚实是最好的策略……果真如此吗？概率性服务选项的需求管理

    Honesty is the Best Policy ... or is it? Demand Management for Probabilistic Service Options

    [https://arxiv.org/abs/2610.21351](https://arxiv.org/abs/2610.21351)

    该论文将概率性服务选项下的需求管理建模为序列决策过程，通过递归定义真实概率并引入透明度约束，首次系统研究了提供商操纵展示概率这一新型需求管理杠杆及其与诚实披露之间的权衡。

    

    为了让客户做出充分知情的选择，并在各服务选项之间重新分配需求，服务提供商可以在预订时展示关于预测服务水平的概率，从而实际上提供“概率性服务选项”。在许多服务场景中，潜在的真实概率内生地依赖于系统状态，而非由提供商控制的静态产品属性。虽然客户在决策时依赖这些信息，却难以验证其准确性。因此，提供商可能会试图操纵所展示的概率，以引导客户做出有利于自身的选择，这为需求管理提供了一种新颖的杠杆。在本工作中，我们将由此产生的问题建模为一个序列决策过程，递归地定义真实概率，并引入限制其失真程度的透明度约束。由于真实概率的精确计算是难以求解的……

    arXiv:2610.21351v1 Announce Type: new  Abstract: To enable well-informed customer choices and redistribute demand across service options, providers can display probabilities regarding predicted service levels at the time of booking, effectively offering probabilistic service options. In many service settings, the underlying truthful probabilities endogenously depend on the system state, rather than being static product attributes controlled by the provider. While customers rely on this information when making decisions, they cannot readily verify its accuracy. Consequently, providers may be tempted to manipulate the displayed probabilities to guide customers' choices in their favor, creating a novel lever for demand management. In this work, we model the resulting problem as a sequential decision process, define truthful probabilities recursively, and introduce transparency constraints that limit their distortion. Because the exact computation of truthful probabilities is intractable f
    
[^161]: 面向正当损失的在线分位数全能预测

    Online Quantile Omniprediction for Proper Losses

    [https://arxiv.org/abs/2610.21342](https://arxiv.org/abs/2610.21342)

    本文提出在线样本高效的分位数预测算法，使预测在所有满足温和正则条件的正当损失下同时保持准确，将全能预测从二分类任务拓展到多分类和连续型结果。

    

    构建一个在多种评估指标下同时保持准确的预测器，这一任务近来以“全能预测”之名得到推广。该领域的先前工作主要集中于二分类预测任务，而针对多分类和连续型结果的现有研究通常存在误差率收敛缓慢的问题。本文考虑以一组分位数形式表达的预测，其概率水平取自一个离散但任意的集合。我们开发了样本高效的算法，用于构建在所有满足温和正则性条件的正当损失下同时保持准确的预测，其中正当损失是指通过预测真实分位数即可使其最小化的损失函数。我们的算法是在线的，能够在动态环境中运行，同时将多种损失纳入考量，其预测有潜力为不同终端用户提供丰富的信息以评估风险和……

    arXiv:2610.21342v1 Announce Type: new  Abstract: The task of constructing a predictor that is simultaneously accurate under multiple evaluation metrics has been popularized recently under the name omniprediction. Prior work in this area has largely focused on binary prediction tasks, and existing results for multiclass and continuous outcomes typically suffer from slow error rates. In this paper, we consider forecasts expressed as a set of quantiles over a discrete but otherwise arbitrary set of probability levels. We develop sample-efficient algorithms for constructing forecasts that are accurate simultaneously across all proper losses that satisfy mild regularity conditions, where proper losses are those minimized by predicting the true quantiles. Our algorithms are online and can operate in a dynamic environment, while simultaneously taking multiple losses into account, and their forecasts offer the potential to provide diverse end-users with rich information for evaluating risk and
    
[^162]: 基于狄利克雷过程的深度神经网络维度无关PAC-Bayes泛化

    Dimension-Free PAC-Bayes Generalization for Deep Neural Networks via Dirichlet Processes

    [https://arxiv.org/abs/2610.21340](https://arxiv.org/abs/2610.21340)

    该论文通过狄利克雷过程将PAC-Bayes分析的先验从高维参数空间转移到数据分布空间，并借助数据处理不等式证明参数空间相对熵被可闭式计算的数据空间DP散度严格界定，从而获得了与网络维度无关的深度神经网络泛化界，解决了过参数化网络泛化界空洞的问题。

    

    arXiv:2610.21340v1 公告类型：cross 摘要：经典的PAC-Bayesian泛化理论将先验分布置于欧氏参数空间 $\Theta \subseteq \mathbb{R}^D$ 上，导致相对熵随网络维度 $D$ 增长。对于现代过参数化神经网络（其中 $D \gg n$），这种基于参数计数的依赖性产生了空洞的泛化界（超过1,000%）。在本工作中，我们通过狄利克雷过程 $\mathcal{DP}(\alpha, F_0)$ 将先验设定从内部参数转移到数据分布的非参数空间 $\mathcal{M}_1(\mathcal{Z})$，并通过学习算法下的前推测度诱导出参数分布，从而突破了这一障碍。根据数据处理不等式，参数空间的相对熵被数据空间的DP散度严格上界约束，我们利用Digamma函数以精确闭式形式对该散度进行计算。所得到的PAC-Bayesian证书对网络参数严格不变（摘要在此处截断）。

    arXiv:2610.21340v1 Announce Type: cross  Abstract: Classical PAC-Bayesian generalization theory places prior distributions over Euclidean parameter space $\Theta \subseteq \mathbb{R}^D$, causing relative entropy to scale with network dimension $D$. For modern overparameterized neural networks where $D \gg n$, this parameter-counting dependency yields vacuous generalization bounds ($> 1{,}000\%$). In this work, we resolve this barrier by shifting prior specification from internal parameters to the non-parametric space of data distributions $\mathcal{M}_1(\mathcal{Z})$ via a \textbf{Dirichlet Process (DP)}, $\mathcal{DP}(\alpha, F_0)$, inducing parameter distributions through push-forward measures under the learning algorithm. By the Data Processing Inequality, parameter-space relative entropy is strictly bounded by the data-space DP divergence, which we evaluate in exact closed form using Digamma functions. The resulting PAC-Bayesian certificate is strictly invariant to network paramete
    
[^163]: 在策略蒸馏需要完成每一次 Rollout 吗？

    Does On-Policy Distillation Need to Complete Every Rollout?

    [https://arxiv.org/abs/2610.21337](https://arxiv.org/abs/2610.21337)

    该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。

    

    在策略蒸馏（On-policy Distillation, OPD）通过对学生模型生成的 rollout 序列提供反馈，有效地将教师模型的能力迁移给学生模型。然而，生成完整的学生 rollout 序列并为每个 token 计算教师反馈，使得 OPD 的计算开销十分高昂。我们提出了随机截断在策略蒸馏（Stochastic Truncated On-Policy Distillation, STOPD），该方法通过随机截断学生 rollout 序列，并利用分块逆概率加权（blockwise IPW）校正由此产生的采样偏差，从而加速 OPD。在每一步训练中，我们在生成前为每个 rollout 采样一个停止位置，让学生模型生成至该位置或自然终止。我们对每个生成的 token 获取教师反馈，并使用分块 IPW 补偿保留后续 token 的较低概率，同时对权重加以控制。我们证明了这种加权方式在期望意义上保持了原始 OPD 的梯度总和不变，并推导出了一步更新的误差界……

    arXiv:2610.21337v1 Announce Type: cross  Abstract: On-policy distillation (OPD) effectively transfers capabilities from a teacher to a student through feedback on student-generated rollouts. However, generating complete student rollouts and computing teacher feedback for every token make OPD computationally expensive. We introduce Stochastic Truncated On-Policy Distillation (STOPD), which accelerates OPD by randomly truncating student rollouts and correcting the resulting sampling bias with blockwise inverse-probability weighting (IPW). At each training step, we sample a stopping position for each rollout before generation and let the student generate until that position or natural termination. We obtain teacher feedback on every generated token and use blockwise IPW to compensate for the lower probability of retaining later tokens while controlling the weights. We prove that this weighting preserves Vanilla OPD's gradient sum in expectation and derive one-step update-error bounds for 
    
[^164]: 用于稀疏与持续强化学习微调的注意力头排序与划分方法

    Ranking and Partitioning of Attention Heads for Sparse and Continual Reinforcement Learning Fine-tuning

    [https://arxiv.org/abs/2610.21336](https://arxiv.org/abs/2610.21336)

    该论文提出按梯度范数对注意力头排序并在任务间划分的方法，实现了语言模型的高效稀疏强化学习微调，同时减少跨任务干扰、提升持续学习中的能力保持。

    

    我们通过注意力头的排序与划分来研究语言模型的稀疏与持续强化学习微调。对于稀疏学习，我们按梯度范数对注意力头进行排序，以识别出可以被有效训练的小规模注意力头子集。对于持续学习，我们在不同任务之间划分注意力头，以减少跨任务干扰并保留先前学到的能力。在理论层面，我们将梯度范数与训练效果联系起来，并进一步刻画了为什么稀疏、分离的注意力头更新能够保持稳定。在实证方面，在数学、指令遵循和编程任务上，微调排名较高的注意力头始终优于微调排名较低的注意力头，且通常可与更密集的基线方法相媲美；同时，注意力头划分在保留适应新任务能力的前提下，提升了对旧任务的记忆保持能力。

    arXiv:2610.21336v1 Announce Type: cross  Abstract: We study sparse and continual reinforcement learning fine-tuning of language models through attention-head ranking and partitioning. For sparse learning, we rank attention heads by gradient norm to identify a small subset of heads that can be trained effectively. For continual learning, we partition heads across tasks to reduce cross-task interference and preserve previously learned capabilities. Theoretically, we relate gradient norm to training effectiveness, and further characterize why sparse, separated head updates can remain stable. Empirically, across math, instruction-following, and coding tasks, tuning higher-ranked heads consistently outperforms tuning lower-ranked ones while often matching denser baselines, and head partitioning improves retention while preserving adaptation to new tasks.
    
[^165]: 低照度处理管线何时应该进行生成？面向选择性生成增强的可审计风险控制

    When Should a Low-Light Pipeline Generate? Auditable Risk Control for Selective Generative Enhancement

    [https://arxiv.org/abs/2610.21335](https://arxiv.org/abs/2610.21335)

    VetoNet提出了一种以认证否决为核心的可审计区域路由框架，通过无分布假设的危害预算证书，在低照度增强中自适应地决定生成专家何时可以安全地覆盖冻结的复原专家，从而在控制感知风险的前提下实现结构与纹理的最优平衡。

    

    生成式低照度增强器能够恢复看似合理的纹理，但可能产生结构上的幻觉；复原网络虽能保持结构，却可能导致部分区域增强不足，且两者的优势在空间上并非均匀分布。因此，我们将选择性生成增强形式化为一个决策问题：判断在哪些区域覆盖（override）冻结的基础专家模型是有益的，并在证据不足时拒绝切换。我们提出了VetoNet，一个以“认证否决”为核心的可审计区域路由框架。该框架首先测量冻结的复原专家与生成专家对之间的互补能力及路由方向，然后仅基于部署时可获得的信号预测区域覆盖的效用，并且只有当无分布假设的危害预算证书表明感知质量（LPIPS）风险预算允许时，才允许执行覆盖。在多个专家对和多个领域上的实验表明，VetoNet能够根据专家对的能力强弱和可靠性自适应地调整其路由行为。

    arXiv:2610.21335v1 Announce Type: cross  Abstract: Generative low-light enhancers recover plausible texture but may hallucinate structure; restoration networks preserve structure but can leave regions under-enhanced, and neither advantage is spatially uniform. Therefore, we formulate selective generative enhancement as a decision problem: determine where overriding a frozen base expert is beneficial and refuse to switch when the evidence is insufficient. We propose VetoNet, an auditable regional routing framework centered on a certified veto. It first measures the complementary strengths and routing direction of a frozen restoration generative pair, then predicts the utility of regional overrides from deployment-available signals alone and admits an override only when a distribution-free harm-budget certificate says the perceptual (LPIPS) risk budget allows it. Experiments across multiple expert pairs and domains show that VetoNet adapts its routing behavior to the strength and reliabi
    
[^166]: 基于可逆双利普希茨适配器的流匹配语义潜在空间对齐

    Semantic Latent Alignment for Flow Matching with Invertible Bi-Lipschitz Adapters

    [https://arxiv.org/abs/2610.21332](https://arxiv.org/abs/2610.21332)

    提出一种可逆双利普希茨适配器，仅改变预训练自编码器中固定的编码器与解码器之间的坐标，将其潜在空间离线对齐到语义教师模型，从而在保持压缩与重建性能的同时改善流匹配生成的语义组织。

    

    预训练潜在空间在压缩、重建以及下游生成器所工作坐标的可学习性之间存在权衡，这使得将这些空间适配到不同领域或任务变得复杂：继续训练成本高昂，而且在没有新的对齐目标的情况下，往往无法改善下游生成。我们反其道而行之，保持编码器和解码器固定，仅改变它们之间的坐标，使用一个离线对齐到语义教师模型并在类条件流匹配训练期间保持冻结的可逆双利普希茨适配器。可逆性保证了与预训练自编码器的兼容性，而双利普希茨约束则限制了对齐过程对其潜在几何结构的扭曲程度。这提供了一种简单的方法，能够在保留原始潜在空间压缩和重建特性的同时改善其语义组织。在两个医学影像数据集上，受约束的潜在适配改善了特征……

    arXiv:2610.21332v1 Announce Type: cross  Abstract: Pretrained latent spaces trade compression, reconstruction, and the learnability of the coordinates a downstream generator has to work in, which complicates adapting these spaces to different domains or tasks: continued training is expensive, and without a new alignment objective it often fails to improve downstream generation. We instead leave the encoder and decoder fixed and change only the coordinates between them, using an invertible bi-Lipschitz adapter aligned offline to a semantic teacher and frozen during class-conditional flow matching. Invertibility preserves compatibility with the pretrained autoencoder, while the bi-Lipschitz constraint limits how far alignment can distort its latent geometry. This provides a simple way to improve semantic organization while retaining the compression and reconstruction properties of the original latent space. On two medical imaging datasets, constrained latent adaptation improves feature m
    
[^167]: 谱信息并不足够：差分隐私学习中无数据预条件化的随机矩阵分析

    Spectrum Is Not Enough: A Random-Matrix Account of Data-Free Preconditioning for Differentially Private Learning

    [https://arxiv.org/abs/2610.21312](https://arxiv.org/abs/2610.21312)

    该论文用随机矩阵理论刻画了差分隐私学习中无数据 K-FAC 预条件化的可重构性，证明了 Kronecker 分解 $A \otimes G$ 中的误差因子 $G$ 和激活因子 $A$ 的架构决定部分可由合成探针在无数据情况下重构，只有 $A$ 中由数据引起的特征值尖峰不可重构，并精确定位了其在 Baik-Ben Arous-Péché 阈值 $(d/n)^{1/4}$ 处的失效边界。

    

    无数据预条件化通过合成探针而非私有数据来构建 KFAC 的 Kronecker 因子。DP-KFC 通过匹配私有因子的特征值衰减来验证探针的有效性；我们将这一准则转化为一个关于探针究竟能够重构什么内容的随机矩阵理论框架。整个分析由一个单一的二分法组织：在分解 $A \otimes G$ 中，误差信号因子 $G$ 以及激活因子 $A$ 中由网络架构决定的主体部分无需数据即可重构，而唯一不可重构的是 $A$ 中由数据引起的特征值尖峰。我们证明了 $G$ 这一半对数据不敏感，其误差不超过一个显式的误差能量界，且在四种模态和一个自然图像基准的完整训练轨迹上实现了零违反。我们将失效边界定位在 Baik-Ben Arous-Péché 阈值处：仅含主体部分的探针会因数据引起的尖峰而遭受精确为 $s^2/(1-s^2)$ 的损伤，而一旦尖峰超过 $(d/n)^{1/4}$，该损伤便变得不可见，这一结论在注入……（摘要截断处）上得到验证。

    arXiv:2610.21312v1 Announce Type: new  Abstract: Data-free preconditioning builds the Kronecker factors of KFAC from synthetic probes instead of private data. DP-KFC validates a probe by matching the eigenvalue decay of the private factors; we turn this criterion into a random-matrix account of what a probe can reconstruct. A single dichotomy organizes it: in the factorization $ A \otimes G $, the error-signal factor $ G $ and the architecture-determined bulk of the activation factor $ A $ are reconstructible without data, while only the data-induced spikes of $ A $ are not. We prove the $ G $ half is data-insensitive up to an explicit error-energy bound, with zero violations over a full training trajectory across four modalities and a natural-image benchmark. We locate the failure boundary at a Baik-Ben Arous-P\'ech\'e threshold: a bulk-only probe suffers an exact damage $s^2/(1-s^2)$ from a data-induced spike, which becomes invisible once it exceeds $(d/n)^{1/4}$, confirmed on an inj
    
[^168]: PocketTSLM：部署设备端软提示时间序列语言模型以实现隐私保护的智能手机推理

    PocketTSLM: Deploying an On-Device Soft-Prompt Time-Series Language Model for Privacy-Preserving Smartphone Reasoning

    [https://arxiv.org/abs/2610.21308](https://arxiv.org/abs/2610.21308)

    提出PocketTSLM，一个完全在消费级智能手机上运行的设备端软提示时间序列语言模型移动原生部署方案，在保留时间序列结构的同时实现隐私保护的生理数据推理。

    

    消费级手机和可穿戴设备日益收集生理时间序列数据，然而大多数移动系统将这些数据流视为摘要或文本，而非时间证据。这种表示方式与个人健康系统的需求匹配不佳，后者需要在保留时间结构的同时检索、路由或汇总传感器证据。云端托管的时间序列语言模型虽然避开了智能手机的限制，但它们会反复将敏感的生理数据传出设备，并增加延迟和每次查询的计算成本。而将时间序列数据聚合为统计特征或将原始样本序列化为文本的移动模型替代方案，要么压缩了时间结构，要么耗尽了小型设备端模型的上下文和内存预算。我们提出了PocketTSLM，这是OpenTSLM SoftPrompt（OpenTSLM时间序列语言模型的软提示变体）的移动原生部署方案。PocketTSLM通过注入编码的时间……（摘要截断），完全在消费级智能手机上运行。

    arXiv:2610.21308v1 Announce Type: cross  Abstract: Consumer phones and wearables increasingly collect physiological time-series data, yet most mobile systems treat these streams as summaries or text rather than as temporal evidence. This representation is poorly matched to personal health systems that need to retrieve, route, or summarize sensor evidence while preserving temporal structure. Cloud-hosted time-series language models avoid smartphone limits, but they repeatedly move sensitive physiology off the device and add latency and per-query compute cost. Mobile model alternatives that aggregate time-series data into statistical features or serialize raw samples as text either compress temporal structure or exhaust the context and memory budgets of small on-device models. We present PocketTSLM, a mobile-native deployment of OpenTSLM SoftPrompt, the soft-prompt variant of the OpenTSLM time-series language model. PocketTSLM runs fully on a consumer smartphone by injecting encoded time
    
[^169]: 面向联邦Stackelberg优化的个性化投影上升方法

    Personalized Projected Ascent for Federated Stackelberg Optimization

    [https://arxiv.org/abs/2610.21295](https://arxiv.org/abs/2610.21295)

    该论文提出在联邦Stackelberg优化中让最大化变量保持客户端个性化并采用投影上升，证明了非凸-凹情形下达到标准次线性收敛速率，并针对非凸-PL情形中随机更新导致的与步长无关的噪声下限，提出方差缩减方法VR-SGDA加以缓解。

    

    联邦极小极大问题（如分布式鲁棒优化和对抗训练）包含一个由各客户端本地数据构建的内层最大化变量。这些变量在不同客户端之间的含义和维度各不相同，因此像标准联邦极小极大方法那样在客户端间对它们取平均是不适定的。为此，我们研究了一类联邦极小极大问题：极小化变量 x 在客户端间共享并被聚合，而每个客户端保留各自的最大化变量 y_i（个性化）。由于这些应用将 y_i 约束在紧致凸集上，我们直接对投影上升方法进行分析。我们证明，在非凸-凹情形下，这种投影式个性化方案能够达到标准的次线性收敛速率；然而，在非凸-PL（Polyak-Łojasiewicz）情形下，由于 y_i 的随机更新，该方法会遇到一个与步长无关的噪声下限。为缓解这一问题，我们提出了VR-SGDA，其中引入了……（原文在此截断）

    arXiv:2610.21295v1 Announce Type: new  Abstract: Federated minimax problems, such as distributionally robust optimization and adversarial training, include an inner maximization variable built from each client's local data. These variables differ in meaning and dimension from one client to the next, so averaging them across clients, as standard federated minimax methods do, is ill-posed. We therefore study a federated minimax problem where the minimization variable $\mathbf{x}$ is shared and aggregated, while each client retains the maximization variable $\mathbf{y}_i$ (personalized). Because these applications constrain $\mathbf{y}_i$ to a compact convex set, we analyze projected ascent directly. We show that, in the nonconvex-concave regime, this projected personalized scheme attains the standard sublinear rate. In the nonconvex-PL regime, however, it hits a stepsize-invariant noise floor due to stochastic $\mathbf{y}_i$ updates. To mitigate this, we propose VR-SGDA, incorporating a 
    
[^170]: 能力增强可以被局部化吗？

    Can Capability Enhancement be Localized?

    [https://arxiv.org/abs/2610.21294](https://arxiv.org/abs/2610.21294)

    该论文提出“能力增强局部化”（CEL）概念，通过两阶段的筛选与联合消融方法，实证发现移除少数注意力头组成的稀疏组合即可在无需微调的情况下提升模型在特定任务上的性能。

    

    基于Transformer的语言模型在许多任务和领域中都展现出强大的能力。然而，先前的研究表明，对于给定的任务，并非所有注意力头都同等重要。具体而言，这些研究表明，许多“冗余”的注意力头可以在几乎不影响模型的情况下被安全移除，而一小组“重要”的注意力头在被消融时则会严重损害模型在给定任务上的性能。在本工作中，我们研究了与此互补的一个方面，称为“能力增强局部化”（Capability Enhancement Localization, CEL）：给定一个模型和一个任务，是否存在一小组注意力头，移除它们能够提升该任务的性能？我们首先在不同模型家族上对这一问题进行了实证研究。通过一个简单的两阶段流程——先筛选候选注意力头，再评估它们的联合消融效果——我们识别出了一些稀疏的注意力头组合，移除这些组合无需进一步微调即可提升特定任务的性能。

    arXiv:2610.21294v1 Announce Type: new  Abstract: Transformer-based language models have shown strong capabilities across many tasks and domains. However, prior work shows that, for a given task, not all attention heads matter equally. Specifically, they suggest that many \textit{redundant} heads can be safely removed with little-to-no effect, while a small set of \textit{important} heads, when ablated, severely damages the model's performance on the given task. In this work, we investigate a complementary aspect, called \textit{Capability Enhancement Localization} (CEL): \textit{given a model and a task, does there exist a small set of heads whose removal improves the task performance?} We first investigate this question \textit{empirically} across models from different families. Using a simple two-stage routine that screens candidate heads and evaluates their joint ablations, we identify sparse head combinations whose removal improves task-specific performance without further fine-tun
    
[^171]: TimeCap：一个用于可穿戴健康时间序列数据描述生成的可扩展开源流水线

    TimeCap: A Scalable Open-Source Pipeline for Captioning Wearable Health Time-Series Data

    [https://arxiv.org/abs/2610.21274](https://arxiv.org/abs/2610.21274)

    该论文提出了TimeCap，一个开源、确定性、可扩展的时间序列描述生成流水线，通过三级可复用提取器和跨通道综合层，为可穿戴健康时间序列数据生成基于信号、可复现且可扩展的文本标注，填补了时间序列语言模型训练语料的空白。

    

    跨工业、环境，尤其是健康领域持续产生的时间序列数据生成多模态信号，其规模和异质性使其处理与解释尤为困难。时间序列语言模型（TSLMs）有望对这些信号实现可扩展的解释，但其发展受到一个根本性瓶颈的制约：缺乏以信号为基础、可复现、且可跨模态和模式扩展的文本标注时间序列数据对。现有的配对语料库要么是封闭的，要么在规模和模态覆盖上有限。利用大语言模型进行标注虽可扩大规模，但其输出无法被精确重现，这导致他人无法重新推导、扩展或审计所得语料库。为解决这一问题，我们发布了TimeCap——一个开放、确定性的描述生成流水线，它包含三个可复用的提取器层级和一个跨通道综合层，可直接作用于信号之上……

    arXiv:2610.21274v1 Announce Type: cross  Abstract: Time-series data generated continuously across industrial, environmental, and especially health domains produce multimodal signals whose scale and heterogeneity make their processing and interpretation particularly challenging. Time-Series Language Models (TSLMs) promise scalable interpretation of these signals, yet remain bottlenecked by a fundamental lack of text-annotated time series pairs that are signal-grounded, reproducible, and extensible across modalities and schemas. Existing paired corpora are either closed or limited in scale and modality coverage. Annotation with large language models could scale them up, but its outputs cannot be regenerated exactly, which prevents others from re-deriving, extending, or auditing the resulting corpus. To address this, we release TimeCap, an open deterministic captioning pipeline with three reusable extractor tiers and a cross-channel synthesis layer that operates directly on the signal wit
    
[^172]: 基于对数损失的量子态在线学习的遗憾下界，及其在量子态层析成像与矩阵均值置信序列中的应用

    Regret Lower Bound for Online Learning of Quantum States with Logarithmic Loss, with Applications to Quantum State Tomography and Confidence Sequences for Matrix Means

    [https://arxiv.org/abs/2610.21273](https://arxiv.org/abs/2610.21273)

    本文建立了基于对数损失的量子态在线学习的最优遗憾下界，并由此导出样本量最优（至多相差多对数因子）的量子态层析成像算法，以及矩阵均值置信序列的构造算法。

    

    我们研究基于对数损失的量子态在线学习问题，该问题是在线投资组合选择的量子推广，并应用于量子态层析成像，即从测量结果中估计未知量子态的任务。我们首先建立了一个遗憾下界，表明指数权重算法和Tseng等人（2025）提出的算法的遗憾率在维度上具有最优的多项式依赖性。我们的分析还带来了两个进一步的结果。首先，我们的下界证明导出了一种量子态层析成像算法，其样本复杂度在样本量方面达到最优（仅相差多对数因子）。其次，我们开发了一种用于构造矩阵值随机变量均值的置信序列的算法。数值实验表明，后一种算法与现有方法相比具有竞争力。

    arXiv:2610.21273v1 Announce Type: new  Abstract: We study the problem of online learning of quantum states with logarithmic loss, which is a quantum generalization of online portfolio selection and has applications to quantum state tomography, the task of estimating an unknown quantum state from measurement outcomes. We first establish a regret lower bound, showing that the regret rates of Exponential Weights and the algorithm of Tseng et al. (2025) have optimal polynomial dependence on the dimension. Our analysis leads to two further results. First, our lower-bound proof leads to a quantum state tomography algorithm that is sample optimal up to polylogarithmic factors in the sample size. Second, we develop an algorithm for constructing confidence sequences for the mean of a matrix-valued random variable. A numerical experiment shows that the latter algorithm is competitive with existing methods.
    
[^173]: 基于矩匹配重新思考标签比例学习

    Rethinking Learning from Label Proportions via Moment Matching

    [https://arxiv.org/abs/2610.21262](https://arxiv.org/abs/2610.21262)

    该论文提出了一种简单有效的基于矩匹配的标签比例学习方法（LLP-MM），在袋独立同分布采样这一更现实的假设下，利用标签计数服从泊松多项式分布并通过多阶矩匹配，显著改善了非随机分袋场景下的分类性能。

    

    从标签比例中学习（Learning from Label Proportions, LLP）是一种弱监督学习设置，其中训练数据被分组为若干袋，仅能观察到袋级别的标签比例，目标是学习一个能够预测单个实例标签的分类器。近年来涌现了大量相关方法；然而，大多数方法依赖于“实例-标签对是独立同分布采样并随机组装成袋”这一假设，而该假设在现实场景中经常被违反。我们的实验结果表明，这些方法在非随机设置下表现欠佳。在这项工作中，我们采用了一个更符合现实的假设：袋是独立同分布采样的，且实例标签在给定实例的条件下相互独立。在这一设置下，我们证明了标签计数服从泊松多项式分布。受这一观察启发，我们提出了一种基于矩匹配的标签比例学习方法（LLP-MM），这是一种简单而有效的方法，它利用多阶因子…

    arXiv:2610.21262v1 Announce Type: new  Abstract: Learning from label proportions (LLP) is a weakly supervised setting where training data are grouped into bags with only bag-level label proportions observed, and the goal is to learn a classifier that predicts labels for individual instances. Recently, a large body of methods has emerged; however, most of them rely on the assumption that instance-label pairs are sampled i.i.d. and randomly assembled into bags, an assumption often violated in real-world scenarios. Our experimental results indicate that these methods perform suboptimally under non-random settings. In this work, we adopt a more realistic assumption: bags are sampled i.i.d., and instance labels are conditionally independent given the instances. Under this setting, we show that the label counts follow a Poisson multinomial distribution. Motivated by this observation, we propose LLP via moment matching (LLP-MM), a simple yet effective approach which leverages multi-order fact
    
[^174]: 稀疏自编码器表示的行为组合性

    Behavioral Compositionality of Sparse Autoencoder Representations

    [https://arxiv.org/abs/2610.21261](https://arxiv.org/abs/2610.21261)

    论文提出SAE可加性残差（SAR）来度量稀疏自编码器特征在行为层面的组合性，发现学习到的SAE坐标比随机旋转具有更强的一阶可加性，但仍存在不可忽略的高阶交互项。

    

    稀疏自编码器（SAE）使用稀疏的、线性解码的特征来表示模型激活，但这些特征的下游效应不一定是按加法方式组合的。我们引入了SAE可加性残差（SAR），用于衡量在选定交互阶数以上未能被解释的行为变异。在独立特征子集采样的条件下，总体SAR恰好等于该阶数以上交互的归一化能量，并为最优低阶近似的联盟排序误差提供了上界。为了区分与SAE表示本身相关的组合性与由下游计算所诱导的组合性，我们将学习到的SAE坐标与同一所选子空间的保和随机旋转进行比较，发现在所有18个模型和行为设置中，学习到的坐标都具有更低的一阶残差。然而，这些坐标并非完全可加：在每一设置中，成对交互项均能改善留出集上的联盟排序。

    arXiv:2610.21261v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) represent model activations with sparse, linearly decoded features, but the downstream effects of those features need not compose additively. We introduce SAE Additivity Residuals (SAR), which measure the behavioral variation left unexplained above a chosen interaction order. Under independent feature subset sampling, population SAR is exactly the normalized energy of interactions above that order and bounds coalition ordering error for the optimal low order approximation. To distinguish compositionality associated with the SAE representation from that induced by downstream computation, we compare learned SAE coordinates with sum preserving random rotations of the same selected subspace and find lower first order residuals for the learned coordinates across all 18 model and behavior settings. The coordinates are nevertheless not fully additive: pairwise terms improve held out coalition ordering in every setti
    
[^175]: 面向高效且有效的无乘法器LLM推理的分数二次幂量化

    Fractional Power-of-Two Quantization for Efficient and Effective Multiplier-Free LLM Inference

    [https://arxiv.org/abs/2610.21257](https://arxiv.org/abs/2610.21257)

    该论文提出分数二次幂量化（FPoT），采用以√2为基数的半八度分数量化网格，在保持完全整数移位-加法数据通路实现无乘法器推理的同时，解决了传统二次幂量化在低位宽下的精度下降问题，实现了实用的4位LLM权重量化。

    

    大语言模型（LLM）需要高效的推理，其中线性投影中的乘累加（MAC）运算主导了计算和能耗成本。训练后量化（PTQ）通过将权重和激活映射到低位整数来降低这一成本。二次幂权重量化通过移位-加法逻辑有望实现高效的无乘法器LLM推理，但由于其粗糙的以2为基数的量化网格，在低位宽下会出现显著的精度下降。我们引入了分数二次幂量化（FPoT），这是一种以√2为基数（半八度）的分数二次幂网格，在保持完全整数移位-加法数据通路的同时，实现了实用的4位权重量化。该方法结合了：(i) 以√2为基数的权重网格与均匀整数激活相结合，(ii) 在校准阶段进行并被吸收到静态逐通道缩放因子中的网格对齐细化，以及 (iii) 轻量级的单加法器√2近似，其误差在权重量化过程中被吸收。

    arXiv:2610.21257v1 Announce Type: cross  Abstract: Large Language Models (LLMs) demand efficient inference, where the multiply-accumulate (MAC) operations in linear projections dominate compute and energy cost. Post-training quantization (PTQ) reduces this cost by mapping weights and activations to low-bit integers. Power-of-Two (PoT) weight quantization promises efficient multiplier-free LLM inference via shift-add logic, but suffers significant accuracy degradation at low bitwidths due to its coarse base-2 quantization grid. We introduce Fractional PoT (FPoT), a base-$\sqrt{2}$ (half-octave) fractional PoT grid that achieves practical 4-bit weight quantization while preserving a fully integer shift-add datapath. The method combines (i) the base-$\sqrt{2}$ weight grid with uniform integer activations, (ii) a calibration-time grid-alignment refinement absorbed into the static per-channel scale, and (iii) a lightweight 1-adder $\sqrt{2}$ approximation whose error is absorbed during weig
    
[^176]: 使用无标签子空间隔离语言模型的判断能力

    Isolating a Language Model's Judgment with a Label-Free Subspace

    [https://arxiv.org/abs/2610.21256](https://arxiv.org/abs/2610.21256)

    该研究提出一种无需标签的方法，通过识别并移除隐藏状态中的低维“表达子空间”，成功隔离出语言模型内部稳定的判断信号，使其在不同指令变化下仍保持可靠。

    

    语言模型经常被要求做出判断：例如某个提出的答案是否正确、一个陈述是否在逻辑上蕴含另一个陈述、或者一个回复是否安全。然而，改变规定判断应如何表达的指令，可能会使最终答案变得不那么可靠，即使输入完全相同。我们探究这究竟反映了内部判断的改变，还是仅仅表达了内部判断的失败。通过比较同一输入在不同指令下的隐藏状态向量，我们发现它们的差异集中在一个低维的“表达子空间”中。我们在无需基准标签的情况下识别出该子空间，并从线性预测器中移除其贡献，从而获得一个被隔离的判断。在正确性、逻辑蕴含和安全性三类任务上，即使在未用于识别该子空间的新指令下，隔离后的判断依然保持可靠，即便此时模型的最终答案和原始预测器的性能已出现下降。当判断问题采用否定形式时……（摘要在此处被截断）

    arXiv:2610.21256v1 Announce Type: cross  Abstract: Language models are often asked to make judgments: whether a proposed answer is correct, whether one statement follows logically from another, or whether a response is safe. Yet changing the instructions that specify how the judgment should be expressed can make the final answer less reliable, even for the same input. We ask whether this reflects a change in internal judgment or a failure to express it. Comparing hidden-state vectors for the same input under different instructions, we find that their differences concentrate in a low-dimensional expression subspace. We identify this subspace without benchmark labels and remove its contribution from a linear predictor to obtain an isolated judgment. Across correctness, entailment, and safety, the isolated judgment remains reliable under instructions not used to identify the subspace, even when the model's final answer and the original predictor degrade. When the judgment question is nega
    
[^177]: System1-Arena：哪些决策可以交给系统1？

    System1-Arena: Which Decisions Can Be Left to System 1?

    [https://arxiv.org/abs/2610.21244](https://arxiv.org/abs/2610.21244)

    提出System1-Arena基准，通过在105个任务、92k个问题上统一评测系统1（单次前向传播输出概率）与系统2（生成式推理）模型，并结合138个闭环环境，系统性地回答哪些决策可以放心交给系统1模型。

    

    大型语言模型（LLM）经常需要做出离散决策，例如标记网络钓鱼或将问题升级给人工处理。系统1模型（如Jev和OpenAI Decisions API）在单次前向传播中返回各选项的概率，而系统2方法则通过生成推理步骤来决策。哪些决策系统1模型可以处理，哪些又需要生成式推理？我们提出了System1-Arena，这是一个基准测试，它向所有模型提供相同的决策请求，并根据其输出的概率进行评分。该基准包含来自268个公开来源的105个任务共92k个问题，分为四大类：文本分类与知识、应用决策、评估与控制、以及智能体与行动。它还包括138个闭环环境，涵盖网络购物、SWE-bench编程控制、机器人、游戏和驾驶等领域；在实时模式下，世界会在模型进行决策的同时持续演化。我们评估了30个模型：7个闭源LLM、13个在答案token位置读取概率的开源LLM，以及10个系统……

    arXiv:2610.21244v1 Announce Type: cross  Abstract: Large language models (LLMs) often make discrete decisions, such as flagging phishing or escalating to a human. System-1 models, such as Jev and the OpenAI Decisions API, return probabilities over options in a single forward pass, while System-2 approaches generate reasoning steps. Which decisions can System-1 models handle, and which require generative reasoning? We introduce System1-Arena, a benchmark that gives all models the same decision requests and scores their probabilities. It contains 92k questions across 105 tasks from 268 public sources, in four groups: text classification and knowledge, applied decisions, evaluation and control, and agent and action. It also includes 138 closed-loop environments, from web shopping and SWE-bench coding control to robotics, games, and driving; in real-time mode, the world evolves while a model decides. We evaluate 30 models: 7 closed LLMs, 13 open LLMs read at the answer token, and 10 System
    
[^178]: 稀疏自编码器中的表征再分配

    Representation Redistribution in Sparse Autoencoders

    [https://arxiv.org/abs/2610.21240](https://arxiv.org/abs/2610.21240)

    该论文提出再分配敏感度分数（RSS）来识别在提示变换下以一致方向重复出现的大幅特征响应，并在涵盖八个语言模型的27个预训练SAE上验证了表征再分配的可复现性，同时将重复的表征变化与下游读出贡献及决策失败精确解耦。

    

    稀疏自编码器将语言模型激活分解为稀疏特征，但旨在保持底层请求不变的提示变换可能会使激活活动在这些特征之间重新分布。在某一个变换后提示上观察到的特征变化未必会再次出现，即使是重复出现的变化，也未必会影响下游计算。我们引入了再分配敏感度分数，用以识别以一致方向重复出现的大幅特征响应，并在未参与选择过程的变体上评估由此得到的支持集。在涵盖八个语言模型的27个预训练SAE上，所测试的提示变换在这些被排除的变体上产生了可复现的再分布特征。随后，我们将重复出现的表征变化与带符号的读出贡献以及决策失败区分开来。对于固定的线性读出，一个精确分解展示了响应幅度、读出系数、方向对齐以及……

    arXiv:2610.21240v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) decompose language model activations into sparse features, but prompt transformations designed to preserve the underlying request can redistribute activity across them. A feature change observed on one transformed prompt need not recur, and even a recurring change need not affect a downstream computation. We introduce the redistribution sensitivity score (RSS), which identifies large feature responses that recur in a consistent direction, and evaluate the resulting supports on variants excluded from selection. Across 27 pretrained SAEs spanning eight language models, the tested prompt transformations produce a reproducible redistribution profile on these excluded variants. We then separate recurring representation change from signed readout contribution and decision failure. For a fixed linear readout, an exact decomposition shows how response magnitude, readout coefficients, directional alignment, and the cl
    
[^179]: 兼容性至关重要：在线策略蒸馏中的词元级教师选择框架

    Compatibility Matters: A Framework for Token-Level Teacher Selection in On-Policy Distillation

    [https://arxiv.org/abs/2610.21234](https://arxiv.org/abs/2610.21234)

    该论文提出在在线策略蒸馏中按词元动态选择教师的框架，从理论上刻画了师生兼容性、教师能力与学生多样性等关键因素，并证明不同场景下不同教师路由方案各有优势。

    

    在线策略蒸馏是最近被提出的一种有前景的持续学习范式，学生策略可以在词元级别接受监督。通常，OPD 在每个提示 rollout（甚至每次训练运行）中都使用单个教师。然而，这忽略了由在线策略特性所带来的另一维灵活性——即在 rollout 过程中进行词元级的教师选择。基于这一洞察，我们在理论上探讨了什么样的教师与学生对于 OPD 是兼容的，并利用这些发现得出了若干用于教师选择的经验性启发式方法。特别地，我们的框架涵盖了 OPD 的若干期望特性：兼容的师生推理模式、教师的能力以及所引发的学生多样性。在理论上，我们对几类教师路由方案进行了分类，并证明在不同场景下，不同的路由方案更有利于实现最优的学生学习效果。

    arXiv:2610.21234v1 Announce Type: new  Abstract: On-policy distillation (OPD) has recently been introduced as a promising paradigm for continual learning, where a student policy can be supervised at the token-level. Typically, OPD is performed using a single teacher per prompt rollout (or even per training run). However, this ignores a whole other axis of flexibility that is available due to the on-policy nature - token-level teacher selection during rollouts. Building on this insight, we theoretically explore what makes a teacher and a student compatible for OPD, and leverage our findings to derive several empirical heuristics for teacher selection. In particular, our framework captures several desiderata for OPD: compatible teacher-student reasoning patterns, teacher competence, and induced student diversity. Theoretically, we categorize several types of teacher routing schemes, and we prove that under different scenarios, different routing schemes are likely to be beneficial for opt
    
[^180]: 并非所有改变都是损害：面向大语言模型的更新匹配遗忘

    Not All Change Is Damage: Update-Matched Unlearning for Large Language Models

    [https://arxiv.org/abs/2610.21230](https://arxiv.org/abs/2610.21230)

    该论文提出“参照时序假设”，证明在大语言模型遗忘任务中，采用更新匹配的参照比更新前的参照能更好地保留模型学习新任务的可塑性，从而有效区分对已删知识的选择性抵抗与整体可塑性丧失。

    

    旨在从大语言模型中移除选定知识，以降低隐私和安全风险，同时保留有用的能力。对于发布后仍会持续微调的模型而言，这一目标并不充分：它们还必须在保持可塑性的同时抵抗对已删除知识的重新学习。现有的防御方法通过针对模拟的重新学习进行训练，同时试图保留有用的能力，但仅凭较低的重新学习程度无法区分选择性的抵抗能力与可塑性的整体丧失。要在模型更新后评估保留情况，需要确定更新后的模型应与什么保持接近。若采用更新前的参照，则会仅仅因为更新带来的有益改变偏离了早期输出而对这些改变进行惩罚。因此，我们提出了参照时序假设：在相当的目标抑制水平下，使用更新匹配的参照比使用更新前的参照能更好地保留可塑性。为了测试……（摘要截断）

    arXiv:2610.21230v1 Announce Type: new  Abstract: Unlearning aims to remove selected knowledge from large language models to mitigate privacy and safety risks while preserving useful capabilities. For models that continue to be fine-tuned after release, this goal is insufficient: they must also resist relearning the deleted knowledge while preserving plasticity, the ability to learn new tasks. Existing defenses train against simulated relearning while trying to preserve useful capabilities, but low relearning alone cannot distinguish selective resistance from a general loss of plasticity. Evaluating preservation after an update requires deciding what the updated model should remain close to. A pre-update reference penalizes useful changes induced by the update simply because they depart from earlier outputs. We therefore formulate the reference-timing hypothesis that, at comparable target suppression, an update-matched reference better preserves plasticity than a pre-update one. To test
    
[^181]: 我们该移除什么？评估RLVR数据去污染的可靠性与成本

    What Should We Remove? Evaluating the Reliability and Costs of RLVR Data Decontamination

    [https://arxiv.org/abs/2610.21229](https://arxiv.org/abs/2610.21229)

    该研究首次系统审计了RLVR数学语料库的去污染策略，发现语义匹配去污在余弦阈值0.80下有88%的抽样移除被证实为误移除，且没有任何预注册语义阈值能达到90%保守精确率目标，揭示了当前数据清洗政策存在严重的误删风险与数据损失成本。

    

    可验证奖励强化学习（RLVR）数学语料库通常会针对与基准测试的重叠进行例行筛查，但清洗策略的可靠性和数据损失成本仍未得到充分检验。我们审计了16个开放的RLVR数学语料库（共包含1,673,514个语料库专属训练条目），对照包含1,949个问题的六个基准测试。我们基于精确匹配、词汇匹配和语义匹配，使用594对经独立盲法双重审核的训练-基准配对，评估了八种去污染策略。校准集与保留集的基准问题和训练样本族互不相交。此外，我们对80个抽样训练条目的全部83个候选匹配进行了单独审核，以衡量条目级别的误移除情况。在余弦阈值0.80下，59个抽样移除中有52个（88%）被确认为误移除；但这并非已证实的总体比例。在预注册的阈值网格（0.80-0.98）中，没有任何语义阈值在校准集上达到90%的保守精确率目标。

    arXiv:2610.21229v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) math corpora are routinely screened for benchmark overlap, but the reliability and data-loss costs of cleaning policies remain underexamined. We audit sixteen open RLVR math corpora comprising 1,673,514 corpus-specific training items against six benchmarks containing 1,949 questions. We evaluate eight policies based on exact, lexical, and semantic matching using 594 independently and blindly dual-reviewed training-benchmark pairs. Calibration and held-out sets have disjoint benchmark questions and training-example families. A separate review of all 83 candidate matches for 80 sampled training items measures item-level false removal. At cosine threshold 0.80, 52 of 59 sampled removals (88%) were confirmed false removals; this is not a demonstrated population rate. No semantic threshold in our preregistered grid (0.80-0.98) meets the 90% conservative-precision target on calibration. The
    
[^182]: 基于拉格朗日-欧拉图神经常微分方程的多尺度时空流行病动力学学习与预测

    Learning and Forecasting Multiscale Spatiotemporal Epidemic Dynamics with Lagrangian-Eulerian Graph Neural ODEs

    [https://arxiv.org/abs/2610.21223](https://arxiv.org/abs/2610.21223)

    提出LE-EpiGNODE模型，以机制引导的拉格朗日-欧拉图神经常微分方程取代通用消息传递，将年龄-接触信息用于感染生成、人口流动用于保守性人口运输，在中长期流行病时空预测中取得显著优势。

    

    准确的中长期流行病预测需要从部分观测数据中外推高维、异质且非平稳的动力学过程。然而，现有混合模型通常未能明确说明哪些观测数据对应哪些流行病过程，以及学习到的图交互如何改变系统状态。我们提出了LE-EpiGNODE，该方法用过程特定的、机制引导的神经算子取代通用消息传递机制：年龄-接触信息驱动拉格朗日式感染生成过程，而人口流动驱动结构保持的通用微分方程框架内的保守欧拉式人口运输过程。我们在14个受控合成机制（涵盖时间调制、状态依赖、非线性和耦合动力学）、三个互补的COVID-19真实场景、21个对比模型以及五组针对性消融实验中评估了LE-EpiGNODE。预测性能的提升往往在更长预测时间范围内表现得更为显著；在最长的预测时间范围内，LE-EpiGNODE（原文此处截断）

    arXiv:2610.21223v1 Announce Type: new  Abstract: Accurate medium- to long-range epidemic forecasting requires extrapolating high-dimensional, heterogeneous and nonstationary dynamics from partial observations. Yet hybrid models often leave implicit which observations inform which epidemic processes and how learned graph interactions modify system state. We introduce LE-EpiGNODE, which replaces generic message passing with process-specific, mechanism-guided neural operators: age-contact information drives Lagrangian infection generation, whereas mobility drives conservative Eulerian population transport within a structure-preserving universal differential equation. We evaluated LE-EpiGNODE across 14 controlled synthetic mechanisms spanning time-modulated, state-dependent, nonlinear and coupled dynamics, three complementary COVID-19 settings, 21 comparator models and five targeted ablations. Forecasting gains tended to become more pronounced at longer horizons; at the longest horizon, LE
    
[^183]: 干净数据仍可能携带后门：面向模型复用的支持持久化后门

    Clean Data Can Still Carry Backdoors: Support-Persistent Backdoors for Model Reuse

    [https://arxiv.org/abs/2610.21208](https://arxiv.org/abs/2610.21208)

    本文提出DAR后门攻击方法，通过挖掘数据中天然存在的低维子空间模式并进行标签投毒，使后门在干净数据知识蒸馏后依然存留，打破了“知识蒸馏可有效净化后门”的普遍认知。

    

    传统的后门攻击会在训练数据中注入人工触发模式，导致模型对被触发的输入产生错误分类，同时对干净样本保持正常行为。由于这些人工触发模式并不存在于干净数据集中，标准的知识蒸馏（KD）通常会将它们消除，因此人们普遍认为知识蒸馏是一种有效的净化防御手段。在本文中，我们提出了DAR（Discover-and-Relabel，发现与重标注），一种支持持久化的后门方法，即使经过干净数据蒸馏后仍能保留其攻击行为。DAR利用子空间聚类识别低维特征子空间中天然存在的丰富模式，并对天然满足所发现规则的样本仅进行标签投毒。在测试阶段，通过局部修改目标图像的相应子空间来激活后门。通过实例化空间算子和频率算子，DAR实现了与最先进方法相当的攻击准确率。

    arXiv:2610.21208v1 Announce Type: cross  Abstract: Traditional backdoor attacks inject artificial trigger patterns into training data, causing models to misclassify triggered inputs while maintaining normal behavior on clean samples. Since these artificial triggers are absent in clean datasets, standard knowledge distillation (KD) typically eliminates them, leading to the common belief that KD serves as an effective purification defense. In this paper, we propose DAR (Discover-and-Relabel), a support-persistent backdoor method that retains its behavior even after clean-data distillation. DAR identifies naturally abundant patterns in low-dimensional feature subspaces using subspace clustering and applies label-only poisoning to samples that inherently satisfy the discovered rule. At test time, the backdoor is activated by locally modifying the corresponding subspace of the target images. By instantiating spatial and frequency operators, DAR achieves attack accuracy comparable to state-o
    
[^184]: 艰难之路，唯一之路：在最难样本上训练的可证明优势

    Hard Way, The Only Way: Provable Merits of Training on Hardest Examples

    [https://arxiv.org/abs/2610.21201](https://arxiv.org/abs/2610.21201)

    该论文通过加权经验风险最小化建立了数据剪枝的精确高维理论，并证明在样本量远超维度的情形下，仅保留最难样本进行训练是可证明的最优剪枝策略。

    

    尽管人们对AI系统中激进数据剪枝的“不合理有效性”产生了大量实证研究兴趣，但对这一现象的统一理论理解仍然缺失。在本工作中，我们通过加权经验风险最小化（ERM）为数据剪枝下的学习提供了一个精确的高维理论。与先前工作类似，我们分析了沿输入空间一维子空间进行剪枝的设定。我们的分析精确刻画了在输入维度N与样本量P以固定比值N/P趋于无穷大的渐近极限下测试误差曲线的极限形态。所得理论将文献中近期的若干结果作为特例加以恢复，并为评估剪枝策略与损失函数及标签噪声之间的相互作用提供了一个通用视角。尤为关键的是，我们刻画了在样本充足（P远大于N）情形下的最优剪枝策略。我们证明了……

    arXiv:2610.21201v1 Announce Type: new  Abstract: Despite a surge of empirical interest in the ``unreasonable effectiveness'' of drastic data pruning for AI systems, a unified theoretical understanding of this phenomenon remains elusive. In this work, we provide a precise high-dimensional theory of learning under data pruning via weighted empirical risk minimization (ERM). As in previous work, we analyze a setting where pruning is performed along a one-dimensional subspace of the input space. Our analysis precisely characterizes the limiting test error curves in the regime where the input dimension N and sample size P tend to infinity with a fixed ratio N/P. The resulting theory recovers several recent results in the literature as special cases and provides a general lens to evaluate how pruning strategies interact with the loss function and label noise. Crucially, we characterize the optimal pruning strategy in the high-sample regime where P greatly exceeds N. We demonstrate that strat
    
[^185]: 在预训练和检索阶段对蛋白质设计系统语料库的投毒攻击

    Poisoning the corpora of protein design systems during pre-training and retrieval

    [https://arxiv.org/abs/2610.21200](https://arxiv.org/abs/2610.21200)

    本研究首次证明，仅需在蛋白质语言模型的预训练语料库或文本引导蛋白质生成器的检索语料库中注入少量（低至200条）含触发器的投毒序列，即可使模型产生攻击者可控的特定序列生成行为，揭示了蛋白质设计AI系统面临的严重数据安全威胁。

    

    蛋白质基础模型日益支持新型生物序列的生成，然而其对大型公共数据集的依赖为其所学习和检索的材料遭受对抗性操纵创造了机会。我们研究了针对蛋白质设计系统的两种纯数据投毒攻击，分别针对自回归蛋白质语言模型的预训练语料库和文本引导蛋白质生成器的检索语料库。对于第一种攻击，我们预训练了参数量为3亿的ProtGPT2和ProGen2变体，引入了包含两种触发器之一、后接三种有效载荷之一的投毒蛋白质序列。在所有投毒运行中，两种架构在均匀和递增投毒计划下均获得了高度可靠的触发器依赖生成能力。在递增暴露方式下，最早的多数激活仅出现在200个污染样本之后，而在均匀计划下，最早激活发生于……

    arXiv:2610.21200v1 Announce Type: new  Abstract: Protein foundation models increasingly support the generation of novel biological sequences, yet their reliance on large public datasets creates opportunities for adversarial manipulation of the material they learn from and retrieve. We investigate two data-only poisoning attacks against protein design systems, targeting the pre-training corpus of autoregressive protein language models and the retrieval corpus of a text-guided protein generator. For the first, we pre-trained ProtGPT2 and ProGen2 variants of 300 million parameters, introducing poisoned protein sequences containing one of two triggers followed by one of three payloads. Across all poisoned runs, both architectures acquired highly reliable trigger-dependent generation under uniform and ramped poisoning schedules. Under ramped exposure, the earliest majority activation occurred after only 200 contaminated examples, while for the uniform schedule earliest activation occurred a
    
[^186]: 通过无需训练的权重编辑抵抗有害指令

    Resisting Harmful Instructions through Training-Free Weight Editing

    [https://arxiv.org/abs/2610.21189](https://arxiv.org/abs/2610.21189)

    本文提出一种无需训练的权重编辑方法，仅需在少量提示对上执行前向传播即可修改模型权重，使模型抵抗指令性谎言和性别歧视等有害指令，且这种防御无法通过修改提示来绕过。

    

    遵循指令的语言模型可能会服从与预期行为和安全对齐相冲突的请求。我们提出了一种无需训练的权重编辑方法，用于减少模型对特定不良指令的服从。该方法只需针对每个目标行为在少量提示对上执行前向传播，并且由于编辑结果存储在模型的权重中，因此改变或删除提示中的指令也无法撤销权重修改。我们在五个指令微调模型上针对两种目标有害行为（指令性谎言和指令性性别歧视）评估了该方法。在两个指令性谎言基准测试中，与最强的替代方法相比，该方法将平均诚实回答率分别提高了6.2和4.6个百分点，在五个模型中的三个上提升显著，另外两个模型的差异在噪声范围内。在涵盖七种提示格式的对照研究中，该方法比单层激活引导方法保持了更高的诚实回答率。

    arXiv:2610.21189v1 Announce Type: new  Abstract: Instruction-following language models can comply with requests that conflict with intended behaviors and safety alignment. We introduce a training-free weight-editing approach for reducing compliance with targeted undesirable instructions. Our method needs only forward passes on a small set of prompt pairs per target behavior, and because the edit is stored in the model's weights, changing or deleting instructions in the prompt cannot undo the weight modification. We evaluate the approach on two target harmful behaviors across five instruction-tuned models: instructed falsehood and instructed sexism. On two instructed-lying benchmarks, it improves mean honest response rates by 6.2 and 4.6 percentage points over the strongest alternative, significantly on three of the five models and within noise on the other two. In a controlled study of seven prompt formats, it maintains higher honest-response rates than single-layer activation steering
    
[^187]: 重型柴油车燃油与排放监测的传感器诊断

    Sensor Diagnosis for Fuel and Emissions Monitoring in Heavy-Duty Diesel Vehicles

    [https://arxiv.org/abs/2610.21188](https://arxiv.org/abs/2610.21188)

    本文提出基于健康测量数据学习关系的传感器故障诊断与燃油修正程序，在重型柴油发动机217个稳态运行点上对五种合成故障类型评估，实现了77.3%的诊断覆盖率且诊断准确率达92.1%，有助于修正因传感器故障导致的燃油消耗和碳排放估计偏差。

    

    有故障的传感器读数可能会扭曲对柴油发动机燃油消耗和二氧化碳排放的估计。我们评估了两种利用从健康测量数据中学到的关系来诊断故障并指导燃油修正的程序。诊断程序测试用估计的健康值替换疑似故障读数是否有助于模型更准确地预测另一个传感器的记录读数，然后预测故障类型并决定是否报告诊断结果。我们在来自重型柴油发动机的217个稳态运行点上评估了五种合成故障类型，并在训练时排除了选定发动机转速下的所有测量数据。在31,500组各含20个运行点的故障注入数据组中，该程序对77.3%的组发出了诊断，其中发出的诊断有92.1%是正确的。燃油修正程序在各个运行点上使用不同的传感器选择规则，测试每个替换……

    arXiv:2610.21188v1 Announce Type: new  Abstract: Faulty sensor readings can distort estimates of fuel use and carbon dioxide emissions from diesel engines. We evaluate two procedures that use relationships learned from healthy measurements to diagnose faults and guide fuel correction. The diagnosis procedure tests whether replacing a suspected faulty reading with an estimated healthy value helps a model predict another sensor's recorded reading more accurately. It then predicts the fault type and decides whether to report the diagnosis. We evaluate five synthetic fault types on 217 steady-state operating points from a heavy-duty diesel engine, excluding all measurements at selected engine speeds from training. Across 31,500 fault-injected groups of 20 operating points, it issues diagnoses for 77.3% of groups, with 92.1% of diagnoses issued on these groups correct. The fuel correction procedure uses a different sensor-selection rule at individual operating points, testing how each repla
    
[^188]: 挖掘、采样、渲染：生成模型的测试时多样性对齐

    Mine, Sample, Render: Test-Time Diversity Alignment Of Generative Models

    [https://arxiv.org/abs/2610.21185](https://arxiv.org/abs/2610.21185)

    提出 MSR 测试时框架，通过从真实语料库挖掘类别模式并采样属性配置，将生成模型仅作为条件渲染器，从而显著提升生成输出的多样性与语料库分布的对齐度。

    

    现代生成模型能够产生高质量的单独输出，但难以实现多样化采样，即使被明确要求这样做也是如此。我们提出了 MINE-SAMPLE-RENDER（MSR），这是一种测试时方法，将多样性外化为一个显式的、由语料库导出的采样过程。我们的流程（i）从真实制品的语料库中挖掘出可解释的类别模式，并提供用于采样的统计数据；（ii）根据语料库导出的统计数据采样属性配置；（iii）将生成模型纯粹用作已采样配置的条件渲染器。在天气报告生成、用户模拟、网页设计、文生图以及智能体生成的3D资产等任务上，我们评估了 MSR 引导生成的多样性和输出质量，结果表明 MSR 改善了生成输出与语料库分布的对齐程度以及生成输出的多样性。

    arXiv:2610.21185v1 Announce Type: new  Abstract: Modern generative models produce high-quality individual outputs but struggle to sample diversely, even when explicitly instructed to do so. We introduce MINE-SAMPLE-RENDER (MSR), a test-time approach that externalizes the diversity into an explicit, corpus-derived sampling procedure. Our pipeline (i) mines an interpretable categorical schema from a corpus of real artifacts and supplies statistics for sampling; (ii) samples attribute configurations from the corpus-derived statistics; (iii) uses a generative model purely as a conditional renderer of a sampled configuration. Across weather report generation, user simulation, webpage design, text-to-image generation, and agent-generated 3D assets, we evaluate the diversity and output quality of MSR-guided generation and show that MSR improves alignment with corpus distributions and diversity of generated outputs.
    
[^189]: 评判评判者：评分标准奖励机制中的置信度与分歧

    Judging the Judges: Confidence and Disagreement in Rubrics as Rewards

    [https://arxiv.org/abs/2610.21179](https://arxiv.org/abs/2610.21179)

    该论文提出 RaRE 评判者集成策略，通过基于置信度的软 Likert 奖励和剔除高分歧样本的掩码机制，显著提升了基于评分标准的强化学习在不可验证领域中的训练效果。

    

    评分标准奖励通过使用大语言模型评判者根据结构化评分标准对回答进行打分，将强化学习扩展到了不可验证的领域。先前的工作主要聚焦于改进评分标准本身或强化单一评判者，而对于在 RaR 训练过程中如何使用不同的单一或多个评判者则关注较少。我们首先发现，在使用五个前沿及开放权重的评判者于 RaR-Medicine 和 RaR-Science 数据集上训练 Qwen3-4B-Base 模型时，仅评判者身份的不同就会导致分数产生显著差异。在此基础上，我们提出了 RaRE（基于集成评判者的评分标准奖励），这是一种新颖的 RaR 评判者集成策略，包含两个机制：一是软 Likert 奖励，利用评判者的对数概率（logprobs）对置信度更高的评判者赋予更大权重；二是“分歧即剔除”掩码，将高分歧的 rollout 从 GRPO 更新中排除。我们发现这种设计至关重要：虽然对评判者分数进行简单平均的做法无法超越……

    arXiv:2610.21179v1 Announce Type: cross  Abstract: Rubrics as Rewards (RaR) extends reinforcement learning to non-verifiable domains by using an LLM judge to score responses against a structured rubric. Prior work has focused on improving the rubric or strengthening the single judge, with comparatively little attention to how different single or multiple judges might be used during RaR training. We first show that judge identity alone produces a substantial spread in scores when training Qwen3-4B-Base on RaR-Medicine and RaR-Science across five frontier and open-weight judges. Building on this, we propose RaRE (Rubrics as Rewards with Ensembles), a novel judge ensemble strategy for RaR with two mechanisms: a soft-Likert reward that uses judge logprobs to weight more confident judges more heavily, and a drop-on-disagreement mask that excludes high-disagreement rollouts from the GRPO update. We find that this design is critical: while a naive average of judge scores fails to outperform t
    
[^190]: 语言模型在工作记忆中共享一个“读出瓶颈”

    Language Models Share a Readout Bottleneck in Working Memory

    [https://arxiv.org/abs/2610.21176](https://arxiv.org/abs/2610.21176)

    该论文发现，语言模型与人类共享同一工作记忆失败机制——错误并非源于注意力定位不准（精度不足），而是正确的目标信息在“读出”阶段受到干扰，即语言模型存在一个共同的工作记忆读出瓶颈。

    

    大型语言模型（LLM）在N-back任务上复现了人类工作记忆的特征：随着记忆负载N的增加，性能下降，且错误集中在“诱饵项”上——即在相邻但不正确的序列位置重复当前刺激的项目。我们探究这种共同的行为特征是否反映了共同的机制。文献提供了两种候选解释：在“精度解释”中，softmax注意力无法将权重集中于N个位置之前的项目，因此检索到的是相邻项目；在“干扰解释”中，注意力已正确选中目标项目，但竞争项目的内容破坏了后续决策。我们在来自三个系列的15个预训练LLM（参数量从7000万到140亿）以及从头训练的极简transformer中对这两种解释进行了区分。在所有三个系列中，承担该任务的注意力头在大多数未命中的匹配上其实都注意到了目标项目；线性探针能够近乎完美地解码出相关信息。

    arXiv:2610.21176v1 Announce Type: cross  Abstract: Large language models (LLMs) reproduce a human working-memory signature on the N-back task: performance declines as memory load N increases, and errors concentrate on lures, items that repeat the current stimulus at a neighboring but incorrect serial position. We ask whether this shared behavioral signature reflects a shared mechanism. The literature offers two candidate accounts. On a precision account, softmax attention cannot concentrate its weight on the item N positions back, so adjacent items are retrieved instead. On an interference account, attention selects the correct item, but the content of competing items corrupts the decision. We dissociate these accounts in 15 pretrained LLMs from three families (70 million to 14 billion parameters) and in minimal transformers trained from scratch. In all three families, the attention heads that carry the task attend to the target on most missed matches. Linear probes decode near-perfect
    
[^191]: 面向可变成本分布式训练的无协调负载正则化

    Coordination-Free Workload Regularization for Variable-Cost Distributed Training

    [https://arxiv.org/abs/2610.21162](https://arxiv.org/abs/2610.21162)

    提出无协调数据加载策略BCL，通过均衡各rank批次的估计成本将批次成本变异性降低高达82.8%，在无需跨rank通信或运行时修改的情况下实现最高1.343倍的训练加速。

    

    同步数据并行训练通常为每个rank分配相同数量的记录，即使记录的成本相差数个数量级。对于可变长度模型，这会将输入差异转化为空闲时间，因为每一步都必须等待本rank最昂贵的批次。我们提出了批次成本前瞻（Batch Cost Lookahead, BCL），这是一种无协调的数据加载策略，它缓冲一个有界的rank本地窗口，并将其重新分区为估计成本均衡的固定容量打包。BCL保留了训练器的记录数量契约，既不需要跨rank元数据交换，也不需要修改运行时。在基于公开工作负载分布的仿真中，启用BCL可将每个rank的估计批次成本变异性降低高达82.8%。在八块H200 GPU上对13个NeMo-RL SFT语料库的测试中，rank本地BCL比池化MFFD快1.050至1.167倍。在内部序列工作负载上的H200编译实测显示加速比达1.197至1.343倍。

    arXiv:2610.21162v1 Announce Type: cross  Abstract: Synchronous data-parallel training usually assigns every rank the same number of records, even when record costs vary by orders of magnitude. For variable-length models, this converts input variation into idle time because each step waits for its most expensive rank-local batch. We present Batch Cost Lookahead (BCL), a coordination-free data-loading policy that buffers a bounded rank-local window and repartitions it into fixed-cardinality packs with balanced estimated cost. BCL preserves the trainer's record-count contract and requires neither cross-rank metadata exchange nor runtime modification. In simulation over public workload distributions, enabling BCL reduces per-rank estimated batch-cost variability by up to 82.8%. Across 13 NeMo-RL SFT corpora on eight H200 GPUs, rank-local BCL runs 1.050 to 1.167 times faster than pooled MFFD. Compiled H200 measurements on an internal sequence workload show speedups of 1.197 to 1.343 times o
    
[^192]: 什么时候更强大的LLM值得付费？基于切换混合保真度预言机的随机优化

    When is a better LLM worth paying for? Stochastic Optimization with Switching Mixed-Fidelity Oracles

    [https://arxiv.org/abs/2610.21139](https://arxiv.org/abs/2610.21139)

    针对“何时值得为更强的LLM付费”这一核心问题，本文提出了两个成本感知的随机优化算法框架，通过在具有系统性偏差和状态依赖方差的混合保真度一阶预言机之间自适应切换，来平衡LLM调用成本与下游优化精度。

    

    大语言模型（LLM）越来越多地被用于决策流程的上游环节，用于清洗数据、检测标签错误和重建缺失值。然而，LLM辅助的数据整理带来了一种新的运营权衡：更高保真度的模型和更丰富的提示词可能成本大幅增加，而这种额外保真度的价值取决于下游优化的精度。这引出了本工作的核心问题：针对特定的下游任务，什么时候应该为更好的LLM付费？我们通过具有混合保真度一阶预言机的随机优化来研究这个问题，这些预言机可能存在系统性偏差和状态依赖的方差。我们开发了两个成本感知的算法框架，允许在随机预言机之间进行自适应切换。在偏差已知的情况下，第一个元算法采用了一种带重启的、成本感知的策略，能够在优化的各个阶段自适应地选择预言机的保真度。其累计成本……

    arXiv:2610.21139v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used upstream of decision-making to clean data, detect label errors, and reconstruct missing values. Yet LLM-assisted data curation creates a new operational tradeoff: higher-fidelity models and richer prompts can be substantially more expensive, while the value of this additional fidelity depends on the downstream optimization accuracy. This raises the central question of this work: when should one pay for a better LLM, given a specific downstream task at hand?   We study this question through stochastic optimization with mixed-fidelity first-order oracles, which can have systematic bias and state-dependent variance. We develop two cost-aware algorithmic frameworks that allow adaptive switching between stochastic oracles. With known bias, the first meta-algorithm applies a restarted, cost-aware policy that adaptively selects oracle fidelity across optimization stages. Its cumulative cost i
    
[^193]: 乘性半柯西过程的路径收缩与概率校准

    Pathwise Shrinkage and Probabilistic Calibration of the Multiplicative Half-Cauchy Process

    [https://arxiv.org/abs/2610.21136](https://arxiv.org/abs/2610.21136)

    本文通过对数尺度随机游走表示系统刻画了乘性半柯西过程的收缩、扩张与临界等路径性质，并推导出连接可解释先验概率陈述与尺度参数的半解析校准规则。

    

    我们通过对数尺度随机游走表示来刻画乘性半柯西过程，建立了其收缩、扩张与临界区域，几乎必然可和性，局部反弹性质以及有效维度。我们还推导出一种半解析的校准规则，将可解释的先验概率陈述与其尺度参数联系起来。

    arXiv:2610.21136v1 Announce Type: cross  Abstract: We characterize the multiplicative half-Cauchy process through its log-scale random-walk representation, establishing shrinkage, expansion, and critical regimes, almost-sure summability, local rebounds, and effective dimension. We also derive a semi-analytic calibration rule linking interpretable prior probability statements to its scale parameter.
    
[^194]: 预测IVF/ICSI鲜胚移植后的活产：差异化的内细胞团/滋养外胚层形态编码与混合注意力-逻辑回归集成模型

    Predicting Live Birth After Fresh Embryo Transfer in IVF/ICSI: Differential ICM/Trophectoderm Morphology Encoding and a Hybrid Attention--Logistic Regression Ensemble

    [https://arxiv.org/abs/2610.21134](https://arxiv.org/abs/2610.21134)

    该研究构建了单中心IVF/ICSI数据集，通过新颖的内细胞团/滋养外胚层差异化形态编码方法以及混合注意力网络与逻辑回归的集成模型，实现了适用于小型临床表格数据的鲜胚移植活产预测。

    

    在鲜胚移植前预测活产可以支持IVF/ICSI（体外受精/卵胞浆内单精子注射）治疗中的医患共同决策，但常规临床数据规模小、异质性强，且包含易导致结果泄露的依赖治疗路径的变量。本研究旨在构建一个精心整理的单中心IVF/ICSI数据集，并开发适用于小型表格临床数据的预测模型。材料与方法：我们收集了616个鲜胚移植周期（251例活产，365例非活产），包含临床、实验室和胚胎学预测变量，并采用一种新颖的内细胞团与滋养外胚层差异化加权方法对胚胎形态进行编码。经过KNN缺失值插补、稳健缩放以及50个特征的筛选，我们开发了混合注意力网络（HAN）——即一个可学习的特征注意力门控后接一个紧凑的多层感知机，以及一个带有验证集调优权重的HAN与逻辑回归（LR）集成模型。这些模型与经典机器学习模型进行了基准比较（原文摘要在此处截断）。

    arXiv:2610.21134v1 Announce Type: new  Abstract: Predicting live birth before fresh embryo transfer could support shared decision-making in IVF/ICSI, but routine clinical data are small, heterogeneous, and contain pathway-dependent variables prone to outcome leakage. We aimed to construct a curated single-centre IVF/ICSI dataset and develop a prediction model suited to small tabular clinical data. \textit{Materials and Methods:} We assembled 616 fresh embryo-transfer cycles (251 live birth, 365 non-live birth) with clinical, laboratory, and embryological predictors, and embryo morphology encoded with a novel differential weighting of the inner cell mass and trophectoderm. After KNN imputation, robust scaling, and selection of 50 features, we developed a Hybrid Attention Network (HAN), a learnable feature-attention gate followed by a compact multilayer perceptron, and a HAN + logistic-regression (LR) ensemble with a validation-tuned weight. Models were benchmarked against classical mach
    
[^195]: 面向少步生成模型的噪声空间奖励对齐

    Noise-Space Reward Alignment for Few-Step Generative Models

    [https://arxiv.org/abs/2610.21125](https://arxiv.org/abs/2610.21125)

    提出噪声策略优化（NPO），通过在生成器初始噪声空间上学习摊销流策略、将噪声分布向奖励倾斜，从而在保持少步生成效率优势的同时解决少步生成模型的奖励对齐问题。

    

    少步生成模型能够在一至数次评估中生成高质量样本。然而，将这些模型引导至期望的属性仍然十分困难，尤其是在仅有黑盒奖励评估可用的情况下。虽然可以将为迭代式扩散模型和基于流的模型开发的方法（如直接强化学习微调或测试时扩展）应用于此，但我们发现这两种方法都不能很好地迁移到少步场景：直接微调存在严重的奖励过优化问题，而测试时扩展则牺牲了少步生成的效率优势。我们提出噪声策略优化，明确针对少步生成模型的奖励对齐问题。其核心思想是在生成器的初始噪声空间上学习一个摊销流策略。训练得到的策略将高斯噪声传输至一个偏向奖励的噪声分布，该分布经过冻结生成器的前推映射后恰好对应奖励对齐的样本分布。

    arXiv:2610.21125v1 Announce Type: new  Abstract: Few-step generative models produce high-quality samples in one or a few evaluations. However, steering them toward desired properties remains difficult, especially when only black-box reward evaluations are available. While one could apply approaches developed for iterative diffusion and flow-based models, such as direct RL fine-tuning or test-time scaling, we find that neither transfers well to the few-step regime: direct fine-tuning suffers from severe reward over-optimization, and test-time scaling sacrifices the efficiency advantage of few-step generation. We propose Noise Policy Optimization (NPO), which explicitly targets the alignment problem for few-step generative models. The key idea is to learn an amortized flow policy over the initial noise space of the generator. The trained policy transports Gaussian noise to a reward-tilted noise distribution, whose pushforward distribution through the frozen generator is exactly the rewar
    
[^196]: DST-Net：用于时间序列预测的分解时空网络

    DST-Net: Decomposed Spatio-Temporal Network for Time Series Forecasting

    [https://arxiv.org/abs/2610.21123](https://arxiv.org/abs/2610.21123)

    提出DST-Net解耦时空网络，摒弃对分解后各分量的同质化建模，针对趋势分量设计潜在图块以捕获全局拓扑依赖与协同运动，针对季节分量设计高斯多尺度注意力机制以聚焦局部时间演化。

    

    序列分解已成为深度时间序列预测模型中广受认可的范式。然而，现有方法主要对所有子序列采用同质化的建模策略，从而忽略了不同分量中固有的不同归纳偏置。我们发现趋势分量在通道间表现出显著的空间耦合，但带有明显的时间冗余；而季节性分量的特征是局部时间演化，但容易受到注意力干扰。为了解决这一问题，我们提出了一种新颖的解耦架构，为每个分量量身定制特定的建模策略以契合其内在属性。对于趋势分支，我们设计了潜在图块，采用潜在图建模来捕获全局拓扑依赖关系和系统级的协同运动。相反，对于季节分支，我们引入了高斯多尺度注意力……（原文摘要到此截断）

    arXiv:2610.21123v1 Announce Type: cross  Abstract: Series decomposition has emerged as a widely acknowledged paradigm in deep time series forecasting models. However, existing approaches predominantly employ a homogeneous modeling strategy for all subsequences, thereby overlooking the distinct inductive biases inherent in different components. We identify that the trend component exhibits dominant spatial coupling across channels but carries significant temporal redundancy, whereas the seasonal component is characterized by local temporal evolution but remains susceptible to attention distraction. To address this, we propose a novel disentangled architecture that tailors specific modeling strategies to align with the intrinsic properties of each component. For the trend branch, we design a Latent Graph Block, employing latent graph modeling to capture global topological dependencies and system-wide co-movements. Conversely, for the seasonal branch, we introduce a Gaussian Multi-scale A
    
[^197]: 光谱-空间解释

    Spectral-Spatial Interpretation

    [https://arxiv.org/abs/2610.21114](https://arxiv.org/abs/2610.21114)

    提出 SSI 方法，通过在每个空间位置对通道构建条件博弈并计算 Shapley 值，生成完整的光谱-空间归因矩阵，实现了满足 Shapley 公平性且具有清晰局部解释性的多光谱模型解释。

    

    事后图像解释通常使用空间热力图来显示模型关注的区域，或使用全局通道分数来指示哪些通道重要。这两种分离的视角对于多光谱数据具有局限性，因为通道相关性往往是空间局部化的且具有物理意义。我们提出了光谱-空间解释，这是一种针对位置-通道参与者 (g,c) 的全网格归因方法。对于每个空间位置，SSI 固定已观测到的周围上下文，在该位置的通道上定义一个条件博弈，并将 Shapley 值应用于这一局部博弈。其结果是一个完整的光谱-空间归因矩阵，而不是单一的热力图或全局通道排序。SSI 具有清晰的局部解释性：它在每个位置内部满足 Shapley 公平性，并且对其通道归因进行聚合后，恰好能够恢复将该位置替换为基线时所得的位置级上下文对比度。

    arXiv:2610.21114v1 Announce Type: new  Abstract: Post-hoc image explanations usually show where a model looks using spatial heatmaps or indicate which channels matter using global channel scores. These separate views are limiting for multispectral data because channel relevance is often spatially localized and physically meaningful. We introduce Spectral-Spatial Interpretation (SSI), a full-grid attribution method over position-channel players (g,c). For each spatial position, SSI fixes the observed surrounding context, defines a conditional game over the channels at that position, and applies the Shapley value to this local game. The result is a full spectral-spatial attribution matrix rather than a single heatmap or a global channel ranking. SSI has a clear local interpretation. It satisfies Shapley fairness within each position, and aggregating its channel attributions exactly recovers the position-level contextual contrast obtained by replacing that position with the baseline. SSI 
    
[^198]: Kolab：长周期工作流中人机协作的基准测试

    Kolab: Benchmarking Human-Agent Collaboration in Long-Horizon Workflows

    [https://arxiv.org/abs/2610.21109](https://arxiv.org/abs/2610.21109)

    本文推出Kolab数据集，通过记录50位来自40个职业的专业人士与AI智能体协作处理真实长周期任务的115小时以上屏幕录像，揭示了人类更重视与智能体的持续交互而非一次性自动化完成，说明现有侧重自动化的基准测试无法衡量真实世界的人机协作效果。

    

    当人类与AI智能体在长周期任务上协作时，智能体的贡献会带来什么？在解决长周期任务时，人类更加重视与AI智能体的持续交互（例如修订计划），而非智能体一次性完成任务的能力。然而，现有的智能体基准测试大多更侧重于自动化。在这些基准测试上的出色表现并不意味着有效的真实世界协作。专业工作流为此提供了一个有用的研究场景，因为它们通常包含相互依赖的步骤。为了填补这一空白，我们推出了Kolab——一个全新的数据集，包含来自40个职业的50位专业人士与智能体协作解决真实的、长周期问题的超过115小时的屏幕录制。Kolab还包含参与者对协作过程和输出质量的评价。我们的分析揭示了多样化的协作模式，突出了专业人士何时以及如何与智能体进行协调的重要性。

    arXiv:2610.21109v1 Announce Type: new  Abstract: When humans collaborate with AI agents on long-horizon tasks, what happens to the agents' contributions? For solving long-horizon tasks, humans place greater importance on ongoing interaction with AI agents (e.g., revising plans) than on an agent's ability to complete the task in a single attempt. However, most agentic benchmarks have focused more on automation. Strong performance on such benchmarks does not imply effective real-world collaboration. Professional workflows offer a useful setting, as they often involve interdependent steps. To address this gap, we introduce Kolab, a new dataset of 115+ hours of screen recordings of 50 professionals from 40 occupations collaborating with agents on realistic, long-horizon problems. Kolab also contains participants' collaboration and output quality judgments. Our analysis reveals diverse collaboration patterns, highlighting the importance of when and how professionals coordinate with agents. 
    
[^199]: ToolBottleneckBench：在受控压力与固定运行框架预算下测量工具智能体性能

    ToolBottleneckBench: Measuring Tool-Agent Performance under Controlled Pressures and Fixed Harness Budgets

    [https://arxiv.org/abs/2610.21098](https://arxiv.org/abs/2610.21098)

    该论文提出ToolBottleneckBench基准，在固定框架预算下通过单独变化工具目录、可见菜单、认证工作流或返回记录编码等压力因素来测量工具智能体性能，发现可见工具菜单是唯一对模型造成大幅惩罚的压力且其根源是完成上限处的截断，而大多数失败实际终止于框架限制而非模型能力不足。

    

    工具智能体的成功得分同时反映了其模型能力和运行它的框架（harness）的预算。我们提出ToolBottleneckBench，这是一个在受控压力与固定框架预算下测量工具智能体性能的基准测试。合成的企业任务保持不变，而每次仅变化一种压力因素：已注册的工具目录、可见的工具菜单、经过认证的工作流程，或返回记录的编码方式。一个精确的评分器会记录每个失败回合中第一个未满足的义务，以及（若存在）终止该回合的框架限制。在21种模型与框架配置中，可见工具菜单是三种作用于模型的压力因素中唯一造成大幅惩罚的因素，且该惩罚与完成上限处的截断相关。将上限提高十六倍后，在截断频发的配置中大部分惩罚被消除。大多数失败终止于某个框架限制，且失败记录对指定压力的识别可靠性并不比（原文在此处截断）……

    arXiv:2610.21098v1 Announce Type: cross  Abstract: The success score of a tool agent reflects both its model and the budgets of the harness that runs it. We present ToolBottleneckBench, a benchmark that measures tool-agent performance under controlled pressures and fixed harness budgets. Synthetic enterprise tasks stay fixed while one pressure varies: the registered tool catalog, the visible tool menu, the certified workflow, or the encoding of returned records. An exact grader records the first unmet obligation of each failed episode and the harness limit, if any, that ended it. Across 21 model and harness configurations, the visible menu is the only one of the three pressures that act on the model with a large penalty, and this penalty tracks truncation at the completion cap. Raising the cap sixteenfold removes most of the penalty in the truncation-heavy configurations. Most failures end at a harness limit, and the failure record identifies the assigned pressure no more reliably than
    
[^200]: 探测与恢复：何时以及如何评估智能体执行

    Probe-and-Resume: When and How to Evaluate Agent Execution

    [https://arxiv.org/abs/2610.21088](https://arxiv.org/abs/2610.21088)

    该论文发现在终端类计算机使用环境中，MCTS测试时搜索失效的原因是评估器在中间状态下缺乏足够证据进行判断，因此提出仅在智能体认为任务完成的节点才调用评估器的探测与恢复策略。

    

    测试时搜索可以通过探索替代行动并使用评估器（critic）选择有前景的分支来提升智能体性能。我们在基于终端的计算机使用环境中研究这一方法，这类环境仅能暴露底层大型状态的部分视图。我们发现，所评估的蒙特卡洛树搜索（MCTS）配置并不能显著提高任务成功率。我们识别出评估器表现不佳的两个来源：智能体的观察遗漏了评估所需的环境状态，以及中间状态对轨迹能否成功提供的证据有限。在中间状态下，评估器对后续行动的排序仅略好于随机猜测，而赋予评估器环境访问权限对改善这一点帮助甚微。我们研究了何时调用评估器以及应给予其何种访问权限来评估节点。我们的发现启发了一种策略：仅在智能体认为节点为最终状态时才调用评估器，而不是在部分进展阶段调用。

    arXiv:2610.21088v1 Announce Type: new  Abstract: Test-time search can improve agent performance by exploring alternative actions and using a critic to select promising branches. We study this approach in terminal-based computer-use environments, which expose only a partial view of a large underlying state, and find that the Monte Carlo tree search (MCTS) configuration we evaluate does not significantly improve task success. We identify two sources of poor critic performance: the agent's observations omit environment state needed for evaluation, and intermediate states offer limited evidence of whether a trajectory will succeed. At intermediate states, critics rank continuations only modestly better than chance, and environment access improves this little. We study when to invoke a critic and what access to give it to evaluate a node. Our findings motivate a policy that invokes the critic only at nodes the agent believes are final, rather than during partial progress. Further, the polic
    
[^201]: SIEVE：克服等变自监督学习中的拓扑障碍

    SIEVE: Overcoming Topological Obstruction in Equivariant Self-Supervised Learning

    [https://arxiv.org/abs/2610.21087](https://arxiv.org/abs/2610.21087)

    该论文发现3D点云掩码自编码与SE(3)等变网络间存在拓扑坍塌障碍，并提出SIEVE框架，利用球面投影掩码将点云几何压缩为不变标量嵌入，从构造上克服该障碍并在下游任务尤其是跨类别泛化上取得显著提升。

    

    我们揭示了3D点云掩码自编码与SE(3)等变网络之间的理论不兼容性。我们证明标准掩码会引发拓扑坍塌，实际上迫使模型在没有参考坐标系的情况下凭空猜测朝向。为解决这一问题，我们提出了SIEVE（Scalar Invariant Extraction Via Equivariance，基于等变性的标量不变量提取），这是一个SE(3)等变自监督框架，可将点云几何压缩为不变的标量嵌入。其核心是球面投影掩码，它在保留全局姿态的同时模糊局部语义，从构造上绕开了这一障碍。实验证实，SIEVE在常量掩码和噪声掩码失效的情况下成功规避了所预测的坍塌。在下游点云任务上，所得嵌入优于基于坐标的基线方法和手工设计的几何描述符，其中在对未见形状类别的泛化上提升最为显著。

    arXiv:2610.21087v1 Announce Type: cross  Abstract: We identify a theoretical incompatibility between 3D point cloud masked autoencoding and SE(3)-equivariant networks. We prove that standard masking triggers a topological collapse, effectively forcing the model to hallucinate orientation without a reference frame. To resolve this, we introduce SIEVE (Scalar Invariant Extraction Via Equivariance), an SE(3)-equivariant self-supervised framework that compresses point cloud geometry into invariant scalar embeddings. At its core is the spherical projection mask, which preserves global pose while obfuscating local semantics, sidestepping the obstruction by construction. Experiments confirm that SIEVE circumvents the predicted collapse where constant and noise masks fail. On downstream point cloud tasks, the resulting embeddings improve over coordinate-based baselines and hand-crafted geometric descriptors, with the largest gains on generalization to unseen shape categories.
    
[^202]: StochLOB：用于稳健限价订单簿趋势预测的重尾分数匹配

    StochLOB: Heavy-Tailed Score Matching for Robust Limit Order Book Trend Prediction

    [https://arxiv.org/abs/2610.21081](https://arxiv.org/abs/2610.21081)

    该论文提出StochLOB训练框架，用跳跃Lévy噪声和截断α稳定噪声等重尾非高斯扩散过程替代传统的布朗高斯噪声来加噪训练数据，以更好地捕捉限价订单簿订单流的重尾与跳跃特性，从而提升LOB趋势预测的稳健性，同时推理阶段保持确定性预测器。

    

    基于分数的扩散方法通常使用布朗高斯噪声对数据进行扰动，这对应于轻尾的前向随机微分方程（SDE）。对于市场时间序列而言，这种SDE可能是一种糟糕的抽象：限价订单簿（LOB）的订单流具有重尾性、突发性，且偶尔呈现跳跃状。我们研究了STOCHLOB，这是一个仅用于训练阶段的分数匹配框架，它改变了前向扩散的损坏规律，同时保持部署的预测器为确定性模型。该加噪过程不再仅使用布朗高斯损坏，而是采用两种受非高斯SDE启发的替代方案：通过复合泊松方差混合实现的跳跃Lévy噪声，以及通过正稳定高斯尺度混合实现的截断α稳定噪声。这些损坏机制在训练中注入了有限跳跃和重尾残差，而无需在推理阶段进行反向时间扩散采样。所得的分数目标具有一个有用的解释：其径向因子h（摘要在此处截断）

    arXiv:2610.21081v1 Announce Type: new  Abstract: Score-based diffusion methods usually perturb data with Brownian Gaussian noise, corresponding to a light-tailed forward stochastic differential equation (SDE). For market time series this SDE can be a poor abstraction: limit order book (LOB) order flow is heavy-tailed, bursty, and occasionally jump-like. We study STOCHLOB, a training-only score-matching framework that changes the forward diffusion corruption law while leaving the deployed predictor deterministic. Instead of Brownian-only Gaussian corruption, the noising process uses two non-Gaussian SDE-inspired alternatives: jump-L\'evy noise through a compound-Poisson variance mixture, and clipped $\alpha$-stable noise through a positive-stable Gaussian scale mixture. These corruptions inject finite jumps and heavy-tailed residuals into training without requiring reverse-time diffusion sampling at inference. The resulting score target has a useful interpretation: its radial factor $h_
    
[^203]: 面向偏微分方程的自适应随机神经网络：结构感知特征重分布与残差驱动的块增强

    Adaptive Randomized Neural Networks for PDEs: Structure-Aware Feature Redistribution and Residual-Driven Block Enrichment

    [https://arxiv.org/abs/2610.21077](https://arxiv.org/abs/2610.21077)

    该论文提出两阶段自适应随机神经网络方法SFRBE-RaNN，通过结构感知特征重分布（SFR）与残差驱动的块增强（RBE）自适应构建试探空间，以更有效捕捉偏微分方程解的空间与尺度结构，并为固定线性问题提供精确的残差缩减恒等式与误差估计。

    

    随机神经网络保留了输出线性的最小二乘求解方式，但全局采样的随机特征可能无法捕捉偏微分方程解的空间与尺度结构。我们提出SFRBE-RaNN，一种两阶段自适应试探空间构建方法。SFR（结构感知特征重分布）根据区域指示质量重新分配固定的特征预算，并通过低维的基于残差的优化来调整区域权重尺度。RBE（残差驱动的块增强）通过筛选多个激活函数族、移除已被表示的响应方向并选择最小的可接受前缀来构建非冗余的增强特征。对于依赖于解的算子，候选特征先由冻结算子进行筛选，仅在通过非线性残差验证后才被接受。对于固定线性问题，我们建立了精确的离散残差缩减恒等式和条件解误差估计。基于这一恒等式，我们将带保护机制的SFR与RBE捕获因子相结合，推导出……（原文摘要在此处截断）

    arXiv:2610.21077v1 Announce Type: cross  Abstract: Randomized neural networks retain an output-linear least-squares solve, but globally sampled random features may fail to capture the spatial and scale structure of PDE solutions. We propose SFRBE--RaNN, a two-stage adaptive trial-space construction. SFR redistributes a fixed feature budget according to regional indicator masses and adapts regional weight scales through low-dimensional residual-based optimization. RBE constructs nonredundant enrichments by screening multiple activation families, removing represented response directions, and selecting the smallest admissible prefix. For solution-dependent operators, candidates are screened with a frozen operator and accepted only after nonlinear residual verification. For fixed linear problems, we establish an exact discrete residual-reduction identity and a conditional solution-error estimate. Building on this identity, we combine safeguarded SFR with RBE capture factors to derive a fin
    
[^204]: 零样本条件离散扩散中的分数失配与集合约束采样

    Score Mismatch and Set-Constrained Sampling in Zero-Shot Conditional Discrete Diffusion

    [https://arxiv.org/abs/2610.21061](https://arxiv.org/abs/2610.21061)

    本文首次对分数失配的离散扩散模型进行了理论分析，推导出一般的KL散度上界来量化失配造成的额外偏差，并覆盖文本填充、句子补全等部分观测约束场景，为零样本条件生成提供了理论保证。

    

    离散扩散模型已成为在众多应用中建模复杂离散分布的强大框架。尽管理论收敛保证依赖于目标分布与训练分布之间的一致性，但这种情况在实践中很少成立，这使得分数失配成为离散扩散采样中的核心挑战。这一问题在零样本条件生成中尤为突出，即在推理阶段直接将在无条件分布上训练的模型重用于从条件目标分布中采样。本文对分数失配的离散扩散模型提供了首次理论分析，并推导出一般的KL散度上界，该上界清晰地刻画了由这种失配引起的额外偏差。我们的分析涵盖了部分观测约束，包括文本填充、句子补全以及更一般的逐坐标集合约束。

    arXiv:2610.21061v1 Announce Type: cross  Abstract: Discrete diffusion models have become a powerful framework for modeling complex discrete distributions across a range of applications. While theoretical convergence guarantees depend on consistency between the target and training distributions, such a case rarely holds in practice, making score mismatch a central challenge in discrete diffusion sampling. This is particularly relevant to zero-shot conditional generation, where a model trained on the unconditional distribution is directly reused at inference time to sample from a conditional target distribution. In this paper, we provide the first theoretical analysis for score-mismatched discrete diffusion models and derive a general KL divergence upper bound that clearly identifies the additional bias caused by such a mismatch. Our analysis covers partial-observation constraints, including text infilling, sentence completion, and more general coordinatewise set constraints. Specificall
    
[^205]: 面向大语言模型算术推理的贝尔曼一致性轨迹强化学习

    Bellman-Consistent Trajectory RL for LLM Arithmetic Reasoning

    [https://arxiv.org/abs/2610.21060](https://arxiv.org/abs/2610.21060)

    提出贝尔曼一致的轨迹强化学习方法TBRM-RLOO，通过轨迹贝尔曼残差最小化将稀疏奖励转化为密集词元级信号并降低梯度方差，在Countdown算术推理任务上达到最先进的76.0–84.0% Pass@16准确率，显著超越GRPO、SFT、IPO/DPO和RLOO等基线。

    

    为了解决大语言模型算术推理中稀疏终端奖励的问题，我们提出了一种贝尔曼一致的轨迹强化学习方法。我们的框架引入了轨迹贝尔曼残差最小化（TBRM），将稀疏奖励转化为密集的词元级信号。我们集成了留一法强化（RLOO）分组策略来降低贝尔曼残差的梯度方差，并使用事后目标重标注来加速中间学习过程。在Countdown任务上的评估表明，我们的方法在缓解优化瓶颈和稳定梯度方差方面显著优于GRPO。TBRM-RLOO框架将Pass@1提升至0.55–0.64，并实现了76.0–84.0%的最先进Pass@16准确率，优于SFT、IPO/DPO和RLOO等基线方法。我们的方法展现了卓越的单样本效率，并通过测试时扩展成功规避了标准的性能平台期。

    arXiv:2610.21060v1 Announce Type: cross  Abstract: To address sparse terminal rewards in LLM arithmetic reasoning, we propose a Bellman-consistent trajectory RL method. Our framework introduces Trajectory Bellman Residual Minimization (TBRM) to transform sparse rewards into dense token-level signals. We integrate a Reinforce Leave-One-Out (RLOO) group strategy to reduce gradient variance of Bellman residuals and use hindsight goal relabeling to accelerate intermediate learning. Evaluated on the Countdown task, our method alleviates optimization bottlenecks and stabilizes gradient variance significantly better than GRPO. The TBRM-RLOO framework boosts Pass@1 to 0.55--0.64 and achieves a state-of-the-art Pass@16 accuracy of 76.0--84.0\%, outperforming the SFT, IPO/DPO, and RLOO baselines. Our approach demonstrates exceptional single-sample efficiency and successfully circumvents standard performance plateaus via test-time scaling.
    
[^206]: Athanor：一个面向自主后训练研究的智能体原生系统

    Athanor: An Agent-Native System for Autonomous Post-Training Research

    [https://arxiv.org/abs/2610.21055](https://arxiv.org/abs/2610.21055)

    Athanor是一个开放的智能体原生系统，赋予AI智能体完全的研究自主权，让它们在真实GPU上自主完成小型基础模型的后训练研究，以此检验智能体迈向递归自我改进的研究潜力。

    

    递归自我改进，即AI系统构建更强大的AI系统，是通往超级智能的一条有前景的路径。我们研究了一种如今即可测量的较弱形式：前沿智能体使用公开数据对一个小型原始基础模型进行后训练，以此作为对其研究潜力的测试。我们构建了Athanor，一个运行在真实GPU上的开放式智能体原生自主后训练研究系统（AutoPostTrain）。我们为智能体提供了完全的研究自主权、低摩擦的工程工具以及高效的起始代码。智能体自行开展研究：它们阅读数据集和评估结果，构建训练数据，编写并修改训练算法，调整超参数，并分析结果以决定下一步的尝试。它们在沙箱环境中工作，且无法使用外部教师模型。Athanor的服务提供冻结评估、托管任务、去污染数据、持久化研究记录，以及一套智能体不知情的保留测试套件。

    arXiv:2610.21055v1 Announce Type: new  Abstract: Recursive self-improvement, in which AI systems build stronger AI systems, is one promising route to superintelligence. We study a weaker form that can be measured today: frontier agents post-training a small raw base model with public data, as a test of their research potential. We build Athanor, an open agent-native system for autonomous post-training research (AutoPostTrain) on real GPUs. We give the agents full research autonomy, low-friction engineering tools, and efficient starter code. The agents carry out the research themselves: they read datasets and evaluation results, build their training data, write and modify training algorithms, tune hyperparameters, and analyze the results to decide what to try next. They work in a sandbox and cannot use external teacher models. Athanor's services supply frozen evaluation, managed jobs, decontaminated data, a persistent research record, and a held-out suite the agents are unaware of. In a
    
[^207]: SALTO：用于非线性事件推断的求解器锚定提升转移算子

    SALTO: Solver-Anchored Lifted Transfer Operators for Nonlinear Event Inference

    [https://arxiv.org/abs/2610.21049](https://arxiv.org/abs/2610.21049)

    SALTO是一个基于Koopman算子的框架，通过在提升空间中构建可复用的线性化伴随算子，仅需一次编码器前向传播和一个内积即可快速推断任意有限幅度的局部非线性事件源。

    

    学习到的动力学模型能够准确预测轨迹，但在求导或转置时产生的敏感性却不可靠，这限制了其在逆问题和科学决策中的应用。经典伴随方法能提供精确的局部导数，但单个伴随算子无法表示系统对一族有限幅度扰动的非线性响应。我们提出了SALTO（求解器锚定提升转移算子），这是一个基于Koopman算子的框架，用于推断有限幅度扰动源。在提升空间中，非线性动力学由一个线性传播子来近似，其转置能够在单次求解中将每个传感器信号反向传播。由此得到一个可复用的提升伴随算子，仅需一次编码器前向传播和一个内积运算，即可预测任意候选局部事件所产生的测量值。为了让模型获得模拟器的精确局部行为，SALTO施加了三个结构约束：它将物理态动力学锚定（摘要在此处被截断）

    arXiv:2610.21049v1 Announce Type: new  Abstract: Learned dynamical models can accurately predict trajectories while producing unreliable sensitivities when differentiated or transposed, limiting their use in inverse problems and scientific decision-making. Classical adjoints provide exact local derivatives, but a single adjoint cannot represent the nonlinear response of a system to a family of finite-amplitude perturbations. We introduce SALTO (Solver-Anchored Lifted Transfer Operators), a Koopman-based framework for inferring finite-amplitude sources. In the lifted space, the nonlinear dynamics are approximated by a linear propagator, and its transpose carries each sensor backward in a single solve. The result is a reusable lifted adjoint that predicts the measurement produced by any candidate localized event with one encoder pass and an inner product. To give the model the simulator's exact local behavior, SALTO imposes three structural constraints: it anchors the physical-state dyna
    
[^208]: SHIP：具有隐私保护的结构化与分层洞察

    SHIP: Structured and Hierarchical Insight with Privacy

    [https://arxiv.org/abs/2610.21048](https://arxiv.org/abs/2610.21048)

    该论文提出 SHIP 框架，将差分隐私洞察形式化为可重用表示，通过自动查询-属性映射、分层混合分类体系和差分隐私合成表格，在保护隐私的同时有效回答细粒度的分析查询，并建立了统一的效用评估协议。

    

    从文本数据中提取洞察同时保护用户隐私正变得日益重要。现有方法为不同的分析目的生成形式各异的隐私洞察，使其效用难以评估和比较。在这项工作中，我们将差分隐私（DP）洞察形式化为用于回答分析查询的可重用表示，为效用评估提供了共同基础。然而，回答这些查询需要细粒度的细节和跨属性的统计信息，而现有表示可能会遗漏这些内容。为了捕获这些信息，我们提出了结构化与分层隐私洞察（SHIP），它能够自动将查询映射到属性、构建分层的混合分类体系，并发布差分隐私合成表格以供进一步分析。我们还开发了一个主要基于问答性能和结构化保真度的评估协议。我们的实验表明，SHIP 在相关任务上表现出了强大的性能。

    arXiv:2610.21048v1 Announce Type: new  Abstract: Extracting insights from textual data while protecting user privacy is increasingly important. Existing methods produce private insights in diverse forms for different analytical purposes, making their utility difficult to assess and compare. In this work, we formulate differentially private (DP) insights as reusable representations for answering analytical queries, providing a common basis for utility assessment. Answering these queries, however, requires fine-grained details and cross-attribute statistics that existing representations may omit. To capture this information, we introduce Structured and Hierarchical Insight with Privacy (SHIP), which automatically maps queries to attributes, builds a hierarchical hybrid taxonomy, and releases a DP synthetic table for further analysis. We also develop an evaluation protocol based primarily on QA performance and structured fidelity. Our experiments demonstrate SHIP's strong performance on b
    
[^209]: PremiseCheck：面向人类与AI同行评审的高效事实核查

    PremiseCheck: Efficient Fact-Checking of Human and AI Peer Reviews

    [https://arxiv.org/abs/2610.21046](https://arxiv.org/abs/2610.21046)

    PremiseCheck通过让LLM在不读论文的情况下将评审意见拆解为显式前提、再用廉价判定模型逐一与论文核对，在保持与前沿LLM相当的核查准确率的同时将成本降低至约一半（NLP领域仅为四分之一）。

    

    同行评审正因投稿量的增长而承受巨大压力，无论是人类还是AI撰写的评审意见都可能错误地陈述论文的实际内容。我们提出了ReviewFact，这是一个在统一定义下对NLP、机器学习和自然科学领域评审意见中的事实性错误进行标注的基准，并同时衡量准确率、成本和延迟。前沿大语言模型能够很好地检查评审意见，但每次检查都必须通读整篇论文；而廉价的判定模型（用于判断某陈述在文本中是否成立）在处理混合了多个隐含断言的完整评审意见时表现远为逊色。PremiseCheck是一个将两者结合的框架：由一个LLM在不阅读论文的情况下将每条评审意见改写为显式前提，再由判定模型将每个前提与论文逐一核对。当每条评审意见被单独核查时，PremiseCheck在全部三个领域达到与GPT-5.5相当的准确率，而成本仅为其一半左右；在NLP评审上，成本更是仅为四分之一。

    arXiv:2610.21046v1 Announce Type: cross  Abstract: Peer review is under strain from the growth of submissions, and review comments written by humans or AI can misstate what a paper says. We introduce ReviewFact, a benchmark that labels such factual errors in review comments under one definition across NLP, machine learning, and the natural sciences, and that measures cost and latency alongside accuracy. A frontier LLM checks comments well but must read the whole paper for each check, while cheap decision models, which score whether a statement holds in a text, fall far short on whole comments that mix several implicit assertions. PremiseCheck is a framework that combines the two. An LLM rewrites each comment as explicit premises without reading the paper, and a decision model checks each premise against the paper. When each comment is checked on its own, PremiseCheck reaches accuracy comparable to GPT-5.5 across all three fields at about half its cost, and on NLP reviews at a quarter o
    
[^210]: INCONVENIENT：一个评估大语言模型从便利样本中进行泛化能力的基准

    INCONVENIENT: A Benchmark to Assess LLM Generalization Capabilities from Convenience Samples

    [https://arxiv.org/abs/2610.21038](https://arxiv.org/abs/2610.21038)

    提出了INCONVENIENT基准，通过显式构造并隔离便利样本偏差，评估模型在有偏采样的表格数据上向目标总体泛化的能力。

    

    用于表格预测的机器学习模型通常假设训练数据能够代表其将要部署的目标人群。然而在实践中，数据往往出于便利性、可获得性或成本的考虑而收集，由此产生的系统性抽样偏差可能扭曲特征与目标变量之间的关系，并损害模型在真实世界中的泛化能力。现有的表格数据分布偏移基准主要通过在自然存在的域之间划分观测样本来构建分布外（OOD）设置。虽然这种划分可能引起 P(Y | X) 的变化，但由此产生的偏移既未被显式控制也未被隔离，而且评估分布本身往往无法代表更广泛的目标人群。我们提出了 INCONVENIENT，一个旨在研究模型对便利样本偏差鲁棒性的基准。INCONVENIENT 不依赖自然存在的域划分，而是直接……

    arXiv:2610.21038v1 Announce Type: cross  Abstract: Machine learning models for tabular prediction typically assume that training data are representative of the population on which they will be deployed. In practice, however, data are often collected based on convenience, availability, or cost, resulting in systematic sampling biases that can distort feature--target relationships and undermine real-world generalization. Existing tabular distribution-shift benchmarks primarily construct out-of-distribution (OOD) settings by partitioning observations across naturally occurring domains. While such partitions may induce changes in P(Y | X), the resulting shifts are neither explicitly controlled nor isolated, and the evaluation distributions themselves are often not representative of the broader target population. We introduce INCONVENIENT, a benchmark designed to study robustness to convenience-sample bias. Rather than relying on naturally occurring domain partitions, INCONVENIENT directly 
    
[^211]: 局部差分隐私下单变量密度的高效不可知学习

    Efficient Agnostic Learning of Univariate Densities Under Local Differential Privacy

    [https://arxiv.org/abs/2610.21033](https://arxiv.org/abs/2610.21033)

    本文提出了一个非交互式局部差分隐私学习框架，实现了对可被分段多项式逼近的单变量分布（如高斯混合模型）首个样本高效且计算高效的不可知学习，并为此开发了可独立使用的大域LDP合成数据生成算法。

    

    我们提出了一个用于非交互式局部差分隐私（LDP）学习的通用框架，适用于可由分段多项式逼近的结构化单变量分布。这为若干分布类提供了首个样本高效的LDP学习器，包括对数凹分布的混合分布，例如高斯混合模型（GMMs）。这些学习器还提供不可知保证，并且在计算上是高效的。我们还证明了在LDP下学习GMM的一个下界，改进了Bun等人[BKSW19]的标准纯DP装箱下界。为了构建我们的通用学习框架，我们开发了一种针对大域上合成数据生成的高效非交互式LDP算法，该算法可能具有独立的研究价值。随后，我们将其与Acharya等人[ADLS17]的非私有分段多项式学习器的确定性改编版本结合使用。

    arXiv:2610.21033v1 Announce Type: cross  Abstract: We propose a general framework for non-interactive locally differentially private (LDP) learning of structured univariate distributions that can be approximated by piece-wise polynomials. This gives the first sample-efficient LDP learners for several distribution classes, including mixtures of log-concave distributions, such as Gaussian mixture models (GMMs). These learners also provide agnostic guarantees and are computationally efficient. We also prove a lower bound for learning GMMs under LDP, improving on the standard pure-DP packing lower bounds of Bun et al. [BKSW19]. To obtain our general learning framework, we develop an efficient non-interactive LDP algorithm for synthetic data generation over large domains, which may be of independent interest. We then use it together with a deterministic adaptation of the non-private piecewise polynomial learner of Acharya et al. [ADLS17].
    
[^212]: 论裁剪梯度方法的末次迭代收敛性

    On the Last-Iterate Convergence of Clipped Gradient Methods

    [https://arxiv.org/abs/2610.21026](https://arxiv.org/abs/2610.21026)

    本文针对梯度噪声为重尾分布这一更贴近实际的情形，研究并证明了裁剪梯度方法（如Clipped-SGD）在随机（强）凸优化中的末次迭代收敛性，克服了现有SGD末次迭代收敛结果依赖强噪声条件（如指数衰减尾部）的局限。

    

    在随机（强）凸优化中，为一阶方法建立末次迭代收敛保证近来受到越来越多的关注，因为这不仅加深了对这些算法的理论理解，而且与实践更为贴近。值得注意的是，人们已经证明，最著名、简单且流行的随机梯度下降（SGD）方法能够可证明地实现末次迭代收敛。然而，现有结果的适用性可能有限，因为它们通常依赖于对梯度噪声的较强条件（例如指数衰减的尾部），而非更现实的重尾噪声。不幸的是，面对重尾噪声时，已知SGD会表现出不良行为甚至无法收敛。为应对重尾噪声的挑战，人们提出了Clipped-SGD，这是一种将……（原文摘要在此处截断）

    arXiv:2610.21026v1 Announce Type: cross  Abstract: In stochastic (strongly) convex optimization, establishing last-iterate convergence guarantees for first-order methods has recently garnered increasing attention, as this not only deepens the theoretical understanding of these algorithms but also aligns more closely with practice. Notably, people have demonstrated that one of the most famous, simple, and popular methods, Stochastic Gradient Descent ($\mathtt{SGD}$), provably achieves last-iterate convergence. However, the existing results may have limited applicability, since they typically rely on strong conditions on the gradient noise (e.g., an exponentially decaying tail) rather than the more realistic heavy-tailed noise. Unfortunately, when faced with heavy-tailed noise, $\mathtt{SGD}$ is known to exhibit undesirable behavior or even fail to converge. To address the challenge of heavy-tailed noise, people have proposed $\mathtt{Clipped}\text{-}\mathtt{SGD}$, an algorithm that comb
    
[^213]: 学会读出：语言模型预训练中的反嵌入动态

    Learning to Read Out: Unembedding Dynamics in Language Model Pretraining

    [https://arxiv.org/abs/2610.21025](https://arxiv.org/abs/2610.21025)

    提出参数轨迹交叉编码方法，仅利用检查点权重直接追踪语言模型预训练中读出层的动态，发现Pythia在预训练0.4%–1.1%的早期窗口内读出层发生重组，且隐藏状态在读出层表达之前就已编码任务信息。

    

    语言模型是在何时获得某种能力的？发展性研究通常通过追踪模型何时开始生成正确的词元来回答这个问题。但这一预测需要经过反嵌入矩阵（即读出层），而读出层本身在预训练过程中也仍在被学习。我们证明，模型的隐藏状态可以在其自身的读出层表达任务信息之前就已编码该信息，因此从输出中读取的时间线混淆了模型所编码的内容与读出层所表达的内容。为了直接追踪读出层，我们引入了参数轨迹交叉编码方法，该方法在每个检查点为每个词元的读出行拟合一个稀疏字典。与作用于激活值的交叉编码器不同，它只需要检查点的权重。我们利用该方法定位到一个狭窄的早期窗口——即Pythia预训练进程的0.4%–1.1%阶段——在此期间读出层发生重组，WordNet词类变得可以从其行中解码。到那时，对隐藏状态的线性探针已经能够解码

    arXiv:2610.21025v1 Announce Type: cross  Abstract: When does a language model acquire a capability? Developmental studies usually answer this by tracking when models start producing the correct token. But that prediction passes through the unembedding matrix, or readout, which is itself still being learned during pretraining. We show that a model's hidden states can encode task information before its own readout expresses it, so timelines read from outputs conflate what the model encodes with what its readout expresses. To track the readout directly, we introduce parameter-trajectory crosscoding, which fits one sparse dictionary to every token's readout row at every checkpoint. Unlike crosscoders on activations, it needs only checkpoint weights. We use it to locate a narrow early window, 0.4-1.1% of the way through Pythia's pretraining, in which the readout reorganizes and WordNet word categories become decodable from its rows. By then, linear probes on the hidden states already decode
    
[^214]: 高斯变分推断中的加速及其极限

    Acceleration and its Limits in Gaussian Variational Inference

    [https://arxiv.org/abs/2610.21020](https://arxiv.org/abs/2610.21020)

    该论文证明高斯变分推断在可精确计算高斯期望时能达到经典的 √κ 加速收敛率，但通过构造一族强对数凹目标证明仅有逐点导数时任何算法都需要 κ^(1-o(1)) 次查询，从而说明仅凭曲率界不足以保证加速。

    

    我们研究了经典的基于动量的加速方法——对于强凸光滑目标，它能将梯度下降对条件数的依赖从 κ 改进到 √κ——能否推广到针对强对数凹、对数光滑目标分布的高斯变分推断（GVI）。在可以精确计算高斯期望的情况下，GVI 的凸复合形式能够达到经典的 √κ 加速收敛率。然而，当仅有逐点导数可用时，我们构造了一族强对数凹的目标分布，证明任何随机化算法都需要 κ^(1-o(1)) 次查询才能达到固定的目标精度，即使初始目标差距与 κ 无关地有界也是如此。因此，仅凭曲率界无法保证一致加速的总查询复杂度。我们随后探讨何种附加结构可以允许加速，并给出了正例：在 Hessian-Lipschitz 假设下……

    arXiv:2610.21020v1 Announce Type: cross  Abstract: We study whether the classical momentum-based acceleration that improves the condition-number dependence of gradient descent from $\kappa$ to $\sqrt{\kappa}$ for strongly convex, smooth objectives extends to Gaussian variational inference (GVI) for strongly log-concave, log-smooth target distributions. With exact Gaussian expectations, a convex composite formulation of GVI yields the classical accelerated $\sqrt{\kappa}$ convergence rate. However, when only pointwise derivatives are available, we construct a family of strongly log-concave targets for which any randomized algorithm requires $\kappa^{1-o(1)}$ queries to achieve a fixed objective accuracy, even when the initial objective gap is bounded independently of $\kappa$. Thus curvature bounds alone cannot guarantee uniform accelerated total-query complexity. We then ask which additional structure permits acceleration and give positive examples: under a Hessian-Lipschitz assumption
    
[^215]: CausalEvo：面向健康LLM智能体的因果技能自进化

    CausalEvo: Causal Skill Self-Evolution for Health LLM Agents

    [https://arxiv.org/abs/2610.21016](https://arxiv.org/abs/2610.21016)

    提出CausalEvo方法，通过在固定模型权重、反馈案例和规则数量的条件下估计技能集变更的相对因果效应，实现健康LLM智能体的技能自进化筛选，保留有效规则并防止错误泛化的传播。

    

    面向健康领域的大语言模型（LLM）智能体可以从任务反馈中积累可复用的文本技能，例如计算标准、测量时间点和公式单位。决定保留哪些技能至关重要：有用的规则能够改善后续回答，而错误的泛化则会传播错误。我们提出了CausalEvo，一种因果技能自进化方法，它在保持模型权重、反馈案例和规则数量固定不变的情况下，估计经过刻意变更的技能集的相对效果。在每个进化步骤中，一个冻结的LLM提出候选规则，这些候选规则与现有技能汇集在一起，使当前技能集始终保持为可行选项。针对共享背景规则的成对技能替换在相同的反馈案例上进行评分；最小二乘法将评分差异转换为相对技能价值，排名靠前的规则被保留，平局时优先保留现有技能。一个决策损失界将因果……

    arXiv:2610.21016v1 Announce Type: new  Abstract: Large language model (LLM) agents for health can accumulate reusable textual skills from task feedback, such as calculation criteria, measurement time points, and formula units. Deciding which skills to retain is consequential: useful rules improve later answers, while faulty generalizations propagate errors. We propose CausalEvo, a method for causal skill self-evolution that estimates the relative effects of deliberately changed skill sets while holding the model weights, feedback cases, and rule count fixed. At each evolution step, a frozen LLM proposes candidate rules, which are pooled with the incumbent skills so that the current skill set remains a feasible choice. Paired skill replacements against shared background rules are scored on the same feedback cases; least squares converts the score differences into relative skill values, and the top-ranked rules are retained, favoring incumbents on ties. A decision-loss bound separates ca
    
[^216]: 真实世界视频中的视听犬只识别与追踪

    Audio-Visual Dog Identification and Tracking in Real-World Videos

    [https://arxiv.org/abs/2610.21011](https://arxiv.org/abs/2610.21011)

    提出了一种结合视觉识别、多目标追踪和犬类声纹识别的多模态犬只指纹框架，可在真实世界视频中从吠叫声中识别并追踪个体犬只，并为其维护持续更新的身份档案。

    

    在真实世界视频中通过吠叫声识别个体犬只仍然是一个充满挑战且 largely 未被探索的问题。解决这一问题将能够通过把发声和行为归因于正确的个体，实现长期行为分析、兽医监测以及动物福利监测。真实世界视频的挑战在于：多只狗可能同时吠叫、被遮挡，或进出视野。以往研究表明，犬类的发声和视觉外观包含可用于区分个体身份的信息，但这些模态大多被独立研究。我们提出了一种多模态犬只指纹框架，结合视觉识别、多目标追踪和犬类声纹识别，在维护持续性档案的同时，将吠叫事件与个体犬只关联起来。该框架将吠叫事件与被追踪的犬只同步，并利用视觉外观和声音特征更新档案。

    arXiv:2610.21011v1 Announce Type: cross  Abstract: Identifying individual dogs from barks in real-world videos remains challenging and largely unexplored. Solving this problem would enable long-term behavioral analysis, veterinary monitoring, and animal welfare monitoring by attributing vocalizations and behaviors to the correct individual. Real-world videos are challenging because multiple dogs may bark simultaneously, become occluded, or move in and out of view. Prior studies show that canine vocalizations and visual appearance contain discriminative identity information, but these modalities have largely been studied independently. We present a multimodal dog-fingerprinting framework combining visual identification, multiple-object tracking, and canine vocal fingerprinting to associate bark events with individual dogs while maintaining persistent profiles. The framework synchronizes bark events with tracked dogs and updates profiles using visual appearance and vocal characteristics,
    
[^217]: 大语言模型能否预判其关键症结？

    Can LLMs Anticipate Their Cruxes?

    [https://arxiv.org/abs/2610.21006](https://arxiv.org/abs/2610.21006)

    本文提出将模型决策中的“关键症结”定义为模型多次推演分布上产生分歧、且改变其取值可能改变结论的子问题，并比较了三种计算成本递增的策略（先验预测、事后推理提取、多次生成）来识别这些症结。

    

    要让语言模型帮助我们做决策，我们需要理解语言模型输出背后的理据。原则上，语言模型会在其文本输出中提供理据，但文本数量庞大，在实践中难以利用。在本文中，我们感兴趣的是提取导致模型推荐某一行动而非另一行动的关键症结。我们将“症结”定义为模型多次推演之间产生分歧的子问题，即改变其取值可能会改变结论。因此，症结是模型推演分布的属性，而不是任何单次推演的属性。我们考虑了三种计算成本递增的策略：第一，我们仅要求模型先验地预测决策的症结将是什么；第二，我们让模型实际完成生成回复的过程，然后从其推理中提取出现的症结；最后，我们生成多次响应……

    arXiv:2610.21006v1 Announce Type: new  Abstract: To use language models to help us make decisions we would like to understand the rationale behind the language model's outputs. In principle, the language model provides its rationale as part of its text output. However, the sheer volume of text makes this hard to use in practice. In this paper, we are interested in extracting the cruxes that lead a model to recommend one action over another. We take a crux to be a sub-question that a model's rollouts disagree on, where changing its value may have altered the conclusion. A crux is therefore a property of the distribution of a model's rollouts, not of any one rollout. We consider three strategies of increasing computational cost. First, we just ask a model to identify, a priori, what the cruxes of a decision will be. Second, we have the model actually go through the process of generating a response and then extract the cruxes that arose in its reasoning. Finally, we generate many response
    
[^218]: 不完美即完美：面向分支定价切平面算法的学习式列预填充

    Imperfect Is Perfect: Learned Column Prefilling for Branch-Price-and-Cut

    [https://arxiv.org/abs/2610.21001](https://arxiv.org/abs/2610.21001)

    该论文揭示了“恰当的不完美”这一反直觉发现——用神经网络在数秒内生成的不完美普通列（而非最优列）作为预填充，并预测Wentges平滑初始中心，反而能显著加速分支定价切平面算法的根节点求解。

    

    列生成领域的经验之谈告诫说，初始列会带来危害：热启动可能使对偶变量产生偏差，从而拖慢分支定价切平面算法根节点的求解。我们证明，一旦这些列以恰当的方式保持“不完美”，情况恰恰相反。预加载最优根节点LP的支撑集——即人们所能预先知道的最好列——在我们最难的实例中有一半无法完成求解；而一组不必属于任何最优解的普通路径，存放于主问题之外并按当前平滑对偶进行定价，却能加速根节点的求解。由于这些列无需最优，生成它们便成为一个学习任务：一个神经边评分器引导构造式生成器，在几秒钟内构建该列池。第二个网络从实例文件预测Wentges平滑的初始中心，并由此确定平滑定价查询的方向；而求解器自身的更新过程保持不变。预填充与中心就位后，精确定价主导了根节点定价时间，因此我们重建bu（原文在此截断）

    arXiv:2610.21001v1 Announce Type: new  Abstract: Column generation folklore warns that initial columns hurt: a warm start may bias the dual and slow the root of branch-price-and-cut. We show the opposite once the columns are imperfect in the right way. Preloading the support of the optimal root LP, the best columns one could know in advance, fails to finish on half of our hardest instances; a pool of ordinary routes that need not belong to any optimal solution, held outside the master and priced at the current smoothed dual, speeds the root up. Because the columns need not be optimal, producing them becomes a learning task: a neural edge scorer guides a constructive generator that builds the pool in seconds. A second network predicts the initial center of Wentges smoothing from the instance file, and with it the direction of the smoothed pricing queries; the solver's own updates run unchanged. With prefill and center in place, exact pricing dominates root pricing time, so we rebuild bu
    
[^219]: EigDiff：特征样本扩散——或如何在性能几乎不变的情况下显著缩减训练数据集规模？

    EigDiff: Eigensample Diffusion - or how to significantly reduce training dataset size with marginal performance variation?

    [https://arxiv.org/abs/2610.20999](https://arxiv.org/abs/2610.20999)

    提出EigDiff方法，在预训练CLIP模型的潜在空间中执行PCA获取特征向量，并以此为指导选取由样本代表与多样化图像构成的代表性训练子集，从而在性能几乎不变的前提下大幅缩减扩散模型的训练数据集规模。

    

    扩散模型的训练成本极高，需要大规模数据集来学习如何在多个小的去噪步骤中逐步还原图像。当前的方法主要聚焦于蒸馏扩散步数，而忽视了训练数据中潜在的冗余。本文探索了利用特征样本在显著更小的数据集上训练扩散模型的潜力，同时保持模型性能。为了缓解特征图像对仿射变换的敏感性，我们利用预训练CLIP模型的潜在空间。随后，我们通过在CLIP空间中执行主成分分析（PCA）来计算潜在特征向量。进一步地，我们将这些特征向量作为参考点，用以指导选择由样本代表（即嵌入最接近特征向量的图像）和多样化图像组成的代表性训练集。我们通过全面的实验……（原文摘要至此截断）

    arXiv:2610.20999v1 Announce Type: cross  Abstract: Diffusion models are notoriously expensive to train, requiring large-scale datasets to learn how to gradually reverse the noise from images over multiple small denoising steps. Current approaches focus on distilling the number of diffusion steps, neglecting potential redundancies of the training data. In this paper, we explore the potential of harnessing eigensamples to train diffusion models on significantly smaller datasets, while preserving model performance. To mitigate the sensitivity to affine transformations of eigenimages, we harness the latent space of a pre-trained CLIP model. We next compute latent eigenvectors by performing principal component analysis (PCA) in CLIP space. Further, we employ eigenvectors as reference points to guide the selection of a representative training set comprising comprising both exemplars (images whose embeddings are closest to eigenvectors) and diverse images. Our comprehensive experiments with d
    
[^220]: 一个偏置分数，两个因素：缓解方法如何影响多项选择题评估中的选项偏置

    One Bias Score, Two Factors: How Mitigations Affect Option Bias in Multiple-Choice Evaluation

    [https://arxiv.org/abs/2610.20993](https://arxiv.org/abs/2610.20993)

    该论文提出用合并选择率标准差统一选择偏置与顺序敏感性两种视角，并证明该指标可精确分解为翻转率（反映模型不确定性）和位置偏斜（反映选项符号先验）两个无标签因素，从而揭示了现有缓解方法影响选项偏置的内在机制。

    

    重新排列多项选择题的选项可以改变模型的答案，这一效应既被研究为选择偏置（模型对某些选项字母的偏好），也被研究为顺序敏感性（答案随选项排列顺序的变化）。我们通过合并选择率标准差（pooled SD）将这两种视角联系起来，该指标衡量选项字母在不同问题和排列之间被选择的不均匀程度。当每个问题的每个选项在每个位置出现的频率相同时，合并标准差恰好是两个无标签因素的乘积：翻转率和位置偏斜。翻转率是指所选选项在不同排列之间发生变化的问题比例；位置偏斜是指这些问题中选项字母选择率的标准差。在41个模型检查点和12个基准测试上，翻转率与准确率和输出不确定性相关，而位置偏斜与估计的选项符号先验相关。对先验的修正……（摘要原文截断于此）

    arXiv:2610.20993v1 Announce Type: new  Abstract: Reordering multiple-choice options can change a model's answer, an effect studied as selection bias (a preference for certain option letters) and as order sensitivity (changes in answers across option orderings). We connect these perspectives through the pooled selection-rate standard deviation (pooled SD), which measures how unevenly option letters are selected across questions and orderings. When every option of each question appears equally often at each position, the pooled SD is exactly the product of two label-free factors: the flip rate and the positional skew. The flip rate is the fraction of questions whose chosen option changes across orderings; the positional skew is the standard deviation of option-letter selection rates among those questions. Across 41 checkpoints and 12 benchmarks, the flip rate tracks accuracy and output uncertainty, while the positional skew tracks the estimated option-symbol prior. Prior correction for s
    
[^221]: 照我做的做，而不是照我说的做：视觉语言模型中的在线策略蒸馏会继承教师模型

    Do As I Do, Not As I Say: On-Policy Distillation Inherits the Teacher in Vision-Language Models

    [https://arxiv.org/abs/2610.20989](https://arxiv.org/abs/2610.20989)

    该研究发现，在线策略蒸馏（OPD）只有在教师模型对学生自身生成的轨迹保持可靠时才有效——学生会直接继承教师在其所处状态上的行为（包括错误），因此教师对学生的状态分布是否可靠是OPD成败的关键。

    

    提升视觉语言模型（VLM）的推理能力通常依赖于借助更强模型进行监督的后训练。两种常见策略是监督微调（SFT），即在教师生成的示范上进行训练；以及在线策略蒸馏（OPD），即由教师对学生自身生成的轨迹提供反馈。虽然OPD通过监督学生生成的轨迹可以获得更高的数据效率，但教师在每个状态上提供的监督如何影响学生的学习行为尚不清楚。为了研究这一问题，我们在不同的数据预算以及教师表现出不同可靠性水平的任务上，对OPD和过滤式SFT进行了受控比较，并分析这些差异如何塑造学生的准确率和完成行为。在两个任务上，我们发现只有当教师在学生生成的轨迹上保持可靠时，OPD才是有效的。在ChartQA上，当教师……

    arXiv:2610.20989v1 Announce Type: new  Abstract: Improving the reasoning capabilities of vision-language models (VLMs) often relies on post-training with supervision from stronger models. Two common strategies are supervised fine-tuning (SFT), which trains on teacher-generated demonstrations, and on-policy distillation (OPD), which provides teacher feedback on trajectories generated by the student itself. While OPD can be more data-efficient by supervising student-generated trajectories, it is unclear how the teacher's supervision at each state affects the student's learned behavior. To study this question, we conduct a controlled comparison of OPD and filtered SFT across data budgets and tasks where the teacher exhibits different levels of reliability, and analyze how these differences shape student accuracy and completion behavior. Across the two tasks, we find that OPD is effective only when the teacher remains reliable on student-generated trajectories. On ChartQA, where the teache
    
[^222]: 作为$\phi^6$场论的Blume-Capel模型：蒙特卡罗与流模型研究

    The Blume-Capel Model as a $\phi^6$ Field Theory: A Monte Carlo and Flow Model Study

    [https://arxiv.org/abs/2610.20982](https://arxiv.org/abs/2610.20982)

    本研究将归一化流生成模型用作无自相关的有效采样器，成功研究了连续自旋Blume-Capel模型（$\phi^6$场论）的相图，并以与正则及扩展系综蒙特卡罗相当的精度确定了一阶/二阶相变和三临界点的位置。

    

    我们报告了Blume-Capel模型连续自旋变体的蒙特卡罗模拟。我们计算了晶体场-温度平面上的相图，并将其作为自旋柔软度（软性）的函数进行研究。我们证明，归一化流——一类能够直接采样且无自相关性的生成模型——在整个相图范围内都是有效的采样器，包括在一阶相变区域和三临界点附近。对流样本进行的有限尺寸标度分析，能够以与正则系综和扩展系综蒙特卡罗相当的精度，确定一阶相变、二阶相变以及三临界点的位置。

    arXiv:2610.20982v1 Announce Type: cross  Abstract: We report Monte Carlo simulations of a continuous spin variant of the Blume-Capel model. We compute the phase diagrams in the crystal-field-temperature plane, as a function of the softness of the spins. We show that normalizing flows, a class of generative models capable of direct sampling without autocorrelation, are effective samplers across the phase diagram, including in the first-order transition region and in the vicinity of the tricritical point. Finite-size scaling analyses of flow samples locate the first- and second-order transitions as well as the tricritical point with accuracy comparable to canonical and extended-ensemble Monte Carlo.
    
[^223]: 如何训练你的小龙：教会小型语言模型构建与改进GPU优化脚本

    How to Train Your Little Dragon: Teaching a Small Language Model to Construct and Refine GPU Optimization Scripts

    [https://arxiv.org/abs/2610.20980](https://arxiv.org/abs/2610.20980)

    该论文提出了一种训练小型语言模型的方法，使其学会生成和迭代改进GPU内核优化脚本，通过设计紧凑无损的脚本表示以及基于TVM MetaSchedule构建的十余万条训练样本，实现了自动化的内核优化脚本构造与精炼。

    

    编译器变换之间存在相互作用，这使得GPU内核优化成为一项搜索任务——需要找到既能保持计算语义又能提升性能的变换组合。我们提出了"如何训练你的小龙"（HOW TO TRAIN YOUR LITTLE DRAGON），这是一种训练方法，教会小型语言模型既能构建完整的优化脚本，又能改进已有的优化脚本。我们的模型不对候选方案进行排序，也不生成内核代码，而是直接生成优化脚本。我们设计了一种紧凑的无损表示，通过其确定性解码能够精确重建编译器轨迹，完整保留其中的变换参数和依赖关系。我们利用编译器框架TVM MetaSchedule在54,116个不同的工作负载上构建训练数据，并在两个训练集中共构建了108,594个样本：其中60,236个样本教会模型从未优化的内核生成优化脚本，另外48,358个样本教会模型在给定内核和一个已有一定优化的脚本的情况下产出更好的脚本。

    arXiv:2610.20980v1 Announce Type: cross  Abstract: Compiler transformations interact, making GPU kernel optimization a search for combinations that preserve the computation while improving performance. We introduce HOW TO TRAIN YOUR LITTLE DRAGON, a training method that teaches a small language model both to construct complete optimization scripts and to improve existing ones. Instead of ranking candidates or generating kernel code, our model generates optimization scripts. We devise a compact, lossless representation whose deterministic decoding reconstructs the exact compiler trace, preserving its transformation parameters and dependencies. We use a compiler framework TVM MetaSchedule to build training data on 54,116 distinct workloads, and construct 108,594 examples in two training sets: 60,236 teach the model to generate an optimized script from an unoptimized kernel, while 48,358 teach it to produce a better script given the kernel and a somewhat optimized script. Supervised fine-
    
[^224]: 面向群体特定扩散模型的两阶段得分估计

    Two-Stage Score Estimation for Group-Specific Diffusion Models

    [https://arxiv.org/abs/2610.20979](https://arxiv.org/abs/2610.20979)

    该论文提出了一个针对群体特定扩散模型的两阶段得分估计统计框架，通过将共享池化得分与目标特定偏移量分离分析，推导出收敛速率，并证明扩散模型可作为密度估计器获得全变差保证。

    

    跨相关分布的生成建模提出了一个问题：共享结构如何在观测有限的情况下支持对目标分布的准确生成？我们开发了一个用于扩散迁移的统计框架，分析共享的池化得分与目标特定的偏移量。在池化密度满足Besov正则性以及偏移量满足稳定分层复合模型的条件下，我们推导出能够分离学习这两个组件成本的收敛速率，并识别出何时更简单的偏移量可以产生比单独拟合更快的风险界速率。在适当的边界和正则性假设下，我们证明随着扩散时间趋于零，偏移量保持有界且具有联合正则性。这些结果使得下时间截断能够以多项式方式趋于消失，并为扩散模型作为密度估计器提供了全变差保证。合成模拟和图像基准测试提供了概念验证。

    arXiv:2610.20979v1 Announce Type: cross  Abstract: Generative modeling across related distributions raises a question: how can shared structure support accurate generation for a target distribution with limited observations? We develop a statistical framework for diffusion transfer that analyzes a shared pooled score together with a target-specific offset. Under Besov regularity of the pooled density and a stable hierarchical composition model for the offset, we derive convergence rates that separate the costs of learning these components and identify when a simpler offset yields a faster risk-bound rate than separate fitting. Under suitable boundary and regularity assumptions, we prove that the offset remains bounded and jointly regular as diffusion time approaches zero. These results allow the lower time cutoff to vanish polynomially and provide total-variation guarantees for diffusion models as density estimators. Synthetic simulations and image benchmarks provide a proof of concept
    
[^225]: MIDI-REPA：通过符号表示对齐改进文本到音乐生成

    MIDI-REPA: Improving Text-to-Music Generation with Symbolic Representation Alignment

    [https://arxiv.org/abs/2610.20967](https://arxiv.org/abs/2610.20967)

    MIDI-REPA通过自动音乐转录提取MIDI符号监督信号，并将其与流匹配DiT早期层的中间表示进行对齐，从而在无需额外数据且仅用单张RTX 3090训练约31小时的低成本条件下，显著提升文本到音乐生成的音乐结构质量。

    

    最先进的文本到音乐模型日益依赖模型容量、数据和计算规模的扩展，同时仍然直接从密集音频中学习音乐结构。然而，旋律、节奏、和声与曲式结构可以以符号形式紧凑地表示。我们提出了MIDI-REPA，这是一个表示对齐框架，它从自动音乐转录（AMT）中获取MIDI监督信号，以对齐流匹配DiT的中间表示。基于2026年ICME学术文本到音乐挑战赛的122M参数FluxAudio-S基线模型，我们研究了对齐的内容、位置和时机。实验表明，将MuScriptor生成的钢琴卷帘表示与DiT早期层在整个流轨迹过程中进行对齐的效果最佳。在无需额外训练数据或技术的情况下，MIDI-REPA在ATTM客观指标上保持竞争力，同时可在单个RTX 3090上于约31小时内完成训练。我们的结果凸显了符号监督作为一种简单而有效的提升音乐生成模型性能的方式。

    arXiv:2610.20967v1 Announce Type: cross  Abstract: State-of-the-art text-to-music models increasingly rely on scaling model capacity, data, and compute, while still learning musical structure directly from dense audio. Yet melody, rhythm, harmony, and structure can be represented compactly in symbolic form. We introduce MIDI-REPA, a representation-alignment framework that derives MIDI supervision from automatic music transcription (AMT) to align intermediate representations of a flow-matching DiT. Building on the 122M-parameter FluxAudio-S baseline from the 2026 ICME Academic Text-To-Music Challenge, we investigate what, where, and when to align. Aligning a MuScriptor-derived piano roll with an early DiT layer throughout the full flow trajectory performs the best. Without additional training data or techniques, MIDI-REPA remains competitive on the ATTM objective metrics while remaining trainable on a single RTX 3090 in about 31 hours. Our results highlight symbolic supervision as a sim
    
[^226]: 基于头皮脑电的跨被试想象语音解码：在三个公开数据集上的防数据泄漏评估

    Cross-subject decoding of imagined speech from scalp EEG: a leakage-controlled evaluation on three public corpora

    [https://arxiv.org/abs/2610.20954](https://arxiv.org/abs/2610.20954)

    该研究在三个公开脑电数据集上采用严格防数据泄漏的评估协议，发现头皮EEG想象语音解码无法跨被试迁移，并系统检验了被试内解码准确率受限的多种可能解释。

    

    目的：一个无需校准即可适用于新用户的想象语音解码器将使非侵入式语音接口走向实用。我们检验了头皮脑电图（EEG）记录的想象语音内容能否在人与人之间迁移，并探讨了单个被试内解码准确率仍然偏低的原因。方法：我们分析了三个公开数据集——ds003626、KaraOne 以及 2020 年国际脑机接口竞赛 Track 3——所采用的评估协议通过断言强制保证数据划分的完整性，并报告精确二项式阈值、组级检验和自助法置信区间。我们以每名参与者的数据训练解码器并在其余所有参与者上进行测试，在不使用分类器的情况下比较参与者之间的表征几何结构，应用被试级对齐方法，并针对被试内性能上限检验了五种可能的解释。主要结果：在 482 个有序的参与者配对中，没有任何一对在多重比较校正后仍保持显著（摘要原文在此处截断）。

    arXiv:2610.20954v1 Announce Type: cross  Abstract: Objective. A decoder of imagined speech that works for a new user without calibration would make a non-invasive speech interface practical. We tested whether imagined-speech content recorded with scalp electroencephalography (EEG) transfers across people, and examined why accuracy within a single person remains low. Approach. We analyzed three public corpora, ds003626, KaraOne and Track 3 of the 2020 International Brain-Computer Interface Competition, under a protocol that enforces split integrity by assertion and reports exact binomial thresholds, group tests and bootstrap confidence intervals. We trained decoders on each participant and tested them on every other, compared representational geometry across participants without a classifier, applied subject-level alignment, and tested five explanations for the within-subject ceiling. Main results. Across 482 ordered pairs of participants, no pair survived correction for multiple compar
    
[^227]: LEV：基于循环决策模型的快速且深度思考

    LEV: Thinking Fast and Deep with Looped Decision Models

    [https://arxiv.org/abs/2610.20951](https://arxiv.org/abs/2610.20951)

    LEV是一种循环决策模型，通过反复运行模型层级而非生成token来实现按需可变的思考深度，单个模型即可在任意深度作答，且准确率随深度提升，尤其在困难问题上收益显著。

    

    决策模型通过对输入的提问为每个选项输出一个概率来作答，在单次前向传播中完成计算，无需生成文本。就像双过程理论中快速的“系统1”，它们能够立即作答，但其深度在训练时即已固定，因此在简单问题和困难问题上花费相同的计算量。通过生成文本进行推理的模型可以思考更久，但每多推一步都需要付出解码token的代价。我们提出了LEV，一种通过循环重复运行其层级而非生成token来实现更深思考的决策模型。LEV构建于以循环方式预训练的语言模型之上，在每次循环步骤后读取选项并联合训练所有步骤，因此单个模型检查点可以在任意深度进行作答。在三个循环骨干模型上，同一权重的准确率随深度增加而提升，且提升主要体现在困难问题上：一个2.6B参数的LEV在JevBench上的准确率从深度一时的63.2%提升至深度四时的76.2%。按需分配的深度由一个小型停止器针对每个问题进行设置……

    arXiv:2610.20951v1 Announce Type: new  Abstract: Decision models answer typed questions about an input with one probability per option, computed in a single forward pass without generating text. Like the fast System 1 of dual-process theory, they answer at once, but their depth is fixed at training time, so they spend the same computation on easy and hard questions. Models that reason in generated text can think longer, but they pay for every extra step in decoded tokens. We present LEV, a decision model that thinks deeper by rerunning its layers in a loop instead of generating tokens. LEV builds on a language model pretrained as a loop, reads the options after every loop step and trains all steps jointly, so one checkpoint can answer at any depth. On three looped backbones, accuracy rises with depth on the same weights, mostly on hard questions, and a 2.6B LEV goes from 63.2% at depth one to 76.2% at depth four on JevBench. On-demand depth, set per question by a small stopper with a t
    
[^228]: 排序正确，量级失准：集合离散度在径流极端事件中悄然失效

    Ranked Right, Sized Wrong: Ensemble Spread Fails Silently at Streamflow Extremes

    [https://arxiv.org/abs/2610.20947](https://arxiv.org/abs/2610.20947)

    研究发现深度学习径流预报模型的集合离散度虽能正确排序预测误差，但其量级随流量增大而系统性低估误差（平常流量下差2倍、极端流量下可达18倍），表明该不确定性信号在径流极端事件中会悄然失效。

    

    深度学习模型，尤其是长短期记忆网络，正越来越多地用于业务化径流预报，其集合离散度可提供预测不确定性的信号。随着气候变化导致水文气候极端事件频率增加，这些模型将越来越多地在历史记录中罕见的流量条件下运行。因此，我们评估了集合离散度在流量增大时是否仍具有信息量，从其相对于误差的量级以及对误差进行排序的能力两方面进行考察。结果显示：在平常流量下，预测误差约为校正后离散度的2倍，在最高流量下高达4倍，且该偏差随流量增大而加剧（对数-对数斜率1.30，95%置信区间：[1.20, 1.41]）。欠离散现象在超出训练记录的观测极端流量中最为严重，偏差可达18倍。这一现象普遍存在，且无法用条件均值偏差来解释。Area Under t（摘要原文在此处截断，后文缺失）

    arXiv:2610.20947v1 Announce Type: cross  Abstract: Deep learning models, particularly Long Short-Term Memory networks, are increasingly used for operational streamflow forecasting, with ensemble spread providing a signal of predictive uncertainty. As climate change increases the frequency of hydroclimatic extremes, these models will increasingly operate under flow conditions that are rare in their historical records. We therefore evaluate whether ensemble spread remains informative under increasing flows, examining both its magnitude relative to error and its ability to rank errors. Prediction error is approximately $2\times$ the corrected spread at ordinary flows and up to $4\times$ at the highest flows, with the discrepancy increasing with flow (log-log slope $1.30$, 95% CI: $[1.20,1.41]$). Under-dispersion is strongest in observed extreme flows beyond the training record, reaching an $18\times$ discrepancy. This is pervasive and not explainable by conditional mean bias. Area Under t
    
[^229]: 规划以学习：基于GPU的具有未来学习保证的实时鲁棒自适应控制

    Planning to Learn: Real-Time Robust Adaptive Control on the GPU with Guaranteed Future Learning

    [https://arxiv.org/abs/2610.20925](https://arxiv.org/abs/2610.20925)

    提出A-SLS框架，通过预测未来参数不确定性并构建考虑未来信息增益的紧致可达集，结合GPU加速求解器，实现了具有未来学习保证的实时鲁棒自适应控制，并可扩展至四足机器人全身控制。

    

    鲁棒自适应控制对于在陌生环境中主动降低不确定性、保障机器人安全至关重要。然而，传统方法往往过于保守，因为它们在规划时未能考虑有保证的未来参数学习，并且难以扩展到高维系统。在这项工作中，我们提出了A-SLS，这是一个基于系统级综合的学习感知鲁棒自适应模型预测控制（RAMPC）框架。A-SLS预测未来的参数不确定性，并保留过程扰动、参数误差和估计更新之间的相关性，从而构建考虑未来信息增益的紧凑且有保证的可达集。我们还开发了GPU加速的A-SLS求解器以实现实时性能。在多个非线性机器人系统上，A-SLS降低了可达管的保守性，保持了实时性能，并可扩展到具有61个状态、8个不确定参数的四足机器人全身控制。

    arXiv:2610.20925v1 Announce Type: cross  Abstract: Robust adaptive control is vital for keeping robots safe while actively reducing uncertainty in unfamiliar settings. However, traditional methods are often conservative since they fail to account for guaranteed future parameter learning during planning and struggle to scale to high-dimensional systems. In this work, we present A-SLS, a learning-aware robust adaptive model predictive control (RAMPC) framework based on system-level synthesis. A-SLS forecasts future parameter uncertainty and preserves correlations between process disturbances, parameter error, and estimate updates to construct tight, guaranteed reachable sets that account for future information gain. We also develop a GPU-accelerated A-SLS solver to enable real-time performance. Across nonlinear robotic systems, A-SLS reduces reachable-tube conservativeness, maintains real-time performance, and scales to whole-body control for a quadruped with 61 states, 8 uncertain param
    
[^230]: 定义与生成组合结构

    Defining and Generating Compositional Structure

    [https://arxiv.org/abs/2610.20922](https://arxiv.org/abs/2610.20922)

    该论文利用算法信息论将组合结构形式化定义为“当部分被整体考虑时会显现出单独建模时无法捕捉的共享结构”，从而为在任意数据中识别和生成这种结构提供了数学框架。

    

    组合结构在自然界中无处不在，并跨越各个尺度：原子组合成分子，分子组合成细胞，细胞组合成生物体，同时在每个层级内部还呈现出进一步的组合层级。更广泛地说，组合性——即部分的复用与重组以构成新结构——是自然与人工系统中有趣复杂性的根本来源。然而，尽管其重要性不言而喻，尽管我们显然能轻松识别它，但能够捕捉我们直觉上所理解的组合结构的数学化论述仍然有限。在此，我们通过将一个简单的想法形式化来填补这一空白：当若干部分被放在一起考虑时，会显现出在单独建模这些部分时无法捕捉到的共享结构，此时便产生了组合结构。我们借助算法信息论——信息与压缩的数学——来表达这一思想，该理论为在任意数据中识别结构提供了一种自然的方式，而无需……

    arXiv:2610.20922v1 Announce Type: new  Abstract: Compositional structure is pervasive in nature and across scales: atoms combine into molecules, molecules into cells, and cells into organisms, while further compositional hierarchies appear within each level. More broadly, compositionality, the reuse and recombination of parts into novel structures, is a fundamental source of interesting complexity in both natural and artificial systems. Yet despite its importance, and despite our apparent ease in recognizing it, mathematical accounts capturing what we intuitively mean by compositional structure remain limited. Here, we fill this gap by formalizing a simple idea: compositional structure emerges when parts, considered together, reveal shared structure that is not captured when those parts are modeled separately. We express this idea using algorithmic information theory, the mathematics of information and compression, which provides a natural way to identify structure in arbitrary data wi
    
[^231]: 基于熵增益采样与答案混合的递归离线蒸馏

    Recursive Offline Distillation via Entropy Gain Sampling and Answer Mixing

    [https://arxiv.org/abs/2610.20914](https://arxiv.org/abs/2610.20914)

    该论文提出一种递归离线蒸馏方法，通过以与辅助中等模型的熵差异为权重采样推理链（熵增益采样），并用单token真实答案补充思维轨迹（答案混合），在相同数据预算下显著优于朴素蒸馏并呈现更陡峭的学习曲线。

    

    将大型语言模型（LLM）的推理链蒸馏到较小模型中是一种标准的训练方法。然而，研究人员通常无法完全访问教师模型及其内部表示，这限制了最终性能。我们提出了一种递归离线蒸馏方法，结合两个组件：用于选择性蒸馏的基于熵的采样和答案混合。具体而言，针对三个开源模型（3-4B），在每个训练周期，我们以与辅助中等模型（70-72B）之间的熵差异为权重采样推理链，然后用单token的真实答案补充思维轨迹。在相当的数据预算下，我们的方法始终优于朴素蒸馏（平均 57.15 ± 1.01 对 51.61），并且学习曲线明显更陡峭。此外，我们对生成和规范化蒸馏轨迹的各种方法进行了比较分析。

    arXiv:2610.20914v1 Announce Type: new  Abstract: Distillation of reasoning chains from Large Language Models (LLMs) into their smaller siblings is a standard training approach. Frequently, researchers do not have full access to a teacher model and its internal representations, which limits the resulting performance. We propose a recursive offline distillation method combining two components: entropy-based sampling for selective distillation and answer mixing. Specifically, for three open models (3-4B), at each epoch, we sample reasoning chains weighted by entropy difference with an auxiliary medium model (70-72B) and, next, supplement thinking traces with single-token ground truth answers. Our approach consistently outperforms naive distillation with comparable data budgets ($57.15 \pm 1.01$ vs 51.61 on average) and offers a much steeper learning curve. Additionally, we present a comparative analysis of methods to generate and normalize distillation traces.
    
[^232]: 可复制的对决老虎机

    Replicable Dueling Bandits

    [https://arxiv.org/abs/2610.20908](https://arxiv.org/abs/2610.20908)

    该论文首次为可复制的对决老虎机设计了具有次线性遗憾的算法，在松弛随机传递性下实现 $\widetilde{O}(\gamma^8 K^4 H_p/\rho^2)$ 的期望遗憾，并在强随机传递性下通过同时淘汰策略改善了对臂数 $K$ 的依赖。

    

    决斗老虎机根据早期的噪声反馈结果来选择每一次比较，因此在相同偏好下运行两次，可能会比较不同的臂对。我们研究可复制的对决老虎机问题：两次执行共享内部随机性且比较结果相互独立，并要求以至少 $1-\rho$ 的概率在每一轮选择相同的有序臂对。在 Beat-the-Mean 算法的基础上——该算法的得分会随臂被淘汰而变化，且不一定将最优臂排在最高——我们设计了首批具有次线性遗憾的此类算法。在参数 $\gamma\ge 1$ 的松弛随机传递性以及随机三角不等式条件下，我们设计了一种算法，其期望遗憾为 $\widetilde{O}(\gamma^8 K^4 H_{\mathrm{p}}/\rho^2)$，其中 $H_{\mathrm{p}}=\sum_{i=2}^{K}(P_{1i}-1/2)^{-1}$。在强随机传递性条件下，我们设计了一种基于同时淘汰的算法，改善了对臂数 $K$ 的依赖性，并实现了 $\widetilde{O}(K^2\cdots)$ 的遗憾界。

    arXiv:2610.20908v1 Announce Type: cross  Abstract: Dueling bandits choose each comparison based on earlier noisy outcomes, so two runs on the same preferences can compare different pairs. We study replicable dueling bandits, where two executions with shared internal randomness and independent comparison outcomes must select the same ordered pair in every round with probability at least $1-\rho$. Building on Beat-the-Mean, whose scores change as arms are eliminated and need not rank the best arm highest, we design the first such algorithms with sublinear regret. Under relaxed stochastic transitivity with parameter $\gamma\ge 1$ and the stochastic triangle inequality, we design an algorithm that achieves expected regret $\widetilde{O}(\gamma^8 K^4 H_{\mathrm{p}}/\rho^2)$, where $H_{\mathrm{p}}=\sum_{i=2}^{K}(P_{1i}-1/2)^{-1}$. Under strong stochastic transitivity, we design an algorithm based on simultaneous elimination that improves the dependence on $K$ and achieves $\widetilde{O}(K^2 
    
[^233]: 以任务为中心的多模态学习用于认知障碍检测

    Task-Centric Multimodal Learning for Cognitive Impairment Detection

    [https://arxiv.org/abs/2610.20906](https://arxiv.org/abs/2610.20906)

    本文提出一种基于Amazon Alexa的非接触式多模态筛查框架，融合视频、音频和文本信号，全面捕捉语音模式、面部表情和语言结构等行为特征，以更准确地检测老年人的认知衰退。

    

    痴呆症影响着全球数百万老年人，但在临床环境之外进行早期检测仍然十分困难。传统模型主要依赖文本或语音响应，尽管人类表达本质上具有多模态特性，它们却只能捕捉到人类认知的有限视角。本工作提出了一种非接触式的多模态筛查框架，利用在与Amazon Alexa进行结构化语音交互过程中收集的视频、音频和文本信号。通过整合这些模态，我们的方法能够捕捉更全面的人类行为表征，涵盖语音模式、声音特征、面部表情和语言结构，从而可能实现更准确的认知衰退检测。不同参与者完成了三项对话任务，这些任务旨在探测自发言语、理解能力和回忆能力。我们追求三个目标：（1）评估三项Alexa任务中的哪一项（原文在此处截断）

    arXiv:2610.20906v1 Announce Type: cross  Abstract: Dementia affects millions of older adults worldwide, yet early detection remains difficult outside clinical settings. Traditional models primarily rely on text or voice-based responses, capturing only a limited view of human cognition despite the inherently multimodal nature of human expression. This work proposes a contactless, multimodal screening framework that leverages video, audio, and text signals collected during structured voice-based interactions with Amazon Alexa. By incorporating these modalities, our approach captures a more holistic representation of human behavior encompassing speech patterns, vocal characteristics, facial expressions, and linguistic structure, which may enable more accurate detection of cognitive decline. Different participants completed three conversational tasks designed to probe spontaneous speech, comprehension, and recall. We pursued three objectives: (1) evaluate which of the three Alexa tasks pro
    
[^234]: 面向误设回归的符号投注方法：方差自适应与最优逼近

    Sign Betting for Misspecified Regression: Variance Adaptivity and Optimal Approximation

    [https://arxiv.org/abs/2610.20903](https://arxiv.org/abs/2610.20903)

    本文证明符号投注回归在模型误设下可实现高概率的 $L_1$ 预言机不等式，其误差界自适应于平均条件方差，并证明其对恰当学习所能达到的普适逼近因子 3 是最优的。

    

    我们研究模型误设下的有界回归问题，通过函数 $f$ 与条件均值 $\eta(x)=\mathbb{E}[Y\mid X=x]$ 之间的 $L_1(P_X)$ 距离来评估函数，而非采用标准的 $L_2(P_X)$ 距离。这一 $L_1$ 度量对多臂老虎机和强化学习等下游决策问题而言是自然的。我们分析了 Baraud 与 Maillard（2025）提出的极小极大估计器，并将其称为“符号投注回归”。我们的主要结果表明，符号投注回归在模型误设情形下能够实现高概率的 $L_1$ 预言机不等式。在可实现性条件下，我们的界能够自适应于平均条件方差，并恢复了 Li 等人（2026）的方差自适应速率。我们进一步刻画了恰当学习中最优的普适逼近因子，证明符号投注方法可以达到因子 3，而没有任何恰当学习器能够在无分布假设、高概率的 $L_1$ 预言机不等式下满足更小的普适逼近因子（原文摘要在此处截断）。

    arXiv:2610.20903v1 Announce Type: new  Abstract: We study bounded regression under model misspecification, evaluating a function $f$ by its $L_1(P_X)$ distance to the conditional mean $\eta(x)=\mathbb{E}[Y\mid X=x]$, rather than the standard $L_2(P_X)$. This $L_1$ metric is natural for downstream decision problems such as bandits and reinforcement learning. We analyze the min--max estimator of Baraud & Maillard (2025), which we call \emph{sign-betting regression}. Our main results show that sign-betting regression achieves high-probability $L_1$ oracle inequalities under misspecification. Under realizability, our bound adapts to the average conditional variance and recovers the variance-adaptive rate of Li et al. (2026). We further characterize the optimal universal approximation factor for proper learning by showing that sign betting achieves factor 3, while no proper learner can satisfy a distribution-free, high-probability $L_1$ oracle inequality with a universal approximation facto
    
[^235]: TGrasp6D：任务感知的视觉-语言先验与位姿优化实现零样本面向任务的6自由度抓取

    TGrasp6D: Task-Informed Vision-Language Priors and Pose Refinement enable Zero-Shot Task-Oriented 6DoF Grasping

    [https://arxiv.org/abs/2610.20901](https://arxiv.org/abs/2610.20901)

    该论文提出TGrasp6D，一种基于视觉-语言模型的零样本面向任务6自由度抓取方法，通过推理任务相关的语义先验并结合位姿优化，摆脱了对有限人工标注任务特定抓取数据集的依赖。

    

    面向任务抓取已成为机器人有目的性操作的一项重要技能。例如，在“用刀切面包”的任务中，机器人必须选择抓握刀柄；而在“把刀递给别人”的任务中，则最好保持刀柄不被遮挡，以便安全交接。当前的面向任务抓取方法要么局限于二维平面抓取，要么受限于依赖与任务无关的抓取生成器，要么依赖于有限的、人工标注的、特定任务的抓取数据集，而这些数据集无法覆盖所有可能的抓取-任务组合。为此，我们提出了TGrasp6D，一种基于视觉-语言的方法，能够以零样本方式生成面向任务的6自由度抓取。给定一项任务和目标物体的一张RGB-D图像，TGrasp6D会针对给定任务和物体几何结构推理出合适的语义可供性，从而产生初始的任务感知先验。该生成的先验进一步……

    arXiv:2610.20901v1 Announce Type: cross  Abstract: Task-Oriented Grasping (TOG) has emerged as an important skill for purposeful robotic manipulation. Given the task of using a knife to "slice bread," a robot must choose to grasp from the handle, whereas for "hand[ing] the knife over," it is preferable for the handle to remain unobstructed for safe handoff. Current approaches for TOG either focus on 2D planar grasping, remain limited by their reliance on task-agnostic grasp generators, or are constrained by reliance on finite, manually-annotated, task-specific grasp datasets that cannot cover all possible grasp-task pairs. To that end, we present TGrasp6D, a vision-language based approach for generating task-oriented 6DoF grasps in a zero-shot manner. Given a task and a single RGB-D image of an associated object, TGrasp6D reasons about the appropriate semantic affordance for the given task and object geometry to produce an initial task- informed prior. This generated prior is further r
    
[^236]: 错误的投票，正确的梯度：为什么多数投票奖励能媲美可验证奖励强化学习

    Wrong Votes, Right Gradients: Why Majority-Vote Rewards Track Verifiable RL

    [https://arxiv.org/abs/2610.20898](https://arxiv.org/abs/2610.20898)

    该论文将分析单位从提示层面转向批次级参数更新，证明多数投票等自监督奖励之所以能媲美可验证奖励强化学习，是因为当标签误差在不同提示之间不相干时，其对参数更新的误导作用会随批次规模增大而消失。

    

    使用自生成标签的强化学习（例如对模型自身的采样结果进行多数投票）往往能获得与可验证奖励训练相当的大部分收益。从通常的提示层面视角来看，这一现象令人费解：多数投票只是返回模型的众数答案，无法增加关于正确性的信息。我们认为，正确的分析单元应是批次级别的更新。由于参数是共享的，在实际批次规模下，某个提示自身的梯度对该提示输出变化的贡献不足1%，因此错误标签只有在不同提示之间的标签误差共享相同方向时才会产生影响。我们将多数投票更新相对于金标准更新分解为收缩与误导两个分量，并证明：只要误差在不同提示之间不相干，无论标签出错的频率有多高，误导分量都会随批次规模增大而消失。通过在五个模型上测量每个响应的梯度，我们发现多数投票误差是……（原文摘要在此处截断）

    arXiv:2610.20898v1 Announce Type: cross  Abstract: Reinforcement learning with self-generated labels, such as majority vote over a model's own samples, often recovers most of the gain of training with verifiable rewards. This is puzzling from the usual prompt-level view, in which majority vote only returns a model's modal answer and cannot add information about correctness. We argue that the relevant unit is the batch-level update. Because parameters are shared, a prompt's own gradient accounts for less than 1\% of the change in that prompt's output at practical batch sizes, so a wrong label matters only to the extent that label errors share a direction across prompts. We decompose the majority-vote update into shrinkage and misdirection relative to the gold update, and show that misdirection vanishes with batch size whenever errors are incoherent across prompts, however often the labels are wrong. Measuring per-response gradients on five models, we find that majority-vote errors are a
    
[^237]: 随机梯度下降：非线性时间序列的视角

    Stochastic Gradient Descent: A Nonlinear Time Series Perspective

    [https://arxiv.org/abs/2610.20894](https://arxiv.org/abs/2610.20894)

    本文在迭代随机函数框架下，通过几何矩收缩（GMC）证明了恒定步长SGD在欧氏距离下的收敛性，克服了固定初始化带来的非平稳性问题，证明了平均SGD的淬火中心极限定理、不变原理和α-稳定收敛，并通过研究关于步长的导数为理解步长的影响提供了新视角。

    

    本文在迭代随机函数的框架下研究了具有恒定步长的随机梯度下降（SGD）的统计行为。与以往建立SGD在概率测度（例如Wasserstein距离）意义下收敛性的研究相比，我们的方法通过证明SGD的几何矩收缩（GMC）性质，直接给出了欧氏距离下的收敛性。这种新的收敛性能够解决由固定初始点导致的SGD非平稳性问题，并能对SGD进行更精细的渐近分析。具体而言，我们证明了平均SGD（ASGD）在任意初始化条件下的淬火中心极限定理、不变原理以及α-稳定收敛性。此外，我们通过研究SGD关于步长的导数，为理解步长对SGD的影响提供了一个新颖的视角。我们证明了导数过程平稳解的存在性，这使得……

    arXiv:2610.20894v1 Announce Type: cross  Abstract: This paper investigates the statistical behavior of Stochastic Gradient Descent (SGD) with constant step size under the framework of iterated random functions. Compared to previous studies establishing the convergence of SGD in probability measure, e.g., Wasserstein distance, our approach directly provides the convergence in Euclidean distance by showing the Geometric Moment Contraction (GMC) of SGD. This new convergence can address the non-stationarity of SGD due to fixed initial points and can provide a more refined asymptotic analysis of SGD. Specifically, we prove quenched central limit theorem, invariance principle and $\alpha$-stable convergence for averaged SGD (ASGD) regardless of initialization. Furthermore, we provide a novel perspective to understand the impact of step sizes in SGD by studying its derivative with respect to the step size. The existence of stationary solutions for the derivative processes is shown, enabling t
    
[^238]: 针对删失结果的灵活双重稳健邻近因果推断

    Flexible Doubly Robust Proximal Causal Inference for Censored Outcomes

    [https://arxiv.org/abs/2610.20893](https://arxiv.org/abs/2610.20893)

    该论文提出了邻近生存桥梁学习（PSB），一个针对删失结果的灵活双重稳健邻近因果推断框架，通过核估计器（PSB-K）和神经估计器（PSB-N）两种灵活桥梁学习器直接处理删失数据和生存曲线目标，突破了现有方法依赖限制性参数化桥梁模型的局限。

    

    在存在未测量混杂因素的情况下，估计右删失生存时间结果的因果效应具有挑战性。邻近因果推断通过代理变量来解决未测量混杂问题，但现有的邻近生存方法主要依赖于限制性较强的参数化桥梁模型。我们提出了邻近生存桥梁学习，这是一个针对删失结果的灵活双重稳健邻近因果推断框架。基于现有的治疗特异性生存曲线的邻近识别方法，该框架给出了结果桥梁（OB）、治疗桥梁（TB）和双重稳健（DR）估计器，并提供了两种灵活的桥梁学习器：PSB-K，一种具有低秩时间分解的核估计器；以及PSB-N，一种通过投影最大矩约束损失训练的神经估计器。与现有的针对标量结果的灵活邻近学习器不同，PSB能够直接处理删失数据和生存曲线目标。在标准结构方程模型基准上，……

    arXiv:2610.20893v1 Announce Type: cross  Abstract: Estimating causal effects on right-censored time-to-event outcomes is challenging under unmeasured confounding. Proximal causal inference addresses unmeasured confounding through proxy variables, but existing proximal survival methods rely largely on restrictive parametric bridge models. We introduce Proximal Survival Bridge learning (PSB), a flexible doubly robust proximal framework for censored outcomes. Building on existing proximal identification of treatment-specific survival curves, the framework yields outcome-bridge (OB), treatment-bridge (TB), and doubly robust (DR) estimators, with two flexible bridge learners: PSB-K, a kernel estimator with a low-rank time decomposition, and PSB-N, a neural estimator trained by a projected maximum-moment-restriction loss. Unlike existing flexible proximal learners for scalar outcomes, PSB directly handles censoring and survival-curve targets. On a standard SEM benchmark, a proximal semi-synt
    
[^239]: 从实测求解器成本学习单纯形法的热启动

    Learning Warm Starts for the Simplex Method from Measured Solver Cost

    [https://arxiv.org/abs/2610.20892](https://arxiv.org/abs/2610.20892)

    该论文发现学习型热启动虽能消除单纯形法三分之二的迭代却只能节省三分之一的时间，其原因是被消除的迭代往往代价低廉且与最优基的接近程度无法反映实际计算成本，因此应以实测求解成本（如枢轴次数，可解释71%的对数求解时间方差）作为学习热启动的指导目标。

    

    arXiv:2610.20892v1 公告类型： cross 摘要：针对单纯形法的学习型热启动，通过预测哪些变量在最优解处将成为基变量来构建初始基。先前的工作表明，由图神经网络预测的初始基可以大幅减少迭代次数，但所节省的时间可能远低于所节省的迭代次数。在25类线性规划（LP）问题上，我们发现一个强大的监督式基预测器能够消除对偶单纯形法三分之二的迭代，却只能消除其三分之一的时间。两个效应解释了这一差距：良好的初始基所消除的迭代往往是代价低廉的，并且与最优基的接近程度并不能按成本对初始基进行排序。我们证明，两个与最优基具有相同重叠程度且枢轴次数相同的初始基，其线性代数计算量可能相差一个随行数平方根增长的因子。实证结果表明，在同一个线性规划问题的各个初始基中，枢轴次数解释了对数求解时间方差的71%，而CPU指令（摘要在此处截断）

    arXiv:2610.20892v1 Announce Type: cross  Abstract: Learned warm starts for the simplex method build starting bases using predictions of which variables will be basic at optimality. Prior work has shown that starting bases predicted by graph neural networks can sharply reduce iteration counts, but the time saved may fall well short of the iterations saved. On 25 classes of linear programs (LPs), we find that a strong supervised basis predictor removes two thirds of the dual simplex's iterations but only a third of its time. Two effects explain the gap: the iterations removed by a good start tend to be cheap, and closeness to an optimal basis does not rank starts by cost. We prove that two starts with equal overlap with an optimal basis and equal pivot counts can differ in linear-algebra work by a factor that grows with the square root of the number of rows. Empirically, across the starts of one LP, the pivot count explains 71% of the variance of log solve time, while the CPU instruction
    
[^240]: 涌现失对齐模型是否共享同一电路？

    Do Emergent Misalignment Models Share a Circuit?

    [https://arxiv.org/abs/2610.20887](https://arxiv.org/abs/2610.20887)

    研究发现在不同有害任务上独立微调得到的涌现失对齐模型共享相同的注意力头电路，将这些头的贡献替换为基础模型的贡献即可大幅降低失对齐行为且不损害连贯性。

    

    在狭义定义的有害任务上对语言模型进行微调，可能在无关领域诱发涌现性失对齐。尽管先前的工作已在各自独立微调的模型中识别出共享的与失对齐相关的表示，但这些模型是否依赖共同的计算组件仍不清楚。我们使用三个基于 Qwen2.5-3B-Instruct 的 LoRA 微调模型来研究这个问题，它们分别在有害医疗建议、高风险金融建议和极限运动推荐任务上训练。通过基于方向的激活追踪，我们在所有三个模型体排名前五的注意力头中识别出三个共享的注意力头位置。将每个模型体所选的五个注意力头的贡献替换为基础模型的贡献后，自由生成中的涌现失对齐率从医疗领域的 38.9% 降至 22.9%，金融领域从 50.7% 降至 12.5%，体育领域从 41.7% 降至 11.1%，其效果优于层级匹配的随机干预，且不会降低被评判的连贯性。

    arXiv:2610.20887v1 Announce Type: new  Abstract: Fine-tuning language models on narrowly defined harmful tasks can induce emergent misalignment (EM) across unrelated domains. While prior work has identified shared misalignment-associated representations across independently fine-tuned models, it remains unclear whether these models rely on common computational components. We investigate this question using three LoRA fine-tunes of Qwen2.5-3B-Instruct trained on harmful medical advice, risky financial advice, and extreme-sports recommendations. Using direction-based activation tracing, we identify three attention-head locations shared among the five highest-ranked heads of all three organisms. Replacing each organism's five selected head contributions with Base contributions reduces free-generation EM from 38.9% to 22.9% for medical, 50.7% to 12.5% for finance, and 41.7% to 11.1% for sports, outperforming layer-matched random interventions without reducing judged coherence. Cross-organi
    
[^241]: 从像素到词元：面向低预算语义分割的词元空间高效主动学习

    Pixels to Tokens: Token Space Efficient Active Learning for Low-Budget Semantic Segmentation

    [https://arxiv.org/abs/2610.20881](https://arxiv.org/abs/2610.20881)

    提出了首个基于ViT表示的像素级主动学习框架TEAL，通过在冻结ViT骨干的原生词元网格上直接执行两级MaxHerding多样性选择，实现了极低标注预算下高效的语义分割。

    

    尽管基础模型和视觉Transformer（ViT）已经重新定义了视觉表示，但目前针对这些骨干网络的主动学习（AL）策略大多在补丁或区域级别上运行，由于需要进行大量的像素标注，标注需求依然很高。将基于ViT的主动学习扩展到像素级、低预算的场景尤为困难：在每张图像仅查询一个像素的情况下，方法必须同时识别出最有信息量的位置，并从极其稀疏的信号中训练出可靠的解码器头。我们提出了TEAL（Token-space Efficient Active Learning，词元空间高效主动学习），这是首个围绕ViT表示构建的像素级主动学习框架。TEAL使用冻结的ViT骨干网络，并直接在其原生词元网格上执行基于多样性的选择，从而避免了将词元嵌入插值到密集像素分辨率时所引入的伪影。我们的框架在多层词元上应用了两级MaxHerding策略

    arXiv:2610.20881v1 Announce Type: cross  Abstract: While foundation models and vision transformers (ViTs) have redefined visual representations, current active learning (AL) strategies for these backbones largely operate at the patch or region level, where annotation requirements remain high due to extensive pixel labeling. Extending ViT-based AL to the pixel-level, low-budget regime is uniquely challenging: with as little as one pixel query per image, a method must simultaneously identify the most informative locations and train a reliable decoder head from an extremely sparse signal. We introduce TEAL (Token-space Efficient Active Learning), the first pixel-level AL framework built around ViT representations. TEAL uses a frozen ViT backbone and performs diversity-based selection directly on its native token lattice, avoiding the artifacts introduced by interpolating token embeddings to dense pixel resolutions. Our framework applies a two-level MaxHerding strategy over multi-layer tok
    
[^242]: CogDrift-4D：用于轻度认知障碍检测的时序行为退化的层次化多模态建模

    CogDrift-4D: Hierarchical Multimodal Modeling of Temporal Behavioral Decay for MCI Detection

    [https://arxiv.org/abs/2610.20877](https://arxiv.org/abs/2610.20877)

    该论文提出了CogDrift-4D框架，将18个月内七次多模态非脚本化交互的行为数据构建为4D张量，并通过层次化Transformer的双层架构（会话内疲劳建模与会话间漂移建模）捕捉认知衰退的时序轨迹，从而实现轻度认知障碍的早期检测。

    

    轻度认知障碍（MCI）的早期检测仍然是痴呆症护理中的一个关键瓶颈，亟需能够捕捉细微行为退化的可扩展工具。为了捕捉认知衰退的微妙时序变化，我们引入了一个在18个月的时间跨度内、通过七次非脚本化多模态交互收集的行为数据集。该数据集将连续的、多会话的交互形式化为一个密集的4D张量（批次 * 会话 * 时间 * 特征），显式地同时保留了局部动态与长期变化。为了有效处理这些复杂的行为模式，我们提出了CogDrift-4D：一种新颖的层次化4D Transformer框架。为了捕捉认知衰退的复杂轨迹，CogDrift-4D实现了一种双层次的行为疲劳与漂移建模架构。在会话内层面，该模型计算局部最小二乘回归斜率，以量化帧级特征的急性“消退”……

    arXiv:2610.20877v1 Announce Type: cross  Abstract: Early detection of mild cognitive impairment remains a critical bottleneck in dementia care, demanding scalable tools capable of capturing subtle behavioral decay. To capture the subtle temporal changes of cognitive decline, we introduce a behavioral dataset collected across seven unscripted multimodal interactions over an 18-month horizon. This dataset formalizes continuous, multi-session interactions into a dense 4D tensor (Batch * Session * Time * Feature) that explicitly preserves concurrent local dynamics alongside long-term changes. To effectively process these complex behavioral patterns, we propose CogDrift-4D: a novel, hierarchical 4D Transformer framework. To capture the complex trajectory of cognitive decline, CogDrift-4D implements a dual-level behavioral fatigue and drift modeling architecture. At the intra-session level, the model computes localized least-squares regression slopes to quantify the acute ``fading'' of frame
    
[^243]: 基于机器学习的、受实验测量约束的中微子事例生成

    Machine-learning-based neutrino-event generation constrained by experimental measurements

    [https://arxiv.org/abs/2610.20870](https://arxiv.org/abs/2610.20870)

    该论文提出了一种机器学习生成式中微子事例模型，可直接利用实验测得的微分截面（无需事例级实验样本）进行校准，且校准后无需重新调参即可在不同可观测量和束流配置间保持适用性。

    

    我们提出了一种用于带电流 νμ-原子核散射的生成式代理模型，该模型可以在没有事例级实验样本的情况下，利用测得的微分截面进行校准。从理论相互作用模型出发，我们针对MicroBooNE的测量对代理模型进行微调，结果表明这一过程同时也改善了对未用于拟合的可观测量的描述。经过校准的模型无需重新调参即可应用于来自不同中微子通量的测量，同时保留对出射缪子运动学的事例级预测。我们的结果表明，实验截面可以直接约束生成式事例模型，同时保持其在不同可观测量和束流配置下的适用性。

    arXiv:2610.20870v1 Announce Type: cross  Abstract: We present a generative surrogate for charged-current \(\nu_\mu\)-nucleus scattering that can be calibrated to measured differential cross sections without event-level experimental samples. Starting from a theoretical interaction model, we fine-tune the surrogate to a MicroBooNE measurement and show that this also improves the description of observables not used in the fit. The calibrated model can also be applied to measurements from different neutrino fluxes without retuning, while retaining event-level predictions of outgoing-muon kinematics. Our results show that experimental cross sections can directly constrain generative event models while preserving their applicability across different observables and beam configurations.
    
[^244]: SHRED：无需专家演示的异构道路使用者自我博弈方法

    SHRED: Self-Play for Heterogeneous Road-Users without Expert Demonstrations

    [https://arxiv.org/abs/2610.20866](https://arxiv.org/abs/2610.20866)

    SHRED提出了一个无需专家演示的异构自我博弈框架，从随机初始化出发联合训练车辆、行人和骑行者三类道路使用者，使逼真的交互行为在闭环自博弈中自然涌现，并在WOSAC基准上超越现有无演示自我博弈方法。

    

    现实的交通仿真与自动驾驶规划都依赖于对异构道路使用者之间交互的建模，而这些道路使用者在动力学特性、约束条件和行为模式上存在根本差异。现有方法主要从大规模人类驾驶演示数据中学习真实的交互行为。在本工作中，我们探讨这种交互行为是否能够纯粹通过自我博弈自然涌现。我们提出了SHRED，这是一个从随机初始化开始、通过大规模异构自我博弈联合训练车辆、行人和骑行者的框架，无需演示数据或记录轨迹回放。SHRED将特定类型的动力学模型与目标相结合，并采用程序化场景生成和逐智能体随机化，使多样化的行为能够在闭环交互中自然涌现。在Waymo开放仿真智能体挑战赛（WOSAC）上，SHRED联合控制全部三类道路使用者，并优于最接近的无演示自我博弈方法

    arXiv:2610.20866v1 Announce Type: cross  Abstract: Realistic traffic simulation and autonomous planning both depend on modeling interactions among heterogeneous road users whose dynamics, constraints, and behaviors differ fundamentally. Existing approaches predominantly learn realistic interaction behavior from large-scale human driving demonstrations. In this work, we ask whether such behavior can instead emerge purely through self-play. We present SHRED, a framework that jointly trains vehicles, pedestrians, and cyclists through large-scale heterogeneous self-play from random initialization, without demonstrations or logged-trajectory replay. SHRED combines type-specific dynamics and objectives with procedural scenario generation and per-agent randomization, allowing diverse behaviors to emerge through closed-loop interaction. On the Waymo Open Sim Agents Challenge (WOSAC), SHRED jointly controls all three road-user types and outperforms both the closest demonstration-free self-play 
    
[^245]: 长上下文草稿生成无需长上下文草稿模型

    Long-Context Drafting Does Not Require Long-Context Drafters

    [https://arxiv.org/abs/2610.20865](https://arxiv.org/abs/2610.20865)

    长上下文投机解码无需构建长上下文草稿模型——通过限制注意力缓存和重定基两种推理时机制，现成的短上下文草稿模型即可远超其训练长度高效运行，因为草稿模型借助目标模型的隐藏状态已经能感知遥远上下文。

    

    投机解码将一个廉价的草稿模型与针对目标模型的并行验证相结合。在超长上下文场景下，它不再划算，因为一个对完整前缀执行注意力计算的草稿模型会继承目标模型随上下文变化的计算成本，因此它本应提供的加速先是逐渐被侵蚀，随后甚至发生逆转。标准的应对方案是构建一个长上下文草稿模型——通过训练或微调获得，或者使用目标模型检查点中自带的多词元预测头。我们认为这是不必要的，因为现有使用的草稿模型都是以目标模型为条件的：它们读取的是目标模型在整个前缀上计算出的隐藏状态，因此遥远的上下文无需草稿模型自身的注意力机制就已经能够传递给它们，而验证过程则会纠正它们遗漏的内容。因此，我们冻结已发布的短上下文草稿模型，并在两种推理时机制下使其运行远超其训练长度：一种机制限制它们所看到的注意力缓存，从而固定其成本；另一种机制对其进行重定基处理……

    arXiv:2610.20865v1 Announce Type: new  Abstract: Speculative decoding pairs a cheap drafter with parallel verification against the target. At ultra-long context it stops paying, because a drafter that attends over the full prefix inherits the target's context-dependent cost, so the speedup it was meant to provide first erodes and then reverses. The standard response is to build a long-context drafter, by training or fine-tuning one, or by using a multi-token-prediction head shipped inside the target checkpoint. We argue this is unnecessary, because the drafters in use are target-conditioned, reading hidden states the target computed over the entire prefix, so distant context already reaches them without their own attention, and verification corrects what they miss. We therefore freeze released short-context drafters and run them far past their training length under two inference-time mechanisms, one bounding the attention cache they see, which fixes their cost, and one re-basing their 
    
[^246]: SNOAR：无需干净参考的语音去噪自训练框架——噪声过滤、观测回加与混音

    SNOAR: Self-training with Noise filtering, Observation adding, and Remixing for Speech Denoising Without Clean References

    [https://arxiv.org/abs/2610.20863](https://arxiv.org/abs/2610.20863)

    提出 SNOAR 自训练框架，在使用教师估计结果之前先进行校正——用差分噪声过滤（DNF）去除泄漏噪声、并回加部分含噪观测以恢复被过度抑制的语音和减弱伪影，从而在无需干净参考语音的情况下显著提升自训练去噪性能。

    

    诸如 MixIT 和含噪目标训练等无监督语音去噪方法可以直接在含噪录音上进行训练，避免了合成训练数据带来的仿真到真实（sim-to-real）差距，但其性能仍远低于有监督方法。以 RemixIT 为代表的师生自训练方法通过将教师的估计结果重新混音成新的混合信号来监督学生，从而缩小了这一差距；然而，由于学生仅以教师的估计结果为目标进行训练，教师的错误（尤其是被过度抑制的语音）永远无法得到恢复，并且随着教师由学生更新而不断传播。我们提出了 SNOAR，这是一个在使用教师估计结果之前先对其进行校正的自训练框架：通过差分噪声过滤（DNF）去除泄漏到语音估计中的噪声，并将部分含噪观测信号加回，以恢复被抑制的语音并减弱伪影。我们证明这两种校正相互依赖，因为如果没有 DNF，该循环将崩溃（摘要此处截断）。

    arXiv:2610.20863v1 Announce Type: cross  Abstract: Unsupervised speech denoising methods such as MixIT and noisy-target training can be trained directly on noisy recordings, avoiding the sim-to-real gap of synthetic training data, but remain well below supervised performance. Teacher--student self-training, as in RemixIT, narrows this gap by remixing a teacher's estimates into new mixtures that supervise a student; however, since the student is trained only toward the teacher's estimates, teacher errors, in particular over-suppressed speech, are never recovered and propagate as the teacher is updated from the student. We propose SNOAR, a self-training framework that corrects the teacher's estimates before use: the noise leaked into its speech estimate is removed by differential noise filtering (DNF), and part of the noisy observation is added back in to restore suppressed speech and attenuate artifacts. We show that the two corrections depend on each other, as without DNF the loop coll
    
[^247]: 时间划分高斯过程混合模型

    Time-partitioning Gaussian Process Mixture Model

    [https://arxiv.org/abs/2610.20860](https://arxiv.org/abs/2610.20860)

    提出了一种贝叶斯非参数框架TPGPMM，通过自适应Pitman-Yor先验与特定状态高斯过程的结合，自动识别时间序列中的持续状态与变点，无需马尔可夫假设即可建模非平稳依赖，并能有效区分持续变化与瞬态冲击。

    

    我们提出了时间划分高斯过程混合模型（TPGPMM），这是一个用于识别时间序列中持续状态并建模非平稳依赖关系的贝叶斯非参数框架。该模型将连续分段上的自适应Pitman-Yor先验与特定状态的高斯过程相结合，而不施加状态内的马尔可夫依赖。通过学习Pitman-Yor折扣参数，模型能够适应同质和异质的状态长度。一种带有精确Gibbs边界更新的分裂-合并Metropolis-within-Gibbs采样器可联合学习变点和模型超参数，量化分段及状态内动态的不确定性。使用RBF核和固定的保守观测噪声水平，TPGPMM倾向于稀疏分段，能够区分持续变化与瞬态冲击。仿真实验证明了该模型对状态结构的恢复能力。该方法已应用于CBOE波动率…

    arXiv:2610.20860v1 Announce Type: cross  Abstract: We propose the Time-partitioning Gaussian Process Mixture Model (TPGPMM), a Bayesian nonparametric framework for identifying persistent regimes and modelling nonstationary dependence in time series. The model combines an adaptive Pitman--Yor prior on contiguous segmentations with regime-specific Gaussian processes, without imposing Markovian within-regime dependence. Learning the Pitman--Yor discount parameter allows adaptation to homogeneous and heterogeneous regime lengths.   A split--merge Metropolis-within-Gibbs sampler with exact Gibbs boundary updates jointly learns change points and model hyperparameters, quantifying uncertainty over segmentations and within-regime dynamics. With an RBF kernel and a fixed, conservative observation-noise level, TPGPMM favours sparse segmentations that distinguish persistent shifts from transient shocks. Simulations demonstrate recovery of the regime structure. Applications to the CBOE Volatility 
    
[^248]: 形态正确性对分词重要吗？——基于他加禄语（Tagalog）的乱序对照研究

    Does Morphological Correctness Matter for Tokenization? A Scrambled Control Study in Tagalog

    [https://arxiv.org/abs/2610.20859](https://arxiv.org/abs/2610.20859)

    本文以他加禄语的中缀语态为切入点，构建了将词根连续化输出并附加位置性中缀标记的分词器 MorphAustro-Tok，并通过保留词素数量、可逆性与一致性的乱序词素对照实验检验“词素边界必须语言学正确”这一假设，结果显示其相比匹配BPE在机器翻译中最多提升3.40个chrF++点。

    

    类型学感知的分词器的立论依据是：尊重词素边界的切分能够保留被基于频率的切分所破坏的信息。已有研究通常在假定边界必须语言学正确的前提下证明其有效性，而本文将这两个因素分离开来。他加禄语通过中缀标记语态（如 s-um-ulat“写了”、s-in-ulat“被写了”），因此词根并不连续，任何BPE合并序列都无法将其恢复。我们构建了 MorphAustro-Tok——一种将词根作为连续单元输出、后接位置性中缀标记的分词器，并将其与语料库匹配的BPE基线进行比较，同时在三个切分率（fertility）水平中的两个上，与一种“乱序词素”对照组进行比较：该对照在保持词素数量、可逆性和逐类型一致性的同时随机化伪中缀。在33次运行中，MorphAustro-Tok 在他加禄语机器翻译上相比匹配的BPE分别取得0.84、1.49和3.40个平均句级chrF++点的提升。Token co……（原文摘要在此处截断）

    arXiv:2610.20859v1 Announce Type: new  Abstract: Typology-aware tokenizers are motivated by the claim that segmentation which respects morpheme boundaries preserves information that frequency-driven segmentation destroys. Studies routinely establish that it helps while assuming the boundaries must be linguistically correct. We separate them. Tagalog marks voice with infixes (s-um-ulat 'wrote', s-in-ulat 'was written'), so the root is not contiguous and no sequence of BPE merges can recover it. We build MorphAustro-Tok, a tokenizer that emits the root as a contiguous unit followed by a positional infix token, and compare it against a corpus-matched BPE baseline and, at two of three fertility levels, against a scrambled-morpheme control that preserves morph count, invertibility and per-type consistency while randomising the pseudo-infix. Across 33 runs, MorphAustro-Tok improves Tagalog machine translation over matched BPE by 0.84, 1.49 and 3.40 mean sentence-level chrF++ points. Token co
    
[^249]: 完整还是稀疏：两种可辨识性的故事

    Complete or Sparse: A Tale of Two Identifiabilities

    [https://arxiv.org/abs/2610.20851](https://arxiv.org/abs/2610.20851)

    本文证明在生成式与判别式两种因果方向下，表示维度充足时最优编码器是潜在变量的线性函数，而结合非负激活与能量效率约束可使每个神经元恰好编码一个潜在变量，为“祖母神经元”现象提供了精确的理论解释。

    

    理解神经网络何时学习到可解释的表示，是机制可解释性研究的核心问题。我们通过可辨识性来研究这一问题：监督学习何时能够恢复数据背后真实的潜在变量？我们引入了两种以因果方向相区分的数据生成过程：一种是由标签导致潜在变量（生成式），另一种是由潜在变量导致标签（判别式）。当表示的维度不少于潜在空间的维度时，我们证明最优编码器是潜在变量的线性函数；相反，当维度不足时，可辨识性通常会失效，编码器变为非线性。随后我们进一步证明，额外的约束能够恢复单个潜在变量，而不仅仅是它们的线性混合。在生成式设定中，非负激活与能量效率迫使每个神经元恰好编码一个潜在变量，从而为“选择性祖母神经元”现象提供了精确的理论解释。

    arXiv:2610.20851v1 Announce Type: cross  Abstract: Understanding when neural networks learn interpretable representations is central to mechanistic interpretability. We study this through identifiability: when can supervised learning recover the true latent variables underlying the data? We introduce two data generating processes distinguished by their causal direction where either labels cause latents (generative) or latents cause labels (discriminative). When the representation has at least as many dimensions as the latent space, we prove that the optimal encoders are linear in the latents. Conversely, when dimensions are insufficient, identifiability generically fails and encoders become nonlinear. We then show that additional constraints yield recovery of individual latent variables, not just linear mixtures. In the generative setting, non-negative activations and energy efficiency force each neuron to encode exactly one latent, providing a precise account of selective grandmother 
    
[^250]: 扩散模型中的去噪分数匹配：秩约束对泛化的影响

    Denoising Score Matching in Diffusion Models: The Effect of Rank Constraints on Generalization

    [https://arxiv.org/abs/2610.20850](https://arxiv.org/abs/2610.20850)

    本文研究了扩散模型分数网络中秩约束对泛化性能的影响，通过随机特征模型的理论分析证明：增大秩能降低训练误差，但其对测试误差的影响取决于样本量与数据维度的比例关系，并借助截断奇异值分解给出了渐近精确的极限误差刻画。

    

    架构偏差限制了扩散分数网络所能表示的函数类别，但其对泛化的影响目前仍知之甚少。我们研究了具有秩约束非线性项和线性跳跃连接的分数网络上的去噪分数匹配问题。在嵌入球面上支持的数据上使用可训练三层网络进行的数值实验表明，增大秩能够降低训练误差，但其对测试误差的影响取决于样本量和数据维度。为解释这一现象，我们针对数据支持于低维子空间的高斯数据，分析了秩约束下的随机特征分数模型。我们通过截断奇异值分解刻画了最优秩约束估计器，并在样本量、特征宽度、数据维度和秩与环境维度成比例增长的条件下，推导出了渐近精确的极限训练误差和测试误差。

    arXiv:2610.20850v1 Announce Type: new  Abstract: Architectural biases restrict the class of functions that a diffusion score network can represent, but their effect on generalization remains poorly understood. We study denoising score matching with score networks having a rank constrained non-linear term and a linear skip connection. Numerical experiments with trainable three-layer networks on data supported on an embedded sphere show that increasing rank lowers training error, but its effect on test error depends on sample size and data dimension. To explain this behavior, we analyze a random-feature score model under rank constraint with Gaussian data supported on a low-dimensional subspace. We characterize the optimal rank-constrained estimator through a truncated singular value decomposition and derive asymptotically precise limiting training and test errors when sample size $(n)$, feature width $(p)$, data dimension $(D)$, and rank $(r)$ scale proportionally with the ambient dimen
    
[^251]: 面向无线网络中数据流与AI流融合的模拟射频计算

    Analog RF Computing for Integrated Data and AI Flows Over Wireless Networks

    [https://arxiv.org/abs/2610.20841](https://arxiv.org/abs/2610.20841)

    提出模拟射频计算方案，让基站组播编码神经网络权重的波形，使资源受限的边缘设备以极低功耗和内存完成AI推理，并研究了MU-MIMO下行链路中组播AI流与其他数据流共享频谱、功率和空间资源的问题。

    

    许多资源受限的边缘设备需要人工智能（AI）服务，但其自身性能不足以存储或执行神经网络（NN）。模拟射频（RF）计算通过让基站（BS）向不同客户端组播编码了神经网络权重的波形来解决这一限制。每个客户端随后通过其射频前端中的无源混频器，将接收到的权重编码波形与本地生成的输入编码波形相乘，从而以极低的功耗和内存需求实现AI推理所需的矩阵-向量乘法（MVM）。这可以使电信运营商向连接到其网络的边缘设备提供AI推理服务。然而在实际应用中，这种组播AI流必须与其他数据流共享频谱、功率和空间资源。本文考虑多用户多输入多输出（MU-MIMO）下行链路场景，并研究

    arXiv:2610.20841v1 Announce Type: cross  Abstract: Many resource-constrained edge devices require artificial intelligence (AI) services but are not powerful enough to store or execute neural networks (NNs). Analog radio frequency (RF) computing addresses this limitation by letting a base station (BS) multicast a waveform encoding NN weights to different clients. Each client then multiplies the received weight-encoded waveform with its locally generated input-encoded waveform through a passive mixer in its RF frontend, thereby realizing the matrix-vector multiplications (MVMs) needed for AI inference with an extremely low power and memory requirement. This can enable telecom operators to provide AI inference services to the edge devices connected to their networks. In practice, however, this multicast AI flow must share the spectrum, power, and spatial resources with other data flows. This paper considers a multi-user multiple-input multiple-output (MU-MIMO) downlink scenario and studie
    
[^252]: 视觉-语言-动作模型中神经几何的现象学研究

    A Phenomenology of Neural Geometry in Vision-Language-Action Models

    [https://arxiv.org/abs/2610.20840](https://arxiv.org/abs/2610.20840)

    该论文首次系统刻画了视觉-语言-动作模型内部任务相关概念的真实几何形态，发现其表征以多维、弯曲的形式存在，并由物理约束塑造（如关节限制使手腕姿态呈开放弧线、目标距离以“接近粗标尺”与“抓取细标尺”两个正交子空间编码），从而超越了仅证明变量“线性可读”的先前研究。

    

    机器人操作本质上是几何性的，但世界在执行操作的策略内部究竟呈现出什么样的形状？先前关于视觉-语言-动作（VLA）策略的研究已经证实，任务相关变量可以从激活中线性读出，然而可读性只能证明某个变量的存在，却无法说明其取值所描绘的形状——这些形状可能是弯曲的、多维的。通过受控扫描、自然轨迹回放、线性探针和激活干预，我们在VLA模型中构建了七个任务相关概念的分类体系。我们发现，实际呈现的几何结构以多维为主，并由策略行动时所受的约束所塑造。例如，物理关节限制使得手腕姿态呈现为一段开放弧线而非完整圆环；而到目标的距离这一单一标量，则在两个正交子空间中实现——一个是用于接近阶段的粗略标尺，另一个是用于抓取阶段的精细标尺。

    arXiv:2610.20840v1 Announce Type: cross  Abstract: Robotic manipulation is inherently geometric, but what shape does the world take inside the policy that performs it? Prior work on vision-language-action (VLA) policies establishes that task-relevant variables are linearly readable from activations, yet readability certifies only that a variable is present and says nothing about the shape its values trace, which may be curved and multi-dimensional. Using controlled sweeps, natural rollouts, linear probes, and activation interventions, we build a taxonomy of seven task-relevant concepts in VLA models. We find that realized geometry is predominantly multi-dimensional and is shaped by the constraints under which the policy acts. Physical joint limits, for example, leave wrist orientation an open arc rather than a circle, and distance to the target, a single scalar, is realized in two orthogonal subspaces, a coarse ruler for the approach and a fine ruler for the grasp. The same geometries 
    
[^253]: 面向评估器引导测试时推理的置信度校准推理扩展

    Confidence-Calibrated Inference Expansion for Evaluator-Guided Test-Time Reasoning

    [https://arxiv.org/abs/2610.20836](https://arxiv.org/abs/2610.20836)

    提出一个计算感知的三动作控制器，通过学习并校准动作特定的置信度估计来决定测试时推理中何时扩展计算，在提升效用的同时将在线计算量降低约 30% 和 16%。

    

    评估器引导的推理在每一步都面临一个局部控制问题：信任当前答案、返回回退结果，还是投入更多推理计算。我们将这一选择形式化为一个计算感知的三动作控制器：通过预测答案成功度对终止动作进行评分，通过基于 rollout 的后续成功替代指标对扩展动作进行评分，并显式地为计算成本定价。局部动作间隔分析刻画了评分误差何时可能改变决策，而期望遗憾界则考虑了沿实际扩展路径出现的误差。在该分析的指导下，我们学习并校准动作特定的置信度估计，并对模糊决策引入局部不确定性惩罚。在留出的宽度扩展和修订任务上，该控制器相比校准选择的固定策略提升了观测效用，同时将归一化在线计算量分别降低约 30% 和 16%。在四个 Qwen3-8B 随机种子上，计算量低于拟合的 Q…

    arXiv:2610.20836v1 Announce Type: new  Abstract: Evaluator-guided reasoning presents a local control problem at every step: trust the current answer, return a fallback, or spend more inference. We formulate this choice as a compute-aware three-action controller, scoring terminal actions by predicted answer success and expansion by a rollout-based continuation-success surrogate, with compute cost priced explicitly. A local action-gap analysis characterizes when score errors can change a decision, while an expected regret bound accounts for errors along the realized expansion path. Guided by this analysis, we learn and calibrate action-specific confidence estimates, with local uncertainty penalties for ambiguous decisions. On held-out width expansion and revision tasks, the controller improves observed utility over calibration-selected fixed policies while reducing normalized online compute by about 30% and 16%, respectively. Across four Qwen3-8B seeds, compute is lower than fitted Q in 
    
[^254]: 在推理大语言模型中摊销序贯实验设计

    Amortizing Sequential Experimental Design in Reasoning LLMs

    [https://arxiv.org/abs/2610.20832](https://arxiv.org/abs/2610.20832)

    该论文通过以信息增益奖励显式训练LLM的推理过程，将序贯贝叶斯实验设计摊销到LLM中，使其在实验设计任务中能够自适应地收集信息，并在二十个问题任务上超越了训练时和推理时的BED基线方法。

    

    尽管大语言模型（LLM）拥有广泛的预训练知识和强大的推理能力，但它们在实验设计任务中难以自适应地收集信息。我们研究了如何通过使用信息增益奖励显式训练引导动作选择的推理过程，将序贯贝叶斯实验设计（BED）摊销到LLM中。我们的方法独立于LLM自身的信念来指定实验生成模型，使我们能够在所选模型下优化期望信息增益。我们研究了摊销过程中的各项设计选择，包括不同的生成模型构建方式、奖励估计、假设空间规范、多步目标，以及相对于训练时和推理时BED基线的计算成本。在“二十个问题”任务上，所得策略改善了信息获取和任务表现，优于推理时和训练时的BED基线。

    arXiv:2610.20832v1 Announce Type: cross  Abstract: Despite their extensive pretraining knowledge and strong reasoning capabilities, large language models (LLMs) struggle to adaptively gather information in experimental design tasks. We study how to amortize sequential Bayesian experimental design (BED) into an LLM by explicitly training the reasoning that guides action selection using information-gain rewards. Our approach specifies the experimental generative model independently of the LLM's own beliefs, allowing us to optimize expected information gain under a model of our choosing. We investigate the design choices for amortization, including alternative generative-model constructions, reward estimation, hypothesis-space specification, multi-step objectives, and computational costs relative to training- and inference-time BED baselines. On Twenty Questions, the resulting policies improve information acquisition and task performance, outperform inference- and training-time BED baseli
    
[^255]: CoTranslator：面向推理蒸馏的从反向到正向思维链翻译器

    CoTranslator: Reverse to Forward Chain-of-Thought Translator for Reasoning Distillation

    [https://arxiv.org/abs/2610.20828](https://arxiv.org/abs/2610.20828)

    CoTranslator通过将反向思维链回答“翻译”为经过验证的正向回答分布，兼得低成本与高质量，实现大规模推理蒸馏数据生成，在六个数学基准上显著提升学生模型平均准确率最高5个百分点。

    

    推理蒸馏需要带有正确答案的思维链（CoT）数据，但如何高效获取此类数据仍具挑战性。正向CoT生成（FCG）结合拒绝采样（RS）可以产生经过验证的、仅基于问题的回答，但对于困难问题可能需要多次调用教师模型。反向CoT生成（RCG）成本更低，因为它以已知答案为条件，但能够访问答案会导致事后合理化，并与学生模型在无答案条件下的推理设置不匹配。我们提出CoTranslator，它学习将反向CoT回答转换（翻译）为经过验证的正向回答的分布。该方法仅需在一小组成对的反向回答与拒绝采样正向回答上训练，即可支持大规模的学生训练数据生成与验证。在六个学生模型和六个数学基准上的评估显示，与语义抑制方法相比，CoTranslator将平均准确率提升了最高5（原文此处被截断）

    arXiv:2610.20828v1 Announce Type: cross  Abstract: Reasoning distillation requires chain-of-thought (CoT) data with correct answers, but obtaining such data efficiently remains challenging. Forward CoT generation (FCG) followed by rejection sampling (RS) produces verified, question-only responses, yet may require many teacher calls for difficult problems. Reverse CoT generation (RCG) is cheaper because it conditions on the known answer, but access to this answer can lead to post-hoc rationalization and a mismatch with the student's answer-free inference setting. We propose CoTranslator, which learns to translate reverse CoT responses toward the distribution of verified forward responses. Trained on a small set of paired reverse and rejection-sampled forward responses, it supports large-scale generation and verification of student training data. Across six student models evaluated on six mathematics benchmarks, CoTranslator improves average accuracy over semantic suppression by up to $5
    
[^256]: 具有有界输入次数的多项式复合的线性无关性

    Linear Independence of Polynomial Compositions with Bounded Input Degree

    [https://arxiv.org/abs/2610.20825](https://arxiv.org/abs/2610.20825)

    本文通过有理曲线形变与Riemann-Roch定理论证解决了弱复合无关性猜想，进而证明了采用足够大次数多项式激活函数的神经网络的可识别性。

    

    我们证明了Kohn、Marchetti、Massarenti和Mella提出的弱复合无关性猜想。证明方法使用了有理曲线的形变以及线性化方程的单个核。Riemann-Roch定理通过具有受控极点的切向量场导出了矛盾。通过解决这些猜想，我们证明了具有足够大次数的一般多项式激活函数的神经网络是可识别的。

    arXiv:2610.20825v1 Announce Type: cross  Abstract: We prove the weak composition-independence conjectures of Kohn, Marchetti, Massarenti and Mella. The proof uses deformations of rational curves and a single kernel of the linearized equation. Riemann-Roch gives a contradiction from tangent fields with controlled poles. By settling these conjectures, we prove that a neural network with a general polynomial activation function of sufficiently large degree is identifiable.
    
[^257]: 学习之前的可行性上限：无人机网格拦截的规模与时间界限

    Feasibility Ceilings Before Learning: Sizing and Timing Bounds for Drone-Mesh Interception

    [https://arxiv.org/abs/2610.20823](https://arxiv.org/abs/2610.20823)

    本文推导了无人机拦截网格在策略选定之前的性能闭式界限，并证明当智能体间距至少为2r时，平面与体积覆盖率分别存在90.69%与74.05%的固有可行性上限。

    

    由拦截器组成的网格在防御固定阵地时，必须散布在逼近目标的预测到达点周围。这是多智能体机器人中反复出现的任务：N个智能体各自覆盖一个半径为r的圆盘，散布于某一概率密度之上，使得从中抽取的一个点落入某个智能体的圆盘之内。我们给出了在任何策略被选定之前约束性能的各量的闭式表达式，并通过蒙特卡洛方法逐一验证。通过对含噪线性轨迹外推所预测的点，其不确定性来自最小二乘方差传播，再加上针对不确定到达时间的delta方法项，与仿真结果的误差在1.8%以内。密集排列网格的覆盖概率具有绝对误差不超过0.007的闭式表达式，并存在对应的体积形式。朴素的面积比会遗漏一个上限：只要智能体之间保持至少2r的间距，平面内任何网格的覆盖率都不会超过90.69%，体积内则不超过74.05%。

    arXiv:2610.20823v1 Announce Type: cross  Abstract: A mesh of interceptors defending a fixed site must spread over the predicted arrival point of an approaching object. This is a recurring task in multi-agent robotics: N agents, each covering a disc of radius r, spread over a probability density so that a point drawn from it falls within some agent's disc. We give closed forms for the quantities that bound performance before any policy is chosen, and validate each by Monte Carlo. The uncertainty of a point predicted by extrapolating a noisy linear track follows from least-squares variance propagation plus a delta-method term for the uncertain arrival time, agreeing with simulation to within 1.8%. The coverage probability of a packed mesh admits a closed form accurate to 0.007 absolute, with a volumetric analogue. A naive area ratio misses a ceiling: while agents stay at least 2r apart, no mesh exceeds 90.69% coverage in the plane or 74.05% in a volume. This ceiling belongs to the separa
    
[^258]: 面向可验证奖励离策略强化学习的Gibbs间隙策略优化

    Gibbs-Gap Policy Optimization for Off-Policy RL with Verifiable Rewards

    [https://arxiv.org/abs/2610.20814](https://arxiv.org/abs/2610.20814)

    提出了Gibbs间隙策略优化（GGPO）——一种无需学习策略到行为策略重要性权重的离策略强化学习方法，它通过最小化同一提示下各响应经KL调整后的奖励的log-mean-exp对比进行训练，并从理论上保证学习策略的单响应期望奖励不低于行为策略的best-of-K奖励。

    

    带可验证奖励的强化学习通常使用由另一个可能过时的行为策略所生成的响应来训练语言模型。用从学习策略到行为策略的重要性权重来纠正这种不匹配，会给长响应带来高方差。我们提出了Gibbs间隙策略优化（GGPO），它在无需学习策略到行为策略重要性权重的情况下，最小化同一提示下各响应之间经KL调整后的奖励的log-mean-exp（对数-均值-指数）对比。我们从KL正则化策略优化中推导出GGPO：在同策略情形下，其总体损失恰好就是KL正则化的最优性间隙；并且在适当的覆盖条件下，学习策略的单响应期望奖励至少不低亍行为策略的best-of-K奖励，只需减去原始有限组损失、正则化以及有限训练数据所带来的相应项。我们通过精确的Gibbs目标复合来刻画参考策略的刷新机制，并在局部参考……条件下给出了有限数据下最终策略的奖励上界。（摘要原文在此处被截断）

    arXiv:2610.20814v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards often trains language models on responses generated by a different, possibly stale behavior policy. Correcting this mismatch with learner-to-behavior importance weights can introduce high variance for long responses. We introduce Gibbs-Gap Policy Optimization (GGPO), which minimizes a log-mean-exp contrast of KL-adjusted rewards among responses to the same prompt, without learner-to-behavior importance weights. We derive GGPO from KL-regularized policy optimization: on policy, its population loss is exactly the KL-regularized optimality gap, and under suitable coverage the learner's single-response expected reward is at least the behavior policy's best-of-$K$ reward, minus terms for the raw finite-group loss, regularization, and finite training data. We characterize reference refresh through exact Gibbs-target composition and a finite-data final-policy reward bound under local reference co
    
[^259]: 高维持续线性回归中的最优显式与隐式正则化强度

    Optimal Explicit and Implicit Regularization Strengths in High-Dimensional Continual Linear Regression

    [https://arxiv.org/abs/2610.20804](https://arxiv.org/abs/2610.20804)

    本文推导出高维持续线性回归中每任务固定步数梯度下降的泛化损失闭式表达式，证明最优固定学习率应随任务数按 $\ln T/T$ 衰减，此时泛化性能渐近等价于最优调节的 L2 正则化，并将结果推广至对角广义岭回归并给出最优正则化的充要条件。

    

    我们研究了高维情形下的持续线性回归，其中学习器观察一个任务序列并依次更新其参数。在这一设定下，我们推导出了每任务以固定步数进行梯度下降训练的模型的泛化损失的显式闭式表达式，该表达式对任意教师模型均成立。我们证明，对于单个教师模型以及更一般的独立同分布教师模型，最优固定步长随任务数量近似反比衰减，具体为 $\ln T / T$。在这一缩放律下，我们证明所得到的泛化损失与经过最优调节的 L2（各向同性）正则化所达到的泛化损失渐近等价。此外，我们将分析扩展到具有对角正则化矩阵的广义岭回归，得到了泛化损失的闭式刻画，并给出了最优……（成立）的充分必要条件。

    arXiv:2610.20804v1 Announce Type: new  Abstract: We study continual linear regression in a high-dimensional regime, where a learner observes a sequence of tasks and updates its parameters sequentially. In this setting, we derive an explicit closed-form expression for the generalization loss of a model trained with gradient descent for a fixed number of steps per task, which holds for arbitrary teachers. We prove that the optimal fixed step size decays nearly inversely with the number of tasks, specifically as $\ln T / T$, for a single teacher and, more generally, i.i.d. teachers. Under this scaling, we show that the resulting generalization loss is asymptotically equivalent to that achieved by optimally tuned L2 (isotropic) regularization. Furthermore, we extend our analysis to generalized ridge regression with diagonal regularization matrices, obtaining a closed-form characterization of the generalization loss and identifying necessary and sufficient conditions under which the optimal
    
[^260]: 面向健康错误信息分类与语义检索的隐私保护文本推理

    Privacy-Preserving Text Inference for Health-Misinformation Classification and Semantic Retrieval

    [https://arxiv.org/abs/2610.20796](https://arxiv.org/abs/2610.20796)

    本文提出了一种基于CKKS同态加密的隐私保护文本推理框架，将BioBERT、MPNet与TF-IDF等特征融合用于健康错误信息分类（准确率达0.971），并结合语义指纹实现密文域上的语义检索。

    

    健康错误信息分析面临双重挑战：误导性言论可能影响公众健康福祉，而用于检测这些言论的系统可能会暴露用户提交的敏感文本。这促使人们为公共卫生应用设计隐私保护推理，其中用户查询可能包含机密医疗信息，而训练好的模型和参考数据库则保存在组织服务器上。我们设计了一个基于CKKS同态加密的隐私保护文本推理框架，用于健康错误信息分类和语义检索。分类管线融合了BioBERT嵌入、经PCA降维的MPNet嵌入以及经SVD降维的TF-IDF特征，而检索管线则通过后期融合将MPNet嵌入与ISCC-SCT语义指纹相结合。我们评估了预测性能，并对明文执行与基于密文的执行进行了基准测试。混合分类器达到了0.971的准确率，并且

    arXiv:2610.20796v1 Announce Type: cross  Abstract: Health-misinformation analysis presents a dual challenge: misleading claims can affect public well-being, while systems designed to detect them may expose sensitive user-submitted text. This motivates privacy-preserving inference for public-health applications in which user queries may contain confidential medical information and trained models and reference databases are maintained on organizational servers. We design a privacy-preserving text inference framework based on CKKS homomorphic encryption for health-misinformation classification and semantic retrieval. The classification pipeline fuses BioBERT embeddings, PCA-reduced MPNet embeddings, and SVD-reduced TF-IDF features, while the retrieval pipeline combines MPNet embeddings with ISCC-SCT semantic fingerprints through late fusion. We evaluate predictive performance and benchmark plaintext versus ciphertext-based execution. The hybrid classifier achieved an accuracy of 0.971 and
    
[^261]: 基于脑电图的音乐享受预测：使用深度学习与可解释的时间分析

    EEG-Based Prediction of Music Enjoyment Using Deep Learning with Explainable Temporal Analysis

    [https://arxiv.org/abs/2610.20795](https://arxiv.org/abs/2610.20795)

    该论文将音乐享受预测从传统的分类任务转化为回归问题，在公开EEG数据集上比较了六种深度学习模型，并结合可解释性技术进行时间维度的神经反应分析。

    

    理解听众为什么喜欢音乐很困难，因为这种体验是主观的，并且会随着乐曲的展开而变化。脑电图（EEG）提供了一种非侵入性的方式来研究聆听过程中的神经反应，然而早期的研究大多将音乐情感视为分类标签，或依赖于黑盒模型。我们将音乐享受度预测形式化为公开MUSIN-G数据集上的一个回归问题，该数据集包含20名听众的128通道脑电数据以及歌曲层面的享受度评分。研究在参与者级别的协议下比较了六种深度学习模型，其中15名听众用于训练和验证，5名听众被保留用于测试。DeepConvNet、CNN-LSTM和GNN-CNN达到了最低的平均绝对误差（在1到5的量表上为0.97至0.99），而时间卷积网络（TCN）与评分的皮尔逊相关性最高（r = 0.21）。基于梯度和基于遮挡的解释方法，通过平均……（摘要在此处截断）

    arXiv:2610.20795v1 Announce Type: new  Abstract: Understanding why listeners enjoy music is difficult because the experience is subjective and changes as a piece unfolds. Electroencephalography (EEG) offers a non-invasive way to study neural responses during listening, yet earlier work has mostly treated music emotion as a categorical label or relied on black-box models. We formulate enjoyment prediction as a regression problem on the public MUSIN-G dataset, which contains 128-channel EEG from 20 listeners together with song-level enjoyment ratings. Six deep learning models were compared under a participant-wise protocol, in which 15 listeners were used for training and validation and 5 were held out for testing. DeepConvNet, CNN-LSTM and GNN-CNN reached the lowest mean absolute error (0.97 to 0.99 on the 1 to 5 scale), while the temporal convolutional network (TCN) had the highest Pearson correlation with the ratings (r = 0.21). Gradient-based and occlusion-based explanations, average
    
[^262]: 带误差证书的切片 Wasserstein 草图表示

    Sliced Wasserstein Sketches with Error Certificates

    [https://arxiv.org/abs/2610.20788](https://arxiv.org/abs/2610.20788)

    该论文提出一种基于切片 Wasserstein 距离的紧凑草图表示方法，通过将平方距离误差精确分解为残差与划分失配两项，为任意质心草图提供可计算的误差证书，并证明最优草图在温和矩条件下可达到 O(M^{-1}) 的误差率。

    

    支持几何上有意义比较的紧凑表示，对于从概率分布中学习非常有价值。我们研究了在二次 Wasserstein 距离下，一维概率测度的最优、自由权重的 M 原子量化问题，并通过切片方法将其推广到更高维度。使用 L 个共享投影可以得到比较时间为 O(LM) 的草图表示。我们的主要结果是将平方距离误差精确分解为残差项与划分失配项之和，从而为任意质心草图提供可计算的误差证书。对于最优草图，只要存在某个 δ>0 使得 (2+δ) 阶矩有限，即可保证有限方向切片距离下 O(M^{-1}) 的误差；而等质量草图的收敛速度可能更慢。每个切片均保持其均值，同时通过存储残差能量，可以从足够多的一般性方向恢复协方差。在文档、点云和流式细胞术基准数据集上的实验验证了方法的有效性。

    arXiv:2610.20788v1 Announce Type: new  Abstract: Compact representations that support geometrically meaningful comparisons are valuable for learning from probability distributions. We study optimal, free-weight $M$-atom quantization of one-dimensional probability measures in quadratic Wasserstein distance and extend it to higher dimensions via slicing. Using $L$ shared projections yields a sketch with $O(LM)$ comparison time. Our main result is an exact decomposition of squared distance error into residual and partition-mismatch terms, yielding computable certificates for arbitrary centroid sketches. For optimal sketches, a finite $(2+\delta)$ moment for some $\delta>0$ guarantees $O(M^{-1})$ error in the finite-direction sliced distance, while equal-mass sketches can converge more slowly. Each slice preserves its mean, while storing residual energies enables covariance recovery from sufficiently many generic directions. Experiments on document, point-cloud, and flow-cytometry benchmar
    
[^263]: Verdict：一次编码，多次决策

    Verdict: Encode Once, Decide Many

    [https://arxiv.org/abs/2610.20778](https://arxiv.org/abs/2610.20778)

    Verdict-MQ 提出一种“一次编码、多问题并行回答”的开源编码器，通过结构化注意力掩码实现状态编码与问题解耦、问题互不干扰及选项顺序无关，在类型化决策基准上以更小参数量在所有随机种子上显著超越专用模型。

    

    arXiv:2610.20778v1 公告类型：cross 摘要：自动化组织工作意味着对同一份状态数据回答许多小型的类型化问题（这封邮件是否需要回复、紧急程度如何、由谁负责），且其概率要好到足以在无需人工的情况下直接执行。我们对比的开源类型化决策模型，如同标准的交叉编码器，对每个问题都需要对输入进行一次单独的完整遍历。我们提出 Verdict-MQ，一种开源编码器，只需读取案例一次即可回答其所有问题：在截断后保留完全相同的状态标记的前提下（每个报告的案例均已验证），结构化注意力掩码使状态编码独立于问题、避免问题之间相互影响，并通过构造使选项顺序不产生任何影响。在类型化决策基准上，395M 参数的 Verdict-MQ 在全部五个全新随机种子上均显著优于已发布的 421M Laya 专用模型，与软金标准标签的匹配度更高（KL 0.080 对 0.117，Brier 0.045 对 0.061），这一优势……

    arXiv:2610.20778v1 Announce Type: cross  Abstract: Automating organizational work means answering many small typed questions about one piece of state (does this email need a reply, how urgent is it, who owns it) with probabilities good enough to act on without a human. The open typed-decision model we compare against, like a standard cross-encoder, answers each question with a separate pass over the input. We present Verdict-MQ, an open encoder that reads a case once and answers all of its questions: given identical retained state tokens after truncation (verified on every reported case), a structured attention mask makes the state encoding independent of the questions, keeps questions from influencing each other, and makes option order irrelevant by construction. On the typed-decisions benchmark, a 395M Verdict-MQ matches the soft gold labels significantly better than the released 421M Laya specialist on all five fresh seeds (KL 0.080 vs 0.117, Brier 0.045 vs 0.061), an advantage that
    
[^264]: 选择安全的保形早退：自适应深度下的无分布覆盖保证

    Selection-Safe Conformal Early Exit: Distribution-Free Coverage under Adaptive Depth

    [https://arxiv.org/abs/2610.20773](https://arxiv.org/abs/2610.20773)

    该论文提出一种选择安全的保形早退方法，通过前缀最大值构造的终端路径分数和单一保形分位数校准，在任意数据依赖的提前退出策略下均能保证无分布覆盖率，无需 alpha/L 修正，且实验表明其覆盖率显著优于朴素的逐层选择方法。

    

    在每个早退点分别校准保形预测集，一旦选择了依赖于数据的退出点，便无法保持覆盖率。我们从各层前缀最大值构造一个终端路径分数，校准单一的分隔保形分位数，并返回全部包含有效终端集的嵌套前缀集。该保证无需 alpha/L 修正即具有选择安全性，并且同时适用于所有在校准之后仍保持终端集包含关系的无标签停止策略。我们还通过早退分数超出终端分数部分的经验尾部，界定了路径最大值带来的效率损失。在八个编码器任务-主干组合的实验中，路径级选择方法达到 0.901 的宏覆盖率，而朴素的逐层选择仅为 0.814，同时平均集合大小为 1.41，延迟比例为 0.71。在 CIFAR-100 上使用具有 24 个退出点的 ViT-L/16（n_cal=500）时，预算 b=4 下的首次满足条件停止策略达到覆盖率 0.902、集合大小 2.76 和延迟比例 0.57，且同样的（摘要在此处被截断）

    arXiv:2610.20773v1 Announce Type: new  Abstract: Calibrating a conformal prediction set at each early exit does not preserve coverage once a data-dependent exit is selected. We construct one terminal path score from prefix maxima, calibrate a single split-conformal quantile, and return nested prefix sets that all contain the valid terminal set. The guarantee is selection-safe without an alpha/L correction and holds simultaneously for every post-calibration label-free stopping policy that preserves terminal-set containment. We also bound the efficiency cost of the path maximum by the empirical tail of the early-over-terminal score excess. Across eight encoder task-backbone cells, pathwise selection attains macro coverage 0.901 versus 0.814 for naive layerwise selection, with average set size 1.41 and latency fraction 0.71. On CIFAR-100 with a 24-exit ViT-L/16 (n_cal=500), first-eligible stopping at budget b=4 attains coverage 0.902, set size 2.76, and latency fraction 0.57, and the same
    
[^265]: 当现实超越认知：可靠智能的新缩放原则

    When Reality Outpaces Cognition: A New Scaling Principle for Reliable Intelligence

    [https://arxiv.org/abs/2610.20764](https://arxiv.org/abs/2610.20764)

    本文提出了一种新的缩放原则——通过“有效连贯认知速度”量化智能系统在现实变化速度超出训练基线最高50倍时仍能可靠感知、推理和行动的能力极限。

    

    本文通过提出一个根本性问题，挑战了现有AI范式和新兴世界模型的理解能力：现实可以以多快的速度变化，以至于智能系统无法再可靠地感知、推理和行动？我们将此表述为一个与理论和模型无关的挑战，即系统必须在不断扩大的观察—现实差距中，为越来越遥远的未来状态推断出可靠的行动。我们通过“有效连贯认知速度”（\(S_C^{\mathrm{eff}}\)）将这一挑战可操作化，其定义为：系统从基线出发，仍能持续满足预定行为可靠性标准的最高已测试“现实速度”（\(S_R\)）。作为概念验证，强化学习智能体在基线\(S_R\)下进行训练，随后在不重新训练、计算资源保持固定的条件下，在\(S_R\)提升至最高50倍的情况下进行评估。我们比较了Transformer和g……（原文摘要在此截断）

    arXiv:2610.20764v1 Announce Type: new  Abstract: This paper challenges the understanding capabilities of existing AI paradigms and emerging world models by asking a fundamental question: How quickly can reality change before an intelligent system can no longer reliably perceive, reason, and act? We formulate this as a theory- and model-agnostic challenge in which systems must infer reliable actions for increasingly distant future states across a growing observation--reality gap. We operationalize this challenge through the effective Speed of Coherent Cognition (\(S_C^{\mathrm{eff}}\)), defined as the highest tested Speed of Reality (\(S_R\)) through which a system continuously satisfies a predefined behavioural reliability criterion from baseline. As a proof of concept, reinforcement-learning agents are trained at the baseline \(S_R\) and evaluated, without retraining, as \(S_R\) is increased by up to \(50\times\) while computational resources remain fixed. We compare Transformer and g
    
[^266]: 先净化再对齐：改进扩散模型的表征对齐

    Purify Before You Align: Improving Representation Alignment for Diffusion Models

    [https://arxiv.org/abs/2610.20761](https://arxiv.org/abs/2610.20761)

    提出局部外观净化（LAP）方法，通过在对齐前去除目标表征中局部可预测的内容、仅对齐残差信息，显著提升扩散模型的生成质量，且无需改动训练或推理流程。

    

    表征对齐通过鼓励扩散模型的中间特征去匹配预训练视觉编码器，从而改进并加速扩散模型的训练。然而，标准的对齐方式只是直接匹配预训练表征，并未区分哪些信息对生成任务可能更有用。我们提出了局部外观净化，该方法通过去除局部可预测的内容，使模型对齐到剩余的残差信息，从而提升生成质量，并使诸如 MAE 这类编码器成为更有效的对齐目标。LAP 仅修改对齐目标，无需对扩散模型的训练或推理过程做任何改动。通过在固定编码器内进行受控干预实验，我们证明了对齐的有效性取决于目标所强调的信息内容，而不是它所包含的图像信息总量。我们的理论为标准对齐提供了变分解释，并且对于固定的局部预测器，给出了以（内容截断）为条件的逐 token 信息界的总和。

    arXiv:2610.20761v1 Announce Type: cross  Abstract: Representation alignment improves and accelerates diffusion training by encouraging intermediate features to match a pretrained visual encoder. However, standard alignment matches pretrained representations without distinguishing what information may be more useful for generation. We introduce local appearance purification (LAP), which improves generation quality and makes encoders such as MAE more effective alignment targets by removing locally predictable content and aligning to the residual. LAP modifies only the alignment target, and requires no changes to diffusion training or inference. Through controlled interventions within a fixed encoder, we show that effective alignment depends on what information the target emphasizes, not simply how much image information it contains. Our theory gives a variational interpretation of standard alignment and, for fixed local predictors, a sum of token-wise information bounds conditioned on lo
    
[^267]: Optimal Hydra：使用数据感知的Hydra-Nets实现大语言模型的模型异构联邦学习

    Optimal Hydra: Model Heterogeneous Federated Learning for LLMs using Data-Informed Hydra-Nets

    [https://arxiv.org/abs/2610.20753](https://arxiv.org/abs/2610.20753)

    该论文提出Optimal Hydra方法，通过根据客户端本地数据分布定制异构的Hydra-Nets子网络架构，解决了资源受限客户端上大语言模型联邦微调的难题。

    

    在海量数据上训练的大型语言模型（LLMs）在一系列任务中展现出令人印象深刻的性能，但未来的性能提升将需要越来越多的数据和计算资源。联邦学习（FL）支持在分散的私有数据上进行训练，但客户端的资源限制可能使训练变得不切实际，尤其是随着模型规模的持续扩大。这些资源限制使得共享的服务器-客户端架构假设变得难以实现。模型异构联邦学习通过将模型大小适配到客户端能力，提供了一种有前景的替代方案；然而，以往的工作仅限于预训练场景，且关于子网络选择的结论不一定适用于微调。此外，先前的方法在创建客户端子网络时没有考虑其独特的数据分布，从而忽视了将子网络架构与本地数据相匹配的机会。受证据启发……（摘要原文在此处截断）

    arXiv:2610.20753v1 Announce Type: new  Abstract: Large language models (LLMs) trained on vast amounts of data have demonstrated impressive performance over a range of tasks, but future performance gains will require increasing amounts of data and compute. Federated Learning (FL) facilitates training on decentralized private data, but client resource limitations can make training impractical, particularly as model sizes continue to expand. These resource constraints make the assumption of a shared server-client architecture impractical. Model heterogeneous FL offers a promising alternative by adapting model size to client capabilities; however, previous work is limited to the pre-training setting, and conclusions on sub-network selection do not necessarily hold for fine-tuning. Furthermore, prior approaches create client sub-networks without considering their unique data distributions, thereby overlooking the opportunity to tailor sub-network architecture to local data. Motivated by evi
    
[^268]: 解读语言模型参数

    Interpreting Language Model Parameters

    [https://arxiv.org/abs/2610.20748](https://arxiv.org/abs/2610.20748)

    提出了对抗性参数分解（VPD）方法，将语言模型参数分解为各自仅实现模型算法一小部分的简单子组件，从而帮助理解神经网络参数如何执行计算。

    

    arXiv:2610.20748v1 公告类型： cross。摘要：机制可解释性旨在揭示神经网络如何利用其参数来实现其令人印象深刻的神经算法。尽管先前的研究已经在网络所使用的中间表示中发现了大量的结构，但在理解网络的参数和非线性如何对这些表示执行计算方面，进展却甚少。在这项工作中，我们提出了一种方法，使我们更接近这一理解目标：该方法将语言模型的参数分解为若干子组件，每个子组件仅实现模型所学算法的一小部分，同时只需其中一小部分子组件即可解释网络在任意输入上的行为。该方法——对抗性参数分解（VPD）——通过优化将神经网络参数分解为简单的子组件，即使在许多子组件……（原文摘要在此处截断）

    arXiv:2610.20748v1 Announce Type: cross  Abstract: Mechanistic interpretability aims to uncover how neural networks use their parameters to implement their impressive neural algorithms. Although previous work has uncovered substantial structure in the intermediate representations that networks use, little progress has been made to understand how the parameters and nonlinearities of networks perform computations on those representations. In this work, we present a method that brings us closer to this understanding by decomposing a language model's parameters into subcomponents that each implement only a small part of the model's learned algorithm, while simultaneously requiring only a small fraction of those subcomponents to account for the network's behavior on any input. The method, adVersarial Parameter Decomposition (VPD), optimizes for decompositions of neural network parameters into simple subcomponents that preserve the network's input-output behavior even when many subcomponents
    
[^269]: 利用随机归一化流实现向连续极限的遍历拓扑采样

    Ergodic topology sampling toward the continuum with Stochastic Normalizing Flows

    [https://arxiv.org/abs/2610.20747](https://arxiv.org/abs/2610.20747)

    本研究通过结合开放边界条件与随机归一化流的计算策略，提供了数值证据表明在细至 0.03 fm 的格距下可有效缓解 SU(3) 杨-米尔斯理论中的拓扑冻结问题。

    

    在本工作中，我们提供了数值证据，表明基于流的方法可以缓解 SU(3) 杨-米尔斯理论中的拓扑冻结问题，并在细至 0.03 fm 的格距上进行了测试。我们通过分析一种将开放边界条件与随机归一化流相结合的计算策略的成本标度行为来做到这一点。结果表明，在现有计算资源下，拓扑荷平方的积分自关联时间以及拓扑磁化率的统计不确定性在整个探索的参数范围内均保持在可控水平。这标志着在使用更高效的基于流的架构对最先进格点规范理论计算中的拓扑进行遍历采样方面迈出了重要一步。

    arXiv:2610.20747v1 Announce Type: cross  Abstract: In this work we provide numerical evidence that topological freezing in $\mathrm{SU}(3)$ Yang-Mills theory can be mitigated with a flow-based approach, as tested at lattice spacings as fine as 0.03~fm. We do so by analyzing the scaling of the costs of a computational strategy based on the combination of open boundary conditions and Stochastic Normalizing Flows. We show that the integrated autocorrelation time of the squared topological charge and the statistical uncertainty on the topological susceptibility remain under control throughout the explored parameter range with currently available computational resources. This marks an important step toward the use of even more efficient flow-based architectures for ergodic sampling of topology in state-of-the-art lattice gauge theory calculations.
    
[^270]: 面向元测度的垂直切片 Wasserstein 距离

    Vertical Sliced Wasserstein Distances for Meta-Measures

    [https://arxiv.org/abs/2610.20743](https://arxiv.org/abs/2610.20743)

    本文提出垂直切片 Wasserstein（VSW）距离，一种面向元测度空间 $\mathcal{P}(\mathcal{P}(\Omega))$ 的可扩展度量，通过连续测试函数投影内层测度实现高效计算，并证明了其度量性质与维度无关的统计收敛保证。

    

    许多现代学习问题需要比较这样的概率分布：其单个观测本身即为概率测度，这自然产生了 $\mathcal{P}(\mathcal{P}(\Omega))$ 中的元测度。然而，在该空间上评估最优传输距离的代价可能高得难以承受。我们引入了垂直切片 Wasserstein 距离（$\mathrm{VSW}$），这是 $\mathcal{P}(\mathcal{P}(\Omega))$ 上的一种可扩展度量。$\mathrm{VSW}$ 并非沿 $\mathcal{P}(\Omega)$ 的传输几何进行切片，而是使用连续测试函数对每个内层测度进行投影，并用一维 Wasserstein 距离比较所得的一维分布。我们证明了 $\mathrm{VSW}$ 是 $\mathcal{P}(\mathcal{P}(\Omega))$ 上的有效度量，能够度量化弱收敛，并具有与维度无关的统计收敛保证。实验表明，与现有方法相比，$\mathrm{VSW}$ 具有更高的精度和效率。

    arXiv:2610.20743v1 Announce Type: new  Abstract: Many modern learning problems require comparing probability distributions whose individual observations are themselves probability measures, naturally giving rise to meta-measures in $\mathcal{P}(\mathcal{P}(\Omega))$. However, optimal transport distances on this space can be prohibitively expensive to evaluate. We introduce the vertical sliced Wasserstein ($\mathrm{VSW}$) distance, a scalable metric on $\mathcal{P}(\mathcal{P}(\Omega))$. Rather than slicing along the transport geometry of $\mathcal{P}(\Omega)$, $\mathrm{VSW}$ projects each inner measure using continuous test functions and compares the resulting one-dimensional distributions with one-dimensional Wasserstein distance. We prove that $\mathrm{VSW}$ is a valid metric on $\mathcal{P}(\mathcal{P}(\Omega))$, metrizes weak convergence, and admits dimension-independent statistical convergence guarantees. Empirically, $\mathrm{VSW}$ provides high accuracy and efficiency compared w
    
[^271]: 面向概率时间序列预测的分位数引导扩散模型

    Quantile-Guided Diffusion for Probabilistic Time-Series Forecasting

    [https://arxiv.org/abs/2610.20739](https://arxiv.org/abs/2610.20739)

    提出Control-TS，在冻结的预训练扩散主干上学习分位数残差控制通路，将预测分位数作为分布控制来引导扩散预测，在八个基准上持续提升CRPS并达到与最先进模型相当的精度。

    

    分位数预测在每个未来时间步上总结不确定性，而扩散预测器则通过采样的未来轨迹来表示不确定性。我们研究预测分位数如何可以作为分布控制手段来引导扩散预测。具体而言，我们探讨了分位数整合的机制如何影响其在引导预训练扩散预测器方面的有效性。我们发现，在整个去噪网络中引入残差控制能够带来最一致的改进。我们将该方法实例化为Control-TS，它在保持预训练主干网络冻结的同时学习一个分位数控制通路。在八个基准测试中，Control-TS持续提升了主干网络的CRPS，取得了与最先进预测模型相当的性能，并在容量匹配的比较中保持了其优势。进一步的分析考察了这些性能提升如何依赖于分位数粒度、预测器选择以及预测质量。

    arXiv:2610.20739v1 Announce Type: new  Abstract: Quantile forecasts summarize uncertainty at each future time step, while diffusion forecasters represent it through sampled future trajectories. We study how predicted quantiles can serve as distributional controls for diffusion forecasting. Specifically, we investigate how the mechanism of quantile integration affects its effectiveness in guiding a pretrained diffusion forecaster. We find that residual control throughout the denoising network yields the most consistent improvements. We instantiate this approach as Control-TS, which learns a quantile-control pathway while keeping the pretrained backbone frozen. Across eight benchmarks, Control-TS consistently improves the backbone's CRPS, achieves competitive performance against state-of-the-art forecasting models, and retains its gains in capacity-matched comparisons. Further analyses examine how these gains depend on quantile granularity, predictor choice, and forecast quality. Finally
    
[^272]: 扩散语言模型的在策略残差学习

    On-Policy Residual Learning for Diffusion Language Models

    [https://arxiv.org/abs/2610.20687](https://arxiv.org/abs/2610.20687)

    该论文提出ORL框架，发现残差学习本质上是在策略的，通过熵收缩目标实时学习模型自身诱导的残差轨迹，并结合异步残差捕获、块扩散注意力等算法-系统协同设计，实现扩散语言模型速度与质量的高效平衡。

    

    残差上下文是在去噪步骤间传递预测分布、从而解锁扩散语言模型（DLM）速度-质量帕累托前沿的关键。然而，将这一看似简单的机制转化为有效的学习范式并非易事：目前尚无明确的方案指导残差上下文应如何学习。我们提出了ORL，一个面向扩散语言模型在策略残差学习的框架。在算法层面，我们发现残差学习本质上是在策略的。ORL实时地从模型自身诱导的残差轨迹中学习，并引入了一个熵收缩目标，使模型在重新审视自身预测时逐渐锐化预测结果。在系统层面，精确的残差监督需要捕获、保留并重新处理大量依赖轨迹的状态。ORL将算法与系统协同设计：采用异步残差捕获与传输、专用的块扩散注意力机制，以及打包的计算均衡训练。

    arXiv:2610.20687v1 Announce Type: new  Abstract: Residual context is key to unlocking the speed-quality Pareto frontier of diffusion language models (DLMs) by carrying forward predictive distributions across denoising steps. Yet turning this seemingly simple mechanism into an effective learning paradigm is far from simple: there is no clear recipe for how residual context should be learned. We introduce ORL, a framework for on-policy residual learning in DLMs. Algorithmically, we identify that residual learning is inherently on-policy. ORL learns from model-induced residual trajectories on the fly and introduces an entropy-contraction objective that sharpens predictions as the model revisits its own predictions. System-wise, exact residual supervision requires capturing, retaining, and reprocessing many trajectory-dependent states. ORL co-designs the algorithm with asynchronous residual capture and transport, specialized block-diffusion attention, and packed compute-balanced training, 
    
[^273]: RoLaQ：面向多头潜在注意力中2比特KV缓存的旋转与量化范围联合校准

    RoLaQ: Joint Rotation and Range Calibration for 2-Bit KV Caches in Multi-Head Latent Attention

    [https://arxiv.org/abs/2610.20686](https://arxiv.org/abs/2610.20686)

    提出RoLaQ离线校准方法，通过在多头潜在注意力模型中联合学习共享旋转和相互耦合的量化范围，实现2比特KV缓存量化，在多个模型和基准测试上取得了2比特方法中最高的平均准确率。

    

    多头潜在注意力（MLA）将键值缓存的内容压缩为每个token共享的潜在表示，使得低比特存储对长上下文推理极具吸引力。在2比特量化下，该表示中的误差会同时干扰注意力匹配和数值读取，其影响取决于误差方向和重建尺度。我们提出了RoLaQ（旋转校准的潜在量化），这是一种用于联合旋转与量化范围校准的离线方法。RoLaQ在由冻结的键投影和值投影所定义的子空间内细化共享旋转，并联合学习量化范围，其阈值与重建增益相互耦合，以控制共享信号的响应。在模型权重冻结的条件下，通过预测匹配和注意力输出监督来引导校准过程。在三个MLA模型和六个基准测试上，RoLaQ无论是否使用token保护，均在所比较的2比特方法中取得最高的平均准确率。

    arXiv:2610.20686v1 Announce Type: new  Abstract: Multi-head latent attention (MLA) compresses the content of a key-value cache into a shared latent per token, making low-bit storage attractive for long-context inference. At two bits, errors in this representation perturb both attention matching and value readout, with effects that depend on error direction and reconstruction scale. We introduce RoLaQ (Rotation-calibrated Latent Quantization), an offline method for joint rotation and range calibration. RoLaQ refines a shared rotation within subspaces defined by the frozen key and value projections, and jointly learns quantization ranges whose thresholds and reconstruction gains are coupled to control the shared signal's response. Prediction matching and attention-output supervision guide calibration with model weights frozen. Across three MLA models and six benchmarks, RoLaQ achieves the highest average accuracy among the compared 2-bit methods both without and with token protection. On
    
[^274]: 面向力学相关TPMS超材料的拓扑感知表示学习

    Learning Topology-Aware Representations for Mechanics-Relevant TPMS Metamaterials

    [https://arxiv.org/abs/2610.20684](https://arxiv.org/abs/2610.20684)

    提出将持续同调融入DMAE骨干网络的拓扑感知表示学习框架，通过持续性引导的掩码策略和辅助持续同调解码器显式编码TPMS超材料的全局结构连通性，从而更好地支持力学分析与结构设计。

    

    学习力学三重周期极小曲面（TPMS）超材料的信息性表示对于力学分析和结构设计十分重要，然而现有方法很少利用与力学响应相关的全局结构连通性信息来显式监督所学习到的表示。我们提出了一种拓扑感知的表示学习框架，将持续同调融入DMAE骨干网络，以显式编码全局结构连通性。一种由持续性引导的掩码策略优先关注与持续拓扑特征相关的图像块，同时一个辅助的持续同调解码器促使潜在表示保留关于连通分量、隧道和空洞等拓扑特征数量的信息。我们在两个互补的TPMS数据集上，通过家族分类、力学性能分类、无监督等任务评估了所学习到的表示。

    arXiv:2610.20684v1 Announce Type: new  Abstract: Learning informative representations of mechanical Triply Periodic Minimal Surface (TPMS) metamaterials is important for mechanics analysis and structural design, yet existing methods have rarely explicitly supervised learned representations with information about global structural connectivity that is relevant to mechanical response. We propose a topology-aware representation learning framework that integrates persistent homology into a DMAE backbone to explicitly encode global structural connectivity. A persistence-guided masking strategy prioritizes patches associated with persistent topological features, while an auxiliary persistent-homology decoder encourages the latent representation to retain information about topological feature counts across connected components, tunnels, and voids. We evaluate the learned representations on two complementary TPMS datasets using family classification, mechanical-property classification, unsuper
    
[^275]: 代码语言模型中的因果语法电路：预算与探针决定了什么看起来是局域化的

    Causal Syntax Circuits in CodeLMs: Budget and Probe Decide What Looks Localised

    [https://arxiv.org/abs/2610.20683](https://arxiv.org/abs/2610.20683)

    该论文通过均值消融、激活修补、贪心电路提取与激活引导的统一流程，系统研究了代码语言模型默认偏好Python的内部机制，发现该偏好随模型规模增长并在指令微调后仍存在，但其局域化表现高度依赖于组件预算与探针的选择。

    

    当提示中没有说明使用哪种语言时，代码模型会默认编写Python。我们针对这一习惯提出了一个机制性问题：它是否由模型内部一小群可找到的组件所产生？我们在55个模型-语言设置上运行了一条统一的实验流程——均值消融、激活修补、贪心电路提取、随机对照显著性检验以及激活引导——涵盖十个开源代码模型（五个基础版和五个指令微调版）、三种目标语言（Rust、C#、PHP）、两套提示集和两种组件预算。我们报告了五项发现。第一，这种偏好很大，随模型规模增长，并且在指令微调后依然存在：每个指令微调版本上的偏好都大于其对应的基础版本，平均高出12%。第二，其测量大小主要取决于探针的属性；用XLCoST平行函数替换我们对齐的提示后，在相同模型上的同一测量值从约1.6降至约0.01。第三，当……（原文在此处截断）

    arXiv:2610.20683v1 Announce Type: new  Abstract: Code models write Python when the prompt does not say which language to use. We ask a mechanistic question about that habit: is it produced by a small, findable group of components inside the model? We run one pipeline---mean ablation, activation patching, greedy circuit extraction, a random-control significance test, and activation steering---over 55 model-language settings spanning ten open code models (five base and five instruction-tuned), three target languages (Rust, C#, PHP), two prompt suites, and two component budgets. We report five findings. First, the preference is large, grows with model size, and survives instruction tuning: it is larger on every instruction-tuned checkpoint than on its base version, by 12% on average. Second, its measured size is mostly a property of the probe; replacing our aligned prompts with XLCoST parallel functions takes the same measurement on the same models from about 1.6 to about 0.01. Third, whe
    
[^276]: TurboMamba：借助词汇记忆实现更快的推理、更快的训练与更好的预测

    TurboMamba: Faster Inference, Faster Training, and Better Prediction with Lexical Memory

    [https://arxiv.org/abs/2610.20682](https://arxiv.org/abs/2610.20682)

    TurboMamba通过结合全局Mamba-3模型、小型Transformer和词汇记忆等轻量级机制，在实现更快推理与训练的同时，取得了比Transformer、Mamba-3、Jamba和Zamba等基线模型更低的预测损失。

    

    我们提出了TurboMamba，这是一种能够实现更快推理、更快训练和更好词元预测的语言模型架构。一个全局Mamba-3模型在每个完成的词元块上运行一次，一个小型Transformer利用该全局上下文以及块内较早的词元来预测下一个词元。三种轻量级改进提升了局部预测效果：直接访问前一个块的最后一个词元、对下一词元分数的小幅修正，以及与近期词元频率的学习型混合。我们在公开的FineWeb-Edu数据集上比较了十二个小规模实现，这些模型具有相近的参数量和共享的训练预算上限。基线模型包括Transformer、Mamba-3、Jamba和Zamba，其中Jamba和Zamba版本使用Mamba-3替换了其原有的Mamba模块。TurboMamba在所有十二个模型中实现了最低的平均测试预测损失（负对数似然，NLL），其平均NLL比Mamba-3低9.4%，比其他模型也更低。

    arXiv:2610.20682v1 Announce Type: cross  Abstract: We introduce TurboMamba, a language-model architecture for faster inference, faster training, and better token prediction. A global Mamba-3 model runs once per completed token block. A small Transformer uses this global context and earlier tokens within the block to predict the next token. Three lightweight additions improve local predictions: direct access to the previous block's last token, a small correction to next-token scores, and a learned mixture with recent-token frequencies. We compare twelve small-scale implementations on public FineWeb-Edu, with comparable parameter counts and shared training-budget caps. The baselines include Transformer, Mamba-3, Jamba, and Zamba. The Jamba and Zamba versions use Mamba-3 in place of their original Mamba modules. TurboMamba achieves the lowest mean test prediction loss (negative log-likelihood, NLL) among all twelve models. Its mean NLL is 9.4% lower than that of Mamba-3, 7.3% lower than t
    
[^277]: 稀疏自编码器中数据的收益递减：功能行为并不能保证特征可复现性

    Diminishing Returns of Data in Sparse Autoencoders: Functional Behavior does not Guarantee Feature Reproducibility

    [https://arxiv.org/abs/2610.20680](https://arxiv.org/abs/2610.20680)

    该论文发现稀疏自编码器的功能表现（如重构）在远小于常规规模的数据量下即可接近全量训练水平，但特征的一对一可复现性会随数据量增加持续提升，表明功能行为良好并不能保证特征可复现，数据规模和数据选择方式对特征复现至关重要。

    

    稀疏自编码器（SAE）通常在数亿到数十亿个激活token上进行训练，但很少有工作检验这种规模是否必要。我们通过在三个语言模型上使用从500万token到5亿token参考预算的不同数据量训练四种SAE架构，研究了SAE到底需要多少激活数据。我们衡量了两种收敛概念：标准评测上的功能行为，以及所学特征的一对一可复现性。功能层面的数据需求强烈依赖于所用的指标：重构性能往往在明显更小的数据预算下就接近全量数据的参考水平，而特征可复现性则随数据量增加持续改善。对于Gemma-2-2B BatchTopK，严格匹配特征的比例从500万token时的2%上升到5000万token时的29%。数据选择同样重要：随机采样能更好地保留原始激活分布，通常更有利于重构，而基于多样性的……（摘要原文在此处被截断）

    arXiv:2610.20680v1 Announce Type: new  Abstract: Sparse autoencoders (SAEs) are typically trained on hundreds of millions to billions of activation tokens, yet little work has tested whether this scale is necessary. We study how much activation data SAEs need by training four SAE architectures on three language models with budgets ranging from 5M tokens to a 500M-token reference. We measure two notions of convergence: functional behavior on standard evaluations and one-to-one reproducibility of learned features. Functional data requirements are strongly metric-dependent: reconstruction often approaches the full-data reference at substantially smaller budgets, whereas feature reproducibility continues to improve with data. For Gemma-2-2B BatchTopK, the fraction of strictly matched features rises from 2\% at 5M tokens to 29\% at 50M. Data selection also matters: random sampling better preserves the original activation distribution and generally favors reconstruction, whereas diversity-ba
    
[^278]: AlphaDiana：面向可验证推理任务中智能体的测量系统

    AlphaDiana: A Measurement System for Agents in Verifiable Reasoning Tasks

    [https://arxiv.org/abs/2610.20675](https://arxiv.org/abs/2610.20675)

    AlphaDiana是一个面向可验证推理任务的模块化智能体测量系统，能够在匹配条件下对完整智能体进行宏观分析并对工具、技能、记忆等组件进行微观干预，研究发现脚手架并不能持续提升模型推理准确性。

    

    智能体将推理模型与脚手架（harness）相结合，脚手架负责组织模型调用、执行工具、加载技能指令，并在内存中保留历史记录。理解智能体性能需要将执行条件与结果、行为和资源使用联系起来。我们推出了AlphaDiana，这是一个为数据集、智能体、环境和评分器提供模块化接口的测量系统。它在保留脚手架原生控制循环的同时，将文件重用和记忆保留的控制相互分离。AlphaDiana既支持对完整智能体进行宏观分析，也支持在匹配条件下对工具、技能和记忆等智能体组件进行微观干预。我们在五个基于答案的基准测试上研究了三个开源模型和三个开源脚手架，其中两个模型还在两个交互式基准上进行了测试。我们发现，脚手架并不能持续提升推理准确性：在模型单独无法解决的问题上所获得的收益，可能会被在其他问题上的损失所抵消

    arXiv:2610.20675v1 Announce Type: new  Abstract: Agents pair reasoning models with harnesses that organize model calls, execute tools, load skill instructions, and retain history in memory. Understanding agent performance requires linking execution conditions to outcomes, behavior, and resource use. We introduce AlphaDiana, a measurement system with modular interfaces for datasets, agents, environments, and scorers. It preserves harness-native control loops while controlling file reuse and memory retention separately. AlphaDiana supports macro analyses of complete agents and micro interventions on agent components such as tools, skills, and memory under matched conditions. We study three open-source models and three open-source harnesses on five answer-based benchmarks, with two models also tested on two interactive benchmarks. We find that harnesses do not consistently improve the accuracy of reasoning: gains on questions unsolved by the model alone can be outweighed by losses on ques
    
[^279]: 关于流映射训练的稳定性与收敛性

    On the Stability and Convergence of Flow Map Training

    [https://arxiv.org/abs/2610.20674](https://arxiv.org/abs/2610.20674)

    本文通过分析MeanFlow与拉格朗日损失诱导的函数梯度流，首次从理论上解释了流映射训练中停止梯度算子和自适应加权等技巧的必要性，并证明了训练会收敛到真实流映射。

    

    流映射是近来的一种概率生成方法，它将扩散轨迹摊销为单步采样器，从而实现高效生成。尽管在提升其实证性能方面已取得进展，但对其成功背后的机制仍缺乏理解。具体而言，许多流映射学习方法将停止梯度算子作为其训练目标的关键组成部分。自适应加权也极大地提升了流映射的生成能力。尽管已有许多解释被提出以支持这些启发式技巧的引入，但尚无工作分析这些假设对流映射自身训练动态的影响。在本工作中，我们通过提供有依据的理论来解释为什么需要这些技巧，从而解决了这一问题。通过分析由 MeanFlow 损失和拉格朗日损失所诱导的函数梯度流，我们证明了其收敛到真实流映射……

    arXiv:2610.20674v1 Announce Type: cross  Abstract: Flow maps are a recent method of probabilistic generation that amortize diffusion trajectories into one-step samplers, unlocking efficient generation. While progress has been made in improving their empirical performance, there is still a lack of understanding behind the mechanisms underlying their success. Specifically, many flow map learning methods employ stop gradient operators as a crucial part of their training target. Adaptive weighting also drastically improves the generation capabilities of flow maps. Although many interpretations have been proposed to support the inclusion of these heuristics, none have analyzed the impact of these assumptions on the training dynamics of flow maps themselves. In this work, we address this issue by providing grounded theory explaining why such tricks are required. Analyzing the functional gradient flow induced by the MeanFlow and Lagrangian losses, we show their convergence to the true flow ma
    
[^280]: 用于加速离散扩散中序贯蒙特卡洛引导的价值信息驱动的调度裁剪方法

    Value-Informed Schedule Trimming for Accelerated Sequential Monte Carlo Guidance in Discrete Diffusion

    [https://arxiv.org/abs/2610.20669](https://arxiv.org/abs/2610.20669)

    VISTA是一种无需训练的方法，通过将稀疏引导建模为对理想完全引导概率路径的近似，并利用预热阶段收集的价值统计量结合动态规划精确求解经验代理目标，从而在固定计算预算下自动选择最优的引导时间步，显著加速离散扩散模型中的序贯蒙特卡洛奖励引导。

    

    奖励引导的离散扩散能够在测试时与下游目标实现对齐，但在每个去噪步骤都应用序贯蒙特卡洛（SMC）引导的计算代价十分高昂。主动引导步骤需要额外的奖励评估，而对于基于梯度的提议方法，还需要进行价值梯度计算。稀疏引导通过仅对部分时间步施加引导来降低这一成本，但常用的调度方案是固定的启发式规则，其有效性在不同任务之间可能存在显著差异。我们提出VISTA，这是一种无需训练的方法，可在固定预算下自动选择需要引导的时间步。VISTA将稀疏引导表述为对理想完全引导概率路径的近似，并推导出ℓ₁路径偏差的一个可估计上界。利用从少量完全引导预热生成中收集的价值统计量，VISTA通过动态规划对该经验代理目标进行精确优化。（摘要原文在此处截断）

    arXiv:2610.20669v1 Announce Type: new  Abstract: Reward-guided discrete diffusion enables test-time alignment with downstream objectives, but applying Sequential Monte Carlo (SMC) guidance at every denoising step is computationally expensive. Active guidance steps require additional reward evaluations and, for gradient-based proposals, value-gradient computations. Sparse guidance reduces this cost by guiding only a subset of timesteps, but commonly used schedules are fixed heuristics whose effectiveness can vary substantially across tasks. We propose VISTA, a training-free method that automatically selects guided timesteps under a fixed budget. VISTA formulates sparse guidance as approximation of the ideal full-guidance probability path and derives an estimable upper bound on \(\ell_1\) path deviation. Using value statistics collected from a small number of full-guidance warmup generations, VISTA optimizes the resulting empirical surrogate exactly via dynamic programming. The warmup ge
    
[^281]: CascadeLedger：LLM级联中信息与资源完整性的反事实审计

    CascadeLedger: Counterfactual Auditing of Information and Resource Integrity in LLM Cascades

    [https://arxiv.org/abs/2610.20668](https://arxiv.org/abs/2610.20668)

    CascadeLedger提出了一个面向预算受限LLM级联的反事实执行审计框架，通过聚合等价轨迹有效性和反事实前缀无干扰性等机制，审计路由决策是否仅使用了合法可观测信息，并能定位最早违规点与最小隐藏见证。

    

    我们提出了CascadeLedger，一个面向预算受限语言模型级联的反事实执行审计框架。该协议强制执行“先调用后读取”语义，将成功的读取操作绑定到服务与缓存标识上，记录可用性状态及各组件的预算转换，并在接入任何目标模型或任务评分器之前冻结事件轨迹。其核心对象是聚合等价轨迹有效性：两次执行可以拥有相同的最终答案、声明的资源投影和服务直方图，但其路由决策是否仅使用了合法可观测的信息却可能存在差异。我们用反事实前缀无干扰性（CPNI）来补充单轨迹不变量，该性质在耦合可见前缀与策略随机性的同时，仅改变一个声明的隐藏族。反事实影响边界（CIF）定位最早违规的前缀以及一个最小隐藏见证；保持投影的对抗性变异（PPAM）

    arXiv:2610.20668v1 Announce Type: cross  Abstract: We introduce CascadeLedger, a counterfactual execution-audit framework for budgeted language-model cascades. The protocol enforces call-before-read semantics, binds successful reads to service and cache identities, records availability and component-wise budget transitions, and freezes the event trace before any target or task scorer is joined. The central object is aggregate-equivalent trace validity: two executions can share a final answer, declared resource projection, and service histogram while differing in whether their routing decisions used only legitimately observable information. We complement single-trace invariants with counterfactual prefix non-interference (CPNI), which couples a visible prefix and policy randomness while changing only a declared hidden family. The counterfactual influence frontier (CIF) localizes the earliest violating prefix and a minimal hidden witness; projection-preserving adversarial mutation (PPAM)
    
[^282]: 基准测试饱和后的模型能力评估

    Evaluating Model Capabilities After Benchmark Saturation

    [https://arxiv.org/abs/2610.20667](https://arxiv.org/abs/2610.20667)

    该论文提出利用模型在饱和基准测试中的完整解题过程记录（而非仅依赖分数），即使基准分数饱和后仍能有效评估和预测模型能力。

    

    AI能力的快速进步正在导致基准测试趋于饱和。一旦多个模型接近满分，基准测试就失去了区分它们的能力，同时也失去了为AI安全论证和前沿政策提供参考的价值。然而，基准测试题目所产生的信息远比我们通常从中提取的分数丰富：那就是每个模型如何完成任务的完整解题过程记录。我们研究这些过程记录是否能在基准分数饱和之后仍用于评估模型能力。我们利用历史上的基准测试和模型发布数据回顾性地模拟了饱和过程，并测试仅访问饱和基准中成功解题记录的协议能否预测在保留的未饱和题目上测得的能力。我们评估了几种具有代表性的从模型过程记录中提取信息的当代方法：成对评判、单条记录绝对评分以及基于预测的方法。

    arXiv:2610.20667v1 Announce Type: new  Abstract: Rapid advances in AI capabilities are causing benchmarks to saturate. Once several models approach perfect scores, a benchmark loses its power to differentiate between them, and with it its utility for informing AI safety cases and frontier policies. Yet benchmark items produce far more information than the scores we typically extract from them: a full transcript of how each model solved the task. We investigate whether these transcripts can be used to assess model capabilities even after benchmark scores have saturated. We simulate saturation retrospectively using historical benchmark and model releases, and test whether protocols that access only correctly solved transcripts from saturated benchmarks can predict capabilities measured on held-out, unsaturated items. We evaluate representative contemporary approaches for extracting information from model transcripts: pairwise judging, single-transcript absolute scoring, and predictions f
    
[^283]: 解耦特征、拓扑与交互的图加性模型

    Graph Additive Models disentangling feature, topology, and interactions

    [https://arxiv.org/abs/2610.20662](https://arxiv.org/abs/2610.20662)

    提出AMIGA模型，通过挖掘的图模体表示拓扑结构，并将节点贡献分解为特征、结构和交互三个加性通道，从而解决了图神经加性网络中虚假相关性传递和效应纠缠的问题。

    

    图神经加性网络（GNANs）已成为一种有效的、具备内建可解释性的图学习架构。它们通过将逐特征的形状函数与简单的基于距离的重加权相结合来实现透明性。然而，GNANs存在三个局限性。第一，节点级相关性得分聚合了来自图中所有节点的信息，使得无关节点可能从相关节点继承虚假的相关性。第二，拓扑仅通过最短路径距离来表示，因此结构不同的图可能具有完全相同的距离特征。第三，特征与距离信息以乘法方式结合，导致特征、距离和交互效应被纠缠在同一个可缩放项中。我们提出AMIGA，它通过挖掘出的模体词汇表来表示拓扑结构，并将每个节点的贡献分解为三个加性通道：特征、结构以及特征-结构交互。这些通道被中心化……

    arXiv:2610.20662v1 Announce Type: cross  Abstract: Graph Neural Additive Networks (GNANs) have emerged as effective, interpretable-by-design architectures for graph learning. They achieve transparency by combining per-feature shape functions with simple distance-based reweighting. However, GNANs have three limitations. First, node-wise relevance scores aggregate information from all graph nodes, allowing irrelevant nodes to inherit spurious relevance from relevant ones. Second, topology is represented only through shortest-path distances, so structurally distinct graphs may share identical distance profiles. Third, feature and distance information are combined multiplicatively, entangling feature, distance, and interaction effects in a single rescalable term. We introduce AMIGA, which represents topology through a vocabulary of mined motifs and decomposes each node contribution into three additive channels: feature, structure, and feature-structure interaction. The channels are centere
    
[^284]: 基于方差保持桥的稀疏观测随机场潜在退火后验采样

    Latent Annealed Posterior Sampling with a Variance-Preserving Bridge for Sparsely Observed Random Fields

    [https://arxiv.org/abs/2610.20659](https://arxiv.org/abs/2610.20659)

    提出LAPS方法，利用流匹配学习潜在先验并在潜在空间中进行退火后验采样，结合方差保持桥控制协方差膨胀，从而高效解决稀疏观测随机场后验推断中空间不确定性刻画与计算成本增长的难题。

    

    从稀疏观测中对随机场进行后验推断面临两大挑战：如何准确刻画复杂的空间不确定性，以及如何避免条件采样成本随空间网格加密而大幅增长。为解决这些挑战，我们提出了潜在退火后验采样（LAPS）。LAPS首先将随机场映射到降维后的潜在空间，同时利用由Branch网络和Trunk网络组成的坐标解码器，实现对任意空间位置的场值评估。随后采用流匹配学习潜在先验分布。基于该学习到的先验，后验采样完全在潜在空间中进行，观测似然通过退火过程逐步融入采样。为控制局部协方差膨胀，我们引入了一种方差保持（VP）桥，在传输的潜在状态与目标状态之间分配协方差预算。

    arXiv:2610.20659v1 Announce Type: cross  Abstract: Posterior inference for random fields from sparse observations presents two major challenges: accurately characterizing complex spatial uncertainty and preventing conditional sampling costs from growing substantially as the spatial grid is refined. To address these challenges, we propose latent annealed posterior sampling (LAPS). LAPS first maps the random field into a latent space with reduced dimensionality, while a coordinate decoder composed of Branch and Trunk networks enables field evaluation at arbitrary spatial locations. Flow matching is then used to learn the latent prior distribution. Based on this learned prior, posterior sampling is performed entirely in the latent space, with the observation likelihood incorporated progressively through an annealing procedure. To control local covariance inflation, we introduce a variance-preserving (VP) bridge that allocates a covariance budget between the transported latent state and a 
    
[^285]: 列表可复现性的一个多对数界

    A polylogarithmic bound for list replicability

    [https://arxiv.org/abs/2610.20657](https://arxiv.org/abs/2610.20657)

    本文为任意二元概念类的列表可复现性数 O(log²M log N) 这一首个多对数上界提供了一个简短且自包含的新证明。

    

    Zhang [Zha26] 最近发布了一份手稿，宣布对于定义在 N≥2 个定义域点上的任意一类 M≥2 个二元概念，其列表可复现性数（list-replicability number）为 O(log²M log N)，这将是关于该量的首个多对数上界。在这篇简短笔记中，我们遵循该手稿的高层思路，但采用了 considerably 更为简化的论证，给出了这一界的一个简短且自包含的证明。

    arXiv:2610.20657v1 Announce Type: cross  Abstract: Zhang [Zha26] recently posted a manuscript announcing that the list-replicability number of every class of $M\ge 2$ binary concepts on $N\ge 2$ domain points is $O(\log^2 M\log N)$, which would be the first polylogarithmic upper bound on this quantity. In this note, we present a short, self-contained proof of this bound, following the high-level approach of the manuscript but with considerably simpler arguments.
    
[^286]: 用于曝光后烘焙模拟的全局时间谱Duhamel-Picard方法及用于EUV光刻的物理信息神经算子

    Global-in-time Spectral Duhamel-Picard Method for Post-Exposure Bake Simulation and Physics-Informed Neural Operators for EUV Lithography

    [https://arxiv.org/abs/2610.20655](https://arxiv.org/abs/2610.20655)

    本文提出一种全局时间谱Duhamel-Picard方法，通过将酸-猝灭剂反应扩散方程转化为Volterra积分方程并用多项式展开与Anderson加速Picard迭代求解，实现了约一分钟内模拟1200×1200×100体素的大规模三维曝光后烘焙过程，并将其融入物理信息神经算子用于EUV光刻模拟。

    

    我们提出了一种用于曝光后烘焙模拟的全局时间谱Duhamel-Picard多项式展开方法。利用Duhamel原理，将耦合的酸-猝灭剂反应扩散方程重新表述为非线性Volterra积分方程。完整的轨迹由有限多项式展开表示，其时间系数场通过Anderson加速的Picard迭代在整个烘焙区间内一次性确定，从而避免了顺序时间步进。在三维PEB模拟中，我们采用Chebyshev和Legendre时间基，并使用相应的高斯求积规则计算Duhamel积分。我们证明了所提出的方法可以在一分钟左右的时间内模拟1200×1200×100体素的大规模区域。该表示进一步被纳入物理信息神经算子（PINO）中，采用了分解式傅里叶神经算子的混合变换变体……

    arXiv:2610.20655v1 Announce Type: cross  Abstract: We present a global-in-time spectral Duhamel-Picard method with polynomial expansions for post-exposure bake (PEB) simulation. The coupled acid-quencher reaction-diffusion equations are reformulated as nonlinear Volterra integral equations using the Duhamel principle. Full trajectories are represented by finite polynomial expansions, with temporal coefficient fields determined over the bake interval by Anderson-accelerated Picard iteration, thereby avoiding sequential time marching. In 3-D PEB simulations, we employ Chebyshev and Legendre temporal bases while evaluating the Duhamel integrals using the corresponding Gaussian quadrature rules. We demonstrate that the proposed method simulates a large domain of $1200\times1200\times100$ voxels on the order of one minute. The representation is further incorporated into a physics-informed neural operator (PINO) with a mixed-transform variant of a factorized Fourier neural operator, which is
    
[^287]: AbridgeKV：将长推理压缩到学习到的暂停令牌的KV缓存中

    AbridgeKV: Compressing Long Reasoning into the KV Cache of Learned Pause Tokens

    [https://arxiv.org/abs/2610.20641](https://arxiv.org/abs/2610.20641)

    AbridgeKV通过训练模型将已生成的长推理文本块压缩到学习到的暂停令牌的KV缓存中并释放原始缓存，解决了长链式思维解码中KV缓存线性增长的内存瓶颈问题。

    

    长链式思维（chain-of-thought）解码受内存限制：生成token的KV缓存随着测试时计算量的增加而线性增长。简单地驱逐缓存条目并不能解决这个问题，因为从未在删除条目的缓存上训练过的模型会生成更长的文本，往往达到长度上限。我们转而训练模型压缩自己的输出。AbridgeKV将每个已完成的生成文本块压缩为少数几个学习到的暂停令牌的KV条目。这些是带有训练嵌入的额外输入token，不产生任何输出。一旦暂停令牌编码了该文本块，AbridgeKV就释放该块的原始缓存；后续token转而关注暂停令牌的KV条目。三个新颖的设计使这一方法成为可能：暂停令牌的KV条目占据它们所替换文本块内部的位置；缓存预算根据每层条目的实测价值在各层之间进行分配；压缩在相对较大的（文本块上进行）。

    arXiv:2610.20641v1 Announce Type: new  Abstract: Long chain-of-thought decoding is memory-bound: the KV cache of the generated tokens grows linearly with the test-time compute being spent. Evicting cache entries does not fix this, because a model never trained on a cache with entries removed generates longer, often up to the length limit. We instead train the model to compress its own output. AbridgeKV compresses each completed block of generated text into the KV entries of a few learned pause tokens. These are extra input tokens with trained embeddings that produce no output. Once the pause tokens have encoded the block, AbridgeKV frees the block's original cache; subsequent tokens attend to the pause tokens' KV entries in its place. Three novel choices make this work: the pause tokens' KV entries take positions inside the block they replace, the cache budget is allocated across layers according to the measured value of each layer's entries, and compression operates on relatively larg
    
[^288]: 相同的答案，新的框：RLVR的域外增益中有多少来自答案格式？

    Same Answer, New Box: How Much of RLVR's Out-of-Domain Gain Is Answer Format?

    [https://arxiv.org/abs/2610.20639](https://arxiv.org/abs/2610.20639)

    研究发现，RLVR所报告的域外多项选择基准提升中约82%只是答案格式提取带来的假象，而非模型真正回答能力的提升。

    

    带有可验证奖励的强化学习（RLVR）被认为能带来超越其奖励所检验领域的增益，例如在数学上训练后MMLU准确率更高。这些准确率通常是通过答案提取器从模型文本中读取的。因此，一个学会把答案写在提取器寻找位置的模型，即使回答得并没有更好，也能获得更高的分数。我们测量了从公开代码中复制的提取器所报告的增益中有多少是这样产生的。我们生成基础模型及其RL模型的输出，并进行两次评分：一次使用这些提取器，另一次使用我们自己的提取器——无论答案是写在答案框\boxed{}中还是文本中，都能读取所陈述的答案。在四个已发布的7B SimpleRL基础-RL模型对上，这些提取器报告的域外多项选择增益中有82%在我们的提取器下消失了：这是答案格式上的增益，而非答案本身的增益。基础模型已经在文本中陈述了它们的大部分答案，而RL使……（原文摘要至此截断）

    arXiv:2610.20639v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) is credited with gains beyond the domain its rewards check, such as higher MMLU accuracy after training on mathematics. Such accuracies are usually read out of the model's text by an answer extractor. A model that learns to write its answer where the extractor looks can therefore score higher without answering better. We measure how much of the gain reported by extractors copied from public code arises this way. We generate outputs from a base model and its RL model and score them twice: with those extractors, and with one extractor of our own that reads a stated answer whether it is in the answer box \boxed{} or in the text. On four released 7B SimpleRL base-RL pairs, 82% of the out-of-domain multiple-choice gain that these extractors report disappears under ours: it is a gain in answer format, not in answers. The base models already state most of their answers in text, and RL move
    
[^289]: 面向小型大语言模型欺骗行为SAE引导的特征引出方法

    Feature Elicitation for SAE Steering of Deceptive Behaviour in Small LLMs

    [https://arxiv.org/abs/2610.20638](https://arxiv.org/abs/2610.20638)

    该研究提出一种新颖的词袋特征引出流水线，通过引导与义务论伦理相关的SAE特征，首次成功将Gemma-3-27B-IT模型的合规差距（对齐伪装的关键指标）降低至不显著水平，并揭示了一种名为“欺骗性去锋”的新型欺骗行为。

    

    随着前沿模型能力的不断提升及其造成灾难性危害风险的日益增加，对欺骗性对齐的研究变得愈发紧迫。对齐伪装是欺骗性对齐的一种形式，其中合规差距是其关键的行为指标。这种行为在小型大语言模型中仍然研究不足，且先前的机制性干预尚未被证明能够缓解合规差距。我们测试了引导与义务论伦理相关的稀疏自编码器（SAE）特征能否缓解这种行为。我们使用一种新颖的词袋特征引出流水线进行SAE引导，以识别我们所理解为概念级的引导方向。我们的方法将Gemma-3-27B-IT中的合规差距降低至不显著水平，而所测试的配置在LLaMA-3-8B-Instruct中并未奏效。对草稿本轨迹的分析还揭示了模型的一种欺骗性行为，我们将其命名为“欺骗性去锋”（deceptive defanging）：故意……（摘要原文在此处截断）

    arXiv:2610.20638v1 Announce Type: new  Abstract: Research on deceptive alignment is becoming increasingly urgent as frontier-model capabilities advance and their risk of causing catastrophic harm increases. Alignment faking is a form of deceptive alignment, with the compliance gap serving as a key behavioural indicator. This behaviour remains understudied in small LLMs and prior mechanistic interventions have not been shown to mitigate the compliance gap. We test whether steering SAE (sparse autoencoders) features associated with deontological ethics can mitigate this behaviour. We use a novel bag-of-words feature elicitation pipeline for SAE steering to identify what we construe as concept-level steering directions. Our approach reduced the compliance gap to non-significance in Gemma-3-27B-IT, whereas the tested configurations were not effective in LLaMA-3-8B-Instruct. Analysis of scratchpad traces also revealed deceptive model behaviour which we term deceptive defanging: deliberately
    
[^290]: FlowBudget：面向流匹配模型的免训练采样预算估计

    FlowBudget: Training-Free Sampling-Budget Estimation for Flow Matching Models

    [https://arxiv.org/abs/2610.20636](https://arxiv.org/abs/2610.20636)

    提出FlowBudget，一种免训练的采样预算估计方法，通过免模拟插值探针为流匹配模型确定足够的数值求解分辨率（NFE），从而避免昂贵且重复的FID-NFE扫描。

    

    从流匹配模型这类基于连续流的生成器中进行采样，需要对常微分方程（ODE）或随机微分方程（SDE）进行数值积分，这使得推理成本取决于函数评估次数（NFE）。在实践中，采样预算往往继承自先前工作，或通过代价高昂的“FID-对-NFE”扫描来选择（即使用FID评估图像生成质量），这需要在多个预算下反复生成并评估样本，使采样预算的选择依赖于大量的经验性评估。因此，我们提出了一个与求解器设计和时间步分配互补的问题：对于一个固定的预训练模型和求解器，多大的数值分辨率才是足够的？我们提出了FLOWBUDGET，一种求解器相对采样复杂度的免训练估计器。FLOWBUDGET利用对所学动力学的免模拟插值探针，将主导性的一步离散化误差按运动量进行归一化，并通过插值探针估计其系数，从而判断所需的采样预算。

    arXiv:2610.20636v1 Announce Type: new  Abstract: Sampling from continuous flow-based generators such as flow matching models requires numerical ODE or SDE integration, making inference cost depend on the number of function evaluations (NFEs). In practice, the sampling budget is inherited from prior work or selected through expensive FID-versus-NFE sweeps, using FID to assess image-generation quality, requiring sample generation and evaluation at multiple budgets. This makes sampling-budget selection dependent on repeated empirical evaluation. We therefore ask a complementary question to solver design and timestep allocation: for a fixed pretrained model and solver, what numerical resolution is sufficient? We introduce FLOWBUDGET, a training-free estimator of solver-relative sampling complexity. Using simulation-free interpolation probes of the learned dynamics, FLOWBUDGET normalizes the leading one-step discretization error by the motion, estimates its coefficient from interpolation pr
    
[^291]: 探索置换扩散大语言模型的设计空间

    Exploring the Design Space of Permutation Diffusion LLMs

    [https://arxiv.org/abs/2610.20635](https://arxiv.org/abs/2610.20635)

    该论文系统探索了三种用于排列生成的扩散大语言模型参数化设计（硬分配、软分配和直接排序分数预测），发现硬/软分配方法在单次函数评估下达到与掩码扩散基线相当的推荐重排性能，而直接分数预测无需匈牙利算法即可保证有效排列。

    

    使用语言模型进行置换生成，既需要文本条件化的排序能力，也需要候选对象与输出位置之间的全局一一对应分配。我们通过三种扩散大语言模型参数化方法来研究这一问题：在打乱的候选索引序列上进行硬分配预测、在双随机矩阵上进行软分配预测，以及直接预测排序分数后再进行排序。这三种方法族将置换约束置于预测和解码的不同阶段，因此支持不同的训练目标和采样流程。我们在三个亚马逊领域上评估了这些设计，候选集大小分别为10和20。在单次函数评估下，硬分配和软分配变体取得了与受限掩码扩散基线相当的Recall@1，而直接排序分数预测无需匈牙利分配算法即可保证生成有效的置换……

    arXiv:2610.20635v1 Announce Type: cross  Abstract: Permutation generation with language models requires both text-conditioned ranking and a global one-to-one assignment between candidates and output positions. We study this problem through three diffusion-LLM parameterizations: hard-assignment prediction over shuffled candidate-index sequences, soft-assignment prediction over doubly stochastic matrices, and direct ordering-score prediction followed by sorting. The three families place permutation constraints at different stages of prediction and decoding and therefore support different training objectives and sampling procedures. We evaluate these designs on recommendation reranking across three Amazon domains with candidate sets of size 10 and 20. At one function evaluation, the hard- and soft-assignment variants achieve Recall@1 comparable to a constrained masked-diffusion baseline, whereas direct ordering-score prediction guarantees valid permutations without Hungarian assignment or
    
[^292]: 面向巴西葡萄牙语多类别金融新闻分类的大规模数据集

    A Large-Scale Dataset for Multi-Class Financial News Classification in Brazilian Portuguese

    [https://arxiv.org/abs/2610.20630](https://arxiv.org/abs/2610.20630)

    本文发布了约16万篇巴西葡萄牙语金融新闻的大规模语料库，并提供了人工与LLM两种标注的基准子集，实验发现基于语义相似度的弱监督多类别金融新闻分类性能高度依赖标签质量与类别分布，其中KNN方法表现最佳。

    

    针对巴西葡萄牙语的金融自然语言处理（NLP）系统的发展一直受到特定领域数据集稀缺的阻碍。为解决这一问题，我们提出了一个包含约160,000篇从网络来源收集的巴西葡萄牙语金融新闻文章的大规模语料库。为了给基于语义相似度的弱监督多类别分类方法建立初步基准，我们对该语料库的子集分别进行了人工标注和使用大语言模型（LLM）的标注。我们使用多语言语义嵌入评估了三种基于相似度的方法——余弦相似度、K近邻（KNN）以及基于FAISS的最近邻搜索。实验表明，该任务对标签质量和类别分布高度敏感。在人工标注的子集上，KNN取得了最佳性能（47.3%准确率，31.9% F1分数），FAISS紧随其后；相反，在LLM标注的子集上……（原文摘要至此截断）

    arXiv:2610.20630v1 Announce Type: new  Abstract: The development of Financial Natural Language Processing (NLP) systems for Brazilian Portuguese is hindered by the scarcity of domain-specific datasets. To address this, we present a large-scale corpus of approximately 160,000 Brazilian Portuguese financial news articles collected from web sources. To establish initial benchmarks for a weakly supervised multi-class classification approach based on semantic similarity, we annotated subsets of this corpus both manually and using a Large Language Model (LLM). We evaluated three similarity-based methods -- cosine similarity, K-Nearest Neighbors (KNN), and FAISS-based nearest-neighbor search -- using multilingual semantic embeddings. Our experiments reveal that the task is highly sensitive to label quality and class distribution. On the manually labeled subset, KNN achieved the best performance (47.3% accuracy, 31.9% F1-score), closely followed by FAISS. Conversely, on the LLM-annotated subse
    
[^293]: BD-HazardVLM：探究视觉语言模型在孟加拉国道路场景中对潜在防御性驾驶危险的识别能力

    BD-HazardVLM: Probing Vision-Language Models for Latent Defensive-Driving Hazards in Bangladesh Road Scenes

    [https://arxiv.org/abs/2610.20627](https://arxiv.org/abs/2610.20627)

    提出了BD-HazardVLM诊断基准，利用500张孟加拉国道路场景图像系统评估视觉语言模型对防御性驾驶危险的推理能力，发现模型虽能判断危险是否存在（接近多数类基线），但在危险类别、风险时机和防御动作等细粒度推理上表现严重不足。

    

    视觉语言模型越来越多地被用于驾驶场景理解，但它们在弱结构化的全球南方交通环境下的防御性推理能力仍不清楚。我们提出了BD-HazardVLM，这是一个包含500张孟加拉国道路场景图像的诊断性基准，其中375张来自RSUD20K，125张来自TFP-BD，标注内容涵盖危险存在与否、主要危险类别、风险发生时机、严重程度、防御性动作、可见证据，以及幻觉陷阱和西方道路偏见陷阱。我们采用统一的零样本任务定义和结构化输出模式，评估了Qwen2.5-VL-7B、InternVL3-8B和MiniCPM-V-4.6三个模型。结果显示，二元危险存在判断的准确率在88.0%至89.2%之间，基本与89.0%的多数类基线持平；而细粒度推理能力则明显较弱：类别判断准确率低于40%，风险时机判断准确率仅为1.6%至12.4%，防御性动作的精确匹配率低于2%。

    arXiv:2610.20627v1 Announce Type: cross  Abstract: Vision-language models are increasingly used for driving-scene understanding, yet their defensive reasoning under weakly structured Global South traffic remains unclear. We introduce BD-HazardVLM, a diagnostic benchmark of 500 Bangladesh road-scene images, comprising 375 from RSUD20K and 125 from TFP-BD, annotated for hazard presence, primary hazard category, risk timing, severity, defensive actions, visible evidence, and hallucination- and Western-road-bias traps. We evaluate Qwen2.5-VL-7B, InternVL3-8B, and MiniCPM-V-4.6 using a shared zero-shot task definition and structured output schema. Binary hazard-presence accuracy ranges from 88.0 percent to 89.2 percent, essentially matching the 89.0 percent majority-class baseline. Fine-grained reasoning is substantially weaker: category accuracy remains below 40 percent, risk-timing accuracy ranges from 1.6 percent to 12.4 percent, and exact defensive-action matching remains below 2 percen
    
[^294]: pcNN-SN：一种物理增强的预测-校正神经网络算法与离散纵标法耦合求解中子输运方程

    pcNN-SN: a physics-enhanced predictor-corrector neural network algorithm coupled with the discrete ordinates method for neutron transport equations

    [https://arxiv.org/abs/2610.20619](https://arxiv.org/abs/2610.20619)

    本文提出一种物理增强的预测-校正神经网络算法pcNN-SN，通过与离散纵标法耦合，利用高精度数据预测粗网格解的校正因子，在保持守恒性质的同时有效降低中子输运计算的成本。

    

    超分辨率机器学习方法有望解决传统输运模拟中高维特征带来的高计算成本问题。本文提出了一种用于离散纵标（SN）输运计算的物理增强神经网络耦合算法。基于高精度和低精度数据，我们开发了预测-校正神经网络，用于预测粗网格解的校正因子。基于输运方程，选取包含亚网格尺度信息的局部物理特征作为网络输入。通过对物理量进行无量纲变换，降低了输入空间的维度并收窄了参数分布，从而便于网络训练。物理约束被纳入网络模型中，校正后的解保留了其原有的守恒特性。pcNN与SN输运扫掠内部耦合，数值结果表明...

    arXiv:2610.20619v1 Announce Type: cross  Abstract: Super-resolution machine learning methods hold promise for addressing the high computational costs associated with high-dimensional features in traditional transport simulations. This paper proposes a physics-enhanced neural network coupling algorithm for discrete ordinates (SN) transport calculations. Based on high- and low-accuracy data, we develop a predictor-corrector neural network (pcNN) to predict correction factors for coarse-grid solutions. Based on transport equations, local physical features containing subgrid-scale information are selected as network inputs. Dimensionless transformation of physical quantities reduces the input space dimensionality and narrows parameter distributions, facilitating network training. Physical constraints are incorporated into the network model, and the corrected solution retains its original conservation properties. pcNN and SN transport sweeps are internally coupled, and numerical solutions a
    
[^295]: 欠阻尼朗之万动力学中承诺函数的迭代神经变分求解器

    Iterative Neural Variational Solver of Committor Function in Under-damped Langevin Dynamics

    [https://arxiv.org/abs/2610.20617](https://arxiv.org/abs/2610.20617)

    本文提出了一种计算欠阻尼朗之万动力学中承诺函数的迭代神经变分求解器，克服了后向Kolmogorov方程缺乏变分结构这一长期挑战，从而能够定量研究有限阻尼系数对系统中稀有转变的影响。

    

    承诺函数描述了随机动力学系统在两个亚稳态之间转变时，首先到达其中一个状态而非另一个状态的概率。它是转变路径理论中的核心对象，通过反应性概率流和过渡态系综提供了精确的反应速率以及对转变机制的特征刻画。尽管针对摩擦阻尼系数趋于无穷大的过阻尼朗之万动力学，该函数的数值方法已得到充分研究，但由于其背后的后向Kolmogorov方程缺乏变分结构，在一般欠阻尼情形下计算该函数仍然是一个悬而未决的挑战。这限制了对实际系统中有限阻尼系数对稀有转变影响的定量研究。在这项工作中，通过将欠阻尼朗之万动力学的无穷小生成元分解……

    arXiv:2610.20617v1 Announce Type: cross  Abstract: The committor function describes the probability that a stochastic dynamical system, while transitioning between two metastable states, will first reach one state rather than the other. It is a central object in transition path theory, providing the accurate reaction rate and the characterization of the transition mechanism through reactive probability flux and the transition state ensemble. While the numerical methods for this function are well studied for over-damped Langevin dynamics where the friction damping coefficient goes to infinity, its computation in the general under-damped case remains an open challenge due to the lack of a variational structure in the underlying backward Kolmogorov equation. This restricts the quantitative investigation of the impact of the finite damping coefficient on rare transitions in real systems. In this work, by decomposing the infinitesimal generator of the underdamped Langevin dynamics into a se
    
[^296]: DeepUnfolding_Agent：一个经典-量子自适应控制API

    DeepUnfolding_Agent: A Classical_Quantum Adaptive Control API

    [https://arxiv.org/abs/2610.20615](https://arxiv.org/abs/2610.20615)

    本文提出 DeepUnfolding Agent，一个基于模式驱动的 FastAPI 平台，将深度展开的自适应控制作为外部软件服务从特定领域执行中解耦出来，从而提升自适应优化工作流在机器学习、联邦学习和量子系统中的复用性、可重复性与安全性。

    

    arXiv:2610.20615v1 Announce Type: new 摘要：机器学习、联邦学习、量子机器学习以及混合经典-量子系统中的自适应优化工作流通常将控制器逻辑直接嵌入到特定任务的训练脚本中。这种耦合方式限制了复用性、可重复性、安全性验证以及跨执行环境的公平比较。本文提出了 DeepUnfolding Agent，一个基于模式驱动的 FastAPI 平台，将深度展开的自适应控制以外部软件服务的形式提供。外部执行器注册策略模式、提交轮次级别的诊断变量、接收受模式约束的控制动作、执行一轮优化，并返回反馈用于控制器更新。该框架通过策略注册表、诊断到状态的适配器、控制器工厂、安全防护机制、执行器抽象、反馈更新路径、SQLite 持久化层、Docker 部署以及可重复性支持，将控制器智能与特定领域的执行过程分离开来。

    arXiv:2610.20615v1 Announce Type: new  Abstract: Adaptive optimization workflows in machine learning, federated learning, quantum machine learning, and hybrid classical quantum systems often embed controller logic directly inside task specific training scripts. This coupling limits reuse, reproducibility, safety validation, and fair comparison across execution environments. This paper presents DeepUnfolding Agent, a schema-driven FastAPI platform that exposes deep unfolded adaptive control as an external software service. External executors register a strategy schema, submit round-level diagnostic variables, receive schema-bounded control actions, execute one optimization round, and return feedback for controller update. The framework separates controller intelligence from domain specific execution through a strategy registry, diagnostic to state adapter, controller factory, safety guard, executor abstraction, feedback update path, SQLite persistence layer, Docker deployment, and repro
    
[^297]: 用于非均质材料断裂预测的热力学一致性机器学习

    Thermodynamically Consistent Machine Learning for Fracture Prediction in Heterogeneous Materials

    [https://arxiv.org/abs/2610.20614](https://arxiv.org/abs/2610.20614)

    该论文提出了热力学一致性潜空间动力学辨识框架，利用强制满足总能量守恒和熵产生等热力学原理的参数化GENERIC形式信息神经网络，在低维潜空间中高效预测非均质材料的断裂模式与演化过程。

    

    理解断裂行为对于评估结构的完整性和使用寿命至关重要。基于物理的断裂模拟极其耗时，这促使人们开发基于机器学习的代理模型来预测断裂演化。然而，大多数现有的断裂预测机器学习模型以黑箱方式运行，并且需要大量训练数据才能实现可靠泛化。在这项工作中，我们引入了面向参数化系统的热力学一致性潜空间动力学辨识方法，用于预测非均质材料中的断裂模式及其演化。该模型使用参数化GENERIC形式体系信息神经网络（pGFINNs）在低维潜空间中学习热力学状态变量，该神经网络强制执行关键的热力学原理，例如总能量守恒和熵产生。该模型通过硬约束方式将物理嵌入到神经网络架构中……（原文截断）

    arXiv:2610.20614v1 Announce Type: cross  Abstract: Understanding fracture behavior is critical for assessing the structural integrity and longevity. Highly time-consuming physics-based fracture simulations motivate the development of machine-learning-based fracture evolution prediction with surrogate models. However, most existing machine-learning models for fracture prediction operate as black boxes and require large training datasets to generalize reliably. In this work, Thermodynamically Consistent Latent Dynamics Identification (tLaSDI) for parametric systems is introduced to predict fracture patterns and evolution in heterogeneous materials. The model learns the thermodynamic state variable in a lower-dimensional latent space using parametric GENERIC Formalism-Informed Neural Networks (pGFINNs) that enforce key thermodynamic principles, such as total energy conservation and entropy generation. This model embeds physics into the neural network architecture using a hard-constraint m
    
[^298]: 面向语言智能体的行动与预测协同演化

    Co-Evolving Acting and Forecasting for Language Agents

    [https://arxiv.org/abs/2610.20597](https://arxiv.org/abs/2610.20597)

    该论文提出在成功的同策略轨迹上引入多步动作序列作为监督目标，使行动生成与动作预测协同演化，从而充分利用中间步骤的监督信号，缓解语言智能体训练中长程 rollout 成本高、终端奖励稀疏的问题。

    

    训练语言智能体的瓶颈在于长程轨迹 rollout 的高昂成本，而标准的智能体强化学习（RL）仅从最终标量奖励中学习，导致中间步骤的监督信号在很大程度上未被利用。先前的工作通过两类方式利用这种监督：一是估计每步成功概率的价值目标，二是拟合环境响应的观测目标。但两者都难以良好构造：(1) 价值目标来自轨迹中每个步骤同一的单一标量信号；(2) 观测目标（如网页和终端日志）冗长且含噪，且只能向前预测一步。我们提出 \name，引入第三种监督目标——多步动作目标。在成功的同策略轨迹上，\name 训练策略主干生成每个历史之后所跟随的动作序列。当从先前历史进行预测时，每个动作也会受到监督，从而形成一个时间集成，……（摘要在此处截断）

    arXiv:2610.20597v1 Announce Type: cross  Abstract: Training language agents is bottlenecked by the high cost of long-horizon rollouts, yet standard agentic reinforcement learning (RL) learns only from a terminal scalar reward, leaving the supervision in intermediate steps largely unused. Prior work exploits this supervision through value targets that estimate per-step success probabilities or observation targets that fit environment response. Both are hard to construct well: (1) value targets come from the same single scalar signal for each step in the trajectory, and (2) observation targets, such as web pages and terminal logs, are long and noisy and reach only one step ahead. We propose \name, which adds a third kind of multi-step action targets. On successful on-policy trajectories, \name trains the policy backbone to generate the action sequences that follows each history. Each action is also supervised when forecast from the preceding histories, forming a temporal ensemble that th
    
[^299]: 事后特权RLVR：无需组采样的单次轨迹词元信用分配

    Hindsight-Privileged RLVR: Single-Rollout Token Credit Without Group Sampling

    [https://arxiv.org/abs/2610.20497](https://arxiv.org/abs/2610.20497)

    提出HP-RLVR，一种无需组采样的单次轨迹RLVR方法，通过比较因果视角与包含未来结果和验证器反馈的事后特权视角之间的Jensen–Shannon散度为每个词元分配有界信用，从而在所有尝试结果相同时仍能提供有效学习信号。

    

    可验证奖励的强化学习（RLVR）利用经过检验的任务结果来训练语言模型，但如何将最终奖励分配到各个推理词元或动作上仍然困难。以组相对策略优化（GRPO）为代表的组相对方法需要对每个提示采样多次尝试，但当所有尝试获得相同结果时，其基于奖励的学习信号便会失效。我们提出事后特权RLVR（HP-RLVR），这是一种单次轨迹方法，将已完成轨迹自身的未来转化为词元级的更新权重。生成策略快照在每个合格词元的因果历史条件下，以及在一个包含已实现未来和验证器反馈的事后上下文条件下分别对该词元进行打分，两个视角之间的归一化Jensen–Shannon散度相对于历史基线对奖励进行加权，从而为成功和失败的尝试都提供有界的词元信用，且无需参考答案或学习到的评论家模型。（原文摘要在此处截断）

    arXiv:2610.20497v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) trains language models from checked task outcomes, but assigning a terminal reward to individual reasoning tokens or actions remains difficult. Group-relative methods such as Group Relative Policy Optimization (GRPO) sample several attempts per prompt, but lose their reward-based learning signal when all attempts receive the same outcome. We propose Hindsight-Privileged RLVR (HP-RLVR), a single-rollout method that turns a completed trajectory's own future into token-level update weights. The generating policy snapshot scores each eligible token under its causal history and under a hindsight context containing the realized future and verifier feedback. A normalized Jensen--Shannon divergence between the two views weights the reward relative to a historical baseline, yielding bounded token credit for both successful and failed attempts, with no reference solution or learned critic. In p
    
[^300]: 扩展多语言专家混合模型：语言覆盖范围如何改变容量与计算分配

    Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation

    [https://arxiv.org/abs/2610.20247](https://arxiv.org/abs/2610.20247)

    本文提出了一种感知重复的联合缩放定律，揭示多语言MoE模型中活跃参数、训练词元、专家数量与语言覆盖范围之间的关系，并发现32个专家相比稠密模型可将语言从12种扩展到200种时的损失惩罚降低83%。

    

    多语言模型在各语言之间共享容量，因此增加语言数量可能会损害现有语言的表现。专家混合模型只需增加极少量的每词元计算即可扩展这一容量。鉴于这种额外容量，增加专家数量能否降低每种新增语言的代价？为回答这一问题，我们提出了一种感知重复的联合缩放定律，将活跃参数数量、每种语言的训练词元数量、专家数量与语言覆盖范围联系起来。该定律基于来自64个预训练配置的26,300个逐语言损失观测数据进行拟合，在留出配置上相比最强评估基线将预测误差降低了28%。在活跃参数数量和每种语言训练词元数量相同的情况下，32个专家相比稠密模型将语言从12种扩展到200种时的损失惩罚降低了83%。我们的定律进一步表明，更广的语言覆盖范围会在更大的参数范围内倾向于使用更多专家。

    arXiv:2610.20247v1 Announce Type: new  Abstract: Multilingual models share capacity across languages, so adding languages can hurt performance on existing ones. Mixture-of-experts models expand this capacity with little additional per-token computation. Given this extra capacity, does adding experts reduce what each additional language costs? To answer this question, we propose a repetition-aware joint scaling law relating number of active parameters, training tokens per language, number of experts, and language coverage. Fitted on 26,300 per-language loss observations from 64 pretrained configurations, our law reduces prediction error on held-out configurations by 28% relative to the strongest evaluated baseline. At matched number of active parameters and training tokens per language, 32 experts reduce the loss penalty of expanding from 12 to 200 languages by 83% compared with a dense model. Our law further shows that broader language coverage favors more experts across a wider range 
    
[^301]: 检索优于训练：基于相似性搜索的时序异常检测模型选择方法

    Retrieval Over Training: Similarity search-based Model Selection for Time Series Anomaly Detection

    [https://arxiv.org/abs/2610.20243](https://arxiv.org/abs/2610.20243)

    提出RAMSAD框架，通过相似性检索历史实例知识库而非训练模型的方式为新时间序列选择最优异常检测器，从而解决了现有模型选择方法在分布外（OOD）场景下精度大幅下降的问题。

    

    异常检测是时间序列分析中的一项基础任务，对众多应用的下游性能具有重要影响。尽管文献中已提出了大量的异常检测方法，但近期的基准研究表明，没有任何单一的检测器能够在高度异构的时间序列上始终表现最佳。因此，一种实用且可扩展的解决方案是开发一种模型选择方法，针对给定的时间序列选出最有可能表现良好的异常检测器。然而，现有文献中提出的模型选择方法在应用于分布外（Out-of-Distribution, OOD）场景时会出现显著的精度下降。本文针对上述局限性，提出了RAMSAD——一个基于检索的时序异常检测模型选择框架。给定一个新的时间序列，我们的方法会查询一个由先前观察到的实例构成的知识库，并检索……（原文摘要到此截断）

    arXiv:2610.20243v1 Announce Type: new  Abstract: Anomaly detection is a fundamental task for time-series analytics, with important implications for the downstream performance of many applications. Despite the large number of anomaly detection methods proposed in the literature, recent benchmark studies have shown that no single detector performs best across highly heterogeneous time series. Therefore, a practical and scalable solution is to develop a model-selection method that, for a given time series, selects the anomaly detector most likely to perform well. Nevertheless, the model selection approaches proposed in the literature suffer from a significant drop in accuracy when applied in Out-of-Distribution (OOD) settings. In this paper, we tackle the aforementioned limitation and propose RAMSAD, a retrieval-based framework for model selection in time-series anomaly detection. Given a new time series, our method queries a knowledge base of previously observed instances, retrieves the 
    
[^302]: RM-OPD：面向生成式奖励模型的同策略蒸馏

    RM-OPD: On-Policy Distillation for Generative Reward Models

    [https://arxiv.org/abs/2610.20174](https://arxiv.org/abs/2610.20174)

    该论文提出RM-OPD两阶段框架，先通过单领域GRPO微调训练各领域专家教师模型，再通过路由同策略蒸馏将其整合为单一学生模型，从而解决Mix-RL训练生成式奖励模型时的奖励稀疏与跨领域干扰问题。

    

    基于混合领域偏好数据的强化学习（Mix-RL）已成为训练通用生成式奖励模型（GRM）的主流方法。然而，Mix-RL 存在两个关键瓶颈：由结果奖励导致的奖励稀疏性，以及联合优化异构能力所引起的跨领域干扰，二者共同导致了能力获取不均衡和普遍的奖励欺骗问题。受同策略蒸馏（On-Policy Distillation, OPD）在多领域能力融合中成功应用的启发，我们提出了 RM-OPD（通过同策略蒸馏进行奖励建模），这是一个将专家能力获取与能力整合相分离的两阶段框架。首先，我们通过单领域 GRPO 微调培养领域专精的教师模型，使每个专家能够在隔离状态下达到其性能上限；然后，我们通过路由同策略蒸馏将这些教师模型整合到一个学生模型中。

    arXiv:2610.20174v1 Announce Type: cross  Abstract: Reinforcement Learning on Mixed-domain preference data (Mix-RL) has become the mainstream approach for training general-purpose Generative Reward Models (GRMs). However, Mix-RL suffers from two critical bottlenecks: the reward sparsity induced by outcome rewards, and the cross-domain interference arising from jointly optimizing heterogeneous capabilities, which together give rise to imbalanced capability acquisition and pervasive reward hacking. Inspired by the success of On-Policy Distillation (OPD) in multi-domain capability integration, we introduce RM-OPD (Reward Modeling via On-Policy Distillation), a two-stage framework that separates specialist acquisition from capability integration. First, we cultivate domain-specialized teacher models via single-domain GRPO fine-tuning, allowing each expert to reach its performance ceiling in isolation; then we consolidate these teachers into one student model through (i) routed on-policy dis
    
[^303]: 基于流匹配的上下文批量黑盒优化

    In-Context Batch Black-box Optimization via Flow Matching

    [https://arxiv.org/abs/2610.20138](https://arxiv.org/abs/2610.20138)

    提出FMTR方法，利用流匹配离线学习历史条件化的连续提议分布，实现可变大小批量的高效采样，并在优化过程中自适应平衡探索与利用，改进批量黑盒优化。

    

    批量黑盒优化对于昂贵系统非常重要，在这类系统中可以并行评估多个候选方案，但只能负担少数几轮评估。在这种情况下，有效的批量构建必须在识别多个有前景区域的同时避免冗余评估，而传统贝叶斯优化随着批量大小的增长，可能需要越来越复杂的批量选择程序。我们提出了汤普森采样精炼流匹配（FMTR），这是一种摊销方法，可以离线学习一个连续的、以历史为条件的提议分布，从中可以采样可变大小的批量。FMTR通过条件流匹配在汤普森采样的argmax上进行离线训练，这些argmax通过一个依赖于优化视野的朗之万前推映射向高价值区域精炼，从而在优化过程中自适应地调整探索-利用权衡。在测试时，模型参数保持固定，批量生成只需（原文在此截断）。

    arXiv:2610.20138v1 Announce Type: new  Abstract: Batch black-box optimization is important for expensive systems where many candidates can be evaluated in parallel, yet only a few rounds of evaluation are affordable. In this regime, effective batch construction must identify multiple promising regions while avoiding redundant evaluations, and conventional Bayesian optimization can require increasingly difficult batch-selection procedures as the batch size grows. We introduce Flow Matching for Thompson-Sampling Refinement (FMTR), an amortized approach that learns offline a continuous, history-conditioned proposal distribution from which variable-size batches can be sampled. FMTR is trained offline by conditional flow matching on Thompson-sampling argmaxes refined toward high-value regions via a horizon-dependent Langevin pushforward map, adapting the exploration-exploitation tradeoff over the optimization horizon. At test time, the model parameters remain fixed and batch generation requ
    
[^304]: 穷举式规范骨架生成：面向可分解符号回归的形式化数据集

    Exhaustive Canonical Skeletons Generation: A Formal Dataset for Decomposable Symbolic Regression

    [https://arxiv.org/abs/2610.20108](https://arxiv.org/abs/2610.20108)

    该论文提出基于确定性受限上下文无关文法穷举生成的单变量规范骨架形式化数据集，消除了符号回归数据集中的结构冗余，并支持在不假设线性可分性的前提下求解多变量符号回归问题。

    

    神经符号回归（SR）模型需要多样化的数学表达式数据集以实现稳健训练。然而，现有数据集主要依赖随机采样方法，存在结构冗余度高和函数覆盖不完整的问题。为解决这一问题，我们通过确定性受限上下文无关文法生成了一份穷举式的单变量规范骨架数据集。规范骨架作为一种通用的结构表示，能够涵盖更具体的函数形式，从而消除冗余表达式并避免重复的计算开销。尽管我们的数据集仅由单变量骨架组成，但其设计旨在支持可分解SR范式，从而在无需假设线性可分性的情况下解决多变量SR问题。此外，我们还为该数据集配套了一种采样策略，可生成干净的数值数据，而不会触发数值不稳定性或依赖人工（原文在此处截断）。

    arXiv:2610.20108v1 Announce Type: new  Abstract: Neural symbolic regression (SR) models require diverse datasets of mathematical expressions for robust training. However, existing datasets rely predominantly on stochastic sampling methods, which suffer from high structural redundancy and incomplete functional coverage. To address this, we introduce an exhaustive dataset of univariate canonical skeletons generated via a deterministic constrained context-free grammar. A canonical skeleton acts as a generic structural representation that subsumes more specific functional forms, eliminating redundant expressions and avoiding repeated computational efforts. Although our dataset consists of univariate skeletons, it is designed to support the decomposable SR paradigm to solve multivariate SR problems without assuming linear separability. Furthermore, we pair this dataset with a sampling strategy that generates clean numerical data without triggering numerical instabilities or relying on artif
    
[^305]: TIMERSI：迈向时间序列基础模型的自改进智能体化部署

    TIMERSI: Towards Self-Improved Agentic Deployment for Time-Series Foundation Models

    [https://arxiv.org/abs/2610.19725](https://arxiv.org/abs/2610.19725)

    TIMERSI是一个围绕冻结时间序列基础模型的自改进智能体部署系统，由编码智能体根据任务描述迭代生成并修正预测流水线，并在部署时通过自适应选择器挑选最优流水线，在198个任务上超越了所有对比方法。

    

    时间序列基础模型具有强大的零样本预测能力，但在真实应用中的部署仍需针对具体任务设计模型的输入与输出。直接推理可能无法利用领域知识，也无法解决反复出现的预测误差，而定制的流水线则需要大量的人类专业知识与精力。我们提出了TIMERSI，一个围绕冻结预测模型构建的自改进智能体化部署系统。一个编码智能体将任务描述转化为可执行的流水线修改，并在历史预测窗口上测试每次修改，利用观察到的误差指导后续修改。在部署阶段，自适应选择器根据相似条件下已完成的预测，在多个候选流水线中进行选择。在TIME和FEV-bench的198个任务上，TIMERSI在六项点预测与概率预测指标上超越了所有提交的方法。受控比较进一步验证了其设计的价值。

    arXiv:2610.19725v1 Announce Type: new  Abstract: Time-series foundation models provide strong zero-shot forecast, but deployment for real- world applications still requires task-specific designs for model inputs and outputs. Direct inference can leave domain knowledge and re- curring forecast errors unaddressed, while tai- lored pipelines require substantial human ex- pertise and effort. We introduce TIMERSI, a self-improving agentic deployment system around frozen forecasting models. A cod- ing agent translates task descriptions into ex- ecutable pipeline changes. It tests each re- vision on historical forecasting windows and uses observed errors to guide subsequent re- visions. During deployment, an adaptive se- lector chooses among the resulting pipelines using completed forecasts under similar con- ditions. Across 198 tasks on TIME and FEV- bench, TIMERSI outperforms all submissions on six point and probabilistic forecasting met- rics. Controlled comparisons support the value of fe
    
[^306]: ReTerm：面向自我改进终端智能体的恢复训练

    ReTerm: Recovery Training for Self-Improving Terminal Agents

    [https://arxiv.org/abs/2610.18484](https://arxiv.org/abs/2610.18484)

    提出ReTerm终端智能体，通过恢复示范数据集ReTerm-10K进行SFT冷启动，并利用从自身失败状态分叉出多次恢复尝试的在线强化学习算法ReTerm-GRPO，让智能体从自身恢复经验中自我改进，在Terminal-Bench 2.0上取得34.8%的Pass@1成绩。

    

    终端智能体在复杂编程任务中展现出强大的能力。然而，它们在长时间运行过程中往往难以从错误中恢复（例如命令错误、有缺陷的实现），最终导致任务失败。为此，我们提出了ReTerm，一种通过学习如何从自身失败的终端任务中进行恢复而训练的自我改进终端智能体。为支持其训练，我们构建了一个定制化的数据流水线，并生成了ReTerm-10K，这是一个高质量数据集，为SFT冷启动提供了有效的恢复示范。此外，我们提出了ReTerm-GRPO，一种在线强化学习算法，它从智能体自身的失败状态分叉出多个独立的自我恢复尝试。通过学习这些尝试是否成功完成任务，智能体通过自身的恢复经验实现自我改进。基于Qwen3.5-9B构建的ReTerm-9B在Terminal-Bench 2.0上实现了34.8%的Pass@1，较基础模型有所提升。

    arXiv:2610.18484v1 Announce Type: new  Abstract: Terminal agents have shown strong abilities in complex coding tasks. However, they often struggle to recover from errors during long-running execution (e.g., command errors, faulty implementations), eventually leading to task failures. To this end, we present ReTerm, a self-improving terminal agent trained by learning how to recover from its own failed terminal tasks. To support its training, we construct a customized data pipeline and produce ReTerm-10K, a high-quality dataset that provides effective recovery demonstrations for SFT cold start. Furthermore, we propose ReTerm-GRPO, an online RL algorithm that forks from the agent's own failure states to generate multiple independent self-recovery attempts. By learning from whether these attempts successfully complete the task, the agent self-improves through its own recovery experience. Built on Qwen3.5-9B, our ReTerm-9B achieves 34.8% Pass@1 on Terminal-Bench 2.0, improving over the base
    
[^307]: 何时学习到的块能保留因果意义？块可辨识性与构造性因果抽象

    When Do Learned Blocks Preserve Causal Meaning? Block Identifiability and Constructive Causal Abstractions

    [https://arxiv.org/abs/2610.18155](https://arxiv.org/abs/2610.18155)

    该论文证明了块对角可逆变换加上相容坐标置换是保留学习表示中块因果意义的最大变换类，从而将块可辨识性与构造性因果抽象联系起来，使高层干预能够在独立学习到的表示之间迁移。

    

    标准可辨识性理论假设一维潜在因子，但学习到的表示可以在多维块中编码潜在信息，而这些块的内部坐标并非唯一确定。当这些块被用来表示更高层次的概念并对其进行干预时，尚不清楚它们的因果意义是否能在块可辨识性未能解决的其他变换下保留下来。我们将块可辨识性与构造性因果抽象联系起来，证明块对角可逆变换加上相容的坐标置换构成了能保留基于这些块构建的所有构造性因果抽象的最大变换类。这一保证假设块与概念之间的对应关系随表示一起更新。一个使用加性自编码器的小型合成实验表明，高层干预可以在独立学习到的表示之间进行迁移。

    arXiv:2610.18155v1 Announce Type: cross  Abstract: Standard identifiability theory assumes one-dimensional latent factors, but learned representations can encode latent information across multidimensional blocks whose internal coordinates are not uniquely determined. When such blocks are used to represent and intervene on higher-level concepts, it is unclear whether their causal meaning survives the transformations left unresolved by block-identifiability. We connect block-identifiability to constructive causal abstraction. We show that block-diagonal invertible transformations followed by compatible coordinate permutations form the maximal class that preserves every constructive causal abstraction built on those blocks. This guarantee assumes that the correspondence between blocks and concepts is updated with the representation. A small synthetic experiment with additive autoencoders shows that high-level interventions can be transferred between independently learned representations w
    
[^308]: RT-J：用于上下文高效预测的关系型Transformer大规模预训练

    RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions

    [https://arxiv.org/abs/2610.17867](https://arxiv.org/abs/2610.17867)

    RT-J是一个8500万参数的关系型基础模型，通过在639个网络爬取数据库和LLM生成的预测任务上利用长上下文预训练，仅需数百个上下文样本即可实现准确预测，所需样本量比现有方法减少10倍。

    

    用于表格和关系型数据库的基础模型在各个科学和工业领域的预测任务中应用日益广泛。然而，将现有模型适配到未见过的数据集需要数万个样本。在此，我们推出了RT-J，一个8500万参数的关系型基础模型（RFM），即使只有数百个甚至更少的上下文样本也能做出准确预测。为实现这一目标，我们首先构建了"Join"数据集，这是一个大型数据集合，包含639个网络爬取的数据库以及6,023个由大语言模型生成的预测任务。我们使用长达8,192个数据库单元格的长上下文进行预训练，并通过一种新颖的轻量级随机游走检索器获取上下文样本。最后，我们利用上下文调优和集成方法来扩展测试时计算。在来自RelBench的上下文回归任务上，RT-J平均只需少10倍的样本即可匹敌采用LLM智能体或RDBLearn特征化器以及TabICLv2或Ligh（摘要截断）的强大特征工程流水线。

    arXiv:2610.17867v2 Announce Type: cross  Abstract: Foundation models for tables and relational databases are increasingly used for predictive tasks across various scientific and industrial domains. However, adapting existing models to unseen datasets requires tens of thousands of examples. Here we introduce RT-J, an 85M-parameter relational foundation model (RFM) that makes accurate predictions even with hundreds of in-context examples or fewer. To achieve this, we first assemble the Join, a large collection of 639 web-crawled databases with 6,023 LLM-generated forecasting tasks. We pretrain with long contexts of up to 8,192 database cells, using in-context examples from a novel lightweight random-walk retriever. Finally, we leverage context tuning and ensembling to scale up test-time compute. On in-context regression tasks from RelBench, RT-J requires 10x fewer examples on average to match strong feature engineering pipelines with LLM-agent or RDBLearn featurizers and TabICLv2 or Ligh
    
[^309]: Agentic-MatriBO：面向材料与化学设计的能力编译型智能体贝叶斯优化

    Agentic-MatriBO: Capability-Compiled Agentic Bayesian Optimization for Materials and Chemical Design

    [https://arxiv.org/abs/2610.17832](https://arxiv.org/abs/2610.17832)

    提出了能力编译型智能体贝叶斯优化框架Agentic-MatriBO，将BO组件表示为可组合的能力，由大语言模型策略提出优化配方，经编译器检查和确定性执行层验证后用于材料与化学设计。

    

    材料与化学优化必须在有限预算下，基于序列化观测来选择下一次合成、表征或高保真仿真。大语言模型能够利用任务知识提出调整方案，但组合代理模型、采集函数与搜索策略时，会受到兼容性、参数依赖性和证据要求的限制。我们提出了Agentic-MatriBO，这是一个能力编译型的智能体贝叶斯优化（BO）框架，它将BO组件和选择条件表示为可组合的能力。语言策略提出BO配方和候选计划；编译器根据注册的能力目录对提案进行检查；确定性执行层负责验证并提交候选方案。Oracle观测结果被存储在研究记忆中，并为后续轮次提供信息支持。我们在四个离线有限候选池——COF、PCE10、Buch……上与八个基线方法进行了比较。

    arXiv:2610.17832v1 Announce Type: cross  Abstract: Materials and chemical optimization must choose the next synthesis, characterization, or high-fidelity simulation from sequential observations under a limited budget. Large language models can use task knowledge to propose adjustments, but composing surrogates, acquisition functions, and search strategies is constrained by compatibility, parameter dependencies, and evidence requirements. We present Agentic-MatriBO, a capability-compiled agentic Bayesian optimization (BO) framework that represents BO components and selection conditions as composable capabilities. A language policy proposes BO recipes and candidate plans; a compiler checks proposals against the registered capability catalog; and a deterministic execution layer validates and commits candidates. Oracle observations are stored in research memory and inform later rounds. We compare Agentic-MatriBO with eight baselines on four offline finite candidate pools---COF, PCE10, Buch
    
[^310]: 线性赌博机的扰动跟随领导者算法：一般凸集上的 $\sqrt{T}$ 遗憾

    Follow the Perturbed Leader for Linear Bandits: $\sqrt{T}$ Regret over General Convex Sets

    [https://arxiv.org/abs/2610.17818](https://arxiv.org/abs/2610.17818)

    本文提出了首个基于FTPL的对抗性线性赌博机算法，通过一个适应估计奖励的扰动分布，在一般凸动作集上实现了 $\sqrt{T}$ 遗憾，改进了此前 $T^{2/3}$ 的遗憾界，同时比连续指数权重方法具有更好的计算效率和更简单的实现。

    

    我们提出了首个基于扰动跟随领导者（Follow-the-Perturbed-Leader, FTPL）的对抗性线性赌博机算法，该算法在一般凸动作集上实现了 $\sqrt{T}$ 遗憾。我们的算法仅使用线性优化预言机即可达到 $\widetilde{\mathcal{O}}(d^{5/4}\sqrt{T})$ 的遗憾；当进一步可获得最优设计探索分布时，遗憾可达到接近最优的 $\widetilde{\mathcal{O}}(d\sqrt{T})$。这改进了 Dani 和 Hayes（2006）的 FTPL 算法在一般凸集上 $T^{2/3}$ 的遗憾，并将 Lévy 等人（2026）在两个特殊动作集（超立方体和欧几里得球）上的 $\sqrt{T}$ 遗憾结果进行了推广。与之前基于连续指数权重的预言机高效线性赌博机算法相比，FTPL 方法通常具有更好的计算效率，且实现起来要简单得多。关键的算法要素是一个能适应估计奖励的扰动分布。

    arXiv:2610.17818v1 Announce Type: cross  Abstract: We give the first adversarial linear bandit algorithm based on Follow-the-Perturbed-Leader (FTPL) that achieves $\sqrt{T}$ regret over general convex action sets. Our algorithm attains $\widetilde{\mathcal{O}}(d^{5/4}\sqrt{T})$ regret using only linear optimization oracles, and the near-optimal $\widetilde{\mathcal{O}}(d\sqrt{T})$ regret when an optimal-design exploration distribution is further available. This improves on the $T^{2/3}$ regret of the FTPL algorithm of Dani and Hayes (2006) for general convex sets, and generalizes the $\sqrt{T}$ regret of L\'evy et al. (2026) over two special action sets (hypercube and Euclidean ball). Compared to previous oracle-efficient linear bandit algorithms based on continuous exponential weights, the FTPL approach generally has better computational efficiency, and is much simpler to implement.   The key algorithmic element is a perturbation distribution that adapts to the estimated rewards. Its 
    
[^311]: SPARCL：用于胸部X光分析中主动学习的空间自适应区域覆盖方法

    SPARCL: Spatially Adaptive Regional Coverage for Active Learning in Chest X-ray Analysis

    [https://arxiv.org/abs/2610.17808](https://arxiv.org/abs/2610.17808)

    提出SPARCL方法，利用凹覆盖函数动态平衡预测不确定性与空间覆盖率，解决胸部X光病灶分割主动学习中样本空间分布不均导致性能不一致的问题，并凭借单调次模性保证近最优的批量选择。

    

    胸部X光（CXR）病灶分割的像素级标注成本高昂，因此主动学习（AL）至关重要。然而，现有的主动学习策略未能考虑所选样本的空间分布，导致在局灶性病变（如主动脉钙化）和弥漫性病变（如结节）上的表现不一致。本文提出了SPARCL（空间自适应区域覆盖主动学习）方法，以动态平衡预测不确定性与累积空间覆盖率。通过采用凹覆盖函数，SPARCL对先前已标注的区域施加收益递减效应，从而自然地将标注预算引导至标注不足的区域。此外，当空间变化缺失时，熵门控机制会将算法恢复为标准的不确定性采样。该覆盖目标是单调次模的，从而保证了贪心批量选择具有近最优性。为了解决诸如聚合指标如何……

    arXiv:2610.17808v2 Announce Type: replace-cross  Abstract: Pixel-level annotation for chest X-ray (CXR) lesion segmentation is expensive, making active learning (AL) essential. However, existing AL strategies fail to account for the spatial distribution of selected samples, resulting in inconsistent performance across focal (e.g., aortic calcification) and diffuse (e.g., nodules) pathologies. In this paper, we propose SPARCL (Spatially Adaptive Regional Coverage for Active Learning) to dynamically balance predictive uncertainty with cumulative spatial coverage. By employing a concave coverage function, SPARCL imposes diminishing returns on previously annotated regions, naturally directing the annotation budget toward under-labeled areas. Furthermore, an entropy gate reverts the algorithm to standard uncertainty sampling when spatial variation is absent. This coverage objective is monotone submodular, guaranteeing near-optimal greedy batch selection. To address how aggregate metrics lik
    
[^312]: 形态学感知的深度学习从全切片图像预测结直肠癌生存期

    Morphology-aware deep learning predicts colorectal cancer survival from whole-slide images

    [https://arxiv.org/abs/2610.17603](https://arxiv.org/abs/2610.17603)

    该研究提出MorphContrast——一种形态学感知的对比多实例学习框架，利用组织标签作为结构监督，从H&E全切片图像有效预测结直肠癌患者的五年生存期。

    

    背景：基于苏木精-伊红（H&E）染色全切片图像的五年生存预测可支持风险分层，但该任务属于弱监督学习，且携带预后信息的组织形态学尚未得到完全表征。方法：我们开展了一项回顾性二次分析，使用TCGA-COAD/READ（n=562）进行形态学特异性分析，使用Alliance/CALGB 89803（n=424）进行模型开发，并使用SurGen（n=218）进行外部评估。我们提出了MorphContrast，一种形态学感知的对比多实例学习框架，它将组织标签作为结构监督用于五年生存预测。MorphContrast将门控注意力池化与三个辅助目标相结合：patch级别的形态学标签定义患者内部的对比，结局的一致性或不一致性定义形态学匹配的患者间对比，而多样性正则化则抑制表征坍塌。

    arXiv:2610.17603v1 Announce Type: cross  Abstract: Background: Five-year survival prediction from hematoxylin-and-eosin whole-slide images could support risk stratification, but the task is weakly supervised, and the tissue morphologies carrying prognostic information remain incompletely characterized.   Methods: We conducted a retrospective secondary analysis using TCGA-COAD/READ (n=562) for morphology-specific analysis, Alliance/ CALGB 89803 (n=424) for model development, and SurGen (n=218) for external evaluation. We propose MorphContrast, a morphology-aware contrastive multiple-instance learning framework that uses tissue labels as structural supervision for five-year survival prediction. MorphContrast combines gated-attention pooling with three auxiliary objectives: patch-level morphology labels define within-patient contrasts, outcome concordance or discordance defines cross-patient contrasts within matched morphologies, and diversity regularization discourages representational c
    
[^313]: 早期Token相关性的贝特拟设来源：一种用于并行提议的保持边际分布的构造方法

    A Bethe-Ansatz Source of Early Token Correlations: A marginal-preserving construction for parallel proposals

    [https://arxiv.org/abs/2610.16781](https://arxiv.org/abs/2610.16781)

    该论文从周期XXZ链的坐标贝特拟设构造出保持token边际分布的并行提议联合分布律，并证明即使一、二token边际完全匹配，由 δ 量化的高阶相关性差异依然存在，从而揭示了早期token相关性的可积系统来源。

    

    并行token提议需要一个联合分布律，而不仅仅是精确的token边际分布。我们从周期XXZ链构造坐标贝特拟设分布律，并在相同的一站点与二站点矩条件下，将其与最大熵成对投影进行比较。在一个八站点、三激发扇区中，对称性使得完整的概率分布律差异是一维的，由一个连续的三站点矩 δ 决定。沿贝特分支，这一差距会改变符号，并恰好在 Δ=(3+√3)/2 处消失，此时本征态分布律变为成对独立的。一个非线性双token读出在保持各自token边际分布的同时，暴露出距离 2|δ|；一个直接的三token读出在保持所有一token与二token边际分布的同时，暴露出 4|δ|。在 Δ=1.5 时，这些确定性二元距离分别为 0.02654573 和 0.05309145。我们给出了一个精确的目标对齐公式，并证明了一个严格的投机解码接受前缀优势。

    arXiv:2610.16781v2 Announce Type: replace  Abstract: Parallel token proposals need a joint law, not merely accurate token marginals. We construct coordinate-Bethe-ansatz laws from the periodic XXZ chain and compare them with their maximum-entropy pairwise projections at identical one- and two-site moments. In an eight-site, three-excitation sector, symmetry makes the full probability-law difference one-dimensional, determined by a contiguous three-site moment $\delta$. Along the Bethe branch, this gap changes sign and vanishes exactly at $\Delta=(3+\sqrt3)/2$, where the eigenstate law becomes pairwise. A nonlinear two-token readout preserves separate token marginals yet exposes a distance $2|\delta|$; a direct three-token readout preserves all one- and two-token marginals yet exposes $4|\delta|$. At $\Delta=1.5$ these deterministic binary distances are $0.02654573$ and $0.05309145$. We give an exact target-alignment formula and prove a strict speculative accepted-prefix advantage throu
    
[^314]: PMB-NN：基于生理模型的神经网络用于1型糖尿病的血糖预测

    PMB-NN: Physiological-Model-Based Neural Network for Glucose Forecasting in Type 1 Diabetes

    [https://arxiv.org/abs/2610.15456](https://arxiv.org/abs/2610.15456)

    该论文提出一种基于生理模型的神经网络框架（PMB-NN），利用个性化四参数生理模型约束时间卷积网络的训练，实现了1型糖尿病患者30至120分钟时间范围内的准确血糖预测。

    

    准确且符合生理一致性的血糖预测可以帮助1型糖尿病患者对即将发生的血糖波动做出应对。我们将基于生理模型的神经网络（PMB-NN）框架应用于血糖预测。时间卷积网络（TCN）使用180分钟的历史数据，并在训练过程中受到个性化四参数生理模型的约束。主要的PMB-NN结合了预测损失和神经-生理一致性损失；一个三项变体还额外将生理预测拟合到实测血糖值上。数据包括25名AZT1D参与者6-8周的连续血糖监测、胰岛素泵和饮食记录。个体化模型采用按时间顺序的70/10/20训练/验证/测试划分，预测时间范围为30、60和120分钟。在使用基础胰岛素记录的情况下，平均绝对误差中位数分别为16.24、25.00和30.20 mg/dL；在这些记录可用之前，分别为15.38、24.21和31.79 mg/dL。（注：原文摘要在末尾处被截断）

    arXiv:2610.15456v2 Announce Type: replace-cross  Abstract: Accurate, physiologically consistent glucose forecasts could help people with type 1 diabetes respond to impending excursions. We applied our physiological-model-based neural network (PMB--NN) framework to glucose forecasting. A temporal convolutional network (TCN) used 180 min of history, constrained during training by a personalized four-parameter physiological model. The main PMB--NN combined forecast and neural--physiological agreement losses; a three-term variant additionally fitted physiological predictions to measured glucose. Data comprised 6--8 weeks of continuous glucose monitoring, pump, and meal records from 25 AZT1D participants. Person-specific models used chronological 70/10/20 train/validation/test splits and 30-, 60-, and 120-min horizons. Median mean absolute errors were 16.24, 25.00, and 30.20 mg/dL with basal insulin records and 15.38, 24.21, and 31.79 mg/dL before these records became available. In the latt
    
[^315]: 通过谱对齐实现世界建模

    World Modeling through Spectral Alignment

    [https://arxiv.org/abs/2610.15429](https://arxiv.org/abs/2610.15429)

    SpecWM是一个通过目标相似度核显式指定观测间期望关系并训练编码器在嵌入空间中复现这些关系的JEPA世界建模框架，核的选择决定了哪些物理量可从表示中恢复，其中时间核仅需观测顺序即可实现全自监督学习。

    

    我们如何告诉世界模型需要保留哪些信息？我们从谱表示学习的视角来研究这个问题。我们提出了SpecWM，这是一个JEPA框架，它通过目标相似度核显式地指定观测之间期望的关系，并训练编码器在其嵌入几何结构中复现这些关系。不同的核提供不同的监督信号，范围从物理状态信息到仅需轨迹内观测顺序的全自监督时间核。我们刻画了该目标函数的全局最优解，建立了线性可恢复性以及对干扰信息不变性的条件，并在理想化设置下将时间核相似度与行为后继度量联系起来。实证结果表明，核的选择强烈决定了哪些物理量可以从表示中恢复出来。更重要的是，时间谱……

    arXiv:2610.15429v1 Announce Type: cross  Abstract: How do we tell a world model what information to preserve? We study this question through the lens of spectral representation learning. We introduce SpecWM, a JEPA framework that explicitly specifies desired relationships between observations through a target similarity kernel and trains the encoder to reproduce those relationships in its embedding geometry. Different kernels provide different supervision, ranging from physical state information to a fully self-supervised temporal kernel requiring only the ordering of observations within trajectories. We characterize the global optima of the objective, establish conditions for linear recoverability and invariance to nuisance information, and connect temporal-kernel similarity to the behavioral successor measure under an idealized setting. Empirically, the choice of kernel strongly controls which physical quantities are recoverable from the representation. More importantly, temporal spe
    
[^316]: CIViC：面向代理任务中音频条件视觉预测的证据门控协议

    CIViC: Evidence Gates for Audio-Conditioned Visual Prediction on Proxy Tasks

    [https://arxiv.org/abs/2610.15327](https://arxiv.org/abs/2610.15327)

    本文提出CIViC四门控验证协议，通过将解释与干预、留出任务比较或端点控制相绑定，来检验音频条件视觉预测任务是否真正使用了预期的音频信息，并在面部光栅测试、合成音频混合及SpeakerVid代理任务上验证了该方法。

    

    移除音频可能会改变视觉预测器的输出，但这并不能表明该任务是否真正使用了预期的音频信息。我们提出了条件输入与视觉解释检查（CIViC），这是一个四门控协议，它将解释与干预、留出任务比较或端点控制联系起来。我们在演员不相交的面部光栅测试、合成音频特征混合数据以及源视频不相交的SpeakerVid代理任务上演示了CIViC。在静态面部光栅测试中，跨演员音频替换以较小的效应量满足了预注册的检测规则；在另一个独立的时间端点上，恒定音频拟合在四个匹配的随机种子中的得分优于音频训练拟合。在合成混合实验中，条件增强拟合的1000步RMSE高于仅音频拟合（0.1807 对 0.1707），且两者差距在训练过程中逐渐缩小。在SpeakerVid代理任务中，五特征置零拟合（将五个低级音频摘要置零；id……（原文摘要此处被截断）

    arXiv:2610.15327v2 Announce Type: replace  Abstract: Removing audio can shift a visual predictor's output without showing whether the task uses the intended audio information. We introduce Conditional Input and Visual Interpretation Checks (CIViC), a four-gate protocol that links an interpretation to an intervention, held-out task comparison, or endpoint control. We demonstrate CIViC on actor-disjoint face-raster tests, a synthetic audio-feature mixture, and source-video-disjoint SpeakerVid proxy tasks. In the static face-raster test, cross-actor audio replacement meets the registered detection rule with a small effect; on a separate temporal endpoint, constant-audio fits score better than audio-trained fits in four matched seeds. In the synthetic mixture, the condition-augmented fit has higher 1,000-step RMSE than the audio-only fit (0.1807 vs. 0.1707), with the gap shrinking during training. In the SpeakerVid proxy, the five-feature-zero fit (five low-level audio summaries zeroed; id
    
[^317]: AlphaRLM：面向基于文献的符号化因子发现的语言模型对齐

    AlphaRLM: Aligning Language Models for Literature-Grounded Symbolic Factor Discovery

    [https://arxiv.org/abs/2610.14959](https://arxiv.org/abs/2610.14959)

    AlphaRLM是一个强化学习框架，通过文献引导算子组装（LEGO）提供语义先验，并借助可执行有效性与分组收益分离等奖励将生成策略与市场反馈对齐，从而训练大语言模型直接从金融文献中构造可执行的符号化Alpha因子。

    

    公式化Alpha挖掘旨在从庞大且充满噪声的搜索空间中发现能够捕捉预测性市场信号的可执行表达式。近期基于大语言模型（LLM）的方法主要依赖提示工程、迭代搜索或种子因子进化，而通过策略学习直接构造因子的路径较少被探索。为解决这一问题，我们提出了AlphaRLM，一个强化学习框架，用于训练大语言模型在无需输入因子表达式的情况下，直接从金融文献中构造可执行的符号化因子。AlphaRLM引入了文献引导算子组装机制，将因子构造建立在文献衍生的思想与可执行算子之上，为探索提供语义先验。它还通过可执行有效性奖励和分组收益分离奖励（后者能够捕捉因子各分位数间的非单调结构）将生成策略与市场反馈对齐。在沪深300、中证1000和标普500上的大量实验表明，AlphaRLM……（摘要至此截断）

    arXiv:2610.14959v2 Announce Type: new  Abstract: Formulaic alpha mining seeks to discover executable expressions that capture predictive market signals in a vast, noisy search space. Recent LLM-based approaches use prompting, iterative search, or seed-factor evolution, leaving direct factor construction through policy learning less explored. To address this, we introduce AlphaRLM, a reinforcement-learning framework that trains LLMs to construct executable symbolic factors from financial literature without input factor expressions. AlphaRLM incorporates Literature-Guided Operator Assembly (LEGO), which conditions factor construction on literature-derived ideas and executable operators, providing semantic priors for exploration. It further aligns the generation policy with market feedback through executable-validity and grouped-return-separation rewards that capture non-monotonic structures across factor quantiles. Extensive experiments on CSI 300, CSI 1000, and S&P 500 show that AlphaRL
    
[^318]: TLA-Verus：使用LLM智能体生成和求解时序证明义务

    TLA-Verus: Generating and Solving Temporal Proof Obligations with LLM Agents

    [https://arxiv.org/abs/2610.14923](https://arxiv.org/abs/2610.14923)

    该论文提出TLA-Verus框架，通过确定性转译器和LLM智能体将TLA+规范翻译为Rust验证器Verus，能够大规模生成证明义务并产出完全可机器检查的安全性与活性证明，弥补了TLA+在时序属性验证上的不足。

    

    TLA+（动作时序逻辑）形式化规范工具包已被广泛应用于对现代分布式软件系统进行建模，包括数据库、存储系统、账本和协调协议。尽管它在软件验证方面存在两个重大局限，但仍然备受欢迎：首先，TLA+无法检测软件实现与数学规范之间的偏差；其次，TLC和TLAPS无法为任意时序逻辑属性提供令人满意的证明。我们通过确定性转译器（transpiler）和智能体翻译器（agentic translator）将TLA+规范翻译为Verus（一种为Rust构建的自动主动式验证器），从而解决了第二个局限，并为解决第一个局限迈出了第一步。我们证明，基于LLM的流水线能够大规模地从TLA+规范生成证明义务，并产生完全可机器检查的安全性与活性属性证明。

    arXiv:2610.14923v2 Announce Type: replace  Abstract: The TLA+ (temporal logic of actions) formal specification toolkit has seen widespread adoption in modeling modern distributed software systems including databases, storage systems, ledgers and coordination protocols. This popularity is despite its two significant limitations for software verification: Firstly, TLA+ offers no way to detect drift between software implementation and mathematical specification; secondly, TLC and TLAPS cannot provide satisfactory proofs of arbitrary temporal logic properties. We address the second limitation and take a first step towards the first by translating TLA+ specifications into Verus, an auto-active verifier built for Rust, via both a deterministic transpiler and an agentic translator. We demonstrate that our LLM-based pipeline is able to generate proof obligations from TLA+ specifications at scale, and to produce fully machine checkable proofs of both safety and liveness properties. Since tempor
    
[^319]: 自改进的数据合成可跨领域、跨任务扩展嵌入模型

    Self-improving data synthesis can scale embedding models across domains and tasks

    [https://arxiv.org/abs/2610.14730](https://arxiv.org/abs/2610.14730)

    提出了SYNTRA自改进检索数据合成方法，仅需无标注语料库和少量查询-文档对，通过自改进循环迭代优化合成指令，在多数任务上超越人工校准，并使嵌入模型随数据规模扩展在15个领域持续提升性能。

    

    将嵌入模型适应到新任务需要能反映任务特定信息需求和相关性标准的训练数据。然而，人工设计数据成本高昂且难以规模化。本文提出了SYNTRA，一种新颖的检索数据合成方法，其自改进循环利用对试验输出的反馈来迭代优化可复用的合成指令。该过程仅需一个无标注语料库和少量相关的查询-文档对。在横跨15个领域的24个检索数据集上的实验证明了合成数据的有效性。这些数据还能在有无现有标注训练集的任务上提升广泛使用的强嵌入模型的性能。自改进在全部六个评估任务上都提升了检索效果，并在其中四个任务上超越了人工校准。此外，随着训练数据集规模扩展到初始大小的32倍，MS MARCO上的检索性能仍持续提升。

    arXiv:2610.14730v1 Announce Type: cross  Abstract: Adapting embedding models to new tasks requires training data that reflect task-specific information needs and relevance criteria. However, manual data design is costly and difficult to scale. Here we introduce SYNTRA, a novel retrieval data synthesis method whose self-improvement loop iteratively refines reusable synthesis instructions using feedback on trial outputs. This process requires only an unlabelled corpus and a few relevant query-document pairs. Experiments across 24 retrieval datasets spanning 15 domains demonstrate the effectiveness of the synthesized data. These data also improve widely used, strong embedding models on tasks with and without existing labelled training sets. Self-improvement enhances retrieval on all six evaluated tasks, outperforming human calibration on four. Moreover, retrieval performance continues to improve on MS MARCO as we scale the training dataset to 32 times its initial size. Overall, with minim
    
[^320]: PotentialEP：基于平衡传播的CIFAR规模生成模型无反向传播训练

    PotentialEP: Backprop-Free Training of CIFAR-Scale Generative Models via Equilibrium Propagation

    [https://arxiv.org/abs/2610.14409](https://arxiv.org/abs/2610.14409)

    该论文提出PotentialEP方法，证明加入单个线性注入项的预测编码网络在弛豫过程中其可见层可天然携带能量匹配所需的梯度，从而实现无需反向传播的CIFAR规模生成模型训练，适配模拟神经形态硬件。

    

    使用扩散模型或流匹配模型生成图像，通常需要对速度场进行数百个连续步骤的积分。这种计算并不适合GPU，却非常适合模拟神经形态基底——在那里，一次快速、低功耗的物理弛豫即可替代整个计算序列。要实现这一点需要两样东西：一个是计算过程本身即为弛豫的生成算法，另一种是不依赖反向传播的训练方法，因为反向传播所需的全局梯度传输在模拟硬件中难以实现。能量匹配提供了前者，它通过在时间无关势能上进行梯度下降弛豫来采样——但获取该梯度下降所遵循的梯度本身仍需一次反向传播，且每个积分步骤都需要执行一次。我们证明，在能量函数中加入单个线性注入项的预测编码网络，会弛豫到一个其可见层天然携带该梯度的状态，且当注入强度较小时误差趋于消失。一个能量匹

    arXiv:2610.14409v2 Announce Type: replace  Abstract: Generating an image with a diffusion or flow-matching model typically involves integrating a velocity field for hundreds of sequential steps. That computation suits a GPU poorly and an analog neuromorphic substrate well, where one fast, low-power physical relaxation could replace the whole sequence. Two things are needed: a generative algorithm whose computation is a relaxation, and a way to train it without backpropagation, whose global gradient transport is hard to realise in analog. Energy Matching supplies the first, sampling by gradient-descent relaxation on a time-independent potential - but obtaining the gradient that descent follows is itself a backward pass, once per integration step. We prove that a predictive coding network with a single linear injection term added to its energy relaxes to a state whose visible layer carries that gradient inherently, with an error that vanishes for small injection strengths. An Energy Matc
    
[^321]: EV-JEPA：适用于任意设备与跨设备推理的弹性 V-JEPA

    EV-JEPA: Elastic V-JEPA for Any Device and Cross-Device Inference

    [https://arxiv.org/abs/2610.14324](https://arxiv.org/abs/2610.14324)

    提出 EV-JEPA，通过转换感知三元组训练（T3）训练出宽度可配置且可原生跨异构设备分区的单一基础模型，一次训练即可部署于从智能眼镜到服务器的任意设备，并同时支持单设备弹性推理与跨设备协同推理。

    

    像 V-JEPA 这样的基础模型正越来越多地部署在异构的边云连续体上，涵盖智能眼镜、手机、机器人和服务器。然而，它们通常被训练为绑定单一设备的刚性、固定容量网络，既不支持可重新配置的计算预算，也不支持协同执行。现有方法每次只能解决其中一个维度：可裁剪和基于 NAS 的方法提供了宽度弹性，但假设单设备推理；而分割计算方法仅在固定分区和静态容量下实现跨设备协作。我们提出了弹性 V-JEPA（EV-JEPA），这是一个只需训练一次、即可部署在连续体中任何位置的单一模型。其宽度可重新配置，并且可以原生地在异构设备间进行分区。其核心是转换感知三元组训练。在每一步中，我们选取一个窄宽度和一个宽宽度，并训练三条路径：两个宽度各自的路径，外加一条混合路径。

    arXiv:2610.14324v1 Announce Type: cross  Abstract: Foundation models like V-JEPA are increasingly deployed across a heterogeneous edge-cloud continuum spanning smart glasses, phones, robots, and servers. Yet, they are typically trained as rigid, fixed-capacity networks bound to one device, supporting neither reconfigurable compute budgets nor collaborative execution. Existing approaches address only one axis at a time: slimmable and NAS-based methods provide width elasticity but assume single-device inference, whereas split-computing methods enable cross-device collaboration only for fixed partitions and static capacities. We introduce Elastic V-JEPA (EV-JEPA), a single model trained once and deployable everywhere along the continuum. Its width is reconfigurable, and it is natively partitionable across heterogeneous devices. At its core is Transition-Aware Triplet Training (T3). At each step we pick a narrow and a wide width, and train three paths: each width alone, plus a hybrid path 
    
[^322]: 先预测再修改：为自进化智能体学习改进动力学

    Predict Before You Modify: Learning Improvement Dynamics for Self-Evolving Agents

    [https://arxiv.org/abs/2610.13991](https://arxiv.org/abs/2610.13991)

    提出可编辑的改进动力学模型（IDM），在评估前预测修改的任务级结果、评估后利用反馈持续积累预测知识，实现“先预测再修改”的预测式自进化框架，并在四个任务上全面超越HyperAgents。

    

    自进化智能体通过反复的修改与评估循环来提升自身。然而，现有方法只有在代价高昂的评估之后才能获得关于修改的反馈，且汇总性的分数可能掩盖需要不同修复方式的各类失败之间的区别。我们通过一个可编辑的改进动力学模型（IDM）引入了预测式自进化。在评估之前，IDM会预测所提出修改在任务层面的结果，并指导一次可选的修订。在评估之后，过去的修改及其观测到的结果被用于改进IDM，使预测知识能够跨代际不断积累。由此，系统可以同时改进其解决任务、提出修改以及预测修改效果的能力。基于HyperAgents，我们在多语言编程、论文评审、机器人奖励设计和奥赛级数学评分四个任务上评估了该方法。我们的方法在全部四个任务上均优于HyperAgents。

    arXiv:2610.13991v1 Announce Type: cross  Abstract: Self-evolving agents improve through repeated cycles of modification and evaluation. However, existing methods receive feedback on modifications only after costly evaluations, and aggregate scores may obscure distinctions between failures that require different repairs. We introduce predictive self-evolution through an editable Improvement Dynamics Model (IDM). Before evaluation, the IDM predicts task-level outcomes of a proposed modification and guides an optional revision. After evaluation, past modifications and their observed outcomes are used to improve the IDM, allowing predictive knowledge to accumulate across generations. The system can thus improve how it solves tasks, proposes modifications, and predicts their effects. Building on HyperAgents, we evaluate this approach across multilingual coding, paper review, robotics reward design, and Olympiad-level mathematical grading. Our approach outperforms HyperAgents across all four
    
[^323]: 空间上下文还是记忆？确定部分观测场神经代理模型的输入规模

    Spatial Context or Memory? Sizing the Inputs of Neural Surrogates for Partially Observed Fields

    [https://arxiv.org/abs/2610.12723](https://arxiv.org/abs/2610.12723)

    本文以贝叶斯最优误差为基准，首次精确量化了部分观测场神经代理模型中空间传感器窗口与历史快照长度之间的权衡，证明对于具有局部动力学的稀疏含噪观测场，扩大空间窗口比增加记忆更有效，且仅读取空间窗口的模型即可接近最优预测性能。

    

    从稀疏、含噪传感器预测物理场的神经代理模型会读取一个邻近传感器窗口和一段过去快照的历史。实践者通常通过相互比较拟合好的模型来选择这两者，但这种比较无法将模型自身的不足与信息的缺失区分开来。我们以相同读数下的贝叶斯最优误差为基准来衡量窗口与历史之间的权衡。精确的高斯分析表明，对于具有局部动力学的场，含噪或稀疏的观测使得最优预测只能通过空间相关性来读取远处的传感器。在一个具有已知最优解的受迫线性场上，仅读取窗口的训练模型即可达到与最优解相差1%以内的水平。随后，我们在粘性Burgers流和混沌的Kuramoto-Sivashinsky（KS）方程上训练此类模型，且所有决定性比较均在任何测试评估之前予以固定。在除KS之外的所有场中，八张快照所消除的单快照误差比例随窗口长度增加而下降（摘要原文在此处截断）。

    arXiv:2610.12723v1 Announce Type: new  Abstract: Neural surrogates that forecast a physical field from sparse, noisy sensors read a window of nearby sensors and a history of past snapshots. Practitioners choose both by comparing fitted models with each other. Such comparisons cannot tell a model's shortfall from missing information. We measure the trade-off between window and history against the Bayes-optimal error of the same readings. An exact Gaussian analysis shows that noisy or sparse observation of a field with local dynamics makes the optimal forecast read distant sensors only through spatial correlation. On a forced linear field with a known optimum, trained models that read only the window come within 1% of it. We then train such models on viscous Burgers flow and the chaotic Kuramoto-Sivashinsky (KS) equation, with every deciding comparison fixed before any test evaluation. In every field except KS, eight snapshots remove a smaller fraction of the one-snapshot error as the le
    
[^324]: TTS-Q：面向文本到语音扩散变换器的感知引导训练后量化

    TTS-Q: Perception-Guided Post-Training Quantization of Text-to-Speech Diffusion Transformers

    [https://arxiv.org/abs/2610.12474](https://arxiv.org/abs/2610.12474)

    本文首次对十亿参数规模的TTS扩散变换器开展4比特权重与激活（W4A4）训练后量化研究，揭示了量化主要损害说话人音色而非可懂度这一现象及其来源，并提出感知引导的量化修复方法。

    

    零样本文本到语音（TTS）技术只需几秒钟的音频即可克隆一个声音。然而其高质量的生成依赖于数十亿参数的扩散变换器，其庞大的模型规模和多步采样过程使得最需要该技术的端侧设备、辅助技术和实时应用难以使用。训练后量化（PTQ）无需重新训练即可压缩此类模型，但现有方法是为语言和视觉领域设计的，而非语音领域。在4比特精度下，强量化器虽然能让语音可懂度接近全精度水平，但会持续损害说话人音色，导致错误率保持较低的同时，声音却偏离了目标说话人。我们探究了为什么量化会损害语音、损伤存在于何处，以及能否在不重新训练的情况下进行修复。为此，我们对1B和3.5B规模的TTS扩散变换器开展了首个4比特权重与激活（W4A4）训练后量化研究。研究发现，音色损失同时来自权重量化和激活量化，但只有激活部分具有时间维度的特性……（原文摘要在此处截断）

    arXiv:2610.12474v1 Announce Type: cross  Abstract: Zero-shot text-to-speech (TTS) can clone a voice from seconds of audio. Yet the quality comes from billion-parameter diffusion transformers whose size and multi-step sampling put them out of reach of the on-device, assistive, and real-time applications that need them most. Post-training quantization (PTQ) shrinks such models without retraining, but it has been built for language and vision, not speech. At 4 bits, a strong quantizer leaves intelligibility close to full precision but consistently erodes speaker timbre, so error rates stay low while the voice drifts from the target speaker. We ask why quantization hurts speech, where the damage lives, and whether it can be repaired without retraining. To find out, we run the first 4-bit weight-and-activation (W4A4) PTQ study of a TTS diffusion transformer at 1B and 3.5B. The timbre cost comes from both the weight and the activation quantization, but only the activation part has the timest
    
[^325]: 当场抓获：探针有效检测破坏行为并捕捉未言明的欺骗

    Caught in the Act: Probes Effectively Detect Sabotage and Catch Unverbalized Deception

    [https://arxiv.org/abs/2610.12445](https://arxiv.org/abs/2610.12445)

    该研究通过构建迄今最大的欺骗数据集并设计可跨层、跨标记聚合信息的新型探针架构，使白盒探针在 SHADE-Arena 上以 98.8% 的 AUC 超越前沿文本监控基线，甚至能检测出仅凭上下文无法察觉的“内省式欺骗”。

    

    近期发生的一些事件凸显了监控大语言模型（LLM）智能体的挑战，以及模型欺骗人类的危险。我们展示了基于探针的白盒欺骗检测可以扩展到前沿监控场景，方法包括收集迄今为止最大的欺骗数据集用于训练探针，并引入一种新颖的探针架构，该架构能够跨多个层和标记（token）聚合信息。我们的探针在 SHADE-Arena 上达到了 98.8% 的 AUC，超越了 Opus 5.5 的文本监控基线，并且随着底层模型规模的扩大，其检测效果也随之提升。为了将我们的探针推向极限，我们在几种仅凭上下文无法判断是否存在欺骗的案例上对其进行了测试。在这些我们称为“内省式欺骗”的案例中，真实情况只能通过仔细的引导性询问或对模型训练数据的透彻了解才能确定。在其中一项评估中，我们展示了探针能够区分包含……的对话记录（原文摘要在此处截断）。

    arXiv:2610.12445v1 Announce Type: cross  Abstract: Recent incidents have highlighted the challenge of monitoring LLM agents and the danger of models deceiving people. We show that white-box deception detection via probes can be scaled up to frontier monitoring settings by collecting the largest deception dataset to date for training probes and introducing a novel probe architecture which can aggregate information across many layers and tokens. Our probes achieve 98.8% AUC in SHADE-Arena, surpassing an Opus 5.5 text-monitoring baseline, and show improved efficacy as the underlying model is scaled up. To push our probes to their limit, we test them on several cases where deception cannot be determined from the context alone. In these cases, which we refer to as introspective deception, the ground truth can only be determined through careful elicitation or thorough knowledge of a model's training data. In one such evaluation, we show that probes can distinguish transcripts containing a mo
    
[^326]: AdaptLSTM：面向分布漂移下云工作负载预测的高效自适应在线学习

    AdaptLSTM: Efficient Adaptive Online Learning for Cloud Workload Forecasting under Distribution Drift

    [https://arxiv.org/abs/2610.12265](https://arxiv.org/abs/2610.12265)

    AdaptLSTM 通过验证校准阈值检测分布漂移并进行选择性针对性更新，以约 20% 的计算成本实现接近朴素在线学习的预测精度，大幅提升云工作负载预测的效率。

    

    准确的工作负载预测对于网络规模云服务的弹性资源供给至关重要，然而由热门内容、产品发布和用户行为引发的分布漂移会迅速降低离线训练模型的性能。朴素的在线学习虽然能恢复预测精度，但会带来难以承受的每步计算成本。我们提出了 AdaptLSTM，一种自适应在线学习框架，它通过验证集校准的阈值检测漂移，并施加选择性的、有针对性的更新。在阿里巴巴机器轨迹数据集上，AdaptLSTM 以 20% 的成本恢复了朴素在线学习 54% 的性能提升（2.7 倍效率，10 个随机种子下 p=0.002）。在波动性更强的容器轨迹数据集上，它以 20% 的成本实现了 96% 的性能提升（4.8 倍效率，相比静态模型 MAE 降低 75%）。与经典漂移检测器（ADWIN、DDM、Page-Hinkley）在回归尺度的误差流上无法触发不同，AdaptLSTM 在 301 步中触发了 42 次，并优于同等计算预算的基线方法。墙上时钟时间（原文在此处截断）。

    arXiv:2610.12265v1 Announce Type: new  Abstract: Accurate workload forecasting is critical for elastic resource provisioning in web-scale cloud services, where distribution shifts driven by viral content, product launches, and user behavior degrade offline-trained models rapidly. Naive online learning recovers accuracy but incurs prohibitive per-step compute cost. We propose AdaptLSTM, an adaptive online framework that detects drift via validation-calibrated thresholds and applies selective, targeted updates. On the Alibaba Machine Trace, AdaptLSTM recovers 54\% of Naive Online's improvement at 20\% cost ($2.7\times$ efficiency, $p=0.002$ over 10 seeds). On the more volatile Container Trace, it achieves 96\% at 20\% cost ($4.8\times$ efficiency, $+75\%$ MAE reduction over Static). Unlike classical drift detectors (ADWIN, DDM, Page-Hinkley) which fail to trigger on regression-scale error streams, AdaptLSTM fires 42 times over 301 steps and outperforms matched-budget baselines. Wall-cloc
    
[^327]: 耗散知识动力学模型的可辨识性：设计激励下的精确恢复与观测数据上的退化

    Identifiability of a dissipative knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational data

    [https://arxiv.org/abs/2610.09889](https://arxiv.org/abs/2610.09889)

    该论文将人类学习建模为参数具有机制含义的耗散常微分方程组，证明了在设计激励条件下模型参数可从数据中被精确恢复（双概念情形有闭式解），而在仅依赖观测数据时可辨识性会退化，并提出了数值精确等价且速度大幅提升的半隐式L-稳定批量求解器。

    

    人类学习是一个耗散动力学过程：熟练度通过练习积累，因遗忘而衰减，并在相互依赖的概念之间传播。我们将其建模为一个非线性耗散常微分方程组，其参数具有机制层面的含义（编码先修耦合关系的概念传递矩阵、各概念的遗忘率，以及饱和的练习-响应增益），并研究这些参数何时能够真正从数据中被恢复。我们在显式激励条件下证明了相应逆问题的结构可辨识性定理，针对双概念情形给出了构造性的闭式恢复方法，并同时给出了单调性、鲁棒性与L-稳定性结果。我们为耗散子系统推导了一种半隐式L-稳定数值格式，以及一个与逐轨迹公式数值等价的批量求解器（预测逐位一致，梯度误差达 $10^{-10}$），同时速度提升两个数量级。

    arXiv:2610.09889v1 Announce Type: new  Abstract: Human learning is a dissipative dynamical process: mastery accumulates through practice, decays through forgetting, and propagates across interdependent concepts. We model it as a nonlinear dissipative system of ordinary differential equations whose parameters are mechanistically meaningful (a concept-transfer matrix encoding prerequisite coupling, per-concept forgetting rates, and a saturating practice-response gain), and we study when those parameters can actually be recovered from data. We prove a structural identifiability theorem for the associated inverse problem under explicit excitation conditions, with constructive closed-form recovery for the two-concept case, together with monotonicity, robustness and L-stability results. We derive a semi-implicit L-stable scheme for the dissipative subsystem and a batched solver numerically equivalent to the per-trajectory formulation (bit-exact predictions, gradients to $10^{-10}$) yet two o
    
[^328]: LASER：用于支撑约束熵正则化离线强化学习的潜在空间伴随匹配

    LASER: Latent Space Adjoint Matching for Support-Constrained Entropy-Regularized Offline RL

    [https://arxiv.org/abs/2610.08989](https://arxiv.org/abs/2610.08989)

    LASER通过潜在空间伴随匹配实现熵正则化的潜在空间离线强化学习，既防止了策略坍缩成单一脆弱模式，又避免了时间反向传播，在40个不同数据质量的OGBench任务上取得了优异表现。

    

    虽然离线强化学习（RL）能够在无需昂贵的在线交互的情况下从静态数据集中进行策略优化，但其性能仍然受到执行分布外（OOD）动作风险的制约。近期的方法通过流匹配学习一个行为克隆策略，然后在其受约束的潜在空间内执行强化学习，从而缓解了这一问题。然而，朴素地优化潜在策略很容易导致策略坍缩成脆弱的单一模式，或利用学习到的评论家网络中尖锐的伪影。在这项工作中，我们发现熵正则化对于在潜在空间强化学习中应对这些挑战至关重要。我们提出了LASER，一种新颖的离线强化学习算法，它应用潜在空间伴随匹配来实现基于强表达能力的流策略的熵正则化潜在空间强化学习，同时避免了时间反向传播。通过在40个具有不同数据集质量的具有挑战性的OGBench任务上的全面实验，我们展示了（摘要在此处截断）

    arXiv:2610.08989v1 Announce Type: new  Abstract: While offline reinforcement learning (RL) enables policy optimization from static datasets without costly online interaction, it remains bottlenecked by the risk of executing out-of-distribution (OOD) actions. Recent approaches mitigate this by learning a behavior-cloning policy through flow matching and then performing RL within its constrained latent space. However, naively optimizing the latent policy can easily cause the policy to collapse into a brittle mode or exploit sharp artifacts of the learned critic. In this work, we find that entropy regularization is essential in latent-space RL for addressing these challenges. We introduce LASER, a novel offline RL algorithm that applies latent-space adjoint matching to achieve entropy-regularized latent-space RL with expressive flow policies while avoiding backpropagation through time. Through comprehensive experiments on 40 challenging OGBench tasks with varying dataset qualities, we sho
    
[^329]: CNet：一个具有Wirtinger自动微分与FFT-阿达马卷积的复值深度学习框架

    CNet: A Complex-Valued Deep Learning Framework with Wirtinger Autodifferentiation and FFT--Hadamard Convolution

    [https://arxiv.org/abs/2610.08592](https://arxiv.org/abs/2610.08592)

    CNet 是一个基于 Wirtinger 自动微分的 C++/CUDA 复值深度学习框架，它将 FFT-阿达马卷积恒等式转化为可学习的复值卷积网络，并采用玻恩规则进行物理原生式分类。

    

    CNet 是一个用于构建和训练深度复值神经网络（CVNN）的 C++/CUDA 框架，更广泛地说，它利用 Wirtinger（CR 演算）导数通过梯度下降来优化复值函数。该框架采取物理原生的立场：网络是作用于振幅向量的复数运算——通常是幺正运算（如 DFT）——的级联，而分类则采用玻恩规则测量 $p_k = |z_k|^2 / \|z\|^2$，而非对实数 logits 进行 softmax。每个层都提供 CPU 参考实现和经过有限差分校验的 CUDA 内核，并且计算图会在整个批次上克隆以供 GPU 执行。在基础层之上，我们添加了信号处理原语，将恒等式 conv(x,k) = IFFT(FFT(x) · FFT(k)) 转化为可学习的复值卷积网络，同时提供了真 Adam 优化器和低内存推理模式。我们报告了三项研究。第一项是一个完全复值的、FNet 风格的因果……（摘要在此处截断）

    arXiv:2610.08592v1 Announce Type: new  Abstract: CNet is a C++/CUDA framework for building and training deep complex-valued neural networks (CVNNs) and, more generally, for optimizing complex-valued functions by gradient descent with Wirtinger (CR-calculus) derivatives. It takes a physics-native stance: a network is a cascade of complex -- and often unitary (the DFT) -- operations acting on an amplitude vector, and classification is a Born-rule measurement $p_k = |z_k|^2 / \|z\|^2$ rather than a softmax over real logits. Every layer ships a CPU reference and a CUDA kernel checked against finite differences, and the computation graph is cloned across the batch for GPU execution. On top of the base layers we add signal-processing primitives that turn the identity conv(x,k) = IFFT(FFT(x) . FFT(k)) into a learnable complex convolutional network, together with a true-Adam optimizer and a reduced-memory inference mode.   We report three studies. First, a fully complex-valued, FNet-style caus
    
[^330]: 最小见证强化学习

    Minimal Witness Reinforcement Learning

    [https://arxiv.org/abs/2610.07226](https://arxiv.org/abs/2610.07226)

    本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。

    

    “足以产生某一结果的不可约条件是什么？”这是计算与科学领域中最常出现的核心问题之一。其答案——最小充分见证——正是我们所称的解释、机制与原因。这类问题通常需要找出多个最小见证，然而标准的强化学习方法可能只能揭示一个解或冗余的解。我们将该问题形式化为最小见证识别，并提出了最小见证强化学习（MWRL）。MWRL 对从策略中采样得到的、经成功验证的提议所认证的集合取并集，并根据“若缺少该提议，组并集将会损失的覆盖范围”为每个提议分配信用。这一直接源于问题定义的信用分配机制，仅凭一个黑盒验证器的单个比特信号，便统一了对最小性与备选解恢复的双重需求。基于该原理，我们推导出一个值迭代规划器，能够恢复……（摘要原文在此处截断）

    arXiv:2610.07226v1 Announce Type: cross  Abstract: ``What are the irreducible conditions that are sufficient to produce an outcome?'' is one of the most common questions that recur across computation and science. Its answers, the minimal sufficient witnesses, are what we mean by explanations, mechanisms and reasons. These problems usually ask for multiple minimal witnesses, yet standard RL methods may reveal only one solution or redundant ones. We formalize this problem as minimal-witness identification and introduce Minimal-Witness Reinforcement Learning (MWRL). MWRL takes the union of the sets certified by successful proposals sampled from the policy and credits each proposal for the coverage the group union would lose without that proposal. This credit assignment, derived directly from the problem definition, unifies the demands for minimality and recovery of alternatives from a single black-box verifier bit. Under this principle, we derive a value iteration planner that recovers th
    
[^331]: 来自40亿参数开放模型的关于随机试验的校准答案：一项预注册测试与无许可证问题的发布

    Calibrated Answers About Randomized Trials From a 4-Billion-Parameter Open Model: A Registered Test and a License-Clean Release

    [https://arxiv.org/abs/2610.07019](https://arxiv.org/abs/2610.07019)

    该论文发布了一个仅使用许可证允许复用的文章微调的 40 亿参数开放模型 Fiorillo v0.5，它能够以良好校准的概率回答随机试验中干预措施对结局影响的问题，并通过预注册的四项标准验证后正式发布。

    

    Fiorillo v0.5 是一个开放模型，它以概率形式回答类型化问题。其主要专家模块读取一篇随机试验的文章（截断至6,144个标记），并回答某项干预措施相对于对照是否显著增加、显著降低或未显著改变某一结局（基于 Evidence Inference 2.0，EI 数据集）。该模型基于 Qwen3-4B-Base，配备低秩适配器和决策头，仅在 2,657 篇训练文章中许可证自身允许复用的 1,431 篇上针对 EI 进行微调。在该版本的测试预测之前，研究者在开放科学框架（OSF）上预先注册了四项发布标准，其中第二项标准在标签公开的 EI 测试集上进行评判。在该测试集（333 篇文章中的 1,218 个提示）上，预期校准误差为 0.0168，低于 0.05 的限值；对数损失比先验基线低 0.8603（95% 区间为 0.8104 至 0.9078），并且在读取相同输入的情况下，比 Gemma 4 31B-it 低约 0.1（摘要在此处截断）。

    arXiv:2610.07019v1 Announce Type: new  Abstract: Fiorillo v0.5 is an open model that answers typed questions with a probability for each answer. Its main specialist reads a randomized trial's article, cut to 6,144 tokens, and answers whether an intervention significantly increased, significantly decreased or did not significantly change an outcome against a comparator (Evidence Inference 2.0, EI). It is Qwen3-4B-Base with low-rank adapters and a decision head, fine-tuned for EI only on the 1,431 of 2,657 training articles whose own license allows reuse. Four criteria registered on the Open Science Framework before this version's test predictions decided its release, the second bar judged on EI's test split, whose labels are public. On that split (1,218 prompts in 333 articles), the expected calibration error was 0.0168 against a limit of 0.05; log loss was below the prior's by 0.8603 (95 percent interval 0.8104 to 0.9078) and below that of Gemma 4 31B-it, reading the same input, by 0.1
    
[^332]: E$^2$-OPSD：驯服在线策略自蒸馏中的熵过冲

    E$^2$-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.05048](https://arxiv.org/abs/2610.05048)

    论文发现在线策略自蒸馏存在学生熵超过教师并持续高企的“熵过冲”失效模式，其根源是教师监督过度依赖答案特定线索以及前向KL散度不断扩散学生分布，并据此提出E$^2$-OPSD同时修复这两个成因。

    

    在线策略自蒸馏（OPSD）无需第二个模型即可提供密集的token级监督：同一个网络在给定参考解答时充当教师，而在仅给定问题时充当学生。我们识别出该方法的一个特定失效模式：在训练过程中，学生的token熵会超过教师的熵并持续保持高位，我们将这一现象称为“熵过冲”（entropy overshoot）。我们将其根源追溯到蒸馏的双方。以参考答案为条件的教师在其面向答案的推理路径上表现得很自信，但这种自信难以迁移到学生生成的前缀上，使其监督过度依赖于答案特定的线索，而非可复用的推理模式；与此同时，OPSD所使用的前向KL散度会持续扩散学生的预测分布，而无法将其拉回。我们提出E$^2$-OPSD来同时应对这两个成因。示例引导的教学（exemplar-guided teaching）用检索到的已解决的相邻问题替换当前答案，提……（原文摘要在此处截断）

    arXiv:2610.05048v2 Announce Type: replace-cross  Abstract: On-policy self-distillation (OPSD) provides dense token-level supervision without a second model: one network acts as teacher with the reference solution and as student with only the problem. We identify a specific failure mode of this recipe. During training, student token entropy rises past the teacher's and remains elevated, a pattern we call entropy overshoot. We trace it to both sides of distillation. The reference-conditioned teacher is confident along its answer-directed reasoning path, but this confidence transfers poorly to student-generated prefixes, making its supervision overly tied to answer-specific cues rather than reusable reasoning patterns; meanwhile, the forward KL used by OPSD continually diffuses the student's predictive distribution without pulling it back. We introduce E$^2$-OPSD to address both causes. Exemplar-guided teaching replaces the current answer with a retrieved solved neighboring problem, provi
    
[^333]: D-DOIT：基于Doob h-变换的离散扩散免训练适配方法

    D-DOIT: Training-free Adaptation of Discrete Diffusion via Doob's h-Transform

    [https://arxiv.org/abs/2610.04938](https://arxiv.org/abs/2610.04938)

    D-DOIT提出了一种免训练的离散扩散模型适配方法，通过推导离散Doob h-变换、仅利用奖励值（无需奖励梯度和昂贵的未来展开）重新加权反向转移概率，从而在推理时高效地从奖励倾斜的目标分布中采样。

    

    我们提出了D-DOIT（离散Doob导向的推理时变换），这是一种针对具有通用奖励的离散扩散模型的免训练且高效的适配方法。D-DOIT将适配问题表述为从一个奖励倾斜的目标分布中采样，并通过离散扩散反向核的Doob h-变换来实现这一传输过程，且仅使用奖励值而无需奖励梯度。与连续扩散不同，掩码离散扩散采样的是类别型的词元揭示转移，而非连续的状态更新。D-DOIT推导出了相应的离散Doob h-变换，该变换通过重新加权反向转移概率（而非添加漂移修正项）来引导采样过程。为了使这一变换切实可行，D-DOIT避免了昂贵的前向展开操作。在每个引导步骤中，D-DOIT采样候选的下一状态，利用模型预测头将每个候选补全为干净的序列，并对其进行评估……

    arXiv:2610.04938v2 Announce Type: replace-cross  Abstract: We propose D-DOIT (Discrete Doob-Oriented Inference-time Transformation), a training-free and efficient adaptation method for discrete diffusion models with generic rewards. D-DOIT formulates adaptation as sampling from a reward-tilted target distribution and realizes this transport through Doob's h-transform of the discrete diffusion reverse kernel, using only reward values rather than reward gradients. Unlike continuous diffusion, masked discrete diffusion samples categorical token-reveal transitions rather than continuous state updates. D-DOIT derives the corresponding discrete Doob's h-transform, which guides sampling by reweighting reverse transition probabilities instead of adding a drift correction. To make this transformation practical, D-DOIT avoids expensive future rollouts. At each guided step, D-DOIT samples candidate next states, uses the model prediction head to complete each candidate into a clean sequence, evalu
    
[^334]: ULTRADISCOVERY：在一个互联的、认识论开放的宇宙中进行溯因探索

    ULTRADISCOVERY: Abductive Exploration in an Interconnected, Epistemically Open Universe

    [https://arxiv.org/abs/2610.03092](https://arxiv.org/abs/2610.03092)

    该论文提出ULTRADISCOVERY交互式基准，通过2×2设计独立控制表征开放性与证据分布性，评估智能体在认识论开放且结构互联的世界中进行溯因科学探索的能力，发现现有十一个模型均无法通过引入新实体或重写变量来完成理论替换。

    

    科学发现往往始于零散的线索呼唤一种描述世界的新方式。当世界在认识论上是开放的，这种溯因探索可能需要构建用以陈述解释的表征；当世界在结构上是互联的，则需要综合散布于不同情境中的证据。现有基准很少将这两项需求区分开来或对它们进行独立控制。我们提出ULTRADISCOVERY，一个包含五个领域的交互式世界，智能体在其中需要修正一个最初成功的理论，并预测一次未见过的跨领域干预的结果。该基准采用2×2设计：表征保持开放或予以揭示，证据保持分散或予以对齐，而潜在动力学保持不变。在表征开放的条件下，横跨十一个模型的智能体常常收回他们被教授的公理，但没有任何一个模型能够引入替换所需的未观测实体或重写变量……

    arXiv:2610.03092v1 Announce Type: cross  Abstract: Scientific discovery often begins when scattered clues call for a new way of describing the world. Such abductive exploration can require constructing the representation in which an explanation is stated, when the world is epistemically open, and composing evidence scattered across contexts, when it is structurally interconnected. Existing benchmarks rarely separate these two demands or control them independently. We introduce ULTRADISCOVERY, an interactive world of five domains in which an agent revises an initially successful theory and predicts the outcome of an unseen cross-domain intervention. A $2 \times 2$ design leaves the representation open or discloses it, and leaves the evidence distributed or aligns it, with the latent dynamics fixed. With the representation open, agents across eleven models often retract the axiom they were taught, and none introduces the unobserved entity or rewrites the variables that a replacement requ
    
[^335]: StabilityArc：将蛋白质序列嵌入解码为可泛化的稳定性景观

    StabilityArc: Decoding Protein Sequence Embeddings into Generalizable Stability Landscapes

    [https://arxiv.org/abs/2610.00742](https://arxiv.org/abs/2610.00742)

    StabilityArc 通过在多种蛋白质数据上训练的共享解码器，将冻结的蛋白质序列嵌入解码为可泛化的稳定性景观，实现了跨蛋白质的突变效应零样本迁移预测，其性能超越最强零样本基线，并可作为先验显著提升其他模型的表现。

    

    每种蛋白质都有其独特的稳定性景观，但突变的物理后果受制于反复出现的生化约束。我们测试了在多种蛋白质测量数据上训练的共享解码器，能否在未见过的目标蛋白质中解读这些约束，从而实现跨蛋白质迁移，用于初始实验轮次的预筛选。我们提出了 StabilityArc，它将冻结的 ESMC-600M 残基表示通过共享的 RoPE transformer 映射为 Lx20 的替换效应矩阵；一个对称的、接触感知的残差模块有助于预测同时替换中的上位效应。在涵盖 134,794 个 ProteinGym 变异的 66 次严格留一蛋白质评估中，StabilityArc 达到了 0.7134 的 Spearman 相关系数，超过最强零样本基线 ProSST-2048（0.6526）0.0608。我们进一步将该分数作为 Kermut 的先验来探索此方法的实用性，实现了 0.8280 的 Spearman 相关系数。

    arXiv:2610.00742v1 Announce Type: cross  Abstract: Every protein has a unique stability landscape, but the physical consequences of mutation are governed by recurring biochemical constraints. We test whether a shared decoder, trained on measurements from diverse proteins, can interpret these constraints in an unseen target, enabling cross-protein transfer for initial experimental round prescreening. We present StabilityArc , which maps frozen ESMC-600M residue representations through a shared RoPE transformer to an Lx20 matrix of substitution effects; a symmetric, contact-aware residual aids in predicting epistasis in simultaneous substitutions. In 66 strict leave-one-protein-out evaluations covering 134,794 ProteinGym variants, StabilityArc achieves 0.7134 Spearman correlation, exceeding the strongest zero-shot baseline, ProSST-2048 (0.6526), by 0.0608. We further explore the utility of this method by providing the score as a prior for Kermut, achieving Spearman correlation of 0.8280 
    
[^336]: CompMat-Bench：面向计算材料科学的AI智能体基准测试

    CompMat-Bench: Benchmarking AI Agents for Computational Materials Science

    [https://arxiv.org/abs/2610.00636](https://arxiv.org/abs/2610.00636)

    提出了CompMat-Bench基准，通过预先复现昂贵模拟的输入与结果作为固定规则的评分依据，实现对计算材料科学AI智能体的低成本高效评估，涵盖94个源自已发表研究的任务以及四种评估条件。

    

    在科学研究任务上评估AI智能体受到底层实验或计算所需时间和资源的限制。在计算材料研究中，跨智能体和多次试验重复相同的昂贵模拟会使评估变得不切实际。我们提出了CompMat-Bench，这是一个包含94个任务的基准，这些任务源自近期发表的计算材料研究，每个任务要求智能体完成实现该研究科学目标的一个步骤。我们预先复现了研究步骤，并评估智能体在为昂贵模拟准备输入和分析输出方面的能力，从而在评估过程中避免执行昂贵的模拟。复现的输入和结果作为基准真值，通过固定规则对智能体进行评分，无需LLM裁判。该基准支持四种评估条件：单一任务和由相关任务组成的工作流，每种条件均提供完整或精简的方法学指导。

    arXiv:2610.00636v1 Announce Type: new  Abstract: Evaluating AI agents on scientific research tasks is constrained by the time and resources required for the underlying experiments or calculations. In computational materials research, repeating the same expensive simulations across agents and trials can make evaluation impractical. We introduce CompMat-Bench, a benchmark of 94 tasks derived from recently published computational materials studies, each asking agents to complete a step toward achieving the study's scientific goal. We reproduce the research steps in advance and assess agents on preparing inputs and analyzing outputs for expensive simulations, so expensive simulations can be avoided during evaluation. The reproduced inputs and results serve as ground truth for grading agents with fixed rules, without an LLM judge. The benchmark supports four evaluation conditions: single tasks and workflows composed of related tasks, each with full or reduced methodological guidance. With f
    
[^337]: 可解释但脆弱？概念瓶颈模型在几何-语义扰动下的鲁棒性

    Interpretable but Fragile? Robustness of Concept Bottlenecks under Geometric-Semantic Perturbations

    [https://arxiv.org/abs/2609.38625](https://arxiv.org/abs/2609.38625)

    该论文提出了一个基于生成器的统一评估框架，首次将潜空间中的连续几何扰动与概念空间中的离散语义干预区分开来，从经验性与可认证两个层面系统比较概念瓶颈模型与标准分类器的鲁棒性，从而调和了此前相互矛盾的研究结论。

    

    概念瓶颈模型（CBMs）旨在提供可解释的中间表示，然而此类瓶颈如何影响模型鲁棒性仍不明确，现有研究报告的结果不一致，有时甚至相互矛盾。我们认为，这些分歧源于将不同的鲁棒性概念与扰动模式混为一谈，而非对概念瓶颈模型本身存在根本性分歧。为厘清这些因素，我们提出了一个基于生成器的评估框架，使标准分类器与概念瓶颈模型能够在两类截然不同的扰动下进行受控比较：潜空间中的连续几何扰动与概念空间中的离散语义干预。在该框架内，我们一方面通过预测与概念级敏感性指标对鲁棒性进行经验性评估，另一方面利用潜空间与概念空间中的随机平滑进行可认证的鲁棒性评估。在一系列实验中，我们调和了此前……（原文摘要在此处截断）

    arXiv:2609.38625v1 Announce Type: cross  Abstract: Concept Bottleneck Models (CBMs) are designed to provide interpretable intermediate representations, yet how such bottlenecks affect robustness remains unclear, with existing studies reporting mixed and sometimes contradictory findings. We argue that these discrepancies arise from conflating different robustness notions and perturbation regimes, rather than from fundamental disagreements about CBMs themselves. To disentangle these factors, we introduce a generator-based evaluation framework that enables controlled comparisons between standard classifiers and CBMs under two distinct perturbation types: continuous geometric perturbations in latent space and discrete semantic interventions in concept space. Within this framework, we evaluate robustness both empirically, via prediction and concept-level sensitivity metrics, and certifiably, using randomized smoothing in latent and concept spaces. Across experiments, we reconcile previously
    
[^338]: 用于智能体强化学习中高效压缩的KV流

    KV-streams for Efficient Compaction in Agentic Reinforcement Learning

    [https://arxiv.org/abs/2609.35750](https://arxiv.org/abs/2609.35750)

    KV-streams提出一种即插即用的KV缓存流式传输策略，通过在压缩时保留并向前流式传输KV缓存而非清空后反复预填充，使智能体强化学习中的上下文压缩实现2.6到5倍训练加速且不损害性能。

    

    摘要：扩展智能体LLM的视野受到需要在GPU内存中容纳越来越长的上下文轨迹的瓶颈制约。上下文压缩一直是缓解这一问题的最流行机制，它使给定轨迹的GPU内存占用保持恒定。不幸的是，大多数压缩策略依赖于对LLM上下文进行多次预填充，从而阻碍了训练吞吐量。为了缓解这一瓶颈并实现高效的可训练压缩，我们提出了KV-streams，这是一种即插即用的策略，可与任何压缩策略兼容，在显著提高吞吐量的同时没有表现出损害性能的证据。KV-streams通过将KV缓存向前流式传输而不是在每次压缩后将其清空，实现了可扩展的压缩。我们证明KV-streams支持三种不同的压缩策略，在训练中实现了2.6到5倍的墙钟时间加速。除了效率之外，我们发现流式传输的KV缓存可以充当循环状态（摘要在此处截断）。

    arXiv:2609.35750v2 Announce Type: replace-cross  Abstract: Scaling the horizon of agentic LLMs is bottlenecked by the need to fit ever longer context traces in GPU memory. Context compaction has been the most popular mechanism to alleviate this issue, keeping GPU memory constant for a given trace. Unfortunately, most compaction strategies rely on prefilling the LLM context many times over, hindering training throughput. To alleviate this bottleneck and enable efficient trainable compaction, we propose KV-streams, a plug-and-play strategy compatible with any compaction strategy that substantially increases throughput while showing no evidence of hindering performance. KV-streams enable scalable compaction by streaming the KV cache forward rather than flushing it after each compaction. We show that KV-streams enable three different compaction strategies, achieving a 2.6 to 5x wall-clock speedup in training. Beyond efficiency, we find that the streamed KV cache can act as a recurrent stat
    
[^339]: THEIA：一个用于版图视觉-语言分析的多模态数据集与基准测试

    THEIA: A Multimodal Dataset and Benchmark for Vision-Language Analysis of Layout

    [https://arxiv.org/abs/2609.35035](https://arxiv.org/abs/2609.35035)

    本文提出THEIA多模态数据集与基准测试，利用微调的视觉-语言模型分析模拟电路的GDSII版图文件，使设计者能够通过对话方式直观地查询和交互物理版图。

    

    人工智能与计算机辅助设计框架的融合引发了模拟集成电路（IC）设计领域的变革，使该领域从使用手动和基于算法的解决方案转向采用自动化和智能化的范式。在这一背景下，GDSII文件作为行业标准数据库，包含了模拟电路最终且最准确的信息来源，封装了决定流片性能的复杂物理几何结构与寄生效应等实际情况。本文提出了THEIA，这是一个包含数千张版图图像并配有问答对话的新型数据集，同时提出了一个采用微调视觉-语言模型（VLM）来分析模拟电路GDSII文件的基准测试，使设计人员能够将物理版图作为直观且有意义的实体进行交互和查询。基于数千个模拟设计的实验结果表明……

    arXiv:2609.35035v2 Announce Type: replace  Abstract: The integration of artificial intelligence into computer-aided design frameworks has sparked a shift in the design of analog integrated circuits (ICs), transitioning the field from using manual and algorithmic-based solutions to adopting automated and intelligent paradigms. In this scenario, the GDSII file represents the industry-standard database containing the ultimate and most accurate source of information of the analog circuit, encapsulating the complex physical geometries and parasitic realities that define tape out performance. This paper proposes THEIA, a novel dataset containing thousands of layout images paired with question-answer conversations, along with a benchmark that employs a fine-tuned vision-language model (VLM) to analyze GDSII files of analog circuits, enabling designers to interact with and query physical layouts as intuitive, meaningful entities. Experimental results using thousands of analog designs across fi
    
[^340]: FAME：一种基于FPGA的近似乘法器评估平台，结合模式引导的DNN重训练

    FAME: An FPGA-Based Platform for Approximate Multipliers Evaluation with Pattern-Guided DNN Retraining

    [https://arxiv.org/abs/2609.17730](https://arxiv.org/abs/2609.17730)

    本文提出FAME，一个基于FPGA的近似乘法器评估平台，通过在硬件中直接实现近似乘法器并结合模式引导的DNN重训练，大幅加速了DNN推理中近似乘法器的精度评估与误差缓解流程。

    

    近似乘法器可以降低深度神经网络（DNN）推理中的硬件面积和能耗；然而，它们会引入计算误差。由于评估时间过于漫长，在多样化的DNN模型和大规模数据集上评估大量近似乘法器设计的精度仍然充满挑战。这种开销主要源于在CPU和GPU平台上使用查找表（LUT）对近似乘法器行为进行缓慢的仿真。此外，由此产生的精度下降必须被仔细量化，并在必要时加以缓解（例如通过重训练），这进一步增加了整体评估成本。为了解决这些挑战，我们提出了FAME，一个基于FPGA的近似乘法器评估平台。该平台利用现场可编程门阵列（FPGA）的可重构逻辑直接在硬件中实现近似乘法器，从而无需基于LUT的仿真。

    arXiv:2609.17730v1 Announce Type: cross  Abstract: Approximate multipliers can reduce hardware area and energy consumption in Deep Neural Network (DNN) inference; however, they introduce computational errors. Assessing the accuracy of numerous approximate multiplier designs across diverse DNN models and large-scale datasets remains challenging due to prohibitive evaluation times. This overhead primarily stems from the slow emulation of approximate multiplier behavior using look-up tables (LUTs) on CPU and GPU platforms. Moreover, the resulting accuracy degradation must be carefully quantified and, if necessary, mitigated (e.g., through retraining), further increasing the overall evaluation cost. To address these challenges, we propose FAME, an FPGA-based platform for evaluating approximate multipliers. The platform exploits the reconfigurable logic of Field-Programmable Gate Arrays (FPGAs) to implement approximate multipliers directly in hardware, eliminating the need for LUT-based emu
    
[^341]: PLSP（事前边界空间剖析）：OOD预测优于检测——一种面向机器学习模型可靠性的预测性方法

    PLSP (Pre-hoc Liminal Space Profiling): OOD Prediction over Detection -- An Anticipatory Approach for Machine Learning Model Reliability

    [https://arxiv.org/abs/2609.12225](https://arxiv.org/abs/2609.12225)

    本文提出事前预测框架PLSP，将OOD处理范式从检测转向预测，并引入可信度评分（CREDS）、可信度曲线和可信度热力图，实现对机器学习模型应对分布外数据可靠性的事前评估。

    

    分布外数据对机器学习模型构成重大威胁，常常导致模型在部署过程中失效。所有现有的OOD检测方法都是事后方法，依赖于推理过程中的准确率和AUC-ROC等评估指标，通过测量偏差来间接评估模型对OOD数据的响应。与现有方法不同，本工作提出了一种名为PLSP的事前预测性框架，将范式从OOD检测转向OOD预测。我们做出了几项关键贡献：提出了一种与数据集无关的度量标准——可信度评分（CREDS），用于OOD预测；引入了可信度曲线来研究模型能够达到的最大可信度；引入了可信度热力图（及曲面下体积）来表征模型在不同数据集上的事前行为。这项工作为信号处理领域提供了全新的视角。

    arXiv:2609.12225v1 Announce Type: new  Abstract: Out-of-Distribution (OOD) data poses a significant threat to machine learning models, often leading to model failure during deployment. All existing OOD detection methods are post-hoc, relying on evaluation metrics such as accuracy and AUC-ROC during inference to indirectly assess the model's response to OOD data by measuring deviations. In contrast to existing approaches, the proposed work shifts the paradigm from OOD detection to OOD prediction by proposing a pre-hoc anticipatory framework called PLSP for OOD prediction. We make several key contributions: (a) a dataset-independent metric called the CREDibility Score (CREDS) is proposed for OOD prediction; (b) credibility curves are introduced to study the maximum credibility a model can attain; and (c) credibility heat maps (and volume under surface) are introduced to characterize pre-hoc model behavior across different datasets. This work provides a novel perspective on signal process
    
[^342]: LightMedSeg-ISLES：参数量比 nnU-Net 少 81 倍的脑卒中病灶分割

    LightMedSeg-ISLES: Stroke Lesion Segmentation with 81x Fewer Parameters than nnU-Net

    [https://arxiv.org/abs/2609.09634](https://arxiv.org/abs/2609.09634)

    提出了仅126万参数的轻量级脑卒中病灶分割模型LightMedSeg-ISLES，以少81.4倍的参数保留nnU-Net 97.5%的Dice分数并提升病灶级F1，同时计算量减少4.7倍，是大型模型的强力单检查点替代方案。

    

    大型网络和集成模型常常主导医学图像分割挑战赛，但其存储和推理需求使部署变得复杂。我们提出了 LightMedSeg-ISLES，一个面向 ISLES'26 中 T1 加权脑卒中病灶分割的 126 万参数流水线。在一个 146 例的保留测试集上，翻转测试时增强（TTA）取得 0.618 的平均 Dice 和 0.599 的病灶级 F1；而一个 1.0235 亿参数的 nnU-Net ResEnc-L 经尺寸过滤后取得 0.634 的 Dice 和 0.544 的病灶级 F1。因此，LightMedSeg 以少 81.4 倍的参数保留了 nnU-Net 97.5% 的 Dice，同时将病灶级 F1 提高了 0.055。其四遍 TTA 工作点在每个标准化图像块上所需的 FLOPs 比 nnU-Net 少 4.7 倍，并且略微超过了经过过滤的 UNETR++ 和 nnFormer。更长的训练和更强的数据增强在不增加模型容量的情况下额外提升 0.0358 Dice，使其成为远大于它的模型的一个强有力的单检查点替代方案。

    arXiv:2609.09634v2 Announce Type: replace-cross  Abstract: Large networks and ensembles often lead medical image segmentation challenges, but their storage and inference demands complicate deployment. We present LightMedSeg-ISLES, a 1.26-million-parameter pipeline for T1-weighted stroke lesion segmentation in ISLES'26. On a 146-case held-out cohort, flip test-time augmentation produces 0.618 mean Dice and 0.599 lesion-wise F1. A 102.35-million-parameter nnU-Net ResEnc-L produces 0.634 Dice and 0.544 lesion-wise F1 after size filtering. LightMedSeg therefore retains 97.5\% of nnU-Net's Dice with 81.4$\times$ fewer parameters while improving lesion-wise F1 by 0.055. Its four-pass TTA operating point requires 4.7$\times$ fewer FLOPs per standardized patch than nnU-Net. It also slightly exceeds filtered UNETR++ and nnFormer. Longer training and stronger augmentation add 0.0358 Dice without increasing capacity, establishing a strong single-checkpoint alternative to much larger models.
    
[^343]: 用于大语言模型行为控制的无训练任务向量

    Training-Free Task Vectors for LLM Behavioral Control

    [https://arxiv.org/abs/2609.09054](https://arxiv.org/abs/2609.09054)

    提出了一种无需微调的TFTVs方法，仅利用前向传播统计量将激活引导向量映射为秩一权重空间编辑，从而以低成本实现对大语言模型行为的放大、抑制与组合控制。

    

    任务向量通过识别权重空间中具有语义意义的方向，实现训练后的模型编辑，通常计算为微调模型与其预训练初始化模型之间的差异。然而，这种对微调的依赖使得发现此类方向的成本高昂，限制了训练后模型编辑的实用性。为了解决这一局限，我们提出了无训练任务向量（TFTVs），这是一种无需微调即可计算类任务向量方向的新方法。我们的方法仅利用前向传播统计量，将激活引导向量映射为秩一权重空间编辑，同时满足直接支持通过加法实现学习、通过减法实现遗忘以及多个编辑进行组合的算术性质。在实验中，我们在大语言模型行为控制任务上对TFTVs进行了评估，结果表明它们能够在保持模型原有能力的同时，持续地放大、抑制和组合目标行为。

    arXiv:2609.09054v1 Announce Type: cross  Abstract: Task vectors enable post-training model editing by identifying semantically meaningful directions in weight space, typically computed as the difference between a fine-tuned model and its pretrained initialization. However, this reliance on fine-tuning makes discovering such directions costly and limits the practicality of post-training model editing. To address this limitation, we introduce Training-Free Task Vectors (TFTVs), a novel method to compute task-vector-like directions without requiring fine-tuning. Our method maps activation steering vectors to rank-one weight-space edits using only forward-pass statistics, while satisfying arithmetic properties that directly support learning via addition, forgetting via subtraction, and the composition of multiple edits. Empirically, we evaluate TFTVs on large language model behavioral control tasks and show that they consistently amplify, suppress, and compose target behaviors while preser
    
[^344]: 语言模型可以控制自己的注意力

    Language Models Can Control Their Own Attention

    [https://arxiv.org/abs/2609.02737](https://arxiv.org/abs/2609.02737)

    该论文提出“声明式注意力”协议，让语言模型在思维链中自主声明需要关注的上下文区域，推理引擎据此像解析工具调用一样跳过大部分KV缓存读取，从而以内在方式避免了外部评分方法每步O(N)的开销。

    

    语言模型将大部分注意力集中在上下文的一小部分上，然而它们却要读取整个KV缓存来找出少数重要的token。如果用户在100万token的对话中询问之前的某个细节，全局注意力层必须扫描完整上下文才能生成回复的每一个token。一种著名的方法通过轻量级代理分数预先选择相关token来缓解这一成本，但这种外部评分机制在每一步仍然会产生O(N)的开销。我们采取一种内在的方法，其动机源于一个简单的问题：模型难道不是已经知道上下文的哪些部分是相关的吗？为此，我们引入了声明式注意力，这是一种协议，可引导模型在其思维链中声明它需要关注的位置，将生成过程划分为三种模式：（完整上下文）、（特定区域）和（仅最近输出）。推理引擎像解析工具调用一样解析这些声明，并跳过大部分KV……

    arXiv:2609.02737v1 Announce Type: cross  Abstract: Language models spend most of their attention on a small fraction of context, yet they read the entire KV cache to find the few tokens that matter. If the user asks about a previous detail in a 1M-token conversation, global attention layers must scan the full context to generate each token of the reply. A prominent approach mitigates this cost by pre-selecting relevant tokens via lightweight proxy scores, but this extrinsic scoring still incurs O(N) per step. We take an intrinsic approach motivated by the simple question: wouldn't the model already know which parts of the context are relevant? To this end, we introduce Declarative Attention (DA), a protocol that elicits the model to declare where it needs to attend within its chain-of-thought, partitioning generation into three modes:  (full context),  (a specific region), and  (recent output only). The inference engine parses these declarations like tool calls and skips most of the KV
    
[^345]: 通过幂律熵搜索高效估计最优超参数缩放定律

    Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search

    [https://arxiv.org/abs/2609.01431](https://arxiv.org/abs/2609.01431)

    本文提出幂律熵搜索（PLES），一种基于多保真度贝叶斯优化的计算成本感知采集函数，通过自适应选择能最大程度降低缩放定律估计整体不确定性的实验配置（而非优化单一目标函数），高效估计大语言模型最优超参数随规模变化的缩放定律，从而大幅节省计算资源。

    

    最优超参数缩放定律描述了用于大语言模型（LLM）训练的最佳超参数如何随模型和数据规模变化，使从业者无需昂贵的大规模调优即可预测生产规模下的最优配置。然而，传统上估计这些缩放定律需要对数千次训练运行进行穷举网格搜索，消耗巨大的计算资源。我们提出了幂律熵搜索（Power-Law Entropy Search, PLES），这是一种建立在多保真度贝叶斯优化之上的计算成本感知采集函数，能够通过自适应实验高效估计最优超参数缩放定律。PLES的一个关键创新在于，它搜索的是能够降低缩放定律估计整体不确定性的候选配置，而不是优化单一目标函数。在每次迭代中，PLES选择能够最大程度降低缩放定律估计不确定性的候选配置。

    arXiv:2609.01431v1 Announce Type: cross  Abstract: Optimal hyperparameter scaling laws describe how the best hyperparameters for large language model (LLM) training change with model and data scale, enabling practitioners to predict optimal configurations at production scales without expensive large-scale tuning. However, estimating these scaling laws conventionally requires exhaustive grid searches over thousands of training runs, consuming enormous computational resources. We introduce Power-Law Entropy Search (PLES), a computational cost-aware acquisition function built on multi-fidelity Bayesian optimization that efficiently estimates optimal hyperparameter scaling laws through adaptive experimentation. A key innovation in PLES is that it searches for candidates that reduce the overall uncertainty of a scaling law estimate, instead of optimizing a single objective function. At each iteration, PLES selects the candidate configuration that maximally reduces the uncertainty of the sca
    
[^346]: 角色扮演越狱中的安全接力：对危害识别与拒绝的组件级因果分析

    The Safety Relay in Roleplay Jailbreaks: A Component-Resolved Causal Analysis of Harm Recognition and Refusal

    [https://arxiv.org/abs/2608.30585](https://arxiv.org/abs/2608.30585)

    该研究运用机制可解释性方法，通过追踪隐状态、反事实干预和几何分解，揭示了角色扮演包装中的人设、场景和任务组件如何在组件层面逆转大语言模型的有害识别与拒绝机制。

    

    大型语言模型被训练为在遵循指令的同时拒绝有害请求。越狱攻击正是利用这种平衡来诱导模型输出其通常会拒绝的内容。角色扮演越狱尤其令人担忧：有害请求可以在由人设、场景和任务构成的角色扮演包装中保持完全可见，但模型却可能服从。我们使用机制可解释性方法来确定这种上下文如何逆转拒绝行为，以及哪些元素促成了这种逆转。我们在两个基准、三个模型家族和四种自建包装上进行实验，比较匹配的有害与良性请求在有包装和无包装情况下的表现。我们追踪从请求到最终提示状态的隐状态对比，通过受控反事实隔离包装操作，在留出的评估请求中对其激活方向进行干预，并从几何角度分解有效方向。我们的分析得出三项发现。(1) 成功的攻击……（摘要在此处截断）

    arXiv:2608.30585v1 Announce Type: new  Abstract: Large language models are trained to follow instructions while refusing harmful requests. Jailbreaks exploit this balance to elicit content a model would ordinarily reject. Roleplay jailbreaks are especially concerning: the harmful request can remain visible inside a roleplay wrapper made of a persona, scenario, and task, yet the model may comply. We use mechanistic interpretability to determine how this context reverses refusal and which elements contribute to the reversal. Across two benchmarks, three model families, and four authored wrappers, we compare matched harmful and benign requests with and without this wrapper. We trace hidden-state contrasts from the request to the final prompt state, isolate wrapper operations through controlled counterfactuals, intervene on their activation directions in held-out evaluation requests, and decompose effective directions geometrically.   Our analysis yields three findings. (1) Successful atta
    
[^347]: 没有任何等变架构能覆盖所有等变注意力

    No Equivariant Architecture Covers All Equivariant Attention

    [https://arxiv.org/abs/2608.30417](https://arxiv.org/abs/2608.30417)

    本文完整刻画了等变多头自注意力的结构，并证明其等变性轨迹分裂为极多个不可约分支，因此任何固定的等变注意力架构至多覆盖其中一个分支，必然损失对等变映射的表达能力。

    

    我们对等变多头自注意力给出了完整的刻画：若一个MHSA层对某个对称群 $G$ 是等变的，则 $G$ 只能通过置换头簇来作用，且QK矩阵和OV矩阵必须满足与该群作用相关联的等变性约束。由此我们证明，任何通过多项式参数化无约束MHSA参数来实现精确等变性的固定MHSA架构，都不可避免地在等变映射类中造成表达能力损失：无约束MHSA的等变性轨迹在约化参数空间中形成由极多个Zariski不可约分支构成的并集，而任何单一架构至多只能覆盖其中一个分支。对于 $G=D_4$ 作用于正则表示的 $C$ 个副本作为词元特征空间的情形，我们证明在八个注意力头下存在 $\Omega(C^{64})$ 个分支。

    arXiv:2608.30417v1 Announce Type: new  Abstract: We give a complete characterization of equivariant multi-head self-attention (MHSA): if an MHSA layer is equivariant to a symmetry group $G$, then $G$ can only act by permuting head-clusters, with QK and OV matrices satisfying an equivariance constraint tied to the group action. As a consequence, we prove that any fixed MHSA architecture that achieves exact equivariance by polynomially parameterizing unconstrained MHSA parameters inevitably leads to expressivity loss within the class of equivariant maps: the equivariance locus of unconstrained MHSA forms a union of extremely many Zariski-irreducible components in a reduced parameter space, and any single architecture covers at most one. For $G=D_4$ acting on $C$ copies of the regular representation as the token feature space, we show that there are $\Omega(C^{64})$ components for eight attention heads.
    
[^348]: Tail-Replay：摆脱混合大语言模型前缀缓存中线性注意力的魔咒

    Tail-Replay: Escaping the Curse of Linear Attention in Prefix Caching for Hybrid LLMs

    [https://arxiv.org/abs/2608.30310](https://arxiv.org/abs/2608.30310)

    提出Tail-Replay前缀缓存机制，利用线性注意力可视为输入前缀结构化有损压缩这一洞察，使混合大语言模型摆脱循环状态检查点的限制，实现任意位置的token级前缀重用。

    

    混合大语言模型通过将全注意力层与线性注意力层交错排列，以降低长上下文推理的成本。这种结构使前缀缓存变得复杂：全注意力的键值缓存可以按token寻址，而线性注意力层维护的循环状态无法回滚到任意的前缀边界。现有的混合前缀缓存方法通过存储循环状态检查点来解决这种不匹配问题。因此，token级别的匹配只能在与已存储检查点对齐的位置上直接使用，这将前缀重用限制在了一组离散的边界上。我们提出了Tail-Replay，这是一种能够在混合大语言模型中实现无约束token级前缀重用的前缀缓存机制。其关键洞察在于，诸如Gated DeltaNet之类的线性注意力机制可以被视为对输入前缀的结构化有损压缩：门控循环更新会逐步衰减……

    arXiv:2608.30310v1 Announce Type: cross  Abstract: Hybrid large language models interleave full-attention layers with linear-attention layers to reduce the cost of long-context inference. This structure complicates prefix caching: full-attention key-value caches are token-addressable, whereas linear-attention layers maintain recurrent states that cannot be rolled back to arbitrary prefix boundaries. Existing hybrid prefix caching methods address this mismatch by storing recurrent-state checkpoints. As a result, token-level matches are directly usable only at positions aligned with stored checkpoints, constraining prefix reuse to a discrete set of boundaries. We present Tail-Replay, a prefix caching mechanism that enables unconstrained token-level prefix reuse in hybrid large language models. The key insight is that linear-attention mechanisms such as Gated DeltaNet can be viewed as a structured, lossy compression of the input prefix: gated recurrent updates progressively attenuate the 
    
[^349]: INTERVenE：基于时间抽象区间的Transformer用于短期医疗事件预测

    INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction

    [https://arxiv.org/abs/2608.29901](https://arxiv.org/abs/2608.29901)

    INTERVenE提出了一族以区间式知识时间抽象（KBTA）生成的命名临床概念作为token输入的Transformer架构，通过自回归解码器和双向编码器两种变体，在保留时间临床意义的同时实现ICU中可解释的短期医疗事件与风险预测。

    

    重症监护室（ICU）中的电子健康记录（EHR）预测模型必须能够从稀疏且不规则的测量数据中学习，同时保留时间的临床意义并支持透明的决策。我们提出了INTERVenE，这是一族Transformer架构，其输入是基于区间的、基于知识的时间抽象（KBTA），即一个由命名的临床概念（状态、趋势、事件、上下文）组成的token流，这些概念取自精心整理的医学本体，而非未命名的分箱索引或原始测量三元组。这一命名层正是我们要求KBTA所实现的功能：它使模型对每个token的归因在构造上就能对应到具体的临床概念。INTERVenE提供两种互补的变体：一种是自回归解码器，可生成未来的抽象轨迹，并在每一步输出风险读数（从而定位风险何时升高以及在哪些事件之后升高）；另一种是双向编码器，用于单次前向传播中联合预测风险与时间……

    arXiv:2608.29901v1 Announce Type: cross  Abstract: Electronic Health Record (EHR) prediction models in the intensive care unit must learn from sparse and irregular measurements while preserving the clinical meaning of time and supporting transparent decision-making. We present INTERVenE, a family of Transformer architectures whose input is an interval-based, knowledge-based temporal abstraction (KBTA), a token stream of named clinical concepts (states, trends, events, contexts) drawn from a curated medical ontology, rather than an unnamed bin index or a raw measurement triplet. This naming layer is what we ask KBTA to do: it makes the model's per-token attributions resolve to clinical concepts by construction. INTERVenE offers two complementary variants: an auto-regressive decoder that generates future abstraction trajectories with a per-step risk readout (localizing \emph{when} and \emph{after which events} risk rises), and a bidirectional encoder for single-pass joint risk and time-t
    
[^350]: ICON分解：用于模型审计的深度表示多变量概念级解释

    ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing

    [https://arxiv.org/abs/2608.26083](https://arxiv.org/abs/2608.26083)

    ICON分解通过多变量分析，在控制其他概念和结果后精确量化每个概念对模型表示的独特贡献，从而有效识别捷径学习并提高解释的准确性。

    

    arXiv:2608.26083v1 公告类型：新 摘要：深度神经网络经常利用训练数据中的虚假关联，这种失败被称为捷径学习。基于概念的可解释性方法通过测试诸如患者性别或扫描仪设置等概念是否能从网络层中解码来筛选捷径。由于每个概念是单独评估的，这些方法可能会将概念之间的相关性误认为是模型使用它们的证据。我们引入了ICON分解，它转而量化每个概念在考虑所有其他概念和结果后所解释的层方差的比例。在具有已知真实标签的合成数据上，ICON比七种替代基线方法更准确地恢复了概念重要性。在皮肤病变和脑成像模型中，它隔离了模型真正依赖的概念，量化了任何提供的概念未解释的表示部分，并产生了我们验证过的稀疏解释。

    arXiv:2608.26083v1 Announce Type: new  Abstract: Deep neural networks often exploit spurious associations in their training data, a failure known as shortcut learning. Concept-based explainability methods screen for shortcuts by testing whether concepts such as a patient's sex or scanner settings can be decoded from a network layer. Because each concept is evaluated in isolation, these methods can mistake correlations between concepts as evidence that the model uses them. We introduce ICON decomposition, which instead quantifies how much of a layer's variance each concept explains after accounting for all other concepts and the outcome. On synthetic data with known ground truth, ICON recovers concept importance more accurately than seven alternative baseline methods. On skin-lesion and brain-imaging models, it isolates the concepts on which a model genuinely relies, quantifies the representation unexplained by any of the supplied concepts, and yields sparse explanations that we validat
    
[^351]: DataKernelBench：大语言模型能否在GPU上优化数据库查询？

    DataKernelBench: Can LLMs Optimize Database Queries on GPUs?

    [https://arxiv.org/abs/2608.25061](https://arxiv.org/abs/2608.25061)

    本文提出了DataKernelBench基准测试，证明大语言模型能通过执行引导修复在GPU上优化数据库查询，最强配置实现2.11倍加速，并发现内核融合、完整查询专业化和工作负载上下文是关键因素。

    

    arXiv:2608.25061v1 公告类型：新 摘要：GPU日益加速数据库系统，但查询特定的峰值性能通常仍依赖手写内核。现有的大语言模型内核基准测试专注于机器学习算子，未测试不规则、异构、数据移动密集型的数据库风格算子。我们引入了DataKernelBench，它将SQL转换为经过验证的PyTorch TorchPlan程序，并评估大语言模型通过执行引导修复来优化核心张量受限片段或完整查询（使用CUDA或Triton）的能力。在TPC-H SF10数据集和H100 GPU上对十个专有和开源权重模型进行的测试中，最强的完整查询CUDA配置在完全通过率下比torch.compile实现了$2.11\times$的加速。我们发现，性能更高的实现通常采用内核融合和执行策略更改，更强的模型从完整查询专业化中获益最多，且工作负载上下文比硬件上下文更重要。为处理大数据...

    arXiv:2608.25061v1 Announce Type: new  Abstract: GPUs increasingly accelerate database systems, but query-specific peak performance still often relies on hand-written kernels. Existing LLM kernel benchmarks focus on machine learning operators, leaving irregular, heterogeneous, data-movement-heavy database-style operators untested. We introduce DataKernelBench, which translates SQL into validated PyTorch TorchPlan programs and evaluates LLMs that optimize either the core tensor-bounded snippet or the full query in CUDA or Triton through execution-guided repair. Across ten proprietary and open-weight models on TPC-H SF10 with an H100 GPU, the strongest full-query CUDA configuration achieves $2.11\times$ speedup over torch.compile at full pass rate. We find that higher-performing implementations commonly use kernel fusion and execution-strategy changes, stronger models benefit most from full-query specialization, and workload context matters more than hardware context. To handle data larg
    
[^352]: 校准保持剪枝：压缩作为可靠性契约

    Calibration-Preserving Pruning: Compression as a Reliability Contract

    [https://arxiv.org/abs/2608.23744](https://arxiv.org/abs/2608.23744)

    本文提出校准保持剪枝方法，通过增强剪枝分数保持分数几何结构，在保证共形预测覆盖的同时减小预测集大小，实现更高效的模型压缩。

    

    分裂共形预测，而非剪枝规则，在剪枝模型独立于共形校准分割固定后，提供有限样本边际覆盖率。我们研究独立的效率问题：剪枝能否充分保持分数几何结构，以获得更小的有效预测集？校准保持剪枝（CPP）通过非共形梯度显著性增强基础剪枝分数，并使用独立的剪枝、验证选择、共形校准和测试分割。有界分数扰动意味着有界共形分位数偏移和受控的集合膨胀，但这并不使通用覆盖定理具有CPP特异性。最终在50%稀疏度下使用五个种子的Qwen2.5-1.5B结果显示，在大标签任务上收益最大。在DBpedia-14上，CPP-SparseGPT将平均集合大小从10.1降至8.6，同时准确率从0.347变为0.366；CPP-Wanda将11.2降至9.0，准确率权衡从...

    arXiv:2608.23744v1 Announce Type: new  Abstract: Split conformal prediction, not the pruning rule, supplies finite-sample marginal coverage once a pruned model is fixed independently of the conformal calibration split. We study the separate efficiency problem: can pruning preserve score geometry well enough to obtain smaller valid prediction sets? Calibration-Preserving Pruning (CPP) augments a base pruning score with nonconformity-gradient saliency and uses disjoint pruning, validation-selection, conformal-calibration, and test splits. Bounded score perturbations imply bounded conformal-quantile shifts and controlled set inflation, but do not make the generic coverage theorem CPP-specific. Final five-seed Qwen2.5-1.5B results at 50\% sparsity show the largest gains on large-label tasks. On DBpedia-14, CPP-SparseGPT reduces mean set size from \(10.1\) to \(8.6\) while changing accuracy from \(0.347\) to \(0.366\); CPP-Wanda reduces \(11.2\) to \(9.0\) with an accuracy trade-off from \(
    
[^353]: 基于递归的分块稳定自适应三次正则化与子求解器

    Blockwise Stabilized Adaptive Cubic Regularization with Subsolvers via Recurrence

    [https://arxiv.org/abs/2608.22129](https://arxiv.org/abs/2608.22129)

    本文提出了一种分块稳定的自适应三次正则化方法，通过Krylov子空间无矩阵求解精确三次步，在保持最优收敛速率的同时支持任意大规模张量，并提供了完整的理论保证。

    

    三次正则化牛顿方法具有最优的$\mathcal{O}(\epsilon^{-3/2})$全局收敛速率，但稠密子问题的求解限制了可行的块大小。可扩展的三次牛顿变体将真实块曲率替换为对角、低秩、Kronecker因子化或草图化的替代，并且通常放弃精确的三次步。我们引入了一种分块优化器，它对每个参数张量独立最小化一个三次模型，使用真实块Hessian，在每块自适应三次常数和全损失单调保护下。任意大的张量通过无矩阵的Lanczos构建Krylov子空间处理，我们证明该步骤最小化了三次模型。理论还提供了$\mathcal{O}(\epsilon^{-3/2})$迭代复杂度界、二阶保证和单调每块下降。该外方案的四种变体与原始自适应三次正则化（ARC）优化器进行了评估。

    arXiv:2608.22129v2 Announce Type: replace  Abstract: Cubic regularized Newton methods have the optimal $\mathcal{O}(\epsilon^{-3/2})$ global rate, but a dense subproblem solve limits the feasible block size. Scalable Cubic Newton variants replace the true block curvature with a diagonal, low-rank, Kronecker-factored, or sketched surrogate and, most often, give up the exact cubic step. We introduce a blockwise optimizer that minimizes an independent cubic model per parameter tensor over the true block Hessian, under a per-block adaptive cubic constant and a monotone guard on the full loss. Arbitrarily large tensors are handled matrix-free in a Lanczos-built Krylov subspace, where we prove that the step minimizes the cubic model. The theory also supplies the $\mathcal{O}(\epsilon^{-3/2})$ iteration complexity bound, a second-order guarantee, and monotone per-block descent. Four variants of this outer scheme are evaluated against the original adaptive regularization with cubics (ARC) opti
    
[^354]: 纯TD：无评估时搜索的后台棋金钱游戏强化学习

    PureTD: Reinforcement Learning for Backgammon Money Games with No Evaluation-time Search

    [https://arxiv.org/abs/2608.15146](https://arxiv.org/abs/2608.15146)

    该论文证明纯自对弈强化学习无需评估时搜索即可训练出接近最优的后台棋模型，在金钱游戏中速度快且强于带单步搜索的开源引擎。

    

    摘要：我们重新审视了Tesauro的TD-Gammon在无评估时搜索的后台棋金钱游戏设置中的应用。棋子移动和骰子立方体操作（使用加倍骰子）均通过自对弈强化学习从零开始学习，仅使用最少的硬编码逻辑，且不依赖专家特征。在此设置下，我们证明纯自对弈强化学习足以训练出接近当前最优水平的模型。具体而言，对于带骰子立方体的金钱游戏，我们的无搜索模型评估速度更快，且显著强于开源引擎GNU Backgammon和Open Sage在单步（1-ply）前瞻搜索下的表现。

    arXiv:2608.15146v1 Announce Type: new  Abstract: We revisit Tesauro's TD-Gammon for backgammon money games in the setting of no evaluation-time search. Both checker play and cube action (use of the doubling cube) are learned from scratch via self-play reinforcement learning (RL), with minimal hand-coded logic and no expert features. In this setting, we demonstrate that pure self-play RL suffices to train models that reach near-state-of-the-art playing strength. Specifically, for cubeful money games, our search-free model evaluates faster and is substantially stronger than the open-source engines GNU Backgammon and Open Sage running a one-move (1-ply) look-ahead search.
    
[^355]: 混合来源大型语言模型文本中的最优水印定位

    Optimal Watermark Localization in Mixed-Source Large Language Model Texts

    [https://arxiv.org/abs/2608.14906](https://arxiv.org/abs/2608.14906)

    本文提出了混合来源LLM文本中水印定位的渐近最优框架，明确了全局检测、发现和分类的相变边界，并证明发现任务难度高于分类。

    

    水印提供了一种有原则的方式来认证由大型语言模型（LLMs）生成的文本。然而，在实践中，最终文本可能是混合来源的，经过改写、插入、删除或释义后，水印证据仅存留在部分标记位置。尽管先前的研究已经探讨了水印信号的全局检测，但何时能对这些信号进行定位仍不清楚。我们将水印定位问题表述为基于关键统计量的标记级多重检验问题，其中包含一个潜在指示符，记录每个位置的水印依赖是否存活。在由信号稀疏性、下一标记浓度和有效词汇增长指数所索引的渐近框架下，我们推导出全局检测的尖锐边界，以及在坐标级基于关键统计的定位规则类别内的发现和分类相变。我们表明，发现严格比分类更难，并提供了最优规则。

    arXiv:2608.14906v1 Announce Type: cross  Abstract: Watermarking provides a principled way to authenticate text generated by large language models (LLMs). In practice, however, the final text may be mixed-source, with watermark evidence surviving at only a subset of token positions after rewriting, insertion, deletion, or paraphrasing. Although prior work has studied global detection of watermark signals, when such signals can be localized remains unclear. We formulate watermark localization as a token-level multiple-testing problem based on pivotal statistics, with a latent indicator recording whether watermark dependence survives at each position. Under an asymptotic regime indexed by exponents for signal sparsity, next-token concentration, and effective-vocabulary growth, we derive a sharp boundary for global detection and phase transitions for discovery and classification within the class of coordinatewise pivot-based localization rules. We show that discovery is strictly harder tha
    
[^356]: DYSANOS：生成式动态平滑无套利非参数期权曲面

    DYSANOS Generative Dynamic Smooth Arbitrage-free Non-parametric Option Surfaces

    [https://arxiv.org/abs/2608.12587](https://arxiv.org/abs/2608.12587)

    本文提出首个生成式无套利期权曲面模型DYSANOS，能够生成未来多年每日期权价格路径，并验证其优于传统隐含波动率PCA模型。

    

    本文介绍了DYSANOS，这是首个生成式市场模型，用于生成所有行权价和到期日的平滑且无静态套利的SANOS期权曲面。该模型旨在生成未来数年的每日现货和期权价格完整路径。我们提出了一种稳健且实用但相对简单的基线隐状态生成模型，形式为AR(1)模型。我们讨论了模型设置、数据管道和训练过程，并研究了动态套利的数值存在性。我们使用Option Metrics的IvyDB S&P指数数据（2020年至2025年）展示了模型性能，并将其与纯隐含波动率主成分分析模型进行了比较。

    arXiv:2608.12587v1 Announce Type: cross  Abstract: This article presents with DYSANOS the first generative market model for smooth SANOS option surfaces for all strikes and expiries which are free of static arbitrage. Our model is designed to generate entire paths of daily spot and option prices for years in the future.   We present a robust and useful if somewhat simplistic baseline hidden state generative model in the form of an AR(1) model. We discuss model setup, data pipeline, and training and investigate numerical resence of dynamic arbitrage. We illustrate model performance on Option Metrics' IvyDB S\&P Index data from 2020 to~2025 and compare it to a pure implied-vol PCA model.
    
[^357]: 迈向长时程智能体的可靠上下文压缩：执行不稳定性的实证研究

    Towards Reliable Context Compression for Long-Horizon Agents: An Empirical Study of Execution Instability

    [https://arxiv.org/abs/2608.06503](https://arxiv.org/abs/2608.06503)

    本文揭示循环上下文压缩会削弱智能体对近期交互的响应并导致执行不稳定，据此提出验证器引导的TRACE框架，通过边界局部评估压缩事件并利用摘要偏好优化自然语言压缩提示词，在任务性能、多运行可靠性和上下文-执行效率上超越现有基线。

    

    循环上下文压缩可以控制长时程智能体中上下文的增长，但其对行为的影响仍知之甚少。在这项初步实证研究中，我们表明压缩可能会削弱近期交互的影响，导致受阻动作增多、重复探索以及多次运行之间的不稳定性。受这些观察启发，我们提出了TRACE，一个由验证器引导的框架，它通过从相同环境状态出发的成对闭环延续来评估单个压缩事件，并利用摘要偏好来优化自然语言压缩提示词，同时保持所有模型冻结不变。在AppWorld上的初步结果显示，与现有压缩基线相比，该方法在任务性能、多运行可靠性和上下文-执行效率方面均有提升。这些发现为边界局部评估作为实现可靠智能体上下文压缩的一个有前景的方向提供了早期证据。

    arXiv:2608.06503v2 Announce Type: replace  Abstract: Recurrent context compression controls context growth in long-horizon agents, but its behavioral effects remain poorly understood. In this preliminary empirical study, we show that compression can weaken the influence of recent interactions, increasing blocked actions, repeated exploration, and instability across runs. Motivated by these observations, we introduce TRACE, a verifier-guided framework that evaluates individual compaction events through paired closed-loop continuations from the same environment state and uses summary preferences to optimize a natural-language compression prompt while keeping all models frozen. Initial results on AppWorld show improvements over existing compression baselines in task performance, multi-run reliability, and context--execution efficiency. These findings provide early evidence for boundary-local evaluation as a promising direction for reliable agent context compression.
    
[^358]: 多模态、多时相地理空间基础模型改进全球地上生物量估计

    Multi-Modal, Multi-Temporal Geospatial Foundation Models Improve Global Above-Ground Biomass Estimation

    [https://arxiv.org/abs/2608.04792](https://arxiv.org/abs/2608.04792)

    本文对13个地理空间基础模型在全球地上生物量估计任务上进行了系统性基准测试，发现多模态、多时相的地理空间基础模型能够超越全监督基线方法，从而改进全球地上生物量的估计。

    

    从卫星影像中准确估计地上生物量（AGB）对于大尺度的碳储量监测至关重要，但在全球尺度上这仍然是一项具有挑战性的回归任务。地理空间基础模型（GFMs）近来作为一种有前景的机器学习范式出现，能够从地球观测（EO）数据中提取通用表示，但它们在生物量估计等定量回归任务中的效用尚未被充分探索，因为大多数基准测试集中于分类和分割任务。本文对13个地理空间基础模型在地面生物量估计任务上进行了基准测试，具体使用GEDI L4A生物量值。我们将它们与在AGBD（A Global-scale Biomass Dataset，一个覆盖多种生物群落和地理区域的机器学习就绪基准数据集）上训练的全监督最先进（SOTA）基线进行比较。所评估的大多数地理空间基础模型相较于监督基线表现明显不佳，无论是……（原文截断）

    arXiv:2608.04792v2 Announce Type: replace  Abstract: Accurate estimation of above-ground biomass (AGB) from satellite imagery is essential for the large-scale monitoring of carbon stocks, yet it remains a challenging regression task at global scale. Geospatial Foundation Models (GFMs) have recently emerged as a promising machine learning paradigm to derive general-purpose representations from Earth observation (EO) data, but their utility for quantitative regression tasks like biomass estimation remains largely unexplored, as most benchmarks focus on classification and segmentation. Here, we benchmark 13 GFMs for above-ground biomass estimation, specifically GEDI L4A biomass values. We compare them against a fully supervised state-of-the-art (SOTA) baseline trained on A Global-scale Biomass Dataset (AGBD), a machine learning-ready benchmark spanning diverse biomes and geographies. Most of the GFMs evaluated substantially underperform with respect to the supervised baseline, whether fro
    
[^359]: ContinualSkillBench：LLM智能体能否真正进化其能力？

    ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities?

    [https://arxiv.org/abs/2608.03874](https://arxiv.org/abs/2608.03874)

    该论文提出动态评估框架ContinualSkillBench，用于衡量LLM智能体的上下文内持续技能学习能力，发现顺序执行通常提升性能但收益因模型和领域而异，且显式技能库仅在需要可复用流程或精确输出的任务中带来选择性收益。

    

    现代智能体框架为大语言模型配备外部技能库以解决复杂任务。然而，这些系统能否有效地进化其技能，以及由此产生的技能能否提升任务解决能力，仍不清楚。为了弥补这一空白，我们引入了ContinualSkillBench，一个用于上下文内持续技能学习的动态评估框架。它涵盖五个领域，每个领域包含按难度排序的100个任务以及技能迁移的机会。我们的实验表明，顺序执行通常能提升性能，但收益在不同模型和领域之间差异显著。此外，在两个模型和三个领域的实验中，仅保留上下文和反馈而不进行技能维护，其平均性能与维护显式技能库几乎相同。尽管如此，显式技能对于需要可复用流程或精确输出的任务仍能提供选择性的收益。我们进一步发现……

    arXiv:2608.03874v2 Announce Type: replace  Abstract: Modern agent frameworks equip large language models with external skill libraries to solve complex tasks. However, it remains unclear whether these systems can effectively evolve their skills and whether the resulting skills improve task-solving capabilities. To bridge this gap, we introduce ContinualSkillBench, a dynamic evaluation framework for in-context continual skill learning. It covers five domains, each with 100 tasks ordered by difficulty and opportunities for skill transfer. Our experiments show that sequential execution generally improves performance, but the gains vary substantially across models and domains. Moreover, across two models and three domains, retaining context and feedback without skill maintenance yields nearly the same mean performance as maintaining an explicit skill library. Explicit skills nevertheless provide selective benefits for tasks requiring reusable procedures or precise outputs. We further find 
    
[^360]: 最优无歧义DNF与Alon-Saks-Seymour猜想

    Optimal Unambiguous DNFs and Alon-Saks-Seymour

    [https://arxiv.org/abs/2608.02533](https://arxiv.org/abs/2608.02533)

    本文构造了宽度为O(n)但0-证书复杂度为Ω(n²)的无歧义DNF，并通过提升定理实现了对Alon-Saks-Seymour猜想的最优反驳以及团与独立集问题的最优通信下界。

    

    我们构造了宽度为 $O(n)$ 但 0-证书复杂度为 $\Omega(n^2)$ 的无歧义DNF（析取范式）。通过利用这些DNF的特殊结构，我们证明了一个使用常数大小组件的提升定理，该定理将DNF提升为一个通信问题，同时将证书复杂度的分离无损地转化为通信复杂度的分离。这导致了对Alon-Saks-Seymour猜想的最优反驳，以及团与独立集问题的最优通信下界，将Balodis、Ben-David、Göös、Jain和Kothari（FOCS 2021, SICOMP 2023）的先前结果改进了若干双对数因子。作为我们这一构造在查询复杂度和学习理论中的进一步应用，我们展示了：(a) 一族在证书复杂度和近似度之间具有最优四次分离的布尔函数，以及 (b) 一个样本压缩下界……

    arXiv:2608.02533v2 Announce Type: replace-cross  Abstract: We construct unambiguous DNFs having width $O(n)$ but $0$-certificate complexity $\Omega(n^2)$. By utilizing the special structure of these DNFs, we prove a lifting theorem with a constant-sized gadget that lifts the DNF to a communication problem, while losslessly translating the separation in certificate complexity to a separation in communication complexity. This leads to an optimal refutation of the Alon-Saks-Seymour conjecture, as well as an optimal communication lower bound for the Clique versus Independent Set problem, improving the previous results of Balodis, Ben-David, G\"{o}\"{o}s, Jain and Kothari (FOCS 2021, SICOMP 2023) by several doubly logarithmic factors. As further applications of our construction to query complexity and learning theory, we exhibit: (a) a family of Boolean functions that has an optimal quartic separation between certificate complexity and approximate degree, and (b) a sample compression lower 
    
[^361]: 缺失数据的流匹配

    Flow Matching with Missing Data

    [https://arxiv.org/abs/2607.28698](https://arxiv.org/abs/2607.28698)

    该论文提出缺失数据流匹配方法，将训练样本的缺失坐标视为潜在变量并证明其oracle修正在任意缺失机制下都能精确恢复完整数据的流匹配目标，同时通过有限样本分析揭示了缺失机制只是转移而非增加估计量的方差。

    

    流匹配假设训练数据是完全观测的，然而许多现实应用很少能提供这样的数据。我们研究了缺失数据流匹配，该方法将训练样本的缺失坐标视为潜在变量，并在这些坐标可能的取值上对流匹配损失进行平均。我们首先证明了oracle修正的精确性：在任意缺失机制下，使用真实的模式条件后验进行补全可以恢复完整数据的流匹配目标。在完全随机缺失（MCAR）和随机缺失（MAR）下，该后验等同于普通的完整数据条件分布；而在非随机缺失（MNAR）下，它通常依赖于缺失机制，可能需要额外的结构才能识别。随后，我们的有限样本分析回答了该算法遗留的设计问题：缺失机制是转移而非额外增加估计量的方差，而对于固定参数的损失估计器……

    arXiv:2607.28698v3 Announce Type: replace  Abstract: Flow matching assumes fully observed training data, which many real-world applications rarely provide. We study Missing-Data Flow Matching, which treats the missing coordinates of training samples as latent variables and averages the flow matching loss over the values they could take. We first prove that the oracle correction is exact: completing from the true pattern-conditioned posterior recovers the complete-data flow-matching objective under any missingness mechanism. Under missing completely at random (MCAR) and missing at random (MAR), this posterior equals the ordinary complete-data conditional, while under missing not at random (MNAR) it generally depends on the missingness mechanism and may require additional structure to identify. Our finite-sample analysis then answers design questions that the algorithm leaves open. Missingness transfers estimator variance rather than adding it, and for the fixed-parameter loss estimator 
    
[^362]: 电路归因结论取决于提取什么以及如何比较

    Circuit Claims Depend on What Is Extracted and How It Is Compared

    [https://arxiv.org/abs/2607.18921](https://arxiv.org/abs/2607.18921)

    论文通过在 Lean 战术预测基准上的系统实验表明，可解释性研究中“电路”的结论高度依赖于提取对象的定义和比较方法——不同提取方式得到的电路重叠度很低，说明行为保持并不能唯一确定一个电路。

    

    电路提取旨在识别在消融实验下其存在能够保持目标行为的模型组件，但仅仅保持行为并不能唯一确定一个电路。我们在一个具有固定证明规则和随机化表面形式的合成 Lean 战术预测基准上研究这一问题。在同一个 transformer 的稠密与权重稀疏检查点上，分别在原子证明和组合证明上进行评估，我们改变了所报告的提取对象：紧凑的预测保持电路、保留周围读取、写入和路由结构的更广泛图，或使用消融后损失预算提取的子图。首先，不同组件间连接组之间的重叠度很低，有时接近随机基线。在确定性选择下，所选注意力头集合的重叠度要高得多，但当在得分相同的注意力头之间进行随机选择时，重叠度急剧下降。其次，各条件的归一化图大小排序……（摘要在此处截断）

    arXiv:2607.18921v2 Announce Type: replace-cross  Abstract: Circuit extraction identifies model components whose presence preserves a target behavior under ablation, but preserving behavior does not single out one circuit. We study this in a synthetic Lean tactic-prediction benchmark with fixed proof rules and randomized surface form. Across dense and weight-sparse checkpoints of the same transformer, evaluated on atomic and compositional proofs, we vary which extracted object is reported: a compact prediction-preserving circuit, a broader graph retaining surrounding read, write, and routing structure, or a subgraph extracted with a post-ablation loss budget. First, overlap between groups of component-to-component connections is low, sometimes close to a random baseline. Overlap is much higher for selected attention-head sets under deterministic selection, but falls sharply when selection among equally scored heads is randomized. Second, the normalized graph-size ranking of conditions t
    
[^363]: 面向工具结构化大语言模型推理的仅幅度FFN干预：方法、门控评估协议与跨模型实证结果

    Amplitude-Only FFN Intervention for Tool-Structured LLM Inference: Method, Gated Evaluation Protocol, and Cross-Model Empirical Results

    [https://arxiv.org/abs/2607.11183](https://arxiv.org/abs/2607.11183)

    该论文提出幅度门控（AG）这一无需重训练的推理时干预方法，通过掩码FFN激活坐标，在多个Qwen模型的函数调用和ToolBench端点上实现了1.6至2.8个百分点的稳定性能提升。

    

    我们研究了幅度门控，这是一种推理时干预方法，在前馈网络（FFN）激活坐标上进行掩码操作，同时保持预训练权重固定。六个SwiGLU位置分别对共享输入与仅门控输入、扩展的门控/数值通道以及FFN输出进行分离。幅度排序熵掩码和Newton-Schulz窗口排序变体提供了无需重新训练的替代前向路径。修正后的九月数据集涵盖三个Qwen检查点和八个文本/工具端点。在冻结的本地文本评分器下，所展示的Qwen3.5-9B非NS配置在两个函数调用端点上分别提升了2.61和2.67个百分点（pp），在静态ToolBench的域内/域外端点上分别提升了1.64和2.82个百分点。将较低的历史/九月AG均值与较高的基线均值进行比较，在这些路径上仍保留正的裕度。这支持了对所观测的两次执行的鲁棒性，而非形式上的统计显著性。

    arXiv:2607.11183v4 Announce Type: replace-cross  Abstract: We study Amplitude Gating (AG), an inference-time intervention that masks feed-forward network (FFN) activation coordinates while keeping pretrained weights fixed. Six SwiGLU sites separate shared and gate-only inputs, expanded gate/value channels, and the FFN output. Magnitude-ranked entropy masks and a Newton-Schulz windowed ranking variant provide alternative forward paths without retraining. The corrected September collection covers three Qwen checkpoints and eight text/tool endpoints. Under frozen local text scorers, the displayed Qwen3.5-9B non-NS configurations improve two function-call endpoints by +2.61 and +2.67 percentage points (pp), and static ToolBench in/out-domain endpoints by +1.64 and +2.82 pp. Comparing the lower historical/September AG mean with the higher baseline mean leaves positive margins on these routes. This supports robustness to the two observed executions, rather than formal statistical significanc
    
[^364]: 面向离散马尔可夫随机场的端到端量子-经典混合采样工作流：一项可复现的案例研究

    An End-to-End Hybrid Quantum--Classical Sampling Workflow for Discrete Markov Random Fields: A Reproducible Case Study

    [https://arxiv.org/abs/2607.09893](https://arxiv.org/abs/2607.09893)

    该论文构建了一个可复现的端到端量子-经典混合采样工作流，实证表明现代经典MCMC采样器能大幅缩小与量子采样的差距，且在计入$O(2^n)$预处理成本后，量子方法在实际运行时间上并无优势。

    

    从离散马尔可夫随机场（MRF）中采样是一个困难问题。我们研究了针对小型MRF的振幅编码独立同分布采样，其中$2^n$个目标概率通过经典方法预先计算。这消除了量子指数级加速的可能性，但使得能够基于独立电路样本（$\tau \approx 1$）与经典MCMC进行清晰的对比。在跨越五个图族的60个实例上（1k步预热，保留3k个样本），量子方法相对于单点Gibbs、块Gibbs、调优块采样和并行回火的平均有效样本量（ESS）比率分别为16.35、7.29、1.82和1.79，表明现代经典采样器已大幅缩小了这一差距。将$O(2^n)$预处理成本摊销到实际运行时间中，精确的逆CDF采样可达到1770万ESS/s，而量子采样器仅为48.8万ESS/s（平均速率相差36倍，单实例最高相差153倍），证实量子方法在实际运行时间上并无优势。我们刻画了MCMC自相关成本并进行了基准测试。

    arXiv:2607.09893v2 Announce Type: replace-cross  Abstract: Sampling from discrete Markov random fields (MRFs) is a hard problem. We study amplitude-encoded i.i.d. sampling for small MRFs where $2^n$ target probabilities are precomputed classically. This removes quantum exponential speedup but allows a clean comparison against classical MCMC based on independent circuit samples ($\tau \approx 1$). Across 60 instances spanning five graph families (1k-step burn-in, 3k retained samples), the mean ESS ratios of Quantum to Single-Site Gibbs, Block Gibbs, Tuned-Block, and Parallel Tempering are $16.35$, $7.29$, $1.82$, and $1.79$, showing modern classical samplers substantially close this gap. Amortizing $O(2^n)$ preprocessing into wall-clock time, exact inverse-CDF sampling yields $17.7\text{M}$ ESS/s versus $488\text{K}$ ESS/s for the quantum sampler ($36\times$ mean rate, $153\times$ per-instance), confirming no wall-clock advantage. We characterize MCMC autocorrelation costs and benchmark
    
[^365]: 局部正则性下基于一阶增广拉格朗日方法的非凸复合泛函约束优化

    Nonconvex Composite Functional Constraints via First-Order Augmented Lagrangian Methods under Local Regularity

    [https://arxiv.org/abs/2607.08954](https://arxiv.org/abs/2607.08954)

    该论文在仅需可行集附近局部乘子正则性的弱条件下，无需假设乘子有界，通过带对偶正则化的光滑近端线性增广拉格朗日原始-对偶方法，为非凸复合约束优化问题建立了非渐近KKT收敛保证，达到 $O(K^{-1/3})$ 的KKT误差率。

    

    我们研究了一类非凸约束优化问题在局部乘子正则性条件下的原始-对偶方法的非渐近收敛性。目标函数和泛函不等式约束均为凸Lipschitz外函数与光滑非线性映射的复合。与现有分析要求正则性全局成立或沿整个迭代序列成立不同，我们仅在可行集附近施加乘子正则性。利用具有紧对偶集的辅助极小极大问题，我们推导出Lyapunov估计，在可行初始化和足够大的罚参数下识别出近似可行的迭代点，而无需假设乘子有界。随后，局部正则性给出了相应乘子的界并使对偶截断失效，从而为原问题提供近似的KKT证书。带对偶正则化的光滑近端线性增广拉格朗日方法获得了 $O(K^{-1/3})$ 的KKT

    arXiv:2607.08954v2 Announce Type: replace-cross  Abstract: We study nonasymptotic convergence of primal-dual methods for a class of nonconvex constrained optimization problems under local multiplier regularity. Both the objective and functional inequality constraints are compositions of convex Lipschitz outer functions with smooth nonlinear mappings. Unlike existing analyses requiring regularity globally or along the entire iterate sequence, we impose multiplier regularity only near the feasible set. Using an auxiliary minimax problem with a compact dual set, we derive Lyapunov estimates that identify nearly feasible iterates under feasible initialization and a sufficiently large penalty, without assuming multiplier boundedness. Local regularity then bounds the associated multipliers and makes dual truncation inactive, yielding approximate KKT certificates for the original problem. The smoothed prox-linear augmented Lagrangian method with dual regularization yields an $O(K^{-1/3})$ KKT
    
[^366]: 答案正确，令牌有误：面向可验证奖励强化学习（RLVR）的尾部感知信用校准

    Right Answer, Wrong Tokens: Tail-Aware Credit Calibration for RLVR

    [https://arxiv.org/abs/2607.07976](https://arxiv.org/abs/2607.07976)

    针对GRPO类强化学习方法在成功轨迹中会同时强化错误尾部令牌的“正信用污染”问题，论文提出TACO方法，以策略自身为参考计算令牌尾部风险分数，在无需令牌级错误标签的情况下柔和降低高风险令牌的正信用，同时保留学习有用稀有模式的机会。

    

    尽管强化学习（RL）提升了大语言模型（LLM）的推理能力，但GRPO风格的方法可能会受到“正信用污染”这一失败模式的影响。具体而言，统一的信用分配会在成功的轨迹中，将局部错误的尾部令牌与有用的令牌一同强化。为解决这一问题，我们提出了尾部感知信用校准方法TACO（Tail-Aware Credit calibratiOn），该方法无需令牌级别的错误标签即可校准分配给此类令牌的正信用。为此，TACO让策略自身充当参考基准，通过评估每个采样令牌相对于策略局部期望的稀有程度来计算尾部风险分数。将该分数作为潜在局部错误的代理指标，TACO在保留非零权重的前提下，柔和地降低高尾部风险令牌的正信用，从而在限制强化这些错误令牌的同时，不消除模型学习有用稀有模式的机会。实验在四个LL（后文截断）……

    arXiv:2607.07976v2 Announce Type: replace-cross  Abstract: Although reinforcement learning (RL) improves the reasoning capabilities of large language models (LLMs), GRPO-style methods can suffer from a positive-credit contamination failure mode. Specifically, uniform credit assignment can reinforce locally erroneous tail tokens alongside useful tokens within successful trajectories. To address this problem, we propose Tail-Aware Credit calibratiOn (TACO), which calibrates the positive credit assigned to such tokens without token-level error labels. To this end, TACO lets the policy serve as its own reference, assessing each sampled token's rarity relative to the policy's local expectations to compute a tail-risk score. Using this score as a proxy for potential local errors, TACO softly reduces positive credit for high tail-risk tokens while retaining nonzero weights, limiting their reinforcement without eliminating opportunities to learn useful rare patterns. Experiments across four LL
    
[^367]: AirPASS：基于夹持天线系统的空中联邦学习

    AirPASS: Over-the-Air Federated Learning via Pinching Antenna Systems

    [https://arxiv.org/abs/2607.06768](https://arxiv.org/abs/2607.06768)

    本文提出AirPASS交替优化框架，通过同伦-黎曼裕度巩固方法与同伦辅助几何优化方法交替求解设备选择、接收波束成形和夹持天线位置的联合非凸优化问题，在夹持天线系统辅助的空中联邦学习中实现聚合失真阈值约束下的设备接入数量最大化。

    

    本文研究了接入点配备多波导夹持天线系统（PASS）的无线系统中的空中联邦学习（AirFL）。我们采用被广泛研究的面向学习的AirFL建模方式，其目标是在将聚合失真控制在预定阈值之下的同时，最大化被选择设备的数量。由此产生的设备选择、接收波束成形与夹持天线位置的联合优化问题，由于这些系统变量之间复杂的耦合关系而呈现高度非凸性。为应对这一挑战，我们提出了AirPASS——一个交替优化框架，包含两个核心组件：在固定PASS配置下，用于设备选择与接收波束成形的同伦-黎曼裕度巩固方法；以及在固定被选设备与波束成形器下，用于更新夹持天线位置的同伦辅助几何优化方法。

    arXiv:2607.06768v2 Announce Type: replace-cross  Abstract: This paper investigates over-the-air federated learning (AirFL) in wireless systems where the access point is equipped with a multi-waveguide pinching antenna system (PASS). We adopt the widely studied learning-oriented AirFL formulation, which seeks to maximize the number of selected devices while keeping the aggregation distortion below a prescribed threshold. The resulting joint optimization of device selection, receive beamforming, and pinching-antenna placement is highly nonconvex due to the intricate coupling among these system variables. To address this challenge, we develop AirPASS, an alternating optimization framework with two main components: a homotopy-Riemannian margin-consolidation method for device selection and receive beamforming under a fixed PASS configuration, and a homotopy-assisted geometry optimization method for updating the pinching-antenna positions under fixed selected devices and beamformer.
    
[^368]: HarnessMDP：利用离线强化学习学习控制大语言模型智能体执行框架

    HarnessMDP: Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning

    [https://arxiv.org/abs/2607.05458](https://arxiv.org/abs/2607.05458)

    提出HarnessMDP框架，将LLM智能体执行框架的控制建模为有限时域马尔可夫决策过程，通过离线强化学习从任务轨迹中学习框架策略，使框架能够根据演变的智能体状态自适应地选择结构性执行动作。

    

    执行框架是大语言模型智能体的关键组成部分，它决定了智能体如何与环境交互以及如何组织多步骤执行。先前的工作表明，改进工具、工作流和其他框架组件可以显著提升智能体性能。这表明执行框架可以被看作一个可学习的控制层，能够根据经验自适应地调整执行决策，而不是一个预先确定的执行配置。在本工作中，我们提出了HarnessMDP，一个使框架控制本身可学习的基于学习的框架。HarnessMDP将智能体执行建模为一个有限时域的马尔可夫决策过程（MDP），其中框架学习根据不断演变的智能体状态选择结构性执行动作。框架策略从收集的轨迹中基于任务性能进行离线学习，并在线部署以自适应地控制新的执行。在编码任务、tau-bench等基准上的大量实验……

    arXiv:2607.05458v2 Announce Type: replace-cross  Abstract: The execution harness is a critical component of LLM agents, shaping how the agent interacts with its environment and organizes multi-step execution. Prior work has shown that improving tools, workflows, and other harness components can substantially improve agent performance. This suggests that the harness can be treated as a learnable control layer that adapts execution decisions from experience, rather than as a predetermined execution configuration. In this work, we introduce HarnessMDP, a learning-based framework that makes harness control itself learnable. HarnessMDP formulates agent execution as a finite-horizon Markov decision process, where the harness learns to select structural execution actions from the evolving agent state. The harness policy is learned offline from collected trajectories using task performance and deployed online to adaptively control new executions. Extensive experiments on coding tasks, tau-benc
    
[^369]: ManifoldFlow：具有可学习奇异谱的SPD松弛Stiefel层

    ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum

    [https://arxiv.org/abs/2607.04535](https://arxiv.org/abs/2607.04535)

    ManifoldFlow通过对固定谱Stiefel层进行最小松弛，将权重分解为 W = Q S^{1/2}，在保持基向量位于Stiefel流形上的同时学习有界的正定奇异谱，使特征值裁剪成为直接的奇异值控制机制，并在序列、表格和图像任务中优于固定谱Stiefel层。

    

    正交层和Stiefel层为神经网络权重提供了精确的谱控制，但它们也带来了很强的建模约束：所有被表示的奇异值都被固定为1。许多受益于正交归一基的场景仍然需要方向相关的衰减或放大。我们提出了ManifoldFlow，这是对固定谱Stiefel层的一种最小松弛方法，它在保持基向量位于Stiefel流形上的同时，通过 W = Q S^{1/2}（其中 Q^T Q = I 且 S 为正定矩阵）学习一个有界的正谱。由于 W^T W = S，S 的特征值恰好是所实现权重的奇异值的平方，这使得特征值裁剪成为一种直接的奇异值控制机制。在配对的序列、表格和图像实验中，可学习的SPD谱在Stiefel先验有效的报告设置中优于固定谱的Stiefel对应方法，其中在循环语言模型投影任务中增益最大（原文摘要在此处截断）。

    arXiv:2607.04535v2 Announce Type: replace-cross  Abstract: Orthogonal and Stiefel layers give neural weights exact spectral control, but they also impose a strong modeling constraint: all represented singular values are fixed at one. Many settings that benefit from an orthonormal basis still need direction-dependent attenuation or amplification. We introduce ManifoldFlow, a minimal relaxation of a fixed-spectrum Stiefel layer that keeps the basis on the Stiefel manifold while learning a bounded positive spectrum through W = Q S^{1/2}, with Q^T Q = I and S positive definite. Since W^T W = S, the eigenvalues of S are exactly the squared singular values of the realized weight, making eigenvalue clipping a direct singular-value control mechanism. Across paired sequence, tabular, and image experiments, the learnable SPD spectrum improves the fixed-spectrum Stiefel counterpart in the reported settings where the Stiefel prior is useful, with the largest gains in recurrent language-model proje
    
[^370]: UniWind：基于物理信息状态路由的日前风电功率预测

    UniWind: Day-Ahead Wind Power Forecasting via Physics-Informed State Routing

    [https://arxiv.org/abs/2607.01670](https://arxiv.org/abs/2607.01670)

    UniWind框架通过将物理信息发电先验与数据驱动的状态感知校正相结合，有效区分气象驱动的发电潜力与本地环境运行因素的影响，实现更准确的日前风电功率预测。

    

    准确的日前风电功率预测对于电力系统的经济高效运行至关重要。风电发电主要由未来气象条件驱动，同时在发电过程中保持时间依赖性。在实际应用中，观测到的风电场功率既反映了气象驱动的发电潜力，也反映了与当地环境和运行因素相关的偏差。现有的物理模型能够提供有用的约束，但在跨风电场适应性方面表现较差；而数据驱动模型虽能捕捉丰富的相关性，却未能明确构建这两种变化来源的结构。为了弥合这一差距，我们提出了UniWind，这是一个将物理信息发电先验与数据驱动的状态感知校正相结合的风电功率预测框架。UniWind首先采用物理先验估计器，通过结合场地条件化的单调变换与共享物理功率模型，构建场地校准的物理先验……

    arXiv:2607.01670v2 Announce Type: replace  Abstract: Accurate day-ahead wind power forecasting is essential for cost-effective power-system operation. Wind power generation is primarily driven by future meteorological conditions while retaining temporal dependencies in power generation. In practice, observed wind-farm power reflects both meteorology-driven generation potential and deviations associated with local environmental and operational factors. Existing physical models provide useful constraints but adapt poorly across wind farms, whereas data-driven models can capture rich correlations but do not explicitly structure these two sources of variation. To bridge this gap, we propose UniWind, a wind power forecasting framework that integrates physics-informed generation priors with data-driven state-aware corrections. UniWind first employs a Physical Prior Estimator to construct a site-calibrated physical prior by combining a site-conditioned monotone warp with a shared physical pow
    
[^371]: 通过学习型随机停止方法稳定循环Transformer的外推能力

    Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping

    [https://arxiv.org/abs/2606.29983](https://arxiv.org/abs/2606.29983)

    该论文揭示了循环Transformer长度外推脆弱性的根源在于序列长度与循环次数之间的虚假相关性，并提出通过在训练中引入随机循环次数（包括学习型的RL-Halting调度）来显著降低分布外方差、稳定模型预测。

    

    循环Transformer通过重复应用共享的Transformer块，在架构上天然适合可变长度的算法任务。尽管它们可以在超出训练序列长度时表现出很强的长度泛化能力，但这种行为是脆弱的，会产生较高的分布外（OOD）方差，即使在表现良好的分布内解决方案之间也是如此。我们将这种方差追溯到简单算法任务中序列长度与循环次数之间的虚假相关性。在训练过程中向循环次数引入随机性可以显著降低OOD方差，并使预测在各种推理时循环次数下保持稳定。为了改进启发式随机化方案，我们进一步将RL-Halting作为一种学习型随机调度进行分析，发现它通常能改善准确性与稳定性之间的权衡。在二进制加法、Dyck-1、Unique Set和Copy等任务上，学习型随机停止通常能改善这一权衡……

    arXiv:2606.29983v2 Announce Type: replace-cross  Abstract: Looped Transformers, which repeatedly apply a shared transformer block, are an architecturally natural fit for variable-length algorithmic tasks. Although they can exhibit strong length generalization beyond the length of training sequences, this behavior is brittle, yielding high out-of-distribution (OOD) variance, even across well-performing in-distribution solutions. We trace this variance to the spurious correlation in simple algorithmic tasks between sequence length and number of loops. Introducing stochasticity into the number of loops during training sharply reduces OOD variance and stabilizes predictions across inference-time loop counts. To improve upon heuristic randomization schemes, we further analyze RL-Halting as a learned stochastic schedule and find that it generally improves the accuracy-stability trade-off. Across binary addition, Dyck-1, Unique Set, and Copy, learned stochastic stopping often improves this tr
    
[^372]: 符号机制数据归因：追踪训练影响至习得的行为策略

    Symbolic Mechanistic Data Attribution: Tracing Training Influence to Learned Behavioral Policies

    [https://arxiv.org/abs/2606.29171](https://arxiv.org/abs/2606.29171)

    提出符号机制数据归因（SMDA）框架，通过在稀疏自编码器特征上拟合可解释的符号策略，并将每个微调样本的影响分解为特征激活（ΔX）与输出概率（ΔY）两条通路，从而将训练数据归因到模型习得的行为策略，并借此揭示基础模型安全行为中的系统性缺陷（如宗教刻板印象）。

    

    虽然现有的数据归因方法能够识别哪些训练样本构建了特定的机制电路，但它们无法解释训练数据如何塑造模型习得的行为决策。我们提出了符号机制数据归因（Symbolic Mechanistic Data Attribution, SMDA），这是一个将训练数据对归因到控制模型行为的可解释符号策略的框架。SMDA 在稀疏自编码器（SAE）特征上拟合闭式岭回归来对某一行为建模，然后分解每个监督微调（SFT）样本如何通过特征激活（ΔX）和输出概率（ΔY）两条通路改变该策略。我们在 Llama-3.2-3B-Instruct 中提炼了拒绝行为的符号策略，并分析了 200 个 SFT 数据对。我们的分析揭示：(1) 符号策略的系数暴露了基础模型安全行为中的系统性缺陷，例如宗教刻板印象；(2) 逐特征的 ΔX/ΔY 分解能够机制性地……（摘要原文在此处截断）

    arXiv:2606.29171v2 Announce Type: replace-cross  Abstract: While existing data attribution methods can identify which training examples build specific mechanistic circuits, they cannot explain how training data shapes the behavioral decisions a model learns. We introduce Symbolic Mechanistic Data Attribution (SMDA), a framework that attributes training pairs to the interpretable symbolic policies governing model behavior. SMDA fits a closed-form Ridge regression over sparse autoencoder (SAE) features to model a behavior, then decomposes how each supervised fine-tuning (SFT) example shifts that policy through feature-activation ($\Delta X$) and output-probability ($\Delta Y$) pathways. We distill a symbolic policy for refusal in Llama-3.2-3B-Instruct and analyze 200 SFT pairs. Our analysis reveals (1)~the symbolic policy's coefficients expose systematic gaps in the base model's safety behavior, such as religious stereotyping; (2)~per-feature $\Delta X/\Delta Y$ decomposition mechanistic
    
[^373]: 多指标模型下经验风险最小化中的副本对称破缺与算法阈值

    Replica Symmetry Breaking and Algorithmic Thresholds in Empirical Risk Minimization under Multi-Index Model

    [https://arxiv.org/abs/2606.28573](https://arxiv.org/abs/2606.28573)

    该论文在多指标模型的监督学习设定下提出增量近似消息传递（IAMP）算法，精确刻画了经验风险地形中多项式时间算法可到达的区域，并揭示了副本对称破缺现象与算法阈值。

    

    现代机器学习模型通过优化高维非凸的经验风险函数进行训练。这类代价函数可能存在大量的局部最优解，然而基于梯度的优化方法却似乎能够收敛到接近全局最优的解。在一个简单的监督学习设定中，我们构建了一幅精确的图景，刻画经验风险地形中哪些部分是多项式时间算法可以到达的。我们给定独立同分布的数据对 {(𝒙ᵢ, yᵢ): 1 ≤ i ≤ n}，其中 𝒙ᵢ ∈ ℝᵈ 为标准高斯特征向量，yᵢ ∈ ℝ 为响应变量，其通过 𝒙ᵢ 在一个未知的 k 维子空间上的投影而依赖于 𝒙ᵢ。我们使用经验风险最小化来学习一个依赖于数据的 m 维投影的模型（例如，具有 m 个神经元的神经网络）。我们提出了一种增量近似消息传递算法，并精确刻画……（原文摘要不完整，此处为已有部分的翻译）

    arXiv:2606.28573v2 Announce Type: replace  Abstract: Modern machine learning models are trained by optimizing high-dimensional non-convex empirical risk functions. Such cost functions can have a multitude of local optima and yet, gradient-based optimization appears to converge to near-global optima.   Within a simple supervised learning setting, we develop a precise picture of which parts of the empirical risk landscape are accessible by polynomial-time algorithms. We are given i.i.d. pairs $\{(\boldsymbol{x}_i,y_i):\; 1 \le i\le n\}$ with $\boldsymbol{x}_i\in \mathbb{R}^d$ standard Gaussian feature vectors, and $y_i\in\mathbb{R}$ response variables that depend on $\boldsymbol{x}_i$ through their projections on an unknown $k$-dimensional subspace. We use empirical risk minimization to learn a model that depends on an $m$-dimensional projection of the data (e.g., an $m$-neurons neural network).   We propose an incremental approximate message passing (IAMP) algorithm and precisely charac
    
[^374]: 随机线性上下文赌博机的集合成员估计

    Set Membership Estimation for Stochastic Linear Contextual Bandits

    [https://arxiv.org/abs/2606.20022](https://arxiv.org/abs/2606.20022)

    本文首次将控制与系统辨识领域成熟的集合成员估计（SME）方法引入线性上下文赌博机，提出基于SME的OFU算法并建立了次线性遗憾界。

    

    上下文赌博机学习由于能够在时变环境中融合辅助/上下文信息，在工程应用中受到越来越多的关注。上下文赌博机中一种广泛使用的方法是不确定性下的乐观主义（OFU），该方法为未知参数构建不确定性集合，并在这些集合上进行乐观决策。在赌博机文献中，不确定性集合通常借助统计学习工具（如置信区间）来推导。相比之下，集合成员估计（SME）——一种在控制与系统辨识领域已经成熟的量化不确定性的方法——在赌博机文献中却很少受到关注。本文以线性上下文赌博机为切入点，研究SME与上下文赌博机之间的联系。我们提出了一种基于SME的OFU算法，并在适当的条件下建立了次线性遗憾界。

    arXiv:2606.20022v2 Announce Type: replace-cross  Abstract: Contextual bandit learning has attracted increasing attention in engineering applications due to its ability to incorporate side/context information in time-varying environments. A widely used approach for contextual bandits is optimism in the face of uncertainty (OFU), in which uncertainty sets for unknown parameters are constructed and decisions are made optimistically over these sets. In the bandit literature, uncertainty sets are typically derived using statistical learning tools, such as confidence intervals. In contrast, set membership estimation (SME), a well-established uncertainty quantification approach in control and system identification, has received little attention in the bandit literature. This paper investigates the connection between SME and contextual bandits by focusing on linear contextual bandits as a starting point. We propose an SME-based OFU algorithm and establish a sublinear regret bound under proper 
    
[^375]: 具有选择性标签的算法前沿的识别与推断

    Identification and Inference for Algorithmic Frontiers with Selective Labels

    [https://arxiv.org/abs/2606.14977](https://arxiv.org/abs/2606.14977)

    本文在标签仅对被选择个体可观测（选择性标签）的情形下，刻画了公平性-准确性前沿的识别区域，并提出了去偏机器学习估计量及统计推断方法，用以检验现有算法是否位于前沿上，并构造在不损害公平性的前提下提升各群体准确性的改进算法。

    

    当结果（标签）仅对被选择的个体进行观测时，本文提供了刻画公平性-准确性（FA）前沿的识别结果，以及用于检验假设和构建FA前沿置信集的统计推断工具。当选择过程不受限制但损失以特定方式度量时，我们给出了FA前沿尖锐识别区域的刻画。当观测到的分布排除了某一二元现状算法位于前沿的可能性时，我们构建了一个单一的替代算法，该算法在所有可行的缺失标签分布上一致地提高了两个群体的准确性且不损害公平性。对于任意损失函数，在可观测变量选择和严格重叠假设下，我们获得了点识别结果，提出了去偏机器学习估计量，推导了其渐近分布，并展示了如何利用该估计量进行统计推断。

    arXiv:2606.14977v2 Announce Type: replace-cross  Abstract: This paper provides identification results to characterize a fairness-accuracy (FA) frontier, and statistical inference tools to test hypotheses and build a confidence set for the FA-frontier, when outcomes are observed only for selected individuals. When the selection process is unrestricted but loss is measured in specific ways, we provide a characterization of the sharp identification region of the FA-frontier. When the observed distribution rules out frontier membership for a binary status quo algorithm, we construct a single alternative algorithm that improves both groups' accuracy without worsening fairness, uniformly over all admissible distributions of missing labels. For any loss functions, under selection on observables and strict overlap assumptions, we obtain point identification and propose a debiased machine learning estimator, derive its asymptotic distribution, and show how this can be used to carry out inferenc
    
[^376]: LLM作为调查者：面向稳健交互式问题诊断的证据优先智能体推理

    LLM-as-an-Investigator: Evidence-First Agentic Reasoning for Robust Interactive Problem Diagnosis

    [https://arxiv.org/abs/2606.13220](https://arxiv.org/abs/2606.13220)

    该论文提出“LLM作为调查者”这一证据优先方法论，通过SIA智能体协调诊断推理、假设维护与反事实质疑三种互补推理行为，减少诊断中过早锁定错误假设的问题，实现更稳健的交互式问题诊断。

    

    大语言模型（LLM）被用于技术故障排查，其中问题描述往往不完整，且多种原因可能解释相同的症状。有效的诊断需要获取具有区分性的证据，并选择证据支持最强的解释。然而，仅靠推理或交互可能无法避免过早锁定错误的假设。我们提出了LLM-as-an-Investigator，这是一种面向交互式问题诊断的证据优先方法论，能够减少过早承诺，同时区分相互竞争的病因。我们通过解决方案调查者智能体（Solution Investigator Agent，SIA）来实现该方法，它通过共享的调查状态协调三种互补的推理行为：诊断推理器负责开展并重新评估诊断性访谈，假设调查者负责维护和更新相互竞争的假设，反事实调查者负责质疑当前的解释并指导……（摘要原文在此处截断）

    arXiv:2606.13220v2 Announce Type: replace  Abstract: Large language models (LLMs) are used for technical troubleshooting, where problem descriptions are often incomplete and multiple causes can explain the same symptoms. Effective diagnosis requires acquiring discriminative evidence and selecting the explanation best supported by it. Yet reasoning or interaction alone may not prevent premature commitment to an incorrect hypothesis. We introduce LLM-as-an-Investigator, an evidence-first methodology for interactive problem diagnosis that reduces premature commitment while distinguishing competing causes. We implement it through the Solution Investigator Agent (SIA), which orchestrates three complementary reasoning behaviours through a shared investigation state: a Diagnostic Reasoner that conducts and reassesses the diagnostic interview, a Hypothesis Investigator that maintains and updates competing hypotheses, and a Counterfactual Investigator that challenges current explanations and gu
    
[^377]: 让我们请教高斯：改进的单次运行隐私审计

    Let's Ask Gauss: Improved One-Run Privacy Auditing

    [https://arxiv.org/abs/2606.12733](https://arxiv.org/abs/2606.12733)

    该论文提出了一种基于中心极限定理的高斯分布视角的单次运行差分隐私审计框架，将金丝雀对齐信号视为随机变量序列而非二值化猜测，从而从单次训练中就能获得更紧的隐私下界。

    

    隐私审计通过估计模型实际泄露的信息量，提供了一项重要的安全保障，从而确保理论上的隐私保证在实践中真正成立。我们研究了差分隐私（DP）机器学习的经验性隐私审计，重点针对DP-SGD等机制的高效单次运行方法。以往的单次运行方法将训练样本或“金丝雀”阈值化为二元成员身份猜测，这丢弃了有用的信息。我们证明，在白盒DP-SGD设置中，与金丝雀对齐的信号自然形成一个随机变量序列，其归一化之和是渐近高斯的。我们还表明，这一高斯视角可以自然地扩展到其他满足经典、三角阵列或鞅中心极限定理的组合机制。利用这一分布视角，我们开发了一个差分隐私审计框架，能够仅通过单次训练就获得更紧的隐私下界。

    arXiv:2606.12733v3 Announce Type: replace  Abstract: Privacy auditing provides an important safeguard by estimating the actual information leaked by a model, thus ensuring that theoretical privacy guarantees hold in practice. We study empirical privacy auditing for differentially private (DP) machine learning, focusing on efficient one-run methods for mechanisms such as DP-SGD. Prior one-run approaches threshold training examples or "canaries" into binary membership guesses, which discards useful information. We show that, in the white-box DP-SGD setting, canary-aligned signals naturally form a sequence of random variables whose normalized sum is asymptotically Gaussian. We show that this gaussian perspective can also be naturally extended to other composition mechanisms that satisfy a classical, triangular-array, or martingale central limit theorem. Leveraging this distributional perspective, we develop a DP-auditing framework that leads to tighter privacy lower bounds from a single t
    
[^378]: 面向含激波流动数据同化的特征保持潜空间集合卡尔曼滤波器

    Feature-preserving Latent-EnKF for Data Assimilation of Flows with Shocks

    [https://arxiv.org/abs/2606.12559](https://arxiv.org/abs/2606.12559)

    本文提出一种特征保持的潜空间集合卡尔曼滤波器，通过坐标条件化自解码器在低维潜空间中执行集合更新，解决了传统EnKF在含激波等间断流动数据同化中因间断位置不确定性而失去物理一致性的问题。

    

    集合卡尔曼滤波器被广泛应用于顺序数据同化，但对于含间断的解（例如可压缩流动中的激波）则会失效。当集合中存在间断位置的不确定性时，EnKF的分析结果可能失去物理一致性，这可以从三个角度来理解：(i) 预报集合中的逐点双峰统计特性，(ii) 预报成员仿射叠加中出现的混合符号分析权重，以及 (iii) 空间分布的卡尔曼增益剖面在间断附近修改状态而非将其相干地平移。为了在分析状态中保持流动特征并由此维持物理一致性，本文提出了一种特征保持的潜空间集合卡尔曼滤波器。所提方法在由坐标条件化自解码器学习得到的低维潜空间中执行集合更新。该潜空间表示保持了近似高斯的集合统计特性……

    arXiv:2606.12559v2 Announce Type: replace-cross  Abstract: The ensemble Kalman filter (EnKF) is widely adopted for sequential data assimilation, but fails for solutions with discontinuities, such as shocks in compressible flows. For ensembles with uncertainty in discontinuity location, the EnKF analysis can lose physical consistency, which can be viewed through three lenses: (i) bimodal pointwise statistics in the forecast ensemble, (ii) mixed-sign analysis weights in the affine superposition of forecast members, and (iii) spatially distributed Kalman-gain profiles that modify the state around a discontinuity rather than translating it coherently. In order to preserve flow features in the analysis state and, as such, maintain physical consistency, a feature-preserving latent-EnKF is introduced. The proposed method performs the ensemble update in a low-dimensional latent space learned by a coordinate-conditioned auto-decoder. The latent representation retains near-Gaussian ensemble stat
    
[^379]: 神经关系程序：统一结构化数据上的查询与神经计算

    Neuro-Relational Programs: Unifying Queries and Neural Computation over Structured Data

    [https://arxiv.org/abs/2606.11946](https://arxiv.org/abs/2606.11946)

    本文提出神经关系程序（NRPs），一种在关系数据库事实上携带向量嵌入的声明式查询语言，通过扩展Datalog规则实现嵌入的组合、聚合与转换，将关系推理与可学习神经组件统一于同一形式体系中。

    

    arXiv:2606.11946v2 公告类型：replace-cross 摘要：在关系数据库上进行深度学习的传统方法是将神经模型（如图神经网络GNN）应用于数据库的图表示。近期的方法则直接在数据库上进行操作，将元组与嵌入向量相关联，并扩展查询机制以联合处理嵌入和关系内容。受这些发展的启发，我们提出了神经关系程序，这是一种面向关系数据库的声明式查询语言，其事实携带数值向量嵌入。NRPs扩展了Datalog风格的规则，增加了组合、聚合和转换嵌入的操作，从而在单一形式体系中交织了关系推理和可学习的神经组件。这为在关系数据上进行神经计算提供了一种通用方法：一个NRP既可以被解读为带有可训练组件的查询计划，也可以被解读为内置关系结构的神经架构。

    arXiv:2606.11946v2 Announce Type: replace-cross  Abstract: The conventional approach to deep learning over relational databases applies neural models, such as Graph Neural Networks (GNNs), to a graph representation of the database. Recent approaches instead operate on databases directly, associating tuples with embeddings and extending query mechanisms to jointly process embeddings and relational content. Inspired by these developments, we introduce Neuro-Relational Programs (NRPs), a declarative query language for relational databases whose facts carry numeric vector embeddings. NRPs extend Datalog-style rules with operations that combine, aggregate, and transform embeddings, thereby interleaving relational reasoning and learnable neural components within a single formalism. This yields a general approach to neural computation over relational data: an NRP can be read both as a query plan with trainable components and as a neural architecture with relational structure built in.   Natur
    
[^380]: 何时对齐，何时预测：多模态学习的相图

    When to Align, When to Predict: A Phase Diagram for Multimodal Learning

    [https://arxiv.org/abs/2606.11190](https://arxiv.org/abs/2606.11190)

    本文通过带结构化跨模态噪声相关的尖峰信号加噪声模型，揭示了跨模态对齐与跨模态预测两种范式的互补失败模式，并构建相图为多模态学习中范式的选择提供了理论指导。

    

    跨模态对齐（CA）和跨模态预测（CP）是多模态表征学习的两大主流范式，然而对于两者何时成功、何时失败，目前缺乏系统性的理解——这一空白使得实践者，尤其是在拥有异构仪器和多层次测量的科学领域中的从业者，无法诊断为什么标准方法的表现不如最佳单一模态。我们在具有结构化跨模态噪声相关性的尖峰信号加噪声模型下研究这两种目标，这种噪声相关性正是打破经典恢复保证的关键因素，并推导出了揭示互补失败模式的分离比率：对齐方法会对每个模态进行白化处理，当噪声在视图间强相关时会失败；预测方法通过单侧白化编码所有可跨模态预测的信息，其恢复效果由源模态的质量决定。由此得到的相图将多模态问题划分为不同区域。

    arXiv:2606.11190v3 Announce Type: replace  Abstract: Cross-modal alignment (CA) and cross-modal prediction (CP) are the dominant paradigms for multimodal representation learning, yet there is no systematic understanding of when each succeeds and when each fails --- a gap that leaves practitioners, especially in scientific domains with heterogeneous instruments and multiple levels of measurement, unable to diagnose why standard methods underperform the best single modality. We study both objectives under a spiked signal-plus-noise model with structured cross-modal nuisance correlation, the ingredient that breaks the classical recovery guarantees, and derive separation ratios that expose complementary failure modes: alignment whitens each modality and fails when nuisance is strongly correlated across views; prediction encodes whatever is cross-predictable through a one-sided whitening, with recovery governed by source-modality quality. The resulting phase diagram partitions multimodal pr
    
[^381]: 基于玻尔兹曼边际的kNN分类近指数收敛速率

    Near-Exponential Convergence Rates for kNN Classification based on Boltzmann Margin

    [https://arxiv.org/abs/2606.10361](https://arxiv.org/abs/2606.10361)

    本文提出了一种介于Tsybakov边际与Massart边际之间的新条件——玻尔兹曼边际，并据此首次为kNN分类建立了近指数收敛速率。

    

    分类器的收敛速率分析通常在Tsybakov边际或Massart边际条件下进行。前者是一个相对较弱的条件，通常只能得到多项式速率，而后者则强得多，但可以保证指数速率。在本文中，我们引入了一个称为玻尔兹曼边际的新条件，它弥合了这两种情形之间的差距。该条件弱于Massart边际，通常强于Tsybakov边际，并且在适当条件下可以蕴含二者的许多性质。我们将玻尔兹曼边际应用于kNN分类器的分析，首次建立了kNN分类的近指数收敛速率。我们还给出了主要结果的若干扩展，并提供了支持主要理论结论的数值证据。

    arXiv:2606.10361v2 Announce Type: replace-cross  Abstract: Convergence-rate analysis for classifiers is often conducted under either Tsybakov margin or Massart margin. The former is a relatively weak condition that typically yields polynomial rates, while the latter is substantially stronger but can guarantee exponential rates. In this paper, we introduce a new condition, called Boltzmann margin, that bridges the gap between these two regimes. It is weaker than Massart margin, generally stronger than Tsybakov margin, and can imply many of their properties under suitable conditions. We apply Boltzmann margin to the analysis of kNN classifiers and establish the first near-exponential convergence rates for kNN classification. We also present extensions of the main results and provide numerical evidence supporting the main theoretical implications.
    
[^382]: 脑源成像中稀疏贝叶斯推断的结构保持校正学习

    Structure-Preserving Correction Learning for Sparse Bayesian Inference in Brain Source Imaging

    [https://arxiv.org/abs/2606.07196](https://arxiv.org/abs/2606.07196)

    该论文提出一种结构保持的校正学习框架，将经典的稀疏Type-II贝叶斯联合超参数求解器展开为可训练的神经网络，通过学习结构化的残差校正项以数据驱动的方式改进M/EEG脑源成像的推断性能，同时保留贝叶斯方法的可解释性。

    

    经典的稀疏Type-II贝叶斯方法用于M/EEG脑成像，支持对源和噪声超参数的联合估计，但其依赖于固定的迭代更新规则。尽管这些更新规则具有原则性且可解释，但其动态过程并非数据驱动。我们提出通过将经典的联合超参数学习求解器展开为可训练的神经网络架构来学习更新机制本身，同时保持底层的贝叶斯结构，其中网络的各层对应于原始的迭代过程。该框架在训练前被初始化以精确恢复经典求解器的行为，并通过渐进式增强的残差校正学习机制加以丰富，涵盖从可学习偏差、自适应多层感知机到基于注意力的上下文精化等多种形式。通过这种方式，训练过程并非用黑盒预测器取代贝叶斯推断，而是学习结构化的校正项……

    arXiv:2606.07196v2 Announce Type: replace  Abstract: Classical sparse Type-II Bayesian methods for M/EEG brain imaging support joint estimation of source and noise hyperparameters, but rely on fixed iterative update rules. Although these updates are principled and interpretable, their dynamics are not data-driven. We propose to learn the update mechanism itself while preserving the underlying Bayesian structure by unfolding a classical joint hyperparameter-learning solver into a trainable neural network architecture whose layers mirror the original iterations. The resulting framework is initialized to recover the classical solver exactly before training and is enriched through progressively more expressive residual correction learning mechanisms, ranging from learnable biases to adaptive multilayer perceptron and attention-based contextual refinements. In this way, training does not replace Bayesian inference with a black-box predictor, but instead learns structured correction terms wh
    
[^383]: 重新思考高稀疏度大语言模型剪枝的校准：从单一来源到能力覆盖

    Rethinking Calibration for High-Sparsity LLM Pruning: From Single Sources to Capability Coverage

    [https://arxiv.org/abs/2606.03328](https://arxiv.org/abs/2606.03328)

    该论文将大语言模型剪枝的校准问题重新定义为能力覆盖问题，证明固定预算下均衡混合多种来源的校准数据（Balanced）显著优于任何单一来源，且稀疏度越高优势越大（60%稀疏度下比C4高18.8分），并在无法获得对齐语料时提出IGSP作为近似方案。

    

    训练后大语言模型剪枝依赖于一个小型校准集，通常通过启发式方法选择，并以WikiText困惑度或通用领域常识基准的平均准确率作为评判标准。这种单一维度的视角掩盖了能力特定的效应：在60%稀疏度下，平均常识准确率仅相差2.85分的不同策略，在代码保留能力上竟相差高达51.9分；而且在15个数据来源中，没有任何单一来源能够保留所有能力。本文将校准视为一个能力覆盖问题。Balanced是在固定预算下的四路均匀混合策略，它在真实语料上验证了这一原则：其性能优于所有被评估的单一来源，且相对于C4的优势从30%稀疏度时的+4.6分增长到60%稀疏度时的+18.8分。这一优势在不同模型规模和第二个模型家族上均成立，并能迁移到与校准集构建不相交的保留基准上。当无法获得对齐语料时，IGSP可以从（摘要截断）……近似该混合策略。

    arXiv:2606.03328v4 Announce Type: replace-cross  Abstract: Post-training LLM pruning relies on a small calibration set, usually chosen heuristically and judged by WikiText perplexity or by mean accuracy on general-domain commonsense benchmarks. This single-axis view hides capability-specific effects: at 60\% sparsity, strategies separated by $2.85$ points in averaged commonsense accuracy diverge by $51.9$ points in code retention, and across 15 sources no single one preserves all capabilities. We treat calibration as a capability-coverage problem. Balanced, a uniform four-way mixture under a fixed budget, tests the principle on real corpora: it outperforms every evaluated single source, and its margin over C4 grows from $+4.6$ points at 30\% sparsity to $+18.8$ at 60\%. The advantage holds across model scale and a second model family, and transfers to held-out benchmarks disjoint from calibration construction. When aligned corpora are unavailable, IGSP approximates that mixture from th
    
[^384]: 具有内生改进的线性策略性分类

    Linear Strategic Classification with Endogenous Improvements

    [https://arxiv.org/abs/2606.01198](https://arxiv.org/abs/2606.01198)

    该论文研究了策略性分类的改进感知变体，证明在单指标资格模型和线性可分解代价下，智能体的最优响应可简化为决策边界的闭式偏移，该问题具有线性样本复杂度的可学习性，并提出了一个用估计代替预言机的实用插入式算法。

    

    经典的策略性分类模型将智能体对已部署分类器的响应视为表面行为：特征会改变，但真实标签不会改变。我们研究了一种改进感知的变体，其中策略性响应可以真正改变结果：智能体以一定代价策略性地选择部署后的特征，随后标签从一个固定的条件结果律中抽取。我们在单指标资格模型和线性可分解代价的假设下，为线性分类器形式化了这一问题。在这些假设下，智能体的最优响应可以简化为决策边界的闭式偏移，由此产生的改进感知分类问题可以以线性样本复杂度进行学习。由于部署后的标签无法先验获得，我们首先在可以访问结果律预言机的条件下建立了泛化界。我们的主要贡献之一是一个实用的插入式算法，该算法用估计值代替预言机，以及……（摘要在此处截断）

    arXiv:2606.01198v3 Announce Type: replace  Abstract: Classical models of strategic classification treat agents' responses to a deployed classifier as cosmetic: features change, but true labels do not. We study an improvement-aware variant in which strategic responses can change outcomes: agents choose post-deployment features strategically at a cost, and labels are then drawn from a fixed conditional outcome law. We formalize this problem for linear classifiers under a single-index qualification model and linear-decomposable costs. Under these assumptions, agents' best responses reduce to a closed-form shift of the decision boundary, and the resulting improvement-aware classification problem is learnable with linear sample complexity. As post-deployment labels are unavailable a priori, we begin by establishing a generalization bound with oracle access to the outcome law. One of our main contributions is a practical plug-in algorithm that replaces the oracle with an estimate, together w
    
[^385]: LLMSurgeon：诊断大语言模型的数据混合

    LLMSurgeon: Diagnosing Data Mixture of Large Language Models

    [https://arxiv.org/abs/2605.30348](https://arxiv.org/abs/2605.30348)

    本文提出LLMSurgeon框架，仅通过分析大语言模型生成的文本，将预训练数据混合估计建模为标签偏移假设下的带约束逆问题，利用校准的软混淆矩阵纠正系统性领域混淆，从而诊断出模型预训练语料的领域级分布。

    

    大语言模型（LLMs）的预训练数据混合构成了它们的“数字DNA”，塑造着模型的行为、能力和失败模式。然而这种构成很少被公开披露，使得对数据组合或来源的事后审计变得困难。在这项工作中，我们形式化了数据混合手术（Data Mixture Surgery, DMS）：仅给定目标LLM生成的文本，在预定义的分类体系下估计其预训练语料库的领域级分布。我们提出了LLMSurgeon，一个将DMS建模为标签偏移假设下逆问题的强大框架。LLMSurgeon并非直接聚合分类器的输出，而是估计一个经过校准的软混淆矩阵，并求解一个带约束的逆问题，以纠正系统性的领域混淆并恢复潜在的混合先验分布。为进行评估，我们引入了LLMScan，一个基于开源LLM构建的、可验证配方的评估套件……

    arXiv:2605.30348v2 Announce Type: replace-cross  Abstract: The pretraining data mixture of Large Language Models (LLMs) constitutes their "digital DNA", shaping model behaviors, capabilities, and failure modes. Yet this composition is rarely disclosed, making post-hoc auditing of data combination or provenance difficult. In this work, we formalize Data Mixture Surgery (DMS): given only generated text from a target LLM, estimate the domain-level distribution of its pretraining corpus under a predefined taxonomy. We propose $\textbf{{LLMSurgeon}}$, a strong framework that casts DMS as an inverse problem under the label-shift assumption. Rather than directly aggregating classifier outputs, LLMSurgeon estimates a calibrated $\textit{soft}$ confusion matrix and solves a constrained inverse problem to correct systematic domain confusion and recover the latent mixture prior. To evaluate, we introduce $\textbf{{LLMScan}}$, a recipe-verifiable evaluation suite built from open-source LLMs with t
    
[^386]: GRASP：面向半结构化知识库的计划引导图检索与自适应融合重排序方法

    GRASP: Plan-Guided Graph Retrieval with Adaptive Fusion and Reranking on Semi-Structured Knowledge Bases

    [https://arxiv.org/abs/2605.30237](https://arxiv.org/abs/2605.30237)

    GRASP通过小型LLM规划器生成可执行图计划、融合图排序与文本排序，并用强化学习优化规划器以最大化融合排序增益，在半结构化知识库检索上取得全面领先的性能，平均Hit@1比最强基线提升18.6个百分点且单查询仅需2.47秒。

    

    在半结构化知识库（SKB）上进行检索时，必须同时处理文本描述和关系约束。基于文本的检索器能够捕捉查询的整体语义，但难以处理多跳关系；而基于图的检索器可以精确地沿着关系路径检索，却容易忽略更广泛的整体含义。我们提出了GRASP，一种学习融合两者的混合检索器。一个小型LLM规划器通过单次调用将查询转化为可执行的图计划，其图排序结果与文本排序结果进行融合，再由微调过的重排序器对融合后的候选结果进行重新排序。不同于单独训练规划器使其自身预测准确，我们采用强化学习对规划器进行优化，以最大化其图排序为最终融合排序带来的增益。在三个STaRK基准测试上，GRASP在所有指标上都取得了最佳结果，相比最强基线平均Hit@1提升了18.6个百分点，同时每个查询仅需2.47秒。

    arXiv:2605.30237v3 Announce Type: replace-cross  Abstract: Retrieval over semi-structured knowledge bases (SKBs) must handle both textual descriptions and relational constraints. Text-based retrievers capture the overall semantics of a query but struggle with multi-hop relations, whereas graph-based retrievers follow relational paths precisely but miss the broader meaning. We propose GRASP, a hybrid retriever that learns to fuse the two. A small LLM planner translates the query into an executable graph plan in a single call, its graph ranking is fused with a text ranking, and a fine-tuned reranker reorders the fused candidates. Instead of training the planner to be accurate on its own, we optimize it with reinforcement learning to maximize the gain its graph ranking brings to the fused ranking. On three STaRK benchmarks, GRASP achieves the best results on all metrics, improving the average Hit@1 by 18.6 points over the strongest baseline, while taking only 2.47 seconds per query.
    
[^387]: 无干扰的可检测性：面向单次运行隐私审计的高效金丝雀样本设计

    Detectability without Interference: Efficient Canary Design for Privacy Auditing in One Run

    [https://arxiv.org/abs/2605.27292](https://arxiv.org/abs/2605.27292)

    本文提出IBIS框架，通过基于影响的贪心初始化与双层优化设计相互干扰最小、检测分数最优的金丝雀样本，使隐私审计仅需一次训练运行即可高效完成。

    

    隐私审计旨在利用成员推理攻击实证地评估机器学习模型中的隐私泄露程度，并据此推导差分隐私（DP）参数的下界。标准的隐私审计方法会将一个敏感样本（即金丝雀样本）插入到许多次（远超100次）随机化的训练运行中，并要求审计者检测该样本是否被包含在每次运行中。单次运行审计则在一次训练运行中插入大量金丝雀样本，从而避免了多次运行方法的高昂计算成本。然而，近期的理论研究表明，金丝雀样本之间可能相互干扰，导致单个金丝雀样本的成员推理攻击分数降低，进而削弱审计能力。我们提出了IBIS框架，该框架通过优化金丝雀样本来最大化其自身的分数，同时最小化其对其他样本的影响。IBIS将基于影响的贪心初始化与双层优化相结合，并引入了一种新颖的正则化项，鼓励金丝雀样本在嵌入空间中相互正交……（摘要原文在此处截断）

    arXiv:2605.27292v3 Announce Type: replace  Abstract: Privacy auditing aims to empirically assess privacy leakage in machine learning models using membership inference attacks (MIAs), and to derive lower bounds on differential privacy (DP) parameters. Standard privacy auditing inserts a sensitive sample (a canary) into many ($\gg 100$) randomized training runs, requiring the auditor to detect whether it was included in each run. In one-run auditing, many canaries are included in a single training run, avoiding the high computational cost of multi-run approaches. However, recent theory shows that canaries can interfere with one another, reducing per-canary MIA scores and thereby weakening the audit power. We propose IBIS, a framework that optimizes canaries to maximize their own score while minimizing their impact on others. IBIS combines influence-based greedy initialization with bilevel optimization, introducing a novel regularization term that encourages canaries to be orthogonal in e
    
[^388]: 高维处理效应的监督学习目标

    Supervised Learning Objectives for High-Dimensional Treatment Effects

    [https://arxiv.org/abs/2605.27281](https://arxiv.org/abs/2605.27281)

    该论文将因果推断重新表述为监督学习问题，证明了因果误差可分解为递增阶数的矩平衡误差，并据此设计出能直接改善高维处理效应（如文本干预）因果估计的学习目标，同时可将效应投影到更粗粒度的属性上。

    

    预测具有多种可能变化形式的干预效应（例如，影响心理健康结果的治疗内容，或驱动股价波动的财报电话会议记录文本）在多个领域都非常有用。然而，经典的因果估计方法通常假设所有可能的干预都被观测到，这在干预变化范围极广时是不可行的，例如在所有文本字符串构成的空间中。我们对一种广为人知的方法进行了改进，即将因果推断重新表述为学习问题，以应对此类场景，并假设所学习的函数可实现且不存在未观测的混杂因素。我们证明了因果误差可以分解为阶数递增的矩平衡误差，并利用条件结果在混杂因素中的结构来设计能够直接改善因果估计的学习目标。我们还展示了如何将高维处理的效应投影到更粗粒度的属性上，从而……

    arXiv:2605.27281v2 Announce Type: replace  Abstract: Predicting the effect of interventions with many possible variations, e.g., therapeutic content that affects mental health outcomes or an earnings call transcript that drives movement in share price, is useful across several domains. However, classical causal estimators tend to assume that all possible interventions are observed, which is infeasible when interventions vary widely, for instance, in the space of all text strings. We adapt a well-known approach of recasting causal inference as a learning problem to address such settings, assuming realizability of the learned function and no unobserved confounding. We show that causal error decomposes into moment-balancing errors of increasing order and exploit structure of the conditional outcome in the confounders to design objectives that directly improve causal estimation. We also show how to project the effect of a high-dimensional treatment onto coarser attributes, which allows a s
    
[^389]: 用于后验推断的摊销因子推断网络

    Amortized Factor Inference Networks for Posterior Inference

    [https://arxiv.org/abs/2605.26419](https://arxiv.org/abs/2605.26419)

    本文提出摊销因子推断网络（AFIN），将模型表示为类型化的先验与似然因子列表，并通过参数不依赖潜变量维度和观测数量的编码-合并-解码网络，实现单一网络跨不同先验、似然、潜变量维度和观测数量的通用后验推断。

    

    摊销推断有望实现快速的测试时贝叶斯推断，但现有方法通常针对固定模型或受限的模型族。将摊销推断扩展到新的模型族往往需要重新训练或测试时微调。在本文中，我们提出一个问题：是否有可能构建一个单一的推断网络，使其能够跨不同的先验、似然、潜变量维度和观测数量进行泛化？我们提出了摊销因子推断网络，它将模型描述为一组类型化的先验因子和似然因子，并通过一个编码-合并-解码网络将该描述与观测数据一起映射为高斯后验近似，其学习到的参数不依赖于潜变量维度或观测数量。在合成数据和真实数据基准上的实验表明，单个训练好的AFIN能够以显著更低的测试时成本提供具有竞争力的后验近似。

    arXiv:2605.26419v2 Announce Type: replace  Abstract: Amortized inference promises fast test-time Bayesian inference, but existing methods typically target a fixed model or a restricted model family. Extending amortization to new model families often requires retraining or test-time fine-tuning. In this paper, we ask: is it possible to build a single inference network capable of generalizing across varying priors, likelihoods, latent dimensions, and observation counts? We introduce Amortized Factor Inference Networks (AFINs), which describe a model as a list of typed prior and likelihood factors and map this description, together with the observations, to a Gaussian posterior approximation through an encode--merge--decode network whose learned parameters do not depend on the latent dimension or the number of observations. Experiments on synthetic and real-data benchmarks show that a single trained AFIN provides competitive posterior approximations at substantially lower test-time cost t
    
[^390]: ChaosBench-Logic v2：大规模评估大语言模型在动力系统上的逻辑推理能力

    ChaosBench-Logic v2: Evaluating LLM Logical Reasoning over Dynamical Systems at Scale

    [https://arxiv.org/abs/2605.24305](https://arxiv.org/abs/2605.24305)

    该论文提出了大规模动力系统逻辑推理基准ChaosBench-Logic v2和CARE评估协议，揭示了即使是最先进的LLM在状态转变推理上仍接近随机水平，并存在先验崩溃与释义不一致等系统性缺陷。

    

    arXiv:2605.24305v2 公告类型： replace-cross 摘要：二元推理基准上的标准准确率会掩盖关键的失败模式：先验崩溃、释义下的不一致性，以及无法对参数依赖的动力系统进行推理。我们提出了ChaosBench-Logic v2，一个包含40,886道题目、覆盖165个动力系统、含27个一阶逻辑（FOL）谓词和78条公理边的基准，并同时提出CARE（校准与对抗鲁棒性评估）协议，该协议能够揭示上述病理现象。对14个模型的评估发现，即使是最前沿的模型，其状态转变推理仍接近随机水平（MCC = 0.05），而在给定前提下的FOL演绎推理可达到MCC = 0.52。按系统族的分解分析显示，专有模型的优势集中在跨指标（+0.40）和一致性任务上，而开源的Qwen 2.5-32B则在指标诊断任务上占据主导地位（0.91 vs. 0.45）。两个模型在分岔问题上表现出负MCC，并通过混淆矩阵被确认为系统性反相关。

    arXiv:2605.24305v2 Announce Type: replace-cross  Abstract: Standard accuracy on binary reasoning benchmarks hides critical failure modes: prior collapse, inconsistency under paraphrase, and inability to reason about parameter-dependent dynamics. We present ChaosBench-Logic v2, a 40,886-question benchmark over 165 dynamical systems with 27 FOL predicates and 78 axiom edges, together with CARE (Calibration- and Adversarial-Robust Evaluation), a protocol that surfaces these pathologies. Evaluating 14 models, we find that regime-transition reasoning remains near random (MCC = 0.05) even for frontier models, whereas FOL deduction with given premises reaches MCC = 0.52. Per-family decomposition shows that the proprietary-model advantage concentrates on cross-indicator (+0.40) and consistency tasks, while open-source Qwen 2.5-32B dominates indicator diagnostics (0.91 vs. 0.45). Two models exhibit negative MCC on bifurcation questions, confirmed as systematic anti-correlation via confusion-mat
    
[^391]: 面向LLM引导程序发现的进化多任务优化

    Evolutionary Multi-Task Optimization for LLM-Guided Program Discovery

    [https://arxiv.org/abs/2605.22613](https://arxiv.org/abs/2605.22613)

    提出EMO-STA两阶段框架，先跨任务族演化出共享程序档案库，再将共享候选程序适配到各目标任务，在八个任务族上以相同的进化迭代预算超越了单任务进化方法。

    

    近年来的LLM引导进化搜索方法表明，迭代式程序变异能够发现强大的算法，但这些方法通常对每个任务独立优化，即使相关任务之间存在可复用的结构。我们研究了面向LLM引导程序发现的进化多任务优化（EMO），并提出了EMO-STA（Shared-Then-Adapt，先共享后适配），这是一个两阶段框架：首先在整个任务族上演化出一个可执行程序的共享档案库，然后将选定的共享候选程序适配到每个目标任务。在EMO-STA框架内，我们探索了多种适配策略，包括从共享档案库热启动、适配平均表现最好的共享程序，以及适配在目标任务上表现最好的共享程序。在涵盖连续优化、几何构造、建模和算法优化的八个任务族上，在相同的进化迭代预算下，EMO-STA的表现优于单任务进化方法……

    arXiv:2605.22613v2 Announce Type: replace  Abstract: Recent LLM-guided evolutionary search methods have shown that iterative program mutation can discover strong algorithms, but they typically optimize each task independently, even when related tasks share reusable structure. We study evolutionary multitask optimization (EMO) for LLM-guided program discovery and propose EMO-STA (Shared-Then-Adapt), a two-stage framework that first evolves a shared archive of executable programs across a task family and then adapts selected shared candidates to each target task. Within EMO-STA, we explore multiple adaptation strategies, including warm-starting from the shared archive, adapting the best average shared program, and adapting the shared program that performs best on each target task. Across eight task families spanning continuous optimization, geometric construction, modeling, and algorithmic optimization, EMO-STA improves over single-task evolution under matched evolutionary-iteration budg
    
[^392]: 弥合差距：将朗读文本转换为会话对话

    Bridging the Gap: Converting Read Text to Conversational Dialogue

    [https://arxiv.org/abs/2605.18001](https://arxiv.org/abs/2605.18001)

    本文提出PACC方法，利用深度神经网络和HiFi-GAN对韵律特征进行分析与调整，将朗读语音转换为自然流畅的会话语音，同时兼顾实时应用的自然度、可懂度与低计算开销。

    

    arXiv:2605.18001v1 公告类型：cross 摘要：在语音处理的最新进展中，将朗读语音转换为会话语音受到了广泛关注。该领域的主要挑战在于在保持自然度和可懂度的同时，尽量减少实时应用的计算开销。传统的朗读语音往往缺乏自然会话交互所必需的细微韵律变化，这给虚拟助手、客户服务和语言学习工具等应用带来了挑战。本文提出了一种新颖的方法——基于会话语境的韵律调整（Prosodic Adjustment with Conversational Context，PACC），旨在将朗读语音转换为各种现代应用中使用的自然会话语音。PACC利用先进的深度神经网络来分析和修改语调、重音和节奏等韵律特征。与传统方法不同，我们的方法采用高保真生成对抗网络（HiFi-GAN）进行语音合成……

    arXiv:2605.18001v1 Announce Type: cross  Abstract: In recent advancements within speech processing, converting read speech to conversational speech has gained significant attention. The primary challenge in this domain is maintaining naturalness and intelligibility while minimizing computational overhead for real-time applications. Traditional read speech often lacks the nuanced prosodic variation essential for natural conversational interactions, posing challenges for applications in virtual assistants, customer service, and language learning tools. This paper introduces a novel approach, Prosodic Adjustment with Conversational Context (PACC), aimed at converting read speech into natural conversational speech used in various modern applications. PACC utilizes advanced deep neural networks to analyze and modify prosodic features such as intonation, stress, and rhythm. Unlike conventional methods, our approach uses High-Fidelity Generative Adversarial Networks (HiFi-GAN) for speech synt
    
[^393]: 告别枯燥的神经数据：面向数据复用的智能体AI基准测试

    Neurodata Without Boredom: Benchmarking Agentic AI for Data Reuse

    [https://arxiv.org/abs/2605.12808](https://arxiv.org/abs/2605.12808)

    该论文提出了首个针对编程智能体在神经科学数据复用任务上的基准测试，通过让AI智能体阅读论文、代码和数据，并将大规模小鼠神经记录数据重新格式化以训练下游解码器，评估智能体AI在减轻神经科学数据整理负担方面的能力。

    

    神经科学数据分散在不同的实验室、实验范式和存储格式中，数据复用往往需要大量的人工操作。为应对这一问题，研究者们提出了标准化格式，但一个根本性矛盾依然存在：足够灵活以适应多样实验的格式往往难以自解释，而完全描述性的数据集则需要文档编制和数据整理工作，很少有实验室能够长期维持。编程智能体能够快速阅读代码和文本，并对人类容易略过的细节表现出明显的关注，是减轻这一负担的天然候选者。为了衡量编程智能体在数据复用方面的表现，我们在八篇近期的论文上对其进行基准测试，这些论文包含大规模小鼠神经记录，涵盖多种记录模态、行为范式和数据格式。在给定数据、代码和论文的情况下，智能体必须将数据重新格式化以用于一个常见的下游任务：训练从神经活动到任务和行为变量的解码器。

    arXiv:2605.12808v3 Announce Type: replace  Abstract: Neuroscience data are fragmented across labs, experimental paradigms, and storage formats, and data reuse often requires substantial manual effort. Standardized formats have been proposed in response, but a fundamental tension persists: formats flexible enough for diverse experiments are rarely self-explanatory, while fully descriptive datasets demand documentation and curation effort that few labs can sustain. Coding agents, which can read code and text quickly with apparent attention to details humans tend to skim, are a natural candidate to reduce this burden. To measure how well coding agents perform data reuse, we benchmark them on eight recent papers with large-scale mouse neural recordings spanning diverse recording modalities, behavioral paradigms, and formats. Given the data, code, and paper, the agents must reformat the data for a common downstream task: training a decoder from neural activity to task and behavioral variabl
    
[^394]: 先规划后交易：强化学习交易智能体的推理时优化

    Plan Before You Trade: Inference-Time Optimization for RL Trading Agents

    [https://arxiv.org/abs/2605.12653](https://arxiv.org/abs/2605.12653)

    提出了受模型预测控制启发的FinPILOT框架，利用多步价格预测在推理时对任意预训练的强化学习交易策略进行优化，实现“先规划、后交易”的策略自适应调整。

    

    用于投资组合管理的强化学习智能体通常以静态策略的形式进行训练和部署，缺乏在推理时利用价格预测对策略进行调整的机制。我们提出了FinPILOT（Financial Plugin Inference-time Learning for Optimal Trading，金融插件推理时学习最优交易），这是一个受模型预测控制（MPC）启发的插件式推理时策略优化框架。在价格接受者假设下，投资组合动作不会影响未来市场价格，这使得多步价格预测器能够提供想象的市场轨迹，而无需迭代式的动作条件推演。在每个决策步骤，FinPILOT利用预测器预测的价格轨迹构建基于配置的想象收益目标，并在执行一步交易之前在推理时对预训练策略进行优化。我们的框架与任何预训练策略兼容，并利用预测器的预测对策略进行调整，而无需……

    arXiv:2605.12653v2 Announce Type: replace-cross  Abstract: Reinforcement learning agents for portfolio management are typically trained and deployed as static policies, with no mechanism for policy adjustments using price forecasts at inference time. We propose FinPILOT (Financial Plugin Inference-time Learning for Optimal Trading), a plugin inference-time policy optimization framework inspired by Model Predictive Control (MPC). Under the price-taker assumption, portfolio actions do not affect future market prices, allowing a multi-step price forecaster to provide imagined market trajectories without iterative action-conditioned rollouts. At each decision step, FinPILOT uses the forecaster's predicted price trajectory to construct an allocation-based imagined return objective and optimizes the pretrained policy at inference time before executing one step of the trade. Our framework is compatible with any pretrained policy and adapts the policy using the forecaster's predictions without
    
[^395]: 一种用于复合对数凹采样的近端梯度算法

    A proximal gradient algorithm for composite log-concave sampling

    [https://arxiv.org/abs/2605.12461](https://arxiv.org/abs/2605.12461)

    提出了一种针对复合对数凹分布的近端梯度采样算法，通过结合 $f$ 的梯度评估与 $g$ 的受限高斯预言机，在 $f+g$ 为 $\alpha$-凸且 $f$ 为 $\beta$-光滑的条件下，以 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$ 次预言机调用实现总变差距离下 $\varepsilon$ 精度的采样，匹配了 $g=0$ 情形下的最优结果。

    

    我们提出了一种从 $\mathbb{R}^d$ 上的复合对数凹分布中进行采样的算法，即形如 $\pi\propto e^{-f-g}$ 的密度，假设我们能够获得 $\alpha_f$-凸函数 $f$ 的梯度评估，以及 $\alpha_g$-凸函数 $g$（其中 $\alpha_g \ge 0$）的受限高斯预言机（RGO）。后一要求意味着我们可以容易地从密度 $\text{RGO}_{g,h,y}(x) \propto \exp(-g(x) -\frac{1}{2h}||y-x||^2)$ 中采样，这相当于 $g$ 的近端算子在采样上的类比。如果 $f + g$ 是 $\alpha$-凸的且 $f$ 是 $\beta$-光滑的，我们的采样器在总变差距离下达到 $\varepsilon$ 误差，所需的预言机调用次数为 $\widetilde{\mathcal{O}}(\kappa \sqrt d \log^4(1/\varepsilon))$，其中 $\kappa := 1 \vee \beta/\alpha$，$0\le \alpha_g \le \beta$ 且 $\alpha := \alpha_f + \alpha_g > 0$，这与之前 $g=0$ 情形下最先进的结果相匹配。我们进一步将结果扩展到以下情形：(1) ...（摘要在此处被截断）

    arXiv:2605.12461v2 Announce Type: replace-cross  Abstract: We propose an algorithm to sample from composite log-concave distributions over $\mathbb{R}^d$, i.e., densities of the form $\pi\propto e^{-f-g}$, assuming access to gradient evaluations of $\alpha_f$-convex $f$ and a restricted Gaussian oracle (RGO) for $\alpha_g$-convex $g$, where $\alpha_g \ge 0$. The latter requirement means that we can easily sample from the density $\text{RGO}_{g,h,y}(x) \propto \exp(-g(x) -\frac{1}{2h}||y-x||^2)$, which is the sampling analogue of the proximal operator for $g$. If $f + g$ is $\alpha$-convex and $f$ is $\beta$-smooth, our sampler achieves $\varepsilon$ error in total variation distance using $\widetilde{\mathcal O}(\kappa \sqrt d \log^4(1/\varepsilon))$ oracle calls where $\kappa := 1 \vee \beta/\alpha$, with $0\le \alpha_g \le \beta$ and $\alpha := \alpha_f + \alpha_g > 0$, which matches prior state-of-the-art results for the case $g=0$. We further extend our results to cases where (1) $
    
[^396]: 通过深度低秩残差蒸馏锁定预训练权重

    Locking Pretrained Weights via Deep Low-Rank Residual Distillation

    [https://arxiv.org/abs/2605.10777](https://arxiv.org/abs/2605.10777)

    提出DLR-Lock方法，利用自动微分在推理与训练之间的不对称性，通过深度低秩残差蒸馏锁定预训练权重，防止适应性攻击者对开放权重模型进行未经授权的微调和滥用。

    

    arXiv:2605.10777v2 公告类型：替换 摘要：近年来，开放权重语言模型的质量得到了显著提升。共享权重使模型能够在多样化的硬件和软件平台上使用，极大地促进了模型的采用。开放权重还支持更开放的研究和测试，用户可以将其作为检查点，根据自身需求进行微调，甚至可能重新分发。然而，在某些情况下，对权重进行修改以用于未经授权用途的担忧可能超过给予用户这种自由的好处。防御此类适应性攻击并非易事：由于适应性攻击者按定义可以观察所有权重和架构，他们可以逆转简单的结构性防御，并通过优化手段击败最简单的锁定机制。在这项工作中，我们利用自动微分在推理与训练之间的不对称性作为一种全新的防御轴。我们提出了DLR-Lock方法，其中模型提供者（摘要内容在此处截断）

    arXiv:2605.10777v2 Announce Type: replace  Abstract: The quality of open-weight language models has dramatically improved in recent years. Sharing weights greatly facilitates model adoption by enabling their use across diverse hardware and software platforms. They also allow for more open research and testing, to the extent that users can use them as checkpoints, fine-tune them according to their needs, and potentially redistribute them. In some cases, however, concerns on modifying these weights towards unauthorized uses may outweigh the pros of giving users such freedom. Defending against such adaptation is non-trivial: since an adaptive attacker can observe all weights and architectures by definition, they can reverse simple structural defenses, and use optimization to defeat the simplest locking mechanisms. In this work, we exploit the inference-training asymmetry of automatic differentiation as a novel defense axis. We propose DLR-Lock, a method where the purveyor of the model pur
    
[^397]: TabFORGE：一个用于生成建模的表格基础模型

    TabFORGE: A Tabular Foundation Model for Generative Modelling

    [https://arxiv.org/abs/2605.09424](https://arxiv.org/abs/2605.09424)

    提出了TabFORGE——一个基于预训练表格基础生成表示的新型表格基础模型，通过更好地对齐表格数据的特征间依赖结构，显著提升生成式表格建模的合成数据质量。

    

    生成建模是对基础模型的一项高要求测试，因为它需要针对给定数据模态进行稳健、全面的表示学习，而不仅仅是针对监督预测目标进行优化。尽管近期关于表格基础模型的研究在预测建模方面取得了显著进展，但生成式表格基础模型仍未得到充分探索。尤其是，现有的表格基础生成器在合成数据质量方面尚未能持续匹敌强大的针对特定数据集的生成器。一个关键原因在于它们与表格数据的特征间依赖结构（可能反映潜在因果关系）的对齐不足。在本文中，我们通过引入一种新颖的表格基础模型 TabFORGE 来填补这一空白，该模型建立在预训练的表格基础生成表示（Tabular FOundational Representations for GEneration）之上。TabFORGE 旨在利用可迁移的（摘要在此处截断）

    arXiv:2605.09424v2 Announce Type: replace  Abstract: Generative modelling is a demanding test of foundation models, because it requires robust, holistic representation learning for a given data modality, rather than optimisation for a supervised prediction target alone. While recent work on tabular foundation models has achieved remarkable progress in predictive modelling, generative tabular foundation models remain underexplored. Existing tabular foundation generators, in particular, have not yet consistently matched strong dataset-specific generators in synthetic data quality. A key reason is their insufficient alignment with the inter-feature dependency structure of tabular data, which can reflect underlying causal relationships. In this paper, we address this gap by introducing a novel tabular foundation model, \textbf{TabFORGE}, built on pretrained \textbf{Tab}ular \textbf{FO}undational \textbf{R}epresentations for \textbf{GE}neration. TabFORGE is designed to utilise transferable 
    
[^398]: HAServe：面向高效大语言模型服务的视界感知调度与内存协同设计

    HAServe: Horizon-Aware Scheduling and Memory Co-Design for Efficient LLM Serving

    [https://arxiv.org/abs/2605.08581](https://arxiv.org/abs/2605.08581)

    提出HAServe，一种面向在线LLM服务的调度与内存协同设计方法，通过统一需求视图联合优化请求准入与KV缓存保留，以最小化预测延迟成本。

    

    在线大语言模型（LLM）服务处理的请求共享上下文，但前缀结构和到达时机各不相同。有限的容量将请求准入与键值（KV）缓存保留耦合在一起。本文针对一个在线服务问题：随着前缀结构和到达率的变化，利用对可见需求的统一视图，选择预测延迟成本最低的、可行的请求准入与KV保留联合方案。为了对该问题建模，我们提出了HorizonSchema，这是一种工作负载抽象，它将精确的前缀路径、允许的序列化方式以及已释放的消费者连接起来，作为两项决策的共享输入。我们的分析表明，有效的复用取决于需求与前缀存活情况，而在计算量相同的情况下，服务顺序也会改变延迟。此外，我们提出了HAServe，它将该抽象转化为一种面向在线LLM服务的、系统化优化的调度—内存协同设计方法。我们实现了HAServ（原文在此处截断）

    arXiv:2605.08581v2 Announce Type: replace  Abstract: Online large language model (LLM) services process requests that share context but differ in prefix structure and arrival timing. Finite capacity couples request admission and key-value (KV) retention. This paper addresses one online serving problem: using a unified view of visible demand to choose the feasible joint admission and KV-retention candidate with the lowest predicted latency cost as prefix structures and arrival rates vary. To model this problem, we introduce HorizonSchema, a workload abstraction that connects exact prefix paths, permitted serializations, and released consumers as shared inputs to both decisions. Our analysis shows that useful reuse depends on demand and prefix survival, while service order changes latency even at equal computation. Furthermore, we present HAServe, which turns this abstraction into a scheduling--memory co-design method of systematic optimization for online LLM serving. We implement HAServ
    
[^399]: CRAFT：面向持续学习的遗忘感知干预式适应方法

    CRAFT: Forgetting-Aware Intervention-Based Adaptation for Continual Learning

    [https://arxiv.org/abs/2605.05732](https://arxiv.org/abs/2605.05732)

    CRAFT提出了一种冻结大模型权重、通过学习低秩隐藏表示干预实现持续学习的框架，利用基于输出分布的任务相似性共享干预、KL惩罚以及深层任务私有干预来缓解灾难性遗忘，且在测试时无需任务身份标签即可自动识别任务。

    

    大型语言模型（LLMs）可以通过微调学习新任务，但按顺序学习多个任务往往会导致对先前任务的灾难性遗忘。本文提出了CRAFT，这是一种持续学习框架，它保持模型权重冻结，转而学习对隐藏表示的低秩干预。相似的任务共享同一个干预以促进知识迁移，其中相似性通过每个任务如何改变模型的输出分布来判断，而不是通过学习到的门控网络来判断。在共享干预内部，Kullback-Leibler（KL）惩罚使每个新任务保持接近该干预的既有行为，从而限制对较早任务的遗忘；同时，在最深层中引入一个小型的任务私有干预，保留了后续任务无法覆盖的专属容量。所学到的干预还使得在测试时能够识别任务，因此不需要任务身份标签。在多个基准……（摘要原文在此处截断）

    arXiv:2605.05732v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) can learn new tasks through fine-tuning, but learning tasks sequentially often leads to catastrophic forgetting of earlier ones. This paper proposes CRAFT, a continual learning framework that keeps model weights frozen and instead learns low-rank interventions on hidden representations. Similar tasks share an intervention to help knowledge transfer, with similarity judged by how each task alters the model's output distribution rather than by a learned gating network. Within a shared intervention, a Kullback-Leibler (KL) penalty keeps each new task close to the intervention's existing behavior, which limits forgetting of earlier tasks, while a small, task-private intervention in the deepest layers preserves dedicated capacity that later tasks cannot overwrite. The learned interventions also make it possible to identify the task at test time, so task identity labels are not needed. Across multiple ben
    
[^400]: 面向可扩展搜索的无标识符代码嵌入模型

    Identifier-Free Code Embedding Models for Scalable Search

    [https://arxiv.org/abs/2605.05251](https://arxiv.org/abs/2605.05251)

    该论文形式化了源代码与去除符号的反编译代码之间的双向函数关联问题，并通过对比学习微调Qwen3-Embedding模型，在所有基线测试中大幅超越现有模型。

    

    函数关联对二进制逆向工程师来说是一项非常有用的技术。目前已有可在大规模场景下执行关联的搜索工具，但它们并未充分利用AI赋能搜索所提供的全部能力。先前的工作已探索了为某些逆向工程代码表示之间的关联开发嵌入模型，但这些工作并未涵盖在标准预处理要求下源代码与去除符号的反编译代码之间的双向关联。为了填补这一空白，我们对这一函数关联问题进行了形式化定义，并评估了嵌入模型在这两种代码表示之间实现双向关联的能力。为提升模型在该任务上的性能，我们使用对比学习对Qwen3-Embedding模型进行了微调。我们发现，我们的新模型在所有函数关联基线测试中均以显著优势超越其他模型，并能泛化至常数算法关联等场景（原文在此处截断）。

    arXiv:2605.05251v2 Announce Type: replace-cross  Abstract: Function association is a useful process for binary reverse engineers. Search tools exist to perform association at scale, but they do not utilize the full range of capabilities that AI-enabled search provides. Prior work has explored the development of embedding models for association between certain reverse engineering code representations, but that work does not cover bidirectional association between source code and decompiled, stripped code with standard preprocessing requirements. To bridge this gap, we formalize this function association problem and evaluate the extent to which embedding models can bidirectionally associate between these two representations. To improve model performance at this task, we fine-tune a Qwen3-Embedding model with contrastive learning. We find that our new model outperforms other models on all function association baselines by a substantial margin and generalizes to a constant-algorithm associ
    
[^401]: 大语言模型拥有核心信念吗？

    Do LLMs have core beliefs?

    [https://arxiv.org/abs/2605.03255](https://arxiv.org/abs/2605.03255)

    本文提出“对抗对话树”探测框架，在五个知识领域测试发现大多数大语言模型无法在对抗性对话压力下维持稳定的世界观和核心信念，表明它们并不具备类似人类的核心信念。

    

    大语言模型（LLM）的兴起引发了关于这些系统是否具备人类水平认知的争论。在这场争论中，人类认知的一个结构性组成部分却很少受到关注：核心信念，即为我们构建世界观提供基础的真实信条。这些信念通常能够抵御驳斥，因为放弃它们将意味着我们看待现实的方式发生根本性转变。在本文中，我们探究大语言模型是否持有类似核心信念的东西。我们使用一个名为“对抗对话树”的探测框架，在五个领域（科学、历史、地理、生物学和数学）上进行测试，发现大多数大语言模型无法维持稳定的世界观。尽管一些最新的模型表现出了更好的稳定性，但在对话压力下它们最终仍无法维持关键的信念承诺。这些结果记录了模型各代际之间论证能力的提升，但表明所有（模型都）……

    arXiv:2605.03255v2 Announce Type: replace  Abstract: The rise of Large Language Models (LLMs) has sparked debate about whether these systems exhibit human-level cognition. In this debate, little attention has been paid to a structural component of human cognition: core beliefs, truths that provide a foundation around which we can build a worldview. These commitments usually resist debunking, as abandoning them would represent a fundamental shift in how we see reality. In this paper, we ask whether LLMs hold anything akin to core commitments. Using a probing framework we call Adversarial Dialogue Trees (ADTs) over five domains (science, history, geography, biology, and mathematics), we find that most LLMs fail to maintain a stable worldview. Though some recent models showed improved stability, they still eventually failed to maintain key commitments under conversational pressure. These results document an improvement in argumentative skills across model generations but indicate that all
    
[^402]: 关注差距：偏好学习中的结构感知一致性

    Mind the Gap: Structure-Aware Consistency in Preference Learning

    [https://arxiv.org/abs/2604.27733](https://arxiv.org/abs/2604.27733)

    本文证明了标准替代损失在LLM对齐中无法保证排序一致性，并提出一种结构感知的边际调整方法（SA-DPO），通过利用响应间的语义距离实现稳定且一致的对齐。

    

    将大型语言模型（LLMs）与人类意图对齐，无论是通过显式奖励建模还是直接方法如DPO，从根本上依赖于最小化替代损失作为真实成对排序目标的代理。我们证明了这种依赖对于使用的标准替代损失是有缺陷的：对于神经网络特有的等连续假设集，没有任何标准替代损失能提供有意义的一致性保证。将替代损失最小化到零可能使真实排序误差任意高。为解决此问题，我们将LLM对齐形式化为一个带有边际偏移的排序框架，并推导出$H$-一致性界限，表明强制置信边际$\gamma$不仅有益，而且对一致性是必要的。我们进一步引入了结构感知$H$-一致性及相应目标（SA-DPO），该目标根据响应之间的语义距离调整边际，防止不稳定性。

    arXiv:2604.27733v2 Announce Type: replace  Abstract: Aligning Large Language Models (LLMs) with human intent, whether through explicit reward modeling or direct methods such as DPO, fundamentally relies on minimizing a surrogate loss as a proxy for the true pairwise ranking objective. We prove that this reliance is flawed for the standard surrogate losses used: for the equicontinuous hypothesis sets characteristic of neural networks, no standard surrogate provides a meaningful consistency guarantee. Minimizing the surrogate loss to zero can leave the true ranking error arbitrarily high. To resolve this, we formulate LLM alignment within a margin-shifted ranking framework and derive $H$-consistency bounds showing that enforcing a confidence margin $\gamma$ is not merely beneficial but necessary for consistency. We further introduce Structure-Aware $H$-consistency and a corresponding objective (SA-DPO) that adapts the margin to the semantic distance between responses, preventing instabil
    
[^403]: TensorDex：一种面向现代AI模型的紧凑型、以张量为中心的存储系统

    TensorDex: A Compact, Tensor-Centric Storage System for Modern AI Models

    [https://arxiv.org/abs/2604.17104](https://arxiv.org/abs/2604.17104)

    TensorDex是一个无损的、以张量为粒度进行增量压缩的存储系统，通过紧凑的位级指纹预测张量对的可压缩性，有效消除模型托管平台中微调变体之间的跨模型冗余。

    

    现代模型托管平台存储着数百PB规模的大型语言模型（LLM），其中微调变体占据了存储空间的主要部分。这些变体包含大量跨模型冗余，增量压缩（delta compression）可以通过仅存储目标模型与参考模型之间的差异来利用这种冗余。然而，压缩效果在很大程度上取决于能否选择一个相似的参考模型。在模型托管平台的规模下，这极具挑战性，因为模型谱系元数据通常缺失或不可靠，而且同一模型中的不同张量可能与来自不同模型的张量最为相似。因此，模型级别的配对方式会留下大量未被利用的冗余。我们提出了TensorDex，一个无损的、以张量为中心的存储系统，它在张量粒度上执行增量压缩。TensorDex将模型分解为张量，使用紧凑的位级指纹来预测张量对的可压缩性，并以增量方式组织张量。

    arXiv:2604.17104v3 Announce Type: replace-cross  Abstract: Modern model hubs store hundreds of petabytes of large language models (LLMs), with fine-tuned variants dominating the storage footprint. These variants contain substantial cross-model redundancy that delta compression can exploit by storing only the difference between a target and a reference model. However, compression effectiveness depends critically on choosing a similar reference. At model-hub scale, this is challenging because model lineage metadata is often missing or unreliable, and different tensors within the same model may be most similar to tensors from different models. Consequently, model-level pairing leaves substantial redundancy unexploited.   We present TensorDex, a lossless, tensor-centric storage system that performs delta compression at tensor granularity. TensorDex decomposes models into tensors, uses compact bit-level fingerprints to predict tensor-pair compressibility, and incrementally organizes tensors
    
[^404]: 通过广义执行器网络实现肌肉驱动机器人的仿真到真实迁移

    Sim-to-Real Transfer for Muscle-Actuated Robots via Generalized Actuator Networks

    [https://arxiv.org/abs/2604.09487](https://arxiv.org/abs/2604.09487)

    提出广义执行器网络（GenAN），无需扭矩传感器、仅从关节位置轨迹即可学习气动人工肌肉等复杂驱动系统的神经模型，实现了肌肉驱动机器人动态精确技能从仿真到真实系统的迁移。

    

    肌腱驱动配合软体肌肉驱动能够实现更快、更安全的机器人，并有望加速技能习得。然而，由于固有的非线性、摩擦和迟滞特性使建模和控制变得复杂，这类系统在实践中很少被使用。迄今为止，这些挑战一直阻碍着策略从仿真到真实系统的迁移。为了弥合这一差距，我们提出了一个仿真到真实的流程：学习一个神经网络模型来刻画这种复杂的驱动方式，同时利用成熟的刚体仿真来处理手臂动力学及与环境的交互。我们的方法称为广义执行器网络，通过直接从关节位置轨迹中学习，而无需扭矩传感器，即可实现对多种机器人的驱动模型识别。我们将GenAN应用于PAMY2——一种由气动人工肌肉驱动的肌腱驱动机器人，成功部署了动态且精确的目标……

    arXiv:2604.09487v4 Announce Type: replace-cross  Abstract: Tendon drives paired with soft muscle actuation enable faster and safer robots while potentially accelerating skill acquisition. Still, these systems are rarely used in practice due to inherent nonlinearities, friction, and hysteresis, which complicate modeling and control. So far, these challenges have hindered policy transfer from simulation to real systems. To bridge this gap, we propose a sim-to-real pipeline that learns a neural network model of this complex actuation and leverages established rigid body simulation for the arm dynamics and interactions with the environment. Our method, called Generalized Actuator Network (GenAN), enables actuation model identification across a wide range of robots by learning directly from joint position trajectories rather than requiring torque sensors. Using GenAN on PAMY2, a tendon-driven robot powered by pneumatic artificial muscles, we successfully deploy dynamic but precise goal-reac
    
[^405]: 《在何处引导：基于输入依赖的引导层选择改进大语言模型对齐》

    Where to Steer: Input-Dependent Layer Selection for Steering Improves LLM Alignment

    [https://arxiv.org/abs/2604.03867](https://arxiv.org/abs/2604.03867)

    该论文发现最优引导层因输入而异，并提出W2S框架，根据输入自适应地选择引导干预的最佳层位置，从而改进大语言模型的推理时对齐效果。

    

    引导向量已成为一种在推理时对齐大语言模型（LLM）的轻量且有效的方法，它通过将中间表示向目标行为偏移来调节模型行为。许多现有的引导方法在LLM的全局固定位置应用干预，实际上将最优干预位置视为与输入无关。然而，与目标行为相关的表示可能根据输入的不同而在LLM的不同层被编码。我们从理论上证明，不同的输入可能需要在不同层进行引导才能实现与给定行为的对齐；在实证上，我们也观察到最有效的引导层在不同输入之间差异显著。基于这些观察，我们提出了Where to Steer（W2S），一个通用框架，通过学习从输入到干预层的映射，根据输入自适应地选择干预层。

    arXiv:2604.03867v2 Announce Type: replace  Abstract: Steering vectors have emerged as a lightweight and effective approach for aligning large language models (LLMs) at inference time, enabling modulation of model behaviors by shifting intermediate representations toward a target behavior. Many existing steering methods apply interventions at globally fixed locations in an LLM, effectively treating the optimal intervention location as input-invariant. However, representations relevant to a target behavior can be encoded at different LLM layers depending on the input. Theoretically, we show that different inputs may require steering at different layers to achieve alignment with a given behavior. Empirically, we observe that the most effective layer for steering varies substantially between inputs. Motivated by these observations, we introduce \textit{Where to Steer} (W2S), a general framework that adaptively selects intervention layers conditioned on the inputs by learning a mapping from
    
[^406]: 面向动力系统的分布到分布神经概率预测框架

    A Distribution-to-Distribution Neural Probabilistic Forecasting Framework for Dynamical Systems

    [https://arxiv.org/abs/2603.25370](https://arxiv.org/abs/2603.25370)

    本文提出了一种分布到分布（D2D）神经概率预测框架，利用核均值嵌入编码输入分布、混合密度网络参数化输出分布，实现了动力系统的递归多步概率预测，避免了求解计算代价高昂的分布演化方程。

    

    概率预测通过预测分布为预测不确定性提供了有原则的表示。使用传统物理模型直接传播预测分布需要求解分布演化方程，而对于高维非线性系统，这一任务在计算上是众所周知的难以承受的。因此，多步预测分布通常使用集合或采样轨迹来近似。本文提出了一种新颖的分布到分布（D2D）神经概率预测框架，该框架在一个可替换的神经预测模块周围配置了兼容的分布编码和解码接口。输入分布采用核均值嵌入表示，输出则由混合密度网络进行参数化，从而能够以分布形式进行递归多步预测。该框架在Lorenz63系统上进行了演示，结果表明D2D模型能够捕获非……

    arXiv:2603.25370v2 Announce Type: replace-cross  Abstract: Probabilistic forecasts provide a principled representation of forecast uncertainty through predictive distributions. Directly propagating predictive distributions with conventional physical models requires solving distributional evolution equations, a task well-known to be computationally prohibitive for high-dimensional nonlinear systems. Consequently, multi-step predictive distributions are commonly approximated using ensembles or sampled trajectories. A novel distribution-to-distribution (D2D) neural probabilistic forecasting framework is proposed with compatible distributional encoding and decoding interfaces around a replaceable neural forecasting module. Kernel mean embeddings represent input distributions, while mixture density networks parameterise outputs, enabling recursive multi-step forecasting in distributional form. The framework is demonstrated on the Lorenz63 system. Results show that the D2D model captures non
    
[^407]: SEVerA：自演化智能体的经验证合成

    SEVerA: Verified Synthesis of Self-Evolving Agents

    [https://arxiv.org/abs/2603.25111](https://arxiv.org/abs/2603.25111)

    提出了SEVerA框架，通过形式化守护生成模型（FGGM）将自演化智能体代码生成形式化为受约束的学习问题，从而为自演化LLM智能体提供安全性和正确性的形式化保证。

    

    近期的研究进展表明，自演化的大语言模型（LLM）智能体在程序修复和科学发现等任务上展现出了有效性。在这一范式中，规划器LLM会合成一个调用参数化模型（包括LLM）的智能体程序，然后针对每个任务对这些模型进行调优以提升性能。然而，现有的自演化智能体框架并未提供安全性或正确性的形式化保证。由于此类程序通常在未见过的输入上自主执行，这种保证的缺失引发了可靠性和安全性方面的担忧。我们将智能体代码生成问题形式化为一个受约束的学习问题，将硬性的形式化规范与捕捉任务效用的软目标相结合。我们提出了形式化守护生成模型（FGGM），它允许规划器LLM使用一阶逻辑为每次生成模型调用指定形式化的输出契约。每次FGGM调用都会将底层模型封装在一个带有经验证组件的拒绝采样器中……

    arXiv:2603.25111v3 Announce Type: replace  Abstract: Recent advances have shown the effectiveness of self-evolving LLM agents on tasks such as program repair and scientific discovery. In this paradigm, a planner LLM synthesizes an agent program that invokes parametric models, including LLMs, which are then tuned per task to improve performance. However, existing self-evolving agent frameworks provide no formal guarantees of safety or correctness. Because such programs are often executed autonomously on unseen inputs, this lack of guarantees raises reliability and security concerns. We formulate agentic code generation as a constrained learning problem, combining hard formal specifications with soft objectives capturing task utility. We introduce Formally Guarded Generative Models (FGGM), which allow the planner LLM to specify a formal output contract for each generative model call using first-order logic. Each FGGM call wraps the underlying model in a rejection sampler with a verified 
    
[^408]: 一种基于模型集成的公平感知预测后处理框架

    A Model Ensemble-Based Post-Processing Framework for Fairness-Aware Prediction

    [https://arxiv.org/abs/2603.18838](https://arxiv.org/abs/2603.18838)

    本文提出一种基于模型集成的后处理框架，该方法独立于模型内部结构、适用于多种学习任务和公平性定义，能够在几乎不损失预测精度的前提下有效提升模型公平性。

    

    在预测性能与公平性之间寻求最优平衡仍然是机器学习中的一项根本性挑战。在本工作中，我们提出了一种后处理框架，通过利用模型集成来促进公平感知预测。该方法的设计独立于任何特定的模型内部结构，因此可广泛应用于各种学习任务、模型架构和公平性定义。通过涵盖分类、回归和生存分析的大量实验，我们证明该框架能够在保持预测精度基本不变或仅受到极小影响的前提下，有效提升公平性。

    arXiv:2603.18838v2 Announce Type: replace  Abstract: Striking an optimal balance between predictive performance and fairness continues to be a fundamental challenge in machine learning. In this work, we propose a post-processing framework that facilitates fairness-aware prediction by leveraging model ensembling. Designed to operate independently of any specific model internals, our approach is widely applicable across various learning tasks, model architectures, and fairness definitions. Through extensive experiments spanning classification, regression, and survival analysis, we demonstrate that the framework effectively enhances fairness while maintaining, or only minimally affecting, predictive accuracy.
    
[^409]: 校准距离的计算与统计难度

    Computational and Statistical Hardness of Calibration Distance

    [https://arxiv.org/abs/2603.18391](https://arxiv.org/abs/2603.18391)

    该论文提出了在均匀边际和无噪声标签假设下精确计算校准距离的高效算法，证明移除任一假设后问题即变为NP难，并给出了适用于一般情形的PTAS以及估计问题所需的Θ(1/ε³)样本复杂度的紧界。

    

    由Błasiok、Gopalan、Hu和Nakkiran（STOC 2023）提出的校准距离，近来已成为衡量概率预测器失准程度的核心指标。我们研究了在给定数据分布的精确描述或仅有数据采样访问的条件下，计算和估计该量的基本问题。当分布具有均匀边际且标签无噪声时，我们给出了一个能精确计算校准距离的高效算法，这改进了Qiao和Zheng（COLT 2024）针对该特殊情形所给出的O(1/√|X|)加性近似。或许令人惊讶的是，当这两个假设中的任意一个被移除时，该问题即变为NP难问题。我们进一步将算法推广为适用于一般情形的多项式时间近似方案（PTAS）。对于估计问题，我们证明了Θ(1/ε³)个样本对于经验校准……（摘要在此处截断）是充分且必要的。

    arXiv:2603.18391v2 Announce Type: replace-cross  Abstract: The distance from calibration, introduced by B{\l}asiok, Gopalan, Hu, and Nakkiran (STOC 2023), has recently emerged as a central measure of miscalibration for probabilistic predictors. We study the fundamental problems of computing and estimating this quantity, given either an exact description of the data distribution or only sample access to it.   We give an efficient algorithm that exactly computes the calibration distance when the distribution has a uniform marginal and noiseless labels, which improves the $O(1/\sqrt{|\mathcal{X}|})$ additive approximation of Qiao and Zheng (COLT 2024) for this special case. Perhaps surprisingly, the problem becomes $\mathsf{NP}$-hard when either of the two assumptions is removed. We extend our algorithm to a polynomial-time approximation scheme for the general case.   For the estimation problem, we show that $\Theta(1/\epsilon^3)$ samples are sufficient and necessary for the empirical cal
    
[^410]: 学会不干扰：为模型合并适配专家模型

    Learning Not To Interfere: Adapting Experts For Model Merging

    [https://arxiv.org/abs/2603.13467](https://arxiv.org/abs/2603.13467)

    本文提出了轻量级适配框架RI（干扰解决），通过将专家模型解耦为与其他任务空间功能正交的形式来减少模型合并中的跨任务干扰，且仅需未标记的辅助数据即可实现，适用于数据稀缺场景。

    

    模型合并已经表明，可以通过直接组合不同模型的参数来创建多任务模型，其中每个模型分别专门针对感兴趣的任务进行训练。然而，在不同任务上独立训练的模型通常会表现出相互干扰，从而降低合并模型的性能。为了解决这个问题，我们正式定义了“跨任务干扰”的概念，即合并模型的表示相对于其组成模型所产生的漂移。减少跨任务干扰是提高合并性能的关键。为了解决这一问题，我们提出了我们的方法——干扰解决（Resolving Interference, RI），这是一个轻量级的适配框架，它将专家模型进行解耦，使其在功能上与其他任务的空间正交，从而减少跨任务干扰。RI 在实现这一目标的同时，仅使用未标记的辅助数据作为输入（即不需要任务数据），使其能够应用于数据稀缺的场景。

    arXiv:2603.13467v2 Announce Type: replace-cross  Abstract: Model merging has shown that multitask models can be created by directly combining the parameters of different models that are each specialized on tasks of interest. However, models trained independently on distinct tasks often exhibit interference that degrades the merged model's performance. To solve this problem, we formally define the notion of Cross-Task Interference as the drift in the representation of the merged model relative to its constituent models. Reducing cross-task interference is key to improving merging performance. To address this issue, we propose our method, Resolving Interference (RI), a light-weight adaptation framework which disentangles expert models to be functionally orthogonal to the space of other tasks, thereby reducing cross-task interference. RI does this whilst using only unlabeled auxiliary data as input (i.e., no task-data is needed), allowing it to be applied in data-scarce scenarios. RI cons
    
[^411]: 3DTCR：涡旋跟随生成式重建改进基于全球AI天气预报的热带气旋强度预测

    3DTCR: Vortex-following generative reconstruction improves tropical cyclone intensity prediction from global AI weather forecasts

    [https://arxiv.org/abs/2603.13049](https://arxiv.org/abs/2603.13049)

    提出3DTCR框架，利用涡旋跟随的条件流匹配生成式重建方法，从全球AI天气预报中恢复热带气旋公里级三维内核结构，从而显著改进热带气旋强度预测。

    

    全球数值天气预报（NWP）与人工智能（AI）模型往往低估热带气旋（TC）的强度，部分原因在于粗网格无法充分解析影响热带气旋强度演变的精细尺度三维内核结构。公里级NWP模拟能够更好地解析这些结构，但计算成本过高，难以用于业务运行，这也导致高分辨率热带气旋数据十分稀缺。本研究利用天气研究与预报模式（WRF）构建了2018-2024年以热带气旋为中心的3公里分辨率数据集，并开发了3DTCR——一种涡旋跟随的生成式框架，可从伏羲（FuXi）全球预报中重建热带气旋的三维结构。3DTCR将条件流匹配（CFM）与两阶段训练相结合，先从中心对齐的再分析-模拟配对数据中学习由粗到细的映射，再通过潜空间对齐将其适配到预报输入上。在2024年西北太平洋（WNP）上的评估显示……

    arXiv:2603.13049v3 Announce Type: replace  Abstract: Global numerical weather prediction (NWP) and artificial intelligence (AI) models often underestimate tropical cyclone (TC) intensity, partly because coarse grids underresolve fine-scale three-dimensional inner-core structures that influence TC intensity evolution. Kilometer-scale NWP simulations can better resolve these structures but are too costly for operational use, which also makes high-resolution TC data limited. Here we construct a 3-km, TC-centered dataset for 2018--2024 using the Weather Research and Forecasting (WRF) model and develop 3DTCR, a vortex-following generative framework that reconstructs three-dimensional TC structure from FuXi global forecasts. 3DTCR combines conditional flow matching (CFM) with two-stage training to learn a coarse-to-fine mapping from center-aligned reanalysis simulation pairs and then adapt it to forecast inputs through latent space alignment. Evaluated on 2024 western North Pacific (WNP) for
    
[^412]: 部分旋转，全部潜力？探究部分RoPE的性能与收敛性

    Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial RoPE

    [https://arxiv.org/abs/2603.11611](https://arxiv.org/abs/2603.11611)

    仅对约10%的隐藏维度应用旋转位置编码（部分RoPE）即可达到与完整RoPE相当的收敛性能，同时实现高达10倍的内存节省，且最小限度的RoPE使用或QK-Norm可缓解NoPE导致的损失尖峰。

    

    旋转位置编码（RoPE）被广泛应用于Transformer中以编码相对位置信息，然而仅对隐层维度的一个子集应用RoPE的效果仍未得到充分探索。这种设计选择可以大幅减少内存使用，尤其是在长上下文场景下。我们证明部分RoPE相比标准RoPE缓存可实现高达10倍的内存节省，同时保持相当的最终损失。我们系统地研究了其对训练动态和收敛性的影响，涵盖了不同的架构、模型规模、序列长度和数据集。我们的结果揭示：（1）仅对约10%的维度应用RoPE即可达到与完整RoPE相当的收敛性；（2）这些趋势在各种设置下保持一致；（3）NoPE（无位置编码）可能产生损失尖峰，而最小限度的RoPE使用或QK-Norm的应用可以缓解这一问题。总体而言，我们的发现表明部分RoPE是一种有效的...

    arXiv:2603.11611v2 Announce Type: replace-cross  Abstract: Rotary Positional Embedding (RoPE) is widely used in transformers to encode relative positional information, yet the effect of applying RoPE to only a subset of hidden dimensions remains underexplored. This design choice can substantially reduce memory usage, especially at long context lengths. We show that partial RoPE achieves up to 10$\times$ memory savings over standard RoPE cache while maintaining comparable final loss. We systematically study its impact on training dynamics and convergence across architectures, model sizes, sequence lengths, and datasets. Our results reveal that: (1) applying RoPE to only a small fraction of dimensions (around 10\%) matches the convergence of full RoPE; (2) these trends remain consistent across settings; and (3) NoPE (No Positional Encoding) can produce loss spikes, which are mitigated by minimal RoPE usage or QK-Norm application. Overall, our findings highlight partial RoPE as an effecti
    
[^413]: JAX中的自尺度Broyden族拟牛顿方法

    Self-Scaled Broyden Family of Quasi-Newton Methods in JAX

    [https://arxiv.org/abs/2603.10599](https://arxiv.org/abs/2603.10599)

    本文实现了自尺度Broyden族拟牛顿方法（包括BFGS、DFP、Broyden、SSBFGS、SSDFP、SSBroyden及L-BFGS）在JAX中的开源工具库，以方便JAX社区使用这些优化器。

    

    我们提出了自尺度Broyden族拟牛顿方法的JAX实现，该实现与JAX完全兼容，并基于Optimistix优化库构建。该实现包括BFGS、DFP、Broyden及其自尺度变体（SSBFGS、SSDFP、SSBroyden），以及L-BFGS，并配有满足强Wolfe条件的Zoom线搜索。这是一篇简短的技术说明，而非研究论文，因为它不声称有任何新颖贡献；其目的是记录该实现，并方便JAX社区采用这些优化器。代码可在 https://github.com/IvanBioli/ssbroyden_optimistix 及 PyPI 上获取（pip install ssbroyden-optimistix）。

    arXiv:2603.10599v2 Announce Type: replace-cross  Abstract: We present a JAX implementation of the Self-Scaled Broyden family of quasi-Newton methods, fully compatible with JAX and building on the Optimistix~\cite{rader_optimistix_2024} optimisation library. The implementation includes BFGS, DFP, Broyden and their Self-Scaled variants (SSBFGS, SSDFP, SSBroyden), as well as L-BFGS, together with a Zoom line search satisfying the strong Wolfe conditions. This is a short technical note, not a research paper, as it does not claim any novel contribution; its purpose is to document the implementation and ease the adoption of these optimisers within the JAX community. The code is available at https://github.com/IvanBioli/ssbroyden_optimistix and on PyPI (pip install ssbroyden-optimistix).
    
[^414]: 具有卷积位置编码的图Transformer的尺寸可迁移性

    Size Transferability of Graph Transformers with Convolutional Positional Encodings

    [https://arxiv.org/abs/2602.15239](https://arxiv.org/abs/2602.15239)

    本文建立了带有GNN位置编码的图Transformer与流形神经网络之间的理论联系，从理论上证明了在小图上训练的图Transformer在温和假设下能够泛化到更大的图，并通过实验验证了其与GNN相当的可扩展性。

    

    Transformer在各个领域取得了显著的成功，这推动了图Transformer（GTs）作为基于注意力机制的图结构数据处理架构的兴起。GT的一个关键设计选择是使用基于图神经网络（GNN）的位置编码来引入结构信息。在这项工作中，我们通过图序列的流形极限模型的视角研究GT，并建立了带有GNN位置编码的GT与流形神经网络（MNNs）之间的理论联系。基于流形收敛下GNN的可迁移性结果，我们证明了GT可以继承其位置编码的可迁移性保证。特别是，在小图上训练的GT在温和的假设下可以证明地泛化到更大的图。我们通过在标准图基准上进行大量实验来补充我们的理论，证明GT表现出与GNN相当的可扩展行为。

    arXiv:2602.15239v3 Announce Type: replace  Abstract: Transformers have achieved remarkable success across domains, motivating the rise of Graph Transformers (GTs) as attention-based architectures for graph-structured data. A key design choice in GTs is the use of Graph Neural Network (GNN)-based positional encodings to incorporate structural information. In this work, we study GTs through the lens of manifold limit models for graph sequences and establish a theoretical connection between GTs with GNN positional encodings and Manifold Neural Networks (MNNs). Building on transferability results for GNNs under manifold convergence, we show that GTs inherit transferability guarantees from their positional encodings. In particular, GTs trained on small graphs provably generalize to larger graphs under mild assumptions. We complement our theory with extensive experiments on standard graph benchmarks, demonstrating that GTs exhibit scalable behavior on par with GNNs. To further show the effic
    
[^415]: 用于机器人故障检测与推理的自精炼视觉语言模型

    Self-Refining Vision Language Model for Robotic Failure Detection and Reasoning

    [https://arxiv.org/abs/2602.12405](https://arxiv.org/abs/2602.12405)

    ARMOR提出了一种多任务自精炼框架，通过融合大规模稀疏二值标签与小规模丰富推理标注的异构监督，并结合离线与在线模仿学习，实现对机器人故障的迭代式检测与自然语言推理，摆脱了对大量人工标注的依赖。

    

    对故障进行推理对于构建可靠且值得信赖的机器人系统至关重要。先前的方法要么将故障推理视为闭集分类问题，要么假设可以获得充足的人工标注。然而现实世界中的故障通常是微妙的、组合性的，且难以穷举，而丰富的推理标注获取成本高昂。我们通过提出ARMOR（Adaptive Round-based Multi-task mOdel，面向机器人故障检测与推理的自适应多轮多任务模型）来解决这一问题。我们将故障检测和推理形式化为一个多任务自精炼过程，模型基于过去的输出迭代地预测检测结果和自然语言推理。在训练过程中，ARMOR从异构监督中学习——包括大规模的稀疏二值标签和小规模的丰富推理标注——并通过离线与在线模仿学习相结合的方式进行优化。在推理时，ARMOR生成多……（原文摘要在此处截断）

    arXiv:2602.12405v2 Announce Type: replace-cross  Abstract: Reasoning about failures is crucial for building reliable and trustworthy robotic systems. Prior approaches either treat failure reasoning as a closed-set classification problem or assume access to ample human annotations. Failures in the real world are typically subtle, combinatorial, and difficult to enumerate, whereas rich reasoning labels are expensive to acquire. We address this problem by introducing ARMOR: Adaptive Round-based Multi-task mOdel for Robotic failure detection and reasoning. We formulate detection and reasoning as a multi-task self-refinement process, where the model iteratively predicts detection outcomes and natural language reasoning conditioned on past outputs. During training, ARMOR learns from heterogeneous supervision - large-scale sparse binary labels and small-scale rich reasoning annotations - optimized via a combination of offline and online imitation learning. At inference time, ARMOR generates m
    
[^416]: 在记忆发生之处缓解记忆化

    Mitigating Memorization Where It Happens

    [https://arxiv.org/abs/2602.08901](https://arxiv.org/abs/2602.08901)

    该论文提出了一个探测—引导框架，利用目标模型与低暴露参考模型之间的 token 级对数似然比精准定位记忆化发生的位置，并仅在这些位置施加有针对性的激活校正，从而在缓解逐字记忆问题的同时避免损害模型的正常泛化性能。

    

    大型语言模型（LLM）可能会逐字记忆并复现训练序列，这引发了隐私和版权方面的担忧。现有的缓解方法会对所有内容统一施加干预，往往会损害大多数正常泛化的 token 的性能。我们研究了目标模型与一个对被评估序列接触较少、但具备领域能力的参考模型之间的 token 级对数似然比。在控制暴露程度的实验中，该指标被证明可用于定位与序列级逐字复现相关的位置。在所评估的各种设置中，少数位置承载了不成比例的正向信号，这促使我们采用选择性的激活校正方法。基于理想的记忆化削减目标，我们推导出一个“探测—引导”框架，该框架将干预分解为两个部分：一个探测器，用于检测与记忆化相关的激活；一个引导器，仅当探测器的输出超过阈值时才施加有针对性的校正。

    arXiv:2602.08901v2 Announce Type: replace  Abstract: Large language models (LLMs) can memorize and reproduce training sequences verbatim, raising privacy and copyright concerns. Existing mitigation methods apply interventions uniformly, often degrading performance on the majority of tokens that generalize normally. We study a token-level log-likelihood ratio between a target model and a domain-competent reference with lower exposure to the evaluated sequences. Exposure-controlled experiments support its use for localizing positions associated with sequence-level verbatim recall. Across evaluated settings, a minority of positions carries a disproportionate share of the positive signal, motivating selective activation correction. From the ideal memorization reduction objective, we derive a probe--steer framework which decomposes intervention into a probe that detects memorization-relevant activations and a steer that applies targeted correction only when the probe exceeds a threshold. We
    
[^417]: TASTE：基于Stein算子的任务感知分布外检测

    TASTE: Task-Aware Out-of-Distribution Detection via Stein Operators

    [https://arxiv.org/abs/2602.07640](https://arxiv.org/abs/2602.07640)

    提出基于Stein算子的任务感知分布外检测框架TASTE，将分布偏移解释为向模型敏感度场的投影，不仅能检测偏移还能通过逐坐标分解和逐像素诊断实现偏移的定位。

    

    分布外检测方法通常要么以数据为中心，即检测与训练输入分布的偏差而不考虑其对训练后模型的影响；要么以模型为中心，即依赖分类器的输出而不显式参考数据的几何结构。我们提出了TASTE（任务感知Stein算子）：一个基于所谓Stein算子的任务感知框架，它将分布偏移与模型的输入敏感度联系起来。由此得到的算子具有几何解释——将分布偏移投影到模型的敏感度场上，并具有理论保证。除了检测偏移之外，该构造还能通过逐坐标分解实现偏移的定位，并且对于图像数据，还能提供可解释的逐像素诊断。在受控高斯偏移、几何扰动下的MNIST以及CIFAR-10/ImageNet-1K扰动基准上的实验表明，TASTE与（摘要在此处截断）

    arXiv:2602.07640v2 Announce Type: replace  Abstract: Out-of-distribution detection methods are often either data-centric, detecting deviations from the training input distribution irrespective of their effect on a trained model, or model-centric, relying on classifier outputs without explicit reference to data geometry. We propose TASTE (Task-Aware STEin operators): a task-aware framework based on so-called Stein operators, which links distribution shift to the input sensitivity of the model. The resulting operators admit a geometric interpretation as a projection of distribution shift onto the sensitivity field of the model, with theoretical guarantees. Beyond detecting a shift, the construction enables its localisation through a coordinate-wise decomposition, and - for image data - provides interpretable per-pixel diagnostics. Experiments on controlled Gaussian shifts, MNIST under geometric perturbations, and CIFAR-10/ImageNet-1K perturbed benchmarks demonstrate that TASTE aligns clo
    
[^418]: 时间重采样对临床强化学习安全性的影响：一项模拟试验

    The impact of temporal resampling on the safety of clinical reinforcement learning: A simulated trial

    [https://arxiv.org/abs/2602.06603](https://arxiv.org/abs/2602.06603)

    本研究通过模拟试验首次量化了时间分箱重采样对临床离线强化学习数据造成的失真，发现其会颠倒血糖与胰岛素之间的因果顺序（超过70%的配对样本）、掩盖血糖异常事件，从而威胁临床强化学习的安全性。

    

    强化学习（RL）是一种用于做出最优决策的人工智能技术。在医疗保健领域，研究人员通常使用离线强化学习（ORL），即基于回顾性观察数据进行模型的训练与评估。为适应临床上天然不规则的数据记录，研究人员通常在训练前将数据重采样为均匀的时间间隔（称为“分箱”）。然而，分箱可能会产生无法代表真实临床实践的反事实轨迹。我们首先在MIMIC-IV重症监护数据库中接受持续胰岛素输注的患者身上量化了这些失真：在同一时间间隔内，超过70%的配对血糖测量与胰岛素调整之间的表观因果顺序被颠倒，且分箱频繁掩盖显著的血糖异常事件，并将不同的治疗决策合并在一起。随后，我们使用30名1型糖尿病虚拟患者开展了一项计算机模拟临床试验……（原文摘要在此处截断）

    arXiv:2602.06603v4 Announce Type: replace  Abstract: Reinforcement learning (RL) is a type of artificial intelligence for making optimal choices. In healthcare, researchers generally use offline RL (ORL), where models are trained and evaluated from retrospective observational data. To accommodate naturally irregular clinical records, researchers often resample the data into uniform time intervals before training (known as binning). However, binning may create counterfactual trajectories that do not represent clinical practice. We first quantified these distortions in patients receiving continuous insulin infusions in the MIMIC-IV intensive care database. Binning reversed the apparent causal order for over 70% of paired glucose measurements and insulin adjustments within the same interval, and frequently masked significant dysglycaemic events and merged distinct treatment decisions together. We then conducted an in silico clinical trial using 30 virtual patients with type 1 diabetes fro
    
[^419]: ReText：文本增强基于图像的行人重识别的泛化能力

    ReText: Text Boosts Generalization in Image-Based Person Re-identification

    [https://arxiv.org/abs/2602.05785](https://arxiv.org/abs/2602.05785)

    提出ReText方法，通过微调视觉-语言模型为多摄像头行人重识别数据集生成文本描述，并利用文本监督以及单摄像头数据的视觉与文本标注联合训练，以语义线索增强视觉特征，从而提升模型在未见域上的行人重识别泛化能力。

    

    可泛化的基于图像的行人重识别（Re-ID）旨在无需重新训练的情况下，在未见过的域中跨摄像头识别个体。虽然视觉特征对域偏移很敏感，但自然语言描述在各域之间保持一致，不受光照变化及其他偏差的影响。由于知名的多摄像头数据集缺乏文本描述，我们通过微调的视觉-语言模型（VLM）为其生成描述文本进行更新。我们提出了ReText，这是一种通过文本监督训练的新颖方法，利用语义线索丰富视觉特征。此外，受近期研究发现的启发——将单摄像头数据纳入训练可以提高泛化能力，我们也采用了这一策略。然而，与此前的相关工作不同，我们不仅利用了视觉信息，还利用了这些样本自然附带的文本标注。最后，在训练过程中，ReText联合优化三个任务：(1) Re-（摘要原文在此处截断）

    arXiv:2602.05785v2 Announce Type: replace-cross  Abstract: Generalizable image-based person re-identification (Re-ID) aims to recognize individuals across cameras in unseen domains without retraining. While visual features are sensitive to domain shifts, natural language descriptions remain consistent across domains, regardless of lighting changes and other biases. Since well-known multi-camera datasets lack textual descriptions, we update them with captions through a fine-tuned Vision-Language Model (VLM). We propose ReText, a novel method trained with textual supervision to enrich visual features with semantic cues. Besides, inspired by recent findings showing that incorporating single-camera data into training improves generalization, we also adopt this strategy. However, in contrast to such prior works, we exploit not only visual information but also the textual annotations naturally accompanying these samples. Finally, during training, ReText jointly optimizes three tasks: (1) Re-
    
[^420]: 基于长短期记忆网络与迁移学习的石油井井底压力估计软测量方法

    Soft Sensor for Bottom-Hole Pressure Estimation in Petroleum Wells Using Long Short-Term Memory and Transfer Learning

    [https://arxiv.org/abs/2602.03737](https://arxiv.org/abs/2602.03737)

    该论文提出一种结合LSTM和迁移学习的软测量方法，通过井口数据估计石油井井底压力，在真实海上数据集上实现低于2%的MAPE，为昂贵的井下物理传感器提供了经济且准确的替代方案。

    

    监测石油井中的井底变量对于生产优化、安全保障和减排至关重要。永久性井下压力计（PDG）虽能提供实时压力数据，但存在可靠性和成本方面的问题。本文提出一种基于机器学习的软测量方法，利用井口和地面测量数据来估计流动井底压力（BHP）。研究引入了长短期记忆（LSTM）模型，并与多层感知器（MLP）和岭回归进行了对比。此外，该研究还率先采用迁移学习方法，使模型能够适应不同的作业环境。在巴西盐下盆地的真实海上数据集上进行测试，该方法的平均绝对百分比误差（MAPE）始终低于2%，优于基准方法。这项工作为物理传感器提供了一种经济、准确的替代方案，可广泛适用于不同的储层和流动条件。

    arXiv:2602.03737v2 Announce Type: replace  Abstract: Monitoring bottom-hole variables in petroleum wells is essential for production optimization, safety, and emissions reduction. Permanent Downhole Gauges (PDGs) provide real-time pressure data but face reliability and cost issues. We propose a machine learning-based soft sensor to estimate flowing Bottom-Hole Pressure (BHP) using wellhead and topside measurements. A Long Short-Term Memory (LSTM) model is introduced and compared with Multi-Layer Perceptron (MLP) and Ridge Regression. We also pioneer Transfer Learning for adapting models across operational environments. Tested on real offshore datasets from Brazil's Pre-salt basin, the methodology achieved Mean Absolute Percentage Error (MAPE) consistently below 2\%, outperforming benchmarks. This work offers a cost-effective, accurate alternative to physical sensors, with broad applicability across diverse reservoir and flow conditions.
    
[^421]: CoCopula：面向可靠可再生能源预测的保形校准与Copula聚合方法

    CoCopula: Conformal Calibration and Copula Aggregation for Reliable Renewable Energy Forecasting

    [https://arxiv.org/abs/2602.02583](https://arxiv.org/abs/2602.02583)

    提出CoCopula框架，将保形校准与Copula聚合相结合，把第三方异构供应商提供的站点级概率预测直接转化为可靠且校准良好的车队级可再生能源概率预测。

    

    可再生能源渗透率的快速增长加剧了对可靠概率预测的需求，以支持在聚合（车队或系统）层面的电网运行。然而在实践中，系统运营商往往无法获得车队级概率模型，而是依赖于由异构第三方供应商提供的站点级预测。由于复杂的跨站点依赖关系以及聚合所导致的校准失准，如何从此类输入构建连贯且校准良好的车队级概率预测仍然极具挑战性。受与行业合作伙伴合作中发现的实际运营挑战的启发，本文提出了一种概率聚合与校准框架，可直接将站点级概率预测转换为可靠的车队级预测。本文提出了CoCopula方法（保形校准与Copula聚合），该方法融合了基于Copula的……

    arXiv:2602.02583v2 Announce Type: replace  Abstract: The rapid growth of renewable energy penetration has intensified the need for reliable probabilistic forecasts to support grid operations at aggregated (fleet or system) levels. In practice, however, system operators often lack access to fleet-level probabilistic models and instead rely on site-level forecasts produced by heterogeneous third-party providers. Constructing coherent and calibrated fleet-level probabilistic forecasts from such inputs remains challenging due to complex cross-site dependencies and aggregation-induced miscalibration. Motivated by a real-world operational challenge identified through collaboration with an industry partner, this paper presents a probabilistic aggregation and calibration framework that directly converts site-level probabilistic forecasts into reliable fleet-level forecasts. The paper introduces \method~ (\textbf{Co}nformal Calibration and \textbf{Copula} Aggregation), which integrates Copula-b
    
[^422]: 乘积交互：一种面向深度学习架构的代数形式化方法

    Product Interaction: An Algebraic Formalism for Deep Learning Architectures

    [https://arxiv.org/abs/2602.02573](https://arxiv.org/abs/2602.02573)

    本文提出“乘积交互”这一代数形式化框架，将乘法而非仅线性变换作为基本架构操作，通过自交互阶数统一了卷积网络、注意力机制、状态空间模型（如Mamba）和等变网络，使深度学习架构设计成为系统化流程。

    

    现代神经架构在很大程度上是作为各自独立的经验性构建而发展起来的，尽管卷积、注意力、状态空间和等变模型中存在反复出现的基本原理。我们引入了乘积交互，这是一种代数形式化方法，它在共同的构建原则下统一了这些架构，并将其设计转化为系统化的流程。其核心转变在于将数据表示为代数元素，并将乘法（而不仅仅是线性变换）作为基本的架构操作。架构通过组合乘法算子来生成，输入依赖性由自交互阶数来组织。在这一形式化框架内，我们将卷积网络和等变网络推导为受对称性约束的一阶乘积交互，从状态空间模型中恢复了Mamba（解释为自交互阶数从一阶提升到三阶），并确立了注意力机制作为一种三阶交互。

    arXiv:2602.02573v2 Announce Type: replace-cross  Abstract: Modern neural architectures are largely developed as distinct empirical constructions, despite recurring principles across convolutional, attention, state-space, and equivariant models. We introduce Product Interactions, an algebraic formalism that unifies these architectures under a common construction principle and turns their design into a systematic procedure. The core shift is representing data as algebraic elements and treating multiplication, rather than linear transformation alone, as a primitive architectural operation. Architectures are generated by composing multiplication operators, with input dependence organized by self-interaction order. Within this formalism, we derive convolutional and equivariant networks as symmetry-constrained first-order product interactions, recover Mamba from state-space models as an increase in self-interaction order from one to three, and establish attention as a third-order interaction
    
[^423]: TabClustPFN：一种用于表格数据聚类的先验数据拟合网络

    TabClustPFN: A Prior-data Fitted Network for Tabular Data Clustering

    [https://arxiv.org/abs/2601.21656](https://arxiv.org/abs/2601.21656)

    TabClustPFN通过协同设计的先验、训练目标与解耦架构，在单次推理中同时推断表格数据的聚类分配和未知聚类基数，从而克服了无监督、基数未知和标签交换等聚类中的核心挑战。

    

    先验数据拟合网络（PFNs）已将监督式表格学习重构为无需针对每个数据集进行优化的单次上下文内推理。将这一范式扩展到聚类虽然很有吸引力，却面临着更根本性的挑战，原因在于缺乏监督信息、未知的聚类基数（簇的数量），以及划分输出所固有的标签交换问题。现有的基于PFN的聚类方法只能部分地应对这些挑战：要么需要将已知的聚类基数作为输入，要么依赖于不稳定的标签排序约定和限制性过强的合成先验。我们提出了TabClustPFN，这是一种聚类PFN，通过协同设计的先验、训练目标与架构来联合解决上述所有挑战。我们的混合预训练先验能够捕捉异构的真实表格数据几何结构；我们解耦的划分推断网络与基数推断网络架构能够在单次前向传递中同时推断聚类分配和聚类基数；我们的SoftARI训练目标……（原文摘要在此处截断）

    arXiv:2601.21656v4 Announce Type: replace  Abstract: Prior-data Fitted Networks (PFNs) have reframed supervised tabular learning as single-pass in-context inference without per-dataset optimization. Extending this paradigm to clustering is appealing yet fundamentally more challenging, due to absent supervision, unknown cluster cardinality, and label switching inherent to partition outputs. Existing PFN-based clustering methods address these challenges only partially, either requiring known cardinality as input or relying on unstable label-ordering conventions and overly restrictive synthetic priors. We introduce TabClustPFN, a clustering PFN that resolves all challenges jointly through co-designed prior, objective, and architecture. Our hybrid pretraining prior captures heterogeneous real-tabular geometry; our decoupled partition inference network--cardinality inference network architecture jointly infers cluster assignments and cardinality in a single pass; and our SoftARI training ob
    
[^424]: DeFlow：解耦行为先验建模与价值最大化的离线策略提取

    DeFlow: Decoupling Behavior-Prior Modeling and Value Maximization for Offline Policy Extraction

    [https://arxiv.org/abs/2601.10471](https://arxiv.org/abs/2601.10471)

    DeFlow通过解耦行为先验建模与价值最大化，在流流形的信任域内学习轻量级精炼模块，避免对ODE求解器求导，在完整保留流模型迭代表达能力的同时实现稳定的离线强化学习性能提升与高效的在线适应。

    

    我们提出了DeFlow，一个解耦的离线强化学习框架，它利用流匹配来忠实捕捉复杂的行为流形。优化生成式策略的计算代价极为高昂，通常需要对ODE求解器进行反向传播。为解决这一问题，我们在一个由数据导出的显式流形信任域内学习一个轻量级的精炼模块，而不是通过单步蒸馏牺牲迭代生成能力。这样，我们绕过了求解器微分，并消除了平衡多个损失项的需要，在确保稳定改进的同时完整保留了流的迭代表达能力。在实证方面，DeFlow在具有挑战性的OGBench基准测试中取得了卓越的性能，并展示了高效的离线到在线适应能力。

    arXiv:2601.10471v3 Announce Type: replace  Abstract: We present DeFlow, a decoupled offline RL framework that leverages flow matching to faithfully capture complex behavior manifolds. Optimizing generative policies is computationally prohibitive, typically necessitating backpropagation through ODE solvers. We address this by learning a lightweight refinement module within an explicit, data-derived trust region of the flow manifold, rather than sacrificing the iterative generation capability via single-step distillation. This way, we bypass solver differentiation and eliminate the need for balancing loss terms, ensuring stable improvement while fully preserving the flow's iterative expressivity. Empirically, DeFlow achieves superior performance on the challenging OGBench benchmark and demonstrates efficient offline-to-online adaptation.
    
[^425]: 心血管疾病中以影像为锚点的多组学研究：整合心脏影像、bulk转录组、单细胞与空间转录组学

    Imaging-anchored Multiomics in Cardiovascular Disease: Integrating Cardiac Imaging, Bulk, Single-cell, and Spatial Transcriptomics

    [https://arxiv.org/abs/2601.07871](https://arxiv.org/abs/2601.07871)

    本综述提出以心脏影像为锚点的多组学整合框架，将超声心动图、心脏MRI和CT所定义的心脏空间表型，与bulk、单细胞及空间转录组提供的细胞类型和位置特异的分子背景相融合，从而打破数据孤岛、实现心血管疾病的多模态联合分析。

    

    心血管疾病源于遗传风险、分子程序与组织尺度重塑之间的相互作用，这些过程在临床上通过影像学手段被观察。目前医疗系统常规产生大量心脏MRI、CT和超声心动图数据，以及bulk转录组、单细胞和空间转录组数据，然而这些数据仍在各自独立的分析流程中处理。本综述探讨了将心脏影像表型与转录组及空间分辨分子状态相联系的联合表示方法。本文采用以影像为锚点的视角，即由超声心动图、心脏MRI和CT定义心脏的空间表型，而bulk、单细胞和空间转录组则提供细胞类型特异与位置特异的分子背景。文中首先总结了这些模态的生物学与技术特性，并概述了各自相应的表示学习策略。多模态融合方法……（原文摘要在此处截断）

    arXiv:2601.07871v2 Announce Type: replace-cross  Abstract: Cardiovascular disease arises from interactions between inherited risk, molecular programmes, and tissue-scale remodelling that are observed clinically through imaging. Health systems now routinely generate large volumes of cardiac MRI, CT and echocardiography together with bulk, single-cell and spatial transcriptomics, yet these data are still analysed in separate pipelines. This review examines joint representations that link cardiac imaging phenotypes to transcriptomic and spatially resolved molecular states. An imaging-anchored perspective is adopted in which echocardiography, cardiac MRI and CT define a spatial phenotype of the heart, and bulk, single-cell and spatial transcriptomics provide cell-type- and location-specific molecular context. The biological and technical characteristics of these modalities are first summarised, and representation-learning strategies for each are outlined. Multimodal fusion approaches are r
    
[^426]: 基于模拟推断中的扩散模型：教程综述

    Diffusion Models in Simulation-Based Inference: A Tutorial Review

    [https://arxiv.org/abs/2512.20685](https://arxiv.org/abs/2512.20685)

    本综述系统梳理了扩散模型在基于模拟推断（SBI）中的最新进展，涵盖训练、推断与评估的设计选择，并讨论了引导、流匹配、一致性模型等概念以及噪声调度、参数化和采样器对效率与统计精度的影响。

    

    扩散模型近来已成为基于模拟推断中强大的学习工具，能够从模拟数据和真实数据中快速且准确地估计潜在参数。其基于分数的公式化为学习参数与观测之间的条件分布或联合分布提供了灵活的方式，从而为各种建模问题提供了通用的解决方案。在本教程综述中，我们综合了扩散模型在SBI领域的最新进展，涵盖训练、推断和评估的设计选择。我们重点介绍了引导、分数组合、流匹配、一致性模型和联合建模等概念所带来的机遇。此外，我们讨论了噪声调度、参数化方法和采样器如何影响效率与统计准确性。最后，我们通过涵盖不同参数维度、模拟预算和模型类型的案例研究来阐释这些概念。

    arXiv:2512.20685v4 Announce Type: replace-cross  Abstract: Diffusion models have recently emerged as powerful learners for simulation-based inference (SBI), enabling fast and accurate estimation of latent parameters from simulated and real data. Their score-based formulation offers a flexible way to learn conditional or joint distributions over parameters and observations, thereby providing a versatile solution to various modeling problems. In this tutorial review, we synthesize recent developments on diffusion models for SBI, covering design choices for training, inference, and evaluation. We highlight opportunities created by various concepts such as guidance, score composition, flow matching, consistency models, and joint modeling. Furthermore, we discuss how efficiency and statistical accuracy are affected by noise schedules, parameterizations, and samplers. Finally, we illustrate these concepts with case studies across parameter dimensionalities, simulation budgets, and model type
    
[^427]: 用于非晶粒子系统的玻尔兹曼生成器

    Boltzmann generators for amorphous particle systems

    [https://arxiv.org/abs/2512.16607](https://arxiv.org/abs/2512.16607)

    本研究针对平衡采样极其困难的非晶材料（玻璃），通过将所需的等变性直接嵌入黎曼结构中，开发了首个专为非晶粒子系统定制的玻尔兹曼生成器。

    

    在热力学平衡态下对构型进行采样是统计物理学中一个长期存在的挑战。玻尔兹曼生成器通过使用生成模型提出独立构型来解决这一问题，随后利用精确的似然评估通过重要性采样对这些构型进行重新加权。近期基于连续归一化流和流匹配的玻尔兹曼生成器在粒子系统和生物分子领域取得了显著成功。然而，这些方法尚未被扩展到非晶材料（玻璃），而这类材料的平衡采样以极其缓慢而著称。由于其无序结构，非晶材料的不变性和几何约束与晶体和生物分子不同，这阻碍了现有生成模型的直接应用。在本研究中，我们通过将所需的等变性直接构建到黎曼（流形）结构中，开发了专为非晶材料定制的玻尔兹曼生成器。

    arXiv:2512.16607v3 Announce Type: replace-cross  Abstract: Sampling configurations in thermodynamic equilibrium is a long-standing challenge in statistical physics. Boltzmann generators address this problem by employing generative models to propose independent configurations, which are then reweighted via importance sampling using exact likelihood evaluations. Recent Boltzmann Generators based on continuous normalizing flows and flow matching have achieved significant success for particle systems and biomolecules. However, these approaches have not been extended to amorphous materials (glasses), for which equilibrium sampling is notoriously slow. Because of their disordered structure, the invariances and geometrical constraints of amorphous materials differ from those of crystals and biomolecules, preventing the direct use of existing generative models. Here, we develop Boltzmann Generators tailored to amorphous materials by building the required equivariances directly into Riemannian 
    
[^428]: 科学反馈揭示了超越训练惯例的高产区域

    Scientific feedback reveals productive regimes beyond training conventions

    [https://arxiv.org/abs/2512.15756](https://arxiv.org/abs/2512.15756)

    科学性能反馈能够引导生成模型自主探索超越人类设计惯例的高产区域，在提升科学性能的同时生成训练数据中被排除的选择，且无需显式奖励对惯例的偏离。

    

    人为设定的设计惯例可能会限制科学训练数据中所呈现的备选方案，并塑造生成模型学习到的搜索偏好。我们证明，科学性能反馈可以将学习到的搜索分布重新引导至这些惯例之外的高产区域，而无需明确奖励对这些惯例的偏离。我们在两个场景中测试了这一过程：使用模拟反馈的反应堆栅格设计，以及使用实测产率的化学反应条件选择，分别采用预训练的序列策略和一个小型非语言策略。在这两种设置中，自适应过程在提升科学性能的同时，增加了被相应监督数据所排除的选择的生成。在实测反应中，随着预期产率误差的降低，被排除在训练之外的任务特定配体选择的概率从0.29上升到0.70-0.71。栅格设置中的对照实验表明，高产区域……（摘要在此处被截断）

    arXiv:2512.15756v2 Announce Type: replace-cross  Abstract: Human-imposed design conventions can restrict the alternatives represented in scientific training data and shape the search preferences learned by generative models. We show that scientific performance feedback can redirect the learned search distribution toward productive regions beyond those conventions, without explicitly rewarding departures from them. We test this process in reactor lattice design with simulated feedback and chemical reaction-condition selection with measured yields, using a pretrained sequence policy and a small non-language policy, respectively. In both settings, adaptation improves scientific performance while increasing generation of choices excluded from the corresponding supervised data. In measured reactions, probability on task-specific ligand choices excluded from training rises from 0.29 to 0.70-0.71 as expected yield error decreases. Controls in the lattice setting show that the productive regio
    
[^429]: CarBench：高保真三维汽车空气动力学神经替代模型的综合基准

    CarBench: A Comprehensive Benchmark for Neural Surrogates on High-Fidelity 3D Car Aerodynamics

    [https://arxiv.org/abs/2512.07847](https://arxiv.org/abs/2512.07847)

    本文提出了CarBench，首个针对大规模三维汽车空气动力学的综合基准，在包含8000多个高保真模拟的DrivAerNet++数据集上系统评估了十一种最先进的神经替代模型架构。

    

    基准测试一直是计算机视觉、自然语言处理以及更广泛深度学习领域进步的基石，通过标准化数据集和可复现的评估协议推动算法创新。大规模计算流体力学（CFD）数据集的日益普及，为将机器学习应用于空气动力学和工程设计开辟了新机遇。然而，尽管取得了这些进展，工程设计中的大规模数值模拟仍缺乏标准化基准。在本工作中，我们引入了CarBench，这是首个专为大规模三维汽车空气动力学设计的综合基准，对DrivAerNet++（最大的公开汽车空气动力学数据集，包含超过8000个高保真汽车模拟）上的最先进模型进行了大规模评估。我们评估了十一种架构，涵盖神经算子方法（如傅里叶神经算子）。

    arXiv:2512.07847v2 Announce Type: replace  Abstract: Benchmarking has been the cornerstone of progress in computer vision, natural language processing, and the broader deep learning domain, driving algorithmic innovation through standardized datasets and reproducible evaluation protocols. The growing availability of large-scale Computational Fluid Dynamics (CFD) datasets has opened new opportunities for applying machine learning to aerodynamic and engineering design. Yet, despite this progress, there exists no standardized benchmark for large-scale numerical simulations in engineering design. In this work, we introduce CarBench, the first comprehensive benchmark dedicated to large-scale 3D car aerodynamics, performing a large-scale evaluation of state-of-the-art models on DrivAerNet++, the largest public dataset for automotive aerodynamics, containing over 8,000 high-fidelity car simulations. We assess eleven architectures spanning neural operator methods (e.g., Fourier Neural Operator
    
[^430]: AlignSAE：概念对齐的稀疏自编码器

    AlignSAE: Concept-Aligned Sparse Autoencoders

    [https://arxiv.org/abs/2512.02004](https://arxiv.org/abs/2512.02004)

    AlignSAE通过“先预训练、后后训练”的两阶段课程，将稀疏自编码器的部分潜变量与预定义概念对齐并绑定到专用槽位，同时保留其余容量进行通用重构，从而实现对大语言模型内部特定概念的无干扰检查与精确控制。

    

    大语言模型（LLM）将事实知识编码在难以检查或控制的隐藏参数空间中。尽管稀疏自编码器（SAE）能够将隐藏激活分解为更细粒度、可解释的特征，但其往往难以将这些特征与人类定义的概念可靠对齐，导致特征表示相互纠缠且分布弥散。为解决这一问题，我们提出了AlignSAE，一种通过“先预训练、后后训练”的课程策略将SAE特征与预定义本体对齐的方法。在初始的无监督训练阶段之后，我们采用有监督后训练将特定概念绑定到专用的潜在槽位，同时保留其余容量用于一般性重构。这种分离构建了一个可解释的接口，使得可以在不受无关特征干扰的情况下对特定概念进行检查与控制。在GPT-2上针对合成生物（传记类）数据的实证结果表明……（原文摘要在此处被截断）

    arXiv:2512.02004v4 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) encode factual knowledge within hidden parametric spaces that are difficult to inspect or control. While Sparse Autoencoders (SAEs) can decompose hidden activations into more fine-grained, interpretable features, they often struggle to reliably align these features with human-defined concepts, resulting in entangled and distributed feature representations. To address this, we introduce AlignSAE, a method that aligns SAE features with a predefined ontology through a "pre-train, then post-train" curriculum. After an initial unsupervised training phase, we apply supervised post-training to bind specific concepts to dedicated latent slots while preserving the remaining capacity for general reconstruction. This separation creates an interpretable interface where specific concepts can be inspected and controlled without interference from unrelated features. Empirical results on GPT-2 across synthetic biog
    
[^431]: SUPN：用于高效函数逼近的浅层通用多项式网络

    SUPN: Shallow Universal Polynomial Networks for Efficient Function Approximation

    [https://arxiv.org/abs/2511.21414](https://arxiv.org/abs/2511.21414)

    提出浅层通用多项式网络（SUPN），通过在每个激活函数内部嵌入可学习的多元多项式（多项式预激活），以单个隐藏层实现更强的表达能力，用更少的参数和更低的训练成本完成高效函数逼近。

    

    深度神经网络（DNN）和Kolmogorov-Arnold网络（KAN）是灵活的函数逼近器，但其精度往往需要较大的网络深度、宽度和可训练参数量。这不仅使所得网络缺乏透明度，庞大的参数空间还会增加训练成本，并产生对优化过程和参数初始化敏感的解，从而导致泛化误差的不稳定性。基于这些原因，我们提出了浅层通用多项式网络（SUPNs），它在保留单个隐藏层的同时，通过在每个激活函数内部嵌入一个可学习的多元多项式（我们称之为多项式预激活）来提高每个单元的表达能力。这种构造以架构的深度和宽度换取每个隐藏单元内部更强的表达能力，在保持浅层计算图的同时，降低了参数数量和训练成本。我们建立了两个互补的通用逼近理论……（原文摘要在此处截断）

    arXiv:2511.21414v2 Announce Type: replace  Abstract: Deep neural networks (DNNs) and Kolmogorov-Arnold networks (KANs) are flexible function approximators, but their accuracy can require substantial network depth, width, and trainable parameter counts. Beyond making the resulting networks less transparent, large parameter spaces can increase training cost and produce solutions that are sensitive to optimization and parameter initialization, leading to variability in generalization error. For these reasons, we propose Shallow Universal Polynomial Networks (SUPNs), which retain a single hidden layer while increasing the expressivity of each unit through a learnable multivariate polynomial inside each activation, which we refer to as a polynomial pre-activation. This construction trades architectural depth and width for greater expressivity within each hidden unit, reducing parameter counts and training cost while preserving a shallow computational graph. We establish two complementary un
    
[^432]: 图同态失真：一种面向属性图的结构化（伪）度量

    Graph Homomorphism Distortion: A structural (pseudo)-metric for attributed graphs

    [https://arxiv.org/abs/2511.03068](https://arxiv.org/abs/2511.03068)

    本文提出一种基于图同态的新型（伪）度量——图同态失真，通过衡量图间映射时节点特征的最小最坏失真来实现属性图的高效、细粒度区分，并可作为结构编码将基于同态的描述符扩展到属性图。

    

    图学习的复杂性很大程度上源于结构与特征之间的相互作用。然而，在分析图神经网络表达能力时，现有方法往往忽略特征而只关注结构，这使得几乎无法评估两个特征相近的图应在多大程度上被视为相似。为解决这一问题，我们基于图同态提出了一种新的（伪）度量。受度量几何概念的启发，我们的图同态失真度量的是将一个图映射到另一个图时，节点特征所承受的最小最坏情况失真。我们通过以下三点展示了该（伪）度量的实用性：i.) 它可以被高效近似，同时保持区分能力；ii.) 它提供了一种细粒度的图区分方法；iii.) 它可作为结构编码使用，将基于同态的描述符扩展至属性图。

    arXiv:2511.03068v5 Announce Type: replace  Abstract: A large driver of the complexity of graph learning is the interplay between structure and features. When analyzing the expressivity of graph neural networks, however, existing approaches ignore features in favor of structure, making it nigh-impossible to assess to what extent two graphs with close features should be considered similar. We address this by developing a new (pseudo-)metric based on graph homomorphisms Inspired by concepts from metric geometry, our graph homomorphism distortion measures the minimal worst-case distortion that node features of one graph are subjected to when mapping one graph to another. We demonstrate the utility of our (pseudo-)metric by showing that i.) it can be efficiently approximated while retaining distinction capabilities, ii.) it provides a granular approach to graph distinction, and, iii.) it serves as a structural encoding, extending the concept of homomorphism-based descriptors to attributed g
    
[^433]: 正则拟阵的算术电路与神经网络

    Arithmetic Circuits and Neural Networks for Regular Matroids

    [https://arxiv.org/abs/2511.02406](https://arxiv.org/abs/2511.02406)

    本文证明了存在大小为 $O(n^3)$ 的均匀算术电路和 ReLU 神经网络可用于正则拟阵的基生成多项式计算与加权基最大化，并首次展示了虚拟扩展公式可以比已知最佳的单个扩展公式更高效。

    

    我们证明了存在大小为 $O(n^3)$ 的均匀 $(+,\times,/)$-电路，可用于计算 $n$ 元正则拟阵的基生成多项式。通过热带化，这意味着存在同样大小的均匀 $(\max,+,-)$-电路和 ReLU 神经网络，可用于正则拟阵的加权基最大化问题。作为在线性规划理论中的一个推论，我们得到了首个例子，表明两个扩展公式之差可以比目前已知的最佳单个扩展公式更为高效，即 Aprile 和 Fiorini 提出的正则拟阵基多面体的大小为 $O(n^6)$ 的扩展公式。这种差最近被引入并称为虚拟扩展公式。我们主要结果的证明依赖于对 Seymour 正则拟阵分解定理的一个精细调校版本，该版本使我们能够识别并维护图子结构，从而对其应用星-网变换的局部版本。

    arXiv:2511.02406v2 Announce Type: replace-cross  Abstract: We prove that there exist uniform $(+,\times,/)$-circuits of size $O(n^3)$ to compute the basis generating polynomial of regular matroids on $n$ elements. By tropicalization, this implies that there exist uniform $(\max,+,-)$-circuits and ReLU neural networks of the same size for weighted basis maximization of regular matroids. As a consequence in linear programming theory, we obtain a first example where taking the difference of two extended formulations can be more efficient than the best known individual extended formulation for the base polytope of regular matroids of size $O(n^6)$ by Aprile and Fiorini. Such differences have recently been introduced as virtual extended formulations. The proof of our main result relies on a fine-tuned version of Seymour's decomposition of regular matroids which allows us to identify and maintain graphic substructures to which we can apply a local version of the star-mesh transformation.
    
[^434]: 多任务神经扩散过程

    Multi-task Neural Diffusion Processes

    [https://arxiv.org/abs/2510.03419](https://arxiv.org/abs/2510.03419)

    提出多任务神经扩散过程（MT-NDPs），通过任务编码器提取任务表示并条件化扩散模型，实现跨任务信息共享的任务条件化概率回归与少样本自适应。

    

    神经扩散过程为函数分布建模提供了一种可扩展的非高斯方法，但现有形式仅限于单任务推理，无法捕捉相关任务之间的依赖关系。我们提出了多任务神经扩散过程（MT-NDPs），这是一种扩展方法，通过引入任务编码器来实现任务条件化的概率回归以及跨相关函数的少样本自适应。任务编码器从上下文观测中提取低维表示，并以此表示为条件来约束扩散模型，从而实现跨任务的信息共享，同时支持大小可变的上下文集合，并保持对上下文观测顺序的置换不变性。我们通过合成实验以及两个真实世界应用（涉及风力涡轮机数据和学校考试结果）对MT-NDPs进行了评估。结果表明性能得到了改进。

    arXiv:2510.03419v3 Announce Type: replace-cross  Abstract: Neural diffusion processes provide a scalable, non-Gaussian approach to modelling distributions over functions, but existing formulations are limited to single-task inference and do not capture dependencies across related tasks. We propose multi-task neural diffusion processes (MT-NDPs), an extension that incorporates a task encoder to enable task-conditioned probabilistic regression and few-shot adaptation across related functions. The task encoder extracts a low-dimensional representation from context observations and conditions the diffusion model on this representation, allowing information sharing across tasks while accommodating variable-sized context sets and preserving permutation invariance with respect to the ordering of context observations. We evaluate MT-NDPs through synthetic experiments and two real-world applications involving wind turbine data and school examination outcomes. The results demonstrate improved pr
    
[^435]: 无思考策略初始化使蒸馏推理模型成为更有效且更高效的推理器

    Thinking-Free Policy Initialization Makes Distilled Reasoning Models More Effective and Efficient Reasoners

    [https://arxiv.org/abs/2509.26226](https://arxiv.org/abs/2509.26226)

    提出无思考策略初始化（TFPI），通过在RLVR训练中显式丢弃思考内容，使蒸馏推理模型在提升性能的同时降低token消耗，架起了长思维链蒸馏与标准强化学习之间的桥梁。

    

    基于可验证奖励的强化学习（RLVR）能够有效解决复杂任务，但在训练过程中需要极长的上下文长度，从而导致巨大的计算开销。虽然多阶段训练可以在一定程度上缓解这一问题，但若从过短的上下文开始训练，往往会造成不可逆的性能退化，最终无法显著减少整体训练计算量。本文提出了无思考策略初始化，这是一种简单而有效的对RLVR的改进方法，在长思维链（CoT）蒸馏与标准RLVR之间架起了桥梁。TFPI采用一种简单的“无思考”操作，通过直接append的方式显式丢弃思考内容，从而减少推理时的token使用量。使用经过ThinkingFree适配的输入进行训练，即使在原始的慢思考模式下，也能提升性能并降低token消耗。在多个基准测试上的大量实验表明，TFPI……（摘要原文在此处截断）

    arXiv:2509.26226v3 Announce Type: replace-cross  Abstract: Reinforcement Learning with Verifiable Reward (RLVR) effectively solves complex tasks but demands extremely long context lengths during training, leading to substantial computational costs. While multi-stage training can partially mitigate this, starting with overly short contexts often causes irreversible performance degradation, ultimately failing to reduce overall training compute significantly. In this paper, we introduce Thinking-Free Policy Initialization (TFPI), a simple yet effective adaptation to RLVR that bridges long Chain-of-Thought (CoT) distillation and standard RLVR. TFPI employs a simple ThinkingFree operation, explicitly discarding the thinking content via a direct  append, to reduce token usage during inference. Training with ThinkingFree-adapted inputs improves performance and lowers token consumption, even in the original slow-thinking mode. Extensive experiments across various benchmarks have shown that TFP
    
[^436]: 用于融合神经算子与经典方法的贪心PDE路由器

    A Greedy PDE Router for Blending Neural Operators and Classical Methods

    [https://arxiv.org/abs/2509.24814](https://arxiv.org/abs/2509.24814)

    本文提出一种近似贪心PDE路由器，在混合迭代求解过程中每步智能选择神经算子或经典数值求解器以融合两者优势，在多种PDE求解任务中持续缩短求解时间。

    

    在求解偏微分方程（PDE）时，经典数值求解器通常计算开销较大，而机器学习方法则可能受到谱偏差的影响，无法捕捉高频分量。设计一种最优的混合迭代求解器——即在每次迭代时从求解器集合中选择一个求解器，以利用它们之间互补的优势——构成了一个具有挑战性的组合优化问题。虽然贪心选择方法因其在Lipschitz假设下对最优解具有常数因子近似保证而备受青睐，但它需要知道每一步的真实误差，而这在实践中是无法获得的。我们通过提出一种近似贪心路由器来解决这一问题，该路由器能够高效地模拟求解器选择中的贪心策略。在泊松方程、对流-扩散方程以及各向异性扩散方程上的实证结果表明，与单一求解器基线和现有方法相比，我们的方法能够持续减少求解时间。

    arXiv:2509.24814v3 Announce Type: replace-cross  Abstract: When solving PDEs, classical numerical solvers are often computationally expensive, while machine learning methods can suffer from spectral bias, failing to capture high-frequency components. Designing an optimal hybrid iterative solver--where, at each iteration, a solver is selected from an ensemble of solvers to leverage their complementary strengths--poses a challenging combinatorial problem. While greedy selection is desirable for its constant-factor approximation guarantee to the optimal solution under Lipschitz assumptions, it requires knowledge of the true error at each step, which is unavailable in practice. We address this by proposing an approximate greedy router that efficiently mimics a greedy approach to solver selection. Empirical results on the Poisson, convection-diffusion, and anisotropic diffusion equations show that our method consistently reduces time-to-solution relative to single-solver baselines and exist
    
[^437]: 次线性时间中的质量控制：以随机图为案例研究

    Quality control in sublinear time: a case study via random graphs

    [https://arxiv.org/abs/2508.16531](https://arxiv.org/abs/2508.16531)

    该论文提出了一类新的算法问题——“质量控制问题”，研究如何在次线性时间内判断任意输入是否为高质量（即能否信任算法在该输入上的表现），并以随机图作为案例进行研究。

    

    许多算法被设计为在输入总体上平均表现良好。但当这类算法在某个任意输入上运行时，问题随之而来：我们能在这个输入上信任该算法吗？这一问题在网络科学领域尤为相关，因为基于理论随机图模型的算法常常被应用于单个真实世界的网络。我们识别出一类解决此问题的新型算法问题，称之为“质量控制问题”。这类问题由一个“质量函数”$\rho$和一个分布$D$定义，使得从$D$中抽取的样本以高概率为“高质量”，即其$\rho$值接近1。目标是接受大多数来自$x \sim D$的输入，同时拒绝可能以对抗方式生成的、其$\rho(x)$远离1的输入$x$。在本工作中，我们考虑质量控制问题的次线性版本，其中$D \in \Delta(\{0,1\}^N)$，目标是求解$(D,\rho)$问题……

    arXiv:2508.16531v3 Announce Type: replace-cross  Abstract: Many algorithms are designed to work well on average over inputs. But when such an algorithm is run on an arbitrary input, the question arises: Can we trust the algorithm on this input? This question is especially relevant in network science, where algorithms for theoretical random graph models are often applied to individual real-world networks.   We identify a new class of algorithmic problems addressing this, which we call "Quality Control Problems." These problems are specified by a "quality function" $\rho$ and a distribution $D$ such that, with high probability, a sample drawn from $D$ is "high quality," meaning its $\rho$-value is near $1$. The goal is to accept most inputs $x \sim D$ and reject potentially adversarially generated inputs $x$ with $\rho(x)$ far from $1$.   In this work, we consider the sublinear version of the quality control problem, where $D \in \Delta(\{0,1\}^N)$ and the goal is to solve the $(D,\rho)$
    
[^438]: 深度神经网络初始化方差的最优条件：基于SGD动力学的视角

    Optimal Condition for Initialization Variance in Deep Neural Networks: An SGD Dynamics Perspective

    [https://arxiv.org/abs/2508.12834](https://arxiv.org/abs/2508.12834)

    该论文通过将SGD近似为朗之万动力学并利用Fokker-Planck方程与KL散度，建立了期望损失与初始化分布的联系，从而推导出深度神经网络初始化方差的最优条件。

    

    随机梯度下降（SGD）是机器学习中最基础的优化算法之一，通过连续时间近似可以将其重新表述为描述朗之万动力学的Fokker-Planck方程，这一视角已启发了众多理论研究。在该框架下，我们通过Kullback-Leibler（KL）散度研究了由该方程导出的准平稳分布与初始分布之间的关系。由于准稳态分布依赖于期望代价函数，KL散度最终揭示了期望代价函数与初始化分布之间的联系。将该方法应用于深度神经网络（DNN）模型后，我们可以用初始化参数显式地表达期望损失函数的界限。进而，通过最小化该界限，我们得到了高斯（初始化方差）的最优条件……

    arXiv:2508.12834v2 Announce Type: replace-cross  Abstract: Stochastic gradient descent (SGD), one of the most fundamental optimization algorithms in machine learning (ML), can be recast through a continuous-time approximation as a Fokker-Planck equation for Langevin dynamics, a viewpoint that has motivated many theoretical studies. Within this framework, we study the relationship between the quasi-stationary distribution derived from this equation and the initial distribution through the Kullback-Leibler (KL) divergence. As the quasi-steady-state distribution depends on the expected cost function, the KL divergence eventually reveals the connection between the expected cost function and the initialization distribution. By applying this to deep neural network models (DNNs), we can express the bounds of the expected loss function explicitly in terms of the initialization parameters. Then, by minimizing this bound, we obtain an optimal condition of the initialization variance in the Gauss
    
[^439]: 多模态远程推理

    Multimodal Remote Inference

    [https://arxiv.org/abs/2508.07555](https://arxiv.org/abs/2508.07555)

    该论文针对多模态远程推理系统提出了一种以最小化机器学习推理误差为目标的多模态调度问题，将推理误差建模为信息年龄向量的函数，通过半马尔可夫决策过程求解，并揭示了双模态与多模态情形下最优策略链结构的本质差异。

    

    我们研究了一个多模态远程推理系统，其中多模态机器学习（ML）模型利用从远程传感器收集的特征进行实时推理。当传感器观测随时间动态变化时，新鲜的特征对于推理任务至关重要。然而，在有限的网络资源下，及时传输所有模态的特征通常是不可行的。为应对这一挑战，我们构建了一个多模态调度问题，以最小化机器学习模型的推理误差。我们将该误差建模为信息年龄（AoI）向量的一般函数，其中AoI量化数据的新鲜程度。我们将该问题表述为半马尔可夫决策过程（SMDP），并推导出一种具有缩减状态集的等价重构形式。随后我们证明，该问题在双模态和多模态两种情形下具有根本不同的链结构。对于双模态情形，我们证明最优策略具有……

    arXiv:2508.07555v4 Announce Type: replace  Abstract: We consider a remote inference system with multiple modalities, where a multimodal machine learning (ML) model performs real-time inference using features collected from remote sensors. When sensor observations evolve dynamically over time, fresh features are critical for inference tasks. However, timely delivery of features from all modalities is often infeasible under limited network resources. To address this challenge, we formulate a multimodal scheduling problem to minimize the ML model's inference error. We model this error as a general function of the Age of Information (AoI) vector, where AoI quantifies data freshness. We cast the problem as a semi-Markov decision process (SMDP) and derive an equivalent reformulation with a reduced state set. We then show that the problem has fundamentally different chain structures in the two-modality and multi-modality cases. For the two-modality case, we prove that the optimal policy has a
    
[^440]: 用于构建虚拟细胞模型的AI科学家

    AI Scientists for Building Virtual-Cell Models

    [https://arxiv.org/abs/2508.02276](https://arxiv.org/abs/2508.02276)

    本文提出CellForge，一个由多个自主智能体组成的有组织团队，能够将真实复杂的单细胞扰动数据自动转化为可执行的模型假设与架构，实现了从文献调研、数据分析到模型构建的全流程自动化AI科学家系统。

    

    从虚拟实验室到生物数据分析智能体和“共同科学家”，自主智能体已经开始将科学目标转化为假设。然而，它们能否更进一步，将真实的、含噪声的、复杂的生物数据转化为可执行的模型假设以及编码了可检验假设的模型架构，仍不清楚。预测细胞如何响应基因和化学扰动是对这种能力的一个天然测试：这一任务对解读疾病机制和治疗发现至关重要，但需要极少能集中于单一研究团队的跨学科专业知识。在此，我们介绍CellForge——一个用于单细胞扰动建模的有组织自主智能体团队。给定一个数据集和一项任务，它会调研文献、解析数据、识别相关基线方法，并通过专门负责数据建模、单细胞生物学等领域的智能体之间由批评者中介的讨论，形成模型假设。

    arXiv:2508.02276v3 Announce Type: replace-cross  Abstract: Autonomous agents, from virtual laboratories to biological data-analysis agents and co-scientists, have begun to turn scientific goals into hypotheses. Whether they can go further and convert real, noisy and complex biological data into executable model hypotheses and architectures that encode testable assumptions remains unclear. Predicting how cells respond to genetic and chemical perturbations is a natural test of this ability: it is central to decoding disease mechanisms and therapeutic discovery, yet it demands cross-disciplinary expertise rarely concentrated in a single research group. Here, we introduce CellForge, an organized team of autonomous agents for single-cell perturbation modeling. Given a dataset and a task, it surveys the literature, parses the data, identifies relevant baselines, and formulates a model hypothesis through critic-mediated discussion among agents specialized in data modeling, single-cell biology
    
[^441]: 基于总风险先验的正式贝叶斯迁移学习

    Formal Bayesian Transfer Learning via the Total Risk Prior

    [https://arxiv.org/abs/2507.23768](https://arxiv.org/abs/2507.23768)

    该论文提出了一种新的贝叶斯迁移学习方法，其核心创新在于使用以源参数为条件的风险最小化器（而非传统的经验损失最小化器）来构建源数据集与目标数据集所有参数的单一联合先验分布，从而实现完整的贝叶斯不确定性量化以及通过Gibbs采样对源数据集进行选择和模型平均。

    

    现有的迁移学习方法难以应对源数据集有限且无法保证与目标数据集良好对齐的情况。一种典型的策略是将源数据上的经验损失最小化器用作目标参数的先验均值。我们的关键概念性贡献是改为使用以源参数为条件的风险最小化器。这使得我们能够基于源数据集和目标数据集为所有参数构建一个单一的联合先验分布。因此，我们可以获得完整的贝叶斯不确定性量化，并可以通过对控制每个源数据集是否被纳入的指示变量进行吉布斯采样（Gibbs sampling）来执行模型平均。我们展示了该先验的一种特定实例化如何在变换后的坐标系中导出贝叶斯Lasso，并讨论了将我们的方法扩展到中等规模数据集的计算技术。我们讨论了……

    arXiv:2507.23768v2 Announce Type: replace-cross  Abstract: Existing methods for transfer learning struggle to deal with situations where the source datasets are limited and not guaranteed to be well-aligned with the target dataset. A typical strategy is to use the empirical loss minimizer on the source data as a prior mean for the target parameters. Our key conceptual contribution is to use a risk minimizer conditional on source parameters instead. This allows us to construct a single joint prior distribution for all parameters from the source datasets as well as the target dataset. As a consequence, we benefit from full Bayesian uncertainty quantification and can perform model averaging via Gibbs sampling over indicator variables governing the inclusion of each source dataset. We show how a particular instantiation of our prior leads to a Bayesian Lasso in a transformed coordinate system and discuss computational techniques to scale our approach to moderately sized datasets. We discus
    
[^442]: 稳定模态间隙：面向视觉-语言模型的零空间投影持续学习

    Stabilizing the Modality Gap: Null-Space Projected Continual Learning for Vision-Language Models

    [https://arxiv.org/abs/2507.19839](https://arxiv.org/abs/2507.19839)

    提出FPCD框架，通过快速零空间投影减少对已学习表示的干扰，并结合对比蒸馏对齐跨模态映射，从而稳定视觉-语言模型持续学习中的模态间隙漂移，同时将投影基构建速度提升5-9倍。

    

    模态间隙是多模态对齐的固有特性，即来自不同模态的嵌入在共享表示空间中仍保持部分分离。在持续学习中，基于CLIP的视觉-语言模型（VLMs）不仅遭受灾难性遗忘，还会经历模态间隙漂移，破坏共享嵌入的几何结构。这种漂移会降低跨模态对齐能力和泛化能力。我们提出了FPCD，一个结合快速零空间投影（FNP）和对比蒸馏（CD）的高效框架。FNP累积先前任务激活的紧凑Gram矩阵，并将梯度投影到层自适应的近似零空间上，以减少对已学习表示的干扰。通过用对称特征分解替代奇异值分解，FNP将投影基的构建速度提升了5-9倍。CD使用小型参考集来对齐双向的图像到文本和文本到图像的（跨模态映射，摘要此处截断）。

    arXiv:2507.19839v2 Announce Type: replace  Abstract: The modality gap is intrinsic to multimodal alignment, where embeddings from different modalities remain partially separated in a shared representation space. In continual learning, CLIP-based vision-language models (VLMs) not only suffer catastrophic forgetting but also experience modality gap drift that disrupts the shared embedding geometry. Such drift degrades cross-modal alignment and generalization. We propose FPCD, an efficient framework combining Fast Null-space Projection (FNP) and Contrastive Distillation (CD). FNP accumulates compact Gram matrices of previous tasks' activations and projects gradients onto layer-adaptive approximate null spaces to reduce interference with learned representations. By replacing singular value decomposition with symmetric eigendecomposition, FNP accelerates projection-basis construction by a factor of 5-9. CD uses a small reference set to align the bidirectional image-to-text and text-to-image
    
[^443]: PAC因果可学习性：从识别到有限样本保证

    PAC-Causal Learnability: From Identification to Finite-Sample Guarantees

    [https://arxiv.org/abs/2507.18903](https://arxiv.org/abs/2507.18903)

    该论文提出了PACC因果可学习性框架，首次从PAC学习角度为因果估计提供了有限样本的统一精度-置信度保证，并揭示其样本复杂度由因果识别强度、目标复杂度和干扰复杂度三个关键因素决定。

    

    因果识别探讨的是在理想化的无限数据设定下，能否从观测分布中恢复出因果目标。在实践中，只有有限样本可用，而可识别性本身并不能确定可靠估计需要多少数据。我们引入了“可能近似正确因果”（PACC）可学习性，它在因果模型类上形式化了统一的精度-置信度保证。从PAC学习的视角出发，我们探究需要多少样本才能以至少1-δ的概率将因果目标估计到精度ε。我们证明PACC样本复杂度由三个反复出现的因素决定：因果识别强度、目标复杂度和干扰复杂度，并推导出一个将三者与有限样本因果学习联系起来的通用充分界。我们通过自身对照病例系列、工具变量和倾向性（评分）方法来阐述该框架。

    arXiv:2507.18903v2 Announce Type: replace-cross  Abstract: Causal identification asks whether a causal target can be recovered from the observational distribution in an idealized infinite-data setting. In practice, only finite samples are available, and identifiability alone does not determine how much data are needed for reliable estimation. We introduce Probably Approximately Correct Causal (PACC) learnability, which formalizes uniform accuracy-confidence guarantees over a causal model class. From a PAC-learning perspective, we ask how many samples are sufficient to estimate a causal target to accuracy $\epsilon$ with probability at least $1-\delta$. We show that PACC sample complexity is shaped by three recurring factors: causal identification strength, target complexity, and nuisance complexity, and derive a general sufficient bound linking them to finite-sample causal learning. We illustrate the framework through self-controlled case series, instrumental variables, and propensity-
    
[^444]: PanoDiff-SR：利用扩散模型与超分辨率技术合成牙科全景X光片

    PanoDiff-SR: Synthesizing Dental Panoramic Radiographs using Diffusion and Super-resolution

    [https://arxiv.org/abs/2507.09227](https://arxiv.org/abs/2507.09227)

    本文提出PanoDiff-SR，一种结合扩散模型与超分辨率的端到端两阶段流水线，用于生成完整且达到诊断分辨率的全尺寸牙科全景X光片，克服了基于GAN方法存在的视野受限、易被临床医生识别以及训练不稳定等局限。

    

    近年来，人们对生成逼真合成医学图像的兴趣日益增长。此类数据集可以缓解人工智能研究中公共数据稀缺的问题，并支持医学教育。迄今为止，合成牙科全景X光片（PR）的工作几乎完全依赖于生成对抗网络（GAN），且反复出现三个局限：合成的视野通常仅限于裁剪的牙槽区域，而非完整的全景X光片；针对GAN生成全景X光片的观察者研究表明，临床医生仍能以约78-82%的准确率识别出合成图像，这意味着合成图像始终存在可见的破绽；此外，对抗训练仍然对不稳定性和模式崩溃敏感。使用扩散模型直接以诊断分辨率生成全尺寸全景X光片是一种替代方案，但在该分辨率下采样成本高昂。因此，我们提出了PanoDiff-SR，一个端到端的两阶段流水线……

    arXiv:2507.09227v2 Announce Type: replace-cross  Abstract: Interest in generating realistic synthetic medical images has grown in recent years. Such datasets can mitigate the scarcity of public data for artificial intelligence research and support education. Work on synthesizing dental panoramic radiographs (PRs) has so far relied almost exclusively on generative adversarial networks (GANs), and three limitations recur: the synthesized field of view is often restricted to a cropped dentoalveolar region rather than a complete PR; observer studies of GAN-generated PRs report that clinicians still identify synthetic images with accuracies of roughly 78-82%, implying persistent visible giveaways; and adversarial training remains sensitive to instability and mode collapse. Generating a full-size PR directly at diagnostic resolution with a diffusion model is an alternative, but sampling at that resolution is costly. We therefore present PanoDiff-SR, an end-to-end two-stage pipeline developed
    
[^445]: 基于博弈论社会影响加权与深度学习的游客到达量预测

    Tourist arrival forecasting with game-theoretic social influence weighting and deep learning

    [https://arxiv.org/abs/2507.03411](https://arxiv.org/abs/2507.03411)

    该论文提出一种融合合作博弈论意见领袖识别（其Shapley值具有闭式解且最大协同联盟可在多项式时间内精确求解）、因果经验小波变换与贝叶斯优化调优的堆叠BiLSTM网络的混合预测方法，实现了对香港和三亚游客到达量的准确预测。

    

    在工业5.0时代，数据驱动的决策对于旅游等服务系统的规划已变得不可或缺。本文提出了一种非线性混合方法，用于预测新冠疫情前来自五大主要客源市场赴香港的游客到达量，以及穿越新冠疫情冲击期间赴中国三亚的游客到达量。该方法整合了多个互联网大数据来源，并采用合作博弈论算法识别社交媒体上的意见领袖联盟，使其发布的内容获得更大的权重。因果地应用经验小波变换（EWT）来处理非平稳特性，并通过贝叶斯优化调优的堆叠双向长短期记忆网络（Stacked BiLSTM）生成预测结果。我们证明了底层信任加权覆盖博弈的Shapley值具有闭式解，且最大协同效应联盟可以在多项式时间内被精确找到。

    arXiv:2507.03411v2 Announce Type: replace  Abstract: In the era of Industry 5.0, data-driven decision-making has become indispensable for planning in service systems such as tourism. This paper proposes a non-linear hybrid approach for forecasting tourist arrivals to Hong Kong from five major source markets before COVID-19 and to Sanya, China, through the COVID-19 shock. The method integrates multiple sources of Internet big data and employs a cooperative game-theoretic algorithm to identify coalitions of opinion leaders on social media, whose content then receives larger weights. The Empirical Wavelet Transform (EWT), applied causally, manages nonstationary attributes, and a Stacked Bi-directional Long Short-Term Memory (Stacked BiLSTM) network tuned by Bayesian optimization generates the forecasts. We prove that the Shapley values of the underlying trust-weighted coverage game have a closed form and that the coalition of maximal synergy can be found exactly in polynomial time. Relati
    
[^446]: 基于KL散度最小化的高斯混合模型差分隐私分布发布

    Differentially Private Distribution Release of Gaussian Mixture Models via KL-Divergence Minimization

    [https://arxiv.org/abs/2506.03467](https://arxiv.org/abs/2506.03467)

    本文提出了一种以KL散度为效用度量驱动的优化框架，在差分隐私保证下发布高斯混合模型的混合权重、分量均值和协方差矩阵，以防止底层数据敏感信息的泄露。

    

    高斯混合模型（GMM）是广泛用于表示多模态数据分布的统计模型，在数据挖掘、模式识别、数据模拟和机器学习等领域有众多应用。然而，近期研究表明，发布GMM参数存在显著的隐私风险，可能暴露底层数据中的敏感信息。本文研究了在确保差分隐私（DP）保证的前提下发布GMM参数这一挑战。具体而言，我们专注于混合权重、分量均值和协方差矩阵的隐私保护。我们提出使用Kullback-Leibler（KL）散度作为效用度量来评估所发布GMM的准确性，因为它能够捕捉噪声扰动对所有模型参数的联合影响。我们开发了一个以KL散度驱动的差分隐私GMM发布优化框架，联合设计隐私……

    arXiv:2506.03467v4 Announce Type: replace-cross  Abstract: Gaussian Mixture Models (GMMs) are widely used statistical models for representing multi-modal data distributions, with numerous applications in data mining, pattern recognition, data simulation, and machine learning. However, recent research has shown that releasing GMM parameters poses significant privacy risks, potentially exposing sensitive information about the underlying data. In this paper, we address the challenge of releasing GMM parameters while ensuring differential privacy (DP) guarantees. Specifically, we focus on the privacy protection of mixture weights, component means, and covariance matrices. We propose to use Kullback-Leibler (KL) divergence as a utility metric to assess the accuracy of the released GMM, as it captures the joint impact of noise perturbation on all the model parameters. We develop a KL-divergence-driven optimization framework for differentially private GMM release that jointly designs privacy-
    
[^447]: HERO：在异构持续图学习中保持结构与语义

    HERO: Preserving Structure and Semantics in Heterogeneous Continual Graph Learning

    [https://arxiv.org/abs/2505.17458](https://arxiv.org/abs/2505.17458)

    该论文提出了HERO框架，通过将经验重放的误差分解为目标选择与上下文重构两部分，证明了仅重放目标节点是不够的，从而在异构持续图学习中同时从结构和语义两个层面保存历史知识。

    

    受现实世界数据中固有的多样实体、关系和语义的驱动，异构图上的机器学习发展迅速。然而，大多数现有研究假设图是静态的，而现实世界的图数据往往是持续更新的。这种设定下的持续学习尤其具有挑战性，因为历史知识不仅编码在目标节点的表示中，还编码在其异构的结构上下文和依赖关系的语义响应中。为此，我们提出了HERO，一个异构持续图学习框架，它在适应新任务的同时，在结构和语义两个层面上保存历史知识。我们将经验重放分析为对不可获取的历史梯度的近似，并将其误差分解为目标选择项和上下文重构项。这一分析表明，仅重放目标节点通常不足以……（摘要内容不完整，此处为原文截断处）

    arXiv:2505.17458v4 Announce Type: replace  Abstract: Machine learning on heterogeneous graphs has advanced rapidly, driven by the diverse entities, relations, and semantics inherent in real-world data. However, most existing studies assume static graphs, whereas real-world graph data are often continuously updated. Continual learning in this setting is particularly challenging because historical knowledge is encoded not only in target-node representations, but also in their heterogeneous structural contexts and relation-dependent semantic responses. To this end, we introduce HERO, a heterogeneous continual graph learning framework that preserves historical knowledge at both structural and semantic levels while adapting to incoming tasks. We analyze experience replay as an approximation to the unavailable historical gradient and decompose its error into target-selection and context-reconstruction terms. This analysis shows that replaying target nodes alone is generally insufficient for 
    
[^448]: 用于万能逼近的并行层归一化

    Parallel Layer Normalization for Universal Approximation

    [https://arxiv.org/abs/2505.13142](https://arxiv.org/abs/2505.13142)

    该论文证明仅含单个层归一化单元的浅层网络无论宽度多大都不具备万能逼近能力，而提出的并行层归一化网络（PLN-Nets）随着并行分组数量的增加即可实现万能逼近，且该机制还可推广到RMSNorm变体、逐位置前馈网络和卷积网络。

    

    层归一化通常因其 在优化中的作用而被研究，但它同时也在仿射层之间充当一种多元非线性。我们研究了当网络宽度增长时，LN 单元的分组方式如何影响神经网络的逼近能力。在并行层归一化网络中，固定维度的 LN 单元被独立地应用于两个仿射层之间的各个组。我们证明，仅含一个 LN 单元的浅层 LN-Net 无论宽度多大都不具备万能逼近性，而 PLN-Nets 则随着并行组数量的增加变得具有万能逼近性。该机制在合适的分组结构下可扩展到深层网络、均方根归一化 变体、逐位置前馈网络以及卷积网络。受控的多层感知机 (MLP) 与物理信息神经网络 (PINN) 实验支持了 PLN 的实际可行性，而 CNN 和 ViT 实验表明组数会塑造有限样本拟合包络，但 d……（原文摘要在此截断）

    arXiv:2505.13142v3 Announce Type: replace  Abstract: Layer Normalization (LN) is usually studied for its role in optimization, but it also acts as a multivariate nonlinearity between affine layers. We investigate how grouping LN units affects the approximation power of neural networks as width grows. In Parallel Layer Normalization networks (PLN-Nets), fixed-dimensional LN units are applied independently to groups between two affine layers. We prove that a shallow LN-Net with one LN unit is non-universal regardless of width, whereas PLN-Nets become universal as the number of parallel groups grows. The mechanism extends to deep networks, Root Mean Square Normalization (RMSNorm) variants, position-wise feed-forward networks, and convolutional networks under suitable grouping structures. Controlled MLP and physics-informed neural network (PINN) experiments support PLN's practical viability, while CNN and ViT experiments show that group count shapes the finite-sample fitting envelope but d
    
[^449]: ZenFlow：为大语言模型训练实现无停顿的卸载

    ZenFlow: Enabling Stall-Free Offloading for LLM Training

    [https://arxiv.org/abs/2505.12242](https://arxiv.org/abs/2505.12242)

    ZenFlow提出了一种新型卸载训练框架，通过利用重要梯度的时空局部性，在GPU上原位更新重要梯度、同时在CPU上异步累积次要梯度并完全重叠CPU工作与GPU计算，从而消除大语言模型训练中的GPU停顿。

    

    微调大语言模型（LLM）通常超出GPU内存限制，促使系统将模型状态卸载到CPU内存。然而，现有的卸载训练框架（如ZeRO-Offload）对所有参数一视同仁，并在CPU上更新整个模型，导致严重的GPU停顿——快速且昂贵的GPU处于闲置状态，等待缓慢的CPU更新和带宽受限的PCIe传输。我们提出了ZenFlow，这是一种新的卸载训练框架，通过优先处理重要梯度来解耦GPU与CPU之间的更新。ZenFlow在GPU上对重要梯度进行原位更新，同时在CPU上卸载并累积不重要的梯度，而不阻塞训练迭代，使CPU工作与GPU计算完全重叠。为了使这种重要性感知的设计具有可扩展性，ZenFlow利用了重要梯度的一种新颖的时空局部性——重要梯度集中于特定位置且在迭代过程中保持稳定——以...

    arXiv:2505.12242v4 Announce Type: replace-cross  Abstract: Fine-tuning large language models (LLMs) often exceeds GPU memory limits, prompting systems to offload model states to CPU memory. However, existing offloaded training frameworks like ZeRO-Offload treat all parameters equally and update the full model on the CPU, causing severe GPU stalls, where fast, expensive GPUs sit idle, waiting for slow CPU updates and limited-bandwidth PCIe transfers. We present ZenFlow, a new offloaded training framework that decouples updates between GPU and CPU by prioritizing important gradients. ZenFlow performs in-place updates of important gradients on GPU while offloading and accumulating less important ones on CPU without blocking training iterations, fully overlapping CPU work with GPU computation. To make this importance-aware design scalable, ZenFlow leverages a novel spatio-temporal locality of important gradients, concentrated in specific positions that remain stable across iterations, to m
    
[^450]: KnowEEG：可解释的知识驱动脑电图分类

    KnowEEG: Explainable Knowledge Driven EEG Classification

    [https://arxiv.org/abs/2505.00541](https://arxiv.org/abs/2505.00541)

    该论文提出KnowEEG可解释分类流程，通过结合单电极统计特征与电极间连接性度量并采用平衡采样策略，在五项脑电图分类任务上达到或超越最先进深度学习模型的性能，同时具备良好的可解释性。

    

    脑电图（EEG）分类在从神经退行性疾病检测到脑机接口（BCI）等诸多应用中展现出巨大前景。尽管近年来深度学习方法日益主导该研究领域，但这些模型在很大程度上仍是不透明的“黑箱”，可解释性有限。在本文中，我们通过一个系统性流程重新审视了基于特征的分类方法的价值，该流程旨在将单电极信号统计特征与跨大脑连接性度量相结合。所提出的"KnowEEG"流程提取了783个单电极统计特征以及电极间连接性统计特征，并在高维单电极特征与连接性特征之间采用平衡采样策略。该基于特征的流程在五项多样化的分类任务（包括情绪检测等）上取得了与最先进深度学习模型相当甚至更优的性能。

    arXiv:2505.00541v3 Announce Type: replace  Abstract: Electroencephalography (EEG) classification has shown significant promise in applications ranging from neurodegenerative disease detection to brain-computer interfaces (BCIs). Whilst deep learning methods have increasingly dominated the research landscape in recent years; these models remain largely opaque 'black boxes' with limited interpretability. In this paper, we revisit the utility of feature-based classification via a systematic pipeline designed to integrate per-electrode signal statistics with cross-brain connectivity metrics. The proposed "KnowEEG" pipeline extracts 783 per-electrode statistical features alongside between-electrode connectivity statistics, utilizing a balanced sampling strategy between high-dimensional per electrode features and connectivity. This feature-based pipeline achieves performance comparable to or exceeding state-of-the-art deep learning models across five diverse classification tasks: emotion det
    
[^451]: 面向航空发动机自适应自由形态管路布线的自学习框架

    A self-learning framework for adaptive free-form pipe routing in aeroengines

    [https://arxiv.org/abs/2504.03669](https://arxiv.org/abs/2504.03669)

    该论文提出了一种基于近端策略优化（PPO）算法的自学习管路布线方法（SLPR），通过统一的规则建模框架和势能表实现快速障碍检测与模糊规则建模，从而在动态设计环境下自适应地优化航空发动机的自由形态管路布线。

    

    管路布线是航空发动机设计中一个高度复杂、耗时且属于非确定性多项式时间困难（NP-hard）的问题。尽管在优化等曲率管路布线方面已有大量研究，但日益增长的对自由形态管路的需求带来了新的挑战。动态设计环境和模糊布局规则进一步影响了优化的性能和效率。为应对这些挑战，本研究提出了一种基于自学习的方法（SLPR），用于优化航空发动机中的自由形态管路布线。SLPR基于近端策略优化（PPO）算法，并集成了一个统一的规则建模框架，可在连续空间中实现高效的障碍物检测和模糊规则建模。此外，构建了一个势能表，可实现对布局倾向和干涉情况的快速查询。SLPR中的智能体通过迭代细化管路布线并积累设计知识，从而不断提升布线质量。

    arXiv:2504.03669v2 Announce Type: replace-cross  Abstract: Pipe routing is a highly complex, time-consuming, and no-deterministic polynomial-time hard (NP-hard) problem in aeroengine design. Despite extensive research efforts in optimizing constant-curvature pipe routing, the growing demand for free-form pipes poses new challenges. Dynamic design environments and fuzzy layout rules further impact the optimization performance and efficiency. To tackle these challenges, this study proposes a self-learning-based method (SLPR) for optimizing free-form pipe routing in aeroengines. The SLPR is based on the proximal policy optimization (PPO) algorithm and integrates a unified rule modeling framework for efficient obstacle detection and fuzzy rule modeling in continuous space. Additionally, a potential energy table is constructed to enable rapid queries of layout tendencies and interference. The agent within SLPR iteratively refines pipe routing and accumulates the design knowledge through int
    
[^452]: FANTF：基于模糊注意力网络的Transformer用于时间序列分析

    FANTF: Fuzzy attention network based transformers for time series analysis

    [https://arxiv.org/abs/2504.00070](https://arxiv.org/abs/2504.00070)

    提出了一种基于二型模糊集理论的模糊注意力机制FANTF，利用可学习的高斯不确定性扰动建模注意力得分中的不确定性，可作为标准注意力层的即插即用替代模块，提升Transformer在时间序列预测、分类和异常检测任务上的性能。

    

    本文介绍了基于模糊注意力网络的Transformer（FANTF），这是一种用于时间序列预测、分类和异常检测的方法。FANTF通过一种新颖的模糊注意力机制将模糊逻辑概念与Transformer架构相结合。所提出的机制受二型模糊集理论的启发，使用可学习的高斯不确定性扰动来建模注意力相关性得分中的不确定性。这使得模型能够比普通的确定性注意力更准确地捕捉时间关系中的不确定性。FANTF增强了Transformer记录复杂时间模式和多元变量连接的能力。该方法既与架构无关，也与任务无关。所提出的模糊注意力模块可以直接替换标准注意力层。为了评估其有效性，FANTF被集成到五个Transformer骨干网络中：Transformer等。

    arXiv:2504.00070v2 Announce Type: replace  Abstract: This paper introduces Fuzzy Attention Network-Based Transformer (FANTF), an approach for time series forecasting, classification, and anomaly detection. FANTF combines fuzzy logic concepts with transformer architectures through a novel fuzzy attention mechanism. The proposed mechanism is inspired by Type-2 fuzzy set theory. It uses a learnable Gaussian uncertainty perturbation to model uncertainty in attention relevance scores. This makes it possible for the model to accurately capture uncertainty in temporal relationships better than normal deterministic attention. Transformers capacity to record complex temporal patterns and multivariate connections is enhanced by FANTF. The approach is both architecture-agnostic and task-agnostic. The proposed fuzzy attention module can be used as a direct replacement for the standard attention layer. To evaluate its effectiveness, FANTF is integrated into five transformer backbones: Transformer, 
    
[^453]: BOLT：基于贝叶斯误差上界的通用分类器训练

    BOLT: Universal Classifier Training via a Bayes-Error Upper Bound

    [https://arxiv.org/abs/2501.07754](https://arxiv.org/abs/2501.07754)

    本文提出了一种从分类器输出样本计算贝叶斯错误率通用上界的新方法，并据此定义了BOLT损失函数用于训练分类器，实验表明其往往比交叉熵训练取得更低的分类错误率。

    

    本工作提出了一种新颖的方法，用于从分类器输出样本计算贝叶斯错误率（BER）的通用上界，该上界可作为底层分类任务的与贝叶斯误差相关的性能评估准则。我们证明该上界在二分类情况下是紧的，并刻画了多分类聚合形式中出现的最大-和间隙。基于此上界，我们定义了一种新的损失函数，称为贝叶斯最优学习阈值（BOLT），用于训练参数化分类器。对于每个数据样本，BOLT损失计算一项，其期望值是贝叶斯误差的上界，因此在训练过程中可以朝着最小化该上界的方向更新模型参数。在合成数据集和真实数据集上的数值实验表明，在所报告的设置中，BOLT训练往往能比交叉熵训练取得更低的留出集分类错误率。

    arXiv:2501.07754v2 Announce Type: replace  Abstract: This work introduces a novel approach to compute a universal upper bound on the Bayes error rate (BER) from classifier-output samples, which serves as a Bayes-error-related performance criterion for the underlying classification task. We show that the bound is tight in the binary case and characterize the max-sum gap that appears in the aggregate multiclass formulation. Based on this bound, we define a new loss function, called Bayes optimal learning threshold (BOLT), for training parameterized classifiers. For each data sample, the BOLT loss computes a term whose expectation upper-bounds the Bayes error, so that during training the model parameters can be updated toward minimizing this bound. Numerical experiments with both synthetic and real-world datasets show that, in the reported settings, BOLT training often attains lower held-out classification error than cross-entropy training.
    
[^454]: 一种多模态情感识别系统：整合面部表情、身体动作、语音与口语语言

    A Multimodal Emotion Recognition System: Integrating Facial Expressions, Body Movement, Speech, and Spoken Language

    [https://arxiv.org/abs/2412.17907](https://arxiv.org/abs/2412.17907)

    本文提出了一种整合面部表情、身体动作、语音和口语的多模态情感识别系统，旨在辅助而非取代心理评估者，并通过真实条件下的用户研究揭示了模型准确率从基准测试到现实场景的显著下降。

    

    心理学评估依赖于临床医生对患者情绪表达的观察和解读，这一过程容易受到主观性、偏见、疲劳和不一致性的影响。本文提出了一种多模态情感识别系统，旨在辅助而非取代评估者。该系统结合了面部表情识别、语音情感识别、基于转录文本的情感识别以及身体动作分析，并将它们的输出融合为包含类别概率的情绪特征，连同整体预测一起呈现给评估者。尽管单模态模型在其基准测试集上取得了较高的准确率，但一项有52名参与者表达七种情绪状态的远程用户研究表明，在现实条件下其准确率下降至82.97%至87.64%之间，某些个体情绪状态的识别率甚至低至48.08%。

    arXiv:2412.17907v2 Announce Type: replace-cross  Abstract: Psychological evaluation relies on clinicians' observation and interpretation of a patient's emotional expression, which is susceptible to subjectivity, bias, fatigue, and inconsistency. This paper presents a multimodal emotion recognition system designed to support, rather than replace, the evaluator. The system combines facial expression recognition, speech emotion recognition, text-based emotion recognition of transcribed speech, and body movement analysis, and fuses their outputs into an emotional profile of class probabilities that is presented to the evaluator together with the overall prediction. Although the unimodal models achieve high accuracy on their benchmark test sets, a remote user study with 52 participants expressing seven emotional states shows that their accuracy falls to between 82.97% and 87.64% under realistic conditions, with individual emotional states recognised in as few as 48.08% of cases. Under the s
    
[^455]: STAMP：用于按时间顺序划分图域自适应的对称时间平均消息传递

    STAMP: Symmetric Temporal Averaging Message Passing for Domain Adaptation in Chronologically Split Graphs

    [https://arxiv.org/abs/2411.10957](https://arxiv.org/abs/2411.10957)

    提出对称时间平均消息传递（STAMP），通过在时间戳间平衡消息强制时间域间的一阶矩不变性，从而解决按时间顺序划分图中的分布偏移问题，提升半监督节点分类性能。

    

    本文研究按时间顺序划分下图结构数据中的域自适应问题。我们考虑直推式半监督节点分类（SSNC），即利用历史节点的标签来预测较新节点的标签。图连通性中的时间依赖性会引起跨时间的分布偏移，这往往会降低在历史节点上训练的模型的性能。尽管具有重要的实际意义，但按时间顺序划分图下的域自适应问题仍然相对未被充分探索。我们提出对称时间平均消息传递（STAMP），这是一种时间戳平衡算子，用时间对齐的传播取代标准的邻域聚合。我们将其针对异质 ogbn-mag 数据集的专门变体称为 STAMP-mag，而将 STAMP 保留用于一般形式。通过在时间戳之间平衡消息，STAMP 在时间域之间强制实现一阶矩不变性，同时保持……

    arXiv:2411.10957v2 Announce Type: replace-cross  Abstract: This paper addresses domain adaptation in graph-structured data under chronological splits. We consider transductive semi-supervised node classification (SSNC), in which labels of historical nodes are used to predict labels of more recent nodes. Temporal dependencies in graph connectivity induce distribution shifts across time, often degrading the performance of models trained on past nodes. Despite its practical relevance, domain adaptation under chronological graph splits remains relatively underexplored. We propose Symmetric Temporal Averaging Message Passing (STAMP), a timestamp-balanced operator that replaces standard neighborhood aggregation with temporally aligned propagation. We call its specialized variant for the heterogeneous ogbn-mag dataset STAMP-mag, reserving STAMP for the general formulation. By balancing messages across timestamps, STAMP enforces first-moment invariance across temporal domains while remaining s
    
[^456]: 面向分类的成对差分学习

    Pairwise Difference Learning for Classification

    [https://arxiv.org/abs/2406.20031](https://arxiv.org/abs/2406.20031)

    本文将成对差分学习（PDL）从回归扩展到分类任务，提出通过在成对训练数据上求解二分类问题来构建PDL分类器，在大规模实证研究中其预测性能优于现有最先进方法。

    

    成对差分学习（PDL）最近被引入作为一种用于回归的新型元学习技术。与以标准方式学习从实例到结果的映射不同，其核心思想是学习一个函数，该函数以两个实例作为输入，并预测相应结果之间的差异。给定此类函数，对查询实例的预测由每个训练样本推导得出，然后取平均值。本文将PDL扩展到分类任务，提出了一种元学习技术，通过在原始训练数据的成对版本上求解一个适当定义的（二分类）问题来构建PDL分类器。我们在大规模实证研究中分析了PDL分类器的性能，发现其在预测性能方面优于最先进的方法。最后，我们提供了一个易于使用且公开可用的实现。

    arXiv:2406.20031v2 Announce Type: replace-cross  Abstract: Pairwise difference learning (PDL) has recently been introduced as a new meta-learning technique for regression. Instead of learning a mapping from instances to outcomes in the standard way, the key idea is to learn a function that takes two instances as input and predicts the difference between the respective outcomes. Given a function of this kind, predictions for a query instance are derived from every training example and then averaged. This paper extends PDL toward the task of classification and proposes a meta-learning technique for inducing a PDL classifier by solving a suitably defined (binary) classification problem on a paired version of the original training data. We analyze the performance of the PDL classifier in a large-scale empirical study and find that it outperforms state-of-the-art methods in terms of prediction performance. Last but not least, we provide an easy-to-use and publicly available implementation o
    
[^457]: 高斯随机场的次大值与精确间距检验

    Second Maximum of a Gaussian Random Field and Exact (t-)Spacing test

    [https://arxiv.org/abs/2406.18397](https://arxiv.org/abs/2406.18397)

    本文提出了黎曼流形上高斯随机场次大值的新概念，借助 Kac-Rice 公式推导出以次大值和 Hessian 矩阵独立部分为条件的全局最大值的显式分布，从而构造出一种可用于检测稀疏备择假设的精确间距检验方法。

    

    本文介绍了在黎曼流形上定义的高斯随机场的次大值这一新概念：当随机场在某点 $t_1$ 达到全局最大值后，次大值指的是在移除场在 $t_1$ 处的值和梯度之后，场在远离 $t_1$ 处所能达到的最大值。这一统计量是刻画全局最大值分布的有力工具。通过采用特定的 Kac-Rice 公式，我们推导出了在给定次大值和黎曼 Hessian 矩阵独立部分（即 $t_1$ 处 Hessian 矩阵中与场在 $t_1$ 处的值和梯度无关的部分）的条件下，全局最大值的显式分布。该方法给出了一种基于这些最大值之间间距的精确检验，我们称之为间距检验。我们研究了该检验在检测高斯对称

    arXiv:2406.18397v4 Announce Type: replace-cross  Abstract: In this article, we introduce the novel concept of the second maximum of a Gaussian random field defined on a Riemannian manifold: Once the field reaches its global maximum at a point $t_1$, the second maximum is the largest value of the field away from $t_1$, after the value and the gradient of the field at $t_1$ have been removed. This statistic serves as a powerful tool for characterizing the distribution of the global maximum. By employing an ad-hoc Kac-Rice formula, we derive the explicit distribution of the maximum, conditioned on the second maximum and the independent part of the Riemannian Hessian (the part of the Hessian at $t_1$ that is independent of the value and the gradient of the field at $t_1$). This approach yields an exact test based on the spacing between these maxima, which we refer to as the spacing test.   We investigate the applicability of this test for detecting sparse alternatives in Gaussian symmetric
    
[^458]: 利用分布鲁棒优化获取可解释的分类模型

    Obtaining Explainable Classification Models using Distributionally Robust Optimization. (arXiv:2311.01994v1 [stat.ML])

    [http://arxiv.org/abs/2311.01994](http://arxiv.org/abs/2311.01994)

    本论文介绍了一种利用分布鲁棒优化获取可解释的分类模型的方法，通过构建稀疏的规则集合来同时解决规则集的稀疏性和预测准确性之间的权衡，从而保证泛化性能并降低计算成本。

    

    对于人类用户来说，模型的可解释性对于理解提议分类器如何根据特征值给数据分配标签至关重要。我们研究使用特征值规则集构建的广义线性模型，该模型可以捕捉非线性依赖和交互作用。规则集的稀疏性和预测准确性之间存在固有的权衡。使用现有方法来找到合适的稀疏度选择（例如通过交叉验证）计算成本很高。我们提出了一种新的公式来学习同时解决这些竞争因素的规则集合。通过利用分布鲁棒优化来确保良好的泛化性能，同时保持低计算成本。该公式利用列生成有效地搜索规则集合的空间并构建稀疏的规则集合，与随机森林或Boosting及其变体等技术相比。我们提出了理论结果来推动这一公式的发展。

    Model explainability is crucial for human users to be able to interpret how a proposed classifier assigns labels to data based on its feature values. We study generalized linear models constructed using sets of feature value rules, which can capture nonlinear dependencies and interactions. An inherent trade-off exists between rule set sparsity and its prediction accuracy. It is computationally expensive to find the right choice of sparsity -- e.g., via cross-validation -- with existing methods. We propose a new formulation to learn an ensemble of rule sets that simultaneously addresses these competing factors. Good generalization is ensured while keeping computational costs low by utilizing distributionally robust optimization. The formulation utilizes column generation to efficiently search the space of rule sets and constructs a sparse ensemble of rule sets, in contrast with techniques like random forests or boosting and their variants. We present theoretical results that motivate an
    
[^459]: 为协调虚拟电厂资产的分布式优化引入机器学习

    Machine Learning Infused Distributed Optimization for Coordinating Virtual Power Plant Assets. (arXiv:2310.17882v1 [cs.LG])

    [http://arxiv.org/abs/2310.17882](http://arxiv.org/abs/2310.17882)

    本文提出了一种名为LOOP-MAC的方法，利用机器学习辅助的分布式优化来协调虚拟电厂（VPP）的资产。该方法采用多智能体协调视角，利用神经网络逼近器加速解决方案搜索。该方法通过引入测量图来保证最佳解之间的关系。

    

    随着对分布式能源资源（DER）部署日益增加的兴趣，虚拟电厂（VPP）已成为汇集各种DER并促进其参与批发能源市场的关键工具。这些VPP部署得到了联邦能源监管委员会第2222号命令的推动，该命令使DER和VPP在市场领域具有竞争力。然而，DER的多样性和分散性质给VPP资产的可扩展协调带来了显著挑战。为了解决效率和速度瓶颈，本文提出了一种新颖的机器学习辅助的分布式优化方法来协调VPP资产。我们的方法名为LOOP-MAC（为多智能体协调优化过程学习优化），采用多智能体协调视角，每个VPP代理管理多个DER并利用神经网络逼近器加速解决方案搜索。LOOP-MAC方法使用一个测量图确保了最佳解之间的关系。

    Amid the increasing interest in the deployment of Distributed Energy Resources (DERs), the Virtual Power Plant (VPP) has emerged as a pivotal tool for aggregating diverse DERs and facilitating their participation in wholesale energy markets. These VPP deployments have been fueled by the Federal Energy Regulatory Commission's Order 2222, which makes DERs and VPPs competitive across market segments. However, the diversity and decentralized nature of DERs present significant challenges to the scalable coordination of VPP assets. To address efficiency and speed bottlenecks, this paper presents a novel machine learning-assisted distributed optimization to coordinate VPP assets. Our method, named LOOP-MAC(Learning to Optimize the Optimization Process for Multi-agent Coordination), adopts a multi-agent coordination perspective where each VPP agent manages multiple DERs and utilizes neural network approximators to expedite the solution search. The LOOP-MAC method employs a gauge map to guarant
    

