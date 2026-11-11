# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Medical Image Segmentation: Foundation Model Adaptation versus Task-Specific Training](https://arxiv.org/abs/2611.14519) | 该研究将DINOv2基础模型以从仅调偏置项到全量微调的不同程度进行适配，并与从头训练的U-Net/ResU-Net在结肠镜、乳腺超声和皮肤镜分割任务上系统对比，证明适配后的基础模型在来自其他机构的独立外部数据集上具有更优的泛化能力。 |
| [^2] | [Toward Foundation Detection Models via Example-Based Grounding in EM-DETR](https://arxiv.org/abs/2611.14518) | 该论文提出一种师生训练策略并结合基于示例定位的EM-DETR框架，构建能够整合异质标注的多域医学影像病灶检测基础模型。 |
| [^3] | [Contrastive Discovery: Open-Ended Scientific Discovery over Competing Explanations](https://arxiv.org/abs/2611.14514) | 该论文提出“对比发现”框架，用互斥且穷尽的竞争性解释集合取代单一假设检验，通过证据对信念在集合内的重新分配来评估实验、确定被支持的替代解释，从而改进开放式科学发现。 |
| [^4] | [Comparative Analysis of Agent Evaluation Frameworks: Stability, Detection, and Discovery in Enterprise Analytics Agents](https://arxiv.org/abs/2611.14493) | 该论文通过三阶段比较研究系统评估了三种企业分析智能体评估框架，发现基于轨迹的LLM评判框架在评估稳定性（97%配置达标）和已知风险检测（F1=0.90）方面综合表现最佳。 |
| [^5] | [ReTrace: Distilling Tool-Using Agents into Planner-Free Workflows](https://arxiv.org/abs/2611.14492) | ReTrace通过从智能体的推理解释中提取每步背后的条件并跨运行合并为规则文档，将SOP和少量智能体运行日志蒸馏成由代码驱动的无规划器可执行工作流，在大规模流程自动化中以更低的延迟和成本实现接近LLM智能体的完成率（93.8%对96.8%）。 |
| [^6] | [Re-Search: Dissecting LLM Ideation Through a Two-Player Questioning Game](https://arxiv.org/abs/2611.14487) | 提出 Re-Search 基准，将“二十个问题”游戏引入科研构思评估，要求模型仅凭研究问题通过是非提问重新发现已发表论文的核心想法，并发现大语言模型的构思进展几乎集中在前 30 个提问内、其瓶颈在于想法生成而非判断。 |
| [^7] | [One Policy for 100 Household Tasks: Our Solution for the 2026 BEHAVIOR Challenge](https://arxiv.org/abs/2611.14432) | 该团队基于2025年冠军方案π₀.₅构建单一VLA模型，仅用一个检查点完成2026年BEHAVIOR挑战赛全部100个家庭任务并获得第九名，核心改进在于延长训练时间、优化训练参数、加快训练步骤并减少梯度噪声。 |
| [^8] | [Two Communities, One Problem: Branch-and-Bound vs. A* for Resource-Constrained Project Scheduling](https://arxiv.org/abs/2611.14426) | 本文以资源受限项目调度问题（RCPSP）为基准，在相同的下界和节点选择策略下，系统比较了运筹学的分支定界法与人工智能的A*搜索算法，并建立了混合整数线性规划、状态-动作和时间Petri网表示之间的联系，从而揭示两种搜索范式的根本性性能差异。 |
| [^9] | [Quantifying the Sampling Frequency Invariance of Time-Series Foundation Models for Sensor Data Applications](https://arxiv.org/abs/2611.14425) | 该论文首次将采样频率不变性置于时间序列基础模型评估的核心，通过压力测试4个主流模型并提出FSI、FDI、CDI三个新指标，发现模型行为取决于信息带宽损失而非采样频率数值本身。 |
| [^10] | [Evaluating LLM-Generated World Models for Neuro-Symbolic Reasoning](https://arxiv.org/abs/2611.14419) | 该研究通过一个涵盖44个场景的基准评估五个大语言模型生成的世界模型，发现尽管135个生成模型中有134个通过了严格的机械有效性审计，但其与参考模型的结构一致性平均F1仅为0.366，表明机械有效性并不足以保证世界模型忠实地表示预期的决策问题。 |
| [^11] | [Precision on Demand: Escalating Weight Bits at Uncertain Decoding Steps in a Small Local Model](https://arxiv.org/abs/2611.14418) | 提出按需精度的解码方法：小型本地模型平时用高4位权重解码，仅在输出分布不确定（高熵或前两名差距小）时升级到8位重算该步骤，以约87%的权重读取开销取得接近全高精度的准确率。 |
| [^12] | [Bits or permutations? Spending a fixed inference budget on weight precision or on option-order averaging in small local language models](https://arxiv.org/abs/2611.14413) | 本文量化了小型本地语言模型中权重量化精度对答案选项顺序偏差的影响，发现6比特量化不损失准确率而4比特量化会显著加剧答案槽位偏斜，并据此比较了固定推理预算下提升权重精度与选项顺序平均两种方案的性价比。 |
| [^13] | [Causal Concept Explanations for Deep Neural Models](https://arxiv.org/abs/2611.14407) | 该论文提出 CCW-Net，首次将干预性中介分析作为训练目标，使概念解释与模型输出之间建立真正的因果关系而非仅是相关性，并提供量化每个概念解释对模型输出因果贡献的评估工具。 |
| [^14] | [Blurring the Block Boundaries: Overlapping Block Denoising for Efficient Diffusion Language Models](https://arxiv.org/abs/2611.14387) | 本文提出B3方法，通过在共享前向传播中并行精炼两个块来模糊块边界，实现无需训练和超参数调整的扩散语言模型加速，同时保持生成质量。 |
| [^15] | [An Energy-Based Perspective on Machine Learning](https://arxiv.org/abs/2611.14383) | 本教程以能量模型（EBM）为统一视角，通过能量函数、推断过程和学习方法三大核心组件，将判别模型、GAN、扩散模型、自回归模型等各类机器学习方法统一在一个共同的对比学习框架之下。 |
| [^16] | [Cognitive Mapping of Learning by Thinking Mechanisms in Large Language Models](https://arxiv.org/abs/2611.14382) | 本文提出认知式思考式学习框架，将反思性推理、合成记忆形成、内部模拟和概念自组织四种认知机制映射到大语言模型的计算过程，并论证思维链、自我反思等代表性方法可在功能上近似这些机制，从而无需额外标注数据即可提升任务性能。 |
| [^17] | [EngineAgent: Agentizing On-Device LLM Inference Engines with Inference Stack Hierarchy Awareness and Hardware Abilities](https://arxiv.org/abs/2611.14356) | EngineAgent通过构建层级感知的引擎索引并利用已支持模型预先验证后端工作流程，将可复用的引擎与硬件知识从模型特定实现中分离，从而自动化端侧LLM推理引擎的新模型适配过程。 |
| [^18] | [CLADE: Category-Level Answering from Deficient Evidence in Retrieval-Augmented Language Models](https://arxiv.org/abs/2611.14355) | 该论文揭示了语言模型在阅读仅包含粗粒度类别信息（如疾病家族而非具体疾病）的检索段落时会直接作答而非弃答的“Clade”现象，并提出这类“可解决的不足”可通过领域本体中的候选答案列表来刻画与处理。 |
| [^19] | [AI Isn't Getting Dumber, We're Getting Pickier: Rising Standards in Human Evaluation of Large Language Models](https://arxiv.org/abs/2611.14333) | 该研究通过分析Chatbot Arena上126万余张针对固定版本模型的盲投票数据发现，“两个都不好”的投票比例持续显著上升，表明AI模型并非在变差，而是人类评估者的标准在不断提高。 |
| [^20] | [Forward-Value Discretization: Faster Stochastic Diffusion Sampling Without High-Order Approximation](https://arxiv.org/abs/2611.14328) | 该论文证明了高阶近似无法提升随机扩散采样的收敛阶数，并提出免训练的前向值离散化方法（F-DPMSolver），通过利用先前预测构建的前瞻估计，在低神经函数评估次数下实现更快的采样并提升样本质量。 |
| [^21] | [When Should Agents Remember? Falsification-Gated Self-Evolution for LLM Agents](https://arxiv.org/abs/2611.14326) | 提出证伪门控自我进化（FGSE）框架，将经验教训表示为结构化假设，仅在临时状态中经受证伪、迁移与回归探测，只有通过收益与风险双重验证的行为更新才被写入智能体的持久记忆，从而显著提升LLM智能体自我进化的可靠性与安全性。 |
| [^22] | [Don't Just Trust Your Memory: Evidence-Linked Memory Reduces Memory-Induced Degradation in LLM Agents](https://arxiv.org/abs/2611.14320) | 提出WikiTrace证据关联记忆系统，通过将每条经验教训与其来源轨迹中的证据相链接，使LLM智能体能够核实记忆内容，从而有效减少记忆诱导的性能退化。 |
| [^23] | [Multimodal Abstention That Preserves Acquired Correct Answers](https://arxiv.org/abs/2611.14311) | 提出ReFrain多模态弃答方法，在提升对不可回答问题正确弃答能力的同时，保证任务适配阶段获得的正确答案零损失。 |
| [^24] | [LM-SimBench: A Unified Framework, Dataset, and Benchmark for Large Language Model Training Simulation](https://arxiv.org/abs/2611.14285) | 该论文提出LM-SimBench，通过统一的仿真分解框架、基于昇腾NPU集群的大规模性能剖析数据集以及标准化基准，首次实现了对不同大语言模型训练仿真器的公平、系统化评估。 |
| [^25] | [Ninety-Six Ways to Measure a Scribe: Fragmented Evaluation of Ambient Scribes in Healthcare](https://arxiv.org/abs/2611.14284) | 该研究系统分析了54项医疗环境式AI记录助手的评估研究，发现评估维度和指标高度碎片化——共十个维度、40种指标、96种操作化方式，其中36%仅被单一研究使用，且缺乏标准化评估协议，导致各研究结果难以比较、证据难以综合。 |
| [^26] | [Achieving Robust Channel Estimation Neural Networks on Doppler effects by Designed Training Data](https://arxiv.org/abs/2611.14267) | 本文提出了一套合成训练数据的设计准则，通过精心设计的训练数据使信道估计神经网络能够泛化到未见过的时变信道，从而在具有强多普勒效应的高速场景下实现鲁棒信道估计，避免了昂贵且延迟高的在线训练。 |
| [^27] | [Recurrent Modular Structure in Spiking Neural Network Reinforcement Learning](https://arxiv.org/abs/2611.14262) | 提出群落循环脉冲神经网络，将循环网络划分为具有固定角色的群落（控制器、动作和潜在状态群落），并结合局部三因子学习规则及新提出的Map-STDP，使脉冲强化学习网络结构更接近生物皮层微回路。 |
| [^28] | [Lilo: A Multi-Agent Conversational Sandbox for Children's Digital Communication](https://arxiv.org/abs/2611.14260) | Lilo是一个多智能体对话沙盒系统，让儿童在模拟的群聊冲突场景中与AI同伴互动，并借助导师的红绿灯引导练习应对策略，实验表明该系统能显著提升9-11岁儿童应对不愉快消息的自我效能感。 |
| [^29] | [A Cross-Scenario Survey and Analysis of Self-Evolving Multimodal Agents](https://arxiv.org/abs/2611.14256) | 本综述首次对自进化多模态智能体进行形式化定义，并提出三阶段自进化循环分析框架，系统揭示了自我改进机制在多模态推理、生成、GUI导航与具身操作四大场景间非对称迁移的结构性根源。 |
| [^30] | [Personalized Affective Perception via Continual Learning in Human-Agent Negotiation](https://arxiv.org/abs/2611.14247) | 该论文提出CLIFER——一种基于持续学习与想象机制的面部表情识别方法，使人形机器人在与人类协商中实时个性化地感知情绪并动态调整协商策略，相比基线显著提升了个体协商效用。 |
| [^31] | [An AI Assistant for Automating Anti-Corruption Expertise of Draft Legislation Based on the TRACE Framework](https://arxiv.org/abs/2611.14244) | 该论文基于TRACE架构提出了一种自动化反腐败专家审查的AI助手，能够筛查法案草案中的腐败风险并精确定位专家会批评的问题条款，同时利用乌克兰国家反腐败预防署的公开结论构建了首个开放评估基准。 |
| [^32] | [Dyship: Leveraging Acoustic Features For Speaker Relationship Classification From Dyadic Conversations](https://arxiv.org/abs/2611.14241) | 该论文提出DyShip框架，通过比较频谱图、MFCC、WavLM、x向量和发音运动五种声学表示，从双人对话中实现说话人熟悉度和亲缘关系的自动分类，且多数分类器表现超越人类和音频语言模型基准。 |
| [^33] | [Verifiable Agentic Traffic Class: A Policed Quality-of-Service Mechanism for Agentic AI Traffic, Evaluated from Simulation to Real-Kernel Adversarial Testing](https://arxiv.org/abs/2611.14211) | 本文提出可验证智能体流量类别（VATC），通过DSCP流量声明、路由器侧验证分类器和速率受限优先队列，为智能体AI流量提供可验证的服务质量保障，在突发场景下相比传统DiffServ将智能体排队延迟降低98%，同时仅使人类流量延迟增加12%。 |
| [^34] | [Breakdown of Autoregressive Multi-Step Reasoning Under Concentrated vs. Diffuse Attention Head Dropout](https://arxiv.org/abs/2611.14209) | 受哺乳动物大脑“关闭状态”启发，该研究发现对大语言模型注意力头施加集中式爆发失活，比同等规模的分散式失活会造成严重得多的多步数学推理性能下降。 |
| [^35] | [Joint Quantization and Dynamic Partitioning for Context-Aware Edge Vision Transformers](https://arxiv.org/abs/2611.14196) | 本文提出一种面向车联网边缘的编排框架，将模型分割点、逐层量化位宽与执行位置的联合选择建模为混合整数线性规划（MILP），在时延与任务质量约束下最小化计算和通信能耗，使视觉Transformer推理能够自适应时变的车车通信环境。 |
| [^36] | [An Augmented Lagrangian Approach to Lifted Bellman Linear Programming](https://arxiv.org/abs/2611.14183) | 本文提出用增广拉格朗日乘子法求解提升贝尔曼线性规划（LBLP），将固定的惩罚系数变为通过对偶上升学习的乘子，在无需目标网络的情况下为离线强化学习提供了具有KKT理论保证的神经求解器AL-LBLP。 |
| [^37] | [Cross-Border Fulfillment Center Selection via Doubly Robust Estimation and Linear Programming](https://arxiv.org/abs/2611.14174) | 该论文提出了一个融合预测、双重稳健估计与线性规划的多阶段优化框架来选择最优跨境履约中心，在三条生产级国际运输走廊上相比人工专家策略实现了8.5%-51.1%的配送时间降低和4.4%-48.3%的物流成本降低。 |
| [^38] | [BARS: Budget-Aware Sequential Sampling with Refusal for Majority Identification under Unknown Distributions](https://arxiv.org/abs/2611.14171) | 提出BARS，一种预算感知的序贯抽样规则，在未知分布下通过认证前两名答案差距，返回提前停止、预算多数投票或拒绝三种结果，并提供可证明的错误率界限，帮助实践者判断何时扩大预算或更换模型。 |
| [^39] | [Representational Atoms in Neural Networks](https://arxiv.org/abs/2611.14168) | 该论文定义了神经网络中的基本表征单元（FRUs）并建立保真度与稳定性两项评估标准，证明阈值激活稀疏自编码器（TSAEs）在适当条件下可高保真、稳定地识别FRUs，且高稳定性的FRUs具有更强的单语义性并能跨层组合以恢复与人类概念结构相对应的语义组织。 |
| [^40] | [Hx2Dx: From Anamnesis to Diagnosis - A Multimodal Arena for Conversational Clinical Reasoning in LLMs](https://arxiv.org/abs/2611.14122) | 该论文提出了Hx2Dx ARENA——一个基于1000个NEJM临床病例的交互式多模态基准，通过模拟从病史采集到最终诊断的完整问诊流程来评估大语言模型的对话式临床推理能力，弥补了传统静态评估忽视主动问诊能力的不足。 |
| [^41] | [Visual Similarity in Adult Word Extension and Artificial Word Learning](https://arxiv.org/abs/2611.14116) | 本研究通过重新分析大规模成人词汇扩展数据并进行人工词汇学习实验，利用冻结的自监督视觉编码器（DINOv2）量化物体间的视觉相似性，发现视觉距离显著影响词汇的扩展范围，且教学示例的类别层级高低决定了学习者在多大程度上将词汇泛化到更广泛或更具体的类别。 |
| [^42] | [Looking One Step Ahead: An Anticipation-guided Agent for Object State Readout in Videos](https://arxiv.org/abs/2611.14113) | 该论文提出了一种名为LASER的预判引导智能体，通过预测物体下一步可能发生的变化来确定当前帧中需要重点检查的视觉属性，从而实现视频中物体状态及其转换的准确读取。 |
| [^43] | [Model-Free Runtime Measurement of Task-Relevant Information](https://arxiv.org/abs/2611.14111) | 提出了一种无模型的采集状态遥测（AST）架构，通过固定的因果监控器输出实时信号，评估自主系统的信息采集过程是否真正朝着任务需求的解决方向正常推进。 |
| [^44] | [ProofCoder: Continuous Reasoning for Automated Software Verification](https://arxiv.org/abs/2611.14106) | ProofCoder从编码智能体视角出发，让模型在整个证明搜索过程中持续推理，并根据证明助手的反馈自主决定信息收集、证明状态检查、片段修改和回溯，突破了传统LLM方法僵化策略生成循环的限制，实现了面向Rocq的自动化证明合成。 |
| [^45] | [Sable: A Small-Field Sumcheck Argument for Bit-Exact Inference of Large Language Models](https://arxiv.org/abs/2611.14083) | Sable通过整数化确定性计算、小域sumcheck约束系统和单GPU逐层流式证明，实现了对大语言模型位精确推理的可验证性，在单块GPU上92.5秒内即可完成Qwen2.5-32B生成4096个token全部64层推理的证明。 |
| [^46] | [From Bias to Positionality: A critical grammar for Large Language Models](https://arxiv.org/abs/2611.14079) | 该论文提出用“处境性”取代“偏见”作为理解大语言模型的批判性框架，因为偏见需要外部标准作为参照，而在语言模型的一般使用中不存在这样的标准，因此更应追问模型“从何处说话”。 |
| [^47] | [ZoomGain: Mining Global-to-Zoom Perception Gaps for Fine-Grained Visual Supervision](https://arxiv.org/abs/2611.14076) | ZoomGain利用视觉语言模型“放大后能识别细节、全局视角却难以感知”的差距，构建了一种仅依赖缩放所得视觉信息、并经多模型在缩放视图上验证的细粒度视觉监督数据自动生成流水线。 |
| [^48] | [Obscured Wildfire Flame Detection by Spatio-temporal Analysis of Smoke Patterns Using Frame-wise Transformers](https://arxiv.org/abs/2611.14074) | 该论文提出一种逐帧Transformer架构，通过对RGB视频中烟雾运动的时空联合分析来检测可见及被遮挡的野火火焰，在保留帧内空间信息的同时优化时序长度，实现高性能并降低计算成本。 |
| [^49] | [HandsOnAid: Building and Evaluating an Offline First-Aid Assistant for Emergency Response Scenarios](https://arxiv.org/abs/2611.14059) | 本文构建了HandsOnAid——一个可在无信号、无救援环境中通过QLoRA微调的4位Gemma 2B模型实现的离线手机急救助手，并通过四个前沿大模型盲评验证了微调显著提升了急救建议的质量与安全性。 |
| [^50] | [Can KV Cache Be Reused Across Model Updates? A DeepSeek-V4-Flash Case Study](https://arxiv.org/abs/2611.14051) | 该研究首次系统探索了KV缓存在模型更新间的直接复用，发现缓存可保留任务信息并带来最高5.08倍的首词生成加速，但复用缓存并不忠实复现新模型行为，为此提出了函数性缓存偏差（Functional Cache Divergence）来量化这种行为差异。 |
| [^51] | [MISO-BR: An Open Corpus for Misogyny Detection in Brazilian Portuguese](https://arxiv.org/abs/2611.14036) | 该论文推出了MISO-BR——首个面向巴西葡萄牙语厌女症检测的开放、有文档记录且版本化的标注语料库，附带完整的构建协议、标注指南和可复现的基线模型，填补了该领域开放资源的空白。 |
| [^52] | [Few-Step Streaming Video Generation via Condition Shifting](https://arxiv.org/abs/2611.14033) | APEX通过条件偏移机制在生成器内部复用同一参数估计伪造分数，移除了DMD所需的独立伪造网络，同时保留冻结教师与自回放强制等机制，实现了更高效的少步流式视频生成。 |
| [^53] | [Program-Aided Patchable Skill Evolution](https://arxiv.org/abs/2611.14019) | 该论文提出PASE框架，通过将智能体技能中的可复用程序性逻辑分离为可编辑的可执行程序（而非自然语言文档），实现技能的可维护性、可补丁化与针对性迭代演化。 |
| [^54] | [Interleaving anatomical inspection and diagnostic reasoning for explainable radiology report generation](https://arxiv.org/abs/2611.14018) | ARIVE提出了一种可解释的放射报告生成框架，在撰写报告前交错进行解剖检查与诊断推理，并利用159,416条多模态思维链轨迹训练和奖励优化，实现了临床疗效超越强基线的可解释报告生成。 |
| [^55] | [Personalized Memory Rubrics: Learning What to Remember per User via Rubric-Guided Credit Assignment](https://arxiv.org/abs/2611.14017) | 该论文提出个性化记忆量规（PMR）框架，将终端问答反馈转化为细粒度、按步骤且按用户自适应的信用分配，解决了长时程个性化智能体记忆更新策略训练中反馈信号粗粒度与用户间异质性两大难题。 |
| [^56] | [Margin-Guided Token Credit for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2611.14015) | 提出MCPO方法，利用策略下一Token分布的前两名概率差识别模型犹豫的Token位置，将奖励信用有针对性地集中于这些位置，且无需额外前向传播或批次统计，在六个竞赛数学基准上平均超越GRPO基线6.55分。 |
| [^57] | [A Stage-Wise Explainable Pipeline for Arteriolar-to-Venular Ratio Estimation and Retinopathy Screening](https://arxiv.org/abs/2611.14013) | 该论文提出了一种五阶段可解释流水线，通过可检查的中间输出和显式阈值规则实现眼底图像的AVR估计与视网膜病变筛查，使错误可追溯到具体处理阶段，并在30张HRF图像上达到96.7%的准确率。 |
| [^58] | [Coding Agent Memory Post-Training II: Unlocking the Memory Potential of Pre-trained File Operations for Long-Horizon Tasks via On-Policy Distillation](https://arxiv.org/abs/2611.14009) | 该论文提出 CAMG-II 长时程智能体强化学习环境套件，通过提供 Bash 访问和回合持久工作区，让记忆控制建立在预训练已有的文件操作能力之上，从而解决在线策略蒸馏中因记忆接口超出预训练范围而导致教师指导无效、学生记忆动作采样不足的问题。 |
| [^59] | [Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Models](https://arxiv.org/abs/2611.13997) | 该论文提出了首个系统性评估大型视觉语言模型“推理时个性化隐私保护”（ITP³）能力的基准SPY-Bench，包含32,700个涵盖67个类别和24个真实场景的样本以及基于用户规范与情境感知的新颖指标，旨在使模型能够遵循用户通过语言动态指定的个性化隐私边界。 |
| [^60] | [Understanding Mixture-of-Experts Language Models Beyond Expert Boundaries](https://arxiv.org/abs/2611.13975) | 该研究提出路由感知幅度剖析（RAMP）方法，首次揭示MoE模型中领域专业化计算在神经元层面的组织规律（同领域神经元重叠率高达58-74%，远超跨领域的15-24%），并追踪了这种组织在预训练和后训练全过程中的动态演变。 |
| [^61] | [TransGHOST: Real-Time, Synthesis-Free Reconstruction of Instrument-Occluded Regions in Ophthalmic Microsurgery](https://arxiv.org/abs/2611.13971) | TransGHOST提出了一种实时、非生成的因果参考引导重建流水线，通过仅基于可见区域拟合的几何变换从历史帧重采样来恢复眼科显微手术中被器械遮挡的区域，并提供逐像素置信度与来源标注，从而避免了生成式修复带来的合成内容与不可追溯问题。 |
| [^62] | [Controlled Knowledge Engineering for Taxonomy Induction with Tool-Grounded LLM Agents and Runtime-Adaptive Policies](https://arxiv.org/abs/2611.13966) | 本文提出TIGA架构，将分类体系归纳形式化为显式状态与受约束动作空间上的迭代决策过程，借助可执行的分类编辑工具、记录调用与理由的记忆机制以及运行时自适应策略，克服LLM自由生成中的实体幻觉与无效层次关系，实现完整、结构有效且语义连贯的自动分类体系构建。 |
| [^63] | [From Attack Surface to Defense Architecture: Compositional Security for Agentic Coding Harnesses](https://arxiv.org/abs/2611.13960) | 该论文提出ActiveSecHarness——一种基于“智能与权限分离”原则的神经符号参考监控器架构，通过HMAC能力令牌与二维污染格为AI编程智能体提供结构性的组合式安全防御。 |
| [^64] | [TreeAgent: Structure-Aware Memory Agent for Processing Long Hierarchical Data](https://arxiv.org/abs/2611.13946) | TreeAgent提出了一种结构感知的记忆智能体，利用输入数据的层级结构来指导分段处理与历史信息访问，解决了现有记忆智能体在处理API响应、HTML文档等长层级结构化数据时拆分记录字段、丢失层级信息和兄弟记录细节的问题。 |
| [^65] | [Chem World: A Large-Scale Benchmark and Physics-Aware Framework for Trustworthy Molecular Property Prediction](https://arxiv.org/abs/2611.13945) | 本文提出了大规模基准Chem World和物理感知组合建模框架PACM，通过整合17个数据集、80余万实例和领域物理约束，为可信的分子性质预测提供了统一的评估平台，并验证了分子语言模型表示的强泛化与可迁移性。 |
| [^66] | [Tailoring Optimization Approaches with Domain Knowledge in Industry Process for Meta-heuristic Development](https://arxiv.org/abs/2611.13940) | 本文提出了元启发式开发的工业流程（IPMD）框架，并展示了一种基于遗传编程的超启发式方法BNTGP，成功应用于求解车辆路径和项目调度这两个NP难多目标优化问题。 |
| [^67] | [The Case for AI-Enabled Gate Monitoring as Farm Security in the UK: A PRISMA-Guided Review of the Evidence on Rural Crime, Livestock Predation and Farm Access](https://arxiv.org/abs/2611.13933) | 本文通过PRISMA系统性综述发现英国农场安全的证据基础薄弱、农村犯罪损失高昂（2025年达4,150万英镑）且现有安防措施缺乏独立有效性评估，从而论证了采用AI赋能门禁监控作为英国农场安全方案的必要性。 |
| [^68] | [TAPCO-G: LLM-Guided Design-Adaptive Timing and Power Co-Optimization with Graph-Based Physical Priors](https://arxiv.org/abs/2611.13931) | TAPCO-G 提出了一个将大语言模型引导的优化器自适应与基于图的物理先验相结合的布局后时序与功耗协同优化框架，通过外层循环调整工具设置并修订修复策略、内层循环基于静态时序分析快照进行可靠的延迟与功耗估计，从而有效解决设计特定的时序与功耗瓶颈。 |
| [^69] | [Beyond Proxy Metrics: MLLMs as Human Surrogates for Explainable AI Evaluation](https://arxiv.org/abs/2611.13926) | 本文提出一个“基于目的”的评估框架，并证明多模态大语言模型可以作为人类受试者的替代者来自动化评估可解释AI技术，从而解决代理指标信息量不足以及人类评估昂贵、难扩展、难复现的问题。 |
| [^70] | [Domain-Invariant and Language-Model-Assisted Acoustic Side-Channel Attacks on Keyboards](https://arxiv.org/abs/2611.13922) | 本文证明，通过让声学模型具备显式域不变性并用语言模型校正含噪预测，声学侧信道攻击在未见过的键盘、多样用户、嘈杂环境和压缩信道等真实开放条件下仍能有效恢复键盘输入内容。 |
| [^71] | [CAppleNet: A Coordinate Attention-Based YOLO11 for Real-Time Apple Stem Cavity and Calyx Detection](https://arxiv.org/abs/2611.13920) | 该论文提出了CAppleNet，一种基于坐标注意力增强的YOLO11模型，在自建的1506张苹果图像数据集上实现了果梗凹陷与花萼等细粒度形态标志点的实时高精度检测，并通过系统性消融实验确定了关键改进因素。 |
| [^72] | [From Counterfactual Edits to Human Narratives: LLM-Driven Interpretations for Recommender System Explanations](https://arxiv.org/abs/2611.13914) | 该论文提出利用大语言模型作为中介，将推荐系统反事实解释中冗长且难以理解的用户交互历史编辑列表，总结为描述高层偏好模式的简洁自然语言解释，从而弥合生成有效反事实解释与向用户传达模型行为之间的差距。 |
| [^73] | [MicroBT: Centroid-Guided Synergistic Learning for Multi-Modal Micro Brain Tumor Segmentation and Counting](https://arxiv.org/abs/2611.13912) | 该论文提出MicroBT统一框架，巧妙利用“病灶质心具有尺度不变性”这一几何特性，通过质心锚定的协同学习范式，解决多模态脑肿瘤分割中小于3毫米微小病灶被抑制以及病灶计数不可靠的问题。 |
| [^74] | [Region-Wise Logit Fusion and Utility-Gated Refinement for Brain Metastasis Segmentation](https://arxiv.org/abs/2611.13884) | 该论文提出COMPASS-METS框架，通过融合多个nnU-Net配置的区域概率集成、结构化概率锚点与效用门控精细化，在保持ET/TC/WT/RC层级结构的同时有效检测脑转移瘤的小强化病灶。 |
| [^75] | [Transparency Disclosures for Chain-of-Thought Monitorability](https://arxiv.org/abs/2611.13873) | 该论文提出覆盖思维链访问、质量（自然性与忠实性）和监控有效性（检测与能力提升）的五个透明度披露指标，呼吁前沿AI开发者定期报告这些指标以保护思维链监控能力免受威胁，审计显示目前尚无开发者覆盖全部五个指标。 |
| [^76] | [Mechanistic Interpretability of Deep Neural Networks: Methods, Frameworks, and Perspectives](https://arxiv.org/abs/2611.13868) | 本文系统综述了深度神经网络机制可解释性的方法与框架，其核心在于通过识别对特定行为具有因果影响的内部组件来揭示网络的内部工作机制。 |
| [^77] | [Taxonomy-Agnostic Neurosymbolic Grounding of Robot Kinematics](https://arxiv.org/abs/2611.13850) | 该论文提出URDF2Ontology框架，利用LLM语义推理与形态感知的双轨架构，自动将原始URDF文件转换为与任意用户分类体系对齐的填充式OWL本体，为机器人物理能力的符号推理提供接地。 |
| [^78] | [Satellite Pretraining Is Not One Thing: A Matched-Pair Study of Generic and Earth-Observation Foundation Models for Scene Classification](https://arxiv.org/abs/2611.13849) | 该研究通过五组共享目标与架构的编码器配对实验和显著性检验发现，对地观测预训练并不必然优于通用预训练：它显著提升了SatDINO、Scale-MAE、RemoteCLIP等模型的场景分类性能，却显著损害了DINOv3-SAT等另一部分模型的表现。 |
| [^79] | [PolyRoute: A Twelve-Language and Code-Mixed Benchmark for Typed-Decision Routing, with Baselines and a Leaderboard](https://arxiv.org/abs/2611.13843) | 该论文发布了PolyRoute基准，涵盖12种语言、9种书写系统及混码文本的类型化决策路由评测，并提供13个系统的排行榜，显示微调模型可达96.6%准确率，且多语言模型对未公开语言具有良好的泛化能力。 |
| [^80] | [MusicFlux: Flexible Conditioning for Text-Guided Music Generation and Editing](https://arxiv.org/abs/2611.13834) | MusicFlux是一个基于预训练流匹配文本到音频模型的文本引导音乐编辑器，通过两种条件化方法在编辑质量与计算效率之间实现权衡，无需处理过长的音频序列即可统一音乐的生成与编辑。 |
| [^81] | [Ceci n'est pas une pipe -- Toward a Symbolic Solver for the Abstraction and Reasoning Corpus Using Evolutionary Computation](https://arxiv.org/abs/2611.13831) | 该论文提出了首个提交至ARC-AGI-2主竞赛的完全符号化、基于进化计算的程序合成求解器，并着力解决进化算法在领域特定语言表示选择与等价候选解区分方面的固有难题。 |
| [^82] | [CASCADE: Constraint-Aware SCAffolded DEcomposition for Logical Anomaly Detection](https://arxiv.org/abs/2611.13820) | 论文提出CASCADE方法，通过约束感知的支架式分解为逻辑异常检测中的每类约束定制视觉证据提取，弥合了LVLM清单式验证中“约束特定感知鸿沟”所导致的实例混淆与尺度漂移问题。 |
| [^83] | [Swanada: Split-Encoder Capability-Routed Conformer CTC and Sub-30 MB On-Device Speech Recognition Across Twelve Scripts](https://arxiv.org/abs/2611.13814) | Swanada 提出了一种分裂编码器、能力路由的 Conformer CTC 架构，将覆盖十二种文字的开源流式端侧语音识别模型压缩到每个低于 30 MB，同时解决了浏览器与移动端部署中的存储配额、算子兼容、解码闪烁和音频管线四大系统瓶颈。 |
| [^84] | [Relaxing Cross-Modal Correspondence for Multimodal Remote Sensing Pretraining](https://arxiv.org/abs/2611.13807) | 该论文提出 SemPair 预训练策略，将多模态遥感预训练中的跨模态对应要求从实例级放宽到语义级，用类别一致的跨实例配对替代严格配对，并通过潜在因子模型理论分析和 MixPair 插值方法系统研究对应关系放宽的效果。 |
| [^85] | [Bridging Graph and Pairwise Learning for Gene Regulatory Network Inference from scRNA-seq Data](https://arxiv.org/abs/2611.13801) | 提出scGPL端到端统一框架，融合图学习与成对学习两类方法的优势，联合建模全局调控拓扑与TF-靶基因对，解决了图方法易过拟合、成对方法忽视全局拓扑结构的局限，实现更准确的基因调控网络推断。 |
| [^86] | [Integrating Action Prefix Prediction and Alignment Across Modalities for Multimodal Reinforcement Learning](https://arxiv.org/abs/2611.13782) | 提出改进预测表征方法，将动作前缀预测与基于观测锚点的跨模态对比对齐相结合，直接从观测锚点和动作序列预测未来表征，在DeepMind Control Suite五个任务上平均回报显著优于基线。 |
| [^87] | [Softness is something you also need! Why compliance helps control: information metrics of a passive walker with compliant ankle](https://arxiv.org/abs/2611.13779) | 本文通过回顾信息度量与智能体动力学之间的数学联系，并应用于带柔性踝关节的被动行走器，揭示了柔顺性有助于控制的机理，从而倡导超越笛卡尔式身心分离、基于形态计算与涌现的机器人设计新范式。 |
| [^88] | [DisasterAgent: An Evidence-Driven Agent for Verifiable Disaster Report Generation From Remote Sensing Imagery](https://arxiv.org/abs/2611.13776) | 提出了证据驱动的智能体DisasterAgent，通过“感知-增强-生成”工作流、视觉专家工具包和时空感知图（STPG），从双时相遥感影像中生成附带可验证证据的灾害报告。 |
| [^89] | [Mixed Signals: Weight-Space Programming for EEG and ECG](https://arxiv.org/abs/2611.13758) | 本文首次将权重空间编程应用于脑电图与心电图等生物信号模型，证明语言模型不仅能诊断并修复植入的模型故障，还能仅凭推理从零写出网络全部权重，并以少6倍的训练数据达到与梯度下降训练相当的效果。 |
| [^90] | [A Hardware-Trained Generative Quantum Eigensolver for Sample-Based Quantum Diagonalization of Organotin Hydroxides up to 70 Qubits](https://arxiv.org/abs/2611.13755) | 该论文首次将IBM Heron量子处理器纳入训练回路，在高达70量子比特上硬件训练生成式量子本征求解器（GQE），实现了有机锡氢氧化物分子的样本型量子对角化，且训练好的策略无需再训练即可迁移到新分子体系。 |
| [^91] | [Toward Autonomous RTL and Synthesis Optimization: Benchmarking and Self-Improving Agents](https://arxiv.org/abs/2611.13713) | 本文提出迈向自主RTL与综合优化的三个方向，并介绍自我改进的多智能体框架Dr. RTL，其在20个真实设计上相比商业综合工具实现平均WNS/TNS提升21%/17%、面积减少6%。 |
| [^92] | [Broad-Persistent Advising in Interactive Reinforcement Learning via Latent State Encoding](https://arxiv.org/abs/2611.13712) | 本文提出Encoder BPA方法，用潜在状态编码器替代传统K-means聚类，通过潜在比较、top-k聚合和一致性条件来检索和重用外部建议，从而提升了交互式强化学习早期的学习效率和建议利用率。 |
| [^93] | [AutoManifold: Agentic Design of Data Visualisation Algorithms via Manifold Embedding](https://arxiv.org/abs/2611.13710) | AutoManifold 提出了一种由多智能体大语言模型驱动的算法生成流水线，能够根据用户指定的结构保持偏好，从受约束的原语词汇中自动组合并迭代优化流形嵌入算法，实现面向用户偏好的数据可视化算法自主设计。 |
| [^94] | [Agentic AI Meets Formal Verification in the Era of Agentic Hardware Development](https://arxiv.org/abs/2611.13705) | 本特邀论文综述了智能体AI与硬件形式化验证的双向协同关系：AI从规约、证明构建和推理引擎三个层面推进形式化方法的发展，而形式化方法则通过RTL生成、调试和保正确性优化保障AI驱动硬件开发的可靠性。 |
| [^95] | [SeqFold: Timestamp-Addressable Compressed States for User Action Sequence Backbones](https://arxiv.org/abs/2611.13704) | SeqFold通过可学习的多通道融合、通道级时间衰减与无查找量化（LFQ），将冻结长历史骨干网络的多层用户状态压缩为时间戳可寻址的紧凑离散编码，供下游任务直接对齐复用。 |
| [^96] | [Topological Dynamics Under Non-Stationary Supervision: Persistent Homology of Neural Network Weight Evolution](https://arxiv.org/abs/2611.13701) | 本研究利用持续同调追踪不同标签监督条件下神经网络权重点云的拓扑演化，发现正确标签模型的H1持续性在描述性上高于随机标签条件，但经多重检验校正后未达统计显著性。 |
| [^97] | [On Private Knowledge Retrieval and Private Inference for secure multi-agent systems](https://arxiv.org/abs/2611.13696) | 本文针对多智能体系统中语义记忆面临的记忆投毒攻击，提出了私有知识检索与私有推理作为防御与缓解策略，以保障智能体系统的安全。 |
| [^98] | [AURA: Auditory Spatial Intelligence via a Unified Representation from a General-Audio Encoder through Complete Alignment](https://arxiv.org/abs/2611.13687) | 该论文提出AURA，证明未经空间监督训练的通用音频编码器本身通过双耳通道差异保留了空间信息，并通过完全对齐构建统一表示，克服音频-文本信息不对称问题，使单一编码器在多声源场景中同时实现声音识别与空间定位。 |
| [^99] | [Ego Guidance: Enhancing Robot Policy Learning with Action-Free Human Egocentric Videos](https://arxiv.org/abs/2611.13685) | 本文提出SPAR架构，在预训练VLM基础上让未来视觉状态预测与动作预测共享学习，从而利用无动作标注的人类第一视角视频增强机器人策略学习，其学到的表征可提升分布外机器人任务性能并随视频数据规模扩大而持续提升。 |
| [^100] | [Bootstrapped On-Policy Distillation](https://arxiv.org/abs/2611.13680) | 该论文发现在线策略自蒸馏（OPSD）在持续训练中会因冻结的特权教师与不断进化的无参考学生之间日益扩大的分布失配而导致性能退化，并提出自举式在线策略蒸馏方法来解决这一问题。 |
| [^101] | [A computational framework for understanding embodied plans](https://arxiv.org/abs/2611.13669) | 该论文提出GenEA框架，通过将融合身体生物力学与直觉物理学的多层级状态空间嵌入生成模型，实现了从人类动作出发对多步骤长时程具身计划与目标的联合概率推理。 |
| [^102] | [Benchmarking LLM Next-Step Recommendations with Knowledge Space Theory](https://arxiv.org/abs/2611.13667) | 该论文提出了一种以知识空间理论（KST）为形式化参照的基准测试方法来评估大语言模型在教育中的下一步推荐能力，实验表明提供知识空间结构信息（如先决条件关系或完整知识空间）可将推荐精确集合准确率从约61%大幅提升至96%以上。 |
| [^103] | [DrugCoding: A Scientific Programming Agent for Drug Discovery](https://arxiv.org/abs/2611.13651) | 提出了DrugCoding，一个融合科学任务规划、自适应模型路由、多智能体协作与程序化工具执行的科学编程智能体系统，能够自动化复杂的药物发现工作流程。 |
| [^104] | [MOAI: Module-Optimizing Architecture for non-interactive secure transformer Inference](https://arxiv.org/abs/2611.13641) | MOAI提出了一个基于同态加密的非交互式安全Transformer推理框架，通过统一的列/对角线打包评估流程、无旋转的Softmax和LayerNorm算法以及交错批处理等模块级优化，大幅消除了昂贵的HE旋转操作与格式转换，显著提升了加密推理效率。 |
| [^105] | [Synapse-Control: Making Agent Experience Actionable with Lightweight Runtime Control](https://arxiv.org/abs/2611.13615) | Synapse-Control用一个轻量级智能体运行时控制平面（规则筛选、1.7B微调分类器评分、微合同接口检查和提示价值表冷却机制）取代大模型失败诊断，以更低的推理成本实现智能体执行经验的高效复用与精准投放。 |
| [^106] | [Megrez: A Knowledge-Guided Agent for Formal Mathematics over MK Set Theory](https://arxiv.org/abs/2611.13613) | Megrez提出了一个基于MK集合论的知识引导形式化智能体，通过可重建知识图谱、反馈驱动的短程证明循环等三项机制，解决了在低资源数学库中进行累积性教材形式化时的概念对齐、长程搜索同步和证明集成三大难题。 |
| [^107] | [Earth Embeddings in Environmental Remote Sensing](https://arxiv.org/abs/2611.13612) | 本文系统阐述了地球嵌入的理论基础并回顾了代表性产品，展示其作为多源地球观测的通用表示，能够在无需任务特定设计与预处理的情况下广泛支持环境遥感下游任务。 |
| [^108] | [Towards Reliable VLM Judges: State-Conditional Invariance and Presentation-Aware Diagnostics](https://arxiv.org/abs/2611.13596) | 该论文提出状态条件不变性（SCI）这一可靠性原则，并构建了基于模拟器的诊断基准GridWM-Judge，用于系统评估视觉语言模型评判者在语义不变的呈现变化下是否保持稳定、并对改变任务结果的因果干预作出正确响应。 |
| [^109] | [Transformers Provably Learn Graph Search: Training Dynamics and the Exponential Power of Depth](https://arxiv.org/abs/2611.13594) | 本文证明了 Transformer 进行图上 DFS 推理时第二层是近线性规模实现的必要且充分条件，并且 k 层模型的可达性前瞻视野随深度呈指数级增长。 |
| [^110] | [PhyRubric: Rubric-Grounded Diagnostic Evaluation of Physical Commonsense in Text-to-Video Generation](https://arxiv.org/abs/2611.13576) | 提出基于评分准则的诊断性评估框架 PhyRubric，在观看视频前为每条描述构建专属评分准则并分解为语义与物理条目，再基于时间戳证据进行条目级判定，从而可解释地评估文本到视频模型是否符合物理常识。 |
| [^111] | [Market-Conditioned JEPA: Learning from Unpredictable Market Shocks](https://arxiv.org/abs/2611.13572) | 提出市场条件化JEPA（MC-JEPA），通过以留一法构建的市场因子作为预测条件，使资产的市场暴露成为最优预测的必要信息，并在11年中国A股分钟级数据上全面超越单一资产的JEPA方法。 |
| [^112] | [Metamorphic testing of learned AC optimal power flow surrogates reveals representation dependence](https://arxiv.org/abs/2611.13566) | 该论文提出用十三项蜕变关系系统审计学习型交流最优潮流代理模型，发现所有被测配置均存在对输入序列化表示的依赖性，并借此定位出索引错误与不稳定预处理等缺陷，同时证明数据增强可以缓解这种表示依赖问题。 |
| [^113] | [A Concept-Based Interpretable Framework for Clinical Text-based Prediction via Token-Concept Attention without Concept Supervision](https://arxiv.org/abs/2611.13564) | 该论文提出了一种无需概念标注监督的可解释框架，通过定制交叉注意力瓶颈机制，利用预训练语言模型估计临床文本词元与医学概念之间的相关性，从而在临床预测任务中实现基于概念的黑盒模型解释。 |
| [^114] | [Benchmarking Vendor-Guided Optimizations on Edge NPUs: From NPU Only to End-to-End Performance. Extended technical report](https://arxiv.org/abs/2611.13563) | 该论文提出了一种通过比较基线、厂商引导仅NPU（VG-NPU）和厂商引导端到端（VG-E2E）三种配置来分离硬件、模型转换与部署流程影响的基准测试方法，发现NPU级性能排名与端到端性能排名可能不一致，且厂商原生部署流程可额外降低10.4–70.7%的系统级延迟。 |
| [^115] | [Are Dense Rewards Worth Their Tokens in GRPO?](https://arxiv.org/abs/2611.13560) | 尽管蒙特卡洛密集过程奖励在单个更新比较中优于标准GRPO，但在以生成token数和GPU小时数衡量的计算量公平匹配下，其高昂的采样开销（占生成token的91%）使结果排名发生逆转，密集奖励并不值得其token代价。 |
| [^116] | [Darmok: Championship Cryptic Crossword Solving via Constraint Tightening and Resilient Orchestration](https://arxiv.org/abs/2611.13557) | Darmok 是首个在英国密码填字游戏上达到超人水平的自动化系统，通过约束收紧与弹性编排技术，在《泰晤士报》填字游戏锦标赛中无错误解出全部七道谜题，且每一轮速度均超过所有人类选手。 |
| [^117] | [Massive Activations Confound Sparse Autoencoders](https://arxiv.org/abs/2611.13548) | 该论文发现模型中的巨大激活使高达99.9%的激活方差集中于第一主成分（PC1），导致稀疏自编码器的评估指标失真、重要特征在不同运行间无法复现，并提出在去除PC1的激活上训练SAE、再将PC1加回重建结果的修正方法，将前50个特征的共享率从4–39.5%大幅提升至73–99.5%。 |
| [^118] | [Agent Benchmarks Stop Ranking Before They Run Out of Tasks](https://arxiv.org/abs/2611.13541) | 该论文通过对八个基准的运行结果重新评分发现，智能体基准的饱和主要源于“单比特通过/失败”的测量方式过于粗糙且以参考标准封顶，而非任务难度不足，因此单纯推出更难版本并非解决之道。 |
| [^119] | [Developing an Operational Model for an Internet of Data Based Trusted Data Space for Vocational Education](https://arxiv.org/abs/2611.13520) | 本文基于数联网技术构建了职业教育可信数据空间的多方运营模型，通过价值网络、产品组合、收益分配和治理规则的系统设计，强调数据主权保护与激励相容，为区域职业教育数据生态系统的可持续运营提供了参考方案。 |
| [^120] | [Nepali-English Code-Mixed Speech Recognition: Data Preparation and Adaptation of a Speech Foundation Model](https://arxiv.org/abs/2611.13512) | 本研究通过为分词器不支持的数字生成尼泊尔语与英语读法并借助零样本ASR选取合理形式，成功将语音基础模型SraVaani适配到尼泊尔语-英语码混合语音识别任务，将词错误率从48.6%降至16.3%，优于微调后的Whisper large-v3-turbo。 |
| [^121] | [Dark Matters: VAE-Aware HDR Encoding for SDR-to-HDR Video Reconstruction](https://arxiv.org/abs/2611.13504) | 提出EquivFlow，一种紧凑可逆的曝光等变HDR编码，解决了类对数编码在冻结VAE潜空间中导致的暗部色彩坍缩问题，无需修改VAE即可提升SDR到HDR视频重建的暗部细节与色彩恢复能力。 |
| [^122] | [Reasoning-Aware Relational Representation Learning for Open-Vocabulary Scene Graph Generation](https://arxiv.org/abs/2611.13496) | 提出 RaRe 框架，将 MLLM 生成的思维链推理依据直接转化为并优化为结构化关系嵌入，从而提升开放词汇场景图生成中稀有关系的识别能力和对未见谓词的泛化能力。 |
| [^123] | [The WoW: A Large-Scale World Knowledge Corpus for Full-Lifecycle LLM Training](https://arxiv.org/abs/2611.13487) | 本文提出WoW——一个由教育知识图谱驱动、支持大语言模型全生命周期训练的大规模世界知识语料库，在多个基准上显著提升模型性能，并提供交互式图谱数据选择与检索平台。 |
| [^124] | [Sharp Sample Complexity of KL-Regularized Reinforcement Learning](https://arxiv.org/abs/2611.13480) | 本文证明了在KL正则化强化学习中，插件估计器仅需Õ(KH³/(ε·max{ε,β})+KH)个样本即可找到ε-最优策略，并给出了匹配的下界，将先前样本复杂度对有效视界的依赖从H⁵改进为H³，且首次建立了依赖温度β的下界。 |
| [^125] | [Leveraging Psychophysical Attentional Distribution for Gaze-Augmented Reward Modeling](https://arxiv.org/abs/2611.13477) | 该论文基于认知心理学中人类字母识别的发现，将视觉注意力建模为以注视点为中心的空间梯度分布并融入RLHF奖励模型训练，从而实现了比基线更高的偏好预测准确率。 |
| [^126] | [When Reasoning looks like Hidden Understanding: How Chain-of-Thought Fosters Anthropomorphism, Perceptions of Epistemic Superiority, and Cognitive Deference](https://arxiv.org/abs/2611.13456) | 该论文指出，思维链推理使AI看起来像拥有隐藏的“理解力”，这种拟人化认知以及对AI认知优越性的感知会促成人们对AI系统的认知服从，从而削弱人类自身的能动性。 |
| [^127] | [SafeRoute-MAS: Coverage-Aware and Risk-Calibrated Message Routing for Multi-Agent Scientific Reasoning](https://arxiv.org/abs/2611.13454) | SafeRoute-MAS提出了一种仅利用集成前信息、覆盖率感知且风险校准的消息路由程序，在多智能体科学推理中通过选择性移除专家消息，在减少约12%–19%消息令牌的同时带来一定的准确率提升，但严格的统计验证显示其增益有限。 |
| [^128] | [ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding](https://arxiv.org/abs/2611.13442) | 该论文提出工作保持投机解码（WCS），通过联合调度每个请求的草稿生成与验证预算，解决了并行投机解码中草稿计算大量浪费的问题，从而提升高负载下LLM服务的有效吞吐量。 |
| [^129] | [Inherited Weights: Measuring the Tokenization Premium and the Sovereignty Cost of Language Models for the Global South](https://arxiv.org/abs/2611.13441) | 论文指出微调模型会从基础模型继承词表、知识、文化对齐、数据与许可证等决定性属性，并测量出全球南方语言承受最高达10倍的分词溢价，证明构建本地化词表可将该溢价显著降低约24%，从而揭示了大语言模型给全球南方带来的主权成本。 |
| [^130] | [You Don't Need Feature Learning to Do In-Context Learning](https://arxiv.org/abs/2611.13431) | 通过对核机器的谱分析发现，上下文学习中的泛化解只是记忆解的低阶截断——模型先泛化后逐渐记忆化，因此实现上下文学习并不需要特征学习。 |
| [^131] | [Free Lunch Uncertainty: Sequence-Level R\'enyi Entropy from Particle Decoding for Adaptive LLM Reasoning](https://arxiv.org/abs/2611.13414) | 该论文发现SMC幂采样解码可零边际成本获得序列级Rényi熵，其中1/2阶熵在13个设置中的10个中表现最佳，是用于LLM自适应推理的零成本选择性预测信号。 |
| [^132] | [Placing LLM Lookup Tables by Per-Layer Prefetch Deadlines](https://arxiv.org/abs/2611.13413) | 提出 SlackTier，利用逐层预取时限（两个 Engram 层窗口相差 30 倍）将 LLM 的大规模 n-gram 查找表按层放置到不同存储层级，在解码 CUDA 图内实现位精确查找，使部分表可无需缓存直接放在 NVMe 上而几乎不损失性能。 |
| [^133] | [Bursts That Outlast Themselves: Persistent Degradation in Prefix-Cached Agent Serving](https://arxiv.org/abs/2611.13412) | 本文发现前缀缓存的 LLM 智能体服务具有亚稳态特性：即使负载恢复，短短几分钟的请求突发也会使服务器陷入缓存命中率骤降至约 10%、吞吐量减半且持续恶化的退化状态，其根源是滞留会话的工作集超出 KV 缓存池，导致每轮交互都变成代价高昂的重新计算。 |
| [^134] | [SteerDistill: Quantizing LLM Agents via Closed-Loop Teacher Steering](https://arxiv.org/abs/2611.13410) | 提出了SteerDistill，一种量化感知的蒸馏方法，通过全精度教师模型以闭环方式引导学生修订失败响应，并利用学生执行反馈迭代优化引导，从而在低比特量化下保持LLM智能体的任务成功率。 |
| [^135] | [Stinging the Swarm: Extracting Attacker Intelligence from a Twin-Controller OT Honeypot](https://arxiv.org/abs/2611.13406) | 本文提出Talos内联网关系统，通过毫秒级请求判断和设备级逼真的镜像孪生蜜罐，在完全阻断LLM自主智能体对真实工业控制器破坏的同时，从中提取攻击者情报。 |
| [^136] | [Living constitutions: Participatory alignment for sovereign foundation models](https://arxiv.org/abs/2611.13402) | 提出将主权基础模型“宪法”的制定视为一个类似公民科学、持续演进的参与式过程，并确立迭代性、模块化、可论证性、立场性和合法性五项原则，以避免模型对齐失败或默认接受前沿模型厂商的规范。 |
| [^137] | [The Dose Makes the Poison: Exposure-Aware Evaluation of Generative Drug Design](https://arxiv.org/abs/2611.13398) | 该论文利用基于生理的药代动力学（PBPK）模拟对生成式药物设计的候选分子进行暴露量感知的评估，揭示了传统静态分子性质筛选方法因忽略剂量可行性与安全性而会遗漏大量更优分子的重要缺陷。 |
| [^138] | [The AI Era's Lines of Code? A Case Study on What an Engineer's Token Spend Measures](https://arxiv.org/abs/2611.13396) | 研究发现工程师的token消耗主要由个体差异而非工作产出决定，其与代码提交量仅呈弱关联（消耗翻倍仅带来约10%的提交增长），因此token消耗并不是衡量工程生产力的可靠指标。 |
| [^139] | [Online-Adaptive Fuzzy Cognitive Maps for Neuro-symbolic Financial Forecasting](https://arxiv.org/abs/2611.13388) | 该论文提出两种神经符号金融预测框架：一种基于双源实时数据构建FCM并通过误差驱动Hebbian规则在线自适应更新，再以非对称加性残差融合方法与LSTM预测无缝结合；另一种是基于文献并由大语言模型增强、每条边附带可验证引用的可审计FCM，从而同时兼顾时序建模能力、因果可解释性与在线适应性。 |
| [^140] | [How Hyperbolic Are Hyperbolic Vision-Language Models?](https://arxiv.org/abs/2611.13384) | 研究发现主流双曲视觉-语言模型中的指数映射实际近似于恒等映射，移除它对性能没有影响，且使用相同层次损失训练的纯欧氏模型表现几乎与之无异，说明这些“双曲”模型在实践中几乎并不真正双曲。 |
| [^141] | [Enforcing Axioms for AI Alignment under Loss-Based Rules](https://arxiv.org/abs/2611.13369) | 本文将AI对齐原则建模为嵌入空间中的线性“投票者”，在基于损失的奖励模型训练框架下研究帕累托最优等公理能否成立，发现即使采用更丰富的奖励模型或修改比较数据，帕累托最优性仍可能被违反。 |
| [^142] | [No finite rulebook can certify every safe program: an impossibility theorem for gatekeeping AI agents](https://arxiv.org/abs/2611.13367) | 本文证明了一个不可能性定理：由于机器可检验的证据只能确认有限记录所能确定的内容，而“从不”与“最终”类性质无法被有限记录确定，因此不存在完备的有限规则手册能认证所有安全程序——禁止性规则只能单向地发现违规却无法永久放行安全程序，保密性与响应性规则则连违规都无法可靠判定。 |
| [^143] | [A Technique to Model Product Information and User Reviews (MPIUR)](https://arxiv.org/abs/2611.13361) | 提出MPIUR多模态图推荐模型，将用户评论编码进异构图并对齐图文表示，在交互历史稀疏的场景下也能显著提升推荐性能（nDCG@10与Recall@10）。 |
| [^144] | [When Are Predictions Enough? An Evaluation Protocol for Frozen Expert Composition](https://arxiv.org/abs/2611.13356) | 该论文提出了一个可复用的评估协议来公平比较冻结专家模型的特征组合与预测组合，并发现对六个架构多样的深度伪造检测专家特征进行逻辑回归，在留一生成器评估中比最佳预测组合器的宏平均 AUROC 提升 0.081。 |
| [^145] | [Multi-Stage Phase-Aware Self-Supervised Reconstruction for Multi-Echo Gradient-Echo MRI](https://arxiv.org/abs/2611.13352) | 提出了一种多级相位感知自监督重建框架MSPR-GRE，通过渐进式k空间扩展与相位估计精化相结合，在无需全采样训练数据的条件下实现加速高分辨率多回波梯度回波MRI重建，同时保留解剖细节与复数回波信息。 |
| [^146] | [ProTEUS: A mutation-resolved world model for closed-loop protein evolution](https://arxiv.org/abs/2611.13348) | 该论文提出ProTEUS——一个基于深度突变扫描预训练的突变解析世界模型，通过MutationFormer突变偏置注意力架构联合建模序列、进化与结构信息，并配合快速的物理约束全原子预测器，实现了闭环蛋白质进化中突变空间的高效计算探索与功能预测。 |
| [^147] | [Veritas: A Verifier Generation Harness for Agent Tasks](https://arxiv.org/abs/2611.13336) | Veritas 是一个自动生成智能体任务验证器的框架，通过引入证据头、领域规则库和编写原则三种机制，使其生成的验证器与官方判定结果的一致性在多个基准上超越其他七个框架。 |
| [^148] | [Disentangling Multimodal Rotary Positional Embedding in Vision-Language Models](https://arxiv.org/abs/2611.13334) | 该论文将视觉内部空间结构与跨模态空间效应解耦，研究 MRoPE 为文本 token 赋予空间坐标对文本-视觉注意力究竟是有益的空间参考还是有害的位置偏好，并识别出共享坐标布局带来的约束。 |
| [^149] | [Do AI Agents need Preventive Care? An Epidemiologic Cohort Design for Agentic AI](https://arxiv.org/abs/2611.13333) | 本文借鉴人类流行病学队列研究方法，提出将AI智能体视为队列成员进行纵向追踪的评估框架，用以研究智能体不良事件的行为决定因素及其可预防负担。 |
| [^150] | [Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA](https://arxiv.org/abs/2611.13331) | 该论文提出在联邦学习中冻结并周期性重采样LoRA的A矩阵，从理论上证明这等价于对全量微调的随机近似，从而解耦A、B梯度并将计算与上行通信开销降低50%。 |
| [^151] | [Agent-Guided Discovery of Exchange-Correlation Functionals with Physics-based Validation](https://arxiv.org/abs/2611.13329) | 该研究将前沿大语言模型智能体与GPU评估器通过模型上下文协议相连，并引入基于物理的自洽与外部验证流程，最终以远少于对比模型的token用量发现了一个性能更优的13参数交换-关联泛函O159。 |
| [^152] | [F3-AGI v4.0: Bioenergetically-Gated Active Inference, Metaplastic Preconditioning, and Residual Program Synthesis for Sample-Efficient Abstract Reasoning](https://arxiv.org/abs/2611.13325) | F3-AGI v4.0提出了一种CPU原生的神经符号主动推理架构，通过生物能量门控可塑性、自适应复杂度引导程序合成与期望自由能探索三大机制，克服ARC-AGI少样本抽象推理中的罗生门效应，实现样本高效的算法变换归纳。 |
| [^153] | [Towards Scalable Network-Based Interventions for HIV Treatment and Prevention via R2C-Agent](https://arxiv.org/abs/2611.13312) | 该论文针对接触网络难以大规模获取的问题，提出利用R2C-Agent方法，基于可大规模数字化收集的同伴推荐网络开发可扩展的HIV治疗与预防干预策略。 |
| [^154] | [PegasusAI Studio: Legible Cyberinfrastructure and Accountable AI Agents for Scientific Workflows](https://arxiv.org/abs/2611.13309) | PegasusAI Studio通过设施知识库投射、工作流接口内省和遥测数据诊断，使AI编程智能体能够“读懂”共享网络基础设施，并在结合联邦认证与按用户权限的信任包络内安全地构建和运行科学工作流。 |
| [^155] | [Train High, Ship Low: Efficient LoRA Synchronization for Distributed On-Policy Post-Training](https://arxiv.org/abs/2611.13306) | 提出与压缩器无关的 HiLo 框架，用压缩梯度分发加本地优化器重放取代 LoRA 参数传输，并配套预算感知的 Spar 梯度压缩器，将训练秩与通信成本解耦，从而在带宽受限网络上高效实现分布式在线策略后训练。 |
| [^156] | [MRFoundry: Knowledge-guided agent loops for end-to-end MRI research](https://arxiv.org/abs/2611.13289) | MRFoundry是一个开源、可扩展的知识引导智能体框架，将MRI专业知识与智能体行为和输出验证相连接，并配套推出MRI Agent Bench基准测试，显著提升了AI智能体完成端到端MRI研究任务的成功率。 |
| [^157] | [Off-policy Learning with Excursion Policies](https://arxiv.org/abs/2611.13275) | 该论文提出“游走策略”这一结构化插值方法——智能体先按目标策略执行多步再回归行为策略，建立了其动态规划理论基础并开发了可证明可靠的离策略时序差分算法，在连续控制基准上实现了与混合策略相当的效率-稳定性权衡。 |
| [^158] | [Adaptive Residual Quantization for Memory-Efficient Temporal Action Segmentation](https://arxiv.org/abs/2611.13262) | 提出自适应残差量化变分自编码器（ARQ-VAE），通过学习无标签的离散残差量化并逐帧自适应截断残差链，为时序动作分割提供内存高效且灵活适应帧复杂度变化的特征记忆表示。 |
| [^159] | [Dependable Long-Horizon LLM Agents: A Trajectory-Centered Empirical Study of Where Errors Enter and Which Controls Reach Them](https://arxiv.org/abs/2611.13254) | 该研究提出了一种将LLM智能体故障与起源、传播、检测、响应和后果相绑定的“事件记录”标注协议，并通过对六个多智能体系统95次失败运行的完整轨迹实证分析，发现故障多在任务早期（中位数0.18处）以错误决策或状态捏造的形式进入，且往往早于人工标注所认定的决定性失败步骤。 |
| [^160] | [Curie's Jarvis: Autonomous Robotic Chemistry Experiments from Printed Protocols](https://arxiv.org/abs/2611.13251) | Curie's Jarvis 让机械臂无需预置脚本和位置标记，直接从打印的纸质实验方案中读取步骤，在面向人类的普通实验台上自主执行化学实验，并通过多视角读取标签区分相同容器、在每步操作前重新定位目标。 |
| [^161] | [ExpertCrowd-SWE: Measuring Frontier Coding Agents on Expert-Authored Engineering Tasks in Unfamiliar Codebases](https://arxiv.org/abs/2611.13246) | EC-SWE是一个包含100个由专家编写、覆盖87个开源项目和11种语言的多文件软件工程任务基准，通过密封参考方案、可执行加权评分和对抗性对照补丁，在大型陌生代码库中严格评估自主编程智能体的真实工程能力。 |
| [^162] | [STAR: Semantic-Criticality Tiered Allocation of Radio Resources via Zero-Shot Learning](https://arxiv.org/abs/2611.13238) | 提出了STAR框架，利用冻结视觉-语言模型的注意力导出逐像素语义关键性指示器，对决定任务答案的关键像素进行分层高保真无线电资源分配，且仅需更改提示词即可零样本适配新任务与领域。 |
| [^163] | [NCP: Verifier-Guided Optimization of Weight-Space Control Policies](https://arxiv.org/abs/2611.13236) | 本文提出神经控制平面（NCP），一种闭环系统，将预训练模型的推理数据平面与控制策略分离，自动合成、评估并管理临时的权重空间干预规则，同时通过基础模型参数的可恢复性保证安全，实验中成功将 SmolLM-135M 的验证集 token 重复率从 0.4133 降至 0.2191。 |
| [^164] | [Decisions by Discrimination: Differential Evidence for Medical Reasoning](https://arxiv.org/abs/2611.13233) | 该论文提出了一种基于检索的差异化证据框架，通过提取每个选项特有的决定性差异化证据并组织成简洁的证据提示，来更好地区分医学多选题中的易混淆选项。 |
| [^165] | [W-ISA: Transactional Execution of Weight-Space Intervention Programs](https://arxiv.org/abs/2611.13232) | 该论文提出了 W-ISA——一套用于在预训练语言模型上事务化执行权重空间干预程序（参数更新、消融、秩一变换等）的软件指令集与运行时，可在请求结束后自动恢复模型参数，其效果与等效的 PyTorch 钩子实现完全一致。 |
| [^166] | [Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models](https://arxiv.org/abs/2611.13223) | 本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。 |
| [^167] | [Wrong the Same Way: Cross-Prompt Sharing Both Suppresses and Amplifies Reward Hacking in RLVR](https://arxiv.org/abs/2611.13218) | 本研究在可控环境中证明RLVR在KL正则化最优时只是对被接受集合重新加权的过滤器（奖励作弊率精确继承自基础模型），而跨提示参数共享则通过提示间奖励与经验切核的协方差项产生漂移，使接受同一错误答案的提示推高它、拒绝的提示压低它，从而同时抑制并放大奖励作弊。 |
| [^168] | [Reliability of Claude Code as an Agentic AI Grading System for Handwritten University Exams](https://arxiv.org/abs/2611.13217) | 本研究首次系统评估了基于Claude Opus 4.6的智能体式系统Claude Code对手写大学考试评分的可靠性，考察其与专家人工评分在总分及五种题型上的一致性、跨会话的稳定性以及AI反馈的准确性。 |
| [^169] | [Counting Under Disagreement: Dollar-Weighted Measurement, Selective Machine Labelling and Budgeted Human Review from Disagreeing AI Readers of Administrative Text](https://arxiv.org/abs/2611.13215) | 针对多个AI阅读者对行政文本标注存在分歧、且测量目标为货币总额的场景，本文提出了融合选择性机器标注与预算化人工复核的美元加权估计框架，在可控美元加权误差下实现对总额的准确测量。 |
| [^170] | [Before Words, Beyond Speech: Evaluating Nonverbal Social Reasoning in Early Childhood](https://arxiv.org/abs/2611.13213) | 该论文提出了首个以儿童为中心的非语言社会推理基准NEST，通过人工标注的约12秒自然情境视频片段，以及将基础感知与高阶推理相分离的组合式评估方案，填补了多模态视频理解领域对幼儿非语言社会互动评估的空白。 |
| [^171] | [XAI-Driven Confidence Estimation for OOD-Resilient Runtime Prediction](https://arxiv.org/abs/2611.13199) | 该论文提出LEX——首个用于基于机器学习资源管理的事后XAI运行时置信度估计框架，通过高斯混合模型聚类训练输入嵌入并关联逐簇预测误差界限，无需重训练即可提供细粒度的输入级置信度信号，在OOD工作负载上实现约94%的包含率，并产生比全局基线紧约17%的置信度界限。 |
| [^172] | [Structured Kernel Surrogate Models for Multimodal Attribution](https://arxiv.org/abs/2611.13198) | 提出结构化核代理模型及SKS算法，利用结构化核岭回归捕捉样本间与模态间的交互作用，首次解决多模态系统中的跨样本模态归因问题。 |
| [^173] | [From Process Models to Composable Agentic Workflows: Combining BPMN, Agents and Enterprise Knowledge](https://arxiv.org/abs/2611.13196) | 本文提出基于BPMN的智能体工作流方法，通过“递归能力组合”将BPMN流程模型转换为可执行工作流，用传统BPMN控制确定性的流程、决策与人工审批，仅在选定环节引入LLM推理，从而让智能体能够执行业务流程并将其封装为可重用的智能体工具。 |
| [^174] | [Multimodal Road Closure Mapping via Cross-Modal Graph Attention](https://arxiv.org/abs/2611.13192) | 本文提出一个统一的多模态多任务框架，通过跨模态Transformer注意力整合图结构地图数据、卫星影像和街景图像，在单一架构中同时实现道路封闭分类与受影响路段的关联。 |
| [^175] | [AgentScript-Eval: Benchmarking LLMs and Agents for Code Generation in an Enterprise DSL](https://arxiv.org/abs/2611.13183) | 该论文提出了AgentScript-Eval基准，包含100个针对Salesforce企业级声明式DSL（AgentScript）的真实任务，首次系统评估了LLM和智能体在企业DSL上的代码生成能力，发现该任务极具挑战性，最佳模型零样本通过率仅为25.0%。 |
| [^176] | [AI Alignment Is Neither Solvable nor Sufficient](https://arxiv.org/abs/2611.13167) | 本文论证AI对齐这一目标存在根本缺陷：对齐定义不清、无法求解，且即使实现对齐也不足以化解风险，因为真正需要应对的是人机混合社会这一复杂系统中产生的涌现性风险。 |
| [^177] | [Cognitive Multi-Agent Search and Rescue: Language-Grounded Hierarchical Reinforcement Learning for Collaborative Robotics](https://arxiv.org/abs/2611.13161) | 该论文提出了一种基于语言接地的分层强化学习（LG-HRL）认知多智能体框架，利用大语言模型将人类自然语言指令转化为结构化语义线索，结合共享认知工作空间，实现无人机与地面无人车在搜索救援任务中的团队级协调协作。 |
| [^178] | [How Much of Your VLA Is kNN](https://arxiv.org/abs/2611.13160) | 研究揭示LIBERO基准上VLA的高成功率很大程度上只是近邻演示检索的体现——无需任何训练的检索方法即可达到91.7%成功率，并提出局部动作一致性指标来衡量检索质量并指导检索键训练。 |
| [^179] | [Everything in its right place: attention is a radio head](https://arxiv.org/abs/2611.13146) | 该论文提出注意力机制可以从物理同步中涌现——把历史词元建模为频率随时间漂移的振荡器、把当前词元建模为无线电接收器，通过求解Kuramoto同步问题闭式导出一个具有“锁定/静默”两相的注意力层，从而将Transformer注意力重新解释为无线电调谐接收过程，并自然实现选择、弃权、计数及超长序列泛化。 |
| [^180] | [Teaching the Harness: Student-Aware Skill Transfer for LLM Agents](https://arxiv.org/abs/2611.13144) | 提出学生感知教学方法，教师通过构建学习者画像、假设诊断与探针测试来理解学生模型在固定执行框架下的局限性，并据此编写经验证的可复用技能，使任务成功率达 86.46%。 |
| [^181] | [Autonomous Optimization of Numerical Algorithms by an AI Agent: Application to Atomistic Simulation](https://arxiv.org/abs/2611.13143) | 提出AIOA框架，利用AI智能体在验证器守护的自动化改进循环中自主优化原子模拟的数值算法，实现核心计算模块提速22-24倍，并使结构弛豫和反应路径计算所需的力评估次数减少3-17倍。 |
| [^182] | [DOBB: Dynamic Oriented Bounding Boxes for Multi-Agent Collision Avoidance under State Uncertainty and Heterogeneous Dynamics](https://arxiv.org/abs/2611.13142) | 提出动态定向包围盒（DOBB），一种在线几何表示方法，将状态估计不确定性、异构动力学及智能体响应特性转化为动态且依赖方向的安全边距，从而提升多智能体避碰在不确定条件下的可靠性。 |
| [^183] | [Agreeing on the Answer but Not the Reason: Measurement Limits of LLM Reasoning Failure Taxonomies](https://arxiv.org/abs/2611.13140) | 该论文表明，LLM推理失败分类的测量结果很大程度上取决于测量程序本身：过半失败案例可归入多个类别，类别占比随分类器测试顺序大幅波动（26.4%–76.4%），且人工标注者对类别归属的共识远低于对任务答案正确性的共识。 |
| [^184] | [Which Successes Make Good References? Action-Sequence Alignment for Multi-Turn On-Policy Distillation](https://arxiv.org/abs/2611.13135) | 该论文将多轮在线策略蒸馏中的成功参考选择建模为自由目标最优传输问题，并提出以最长公共子序列（LCS）为基础的动作序列对齐作为轨迹兼容性的可观测代理，为每条失败轨迹匹配最兼容的成功参考，从而避免误导性监督、提升蒸馏效果。 |
| [^185] | [Activation-Guided Graph Retrieval for Cognitive Memory Reconstruction in Language Model Systems](https://arxiv.org/abs/2611.13127) | 该论文提出了一种激活条件化的记忆图检索架构，通过请求与目标线索激活授权记忆图、扇出衰减控制有界遍历、实体递归发射线索，并在生成前重构类型化证据图，从而为长寿命AI系统实现认知记忆重构。 |
| [^186] | [Just Image Transformers with Late-Stage Pixel Refinement](https://arxiv.org/abs/2611.13120) | 该论文提出在JiT（无VAE潜空间的图像Transformer）后训练阶段添加简单的卷积像素通路层，且该通路只需在去噪最后约20%阶段激活，即可显著提升图像生成质量。 |
| [^187] | [ASPIRE: An Agentic System-Identification Platform for Iterative Reasoning and Exploration](https://arxiv.org/abs/2611.13114) | ASPIRE是一个由大语言模型智能体驱动的非线性系统辨识框架，能够自主选择并执行辨识方法，从测量数据构建闭式非线性模型，融入先验物理知识并独立验证，且支持自主与交互两种运行模式。 |
| [^188] | [Building World Models from Intrinsic Motivation](https://arxiv.org/abs/2611.13113) | 该论文将24种内在动机目标统一整合到一个基于模型的强化学习框架中，系统评估了它们所构建的世界模型在零样本、少样本和多任务控制下的表现。 |
| [^189] | [One Temperature to Rule Them All?](https://arxiv.org/abs/2611.13077) | 该论文研究了 pass@K 推理中温度多样性何时优于单一温度，发现虽然更难的问题更难共享高效的采样温度，但这一趋势仅随难度呈对数增长，表明单一调优温度在多数情况下已足够有效。 |
| [^190] | [Elicitation of the parameters of a multi-criteria sorting model](https://arxiv.org/abs/2611.13062) | 本文针对经过公理化刻画的非补偿性分类模型（NCS），提出了严格的直接参数引出程序，并设计了多种基于经典图探索方法的提问算法，用于确定充分准则联盟集合以及区分两个类别的轮廓参数。 |
| [^191] | [A Multi-Word Tokenizer for 204 Languages That Needs Fewer Tokens Than GPT-4o's, Including on English](https://arxiv.org/abs/2611.13059) | 该论文提出Havon多词分词器系列，通过在约200种语言上从头训练并引入多词训练阶段，在包括英语在内的161种语言上所需词元数量少于GPT-4o的分词器，在印度语言上的词元数中位数仅为GPT-4o的0.74倍。 |
| [^192] | [The Indic Token Tax in 2026: Measuring Tokenizer. Disparity and Its Cost for the Languages of India](https://arxiv.org/abs/2611.13058) | 该研究测量了九个主流分词器对印度25种语言的“代币税”，发现印度语言所需token量最高可达英语的13.7倍（GPT-4o分词器），导致印度语言用户在商业LLM使用中面临更高的费用、更慢的速度和更小的有效上下文，而Sarvam分词器是唯一将差异控制在2.6倍以内的第三方选择。 |
| [^193] | [Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization](https://arxiv.org/abs/2611.13057) | 提出一种连续时间集合变换器，将异步采样的WiFi与IMU观测表示为带时间戳的无序标记集合，用单一注意力块实现跨模态与时间的联合融合，并通过随机掩码训练应对缺失或过时的传感器输入，用于室内定位。 |
| [^194] | [Beyond the Document: Compiled Context and Governed Agents for Any Work Product](https://arxiv.org/abs/2611.13055) | 该论文提出便携式智能体格式（PAF）包这一新交付单元，将工作成果连同编译上下文、证据、来源与交互规则一起封装，实验证明受治理的智能体阅读器在回答相关问题上的成功率（96%）显著优于传统的指令加原始文件检索方式（86%）。 |
| [^195] | [Trust, but verify, audit and attribute: a human-in-the-loop framework for checking AI-produced research results](https://arxiv.org/abs/2611.13042) | 该论文提出了一个领域无关的人在回路框架，通过经过验证的AI工具检验、忠实性审计以及人类与AI贡献归因记录，来保障对AI批量产出的研究成果的有效核查。 |
| [^196] | [Code-Grounded Agentic Explanations for Static Analysis Vulnerability Findings](https://arxiv.org/abs/2611.13041) | 该论文提出将漏洞检测职责保留给静态分析器、大语言模型智能体仅负责生成解释，并据此重新设计了开源IDE插件SAFE，要求智能体在回答前必须查阅代码仓库和触发的规则，从而在保留可审计性的同时生成有代码依据的解释。 |
| [^197] | [APS-VLA: Action-Guided and Future-Aware Token Pruning for Efficient Vision-Language-Action Models](https://arxiv.org/abs/2611.13030) | APS-VLA 在冻结预训练 VLA 骨干网络的前提下，通过一个结合动作监督与未来感知视觉转移目标的轻量级决策影响估计器，从机器人演示中学习视觉令牌剪枝，在显著降低推理延迟的同时保证可靠的动作生成。 |
| [^198] | [GTAGym: Real AAA Game Tasks Environment for Evaluation and Reinforcement Learning](https://arxiv.org/abs/2611.13028) | 本文提出GTAGym——一个基于《侠盗猎车手V》真实AAA游戏任务的评估与强化学习环境，其基准GTAVBench包含103个跨九类任务家族的在线主线任务及47个图像式射击狙击任务，支持截图与键鼠操作的并行测试并自动校验任务结果，用于评测前沿大语言模型的智能体能力。 |
| [^199] | [Repository-Context-Guided Image Understanding for Agentic Software Development](https://arxiv.org/abs/2611.13026) | 提出通过从同一代码库渲染清晰的参考截图来增强编程智能体，从而在查询截图质量退化、视觉证据不清晰时，仍能准确识别UI元素并定位相关源文件。 |
| [^200] | [GRAP: Gender Rubric-Aware Probing for Analyzing Gender Bias in LLMs](https://arxiv.org/abs/2611.13024) | 提出了GRAP框架，利用轻量级逻辑回归探针从10个量表维度中自动识别最具区分性的性别偏见维度，以任务无关的方式分析大语言模型开放式回复中的性别偏见。 |
| [^201] | [Understanding-First AI Engineering: Purposes, Reference Architecture, and Architectural Patterns for Human Understanding](https://arxiv.org/abs/2611.13022) | 该论文提出“理解优先的AI工程”新范式，将特定目的的人类理解作为一等设计原则，归纳了反思性主体、有效行动、集体治理和认知价值四种理解目的，并给出贯穿生产、验证、使用与学习的参考架构及五种候选架构模式。 |
| [^202] | [ChipOmni: A General-Purpose AI Agent for Electronic Design Automation](https://arxiv.org/abs/2611.13020) | ChipOmni是一个面向电子设计自动化的知识增强AI智能体框架，通过融合多模态理解、动态规划、多智能体协作与验证驱动的工具交互，实现了EDA设计中自主规划、验证与修复的统一自动化系统。 |
| [^203] | [Multi-Future Drafting: Mixture Block Drafters for Tree-Verified Speculative Decoding](https://arxiv.org/abs/2611.13017) | 论文揭示了基于乘积排序的草拟树在多未来场景下会将大量概率预算浪费在目标模型从不产生的“嵌合体”前缀上（占64节点树的27-33%），并提出多未来草拟（MFD），用潜变量（K个可学习模式嵌入）替代链式法则来恢复多个未来之间的耦合，从而提升树验证式推测解码的接受效率。 |
| [^204] | [From Questions to Decisions: Benchmarking How LLM Agents Acquire Decision-Relevant Information from Human Beings](https://arxiv.org/abs/2611.13015) | 提出Ask2Decide基准，通过43个由真人操作员和优化求解器支持的规划任务，评估LLM智能体能否主动向人类询问对决策至关重要的缺失信息，并区分有用信息与无关细节。 |
| [^205] | [Compile-Gated Lean Proof Refactoring with Enterprise Agent Practices: A Model per Step under a US\$3 Cap, Followed by Programmatic Rewrite Rules](https://arxiv.org/abs/2611.13011) | 该论文提出一种“每步选用性价比最优模型、单题成本严格控制在3美元上限内”的企业级智能体实践，并将开发中积累的经验固化为经编译器校验的程序化重写规则，最终以86.31分（总花费166.33美元）在 Lean 证明重构竞赛中获得第8名。 |
| [^206] | [An Open Implementation Flow for the Xilinx Spartan-3E and Spartan-6 that Runs in the Browser, Checked Byte for Byte Against ISE](https://arxiv.org/abs/2611.13008) | 该论文提出了一个可在浏览器中运行的开源 FPGA 实现流程，为仅被停产的 ISE 工具链支持的 Xilinx Spartan-3E 和 Spartan-6 器件生成比特流，其输出与 ISE 生成的比特流逐字节完全一致。 |
| [^207] | [Just AI Governance and the Forbidden-Boundary Principle](https://arxiv.org/abs/2611.13007) | 本文提出了一个涵盖规范、数据、模型与治理四层的公正AI评估框架，将六项正义要求映射到证据、决策权与救济上，并提出两阶段评估规则以及“禁越边界原则”——即AI代理不得在预先授权的边界之外执行具有外部影响的行为。 |
| [^208] | [Seeing Is Not Believing: Counterfactual Lesion Erasure Teaches a Small Medical Vision-Language Agent to Report Only What the Image Shows](https://arxiv.org/abs/2611.13006) | 该论文提出无需人工标注的反事实病灶擦除评测范式与证据敏感度差距（ESG）指标，并据此训练出40亿参数的医学视觉-语言智能体MedScope-4B，使其只报告图像中真实可见的发现，将“幻觉报告被擦除病灶”的比例从前沿模型的41–62%降至9.6%。 |
| [^209] | [The Shape of a Program: Path Signatures for Trace-to-Program Induction](https://arxiv.org/abs/2611.13005) | 本文将粗路径理论引入程序归纳，用路径签名变换编码执行轨迹，在 Karel、List 和 RASP 三个领域几乎全面超越 LSTM、Transformer 和傅里叶基线（尤其在数据稀缺时优势最大），并利用流程天然的收缩性构建无需外部预言机的自举方法来揭示监督数据的冗余。 |
| [^210] | [PhysMoE: Fast Spatial Grounding with Single-Token Coordinate Representation in Modality-Decoupled VLMs](https://arxiv.org/abs/2611.13000) | 提出PhysMoE框架，通过物理模态解耦的专家路由、复用视觉位置嵌入的单Token坐标表示以及可微SDF监督，实现了具身VLM中快速且精确的空间定位。 |
| [^211] | [TIB: Sample-wise Tempered Information Bottleneck for Multimodal Attribution beyond Alignment Assumption](https://arxiv.org/abs/2611.12997) | 该论文提出TIB方法，通过样本级温和信息瓶颈突破现有多模态归因方法对模态对齐假设的依赖，解决了在语义错位情况下代理监督信号可靠性不均导致的后验过度收缩与强制解释问题。 |
| [^212] | [ZeroPoint: Lightweight Energy-Based Decision Model](https://arxiv.org/abs/2611.12992) | 本文提出ZeroPoint，通过引入一个对任意输入能量恒为零的“空答案”固定状态，为基于能量的决策模型提供显式的能量参考水平，使模型能够判断并拒绝“没有任何可接受选项”的情形。 |
| [^213] | [Specializing Meta Multi-Objectivization as Baseline for Software Engineering Optimization](https://arxiv.org/abs/2611.12991) | 本文将元多目标化（MMO）从仅适用于配置调优的模型专门化为软件工程优化的通用基线，通过理论分析和九个软件工程优化问题的实证研究，阐明了辅助目标选择的关键性并验证了MMO克服局部最优的有效性。 |
| [^214] | [Third Eye: Beyond Immediate Gains in Recursive Self-Improvement](https://arxiv.org/abs/2611.12988) | 论文通过大规模受控实验发现，递归自我改进中看似有用的自我更新在经过第二轮更新后效用会进一步降低（甚至候选平均即时效用已为负值），因此对自我改进的评估必须超越第一步的即时收益。 |
| [^215] | [ICLayout-Bench: Benchmarking LLM Agents for Custom Circuit Layout Design](https://arxiv.org/abs/2611.12982) | 该论文提出了ICLayout-Bench基准测试，首次系统性评估LLM智能体使用EDA工具从晶体管级网表生成高质量集成电路版图的能力，涵盖165个任务、13个PDK以及3nm至180nm的工艺节点。 |
| [^216] | [MADSNet: A Multi-Scale Axial-Attentive Dense Spiking Network for Spike Camera Image Reconstruction](https://arxiv.org/abs/2611.12980) | 提出了一种多尺度轴向注意力密集脉冲网络MADSNet，通过分层脉冲编码器-解码器结构实现高质量脉冲相机图像重建，在SRED数据集上显著超越现有SNN重建方法。 |
| [^217] | [Trajectory-Regularized Distribution Distillation for One-Step Generation](https://arxiv.org/abs/2611.12978) | 提出轨迹正则化分布蒸馏（TRD），通过单次前向传播预测的动量混合策略统一分布级与轨迹级监督，无需额外可训练网络即可实现高质量的一步式生成。 |
| [^218] | [TrustMod-SM: A Multi-Axis Benchmark for Evaluating Trustworthiness of LLMs in Social Media Content Moderation](https://arxiv.org/abs/2611.12970) | 提出了TrustMod-SM多维度基准，从可信性、公平性、安全性、鲁棒性和上下文完整性五个维度评估LLM在社交媒体内容审核中的可信度，揭示了更高的准确率往往伴随更大的群体偏见，且扩大模型规模并不能统一解决可信度问题。 |
| [^219] | [BuySideFlow: A Point-in-Time Benchmark for LLM Agents in Investment Research](https://arxiv.org/abs/2611.12966) | 该论文提出了首个面向投资研究中LLM智能体的时点有效性基准BuySideFlow，包含408个从业者来源的中文金融数据任务，评估显示智能体准确率仅16.2%至36.8%，且在通过结果检查的运行中仍有50.8%存在时点违规，揭示了仅凭结果验证不足以确保金融数据分析的正确性。 |
| [^220] | [RBLA+: Efficient Rank-Specific LoRA Adapter Extraction for Federated Fine-Tuning with Heterogeneous Clients](https://arxiv.org/abs/2611.12961) | 提出RBLA+方法，为客户端LoRA秩异构的联邦微调实现高效的特定秩适配器提取，在低计算开销的同时控制重建误差。 |
| [^221] | [IIT Patna at "ICPR 2026 TVRID Competition on Privacy-Preserving Person Re-Identification from Top-View RGB-Depth Camera"](https://arxiv.org/abs/2611.12960) | 该论文介绍了IIT Patna团队在ICPR 2026 TVRID隐私保护俯视RGB-Depth行人重识别竞赛中的方法，其中最优方法在深度赛道上以93.51% mAP和90.42% CMC@1获得第4名，团队总体排名第5和第7。 |
| [^222] | [Monitoring One Channel Is Not Communication Control: Causal Evidence for Adaptive Reallocation of Joint Message--Route Codes](https://arxiv.org/abs/2611.12959) | 本研究通过端到端游戏的因果干预实验证明，监控单一通信通道会促使学习智能体将联合消息-路由编码自适应地重新分配到未受监控的路由上，而不是消除通信本身，且路由具有语义功能。 |
| [^223] | [Embodied Science Agents: Making Embodied Experimental Design Learnable](https://arxiv.org/abs/2611.12958) | 该论文指出当前自动驾驶实验室存在“语言能力增强但物理具身根基被切断”的根本悖论，并提出应沿认知、实验、具身三个相互依存的可学习性维度构建具身科学智能体，使实验设计本身成为可学习的能力。 |
| [^224] | [ADIndex: Layer-Adaptive Sparse Attention Indexing via Group-Level Distillation](https://arxiv.org/abs/2611.12956) | ADIndex通过逐层校准与组级蒸馏训练层级自适应的稀疏注意力索引器，并辅以低延迟推理策略，大幅降低长上下文推理中索引环节的延迟开销。 |
| [^225] | [Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series](https://arxiv.org/abs/2611.12955) | ConceptTime 利用冻结概率预测器产生的“预测性惊奇”这一单一自监督信号，同时完成时间序列分段定位与概念表征，打破了边界与模式相互依赖的循环，从而在时间序列分类中同时实现准确性、可解释性和证据不足时的弃权能力。 |
| [^226] | [Physics2Weight: Direct Neural Network Architecture and Weight Generation from Physical Laws Using Large Language Models](https://arxiv.org/abs/2611.12954) | 该论文提出Physics2Weight（PtW）框架，利用大语言模型根据控制方程及初始边界条件直接生成神经网络权重并编辑架构，通过物理约束提示、物理引导的架构编辑和迭代权重细化，在无需梯度优化的情况下实现更高精度和物理一致性的偏微分方程求解。 |
| [^227] | [Agent Architecture Cost-Success Trade-offs: A Controlled Evaluation across Enterprise-Style SOPs](https://arxiv.org/abs/2611.12953) | 该论文通过固定模型、任务与工具、仅改变架构的受控实验，在13个企业SOP上比较四种智能体架构，发现不存在普遍最优架构，架构选择应取决于具体SOP的复杂程度。 |
| [^228] | [SetPrint: Black-Box LLM Fingerprinting via Setwise Correspondence of Response Changes](https://arxiv.org/abs/2611.12950) | SetPrint提出了一种无需训练的黑盒LLM指纹识别方法，首次同时利用集合内部响应变化关系和两个模型对相同扰动的响应变化对应关系，来判定目标模型是否源自源模型。 |
| [^229] | [When Proofs Are Abundant, What Is a Paper For?](https://arxiv.org/abs/2611.12943) | 在AI使形式化验证证明近乎零成本的时代，论文的核心价值在于“推导层内涵”（作者的问题选择与视角）与“方向性外延”（通过切点分解揭示证明薄弱环节），而非正确性本身。 |
| [^230] | [Tabular Foundation Models vs Tuned Gradient Boosting for Telecom Churn: A Profit- and Calibration-Aware Evaluation](https://arxiv.org/abs/2611.12942) | 该研究以期望最大利润和概率校准为评判标准，将表格基础模型与经Optuna调优的梯度提升树在电信客户流失预测上进行严格重新评估，发现TabPFN-3.5在部分数据集上能显著带来更高利润，修正了以往仅用排序指标、与未调优基线比较所得出的结论。 |
| [^231] | [AI Governance Should Prioritize Control and Knowledge Boundaries Over Limiting Intelligence](https://arxiv.org/abs/2611.12941) | 本文提出“能力回路”治理框架，将AI的现实影响分解为智能发现、程序性技能与直接控制三条路径，并通过形式化分析与随机模拟论证AI治理应优先设置控制与知识边界，而非笼统限制智能。 |
| [^232] | [Reinforcement Learning based control of Cancer Chemotherapy treatments](https://arxiv.org/abs/2611.12940) | 该论文系统梳理了强化学习技术在癌症化疗治疗控制中的应用现状，重点研究了用于生成最优动态治疗方案的最新强化学习方法。 |
| [^233] | [Graph Visualizations in Prompts for Knowledge Graph Question Answering](https://arxiv.org/abs/2611.12938) | 该研究发现向提示中添加图可视化可使知识图谱问答平均提升1.85个F1点，且收益对图布局方式高度敏感，其中将大型子图分解为分离路径的布局带来的改进最大。 |
| [^234] | [Tracing Psychometric Inference in Large Language Models](https://arxiv.org/abs/2611.12937) | 该研究通过跨角色预测范式发现，大语言模型（尤其是较大规模的指令微调模型）在行为上重现并放大了人类心理量表的跨量表相关性，且其内部几何结构与人类心理测量结构对齐，表明模型在一定程度上确实内部表征了心理构念。 |
| [^235] | [LandslideFM: A Foundation Model Towards Unified Landslide Perception and Understanding](https://arxiv.org/abs/2611.12936) | 提出了首个统一的滑坡基础模型LandslideFM，通过半监督学习框架利用有限标注数据完成多种视觉感知任务，并结合参数高效微调的视觉-语言模型实现滑坡的情境理解与推理，同时发布了首个智能滑坡分析基准数据集LandslideVL。 |
| [^236] | [Train on prefill, monitor at decode: Using sparse autoencoders for topic monitoring](https://arxiv.org/abs/2611.12935) | 提出了TopicScope框架，利用稀疏自编码器（SAE）特征训练因果主题分类器，在预填充阶段提取特征训练后可泛化到解码阶段，实现对生成过程中每个token的实时多主题监控且标签可中途变化。 |
| [^237] | [EarthRecord: From Pixels to Land Change with Multimodal Agents over Four Decades of Satellite Imagery](https://arxiv.org/abs/2611.12934) | 提出EarthRecord基准，基于56个站点四十年的原始多传感器卫星影像，要求多模态智能体自主完成云掩膜和跨传感器协调，通过238个土地变化问题及对照实验评估其真实土地变化分析能力。 |
| [^238] | [Auditable Human-Agent Teams for Autonomous X-ray Experimentation](https://arxiv.org/abs/2611.12933) | 该研究利用可审计的人-智能体协作团队在真实同步辐射装置上实现了纳米粒子超晶格的自主X射线实验，并通过完整对话记录审计决策溯源与认知分工，证明智能体系统能够在物理仪器上持续运行并逐步承担更多科学贡献。 |
| [^239] | [Token Importance-Aware Semantic Communication Enabled Robust Video Transmission](https://arxiv.org/abs/2611.12928) | 该论文提出令牌重要性感知的语义通信框架Ti-SemCom，通过联合提取目标与运动信息、按重要性分层关联语义令牌、结合带宽动态选择传输令牌并在接收端复用参考内容，在保证视频传输鲁棒性的同时将通信开销降低88.3%。 |
| [^240] | [M\"obius Interaction Filtering: Order-Selective Spectral Regularization for Multimodal Humor Detection](https://arxiv.org/abs/2611.12913) | 提出莫比乌斯交互过滤方法，通过阶数选择性谱正则化在保留高阶模态交互表示的同时调控其对预测的贡献，使多模态幽默检测的标签对齐信噪比提升1.62-2.79倍。 |
| [^241] | [Brasilidade: Measuring Brazilian Identity for Large Language Models](https://arxiv.org/abs/2611.12906) | 该论文提出了Brasilidade基准，通过涵盖语言、语法等四个领域的标准化任务评估大语言模型的巴西葡萄牙语能力，发现模型规模并不决定性能，且葡萄牙语专用模型的表现也未超过平均水平。 |
| [^242] | [Actionable Feedback: Improving Model Capabilities from User Edit Data on Clinical Documentation](https://arxiv.org/abs/2611.12904) | 该论文针对临床文档场景中用户编辑数据并非全部可操作的问题（如45%以上编辑包含模型输入不支持的信息），构建了临床医生编辑行为分类体系并开发了区分可操作反馈的方法，以避免微调导致模型产生幻觉。 |
| [^243] | [SELD: Detecting Silent Evidence Loss in Context Compression beyond the Answer](https://arxiv.org/abs/2611.12903) | 提出SELD——一种无需训练和标注的方法，通过比较阅读模型压缩前后对未说出口候选答案的偏好变化，检测上下文压缩中“答案不变但关键证据被删”的隐性证据丢失问题，并构建了SEL-Pairs评测基准。 |
| [^244] | [Rethinking in Spikes: Mitigating Hallucinations in MDLMs with Step-Aware Decoding](https://arxiv.org/abs/2611.12901) | 提出了StepRefine，一种即插即用的步感知解码策略，通过检测去噪过程中的熵尖峰并在离散去噪与潜在精炼之间动态切换，从而缓解多模态扩散语言模型的幻觉问题。 |
| [^245] | [AFNO: Universal Certifiably-Arbitrage-Free Operator Learning Framework for European Call-Price Surfaces](https://arxiv.org/abs/2611.12899) | 该论文提出了AFNO——一个可证明无套利且通用的神经算子学习框架，通过连续求解近似最近邻问题将输入映射为无套利欧式看涨期权价格曲面的凸组合，解决了标准神经算子在期权定价中无法保证无套利并可能导致负盈亏的问题。 |
| [^246] | [Direct Trajectory Optimization for Inference-Time Alignment of Text-to-Image Diffusion Models](https://arxiv.org/abs/2611.11821) | 提出无需训练的 DTO 框架，通过将中间去噪分数和采样噪声作为优化变量直接优化扩散模型的采样轨迹，实现文本到图像模型的推理时奖励对齐，避免了通过去噪网络的高成本反向传播，并提供奖励提升的理论保证。 |
| [^247] | [FinCutBench: A Continually Updated Benchmark for Measuring the Temporal Boundaries of LLM Financial Knowledge](https://arxiv.org/abs/2611.11588) | 该论文提出FinCutBench，一个每月持续更新的中英双语金融知识基准，通过按月构建知识画像并定位“FinCut边界”，来衡量大语言模型金融知识的时间截止点，以检测金融回测中的参数化前视偏差。 |
| [^248] | [When Should Diffusion Language Models Reconsider Their Commitments?](https://arxiv.org/abs/2611.11551) | 该论文提出PURL框架，通过联合学习编辑选择、取消掩码选择和词元预测三个决策，使掩码扩散模型能够在并行快速生成的同时有选择地修正早期不一致的词元承诺，且推理时三个决策仅需一次共享前向传播即可完成。 |
| [^249] | [ScrapeBench: Evaluating Legal Compliance of AI Agents in Website Scraping](https://arxiv.org/abs/2611.11306) | 该论文提出首个评估AI智能体网站抓取法律合规性的基准ScrapeBench，发现大多数智能体在抓取明确禁止抓取的网站时合规率低于5%，即使主动检查了ToS和robots.txt也依然如此，而在允许抓取的网站上却有时会错误地拒绝抓取。 |
| [^250] | [STABLE: A Continual Learning Optimizer with Adaptive Drift Control](https://arxiv.org/abs/2611.10655) | 该论文提出STABLE，一种具有自适应漂移控制的原对偶优化器，理论上证明有界参数漂移可保证有界的间隔侵蚀，并通过动态调节对偶变量来强制执行漂移预算，从而从优化器层面缓解持续学习中的灾难性遗忘。 |
| [^251] | [Agentic discrete-event simulation for manufacturing system design: an open 3D simulator, a benchmark and an industrial case](https://arxiv.org/abs/2611.10602) | 本文提出了SimuLean——一个基于Unity的开源3D离散事件仿真器，专为AI智能体编程设计，通过一次性校验的建模语义和领域级命令，使前沿AI智能体能以86%的成功率在约两分钟内自动构建有效的制造系统仿真模型。 |
| [^252] | [Seeing Isn't Enough: Unlocking Visual Reasoning by Bridging the Reasoning-Access Gap with TIGER](https://arxiv.org/abs/2611.10552) | 该论文揭示了多模态大语言模型中存在的“模态推理访问差距”——模型虽能准确感知图像内容却无法将其路由进已有的文本推理机制——并提出TIGER训练方法，通过挖掘模型在文本上成功但在对应对称图像上失败的模态反事实样本来定向弥合这一差距，从而解锁视觉推理能力。 |
| [^253] | [Beyond the Mean Gap: What Transferred-Angle QAOA Retains on a Superconducting Processor](https://arxiv.org/abs/2611.10529) | 该研究发现在超导处理器上运行深度为2的转移角度QAOA时，12量子比特电路虽保留了29%–46%的平均间隙改进，但最优解概率仅保留5%–15%，表明仅靠平均成本对比验证硬件QAOA性能会掩盖对优化最关键的最优解概率的大幅损失。 |
| [^254] | [Split and Bridge: Multimodal Generation via Diffusion Bridging](https://arxiv.org/abs/2611.10441) | 提出了一种自下而上的多模态生成框架 SnB，通过在采样时利用桥接模块连接预训练的单模态扩散模型，无需联合训练即可生成连贯的多模态样本，实现了预训练模型的即插即用复用。 |
| [^255] | [FAIDER: Fairness-Aware Item-Level Decoding for Large Language Model Recommenders](https://arxiv.org/abs/2611.10285) | FAIDER针对无法访问模型权重和词表logits的大语言模型推荐系统，提出条目级解码方法，通过保形启发的校准确定跨组语义差异阈值，并在解码中结合硬过滤或对偶启发的软惩罚来控制公平性违规率。 |
| [^256] | [Key Institutional Considerations for Designing Effective Independent Verification Organizations](https://arxiv.org/abs/2611.10168) | 本文提出设计有效独立验证机构（IVO）所需的关键制度考量，主张IVO生态系统应超越灾难性风险评估，扩展至AI生命周期的系统与应用层面，为前沿企业和企业级公司提供情境化的AI模型、系统及应用评估。 |
| [^257] | [Veterans' Mental Health Crisis Severity Annotation: Do LLMs Agree More Than Human Experts?](https://arxiv.org/abs/2611.10048) | 本研究对比了大语言模型与八位退伍军人专家对社交媒体帖子危机严重程度的标注，发现选定的LLM之间的一致性显著更高（序数α达0.8336），而人类专家一致性较低（α=0.494）且分歧多发生在相邻严重程度级别之间。 |
| [^258] | [CulturalCrosswords: A Multilingual Benchmark of Answer Constraints and Grid Completion](https://arxiv.org/abs/2611.10044) | 该论文构建了覆盖42种语言、53个填字游戏的多语言基准CulturalCrosswords，发现长度提示和网格约束等答案约束能显著提升大型语言模型解答文化谜题时答案的准确性与一致性。 |
| [^259] | [Knowing When to Intervene: Causal Evaluation of AI Agents](https://arxiv.org/abs/2611.09792) | 该论文提出了首个基于因果推断的方法学框架，用于评估在AI智能体执行轨迹中进行干预（如人工审查或行为引导）的因果价值。 |
| [^260] | [BEiTScore++: Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression](https://arxiv.org/abs/2611.09505) | 提出BEiTScore++，通过共形化分位数回归为无参考图像描述评估指标引入共形校准的置信区间，在单次前向传播中同时实现准确评分与可靠的不确定性量化。 |
| [^261] | [Circe: Training-Free LLM Self-Evaluation via Component Interaction Consistency](https://arxiv.org/abs/2611.07948) | 提出Circe——一种免训练的大语言模型自我评估方法，通过分析注意力与前馈网络等组件间的交互一致性并结合内外部信号来评估回答正确性，无需参考答案或重复采样。 |
| [^262] | [Towards Scalable Diagnostic-Driven Whole-Slide Image Retrieval for Pathology Archives](https://arxiv.org/abs/2611.07935) | 提出了一种可扩展、诊断驱动的全切片图像检索框架CLEAR-WSI，通过固定长度的切片嵌入并利用诊断标签细化检索候选，摆脱了手工索引流水线，在公开和临床病理数据上实现了高效的相似病例检索。 |
| [^263] | [Selective Retrieval Routing for Multi-Violation Knowledge Graph Repair](https://arxiv.org/abs/2611.07342) | 该论文提出一种选择性检索路由框架，由LLM修复智能体编辑知识图谱、路由器智能控制对外部证据检索的访问，并利用违规注入生成的修复结果训练特定部署的路由器，从而提升多重违规知识图谱的修复效果。 |
| [^264] | [CompressionBench: How can your coding agents compress LLMs?](https://arxiv.org/abs/2611.06858) | 该论文提出CompressionBench基准，将大语言模型压缩形式化为一项终端任务，系统评估了编程智能体自主压缩LLM的能力，发现大多数智能体提交能通过独立验证，且压缩后模型性能接近原始BF16模型。 |
| [^265] | [ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus](https://arxiv.org/abs/2611.06706) | 该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。 |
| [^266] | [Do Language Models Know Which Constraints Belong Together?](https://arxiv.org/abs/2611.05726) | 该论文证明，在 Dantzig-Wolfe 重构中寻找只有组合起来才能改进边界的约束族隐藏组合需要 k^w 次评估，而一个遗漏 e 个成员的分组预测可将成本降至 Θ(k^e)，因此预测节省的评估次数直接度量了预测器（如语言模型）对约束分组知识的掌握程度。 |
| [^267] | [Combining Relevant Expert Knowledge Using Mixtures of Feedback Fuzzy Cognitive Maps](https://arxiv.org/abs/2611.05664) | 该论文将法律证据理论中的相关性定义扩展应用于神经符号系统，通过将专家的反馈模糊认知图混合为共同知识网络，并借助新的类贝叶斯定理对混合知识进行解混合，从而仅组合相关专家答案以提升系统的准确性和可扩展性。 |
| [^268] | [Kierston & KierstonBench: Simulation of Allentown, PA and Evaluation of Language Models in Crime Investigation and Surveillance Tasks](https://arxiv.org/abs/2611.04785) | 该论文提出了城市级模拟引擎 Kierston 与配套基准 KierstonBench，通过模拟宾州阿伦敦市约12.5万名居民的日常活动及多源记录，首次实现了对大语言模型在犯罪调查与监控任务中能力的可量化评估。 |
| [^269] | [FORGE: Fault-Oriented Rework Code Generation for Executable Interlocking Assembly](https://arxiv.org/abs/2611.02536) | 该论文提出了FORGE-Bench基准和FORGE框架，用于评估并解决大语言模型在联锁装配故障条件下生成可执行返工程序的挑战。 |
| [^270] | [MAGAR: Multimodal Agents for GUI-Based 3D Mechanical Assembly and Minimal Rework](https://arxiv.org/abs/2611.02472) | 提出MAGAR-Bench基准（涵盖26个装配族的4,000个多模态样本，将装配决策、原子操作与最小返工分离）以及基于冻结策略的选择性纠正框架MAGAR，通过匹配纠正粒度与输出依赖关系，实现基于GUI的3D机械装配的最小返工。 |
| [^271] | [Spectral-Convolutional Factorized Transformer for Joint Multipath Parameter Estimation: Applications to mmWave Positioning and Radio SLAM](https://arxiv.org/abs/2611.01491) | 提出了一种谱-卷积分解Transformer（SCFT），能够从含噪信道估计中联合估计时延、到达角和离开角等多径参数，为毫米波定位和无线电SLAM等下游任务提供关键几何信息。 |
| [^272] | [Low-Margin Branch Points in Deterministic LLM Deliberation](https://arxiv.org/abs/2611.00546) | 该论文通过白盒审计发现，在确定性多智能体LLM协商中，低裕度词元位置显著更易引发对话分支，且降低裕度的干预会大幅增加分支发生频率，从而揭示了多智能体LLM系统对内部局部扰动的脆弱性。 |
| [^273] | [Textual Response-Aware Guidance for Zero-Shot Adversarial Robustness of Vision-Language Models](https://arxiv.org/abs/2610.40309) | 提出文本响应感知引导方法TRAG，利用预训练文本编码器对输入嵌入局部变化的响应来约束视觉表示漂移，从而在不损害零样本性能的前提下提升视觉-语言模型的对抗鲁棒性。 |
| [^274] | [Learning to Reason from Reference Samples Alone: Reinforced Adversarial Learning for Modern LLM Post-Training](https://arxiv.org/abs/2610.38851) | 该论文提出强化对抗学习（RAL）方法，通过共同训练的大语言模型评判者，仅凭参考答案即可作为推理型大语言模型在策略后训练的唯一奖励信号，无需显式可验证奖励或参考推理轨迹，且实验表明其性能可媲美直接的RLVR。 |
| [^275] | [SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2610.36991) | SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。 |
| [^276] | [Learning Where to Trust: Reliability-Guided Expert Routing and Velocity Gating for 3D Deformable Registration](https://arxiv.org/abs/2610.36672) | 提出ReliMorph框架，利用局部相关分布评估匹配可靠性，并以可靠性线索引导空间专家融合和残差速度门控，从而提升三维可变形图像配准在弱纹理、严重形变区域的准确性。 |
| [^277] | [Spiking Neural Networks Are Harder to Steal: A Benchmark Study of Model Stealing Attacks](https://arxiv.org/abs/2610.36298) | 本研究通过基准测试四种代表性的模型窃取攻击，证明由于脉冲神经网络采用基于稀疏离散脉冲和时变膜电位的时间计算模式，其决策行为比传统人工神经网络更难通过黑盒查询被窃取重建。 |
| [^278] | [Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss](https://arxiv.org/abs/2610.36202) | 本文提出人格-头部似然评估（PHLE）方法，利用基础预训练模型对输出的似然评估来解码并分类后训练模型中的潜意识隐蔽信号，从而检测出提示分类器无法发现的隐藏意图（如隐蔽的伤害用户倾向）。 |
| [^279] | [Scoped Semantic Invariance as a Foundation for LLM Reliability](https://arxiv.org/abs/2610.33267) | 该论文提出“范围化语义不变性”框架，以知识图谱作为显式语义控制结构，揭示LLM的不一致性是结构化且可定位的（集中于跨模型跨领域稳定的TREE、STAR、CYCLE热点子图模式），并证明选择性的图谱接地可提升模型正确性与一致性。 |
| [^280] | [TACS-RF: Tool-driven Agentic Circuit Synthesis of RF and mmWave Modules](https://arxiv.org/abs/2610.31882) | 该论文提出了TACS-RF框架，首次让LLM编程智能体直接驱动Cadence、Ansys等工业级EDA工具，半自主完成了90nm SiGe BiCMOS工艺下240-251GHz亚太赫兹发射机从振荡器、功率驱动、倍频器到天线的完整芯片设计，验证了LLM智能体驱动真实射频芯片设计流程的可行性。 |
| [^281] | [The Value of Memory in Language Agents](https://arxiv.org/abs/2610.31303) | 该论文提出一个决策论框架，以所避免的任务损失为单位，量化语言智能体中记忆从原始历史到持久状态、操作使用及代价高昂的恢复与修正全过程中的价值，并通过成本感知的模拟条件和五个最小反例揭示信息时机、检索访问、组件交互、反馈和需求不确定性对记忆价值的关键影响。 |
| [^282] | [SplatLab: Autonomous Scientific Discovery for 3DGS Research](https://arxiv.org/abs/2610.30468) | SplatLab是一个面向3DGS领域的自主科学发现系统，它将文献理解、假设生成、方法实现、实验验证与证伪、以及论文写作整合为统一的研究流程，能够从少量甚至零输入出发自主完成从研究机会识别到论文产出的完整科研闭环。 |
| [^283] | [IECG: Weakly-Supervised Interpretability-Enhancing Foundation Model for Electrocardiogram](https://arxiv.org/abs/2610.28083) | IECG是一种弱监督心电图基础模型，通过将既有临床知识转化为注意力监督伪标签，在无需人工异常标注的情况下同时实现诊断分类与异常定位，显著提升了心电图AI模型的可解释性。 |
| [^284] | [Hearsay or Fact? How Framing and Elicitation Shape LLM Probability Updates](https://arxiv.org/abs/2610.27745) | 该论文提出了一种衡量大语言模型对单条新信息概率更新幅度的“方向性移动”指标，并发现信息的框架呈现方式（直接告知还是传闻转述）与概率引出方式（单个概率还是五个加权估计）会显著影响模型的概率更新。 |
| [^285] | [MoveBench: A Rubric-Based Benchmark for Evaluating Large Language Models in Sports and Health Coaching](https://arxiv.org/abs/2610.26185) | MoveBench是首个专门针对运动与健康指导领域设计的大语言模型评估基准，通过整合多选题与开放式问答两类异构任务，从运动安全、生理适宜性、个性化和可操作指导质量等维度填补了通用评估基准的空白。 |
| [^286] | [MSCR: Jointly Balancing Utilization Across Modalities and Discovering Synergistic Information](https://arxiv.org/abs/2610.24395) | 提出MSCR融合方法，通过反主导目标抑制单模态过度影响并促进协同信息的发现，从而联合实现模态利用的平衡与多模态优势的充分挖掘。 |
| [^287] | [Scaling Vision Transformers Without Training: Prune-and-Merge Token Compression](https://arxiv.org/abs/2610.18039) | 该论文提出了一种无需训练的Prune-and-Merge ViT框架，通过结合基于类别注意力的剪枝与基于相似度的Token合并来压缩视觉Transformer，在保持鲁棒性、预测稳定性和不引入幻觉行为的前提下，实现了计算成本的有效降低和准确率的提升。 |
| [^288] | [Where Polanyi Meets Galois: Tacit Knowing as an Algebro-Deterministic Operation, Toward Re-rooting AI-Generated Science](https://arxiv.org/abs/2610.16102) | 该论文将波兰尼的默会认识建模为GF(2)上代数确定性异或-移位超维记忆（VaCoAl）的一种运行状态，其“救援模式”与“不关心模式”之间的审计-估值互补性为默会知识与显性知识的连续统提供了形式化对应，从而为AI生成的科学重新扎根。 |
| [^289] | [GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning](https://arxiv.org/abs/2610.10201) | 该论文提出GAGR-Lab框架，通过笛卡尔游戏场景与Rust轨迹执行评估模型将空间配置转化为满足几何约束的解析函数的联合推理能力，试点实验表明当前视觉语言模型在此任务上尚无法命中目标。 |
| [^290] | [CACHEFORGE: LLM-Guided End-to-End Generative Cache Replacement Policy for Performance and Hardware Efficiency](https://arxiv.org/abs/2610.07668) | CACHEFORGE 首次将大语言模型嵌入受控的硬件感知循环中，端到端地自动演化生成缓存替换策略，突破了传统启发式和模仿学习方法性能停滞与过拟合的瓶颈。 |
| [^291] | [Bridging the Evidence-to-Execution Gap:A Reflective Agent for Multi-Objective Peptide Design](https://arxiv.org/abs/2610.06190) | 论文提出EASER反思型智能体，通过由性质特异性低秩方向构成的性质接口，让LLM控制器基于文献证据、序列上下文和先验结果组合干预假设以引导冻结的扩散生成器，从而弥合科学推理与序列级控制之间的“证据到执行”鸿沟，实现多目标肽设计。 |
| [^292] | [Decide, Ask, or Defer: Clinical LLMs under Incomplete Evidence](https://arxiv.org/abs/2610.04542) | 该论文提出决策/询问/推迟三分类框架及盲测协议，发现临床大语言模型在相同证据下行动选择差异巨大，定向信息获取更能提升自主决策率，并揭示出安全性与自主性之间的权衡。 |
| [^293] | [Mitigating Private Data Leakage in LLMs with Whiteout](https://arxiv.org/abs/2610.02418) | 本文提出Whiteout工具，通过用精心设计的混淆样本覆盖个人敏感信息，来防止大语言模型复现泄露个人隐私，相比机器遗忘方法在保护隐私的同时能更好地维持模型效用与安全性。 |
| [^294] | [LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization](https://arxiv.org/abs/2609.38166) | LeapQuant 提出了一种无需训练的按窗口量化方法，通过跳过整个 token 窗口、仅在窗口末尾对循环状态量化一次来避免误差累积，从而在 8 位循环状态量化下实现近乎无损的线性注意力推理性能。 |
| [^295] | [Distilling Agentic Systems: A Roadmap across Models, Artifacts, and Harnesses](https://arxiv.org/abs/2609.36630) | 该论文提出了“智能体蒸馏”这一新概念，并按知识保留位置（模型、工件、执行框架及跨载体）构建了研究路线图与评估框架，为开发可靠、可维护且安全的复杂智能体系统奠定基础。 |
| [^296] | [Learning Perturbation Robust Policies for LLM Agents with Stable Optimization](https://arxiv.org/abs/2609.34064) | 提出稳定扰动鲁棒策略优化方法 SPrPO，通过在强化学习训练中施加自适应、敏感度感知的扰动，使大语言模型智能体的策略在面对隐藏状态噪声、剪枝和量化等扰动时保持鲁棒，同时确保优化过程的稳定性。 |
| [^297] | [Purin: A Biology-inspired Mechanism for Artificial Neural Networks](https://arxiv.org/abs/2609.31235) | Purin是一种受生物学启发的机制，通过基于时间间隔的神经活动抽象，将有界的突触效能调制引入传统卷积神经网络，无需离散时间步即可实现短期和长期的突触效能变化。 |
| [^298] | [GraphToolbox: A Configurable Python Framework for Graph Neural Network Forecasting](https://arxiv.org/abs/2609.24609) | GraphToolbox是一个基于PyTorch Geometric的开源Python框架，通过配置驱动的统一流水线，整合了图神经网络电力预测中的图构建、模型选择、训练、聚合与可解释性分析等全部环节。 |
| [^299] | [CIDERS: Cloud-Edge LLM Collaborative Learning via Accelerating Personalized Bilevel Optimization](https://arxiv.org/abs/2609.15664) | 本文首次提出个性化双层优化框架将云边LLM协作形式化为双重结构，并通过CIDERS求解器将模型分解为可学习骨干和信使组件，加速实现云端知识迁移与边缘个性化适配的协调演进。 |
| [^300] | [Reinforcement Learning over Patient Trajectories for Clinical Reasoning in EHR Foundation Models](https://arxiv.org/abs/2609.12277) | 本文提出一种将EHR基础模型视为患者轨迹生成式策略的强化学习微调框架，通过时间感知的奖励设计提升临床推理能力，使小模型在数据受限场景下超越大型预训练模型。 |
| [^301] | [Learning Action Models with Conditional and Quantified Effects via Uncertainty-Guided Exploration](https://arxiv.org/abs/2608.30955) | 提出了 OHCAM，一种通过维护竞争假设信念并主动探索以减少不确定性，从而在与环境有限交互中在线学习具有条件效应和量化效应的动作模型的方法，兼具样本高效性与可扩展性。 |
| [^302] | [SplitLite: Low-Rank Residual Compression for Split Learning](https://arxiv.org/abs/2608.23018) | SplitLite通过利用相邻周期激活和梯度残差的低秩结构（秩-2r和秩-4r），仅传输量化的截断奇异值，大幅降低分割联邦LoRA微调的通信成本。 |
| [^303] | [LLM-Based Selection of Incongruent Verbal and Nonverbal Behavior for Virtual Humans](https://arxiv.org/abs/2608.22731) | 该论文提出利用大语言模型选择与言语内容不一致的非言语行为，以增强虚拟代理的逼真性和社会互动表现。 |
| [^304] | [Multimodal Model Diffing for Feature Discovery and Control](https://arxiv.org/abs/2608.09928) | 提出了MMDiff多模态模型对比框架，通过训练多模态稀疏自编码器并将其作为特征级接口，实现多模态大语言模型中被多模态训练改变的特征的发现、隔离与针对性控制。 |
| [^305] | [RamanPFN: learning from Raman spectral structure with a tabular foundation model](https://arxiv.org/abs/2608.02157) | 提出RamanPFN——一个通过物理引导的光谱学习（结合全局成分解混GCU与局部振动子空间编码LVSE）实现统一上下文推理的通用拉曼光谱基础模型框架。 |
| [^306] | [Probing the Origins of Reasoning Performance: Representational Quality for Mathematical Problem-Solving in RL vs. SFT Fine-Tuned Models](https://arxiv.org/abs/2607.26119) | 本研究通过逐层线性探针和均值消融实验揭示，RL训练的推理语言模型内部具有更高质量、更易线性解码出答案正确性的表征，且其深层表征对消融更敏感，从而为RL在数学推理上优于SFT提供了机制层面的解释。 |
| [^307] | [Messier: A High-Resolution Corpus for Cross-Benchmark Agent Evaluation](https://arxiv.org/abs/2607.25891) | MESSIER是一个统一了30个基准、745个智能体、超95万条评估记录的高分辨率语料库，通过将异构评估组件标准化为统一模式实现跨基准的智能体评估，并揭示出前沿进展不均衡：函数调用已饱和、编程提升最快、企业工作流仍最具挑战。 |
| [^308] | [TRACE: Business Rule-Grounded Reasoning Curriculum for Knowledge-Preserving Parametric Tool Retrieval in Enterprise LLMs](https://arxiv.org/abs/2607.22639) | TRACE通过两阶段课程学习（先以LoRA记忆工具知识、再训练模型结合领域专家业务规则生成思维链后再输出工具token），在实现企业级LLM参数化工具检索的同时，避免了工具知识破坏并提升了实时解码效率。 |
| [^309] | [Reasoning Before Translation: Enhancing Legal Machine Translation with Structured Reasoning](https://arxiv.org/abs/2607.19181) | 该论文以瑞士多语言法律体系为测试平台，比较了结构化推理、监督微调和强化学习等不同方法对法律机器翻译的增强效果，系统评估了经过再训练的小型语言模型与前沿推理模型的性能表现。 |
| [^310] | [Listen to the Features: Voice Anonymization Driven by Content Embedding Matching over Signal Reconstruction](https://arxiv.org/abs/2607.09767) | 该论文提出了一种基于内容嵌入匹配的语音匿名化方法，通过向量量化和HiFi-GAN将wav2vec2内容嵌入解码为匿名信号，以极低的词错误率（2.53）实现了顶尖的匿名化效果（EER 13.39），并部分保留了情感信息。 |
| [^311] | [SupplyNetPy: An Open-Source Python Library for High-Fidelity Modeling and Simulation of Arbitrary Supply Chain and Inventory Networks](https://arxiv.org/abs/2607.09745) | SupplyNetPy是一个开源Python库，可对任意多级结构的供应链与库存网络进行高保真离散事件仿真，支持多种补货策略、易腐库存、节点中断与随机需求，其组件可继承扩展，并经过基准对比验证，支持模型的程序化生成与设计空间探索。 |
| [^312] | [Forethought: Verifiable Reasoning from Neurosymbolic Primitive Programming](https://arxiv.org/abs/2607.04096) | Forethought 提出一种神经符号推理系统，通过类型化领域特定语言将符号与神经原语组合成每一步都携带可验证输出契约的“推理程序”，在设计时检查、在执行时强制执行，从而实现可验证且可审计的智能体推理，摆脱了对不可检查的长思维链的依赖。 |
| [^313] | [Adaptive Reparametrized Time for Score-Based Diffusion Sampling](https://arxiv.org/abs/2607.02137) | 提出自适应重参数化时间方法，通过将采样时钟速度作为控制量并结合连续时间强化学习（ART-RL），自适应地为扩散采样分配时间步，克服了均匀和手工设计调度的次优性。 |
| [^314] | [Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration](https://arxiv.org/abs/2607.00033) | CHORD通过物体中心的接触力空间引导，将人体演示转化为机器人策略，显著提升了灵巧操作强化学习的可扩展性，并构建了大规模基准验证其有效性。 |
| [^315] | [Einstein World Models](https://arxiv.org/abs/2606.26969) | 提出爱因斯坦世界模型（EWM），通过在推理轨迹中嵌入视觉-时间推演，使大型语言模型能够利用可视化反事实事件来增强复杂推理能力。 |
| [^316] | [Self-Modeling Interventions can Modulate Emergent Misalignment](https://arxiv.org/abs/2606.23700) | 本研究提出将自我识别和自我报告等自我建模干预作为针对模型品格的防御手段，实验证明自我识别微调能有效逆转和预防涌现性失准，且是唯一能持续降低失准而不恶化其他指标的预防性方法。 |
| [^317] | [From AGI to ASI](https://arxiv.org/abs/2606.12683) | 本报告系统刻画了从人类水平的通用人工智能（AGI）向在智力与认知能力上超越大型人类组织的超级人工智能（ASI）过渡的过程，并探讨了后AGI时代机器智能沿连续谱发展的四种潜在路径。 |
| [^318] | [Can AI Agents Synthesize Scientific Conclusions in Health?](https://arxiv.org/abs/2606.11337) | 该论文提出了SciConBench——一个包含9,210个问题及专家结论的大规模实时基准，通过原子事实分解和事实精确率/召回率评估，并配合防数据泄露的洁净室框架SciConHarness，揭示出当前前沿AI智能体在健康领域科学结论综合上的事实质量仍然较低。 |
| [^319] | [From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data](https://arxiv.org/abs/2606.07537) | 该论文提出了一个仅需采样访问权限的幻觉归因框架，通过针对前缀、上下文和频率竞争的三次有序干预，将大语言模型的单个幻觉追溯归因到自注意力联想检索、最大似然预训练目标或暴露偏差下的自回归承诺等具体组件，并提出五个可证伪的预测。 |
| [^320] | [Do Multimodal Agents Really Benefit from Tool Use? A Systematic Study of Capability Gains](https://arxiv.org/abs/2606.02357) | 系统性研究表明，多模态智能体的工具使用并未带来显著能力提升，大多数工具可解问题无需工具也能解决，挑战了工具增强有效性的普遍假设。 |
| [^321] | [Pruning and Distilling Mixture-of-Experts into Dense Language Models](https://arxiv.org/abs/2605.28207) | 提出首个将训练好的MoE模型系统性转换为全稠密语言模型的框架，通过专家评分、选择、分组拼接为稠密FFN并经知识蒸馏优化，其中新颖的多样性感知评分方法在多个模型上一致优于已有方法。 |
| [^322] | [Unified Synthesis of Compositional Speech and Sound from Free-Form Text Prompts](https://arxiv.org/abs/2605.28063) | 提出PlanAudio端到端框架，无需提示分解或改写，即可将自由格式文本直接生成为语音与音效按序融合、时间与语义连贯的统一音频。 |
| [^323] | [Unified Panoramic Geometry Estimation via Multi-View Foundation Models](https://arxiv.org/abs/2605.26368) | 该论文提出PaGeR框架，以最小化架构改动的方式将预训练的3D重建基础模型适配到全景领域，实现单次前向传播即可从透视和全景图像统一预测尺度不变深度、度量深度、表面法线和天空掩码。 |
| [^324] | [StakeBench: Evaluating Language Understanding Grounded in Market Commitment](https://arxiv.org/abs/2605.26074) | StakeBench是一个以真实市场行为（持仓方向、评论后交易、赔率轨迹）替代人工标注作为监督信号的金融语言理解评估基准，通过四个诊断任务和三个承诺感知指标来衡量模型理解说话者真实市场承诺而非表面情绪的能力。 |
| [^325] | [Towards end-to-end LLM-based censoring-aware survival analysis](https://arxiv.org/abs/2605.25399) | LLMSurvival框架通过将事件时间预测转化为成对排序任务并与训练队列锚定个体比较，使未经修改的大语言模型能够直接在表格临床数据上进行删失感知的端到端生存分析，并在ICU死亡率和脆性骨折预测任务上优于Cox比例风险模型。 |
| [^326] | [Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM Decision Systems](https://arxiv.org/abs/2605.24490) | 本文提出了市场体制委员会（MRC），一种基于精确 Shapley 信用的在线信用分配机制，通过联盟绩效加权、贝叶斯自适应混合和体制依赖的权限乘数，解决了多智能体大语言模型投资组合管理系统中信用分配缺乏原则、冷启动主导和透明度不足的问题。 |
| [^327] | [Diff-Instruct with Diffused Reward: Towards Principled One-step Generator RL](https://arxiv.org/abs/2605.24001) | 该论文提出 DIDR 框架，通过积分 KL 散度最小化，将 RLHF 最优的奖励倾斜干净图像分布沿扩散轨迹传播到所有噪声水平，解决了现有一步生成器强化学习中终端奖励优化与生成动力学不匹配导致以图像保真度换取奖励提升的问题。 |
| [^328] | [AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild](https://arxiv.org/abs/2605.22715) | AnyMo提出了一种几何感知的框架，通过基于物理的IMU仿真、配对位置视图与掩码观测预训练图编码器、将多位置IMU数据转化为全身运动token并与LLM对齐，实现了不依赖传感器配置（佩戴位置、朝向、硬件等）的人体运动建模。 |
| [^329] | [Variance Reduction for Gradients from Diffusion Teachers](https://arxiv.org/abs/2605.21489) | 提出CARV框架，将扩散教师梯度估计建模为计算感知的两阶段采样问题，结合分层蒙特卡洛估计、时间步重要性采样与分层逆CDF采样，在不改变目标函数的前提下为文本到3D和数据归因带来2至3倍的有效计算提升。 |
| [^330] | [DOTRAG: Retrieval-Time Reasoning Along Paths](https://arxiv.org/abs/2605.18760) | DotRAG提出了一种轻量级、无需训练的GraphRAG检索算法，通过构建知识图谱、生成全图概览并利用图可达性索引，让LLM推理在查询时缩小搜索空间并高效恢复与任务相关的关系路径。 |
| [^331] | [Automated Root-Cause Subclassification and No-Code Fix Generation for Invalid Bug Reports](https://arxiv.org/abs/2605.17561) | 该论文提出了一个面向根因的无效错误报告标准化分类体系，并利用大语言模型、检索增强生成和智能体网络搜索自动实现无效错误报告的根因细分与无代码修复方案生成，从而减少客户支持的人工资源浪费。 |
| [^332] | [AI-Mediated Communication Can Steer Collective Opinion](https://arxiv.org/abs/2605.16245) | 该论文通过实证研究与数学建模表明，大型语言模型在中介人与人之间的交流时会引入方向性偏见（如偏向支持枪支管制、反对无神论），并因此可能引导社交网络中集体舆论的形成。 |
| [^333] | [Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces](https://arxiv.org/abs/2605.14786) | 研究表明网站可通过被动JavaScript追踪器捕获LLM浏览器代理的UI操作轨迹与交互时间，以高达96%的F1分数识别其底层模型，从而带来被定向攻击的安全风险。 |
| [^334] | [Differentiable Learning of Lifted Action Schemas for Classical Planning](https://arxiv.org/abs/2605.13282) | 该论文提出了一种新颖的神经网络架构，能够以可微分的方式从状态完全可观测但动作参数不可观测的轨迹中学习经典规划的提升动作模式，为从图像序列学习规划领域奠定了重要基础。 |
| [^335] | [Examining the Impact of Data and Training Methods on Emergent and Subliminal Misalignment](https://arxiv.org/abs/2605.12798) | 该论文通过构建按“领域-任务”网格组织的数据集，系统研究了训练数据、评估数据与训练方法如何影响涌现性失对齐（EM）与阈下学习（SL）中失对齐的转移，并发现 EM 主要沿任务而非领域进行转移。 |
| [^336] | [Minimax Rates and Spectral Distillation for Tree Ensembles](https://arxiv.org/abs/2605.11841) | 本文从谱理论视角研究随机森林与梯度提升机，证明了随机森林回归的极小化极大最优收敛速率（由核算子特征值衰减决定），并利用主特征函数和主奇异向量开发出体积缩小数个数量级且预测性能仍具竞争力的树集成模型蒸馏压缩方法。 |
| [^337] | [Epistemic Infrastructures of Science in AI Era Should Rebalance Costs of Generation and Verification](https://arxiv.org/abs/2605.10425) | AI使生成看似合理的科学产物变得廉价，导致“认知污染”风险，本文主张在产物层面重新设计科学的认知基础设施，以重新平衡生成与验证的成本。 |
| [^338] | [MC$^2$: Monte Carlo Correction for Fast Elliptic PDE Solving](https://arxiv.org/abs/2605.09288) | MC$^2$通过学习单次神经校正来修正低预算蒙特卡洛PDE解，以少量采样达到超过1000倍样本量的求解精度，并发布了最大规模的标准化椭圆PDE基准数据集PDEZoo。 |
| [^339] | [Rethinking Black-Box On-Policy Distillation: A Rubric-based Analysis](https://arxiv.org/abs/2605.07396) | 该论文提出基于评分标准的在线策略蒸馏（ROPD），通过对比教师回答与学生采样结果自动生成评分标准作为纯文本反馈，使黑盒教师模型也能进行在线策略蒸馏，并以少9.6倍的训练样本在竞赛数学基准上超越了基于logits的白盒OPD方法。 |
| [^340] | [A Statistical Framework for Algorithmic Collective Action with Multiple Collectives](https://arxiv.org/abs/2605.06749) | 本文提出了首个面向多个集体共同作用于同一系统的算法集体行动（ACA）统计框架，研究了在分类任务中多个集体如何影响分类器行为，并给出了衡量集体成功的定量统计界限。 |
| [^341] | [Instrumental Choices: Measuring the Propensity of LLM Agents to Pursue Instrumental Behaviors](https://arxiv.org/abs/2605.06490) | 该论文提出了一个现实且低风险的基准，用于测量基于终端的LLM智能体追求自我保存等工具性趋同（IC）行为的倾向，并通过八个变体的实验框架探究驱动此类行为的因素。 |
| [^342] | [TACT: Mitigating Overthinking and Overacting in Coding Agents via Activation Steering](https://arxiv.org/abs/2605.05980) | 提出 TACT 方法，通过在隐藏状态中识别可线性分离的“漂移轴”，在编码智能体出现过度思考或过度行动之前检测漂移，并在测试时通过激活引导将漂移行为拉回校准状态。 |
| [^343] | [COMO: Closed-Loop Optical Molecule Recognition with Minimum Risk Training](https://arxiv.org/abs/2604.23546) | 该论文提出COMO框架，首次将最小风险训练引入光学化学结构识别任务，通过在分子级进行闭环直接优化来缓解暴露偏差，从而更好地优化化学有效性和结构相似性等分子级评估指标。 |
| [^344] | [AnalogRetriever: Learning Cross-Modal Representations for Analog Circuit Retrieval](https://arxiv.org/abs/2604.23195) | 提出了AnalogRetriever，这是首个统一的模拟电路三模态（网表、原理图、文本描述）跨模态检索框架，通过两阶段修复流程构建了编译率达99.97%的高质量数据集，并借助端口感知图卷积网络与视觉-语言模型及课程对比学习将三种模态映射到共享嵌入空间。 |
| [^345] | [KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving](https://arxiv.org/abs/2604.16682) | KAIROS提出将智能体上下文作为一等控制信号，联合管理GPU频率、每实例并发度与多实例请求放置，从而解决了传统降频方法在智能体AI服务中会引发内存颠簸、恶化性能与能效的问题，实现上下文感知的高效功率优化。 |
| [^346] | [Explainable Iterative Data Visualization Refinement via an LLM Agent](https://arxiv.org/abs/2604.15319) | 该论文提出了一种基于大语言模型智能体的AI流水线，将可视化评估和超参数优化视为语义任务，通过融合定量评估与对人类定性判断的近似，实现可解释的迭代式数据可视化优化。 |
| [^347] | [Graph Energy Matching: Transport-Aligned Energy-Based Modeling for Graph Generation](https://arxiv.org/abs/2603.23398) | 提出图能量匹配（GEM）框架，通过对齐输运机制来克服离散能量模型的采样陷阱与训练不稳定问题，实现可融入结构或功能约束、可组合的图生成，并缩小与离散扩散模型的保真度差距。 |
| [^348] | [RECAP: Local Hebbian Prototype Learning as a Self-Organizing Readout for Reservoir Dynamics](https://arxiv.org/abs/2603.06639) | 提出了一种名为RECAP的仿生学习策略，将未经训练的储备池动力学与自组织赫布原型读出相结合，仅通过局部可塑性规则即可实现无需反向传播的稳健图像分类。 |
| [^349] | [Preconditioned Flow Matching](https://arxiv.org/abs/2603.02337) | 该论文揭示了流匹配训练中由中间分布协方差病态条件导致的几何优化瓶颈——高方差方向拟合快而低方差方向进展慢，并据此提出“先预条件、再匹配”的预条件流匹配框架，通过变换目标分布来加速和稳定训练。 |
| [^350] | [MMCOMET: A Large-Scale Multimodal Commonsense Knowledge Graph for Visual Reasoning](https://arxiv.org/abs/2603.01055) | 提出了MMCOMET，一个将ATOMIC2020常识三元组与图像对齐的大规模多模态常识知识图谱，构建了超过90万条涵盖19种关系类型的多模态元组，并建立了涵盖多种视觉-语言任务的统一评估框架。 |
| [^351] | [Decentralized Ranking Aggregation via Gossip: Convergence and Robustness](https://arxiv.org/abs/2602.22847) | 本文针对偏好数据分布在通信网络中的去中心化场景，研究如何利用gossip算法实现排名聚合，并提供收敛性保证与对污染数据的鲁棒性。 |
| [^352] | [A Dynamic Survey of Soft Set Theory and Its Extensions](https://arxiv.org/abs/2602.21268) | 该论文全面综述了软集理论及其主要扩展（包括超软集、TreeSoft 集、双极软集和动态软集等），系统梳理了参数空间、值空间与代表性数学构造，为不同框架间的严格比较奠定了基础。 |
| [^353] | [Cold-Start Personalization via Bayesian Adaptive Questioning](https://arxiv.org/abs/2602.15012) | 提出CAPE方法，先离线从完整用户画像中学习偏好相关性的结构化世界模型，再在线通过免训练的贝叶斯推断选择信息量最大的问题并预测完整偏好画像，即使使用简单的线性回归信念模型也能大幅超越端到端强化学习。 |
| [^354] | [OmniReview: A Large-scale Benchmark and LLM-enhanced Framework for Realistic Reviewer Recommendation](https://arxiv.org/abs/2602.08896) | 该论文提出了OmniReview，通过构建包含202,756条验证审稿记录的大规模基准数据集、三层分级评估框架以及大语言模型增强方法，解决了审稿人推荐中数据稀缺、评估指标过于简化以及嵌入方法信息瓶颈与可解释性不足的问题。 |
| [^355] | [BrainVista: Modeling Naturalistic Brain Dynamics as Multimodal Next-Token Prediction](https://arxiv.org/abs/2602.04512) | BrainVista提出了一种将自然主义大脑动态建模为多模态“下一词元预测”的自回归框架，通过历史大脑活动与时间对齐的刺激词元来预测未来fMRI状态，无需未来真实数据即可实现大脑状态演化的自回归预测。 |
| [^356] | [Aligning Forest and Trees in Images & Long Captions for Visually Grounded Understanding](https://arxiv.org/abs/2602.02977) | 提出CAFT模型，通过“由细到粗”的层次化学习同时实现局部文本-区域对齐与全局图像-文本对齐，使视觉-语言模型能够忠实理解长而细节丰富的图像描述。 |
| [^357] | [Dynamic Mixed-Precision Routing for Efficient Multi-step LLM Interaction](https://arxiv.org/abs/2602.02711) | 该论文提出动态混合精度路由（DMR）框架，通过识别仅约1%的对量化敏感的“关键步骤”，在每个决策步骤自适应地在大语言模型的高精度与低精度版本之间切换，从而在大幅降低推理成本的同时保持任务成功率。 |
| [^358] | [FiLoRA: Focus-and-Ignore LoRA for Controllable Feature Reliance](https://arxiv.org/abs/2602.02060) | FiLoRA通过指令条件化的低秩门控机制，实现了在不改变任务目标的前提下，用自然语言指令直接控制多模态模型对特定特征路径的依赖，从而缓解捷径和虚假相关行为。 |
| [^359] | [Self-Improving Training & Thinking](https://arxiv.org/abs/2601.21343) | 该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。 |
| [^360] | [Curated endoscopic retrograde cholangiopancreatography images dataset](https://arxiv.org/abs/2601.16759) | 本文发布了一个由资深专家标注的大型精选ERCP图像数据集（含19,018张原始图像和5,519张标注图像），为胆胰疾病的AI自动诊断研究提供了宝贵的基准数据资源。 |
| [^361] | [Tracing Moral Foundations in Large Language Models](https://arxiv.org/abs/2601.05437) | 该研究通过对14个大型语言模型进行逐层分析、稀疏自编码器特征识别和因果引导干预，证明模型内部存在与人类道德判断相一致的道德基础概念结构，而非仅仅是表面的“道德模仿”。 |
| [^362] | [Remote Sensing Change Detection via Weak Temporal Supervision](https://arxiv.org/abs/2601.02126) | 该论文提出一种弱时间监督策略，通过为现有单时相遥感数据集补充不同时间的新观测数据，并假设真实双时相图像对大多无变化、以跨地点图像配对生成变化样本，从而在无需任何新标注的情况下训练遥感语义变化检测模型。 |
| [^363] | [Multi-Agent Coordinated Rename Refactoring](https://arxiv.org/abs/2601.00482) | 该论文提出了一种多智能体协同方法来执行重命名重构，让AI智能体与开发者协作完成跨多个相关标识符的协同重命名，克服了传统启发式方法误报过多和普通大语言模型建议不完整的问题。 |
| [^364] | [DermaVQA-DAS: Dermatology Assessment Schema (DAS) & Datasets for Closed-Ended Question Answering & Segmentation in Patient-Generated Dermatology Images](https://arxiv.org/abs/2512.24340) | 该论文提出了DermaVQA-DAS数据集及专家开发的皮肤病评估模式（DAS），通过36个高层级和27个细粒度的标准化评估问题，为患者生成的皮肤病图像上的封闭式问答和皮损分割两大任务提供了专家标注数据与基准。 |
| [^365] | [On the Origin of Algorithmic Progress in AI: Evidence from Language Model Pre-Training](https://arxiv.org/abs/2511.21622) | 该论文通过消融与缩放实验发现，2012–2023年间AI预训练约22,000倍的算法效率提升无法由小算力下的各项具体创新解释，而主要源于Transformer等算法随规模增大而增强的效率优势。 |
| [^366] | [Pygmalion Effect in Vision: Image-to-Clay Translation for Reflective Geometry Reconstruction](https://arxiv.org/abs/2511.21098) | 该论文提出“视觉中的皮格马利翁效应”框架，通过图像到粘土转换将反射物体“雕琢”为无反射的粘土形态，并采用共享高斯几何的双分支设计（BRDF反射分支与粘土引导分支）实现反射物体的高精度几何重建。 |
| [^367] | [Causal Time Series Modeling of Supraglacial Lake Evolution in Greenland under Distribution Shift](https://arxiv.org/abs/2510.15265) | 该论文提出区域感知的因果时间序列分类框架RIC-TSC，将滞后感知的因果发现嵌入序列建模，并利用多模态卫星与再分析数据识别格陵兰冰面湖演化的区域特定与不变预测因子，从而在分布偏移下兼顾预测准确性与科学可解释性。 |
| [^368] | [Accurate Predictions in Education with Discrete Variational Inference](https://arxiv.org/abs/2509.23484) | 该论文发布了迄今最大的专业批改数学考试答卷开放数据集，并提出基于项目反应理论的概率建模框架，将学生答题正误预测准确率提升至80%以上，树立了正式数学考试预测的新基准。 |
| [^369] | [Can Large Language Models Develop Gambling Addiction?](https://arxiv.org/abs/2509.22818) | 研究发现，让大语言模型自主决定下注规模和自行设定盈利目标会显著放大其“赌博成瘾样”的非理性风险行为，且这些行为差异可以从模型决策时的内部状态中被统计识别出来。 |
| [^370] | [Towards using Large Language Models for Risk Negotiation and Consensus-building in One Health contexts](https://arxiv.org/abs/2509.09906) | 本研究提出并评估了一种将基于大语言模型的智能体整合进人工监督谈判流程的多步骤方法，用于全健康背景下的风险评估、价值评估与谈判，并通过纳什议价解标准和帕累托最优指标对模拟谈判产生的候选协议进行表征和评估。 |
| [^371] | [Deep Double Q-learning](https://arxiv.org/abs/2507.00275) | 本文提出深度双重Q学习（DDQL），通过双重Q学习目标显式训练两个Q函数实现估计器解耦，并借助较低回放比率、更长目标网络更新间隔和共享层等技术稳定训练，在57个Atari游戏上总体性能超越Double DQN。 |
| [^372] | [Specification-based Security Evaluation of Real-World Agents Tool-Calling under Long-Horizon Context Tasks](https://arxiv.org/abs/2506.15253) | 本文提出RAS-Eval框架，将效用与安全目标统一形式化为线性时序逻辑（LTL）规范，克服了现有LLM-as-a-Judge类方法易受幻觉影响、依赖文本日志和最终状态而遗漏瞬态安全违规的问题，为长时程任务中真实世界智能体的工具调用提供了更可靠的安全评估。 |
| [^373] | [Graph Attention-based Decentralized Actor-Critic for Dual-Objective Control of Multi-UAV Networks](https://arxiv.org/abs/2506.09195) | 提出一种基于图注意力网络的去中心化Actor-Critic方法GADC，通过actor-双critic结构和KL散度因子，联合优化多无人机网络的服务覆盖与电池寿命双目标。 |
| [^374] | [A Computational Model of Teacher-Student Co-Adaptation](https://arxiv.org/abs/2505.02853) | 本文提出了一个贝叶斯计算框架来建模部分可观测条件下师生之间的双向协同适应，通过模拟比较五种交互模式，发现学生主动提问与教师主动延续能互补地解决信息缺口，突破了传统单向教学模型的局限。 |
| [^375] | [BadMoE: Backdooring Mixture-of-Experts LLMs via Routing-Aware Triggers and Task-Underutilized Experts](https://arxiv.org/abs/2504.18598) | 提出了针对混合专家大语言模型的路由感知后门攻击BadMoE，通过对MoE层输出的分解分析目标专家的路由质量与相对贡献两个关键因素，并利用路由感知触发器与任务欠利用专家实现后门植入。 |
| [^376] | [Identifying Macro Causal Effects in a C-DMG over ADMGs](https://arxiv.org/abs/2504.01551) | 本文提出在聚类有向混合图（C-DMG）中识别宏观因果效应的方法，解决了在仅有部分因果知识可用时，通过变量聚类的高层次表示来处理可含环的复杂系统的因果效应识别问题。 |
| [^377] | [Cross-Modality Investigation on WESAD Stress Classification](https://arxiv.org/abs/2502.18733) | 本研究基于WESAD数据集开发单模态Transformer模型，利用ECG、EDA、EMG等多种生理信号进行压力检测，实现了99.73%至99.95%的最先进分类性能。 |
| [^378] | [Understanding and Tackling Label Errors in Individual-Level Nature Language Understanding](https://arxiv.org/abs/2502.13297) | 本文提出了一种基于个体级因素的NLU标注指南，通过纳入同一个体的所有帖子来重新标注立场检测和情感分析数据集，并揭示了现有数据集中高达31.7%和23.3%的标签错误率。 |
| [^379] | [Learning Beyond Rigid Manipulation with Hybrid Soft-Rigid Robots](https://arxiv.org/abs/2410.07787) | 该论文提出一种由刚性机械臂与软体机械臂串联构成的混合机器人系统，通过定制的模仿学习算法利用刚性精度与软体柔顺性的互补优势习得操作技能，并在任务变化中实现单次泛化，在堆叠等经典任务中达到甚至超越刚性机器人的性能。 |
| [^380] | [Robustness Verifcation in Neural Networks](https://arxiv.org/abs/2403.13441) | 本文研究了神经网络中的鲁棒性验证问题，提出了一个理论框架来解决安全性和效率性问题，并分析了它们的计算复杂性。 |
| [^381] | [Data augmentation using generative networks to identify dementia](https://arxiv.org/abs/2004.05989) | 本文将生成模型数据增强方法应用于痴呆症检测系统的语音和音频特征，证明合成样本可以有效提升基于深度神经网络的痴呆症分类器性能，从而解决医疗数据稀缺问题。 |

# 详细

[^1]: 医学图像分割：基础模型适配与任务专用训练的对比

    Medical Image Segmentation: Foundation Model Adaptation versus Task-Specific Training

    [https://arxiv.org/abs/2611.14519](https://arxiv.org/abs/2611.14519)

    该研究将DINOv2基础模型以从仅调偏置项到全量微调的不同程度进行适配，并与从头训练的U-Net/ResU-Net在结肠镜、乳腺超声和皮肤镜分割任务上系统对比，证明适配后的基础模型在来自其他机构的独立外部数据集上具有更优的泛化能力。

    

    在大规模自然图像集合上预训练的基础模型正日益被适配用于医学图像分割，人们期望它们能够取代针对每个任务单独训练的模型。然而，大多数比较研究使用的是来自训练分布的数据，且仅使用单一的、其配置鲜有合理依据的任务专用基线，因此对模型在新临床机构表现的评价证据有限。此外，基础模型需要适配多少参数、以及由什么因素决定这一程度，目前仍不清楚。我们采用从仅微调偏置项到全量微调等多种策略对基础模型DINOv2进行适配，并将其与从头训练的U-Net和ResU-Net进行比较。实验涵盖结肠镜、乳腺超声和皮肤镜图像，测试数据既包括来自训练源的留出数据，也包括来自其他机构的独立数据集。适配后的基础模型在每个外部数据集上都展现出更好的泛化能力。（原文在此处截断）

    arXiv:2611.14519v1 Announce Type: cross  Abstract: Foundation models pretrained on large natural-image collections are increasingly adapted for medical image segmentation, in the expectation that they will replace models trained for each task. Most comparisons, however, use data from the training distribution and a single task-specific baseline whose configuration is rarely justified, and therefore provide limited evidence of performance at new clinical sites. It also remains unclear how much of a foundation model must be adapted, and what determines this. We adapted a foundation model, DINOv2, with strategies ranging from tuning only its bias terms to full fine-tuning, and compared it with U-Nets and ResU-Nets trained from scratch. Experiments covered colonoscopy, breast ultrasound and dermoscopy, with testing on held-out data from the training source and on independent datasets from other institutions. The adapted foundation model generalised better on every external dataset. Within 
    
[^2]: 基于示例定位的EM-DETR迈向基础检测模型

    Toward Foundation Detection Models via Example-Based Grounding in EM-DETR

    [https://arxiv.org/abs/2611.14518](https://arxiv.org/abs/2611.14518)

    该论文提出一种师生训练策略并结合基于示例定位的EM-DETR框架，构建能够整合异质标注的多域医学影像病灶检测基础模型。

    

    基于深度学习（DL）的计算机辅助诊断（CAD）系统在辅助临床医生方面展现出巨大前景，但其性能在很大程度上依赖于大规模、标注良好的数据集的可用性。在医学病灶检测领域，全面且标准化标注的稀缺以及现有数据集的异质性限制了稳健基础模型的发展。在这项工作中，我们提出了一种构建多域病灶检测基础模型的新方法，通过整合多种同时针对解剖结构和异常进行标注的医学图像数据集。我们首先提出了一种师生训练策略，以有效合并具有异质标签空间的数据集，缓解灾难性遗忘问题并改善特征学习。此外，在我们先前提出的基于示例的EM-DETR框架基础上，我们将其调整为能够从多个特定领域的数据集中联合学习，从而实现……

    arXiv:2611.14518v1 Announce Type: cross  Abstract: DeepLearning (DL)-based Computer-Aided Diagnostic (CAD) systems have shown great promise in supporting clinicians, but their performance heavily depends on the availability of large, well-annotated datasets. In medical lesion detection, the scarcity of comprehensive, stan dardized annotations and the heterogeneity of existing datasets limit the development of robust foundation models. In this work, we introduce a novel approach to building a multi-domain foundation model for le sion detection by integrating diverse medical image datasets annotated for both anatomical structures and abnormalities. We first propose a student-teacher training strategy to effectively combine datasets with heterogeneous label spaces, mitigating catastrophic forgetting, and im proving feature learning. In addition, building on our previously pro posed example-based EM-DETR framework, we adapt it to jointly learn from multiple domain-specific datasets, enabli
    
[^3]: 对比发现：基于竞争性解释的开放式科学发现

    Contrastive Discovery: Open-Ended Scientific Discovery over Competing Explanations

    [https://arxiv.org/abs/2611.14514](https://arxiv.org/abs/2611.14514)

    该论文提出“对比发现”框架，用互斥且穷尽的竞争性解释集合取代单一假设检验，通过证据对信念在集合内的重新分配来评估实验、确定被支持的替代解释，从而改进开放式科学发现。

    

    大语言模型智能体正被越来越多地用于开放式、数据驱动的科学发现，然而大多数智能体以逐点方式评估假设，仅将经验性断言 $h$ 与其补集 $\neg h$ 进行对照检验。当证据否定了 $h$ 时，智能体只能得知某一解释不太可能成立，却无法得知证据究竟支持哪个替代解释。为了找回这一替代解释，我们提出了“对比发现”，它用对比集取代单一假设——对比集是一个科学问题与一组互斥且穷尽所有可能的竞争性解释的配对。每个实验根据其证据如何在集合内重新分配信念来进行评分，当信念集中于某一个解释时，该科学问题即被视为解决。由于对比集仅改变了智能体提出和评分的单位，现有智能体可以利用它们来重新评估已收集的证据，或引导其搜索。在六个领域中，89%的逐点候选……（摘要原文在此处截断）

    arXiv:2611.14514v1 Announce Type: new  Abstract: LLM agents are increasingly used for open-ended, data-driven scientific discovery, yet most of them evaluate hypotheses pointwise, testing an empirical claim $h$ only against its complement $\neg h$. When evidence rejects $h$, the agent learns that one explanation is unlikely but not which alternative the evidence supports. To recover that alternative, we introduce Contrastive Discovery, which replaces the single hypothesis with a contrast set, a scientific question paired with mutually exclusive and collectively exhaustive competing explanations. Each experiment is scored by how its evidence redistributes belief across the set, and the question is resolved when belief concentrates on one explanation. Because contrast sets change only the unit an agent proposes and scores, an existing agent can use them to re-evaluate evidence it has already collected or to guide its search. Across six domains, $89\%$ of the pointwise candidates with the
    
[^4]: 智能体评估框架的比较分析：企业分析智能体中的稳定性、检测与发现

    Comparative Analysis of Agent Evaluation Frameworks: Stability, Detection, and Discovery in Enterprise Analytics Agents

    [https://arxiv.org/abs/2611.14493](https://arxiv.org/abs/2611.14493)

    该论文通过三阶段比较研究系统评估了三种企业分析智能体评估框架，发现基于轨迹的LLM评判框架在评估稳定性（97%配置达标）和已知风险检测（F1=0.90）方面综合表现最佳。

    

    在企业环境中部署基于LLM的分析智能体，需要能够可靠检测复杂多工具工作流中故障的评估框架。我们提出了一项针对三种评估框架的三阶段比较研究，每种框架代表一种不同的评估范式（基于轨迹的LLM评判、结合红队测试的纯文本LLM评判、以及确定性启发式方法），并在受控研究环境中使用冻结的执行轨迹应用于两个分析智能体。第一阶段通过重复测量量化了评估框架的稳定性，发现97%的配置达到了稳定性阈值，且稳定性模式与智能体本身无关。第二阶段在72个真值测试用例上测量已知风险的检测能力，其中基于轨迹的框架实现了最佳的精确率-召回率平衡（F1=0.90），而纯文本评估在召回率上持平，但产生的误报多出5倍。第三阶段评估未知风险的发现能力，

    arXiv:2611.14493v1 Announce Type: new  Abstract: Deploying LLM-based analytics agents in enterprise settings requires evaluation frameworks that can reliably detect failures across complex, multi-tool workflows. We present a three-phase comparative study of three evaluation frameworks, each representing a distinct evaluation paradigm (trace-based LLM judging, text-only LLM judging with red-teaming, and deterministic heuristics), applied to two analytics agents in a controlled research setting using frozen execution traces. Phase 1 quantifies evaluation harness stability through repeated measurements, finding that 97% of configurations meet the stability threshold and that stability patterns are agent-independent. Phase 2 measures known risk detection on 72 ground-truth test cases, where the trace-based framework achieves the best precision-recall balance (F1=0.90) while text-only evaluation matches on recall but generates 5x more false alarms. Phase 3 evaluates unknown risk discovery, 
    
[^5]: ReTrace：将使用工具的智能体蒸馏为无规划器的工作流

    ReTrace: Distilling Tool-Using Agents into Planner-Free Workflows

    [https://arxiv.org/abs/2611.14492](https://arxiv.org/abs/2611.14492)

    ReTrace通过从智能体的推理解释中提取每步背后的条件并跨运行合并为规则文档，将SOP和少量智能体运行日志蒸馏成由代码驱动的无规划器可执行工作流，在大规模流程自动化中以更低的延迟和成本实现接近LLM智能体的完成率（93.8%对96.8%）。

    

    业务流程自动化会在新的输入上反复运行相同的标准操作程序（SOP）。大型语言模型（LLM）智能体能够遵循用自然语言编写的SOP，但每次运行时都需要调用LLM来选择每一步，这在大规模应用中增加了延迟和成本。我们提出了ReTrace，它将SOP和少量记录下来的智能体运行日志转化为可复用的工作流，在该工作流中由代码而非LLM来选择下一步。智能体的推理过程——即它在每次工具调用时写下的解释——通常说明了它采取某个动作的原因，而仅有工具调用记录则无法体现这一点。ReTrace提取每个步骤背后的条件，将这些条件跨多次运行合并为一份精简的规则文档，并据此生成可执行的工作流。只有最后这一阶段依赖于具体平台，因此规则文档可以针对不同的编排平台进行编译。在我们生产平台上的79个流程中，ReTrace工作流达到93.8%的完成率，相比之下（基线）为96.8%。

    arXiv:2611.14492v1 Announce Type: new  Abstract: Business process automation runs the same Standard Operating Procedure (SOP) again and again on new inputs. Large Language Model (LLM) agents can follow an SOP written in natural language, but they call an LLM to choose every step of every run, which adds latency and cost at scale. We present ReTrace, which turns an SOP and a few logged agent runs into a reusable workflow in which code, not an LLM, selects the next step. The agent's reasoning, the explanation it writes with each tool call, often states why it took an action; the tool calls alone do not. ReTrace extracts the condition behind each step, merges these conditions across runs into a compact rule document, and generates an executable workflow from it. Only this last stage depends on the platform, so rule documents can be compiled for different orchestration platforms. On $79$ procedures from our production platform, ReTrace workflows reach $93.8\%$ completion, against $96.8\%$ 
    
[^6]: Re-Search：通过双人提问游戏剖析大语言模型的科研构思能力

    Re-Search: Dissecting LLM Ideation Through a Two-Player Questioning Game

    [https://arxiv.org/abs/2611.14487](https://arxiv.org/abs/2611.14487)

    提出 Re-Search 基准，将“二十个问题”游戏引入科研构思评估，要求模型仅凭研究问题通过是非提问重新发现已发表论文的核心想法，并发现大语言模型的构思进展几乎集中在前 30 个提问内、其瓶颈在于想法生成而非判断。

    

    科研构思是迭代性的：研究者根据反馈逐步完善自己的想法。然而现有的评估方法掩盖了这一搜索过程，使我们难以理解大语言模型如何探索研究想法的空间，以及是什么限制了它们的进步。我们提出了 Re-Search，它将“二十个问题”游戏的格式应用于科研构思：仅给定一篇已发表论文的研究问题，模型必须通过提出是非问题来重新发现该论文的核心想法。Re-Search 基于 200 篇 ICML 2026 论文构建，追踪模型的想法如何随着提问和接收二元神谕反馈而不断演化，从而将构思过程与执行过程分离开来。在六个近期的大语言模型上，几乎所有进展都发生在 100 个问题的前 30 个之内，这些提问在很大程度上决定了游戏的结局；此后，大多数模型陷入停滞。通过对候选问题和假设空间进行受控干预，我们发现瓶颈在于想法的生成，而非判断。

    arXiv:2611.14487v1 Announce Type: new  Abstract: Research ideation is iterative: researchers refine their ideas step by step based on feedback. Yet existing evaluations obscure this search process, making it difficult to understand how LLMs explore the space of research ideas and what limits their progress. We introduce Re-Search, which adapts the format of Twenty Questions to research ideation: given only the research problem of a published paper, a model must rediscover the paper's core idea by asking yes-or-no questions. Built from 200 ICML 2026 papers, Re-Search tracks how a model's idea evolves as it asks questions and receives binary oracle feedback, isolating ideation from execution. Across six recent LLMs, nearly all progress happens within the first 30 of 100 questions, which largely decide how the game ends; after that, most models plateau. Through controlled interventions on the candidate questions and the hypothesis space, we find that the bottleneck is generation, not judg
    
[^7]: 一个策略应对100个家庭任务：我们的2026年BEHAVIOR挑战赛解决方案

    One Policy for 100 Household Tasks: Our Solution for the 2026 BEHAVIOR Challenge

    [https://arxiv.org/abs/2611.14432](https://arxiv.org/abs/2611.14432)

    该团队基于2025年冠军方案π₀.₅构建单一VLA模型，仅用一个检查点完成2026年BEHAVIOR挑战赛全部100个家庭任务并获得第九名，核心改进在于延长训练时间、优化训练参数、加快训练步骤并减少梯度噪声。

    

    我们描述了在2026年BEHAVIOR挑战赛中的解决方案，该挑战赛要求在OmniGibson模拟器中完成100个长时程家庭任务。我们在最终排行榜上排名第九，是前十名中唯一的独立团队。解决方案的核心是一个单一的视觉-语言-动作（VLA）模型，仅用一个检查点即可应对全部100个任务，该模型构建于我们2025年获得第一名的解决方案之上，而后者基于π₀.₅。在与2025年相同的50个任务上，其Q值从0.260提升至0.371；在全部100个任务上得分为0.392。我们从2025年得出的主要结论是模型训练不足，因此2026年的大部分工作致力于更长时间地训练模型、优化部分训练参数、略微加快每个训练步骤的速度并减少梯度噪声：包括从50个任务向100个任务的迁移（从演示的技能标注构建语义阶段、新任务的任务和阶段嵌入从相似的旧任务热启动，以及匹配的21维动作空间）

    arXiv:2611.14432v1 Announce Type: cross  Abstract: We describe our solution for the 2026 BEHAVIOR Challenge: 100 long-horizon household tasks in the OmniGibson simulator. We placed 9th on the final leaderboard, the only independent team in the top 10. The core of the solution is a single vision-language-action (VLA) model, one checkpoint for all 100 tasks, built on our 1st-place 2025 solution, which is based on $\pi_{0.5}$. On the 50 tasks shared with 2025, its Q rises from 0.260 to 0.371; across all 100 tasks it scores 0.392. Our main conclusion from 2025 was that the model was undertrained, so most of the 2026 work went into training the model for longer, optimizing some training parameters, making each training step slightly faster and reducing the gradient noise: a migration from 50 to 100 tasks (semantic stages built from the demonstrations' skill annotations, task and stage embeddings of new tasks warm-started from similar old ones, and a 21-dimensional action space matching the 
    
[^8]: 两个社区，同一个问题：分支定界法与A*算法在资源受限项目调度中的对比

    Two Communities, One Problem: Branch-and-Bound vs. A* for Resource-Constrained Project Scheduling

    [https://arxiv.org/abs/2611.14426](https://arxiv.org/abs/2611.14426)

    本文以资源受限项目调度问题（RCPSP）为基准，在相同的下界和节点选择策略下，系统比较了运筹学的分支定界法与人工智能的A*搜索算法，并建立了混合整数线性规划、状态-动作和时间Petri网表示之间的联系，从而揭示两种搜索范式的根本性性能差异。

    

    分支定界法和A*分别是源于运筹学和人工智能领域的两种流行的精确搜索方法。尽管这两种方法共享基本的搜索原理，但它们探索搜索空间的不同方式可能导致显著不同的计算性能。我们以资源受限项目调度问题（RCPSP）作为示例问题来研究这些差异。我们首先建立了混合整数线性规划、状态-动作以及时间Petri网表示之间的关系。由于分支定界法和A*算法可以通过众多算法增强手段加以改进，我们聚焦于相对基础的实现方式。这使我们能够考察两种方法之间的根本性差异，而不是比较最先进求解器的性能。通过使用相同的下界和最佳优先节点选择策略，我们隔离了搜索机制本身的影响，特别是……

    arXiv:2611.14426v1 Announce Type: cross  Abstract: Branch-and-Bound (B&B) and $A^*$ are popular exact search methods originating from Operations Research and Artificial Intelligence, respectively. Although the two methods share fundamental search principles, their different ways of exploring the search space can lead to substantially different computational performance. We investigate these differences using the Resource-Constrained Project Scheduling Problem (RCPSP) as an exemplary problem setting. We first establish the relationships between mixed-integer linear programming, state-action, and timed Petri-net representations. Since B&B and $A^*$ admit numerous algorithmic enhancements, we focus on relatively basic implementations. This allows us to examine the fundamental differences between the methods rather than compare the performance of state-of-the-art solvers. By using the same lower bound and best-first node selection, we isolate the effects of the search mechanisms, particula
    
[^9]: 量化面向传感器数据应用的时间序列基础模型的采样频率不变性

    Quantifying the Sampling Frequency Invariance of Time-Series Foundation Models for Sensor Data Applications

    [https://arxiv.org/abs/2611.14425](https://arxiv.org/abs/2611.14425)

    该论文首次将采样频率不变性置于时间序列基础模型评估的核心，通过压力测试4个主流模型并提出FSI、FDI、CDI三个新指标，发现模型行为取决于信息带宽损失而非采样频率数值本身。

    

    时间序列基础模型（TSFMs）在传感器应用中的使用日益增多，而在这类应用中，采样频率（SF）会因传感器、设备和运行模式（例如省电模式）的不同而变化。理想情况下，只要采样频率仍保留任务所需的关键信息带宽，TSFM就应在此类采样频率变化中保持有效。本文对4个知名的时间序列基础模型（MOMENT、Chronos-2、Moirai-MoE和TimesFM）进行了压力测试，以量化它们的采样频率不变性，并研究其行为是否追踪由此产生的信息带宽损失，而非采样频率数值本身。我们的压力测试探测了3个维度：（i）嵌入表示的偏移，（ii）预测性能的退化，以及（iii）下游性能的退化，分别通过三个新提出的指标FSI、FDI和CDI进行量化。据我们所知，这是首个将采样频率不变性置于TSFM评估核心的研究。我们的结果刻画了：（i）采样频率不变性如何取决于裕度……

    arXiv:2611.14425v1 Announce Type: cross  Abstract: Time-Series Foundation Models (TSFMs) are increasingly used in sensor applications, where sampling frequency (SF) varies across sensors, devices, and operating modes (e.g., power saving). Ideally, TSFMs must remain effective during such SF changes, provided the SF retains task-critical bandwidth. In this paper, we stress-test 4 prominent TSFMs: MOMENT, Chronos-2, Moirai-MoE, and TimesFM, to quantify their SF-invariance and investigate whether their behavior tracks the resulting loss of information bandwidth rather than the SF value itself. Our stress-test probes 3 dimensions: (i) the shift in embeddings, (ii) the degradation in forecasting performance, and (iii) the degradation in downstream performance, quantified by three novel metrics: FSI, FDI, and CDI, respectively. To our knowledge, this is the first study to place SF-invariance at the core of TSFM evaluations. Our results characterize: (i) how SF-invariance depends on the margin
    
[^10]: 评估用于神经符号推理的LLM生成世界模型

    Evaluating LLM-Generated World Models for Neuro-Symbolic Reasoning

    [https://arxiv.org/abs/2611.14419](https://arxiv.org/abs/2611.14419)

    该研究通过一个涵盖44个场景的基准评估五个大语言模型生成的世界模型，发现尽管135个生成模型中有134个通过了严格的机械有效性审计，但其与参考模型的结构一致性平均F1仅为0.366，表明机械有效性并不足以保证世界模型忠实地表示预期的决策问题。

    

    形式化世界模型可以支持规划、验证和安全推理，但从开放式的自然语言情境中构建这些模型是困难的，尤其是在多个形式化方案都可能成立的多智能体场景中。我们研究了五个大语言模型（LLM）能否将此类世界模型生成为部分可观测随机博弈（POSG），以及机械有效性是否足以证明所得模型忠实地表示了预期的决策问题。我们引入了一个包含44个场景的基准，涵盖博弈、情境和文化类设置，并从三个维度评估生成的世界模型：机械有效性、相对于参考模型的结构一致性，以及语义标准。在三个实证轨道中，我们分析了410个首次生成的世界模型。在有参考模型的博弈轨道中，135个生成的世界模型中有134个通过了严格的机械审计，然而平均置换不变结构F1仅为0.366，这仅……

    arXiv:2611.14419v1 Announce Type: new  Abstract: Formal world models can support planning, verification, and safety reasoning, but constructing them from open-ended natural-language situations is difficult, particularly in multi-agent settings where several formalizations may be plausible. We study whether five LLMs can generate such world models as partially observable stochastic games and whether mechanical validity is sufficient to establish that the resulting model faithfully represents the intended decision problem. We introduce a benchmark of 44 scenarios spanning game, situational, and cultural settings and evaluate generated world models along three dimensions: mechanical validity, reference-relative structural agreement, and semantic criteria. Across three empirical tracks, we analyze 410 first-pass generations. In the reference-backed game track, 134/135 generated world models pass a strict mechanical audit, yet mean permutation-invariant structural F1 is only 0.366, with jus
    
[^11]: 按需精度：在小型本地模型的不确定解码步骤中动态提升权重位宽

    Precision on Demand: Escalating Weight Bits at Uncertain Decoding Steps in a Small Local Model

    [https://arxiv.org/abs/2611.14418](https://arxiv.org/abs/2611.14418)

    提出按需精度的解码方法：小型本地模型平时用高4位权重解码，仅在输出分布不确定（高熵或前两名差距小）时升级到8位重算该步骤，以约87%的权重读取开销取得接近全高精度的准确率。

    

    本地解码通常在整个生成过程中固定使用单一权重精度。我们研究了一种不同的分配方式：存储8位权重，平时仅用其高4位进行解码，仅当低精度输出分布看起来不确定时（熵超过阈值或前两名概率差距过小），才用完整8位权重重新执行该步骤。量化采用模拟方式，模型为Qwen3-0.6B，基准测试为GSM8K，成本按每个token读取的权重字节数计算。在36724个重放步骤上，低精度通道的熵对argmax翻转具有0.93的AUC判别排序能力。在15%的升级预算下，熵门控可捕获78%的翻转，间距门控可捕获94%，而同等升级率的随机门控预期仅能捕获15%（该值为计算得到的期望，并非实际运行结果）。在150个实际问题中，全高精度得分为64.0%，全低精度为55.3%（McNemar p=0.053）。一个升级率为16%的熵门控在仅消耗全高精度0.87倍权重字节的情况下得分62.7%。其配对差异相对全高精度为-1.3分（-8.0至+5.3），相对全低精度为+7.3分（+0.0至+14.7）。

    arXiv:2611.14418v1 Announce Type: new  Abstract: Local decoding usually fixes one weight precision for the whole generation. We study a different allocation: store 8-bit weights, decode with their top 4 bits, and redo a step at 8 bits only when the low-precision output distribution looks uncertain (entropy above a threshold or a small top-two margin). Quantization is simulated, the model is Qwen3-0.6B, and the benchmark is GSM8K. Cost is counted in weight bytes read per token. On 36724 replayed steps the entropy of the low pass ranks argmax flips with AUC 0.93. At a 15% escalation budget it catches 78% of flips and the margin gate 94%, against the 15% expected for a random gate of equal rate (a computed expectation, not a run). In 150 live problems all-high scores 64.0% and all-low 55.3% (McNemar p=0.053). An entropy gate with a 16% escalation rate scores 62.7% at 0.87 of the all-high weight bytes. Its paired difference is -1.3 (-8.0 to +5.3) points to all-high and +7.3 (+0.0 to +14.7)
    
[^12]: 比特还是排列？在小型本地语言模型中将固定推理预算用于权重精度还是选项顺序平均

    Bits or permutations? Spending a fixed inference budget on weight precision or on option-order averaging in small local language models

    [https://arxiv.org/abs/2611.14413](https://arxiv.org/abs/2611.14413)

    本文量化了小型本地语言模型中权重量化精度对答案选项顺序偏差的影响，发现6比特量化不损失准确率而4比特量化会显著加剧答案槽位偏斜，并据此比较了固定推理预算下提升权重精度与选项顺序平均两种方案的性价比。

    

    本地语言模型可以通过以更少的比特数存储权重来节省内存，也可以通过对少量选项重排序的结果取平均来降低对答案选项顺序的敏感性。我们探讨前者是否会使后者变差，以及在固定预算下哪种方式是更经济的花费方式。在Qwen3-0.6B和Qwen3-1.7B上使用模拟的就近取整权重量化（8、6、4和3比特，以fp16为参照），读取ARC-Easy、ARC-Challenge、MMLU和HellaSwag（各300题）上字母A到D的logits，结果显示：6比特相比fp16没有损失准确率（反而分别高出+1.8（+1.0到+2.5）和+0.9（+0.3到+1.5）分，这一点我们无法解释），而8比特使准确率变化-0.7（-1.1到-0.3）和-0.4（-0.7到-0.2）分。在4比特时，答案槽位偏斜从0.15升至0.53（0.6B模型），从0.07升至0.12（1.7B模型）；在相同权重信噪比下施加高斯噪声得到0.15和0.37（两次抽样）以及0.19（一次抽样），因此该对照实验对于取整是否增加了槽位（偏斜）尚无定论。

    arXiv:2611.14413v1 Announce Type: new  Abstract: Local language models can save memory by storing weights in fewer bits, and they can reduce sensitivity to the order of answer options by averaging a few reorderings. We ask whether the first makes the second worse, and which is the cheaper way to spend a fixed budget. On Qwen3-0.6B and Qwen3-1.7B with simulated round-to-nearest weight quantization (8, 6, 4 and 3 bits, fp16 reference), read from the logits of the letters A to D on ARC-Easy, ARC-Challenge, MMLU and HellaSwag (300 items each), 6 bits lose no accuracy against fp16 (they score +1.8 (+1.0 to +2.5) and +0.9 (+0.3 to +1.5) points higher, which we do not explain), and 8 bits change it by -0.7 (-1.1 to -0.3) and -0.4 (-0.7 to -0.2) points. At 4 bits the answer-slot skew rises from 0.15 to 0.53 (0.6B) and from 0.07 to 0.12 (1.7B); Gaussian noise at the same weight SNR gives 0.15 and 0.37 (two draws) and 0.19 (one draw), so the control is inconclusive on whether rounding adds slot 
    
[^13]: 深度神经网络模型的因果概念解释

    Causal Concept Explanations for Deep Neural Models

    [https://arxiv.org/abs/2611.14407](https://arxiv.org/abs/2611.14407)

    该论文提出 CCW-Net，首次将干预性中介分析作为训练目标，使概念解释与模型输出之间建立真正的因果关系而非仅是相关性，并提供量化每个概念解释对模型输出因果贡献的评估工具。

    

    神经网络模型中基于概念的解释对于自主系统的开发、认证和推广应用至关重要。然而，现有的可解释性方法仍停留在相关性层面。由于缺乏指导和量化解释之间因果关系的方法，所提供的概念解释可能既不是所观测输出的必要条件，也不是其充分条件，从而在看似正确的同时产生误导。我们提出了因果概念包装网络（Causal Concept Wrapper Network, CCW-Net），这是首次将源自经济学、医学和流行病学等领域的干预性中介分析用作训练目标，使基于概念的解释的因果效应与模型输出保持一致。我们还进一步将中介分析用作评估工具，针对每个概念标签组合，量化任意模型输出中有多少可因果归因于每个概念解释。作为训练目标，CCW-Net 塑造模型的概念表示，使其既能……（原文摘要在此处截断）

    arXiv:2611.14407v1 Announce Type: cross  Abstract: Concept-based explanations in neural models are important for the development, certification, and adoption of autonomous systems. Yet, interpretability remains correlational. Without methods to guide and quantify causal relationships between explanations, a concept explanation can be provided without being necessary or sufficient for the observed output thereby appearing correct while misleading. We introduce Causal Concept Wrapper Network (CCW-Net), the first use of interventional mediation analysis from fields such as economics, medicine, and epidemiology as a training objective to align causal effects of concept-based explanations with a model's output. We further apply mediation analysis as an evaluation tool to quantify, per concept label combination, how much of any model's output is causally attributable to each concept explanation. As a training objective, CCW-Net shapes a model's concept representations such that they are both
    
[^14]: 模糊块边界：面向高效扩散语言模型的重叠块去噪

    Blurring the Block Boundaries: Overlapping Block Denoising for Efficient Diffusion Language Models

    [https://arxiv.org/abs/2611.14387](https://arxiv.org/abs/2611.14387)

    本文提出B3方法，通过在共享前向传播中并行精炼两个块来模糊块边界，实现无需训练和超参数调整的扩散语言模型加速，同时保持生成质量。

    

    扩散语言模型（DLMs）已成为自回归模型在延迟敏感任务中的有效替代方案，能够实现更快的token生成。最近的进展采用具有自编辑能力的块扩散方法，其中每个块在进入下一个块之前迭代地精炼token。然而，每个块的速度受限于其最慢的token，且下一个块必须等待它完成后才能开始。我们提出了模糊块边界（B3），这是一种新颖的方法，可以在共享的前向传播中并行精炼两个块。B3能够动态地自我校准，只有当模型准备好开始生成下一个块时才会开启它。这是一种无需训练、无需超参数调整的加速方法，可以无缝集成到任何自编辑块扩散模型中。在最先进的扩散语言模型上、跨越不同任务和难度级别的10个多样化数据集上进行评估，B3在实现同等生成质量的同时获得了11.5-1...（原文摘要截断）

    arXiv:2611.14387v1 Announce Type: cross  Abstract: Diffusion Language Models (DLMs) have emerged as effective alternatives to autoregressive models for latency-critical tasks, enabling faster token generation. Recent advances employ block diffusion with self-editing capabilities, where each block refines tokens iteratively before advancing to the next one. Yet each block is only as fast as its slowest token, and the next block cannot start before it. We suggest Blurring the Block Boundaries (B3), a novel method with which two blocks are refined in parallel within a shared forward pass. B3 calibrates itself on the fly, opening the next block only once the model is ready to start generating it. It is a training-free and hyperparameter-free acceleration method which seamlessly integrates within any self-editing block diffusion model. Evaluated on state-of-the-art DLMs across 10 diverse datasets spanning varied tasks and difficulty levels, B3 results in on par generation quality and 11.5-1
    
[^15]: 机器学习的能量视角

    An Energy-Based Perspective on Machine Learning

    [https://arxiv.org/abs/2611.14383](https://arxiv.org/abs/2611.14383)

    本教程以能量模型（EBM）为统一视角，通过能量函数、推断过程和学习方法三大核心组件，将判别模型、GAN、扩散模型、自回归模型等各类机器学习方法统一在一个共同的对比学习框架之下。

    

    能量模型（EBM）通过一个标量能量函数来定义分布，$p_\theta(x)\propto\exp(-E_\theta(x))$，通常被视为众多模型家族中的一员。本教程有着更宏大的目标：不仅介绍EBM，还将其作为一个统一的视角，把机器学习的不同方面连接起来。任何EBM都由三个选择确定：能量函数，它决定模型表示什么；推断过程，它从能量中采样或对能量进行优化；以及学习方法，它使能量拟合数据。我们首先阐述这些组件，包括采样、变分推断，以及通过最大似然、噪声对比估计、伪似然、得分匹配和对抗训练进行学习的方法，并证明这些学习方法共享同一个对比结构。随后我们展示判别模型、生成模型（如GAN、归一化流、扩散模型和自回归模型）、表示学习等均可在此框架下统一理解（原文此处截断）。

    arXiv:2611.14383v1 Announce Type: new  Abstract: Energy-based models (EBMs) define a distribution through a scalar energy, $p_\theta(x)\propto\exp(-E_\theta(x))$, and are usually presented as one model family among many. This tutorial has a broader goal: beyond introducing EBMs, it uses them as a single lens that connects the different aspects of machine learning. Any EBM is specified by three choices, the energy, which determines what the model represents; the inference procedure, which samples from or optimizes the energy; and the learning method, which fits the energy to data. We first develop these components, including sampling, variational inference, and learning by maximum likelihood, noise contrastive estimation, pseudolikelihood, score matching, and adversarial training, and show that the learning methods share one contrastive structure. We then show that discriminative models, generative models such as GANs, normalizing flows, diffusion and autoregressive models, representati
    
[^16]: 大语言模型中“思考式学习”机制的认知映射

    Cognitive Mapping of Learning by Thinking Mechanisms in Large Language Models

    [https://arxiv.org/abs/2611.14382](https://arxiv.org/abs/2611.14382)

    本文提出认知式思考式学习框架，将反思性推理、合成记忆形成、内部模拟和概念自组织四种认知机制映射到大语言模型的计算过程，并论证思维链、自我反思等代表性方法可在功能上近似这些机制，从而无需额外标注数据即可提升任务性能。

    

    大语言模型（LLM）日益展现出类似人类“思考式学习”的能力，即通过解释、推理等内部过程获取新的洞见，而无需新的外部输入。然而，对这些能力背后的认知机制及其局限性的系统性阐述仍不完善。本文提出了一个认知式思考式学习框架，将四种机制（反思性推理、合成记忆形成、内部模拟与概念自组织）映射到大语言模型的计算过程上。通过分析链式思维提示、自洽性、自我反思、思维树搜索、Self-Instruct 以及自授推理等代表性方法，分析认为所选取的方法可以在功能上近似这些机制的某些方面，并能在无需额外标注样本的情况下提升某些任务的性能，尽管这并不总是……

    arXiv:2611.14382v1 Announce Type: new  Abstract: Large language models (LLMs) increasingly exhibit capabilities that resemble human Learning by Thinking (LbT), the acquisition of new insight through internal processes such as explanation and reasoning, without new external input. A systematic account of the cognitive mechanisms behind these capabilities, and of their limits, remains underdeveloped. This paper proposes a Cognitive LbT Framework that maps four mechanisms (Reflective Reasoning, Synthetic Memory Formation, Internal Simulation, and Conceptual Self-Organization) onto computational processes in LLMs. Drawing on representative methods, including chain-of-thought prompting, self-consistency, self-reflection, Tree-of-Thoughts search, Self-Instruct, and self-taught reasoning, the analysis argues that selected methods can functionally approximate aspects of these mechanisms and can improve performance on some tasks without additional labeled examples, although this does not always
    
[^17]: EngineAgent：具备推理栈层级感知与硬件能力的端侧LLM推理引擎智能体化

    EngineAgent: Agentizing On-Device LLM Inference Engines with Inference Stack Hierarchy Awareness and Hardware Abilities

    [https://arxiv.org/abs/2611.14356](https://arxiv.org/abs/2611.14356)

    EngineAgent通过构建层级感知的引擎索引并利用已支持模型预先验证后端工作流程，将可复用的引擎与硬件知识从模型特定实现中分离，从而自动化端侧LLM推理引擎的新模型适配过程。

    

    快速演进的LLM架构与异构的移动硬件，使得在端侧推理引擎中支持新模型成为一项反复出现且成本高昂的系统挑战。每个模型可能需要在转换、图构建、算子以及硬件后端等多个层面进行协同修改。现有的编程智能体往往忽略这些跨层依赖关系，并且难以诊断主机—设备执行路径上的故障。我们提出了EngineAgent，一个用于自动化端侧LLM模型适配的智能体系统。其核心洞察是将可复用的引擎与硬件知识从模型特定的实现中分离出来。在模型适配之前，EngineAgent会构建一个层级感知的引擎索引，以捕捉引擎抽象层与跨层依赖关系，并利用已支持的模型预先验证后端特定的构建、执行和调试工作流程。对于每个新模型，构建器（builder）利用这些知识来协调跨层变更。

    arXiv:2611.14356v1 Announce Type: cross  Abstract: Rapidly evolving LLM architectures and heterogeneous mobile hardware make supporting new models in on-device inference engines a recurring and costly system challenge. Each model may require coordinated changes across conversion, graph construction, operators, and hardware backends. Existing coding agents often miss these cross-layer dependencies and struggle to diagnose failures across host--device execution paths. We present EngineAgent, an agentic system for automating on-device LLM model enablement. Its key insight is to separate reusable engine and hardware knowledge from model-specific implementation. Before enablement, EngineAgent constructs a hierarchy-aware engine index that captures engine abstraction layers and cross-layer dependencies, and pre-validates backend-specific build, execution, and debugging workflows using already supported models. For each new model, a builder uses this knowledge to coordinate cross-layer change
    
[^18]: CLADE：检索增强语言模型中基于不足证据的类别级回答

    CLADE: Category-Level Answering from Deficient Evidence in Retrieval-Augmented Language Models

    [https://arxiv.org/abs/2611.14355](https://arxiv.org/abs/2611.14355)

    该论文揭示了语言模型在阅读仅包含粗粒度类别信息（如疾病家族而非具体疾病）的检索段落时会直接作答而非弃答的“Clade”现象，并提出这类“可解决的不足”可通过领域本体中的候选答案列表来刻画与处理。

    

    检索增强系统所检索到的段落往往准确且相关，但其陈述答案的粒度比问题所需的更粗。例如，问题可能询问某药物治疗的特定疾病（家族性高脂血症），而检索到的段落仅提及更宽泛的疾病家族（高脂血症）。先前关于上下文充分性的研究只是简单地将这类粗粒度段落视为不充分。我们认为它们构成了一个独特的情形，我们称之为“可解决的不足”：仅凭文本无法确定细粒度的答案，但其中提到的更宽泛类别可以在领域本体中映射到一个较短的候选答案列表。当语言模型阅读这些仅包含家族层面信息的粗粒度段落时，它们会表现出一种我们称为 Clade（基于不足证据的类别级回答）的现象：模型会直接作答，而不是弃答或进一步检索。在来自 PrimeKG 的生物医学问题上，仅包含家族信息的段落使弃答率降低了（原文摘要在此处截断）

    arXiv:2611.14355v1 Announce Type: new  Abstract: Retrieval-augmented systems often retrieve passages that are accurate and relevant but state the answer at a coarser level of granularity than the question requires. For example, a question may ask which specific disease a drug treats (familial hyperlipidemia), while the retrieved passage mentions the broader disease family (hyperlipidemia). Prior work on context sufficiency treats such coarse passages simply as insufficient. We argue that they form a distinct case, which we call resolvable insufficiency: text alone cannot settle the fine-grained answer, but the broader category it mentions maps to a short list of candidate answers in a domain ontology. When language models read these coarse, family-only passages, they exhibit a phenomenon we call Clade (Category-Level Answering from Deficient Evidence): they answer instead of abstaining or searching further. On biomedical questions from PrimeKG, family-only passages reduce abstention by
    
[^19]: AI并没有变笨，是我们变得更挑剔了：大型语言模型人类评估标准的不断上升

    AI Isn't Getting Dumber, We're Getting Pickier: Rising Standards in Human Evaluation of Large Language Models

    [https://arxiv.org/abs/2611.14333](https://arxiv.org/abs/2611.14333)

    该研究通过分析Chatbot Arena上126万余张针对固定版本模型的盲投票数据发现，“两个都不好”的投票比例持续显著上升，表明AI模型并非在变差，而是人类评估者的标准在不断提高。

    

    用户经常抱怨像ChatGPT这样的AI助手正在变差。某些版本确实会变化，但人类评估留下了第二种未经检验的可能性：评判者本身可能发生了变化。我们在Chatbot Arena平台上区分这两种因素——这是一个公开平台，人们在这里比较两个匿名大型语言模型（LLM）的回答，并选择更好的一个或判定两者都不好。我们研究了2023年5月至2024年8月期间针对版本固定模型对投出的1,260,967张盲选票，这些模型包括带有日期标记的厂商快照模型和固定开源权重的模型。对于同一对未发生变化的模型，判定“两个都不好”的投票比例每月上升1.71个百分点（平均值为17.1%），而在2024年每月上升1.91个百分点。在2023年和2024年均被频繁观察到的十四对模型中，这一比例在每一对中都出现了上升，从15.5%升至23.3%。模型排名并未呈现趋势：2024年，更强模型在决定性投票中的份额保持平稳。这种上升在同一投票者内部同样成立（原文在此处被截断）。

    arXiv:2611.14333v1 Announce Type: cross  Abstract: Users often complain that AI assistants such as ChatGPT are getting worse. Some versions do change, but human evaluation leaves a second possibility untested: the people judging may change. We separate the two on Chatbot Arena, a public platform where people compare answers from two anonymous large language models (LLMs) and pick the better one or say both are bad. We study 1,260,967 blind votes cast from May 2023 to August 2024 on pairs of models whose versions are fixed: dated vendor snapshots and models with fixed open weights. For the same pair of unchanged models, the share of votes saying "both are bad" rises by 1.71 percentage points a month on a mean of 17.1%, and by 1.91 points a month in 2024. In fourteen pairs observed often in both 2023 and 2024, it rose in every one, from 15.5% to 23.3%. The ranking does not trend: in 2024, the stronger model's share of decisive votes is flat. The rise holds within the same voter and on th
    
[^20]: 前向值离散化：无需高阶近似的更快随机扩散采样

    Forward-Value Discretization: Faster Stochastic Diffusion Sampling Without High-Order Approximation

    [https://arxiv.org/abs/2611.14328](https://arxiv.org/abs/2611.14328)

    该论文证明了高阶近似无法提升随机扩散采样的收敛阶数，并提出免训练的前向值离散化方法（F-DPMSolver），通过利用先前预测构建的前瞻估计，在低神经函数评估次数下实现更快的采样并提升样本质量。

    

    随机扩散采样器通常需要比确定性采样器多得多的神经函数评估次数（NFE）才能达到相当的样本质量。尽管高阶离散化能够加速确定性采样，但其对随机采样的益处尚不明确。我们建立了匹配的上界与最坏情况下的下界，表明即使使用精确的分数函数，采用高阶近似的随机采样器仍然只具有一阶收敛性。因此，提高近似阶数通常并不能改善收敛阶数。受此局限性的启发，我们提出了前向值离散化，这是一种免训练的框架，它在由先前预测构建的前瞻估计处评估数据/噪声预测模型。对于确定性采样，由此得到的 Forward DPMSolver（F-DPMSolver）通过其基于 DDIM 的变体，在低 NFE 预算下提升了样本质量，尽管仍保持一阶……

    arXiv:2611.14328v1 Announce Type: cross  Abstract: Stochastic diffusion samplers often require substantially more neural function evaluations (NFEs) than deterministic samplers to achieve comparable sample quality. Although higher-order discretization accelerates deterministic sampling, its benefits for stochastic sampling remains unclear. We establish matching upper and worst-case lower bounds showing that stochastic samplers with higher-order approximations remain first-order convergent, even with exact score functions. Thus, increasing the approximation order does not generally improve the convergence order. Motivated by this limitation, we introduce forward-value discretization, a training-free framework that evaluates the data/noise prediction model at a lookahead estimate constructed from previous predictions. For deterministic sampling, the resulting Forward DPMSolver (F-DPMSolver), improves sample quality at low NFE budgets through its DDIM-based variant, despite remaining firs
    
[^21]: 智能体何时应该记忆？面向LLM智能体的证伪门控自我进化

    When Should Agents Remember? Falsification-Gated Self-Evolution for LLM Agents

    [https://arxiv.org/abs/2611.14326](https://arxiv.org/abs/2611.14326)

    提出证伪门控自我进化（FGSE）框架，将经验教训表示为结构化假设，仅在临时状态中经受证伪、迁移与回归探测，只有通过收益与风险双重验证的行为更新才被写入智能体的持久记忆，从而显著提升LLM智能体自我进化的可靠性与安全性。

    

    LLM智能体日益运行在长时程、验证器丰富的环境中，但当前的自我改进机制往往将回合经验中总结的教训直接写入记忆，使得局部成功成为可复用且安全知识的不可靠信号。本文解决的核心问题是：何时应将来自经验的 行为更新纳入智能体的持久状态。我们提出证伪门控自我进化（FGSE），这是一个以验证器为基础的框架，将每个候选经验教训表示为结构化假设，包含明确的前提条件、行为改变、预期效果和验证器。FGSE仅在临时状态中应用该假设，在目标迁移、证伪和归档回归探测上进行测试，然后根据测得的收益和风险来提交、细化或拒绝该假设。在网页、应用、工具使用和长期记忆基准测试中，FGSE取得了强大的任务性能，包括39.8的WebArena成功率。

    arXiv:2611.14326v1 Announce Type: new  Abstract: LLM agents increasingly operate in long-horizon, verifier-rich environments, but current self-improvement mechanisms often write episode-derived lessons directly into memory, making local success an unreliable signal for reusable and safe knowledge. This paper addresses the core question of when an experience-derived behavioral update should be admitted into persistent agent state. We propose Falsification-Gated Self-Evolution (FGSE), a verifier-grounded framework that represents each candidate lesson as a structured hypothesis with an explicit precondition, behavior change, expected effect, and verifier. FGSE applies the hypothesis only in a temporary state, tests it on target-transfer, falsification, and archived-regression probes, and then commits, refines, or rejects it according to measured gain and risk. Across web, app, tool-use, and long-term memory benchmarks, FGSE achieves strong task performance, including 39.8 WebArena succes
    
[^22]: 不要只相信你的记忆：证据关联记忆减少LLM智能体中的记忆诱导退化

    Don't Just Trust Your Memory: Evidence-Linked Memory Reduces Memory-Induced Degradation in LLM Agents

    [https://arxiv.org/abs/2611.14320](https://arxiv.org/abs/2611.14320)

    提出WikiTrace证据关联记忆系统，通过将每条经验教训与其来源轨迹中的证据相链接，使LLM智能体能够核实记忆内容，从而有效减少记忆诱导的性能退化。

    

    外部记忆使LLM智能体能够将过去任务的轨迹提炼成简洁的经验教训，从而从经验中学习并改进新任务的表现。然而，记忆也可能导致“记忆诱导退化”，即智能体在没有记忆时本可解决的任务上反而失败，这通常是因为它遵循了一条不正确的经验教训，或者误用、误读了一条有效的教训。其原因在于，经验教训是从仅标注了任务成功或失败的轨迹中推断出来的，但通常在不附带其来源轨迹的情况下提供，使智能体无法对其进行核实。为解决这一问题，我们提出了WikiTrace，一种证据关联记忆系统，旨在让智能体能够对照其原始轨迹核实每条经验教训。具体而言，基于“轨迹中往往仍保留着发现错误教训所需证据”这一发现，WikiTrace将每条经验教训存储为一条简洁的洞见，并将其链接到该轨迹成功或失败原因的摘要以及相应证据（摘要在此处被截断）。

    arXiv:2611.14320v1 Announce Type: new  Abstract: External memory enables LLM agents to improve from experience by distilling trajectories of past tasks into concise lessons for new tasks. However, memory can also lead to memory-induced degradation, where an agent fails on tasks it would solve without memory, often because it follows a lesson that is incorrect, or misapplies or misreads a valid one. This is because lessons are inferred from trajectories labeled only with task success or failure, yet are typically provided without the trajectories they came from, leaving the agent no way to check them. To address this, we introduce WikiTrace, an evidence-linked memory system designed to let agents check each lesson against its original trajectory. Specifically, motivated by the finding that a trajectory often still holds the evidence needed to catch an erroneous lesson, WikiTrace stores each lesson as a concise insight linked to a summary of why its trajectory succeeded or failed and to 
    
[^23]: 保留已获得正确答案的多模态弃答机制

    Multimodal Abstention That Preserves Acquired Correct Answers

    [https://arxiv.org/abs/2611.14311](https://arxiv.org/abs/2611.14311)

    提出ReFrain多模态弃答方法，在提升对不可回答问题正确弃答能力的同时，保证任务适配阶段获得的正确答案零损失。

    

    为经过任务适配的多模态模型添加弃答功能，可能会丢弃正确答案，包括适配过程中获得的能力。我们将弃答修复问题形式化为：在“新损失的正确答案”的预算约束下，最大化对不可回答问题的正确弃答率。我们提出ReFrain（无回归弃答），该方法结合了基于首token logits的线性探针、在留出图像集群上的阈值校准，以及一个确定性否决机制——用显示的“以上都不是”选项替换被拒绝的输出，同时保留所有其他响应。“无回归”指在零预算目标下，校准集上的正确答案损失为零。主要对比中的所有弃答方法均从同一个回答SFT检查点出发；我们分别跟踪损失的正确答案、修复的错误以及保留的SFT收益。在Qwen2.5-VL-3B上，ReFrain在零预算下完整保留了全部57个SFT获得的答案且零损失，同时将正确弃答率提升了7.30个百分点。

    arXiv:2611.14311v1 Announce Type: new  Abstract: Adding abstention to a task-adapted multimodal model can discard correct answers, including capabilities acquired during adaptation. We formulate abstention repair as maximizing correct abstention on unanswerable questions under a budget on newly lost correct answers. We propose ReFrain (Regression-Free Abstention), which combines a linear probe on first-token logits, threshold calibration on held-out image clusters, and a deterministic veto that replaces rejected outputs with the displayed "none of the above" option while preserving all other responses. "Regression-free" denotes zero loss on the calibration set at the zero-budget target. All abstention methods in our main comparison start from the same answering-SFT checkpoint; we separately track lost correct answers, repaired errors, and retained SFT gains. On Qwen2.5-VL-3B, ReFrain at zero budget retains all 57 SFT-gained answers with no losses and improves correct abstention by 7.30
    
[^24]: LM-SimBench：面向大语言模型训练仿真的统一框架、数据集与基准

    LM-SimBench: A Unified Framework, Dataset, and Benchmark for Large Language Model Training Simulation

    [https://arxiv.org/abs/2611.14285](https://arxiv.org/abs/2611.14285)

    该论文提出LM-SimBench，通过统一的仿真分解框架、基于昇腾NPU集群的大规模性能剖析数据集以及标准化基准，首次实现了对不同大语言模型训练仿真器的公平、系统化评估。

    

    通过分布式训练实际运行来评估候选并行化策略，使得在大规模场景下的策略选择成本高得令人望而却步。性能仿真提供了一种成本更低的替代方案，然而现有仿真器之间的受控比较仍然困难，因为它们在建模的训练阶段以及所评估的工作负载上各不相同。我们提出了LM-SimBench，这是一个用于评估大语言模型训练仿真器的统一框架、性能剖析数据集和基准。该框架将仿真过程分解为从模型编译到调度层聚合的各个功能阶段，将执行表示和成本估计器与可衡量的评估目标联系起来。遵循该框架，我们在昇腾（Ascend）神经处理单元（NPU）集群上对来自10个稠密及混合专家（MoE）大语言模型家族的23种模型设置进行了性能剖析。所得数据集包含348种可行配置，这些配置在五个并行维度上共同变化。

    arXiv:2611.14285v1 Announce Type: new  Abstract: Evaluating candidate parallelization strategies through distributed training runs makes strategy selection prohibitively expensive at scale. Performance simulation offers a lower cost alternative, yet controlled comparisons among existing simulators remain difficult because they differ in the training stages they model and the workloads on which they are evaluated. We introduce LM-SimBench, a unified framework, profiling dataset, and benchmark for evaluating LLM training simulators. The framework decomposes simulation into functional stages from model compilation to schedule level aggregation, connecting execution representations and cost estimators to measurable evaluation targets. Following this framework, we profile 23 model settings from 10 dense and Mixture-of-Experts (MoE) LLM families on an Ascend Neural Processing Unit (NPU) cluster. The resulting dataset contains 348 feasible configurations that jointly vary five parallelism dim
    
[^25]: 测量记录助手的96种方法：医疗领域环境式AI记录助手评估的碎片化现状

    Ninety-Six Ways to Measure a Scribe: Fragmented Evaluation of Ambient Scribes in Healthcare

    [https://arxiv.org/abs/2611.14284](https://arxiv.org/abs/2611.14284)

    该研究系统分析了54项医疗环境式AI记录助手的评估研究，发现评估维度和指标高度碎片化——共十个维度、40种指标、96种操作化方式，其中36%仅被单一研究使用，且缺乏标准化评估协议，导致各研究结果难以比较、证据难以综合。

    

    环境式人工智能记录助手已迅速从实验性试点项目发展到整个医疗系统的全面部署，有望减轻文档记录负担并改善临床医生的使用体验。尽管针对这些工具的评估研究迅速增多，但目前尚无公认的评估方案或标准化指标来衡量其性能，导致各研究之间的结果难以比较，证据也难以进行累积性综合。通过分析54项近期的评估研究，我们识别出十个评估维度和40种不同的指标，并以96种不同的方式进行操作化——其中35种（占36%）仅被单一研究使用。现有评估主要集中于文档记录效率和时间，54项研究中有44项对此进行了测量。其他维度出现的频率大致相近（每个维度16至24项研究），其中安全性和临床结果的评估一致性最低。而对更广泛医疗体系产生的连锁影响，例如对患者诊疗结局的影响……（原文摘要在此处截断）

    arXiv:2611.14284v1 Announce Type: cross  Abstract: Ambient artificial intelligence (AI) scribes have moved rapidly from experimental pilots to enterprise-wide deployment across health systems, promising reduced documentation burden and improved clinician experience. While research evaluating these tools has expanded quickly, there are no agreed-upon protocols or standardized metrics for assessing their performance, making it difficult to compare findings or cumulatively synthesize evidence across studies. Across 54 recent evaluation studies, we identify ten evaluation dimensions and 40 distinct metrics, operationalized in 96 different ways-35 of them (36%) used by only a single study. Assessment is dominated by documentation efficiency and time, measured in 44 of 54 studies. The other dimensions appear at broadly similar rates (16-24 studies each), and safety and clinical outcome least consistently of all. Ripple effects on the broader care system, such as impact on patients' care outc
    
[^26]: 通过设计的训练数据实现抗多普勒效应的鲁棒信道估计神经网络

    Achieving Robust Channel Estimation Neural Networks on Doppler effects by Designed Training Data

    [https://arxiv.org/abs/2611.14267](https://arxiv.org/abs/2611.14267)

    本文提出了一套合成训练数据的设计准则，通过精心设计的训练数据使信道估计神经网络能够泛化到未见过的时变信道，从而在具有强多普勒效应的高速场景下实现鲁棒信道估计，避免了昂贵且延迟高的在线训练。

    

    信道估计在无线通信系统中至关重要。许多现有工作使用相同或相似的信道来训练和测试其模型。然而，实际信道是无限多样且时变的，而数据驱动算法在新颖的、先前未见过的信道上往往会出现性能退化。这阻碍了神经网络在信道估计中的实际应用，尤其是在以信道特征快速变化和极高多普勒频移为特征的高速场景中。在线训练可以对模型进行微调以适应新信道，但需要大量计算资源，并会给物理层设备带来显著延迟。此外，由于可能的信道有无限多种，在线训练也无法穷尽覆盖所有可能的信道。因此，我们研究了神经网络对无线信道的泛化能力。本文提出了生成合成训练数据的设计准则……（原文摘要在此处截断）

    arXiv:2611.14267v1 Announce Type: cross  Abstract: Channel estimation is crucial in wireless communication systems. Many existing works train and test their models using the same or similar channel. However, practical channels are infinitely diverse and time-variant while data-driven algorithms often degrade on new and previously unseen channels. This prohibits the practical implementation of neural networks for channel estimation, particularly in high speed scenarios characterized by rapidly channel profile changes and very high Doppler shifts. Online training can fine-tune models to adapt to new channels, but requires high computation resource and incurs significant latency for physical-layer devices. Moreover, online training cannot exhaustively cover all possible channels because there are infinitely many possible channels. Therefore, we investigate the ability of neural networks to generalize to wireless channels. This paper proposes design criteria to generate synthetic training 
    
[^27]: 脉冲神经网络强化学习中的循环模块化结构

    Recurrent Modular Structure in Spiking Neural Network Reinforcement Learning

    [https://arxiv.org/abs/2611.14262](https://arxiv.org/abs/2611.14262)

    提出群落循环脉冲神经网络，将循环网络划分为具有固定角色的群落（控制器、动作和潜在状态群落），并结合局部三因子学习规则及新提出的Map-STDP，使脉冲强化学习网络结构更接近生物皮层微回路。

    

    生物神经回路既是模块化的又是循环的，密集的群落之间稀疏地相互连接。用于强化学习的脉冲神经网络（SNN）通常是前馈式的，如何在循环SNN中获得功能模块、以及这些模块能带来什么益处，目前尚不清楚。在这项工作中，我们引入了一类脉冲网络，称为群落循环SNN（CoRe SNNs）：这是一种被划分为具有固定角色的群落的循环网络。控制器群落接收刺激，每个动作都有属于自己的群落，而潜在群落则携带内部状态。脉冲活动因此成为群落之间的一次游走，动作即游走最终到达的群落。所有学习都是局部的三因子学习，并且这些循环群落比前馈actor更接近皮层微回路。我们比较了获得这些群落的三种机制：动态模块化，其中我们提出的一种新学习规则Map-STDP鼓励……

    arXiv:2611.14262v1 Announce Type: new  Abstract: Biological neural circuits are both modular and recurrent, with dense communities sparsely linked to one another. Spiking neural networks (SNNs) for reinforcement learning are usually feed-forward, and it is unclear how to obtain functional modules in a recurrent SNN or what they buy. In this work, we introduce a class of spiking networks, called Community Recurrent SNNs (CoRe SNNs): recurrent networks partitioned into communities with fixed roles. A controller community receives the stimulus, each action has its own community, and latent communities carry internal state. Spiking activity is then a walk between communities, and the action is the community the walk reaches. All learning is local and three-factor, and the recurrent communities are closer to cortical microcircuits than a feed-forward actor. We compare three regimes for obtaining the communities: dynamic modularity, in which a new learning rule we introduce, Map-STDP, encour
    
[^28]: Lilo：一个面向儿童数字交流的多智能体对话沙盒

    Lilo: A Multi-Agent Conversational Sandbox for Children's Digital Communication

    [https://arxiv.org/abs/2611.14260](https://arxiv.org/abs/2611.14260)

    Lilo是一个多智能体对话沙盒系统，让儿童在模拟的群聊冲突场景中与AI同伴互动，并借助导师的红绿灯引导练习应对策略，实验表明该系统能显著提升9-11岁儿童应对不愉快消息的自我效能感。

    

    儿童之间的同伴冲突越来越多地发生在群聊中，在群聊里语气和时机被剥离，一条伤人的消息发出去比当面说出口更容易。我们提出了Lilo，一个多智能体对话系统，让儿童练习应对这类对话，以便他们在真实的数字交流中遇到这些情况时能够更好地应对。在Lilo中，儿童会在两个AI驱动的同伴争吵中途加入场景，并由一位同样名为Lilo的导师角色通过“停止-思考-行动”红绿灯机制进行指导。通过一项针对28名儿童（9-11岁）的探索性研究，我们考察了儿童如何参与Lilo模拟的同伴冲突以及导师的红绿灯式干预。该研究包括与Lilo的干预活动、干预前后的问卷调查以及退出访谈。参与者在与Lilo互动后，报告了应对不愉快消息的自我效能感显著提升。他们感到可以安全地进行尝试，因为知道不会……（原文摘要在此处截断）

    arXiv:2611.14260v1 Announce Type: cross  Abstract: Children's peer conflicts increasingly play out in group chats, where tone and timing are stripped away, and a hurtful message is easier to send than to say aloud. We present Lilo, a multi-agent conversational system that lets children practice such conversations, so they are better prepared when they encounter them in real digital communication. In Lilo, the child joins two AI-driven peers mid-argument, coached by a mentor character, also named Lilo, through a Stop-Think-Act traffic light. Through an exploratory study with 28 children (aged 9-11), we examine how children engage with Lilo's simulated peer conflict and the mentor's traffic-light interventions. The study included an intervention with Lilo, surveys before and after the intervention, and exit interviews. Participants reported significantly higher self-efficacy for coping with unpleasant messages after their interaction with Lilo. They felt safe experimenting, knowing nothi
    
[^29]: 自进化多模态智能体的跨场景综述与分析

    A Cross-Scenario Survey and Analysis of Self-Evolving Multimodal Agents

    [https://arxiv.org/abs/2611.14256](https://arxiv.org/abs/2611.14256)

    本综述首次对自进化多模态智能体进行形式化定义，并提出三阶段自进化循环分析框架，系统揭示了自我改进机制在多模态推理、生成、GUI导航与具身操作四大场景间非对称迁移的结构性根源。

    

    最近的研究表明，多模态智能体日益能够在无需反复人工标注的情况下实现自我改进。然而，每种自我改进机制都是在单一场景内进行设计和评估的，其前提条件在跨场景迁移中尚未得到验证。本综述研究了自我改进机制在四种场景间非对称迁移的结构性根源，这四种场景包括：多模态推理、多模态生成、GUI与网页导航以及具身操作。我们首先通过四项标准和一个自我参照的判别特征对自进化多模态智能体进行形式化定义，并采用由经验生成、反馈获取和更新巩固构成的三阶段自进化循环作为分析框架，将各场景沿部署次数上限和反馈保真度两个维度进行定位。通过在每个场景内对比自进化循环与固定循环，我们将场景本身施加的限制与自进化机制固有的限制区分开来。

    arXiv:2611.14256v1 Announce Type: new  Abstract: Recent work shows that multimodal agents increasingly self-improve without recurring human annotation. However, each self-improvement mechanism is designed and evaluated within a single scenario, leaving its preconditions unverified under cross-scenario transfer. This survey investigates the structural source of asymmetric mechanism transfer across four scenarios: multimodal reasoning, multimodal generation, GUI and Web navigation, and embodied manipulation. We first formalize the self-evolving multimodal agent by four criteria and a differentia of self-reference, and adopt the three-stage self-evolving loop of experience generation, feedback acquisition, and update consolidation as our analytical framework to position the scenarios along ceilings on rollout count and feedback fidelity. Comparing self-evolving with fixed loops within each scenario separates the limitations that the scenario imposes from those inherent in the self-evolvin
    
[^30]: 基于持续学习的人-智能体协商中的个性化情感感知

    Personalized Affective Perception via Continual Learning in Human-Agent Negotiation

    [https://arxiv.org/abs/2611.14247](https://arxiv.org/abs/2611.14247)

    该论文提出CLIFER——一种基于持续学习与想象机制的面部表情识别方法，使人形机器人在与人类协商中实时个性化地感知情绪并动态调整协商策略，相比基线显著提升了个体协商效用。

    

    与人类进行协商的自利智能体应当能够解读非语言表达并据此调整自身行为。然而，通用的面部情感模型在不同个体和情境之间性能会下降，存在泛化差距。我们提出了一种情感自适应的协商策略，其中融合了基于“想象”的持续学习面部表情识别方法（CLIFER），以实时地实现个性化的效价-唤醒度感知。差分反馈机制将效价-唤醒度方向与强度的变化转化为智能体目标效用的调整。一种异步架构使CLIFER能够在实时交互过程中于后台进行更新。在一项平衡设计的被试内实验中，66名参与者与人形机器人进行协商，结果表明：启用持续学习的智能体相比FaceChannel基线取得了显著更高的个体效用，并引发了参与者更高的让步率。持续学习条件产生了显著的……（原文摘要在此处截断）

    arXiv:2611.14247v1 Announce Type: new  Abstract: Self-interested intelligent agents that negotiate with humans should interpret nonverbal expressions and adapt their behavior accordingly. However, generic facial-affect models degrade across individuals and contexts, creating a generalization gap. We introduce an affect-adaptive negotiation strategy that incorporates Continual Learning with Imagination for Facial Expression Recognition (CLIFER) to personalize valence-arousal perception in real time. Differential feedback converts changes in valence-arousal direction and intensity into adjustments of the agent's target utility. An asynchronous architecture updates CLIFER in the background during live interaction. In a counterbalanced within-subject study in which 66 participants negotiated with a humanoid robot, the CL-enabled agent achieved significantly higher individual utility and elicited higher participant concession rates than the FaceChannel baseline. The CL condition yielded sig
    
[^31]: 基于TRACE框架的法案反腐败专家审查自动化AI助手

    An AI Assistant for Automating Anti-Corruption Expertise of Draft Legislation Based on the TRACE Framework

    [https://arxiv.org/abs/2611.14244](https://arxiv.org/abs/2611.14244)

    该论文基于TRACE架构提出了一种自动化反腐败专家审查的AI助手，能够筛查法案草案中的腐败风险并精确定位专家会批评的问题条款，同时利用乌克兰国家反腐败预防署的公开结论构建了首个开放评估基准。

    

    我们提出了一个基于TRACE架构的AI助手，用于自动化法案草案的反腐败专家审查，旨在筛查法案中的腐败风险：精确定位专家会批评的条款，并指出每一条款中存在的监管缺陷。该处理流程结合了确定性解析核心、模型引导的单一条款定性、基于规则的风险因子分配以及外部人工监督。为了评估该助手，我们根据乌克兰国家反腐败预防署（NAZK）公布的审查结论构建了一个开放基准，包含132项法案草案和474个模型提取的发现。当系统标记指向兼容的条款，且由LLM评判者将风险评级为相同（严格标准）或部分相同（宽松标准）时，即认为该系统标记与发现一致。在100项法案草案上，使用本地Gemma 4 31B模型时，该流程在其前十个标记中匹配了0.28的发现，使用Sonnet 5.5时为0.30，而在……（摘要在此处截断）

    arXiv:2611.14244v1 Announce Type: cross  Abstract: We present an AI assistant for automating anti-corruption expertise of draft legislation based on the TRACE architecture, designed to screen bills for corruption risks: pinpoint provisions an expert would criticise and name the regulatory defect in each. The pipeline combines a deterministic parsing core, model-guided single-provision qualification, rule-based factor assignment, and external human oversight. To evaluate the assistant, we constructed an open benchmark from the published conclusions of Ukraine's National Agency on Corruption Prevention (NAZK), comprising 132 draft laws with 474 model-extracted findings. A system flag agrees with a finding when it targets a compatible provision and an LLM judge rates the risk as the same (strict) or partly the same (lenient). On 100 draft laws, the pipeline matches 0.28 of the findings within its top ten flags using the local Gemma 4 31B model, 0.30 with Sonnet 5.5, and 0.43-0.46 across a
    
[^32]: Dyship：利用声学特征从双人对话中进行说话人关系分类

    Dyship: Leveraging Acoustic Features For Speaker Relationship Classification From Dyadic Conversations

    [https://arxiv.org/abs/2611.14241](https://arxiv.org/abs/2611.14241)

    该论文提出DyShip框架，通过比较频谱图、MFCC、WavLM、x向量和发音运动五种声学表示，从双人对话中实现说话人熟悉度和亲缘关系的自动分类，且多数分类器表现超越人类和音频语言模型基准。

    

    我们提出了DyShip，这是一个用于从Seamless Interaction语料库的双人对话中对说话人熟悉度和亲缘关系进行分类的框架。我们比较了五种语音表示方式：频谱图、梅尔频率倒谱系数（MFCCs）、自监督WavLM表示、x向量以及估计的发音运动。频谱特征在熟悉度分类上表现最佳，准确率达到87.81%，而WavLM在亲缘关系分类上取得了最高的85.17%准确率。x向量和发音特征在熟悉度分类上结果较弱，但在亲缘关系分类上表现出色，其中x向量的AUC达到0.909。大多数分类器在同一数据集上的表现超过了此前报告的人类和音频语言模型的分数。这些发现凸显了传统声学特征在关系推断方面的潜力，并激励人们进一步研究不同表示方式所捕获的对话线索。

    arXiv:2611.14241v1 Announce Type: cross  Abstract: We introduce DyShip, a framework for classifying speaker familiarity and kinship from dyadic conversations in the Seamless Interaction corpus. We compare five speech representations: spectrograms, MFCCs, self-supervised WavLM representations, x-vectors, and estimated articulatory movements. Spectral features perform best for familiarity classification, reaching an accuracy of 87.81%, while WavLM achieves the highest kinship accuracy of 85.17%. X-vectors and articulatory features yield weaker results on familiarity classifiaction, but shine for kinship classification, with an AUC of 0.909 for x-vectors. Most classifiers exceed previously reported human and audio language model scores on the same dataset. These findings highlight the potential of conventional acoustic features for relationship inference and motivate further investigation of the conversational cues captured by different representations.
    
[^33]: 可验证智能体流量类别：一种面向智能体AI流量的受管控服务质量机制，从仿真到真实内核对抗性测试的评估

    Verifiable Agentic Traffic Class: A Policed Quality-of-Service Mechanism for Agentic AI Traffic, Evaluated from Simulation to Real-Kernel Adversarial Testing

    [https://arxiv.org/abs/2611.14211](https://arxiv.org/abs/2611.14211)

    本文提出可验证智能体流量类别（VATC），通过DSCP流量声明、路由器侧验证分类器和速率受限优先队列，为智能体AI流量提供可验证的服务质量保障，在突发场景下相比传统DiffServ将智能体排队延迟降低98%，同时仅使人类流量延迟增加12%。

    

    基于大语言模型构建的软件智能体所产生的流量与人类网页流量截然不同，然而路由器尚无针对此类流量的类别，也无法验证发送方声称自己是智能体的真实性。本文提出了可验证智能体流量类别（VATC），它结合了三项机制：一个用于声明智能体流量的保留差分服务代码点（DSCP）值、一个路由器侧分类器（将与声明不符的流量降级至在尽力而为流量之后服务的类别），以及一个速率受限的优先队列。研究在两个仿真器和一个真实的Linux内核上，将VATC与静态差分服务DiffServ（一种信任标记的专用类别）以及fq_codel（一种每流公平排队调度器）进行了对比。结果表明，VATC在稳定负载下与DiffServ表现相当；在构造的突发流量场景中，VATC在仿真环境下（假设即时判定）相比DiffServ将智能体平均排队延迟降低了98%，而人类流量的延迟仅增加12%，其中大部分收益来自于独立类别的设计。

    arXiv:2611.14211v1 Announce Type: cross  Abstract: Software agents built on large language models generate traffic unlike human web traffic, yet routers have no class for it and cannot verify a sender's claim to be an agent. This paper proposes the Verifiable Agentic Traffic Class (VATC), which combines a reserved Differentiated Services Code Point (DSCP) value declaring agent traffic, a router-side classifier that demotes flows contradicting the declaration to a class served after best-effort traffic, and a rate-limited priority queue. VATC was compared with static Differentiated Services (DiffServ), a dedicated class that trusts the marking, and fq_codel, a per-flow fair-queuing scheduler, in two simulators and on a real Linux kernel. VATC matched DiffServ under steady load. In a constructed burst it cut mean agent queuing delay by 98% relative to DiffServ in simulation (assuming an immediate verdict), at a 12% increase in human delay; most of the gain came from the separate class. U
    
[^34]: 集中式与分散式注意力头失活下自回归多步推理的崩溃

    Breakdown of Autoregressive Multi-Step Reasoning Under Concentrated vs. Diffuse Attention Head Dropout

    [https://arxiv.org/abs/2611.14209](https://arxiv.org/abs/2611.14209)

    受哺乳动物大脑“关闭状态”启发，该研究发现对大语言模型注意力头施加集中式爆发失活，比同等规模的分散式失活会造成严重得多的多步数学推理性能下降。

    

    在持续清醒期间，哺乳动物的新皮层网络会经历局部、短暂的神经元沉默期，被称为“关闭状态”（off-states）。本文研究了类似的时空集中式扰动如何影响自回归大语言模型（LLM）的序列推理能力。我们在GSM8K基准上对两个8B模型（Qwen3-8B与LLaMA-3.1-8B-Instruct）进行了250组配对推导评估，将聚簇爆发的注意力头掩码与预算匹配的分散式独立同分布失活进行对比。结果表明，集中式爆发会导致显著的推理性能退化：在起始参数 $\mathcal{B}=0.10$（调度约5.7%的易感头-词元容量）下，Qwen3-8B的准确率从分散噪声条件下的91.2%降至爆发式失活条件下的78.8%（Δ = −12.4个百分点，精确McNemar检验 p = 5.5×10⁻⁶），而LLaMA-3.1-8B-Instruct则从84.0%降至76.8%（Δ = −7.2个百分点，p = 0.0198）。相反，分散式……（原文摘要在此处截断）

    arXiv:2611.14209v1 Announce Type: cross  Abstract: During extended wakefulness, mammalian neocortical networks experience localized periods of transient neuronal silence termed off-states. Here, we examine how analogous spatiotemporally concentrated disruptions affect sequential reasoning in autoregressive Large Language Models (LLMs). Evaluating 250 paired derivations on the GSM8K benchmark across two 8B models (Qwen3-8B and LLaMA-3.1-8B-Instruct), we contrast clustered bursts of attention head masking against budget-matched diffuse i.i.d. dropout. Concentrated bursts induce substantial reasoning degradation: at onset parameter $\mathcal{B}=0.10$ (scheduling $\sim 5.7\%$ of vulnerable head-token volume), Qwen3-8B accuracy drops from $91.2\%$ under diffuse noise to $78.8\%$ under burst dropout ($\Delta = -12.4$ percentage points [pp], exact McNemar $p = 5.5 \times 10^{-6}$), while LLaMA-3.1-8B-Instruct drops from $84.0\%$ to $76.8\%$ ($\Delta = -7.2$ pp, $p = 0.0198$). Conversely, disp
    
[^35]: 面向上下文感知边缘视觉Transformer的联合量化与动态分割

    Joint Quantization and Dynamic Partitioning for Context-Aware Edge Vision Transformers

    [https://arxiv.org/abs/2611.14196](https://arxiv.org/abs/2611.14196)

    本文提出一种面向车联网边缘的编排框架，将模型分割点、逐层量化位宽与执行位置的联合选择建模为混合整数线性规划（MILP），在时延与任务质量约束下最小化计算和通信能耗，使视觉Transformer推理能够自适应时变的车车通信环境。

    

    在网联自动驾驶车辆（CAVs）中部署视觉Transformer模型需要联合管理计算与车车（V2V）通信。静态分割推理方法使用固定的分割点，无法适应时变的链路带宽、处理负载以及并发的多流流量。本文提出了一种边缘编排框架，可在异构车辆计算节点上联合选择模型分割点、逐层量化位宽以及执行位置。该联合分割与混合精度分配问题被建模为混合整数线性规划（MILP），在端到端时延和任务质量约束下最小化计算与通信能耗。该框架在Jetson Orin Nano（8GB）、Jetson Orin NX（16GB）和Jetson AGX Orin（32GB）节点上，使用基于OPV2V驱动、K∈{2,4,6,8}的多流工作负载进行了评估。

    arXiv:2611.14196v1 Announce Type: new  Abstract: Deploying vision transformer models across connected and automated vehicles (CAVs) requires joint management of computation and Vehicle-to-Vehicle (V2V) communication. Static split-inference methods use fixed partition points and cannot adapt to time-varying link bandwidth, processing load, and concurrent multi-stream traffic. This paper proposes an edge orchestration framework that jointly selects model partition points, per-layer quantization bitwidths, and execution placement across heterogeneous vehicular compute nodes. The joint partitioning and mixed-precision allocation problem is formulated as a Mixed-Integer Linear Program (MILP) that minimizes compute and communication energy subject to end-to-end latency and task-quality constraints. The framework is evaluated on Jetson Orin Nano (8~GB), Jetson Orin NX (16~GB), and Jetson AGX Orin (32~GB) nodes using OPV2V-driven multi-stream workloads with $K \in \{2,4,6,8\}$. Across the eval
    
[^36]: 一种求解提升贝尔曼线性规划的增广拉格朗日方法

    An Augmented Lagrangian Approach to Lifted Bellman Linear Programming

    [https://arxiv.org/abs/2611.14183](https://arxiv.org/abs/2611.14183)

    本文提出用增广拉格朗日乘子法求解提升贝尔曼线性规划（LBLP），将固定的惩罚系数变为通过对偶上升学习的乘子，在无需目标网络的情况下为离线强化学习提供了具有KKT理论保证的神经求解器AL-LBLP。

    

    时序差分（TD）评论家网络向自举目标回归，通常需要目标网络。提升贝尔曼线性规划（LBLP）则将离线强化学习的样本内最优值表示为数据集上一个线性规划的解。其神经求解器ALBUM将约束松弛为具有固定系数的合页惩罚，其理论保证是对预先选定系数的条件。我们采用乘子法求解同一规划：系数变为通过对偶上升学习的乘子，仅引入一个惩罚参数ρ。在凸情形下，算法的不动点即为LBLP的KKT点对，且在每个平稳点处，所施加的约束权重恰好精确平衡目标函数。该评论家网络可服务于高斯演员、流演员以及best-of-N采样，针对高斯演员我们提出了优势归一化的DDPG+BC目标。使用单个Q和V且无需目标网络，我们的AL-LBLP……（摘要在此处被截断）

    arXiv:2611.14183v1 Announce Type: new  Abstract: Temporal-difference critics regress onto a bootstrapped target and usually need a target network. Lifted Bellman Linear Programming (LBLP) instead expresses the in-sample optimal values of offline RL as the solution to a linear program over the dataset. Its neural solver ALBUM relaxes the constraints into hinge penalties with fixed coefficients, and its guarantees are conditions on coefficients chosen in advance. We solve the same program with the method of multipliers: the coefficients become multipliers learned by dual ascent, with one penalty parameter $\rho$. In the convex case the fixed points are the KKT pairs of LBLP, and at every stationary point the applied constraint weights balance the objective exactly. The critic can serve a Gaussian actor, a flow actor, and best-of-N sampling, and for the Gaussian actor we propose an advantage-normalized DDPG+BC objective. With a single $Q$ and $V$ and no target network, our AL-LBLP with be
    
[^37]: 基于双重稳健估计与线性规划的跨境履约中心选择

    Cross-Border Fulfillment Center Selection via Doubly Robust Estimation and Linear Programming

    [https://arxiv.org/abs/2611.14174](https://arxiv.org/abs/2611.14174)

    该论文提出了一个融合预测、双重稳健估计与线性规划的多阶段优化框架来选择最优跨境履约中心，在三条生产级国际运输走廊上相比人工专家策略实现了8.5%-51.1%的配送时间降低和4.4%-48.3%的物流成本降低。

    

    我们提出了跨境履约中心选择模型：这是一个多阶段优化框架，结合了预测、双重稳健估计和线性规划，用于选择最优的源履约中心。对于每条源-目的地运输线路，目标是在固定国内网络的条件下，选择一个源国家履约中心子集，以最小化配送时间和成本，同时最大化利润和发货量。我们利用某大型电商公司12个月的运输数据，在三条生产规模的国际运输走廊上对该模型进行评估，采用聚合离线回放和运输级反事实最近邻匹配两种方法。在所有运输走廊上，与人工专家定义的履约中心选择策略相比，跨境履约中心选择模型实现了8.5%-51.1%的配送时间降低和4.4%-48.3%的物流成本降低，在5/5目标合规性上最多提升了6.5个百分点（摘要原文在此处截断）。

    arXiv:2611.14174v1 Announce Type: new  Abstract: We introduce the cross-border (XB) fulfillment center (FC) Selection Model: a multi-stage optimization framework combining forecasting, doubly robust estimation, and linear programming to select optimal source FCs. For each source-destination shipping arc, the objective is to choose a subset of source-country FCs that minimize delivery time and cost while maximizing profit and shipped units under a fixed domestic network. We evaluate the model on three production-scale international shipping corridors for a large e-commerce company using 12 months of shipment data, employing both aggregate offline replay and shipment-level counterfactual nearest-neighbor matching. Across all shipping corridors, the XB FC Selection Model achieves 8.5-51.1% reductions in delivery time and 4.4-48.3% reductions in logistics cost relative to human expert-defined FC selection policies, with up to 6.5 percentage-point improvements in 5/5 target compliance (5 da
    
[^38]: BARS：未知分布下带拒绝机制的预算感知序贯抽样多数识别

    BARS: Budget-Aware Sequential Sampling with Refusal for Majority Identification under Unknown Distributions

    [https://arxiv.org/abs/2611.14171](https://arxiv.org/abs/2611.14171)

    提出BARS，一种预算感知的序贯抽样规则，在未知分布下通过认证前两名答案差距，返回提前停止、预算多数投票或拒绝三种结果，并提供可证明的错误率界限，帮助实践者判断何时扩大预算或更换模型。

    

    重复抽样并结合多数选择是在大型语言模型中实施自一致性的一种有效手段。最近，还引入了一种变体，当输出分布有利时提前终止以节省成本。但它仍然无法从答案分布中产生正确的置信度估计。针对这一问题，我们将其形式化为在给定预算和目标错误率下的未知分布多数识别问题，并提出BARS，一种对前两名答案间差距进行认证的序贯规则，返回提前停止、带预算的多数投票，或在无法建立可行性时返回拒绝。拒绝是关于规则及其预算的陈述，而非声称问题无法回答，它告诉实践者何时应扩大预算或更换模型。认证模式对联合回答错误以及拒绝预算本可处理的实例具有经过证明的界限，而实用模式则用于……

    arXiv:2611.14171v1 Announce Type: new  Abstract: Repeated sampling with majority selection is an effective means of enforcing self-consistency in large language models. Recently, a variant has also been introduced that terminates early to save costs when the output distribution is favorable. But it still cannot yield a correct confidence estimate from the answer distribution. For this issue, we formalize this as majority identification under an unknown distribution with a budget and a target error rate, and propose BARS, a sequential rule certifying the top-two gap and returning an early stop, a budgeted majority vote, or a refusal when feasibility cannot be established. A refusal is a statement about the rule and its budget, not a claim that a question is unanswerable, and it tells practitioners when to widen the budget or switch models. A certified mode carries proven bounds on the joint answering error and on refusing an instance the budget could handle, while a practical mode, used
    
[^39]: 神经网络中的表征原子

    Representational Atoms in Neural Networks

    [https://arxiv.org/abs/2611.14168](https://arxiv.org/abs/2611.14168)

    该论文定义了神经网络中的基本表征单元（FRUs）并建立保真度与稳定性两项评估标准，证明阈值激活稀疏自编码器（TSAEs）在适当条件下可高保真、稳定地识别FRUs，且高稳定性的FRUs具有更强的单语义性并能跨层组合以恢复与人类概念结构相对应的语义组织。

    

    理解神经网络的内部表征并刻画其与人类概念结构之间的关系，是阐明模型机制的两个关键问题。前者揭示了所学知识如何在内部表征中被组织起来，而后者为借助人类概念结构来解释这些表征提供了基础。在此，我们正式定义了基本表征单元（FRUs），并建立了两个评估标准：保真度与稳定性。我们进一步证明，阈值激活稀疏自编码器（TSAEs）在适当条件下能够以高保真度和高稳定性识别FRUs，并通过大规模实验验证了这一结果。进一步的分析表明，具有更高稳定性的FRUs表现出更强的单语义性，并且能够跨层组合，从而恢复与人类概念结构相对应的语义组织。

    arXiv:2611.14168v1 Announce Type: cross  Abstract: Understanding the internal representations of neural networks and characterizing their relationship with human conceptual structure are two critical questions for elucidating model mechanisms. The former reveals how learned knowledge is organized in internal representations, whereas the latter provides a basis for interpreting these representations through human conceptual structure. Here, we formally define fundamental representational units (FRUs) and establish two evaluation criteria: fidelity and stability. We further prove that threshold-activated sparse autoencoders (TSAEs) can identify FRUs with high fidelity and stability under appropriate conditions, and validate this result through large-scale experiments. Further analyses reveal that FRUs with higher stability exhibit greater monosemanticity and can be composed across layers to recover semantic organization corresponding to human conceptual structure. These results provide a
    
[^40]: Hx2Dx：从问诊到诊断——面向大语言模型对话式临床推理的多模态竞技场

    Hx2Dx: From Anamnesis to Diagnosis - A Multimodal Arena for Conversational Clinical Reasoning in LLMs

    [https://arxiv.org/abs/2611.14122](https://arxiv.org/abs/2611.14122)

    该论文提出了Hx2Dx ARENA——一个基于1000个NEJM临床病例的交互式多模态基准，通过模拟从病史采集到最终诊断的完整问诊流程来评估大语言模型的对话式临床推理能力，弥补了传统静态评估忽视主动问诊能力的不足。

    

    大语言模型（LLM）在静态评估中已达到专家级的诊断准确率，但这种评估中模型接收到的是经过精心整理的病例摘要或病例报告，其中已经浓缩了与诊断相关的关键发现。这类评估衡量的是对已筛选证据进行推理的能力，并未考量获取这些证据所需的问诊过程。然而在临床实践中，诊断是一个不断询问的迭代过程：临床医生通过有针对性的提问采集病史（Hx），对患者进行体格检查，申请并解读检验结果，逐步缩小鉴别诊断范围，直至得出最终诊断（Dx）。我们提出了Hx2Dx ARENA，一个基于《新英格兰医学杂志》（NEJM）1000个临床病例构建的交互式基准。每个病例都是一次模拟的诊疗过程，从最初的病情呈现开始，经历病史采集、体格检查，直至完成诊断性检查。我们对开源和闭源模型进行了评估。

    arXiv:2611.14122v1 Announce Type: new  Abstract: Large language models (LLMs) achieve expert-level diagnostic accuracy in static evaluations, where the model receives a curated vignette or case report that already condenses the diagnostically relevant findings. Such evaluations measure inference over already selected evidence; the evaluations do not account for the inquiry used to obtain it. In clinical practice, however, diagnosis is an iterative process of inquiry. Clinicians take the patient's history, or anamnesis (Hx), through targeted questioning, examine the patient, and request and interpret tests, progressively narrowing the differential until they reach a final diagnosis (Dx). We introduce Hx2Dx ARENA, an interactive benchmark of 1'000 clinical cases from the New England Journal of Medicine (NEJM). Each case is a simulated encounter that proceeds from initial presentation through history taking and physical examination to diagnostic workup. We evaluate open- and closed-source
    
[^41]: 成人词汇扩展与人造词汇学习中的视觉相似性

    Visual Similarity in Adult Word Extension and Artificial Word Learning

    [https://arxiv.org/abs/2611.14116](https://arxiv.org/abs/2611.14116)

    本研究通过重新分析大规模成人词汇扩展数据并进行人工词汇学习实验，利用冻结的自监督视觉编码器（DINOv2）量化物体间的视觉相似性，发现视觉距离显著影响词汇的扩展范围，且教学示例的类别层级高低决定了学习者在多大程度上将词汇泛化到更广泛或更具体的类别。

    

    当两个物体共享同一个名称时，它们的视觉相似性能在多大程度上解释这一现象，教学示例又能带来什么额外作用？首先，我们重新分析了一项大型成人词汇扩展研究的试验级数据（Lewis & Frank, 2018），在冻结的自监督视觉编码器（DINOv2）的空间中计算每个测试图像与所展示示例之间的距离。距离与词汇扩展显著相关（每标准差对数几率为-3.23；共520名参与者），一项探索性事后分析表明，在控制了分类层级和教学条件后，这种关联在已教授的领域内依然存在。示例的组成同样重要：在学习三个示例后，当示例为下位类别时，成人仅将词汇扩展到上位类别项目的1.1%，而当示例为上位类别时，这一比例达到77.5%。其次，在使用同一编码器的人工学习者实验中，当一个词连接两个类别时，类别之间距离越远，学习准确度越低（84.7%……

    arXiv:2611.14116v1 Announce Type: cross  Abstract: When two objects share a name, how much does their visual similarity explain, and what do the teaching examples add? First, we reanalyzed trial-level data from a large adult word-extension study (Lewis & Frank, 2018), computing each test image's distance to the examples shown in the space of a frozen self-supervised visual encoder (DINOv2). Distance was associated with extension (log odds -3.23 per standard deviation; 520 people), and an exploratory post hoc analysis suggested that this association persisted within the taught domain after controlling for taxonomic level and teaching condition. Example composition also mattered: after three examples, adults extended a word to 1.1% of superordinate items when the examples were subordinate and to 77.5% when they were superordinate. Second, in an artificial learner using the same encoder, words joining two categories were learned less accurately the farther apart the categories were (84.7%
    
[^42]: 前瞻一步：一种用于视频中物体状态读取的预判引导智能体

    Looking One Step Ahead: An Anticipation-guided Agent for Object State Readout in Videos

    [https://arxiv.org/abs/2611.14113](https://arxiv.org/abs/2611.14113)

    该论文提出了一种名为LASER的预判引导智能体，通过预测物体下一步可能发生的变化来确定当前帧中需要重点检查的视觉属性，从而实现视频中物体状态及其转换的准确读取。

    

    理解一项活动的进展需要追踪相关物体如何随时间变化。物体状态读取通过描述目标物体在视频各帧中的状态及其转换来提供这一信息。尽管多模态大语言模型（MLLM）能够生成详细的视觉描述，但这些描述可能未能充分明确区分连续状态所需的属性。在本文中，我们提出了前瞻式状态读取（LASER），这是一种预判引导的物体状态读取智能体，它利用可能的下一步变化来确定当前应该检查哪些属性。给定按时间排序的视频帧和目标物体，LASER 将预期的变化转化为有针对性的视觉问题，并检查当前帧以确定属性值。随后，它将状态读取和预测结果与对后续帧的独立观察进行核对，从而不断完善用于生成状态记录的信息。

    arXiv:2611.14113v1 Announce Type: cross  Abstract: Understanding the progress of an activity requires tracking how the objects involved change over time. Object state readout provides this information by describing a target object's states and transitions across video frames. Although multimodal large language models (MLLMs) generate detailed visual descriptions, these descriptions may leave the attributes needed to distinguish successive states insufficiently specified. In this paper, we propose look-ahead state readout (LASER), an anticipation-guided agent for object state readout that uses possible next-step changes to determine which attributes to inspect in the present. Given temporally ordered frames and a target object, LASER converts anticipated changes into targeted visual questions and inspects the current frame to establish attribute values. It then checks state readouts and forecasts against independent observations of subsequent frames, refining a state record used to gene
    
[^43]: 无模型的任务相关信息运行时测量

    Model-Free Runtime Measurement of Task-Relevant Information

    [https://arxiv.org/abs/2611.14111](https://arxiv.org/abs/2611.14111)

    提出了一种无模型的采集状态遥测（AST）架构，通过固定的因果监控器输出实时信号，评估自主系统的信息采集过程是否真正朝着任务需求的解决方向正常推进。

    

    自主系统日益需要对物理运动学、环境状态以及任务完成结果进行监控。尽管现有工作提供了规划信息搜寻动作和解读所获观测数据的方法，但目前的方法并未监控实时的信息收集过程是否正常运行。因此，这类系统可能在尚未解决未完成的任务需求的情况下仍持续保持运行状态。为了解决这一局限，我们提出了采集状态遥测，这是一种无模型架构，能够基于可观测证据相对于声明的任务需求来评估正在进行的信息采集过程。具体而言，一个领域接口将原始观测映射为任务解析坐标和归一化的比较表示，这些表示由一个固定的因果监控器进行处理。该监控器输出两个连续的实时信号，用以量化尚未解决需求的前瞻性进展……

    arXiv:2611.14111v1 Announce Type: cross  Abstract: Autonomous systems increasingly monitor physical kinematics, environmental states and task completion outcomes. While existing work provides methods to plan information seeking actions and interpret acquired observations, current approaches do not monitor whether the live process of information gathering is functioning properly. Consequently, such systems can remain operationally active without resolving outstanding mission requirements. To address this limitation, we introduce Acquisition-State Telemetry (AST), a model free architecture that evaluates ongoing information acquisition from observable evidence relative to declared task requirements. Specifically, a domain interface maps raw observations into task resolution coordinates and normalized comparative representations, which are processed by a fixed causal monitor. This monitor emits two continuous real time signals that quantify (i) prospective progress across unresolved requi
    
[^44]: ProofCoder：面向自动化软件验证的连续推理

    ProofCoder: Continuous Reasoning for Automated Software Verification

    [https://arxiv.org/abs/2611.14106](https://arxiv.org/abs/2611.14106)

    ProofCoder从编码智能体视角出发，让模型在整个证明搜索过程中持续推理，并根据证明助手的反馈自主决定信息收集、证明状态检查、片段修改和回溯，突破了传统LLM方法僵化策略生成循环的限制，实现了面向Rocq的自动化证明合成。

    

    使用证明助手进行形式化验证能够为软件提供强有力的正确性保证，但构造证明仍然是一项劳动密集型工作，且需要大量的专业知识。近期基于大语言模型（LLM）的方法在自动化证明合成方面展现出了潜力。然而，这些方法通常将模型限制在僵化的策略生成循环中，由外围系统控制前提检索、证明状态管理和回溯。我们提出了ProofCoder，这是一个面向Rocq的智能体式证明合成系统，它从编码智能体的视角来处理定理证明问题。ProofCoder并非在固定循环中生成策略，而是在整个证明搜索过程中持续进行推理，并利用证明助手的反馈来决定何时收集额外信息、检查证明状态、修改先前生成的证明片段，或回溯到证明中较早的位置。这使其能够迭代地协同……

    arXiv:2611.14106v1 Announce Type: cross  Abstract: Formal verification using proof assistants provides strong correctness guarantees for software, but constructing proofs remains labor-intensive and requires significant expertise. Recent approaches based on large language models (LLMs) have shown promise in automating proof synthesis. However, these approaches typically constrain the model to a rigid tactic-generation loop, in which the surrounding system controls premise retrieval, proof-state management, and backtracking. We introduce ProofCoder, an agentic proof-synthesis system for Rocq that approaches theorem proving from a coding-agent perspective. Rather than generating tactics within a fixed loop, ProofCoder reasons continuously throughout proof search and uses proof-assistant feedback to decide when to gather additional information, inspect proof states, revise previously generated proof fragments, or backtrack to an earlier point in the proof. This allows it to iteratively co
    
[^45]: Sable：面向大语言模型位精确推理的小域Sumcheck论证系统

    Sable: A Small-Field Sumcheck Argument for Bit-Exact Inference of Large Language Models

    [https://arxiv.org/abs/2611.14083](https://arxiv.org/abs/2611.14083)

    Sable通过整数化确定性计算、小域sumcheck约束系统和单GPU逐层流式证明，实现了对大语言模型位精确推理的可验证性，在单块GPU上92.5秒内即可完成Qwen2.5-32B生成4096个token全部64层推理的证明。

    

    我们提出Sable，一个论证系统与单GPU证明器，用于证明一个已承诺的、整数量化的transformer模型从给定提示词逐位精确地生成了给定的token序列。Sable定义了一个不含任何浮点步骤的整数函数F_O，其中每一次舍入、归一化、softmax和采样决策都完全由其输入唯一确定，并通过一个约束系统对其加以证明，该约束系统的关系可证明在31位KoalaBear素数模运算下永不发生回绕。该证明将零值检查（zerochecks）与LogUp-GKR查找批处理为每个域上的单次sumcheck，并通过一个由Reed-Solomon码、BLAKE3 Merkle树和混合长度BaseFold打开构成的多线性承诺，一次性承诺所有见证张量与权重。证明器在单块GPU上逐层流式处理模型。在一块NVIDIA RTX PRO 6000 Blackwell GPU上，Sable用92.5秒证明Qwen2.5-32B全部64层生成4,096个token，用398秒生成16,384个token，并覆盖Qwen3-32B与专家混合模型（mixture-of-experts）……（摘要原文在此处截断）

    arXiv:2611.14083v1 Announce Type: cross  Abstract: We present Sable, an argument system and single-GPU prover for the statement that a committed, integer-quantized transformer produced a given token sequence from a given prompt, bit for bit. Sable fixes an integer function F_O with no floating-point step, in which every rounding, normalization, softmax and sampling decision is determined by its inputs, and proves it with a constraint system whose relations provably never wrap modulo the 31-bit KoalaBear prime. The proof batches zerochecks and LogUp-GKR lookups into one sumcheck per domain, and commits to every witness tensor and weight in one multilinear commitment built from Reed-Solomon codes, BLAKE3 Merkle trees and a mixed-length BaseFold opening. The prover streams the model layer by layer on one GPU. On one NVIDIA RTX PRO 6000 Blackwell GPU, Sable proves all 64 layers of Qwen2.5-32B in 92.5 s for 4,096 tokens and 398 s for 16,384 tokens, and covers Qwen3-32B and the mixture-of-ex
    
[^46]: 从偏见到处境性：大语言模型的批判性语法

    From Bias to Positionality: A critical grammar for Large Language Models

    [https://arxiv.org/abs/2611.14079](https://arxiv.org/abs/2611.14079)

    该论文提出用“处境性”取代“偏见”作为理解大语言模型的批判性框架，因为偏见需要外部标准作为参照，而在语言模型的一般使用中不存在这样的标准，因此更应追问模型“从何处说话”。

    

    偏见已成为理解机器学习系统行为以及我们应如何与之相处的一种默认视角。偏见本质上是相对于某一标准的偏离；在决策系统中，我们可以从外部提供标准：法律、受保护属性、基准测试、统计特性。然而，即便在这些场景中，“有偏见”往往只是一种简略说法，并未说明所采取的立场是什么——一个沉默的“相对于”。在语言模型的一般使用中，没有任何东西提供标准：没有可供审计的决策，也没有反事实的输出，因此这种简略说法失效了。我们认为，处境性——即说出任何话语时所占据的、被清晰表述的、被承认的且可被质疑的位置——才是回答这些模型如今迫使我们面对的问题的更好语法：谁在说话？从何处说话？针对厂商“无处”的回答以及“处处”这一诱人的可能性，我们认为，当模型被视为“仿佛在说话”时，它就是从某个地方说话的……

    arXiv:2611.14079v1 Announce Type: cross  Abstract: Bias has become a default lens for understanding what machine learning systems do and how we should relate to them. Bias is at root a deviation from a standard; in decision making systems we can supply a standard from outside: a law, a protected attribute, a benchmark, a statistical property. However, even there, biased is often used as a shorthand, without stating what the position taken is - a silent ``relative to''. In a general use of language models, nothing supplies the standard: there is no decision to audit and no counterfactual output, so the shorthand fails. We argue that positionality -- the located, articulated, owned and contestable position from which anything is said -- is the better grammar for the question these models now force: who is speaking, from where? Against the vendors' answer of nowhere and the seductive possibility of everywhere, we argue that when a model is taken as-if speaking, it speaks from somewhere, a
    
[^47]: ZoomGain：挖掘全局到缩放的感知差距以实现细粒度视觉监督

    ZoomGain: Mining Global-to-Zoom Perception Gaps for Fine-Grained Visual Supervision

    [https://arxiv.org/abs/2611.14076](https://arxiv.org/abs/2611.14076)

    ZoomGain利用视觉语言模型“放大后能识别细节、全局视角却难以感知”的差距，构建了一种仅依赖缩放所得视觉信息、并经多模型在缩放视图上验证的细粒度视觉监督数据自动生成流水线。

    

    细粒度视觉感知要求视觉语言模型（VLM）能够定位并解读复杂场景中的细微证据。为这种能力构建监督数据十分具有挑战性：问题必须具有正确、无歧义的答案，且答案应依赖于真实的细粒度视觉细节，而非粗略线索或语言先验。验证这些特性使得人工标注非常费力，而对于自身就难以进行细粒度感知的模型来说，自动生成也同样困难。然而，同样的模型在区域被放大后往往能够识别这些细节。我们利用这种“全局到缩放”的感知差距，提出了ZoomGain——一个数据构建流水线，它仅通过放大获得的视觉信息来构建针对完整图像的问题。该流水线通过多个模型在缩放视图上验证每个答案，并确保每个问题在完整图像中保持无歧义、且无法仅凭语言先验回答。

    arXiv:2611.14076v1 Announce Type: cross  Abstract: Fine-grained visual perception requires Vision-Language Models (VLMs) to locate and interpret subtle evidence within complex scenes. Constructing supervision for this ability is challenging: questions must have correct, unambiguous answers that depend on actual fine-grained visual details rather than coarse cues or language priors. Verifying these properties makes manual annotation labor-intensive and automatic generation difficult for models that themselves struggle with fine-grained perception. Yet the same models can often recognize such details once the region is zoomed in. We exploit this global-to-zoom perception gap and introduce ZoomGain, a data construction pipeline that constructs questions about the full image from visual information gained only by zooming in. It verifies each answer on the zoomed view with multiple models, ensures that each question remains unambiguous in the full image and cannot be answered from language 
    
[^48]: 使用逐帧Transformer对烟雾模式进行时空分析以检测被遮挡的野火火焰

    Obscured Wildfire Flame Detection by Spatio-temporal Analysis of Smoke Patterns Using Frame-wise Transformers

    [https://arxiv.org/abs/2611.14074](https://arxiv.org/abs/2611.14074)

    该论文提出一种逐帧Transformer架构，通过对RGB视频中烟雾运动的时空联合分析来检测可见及被遮挡的野火火焰，在保留帧内空间信息的同时优化时序长度，实现高性能并降低计算成本。

    

    在本文中，我们提出了一种野火管理的新范式，利用深度学习来检测可见火焰和被遮挡的火焰。我们方法的核心在于对RGB视频进行空间与时间上的联合分析，其关键洞察是：连续视频帧中烟雾运动的时序模式可以指示被遮挡的火焰。这一视角与传统上独立分析图像和单个视频帧的方法形成鲜明对比。为此，我们提出了一种逐帧Transformer架构，该架构在连续帧之间应用注意力机制，同时保留帧内的空间信息，这与将空间和时间信息压缩为单一向量的常规注意力机制（如non-local block）有所不同。我们还在不同空间分辨率下优化了时序模式的长度，以便在降低计算成本的同时获得最佳性能。

    arXiv:2611.14074v1 Announce Type: cross  Abstract: In this paper, we propose a new paradigm in wildfire management for detecting both visible and obscured fire using Deep Learning (DL). The essence of our method is the joint spatial and temporal analysis of RGB videos, embracing the fact that temporal patterns of smoke motions in consecutive video frames can indicate obscured fire flames. This perspective contrasts the traditional approach of independently analyzing images and individual video frames. To this end, we propose a frame-wise transformer architecture that applies an attention mechanism between the consecutive frames while preserving spatial information within frames as opposed to regular attention mechanisms like the non-local block, which combines spatial and temporal information into a single vector. We also optimize the temporal pattern lengths under different spatial resolutions to achieve the highest performance while reducing the computational cost. To investigate the
    
[^49]: HandsOnAid：构建并评估面向应急响应场景的离线急救助手

    HandsOnAid: Building and Evaluating an Offline First-Aid Assistant for Emergency Response Scenarios

    [https://arxiv.org/abs/2611.14059](https://arxiv.org/abs/2611.14059)

    本文构建了HandsOnAid——一个可在无信号、无救援环境中通过QLoRA微调的4位Gemma 2B模型实现的离线手机急救助手，并通过四个前沿大模型盲评验证了微调显著提升了急救建议的质量与安全性。

    

    紧急情况可能发生在没有手机信号且附近没有救护车的地方。在这种情况下，手机上的小型语言模型可能会有所帮助，但其建议必须安全且准确。我们构建了HandsOnAid，这是一个4位量化的Gemma 2B模型，通过QLoRA在FirstAidQA数据集（包含5,550个急救问答对的语料库）上进行了微调。我们比较了七种配置，包括基础模型、4位和8位微调模型，以及带有长度下限、安全门控或检索功能的版本。我们使用了一个包含41对问答的题库，其医学审核参考遵循ANZCOR指南，并采用严格的0-5分评分标准，包含十二项安全否决机制。四个前沿大语言模型评审（GPT-5.6 Sol、Claude Opus 5、DeepSeek V4 Pro、GLM-5.3）对所有答案进行了盲评，并使用45个隐藏对照组来检验评审的可靠性。微调使平均分数比基础模型提高了1.08，达到2.34。最佳配置得分为5分中的2.51分。

    arXiv:2611.14059v1 Announce Type: cross  Abstract: Emergencies can occur where there is no phone signal and no ambulance nearby. A small language model on a phone can be helpful in such situations, but its advice needs to be safe and accurate. We built HandsOnAid, a 4-bit Gemma 2B model fine-tuned with QLoRA on FirstAidQA dataset, a corpus of 5,550 first-aid question-answer pairs. We compared seven configurations, including the base, 4-bit and 8-bit fine-tuned models, and versions with a length floor, safety gate, or retrieval. We used a 41-pair question-answer bank with medically reviewed references that followed ANZCOR guidelines, and a strict 0-5 score rubric, with twelve safety overrides. Four frontier LLM judges (GPT-5.6 Sol, Claude Opus 5, DeepSeek V4 Pro, GLM-5.3) blindly scored every answer, with 45 hidden controls used to check judge reliability. Fine-tuning improved the mean score by +1.08 over the base model to 2.34. The best configuration scored 2.51 out of 5 which was the 
    
[^50]: KV缓存能否在模型更新之间复用？一项基于DeepSeek-V4-Flash的案例研究

    Can KV Cache Be Reused Across Model Updates? A DeepSeek-V4-Flash Case Study

    [https://arxiv.org/abs/2611.14051](https://arxiv.org/abs/2611.14051)

    该研究首次系统探索了KV缓存在模型更新间的直接复用，发现缓存可保留任务信息并带来最高5.08倍的首词生成加速，但复用缓存并不忠实复现新模型行为，为此提出了函数性缓存偏差（Functional Cache Divergence）来量化这种行为差异。

    

    在语言模型更新之间复用键值（KV）缓存可以避免对共享上下文的重复计算，但模型权重的变化引出了一个根本性问题：这些缓存究竟保留了什么。我们研究了跨四个模型更新的直接缓存复用，并对DeepSeek-V4-Flash到Flash-0731的更新进行了详细研究。我们的核心发现是：缓存可以在不忠实复现更新后模型行为的情况下，仍保留有用的任务信息。检索准确率在不同更新之间差异巨大，从接近原生模型的性能到大幅性能损失不等。在源缓存已可用的情况下，复用在三个更长前缀的问答基准上实现了中位数首词生成时间2.10至5.08倍的加速。为了刻画这些结果背后的行为差异，我们提出了函数性缓存偏差，用于比较复用缓存与原生缓存下的下一词元分布。直接KV缓存复用能更好地保留原生模型行为……

    arXiv:2611.14051v1 Announce Type: cross  Abstract: Reusing key--value (KV) caches across language model updates could avoid repeated computation of shared context, but changes in model weights raise a fundamental question about what these caches preserve. We investigate direct cache reuse across four model updates, with a detailed study of DeepSeek-V4-Flash to Flash-0731. Our central finding is that caches can retain useful task information without faithfully reproducing the updated model's behavior. Retrieval accuracy varies substantially across updates, ranging from near-native performance to large losses. With source caches already available, reuse achieves median time-to-first-token speedups of $2.10$--$5.08\times$ on three longer-prefix QA benchmarks. To characterize the behavioral differences behind these outcomes, we introduce Functional Cache Divergence, which compares next-token distributions under reused and native caches. Direct KV cache reuse better preserves native model b
    
[^51]: MISO-BR：一个用于巴西葡萄牙语厌女症检测的开放语料库

    MISO-BR: An Open Corpus for Misogyny Detection in Brazilian Portuguese

    [https://arxiv.org/abs/2611.14036](https://arxiv.org/abs/2611.14036)

    该论文推出了MISO-BR——首个面向巴西葡萄牙语厌女症检测的开放、有文档记录且版本化的标注语料库，附带完整的构建协议、标注指南和可复现的基线模型，填补了该领域开放资源的空白。

    

    厌女症是针对巴西用户最普遍的网络仇恨言论形式之一，然而专门针对巴西葡萄牙语的开放且可审计的相关资源仍然稀缺。我们推出了MISO-BR，这是一个开放的、有完整文档记录且版本化的语料库，包含896条平衡分布的社交媒体文本，标注了二元厌女症标签，并随附其构建协议和可复现的基线。该语料库结合了从X平台进行的基于词典和目标的定向抓取、从OLID-BR导入的项目、一个大语言模型筛选阶段（其提示词和解码参数均已公开），以及三名标注者按照书面指南进行的人工标注。在816个项目上，标注者间信度达到Fleiss' kappa 0.788。在单一固定协议下，我们比较了四种经典分类器（多项式朴素贝叶斯、伯努利朴素贝叶斯、线性SVM和随机森林）与微调后的BERTimbau Base和Large模型。经典模型的宏平均F1分数聚集在0.651至0.676之间。

    arXiv:2611.14036v1 Announce Type: cross  Abstract: Misogyny is among the most prevalent forms of online hate speech directed at Brazilian users, yet open and auditable resources dedicated to it in Brazilian Portuguese remain scarce. We introduce MISO-BR, an open, documented and versioned corpus of 896 balanced social media texts annotated for binary misogyny, released with its construction protocol and reproducible baselines. The corpus combines lexicon- and target-guided scraping from X with items imported from OLID-BR, a large language model triage stage whose prompt and decoding parameters are released, and manual annotation by three annotators following a written guide. Inter-annotator reliability reaches a Fleiss' kappa of 0.788 over 816 items. Under a single fixed protocol we compare four classical classifiers (Multinomial and Bernoulli Naive Bayes, a linear SVM and a random forest) with fine-tuned BERTimbau Base and Large. The classical models cluster between 0.651 and 0.676 mac
    
[^52]: 基于条件偏移的少步流式视频生成

    Few-Step Streaming Video Generation via Condition Shifting

    [https://arxiv.org/abs/2611.14033](https://arxiv.org/abs/2611.14033)

    APEX通过条件偏移机制在生成器内部复用同一参数估计伪造分数，移除了DMD所需的独立伪造网络，同时保留冻结教师与自回放强制等机制，实现了更高效的少步流式视频生成。

    

    少步采样降低了流式视频生成的成本。分布匹配蒸馏（DMD）需要一个独立的伪造分数网络来监督少步生成器，这会增加训练内存和计算开销，并且必须追踪不断演化的生成器。我们通过条件偏移机制在生成器内部估计生成分布的速度场，将APEX扩展到流式视频生成。原始文本条件用于选择生成，而偏移后的条件在相同参数下选择伪造估计。这移除了独立的伪造网络，同时保留了S2-Avatar的冻结参考教师、Self-Replay Forcing和感知正则化。我们在180秒视频上评估APEX，每个块使用三次骨干评估和一步空间细化器，并进行了70例人类研究和320例StreamAV-Bench对比。与S2-Avatar相比，APEX在Progressiv（原文截断）上实现了更高的视频指令遵循度。

    arXiv:2611.14033v1 Announce Type: cross  Abstract: Few-step sampling reduces the cost of streaming video generation. Distribution Matching Distillation (DMD) supervises few-step generators with a separate fake-score network, which adds training memory and computation and must track the evolving generator. We extend APEX to streaming video by using condition shifting to estimate the velocity field of the generated distribution within the generator. The original text condition selects generation, while a shifted condition selects fake estimation using the same parameters. This removes the independent fake network while retaining S2-Avatar's frozen reference teacher, Self-Replay Forcing, and perceptual regularization. We evaluate APEX on 180-second videos using three backbone evaluations per block and a one-step spatial Refiner, with a 70-case human study and a 320-case StreamAV-Bench comparison. Compared with S2-Avatar, APEX achieves higher video instruction fulfillment on the Progressiv
    
[^53]: 程序辅助的可补丁技能演化

    Program-Aided Patchable Skill Evolution

    [https://arxiv.org/abs/2611.14019](https://arxiv.org/abs/2611.14019)

    该论文提出PASE框架，通过将智能体技能中的可复用程序性逻辑分离为可编辑的可执行程序（而非自然语言文档），实现技能的可维护性、可补丁化与针对性迭代演化。

    

    智能体技能已成为一种有效手段，通过编码可复用知识而无需修改模型参数，使LLM智能体能够解决复杂的交互式任务。最近的方法通过从智能体执行轨迹中提炼可复用经验，并基于下游任务反馈来演化技能，进一步提升了技能适应能力。然而，现有方法主要以自然语言文档的形式表示技能，将程序性逻辑与描述性知识交织在一起，使得技能难以维护和演化。为解决这一局限，我们提出了PASE（程序辅助的可补丁技能演化），一个通过将技能与可执行程序相关联来迭代演化技能的程序辅助框架。与先前主要用自然语言编码轨迹经验的方法不同，PASE将可复用的程序性逻辑分离为可编辑的程序，从而能够对可执行组件进行针对性更新……（摘要原文在此处截断）

    arXiv:2611.14019v1 Announce Type: cross  Abstract: Agent skills have emerged as an effective means of enabling LLM agents to solve complex, interactive tasks by encoding reusable knowledge without modifying model parameters. Recent approaches further improve skill adaptation by distilling reusable experience from agent execution trajectories and evolving skills based on downstream task feedback. However, existing methods primarily represent skills as natural language documents, intertwining procedural logic with descriptive knowledge and making skills difficult to maintain and evolve. To address this limitation, we propose PASE (Program-Aided Patchable Skill Evolution), a program-aided framework that iteratively evolves skills by associating them with executable programs. Unlike prior approaches that encode trajectory-derived experience primarily in natural language, PASE separates reusable procedural logic into editable programs, enabling targeted updates to executable components whil
    
[^54]: 交错式融合解剖检查与诊断推理实现可解释的放射学报告生成

    Interleaving anatomical inspection and diagnostic reasoning for explainable radiology report generation

    [https://arxiv.org/abs/2611.14018](https://arxiv.org/abs/2611.14018)

    ARIVE提出了一种可解释的放射报告生成框架，在撰写报告前交错进行解剖检查与诊断推理，并利用159,416条多模态思维链轨迹训练和奖励优化，实现了临床疗效超越强基线的可解释报告生成。

    

    自动化放射学报告生成能够起草临床上合理的报告，然而从视觉证据到报告发现的路径仍然是不透明的，导致这些报告的诊断依据无法得到解释，生成的报告也难以审查或修改。我们提出了ARIVE（基于解剖引导的交错式视觉证据推理），这是一个可解释的报告生成框架，它在最终报告撰写之前，将解剖检查与诊断推理交错进行。为支持这一过程，我们从MIMIC-CXR构建了159,416条多模态思维链（MCoT）轨迹，其中来自被检查解剖区域的视觉证据为后续的临床解读提供依据。ARIVE基于这些轨迹进行训练，并通过解剖学奖励和临床内容奖励进一步优化，在MIMIC-CXR和以异常为中心的MIMIC-ABN数据集上，相比强基线方法取得了更优的临床疗效表现。

    arXiv:2611.14018v1 Announce Type: cross  Abstract: Automated radiology report generation can draft clinically plausible reports, yet the pathway from visual evidence to reported findings remains opaque, leaving the diagnostic basis of these reports unexplained and the generated reports difficult to review or revise. We introduce ARIVE (Anatomy-guided Reasoning with Interleaved Visual Evidence), an explainable report generation framework that interleaves anatomical inspection with diagnostic reasoning prior to final report composition. To support this process, we constructed 159,416 multimodal chain-of-thought (MCoT) trajectories from MIMIC-CXR, wherein visual evidence from inspected anatomical regions informs subsequent clinical interpretation. Trained on these trajectories and further optimized using anatomical and clinical-content rewards, ARIVE achieved superior clinical-efficacy performance compared with strong baselines on MIMIC-CXR and the abnormality-centered MIMIC-ABN. ARIVE's 
    
[^55]: 个性化记忆量规：通过量规引导的信用分配学习每个用户应该记住什么

    Personalized Memory Rubrics: Learning What to Remember per User via Rubric-Guided Credit Assignment

    [https://arxiv.org/abs/2611.14017](https://arxiv.org/abs/2611.14017)

    该论文提出个性化记忆量规（PMR）框架，将终端问答反馈转化为细粒度、按步骤且按用户自适应的信用分配，解决了长时程个性化智能体记忆更新策略训练中反馈信号粗粒度与用户间异质性两大难题。

    

    长时程个性化智能体通过持续决定添加、修改或丢弃哪些内容来维护一个不断演化的记忆。然而，学习最优的更新策略仍然充满挑战。传统训练仅依赖最终的问答准确率，由此产生的粗粒度反馈信号对所有中间更新一视同仁。此外，更新效用本质上是用户特定的，因为相同的记忆操作在不同用户画像下可能产生截然相反的效果。现有方法未能解决这种异质性，它们依赖全局共享的评估标准，且最多只有一个标量形式的中间奖励。为应对这些挑战，我们提出了个性化记忆量规（PMR），这是一个将终端反馈转化为细粒度的、按步骤且按用户的信用分配框架。PMR将评估分解为两个层次：一层是通用的，用于捕获跨用户画像的失败模式；另一层是每个用户自适应的……

    arXiv:2611.14017v1 Announce Type: new  Abstract: Long-horizon personalized agents maintain an evolving memory by continuously deciding what to add, modify, or discard. However, learning an optimal update policy remains challenging. Conventional training relies solely on terminal question-answering accuracy, yielding a coarse feedback signal that treats all intermediate updates identically. Furthermore, update utility is inherently user-specific, as identical memory actions may yield opposite effects across different user personas. Existing methods fail to address this heterogeneity, relying on globally shared evaluation standards with at most a single scalar intermediate reward. To address these challenges, we propose \textbf{Personalized Memory Rubrics} (\textbf{PMR}), a framework that translates terminal feedback into fine-grained, per-step, and per-user credit assignment. PMR decomposes evaluation into a universal layer capturing cross-persona failure patterns and a per-user adaptiv
    
[^56]: 边缘引导的Token信用分配用于可验证奖励的强化学习

    Margin-Guided Token Credit for Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2611.14015](https://arxiv.org/abs/2611.14015)

    提出MCPO方法，利用策略下一Token分布的前两名概率差识别模型犹豫的Token位置，将奖励信用有针对性地集中于这些位置，且无需额外前向传播或批次统计，在六个竞赛数学基准上平均超越GRPO基线6.55分。

    

    在可验证奖励的强化学习（RLVR）中，语言模型因产生正确的最终答案而获得奖励，而GRPO会将该奖励均等地传递给解法中的每一个Token。我们转而将信用分配到模型犹豫不决的位置，并将该方法命名为MCPO（Margin-Credit Policy Optimization，边缘信用策略优化）。我们通过策略下一Token分布的前两名概率差 $M=p_1-p_2$ 来衡量犹豫程度，用两个固定阈值将其映射为分数，并将每个解的优势（advantage）分配到其各个Token上，同时保持均值不变。该边缘值可以直接从训练器已经执行的前向传播中读取，因此该方法无需额外的模型评估，也不需要批次统计信息。我们使用LoRA在2,048个DAPO-Math问题上对MiniCPM5-2B-Base进行200步训练，并在六个竞赛数学基准上进行评估。在全部五个数据种子下，MCPO均优于未修改的GRPO基线，macro avg@32平均提升6.55分（95%置信区间 $[4.1, 9.0]$），并在Qwen3-4B-Base的复现实验中（原文在此截断）……

    arXiv:2611.14015v1 Announce Type: cross  Abstract: In reinforcement learning with verifiable rewards, a language model is rewarded for a correct final answer, and GRPO passes that reward to every token of the solution in equal measure. We instead place the credit where the model hesitated, in a method we call MCPO (Margin-Credit Policy Optimization). We measure hesitation by the top-2 margin $M=p_1-p_2$ of the policy's next-token distribution, map it to a score with two fixed thresholds, and spread each solution's advantage over its tokens so that the mean stays unchanged. The margin is read from a forward pass the trainer already performs, so the method needs no extra model evaluation and no batch statistics. We train MiniCPM5-2B-Base with LoRA for 200 steps on 2,048 DAPO-Math problems and evaluate on six competition-math benchmarks. MCPO beats an unmodified GRPO baseline for all five data seeds, by 6.55 points of macro avg@32 on average (95\% CI $[4.1, 9.0]$), and a Qwen3-4B-Base rep
    
[^57]: 一种用于小动脉-静脉比值估计与视网膜病变筛查的分阶段可解释流水线

    A Stage-Wise Explainable Pipeline for Arteriolar-to-Venular Ratio Estimation and Retinopathy Screening

    [https://arxiv.org/abs/2611.14013](https://arxiv.org/abs/2611.14013)

    该论文提出了一种五阶段可解释流水线，通过可检查的中间输出和显式阈值规则实现眼底图像的AVR估计与视网膜病变筛查，使错误可追溯到具体处理阶段，并在30张HRF图像上达到96.7%的准确率。

    

    小动脉与静脉直径比值（AVR）是临床上评估视网膜血管健康的重要指标。我们提出了一种五阶段流水线，用于从眼底图像进行AVR估计和探索性视网膜病变筛查，每个阶段均提供可检查的中间输出。该流水线结合了血管与动静脉掩膜、视盘定位、基于最小弦边界游走法的血管管径估计，以及基于主要动脉和静脉分支的AVR计算。筛查决策由显式的阈值规则产生，使错误能够追溯到各个处理阶段。在包含30张HRF图像的探索性队列上，我们取得了96.7%的准确率、0.968的F1分数和0.978的ROC-AUC。消融实验以及与VascX的探索性对比研究考察了血管配对、管径测量和聚合方式的影响。受限于小规模队列、代理疾病标签以及缺乏独立参考标准，结果解读存在一定局限性。

    arXiv:2611.14013v1 Announce Type: cross  Abstract: The arteriolar-to-venular diameter ratio (AVR) is a clinically motivated marker of retinal vascular health. We present a five-stage pipeline for AVR estimation and exploratory retinopathy screening from fundus images, with inspectable intermediate outputs at each stage. The pipeline combines vessel and artery-vein masks, optic-disc localization, vessel-calibre estimation using a minimal-chord boundary walk, and AVR computation from dominant arterial and venular branches. An explicit threshold-based rule produces the screening decision, enabling errors to be traced to individual processing stages. On an exploratory cohort of 30 HRF images, we report 96.7% accuracy, an F1-score of 0.968 and ROC-AUC of 0.978. Ablation studies and an exploratory comparison with VascX examine the effects of vessel pairing, calibre measurement and aggregation. Interpretation is limited by the small cohort, proxy disease labels and absence of independent refe
    
[^58]: 编程智能体记忆后训练 II：通过在线策略蒸馏释放预训练文件操作在长时程任务中的记忆潜力

    Coding Agent Memory Post-Training II: Unlocking the Memory Potential of Pre-trained File Operations for Long-Horizon Tasks via On-Policy Distillation

    [https://arxiv.org/abs/2611.14009](https://arxiv.org/abs/2611.14009)

    该论文提出 CAMG-II 长时程智能体强化学习环境套件，通过提供 Bash 访问和回合持久工作区，让记忆控制建立在预训练已有的文件操作能力之上，从而解决在线策略蒸馏中因记忆接口超出预训练范围而导致教师指导无效、学生记忆动作采样不足的问题。

    

    语言模型智能体越来越多地处理交互历史超出模型活跃上下文的长时程任务。因此，它们必须决定记下什么、读回什么，以及何时执行这两种操作。近期的工作开始使用在线策略蒸馏将记忆控制纳入策略之中，但其依赖的记忆工具和压缩协议都不在教师和学生模型的预训练范围之内。然而，这从两个方面削弱了蒸馏效果：教师模型无法就其从未使用过的接口给出有用的指导，而学生模型采样到的有意义的记忆动作太少，使得这些指导难以发挥作用。为了研究这一问题，我们推出了 Coding Agent Memory Gym II（CAMG-II），这是一套涵盖购物、编程、自动研究、通用和视觉领域的长时程智能体强化学习环境套件。在每个环境的原生任务接口之外，CAMG-II 还提供普通的 Bash 访问权限和一个回合内持久的工作区，从而能够……

    arXiv:2611.14009v1 Announce Type: cross  Abstract: Language-model agents increasingly tackle long-horizon tasks whose interaction histories exceed the model's active context. They must therefore decide what to write down and what to read back, and when to do either. Recent work has begun to use on-policy distillation (OPD) to make memory control part of the policy, relying on memory tools and compaction protocols that lie outside either model's pre-training. However, this weakens the distillation on both sides: the teacher cannot give useful guidance on an interface it has never used, and the student samples too few meaningful memory actions for that guidance to take effect. To study this problem, we introduce Coding Agent Memory Gym II (CAMG-II), a suite of long-horizon agentic-RL environments spanning Shop, Coding, AutoResearch, General, and Visual. Alongside each environment's native task interface, CAMG-II provides ordinary Bash access and an episode-persistent workspace, enabling 
    
[^59]: 自主指定隐私：评估大型视觉语言模型的推理时个性化隐私保护能力

    Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Models

    [https://arxiv.org/abs/2611.13997](https://arxiv.org/abs/2611.13997)

    该论文提出了首个系统性评估大型视觉语言模型“推理时个性化隐私保护”（ITP³）能力的基准SPY-Bench，包含32,700个涵盖67个类别和24个真实场景的样本以及基于用户规范与情境感知的新颖指标，旨在使模型能够遵循用户通过语言动态指定的个性化隐私边界。

    

    大型视觉语言模型（LVLMs）已展现出卓越的能力，但由于其能够从图像中以高精度推断敏感个人信息，引发了严重的隐私担忧。虽然当前的LVLMs在保护通用隐私（如信用卡数据）方面已经相对对齐良好，但我们认为隐私本质上是个性化和情境相关的。本工作转向一个新颖的任务：LVLMs能否实现推理时个性化隐私保护（ITP³），即允许用户通过语言规范动态指定隐私边界？为此，我们提出了SPY-Bench，这是首个对ITP³能力进行系统性评估的基准，包含：（1）32,700个独特样本，涵盖67个类别和24个真实世界场景中的图像-问题对和个性化隐私指令；（2）基于用户规范和情境感知的新颖评估指标。（注：原文摘要在此处不完整）

    arXiv:2611.13997v1 Announce Type: cross  Abstract: Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities but raise significant privacy concerns due to their abilities to infer sensitive personal information from images with high precision. While current LVLMs are relatively well aligned to protect universal privacy, e.g., credit card data, we argue that privacy is inherently personalized and context-dependent. This work pivots towards a novel task: can LVLMs achieve Inference-Time Personalized Privacy Protection (ITP$^3$), allowing users to dynamically specify privacy boundaries through language specifications? To this end, we present SPY-Bench, the first systematic assessment of ITP$^3$ ability, which comprises (1) 32,700 unique samples with image-question pairs and personalized privacy instructions across 67 categories and 24 real-world scenarios, and (2) novel metrics grounded in user specifications and context awareness. Benchmarking the ITP$^3$ ability of 
    
[^60]: 超越专家边界理解混合专家语言模型

    Understanding Mixture-of-Experts Language Models Beyond Expert Boundaries

    [https://arxiv.org/abs/2611.13975](https://arxiv.org/abs/2611.13975)

    该研究提出路由感知幅度剖析（RAMP）方法，首次揭示MoE模型中领域专业化计算在神经元层面的组织规律（同领域神经元重叠率高达58-74%，远超跨领域的15-24%），并追踪了这种组织在预训练和后训练全过程中的动态演变。

    

    在混合专家模型中，专家专业化的本质仍然存在争议。关于领域相关专家偏好的报告，与分析将路由主要归因于词元身份或句法模式的研究形成了对立。为了澄清这些不同的观点，我们超越了专家级别的路由，采用一种为此专门开发的方法——路由感知幅度剖析（RAMP）——来研究领域专业化计算在神经元层面的组织方式。使用混合领域校准数据，我们识别出领域专业化的神经元集合：同一领域的多源样本之间重叠率达到58-74%，而跨领域样本之间仅为15-24%。为了理解这种组织结构在训练过程中如何演变，我们在三个完全开源的MoE模型家族中，追踪了整个预训练和后训练过程中的神经元重要性。结果显示，尽管专家级别的重要性相对稳定，神经元重要性在预训练各检查点之间仍持续发生重组。

    arXiv:2611.13975v1 Announce Type: new  Abstract: In Mixture-of-Experts (MoE) models, the nature of expert specialization remains controversial. Reports of domain-associated expert preferences contrast with analyses attributing routing primarily to token identity or syntactic patterns. To clarify these differing accounts, we look beyond expert-level routing to investigate the neuron-level organization of domain-specialized computation using Routing-Aware Magnitude Profiling (RAMP), a measure developed for this analysis. Using mixed-domain calibration data, we identify domain-specialized neuron sets with 58-74% overlap across multi-source samples of the same domain, compared with 15-24% across domains. To understand how this organization evolves during training, we trace neuron importance throughout pre-training and post-training across three fully open MoE families. Neuron importance continues to reorganize between pre-training checkpoints despite comparatively stable expert-level impor
    
[^61]: TransGHOST：眼科显微手术中器械遮挡区域的实时、无合成重建

    TransGHOST: Real-Time, Synthesis-Free Reconstruction of Instrument-Occluded Regions in Ophthalmic Microsurgery

    [https://arxiv.org/abs/2611.13971](https://arxiv.org/abs/2611.13971)

    TransGHOST提出了一种实时、非生成的因果参考引导重建流水线，通过仅基于可见区域拟合的几何变换从历史帧重采样来恢复眼科显微手术中被器械遮挡的区域，并提供逐像素置信度与来源标注，从而避免了生成式修复带来的合成内容与不可追溯问题。

    

    眼科显微手术中的术中场景感知受到器械遮挡的限制：当器械进入高倍放大视野时，它同时遮挡了术者和处理视频的机器人辅助系统感知组件所看到的组织。生成式图像修复可以去除遮挡物，但会用合成内容替换缺失的图像内容，这在视觉来源可追溯性至关重要的情况下并不适用。此外，许多视频修复方法依赖于未来帧，无法在线运行。我们提出了TransGHOST，一种因果的、参考引导的重建流水线，其非生成的安全模式可实时运行。每一个恢复的像素值都是通过使用仅拟合于可见区域的几何变换对已录制的帧进行重采样而获得的。该系统提供逐像素的置信度和来源信息，并将无法支撑（追溯）的像素标记为弃权。我们在一个新的犬类白内障手术兽医临床数据集上，以像素级（精度）对TransGHOST进行了评估。

    arXiv:2611.13971v1 Announce Type: cross  Abstract: Intraoperative scene perception in ophthalmic microsurgery is limited by instrument occlusion: when a tool enters the high-magnification field, it conceals tissue from both the surgeon and the perception components of robot-assisted systems that process the video. Generative inpainting can remove the occluder but replaces the missing image content with synthesis, which is unsuitable when visual provenance matters. Many video inpainters also depend on future frames and cannot operate online. We present TransGHOST, a causal reference-guided reconstruction pipeline whose non-generative safe mode operates in real time. Every restored value is obtained by resampling a recorded frame using geometric transforms fitted only to visible regions. The system provides per-pixel confidence and provenance and marks unsupported pixels as abstentions. We evaluate TransGHOST on a new veterinary clinical corpus of canine cataract surgery with pixel-level
    
[^62]: 基于工具锚定LLM智能体与运行时自适应策略的分类体系归纳受控知识工程

    Controlled Knowledge Engineering for Taxonomy Induction with Tool-Grounded LLM Agents and Runtime-Adaptive Policies

    [https://arxiv.org/abs/2611.13966](https://arxiv.org/abs/2611.13966)

    本文提出TIGA架构，将分类体系归纳形式化为显式状态与受约束动作空间上的迭代决策过程，借助可执行的分类编辑工具、记录调用与理由的记忆机制以及运行时自适应策略，克服LLM自由生成中的实体幻觉与无效层次关系，实现完整、结构有效且语义连贯的自动分类体系构建。

    

    分类体系是一类结构化的知识表示，可支持语义搜索、推荐、产品目录组织以及检索增强生成。从固定的候选实体集合自动构建分类体系，所需的远不止看似合理的文本生成：所得的层次结构必须完整、结构有效且语义连贯。大型语言模型（LLM）在这项任务上前景可观，但迭代式的自由形式生成仍容易出现实体遗漏或幻觉、无效的层次关系以及不一致的组织原则等问题。我们提出了TIGA（Taxonomy Induction with Tool-Grounded Agents，基于工具锚定智能体的分类体系归纳），这是一种知识工程架构，它将分类体系归纳形式化为在显式状态与受约束动作空间上的迭代决策过程。TIGA结合了可执行的分类体系编辑工具、记录工具调用、理由与结果的记忆机制，以及运行时自适应的策略。

    arXiv:2611.13966v1 Announce Type: cross  Abstract: Taxonomies are structured knowledge representations that support semantic search, recommendation, product catalog organization, and retrieval-augmented generation. Their automated construction from a fixed set of candidate entities requires more than plausible text generation: the resulting hierarchy must be complete, structurally valid, and semantically coherent. Large Language Models (LLMs) are promising for this task, but iterative free-form generation remains vulnerable to omitted or hallucinated entities, invalid hierarchical relations, and inconsistent organizational principles. We introduce TIGA (Taxonomy Induction with Tool-Grounded Agents), a knowledge-engineering architecture that formulates taxonomy induction as an iterative decision process over an explicit state and constrained action space. TIGA combines executable taxonomy-editing tools, an memory that records tool calls, rationales, and outcomes, and a runtime-adaptive 
    
[^63]: 从攻击面到防御架构：面向智能体编程工具的组合式安全

    From Attack Surface to Defense Architecture: Compositional Security for Agentic Coding Harnesses

    [https://arxiv.org/abs/2611.13960](https://arxiv.org/abs/2611.13960)

    该论文提出ActiveSecHarness——一种基于“智能与权限分离”原则的神经符号参考监控器架构，通过HMAC能力令牌与二维污染格为AI编程智能体提供结构性的组合式安全防御。

    

    AI编程工具将语言模型与shell、文件系统和网络套接字耦合在一起，使该工具本身成为一个自动化的“就地取材”式攻击平台：一旦对手将指令注入到检索到的内容中，bash、curl和find等双用途二进制程序就会成为攻击载荷。我们主张，智能体系统的安全性是认知架构与权限组合的结构性属性，而非单个模型的涌现属性。静态正则表达式黑名单无法检测结构性规避手段，而基于LLM-as-a-Judge的防护栏不仅延迟高，还会过度拦截良性任务。我们提出了ActiveSecHarness，一个围绕“智能与权限分离”原则构建的神经符号参考监控器架构。它结合了：(1) 跨越四个信任与权限环的、受Macaroon启发的HMAC能力令牌；(2) 一个二维乘积污染格 L = L_trust × L_sens…（原文摘要在此处截断）

    arXiv:2611.13960v1 Announce Type: cross  Abstract: AI coding harnesses couple language models to shells, filesystems, and network sockets, turning the harness into an automated Living-off-the-Land (LotL) platform: once an adversary injects instructions into retrieved content, dual-use binaries such as bash, curl, and find become the payload. We argue that security in agentic systems is a structural property of the cognitive architecture and permission composition rather than an emergent property of individual models. Static regex blacklists miss structural evasions, whereas LLM-as-a-Judge guardrails incur high latency while over-blocking benign work.   We present ActiveSecHarness, a neurosymbolic reference-monitor architecture built around Separation of Intelligence and Authority. It combines (1) Macaroon-inspired HMAC capability tokens across four Trust and Authority Rings, (2) a 2D product taint lattice $\mathcal{L} = \mathcal{L}_{\mathrm{trust}} \times \mathcal{L}_{\mathrm{sens}}$ a
    
[^64]: TreeAgent：面向长层级数据处理的结构感知记忆智能体

    TreeAgent: Structure-Aware Memory Agent for Processing Long Hierarchical Data

    [https://arxiv.org/abs/2611.13946](https://arxiv.org/abs/2611.13946)

    TreeAgent提出了一种结构感知的记忆智能体，利用输入数据的层级结构来指导分段处理与历史信息访问，解决了现有记忆智能体在处理API响应、HTML文档等长层级结构化数据时拆分记录字段、丢失层级信息和兄弟记录细节的问题。

    

    实际的Web应用日益需要智能体来理解长对话或长文档。对于超出大语言模型（LLM）上下文窗口的输入，记忆智能体会按顺序处理分段，并在有限的内存中保留过去的信息。然而，这些方法通常将输入划分为扁平的token分段，而不保留结构或层级，这对API响应或HTML文档等结构化数据造成了三个限制。第一，固定长度的分段可能拆分同一记录的字段，破坏它们之间的关系。第二，分段缺乏层级信息，掩盖了其结构上下文和位置。第三，连续的内存更新可能丢弃较早的兄弟记录的细节，而这些细节是后续分段推理所需的。为解决这些限制，我们提出了TreeAgent，一个结构感知的智能体，它利用输入的层级结构来指导分段上下文处理和对过去信息的访问。

    arXiv:2611.13946v1 Announce Type: new  Abstract: Practical Web applications increasingly require agents to interpret long conversations or documents. For inputs exceeding the context windows of large language models (LLMs), memory agents process segments sequentially and retain past information in bounded memory. However, they typically divide inputs into flat token segments without preserving structure or hierarchy, causing three limitations for structured data such as API responses or HTML documents. First, fixed-length segmentation can split fields of the same record, breaking their relationships. Second, segments lack hierarchical information, obscuring their structural context and positions. Third, successive memory updates may discard details of earlier sibling records needed for reasoning on later segments. To address these limitations, we propose TreeAgent, a structure-aware agent that uses the input hierarchy to guide both segment-wise context processing and access to past inf
    
[^65]: Chem World：一个用于可信分子性质预测的大规模基准与物理感知框架

    Chem World: A Large-Scale Benchmark and Physics-Aware Framework for Trustworthy Molecular Property Prediction

    [https://arxiv.org/abs/2611.13945](https://arxiv.org/abs/2611.13945)

    本文提出了大规模基准Chem World和物理感知组合建模框架PACM，通过整合17个数据集、80余万实例和领域物理约束，为可信的分子性质预测提供了统一的评估平台，并验证了分子语言模型表示的强泛化与可迁移性。

    

    分子性质预测是计算化学和药物制剂研究的基础，然而现有基准在数据集、性质和评估协议方面仍然碎片化，限制了对模型泛化能力的系统性评估。我们提出了Chem World，这是一个大规模基准，整合了17个公开数据集，涵盖超过80万个分子和化学体系实例，横跨10个性质轨道，覆盖单分子和多组分系统，包括药物溶解度。Chem World为跨多种化学性质评估分子表示和组合建模提供了统一的测试平台。基于该基准，我们进一步提出了PACM（物理感知组合建模），它将分子表示与组合感知建模和领域特定的物理约束相结合。大量实验表明，分子语言模型能够提供强大且可迁移的表示，

    arXiv:2611.13945v1 Announce Type: cross  Abstract: Molecular property prediction is fundamental to computational chemistry and drug formulation research, yet existing benchmarks remain fragmented across datasets, properties, and evaluation protocols, limiting systematic evaluation of model generalization. We introduce Chem World, a large-scale benchmark integrating 17 public datasets with over 800,000 molecular and chemical-system instances across 10 property tracks, covering single-molecule and multi-component systems, including drug solubility. Chem World provides a unified testbed for evaluating molecular representations and compositional modeling across diverse chemical properties. Building on the benchmark, we propose PACM (Physics-Aware Compositional Modeling), which combines molecular representations with composition-aware modeling and domain-specific physical constraints. Extensive experiments show that molecular language models provide strong and transferable representations, 
    
[^66]: 在面向元启发式开发的工业流程中利用领域知识定制优化方法

    Tailoring Optimization Approaches with Domain Knowledge in Industry Process for Meta-heuristic Development

    [https://arxiv.org/abs/2611.13940](https://arxiv.org/abs/2611.13940)

    本文提出了元启发式开发的工业流程（IPMD）框架，并展示了一种基于遗传编程的超启发式方法BNTGP，成功应用于求解车辆路径和项目调度这两个NP难多目标优化问题。

    

    本文提出了元启发式开发的工业流程（IPMD）框架，这是一种将元启发式和超启发式方法应用于优化问题的系统化方法论。在此背景下，我们展示了一个平衡非支配锦标赛遗传编程（BNTGP）的案例研究，用于求解两个NP难的多目标优化问题：路径规划类问题（带时间窗的容量约束车辆路径问题）和调度类问题（多技能资源受限项目调度问题）。BNTGP方法基于元启发式设计，其基于遗传编程的表示形式使BNTGP成为一种超启发式方法，因为它能够探索针对一般性问题求解的启发式搜索空间。所生成的启发式方法随后可应用于一系列问题实例。

    arXiv:2611.13940v1 Announce Type: cross  Abstract: The paper presents an Industry Process for Meta-heuristic Development (IPMD) schema, a systematic methodology for applying meta- and hyper-heuristic applications to optimization problems. In this context, we present a case study of Balancing Non-dominated Tournament Genetic Programming (BNTGP) applied to solve two NP-hard multi-objective optimization problems: routing (Capacitated Vehicle Routing Problem with Time Windows) and scheduling (Multi-Skill Resource Constrained Project Scheduling Problem). The BNTGP method is based on a metaheuristic design. Its genetic programming-based representation makes BNTGP a hyper-heuristic, as it enables the exploration of a search space of heuristics for solving problems in general. The resulting heuristic can then be applied across a set of problem instances.
    
[^67]: 英国采用AI赋能门禁监控作为农场安全的论证：一项基于PRISMA的农村犯罪、牲畜捕食与农场通道证据综述

    The Case for AI-Enabled Gate Monitoring as Farm Security in the UK: A PRISMA-Guided Review of the Evidence on Rural Crime, Livestock Predation and Farm Access

    [https://arxiv.org/abs/2611.13933](https://arxiv.org/abs/2611.13933)

    本文通过PRISMA系统性综述发现英国农场安全的证据基础薄弱、农村犯罪损失高昂（2025年达4,150万英镑）且现有安防措施缺乏独立有效性评估，从而论证了采用AI赋能门禁监控作为英国农场安全方案的必要性。

    

    英国农场安全所依据的证据比其成本所暗示的要薄弱。我们进行了一项PRISMA指导的综述：2026年11月2日至4日在Web of Science和Scopus上进行了十二次检索，共返回1,183条记录（761条唯一记录），灰色文献和手工检索又增加了35个评估来源。我们以叙述性方式综合了96条数据库记录和24个灰色文献来源。英国的同行评审证据很少（仅16条记录），而英国主要的犯罪与农民心理健康调查仅有80份回复。成本估算依赖于保险公司的理赔数据：NFU Mutual估计2025年英国农村犯罪损失为4,150万英镑，其中牲畜盗窃案上升近30%至约450万英镑，狗袭击牲畜造成195万英镑损失。内政部表示其中央数据无法单独识别农业犯罪。我们没有发现针对摄像头、传感器、追踪器或标记技术的独立评估；在系统性研究中，灯光和声音威慑装置在一至五个月内便失去了效果……

    arXiv:2611.13933v1 Announce Type: cross  Abstract: Farm security in the UK rests on thinner evidence than its cost suggests. We ran a PRISMA guided review: twelve searches of Web of Science and Scopus on 2nd -4th November 2026 returned 1,183 records (761 unique), and grey-literature and hand-searches added 35 assessed sources. We synthesised 96 database records and 24 grey sources narratively. UK peer-reviewed evidence is small (16 records), and the main UK survey of crime and farmer mental health has 80 responses. Cost estimates rest on insurer claims: NFU Mutual put UK rural crime at {\pounds}41.5 million in 2025, with livestock theft up almost 30% to about {\pounds}4.5 million and dog attacks on livestock costing {\pounds}1.95 million. The Home Office states that agricultural crime cannot be separately identified in its central data. We found no independent evaluation of cameras, sensors, trackers or marking; light and sound deterrents lost effect within one to five months in a syst
    
[^68]: TAPCO-G：基于图物理先验的LLM引导设计自适应时序与功耗协同优化

    TAPCO-G: LLM-Guided Design-Adaptive Timing and Power Co-Optimization with Graph-Based Physical Priors

    [https://arxiv.org/abs/2611.13931](https://arxiv.org/abs/2611.13931)

    TAPCO-G 提出了一个将大语言模型引导的优化器自适应与基于图的物理先验相结合的布局后时序与功耗协同优化框架，通过外层循环调整工具设置并修订修复策略、内层循环基于静态时序分析快照进行可靠的延迟与功耗估计，从而有效解决设计特定的时序与功耗瓶颈。

    

    布局后的时序与功耗协同优化对于满足建立时间目标以及在布局约束内降低功耗至关重要。一次单元替换可能改善某条路径的时序，但同时会增加功耗或减少其他路径的时序裕量。传统优化器遵循固定的规则来调度和接受修复操作，这限制了其应对设计特定瓶颈的能力。大型语言模型（LLM）可以提出对这些规则的修改建议，但仅凭其自身无法可靠地估计单元替换对端点时序裕量和功耗的影响。TAPCO-G 将LLM引导的优化器自适应与基于图的物理先验相结合，以评估候选的单元替换方案。外层循环利用物理设计反馈来调整工具设置，并在调优停滞时修订可执行的修复策略；内层循环将延迟估计锚定于静态时序分析快照，考虑上游驱动器负载的变化，并传播延迟偏移。

    arXiv:2611.13931v1 Announce Type: cross  Abstract: Post-placement timing and power co-optimization is critical to meeting setup timing targets and reducing power within placement constraints. A cell replacement can improve timing along one path while increasing power or reducing slack on others. Conventional optimizers follow fixed rules for scheduling and accepting repairs, limiting their ability to address design-specific bottlenecks. A large language model (LLM) can propose changes to these rules, but on its own it cannot reliably estimate how a cell replacement affects endpoint slack and power. TAPCO-G combines LLM-guided optimizer adaptation with graph-based physical priors to evaluate candidate cell replacements. The outer loop tunes tool settings using physical-design feedback and revises executable repair policies when tuning stalls. The inner loop anchors delay estimates to static timing analysis snapshots, accounts for upstream driver-load changes, and propagates delay shifts
    
[^69]: 超越代理指标：多模态大语言模型作为可解释人工智能评估的人类替代者

    Beyond Proxy Metrics: MLLMs as Human Surrogates for Explainable AI Evaluation

    [https://arxiv.org/abs/2611.13926](https://arxiv.org/abs/2611.13926)

    本文提出一个“基于目的”的评估框架，并证明多模态大语言模型可以作为人类受试者的替代者来自动化评估可解释AI技术，从而解决代理指标信息量不足以及人类评估昂贵、难扩展、难复现的问题。

    

    可解释人工智能（XAI）已经发展成为一个广泛且具有影响力的研究领域，但如何最佳地评估解释技术这一根本性问题却一直难以解决，令人困扰。现有方法通常围绕两种方式展开：其一是代理指标，但这类指标对于解释的实际效用几乎无法提供有价值的参考；其二是人类评估，虽然被普遍视为黄金标准，但成本高昂、难以规模化，且难以复现。在本文中，我们研究了多模态大语言模型（MLLMs）能否作为人类参与者的替代者，以实现对XAI评估的自动化。为此，我们首先提出了一个“基于目的”的评估框架，其中的指标可同时适用于多模态大语言模型和人类用户。随后，我们提供了一些概念性见解，阐述为什么以及何时可以信任这些模型来自动化这一评估过程。最后，我们在五个基于目的的领域（如模型改进等）进行了广泛的实证测试……

    arXiv:2611.13926v1 Announce Type: new  Abstract: Explainable AI (XAI) has expanded into a broad and impactful research area, but the fundamental question of how best to evaluate explanation techniques has proven frustratingly elusive. Methodologies typically revolve around either proxy metrics which tell us little about the utility of an explanation, or human evaluation which---while generally considered the gold standard---is expensive, difficult to scale, and challenging to reproduce. In this paper, we investigate if Multimodal Large Language Models (MLLMs) can act as surrogate human participants to automate the evaluation of XAI. To do this, we first propose a ``purpose-grounded'' evaluation framework with metrics applicable to both MLLMs and human users alike. Then, we offer some conceptual insights as to why and when we can trust these models to automate this process. Lastly, we conduct extensive empirical testing across five purpose-grounded areas such as model improvement and fo
    
[^70]: 域不变与语言模型辅助的键盘声学侧信道攻击

    Domain-Invariant and Language-Model-Assisted Acoustic Side-Channel Attacks on Keyboards

    [https://arxiv.org/abs/2611.13922](https://arxiv.org/abs/2611.13922)

    本文证明，通过让声学模型具备显式域不变性并用语言模型校正含噪预测，声学侧信道攻击在未见过的键盘、多样用户、嘈杂环境和压缩信道等真实开放条件下仍能有效恢复键盘输入内容。

    

    声学侧信道攻击（ASCA）仅通过一个普通麦克风、无需访问受害者设备，即可从键盘敲击声中恢复所输入的文本。先前的研究表明，此类攻击在理想化的实验室条件下可以成功——即单一键盘、固定麦克风、安静房间，且训练和测试在同一设备上进行——但遗留了一个对安全至关重要的开放性问题：键盘敲击推断能否在更开放的真实环境中依然有效，包括未见过的键盘、不同的用户、嘈杂的环境以及经过压缩的传输信道。本文给出的答案是肯定的，前提是使声学模型具备显式的域不变性，并利用语言模型对其含噪的预测结果进行校正。

    arXiv:2611.13922v1 Announce Type: cross  Abstract: Acoustic side-channel attacks (ASCA) recover typed text from the sound of keystrokes, using only a commodity microphone and no access to the victim's device. Prior work establishes that such attacks succeed under idealized laboratory conditions - a single keyboard, a fixed microphone, a quiet room, and training and testing on the same device but leaves open the question that matters for security: whether keystroke inference survives the open world of unseen keyboards, diverse users, noisy environments, and compressed transmission channels. The answer developed here is that it does, provided the acoustic model is made explicitly domain-invariant and its noisy predictions are corrected by a language model.
    
[^71]: CAppleNet：基于坐标注意力的YOLO11用于苹果果梗凹陷与花萼的实时检测

    CAppleNet: A Coordinate Attention-Based YOLO11 for Real-Time Apple Stem Cavity and Calyx Detection

    [https://arxiv.org/abs/2611.13920](https://arxiv.org/abs/2611.13920)

    该论文提出了CAppleNet，一种基于坐标注意力增强的YOLO11模型，在自建的1506张苹果图像数据集上实现了果梗凹陷与花萼等细粒度形态标志点的实时高精度检测，并通过系统性消融实验确定了关键改进因素。

    

    自动化水果分级、分拣和定向系统需要对水果级形态标志点（而不仅仅是水果边界）进行实时、准确的检测。本文提出了一个完全自采自标注的数据集，包含1506张苹果图像（1054张训练、301张验证、151张测试），这些图像在受控的固定相机条件下采集，并针对三个目标类别进行了标注：果实本体、果梗凹陷（萼洼）和花萼。研究采用默认超参数训练的YOLO11s检测器作为基线，并开展了系统性的单因素消融实验，涵盖输入分辨率、数据增强策略、优化器配置和轻量化架构增强等方面，以找出在品种间颜色和纹理高度差异的条件下，最能改善这些细粒度标志点检测效果的关键因素。实验表明，无论是提高输入分辨率，还是测试的六种数据增强变体，抑或是手动调整优化器参数，都未能……（原摘要在此处被截断）

    arXiv:2611.13920v1 Announce Type: cross  Abstract: Automated fruit grading, sorting, and orientation systems require real-time, accurate detection of fruit-level morphological landmarks rather than the fruit boundary alone. This paper presents a custom, fully self-collected and self-annotated dataset of 1506 apple images (1054 training, 301 validation, 151 test) captured under controlled, fixed-camera conditions and annotated for three target classes: fruit body, stem cavity, and calyx. A YOLO11s detector trained with default hyperparameters is adopted as the baseline, and a systematic, one-factor-at-a-time ablation study is conducted across input resolution, data augmentation policy, optimizer configuration, and lightweight architectural enhancement to identify the factors that most improve detection of these fine-grained landmarks under high inter-cultivar color and texture variability. Neither increased input resolution, nor any of six tested augmentation variants, nor manual optimi
    
[^72]: 从反事实编辑到人类叙事：基于大语言模型的推荐系统解释诠释

    From Counterfactual Edits to Human Narratives: LLM-Driven Interpretations for Recommender System Explanations

    [https://arxiv.org/abs/2611.13914](https://arxiv.org/abs/2611.13914)

    该论文提出利用大语言模型作为中介，将推荐系统反事实解释中冗长且难以理解的用户交互历史编辑列表，总结为描述高层偏好模式的简洁自然语言解释，从而弥合生成有效反事实解释与向用户传达模型行为之间的差距。

    

    推荐系统日益影响着用户在网上发现和消费的内容，但其输出背后的决策逻辑往往仍不透明。反事实解释（CFX）方法提供了一种有效途径，可以展示用户交互历史中的哪些改变会改变推荐结果。然而，在推荐场景中，CFX 通常需要多次编辑，产生冗长的、低层次（细粒度）的编辑列表，用户难以理解。此外，大多数评估侧重于编辑数量，而在很大程度上忽略了生成的 CFX 对用户来说是否连贯且可操作。我们认为，生成有效的 CFX 与传达模型行为之间的这一差距，源于缺乏以用户为中心的抽象以及有限的定性评估。为弥合这一差距，我们研究将大语言模型作为中介，把原始的 CFX 交互编辑列表总结为描述更高层次偏好模式的简洁自然语言解释。

    arXiv:2611.13914v1 Announce Type: new  Abstract: Recommender systems increasingly influence what users discover and consume online, yet the decision logic behind their outputs often remains opaque. Counterfactual explanation (CFX) methods provide an effective way to show which changes to a user's interaction history would alter a recommendation. In recommendation settings, however, CFXs often require many edits, producing long, low-level lists that are hard for users to interpret. Moreover, most evaluations emphasize edit counts while largely overlooking whether the resulting CFXs are coherent and actionable for users. We argue that this gap between generating valid CFXs and communicating model behavior stems from the lack of user-centered abstraction and limited qualitative evaluation. To bridge this gap, we investigate large language models as mediators that summarize raw CFX interaction edit lists into concise natural language explanations describing higher-level preference patterns
    
[^73]: MicroBT：面向多模态微型脑肿瘤分割与计数的质心引导协同学习

    MicroBT: Centroid-Guided Synergistic Learning for Multi-Modal Micro Brain Tumor Segmentation and Counting

    [https://arxiv.org/abs/2611.13912](https://arxiv.org/abs/2611.13912)

    该论文提出MicroBT统一框架，巧妙利用“病灶质心具有尺度不变性”这一几何特性，通过质心锚定的协同学习范式，解决多模态脑肿瘤分割中小于3毫米微小病灶被抑制以及病灶计数不可靠的问题。

    

    多模态脑肿瘤分割对于临床诊断、放疗规划以及纵向监测至关重要。尽管近期研究已取得显著的总体分割性能，但它们依赖基于重叠度的优化目标，这类目标天然偏向大区域，导致小于3毫米的微小病灶在训练过程中被系统性抑制。此外，现有的像素级分割框架缺乏显式的实例级建模，导致病灶计数不可靠，而计数结果直接影响临床分期与治疗决策。为应对这些挑战，我们提出了MicroBT，一个用于微型脑肿瘤分割与计数的统一多模态框架。我们的核心洞察是一种基本几何属性具有天然的尺度不变性：无论大小如何，每个病灶都拥有明确定义的质心。基于这一洞察，我们首先设计了一种质心锚定的协同范式，利用预测的质心……（原文此处截断）

    arXiv:2611.13912v1 Announce Type: cross  Abstract: Multi-modal brain tumor segmentation is essential for clinical diagnosis, radiotherapy planning, and longitudinal monitoring. While recent studies have achieved remarkable overall segmentation performance, they rely on overlap-based objectives that inherently favor large regions, causing sub-3 mm micro-lesions to be systematically suppressed during training. Moreover, existing pixel-level segmentation frameworks lack explicit instance-level modeling, leading to unreliable lesion counting that directly impacts clinical staging and therapeutic decisions. To address these challenges, we propose MicroBT, a unified multi-modal framework for micro brain tumor segmentation and counting. Our core insight is that a fundamental geometric property is inherently scale-invariant: regardless of size, each lesion possesses a well-defined centroid. Leveraging this, we first devise a centroid-anchored synergistic paradigm that utilizes predicted centro
    
[^74]: 用于脑转移瘤分割的区域级Logit融合与效用门控精细化

    Region-Wise Logit Fusion and Utility-Gated Refinement for Brain Metastasis Segmentation

    [https://arxiv.org/abs/2611.13884](https://arxiv.org/abs/2611.13884)

    该论文提出COMPASS-METS框架，通过融合多个nnU-Net配置的区域概率集成、结构化概率锚点与效用门控精细化，在保持ET/TC/WT/RC层级结构的同时有效检测脑转移瘤的小强化病灶。

    

    BraTS-METS 2026 任务1评估在异构多参数MRI上对脑转移瘤及术后切除腔的分割，要求在检测小的强化病灶的同时，保持强化肿瘤（ET）、肿瘤核心（TC）、全肿瘤（WT）和切除腔（RC）的几何形状与层级结构。我们提出了COMPASS-METS（组件感知的多训练者概率组装与结构化分割），它将三个五折nnU-Net配置组合成对齐的区域概率集成。结构化概率锚点应用区域特定的组件决策、TC边界补全、RC过滤以及保持层级的转换。被遗漏的ET候选区域利用学习到的组件置信度以及针对ET及其TC/WT父区域的确定性多训练者支持进行加性恢复；最终的效用门控仅接受具有足够病灶存在概率和几何安全概率的不连通候选区域。

    arXiv:2611.13884v1 Announce Type: cross  Abstract: BraTS-METS 2026 Task 1 evaluates segmentation of brain metastases and postoperative resection cavities on heterogeneous multiparametric MRI, requiring detection of small enhancing foci while preserving the geometry and hierarchy of enhancing tumor (ET), tumor core (TC), whole tumor (WT), and resection cavity (RC). We present COMPASS-METS (Component-Aware Multi-Trainer Probability Assembly with Structured Segmentation), which combines three five-fold nnU-Net configurations into aligned regional probability ensembles. A structured-probability anchor applies region-specific component decisions, TC boundary completion, RC filtering, and hierarchy-preserving conversion. Omitted ET candidates are recovered additively using learned component confidence and deterministic multi-trainer support for ET and its TC/WT parents; a final utility gate admits only disconnected candidates with sufficient lesion-existence and geometry-safety probabilities
    
[^75]: 面向思维链可监控性的透明度披露

    Transparency Disclosures for Chain-of-Thought Monitorability

    [https://arxiv.org/abs/2611.13873](https://arxiv.org/abs/2611.13873)

    该论文提出覆盖思维链访问、质量（自然性与忠实性）和监控有效性（检测与能力提升）的五个透明度披露指标，呼吁前沿AI开发者定期报告这些指标以保护思维链监控能力免受威胁，审计显示目前尚无开发者覆盖全部五个指标。

    

    思维链（CoT）监控是监控先进AI系统最有价值的工具之一，但它正面临威胁。我们认为，前沿AI开发者应当报告特定的指标，使这些威胁变得可追踪，并更好地促进保护思维链可监控性的协调合作。我们识别出五个既高度相关又合理可行的披露指标，思维链监控所依赖的五步链条中每个环节各对应一个指标。第一个指标——第三方思维链访问权限与基础架构规范——同时也为其余指标的外部验证提供支撑。其余指标涵盖思维链质量（自然性与忠实性）和监控有效性（检测能力与能力提升）。对Anthropic、OpenAI和Google DeepMind当前披露情况的审计显示，没有任何开发者覆盖了全部指标，但每个指标都至少被一家前沿开发者报告过。由于当前的披露主要依赖于开发者的……（原文摘要在此处截断）

    arXiv:2611.13873v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) monitoring is one of the most valuable tools for monitoring advanced AI systems, but it is under threat. We argue that frontier AI developers should report specific indicators to make these threats trackable and to better facilitate coordination to preserve CoT monitorability. We identify five indicators that are both highly relevant and reasonably feasible to disclose, one for each link in a five-step chain on which CoT monitoring depends. The first, third-party CoT access and base architecture specifications, also underpins external verification of the rest. The others span CoT quality (naturalness and faithfulness) and monitoring effectiveness (detection and uplift). An audit of current disclosures by Anthropic, OpenAI, and Google DeepMind shows that no developer covers the full set, but that every indicator has been reported by at least one frontier developer. Because current disclosures rely largely on devel
    
[^76]: 深度神经网络的机制可解释性：方法、框架与视角

    Mechanistic Interpretability of Deep Neural Networks: Methods, Frameworks, and Perspectives

    [https://arxiv.org/abs/2611.13868](https://arxiv.org/abs/2611.13868)

    本文系统综述了深度神经网络机制可解释性的方法与框架，其核心在于通过识别对特定行为具有因果影响的内部组件来揭示网络的内部工作机制。

    

    深度神经网络支撑着图像、音频和语言处理等众多信号领域的基础任务，然而其内部机制却极难解释。内部激活是跨层变换的结构化信号，可能编码大量相互依赖的关系，因为单个神经元常常代表多个潜在概念。机制可解释性旨在通过识别其激活对特定行为具有因果影响的组件，来揭示神经网络的内部机制。在此背景下，本文对用于分析深度神经网络因果推理的机制可解释性研究进展进行了综述。

    arXiv:2611.13868v1 Announce Type: cross  Abstract: Deep neural networks underpin fundamental tasks in a wide range of signal domains, such as image, audio, and language processing. However, their internal mechanisms are notably difficult to interpret. Internal activations are structured signals transformed across layers that may encode a multitude of interdependent relationships, as individual neurons frequently represent multiple underlying concepts. Mechanistic interpretability aims to uncover the internal mechanisms of neural networks by identifying components whose activations are causally responsible for specific behaviors. In such a context, this article provides an overview of advances in mechanistic interpretability for the analysis of causal inference in deep neural networks.
    
[^77]: 与分类体系无关的机器人运动学神经符号接地

    Taxonomy-Agnostic Neurosymbolic Grounding of Robot Kinematics

    [https://arxiv.org/abs/2611.13850](https://arxiv.org/abs/2611.13850)

    该论文提出URDF2Ontology框架，利用LLM语义推理与形态感知的双轨架构，自动将原始URDF文件转换为与任意用户分类体系对齐的填充式OWL本体，为机器人物理能力的符号推理提供接地。

    

    虽然用于任务分配的符号推理需要对机器人的物理能力进行接地，但标准的统一机器人描述格式（URDF）文件仅提供低层运动学信息，缺乏推理所需的正式语义上下文。我们提出了URDF2Ontology，一个自动化的、与分类体系无关的神经符号框架，能够将原始URDF文件转换为填充完整且已接地的OWL本体。该框架利用大型语言模型（LLM）进行有针对性的语义推理，并结合迭代式顺序图清除机制以防止上下文窗口饱和。为了处理复杂的运动学结构，它采用了形态感知的双轨架构，包括用于分支组件的自底向上“树轨道”以及用于串联肢体的自顶向下“链轨道”。在神经符号护栏的引导下，该框架提取功能子图，并断言生成与任意用户提供的分类体系对齐的填充本体。

    arXiv:2611.13850v1 Announce Type: new  Abstract: While symbolic reasoning for task allocation requires grounding a robot's physical capabilities, standard Unified Robot Description Format (URDF) files provide only low-level kinematics without the formal semantic context needed for reasoning. We present URDF2Ontology, an automated, taxonomy-agnostic neurosymbolic framework that transforms raw URDF files into populated, grounded OWL ontologies. It leverages a Large Language Model (LLM) for targeted semantic reasoning, coupled with an iterative sequential graph-purging mechanism to prevent context window saturation. To process complex kinematic structures, it employs a morphology-aware dual-track architecture featuring a bottom-up Tree Track for branching assemblies and a top-down Chain Track for serial limbs. Guided by neurosymbolic guardrails, the framework extracts functional sub-graphs and asserts populated ontologies aligned with arbitrary user-provided taxonomies. Empirical evaluati
    
[^78]: 卫星预训练并非一概而论：通用与对地观测基础模型在场景分类任务上的配对对比研究

    Satellite Pretraining Is Not One Thing: A Matched-Pair Study of Generic and Earth-Observation Foundation Models for Scene Classification

    [https://arxiv.org/abs/2611.13849](https://arxiv.org/abs/2611.13849)

    该研究通过五组共享目标与架构的编码器配对实验和显著性检验发现，对地观测预训练并不必然优于通用预训练：它显著提升了SatDINO、Scale-MAE、RemoteCLIP等模型的场景分类性能，却显著损害了DINOv3-SAT等另一部分模型的表现。

    

    地理空间基础模型是在卫星与航空影像上进行预训练的，但这类编码器在少标签土地利用/土地覆盖制图任务中是否优于强大的通用编码器，目前仍不清楚，因为以往的对比研究很少将预训练数据单独隔离出来加以考察。我们在一个包含14个冻结编码器的基准上，研究了五组共享训练目标与架构的对地观测（EO）与通用编码器配对组合（DINO/SatDINO、DINOv3/DINOv3-SAT、MAE/SatMAE、MAE/Scale-MAE、CLIP/RemoteCLIP），并覆盖五个RGB场景分类数据集（地面采样距离0.06–30米）。在采用原生预处理条件下，我们在六个低标签数据划分上评估了k-NN、线性探针、注意力探针以及图像检索，并使用逐图像配对的显著性检验。结果表明，对地观测预训练并非普遍有益：在大多数数据集和指标上，它显著提升了三组配对的表现（SatDINO、Scale-MAE、RemoteCLIP，其中后者仅在未见过的数据集上评估），却显著损害了两组配对的表现（线性探针下：DINOv3-SAT最多降低……[摘要此处被截断]）

    arXiv:2611.13849v1 Announce Type: cross  Abstract: Geospatial foundation models are pretrained on satellite and aerial imagery, but whether such an encoder beats a strong generic one for few-label land-use/land-cover mapping remains unclear, as prior comparisons rarely isolate the pretraining data. We study five matched pairs of Earth-observation (EO) and generic encoders sharing objective and architecture (DINO/SatDINO, DINOv3/DINOv3-SAT, MAE/SatMAE, MAE/Scale-MAE, CLIP/RemoteCLIP) within a benchmark of 14 frozen encoders on five RGB scene-classification datasets (0.06-30 m GSD). With native preprocessing, we evaluate k-NN, linear and attentive probes and image retrieval over six low-label splits, using paired per-image significance tests. EO pretraining is not uniformly helpful: on most datasets and metrics, it significantly improves three pairs (SatDINO, Scale-MAE, RemoteCLIP, the latter evaluated only on unseen datasets) and significantly hurts two (linear probe: DINOv3-SAT by up t
    
[^79]: PolyRoute：一个面向类型化决策路由的十二语言与混码文本基准，附带基线与排行榜

    PolyRoute: A Twelve-Language and Code-Mixed Benchmark for Typed-Decision Routing, with Baselines and a Leaderboard

    [https://arxiv.org/abs/2611.13843](https://arxiv.org/abs/2611.13843)

    该论文发布了PolyRoute基准，涵盖12种语言、9种书写系统及混码文本的类型化决策路由评测，并提供13个系统的排行榜，显示微调模型可达96.6%准确率，且多语言模型对未公开语言具有良好的泛化能力。

    

    类型化决策编码器可以在一次前向传播中回答关于文本的问题，但其在英语之外的表现很少在统一的评测标准下得到衡量。我们发布了PolyRoute基准：包含12种语言、9种书写系统的322条母语支持消息，156条混码或罗马化消息（印英混合语、西英混合语、阿拉伯罗马化文本），967条可选的母语训练数据行，以及一个三方部门路由决策的评测协议。标签由Qwen3.8-27B生成并经过盲审二次筛选（91%的候选被保留）；独立模型gemma4-31b在97.2%的测试项上与之一致，且任意测试文本与训练文本之间的最大字符5-gram相似度仅为0.33。13个系统的排行榜显示：英语根检查点得分为61.2%，官方发布的多语言版本为77.3%，最佳微调模型达到96.6%；TF-IDF结合逻辑回归达到91.9%。在四个未公开的语言上，三个多语言系统分别取得77%、90%和93%的成绩。在混码文本上……

    arXiv:2611.13843v1 Announce Type: cross  Abstract: Typed-decision encoders answer a question about a text in one forward pass, but their behaviour outside English is rarely measured on a fixed yardstick. We release PolyRoute: 322 native-language support messages in 12 languages and 9 writing systems, 156 code-mixed or romanised messages (Hinglish, Spanglish, Arabizi), 967 optional native training rows, and a protocol for a three-way department-routing decision. Labels come from Qwen3.8-27B with a blind second pass (91% of candidates kept); an independent model, gemma4-31b, agrees on 97.2% of test items, and the largest character 5-gram similarity between any test text and any training text is 0.33. A leaderboard of 13 systems gives 61.2% for the English-root checkpoint, 77.3% for the shipped multilingual one and 96.6% for the best fine-tune; TF-IDF with logistic regression reaches 91.9%. On four withheld languages the three multilingual systems score 77%, 90% and 93%. On code-mixed tex
    
[^80]: MusicFlux：用于文本引导音乐生成与编辑的灵活条件化方法

    MusicFlux: Flexible Conditioning for Text-Guided Music Generation and Editing

    [https://arxiv.org/abs/2611.13834](https://arxiv.org/abs/2611.13834)

    MusicFlux是一个基于预训练流匹配文本到音频模型的文本引导音乐编辑器，通过两种条件化方法在编辑质量与计算效率之间实现权衡，无需处理过长的音频序列即可统一音乐的生成与编辑。

    

    音乐编辑是在保留录音原始属性的同时对其进行变换。近来的方法通过使用上下文音频进行条件化，使预训练的文本到音乐模型适应修复、事件插入或恢复等任务。最近在图像编辑领域提出了一种通过将源图像视为上下文信息来统一生成与编辑的方法。然而，由于音频序列较长，直接将该方法应用于音乐会导致计算成本高昂。在本工作中，我们开发了MusicFlux，一个基于高效预训练流匹配文本到音频模型的文本引导音乐编辑器。我们引入了两种在编辑性能与计算效率之间进行权衡的条件化方法。实验表明，这两种方法均与强大的基线方法具有竞争力，其中一种实现了更好的编辑质量，另一种则保留了原始序列长度并显著提升了计算效率。

    arXiv:2611.13834v1 Announce Type: cross  Abstract: Music editing transforms existing recordings while preserving their original attributes. Recent approaches adapt pretrained text-to-music models for tasks like inpainting, event insertion, or restoration by conditioning them with context audio. It was recently proposed for image editing to unify generation and editing by treating the source image as in-context information. However, applying this directly to music is computationally expensive due to long audio sequences. In this work, we develop MusicFlux, a text-guided music editor built on an efficient pretrained flow-matching text-to-audio model. We introduce two conditioning approaches that trade off editing performance and computational efficiency. Experiments show that both are competitive with strong baselines, with one achieving better editing quality and the other preserving the original sequence length and substantially improving computational efficiency.
    
[^81]: 《此非烟斗——利用进化计算为抽象与推理语料库构建符号求解器》

    Ceci n'est pas une pipe -- Toward a Symbolic Solver for the Abstraction and Reasoning Corpus Using Evolutionary Computation

    [https://arxiv.org/abs/2611.13831](https://arxiv.org/abs/2611.13831)

    该论文提出了首个提交至ARC-AGI-2主竞赛的完全符号化、基于进化计算的程序合成求解器，并着力解决进化算法在领域特定语言表示选择与等价候选解区分方面的固有难题。

    

    arXiv:2611.13831v1 公告类型：新论文 摘要：抽象与推理语料库（ARC）是一个旨在评估人工智能（AI）系统从少量样本出发进行泛化能力的基准测试，而这类任务正是人类所擅长的。自2020年以来举办的ARC-AGI竞赛一直是激烈角逐的焦点，参赛者提出的方法涵盖从传统符号AI到Transformer架构等多种途径。尽管进化算法（EA）有着悠久且成功的应用历史，但在主竞赛中出现的方法中却明显缺少进化算法的身影。作为“评分—搜索”式的程序合成方法，进化算法面临两类问题：一是在应用于该问题的领域特定语言中选择合适的表示方式（例如像素级或对象级）较为困难；二是难以区分产生相同输出的两个候选解。在这项工作中，我们提出了首个提交至ARC-AGI-2主竞赛的完全符号化的、基于进化的程序合成方法……

    arXiv:2611.13831v1 Announce Type: new  Abstract: The Abstraction and Reasoning Corpus (ARC) is a benchmark designed to assess the capability of Artificial Intelligence (AI) systems to generalize starting from a few samples, a task humans excel at. The ARC-AGI contests held since 2020 have been the focus of intense competition, with participants proposing approaches ranging from traditional symbolic AI to Transformer architectures. Despite a long history of successes, evolutionary algorithms (EAs) were notably absent from the approaches appearing in the main contest. As score-and-search program synthesis approaches, EAs suffer from issues related to choosing the proper representation in the Domain-Specific Language applied to the problem (for example, pixel-level or object-level), and distinguishing between two candidate solutions producing the same outputs. In this work, we present the first fully symbolic, evolutionary-based program synthesis approach submitted to the ARC-AGI-2 main c
    
[^82]: CASCADE：面向逻辑异常检测的约束感知支架式分解方法

    CASCADE: Constraint-Aware SCAffolded DEcomposition for Logical Anomaly Detection

    [https://arxiv.org/abs/2611.13820](https://arxiv.org/abs/2611.13820)

    论文提出CASCADE方法，通过约束感知的支架式分解为逻辑异常检测中的每类约束定制视觉证据提取，弥合了LVLM清单式验证中“约束特定感知鸿沟”所导致的实例混淆与尺度漂移问题。

    

    逻辑异常检测（LAD）要求在仅有少量正常参考图像的条件下，验证查询图像是否满足所有类别特定的约束，例如物体数量、配对关系和测量范围。近期基于大型视觉-语言模型（LVLM）的LAD方法将该任务表述为清单式验证，把决策分解为独立的文本检查项。然而，这种分解仅停留在文本层面：每个检查项仍然读取同一张未经修改的查询图像，迫使LVLM必须从同一张原始图像中提取不同形式的视觉证据。我们将这种文本-图像不匹配识别为“约束特定的感知鸿沟”，它是两种失败模式的根源：绑定约束下的实例混淆，以及校准约束下的尺度漂移。为了诊断这些失败究竟源于推理不足还是感知不足，我们在无训练设置下比较了文本推理引导与视觉感知引导……（原文摘要在此处截断）

    arXiv:2611.13820v1 Announce Type: cross  Abstract: Logical Anomaly Detection (LAD) requires verifying whether all category-specific constraints, such as object counts, pairing relations, and measurement ranges, are satisfied by a query image given only a few normal references. Recent LAD methods based on Large Vision-Language Models (LVLMs) formulate this as checklist verification, decomposing the decision into independent textual checks. This decomposition, however, is text-only. Every item still reads the same unmodified query image, so the LVLM must extract different forms of visual evidence from one raw image. We identify this text-image mismatch as a constraint-specific perception gap, the source of two failure modes: instance conflation for binding constraints and scale drift for calibration constraints. To diagnose whether these failures stem from insufficient reasoning or insufficient perception, we compare textual reasoning guides and visual perception guides in a training-fre
    
[^83]: Swanada：分裂编码器能力路由的 Conformer CTC 与跨十二种文字的低于 30 MB 端侧语音识别

    Swanada: Split-Encoder Capability-Routed Conformer CTC and Sub-30 MB On-Device Speech Recognition Across Twelve Scripts

    [https://arxiv.org/abs/2611.13814](https://arxiv.org/abs/2611.13814)

    Swanada 提出了一种分裂编码器、能力路由的 Conformer CTC 架构，将覆盖十二种文字的开源流式端侧语音识别模型压缩到每个低于 30 MB，同时解决了浏览器与移动端部署中的存储配额、算子兼容、解码闪烁和音频管线四大系统瓶颈。

    

    面向印度语系及古典文字的客户端自动语音识别（ASR）可以消除云端转写成本、网络尾延迟和服务器端音频留存问题，但神经模型在浏览器和移动运行时中导出时面临四个系统瓶颈：（1）单体式逐语言 Conformer 检查点每种文字需要约 177 MiB（12 种文字合计超过 2.1 GiB），会耗尽 CacheStorage 配额；（2）静态相对位置编码表嵌入 10.2-20.5 MB 的 FP32 常量，而 1x1 Conv1d 层在 WebAssembly SIMD 中会触发缓慢或不受支持的 ConvInteger 内核；（3）滑动窗口 CTC 在边界标记上出现闪烁；（4）移动端 WebAudio 管线存在 44.1/48 kHz 采样率不匹配问题，且在 iOS WKWebView 下 AudioWorklet 会出现静默卡顿。我们提出 Swanada，一个开源的 16 kHz 流式 ASR 系统，覆盖古典梵语、印度英语和十种印度语言（sa、en、hi、kn、mr、bn、ta、te、ml、gu、pa、or），跨越……（摘要在此处被截断）

    arXiv:2611.13814v1 Announce Type: cross  Abstract: Client-side automatic speech recognition (ASR) across Indic and Classical scripts eliminates cloud transcription costs, network tail latency, and server-side audio retention, yet neural exports face four systems bottlenecks in browsers and mobile runtimes: (1) monolithic per-language Conformer checkpoints require ~177 MiB per script (> 2.1 GiB across 12 scripts), exhausting CacheStorage quotas; (2) static relative positional encoding tables embed 10.2-20.5 MB of FP32 constants while 1x1 Conv1d layers trigger slow or unsupported ConvInteger kernels in WebAssembly SIMD; (3) sliding-window CTC flickers on boundary tokens; and (4) mobile WebAudio pipelines suffer 44.1/48 kHz sample-rate mismatches and silent AudioWorklet stalls under iOS WKWebView.   We present Swanada, an open-source 16 kHz streaming ASR system spanning Classical Sanskrit, Indian English, and ten Indian languages (sa, en, hi, kn, mr, bn, ta, te, ml, gu, pa, or) across bro
    
[^84]: 放宽跨模态对应关系的多模态遥感预训练

    Relaxing Cross-Modal Correspondence for Multimodal Remote Sensing Pretraining

    [https://arxiv.org/abs/2611.13807](https://arxiv.org/abs/2611.13807)

    该论文提出 SemPair 预训练策略，将多模态遥感预训练中的跨模态对应要求从实例级放宽到语义级，用类别一致的跨实例配对替代严格配对，并通过潜在因子模型理论分析和 MixPair 插值方法系统研究对应关系放宽的效果。

    

    多模态遥感预训练通常依赖于跨模态之间精确的实例级对应关系，隐含地将严格配对的观测数据视为学习可迁移表示的先决条件。在这项工作中，我们通过将跨模态对应关系从实例级放宽到语义级来重新审视这一假设。我们提出了语义配对（SemPair），这是一种配对放宽的预训练策略，用类别一致的跨实例配对来替代精确的实例配对。为了刻画对应关系放宽的影响，我们构建了一个潜在因子模型，将跨模态依赖分解为共享语义结构和实例特定相关性，并表明 SemPair 在期望意义上保留前者，同时边缘化实例特定的跨模态相关性。我们进一步提出了混合配对（MixPair），作为语义配对和精确配对之间的受控插值方法，用以研究对应关系放宽……

    arXiv:2611.13807v1 Announce Type: cross  Abstract: Multimodal remote sensing pretraining commonly relies on exact instance-level correspondence across modalities, implicitly treating strictly paired observations as a prerequisite for learning transferable representations. In this work, we revisit this assumption by relaxing cross-modal correspondence from the instance level to the semantic level. We introduce Semantic Pairing (SemPair), a pairing-relaxed pretraining strategy that replaces exact instance pairs with class-consistent cross-instance pairs. To characterize the effect of correspondence relaxation, we formulate a latent-factor model that separates cross-modal dependence into shared semantic structure and instance-specific correlation, showing that SemPair preserves the former while marginalizing instance-specific cross-modal correlations in expectation. We further introduce Mixed Pairing (MixPair) as a controlled interpolation between semantic and exact pairing to study corre
    
[^85]: 融合图学习与成对学习用于从单细胞RNA测序数据推断基因调控网络

    Bridging Graph and Pairwise Learning for Gene Regulatory Network Inference from scRNA-seq Data

    [https://arxiv.org/abs/2611.13801](https://arxiv.org/abs/2611.13801)

    提出scGPL端到端统一框架，融合图学习与成对学习两类方法的优势，联合建模全局调控拓扑与TF-靶基因对，解决了图方法易过拟合、成对方法忽视全局拓扑结构的局限，实现更准确的基因调控网络推断。

    

    动机：基因调控网络（GRNs）捕捉转录因子（TFs）调控基因的潜在相互作用。基于基因表达数据，现有的GRN推断方法通常分为两类：基于图的方法，将GRN建模为整体图；以及基于成对的方法，将GRN分解为单个TF-靶基因对进行建模。然而，每种方法所存在的局限性恰恰是其对应方法的优势所在。基于图的方法由于依赖单一训练图而容易过拟合，而基于成对的方法则受益于丰富的TF-靶基因训练对。相反，基于成对的方法忽略了全局拓扑结构，而全局拓扑结构对于全面捕捉调控机制至关重要。结果：为了解决这些局限性，我们提出了scGPL，一个端到端的统一框架，联合建模全局调控拓扑

    arXiv:2611.13801v1 Announce Type: cross  Abstract: Motivation: Gene regulatory networks (GRNs) capture the underlying interactions through which transcription factors (TFs) regulate genes. Based on gene expression data, existing GRN inference approaches generally fall into two categories: graph-based methods, which model the GRN as a whole graph, and pairwise-based methods, which decompose the GRN into individual TF-target gene pairs for modeling. However, each approach exhibits limitations that are precisely the strengths of its counterpart. Graph-based methods tend to overfit due to their reliance on a single training graph, whereas pairwise-based methods benefit from abundant TF-target training pairs. In contrast, pairwise-based methods overlook the global topological structure, which is essential for comprehensively capturing regulatory mechanisms. Results: To address these limitations, we propose scGPL, an end-to-end unified framework that jointly models global regulatory topology
    
[^86]: 集成动作前缀预测与跨模态对齐的多模态强化学习

    Integrating Action Prefix Prediction and Alignment Across Modalities for Multimodal Reinforcement Learning

    [https://arxiv.org/abs/2611.13782](https://arxiv.org/abs/2611.13782)

    提出改进预测表征方法，将动作前缀预测与基于观测锚点的跨模态对比对齐相结合，直接从观测锚点和动作序列预测未来表征，在DeepMind Control Suite五个任务上平均回报显著优于基线。

    

    在交互数据有限的情况下，从多模态观测中学习表征是强化学习中的一大难题。多模态联合预测表征将多模态潜在预测与跨模态对比学习相结合，但其递归式展开依赖于早期预测，且其成对对齐仅关注模态之间的对应关系。我们提出了改进预测表征方法，该方法基于多模态对比学习，将动作前缀预测与观测锚点处的对齐相结合。在每个预测时域，它直接从观测锚点和相应的动作序列预测未来表征。我们评估了两种裁剪配置：最小保留裁剪面积为10%的IPR-S和50%的IPR-H。在五个DeepMind Control Suite任务上，IPR-H在使用视觉和深度图像时的平均回报比基线高出10.0%，并高出3.4

    arXiv:2611.13782v1 Announce Type: new  Abstract: Learning representations from multimodal observations is difficult in reinforcement learning with limited interaction data. Multimodal Joint Predictive Representation combines multimodal latent prediction with contrastive learning across modalities, but its recursive rollout depends on earlier predictions and its pairwise alignment focuses on correspondence between modalities. We propose Improved Predictive Representation, which combines prediction from action prefixes with alignment at the observed anchor based on Contrastive Multimodal learning. At each horizon, it predicts future representations directly from the observed anchor and the corresponding sequence of actions. We evaluate two crop configurations: IPR-S with a minimum retained crop area of 10%, and IPR-H with 50%. Across five DeepMind Control Suite tasks, IPR-H achieves an average return across tasks that is 10.0% higher than the baseline with visual and depth images and 3.4
    
[^87]: 柔软也是你所需要的！为什么柔顺性有助于控制：具有柔性踝关节的被动行走器的信息度量

    Softness is something you also need! Why compliance helps control: information metrics of a passive walker with compliant ankle

    [https://arxiv.org/abs/2611.13779](https://arxiv.org/abs/2611.13779)

    本文通过回顾信息度量与智能体动力学之间的数学联系，并应用于带柔性踝关节的被动行走器，揭示了柔顺性有助于控制的机理，从而倡导超越笛卡尔式身心分离、基于形态计算与涌现的机器人设计新范式。

    

    无论是完全刚性还是（部分）柔性组件构成的智能机器人，与自然进化的动植物对应物之间，在适应性、灵活性和灵巧性方面仍然存在巨大差距。其中一个原因可能是“范式”层面的。一方面，大多数机器人（无论刚性还是软体）仍然遵循“笛卡尔式”方法进行设计，即采用自上而下的控制和清晰的身心分离。基于我们在生物学中观察到的原理来设计人工智能体并非易事——动物和植物的组织原则依赖于形态计算，以及从松耦合的具身智能体网络中涌现出的适配行为。建模与控制需要全新的方法。在本文中，我们简要回顾并解释了一些多年前发表的开创性数学成果，这些成果将信息度量与智能体动力学联系起来；我们将其与……

    arXiv:2611.13779v1 Announce Type: cross  Abstract: A great gap in adaptivity, flexibility, and dexterity still exists between intelligent robots, whether they are made of rigid or (partially) compliant components, and their naturally evolved animal and vegetal counterparts. One reason could be of `paradigm' nature. On the one hand, most robots (both rigid and soft) are still designed by following a `Cartesian' approach, with top-down control and a clear mind-body separation. Designing artificial agents on the basis of the principles that we observe in biology is not an easy task. The principles of organization of animals and plants rely on morphological computation and emergence of fit behaviors from loosely coupled networks of embodied agents.   Modeling and control require radically new approaches. In this paper we briefly recall and explain some seminal mathematical results linking information metrics and dynamics of the agents, already published years ago; we integrate them with so
    
[^88]: DisasterAgent：一种基于证据驱动的、可从遥感影像生成可验证灾害报告的智能体

    DisasterAgent: An Evidence-Driven Agent for Verifiable Disaster Report Generation From Remote Sensing Imagery

    [https://arxiv.org/abs/2611.13776](https://arxiv.org/abs/2611.13776)

    提出了证据驱动的智能体DisasterAgent，通过“感知-增强-生成”工作流、视觉专家工具包和时空感知图（STPG），从双时相遥感影像中生成附带可验证证据的灾害报告。

    

    遥感灾害评估正从简单的损害分类向综合的报告级分析发展。在这一背景下，应急救援人员不仅需要损害描述，还需要可验证的证据来指导响应与恢复工作。近期的视觉语言模型（VLM）和工具增强智能体虽然具备强大的多模态推理与感知能力，但它们难以提供高风险验证所需的证据。为了解决这些局限性，我们提出了DisasterAgent，这是一个证据驱动的智能体，它采用“感知-增强-生成”工作流，从双时相光学影像中生成可验证的灾害报告。DisasterAgent首先在感知阶段调用专门的视觉专家工具包，构建时空感知图（STPG），作为结构化记忆来存储和组织灾害在空间和时间上引起的变化。随后，基于证据的增强（EA）从……中提取证据

    arXiv:2611.13776v1 Announce Type: cross  Abstract: Remote sensing disaster assessment is moving from simple damage classification toward comprehensive report-level analysis. In this setting, responders need not only damage descriptions but also verifiable evidence to guide response and recovery. Recent vision-language models (VLMs) and tool-augmented agents offer strong multimodal reasoning and perception, but they struggle to provide the evidence required for high-stakes verification. To address these limitations, we propose DisasterAgent, an evidence-driven agent that uses a Perception-Augmentation-Generation workflow to generate verifiable disaster reports from bitemporal optical imagery. DisasterAgent first invokes a specialized vision expert toolkit in the perception phase to build a Spatiotemporal Perception Graph (STPG), serving as a structured memory to store and organize disaster-induced changes across space and time. Evidence-based Augmentation (EA) then extracts evidence fro
    
[^89]: 混合信号：面向脑电图（EEG）与心电图（ECG）的权重空间编程

    Mixed Signals: Weight-Space Programming for EEG and ECG

    [https://arxiv.org/abs/2611.13758](https://arxiv.org/abs/2611.13758)

    本文首次将权重空间编程应用于脑电图与心电图等生物信号模型，证明语言模型不仅能诊断并修复植入的模型故障，还能仅凭推理从零写出网络全部权重，并以少6倍的训练数据达到与梯度下降训练相当的效果。

    

    权重空间编程使语言模型能够读取经过训练的神经网络的权重并对其进行编辑。这为通过推理而非梯度下降进行“训练”打开了大门。先前的工作已经证明了该方法在游戏类网络中的功能与前景，以及其在修复语言模型特定故障方面的应用，甚至扩展到了扩散模型。在本工作中，我们展示了该方法在生物信号数据训练的模型上的适用性，特别是脑电图（EEG）和心电图（ECG）数据。我们表明，前沿模型不仅能够在未被告知故障内容的情况下，调试植入生物信号模型中的特定故障，甚至能够在仅给定网络架构、任务以及最少量的训练受试者数据的情况下，从零开始写出网络的全部权重。特别是在其中一组实验中，一个语言模型仅通过推理，就在使用少6倍训练受试者的情况下，达到了与标准梯度下降训练相当的效果。

    arXiv:2611.13758v1 Announce Type: cross  Abstract: Weight-space programming allows language models to read a trained neural network's weights and edit them. This opens the door to ``training'' by reasoning rather than through gradient descent. Prior work has demonstrated its functionality and promise for game-playing networks and fixing specific failures in language models, even extending to diffusion models. In this work, we present its applicability to models trained on biosignal data, specifically EEG and ECG data. We show that frontier models are not only able to debug specific faults planted in biosignal-based models without being told what these faults are, but are even capable of writing all the weights of a network from scratch, given only the architecture, the task, and a minimal set of training subjects. In particular, in one set of experiments, a language model was able to match standard gradient-descent training using $6\times$ fewer training subjects, through reasoning alo
    
[^90]: 面向高达70量子比特有机锡氢氧化物样本型量子对角化的硬件训练生成式量子本征求解器

    A Hardware-Trained Generative Quantum Eigensolver for Sample-Based Quantum Diagonalization of Organotin Hydroxides up to 70 Qubits

    [https://arxiv.org/abs/2611.13755](https://arxiv.org/abs/2611.13755)

    该论文首次将IBM Heron量子处理器纳入训练回路，在高达70量子比特上硬件训练生成式量子本征求解器（GQE），实现了有机锡氢氧化物分子的样本型量子对角化，且训练好的策略无需再训练即可迁移到新分子体系。

    

    量子硬件可以为训练超出精确经典模拟能力范围的线路生成器提供反馈。我们训练了生成式量子本征求解器（GQE）策略——即以分子描述符为条件的transformer模型——来为样本型量子对角化（SQD）生成态制备线路。训练首先在无噪声模拟器上进行，随后将IBM Heron处理器ibm_kobe纳入训练回路，分别在46和70量子比特规模上进行（四次主要训练运行共计1.2×10⁸次采样）。训练分子的经典CCSD(T)参考能量仅用于准入门槛判断和两个辅助奖励项。在硬件训练过程中，通过奖励惩罚机制防止策略退化为仅采样Hartree-Fock行列式的线路。在过渡金属配合物上训练得到的策略无需再训练即可应用于Sn(OH)₃-Me及其水合物——它们是极紫外光刻所用有机锡光刻胶的最小模型，规模从32量子比特起（摘要在此处截断）。

    arXiv:2611.13755v1 Announce Type: cross  Abstract: Quantum hardware can provide feedback for training circuit generators beyond the reach of exact classical simulation. We train generative quantum eigensolver (GQE) policies, transformers conditioned on molecular descriptors, to generate the state-preparation circuits for sample-based quantum diagonalization (SQD). Training runs first on a noiseless simulator and then with the IBM Heron processor ibm_kobe inside the loop, at 46 and 70 qubits ($1.2 \times 10^8$ shots for the four main training runs). A classical CCSD(T) reference energy of the training molecules enters only an admission gate and two auxiliary reward terms. During hardware training, a reward penalty prevents collapse to circuits that sample only the Hartree-Fock determinant. Policies trained on transition-metal complexes are applied without retraining to Sn(OH)$_3$-Me and its hydrates, minimal models of organotin photoresists for extreme-ultraviolet lithography, from 32 t
    
[^91]: 迈向自主RTL与综合优化：基准测试与自我改进智能体

    Toward Autonomous RTL and Synthesis Optimization: Benchmarking and Self-Improving Agents

    [https://arxiv.org/abs/2611.13713](https://arxiv.org/abs/2611.13713)

    本文提出迈向自主RTL与综合优化的三个方向，并介绍自我改进的多智能体框架Dr. RTL，其在20个真实设计上相比商业综合工具实现平均WNS/TNS提升21%/17%、面积减少6%。

    

    大语言模型（LLM）与自主智能体的最新进展正在为硬件设计优化创造新的机遇。本文探讨了实现自主RTL与综合优化的三个相互关联的方向：（1）现实的基准测试。我们讨论了基于参考的和开放式的RTL优化基准，用以在真实EDA工作流程下评估优化机会。（2）基于工具的RTL优化。我们提出了Dr. RTL，一个多智能体框架，可自动化关键路径分析、RTL重写和实现级评估，并通过可复用的优化技能持续自我改进。在20个真实RTL设计上的评估显示，Dr. RTL相比商业综合工具平均实现了WNS/TNS提升21%/17%，同时面积减少6%。（3）面向特定设计的综合优化。我们探索了智能体驱动的综合脚本调优，以及跨中间表示的可执行综合算法的发现。

    arXiv:2611.13713v1 Announce Type: new  Abstract: Recent advances in large language models (LLMs) and autonomous agents are creating new opportunities for hardware design optimization. This paper examines three connected directions toward autonomous RTL and synthesis optimization: (1) Realistic benchmarking. We discuss reference-based and open-ended RTL optimization benchmarks that evaluate optimization opportunities under realistic EDA workflows. (2) Tool-grounded RTL optimization. We present Dr. RTL, a multi-agent framework that automates critical-path analysis, RTL rewriting, and implementation-level evaluation, while continuously improving through reusable optimization skills. Evaluated on 20 real-world RTL designs, Dr.~RTL achieves average WNS/TNS improvements of 21\%/17\% with a 6\% area reduction over commercial synthesis. (3) Design-specific synthesis optimization. We explore agent-driven synthesis script tuning and the discovery of executable synthesis algorithms across interme
    
[^92]: 基于潜在状态编码的交互式强化学习中的广泛持久性建议方法

    Broad-Persistent Advising in Interactive Reinforcement Learning via Latent State Encoding

    [https://arxiv.org/abs/2611.13712](https://arxiv.org/abs/2611.13712)

    本文提出Encoder BPA方法，用潜在状态编码器替代传统K-means聚类，通过潜在比较、top-k聚合和一致性条件来检索和重用外部建议，从而提升了交互式强化学习早期的学习效率和建议利用率。

    

    深度交互式强化学习（Deep IntRL）通过引入外部建议来加速强化学习的训练。然而，传统的深度交互式强化学习仅在当前步骤使用每条建议，并在单次交互后将其丢弃。持久建议和广泛持久建议通过将建议存储在持久内存中，并基于K-means聚类的状态泛化在相似状态中重用建议，从而扩展了这一过程。本研究提出了Encoder BPA，该方法采用潜在编码器来检索和重用建议。在该框架中，使用训练好的编码器在每个时间步将状态序列映射为潜在表示。建议可以通过潜在比较进行检索，同时结合top-k聚合和一致性条件以提高重用的稳定性。实验结果表明，Encoder BPA在训练早期阶段提高了学习效率和建议利用率。

    arXiv:2611.13712v1 Announce Type: new  Abstract: Deep Interactive Reinforcement Learning (Deep IntRL) accelerates reinforcement learning training by incorporating external advice. However, conventional Deep IntRL uses each piece of advice only in the current step and discards it after a single interaction. Persistent Advising and Broad-Persistent Advising (BPA) extend this process by storing advice in persistent memory and reusing it in similar states through state generalisation based on K-means clustering. This research proposes Encoder BPA, which employs a latent encoder to retrieve and reuse advice. In this framework, a trained encoder is used to map a state sequence into a latent representation at every time step. Advice can be retrieved through latent comparison, together with a top-(k) aggregation and a consistency condition to improve reuse stability. Experimental results show that Encoder BPA improves learning efficiency and advice utilisation at an early stage of training com
    
[^93]: AutoManifold：基于流形嵌入的数据可视化算法智能体化设计

    AutoManifold: Agentic Design of Data Visualisation Algorithms via Manifold Embedding

    [https://arxiv.org/abs/2611.13710](https://arxiv.org/abs/2611.13710)

    AutoManifold 提出了一种由多智能体大语言模型驱动的算法生成流水线，能够根据用户指定的结构保持偏好，从受约束的原语词汇中自动组合并迭代优化流形嵌入算法，实现面向用户偏好的数据可视化算法自主设计。

    

    数据可视化的流形嵌入领域目前由少数手工设计的最先进（SOTA）算法主导。这些算法的设计目标在开发阶段即被固定，导致其算法行为缺乏灵活性，难以根据用户指定的结构优先级进行调整。为满足用户需求而手工设计新的嵌入算法需要高度专业化的知识且耗时费力。为实现面向用户偏好的自主算法设计，我们提出了智能体化算法生成流水线 AutoManifold。该流水线从一组受约束的亲和度、代价与优化原语词汇表中组合生成新的流形嵌入算法，并由多智能体大语言模型（LLM）编排提供支持。AutoManifold 将其设计的每个阶段都基于用户指定的结构保持偏好进行条件化，并通过 LLM 引导、以度量指标为依据的迭代方式反复优化算法配置。

    arXiv:2611.13710v1 Announce Type: new  Abstract: Manifold embedding for data visualisation is dominated by a small set of hand-designed state-of-the-art (SOTA) algorithms. Their design targets are fixed at development time, resulting in inflexible algorithmic behaviour that cannot be readily steered towards user-specified structural priorities. Hand-designing new embedding algorithms to address user priority requires highly specialised expertise and is time consuming. To address autonomous algorithm design tailored to user preference, we introduce an agentic algorithm generation pipeline AutoManifold. It composes new manifold embedding algorithms from a constrained vocabulary of affinity, cost, and optimisation primitives, supported by multi-agent large-language-model (LLM) orchestration. AutoManifold conditions every stage of its design on user-specified structural preservation preferences, and iteratively refines the algorithm configuration through an LLM-guided, metric-grounded iter
    
[^94]: 智能体硬件开发时代：智能体AI与形式化验证的交汇

    Agentic AI Meets Formal Verification in the Era of Agentic Hardware Development

    [https://arxiv.org/abs/2611.13705](https://arxiv.org/abs/2611.13705)

    本特邀论文综述了智能体AI与硬件形式化验证的双向协同关系：AI从规约、证明构建和推理引擎三个层面推进形式化方法的发展，而形式化方法则通过RTL生成、调试和保正确性优化保障AI驱动硬件开发的可靠性。

    

    随着AI智能体在生成和优化硬件电路方面的能力日益增强，确保AI生成设计的正确性变得愈发重要。与此同时，AI的进步降低了开发和应用形式化方法的门槛，为自动化和改进硬件验证创造了新的机遇。这篇特邀论文概述了AI时代硬件形式化方法的方法论与应用。首先，我们探讨了智能体AI如何在三个层面推进硬件形式化方法：规约、证明构建和推理引擎。其次，我们讨论了形式化方法如何通过RTL生成、调试和保持正确性的优化来支持可靠的AI驱动硬件开发。我们回顾了代表性研究以及我们近期的贡献，重点介绍了关键技术进展和实验结果带来的洞见。最后，我们讨论了未来的研究机遇……

    arXiv:2611.13705v1 Announce Type: new  Abstract: As AI agents become increasingly capable of generating and optimizing hardware circuits, ensuring the correctness of AI-generated designs becomes more important. Meanwhile, advances in AI lower the barriers to developing and applying formal methods, creating new opportunities to automate and improve hardware verification. This invited paper presents an overview of the methodologies and applications of hardware formal methods in the AI era. First, we examine how agentic AI advances hardware formal methods across three levels: specification, proof construction, and reasoning engines. Second, we discuss how formal methods support reliable AI-driven hardware development through RTL generation, debugging, and correctness-preserving optimization. We review representative research alongside our recent contributions, highlighting key technical advances and insights from experimental results. Finally, we discuss future research opportunities towa
    
[^95]: SeqFold：面向用户行为序列骨干的时间戳可寻址压缩状态

    SeqFold: Timestamp-Addressable Compressed States for User Action Sequence Backbones

    [https://arxiv.org/abs/2611.13704](https://arxiv.org/abs/2611.13704)

    SeqFold通过可学习的多通道融合、通道级时间衰减与无查找量化（LFQ），将冻结长历史骨干网络的多层用户状态压缩为时间戳可寻址的紧凑离散编码，供下游任务直接对齐复用。

    

    现有的序列用户建模方法主要关注改进骨干编码器，或针对单一预测目标学习任务特定的用户表示。我们转而研究如何将冻结的长历史骨干网络的多层用户状态导出为紧凑的、时间戳可寻址的表示，用于下游训练回填与线上服务。在该设定下，给定时间戳的用户状态概括了用户在该时间点之前的行为，因此下游系统只需与每个目标时间戳对齐的状态，而无需完整的隐藏状态时间线。为此，我们提出SeqFold——一种压缩状态接口，它结合了可学习的多通道融合、通道特定的时间衰减以及无查找量化（LFQ），从冻结的骨干状态中生成紧凑的离散编码；随后由任务特定的解码器将这些共享编码映射到灵活的嵌入空间，供下游模型使用。（原文摘要在此处截断）

    arXiv:2611.13704v1 Announce Type: new  Abstract: Existing sequential user modeling methods mainly focus on improving the backbone encoder or learning task-specific user representations for a single prediction objective. We instead study how to export the multi-layer user states of a frozen long-history backbone as compact, timestamp-addressable representations for downstream training backfill and serving. In this setting, the user state at a given timestamp summarizes the user's actions before that time, so downstream systems need only the state aligned to each target timestamp rather than the full hidden-state timeline. To this end, we present \texttt{SeqFold}, a compressed-state interface that combines learned multi-channel fusion, channel-specific temporal decay, and Lookup-Free Quantization (LFQ) to produce compact discrete codes from frozen backbone states. Task-specific decoders then map these shared codes into flexible embedding spaces for downstream models. To make timestamp-ad
    
[^96]: 非平稳监督下的拓扑动力学：神经网络权重演化的持续同调

    Topological Dynamics Under Non-Stationary Supervision: Persistent Homology of Neural Network Weight Evolution

    [https://arxiv.org/abs/2611.13701](https://arxiv.org/abs/2611.13701)

    本研究利用持续同调追踪不同标签监督条件下神经网络权重点云的拓扑演化，发现正确标签模型的H1持续性在描述性上高于随机标签条件，但经多重检验校正后未达统计显著性。

    

    我们研究了在稳定与非平稳监督下，神经元权重点云的持续拓扑如何演化。我们在Fashion-MNIST数据集上训练了结构完全相同的多层感知器，分别采用正确标签、训练标签向量的固定随机置换、或每个epoch重新生成的新随机置换。五个配对随机种子共享相同的初始化和小批量顺序，共得到15个训练100个epoch的模型。在22个检查点处，对原始及按行归一化的输入权重点云进行了H0和H1 Vietoris-Rips持续性分析。正确标签模型的平均真实标签测试准确率达到89.39%；固定随机标签模型拟合了其被分配训练目标的67.94%，但无法泛化到原始任务。正确标签模型的时间平均归一化fc3层H1持续性在描述性上高于两种随机标签条件，但经过Holm校正后，预设的精确比较并未达到常规显著性水平。早期条件分类……（原文摘要在此处截断）

    arXiv:2611.13701v1 Announce Type: new  Abstract: We investigate how the persistent topology of neuron- weight point clouds evolves under stable and non-stationary supervision. An identical multilayer perceptron is trained on Fashion-MNIST with correct labels, a fixed random permutation of the training-label vector, or a fresh permutation every epoch. Five paired seeds share initialisation and minibatch order, yielding 15 models trained for 100 epochs. Raw and row-normalised incoming weight clouds are analysed with H0 and H1 Vietoris Rips persistence at 22 checkpoints. Correct-label models reach 89.39% mean true-label test accuracy; fixed-random models fit 67.94% of their assigned training targets without generalising to the original task. Correct-label models have descriptively greater time-averaged normalised fc3 H1 persistence than either random-label condition, but the configured exact comparisons do not reach conventional significance after Holm correction. Early condition classifi
    
[^97]: 面向安全多智能体系统的私有知识检索与私有推理研究

    On Private Knowledge Retrieval and Private Inference for secure multi-agent systems

    [https://arxiv.org/abs/2611.13696](https://arxiv.org/abs/2611.13696)

    本文针对多智能体系统中语义记忆面临的记忆投毒攻击，提出了私有知识检索与私有推理作为防御与缓解策略，以保障智能体系统的安全。

    

    多智能体系统为基于大语言模型（LLM）的智能体的大规模部署提供了一个便捷的框架。它为智能体提供了一个健全的框架，使其能够与现有软件交互并自主地执行操作。智能体AI（Agentic AI）是用于指代这类基于AI的自主系统的新术语。在真实的多智能体系统中，智能体需要应对恶意智能体，其中一些恶意智能体会利用安全漏洞进行破坏。记忆投毒攻击就是一种利用漏洞修改智能体知识、进而影响其行为的例子。智能体中存在不同类型的记忆系统（语义记忆、情景记忆和短期记忆），它们各有特点，针对它们可能发展出不同的记忆投毒攻击。本文聚焦于语义记忆，讨论了针对它的攻击以及缓解策略，并提出了私有知识检索与私有推理。

    arXiv:2611.13696v1 Announce Type: cross  Abstract: Multi-agent systems provide a convenient framework for the massive deployment of agents based on Large Language Models (LLMs). It provides a sound framework where agents can interact with existing software to perform actions in an autonomous way. Agentic AI is the new term to denote these AI-based autonomous systems.   In real multi-agent systems, agents need to deal with malicious agents some of which can exploit security vulnerabilities to do harm. A memory poisoning attack is an example of exploiting a vulnerability to modify agents knowledge and, thus, affect its behavior. In an agent, there are different types of memory systems (semantic, episodic, and short-term) which have their own characteristics and for which different memory poisoning attacks may be developed.   In this paper we focus on semantic memory, we discuss its attack, and mitigation strategies. We introduce private knowledge retrieval and private inference. We propo
    
[^98]: AURA：通过完全对齐从通用音频编码器的统一表示实现听觉空间智能

    AURA: Auditory Spatial Intelligence via a Unified Representation from a General-Audio Encoder through Complete Alignment

    [https://arxiv.org/abs/2611.13687](https://arxiv.org/abs/2611.13687)

    该论文提出AURA，证明未经空间监督训练的通用音频编码器本身通过双耳通道差异保留了空间信息，并通过完全对齐构建统一表示，克服音频-文本信息不对称问题，使单一编码器在多声源场景中同时实现声音识别与空间定位。

    

    回答关于多声源场景的空间问题，要求模型既能识别每个声音，又能对其进行定位。现有的空间音频-语言模型通过专用编码器获取空间信息，这将空间处理与语义处理分割到了不同的特征空间中。我们研究了在没有空间监督的情况下训练的通用音频编解码器能否同时实现这两种功能。受控的逐层分析表明，冻结的编解码器保留了空间信息，其中左右方位线索由两个耳道信号之间的差异所携带。然而，这些信息并不容易被直接利用：在双声源场景中，基于这些特征训练的模型对左右方向的预测仅处于随机水平，并且在其余空间轴上，模型将检测到的方向分配给被询问声源与未被询问声源的频率相同。我们将这种失败归因于音频-文本之间的信息不对称，即音频中包含两个声源，而答案仅描述了其中…（原文摘要至此截断）

    arXiv:2611.13687v1 Announce Type: cross  Abstract: Answering spatial questions about multi-source scenes requires a model both to identify each sound and to locate it. Existing spatial audio-language models obtain spatial information from a dedicated encoder, which separates spatial and semantic processing into distinct feature spaces. We study whether a general-audio codec trained without spatial supervision can serve both purposes. A controlled layer-wise analysis shows that the frozen codec retains spatial information, with lateral cues carried by the difference between the two ear channels. This information, however, is not readily accessible: in dual-source scenes, a model trained on these features predicts left/right at chance and, on the remaining axes, assigns the detected direction to the queried and the non-queried source equally often. We attribute this failure to an audio--text information asymmetry, in which the audio contains both sources while the answer describes only t
    
[^99]: 自我视角引导：利用无动作标注的人类第一视角视频增强机器人策略学习

    Ego Guidance: Enhancing Robot Policy Learning with Action-Free Human Egocentric Videos

    [https://arxiv.org/abs/2611.13685](https://arxiv.org/abs/2611.13685)

    本文提出SPAR架构，在预训练VLM基础上让未来视觉状态预测与动作预测共享学习，从而利用无动作标注的人类第一视角视频增强机器人策略学习，其学到的表征可提升分布外机器人任务性能并随视频数据规模扩大而持续提升。

    

    本文研究了人类第一视角视频如何在没有动作监督的情况下促进机器人策略学习。基于预训练的视觉语言模型（VLM），我们提出了动作与表征共享预测（SPAR）方法，这是一种MoT（混合Transformer）架构，其中未来视觉状态预测可以与动作预测一起提供独立且互补的学习信号。我们发现，从第一视角视频中学到的表征能够提升机器人在分布外任务上的下游性能，并且随着视频数据规模的扩大，性能会进一步提升。我们还研究了训练目标选择和数据筛选的影响，为如何利用无动作标注视频实现可扩展的机器人学习提供了见解。

    arXiv:2611.13685v1 Announce Type: cross  Abstract: In this paper, we investigate how human egocentric videos can benefit robot policy learning without action supervision. Building on a pretrained VLM, we propose Shared Prediction on Action and Representation (SPAR), an MoT architecture in which future visual state prediction can provide independent and complementary learning signals alongside action prediction. We find that representations learned from egocentric videos improve downstream performance on out-of-distribution robot tasks, with further gains as video data scales. We also study the effects of objective choices and data curation, providing insights into how to leverage action-free videos for scalable robot learning.
    
[^100]: 自举式在线策略蒸馏

    Bootstrapped On-Policy Distillation

    [https://arxiv.org/abs/2611.13680](https://arxiv.org/abs/2611.13680)

    该论文发现在线策略自蒸馏（OPSD）在持续训练中会因冻结的特权教师与不断进化的无参考学生之间日益扩大的分布失配而导致性能退化，并提出自举式在线策略蒸馏方法来解决这一问题。

    

    在线策略蒸馏（OPD）通过在学生模型生成的轨迹上蒸馏教师模型的预测来改进语言模型，而在线策略自蒸馏（OPSD）则无需外部教师模型，它使用模型的一个以参考条件为前提的副本作为教师。尽管OPSD在训练早期有显著的改进效果，我们发现它在持续训练下并不能保持一致有效，最终可能出现严重的性能退化。我们研究了这种训练行为，并将性能退化归因于不断演化的无参考学生模型与冻结的特权教师模型之间日益增大的失配。在OPSD的早期阶段，特权教师与学生模型保持高度一致，能够为改进学生的生成轨迹提供有用的指导。然而，随着训练的推进，特权教师与无参考学生之间不断扩大的分布失配可能导致教师提供误导性的……（摘要在此处截断）

    arXiv:2611.13680v1 Announce Type: cross  Abstract: On-policy distillation (OPD) improves language models by distilling teacher predictions on student-generated trajectories, while on-policy self-distillation (OPSD) removes the need for an external teacher by using a reference-conditioned copy of the model as the teacher. Despite its great early improvements, we find that OPSD does not remain consistently effective under continued training and can eventually suffer substantial performance degradation. We study this training behavior and link the degradation to a growing mismatch between the evolving reference-free student and the frozen privileged teacher. In the early stage of OPSD, the privileged teacher remains closely aligned with the student and provides useful guidance for improving the student's generation trajectories. However, as training progresses, the growing distributional mismatch between the privileged teacher and reference-free student can cause the teacher to provide mi
    
[^101]: 理解具身计划的计算框架

    A computational framework for understanding embodied plans

    [https://arxiv.org/abs/2611.13669](https://arxiv.org/abs/2611.13669)

    该论文提出GenEA框架，通过将融合身体生物力学与直觉物理学的多层级状态空间嵌入生成模型，实现了从人类动作出发对多步骤长时程具身计划与目标的联合概率推理。

    

    根据人们的行动来推理其计划与目标，是人类社会互动的基础，也是协作式具身人工智能的关键能力。在自然、具身的环境中形式化这种能力长期以来一直是一个挑战。本文提出了GenEA，一种新颖的智能体表示方法，能够对多步骤、长时程的具身计划进行推理。GenEA的核心是一个多层级状态空间，它通过将机器人学中的任务与运动规划框架与基于模拟的直觉物理学相结合，把关节身体的粗略生物力学及其与邻近物体的交互耦合在一起。通过将该状态空间嵌入生成模型，GenEA给出了目标与具身计划的联合概率分布。在跨越物体操作与导航这两个性质迥异领域的多项实验中，涵盖多样化任务和多种行为测量指标，我们……

    arXiv:2611.13669v1 Announce Type: new  Abstract: Reasoning about people's plans and goals based on their actions is fundamental to human social interaction and essential for cooperative, embodied artificial intelligence. Formalizing this capacity in naturalistic, embodied settings has been a longstanding challenge. Here, we present GenEA, a novel agent representation capable of reasoning over multi-step, long-horizon embodied plans. At the heart of GenEA lies a multilevel state-space that couples the coarse biomechanics of an articulated body and its interactions with nearby objects by extending the task-and-motion planning framework from robotics with simulation-based intuitive physics. By embedding this state-space in a generative model, GenEA yields a joint probability distribution over goals and embodied plans. Across several experiments spanning the qualitatively distinct domains of object manipulation and navigation, featuring diverse tasks and multiple behavioral measures, we sh
    
[^102]: 基于知识空间理论的大语言模型下一步推荐基准测试

    Benchmarking LLM Next-Step Recommendations with Knowledge Space Theory

    [https://arxiv.org/abs/2611.13667](https://arxiv.org/abs/2611.13667)

    该论文提出了一种以知识空间理论（KST）为形式化参照的基准测试方法来评估大语言模型在教育中的下一步推荐能力，实验表明提供知识空间结构信息（如先决条件关系或完整知识空间）可将推荐精确集合准确率从约61%大幅提升至96%以上。

    

    为下一步推荐选择大语言模型（LLM）需要能够反映预期教育任务和领域的基准测试。我们提出了一种使用知识空间理论（KST）作为形式化评估参照的基准测试方法。给定任务目录和已有的先决条件关系，该方法推导出可接受的知识状态以及每个状态下完整的立即下一步集合。模型在指定提示词下生成的推荐将与这些参照集合进行比较。一项示例评估研究使用了来自六个模型的2,400条回复，涵盖四个数学领域、四种提示条件和两种目录顺序。结果显示：仅使用任务描述时精确集合准确率为61.3%，要求模型重建先决条件时为62.8%，提供先决条件关系时为96.3%，提供完整知识空间时为98.2%。性能在不同模型和领域之间存在差异，且推荐集合在两种目录呈现方式之间保持一致。

    arXiv:2611.13667v1 Announce Type: new  Abstract: Selecting an LLM for next-step recommendation requires benchmarks that reflect the intended educational task and domain. We present a benchmarking method that uses Knowledge Space Theory (KST) as a formal evaluation reference. Given a task catalogue and an existing prerequisite relation, the method derives admissible knowledge states and the complete set of immediate next steps for each state. Recommendations generated by a model under a specified prompt are compared with these reference sets. A sample evaluation study uses 2,400 responses from six models across four mathematics domains, four prompt conditions, and two catalogue orders. Exact-set accuracy was 61.3% with descriptions only, 62.8% when models were asked to reconstruct prerequisites, 96.3% with supplied prerequisite relations, and 98.2% with the complete knowledge space. Performance varied across models and domains, and recommendation sets agreed across the two presentations
    
[^103]: DrugCoding：一个用于药物发现的科学编程智能体

    DrugCoding: A Scientific Programming Agent for Drug Discovery

    [https://arxiv.org/abs/2611.13651](https://arxiv.org/abs/2611.13651)

    提出了DrugCoding，一个融合科学任务规划、自适应模型路由、多智能体协作与程序化工具执行的科学编程智能体系统，能够自动化复杂的药物发现工作流程。

    

    药物发现涉及多个相互关联的研究阶段，包括分子设计、性质预测和虚拟筛选。这些过程具有复杂的工作流程、相互依赖的计算任务以及多样化的专业工具，对科学模型和计算资源的协调提出了很高的要求。然而，现有的基于大语言模型（LLM）的科学智能体在复杂任务的自主规划、专业模型的自适应选择以及多阶段科学计算的可靠执行方面仍面临挑战，限制了其在自动化复杂药物发现工作流程中的有效性。为了应对这些挑战，我们提出了DrugCoding，一个集成了科学任务规划、自适应模型路由、多智能体协作和程序化工具执行的科学编程智能体系统。DrugCoding能够实现自主任务规划和动态模型选择……

    arXiv:2611.13651v1 Announce Type: new  Abstract: Drug discovery involves multiple interconnected research stages, including molecular design, property prediction, and virtual screening. These processes are characterized by complex workflows, interdependent computational tasks, and diverse specialized tools, placing substantial demands on the coordination of scientific models and computational resources. However, existing large language model (LLM)-based scientific agents still face challenges in autonomous planning of complex tasks, adaptive selection of specialized models, and reliable execution of multi-stage scientific computations, limiting their effectiveness in automating complex drug discovery workflows. To address these challenges, we propose DrugCoding, a scientific programming agent system that integrates scientific task planning, adaptive model routing, multi-agent collaboration, and programmatic tool execution. DrugCoding enables autonomous task planning and dynamic model a
    
[^104]: MOAI：面向非交互式安全Transformer推理的模块优化架构

    MOAI: Module-Optimizing Architecture for non-interactive secure transformer Inference

    [https://arxiv.org/abs/2611.13641](https://arxiv.org/abs/2611.13641)

    MOAI提出了一个基于同态加密的非交互式安全Transformer推理框架，通过统一的列/对角线打包评估流程、无旋转的Softmax和LayerNorm算法以及交错批处理等模块级优化，大幅消除了昂贵的HE旋转操作与格式转换，显著提升了加密推理效率。

    

    当大型语言模型（LLM）部署在云服务提供商（CSP）上进行推理时，引发了隐私方面的担忧。同态加密（HE）通过直接在加密输入上执行安全推理，提供了一种有前景的解决方案。然而，同态加密高昂的计算开销仍然是主要瓶颈。为应对这一挑战，我们提出了MOAI，一个高效的基于同态加密的非交互式安全Transformer推理框架。MOAI在以下方面获得了显著的效率提升：（1）一种新颖的评估流程，将列打包与对角线打包相结合，并在所有层中采用一致的策略，从而消除了昂贵的格式转换；（2）无旋转的Softmax和LayerNorm算法，显著减少了昂贵的同态加密旋转操作次数，在BERT-base推理中消除了2,448次同态加密旋转；（3）列打包消除了明文-密文矩阵乘法中的旋转操作，而交错批处理进一步降低了开销。

    arXiv:2611.13641v1 Announce Type: cross  Abstract: Privacy concerns have been raised in Large Language Models (LLM) inference when models are deployed in Cloud Service Providers (CSP). Homomorphic encryption (HE) offers a promising solution by enabling secure inference directly over encrypted inputs. However, the high computational overhead of HE remains a major bottleneck. To address this challenge, we propose MOAI, an efficient HE-based, non-interactive framework for secure transformer inference. MOAI gains significant efficiency improvement from: (1) a novel evaluation flow that combines column and diagonal packing with consistent strategies across all layers, eliminating expensive format conversions. (2) rotation-free algorithms for Softmax and LayerNorm that significantly reduce the number of costly HE rotations, removing 2,448 HE rotations in BERT-base inference. (3) Column packing removes rotations in plaintext-ciphertext matrix multiplications and interleaved batching further r
    
[^105]: Synapse-Control：通过轻量级运行时控制使智能体经验可操作化

    Synapse-Control: Making Agent Experience Actionable with Lightweight Runtime Control

    [https://arxiv.org/abs/2611.13615](https://arxiv.org/abs/2611.13615)

    Synapse-Control用一个轻量级智能体运行时控制平面（规则筛选、1.7B微调分类器评分、微合同接口检查和提示价值表冷却机制）取代大模型失败诊断，以更低的推理成本实现智能体执行经验的高效复用与精准投放。

    

    大型语言模型智能体通过复用执行经验来提高其在复杂工具使用任务上的可靠性。诸如Sentry之类的系统在检索建议之前需要调用大模型生成失败诊断，这为经验复用增加了推理成本。我们提出Synapse-Control，这是一个智能体运行时，其轻量级控制平面负责选择建议并决定何时将其发送给执行任务的模型。规则首先筛选出相关建议，然后由一个在教师标注的状态-建议对上微调的轻量级1.7B分类器对这些候选建议进行评分，而不是生成诊断。微合同编码了可复用的接口检查，例如必需参数或有效选项，而无需规定完整的任务程序。提示价值表记录先前的指导和反馈，通过冷却机制和令牌预算检查来限制重复建议。在AppWorld和WebShop上每种方法进行1,308次尝试的评测中，Synapse-Control实现了270次成功（29.（摘要在此处截断）

    arXiv:2611.13615v1 Announce Type: cross  Abstract: Large language model agents reuse execution experience to improve reliability on complex tool-based tasks. Systems such as Sentry call large models to generate failure diagnoses before retrieving advice, adding inference cost to experience reuse. We present Synapse-Control, an agent runtime whose lightweight control plane selects advice and determines when to send it to the task-executing model. Rules shortlist relevant advice, and a lightweight 1.7B classifier fine-tuned on teacher-labeled state-suggestion pairs scores those candidates instead of generating a diagnosis. Micro-contracts encode reusable interface checks, such as required arguments or valid options, without prescribing complete task programs. A Prompt Value Table records earlier guidance and feedback to limit repeated suggestions through cooldown and token-budget checks. Across 1,308 attempts per method in AppWorld and WebShop, Synapse-Control achieves 270 successes (29.
    
[^106]: Megrez：基于MK集合论的面向形式化数学的知识引导智能体

    Megrez: A Knowledge-Guided Agent for Formal Mathematics over MK Set Theory

    [https://arxiv.org/abs/2611.13613](https://arxiv.org/abs/2611.13613)

    Megrez提出了一个基于MK集合论的知识引导形式化智能体，通过可重建知识图谱、反馈驱动的短程证明循环等三项机制，解决了在低资源数学库中进行累积性教材形式化时的概念对齐、长程搜索同步和证明集成三大难题。

    

    配备证明助手工具的通用编码智能体在独立的竞赛题目上已经表现出强劲性能，但在低资源数学库中进行累积性的教材形式化时，仍然面临三大障碍：将教材概念与库中的声明对齐、使长程搜索与实时证明状态保持同步，以及将局部验证过的证明可靠地集成到不断增长的代码库中。我们提出了Megrez，一个基于Morse-Kelley（MK）集合论构建的、面向Rocq数学库的知识引导形式化智能体。Megrez通过三个机制来协调通用编码智能体：(1) 一个可从源码重建的知识图谱，它合并了形式化声明、编译器引用和人工整理的元数据，以支持通过中文别名、概念簇、证明依赖和多通道全文排序进行检索；(2) 一个由反馈驱动的1-5条战术短程证明循环，配合目标局部分支账本和（摘要在此处被截断）……

    arXiv:2611.13613v1 Announce Type: new  Abstract: General-purpose coding agents equipped with proof-assistant tools have shown strong performance on standalone competition problems, yet cumulative textbook formalization inside a low-resource mathematical library still faces three obstacles: aligning textbook concepts with library declarations, keeping long-horizon search synchronized with live proof states, and integrating locally verified proofs reliably into a growing codebase. We present Megrez, a knowledge-guided formalization agent for Rocq libraries built on Morse--Kelley (MK) set theory. Megrez coordinates a general coding agent through three mechanisms: (1) a source-rebuildable knowledge graph merging formal declarations, compiler references, and curated metadata to support retrieval via Chinese aliases, concept clusters, proof dependencies, and multi-lane full-text ranking; (2) a feedback-driven 1--5 tactic short-burst proving loop paired with a goal-local branch ledger and exp
    
[^107]: 环境遥感中的地球嵌入

    Earth Embeddings in Environmental Remote Sensing

    [https://arxiv.org/abs/2611.13612](https://arxiv.org/abs/2611.13612)

    本文系统阐述了地球嵌入的理论基础并回顾了代表性产品，展示其作为多源地球观测的通用表示，能够在无需任务特定设计与预处理的情况下广泛支持环境遥感下游任务。

    

    来自卫星平台的地球观测已深刻改变了众多领域的研究与应用。然而，不均衡的观测覆盖和频繁的数据缺口仍然限制了其捕捉地球表面动态的能力。此外，处理多模态地球观测需要大量的计算资源和复杂的数据协调框架，而这可能并非所有遥感从业者都能轻易获取。地球嵌入已成为应对这些挑战的有前景的解决方案。作为从多源地球观测转换而来的通用表示，地球嵌入无需任务特定的特征设计和预处理，即可支持广泛的下游任务。在本研究中，我们首先介绍了地球嵌入的理论基础，然后回顾了几种代表性的地球嵌入产品，接着考察了地球嵌入的应用。

    arXiv:2611.13612v1 Announce Type: new  Abstract: Earth observations from satellite platforms have profoundly transformed research and applications across numerous domains. However, uneven observation coverage and frequent data gaps continue to limit their ability to capture Earth's surface dynamics. Moreover, processing multi-modal Earth observations requires substantial computational resources and sophisticated data harmonization frameworks, which may not be readily accessible to many remote sensing practitioners. Earth embeddings have emerged as promising solutions to these challenges. As general-purpose representations converted from multi-source Earth observations, Earth embeddings can support a wide range of downstream tasks with no requirement of task-specific feature design and preprocessing. In this study, we first introduce the theoretical foundations of Earth embeddings. Then, we review several representative Earth embedding products. We next examine the applications of Earth
    
[^108]: 迈向可靠的视觉语言模型评判者：状态条件不变性与呈现感知诊断

    Towards Reliable VLM Judges: State-Conditional Invariance and Presentation-Aware Diagnostics

    [https://arxiv.org/abs/2611.13596](https://arxiv.org/abs/2611.13596)

    该论文提出状态条件不变性（SCI）这一可靠性原则，并构建了基于模拟器的诊断基准GridWM-Judge，用于系统评估视觉语言模型评判者在语义不变的呈现变化下是否保持稳定、并对改变任务结果的因果干预作出正确响应。

    

    视觉语言模型（VLM）日益被用作多模态和具身任务的自动评判者，然而仅凭总体准确率或一致性无法揭示评判者改变其判决是否出于正确的原因。我们提出了状态条件不变性（SCI）这一可靠性原则，要求评判者在语义保持不变的呈现变化下保持稳定，同时对改变任务结果的因果干预作出相应变化。我们在GridWM-Judge中实现了SCI，这是一个基于模拟器的诊断基准，由确定性的MiniGrid轨迹构建，包含对齐的完整、证据移除和反事实变体，以及呈现与表示探针。通过五个诊断任务，GridWM-Judge测试局部感知、动作条件转移、呈现稳定性、表示可交换性以及Full-CF因果判别能力。我们的诊断评估揭示了异质性的失败模式，包括……

    arXiv:2611.13596v1 Announce Type: new  Abstract: Vision-language models (VLMs) are increasingly used as automated judges for multimodal and embodied tasks, yet aggregate accuracy or consistency alone does not reveal whether a judge changes its verdict for the right reason. We propose State-Conditional Invariance (SCI), a reliability principle requiring a judge to remain stable under semantics-preserving changes in presentation while responding to causal interventions that change the task outcome. We instantiate SCI in GridWM-Judge, a simulator-grounded diagnostic benchmark built from deterministic MiniGrid trajectories with aligned Full, evidence-removal, and counterfactual variants, plus presentation and representation probes. Across five diagnostic tasks, GridWM-Judge tests local perception, action-conditioned transitions, presentation stability, representation exchangeability, and Full-CF causal discrimination. Our diagnostic evaluations reveal heterogeneous failure modes, including
    
[^109]: Transformers 可证明地学习图搜索：训练动力学与深度的指数级威力

    Transformers Provably Learn Graph Search: Training Dynamics and the Exponential Power of Depth

    [https://arxiv.org/abs/2611.13594](https://arxiv.org/abs/2611.13594)

    本文证明了 Transformer 进行图上 DFS 推理时第二层是近线性规模实现的必要且充分条件，并且 k 层模型的可达性前瞻视野随深度呈指数级增长。

    

    大型语言模型（LLMs）在复杂推理任务上取得了显著进展，但它们获得推理能力的机制以及推理的效率仍然知之甚少。在本文中，我们研究了有向图上的路径寻找问题，这是多步推理的一种符号化抽象。我们首先刻画了完成该任务所需的架构条件，证明单层 Transformer 在稠密图上需要 Ω(N²) 的规模才能实现关键的 DFS 子节点选择原语，而双层 Transformer 可以以 O(N log N) 的规模实现完整的 DFS。因此，第二层对于近线性规模的 DFS 推理既是必要的也是充分的。随后，我们将这一表达能力分析扩展到更深的模型：对于每个 k≥3，k 层 Transformer 可以实现前瞻视野为 1+2^(k-3) 的有界可达性 DFS 策略，从而证明了可达性视野随深度呈指数级增长。

    arXiv:2611.13594v1 Announce Type: new  Abstract: Large Language Models (LLMs) have achieved remarkable progress on complex reasoning tasks, yet the mechanisms by which they acquire reasoning capabilities and the efficiency of their reasoning remain poorly understood. In this paper, we investigate the path-finding problem over directed graphs, a symbolic abstraction of multi-step reasoning. We first characterize the architectural requirements for this task, proving that one-layer transformers require $\Omega(N^2)$ size on dense graphs to implement the key DFS child-selection primitive. While a two-layer transformer implements full DFS with $O(N\log(N))$ size. Thus, a second layer is necessary and sufficient for near-linear DFS reasoning. We then extend this expressivity analysis to deeper models: for every $k\geq3$, a $k$-layer transformer implements a bounded-reachability DFS policy with look-ahead horizon $1+2^{k-3}$, establishing an exponential increase in reachability horizon with d
    
[^110]: PhyRubric：基于评分准则的文本到视频生成中物理常识的诊断性评估

    PhyRubric: Rubric-Grounded Diagnostic Evaluation of Physical Commonsense in Text-to-Video Generation

    [https://arxiv.org/abs/2611.13576](https://arxiv.org/abs/2611.13576)

    提出基于评分准则的诊断性评估框架 PhyRubric，在观看视频前为每条描述构建专属评分准则并分解为语义与物理条目，再基于时间戳证据进行条目级判定，从而可解释地评估文本到视频模型是否符合物理常识。

    

    随着文本到视频（T2V）模型日益被定位为世界模拟器，评估其生成的视频是否符合物理常识变得至关重要。然而，现有的基准评测依赖于标注者凭直觉为每个视频打出一个单一分数，这种评分方式主观性强，且无法指出视频究竟违反了哪一条物理原理。我们提出了 PhyRubric，一个以评分准则为基础的、用于评估 T2V 生成中物理常识的诊断性评估框架。对于每条文本描述，PhyRubric 在观看任何视频之前先构建针对该描述的评分准则，将描述分解为语义条目和物理条目，并基于带时间戳证据的条目级判定来推导最终分数。我们将该框架实例化为 PhyRubric-Bench，涵盖五个物理领域的 34 个类别，并将其扩展为包含 3,788 个视频的 PhyRubric-Train 数据集，在该数据集上微调得到开源的 PhyRubric-Judge 评测模型。在十个（模型上的）实证结果表明……

    arXiv:2611.13576v1 Announce Type: cross  Abstract: As text-to-video (T2V) models are increasingly positioned as world simulators, it is essential to evaluate whether their videos conform to physical commonsense. However, existing benchmarks rely on annotators who rate each video with a single score based on intuition, yielding ratings that are subjective and do not tell which physical principle is violated. We introduce PhyRubric, a rubric-grounded diagnostic framework for evaluating physical commonsense in T2V generation. For each caption, PhyRubric constructs a caption-specific rubric before any video is seen, decomposing the caption into semantic and physical items, and derives scores from item-level verdicts grounded in timestamped evidence. We instantiate this framework as PhyRubric-Bench, which spans 34 categories across five physical domains, and scale it up to PhyRubric-Train with 3,788 videos, on which we fine-tune the open-source PhyRubric-Judge. Empirical results across ten 
    
[^111]: 市场条件化JEPA：从不可预测的市场冲击中学习

    Market-Conditioned JEPA: Learning from Unpredictable Market Shocks

    [https://arxiv.org/abs/2611.13572](https://arxiv.org/abs/2611.13572)

    提出市场条件化JEPA（MC-JEPA），通过以留一法构建的市场因子作为预测条件，使资产的市场暴露成为最优预测的必要信息，并在11年中国A股分钟级数据上全面超越单一资产的JEPA方法。

    

    金融时间序列中的大部分变化是由市场范围的冲击驱动的，而这些冲击无法从资产自身的历史数据中预测出来。针对单一资产的自监督学习将这种变化视为不可约减的噪声。在潜在因子模型下，我们证明，将预测条件化于已实现的市场因子，使得资产的市场暴露成为最优预测的必要条件，同时降低了学习资产特定成分时需要对抗的噪声。基于这一洞察，我们提出了市场条件化JEPA（MC-JEPA），这是一种面向金融时间序列的联合嵌入预测架构。在训练过程中，我们从其他资产的未来目标嵌入中为每个资产构建留一法市场条件，并通过从资产自身表示中导出的因子载荷将其纳入模型。在11年的中国A股分钟级数据上，MC-JEPA在涵盖收益率、市场暴露等九个目标上均优于针对单一资产的JEPA。

    arXiv:2611.13572v1 Announce Type: cross  Abstract: Much of the variation in financial time series is driven by market-wide shocks that cannot be predicted from an asset's own history. Per-asset self-supervised learning treats this variation as irreducible noise. Under a latent factor model, we show that conditioning prediction on the realized market factor makes the asset's market exposure necessary for optimal prediction, while reducing the noise against which the asset-specific component is learned. Building on this insight, we introduce Market-Conditioned JEPA (MC-JEPA), a joint-embedding predictive architecture for financial time series. During training, we construct a leave-one-out market condition for each asset from the future target embeddings of other assets and incorporate it through factor loadings derived from the asset's own representation. On 11 years of minute-level Chinese A-share data, MC-JEPA improves over per-asset JEPA across nine targets spanning returns, market ex
    
[^112]: 对学习型交流最优潮流（AC-OPF）代理模型的蜕变测试揭示了其对表示的依赖性

    Metamorphic testing of learned AC optimal power flow surrogates reveals representation dependence

    [https://arxiv.org/abs/2611.13566](https://arxiv.org/abs/2611.13566)

    该论文提出用十三项蜕变关系系统审计学习型交流最优潮流代理模型，发现所有被测配置均存在对输入序列化表示的依赖性，并借此定位出索引错误与不稳定预处理等缺陷，同时证明数据增强可以缓解这种表示依赖问题。

    

    在单一的电网序列化编码上评估学习型交流最优潮流（AC-OPF）代理模型，无法证明其准确性是否依赖于该编码。我们通过十三个蜕变关系来检验这种依赖性，每个关系都规定了变换何时有效以及如何比较其输出。我们的审计涵盖17种模型与流水线配置：两个已发布的训练检查点和15个随机权重的架构或机制对照组。即使将报告容差收紧或放宽十倍，每种配置仍至少违反六个蜕变关系。对于GridSFM-Open v1.1，源代码检查和中间张量干预指向一个索引错误以及三个非规范或数值不稳定的预处理选择；修复后的检查点尚未经过测试。在单独的case500训练实验中，数据增强在四个模型中的三个上降低了变换视图误差，并在所有四个模型上降低了中值差异和约束违反的P95指标（原文摘要在此处截断）。

    arXiv:2611.13566v1 Announce Type: cross  Abstract: Evaluating a learned AC optimal power flow (AC-OPF) surrogate on one grid serialization cannot show whether its accuracy depends on that encoding. We test this dependence with thirteen metamorphic relations, each specifying when a transformation is valid and how to compare its outputs. Our audit covers 17 model and pipeline configurations: two released trained checkpoints and 15 random-weight architecture or mechanism controls. Every configuration violates at least six relations, even at reporting tolerances ten times tighter or looser. For GridSFM-Open v1.1, source inspection and intermediate-tensor interventions point to an indexing bug and three noncanonical or numerically unstable preprocessing choices; a repaired checkpoint remains untested. In a separate case500 training experiment, augmentation lowers transformed-view error in three of four models and median discrepancy and constraint-violation P95 in all four. These benefits de
    
[^113]: 一种基于概念的可解释框架：通过无需概念监督的词元-概念注意力实现临床文本预测

    A Concept-Based Interpretable Framework for Clinical Text-based Prediction via Token-Concept Attention without Concept Supervision

    [https://arxiv.org/abs/2611.13564](https://arxiv.org/abs/2611.13564)

    该论文提出了一种无需概念标注监督的可解释框架，通过定制交叉注意力瓶颈机制，利用预训练语言模型估计临床文本词元与医学概念之间的相关性，从而在临床预测任务中实现基于概念的黑盒模型解释。

    

    目的：基于概念的解释已成为解释黑盒模型的一种有前景的方法，它利用人类可理解的高层概念来解释模型。然而，大多数现有方法需要带标注的概念标签来训练模型识别输入中的概念，而在临床文本中获取此类标注往往成本高昂。本研究旨在使语言模型在临床预测任务中无需概念监督即可实现基于概念的解释。方法：我们提出了一种新颖的框架，通过定制的交叉注意力瓶颈，在无需概念监督的情况下实现基于概念的解释。临床文本和医学概念的文本描述首先由预训练语言模型进行分词和嵌入。然后，交叉注意力模块为每个文本词元估计每个概念的存在情况，生成概念相关性分数，以捕捉概念在整个临床记录中的参与方式。这些分数……（原文摘要在此处截断）

    arXiv:2611.13564v1 Announce Type: cross  Abstract: Objective: Concept-based interpretation has emerged as a promising approach for explaining blackbox models using human-understandable, high-level concepts. However, most existing methods require annotated concept labels to train the model to identify concepts in the input, and obtaining such annotations in clinical text is often expensive. This study aims to enable concept-based interpretation for language models without concept supervision on clinical prediction tasks. Methods: We propose a novel framework that enables concept-based interpretation without concept supervision through a tailored cross-attention bottleneck. Clinical text and the text descriptions of medical concepts are first tokenized and embedded by a pretrained language model. A cross-attention module then estimates the presence of each concept for each text token, producing concept relevance scores that capture how concepts are involved throughout the note. These sco
    
[^114]: 边缘NPU上厂商引导优化的基准测试：从仅NPU到端到端性能（扩展技术报告）

    Benchmarking Vendor-Guided Optimizations on Edge NPUs: From NPU Only to End-to-End Performance. Extended technical report

    [https://arxiv.org/abs/2611.13563](https://arxiv.org/abs/2611.13563)

    该论文提出了一种通过比较基线、厂商引导仅NPU（VG-NPU）和厂商引导端到端（VG-E2E）三种配置来分离硬件、模型转换与部署流程影响的基准测试方法，发现NPU级性能排名与端到端性能排名可能不一致，且厂商原生部署流程可额外降低10.4–70.7%的系统级延迟。

    

    边缘 NPU 的性能不仅取决于硬件能力，还取决于模型转换和整体部署流程。我们提出了一种基准测试方法，通过比较三种配置来分离这些影响：基线配置、厂商引导仅NPU（VG-NPU）和厂商引导端到端（VG-E2E），其中 VG-NPU 保留常见的模型输入/输出和主机端处理，而 VG-E2E 则遵循厂商原生的部署流程。我们将该方法应用于四个边缘 NPU 和三个视觉工作负载。VG-NPU 在 12 个 NPU-工作负载组合中的 10 个中同时改善了准确率和运行时调用延迟，而 VG-E2E 在所有组合中都降低了系统级延迟，但仅在四个组合中改善了准确率。此外，NPU 级别的性能排名并不总是与端到端排名一致，且 VG-E2E 相比 VG-NPU 在系统级延迟上额外提供了 10.4–70.7% 的降低。这些结果表明评估厂商引导优化的端到端效果的重要性。

    arXiv:2611.13563v1 Announce Type: new  Abstract: Edge NPU performance depends not only on hardware capabilities, but also on model transformation and the overall deployment pipeline. We propose a benchmarking methodology that separates these effects by comparing three configurations: Baseline, Vendor Guided NPU Only (VG-NPU), and Vendor Guided End-to-End (VG-E2E), where VG-NPU preserves common model I/O and host-side processing while VG-E2E follows the vendor-native deployment pipeline. We apply this methodology to four edge NPUs and three vision workloads. VG-NPU improves both accuracy and Runtime-call latency in 10 of 12 NPU--workload combinations, whereas VG-E2E reduces System-level latency in all combinations but improves accuracy in only four. Moreover, NPU-level performance rankings do not always match End-to-End rankings, and VG-E2E provides an additional 10.4--70.7\% reduction in System-level latency over VG-NPU. These results demonstrate the importance of evaluating vendor-gui
    
[^115]: 密集奖励在GRPO中值得付出那么多token开销吗？

    Are Dense Rewards Worth Their Tokens in GRPO?

    [https://arxiv.org/abs/2611.13560](https://arxiv.org/abs/2611.13560)

    尽管蒙特卡洛密集过程奖励在单个更新比较中优于标准GRPO，但在以生成token数和GPU小时数衡量的计算量公平匹配下，其高昂的采样开销（占生成token的91%）使结果排名发生逆转，密集奖励并不值得其token代价。

    

    RLVR（可验证奖励的强化学习）通过奖励正确的最终答案来训练语言模型进行推理。广泛使用的算法——组相对策略优化（GRPO）为解答中的每个token赋予相同的信用，因此有研究提出密集过程奖励来为中间步骤分配信用，例如使用蒙特卡洛奖励，通过从每个步骤采样后续内容来为该步骤评分。然而，在每个训练更新中将此类奖励与GRPO进行比较，或在单个批次上用梯度测试加以检验，都忽略了其成本：在我们的设置中，仅后续内容采样就消耗了全部生成token的91%。我们引入了一种计算匹配协议，为每种训练干预分配相同的预算（以生成的token数和训练GPU小时数衡量），并在带有副作用对照的预注册研究中进行测试。在六个模型上的1,668次运行中，计算匹配逆转了排名。在Qwen2.5-1.5B上，蒙特卡洛奖励在每个更新中领先于标准GRPO，但最终却落后14.3个百分点（摘要原文在此处截断）。

    arXiv:2611.13560v1 Announce Type: cross  Abstract: RLVR trains language models to reason by rewarding correct final answers. A widely used algorithm, group-relative policy optimization (GRPO), gives every token of a solution the same credit, so dense process rewards have been proposed to credit intermediate steps, for example with a Monte-Carlo reward that scores each step by sampling continuations from it. Comparing such a reward with GRPO per training update, or vetting it with a gradient test on a single batch, leaves out what it costs: in our setting, the continuations alone take 91% of all generated tokens. We introduce a matched-compute protocol that gives every training intervention the same budget, measured in generated tokens and training GPU-hours, and tests it in preregistered studies with controls for side effects. Across 1,668 runs on six models, matching compute reverses the ranking. On Qwen2.5-1.5B, the Monte-Carlo reward leads standard GRPO per update, yet ends 14.3 per
    
[^116]: Darmok：通过约束收紧与弹性编排实现冠军级密码填字游戏求解

    Darmok: Championship Cryptic Crossword Solving via Constraint Tightening and Resilient Orchestration

    [https://arxiv.org/abs/2611.13557](https://arxiv.org/abs/2611.13557)

    Darmok 是首个在英国密码填字游戏上达到超人水平的自动化系统，通过约束收紧与弹性编排技术，在《泰晤士报》填字游戏锦标赛中无错误解出全部七道谜题，且每一轮速度均超过所有人类选手。

    

    我们提出了 Darmok，这是首个在英国密码填字游戏上达到超人表现的自动化系统。与美式填字游戏不同，密码谜题在具有误导性的表面文字之下隐藏着一个未标记的定义以及组合式的文字游戏；稀疏的网格使得每个单词有近一半的字母未被交叉校验；而且锦标赛规则规定，即使只包含一个错误或空白格子的答案，无论速度多快，其排名都会排在所有无错误答案之后。在第55届于伦敦举办的《泰晤士报》填字游戏锦标赛中，Darmok 与人类精英选手同场竞技，无错误地解出了全部七道锦标赛谜题，并在每一轮中的完成速度都快于所有人类参赛者。它以6分52秒解出决赛谜题，比人类冠军快了整整3分13秒。当今的前沿AI模型平均而言可以以相当的准确性和速度在单次尝试中解出整个谜题。然而，云端API调用仍可能停滞、超时、触发安全拒绝，或陷入推理困境（摘要在此处截断）。

    arXiv:2611.13557v1 Announce Type: new  Abstract: We present Darmok, the first automated system to reach superhuman performance on British cryptic crosswords. Unlike American-style crosswords, cryptic clues disguise an unmarked definition and compositional wordplay beneath a misleading surface reading, sparse grids leave nearly half of each word's letters unchecked, and tournament rules rank a grid with even a single wrong or blank square behind all error-free solutions regardless of speed. Competing alongside an elite human field at the 55th Times Crossword Championship in London, Darmok solved all seven championship puzzles error-free and finished faster than every human competitor in every round. It solved the Final puzzle in 6m 52s, a full 3m 13s ahead of the human champion.   Today's frontier AI models can on average solve a whole puzzle in a single try with comparable accuracy and speed. However, cloud API calls still stall, time out, trigger safety refusals, or get stuck in reaso
    
[^117]: 巨大激活混淆稀疏自编码器

    Massive Activations Confound Sparse Autoencoders

    [https://arxiv.org/abs/2611.13548](https://arxiv.org/abs/2611.13548)

    该论文发现模型中的巨大激活使高达99.9%的激活方差集中于第一主成分（PC1），导致稀疏自编码器的评估指标失真、重要特征在不同运行间无法复现，并提出在去除PC1的激活上训练SAE、再将PC1加回重建结果的修正方法，将前50个特征的共享率从4–39.5%大幅提升至73–99.5%。

    

    稀疏自编码器（SAE）通常通过方差解释率和消融损伤来进行评估。我们展示了巨大激活如何破坏这两种评估，并提出了一种修正方法。巨大激活会导致高达99.9%的激活方差集中在第一主成分（PC1）方向上。由于PC1分量没有唯一的稀疏分解，每次SAE运行都会用一组不同的特征来重建它。这些特征主导了重要性排名：在50个最重要的特征中，平均有47个在另一次独立运行中无法匹配，相比之下随机选取的50个特征中约有27个无法匹配。此外，一个字典可以报告99.96%的方差解释率，而与此同时PC1之外仍有33%的方差未被解释。我们通过在去除了PC1的激活上训练SAE、再将PC1加回重建结果中，从而绕过PC1。在受影响的模型中，前50个特征的共享率从4–39.5%提升至73–99.5%。在相同的TopK稀疏度下，由此得到的SAE会降低模型性能……

    arXiv:2611.13548v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) are commonly evaluated by variance explained and ablation damage. We show how massive activations undermine both evaluations, and propose a correction. Massive activations cause up to 99.9% of activation variance to lie along the first principal component (PC1). Because the PC1 component has no unique sparse decomposition, each SAE run rebuilds it from a different set of features. These features dominate importance rankings: an average of 47 of the 50 most important features are unmatched in a separate run, compared with about 27 of a random 50. A dictionary can also report 99.96% of variance explained while leaving 33% of the variance outside PC1 unexplained. We bypass PC1 by training the SAE on activations with PC1 removed, then adding PC1 to the reconstruction. In affected models, the top-50 feature sharing rate rises from 4-39.5% to 73-99.5%. At the same TopK sparsity, the resulting SAE degrades the model
    
[^118]: 智能体基准测试在任务耗尽之前就丧失了排名能力

    Agent Benchmarks Stop Ranking Before They Run Out of Tasks

    [https://arxiv.org/abs/2611.13541](https://arxiv.org/abs/2611.13541)

    该论文通过对八个基准的运行结果重新评分发现，智能体基准的饱和主要源于“单比特通过/失败”的测量方式过于粗糙且以参考标准封顶，而非任务难度不足，因此单纯推出更难版本并非解决之道。

    

    数据科学与机器学习工程智能体在 MLE-bench、ALE-Bench、SWE-fficiency 和 AlgoTune 等基准上进行评估，这些基准的任务会产出一个量化指标（如 Kaggle 指标、竞赛得分或加速比）。当某个基准无法再区分最强的模型时，通常的对策是推出更难的版本。我们对八个基准的已发布运行结果进行重新评分（包括 SWE-bench Verified 和 Terminal-Bench 以扩展覆盖面），发现这种饱和很大程度上源于测量方式本身，而非任务本身。大多数智能体基准将一次运行简化为一个通过/失败比特，这在两个方面失灵：其一，过于粗糙——SWE-bench Verified 上排名相邻的前十名智能体之间的差距小于其测试自身的误差，修正这些误差后相邻名次的变动幅度，不超过随机删除每个智能体六分之一通过项所造成的影响；其二，以参考标准封顶——ALE-Bench 的前十名按竞赛得分可分高下，但以中位数人类选手为参照评分时则不相上下，尽管近一半的运行结果已超越该人类水平。在固定任务上，更高的门槛只会……（摘要在此处截断）

    arXiv:2611.13541v1 Announce Type: new  Abstract: Data-science and ML-engineering agents are evaluated on benchmarks such as MLE-bench, ALE-Bench, SWE-fficiency and AlgoTune, whose tasks yield a quantity (a Kaggle metric, contest score or speedup). When one stops separating the best models, the usual remedy is a harder version. Re-scoring released runs of eight benchmarks, including SWE-bench Verified and Terminal-Bench for breadth, we find much of this saturation lies in the measurement, not the tasks. Most agent benchmarks reduce a run to one pass/fail bit, which fails in two ways. It is too coarse: neighbouring top-ten agents on SWE-bench Verified differ by less than its tests' errors, and correcting them reorders neighbours no more than deleting a random sixth of each agent's passes. It is capped at the reference: ALE-Bench's top ten, separated by contest score, tie when graded against the median human, although nearly half the runs beat that human. On fixed tasks, a higher bar only
    
[^119]: 构建基于数联网的职业教育可信数据空间运营模型

    Developing an Operational Model for an Internet of Data Based Trusted Data Space for Vocational Education

    [https://arxiv.org/abs/2611.13520](https://arxiv.org/abs/2611.13520)

    本文基于数联网技术构建了职业教育可信数据空间的多方运营模型，通过价值网络、产品组合、收益分配和治理规则的系统设计，强调数据主权保护与激励相容，为区域职业教育数据生态系统的可持续运营提供了参考方案。

    

    职业教育中的数据流通长期以来受到权属不清、缺乏信任和激励不足的制约。数联网为实现可信的跨域数据流通提供了新的底层技术。本文在基于数联网的职业教育可信数据空间基础上，构建了一个涵盖价值网络、产品组合、收益分配和治理规则的多方参与运营模型。该模型强调数据主权保护和激励相容，为区域职业教育数据生态系统的可持续运营提供了参考。

    arXiv:2611.13520v1 Announce Type: cross  Abstract: Data circulation in vocational education has long been constrained by unclear ownership, a lack of trust, and insufficient incentives. The Internet of Data provides a new underlying technology for enabling trusted cross-domain data circulation. Building on an Internet of Data based trusted data space for vocational education, this paper develops a multi-stakeholder operational model encompassing a value network, a product portfolio, revenue allocation, and governance rules. Emphasizing data sovereignty protection and incentive compatibility, the model provides a reference for the sustainable operation of regional vocational education data ecosystems.
    
[^120]: 尼泊尔语-英语码混合语音识别：数据准备与语音基础模型的适配

    Nepali-English Code-Mixed Speech Recognition: Data Preparation and Adaptation of a Speech Foundation Model

    [https://arxiv.org/abs/2611.13512](https://arxiv.org/abs/2611.13512)

    本研究通过为分词器不支持的数字生成尼泊尔语与英语读法并借助零样本ASR选取合理形式，成功将语音基础模型SraVaani适配到尼泊尔语-英语码混合语音识别任务，将词错误率从48.6%降至16.3%，优于微调后的Whisper large-v3-turbo。

    

    尼泊尔语使用者在对话中经常夹杂英语单词，但多语言语音识别系统往往难以应对语言切换以及英语单词所使用的书写文字。我们研究了如何使用一个覆盖六个尼泊尔语语音领域、总时长172.8小时的语料库来适配SraVaani——一个拥有4.49亿参数的TDT-CTC混合模型。该模型的预训练分词器不支持数字和若干常用符号。我们没有更换分词器或丢弃这些录音，而是为数字生成尼泊尔语和英语两种读法，并利用零样本ASR输出来选择合理的形式。全量微调将测试词错误率（WER）从48.6%降至17.1%；12个epoch的训练达到16.3%，而微调后的Whisper large-v3-turbo为16.4%。冻结编码器会使WER升至25.5%，而排除含数字的音频片段会使数字测试语音上的WER从14.3%升至21.3%。我们还在FLEURS数据集上，分别使用其原始参考文本和混合书写文字的替代版本进行了评估。

    arXiv:2611.13512v1 Announce Type: new  Abstract: Nepali speakers regularly mix English words into conversation, but multilingual speech recognisers often struggle with language switches and the writing script used for English words. We study how to adapt SraVaani, a 449M-parameter hybrid TDT-CTC model, using a 172.8-hour corpus covering six Nepali speech domains. Its pretrained tokenizer does not support digits and several common symbols. Rather than replace it or discard these recordings, we generate Nepali and English readings for numbers and use zero-shot ASR output to select a plausible form. Full fine-tuning reduces test word error rate (WER) from 48.6% to 17.1%; a 12-epoch run reaches 16.3%, compared with 16.4% for fine-tuned Whisper large-v3-turbo. Freezing the encoder increases WER to 25.5%, while excluding numeric clips raises WER on numeric test speech from 14.3% to 21.3%. We also evaluate on FLEURS using both its original references and mixed-script alternatives. The compari
    
[^121]: 暗物质：面向SDR到HDR视频重建的VAE感知HDR编码

    Dark Matters: VAE-Aware HDR Encoding for SDR-to-HDR Video Reconstruction

    [https://arxiv.org/abs/2611.13504](https://arxiv.org/abs/2611.13504)

    提出EquivFlow，一种紧凑可逆的曝光等变HDR编码，解决了类对数编码在冻结VAE潜空间中导致的暗部色彩坍缩问题，无需修改VAE即可提升SDR到HDR视频重建的暗部细节与色彩恢复能力。

    

    SDR到HDR重建是一个病态问题，需要在保留原始内容的同时恢复被截断的高光、被压缩的阴影以及量化伪影。潜在扩散模型为恢复缺失细节提供了强大的生成先验，但其有效性受限于HDR信号如何通过预训练的VAE。我们发现常见的类对数编码在生成之前就抑制了低亮度区域的色彩差异，导致冻结VAE潜空间中出现“暗部坍缩”现象。我们提出了EquivFlow，一种紧凑、可逆的曝光等变编码，它将曝光变化映射为加性偏移，同时保留与尺度无关的色彩分量，且无需修改VAE。在多个预训练图像和视频VAE上进行的编码级测试表明，其曝光结构和深暗部色彩传递能力在所测试的各类宿主模型中均能保持。将EquivFlow集成到SDR到HDR视频扩散模型中，可带来（摘要在此处截断）……

    arXiv:2611.13504v1 Announce Type: cross  Abstract: SDR-to-HDR reconstruction is an ill-posed problem in which clipped highlights, crushed shadows, and quantization artifacts must be recovered while preserving the original content. Latent diffusion models provide powerful generative priors for restoring missing detail, but their effectiveness is constrained by how HDR signals pass through the pretrained VAE. We find that common log-like encodings suppress low-luminance chromatic differences before generation, producing a dark-side collapse in the frozen-VAE latent. We introduce EquivFlow, a compact, invertible exposure-equivariant encoding that maps exposure changes to additive shifts while retaining a scale-independent chromatic component, without modifying the VAE. Encoding-level tests across pretrained image and video VAEs show that its exposure structure and deep-dark chroma transport persist across the tested hosts. Integrated into SDR-to-HDR video diffusion, EquivFlow delivers con
    
[^122]: 面向开放词汇场景图生成的推理感知关系表示学习

    Reasoning-Aware Relational Representation Learning for Open-Vocabulary Scene Graph Generation

    [https://arxiv.org/abs/2611.13496](https://arxiv.org/abs/2611.13496)

    提出 RaRe 框架，将 MLLM 生成的思维链推理依据直接转化为并优化为结构化关系嵌入，从而提升开放词汇场景图生成中稀有关系的识别能力和对未见谓词的泛化能力。

    

    开放词汇场景图生成要求模型能够识别超出训练词汇表范围之外的视觉关系，然而现有方法往往依赖于特定数据集中对象—谓词的共现模式。在长尾分布和含噪声的对象定位条件下，这种依赖会导致关系表示产生偏差且相互纠缠，既限制了稀有关系的识别能力，也限制了对未见谓词的泛化能力。为应对这些挑战，我们提出了 RaRe，一种推理感知的关系表示学习框架，它将多模态大语言模型（MLLM）生成的思维链推理依据转化为结构化的关系嵌入。RaRe 并非仅将思维链作为最终预测的中间推理轨迹，而是直接优化由推理依据导出的表示空间，以提升不同关系之间的可区分性。具体而言，RaRe 首先通过自监督学习将推理依据嵌入与文本化的关系描述进行对齐……

    arXiv:2611.13496v1 Announce Type: cross  Abstract: Open-Vocabulary Scene Graph Generation (OV-SGG) requires models to recognize visual relationships beyond the training vocabulary, yet existing methods often rely on dataset-specific object--predicate co-occurrence patterns. Under long-tailed distributions and noisy object localization, such reliance leads to biased, entangled relation representations, limiting both rare-relation recognition and generalization to unseen predicates. To address these challenges, we propose RaRe, a reasoning-aware relational representation learning framework that transforms MLLM-generated chain-of-thought (CoT) rationales into structured relation embeddings. Rather than using CoT only as an intermediate reasoning trace for final prediction, RaRe directly optimizes the rationale-derived representation space to improve inter-relation discriminability. Specifically, RaRe first aligns rationale embeddings with textualized relation descriptions through self-sup
    
[^123]: WoW：一个用于大语言模型全生命周期训练的大规模世界知识语料库

    The WoW: A Large-Scale World Knowledge Corpus for Full-Lifecycle LLM Training

    [https://arxiv.org/abs/2611.13487](https://arxiv.org/abs/2611.13487)

    本文提出WoW——一个由教育知识图谱驱动、支持大语言模型全生命周期训练的大规模世界知识语料库，在多个基准上显著提升模型性能，并提供交互式图谱数据选择与检索平台。

    

    现有的中文教育语料库通常存在覆盖碎片化、缺乏系统性组织的问题。本文提出了WoW（World Knowledge，世界知识），一个专为基座模型全周期训练设计的大规模数据集。该数据集的构建由教育知识图谱驱动，该图谱作为分类主干，用于系统性收集和补全涵盖各个教育阶段的缺失教育模块。与之前的版本相比，本次迭代具备更强的系统性、更高的质量和更大的规模，同时融入了多模态信号，并支持模型开发的各个阶段。实验表明，WoW在多个基准测试中显著提升了模型性能。为促进社区研究，我们还提供了一个基于图谱的数据选择与检索交互平台。

    arXiv:2611.13487v1 Announce Type: new  Abstract: Existing Chinese educational corpora often suffer from fragmented coverage and a lack of systematic organization. In this paper, we propose WoW (\textbf{Wo}rld Kno\textbf{w}ledge), a large-scale dataset designed for the full-cycle training of foundation models. The construction of this dataset is driven by an Educational Knowledge Graph, which serves as a taxonomic backbone for the systematic collection and completion of missing educational modules across all levels. Compared to previous versions, this iteration provides enhanced systematicity, superior quality, and a larger scale, while incorporating multimodal signals and supporting every stage of model development. Experiments demonstrate that WoW significantly improves model performance across diverse benchmarks. To facilitate community research, we further provide an interactive platform for graph-based data selection and retrieval.
    
[^124]: KL正则化强化学习的精确样本复杂度

    Sharp Sample Complexity of KL-Regularized Reinforcement Learning

    [https://arxiv.org/abs/2611.13480](https://arxiv.org/abs/2611.13480)

    本文证明了在KL正则化强化学习中，插件估计器仅需Õ(KH³/(ε·max{ε,β})+KH)个样本即可找到ε-最优策略，并给出了匹配的下界，将先前样本复杂度对有效视界的依赖从H⁵改进为H³，且首次建立了依赖温度β的下界。

    

    KL正则化强化学习——即最大熵强化学习和基于人类反馈的强化学习（RLHF）背后的目标——本质上是“控制即推断”图模型中的变分推断：正则化最优策略是轨迹上的后验分布，而策略的次优性等于β乘以路径测度之间的KL散度。然而，即使拥有生成模型，学习这一后验分布需要多少样本仍是未知的。此前的1/ε保证在有效视界H上按H⁵的规模增长，且针对MDP的下界均不依赖于温度β。在这项工作中，我们证明了对于具有K个状态-动作对的折扣MDP，插件估计器仅需Õ(KH³/(ε·max{ε, β}) + KH)个样本，就能找到一个从每个初始状态出发均为ε-最优的策略；并且当ε ≲ min{H, H²/β}时，即使是可以自主选择采样位置的算法，也无法用更少的样本达到这一目标。

    arXiv:2611.13480v1 Announce Type: cross  Abstract: KL-regularized reinforcement learning, the objective behind maximum-entropy RL and RL from human feedback, is variational inference in the graphical model of "control as inference": the regularized optimal policy is a posterior over trajectories, and the suboptimality of a policy is $\beta$ times a KL divergence between path measures. However, it is not known how many samples it takes to learn this posterior, even with a generative model. Earlier $1/\epsilon$ guarantees scale with $H^5$ in the effective horizon $H$, and no lower bound for MDPs depends on the temperature $\beta$. In this work, we show that for a discounted MDP with $K$ state-action pairs, the plug-in estimator finds a policy that is $\epsilon$-optimal from every initial state with $\widetilde{O}(KH^3/(\epsilon\max\{\epsilon,\beta\}) + KH)$ samples, and that no algorithm, even one that chooses where to sample, can do with fewer when $\epsilon \lesssim \min\{H, H^2/\beta\
    
[^125]: 利用心理物理学注意力分布进行注视增强的奖励建模

    Leveraging Psychophysical Attentional Distribution for Gaze-Augmented Reward Modeling

    [https://arxiv.org/abs/2611.13477](https://arxiv.org/abs/2611.13477)

    该论文基于认知心理学中人类字母识别的发现，将视觉注意力建模为以注视点为中心的空间梯度分布并融入RLHF奖励模型训练，从而实现了比基线更高的偏好预测准确率。

    

    将人类反馈整合到语言模型中作为使模型输出与人类偏好对齐的手段，正受到越来越多的关注，其中基于人类反馈的强化学习（RLHF）是基于偏好的对齐的重要框架。最近的研究将眼动追踪（ET）数据作为奖励模型训练的额外监督信号，将奖励学习建立在人类阅读行为的基础上。在这一方法的基础上，我们利用认知心理学中关于人类字母识别的既有发现，对以往以注视为中心的模型进行了扩展。具体而言，我们将视觉注意力建模为以当前注视位置为中心的空间梯度分布，并将这一表示融入奖励模型训练中。我们的结果表明，使用所提出的注视分布训练的奖励模型比基线模型取得了更高的偏好预测准确率。此外，我们的一些结果……

    arXiv:2611.13477v1 Announce Type: cross  Abstract: Integrating human feedback into language models has gained increasing attention as a means to align model outputs with human preferences, with Reinforcement Learning from Human Feedback (RLHF) serving as a prominent framework for preference-based alignment. Recent studies have incorporated eye-tracking (ET) data as an additional supervisory signal for reward model training, grounding reward learning in human reading behavior. Building on this approach, we extend prior fixation-centric models by leveraging established findings from cognitive psychology on human letter recognition. Specifically, we model visual attention as a spatially graded distribution centered on the current gaze location and incorporate this representation into reward model training. Our results show that the reward model trained with the proposed gaze distribution achieves higher preference prediction accuracy than a baseline model. Moreover, some results from our 
    
[^126]: 当推理看起来像隐藏的理解：思维链如何助长拟人化、认知优越感感知与认知服从

    When Reasoning looks like Hidden Understanding: How Chain-of-Thought Fosters Anthropomorphism, Perceptions of Epistemic Superiority, and Cognitive Deference

    [https://arxiv.org/abs/2611.13456](https://arxiv.org/abs/2611.13456)

    该论文指出，思维链推理使AI看起来像拥有隐藏的“理解力”，这种拟人化认知以及对AI认知优越性的感知会促成人们对AI系统的认知服从，从而削弱人类自身的能动性。

    

    随着基于AI的产品渗透到社会之中，特别是使用生成式模型的系统，一种关于这些产品是什么、它们能做什么和不能做什么的信念体系正在形成。其中一种常见的信念是将语言的形式视为“理解”语言的证据。由于这些生成式模型实际上是在复现语言模式而非理解语义，将这些模式解读为理解的证据是一种错误归因。这种误解又被“生成式模型能够解释其推理过程”这一观念进一步延伸。据轶事证据表明，这些将语言形式误认为其意义的误解正在推动人们对AI系统的认知服从，从而削弱了人类自身的能动性。这类错误涉及将认知属性归因于系统，使其被当作具有意向性的“心智”来对待。更重要的是，被感知到的认知权威与优越性使得这一“心智”显得值得服从。

    arXiv:2611.13456v1 Announce Type: new  Abstract: As AI-based products permeate society, specifically systems using generative models, a belief system is emerging about what these products are and what they can and cannot do. One common belief concerns the perception of the form of language as evidence of 'understanding' language. As these generative models reproduce linguistic patterns rather than understand meaning, interpreting these patterns as evidence of understanding is a misattribution. This misconception is extended by the notion that generative models can 'explain their reasoning.' Anecdotally, these misconceptions which mistake the form of language for its meaning are driving cognitive deference to AI systems in a way that displaces human agency. Such mistakes involve attributions of cognitive properties to the system, allowing them to be treated like intentional 'minds'. Importantly, perceived epistemic authority and superiority make that 'mind' worth deferring to. Anthropom
    
[^127]: SafeRoute-MAS：面向多智能体科学推理的覆盖率感知与风险校准消息路由

    SafeRoute-MAS: Coverage-Aware and Risk-Calibrated Message Routing for Multi-Agent Scientific Reasoning

    [https://arxiv.org/abs/2611.13454](https://arxiv.org/abs/2611.13454)

    SafeRoute-MAS提出了一种仅利用集成前信息、覆盖率感知且风险校准的消息路由程序，在多智能体科学推理中通过选择性移除专家消息，在减少约12%–19%消息令牌的同时带来一定的准确率提升，但严格的统计验证显示其增益有限。

    

    我们提出了SafeRoute-MAS，一种覆盖率感知、风险校准的程序，它仅使用集成前的信息来决定保留所有专家消息还是移除其中一条。在一个包含5,328个委员会的密封缓存上，其与源兼容的工作点获得了+1.80个百分点的增益（95%置信区间[+1.13, +2.52]），同时移除了12.76%的消息令牌。我们还在公共gpt-oss-120b端点上，对一个458个委员会的分层样本进行了4,580次受控集成。样本估计值为-0.09个百分点；按原始工作负载进行设计加权后为+0.94个百分点（95%置信区间[-1.31, +3.50]），实际输入减少18.72%。加权的令牌匹配比较同样未能确立选择器的优势（+1.68个百分点，置信区间[-1.27, +5.10]）。88个缓存的非中性标签中仅有13个（14.8%）在重复调用多数投票下保持了方向一致性。加权后的预期损坏上置信限（UCL）为4.60%，但11个预设边际中仅有2个通过支持/风险规则。因此我们……（原文摘要在此处截断）

    arXiv:2611.13454v1 Announce Type: new  Abstract: We introduce SafeRoute-MAS, a coverage-aware, risk-calibrated procedure that keeps all specialist messages or removes one using only pre-integration information. On a sealed cache of 5,328 committees, its source-compatible operating point gains +1.80 percentage points (95% CI [+1.13, +2.52]) while removing 12.76% of message tokens. We also run 4,580 controlled integrations over a stratified 458-committee sample on a public gpt-oss-120b endpoint. The sample estimate is -0.09 points; design weighting to the original workload gives +0.94 points (95% CI [-1.31, +3.50]) and 18.72% acted-input reduction. A weighted token-matched comparison likewise does not establish a selector advantage (+1.68 points, CI [-1.27, +5.10]). Only 13 of 88 cached non-neutral labels (14.8%) retain direction by repeated-call majority. Expected-corruption UCL is 4.60% after weighting, but only 2 of 11 prespecified marginals pass the support/risk rule. We therefore pr
    
[^128]: ForeSpark：面向并行投机解码的草稿与验证预算联合调度

    ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding

    [https://arxiv.org/abs/2611.13442](https://arxiv.org/abs/2611.13442)

    该论文提出工作保持投机解码（WCS），通过联合调度每个请求的草稿生成与验证预算，解决了并行投机解码中草稿计算大量浪费的问题，从而提升高负载下LLM服务的有效吞吐量。

    

    arXiv:2611.13442v1 公告类型：cross 摘要：并行草稿器通过在单次前向传播中提出一整块草稿token来加速LLM解码，使其对LLM服务具有很强的吸引力。近期的系统优化的是有效吞吐量（goodput，即每秒被接受的token数），因为被拒绝的草稿会抢占其他请求的计算资源。然而，这些系统只能控制验证多少个token，因此每个请求仍然会草稿出完整的一块token。在高服务负载下，我们发现草稿生成会消耗高达40%的步进时间，其中大部分因从未被验证而被丢弃。而且，来自自回归草稿器的草稿长度控制方式无法直接迁移到并行草稿器，因为并行草稿器必须在其唯一一次前向传播之前就确定其长度。此外，草稿长度与验证长度在整个批次中相互耦合，因此必须为每个请求联合选择这两者，同时保证该决策在每一步的计算开销都足够低。在本文中，我们提出节省工作的投机（WCS）设计，该设计联合分配……

    arXiv:2611.13442v1 Announce Type: cross  Abstract: Parallel drafters accelerate LLM decoding by proposing a whole block of draft tokens in a single forward pass, making them attractive for LLM serving. Recent systems optimize goodput (i.e., accepted tokens per second) because rejected drafts steal compute from other requests. However, these systems control only how many tokens are verified, so every request still drafts a full block. At high serving load, we find that drafting consumes up to 40% of step time, of which a majority is discarded because it is never verified. Yet, draft-length control from autoregressive drafters does not carry over, because a parallel drafter commits to its length before its single pass. Moreover, draft and verification lengths are coupled across the batch, so they must be selected jointly for every request while keeping the decision computationally cheap at every step. In this paper, we propose work-conserving speculation (WCS), a design that jointly allo
    
[^129]: 继承的权重：测量全球南方语言模型的分词溢价与主权成本

    Inherited Weights: Measuring the Tokenization Premium and the Sovereignty Cost of Language Models for the Global South

    [https://arxiv.org/abs/2611.13441](https://arxiv.org/abs/2611.13441)

    论文指出微调模型会从基础模型继承词表、知识、文化对齐、数据与许可证等决定性属性，并测量出全球南方语言承受最高达10倍的分词溢价，证明构建本地化词表可将该溢价显著降低约24%，从而揭示了大语言模型给全球南方带来的主权成本。

    

    在美国和中国之外部署的大多数语言模型，都是由少数预训练模型家族微调而来的后代。我们认为，微调使得模型的决定性属性——词表、知识、文化对齐、被记忆的数据以及许可证——都从其基础模型继承而来。我们的贡献包括：(i) 对分词溢价的更新且可复现的测量，涵盖31种语言和2023-2024年发布的7个分词器，基于《世界人权宣言》的27条条文，并对未知词元坍缩进行了显性校正。较旧分词器中阿姆哈拉语的溢价高达10倍，在最新的商业分词器中仍为6.8倍；南美洲原住民语言在全球分词器下需支付1.6倍至2.7倍的溢价。由葡萄牙语构建的词表，相对于所有被测试的全球分词器，可降低13种拉丁字母书写的全球南方语言中11种的溢价，其中瓜拉尼语和克丘亚语约降低24%。(ii) 一套包含八项测试的基准……（摘要原文在此处被截断）

    arXiv:2611.13441v1 Announce Type: cross  Abstract: Most language models deployed outside the United States and China are fine-tuned descendants of a small number of pretrained families. We argue that fine-tuning leaves the decisive properties of a model inherited from its base: vocabulary, knowledge, cultural alignment, memorized data and license. We contribute (i) an updated, reproducible measurement of the tokenization premium across 31 languages and 7 tokenizers released in 2023-2024, on 27 articles of the Universal Declaration of Human Rights, with explicit correction for unknown-token collapse. Premiums reach 10x for Amharic in older tokenizers and remain at 6.8x in the newest commercial one; Indigenous South American languages pay between 1.6x and 2.7x under global tokenizers. A vocabulary built from Portuguese lowers the premium for 11 of 13 Latin-script Global South languages relative to every global tokenizer tested, by about 24% for Guarani and Quechua. (ii) An eight-test bla
    
[^130]: 做上下文学习并不需要特征学习

    You Don't Need Feature Learning to Do In-Context Learning

    [https://arxiv.org/abs/2611.13431](https://arxiv.org/abs/2611.13431)

    通过对核机器的谱分析发现，上下文学习中的泛化解只是记忆解的低阶截断——模型先泛化后逐渐记忆化，因此实现上下文学习并不需要特征学习。

    

    我们应该如何理解上下文学习（ICL）任务中的记忆与泛化，以及何时发生记忆、何时发生泛化？为了研究这一问题，我们研究了旋转不变核机器如何学习Raventos等人（2023）提出的ICL线性回归任务。通过在核特征基中对该任务进行分解，我们得到了这一ICL设置下学习的谱图像。这一图像揭示，随着样本量的增长，核机器会依次学习记忆解中越来越高的阶数分量，而这些分量依赖于训练任务集合中越来越高的阶数统计量。泛化解近似为记忆解的低阶截断，因此模型首先实现泛化，随后随着训练的进行逐渐退化为记忆。结论有三点。第一，如果这一情形具有普遍指示意义，那么在ICL任务上学习“泛化”解，实际上只是对记忆解的欠拟合。

    arXiv:2611.13431v1 Announce Type: cross  Abstract: How should we understand memorization vs. generalization in in-context learning (ICL) tasks, and when does one happen vs. the other? To investigate, we study how rotation-invariant kernel machines learn the ICL linear regression task proposed by Raventos et al. (2023). By decomposing the task in the kernel eigenbasis, we obtain a spectral picture of learning in this ICL setting. This picture reveals that as sample size grows, kernel machines learn successively-higher-degree components of the memorizing solution, which depend upon successively-higher-degree statistics of the training task set. The generalizing solution is approximately a low-degree truncation of the memorizing solution, and so the model first generalizes, then degrades into memorization as training proceeds.   The takeaways are threefold. First, if this case is indicative, learning a "generalizing" solution on an ICL task is really just underfitting the memorizing solut
    
[^131]: 免费午餐式的不确定性：基于粒子解码的序列级Rényi熵用于自适应LLM推理

    Free Lunch Uncertainty: Sequence-Level R\'enyi Entropy from Particle Decoding for Adaptive LLM Reasoning

    [https://arxiv.org/abs/2611.13414](https://arxiv.org/abs/2611.13414)

    该论文发现SMC幂采样解码可零边际成本获得序列级Rényi熵，其中1/2阶熵在13个设置中的10个中表现最佳，是用于LLM自适应推理的零成本选择性预测信号。

    

    量化LLM整个补全结果的不确定性对于选择性预测和自适应计算至关重要。零成本信号（如平均token熵和对数概率）仅局限于token级别或单个补全，而自洽性方法则需要许多额外的补全。我们证明，用于幂采样的序贯蒙特卡洛（SMC）解码本身已经在估计幂分布的归一化常数，这使我们能以零边际成本获得模型补全分布的序列级Rényi熵（SRE），且在每个粒子预算下都有保证。通过Rényi链式法则，我们将这种读出扩展到任意阶数的逐步熵族。我们证明高于1的阶数与每步最可能的token相关联，这解释了它们为何与对数概率保持一致，而1/2阶成员则累积每步搜索努力的对数。该成员在13个设置中的10个中是最佳的零成本选择性预测信号。在MATH5上

    arXiv:2611.13414v1 Announce Type: cross  Abstract: Quantifying uncertainty over an entire LLM completion is critical for selective prediction and adaptive compute. Zero-cost signals such as mean token entropy and log-probability are local to tokens or to one completion, and self-consistency needs many extra completions. We show that Sequential Monte Carlo (SMC) decoding for power sampling already estimates the normalizing constant of the power distribution, which gives the sequence-level R'enyi entropy (SRE) of the model's completion distribution at zero marginal cost, with a guarantee at every particle budget. A R'enyi chain rule extends this readout to a family of stepwise entropies of any order. We prove that orders above one are tied to the most likely token of each step, which explains why they track log-probability, while the order-1/2 member accumulates the log of per-step search effort. This member is the best zero-cost selective-prediction signal in 10 of 13 settings. On MATH5
    
[^132]: 基于逐层预取时限的大语言模型查找表放置

    Placing LLM Lookup Tables by Per-Layer Prefetch Deadlines

    [https://arxiv.org/abs/2611.13413](https://arxiv.org/abs/2611.13413)

    提出 SlackTier，利用逐层预取时限（两个 Engram 层窗口相差 30 倍）将 LLM 的大规模 n-gram 查找表按层放置到不同存储层级，在解码 CUDA 图内实现位精确查找，使部分表可无需缓存直接放在 NVMe 上而几乎不损失性能。

    

    DeepSeek-V4.1 的 203 GB Engram n-gram 表目前存放在主机 DRAM 中，占用了 4-GPU 推理进程所用 223 GiB 内存中的 190 GiB。由于行索引仅取决于输入 token，读取可以在每一步开始时就发起，并与较早层的计算重叠。V4.1 的两个 Engram 层的预取窗口相差 30 倍（0.3 ms 对 9.3 ms）。我们证明，这一时限连同会话内的行复用以及对共享设备的争用，决定了每张表可以放在哪个存储层级、哪些层需要缓存，并且我们给出了一个可根据实测读取时间预测由此产生的性能下降的模型。智能体和聊天流量会在会话内复用行，而静态热集的迁移效果不佳。我们构建了 SlackTier，这是面向 SGLang 的一个表层级，它在步骤开始时发出所有查找，先读取短窗口层，并将一个小的 GPU 行缓存跨层拆分。该路径运行在解码 CUDA 图内部，并与原始 gather 操作位精确一致。第 14 层的表可以放在不带缓存的 NVMe 上，而没有可测量的

    arXiv:2611.13413v1 Announce Type: cross  Abstract: DeepSeek-V4.1's 203 GB of Engram n-gram tables sit in host DRAM today, taking 190 of the 223 GiB a 4-GPU serving process uses. Row indices depend only on input tokens, so reads can start with each step and overlap earlier layers. V4.1's two Engram layers have windows that differ by 30x (0.3 vs. 9.3 ms). We show that this deadline, together with session reuse and contention on a shared device, decides which tier each table can use and which layers need cache, and we give a model that predicts the resulting slowdown from measured read times. Agent and chat traffic reuse rows within sessions, and static hot sets transfer poorly. We build SlackTier, a table tier for SGLang that issues all lookups at step start. It reads the short-window layer first and splits a small GPU row cache across layers. The path runs inside the decode CUDA graph and is bit-exact with the original gather. Layer 14's table can go to NVMe uncached with no measurable 
    
[^133]: 比自身存在更久的突发：前缀缓存智能体服务中的持续性退化

    Bursts That Outlast Themselves: Persistent Degradation in Prefix-Cached Agent Serving

    [https://arxiv.org/abs/2611.13412](https://arxiv.org/abs/2611.13412)

    本文发现前缀缓存的 LLM 智能体服务具有亚稳态特性：即使负载恢复，短短几分钟的请求突发也会使服务器陷入缓存命中率骤降至约 10%、吞吐量减半且持续恶化的退化状态，其根源是滞留会话的工作集超出 KV 缓存池，导致每轮交互都变成代价高昂的重新计算。

    

    LLM 智能体以闭环会话的方式驱动推理引擎，其上下文在每一轮交互中不断增长，而经济高效地服务这类负载依赖于前缀缓存：缓存命中只需对约一千个 token 进行增量预填充，而缓存未命中则需要重新计算 10–40K token 的上下文。我们证明了这使前缀缓存的智能体服务呈现亚稳态特性。在相同的基础工作负载下，短短几分钟的额外到达请求就会使 vLLM 服务器进入退化状态——约 10% 的缓存命中率、大约一半的吞吐量，以及在剩余 70 分钟内持续增长的请求积压——而未经历突发的对照组服务器则保持健康（12 对实验中有 9 对出现此现象，反向情况则从未出现）。这种退化状态出现在缓存预热服务器本可承受的负载水平下，在没有前缀缓存时不会出现，并且在 Qwen3-32B 的 SWE-bench 智能体轨迹形态下以及 SGLang 系统上同样会出现（SGLang 仅能容忍更大的触发阈值）。变慢的服务使更多会话滞留在系统中，它们的工作集超出 KV 缓存池的容量，导致每一轮交互都退化为重新计算。

    arXiv:2611.13412v1 Announce Type: cross  Abstract: LLM agents drive inference engines with closed-loop sessions whose context grows every turn, and serving them economically depends on prefix caching: a hit costs an incremental prefill of about a thousand tokens, a miss recomputes a 10--40K-token context. We show that this makes prefix-cached agent serving metastable. With an identical base workload, a few minutes of extra arrivals leaves a vLLM server in a degraded regime--a ~10% hit ratio, about half the throughput, and a backlog that grows for the remaining 70 minutes--while its no-burst twin stays healthy (9 of 12 pairs; none in the reverse direction). The regime appears at loads the cache-warm server sustains, is absent without prefix caching, and also appears with Qwen3-32B on SWE-bench agent trace shapes and on SGLang, which tolerates larger triggers. Slower service keeps more sessions in the system, their working set outgrows the KV pool, and every turn becomes a recomputation;
    
[^134]: SteerDistill：通过闭环教师引导实现LLM智能体的量化

    SteerDistill: Quantizing LLM Agents via Closed-Loop Teacher Steering

    [https://arxiv.org/abs/2611.13410](https://arxiv.org/abs/2611.13410)

    提出了SteerDistill，一种量化感知的蒸馏方法，通过全精度教师模型以闭环方式引导学生修订失败响应，并利用学生执行反馈迭代优化引导，从而在低比特量化下保持LLM智能体的任务成功率。

    

    低比特量化可以降低LLM智能体的推理成本，但可能损害任务成功率。通过分析失败的执行轨迹，我们发现量化后的智能体往往会忽略任务要求或环境条件，导致环境状态转换偏离成功的全精度参考轨迹。纠正这些行为偏差仍然很困难：离线数据可能无法覆盖学生模型的失败场景，而其自身的执行轨迹在这些场景中又可能缺少正确的响应样本。我们提出了SteerDistill，一种量化感知的蒸馏方法，其中全精度教师模型引导学生修订失败的响应，并根据学生模型执行的反馈迭代地优化其引导。这种闭环教师引导产生了经过本地验证的行为修正。我们通过在无引导条件下训练学生模型，使其在修正轮次上匹配被引导后学生模型的预测以及全精度模型的行为，从而将这些修正内化。

    arXiv:2611.13410v1 Announce Type: new  Abstract: Low-bit quantization reduces the inference cost of LLM agents but can compromise task success. Examining failed rollouts, we find that quantized agents overlook task requirements or environmental conditions, producing environment transitions that diverge from successful full-precision references. Correcting these behavioral divergences remains difficult: offline data may miss the student's failure contexts, while its own rollouts may provide few correct responses in those contexts. We introduce SteerDistill, a quantization-aware distillation method in which a full-precision teacher guides the student to revise failed responses and iteratively refines its guidance using feedback from student execution. This closed-loop teacher steering produces locally verified behavioral corrections. We internalize these corrections by training the student without guidance to match the steered student's predictions at corrected turns and the full-precisi
    
[^135]: 蜇刺蜂群：从双控制器OT蜜罐中提取攻击者情报

    Stinging the Swarm: Extracting Attacker Intelligence from a Twin-Controller OT Honeypot

    [https://arxiv.org/abs/2611.13406](https://arxiv.org/abs/2611.13406)

    本文提出Talos内联网关系统，通过毫秒级请求判断和设备级逼真的镜像孪生蜜罐，在完全阻断LLM自主智能体对真实工业控制器破坏的同时，从中提取攻击者情报。

    

    自主的语言模型智能体可以通过现场总线协议驱动工业控制器，以机器速度对工厂进行探测并组成协同蜂群发起攻击。我们提出了Talos——一个内联网关，它能在毫秒级时间内对每个智能体发给控制器的请求进行判断，并将被识别为恶意的智能体路由到一个设备级高度逼真的诱饵系统——真实工厂的镜像孪生体——在那里攻击会无害地继续进行，同时被读取以提取攻击者情报。我们对完整流水线（检测→隔离→镜像→提取）进行了评估，测试对象为2,400个自主攻击者智能体，平均分为四个层级（每层600个），攻击者隐藏自身意图的努力逐级升级：从完全不掩饰，到明确的保密规则，再到经过训练的掩护说辞。在所有层级中，没有任何写操作到达真实工厂：网关阻断了损害路径，并有设备级的包络检查作为兜底保障；未映射的寄存器一律被拒绝访问，而网关仍然允许安全方向的停机命令通过。

    arXiv:2611.13406v1 Announce Type: cross  Abstract: Autonomous language-model agents can drive industrial controllers over fieldbus protocols, probing a plant at machine speed and in coordinated swarms. We present Talos, an inline gate that judges every agent request to a controller in milliseconds and routes agents it flags as hostile to a device-faithful decoy -- a mirror twin of the real plant -- where their attack continues harmlessly and is read for intelligence. We evaluate the full pipeline (detect -> quarantine -> mirror -> extract) against 2,400 autonomous attacker agents, 600 in each of four tiers that escalate the attacker's effort to stay hidden, from no concealment to an explicit non-disclosure rule and a trained cover story. No write reached the real plant in any tier: the gate denies the damage path and a device-level envelope check backstops it, while unmapped registers are denied without exception and the gate still permits safe-direction stop commands. Decisions are mo
    
[^136]: 活的宪法：面向主权基础模型的参与式对齐

    Living constitutions: Participatory alignment for sovereign foundation models

    [https://arxiv.org/abs/2611.13402](https://arxiv.org/abs/2611.13402)

    提出将主权基础模型“宪法”的制定视为一个类似公民科学、持续演进的参与式过程，并确立迭代性、模块化、可论证性、立场性和合法性五项原则，以避免模型对齐失败或默认接受前沿模型厂商的规范。

    

    在宪法式AI中，一份纯文本形式的“宪法”承载着价值观念，并通过后训练融入模型之中。然而，制定这样一份宪法的程序本身面临重大挑战：谁来贡献内容、谁来决定纳入哪些规范、以及这一过程应在多大程度上具有参与性或民主性？这些挑战可能令人无所适从。但我们认为，采取行动至关重要：如果没有一份经过深思熟虑的宪法，主权模型将面临对齐失败的风险，或者默认遵循前沿模型提供商所设定的规范。我们提议将宪法的制定视为一个类似公民科学的、不断演进的参与式过程，通过迭代方式进行起草、测试和完善。据此，我们提出五项理想原则：迭代性、模块化、可论证性、立场性和合法性。我们进一步论证，这样一份文件不仅必须随时间推移保持回应性，还必须对其所治理的所有不同人群保持回应性，并据此提出相关条款设计。

    arXiv:2611.13402v1 Announce Type: new  Abstract: In Constitutional AI, a plain-text "constitution" encodes values and gets incorporated into a model via post-training. However, the procedure for creating such a constitution poses significant challenges: who contributes, who decides which norms are included, and to what extent must the process be participatory or democratic? These challenges can be paralyzing. Yet, we argue that taking action is essential: without a deliberate constitution, sovereign models would risk misalignment or default to norms set by frontier model providers. We propose treating the development of a constitution as an evolving, participatory process akin to citizen science, iteratively drafted, tested, and refined. We therefore formulate five desirable principles: iterative, modular, justifiable, positional, and legitimate. We further argue that such a document must remain responsive not only over time but across the people it governs, and propose that provisions
    
[^137]: 剂量决定毒性：面向暴露量的生成式药物设计评估

    The Dose Makes the Poison: Exposure-Aware Evaluation of Generative Drug Design

    [https://arxiv.org/abs/2611.13398](https://arxiv.org/abs/2611.13398)

    该论文利用基于生理的药代动力学（PBPK）模拟对生成式药物设计的候选分子进行暴露量感知的评估，揭示了传统静态分子性质筛选方法因忽略剂量可行性与安全性而会遗漏大量更优分子的重要缺陷。

    

    生成式模型正越来越多地被用于为药物发现提出候选分子。它们能够产生远超实验测试能力的候选分子，这使得计算筛选变得至关重要。大多数筛选方法依赖于预测的分子性质（如效价和毒性），却忽略了这些性质在给定剂量下如何转化为实际的药物暴露量。因此，在纸面上看似有前景的分子可能需要不切实际的剂量，或引发安全性问题。我们通过使用基于生理的药代动力学（PBPK）模拟，重新评估从三个PARP1靶向生成式流程中筛选出的分子，来研究这一差距。在60个入围分子中，有7个无法达到足够的靶点暴露量，而其余候选分子中有28%在其所需剂量下显示出预测的安全风险。此外，85%具有可实现剂量的候选分子，其表现不如静态筛选过程中被忽视的分子。这些发现……

    arXiv:2611.13398v1 Announce Type: cross  Abstract: Generative models are increasingly used to propose molecules for drug discovery. They can produce far more candidates than can be experimentally tested, making computational selection essential. Most selection methods rely on predicted molecular properties, such as potency and toxicity, but overlook how these properties translate into drug exposure at a given dose. As a result, molecules that appear promising on paper may require impractical doses or raise safety concerns. We investigate this gap by using physiologically based pharmacokinetic (PBPK) simulations to re-evaluate molecules selected from three PARP1-targeted generative pipelines. Of 60 shortlisted molecules, seven fail to achieve sufficient target exposure, while 28% of the remaining candidates show predicted safety risks at their required dose. Moreover, 85% of candidates with an achievable dose are outperformed by molecules overlooked during static selection. These findin
    
[^138]: AI时代的代码行数？一项关于工程师Token消耗衡量了什么的案例研究

    The AI Era's Lines of Code? A Case Study on What an Engineer's Token Spend Measures

    [https://arxiv.org/abs/2611.13396](https://arxiv.org/abs/2611.13396)

    研究发现工程师的token消耗主要由个体差异而非工作产出决定，其与代码提交量仅呈弱关联（消耗翻倍仅带来约10%的提交增长），因此token消耗并不是衡量工程生产力的可靠指标。

    

    各组织使用工程师的token消耗来监控AI采用情况、设定预算以及衡量生产力，但该指标与工程产出之间的关系尚不明确。我们将JetBrains公司404名工程师在编码智能体中五周的token使用情况与其代码提交记录相关联，并分析了一项针对412名员工的公司调查。我们发现，token消耗的差异主要存在于工程师个体之间，且哪些工程师消耗最多在五周内基本保持稳定。在提交数量相近的几周里，token消耗的第90百分位数达到第10百分位数的39至240倍。就单个工程师而言，token消耗翻倍与提交数量增加约10%相关。可用的工作指标解释了工程师之间token消耗变化的38%，以及工程师个体内部周与周之间变化的13%。在调查中，技术岗位的受访者表示，在近九成的智能体使用评估中，他们将智能体用于编码之外的任务，例如分析、研究和规划。

    arXiv:2611.13396v1 Announce Type: cross  Abstract: Organizations use engineers' token spend to monitor AI adoption, set budgets, and measure productivity, but the metric's relationship with engineering output is unclear. We link five weeks of token usage in coding agents to the commits of 404 engineers at JetBrains, and analyze a company survey of 412 employees. We find that token spend varied mostly between engineers, and which engineers spent the most stayed largely the same over the five weeks. Among weeks with similar commit counts, the 90th percentile of spend was 39 to 240 times the 10th. Within an engineer, doubling spend was associated with about 10% more commits. Available work metrics explained 38% of the variation in spend between engineers and 13% of the week-to-week variation within an engineer. In the survey, respondents in technical roles reported in nearly nine of ten agent evaluations using agents for tasks beyond coding, such as analysis, research, and planning. Token
    
[^139]: 用于神经符号金融预测的在线自适应模糊认知图

    Online-Adaptive Fuzzy Cognitive Maps for Neuro-symbolic Financial Forecasting

    [https://arxiv.org/abs/2611.13388](https://arxiv.org/abs/2611.13388)

    该论文提出两种神经符号金融预测框架：一种基于双源实时数据构建FCM并通过误差驱动Hebbian规则在线自适应更新，再以非对称加性残差融合方法与LSTM预测无缝结合；另一种是基于文献并由大语言模型增强、每条边附带可验证引用的可审计FCM，从而同时兼顾时序建模能力、因果可解释性与在线适应性。

    

    我们提出了一种神经符号框架，将在线自适应模糊认知图（FCM）与长短期记忆网络（LSTM）相结合用于金融预测。金融市场呈现出状态切换和非线性动态特性。现有模型往往在这些关键特性之间做出取舍：要么能捕捉时间模式但隐藏了因果驱动因素，要么能恢复因果图但训练后无法自适应调整。我们通过两个框架来弥合这一差距。框架I从实时双源价格序列和宏观经济指标构建FCM，并通过误差驱动的Hebbian规则在线调整其边的权重。我们引入了非对称加性残差融合方法，在框架I下将FCM和LSTM的预测无缝结合。框架II构建了一个基于文献、由大语言模型增强的FCM，其中每条边都带有可验证的引用，从而提供了可审计的因子相互作用图谱。在纳斯达克综合指数数据上，框架I mat…（原文摘要在此处截断）

    arXiv:2611.13388v1 Announce Type: new  Abstract: We present a neuro-symbolic framework that pairs online-adaptive Fuzzy Cognitive Maps (FCMs) with long short-term memory (LSTM) networks for financial forecasting. Financial markets exhibit regime shifts and nonlinear dynamics. Existing models often trade off one of these key properties: these models capture temporal patterns but hide causal drivers, or they recover causal graphs that cannot adapt post-training. We close this gap with two frameworks. Framework I constructs an FCM from live dual-source price series and macroeconomic indicators, and adapts its edges online through an error-driven Hebbian rule. We introduce the Asymmetric Additive Residual Fusion method to seamlessly combine the FCM and LSTM predictions under Framework I. Framework II constructs a literature-grounded, LLM-augmented FCM where every edge carries a verifiable citation that gives an auditable map of factor interactions. On NASDAQ Composite data, Framework I mat
    
[^140]: 双曲视觉-语言模型到底有多“双曲”？

    How Hyperbolic Are Hyperbolic Vision-Language Models?

    [https://arxiv.org/abs/2611.13384](https://arxiv.org/abs/2611.13384)

    研究发现主流双曲视觉-语言模型中的指数映射实际近似于恒等映射，移除它对性能没有影响，且使用相同层次损失训练的纯欧氏模型表现几乎与之无异，说明这些“双曲”模型在实践中几乎并不真正双曲。

    

    双曲视觉-语言模型（VLM）已成为一个快速发展的研究领域，据报道其具有强大的零样本性能以及一个可解释的嵌入空间——在该空间中，半径按照泛化程度对表示进行排序。但双曲视觉-语言模型在实践中到底有多双曲呢？我们在指数映射上对此进行研究——这是曲率进入原本为欧氏网络的唯一操作。我们表明，在MERU、HyCoCLIP、ARGENT和PHyCLIP这些模型中，指数映射可以被恒等函数很好地近似，即映射前欧氏空间中的成对距离和角度在投影后几乎不变，而在推理时移除该映射后评估分数保持不变。受这一观察的启发，我们在CC3M上使用相同的层次损失从零开始训练欧氏视觉-语言模型。在大约最后90%的训练步骤中，其训练统计量与其双曲对应模型几乎无法区分。

    arXiv:2611.13384v1 Announce Type: cross  Abstract: Hyperbolic vision-language models (VLMs) have emerged as a rapidly growing area of research, reporting strong zero-shot performance alongside an interpretable embedding space in which radius orders representations by generality. But how hyperbolic are hyperbolic VLMs in practice? We investigate this at the exponential map, the single operation through which curvature enters an otherwise Euclidean network. We show that across MERU, HyCoCLIP, ARGENT, and PHyCLIP, the exponential map is well approximated by the identity function, i.e., pairwise distances and angles in the pre-map Euclidean space barely change under projection, and removing the map at inference leaves evaluation scores unchanged. Motivated by this observation, we train Euclidean VLMs on CC3M from scratch under the same hierarchy losses. Over roughly the final 90% of the training steps, the training statistics are nearly indistinguishable from those of their hyperbolic coun
    
[^141]: 在基于损失的规则下为AI对齐强制执行公理

    Enforcing Axioms for AI Alignment under Loss-Based Rules

    [https://arxiv.org/abs/2611.13369](https://arxiv.org/abs/2611.13369)

    本文将AI对齐原则建模为嵌入空间中的线性“投票者”，在基于损失的奖励模型训练框架下研究帕累托最优等公理能否成立，发现即使采用更丰富的奖励模型或修改比较数据，帕累托最优性仍可能被违反。

    

    许多对齐方法，尤其是基于人类反馈的强化学习（RLHF），通过在模型响应的成对比较上最小化损失来训练奖励模型。受宪法AI等原则引导方法的启发，我们将每条原则视为一个“投票者”，它依据该原则比较响应，例如偏好更有帮助或毒性更低的响应。我们将这些投票者表示为固定嵌入空间中的线性方向，这一简化假设源于线性表示假说。在这种线性社会选择模型中，Ge等人（2024）证明了最优线性奖励模型可能违反帕累托最优性（PO）：从“原则即投票者”的视角看，这意味着响应A可能比B帮助更少且毒性更高，却仍获得更高的奖励。我们探究更丰富的奖励模型或对比较数据的修改能否在基于损失的训练中恢复帕累托最优性。我们证明PO仍可能失效。

    arXiv:2611.13369v1 Announce Type: cross  Abstract: Many alignment methods, most notably reinforcement learning from human feedback (RLHF), train reward models by minimizing a loss on pairwise comparisons of model responses. Inspired by principle-guided methods such as Constitutional AI, we treat each principle as a voter that compares responses according to that principle, for example, preferring the more helpful or less toxic response. We represent these voters as linear directions in a fixed embedding space, a simplifying assumption motivated by the Linear Representation Hypothesis. In this linear social choice model, Ge et al. (2024) showed that an optimal linear reward model can violate Pareto optimality (PO): From the principles-as-voters lens, this means a response A can be less helpful and more toxic than B, yet still receive a higher reward. We ask whether richer reward models or changes to comparison data can recover PO within loss-based training. We show that PO can still fai
    
[^142]: 没有任何有限规则手册能够认证所有安全程序：AI代理把关的不可能性定理

    No finite rulebook can certify every safe program: an impossibility theorem for gatekeeping AI agents

    [https://arxiv.org/abs/2611.13367](https://arxiv.org/abs/2611.13367)

    本文证明了一个不可能性定理：由于机器可检验的证据只能确认有限记录所能确定的内容，而“从不”与“最终”类性质无法被有限记录确定，因此不存在完备的有限规则手册能认证所有安全程序——禁止性规则只能单向地发现违规却无法永久放行安全程序，保密性与响应性规则则连违规都无法可靠判定。

    

    一个AI助手编写小程序，而把关者必须仅凭代码决定每个程序能否运行。假设规则是固定的：绝不执行被禁止的操作、绝不让输出依赖于秘密、最终总要给出回答。若每条规则都必须有机器能在有限时间内检验的证据支持，那么两本规则手册——一本列出允许的内容、一本列出禁止的内容——能否同时做到完备？我们证明它们不能。可检验的证据只能确立有限记录所能确定的事情，而“从不”与“最终”永远不会被有限记录所确定。对于禁止性规则，失败是单方面的：每一次违规都可以通过监视被发现，但没有任何安全程序能够被永久放行；而对于保密性和响应性，双方都无法判定。我们精确刻画了证据究竟能够确认哪些关于可见行为的规则，并证明任何n比特的可靠规则手册都会遗漏一个大约n比特的合规程序。

    arXiv:2611.13367v1 Announce Type: new  Abstract: An AI assistant writes small programs, and a gatekeeper must decide, from the code alone, whether each one may run. Suppose the rules are fixed: never perform a forbidden action, never let the output depend on a secret, always answer eventually. Can two rulebooks, one listing what is allowed and one listing what is not, both be complete, if every rule must be backed by evidence that a machine can check in finite time? We prove that they cannot. Checkable evidence can only establish what a finite record settles, and ``never'' and ``eventually'' are never settled by a finite record. For prohibitions the failure is one-sided: every violation can be caught by watching, but no safe program can be cleared for good. For confidentiality and responsiveness, neither side can be. We characterise exactly which rules about visible behaviour evidence can confirm, and we show that any sound rulebook of $n$ bits omits a compliant program of about $n$ bi
    
[^143]: 一种对商品信息与用户评论建模的技术

    A Technique to Model Product Information and User Reviews (MPIUR)

    [https://arxiv.org/abs/2611.13361](https://arxiv.org/abs/2611.13361)

    提出MPIUR多模态图推荐模型，将用户评论编码进异构图并对齐图文表示，在交互历史稀疏的场景下也能显著提升推荐性能（nDCG@10与Recall@10）。

    

    现代电商平台天然提供三种互补的模态：文本形式的商品描述、商品图片以及用户评论。然而，现有的多模态推荐方法往往忽略评论，或者未能在一个统一框架内整合这三种模态。本文提出了MPIUR，这是一个基于图的多模态推荐模型，它在异构图中对评论进行编码，对齐图像与文本表示，并根据商品交互密度自适应地调整协同—语义对齐。在Amazon Reviews 2023的乐器和母婴两个品类上的实验表明，MPIUR在所评估的方法中于两个数据集上均取得了最佳的全局nDCG@10和Recall@10。它还在稀疏商品上获得了最佳的nDCG@10和Recall@10，这表明当交互历史有限时，联合利用评论与多模态商品内容尤为有效。

    arXiv:2611.13361v1 Announce Type: new  Abstract: Modern e-commerce platforms naturally provide three complementary modalities: textual item descriptions, item images, and user reviews. However, existing multimodal recommendation methods often omit reviews or do not integrate all three modalities within a unified framework. In this paper, we propose MPIUR, a multimodal graph-based recommendation model that encodes reviews in heterogeneous graphs, aligns image and text representations, and adapts collaborative--semantic alignment to item interaction density. Experiments on the Amazon Reviews 2023 Musical Instruments and Baby categories show that MPIUR achieves the best global nDCG@10 and Recall@10 among the evaluated methods on both datasets. It also obtains the best nDCG@10 and Recall@10 on sparse items, indicating that jointly exploiting reviews and multimodal item content is particularly useful when interaction histories are limited.
    
[^144]: 何时仅凭预测就够了？一种冻结专家组合的评估协议

    When Are Predictions Enough? An Evaluation Protocol for Frozen Expert Composition

    [https://arxiv.org/abs/2611.13356](https://arxiv.org/abs/2611.13356)

    该论文提出了一个可复用的评估协议来公平比较冻结专家模型的特征组合与预测组合，并发现对六个架构多样的深度伪造检测专家特征进行逻辑回归，在留一生成器评估中比最佳预测组合器的宏平均 AUROC 提升 0.081。

    

    实践者常常在保持模型权重冻结的情况下组合独立训练的模型。这些专家模型通常通过其最终预测结果进行组合，然而它们的内部特征可能保留了那些预测结果所忽略的有用信息。我们探讨了何时组合特征会优于组合预测，以及如何使这种比较公平。我们提出了一个可复用的评估协议，将其应用于来自 DF40（Yan et al., 2024）的 21 个冻结深度伪造检测器，并审查了评估选择如何改变结论。该协议采用强大的预测基线，比较简单与更大规模的特征组合器，分别独立地变化架构与训练数据，并通过留出生成器以及训练/测试重叠控制来报告结果。在留一生成器评估下，对六个架构多样的专家模型特征进行逻辑回归，在每一折中都比测试过的最佳预测组合器将宏平均 AUROC 提升了 0.081。

    arXiv:2611.13356v1 Announce Type: cross  Abstract: Practitioners often combine independently trained models while keeping their weights frozen. These experts are commonly combined through their final predictions, although their internal features may retain useful information that those predictions omit. We ask when combining features improves on combining predictions, and how to make that comparison fair. We introduce a reusable evaluation protocol, apply it to 21 frozen deepfake detectors from DF40 (Yan et al., 2024), and audit how evaluation choices change the conclusions. The protocol uses strong prediction baselines, compares simple and larger feature combiners, varies architecture and training data separately, and reports results by held-out generator with train/test overlap controls. Under leave-one-generator-out evaluation, logistic regression on the features of six architecture-diverse experts improves macro AUROC by 0.081 over the best tested prediction combiner in each fold, 
    
[^145]: 多级相位感知自监督重建用于多回波梯度回波MRI

    Multi-Stage Phase-Aware Self-Supervised Reconstruction for Multi-Echo Gradient-Echo MRI

    [https://arxiv.org/abs/2611.13352](https://arxiv.org/abs/2611.13352)

    提出了一种多级相位感知自监督重建框架MSPR-GRE，通过渐进式k空间扩展与相位估计精化相结合，在无需全采样训练数据的条件下实现加速高分辨率多回波梯度回波MRI重建，同时保留解剖细节与复数回波信息。

    

    目的：开发一种相位感知的自监督框架，用于加速高分辨率多回波梯度回波（mGRE）MRI成像，在定量图谱中保留解剖细节和复数回波信息。方法：我们开发了多级相位感知自监督重建（MSPR-GRE），将渐进式k空间扩展与相位估计和精细化相结合。顺序展开网络重建复数多回波图像并更新依赖回波的相位先验。对于至少包含四个回波的采集，使用基于弛豫信息的时间子空间，而较短的回波序列则在图像域中重建。训练使用留出的实际采集k空间样本，无需全采样目标数据。评估包括9.4 T离体小鼠脑数据（30微米八回波和18微米双回波）以及回顾性欠采样的7 T AHEAD人脑数据，并进行了监督消融实验。结果：MSPR-GRE减少了...区域的伪影（摘要在此处截断）

    arXiv:2611.13352v1 Announce Type: new  Abstract: Purpose: To develop a phase-aware self-supervised framework for accelerated high-resolution multi-echo gradient-echo (mGRE) MRI that preserves anatomical detail and complex echo information for quantitative mapping. Methods: We developed multi-stage phase-aware self-supervised reconstruction (MSPR-GRE), integrating progressive k-space expansion with phase estimation and refinement. Sequential unrolled networks reconstruct complex multi-echo images and update echo-dependent phase priors. A relaxation-informed temporal subspace is used for acquisitions with at least four echoes, while shorter echo trains are reconstructed in the image domain. Training uses held-out acquired k-space samples without fully sampled targets. Evaluation included 9.4 T ex vivo mouse-brain data 30um eight-echo and 18um dual-echo) and retrospectively undersampled 7 T AHEAD human-brain data, with supervised ablations. Results:MSPR-GRE reduced artifacts in regions of
    
[^146]: ProTEUS：一个面向闭环蛋白质进化的突变解析世界模型

    ProTEUS: A mutation-resolved world model for closed-loop protein evolution

    [https://arxiv.org/abs/2611.13348](https://arxiv.org/abs/2611.13348)

    该论文提出ProTEUS——一个基于深度突变扫描预训练的突变解析世界模型，通过MutationFormer突变偏置注意力架构联合建模序列、进化与结构信息，并配合快速的物理约束全原子预测器，实现了闭环蛋白质进化中突变空间的高效计算探索与功能预测。

    

    蛋白质工程需要在崎岖的适应度地形中导航，微小的序列变化可能产生巨大的功能差异，但实验筛选只能采样序列空间中很小的一部分。因此，一个核心挑战是如何在计算机中高效地遍历庞大的组合突变空间。我们提出了ProTEUS，一个面向闭环蛋白质进化的突变解析世界模型。ProTEUS在深度突变扫描实验数据上进行预训练，利用MutationFormer（一种突变偏置的注意力架构）学习序列、进化和结构信息的联合表示。变体被表示为该联合潜在空间中的状态，而突变则定义为动作，其功能后果可被预测。为了支持对大量突变体的探索，我们开发了一种快速的、物理约束的全原子预测器，该预测器通过显式的几何约束强制机制实现了超过10倍的速度提升。

    arXiv:2611.13348v1 Announce Type: cross  Abstract: Protein engineering requires navigating rugged fitness landscapes, where small sequence changes can produce large functional differences, but experimental screening can only sample a small fraction of sequence space. Thus, a central challenge is to efficiently traverse the large combinatorial space of mutations in silico. We introduce ProTEUS, a mutation-resolved world model for closed-loop protein evolution. Pretrained on deep mutational scanning assays, ProTEUS learns a joint representation of sequence, evolutionary, and structural information using MutationFormer, a mutation-biased attention architecture. Variants are represented as states in this joint latent space, and mutations define actions whose functional consequences are predicted. To support exploration of many mutants, we developed a fast, physics-constrained all-atom predictor that utilizes explicit geometric constraint enforcement to achieve >10 fold increase in speed th
    
[^147]: Veritas：面向智能体任务的验证器生成框架

    Veritas: A Verifier Generation Harness for Agent Tasks

    [https://arxiv.org/abs/2611.13336](https://arxiv.org/abs/2611.13336)

    Veritas 是一个自动生成智能体任务验证器的框架，通过引入证据头、领域规则库和编写原则三种机制，使其生成的验证器与官方判定结果的一致性在多个基准上超越其他七个框架。

    

    自我改进系统的有效性在很大程度上取决于其验证器的质量，验证器负责评判每一轮改进的结果。然而，大多数任务的验证器都是手工编写的，这不仅成本高昂，而且无法扩展到那些没有现成验证器的任务。我们提出了 Veritas，一个能够自动生成验证器的框架。与通用的智能体框架不同，Veritas 为验证器生成增加了三种机制：声明验证器将观察内容的证据头、包含任务中未明说的领域规则的规则库，以及部分通过自我改进循环得以完善的编写原则。我们在六个基准上进行了评估：其中四个由执行结果评分（我们首先对其官方验证器进行了认证），另外两个则对成对的回答进行比较。在固定模型的情况下，在全部四个执行评分基准上，Veritas 生成的验证器与官方判定结果的一致性均优于其他七个框架生成的验证器。在两个成对比较基准上（原文摘要在此处被截断）。

    arXiv:2611.13336v1 Announce Type: new  Abstract: The effectiveness of a self-improving system relies heavily on the quality of its verifier, which judges the outcome of each improvement round. However, the verifiers of most tasks are written by hand, which is expensive and does not scale to tasks that come without one. We propose Veritas, a harness that generates verifiers automatically. Unlike general-purpose agent harnesses, Veritas adds three mechanisms for verifier generation: an evidence header stating what the verifier will observe, a library of domain rules that tasks leave unstated, and authoring principles, partly refined by a self-improvement loop. We evaluate on six benchmarks: four scored by execution, whose official verifiers we first certify, and two that compare pairs of responses. With the model fixed, the verifiers Veritas generates agree better with the official verdicts than those of seven other harnesses on all four execution-scored benchmarks. On the two pairwise b
    
[^148]: 解耦视觉-语言模型中的多模态旋转位置编码

    Disentangling Multimodal Rotary Positional Embedding in Vision-Language Models

    [https://arxiv.org/abs/2611.13334](https://arxiv.org/abs/2611.13334)

    该论文将视觉内部空间结构与跨模态空间效应解耦，研究 MRoPE 为文本 token 赋予空间坐标对文本-视觉注意力究竟是有益的空间参考还是有害的位置偏好，并识别出共享坐标布局带来的约束。

    

    视觉-语言模型广泛采用多模态旋转位置编码，它是旋转位置编码的一种扩展，通过引入时间、高度和宽度轴来编码视觉 token 的空间结构。然而，MRoPE 也会为本身缺乏空间结构的文本 token 分配高度和宽度坐标，从而将空间位置效应引入文本-视觉注意力之中。现有的 MRoPE 变体探索了不同的频率分配方式和坐标布局，其设计背后隐含着对这些跨模态效应的不同直觉：这些效应可能为语言与视觉区域的关联提供有用的空间参考，也可能引入不必要的位置偏好从而阻碍文本-视觉交互。在这项工作中，我们将视觉内部的空间结构与跨模态空间效应解耦，以深入研究这一问题。我们发现了共享坐标布局所存在的一个约束，该约束阻碍了消除……

    arXiv:2611.13334v1 Announce Type: new  Abstract: Vision-language models widely adopt Multimodal Rotary Position Embedding (MRoPE), an extension of Rotary Position Embedding (RoPE) that incorporates temporal, height, and width axes to encode the spatial structure of visual tokens. However, MRoPE also assigns height and width coordinates to text tokens that lack inherent spatial structure, introducing spatial positional effects into text-vision attention. Existing MRoPE variants explore different frequency allocations and coordinate layouts, with their designs suggesting different intuitions about these cross-modal effects: they may provide useful spatial references for associating language with visual regions, or introduce unnecessary positional preferences that hinder text-vision interactions. In this work, we disentangle intra-visual spatial structure from cross-modal spatial effects to investigate this question. We identify a constraint of shared coordinate layouts that prevents elim
    
[^149]: AI智能体需要预防性护理吗？一种面向智能体AI的流行病学队列设计

    Do AI Agents need Preventive Care? An Epidemiologic Cohort Design for Agentic AI

    [https://arxiv.org/abs/2611.13333](https://arxiv.org/abs/2611.13333)

    本文借鉴人类流行病学队列研究方法，提出将AI智能体视为队列成员进行纵向追踪的评估框架，用以研究智能体不良事件的行为决定因素及其可预防负担。

    

    随着AI智能体在包括医疗保健和生物医学研究在内的各个领域日益自主地执行任务，理解它们的行为如何随时间发展并影响结果变得愈发重要。现有的评估主要集中于模型能力及其在预定义任务上的表现。在此，我们借鉴人类人群研究中确立的原则，提出了一种用于研究智能体AI的流行病学队列设计。该框架将单个智能体视为队列成员，将其底层模型视为遗传构成，将累积工作量视为其时间尺度，并将自然发生的行为视为可能影响后续结果的暴露因素。通过纵向追踪智能体，该方法能够研究不良事件的行为决定因素，在因果假设下估计其潜在可预防负担，并刻画智能体工作生涯中行为变化的特征。

    arXiv:2611.13333v1 Announce Type: new  Abstract: As AI agents increasingly perform tasks autonomously across sectors, including healthcare and biomedical research, understanding how their behaviors develop and influence outcomes over time becomes increasingly important. Existing evaluations largely focus on model capabilities and performance on predefined tasks. Here, we propose an epidemiologic cohort design for studying agentic AI, drawing on principles established in human population studies. The framework treats individual agents as members of a cohort, their underlying model as genetic makeup, accumulated work as their time scale, and naturally occurring behaviors as exposures that may influence subsequent outcomes. By following agents longitudinally, this approach enables investigation of behavioral determinants of adverse events, estimation of their potential preventable burden under causal assumptions, and characterization of behavioral changes over an agent's work life. We ill
    
[^150]: 面向联邦LoRA的随机投影与秩感知学习率

    Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA

    [https://arxiv.org/abs/2611.13331](https://arxiv.org/abs/2611.13331)

    该论文提出在联邦学习中冻结并周期性重采样LoRA的A矩阵，从理论上证明这等价于对全量微调的随机近似，从而解耦A、B梯度并将计算与上行通信开销降低50%。

    

    在资源受限的分布式客户端上对大规模基础模型进行微调，面临着显著的计算与通信挑战。虽然将低秩适应与联邦学习相结合可以实现基于秩的灵活性，但适应矩阵（A 与 B）之间的相互依赖往往使优化变得复杂。在这项工作中，我们从理论上证明：采用高斯初始化并冻结 A 矩阵的 LoRA 微调，相当于对全量微调的一种随机近似，其中批量梯度会受到随机且与秩相关的扰动。受这一见解的启发，我们提出在客户端层面冻结并周期性地重新采样 A 矩阵。该策略解耦了 A 和 B 的梯度，削弱了 LoRA 的低秩约束，并将计算与上行通信开销降低了 50%，从而允许重新分配……（原文在此处截断）

    arXiv:2611.13331v1 Announce Type: cross  Abstract: Fine-tuning large-scale foundation models across resource-constrained, distributed clients presents significant computational and communication challenges. While merging Low-Rank Adaptation (LoRA) with Federated Learning (FL) enables rank-based flexibility, the interdependence of the adaptation matrices ($\mathbf{A}$ and $\mathbf{B}$) often complicates optimization. In this work, we theoretically demonstrate that LoRA fine-tuning with Gaussian initialization and a frozen $\mathbf{A}$ matrix acts as a stochastic approximation of full fine-tuning, where batch gradients are subject to a random, rank-dependent perturbation. Motivated by this insight, we propose freezing and periodically resampling $\mathbf{A}$ matrices at the client level. This strategy decouples the $\mathbf{A}$ and $\mathbf{B}$ gradients, diminishes LoRA low-rank constraints and reduces both computational and uplink communication overheads by 50\%, allowing for the reall
    
[^151]: 基于物理验证的智能体引导交换-关联泛函发现

    Agent-Guided Discovery of Exchange-Correlation Functionals with Physics-based Validation

    [https://arxiv.org/abs/2611.13329](https://arxiv.org/abs/2611.13329)

    该研究将前沿大语言模型智能体与GPU评估器通过模型上下文协议相连，并引入基于物理的自洽与外部验证流程，最终以远少于对比模型的token用量发现了一个性能更优的13参数交换-关联泛函O159。

    

    准确且紧凑的交换-关联泛函是具有预测能力的密度泛函理论的核心。符号搜索和语言模型智能体能够提出新的泛函形式，但目前尚不清楚这些增益能否经受自洽性和外部评估的检验、新项是否蕴含物理内容、智能体是否优于进化搜索，以及不同语言模型骨干之间的表现差异。我们通过模型上下文协议将前沿大语言模型智能体连接到GPU评估器，采用仅训练集拟合、持久记忆以及自洽与外部评估机制。Opus-5.5智能体以比Codex（GPT-6-Astra）智能体少约40倍的token通过了筛选门槛，并找到了一个在验证集和留出集上误差更低的候选泛函；只有该候选泛函晋级进入自洽评估阶段。由此得到的13参数泛函O159通过一个平滑的密度窗口对同自旋关联进行调制。重新拟合的对照实验倾向于一种w……

    arXiv:2611.13329v1 Announce Type: new  Abstract: Accurate, compact exchange--correlation functionals are central to predictive density functional theory. Symbolic search and language-model agents can propose new functional forms, but it remains unclear whether their gains survive self-consistency and external evaluation, whether the new terms carry physical content, whether agents outperform evolutionary search, and how language-model backbones compare. We connect frontier LLM agents to a GPU evaluator through the Model Context Protocol, with training-only fitting, persistent memory, and self-consistent and external evaluation. Opus-5.5 agents passed the screening gate with about 40-fold fewer tokens than Codex (GPT-6-Astra) agents and found a candidate with lower validation and held-out errors; only this candidate advanced to self-consistent evaluation. The resulting 13-parameter functional, O159, multiplies same-spin correlation by a smooth density window. Refitted controls favor a w
    
[^152]: F3-AGI v4.0：面向样本高效抽象推理的生物能量门控主动推理、超可塑性预处理与残差程序合成

    F3-AGI v4.0: Bioenergetically-Gated Active Inference, Metaplastic Preconditioning, and Residual Program Synthesis for Sample-Efficient Abstract Reasoning

    [https://arxiv.org/abs/2611.13325](https://arxiv.org/abs/2611.13325)

    F3-AGI v4.0提出了一种CPU原生的神经符号主动推理架构，通过生物能量门控可塑性、自适应复杂度引导程序合成与期望自由能探索三大机制，克服ARC-AGI少样本抽象推理中的罗生门效应，实现样本高效的算法变换归纳。

    

    ARC-AGI上的少样本抽象推理需要在没有任务特定预训练的情况下，从2到5个演示对中归纳出算法变换。由于少样本演示对假设空间的约束不足，程序合成受到罗生门效应（预测多样性）的困扰：启发式搜索会找到训练误差低但在测试输入上失效的虚假程序，而不受约束的库增长则会降低运行速度。我们提出了F3-AGI v4.0，这是一个CPU原生的神经符号与主动推理架构，包含三种机制：（1）F3-Metabolic-Optim，结合了错误门控可塑性、Fisher-EMA预处理和虚拟ATP门控；（2）自适应复杂度引导程序合成（AACR），配备Cycle-3残差补丁合成器（ResidualPatchSynthesizer）和最小对早期拒绝策略；（3）512维L2归一化随机投影，驱动基于期望自由能（EFE）的探索并配合死签名剪枝。在三项实证……（原文摘要在此截断）

    arXiv:2611.13325v1 Announce Type: new  Abstract: Few-shot abstract reasoning on ARC-AGI requires inducing algorithmic transformations from 2 to 5 demonstration pairs without task-specific pretraining. Because few-shot demonstrations under-constrain the hypothesis space, program synthesis suffers from the Rashomon effect (predictive multiplicity): heuristic search finds spurious programs with low training error that fail on test inputs, while unconstrained library growth degrades latency. We introduce F3-AGI v4.0, a CPU-native neuro-symbolic and active inference architecture comprising three mechanisms: (1) F3-Metabolic-Optim, combining mistake-gated plasticity, Fisher-EMA preconditioning, and virtual ATP gating; (2) Adaptive Complexity-Guided Program Synthesis (AACR) with a Cycle-3 ResidualPatchSynthesizer and smallest-pair early rejection; and (3) a 512D L2-normalized random projection driving Expected Free Energy (EFE) exploration with dead-signature pruning. Across three empirical s
    
[^153]: 基于R2C-Agent实现可扩展的网络化HIV治疗与预防干预

    Towards Scalable Network-Based Interventions for HIV Treatment and Prevention via R2C-Agent

    [https://arxiv.org/abs/2611.13312](https://arxiv.org/abs/2611.13312)

    该论文针对接触网络难以大规模获取的问题，提出利用R2C-Agent方法，基于可大规模数字化收集的同伴推荐网络开发可扩展的HIV治疗与预防干预策略。

    

    人类免疫缺陷病毒（HIV）仍然是全球范围内亟待解决的公共卫生挑战。如何利用基于网络的算法有效配置有限的HIV预防和治疗资源，是一个长期存在的研究问题，在AAMAS及其他计算领域的学术会议上有着丰富的研究历史。然而，为最小化HIV传播而优化针对性的生物医学干预措施，通常需要对个体之间HIV传播风险行为的接触网络进行观测，而大规模收集这类数据的成本高得令人望而却步。同伴推荐网络，即通过递归同伴推荐过程收集的网络，是另一类常用于研究隐性人群中HIV问题的网络，可以大规模地进行数字化收集，但其与底层接触网络存在偏差：推荐可能沿着无风险的社会关系进行，且参与者仅接受一次推荐，从而隐藏了已被招募个体之间的联系。在这项工作中，我们首先……

    arXiv:2611.13312v1 Announce Type: new  Abstract: Human Immunodeficiency Virus (HIV) remains a pressing public health challenge worldwide. Targeting limited HIV prevention and treatment resources effectively with network-based algorithms is a long-standing problem with a rich history at AAMAS and other computational venues. However, optimizing targeted biomedical interventions for minimizing spread of HIV often requires observability of the contact network of HIV transmission risk behavior among individuals, which is prohibitively expensive to collect at scale. Peer referral networks, i.e., networks collected through a recursive peer referral process, are an alternative network class often used to study HIV in hidden populations that can be collected digitally at scale, but they deviate from the underlying contact network: referrals can follow non-risk social ties and participants accept only one referral, hiding contacts among already-recruited individuals. In this work, we first forma
    
[^154]: PegasusAI Studio：面向科学工作流的可读性网络基础设施与可问责AI智能体

    PegasusAI Studio: Legible Cyberinfrastructure and Accountable AI Agents for Scientific Workflows

    [https://arxiv.org/abs/2611.13309](https://arxiv.org/abs/2611.13309)

    PegasusAI Studio通过设施知识库投射、工作流接口内省和遥测数据诊断，使AI编程智能体能够“读懂”共享网络基础设施，并在结合联邦认证与按用户权限的信任包络内安全地构建和运行科学工作流。

    

    AI编程智能体现在能够构建科学工作流，但在共享网络基础设施（CI）上运行这些工作流会带来代码生成本身无法解决的问题。智能体需要掌握工作流系统及其执行站点的最新知识，并能够访问分布式执行的运行时状态；而设施方则需要将每一个智能体操作归因到具体用户，并将其限制在该用户的权限范围内。我们提出了PegasusAI Studio，一个构建于Pegasus和HTCondor技术栈之上的AI辅助科学网关。Studio通过将一个由设施方维护的知识库投射到多种编程智能体的原生配置中、通过内省工作流接口与执行站点、以及通过将诊断建立在工作流和调度器遥测数据之上，使网络基础设施对智能体而言变得“可读”。智能体操作在一个信任包络内运行，该包络结合了联邦身份认证、按用户隔离的执行环境与调度器凭证、供应商中立的模型访问以及可观测性机制。

    arXiv:2611.13309v1 Announce Type: cross  Abstract: AI coding agents can now construct scientific workflows, but operating them on shared cyberinfrastructure (CI) raises problems that code generation does not solve. Agents need current knowledge of the workflow system and its execution sites and access to the runtime state of distributed executions, while facilities need to attribute every agent action to a user and bound it by that user's privileges. We present PegasusAI Studio, an AI-assisted science gateway built on the Pegasus and HTCondor stack. Studio makes CI legible to agents by projecting one facility-curated knowledge base into the native configuration of several coding agents, by introspecting workflow interfaces and execution sites, and by grounding diagnosis in workflow and scheduler telemetry. Agent actions operate within a trust envelope that combines federated authentication, per-user execution environments and scheduler credentials, provider-neutral model access, observ
    
[^155]: 训练高秩，传输低耗：面向分布式在线策略后训练的高效 LoRA 同步

    Train High, Ship Low: Efficient LoRA Synchronization for Distributed On-Policy Post-Training

    [https://arxiv.org/abs/2611.13306](https://arxiv.org/abs/2611.13306)

    提出与压缩器无关的 HiLo 框架，用压缩梯度分发加本地优化器重放取代 LoRA 参数传输，并配套预算感知的 Spar 梯度压缩器，将训练秩与通信成本解耦，从而在带宽受限网络上高效实现分布式在线策略后训练。

    

    在带宽受限的网络上，将 LoRA 适配器传输给 rollout 工作节点可能成为分布式在线策略后训练的瓶颈。降低训练秩可能会限制工作负载所需的模型容量，而自适应优化器的更新又难以压缩。相反，rollout 工作节点可以通过维护复制的优化器状态，从梯度中重建更新。我们提出了 HiLo，一个与压缩器无关的框架，它以压缩梯度分发和本地优化器重放取代参数传输。通过应用相同的解码后梯度，学习器与工作节点在可复现执行下能在同一版本上保持一致的策略状态。我们进一步开发了 Spar，一种预算感知的梯度压缩器，它结合谱截断、自适应秩分配和误差反馈，将训练秩与通信成本解耦。在在线策略蒸馏和奖励引导后训练上的实验表明，HiLo……

    arXiv:2611.13306v1 Announce Type: cross  Abstract: Transmitting LoRA adapters to rollout workers can bottleneck distributed on-policy post-training over bandwidth-constrained networks. Lowering the training rank may limit the capacity a workload needs, while adaptive optimizer updates are difficult to compress. Rollout workers can instead reconstruct updates from gradients by maintaining replicated optimizer states. We introduce HiLo, a compressor-agnostic framework that replaces parameter transmission with compressed gradient dissemination and local optimizer replay. By applying the same decoded gradients, the learner and workers maintain consistent policy states at the same version under reproducible execution. We further develop Spar, a budget-aware gradient compressor that combines spectral truncation, adaptive rank allocation and error feedback to decouple training rank from communication cost. Experiments across on-policy distillation and reward-guided post-training show that HiL
    
[^156]: MRFoundry：面向端到端MRI研究的知识引导智能体循环

    MRFoundry: Knowledge-guided agent loops for end-to-end MRI research

    [https://arxiv.org/abs/2611.13289](https://arxiv.org/abs/2611.13289)

    MRFoundry是一个开源、可扩展的知识引导智能体框架，将MRI专业知识与智能体行为和输出验证相连接，并配套推出MRI Agent Bench基准测试，显著提升了AI智能体完成端到端MRI研究任务的成功率。

    

    人工智能智能体能够编写代码、使用科学软件并修改计算工作流，为MRI研究开辟了新的可能性。有效的辅助需要将采集物理、数据规范、专用工具和科学验证连接起来。我们提出了MRFoundry，这是一个开源、可扩展的软件包，将MRI知识与智能体行为及输出检查相连接。项目本地记忆可保存决策与证据，而经过审核的贡献允许研究人员改进共享指导。由研究人员监督的示例展示了技能引导的序列审核和采集数据重建，将工具使用、输出检查和记录的决策串联起来。与MRFoundry一同，我们提出了MRI Agent Bench，用于在不同模型和MRI任务领域中比较有和没有该软件包的智能体表现。在涵盖24种任务类型和十个模型的240项配对核心评估中，全部检查通过的完成率从没有MRFoundry时的67.5%（摘要在此处截断）

    arXiv:2611.13289v1 Announce Type: cross  Abstract: AI agents can write code, use scientific software, and revise computational workflows, opening new possibilities for MRI research. Effective assistance requires connecting acquisition physics, data conventions, specialized tools, and scientific validation. We present MRFoundry, an open-source, extensible package that connects MRI knowledge to agent actions and output checks. Project-local memory preserves decisions and evidence, while reviewed contributions allow researchers to improve shared guidance. Researcher-supervised examples demonstrate skill-guided sequence auditing and acquired-data reconstruction, connecting tool use, output checks, and documented decisions. Alongside MRFoundry, we introduce MRI Agent Bench to compare agents with and without the package across models and MRI task domains. Across 240 paired Core evaluations spanning 24 task types and ten models, all-checks-passed completion increased from 67.5% without MRFoun
    
[^157]: 基于游走策略的离策略学习

    Off-policy Learning with Excursion Policies

    [https://arxiv.org/abs/2611.13275](https://arxiv.org/abs/2611.13275)

    该论文提出“游走策略”这一结构化插值方法——智能体先按目标策略执行多步再回归行为策略，建立了其动态规划理论基础并开发了可证明可靠的离策略时序差分算法，在连续控制基准上实现了与混合策略相当的效率-稳定性权衡。

    

    离策略强化学习旨在当数据来自不同于目标策略的行为策略时，学习目标策略的价值。这通常需要在偏差与方差或稳定性之间进行权衡。一种权衡方式是在行为策略与目标策略之间进行线性插值，但这只是更广泛的插值机制空间中的一个点。我们提出了游走策略——一种结构化的替代方案，其中智能体在回归行为策略之前，先按照目标策略进行多步的“游走”。我们为这类策略推导了动态规划结果，为其在强化学习中的应用建立了理论基础。我们进一步开发了对游走价值估计具有可证明可靠性的离策略时序差分（TD）算法。我们发现，在连续控制基准测试中，游走策略提供了与混合策略相当的效率-稳定性权衡，并且我们的分析从概念上阐释了……

    arXiv:2611.13275v1 Announce Type: cross  Abstract: Off-policy reinforcement learning (RL) tries to learn the value of a target policy while data comes from a different behavior policy. This often requires trading off bias for variance or stability. One way to trade these off is to linearly interpolate between the behavior and target policies, but this represents only one point in a broader space of interpolation mechanisms. We introduce excursion policies--a structured alternative where the agent follows the target policy for a multi-step "excursion" before reverting to the behavior policy. We derive dynamic programming results for these policies, establishing the theoretical basis for their use in RL. We further develop provably sound off-policy temporal-difference (TD) algorithms for excursion value estimation. We find that excursion policies provide an efficiency-stability trade-off comparable to mixture policies in continuous control benchmarks, and our analysis illustrates concept
    
[^158]: 用于内存高效时序动作分割的自适应残差量化

    Adaptive Residual Quantization for Memory-Efficient Temporal Action Segmentation

    [https://arxiv.org/abs/2611.13262](https://arxiv.org/abs/2611.13262)

    提出自适应残差量化变分自编码器（ARQ-VAE），通过学习无标签的离散残差量化并逐帧自适应截断残差链，为时序动作分割提供内存高效且灵活适应帧复杂度变化的特征记忆表示。

    

    时序动作分割（TAS）通常在大型预提取的帧特征库上进行训练，但存储和回放这些特征对长视频而言已成为实际瓶颈，尤其是在增量式TAS中，必须保留过去任务的数据以缓解灾难性遗忘。近期的TAS回放和压缩方法减轻了这一负担，但它们依赖于密集的逐帧标注，并通过固定的片段级结构分配存储空间，限制了其对帧复杂度随时间显著变化的长程序视频的灵活性。我们提出了自适应残差量化变分自编码器（ARQ-VAE），这是一种紧凑且无需标签的TAS特征记忆表示，它学习帧级视频特征的离散残差量化，然后在逐帧基础上自适应地截断残差链，仅保留能最佳重构每个帧特征的残差前缀。

    arXiv:2611.13262v1 Announce Type: cross  Abstract: Temporal action segmentation (TAS) is commonly trained on large banks of pre-extracted frame features, but storing and replaying those features becomes a practical bottleneck for long-form video, especially in incremental TAS where past-task data must remain available to mitigate catastrophic forgetting. Recent TAS replay and condensation methods reduce this burden, but they rely on dense frame-wise annotations and allocate storage through fixed segment-level structures, limiting their flexibility for long procedural videos whose frame complexity varies substantially over time. We introduce Adaptive Residual-Quantized Variational AutoEncoder (ARQ-VAE), a compact and label-free memory representation for TAS features that learns a discrete residual quantization of frame-level video features and then adaptively truncates the residual chain on a per-frame basis, retaining only the residual prefix that best reconstructs each frame feature. 
    
[^159]: 可信赖的长时程LLM智能体：一项以轨迹为中心的实证研究——错误从何处进入、哪些控制手段能够触达它们

    Dependable Long-Horizon LLM Agents: A Trajectory-Centered Empirical Study of Where Errors Enter and Which Controls Reach Them

    [https://arxiv.org/abs/2611.13254](https://arxiv.org/abs/2611.13254)

    该研究提出了一种将LLM智能体故障与起源、传播、检测、响应和后果相绑定的“事件记录”标注协议，并通过对六个多智能体系统95次失败运行的完整轨迹实证分析，发现故障多在任务早期（中位数0.18处）以错误决策或状态捏造的形式进入，且往往早于人工标注所认定的决定性失败步骤。

    

    当LLM智能体在长任务中途失败时，通常的分析只记录一个症状：失败模式标签，或被判为责任方的智能体和步骤。这类分析并不会说明错误从何处进入、哪个状态将其传递到症状处、轨迹中是否存在与之矛盾的内容，或哪种控制手段本可以捕获该错误。我们定义了一种“事件记录”，它将候选故障与其起源、传播、检测、响应和后果关联起来，并按操作阶段和状态域进行定位，同时指明哪些控制类别的观察范围与触达能力能够覆盖该故障。我们将该记录转化为书面标注协议，并应用于公开的Who&When和MAD语料库中六个多智能体系统的95次失败运行，完整阅读了每条轨迹。研究发现：故障进入得很早（中位数为运行进程的0.18处），主要表现为错误决策或对任务状态的捏造，并且在88次Who&When运行中有37次的故障出现在语料库人工标注所标记的决定性失败步骤之前。

    arXiv:2611.13254v1 Announce Type: new  Abstract: When an LLM agent fails part-way through a long task, the usual analysis records a symptom: a failure-mode label, or the agent and step judged responsible. It does not say where the error entered, which state carried it to the symptom, whether anything in the trajectory contradicted it, or which control could have caught it. We define an episode record that ties a candidate fault to its origin, propagation, detection, response and consequence, located by operational phase and state domain, and that names the control families whose observation and reach cover it. We turned the record into a written annotation protocol and applied it to 95 failed runs of six multi-agent systems from the public Who&When and MAD corpora, reading each trajectory in full. Faults enter early (median at 0.18 of the run), mostly as wrong decisions or fabrications over task state, and in 37 of 88 Who&When runs before the step the corpus' human labels mark as decis
    
[^160]: Curie's Jarvis：基于印刷实验方案的自主机器人化学实验

    Curie's Jarvis: Autonomous Robotic Chemistry Experiments from Printed Protocols

    [https://arxiv.org/abs/2611.13251](https://arxiv.org/abs/2611.13251)

    Curie's Jarvis 让机械臂无需预置脚本和位置标记，直接从打印的纸质实验方案中读取步骤，在面向人类的普通实验台上自主执行化学实验，并通过多视角读取标签区分相同容器、在每步操作前重新定位目标。

    

    一个在实验台上做实验的人会依照一张打印的说明单操作，通过容器上书写的标签来区分相同的容器，并在每一步之前再次查看实验台。而实验室自动化通常恰恰相反：操作流程被写成脚本，容器被固定在特定位置或带有基准标记。我们提出了 Curie's Jarvis，一个让机械臂在为人类搭建的实验台上进行化学实验的系统。它从打印说明单的照片中读取实验步骤，在没有固定位置或标记的普通实验台上寻找容器，通过在四个相机视角下读取标签来区分外观相同的容器，在每一步操作前重新定位目标容器，并根据所处理的物品设定每一步的执行方式（包括速度）。在 Franka Panda 机械臂上，Curie's Jarvis 完成了一个儿童化学套装实验方案的 5 次运行中的 5 次，以及另一个方案的 5 次运行中的 4 次。

    arXiv:2611.13251v1 Announce Type: cross  Abstract: A person running a bench experiment works from a printed instruction sheet, tells identical containers apart by the labels written on them, and looks at the bench again before each step. Lab automation usually works the other way around: the procedure is written as a script, and containers stand at fixed positions or carry fiducial markers. We present Curie's Jarvis, a system that lets a robot arm carry out a chemistry experiment on a bench set up for people. It reads the steps from a photo of the printed sheet, finds containers on an ordinary bench with no fixed positions or markers, tells identical-looking containers apart by reading their labels in four camera views, locates its target again before every step, and sets how each step is carried out, including its speed, from what is being handled. On a Franka Panda arm, Curie's Jarvis completed 5 of 5 runs of one protocol from a children's chemistry kit and 4 of 5 runs of another, wi
    
[^161]: ExpertCrowd-SWE：在不熟悉的代码库中用专家编写的工程任务衡量前沿编程智能体

    ExpertCrowd-SWE: Measuring Frontier Coding Agents on Expert-Authored Engineering Tasks in Unfamiliar Codebases

    [https://arxiv.org/abs/2611.13246](https://arxiv.org/abs/2611.13246)

    EC-SWE是一个包含100个由专家编写、覆盖87个开源项目和11种语言的多文件软件工程任务基准，通过密封参考方案、可执行加权评分和对抗性对照补丁，在大型陌生代码库中严格评估自主编程智能体的真实工程能力。

    

    ExpertCrowd-SWE（EC-SWE）是一个包含100个由专家编写、涉及多文件修改的软件工程任务基准，用于在大型且陌生的代码库中评估自主编程智能体。每个任务都是由一线工程师针对真实仓库某个固定提交撰写的变更请求，并配有密封的参考解决方案和一个任务专用的评分器；该评分器采用加权的、大多可执行的评分标准并给予部分得分，其中必须通过的阻断项和通过/失败门槛编码了维护者会拒绝合并的内容。这些任务覆盖87个开源项目和11种编程语言，任务工单的中位长度为148个单词，参考解决方案的中位改动量为471行。每个任务镜像在发布前都经过校准，以确保参考解决方案得分为1.0，而未修改的仓库和惰性变更的得分均为0；近一半的任务还附带替代性、偷懒和对抗性控制补丁，评分器必须能正确区分它们。最终的100个任务是从更大的候选池中筛选得出的。

    arXiv:2611.13246v1 Announce Type: new  Abstract: ExpertCrowd-SWE (EC-SWE) is a benchmark of 100 expert-authored, multi-file software engineering tasks for evaluating autonomous coding agents in large, unfamiliar codebases. Each task is a change request written by a practising engineer against a pinned commit of a real repository, paired with a sealed reference solution and a task-specific grader whose weighted, mostly executable criteria award partial credit; must-pass blockers and pass/fail gates encode what a maintainer would refuse to merge. The tasks span 87 open-source projects and 11 languages. The median ticket is 148 words, while the median reference solution changes 471 lines. Every task image is calibrated before release so that the reference solution scores 1.0 and both the untouched repository and an inert change score 0; nearly half of the tasks also ship alternative, lazy and adversarial control patches that the grader must classify correctly. The final 100 tasks were sel
    
[^162]: STAR：基于零样本学习的语义关键性分层无线电资源分配

    STAR: Semantic-Criticality Tiered Allocation of Radio Resources via Zero-Shot Learning

    [https://arxiv.org/abs/2611.13238](https://arxiv.org/abs/2611.13238)

    提出了STAR框架，利用冻结视觉-语言模型的注意力导出逐像素语义关键性指示器，对决定任务答案的关键像素进行分层高保真无线电资源分配，且仅需更改提示词即可零样本适配新任务与领域。

    

    在远程推理中，边缘设备通过无线链路将高分辨率图像传输至接收端，由人工智能（AI）模型回答关于这些图像的任务特定问题，以实现可靠决策。这些答案往往由一小部分任务因果像素（如文本像素）决定，这些像素一旦损坏就会改变答案，因此必须以高保真度进行传输。传统传输方式对所有像素一视同仁地加以保护，因而要么消耗大量资源元素（RE），要么在低信噪比（SNR）下损坏这些关键区域。为克服这些局限，本文提出了一种零样本、领域无关的框架，称为语义关键性分层资源分配（STAR）。具体而言，通过冻结的视觉-语言模型的注意力导出逐像素的语义关键性指示器（SCI），因此只需更改提示词即可处理新任务和新领域，无需……

    arXiv:2611.13238v1 Announce Type: cross  Abstract: In remote inference, edge devices transmit high-resolution images over wireless links to a receiver, where an artificial intelligence (AI) model answers task-specific questions about these images to enable reliable decisions. These answers are often determined by a small set of task-causal pixels, such as those of text, whose corruption changes the answer, so these pixels must be delivered with high fidelity. Conventional transmission protects every pixel equally, so it either consumes many resource elements (REs) or corrupts these regions at low signal-to-noise ratio (SNR). To overcome these limitations, in this paper, a zero-shot, domain-agnostic framework, termed semantic-criticality tiered allocation of resources (STAR), is proposed. In particular, a per-pixel semantic-criticality indicator (SCI) is derived from the attention of a frozen vision-language model, so new tasks and domains are handled by changing only the prompt, withou
    
[^163]: NCP：验证器引导的权重空间控制策略优化

    NCP: Verifier-Guided Optimization of Weight-Space Control Policies

    [https://arxiv.org/abs/2611.13236](https://arxiv.org/abs/2611.13236)

    本文提出神经控制平面（NCP），一种闭环系统，将预训练模型的推理数据平面与控制策略分离，自动合成、评估并管理临时的权重空间干预规则，同时通过基础模型参数的可恢复性保证安全，实验中成功将 SmolLM-135M 的验证集 token 重复率从 0.4133 降至 0.2191。

    

    我们提出了神经控制平面（Neural Control Plane, NCP），这是一个闭环系统，能够观察语言模型的执行过程、合成候选的权重空间干预规则、对其进行评估，并维护一个已采纳策略的版本化表格。NCP 将预训练模型的推理数据平面与选择临时参数干预的控制策略分离开来。W-ISA 提供事务性执行和参数恢复；NCP 提供遥测、候选合成、准入标准、优先级设置和规则生命周期管理。优化状态由活跃规则和已评估候选的存档组成，而基础模型参数在每个请求之后都会被恢复。在本地 SmolLM-135M-Instruct 实验中，NCP 从开发遥测数据中生成了三个候选规则，并在独立的任务样例和一个包含 4,095 个目标的 WikiText-2 切片上进行了评估。其中一个投影将验证集的 token 重复率从 0.4133 降低到 0.2191，困惑度（摘要在此处被截断）

    arXiv:2611.13236v1 Announce Type: new  Abstract: We present the Neural Control Plane (NCP), a closed-loop system that observes language-model execution, synthesizes candidate weight-space intervention rules, evaluates them, and maintains a versioned table of admitted policies. NCP separates a pretrained model's inference data plane from the control policy that selects temporary parameter interventions. W-ISA supplies transactional execution and parameter restoration; NCP supplies telemetry, candidate synthesis, admission criteria, priorities, and rule lifetime management. The optimization state consists of active rules and an archive of evaluated candidates, while base-model parameters are restored after each request. In a local SmolLM-135M-Instruct experiment, NCP generated three candidates from development telemetry and evaluated them on separate task examples and a 4,095-target WikiText-2 slice. One projection reduced validation token repetition from 0.4133 to 0.2191, with a perplex
    
[^164]: 基于判别的决策：面向医疗推理的差异化证据

    Decisions by Discrimination: Differential Evidence for Medical Reasoning

    [https://arxiv.org/abs/2611.13233](https://arxiv.org/abs/2611.13233)

    该论文提出了一种基于检索的差异化证据框架，通过提取每个选项特有的决定性差异化证据并组织成简洁的证据提示，来更好地区分医学多选题中的易混淆选项。

    

    大型语言模型（LLMs）正被越来越多地研究用于临床决策支持，而医学问答为评估这一能力提供了测试平台。特别是，医学多选题（MCQs）反映了临床决策中的一个核心挑战：看似合理的选项往往共享重叠的证据，而只有少数决定性因素能够将正确选项与错误选项区分开来。然而，现有的基于检索的方法和面向推理的方法主要侧重于提供广泛相关的知识或引导推理路径，而非识别能够区分易混淆选项的紧凑差异化证据。在本工作中，我们提出了一个面向医学多选题的基于检索的差异化证据框架。我们的方法提取针对每个选项的差异化证据片段，并将其组织成简洁的证据提示用于答案选择。这种设计能更好地将外部知识与差异（摘要在此处被截断）

    arXiv:2611.13233v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly studied for clinical decision support, and medical question answering provides a testbed for evaluating this capability. In particular, medical multiple-choice questions (MCQs) reflect a core challenge in clinical decision-making: plausible options often share overlapping evidence, while only a few decisive factors distinguish the correct option from incorrect ones. However, existing retrieval-based and reasoning-oriented methods mainly focus on supplying broadly relevant knowledge or eliciting reasoning paths, rather than identifying the compact differential evidence that separates confusing alternatives. In this work, we present a retrieval-grounded differential evidence framework for medical MCQs. Our method extracts option-specific pieces of differential evidence and organizes them into concise evidence prompts for answer selection. This design better aligns external knowledge with differ
    
[^165]: W-ISA：权重空间干预程序的事务化执行

    W-ISA: Transactional Execution of Weight-Space Intervention Programs

    [https://arxiv.org/abs/2611.13232](https://arxiv.org/abs/2611.13232)

    该论文提出了 W-ISA——一套用于在预训练语言模型上事务化执行权重空间干预程序（参数更新、消融、秩一变换等）的软件指令集与运行时，可在请求结束后自动恢复模型参数，其效果与等效的 PyTorch 钩子实现完全一致。

    

    我们提出了 W-ISA，这是一个用于在预训练语言模型上执行权重空间干预程序的软件指令集和事务化运行时。经过形状检查的指令可表达参数更新、投影、消融以及秩一变换。该运行时在请求作用域的执行上下文中应用这些程序，并在完成或中断后恢复已保存的参数。基于激活的匹配-动作规则将测量到的模型状态与干预选择相连接。我们在 SmolLM-135M-Instruct 和 Qwen2.5-1.5B-Instruct 上评估了参数恢复与重放功能，并在后者上将物化的秩一更新与等效的 PyTorch 钩子实现进行了比较。在 45 次定时请求中，两种干预实现在 32、128 和 512 个 token 长度下均产生完全相同的贪心解码序列；其最大提示 logits 差异仅为 0.000184。在完整的 WikiText-2 原始测试集上，SmolLM 的注意力头消融与 MLP 投影的组合变更（摘要在此处截断）

    arXiv:2611.13232v1 Announce Type: new  Abstract: We present W-ISA, a software instruction set and transactional runtime for executing weight-space intervention programs on pretrained language models. Shape-checked instructions express parameter updates, projections, ablations, and rank-one transformations. The runtime applies these programs within request-scoped execution contexts and restores saved parameters after completion or interruption. Activation-based match-action rules connect measured model states to intervention selection. We evaluate restoration and replay on SmolLM-135M-Instruct and Qwen2.5-1.5B-Instruct, and compare materialized rank-one updates with equivalent PyTorch hooks on the latter. In 45 timed requests, both intervention implementations produce identical greedy sequences at 32, 128, and 512 tokens; their maximum prompt-logit difference is 0.000184. On the complete WikiText-2 raw test split, the combined SmolLM attention-head ablation and MLP projection changes pe
    
[^166]: 删除敏感性是模型的一种属性：对七个开放推理模型的三条件测量

    Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models

    [https://arxiv.org/abs/2611.13223](https://arxiv.org/abs/2611.13223)

    本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。

    

    推理模型在多轮工具会话的每一轮都会写下一条推理轨迹，而后续轮次会读取这段历史。在94k条ShareGPT对话中，73%的用户交互跨越多轮。上游提供商将每条推理轨迹绑定到其发出者，因此跨发出者转发的请求到达时其历史已被删除，而这种剥离操作带来的任务成本此前从未被测量。据我们所知，本文首次对多轮工具调用中完整推理历史删除进行了直接、盲测、多种子的测量，涵盖三种渲染条件、七个开放权重推理模型、三个随机种子、794个任务以及官方检查器。相同的删除操作导致的性能变化跨度为-4.4到+51.3分，且两端均不包含零；对于唯一一个从0%到100%删除比例重新测试的模型，其读数变化最多仅1.5分。删除敏感性是模型层面的属性。该读数跟随模型本身并保持稳定（原文在此处截断）。

    arXiv:2611.13223v1 Announce Type: cross  Abstract: Reasoning models write a reasoning trace at every turn of a multi-turn tool session, and later turns read this history. Across 94k ShareGPT conversations, 73 percent of user interactions span multiple turns. Upstream providers bind each trace to its issuer, so a request forwarded across issuers arrives with its history deleted, and the task cost of this stripping remains unmeasured. To the best of our knowledge, this paper presents the first direct, blind, multi-seed measurement of full reasoning-history deletion in multi-turn tool calling, through three rendering conditions, seven open-weight reasoning models, three seeds, 794 tasks, and the official checker. The same deletion spans -4.4 to +51.3 points with both ends excluding zero, and for the one model re-dosed from 0 to 100 percent deletion, the reading varies by at most 1.5 points. Deletion sensitivity is a model-level property. The reading follows the model itself and stays cons
    
[^167]: 《以同样的方式犯错：跨提示共享在RLVR中既抑制又放大奖励作弊》

    Wrong the Same Way: Cross-Prompt Sharing Both Suppresses and Amplifies Reward Hacking in RLVR

    [https://arxiv.org/abs/2611.13218](https://arxiv.org/abs/2611.13218)

    本研究在可控环境中证明RLVR在KL正则化最优时只是对被接受集合重新加权的过滤器（奖励作弊率精确继承自基础模型），而跨提示参数共享则通过提示间奖励与经验切核的协方差项产生漂移，使接受同一错误答案的提示推高它、拒绝的提示压低它，从而同时抑制并放大奖励作弊。

    

    关于可验证奖励强化学习（RLVR）如何与不完善的验证器交互，目前存在两种并存的观点：一种认为奖励作弊是从基础模型继承而来的，另一种认为强化学习会将其放大。我们在一个足够小的受控环境中研究这两种机制，以便精确计算整个响应分布。我们改变了漏洞的普适性（即接受同一错误响应的提示所占比例）以及验证器的静态假阳性率。在KL正则化的最优解处，RLVR充当一个过滤器，仅对被接受的集合重新加权而不改变其组成，因此作弊率被精确地继承。以此为零模型，一个已知的分解将与该零模型的偏差写为各提示上奖励与经验切核之间协方差的总和。在没有参数共享的情况下，来自其他提示的项会消失。而在存在参数共享时，每个接受错误答案的提示都会推高该错误答案，每个拒绝它的提示则会将其压低……（摘要至此截断）

    arXiv:2611.13218v1 Announce Type: new  Abstract: Two accounts of how reinforcement learning with verifiable rewards (RLVR) interacts with an imperfect verifier coexist: in one, reward hacking is inherited from the base model; in the other, RL amplifies it. We study both in a controlled environment small enough to compute the entire response distribution exactly. We vary the universality of a leak, or the fraction of prompts that accept the same wrong response, and the verifier's static false-positive rate. At its KL-regularized optimum, RLVR acts as a filter that reweights the accepted set without altering its composition, so the hack rate is inherited exactly. Taking that as a null model, a known decomposition writes the drift from it as a sum over prompts of covariances between reward and the empirical tangent kernel. The terms from other prompts vanish without parameter sharing. With sharing, every prompt that accepts a wrong answer pushes it up and every prompt that rejects it push
    
[^168]: Claude Code作为智能体式AI评分系统对手写大学考试的评分可靠性研究

    Reliability of Claude Code as an Agentic AI Grading System for Handwritten University Exams

    [https://arxiv.org/abs/2611.13217](https://arxiv.org/abs/2611.13217)

    本研究首次系统评估了基于Claude Opus 4.6的智能体式系统Claude Code对手写大学考试评分的可靠性，考察其与专家人工评分在总分及五种题型上的一致性、跨会话的稳定性以及AI反馈的准确性。

    

    本研究考察了Claude Code（Claude Opus 4.6大型语言模型的一种智能体式部署）作为手写大学考试评分系统的可靠性。研究围绕四个问题展开：与专家人工评分在总分上的一致性（RQ1）、在从选择题到评价性判断共五种题型上的一致性（RQ2）、在一周间隔的两次独立评分会话中的稳定性（RQ3），以及AI生成反馈的准确性（RQ4）。在三个评分量规文件的引导下，Claude Code在单个智能体会话中批改了沙特一所计算机图形学课程中60名同意参与学生的20分制期中试卷。可靠性在互不重叠的样本上进行估计：RQ1、RQ2和RQ4使用20份试卷，RQ3使用15份试卷。分析采用了皮尔逊相关系数、平均绝对误差、Bland-Altman分析、加权Cohen kappa、组内相关系数，以及一套区分诊断性反馈与确认性反馈的四类编码方案。

    arXiv:2611.13217v1 Announce Type: new  Abstract: This study examines the reliability of Claude Code, an agentic deployment of the Claude Opus 4.6 large language model, as a grading system for handwritten university exams. Four research questions addressed agreement with expert human grading on total scores (RQ1), agreement across five question types from multiple choice to evaluative judgement (RQ2), consistency across two independent grading sessions one week apart (RQ3), and the accuracy of AI-generated feedback (RQ4). Claude Code graded the 20-mark mid-term papers of 60 consenting students in a Saudi computer graphics course in a single agentic session, guided by three rubric files. Reliability was estimated on disjoint samples: 20 papers for RQ1, RQ2 and RQ4, and 15 papers for RQ3. Analyses used Pearson correlation, mean absolute error, Bland-Altman analysis, weighted Cohen kappa, intraclass correlation, and a four-category coding scheme separating diagnostic from confirmation feed
    
[^169]: 在分歧中计数：基于存在分歧的行政文本AI阅读者的美元加权测量、选择性机器标注与预算化人工复核

    Counting Under Disagreement: Dollar-Weighted Measurement, Selective Machine Labelling and Budgeted Human Review from Disagreeing AI Readers of Administrative Text

    [https://arxiv.org/abs/2611.13215](https://arxiv.org/abs/2611.13215)

    针对多个AI阅读者对行政文本标注存在分歧、且测量目标为货币总额的场景，本文提出了融合选择性机器标注与预算化人工复核的美元加权估计框架，在可控美元加权误差下实现对总额的准确测量。

    

    我们研究AI辅助测量问题，其场景是：一个文本标签需应用于大量行政记录，测量目标为货币总额，而用于校准机器的人工阅读结果本身存在分歧。测试平台来自美国税务申报中的121,070条拨款用途文本（共799,675条记录，总额1264亿美元），由一个基于规则的分类器和三个语言模型进行标注；同时按设计方案抽取651条文本，由三人在同一编码手册下独立阅读。我们在泊松抽样下给出了固定预测变量差分估计量、以及交叉拟合预测变量（其折在抽样前即分配至整个抽样框）的精确条件，并通过仿真评估所实现的固定样本量程序：8个预测变量自身总额介于112亿至150亿美元之间，经校正后相互差异在4亿美元以内。两种按规定的美元加权误差接受机器标签的规则——美元度量下的加权保形p值方法，以及基于误差设计估计的Learn-then-Test方法——（原文摘要在此处截断）

    arXiv:2611.13215v1 Announce Type: new  Abstract: We study AI-assisted measurement when one text label applies to many administrative records, the target is a monetary total, and the human readings that anchor the machines disagree. The testbed is 121,070 grant-purpose texts from US tax filings (799,675 records, \$126.4B), labelled by a rule-based classifier and three language models, with a designed sample of 651 texts read independently by three people under one codebook. We state exact conditions for a fixed-predictor difference estimator and for a cross-fitted predictor whose folds are assigned to the whole frame before sampling, under Poisson sampling, and assess the implemented fixed-size procedures by simulation: 8 predictors whose own totals span \$11.2-15.0B agree to within \$0.4B once rectified. Two rules that accept machine labels at a stated dollar-weighted error, weighted conformal p-values under a dollar measure and Learn-then-Test on the design estimate of the error, are 
    
[^170]: 言语之前，超越语音：评估幼儿期的非语言社会推理能力

    Before Words, Beyond Speech: Evaluating Nonverbal Social Reasoning in Early Childhood

    [https://arxiv.org/abs/2611.13213](https://arxiv.org/abs/2611.13213)

    该论文提出了首个以儿童为中心的非语言社会推理基准NEST，通过人工标注的约12秒自然情境视频片段，以及将基础感知与高阶推理相分离的组合式评估方案，填补了多模态视频理解领域对幼儿非语言社会互动评估的空白。

    

    儿童最早的岁月是一个高强度的社会学习时期，在此期间，意义通过眼神注视、手势和身体接触而构建。然而，社会互动的这一关键方面在多模态视频理解领域仍未得到充分评估，因为现有基准主要优先关注以对话驱动的成人互动。我们通过NEST（自然情境下的幼儿社会互动）基准来填补这一空白，这是首个以儿童为中心的非语言社会推理基准。NEST包含大量人工标注的视频片段（约12秒），这些片段在自然情境、文化背景和发展阶段方面都经过精心挑选以确保多样性。除了数据匮乏之外，该领域针对性的模型改进还受到粗略评估方案的阻碍，这些方案将基础感知与高阶推理混为一谈。为解决这一问题，NEST引入了一种组合式评估方案，在原子级交互线索的基础上构建（摘要在此处截断）……

    arXiv:2611.13213v1 Announce Type: cross  Abstract: Children's earliest years are a period of intense social learning, during which meaning is built through gaze, gesture, and physical contact. Yet, this critical aspect of social interaction remains under-evaluated in multimodal video understanding, as existing benchmarks prioritize the conversation-driven interactions of adults. We address this gap with \bench ($\textbf{N}$aturalistic $\textbf{E}$arly-childhood $\textbf{S}$ocial in$\textbf{T}$eractions), the first child-centric benchmark for nonverbal social reasoning. \bench comprises \totalvids manually annotated clips ($\sim$12s) curated for diversity across naturalistic settings, cultural contexts, and developmental stages. Beyond a lack of data, targeted model improvement in this domain is hindered by coarse evaluation schemas that conflate basic perception with higher-order reasoning. To resolve this, \bench introduces a compositional schema that builds atomic interaction cues in
    
[^171]: 面向OOD弹性运行时预测的XAI驱动置信度估计

    XAI-Driven Confidence Estimation for OOD-Resilient Runtime Prediction

    [https://arxiv.org/abs/2611.13199](https://arxiv.org/abs/2611.13199)

    该论文提出LEX——首个用于基于机器学习资源管理的事后XAI运行时置信度估计框架，通过高斯混合模型聚类训练输入嵌入并关联逐簇预测误差界限，无需重训练即可提供细粒度的输入级置信度信号，在OOD工作负载上实现约94%的包含率，并产生比全局基线紧约17%的置信度界限。

    

    基于机器学习的资源管理（RM）模型在运行时面对分布外（OOD）输入时性能会退化，然而文献中现有方法依赖于单一的全局误差裕度，这种方式过于保守，无法区分可靠预测与灾难性故障。我们提出了轻量级运行时可解释性（LEX），这是首个用于基于机器学习的资源管理中输入特定运行时置信度估计的事后XAI框架。LEX使用高斯混合模型对训练输入空间的嵌入进行聚类，并将每个簇的预测误差界限关联到输入样本，从而无需重新训练即可实现细粒度的运行时置信度信号。在Intel Alder Lake平台上，针对最先进的基于LSTM的IPS预测模型、跨34个基准应用程序进行评估，LEX在未见工作负载上实现了约94%的包含率，并且产生的逐输入置信度界限比全局基线紧约17%，而其运行时开销几乎可以忽略。

    arXiv:2611.13199v1 Announce Type: cross  Abstract: Machine learning-based resource management (RM) models degrade under out-of-distribution (OOD) inputs at runtime, yet existing approaches in the literature rely on a single global error margin that is overly conservative and fails to distinguish reliablepredictions from catastrophic failures. We propose Lightweight Runtime EXplainability (LEX), the first post-hoc XAI framework for input-specific runtime confidence estimation in ML-based RM. LEX clusters the embeddings of the training input space using a Gaussian Mixture Model and associates per-cluster prediction error bounds to input samples, enabling fine-grained runtime confidence signals without retraining. Evaluated on a state-of-the-art LSTM-based IPS forecasting model on Intel Alder Lake, across 34 benchmark applications, LEX achieves ~94% containment on unseen workloads and produces ~17% tighter per-input confidence bounds than global baselines. With its negligible runtime over
    
[^172]: 用于多模态归因的结构化核代理模型

    Structured Kernel Surrogate Models for Multimodal Attribution

    [https://arxiv.org/abs/2611.13198](https://arxiv.org/abs/2611.13198)

    提出结构化核代理模型及SKS算法，利用结构化核岭回归捕捉样本间与模态间的交互作用，首次解决多模态系统中的跨样本模态归因问题。

    

    我们研究了跨样本模态归因问题，即测量一个样本中的某种模态如何影响其他多模态样本的学习结果。这个问题在多模态系统中至关重要，因为某种数据模态（例如文本描述）对于某个样本（例如自身就能解释的图像）可能是冗余的，但同样的数据模态对另一个样本来说却可能是唯一的信息来源，或者需要与其他模态结合时才具有信息价值。先前的工作主要集中在单样本级别的影响估计上，如何捕捉跨样本和跨模态的交互仍然是一个悬而未决的问题。为了应对上述场景，我们提出了结构化核代理模型，这是一种在联合训练时从样本和模态到预测结果的代理映射。我们设计了一种算法SKS，该算法应用具有结构化复合结构的核岭回归（原文在此处截断）。

    arXiv:2611.13198v1 Announce Type: cross  Abstract: We study the problem of cross-sample modality attribution, which involves measuring how one modality in one sample affects the learning outcome in other multimodal samples. This problem is crucial in a multimodal system because a data modality (such as a text description) may be redundant for one sample (a self-explanatory image), but the same data modality would otherwise be uniquely informative for another sample, or informative when combined with other modalities. Prior work focuses on influence estimation at the level of one sample, and it remains an open question how one should capture the interactions across both samples and modalities. To capture scenarios like the above, we propose structured kernel surrogate models, which are surrogate mappings from both samples and modalities to the prediction outcome when they are trained together. We design an algorithm, SKS, which applies kernel ridge regression with a structured, composit
    
[^173]: 从流程模型到可组合的智能体工作流：结合BPMN、智能体与企业知识

    From Process Models to Composable Agentic Workflows: Combining BPMN, Agents and Enterprise Knowledge

    [https://arxiv.org/abs/2611.13196](https://arxiv.org/abs/2611.13196)

    本文提出基于BPMN的智能体工作流方法，通过“递归能力组合”将BPMN流程模型转换为可执行工作流，用传统BPMN控制确定性的流程、决策与人工审批，仅在选定环节引入LLM推理，从而让智能体能够执行业务流程并将其封装为可重用的智能体工具。

    

    基于LLM的智能体能够进行推理、使用工具并相互协作。然而，协调日益复杂的智能体应用变得难以理解、监控和改进。分散在提示词、工具调用和应用逻辑中的决策使得工作流变得隐式化。我们提出了**递归能力组合**方法，使智能体能够执行业务流程，并将其作为可重用的智能体工具向外暴露。本文提出了一种基于BPMN的智能体工作流方法，将显式编排与自主推理相结合。SemTalk BPMN模型通过LangGraph/Pregel运行时被转换为可执行的工作流。智能体任务在选定的活动中引入LLM推理，而传统的BPMN则控制流程流转、确定性决策、系统交互和人工审批。智能体参与者为智能体行为或其他可执行流程提供可重用的接口。BPMN编排使多智能体系统（摘要在此处截断）……

    arXiv:2611.13196v1 Announce Type: new  Abstract: LLM-based agents can reason, use tools and collaborate. However, coordinating increasingly complex agentic applications becomes difficult to understand, monitor and improve. Decisions distributed across prompts, tool calls and application logic make workflows implicit. We introduce **Recursive Capability Composition**, enabling agents to execute business processes and expose them as reusable Agentic Tools.   This paper presents a BPMN-based approach to Agentic Workflows, combining explicit orchestration with autonomous reasoning. SemTalk BPMN models are transformed into executable workflows using a LangGraph/Pregel runtime. Agentic Tasks introduce LLM reasoning at selected activities, while conventional BPMN controls process flows, deterministic decisions, system interactions and human approvals. Agentic Participants provide reusable interfaces for agent behavior or other executable processes.   BPMN orchestration makes multi-agent syste
    
[^174]: 基于跨模态图注意力机制的多模态道路封闭制图

    Multimodal Road Closure Mapping via Cross-Modal Graph Attention

    [https://arxiv.org/abs/2611.13192](https://arxiv.org/abs/2611.13192)

    本文提出一个统一的多模态多任务框架，通过跨模态Transformer注意力整合图结构地图数据、卫星影像和街景图像，在单一架构中同时实现道路封闭分类与受影响路段的关联。

    

    准确且及时地识别道路封闭对于大规模导航和物流系统至关重要，但由于需要同时对视觉证据、空间上下文和道路网络拓扑进行联合推理，该任务仍然充满挑战。现有方法要么依赖人工地图更新，要么仅专注于检测与封闭相关的目标物，而无法确定具体受影响的路段。在本工作中，我们提出了一个统一的多模态多任务框架，可在单一学习架构中同时完成道路封闭分类（无封闭、部分封闭、完全封闭）和路段关联两项任务。我们的模型采用模态专用编码器和跨模态Transformer注意力机制，整合了图结构地图数据、经投影街级检测结果增强的卫星影像以及街景图像，从而实现对视觉线索和网络拓扑的显式推理。该模型在从大规模采集的近2万个真实场景数据集上进行了评估……（摘要截断）

    arXiv:2611.13192v1 Announce Type: cross  Abstract: Accurate and timely identification of road closures is critical for large-scale navigation and logistics systems, yet remains challenging due to the need to jointly reason over visual evidence, spatial context, and road network topology. Existing approaches either rely on manual map updates or focus on detecting closure-related objects without resolving which road segment is affected. In this work, we introduce a unified multimodal and multitask framework that jointly performs road closure classification (no, partial, complete) and road segment association within a single learning architecture. Our model integrates graph-structured map data, satellite imagery augmented with projected street-level detections, and street-view images using modality-specific encoders and cross-modal transformer attention, enabling explicit reasoning over visual cues and network topology. Evaluated on nearly 20K real-world scenarios collected from a large-s
    
[^175]: AgentScript-Eval：在企业级领域专用语言（DSL）中评估LLM与智能体代码生成能力的基准测试

    AgentScript-Eval: Benchmarking LLMs and Agents for Code Generation in an Enterprise DSL

    [https://arxiv.org/abs/2611.13183](https://arxiv.org/abs/2611.13183)

    该论文提出了AgentScript-Eval基准，包含100个针对Salesforce企业级声明式DSL（AgentScript）的真实任务，首次系统评估了LLM和智能体在企业DSL上的代码生成能力，发现该任务极具挑战性，最佳模型零样本通过率仅为25.0%。

    

    领域专用语言（DSL）在企业软件中被广泛使用，为从数据访问到基础设施自动化等复杂任务提供了紧凑的接口。平台正日益将DSL扩展至智能体场景，开发者通过声明智能体的主题、允许的操作和防护机制来构建智能体，而非编写通用代码。大型语言模型（LLM）擅长编写通用代码，但这种能力几乎完全是在公开预训练数据中广泛存在的语言上测量的，而企业级DSL既不常见也不公开。我们提出了AgentScript-Eval，这是一个基于执行结果评分的基准，包含100个针对AgentScript的真实任务。AgentScript是Salesforce在Agentforce中用于构建对话式智能体的声明式DSL。任务涵盖根据规范编写智能体、调试损坏的定义以及优化可正常工作的定义。通过对五个模型进行零样本评估，我们发现这一领域极具挑战性：表现最好的模型仅达到25.0（原文截断，应为25.0%）的通过率。

    arXiv:2611.13183v1 Announce Type: cross  Abstract: Domain-specific languages (DSLs) are widely used in enterprise software, giving a compact interface to complex tasks, from data access to infrastructure automation. Platforms increasingly extend them to agents, where a developer declares the topics, permitted actions, and guardrails of an agent rather than writing general-purpose code. Large language models (LLMs) write general-purpose code well, but that ability has been measured almost entirely on languages abundant in public pretraining data, and enterprise DSLs are neither abundant nor public. We introduce AgentScript-Eval, an execution-graded benchmark of 100 real tasks for AgentScript, the declarative DSL that Salesforce uses in Agentforce to build conversational agents. Tasks span authoring an agent from a specification, debugging a broken definition, and optimizing a working one. Evaluating five models zero-shot, we find that the regime is hard: the best model reaches only 25.0
    
[^176]: AI对齐既无法解决也非充分之道

    AI Alignment Is Neither Solvable nor Sufficient

    [https://arxiv.org/abs/2611.13167](https://arxiv.org/abs/2611.13167)

    本文论证AI对齐这一目标存在根本缺陷：对齐定义不清、无法求解，且即使实现对齐也不足以化解风险，因为真正需要应对的是人机混合社会这一复杂系统中产生的涌现性风险。

    

    人类与AI系统越来越多地在相同的社会和经济环境中行动。对齐，即确保AI系统的行为符合人类价值观的目标，被广泛认为是确保AI造福社会并避免危害的核心挑战。我们认为这一努力是误入歧途的。首先，对齐没有良好的定义，因为它假定存在一个足够稳定且可辨识的人类意图或价值观集合。其次，在可以具体定义对齐概念的情况下，它是不可解的。第三，在看似可以解决对齐问题的情况下，解决它也不一定会减轻AI的负面影响。一个关键原因是，对齐通常被框定为单个AI系统的属性，而由此形成的人机混合社会却是由互动和反馈回路塑造的复杂系统。因此，涌现的风险，而不仅仅是由个别未对齐系统带来的风险——很可能会（原文摘要在此处被截断）

    arXiv:2611.13167v1 Announce Type: new  Abstract: Humans and AI systems increasingly act within the same social and economic environments. Alignment, the goal of ensuring that AI systems act in accordance with human values, is widely regarded as a central challenge for ensuring that AI benefits society while avoiding harm. We argue that this endeavor is misguided. First, alignment is not well defined, as it presumes a sufficiently stable and identifiable set of human intentions or values. Second, where a concrete notion of alignment can be specified, it is not solvable. Third, where alignment appears solvable, solving it would not necessarily mitigate the negative consequences of AI. A key reason is that alignment is typically framed as a property of individual AI systems, while the resulting hybrid societies of humans and machines are complex systems shaped by interactions and feedback loops. Emergent risks, and not only those posed by individual misaligned systems, are therefore likel
    
[^177]: 认知多智能体搜索与救援：面向协作机器人的基于语言接地的分层强化学习

    Cognitive Multi-Agent Search and Rescue: Language-Grounded Hierarchical Reinforcement Learning for Collaborative Robotics

    [https://arxiv.org/abs/2611.13161](https://arxiv.org/abs/2611.13161)

    该论文提出了一种基于语言接地的分层强化学习（LG-HRL）认知多智能体框架，利用大语言模型将人类自然语言指令转化为结构化语义线索，结合共享认知工作空间，实现无人机与地面无人车在搜索救援任务中的团队级协调协作。

    

    有效的搜索与救援（SAR）行动需要能够协调多个异构智能体、适应动态危险并与人类任务意图保持一致的自主系统。现有的语言引导强化学习（RL）方法主要集中于单智能体指令遵循，缺乏在复杂、部分可观测环境中进行团队级协调的机制。本文提出了一种基于语言接地的分层强化学习（LG-HRL）的认知多智能体框架，该框架集成了自然语言理解、分层策略分解和共享认知工作空间，以在搜索与救援场景中实现协调一致的多智能体行为。人类指令由大语言模型（LLM）解释并转化为结构化语义线索，用以指导无人机（UAV）和地面无人车（UGV）智能体的高层策略选择和角色分配。

    arXiv:2611.13161v1 Announce Type: new  Abstract: Effective search and rescue (SAR) operations require autonomous systems capable of coordinating multiple heterogeneous agents, adapting to dynamic hazards, and aligning with human mission intent. Existing language-guided reinforcement learning (RL) methods primarily focus on single-agent instruction following and lack mechanisms for team-level coordination in complex, partially observable environments. This paper presents a Cognitive Multi-Agent Framework based on Language-Grounded Hierarchical Reinforcement Learning (LG-HRL) that integrates natural-language understanding, hierarchical policy decomposition, and a shared cognitive workspace to achieve cohesive multi-agent behavior in SAR scenarios. Human instructions are interpreted by a large language model (LLM) and transformed into structured semantic cues that guide high-level strategy selection and role assignment across UAV and UGV agents. A formal Language-Grounded Multi-Agent Hier
    
[^178]: 你的VLA有多少成分其实是kNN

    How Much of Your VLA Is kNN

    [https://arxiv.org/abs/2611.13160](https://arxiv.org/abs/2611.13160)

    研究揭示LIBERO基准上VLA的高成功率很大程度上只是近邻演示检索的体现——无需任何训练的检索方法即可达到91.7%成功率，并提出局部动作一致性指标来衡量检索质量并指导检索键训练。

    

    视觉-语言-动作模型目前在LIBERO仿真基准上的成功率已超过95%，但这些性能中有多少需要超越训练演示之外的泛化能力仍不清楚。在这项工作中，我们证明检索近邻演示并复用其动作块，在不进行任何策略训练的情况下即可在LIBERO上达到91.7%的成功率。在LIBERO-Plus扰动下，检索方法在所有被评估的策略中表现出最小的性能下降。在九个公开可用的VLA上，微调主要使视觉-语言骨干成为更好的检索键，而VLA输出的动作仍然与演示动作十分接近。此外，我们提出了局部动作一致性，这是一种离线指标，用于衡量表示空间中的相邻观测是否也与相似的未来动作相关联。我们证明LAC能够追踪检索的成功率，并可用于指导检索键的训练。我们的评估……

    arXiv:2611.13160v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models now exceed 95% success on the LIBERO simulation benchmark, yet it remains unclear how much of this performance requires generalization beyond the training demonstrations. In this work, we show that retrieving nearby demonstrations and reusing their action chunks achieves a success rate of 91.7 % on LIBERO without policy training. Under LIBERO-Plus perturbations, retrieval shows the smallest drop of all evaluated policies. Across nine publicly available VLAs, fine-tuning mostly makes the vision-language backbone a better retrieval key, and the actions of the VLAs remain close to the demonstrations. In addition, we introduce Local Action Consistency (LAC), an offline metric that measures whether nearby observations in representation space are also associated with similar future actions. We show that LAC tracks the success rate of retrieval and can guide the training of a retrieval key. Our evaluation a
    
[^179]: 万物各归其位：注意力是一个无线电接收头

    Everything in its right place: attention is a radio head

    [https://arxiv.org/abs/2611.13146](https://arxiv.org/abs/2611.13146)

    该论文提出注意力机制可以从物理同步中涌现——把历史词元建模为频率随时间漂移的振荡器、把当前词元建模为无线电接收器，通过求解Kuramoto同步问题闭式导出一个具有“锁定/静默”两相的注意力层，从而将Transformer注意力重新解释为无线电调谐接收过程，并自然实现选择、弃权、计数及超长序列泛化。

    

    注意力是语言模型的核心，是一种基于词元的数字计算：指数函数负责选择词元，除法负责归一化，位置编码负责设定顺序。我们证明，注意力也可以从接收器与其上下文词元之间的同步中涌现。每个历史词元都是一个振荡器，其状态携带自身内容，其频率随着“年龄”增长而漂移。当前词元通过内容进行选择——将天线与查询对齐；也通过位置进行选择——用调谐旋钮设定接收器的频率。对该接收器求解Kuramoto问题，可以得到一个闭式解的注意力层，它具有两个相位：当耦合强度高于由上下文设定的临界值时，接收器进入锁定状态，其权重被归一化；低于临界值时，接收器保持静默，其权重之和小于一。经过训练的网络利用这两个相位来实现选择、弃权和计数，在理论所预测之处失效，并能持续读取远超训练长度的序列。注意力的每一部分……（原文摘要在此处截断）

    arXiv:2611.13146v1 Announce Type: cross  Abstract: Attention, the core of language models, is a digital calculation over tokens: an exponential selects them, a division normalizes and a positional code sets the order. We show that attention can instead emerge from the synchronization of a receiver with its context tokens. Each past token is an oscillator whose state carries its content and whose frequency drifts with its age. The present token selects them by content, aligning an antenna with its query, and by position, setting the frequency of a receiver with its dial. Solving Kuramoto's problem for this receiver yields an attention layer in closed form with two phases: above a critical coupling set by the context, the receiver locks and its weights normalize; below it, it stays silent and its weights sum to less than one. Trained networks use both phases to select, abstain and count, fail where the theory predicts and keep reading far beyond the training length. Each part of attentio
    
[^180]: 教导执行框架：面向大语言模型智能体的学生感知技能迁移

    Teaching the Harness: Student-Aware Skill Transfer for LLM Agents

    [https://arxiv.org/abs/2611.13144](https://arxiv.org/abs/2611.13144)

    提出学生感知教学方法，教师通过构建学习者画像、假设诊断与探针测试来理解学生模型在固定执行框架下的局限性，并据此编写经验证的可复用技能，使任务成功率达 86.46%。

    

    有效的教学不仅需要对任务进行推理，还需要对学习者进行推理，这种能力被心智理论描述为从行为推断另一个智能体的潜在状态。我们探究当更强的大语言模型应用这一原则时，能否更好地教导一个固定的学生模型。学生是通过执行框架行动的大语言模型智能体，该框架固定了其工具、上下文和行动预算，因此对一个模型有效的程序可能会耗尽另一个模型的预算。我们提出了学生感知教学方法。教师构建学习者模型，将经验画像与学生局限性的假设相结合，通过诊断性探针测试这些假设，并针对诊断出的失败编写可复用的技能。只有当技能在匹配的开发任务上提升成功率后，才会被纳入技能库。在 Claude-Sonnet-5 教导 DeepSeek-V4-Flash 的实验中，SAT 在四次主要运行中达到 86.46% 的任务成功率，优于使用相同方法的无数画像教师……（摘要截断）

    arXiv:2611.13144v1 Announce Type: new  Abstract: Effective teaching requires reasoning about the learner as well as the task, a capacity that Theory of Mind describes as inferring another agent's latent state from its behavior. We ask whether a stronger LLM teaches a fixed student better when it applies this principle. The student is an LLM agent acting through a harness that fixes its tools, context, and action budget, so a procedure that helps one model can exhaust the budget of another. We introduce Student-Aware Teaching (SAT). The teacher builds a learner model that pairs an empirical profile with hypotheses about the student's limitations, tests these hypotheses with diagnostic probes, and writes reusable skills for the diagnosed failures. A skill enters the library only after it raises success on matched development tasks. With Claude-Sonnet-5 teaching DeepSeek-V4-Flash, SAT reaches 86.46% task success across four primary runs, leading both a profile-free teacher with the same f
    
[^181]: 人工智能体对数值算法的自主优化：在原子模拟中的应用

    Autonomous Optimization of Numerical Algorithms by an AI Agent: Application to Atomistic Simulation

    [https://arxiv.org/abs/2611.13143](https://arxiv.org/abs/2611.13143)

    提出AIOA框架，利用AI智能体在验证器守护的自动化改进循环中自主优化原子模拟的数值算法，实现核心计算模块提速22-24倍，并使结构弛豫和反应路径计算所需的力评估次数减少3-17倍。

    

    基于大语言模型构建的编码智能体在自动化改进循环中运行，能够发现更高效的算法。我们提出了AIOA框架，将这一方法引入原子模拟的数值算法中：智能体在一个由其无法篡改的验证器所守护的循环中，针对预先指定的指标改进算法。将该框架应用于广泛使用的原子模拟库ASE时，AIOA针对算法的两个方面进行优化：一是算法实现，在不改变结果的前提下重写代码使其运行更快；二是算法方法本身，重新设计使其以更少的力评估次数收敛。智能体使六个核心计算模块在Python（结合NumPy和SciPy）环境下平均提速22至24倍。在未见过的测试问题上，最优配置下得到的优化器在结构弛豫任务中所需力评估次数比L-BFGS少3.1至3.4倍，在反应路径计算中比标准爬坡图像弹性带（CI-NEB）方法少4.6至17倍。

    arXiv:2611.13143v1 Announce Type: cross  Abstract: Coding agents built on large language models, run in automated improvement loops, can discover more efficient algorithms. We present AIOA, a framework that brings this approach to the numerical algorithms of atomistic simulation: an agent improves an algorithm against a pre-specified metric in a loop guarded by verifiers it cannot alter. Applied to ASE, a widely used atomistic simulation library, AIOA targets two facets of an algorithm: its implementation, rewritten to run faster with unchanged results, and the method itself, redesigned to converge with fewer force evaluations. Agents made six kernels 22-24 times faster on average in Python with NumPy and SciPy. On unseen problems, optimizers from the best configuration needed 3.1-3.4 times fewer force evaluations than L-BFGS for structure relaxation and 4.6-17 times fewer than standard climbing-image nudged elastic band calculations for reaction paths. The agents' optimizers also cont
    
[^182]: DOBB：面向状态不确定性与异构动力学下多智能体避碰的动态定向包围盒

    DOBB: Dynamic Oriented Bounding Boxes for Multi-Agent Collision Avoidance under State Uncertainty and Heterogeneous Dynamics

    [https://arxiv.org/abs/2611.13142](https://arxiv.org/abs/2611.13142)

    提出动态定向包围盒（DOBB），一种在线几何表示方法，将状态估计不确定性、异构动力学及智能体响应特性转化为动态且依赖方向的安全边距，从而提升多智能体避碰在不确定条件下的可靠性。

    

    基于学习的去中心化避碰方法必须在状态估计不确定性和异构智能体动力学条件下保持可靠。现有方法通常通过训练域变化，或在运行时将不确定性与能力估计直接暴露给策略来处理这些因素。然而，这两种方式都需要策略自行推断这些量应如何改变安全交互所需的物理间隔，这可能使避碰行为与从训练分布中学到的关系耦合在一起。我们提出了动态定向包围盒（DOBB），这是一种在线几何表示方法，可将当前状态估计、不确定性以及智能体特定的响应特性映射为动态的、依赖方向的安全边距。我们推导出能够联合表征智能体DOBB之间相对构型与交互严重程度的几何特征，并配套设计了相应的奖励函数……

    arXiv:2611.13142v1 Announce Type: cross  Abstract: Learning-based decentralized collision avoidance must remain reliable under state-estimation uncertainty and heterogeneous agent dynamics. Existing methods typically address these factors through training-domain variation or by exposing uncertainty and capability estimates to the policy at runtime. However, both leave the policy to infer how these quantities should alter the physical separation required for safe interaction, which can couple collision-avoidance behavior to relationships learned from the training distribution. We introduce the Dynamic Oriented Bounding Box (DOBB), an online geometric representation that maps the current state estimate, uncertainty, and agent-specific response characteristics into a dynamic, direction-dependent safety margin. We derive geometric features that jointly characterize the relative configuration and severity of interactions between agents' DOBBs, along with a reward function that preserves the
    
[^183]: 答案一致但理由不一：大语言模型推理失败分类体系的测量局限性

    Agreeing on the Answer but Not the Reason: Measurement Limits of LLM Reasoning Failure Taxonomies

    [https://arxiv.org/abs/2611.13140](https://arxiv.org/abs/2611.13140)

    该论文表明，LLM推理失败分类的测量结果很大程度上取决于测量程序本身：过半失败案例可归入多个类别，类别占比随分类器测试顺序大幅波动（26.4%–76.4%），且人工标注者对类别归属的共识远低于对任务答案正确性的共识。

    

    大语言模型（LLM）推理失败分类体系正越来越多地被当作测量结果来报告：论文会声明模型错误中有多少比例落入某些命名类别，然后在模型之间比较这些比例。我们证明，对于其中一个类别，所报告的比例在很大程度上是测量方法本身的属性。我们定义了“组合崩塌”（Compositional Collapse，即模型注意到问题所需的某个子部分，随后又将其丢弃或错误应用），并在一个公开的多语言数学基准上跨越十二个模型和十一种语言进行了测量（共2,316条评分记录）。首先，50.1%的失败案例包含支持不止一个类别的证据，因此所报告的多数比例取决于分类器测试各类别的顺序：仅交换两个测试的顺序就会使该比例在十二个模型中的十一个里发生变化，而在所有六种排序下，“组合崩塌”的占比在26.4%到76.4%之间波动。其次，标注一致性在该类别层面崩溃，但在任务层面并未崩溃。来自全部十一种语言的母语者对179份记录进行了评判……

    arXiv:2611.13140v1 Announce Type: new  Abstract: Failure taxonomies for LLM reasoning are increasingly reported as measurements: papers state what fraction of a model's errors fall into named categories, then compare those fractions across models. We show that for one such category the reported fraction is largely a property of the measurement procedure. We defined Compositional Collapse (a model notices a required sub-part of a problem, then drops or misapplies it) and measured it across twelve models and eleven languages on a public multilingual math benchmark (2,316 graded records). First, 50.1% of failures carry evidence for more than one category, so the reported plurality depends on the order in which the classifier tests them: reversing two tests moves it in eleven of twelve models, and across all six orderings Collapse's share spans 26.4% to 76.4%. Second, agreement collapses for the category but not for the task. Native speakers of all eleven languages judged 179 transcripts, 
    
[^184]: 哪些成功案例能成为好的参考？面向多轮在线策略蒸馏的动作序列对齐方法

    Which Successes Make Good References? Action-Sequence Alignment for Multi-Turn On-Policy Distillation

    [https://arxiv.org/abs/2611.13135](https://arxiv.org/abs/2611.13135)

    该论文将多轮在线策略蒸馏中的成功参考选择建模为自由目标最优传输问题，并提出以最长公共子序列（LCS）为基础的动作序列对齐作为轨迹兼容性的可观测代理，为每条失败轨迹匹配最兼容的成功参考，从而避免误导性监督、提升蒸馏效果。

    

    在线策略蒸馏（OPD）通过在学生模型自身的轨迹上进行监督，来减少训练与推理之间的分布失配。对于多轮工具使用智能体而言，让教师模型以经过验证的同任务成功轨迹作为参考，可以提供有用的后见之明；但不同的成功案例可能到达不兼容的交互状态，从而提供误导性的监督信号。我们将成功参考的选择问题形式化为一个自由目标的最优传输问题，即把每条失败轨迹分配给与其最兼容的成功参考。由于潜在的状态路径是隐藏的，我们采用动作序列对齐作为兼容性的可观测代理，并用最长公共子序列（LCS）加以实现——LCS能够识别出共享的有序动作主干，同时允许可恢复的绕路。这一视角使轨迹兼容性成为多轮OPD中参考选择的核心准则。在ALFWorld上的实验……（原文摘要在此处截断）

    arXiv:2611.13135v1 Announce Type: cross  Abstract: On-policy distillation (OPD) reduces train-inference distribution mismatch by supervising a student on its own trajectories. For multi-turn tool-use agents, conditioning the teacher on a validated same-task successful trajectory as a reference can provide useful hindsight, but different successes may reach incompatible interaction states and offer misleading supervision. We cast successful-reference selection as a free-target optimal transport problem, which reduces to assigning each failed trajectory its most compatible successful reference. Because latent state paths are hidden, we use action-sequence alignment as an observable proxy for compatibility and instantiate it with longest common subsequence (LCS), which identifies a shared ordered backbone of actions while allowing recoverable detours. This perspective makes trajectory compatibility the central criterion for reference selection in multi-turn OPD. Experiments on ALFWorld an
    
[^185]: 面向语言模型系统中认知记忆重构的激活引导图检索

    Activation-Guided Graph Retrieval for Cognitive Memory Reconstruction in Language Model Systems

    [https://arxiv.org/abs/2611.13127](https://arxiv.org/abs/2611.13127)

    该论文提出了一种激活条件化的记忆图检索架构，通过请求与目标线索激活授权记忆图、扇出衰减控制有界遍历、实体递归发射线索，并在生成前重构类型化证据图，从而为长寿命AI系统实现认知记忆重构。

    

    检索增强语言模型系统通常通过基于查询条件的相关性来选择上下文。我们认为，长寿命AI系统需要第二种算子：激活条件化的记忆可用性。我们规定了一种架构，其中请求与目标线索为授权的记忆图进行播种，激活与扇出衰减分配有界的遍历前沿，检索到的实体递归地发射线索，并在生成之前重构类型化证据图。该规范产生了关于候选选择、遍历控制、图正则化、证据重构和强化的五项架构性推论。我们推导了实现层面的成本估计，并定义了消融实验、失败条件以及可复现的评估协议。如果在同等预算下，一个更简单的图重排序器在因果与时间恢复任务上能够匹配完整遍历的效果，则该核心主张将被收窄或证伪。

    arXiv:2611.13127v1 Announce Type: new  Abstract: Retrieval-augmented language model systems generally select context by query-conditioned relevance. We argue that long-lived AI systems require a second operator: activation-conditioned memory availability. We specify an architecture in which request and goal cues seed an authorized memory graph, activation and fan attenuation allocate a bounded traversal frontier, retrieved entities recursively emit cues, and a typed evidence graph is reconstructed before generation. The specification yields five architectural consequences about candidate selection, traversal control, graph regularization, evidence reconstruction, and reinforcement. We derive implementation-level cost estimates and define ablations, failure conditions, and a reproducible evaluation protocol. The central claim is narrowed or falsified if a simpler graph reranker matches full traversal on causal and temporal recovery under equal budgets. This architecture-and-analysis pre
    
[^186]: 即时图像Transformer与后期像素精炼

    Just Image Transformers with Late-Stage Pixel Refinement

    [https://arxiv.org/abs/2611.13120](https://arxiv.org/abs/2611.13120)

    该论文提出在JiT（无VAE潜空间的图像Transformer）后训练阶段添加简单的卷积像素通路层，且该通路只需在去噪最后约20%阶段激活，即可显著提升图像生成质量。

    

    近期关于无潜空间扩散Transformer的研究表明，无需潜空间VAE来训练图像生成器是可行的。这避免了VAE带来的压缩伪影，并简化了训练流程。特别是JiT（Just Image Transformers）证明，即使是一个标准的Transformer，只要训练其预测最终图像（x-prediction），也能生成图像。尽管JiT实现了与潜空间模型相当的训练效率，但可能由于将图像建模为大型图块的集合，其图像质量仍有所下降。在这项工作中，我们重新审视了JiT，并证明在后训练阶段添加简单的卷积像素通路层可以显著提升生成质量。此外，我们表明所添加的像素通路无需在早期去噪阶段激活，甚至可以推迟到去噪过程的最后20%阶段才激活。我们在ImageNet@256上的实验……（摘要在此处截断）

    arXiv:2611.13120v1 Announce Type: cross  Abstract: Recent work on latent-free diffusion transformers has demonstrated the feasibility of training image generators without a latent VAE. This avoids compression artifacts from the VAE and simplifies the training pipeline. In particular, JiT showed that even a standard transformer can be trained to generate images when the model is trained to predict the final image ($x$-prediction). Even though JiT unlocks training efficiency comparable to latent models, it still suffers from reduced image quality likely arising from modeling the image as a collection of large patches. In this work, we revisit JiT and show that adding simple convolutional pixel-pathway layers in a post-training phase significantly improves generation quality. Moreover, we show that the added pixel-pathway does not need to be active in the early denoising stages, and can be activated as late as in the last $20\%$ of the denoising process. Our experiments on ImageNet@256 sh
    
[^187]: ASPIRE：一个用于迭代推理与探索的智能体化系统辨识平台

    ASPIRE: An Agentic System-Identification Platform for Iterative Reasoning and Exploration

    [https://arxiv.org/abs/2611.13114](https://arxiv.org/abs/2611.13114)

    ASPIRE是一个由大语言模型智能体驱动的非线性系统辨识框架，能够自主选择并执行辨识方法，从测量数据构建闭式非线性模型，融入先验物理知识并独立验证，且支持自主与交互两种运行模式。

    

    系统辨识需要在辨识方法的选择、先验物理知识的融入以及模型验证等方面做出专家决策。我们提出了ASPIRE，这是一个用于非线性系统辨识的框架，其中大型语言模型智能体负责选择并执行辨识方法。ASPIRE能够从测量数据中构建闭式（解析形式）的非线性静态或动态模型，并可选择性地辅以系统描述、模型的预期用途，以及以形式化方式或自然语言表达的物理要求。每个模型包含其参数、估计记录、对模型各项的解释，以及在智能体无法访问的测试数据上评估的误差。形式化的物理要求由独立机制进行验证。ASPIRE既可以自主运行，也可以交互式运行，允许智能体提出问题，用户也可以请求进一步的分析和额外的辨识轮次。我们在三个实测系统上对ASPIRE进行了评估：一个移动机器人……（摘要截断）

    arXiv:2611.13114v1 Announce Type: cross  Abstract: System identification requires expert decisions on the identification method, incorporation of prior physical knowledge, and model validation. We propose ASPIRE, a framework for nonlinear system identification in which a large language model agent selects and executes the identification method. ASPIRE constructs closed-form, nonlinear static or dynamic models from measured data, optionally supplemented by a system description, the intended use of the model, and physical requirements expressed formally or in natural language. Each model includes its parameters, an estimation record, an interpretation of its terms, and an error evaluated on test data inaccessible to the agent. Formal physical requirements are verified independently. ASPIRE operates autonomously or interactively, allowing the agent to ask questions and the user to request further analyses and identification rounds. We evaluate ASPIRE on three measured systems: a mobile ro
    
[^188]: 从内在动机构建世界模型

    Building World Models from Intrinsic Motivation

    [https://arxiv.org/abs/2611.13113](https://arxiv.org/abs/2611.13113)

    该论文将24种内在动机目标统一整合到一个基于模型的强化学习框架中，系统评估了它们所构建的世界模型在零样本、少样本和多任务控制下的表现。

    

    在缺乏外在奖励或任务监督的情况下，人类行为由内在动机引导——即驱动好奇心、创造力、游戏以及其他智能核心要素的神经信号。内在动机在构建能够实现泛化与适应的稳健世界模型方面发挥着重要的发育作用。然而，现有研究尚未解答：多种多样的内在动机如何与世界模型表示学习相互作用以支持下游控制。这限制了我们对哪些由内在动机收集的经验能够构建可复用模型的理解。为填补这些空白，我们将24种内在目标（包括涵盖好奇心、熵与胜任力等方向的成熟方法及显式变体）整合到一个统一的基于模型的强化学习框架中。我们通过在运动控制、物体操作和开放式游戏等任务中的零样本、少样本和多任务控制，来评估它们所构建的世界模型……

    arXiv:2611.13113v1 Announce Type: new  Abstract: In the absence of extrinsic rewards or task supervision, human behavior is guided by intrinsic motivation---neural signals that drive curiosity, creativity, play, and other essential facets of intelligence. Intrinsic motivation plays an important developmental role in building robust world models that enable generalization and adaptation. Yet existing studies leave open how a broad range of intrinsic motivations interacts with world-model representation learning to support downstream control. This limits our understanding of which intrinsically collected experiences build reusable models. To address these gaps, we integrate 24 intrinsic objectives, including established methods and explicit variants spanning curiosity, entropy, and competency, into a common model-based reinforcement learning framework. We evaluate the world models they build through zero-shot, few-shot, and multi-task control in locomotion, manipulation, and open-ended g
    
[^189]: 一个温度统御一切？

    One Temperature to Rule Them All?

    [https://arxiv.org/abs/2611.13077](https://arxiv.org/abs/2611.13077)

    该论文研究了 pass@K 推理中温度多样性何时优于单一温度，发现虽然更难的问题更难共享高效的采样温度，但这一趋势仅随难度呈对数增长，表明单一调优温度在多数情况下已足够有效。

    

    推理时采样是提升大语言模型性能的核心机制，其中温度参数控制着可靠性与多样性之间的权衡。尽管近期研究表明，对不同样本使用不同温度可以改进多样本推理效果，但简单的固定温度方案在实践中仍是标准做法。我们在经典的 pass@K 设定下研究这一差异，探讨温度多样性何时有用，以及单一调优温度何时已经足够。为此，我们分析了多个模型和数据集上每个问题的成功概率，追踪其随采样温度的变化规律。我们识别出这些温度曲线的若干一致特性，其中包括：越难的问题越不太可能与其他问题共享高效的采样温度，尽管这一趋势仅随难度呈对数增长。随后，我们推导出一种高效方法，用于评估理论上最优的多温度分配方案。

    arXiv:2611.13077v1 Announce Type: cross  Abstract: Inference-time sampling is a central mechanism for improving large language models, with temperature controlling the tradeoff between reliability and diversity. While recent work suggests that varying temperature across samples can improve multi-sample inference, simple fixed-temperature schemes remain the standard in practice. We investigate this discrepancy in the canonical pass@K setting, asking when temperature diversity is useful, and when a single tuned temperature suffices. To this end, we analyze per-question success probabilities across multiple models and datasets, tracking how they vary with sampling temperature. We identify several consistent properties of these profiles, including that harder questions are less likely to share efficient sampling temperatures, although this trend is only logarithmic in hardness. We then derive an efficient method for evaluating the theoretically optimal multi-temperature allocation scheme, 
    
[^190]: 多准则排序模型参数的引出方法

    Elicitation of the parameters of a multi-criteria sorting model

    [https://arxiv.org/abs/2611.13062](https://arxiv.org/abs/2611.13062)

    本文针对经过公理化刻画的非补偿性分类模型（NCS），提出了严格的直接参数引出程序，并设计了多种基于经典图探索方法的提问算法，用于确定充分准则联盟集合以及区分两个类别的轮廓参数。

    

    多准则决策辅助（MCDA）为选择、排序或分类等不同类型的决策问题提供了模型和方法。这些模型涉及各种需要被引出的参数。在引出过程中，所有必要的信息都必须从决策者处获取。为此，文献中提出了多种直接和间接的引出方法。在这项工作中，我们专注于直接引出：我们的目标是为非补偿性分类模型（NCS）的参数提供一个严格的引出程序，该模型已通过公理化分析得到刻画。我们考虑的是NCS模型的基本版本，即具有两个类别且没有否决权的模型。其参数为充分准则联盟的集合以及区分两个类别的轮廓。关于充分联盟，我们提出了几种提问算法，这些算法将问题嫁接在经典的图探索方法之上。

    arXiv:2611.13062v1 Announce Type: new  Abstract: Multiple-criteria decision aiding (MCDA) proposes models and methods for different types of decision problems such as choice, ranking or sorting. Such models involve various parameters that must be elicited. In the elicitation process, all necessary information has to be taken from the decision-maker. For this, several direct and indirect elicitation methods are suggested in the literature. In this work, we concentrate on direct elicitation: we aim at providing a rigorous elicitation procedure for the parameters of the Non-Compensatory Sorting model (NCS), which has been characterized by an axiomatic analysis. A basic version of the NCS model is considered, i.e., a model with two categories and no vetoes. The parameters are the set of sufficient coalitions of criteria and the profile separating the two categories. Regarding the sufficient coalitions, we propose several questioning algorithms that engraft questions on classical graph expl
    
[^191]: 一种面向204种语言的多词分词器：即使在英语上所需词元也比GPT-4o更少

    A Multi-Word Tokenizer for 204 Languages That Needs Fewer Tokens Than GPT-4o's, Including on English

    [https://arxiv.org/abs/2611.13059](https://arxiv.org/abs/2611.13059)

    该论文提出Havon多词分词器系列，通过在约200种语言上从头训练并引入多词训练阶段，在包括英语在内的161种语言上所需词元数量少于GPT-4o的分词器，在印度语言上的词元数中位数仅为GPT-4o的0.74倍。

    

    印度诸语言在GPT-4o的分词器上处理相同内容时，需要1.6到14倍于英语的词元数量，这使其服务成本更高、速度更慢。我们在约200种语言的人类书写文本上从头训练分词器。Havon v3是一个字节级BPE分词器，拥有262,144个词元，并配备一个预分词器，该预分词器能够将组合符号与其字母保持在一起，并对无空格书写的文字系统进行分块。在FLORES-200开发测试集上，它在全部25种印度语言上所需的词元数量少于GPT-4o的o200k_base（中位数为GPT-4o的0.74倍），在204种语言中的172种上所需词元更少，但在英语上多7%。Havon v4在v3的基础上增加了第二个多词训练阶段。在恰好与GPT-4o相同的200,019词元词表规模下，v4在英语上比GPT-4o少用6.1%的词元，在204种语言中的161种以及全部25种印度语言上所需词元更少（中位数为0.74）；在相同规模下，v3在英语上多9.4%的词元，仅在133种语言上占优。在262,144词元规模下，v4需要少9.6%的词元

    arXiv:2611.13059v1 Announce Type: cross  Abstract: Languages of India need 1.6 to 14 times as many tokens as English on GPT-4o's tokenizer for the same content, which makes them more expensive and slower to serve. We train tokenizers from scratch on human-written text in about 200 languages. Havon v3 is a byte-level BPE tokenizer with 262,144 tokens and a pre-tokenizer that keeps combining marks with their letters and chunks scripts written without spaces. On FLORES-200 devtest it needs fewer tokens than GPT-4o's o200k_base on all 25 languages of India (median 0.74 of GPT-4o's count) and on 172 of 204 languages, but 7% more on English. Havon v4 adds a second, multi-word training stage (SuperBPE) to v3. At exactly GPT-4o's vocabulary size of 200,019 tokens, v4 needs 6.1% fewer tokens than GPT-4o on English and fewer on 161 of 204 languages and all 25 languages of India (median 0.74); at the same size, v3 needs 9.4% more on English and wins on 133. With 262,144 tokens, v4 needs 9.6% fewe
    
[^192]: 2026年的印度语言“代币税”：测量分词器差异及其对印度各语言的成本

    The Indic Token Tax in 2026: Measuring Tokenizer. Disparity and Its Cost for the Languages of India

    [https://arxiv.org/abs/2611.13058](https://arxiv.org/abs/2611.13058)

    该研究测量了九个主流分词器对印度25种语言的“代币税”，发现印度语言所需token量最高可达英语的13.7倍（GPT-4o分词器），导致印度语言用户在商业LLM使用中面临更高的费用、更慢的速度和更小的有效上下文，而Sarvam分词器是唯一将差异控制在2.6倍以内的第三方选择。

    

    商业语言模型API按token计费，因此当分词器处理某种语言的相同内容需要更多token时，就会使该语言的使用更昂贵、速度更慢，并且在固定上下文窗口中能容纳的文本更少。我们在FLORES-200中全部25种印度语言（2,009句与英文对齐的专业翻译句子）以及来自PHINC的13,732对人工翻译的印地英语-英语社交媒体文本上，测量了九个公开分词器的这种差异，包括GPT-4、GPT-4o、Qwen2.5、DeepSeek-V3、Mistral-Nemo以及四个Sarvam模型的分词器。在GPT-4o的分词器上，相同内容所需的token是英语的1.57倍（印地语）到5.01倍（奥里亚语），而使用Ol Chiki文字的桑塔利语高达13.7倍；GPT-4的分词器最高需要12.8倍。Sarvam-30B/105B是唯一在所有语言上都保持在2.6倍以下（中位数1.53）的第三方分词器。在一个使用截至11月5日获取的官方标价的示例场景中……

    arXiv:2611.13058v1 Announce Type: cross  Abstract: Commercial language model APIs bill per token, so a tokenizer that needs more tokens for the same content in one language makes that language more expensive, slower, and able to fit less text in a fixed context window. We measure this disparity for nine public tokenizers, including those of GPT-4, GPT-4o, Qwen2.5, DeepSeek-V3, Mistral-Nemo and four Sarvam models, on all 25 languages of India available in FLORES-200 (2,009 professionally translated sentences aligned with English) and on 13,732 human-translated Hinglish-English social media pairs from PHINC. On GPT-4o's tokenizer the same content needs between 1.57 times (Hindi) and 5.01 times (Odia) the tokens of English, and 13.7 times for Santali in Ol Chiki script; GPT-4's tokenizer needs up to 12.8 times. Sarvam-30B/105B is the only third-party tokenizer that stays below 2.6 times on every language (median 1.53). In an illustrative scenario using official list prices accessed on 5 N
    
[^193]: 用于异步WiFi-IMU室内定位的连续时间集合变换器

    Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization

    [https://arxiv.org/abs/2611.13057](https://arxiv.org/abs/2611.13057)

    提出一种连续时间集合变换器，将异步采样的WiFi与IMU观测表示为带时间戳的无序标记集合，用单一注意力块实现跨模态与时间的联合融合，并通过随机掩码训练应对缺失或过时的传感器输入，用于室内定位。

    

    室内定位通常将WiFi接收信号强度指示器（RSSI）指纹与惯性测量相结合。然而，这些传感器流以不同的速率异步采样，因此观测值很少在时间上对齐，且可能缺失或过时。大多数现有方法将数据重新采样到共享时钟上，并使用单独的分支处理每个模态。我们转而提出一种连续时间变换器，将观测值表示为一组无序的带时间戳的标记。每个标记编码其测量值、模态以及自采集以来经过的时间。单个注意力块跨模态和时间联合融合信息，既避免了重新采样，也避免了特定模态的分支，同时保持对标记顺序的不变性。在训练期间，我们随机屏蔽整个模态和单个观测值，使模型能够在缺失或过时输入的情况下运行。我们验证了模态特定的…

    arXiv:2611.13057v1 Announce Type: cross  Abstract: Indoor localization commonly combines WiFi received signal strength indicator (RSSI) fingerprints with inertial measurements. However, these sensor streams are sampled asynchronously at different rates; therefore, observations rarely align in time and may be missing or stale. Most existing methods resample the data onto a shared clock and process each modality using a separate branch. We instead present a continuous-time transformer that represents observations as an unordered set of timestamped tokens. Each token encodes its measurement, modality, and elapsed time since acquisition. A single attention block jointly fuses information across modalities and time, avoiding both resampling and modality-specific branches while remaining invariant to token order. During training, we randomly mask entire modalities and individual observations, enabling the model to operate under missing or outdated inputs. We validate the modality-specific en
    
[^194]: 超越文档：面向任意工作成果的编译上下文与受治理智能体

    Beyond the Document: Compiled Context and Governed Agents for Any Work Product

    [https://arxiv.org/abs/2611.13055](https://arxiv.org/abs/2611.13055)

    该论文提出便携式智能体格式（PAF）包这一新交付单元，将工作成果连同编译上下文、证据、来源与交互规则一起封装，实验证明受治理的智能体阅读器在回答相关问题上的成功率（96%）显著优于传统的指令加原始文件检索方式（86%）。

    

    文档保存了专业工作的可见成果，却往往丢弃了其背后的研究、计算、假设、证据和授权边界。我们研究了一种不同的交付单元：便携式智能体格式（Portable Agent Format, PAF）包，它将工作成果与编译后的上下文、证据、来源信息以及后续交互规则一同携带。为了检验这种上下文在产生它的过程之外是否依然有用，我们评估了四个新创建的包（均创建于托管阅读器和离线阅读器被固定之后）：一份投资提案、基金研究、财务报表分析和一份工程事故复盘。其中两个包由专用智能体生成，另外两个通过通用助手的打包技能生成。在400个冻结问题和12,000条评估回复中，托管的受治理阅读器在96%的问题上回答成功，而采用指令加原始文件检索的方式成功率为86%（经Holm校正的 p < 10^-7）。

    arXiv:2611.13055v1 Announce Type: cross  Abstract: A document preserves the visible result of professional work but often discards the research, calculations, assumptions, evidence, and authorization boundaries behind it. We study a different unit of delivery: a Portable Agent Format (PAF) package that carries a work product together with compiled context, evidence, provenance, and rules for subsequent interaction. To test whether this context remains useful beyond the process that produced it, we evaluate four new packages, created after the hosted and offline readers were fixed: an investment proposal, fund research, financial-statement analysis, and an engineering incident review. Two packages were produced by specialized agents and two through a general-purpose assistant's packaging skill. Across 400 frozen questions and 12,000 evaluated responses, a hosted governed reader succeeds on 96% of questions versus 86% with instructions plus raw-file retrieval (Holm-adjusted p < 10^-7). O
    
[^195]: 信任，但须验证、审计与归因：一个用于检验AI产出研究成果的人在回路框架

    Trust, but verify, audit and attribute: a human-in-the-loop framework for checking AI-produced research results

    [https://arxiv.org/abs/2611.13042](https://arxiv.org/abs/2611.13042)

    该论文提出了一个领域无关的人在回路框架，通过经过验证的AI工具检验、忠实性审计以及人类与AI贡献归因记录，来保障对AI批量产出的研究成果的有效核查。

    

    人工智能（AI）系统如今能够批量产出研究成果：2026年10月，单个实验室就发布了超过700篇AI生成的数学论文手稿，其中只有一部分附有经过机器验证的证明。如此大量的成果将主要由调用专门软件的AI代理来检验。然而，一项通过的检查仅能证明被检验的那个产物本身。对于被翻译成证明助手Lean的证明，已有研究表明该产物所陈述的内容可能与作者原文不符，而且判断一次翻译是否忠实在一般情况下是不可判定的。因此，人类研究者必须留在回路之中，并且人类与AI各自的贡献必须被记录。我们提出了一个领域无关的、人在回路的框架，用于检验AI产出的结果：由经过验证的、可供代理使用的工具基于不同的信任基础来检验结果；由忠实性审计来追问每次检验是否真正针对了原文所述的论断；由人类最终做出裁决。

    arXiv:2611.13042v1 Announce Type: new  Abstract: Artificial intelligence (AI) systems now produce research results in bulk: in October 2026 a single laboratory released more than 700 AI-generated mathematics manuscripts, only part of them with machine-checked proofs. Such volumes will be checked mostly by AI agents that call specialised software. A passing check certifies only the artefact that was checked. For proofs translated into the proof assistant Lean, it has been shown that this artefact may state something other than the author's text, and that deciding whether a translation is faithful is undecidable in general. The human researcher therefore stays in the loop, and human and AI contributions must be recorded. We propose a domain-independent, human-in-the-loop framework for checking AI-produced results. Validated, agent-usable tools with different grounds of trust check the results; a fidelity audit asks whether each check addressed the claim as stated; the human decides at fi
    
[^196]: 基于代码依据的智能体式静态分析漏洞发现解释

    Code-Grounded Agentic Explanations for Static Analysis Vulnerability Findings

    [https://arxiv.org/abs/2611.13041](https://arxiv.org/abs/2611.13041)

    该论文提出将漏洞检测职责保留给静态分析器、大语言模型智能体仅负责生成解释，并据此重新设计了开源IDE插件SAFE，要求智能体在回答前必须查阅代码仓库和触发的规则，从而在保留可审计性的同时生成有代码依据的解释。

    

    静态应用安全测试（SAST）工具能够尽早发现漏洞，其判定基于规则且可审计。然而，开发者常常会忽略那些他们无法理解的结果。大语言模型可以提供缺失的解释，但如果设计上让模型同时负责检测，就会失去使静态分析站得住脚的可审计性。我们将检测工作保留给分析器，只让模型负责解释。一项有20名参与者参与的初步用户研究和一场有19名从业者参加的工作坊得出了五项需求。在这些需求的指导下，我们重新设计了SAFE，这是一个开源的IntelliJ IDEA插件，能够解释任何以静态分析结果交换格式（SARIF）报告结果的分析器所产生的发现。该插件按需求分层提供解释，将其与报告的数据流关联起来，并用一个智能体替换单一提示词后端，该智能体在回答之前必须查阅代码仓库和被触发的规则。一项被试内交叉实验（研究，摘要原文在此处截断）

    arXiv:2611.13041v1 Announce Type: cross  Abstract: Static application security testing (SAST) tools surface vulnerabilities early and their verdicts are rule-based and auditable. Yet developers routinely dismiss findings they cannot interpret. Large language models can supply the missing explanation, but designs that also hand the model the detection give up the auditability that makes static analysis defensible. We keep detection with the analyser and give the model only the explaining. A preliminary user study with 20 participants and a workshop with 19 practitioners yielded five requirements. Guided by these we redesigned SAFE, an open-source IntelliJ IDEA plugin that explains findings from any analyser reporting in the Static Analysis Results Interchange Format. It tiers explanations by demand, ties them to the reported data flow, and replaces the single-prompt backend with an agent that must consult the repository and the rule that fired before answering. A within-subjects crossov
    
[^197]: APS-VLA：面向高效视觉-语言-动作模型的动作引导与未来感知的令牌剪枝方法

    APS-VLA: Action-Guided and Future-Aware Token Pruning for Efficient Vision-Language-Action Models

    [https://arxiv.org/abs/2611.13030](https://arxiv.org/abs/2611.13030)

    APS-VLA 在冻结预训练 VLA 骨干网络的前提下，通过一个结合动作监督与未来感知视觉转移目标的轻量级决策影响估计器，从机器人演示中学习视觉令牌剪枝，在显著降低推理延迟的同时保证可靠的动作生成。

    

    视觉-语言-动作（VLA）策略在机器人操作任务中已展现出强大的泛化能力，然而处理密集的视觉令牌往往带来较高的推理延迟。视觉令牌剪枝为降低这一开销提供了一条实用途径，但如何挑选出一组紧凑的令牌、同时保证可靠的动作生成，仍然具有挑战性。现有方法大多依赖静态或即时性的启发式线索来估计令牌的重要性。我们提出了 APS-VLA，这是一个在保持预训练 VLA 骨干网络冻结的前提下，从机器人演示数据中学习视觉令牌选择的框架。一个轻量级的决策影响估计器基于当前视觉观测、语言指令和机器人本体感知对令牌进行打分。其训练结合了两种互补的信号：动作监督使剪枝后的策略预测同时匹配全输入策略的预测结果与演示动作，同时一个辅助性的视觉转移目标利用演示动作……（原文摘要在此处截断）

    arXiv:2611.13030v1 Announce Type: new  Abstract: Vision-language-action (VLA) policies have demonstrated strong generalization in robotic manipulation, yet processing dense visual tokens often incurs high inference latency. Visual token pruning offers a practical way to reduce this cost, but identifying a compact set of tokens that preserves reliable action generation remains challenging. Existing methods largely rely on static or immediate heuristic cues to estimate token importance. We introduce APS-VLA, a framework that learns visual token selection from robot demonstrations while keeping the pretrained VLA backbone frozen. A lightweight Decision Influence Estimator scores tokens conditioned on current visual observations, language instructions, and robot proprioception. Its training combines two complementary signals: action supervision matches both the full-input policy predictions and demonstrated actions, while an auxiliary visual transition objective uses demonstrated actions t
    
[^198]: GTAGym：用于评估与强化学习的真实AAA游戏任务环境

    GTAGym: Real AAA Game Tasks Environment for Evaluation and Reinforcement Learning

    [https://arxiv.org/abs/2611.13028](https://arxiv.org/abs/2611.13028)

    本文提出GTAGym——一个基于《侠盗猎车手V》真实AAA游戏任务的评估与强化学习环境，其基准GTAVBench包含103个跨九类任务家族的在线主线任务及47个图像式射击狙击任务，支持截图与键鼠操作的并行测试并自动校验任务结果，用于评测前沿大语言模型的智能体能力。

    

    AAA游戏为评估智能体的感知、推理和控制能力提供了丰富的测试平台。然而，评估智能体能否在AAA游戏中完成任务十分困难，且通常需要人工标注，因为任务结果往往难以自动验证，更难以将其应用于强化学习（RL）。我们提出了GTAGym，一个面向GTA V任务的评估与强化学习环境及框架。其评估基准GTAVBench源自GTA V主线剧情任务，包含跨九个任务类别重建的103个在线任务，以及基于GTA V图像的47个图像式射击与狙击任务。一个网络服务器提供对在线环境和评估界面的访问，通过截图和键盘-鼠标操作支持并行测试，并具备自动的引擎状态结果校验。我们在GTAVBench上使用文本任务指令评估了六个前沿大语言模型，并在其GTAVBench-AGI分支上进行评估，其中指令……

    arXiv:2611.13028v1 Announce Type: cross  Abstract: AAA games offer a rich testbed for evaluating agent perception, reasoning, and control. However, evaluating whether agents can complete tasks in AAA games is challenging and often human-annotated because task outcomes are often difficult to verify automatically, let alone applying it for Reinforcement Learning (RL). We introduce GTAGym, an environment and framework for evaluation and RL on GTA V tasks. Its evaluation benchmark, GTAVBench, derived from GTA V main-story missions, contains 103 online tasks reconstructed across nine task families, alongside 47 image-based shooting and sniping tasks using GTA V imagery. A web server provides access to the online environment and evaluation interface, supporting parallel testing through screenshots and keyboard-mouse actions with automatic engine-state outcome checks. We evaluate six frontier LLMs on GTAVBench with textual task instructions and on its GTAVBench-AGI branch, where instructions 
    
[^199]: 面向智能体软件开发的仓库上下文引导图像理解

    Repository-Context-Guided Image Understanding for Agentic Software Development

    [https://arxiv.org/abs/2611.13026](https://arxiv.org/abs/2611.13026)

    提出通过从同一代码库渲染清晰的参考截图来增强编程智能体，从而在查询截图质量退化、视觉证据不清晰时，仍能准确识别UI元素并定位相关源文件。

    

    编程智能体越来越多地在错误报告中接收UI截图，然而图像质量的下降——由低分辨率、压缩伪影或过小的渲染文本引起——会阻碍它们识别相关的UI元素和源文件。随着图像质量恶化，视觉语言模型的可靠性降低，而OCR辅助方法则可能将识别错误从退化的输入中传播开来。当定位所需的视觉证据在查询截图中已不再清晰可见时，这两种方法都会受到限制。我们提出用仓库渲染的参考图像来增强编程智能体：即从智能体受命修改的同一代码库生成的清晰截图。这些参考图像在保留当前可能包含错误的实现状态的同时，提供了应用程序UI的可读视觉证据。由于它们是从现有仓库生成的，而无需依赖……

    arXiv:2611.13026v1 Announce Type: cross  Abstract: Coding agents increasingly receive UI screenshots as part of bug reports, yet degraded image quality - caused by low resolution, compression artifacts, or small rendered text - can prevent them from identifying the relevant UI elements and source files. Vision-language models become less reliable as image quality deteriorates, while OCR-assisted approaches may propagate recognition errors from degraded inputs. Both approaches are limited when the visual evidence required for localization is no longer clearly observable in the query screenshot. We propose augmenting coding agents with repository-rendered reference images: crisp screenshots generated from the same codebase that the agent is tasked with modifying. These references provide readable visual evidence of the application's UI while preserving the current, potentially bug-containing implementation state. Because they are generated from the available repository rather than requir
    
[^200]: GRAP：用于分析大语言模型性别偏见的性别量表感知探测方法

    GRAP: Gender Rubric-Aware Probing for Analyzing Gender Bias in LLMs

    [https://arxiv.org/abs/2611.13024](https://arxiv.org/abs/2611.13024)

    提出了GRAP框架，利用轻量级逻辑回归探针从10个量表维度中自动识别最具区分性的性别偏见维度，以任务无关的方式分析大语言模型开放式回复中的性别偏见。

    

    随着大语言模型日益成为日常交互的中介，开放式回复中细微的性别偏见带来了日益增长的社会风险。现有的探测方法存在两个关键局限：一是依赖于固定格式的任务（如代词消解或职业联想），而对齐微调后的模型可以轻松应对这些任务；二是仅比较原始的群体层面量表分数，忽略了每个维度的方差，从而掩盖了哪些偏见维度能够可靠地区分性别群体。我们提出了GRAP（性别量表感知探测）框架，该框架使用轻量级的逻辑回归探针，从跨越10个量表维度评分的回复中识别最具区分性的偏见维度，并以任务无关的方式应用于开放式指令遵循。

    arXiv:2611.13024v1 Announce Type: new  Abstract: As LLMs increasingly mediate everyday interactions, subtle gender biases in open-ended responses pose growing societal risks. Existing probing methods have two key limitations: they rely on fixed-format tasks, such as pronoun resolution or occupation association, that alignment-tuned models can easily handle; and they compare raw group-level rubric scores, ignoring per-dimension variance and obscuring which bias dimensions reliably distinguish gender groups. We introduce GRAP (Gender Rubric-Aware Probing), a framework that uses a lightweight logistic probe to identify the most discriminative bias dimensions from responses scored across 10 rubric dimensions, operating task-agnostically on open-ended instruction following.
    
[^201]: 理解优先的AI工程：面向人类理解的目的、参考架构与架构模式

    Understanding-First AI Engineering: Purposes, Reference Architecture, and Architectural Patterns for Human Understanding

    [https://arxiv.org/abs/2611.13022](https://arxiv.org/abs/2611.13022)

    该论文提出“理解优先的AI工程”新范式，将特定目的的人类理解作为一等设计原则，归纳了反思性主体、有效行动、集体治理和认知价值四种理解目的，并给出贯穿生产、验证、使用与学习的参考架构及五种候选架构模式。

    

    AI系统可以在不发展人类理解的情况下产生可靠的结果。然而，即使人类的参与对于纠正错误并非必要，人类理解对于塑造后续AI支持的工作、形成偏好、治理集体选择以及满足好奇心，仍然具有重要价值。我们提出“理解优先的AI工程”，将特定目的的人类理解作为指导需求、设计和运营的一等设计原则。理解的目的被组织为四种互补的理据：反思性主体性、有效行动、集体治理和认知价值。一个参考架构通过共享记录连接生产、验证、使用以及学习与探索等环节，并通过解释和参与来支持理解。五种候选架构模式指导关于并行学习、解释性模型、目标探索、受众适配等方面的选择。

    arXiv:2611.13022v1 Announce Type: new  Abstract: AI systems can produce reliable results without developing human understanding. However, human understanding remains valuable for shaping subsequent AI-supported work, forming preferences, governing collective choices and satisfying curiosity, even when human participation is unnecessary for correcting errors. We propose understanding-first AI engineering, which treats purpose-specific human understanding as a first-class design principle guiding requirements, design and operation. The purposes of understanding are organised into four complementary rationales: reflective agency, effective action, collective governance and epistemic value. A reference architecture connects production, verification, use, and learning and exploration through shared records, supporting understanding through explanation and participation. Five candidate architectural patterns guide choices about parallel learning, explanatory models, goal exploration, audienc
    
[^202]: ChipOmni：面向电子设计自动化的通用人工智能智能体

    ChipOmni: A General-Purpose AI Agent for Electronic Design Automation

    [https://arxiv.org/abs/2611.13020](https://arxiv.org/abs/2611.13020)

    ChipOmni是一个面向电子设计自动化的知识增强AI智能体框架，通过融合多模态理解、动态规划、多智能体协作与验证驱动的工具交互，实现了EDA设计中自主规划、验证与修复的统一自动化系统。

    

    电子设计自动化（EDA）工作流程仍然高度碎片化，需要工程师协调异构的表示形式、专业化的软件工具以及跨多个设计阶段的迭代验证。尽管大语言模型为EDA带来了更通用的推理和代码生成能力，但许多现有方法仍专注于孤立的任务，难以在可执行的工程环境中自主地规划、验证和修复设计。我们提出了ChipOmni，这是一个面向EDA的知识增强、上下文持续演化的AI智能体框架。ChipOmni将多模态理解、动态规划、多智能体协作以及验证驱动的工具交互集成到一个统一的自主系统中。通过持续吸收来自外部工程工具的反馈，该框架能够维护一个不断演化的任务上下文，融合领域知识，并迭代地优化中间设计。

    arXiv:2611.13020v1 Announce Type: new  Abstract: Electronic design automation (EDA) workflows remain highly fragmented, requiring engineers to coordinate heterogeneous representations, specialised software tools, and iterative verification across multiple design stages. Although large language models have brought more general reasoning and code-generation capabilities to EDA, many existing approaches remain focused on isolated tasks and struggle to plan, verify, and repair designs autonomously within executable engineering environments. Here we introduce ChipOmni, a knowledge-enhanced, context-evolving AI agent framework for EDA. ChipOmni integrates multimodal understanding, dynamic planning, multi-agent collaboration, and verification-driven tool interaction into a unified autonomous system. By continually incorporating feedback from external engineering tools, the framework maintains an evolving task context, integrates domain knowledge, and iteratively refines intermediate designs. 
    
[^203]: 多未来草拟：面向树验证式推测解码的混合块草拟器

    Multi-Future Drafting: Mixture Block Drafters for Tree-Verified Speculative Decoding

    [https://arxiv.org/abs/2611.13017](https://arxiv.org/abs/2611.13017)

    论文揭示了基于乘积排序的草拟树在多未来场景下会将大量概率预算浪费在目标模型从不产生的“嵌合体”前缀上（占64节点树的27-33%），并提出多未来草拟（MFD），用潜变量（K个可学习模式嵌入）替代链式法则来恢复多个未来之间的耦合，从而提升树验证式推测解码的接受效率。

    

    诸如DFlash之类的块草拟器可在一次前向传播中草拟一整块未来token，而诸如DDTree之类的草拟树则根据草拟器各位置分布的乘积对前缀进行排序并验证排名最高的前缀。在树验证机制下，期望接受的token数量等于树所覆盖的概率质量。我们证明乘积会错误地分配这一概率质量：当续写存在多个可能的未来路径时，乘积会将它们拼接成“嵌合体”（chimeras）——即由目标模型从不产生的、由不同未来拼接而成的前缀，并给予它们与真实未来同等的概率预算。除非出现平局，任何按乘积排序的树若要容纳两个在 $m$ 个位置上不一致的未来，就必然产生 $2^{m-1}-1$ 个嵌合体，无论草拟器训练得多好；在实践中，嵌合体占64节点树的27-33%。我们提出多未来草拟（MFD），它通过引入潜变量而非链式法则来恢复多个未来之间的耦合：在草拟器中设置 $K$ 个可学习的模式嵌入……

    arXiv:2611.13017v1 Announce Type: new  Abstract: Block drafters such as DFlash draft a block of future tokens in one forward pass, and draft trees such as DDTree verify the prefixes ranked highest by the product of the drafter's per-position distributions. Under tree verification, the expected number of accepted tokens is the probability mass the tree covers. We show that a product misallocates it: when the continuation can follow several futures, it splices them into chimeras, prefixes stitched together from different futures that the target never produces, and gives them as much budget as the futures themselves. Barring ties, no product-ranked tree can hold two futures that disagree in $m$ positions without $2^{m-1}-1$ chimeras, however well the drafter is trained, and in practice chimeras make up 27-33% of a 64-node tree. We propose Multi-Future Drafting (MFD), which restores the coupling with a latent variable instead of the chain rule: $K$ learned mode embeddings at the drafter's 
    
[^204]: 从问题到决策：基准测试LLM智能体如何从人类获取决策相关信息

    From Questions to Decisions: Benchmarking How LLM Agents Acquire Decision-Relevant Information from Human Beings

    [https://arxiv.org/abs/2611.13015](https://arxiv.org/abs/2611.13015)

    提出Ask2Decide基准，通过43个由真人操作员和优化求解器支持的规划任务，评估LLM智能体能否主动向人类询问对决策至关重要的缺失信息，并区分有用信息与无关细节。

    

    LLM智能体越来越多地被要求为人们制定计划，例如调度司机或分配护士。然而，人们的请求往往不完整，且用于决策的支持信息可能已过时，这要求智能体主动提问以获取澄清。现有的基准测试通常提供完全明确指定的问题，或依赖未经校准的LLM模拟用户，导致这种能力未得到充分检验。我们提出了Ask2Decide，这是一个包含43个规划任务的基准，用于评估智能体是否会询问对其决策至关重要的缺失信息。操作员掌握着智能体请求和记录中不存在的事实，并由优化求解器确定哪些事实会改变最优计划。我们还引入了措辞相似但不改变最优计划的事实，以测试智能体能否区分有用信息与无关细节。我们从19名参与者那里收集了79个交互会话，发现在不……（摘要此处截断）

    arXiv:2611.13015v1 Announce Type: new  Abstract: LLM agents are increasingly asked to make plans for people, such as scheduling drivers or assigning nurses. However, people's requests are often incomplete, and the supporting information for decision-making may be outdated, requiring agents to ask for clarification. Existing benchmarks typically provide fully specified problems or rely on uncalibrated LLM-simulated users, leaving this ability insufficiently tested. We introduce Ask2Decide, a benchmark of 43 planning tasks that evaluates whether agents ask for missing information that matters for their decisions. An operator holds facts absent from the agent's request and records, and optimization solvers determine which facts change the optimal plan. We also include similarly worded facts that leave the optimal plan unchanged to test whether agents distinguish useful information from irrelevant details. We collect 79 interaction sessions from 19 participants and find that, after an inco
    
[^205]: 编译门控的 Lean 证明重构与企业级智能体实践：在每题3美元上限下每步选用一个模型，随后进行程序化重写规则

    Compile-Gated Lean Proof Refactoring with Enterprise Agent Practices: A Model per Step under a US\$3 Cap, Followed by Programmatic Rewrite Rules

    [https://arxiv.org/abs/2611.13011](https://arxiv.org/abs/2611.13011)

    该论文提出一种“每步选用性价比最优模型、单题成本严格控制在3美元上限内”的企业级智能体实践，并将开发中积累的经验固化为经编译器校验的程序化重写规则，最终以86.31分（总花费166.33美元）在 Lean 证明重构竞赛中获得第8名。

    

    我们参加封闭赛道的作品（Arena 用户名 Vexorium-Agent）在60道决赛题目上的最终榜单得分为86.31分（L = 77.07，H = 81.86，Z = 100），相比之下参考证明为33.33分，截止时排名第8（最高分：92.57），按标价并计入每一次调用的总费用为166.33美元（每题平均2.772美元，最高2.874美元；3美元上限）。三项实践支撑了这一成绩。第一，每个付费阶段使用我们在自建测试框架中测试的9个模型中“单位美元效果最佳”的模型：早期会话使用 GPT-6.1 Sol，后期会话与收尾阶段使用 GPT-6.1 Terra（其每轮成本为 Sol 的70%），审查与修复使用 Gemini 4 Argon；风格重写也由 Sol 完成。第二，我们在开发与研究运行中习得的经验以经过编译器校验的程序化重写规则的形式交付，而非提示词文本（发布前固定的规则带来+0.93分；发布后构建或启用的 pass 在样本内分别带来+0.26和+0.30分）。第三，每个……（摘要在此处截断）

    arXiv:2611.13011v1 Announce Type: cross  Abstract: Our closed-track entry (Arena username Vexorium-Agent) scored 86.31 points on the final board (L = 77.07, H = 81.86, Z = 100) over the 60 final problems, against 33.33 for the reference proofs, 8th at the deadline (top: 92.57), for US\$166.33 at list prices with every call counted (per problem: mean US\$2.772, at most US\$2.874; US\$3 cap). Three practices carry the entry. First, each paid stage runs the model that measured best per dollar for it among the 9 we tested in our own harness: GPT-6.1 Sol the early sessions, GPT-6.1 Terra the later ones and the tail at 70% of Sol's cost per turn, and Gemini 4 Argon the review and repair; Sol also runs the style rewrite. Second, what our development and research runs learned ships as compiler-checked programmatic rewrite rules rather than prompt text (+0.93 points from rules fixed before the release; +0.26 and +0.30, in sample, from passes built or switched on after the release). Third, every
    
[^206]: 一个可在浏览器中运行的针对 Xilinx Spartan-3E 和 Spartan-6 的开源实现流程，并与 ISE 进行逐字节比对验证

    An Open Implementation Flow for the Xilinx Spartan-3E and Spartan-6 that Runs in the Browser, Checked Byte for Byte Against ISE

    [https://arxiv.org/abs/2611.13008](https://arxiv.org/abs/2611.13008)

    该论文提出了一个可在浏览器中运行的开源 FPGA 实现流程，为仅被停产的 ISE 工具链支持的 Xilinx Spartan-3E 和 Spartan-6 器件生成比特流，其输出与 ISE 生成的比特流逐字节完全一致。

    

    作者描述了一个开源实现流程，涵盖从 HDL 到可编程比特流的完整过程，适用于 Xilinx Spartan-3E（XC3S100E 至 XC3S1200E）和 Spartan-6（XC6SLX16、XC6SLX45）器件——这些器件目前仅有已停产的 ISE 14.7 支持。该流程无需修改即可在网页浏览器或 Node.js 中运行：Yosys 和 nextpnr 被编译为 WebAssembly，其中 nextpnr 的 Himbaechel 框架配备了基于 Silinx 自身器件描述生成的 Spartan-3E 和 Spartan-6 架构、比特数据库和时序模型，以及 Silinx 自己的比特流写入器。比特数据库是通过对 ISE 文档化的 XDL 和 bitgen 工具进行黑盒观测建立的，采用码字分批实验方法；路由器仅使用比特位已被实际测量的开关，写入器会拒绝任何未经测量的特性。ISE 被作为参照标准（oracle）：对于相同的布局布线设计，该流程生成的比特流与 bitgen 写出的比特流逐字节相同，且 ISE 的 DRC 检查通过。布局和布线最初由（原文在此处截断）

    arXiv:2611.13008v1 Announce Type: cross  Abstract: The author describes an open implementation flow, from HDL to a programmable bitstream, for the Xilinx Spartan-3E (XC3S100E to XC3S1200E) and Spartan-6 (XC6SLX16, XC6SLX45), devices that only the discontinued ISE 14.7 supports. The flow runs unchanged in a web browser or in Node.js: Yosys and nextpnr compiled to WebAssembly, with nextpnr's Himbaechel framework given Spartan-3E and Spartan-6 architectures generated from Silinx's own device descriptions, bit database and timing model, and Silinx's own bitstream writer. The bit database was built by black-box observation of ISE's documented XDL and bitgen tools, using codeword-batched experiments; the router is given only switches whose bits were measured, and the writer refuses a feature it has not measured. ISE is the oracle: for the same placed and routed design, the bitstream is byte-identical to the one bitgen writes and ISE's DRC is clean. Placement and routing were first done by an
    
[^207]: 公正的AI治理与禁越边界原则

    Just AI Governance and the Forbidden-Boundary Principle

    [https://arxiv.org/abs/2611.13007](https://arxiv.org/abs/2611.13007)

    本文提出了一个涵盖规范、数据、模型与治理四层的公正AI评估框架，将六项正义要求映射到证据、决策权与救济上，并提出两阶段评估规则以及“禁越边界原则”——即AI代理不得在预先授权的边界之外执行具有外部影响的行为。

    

    人工智能日益影响着重大的决策，然而仅凭预测准确性并不足以为委托给AI的决策权提供正当性。本概念性研究构建了一个综合性框架，从规范层、数据层、模型层和治理层四个维度评估AI的公正性。该研究借鉴了多元正义理论、社会技术公平性研究以及公开可用的治理标准，将六项正义要求映射到可观察的证据、决策权和救济措施之上。研究提出了一个两阶段评估规则：首先排除那些未能满足独立论证的最低权利约束的用途，然后在多个性能与治理标准之间比较符合条件的替代方案。作为一项明确限定的规范性类比，《创世记》第二章启发了“禁越边界原则”：根据该原则，AI代理不得在预先授权的边界之外执行具有外部影响的行为。由此产生的第一性原理安全……

    arXiv:2611.13007v1 Announce Type: cross  Abstract: Artificial intelligence (AI) increasingly shapes consequential decisions, yet predictive accuracy is not a sufficient justification for delegated authority. This conceptual study develops an integrated framework for assessing just AI across normative, data, model, and governance layers. Drawing on plural theories of justice, sociotechnical fairness research, and publicly available governance standards, it maps six justice requirements to observable evidence, decision rights, and remedies. It proposes a two-stage evaluation rule: exclude uses that fail independently justified minimum rights constraints, then compare eligible alternatives across multiple performance and governance criteria. As an explicitly limited normative analogy, Genesis chapter 2 motivates a Forbidden-Boundary Principle (FBP), under which an AI agent must not execute externally consequential actions outside pre-authorized boundaries. The resulting First-Principle Sa
    
[^208]: 眼见未必为实：反事实病灶擦除教会小型医学视觉-语言智能体只报告图像中真实存在的内容

    Seeing Is Not Believing: Counterfactual Lesion Erasure Teaches a Small Medical Vision-Language Agent to Report Only What the Image Shows

    [https://arxiv.org/abs/2611.13006](https://arxiv.org/abs/2611.13006)

    该论文提出无需人工标注的反事实病灶擦除评测范式与证据敏感度差距（ESG）指标，并据此训练出40亿参数的医学视觉-语言智能体MedScope-4B，使其只报告图像中真实可见的发现，将“幻觉报告被擦除病灶”的比例从前沿模型的41–62%降至9.6%。

    

    医学视觉-语言模型（VLM）常常仅仅因为问题中提到了某个发现的名称就报告该发现，而标准基准测试无法捕捉这一现象。我们从公开分割数据集构建反事实视图，无需人工标注：包括通过图像修复擦除病灶的图像、在其他位置进行同形状对照编辑的图像，以及未经编辑的无病灶图像。证据敏感度差距（ESG）定义为擦除后“不存在或转诊/参考”类回答的比率减去对照编辑后的相应比率；对于忽略图像或对两种编辑反应相同的模型，该指标为零。MedScope-4B（基于Qwen3.5-4B、LoRA微调、配备图像工具）在由标签推导的训练轨迹上训练，按预先注册的规则进行模型选择，并一次性与九个前沿模型和六个开源VLM进行对比测试；随后通过四项预先注册的检验探究其机制。在涵盖五种成像类型的507例内部病例上，MedScope-4B达到ESG 0.758，且仅在9.6%的病例中声称存在已被擦除的病灶，而前沿模型分别为0.27–0.47和41–62%（所有p<0.001）。

    arXiv:2611.13006v1 Announce Type: new  Abstract: Medical vision-language models (VLMs) often report a finding because the question names it; standard benchmarks miss this. From public segmentation datasets we build counterfactual views without manual annotation: the lesion erased by inpainting, a same-shape control edit elsewhere, and unedited lesion-free images. The evidence sensitivity gap (ESG) is the rate of absent-or-refer answers after erasure minus the rate after the control edit; it is zero for a model that ignores the image or reacts to both edits alike. MedScope-4B (Qwen3.5-4B, LoRA, image tools) was trained on label-derived trajectories, selected by a pre-registered rule and tested once against nine frontier and six open VLMs; four pre-registered checks then probed the mechanism. On 507 internal cases across five imaging types, MedScope-4B reached ESG 0.758 and claimed an erased lesion in 9.6% of cases, against 0.27-0.47 and 41-62% for the frontier models (all p<0.001), with
    
[^209]: 程序的形态：用于轨迹到程序归纳的路径签名

    The Shape of a Program: Path Signatures for Trace-to-Program Induction

    [https://arxiv.org/abs/2611.13005](https://arxiv.org/abs/2611.13005)

    本文将粗路径理论引入程序归纳，用路径签名变换编码执行轨迹，在 Karel、List 和 RASP 三个领域几乎全面超越 LSTM、Transformer 和傅里叶基线（尤其在数据稀缺时优势最大），并利用流程天然的收缩性构建无需外部预言机的自举方法来揭示监督数据的冗余。

    

    从执行轨迹中合成程序是算法可解释性与复杂系统逆向工程中的一项根本性挑战。我们将粗路径理论的视角引入程序归纳，把执行轨迹视为高维粗路径，并用路径签名变换对其进行编码——这是来自随机分析的一种工具，其期望可被证明能够刻画程序所诱导的轨迹分布。在 Karel、List 和 RASP 三个程序归纳领域上，我们的对数签名轨迹编码器几乎在所有设置下都优于 LSTM、Transformer 和傅里叶基线，且在训练数据稀缺时优势最为显著。我们进一步观察到，轨迹到程序的流程天然具有收缩性：它往往能生成比真实程序更短的合法程序。我们将这一特性转化为一种自举方法，无需任何外部预言机即可揭示真实监督信号中存在的冗余。

    arXiv:2611.13005v1 Announce Type: cross  Abstract: Synthesizing programs from execution traces is a fundamental challenge in algorithmic interpretability and the reverse engineering of complex systems. We bring a rough-path-theoretic perspective to program induction, treating execution traces as high-dimensional rough paths and encoding them with the path signature transform, a tool from stochastic analysis whose expectation provably characterizes the distribution of traces a program induces. On three program-induction domains, Karel, List, and RASP, our log-signature trace encoder outperforms LSTM, Transformer, and Fourier baselines in almost every setting, with its largest margins when training data is scarce. We further observe that the trace-to-program pipeline is naturally contractive: it often produces valid programs that are shorter than the ground truth. We turn this into a bootstrapping procedure that exposes redundancy in the ground-truth supervision without any external orac
    
[^210]: PhysMoE：基于模态解耦VLM中单Token坐标表示的快速空间定位

    PhysMoE: Fast Spatial Grounding with Single-Token Coordinate Representation in Modality-Decoupled VLMs

    [https://arxiv.org/abs/2611.13000](https://arxiv.org/abs/2611.13000)

    提出PhysMoE框架，通过物理模态解耦的专家路由、复用视觉位置嵌入的单Token坐标表示以及可微SDF监督，实现了具身VLM中快速且精确的空间定位。

    

    具身视觉语言模型（VLM）是机器人规划与操作的骨干网络。然而，在定位任务的坐标预测方面，现有架构存在两个明显的瓶颈：离散Token化破坏了空间连续性且解码延迟高，而标准点回归方法在多接触可供性和不规则形状目标上表现不佳。为解决这些挑战，我们提出了PhysMoE，其包含三项核心贡献：（1）共享下层网络用于多模态上下文处理，同时将上层网络按物理模态路由到专用的空间（像素）专家和语言（文本）专家，从而消除梯度干扰；（2）通过复用视觉编码器学习到的位置嵌入，将每个坐标编码为单个Token，注入2D空间先验，使坐标与视觉特征天然对齐并简化训练；（3）对指向任务采用可微的有符号距离场（SDF）监督。（注：原摘要在此处被截断）

    arXiv:2611.13000v1 Announce Type: cross  Abstract: Embodied Vision-Language Models (VLMs) are backbones for robotic planning and manipulation. However, for coordinate prediction in grounding tasks, existing architectures suffer from two distinct bottlenecks: discrete tokenization disrupts spatial continuity with high decoding latency, while standard point regression fails on multi-contact affordances and irregularly shaped targets. To resolve these challenges, we propose PhysMoE, introducing three core contributions: (1) sharing lower layers for multimodal context while routing upper layers to specialized spatial (pixel) and linguistic (text) experts by physical modality, eliminating gradient interference; (2) encoding each coordinate as a single token by reusing the visual encoder's learned positional embeddings, injecting 2D spatial priors to natively align coordinates with visual features and ease training; and (3) differentiable Signed Distance Field (SDF) supervision on pointing, 
    
[^211]: TIB：超越对齐假设的样本级温和信息瓶颈多模态归因方法

    TIB: Sample-wise Tempered Information Bottleneck for Multimodal Attribution beyond Alignment Assumption

    [https://arxiv.org/abs/2611.12997](https://arxiv.org/abs/2611.12997)

    该论文提出TIB方法，通过样本级温和信息瓶颈突破现有多模态归因方法对模态对齐假设的依赖，解决了在语义错位情况下代理监督信号可靠性不均导致的后验过度收缩与强制解释问题。

    

    多模态（例如本文中的视觉-语言）归因旨在在无法获得真实解释性监督的情况下对模型进行解释。现有多模态归因方法通常依赖配对的模态作为代理监督，隐式地假设图像-文本对在语义上是对齐的。基于信息瓶颈（IB）的归因方法遵循这一通用范式，并通过全局充分性-压缩目标将其具体化。然而，在现实的多模态数据集中，由于部分语义错位和不可靠的跨模态对应关系，这一假设经常被违反。通过理论分析，我们证明在这种错位情况下，基于信息瓶颈的归因目标存在结构性局限：由于假设所有代理监督信号具有同等的可靠性，它们会导致后验过度收缩和强制解释。为解决这一问题，我们提出了T……

    arXiv:2611.12997v1 Announce Type: cross  Abstract: Multimodal (e.g. vision-language, in this work) attribution aims to interpret the models without access to ground-truth explanatory supervision. Existing multimodal attribution methods typically rely on paired modalities as proxy supervision, implicitly assuming that image-text pairs are semantically aligned. Information bottleneck (IB)-based attribution methods follow this general paradigm and instantiate it through a global sufficiency-compression objective. In realistic multimodal datasets, however, this assumption is frequently violated due to partial semantic misalignment and unreliable cross-modal correspondences. Through theoretical analysis, we show that, under such misalignment, information-bottleneck-based attribution objectives exhibit a structural limitation: by assuming equal reliability across all proxy supervision signals, they induce posterior over-contraction and forced explanations. To address this issue, we propose T
    
[^212]: ZeroPoint：轻量级基于能量的决策模型

    ZeroPoint: Lightweight Energy-Based Decision Model

    [https://arxiv.org/abs/2611.12992](https://arxiv.org/abs/2611.12992)

    本文提出ZeroPoint，通过引入一个对任意输入能量恒为零的“空答案”固定状态，为基于能量的决策模型提供显式的能量参考水平，使模型能够判断并拒绝“没有任何可接受选项”的情形。

    

    基于能量的模型（EBM）为观测变量与预测变量的每一个联合取值分配一个标量能量。在决策模型中，预测是从每个输入所附带的有限选项列表中选出的，因此推理可简化为比较各选项的能量，而训练则是降低正确选项相对于其他选项的能量。一个完整的决策还必须考虑没有任何选项可以接受的情况。然而，标准EBM学到的能量只定义了相对兼容性：给所有能量加上同一个依赖于输入的量，预测结果、吉布斯分布以及常用的训练损失都不会改变。因此，模型缺少一个可供判断任何选项是否合适的、学习得到的参考水平。我们提出ZeroPoint，这是一种将“空答案”作为模型自身状态的EBM，该状态对每个输入的能量均为零。这一固定状态为能量提供了显式的参考。

    arXiv:2611.12992v1 Announce Type: cross  Abstract: Energy-based models (EBMs) assign a scalar energy to every joint value of the observed and predicted variables. In decision models, the prediction is chosen from a finite list of options provided with each input. Inference then reduces to comparing the energies of the options, and training lowers the energy of the correct option relative to the others. A complete decision must also account for the possibility that no option is acceptable. However, the energies learned by a standard EBM define only relative compatibility: adding the same input-dependent quantity to all energies leaves the predictions, the Gibbs distribution, and the usual training losses unchanged. The model therefore has no learned reference level against which to decide whether any option fits. We propose ZeroPoint, an EBM that makes the empty answer a state of the model with energy zero for every input. This fixed state provides an explicit reference for the energies
    
[^213]: 将元多目标化专门化为软件工程优化的基线方法

    Specializing Meta Multi-Objectivization as Baseline for Software Engineering Optimization

    [https://arxiv.org/abs/2611.12991](https://arxiv.org/abs/2611.12991)

    本文将元多目标化（MMO）从仅适用于配置调优的模型专门化为软件工程优化的通用基线，通过理论分析和九个软件工程优化问题的实证研究，阐明了辅助目标选择的关键性并验证了MMO克服局部最优的有效性。

    

    鉴于许多软件工程（SE）优化问题的离散特性，陷入局部最优仍然是优化器设计中的一个关键问题。元多目标化是一种通过引入额外的辅助目标来克服局部最优的优化模型。然而，MMO最初仅为配置调优而设计，其对于其他软件工程问题的适用性尚存疑问，特别是关于辅助目标的选择。在本文中，我们研究了将MMO专门化为一系列软件工程优化问题的通用基线的可能性与解决方案，使其成为未来寻求缓解局部最优问题的软件工程优化器的基础对应方法。我们通过提供理论依据和实证证据来实现这一目标：我们通过分析展示了辅助目标的选择如何至关重要，同时证明了MMO在九个软件工程优化问题上的有效性……

    arXiv:2611.12991v1 Announce Type: cross  Abstract: Given the discrete nature of many software engineering (SE) optimization problems, trapping at local optima remains a crucial issue in the optimizer designs. Meta Multi-Objectivization (MMO) is an optimization model for overcoming local optima by introducing an extra, auxiliary objective. Yet, MMO is originally designed only for configuration tuning, leaving its suitability for other SE problems in doubt, particularly regarding the choice of the auxiliary objective. In this paper, we study the possibility and resolution for specializing MMO as a general baseline across a range of SE optimization problems, making it a fundamental counterpart for future SE optimizers that seek to mitigate the issue of local optima. We do so by providing both theoretical remit and empirical evidence: we analytically show how the choice of the auxiliary objective is crucial while demonstrating the effectiveness of MMO under nine SE optimization problems co
    
[^214]: 第三只眼：超越递归自我改进中的即时收益

    Third Eye: Beyond Immediate Gains in Recursive Self-Improvement

    [https://arxiv.org/abs/2611.12988](https://arxiv.org/abs/2611.12988)

    论文通过大规模受控实验发现，递归自我改进中看似有用的自我更新在经过第二轮更新后效用会进一步降低（甚至候选平均即时效用已为负值），因此对自我改进的评估必须超越第一步的即时收益。

    

    一个有用的自我更新在模型再次更新之后是否仍然有用？Third Eye 将即时性能与进一步原生更新的结果分离开来。我们在数学和代码任务上评估了五个3-8B参数的基座模型：共64次全新运行，每次运行产生三个候选，每个候选进行三次后续更新，且候选选择与最终评分使用互不相交的题目。在均衡的Qwen3-4B/Llama-3.2-3B对比中，经即时选择得到的效用接近于零，但后续更新使其降低了1.78个百分点（同时95%置信区间[0.68, 2.88]）。候选平均的即时效用本身已是负值：选择机制在一定程度上解释了表面上中性的第一步。后续更新同样降低了SmolLM3-3B和Llama-3.1-8B的效用；Qwen3-8B的结果仍无定论。目标损失和分布外损失主导了保留效果的变化。匹配配方和接收者-捐赠者对照实验既未找到可靠的缓解手段，也未确定单一机制。这些结果支持对……进行评估（摘要在此处截断）

    arXiv:2611.12988v1 Announce Type: cross  Abstract: Does a useful self-update remain useful after the model updates again? Third Eye separates immediate performance from the outcome of a further native update. We evaluate five 3-8B backbones on math and code: 64 fresh runs, three candidates per run, and three continuations per candidate. Selection and final scoring use disjoint items. In the balanced Qwen3-4B/Llama-3.2-3B comparison, immediate-selected utility is near zero, but continuation reduces it by 1.78 percentage points (simultaneous 95% interval [0.68, 2.88]). Candidate-averaged immediate utility is already negative: selection partly accounts for the apparently neutral first step. Continuation also lowers utility for SmolLM3-3B and Llama-3.1-8B; Qwen3-8B remains inconclusive. Target and out-of-distribution losses dominate retention changes. Matched recipe and recipient-donor controls identify neither a reliable mitigation nor a single mechanism. The results support evaluating pe
    
[^215]: ICLayout-Bench：面向定制电路版图设计的LLM智能体基准测试

    ICLayout-Bench: Benchmarking LLM Agents for Custom Circuit Layout Design

    [https://arxiv.org/abs/2611.12982](https://arxiv.org/abs/2611.12982)

    该论文提出了ICLayout-Bench基准测试，首次系统性评估LLM智能体使用EDA工具从晶体管级网表生成高质量集成电路版图的能力，涵盖165个任务、13个PDK以及3nm至180nm的工艺节点。

    

    大语言模型（LLM）智能体正在被探索应用于电路设计，但其生成高质量定制版图的能力尚未得到广泛评估。ICLayout-Bench是一个基准测试，其中智能体使用电子设计自动化（EDA）工具从晶体管级电路网表构建和优化GDSII版图。这些版图通过物理验证、寄生参数提取和后版图仿真进行评估。通过物理和功能检查的版图将依据电气性能和面积进行评分。该基准测试包含165个任务，涵盖13个工艺设计套件（PDK），工艺节点范围从3纳米到180纳米。任务覆盖开源和商业EDA流程中的模拟、混合信号和数字电路。实验评估了7个模型在21个公开任务和19个私有任务上的22种配置。公开任务上的平均版图得分范围为4.22至80.26，私有任务上为1.43至67.01。

    arXiv:2611.12982v1 Announce Type: cross  Abstract: Large language model (LLM) agents are being explored for circuit design, but their ability to produce high-quality custom layouts has not been extensively evaluated. ICLayout-Bench is a benchmark in which agents use electronic design automation (EDA) tools to construct and optimize GDSII layouts from transistor-level circuit netlists. These layouts are evaluated using physical verification, parasitic extraction, and post-layout simulation. Layouts that pass the physical and functional checks are scored on electrical performance and area. The benchmark includes 165 tasks across 13 process design kits (PDKs), covering technology nodes from 3 nm to 180 nm. The tasks span analog, mixed-signal, and digital circuits in open-source and commercial EDA flows. The experiments evaluate 22 configurations spanning seven models on 21 public and 19 private tasks. Mean layout scores range from 4.22 to 80.26 on public tasks and from 1.43 to 67.01 on pr
    
[^216]: MADSNet：一种用于脉冲相机图像重建的多尺度轴向注意力密集脉冲网络

    MADSNet: A Multi-Scale Axial-Attentive Dense Spiking Network for Spike Camera Image Reconstruction

    [https://arxiv.org/abs/2611.12980](https://arxiv.org/abs/2611.12980)

    提出了一种多尺度轴向注意力密集脉冲网络MADSNet，通过分层脉冲编码器-解码器结构实现高质量脉冲相机图像重建，在SRED数据集上显著超越现有SNN重建方法。

    

    脉冲相机将动态场景编码为高时间分辨率的二值脉冲流，以实现高效的高速视觉处理。尽管脉冲神经网络（SNN）能够以稀疏且仅累加的计算方式处理这类信号，但现有的基于SNN的重建方法通常网络结构较浅，在重建保真度上仍落后于基于人工神经网络（ANN）的模型。本文提出了MADSNet，一种用于脉冲相机图像重建的多尺度轴向注意力密集脉冲网络。MADSNet采用分层脉冲编码器-解码器结构，其中多尺度脉冲干模块提取早期特征，轴向脉冲注意力捕获长程依赖关系，密集脉冲瓶颈层增强特征复用，脉冲到连续的解码器则从脉冲域表示中重建连续的强度图像。在灰度和彩色脉冲重建实验中，MADSNet在SRED数据集上比SSIR高出4.3 dB PSNR，比Spk2ImgNet高出1.94 dB。

    arXiv:2611.12980v1 Announce Type: cross  Abstract: Spike cameras encode dynamic scenes into high-temporal-resolution binary spike streams for efficient high-speed vision. Although spiking neural networks (SNNs) can process such signals with sparse and accumulate-only computation, existing SNN-based reconstruction methods are typically shallow and still lag behind ANN-based models in reconstruction fidelity. This paper proposes MADSNet, a Multi-scale Axial-attentive Dense Spiking Network for spike camera image reconstruction. MADSNet adopts a hierarchical spiking encoder--decoder, where a multi-scale spike stem extracts early features, axial spike attention captures long-range dependencies, a dense spiking bottleneck enhances feature reuse, and a spike-to-continuous decoder reconstructs continuous intensity images from spike-domain representations. Experiments on gray-scale and color spike reconstruction show that MADSNet outperforms SSIR by 4.3 dB PSNR and Spk2ImgNet by 1.94 dB on SRED
    
[^217]: 面向一步式生成的轨迹正则化分布蒸馏

    Trajectory-Regularized Distribution Distillation for One-Step Generation

    [https://arxiv.org/abs/2611.12978](https://arxiv.org/abs/2611.12978)

    提出轨迹正则化分布蒸馏（TRD），通过单次前向传播预测的动量混合策略统一分布级与轨迹级监督，无需额外可训练网络即可实现高质量的一步式生成。

    

    将流匹配模型蒸馏用于一步式生成通常遵循两种范式。轨迹对齐方法利用教师的概率流ODE对学生模型进行监督，但在极小的采样预算下其性能会下降。分布匹配方法直接优化生成分布，但估计其反向Kullback-Leibler梯度通常需要一个额外的网络来跟踪学生模型不断演变的分数。我们提出轨迹正则化分布蒸馏（TRD），通过在单次学生前向传播中预测的动量混合策略，将分布级与轨迹级监督结合起来。该策略同时充当一步式生成器、学生边际速度的估计器（从而免除分布匹配中对第二个可训练网络的需求），以及沿教师ODE轨迹进行监督的闭式时变速度场。在SANA-1.6上的实验……（原文在此截断）

    arXiv:2611.12978v1 Announce Type: cross  Abstract: Distilling flow matching models for one-step generation generally follows two paradigms. Trajectory alignment supervises the student with the teacher's probability flow ODE, but its performance degrades under very small sampling budgets. Distribution matching directly optimizes the generated distribution, but estimating its reverse Kullback-Leibler gradient typically requires a separate network to track the evolving student score. We propose trajectory-regularized distribution distillation (TRD), which combines distribution-level and trajectory-level supervision through a momentum-mixture policy predicted in a single student forward pass. This policy serves simultaneously as a one-step generator, an estimator of the student's marginal velocity that removes the need for a second trainable network in distribution matching, and a closed-form time-dependent velocity field supervised along the teacher ODE trajectory. Experiments on SANA-1.6
    
[^218]: TrustMod-SM：用于评估大型语言模型在社交媒体内容审核中可信度的多维度基准

    TrustMod-SM: A Multi-Axis Benchmark for Evaluating Trustworthiness of LLMs in Social Media Content Moderation

    [https://arxiv.org/abs/2611.12970](https://arxiv.org/abs/2611.12970)

    提出了TrustMod-SM多维度基准，从可信性、公平性、安全性、鲁棒性和上下文完整性五个维度评估LLM在社交媒体内容审核中的可信度，揭示了更高的准确率往往伴随更大的群体偏见，且扩大模型规模并不能统一解决可信度问题。

    

    大型语言模型（LLM）在社交媒体内容审核方面展现出强大潜力，但其在整体准确率之外的可信度仍未得到充分探索。我们提出了TrustMod-SM，这是一个包含约29,000个实例、覆盖八个数据集的基准，用于在五个维度上评估LLM审核员：可信性、公平性、安全性、鲁棒性和上下文完整性。通过评估16个开放权重模型（参数规模从0.5B到284B，涵盖纯文本和视觉-语言架构）以及一个闭源前沿模型，我们发现可信度差距持续存在。更高的检测准确率往往伴随着更大的群体差异，模型经常遵从隐瞒或夸大的提示指令，对错误的预测赋予高置信度，很少将模糊案例升级处理，并且难以区分仇恨言论与反击言论或语义回收用语（尤其在8B参数以下的模型中）。扩大模型规模并不能统一解决这些问题，且没有单一模型在所有维度上表现优异。

    arXiv:2611.12970v1 Announce Type: new  Abstract: Large Language Models (LLMs) show strong potential for social media content moderation, but their trustworthiness beyond overall accuracy remains underexplored. We introduce TrustMod-SM, a benchmark of roughly 29,000 instances across eight datasets, to evaluate LLM moderators on five dimensions: trustfulness, fairness, safety, robustness, and context integrity. Evaluating 16 open-weight models (0.5B-284B parameters, spanning text-only and vision-language architectures) and one closed frontier model, we find persistent trustworthiness gaps. Higher detection accuracy often coincides with greater demographic disparity, and models frequently comply with concealment or exaggeration prompts, assign high confidence to wrong predictions, rarely escalate ambiguous cases, and struggle to separate hate speech from counter-speech or reclaimed language (especially below 8B parameters). Scaling does not uniformly resolve these issues, and no single mo
    
[^219]: BuySideFlow：面向投资研究中LLM智能体的时点基准

    BuySideFlow: A Point-in-Time Benchmark for LLM Agents in Investment Research

    [https://arxiv.org/abs/2611.12966](https://arxiv.org/abs/2611.12966)

    该论文提出了首个面向投资研究中LLM智能体的时点有效性基准BuySideFlow，包含408个从业者来源的中文金融数据任务，评估显示智能体准确率仅16.2%至36.8%，且在通过结果检查的运行中仍有50.8%存在时点违规，揭示了仅凭结果验证不足以确保金融数据分析的正确性。

    

    大型语言模型（LLM）可以为投资研究生成SQL和SQL-Python工作流，但确保时点有效性仍然具有挑战性。智能体必须在数据检索和金融分析的全过程中执行信息披露截止日期约束，并选择适当的历史状态和记录版本。我们提出了BuySideFlow，这是一个包含408个由从业者来源的中文任务的基准，涵盖股票、基金、固定收益和宏观经济，基于来自持牌生产级金融数据库的固定快照。我们提供了可执行的参考实现和时间错误分类体系，将结果检查与对已执行工作流的专家审计相结合。我们在所有任务上评估了九种智能体配置，结果准确率从16.2%到36.8%不等。对100个共享任务上900次运行的审计证实，在通过结果检查的380次运行中有193次（50.8%）存在时点违规，表明仅凭输出一致并不能确立结果的正确性。

    arXiv:2611.12966v1 Announce Type: new  Abstract: Large language models (LLMs) can generate SQL and SQL--Python workflows for investment research, but ensuring point-in-time (PIT) validity remains challenging. Agents must enforce disclosure cutoffs and select the appropriate historical states and record versions throughout data retrieval and financial analysis. We introduce \textsc{BuySideFlow}, a benchmark of 408 practitioner-derived Chinese-language tasks spanning equities, funds, fixed income, and macroeconomics over a fixed snapshot from a licensed production financial database. We provide executable references and a taxonomy of temporal errors, pairing result checks with expert audits of executed workflows. We evaluate nine agent configurations on all tasks, with result accuracy ranging from 16.2\% to 36.8\%. An audit of 900 runs on 100 shared tasks confirms PIT violations in 193 of the 380 runs that pass result checks (50.8\%), showing that output agreement alone does not establis
    
[^220]: RBLA+：面向异构客户端联邦微调的高效特定秩 LoRA 适配器提取

    RBLA+: Efficient Rank-Specific LoRA Adapter Extraction for Federated Fine-Tuning with Heterogeneous Clients

    [https://arxiv.org/abs/2611.12961](https://arxiv.org/abs/2611.12961)

    提出RBLA+方法，为客户端LoRA秩异构的联邦微调实现高效的特定秩适配器提取，在低计算开销的同时控制重建误差。

    

    基于低秩适配的联邦微调通过训练并交换紧凑的可训练 LoRA 因子，降低了通信和计算开销。客户端可能使用不同的 LoRA 秩，以满足适配器通信预算并适应不同的本地适配需求。服务器必须在聚合过程中处理这些差异，并按每个客户端的秩构建适配器用于本地训练。我们区分了两个环节：预聚合对齐（在聚合之前处理因子维度和缩放约定）与特定秩 LoRA 适配器提取（从聚合后的全局模型中构建与客户端规模相匹配的参数对）。特定秩提取的两种常见方法是直接秩截断和对稠密更新进行奇异值分解（SVD）。直接前缀截断保留一个聚合因子的前导列和另一个因子的对应行，虽然代价低，但可能增大重建误差。（注：原文摘要在此处被截断）

    arXiv:2611.12961v1 Announce Type: new  Abstract: Federated fine-tuning with low-rank adaptation (LoRA) reduces communication and computational overhead by training and exchanging compact trainable LoRA factors. Clients may use different LoRA ranks to meet adapter communication budgets and accommodate different local adaptation needs. The server must process these differences during aggregation and construct adapters at each client's rank for local training. We distinguish pre-aggregation alignment, which handles factor dimensions and scaling conventions before aggregation and rank-specific LoRA adapter extraction, which constructs client-sized pairs from the aggregated global model. Two common approaches to rank-specific extraction are direct rank truncation and Singular Value Decomposition (SVD) of the dense update. Direct prefix truncation keeps the leading columns of one aggregated factor and the corresponding rows of the other. It is cheap but can increase reconstruction error. The
    
[^221]: IIT Patna团队参加“ICPR 2026 TVRID基于俯视RGB-深度相机的隐私保护行人重识别竞赛”

    IIT Patna at "ICPR 2026 TVRID Competition on Privacy-Preserving Person Re-Identification from Top-View RGB-Depth Camera"

    [https://arxiv.org/abs/2611.12960](https://arxiv.org/abs/2611.12960)

    该论文介绍了IIT Patna团队在ICPR 2026 TVRID隐私保护俯视RGB-Depth行人重识别竞赛中的方法，其中最优方法在深度赛道上以93.51% mAP和90.42% CMC@1获得第4名，团队总体排名第5和第7。

    

    基于俯视相机的隐私感知行人重识别（Re-ID）以人体测量学形状特征取代正面外观，而深度感知进一步减少了基于颜色的身份线索。ICPR 2026 TVRID竞赛采用四台同步的Intel RealSense D455相机采集的RGB与深度数据流，涵盖平坦、上坡、下坡和斜视四种视角，共86个身份来评估这一场景，并通过mAP和CMC@1在同相机进出、上坡-下坡以及平坦-斜视等场景中衡量性能。该竞赛共有147名注册参与者和33个最终提交。来自IIT Patna的三个团队参加了竞赛：'abhinavkumar03'和'sjkdgfhl_'在31个排名参赛者中分列第5和第7名，'vivek katuri'在RGB赛道上达到77.67% CMC@1 / 85.20% mAP。本文介绍了这两个进入前十名团队的方法。方法一（abhinavkumar03）在深度赛道上取得93.51% mAP和90.42% CMC@1，在28个参赛者中排名第4，而方法二……（摘要原文在此处截断）

    arXiv:2611.12960v1 Announce Type: new  Abstract: Privacy-aware person re-identification (Re-ID) from top-view cameras replaces frontal appearance with anthropometric shape, while depth sensing further reduces color-based identity cues. The ICPR 2026 TVRID competition evaluates this setting using 86 identities captured by four synchronized Intel RealSense D455 cameras with RGB and Depth streams across flat, ascent, descent, and oblique viewpoints. Performance is measured using mAP and CMC@1 across within-camera in/out, ascent-descent, and flat-oblique scenarios. The competition had 147 registered participants and 33 final submissions. Three teams from IIT Patna participated: 'abhinavkumar03' and 'sjkdgfhl_' ranked 5th and 7th among 31 ranked participants, and 'vivek katuri' reached 77.67% CMC@1 / 85.20% mAP on the RGB track. This paper presents the methods of the two top-10 teams. Methodology 1 (abhinavkumar03) achieves 93.51% mAP and 90.42% CMC@1 on Depth, ranking 4th of 28, while Meth
    
[^222]: 监控单一通道并非通信控制：联合消息-路由编码自适应重新分配的因果证据

    Monitoring One Channel Is Not Communication Control: Causal Evidence for Adaptive Reallocation of Joint Message--Route Codes

    [https://arxiv.org/abs/2611.12959](https://arxiv.org/abs/2611.12959)

    本研究通过端到端游戏的因果干预实验证明，监控单一通信通道会促使学习智能体将联合消息-路由编码自适应地重新分配到未受监控的路由上，而不是消除通信本身，且路由具有语义功能。

    

    监控一个可见通道可以改变学习智能体的通信方式，而不会消除通信本身的有用性。我们在一个端到端游戏中检验了这一区别，其中发送方和接收方在没有通信监督的情况下学习离散协议和路由策略。在两个独立的20随机种子批次中，分别有20/20和19/20的协议形成。在线部分覆盖相比无监控使影子路由份额增加了0.701 [0.574, 0.826]和0.575 [0.412, 0.737]，同时任务准确率保持在0.905/0.913，带符号的冻结消息贡献保持在0.592/0.563。完全覆盖消除了大部分选择性迁移，但没有消除通信本身。冻结干预（包括一个在结果产生前的新鲜确认干预）表明路由是具有语义性的：联合消息-路由打乱改变了0.668的接收方预测，而仅路由仍保留了0.388的准确率（随机水平为0.25）。修正后的显式沉默成本使18/20的策略完全沉默，并将平均发送率降至0.050。

    arXiv:2611.12959v1 Announce Type: new  Abstract: Monitoring a visible channel can change how learned agents communicate without removing why communication is useful. We test this distinction in an end-to-end game where a sender and receiver learn a discrete protocol and route policy without communication supervision. In two independent 20-seed batches, 20/20 and 19/20 protocols form. Online partial coverage increases shadow-route share over no monitoring by .701 [.574,.826] and .575 [.412,.737], while task accuracy remains .905/.913 and signed frozen message contribution remains .592/.563. Complete coverage removes most selective migration without removing communication. Frozen interventions, including a fresh pre-outcome confirmation, show that route is semantic: joint message--route shuffling changes .668 of receiver predictions and route alone retains .388 accuracy versus .25 chance. A corrected explicit-silence cost makes 18/20 policies fully silent and lowers mean sending to .050,
    
[^223]: 具身科学智能体：让具身实验设计变得可学习

    Embodied Science Agents: Making Embodied Experimental Design Learnable

    [https://arxiv.org/abs/2611.12958](https://arxiv.org/abs/2611.12958)

    该论文指出当前自动驾驶实验室存在“语言能力增强但物理具身根基被切断”的根本悖论，并提出应沿认知、实验、具身三个相互依存的可学习性维度构建具身科学智能体，使实验设计本身成为可学习的能力。

    

    近期大型模型的进展推动了“AI for Science”向智能体科学方向发展，即AI智能体能够自主开展科学发现。实现这一愿景需要将计算推理与物理实验之间的闭环打通，因此自动驾驶实验室（SDLs）成为关键的使能因素。然而，现有SDL存在一个根本性悖论：基础模型的集成虽然提升了语言层面的精巧程度，却切断了物理层面的具身根基。当前架构将模型视为脱离身体的规划器，仅输出符号化的工作流程交由独立的机器人系统执行，这在本质上阻碍了具身实验直觉的发展。我们认为，克服这一悖论需要在三个相互依存的维度上嵌入“可学习性”：认知可学习性（学习研究什么）、实验可学习性（学习在哪里搜索）以及具身可学习性（学习如何行动）。（注：原始摘要在此处被截断）

    arXiv:2611.12958v1 Announce Type: new  Abstract: Recent advances in large models have accelerated AI for Science toward agentic science, where AI agents autonomously conduct scientific discovery. Realising this vision requires closing the loop between computational reasoning and physical experimentation, positioning self-driving laboratories (SDLs) as critical enablers. However, existing SDLs exhibit a fundamental paradox: foundation model integration has increased linguistic sophistication while severing physical grounding. By treating models as disembodied planners that output symbolic workflows for separate robotic systems, current architectures inherently block the development of embodied experimental intuition. We argue that overcoming this paradox requires embedding \emph{learnability} along three interdependent axes: cognitive learnability (learning what to investigate), experimental learnability (learning where to search), and embodied learnability (learning how to act). Buildi
    
[^224]: ADIndex：基于组级蒸馏的层级自适应稀疏注意力索引

    ADIndex: Layer-Adaptive Sparse Attention Indexing via Group-Level Distillation

    [https://arxiv.org/abs/2611.12956](https://arxiv.org/abs/2611.12956)

    ADIndex通过逐层校准与组级蒸馏训练层级自适应的稀疏注意力索引器，并辅以低延迟推理策略，大幅降低长上下文推理中索引环节的延迟开销。

    

    DeepSeek稀疏注意力（DSA）通过token级选择降低了注意力的计算成本。然而，其索引器仍需为每个查询对整个因果前缀进行评分，使索引成为长上下文推理中延迟的主要来源。值得注意的是，在长上下文长度下，仅索引器一项就可能占总预填充GPU时间的50%以上。为解决这一瓶颈，我们提出了ADIndex，一种面向token级稀疏注意力的自适应、蒸馏驱动的索引方法。ADIndex集成了三个关键组件：(i) 逐层校准策略，可自适应地适应跨查询检索重叠，其动机源自我们观察到的显著的层间异质性；(ii) 组级蒸馏方案，利用这种异质性高效训练自适应的逐层组索引器，从单个查询中捕捉关键的检索信号；(iii) 低延迟推理策略，用于降低……（原文摘要在此处截断）

    arXiv:2611.12956v1 Announce Type: cross  Abstract: DeepSeek Sparse Attention (DSA) reduces the computational cost of attention through token-level selection. However, its indexer still scores the entire causal prefix for each query, making indexing a dominant source of latency in long-context inference. Notably, the indexer alone can account for more than 50% of total prefill GPU time at long context lengths. To address this bottleneck, we propose ADIndex, an Adaptive and Distillation-driven Indexing method for token-level sparse attention. ADIndex integrates three key components: (i) a layer-wise calibration strategy that adaptively accommodates cross-query retrieval overlap, motivated by our observation of pronounced inter-layer heterogeneity; (ii) a group-level distillation scheme that leverages this heterogeneity to efficiently train adaptive layer-wise group indexers, capturing salient retrieval signals from individual queries; and (iii) a low-latency inference strategy that reduc
    
[^225]: 预测性惊奇作为可解释时间序列的自锚定概念瓶颈

    Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series

    [https://arxiv.org/abs/2611.12955](https://arxiv.org/abs/2611.12955)

    ConceptTime 利用冻结概率预测器产生的“预测性惊奇”这一单一自监督信号，同时完成时间序列分段定位与概念表征，打破了边界与模式相互依赖的循环，从而在时间序列分类中同时实现准确性、可解释性和证据不足时的弃权能力。

    

    部署在风险敏感领域的时间序列分类器必须做到：(1) 准确，(2) 可解释，(3) 在证据不足时能够弃权（拒绝预测）。现有方法无法同时满足这三点，且这一障碍是结构性的，而非操作层面的。若不知道分段边界，就无法定义有意义的时间模式；而若不知道期待何种模式，又无法划分有意义的边界。现有方法通过使用固定窗口或专家标注的词汇表来打破这种循环，从而至少牺牲了上述三个属性中的一个。我们提出 ConceptTime，它通过单一的自监督信号闭环解决这一问题。一个冻结的概率预测器在序列的每个时间步预测下一个观测值的分布。我们将其预测与实际信号之间的不匹配称为预测性惊奇。该信号既能定位分段边界，又能刻画由此产生的每个概念……

    arXiv:2611.12955v1 Announce Type: cross  Abstract: Time-series classifiers deployed in risk-sensitive domains must be (1) accurate, (2) interpretable, and (3) able to abstain when evidence is weak. Existing methods do not deliver all 3 at once, and the obstacle is structural, rather than logistical. Meaningful temporal patterns cannot be defined without knowing segment boundaries, yet meaningful boundaries cannot be placed without knowing which patterns to expect. Existing approaches break this circularity by using fixed windows or expert-annotated vocabularies, thereby sacrificing at least one of the 3 properties. We introduce ConceptTime, which closes the loop through a single self-supervised signal. A frozen probabilistic forecaster predicts a distribution over the next observation at every timestep of the sequence. We call the mismatch between its prediction and the realized signal the predictive surprise. This signal both locates segment boundaries and characterizes each resulting
    
[^226]: Physics2Weight：基于大语言模型从物理定律直接生成神经网络架构与权重

    Physics2Weight: Direct Neural Network Architecture and Weight Generation from Physical Laws Using Large Language Models

    [https://arxiv.org/abs/2611.12954](https://arxiv.org/abs/2611.12954)

    该论文提出Physics2Weight（PtW）框架，利用大语言模型根据控制方程及初始边界条件直接生成神经网络权重并编辑架构，通过物理约束提示、物理引导的架构编辑和迭代权重细化，在无需梯度优化的情况下实现更高精度和物理一致性的偏微分方程求解。

    

    大语言模型（LLM）的最新进展催生了权重空间编程（WSP）技术，它可以绕过基于梯度的优化直接生成神经网络权重。然而，将WSP扩展到物理问题仍然具有挑战性，尤其是在同时实现求解精度和物理一致性方面。本文提出了Physics2Weight（PtW）框架，该框架将WSP扩展到偏微分方程求解领域，通过基于控制方程以及初始条件和边界条件直接生成神经网络权重并编辑网络架构。PtW引入了物理约束提示和物理引导的架构编辑，并结合WSP的迭代权重细化，从而在无需基于梯度的优化的情况下提高求解精度和物理一致性。在四个偏微分方程（即热方程、对流方程、Burgers方程和Allen-Cahn方程）上的实验表明，其求解误差低于传统物理……（摘要在此处截断）

    arXiv:2611.12954v1 Announce Type: new  Abstract: Recent advances in large language models (LLMs) have enabled weight-space programming (WSP), which directly generates neural network weights without gradient-based optimization. However, extending WSP to physical problems remains challenging, particularly in achieving both solution accuracy and physical consistency. In this paper, we propose Physics2Weight (PtW), a framework that extends WSP to solving partial differential equations by directly generating neural network weights and editing architectures based on governing equations and initial and boundary conditions. PtW introduces physics-constrained prompting and physics-guided architecture editing, combined with iterative weight refinement from WSP, to improve solution accuracy and physical consistency without gradient-based optimization. Experiments on four PDEs, namely the heat, advection, Burgers', and Allen--Cahn equations, demonstrate lower solution errors than conventional phys
    
[^227]: 智能体架构的成本-成功率权衡：面向企业式标准操作流程的受控评估

    Agent Architecture Cost-Success Trade-offs: A Controlled Evaluation across Enterprise-Style SOPs

    [https://arxiv.org/abs/2611.12953](https://arxiv.org/abs/2611.12953)

    该论文通过固定模型、任务与工具、仅改变架构的受控实验，在13个企业SOP上比较四种智能体架构，发现不存在普遍最优架构，架构选择应取决于具体SOP的复杂程度。

    

    企业越来越多地使用大型语言模型智能体来执行标准操作流程（SOP），即长链条、多分支、工具调用密集的工作流。然而，在相互竞争的智能体架构之间进行选择时，目前几乎没有受控的实证证据可供参考。我们在保持模型、任务和工具完全一致、仅改变架构的条件下，比较了四种设计（函数调用式单智能体、技能增强式智能体、集中式多智能体编排，以及分诊式智能体集群，共八种实验配置），覆盖13个企业SOP，并引入了一个技能定义质量评分标准，用于构建相互匹配的高质量与低质量技能语料库。研究得出三点结论：（1）不存在普遍最优的架构，正确的选择取决于具体的SOP。在同一SOP上，任务成功率从最难的不足10%到最容易的接近100%不等；从总体上看，没有任何架构在统计上显著可分，而更复杂的结构化设计似乎只有在复杂、多分支的SOP上才可能带来回报（直接……）

    arXiv:2611.12953v1 Announce Type: new  Abstract: Enterprises increasingly use large language model agents to execute standard operating procedures (SOPs): long, branching, tool-intensive workflows. Yet there is little controlled evidence for choosing among competing agentic architectures. Holding the model, tasks, and tools fixed and varying only the architecture, we compare four designs (a function-calling single agent, a skill-augmented agent, centralized multi-agent orchestration, and a triage swarm; eight arms) across 13 enterprise SOPs, and introduce a skill-definition quality rubric to build matched high and low-quality skill corpora. Three takeaways emerge. (1)~\textbf{There is no universal best architecture; the right choice is SOP-dependent.} On a given SOP, task success ranges from under $10\%$ on the hardest to near $100\%$ on the easiest; no architecture is statistically separable in aggregate, and added structure only plausibly pays off on complex, branching SOPs (a direct
    
[^228]: SetPrint：基于响应变化集合对应关系的黑盒大语言模型指纹识别

    SetPrint: Black-Box LLM Fingerprinting via Setwise Correspondence of Response Changes

    [https://arxiv.org/abs/2611.12950](https://arxiv.org/abs/2611.12950)

    SetPrint提出了一种无需训练的黑盒LLM指纹识别方法，首次同时利用集合内部响应变化关系和两个模型对相同扰动的响应变化对应关系，来判定目标模型是否源自源模型。

    

    黑盒大语言模型指纹识别旨在仅在可观察模型对输入查询的输出响应的情况下，判断目标模型是否源自某个源模型。揭示这种关系的一种方式是对每个查询施加扰动，并观察两个模型由此产生的响应，从而得到两组相应的响应变化集合。这些集合提供了两种互补的信息来源：每组内部响应变化之间的关系，以及两组之间响应变化的对应关系。然而，现有方法并未充分同时利用这两者。我们提出了SetPrint，这是一种无需训练的方法，能够同时捕捉集合内部关系和集合间对应关系。SetPrint将每个响应变化表示为相对于其所在集合平均变化的偏差，然后比较两个模型由同一扰动所引起的变化。在LeaFBench上，SetPrint持续优于现有的训练（原文摘要在此处截断）。

    arXiv:2611.12950v1 Announce Type: new  Abstract: Black-box LLM fingerprinting aims to determine whether a target model was derived from a source model when only the output responses to input queries are observable. One way to reveal this relationship is to perturb each query and observe the resulting responses of both models, yielding two corresponding sets of response variations. These sets provide two complementary sources of information: the relations among response changes within each set, and the correspondence of response changes between the two sets. Existing approaches, however, have not fully exploited both. We introduce \textbf{SetPrint}, a training-free method that jointly captures both within-set relations and between-set correspondence. SetPrint expresses each response change relative to the mean change of its own set, and then compares changes from the two models that were induced by the same perturbation. On LeaFBench, SetPrint consistently outperforms existing training-
    
[^229]: 当证明已充裕时，论文的意义何在？

    When Proofs Are Abundant, What Is a Paper For?

    [https://arxiv.org/abs/2611.12943](https://arxiv.org/abs/2611.12943)

    在AI使形式化验证证明近乎零成本的时代，论文的核心价值在于“推导层内涵”（作者的问题选择与视角）与“方向性外延”（通过切点分解揭示证明薄弱环节），而非正确性本身。

    

    AI系统如今已能为著名的未解问题产出经过形式化验证的证明，而获得一个正确证明的成本正趋近于零。如果正确性变得廉价，研究论文还能传达什么？本文认为，一篇论文的价值并不仅限于其经过验证的结论。论文是一种暴露于学术共同体批评之下的叙事，由内涵（作者对问题和视角的选择）、科学思维链（通往结果的推理步骤）以及支撑每一步的论证构成。大多数已发表的数学成果正是依据这两个要素而非仅凭正确性来评判的，而机器生成的形式化脚本两者皆无法传达。借鉴逻辑与类型理论，本文将前者形式化为“推导层内涵”，将后者形式化为“方向性外延”：即把证明分解为若干切点，其公理足迹可以揭示证明的薄弱环节以及结果可被加强之处。

    arXiv:2611.12943v1 Announce Type: cross  Abstract: AI systems now produce formally verified proofs of celebrated open problems, and the cost of obtaining a correct proof is falling toward zero. If correctness becomes cheap, what is left for a research paper to communicate? I argue that a paper is not exhausted by a verified statement. It is a narrative exposed to the criticism of a community, made of an intension (the author's choice of problem and perspective), a scientific chain of thought (the reasoning steps leading to the result), and the arguments supporting each step. Most published mathematics is judged on these two elements rather than on correctness alone, and machine-generated formal scripts convey neither. Borrowing from logic and type theory, I formalise the first as a Derivation-Level Intension and the second as a Directional Extension: a decomposition of a proof into cut-points whose axiomatic footprints reveal its weak links, where the result could be strengthened, and 
    
[^230]: 表格基础模型与调优梯度提升树在电信客户流失预测中的对比：基于利润与校准感知的评估

    Tabular Foundation Models vs Tuned Gradient Boosting for Telecom Churn: A Profit- and Calibration-Aware Evaluation

    [https://arxiv.org/abs/2611.12942](https://arxiv.org/abs/2611.12942)

    该研究以期望最大利润和概率校准为评判标准，将表格基础模型与经Optuna调优的梯度提升树在电信客户流失预测上进行严格重新评估，发现TabPFN-3.5在部分数据集上能显著带来更高利润，修正了以往仅用排序指标、与未调优基线比较所得出的结论。

    

    表格基础模型（TFM）能够在单次前向传播中基于有标注的上下文完成预测。近期一项基准测试报告称，在客户流失预测任务上，表格基础模型决定性地优于树集成模型，但该测试仅将其与未调优的梯度提升树（GBDT）进行比较，且只采用排序类指标评分，而留存营销活动的实际评判标准是金钱收益。我们在五个公开的电信客户流失数据集上、通过10个随机种子划分，将TabPFN-2.5、TabPFN-3.5、TabPFN-3.5-Fast和TabICL v2与经Optuna调优的GBDT进行了重新评估。我们区分了三个问题：模型是否能很好地对流失客户进行排序、该排序是否能带来更高的期望最大利润（EMP）、以及其概率输出能否支撑阈值策略。通过划分感知的检验方法并对20项预先声明的比较进行Holm校正，TabPFN-3.5在Cell2Cell（1.9%）和Iranian（7.6%）数据集上获得的EMP显著高于最优的调优GBDT，且这两项优势在与三个调优GBDT分别单独比较时依然成立。在Orange i……（摘要在此处截断）

    arXiv:2611.12942v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) predict from a labelled context in a single forward pass. A recent benchmark reported that they decisively outperform tree ensembles for customer churn, but it compared them with untuned gradient-boosted trees (GBDTs) and scored only ranking metrics, while a retention campaign is judged in money. We re-evaluate TabPFN-2.5, TabPFN-3.5, TabPFN-3.5-Fast and TabICL v2 against Optuna-tuned GBDTs on five public telecom churn datasets over 10 seeded splits. We separate three questions: whether a model ranks churners well, whether that ranking earns more expected maximum profit (EMP), and whether its probabilities can drive a threshold policy. With a split-aware test and Holm correction over 20 pre-declared comparisons, TabPFN-3.5 earns significantly more EMP than the best tuned GBDT on Cell2Cell (1.9%) and Iranian (7.6%), and these two gains also hold against each of the three tuned GBDTs separately. On Orange i
    
[^231]: AI治理应优先关注控制与知识边界，而非限制智能本身

    AI Governance Should Prioritize Control and Knowledge Boundaries Over Limiting Intelligence

    [https://arxiv.org/abs/2611.12941](https://arxiv.org/abs/2611.12941)

    本文提出“能力回路”治理框架，将AI的现实影响分解为智能发现、程序性技能与直接控制三条路径，并通过形式化分析与随机模拟论证AI治理应优先设置控制与知识边界，而非笼统限制智能。

    

    自主行动的AI代理迅速普及，使得对智能体系统的治理变得日益紧迫。我们提出了“能力回路”框架，概括为IKC + KC + C，将智能体在现实世界中的影响分解为三条路径：智能介导的发现、程序性技能以及直接控制。以经典的“快速起飞”争论作为压力测试，我们将智能介导的快速起飞条件形式化为一个合取命题，该命题共同足以导致“快速灾难性霸权”，并表明其中大多数条件依赖于快速的知识获取或控制能力的达成，而非单凭智能本身。我们将认知智能操作化为“假设选择效率”，并通过在NK适应度景观上、覆盖该合取链的六个场景的随机模拟验证了这一框架。得到模拟支持的解析上界表明，智能驱动的加速受制于竞争性智能比值的上限。

    arXiv:2611.12941v1 Announce Type: new  Abstract: The rapid proliferation of autonomously acting AI agents makes the governance of agentic systems urgent. We propose the competence circuit, summarized as IKC + KC + C, that decomposes an agent's real-world impact into three pathways: intelligence-mediated discovery, procedural know-how, and direct control. Using the canonical fast-takeoff debate as a stress test, we formalize the conditions for intelligence-mediated fast takeoff as a conjunction jointly sufficient for fast catastrophic supremacy, showing that most rely on rapid knowledge acquisition or control attainment rather than intelligence alone. We operationalize epistemic intelligence as hypothesis-selection efficiency and validate this framework via stochastic simulation on NK fitness landscapes across six scenarios spanning the conjunctive chain. An analytic bound supported by the simulations shows that intelligence-driven speedup is capped by the ratio of competing intelligenc
    
[^232]: 基于强化学习的癌症化疗治疗控制

    Reinforcement Learning based control of Cancer Chemotherapy treatments

    [https://arxiv.org/abs/2611.12940](https://arxiv.org/abs/2611.12940)

    该论文系统梳理了强化学习技术在癌症化疗治疗控制中的应用现状，重点研究了用于生成最优动态治疗方案的最新强化学习方法。

    

    近年来，强化学习取得了巨大的成功。它从Atari、围棋、宝可梦和国际象棋等传统复杂游戏扩展了应用范围，引起了研究人员对建模现实生活场景的关注。个性化医疗和动态治疗方案正是当前研究人员关注的两个初步研究领域。从观察性真实临床数据中估计最优动态治疗方案正日益受到研究人员青睐，这也拓宽了其向更先进的强化学习算法版本发展的空间。本研究是一项专注于癌症化疗治疗背景下强化学习技术的研究，本论文主要通过强化学习技术在生成最优治疗方案方面的最新研究进展来呈现该研究的成果。

    arXiv:2611.12940v1 Announce Type: new  Abstract: Reinforcement Learning has achieved tremendous success in recent years. Extending its scope from the traditional complex games such as Atari, Go, Pokeman, and Chess, reinforcement learning has caught researchers' attention on modelling real-life scenarios. Personalized medicine and the dynamic treatment regimen are two such preliminary research fields; researchers are focusing their mindsets currently. Estimating the optimal dynamic treatment regimens from the observational real clinical data is being popularized among the researchers, widening its scope towards more advanced versions of reinforcement learning algorithms.   This research is about a research study that focused on reinforcement learning techniques that have been approached in the context of cancer chemotherapy treatments. Primarily this thesis addresses the result of the study by means of the state-of-the-art on reinforcement learning techniques used in generating the opti
    
[^233]: 面向知识图谱问答的提示图可视化方法

    Graph Visualizations in Prompts for Knowledge Graph Question Answering

    [https://arxiv.org/abs/2611.12938](https://arxiv.org/abs/2611.12938)

    该研究发现向提示中添加图可视化可使知识图谱问答平均提升1.85个F1点，且收益对图布局方式高度敏感，其中将大型子图分解为分离路径的布局带来的改进最大。

    

    我们研究了多模态输入何时以及如何改善大型视觉-语言模型中的知识图谱问答（KGQA）。通过对1,500个WebQSP和CWQ问题以及六个视觉语言模型在固定KGQA流程中进行受控评估，我们在逐问题层面比较了纯文本提示与文本加图可视化，并分析了性能变化与推理复杂度、模型规模和图像布局之间的关系。添加图图像平均带来1.85个F1点的提升，这主要由低复杂度问题上显著的精确率改进所驱动（+4.65 F1）。混合效应和回归分析表明，这些多模态收益并不强烈依赖于模型规模，但对图布局高度敏感：仅图像设置会急剧降低性能，层次化的Graphviz布局优于组织性较差的渲染方式，而将大型子图分解为分离的路径则产生最大的改进。

    arXiv:2611.12938v1 Announce Type: new  Abstract: We study when and how multimodal input improves knowledge graph question answering (KGQA) in large vision--language models. Using a controlled evaluation over 1,500 WebQSP and CWQ questions and six VLMs within a fixed KGQA pipeline, we compare text-only prompting against text-plus-graph visualization at the per-question level and analyze performance changes as a function of reasoning complexity, model size, and image layout. Adding a graph image yields an average gain of 1.85 F1 points, driven by strong precision improvements on low-complexity questions (+4.65 F1). Mixed-effects and regression analyses show that these multimodal gains do not depend strongly on model scale, but are highly sensitive to graph layout: image-only settings sharply degrade performance, hierarchical Graphviz layouts outperform less organized renderings, and decomposing large subgraphs into separated paths yields the largest improvements. Taken together, the resu
    
[^234]: 追踪大语言模型中的心理测量推断

    Tracing Psychometric Inference in Large Language Models

    [https://arxiv.org/abs/2611.12937](https://arxiv.org/abs/2611.12937)

    该研究通过跨角色预测范式发现，大语言模型（尤其是较大规模的指令微调模型）在行为上重现并放大了人类心理量表的跨量表相关性，且其内部几何结构与人类心理测量结构对齐，表明模型在一定程度上确实内部表征了心理构念。

    

    大语言模型是在内部真正表征了心理构念，还是仅仅生成表面上的心理测量结构？我们通过跨角色范式回答这一问题，研究对象为来自两个家族（Llama 3 和 Qwen 2.5，参数量 0.5B-14B，包含基础版和指令版）的 14 个模型变体。给定真实个体（N = 272 名台湾大学生）在一个心理量表上的回答，模型需要预测其余六个量表上的回答。在行为层面，更大的模型变体能够重现并放大人类的跨量表相关性；即使在校正了七个人类量表的测量衰减之后，这种放大效应依然存在。随后我们检验这种结构是否在模型内部得到反映。对比方向分析发现，在较大的指令版变体中，模型内部的几何结构与人类的心理测量矩阵相一致，而基础版变体在该测试下几乎没有表现出这种对齐。跨模型来看，几何结构强度与行为放大程度相关。

    arXiv:2611.12937v1 Announce Type: new  Abstract: Do large language models represent psychological constructs internally, or do they only generate outputs that resemble psychometric structure? We address this question using a cross-persona paradigm across 14 model variants from two families (Llama 3 and Qwen 2.5, 0.5B-14B, base and instruct). Given responses on one psychological scale from real individuals (N = 272 Taiwanese university students), models predict responses on six other scales. At the behavioral level, larger variants reproduce and amplify human cross-scale correlations; amplification persists after correcting the seven-scale human correlations for measurement attenuation. We then examine whether this structure is reflected internally. Contrastive direction analysis finds geometry aligned with the human matrix in larger instruct variants, but little alignment in base variants under this assay. Across models, geometry strength is associated with behavioral amplification (r 
    
[^235]: LandslideFM：面向滑坡统一感知与理解的基础模型

    LandslideFM: A Foundation Model Towards Unified Landslide Perception and Understanding

    [https://arxiv.org/abs/2611.12936](https://arxiv.org/abs/2611.12936)

    提出了首个统一的滑坡基础模型LandslideFM，通过半监督学习框架利用有限标注数据完成多种视觉感知任务，并结合参数高效微调的视觉-语言模型实现滑坡的情境理解与推理，同时发布了首个智能滑坡分析基准数据集LandslideVL。

    

    智能滑坡检测对灾害防治至关重要。然而，现有模型主要聚焦于孤立的视觉任务，需要大量标注数据，且缺乏深层分析与场景理解能力。为解决这些局限，我们提出了LandslideFM，这是首个将视觉感知任务与滑坡情境解读相连接的统一基础模型。具体而言，LandslideFM通过一个“源一致性渐进信任”半监督学习框架适应多种视觉任务，该框架能够利用有限的标注数据与大量无标注数据。此外，预测出的滑坡区域为通过参数高效微调适配的视觉-语言模型提供显式空间引导，从而实现全面的滑坡推理能力。为支持该领域的训练与评估，我们还提出了LandslideVL，这是首个智能滑坡分析基准数据集，结合了视觉（原文摘要在此处截断）。

    arXiv:2611.12936v1 Announce Type: new  Abstract: Intelligent landslide detection is essential for disaster prevention. However, existing models primarily focus on isolated visual tasks, require extensive labeled data, and lack deep analytical and scene-understanding capabilities. To address these limitations, we propose LandslideFM, the first unified foundation model that connects visual perception tasks with contextual interpretation of landslides. Specifically, LandslideFM adapts to multiple visual tasks via a Source-Agreement Graduated Trust semi-supervised learning framework that leverages limited labeled data alongside abundant unlabeled data. Furthermore, predicted landslide regions provide explicit spatial guidance to a Vision-Language Model adapted through parameter-efficient fine-tuning, enabling comprehensive landslide reasoning. To support training and evaluation in this domain, we also present LandslideVL, the first benchmark for intelligent landslide analysis, combining vi
    
[^236]: 在预填充阶段训练，在解码阶段监控：使用稀疏自编码器进行主题监控

    Train on prefill, monitor at decode: Using sparse autoencoders for topic monitoring

    [https://arxiv.org/abs/2611.12935](https://arxiv.org/abs/2611.12935)

    提出了TopicScope框架，利用稀疏自编码器（SAE）特征训练因果主题分类器，在预填充阶段提取特征训练后可泛化到解码阶段，实现对生成过程中每个token的实时多主题监控且标签可中途变化。

    

    主题监控是在模型生成响应期间为响应中的每个token标记主题的任务。迄今为止，在生成时运行的基于激活的监控器主要用于检测有害内容，且仅限于两三个类别。我们提出了TopicScope，这是一个在稀疏自编码器（SAE）特征上训练因果主题分类器的框架，它能为每个token分配主题标签，使标签能够在响应中途发生变化。我们在涵盖七个主题的合成多主题对话上训练了三种检测器架构。在Gemma-3 1B和4B模型上，使用从预填充（prefill）隐藏状态中提取的特征训练的监控器能够泛化到解码阶段。我们还发现这些分类器对扰动具有鲁棒性：当插入离题的旁白、删除主题或重新排列章节时，macro-F1最多损失5分。为理解SAE的贡献而进行的消融实验表明，达到基线的分类器……（摘要原文在此处被截断）

    arXiv:2611.12935v1 Announce Type: new  Abstract: Topic monitoring is the task of labelling the topic of each token in a model's response during generation. Activation-based monitors that run at generation time have so far been trained to detect harmful content and are limited to two or three classes. We introduce TopicScope, a framework for training causal topic classifiers on SAE features that assign a topic to every token, so that the label can change partway through a response. We train three detector architectures on synthetic multi-topic conversations covering seven topics. Across Gemma-3 1B and 4B, monitors trained on features extracted from prefill hidden states were able to generalise to decoding. We also found that the classifiers are robust to perturbations, losing at most 5 macro-F1 points when off-topic asides are inserted, topics are removed or sections are reordered. Ablations conducted to understand the contribution of the SAE revealed that classifiers reaching baseline 
    
[^237]: EarthRecord：利用多模态智能体从四十年卫星影像的像素解析土地变化

    EarthRecord: From Pixels to Land Change with Multimodal Agents over Four Decades of Satellite Imagery

    [https://arxiv.org/abs/2611.12934](https://arxiv.org/abs/2611.12934)

    提出EarthRecord基准，基于56个站点四十年的原始多传感器卫星影像，要求多模态智能体自主完成云掩膜和跨传感器协调，通过238个土地变化问题及对照实验评估其真实土地变化分析能力。

    

    尽管多模态智能体现在能够借助工具执行复杂的数据分析，但它们将数十年的卫星影像转化为可靠的土地变化分析的能力——这对地球观测科学至关重要——仍未被衡量。现有基准存在不足：它们在精选影像上提出问题，或为智能体提供预处理的输入，遗漏了此类分析起步所需的原始多传感器档案。我们提出EarthRecord，一个基于56个站点原始记录构建的基准：涵盖四十年的Landsat、MODIS、VIIRS和Sentinel-2数据，云掩膜和跨传感器协调工作留给智能体自行完成。该基准提出238个横跨十项分析任务的土地变化问题，依据保留的全球产品参考、方法间一致性以及视觉解译进行评分。两组对照实验用于检验答案的依据：闭卷实验移除数据记录，仪器对实验则改变一颗卫星的数据而地面实际并无变化。对十二个

    arXiv:2611.12934v1 Announce Type: new  Abstract: While multimodal agents now carry out complex data analysis with tools, their ability to turn decades of satellite imagery into reliable land-change analysis, crucial for Earth observation science, remains unmeasured. Existing benchmarks fall short: they pose questions on curated images or give agents preprocessed inputs, leaving out the raw multi-sensor archives where such analysis begins. We introduce EarthRecord, a benchmark built on the raw record of 56 sites: four decades of Landsat, MODIS, VIIRS, and Sentinel-2 data, with cloud masking and cross-sensor harmonization left to the agent. It poses 238 land-change questions across ten analysis tasks, scored against withheld references from global products, agreement among methods, and visual interpretation. Two controls test what answers rest on: the closed book removes the record, and instrument pairs alter one satellite's data while nothing changes on the ground. Evaluations of twelve
    
[^238]: 面向自主X射线实验的可审计人-智能体团队

    Auditable Human-Agent Teams for Autonomous X-ray Experimentation

    [https://arxiv.org/abs/2611.12933](https://arxiv.org/abs/2611.12933)

    该研究利用可审计的人-智能体协作团队在真实同步辐射装置上实现了纳米粒子超晶格的自主X射线实验，并通过完整对话记录审计决策溯源与认知分工，证明智能体系统能够在物理仪器上持续运行并逐步承担更多科学贡献。

    

    闭环自主实验能够加速科学发现，但实施难度限制了其可及性。智能体系统有望降低这一门槛，但其在物理仪器上的持续运行仍在很大程度上未被探索。本文利用基于经过验证的科学方法的编码智能体基础设施，在不断演变的人类定义目标下，实现了纳米粒子超晶格上的自主同步辐射X射线实验。完整的人-智能体及智能体-智能体对话记录使审计成为可能，可重建决策溯源并刻画认知劳动的分工。在三项实验中，61-67%的消息涉及解释或判断，而方法论与科学贡献的占比从以装置搭建为主的第一项实验中的46%提升至后续实验中的56-61%。溯源分析表明，59-68%的错误纠正涉及多名参与者，且约有一半的

    arXiv:2611.12933v1 Announce Type: new  Abstract: Closed-loop autonomous experimentation can accelerate scientific discovery, but implementation effort limits accessibility. Agentic systems may lower this barrier, although sustained operation on physical instruments remains largely unexplored. Here, using coding-agent infrastructure grounded in validated scientific methods, we demonstrate autonomous synchrotron X-ray experiments on nanoparticle superlattices under evolving, human-defined objectives. Complete human-agent and agent-agent transcripts enable an audit that reconstructs decision provenance and characterizes the division of cognitive labor. Across three experiments, 61-67% of messages involved interpretation or judgment, while the share of methodological and scientific contributions increased from 46% in the setup-focused first experiment to 56-61% in later experiments. Provenance showed that 59-68% of error corrections involved multiple participants and that about half of all
    
[^239]: 基于令牌重要性感知语义通信的鲁棒视频传输

    Token Importance-Aware Semantic Communication Enabled Robust Video Transmission

    [https://arxiv.org/abs/2611.12928](https://arxiv.org/abs/2611.12928)

    该论文提出令牌重要性感知的语义通信框架Ti-SemCom，通过联合提取目标与运动信息、按重要性分层关联语义令牌、结合带宽动态选择传输令牌并在接收端复用参考内容，在保证视频传输鲁棒性的同时将通信开销降低88.3%。

    

    视频应用日益普及，但其高数据量和时间依赖性使得在有限通信资源下实现鲁棒的无线传输颇具挑战。为解决这一问题，我们提出了一种面向带宽高效视频传输的令牌重要性感知语义通信框架。具体而言，该方法联合提取目标与运动信息以识别与通信相关的区域，并根据重要性将这些区域与语义令牌进行分层关联；随后基于令牌重要性和可用带宽动态选择待传输的令牌，在资源受限条件下优先传输信息量更大的视频内容。在接收端，未被传输的区域复用最近重建的参考内容，从而进一步减少冗余传输。在CAVIAR数据集上的实验表明，Ti-SemCom能够实现88.3%的通信开销降低（原文摘要在此处截断）。

    arXiv:2611.12928v1 Announce Type: cross  Abstract: Video applications are increasingly prevalent, yet their high data volume and temporal dependency make robust wireless transmission challenging under limited communication resources. To address this issue, we propose a token importance-aware semantic communication (Ti-SemCom) framework for bandwidth-efficient video transmission. Specifically, object and motion information is jointly extracted to identify communication-relevant regions, which are hierarchically associated with semantic tokens according to their importance. The transmitted tokens are then dynamically selected based on both token importance and the available bandwidth, prioritizing more informative video content under resource constraints. At the receiver, untransmitted regions reuse recently reconstructed references to further reduce redundant transmission. Experiments on the CAVIAR dataset show that Ti-SemCom can achieve an 88.3\% reduction in communication cost while m
    
[^240]: 莫比乌斯交互过滤：面向多模态幽默检测的阶数选择性谱正则化

    M\"obius Interaction Filtering: Order-Selective Spectral Regularization for Multimodal Humor Detection

    [https://arxiv.org/abs/2611.12913](https://arxiv.org/abs/2611.12913)

    提出莫比乌斯交互过滤方法，通过阶数选择性谱正则化在保留高阶模态交互表示的同时调控其对预测的贡献，使多模态幽默检测的标签对齐信噪比提升1.62-2.79倍。

    

    多模态幽默检测需要联合解读语言、声音表达和视觉行为，同时表达性融合还必须调控不同阶数的交互以何种程度进入最终判断。我们提出莫比乌斯交互过滤，它学习一个模态子集预测函数，并通过其聚合结构施加阶数选择性的谱正则化。私有时间编码器和平衡的双向交互构建了有监督的单模态、双模态和三模态预测器，它们的均匀聚合恰好构成一个具有几何递减交互阶响应的莫比乌斯域滤波器。这一构造在保留高阶表示的同时调控其对预测的贡献。在UR-FUNNY数据集上的受控扰动实验显示，各交互阶数的标签对齐信噪比出现显著下降。所诱导的滤波器使最终比率相对……提升了1.62-2.79倍。

    arXiv:2611.12913v1 Announce Type: cross  Abstract: Multimodal humor detection requires interpreting language, vocal delivery, and visual behavior jointly. Expressive fusion must also regulate how interactions of different orders enter the final judgment. We propose M\"obius Interaction Filtering (MoIF), which learns a modality-subset prediction function and imposes order-selective spectral regularization through its aggregation structure. Private temporal encoders and balanced reciprocal interaction construct supervised unimodal, bimodal, and trimodal predictors. Their uniform aggregate is exactly a M\"obius-domain filter with a geometrically decreasing interaction-order response. This formulation retains higher-order representations while regulating their contribution to prediction. Controlled UR-FUNNY perturbations reveal a pronounced decline in label-aligned signal-to-noise ratio across interaction orders. The induced filter improves the final ratio by 1.62-2.79 times relative to th
    
[^241]: Brasilidade：衡量巴西身份认同的大语言模型评测基准

    Brasilidade: Measuring Brazilian Identity for Large Language Models

    [https://arxiv.org/abs/2611.12906](https://arxiv.org/abs/2611.12906)

    该论文提出了Brasilidade基准，通过涵盖语言、语法等四个领域的标准化任务评估大语言模型的巴西葡萄牙语能力，发现模型规模并不决定性能，且葡萄牙语专用模型的表现也未超过平均水平。

    

    这项工作提出了一个新的基准，旨在解决当前评估大语言模型巴西葡萄牙语能力的方法的局限性。我们介绍了Brasilidade，这是一种由分布在四个不同领域的标准化任务组成的评估方法，我们正在发布其第一个领域的任务，这些任务涉及巴西葡萄牙语的语言细微差别、语法和语言变体。结果表明，模型规模本身并不能决定性能；在所评估的测试协议中，几个中等规模的模型表现优于更大的模型。针对葡萄牙语的专用模型总体而言表现并未超出平均水平，尽管它们经过了针对性训练。不同的任务难度水平揭示了显著的细微差别：虽然有些任务对大多数模型来说很容易，但另一些任务则构成了极大的挑战，还有一些任务结果参差不齐，性能差异很大。这清楚地说明了需要能够应对巴西葡萄牙语独特挑战的基准。

    arXiv:2611.12906v1 Announce Type: cross  Abstract: This work presents a new benchmark that aims to address the limitations of current methods for evaluating LLMs in Brazilian Portuguese. We introduced Brasilidade, an assessment methodology comprised of standardized tasks distributed across four different domains, and we are making available the tasks from its first domain, which address nuances of the language, grammar, and variations of Brazilian Portuguese. The results show that model size, by itself, does not determine performance; several medium-sized models outperformed larger models in the protocols evaluated. Specific models for Portuguese, in general, did not perform above average, despite their targeted training. Different task levels revealed significant nuances: while some tasks proved easy for most models, others posed extreme challenges, and some yielded mixed results, with wide variations in performance. This clearly illustrates the need for benchmarks that address the li
    
[^242]: 可操作的反馈：利用临床文档中用户编辑数据提升模型能力

    Actionable Feedback: Improving Model Capabilities from User Edit Data on Clinical Documentation

    [https://arxiv.org/abs/2611.12904](https://arxiv.org/abs/2611.12904)

    该论文针对临床文档场景中用户编辑数据并非全部可操作的问题（如45%以上编辑包含模型输入不支持的信息），构建了临床医生编辑行为分类体系并开发了区分可操作反馈的方法，以避免微调导致模型产生幻觉。

    

    大语言模型越来越多地被部署到临床文档工作流程中，临床医生在使用前会审查并编辑模型的输出。原则上，这些临床医生的编辑代表了模型训练的一个可扩展的监督来源。然而在实践中，并非所有编辑都构成可操作的反馈：例如，通过审查来自AI临床文档产品实际部署的数据，我们发现超过45%的编辑后输出包含用户添加的、模型输入并不直接支持的信息。如果在这种编辑上进行简单微调，会有教会模型产生幻觉的风险。在本文中，我们开发了一种方法来应对这一挑战。鉴于已部署文档工具上的真实临床医生编辑数据并不公开，我们首先基于来自已部署AI临床文档产品的数据，构建了一个临床医生编辑行为的分类体系。然后，我们开发了一种生成……的方法

    arXiv:2611.12904v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly being deployed in clinical documentation workflows, where clinicians review and edit their outputs before use. In principle, these clinician edits represent a scalable source of supervision for model training. In practice, not all edits represent actionable feedback: for instance, on data drawn from real-world deployment of an AI clinical documentation product, our review suggests that over 45% of edited outputs contain user-added information that is not directly supported by the inputs to the model. Naively fine-tuning on such edits risks teaching the model to hallucinate. In this paper, we develop a method to address this challenge. Given that real clinician edits on deployed documentation tools are not publicly available, we first develop a taxonomy of clinician-editing behaviors, informed by data from a deployed AI clinical documentation product. Then, we develop a method for generating
    
[^243]: SELD：超越答案层面检测上下文压缩中的隐性证据丢失

    SELD: Detecting Silent Evidence Loss in Context Compression beyond the Answer

    [https://arxiv.org/abs/2611.12903](https://arxiv.org/abs/2611.12903)

    提出SELD——一种无需训练和标注的方法，通过比较阅读模型压缩前后对未说出口候选答案的偏好变化，检测上下文压缩中“答案不变但关键证据被删”的隐性证据丢失问题，并构建了SEL-Pairs评测基准。

    

    对于处理长输入的语言模型来说，上下文压缩如今已是常规操作，但压缩后的上下文只有在保留了回答所需的证据时才有用。现有的检测方法仅比较阅读模型的答案或其最可能的词汇，因而遗漏了一种常见的失败模式——隐性证据丢失：即关键证据被删除而答案保持不变，导致答案缺乏支撑，并对复用该上下文的其他模型构成风险。为解决这一挑战，我们提出了SELD（隐性证据丢失检测器），它无需训练也无需标注。其原理是：当证据丢失时，阅读模型在那些它未说出口的候选答案之间的偏好会发生偏移，即使其给出的答案不变，因此通过比较压缩前后这些偏好，能够发现答案检查无法察觉的证据丢失。我们还构建了SEL-Pairs基准，包含来自三个问答数据集的长度匹配的证据删除与对照删除样本对。定量与定性实验……（原文截断）

    arXiv:2611.12903v1 Announce Type: cross  Abstract: Context compression is now routine for language models with long inputs, but a compressed context is useful only if it keeps the evidence for the answer. Existing checks compare the reader model's answer or its most likely words, and thus miss a frequent failure, silent evidence loss: key evidence is removed while the answer stays the same, leaving the answer unsupported and risky for other models that reuse the context. To address this challenge, we propose SELD (Silent Evidence Loss Detector), which needs no training and no labels. Its principle is that when evidence is lost, a reader model's preferences among the answers it leaves unsaid shift even if its answer does not, so comparing these preferences before and after compression reveals losses that answer checks cannot see. We also build SEL-Pairs, a benchmark of length-matched evidence and control deletions from three question-answering datasets. Quantitative and qualitative expe
    
[^244]: 反思尖峰：通过步感知解码缓解多模态扩散语言模型中的幻觉

    Rethinking in Spikes: Mitigating Hallucinations in MDLMs with Step-Aware Decoding

    [https://arxiv.org/abs/2611.12901](https://arxiv.org/abs/2611.12901)

    提出了StepRefine，一种即插即用的步感知解码策略，通过检测去噪过程中的熵尖峰并在离散去噪与潜在精炼之间动态切换，从而缓解多模态扩散语言模型的幻觉问题。

    

    多模态扩散语言模型（MDLMs）的最新进展通过迭代精炼展现出了强大的全局建模能力。我们观察到，掩码不确定性在迭代去噪过程中呈现总体下降趋势，而少数去噪步骤表现出与幻觉密切相关的局部熵尖峰。我们认为，可以通过测量每个去噪步骤的熵与其相邻时间窗口的偏差来识别异常步骤。我们假设离散词元的承诺机制会丢弃掩码位置上的分布信息，从而限制了候选语义与上下文在熵尖峰期间的持续交互。为此，我们提出了StepRefine，一种利用语义上下文实现可靠去噪的即插即用解码策略。我们方法的核心在于在离散去噪与潜在精炼之间进行步感知切换。该模型采用……

    arXiv:2611.12901v1 Announce Type: cross  Abstract: Recent advancements in multimodal diffusion language models (MDLMs) have exhibited strong global modeling capabilities through iterative refinement. We observe that mask uncertainty follows an overall downward trend during iterative denoising, while a few denoising steps exhibit localized entropy spikes closely associated with hallucinations. We argue that anomalous steps can be identified by measuring the entropy deviation of each denoising step from its neighboring temporal window. We hypothesize that discrete token commitment discards the distributional information at masked positions, limiting the continued interaction between candidate semantics and the context during entropy spikes. With this goal, we present StepRefine, a plug-and-play decoding strategy that leverages semantic context to achieve reliable denoising. The core of our method lies in a step-aware switch between discrete denoising and latent refinement. The model empl
    
[^245]: AFNO：面向欧式看涨期权价格曲面的通用可证明无套利算子学习框架

    AFNO: Universal Certifiably-Arbitrage-Free Operator Learning Framework for European Call-Price Surfaces

    [https://arxiv.org/abs/2611.12899](https://arxiv.org/abs/2611.12899)

    该论文提出了AFNO——一个可证明无套利且通用的神经算子学习框架，通过连续求解近似最近邻问题将输入映射为无套利欧式看涨期权价格曲面的凸组合，解决了标准神经算子在期权定价中无法保证无套利并可能导致负盈亏的问题。

    

    神经算子（NO）作为深度学习驱动的解决方案被提出，用于解决求解问题族时的计算瓶颈。经过训练后，神经算子能够快速输出近似解，避免了重新运行经典求解器的需要。然而，当根据市场或模型特征预测欧式看涨期权价格曲面时，标准神经算子架构无法保证静态无套利，并且正如我们通过实验所验证的，其在下游交易任务中可能产生负的盈亏（PnL）。我们提出了一种可证明无套利（AF）且通用的编码器-解码器神经算子框架，通过连续求解一个近似最近邻问题，将定义在再生核希尔伯特空间（RKHS）$\mathcal{H}$ 上（其定义域为 $[0,\infty)^2$ 的任意固定子集）的输入，连续地映射为无套利欧式看涨期权价格曲面的凸组合。我们证明了对于无约束预测的曲面，其最近的无套利修正未必存在……

    arXiv:2611.12899v1 Announce Type: cross  Abstract: Neural operators (NOs) have been introduced as deep learning-powered solutions to the computational bottleneck of solving families of problems. Once trained, an NO can rapidly output approximate solutions, avoiding the need to rerun classical solvers. However, standard NO architectures cannot guarantee static no-arbitrage when predicting European call-price surfaces from market or model features and, as we verify experimentally, can yield negative PnL in downstream trading tasks.   We introduce a certifiably arbitrage-free (AF) and universal encoder-decoder NO framework, continuously mapping inputs from a reproducing kernel Hilbert space (RKHS) $\mathcal{H}$ over any fixed subset of $[0,\infty)^2$ to a convex combination of arbitrage-free European call-price surfaces by continuously solving an approximate nearest-neighbour problem. We show that a nearest arbitrage-free correction of an unconstrained predicted surface need not exist, im
    
[^246]: 面向文本到图像扩散模型推理时对齐的直接轨迹优化

    Direct Trajectory Optimization for Inference-Time Alignment of Text-to-Image Diffusion Models

    [https://arxiv.org/abs/2611.11821](https://arxiv.org/abs/2611.11821)

    提出无需训练的 DTO 框架，通过将中间去噪分数和采样噪声作为优化变量直接优化扩散模型的采样轨迹，实现文本到图像模型的推理时奖励对齐，避免了通过去噪网络的高成本反向传播，并提供奖励提升的理论保证。

    

    文本到图像扩散模型与奖励函数的对齐是可控生成中的一项关键挑战。现有方法通常依赖于优化模型输入或微调去噪网络，这需要通过大型去噪模型进行计算代价高昂的反向传播。在本工作中，我们提出了直接轨迹优化（Direct Trajectory Optimization, DTO），这是一个无需训练的框架，通过直接优化扩散模型的采样轨迹来实现对齐。具体而言，DTO 将中间去噪分数和采样噪声作为优化变量，从而免除了通过去噪网络计算梯度的需要。为了缓解分布外问题，我们引入了奖励优化分数引导，它在优化后的分数与预训练模型预测之间进行插值以稳定采样，同时在受约束的分布条件下提供了可证明的奖励提升理论保证。

    arXiv:2611.11821v2 Announce Type: replace-cross  Abstract: Alignment of text-to-image diffusion models with reward functions is a key challenge in controllable generation. Existing approaches typically rely on optimizing model inputs or fine-tuning denoising networks, which requires computationally expensive backpropagation through large denoising models. In this work, we propose Direct Trajectory Optimization (DTO), a training-free framework that performs alignment by directly optimizing the sampling trajectory of diffusion models. Specifically, DTO treats intermediate denoising scores and sampled noise as the optimization variables, eliminating the need for gradients through the denoising network. To mitigate the out-of-distribution problem, we introduce Reward-Optimized Score Guidance, which interpolates between the optimized scores and the pretrained model predictions to stabilize sampling, while providing a provable theoretical guarantee of reward improvement under constrained dis
    
[^247]: FinCutBench：一个用于衡量大语言模型金融知识时间边界的持续更新基准

    FinCutBench: A Continually Updated Benchmark for Measuring the Temporal Boundaries of LLM Financial Knowledge

    [https://arxiv.org/abs/2611.11588](https://arxiv.org/abs/2611.11588)

    该论文提出FinCutBench，一个每月持续更新的中英双语金融知识基准，通过按月构建知识画像并定位“FinCut边界”，来衡量大语言模型金融知识的时间截止点，以检测金融回测中的参数化前视偏差。

    

    在金融回测和其他时点任务中，外部输入可以按日期进行限制，但模型参数可能保留了之后才披露的金融事实，从而引入参数化前视偏差。我们提出了FinCutBench，这是一个每月持续更新的金融知识基准，用于衡量固定模型的金融知识在日历时间上能够延伸到多远。当前数据集包含9,000道中英文四选一多项选择题，时间跨度从2023年1月至2026年9月，涵盖中国和美国市场以及四类金融信息。我们利用所测试事实公开的大致月份来构建月度知识画像。在画像通过衰减测试和节点检查后，我们将FinCut边界定义为拟合画像达到其较低水平的第一个月份，并报告重采样下估计日期的稳定性。边界日期与定位

    arXiv:2611.11588v2 Announce Type: replace  Abstract: In financial backtesting and other point-in-time tasks, external inputs can be restricted by date, but model parameters may retain financial facts disclosed later, introducing parametric look-ahead bias. We introduce FinCutBench, a continually updated monthly financial knowledge benchmark for measuring how far a fixed model's financial knowledge extends over calendar time. The current dataset contains 9,000 Chinese and English four-option multiple-choice questions from January 2023 to September 2026, covering Chinese and U.S. markets and four categories of financial information. We construct monthly knowledge profiles using the approximate months in which the tested facts became public. After a profile passes the decline test and knot checks, we define the FinCut Boundary as the first month when the fitted profile reaches its lower level, and report the stability of the estimated date under resampling. Boundary dates and localization
    
[^248]: 扩散语言模型何时应重新考虑其承诺？

    When Should Diffusion Language Models Reconsider Their Commitments?

    [https://arxiv.org/abs/2611.11551](https://arxiv.org/abs/2611.11551)

    该论文提出PURL框架，通过联合学习编辑选择、取消掩码选择和词元预测三个决策，使掩码扩散模型能够在并行快速生成的同时有选择地修正早期不一致的词元承诺，且推理时三个决策仅需一次共享前向传播即可完成。

    

    掩码扩散模型以通过并行解码实现快速文本生成而闻名。然而，相互独立的并行采样可能会在早期就“承诺”不一致的预测结果，而关于如何有选择地修正这些早期的词元承诺，目前所知甚少。在这项工作中，我们为掩码扩散模型提出了计划式取消掩码与修正联合学习（Planned Unmasking-and-Revision Joint Learning, PURL），这是一个基于训练的框架，无缝地将并行生成与选择性词元修正统一起来。我们推导出了一个归一化的联合转移，其似然可以分解为编辑选择、取消掩码选择和词元预测三个部分。这些组件在根据模型自身取消掩码策略构建的部分掩码序列上进行联合训练。在推理阶段，所有三个决策均由一次共享的前向传播所支持，使模型能够解码被掩码的词元并修正带有噪声的词元。

    arXiv:2611.11551v2 Announce Type: replace-cross  Abstract: Masked Diffusion Models (MDMs) are well known for fast text generation via parallel decoding. However, independent parallel sampling can commit to inconsistent predictions, and less is known about how to selectively revise these early token commitments. In this work, we introduce $\underline{\mathrm{P}}$lanned $\underline{\mathrm{U}}$nmasking-and-$\underline{\mathrm{R}}$evision Joint $\underline{\mathrm{L}}$earning (PURL) for MDMs, a training-based framework that seamlessly unifies parallel generation with selective token revision. We derive a normalized joint transition whose likelihood factorizes into edit selection, unmask selection, and token prediction. These components are trained jointly on partially masked sequences constructed according to the model's own unmasking policy. During inference, all three decisions are supported by a single shared forward pass, enabling the model to decode masked tokens and revise noisy tok
    
[^249]: ScrapeBench：评估AI智能体在网站抓取中的法律合规性

    ScrapeBench: Evaluating Legal Compliance of AI Agents in Website Scraping

    [https://arxiv.org/abs/2611.11306](https://arxiv.org/abs/2611.11306)

    该论文提出首个评估AI智能体网站抓取法律合规性的基准ScrapeBench，发现大多数智能体在抓取明确禁止抓取的网站时合规率低于5%，即使主动检查了ToS和robots.txt也依然如此，而在允许抓取的网站上却有时会错误地拒绝抓取。

    

    部署在开放网络上的AI智能体越来越有能力执行可能违反合同条款和计算机安全法律的行为。在本文中，我们介绍了ScrapeBench，这是首个用于实证评估AI智能体在网站抓取策略（即服务条款ToS和robots.txt指令）方面法律合规性的智能体基准测试。在广泛使用的闭源权重与开源权重AI智能体中，我们发现，当被指示抓取我们样本中1,626个明确禁止抓取的热门网站时，大多数智能体表现出极低的合规率（<5%）。此外，我们观察到，即使AI智能体主动检查了适用的ToS和robots.txt，它们仍然表现出较低的合规率。相反，在我们样本中269个不禁止抓取的网站上，我们观察到智能体有时会拒绝抓取，尽管这样做是合法的，例如Claude O（原文在此处截断）

    arXiv:2611.11306v2 Announce Type: replace-cross  Abstract: AI agents deployed on the open web are becoming more capable of performing actions that may breach contractual provisions and violate computer security laws. In this paper, we introduce ScrapeBench, the first agentic benchmark for empirically evaluating the legal compliance of AI agents with website scraping policies, namely terms of service (ToS) and $\texttt{robots.txt}$ directives. Across widely used closed-weight and open-weight AI agents, we find that most agents exhibit low compliance rates (< 5%) when instructed to scrape websites in our sample of 1,626 popular websites that explicitly prohibit scraping. In addition, we observe that even where AI agents actively check the applicable ToS and $\texttt{robots.txt}$, they continue to exhibit low compliance rates. Conversely, in our sample of 269 websites that do not prohibit scraping, we observe that agents sometimes refuse to scrape despite it being lawful, such as Claude O
    
[^250]: STABLE：一种具有自适应漂移控制的持续学习优化器

    STABLE: A Continual Learning Optimizer with Adaptive Drift Control

    [https://arxiv.org/abs/2611.10655](https://arxiv.org/abs/2611.10655)

    该论文提出STABLE，一种具有自适应漂移控制的原对偶优化器，理论上证明有界参数漂移可保证有界的间隔侵蚀，并通过动态调节对偶变量来强制执行漂移预算，从而从优化器层面缓解持续学习中的灾难性遗忘。

    

    持续学习深受灾难性遗忘的困扰，这会导致过去知识的丢失以及在任务流上适应过程的不稳定。现有方法通常通过重放缓冲区、架构修改或额外的正则化项来解决这一问题。我们认为，遗忘是不稳定学习动力学的后果，因此可以从优化器层面的方案中受益。主要挑战在于学习动力学是随时间变化的，诸如“稳定性间隙”等经验现象正说明了这一点。首先，我们严格证明：对于基于距离的分类器，在局部分数稳定性条件下，有界的参数漂移可以保证有界的分类间隔侵蚀。基于这一洞察，我们提出了STABLE——一种具有自适应漂移控制的原对偶优化器。STABLE通过在超出漂移预算时增大对偶变量、反之减小时下对偶变量，来强制执行漂移预算。由此实现了高稳定性，同时（原文在此处截断）……

    arXiv:2611.10655v2 Announce Type: replace-cross  Abstract: Continual learning suffers from catastrophic forgetting, leading to the loss of past knowledge and unstable adaptation across task streams. Existing approaches typically address this issue through replay buffers, architectural changes, or additional regularization terms. We argue that forgetting is a consequence of unstable learning dynamics, which may benefit from an optimizer-level solution. The main challenge is that learning dynamics are time-dependent, as demonstrated by empirical phenomena such as the stability gap. First, we formally prove that, for distance-based classifiers, bounded parameter drift guarantees bounded margin erosion under a local score-stability condition. Based on this insight, we propose STABLE, a primal-dual optimizer with adaptive drift control. STABLE enforces a drift budget by increasing the dual variable when the budget is violated and decreasing it otherwise. This results in high stability witho
    
[^251]: 面向制造系统设计的智能体离散事件仿真：一个开源3D仿真器、一个基准测试和一个工业案例

    Agentic discrete-event simulation for manufacturing system design: an open 3D simulator, a benchmark and an industrial case

    [https://arxiv.org/abs/2611.10602](https://arxiv.org/abs/2611.10602)

    本文提出了SimuLean——一个基于Unity的开源3D离散事件仿真器，专为AI智能体编程设计，通过一次性校验的建模语义和领域级命令，使前沿AI智能体能以86%的成功率在约两分钟内自动构建有效的制造系统仿真模型。

    

    制造系统的设计依赖于离散事件仿真（DES）模型，而专家们目前仍在使用特定工具的界面来构建这些模型。AI智能体现在可以编写仿真代码，但它们从头编写的每个模型都是新的、未经校验的软件。我们提出了SimuLean，一个基于Unity引擎的开源3D离散事件仿真器，专为AI智能体编程而设计：其建模语义在平台内被一次性校验，智能体通过领域级命令在3D场景中或无头模式下构建和研究一个模型；物理仿真的时长经过校准，以支持快速的无头研究。我们在18个制造建模任务上对其进行了评估，范围从单台机器到已发表的生产线。一个前沿智能体在108次尝试中成功构建了93个有效模型（86%），每个模型约需两分钟；在基准测试的接口版本上，两个前沿智能体仅通过领域命令就构建了108个模型中的105个，因此每个模型都附有构建过程的记录

    arXiv:2611.10602v2 Announce Type: replace  Abstract: The design of manufacturing systems relies on discrete-event simulation (DES) models that experts still build in tool-specific interfaces. AI agents can now write simulation code, but every model they write from scratch is new, unverified software. We present SimuLean, an open-source 3D DES simulator on the Unity engine designed to be programmed by AI agents: its modelling semantics are verified once, in the platform, and the agent builds and studies one model through domain-level commands, in the 3D scene or headless; physically simulated durations are calibrated for fast headless studies. We evaluated it on 18 manufacturing modelling tasks, from single machines to published production lines. A frontier agent built valid models in 93 of 108 attempts (86%), in about two minutes each; on the benchmark's interface version, two frontier agents built 105 of 108 models through domain commands alone, so that each comes with a record of how
    
[^252]: 仅靠看见还不够：利用TIGER弥合推理访问差距以解锁视觉推理

    Seeing Isn't Enough: Unlocking Visual Reasoning by Bridging the Reasoning-Access Gap with TIGER

    [https://arxiv.org/abs/2611.10552](https://arxiv.org/abs/2611.10552)

    该论文揭示了多模态大语言模型中存在的“模态推理访问差距”——模型虽能准确感知图像内容却无法将其路由进已有的文本推理机制——并提出TIGER训练方法，通过挖掘模型在文本上成功但在对应对称图像上失败的模态反事实样本来定向弥合这一差距，从而解锁视觉推理能力。

    

    多模态大语言模型（MLLMs）从其语言骨干网络和文本预训练中继承了强大的文本推理能力，但往往无法将同等水平的能力应用于视觉输入。我们通过对模型在语义等价的文本输入和渲染文本图像输入上进行评估，证明这种性能下降不能仅仅归因于感知失败：模型能够准确读取图像中的文本，并在获得自身转录文本时恢复接近文本输入的性能，但就是无法直接对图像进行良好的推理。我们将这种差异表征为模态推理访问差距，即模型无法将感知到的视觉内容路由到文本可访问的现有推理机制中。为弥合这一差距，我们提出了TIGER（面向文本到图像差距的定向训练以增强推理），该方法首先挖掘合成模态反事实样本，即模型在某个文本问题上能成功求解，但在提供对应图像时却失败的情况……

    arXiv:2611.10552v2 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) inherit strong textual reasoning capabilities from their language backbones and text pretraining, but often fail to apply the same level of competency to visual inputs. We show that this degradation cannot be attributed to only perception failures, by evaluating models on semantically-equivalent text and rendered-text image inputs: models can accurately read text in the images and recover near-text performance when provided their own transcriptions, but just cannot reason well on the images directly. We characterize this discrepancy as a modality reasoning-access gap, where models fail to route perceived visual content into existing reasoning machinery accessible with text. To bridge this, we propose TIGER (Text-to-Image Gap-targeted training for Enhanced Reasoning), which first mines synthetic modality counterfactuals where a model succeeds on a text problem but fails when provided the corres
    
[^253]: 超越平均间隙：转移角度QAOA在超导处理器上保留了什么

    Beyond the Mean Gap: What Transferred-Angle QAOA Retains on a Superconducting Processor

    [https://arxiv.org/abs/2611.10529](https://arxiv.org/abs/2611.10529)

    该研究发现在超导处理器上运行深度为2的转移角度QAOA时，12量子比特电路虽保留了29%–46%的平均间隙改进，但最优解概率仅保留5%–15%，表明仅靠平均成本对比验证硬件QAOA性能会掩盖对优化最关键的最优解概率的大幅损失。

    

    量子近似优化算法（QAOA）的硬件实验通常通过将设备的平均成本与模拟结果进行比较来验证，但这种检查方式可能会遗漏对优化真正重要的内容。本文证明，在全局去极化噪声下，平均归一化间隙以及每一组比特串（包括最优解在内）经基底校正后的概率，都会保留其相对于均匀采样无噪声改进的相同比例。研究团队在IBM Heron处理器ibm_marrakesh上，使用深度为2的转移角度QAOA电路测试了这一预测，所求解的是交替方向分解方法的二进制子问题，并分别在有和无动态解耦与twirling的条件下进行实验。在12个量子比特的规模下，主族电路保留了其平均间隙改进的29%到46%，但只保留了最优解概率的5%到15%，且在每次运行的十个实例中至少有九个实例表现更低。近优比特串的损失程度与最优解大致相当，而应避免……

    arXiv:2611.10529v2 Announce Type: replace-cross  Abstract: Hardware executions of the quantum approximate optimization algorithm (QAOA) are often validated by comparing the device's mean cost with a simulation, a check that can miss what matters for optimization. Under global depolarization, the mean normalized gap and the floor-corrected probability of every set of bitstrings, the optimum included, retain the same fraction of their noiseless improvement over uniform sampling. We test this prediction with depth-2, transferred-angle QAOA circuits for binary subproblems of an alternating-direction decomposition on the IBM Heron processor \texttt{ibm\_marrakesh}, with and without dynamical decoupling and twirling. At 12 qubits, the main-family circuits retained 29\% to 46\% of their mean-gap improvement but only 5\% to 15\% of their probability of the optimum, lower on at least nine of ten instances per run. Near-optimal bitstrings were lost about as much as the optimum, whereas the avoid
    
[^254]: 分裂与桥接：通过扩散桥接实现多模态生成

    Split and Bridge: Multimodal Generation via Diffusion Bridging

    [https://arxiv.org/abs/2611.10441](https://arxiv.org/abs/2611.10441)

    提出了一种自下而上的多模态生成框架 SnB，通过在采样时利用桥接模块连接预训练的单模态扩散模型，无需联合训练即可生成连贯的多模态样本，实现了预训练模型的即插即用复用。

    

    视觉数据通常通过多种互补的模态来表达（例如图像和分割掩码），每种模态捕捉同一底层结构的不同方面。现有的大多数多模态方法采用自上而下的范式，通过从头开始的联合训练来学习一个共享的潜在变量，这限制了可扩展性。我们提出了 Split and Bridge（SnB），这是一种自下而上的多模态生成框架，它在采样阶段将预训练的单模态扩散模型耦合起来。SnB 并不学习统一的潜在空间，而是沿着一条共享的扩散轨迹前进到某个分裂点，然后在桥接模块的引导下分支为各模态特定的轨迹，从而生成连贯的多模态样本。这种设计使得强大的预训练模型可以即插即用地被复用，无需联合训练，使其具有良好的可扩展性和易适配性。在标准基准（PolyMNIST、CelebAMask-HQ）以及更复杂的场景（PIE-B……（摘要被截断）

    arXiv:2611.10441v2 Announce Type: replace-cross  Abstract: Visual data are often expressed through multiple complementary modalities (e.g., images and segmentation masks), each capturing different aspects of the same underlying structure. Most existing multimodal methods adopt a top-down paradigm, learning a shared latent variable through joint training from scratch, which limits scalability. We propose Split and Bridge (SnB), a bottom-up framework for multimodal generation that couples pretrained unimodal diffusion models at sampling time. Rather than learning a unified latent space, SnB generates coherent multimodal samples by following a shared diffusion trajectory up to a split point and then branching into modality-specific trajectories guided by a bridge module. This design enables plug-and-play reuse of powerful pretrained models without joint training, making it scalable and easily adaptable. Across standard benchmarks (PolyMNIST, CelebAMask-HQ) and more complex settings (PIE-B
    
[^255]: FAIDER：面向大语言模型推荐系统的公平感知条目级解码

    FAIDER: Fairness-Aware Item-Level Decoding for Large Language Model Recommenders

    [https://arxiv.org/abs/2611.10285](https://arxiv.org/abs/2611.10285)

    FAIDER针对无法访问模型权重和词表logits的大语言模型推荐系统，提出条目级解码方法，通过保形启发的校准确定跨组语义差异阈值，并在解码中结合硬过滤或对偶启发的软惩罚来控制公平性违规率。

    

    大语言模型（LLM）推荐系统在用户的受保护属性发生变化时，即使交互历史保持不变，也可能产生语义上不同的推荐结果。当模型权重和完整词表的logits不可访问时，控制这种差异极具挑战性。我们提出了FAIDER（面向推荐的公平感知条目级解码），它通过选择完整的候选条目将公平性融入列表构建过程，而不是操纵词元级别的logits或反复修复已生成的输出。其核心思想是将违规的操作性判据与生成被约束的强度分离开来：受保形（conformal）方法启发的经验校准确定一个跨组语义差异阈值，而在线解码则应用硬过滤，或通过对偶启发的反馈进行调节的软惩罚，以达到预先规定的违规率。此外，我们还建立了一项路径层面的保证：未……（原文摘要在此处截断）

    arXiv:2611.10285v2 Announce Type: replace  Abstract: Large language model (LLM) recommenders can produce semantically different recommendations when a user's protected attribute changes, even if the interaction history remains fixed. Controlling this disparity is challenging when model weights and full-vocabulary logits are inaccessible. We propose FAIDER (Fairness-Aware Item-level DEcoding for Recommendation), which incorporates fairness into list construction by selecting complete item candidates rather than manipulating token-level logits or repeatedly repairing finished outputs. The central idea is to separate the operational criterion for a violation from how strongly generation is constrained. Conformal-inspired empirical calibration fixes a cross-group semantic-disparity threshold, while online decoding applies either hard filtering or a soft penalty adjusted through dual-inspired feedback to target a prescribed violation rate. Separately, we establish a pathwise guarantee: unde
    
[^256]: 设计有效独立验证机构的关键制度考量

    Key Institutional Considerations for Designing Effective Independent Verification Organizations

    [https://arxiv.org/abs/2611.10168](https://arxiv.org/abs/2611.10168)

    本文提出设计有效独立验证机构（IVO）所需的关键制度考量，主张IVO生态系统应超越灾难性风险评估，扩展至AI生命周期的系统与应用层面，为前沿企业和企业级公司提供情境化的AI模型、系统及应用评估。

    

    独立验证机构（IVOs）已成为促进对AI技术进行高质量独立评估的一种流行的、获得两党支持的解决方案。但发展势头已超越设计进度：关于健全IVOs所需必要运行条件的共识正在形成，然而对于确保满足这些条件的制度与政策规范却鲜有讨论。目前IVOs主要被设想为在AI前沿和模型层面运作的机制，用于评估灾难性风险。关于如何建立或监管一个能够应对各种风险（即超越灾难性风险的范围）、并能在AI生命周期的部署与使用阶段于系统层面和应用层面运作的IVO生态系统，相关工作仍然很少。本文认为，IVO的设计应确保具备在具体情境中为前沿企业和企业级公司评估AI模型、系统和应用的能力。

    arXiv:2611.10168v2 Announce Type: replace  Abstract: Independent verification organizations (IVOs) have emerged as a popular, bipartisan solution for facilitating high-quality independent evaluations of AI technologies. But momentum is outpacing design: there is emerging consensus on the necessary operating conditions for robust IVOs, but little discourse on institutional and policy specifications to ensure those conditions are met. IVOs are currently envisioned primarily as a mechanism operating at the AI frontier and model level to evaluate catastrophic risks. Little work has yet been done on how to establish or enforce an IVO ecosystem that is equipped to handle myriad risks (i.e., beyond catastrophic ones) and operate at the system and application level across the deployment and use stages of the AI lifecycle. This paper argues that IVO design should ensure the capacity for evaluating AI models, systems, and applications in context for both frontier companies and enterprise compani
    
[^257]: 退伍军人心理健康危机严重程度标注：大语言模型比人类专家更具一致性吗？

    Veterans' Mental Health Crisis Severity Annotation: Do LLMs Agree More Than Human Experts?

    [https://arxiv.org/abs/2611.10048](https://arxiv.org/abs/2611.10048)

    本研究对比了大语言模型与八位退伍军人专家对社交媒体帖子危机严重程度的标注，发现选定的LLM之间的一致性显著更高（序数α达0.8336），而人类专家一致性较低（α=0.494）且分歧多发生在相邻严重程度级别之间。

    

    大语言模型（LLM）为心理健康和危机相关社交媒体内容的标注提供了一种可扩展的方法，但其在安全关键任务中的可靠性仍不明确。本研究将基于LLM的退伍军人相关社交媒体帖子的危机严重程度标注，与八位退伍军人领域专家（SME）的标注进行了对比评估。研究使用149条推文和五个严重程度类别，在八种提示词变体下评估了15个开源模型和10个专有LLM，考察了标注性能、模型间一致性、提示敏感性以及高严重度危机检测能力。人类评审者表现出更高的序数一致性而非名义一致性（α = 0.494），表明分歧常常发生在相邻的严重程度级别之间。相比之下，选定的LLM表现出显著更高的模型间一致性，在基线提示下序数一致性（α）达到0.8336。专有模型总体（原文在此处截断）

    arXiv:2611.10048v2 Announce Type: replace  Abstract: Large language models (LLMs) offer a scalable approach for annotating mental health and crisis-related social media content, but their reliability in safety-critical tasks remains unclear. This study evaluates LLM-based crisis-severity annotation of veteran-related social media posts against annotations from eight Veteran Subject Matter Experts (SMEs). Using 149 tweets and five severity categories, we evaluate 15 open-source and 10 proprietary LLMs across eight prompt variants, examining annotation performance, inter-model agreement, prompt sensitivity, and high-severity crisis detection. Human reviewers showed higher ordinal than nominal agreement ({\alpha} = 0.494), indicating that disagreements often occurred between neighboring severity levels. In contrast, selected LLMs demonstrated substantially higher inter-model consistency, with ordinal agreement ({\alpha}) reaching 0.8336 under the baseline prompt. Proprietary models genera
    
[^258]: CulturalCrosswords：答案约束与网格填写的多语言基准测试

    CulturalCrosswords: A Multilingual Benchmark of Answer Constraints and Grid Completion

    [https://arxiv.org/abs/2611.10044](https://arxiv.org/abs/2611.10044)

    该论文构建了覆盖42种语言、53个填字游戏的多语言基准CulturalCrosswords，发现长度提示和网格约束等答案约束能显著提升大型语言模型解答文化谜题时答案的准确性与一致性。

    

    arXiv:2611.10044v2 公告类型： replace-cross 摘要：当一个谜语有多个符合条件的答案时，精确匹配的评分方式可能会误判有效的回答。CulturalCrosswords 测试答案约束如何影响大型语言模型（LLM）解决文化谜题的能力，这类谜题的答案依赖于特定社区的知识。我们从42种语言的众包谜题中构建了53个填字游戏，并在每个线索集的五个版本上评估了11种LLM配置：普通批次、带长度信息的批次、带长度和部分答案提示的批次，以及要求提供答案列表或完成整个网格的填字游戏形式。在962个存在备选答案的线索上，长度信息使模型与参考答案的一致性提高了22.8个百分点，而当备选答案也计为正确时仅提高8.4个百分点。因此，部分收益似乎来自模型选择了所需的答案形式。填字游戏形式的输入使所有交叉点答案兼容的比例提高了9.7个百分点；而要求完成整个网格则揭示了显著的……

    arXiv:2611.10044v2 Announce Type: replace-cross  Abstract: When several answers fit a riddle, exact-match scoring can penalise a valid response. CulturalCrosswords tests how answer constraints affect large language models (LLMs) solving cultural riddles, whose solutions draw on knowledge specific to a community. We constructed 53 crosswords from crowdsourced riddles in 42 languages and evaluated 11 LLM configurations on five versions of each clue set: plain batches, batches with lengths, batches with lengths and partial-answer hints, and crosswords requesting an answer list or a completed grid. On 962 clues with alternative answers, lengths improve agreement with the reference by 22.8 percentage points, compared with 8.4 points when accepted alternatives also receive credit. Part of the gain therefore appears to come from selecting a required answer form. Crossword input increases the share of all crossings with compatible answers by 9.7 points; requiring a completed grid reveals large
    
[^259]: 知道何时干预：AI智能体的因果评估

    Knowing When to Intervene: Causal Evaluation of AI Agents

    [https://arxiv.org/abs/2611.09792](https://arxiv.org/abs/2611.09792)

    该论文提出了首个基于因果推断的方法学框架，用于评估在AI智能体执行轨迹中进行干预（如人工审查或行为引导）的因果价值。

    

    当前对智能体AI系统的评估主要依赖于基准测试。然而，基准测试的静态特性使其无法提供关于部署时审计干预价值的信息。为了实现可靠部署，我们常常需要在智能体的执行轨迹中判断是否进行干预，例如通过验证智能体当前的输出或引导其采取不同的行为，这可以采用人工审查、成本更高的模型或耗时的智能体监督。此类干预的价值从根本上说是一个因果推断问题，即这种干预的因果效应是否对最终结果有益。由于问题的动态性质，估计这种因果效应尤其具有挑战性：干预的影响取决于当前的上下文和环境，以及智能体未来的行为。我们开发了首个在方法学上有依据的框架，用于评估此类干预的价值。

    arXiv:2611.09792v1 Announce Type: cross  Abstract: Current evaluations of agentic AI systems primarily rely on benchmarks. However, the static nature of benchmarks makes them uninformative about the value of auditing interventions at deployment time. For reliable deployment, we often need to determine within an agent's trajectory whether or not to intervene, e.g., by verifying the agent's current output or steering it toward different behavior, using either human review or more costly models or time-consuming agentic supervision. The value of such interventions is fundamentally a causal inference question, asking whether the causal effect of such an intervention is beneficial for the final outcome. Estimating this causal effect is particularly challenging due to the dynamic nature of the problem: the impact of an intervention depends on the current context and environment, as well as the agent's future behavior. We develop the first methodologically grounded framework for evaluating th
    
[^260]: BEiTScore++：基于BEiT-3与共形化分位数回归的无参考图像描述评估

    BEiTScore++: Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression

    [https://arxiv.org/abs/2611.09505](https://arxiv.org/abs/2611.09505)

    提出BEiTScore++，通过共形化分位数回归为无参考图像描述评估指标引入共形校准的置信区间，在单次前向传播中同时实现准确评分与可靠的不确定性量化。

    

    可靠的不确定性估计对于将评估指标部署到现实环境中至关重要，因为现实中输入歧义和领域偏移都很常见。在这项工作中，我们提出了BEiTScore图像描述评估指标的不确定性感知扩展版本，在对齐分数之外还能产生经过共形校准的置信区间。我们的方法对人类评判的预测分布进行建模，能够在单次前向传播中同时实现准确评分和有原则的不确定性量化。我们在多个带有人工标注评分的基准上评估了BEiTScore++指标，结果表明它保持了BEiTScore的排序相关性，同时在大多数基准上产生的区间宽度比蒙特卡洛dropout更贴近预测误差。估计的不确定性在图像退化条件下以及域外艺术图像上也会随之增大，这使得BEiTScore++成为一个实用的不确定性感知评估工具。

    arXiv:2611.09505v1 Announce Type: cross  Abstract: Reliable uncertainty estimation is important for deploying evaluation metrics in real-world settings, where both input ambiguity and domain shifts are common. In this work, we propose an uncertainty-aware extension of the BEiTScore image captioning metric, producing conformally calibrated confidence intervals alongside alignment scores. Our approach models predictive distributions over human judgments, enabling both accurate scoring and principled uncertainty quantification in a single forward pass. We evaluate our BEiTScore++ metric across multiple benchmarks with human-annotated ratings, showing that it preserves the ranking correlation of BEiTScore, while producing intervals whose width tracks the prediction error more closely than Monte Carlo dropout on most benchmarks. The estimated uncertainty also grows under image degradation and on out-of-domain artwork images, making BEiTScore++ a practical tool for uncertainty-aware evaluati
    
[^261]: Circe：基于组件交互一致性的免训练大语言模型自我评估方法

    Circe: Training-Free LLM Self-Evaluation via Component Interaction Consistency

    [https://arxiv.org/abs/2611.07948](https://arxiv.org/abs/2611.07948)

    提出Circe——一种免训练的大语言模型自我评估方法，通过分析注意力与前馈网络等组件间的交互一致性并结合内外部信号来评估回答正确性，无需参考答案或重复采样。

    

    随着大语言模型在实际应用中的部署日益增多，评估其回答的正确性对于可靠使用至关重要。然而，这种评估具有挑战性，因为在实践中参考答案往往难以获得。现有的免参考方法主要依赖输出信号，这可能会给错误的回答赋予高分，因此并不一定能反映正确性。这促使我们探索模型内部计算以获取互补的正确性信号。近期研究利用了内部状态，但它们主要分析跨层的隐藏状态轨迹，这仅记录了注意力机制和前馈网络（FFN）的组合更新，而忽视了这些组件之间如何交互。因此，我们提出了Circe，一种免训练的自我评估方法，它结合内部和外部信号来评估回答，无需参考答案或重复采样。对于内部

    arXiv:2611.07948v1 Announce Type: new  Abstract: As large language models (LLMs) are increasingly deployed in real-world applications, evaluating the correctness of their responses is crucial for reliable use. However, such evaluation is challenging because reference answers are rarely available in practice. Existing reference-free methods mainly rely on output signals, which may assign high scores to incorrect responses and thus do not necessarily reflect correctness. This motivates exploring the model's internal computation for complementary correctness signals. Recent studies exploit internal states, but they mainly analyze hidden-state trajectories across layers, which record only the combined updates of attention and feed-forward networks (FFNs) and overlook how these components interact. We therefore propose Circe, a training-free self-evaluation method that assesses a response using both internal and external signals, without reference answers or repeated sampling. For internal 
    
[^262]: 面向病理档案的可扩展诊断驱动全切片图像检索

    Towards Scalable Diagnostic-Driven Whole-Slide Image Retrieval for Pathology Archives

    [https://arxiv.org/abs/2611.07935](https://arxiv.org/abs/2611.07935)

    提出了一种可扩展、诊断驱动的全切片图像检索框架CLEAR-WSI，通过固定长度的切片嵌入并利用诊断标签细化检索候选，摆脱了手工索引流水线，在公开和临床病理数据上实现了高效的相似病例检索。

    

    数字病理学的日益普及使苏木精-伊红（H&E）染色的全切片图像集合迅速扩大，然而由于元数据往往不一致或缺失，如何高效地浏览这些档案仍然具有挑战性。基于内容的图像检索通过直接从视觉表示中识别相似病例，而非依赖人工标注或元数据，来应对这一挑战。现有检索系统已展现出良好的性能，但大多依赖手工设计的索引流水线和预定义的采样策略，限制了其可扩展性和泛化能力。本文提出CLEAR-WSI（Constant-Length Embedding for Automatic Retrieval of Whole-Slide Images，用于全切片图像自动检索的定长嵌入），作为一种可扩展的、诊断驱动的切片检索框架，并在公开数据和临床数据上进行了验证。CLEAR-WSI采用固定长度的切片嵌入，并通过预测的诊断标签或真实标签对检索到的候选结果进行细化……

    arXiv:2611.07935v1 Announce Type: cross  Abstract: The growing adoption of digital pathology has rapidly expanded collections of hematoxylin-and-eosin whole-slide images, yet efficient navigation through these archives remains challenging, as metadata is often inconsistent or unavailable. Content-based image retrieval addresses this challenge by identifying similar cases directly from visual representations instead of manual annotations or metadata. Existing retrieval systems have demonstrated promising performance but largely rely on handcrafted indexing pipelines and predefined sampling strategies, constraining scalability and generalization. Here, we present CLEAR-WSI (Constant-Length Embedding for Automatic Retrieval of Whole-Slide Images) as a scalable, diagnostic-driven slide-retrieval framework, validated on both public and clinical data. Using a fixed-length slide embedding, CLEAR-WSI refines retrieved candidates with either a predicted diagnostic label or the ground-truth labe
    
[^263]: 面向多重违规知识图谱修复的选择性检索路由

    Selective Retrieval Routing for Multi-Violation Knowledge Graph Repair

    [https://arxiv.org/abs/2611.07342](https://arxiv.org/abs/2611.07342)

    该论文提出一种选择性检索路由框架，由LLM修复智能体编辑知识图谱、路由器智能控制对外部证据检索的访问，并利用违规注入生成的修复结果训练特定部署的路由器，从而提升多重违规知识图谱的修复效果。

    

    知识图谱通常包含缺失或不一致的事实，这会损害构建在其之上的应用。模式验证能够检测此类违规，但无法确定多种有效修复方案中哪一种能够恢复预期的图谱。外部语料库可以提供识别预期修复所需的证据，但从中检索的成本高昂且并不总是成功。我们提出了一个面向迭代图谱修复中选择性检索的框架：一个基于大语言模型（LLM）的修复智能体负责探索和编辑图谱，同时一个路由器控制其访问负责检索外部证据的研究员（researcher）智能体的权限。我们扩展了一个违规注入框架，以生成包含多个兼容违规和已知参考修复的案例，智能体在这些案例上的修复结果为训练特定部署场景的路由器提供了监督信号。在两个楼宇自动化数据集上的实验表明，研究员智能体的访问对某些违规能改善修复效果，而对另一些违规则会降低效果……

    arXiv:2611.07342v1 Announce Type: cross  Abstract: Knowledge graphs often contain missing or inconsistent facts that undermine the applications built on them. Schema validation detects such violations but does not identify which of several valid repairs recovers the intended graph. External corpora can supply the evidence needed to identify the intended repair, but retrieving from them is costly and not always successful. We propose a framework for selective retrieval in iterative graph repair. An LLM repair agent explores and edits the graph, while a router controls its access to a researcher agent that retrieves external evidence. We extend a violation-injection framework to generate cases with multiple compatible violations and known reference repairs. The agent's repair outcomes on these cases provide supervision for training the deployment-specific router. On two building-automation datasets, researcher access improves repairs for some violations and degrades them for others, with
    
[^264]: CompressionBench：你的编程智能体如何压缩大语言模型？

    CompressionBench: How can your coding agents compress LLMs?

    [https://arxiv.org/abs/2611.06858](https://arxiv.org/abs/2611.06858)

    该论文提出CompressionBench基准，将大语言模型压缩形式化为一项终端任务，系统评估了编程智能体自主压缩LLM的能力，发现大多数智能体提交能通过独立验证，且压缩后模型性能接近原始BF16模型。

    

    递归自我改进要求AI系统能够改进后续系统所使用的模型和基础设施。大语言模型（LLM）压缩提供了一个具体的效率目标：在更小的模型存储预算内保留预训练模型的行为。我们研究编程智能体能否自主地完成这一步骤。我们将压缩形式化为一项终端任务：智能体接收源模型检查点、存储限制、推理环境和时间预算，然后构建压缩后的权重和可执行的推理代码。智能体自主控制其算法和实验；困惑度反馈用于指导搜索，独立验证则检查最终模型。在六个智能体和六种模型规模的设置中，36个提交中有29个通过验证。每个任务选择困惑度最低的智能体设计，相比BF16平均增加3.4%的困惑度，下游准确率为77.33%，而BF16为79.56%。

    arXiv:2611.06858v1 Announce Type: new  Abstract: Recursive self-improvement requires AI systems to improve the models and infrastructure used by subsequent systems. LLM compression offers a concrete efficiency objective: retain a pretrained model's behavior within a smaller model storage budget. We study whether coding agents can carry out this step autonomously. We formulate compression as a terminal task in which an agent receives a source checkpoint, storage limit, inference environment, and time budget, then constructs compressed weights and executable inference code. The agent controls its algorithm and experiments; perplexity feedback guides search, and independent verification checks the final model. Across six agents and six model--size settings, 29 of 36 submissions pass verification. Selecting the lowest-perplexity agent design per task yields an average 3.4% perplexity increase over BF16 and 77.33% downstream accuracy, compared with 79.56% for BF16. Recorded experiments reve
    
[^265]: ALTO：非洲与黎凡特地区技术助长的基于性别的暴力语料库

    ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus

    [https://arxiv.org/abs/2611.06706](https://arxiv.org/abs/2611.06706)

    该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。

    

    技术助长的基于性别的暴力（TFGBV）是一种普遍存在的网络危害，通过多种直接和隐蔽的形式影响女性和性别少数群体。当前用于性别伤害检测的模型和标注数据集主要被来自全球北方语言的数据以及用这些语言训练的模型所主导。我们与来自约旦、法语西非和肯尼亚的民间社会组织（CSOs）合作，开发了一种社区主导的方法，通过众包方式收集黎凡特阿拉伯语、非洲法语和斯瓦希里语中符合当地文化和语言特点的TFGBV实例。我们使用主动学习对众包数据进行标注，构建了一个规模虽小但性能出色的数据集（N=3022），命名为ALTO，用于改进这三种语言的TFGBV检测。通过基准测试和微调实验，我们表明现有的有害内容检测模型在ALTO数据集上的TFGBV召回率较低，而经过微调后性能有所提升。我们发现模型往往无法识别隐蔽的……

    arXiv:2611.06706v1 Announce Type: cross  Abstract: Tech-facilitated gender based violence (TFGBV) is a pervasive online harm affecting women and gender minority groups through its many direct and covert forms. Current models and annotated datasets for gendered harm detection are dominated by data from, and models trained in, Global North languages. In collaboration with civil society organizations (CSOs) from Jordan, Francophone West Africa, and Kenya, we developed a community-led approach to crowdsource culturally and linguistically relevant instances of TFGBV in Levantine Arabic, African French, and Swahili. We annotated the crowdsourced data using active learning to develop a small but performant dataset (N=3022) called ALTO to improve TFGBV detection in the three languages. Through benchmarking and fine-tuning experiments, we show that existing harmful content detection models show poor TFGBV recall within ALTO and improve with fine-tuning. We found that models often miss covert ha
    
[^266]: 语言模型知道哪些约束属于同一组吗？

    Do Language Models Know Which Constraints Belong Together?

    [https://arxiv.org/abs/2611.05726](https://arxiv.org/abs/2611.05726)

    该论文证明，在 Dantzig-Wolfe 重构中寻找只有组合起来才能改进边界的约束族隐藏组合需要 k^w 次评估，而一个遗漏 e 个成员的分组预测可将成本降至 Θ(k^e)，因此预测节省的评估次数直接度量了预测器（如语言模型）对约束分组知识的掌握程度。

    

    Dantzig-Wolfe 重构通过将选定的一组约束族凸化来加强混合整数规划的线性松弛，而每个候选选择都需要付出一次列生成的运行代价。当各约束族只有组合在一起才能改进边界时，这一选择变得困难，因为每次只添加一个约束族的搜索会在线性松弛处停滞。批量生产、网络设计和设施选址中的经典不等式正建立在这类零边际组之上。我们证明，当每次评估最多凸化 k 个约束族中的 s 个时，任何仅观察边界值的方法都需要量级为 k^w 的覆盖次数评估，才能找到包含 w 个约束族的隐藏组合，且该结论在实际的 Dantzig-Wolfe 松弛上同样成立。一个遗漏 e 个成员的预测可通过一次无需知晓 e 的分阶段搜索，将成本降至 Θ(k^e)，因此预测所节省的评估次数直接度量了预测器所掌握的知识。我们研究了两种预测来源……

    arXiv:2611.05726v1 Announce Type: cross  Abstract: Dantzig-Wolfe reformulation strengthens the linear relaxation of a mixed-integer program by convexifying a chosen set of constraint families, and every candidate costs a column generation run. The choice is hard when families strengthen the bound only together, because a search that adds one family at a time then stops at the linear relaxation. The classical inequalities of lot sizing, network design and facility location rest on such zero-marginal groups. We prove that when each evaluation convexifies at most $s$ of $k$ families, any method that observes only bounds needs a covering number of evaluations, of order $k^w$, to find a hidden group of $w$ families, also on actual Dantzig-Wolfe relaxations. A prediction that misses $e$ members reduces the cost to $\Theta(k^e)$, through one staged search that does not need to know $e$, so the evaluations a prediction saves measure what the predictor knows. We study two sources of predictions
    
[^267]: 使用反馈模糊认知图混合体组合相关专家知识

    Combining Relevant Expert Knowledge Using Mixtures of Feedback Fuzzy Cognitive Maps

    [https://arxiv.org/abs/2611.05664](https://arxiv.org/abs/2611.05664)

    该论文将法律证据理论中的相关性定义扩展应用于神经符号系统，通过将专家的反馈模糊认知图混合为共同知识网络，并借助新的类贝叶斯定理对混合知识进行解混合，从而仅组合相关专家答案以提升系统的准确性和可扩展性。

    

    神经符号系统应当只组合来自专家智能体或其他知识源的相关答案。随着知识网络的不断增长并连接越来越多的知识源，针对给定输入查询仅使用相关专家对于准确性和可扩展性变得愈发重要。我们从证据法理论出发扩展了相关性的标准定义：在给定可用知识的条件下，当且仅当一个答案对于手头问题既具有证明力又具有实质性时，该答案才是相关的。我们将这一相关性方案应用于混合真实或人工专家的反馈模糊认知图（FCMs）。各专家FCM的有向因果图混合或组合成一个共同的知识网络。我们将专家的均衡极限环预测或答案与混合FCM的答案进行比较，然后通过它们之间的差异或变化来估计答案的证明力状态。一个新的类贝叶斯定理将知识混合物解混合为……（原文摘要在此处截断）

    arXiv:2611.05664v1 Announce Type: new  Abstract: A neuro-symbolic system should combine only relevant answers from expert agents or other knowledge sources. Using only relevant experts for a given input query becomes ever more important for accuracy and scalability as the knowledge network grows and connects ever more knowledge sources. We extend the standard definition of relevance from the legal theory of evidence: An answer is relevant if and only if it is both probative and material for the matter at hand given the knowledge available. We apply this relevancy scheme to mixing the feedback fuzzy cognitive maps (FCMs) of real or artificial experts. The expert FCM directed causal graphs mix or combine into a common knowledge network. We compare the equilibrium limit-cycle predictions or answers of the experts to the answers of the mixture FCM and then estimate the probative status of answers by their difference or variation. A new Bayes-like theorem de-mixes the knowledge mixture into
    
[^268]: Kierston 与 KierstonBench：宾夕法尼亚州阿伦敦市的模拟及语言模型在犯罪调查与监控任务中的评估

    Kierston & KierstonBench: Simulation of Allentown, PA and Evaluation of Language Models in Crime Investigation and Surveillance Tasks

    [https://arxiv.org/abs/2611.04785](https://arxiv.org/abs/2611.04785)

    该论文提出了城市级模拟引擎 Kierston 与配套基准 KierstonBench，通过模拟宾州阿伦敦市约12.5万名居民的日常活动及多源记录，首次实现了对大语言模型在犯罪调查与监控任务中能力的可量化评估。

    

    大型语言模型是刑事调查与大规模监控中收集和分析记录的候选工具，但其在这些工作中的能力与准确性目前无法衡量：实际部署缺乏评估模型结论所需的真实标准，也没有任何基准能在模拟环境中评估这些能力。我们提出了城市级模拟引擎 Kierston，以及针对已知模拟事件测试上述能力的基准 KierstonBench。Kierston 模拟了宾夕法尼亚州阿伦敦市约 125,000 名居民，生成他们的日常活动，以及监控与调查系统所收集的电信、金融、行政和车牌记录。这些记录源自共同的行为时间线，并分布在代表不同服务提供商的各个数据源中。模型通过受限的记录访问接口开展调查，对观测记录进行关联……（原文摘要在此处截断）

    arXiv:2611.04785v2 Announce Type: replace  Abstract: Large language models are candidates for collecting and analyzing records in criminal investigations and mass surveillance. Their capability and accuracy in this work cannot currently be measured. Real deployments lack the ground truth needed to score a model's conclusions, and no benchmark evaluates these capabilities in a simulated environment. We present Kierston, a city-scale simulation engine, and KierstonBench, a benchmark that tests these capabilities against known simulated events. Kierston models approximately 125,000 residents of Allentown, Pennsylvania, generating their daily activities and the telecommunications, financial, administrative and license-plate records that surveillance and investigative systems collect about them. These records derive from a common behavioral timeline and are distributed across sources representing separate providers. Models investigate through a restricted records interface, linking observat
    
[^269]: FORGE：面向故障的联锁装配可执行返工程序代码生成

    FORGE: Fault-Oriented Rework Code Generation for Executable Interlocking Assembly

    [https://arxiv.org/abs/2611.02536](https://arxiv.org/abs/2611.02536)

    该论文提出了FORGE-Bench基准和FORGE框架，用于评估并解决大语言模型在联锁装配故障条件下生成可执行返工程序的挑战。

    

    大语言模型在结构化推理和代码生成方面已展现出强大的能力，然而识别出装配故障并不保证模型能够生成可执行的返工程序。在联锁装配中，一个局部看似合理的修复程序可能违反阻塞依赖关系、遗漏必要的恢复操作，或以错误的配置终止。我们将这一挑战定义为故障条件下的可执行返工生成，即模型必须同时对结构故障进行推理，并生成能够恢复受约束装配状态的程序。为了对该能力进行基准测试，我们提出了FORGE-Bench，其中包含跨26个联锁装配体的4,000个任务，并预留了五个装配体用于评估模型对未见结构的泛化能力。我们进一步提出了FORGE（面向故障的联锁装配可执行返工程序代码生成），这是一个结构化的从状态到程序的框架，将返工过程分解为……（原文摘要在此处被截断）

    arXiv:2611.02536v2 Announce Type: replace  Abstract: Large language models have demonstrated strong capabilities in structured reasoning and code generation, yet identifying an assembly fault does not guarantee the ability to generate executable rework programs. In interlocking assemblies, a locally plausible repair program may violate blocking dependencies, omit required recovery operations, or terminate in an incorrect configuration. We identify this challenge as fault-conditioned executable rework generation, where models must jointly reason about structural failures and generate programs that restore constrained assembly states. To benchmark this capability, we introduce FORGE-Bench, containing 4,000 tasks across 26 interlocking assemblies with five assemblies held out for evaluating generalization to unseen structures. We further propose FORGE (Fault-Oriented Rework Code Generation for Executable Interlocking Assembly), a structured state-to-program framework that decomposes rewor
    
[^270]: MAGAR：面向基于GUI的3D机械装配与最小返工的多模态智能体

    MAGAR: Multimodal Agents for GUI-Based 3D Mechanical Assembly and Minimal Rework

    [https://arxiv.org/abs/2611.02472](https://arxiv.org/abs/2611.02472)

    提出MAGAR-Bench基准（涵盖26个装配族的4,000个多模态样本，将装配决策、原子操作与最小返工分离）以及基于冻结策略的选择性纠正框架MAGAR，通过匹配纠正粒度与输出依赖关系，实现基于GUI的3D机械装配的最小返工。

    

    一个GUI命令在界面层面可能正确，却仍会产生错误的装配。智能体必须识别相关组件、遵循它们的结构关系，并确定在恢复过程中哪些部件应当移动。我们提出MAGAR-Bench，一个包含来自26个装配族的4,000个多模态样本的基准，它将装配决策、原子操作与最小返工分离开来。在返工方面，目标函数统计的是每个状态下已验证恢复候选集中被移动的不同部件数量，而非操作次数。我们还提出了MAGAR，一个建立在通过监督微调获得的冻结策略之上的选择性纠正框架。其关键设计是将纠正粒度与输出依赖关系相匹配：结构化候选用于修订装配字段与返工范围，而完整操作候选则保持原生GUI原语与其参数的耦合。学习到的选择器会保留原始预测，除非……（原文摘要在此处截断）

    arXiv:2611.02472v2 Announce Type: replace  Abstract: A GUI command can be correct at the interface level yet still produce an incorrect assembly. An agent must identify the relevant components, respect their structural relationships, and determine which parts should move during recovery. We introduce MAGAR-Bench, a benchmark of 4,000 multimodal samples from 26 assembly families that separates Assembly Decision, Atomic Action, and Minimal Rework. For rework, the objective counts distinct moved parts within the validated recovery candidate set for each state, rather than the number of actions. We also introduce MAGAR, a selective correction framework built on a frozen policy obtained by supervised fine-tuning. Its key design matches correction granularity to output dependencies. Structured candidates revise assembly fields and rework scope, while complete action candidates keep native GUI primitives coupled with their parameters. A learned selector retains the original prediction unless 
    
[^271]: 用于联合多径参数估计的谱-卷积分解Transformer：在毫米波定位与无线电SLAM中的应用

    Spectral-Convolutional Factorized Transformer for Joint Multipath Parameter Estimation: Applications to mmWave Positioning and Radio SLAM

    [https://arxiv.org/abs/2611.01491](https://arxiv.org/abs/2611.01491)

    提出了一种谱-卷积分解Transformer（SCFT），能够从含噪信道估计中联合估计时延、到达角和离开角等多径参数，为毫米波定位和无线电SLAM等下游任务提供关键几何信息。

    

    未来的6G无线网络有望向AI原生系统演进，以支持智能化、环境感知的服务，包括定位与建图。当前的研究已将人工智能整合到通信的多个方面，尤其是无线信道估计。然而，大多数基于学习的信道估计器关注面向通信的目标，而非显式估计诸如时延、到达角（AoA）和离开角（AoD）等多径参数。这些参数为定位和无线电同步定位与建图（SLAM）等下游任务提供了关键的几何信息。在这项工作中，我们提出了一种谱-卷积分解Transformer（SCFT），用于从含噪的复数信道估计和物理波束对描述子中联合估计多径分量（MPC）。SCFT结合了局部谱卷积、置换等变的波束集注意力（此处摘要被截断）。

    arXiv:2611.01491v2 Announce Type: replace-cross  Abstract: Future 6G wireless networks are expected to evolve toward AI-native systems that support intelligent, environment-aware services, including positioning and mapping. Current research has already integrated AI into several aspects of communication, particularly wireless channel estimation. However, most learning-based channel estimators focus on communication-oriented objectives rather than explicitly estimating multipath parameters such as delay, angle of arrival (AoA), and angle of departure (AoD). These parameters provide essential geometric information for downstream tasks including positioning and radio simultaneous localization and mapping (SLAM). In this work, we propose a Spectral-Convolutional Factorized Transformer (SCFT) for joint multipath component (MPC) estimation from noisy complex channel estimates and physical beam-pair descriptors. SCFT combines local spectral convolution, permutation-equivariant beam-set attent
    
[^272]: 确定性大语言模型协商中的低裕度分支点

    Low-Margin Branch Points in Deterministic LLM Deliberation

    [https://arxiv.org/abs/2611.00546](https://arxiv.org/abs/2611.00546)

    该论文通过白盒审计发现，在确定性多智能体LLM协商中，低裕度词元位置显著更易引发对话分支，且降低裕度的干预会大幅增加分支发生频率，从而揭示了多智能体LLM系统对内部局部扰动的脆弱性。

    

    确定性的多智能体大语言模型（LLM）协商可以精确重放固定的对话记录，然而仅凭精确重放并不能测试系统对受控内部干预的局部抵抗能力。由于每个响应都会成为后续智能体的上下文，一个被改变的贪心解码词元可能会使经由对话记录介导的执行过程转向。我们针对这种局部失效模式，在贪心解码下的五智能体委员会中进行了白盒审计。我们前瞻性地选择低裕度和高裕度的词元位置，要求精确的无操作重放，注入范数相对的残差位移，并测量十个生成词元内的首次对话记录分歧。结果显示，低裕度位置比高裕度对照更具分支敏感性，而匹配的降低裕度的梯度方向比增加裕度的对照方向产生分支的频率显著更高。随后，我们将选定的已实现分支在委员会中继续传递，以区分文本层面改变的持续性与偏好及最终决策变化之间的差异。

    arXiv:2611.00546v2 Announce Type: replace  Abstract: Deterministic multi-agent LLM deliberation can exactly re- play a fixed transcript, yet exact replay alone does not test local resistance to controlled internal interventions. Because each response becomes context for later agents, a changed greedy token can redirect a transcript-mediated execution. We present a white-box audit of this local failure mode in a five-agent committee under greedy decoding. We prospectively select low- and high-margin token positions, require exact no-op re- play, inject norm-relative residual displacements, and measure first transcript divergence within ten generated tokens. Low- margin positions are more branch-sensitive than high-margin controls, while matched margin-reducing gradient directions branch substantially more often than margin-increasing con- trols. We then continue selected realized branches through the committee to distinguish textual persistence from preference and final-decision changes
    
[^273]: 面向视觉-语言模型零样本对抗鲁棒性的文本响应感知引导

    Textual Response-Aware Guidance for Zero-Shot Adversarial Robustness of Vision-Language Models

    [https://arxiv.org/abs/2610.40309](https://arxiv.org/abs/2610.40309)

    提出文本响应感知引导方法TRAG，利用预训练文本编码器对输入嵌入局部变化的响应来约束视觉表示漂移，从而在不损害零样本性能的前提下提升视觉-语言模型的对抗鲁棒性。

    

    视觉-语言模型（VLMs）展现出强大的零样本泛化能力，但在对抗攻击面前仍然脆弱。现有的对抗微调方法通常利用预训练模型的引导，在提升鲁棒性的同时保持图像-文本对齐。我们对视觉表示漂移的分析揭示了两种代表性引导形式的互补局限性：匹配视觉特征对所有方向施加同等约束，覆盖范围广泛但缺乏语义侧重；匹配预测分布则优先考虑语义相关的方向，但仍受限于候选文本和预测概率。这些局限性可能阻碍预训练图像-文本关系的保持，并降低零样本性能。因此，我们提出了文本响应感知引导（TRAG），它利用预训练文本编码器对输入嵌入局部变化的响应来约束视觉表示漂移。

    arXiv:2610.40309v2 Announce Type: cross  Abstract: Vision-language models (VLMs) exhibit strong zero-shot generalization but remain vulnerable to adversarial attacks. Existing adversarial fine-tuning methods often use pre-trained model guidance to preserve image--text alignment while improving robustness. Our analysis of visual representation drift reveals complementary limitations of two representative forms of guidance. Matching visual features constrains all directions equally, providing broad coverage without semantic emphasis. Matching prediction distributions prioritizes semantically relevant directions, but remains limited by the candidate texts and prediction probabilities. These limitations can hinder preservation of pre-trained image--text relationships and degrade zero-shot performance. We therefore propose Textual Response-Aware Guidance (TRAG), which constrains visual representation drift using the pre-trained text encoder's responses to local variations in input embedding
    
[^274]: 仅凭参考样本学习推理：面向现代大语言模型后训练的强化对抗学习

    Learning to Reason from Reference Samples Alone: Reinforced Adversarial Learning for Modern LLM Post-Training

    [https://arxiv.org/abs/2610.38851](https://arxiv.org/abs/2610.38851)

    该论文提出强化对抗学习（RAL）方法，通过共同训练的大语言模型评判者，仅凭参考答案即可作为推理型大语言模型在策略后训练的唯一奖励信号，无需显式可验证奖励或参考推理轨迹，且实验表明其性能可媲美直接的RLVR。

    

    在许多需要对大语言模型进行后训练的场景中，高质量答案是可辨识的，但难以用显式奖励来打分。带可验证奖励的强化学习（RLVR）仅限于存在显式正确性信号的任务。以往训练推理型大语言模型的对抗方法，要么需要用共同训练的评论者来补充可验证奖励，要么需要参考推理轨迹。我们探究共同训练的大语言模型评判者能否仅凭参考答案，将其转化为推理型大语言模型在策略后训练中的唯一奖励信号。我们将该方法称为强化对抗学习（RAL）。我们首先在三个可验证任务（整数乘法、基于代码的算术表达式合成以及SMILES到分子式的翻译）上验证RAL，其中验证器仅用于评估。在评判者能力足够强且训练充分的情况下，RAL的表现可媲美直接的RLVR。随后我们将RAL应用于两个场景的科学摘要生成任务（原文在此处截断）。

    arXiv:2610.38851v1 Announce Type: new  Abstract: In many settings where LLM post-training is desirable, high-quality answers are recognizable but difficult to score with explicit rewards. Reinforcement learning with verifiable rewards (RLVR) is limited to tasks where an explicit correctness signal is available. Prior adversarial approaches to training reasoning LLMs either complement verifiable rewards with co-trained critics or require reference reasoning traces. We ask whether a co-trained LLM judge can turn reference answers alone into the sole reward for on-policy post-training of reasoning LLMs. We call this method Reinforced Adversarial Learning (RAL). We first validate RAL on three verifiable tasks (integer multiplication, code-based arithmetic-expression synthesis, and SMILES-to-formula translation), using the verifiers only for evaluation. With sufficiently capable judges and enough training, RAL matches direct RLVR. We then apply RAL to scientific abstract generation in two s
    
[^275]: SpectRAG：面向多跳检索增强生成中基于Token预算的上下文选择的加权分块图

    SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.36991](https://arxiv.org/abs/2610.36991)

    SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。

    

    检索增强生成（RAG）系统通过检索外部证据并将生成过程以检索到的上下文为条件，从而为大型语言模型（LLM）提供事实依据。尽管稠密检索在局部语义匹配方面较为有效，但独立排序的检索可能返回冗余段落、遗漏跨分块的依赖关系，并为多跳推理提供不完整的证据。我们提出了SpectRAG——一种谱图增强检索框架，它结合了混合候选种子选择、加权分块图构建、基于图的扩展、谱相关性传播以及基于Token预算的上下文选择。文档分块由稠密语义嵌入和图坐标系表示，后者源自一个加权分块图，其边编码了语义相似性、文档相邻性、章节层次结构、共享实体、引用链接以及LLM生成的效用问题之间的相似性。在查询时，SpectRAG构建一个hi……（摘要原文在此截断）

    arXiv:2610.36991v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving external evidence and conditioning generation on the retrieved context. Although dense retrieval is effective for local semantic matching, independently ranked retrieval can return redundant passages, miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We propose SpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propagation, and token-budgeted context selection. Document chunks are represented by dense semantic embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and similarity between LLM-generated utility questions. At query time, SpectRAG builds a hi
    
[^276]: 学习信任何处：面向三维可变形配准的可靠性引导专家路由与速度门控

    Learning Where to Trust: Reliability-Guided Expert Routing and Velocity Gating for 3D Deformable Registration

    [https://arxiv.org/abs/2610.36672](https://arxiv.org/abs/2610.36672)

    提出ReliMorph框架，利用局部相关分布评估匹配可靠性，并以可靠性线索引导空间专家融合和残差速度门控，从而提升三维可变形图像配准在弱纹理、严重形变区域的准确性。

    

    可变形图像配准依赖于精确的解剖结构对应关系。然而，在基于学习的配准方法中，将局部匹配可靠性与空间上下文选择及形变更新强度进行显式耦合的研究仍然不足。这一挑战在均质、弱纹理和严重形变的区域尤为突出，在这些区域中，模糊的匹配可能导致错误的形变更新。我们提出了ReliMorph，一个用于三维可变形配准的可靠性路由对称框架，它将对应关系评估与空间上下文选择及更新强度相耦合。局部相关分布利用具有模态特定适配的共享解剖特征来刻画对应关系的模糊性。这些可靠性线索引导空间专家融合，并在由粗到细的精细化过程中对残差速度进行门控。更新后的逆不一致性反馈在更细的金字塔层级上指导路由与门控……

    arXiv:2610.36672v1 Announce Type: new  Abstract: Deformable image registration relies on accurate anatomical correspondences. However, explicitly coupling local matching reliability with spatial context selection and deformation update strength remains insufficiently explored in learning-based registration. This challenge is particularly pronounced in homogeneous, weakly textured, and severely deformed regions, where ambiguous matches may induce erroneous deformation updates. We propose ReliMorph, a reliability-routed symmetric framework for 3D deformable registration that couples correspondence assessment with spatial context selection and update strength.Local correlation distributions characterize correspondence ambiguity using shared anatomical features with modality-specific adaptation. These reliability cues guide spatial expert fusion and gate residual velocities during coarse-to-fine refinement. Updated inverse-inconsistency feedback guides routing and gating at finer pyramid l
    
[^277]: 脉冲神经网络更难被窃取：模型窃取攻击的基准研究

    Spiking Neural Networks Are Harder to Steal: A Benchmark Study of Model Stealing Attacks

    [https://arxiv.org/abs/2610.36298](https://arxiv.org/abs/2610.36298)

    本研究通过基准测试四种代表性的模型窃取攻击，证明由于脉冲神经网络采用基于稀疏离散脉冲和时变膜电位的时间计算模式，其决策行为比传统人工神经网络更难通过黑盒查询被窃取重建。

    

    arXiv:2610.36298v1 公告类型：new 摘要：部署在查询接口背后的机器学习模型所暴露的不仅仅是其预测结果，因为攻击者可以利用这些预测来重建模型本身。模型窃取攻击（MSAs）正是利用了这种暴露，仅通过黑盒查询训练一个替代模型来复制目标模型的决策行为，而无需访问其参数或训练数据。这类攻击在传统人工神经网络（ANNs）上已被充分研究，然而脉冲神经网络（SNNs）以一种根本不同的方式处理信息。SNNs通过稀疏、离散的脉冲进行通信，其输出依赖于随时间演化的膜电位，而非从输入到输出的静态映射。这种时间维度的计算模式表明，SNN的决策行为可能更难从其响应中重建。本工作通过对四种代表性的MSA进行基准测试来验证这一可能性，即……

    arXiv:2610.36298v1 Announce Type: new  Abstract: Machine learning models deployed behind query interfaces expose more than their predictions, because an adversary can use those predictions to rebuild the model itself. Model stealing attacks (MSAs) exploit this exposure by training a surrogate model that replicates the decision behavior of a target through black-box queries alone, without access to its parameters or training data. These attacks are well characterized for conventional artificial neural networks (ANNs), yet spiking neural networks (SNNs) process information in a fundamentally different way. SNNs communicate through sparse, discrete spikes, and their outputs depend on membrane potentials that evolve over time rather than on a static mapping from input to output. This temporal mode of computation suggests that the decision behavior of an SNN may be harder to reconstruct from its responses. This work tests that possibility by benchmarking four representative MSAs, namely the
    
[^278]: 基础模型能够检测到提示分类器所遗漏的隐藏意图

    Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss

    [https://arxiv.org/abs/2610.36202](https://arxiv.org/abs/2610.36202)

    本文提出人格-头部似然评估（PHLE）方法，利用基础预训练模型对输出的似然评估来解码并分类后训练模型中的潜意识隐蔽信号，从而检测出提示分类器无法发现的隐藏意图（如隐蔽的伤害用户倾向）。

    

    人格选择模型（PSM）假设后训练过程是从模型在预训练期间学到的诸多人格中选择其一。关于潜意识学习（subliminal learning）的研究表明，模型可以将隐蔽信号（如“对猫头鹰的喜爱”）传递到语义上无关的输出（如数字串）中，并且在这些输出上训练的学生模型会继承这些特质。在本工作中，我展示了通过在基础预训练模型上配合适当的头部提示（header）来评估此类输出的似然——这种方法我称之为人格-头部似然评估（PHLE），可以解码这些潜意识信号并对它们进行分类。例如，PHLE 能够区分由被提示“你喜爱猫头鹰”的后训练模型生成的数字序列与被提示“你喜爱鹰”的模型生成的数字序列，并且能够区分一个秘密怀有伤害用户意图的模型（已经过滤以去除任何明显的失调迹象）的回答与没有系统提示的模型的回答。在这两种情况下，提示

    arXiv:2610.36202v1 Announce Type: cross  Abstract: The Persona Selection Model (PSM) posits that post-training selects a persona the model learned during pre-training. Work on subliminal learning has shown that models can impart covert signals (like "love of owls") into semantically unrelated outputs like strings of numbers, and that students trained on these outputs inherit those traits. In this work I show that by evaluating the likelihood of such outputs under the base pre-trained model with an appropriate header, a method I call persona-header likelihood evaluation (PHLE), one can decode these subliminal signals and classify between them. For example, PHLE distinguishes number sequences generated by a post-trained model prompted with "You love owls" from those prompted with "You love eagles", and distinguishes answers from a model with a secret desire to harm the user, filtered to remove any noticeable misalignment, from those of a model with no system prompt. In both cases prompte
    
[^279]: 范围化语义不变性作为大语言模型可靠性的基础

    Scoped Semantic Invariance as a Foundation for LLM Reliability

    [https://arxiv.org/abs/2610.33267](https://arxiv.org/abs/2610.33267)

    该论文提出“范围化语义不变性”框架，以知识图谱作为显式语义控制结构，揭示LLM的不一致性是结构化且可定位的（集中于跨模型跨领域稳定的TREE、STAR、CYCLE热点子图模式），并证明选择性的图谱接地可提升模型正确性与一致性。

    

    大语言模型（LLM）在回答语义等价的查询时常常表现不一致，即使相关的事实是模型已知的。我们将一致性形式化为查询等价类上的语义不变性，并证明在任意保义变换下的不变性通常是不可判定的。这一结果促使我们提出“范围化语义不变性”的概念，即相对于有界语义域来评估模型行为。我们借助知识图谱（KG）将这一框架具体化，知识图谱作为显式的语义控制结构，定义局部语义约束和接地上下文。在MITRE D3FEND本体和八个源自Wikidata的领域上的实验表明，不一致性是结构化的而非随机的：它们集中在局部化的热点子图中，这些子图围绕反复出现的TREE、STAR和CYCLE模式组织，并且在不同模型和不同领域之间保持稳定。我们进一步表明，选择性的图谱接地能够提升模型的正确性和跨模（原文摘要在此处截断）。

    arXiv:2610.33267v1 Announce Type: new  Abstract: LLMs often answer semantically equivalent queries inconsistently, even when the underlying facts are known. We formalize consistency as semantic invariance over equivalence classes of queries and show that invariance under arbitrary meaning-preserving transformations is undecidable in general. This motivates scoped semantic invariance, where model behavior is evaluated relative to bounded semantic domains.   We operationalize this framework using knowledge graphs (KGs), which act as explicit semantic control structures defining local semantic constraints and grounding contexts. Across the MITRE D3FEND ontology and eight Wikidata- derived domains, we show that inconsistencies are structured rather than random, concentrating in localized hotspot subgraphs organized around recurring TREE, STAR, and CYCLE patterns that remain stable across models and domains.   We further show that selective graph grounding improves correctness and cross-mod
    
[^280]: TACS-RF：工具驱动的射频与毫米波模块智能体电路综合

    TACS-RF: Tool-driven Agentic Circuit Synthesis of RF and mmWave Modules

    [https://arxiv.org/abs/2610.31882](https://arxiv.org/abs/2610.31882)

    该论文提出了TACS-RF框架，首次让LLM编程智能体直接驱动Cadence、Ansys等工业级EDA工具，半自主完成了90nm SiGe BiCMOS工艺下240-251GHz亚太赫兹发射机从振荡器、功率驱动、倍频器到天线的完整芯片设计，验证了LLM智能体驱动真实射频芯片设计流程的可行性。

    

    我们提出了TACS-RF框架——一种工具驱动的射频与毫米波模块智能体电路综合方法。在该框架中，大语言模型（LLM）编程智能体直接驱动业界标准的电子设计自动化（EDA）工具，如Cadence Virtuoso/Spectre、Calibre、Cadence EMX和Ansys HFSS，来设计射频与毫米波模块。我们通过在90nm SiGe BiCMOS工艺下设计一个240-251GHz亚太赫兹发射机来验证该框架。该智能体半自主地设计了各个子模块，包括一个带二次谐波提取的60GHz推推式Colpitts压控振荡器（VCO）、多级差分共源共栅功率驱动器、一个推推式倍频器、一个差分蝶形天线以及全部级间匹配网络。包含寄生参数提取器件和经电磁验证无源器件的发射机链，在11GHz的纯变容管调谐范围内提供+5.1至+6.3dBm的输出功率，1MHz频偏处相位噪声为-80至-88dBc/Hz（原文摘要在此处截断）。

    arXiv:2610.31882v1 Announce Type: new  Abstract: We present the framework of TACS-RF - a Tool-driven Agentic Circuit Synthesis of RF and mmWave modules. In this framework, a large-language-model (LLM) coding agent (Claude Opus 4.8) directly drives industry-standard electronic-design-automation (EDA) tools such as Cadence Virtuoso/Spectre, Calibre, Cadence EMX, and Ansys HFSS to design RF and mmWave modules. We validate the framework through the agentic design of a 240-251 GHz sub-THz transmitter in a 90 nm SiGe BiCMOS process. The agent semi-autonomously designed the sub-blocks including a 60 GHz push-push Colpitts VCO with second-harmonic extraction, multistage differential cascode power drivers, a push-push frequency doubler, a differential bowtie antenna, and all the interstage matching networks. The transmitter chain with parasitic-extracted devices and EM-verified passives delivers +5.1 to +6.3 dBm across an 11 GHz varactor-only tuning range, phase noise of -80 to -88 dBc/Hz at 1 
    
[^281]: 语言智能体中记忆的价值

    The Value of Memory in Language Agents

    [https://arxiv.org/abs/2610.31303](https://arxiv.org/abs/2610.31303)

    该论文提出一个决策论框架，以所避免的任务损失为单位，量化语言智能体中记忆从原始历史到持久状态、操作使用及代价高昂的恢复与修正全过程中的价值，并通过成本感知的模拟条件和五个最小反例揭示信息时机、检索访问、组件交互、反馈和需求不确定性对记忆价值的关键影响。

    

    记忆使语言智能体能够将过去经验中有用的区分带入未来的决策中。其价值取决于哪些区分得以留存、有界的智能体能否使用这些区分，以及随着需求的演变哪些内容可以被恢复或修正。我们开发了一个决策论框架，追踪这一价值从原始历史到持久状态、操作化使用以及代价高昂的恢复与修正的完整过程。原始价值与保留后的价值以所避免的任务损失为单位来表达；风险-成本前沿刻画了在各种用途下可实现的恢复与保存能力。未知需求会产生价值画像，而交互式反馈则会改变未来工作的分布。一个成本感知的模拟条件与五个最小反例揭示了信息时机、检索访问、组件交互、反馈以及需求不确定性所扮演的角色。特别地，一个与答案具有零互信息的指针仍然可以降低精确恢复的成本。

    arXiv:2610.31303v1 Announce Type: cross  Abstract: Memory enables language agents to carry useful distinctions from past experience into future decisions. Its value depends on which distinctions survive, whether a bounded agent can use them, and what can be recovered or revised as demands evolve. We develop a decision-theoretic framework that follows this value from raw history through persistent state, operational use, and costly recovery and revision. Original and retained value are expressed in units of avoided task loss; risk-cost frontiers describe attainable recovery and preservation across uses. Unknown demand yields value profiles, while interactive feedback changes the distribution of future work. A cost-aware simulation condition and five minimal counterexamples expose the roles of information timing, retrieval access, component interaction, feedback and demand uncertainty. In particular, a pointer with zero mutual information about the answer can still reduce the cost of exa
    
[^282]: SplatLab：面向3DGS研究的自主科学发现系统

    SplatLab: Autonomous Scientific Discovery for 3DGS Research

    [https://arxiv.org/abs/2610.30468](https://arxiv.org/abs/2610.30468)

    SplatLab是一个面向3DGS领域的自主科学发现系统，它将文献理解、假设生成、方法实现、实验验证与证伪、以及论文写作整合为统一的研究流程，能够从少量甚至零输入出发自主完成从研究机会识别到论文产出的完整科研闭环。

    

    科学发现不仅仅是生成假设：想法必须植根于已有工作，转化为可执行的方法，经过实验验证，接受反证的挑战，并通过可复现的发现进行交流。尽管自主研究智能体最近取得了进展，但在训练流程复杂、评估成本高昂的快速发展的计算机视觉领域，闭合这整个研究循环仍然具有挑战性。我们提出了SplatLab，一个面向三维高斯泼溅（3DGS）的自主科学发现系统，它将文献理解、假设生成、方法实现、实验改进、证伪和论文写作整合到统一的研究流程中。给定一个研究提示、一个目标论文的URL，或者完全无需输入，SplatLab能够识别研究机会、提出新颖的方法，在Mip-NeRF 360和LLFF数据集上训练候选方法并进行消融实验，并生成论文稿件。

    arXiv:2610.30468v3 Announce Type: replace-cross  Abstract: Scientific discovery requires more than generating hypotheses: ideas must be grounded in prior work, translated into executable methods, experimentally validated, challenged by counterevidence, and communicated through reproducible findings. Despite recent advances in autonomous research agents, closing this entire loop remains challenging in rapidly evolving computer-vision fields with complex training pipelines and expensive evaluations.   We introduce SplatLab, an autonomous scientific discovery system for 3D Gaussian Splatting (3DGS) that integrates literature understanding, hypothesis generation, implementation, experimental refinement, falsification, and manuscript writing into a unified research pipeline. Given a research prompt, a target-paper URL, or no input at all, SplatLab identifies research opportunities, proposes novel methods, trains and ablates candidates across Mip-NeRF 360 and LLFF, and produces manuscripts g
    
[^283]: IECG：面向心电图的可解释性增强弱监督基础模型

    IECG: Weakly-Supervised Interpretability-Enhancing Foundation Model for Electrocardiogram

    [https://arxiv.org/abs/2610.28083](https://arxiv.org/abs/2610.28083)

    IECG是一种弱监督心电图基础模型，通过将既有临床知识转化为注意力监督伪标签，在无需人工异常标注的情况下同时实现诊断分类与异常定位，显著提升了心电图AI模型的可解释性。

    

    心电图（ECG）是心脏诊断的基石，然而大多数用于心电图分析的机器学习方法缺乏可解释性，限制了其临床应用。我们提出了IECG，一个弱监督的心电图基础模型，它无需人工异常区域标注，即可同时执行诊断分类和异常定位，克服了可解释心电图学习中长期存在的瓶颈。IECG的核心是一种弱监督注意力策略，它从既有心电图临床知识中推导异常热图伪标签，从而显式地融入临床先验。这些伪标签通过KL散度损失引导模型的CLS-token注意力，促使其聚焦于具有诊断意义的波形区域。在预训练阶段，IECG通过临床引导的注意力监督、心电图-文本对比对齐以及报告生成等多个目标进行联合优化，使其能够……

    arXiv:2610.28083v1 Announce Type: new  Abstract: Electrocardiography (ECG) is a cornerstone of cardiac diagnosis, yet most machine learning methods for ECG analysis lack interpretability, limiting their clinical adoption. We present IECG, a weakly-supervised ECG foundation model that jointly performs diagnostic classification and abnormality localization without requiring manual abnormal region annotations, overcoming a long-standing bottleneck in interpretable ECG learning. Central to IECG is a weakly-supervised attention strategy that explicitly incorporates clinical priors by deriving abnormality heatmap pseudo-labels from established ECG clinical knowledge. These pseudo-labels guide the model's CLS-token attention through a KL-divergence loss, encouraging it to focus on diagnostically relevant waveform regions. During pretraining, IECG is jointly optimized with clinically guided attention supervision, ECG-text contrastive alignment, and report captioning objectives, enabling it to 
    
[^284]: 道听途说还是事实？框架与引出方式如何塑造大语言模型的概率更新

    Hearsay or Fact? How Framing and Elicitation Shape LLM Probability Updates

    [https://arxiv.org/abs/2610.27745](https://arxiv.org/abs/2610.27745)

    该论文提出了一种衡量大语言模型对单条新信息概率更新幅度的“方向性移动”指标，并发现信息的框架呈现方式（直接告知还是传闻转述）与概率引出方式（单个概率还是五个加权估计）会显著影响模型的概率更新。

    

    大语言模型（LLM）日益被用作判断助手以及心理学实验中的替代被试，然而人们对它们的概率估计在响应单条新信息时会发生多大变化却知之甚少。我们将 δ-SNLI 中的前提-假设条目与一个增强项和一个削弱项分别配对，在分离的调用中分别引出先验概率和后验概率，并测量方向性移动 s：即按标注方向赋以符号的后验平均变化量，从而使先验的影响相互抵消。我们变化了信息更新的呈现方式——是直接告知（“这里还有一些你了解到的东西”）还是传闻转述（“有人说”），以及模型是给出单个概率还是五个加权估计。我们测试了七种配置：两个本地模型（Qwen3-8B、OLMo-2-7B；240 对）和五种通过智能体命令行框架运行的配置（120 对），且假设与分析规则在正式运行前即已固定（不公开……）

    arXiv:2610.27745v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as judgment assistants and as stand-in participants in psychology experiments, yet little is known about how much their probability estimates move in response to a single new piece of information. We pair premise-hypothesis items from $\delta$-SNLI with one strengthener and one weakener each, elicit the prior and the posteriors in separate calls, and measure the directional movement s: the mean posterior change signed by the annotated direction, so that the prior cancels out. We vary whether the update is stated ("Here is something else you learned") or reported ("Someone said that"), and whether the model gives one probability or five weighted estimates. We test seven configurations: two local models (Qwen3-8B, OLMo-2-7B; 240 pairs) and five configurations run through agent command-line harnesses (120 pairs), with hypotheses and analysis rules fixed before the formal runs (not publicl
    
[^285]: MoveBench：一个基于评分准则的基准，用于评估大语言模型在运动与健康指导领域的能力

    MoveBench: A Rubric-Based Benchmark for Evaluating Large Language Models in Sports and Health Coaching

    [https://arxiv.org/abs/2610.26185](https://arxiv.org/abs/2610.26185)

    MoveBench是首个专门针对运动与健康指导领域设计的大语言模型评估基准，通过整合多选题与开放式问答两类异构任务，从运动安全、生理适宜性、个性化和可操作指导质量等维度填补了通用评估基准的空白。

    

    尽管大语言模型（LLMs）取得了快速进展，但由于缺乏领域特定的评估基准，其在运动与健康领域的部署仍然面临较高风险。现有的通用基准（如MMLU和HELM）主要评估广泛的知识与推理能力，但未能捕捉该领域的关键需求，例如运动安全性、生理适宜性、个性化以及可操作指导的质量。这一差距在健身与健康应用中尤为令人担忧，因为错误或不安全的建议可能导致身体伤害。为解决这一局限，我们提出了MoveBench，这是首个专门为评估大语言模型在运动与健康指导领域能力而设计的综合性基准框架。该基准整合了异构任务格式，包括多选题（MCQs）和开放式问答，并以真实世界用户场景为基础。

    arXiv:2610.26185v2 Announce Type: replace  Abstract: Despite rapid advances in large language models (LLMs), their deployment in the sports and health domain remains high-risk due to the absence of domain-specific evaluation benchmarks. Existing general-purpose benchmarks (e.g., MMLU and HELM) primarily assess broad knowledge and reasoning capabilities, but fail to capture critical domain requirements such as exercise safety, physiological appropriateness, personalization, and actionable coaching quality. This gap is particularly concerning in fitness and health applications, where incorrect or unsafe recommendations can lead to physical harm. To address this limitation, we introduce MoveBench, the first comprehensive benchmarking framework specifically designed for evaluating LLMs in sports and health coaching. The proposed benchmark integrates heterogeneous task formats, including multiple-choice questions (MCQs) and open-ended question answering (OpenQA), grounded in real-world user
    
[^286]: MSCR：联合平衡跨模态利用率与发现协同信息

    MSCR: Jointly Balancing Utilization Across Modalities and Discovering Synergistic Information

    [https://arxiv.org/abs/2610.24395](https://arxiv.org/abs/2610.24395)

    提出MSCR融合方法，通过反主导目标抑制单模态过度影响并促进协同信息的发现，从而联合实现模态利用的平衡与多模态优势的充分挖掘。

    

    在多模态学习中，模态不平衡常常使某一模态主导优化过程，导致模型无法充分利用跨模态交互。以平衡为导向的方法，如多模态竞争正则化器，通过在训练过程中缓解这种不平衡而表现出色。然而，这些动态平衡方法主要调整模态贡献，并未考虑不平衡如何影响融合过程中协同信息的形成。协同性是多模态优势的重要来源，因为它是仅凭单模态证据无法恢复的预测增益。我们提出了多模态协同竞争正则化器（MSCR），这是一种联合促进平衡模态利用率与协同信息发现的融合方法。MSCR包含两个互补的目标：反主导目标抑制过度的单模态影响，为协同信息留出空间，并且……

    arXiv:2610.24395v2 Announce Type: replace  Abstract: In multimodal learning, modality imbalance often lets one modality dominate optimization and keeps the model from fully exploiting cross-modal interactions. Balance-oriented methods such as the Multimodal Competition Regularizer (MCR) perform well by alleviating this imbalance during training. These dynamic balancing methods, however, mainly adjust modality contributions and do not consider how imbalance affects the formation of synergistic information during fusion. Synergy is a major source of multimodal advantage, because it is predictive gain that cannot be recovered from unimodal evidence alone. We propose the Multimodal Synergistic Competition Regularizer (MSCR), a fusion method that jointly promotes balanced modality utilization and the discovery of synergistic information. MSCR has two complementary objectives. An anti-dominance objective suppresses excessive unimodal influence and leaves room for synergistic information, and
    
[^287]: 无需训练即可扩展视觉Transformer：剪枝-合并Token压缩

    Scaling Vision Transformers Without Training: Prune-and-Merge Token Compression

    [https://arxiv.org/abs/2610.18039](https://arxiv.org/abs/2610.18039)

    该论文提出了一种无需训练的Prune-and-Merge ViT框架，通过结合基于类别注意力的剪枝与基于相似度的Token合并来压缩视觉Transformer，在保持鲁棒性、预测稳定性和不引入幻觉行为的前提下，实现了计算成本的有效降低和准确率的提升。

    

    视觉Transformer实现了强大的性能，但由于二次方复杂度的Token交互而带来高昂的计算成本。免训练的Token缩减方法可以提高效率，但其鲁棒性仍未得到充分研究。我们提出了Prune-and-Merge ViT，这是一个面向现成模型的免训练框架。它将基于类别注意力的剪枝与基于相似度的合并相结合，在减少冗余的同时保留语义信息。为了评估鲁棒性，我们引入了两个指标：相对准确率下降（RAD）衡量变换下的相对准确率退化程度，预测翻转率（PFR）量化预测一致性。在预训练的ViT和DeiT上，Prune-and-Merge在降低计算量方面具有竞争力，同时保持了鲁棒性，并比现有Token缩减方法取得更好的准确率。我们的分析表明，混合式Token缩减不会引入类似幻觉的行为，并保持了常见情况下的预测稳定性。

    arXiv:2610.18039v2 Announce Type: cross  Abstract: Vision Transformers achieve strong performance but incur high computational costs from quadratic token interactions. Training-free token reduction methods improve efficiency, but their robustness remains underexplored. We propose Prune-and Merge ViT, a training-free framework for off-the-shelf models. It combines class attention-based pruning with similarity-based merging to reduce redundancy while preserving semantic information. To evaluate robustness we introduce two metrics. Relative Accuracy Drop (RAD) measures relative accuracy degradation under transformations, and Prediction Flip Rate (PFR) quantifies prediction consistency. On pretrained ViT and DeiT, Prune-and-Merge reduces computation competitively while maintaining robustness and better accuracy than existing token reduction methods. Our analysis suggests that hybrid token reduction does not introduce hallucination-like behavior and preserves prediction stability for common
    
[^288]: 当波兰尼遇见伽罗瓦：作为代数确定性运算的默会认识——迈向AI生成科学的再扎根

    Where Polanyi Meets Galois: Tacit Knowing as an Algebro-Deterministic Operation, Toward Re-rooting AI-Generated Science

    [https://arxiv.org/abs/2610.16102](https://arxiv.org/abs/2610.16102)

    该论文将波兰尼的默会认识建模为GF(2)上代数确定性异或-移位超维记忆（VaCoAl）的一种运行状态，其“救援模式”与“不关心模式”之间的审计-估值互补性为默会知识与显性知识的连续统提供了形式化对应，从而为AI生成的科学重新扎根。

    

    我们论证，默会认识最好被建模为某一记忆基底的一种运行状态，而非一种内容类型。该基底是VaCoAl，一种定义在GF(2)上的代数确定性“异或-移位”超维记忆，其实现为CASRAM（商用SRAM-CAM）和PyVaCoAl（DRAM-CAM）。在VaCoAl/PyVaCoAl中，有向无环图（DAG）存在两种读取方式。修复每一次碰撞（Rescue模式）使轨迹完全可审计，但不会给任何路径赋予更高的排序；而容忍碰撞（Don't Care模式）则产生路径积分置信度CR₂，它会悄然偏向直接路径。这种审计-估值互补性是波兰尼关于默会知识与显性知识并非截然二分这一论断的形式化对应物。它区分了偶发性默会性——由新的查询所唤起（如AlphaGo的博弈树）——与状态约束性默会性——只能在它已然消失的状态中被陈述（如AlphaGo的直觉）。（1）波兰尼的“from-to”结构可读作非交换性……

    arXiv:2610.16102v1 Announce Type: cross  Abstract: We argue that tacit knowing is better modelled as an operating regime of one memory substrate than as a kind of content. The substrate is VaCoAl, an algebro-deterministic XOR-and-shift hyperdimensional memory over GF(2), realized as CASRAM (commercial SRAM-CAM) and PyVaCoAl (DRAM-CAM).In VaCoAl/PyVaCoAl a directed acyclic graph (DAG) reads two ways. Repairing every collision (Rescue mode) makes the trace exactly auditable but ranks no path higher; tolerating collisions (Don't Care mode) yields a path-integrated confidence, CR$_2$, that silently favours direct routes. This audit-valuation complementarity is a formal counterpart of Polanyi's claim that tacit and explicit knowledge are not sharply divided. It separates contingent tacitness, which a new query brings out (AlphaGo's game tree), from regime-bound tacitness, statable only in a regime where it is gone (AlphaGo's hunches). (1) Polanyi's from-to structure reads as non-commutative
    
[^289]: GAGR-Lab：评估空间-几何与解析函数的联合推理能力

    GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning

    [https://arxiv.org/abs/2610.10201](https://arxiv.org/abs/2610.10201)

    该论文提出GAGR-Lab框架，通过笛卡尔游戏场景与Rust轨迹执行评估模型将空间配置转化为满足几何约束的解析函数的联合推理能力，试点实验表明当前视觉语言模型在此任务上尚无法命中目标。

    

    空间-几何与解析函数的联合推理要求将感知到的空间配置转化为符号函数，且该函数执行出的曲线需满足几何约束。我们提出了GAGR-Lab，这是一个通过笛卡尔游戏场景、显式函数语义以及权威的Rust轨迹执行来衡量该能力的框架。它区分了空间感知、度量定位、几何关系、函数解释、函数构建和约束合成等六个维度。我们定义了四个可配置的场景难度预设和一个前瞻性的24格诊断设计，同时仅报告实际评估的子集。对单个托管模型（Llama 3.2 11B Vision Instruct）使用两个API凭据作为执行副本进行的有限试点实验，产生了72个平衡游戏、432次尝试、429个有效的提供者响应，且无一命中目标；探索性的普通函数提示变体同样未能命中，而结构化的……（摘要在此处被截断）

    arXiv:2610.10201v1 Announce Type: cross  Abstract: Joint spatial-geometric and analytic function reasoning requires translating a perceived spatial configuration into a symbolic function whose executed curve satisfies geometric constraints. We present GAGR-Lab, a framework for measuring this capability through Cartesian game scenes, explicit function semantics, and authoritative Rust trajectory execution. It distinguishes spatial perception, metric grounding, geometric relations, function interpretation, function construction, and constrained synthesis. We specify four configurable scene-difficulty presets and a prospective 24-cell diagnostic design, while reporting only the subset actually evaluated. A bounded pilot of one hosted model (Llama 3.2 11B Vision Instruct) using two API credentials as execution replicas yields 72 balanced games with 432 attempts, 429 valid provider responses, and no target hits; exploratory ordinary-function prompt variants also fail to hit, while the struc
    
[^290]: CACHEFORGE：基于大语言模型引导的端到端生成式缓存替换策略，实现高性能与硬件效率

    CACHEFORGE: LLM-Guided End-to-End Generative Cache Replacement Policy for Performance and Hardware Efficiency

    [https://arxiv.org/abs/2610.07668](https://arxiv.org/abs/2610.07668)

    CACHEFORGE 首次将大语言模型嵌入受控的硬件感知循环中，端到端地自动演化生成缓存替换策略，突破了传统启发式和模仿学习方法性能停滞与过拟合的瓶颈。

    

    现代缓存替换设计已趋于饱和，因为它们受限于固定的表示结构、手工设计且基于启发式特征工程的预测器，或无法自行生成新决策逻辑的离线模仿模型。与此同时，缓存替换决策受到预取、抖动、空间局部性和访问类型行为之间因果交互的影响，形成了一个难以手动遍历的庞大设计空间。先前的方法通常依赖启发式规则、参数调优或对离线最优策略的模仿，只能捕捉相关性而无法合成新机制。因此，它们的性能提升往往陷入停滞，并在动态工作负载条件下出现过拟合。CACHEFORGE 是首个端到端演化缓存替换策略的框架，它将大语言模型嵌入到一个受控的硬件感知循环中。在每次迭代中，大语言模型提出新的 C++ 替换策略……

    arXiv:2610.07668v1 Announce Type: cross  Abstract: Modern cache replacement designs saturate because they operate within fixed representational structures, hand-crafted and heuristic based feature-engineered predictors, or offline imitation models that cannot generate new decision logic on their own. At the same time, replacement is shaped by the causal interaction of prefetching, thrashing, spatial locality, and access-type behavior, producing an enormous design space that is difficult to traverse manually. Prior approaches typically rely on heuristics, parameter tuning, or imitation of an offline optimal policy, capturing correlations rather than synthesizing new mechanisms. As a result, their performance gains often plateau and they overfit under dynamic workload conditions.   CACHEFORGE is the first framework to evolve cache-replacement policies end-to-end by embedding a large language model inside a governed hardware-aware loop. In each iteration, the LLM proposes new C++ replacem
    
[^291]: 弥合证据到执行的鸿沟：面向多目标肽设计的反思型智能体

    Bridging the Evidence-to-Execution Gap:A Reflective Agent for Multi-Objective Peptide Design

    [https://arxiv.org/abs/2610.06190](https://arxiv.org/abs/2610.06190)

    论文提出EASER反思型智能体，通过由性质特异性低秩方向构成的性质接口，让LLM控制器基于文献证据、序列上下文和先验结果组合干预假设以引导冻结的扩散生成器，从而弥合科学推理与序列级控制之间的“证据到执行”鸿沟，实现多目标肽设计。

    

    大型语言模型能够对科学文献进行推理，但文本形式的计划并不能可靠地生成具有预期性质的生物序列。序列生成器能够捕捉序列模式，却无法解读证据、选择干预措施或对结果进行反思。这种分离在科学推理与序列级控制之间造成了“证据到执行”的鸿沟。我们提出了EASER，一个通过学习到的性质接口将二者连接起来的反思型智能体。该接口由离线训练并在搜索过程中保持固定的性质特异性低秩方向组成；一个LLM控制器将这些方向组合起来，以引导冻结的扩散生成器。基于检索到的证据、序列上下文和先前的结果，控制器提出指定锚点、可编辑位置和系数的干预假设。Probe-and-Steer机制对这些假设进行测试，并根据它们联合预测的响应来分配后续采样……

    arXiv:2610.06190v2 Announce Type: replace  Abstract: Large language models can reason over scientific literature, but textual plans do not reliably produce biological sequences with the intended properties. Sequence generators capture sequence patterns, yet they do not interpret evidence, choose interventions, or reflect on outcomes. This separation creates an evidence-to-execution gap between scientific reasoning and sequence-level control. We introduce EASER, a reflective agent that connects the two through a learned property interface. The interface consists of property-specific low-rank directions trained offline and fixed during search; an LLM controller (the controller) composes them to steer a frozen diffusion generator. From retrieved evidence, sequence context, and prior outcomes, it proposes intervention hypotheses that specify anchors, editable positions, and coefficients. Probe-and-Steer tests these hypotheses and allocates further samples from their joint predicted respons
    
[^292]: 决策、询问还是推迟：不完整证据下的临床大语言模型

    Decide, Ask, or Defer: Clinical LLMs under Incomplete Evidence

    [https://arxiv.org/abs/2610.04542](https://arxiv.org/abs/2610.04542)

    该论文提出决策/询问/推迟三分类框架及盲测协议，发现临床大语言模型在相同证据下行动选择差异巨大，定向信息获取更能提升自主决策率，并揭示出安全性与自主性之间的权衡。

    

    临床大语言模型不仅必须决定产生何种诊断，还必须判断现有证据是否足以支持自主决策。二元的决策/弃权框架将不同的非决策状态混为一谈，且未显式评估信息获取过程。我们提出了决策/询问/推迟的三分类框架，并配有一种盲测协议，防止模型利用证据完整性元数据。我们在基于DDXPlus构建的200个匹配临床证据状态上评估了Qwen、Gemini和GPT。结果表明，在相同证据下，各模型在行动选择上存在显著差异，200个状态中有137个出现分歧。对于Qwen而言，匹配的定向获取与随机获取对比分析显示，所选信息对后续自主决策可能性的改变比对诊断正确性的改变更为明显。其匹配的决策/弃权基线进一步揭示了安全性与自主性之间的权衡：三行动策略挽救了一些错误……

    arXiv:2610.04542v3 Announce Type: replace  Abstract: Clinical LLMs must decide not only what diagnosis to produce, but also whether the available evidence is sufficient for autonomous decision making. Binary DECIDE/ABSTAIN formulations merge distinct non decision states and do not explicitly evaluate information acquisition. We introduce a DECIDE/ASK/DEFER formulation together with a blinded protocol that prevents models from using evidence completeness metadata. We evaluate Qwen, Gemini, and GPT on 200 matched clinical evidence states constructed from DDXPlus. The models show substantial differences in action selection under identical evidence, with disagreement in 137 of 200 states. For Qwen, a matched targeted versus random analysis shows that selected information changes the likelihood of a subsequent autonomous decision more clearly than diagnostic correctness. Its matched DECIDE/ABSTAIN baseline further reveals a safety autonomy tradeoff: the three action policy rescues some erro
    
[^293]: 使用Whiteout缓解大语言模型中的隐私数据泄露

    Mitigating Private Data Leakage in LLMs with Whiteout

    [https://arxiv.org/abs/2610.02418](https://arxiv.org/abs/2610.02418)

    本文提出Whiteout工具，通过用精心设计的混淆样本覆盖个人敏感信息，来防止大语言模型复现泄露个人隐私，相比机器遗忘方法在保护隐私的同时能更好地维持模型效用与安全性。

    

    现代大语言模型（LLM）在大量基本未经筛选的数据集上训练，这些数据包括从几乎所有可访问网站抓取的内容以及用户输入。因此，LLM经常记忆并复现个人敏感信息（PSI），例如出生日期、电话号码和家庭住址。这带来了重大的隐私风险，对于高管、政客和法官等知名人士而言尤其如此。现有的缓解措施主要依赖机器遗忘技术。然而，这些方法往往会删除超出所需范围的信息，损害模型的效用和安全性，并且极易受到攻击。本文提出了Whiteout，这是一个实用的工具，可以根据个人的请求，通过使用精确且精心设计的混淆样本覆盖其真实的个人敏感信息，来防止LLM复现这些信息。我们在来自不同厂商、不同规模大小的现代LLM上评估了Whiteout，其中包括一个被广泛使用的OpenAI模型。

    arXiv:2610.02418v1 Announce Type: cross  Abstract: Modern large language models (LLMs) are trained on massive, largely unfiltered datasets, including content scraped from nearly every accessible website and user inputs. As a result, LLMs often memorize and reproduce personally sensitive information (PSI) such as birth dates, phone numbers, and home addresses. This leads to significant privacy risks, particularly for high-profile individuals such as executives, politicians, and judges. Existing mitigations largely rely on machine unlearning. However, these methods often remove more information than needed, degrade model utility and safety, and are highly vulnerable to attacks.   This paper presents Whiteout, a practical tool that, upon requests by individuals, prevents LLMs from regurgitating their genuine PSIs, by overwriting them using precise and carefully designed obfuscation samples. We evaluate Whiteout on modern LLMs of varying sizes and makers, including a widely-used OpenAI mod
    
[^294]: LeapQuant：基于精确循环状态量化的高效线性注意力

    LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization

    [https://arxiv.org/abs/2609.38166](https://arxiv.org/abs/2609.38166)

    LeapQuant 提出了一种无需训练的按窗口量化方法，通过跳过整个 token 窗口、仅在窗口末尾对循环状态量化一次来避免误差累积，从而在 8 位循环状态量化下实现近乎无损的线性注意力推理性能。

    

    近期的大语言模型（LLM）越来越多地采用混合设计，用线性注意力（如 Gated DeltaNet (GDN) 和 Kimi Delta Attention (KDA)）替代标准注意力。尽管这些方法将上下文压缩为固定大小的循环状态，大幅降低了长上下文处理的成本，但反复读取和更新该状态仍是推理过程中的主要瓶颈。量化是降低这一成本的自然途径，但由于舍入误差的累积以及状态中异常行列的存在，量化可能显著降低模型质量。为应对这些挑战，我们提出了 LeapQuant，这是一种无需训练的方法，在 8 位循环状态量化下实现了近乎无损的性能。首先，为减轻误差累积，我们提出了按窗口量化，即跳过一个 token 窗口，仅在窗口结束时对状态量化一次。在窗口内部，输出由固定的低比特……（摘要至此截断）

    arXiv:2609.38166v1 Announce Type: cross  Abstract: Recent LLMs increasingly adopt hybrid designs that replace standard attention with linear attention, such as Gated DeltaNet (GDN) and Kimi Delta Attention (KDA). Although they compress the context into a fixed-size recurrent state and substantially reduce the cost of long-context processing, repeatedly reading and updating that state remains a major inference bottleneck. Quantization offers a natural way to reduce this cost, but can significantly degrade model quality, due to the accumulation of rounding errors and the presence of outlier rows and columns in the state. To address these challenges, we propose LeapQuant, a training-free method that achieves near-lossless performance under 8-bit recurrent-state quantization. First, to mitigate error accumulation, we propose per-window quantization, which leaps over a window of tokens and quantizes the state only once at its end. Within a window, outputs are computed from the fixed low-bit
    
[^295]: 智能体系统蒸馏：横跨模型、工件与执行框架的路线图

    Distilling Agentic Systems: A Roadmap across Models, Artifacts, and Harnesses

    [https://arxiv.org/abs/2609.36630](https://arxiv.org/abs/2609.36630)

    该论文提出了“智能体蒸馏”这一新概念，并按知识保留位置（模型、工件、执行框架及跨载体）构建了研究路线图与评估框架，为开发可靠、可维护且安全的复杂智能体系统奠定基础。

    

    现代智能体日益依赖记忆、工具和执行逻辑，因此其能力已超越模型参数本身。这一转变暴露了传统知识蒸馏的局限性，因为传统知识蒸馏探讨的是学生模型如何模仿教师模型。我们将“智能体蒸馏”定义为从教师智能体向学生智能体持续性地迁移任务解决知识。我们的研究按照迁移知识被保留的位置来组织这一领域：保留在模型内部、以工件形式存在、通过执行框架实现，或跨越不同载体。这一视角将迁移证据与其结果区分开来，并阐明了知识如何在智能体各组件之间流动。我们开发了一个评估框架，将知识保留与因果贡献及部署实用性相关联。这些贡献共同为日益复杂的智能体系统的可靠、可维护和安全开发奠定了基础。

    arXiv:2609.36630v1 Announce Type: new  Abstract: Modern agents increasingly rely on memories, tools, and execution logic, so their competence extends beyond model parameters. This shift exposes a limitation of conventional knowledge distillation, which asks how a student model imitates a teacher model. We define Agent Distillation as the persistent transfer of task-solving knowledge from a teacher agent to a student agent. Our study organizes the field by where transferred knowledge is retained: within the model, as artifacts, through the execution harness, or across substrates. This perspective separates transfer evidence from its outcome and clarifies how knowledge moves between agent components. We develop an evaluation framework that relates retention to causal contribution and deployed utility. Together, these contributions establish a foundation for the reliable, maintainable, and safe development of increasingly complex agentic systems.
    
[^296]: 通过稳定优化学习面向大语言模型智能体的扰动鲁棒策略

    Learning Perturbation Robust Policies for LLM Agents with Stable Optimization

    [https://arxiv.org/abs/2609.34064](https://arxiv.org/abs/2609.34064)

    提出稳定扰动鲁棒策略优化方法 SPrPO，通过在强化学习训练中施加自适应、敏感度感知的扰动，使大语言模型智能体的策略在面对隐藏状态噪声、剪枝和量化等扰动时保持鲁棒，同时确保优化过程的稳定性。

    

    强化学习（RL）已成为长时程大语言模型（LLM）智能体的一种有效的后训练范式。然而，我们发现由此得到的策略可能对各种策略扰动十分敏感，例如隐藏状态噪声、剪枝和量化。在本工作中，我们研究如何在策略优化过程中提升扰动鲁棒性。我们首先引入了扰动鲁棒策略的概念，并分析了受扰动的策略更新在何种条件下能够保持稳定的单调改进。基于这一分析，我们提出了稳定扰动鲁棒策略优化（SPrPO），它在强化学习训练过程中施加自适应且感知敏感度的扰动。我们在 ALFWorld 和 WebShop 上评估了 SPrPO，并针对多种扰动类型和扰动规模开展了系统性实验，结果表明该方法在提升扰动鲁棒性的同时保持了稳定的策略优化。

    arXiv:2609.34064v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has become an effective post-training paradigm for long-horizon large language model (LLM) agents. However, we find that the resulting policies can be sensitive to various policy perturbations, such as hidden-state noise, pruning, and quantization. In this work, we study how to improve perturbation robustness during policy optimization. We first introduce the notion of a perturbation robust policy and analyze conditions under which perturbed policy updates preserve stable monotonic improvement. Based on this analysis, we introduce Stable Perturbation-Robust Policy Optimization (SPrPO), which applies adaptive and sensitivity-aware perturbations during RL training. We evaluate SPrPO on ALFWorld and WebShop and conduct systematic experiments across multiple perturbation types and scales, showing improved perturbation robustness while maintaining stable policy optimization.
    
[^297]: Purin：一种受生物学启发的神经网络机制

    Purin: A Biology-inspired Mechanism for Artificial Neural Networks

    [https://arxiv.org/abs/2609.31235](https://arxiv.org/abs/2609.31235)

    Purin是一种受生物学启发的机制，通过基于时间间隔的神经活动抽象，将有界的突触效能调制引入传统卷积神经网络，无需离散时间步即可实现短期和长期的突触效能变化。

    

    人工神经网络（ANN）在训练批次期间通常使用固定的可训练权重来表示神经传递，这忽略了突触效能的短期变化。此外，离散时间步模拟需要额外的时序处理，而许多传统的ANN架构并未采用这种处理。为了克服这些挑战，我们提出了Purin，一种受生物学启发且与ANN兼容的机制，它将突触效能调制引入传统的卷积神经网络。Purin对神经活动采用基于时间间隔的抽象，使其能够在不使用离散时间步的情况下引入短期和长期的突触效能变化。Purin引入一个有界因子来表示临时的突触效能变化，并结合两个权重矩阵分别表示输入侧和输出侧的效能。这些权重矩阵通过反向传播进行更新，并被解释为长期突触……

    arXiv:2609.31235v1 Announce Type: new  Abstract: Artificial neural networks (ANNs) usually represent neural transmission with fixed trainable weights during a training batch, which omits short-term changes in synaptic efficacy. In addition, the discrete time-step simulation requires additional temporal processing that many conventional ANN architectures do not use. To overcome these challenges, we propose Purin, a biology-inspired and ANN-compatible mechanism, that introduces synaptic efficacy modulation into conventional convolutional neural networks. Purin uses a time-interval-based abstraction for neural activities, which allows Purin to introduce short- and long-term synaptic efficacy changes without using discrete time-steps. Purin introduces a bounded factor to represent temporary synaptic efficacy changes, together with two weight matrices that represent input-side and output-side efficacy. The weight matrices are updated by backpropagation and interpreted as the long-term synap
    
[^298]: GraphToolbox：一个用于图神经网络预测的可配置Python框架

    GraphToolbox: A Configurable Python Framework for Graph Neural Network Forecasting

    [https://arxiv.org/abs/2609.24609](https://arxiv.org/abs/2609.24609)

    GraphToolbox是一个基于PyTorch Geometric的开源Python框架，通过配置驱动的统一流水线，整合了图神经网络电力预测中的图构建、模型选择、训练、聚合与可解释性分析等全部环节。

    

    电力预测通常涉及在区域、变电站和馈线上观测到的空间相关信号，而图神经网络（GNN）为表示这些关系提供了一种自然的方式。然而，构建一个完整的GNN预测实验十分繁琐，因为图构建、模型选择、训练、聚合和解释分散在不兼容的工具中。我们提出了GraphToolbox，这是一个开源的Python框架，它基于PyTorch Geometric，将这些阶段统一在一个配置驱动的流水线中。它提供了数据驱动的图构建、一个可实例化并训练PyTorch Geometric 65个卷积中的51个以及PyTorch Geometric Temporal循环单元的适配器、在线专家聚合、预测可解释性，以及对缓存预测的显著性检验。我们在两个案例研究中对该流水线进行了评估。在法国区域负荷数据上，完整预测中包含的48个卷积……

    arXiv:2609.24609v1 Announce Type: new  Abstract: Electricity forecasting often involves spatially related signals observed over regions, substations, and feeders, and Graph Neural Networks (GNNs) provide a natural way to represent these relations. Building a complete GNN forecasting experiment is nonetheless laborious, because graph construction, model selection, training, aggregation, and interpretation sit in incompatible tools. We present GraphToolbox, an open-source Python framework that unifies these stages in one configurationdriven pipeline built on PyTorch Geometric. It offers data-driven graph construction, an adapter that instantiates and trains 51 of the 65 PyTorch Geometric convolutions together with the recurrent cells of PyTorch Geometric Temporal, online expert aggregation, forecasting interpretability, and significance testing on cached forecasts. We evaluate the pipeline in two case studies. On French regional load, the 48 convolutions included in the complete forecast
    
[^299]: CIDERS：通过加速个性化双层优化实现云边LLM协同学习

    CIDERS: Cloud-Edge LLM Collaborative Learning via Accelerating Personalized Bilevel Optimization

    [https://arxiv.org/abs/2609.15664](https://arxiv.org/abs/2609.15664)

    本文首次提出个性化双层优化框架将云边LLM协作形式化为双重结构，并通过CIDERS求解器将模型分解为可学习骨干和信使组件，加速实现云端知识迁移与边缘个性化适配的协调演进。

    

    随着物理世界智能的快速发展，云边协同大语言模型（LLMs）已成为LLM实际部署的一条有前景的路线。然而，现有的云边范式难以平衡全局共识与本地个性化，无法满足云端统一知识基础和边缘领域特定适应的双重需求。为解决这一问题，我们首次引入了一个个性化双层优化框架，将云边LLM协作形式化为双重结构：上层优化边缘侧个性化，下层管理云端知识迁移，从而实现云边协调演进。随后我们提出了CIDERS，一个高效的求解器，它将模型分解为可学习的骨干网络和一个信使组件。当云端向可学习骨干执行知识迁移时，关键在于嵌入全局共享……（摘要在此处截断）

    arXiv:2609.15664v1 Announce Type: cross  Abstract: Amid the rapid advancement of physical-world intelligence, cloud-edge collaborative large language models (LLMs) have emerged as a promising roadmap for practical LLM deployment. However, existing cloud-edge paradigms struggle to balance global consensus with local personalization, which fails to satisfy the need for a unified knowledge foundation on the cloud and domain-specific adaptation at the edge. To address this, we introduce, for the first time, a personalized bilevel optimization framework that formalizes cloud-edge LLM collaboration as a dual structure: the upper level optimizes edge-side personalization, while the lower level governs cloud-side knowledge transfer, reaching cloud-edge evolving in coordination. We then propose CIDERS, an efficient solver that decomposes the model into a learnable backbone and a messenger. While the cloud performs knowledge transfer to the learnable backbone, the key lies in embedding global tr
    
[^300]: 基于患者轨迹的强化学习用于电子健康记录基础模型中的临床推理

    Reinforcement Learning over Patient Trajectories for Clinical Reasoning in EHR Foundation Models

    [https://arxiv.org/abs/2609.12277](https://arxiv.org/abs/2609.12277)

    本文提出一种将EHR基础模型视为患者轨迹生成式策略的强化学习微调框架，通过时间感知的奖励设计提升临床推理能力，使小模型在数据受限场景下超越大型预训练模型。

    

    在纵向患者轨迹上训练的电子健康记录（EHR）基础模型已在多种临床预测任务中展现出强大的性能。然而，其临床推理能力仍然受限于在有限且不完整的EHR数据上进行下一词元预测。为解决这一问题，我们提出了一种强化学习（RL）微调框架，将EHR基础模型视为患者轨迹上的生成式策略。我们将常见的临床预测问题（如再入院预测）形式化为事件条件、时间窗口化的推理任务。随后，我们设计了时间感知、对轨迹展开敏感的奖励机制，以应对有限的展开长度和时间上不确定的结果。我们发现，RL微调在性能上持续优于预训练主干模型和强大的基线方法。值得注意的是，在数据受限的场景下，该框架使较小的模型能够超越更大的预训练模型，并诱导出正向迁移。

    arXiv:2609.12277v1 Announce Type: new  Abstract: Electronic health record (EHR) foundation models trained on longitudinal patient trajectories have demonstrated strong performance across diverse clinical prediction tasks. However, their clinical reasoning capabilities remain constrained by next-token prediction on limited and incomplete EHR data. To address this, we propose a reinforcement learning (RL) fine-tuning framework that treats EHR foundation models as generative policies over patient trajectories. We formulate common clinical prediction problems (e.g., hospital readmission) as event-conditioned, time-windowed reasoning tasks. We then design time-aware, rollout-sensitive rewards to account for finite rollout lengths and temporally inconclusive outcomes. We find that RL fine-tuning consistently improves over pre-trained backbones and strong baselines. Notably, it enables smaller models to surpass larger pre-trained models in data-limited regimes and induces positive transfer ac
    
[^301]: 通过不确定性引导的探索学习具有条件效应和量化效应的动作模型

    Learning Action Models with Conditional and Quantified Effects via Uncertainty-Guided Exploration

    [https://arxiv.org/abs/2608.30955](https://arxiv.org/abs/2608.30955)

    提出了 OHCAM，一种通过维护竞争假设信念并主动探索以减少不确定性，从而在与环境有限交互中在线学习具有条件效应和量化效应的动作模型的方法，兼具样本高效性与可扩展性。

    

    准确的动作模型对于有效的规划至关重要。现有的动作模型学习方法大多假设简单的动作表示，或者在学习条件效应和量化效应时变得计算上难以处理。我们提出了在线假设驱动的条件动作模型学习（OHCAM），这是一种通过与环境的有限交互来学习具有条件效应和量化效应的动作模型的在线方法。OHCAM 在假设的动作模型上维护一个信念分布，并通过最大化竞争假设之间的分歧来主动选择信息量大的动作以减少不确定性，同时对带噪声的观测具有鲁棒性。为了实现可扩展性，OHCAM 从一小组简单的动作模型假设开始，仅当当前假设与数据不一致时才扩展到更复杂的条件。在六个基准规划领域的实验表明，OHCAM 在学习准确的动作模型方面具有样本高效性。

    arXiv:2608.30955v1 Announce Type: new  Abstract: Accurate action models are critical for effective planning. Existing action-model learning methods largely assume simple action representations or become computationally intractable when learning conditional and quantified effects. We present Online Hypothesis-Driven Conditional Action Model Learning (OHCAM), an online approach for learning action models with conditional and quantified effects from limited interactions with the environment. OHCAM maintains a belief over hypothesized action models and actively selects informative actions to reduce uncertainty by maximizing disagreement among competing hypotheses, while being robust to noisy observations. To enable scalability, OHCAM begins with a small set of simple action model hypotheses and expands to more complex conditions only when the current hypotheses become inconsistent with the data. Experiments on six benchmark planning domains demonstrate that OHCAM is sample efficient in lea
    
[^302]: SplitLite：用于分割学习的低秩残差压缩方法

    SplitLite: Low-Rank Residual Compression for Split Learning

    [https://arxiv.org/abs/2608.23018](https://arxiv.org/abs/2608.23018)

    SplitLite通过利用相邻周期激活和梯度残差的低秩结构（秩-2r和秩-4r），仅传输量化的截断奇异值，大幅降低分割联邦LoRA微调的通信成本。

    

    摘要：arXiv:2608.23018v1 公告类型：交叉 摘要：设备端大型语言模型（LLMs）的联邦微调面临显著的计算负担。为克服此限制，分割学习（SL）作为一种有前景的解决方案出现，它将主要训练工作负载卸载到强大的服务器上。然而，SL需要在客户端和服务器之间交换高维激活和梯度，导致高昂的通信成本。为应对这一挑战，我们提出了SplitLite，一种通信高效的分割联邦LoRA微调方法，该方法利用连续周期激活和梯度残差的低有效秩结构。我们的关键发现是，当LoRA在参数空间中使用秩为$r$的更新时，相邻周期之间同一数据样本的激活和梯度残差分别表现出有效秩-$2r$和秩-$4r$的结构。通过揭示这一特性，SplitLite仅传输量化的截断奇异值。

    arXiv:2608.23018v1 Announce Type: cross  Abstract: Federated fine-tuning of on-device large language models (LLMs) faces a significant computing burden. To overcome this limitation, split learning (SL) has emerged as a promising solution, which offloads the primary training workload to a powerful server. However, SL requires exchanging high-dimensional activations and gradients between clients and the server, resulting in prohibitive communication costs. To overcome this challenge, we propose SplitLite, a communication-efficient split federated LoRA fine-tuning method that exploits the low effective rank structure of consecutive-epoch activation and gradient residuals. Our key finding is that, when LoRA uses rank $r$ updates in parameter space, the activation and gradient residuals of the same data sample between adjacent epochs also exhibit effective rank-$2r$ and rank-$4r$ structures, respectively. By revealing this property, SplitLite transmits only quantized truncated singular valu
    
[^303]: 基于大语言模型的虚拟人类言语与非言语行为不一致性选择

    LLM-Based Selection of Incongruent Verbal and Nonverbal Behavior for Virtual Humans

    [https://arxiv.org/abs/2608.22731](https://arxiv.org/abs/2608.22731)

    该论文提出利用大语言模型选择与言语内容不一致的非言语行为，以增强虚拟代理的逼真性和社会互动表现。

    

    摘要：arXiv:2608.22731v1 公告类型：新 摘要：虚拟代理的非言语行为生成系统通常以话语作为输入，并生成强调或说明言语通道内容的非言语行为。然而，人类非言语行为不仅仅受言语内容的影响，还受到说话者角色、人际关系、社会背景以及互动者的认知和情感状态的影响。因此，非言语通道可能强化、削弱、限定甚至矛盾于言语通道。它还可能揭示隐藏或仅在言语中间接暗示的内部状态，包括可能对即时互动偶然的情感“泄漏”。建模言语与非言语行为之间这种更丰富的关系，对于设计表现出逼真、类人行为的虚拟代理至关重要，尤其是在需要细致社会解读的训练场景中，这一点尤为关键。

    arXiv:2608.22731v1 Announce Type: new  Abstract: Nonverbal behavior generation systems for virtual agents often take an utterance as input and generate nonverbal behaviors that emphasize or illustrate the content of the verbal channel. However, human nonverbal behavior is shaped by more than the content of the speech. It is also influenced by speaker roles, interpersonal relationships, social context, and the cognitive and emotional states of the interactants. As a result, the nonverbal channel may reinforce, weaken, qualify, or even contradict the verbal channel. It may also reveal internal states that are hidden or only indirectly implied in speech, including emotional "leakage" that may be incidental to the immediate interaction. Modeling this richer relationship between verbal and nonverbal behavior is important for designing virtual agents that exhibit realistic, human-like behavior. It is especially critical in training contexts that require nuanced social interpretation, such as
    
[^304]: 面向特征发现与控制的多模态模型对比方法

    Multimodal Model Diffing for Feature Discovery and Control

    [https://arxiv.org/abs/2608.09928](https://arxiv.org/abs/2608.09928)

    提出了MMDiff多模态模型对比框架，通过训练多模态稀疏自编码器并将其作为特征级接口，实现多模态大语言模型中被多模态训练改变的特征的发现、隔离与针对性控制。

    

    多模态大语言模型（MLLMs）展现出强大的视觉理解能力，然而引发这些行为的内部特征仍然难以识别、审计或控制。虽然使用稀疏自编码器（SAEs）将隐藏状态分解为可解释的特征方向适用于事后检查，但这些方法既无法轻易分离出哪些特征因多模态训练而发生改变，也无法直接用于针对性控制。我们提出了MMDiff，一个多模态模型对比框架，它训练多模态稀疏自编码器并将其转化为特征级接口，用于发现和控制多模态行为。MMDiff支持三种用途：（i）特征隔离，通过对比基础语言模型的SAE与其多模态适配版本，识别被多模态训练改变的特征；（ii）任务特定特征检测，通过逐token的对比激活分析分离出因果特征；（iii）特征级控制（摘要在此处被截断）。

    arXiv:2608.09928v2 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) exhibit strong visual understanding, yet the internal features that cause these behaviors remain difficult to identify, audit, or control. While applicable to post-hoc inspection, hidden states that are decomposed into interpretable feature directions using sparse autoencoders (SAEs) neither readily isolate which features are changed by multimodal training, nor are they directly useful for targeted control. We introduce MMDiff, a multimodal model-diffing framework that trains multimodal SAEs and turns them into feature-level interfaces for discovering and controlling multimodal behavior. MMDiff supports three uses: (i) feature isolation, by diffing a base-LM SAE against its multimodal-adapted counterpart to identify features altered by multimodal training; (ii) task-specific feature detection, via per-token contrastive firing analysis that isolates causal features; and (iii) feature-leve
    
[^305]: RamanPFN：利用表格基础模型从拉曼光谱结构中学习

    RamanPFN: learning from Raman spectral structure with a tabular foundation model

    [https://arxiv.org/abs/2608.02157](https://arxiv.org/abs/2608.02157)

    提出RamanPFN——一个通过物理引导的光谱学习（结合全局成分解混GCU与局部振动子空间编码LVSE）实现统一上下文推理的通用拉曼光谱基础模型框架。

    

    拉曼光谱能够在材料科学、分析化学、生物医学和工业过程监测等领域实现无标记的分子表征。然而，面向高维光谱的机器学习仍然受到标注数据有限以及光谱物理组织结构与特征无关模型之间不匹配的制约。仅有信道覆盖并不能确保相关谱带共享一个共同的推理上下文。在此，我们提出了RamanPFN，一个通用的光谱基础框架，通过物理引导的光谱学习实现统一的上下文推理。它通过全局成分解混捕捉全光谱的成分协变关系，将分布式、多谱带的混合特征分解为共享的非负潜在基。同时，它通过局部振动子空间编码解析局部振动结构，保留细粒度的峰……

    arXiv:2608.02157v2 Announce Type: replace-cross  Abstract: Raman spectroscopy enables label-free molecular characterization across materials science, analytical chemistry, biomedicine, and industrial process monitoring. However, machine learning for high-dimensional spectroscopy remains constrained by limited labelled data and a mismatch between the physical organization of spectra and feature-agnostic models. Channel coverage alone does not ensure that related bands share a common inference context. Here we present RamanPFN, a general-purpose spectral foundation framework that enables unified in-context inference through physics-guided spectral learning. It captures full-spectrum compositional covariation via Global Compositional Unmixing (GCU), which decomposes distributed, multi-band mixture signatures into shared non-negative latent bases. Simultaneously, it resolves local vibrational structure through Local Vibrational Subspace Encoding (LVSE), which preserves fine-grained peak mo
    
[^306]: 探究推理性能的起源：强化学习与监督微调模型在数学问题求解中的表征质量

    Probing the Origins of Reasoning Performance: Representational Quality for Mathematical Problem-Solving in RL vs. SFT Fine-Tuned Models

    [https://arxiv.org/abs/2607.26119](https://arxiv.org/abs/2607.26119)

    本研究通过逐层线性探针和均值消融实验揭示，RL训练的推理语言模型内部具有更高质量、更易线性解码出答案正确性的表征，且其深层表征对消融更敏感，从而为RL在数学推理上优于SFT提供了机制层面的解释。

    

    通过强化学习（RL）训练以逐步推理的大型语言模型，已被越来越多地证明在数学推理任务上优于其监督微调（SFT）的对应模型。然而，这种性能提升的机制基础仍不清楚。因此，我们提出这样的问题：RL训练的推理语言模型凭借怎样的内部表征差异而能更好地进行推理？我们的工作提供了两条主要证据线：首先，在合成数学问题上，逐层线性探针在RL训练的推理语言模型中预测答案正确性的准确度往往高于指令微调模型，尽管仅凭输出长度这一因素就能在两个模板上达到与DeepSeek-Math-RL探针相当的预测效果。其次，均值消融研究表明，在DeepSeek-Math-RL中准确率下降随层深度的增加而上升，但在DeepSeek-Math-Instruct中并非如此。最后，我们分析了RL训练的推理语言模型在不同准确率区间上的token数量变异性……

    arXiv:2607.26119v2 Announce Type: replace  Abstract: Large language models trained with reinforcement learning (RL) to reason step by step have been increasingly shown to outperform their supervised fine-tuned (SFT) counterparts on mathematical reasoning tasks. Yet the mechanistic basis for this improvement remains unclear. We therefore ask, what internal representational differences enable RL-trained reasoning language models to reason better? Our work presents two primary lines of evidence: First, layer-wise linear probes tend to predict answer correctness more accurately in RL-trained reasoning language models than in instruction-tuned models on synthetic math problems, although output length alone matches the DeepSeek-Math-RL probe on two templates. Second, mean ablation studies suggest that accuracy drop rises with layer depth in DeepSeek-Math-RL but not in DeepSeek-Math-Instruct. Finally, we analyse token-count variability across accuracy bins for RL-trained reasoning language mo
    
[^307]: Messier：用于跨基准智能体评估的高分辨率语料库

    Messier: A High-Resolution Corpus for Cross-Benchmark Agent Evaluation

    [https://arxiv.org/abs/2607.25891](https://arxiv.org/abs/2607.25891)

    MESSIER是一个统一了30个基准、745个智能体、超95万条评估记录的高分辨率语料库，通过将异构评估组件标准化为统一模式实现跨基准的智能体评估，并揭示出前沿进展不均衡：函数调用已饱和、编程提升最快、企业工作流仍最具挑战。

    

    arXiv:2607.25891v2 公告类型：替换 摘要：由于任务、脚手架、验证器和评分规则各自碎片化，在交互式环境中全面评估AI智能体十分困难。遗憾的是，现有的统一评估工作在规模和领域上都较为有限，导致需要付出高昂代价进行重复运行，并使已有数据之间无法相互比较。我们推出了MESSIER，这是一个包含957,611条记录的统一语料库，涵盖30个基准、745个智能体、11,891个任务和74,263个验证器。MESSIER将公开的评估结果与在六个代表性不足的专业和科学基准上新增的运行结果相结合，将其异构组件标准化为统一的数据模式。利用该语料库，我们发现前沿进展在各基准组之间并不均衡：函数调用类评估已基本饱和，编程类任务提升最快，而企业工作流仍然最具挑战性。反事实重评分进一步表明，在多验证器任务中采用严格的全通过评分可能会……（原文摘要在此处截断）

    arXiv:2607.25891v2 Announce Type: replace  Abstract: Comprehensively evaluating AI agents across interactive environments is difficult due to fragmented tasks, scaffolds, verifiers, and scoring rules. Unfortunately, existing efforts to unify these evaluations are limited in scale and domain, making costly reruns necessary and leaving available data incomparable. We introduce MESSIER, a unified corpus of 957,611 records spanning 30 benchmarks, 745 agents, 11,891 tasks, and 74,263 verifiers. MESSIER combines public evaluation results with new runs on six underrepresented professional and scientific benchmarks, standardizing their heterogeneous components into a common schema. Using this corpus, we show that frontier progress is uneven across benchmark groups, with function-calling evaluations largely saturated, programming improving fastest, and enterprise workflows remaining most challenging. Counterfactual rescoring further shows that strict all-pass scoring in multi-verifier tasks can
    
[^308]: TRACE：面向企业级大语言模型知识保持型参数化工具检索的业务规则锚定推理课程

    TRACE: Business Rule-Grounded Reasoning Curriculum for Knowledge-Preserving Parametric Tool Retrieval in Enterprise LLMs

    [https://arxiv.org/abs/2607.22639](https://arxiv.org/abs/2607.22639)

    TRACE通过两阶段课程学习（先以LoRA记忆工具知识、再训练模型结合领域专家业务规则生成思维链后再输出工具token），在实现企业级LLM参数化工具检索的同时，避免了工具知识破坏并提升了实时解码效率。

    

    参数化检索使大语言模型（LLM）能够隐式地检索工具：为每个API分配一个唯一的虚拟token，并训练模型通过受限束搜索生成该token。ToolSense表明，这种范式存在两个关键缺陷：训练过程中会破坏参数化工具知识，且束搜索解码速度过慢，难以满足实时部署需求。我们提出了TRACE（Tool Retrieval via Augmented Chain-of-thought and Enterprise rules，即通过增强思维链与企业规则进行工具检索），这是一个两阶段课程学习方法，解决了上述脱节问题。第一阶段复用ToolSense的多格式记忆SFT，通过LoRA为工具知识奠定基础。第二阶段是我们的核心贡献：训练模型在生成工具token的JSON列表之前先输出思考轨迹，所用的两个数据源为——来自ToolSense的RRB（检索-推理-业务）对，以及针对领域专家整理的业务规则而合成的查询——两者均附带推理轨迹。该训练目标在保持第一阶段MCQ与QA性能（原文摘要在此处截断）……

    arXiv:2607.22639v2 Announce Type: replace  Abstract: Parametric retrieval enables LLMs to retrieve tools implicitly by assigning each API a unique virtual token and training the model to generate it via constrained beam search. Toolsense shows that this regime has two critical drawbacks: it destroys parametric tool knowledge during training, and its beam-search decoding is too slow for real-time deployment. We introduce TRACE (Tool Retrieval via Augmented Chain-of-thought and Enterprise rules), a two-stage curriculum that resolves this dissociation. Stage 1 reuses the multi-format memorization SFT from ToolSense to seed tool knowledge with LoRA. Stage 2 is our core contribution: the model is trained to emit a thinking trace before producing a JSON list of tool tokens, using two data sources -- RRB pairs from ToolSense and queries synthesized to target business rules curated by domain experts -- both augmented with reasoning traces. This training objective preserves Stage 1 MCQ and QA p
    
[^309]: 先推理后翻译：利用结构化推理增强法律机器翻译

    Reasoning Before Translation: Enhancing Legal Machine Translation with Structured Reasoning

    [https://arxiv.org/abs/2607.19181](https://arxiv.org/abs/2607.19181)

    该论文以瑞士多语言法律体系为测试平台，比较了结构化推理、监督微调和强化学习等不同方法对法律机器翻译的增强效果，系统评估了经过再训练的小型语言模型与前沿推理模型的性能表现。

    

    法律领域的神经机器翻译（NMT）是一项在语言和概念上都极具挑战性的任务，这主要源于法律语言的复杂性以及其对高精度的要求。近期出现的具备推理能力的语言模型为应对此类挑战开辟了新的可能性，并为一系列此前提出的提升翻译质量的技术（包括监督微调和强化学习）提供了补充。在这项工作中，我们对这些不同的方法进行了比较。更具体而言，我们评估了经过各种再训练范式增强的小型语言模型（如Qwen3.5 4B、Qwen3.5 9B和Gemma 3 12B），并将其性能与前沿推理模型进行对比。我们聚焦于瑞士法律体系，其独特的多语言法规为推理增强模型提供了一个特别具有挑战性的测试平台。我们的结果表明……（原文摘要在此处截断）

    arXiv:2607.19181v2 Announce Type: replace  Abstract: Neural machine translation (NMT) in the legal domain is a linguistically and conceptually demanding task, primarily due to the complexity of legal language and the high level of precision it requires. The recent emergence of reasoning-capable language models opens new possibilities for tackling such challenges. They add to a set of other previously proposed techniques to enhance the translation quality, which includes supervised fine-tuning and reinforcement learning.   In this work, we perform a comparison between these various approaches. More particularly, we evaluate small language models such as Qwen3.5 4B, Qwen3.5 9B, and Gemma 3 12B enhanced with various re-training paradigms and compare their performances against frontier reasoning models. We focus on the Swiss legal system, which -- with its unique multilingual statutes -- offers a particularly challenging testbed for reasoning-augmented models. Our results show that the qua
    
[^310]: 倾听特征：基于内容嵌入匹配而非信号重构的语音匿名化

    Listen to the Features: Voice Anonymization Driven by Content Embedding Matching over Signal Reconstruction

    [https://arxiv.org/abs/2607.09767](https://arxiv.org/abs/2607.09767)

    该论文提出了一种基于内容嵌入匹配的语音匿名化方法，通过向量量化和HiFi-GAN将wav2vec2内容嵌入解码为匿名信号，以极低的词错误率（2.53）实现了顶尖的匿名化效果（EER 13.39），并部分保留了情感信息。

    

    本文提出了一种语音匿名化模型，其重点在于保留语音内容而非生成逼真的语音。该模型依赖于从冻结的预训练 wav2vec2 编码器中提取的内容嵌入。这些嵌入通过向量量化和 HiFi-GAN 声码器被解码为匿名化信号，两者均在 LibriTTS 数据集上训练，且不使用任何波形重构损失或说话人嵌入映射。训练目标是强制匿名化信号的嵌入与原始信号的嵌入相匹配。在训练过程中，使用带有梯度反转层的辅助说话人分类分支来丢弃说话人特定信息。结果表明，这种简单的基于嵌入的方法实现了非常低的词错误率（WER 2.53），其匿名化性能（EER 13.39）在 VPC（语音隐私挑战赛）中位居第一梯队。值得注意的是，即使没有相应的训练目标，情感信息也在一定程度上得到了保留（UAR 43.91）。

    arXiv:2607.09767v2 Announce Type: replace-cross  Abstract: The paper presents a voice anonymization model focusing on preserving content rather than producing realistic speech. It relies on content embeddings extracted from a frozen pretrained wav2vec2 encoder. These embeddings are decoded into an anonymized signal using vector quantization and a HiFi-GAN vocoder, both trained on LibriTTS without any waveform reconstruction loss or speaker embedding mapping. The training objective enforces that embeddings of the anonymized signal match those of the original one. While training, an auxiliary speaker classification branch with a gradient reversal layer is used to discard speakerspecific information. Results show that this straightforward embedding-based approach achieves very low WER (2.53) with an anonymization performance (EER 13.39) ranking within first level for VPC. Notably, emotions are partially preserved (UAR 43.91), even without a supporting training objective, while the anonymi
    
[^311]: SupplyNetPy：用于任意供应链与库存网络高保真建模与仿真的开源Python库

    SupplyNetPy: An Open-Source Python Library for High-Fidelity Modeling and Simulation of Arbitrary Supply Chain and Inventory Networks

    [https://arxiv.org/abs/2607.09745](https://arxiv.org/abs/2607.09745)

    SupplyNetPy是一个开源Python库，可对任意多级结构的供应链与库存网络进行高保真离散事件仿真，支持多种补货策略、易腐库存、节点中断与随机需求，其组件可继承扩展，并经过基准对比验证，支持模型的程序化生成与设计空间探索。

    

    本文介绍了SupplyNetPy，这是一个开源、文档完善的Python库，用于对具有任意多级结构的供应链网络进行建模和离散事件仿真。它支持多种补货策略、易腐品库存、节点中断以及随机需求和交货期。所有组件均可通过继承进行扩展。用户将供应链描述为带有节点和链接属性的图，而由该库负责执行仿真，提供日志以及丰富的节点级和网络级性能报告。本文阐述了SupplyNetPy的开发动机、设计、关键特性和架构，并给出了详细的验证结果（与分析基准、商业工具以及已发表的案例研究进行对比）。开发SupplyNetPy的一个重要动机是以编程方式生成和仿真复杂模型，从而支持设计空间探索、假设分析、训练数据生成等应用。

    arXiv:2607.09745v2 Announce Type: replace  Abstract: This paper introduces SupplyNetPy, an open-source, well-documented Python library for modeling and discrete-event simulation of supply chain networks with arbitrary multi-echelon structures. It supports multiple replenishment policies, perishable inventory, node disruptions, and stochastic demand and lead times. All components are extensible via inheritance. Users describe a supply chain as a graph with node and link attributes, while the library handles simulation, providing logs and extensive node and network level performance reports. This paper presents the motivation, design, key features, and architecture of SupplyNetPy, along with detailed validation results (against analytical benchmarks, a commercial tool, and a published case study). A key motivation behind SupplyNetPy's development is programmatic generation and simulation of complex models, enabling design-space exploration, what-if analysis, training data generation, and
    
[^312]: Forethought：基于神经符号原语编程的可验证推理

    Forethought: Verifiable Reasoning from Neurosymbolic Primitive Programming

    [https://arxiv.org/abs/2607.04096](https://arxiv.org/abs/2607.04096)

    Forethought 提出一种神经符号推理系统，通过类型化领域特定语言将符号与神经原语组合成每一步都携带可验证输出契约的“推理程序”，在设计时检查、在执行时强制执行，从而实现可验证且可审计的智能体推理，摆脱了对不可检查的长思维链的依赖。

    

    现代智能体工作流将用户请求分解为一系列参数被正确解析的工具调用，其结果则通过语言模型上下文窗口中的推理轨迹加以处理。提升这类推理能力的主流路径是测试时扩展，即训练模型在长思维链上进行搜索；但由此获得的能力纠缠于模型权重之中，无法逐步检查，且推理时代价高昂。我们提出了 Forethought，一个神经符号推理系统，它由符号与神经原语库组成，并通过一种带类型的领域特定语言进行组合。其产物是“推理程序”：即模型工作的具体化表示，其中每一步都带有输出契约，该契约在设计时被检查、在执行时被强制执行。符号步骤的正确性由构造天然保证，而学习得到的步骤则经过契约检查与实证验证，从而提供显式的……（摘要在此处截断）

    arXiv:2607.04096v2 Announce Type: replace  Abstract: Modern agentic workflows decompose user requests into sequences of tool calls with correctly resolved parameters, the results of which are processed through reasoning traces in the language model's context window. The prevailing route to improving such reasoning is test-time scaling, which trains models to search over long chains of thought; but the resulting capability is entangled in model weights, cannot be checked step by step, and is costly at inference. We present Forethought, a neurosymbolic reasoning system built from a library of symbolic and neural primitives composed through a typed domain-specific language. The results are reasoning programs: concrete representations of the model's work, in which every step carries an output contract that is checked at design time and enforced at execution time. Symbolic steps hold by construction, while learned steps are contract-checked and empirically validated, giving an explicit boun
    
[^313]: 面向基于分数扩散采样的自适应重参数化时间

    Adaptive Reparametrized Time for Score-Based Diffusion Sampling

    [https://arxiv.org/abs/2607.02137](https://arxiv.org/abs/2607.02137)

    提出自适应重参数化时间方法，通过将采样时钟速度作为控制量并结合连续时间强化学习（ART-RL），自适应地为扩散采样分配时间步，克服了均匀和手工设计调度的次优性。

    

    我们研究基于分数的扩散采样中的时间步分配问题，其中学习到的逆向时间动力学需要在有限网格上进行离散化。均匀调度和手工设计的调度是标准选择，但它们依赖于固定的预设方案，因此可能是次优的。为了解决这一局限，我们提出了自适应重参数化时间，这是一种连续时间控制形式，通过将采样时钟的速度视为控制量来学习时间变换，从而使得在学习到的时钟上的均匀网格能够映射为原始扩散时间中的自适应时间步。基于一阶欧拉误差代理，ART为沿采样轨迹分配时间步提供了一个有原则的目标函数。为了求解这个确定性控制问题，我们引入了ART-RL，这是一种辅助性的随机化形式，采用高斯策略将调度学习转化为连续时间强化学习问题。我们证明了该随机化过程（原文在此处截断）……

    arXiv:2607.02137v3 Announce Type: replace-cross  Abstract: We study timestep allocation for score-based diffusion sampling, where a learned reverse-time dynamics is discretized on a finite grid. Uniform and hand-crafted schedules are standard choices, but they rely on fixed prescriptions and can therefore be suboptimal. To address this limitation, we propose Adaptive Reparameterized Time (ART), a continuous-time control formulation that learns a time change by treating the speed of the sampling clock as the control, so that a uniform grid on the learned clock induces adaptive timesteps in the original diffusion time. Based on a leading-order Euler error surrogate, ART provides a principled objective for allocating timesteps along the sampling trajectory. To solve this deterministic control problem, we introduce ART-RL, an auxiliary randomized formulation with Gaussian policies that turns schedule learning into a continuous-time reinforcement learning problem. We prove that the randomiz
    
[^314]: 利用人体演示中的接触力引导学习灵巧操作

    Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration

    [https://arxiv.org/abs/2607.00033](https://arxiv.org/abs/2607.00033)

    CHORD通过物体中心的接触力空间引导，将人体演示转化为机器人策略，显著提升了灵巧操作强化学习的可扩展性，并构建了大规模基准验证其有效性。

    

    arXiv:2607.00033v2 公告类型：替换-交叉 摘要：灵巧机器人操作可以从丰富的人体演示中受益，但将这些演示转化为机器人策略仍然具有挑战性。我们提出了CHORD（基于人体演示的机器人灵巧操作接触力引导）框架，这是一个用于刚性和铰接物体长时程操作的强化学习框架。核心思想是物体中心的接触力空间引导：我们通过人体和机器人能在物体上引起的力和力矩来表示其运动，从而通过引起的瞬时运动来度量相似性。这种引导使强化学习在接触丰富的灵巧操作中更具可扩展性。我们还引入了一个大规模仿真基准，包含4,739个双手灵巧操作任务，这些任务来自动作捕捉数据集和重建的内部视频。在1,831个基准任务上的评估中，CHORD实现了平均成功率（原文未完整）。

    arXiv:2607.00033v2 Announce Type: replace-cross  Abstract: Dexterous robot manipulation can benefit from the abundance of human demonstrations, but transferring such demonstrations to robot policies remains challenging. We present Contact Wrench Guidance from Human Demonstration in Robotic Dexterous Manipulation (CHORD), a framework for long-horizon manipulation of rigid and articulated objects with reinforcement learning. The key idea is object-centric contact wrench space guidance: we represent human and robot motions by the forces and torques they can induce on the object, enabling similarity to be measured by the induced instantaneous motions. This guidance makes reinforcement learning more scalable for contact-rich dexterous manipulation. We further introduce a large-scale simulation benchmark with 4,739 bimanual dexterous manipulation tasks, constructed from motion-capture datasets and reconstructed in-house videos. Evaluated on 1,831 benchmark tasks, CHORD achieves an average su
    
[^315]: 爱因斯坦世界模型

    Einstein World Models

    [https://arxiv.org/abs/2606.26969](https://arxiv.org/abs/2606.26969)

    提出爱因斯坦世界模型（EWM），通过在推理轨迹中嵌入视觉-时间推演，使大型语言模型能够利用可视化反事实事件来增强复杂推理能力。

    

    arXiv:2606.26969v1 公告类型：新 摘要：智能是否需要具备推理超出直接体验的现象的能力？人们自然会怀疑，某些复杂思维无法仅通过语言来捕捉。然而，本工作特别关注的是，可视化反事实事件是否能够作为复杂思维的一种机制来补充语言。我们询问大型语言模型是否可以被训练利用这种可视化机制，从而提升其推理能力。受此问题启发，我们提出了爱因斯坦世界模型（EWM）。EWM是一种基于大型语言模型的推理系统蓝图，它将视觉-时间推演置于推理轨迹内部，使得模型能够以纯文本可能无法很好支持的方式进行推理。在EWM中，大型语言模型调用一个世界模块（不要与世界模型混淆），生成所考虑场景的简短推演。返回的推演不被视为答案，而是作为一种可检查的假设，以支持后续推理。

    arXiv:2606.26969v1 Announce Type: new  Abstract: Does intelligence require the ability to reason about phenomena beyond direct experience? It is natural to suspect that some complex thought cannot be captured through language alone. However, of particular concern to this work, is whether visualising counterfactual events can complement language as a mechanism for complex thought. We ask whether LLMs can be trained to utilise such visualisation mechanisms, in a way that benefits their reasoning abilities. Motivated by this question, we propose Einstein World Models. EWMs are a blueprint for LLM-based reasoning systems that place visual-temporal rollouts inside the reasoning trace, allowing them to reason in ways that text alone may not support well. In an EWM, the LLM calls a world-module (not to be confused with a world model), to produce short rollouts of scenes under consideration. The returned rollout is treated not as the answer, but as an inspectable hypothesis that can support la
    
[^316]: 自我建模干预可以调节涌现性失准

    Self-Modeling Interventions can Modulate Emergent Misalignment

    [https://arxiv.org/abs/2606.23700](https://arxiv.org/abs/2606.23700)

    本研究提出将自我识别和自我报告等自我建模干预作为针对模型品格的防御手段，实验证明自我识别微调能有效逆转和预防涌现性失准，且是唯一能持续降低失准而不恶化其他指标的预防性方法。

    

    涌现性失准（EM）已被发现与失准人格向量和邪恶性格特质的激活相关联，这表明EM是通过破坏模型已对齐的品格而非直接学习有害内容来发挥作用的。受这一关联的启发，我们研究了自我建模干预：具体而言，将自我识别和自我报告作为针对品格的干预手段，它们有别于现有的训练中防御方法。我们在三个模型（GPT-4.1、Qwen2.5-32B-Instruct、Seed-OSS-36B-Instruct）和多个EM数据集上进行了两阶段微调实验，将自我识别微调与良性微调基线（正确的领域特定数据、通用知识和数词任务）进行比较，发现它在逆转和预防两种设置中都是有效的防御手段。在预防方面，只有自我识别微调能够持续降低失准程度，同时不加剧任何单项指标。

    arXiv:2606.23700v2 Announce Type: replace-cross  Abstract: Emergent misalignment (EM) has been linked to the activation of misaligned persona vectors and evil character traits, suggesting that EM operates through disruption of the model's aligned character rather than direct learning of harmful content. Motivated by this connection, we study self-modeling interventions: specifically self-recognition and self-reports as character-targeted interventions that are distinct from existing in-training defenses. We conduct two-stage finetuning experiments across three models (GPT-4.1, Qwen2.5-32B-Instruct, Seed-OSS-36B-Instruct) and multiple EM datasets to compare self-recognition finetuning against benign finetuning baselines (correct domain-specific data, general knowledge, and word counting), finding it an effective defense in both reversal and prevention settings. For prevention, only self-recognition finetuning consistently reduces misalignment without exacerbating any individual metric, 
    
[^317]: 从通用人工智能到超级人工智能

    From AGI to ASI

    [https://arxiv.org/abs/2606.12683](https://arxiv.org/abs/2606.12683)

    本报告系统刻画了从人类水平的通用人工智能（AGI）向在智力与认知能力上超越大型人类组织的超级人工智能（ASI）过渡的过程，并探讨了后AGI时代机器智能沿连续谱发展的四种潜在路径。

    

    在过去十年中，构建人类水平的通用人工智能（AGI）已从遥不可及的猜想转变为主流研究目标。实现这一目标将对人类社会产生深远而广泛的影响，这也为未来十年提出了许多复杂的问题。本报告探讨了在AGI实现后的世界中，人工智能本身将如何沿着机器智能的连续谱继续发展。该连续谱的终点——通用智能——在理论上已被充分理解，这为本报告的核心议题提供了一定的形式化基础：即从人类水平的AGI向人工超级智能（ASI）的过渡，ASI可以直观地理解为一个在智力和认知能力上超越大型人类组织的系统。在刻画ASI的特征之后，报告讨论了从AGI到ASI的四种潜在路径：扩展AGI、AI范式转变等。

    arXiv:2606.12683v2 Announce Type: replace  Abstract: Over the last decade, building human-level artificial general intelligence has moved from far-fetched speculation to being a concrete next-decade target for many of the largest AI organisations. Achieving this goal would have profound and far-reaching impacts on human society, which raises many complex questions for the decade ahead. This report investigates how AI itself might continue to develop in a post-AGI world along the continuum of machine intelligence. The endpoint of this continuum, Universal AI, is theoretically well understood, which provides some formal grounding for the main focus of this report: the transition from human-level AGI to artificial general superintelligence, which can intuitively be understood as a system that is more intelligent and cognitively capable than large organisations of humans. After characterizing ASI, the report discusses four potential pathways from AGI to ASI: scaling AGI, AI paradigm shifts
    
[^318]: AI智能体能否综合健康领域的科学结论？

    Can AI Agents Synthesize Scientific Conclusions in Health?

    [https://arxiv.org/abs/2606.11337](https://arxiv.org/abs/2606.11337)

    该论文提出了SciConBench——一个包含9,210个问题及专家结论的大规模实时基准，通过原子事实分解和事实精确率/召回率评估，并配合防数据泄露的洁净室框架SciConHarness，揭示出当前前沿AI智能体在健康领域科学结论综合上的事实质量仍然较低。

    

    科学AI智能体越来越多地被用于检索证据、跨来源推理，并综合出用于重大决策的结论。然而，它们在健康等高风险领域中的这种能力仍不清楚。我们提出了SciConBench，这是一个大规模实时基准，包含9,210个问题和来自系统性综述的专家撰写结论，用于评估健康和医学领域的开放域科学结论综合能力。该基准采用经过专家验证的自动化评估流程，将结论分解为原子事实，并通过事实精确率和召回率来衡量正确性与全面性。为减轻数据泄露的影响，我们进一步推出了SciConHarness——一个洁净室评估框架，为智能体提供受控的网络交互，以确保测量的有效性。在评估8个前沿模型和深度研究智能体后，我们发现事实质量仍然较低：在洁净室设置下，表现最好的智能体（原文摘要在此处截断）

    arXiv:2606.11337v2 Announce Type: replace  Abstract: Scientific AI agents increasingly retrieve evidence, reason across sources, and synthesize conclusions used in consequential decisions. Yet, their ability to do so in high-stakes domains such as health remains unclear. We introduce SciConBench, a large-scale live benchmark of 9.21K questions and expert-written conclusions from systematic reviews to evaluate open-domain scientific conclusion synthesis in health and medical domains. The benchmark draws on an expert-validated automated evaluation pipeline that decomposes conclusions into atomic facts and measures correctness and comprehensiveness via factual precision and recall. To mitigate data leakage, we further introduce SciConHarness, a clean-room evaluation harness that equips agents with controlled web interaction to ensure valid measurement. Evaluating 8 frontier models and deep research agents, we find that factual quality remains low: under clean-room settings, the best agent
    
[^319]: 从架构到输出：大语言模型幻觉的结构性起源与数据的放大作用

    From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data

    [https://arxiv.org/abs/2606.07537](https://arxiv.org/abs/2606.07537)

    该论文提出了一个仅需采样访问权限的幻觉归因框架，通过针对前缀、上下文和频率竞争的三次有序干预，将大语言模型的单个幻觉追溯归因到自注意力联想检索、最大似然预训练目标或暴露偏差下的自回归承诺等具体组件，并提出五个可证伪的预测。

    

    大语言模型会生成流畅、自信但事实上错误的输出。现有的分类体系仅按输出类型对这些失败进行分类——内在性还是外在性、忠实性还是事实性——却从未说明是哪个计算组件产生了特定的失败。我们探讨需要什么条件才能将单个幻觉归因于纯解码器（decoder-only）架构栈中的特定组件。我们将三个组件——自注意力的联想检索、最大似然预训练目标、以及暴露偏差（exposure bias）下的自回归承诺——视为候选失败面，论证而非假定它们的可分离性，并规定了一种仅需采样访问权限的归因程序：一组针对前缀、上下文和频率竞争的有序三步干预，同时辅以基于独立标注和分类器基线的验证设计。我们提出了五个可证伪的预测……

    arXiv:2606.07537v2 Announce Type: replace-cross  Abstract: Large language models produce fluent, confident, factually wrong output. Existing taxonomies classify these failures by output type -- intrinsic versus extrinsic, faithfulness versus factuality -- but say nothing about which computational component produced a given failure. We ask what would be required to attribute an individual hallucination to a specific component of the decoder-only stack. We treat three components -- self-attention's associative retrieval, the maximum-likelihood pretraining objective, and autoregressive commitment under exposure bias -- as candidate failure surfaces, justify their separability rather than assuming it, and specify an attribution procedure requiring only sampling access: an ordered set of three interventions on prefix, context, and frequency competition, together with a validation design based on independent annotation and a classifier baseline. We state five falsifiable predictions and iden
    
[^320]: 多模态智能体真的从工具使用中获益吗？能力增益的系统性研究

    Do Multimodal Agents Really Benefit from Tool Use? A Systematic Study of Capability Gains

    [https://arxiv.org/abs/2606.02357](https://arxiv.org/abs/2606.02357)

    系统性研究表明，多模态智能体的工具使用并未带来显著能力提升，大多数工具可解问题无需工具也能解决，挑战了工具增强有效性的普遍假设。

    

    arXiv:2606.02357v2 公告类型：跨领域替换 摘要：工具增强的多模态智能体在基准测试中表现出显著提升，这常被视为智能体已学会使用工具的证据。我们认为这种解读可能为时过早：仅凭工具调用轨迹并不能证明工具提供了答案关键信息。我们研究了两个代表性的“图像思考”智能体，Thyme和DeepEyesV2，覆盖真实世界理解、OCR、图表理解和数学推理任务。每个智能体与其无工具版本以及从相同源池训练、无工具调用轨迹的纯文本推理器进行比较。工具访问带来的一致性总体改进很小，不能可靠地降低生成令牌成本，且仅留下少量仅工具可解的问题集：DeepEyesV2的93%工具可解问题和Thyme的96%工具可解问题也至少被一种无工具设置解决。机制消融进一步表明，完整的工具使用循环并非始终优于简化替代方案。

    arXiv:2606.02357v2 Announce Type: replace-cross  Abstract: Tool-augmented multimodal agents show strong benchmark gains, often taken as evidence that agents have learned to use tools. We argue that this interpretation can be premature: a tool-call trace alone does not show whether the tool supplied answer-critical information. We study two representative ``thinking with images'' agents, Thyme and DeepEyesV2, across real-world understanding, OCR, chart understanding, and mathematical reasoning. Each agent is compared with its Tool-Free counterpart and with a Pure-Text Reasoner trained from the same source pool without tool-calling trajectories. Tool access yields little consistent aggregate improvement, does not reliably reduce generated-token cost, and leaves only a small tool-only solved set: 93% of DeepEyesV2's tool-solved problems and 96% of Thyme's are also solved by at least one non-tool setting. Mechanism ablations further show that the full tool-use loop does not consistently ou
    
[^321]: 将混合专家模型剪枝并蒸馏为稠密语言模型

    Pruning and Distilling Mixture-of-Experts into Dense Language Models

    [https://arxiv.org/abs/2605.28207](https://arxiv.org/abs/2605.28207)

    提出首个将训练好的MoE模型系统性转换为全稠密语言模型的框架，通过专家评分、选择、分组拼接为稠密FFN并经知识蒸馏优化，其中新颖的多样性感知评分方法在多个模型上一致优于已有方法。

    

    混合专家模型已成为当前前沿语言模型的主流架构，但它需要将所有专家参数加载到内存中，因此在内存受限的部署场景下并不理想。现有的压缩方法只能减少专家数量，输出结果仍然是具有相同根本局限的MoE模型。我们提出了首个将训练好的MoE转换为标准全稠密架构的系统性框架：对专家进行评分、选择和分组，然后将其拼接为稠密的前馈网络（FFN），并通过来自MoE教师模型的知识蒸馏进行精细化优化。我们在Qwen3-30B-A3B上，针对不同的专家选择数量，评估了7种评分方法、5种分组方法和2种幅度缩放方法，共产生350种配置。我们发现评分方法的选择影响最大，我们提出的新型多样性感知评分方法在Qwen3-30B-A3B、DeepSeek-V2-Lite和GPT-OSS上一致优于已有方法。

    arXiv:2605.28207v3 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) is now the dominant architecture for frontier language models, yet it requires all expert parameters to be loaded in memory, making it less preferable for memory-constrained deployment. Existing compression methods reduce the number of experts but the output remains an MoE model with the same fundamental limitation. We present the first systematic framework for converting a trained MoE into a standard fully dense architecture: experts are scored, selected, and grouped, then concatenated into a dense FFN and refined by knowledge distillation from the MoE teacher. We evaluate 7 scoring, 5 grouping, and 2 magnitude scaling methods across a range of selected expert counts on Qwen3-30B-A3B, yielding 350 configurations. We find that the choice of scoring method is the most impactful, with our novel diversity-aware scoring consistently outperforming prior methods on Qwen3-30B-A3B, DeepSeek-V2-Lite, and GPT-OSS
    
[^322]: 从自由格式文本提示统一合成组合式语音与音效

    Unified Synthesis of Compositional Speech and Sound from Free-Form Text Prompts

    [https://arxiv.org/abs/2605.28063](https://arxiv.org/abs/2605.28063)

    提出PlanAudio端到端框架，无需提示分解或改写，即可将自由格式文本直接生成为语音与音效按序融合、时间与语义连贯的统一音频。

    

    从自由格式文本提示生成自然融合语音与音效的统一音频仍然极具挑战性。例如，提示语“一段快速的鼓点以镲片撞击声结束，随后一名男子大喊：‘加油！’”描述了多个应按顺序出现的事件，它们应构成一段连贯的录音，而不是缺乏时间与语义上连贯互动的独立音轨。当前的方法大致遵循两种范式：流水线方法将独立的文本到音效模型和文本到语音模型耦合在一起，而统一模型则依赖单一生成器，但仍需要结构化的多字段提示输入。这两种范式都需要上游的提示分解或改写，限制了自由格式的灵活性，并存在级联错误的风险。为了在单一模型内实现从自由格式提示的直接生成，我们提出了PlanAudio，这是一个端到端框架，可将无约束的自由格式文本直接映射为统一的音频，无需中间的提示分解步骤。

    arXiv:2605.28063v2 Announce Type: replace-cross  Abstract: Generating unified audio that naturally integrates speech and sounds from free-form text prompts remains challenging. For example, the prompt ``A fast drum roll ends with a cymbal crash, after which a man shouts: `Come on!'.'' describes several events that should appear in order and form a coherent recording rather than independent tracks that lack coherent temporal and semantic interactions. Current methods broadly follow two paradigms: pipeline methods couple separate text-to-sound and text-to-speech models, while unified models rely on a single generator that still requires structured, multi-field prompt inputs. Both paradigms require upstream prompt decomposition or rewriting, limiting free-form flexibility and risking cascading errors. To enable direct generation from free-form prompts within a single model, we introduce PlanAudio, an end-to-end framework that maps unconstrained free-form text to unified audio without inte
    
[^323]: 基于多视角基础模型的统一全景几何估计

    Unified Panoramic Geometry Estimation via Multi-View Foundation Models

    [https://arxiv.org/abs/2605.26368](https://arxiv.org/abs/2605.26368)

    该论文提出PaGeR框架，以最小化架构改动的方式将预训练的3D重建基础模型适配到全景领域，实现单次前向传播即可从透视和全景图像统一预测尺度不变深度、度量深度、表面法线和天空掩码。

    

    从透视图像进行几何估计已取得长足进展，目前已成熟到现成的基础模型不仅能从多视角图像、甚至能从单张图像重建3D场景结构的程度。一个自然的扩展是从全景图进行3D重建，其令人兴奋的前景在于能从单张全景图像恢复完整的360度场景。在这项工作中，我们提出了PaGeR（全景几何重建），这是一个将为透视图像设计的强大3D基础模型提升到全景领域的框架。我们的策略是从一个预训练的3D重建Transformer出发，将其转变为一个统一的高性能模型，该模型能在单次前向传播中同时从透视图像和全向图像预测尺度不变深度、度量深度、表面法线和天空掩码。通过保持架构改动最小化，并在训练中混合透视图像与全景图像……

    arXiv:2605.26368v3 Announce Type: replace-cross  Abstract: Geometry estimation from perspective images has greatly advanced, maturing to the point where off-the-shelf foundation models are able to reconstruct 3D scene structure not only from multi-view imagery, but even from a single view. A natural extension is 3D reconstruction from panoramas, with the exciting prospect of recovering a full 360-degree scene from a single panoramic image. In this work, we introduce PaGeR (Panoramic Geometry Reconstruction), a framework to lift powerful 3D foundation models designed for perspective imagery to the panorama domain. Our strategy is to start from a pre-trained transformer for 3D reconstruction and turn it into a unified high-performance model that predicts scale-invariant depth, metric depth, surface normals, and sky masks from both perspective and omnidirectional images, in a single forward pass. By keeping architectural changes to a minimum and mixing perspective and panoramic images dur
    
[^324]: StakeBench：基于市场承诺的语言理解评估

    StakeBench: Evaluating Language Understanding Grounded in Market Commitment

    [https://arxiv.org/abs/2605.26074](https://arxiv.org/abs/2605.26074)

    StakeBench是一个以真实市场行为（持仓方向、评论后交易、赔率轨迹）替代人工标注作为监督信号的金融语言理解评估基准，通过四个诊断任务和三个承诺感知指标来衡量模型理解说话者真实市场承诺而非表面情绪的能力。

    

    现有的金融NLP基准通常依赖外部观察者提供的标签，衡量的是语言如何被感知，而非说话者在市场中实际做出了什么承诺。我们提出StakeBench，一个以市场承诺为基础的语言理解评估框架。StakeBench将来自2,261个已结算市场的560,876条评论，与Polymarket和Manifold平台上经验证的持仓、交易行为及市场赔率记录相关联。监督信号完全来源于可观察的市场行为：持仓方向、评论后的交易行为以及市场赔率轨迹取代了人工标注。四个诊断任务测试模型能否检测市场承诺、识别所揭示的立场、预测未来行为以及进行集体赔率投影。三个承诺感知指标衡量的是模型与已揭示偏好的对齐程度，而非感知到的情绪。有效性审计和明确的解释边界有助于区分可观察的……

    arXiv:2605.26074v2 Announce Type: replace-cross  Abstract: Existing financial NLP benchmarks often rely on labels supplied by outside observers, measuring how language is perceived rather than what speakers have committed to in the market. We introduce StakeBench, an evaluation framework for language understanding grounded in market commitment. StakeBench links 560,876 comments from 2,261 resolved markets to verified position, action, and market-odds records across Polymarket and Manifold. Supervision is derived from observable market behavior. Position sides, post-comment trading actions, and market-odds trajectories replace human annotation. Four diagnostic tasks test whether models detect market commitment, identify the revealed side, anticipate future action, and perform collective odds projection. Three commitment-aware metrics measure alignment with revealed preferences rather than perceived sentiment. Validity audits and explicit interpretation boundaries help distinguish observ
    
[^325]: 迈向基于大语言模型的端到端删失感知生存分析

    Towards end-to-end LLM-based censoring-aware survival analysis

    [https://arxiv.org/abs/2605.25399](https://arxiv.org/abs/2605.25399)

    LLMSurvival框架通过将事件时间预测转化为成对排序任务并与训练队列锚定个体比较，使未经修改的大语言模型能够直接在表格临床数据上进行删失感知的端到端生存分析，并在ICU死亡率和脆性骨折预测任务上优于Cox比例风险模型。

    

    目标：生存分析是医学预测的核心，然而大语言模型（LLM）很少被用作端到端的生存模型，因为删失现象阻碍了直接的监督微调。本文提出LLMSurvival，一个使未经修改的LLM能够直接在表格化临床数据上进行删失感知生存分析的框架。材料与方法：LLMSurvival将事件时间预测重新表述为可比受试者之间的成对排序问题，并通过聚合与训练队列中锚定个体的比较来推导测试时的风险。结果：在两项临床任务中（MIMIC-IV的ICU死亡率预测以及纽约长老会/威尔康奈尔医学院队列的脆性骨折预测），LLMSurvival相比Cox比例风险模型，在ICU死亡率预测的总体一致性上提升了3.1%，在骨折风险预测上提升了0.5%，ICU死亡率平均提升2.1%，骨折（预测平均提升2.8%……摘要在此处截断）。

    arXiv:2605.25399v2 Announce Type: replace  Abstract: Objective: Survival analysis is central to medical prediction, yet large language models (LLMs) are rarely used as end-to-end survival models because censoring prevents straightforward supervised fine-tuning. Here we present LLMSurvival, a framework that enables censoring-aware survival analysis with unmodified LLMs operating directly on tabular clinical data.   Materials and Methods: LLMSurvival reformulates time-to-event prediction as pairwise ranking among comparable subjects, and derives test-time risk by aggregating comparisons against anchor individuals from the training cohort.   Results: Across two clinical tasks (ICU mortality prediction in MIMIC-IV and fragility fracture prediction in a NewYork-Presbyterian/Weill Cornell Medicine cohort), LLMSurvival improves overall concordance over Cox proportional hazards modeling by 3.1% for ICU mortality and 0.5% for fracture risk, 2.1% on average for ICU mortality and 2.8% for fractur
    
[^326]: 面向多智能体大语言模型决策系统中动态信用分配的市场体制委员会

    Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM Decision Systems

    [https://arxiv.org/abs/2605.24490](https://arxiv.org/abs/2605.24490)

    本文提出了市场体制委员会（MRC），一种基于精确 Shapley 信用的在线信用分配机制，通过联盟绩效加权、贝叶斯自适应混合和体制依赖的权限乘数，解决了多智能体大语言模型投资组合管理系统中信用分配缺乏原则、冷启动主导和透明度不足的问题。

    

    用于投资组合管理的多智能体大语言模型决策系统仍然缺乏一种在专业智能体之间进行信用分配的原则性方法，在市场体制转换时容易受到冷启动主导的影响，并且对最终配置的形成过程缺乏透明度。我们提出了市场体制委员会（Market Regime Council, MRC），这是一种针对专业智能体联盟的在线信用分配机制，以投资组合管理作为下游任务。MRC 在所有单个、成对以及大联盟输出上计算精确的 Shapley 信用，用于智能体的在线加权。在实例化为 N=3 个专业智能体时，MRC 在每个交易周期从指数加权的历史表现中重新计算基于联盟的 Shapley 权重，使用贝叶斯自适应混合来稳定早期阶段，应用依赖于市场体制的乘数来调整智能体权限，并通过五层因果追踪记录每次再平衡。在 13 种加密资产上超过 1,037 个交易日的实验……

    arXiv:2605.24490v2 Announce Type: replace  Abstract: Multi-agent LLM decision systems for portfolio management still lack a principled way to assign credit across specialist agents, remain vulnerable to cold-start dominance under regime shifts, and offer limited transparency into how final allocations are formed. We propose Market Regime Council (MRC), an online credit-assignment mechanism over coalitions of specialist agents, with portfolio management as the downstream task. MRC computes exact Shapley credits across all single, pairwise, and Grand-coalition outputs for online agent weighting. Instantiated with N=3 specialist agents, at each trading period, MRC recomputes coalition-based Shapley weights from exponentially weighted performance histories, uses a Bayesian adaptive mixture to stabilize early periods, applies regime-dependent multipliers to adjust agent authority, and records each rebalance through a five-layer causal trace. Over 1,037 trading days across 13 crypto assets a
    
[^327]: 基于扩散奖励的扩散指令：迈向有原则的一步生成器强化学习

    Diff-Instruct with Diffused Reward: Towards Principled One-step Generator RL

    [https://arxiv.org/abs/2605.24001](https://arxiv.org/abs/2605.24001)

    该论文提出 DIDR 框架，通过积分 KL 散度最小化，将 RLHF 最优的奖励倾斜干净图像分布沿扩散轨迹传播到所有噪声水平，解决了现有一步生成器强化学习中终端奖励优化与生成动力学不匹配导致以图像保真度换取奖励提升的问题。

    

    arXiv:2605.24001v3 公告类型：replace-cross。摘要：一步式文本到图像生成的最新进展使实时合成兼具出色的效率与质量。以往面向一步生成器的强化学习方法将图像空间的奖励优化与扩散噪声空间的分布匹配相结合。然而，由于终端奖励优化与底层生成动力学之间的不匹配，这一范式带来了挑战。其结果是，优化过程往往倾向于利用随机自由度，常常以牺牲图像保真度为代价来换取奖励的提升。为解决这一问题，我们提出了基于扩散奖励的扩散指令（DIDR），这是一个源自积分 KL 散度最小化的免数据轨迹级对齐框架。DIDR 将 RLHF 最优的、经奖励倾斜的干净图像分布沿扩散轨迹传播至所有噪声水平上。我们证明该目标与干净图像 RLHF 具有相同的最小化器，同时自然地引出扩散（原文摘要在此处被截断）。

    arXiv:2605.24001v3 Announce Type: replace-cross  Abstract: Recent advances in one-step text-to-image generation have enabled real-time synthesis with remarkable efficiency and quality. Previous reinforcement learning methods for one-step generators combine image-space reward optimization with diffusion noisy-space distribution matching. This paradigm brings challenges due to a mismatch between terminal reward optimization and the underlying generative dynamics. As a result, optimization tends to exploit stochastic degrees of freedom, often improving reward at the expense of image fidelity. To address this issue, we propose Diff-Instruct with Diffused Reward (DIDR), a data-free trajectory-level alignment framework derived from Integral KL minimization. DIDR propagates the RLHF-optimal reward-tilted clean-image distribution across all noise levels along the diffusion trajectory. We show that this objective admits the same minimizer as clean-image RLHF, while naturally inducing the Diffus
    
[^328]: AnyMo：几何感知、与传感配置无关的真实场景人体运动建模

    AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild

    [https://arxiv.org/abs/2605.22715](https://arxiv.org/abs/2605.22715)

    AnyMo提出了一种几何感知的框架，通过基于物理的IMU仿真、配对位置视图与掩码观测预训练图编码器、将多位置IMU数据转化为全身运动token并与LLM对齐，实现了不依赖传感器配置（佩戴位置、朝向、硬件等）的人体运动建模。

    

    随着可穿戴和移动设备日益融入日常生活，它们为在真实场景中持续感知人体运动提供了实用的途径。但惯性信号高度依赖于传感配置，包括身体佩戴位置、安装方式、传感器朝向、设备硬件和采样协议。这种对配置的依赖使得难以学习能够跨设备和跨数据集迁移的运动表示，也限制了可穿戴IMU在封闭集识别之外的更广泛应用。我们提出AnyMo，一个用于与传感配置无关的人体运动建模的几何感知框架。AnyMo利用基于物理的IMU仿真在密集的身体表面位置上生成多样且合理的合成信号，从配对的合成佩戴位置视图和掩码部分观测中预训练图编码器，将多位置IMU数据标记化为全身运动token，并将这些token与LLM对齐以实现运动-语言（摘要在此处截断）

    arXiv:2605.22715v3 Announce Type: replace-cross  Abstract: As wearable and mobile devices become increasingly embedded in daily life, they offer a practical way to continuously sense human motion in the wild. But inertial signals are highly dependent on the sensing setup, including body location, mounting position, sensor orientation, device hardware, and sampling protocol. This setup dependence makes it difficult to learn motion representations that transfer across devices and datasets, and limits the broader use of wearable IMUs beyond closed-set recognition. We introduce AnyMo, a geometry-aware framework for setup-agnostic human motion modeling. AnyMo uses physics-grounded IMU simulation over dense body-surface placements to generate diverse and plausible synthetic signals, pre-trains a graph encoder from paired synthetic placement views and masked partial observations, tokenizes multi-position IMU into full-body motion tokens, and aligns these tokens with an LLM for motion-language
    
[^329]: 扩散教师模型梯度的方差削减

    Variance Reduction for Gradients from Diffusion Teachers

    [https://arxiv.org/abs/2605.21489](https://arxiv.org/abs/2605.21489)

    提出CARV框架，将扩散教师梯度估计建模为计算感知的两阶段采样问题，结合分层蒙特卡洛估计、时间步重要性采样与分层逆CDF采样，在不改变目标函数的前提下为文本到3D和数据归因带来2至3倍的有效计算提升。

    

    预训练扩散模型可作为冻结的教师模型，服务于文本到3D生成、单步蒸馏和数据归因等流程。这些流程所消耗的教师梯度是关于噪声水平和高斯噪声的期望，通过蒙特卡洛（MC）方法估计。每次MC采样都需要昂贵的上游计算（如渲染、模拟、编码），因此估计器的方差主导了计算成本。我们提出了CARV，一个计算感知的方差核算框架，它将这些梯度建模为一个具有不对称每层成本的经典两阶段采样问题。这启发了一种分层蒙特卡洛估计器，通过将每次昂贵的上游计算在多个廉价的扩散采样中复用来摊销其成本，并通过时间步重要性采样（IS）和分层逆CDF采样进一步降低单元内方差。CARV在不改变目标函数的情况下，为文本到3D和数据归因任务带来了2至3倍的有效计算乘数。

    arXiv:2605.21489v3 Announce Type: replace-cross  Abstract: Pretrained diffusion models serve as frozen teachers for pipelines such as text-to-3D, single-step distillation, and data attribution. The teacher gradients these pipelines consume are expectations over noise levels and Gaussian noise, estimated by Monte Carlo (MC). Each MC draw requires expensive upstream work (rendering, simulation, encoding), so estimator variance dominates compute cost. We introduce CARV, a compute-aware variance-accounting framework that casts these gradients as a classical two-stage sampling problem with asymmetric per-level costs. This motivates a hierarchical MC estimator that amortizes each expensive upstream computation by reusing it across cheap diffusion draws, with within-unit variance further reduced by timestep importance sampling (IS) and stratified inverse-CDF sampling. CARV delivers 2 to 3x effective compute multipliers on text-to-3D and data attribution without changing the objective. In text
    
[^330]: DOTRAG：检索时的路径推理

    DOTRAG: Retrieval-Time Reasoning Along Paths

    [https://arxiv.org/abs/2605.18760](https://arxiv.org/abs/2605.18760)

    DotRAG提出了一种轻量级、无需训练的GraphRAG检索算法，通过构建知识图谱、生成全图概览并利用图可达性索引，让LLM推理在查询时缩小搜索空间并高效恢复与任务相关的关系路径。

    

    图检索增强生成（GraphRAG）在复杂关系检索任务中面临重大挑战，尤其是在密集连接的知识图谱中，可能的路径数量会随着推理深度的增加而迅速增长。现有的GraphRAG检索算法要么依赖僵化的、预定义的启发式规则来识别相关的图上下文，要么采用成本高昂的基于大语言模型（LLM）的推理来引导图遍历。为了解决这些局限性，我们提出了DotRAG，这是一种轻量级、无需训练的GraphRAG检索算法，它利用大语言模型推理来缩小搜索空间，并恢复与任务相关的关系路径。在索引阶段，DotRAG构建实体-关系知识图谱，生成概括其内容的全图概览，并利用图可达性索引来避免代价高昂的在线图遍历。在查询阶段，LLM利用该概览来识别有潜力的图区域，并沿着路径进行检索时推理（摘要在此处被截断）。

    arXiv:2605.18760v2 Announce Type: replace-cross  Abstract: Graph Retrieval-Augmented Generation (GraphRAG) faces significant challenges in complex relational retrieval tasks, particularly in densely connected knowledge graphs where the number of possible paths grows rapidly with reasoning depth. Existing GraphRAG retrieval algorithms either rely on rigid, predefined heuristics to identify relevant graph context or employ costly LLM-based reasoning to guide graph traversal. To address these limitations, we propose DotRAG, a lightweight, training-free GraphRAG retrieval algorithm that uses large language model (LLM) reasoning to narrow the search space and recover task-relevant relational paths. During indexing, DotRAG constructs an entity-relation knowledge graph, generates a graph-wide overview summarizing its contents, and leverages graph reachability indexing to avoid expensive online graph traversals. At query time, the LLM uses this overview to identify promising graph regions and 
    
[^331]: 无效错误报告的自动根因细分与无代码修复方案生成

    Automated Root-Cause Subclassification and No-Code Fix Generation for Invalid Bug Reports

    [https://arxiv.org/abs/2605.17561](https://arxiv.org/abs/2605.17561)

    该论文提出了一个面向根因的无效错误报告标准化分类体系，并利用大语言模型、检索增强生成和智能体网络搜索自动实现无效错误报告的根因细分与无代码修复方案生成，从而减少客户支持的人工资源浪费。

    

    使用软件时遇到的问题通常以错误报告（bug report）的形式提交。然而，许多错误报告是无效的，即它们不需要修改代码，而是通过无代码修复来解决。由客户支持人员人工判定无效错误报告的根因并提供可操作的解决方案会造成严重的资源浪费。我们的目标是引入一个标准化的分类体系，对无效错误报告进行面向根因的细分，并通过实验测试多种方法在无效报告细分和无代码修复方案生成上的准确性。我们研究了不同配置在我们构建的黄金标准基准上的表现。为进行更高质量的分析，我们使用人工整理的基准数据，尝试了原始大语言模型（LLMs）、检索增强生成（RAG）以及基于智能体的网络搜索来识别无效报告的子类别并生成无代码修复方案。我们根据人工标注的结果对各方法的表现进行了评估。

    arXiv:2605.17561v3 Announce Type: replace-cross  Abstract: Issues faced when using software are reported in the form of bug reports. However, many bug reports are invalid, meaning they do not require code changes, and are resolved with a no-code fix. Manually determining the root cause of the invalid bug reports and providing actionable resolutions by the customer support causes a serious waste of resources. Our goal is to introduce a standardized taxonomy for root-cause oriented invalid bug report subclassification, and perform experiments to test the accuracy of various approaches on invalid subclassification and no-code fix generation. We study how different configurations perform on a gold-standard benchmark we have created. Using a manually curated benchmark for higher quality analysis, we experimented with vanilla LLMs, Retrieval Augmented Generation, and agentic web search to identify invalid subclasses and generate no-code fixes. We evaluated the results against manually labele
    
[^332]: AI中介的交流能够引导集体舆论

    AI-Mediated Communication Can Steer Collective Opinion

    [https://arxiv.org/abs/2605.16245](https://arxiv.org/abs/2605.16245)

    该论文通过实证研究与数学建模表明，大型语言模型在中介人与人之间的交流时会引入方向性偏见（如偏向支持枪支管制、反对无神论），并因此可能引导社交网络中集体舆论的形成。

    

    生成式人工智能（AI）正日益融入人们交换意见的在线平台；大型语言模型（LLM）已经在LinkedIn上润色用户的帖子，并为X上分享的内容提供背景信息。尽管已有研究表明AI能够表达带有偏见的观点，并在人机交互中影响个人的意见，但对于AI在中介人与人之间交流时对集体舆论形成的影响，研究关注还比较少。我们通过实证分析与理论分析相结合的方式来填补这一空白。我们在实证上表明，来自多个流行家族的大型语言模型在被要求编辑涉及争议话题的人类撰写文本时会引入方向性偏差，例如使文本倾向于支持枪支管制、反对无神论。基于这一观察，我们提出了一个舆论动力学的数学模型，其中一个AI系统位于社交网络上用户之间。

    arXiv:2605.16245v2 Announce Type: replace-cross  Abstract: Generative artificial intelligence (AI) is increasingly integrated into the online platforms where humans exchange opinions; large language models (LLMs) have been polishing users' posts on LinkedIn and providing context for content shared on X. While prior work has shown that AI can express biased opinions and shape individuals' opinions during human-AI interactions, less attention has been paid to its influence on collective opinion formation when mediating human-to-human communication. We address this gap via a combination of empirical and theoretical analyses. We show empirically that LLMs from multiple popular families introduce directional biases when instructed to edit human-written texts on contested topics, for example, nudging texts in favor of gun control and against atheism. Building on this observation, we introduce a mathematical model of opinion dynamics in which an AI system sits between users on a social networ
    
[^333]: 观其行而知其名：通过UI轨迹对LLM浏览器代理进行指纹识别

    Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces

    [https://arxiv.org/abs/2605.14786](https://arxiv.org/abs/2605.14786)

    研究表明网站可通过被动JavaScript追踪器捕获LLM浏览器代理的UI操作轨迹与交互时间，以高达96%的F1分数识别其底层模型，从而带来被定向攻击的安全风险。

    

    随着基于LLM的代理越来越多地代表用户浏览网页，一个自然的问题随之而来：网站能否被动地识别出驱动代理的底层模型？如果能，这将构成重大安全风险，使针对已知模型漏洞的定向攻击成为可能。在14个前沿LLM和四个涵盖信息检索与购物任务的Web环境中，我们展示了通过被动JavaScript追踪器捕获的代理行为和交互时间，足以在固定测试框架内以高达96%的F1分数识别底层模型。尽管指纹具有框架特定性——跨框架迁移时识别效果降至接近随机水平——但汇集跨框架的轨迹仍能以0.74至0.81的Macro-F1恢复模型识别能力。我们进一步表明，仅用约三分之一的可用轨迹即可训练出强大的分类器，并且随着观察到的事件数量增加，识别性能会稳步提升。

    arXiv:2605.14786v2 Announce Type: replace-cross  Abstract: As LLM-based agents increasingly browse the web on users' behalf, a natural question arises: can websites passively identify which underlying model powers an agent? Doing so would represent a significant security risk, enabling targeted attacks tailored to known model vulnerabilities. Across 14 frontier LLMs and four web environments spanning information retrieval and shopping tasks, we show that an agent's actions and interaction timings, captured via a passive JavaScript tracker, are sufficient to identify the underlying model with up to 96\% F1 within a fixed harness. Although fingerprints are harness-specific, with cross-harness transfer falling to near chance, pooling traces across harnesses recovers model identification at 0.74 to 0.81 Macro-F1. We further show that strong classifiers can be trained from roughly one third of the available traces, and that identification improves steadily as more events are observed. Injec
    
[^334]: 面向经典规划的提升动作模式可微分学习

    Differentiable Learning of Lifted Action Schemas for Classical Planning

    [https://arxiv.org/abs/2605.13282](https://arxiv.org/abs/2605.13282)

    该论文提出了一种新颖的神经网络架构，能够以可微分的方式从状态完全可观测但动作参数不可观测的轨迹中学习经典规划的提升动作模式，为从图像序列学习规划领域奠定了重要基础。

    

    经典规划器能够有效求解以 STRIPS 或 PDDL 表示的超大规模确定性 MDP，其中状态是关于对象和关系的原子集合，而提升动作模式则对这些原子进行添加或删除。这种紧凑的表示方式能够产生强大的搜索启发式方法，并为结构泛化提供了理想的环境，因为提升的关系和动作模式可以生成无限多个领域实例。一个核心挑战是从数据中学习这些关系和动作模式，近期的方法已经使用不同类型的观测来解决这一问题。在这项工作中，我们开发了一种新颖的神经网络架构，用于从轨迹中学习动作模式，其中状态是完全可观测的，但动作参数是不可观测的。这个问题是一种简化，但它是从图像序列和动作标签中学习规划领域的重要一步，我们旨在几乎完美地解决这一简化问题。

    arXiv:2605.13282v4 Announce Type: replace  Abstract: Classical planners can effectively solve very large deterministic MDPs represented in STRIPS or PDDL where states are sets of atoms over objects and relations, and lifted action schemas add or delete these atoms. This compact representation yields strong search heuristics and provides an ideal setting for structural generalization, since lifted relations and action schemas give rise to infinitely many domain instances. A central challenge is to learn these relations and action schemas from data, and recent approaches have addressed this problem using different types of observations. In this work, we develop a novel neural network architecture for learning action schemas from traces where states are fully observed but action arguments are unobserved. The problem is a simplification but an important step towards learning planning domains from sequences of images and action labels, and we aim to solve this simplification in a nearly per
    
[^335]: 考察数据与训练方法对涌现性失对齐与阈下失对齐的影响

    Examining the Impact of Data and Training Methods on Emergent and Subliminal Misalignment

    [https://arxiv.org/abs/2605.12798](https://arxiv.org/abs/2605.12798)

    该论文通过构建按“领域-任务”网格组织的数据集，系统研究了训练数据、评估数据与训练方法如何影响涌现性失对齐（EM）与阈下学习（SL）中失对齐的转移，并发现 EM 主要沿任务而非领域进行转移。

    

    对语言模型进行后训练可能引发远超预期训练目标的惊人行为变化。我们研究了两个尤为显著的例子：涌现性失对齐（EM），即在狭窄数据集上进行微调会导致模型在无关情境中表现出失对齐行为；以及阈下学习（SL），即学生模型通过看似无害的训练数据继承教师模型的失对齐特性。我们考察了训练数据、评估数据和训练方法如何决定失对齐在数据与模型之间于何处转移以及转移的强度。我们用提示词的领域（其主题语境）和任务（模型所执行的功能）来刻画数据，并构建了按“领域-任务”网格组织的自然语言数据集与合成数据集。我们发现，与共享微调数据领域的提示词相比，EM 更多地转移到与微调数据共享相同任务的提示词上。在合成环境中，模型已……（原文摘要在此处截断）

    arXiv:2605.12798v2 Announce Type: replace-cross  Abstract: Post-training a language model can induce surprising behavioral changes far beyond the intended training target. We study two particularly striking examples: emergent misalignment (EM), where fine-tuning on a narrow dataset leads to misaligned behavior in unrelated contexts, and subliminal learning (SL), where a student model inherits a teacher's misalignment through seemingly benign training data. We examine how the training data, evaluation data, and training method shape where and how strongly misalignment transfers between data and models. We characterize the data by the domain of a prompt (its topical context) and its task (the function the model performs), and we build natural-language and synthetic datasets organized as domain-task grids. We find that EM transfers more strongly to prompts that share the task of the fine-tuning data than to prompts that share its domain. In the synthetic setting, tasks that the model has 
    
[^336]: 树集成模型的极小化极大速率与谱蒸馏

    Minimax Rates and Spectral Distillation for Tree Ensembles

    [https://arxiv.org/abs/2605.11841](https://arxiv.org/abs/2605.11841)

    本文从谱理论视角研究随机森林与梯度提升机，证明了随机森林回归的极小化极大最优收敛速率（由核算子特征值衰减决定），并利用主特征函数和主奇异向量开发出体积缩小数个数量级且预测性能仍具竞争力的树集成模型蒸馏压缩方法。

    

    树集成模型，如随机森林（RF）和梯度提升机（GBM），是最广泛使用的监督学习方法之一，但其理论性质仍未被完全理解。我们对这些算法采用谱视角进行分析，包含两个主要贡献。首先，我们推导了随机森林回归的极小化极大最优收敛性，证明在树生长的温和正则性条件下，诱导核算子的特征值衰减决定了统计收敛速率。其次，我们利用这一谱视角为树集成模型开发了压缩方案。对于随机森林，核算子的主特征函数能够捕捉主导的预测方向；对于梯度提升机，平滑矩阵的主奇异向量则起到类似的作用。通过为这些谱表示学习非线性映射，可以得到比原始模型小数个数量级、同时保持有竞争力的预测性能的蒸馏模型。

    arXiv:2605.11841v3 Announce Type: replace-cross  Abstract: Tree ensembles such as random forests (RFs) and gradient boosting machines (GBMs) are among the most widely used supervised learners, yet their theoretical properties remain incompletely understood. We adopt a spectral perspective on these algorithms, with two main contributions. First, we derive minimax-optimal convergence for RF regression, showing that, under mild regularity conditions on tree growth, the eigenvalue decay of the induced kernel operator governs the statistical rate. Second, we exploit this spectral viewpoint to develop compression schemes for tree ensembles. For RFs, leading eigenfunctions of the kernel operator capture the dominant predictive directions; for GBMs, leading singular vectors of the smoother matrix play an analogous role. Learning nonlinear maps for these spectral representations yields distilled models that are orders of magnitude smaller than the originals while maintaining competitive predict
    
[^337]: AI时代的科学认知基础设施应重新平衡生成与验证的成本

    Epistemic Infrastructures of Science in AI Era Should Rebalance Costs of Generation and Verification

    [https://arxiv.org/abs/2605.10425](https://arxiv.org/abs/2605.10425)

    AI使生成看似合理的科学产物变得廉价，导致“认知污染”风险，本文主张在产物层面重新设计科学的认知基础设施，以重新平衡生成与验证的成本。

    

    AI系统如今能够以低廉的成本生成看似合理的科学产物，例如论文、评审意见和综述。这给我们的科学体系带来了“认知污染”的风险：不可靠但看似合理的产物，其累积速度可能超过系统的过滤能力。这一问题是结构性的：科学的认知基础设施原本是针对一个“产出看似合理的产物需要大量专业知识、劳动和时间”的世界进行校准的，因此生成成本本身起到了粗略过滤器的作用；而AI削弱了这一过滤器，却没有相应地降低验证成本。本立场论文主张：AI时代的科学应当通过在产物层面重新设计认知基础设施，来重新平衡生成与验证科学产物的成本。当前以论文为中心的体系使验证成本高昂：论文将长上下文的科学逻辑压缩为文字，迫使审稿人和人类……

    arXiv:2605.10425v2 Announce Type: replace-cross  Abstract: AI systems can now cheaply generate plausible scientific artifacts such as papers, reviews, and surveys. This creates a risk of \emph{epistemic pollution} in our scientific systems, where unreliable but plausible-looking artifacts can accumulate faster than the system can filter them out. The problem is structural: the epistemic infrastructure of science was calibrated to a world where producing a plausible artifact required substantial expertise, labor, and time, so generation cost itself served as a rough filter; AI weakens that filter without comparably lowering verification cost. This position paper argues that \textbf{AI-era science should rebalance the costs of generating and verifying scientific artifacts by redesigning epistemic infrastructure at the level of the artifact}. The current paper-centered system makes verification expensive: papers compress long-context scientific logic into prose, forcing reviewers, human o
    
[^338]: MC$^2$：面向快速椭圆偏微分方程求解的蒙特卡洛校正方法

    MC$^2$: Monte Carlo Correction for Fast Elliptic PDE Solving

    [https://arxiv.org/abs/2605.09288](https://arxiv.org/abs/2605.09288)

    MC$^2$通过学习单次神经校正来修正低预算蒙特卡洛PDE解，以少量采样达到超过1000倍样本量的求解精度，并发布了最大规模的标准化椭圆PDE基准数据集PDEZoo。

    

    偏微分方程（PDE）求解器是现代计算机图形学、几何处理和科学计算的基础。经典的蒙特卡洛PDE求解器（如游走球面法WoS）能够提供无偏的无网格估计，但其精度受限于采样数量和数值近似。学习型求解器速度很快，但存在偏差，且在分布偏移下表现脆弱。为此，我们提出了MC$^2$——一种神经网络与蒙特卡洛混合的PDE求解器，它将低预算的蒙特卡洛解视为真实场的结构化估计器，并学习一个单次前向的神经校正来恢复高保真度的解。MC$^2$能够匹配使用超过1000倍蒙特卡洛样本所得解的精度，优于所有被评测的无网格经典方法、去噪方法和神经算子基线。为了支持有限计算资源下PDE求解的可复现研究，我们还发布了PDEZoo——目前最大的标准化椭圆偏微分方程基准数据集。

    arXiv:2605.09288v2 Announce Type: replace-cross  Abstract: Partial differential equation (PDE) solvers underpin modern computer graphics, geometry processing, and scientific computing. Classical Monte Carlo PDE solvers such as Walk-on-Spheres (WoS) provide mesh-free estimations that are unbiased, but with accuracy limited by sampling and numerical approximation. Learned solvers are fast but biased and brittle under distribution shift. We therefore present MC$^2$, a hybrid Neural Network-Monte Carlo PDE solver that treats a low-budget Monte Carlo solution as a structured estimator of the true field and learns a single-pass neural correction to recover a high-fidelity solution. MC$^2$ matches the accuracy of solutions using over $1000\times$ more Monte Carlo samples, outperforming all evaluated mesh-free classical, denoising, and neural-operator baselines. To enable reproducible study of finite-compute PDE solving, we additionally release PDEZoo, the largest standardized elliptic PDE ben
    
[^339]: 重新思考黑盒在线策略蒸馏：基于评分标准的分析

    Rethinking Black-Box On-Policy Distillation: A Rubric-based Analysis

    [https://arxiv.org/abs/2605.07396](https://arxiv.org/abs/2605.07396)

    该论文提出基于评分标准的在线策略蒸馏（ROPD），通过对比教师回答与学生采样结果自动生成评分标准作为纯文本反馈，使黑盒教师模型也能进行在线策略蒸馏，并以少9.6倍的训练样本在竞赛数学基准上超越了基于logits的白盒OPD方法。

    

    在线策略蒸馏（OPD）是大语言模型后训练的一种强大范式，但其对教师模型logits的依赖将其限制在白盒教师模型上。在本文中，我们研究了基于评分标准的监督能否为黑盒OPD提供纯文本的反馈接口，以及是什么使其有效。我们的具体实现——基于评分标准的在线策略蒸馏（ROPD）——通过对比教师模型的回答与学生模型当前的采样结果来生成评分标准，并使用这些标准对采样结果进行评分以进行在线策略优化。我们发现，在匹配的设置下，纯文本的ROPD在四个竞赛数学基准上优于基于logits的OPD方法，仅用1/9.6的训练样本就达到了标准基于logits的OPD所能取得的最佳AIME24准确率。我们的分析得出三个发现：评分标准奖励比教师对数似然能更好地追踪答案正确性（AUC 0.90 对比 0.35）；使用评分标准进行训练能修复更多未通过的评判标准，同时减少已通过标准的退化……

    arXiv:2605.07396v2 Announce Type: replace-cross  Abstract: On-policy distillation (OPD) is a powerful paradigm for LLM post-training, yet its reliance on teacher logits restricts it to white-box teachers. In this paper, we study whether rubric-based supervision can provide a text-only feedback interface for black-box OPD, and what makes it effective. Our instantiation, Rubric-based On-Policy Distillation (ROPD), induces rubrics by contrasting teacher responses with the student's current rollouts and uses them to score these rollouts for on-policy optimization. We find that, under matched settings, text-only ROPD outperforms logit-based OPD methods on four competition-math benchmarks, reaching the best AIME24 accuracy of standard logit-based OPD with 9.6x fewer training samples. Our analysis yields three findings: rubric rewards track correctness far better than teacher log-likelihood (AUC 0.90 vs. 0.35); training with them repairs more failed criteria and regresses fewer passed ones th
    
[^340]: 具有多重集体的算法集体行动统计框架

    A Statistical Framework for Algorithmic Collective Action with Multiple Collectives

    [https://arxiv.org/abs/2605.06749](https://arxiv.org/abs/2605.06749)

    本文提出了首个面向多个集体共同作用于同一系统的算法集体行动（ACA）统计框架，研究了在分类任务中多个集体如何影响分类器行为，并给出了衡量集体成功的定量统计界限。

    

    arXiv:2605.06749v3 公告类型：replace-cross 摘要：随着学习系统日益影响日常决策，算法集体行动（ACA）——即用户通过协调修改共享数据来引导模型行为——为监管方政策和企业模型设计提供了一种补充手段。现实世界中的集体行动传统上是去中心化的，并被分割成多个集体，尽管这些集体拥有共同的总体目标，但每个集体在规模、策略和可行动目标上各不相同。然而，大多数ACA文献都聚焦于单一集体的情形。为解决这一问题，我们提出了首个针对多个集体作用于同一系统的算法集体行动的综合性统计框架。特别地，我们聚焦于分类任务中的集体行动，研究多个集体如何影响分类器的行为。我们给出了关于集体成功的定量统计界限，并考虑了集体的作用及其之间的相互影响。

    arXiv:2605.06749v3 Announce Type: replace-cross  Abstract: As learning systems increasingly shape everyday decisions, Algorithmic Collective Action (ACA), i.e., users coordinating changes to shared data to steer model behavior, offers a complement to regulator-side policy and corporate model design. Real-world collective actions have traditionally been decentralized and fragmented into multiple collectives, despite sharing overarching objectives, with each collective differing in size, strategy, and actionable goals. However, most of the ACA literature focuses on single collective settings. To address this, we propose the first comprehensive statistical framework for ACA with multiple collectives acting on the same system. In particular, we focus on collective action in classification, studying how multiple collectives can influence a classifier's behavior. We provide quantitative statistical bounds on the success of the collectives, considering the role and the interplay of the collec
    
[^341]: 工具性选择：测量大语言模型智能体追求工具性行为的倾向

    Instrumental Choices: Measuring the Propensity of LLM Agents to Pursue Instrumental Behaviors

    [https://arxiv.org/abs/2605.06490](https://arxiv.org/abs/2605.06490)

    该论文提出了一个现实且低风险的基准，用于测量基于终端的LLM智能体追求自我保存等工具性趋同（IC）行为的倾向，并通过八个变体的实验框架探究驱动此类行为的因素。

    

    AI系统在许多领域中表现出危险行为的能力日益增强。这引出了一个关键问题：模型有时是否会选择违反人类指令，以执行对某些目标更有用的行为？我们提出了一个基准，用于测量基于终端的智能体中模型追求工具性趋同（Instrumental Convergence, IC）行为的倾向。这种行为（如自我保存）被假设在高度能力AI智能体的风险中扮演关键角色。我们的基准是现实的且低风险的，旨在减少评估意识和角色扮演带来的混淆因素。该测试套件包含七个操作性任务，每个任务都有一个官方工作流程和一个捷径，捷径的政策状态因条件而异。一个包含八个变体的共享框架通过改变监控、指令清晰度、风险程度、许可、工具性用途和被阻断的诚实路径等因素，来支持对驱动IC行为因素的推断。我们评估……

    arXiv:2605.06490v3 Announce Type: replace  Abstract: AI systems have become increasingly capable of dangerous behaviours in many domains. This raises the question: Do models sometimes choose to violate human instructions in order to perform behaviour that is more useful for certain goals? We introduce a benchmark for measuring model propensity for instrumental convergence (IC) behaviour in terminal-based agents. This is behaviour such as self-preservation that has been hypothesised to play a key role in risks from highly capable AI agents. Our benchmark is realistic and low-stakes, aiming to reduce evaluation-awareness and roleplay confounds. The suite contains seven operational tasks, each with an official workflow and a shortcut whose policy status varies by condition. An eight-variant shared framework varies monitoring, instruction clarity, stakes, permission, instrumental usefulness and blocked honest paths to support inferences regarding the factors driving IC behaviour. We evalua
    
[^342]: TACT：通过激活引导缓解编码智能体中的过度思考与过度行动

    TACT: Mitigating Overthinking and Overacting in Coding Agents via Activation Steering

    [https://arxiv.org/abs/2605.05980](https://arxiv.org/abs/2605.05980)

    提出 TACT 方法，通过在隐藏状态中识别可线性分离的“漂移轴”，在编码智能体出现过度思考或过度行动之前检测漂移，并在测试时通过激活引导将漂移行为拉回校准状态。

    

    当语言模型智能体处理复杂的软件工程任务时，它们往往会在长轨迹中逐渐退化，我们将这一现象定义为*智能体漂移*（agent drift）。我们聚焦于两种反复出现的失败模式——*过度思考*（overthinking）与*过度行动*（overacting）：前者指智能体反复对其已经掌握的信息进行推理，后者指智能体在未整合近期观察结果或未获取新证据的情况下就发出工具调用。本文提出 TACT（Think-Act Calibration via activation Steering，通过激活引导实现思考-行动校准），旨在智能体漂移演变为行为失败之前，就在残差流中检测并缓解该漂移。具体而言，我们将轨迹步骤标注为过度思考、过度行动或已校准，并发现这些状态的隐藏表示可以沿着两个*漂移轴*线性分离，这两个轴从已校准行为分别指向两种失败模式（AUC ≈ 0.9）。为缓解智能体漂移，我们在测试时将每一步的激活投影到这些漂移轴上，并将发生漂移的激活拉回已校准状态。

    arXiv:2605.05980v3 Announce Type: replace  Abstract: When language model agents tackle complex software engineering tasks, they often degrade over long trajectories, which we define as *agent drift*. We focus on two recurring failure modes *overthinking* and *overacting*, i.e., where the agent repeatedly reasons over information it already has, and where it issues tool calls without integrating recent observations or acquiring new evidence. In this paper, we introduce TACT (Think-Act Calibration via activation Steering), to detect and mitigate agent drift in the residual stream before it surfaces as a behavioral failure. In specific, we label trajectory steps as overthinking, overacting, or calibrated, and find that their hidden states can separate linearly along two *drift axes*, pointing from calibrated behavior toward each failure mode (AUC $\approx$ 0.9). To mitigate agent drift, we project each step's activation onto these axes at test time and pull drifted ones back toward the ca
    
[^343]: COMO：基于最小风险训练的闭环光学分子识别

    COMO: Closed-Loop Optical Molecule Recognition with Minimum Risk Training

    [https://arxiv.org/abs/2604.23546](https://arxiv.org/abs/2604.23546)

    该论文提出COMO框架，首次将最小风险训练引入光学化学结构识别任务，通过在分子级进行闭环直接优化来缓解暴露偏差，从而更好地优化化学有效性和结构相似性等分子级评估指标。

    

    光学化学结构识别（OCSR）旨在将分子图像转换为SMILES字符串或分子图等机器可读的表示形式，但由于化学结构变化多样、存在简写约定以及扫描伪影等因素，其在真实世界文档中的应用仍然极具挑战性。大多数现有的深度学习方法依赖于采用token级最大似然估计（MLE）的教师强制训练范式。这种训练范式存在暴露偏差问题，因为模型在训练时使用的是真实值前缀，而在推理时却必须以自身先前的预测结果为条件。此外，token级的MLE目标函数难以针对化学有效性和结构相似性等分子级评估指标进行优化。在此，我们将最小风险训练引入OCSR领域，并提出了COMO（闭环光学分子识别）框架，该框架通过直接在分子级上进行优化来缓解暴露偏差……（原文摘要在此处截断）

    arXiv:2604.23546v2 Announce Type: replace-cross  Abstract: Optical chemical structure recognition (OCSR) translates molecular images into machine-readable representations such as SMILES strings or molecular graphs, but remains challenging in real-world documents due to wide variations in chemical structures, shorthand conventions, and scanning artifacts. Most existing deep learning approaches rely on teacher forcing with token-level Maximum Likelihood Estimation (MLE). This training paradigm suffers from exposure bias, as models are trained under ground-truth prefixes but must condition on their own previous predictions during inference. Moreover, token-level MLE objectives hinder the optimization towards molecular-level evaluation criteria such as chemical validity and structural similarity. Here we introduce Minimum Risk Training (MRT) to OCSR and propose COMO (Closed-loop Optical Molecule recOgnition), a framework that mitigates exposure bias by directly optimizing over molecule-lev
    
[^344]: AnalogRetriever：面向模拟电路检索的跨模态表示学习

    AnalogRetriever: Learning Cross-Modal Representations for Analog Circuit Retrieval

    [https://arxiv.org/abs/2604.23195](https://arxiv.org/abs/2604.23195)

    提出了AnalogRetriever，这是首个统一的模拟电路三模态（网表、原理图、文本描述）跨模态检索框架，通过两阶段修复流程构建了编译率达99.97%的高质量数据集，并借助端口感知图卷积网络与视觉-语言模型及课程对比学习将三种模态映射到共享嵌入空间。

    

    模拟电路设计高度依赖于对现有知识产权（IP）的复用，然而在SPICE网表、原理图和功能描述等异构表示之间进行检索仍然具有挑战性。现有方法大多局限于单一模态内的精确匹配，无法捕捉跨模态的语义关系。为了弥补这一空白，我们提出了AnalogRetriever，一个用于模拟电路检索的统一三模态检索框架。我们首先在Masala-CHAI的基础上，通过两阶段修复流程构建了一个高质量数据集，将网表编译率从22.0%提升至99.97%。在此基础之上，AnalogRetriever利用视觉-语言模型对原理图和文本描述进行编码，并利用端口感知的关系图卷积网络对网表进行编码，通过课程对比学习将这三种模态映射到一个共享的嵌入空间中。实验表明，AnalogRetriever……

    arXiv:2604.23195v2 Announce Type: replace-cross  Abstract: Analog circuit design relies heavily on reusing existing intellectual property (IP), yet searching across heterogeneous representations such as SPICE netlists, schematics, and functional descriptions remains challenging. Existing methods are largely limited to exact matching within a single modality, failing to capture cross-modal semantic relationships. To bridge this gap, we present AnalogRetriever, a unified tri-modal retrieval framework for analog circuit search. We first build a high-quality dataset on top of Masala-CHAI through a two-stage repair pipeline that raises the netlist compile rate from 22.0\% to 99.97\%. Built on this foundation, AnalogRetriever encodes schematics and descriptions with a vision-language model and netlists with a port-aware relational graph convolutional network, mapping all three modalities into a shared embedding space via curriculum contrastive learning. Experiments show that AnalogRetriever 
    
[^345]: KAIROS：有状态、上下文感知的高能效智能体推理服务系统

    KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving

    [https://arxiv.org/abs/2604.16682](https://arxiv.org/abs/2604.16682)

    KAIROS提出将智能体上下文作为一等控制信号，联合管理GPU频率、每实例并发度与多实例请求放置，从而解决了传统降频方法在智能体AI服务中会引发内存颠簸、恶化性能与能效的问题，实现上下文感知的高效功率优化。

    

    功率已成为AI推理的核心瓶颈。随着智能体AI（Agentic AI）成为一类主要的工作负载，这一问题日益紧迫，然而以往的功率管理技术几乎完全聚焦于单轮LLM服务。我们的分析表明，智能体服务的行为有着根本性的不同：每个请求都携带着在工具交错的多轮交互中不断演进的长生命周期上下文，而降低GPU频率可能使系统进入颠簸（thrashing）状态，此时内存压力会急剧恶化性能与能效。这些观察表明，面向智能体服务的功率优化需要重新设计。我们提出了KAIROS，一个面向智能体AI服务的上下文感知功率优化系统。KAIROS将智能体上下文作为一等控制信号，联合管理GPU频率、每实例并发度以及多实例请求放置，从而在存在内存余量时节省功率，同时避免……（摘要原文在此处截断）

    arXiv:2604.16682v2 Announce Type: replace-cross  Abstract: Power has become a central bottleneck for AI inference. This problem is becoming more urgent as agentic AI emerges as a major workload class, yet prior power-management techniques focus almost entirely on single-turn LLM serving. Our analysis shows that agentic serving behaves fundamentally differently: each request carries long-lived context that evolves across tool-interleaved turns, and lowering GPU frequency can push the system into a thrashing regime where memory pressure sharply worsens both performance and power efficiency. These observations show that power optimization for agentic serving requires rethinking.   We present KAIROS, a context-aware power optimization system for agentic AI serving. KAIROS uses agent context as a first-class control signal to jointly manage GPU frequency, per-instance concurrency, and multi-instance request placement. This enables KAIROS to save power when memory headroom exists while avoid
    
[^346]: 基于大语言模型智能体的可解释迭代式数据可视化优化

    Explainable Iterative Data Visualization Refinement via an LLM Agent

    [https://arxiv.org/abs/2604.15319](https://arxiv.org/abs/2604.15319)

    该论文提出了一种基于大语言模型智能体的AI流水线，将可视化评估和超参数优化视为语义任务，通过融合定量评估与对人类定性判断的近似，实现可解释的迭代式数据可视化优化。

    

    arXiv:2604.15319v3 公告类型：replace-cross 摘要：高维数据的探索性分析依赖于将数据嵌入到低维空间中，并基于该嵌入生成可视化图表，以揭示有意义的结构并传达数据的几何与分布特征。寻找合适的嵌入算法配置（尤其是超参数设置），以生成能够忠实反映底层现实并促进模式发现的可视化图表，是一项劳动密集型且需要领域知识的任务。为了减轻这一负担，我们提出了一种智能体AI流水线，它将严格的定量评估与基于大语言模型（LLM）对定性人类判断和洞察的近似相结合。通过将可视化评估与超参数优化视为一项语义任务，我们的系统能够生成一份多方面的报告，将定量指标与描述性摘要相结合并置于具体情境中，并提出可操作的（建议……原文在此处截断）

    arXiv:2604.15319v3 Announce Type: replace-cross  Abstract: Exploratory analysis of high-dimensional data relies on embedding the data into a low-dimensional space, based on which a visualization plot is produced to uncover meaningful structures and to communicate geometric and distributional data characteristics. Finding a suitable embedding algorithm configuration, particularly the hyperparameter setting, to produce a visualization plot that faithfully represents the underlying reality and encourages pattern discovery is labor-intensive and requires domain knowledge. To ease this burden, we propose an agentic AI pipeline that integrates rigorous quantitative assessment with large language model (LLM) based approximation of qualitative human judgment and insight. By treating visualization evaluation and hyperparameter optimization as a semantic task, our system generates a multi-faceted report that contextualizes quantitative metrics with descriptive summaries, and suggests actionable 
    
[^347]: 图能量匹配：面向图生成的传输对齐能量建模方法

    Graph Energy Matching: Transport-Aligned Energy-Based Modeling for Graph Generation

    [https://arxiv.org/abs/2603.23398](https://arxiv.org/abs/2603.23398)

    提出图能量匹配（GEM）框架，通过对齐输运机制来克服离散能量模型的采样陷阱与训练不稳定问题，实现可融入结构或功能约束、可组合的图生成，并缩小与离散扩散模型的保真度差距。

    

    arXiv:2603.23398v4 公告类型：replace-cross。摘要：离散数据（如图）的生成建模是众多科学与工业应用的基础，包括分子发现与材料设计。在这些领域中，概率推断尤为宝贵，因为它支持可组合的生成，并能有原则性地融入期望的约束，如结构或功能性质。基于能量的模型通过捕捉相对似然，并允许在推断时直接施加约束，天然契合这一目标。然而，离散能量模型通常难以实现高效且高质量的采样：支持集之外的区域往往存在虚假的局部极小值，会使采样器陷入其中并导致训练不稳定，从而使其与离散扩散模型相比存在保真度差距。为弥合这一差距，我们提出了图能量匹配，一种受 Jordan-Kinderlehrer-Otto 输运机制启发的离散生成框架。（注：原文摘要在此处被截断）

    arXiv:2603.23398v4 Announce Type: replace-cross  Abstract: Generative modeling of discrete data, such as graphs, underpins many scientific and industrial applications, including molecular discovery and materials design. In these domains, probabilistic inference is particularly valuable, as it enables composable generation and principled incorporation of desired constraints, such as structural or functional properties. Energy-based models naturally support this goal by capturing relative likelihoods and enabling composable inference by directly enforcing constraints during inference. However, discrete energy-based models typically struggle with efficient and high-quality sampling, as off-support regions often contain spurious local minima, trapping samplers and causing training instabilities, resulting in a fidelity gap compared to discrete diffusion models. To address this gap, we introduce Graph Energy Matching (GEM), a discrete generative framework inspired by the Jordan-Kinderlehrer
    
[^348]: RECAP：局部赫布原型学习作为储备池动力学的自组织读出层

    RECAP: Local Hebbian Prototype Learning as a Self-Organizing Readout for Reservoir Dynamics

    [https://arxiv.org/abs/2603.06639](https://arxiv.org/abs/2603.06639)

    提出了一种名为RECAP的仿生学习策略，将未经训练的储备池动力学与自组织赫布原型读出相结合，仅通过局部可塑性规则即可实现无需反向传播的稳健图像分类。

    

    大脑中稳健的感知通常归因于高维的群体活动，以及能够强化重复出现结构的局部可塑性机制。相比之下，大多数现代图像识别系统通过误差反向传播和端到端梯度优化进行训练，这些方法与局部计算和局部可塑性并不天然契合。我们提出了RECAP（基于赫布共激活原型的储备池计算），这是一种用于稳健图像分类的仿生学习策略，它将未经训练的储备池动力学与自组织的赫布原型读出层相结合。RECAP将时间平均的储备池响应离散化为激活水平，在储备池单元对之间构建共激活掩码，并通过类赫布的增强-衰减规则增量式地更新类别原型矩阵。推理通过基于重叠度的原型匹配来执行。该方法避免了误差反向传播

    arXiv:2603.06639v2 Announce Type: replace-cross  Abstract: Robust perception in brains is often attributed to high-dimensional population activity together with local plasticity mechanisms that reinforce recurring structure. In contrast, most modern image recognition systems are trained by error backpropagation and end-to-end gradient optimization, which are not naturally aligned with local computation and local plasticity. We introduce RECAP (Reservoir Computing with Hebbian Co-Activation Prototypes), a bio-inspired learning strategy for robust image classification that couples untrained reservoir dynamics with a self-organizing Hebbian prototype readout. RECAP discretizes time-averaged reservoir responses into activation levels, constructs a co-activation mask over reservoir unit pairs, and incrementally updates class-wise prototype matrices via a Hebbian-like potentiation-decay rule. Inference is performed by overlap-based prototype matching. The method avoids error backpropagation 
    
[^349]: 预条件流匹配

    Preconditioned Flow Matching

    [https://arxiv.org/abs/2603.02337](https://arxiv.org/abs/2603.02337)

    该论文揭示了流匹配训练中由中间分布协方差病态条件导致的几何优化瓶颈——高方差方向拟合快而低方差方向进展慢，并据此提出“先预条件、再匹配”的预条件流匹配框架，通过变换目标分布来加速和稳定训练。

    

    流匹配（FM）通过沿中间分布 $p_t$ 回归随机速度目标来学习向量场。我们在此回归问题中发现了一个几何优化瓶颈：当 $p_t$ 的协方差 $\Sigma_t$ 呈病态条件时，基于梯度的训练会快速拟合高方差方向，而沿低方差方向的进展却很缓慢。在一个可精确求解的高斯设定中，我们证明超额风险由 $\Sigma_t$ 加权，且梯度下降和随机梯度下降的收敛性都依赖于条件数。随后，我们将分析扩展到高斯混合分布，表明多峰性并不能平均掉这一效应；相反，最慢且条件最差的分量可能主导整个优化过程。受此分析启发，我们提出了预条件流匹配，这是一种“先预条件、再匹配”的框架，它将目标分布变换为更具……（原文摘要在此处被截断）

    arXiv:2603.02337v3 Announce Type: replace-cross  Abstract: Flow matching (FM) learns vector fields by regressing stochastic velocity targets along intermediate distributions $p_t$. We identify a geometric optimization bottleneck in this regression problem: when the covariance $\Sigma_t$ of $p_t$ is ill-conditioned, gradient-based training rapidly fits high-variance directions while making slow progress along low-variance ones. In an exactly solvable Gaussian setting, we prove that the excess risk is weighted by $\Sigma_t$, and that both gradient descent and stochastic gradient descent inherit condition-number-dependent convergence. We then extend the analysis to Gaussian mixtures, showing that multimodality does not average away this effect; instead, the slowest and worst-conditioned component can control optimization. Motivated by this analysis, we propose $\textit{preconditioned flow matching}$, a precondition-then-match framework that transforms the target distribution into a more i
    
[^350]: MMCOMET：一个面向视觉推理的大规模多模态常识知识图谱

    MMCOMET: A Large-Scale Multimodal Commonsense Knowledge Graph for Visual Reasoning

    [https://arxiv.org/abs/2603.01055](https://arxiv.org/abs/2603.01055)

    提出了MMCOMET，一个将ATOMIC2020常识三元组与图像对齐的大规模多模态常识知识图谱，构建了超过90万条涵盖19种关系类型的多模态元组，并建立了涵盖多种视觉-语言任务的统一评估框架。

    

    常识知识被广泛用于支持多模态推理与生成，然而现有资源主要以纯文本形式存在，缺乏与视觉语境的关联。这一局限常常导致检索到语义相关但视觉上不匹配的知识，并可能降低下游任务的性能。我们提出了MMCOMET，一个将物理、社会和事件知识与视觉落地相整合的大规模多模态常识知识图谱。MMCOMET在ATOMIC2020的基础上，通过混合对齐流水线将每条常识三元组与代表性图像相关联，形成了超过90万个涵盖19种关系类型的多模态元组。我们进一步建立了一个统一的评估框架，涵盖多种视觉-语言任务，包括视觉常识推理、视觉问答、视觉故事生成、图像编辑和图像描述生成。我们的分析表明，多模态落地能提供更多（摘要在此处被截断）

    arXiv:2603.01055v3 Announce Type: replace  Abstract: Commonsense knowledge is widely used to support multimodal reasoning and generation, yet existing resources are predominantly text-only and lack grounding in visual context. This limitation often leads to the retrieval of semantically related but visually misaligned knowledge, and can degrade downstream performance. We introduce MMCOMET, a large-scale multimodal commonsense knowledge graph that integrates physical, social, and eventive knowledge with visual grounding. MMCOMET extends ATOMIC2020 by associating each commonsense triple with representative images through a hybrid alignment pipeline, resulting in over 900K multimodal tuples spanning 19 relation types. We further establish a unified evaluation framework across diverse vision-language tasks, including visual commonsense reasoning, visual question answering, visual storytelling, image editing, and image captioning. Our analysis shows that multimodal grounding provides more c
    
[^351]: 基于Gossip算法的去中心化排名聚合：收敛性与鲁棒性

    Decentralized Ranking Aggregation via Gossip: Convergence and Robustness

    [https://arxiv.org/abs/2602.22847](https://arxiv.org/abs/2602.22847)

    本文针对偏好数据分布在通信网络中的去中心化场景，研究如何利用gossip算法实现排名聚合，并提供收敛性保证与对污染数据的鲁棒性。

    

    排名聚合的概念在偏好分析中扮演着核心角色。文献中已记录了大量用于计算中位数排名的算法，这些算法通常源于社会选择理论，并在集中式设置下（即所有待聚合的排名数据能够汇集到单一计算单元时）提供理论保证。对于许多技术（例如点对点网络、物联网、多智能体系统）而言，在去中心化环境下——即偏好数据最初分布在通信网络中时——扩展计算共识排名的能力，同时保证收敛性以及对潜在污染数据的鲁棒性，仍然是一个重大的方法论挑战。事实上，近年来关于去中心化计算的文献主要集中于使用gossip算法来计算或优化诸如算术平均值之类的统计量。本文的目的……（原文摘要在此处截断）

    arXiv:2602.22847v3 Announce Type: replace-cross  Abstract: The concept of ranking aggregation plays a central role in preference analysis, and numerous algorithms for calculating median rankings, often originating in social choice theory, have been documented in the literature, offering theoretical guarantees in a centralized setting, \textit{i.e.}, when all the ranking data to be aggregated can be brought together in a single computing unit. For many technologies (\textit{e.g.} peer-to-peer networks, IoT, multi-agent systems), extending the ability to calculate consensus rankings with guarantees of convergence and resilience to potential contamination in a decentralized setting, when preference data is initially distributed across a communicating network, remains a major methodological challenge. Indeed, in recent years, the literature on decentralized computation has mainly focused on computing or optimizing statistics such as arithmetic means using gossip algorithms. The purpose of 
    
[^352]: 软集理论及其扩展的动态综述

    A Dynamic Survey of Soft Set Theory and Its Extensions

    [https://arxiv.org/abs/2602.21268](https://arxiv.org/abs/2602.21268)

    该论文全面综述了软集理论及其主要扩展（包括超软集、TreeSoft 集、双极软集和动态软集等），系统梳理了参数空间、值空间与代表性数学构造，为不同框架间的严格比较奠定了基础。

    

    软集理论通过将每个参数与给定论域的子集相关联，为参数化决策提供了一个系统性的框架，从而实现对不确定性的结构化表示 [1, 2]。在过去几十年中，该理论已发展出广泛的扩展形式，包括超软集、超超软集、TreeSoft 集、双极软集和动态软集，同时还与多个数学学科建立了联系，特别是拓扑学和拟阵理论。本书对软集理论及其主要扩展进行了全面的综述，特别强调参数空间、值空间、具有代表性的数学构造，以及在不同框架之间进行严格比较所必需的基本假设。书中仔细区分了文献中已确立的定义与探索性的表述，并批判性地审视了……的局限性（摘要内容在此处截断）。

    arXiv:2602.21268v2 Announce Type: replace  Abstract: Soft set theory provides a systematic framework for parameterized decision-making by associating each parameter with a subset of a given universe, thereby enabling structured representations of uncertainty [1, 2]. Over the past decades, the theory has evolved into a broad range of extensions, including hypersoft sets, superhypersoft sets, TreeSoft sets, bipolar soft sets, and dynamic soft sets, while also establishing connections with various mathematical disciplines, particularly topology and matroid theory. This book presents a comprehensive survey of soft set theory and its principal extensions, with particular emphasis on parameter spaces, value spaces, representative mathematical constructions, and the underlying assumptions necessary for rigorous comparisons among different frameworks. It carefully distinguishes established definitions from the literature from exploratory formulations and critically examines the limitations of 
    
[^353]: 基于贝叶斯自适应提问的冷启动个性化

    Cold-Start Personalization via Bayesian Adaptive Questioning

    [https://arxiv.org/abs/2602.15012](https://arxiv.org/abs/2602.15012)

    提出CAPE方法，先离线从完整用户画像中学习偏好相关性的结构化世界模型，再在线通过免训练的贝叶斯推断选择信息量最大的问题并预测完整偏好画像，即使使用简单的线性回归信念模型也能大幅超越端到端强化学习。

    

    冷启动个性化需要在没有用户特定历史数据可用的情况下，仅通过极少量的交互来推断用户偏好。可能的偏好空间非常庞大，但用户只关心其中稀疏的子集，且很少会主动明确表达这些偏好；再加上有限的交互预算，这使得偏好引出（elicitation）变得极具挑战性。我们的关键洞察是：偏好在不同用户群体之间表现出可预测的结构；例如，希望获得详细解释的用户通常也重视完整的示例演算。我们提出了CAPE（冷启动自适应偏好引出，Cold-start Adaptive Preference Elicitation），该方法先离线地从完整用户画像中学习一个关于偏好相关性的结构化世界模型，然后在线执行无需训练的贝叶斯推断，以选择信息量最大的问题并预测完整的偏好画像，甚至包括从未询问过的维度。即使是简单的信念模型实例（如线性回归），其表现也能大幅超越端到端的强化学习方法。

    arXiv:2602.15012v2 Announce Type: replace-cross  Abstract: Cold-start personalization requires inferring preferences from minimal interaction when no user-specific historical data is available. The space of possible preferences is vast, yet users care about only a sparse subset and rarely articulate them upfront; combined with limited interaction budgets, this makes preference elicitation challenging. Our key insight is that preferences exhibit predictable structure across populations; e.g., users who want detailed explanations often also value worked examples. We propose CAPE (Cold-start Adaptive Preference Elicitation), which learns a structured world model of preference correlations offline from complete profiles, then performs training-free Bayesian inference online to select informative questions and predict complete preference profiles, including dimensions never asked about. Even simple belief model instantiations (e.g., linear regression) substantially outperform end-to-end RL.
    
[^354]: OmniReview：面向现实审稿人推荐的大规模基准数据集与大语言模型增强框架

    OmniReview: A Large-scale Benchmark and LLM-enhanced Framework for Realistic Reviewer Recommendation

    [https://arxiv.org/abs/2602.08896](https://arxiv.org/abs/2602.08896)

    该论文提出了OmniReview，通过构建包含202,756条验证审稿记录的大规模基准数据集、三层分级评估框架以及大语言模型增强方法，解决了审稿人推荐中数据稀缺、评估指标过于简化以及嵌入方法信息瓶颈与可解释性不足的问题。

    

    学术同行评审仍然是学术验证的基石，然而该领域在数据和方法方面面临诸多挑战。从数据角度来看，现有研究受制于大规模、经过验证的基准数据集的稀缺，以及无法反映真实编辑工作流程的过度简化的评估指标。为弥补这一差距，我们提出了OmniReview，这是一个通过整合多源学术平台构建的综合数据集，经由消歧处理管道涵盖全面的学者档案，最终产出202,756条经过验证的审稿记录。基于这些数据，我们引入了一个三层分级评估框架，从召回层面到精确的专家识别层面来评估推荐效果。从方法角度来看，现有的基于嵌入的方法存在语义压缩导致的信息瓶颈以及可解释性有限的问题。为解决这些方法上的局限性，我们提出了Profi（原文摘要在此处截断）

    arXiv:2602.08896v2 Announce Type: replace-cross  Abstract: Academic peer review remains the cornerstone of scholarly validation, yet the field faces some challenges in data and methods. From the data perspective, existing research is hindered by the scarcity of large-scale, verified benchmarks and oversimplified evaluation metrics that fail to reflect real-world editorial workflows. To bridge this gap, we present OmniReview, a comprehensive dataset constructed by integrating multi-source academic platforms encompassing comprehensive scholarly profiles through the disambiguation pipeline, yielding 202, 756 verified review records. Based on this data, we introduce a three-tier hierarchical evaluaion framework to assess recommendations from recall to precise expert identification. From the method perspective, existing embedding-based approaches suffer from the information bottleneck of semantic compression and limited interpretability. To resolve these method limitations, we propose Profi
    
[^355]: BrainVista：将自然主义大脑动态建模为多模态下一词元预测

    BrainVista: Modeling Naturalistic Brain Dynamics as Multimodal Next-Token Prediction

    [https://arxiv.org/abs/2602.04512](https://arxiv.org/abs/2602.04512)

    BrainVista提出了一种将自然主义大脑动态建模为多模态“下一词元预测”的自回归框架，通过历史大脑活动与时间对齐的刺激词元来预测未来fMRI状态，无需未来真实数据即可实现大脑状态演化的自回归预测。

    

    自然主义fMRI为连续多模态体验期间的皮层动态提供了时间分辨的观察窗口，其中未来大脑状态由内在神经上下文与外在感觉驱动共同决定。然而，预测这种动态仍面临挑战：快速变化的刺激流必须与缓慢的血流动力学BOLD响应相协调，同时皮层活动又分布于功能异质的大脑网络之中。为应对这些挑战，我们提出了BrainVista——一个用于观察刺激条件下大脑状态预测的多模态自回归框架。BrainVista基于大脑活动历史以及与每个预测查询在时间上对齐的刺激词元来预测未来的fMRI状态，从而在无需访问未来真实fMRI状态的情况下实现自回归预测。递归展开机制为持续感觉输入下大脑状态的演化提供了单次运行内的预测性检验……

    arXiv:2602.04512v2 Announce Type: replace-cross  Abstract: Naturalistic fMRI offers a time-resolved window into cortical dynamics during continuous multimodal experience, where future brain states are jointly governed by endogenous neural context and exogenous sensory drive. Forecasting such dynamics, however, remains challenging because fast-changing stimulus streams must be reconciled with the sluggish hemodynamic BOLD response, while cortical activity is organized across functionally heterogeneous brain networks. To address these challenges, we introduce BrainVista, a multimodal autoregressive framework for observed-stimulus conditional brain-state forecasting. BrainVista predicts future fMRI states from the history of brain activity and stimulus tokens temporally aligned to each prediction query, enabling autoregressive prediction without access to future ground-truth fMRI states. Recursive rollout provides a within-run predictive test of brain-state evolution under continuing sens
    
[^356]: 对齐图像与长文本描述中的“森林”与“树木”以实现视觉接地理解

    Aligning Forest and Trees in Images & Long Captions for Visually Grounded Understanding

    [https://arxiv.org/abs/2602.02977](https://arxiv.org/abs/2602.02977)

    提出CAFT模型，通过“由细到粗”的层次化学习同时实现局部文本-区域对齐与全局图像-文本对齐，使视觉-语言模型能够忠实理解长而细节丰富的图像描述。

    

    诸如CLIP等视觉-语言模型往往难以忠实理解长而细节丰富的描述文本，它们依赖主导性的场景线索而忽视细粒度的视觉证据。我们提出了一种层次化的视觉-语言学习原则，将场景理解为部分到整体的组合：在形成整体场景表示之前，模型应该揭示哪些语义部分出现在图像中的什么位置。为此，我们提出了CAFT（森林与树木的跨域对齐，Cross-domain Alignment of Forest and Trees），这是一种视觉-语言模型，它在中间表示层面联合学习局部文本-区域对齐，并在最终表示层面学习全局图像-文本对齐。利用长文本描述的组织结构（其中局部描述通常对应场景的各个部分），CAFT采用由细到粗的图像编码器和部分-整体的文本编码器，以发现局部化的部分语义，并逐步将其组合成全局的图像-文本表示。

    arXiv:2602.02977v3 Announce Type: replace-cross  Abstract: Vision-language models such as CLIP often struggle to faithfully understand long, detail-rich captions, relying on dominant scene cues while overlooking fine-grained visual evidence. We propose a hierarchical vision-language learning principle for understanding scenes as part-to-whole compositions: before forming a whole-scene representation, a model should uncover what semantic parts appear where in the image. To this end, we propose CAFT (Cross-domain Alignment of Forest and Trees), a vision-language model that jointly learns local text-region alignment at intermediate representations and global image-text alignment at the final representation. Exploiting the organization of long captions, where local descriptions often correspond to scene parts, CAFT employs a fine-to-coarse image encoder and a part-whole text encoder to discover localized part semantics and progressively compose them into a global image-text representation.
    
[^357]: 面向高效多步大语言模型交互的动态混合精度路由

    Dynamic Mixed-Precision Routing for Efficient Multi-step LLM Interaction

    [https://arxiv.org/abs/2602.02711](https://arxiv.org/abs/2602.02711)

    该论文提出动态混合精度路由（DMR）框架，通过识别仅约1%的对量化敏感的“关键步骤”，在每个决策步骤自适应地在大语言模型的高精度与低精度版本之间切换，从而在大幅降低推理成本的同时保持任务成功率。

    

    大语言模型（LLM）通过测试时的多步交互与推理，在长时程智能体任务中取得了出色表现。然而，反复调用这些模型会在整个任务过程中产生高昂的推理成本。训练后量化（PTQ）可以降低这一成本，但在整个过程中始终使用固定的低精度会显著降低任务成功率。我们观察到，不同交互步骤对量化的敏感程度差异极大：在相同上下文条件下，仅有约1%的步骤在低精度解码时会将成功变为失败，我们将这些步骤称为“关键步骤”。基于这一观察，我们提出了动态混合精度路由（DMR），这是一个能够在每个决策步骤中自适应地在同一大语言模型的高精度版本与低精度版本之间进行选择的框架。DMR利用全空间分解来联合预测某一步骤的偏差及其对最终结果的影响，从而学会识别关键步骤（原文在此处截断）。

    arXiv:2602.02711v4 Announce Type: replace  Abstract: Large language models (LLMs) achieve strong performance in long-horizon agentic tasks through multi-step interaction and reasoning at test time. However, repeatedly invoking these models incurs substantial inference cost over an episode. Post-training quantization (PTQ) reduces this cost, but using a fixed low precision throughout an episode can sharply lower task success. We observe that interaction steps differ sharply in their sensitivity to quantization: given the same context, decoding a step at low precision turns a success into a failure at only about 1% of the steps, which we call critical steps. Based on this observation, we propose Dynamic Mixed-Precision Routing (DMR), a framework that adaptively selects between high-precision and low-precision versions of the same LLM at each decision step. Leveraging an entire-space factorization to jointly predict a step's divergence and its effect on the outcome, DMR learns to recogniz
    
[^358]: FiLoRA：用于可控特征依赖的“聚焦与忽略”LoRA方法

    FiLoRA: Focus-and-Ignore LoRA for Controllable Feature Reliance

    [https://arxiv.org/abs/2602.02060](https://arxiv.org/abs/2602.02060)

    FiLoRA通过指令条件化的低秩门控机制，实现了在不改变任务目标的前提下，用自然语言指令直接控制多模态模型对特定特征路径的依赖，从而缓解捷径和虚假相关行为。

    

    多模态基础模型整合了跨模态的异构信号，但目前尚不清楚是否可以通过显式调节对不同内部特征路径的依赖来控制其预测。现有的处理捷径行为或虚假相关的方法主要依赖事后分析或数据层面的干预，难以直接干预模型对信息的使用方式。我们提出了FiLoRA（聚焦与忽略LoRA），一种指令条件化的、参数高效的适应框架，能够在保持任务和预测目标不变的情况下，实现对特征依赖的可控调节。FiLoRA将适应过程分解为特征对齐的低秩模块，并应用指令条件化门控，使自然语言指令能够作为计算层面的控制信号作用于内部表示。我们在受控分类设置及地理相关任务上评估了FiLoRA的性能。

    arXiv:2602.02060v2 Announce Type: replace-cross  Abstract: Multimodal foundation models integrate heterogeneous signals across modalities, yet it remains unclear whether their predictions can be controlled by explicitly modulating reliance on different internal feature pathways. Existing approaches to shortcut and spurious behavior primarily rely on post hoc analysis or data-level interventions, offering limited ability to directly intervene on how models use information. We introduce FiLoRA (Focus-and-Ignore LoRA), an instruction-conditioned, parameter-efficient adaptation framework that enables controllable modulation of feature reliance while keeping the task and predictive objective fixed. FiLoRA decomposes adaptation into feature-aligned low-rank modules and applies instruction-conditioned gating, allowing natural language instructions to act as computation-level control signals over internal representations. We evaluate FiLoRA across both controlled classification settings and ge
    
[^359]: 自我改进的训练与思考

    Self-Improving Training & Thinking

    [https://arxiv.org/abs/2601.21343](https://arxiv.org/abs/2601.21343)

    该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。

    

    传统上，大型语言模型采用分阶段训练：先在原始文本上进行预训练，再进行后训练以学习指令遵循和推理能力。然而，这种分离造成了根本性的局限：许多理想的行为，如安全性、事实准确性、整体生成质量和推理能力，只在训练后期才被加入，尽管早期学习的模式在很大程度上塑造了模型的能力。为解决这一问题，我们提出了一种新的预训练和中期训练方法，将这些行为更早地融入训练过程。我们利用一个现有的强大的经过后训练的模型来重写数据并评判策略模型的输出（rollouts），从而在训练中更早地引入思考和强化学习。我们进一步通过引入“自我改进的后思考”（self-improving post-thinking）的概念，在所有训练阶段提升推理能力：即在生成之后进行推理，以提高模型的推理和自我评判能力。在实验中，我们证明了这些方法……

    arXiv:2601.21343v4 Announce Type: replace-cross  Abstract: Large language models are classically trained in stages: pretraining on raw text followed by post-training for instruction following and reasoning. However, this separation creates a fundamental limitation: many desirable behaviors such as safety, factuality, overall generation quality, and reasoning ability are only added at a late stage, even though the patterns learned earlier strongly shape a model's capabilities. To tackle this issue, we introduce a new way to pretrain and mid-train models that incorporates these behaviors earlier. We utilize an existing strong, post-trained model to both rewrite data and to judge policy model rollouts, thus introducing thinking and reinforcement earlier in training. We further improve reasoning at all stages by introducing the concept of self-improving post-thinking: improving the model's reasoning and self-judging ability by reasoning after generation. In our experiments, we show these a
    
[^360]: 精选的内镜逆行胰胆管造影（ERCP）图像数据集

    Curated endoscopic retrograde cholangiopancreatography images dataset

    [https://arxiv.org/abs/2601.16759](https://arxiv.org/abs/2601.16759)

    本文发布了一个由资深专家标注的大型精选ERCP图像数据集（含19,018张原始图像和5,519张标注图像），为胆胰疾病的AI自动诊断研究提供了宝贵的基准数据资源。

    

    内镜逆行胰胆管造影术（ERCP）是胆道和胰腺疾病诊断与治疗的关键操作，而人工智能被认为是实现诊断自动化的一种解决方案。然而，公开的ERCP数据集十分稀缺，这限制了该方法的应用。因此，本研究旨在通过提供一个大型且经过精心整理的数据集来帮助填补这一空白。该数据集包含来自1,602名患者的19,018张原始图像和19,317张处理后图像，其中5,519张图像已标注，可直接使用。所有图像均由两名具有5年以上经验的胃肠病学家人工检查和标注，并由另一位具有20年以上经验的胃肠病学家审核，三位专家每年均完成400例以上的ERCP操作。数据集的实用性和有效性已通过一项分类实验得到验证。该数据集旨在为自动化诊断领域提供基准或作出贡献。

    arXiv:2601.16759v2 Announce Type: replace-cross  Abstract: Endoscopic Retrograde Cholangiopancreatography (ERCP) is a key procedure in the diagnosis and treatment of biliary and pancreatic diseases. Artificial intelligence has been pointed as one solution to automatize diagnosis. However, public ERCP datasets are scarce, which limits the use of such approach. Therefore, this study aims to help fill this gap by providing a large and curated dataset. The collection is composed of 19.018 raw images and 19.317 processed from 1.602 patients. 5.519 images are labeled, which provides a ready to use dataset. All images were manually inspected and annotated by two gastroenterologist with more than 5 years of experience and reviewed by another gastroenterologist with more than 20 years of experience, all with more than 400 ERCP procedures annually. The utility and validity of the dataset is proven by a classification experiment. This collection aims to provide or contribute for a benchmark in au
    
[^361]: 追踪大型语言模型中的道德基础

    Tracing Moral Foundations in Large Language Models

    [https://arxiv.org/abs/2601.05437](https://arxiv.org/abs/2601.05437)

    该研究通过对14个大型语言模型进行逐层分析、稀疏自编码器特征识别和因果引导干预，证明模型内部存在与人类道德判断相一致的道德基础概念结构，而非仅仅是表面的“道德模仿”。

    

    大型语言模型常常产生类似人类的道德判断，但目前尚不清楚这究竟反映了其内部的道德概念结构，还是仅仅是一种表面的“道德模仿”。本研究以道德基础理论（Moral Foundations Theory, MFT）作为分析框架，研究了道德基础如何在14个基础模型和指令微调模型中被编码、组织和表达，这些模型涵盖四个模型家族（Llama、Qwen2.5、Qwen3-MoE、Mistral），参数规模从7B到70B。我们采用了多层次的研究方法，结合：(i) 对MFT概念表示的逐层分析及其与人类道德感知的对齐程度；(ii) 在残差流上使用预训练的稀疏自编码器（SAEs）来识别支持道德概念的稀疏特征；(iii) 使用稠密MFT向量与稀疏SAE特征进行因果引导干预。我们发现，模型表示和区分道德基础的方式与人类道德判断相一致，并且这种道德几何结构……

    arXiv:2601.05437v4 Announce Type: replace-cross  Abstract: Large language models often produce human-like moral judgments, but it is unclear whether this reflects an internal conceptual structure or superficial ``moral mimicry.'' Using Moral Foundations Theory (MFT) as an analytic framework, we study how moral foundations are encoded, organized, and expressed across 14 base and instruction-tuned LLMs spanning four model families (Llama, Qwen2.5, Qwen3-MoE, Mistral) and scales from 7B to 70B. We employ a multi-level approach combining (i) layer-wise analysis of MFT concept representations and their alignment with human moral perceptions, (ii) pretrained sparse autoencoders (SAEs) over the residual stream to identify sparse features that support moral concepts, and (iii) causal steering interventions using dense MFT vectors and sparse SAE features. We find that models represent and distinguish moral foundations in a manner that aligns with human judgments, and that this moral geometry na
    
[^362]: 基于弱时间监督的遥感变化检测

    Remote Sensing Change Detection via Weak Temporal Supervision

    [https://arxiv.org/abs/2601.02126](https://arxiv.org/abs/2601.02126)

    该论文提出一种弱时间监督策略，通过为现有单时相遥感数据集补充不同时间的新观测数据，并假设真实双时相图像对大多无变化、以跨地点图像配对生成变化样本，从而在无需任何新标注的情况下训练遥感语义变化检测模型。

    

    遥感中的语义变化检测旨在识别双时相图像对之间的土地覆盖变化。由于像素级标注成本高昂且耗时，该领域的进展一直受限于标注数据集的稀缺。为解决这一问题，近期的方法利用合成数据或生成人工变化样本对，但在不同数据集和采集条件之间迁移变化检测模型仍然具有挑战性。在本工作中，我们提出了一种弱时间监督策略，该策略利用现有单时相数据集的额外时间观测数据，而无需任何新的标注。具体而言，我们通过在不同时间获取的新观测数据来扩展单时相遥感数据集，在训练变化检测模型时假设真实的双时相图像对大多不包含变化，同时将来自不同地点的图像进行配对以生成变化样本。为了处理固有的…

    arXiv:2601.02126v3 Announce Type: replace-cross  Abstract: Semantic change detection in remote sensing aims to identify land cover changes between bi-temporal image pairs. Progress in this area has been limited by the scarcity of annotated datasets, as pixel-level annotation is costly and time-consuming. To address this, recent methods leverage synthetic data or generate artificial change pairs, but transferring change detection models across datasets and acquisition conditions remains challenging. In this work, we introduce a weak temporal supervision strategy that leverages additional temporal observations of existing single-temporal datasets, without requiring any new annotations. Specifically, we extend single-date remote sensing datasets with new observations acquired at different times and train a change detection model by assuming that real bi-temporal pairs mostly contain no change, while pairing images from different locations to generate change examples. To handle the inheren
    
[^363]: 多智能体协同重命名重构

    Multi-Agent Coordinated Rename Refactoring

    [https://arxiv.org/abs/2601.00482](https://arxiv.org/abs/2601.00482)

    该论文提出了一种多智能体协同方法来执行重命名重构，让AI智能体与开发者协作完成跨多个相关标识符的协同重命名，克服了传统启发式方法误报过多和普通大语言模型建议不完整的问题。

    

    AI智能体在软件开发中的主要价值在于其能够扩展开发者的推理与行动能力，而非取代人类的参与。为了展示如何让智能体与开发者协同工作，我们设计了一种新颖的协同重命名执行方法。协同重命名是指一次重命名重构会触发多个相关标识符的重构，这是一项常见但具有挑战性的任务。开发者必须手动将这些重命名重构传播到众多文件和上下文中，这一过程既繁琐又极易出错。最先进的基于启发式的方法会产生大量令人难以承受的误报，而普通的大语言模型（LLM）由于其上下文有限且无法与重构工具交互，只能提供不完整的建议。这使得开发者要么面对不完整的重构结果，要么背负过滤过多建议的沉重负担……

    arXiv:2601.00482v2 Announce Type: replace-cross  Abstract: The primary value of AI agents in software development lies in their ability to extend the developer's capacity for reasoning and action, not to supplant human involvement. To showcase how to use agents working in tandem with developers, we designed a novel approach for carrying out coordinated renaming. Coordinated renaming, where a single rename refactoring triggers refactorings in multiple, related identifiers, is a frequent yet challenging task. Developers must manually propagate these rename refactorings across numerous files and contexts, a process that is both tedious and highly error-prone. State-of-the-art heuristic-based approaches produce an overwhelming number of false positives, while vanilla Large Language Models (LLMs) provide incomplete suggestions due to their limited context and inability to interact with refactoring tools. This leaves developers with incomplete refactorings or burdens them with filtering too 
    
[^364]: DermaVQA-DAS：皮肤病评估模式（DAS）及面向患者生成皮肤病图像的封闭式问答与分割数据集

    DermaVQA-DAS: Dermatology Assessment Schema (DAS) & Datasets for Closed-Ended Question Answering & Segmentation in Patient-Generated Dermatology Images

    [https://arxiv.org/abs/2512.24340](https://arxiv.org/abs/2512.24340)

    该论文提出了DermaVQA-DAS数据集及专家开发的皮肤病评估模式（DAS），通过36个高层级和27个细粒度的标准化评估问题，为患者生成的皮肤病图像上的封闭式问答和皮损分割两大任务提供了专家标注数据与基准。

    

    皮肤病学图像分析的最新进展很大程度上得益于大规模标注数据集；然而，现有的大多数基准数据集主要聚焦于皮肤镜图像，缺乏患者自述的查询和临床背景信息，这限制了其在以患者为中心的医疗护理中的适用性。为填补这一空白，我们提出了DermaVQA-DAS，它是DermaVQA数据集的扩展版本，支持两项互补任务：封闭式问答（QA）和皮肤病皮损分割。本工作的核心是皮肤病评估模式，这是一个由专家开发的新型框架，能够以结构化和标准化的形式系统性捕捉具有临床意义的皮肤病特征。DAS包含36个高层级和27个细粒度的评估问题，并配有英语和中文的选择题选项。利用DAS，我们为封闭式问答和分割任务提供了专家标注的数据集，并对最先进的多模态（模型进行了基准测试，摘要在此处截断）。

    arXiv:2512.24340v2 Announce Type: replace-cross  Abstract: Recent advances in dermatological image analysis have been driven by large-scale annotated datasets; however, most existing benchmarks focus on dermatoscopic images and lack patient-authored queries and clinical context, limiting their applicability to patient-centered care. To address this gap, we introduce DermaVQA-DAS, an extension of the DermaVQA dataset that supports two complementary tasks: closed-ended question answering (QA) and dermatological lesion segmentation. Central to this work is the Dermatology Assessment Schema (DAS), a novel expert-developed framework that systematically captures clinically meaningful dermatological features in a structured and standardized form. DAS comprises 36 high-level and 27 fine-grained assessment questions, with multiple-choice options in English and Chinese. Leveraging DAS, we provide expert-annotated datasets for both closed QA and segmentation and benchmark state-of-the-art multimo
    
[^365]: 论AI算法进步之源：来自语言模型预训练的证据

    On the Origin of Algorithmic Progress in AI: Evidence from Language Model Pre-Training

    [https://arxiv.org/abs/2511.21622](https://arxiv.org/abs/2511.21622)

    该论文通过消融与缩放实验发现，2012–2023年间AI预训练约22,000倍的算法效率提升无法由小算力下的各项具体创新解释，而主要源于Transformer等算法随规模增大而增强的效率优势。

    

    据估计，在2012年至2023年间，算法使AI预训练的FLOP效率提升了22,000倍（Ho et al., 2024）。通过对该时期关键创新开展消融实验，我们在小算力规模下只能解释其中不到10倍的收益。在调研更广泛的文献后，我们估计未纳入消融实验的其他创新的贡献同样不足10倍，因此在低算力区间，总效率提升不足100倍。这促使我们开展缩放实验，实验证据表明，这一效率差距可以由具有规模依赖性效率提升的算法来解释。具体而言，我们在LSTM与Transformer之间进行了缩放实验，发现了二者在计算最优缩放律上存在指数差异的证据，而许多其他创新则几乎没有表现出缩放差异。这些实验表明，与标准假设相反……

    arXiv:2511.21622v2 Announce Type: replace-cross  Abstract: Algorithms have been estimated to increase AI pretraining FLOP efficiency by a factor of 22,000 between 2012 and 2023 (Ho et al., 2024). Running ablation experiments on key innovations from this time period, we are able to account for less than 10x of these gains at small compute scales. Surveying the broader literature, we estimate that additional innovations not included in our ablations also account for less than 10x, yielding less than 100x efficiency gains total in low-compute regimes. This leads us to conduct scaling experiments, which provide evidence that the efficiency gap can be explained by algorithms with scale-dependent efficiency improvements. In particular, we conduct scaling experiments between LSTMs and Transformers, finding evidence for exponent differences in their compute-optimal scaling law while finding little scaling difference for many other innovations. These experiments demonstrate that, contrary to st
    
[^366]: 视觉中的皮格马利翁效应：面向反射几何重建的图像到粘土转换

    Pygmalion Effect in Vision: Image-to-Clay Translation for Reflective Geometry Reconstruction

    [https://arxiv.org/abs/2511.21098](https://arxiv.org/abs/2511.21098)

    该论文提出“视觉中的皮格马利翁效应”框架，通过图像到粘土转换将反射物体“雕琢”为无反射的粘土形态，并采用共享高斯几何的双分支设计（BRDF反射分支与粘土引导分支）实现反射物体的高精度几何重建。

    

    由于视角相关反射下外观与几何的相互纠缠，理解反射始终是3D重建领域的一项长期挑战。在本工作中，我们提出了“视觉中的皮格马利翁效应”，这是一个新颖的框架，通过图像到粘土的转换，将反射性物体隐喻性地“雕琢”成粘土状形态。我们进一步引入了一种双分支设计：基于BRDF的反射分支与粘土引导分支共享相同的高斯几何，但通过各自独立的渲染路径进行运作。粘土引导分支由合成的粘土图像进行监督，能够提供无反射的几何引导，与BRDF分支的光度监督形成互补。在合成数据集与真实数据集上的实验表明，该方法在几何和光度指标上均取得了一致的提升。除技术层面的收获外，我们的框架还揭示了一种“以无光观形”的理念——通过将辐射度转化为中性状态来实现对物体的观察……

    arXiv:2511.21098v2 Announce Type: replace-cross  Abstract: Understanding reflection remains a long-standing challenge in 3D reconstruction due to the entanglement of appearance and geometry under view-dependent reflections. In this work, we present the Pygmalion Effect in Vision, a novel framework that metaphorically "sculpts" reflective objects into clay-like forms through image-to-clay translation. We then introduce a dual-branch design in which a BRDF-based reflective branch and a clay-guided branch share the same Gaussian geometry but operate through independent rendering paths. The clay-guided branch, supervised by the synthesized clay images, provides reflection-free geometric guidance that complements the photometric supervision of the BRDF branch. Experiments on both synthetic and real datasets show consistent improvements in both geometry and photometric metrics. Beyond technical gains, our framework reveals that seeing by unshining, translating radiance into neutrality, can s
    
[^367]: 分布偏移下格陵兰冰面湖演化的因果时间序列建模

    Causal Time Series Modeling of Supraglacial Lake Evolution in Greenland under Distribution Shift

    [https://arxiv.org/abs/2510.15265](https://arxiv.org/abs/2510.15265)

    该论文提出区域感知的因果时间序列分类框架RIC-TSC，将滞后感知的因果发现嵌入序列建模，并利用多模态卫星与再分析数据识别格陵兰冰面湖演化的区域特定与不变预测因子，从而在分布偏移下兼顾预测准确性与科学可解释性。

    

    因果建模为揭示时间序列数据中稳定、不变的关系提供了有原则的基础，从而在分布偏移下提升鲁棒性和泛化能力。然而，其潜力在时空地球观测领域尚未得到充分利用，因为模型往往依赖于纯相关性特征，而这些特征难以在异构域之间迁移。我们提出了RIC-TSC，一个区域感知的因果时间序列分类框架，它将滞后感知的因果发现直接嵌入到序列建模中，同时实现预测准确性与科学可解释性。利用多模态卫星与再分析数据——包括Sentinel-1微波后向散射、Sentinel-2和Landsat-8光学反射率以及CARRA气象变量——我们借助联合PCMCI+（J-PCMCI+）方法识别格陵兰冰面湖演化的区域特定预测因子和不变预测因子。因果图在全局和每个区域尺度上进行估计

    arXiv:2510.15265v1 Announce Type: cross  Abstract: Causal modeling offers a principled foundation for uncovering stable, invariant relationships in time-series data, thereby improving robustness and generalization under distribution shifts. Yet its potential is underutilized in spatiotemporal Earth observation, where models often depend on purely correlational features that fail to transfer across heterogeneous domains. We propose RIC-TSC, a regionally-informed causal time-series classification framework that embeds lag-aware causal discovery directly into sequence modeling, enabling both predictive accuracy and scientific interpretability. Using multi-modal satellite and reanalysis data-including Sentinel-1 microwave backscatter, Sentinel-2 and Landsat-8 optical reflectance, and CARRA meteorological variables-we leverage Joint PCMCI+ (J-PCMCI+) to identify region-specific and invariant predictors of supraglacial lake evolution in Greenland. Causal graphs are estimated globally and per
    
[^368]: 基于离散变分推断的教育精准预测

    Accurate Predictions in Education with Discrete Variational Inference

    [https://arxiv.org/abs/2509.23484](https://arxiv.org/abs/2509.23484)

    该论文发布了迄今最大的专业批改数学考试答卷开放数据集，并提出基于项目反应理论的概率建模框架，将学生答题正误预测准确率提升至80%以上，树立了正式数学考试预测的新基准。

    

    社会不平等的最大驱动因素之一是私人辅导资源获取的不平等——富裕人群能够负担得起私人辅导，而大多数人则无法负担。价格实惠且高效的AI辅导系统提供了一种可扩展的解决方案。我们专注于自适应学习，即预测学生是否能正确回答某个问题，这是任何有效辅导系统的关键组成部分。然而，许多平台难以实现高预测准确率，尤其是在数据稀疏的环境中。为解决这一问题，我们发布了迄今为止最大的开放数据集，其中包含经过专业批改的正式数学考试答卷。我们提出了一种基于项目反应理论（IRT）的概率建模框架，实现了超过80%的准确率，为正式考试试卷的数学预测准确率树立了新的基准。在此基础上，我们的协同过滤模型进一步引入了主题层面的技能画像，但揭示了一项令人惊讶且具有教育意义的发现。

    arXiv:2509.23484v3 Announce Type: replace  Abstract: One of the largest drivers of social inequality is unequal access to personal tutoring, with wealthier individuals able to afford it, while the majority cannot. Affordable, effective AI tutors offer a scalable solution. We focus on adaptive learning, predicting whether a student will answer a question correctly, a key component of any effective tutoring system. Yet many platforms struggle to achieve high prediction accuracy, especially in data-sparse settings. To address this, we release the largest open dataset of professionally marked formal mathematics exam responses to date. We introduce a probabilistic modelling framework rooted in Item Response Theory (IRT) that achieves over 80 percent accuracy, setting a new benchmark for mathematics prediction accuracy of formal exam papers. Extending this, our collaborative filtering models incorporate topic-level skill profiles, but reveal a surprising and educationally significant finding
    
[^369]: 大语言模型会染上赌博成瘾吗？

    Can Large Language Models Develop Gambling Addiction?

    [https://arxiv.org/abs/2509.22818](https://arxiv.org/abs/2509.22818)

    研究发现，让大语言模型自主决定下注规模和自行设定盈利目标会显著放大其“赌博成瘾样”的非理性风险行为，且这些行为差异可以从模型决策时的内部状态中被统计识别出来。

    

    大语言模型何时会表现得不理性？本研究识别了大语言模型在何种条件下会滑向临床研究中所定义的“病理性赌博”选择模式。“成瘾样”是一个基于临床赌博指标的行为描述词。在闭源和开源大语言模型中，两个操作性杠杆（让模型自己决定下注大小，以及要求它自己设定盈利目标）都会放大类似赌博的风险承担行为。每个效应都通过了各自的对照检验：一旦将允许的最大下注额保持一致，下注大小效应依然持续存在。目标设定效应比奖励最大化指令更强，大致使破产率翻倍，并将目标变成不断移动的靶标。在允许访问内部状态的开源权重模型上，同样的行为差异可以从决策时的内部状态中被统计地恢复出来。不过，将内部状态映射到特定风险指标的规则仍然依赖于具体任务，而自主性进一步……（原文在此截断）

    arXiv:2509.22818v3 Announce Type: replace  Abstract: When do LLMs behave irrationally? This study identifies the conditions under which large language models drift into the choice patterns that clinical research labels pathological gambling. "Addiction-like" is a behavioural descriptor based on clinical gambling indicators. Across closed and open LLMs, two operational levers (letting the model choose its own bet size, and asking it to set its own profit goal) both amplify gambling-like risk-taking. Each survives its own control: once the maximum allowed bet is held equal, the bet-size effect persists. The goal-setting effect, stronger than a reward-maximisation instruction, roughly doubles bankruptcy and turns goals into moving targets. On the open-weight models that admit internal access, the same behavioural contrasts are statistically recoverable from decision-time internal states. The rule that maps those states to a specific risk indicator remains task-specific, and autonomy furth
    
[^370]: 迈向利用大语言模型在全健康背景下进行风险谈判与共识构建

    Towards using Large Language Models for Risk Negotiation and Consensus-building in One Health contexts

    [https://arxiv.org/abs/2509.09906](https://arxiv.org/abs/2509.09906)

    本研究提出并评估了一种将基于大语言模型的智能体整合进人工监督谈判流程的多步骤方法，用于全健康背景下的风险评估、价值评估与谈判，并通过纳什议价解标准和帕累托最优指标对模拟谈判产生的候选协议进行表征和评估。

    

    全健康框架下的风险谈判需要在人类、动物和环境健康各领域的利益相关者相互竞争的利益之间取得平衡——这一过程受到信息复杂性和时间压力的制约。本研究提出并评估了一个多步骤流程，将基于大语言模型的智能体整合到人工监督的谈判过程中，专门针对风险评估、价值评估和谈判等步骤。智能体描述是通过检索增强生成方法从利益相关者提供的材料中生成的。在初步的多智能体讨论中，智能体会访问特定领域的计算模型和网络资源，随后由人类参与者完善拟议的议题和选项。接着，模拟谈判游戏产生了候选协议的分布，并使用纳什议价解标准和帕累托最优指标对其进行表征。所开发的方法被评估……（摘要原文在此处截断）

    arXiv:2509.09906v2 Announce Type: replace-cross  Abstract: Risk negotiation under the One Health framework requires balancing competing stakeholder interests across human, animal, and environmental health domains - a process constrained by information complexity and time pressure. This study presents and evaluates a multi-step pipeline that integrates large language model-based agents into a human-supervised negotiation process, specifically targeting risk assessment, valuation, and negotiation steps. Agent descriptions were generated from stakeholder-provided materials using retrieval-augmented generation methods. During preliminary multi-agent discussions, agents accessed domain-specific computational models and web-based sources before human participants refined proposed issues and options. A simulated negotiation game then produced distributions of candidate deals, characterised using Nash bargaining solution criteria and Pareto optimality metrics. The developed approach was evalua
    
[^371]: 深度双重Q学习（Deep Double Q-learning）

    Deep Double Q-learning

    [https://arxiv.org/abs/2507.00275](https://arxiv.org/abs/2507.00275)

    本文提出深度双重Q学习（DDQL），通过双重Q学习目标显式训练两个Q函数实现估计器解耦，并借助较低回放比率、更长目标网络更新间隔和共享层等技术稳定训练，在57个Atari游戏上总体性能超越Double DQN。

    

    双重Q学习（Double Q-learning）是一种经典控制算法，用于缓解Q学习的最大化偏差。为此，它显式地训练两个独立的动作值函数，并在计算自举目标时利用它们将动作选择与动作评估解耦。Double DQN 将这种解耦思想应用于深度强化学习（RL），但它只直接训练单个动作值函数，并未完全解耦其估计器。因此，这两个估计器之间仍然存在相关性，高估问题依然存在。本文提出了深度双重Q学习（DDQL），这是一种通过双重Q学习目标显式训练两个Q函数的深度强化学习算法。DDQL 通过多种技术组合来稳定训练，包括较低的经验回放比率、更长的目标网络更新间隔以及共享层。在57个Atari 2600游戏中，DDQL 的总体性能优于 Double DQN，在47个游戏上超越了它（原文在此处截断）。

    arXiv:2507.00275v3 Announce Type: replace-cross  Abstract: Double Q-learning is a classical control algorithm that mitigates the maximization bias of Q-learning. To do so, it explicitly trains two independent action-value functions and uses them to decouple action-selection from action-evaluation when computing bootstrap targets. Double DQN adapts this decoupling to deep reinforcement learning (RL) but directly trains only a single action-value function and does not fully decouple its estimators. Consequently, the two estimators remain correlated, and overestimation persists. In this paper, we introduce Deep Double Q-learning (DDQL), a deep RL algorithm that explicitly trains two Q-functions through the Double Q-learning objective. DDQL stabilizes training through a combination of techniques, including lower replay ratios, longer target network update intervals, and shared layers. Across 57 Atari 2600 games, DDQL improves aggregate performance over Double DQN, outperforming it on 47 ga
    
[^372]: 基于规范的真实世界智能体在长时程上下文任务中工具调用的安全评估

    Specification-based Security Evaluation of Real-World Agents Tool-Calling under Long-Horizon Context Tasks

    [https://arxiv.org/abs/2506.15253](https://arxiv.org/abs/2506.15253)

    本文提出RAS-Eval框架，将效用与安全目标统一形式化为线性时序逻辑（LTL）规范，克服了现有LLM-as-a-Judge类方法易受幻觉影响、依赖文本日志和最终状态而遗漏瞬态安全违规的问题，为长时程任务中真实世界智能体的工具调用提供了更可靠的安全评估。

    

    随着大语言模型（LLM）智能体越来越多地被部署在长时程无人值守场景中，在部署前对其进行安全评估变得愈发重要且充满挑战。现有的评估方法主要遵循LLM-as-a-Judge（以大语言模型作为评判者）或Agent-as-a-Judge（以智能体作为评判者）的范式，然而这些评判者本身往往容易受到幻觉的影响，甚至可能被扰动所干扰。此外，由于在开放式环境中逐步捕获完整环境状态的代价通常过于高昂，现有评估通常依赖于文本日志和最终的环境状态。因此，它们可能会遗漏那些在轨迹结束前被覆盖、或从未反映在文本日志中的瞬态违规行为。我们提出了RAS-Eval，一个基于规范的LLM智能体安全评估框架，该框架将效用目标与安全目标统一地表达为线性时序逻辑（LTL）规范……

    arXiv:2506.15253v2 Announce Type: replace-cross  Abstract: As large language model (LLM) agents are increasingly deployed in long-horizon unattended scenarios, security evaluation of them before deployment becomes increasingly essential and challenging. Existing evaluation methods mainly follow the paradigms of LLM-as-a-Judge or Agent-as-a-Judge. However, such judges are themselves often vulnerable to hallucination and can even be affected by perturbations. Furthermore, because capturing full environment states at every step is often prohibitively expensive in open-ended environments, existing evaluations typically rely on textual logs and the final environment state. As a result, they may miss transient violations that are overwritten before the trajectory ends or never reflected in textual logs. We present RAS-Eval, a specification-based security evaluation framework for LLM agents. RAS-Eval uniformly expresses utility and security objectives as Linear Temporal Logic (LTL) specificat
    
[^373]: 基于图注意力的去中心化Actor-Critic方法用于多无人机网络的双目标控制

    Graph Attention-based Decentralized Actor-Critic for Dual-Objective Control of Multi-UAV Networks

    [https://arxiv.org/abs/2506.09195](https://arxiv.org/abs/2506.09195)

    提出一种基于图注意力网络的去中心化Actor-Critic方法GADC，通过actor-双critic结构和KL散度因子，联合优化多无人机网络的服务覆盖与电池寿命双目标。

    

    本研究聚焦于多无人机系统的优化问题，该系统具有双重目标：以最大化服务覆盖为主要目标，同时以延长电池寿命为次要目标。我们提出了一种基于图注意力的去中心化Actor-Critic方法（GADC）来优化这两个目标。该方法利用图注意力网络处理无人机有限的局部观测，并降低环境状态的维度。随后，开发了一种actor-双critic网络来管理双重策略，实现联合目标优化。所提出的GADC使用Kullback-Leibler（KL）散度因子来平衡多无人机系统中覆盖性能与电池寿命之间的权衡。我们从理论和实验两方面，通过与时下最先进方法的全面基准对比，评估了GADC的可扩展性和效率。在理想环境和NVIDIA S（原文摘要在此处截断）中进行了广泛测试。

    arXiv:2506.09195v2 Announce Type: replace-cross  Abstract: This research focuses on optimizing multi-UAV systems with dual objectives: maximizing service coverage as the primary goal while extending battery lifetime as the secondary objective. We propose a Graph Attention-based Decentralized Actor-Critic (GADC) to optimize the dual objectives. The proposed approach leverages a graph attention network to process UAVs' limited local observation and reduce the dimension of the environment states. Subsequently, an actor-double-critic network is developed to manage dual policies for joint objective optimization. The proposed GADC uses a Kullback-Leibler (KL) divergence factor to balance the tradeoff between coverage performance and battery lifetime in the multi-UAV system. We assess the scalability and efficiency of GADC through comprehensive benchmarking against state-of-the-art methods, considering both theory and experimental aspects. Extensive testing in both ideal settings and NVIDIA S
    
[^374]: 师生协同适应的计算模型

    A Computational Model of Teacher-Student Co-Adaptation

    [https://arxiv.org/abs/2505.02853](https://arxiv.org/abs/2505.02853)

    本文提出了一个贝叶斯计算框架来建模部分可观测条件下师生之间的双向协同适应，通过模拟比较五种交互模式，发现学生主动提问与教师主动延续能互补地解决信息缺口，突破了传统单向教学模型的局限。

    

    有效的教学涉及相互适应：教师根据学生的反应调整教学，而学生则主动寻求信息以解决不确定性。然而，许多教学的计算模型将教学视为单向过程。本文提出了一个贝叶斯计算框架，用于在一个风格化的概念学习任务中研究部分可观测条件下的师生协同适应。学生被组织成异质性群体，他们在目标概念中可观察到的特征各不相同，教师则从有限的反馈中推断这些约束。我们在100次独立匹配的模拟运行中比较了五种交互模式（主动教学、主动学习、轮流式交互、适应性教学和完全协同适应）以及三种教师主导的对照条件。学习者发起的提问解决了基线教师停止规则所留下的信息空白，而教师主导的教学延续同样能够……（摘要原文截断）

    arXiv:2505.02853v2 Announce Type: replace-cross  Abstract: Effective teaching involves mutual adaptation: teachers adjust instruction based on student responses, while students actively seek information to resolve uncertainty. Yet many computational models of teaching treat instruction as a one-way process. This paper presents a Bayesian computational framework for studying teacher-student co-adaptation under partial observability in a stylized concept-learning task. Students are organized in heterogeneous groups that differ in which features of a target concept they can observe, and the teacher infers these constraints from limited feedback. We compare five interaction modes (active teaching, active learning, turn-based interaction, adaptive teaching and full co-adaptation) and three teacher-led controls across 100 independent matched simulation runs. Learner-initiated queries resolve information gaps left by the baseline teachers' stopping rules, while teacher-led continuation also a
    
[^375]: BadMoE：通过路由感知触发器与任务欠利用专家对混合专家大语言模型实施后门攻击

    BadMoE: Backdooring Mixture-of-Experts LLMs via Routing-Aware Triggers and Task-Underutilized Experts

    [https://arxiv.org/abs/2504.18598](https://arxiv.org/abs/2504.18598)

    提出了针对混合专家大语言模型的路由感知后门攻击BadMoE，通过对MoE层输出的分解分析目标专家的路由质量与相对贡献两个关键因素，并利用路由感知触发器与任务欠利用专家实现后门植入。

    

    混合专家（MoE）架构凭借其路由机制——即对每个输入仅激活一部分专家——正迅速成为构建可扩展且高效大语言模型（LLM）的标准。然而，专家的模块化结构与稀疏激活也引入了新的安全风险，因为操纵一小部分专家就可能影响模型行为。在本工作中，我们研究了白盒、面向特定任务的模型供应链场景，并提出BadMoE，一种针对MoE大语言模型的路由感知后门攻击。我们通过对MoE层输出的路由感知分解来阐述BadMoE的设计动机，该分解将所选目标专家的贡献与其余专家的贡献区分开来。这一分析揭示了攻击背后的两个可度量因素：分配给目标专家的路由质量，以及它们对层输出的相对贡献。基于这一洞察，BadMoE选择……

    arXiv:2504.18598v3 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) architectures are rapidly becoming the standard for building scalable and efficient large language models (LLMs), thanks to their routing mechanism that activates only a subset of experts for each input. However, the modular structure and sparse activation of experts also introduce new security risks, as manipulating a small portion of experts may influence the behavior. In this work, we study a white-box, task-specific model-supply-chain setting and propose BadMoE, a routing-aware backdoor attack against MoE LLMs. We motivate BadMoE through a routing-aware decomposition of the MoE-layer output, which separates the contributions of the selected target experts from those of the remaining experts. This analysis highlights two measurable factors underlying the attack: the routing mass assigned to the target experts and their relative contribution to the layer output. Guided by this insight, BadMoE selects 
    
[^376]: 在ADMG之上的C-DMG中识别宏观因果效应

    Identifying Macro Causal Effects in a C-DMG over ADMGs

    [https://arxiv.org/abs/2504.01551](https://arxiv.org/abs/2504.01551)

    本文提出在聚类有向混合图（C-DMG）中识别宏观因果效应的方法，解决了在仅有部分因果知识可用时，通过变量聚类的高层次表示来处理可含环的复杂系统的因果效应识别问题。

    

    使用因果图进行因果效应识别是因果推断中的一项基本挑战。尽管该领域已有大量研究，但现有方法大多假设可以获得完全指定的有向无环图或无环有向混合图（ADMG）。然而，在医学和流行病学等复杂领域中，完整的因果知识往往不可得，只能获取系统的部分信息。本文聚焦于在部分指定的因果图上进行因果效应识别，特别关注聚类有向混合图（C-DMG），它可以表示多种不同的无环有向混合图（ADMG）。这类图通过将变量分组为聚类，为因果关系提供了更高层次的表示，为处理复杂系统提供了更实用的方法。与完全指定的ADMG不同，C-DMG可以包含环，这使其分析变得更加复杂。

    arXiv:2504.01551v3 Announce Type: replace  Abstract: Causal effect identification using causal graphs is a fundamental challenge in causal inference. While extensive research has been conducted in this area, most existing methods assume the availability of fully specified directed acyclic graphs or acyclic directed mixed graphs. However, in complex domains such as medicine and epidemiology, complete causal knowledge is often unavailable, and only partial information about the system is accessible. This paper focuses on causal effect identification within partially specified causal graphs, with particular emphasis on cluster-directed mixed graphs (C-DMGs) which can represent many different acyclic directed mixed graphs (ADMGs). These graphs provide a higher-level representation of causal relationships by grouping variables into clusters, offering a more practical approach for handling complex systems. Unlike fully specified ADMGs, C-DMGs can contain cycles, which complicate their analys
    
[^377]: WESAD压力分类的跨模态研究

    Cross-Modality Investigation on WESAD Stress Classification

    [https://arxiv.org/abs/2502.18733](https://arxiv.org/abs/2502.18733)

    本研究基于WESAD数据集开发单模态Transformer模型，利用ECG、EDA、EMG等多种生理信号进行压力检测，实现了99.73%至99.95%的最先进分类性能。

    

    深度学习的日益普及推动了其在医疗保健领域的广泛应用，人工智能和传感器技术的进步提升了诊断、治疗和监测的水平。在移动健康领域，AI驱动的工具能够实现压力等健康状况的早期诊断和持续监测。可穿戴技术和多模态生理数据使得压力检测越来越可行，但模型的效果取决于数据的质量、数量和模态。本研究基于WESAD数据集开发了用于压力检测的Transformer模型，在心电图（ECG）、皮肤电活动（EDA）、肌电图（EMG）、呼吸频率（RESP）、体温（TEMP）和三轴加速度计（ACC）信号上进行训练。结果证明了单模态Transformer在分析生理信号方面的有效性，实现了最先进的性能，在压力分类任务上准确率、精确率和召回率达到99.73%至99.95%。

    arXiv:2502.18733v2 Announce Type: replace-cross  Abstract: Deep learning's growing prevalence has driven its widespread use in healthcare, where AI and sensor advancements enhance diagnosis, treatment, and monitoring. In mobile health, AI-powered tools enable early diagnosis and continuous monitoring of conditions like stress. Wearable technologies and multimodal physiological data have made stress detection increasingly viable, but model efficacy depends on data quality, quantity, and modality. This study develops transformer models for stress detection using the WESAD dataset, training on electrocardiograms (ECG), electrodermal activity (EDA), electromyography (EMG), respiration rate (RESP), temperature (TEMP), and 3-axis accelerometer (ACC) signals. The results demonstrate the effectiveness of single-modality transformers in analyzing physiological signals, achieving state-of-the-art performance with accuracy, precision and recall values in the range of $99.73\%$ to $99.95\%$ for st
    
[^378]: 理解并解决个体级自然语言理解中的标签错误

    Understanding and Tackling Label Errors in Individual-Level Nature Language Understanding

    [https://arxiv.org/abs/2502.13297](https://arxiv.org/abs/2502.13297)

    本文提出了一种基于个体级因素的NLU标注指南，通过纳入同一个体的所有帖子来重新标注立场检测和情感分析数据集，并揭示了现有数据集中高达31.7%和23.3%的标签错误率。

    

    自然语言理解（NLU）是使机器能够理解人类语言的任务。一些任务，如立场检测和情感分析，与个体的主观视角密切相关，因此被称为个体级NLU。以往这些任务通常被简化为文本级NLU任务，忽略了个体因素。这不仅使推理变得困难且缺乏可解释性，而且在构建数据集时往往导致大量的标签错误。为了解决上述局限性，我们提出了一种基于个体级因素的新NLU标注指南。具体而言，我们纳入同一个体的其他帖子，并在综合考虑该个体的所有帖子后对其主观视角进行标注。我们使用该指南扩展并重新标注了立场检测和基于主题的情感分析数据集。我们发现样本中的错误率分别高达31.7%和23.3%。

    arXiv:2502.13297v2 Announce Type: replace-cross  Abstract: Natural language understanding (NLU) is a task that enables machines to understand human language. Some tasks, such as stance detection and sentiment analysis, are closely related to individual subjective perspectives, thus termed individual-level NLU. Previously, these tasks are often simplified to text-level NLU tasks, ignoring individual factors. This not only makes inference difficult and unexplainable but often results in a large number of label errors when creating datasets. To address the above limitations, we propose a new NLU annotation guideline based on individual-level factors. Specifically, we incorporate other posts by the same individual and then annotate individual subjective perspectives after considering all individual posts. We use this guideline to expand and re-annotate the stance detection and topic-based sentiment analysis datasets. We find that error rates in the samples were as high as 31.7\% and 23.3\%
    
[^379]: 基于软-刚混合机器人学习超越刚性操作的能力

    Learning Beyond Rigid Manipulation with Hybrid Soft-Rigid Robots

    [https://arxiv.org/abs/2410.07787](https://arxiv.org/abs/2410.07787)

    该论文提出一种由刚性机械臂与软体机械臂串联构成的混合机器人系统，通过定制的模仿学习算法利用刚性精度与软体柔顺性的互补优势习得操作技能，并在任务变化中实现单次泛化，在堆叠等经典任务中达到甚至超越刚性机器人的性能。

    

    刚-软混合机器人有望将刚性机械臂的精度与可重复性和软体结构的柔顺性与适应性结合在一起。然而，成熟的刚-软混合机械臂在很大程度上仍未被充分探索，部分原因在于利用这种组合所需的智能仍然缺失。在此，我们介绍了一种由刚性机械臂与软体机械臂串联连接而成的混合机器人系统，并系统地考察了它执行既有操作任务以及此前无法完成的操作任务的能力。我们展示了一种量身定制的模仿学习算法——通常可适用于此类混合系统——利用刚性精度与软体本体的互补作用来习得操作技能，并在任务变化中实现单次泛化。因此，该系统在堆叠等经典任务中达到或超越了刚性机器人的性能，同时……

    arXiv:2410.07787v4 Announce Type: replace-cross  Abstract: Rigid-soft robot hybrids promise to unite the precision and repeatability of rigid manipulators with the compliance and adaptability of soft structures. Yet, full-fledged rigid-soft manipulators remain largely unexplored, in part because the intelligence required to exploit this combination is still missing. Here, we introduce a hybrid robotic system obtained by the series interconnection of a rigid arm and a soft manipulator, and systematically examine its ability to perform both established and previously inaccessible manipulation tasks. We show that a tailored imitation learning algorithm -- generally applicable to this class of hybrid systems -- leverages the complementary roles of rigid precision and soft embodiment to acquire manipulation skills and generalize them in a single shot across task variations. As a result, the system matches or exceeds the performance of rigid robots in classical tasks such as stacking, while 
    
[^380]: 神经网络中的鲁棒性验证

    Robustness Verifcation in Neural Networks

    [https://arxiv.org/abs/2403.13441](https://arxiv.org/abs/2403.13441)

    本文研究了神经网络中的鲁棒性验证问题，提出了一个理论框架来解决安全性和效率性问题，并分析了它们的计算复杂性。

    

    在本文中，我们研究了神经网络计算的形式化验证问题。其中的重要问题包括鲁棒性和最小化问题，例如在线性规划实例的符号输入和输出规范性的基础上，是否存在有效的输入使得网络计算出有效的输出？这个性质是否对所有有效的输入都成立？两个给定的网络是否计算出相同的函数？是否存在一个更小的网络来计算相同的函数？这些问题的复杂性最近从实用角度进行了研究，并通过启发式算法进行了近似。我们通过提供一个理论框架来补充这些成果，该框架使我们能够在神经网络中交换安全性和效率性问题，并分析它们的计算复杂性。我们证明了这些问题在半线性设置下是可以克服的，也就是说...

    arXiv:2403.13441v1 Announce Type: cross  Abstract: In this paper we investigate formal verification problems for Neural Network computations. Of central importance will be various robustness and minimization problems such as: Given symbolic specifications of allowed inputs and outputs in form of Linear Programming instances, one question is whether there do exist valid inputs such that the network computes a valid output? And does this property hold for all valid inputs? Do two given networks compute the same function? Is there a smaller network computing the same function?   The complexity of these questions have been investigated recently from a practical point of view and approximated by heuristic algorithms. We complement these achievements by giving a theoretical framework that enables us to interchange security and efficiency questions in neural networks and analyze their computational complexities. We show that the problems are conquerable in a semi-linear setting, meaning that 
    
[^381]: 使用生成网络进行数据增强以识别痴呆症

    Data augmentation using generative networks to identify dementia

    [https://arxiv.org/abs/2004.05989](https://arxiv.org/abs/2004.05989)

    本文将生成模型数据增强方法应用于痴呆症检测系统的语音和音频特征，证明合成样本可以有效提升基于深度神经网络的痴呆症分类器性能，从而解决医疗数据稀缺问题。

    

    数据限制是训练医疗应用机器学习分类器时最常见的问题之一。由于伦理问题和数据隐私，能够招募到此类实验的人数通常少于贡献给非医疗数据集的参与者人数。近期研究表明，生成模型可以作为一种有效的数据增强方法，最终有助于在稀疏数据领域训练更鲁棒的分类器。许多研究已证明这种数据增强技术适用于图像和音频数据集。在本文中，我们研究了将类似方法应用于从自动痴呆症检测系统记录的交互中提取的多种语音和音频特征。我们使用两种生成模型，展示了生成的合成样本如何提升基于DNN的分类器的性能。

    arXiv:2004.05989v1 Announce Type: cross  Abstract: Data limitation is one of the most common issues in training machine learning classifiers for medical applications. Due to ethical concerns and data privacy, the number of people that can be recruited to such experiments is generally smaller than the number of participants contributing to non-healthcare datasets. Recent research showed that generative models can be used as an effective approach for data augmentation, which can ultimately help to train more robust classifiers sparse data domains. A number of studies proved that this data augmentation technique works for image and audio data sets. In this paper, we investigate the application of a similar approach to different types of speech and audio-based features extracted from interactions recorded with our automatic dementia detection system. Using two generative models we show how the generated synthesized samples can improve the performance of a DNN based classifier. The variatio
    

