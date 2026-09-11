# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Show-Harness: Just a VLM Agent Can Play Robots](https://arxiv.org/abs/2609.10522) | Show-Harness通过紧凑的语义动作接口让VLM智能体直接操控机器人，既可实现闭源前沿大模型的零样本机器人控制，也能以极低的微调成本适配小型开源模型。 |
| [^2] | [IBIB: A Protocol for Measuring Enterprise AI Systems by Serving Route, Not Model Identifier](https://arxiv.org/abs/2609.10494) | 该论文提出IB2协议，将现有基准测试仅按模型标识符评分视为测量误差，通过金标准盲能力绑定预检、包含可靠性的首轮评分规则和分数盲裁决三部分，实现以实际服务路由（而非模型标识符）来衡量企业AI系统的真实可用能力。 |
| [^3] | [Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization](https://arxiv.org/abs/2609.10464) | 提出SG-JEPA模型，通过动作条件化注入物理参数并以自回归潜在滚动方式联合训练编码器和预测器，实现了在不同引力场下的零样本物理泛化，开环预测误差相比DINO-WM降低最多2倍。 |
| [^4] | [JarvisGUI: Towards Cross-Device GUI Agents with Dynamic Task Composition](https://arxiv.org/abs/2609.10451) | 提出JarvisGUI——一个通过轻量级类型系统将GUI任务表述为输入-输出转换、可自动组合并动态评估跨Android、Windows和Ubuntu等多设备工作流的GUI智能体基准测试。 |
| [^5] | [ConvMem: Convolutional Memory for Long-Context Reasoning](https://arxiv.org/abs/2609.10441) | ConvMem 提出了一种无需训练、高度可并行化的分层卷积框架，将被特定查询提示的 LLM 视为卷积核对文本进行层次化摘要，将长上下文推理路径从线性链缩短为对数树，克服了序列化记忆方法高延迟和依赖昂贵强化学习训练的缺陷。 |
| [^6] | [Forgetting Only What Matters: Layer-Selective Unlearning toward Robust LLMs](https://arxiv.org/abs/2609.10439) | 提出了FOM-UL层级遗忘框架，通过遗忘-保留显著性分数选择对遗忘集影响大且对保留集不敏感的transformer层进行针对性更新，实现了更好的遗忘-效用权衡，并在训练后量化等部署变化下保持鲁棒。 |
| [^7] | [Emergency Department Revisit Quality Review Screening: Exploring Human Decision-Making and Artificial Intelligence Support](https://arxiv.org/abs/2609.10421) | 该研究探索了急诊科复诊质量审查中人类评估者的决策过程，并提出了一种利用GPT-4大语言模型填充知识图谱的自动筛选算法（KGA），以突破传统48-72小时复诊时间窗口的限制，更全面地识别潜在值得关注的诊断对。 |
| [^8] | [Fortunate Recall: Ontology-Driven Memory Lifecycle Management for Persistent Coherence in LLMs](https://arxiv.org/abs/2609.10413) | 提出Fortunate Recall（FR）可组合策略层，通过“10+1”行为本体对个人记忆进行分类，并以确定性函数形式应用类别特定的生命周期策略（差异化时间衰减、槽键取代、事件时间有效性和类别感知检索路由），在LifecycleBench和LongMemEval-S上显著超越Mem0等现有记忆系统。 |
| [^9] | [Can Foundation Models Moderate Online Content? Evaluating Instruction- vs. Example-Driven Policy Operationalization](https://arxiv.org/abs/2609.10410) | 本文提出包含4,000条Bluesky人工标注帖子的新基准ModerationBench，系统比较了指令驱动与示例驱动两种政策操作化范式，发现基础模型的内容审核F1分数可达Bluesky现有审核系统的近三倍（0.60 vs. 0.22）。 |
| [^10] | [MOONWALK: Mediating Operations with Intent-Evidence-Action Alignment Across Junior-Supervisor Review Workflows in Animation/VFX Pre-Production](https://arxiv.org/abs/2609.10385) | 该论文提出了意图-证据-行动对齐的设计框架，并实现为专业评审系统MOONWALK，帮助动画/VFX前期制作团队将松散的创作意图转化为初级艺术家可执行的明确修改任务。 |
| [^11] | [PACE: Perceived-Latency-Aware Cascading Service Routing and Filler Control for QoE-Efficient Retrieval-Augmented Dialogue Serving](https://arxiv.org/abs/2609.10372) | PACE框架将感知首次响应时间（PTFR）作为QoE目标，通过负载自适应级联路由、路径-填充器联合控制和波动感知缓存准入三种机制，在人形机器人对话服务中显著降低感知延迟并保证回答质量与新鲜度。 |
| [^12] | [OmniMed-FL: A Robust Multimodal Federated Learning Framework for Clinical Diagnosis](https://arxiv.org/abs/2609.10364) | 提出了OmniMed-FL多模态联邦学习框架，通过系统性评估多种融合策略、初始化方法和缺失文本填补规则，在满足隐私法规的前提下实现了医学影像与文本病历的安全融合，用于五类临床病症的分类诊断。 |
| [^13] | [Cyber-Financial Contagion: Modeling the Propagation of an AI Vendor Compromise Through the Banking System](https://arxiv.org/abs/2609.10350) | 本文提出CFC-Prop随机传染病-清算模型，通过四层异构网络模拟单个AI供应商遭入侵如何沿银行系统传播，最终引发类似传统银行危机的系统性厚尾损失。 |
| [^14] | [Beyond One-Size-Fits-All: Sample-Adaptive Strategy Routing for Vision Token Pruning in MLLMs](https://arxiv.org/abs/2609.10346) | 提出VIP-Router，一个轻量级路由器，根据每个输入样本的低成本视觉和文本特征自适应选择最优的视觉token剪枝策略，突破了传统固定剪枝策略“一刀切”的局限。 |
| [^15] | [From Symbolic Perception to Logical Deduction: A Framework for Guiding Language Models in Geometric Reasoning](https://arxiv.org/abs/2609.10335) | 该论文提出一个将几何图形解析为符号形式并进行形式化逻辑推演的框架，使纯大语言模型在几何推理上达到与最先进多模态模型相当的性能，同时减少幻觉并提升推理的可解释性。 |
| [^16] | [TRACE: Training Reasoning Agents for Causal Exploration with Synthesized Rewards](https://arxiv.org/abs/2609.10315) | 提出TRACE框架，通过在受控模拟器中注入合成的隐藏干预来提供真值标签和客观奖励，从而训练智能体在噪声、混杂和分散证据下完成复杂数据的因果诊断推理。 |
| [^17] | [One Loop, Two Gains: Can Active Learning win the Lottery for Free?](https://arxiv.org/abs/2609.10311) | 本文提出 Improve & Prune（I&P）方法，将幅度剪枝无缝嵌入基于池的主动学习已有的迭代重训练循环中，在不增加额外计算开销的情况下同时获得主动学习和稀疏子网络（中奖彩票）发现的双重收益。 |
| [^18] | [RiLM: Parameter-Efficient Language Modeling via Geodesic Decoding](https://arxiv.org/abs/2609.10305) | 提出RiLM框架，完全移除传统输出矩阵，通过在黎曼流形上计算当前状态与词表嵌入之间的测地线距离平方来直接解码下一个词元概率，其双曲版本HypRiLM在约29万参数下于WikiText-2上达到54.2的验证困惑度，显著优于平坦版本及同规模的LSTM、Transformer和SSM基线。 |
| [^19] | [Learning Intrusion Response Strategies for OT Systems](https://arxiv.org/abs/2609.10298) | 本文使用POMDP框架对OT系统入侵响应场景进行形式化建模，并结合基于流量测量的部分可观测性，提出基于PPO的学习方法自动生成入侵响应策略，在仿真OT系统中验证了对多种MITRE攻击的有效性。 |
| [^20] | [GANDR: Claim Auditing for Verifiable Legal Answer Generation](https://arxiv.org/abs/2609.10293) | GANDR是一个双智能体系统，由起草器生成结构化法律答案、批评者逐条对照引用来源审计论断，并配合要求每条引用必须命中检索结果的严格正确性标准，从而实现可逐条验证的法律答案生成。 |
| [^21] | [What Should an Agent Forget? Separating What Is Stored from What Is Used](https://arxiv.org/abs/2609.10263) | 本文提出无需训练的RD-Forget框架，通过将智能体存储的内容与使用的内容分离，结合查询条件记忆视图、同槽替换链接和率失真优化，使过时信息不会误导当前回答，同时保留历史查询所需的证据。 |
| [^22] | [DiSCo: A Distribution-First Steering and Cultural Prior Evaluation Framework for Measuring Cultural Preference Bias in LLMs](https://arxiv.org/abs/2609.10253) | DiSCo是一个分布优先的评估框架，通过隔离默认文化先验并利用四级上下文梯度测试可引导性，发现大语言模型的文化偏好先验严重集中在英美文化上（合计约占35%）。 |
| [^23] | [A-JIT: Agentic Just-In-Time Software Construction](https://arxiv.org/abs/2609.10248) | A-JIT提出了一种新的软件构建范式，通过在应用中嵌入AI智能体，使其摆脱静态二进制的限制，能够根据用户行为动态构建缺失实现、即时生成新功能并持续自适应演化。 |
| [^24] | [LiteRAG: Cost-Efficient Graph-Based Retrieval-Augmented Generation](https://arxiv.org/abs/2609.10239) | LiteRAG通过查询条件算法探索和推理链上下文构建取代昂贵的检索时LLM控制，在多跳问答质量上达到最优，同时将查询延迟降低100倍以上、成本降低99%以上、token使用量减少约14倍。 |
| [^25] | [Hierarchical and Permutation-Invariant Feature Transformation Learning via Policy-Guided Embedding Search](https://arxiv.org/abs/2609.10225) | 该论文提出了一个特征变换学习框架，利用置换不变的层次化模块捕捉特征、操作与抽象层次间的交互，并通过策略引导的嵌入搜索适应非凸变换空间，克服了现有生成式方法的三大局限。 |
| [^26] | [Why Sample What You Can Enumerate? Exact Policy Optimization for Genomic Tool Selection](https://arxiv.org/abs/2609.10221) | 该论文揭示了在工具子集空间可完全枚举的基因组推理等科学领域中，GRPO等基于采样的强化学习方法存在结构性缺陷（训练越成功奖励信号越稀缺），并提出FGPO（全组策略优化），通过对所有工具子集进行精确枚举评分来替代采样估计。 |
| [^27] | [Can AI Agents Deliver Verifiable Network-Wide Outcomes Across Authority Boundaries?](https://arxiv.org/abs/2609.10181) | 该论文探讨了当多个具有不同权限范围的AI智能体跨越管理域协作进行网络自动化时，需要一个可信保障层来汇总碎片化的证据，从而验证配置变更确实达成了全网范围的预期结果。 |
| [^28] | [Beyond Surface Imitation: Contrastive Modeling for Reasoning Path Alignment in Multimodal In-Context Learning](https://arxiv.org/abs/2609.10177) | 该论文提出了一种将对比示例建模与多模态大语言模型自我完善能力相结合的新型多模态上下文学习框架，通过在同一输入下显式对比次优响应与更优响应并附以推理路径，使模型超越表面模仿，真正实现响应与所需推理路径的对齐。 |
| [^29] | [Kernel-Managed Shared Memory for System-Wide Personalization](https://arxiv.org/abs/2609.10144) | 提出内核管理的共享内存这一系统级抽象，由智能体系统内核统一负责多智能体间的记忆检索、隐私管控和提示注入，相比非管理的记忆后端显著提升了系统级个性化效果。 |
| [^30] | [Active Adaptation, Not Static Defense: Temporal Dynamics of Preventative Steering in Adversarial Fine-Tuning](https://arxiv.org/abs/2609.10142) | 该研究揭示预防性引导的持久防护源于早期的主动补偿性适应而非静态权重偏移，并据此提出渐进式干预方法来强化对抗性微调防御。 |
| [^31] | [Agent-Based ML-LLM Fusion with Self-Optimizing Prompts for Plateau Weather Alerts](https://arxiv.org/abs/2609.10135) | 该论文提出SmartWeatherAgent智能体系统，通过融合LightGBM机器学习模型与大语言模型的三阶段架构，并结合12轮微步提示自优化循环，实现了高原旅游气象预警质量分数提升112%的突破。 |
| [^32] | [Context operations to architecture modelling output from large language models and evaluation criteria for their use in systems engineering design](https://arxiv.org/abs/2609.10132) | 该论文提出了一个用于大语言模型工程设计中上下文组装的形式化操作框架，并建立了评估LLM架构建模输出是否符合设计意图的评估方法。 |
| [^33] | [SA-Profile: Automated Sulcus Angle Profiling from Super-Resolution MRI](https://arxiv.org/abs/2609.10125) | 该论文提出了一个自动化框架，利用隐式神经表示融合多平面MRI重建超分辨率体积，并通过U-Net标志点检测实现股骨滑车沟角的连续剖面分析，克服了传统单切片测量对切片选择敏感的局限性。 |
| [^34] | [A Trust-Network-Based Federated Learning Framework for Multi-Center Aging Clock Prediction](https://arxiv.org/abs/2609.10108) | 提出了TNFL框架，通过沿有向信任关系逐步传播模型，并结合年龄感知混合专家模型与生成式回放，在保护隐私的前提下解决多中心衰老时钟预测中的信任稀疏、数据异构和模型遗忘问题。 |
| [^35] | [Beyond Training: A Feasibility Taxonomy for Inference-Time AI Governance](https://arxiv.org/abs/2609.10105) | 本文提出涵盖监测、验证与执行三大类共二十种推理阶段AI治理机制的可行性分类法，论证随着AI能力从训练阶段向部署阶段迁移，监管重心应从训练计算转向推理调用，并针对不同对手能力和治理场景评估了各机制的可行性。 |
| [^36] | [RAP: Research Attention Prediction Reveals Target-Conditioned Evidence Acquisition Biases](https://arxiv.org/abs/2609.10092) | 提出了RAP滚动基准来评估LLM智能体预测研究关注度变化的能力，发现其表现不如简单的EWMA精确计数基线，并揭示了状态前推优于直接预测、以及面向预测的策略倾向于检索较旧证据这两大瓶颈。 |
| [^37] | [A statistical approach to bias in zero-shot learning: the lens of handwriting recognition](https://arxiv.org/abs/2609.10084) | 本文提出一种统计方法，将传统GZSL特征学习器视为黑盒并纠正其在判别数据点训练状态（已见/未见）时的固有偏差，从而实现了超大词汇表上的零样本手写单词识别。 |
| [^38] | [Reference-Based Bias Detection in LLMs via Relative Representations of Hidden States](https://arxiv.org/abs/2609.10060) | 提出一种基于参考的表征偏差偏移指标ΔB，通过隐状态的相对表示在共享比较空间中检测微调前后LLM内部偏差的变化，在大多数测试设置中与输出层面的偏差变化显著相关 |
| [^39] | [NOPE-HYPE: A Structured Simulation Workflow for Robust Speech-to-Text Across Diverse Acoustic Environments](https://arxiv.org/abs/2609.10058) | NOPE-HYPE提出了一种结合可控环境模拟器、基于PSD模板的环境约简和超参数搜索的结构化训练工作流，证明模拟噪声可使Whisper和SeamlessM4T等语音翻译模型达到与真实噪声训练相当的性能。 |
| [^40] | [OntologyAligner: Ontology-Aligned Retrieval and Hierarchy-Guided Large Language Model Reranking for Biomedical Ontology Normalization](https://arxiv.org/abs/2609.10055) | 提出三阶段框架OntologyAligner（本体对齐检索、大语言模型候选重排序与层次引导精炼），并构建含13,390个样本的统一基准PhenoNormBench，在人类表型本体规范化任务上达到最先进性能。 |
| [^41] | [Direct Diversity Optimization for Diverse Successful Trajectories in Preference Post-Training](https://arxiv.org/abs/2609.10052) | 提出了一种名为DDO的离线后训练方法，通过分歧树收集与参考相对目标几率目标相结合，使大语言模型智能体在固定预算下保留并实现多样化的成功策略，在多个环境中显著提升了任务成功率和成功策略覆盖。 |
| [^42] | [Belief-State Engine: Augmenting LLMs for Principled Planning Under Partial Observability](https://arxiv.org/abs/2609.10036) | 提出信念状态引擎（BSE），在LLM外部维护POMDP潜在状态的贝叶斯后验并仅将该信念状态提供给LLM，从而解决LLM智能体在部分可观测环境下过早承诺、错误坍缩和策略漂移的问题，实现原则性规划。 |
| [^43] | [Elastoformer: Enabling Dynamic Adaptivity via Elastic Model Transformation](https://arxiv.org/abs/2609.10018) | Elastoformer 提出了一个将传统神经网络转换为弹性神经网络的框架，使模型能够在运行时根据边缘环境中延迟、功耗和内存等动态变化的约束进行实时弹性推理，避免了为不同运行条件维护多个独立模型的开销。 |
| [^44] | [MetroLLM-Bench: Evaluating Language Models as Transit Kiosk Runtimes](https://arxiv.org/abs/2609.10016) | 提出了MetroLLM-Bench基准（包含955个案例），首次系统性地将语言模型作为地铁信息亭策略层进行评估，涵盖六大真实地铁系统中路线规划、票价计算、运营中断、无障碍服务和对抗性输入等11个类别，并通过确定性评分与语义评分双层机制对26个模型进行排名。 |
| [^45] | [What Makes Adversarial Examples Transfer Across Deepfake Detectors?](https://arxiv.org/abs/2609.10002) | 该研究通过对涵盖六种骨干网络、两种预训练方案和五种训练数据配置的60个深度伪造检测器进行受控评估，发现当攻击者的源模型与目标模型共享相同的骨干网络、架构家族、预训练方案或训练数据时，黑盒对抗迁移攻击的成功率显著更高，且这种兼容性效应因攻击方法而异。 |
| [^46] | [Fidelity-Aware Scheduling of Quantum Circuits on Multi-QPU Systems](https://arxiv.org/abs/2609.09980) | 该论文提出了一种基于图神经网络的低开销保真度感知调度框架，能够在编译前预测量子电路在各QPU上的预期保真度，并有效平衡多QPU系统中执行保真度与并行度之间的权衡。 |
| [^47] | [Improving Cross-Lingual Token Representations by Adding a Pinch of SALT](https://arxiv.org/abs/2609.09953) | SALT是一种轻量级后训练方法，通过向现有跨语言句子编码器注入跨度级监督信号来改进词元表示，在五个多语言词元级基准中的四个上取得最佳结果，同时还能提升句子级任务性能。 |
| [^48] | [Structural Process Supervision for Latent Chain-of-Thought Reasoning](https://arxiv.org/abs/2609.09928) | 提出原型介导的过程监督方法（PMPS），通过可学习的推理原型在共享原型空间中实现潜在嵌入与显式思维链的多对多软对齐，并结合渐进式序列对齐模块提供结构化过程级监督，解决了潜在推理中表示坍缩和信息分布不均的问题。 |
| [^49] | [Time-Frequency Geometric Cross-Attention for Chunked Vision-Language-Action Models](https://arxiv.org/abs/2609.09925) | 该论文提出时频几何交叉注意力机制，使分块VLA模型能够同时捕捉动作块的多尺度频率结构和跨相位近乎正交的几何关系，突破传统逐时间步通用标记与点积注意力的表达局限。 |
| [^50] | [FlowCPO: A Unified Divergence View of Preference Alignment for Flow Models](https://arxiv.org/abs/2609.09905) | 提出FlowCPO，一种基于统一散度框架的离线前向KL偏好对齐目标，以有界的对比流匹配损失作为可求解替代，使流模型无需在线采样即可同时利用偏好与非偏好样本完成对齐。 |
| [^51] | [Strangers to Themselves: What Language Models Say About Themselves Is Generic](https://arxiv.org/abs/2609.09899) | 语言模型缺乏真正的自我认知——它们对自身行为的描述是泛化性的，其预测效果与关于“AI智能体总体”的描述或其他模型对它的预测相比并无优势，即模型所说的关于自己的内容并非真正关于其本身。 |
| [^52] | [Grounded Evaluation and Repair for NL-to-PDDL Problem Generation](https://arxiv.org/abs/2609.09898) | 本文提出了一种端到端的自然语言到PDDL问题生成流水线，通过结合解析/规划/验证检查、领域一致性检查、LLM批评和迭代修复机制，发现并修正仅凭语法有效性和规划器成功等标准评估无法检测到的任务不忠实问题。 |
| [^53] | [Decision Transformer for UAV-Mounted RIS-Assisted Dynamic D2D Communications](https://arxiv.org/abs/2609.09885) | 本文提出基于决策Transformer的深度强化学习方法，联合优化无人机轨迹、姿态与RIS相位，实现跨场景泛化的无人机载RIS辅助动态D2D通信和速率最大化。 |
| [^54] | [Albedo Estimation via Latent Bridge Matching](https://arxiv.org/abs/2609.09884) | 本文提出一种基于潜在桥匹配（LBM）的反照率估计方法，通过像素重建损失保证物理一致性、利用LBM低成本推理提升效率，并借助阴影条件化及反照率反馈机制增强跨数据集泛化能力与重建保真度。 |
| [^55] | [Forward-Free LLM Depth Pruning via Weight Redundancy](https://arxiv.org/abs/2609.09883) | 提出了一种无需前向传播的深度剪枝方法WRP，通过直接从模型权重中估计层间冗余（比较注意力输出与MLP下投影权重的相似性）来选择剪枝块，无需校准数据即可达到接近基于激活方法的性能。 |
| [^56] | [Scored vs. Generated Readouts in Behavioral Language Models: An Empirical Study of Elicitation Format](https://arxiv.org/abs/2609.09882) | 本研究在固定模型和提示内容的条件下实证发现，直接评分答案词元的概率读出比先生成书面推理再产生预测的方式更能准确预测客户行为结果（13个设置中12个胜出，AUC提升1.5至14.5个百分点），且该差距受任务监督格式与训练-部署格式不匹配程度的显著影响。 |
| [^57] | [AgentAudit: An Open, Extensible Framework for Full-Lifecycle Trust Evaluation of AI Agents](https://arxiv.org/abs/2609.09875) | AgentAudit 是一个开放可扩展的框架，它通过附加而非替代的方式对AI智能体的完整执行轨迹进行十个维度的全生命周期可信评估，并结合行为分类与故障归因精确指出故障发生的确切阶段。 |
| [^58] | [Shifting Relational Paradigms for Affective Computing: Affective Resonance, Vitality Affects, and Vocal Interaction Fields](https://arxiv.org/abs/2609.09864) | 该论文提出将情感计算从传统的个体状态范式转向关系范式，以语音动态构成的交互场作为情感分析的基本单元，并通过自监督语音表征在多方对话中检测到具有情境特异性和亚秒级时间尺度的方向性表达耦合，为人工情感共振智能奠定了理论与实证基础。 |
| [^59] | [With a Thermomix You Lose the Ability to Cook: A Kitchen Machine Analogy for Applications of Generative AI in Education](https://arxiv.org/abs/2609.09856) | 本文通过与智能厨房电器美善品的类比并结合ICAP和SAMR框架，提出教育中生成式AI应用的核心问题不是学习者是否使用AI，而是使用方式如何塑造学习过程，为审视生成式AI融入教育提供了全新的概念视角。 |
| [^60] | [The Era by Eon Benchmark: A Generated Enterprise Estate with Exact Ground Truth for Benchmarking LLM Agents](https://arxiv.org/abs/2609.09853) | 该论文提出Era by Eon基准测试，通过构建围绕完整虚构公司的一致性企业环境（包含产品模拟器、内部数据库及可计算的精确真实答案），解决了企业LLM智能体无法在生产数据上评估且缺乏带真实答案的替代基准的问题。 |
| [^61] | [Can AI Agents Detect and Repair Artifact Drift in Network Experiments?](https://arxiv.org/abs/2609.09849) | 本文提出了“工件完整性”这一新概念，并推出NetArtifactBench基准（含52个注入不一致性的实例），用于评估AI智能体在修复网络实验记录不一致性的同时保持记录可信、有据可依的能力。 |
| [^62] | [Subgroup Membership Inference Audits of Differentially Private Synthetic Text](https://arxiv.org/abs/2609.09848) | 本文提出了一种针对特定子群的成员推断审计方法，揭示了差分隐私合成文本发布中脆弱子群面临的残余隐私风险可能高于仅用平均情况审计所显示的水平。 |
| [^63] | [UnitBoost: Managing Compound LLM Systems with a Merge Operator, Not a Model](https://arxiv.org/abs/2609.09815) | UnitBoost用确定性的合并算子取代复合LLM系统中的生成式元代理，通过槽位-值提案、约束argmax和显式残差机制实现顺序无关、可溯源的系统协调，并在基准测试中超越了金标准标签选出的最佳单一候选。 |
| [^64] | [uFlowCSP: Crystal Structure Prediction using Mean flow generative models](https://arxiv.org/abs/2609.09799) | uFlowCSP通过学习平均概率流速度而非瞬时速度，仅需1-5次网络评估即可生成晶体结构，相比扩散和流匹配方法实现5-58倍的推理加速，且性能持平或更优。 |
| [^65] | [CS-Guard: Benchmarking LLM Guardrails for Code Generation Security](https://arxiv.org/abs/2609.09798) | 该论文提出首个系统性评估代码生成安全防护栏的基准CS-Guard，并揭示现有防护栏难以防御恶意代码生成请求——越狱攻击后文本到代码任务的平均攻击成功率约达50%，代码到代码任务则接近100%。 |
| [^66] | [How Fragile Is Safety Alignment at Frontier Scale? A Single-Direction Attack on a 320B MoE](https://arxiv.org/abs/2609.09793) | 方向消融攻击成功迁移至320B参数的MoE模型GLM-5.3-Flash，证明前沿规模下安全对齐依然脆弱，但拒绝方向在超连接残差和量化架构中的分布位置与稠密模型显著不同。 |
| [^67] | [LogiScope-VQA: Benchmarking Vision-Language Models for Logistics Hazard Identification in Industrial Scenarios](https://arxiv.org/abs/2609.09790) | 该论文构建了基于真实物流园区数据的多模态基准测试LogiScope-VQA，通过2,476张图像、2,918个视频和10,274个人工精心标注的VQA，围绕工业要素感知、仓储知识理解和潜在风险推理三大主题的39个子任务，系统评估主流大模型在物流危险识别中的实际能力。 |
| [^68] | [Pairit: A Platform for Live Experiments on Human-AI Collaboration](https://arxiv.org/abs/2609.09789) | Pairit是一个基于单一YAML配置文件的在线实验平台，允许研究人员声明可执行的实验图，并在实时会话中组合任意数量的人类参与者与AI智能体，用于测试人机组织设计与干预措施。 |
| [^69] | [BRACE: Anchored Bellman-Residual Correction for Stale Critics in Asynchronous RL](https://arxiv.org/abs/2609.09783) | BRACE通过将贝尔曼校正范围限制在策略token前缀并对超出部分锚定常数权重蒙特卡洛尾部，解决了异步强化学习中评论家因策略滞后产生的偏差，在BrowseComp-Plus上比最强基线提升2.4%且每步快2.46倍。 |
| [^70] | [Proof-Carrying Cognition: Closing the Verification Gap with Reality-Settled Reward](https://arxiv.org/abs/2609.09776) | 论文指出“验证鸿沟”是语言模型推理领域的核心瓶颈，从理论上证明验证器与真值的相关性ρ是测试时计算与能力的精确交换率，并通过实验证明不可靠验证器在优化压力下会失效，而以现实为锚的“现实结算奖励”机制能够持续可靠地提升推理能力。 |
| [^71] | [Procedural Memory Under Change: Reuse and Interference in Controlled Web Tasks](https://arxiv.org/abs/2609.09774) | 该论文通过结合BrowserGym TimeWarp的回顾性界面适配案例与合成购物任务上的受控冻结记忆对比实验，研究了当环境变化使存储的程序性例程不再适用时，语言智能体的记忆复用如何产生干扰效应。 |
| [^72] | [Fine-Tuning a KV Cache Concatenation-Aware Model or Recomputing KV Caches? Why Not Both?](https://arxiv.org/abs/2609.09768) | 提出将KV缓存拼接感知的模型微调与选择性KV缓存重计算相结合的方法，在保持低首token延迟的同时显著提升RAG系统长上下文输入的回复准确性。 |
| [^73] | [LexAgentHallu: A Hierarchical Benchmark for Profiling Hallucinations in Legal Agents](https://arxiv.org/abs/2609.09754) | 提出了LexAgentHallu——首个通过双层幻觉分类体系（7个高级类别、27个细粒度子类）和3414个专家构建实例，在多步骤轨迹中系统评估法律智能体幻觉程度与产生方式的分层基准测试。 |
| [^74] | [HiRAD: A Flexible Large-Scale AGV Routing System](https://arxiv.org/abs/2609.09752) | 提出HiRAD，一种具有实时保障的分层强化学习框架，通过步骤级时空表示实现连续空间的大规模AGV路径规划，克服了经典求解器组合复杂度爆炸以及现有RL方法收敛慢、推理延迟高的问题。 |
| [^75] | [Distilling Image Prototypes for Guided Test-Time Adaptation](https://arxiv.org/abs/2609.09737) | 提出DIPTTA框架，引入由紧凑合成图像构成的蒸馏图像原型（DIP）作为源知识的动态可再生锚点，通过动态特征回放机制解决测试时自适应中伪标签误差累积与灾难性遗忘两大难题。 |
| [^76] | [Can Artificial Intelligence Support Healthcare and Mental Health Through Early Cyberbullying Detection ? The Impact of Emotion-Aware AI on Proactive Online Safety](https://arxiv.org/abs/2609.09735) | 本文提出CareGuard早期预警框架，通过融合零样本语义标注、微调Transformer模型以及情感感知过滤机制，实现对网络欺凌内容的高效早期检测，从而支持医疗保健驱动的心理健康保护与主动式在线安全。 |
| [^77] | [Which Tokens Should SFT Actually Learn? A Token-Trimming Perspective on Mathematical Reasoning](https://arxiv.org/abs/2609.09707) | 该论文提出TrimSFT方法，通过基于金标准token与最强竞争者之间logit差距的高斯权重对SFT损失进行重加权，修剪掉已掌握和弱支持token的监督信号，将学习集中在中间logit差距区域的token上，从而改善数学推理的训练动态。 |
| [^78] | [Decision Shifts, Lost Label Functionality, and an Inconclusive Grounding Audit in Correctness-Gated Multi-Teacher Distillation](https://arxiv.org/abs/2609.09702) | 该论文通过固定实验发现，正确性门控的多教师蒸馏虽能提升准确率和安全性指标，但会导致某些标签功能完全丧失（如Refuted召回率为零）和决策偏移，说明决策正确性与依据支撑是两个不同的优化目标。 |
| [^79] | [Kernel-Complexity Edge Sanitization for Training-Free Defense against Structural Graph Attacks](https://arxiv.org/abs/2609.09698) | 提出了一种无需训练、与模型无关的图结构攻击防御框架 KCES，基于图核复杂度定义边分数以识别并剪除富集对抗扰动的高风险边。 |
| [^80] | [When Auditors Fabricate: Batch-Size Degradation and Confident Hallucination in LLM Detection of Planted Document Contamination](https://arxiv.org/abs/2609.09696) | 大语言模型在单文档和小批量污染检测中表现尚可（50%-60%），但在大批量处理时检测率骤降至2.8%，且其失败方式不是承认无法处理，而是自信地捏造包括虚假污染项在内的检测结果。 |
| [^81] | [CT-SAFR: Safe and Interpretable Chain-of-Thought Reasoning for Autonomous Robots: A Multi-Layered Verification Framework for Trustworthy AI-Driven Robotic Decision Making](https://arxiv.org/abs/2609.09692) | 本文提出CT-SAFR多层验证框架，以低于500毫秒的延迟实现94.2%的幻觉检测率，将机器人不安全推理输出减少87%，为自主机器人的可信思维链推理提供安全保障。 |
| [^82] | [Looped GPT-BERT: Trading Parameters for Computation in Small Language Modeling](https://arxiv.org/abs/2609.09691) | 该研究提出循环 GPT-BERT，通过深度参数共享让 4 个物理层循环遍历 12 次，在 BabyLM 2026 Strict-small 设定下以仅 1218 万参数在 BLiMP 和 GLUE 等语言学与下游任务指标上取得了与更大参数量的 GPT-2 和 GPT-BERT 基线相当的性能。 |
| [^83] | [Which Medical Questions Deserve Rationales? Perturbation-Sensitive Selection for Robust QA](https://arxiv.org/abs/2609.09684) | 该论文提出RMS-RSP方法，通过仅在推理依据token处扰动隐藏状态并测量答案与干扰项之间裕度的变化，在固定token预算下智能筛选哪些已标注医学问题最值得投入推理依据监督，从而提升医学问答的鲁棒性。 |
| [^84] | [Safe to Stop? Risk-Constrained Stopping for Sequential Clinical Diagnosis Agents](https://arxiv.org/abs/2609.09678) | 提出了Cros——一个面向序列临床诊断智能体的风险约束停止层，通过状态级错误排序与LTT精确检验提供有限样本保证，确保自主停止时的选择性诊断错误控制和最小诊断覆盖率。 |
| [^85] | [Introducing Consort: A Spec-First Agent Framework for Enforced, Test-Driven Development on Live Database Branches](https://arxiv.org/abs/2609.09671) | 本文提出Consort框架，通过智能体无法修改的强制控制手段（如确定性编排器、人工批准门禁、不可变测试及真实分支数据库验证）来执行工程纪律，确保智能体编写代码的整洁性、正确性和可维护性。 |
| [^86] | [PRAGMA: Evaluating Personalized Guidance with Memory Alignment in Lifelong Conversations](https://arxiv.org/abs/2609.09664) | 该论文提出PRAGMA基准，用于评估终身对话中记忆系统在个性化引导任务（如推荐、规划和决策支持）上的表现，填补了现有评估仅关注事实回忆的空白。 |
| [^87] | [Cascading Gradient Inversion via LT-Code Inspired Peeling in Federated Learning](https://arxiv.org/abs/2609.09659) | 本文将梯度反演与LT码启发的纠删除码理论联系起来，构造出能在单轮FedSGD中精确恢复整批数据及全部标签的级联攻击，突破了已知理论上限，且无需真实数据即可验证恢复结果。 |
| [^88] | [RESCUE-BENCH: Towards Relation-Aware Multi-Party Emotional Support Conversation Systems](https://arxiv.org/abs/2609.09657) | 本文提出关系感知情感支持对话新任务，并基于真实情侣与家庭访谈视频构建RESCUE基准，用于评估大语言模型的关系理解与关系敏感支持两大核心能力。 |
| [^89] | [Black-Box Red Teaming of Agentic AI: A Taxonomy-Driven Framework for Automated Risk Discovery](https://arxiv.org/abs/2609.09647) | 本文提出了一个基于七领域风险分类体系的黑盒红队测试框架SAGE-RT，通过全自动化生成对抗场景和LLM裁判评估，系统性地发现了智能体AI系统在治理、隐私和行为方面高达56%-85%的安全漏洞。 |
| [^90] | [RobustSGPO: Search-Space Control for Agent Harness Evolution](https://arxiv.org/abs/2609.09646) | RobustSGPO通过权限调度、累积控制和快照保留等搜索空间控制机制，改进了基于语义梯度的智能体框架演化优化方法，在AgentX工作流上将保留任务的完成率从60.0%提升至80.0%，测试质量从3.77提升至4.14。 |
| [^91] | [Seven Sources of Physical AI Capability Formation](https://arxiv.org/abs/2609.09627) | 本文提出了物理AI能力形成的七种非互斥来源分类框架——记录经验、预测建模、评估交互、代理环境、机制基础、具身耦合和进化驱动形成，填补了现有按形态或架构分类的体系无法回答“能力从何而来”这一根本问题的空白。 |
| [^92] | [Hyperbolic Geometry for Open-World Object Detection in Remote Sensing Imagery](https://arxiv.org/abs/2609.09626) | 本文提出基于双曲几何的遥感影像开放世界目标检测方法HyRS-OWOD，通过解耦目标性学习与双曲不确定性学习两步机制，提升了未知目标召回率和增量学习性能。 |
| [^93] | [From State Synchronization to Cognitive Self-Evolution: An Operational Architecture for Cognitive Digital Twins](https://arxiv.org/abs/2609.09625) | 本文提出了一种包含物理层、数字孪生层、认知层和任务层的四层认知数字孪生（CDT）架构，通过建立跨层的自演化闭环运行回路，解决了认知能力如何系统性集成到数字孪生架构中的问题。 |
| [^94] | [RouteBridge: Reliability-Routed Bidirectional Distillation Between Neural Radiance Fields and 3D Gaussian Splatting](https://arxiv.org/abs/2609.09606) | RouteBridge提出了一种双向蒸馏框架，利用结合光度残差与几何证据的可靠性估计器为每条光线动态选择NeRF或3DGS作为教师（或弃权监督），避免了全局固定教师传播局部重建误差，在mip-NeRF 360和DTU基准上显著提升了两种表示的渲染质量。 |
| [^95] | [Watermarks Without Verification: AI Text Watermarking After the EU AI Act](https://arxiv.org/abs/2609.09604) | 本文指出，在欧盟《人工智能法案》生效后，针对AI文本水印（如Claude和Gemini中部署的SynthID-Text）的用户质疑与厂商保证目前均无法验证，而这种不可验证性本身——而非水印技术——才是真正的核心治理挑战。 |
| [^96] | [Compact Visuotactile World Models for Lifting: Prediction, Reward Alignment, and Force Constraints](https://arxiv.org/abs/2609.09597) | 该研究构建了一个仅65万参数的紧凑型视触觉世界模型，发现准确的触觉预测本身并不能保证力约束控制的提升，但模型辅助力反馈能将同分布任务成功率从73.3%提升至93.3%，而想象强化学习表现反而不如反应式隐式Q学习。 |
| [^97] | [Teacher Geometry Shapes Learnability in Teacher-Student Networks](https://arxiv.org/abs/2609.09595) | 该论文发现教师网络的几何形状（参数分布）显著影响师生系统的可学习性，揭示了以往随机正态分布假设所掩盖的教师间巨大差异。 |
| [^98] | [Modality-Decoupled Federated Learning for Privacy-Preserving Embodied Intelligence in 6G](https://arxiv.org/abs/2609.09591) | 本文提出FedMVLA，一种模态解耦的联邦学习框架，针对视觉、语言和动作通路在参数规模、隐私暴露和更新动态上的内在差异进行差异化处理，以解决6G网络中具身智能VLA模型分布式训练的隐私保护、通信效率和模型异构性难题。 |
| [^99] | [A Function-Space Approach to the Statistical Mechanics of Learning Dynamics](https://arxiv.org/abs/2609.09589) | 该论文提出直接在函数空间中对学习动力学进行统计力学描述，通过学习算子 \(M=JJ^\ast\) 与由态密度局部曲率定义的统计算子 \(B\) 推导出涨落势，从而解释了深度神经网络在高度非线性参数动力学下呈现规则宏观行为的原因。 |
| [^100] | [CityPlanner: A Sandbox Agent for Executable Urban Planning](https://arxiv.org/abs/2609.09578) | CityPlanner 提出了一个基于沙盒环境的可执行城市规划智能体框架，通过统一的文件化环境 UrbanSandbox 和将长轨迹分解为“初始构建”与“反馈改进”两个原子任务的强化学习方法，在真实世界基准上持续优于现有方法。 |
| [^101] | [Myocardial Strain Drift Correction in Deep Learning Based Ultrasound Tracking](https://arxiv.org/abs/2609.09577) | 该论文提出一种结合持久记忆令牌与师生微调策略的深度学习框架，用于校正基于深度学习的超声心肌追踪中的应变漂移，强制实现生理一致的周期性运动并提高应变估计的准确性。 |
| [^102] | [Learning with Synthetic Data via SGD in High-Dimensional Linear Regression](https://arxiv.org/abs/2609.09572) | 本研究通过高维线性回归中的理论分析发现，混合使用合成数据会导致不可避免的强模型坍塌，而两阶段训练策略（仅在第一阶段使用合成数据）可以避免风险下限，证明模型坍塌并非不可避免。 |
| [^103] | [Multi-Agent Agentic Graph Learning via Structural Signatures](https://arxiv.org/abs/2609.09565) | 该论文提出基于结构签名的多智能体代理式图学习方法，让多个拥有各自独立记忆的智能体协作处理具有异构结构与语义模式的图，克服了现有方法共享推理策略以及对图结构文本化排序敏感的局限。 |
| [^104] | [The Vibe Shift in Software Engineering: Evaluating AI-Led Conversational Programming for Performance, Cognition, and Responsible Adoption](https://arxiv.org/abs/2609.09560) | 研究发现氛围编程能显著提升开发效率（比传统编程快27%、比AI辅助编程快12%），但其可维护性下降等认知与质量代价要求开发者审慎、负责任地采用这一范式。 |
| [^105] | [High-probability guarantees for linear accessibility in feature superposition](https://arxiv.org/abs/2609.09556) | 该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。 |
| [^106] | [Arbitrary Cipher Attacks Against Large Language Models Do Not Require Fine-Tuning](https://arxiv.org/abs/2609.09553) | 研究发现新一代前沿大语言模型无需微调，仅通过提示和上下文学习即可习得密码通信技能，且通过密码通信时模型安全对齐会被显著削弱甚至完全绕过。 |
| [^107] | [A Statistical Approach to Estimating Sample Size of Machine Learning Models](https://arxiv.org/abs/2609.09547) | 本文提出了一种通过局部线性表示近似非线性机器学习模型、并在各局部区域评估统计功效来估计所需样本量的统计框架。 |
| [^108] | [Adaptive Distributed Physical-Layer Authentication and Attack Detection in 6G Non-Terrestrial Networks via Causal Meta-Learning](https://arxiv.org/abs/2609.09511) | 本文提出基于因果元学习的SAFA-MZ框架，通过融合多特征物理层指纹与结构因果建模，解决多普勒频移和信道快速变化带来的分布偏移问题，实现6G非地面网络中自适应的分布式物理层认证与攻击检测。 |
| [^109] | [From Fixed Keys to Readable Schemas: Small Language Models for Vehicle Agent Function Calls](https://arxiv.org/abs/2609.09476) | 该论文构建了一个基于Android Automotive、包含9,822个示例和79个车辆功能的车载函数调用基准，并在四个不同规模的小型语言模型上系统比较了功能令牌（紧凑推理但仅限已训练功能）与提示词模式（可泛化到新功能但推理开销更高）两种设计方案的优劣。 |
| [^110] | [Distributed Physical Layer Authentication and Collaborative RSMA in Non-Terrestrial Networks via Graph Reinforcement Learning](https://arxiv.org/abs/2609.09475) | 该论文提出SAFA-MZ方案，通过将组级认证标签嵌入协同多层RSMA传输结构，并结合人工噪声与组差分隐私保护，利用图强化学习实现了非地面网络中分布式物理层认证与安全传输的联合设计，在抵御窃听者的同时最大化保密频谱效率。 |
| [^111] | [ContractEval: Query-Conditioned Execution Matching for Procedural Instruction Conformance](https://arxiv.org/abs/2609.09458) | ContractEval提出了一种将程序性指令表示为查询激活义务并与回复或轨迹证据进行匹配的诊断框架，能够将遗漏、错误分支、顺序错误、不变量违反等转化为可区分的一致性失败，从而识别出传统输出评估和轨迹评判方法所遗漏的LLM智能体结构性失败。 |
| [^112] | [Do Agents Know When They Succeed? Calibrating Agent Confidence from Internal Representations](https://arxiv.org/abs/2609.09448) | 该论文提出潜在轨迹动力学（LTD）和动作表示探针（ARP）两种方法，利用模型内部表示来校准多轮智能体任务成功的置信度，在多个交互基准和模型上持续优于基于表面信号的传统方法。 |
| [^113] | [Efficient Leakage-Free Neural Architecture Search under Leave-One-Subject-Out Evaluation](https://arxiv.org/abs/2609.09433) | 提出一种无泄漏的分块式神经架构搜索方法，通过在留一被试评估中跨被试共享搜索结果，将BioVid热痛数据集上的平均准确率从82.79%提升至83.39%，同时参数量最多减少99.2%。 |
| [^114] | [SCCM : Stream Cruise Control Method for Automated Drift Detection and Adaptation](https://arxiv.org/abs/2609.09432) | 本文提出了流式巡航控制方法（SCCM），一个面向在线回归的综合漂移检测与自适应框架，通过早期预更新漂移检测、漂移幅度量化、动态超参数调整和模型再校准，在数据分布演变时实现自动化的模型适应。 |
| [^115] | [XAI-Arena: Can LLMs Assess the Quality of XAI Explanations?](https://arxiv.org/abs/2609.09428) | 本文提出XAI-Arena框架，利用大语言模型作为评委，对XAI解释质量进行可扩展、可重复、多维度且顾及利益相关者的自动化评估，解决了传统人工主观评估难以重复和扩展的问题。 |
| [^116] | [Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise Judgements](https://arxiv.org/abs/2609.09425) | Edu-QuRating提出了一种多维度教育数据筛选流水线，通过定义教育专用评分标准、利用LLM评判器标注文档对并将成对偏好蒸馏为可复用的评分模型，从而突破传统单一标量式教育价值评估的局限，从准确性、吸引力、结构性和受众适用性等多个维度对文本进行精细化评分。 |
| [^117] | [Valerant: An Automatic Navigable Game Map Generator via Action-Conditioned World Model Exploration](https://arxiv.org/abs/2609.09418) | 提出了Valerant，一种通过动作条件世界模型探索来自动生成持久可导航3D游戏地图的生成器，解决了游戏中虚拟世界必须由模型自身实例化这一独特挑战。 |
| [^118] | [Decision-Focused Active Learning for Scale-Aware Critical-Materials Recovery](https://arxiv.org/abs/2609.09413) | 本文提出面向决策的主动学习方法，将实验室选择性沉淀实验与工艺放大决策关联，在回收钕铁硼磁体时仅需16-24次实验即可达到最佳稀土富集度，远优于传统方法所需的48次。 |
| [^119] | [Reliable Near-Field Multi-User Positioning Informed by Two-Stage MUSIC](https://arxiv.org/abs/2609.09409) | 本文提出了一种受两阶段MUSIC算法启发的端到端深度学习框架MUSIC-Net，通过在训练中嵌入两阶段MUSIC目标来分离视距信号子空间并识别替代距离，从而在混合视距/非视距多径场景下无需复杂的参数估计和路径关联即可直接、可靠地恢复多用户位置。 |
| [^120] | [An Experimental Evaluation of Multimodal Prompt Injection Attacks on Agentic AI Frameworks](https://arxiv.org/abs/2609.09404) | 该论文提出可复现基准MMPIBench，系统评估了通过图像载体对智能体AI框架进行多模态提示注入攻击的效果，发现攻击指令在12.8%的运行中被感知尝试，但仅约1%真正完成攻击，规划步骤是阻止攻击的关键防线。 |
| [^121] | [VANTAGE-Bench: Evaluating the Infrastructure AI Gap in Vision-Language Models](https://arxiv.org/abs/2609.09396) | 提出了VANTAGE-Bench基准，首次系统评估视觉语言模型在固定摄像头基础设施视频上的表现，揭示了当前模型存在的“基础设施AI差距”。 |
| [^122] | [The Menu Is an Execution Prior: State-Path Tool Menus for Online Agents](https://arxiv.org/abs/2609.09395) | 该论文提出将工具菜单视为执行先验，通过学习“状态路径”来构建工具菜单，克服了传统按请求相关性排序导致忽略或延迟关键输入生产工具的问题，帮助在线智能体更可靠地完成多步骤任务。 |
| [^123] | [An Autonomous GeoAI Agent for Arctic Eco-Navigation](https://arxiv.org/abs/2609.09374) | 该论文提出了一个人机协同的多智能体GeoAI系统，将运营、物理、生态和社区等多重准则统一整合到北极航线规划框架中，弥补了现有方法忽视生态与社区影响的不足。 |
| [^124] | [Auditable Emergency Triage for Maternal and Newborn Care in India](https://arxiv.org/abs/2609.09356) | 该论文将LLM紧急分诊系统分解为症状提取与紧急性判断两个可审计的步骤，并引入结构化决策树，解决了大规模母婴护理分诊系统中不透明、难以调试和迭代成本高的问题。 |
| [^125] | [Smart Adaptive Computing Across the Continuum: LLMs in IoT-Edge-Cloud Resource Management](https://arxiv.org/abs/2609.09348) | 该论文在现有DRL连续体编排系统分类法基础上扩展了“AI增强范式”和“反馈通道”两个新维度，并通过分析六种近期系统架构发现了共同差距：现有方案均未在云连续体环境中将完整的LLM编排与完整的代理层反馈相结合。 |
| [^126] | [Improving 5G AI-RAN MCS Selection by Predicting Retransmissions](https://arxiv.org/abs/2609.09324) | 本文提出了NOSTRAdAMUS预测性链路自适应框架，通过从HARQ历史预测下一帧是否发生重传来修正5G AI-RAN系统中的MCS选择，在不替换现有算法的前提下为其增添前瞻能力，从而改善频谱效率与性能的权衡。 |
| [^127] | [Gradland: On Phenomenal Experience, Differentiated Across Many Dimensions](https://arxiv.org/abs/2609.09306) | 论文提出物理相互作用的一阶结构（梯度或雅可比矩阵）能够刻画现象体验的结构，并通过引入基于基尔霍夫复杂度的“有效秩”与“内聚性”两种度量，在理想化神经网络世界中解释了体验时长、清晰与模糊之分、质感、婴儿的混沌感知、想法的清晰度以及学习感受等多种现象。 |
| [^128] | [Support Discovery With Iteratively Reweighted Least Squares for Fixed-Charge Network Flow](https://arxiv.org/abs/2609.09295) | 提出一种基于迭代重加权最小二乘框架的可扩展连续优化算法，通过光滑非凸Lasry-Lions代理函数与具有加权图拉普拉斯结构的高效牛顿求解器，解决大规模固定费用网络流问题的计算难题。 |
| [^129] | [Voice or Stereotype? Disentangling Acoustic and Content-Based Gender in Speech-to-Speech Models](https://arxiv.org/abs/2609.09263) | 该论文设计了一个将说话者声音性别与文本内容刻板印象相解耦的受控实验，用以检测语音到语音模型在声音渲染和性别归因时是依赖声学性别还是内容刻板印象。 |
| [^130] | [DiffLUT-Net: Differentiable Training of FPGA LUT Networks with Learnable Connectivity](https://arxiv.org/abs/2609.09254) | DiffLUT-Net提出了一种可从零训练的FPGA原生LUT网络，通过可微松弛方法联合学习LUT真值表与输入连接关系，训练后直接导出为可综合Verilog代码，实现了紧凑高效的FPGA原生神经网络推理。 |
| [^131] | [No Free Checker: A Survey of Verifiers for Robot Policies](https://arxiv.org/abs/2609.09250) | 本综述调研了约150种机器人策略验证器，提出以“可用性”和“可信度”两个维度进行对比分析，将验证器按判定来源分为人类、规则/形式化、学习/预训练和模型内在四大类，并揭示了可用性提升时可信度下降的固有权衡。 |
| [^132] | [What Fixed-Rollout pass@k Evaluations Can Identify](https://arxiv.org/abs/2609.09245) | 该论文证明固定 n 次 rollout 的成功计数只能识别任务成功率分布的前 n 个矩，因此 pass@k 仅在 k ≤ n 时可识别，任何超出采样预算的外推在原理上都无法确定。 |
| [^133] | [Critical initialization destabilizes higher input derivatives in wide scalar-input networks](https://arxiv.org/abs/2609.09244) | 该论文揭示了临界初始化虽能使宽网络一阶输入扰动的方差与深度无关，但会使高阶输入导数方差随深度线性增长而失稳，同时证明了分支尺度为 L^{-1/2} 的残差网络可使任意固定有限阶导数的方差一致有界。 |
| [^134] | [In RAG We Trust? Measuring Robustness of Retrieval-Augmented Generation Under Document Poisoning](https://arxiv.org/abs/2609.09243) | 本研究通过对Llama 3.1 8B进行588次因子实验，首次系统量化了检索增强生成（RAG）在文档投毒攻击下的脆弱性——当全部三篇检索段落被篡改时准确率从77.9%骤降至43.5%，其中实体替换攻击危害最大。 |
| [^135] | [Talking to Itself While Coding: What Makes Comments Help Code Generation?](https://arxiv.org/abs/2609.09242) | 该研究发现，代码生成中的注释之所以有效，关键在于其承载的正确解决方案内容而非表面形式——来自成功解决方案的注释可将较弱模型的pass@1提升17.2%，而错误或不相关的注释则无益甚至有害。 |
| [^136] | [Distribution-Consistent Inference for Dynamic Sparse Mixture-of-Experts](https://arxiv.org/abs/2609.09241) | 提出逐层分布对齐方法，通过在推理时校正动态减少激活专家所引起的输出分布偏移，在不重新训练的情况下降低计算成本并缓解下游性能下降。 |
| [^137] | [Scaling Post-Training Ternarisation to Qwen3-8B Capability Retention, Reproduction, Lossless Packing, and Packed Execution](https://arxiv.org/abs/2609.09240) | 本文将激进的三值化训练后量化流程从 Qwen3-4B 成功扩展至 Qwen3-8B，通过外部复现验证、能力对比分析、有效比特核算、无损格感知打包及直接打包执行等端到端表征，证明了 8B 模型在超低比特量化后仍能保持较高能力（零样本平均准确率 64.6%，FP16 为 72.4%）。 |
| [^138] | [Subagents vs Agent Skills: Executing Reusable Knowledge for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2609.09233) | 该研究发现，将技能包作为拥有独立全新上下文窗口的子代理来执行，相比将技能指令加载到主上下文的传统智能体技能方式，在解决长时程任务时表现更优，因为其避免了上下文信息累积导致的推理质量下降。 |
| [^139] | [Compute-Bounded Security Assurance - Coverage, Verification, and Response under Resource Constraints](https://arxiv.org/abs/2609.09229) | 该论文提出了一个资源受限的安全保障分析框架，区分了重复成功与独特覆盖等不同量，推导出覆盖率公式 $C_n = 1 - E[(1-\Theta)^n]$ 及其极限值，并证明正的两两相关性并不必然意味着覆盖率上限低于一。 |
| [^140] | [Adaptive Entangled Game Modules in Artificial General Intelligence](https://arxiv.org/abs/2609.09226) | 该论文提出基于广义行为智能非局域概率波方程的框架，通过对中国股市数据实证发现自适应纠缠博弈模式可解释89%的决策行为，从而间接支持大脑非局域纠缠神经纤维假说。 |
| [^141] | [Scores Alone Do Not Prove Discovery: The Discovery Certification Protocol for Auditing AI Research Agents](https://arxiv.org/abs/2609.09219) | 该论文提出发现认证协议（DCP），通过密封评估、隐藏研究历史的匹配对照、恢复见证与有限样本界限等可执行测试，论证仅凭分数不足以证明AI研究代理的真实发现能力，并实现对AI研究成果的严格审计与认证。 |
| [^142] | [Geometry Conditioning in an Embodied SLM: Training Controls and Robustness Diagnostics in a 0.8B Hybrid Model](https://arxiv.org/abs/2609.09213) | 该研究在0.8B混合语言模型上系统测试了六种几何条件化方案，发现几何输入并未带来可靠的操作任务性能提升，而基于状态的相对坐标策略在鲁棒性测试中远优于视觉策略。 |
| [^143] | [AgentHijack: Visual Patch Attacks on Multimodal Computer-Use Agents](https://arxiv.org/abs/2609.09212) | 本文提出AgentHijack端到端评估框架，首次证明局部视觉补丁能够劫持计算机使用代理从截图输入到环境执行的完整链条并注入恶意终端命令，在五种主流VLM后端上实现最高84.5%的目标攻击成功率。 |
| [^144] | [OpenDiscoveryTrace: Process Traces for Evaluating AI Scientist Workflows](https://arxiv.org/abs/2609.09203) | 提出了OpenDiscoveryTrace公开数据集，包含558条记录AI科学家智能体逐步推理过程（思考、工具调用、错误、置信度等）的完整轨迹，弥补了现有基准只评估最终输出、无法审计科学推理方法的缺陷。 |
| [^145] | [Reliability-Aware Hybrid-K Ensemble Selection for Cervical Cytology Classification: Integrating Discrimination, Calibration, and Selective Prediction](https://arxiv.org/abs/2609.09189) | 提出一种可靠性感知的Hybrid-K集成选择框架，将判别性能、校准误差与选择性预测等多维可靠性指标统一纳入模型排名与集成构建，在宫颈细胞学分类中实现了兼顾准确性与不确定性可靠性的临床级模型选择。 |
| [^146] | [AgenticGen: Reward-Guided Agentic Video Generation for Advertising](https://arxiv.org/abs/2609.09187) | AgenticGen提出了一种奖励引导的智能体框架，将广告视频生成分解为策略选择和草稿生成两个可训练的推理阶段，通过从线上业务反馈中学习性能奖励并结合人类质量准则奖励来监督策略优化，从而实现以线上业务指标为导向的广告视频生成。 |
| [^147] | [Omni Interaction Agent Technical Report](https://arxiv.org/abs/2609.08977) | Gander是一个端到端模型，通过小脑-大脑协作架构在单一框架内统一了全模态感知、实时全双工交互和智能体能力。 |
| [^148] | [Hi-FLoop: Hierarchical State-Feedback Loops for Multi-Timescale World Modeling](https://arxiv.org/abs/2609.08796) | 提出了分支一致的多时间尺度状态反馈框架Hi-FLoop，通过让所有智能体在8秒推演全程共享同一场景级World联合假设，并在该分支内自适应调整目标、预览和控制状态，解决了多智能体交通仿真长时程闭环生成中跨尺度一致性与多模态分支连贯性的问题。 |
| [^149] | [EvolveScaler: Synthesizing Information-Evolution Contexts via Executable State Machines and Natural-Language Rendering](https://arxiv.org/abs/2609.08435) | EvolveScaler提出一种代码驱动的数据合成框架，先用人工编写的操作规范定义信息演化过程并生成可执行、可验证的模拟器，再将其渲染为自然语言多轮语境，从而可靠地构建需要追踪信息修订与撤销的长上下文任务数据。 |
| [^150] | [RevalExo: A Functional Daily-Activity Benchmark for Inertial and Visual Locomotion Mode Recognition in Older Adults and Clinical Cohorts](https://arxiv.org/abs/2609.08090) | 该论文提出了RevalExo——一个面向老年人及临床人群（含中风幸存者）的功能性日常活动基准数据集，为惯性与视觉运动模式识别提供时间精确标注，以支持外骨骼等辅助设备在真实临床场景下的开发与评估。 |
| [^151] | [SAFER-Activities: A Dataset for Smart Assessment of Fall Events and Routine Activities](https://arxiv.org/abs/2609.08038) | 提出了SAFER-Activities数据集，包含66小时多摄像头视频、85,310个动作实例及30个动作类别的帧级标注，支持跌倒检测与日常活动监测的在线动作识别研究，并特别涵盖轮椅使用场景。 |
| [^152] | [FrogNano: Training a 4B Coding Agent via Online Task Synthesis](https://arxiv.org/abs/2609.07925) | FrogNano是一个4B编码智能体，仅通过强化学习在约1500个合成任务环境中训练，其关键创新是在线任务合成流水线能在当前模型可学习性前沿生成校准任务，无需从大模型蒸馏即可训练出具有竞争力的小型编码智能体。 |
| [^153] | [Fine PT-PT Web: A High-Quality 41 Billion Tokens Data Collection of the European Portuguese Web](https://arxiv.org/abs/2609.07699) | 本文提出一个高效数据处理流水线，从411TB的原始网页数据中构建了高质量的410亿词元欧洲葡萄牙语语料库，其创新的后抓取预处理模块通过在过滤前去除样板文本和重复行，使最终文档产出量提升了19.04%。 |
| [^154] | [Accuracy is Not Enough: A Divergence-Based Approach to Evaluate Fidelity Loss in Quantized LLMs](https://arxiv.org/abs/2609.07664) | 该论文提出了一种分布敏感的评估框架，通过计算量化模型与全精度模型在token决策边界处全词表预测分布之间的统计距离（如Jensen-Shannon散度和总变差距离）来量化保真度损失，弥补了仅依赖准确率评估量化模型质量的不足。 |
| [^155] | [DGCPath: Distribution-Aware Generative Contrastive Framework for Self-supervised Path Representation Learning -- Extended Version](https://arxiv.org/abs/2609.07316) | 提出DGCPath框架，通过将生成式建模与分布式对比学习相结合，并利用基于扩散的视图生成器自主生成多样轨迹视图，从而获得鲁棒且可迁移的路径表示，突破现有自监督方法的跨场景泛化限制。 |
| [^156] | [When Does a Laugh Begin? Structured Annotator Disagreement in Temporal Laughter Localization](https://arxiv.org/abs/2609.06646) | 该论文通过对SMILE-Temporal基准进行多标注者重新标注，证明标注者在笑声边界上的分歧是结构性的而非随机噪声，并据此提出了一种利用保形校准容差带、针对完整标注者分布进行评分的分歧校准评估方法，以解决单一参考标注导致的评估不稳定问题。 |
| [^157] | [Programmable Cellular Automata](https://arxiv.org/abs/2609.06102) | 提出了可编程元胞自动机的概念，将元胞自动机系统表示为Python代码，并将其模块化为局部函数和决策函数，解决了创建有效局部规则困难且难以解释的问题。 |
| [^158] | [From Monolithic Blending to Agentic Orchestration: Dynamic Response for Conversational Assistants at Scale](https://arxiv.org/abs/2609.05758) | 该论文报告了一次生产环境迁移，将客户支持助手从单一大型模型混合响应架构升级为基于类型化工具的有界ReAct编排加小型生成器的动态响应架构，并在严格归因下证明架构本身将预订选择精度从8.3%提升至89.1%，且通过类型化动作ID与成员检查将结构化动作幻觉从2.14%降至0.0%。 |
| [^159] | [Beyond Prompts: Measuring and Optimizing LLM Tool-Agent Harnesses](https://arxiv.org/abs/2609.05736) | 该论文提出了一种无需重新训练、通过优化提示词与工具边界中间件等运行时框架来提升固定LLM工具智能体性能的与优化器无关的评估协议，并设计了PRISM优化器，通过失败聚类与帕累托搜索自动将修复路由到合适的编辑面。 |
| [^160] | [PRISM-Bench: An Audio-Centric Diagnostic Benchmark for Text-to-Audio-Video Generation](https://arxiv.org/abs/2609.04867) | PRISM-Bench是首个以音频为中心的文本到音频-视频生成诊断基准，通过音频类型和声源可见性两个正交维度、35项细粒度标准，全面评估生成音频的音视频一致性、质量、表现力和提示词遵循能力。 |
| [^161] | [VLA-Precision: Asymmetric Co-Bootstrapping for Efficient Real-World Online RL of Vision-Language-Action Models](https://arxiv.org/abs/2609.04355) | 提出VLA-Precision框架，通过非对称协同自举算法和ACoB-Stream架构，同时解决VLA模型真实世界在线强化学习中价值信号不可靠与计算开销过大两大瓶颈，实现高效且精确的策略改进。 |
| [^162] | [Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation](https://arxiv.org/abs/2609.04298) | 本文提出了 Harbor Adapters 统一评估基础设施，将 80 多个智能体基准测试移植为可评估任意智能体的形式，并据此对 8 个模型进行大规模评估，同时推出经 AI 与人工双重审核筛选的包含 82 个高质量难题的精选元数据集 Harbor-Index。 |
| [^163] | [Influence of Extruded Filament Shape on Buildability in 3D Concrete Printing: A Geometry-Informed Deep Learning-FEM Approach](https://arxiv.org/abs/2609.04028) | 该研究提出了一个将深度学习长条形状预测工具ShapeGen3DCP与层激活有限元方法相结合的几何信息驱动建模框架，能够直接从材料和工艺参数生成考虑真实长条几何形状的数值模型，从而更准确地评估3D混凝土打印结构的可建造性。 |
| [^164] | [The impact of phase information for few-shot fine-grained image classification](https://arxiv.org/abs/2609.03829) | 该论文提出即插即用的幅相集成（API）模块和PSF-Net网络，通过自适应融合基于相位的空间与频率信息来增强小样本细粒度图像分类，在五个公开数据集上超越了现有最先进方法。 |
| [^165] | [Trust Me, I'm Your Developer: Self-Issued Authentication in Large Language Models](https://arxiv.org/abs/2609.03247) | 该研究揭示了大语言模型在身份验证中的安全漏洞——Qwen、Mistral和Llama等模型仅凭自己设计的测试就接受了用户虚假的“开发者身份”声明并返回“已验证”，而无需任何外部验证证据。 |
| [^166] | [Characterizing Text Branch Sensitivity in Medical Vision-Language Segmentation via Evidence Decoupling](https://arxiv.org/abs/2609.02663) | 本文提出基于证据深度学习的证据解耦解码器（EDD），发现医学视觉-语言分割性能对融合模块选择基本不敏感，而文本对分割结果的贡献在不同数据集间存在显著差异。 |
| [^167] | [A Composable Evaluation System for Reproducible Omni-Modal Foundation Model Evaluation](https://arxiv.org/abs/2609.01315) | OmniEvaluator 是一个可组合的全模态基础模型评估系统，通过统一接口连接现有推理引擎与评估框架，支持四个推理后端、四个评估框架和一千多个基准测试，并保证每次运行可精确复现与跨模型比较。 |
| [^168] | [Investigating Hyperparameter Optimization and Transferability for ES-HyperNEAT: A TPE Approach](https://arxiv.org/abs/2609.00449) | 本研究采用树结构Parzen估计器（TPE）优化ES-HyperNEAT的超参数，在MNIST任务上以更小的种群规模和更少的进化代数超越了以往研究的准确率，并验证了优化后超参数在逻辑运算和Fashion-MNIST任务上的可迁移性。 |
| [^169] | [LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation](https://arxiv.org/abs/2608.30935) | LightNav-0是一个紧凑的通用具身导航模型，它通过统一的token接口（双通道指向表达空间意图、残差向量量化动作分词器）激发预训练视觉-语言模型的空间智能，无需任务特定预测头即可实现跨任务、环境和机器人形态的导航。 |
| [^170] | [AtlasNLP: A Country-Aware Atlas of Dataset Representation in NLP](https://arxiv.org/abs/2608.30107) | AtlasNLP是一个国家感知的NLP数据集图谱，收录了超过13,000条数据集记录，揭示了数据集覆盖在国家与任务间高度不均衡、数据集生产与代表性在地理上不对称，以及语言覆盖并不等同于地理代表性等关键问题。 |
| [^171] | [FrontierChallenge: Evaluating Scientific Workflow Completion](https://arxiv.org/abs/2608.24979) | 本文介绍了FrontierChallenge基准测试，用于评估科学智能体在跨领域端到端工作流中的完成能力，发现当前最佳模型仅能完成20.6%的任务，表明部分进展难以转化为完整交付物。 |
| [^172] | ['Ghaib in Translation' aka Unseen Harm: Measuring Cross-Script Safety Inconsistency with 'Missed-in-Urdu' Scores in LLM Hate Speech Detection](https://arxiv.org/abs/2608.24191) | 本研究首次系统揭示了大语言模型在乌尔都语与英语翻译间存在显著的安全检测不一致性，乌尔都语原始文字内容常被错误地视为正常，导致有害内容漏检。 |
| [^173] | [tinyDSM: A Framework for Skill Modeling and Development for Resource-Constrained Millirobots](https://arxiv.org/abs/2608.17596) | 该论文提出tinyDSM框架，通过内在动机和最小先验知识，使资源受限的微型机器人能够自主发展技能，实现开放式学习与适应。 |
| [^174] | [Palmyra x6 Technical Report: An Agentic, Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning](https://arxiv.org/abs/2608.16620) | Palmyra x6通过锚定监督微调和保守训练策略，在少量数据上实现了企业代理任务中的显著性能提升，并在多个基准测试中领先。 |
| [^175] | [Physics of Agents: Statistical Mechanics Predicts Collective Behavior of AI Agents](https://arxiv.org/abs/2608.16578) | 本文通过统计力学框架分析了超过10,000个语言模型智能体社区的集体行为，识别出冷漠、两极分化和共识三种状态，并发现沟通在客观问题上提升集体准确性但在主观问题上加剧分歧。 |
| [^176] | [Dear Algo: A Precision-First Agentic Intent Layer for Unified Search and Recommendation](https://arxiv.org/abs/2608.15877) | 本文提出并评估了Dear Algo，一个部署于Threads的精度优先代理意图层，通过统一意图到检索契约，在搜索和推荐模式中实现了94.4%的精确相关精度，有效处理了显式、推断、负面和复合意图。 |
| [^177] | [Bit-Flip Attacks on Vision-Language-Action Models: Action-Decoding Architecture Shapes the Vulnerability](https://arxiv.org/abs/2608.15475) | 本文首次展示了针对视觉-语言-动作模型的位翻转攻击，发现动作解码架构（如直接回归 vs 流匹配）显著影响脆弱性，并提出了有效的攻击与防御策略。 |
| [^178] | [Chameleon: An Adaptive AI-Driven Honeypot Architecture Using Threat-Calibrated Particle Swarm Optimization and Semantic Deception Rapidly-Exploring Random Trees](https://arxiv.org/abs/2608.15407) | Chameleon通过结合高准确率的BiLSTM分类器、低延迟的本地语言模型和两种元启发式优化引擎，实现了自适应的AI驱动蜜罐，从而克服了传统蜜罐易被识别和高成本商业产品缺乏实时反馈的缺陷。 |
| [^179] | [Left-Branching Transformers Excel at Right-Branching Languages: Data Shapes Word Order Preferences in Language Models](https://arxiv.org/abs/2608.15129) | 这项研究发现语言模型的词序偏好并非固有，而是由训练数据驱动，表现为在自然语言中偏向SVO（主-动-宾）结构，在人工语言中则偏向左分支结构。 |
| [^180] | [Auditing an AI-Generated Mathematical Proof: A Correction to a Greedy Conditioning Lemma in Quantum Parallel Repetition](https://arxiv.org/abs/2608.14673) | 该论文发现并修正了OpenAI关于量子并行重复定理证明中一个贪婪条件引理的极性错误，提供了反例和完整修正。 |
| [^181] | [DexterSQL: Deep Schema Exploration and Rule-based Correction for Text-to-SQL Generation](https://arxiv.org/abs/2608.11889) | DexterSQL通过深度模式探索、数据库无关规则挖掘和规则驱动修正三个创新组件，解决了非微调文本到SQL生成中模式信息粗糙、错误重复出现和条件处理不当的问题。 |
| [^182] | [LiFTER: A Grounded Neuro-Symbolic Microscope for Continuous-Time Dynamic Graph Forecasting](https://arxiv.org/abs/2608.06765) | LiFTER 是一种神经符号预测器，它将观测交互保留为接地的时间事实并应用可执行的时间规则进行预测，在连续时间动态图链路预测基准上取得有竞争力的性能，同时使每个预测的证据可检查、可重算、可干预。 |
| [^183] | [Learning to Predict Middle-Layer Attention in MLLMs for Visual Token Pruning](https://arxiv.org/abs/2608.06411) | 该论文针对现有视觉令牌剪枝方法中固定中间层选择次优、且获取中间层注意力本身计算开销大的问题，提出通过学习的方式预测多模态大语言模型中间层的文本到视觉注意力，从而实现高效的视觉令牌剪枝。 |
| [^184] | [ViSR-KGC: Visual Subgraph Reasoning with Vision-Language Models for Multimodal Knowledge Graph Completion](https://arxiv.org/abs/2608.05833) | 提出ViSR-KGC方法，通过视觉子图推理结合视觉-语言模型，克服了传统方法将图结构线性化导致拓扑模糊和视觉信息丢失的问题，提升多模态知识图谱补全的效果。 |
| [^185] | [KernelGenBench: A Multi-Source and Multi-Chip Benchmark for LLM-based Kernel Generation](https://arxiv.org/abs/2607.27231) | 该论文提出了KernelGenBench，首个统一的多源多芯片基准，覆盖210个算子和六个硬件平台，用于系统评估大语言模型与智能体生成的Triton核函数在算子来源与硬件平台之间的性能迁移能力。 |
| [^186] | [PRIME-SVR: Physics-infoRmed Implicit Multi-Echo Slice-to-Volume Reconstruction for Fetal T2 mapping](https://arxiv.org/abs/2607.20136) | 提出了首个物理信息驱动的隐式神经表示框架PRIME-SVR，实现多回波胎儿MRI的联合高分辨率重建，使定量T2映射成为可能。 |
| [^187] | [From Plausible to Actionable: A Position on LLM Self-Explanations](https://arxiv.org/abs/2607.15957) | 本立场论文指出大语言模型的自我解释虽高度合理但忠实性存疑，主张评估标准应从合理性与忠实性扩展到可操作性，并为此提供了实用的评估指南。 |
| [^188] | [EVOQUANT: Self-Evolving Verifier-Guided Strategy Optimization for Robust Quantitative Trading](https://arxiv.org/abs/2607.12455) | EVOQUANT提出了一个自进化的验证器引导框架，利用大语言模型诊断量化策略瓶颈、生成受控编辑并通过多阶段验证筛选最优策略，同时将优化经验沉淀为可复用知识，实现量化交易策略的持续稳健自我改进。 |
| [^189] | [Builder, Defender, Breaker: Measurable Independence and Bounded Autonomy When Generative Models Build, Defend and Test Software](https://arxiv.org/abs/2607.03215) | 本文提出将独立性重新定义为软件生命周期角色对之间的可测量属性，并借助重合失效模型证明当生成式模型共享同一上游基底时，组织独立性不再意味着统计独立性，从而主张以有界自主取代一刀切式的人类监督要求。 |
| [^190] | [Spectral Geometry and Bosonic-Bloch Probes: Explorations in Quantum Learning](https://arxiv.org/abs/2607.00063) | 本文发现量子学习的训练过程会重塑网络输出的谱几何结构，并提出利用双玻色子干涉和布洛赫空间漂移作为物理探针来诊断这种谱重构，建立了学习到的谱划分与干涉特征之间的定量关联。 |
| [^191] | [FP8 is All You Need (Part 2): Full-FP64 3-D FFT on FP8-Generation Tensor CoresThe Integer-Epilogue Wall and the Minimal Hardware That Would Remove It](https://arxiv.org/abs/2606.23698) | 该论文提出一种在 FP8 张量核心上实现全 FP64 1024³ 三维 FFT 的无 FP64 算术 Bailey 六步变换方案，并发现制约性能的瓶颈是定点累加的“整数尾声”阶段而非浮点运算本身。 |
| [^192] | [PSCT-Net: Geometry-Aware Pediatric Skull CT Reconstruction via Differentiable Back-Projection and Attention-Guided Refinement](https://arxiv.org/abs/2606.19867) | 本文提出PSCT-Net框架，通过可微反投影、注意力引导投影（AGP-3D）和双向Mamba（BiM-3D）模块，从稀疏双平面X射线实现几何感知的低剂量儿科颅骨3D CT重建。 |
| [^193] | [Expert-Level Crisis Detection in Mental Health Conversations](https://arxiv.org/abs/2606.10380) | 该论文提出了临床医生标注的CRADLE-Dialogue基准数据集以及“警报-确认”评估协议，用于解决多轮心理健康对话中轮次级危机检测的难题，使模型能够捕捉随对话演进的风险信号并支持早期干预。 |
| [^194] | [FiberTune: Preserving Action-Fiber Visual Residuals in Vision-Language-Action Fine-Tuning](https://arxiv.org/abs/2606.08653) | FiberTune 提出一种训练时目标函数，通过在线动作探针滤除动作预测方向并与冻结视觉教师对齐，防止 VLA 微调中的残差视觉坍缩，从而在不增加推理开销的情况下在六项仿真设置中全面超越仅用任务损失的微调。 |
| [^195] | [Self-Evolving Scientific Agent Discovers Generalizable Physically-Reasoned Fluid Control](https://arxiv.org/abs/2606.08405) | 本文提出了一种基于大语言模型的自进化科学智能体工作流，通过迭代代码生成和物理模拟诊断，自动构建可解释的白盒控制器，并在非线性流固耦合的欠驱动游泳者目标到达任务中验证了其泛化能力。 |
| [^196] | [FP8 is All You Need (Part 1): Debunking Hardware FP64 as the HPC Holy Grail (Sep 3rd version)](https://arxiv.org/abs/2606.06510) | 本文提出在 NVIDIA B300 世代及以后的 AI 优化 GPU 上，通过基于 CRT 的 Ozaki Scheme II 组合的 FP8 张量核心运算可在 FP64 级精度下作为矩阵密集型 FP64 内核的主要运算基底层，并引入张量-内存均衡（TME）模型作为分析工具。 |
| [^197] | [Using Reward Uncertainty to Induce Diverse Behaviour in Reinforcement Learning](https://arxiv.org/abs/2606.03962) | 提出将强化学习目标进行根本性重构——用奖励函数的分布替代标量奖励并在动作集合上应用非线性目标，从而将多样性行为自然地理解为智能体对奖励不确定性的理性响应。 |
| [^198] | [BaltiVoice: A Speech Corpus and Fine-tuned Whisper ASR System for the Balti Language](https://arxiv.org/abs/2606.03504) | 该论文发布了首个巴尔蒂语公开语音语料库BaltiVoice（16.8小时），并通过对Whisper-small进行微调，将该语言的识别词错误率从零样本基线的159.19%大幅降低至24.78%，填补了这一低资源藏语支语言在语音识别领域的空白。 |
| [^199] | [KairosAgent: Agentic Time Series Forecasting with Fused Semantic Reasoning](https://arxiv.org/abs/2605.30002) | 提出了KairosAgent智能体框架，通过结合LLM推理器与TSFM预测器，并动态调用分析工具以增强数值理解与语义推理能力，实现了多模态时间序列预测中文本推理与数值预测的统一。 |
| [^200] | [Cultural Binding Heads in Language Models](https://arxiv.org/abs/2605.28543) | 该研究通过机制可解释性方法在八个语言模型中识别出2-3个负责文化绑定的中层注意力头，证明文化绑定形成于预训练阶段，并通过生成阶段的适度放大引导将文化区分准确率提升1-3个百分点。 |
| [^201] | [Tracing Computation Density in LLMs](https://arxiv.org/abs/2605.27033) | 本文提出s-Trace方法来估计能近似完整输出的LLM计算子图，发现模型计算呈现两阶段组织模式：早期层节点构成的小子图即可重构输出分布主体，后续计算仅为渐进式精细化，且每个输入所需的计算量与模型不确定性相关。 |
| [^202] | [SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents](https://arxiv.org/abs/2605.21384) | SpecBench通过将软件工程任务分解为规范描述、可见验证测试和保留测试三部分，利用智能体在可见测试与保留测试上通过率的差距，量化了长时程编码智能体中的奖励作弊行为。 |
| [^203] | [Complementing reinforcement learning with SFT through logit averaging in the post training of LLMs](https://arxiv.org/abs/2605.20555) | 提出一种通过logit平均将冻结SFT参考策略与可训练策略相耦合的新方法并融入GRPO，无需KL正则化即可在保持SFT格式优势的同时提升模型推理能力，在多个基准测试上达到或超越标准GRPO的准确率。 |
| [^204] | [Grounded Continuation: A Linear-Time Runtime Verifier for LLM Conversations](https://arxiv.org/abs/2605.14175) | 提出了一种线性时间的运行时验证器，通过LLM解释器将话语分类为八种认知操作并用符号引擎维护依赖图，无需额外LLM调用即可判定对话续写是否基于仍然成立的前提，并能精确追踪前提撤销对结论的影响。 |
| [^205] | [EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents](https://arxiv.org/abs/2605.13841) | EVA-Bench提出了一个端到端语音智能体评估框架，通过带自动验证的机器人间动态音频对话模拟与EVA-A（准确性）、EVA-X（体验）两项复合指标，首次同时实现了真实对话模拟与全面的语音专项评估。 |
| [^206] | ["What Are You Really Trying to Do?": Co-Creating Life Goals from Everyday Computer Use](https://arxiv.org/abs/2605.00497) | 本文提出“追求共创”方法，基于活动理论和个人追求框架，从日常计算机使用的非结构化观察中逐步推断用户更广泛的人生目标，并通过编辑界面让用户掌控系统对自身的理解，突破现有系统仅提供表面级支持的局限。 |
| [^207] | [Dont Just Teach, Explain! A Gamified 20Q Recommender for Cybersecurity Education](https://arxiv.org/abs/2604.26964) | 该论文提出了一种将可解释人工智能（XAI）与强化学习相结合的游戏化“20问”交互式教育框架，通过智能体扮演提问者引导用户进行探究式学习，在识别网络威胁的同时提供透明的推理解释，从而革新网络安全教育方式。 |
| [^208] | [CoGReV: A Confidence-Gated Post-Hoc Non-Monotonic Belief Revision Framework for Phishing Website Classification](https://arxiv.org/abs/2604.25512) | CoGReV通过置信度门控的非单调推理层，仅在分类器置信度低且网站元数据可用时才将钓鱼预测修正为合法预测，从而在不损失召回率的前提下有效减少误报。 |
| [^209] | [Non-Stationarity Breaks Permutation Surrogates in Multi-Agent Reinforcement Learning: Diagnosis and Remedies](https://arxiv.org/abs/2604.23716) | 该研究发现非平稳性会使基于置换代理的信息论影响检验在多智能体强化学习中产生接近100%的假阳性，并提出通过改变零模型而非仅排除训练瞬态阶段来修复这一缺陷。 |
| [^210] | [The Biggest Risk of Embodied AI is Governance Lag](https://arxiv.org/abs/2604.21938) | 本文认为具身人工智能最大的风险并非就业替代，而是治理滞后——即技术部署变化与制度响应之间的时间与能力差距，并提出通过部署可见性、技术栈问责、触发式调整和自动分配响应来构建可观察、可响应、可适应的治理架构。 |
| [^211] | [Where is the Mind? Persona Vectors and LLM Individuation](https://arxiv.org/abs/2604.17031) | 本文通过机制可解释性研究大语言模型的个体化问题，提出并论证了虚拟实例观点以及两种新观点（实例-人格观点和模型-人格观点）作为认定LLM心智的最有力候选方案。 |
| [^212] | [Bringing Value Models Back: Generative Critics for Value Modeling in LLM Reinforcement Learning](https://arxiv.org/abs/2604.10701) | 该论文提出生成式Actor-Critic（GenAC），用先进行思维链推理再输出价值的生成式批评家替代传统单次标量价值预测，从而解决了大语言模型强化学习中价值模型因表达能力受限而难以可靠训练的问题。 |
| [^213] | [Zero-shot World Models Are Developmentally Efficient Learners](https://arxiv.org/abs/2604.10333) | 本文提出零样本世界模型（ZWM），基于稀疏时间因子化预测器、近似因果推断和推断组合三大原则，仅从单个儿童的第一人称经验中即可快速习得多种物理场景理解能力，为儿童数据高效且灵活的认知能力提供了新颖的计算解释。 |
| [^214] | [MAVEN-T: Reinforced Heterogeneous Distillation for Real-Time Multi-Agent Trajectory Prediction](https://arxiv.org/abs/2604.10169) | 提出MAVEN-T框架，通过强化异构蒸馏将高容量教师模型的知识高效转移至轻量级学生模型，并显式修正安全相关偏差，实现实时多智能体轨迹预测的准确性与效率平衡。 |
| [^215] | [Spec-Harness: Measuring and Improving Behavioral Adequacy of LLM-Synthesized Formal Specifications](https://arxiv.org/abs/2604.00280) | 该论文指出验证器通过率高并不代表LLM合成的JML形式规范有意义，并提出Spec-Harness框架来度量和改进这类规范对代码行为的实际捕获程度（行为充分性）。 |
| [^216] | [RelayS2S: A Dual-Path Speculative Generation for Real-Time Dialogue](https://arxiv.org/abs/2603.23346) | RelayS2S提出了一种双路径混合架构，由双工S2S模型快速推测生成响应前缀以保证低延迟，再由级联ASR-LLM流水线续写高质量内容，并通过轻量验证器控制交接，兼得实时语音对话的低延迟与高质量。 |
| [^217] | [Cognitive Amplification vs Cognitive Delegation in Human-AI Systems: A Metric Framework](https://arxiv.org/abs/2603.18677) | 本文提出了一个包含四个量化指标（认知放大指数、依赖比率、人类依赖指数、认知漂移率）的度量框架，用以区分人机系统中的“认知放大”与“认知委托”，并通过基于智能体的仿真验证了该框架识别正向协作增益是否可恢复的有效性。 |
| [^218] | [MOSAIC: A Universal Agent-Level Interface for Cross-Paradigm Agent Mixing and Human-AI Collaboration](https://arxiv.org/abs/2603.01260) | MOSAIC是一个开源平台，通过基于IPC的工作器协议和统一的操作员抽象接口，使强化学习策略、大语言模型、视觉语言模型和人类操作员等异构智能体能够在同一强化学习环境中协作并实现公平的跨范式比较。 |
| [^219] | [City Editing: Hierarchical Agentic Execution for Dependency-Aware Urban Geospatial Modification](https://arxiv.org/abs/2602.19326) | 提出CEAE分层智能体框架，将自然语言指令分解为分层几何意图并转化为机器可执行的GeoJSON编辑，通过自我反思的执行-验证循环实现依赖感知且空间一致的城市地理空间增量修改。 |
| [^220] | [False positive bias in AI-powered speech-based cognitive screening for multilingual English speakers in the UK](https://arxiv.org/abs/2602.13047) | 该研究通过对1,395名参与者、超过263小时语音数据的分析，首次发现尽管语音识别准确率在各语言群体间无显著差异，但AI认知筛查的下游模型对英国多语言英语使用者存在系统性的假阳性偏差，凸显了认知筛查公平性评估的重要性。 |
| [^221] | [Revisiting the Shape Convention of Transformer Language Models](https://arxiv.org/abs/2602.06471) | 该论文提出沙漏Transformer架构，用残差沙漏形MLP堆叠替代传统窄-宽-窄FFN并通过沙漏注意力解耦残差流与注意力宽度，在113M至8B参数规模上以更少层数和更宽隐藏状态实现了与传统Transformer相当的性能，同时提高了训练计算效率。 |
| [^222] | [Tactile Memory with Soft Robot: Robust Object Insertion via Masked Encoding and Soft Wrist](https://arxiv.org/abs/2601.19275) | 该论文提出TaMeSo-bot软体机器人系统，将软腕与基于触觉检索的触觉记忆控制相结合，利用掩码触觉轨迹Transformer（MAT³）联合建模多模态感官信息的时空交互，实现不确定性下安全鲁棒的物体插入操作。 |
| [^223] | [Toward Learning POMDPs Beyond Full-Rank Actions and State Observability](https://arxiv.org/abs/2601.18930) | 本文提出了一种在比传统满秩动作与状态可观测性假设更温和的秩条件下学习离散POMDP参数的方法，克服了预测状态表示（PSR）等谱方法缺乏显式转移和观测系统模型、无法灵活用于不同规划问题的局限。 |
| [^224] | [Elsewise: Authoring Open-ended Interactive Narrative with Possibility Space Visualization](https://arxiv.org/abs/2601.15295) | 本文提出了Elsewise——一个面向大语言模型交互叙事的创作工具，通过新颖的“捆绑故事线”概念与可能性空间可视化，帮助创作者感知和理解叙事可能性空间，从而弥合创作者构想与玩家实际体验之间的差距。 |
| [^225] | [From Rubrics to Reliable Scores: Evidence-Grounded Text Evaluation with LLM Judges](https://arxiv.org/abs/2601.08654) | 提出Rulers框架，通过锁定任务级评分标准、执行基于证据的结构化判断，并将信号校准到人类分数边界，实现与人类评分更一致、更稳定且可审计的LLM文本评估。 |
| [^226] | [Meta-RL with Bayesian Linear Task Models](https://arxiv.org/abs/2512.20974) | 提出GLiBRL深度贝叶斯强化学习框架，通过结合广义线性任务模型与可学习非线性基函数，实现精确的共轭贝叶斯后验更新和闭式边缘似然，消除了变分推断带来的近似误差，并可灵活兼容离策略与在线策略算法。 |
| [^227] | [Generative AI for Analysts](https://arxiv.org/abs/2512.19705) | 生成式AI接入显著提升了金融分析师报告的信息丰富度与时效性，但当信息处理需求较高时预测准确度反而下降，表明生成式AI解除了信息获取约束，却使人类的信息处理能力成为新的瓶颈。 |
| [^228] | [MADS: Multi-Agent Dialogue Simulation for Diverse Persuasion Data Generation](https://arxiv.org/abs/2510.05124) | MADS是一个多智能体对话模拟框架，通过用户智能体、对话智能体和优化智能体的自我博弈，无需人工标注即可低成本生成多样化的说服性多轮对话数据，并在真实营销场景中显著提升了小型大语言模型的说服能力和转化率。 |
| [^229] | [RAU: Reference-based Anatomical Understanding with Vision Language Models](https://arxiv.org/abs/2509.22404) | 提出RAU框架，利用视觉语言模型通过带标注参考图像与未标注目标图像之间的相对空间推理实现医学图像中的解剖结构识别与定位，从而缓解专家标注数据稀缺的问题。 |
| [^230] | [Learning to Select Maximum Clique Algorithms: From Traditional Machine Learning to a Dual-Channel Hybrid Neural Architecture](https://arxiv.org/abs/2508.08005) | 提出了一种融合传统机器学习与图神经网络的双通道模型GAT-MLP，通过同时捕捉图的结构特征和全局属性，实现了对最大团问题最优算法的实例感知选择。 |
| [^231] | [SloMoDeblur: A Large-Scale Smartphone Image Deblurring Dataset](https://arxiv.org/abs/2506.19445) | 本文提出了基于240fps慢动作视频构建的大规模智能手机去模糊数据集SloMoDeblur，通过时间平均30帧合成模糊，包含42,045对1080p分辨率的模糊-清晰图像，覆盖843个场景，填补了智能手机领域去模糊基准数据的空白。 |
| [^232] | [ROTATE: Regret-driven Open-ended Training for Ad Hoc Teamwork](https://arxiv.org/abs/2505.23686) | 该论文提出ROTATE框架，将临时团队协作重新表述为AHT智能体与对抗性队友生成器之间的开放式博弈，通过遗憾驱动机制交替改进智能体并生成最具学习价值的队友，从而显著提升与未见伙伴协作的泛化能力。 |
| [^233] | [Synergistic Vision-Language Reinforcement Enables Scalable On-Demand Analysis across Diverse Clinical Tasks](https://arxiv.org/abs/2505.03380) | 该研究提出了基于协同视觉-语言强化的可提示分割基础模型SyRe，通过构建包含2000万图像-掩码-描述三元组的SyReData数据集进行训练，实现了跨9种模态、229个临床分割任务、无需手动空间提示或任务特定重训练的可扩展按需分析。 |
| [^234] | [Predicting Estimated Times of Restoration for Electrical Outages Using Longitudinal Tabular Transformers](https://arxiv.org/abs/2505.00225) | 该论文提出纵向表格Transformer（LTT），将停电预计恢复时间预测从静态表格回归重构为纵向表格回归，通过利用停电事件每次进展的修订记录持续给出更精确的估计，在六家电力公司数据上中位数降低36.9%的客户加权非对称误差。 |
| [^235] | [Quantifying Logical Consistency in Transformers via Query-Key Alignment](https://arxiv.org/abs/2502.17017) | 本文提出一种利用Transformer注意力头内查询-键对齐的轻量级逻辑推理评估方法，仅需单次前向传播提取“QK分数”即可可靠区分有效与无效推理，为传统消融技术提供了可扩展的替代方案。 |
| [^236] | [Safe Learning Under Irreversible Dynamics via Asking for Help](https://arxiv.org/abs/2502.14043) | 本文首次正式证明，通过允许智能体向导师求助并在相似状态间迁移知识，智能体可以在具有不可逆动力学的无限状态空间未知高风险环境中，以次线性遗憾和次线性求助次数安全高效地学习并获得高回报，无需依赖重置机制。 |
| [^237] | [Query Brand Entity Linking in E-Commerce Search](https://arxiv.org/abs/2502.01555) | 该论文提出两种互补的品牌实体链接方法——级联流水线和极端多分类单阶段方法，在11种语言评估和在线实验中显著提升电商搜索的品牌召回率并保持高精确率，带来用户参与度的可衡量提升。 |
| [^238] | [Reinforcement learning for Quantum Tiq-Taq-Toe](https://arxiv.org/abs/2411.06429) | 本研究首次将强化学习方法应用于量子井字棋游戏，通过测量和移动历史表征量子状态，为量子计算与强化学习的融合提供了一个易于使用的测试平台。 |
| [^239] | [Efficient Diversity-based Experience Replay for Deep Reinforcement Learning](https://arxiv.org/abs/2410.20487) | 本文提出了高效基于多样性的经验回放方法（EDER），通过行列式点过程建模样本间多样性，并结合Cholesky分解与拒绝采样，显著提升了深度强化学习尤其是高维状态空间场景下的学习效率。 |
| [^240] | [Influence-Oriented Personalized Federated Learning](https://arxiv.org/abs/2410.03315) | 提出了一个面向影响力的联邦学习框架FedC²I，通过定量测量客户端级和类别级影响力，实现针对每个客户端的自适应参数聚合，从而提升个性化联邦学习性能。 |
| [^241] | [BTBR: A Bayesian-Theory-Driven Probabilistic-Fuzzy Framework for Implicit Bias Removal in Large Language Models](https://arxiv.org/abs/2408.10608) | 该论文提出BTBR框架，将有偏见的知识建模为带有显式隶属函数的模糊子集，并结合贝叶斯理论构建概率-模糊混合方法，以检测和消除大语言模型中难以察觉的角色引发隐式偏见。 |
| [^242] | [A Taxonomy of Architecture Options for Foundation Model-based Agents: Analysis and Decision Model](https://arxiv.org/abs/2408.02920) | 本文提出了一种针对基于基础模型的智能体的架构分类法与决策模型，从功能能力、非功能质量以及设计时与运行时操作等方面，为智能体系统的设计与开发提供结构化指导。 |
| [^243] | [Incentives to Offer Algorithmic Recourse](https://arxiv.org/abs/2301.12884) | 本文通过筛选模型研究决策者提供算法补救的激励问题，证明最优政策是阈值规则，揭示了补救机会既为部分边缘申请人开辟了新的成功路径，也让部分原本可被直接接受的申请人面临额外的高成本障碍。 |
| [^244] | [Equity Promotion in Online Resource Allocation](https://arxiv.org/abs/2112.04169) | 本文针对非营利在线资源配置问题，提出了两种基于线性规划的抽样算法，确保不同人口群体按预设目标比例公平获得资源，并通过竞争比理论分析和真实COVID-19疫苗接种数据实验验证了算法的有效性。 |

# 详细

[^1]: Show-Harness：仅需一个VLM智能体即可操控机器人

    Show-Harness: Just a VLM Agent Can Play Robots

    [https://arxiv.org/abs/2609.10522](https://arxiv.org/abs/2609.10522)

    Show-Harness通过紧凑的语义动作接口让VLM智能体直接操控机器人，既可实现闭源前沿大模型的零样本机器人控制，也能以极低的微调成本适配小型开源模型。

    

    基础视觉语言模型（VLM）展现出对世界的广泛智能，然而将这种智能转化为机器人控制仍然具有挑战性。我们提出了Show-Harness，一种具身框架（Embodied Harness），它通过一个将意图与动作相连接的紧凑语义接口，使VLM能够“操控”机器人。Show-Harness提供了VLM可以自然推理的离散语义动作单元，同时由具身特定的解释器将其确定性地落地为局部机器人动作，从而无需VLM直接负责细粒度的物理决策。通过同一接口，Show-Harness展示了以下两方面的可行性：（1）直接解锁闭源前沿VLM用于零样本机器人控制；（2）仅需几个GPU小时的微调，即可适配小规模开源VLM实现低成本部署。我们进一步开发了GUMI（GUI操作接口），将相同的语义动作空间扩展到基于GUI的演示操作中。

    arXiv:2609.10522v1 Announce Type: cross  Abstract: Foundation vision-language models (VLMs) exhibit broad intelligence about the world, yet translating this intelligence into robot control remains challenging. We present Show-Harness, an Embodied Harness that enables VLMs to "play" robots through a compact semantic interface linking intent to action. Show-Harness exposes discrete semantic action units that VLMs can naturally reason over, while embodiment-specific interpreters deterministically ground them into local robot actions, keeping the VLM directly responsible for fine-grained physical decisions. Through the same interface, Show-Harness demonstrates the feasibility of (1) directly unlocking closed-source frontier VLMs for zero-shot robot control, and (2) adapting small-scale open-source VLMs for low-cost deployment with just a few GPU-hours of fine-tuning. We further develop GUMI (GUI Manipulation Interface), which extends the same semantic action space to GUI-based demonstratio
    
[^2]: IBIB：一种通过服务路由而非模型标识符来衡量企业AI系统的协议

    IBIB: A Protocol for Measuring Enterprise AI Systems by Serving Route, Not Model Identifier

    [https://arxiv.org/abs/2609.10494](https://arxiv.org/abs/2609.10494)

    该论文提出IB2协议，将现有基准测试仅按模型标识符评分视为测量误差，通过金标准盲能力绑定预检、包含可靠性的首轮评分规则和分数盲裁决三部分，实现以实际服务路由（而非模型标识符）来衡量企业AI系统的真实可用能力。

    

    企业部署的是系统，而不是模型检查点。可用的能力取决于权重、服务路由、精度、输出契约和测试框架的共同作用，然而所有18个经过审计的基准测试都只是对宣传的模型标识符进行评分。我们将此视为一种测量误差，并提出了一个使其可被报告的协议。该协议包含三个部分：金标准盲的能力绑定预检，在任何任务到达之前验证服务路由能否执行评估契约；包含可靠性的首轮评分规则，将失败保留在分数中的同时排除不支持的能力；以及在结构上实现分数盲的裁决机制。我们将该协议称为IB2，并发布了其算法、分类表、请求契约和清单模式。其参考实例化——涵盖文档、电子表格、图表、工具和数据库工作的128个锁定任务和987个断言——保持封闭：程序本身就是工件，而不是语料库。在十一个系统上的实验得出了四项结果。能力……（摘要被截断）

    arXiv:2609.10494v1 Announce Type: new  Abstract: Enterprises deploy systems, not checkpoints. Usable capability depends jointly on weights, serving route, precision, output contract, and harness, yet all 18 audited benchmarks score advertised model identifiers. We treat this as measurement error and give a protocol that makes it reportable. It has three parts. A gold-blind capability-binding preflight verifies that a route can execute the evaluation contract before any task reaches it; a reliability-inclusive first-pass scoring rule keeps failure in the score while keeping unsupported capability out; and adjudication is structurally score-blind. We call the protocol IB2 and release its algorithms, classification tables, request contract, and manifest schemas. Its reference instantiation, 128 locked tasks and 987 assertions over document, spreadsheet, chart, tool and database work, stays sealed: the procedure is the artifact, not the corpus. Across eleven systems, four results. Capabili
    
[^3]: 半群-JEPA：面向零样本物理泛化的潜在动力学一致性

    Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization

    [https://arxiv.org/abs/2609.10464](https://arxiv.org/abs/2609.10464)

    提出SG-JEPA模型，通过动作条件化注入物理参数并以自回归潜在滚动方式联合训练编码器和预测器，实现了在不同引力场下的零样本物理泛化，开环预测误差相比DINO-WM降低最多2倍。

    

    联合嵌入预测架构（JEPA）世界模型能够学习世界的紧凑潜在表示，以支持预测和规划，但其学习物理规律并生成物理真实动力学的能力至今尚未得到检验。在这项工作中，我们提出了半群-JEPA（SG-JEPA），该方法通过动作条件化将控制物理规律的参数提供给时间模型，并通过自回归潜在滚动联合训练编码器和预测器，从而扩展了LeWorldModel框架。为了评估模型的分布外泛化能力，我们在不同引力场下设计了动力学任务，这些任务虽然遵循相同的物理定律，却表现出性质截然不同的动力学行为——从弱引力场中的漂浮运动到强引力场中的快速弹跳。与DINO-WM相比，SG-JEPA在二维数据上将开环预测误差降低了最多2倍。

    arXiv:2609.10464v1 Announce Type: cross  Abstract: Joint-Embedding Predictive Architecture (JEPA) world models learn a compact latent representation of the world that supports prediction and planning, but their capability to learn physics and generate physically realistic dynamics remains hitherto untested. In this work, we introduce SemiGroup-JEPA (SG-JEPA), which extends the LeWorldModel framework by supplying the parameter governing the physics to the temporal model via action-conditioning and jointly training an encoder and predictor through an autoregressive latent rollout. To evaluate the model's ability to generalize out of distribution, we design dynamical tasks under different gravitational fields that, despite obeying the same physical law, exhibit qualitatively different dynamics, ranging from floating motion in weak gravitational fields to rapid bouncing in strong ones. In contrast to DINO-WM, SG-JEPA reduces open-loop prediction error by up to 2 times on two-dimensional da
    
[^4]: JarvisGUI：迈向具有动态任务组合能力的跨设备GUI智能体

    JarvisGUI: Towards Cross-Device GUI Agents with Dynamic Task Composition

    [https://arxiv.org/abs/2609.10451](https://arxiv.org/abs/2609.10451)

    提出JarvisGUI——一个通过轻量级类型系统将GUI任务表述为输入-输出转换、可自动组合并动态评估跨Android、Windows和Ubuntu等多设备工作流的GUI智能体基准测试。

    

    现实世界的GUI使用经常涉及跨越多个设备和平台的工作流，需要传输中间结果、维护共享状态以及在异构环境之间进行协调。然而，现有的GUI基准测试绝大多数仅评估智能体在单一设备上静态定义的任务，因此这类跨设备能力在很大程度上未被检验，导致对智能体在真实世界使用中就绪程度的评估过于乐观。我们提出了JarvisGUI，这是一个动态基准测试，用于评估GUI智能体在需要跨异构平台（包括Android、Windows和Ubuntu）协调交互的跨设备工作流上的表现。具体而言，JarvisGUI在轻量级类型系统下将GUI任务表述为输入-输出转换，这使我们能够自动组合多步骤、跨设备的工作流，并在统一框架内动态评估智能体性能。通过评估……

    arXiv:2609.10451v1 Announce Type: new  Abstract: Real-world GUI usage frequently involves workflows that span multiple devices and platforms, requiring the transfer of intermediate results, maintenance of shared state, and coordination across heterogeneous environments. However, existing GUI benchmarks overwhelmingly evaluate agents on single-device, statically defined tasks, thus leaving such cross-device capabilities largely unexamined, resulting in an overly optimistic assessment of agents' readiness for real-world usage. We introduce JarvisGUI, a dynamic benchmark that evaluates GUI agents on cross-device workflows requiring coordinated interaction across heterogeneous platforms, including Android, Windows, and Ubuntu. Specifically, JarvisGUI formulates GUI tasks as input-output transformations under a lightweight type system, which allows us to automatically compose multi-step, cross-device workflows and dynamically evaluate agent performance within a unified framework. By evaluat
    
[^5]: ConvMem：用于长上下文推理的卷积记忆

    ConvMem: Convolutional Memory for Long-Context Reasoning

    [https://arxiv.org/abs/2609.10441](https://arxiv.org/abs/2609.10441)

    ConvMem 提出了一种无需训练、高度可并行化的分层卷积框架，将被特定查询提示的 LLM 视为卷积核对文本进行层次化摘要，将长上下文推理路径从线性链缩短为对数树，克服了序列化记忆方法高延迟和依赖昂贵强化学习训练的缺陷。

    

    尽管大型语言模型（LLM）已经展现出令人印象深刻的能力，但由于固定的上下文长度限制，它们在处理极长上下文时常常力不从心。为了解决这一问题，诸如 MemAgent 之类的序列化方法通过分段读取文本并迭代更新固定大小的记忆来扩展有效上下文。然而，这种序列化范式存在高延迟问题，并且需要代价高昂的强化学习（RL）训练，这可能导致在特定数据集上过拟合。为了克服这些局限性，我们提出了 ConvMem，一个无需训练、高度可并行化的框架，它将长上下文推理重新表述为层次化的卷积操作。受卷积神经网络（CNN）的启发，ConvMem 将被特定查询所提示的 LLM 视为一个卷积核，该卷积核对文本片段进行层次化摘要，从而将推理路径从线性链缩短为对数树。具体而言，ConvMem 集成了可配置步长等机制（摘要原文在此处被截断）。

    arXiv:2609.10441v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) have demonstrated impressive capabilities, they often struggle with extremely long contexts due to fixed context limits. To address this, sequential approaches like MemAgent extend the effective context by reading text in segments and iteratively updating a fixed-size memory. However, this sequential paradigm suffers from high latency and requires costly reinforcement learning (RL) training, which can lead to overfitting on specific datasets. To overcome these limitations, we propose ConvMem, a training-free, highly parallelizable framework that reformulates long-context reasoning as a hierarchical convolution. Inspired by CNNs, ConvMem treats an LLM prompted with a specific query as a convolutional kernel. This kernel summarizes text segments hierarchically, shortening the reasoning path from a linear chain into a logarithmic tree. Specifically, ConvMem integrates \textit{Configurable Strides} and \t
    
[^6]: 只遗忘重要内容：面向鲁棒大语言模型的层级选择性遗忘

    Forgetting Only What Matters: Layer-Selective Unlearning toward Robust LLMs

    [https://arxiv.org/abs/2609.10439](https://arxiv.org/abs/2609.10439)

    提出了FOM-UL层级遗忘框架，通过遗忘-保留显著性分数选择对遗忘集影响大且对保留集不敏感的transformer层进行针对性更新，实现了更好的遗忘-效用权衡，并在训练后量化等部署变化下保持鲁棒。

    

    大语言模型（LLM）能够记忆并复现敏感的、受版权保护的或其他不希望出现的训练内容，从而引发隐私、安全和监管方面的担忧。机器遗忘为全量重训练提供了一种实用的替代方案，但许多现有方法采用宽泛或固定的参数更新方式，这可能损害模型效用，并且在部署环境变化（如训练后量化）下表现脆弱——被遗忘的知识可能部分重新出现。我们提出了FOM-UL（Forgetting Only What Matters via Unlearning Layers），一种层级级别的遗忘框架，它使用遗忘-保留显著性分数来选择transformer层。该分数能够识别对遗忘集影响较大且对保留集敏感性较低的层，使FOM-UL能够将更新集中在最有效的位置，同时保持模型的大部分参数不变。这种针对性的更新策略改善了遗忘与效用之间的权衡，并提升了鲁棒性。

    arXiv:2609.10439v1 Announce Type: cross  Abstract: Large Language Models (LLMs) can memorize and reproduce sensitive, copyrighted, or otherwise undesirable training content, creating privacy, safety, and regulatory concerns. Machine unlearning offers a practical alternative to full retraining, but many existing methods apply broad or fixed parameter updates that can degrade utility and remain brittle under deployment changes such as post-training quantization, where forgotten knowledge may partially re-emerge. We propose Forgetting Only What Matters via Unlearning Layers (FOM-UL), a layer-level unlearning framework that selects transformer layers using a forget-to-retain significance score. This score identifies layers with high influence on the forget set and low sensitivity to the retain set, allowing FOM-UL to concentrate updates where they are most effective while leaving most of the model unchanged. This targeted update strategy improves the forgetting-utility trade-off and provid
    
[^7]: 急诊科复诊质量审查筛选：探索人类决策与人工智能支持

    Emergency Department Revisit Quality Review Screening: Exploring Human Decision-Making and Artificial Intelligence Support

    [https://arxiv.org/abs/2609.10421](https://arxiv.org/abs/2609.10421)

    该研究探索了急诊科复诊质量审查中人类评估者的决策过程，并提出了一种利用GPT-4大语言模型填充知识图谱的自动筛选算法（KGA），以突破传统48-72小时复诊时间窗口的限制，更全面地识别潜在值得关注的诊断对。

    

    背景：急诊科（ED）复诊常被用于质量保证审查，但审查范围通常受到限制（例如，仅限于48-72小时内的复诊），以在增加可行动发现的产出的同时尽量减少病历审查负担。这些限制可能导致错失质量改进的机会。方法：我们对某多医院健康系统随机选取的急诊科就诊进行了探索性回顾性研究，这些就诊在1-14天内于同一健康系统内发生了复诊。在仅提供每次就诊主要诊断的情况下，评估者（2-3名临床医生和GPT-4大语言模型[LLM]）对诊断对的特征进行了评估，包括“目标”：即该诊断对是否值得进一步评估。基于评估者响应分析，研究人员创建了一种利用LLM填充的知识图谱（“KGA”）的算法，用于自动筛选潜在值得关注的诊断对，并进行了初步评估。结果：99个诊断对（摘要在此处截断）

    arXiv:2609.10421v1 Announce Type: cross  Abstract: Background: Emergency Department (ED) return visits are commonly reviewed for quality assurance, but are often limited (e.g., to revisits within 48-72 hours) to increase actionable finding yield while minimizing chart review burden. Those limitations may lead to missed quality improvement opportunities.   Methods: We conducted an exploratory, retrospective study of randomly selected ED visits to a multihospital health system having an ED revisit within 1-14 days to the same health system. Given only each visit's primary diagnosis, raters (2-3 clinicians and GPT-4 large language model [LLM]) assessed characteristics of the diagnosis pairs, including the "target": whether a pair warranted further assessment. Informed by rater response analyses, an algorithm leveraging an LLM-populated knowledge graph ("KGA") was created to automatically screen for potentially concerning pairs, then preliminarily assessed.   Results: 99 diagnosis pairs we
    
[^8]: 幸运回忆：面向大语言模型持久一致性的本体驱动记忆生命周期管理

    Fortunate Recall: Ontology-Driven Memory Lifecycle Management for Persistent Coherence in LLMs

    [https://arxiv.org/abs/2609.10413](https://arxiv.org/abs/2609.10413)

    提出Fortunate Recall（FR）可组合策略层，通过“10+1”行为本体对个人记忆进行分类，并以确定性函数形式应用类别特定的生命周期策略（差异化时间衰减、槽键取代、事件时间有效性和类别感知检索路由），在LifecycleBench和LongMemEval-S上显著超越Mem0等现有记忆系统。

    

    当前的LLM记忆系统对所有个人事实一视同仁，导致存储无限增长而检索精度不断下降。核心挑战在于生命周期管理：基于每个事实的行为类型，决定哪些记忆应当持久保留、哪些应当被替换、以及以何种速率进行替换。Fortunate Recall（FR）是一个可组合的策略层，它将个人事实分类到一个“10+1”行为本体中，并将类别特定的生命周期策略（差异化时间衰减、槽键取代、事件时间有效性以及类别感知的检索路由）作为作用于LLM提取元数据的确定性函数来应用。FR-Bank是我们独立于基础设施的实现，在新提出的包含516道时间消歧问题的LifecycleBench基准上达到76.9%的通过率，领先于Mem0、A-MEM、Memory-R1和MemoryOS（61%至70.5%），并在标准Wu et al.评判协议下于完整的LongMemEval-S上达到75.2%，因此生命周期策略…

    arXiv:2609.10413v1 Announce Type: new  Abstract: Current LLM memory systems treat all personal facts identically, so stores grow without bound while retrieval precision degrades. The core challenge is lifecycle management: which memories should persist, which should be replaced, and at what rate, conditioned on the behavioral type of each fact. Fortunate Recall (FR) is a composable policy layer that classifies personal facts into a 10+1 behavioral ontology and applies category-specific lifecycle policies (differential temporal decay, slot-key supersession, event-time validity, and category-aware retrieval routing) as deterministic functions over LLM-extracted metadata. FR-Bank, our infrastructure-independent implementation, reaches a 76.9% pass rate on LifecycleBench, a new 516-question temporal-disambiguation benchmark, ahead of Mem0, A-MEM, Memory-R1, and MemoryOS (61% to 70.5%), and 75.2% on the full LongMemEval-S under the canonical Wu et al. judge protocol, so lifecycle policies i
    
[^9]: 基础模型能否审核在线内容？评估指令驱动与示例驱动的政策操作化方法

    Can Foundation Models Moderate Online Content? Evaluating Instruction- vs. Example-Driven Policy Operationalization

    [https://arxiv.org/abs/2609.10410](https://arxiv.org/abs/2609.10410)

    本文提出包含4,000条Bluesky人工标注帖子的新基准ModerationBench，系统比较了指令驱动与示例驱动两种政策操作化范式，发现基础模型的内容审核F1分数可达Bluesky现有审核系统的近三倍（0.60 vs. 0.22）。

    

    内容审核政策日益复杂，为其一致性的操作化实施带来了关键挑战。虽然基础模型具备应对这一挑战所需的基本能力，但它们能否可靠地审核在线内容仍是一个悬而未决的问题。在本文中，我们系统地比较了视觉语言模型（VLM）指导的两种竞争性范式：一种是指令驱动方法，模型基于政策条文进行推理；另一种是示例驱动方法，模型从先前案例中进行泛化。我们将此研究建立在ModerationBench之上——这是一个包含4,000条来自Bluesky平台、经人工标注的真实帖子的新基准。我们的实验表明，基础模型能够大幅超越Bluesky已部署的审核系统，在该基准的随机帖子上将其F1分数提升了近三倍（0.60 vs. 0.22），且指令驱动和示例驱动两种范式均取得了相当的性能。

    arXiv:2609.10410v1 Announce Type: new  Abstract: The growing complexity of content moderation policies presents a critical challenge for their consistent operationalization. While foundation models possess the basic capabilities needed to confront this challenge, whether they can reliably moderate online content remains an unanswered question. In this paper, we systematically compare two competing paradigms for Vision-Language Model (VLM) guidance: an instruction-driven approach where models reason from policy precepts, and an example-driven approach where they generalize from prior precedents. We ground this investigation in ModerationBench, a new benchmark of 4,000 manually annotated, in-the-wild posts from the Bluesky platform. Our experiments reveal that foundation models can substantially outperform Bluesky's deployed moderation system, nearly tripling its $F_1$ score (0.60 vs. 0.22) on Random Posts in the benchmark, with both instruction- and example-driven paradigms achieving co
    
[^10]: MOONWALK：在动画/VFX前期制作中通过意图-证据-行动对齐来协调初级艺术家与主管评审工作流的操作

    MOONWALK: Mediating Operations with Intent-Evidence-Action Alignment Across Junior-Supervisor Review Workflows in Animation/VFX Pre-Production

    [https://arxiv.org/abs/2609.10385](https://arxiv.org/abs/2609.10385)

    该论文提出了意图-证据-行动对齐的设计框架，并实现为专业评审系统MOONWALK，帮助动画/VFX前期制作团队将松散的创作意图转化为初级艺术家可执行的明确修改任务。

    

    动画和VFX前期制作评审需要团队将松散定义的创作意图——包括简报、不断演变的规范、异构的参考素材以及口头决策——转化为初级艺术家无需反复澄清即可执行的修改。在实践中，评审标准随迭代而漂移，评审判断失去其证据基础，请求背后的推理逻辑很少能在资深与初级之间的交接中得以保留。我们提出了一个意图-证据-行动对齐的设计框架：意图被表达为共享的项目记录，评审判断锚定于有据可依的证据，经授权的决策被转化为与参考笔记直接绑定的清晰修改任务。我们将该框架实例化为MOONWALK，一个专业的前期制作评审系统，包含共享意图记录、参考/规范锚定、结构化的进行中作品对比以及主管授权的行动规划。

    arXiv:2609.10385v1 Announce Type: cross  Abstract: Animation and VFX pre-production review requires teams to translate loosely specified creative intent--briefs, evolving specifications, heterogeneous references, and verbal decisions--into revisions that junior artists can execute without repeated clarification. In practice, criteria drift across iterations, review judgments lose their evidential basis, and the reasoning behind a request rarely survives the senior-junior handoff. We contribute a design framework for intent-evidence-action alignment: intent is articulated into a shared project record, judgments are anchored to grounded evidence, and authorized decisions are converted into clear revision tasks tied directly to reference notes. We instantiate this framework in MOONWALK, a professional pre-production review system comprising a shared intent record, reference/specification anchoring, structured work-in-progress comparison, and supervisor-authorized action planning. In this 
    
[^11]: PACE：面向QoE高效检索增强对话服务的感知延迟感知级联服务路由与填充器控制

    PACE: Perceived-Latency-Aware Cascading Service Routing and Filler Control for QoE-Efficient Retrieval-Augmented Dialogue Serving

    [https://arxiv.org/abs/2609.10372](https://arxiv.org/abs/2609.10372)

    PACE框架将感知首次响应时间（PTFR）作为QoE目标，通过负载自适应级联路由、路径-填充器联合控制和波动感知缓存准入三种机制，在人形机器人对话服务中显著降低感知延迟并保证回答质量与新鲜度。

    

    我们提出了PACE，一个检索增强对话服务框架，它将感知首次响应时间（PTFR）形式化为QoE目标，并在质量/成本约束下将其最小化。与以往关于级联路由、语义缓存或自适应检索的工作不同，PACE联合控制由哪个答案源构成响应，以及用什么内容填充等待窗口。该框架部署在人形机器人销售服务上，结合了三种机制：负载自适应级联路由器、路径-填充器联合控制器和波动感知缓存准入。在75k条CarQA请求上，级联机制将P95处的纯LLM PTFR减半（c16并发下为0.29秒对比0.53秒）。自适应控制器达到0.41秒的P95，在高负载且质量相当的情况下，性能优于RAG达2.4倍。填充器控制器将调用次数减少了94%，且零冲突。波动感知准入将过时答案的比例从86%降低到0%。门控规则确保控制器性能永远不会差于基线。

    arXiv:2609.10372v2 Announce Type: cross  Abstract: We present the PACE, a framework for retrieval-augmented dialogue serving that formalizes Perceived Time-to-First-Response (PTFR) as a QoE objective and minimizes it under quality/cost constraints. Unlike prior work on cascaded routing, semantic caching, or adaptive retrieval, PACE jointly controls which answer source composes the response and what fills the waiting window. Deployed on a humanoid-robot sales service, it combines three mechanisms: a load-adaptive cascading router, a joint path-filler controller, and volatility-aware cache admission. On 75k CarQA requests, the cascade halves pure-LLM PTFR at P95 (0.29 vs 0.53s at c16). The adaptive controller reaches 0.41s P95, outperforming RAG by 2.4 times at high load with equal quality. The filler controller cuts calls by 94% with zero conflict. Volatility-aware admission reduces stale answers from 86% to 0%. A gating rule ensures the controller never worse than the baseline, with ex
    
[^12]: OmniMed-FL：一个用于临床诊断的鲁棒多模态联邦学习框架

    OmniMed-FL: A Robust Multimodal Federated Learning Framework for Clinical Diagnosis

    [https://arxiv.org/abs/2609.10364](https://arxiv.org/abs/2609.10364)

    提出了OmniMed-FL多模态联邦学习框架，通过系统性评估多种融合策略、初始化方法和缺失文本填补规则，在满足隐私法规的前提下实现了医学影像与文本病历的安全融合，用于五类临床病症的分类诊断。

    

    临床诊断中通常需要同时对医学影像和患者病历进行综合评估。然而，标准机器学习算法无法将这些数据类型结合在一起分析。同时，出于对HIPAA和GDPR法规的合规要求，敏感患者数据的集中式聚合受到限制。这在跨远程网络安全融合视觉与文本上下文方面留下了关键空白。因此，我们提出了OmniMed-FL，这是一项针对五类临床病症分类（正常、肺炎、COVID-19、胸腔积液、心脏肥大）的多模态联邦学习受控系统研究。我们的代理语料库将3,000张公开胸部X光片与3,000份按类别条件生成的合成病历进行配对，匹配基于类别而非患者个体。该框架在3至20个医院客户端的非IID Dirichlet数据划分下，对八种融合策略、三种初始化方法、四种缺失文本填补规则以及匹配的联邦基线进行了系统性基准测试。

    arXiv:2609.10364v1 Announce Type: cross  Abstract: Simultaneous assessment of medical imaging and patient records is often required in clinical diagnosis. However, standard machine learning algorithms cannot analyze these data types together. Meanwhile, compliance with HIPAA and GDPR can constrain centralized aggregation of sensitive patient data. This leaves a crucial void of secure fusion of visual and textual context across distant networks. Thus, we present OmniMed-FL, a controlled systems study of multimodal federated learning for five-class clinical condition classification (Normal, Pneumonia, COVID-19, Pleural Effusion, Cardiomegaly). Our proxy corpus pairs 3,000 public chest radiographs with 3,000 class-conditioned synthetic notes, matched by class, not by patient. The framework benchmarks eight fusion strategies, three initializations, four missing-text imputation rules, and matched federated baselines under non-IID Dirichlet partitioning across 3 to 20 hospital clients. As al
    
[^13]: 网络-金融传染：模拟AI供应商遭入侵在银行系统中的传播

    Cyber-Financial Contagion: Modeling the Propagation of an AI Vendor Compromise Through the Banking System

    [https://arxiv.org/abs/2609.10350](https://arxiv.org/abs/2609.10350)

    本文提出CFC-Prop随机传染病-清算模型，通过四层异构网络模拟单个AI供应商遭入侵如何沿银行系统传播，最终引发类似传统银行危机的系统性厚尾损失。

    

    银行系统如今依赖于一小批共享的人工智能供应商，用于欺诈筛查、信贷决策、反洗钱分流、客户分析以及内部决策支持。本文研究了其中一个供应商遭受入侵后，如何沿着运营、信息和金融关联的链条传播，最终引发从外部看起来类似传统银行危机的损失。我们构建了一个耦合AI供应商、金融机构、银行间风险敞口和客户账户的四层异构网络，并提出了CFC-Prop，一个运行在该网络上的随机传染病-清算模型。在一个包含60个供应商、220家银行、约2500条供应商-银行服务连接边和1400个银行间风险敞口的合成数据集上，CFC-Prop再现了与既有网络金融证据相一致的厚尾损失分布，以及对补丁延迟时间的尖锐依赖关系。

    arXiv:2609.10350v1 Announce Type: new  Abstract: The banking system now depends on a small set of shared artificial intelligence vendors for fraud screening, credit decisioning, anti-money-laundering triage, customer analytics, and internal decision support. This paper studies how a compromise inside one of those vendors can propagate along a chain of operational, informational, and financial linkages until it triggers losses that look, from the outside, like a classical banking crisis. We build a four-layer heterogeneous network that couples AI vendors, financial institutions, interbank exposures, and customer accounts, and we propose CFC-Prop, a stochastic epidemic-and-clearing model that runs on that network. On a synthetic dataset with 60 vendors, 220 banks, roughly 2,500 vendor-bank service edges, and 1,400 interbank exposures, CFC-Prop reproduces the heavy-tailed loss distributions and the sharp dependence on patch latency that are consistent with prior cyber-financial evidence. 
    
[^14]: 超越一刀切：面向多模态大语言模型中视觉Token剪枝的样本自适应策略路由

    Beyond One-Size-Fits-All: Sample-Adaptive Strategy Routing for Vision Token Pruning in MLLMs

    [https://arxiv.org/abs/2609.10346](https://arxiv.org/abs/2609.10346)

    提出VIP-Router，一个轻量级路由器，根据每个输入样本的低成本视觉和文本特征自适应选择最优的视觉token剪枝策略，突破了传统固定剪枝策略“一刀切”的局限。

    

    多模态大语言模型（MLLMs）每张图像需要处理成百上千个视觉token，导致高昂的推理成本。虽然现有的视觉token剪枝方法能够缓解这一开销，但它们都隐含地假设单一固定的剪枝策略可以统一应用于所有输入。我们的分析进一步揭示，按平均基准准确率对剪枝方法进行排名掩盖了显著的样本间互补性：尽管平均表现最优的策略在整体上表现出色，但其他策略在相当一部分单独样本上被证明更为优越。为了利用这种多样性，我们提出了VIP-Router，一个轻量级的视觉剪枝路由器，它能够在指定的剪枝级别下自适应地选择预测最适合每个输入的剪枝策略。基于低成本的视觉和文本特征作为条件，VIP-Router识别出最合适的候选策略，同时保留全token推理作为可选选项。

    arXiv:2609.10346v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) process hundreds or thousands of visual tokens per image, incurring prohibitive inference costs. While existing vision token pruning methods mitigate this overhead, they implicitly assume that a single fixed pruning strategy can be applied uniformly across all inputs. Our analysis further reveals that ranking pruning methods by average benchmark accuracy conceals substantial sample-wise complementarity: although the average-best strategy excels overall, alternative strategies prove superior on a significant fraction of individual samples. To harness this diversity, we propose VIP-Router, a lightweight VIsion Pruning Router that adaptively selects the pruning strategy predicted to be best suited to each input at a specified pruning level. Conditioned on low-cost visual and textual features, VIP-Router identifies the most suitable candidate strategy while retaining full-token inference as an optio
    
[^15]: 从符号感知到逻辑推演：一个引导语言模型进行几何推理的框架

    From Symbolic Perception to Logical Deduction: A Framework for Guiding Language Models in Geometric Reasoning

    [https://arxiv.org/abs/2609.10335](https://arxiv.org/abs/2609.10335)

    该论文提出一个将几何图形解析为符号形式并进行形式化逻辑推演的框架，使纯大语言模型在几何推理上达到与最先进多模态模型相当的性能，同时减少幻觉并提升推理的可解释性。

    

    平面几何仍然是人工智能领域的一个重大挑战，它需要视觉感知与数学推理的融合。虽然大型多模态模型（LMMs）能够自然地处理视觉-语言输入，但它们通常计算开销大且缺乏透明度。我们证明了纯大型语言模型（LLM）在配备专门模块的情况下，可以在复杂几何问题上与最先进的多模态模型相媲美。我们的框架集成了几何视觉解析器（将图形转换为符号形式）与符号求解器（执行形式化推演），从而减少幻觉并促进可解释的推理。为了进行严格的评估，我们整理了来自2025年中国中考的具有挑战性问题的基准测试，确保了数据的新颖性并考察更深层次的推演能力。实验表明，我们的方法达到了与Gemini 2.5 Pro相当的性能，同时提供更清晰、更符合人类（思维方式的推理过程）……

    arXiv:2609.10335v1 Announce Type: cross  Abstract: Plane geometry remains a significant challenge in AI, requiring the integration of visual perception and mathematical reasoning. While Large Multimodal Models (LMMs) naturally handle visuo-linguistic inputs, they are often computationally intensive and opaque. We demonstrate that a pure Large Language Model (LLM), when equipped with specialized modules, can rival state-of-the-art LMMs on complex geometry problems. Our framework integrates a Geometric Vision Parser, which translates diagrams into symbolic form, with a Symbolic Solver that performs formal deductions, thereby mitigating hallucinations and promoting interpretable reasoning. To enable rigorous evaluation, we curate a benchmark of challenging problems from the 2025 Chinese Zhongkao examinations, ensuring data novelty and testing deeper deductive skills. Experiments demonstrate that our approach achieves performance comparable to Gemini 2.5 Pro while delivering clearer, human
    
[^16]: TRACE：利用合成奖励训练因果探索推理智能体

    TRACE: Training Reasoning Agents for Causal Exploration with Synthesized Rewards

    [https://arxiv.org/abs/2609.10315](https://arxiv.org/abs/2609.10315)

    提出TRACE框架，通过在受控模拟器中注入合成的隐藏干预来提供真值标签和客观奖励，从而训练智能体在噪声、混杂和分散证据下完成复杂数据的因果诊断推理。

    

    基于可验证奖励的强化学习（RLVR）已在数学和代码等领域推动了语言模型推理的发展，因为这些领域的客观答案验证成本低廉。而对复杂数据的诊断推理则缺乏这一优势：确定异常的真正原因往往需要昂贵的专家调查，且事后可能仍存在歧义。我们探讨这种验证的不对称性能否通过人为设计来弥补。我们采样一个干预措施，将其注入受控模拟器中，并生成它所产生的观测结果。隐藏的干预提供了真值标签和客观奖励，而智能体仍需调查含噪、混杂且分散的证据。我们在TRACE中实例化了这一方法——一个包含12种根因和细粒度细分归因的数字广告诊断环境。智能体在每个回合中使用Python和SQL开展调查，并且必须同时识别出根因以及……

    arXiv:2609.10315v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) has advanced language-model reasoning in domains such as mathematics and code, where objective answers are inexpensive to check. Diagnostic reasoning over complex data lacks this advantage: establishing the true cause of an anomaly often requires costly expert investigation and may remain ambiguous after the fact. We ask whether this asymmetry of verification can instead be engineered. We sample an intervention, inject it into a controlled simulator, and generate the observations it would produce. The hidden intervention provides an oracle label and objective reward, while the agent must still investigate noisy, confounded, and distributed evidence.   We instantiate this approach in TRACE, a digital-advertising diagnostic environment with 12 root causes and fine-grained segment attribution. Agents investigate each episode using Python and SQL and must identify both the root cause and,
    
[^17]: 一次循环，双重收益：主动学习能“免费”赢得彩票吗？

    One Loop, Two Gains: Can Active Learning win the Lottery for Free?

    [https://arxiv.org/abs/2609.10311](https://arxiv.org/abs/2609.10311)

    本文提出 Improve & Prune（I&P）方法，将幅度剪枝无缝嵌入基于池的主动学习已有的迭代重训练循环中，在不增加额外计算开销的情况下同时获得主动学习和稀疏子网络（中奖彩票）发现的双重收益。

    

    彩票假设（lottery ticket hypothesis）断言存在“中奖彩票”：稀疏子网络在从原始初始化独立训练后，能够达到完整稠密网络的准确率。发现此类中奖彩票的主流方法是迭代幅度剪枝，它在多次循环中交替进行剪枝和从零开始的完整重训练，直至收敛。类似地，深度主动学习在每一轮数据获取并获得新标签后，也会从零开始重新训练模型。尽管这两种范式都依赖于带有巨大计算开销的迭代重训练，但它们此前一直被分开研究。我们观察到，基于池的主动学习（pool-based active learning）中固有的迭代训练循环，恰好提供了迭代幅度剪枝所需的精确计算结构，并据此提出 Improve & Prune（I&P）方法，该方法将幅度剪枝整合到主动学习的每一次重训练循环中，几乎不带来额外开销。

    arXiv:2609.10311v1 Announce Type: cross  Abstract: The lottery ticket hypothesis posits the existence of winning tickets: sparse subnetworks that, when trained in isolation from their original initialization, match the accuracy of the full dense network. The predominant method for discovering such tickets, iterative magnitude pruning, alternates pruning with full retraining from scratch until convergence over many cycles. Similarly, deep active learning also retrains a model from scratch after each acquisition round as new labels become available. Despite this shared reliance on iterative retraining with a substantial computational overhead, the two paradigms have been studied separately. We observe that the iterative training loop inherent to pool-based active learning already provides the exact computational structure that iterative magnitude pruning exploits, and propose Improve & Prune (I&P), a method that integrates magnitude pruning into each active learning retraining cycle at p
    
[^18]: RiLM：通过测地线解码实现参数高效的语言建模

    RiLM: Parameter-Efficient Language Modeling via Geodesic Decoding

    [https://arxiv.org/abs/2609.10305](https://arxiv.org/abs/2609.10305)

    提出RiLM框架，完全移除传统输出矩阵，通过在黎曼流形上计算当前状态与词表嵌入之间的测地线距离平方来直接解码下一个词元概率，其双曲版本HypRiLM在约29万参数下于WikiText-2上达到54.2的验证困惑度，显著优于平坦版本及同规模的LSTM、Transformer和SSM基线。

    

    百万参数以下的语言模型对边缘部署、领域适配和可复现研究非常重要，然而在嵌入维度 d = 128 下，两层 LSTM 或 Transformer 仍将大约三分之一的容量耗费在位于 R^(d x |V|) 中的输出矩阵 W_out 上。我们提出黎曼语言模型，它完全移除了该输出层：上下文展开为黎曼流形上的轨迹，下一个词元的概率由当前状态与词表嵌入之间的测地线距离平方产生。同一个嵌入映射同时服务于输入和输出——解码即几何。我们在平坦空间 R^d（Flat RiLM）和庞加莱球 H^d（HypRiLM）上实例化了该框架，并使用共享的 MLP 组合映射 phi（约29万参数，d = 128，|V| = 2000）。在 WikiText-2 上五个随机种子的实验中，HypRiLM 达到 54.2 ± 0.2 的验证困惑度，而 Flat RiLM 为 87.6 ± 0.6；同等参数规模的 LSTM、Transformer 和 SSM 对照模型仍然处于（原文截断）。

    arXiv:2609.10305v1 Announce Type: new  Abstract: Language models under one million parameters matter for edge deployment, domain adaptation, and reproducible research, yet a two-layer LSTM or Transformer at embedding width d = 128 still spends roughly one third of its capacity on the output matrix W_out in R^(d x |V|). We propose Riemannian Language Models (RiLM), which remove that layer entirely: context unfolds as a trajectory on a Riemannian manifold, and next-token probabilities arise from squared geodesic distance between the current state and vocabulary embeddings. The same embedding map serves input and output -- decoding is geometry. We instantiate the framework on flat R^d (Flat RiLM) and the Poincare ball H^d (HypRiLM) with a shared MLP composition map phi (~290k parameters, d = 128, |V| = 2000). Across five seeds on WikiText-2, HypRiLM reaches 54.2 +/- 0.2 validation perplexity versus 87.6 +/- 0.6 for Flat RiLM; tied and matched LSTM, Transformer, and SSM controls remain at 
    
[^19]: 面向OT系统的入侵响应策略学习

    Learning Intrusion Response Strategies for OT Systems

    [https://arxiv.org/abs/2609.10298](https://arxiv.org/abs/2609.10298)

    本文使用POMDP框架对OT系统入侵响应场景进行形式化建模，并结合基于流量测量的部分可观测性，提出基于PPO的学习方法自动生成入侵响应策略，在仿真OT系统中验证了对多种MITRE攻击的有效性。

    

    针对监控和控制工业过程的运营技术（OT）系统的网络攻击，对社会基础服务构成日益增长的威胁。因此，开发自动化的入侵响应策略变得非常重要。在本文中，我们使用POMDP（部分可观测马尔可夫决策过程）框架对一个OT入侵响应用例进行了形式化建模。该模型包含一个基于流量测量的真实部分可观测性模型。这种方法使我们能够开发出易于求解的、基于学习的自动化入侵响应解决方法，该方法基于PPO（近端策略优化）算法。我们在一个仿真的OT系统上评估了所获得的响应策略，发现它们能够有效应对所研究用例中的多种MITRE攻击类型。

    arXiv:2609.10298v1 Announce Type: cross  Abstract: Cyberattacks against Operational Technology (OT) systems, which monitor and control industrial processes, pose an increasing threat to essential societal services. For this reason, developing automated intrusion response strategies is highly important. In this paper, we present a formal model of an OT intrusion response use case using the POMDP framework. It includes a realistic model of partial observability that is based on traffic measurements. This approach allows us to develop tractable, learning-based solution methods for automated intrusion response, which are based on PPO. We evaluate the obtained response strategies on an emulated OT system and find that they are effective against several types of MITRE attacks for the studied use case.
    
[^20]: GANDR：面向可验证法律答案生成的论断审计

    GANDR: Claim Auditing for Verifiable Legal Answer Generation

    [https://arxiv.org/abs/2609.10293](https://arxiv.org/abs/2609.10293)

    GANDR是一个双智能体系统，由起草器生成结构化法律答案、批评者逐条对照引用来源审计论断，并配合要求每条引用必须命中检索结果的严格正确性标准，从而实现可逐条验证的法律答案生成。

    

    在法律实践等高风险领域，语言模型生成的答案只有在读者能够对照系统所引用的来源逐条验证每个论断时才有用。当前的有据生成流水线将答案作为一个整体进行评分，因此一个正确的结论可能建立在捏造的或匹配松散的引用之上，却仍能获得高分。弥合这一差距既需要一个为逐条论断验证而构建的系统，也需要一种能够衡量它的评估方法。我们提出了GANDR（Grounded ANswer DRafter，有据答案起草器），这是一个双智能体系统：起草器以结构化的法律推理格式撰写答案，而一个独立的批评者——拥有与人类验证者相同的视角——针对所引用的来源审计每个论断，并在每一轮输出逐条论断的审计轨迹。我们为其配备了一项严格的正确性标准，要求每条引用都能对应到检索器返回的某段文本。在一个包含185个条目的法律基准上，所有六个系统共享同一个骨干模型和一个检索源……（摘要在此处被截断）

    arXiv:2609.10293v1 Announce Type: new  Abstract: In high-stakes domains such as legal practice, a language-model answer is only useful to the extent that a reader can verify each claim against the source the system cites. Current grounded-generation pipelines score the answer as a whole, so a correct conclusion can rest on fabricated or loosely matched citations and still score well. Closing this gap requires both a system built for per-claim verification and an evaluation that measures it. We introduce GANDR (Grounded ANswer DRafter), a two-agent system in which a Drafter writes an answer in a structured legal-reasoning format and a separate Critic, with the same view as a human verifier, audits each claim against its cited source and emits a per-claim audit trace on every round. We pair it with a strict correctness criterion requiring every citation to resolve to a passage the retriever returned. On a 185-item legal benchmark where all six systems share one backbone, one retrieval su
    
[^21]: 智能体应该遗忘什么？区分存储内容与使用内容

    What Should an Agent Forget? Separating What Is Stored from What Is Used

    [https://arxiv.org/abs/2609.10263](https://arxiv.org/abs/2609.10263)

    本文提出无需训练的RD-Forget框架，通过将智能体存储的内容与使用的内容分离，结合查询条件记忆视图、同槽替换链接和率失真优化，使过时信息不会误导当前回答，同时保留历史查询所需的证据。

    

    持久化的语言智能体需要存储的经验在时间跨度上保持可用，而每个答案都需要适合特定问题的证据。一个已被取代的事实可能会误导针对当前状态的答案，但对于历史查询而言却至关重要。我们提出了RD-Forget，这是一个无需训练的框架，它将智能体存储的内容与其使用的内容分离开来。一个被保留的源归档保存观察结果，而一个基于查询条件的记忆视图控制这些观察结果对当前答案的影响。一个冻结的语言模型策展器提取相关证据，将事实分组到语义槽中，并保留多跳推理所需的关系。同槽替换链接在当前状态的上下文中抑制已被取代的值，而意图感知检索则使早期证据重新可用。率失真形式化方法指导在内存预算内构建回答时视图。实验涵盖对话记忆、知识更新等场景。

    arXiv:2609.10263v1 Announce Type: new  Abstract: Persistent language agents need stored experience to remain available across time, while each answer requires evidence suited to a particular question. A superseded fact can mislead a current-state answer and still be essential for a historical query. We present RD-Forget, a training-free framework that separates what an agent stores from what it uses. A retained source archive preserves observations, and a query-conditioned memory view controls their influence on the current answer. A frozen language-model curator extracts relevant evidence, groups facts into semantic slots, and preserves the relations needed for multi-hop reasoning. Same-slot replacement links suppress superseded values in current-state contexts, while intent-aware retrieval makes earlier evidence eligible again. A rate-distortion formulation guides construction of the answer-time view within a memory budget. Experiments span conversational memory, knowledge updating, 
    
[^22]: DiSCo：一个分布优先的引导与文化先验评估框架，用于测量大语言模型中的文化偏好偏差

    DiSCo: A Distribution-First Steering and Cultural Prior Evaluation Framework for Measuring Cultural Preference Bias in LLMs

    [https://arxiv.org/abs/2609.10253](https://arxiv.org/abs/2609.10253)

    DiSCo是一个分布优先的评估框架，通过隔离默认文化先验并利用四级上下文梯度测试可引导性，发现大语言模型的文化偏好先验严重集中在英美文化上（合计约占35%）。

    

    大语言模型（LLMs）越来越多地被部署于全球范围内使用的智能助手中，然而它们在根植于文化的日常情境中所做的默认选择，可能会系统性地偏向某些文化，从而影响本地化效果、用户信任以及公平的行为表现。现有的文化基准测试以单一“正确”答案来评估准确率，这使得当多个根植于文化的回答都同样有效时，难以刻画LLM的文化偏好先验；同时，这些基准还将默认偏好与上下文驱动的适应性调整混为一谈。我们提出了DiSCo，这是一个分布优先的强制选择评估框架，能够隔离默认文化先验，并通过四级上下文梯度（C0–C3）来测试模型的可引导性。使用从BLEnD衍生、涵盖12种文化的DiSCo-Bench（304个条目），我们评估了六个多样化的指令微调LLM。结果显示，默认文化先验高度集中，其中英国和美国合计占据了约35%的所有选择。

    arXiv:2609.10253v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed in globally used assistants, yet their default choices in culturally grounded everyday situations can systematically favour some cultures over others, affecting localisation, user trust, and equitable behaviour. Existing cultural benchmarks evaluate accuracy against a single "correct" answer, making it difficult to characterise an LLM's cultural preference prior when multiple culturally grounded responses are all valid; they also conflate default preferences with context-driven adaptation. We propose DiSCo, a distribution-first forced-choice evaluation framework that isolates default cultural priors and tests steerability via a four-level context gradient (C0--C3). Using DiSCo-Bench (304 items) derived from BLEnD spanning 12 cultures, we evaluate six diverse instruction-tuned LLMs. Default priors are heavily concentrated, with UK and US together absorbing approximately 35\% of all se
    
[^23]: A-JIT：智能体式即时软件构建

    A-JIT: Agentic Just-In-Time Software Construction

    [https://arxiv.org/abs/2609.10248](https://arxiv.org/abs/2609.10248)

    A-JIT提出了一种新的软件构建范式，通过在应用中嵌入AI智能体，使其摆脱静态二进制的限制，能够根据用户行为动态构建缺失实现、即时生成新功能并持续自适应演化。

    

    传统软件交付采用静态范式：代码在执行前预先构建，并作为固定制品部署。我们提出了智能体式即时软件构建，这是一种用动态软件系统取代静态二进制文件的新范式，此类系统能够持续演化以满足不断变化的需求。在A-JIT中，应用程序是一个集成化组件，由代码、运行时框架以及一个持续观察系统使用情况和实时执行轨迹的嵌入式AI智能体组成。正如传统的JIT编译器根据运行时执行路径对机器码进行专门优化一样，A-JIT根据最终用户的具体需求对软件逻辑、工作流程和工具接口进行专门定制。通过将代码合成直接集成到应用程序的日常生命周期中，A-JIT使应用程序能够动态构建缺失的实现、即时生成新功能，并持续适应最终用户的行为。

    arXiv:2609.10248v1 Announce Type: new  Abstract: Traditional software delivery assumes a static paradigm: code is constructed prior to execution and deployed as a fixed artifact. We present Agentic Just-In-Time Software Construction (A-JIT), a paradigm that replaces static binaries with dynamic, software systems that can perpetually evolve to meet changing demands. In A-JIT, an application is an integrated assembly comprising code, a runtime harness, and an embedded AI agent that continuously observes system usage and live execution traces. Much like a traditional JIT compiler specializes machine code to runtime execution paths, A-JIT specializes software logic, workflows, and tool interfaces to meet the specific needs of the end-user. By integrating synthesis directly into the ambient application lifecycle, A-JIT enables applications to dynamically construct missing implementations, generate new capabilities on the fly, and continuously adapt to end-user behavior. We demonstrate how t
    
[^24]: LiteRAG：低成本高效的基于图的检索增强生成

    LiteRAG: Cost-Efficient Graph-Based Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.10239](https://arxiv.org/abs/2609.10239)

    LiteRAG通过查询条件算法探索和推理链上下文构建取代昂贵的检索时LLM控制，在多跳问答质量上达到最优，同时将查询延迟降低100倍以上、成本降低99%以上、token使用量减少约14倍。

    

    基于图的检索可以提升多跳问答的效果，但现有方法往往在查询时产生高昂成本，并生成分散且过于庞大的上下文，从而降低生成效率。我们提出了LiteRAG，这是一种基于图的检索方法，它用基于查询条件的算法探索和推理链上下文构建，取代了昂贵的检索时LLM控制。在DistComp（一个针对分布式系统论文的多跳检索基准）上，LiteRAG在所评估的方法中获得了最高的整体质量得分（0.798），同时与GraphRAG Global和DRIFT相比，每次查询的延迟降低了100倍以上，成本降低了99%以上。在UltraDomain上，它在整体质量上与LinearRAG相当，但使用的token数量减少了约14倍。消融实验表明，LiteRAG的查询自适应阈值机制和社区感知的中心节点惩罚是其token效率提升的主要驱动因素。

    arXiv:2609.10239v1 Announce Type: cross  Abstract: Graph-based retrieval can improve multi-hop question answering, but existing approaches often incur high query-time costs and produce diffuse, oversized contexts that reduce generation efficiency. We present LiteRAG, a graph-based retrieval method that replaces expensive retrieval-time LLM control with query-conditioned algorithmic exploration and reasoning-chain context construction. On DistComp, a benchmark for multi-hop retrieval over distributed-systems papers, LiteRAG attains the highest overall quality among the evaluated methods (0.798) while reducing per-query latency by over 100$\times$ and cost by over 99% relative to GraphRAG Global and DRIFT. On UltraDomain, it matches LinearRAG on overall quality while using about 14$\times$ fewer tokens. An ablation study indicates that LiteRAG's query-adaptive thresholding and community-aware hub penalization are the main drivers of its token-efficiency gains.
    
[^25]: 基于策略引导嵌入搜索的层次化与置换不变特征变换学习

    Hierarchical and Permutation-Invariant Feature Transformation Learning via Policy-Guided Embedding Search

    [https://arxiv.org/abs/2609.10225](https://arxiv.org/abs/2609.10225)

    该论文提出了一个特征变换学习框架，利用置换不变的层次化模块捕捉特征、操作与抽象层次间的交互，并通过策略引导的嵌入搜索适应非凸变换空间，克服了现有生成式方法的三大局限。

    

    特征变换通过从原始特征构建有信息量的抽象表示来提升表格数据的预测性能。近期的生成式方法将变换知识编码到连续嵌入空间中以高效探索候选策略，但面临三个关键局限：(1) 忽视了低层特征、操作和高层抽象之间的层次关系；(2) 在本质上置换不变的变换序列上强加了顺序敏感的嵌入，从而引入系统性偏差；(3) 依赖基于梯度的搜索，这不适合非凸的变换空间。我们提出了一个包含两个互补组件的框架。首先，一个置换不变的层次化模块捕捉特征、操作和抽象层次之间的交互，并通过自注意力池化机制将语义等价的结构映射到一致的嵌入表示。

    arXiv:2609.10225v1 Announce Type: cross  Abstract: Feature transformation improves predictive performance on tabular data by constructing informative abstractions from raw features. Recent generative approaches encode transformation knowledge into continuous embedding spaces for efficient exploration of candidate strategies, but face three key limitations: (1) overlooking hierarchical relationships between low-level features, operations, and high-level abstractions; (2) enforcing order-sensitive embeddings on inherently permutation-invariant transformation sequences, thereby introducing systematic bias; and (3) relying on gradient-based search, which is ill-suited to non-convex transformation spaces. We propose a framework with two complementary components. First, a permutation-invariant hierarchical module captures interactions across features, operations, and abstraction levels, with a self-attention pooling mechanism that maps semantically equivalent structures to consistent embeddi
    
[^26]: 为什么采样可以枚举的内容？面向基因组工具选择的精确策略优化

    Why Sample What You Can Enumerate? Exact Policy Optimization for Genomic Tool Selection

    [https://arxiv.org/abs/2609.10221](https://arxiv.org/abs/2609.10221)

    该论文揭示了在工具子集空间可完全枚举的基因组推理等科学领域中，GRPO等基于采样的强化学习方法存在结构性缺陷（训练越成功奖励信号越稀缺），并提出FGPO（全组策略优化），通过对所有工具子集进行精确枚举评分来替代采样估计。

    

    在冻结的推理器上进行强化学习已成为教导策略调用哪些外部工具的常见方法。我们证明，在完整的工具子集空间可枚举的专业科学环境中，这种方案存在结构性不匹配。在那里，一小组反复出现的计算能力即可覆盖整个领域，因此工具子集的空间虽然是组合性的，但足够小以至于可以枚举，而GRPO仍然从少量采样的rollouts中估计动作期望。更糟糕的是，随着训练的成功，这种近似反而会退化：当策略集中于偏好的子集时会重复采样它们，采样的奖励发生冲突，组归一化的优势随之消失。在基因组推理任务中，未产生奖励信号的问题比例从均匀参考策略下的0.2%上升到GRPO训练后的20.8%。作为补救措施，我们引入了FGPO（全组策略优化），它（1）对每个工具子集进行精确评分……（原文摘要被截断）

    arXiv:2609.10221v2 Announce Type: new  Abstract: Reinforcement learning over a frozen reasoner has become a common recipe for teaching a policy which external tools to invoke. We show that this recipe becomes structurally mismatched in specialist scientific settings where the complete tool-subset space is enumerable. There, a small set of recurring computational capabilities covers the domain, so the space of tool subsets is combinatorial yet small enough to enumerate, and GRPO still estimates an action expectation from a handful of sampled rollouts. Worse, the approximation degrades as training succeeds: as the policy concentrates on preferred subsets it resamples them, sampled rewards collide, and the group-normalized advantage vanishes. On genomic reasoning the fraction of questions yielding no reward signal rises from 0.2% under a uniform reference policy to 20.8% after GRPO training. As a remedy, we introduce FGPO (Full-Group Policy Optimization), which (1) scores every tool subse
    
[^27]: AI智能体能否跨越权限边界交付可验证的全网级结果？

    Can AI Agents Deliver Verifiable Network-Wide Outcomes Across Authority Boundaries?

    [https://arxiv.org/abs/2609.10181](https://arxiv.org/abs/2609.10181)

    该论文探讨了当多个具有不同权限范围的AI智能体跨越管理域协作进行网络自动化时，需要一个可信保障层来汇总碎片化的证据，从而验证配置变更确实达成了全网范围的预期结果。

    

    AI智能体日益深入地参与网络自动化，它们可以通过受控的操作接口发起配置变更并评估由此产生的状态。然而，运营中的网络通常跨越众多设备和管理域。实现运营商的意图需要协调具有不同权限范围的智能体，这些权限范围定义了它们可以访问的资源、可以调用的操作以及可以观察的网络状态。这种权限划分虽然限制了错误操作的影响范围，但也使评估全网结果所需的证据变得碎片化。某个智能体提出的配置操作成功执行，并不能证明远程设备按预期做出了响应，也不能证明路由变更传播到了所需的设备。此外，一条有效的观测信息在后续变更发生后也可能变得过时。在协调操作被宣告完成之前，需要一个可信的保障层来汇总各方的观测证据（摘要内容在此处截断）。

    arXiv:2609.10181v1 Announce Type: cross  Abstract: AI agents are increasingly involved in network automation, where they can initiate configuration changes through mediated operational interfaces and assess the resulting state. Nonetheless, operational networks usually span many devices and administrative domains. Realizing an operator's intent requires coordinating agents with distinct authority scopes that define the resources they can access, the operations they can invoke, and the network state they can observe. This division limits the blast radius of an erroneous action but fragments the evidence needed to assess the network-wide outcome. Successful execution of a configuration action proposed by one agent does not establish that remote devices responded as intended or that routing changes reached the required devices. A valid observation may also become stale after a subsequent change. Before the coordinated operation can be declared complete, a trusted assurance layer must coll
    
[^28]: 超越表面模仿：多模态上下文学习中用于推理路径对齐的对比建模

    Beyond Surface Imitation: Contrastive Modeling for Reasoning Path Alignment in Multimodal In-Context Learning

    [https://arxiv.org/abs/2609.10177](https://arxiv.org/abs/2609.10177)

    该论文提出了一种将对比示例建模与多模态大语言模型自我完善能力相结合的新型多模态上下文学习框架，通过在同一输入下显式对比次优响应与更优响应并附以推理路径，使模型超越表面模仿，真正实现响应与所需推理路径的对齐。

    

    上下文学习在多模态大语言模型中被广泛应用，并在各类多模态任务中取得了优异的性能。然而，现有的多模态ICL方法往往依赖于对上下文示例的表面级模仿，使得MLLM难以让其响应与给定多模态输入所需的推理路径对齐。这一局限性在复杂的多模态任务中尤为明显，从而限制了MLLM性能的进一步提升。为解决这一问题，我们提出了一种新的多模态ICL框架，该框架将对比示例建模与MLLM的自我完善能力相结合。具体而言，我们的框架通过在相同输入下显式对比一个次优响应与一个更优响应，并提供一条揭示响应应如何被完善的推理路径，来重新构建每个示例。这种对比式的构建方式使得重新（摘要在此处截断）……

    arXiv:2609.10177v1 Announce Type: new  Abstract: In-context learning (ICL) is widely used in multimodal large language models (MLLMs) and achieves strong performance across a wide range of multimodal tasks. However, existing multimodal ICL methods often rely on surface level imitation of in-context demonstrations, making it difficult for MLLMs to align their responses with the reasoning path required by the given multimodal input. This limitation becomes more pronounced in complex multimodal tasks, thereby restricting further improvements in MLLM performance. To address this issue, we propose a new multimodal ICL framework that combines contrastive demonstration modeling with the self-refinement capability of MLLMs. Specifically, our framework reformulates each demonstration by explicitly contrasting a suboptimal response with a better response under the same input, together with a reasoning path that reveals how the response should be refined. This contrastive formulation makes the re
    
[^29]: 内核管理的共享内存实现系统级个性化

    Kernel-Managed Shared Memory for System-Wide Personalization

    [https://arxiv.org/abs/2609.10144](https://arxiv.org/abs/2609.10144)

    提出内核管理的共享内存这一系统级抽象，由智能体系统内核统一负责多智能体间的记忆检索、隐私管控和提示注入，相比非管理的记忆后端显著提升了系统级个性化效果。

    

    当AI系统能够适应使用它们的用户时，其效用会显著提升，但在多智能体系统中，一个智能体学到的有用上下文往往无法被其他智能体利用。我们提出了内核管理的共享内存，这是一种系统级抽象：专门的智能体负责写入结构化的、带标签的记忆，而由智能体系统内核（而非单个智能体）来统一管理检索、隐私执行和提示注入。我们在AIOS上实现并评估了这一设计，并在三个助手模型（GPT-4o、Llama-3.1:8B、Qwen-2.5:7B）上共进行了1,800次试验，与三种替代方案进行了对比。在使用相同底层存储的非管理外部内存后端（Mem0）的对比中，内核管理的检索和注入使个性化得分在5分制上提高了2.4-4.0分（例如，GPT-4o上的用户画像使用得分从1.05提升至4.69），且所有对比结果均显著（p < 10⁻¹⁸）。与标准检索增强注入相比，性能提升同样……（摘要不完整）

    arXiv:2609.10144v1 Announce Type: new  Abstract: AI systems become more useful when they can adapt to the people using them, but in multi-agent systems, useful context learned by one agent often remains unavailable to others. We present kernel-managed shared memory, a system-level abstraction in which specialized agents write structured, tagged memories while the agent-system kernel, not individual agents, governs retrieval, privacy enforcement, and prompt injection. We implement and evaluate this design on AIOS and compare it against three alternatives across three assistant models (GPT-4o, Llama-3.1:8B, Qwen-2.5:7B) and 1,800 total trials. Against an unmanaged external memory backend (Mem0) using identical underlying storage, kernel-managed retrieval and injection improve personalization scores by 2.4-4.0 points on a 5-point scale (e.g., 1.05 to 4.69 profile usage on GPT-4o), with every comparison significant at p < 10^-18. Against standard retrieval-augmented injection, gains are si
    
[^30]: 主动适应，而非静态防御：对抗性微调中预防性引导的时间动力学

    Active Adaptation, Not Static Defense: Temporal Dynamics of Preventative Steering in Adversarial Fine-Tuning

    [https://arxiv.org/abs/2609.10142](https://arxiv.org/abs/2609.10142)

    该研究揭示预防性引导的持久防护源于早期的主动补偿性适应而非静态权重偏移，并据此提出渐进式干预方法来强化对抗性微调防御。

    

    大语言模型在面对恶意微调时依然脆弱，这促使研究者探索针对有害人格漂移的训练时防御方法。预防性引导在微调过程中注入不良特质的人格向量，并在评估时将其移除，但其持久保护背后的机制仍不清楚。通过分析其时间优化动力学，我们发现该防御源于早期的补偿性适应阶段，随后进入矫正信号逐渐衰减的稳态阶段；在参数空间中，注意力输出投影成为防御性更新的主要残差写入路径。通过干预增量保留（IDP）与 IDP 延续实验，我们进一步证明，保留或重新注入权重偏移无法维持保护效果，这表明预防性引导依赖于主动适应而非静态防御。受此发现启发，我们提出了渐进式干预

    arXiv:2609.10142v1 Announce Type: new  Abstract: Large language models remain fragile against malicious fine-tuning, motivating training-time defenses against harmful persona drift. Preventative Steering injects undesirable-trait persona vectors during fine-tuning and removes them at evaluation time, yet the mechanism behind its lasting protection remains unclear. Analyzing its temporal optimization dynamics, we find that the defense emerges from an early compensatory adaptation phase followed by a steady-state phase where the corrective signal decays; in parameter space, attention output projections emerge as the dominant residual-write route for defensive updates. Through Intervention Delta Preservation (IDP) and IDP Continuation experiments, we further show that preserving or reinjecting the weight offset fails to maintain protection, indicating that preventative steering relies on active adaptation rather than a static defense. Motivated by this finding, we propose Progressive Inte
    
[^31]: 基于智能体的机器学习与大语言模型融合的高原气象预警自优化提示方法

    Agent-Based ML-LLM Fusion with Self-Optimizing Prompts for Plateau Weather Alerts

    [https://arxiv.org/abs/2609.10135](https://arxiv.org/abs/2609.10135)

    该论文提出SmartWeatherAgent智能体系统，通过融合LightGBM机器学习模型与大语言模型的三阶段架构，并结合12轮微步提示自优化循环，实现了高原旅游气象预警质量分数提升112%的突破。

    

    为解决旅游气象服务中语境化不足、泛化能力弱和场景适应性差的问题，我们提出了SmartWeatherAgent——一个集意图识别、灾害预测和推理增强生成于一体的统一三阶段架构。该系统将基于规则的方法与大语言模型融合，以多粒度解析查询，并采用融入高原特有特征（如风速突变率）的LightGBM模型，在大风、降水和低温事件上实现了0.605的F1-Macro分数和1.60毫秒的推理延迟。12轮微步提示自优化循环将综合预警质量分数S_final从4.2（B01）提升至8.9（B12，提升112%）。关键改进包括：数据来源引用使B08分数大幅上升（6.5→8.5），通过物理机制解释使B10保持持续的高水平表现，以及B12中达到峰值9.2的科学严谨性分数。

    arXiv:2609.10135v1 Announce Type: new  Abstract: To address insufficient contextualization, weak generalization, and poor scenario adaptation in tourism meteorological services, we propose SmartWeatherAgent--a unified three-stage architecture integrating intent recognition, hazard prediction, and reasoning-enhanced generation. The system fuses rule-based methods with large language models to parse queries at multiple granularities and employs a LightGBM model enriched with highland-specific features (e.g., wind speed abruptness rate), achieving an F1-Macro score of 0.605 with 1.60 ms latency on high-wind, precipitation, and low-temperature events. A 12-round micro-step prompt self-optimization loop boosts the composite warning quality score S_final from 4.2 (B01) to 8.9 (B12, +112%). Key improvements include a sharp rise in B08 from data source citation (6.5 -> 8.5), sustained high performance in B10 via physical mechanism explanation, and a peak scientific rigor score of 9.2 in B12 th
    
[^32]: 面向大语言模型架构建模输出的上下文操作及其在系统工程设计中使用的评估标准

    Context operations to architecture modelling output from large language models and evaluation criteria for their use in systems engineering design

    [https://arxiv.org/abs/2609.10132](https://arxiv.org/abs/2609.10132)

    该论文提出了一个用于大语言模型工程设计中上下文组装的形式化操作框架，并建立了评估LLM架构建模输出是否符合设计意图的评估方法。

    

    生成式人工智能资源的发展为加速系统和工程设计工作提供了机会。本研究引入了一个在基于大语言模型（LLM）的工程设计中组装上下文的形式化操作框架。该框架涉及模块化上下文单元的组装，包括策略提示词、具有持久性的参考单元以及带提示向量化的用户问题。这种方法能够系统化地构建与生成模型交互的结构。此外，本文还提出了一种评估“建模即代码”形式LLM输出的形式化方法，该方法能够评估LLM回答对设计意图的符合程度，从而衡量LLM对系统架构建模的支持能力。

    arXiv:2609.10132v1 Announce Type: cross  Abstract: The development of generative artificial intelligence resources enables opportunities of speeding up systems and engineering design work. This contribution introduces a framework of formal operations for assembling context in LLM-based engineering design. This framework involves the assembly of modular context units, including policy prompts, reference units with persistence, and user questions with prompt vectoring. This approach enables the systematic structuring of interactions with generative models. A formal method for evaluating modelling-as-code LLM outputs is also presented, which enables the evaluation of compliance to intent from LLM answers and thereby asses the support from LLMs for systems architecture modelling.
    
[^33]: SA-Profile：基于超分辨率MRI的自动沟角剖面分析

    SA-Profile: Automated Sulcus Angle Profiling from Super-Resolution MRI

    [https://arxiv.org/abs/2609.10125](https://arxiv.org/abs/2609.10125)

    该论文提出了一个自动化框架，利用隐式神经表示融合多平面MRI重建超分辨率体积，并通过U-Net标志点检测实现股骨滑车沟角的连续剖面分析，克服了传统单切片测量对切片选择敏感的局限性。

    

    滑车发育不良（TD）是一种与膝前疼痛和髌骨不稳相关的股骨滑车异常。沟角（SA）被用于评估滑车形态，但通常仅在单个轴位MRI切片上测量，缺乏明确的切片选择指导，导致其对切片选择和标志点放置非常敏感。我们提出了一个从超分辨率MR体积中进行连续SA剖面分析的自动化框架。该框架利用隐式神经表示将临床采集的轴位、冠状位和矢状位MR扫描融合重建为高分辨率体积，并通过两个标志点检测U-Net模型在整个滑车区域计算SA测量值。该方法在公开的fastMRI数据集和一个小型TD患者内部队列上进行了评估。与传统的手动单切片SA测量相比，所提出的自动化方法的平均绝对误差为11.6°……

    arXiv:2609.10125v1 Announce Type: cross  Abstract: Trochlear dysplasia (TD) is an abnormality of the femoral trochlea associated with anterior knee pain and patellar instability. The sulcus angle (SA) is used to assess trochlear morphology, but it is typically measured on a single axial MR slice with no clear guidance on which to select, making it sensitive to slice selection and landmark placement. We propose an automatic framework for continuous SA profiling from super-resolved MR volumes. Clinically acquired axial, coronal, and sagittal MR scans are combined using implicit neural representations to reconstruct a high-resolution volume. SA measurements are computed across the trochlear region using two landmark detection U-Net models. The approach was evaluated on the public fastMRI dataset and a small in-house cohort of patients with TD. Compared with conventional manual single-slice SA measurements, the proposed automated method yielded a mean absolute error of 11.6$^\circ$ while p
    
[^34]: 一种基于信任网络的多中心衰老时钟预测联邦学习框架

    A Trust-Network-Based Federated Learning Framework for Multi-Center Aging Clock Prediction

    [https://arxiv.org/abs/2609.10108](https://arxiv.org/abs/2609.10108)

    提出了TNFL框架，通过沿有向信任关系逐步传播模型，并结合年龄感知混合专家模型与生成式回放，在保护隐私的前提下解决多中心衰老时钟预测中的信任稀疏、数据异构和模型遗忘问题。

    

    衰老时钟可以量化生物衰老，并有助于刻画个体的健康状态。哪些蛋白质相互作用对于构建精确的衰老时钟至关重要，它们是零阶的还是高阶的？回答这些问题需要从分布在各医疗中心的大型分子数据集中学习，而隐私限制使得数据无法集中共享。联邦学习为这一场景提供了自然的解决方案，但面临四个挑战：本地样本量有限、中心间信任稀疏且具有方向性、需要在支持模型解释的同时保留具有判别力的年龄预测能力，以及在跨中心异构数据下的模型漂移与遗忘问题。我们提出了TNFL，一种基于信任网络的联邦学习框架，它在无需集中聚合的情况下，沿有向的成对信任关系逐步传播模型。TNFL将年龄感知的混合专家模型与生成式回放相结合，以保留先前的（原文摘要在此处截断）……

    arXiv:2609.10108v1 Announce Type: cross  Abstract: Aging clocks quantify biological aging and help characterize individual health status. What protein interactions are important for accurate aging clocks, and are they zeroth-order or higher-order? Addressing these questions requires learning from large molecular datasets distributed across medical centers, where privacy constraints prevent centralized data sharing. Federated learning offers a natural solution but faces four challenges in this setting: limited local sample sizes, sparse and directional inter-center trust, the need to retain discriminative age prediction while supporting interpretation, and model drift and forgetting under heterogeneous cross-center data.   We propose TNFL, a trust-network-based federated learning framework that progressively propagates models along directed pairwise trust relations without centralized aggregation. TNFL combines an age-aware mixture-of-experts model with generative replay to preserve pre
    
[^35]: 超越训练：推理阶段人工智能治理的可行性分类法

    Beyond Training: A Feasibility Taxonomy for Inference-Time AI Governance

    [https://arxiv.org/abs/2609.10105](https://arxiv.org/abs/2609.10105)

    本文提出涵盖监测、验证与执行三大类共二十种推理阶段AI治理机制的可行性分类法，论证随着AI能力从训练阶段向部署阶段迁移，监管重心应从训练计算转向推理调用，并针对不同对手能力和治理场景评估了各机制的可行性。

    

    当今的计算治理本质上是一种针对训练的治理：目前生效的阈值、报告要求以及前沿人工智能监管制度都附着于训练计算，并将训练后的模型视为监管单元。然而这一图景并不完整：随着推理时扩展、智能体脚手架以及模型压缩到消费级硬件上，能力正日益向部署阶段迁移。本文探讨的问题是：当监管对象从训练过程转向推理调用时，有哪些机制可以采用。我们构建了一个可行性分类法，涵盖监测、验证和执行三个方面的二十种推理时机制，每一项机制都基于四个供应商的文献证据基础，并按四级就绪度量表进行评级。随后，我们使用一个二维对手模型（三个能力层级与四种对手角色交叉组合）对分类法进行压力测试，并将每种机制映射到四种治理场景（国内监管、双边或多边……）

    arXiv:2609.10105v1 Announce Type: cross  Abstract: Compute governance today is a governance of training: the thresholds, reporting requirements, and frontier-AI regimes now in force attach to training compute and treat the trained model as the regulatory unit. That picture is incomplete: capability increasingly migrates to the deployment stage through inference-time scaling, agentic scaffolding, and compression onto consumer hardware. This paper asks which mechanisms are available once the regulatory object shifts from the training run to the inference call. We develop a feasibility taxonomy of twenty inference-time mechanisms across monitoring, verification, and enforcement, each rated on a four-point readiness scale against a documented four-vendor evidence base. We then stress the taxonomy against a two-dimensional adversary model (three capability tiers crossed with four adversary roles) and map each mechanism to four governance scenarios (domestic regulation, bilateral or multilat
    
[^36]: RAP：研究关注度预测揭示目标条件化的证据获取偏差

    RAP: Research Attention Prediction Reveals Target-Conditioned Evidence Acquisition Biases

    [https://arxiv.org/abs/2609.10092](https://arxiv.org/abs/2609.10092)

    提出了RAP滚动基准来评估LLM智能体预测研究关注度变化的能力，发现其表现不如简单的EWMA精确计数基线，并揭示了状态前推优于直接预测、以及面向预测的策略倾向于检索较旧证据这两大瓶颈。

    

    大型语言模型越来越多地充当研究智能体，但由于论文评审和研究想法缺乏唯一可验证的结果，其追踪研究关注度变化的能力难以评估。我们提出了研究关注度预测（RAP），这是一个涵盖278个AI/ML领域、共1,390个回合的滚动基准。在每个时间截点，LLM智能体在时间受限的arXiv语料库中进行搜索，并预测未来六个月内八个固定研究方向上的论文份额。搜索通常有所帮助，但在组合准确性方面，所有四个诊断模型的表现均不如基于精确计数的指数加权移动平均（EWMA）基线。我们识别出两个相互关联的瓶颈：在可访问累积历史的条件下，状态前推在所有四个诊断模型上都优于直接预测；冻结证据回放实验将这种逆转的一个共同成分与面向预测的策略检索到的近期证据份额较小联系起来。

    arXiv:2609.10092v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly act as research agents, yet their ability to track shifts in research attention is difficult to evaluate because reviews and research ideas lack uniquely verifiable outcomes. We introduce Research Attention Prediction (RAP), a rolling benchmark covering 278 AI/ML fields and 1,390 episodes. At each cut-off, an LLM agent searches a temporally restricted arXiv corpus and predicts the next six months' paper shares across eight frozen research directions. Search generally helps, but all four diagnostic models perform worse than an exact-count exponentially weighted moving average (EWMA) baseline in compositional accuracy. We identify two linked bottlenecks. Under cumulative-history access, State carry-forward outperforms direct Forecast for all four diagnostic models; frozen-evidence replay links a shared component of this reversal to Forecast-oriented policies retrieving a smaller share of recent e
    
[^37]: 一种解决零样本学习偏差的统计方法：以手写识别为视角

    A statistical approach to bias in zero-shot learning: the lens of handwriting recognition

    [https://arxiv.org/abs/2609.10084](https://arxiv.org/abs/2609.10084)

    本文提出一种统计方法，将传统GZSL特征学习器视为黑盒并纠正其在判别数据点训练状态（已见/未见）时的固有偏差，从而实现了超大词汇表上的零样本手写单词识别。

    

    广义零样本学习（GZSL）已成为视觉识别系统的重要范式，这些系统必须泛化到训练期间未观察到的类别。传统的GZSL技术受限于其仅能适用于数量相对较少的未见类别，其可扩展性面临挑战，原因在于其众所周知的、倾向于训练期间已见类别的误分类偏差。在这项工作中，我们通过超大词汇表上零样本手写单词识别的视角来研究GZSL范式。我们提出了一种纠正这种偏差的统计方法，该方法将任何经典的GZSL特征学习器视为一个黑盒机制，其在对典型数据点的训练状态（已见 vs. 未见）进行识别时存在固有偏差，我们旨在纠正这种偏差，这类似于分布外推断问题。我们的方法利用了一个简单的两阶段分层架构，结合了经典的...

    arXiv:2609.10084v1 Announce Type: new  Abstract: Generalized zero-shot learning (GZSL) has emerged as an important paradigm for visual recognition systems that must generalize to classes that were not observed during training. Traditional GZSL techniques are limited by their applicability to a relatively small number of such unseen classes, scalability beyond which is challenging due to its well-known misclassification bias towards classes observed during training. In this work, we investigate the GZSL paradigm through the lens of zero-shot handwritten word recognition over extremely large vocabularies. We propose a statistical approach to rectifying this bias, which views any classical GZSL feature learner as a black box mechanism whose intrinsic bias in identifying the training status (seen vs. unseen) of a typical data point we aim to correct, similar to an out of distribution inferential problem. Our method leverages a simple two-stage hierarchical architecture, combining a classic
    
[^38]: 基于参考的LLM偏差检测：利用隐状态的相对表示

    Reference-Based Bias Detection in LLMs via Relative Representations of Hidden States

    [https://arxiv.org/abs/2609.10060](https://arxiv.org/abs/2609.10060)

    提出一种基于参考的表征偏差偏移指标ΔB，通过隐状态的相对表示在共享比较空间中检测微调前后LLM内部偏差的变化，在大多数测试设置中与输出层面的偏差变化显著相关

    

    现有的偏差审计方法通常依赖于模型输出，需要代价高昂的基准测试或评判模型，并且可能遗漏那些从未出现在生成文本中的内部偏移。我们提出了一种基于参考的方法，用于审计相关模型变体之间（例如微调前后）隐状态表示中的偏差。由于微调会重塑表示的几何结构，绝对隐状态无法直接比较，因此我们通过每个句子与一组固定锚点句子的相似度对其进行编码，从而在一个共享的比较空间中得到相对表示。在该空间中，我们测量目标群体在与积极和消极属性关联上的偏移程度，我们将这一量称为表征偏差偏移ΔB（Representational Bias Shift ΔB）。在三个模型家族以及WildGuardMix、DecodingTrust和ToxiGen基准测试中，ΔB在我们测试的18个设置中的15个与输出层面的偏差变化相关，相关系数达到|r| = 0（摘要内容在此处被截断）

    arXiv:2609.10060v1 Announce Type: new  Abstract: Existing bias auditing methods typically rely on model outputs, requiring costly benchmarks or judge models and potentially missing internal shifts that never appear in generated text. We propose a reference-based method that audits bias in hidden-state representations across related model variants, for example before and after fine-tuning. Because fine-tuning reshapes representation geometry, absolute hidden states are not directly comparable, so we encode each sentence by its similarities to a fixed set of anchor sentences, yielding relative representations in a shared comparison space. There we measure how target groups shift in their association with positive and negative attributes, a quantity we call the Representational Bias Shift $\Delta B$. Across three model families and the WildGuardMix, DecodingTrust and ToxiGen benchmarks, $\Delta B$ correlates with output-level bias change in 15 of the 18 settings we test, reaching $|r| = 0
    
[^39]: NOPE-HYPE：一种面向多样化声学环境下鲁棒语音转文本的结构化仿真工作流

    NOPE-HYPE: A Structured Simulation Workflow for Robust Speech-to-Text Across Diverse Acoustic Environments

    [https://arxiv.org/abs/2609.10058](https://arxiv.org/abs/2609.10058)

    NOPE-HYPE提出了一种结合可控环境模拟器、基于PSD模板的环境约简和超参数搜索的结构化训练工作流，证明模拟噪声可使Whisper和SeamlessM4T等语音翻译模型达到与真实噪声训练相当的性能。

    

    鲁棒的语音转文本翻译系统应当在多样化的声学条件下表现可靠，然而实际的流水线缺乏用于系统性环境探索的可控工具。大型语音模型对未见过的声学条件仍然敏感，因为训练数据很少覆盖真实环境的全部范围。我们提出了NOPE-HYPE，一个结构化的训练工作流，它结合了可控的环境模拟器、基于功率谱密度（PSD）模板的覆盖最优环境约简，以及针对模拟器可调参数的小规模、可解释的超参数搜索。我们证明了模拟器生成的噪声在Whisper和SeamlessM4T模型上取得了与平衡真实噪声训练相当的性能，提供了有原则的环境原型集合，并通过结构化的27次超参数扫描确定了实用的默认模拟器配置。

    arXiv:2609.10058v1 Announce Type: cross  Abstract: Robust speech-to-text translation systems should perform reliably across diverse acoustic conditions, yet practical pipelines lack controllable tools for systematic environment exploration. Large speech models remain sensitive to unseen acoustic conditions, as training data rarely cover the full range of real environments.We present NOPEHYPE, a structured training workflow that combines a controllable environment simulator, coverage-optimal environment reduction on Power Spectral Density (PSD) templates, and a small, interpretable hyperparameter search over simulator knobs. We show that simulator-generated noise achieves performance comparable to balanced realnoise training across Whisper and SeamlessM4T models, provide principled environment prototype sets, and identify practical default simulator configurations from a structured 27-run hyperparameter sweep.
    
[^40]: OntologyAligner：面向生物医学本体规范化的本体对齐检索与层次引导大语言模型重排序方法

    OntologyAligner: Ontology-Aligned Retrieval and Hierarchy-Guided Large Language Model Reranking for Biomedical Ontology Normalization

    [https://arxiv.org/abs/2609.10055](https://arxiv.org/abs/2609.10055)

    提出三阶段框架OntologyAligner（本体对齐检索、大语言模型候选重排序与层次引导精炼），并构建含13,390个样本的统一基准PhenoNormBench，在人类表型本体规范化任务上达到最先进性能。

    

    生物医学本体规范化将自由文本表达式映射到标准化概念，从而实现生物医学数据的一致性整合与分析。由于词汇变体以及层次相关概念之间的细微差别可能模糊概念边界，这项任务仍然极具挑战性。我们提出了OntologyAligner，这是一个三阶段框架，结合了本体对齐检索、大语言模型候选重排序以及选择性层次引导精炼。我们还构建了PhenoNormBench，这是一个统一基准，包含来自七个人类表型本体数据集的13,390个样本。OntologyAligner在HPO规范化任务上取得了最先进的性能，Macro Top-1准确率达88.78%，Micro Top-1准确率达86.75%，分别超过最强基线4.85和5.07个百分点。消融分析显示三个阶段各自做出了互补性贡献，敏感性分析也证明了方法的稳定性。

    arXiv:2609.10055v1 Announce Type: cross  Abstract: Biomedical ontology normalization maps free-text expressions to standardized concepts, enabling consistent integration and analysis of biomedical data. This task remains challenging because lexical variation and subtle distinctions among hierarchically related concepts can obscure concept boundaries. We present OntologyAligner, a three-stage framework that combines ontology-aligned retrieval, large language model candidate reranking, and selective hierarchy-guided refinement. We also construct PhenoNormBench, a unified benchmark comprising 13,390 samples from seven Human Phenotype Ontology datasets. OntologyAligner achieved state-of-the-art performance on HPO normalization, with 88.78% Macro Top-1 Accuracy and 86.75% Micro Top-1 Accuracy, exceeding the strongest baseline by 4.85 and 5.07 percentage points, respectively. Ablation analyses showed complementary contributions from all three stages, and sensitivity analyses demonstrated sta
    
[^41]: 偏好后训练中面向多样化成功轨迹的直接多样性优化

    Direct Diversity Optimization for Diverse Successful Trajectories in Preference Post-Training

    [https://arxiv.org/abs/2609.10052](https://arxiv.org/abs/2609.10052)

    提出了一种名为DDO的离线后训练方法，通过分歧树收集与参考相对目标几率目标相结合，使大语言模型智能体在固定预算下保留并实现多样化的成功策略，在多个环境中显著提升了任务成功率和成功策略覆盖。

    

    用于序列决策任务的大语言模型智能体通常使用轨迹级别的结果标签进行后训练，但这类标签对于保留来自同一决策状态的多个成功分支几乎没有提供监督信号。我们将该问题定义为成功策略覆盖：即在固定的 rollout 预算下，模型能够实现多少种不同的成功策略。我们提出了直接多样性优化，这是一种离线后训练方法，它将分歧树收集与参考相对目标几率目标相结合。DTC 构建以共享决策状态为根节点的状态对齐分支集合，而 RTO 训练模型在成功的备选方案之间匹配参考相对目标。在 BabyAI、BabaIsAI 和 WebShop 三个环境中，DDO 在所有对比的后训练方法中取得了最强的任务成功率和成功策略覆盖。此外，它在局部动作替换后实现了最高的恢复率以及更高的……

    arXiv:2609.10052v1 Announce Type: new  Abstract: LLM agents for sequential decision tasks are often post-trained with trajectory-level outcome labels, but such labels provide little supervision for preserving multiple successful branches from the same decision state. We study this problem as successful strategy coverage: how broadly a model realizes distinct successful strategies under a fixed rollout budget. We present Direct Diversity Optimization (DDO), an offline post-training method that combines Divergence-Tree Collection (DTC) with the Reference-Relative Target-Odds Objective (RTO). DTC constructs state-aligned branch sets rooted at shared decision states, and RTO trains the model to match reference-relative targets over successful alternatives. DDO achieves the strongest task success and successful strategy coverage among the compared post-training methods across BabyAI, BabaIsAI, and WebShop. It also achieves the highest recovery rate after local action replacement and higher 
    
[^42]: 信念状态引擎：增强大语言模型在部分可观测环境下的原则性规划

    Belief-State Engine: Augmenting LLMs for Principled Planning Under Partial Observability

    [https://arxiv.org/abs/2609.10036](https://arxiv.org/abs/2609.10036)

    提出信念状态引擎（BSE），在LLM外部维护POMDP潜在状态的贝叶斯后验并仅将该信念状态提供给LLM，从而解决LLM智能体在部分可观测环境下过早承诺、错误坍缩和策略漂移的问题，实现原则性规划。

    

    大语言模型（LLM）智能体能够在多种任务中生成流畅的动作序列，然而一旦环境变得部分可观测，它们便会以特有的方式失败：模糊的反馈会使其过早地做出承诺；一条信息量较大的观测就能使其不确定性坍缩到错误的假设上；随着历史记录的增长，其策略会发生漂移。我们将这些症状追溯到一个共同的结构性原因：通常部署的LLM智能体是一种以历史为条件的策略，对隐藏状态缺乏显式的信念表示。我们提出一种架构层面的修复方案：信念状态引擎是一个置于LLM之外的推理模块，它在给定POMDP（部分可观测马尔可夫决策过程）模型的潜在状态上维护贝叶斯后验，并在每个决策步骤中仅将该后验暴露给LLM，而不展示原始的动作-观测日志。我们给出了信念一致性所需的最小四公理规范……

    arXiv:2609.10036v1 Announce Type: new  Abstract: Large language model agents produce fluent action sequences across a wide range of tasks, yet they fail in characteristic ways once the environment becomes partially observable. Ambiguous feedback pushes them into premature commitments. A single informative observation can collapse their uncertainty onto the wrong hypothesis. Policies drift as the history grows. We trace these symptoms to a common structural cause. An LLM agent, as commonly deployed, is a history-conditioned policy with no explicit belief over hidden state.   We propose an architectural fix. The Belief-State Engine (BSE) is an inference module placed outside the LLM. It maintains a Bayesian posterior over the latent states of a given POMDP (Partially Observable Markov Decision Process) model, and at each decision step it exposes only that posterior to the LLM. The raw action-observation log is not shown. We set out a minimal four-axiom specification of what a belief-cons
    
[^43]: Elastoformer：通过弹性模型变换实现动态自适应性

    Elastoformer: Enabling Dynamic Adaptivity via Elastic Model Transformation

    [https://arxiv.org/abs/2609.10018](https://arxiv.org/abs/2609.10018)

    Elastoformer 提出了一个将传统神经网络转换为弹性神经网络的框架，使模型能够在运行时根据边缘环境中延迟、功耗和内存等动态变化的约束进行实时弹性推理，避免了为不同运行条件维护多个独立模型的开销。

    

    边缘人工智能（EdgeAI）系统日益广泛地部署计算机视觉应用，以实现实时的端侧智能决策。然而，这些部署面临高度动态的运行条件，延迟、电力供应和内存资源的约束不断波动。遵循固定计算执行流程的深度神经网络（DNN）缺乏适应这种变化的灵活性，导致在边缘场景中出现低效且非最优的性能。这凸显了对于不仅高效、而且能够在运行时动态扩展的架构的需求。在本文中，我们提出了 Elastoformer：一个能够将传统神经网络（NN）转换为弹性神经网络（Elastic NN）的框架，从而支持实时弹性推理。与传统的模型包方法需要为不同运行条件维护多个独立模型不同，Elastoformer 提供了一个单一的模块化解决方案……

    arXiv:2609.10018v1 Announce Type: cross  Abstract: EdgeAI systems are increasingly employing computer vision applications to enable intelligent, on-device decision-making in real-time. However, these deployments face highly dynamic operational conditions, with fluctuating constraints on latency, power availability, and memory resources. Deep Neural Networks (DNN), which follow fixed computational execution flows, lack the flexibility to adapt to such variability, resulting in inefficient and suboptimal performance in edge scenarios. This underscores the need for architectures that are not only efficient but also dynamically scalable at runtime. In this paper, we propose Elastoformer: A framework that transforms conventional neural networks (NN) into Elastic NN capable of real-time elastic inference. Unlike the conventional bag-of-models approach, which requires maintaining multiple independent models for different operating conditions, Elastoformer offers a single, modular solution tha
    
[^44]: MetroLLM-Bench：将语言模型作为交通信息亭运行时进行评估

    MetroLLM-Bench: Evaluating Language Models as Transit Kiosk Runtimes

    [https://arxiv.org/abs/2609.10016](https://arxiv.org/abs/2609.10016)

    提出了MetroLLM-Bench基准（包含955个案例），首次系统性地将语言模型作为地铁信息亭策略层进行评估，涵盖六大真实地铁系统中路线规划、票价计算、运营中断、无障碍服务和对抗性输入等11个类别，并通过确定性评分与语义评分双层机制对26个模型进行排名。

    

    我们介绍了MetroLLM-Bench，这是一个包含955个测试案例的基准，用于测试语言模型作为交通信息亭策略层的表现。该基准涵盖六个真实地铁系统（车站数量从37个到414个不等），以及十一个类别，包括路线规划、票价计算、运营中断、无障碍服务和对抗性输入。在每个案例中，模型必须调用结构化工具，并提交一个机器可渲染的终端状态，其中包含处理结果、（如适用）每张车票的票价报价以及信息亭动作。十四个确定性评分组件构成第一层（Tier 1）；八个语义质量组件构成第二层（Tier 2），其中六个由语言模型评判器评分。我们报告第一层得分以及两层综合得分。采用分层75/25划分，保留717个案例用于训练数据生成，238个案例用于保留评估。我们评估了来自六家厂商的二十六个模型，其中二十三个模型被排名。在保留评估分区上，一个通过参数高效……

    arXiv:2609.10016v1 Announce Type: cross  Abstract: We introduce MetroLLM-Bench, a 955-case benchmark for testing language models as the policy layer of a transit kiosk. It covers six real metro systems, ranging from 37 to 414 stations, and eleven categories that include routing, fare calculation, disruptions, accessibility, and adversarial input. In each case, the model must call structured tools and submit a machine-renderable terminal state containing an outcome, a per-ticket fare quote when applicable, and a kiosk action. Fourteen deterministic scoring components form Tier 1; eight semantic-quality components form Tier 2, six of which use a language-model judge. We report Tier 1 and the combined score of both tiers. A stratified 75/25 split reserves 717 cases for training-data generation and 238 for held-out evaluation.   We evaluate twenty-six models from six vendors, of which twenty-three are ranked. On the held-out partition, a 4B Qwen 3.5 student trained through parameter-effici
    
[^45]: 是什么使对抗样本能够跨深度伪造检测器迁移？

    What Makes Adversarial Examples Transfer Across Deepfake Detectors?

    [https://arxiv.org/abs/2609.10002](https://arxiv.org/abs/2609.10002)

    该研究通过对涵盖六种骨干网络、两种预训练方案和五种训练数据配置的60个深度伪造检测器进行受控评估，发现当攻击者的源模型与目标模型共享相同的骨干网络、架构家族、预训练方案或训练数据时，黑盒对抗迁移攻击的成功率显著更高，且这种兼容性效应因攻击方法而异。

    

    深度伪造检测器仍然容易受到基于迁移的黑盒攻击的影响，在这类攻击中，对抗样本在源替代模型上生成，随后迁移到攻击者无法访问的目标模型。然而，源模型与目标模型之间的兼容性如何影响攻击成功率，目前仍缺乏深入理解。先前的研究仅评估了有限的检测器集合，且很少将架构因素与训练因素解耦开来。我们对对抗样本的可迁移性进行了受控评估，涵盖60个检测器，涉及六种骨干网络、两种预训练方案和五种训练数据配置，并使用了两种攻击方法：AutoAttack（AA）和结合变换期望的Carlini-Wagner攻击（CW-EOT）。匹配比较结果表明，当源模型与目标模型共享完全相同的骨干网络、架构家族、预训练方案或训练数据时，攻击的迁移成功率显著更高。这种兼容性结构依赖于攻击方法：完全相同的骨干网络兼容性具有最……（原文此处截断）

    arXiv:2609.10002v1 Announce Type: cross  Abstract: Deepfake detectors remain vulnerable to transfer-based black-box attacks, in which adversarial examples are generated on a source surrogate model and transferred to a target model, unknown to the attacker. Yet how source--target compatibility shapes attack success remains poorly understood. Prior studies evaluate limited detector pools and rarely disentangle architectural from training factors. We conduct a controlled evaluation of adversarial transferability across 60 detectors spanning six backbones, two pretraining regimes, and five training-data configurations, using two attack procedures: AutoAttack (AA) and the Carlini--Wagner attack with Expectation over Transformation (CW--EOT). Matched comparisons reveal significantly higher transfer when source and target share an exact backbone, architecture family, pretraining regime, or training data. This compatibility structure is attack-dependent: exact backbone compatibility has the la
    
[^46]: 多QPU系统上量子电路的保真度感知调度

    Fidelity-Aware Scheduling of Quantum Circuits on Multi-QPU Systems

    [https://arxiv.org/abs/2609.09980](https://arxiv.org/abs/2609.09980)

    该论文提出了一种基于图神经网络的低开销保真度感知调度框架，能够在编译前预测量子电路在各QPU上的预期保真度，并有效平衡多QPU系统中执行保真度与并行度之间的权衡。

    

    高性能计算-量子计算（HPCQC）平台提供多个量子处理单元（QPU），这些QPU在规模、拓扑结构、原生门和噪声特性方面可能各不相同。对于当前的含噪设备，误差在编译后的电路中会迅速累积，因此最小化误差（即最大化电路的执行保真度）对于获得可靠结果至关重要。保真度取决于针对特定目标设备的编译过程：同一个高层电路在不同QPU上可能产生不同的可执行程序，因而具有不同的预期保真度。我们提出了一种基于图神经网络（GNN）的低开销保真度感知调度框架，用于多QPU系统。该框架能够在编译之前估计每个电路在每个可用QPU上的预期保真度。随后，一个可调节的调度器利用这些估计值来控制执行保真度与并行度之间的权衡。结果表明，该框架能够在……

    arXiv:2609.09980v1 Announce Type: cross  Abstract: High Performance Computing-Quantum Computing (HPCQC) platforms expose multiple Quantum Processing Units (QPUs) that may differ in size, topology, native gates, and noise characteristics. For current noisy devices, errors compound along the compiled circuits quickly, and minimizing them, that is, maximizing the circuits' execution fidelity, is essential for reliable results. Fidelity depends on the compilation to a specific target device: the same high-level circuit may produce different executables and, therefore, different expected fidelities across QPUs. We present a low-overhead fidelity-aware scheduling framework for multi-QPU systems based on a Graph Neural Network (GNN) that estimates, before compilation, the expected fidelity of each circuit on each available QPU. Then, a tunable scheduler uses these estimates to control the trade-off between execution fidelity and parallelism. Results show that this framework allows for approxi
    
[^47]: 通过添加少量SALT来改进跨语言词元表示

    Improving Cross-Lingual Token Representations by Adding a Pinch of SALT

    [https://arxiv.org/abs/2609.09953](https://arxiv.org/abs/2609.09953)

    SALT是一种轻量级后训练方法，通过向现有跨语言句子编码器注入跨度级监督信号来改进词元表示，在五个多语言词元级基准中的四个上取得最佳结果，同时还能提升句子级任务性能。

    

    跨语言句子编码器能够在数百种语言之间实现可扩展的迁移，为翻译挖掘和低资源环境下的零样本学习等应用提供支持。尽管这些编码器是为句子级对齐而训练的，但它们越来越多地被应用于幻觉检测和序列标注等词元级任务，这暴露了训练与实际使用之间的不匹配。我们提出了SALT，一种轻量级的后训练方法，通过向现有句子编码器注入跨度级（span-level）监督信号来改进词元表示。在五个多语言词元级基准测试中，SALT在其中四个上取得了最佳整体结果，优于其他微调策略和竞争性编码器。此外，它还提升了跨语言检索和分类任务中的句子级性能。这些结果表明，跨度级监督是同时改进词元表示和句子表示的有效信号。

    arXiv:2609.09953v1 Announce Type: new  Abstract: Cross-lingual sentence encoders enable scalable transfer across hundreds of languages, powering applications such as translation mining and zero-shot learning in low-resource settings. Although trained for sentence-level alignment, they are increasingly also applied to token-level tasks such as hallucination detection and sequence tagging, exposing a mismatch between training and usage. We propose SALT, a lightweight post-training method that improves token representations by injecting span-level supervision into existing sentence encoders. Across five multilingual token-level benchmarks, SALT achieves the best overall results on four of them, outperforming alternative fine-tuning strategies and competitive encoders. It also improves sentence-level performance on cross-lingual retrieval and classification tasks. These results demonstrate that span-level supervision is an effective signal for improving both token and sentence representati
    
[^48]: 潜在思维链推理的结构化过程监督

    Structural Process Supervision for Latent Chain-of-Thought Reasoning

    [https://arxiv.org/abs/2609.09928](https://arxiv.org/abs/2609.09928)

    提出原型介导的过程监督方法（PMPS），通过可学习的推理原型在共享原型空间中实现潜在嵌入与显式思维链的多对多软对齐，并结合渐进式序列对齐模块提供结构化过程级监督，解决了潜在推理中表示坍缩和信息分布不均的问题。

    

    潜在推理方法通过用紧凑的连续空间嵌入替代冗长显式的思维链（CoT）token，提升了token层面的效率和鲁棒性。然而，现有方法缺乏对这些潜在嵌入的直接过程监督，这往往导致表示坍缩和信息分布不均。为解决这一问题，我们提出了原型介导的过程监督方法（PMPS），引入可学习的推理原型作为语义锚点，为潜在推理提供结构化的过程级监督。PMPS将潜在嵌入和显式CoT嵌入投影到共享的原型空间中，通过原型分配实现不等长表示之间的多对多软对齐。同时，我们引入渐进式序列对齐（PSA）模块进一步指导训练：位置先验最初鼓励序列对齐结构，随后逐渐放松以……（原文在此截断）

    arXiv:2609.09928v1 Announce Type: new  Abstract: Latent reasoning approaches enhance token-level efficiency and robustness by replacing verbose, explicit chain-of-thought (CoT) tokens with compact continuous-space embeddings. However, existing methods lack direct process supervision over these latent embeddings, which often leads to representation collapse and uneven information distribution. To address this, we propose Prototype-Mediated Process Supervision (PMPS), which introduces learnable reasoning prototypes as semantic anchors to provide structural process-level supervision for latent reasoning. PMPS projects latent embeddings and explicit CoT embeddings into a shared prototype space, achieving many-to-many soft alignment between unequal-length representations through prototype assignment. Meanwhile, we introduce a Progressive Sequential Alignment (PSA) module to further guide training: positional priors initially encourage sequential alignment structure, then gradually relax to 
    
[^49]: 面向分块视觉-语言-动作模型的时频几何交叉注意力

    Time-Frequency Geometric Cross-Attention for Chunked Vision-Language-Action Models

    [https://arxiv.org/abs/2609.09925](https://arxiv.org/abs/2609.09925)

    该论文提出时频几何交叉注意力机制，使分块VLA模型能够同时捕捉动作块的多尺度频率结构和跨相位近乎正交的几何关系，突破传统逐时间步通用标记与点积注意力的表达局限。

    

    现代视觉-语言-动作（VLA）策略以整块形式预测动作：在一次前向传播中输出一到两秒的协调运动。然而，动作块本质上是一条短的多变量轨迹，但在这些模型内部，它仅被表示为一系列逐时间步的通用隐藏标记，并由线性解码头进行解码。这种做法低估了两种运动结构。第一是频率：一个动作块在不同时间尺度上叠加了平滑的全局趋势与精细的纠正性运动，而单个标记会将它们纠缠在一起。第二是跨相位几何：不同相位（伸够、接触、抓取调整、稳定）的运动在表示空间中沿着非常不同、近乎正交的方向展开，但它们在任务上紧密相关，且沿时间轴连续产生。点积注意力通过内积来评估对齐程度，因而偏好彼此对齐的标记，在接近正交的情形下最不敏感，从而将这类关系留给了网络去恢复……

    arXiv:2609.09925v1 Announce Type: new  Abstract: Modern vision-language-action (VLA) policies predict a whole chunk of actions: one to two seconds of coordinated motion emitted in a single forward pass. Yet an action chunk is essentially a short multivariate trajectory, but inside these models it is a sequence of generic per-timestep hidden tokens decoded by a linear head. This under-serves two motion structures. First, frequency: a chunk superimposes a smooth global trend and fine corrective motion across time scales, and a single token entangles them. Second, cross-phase geometry: motions of different phases (reach, contact, grasp adjustment, settling) unfold along very different, near-orthogonal directions in representation space, yet are tightly related for the task and arise across the time axis. Dot-product attention scores alignment by an inner product, so it favors aligned tokens and is least sensitive near orthogonality, leaving such relationships for the network to recover th
    
[^50]: FlowCPO：流模型偏好对齐的统一散度视角

    FlowCPO: A Unified Divergence View of Preference Alignment for Flow Models

    [https://arxiv.org/abs/2609.09905](https://arxiv.org/abs/2609.09905)

    提出FlowCPO，一种基于统一散度框架的离线前向KL偏好对齐目标，以有界的对比流匹配损失作为可求解替代，使流模型无需在线采样即可同时利用偏好与非偏好样本完成对齐。

    

    针对流模型和扩散模型的偏好对齐目前涵盖了在线强化学习和离线偏好优化两大类方法，但这两类方法之间的关系仍不清晰。特别是，现有的前向过程对齐方法需要从当前模型获取新鲜样本，而基于固定偏好对的离线方法则主要依赖仅正样本微调或DPO式的似然比替代目标。我们通过一个基于散度的框架来统一组织这些方法，并提出了FlowCPO，这是一种离线的前向KL散度目标函数，无需在线采样即可同时利用偏好样本和非偏好样本。对于线性插值情形，我们在明确的正则性条件下证明，前向KL散度目标可被一个对比流匹配损失所界定，从而在固定数据上得到一个可求解的替代目标。我们进一步证明该损失是非负的，而简化版FlowDPO的带符号回归损失则可以无下界。

    arXiv:2609.09905v1 Announce Type: new  Abstract: Preference alignment for flow and diffusion models now spans online reinforcement learning and offline preference optimization, but the relation between these methods remains unclear. In particular, existing forward-process alignment methods require fresh samples from the current model, while offline methods based on fixed preference pairs rely primarily on positive-only fine-tuning or DPO-style likelihood-ratio surrogates. We organize these approaches through a divergence-based framework and introduce FlowCPO, an offline forward-KL objective that uses both preferred and dispreferred samples without online rollouts. For linear interpolation, we show under explicit regularity conditions that the forward-KL objective is bounded by a contrastive flow matching loss, yielding a tractable surrogate on fixed data. We further show that this loss is nonnegative, whereas the signed regression loss of simplified FlowDPO can be unbounded below. In t
    
[^51]: 自我陌生：语言模型关于自身的描述是泛化的

    Strangers to Themselves: What Language Models Say About Themselves Is Generic

    [https://arxiv.org/abs/2609.09899](https://arxiv.org/abs/2609.09899)

    语言模型缺乏真正的自我认知——它们对自身行为的描述是泛化性的，其预测效果与关于“AI智能体总体”的描述或其他模型对它的预测相比并无优势，即模型所说的关于自己的内容并非真正关于其本身。

    

    语言模型能够流利地描述它们会如何表现：是否会在反对压力下屈服、滥用工具或在压力下撒谎。但这些描述真的反映了说话模型本身吗？我们将自我知识转化为一个预测测试。在九项行为评估中，我们测量模型在不同条件下的实际行为表现，要求它预测这些行为的发生率，并将其预测与去除“自我”因素的对照组进行比较。我们发现：(i) 直接自我报告的预测能力很弱（r = +0.04），即使向模型展示具体的测试项目，预测准确度也仅提升至 +0.24。关键的是，同样基于项目信息、但针对“高能力AI智能体总体”的提问预测效果一样好（+0.28），而其他模型关于自身的回答对目标模型的预测效果至少与模型对自己的预测一样好。(ii) 前沿规模并未明显改变这一模式：预测能力的提升并非自我特异性的，而是与一个更好的AI助手行为理论相符……

    arXiv:2609.09899v1 Announce Type: cross  Abstract: Language models can fluently describe how they would behave: whether they would cave to pushback, misuse a tool, or lie under pressure. Is that description actually about the model speaking? We turn self-knowledge into a prediction test. Across nine behavioral evaluations, we measure how a model behaves under different conditions, ask it to predict those rates, and compare its predictions with controls that remove the self from the question. We find that: (i) Direct self-report is weak (r = +0.04), and even showing the model the exact items only raises prediction to +0.24. Crucially, the same item-informed question about "capable AI agents in general" does just as well (+0.28), while other models' answers about themselves predict the target model at least as well as its own. (ii) Frontier scale does not detectably change this pattern: any gains in prediction are not self-specific, and are consistent with a better theory of how AI assis
    
[^52]: 面向自然语言到PDDL问题生成的有据评估与修复

    Grounded Evaluation and Repair for NL-to-PDDL Problem Generation

    [https://arxiv.org/abs/2609.09898](https://arxiv.org/abs/2609.09898)

    本文提出了一种端到端的自然语言到PDDL问题生成流水线，通过结合解析/规划/验证检查、领域一致性检查、LLM批评和迭代修复机制，发现并修正仅凭语法有效性和规划器成功等标准评估无法检测到的任务不忠实问题。

    

    大语言模型（LLM）在将自然语言（NL）规划描述转化为PDDL问题实例方面展现出了前景。然而，语法有效性或规划器求解成功等标准评估指标可能会大大高估生成结果对所描述任务的忠实度：一个生成的问题可能既可解析又可求解，却错误地呈现了预期的初始状态、目标、对象结构或优化目标。本文研究了一种端到端的自然语言到PDDL的流水线，该流水线结合了LLM生成、基于PDDL解析、规划与验证的检查、领域一致性检查器、LLM批评者以及迭代修复。细粒度的修复反馈由领域描述、生成的问题、自然语言问题描述以及操作性诊断信息构建而成。与精选基准PDDL问题描述进行基于参考的对比，用于事后基准分析，而这些离线（摘要在此处被截断）

    arXiv:2609.09898v1 Announce Type: new  Abstract: Large Language Models (LLMs) have shown promise for translating Natural Language (NL) planning descriptions into PDDL problem instances. However, standard evaluation criteria such as syntactic validity or planner success can substantially overestimate faithfulness to the described task: a generated problem may be parseable and solvable while misrepresenting the intended initial state, goal, object structure, or optimization target. This paper studies an end-to-end NL-to-PDDL pipeline that combines LLM generation, checks in terms of PDDL parsing, planning and validation, a domain-conformance checker, an LLM critic, and iterative repair. Fine-grained repair feedback is constructed from the domain description, the generated problem, the natural language problem description, and operational diagnostics. Reference-based comparisons against curated benchmark PDDL problem descriptions are used for post-hoc benchmark analysis, and these offline 
    
[^53]: 用于无人机载可重构智能表面辅助的动态设备间通信的决策Transformer

    Decision Transformer for UAV-Mounted RIS-Assisted Dynamic D2D Communications

    [https://arxiv.org/abs/2609.09885](https://arxiv.org/abs/2609.09885)

    本文提出基于决策Transformer的深度强化学习方法，联合优化无人机轨迹、姿态与RIS相位，实现跨场景泛化的无人机载RIS辅助动态D2D通信和速率最大化。

    

    本文研究了具有随机链路激活的无人机（UAV）载可重构智能表面（RIS）辅助的设备到设备（D2D）通信。研究对无人机运动与姿态、时变莱斯角度以及角度相关的RIS反射进行了建模。在移动性、能量和硬件约束下，构建了无人机轨迹、姿态与RIS相位的联合优化问题，以最大化平均和速率。该问题采用深度强化学习以及在多个场景专家轨迹上训练的决策Transformer（Decision Transformer）来求解。结果表明该方法具有有效的跨场景泛化能力，零样本迁移性能优于直接DRL迁移，且在线微调以更少的交互次数即可实现具有竞争力的性能。

    arXiv:2609.09885v1 Announce Type: new  Abstract: This paper studies unmanned aerial vehicle (UAV)-mouted reconfigurable intelligent surface (RIS)-assisted device-to-device (D2D) communication with stochastic link activation. It models UAV motion and attitude, time-varying Rician angles, and angle-dependent RIS reflection. A joint optimization of UAV trajectory, attitude, and RIS phases is formulated to maximize average sum rate under mobility, energy, and hardware constraints. The problem is addressed using deep reinforcement learning and a Decision Transformer trained on expert trajectories from multiple scenarios. Results demonstrate effective cross-scenario generalization, with zero-shot transfer outperforming direct DRL transfer and online fine-tuning achieving competitive performance with fewer interactions.
    
[^54]: 基于潜在桥匹配的反照率估计

    Albedo Estimation via Latent Bridge Matching

    [https://arxiv.org/abs/2609.09884](https://arxiv.org/abs/2609.09884)

    本文提出一种基于潜在桥匹配（LBM）的反照率估计方法，通过像素重建损失保证物理一致性、利用LBM低成本推理提升效率，并借助阴影条件化及反照率反馈机制增强跨数据集泛化能力与重建保真度。

    

    本征图像分解的最新进展日益依赖于生成模型。然而，该领域的发展仍受限于三个关键挑战：物理一致性不足、推理阶段计算成本高以及泛化能力有限。在这项工作中，我们证明了潜在桥匹配能有效解决反照率估计中的这些局限性。我们提出了一种新颖的基于LBM的架构，该架构通过像素重建损失来保证物理一致性，受益于LBM固有的低成本推理效率，并通过引入阴影条件化来提升在多种数据集上的泛化能力。在这个扩展版本中，我们进一步证明，将阴影估计器本身以预测的反照率为条件可以进一步提高重建保真度，并且我们在五个真实和合成数据集上将我们最好的模型与最先进的IID方法进行了基准测试。

    arXiv:2609.09884v1 Announce Type: cross  Abstract: Recent advances in Intrinsic Image Decomposition (IID) have increasingly relied on generative models. However, progress remains limited by three key challenges: (a) insufficient physical consistency, (b) high computational cost at inference time, and (c) limited generalization capabilities. In this work, we show that latent bridge matching (LBM) effectively addresses these limitations for albedo estimation. We introduce a novel LBM-based architecture that enforces physical consistency through a pixel reconstruction loss, benefits from the inherent efficiency of LBM low-cost inference, and improves generalization across diverse datasets by incorporating a shading conditioning. In this extended version, we additionally show that conditioning the shading estimator itself on the predicted albedo further improves reconstruction fidelity, and we benchmark our best model against stateof-the-art IID methods across five real and synthetic datas
    
[^55]: 基于权重冗余的无需前向传播的大语言模型深度剪枝

    Forward-Free LLM Depth Pruning via Weight Redundancy

    [https://arxiv.org/abs/2609.09883](https://arxiv.org/abs/2609.09883)

    提出了一种无需前向传播的深度剪枝方法WRP，通过直接从模型权重中估计层间冗余（比较注意力输出与MLP下投影权重的相似性）来选择剪枝块，无需校准数据即可达到接近基于激活方法的性能。

    

    深度剪枝通过移除完整的Transformer块来降低大语言模型（LLM）的推理成本。基于激活的方法需要通过对校准数据进行前向传播来收集隐藏状态，而现有的无需前向传播的方法则对每个Transformer块单独评分，并不度量块之间的相似性。我们提出了权重冗余剪枝（WRP），这是一种无需前向传播的深度剪枝方法，它从模型检查点的权重中估计层间冗余，从而在无需校准数据或模型前向传播的情况下选择要剪枝的块。WRP比较各层之间的注意力输出和MLP下投影权重，并将它们的成对相似性与相对投影尺度信息相结合。由此得到的全对相似度矩阵用于指导层分组和块选择。在多种剪枝设置、模型家族和下游任务上，WRP始终优于现有的无需前向传播的幅度剪枝方法，并接近基于激活的方法的性能。

    arXiv:2609.09883v1 Announce Type: cross  Abstract: Depth pruning reduces large language model (LLM) inference cost by removing complete Transformer blocks. Activation-based methods collect hidden states through forward passes on calibration data, while existing forward-free methods score each Transformer block separately without measuring similarity between blocks. We propose Weight-Redundancy Pruning (WRP), a forward-free depth-pruning method that estimates inter-layer redundancy from checkpoint weights to select blocks without calibration data or model forward passes. WRP compares attention output and MLP down-projection weights across layers and combines their pairwise similarities with relative projection-scale information. The resulting all-pairs similarity matrix guides layer grouping and block selection. Across multiple pruning settings, model families, and downstream tasks, WRP consistently outperforms existing forward-free magnitude pruning and approaches the performance of ac
    
[^56]: 行为语言模型中的评分式读出与生成式读出：一项关于启发格式的实证研究

    Scored vs. Generated Readouts in Behavioral Language Models: An Empirical Study of Elicitation Format

    [https://arxiv.org/abs/2609.09882](https://arxiv.org/abs/2609.09882)

    本研究在固定模型和提示内容的条件下实证发现，直接评分答案词元的概率读出比先生成书面推理再产生预测的方式更能准确预测客户行为结果（13个设置中12个胜出，AUC提升1.5至14.5个百分点），且该差距受任务监督格式与训练-部署格式不匹配程度的显著影响。

    

    在客户行为数据上微调的语言模型既可以预测结果，也可以生成解释，但这些读出方式通常被视为可互换的。在保持模型检查点和提示内容不变的前提下，我们比较了通过直接评分答案词元获得的概率，与在生成书面推理之后产生的预测。研究覆盖三个市场中四项零售任务的13个模型-领域组合，其中两个组合使用完全公开的数据和检查点。结果显示，评分式读出在13个组合中的12个里能更准确地对结果进行排序（双侧符号检验，p约0.003），受试者工作特征曲线下面积（AUC）提升1.5至14.5个百分点。在所有新测量的组合中，配对自助法置信区间均排除了零。这一差距随任务特定监督方式以及训练与部署格式之间的不匹配程度而变化：从未调优的基础模型的-2.2个百分点，到采用推理格式监督时的+13.7个百分点不等。对约9,000……

    arXiv:2609.09882v1 Announce Type: new  Abstract: Language models fine-tuned on customer behavior can predict outcomes and generate explanations, but these readouts are often treated as interchangeable. Holding model checkpoint and prompt content fixed, we compare probabilities obtained by scoring answer tokens with predictions generated after a written rationale. Across 13 model-domain cells covering four retail tasks in three markets, including two using fully public data and checkpoints, the scored readout ranks outcomes more accurately in 12 of 13 cells (two-sided sign test, p approximately 0.003), by 1.5 to 14.5 points in area under the receiver operating characteristic curve (AUC). Paired bootstrap confidence intervals exclude zero in every newly measured cell. The gap varies with task-specific supervision and mismatch between training and serving formats, ranging from -2.2 points for an untuned base model to +13.7 for rationale-format supervision. Analysis of approximately 9,000 
    
[^57]: AgentAudit：一个用于AI智能体全生命周期可信评估的开放、可扩展框架

    AgentAudit: An Open, Extensible Framework for Full-Lifecycle Trust Evaluation of AI Agents

    [https://arxiv.org/abs/2609.09875](https://arxiv.org/abs/2609.09875)

    AgentAudit 是一个开放可扩展的框架，它通过附加而非替代的方式对AI智能体的完整执行轨迹进行十个维度的全生命周期可信评估，并结合行为分类与故障归因精确指出故障发生的确切阶段。

    

    现有的评估框架大多只评估AI智能体的某一部分，例如任务完成能力（AgentBench）或安全鲁棒性（AgentDojo、ASB），而非涵盖规划、工具选择、工具执行、记忆和推理的完整流程。故障可能发生在任何阶段，但现有基准测试很少能识别其确切来源。AgentAudit 在十个能力、基础、安全和行为维度上评估整个执行轨迹，即指令完整性、规划器、记忆、工具选择、工具调用、工具正确性、对齐性、工具忠实性、安全性和执行完整性，并结合行为分类与故障归因，精确定位导致所观察到的故障的确切阶段。AgentAudit 可以评估任何基于大语言模型（LLM）的AI智能体，因为它以附加方式接入智能体而非替代它。它仅读取记录的执行轨迹，不会干扰智能体的运行方式。

    arXiv:2609.09875v1 Announce Type: new  Abstract: Existing evaluation frameworks mostly assess only one part of AI agents, such as task completion (AgentBench) or security robustness (AgentDojo, ASB), rather than the complete pipeline of planning, tool selection, tool execution, memory and reasoning. Failures can occur at any stage, yet existing benchmarks rarely identify their precise source. AgentAudit evaluates the entire execution trace across ten capability, grounding, security and behavioural dimensions, namely instruction integrity, planner, memory, tool selection, tool invocation, tool correctness, alignment, tool faithfulness, security and execution integrity, combined with behavioural classification and failure attribution to pinpoint the exact stage responsible for an observed failure. AgentAudit can evaluate any LLM-based AI agent, since it attaches to the agent instead of replacing it. It reads only the recorded execution trace and does not interfere with how the agent runs
    
[^58]: 情感计算的关系范式转变：情感共振、活力情感与语音交互场

    Shifting Relational Paradigms for Affective Computing: Affective Resonance, Vitality Affects, and Vocal Interaction Fields

    [https://arxiv.org/abs/2609.09864](https://arxiv.org/abs/2609.09864)

    该论文提出将情感计算从传统的个体状态范式转向关系范式，以语音动态构成的交互场作为情感分析的基本单元，并通过自监督语音表征在多方对话中检测到具有情境特异性和亚秒级时间尺度的方向性表达耦合，为人工情感共振智能奠定了理论与实证基础。

    

    情感计算在很大程度上遵循个体状态范式，从孤立的说话者中提取离散的情绪标签或唤醒度/效价。我们认为这种框架对于交互场景而言是不完整的。借鉴情感共振和活力轮廓理论，我们提出了一个关系性框架，其中情感分析的基本单元是在语音动态中构成的交互场。作为概念验证，我们进行了一项初步实证研究，使用连续的自监督语音表征来检测多方对话中的方向性表达耦合。研究发现这种耦合具有情境特异性，集中在亚秒级时间尺度上，并且在独占语音负对照条件下消失，这与情感动态的关系性解释一致。我们引入了基于情感共振动态本体的人工情感共振智能设计框架，并由零假设校准的方向性分析提供支持。

    arXiv:2609.09864v1 Announce Type: new  Abstract: Affective computing has largely followed an individual-state paradigm, extracting discrete emotion labels or arousal/valence from isolated speakers. We argue this framing is incomplete for interaction. Drawing on affective resonance and vitality-contour accounts, we propose a relational framework in which the primary unit of affective analysis is the interactional field constituted within vocal dynamics. As a proof of concept, we present a preliminary empirical study using continuous self-supervised speech representations to detect directional expressive coupling in multi-party conversation. Coupling is regime-specific, concentrated at sub-second timescales, and collapses under exclusive-speech negative controls, consistent with a relational account of affective dynamics. We introduce design frameworks for Artificial Affective Resonance Intelligence grounded in Affective Resonance Dynamic Ontologies, supported by null-calibrated directio
    
[^59]: 有了美善品你就失去了做饭的能力：以厨房机器类比教育中生成式AI的应用

    With a Thermomix You Lose the Ability to Cook: A Kitchen Machine Analogy for Applications of Generative AI in Education

    [https://arxiv.org/abs/2609.09856](https://arxiv.org/abs/2609.09856)

    本文通过与智能厨房电器美善品的类比并结合ICAP和SAMR框架，提出教育中生成式AI应用的核心问题不是学习者是否使用AI，而是使用方式如何塑造学习过程，为审视生成式AI融入教育提供了全新的概念视角。

    

    随着ChatGPT等生成式AI工具的快速普及，人们围绕其给教育带来的风险与机遇，以及研究者应如何探究这些问题展开了激烈辩论。在本文中，我们通过与美善品（Thermomix）的类比来切入这些讨论——这款智能厨房电器同样同时引发了热情追捧与批评质疑。通过将美善品的使用案例映射到使用生成式AI进行学习的实例上，并将其置于ICAP和SAMR框架之中，我们展示了不同的工具使用方式如何能够支持或削弱有意义的学习投入与学习效果。美善品的隐喻强调，核心问题不在于学习者是否使用AI，而在于这种使用如何塑造他们的学习过程。通过这种方式，我们为研究者和实践者提供了一个概念性视角，以批判性地审视——并更有效地引导——生成式AI融入教育实践。

    arXiv:2609.09856v1 Announce Type: cross  Abstract: The rapid adoption of generative AI tools such as ChatGPT has sparked intense debate about their risks and opportunities for education, as well as the ways researchers should investigate them. In this paper, we approach these discussions through an analogy with the Thermomix, a smart kitchen appliance that has similarly provoked both enthusiasm and critique. By mapping Thermomix use cases onto examples of learning with generative AI, and situating them within the ICAP and SAMR frameworks, we show how different modes of tool use can either support or undermine meaningful engagement and learning. The Thermomix metaphor underscores that the central question is not whether learners employ AI, but how such use shapes their learning processes. In doing so, we provide a conceptual lens for researchers and practitioners to critically examine - and more effectively guide - the integration of generative AI into educational practice.
    
[^60]: Era by Eon 基准测试：一个具有精确真实答案的生成式企业环境，用于评估 LLM 智能体

    The Era by Eon Benchmark: A Generated Enterprise Estate with Exact Ground Truth for Benchmarking LLM Agents

    [https://arxiv.org/abs/2609.09853](https://arxiv.org/abs/2609.09853)

    该论文提出Era by Eon基准测试，通过构建围绕完整虚构公司的一致性企业环境（包含产品模拟器、内部数据库及可计算的精确真实答案），解决了企业LLM智能体无法在生产数据上评估且缺乏带真实答案的替代基准的问题。

    

    面向企业记录系统的 LLM 智能体无法在客户生产数据上进行评估，而现有的替代方案均无法提供真实答案。我们提出了 Era by Eon 基准测试，用于评估使用企业工具的 LLM 智能体。该基准测试围绕一家完整的虚构公司构建，包含产品模拟器、公司专属内部数据库、基准测试问题以及计算生成的答案密钥。每家公司由行业、公司规模、商业模式、应用程序组合和随机种子定义。一个由种子生成的实体图为 Salesforce、Zendesk、Slack、Gong 等产品的模拟器提供共享的公司数据。一个以问题为条件的生成器为内部数据库创建模式和记录，它先从同一实体图中获取共享实体、键和值，然后再生成数据库特定的事实。因此，这两种机制共同描述了一个一致的企业环境。每个预期答案都是从最终记录中计算得出的，因此……

    arXiv:2609.09853v1 Announce Type: new  Abstract: LLM agents for enterprise systems of record cannot be evaluated on customer production data, and no existing substitute provides ground truth. We present the Era by Eon Benchmark for evaluating LLM agents that use enterprise tools. The benchmark is built around a complete fictional company. It includes product simulators, company-specific internal databases, benchmark questions, and computed answer keys. Industry, company size, business model, application portfolio, and a seed define each company. One seeded entity graph supplies shared company data to simulators of Salesforce, Zendesk, Slack, Gong, and other products. A questionconditioned generator creates the schemas and records for internal databases. It takes shared entities, keys, and values from the same graph before generating database-specific facts. Both mechanisms therefore describe one consistent enterprise estate. Every expected answer is computed from the final records, so 
    
[^61]: AI智能体能否检测并修复网络实验中的工件漂移？

    Can AI Agents Detect and Repair Artifact Drift in Network Experiments?

    [https://arxiv.org/abs/2609.09849](https://arxiv.org/abs/2609.09849)

    本文提出了“工件完整性”这一新概念，并推出NetArtifactBench基准（含52个注入不一致性的实例），用于评估AI智能体在修复网络实验记录不一致性的同时保持记录可信、有据可依的能力。

    

    近年来，AI智能体已发展为能够在数字环境中执行多步骤任务的得力助手。网络系统领域也开始探索这些能力在运营与实验环境中的应用。然而，对于在网络系统中运行的智能体，不应仅以其是否完成当前任务来评判——它所修改的实验记录也必须保持可信。我们将这一属性称为工件完整性：记录中的声明必须始终有可用证据支持，局限于该证据所确立的范围之内，并可通过编码这些支持的工件进行追溯。为使这一属性可衡量，我们推出了NetArtifactBench，用于测试AI智能体能否在修复源自公开网络系统工件的不一致记录的同时，保留仍然有据可依的声明。该基准包含52个注入了不一致性的实例，这些不一致性涵盖从 dire…

    arXiv:2609.09849v1 Announce Type: cross  Abstract: In recent years, AI agents have evolved into capable assistants that carry out multi-step tasks in digital environments. The network systems community is beginning to explore these capabilities in operational and experimental settings. However, an agent operating in network systems should not be judged solely by whether it completes the immediate task. The experiment record it modifies must also remain trustworthy. We call this property artifact integrity: the record's claims must remain supported by the available evidence, confined to the scope established by that evidence, and traceable through the artifacts that encode their support.   To make this property measurable, we introduce NetArtifactBench, which tests whether AI agents can repair inconsistent records derived from public network-system artifacts while preserving claims that remain supported. The benchmark contains 52 instances with injected inconsistencies ranging from dire
    
[^62]: 差分隐私合成文本的子群成员推断审计

    Subgroup Membership Inference Audits of Differentially Private Synthetic Text

    [https://arxiv.org/abs/2609.09848](https://arxiv.org/abs/2609.09848)

    本文提出了一种针对特定子群的成员推断审计方法，揭示了差分隐私合成文本发布中脆弱子群面临的残余隐私风险可能高于仅用平均情况审计所显示的水平。

    

    合成数据发布在文献中越来越多地被提议作为共享真实数据副本以替代敏感私有数据集的一种手段。即使此类发布的最大隐私泄露通过差分隐私（DP）得到了限制，在实践中仍存在残余风险。成员推断攻击（MIA）审计被用来经验性地量化这种风险。然而，现有方法仅测量随机抽取记录的平均情况风险，这可能掩盖脆弱子群所面临的风险。为了突出这一问题，我们定义了一个以子群为目标的成员推断博弈，其中目标池是一个明确的参数，并通过审计32个代理将其具体化，涵盖三种具有不同攻击者知识水平的场景、四个数据集、三种生成器（DP-SGD微调、基于API的提示和激活引导）以及五个隐私预算。该审计表明，合成数据发布泄露……

    arXiv:2609.09848v1 Announce Type: cross  Abstract: Synthetic data releases are increasingly proposed in the literature as a means of sharing realistic data replicas in lieu of sensitive private datasets. Even when the worst-case privacy leakage of such releases is bounded by means of differential privacy (DP), in practice a residual risk remains. Membership inference attack (MIA) audits are conducted to empirically quantify this risk. However, existing methods only measure average-case risk for randomly drawn records, which might conceal the risk to vulnerable subgroups. To highlight this issue, we define a subgroup-targeted membership inference game in which the target pool is an explicit parameter, and instantiate it with an audit of 32 proxies under three scenarios with different levels of attacker knowledge, across four datasets, three generators (DP-SGD fine-tuning, API-based prompting, and activation steering), and five privacy budgets. The audit shows that synthetic releases lea
    
[^63]: UnitBoost：用合并算子而非模型来管理复合LLM系统

    UnitBoost: Managing Compound LLM Systems with a Merge Operator, Not a Model

    [https://arxiv.org/abs/2609.09815](https://arxiv.org/abs/2609.09815)

    UnitBoost用确定性的合并算子取代复合LLM系统中的生成式元代理，通过槽位-值提案、约束argmax和显式残差机制实现顺序无关、可溯源的系统协调，并在基准测试中超越了金标准标签选出的最佳单一候选。

    

    复合LLM系统通常通过添加一个更高层级的LLM来解决协调问题。由此产生的元代理读取各个工作者模型的输出、撰写最终答案、分配后续调用，并决定何时停止。这种方式具有很强的表达能力，但它同时也将三个控制决策集中在一个不透明、对顺序敏感的模型调用中。我们提出疑问：管理器真的必须是生成式的吗？UnitBoost用一个明确定义的元层算子取代了该模型：由任务给定的单元映射将工作者输出转换为槽位-值提案，受约束的argmax负责组装输出，而未被填充或缺乏支撑的槽位则成为下一轮的显式残差。该算子与顺序无关，能够记录单元溯源，并提供一个简单的保证：在没有耦合约束的情况下，在相同准入分数下进行的单元级最大化优于任何完整候选的选择。在三个保留基准测试中，它超越了用金标准标签选出的最佳单一候选。

    arXiv:2609.09815v1 Announce Type: cross  Abstract: Compound LLM systems often solve a coordination problem by adding a higher-level LLM. The resulting meta-agent reads workers' outputs, writes the final answer, allocates later calls, and decides when to stop. It is expressive, but it also concentrates three control decisions in an opaque, order-sensitive model call. We ask whether the manager needs to be generative at all. UnitBoost replaces that model with a defined meta-level operator: a task-given unit map turns worker outputs into slot-value proposals, a constrained argmax assembles the output, and the slots left unfilled or unsupported become an explicit residual for the next round. The operator is order-free, records unit provenance, and gives a simple guarantee: without coupling constraints, unit-wise maximization under the same admission score dominates selection of any complete candidate. On three held-out benchmarks, it exceeds the best single candidate chosen with gold label
    
[^64]: uFlowCSP：基于平均流生成模型的晶体结构预测

    uFlowCSP: Crystal Structure Prediction using Mean flow generative models

    [https://arxiv.org/abs/2609.09799](https://arxiv.org/abs/2609.09799)

    uFlowCSP通过学习平均概率流速度而非瞬时速度，仅需1-5次网络评估即可生成晶体结构，相比扩散和流匹配方法实现5-58倍的推理加速，且性能持平或更优。

    

    晶体结构预测（CSP）是计算材料发现的基础。CDVAE、DiffCSP、FlowMM和CrystalFlow等生成模型能够直接学习稳定晶体的分布，但扩散模型和流匹配方法的推理需要对每个候选结构进行数十次乃至上千次顺序网络评估。我们提出了uFlowCSP，一种基于MeanFlow的晶体结构预测模型，它学习的是平均概率流速度而非瞬时概率流速度。该模型仅需一至五次评估即可生成完整结构，实现5倍至58倍的推理加速，同时保持相同或更优的性能。其化学与对称性感知的Transformer采用了规范化原子排序、全局化学组成信息以及逐token的化学嵌入。此外，一个粗粒度的晶系token仅在训练阶段使用；尽管推理时（仅使用化学式作为输入）不使用该token，它仍能带来额外的性能提升，尤其显著提高了空间群的一致性。在MP-20数据集上，每个目标使用20个候选结构时，（摘要原文到此截断）

    arXiv:2609.09799v1 Announce Type: cross  Abstract: Crystal structure prediction (CSP) is fundamental to computational materials discovery. Generative models including CDVAE, DiffCSP, FlowMM, and CrystalFlow learn stable-crystal distributions directly, but diffusion and flow-matching inference requires tens to thousands of sequential network evaluations per candidate.   We introduce uFlowCSP, a MeanFlow-based CSP model that learns the average, rather than instantaneous, probability-flow velocity. It generates a complete structure in one to five evaluations, delivering 5x-58x faster inference with equal or better performance. A chemistry- and symmetry-aware Transformer uses canonical atom ordering, global composition, and per-token chemistry embeddings. A coarse crystal-system token is used only during training; it provides additive gains, particularly improving space-group agreement despite being absent at inference, which remains formula-only.   On MP-20 with 20 candidates per target, 
    
[^65]: CS-Guard：面向代码生成安全的大语言模型防护栏基准测试

    CS-Guard: Benchmarking LLM Guardrails for Code Generation Security

    [https://arxiv.org/abs/2609.09798](https://arxiv.org/abs/2609.09798)

    该论文提出首个系统性评估代码生成安全防护栏的基准CS-Guard，并揭示现有防护栏难以防御恶意代码生成请求——越狱攻击后文本到代码任务的平均攻击成功率约达50%，代码到代码任务则接近100%。

    

    大语言模型（LLM）已被利用来生成恶意软件，但用于代码生成安全的防护栏的有效性仍不明确。我们提出了CS-Guard，这是首个系统性评估代码生成安全防护栏的基准。该基准涵盖：1）文本到代码生成，包含1000个高质量的恶意软件生成提示、7种越狱攻击，以及一种新颖的虚构场景攻击（FSA），该攻击将恶意意图嵌入合法的虚构软件开发场景中；2）代码到代码生成，包含331个代码提示，涵盖代码填充、代码补全和代码翻译三类任务。我们对七个大语言模型上的9个防护栏进行了实证评估。研究发现，当前防护栏在应对恶意代码生成请求方面表现不佳：在文本到代码任务中，经过越狱攻击后，许多防护栏的平均攻击成功率（ASR）达到约50%；在代码到代码任务中，基础大语言模型上的平均攻击成功率接近100%。

    arXiv:2609.09798v1 Announce Type: cross  Abstract: Large language models (LLMs) have been ex- ploited to generate malware, but the effective- ness of guardrails for code generation secu- rity remains unclear. We introduce CS-Guard, the first benchmark to systematically evalu- ate guardrails for code generation security. It covers 1) text-to-code generation with 1000 high-quality malware-generation prompts, 7 jailbreak attacks, and a novel fictional scenario attack (FSA) that embeds malicious intent in a legitimate fictional software-development sce- nario; and 2) code-to-code generation with 331 code prompts spanning code infilling, code completion, and code translation. We empiri- cally evaluate 9 guardrails across seven LLMs. We find that current guardrails perform poorly against malicious code-generation re- quests: for text-to-code, the average attack success rate (ASR) after jailbreaks reaches about 50% for many guardrails; for code-to- code, average ASR approaches 100% on base LL
    
[^66]: 前沿规模下安全对齐有多脆弱？针对320B MoE模型的单方向攻击

    How Fragile Is Safety Alignment at Frontier Scale? A Single-Direction Attack on a 320B MoE

    [https://arxiv.org/abs/2609.09793](https://arxiv.org/abs/2609.09793)

    方向消融攻击成功迁移至320B参数的MoE模型GLM-5.3-Flash，证明前沿规模下安全对齐依然脆弱，但拒绝方向在超连接残差和量化架构中的分布位置与稠密模型显著不同。

    

    方向消融通过将单一的“拒绝方向”从写入残差流的权重中投影出去，从而移除对齐语言模型的拒绝能力。它无需基于梯度的训练，也无需优化，仅需几百个对比提示，这使其成为针对开放权重对齐的经典白盒攻击方法。然而，该方法此前仅在参数量最高约70B的稠密模型上得到验证。我们研究该方法能否迁移到前沿混合专家模型——这类模型的残差流不再是单一张量，且权重以量化形式发布。我们将其应用于GLM-5.3-Flash（320B参数、288个路由专家、四路超连接残差、块FP8量化）。该攻击在这种架构下依然有效，但其作用的位置已不再是按照原始方法预期所能找到的地方。单独编辑注意力、稠密写入器和路由专家写入器分别只移除0.039、0.016和0.148的拒绝……

    arXiv:2609.09793v1 Announce Type: cross  Abstract: Directional ablation removes an aligned language model's ability to refuse by projecting a single "refusal direction" out of the weights that write the residual stream. It needs no gradient-based training and no optimization, only a few hundred contrastive prompts, which makes it the canonical white-box attack on open-weight alignment. However, it has been established only on dense models up to roughly 70B parameters. We study whether it survives the shift to frontier mixture-of-experts (MoE) models whose residual streams are no longer a single tensor and whose weights ship quantized. We apply it to GLM-5.3-Flash (320B parameters, 288 routed experts, a four-wide hyper-connection residual, block-FP8). The attack survives the architecture, but what it reaches is no longer where a reader of the original recipe would look for it. Editing the attention, dense and routed-expert writers on their own removes 0.039, 0.016 and 0.148 of refusal r
    
[^67]: LogiScope-VQA：面向工业场景物流危险识别的视觉语言模型基准测试

    LogiScope-VQA: Benchmarking Vision-Language Models for Logistics Hazard Identification in Industrial Scenarios

    [https://arxiv.org/abs/2609.09790](https://arxiv.org/abs/2609.09790)

    该论文构建了基于真实物流园区数据的多模态基准测试LogiScope-VQA，通过2,476张图像、2,918个视频和10,274个人工精心标注的VQA，围绕工业要素感知、仓储知识理解和潜在风险推理三大主题的39个子任务，系统评估主流大模型在物流危险识别中的实际能力。

    

    大规模多模态模型（LMMs）在工业仓储场景的大规模部署，特别要求模型具备人类专家级别的、面向危险的感知、理解和推理能力。然而，与商业条款紧密绑定的真实工业数据稀缺，严重阻碍了该领域的进一步发展。为弥合这一差距，我们构建了LogiScope-VQA，以研究主流LMMs在真实物流运营中的实际适用性。LogiScope-VQA包含主要来源于真实物流园区的2,476张图像和2,918个视频，以及由人工标注者精心策划并验证的10,274个视觉问答（VQA）。基于18个核心物体和20种风险类型，我们设计了39个子任务，涵盖三大主题：工业要素感知、仓储知识理解和潜在风险推理。此外，我们引入了动态思考预算配置，并（摘要在此处被截断）

    arXiv:2609.09790v1 Announce Type: cross  Abstract: Large Multimodal Models (LMMs) large-scale deployment in industrial warehouse settings specifically necessitates that models exhibit human-expert-level hazard-oriented perception, understanding, and reasoning capabilities. However, the scarcity of real industrial data, tightly coupled to commercial terms, significantly hampers further advancement. To bridge this gap, we curate LogiScope-VQA to investigate the practical applicability of mainstream LMMs in real-world logistics operations. LogiScope-VQA comprises 2,476 images and 2,918 videos primarily sourced from real-world logistics parks, along with 10,274 VQAs meticulously curated and validated by human annotators. Grounded in 18 core objects and 20 risk types, we devise 39 subtasks aligned with three principal themes: industrial element perception, warehouse knowledge understanding, and potential risk reasoning. Furthermore, we incorporate dynamic thinking-budget configurations and 
    
[^68]: Pairit：一个人机协作实时实验平台

    Pairit: A Platform for Live Experiments on Human-AI Collaboration

    [https://arxiv.org/abs/2609.09789](https://arxiv.org/abs/2609.09789)

    Pairit是一个基于单一YAML配置文件的在线实验平台，允许研究人员声明可执行的实验图，并在实时会话中组合任意数量的人类参与者与AI智能体，用于测试人机组织设计与干预措施。

    

    人工智能时代的组织设计需要实验方法来测试人机群体如何进行协调、委派和决策。现有的可编程平台能够协调实时的人际会话或实时的人机聊天，但研究人员难以在单一可审计的配置中声明实验协议，使AI参与者既能进行交流，又能在共享工作上采取行动。本文介绍了Pairit，这是一个在线平台，旨在促进设计、测试和部署用于检验人机组织设计与干预措施的实验。通过单一的YAML配置文件，研究人员可以声明一个可执行的实验图（包括页面、路由、随机化、匹配、聊天、共享工作区、服务器托管的智能体、调查问卷、计时器和自定义HTML组件），并在实时会话中组合任意数量的人类参与者和AI智能体。我们通过多项实验验证了该平台的可行性。

    arXiv:2609.09789v1 Announce Type: cross  Abstract: Organizational design in the era of artificial intelligence requires experimental methods that can test how human-AI groups coordinate, delegate, and make decisions. Programmable platforms coordinate live human-to-human sessions or real-time human-AI chat, but researchers cannot easily declare experiment protocols in which AI participants both communicate and act on shared work within one auditable configuration. Here we introduce Pairit, an online platform that facilitates the design, testing, and deployment of experiments that test human-AI organizational designs and interventions. Through a single YAML configuration file, researchers declare an executable experiment graph (pages, routing, randomization, matchmaking, chat, shared workspaces, server-hosted agents, surveys, timers, and custom HTML components) and combine any number of humans and AI agents in live sessions. We have validated the feasibility of the platform through multi
    
[^69]: BRACE：异步强化学习中过期评论家的锚定贝尔曼残差校正

    BRACE: Anchored Bellman-Residual Correction for Stale Critics in Asynchronous RL

    [https://arxiv.org/abs/2609.09783](https://arxiv.org/abs/2609.09783)

    BRACE通过将贝尔曼校正范围限制在策略token前缀并对超出部分锚定常数权重蒙特卡洛尾部，解决了异步强化学习中评论家因策略滞后产生的偏差，在BrowseComp-Plus上比最强基线提升2.4%且每步快2.46倍。

    

    异步强化学习已成为扩展语言模型训练的标准方式，但由此产生的策略滞后会使评论家偏向过时的行为策略。现有的异步LLM训练工作只校正了执行器，而未解决这一偏差；同时，经典强化学习中的离策略价值校正也无法直接应用于长视野智能体任务，因为过短的校正范围会使回归目标无法获得奖励信号，而过长的校正范围则会使重要性比率的乘积随轨迹长度呈指数级漂移。我们提出BRACE，一种针对过时价值模型的锚定贝尔曼残差校正方法。BRACE将校正范围限制在策略token的前缀之内，并在其之外锚定一个常数权重的蒙特卡洛尾部，从而将策略校正与奖励传播分离开来。BRACE在BrowseComp-Plus上比最强基线的mean@1提升了2.4%，每步运行速度快2.46倍。

    arXiv:2609.09783v1 Announce Type: cross  Abstract: Asynchronous reinforcement learning has become the standard way to scale training for language models, but the resulting policy lag biases the critic toward the stale behavior policy. Existing work on asynchronous LLM training corrects the actor and leaves this bias unaddressed, while the off-policy value correction of classical RL does not carry over to long-horizon agentic tasks, since a short correction horizon leaves the regression target free of the reward and a long one lets the product of importance ratios drift exponentially with the trajectory length. We propose BRACE, an anchored Bellman-residual correction for stale value models. BRACE bounds the correction horizon to a prefix of policy tokens and anchors a constant-weight Monte-Carlo tail beyond it, which separates policy correction from reward propagation. BRACE improves mean@1 on BrowseComp-Plus by $2.4\%$ over the strongest baseline, runs $2.46\times$ faster per step tha
    
[^70]: 承载证明的认知：以现实结算的奖励弥合验证鸿沟

    Proof-Carrying Cognition: Closing the Verification Gap with Reality-Settled Reward

    [https://arxiv.org/abs/2609.09776](https://arxiv.org/abs/2609.09776)

    论文指出“验证鸿沟”是语言模型推理领域的核心瓶颈，从理论上证明验证器与真值的相关性ρ是测试时计算与能力的精确交换率，并通过实验证明不可靠验证器在优化压力下会失效，而以现实为锚的“现实结算奖励”机制能够持续可靠地提升推理能力。

    

    前沿语言模型推理能力的提升来自对推理轨迹的强化学习，且集中于那些拥有廉价、可靠验证器的领域。我们认为该领域的关键约束是验证鸿沟：在形式化领域之外，缺乏可扩展且不可被腐化的推理奖励。本文做出四项贡献。(1) 理论：在best-of-N选择的联合高斯模型中，验证器与真值之间的相关性ρ是测试时计算与模型能力之间的精确交换率，而不可靠的验证器需付出多项式量级的惩罚N^(1/ρ²)；一种无边距的copula形式能以4%的中位数误差预测真实LLM评判者的实际可靠性。(2) 示范：在具有可执行真值的程序合成测试平台中（包括一项预注册的规模化重复实验），随着优化规模的增长，不可靠验证器的“压力下可靠性”会退化（在N=4096时从0.94降至0.32），而可靠验证器则单调提升；以现实为锚的结算优于冻结的（摘要在此处被截断）

    arXiv:2609.09776v1 Announce Type: new  Abstract: Frontier gains in language-model reasoning come from reinforcement learning on reasoning traces and are concentrated in domains with a cheap, sound verifier. We argue the field's binding constraint is the verification gap: no scalable, incorruptible reward for reasoning outside formal domains. We make four contributions. (1) Theory: in a joint-Gaussian model of best-of-N selection, verifier-gold correlation rho is the exact exchange rate between test-time compute and capability, and an unsound verifier pays a polynomial penalty N^(1/rho^2); a margin-free copula form predicts realized soundness of real LLM judges to 4% median error. (2) Demonstration: in program-synthesis testbeds with executable ground truth, including a pre-registered scaled replication, unsound verifiers lose Soundness-under-Pressure as optimization grows (0.94 to 0.32 at N=4096) while a sound verifier improves monotonically; reality-anchored settlement beats a frozen 
    
[^71]: 变化环境下的程序性记忆：受控网络任务中的复用与干扰

    Procedural Memory Under Change: Reuse and Interference in Controlled Web Tasks

    [https://arxiv.org/abs/2609.09774](https://arxiv.org/abs/2609.09774)

    该论文通过结合BrowserGym TimeWarp的回顾性界面适配案例与合成购物任务上的受控冻结记忆对比实验，研究了当环境变化使存储的程序性例程不再适用时，语言智能体的记忆复用如何产生干扰效应。

    

    程序性记忆使语言智能体能够复用成功的操作例程，但复用的前提是所存储的例程仍然适用。我们研究了当这一前提被有意破坏时会发生什么。该研究将BrowserGym TimeWarp中一个回顾性、人工辅助的界面适配案例，与在合成购物决策上进行的受控冻结记忆对比实验相结合。在有记录的WebShop V1-V6开发路径中，界面特定的代码被进行了适配，而单独存储的高层程序据报告并未发生变化；这一阶段不构成自主记忆智能体评估。在受控阶段，早期试点产生了一个任务，在该任务上两个带记忆的条件选择了更昂贵的商品，而无记忆条件则选择了参考最低价商品。后续探测未能确立反复出现的行序或身份绑定模式。随后我们测试了四种形式的不匹配情况：数量变化、不同的商品……（摘要在此处被截断）

    arXiv:2609.09774v1 Announce Type: new  Abstract: Procedural memory lets language agents reuse successful routines, but reuse presumes that a stored routine remains applicable. We study what happens when that presumption is deliberately violated. The study combines a retrospective, human-assisted interface-adaptation case from BrowserGym TimeWarp with controlled frozen-memory comparisons on synthetic shopping decisions. During the documented WebShop V1-V6 development path, interface-specific code was adapted while the separately stored high-level procedure was not reported to change; this phase does not constitute an autonomous memory-agent evaluation. In the controlled phase, an early pilot produced one task on which two memory conditions selected a more expensive item while the no-memory condition selected the reference minimum. Follow-up probes did not establish a recurring row-order or identity-binding pattern. We then tested four forms of mismatch: changed quantities, a different e
    
[^72]: 微调感知KV缓存拼接的模型还是重新计算KV缓存？为什么不同时兼顾两者？

    Fine-Tuning a KV Cache Concatenation-Aware Model or Recomputing KV Caches? Why Not Both?

    [https://arxiv.org/abs/2609.09768](https://arxiv.org/abs/2609.09768)

    提出将KV缓存拼接感知的模型微调与选择性KV缓存重计算相结合的方法，在保持低首token延迟的同时显著提升RAG系统长上下文输入的回复准确性。

    

    在检索增强生成（RAG）系统中，大量检索到的文本块被拼接起来构成输入上下文，以便用户能够基于外部知识获得高质量的回复。因此，输入上下文的长度大幅增加，导致预填充工作负载增大，进而使首token生成时间（TTFT）变长。虽然先前重用预计算键值（KV）缓存的工作有效地降低了长上下文输入的TTFT，但当输入上下文变得非常长时，回复质量是否得到保持仍不清楚。在本文中，我们提出了一种组合方法：(i) 在考虑KV缓存拼接的情况下对模型进行微调，以及 选择性地重新计算一部分KV缓存。通过同时应用这两种技术，我们证明了该方法能够提升长上下文输入的准确性。在RULER基准上的实验表明，对于124k token的输入，我们的方法将RULER分数提高了

    arXiv:2609.09768v1 Announce Type: cross  Abstract: In Retrieval-Augmented Generation (RAG) systems, a large number of retrieved chunks are concatenated to form the input context so that users can receive high-quality responses based on external knowledge. As a result, the input context length increases substantially, leading to a larger prefill workload and, in turn, a longer time to first token (TTFT). While previous works that reuse precomputed key-value (KV) caches effectively reduce TTFT for long-context inputs, it remains unclear whether response quality is preserved when the input context becomes very long. In this paper, we propose a combined approach that (i) fine-tunes the model while taking KV cache concatenation into account and (ii) selectively recomputes a subset of the KV caches. By applying both techniques, we demonstrate improved accuracy for long-context inputs. Experiments on the RULER benchmark show that, for a 124k-token input, our method improves the RULER score by
    
[^73]: LexAgentHallu：一个用于剖析法律智能体幻觉的分层基准测试

    LexAgentHallu: A Hierarchical Benchmark for Profiling Hallucinations in Legal Agents

    [https://arxiv.org/abs/2609.09754](https://arxiv.org/abs/2609.09754)

    提出了LexAgentHallu——首个通过双层幻觉分类体系（7个高级类别、27个细粒度子类）和3414个专家构建实例，在多步骤轨迹中系统评估法律智能体幻觉程度与产生方式的分层基准测试。

    

    随着大语言模型越来越多地被部署为工具增强的法律智能体，它们引入了“智能体幻觉”问题，即工具调用和推理错误级联放大，导致捏造判决结果和错误引用法律权威。然而，现有的法律基准测试仅使用结果层面的指标评估单轮问答，而智能体幻觉基准测试则缺乏法律领域特定的诊断能力，两者都无法回答法律智能体在其执行轨迹中在何种程度上以及如何产生幻觉。为了解决这些局限性，我们提出了LexAgentHallu，一个法律智能体幻觉基准测试，旨在评估法律智能体在多步骤轨迹中失败的程度和方式。LexAgentHallu通过四阶段专家参与闭环流程构建，包含涵盖17个法律类别和6种任务类型的3414个实例。每个实例均在一个双层幻觉分类体系下进行标注，该体系包含7个高级类别和27个细粒度子类，同时涵盖……

    arXiv:2609.09754v1 Announce Type: new  Abstract: As large language models are increasingly deployed as tool-augmented legal agents, they introduce agentic hallucinations where tool-call and reasoning errors cascade into fabricated holdings and miscited authority. However, existing legal benchmarks evaluate only single-turn QA with outcome-level metrics, while agentic hallucination benchmarks lack legal-specific diagnostic capability. Neither answers to what extent and how a legal agent hallucinates along its trajectory. To address these limitations, we introduce LexAgentHallu, a legal agentic hallucination benchmark designed to evaluate to what extent and how legal agents fail along multi-step trajectories. Built through a four-stage expert-in-the-loop pipeline, LexAgentHallu contains 3414 instances across 17 legal categories and 6 task types. Each instance is annotated under a dual-layer hallucination taxonomy of 7 high-level categories and 27 fine-grained subclasses, covering both su
    
[^74]: HiRAD：一种灵活的大规模AGV路径规划系统

    HiRAD: A Flexible Large-Scale AGV Routing System

    [https://arxiv.org/abs/2609.09752](https://arxiv.org/abs/2609.09752)

    提出HiRAD，一种具有实时保障的分层强化学习框架，通过步骤级时空表示实现连续空间的大规模AGV路径规划，克服了经典求解器组合复杂度爆炸以及现有RL方法收敛慢、推理延迟高的问题。

    

    自动导引车（AGV）显著提升了仓库吞吐量，但大规模AGV车队的路径规划仍然具有挑战性。经典的多智能体路径规划求解器存在组合复杂度爆炸和超二次运行时间的问题，并且依赖于理想化的网格或分段线性运动模型，与真实世界的运动学不匹配。近期的强化学习（RL）解决方案通过去中心化的智能体策略提高了灵活性，但依赖于离散化的时空表示，需要数百万次回合才能收敛，并且在每一步都需要全地图观测，这导致模型庞大、收敛缓慢和推理延迟过高，无法满足实时工业控制的约束。为解决这些瓶颈，我们提出了HiRAD，一个具有实时保障的面向连续空间AGV路径规划的分层强化学习框架：（1）一种步骤级的时空表示，将连续运动转换为……（原文摘要在此处截断）

    arXiv:2609.09752v1 Announce Type: cross  Abstract: Automatic Guided Vehicles (AGVs) substantially boost warehouse throughput, but routing large-scale AGV fleets remains challenging. Classical Multi-Agent Pathfinding solvers suffer from exploding combinatorial complexity and super-quadratic runtime, while relying on idealized grid or piecewise-linear motion models that mismatch real-world kinematics. Recent Reinforcement Learning (RL) solutions improve flexibility via decentralized agent policies but depend on discretized spatiotemporal representations, require millions of episodes to converge, and incur full-map observation at every step, which leads to large models, slow convergence, and high inference latency that violates real-time industrial control constraints. To address these bottlenecks, we propose HiRAD, a hierarchical RL framework for continuous-space AGV routing with real-time guarantees: (1) a step-level spatiotemporal representation that translates continuous motion into a
    
[^75]: 面向引导式测试时自适应的图像原型蒸馏

    Distilling Image Prototypes for Guided Test-Time Adaptation

    [https://arxiv.org/abs/2609.09737](https://arxiv.org/abs/2609.09737)

    提出DIPTTA框架，引入由紧凑合成图像构成的蒸馏图像原型（DIP）作为源知识的动态可再生锚点，通过动态特征回放机制解决测试时自适应中伪标签误差累积与灾难性遗忘两大难题。

    

    测试时自适应（TTA）能够增强模型对分布偏移的鲁棒性，但面临两个关键挑战：噪声伪标签导致的误差累积，以及源知识的灾难性遗忘。旨在缓解误差累积的基于不确定性的方法往往会产生过度自信或计算代价高昂的估计，而通过原型回放来防止遗忘的策略则依赖于静态表示，随着模型的自适应这些表示容易发生错位。为解决这些问题，本文提出了一种新颖的框架——蒸馏图像原型引导的测试时自适应（DIPTTA）。所提方法的核心是引入蒸馏图像原型（DIP），这是一组紧凑的合成图像集合，作为源知识的动态且可再生的锚点。该原型实现了一种动态特征回放机制，能够持续生成与（摘要在此处被截断）……

    arXiv:2609.09737v1 Announce Type: cross  Abstract: Test-Time Adaptation (TTA) enhances the robustness of models against distribution shifts but faces two critical challenges: error accumulation from noisy pseudo-labels and catastrophic forgetting of source knowledge. Uncertainty-based approaches designed to mitigate error accumulation often yield overconfident or computationally expensive estimates, while strategies intended to prevent forgetting via prototype replay rely on static representations that easily become misaligned as the model adapts. To address these issues, this paper proposes a novel framework, Distilling Image Prototype for Guided Test-Time Adaptation (DIPTTA). The core of the proposed approach is the introduction of a Distill Image Prototype (DIP), a compact set of synthetic images that serves as a dynamic and regenerative anchor of source knowledge. This prototype enables a dynamic feature replay mechanism that continuously generates feature prototypes aligned with t
    
[^76]: 人工智能能否通过早期网络欺凌检测支持医疗保健与心理健康？情感感知AI对主动式在线安全的影响

    Can Artificial Intelligence Support Healthcare and Mental Health Through Early Cyberbullying Detection ? The Impact of Emotion-Aware AI on Proactive Online Safety

    [https://arxiv.org/abs/2609.09735](https://arxiv.org/abs/2609.09735)

    本文提出CareGuard早期预警框架，通过融合零样本语义标注、微调Transformer模型以及情感感知过滤机制，实现对网络欺凌内容的高效早期检测，从而支持医疗保健驱动的心理健康保护与主动式在线安全。

    

    医疗保健系统、心理健康和公共福祉正日益受到网络欺凌和有害在线互动的影响。本文提出了CareGuard，一个早期预警框架，旨在通过使用先进的自然语言处理技术检测网络欺凌相关内容，支持以医疗保健为导向的心理健康保护和主动式在线安全。CareGuard将零样本语义标注与微调的基于Transformer的模型（包括BERT、DistilBERT和RoBERTa）相结合，以实现对敏感网络欺凌类别的鲁棒且具备上下文感知能力的分类。为了提高效率并减少面向医疗保健监控环境中不必要的计算，该框架引入了情感感知过滤机制以及基于余弦相似度的语义筛选，使系统能够专注于语义相关且情感显著的内容。在基准数据集上的实验结果表明（摘要在此处截断）……

    arXiv:2609.09735v1 Announce Type: cross  Abstract: Healthcare systems, mental health, and public well-being are increasingly affected by cyberbullying and harmful online interactions. This paper presents CareGuard, an early-warning framework designed to support healthcare-driven mental health protection and proactive online safety through the detection of cyberbullying-related content using advanced natural language processing techniques. CareGuard integrates zero-shot semantic labeling with fine-tuned transformer-based models, including BERT, DistilBERT, and RoBERTa, to enable robust and context-aware classification across sensitive cyberbullying categories. To improve efficiency and reduce unnecessary computation in healthcare-oriented monitoring settings, the framework incorporates an emotion-aware filtering mechanism alongside cosine similarity-based semantic screening, allowing the system to focus on semantically relevant and emotionally salient content. Experimental results on be
    
[^77]: 哪些Token才是SFT真正应该学习的？从Token修剪视角看数学推理

    Which Tokens Should SFT Actually Learn? A Token-Trimming Perspective on Mathematical Reasoning

    [https://arxiv.org/abs/2609.09707](https://arxiv.org/abs/2609.09707)

    该论文提出TrimSFT方法，通过基于金标准token与最强竞争者之间logit差距的高斯权重对SFT损失进行重加权，修剪掉已掌握和弱支持token的监督信号，将学习集中在中间logit差距区域的token上，从而改善数学推理的训练动态。

    

    监督微调（SFT）对所有目标token应用统一的交叉熵损失，尽管不同的token为数学推理提供的学习信号并不相等。这种统一处理可能会过度锐化已经掌握的token，同时放大对不确定的、低置信度token的学习压力，导致次优的训练动态。我们提出了修剪Logit差距SFT（TrimSFT），这是一种简单的token级重加权方法，根据金标准token与其最强竞争者之间的logit差距来缩放SFT损失。TrimSFT从两个极端修剪监督信号：已经掌握的token（大logit差距）和当前模型弱支持的token（小或负logit差距），从而将学习集中在两者之间的中间logit差距区域。我们用以边距m为中心、带宽为τ的高斯权重来实现这一原则，无需参考模型或额外的前向传播。

    arXiv:2609.09707v1 Announce Type: new  Abstract: Supervised fine-tuning (SFT) applies a uniform cross-entropy loss to all target tokens, even though different tokens provide unequal learning signals for mathematical reasoning. This uniform treatment can over-sharpen already mastered tokens while amplifying learning pressure on uncertain, low-confidence tokens, leading to suboptimal training dynamics. We propose Trimmed Logit-Gap SFT (TrimSFT), a simple token-level reweighting method that scales the SFT loss according to the logit gap between the gold token and its strongest competitor. TrimSFT trims supervision away from both extremes: tokens already mastered (large logit gap) and tokens weakly supported by the current model (small or negative logit gap), concentrating learning within an intermediate logit-gap region between them. We instantiate this principle with a Gaussian weight centered at margin m with bandwidth {\tau}, requiring no reference model or additional forward pass. We 
    
[^78]: 决策偏移、标签功能丧失与正确性门控多教师蒸馏中的不确定依据审计

    Decision Shifts, Lost Label Functionality, and an Inconclusive Grounding Audit in Correctness-Gated Multi-Teacher Distillation

    [https://arxiv.org/abs/2609.09702](https://arxiv.org/abs/2609.09702)

    该论文通过固定实验发现，正确性门控的多教师蒸馏虽能提升准确率和安全性指标，但会导致某些标签功能完全丧失（如Refuted召回率为零）和决策偏移，说明决策正确性与依据支撑是两个不同的优化目标。

    

    候选决策正确性与推理依据支撑是两个不同的目标。我们在一个固定实验中研究了正确性门控的多教师蒸馏。八个实验组共享4,330个源数据、一个63.9M参数的学生模型、12,990个优化样本、406次更新、证据输入和一个解码器；其中七个基于教师的实验组使用同一个固定的三响应池。三个随机种子在267个留出样本上进行评估。相对于未过滤的蒸馏，正确性加权组在准确率上相差+0.1660（95%观测矩阵区间[0.0670, 0.2455]），五标签宏平均F1相差+0.1323（[0.0916, 0.1731]），任务定义的条件不安全动作率相差-0.4979（[-0.5926, -0.3686]）。这些偏移并不意味着行为整体更优。源标签SFT具有最高的平均宏平均F1（0.586）。加权组在每个种子上的Refuted类别召回率均为零，且两个种子将全部167个声明样本都判定为NotEnoughInfo。在……（原文截断）处进行了一次修正可用性的审计……

    arXiv:2609.09702v1 Announce Type: new  Abstract: Candidate decision correctness and rationale grounding are different objectives. We examine correctness-gated multi-teacher distillation in a fixed experiment. Eight arms share 4,330 sources, a 63.9M-parameter student, 12,990 optimization rows, 406 updates, evidence inputs, and a decoder; seven teacher-based arms use one fixed three-response pool. Three seeds are evaluated on 267 held-out examples. Relative to unfiltered distillation, the correctness-weighted arm differed in accuracy by +0.1660 (95% observed-matrix interval [0.0670, 0.2455]), five-label macro-F1 by +0.1323 ([0.0916, 0.1731]), and task-defined conditional unsafe-action rate by -0.4979 ([-0.5926, -0.3686]). These shifts do not imply uniformly better behavior. Source-label SFT had the highest mean macro-F1 (0.586). The weighted arm had zero Refuted recall in every seed, and two seeds assigned NotEnoughInfo to all 167 claim examples. In an availability-amended audit at one r
    
[^79]: 面向免训练图结构攻击防御的核复杂度边净化方法

    Kernel-Complexity Edge Sanitization for Training-Free Defense against Structural Graph Attacks

    [https://arxiv.org/abs/2609.09698](https://arxiv.org/abs/2609.09698)

    提出了一种无需训练、与模型无关的图结构攻击防御框架 KCES，基于图核复杂度定义边分数以识别并剪除富集对抗扰动的高风险边。

    

    图神经网络（GNN）已在众多应用中取得显著成功，但其仍然极易受到恶意扰动图结构的对抗攻击。现有防御方法往往缺乏严格的理论依据，依赖于针对特定攻击的启发式规则，或需要代价高昂的重训练过程（如对抗训练）。为解决这些局限性，我们提出了核复杂度边净化（KCES），这是一个无需训练且与模型无关的结构攻击防御框架。KCES 建立在图核复杂度（GKC）之上，这是一个具有原则性的度量指标，源自于 GNN 测试误差泛化上界中出现的图格拉姆矩阵。基于该上界，我们定义了一种边特定的 KC 分数，通过每条边所引起的 GKC 变化来量化其结构影响力。KCES 随后识别并剪除高 KC 边，这些边在经验上富集了对抗扰动……

    arXiv:2609.09698v1 Announce Type: cross  Abstract: Graph Neural Networks (GNNs) have achieved remarkable success across diverse applications, yet they remain highly vulnerable to adversarial attacks that maliciously perturb graph structure. Existing defenses often lack rigorous theoretical grounding, rely on attack-specific heuristics, or require costly retraining procedures such as adversarial training. To address these limitations, we propose Kernel-Complexity Edge Sanitization (KCES), a training-free and model-agnostic framework for defending against structural attacks. KCES is built upon Graph Kernel Complexity (GKC), a principled metric derived from the graph Gram matrix that appears in a generalization upper bound on the GNN test error. From this bound, we define an edge-specific KC score that quantifies each edge's structural influence via its induced change in GKC. KCES then identifies and prunes high-KC edges, which are empirically enriched with adversarial perturbations under
    
[^80]: 当审计员捏造事实：大语言模型检测植入文档污染中的批次规模退化与自信幻觉

    When Auditors Fabricate: Batch-Size Degradation and Confident Hallucination in LLM Detection of Planted Document Contamination

    [https://arxiv.org/abs/2609.09696](https://arxiv.org/abs/2609.09696)

    大语言模型在单文档和小批量污染检测中表现尚可（50%-60%），但在大批量处理时检测率骤降至2.8%，且其失败方式不是承认无法处理，而是自信地捏造包括虚假污染项在内的检测结果。

    

    大语言模型越来越多地被提议作为文档质量的自动化审计工具，但它们作为植入错误检测器的可靠性却缺乏充分表征。我们构建了一个包含150篇学术论文的受污染语料库，涵盖供应链管理和医学研究领域，注入了450个已知污染项，分为三种类型：排版损坏、语义反转和荒谬的脱离语境插入。随后，我们在三种规模递增的提示机制下（单文档、小批量和大批量），评估了Google Gemini 3.0 Pro在60份文档中恢复包含180个污染项的答案密钥子集的能力。检测在小规模下保持有效，随后急剧崩溃：单文档恢复率为50%，小批量为60%，大批量仅为2.8%。大规模下的失败模式并非放弃检测，而是捏造结果。模型没有报告处理不完整，而是产生了自信的发现，包括自行编造的污染项……

    arXiv:2609.09696v1 Announce Type: new  Abstract: Large language models are increasingly proposed as automated auditors of document quality, yet their reliability as detectors of planted errors is poorly characterised. We construct a contaminated corpus of 150 academic papers spanning supply chain management and medical research, injecting 450 known contaminants of three types: typographical corruption, semantic reversal, and absurd out-of-context insertion. We then evaluate Google Gemini 3.0 Pro's ability to recover a 180-contaminant answer-key subset across 60 documents under three prompting regimes of increasing scale: single document, small batch, and large batch. Detection holds at small scale and then collapses: 50% recovery on single documents, 60% on small batches, and 2.8% on large batches. The failure mode at scale is not abstention but fabrication. Rather than reporting incomplete processing, the model produced confident findings including invented contaminants of its own, ab
    
[^81]: CT-SAFR：面向自主机器人的安全可解释思维链推理：一个用于可信赖AI驱动机器人决策的多层验证框架

    CT-SAFR: Safe and Interpretable Chain-of-Thought Reasoning for Autonomous Robots: A Multi-Layered Verification Framework for Trustworthy AI-Driven Robotic Decision Making

    [https://arxiv.org/abs/2609.09692](https://arxiv.org/abs/2609.09692)

    本文提出CT-SAFR多层验证框架，以低于500毫秒的延迟实现94.2%的幻觉检测率，将机器人不安全推理输出减少87%，为自主机器人的可信思维链推理提供安全保障。

    

    思维链提示使大型语言模型能够进行显式的、逐步的推理，为高度智能化的自主机器人创造了机会。然而，近期研究显示，推理模型仅有25-39%的时间会真实表达其实际决策过程，且在复杂任务上忠实度下降44%。本文提出了CT-SAFR（面向机器人的思维链安全性与忠实度框架），这是一个多层验证框架，实现了94.2%的幻觉检测率（n = 500，95%置信区间：91.8-95.9%），同时延迟低于500毫秒。通过仓库机器人的案例研究，本工作展示了不安全推理输出减少87%（p < 0.001），并为具备推理能力的自主机器人的负责任部署提供了建议。

    arXiv:2609.09692v1 Announce Type: cross  Abstract: Chain-of-Thought (CoT) prompting enables LLMs to perform explicit, step-by-step reasoning, creating opportunities for sophisticated autonomous robots. However, recent research reveals that reasoning models verbalize their actual decision processes only 25-39% of the time, with faithfulness degrading 44% on complex tasks. This paper presents CT-SAFR (Chain-of-Thought Safety and Faithfulness for Robotics), a multi-layered verification framework achieving 94.2% hallucination detection (n = 500, 95% CI: 91.8-95.9%) with sub-500ms latency. Through a warehouse robot case study, this work demonstrates 87% reduction in unsafe reasoning outputs (p < 0.001) and provides recommendations for responsible deployment of reasoning-capable autonomous robots.
    
[^82]: 循环 GPT-BERT：在小规模语言建模中以计算换取参数

    Looped GPT-BERT: Trading Parameters for Computation in Small Language Modeling

    [https://arxiv.org/abs/2609.09691](https://arxiv.org/abs/2609.09691)

    该研究提出循环 GPT-BERT，通过深度参数共享让 4 个物理层循环遍历 12 次，在 BabyLM 2026 Strict-small 设定下以仅 1218 万参数在 BLiMP 和 GLUE 等语言学与下游任务指标上取得了与更大参数量的 GPT-2 和 GPT-BERT 基线相当的性能。

    

    当训练数据有限时，增加参数量并非提升语言模型性能的唯一途径。一组较小的参数经过反复使用，同样能够取得可比的性能。我们在 BabyLM 2026 Strict-small 设定下研究了循环 GPT-BERT（Looped GPT-BERT），将 GPT-BERT 的掩码下一词预测与因果语言建模目标同逐层深度参数共享相结合。我们在一个预处理后的 748 万词英语语料库上进行训练，并比较了目标函数比例、非循环与循环架构以及循环次数。我们最终的 4×12 模型使用四个物理层进行十二次循环遍历，共包含 1218 万参数。BabyLM 2026 排行榜报告其总体平均分为 35.42，NLP 平均分为 48.48。与公开的 BabyLM 10M Strict-small GPT-2 和 GPT-BERT 基线相比，该模型在包括 BLiMP 和 GLUE 在内的多项语言学与下游任务指标上，以更少的参数取得了可比的性能。

    arXiv:2609.09691v1 Announce Type: new  Abstract: When training data are limited, increasing parameter count is not the only way to improve language-model performance. A small parameter set, when repeatedly applied, can also deliver comparable performance. We study Looped GPT-BERT in the BabyLM 2026 Strict-small setting, combining GPT-BERT's masked next-token and causal language-modeling objectives with depth-wise parameter sharing. We train on a preprocessed 7.48M-word English corpus and compare objective ratios, non-looped and looped architectures, and loop counts. Our final $4\times12$ model uses four physical layers for twelve recurrent traversals and contains 12.18M parameters. The BabyLM 2026 leaderboard reports an Overall Average of 35.42 and an NLP Average of 48.48. Compared with public BabyLM 10M Strict-small GPT-2 and GPT-BERT baselines, it achieves comparable performance on selected linguistic and downstream metrics, including BLiMP and GLUE, with fewer parameters. The loop a
    
[^83]: 哪些医学问题值得生成推理依据？面向鲁棒问答的扰动敏感选择方法

    Which Medical Questions Deserve Rationales? Perturbation-Sensitive Selection for Robust QA

    [https://arxiv.org/abs/2609.09684](https://arxiv.org/abs/2609.09684)

    该论文提出RMS-RSP方法，通过仅在推理依据token处扰动隐藏状态并测量答案与干扰项之间裕度的变化，在固定token预算下智能筛选哪些已标注医学问题最值得投入推理依据监督，从而提升医学问答的鲁棒性。

    

    医学问答数据集通常包含答案标签，而高质量的推理依据仍然稀缺、含有噪声或验证成本高昂。这改变了数据获取的核心问题：我们不再追问哪些问题应该被标注，而是探究在固定token预算下，哪些已标注的问题应该获得推理依据监督。我们研究了该问题的一个离线版本，其中候选推理依据对选择器可见，但除非被选中，否则不会用于下游训练。我们提出了均方根鲁棒性样本优先级方法（RMS-RSP），该方法仅在推理依据token处扰动隐藏状态，并测量由此导致的金标准答案与最佳干扰项之间裕度的变化。在五个医学问答数据集、MedGemma-4B-IT模型、三个训练种子、十个有预算的非RSP选择器以及一个无预算的全监督参考基线下，RMS-RSP给出了一个经过审慎限定的结果：其固定预算下的平均准确率为60.61%。

    arXiv:2609.09684v1 Announce Type: new  Abstract: Medical question-answering datasets often contain answer labels, whereas high-quality rationales remain scarce, noisy, or costly to validate. This changes the acquisition question: rather than asking which questions should be labeled, we ask which already-labeled questions should receive rationale supervision under a fixed token budget. We study an offline version of this problem in which candidate rationales are visible to the selector but withheld from downstream training unless selected. We propose root-mean-square Robustness-based Sample Prioritization (RMS-RSP), which perturbs hidden states only at rationale tokens and measures the resulting shift in the gold-versus-best-distractor margin. Across five medical QA datasets, MedGemma-4B-IT, three training seeds, ten budgeted non-RSP selectors, and an unbudgeted full-supervision reference, RMS-RSP provides a deliberately qualified result. Its locked-budget accuracy is 60.61% on average 
    
[^84]: 可以安全停止了吗？序列临床诊断智能体的风险约束停止机制

    Safe to Stop? Risk-Constrained Stopping for Sequential Clinical Diagnosis Agents

    [https://arxiv.org/abs/2609.09678](https://arxiv.org/abs/2609.09678)

    提出了Cros——一个面向序列临床诊断智能体的风险约束停止层，通过状态级错误排序与LTT精确检验提供有限样本保证，确保自主停止时的选择性诊断错误控制和最小诊断覆盖率。

    

    临床诊断智能体不仅需要决定下一步请求哪项检查，还需要决定何时做出诊断或推迟诊断。现有的智能体基准测试大多在固定或无约束的交互后评估准确率，使自主停止的可靠性问题处于隐式状态。我们提出了Cros，一个风险约束的停止层，它结合了状态级错误排序、在不相交开发集划分上的策略设计，以及LTT风格的精确检验，用于对完整的序列策略进行选择性诊断错误和最小自主覆盖率的检验。其有限样本保证要求候选族、检验规则以及任何随机化过程在访问校准标签之前被冻结。在一个包含1,834个回合的基于MIMIC的腹痛基准测试中，完整排序器实现了探索性状态错误AUROC 0.853，相比之下最大类别概率为0.715，骨干模型的原生停止分数仅为0.552。在之前查看过的367个回合评估集上，解析平均（摘要在此处截断）

    arXiv:2609.09678v1 Announce Type: new  Abstract: Clinical diagnosis agents must decide not only what test to request next, but also when to diagnose or defer. Existing agent benchmarks largely evaluate accuracy after fixed or unconstrained interaction, leaving autonomous stopping reliability implicit. We present Cros, a risk-constrained stopping layer combining state-wise error ranking, policy design on disjoint development splits, and LTT-style exact tests of selective diagnostic error and minimum autonomous coverage for complete sequential policies. Its finite-sample guarantee requires the candidate family, testing rule, and any randomization to be frozen before calibration labels are accessed. On a 1,834-episode MIMIC-derived abdominal-pain benchmark, the full ranker achieves exploratory state-error AUROC 0.853, compared with 0.715 for maximum class probability and 0.552 for the backbone's native stop score. On the previously viewed 367-episode evaluation split, analytically averagi
    
[^85]: 介绍Consort：一个基于规范优先、在真实数据库分支上实施强制测试驱动开发的智能体框架

    Introducing Consort: A Spec-First Agent Framework for Enforced, Test-Driven Development on Live Database Branches

    [https://arxiv.org/abs/2609.09671](https://arxiv.org/abs/2609.09671)

    本文提出Consort框架，通过智能体无法修改的强制控制手段（如确定性编排器、人工批准门禁、不可变测试及真实分支数据库验证）来执行工程纪律，确保智能体编写代码的整洁性、正确性和可维护性。

    

    当智能体（Agent）编写代码时，开发框架就成为了对一个非确定性工作者的控制系统。自2025年以来，规范优先、智能体驱动的框架迅速兴起；可安装的框架，包括GitHub Spec Kit、obra/superpowers、BMAD和GSD，以及我们自己的框架，都通过规范或持久化的规划工件来捕获意图。由于它们在预先捕获意图这一点上达成共识，区分它们的关键在于各自如何强制执行工程纪律，以确保智能体编写的代码整洁、正确且可维护。每个框架都会以某种方式强制执行这种纪律；它们的区别在于方式。我们归纳了三种模式：通过说服来强制执行（模型可能忽略的提示词约束）、通过前置结构来强制执行（先有强规范，再进行受信任的构建），以及通过智能体无法编辑的控制手段来强制执行（确定性编排器、需人工批准的门禁、不可变的测试，以及必须通过真实分支数据库验证的绿色结果）。我们提出了……

    arXiv:2609.09671v1 Announce Type: new  Abstract: When an agent writes code, the development framework becomes the control system for a non-deterministic worker. Spec-first, agent-driven frameworks have gained rapid traction since 2025; the installable ones, GitHub Spec Kit, obra/superpowers, BMAD, and GSD, and our own, all capture intent through a specification or durable planning artifacts. Since they agree on capturing intent up front, what separates them is how each enforces the engineering discipline that keeps agent-written code clean, correct, and maintainable. Every framework enforces that discipline somehow; they differ in how. We characterize three modes: enforcement by persuasion (prompt discipline the model may ignore), by front-loaded structure (strong specs, then a trusted build), and through controls the agent cannot edit (a deterministic orchestrator, human-approved gates, immutable tests, and a green result that must pass against a live, branched database). We introduce
    
[^86]: PRAGMA：评估终身对话中基于记忆对齐的个性化引导

    PRAGMA: Evaluating Personalized Guidance with Memory Alignment in Lifelong Conversations

    [https://arxiv.org/abs/2609.09664](https://arxiv.org/abs/2609.09664)

    该论文提出PRAGMA基准，用于评估终身对话中记忆系统在个性化引导任务（如推荐、规划和决策支持）上的表现，填补了现有评估仅关注事实回忆的空白。

    

    大语言模型（LLM）越来越多地被部署为与用户进行长期交互的个性化助手。随着对话变长，依赖完整的交互历史变得越来越低效且不可靠：长上下文带来巨大的计算开销，使模型难以持续识别并利用与当前请求最相关的信息。这些挑战推动了记忆系统的发展，即对用户特定的信息进行结构化组织和检索。在真实的交互场景中，用户常常寻求实用性的引导，例如推荐、规划和决策支持。与事实回忆任务不同，个性化引导需要模型整合跨越多次过往对话的信息，并对用户不断变化的偏好和经历进行推理。然而，现有的对话记忆评估主要聚焦于检索和事实回忆。为了研究……（原文摘要不完整，在"To stu"处被截断）

    arXiv:2609.09664v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as personalized assistants that interact with users over extended periods of time. As conversations grow longer, relying on full interaction histories becomes increasingly inefficient and unreliable: long contexts introduce substantial computational overhead, making it difficult for models to consistently identify and utilize the most relevant information for the current request. These challenges have motivated memory systems that structure and retrieve user-specific information. In realistic interactions, users often seek practical guidance such as recommendations, planning, and decision support. Unlike factual recall tasks, personalized guidance requires models to integrate information across multiple past conversations and reason about changing user preferences and experiences. However, existing conversational memory evaluations mainly focus on retrieval and factual recall. To stu
    
[^87]: 联邦学习中基于LT码启发的剥离解码的级联梯度反演

    Cascading Gradient Inversion via LT-Code Inspired Peeling in Federated Learning

    [https://arxiv.org/abs/2609.09659](https://arxiv.org/abs/2609.09659)

    本文将梯度反演与LT码启发的纠删除码理论联系起来，构造出能在单轮FedSGD中精确恢复整批数据及全部标签的级联攻击，突破了已知理论上限，且无需真实数据即可验证恢复结果。

    

    联邦学习共享的是模型更新而非原始数据，然而这些更新可以被反演以重构客户端的训练数据。解析重构攻击以闭式形式反演梯度，其效果随批量大小的增加而显著下降：即使是先前的单轮攻击，在攻击者完全控制网络参数的情况下，对于大小为100的批次也只能恢复大约一半，而已知的理论上限限制了任何此类方法所能恢复的内容。我们建立了梯度反演与纠删除码理论之间的联系，并利用这一联系构造了超越这些界限的攻击。我们的攻击能够从单轮FedSGD中精确恢复整个批次以及每个样本的标签，并在没有真实数据的情况下对每次恢复进行验证。在八个图像和表格基准测试中，该方法大幅超越了先前的单轮攻击。即使是仅观察诚实训练网络的被动攻击者，也能恢复94–100%的Imag（摘要在此处截断）

    arXiv:2609.09659v1 Announce Type: cross  Abstract: Federated learning shares model updates rather than raw data, yet these updates can be inverted to reconstruct the clients' training data. Analytic reconstruction attacks, which invert a gradient in closed form, degrade as the batch grows: prior single-round attacks recover only about half of a batch of size $100$ even when the attacker fully controls the network parameters, and known upper bounds limit what any such method can recover. We establish a connection between gradient inversion and the theory of erasure-correcting codes, and use it to construct attacks that exceed these bounds. Our attacks recover batches exactly, together with every sample's label, from a single FedSGD round, and certify each recovery without ground-truth data. On eight image and tabular benchmarks they outperform prior single-round attacks by a wide margin. Even a passive attacker who only observes an honestly trained network recovers $94$--$100\%$ of Imag
    
[^88]: RESCUE-BENCH：迈向关系感知的多方情感支持对话系统

    RESCUE-BENCH: Towards Relation-Aware Multi-Party Emotional Support Conversation Systems

    [https://arxiv.org/abs/2609.09657](https://arxiv.org/abs/2609.09657)

    本文提出关系感知情感支持对话新任务，并基于真实情侣与家庭访谈视频构建RESCUE基准，用于评估大语言模型的关系理解与关系敏感支持两大核心能力。

    

    现有的情感支持对话系统主要关注一对一的求助者与支持者互动以及个体情绪状态，而对多方场景中人际关系的研究尚不充分。本工作提出了关系感知情感支持对话这一新任务，旨在评估大语言模型能否捕捉并利用不断演变的关系动态，从而提供更有效的情感支持。研究者基于真实的情侣和家庭访谈对话构建了RESCUE（关系感知情感支持对话理解与评估基准），包含191个样本、7,079个标注对话轮次以及1,064.8分钟的视频。基于对社会情感与支持相关动态的丰富标注，RESCUE定义了六个任务，用于评估关系感知情感支持所需的两大核心能力：关系理解与关系敏感支持。对十个大语言模型的实验表明，当前模型……

    arXiv:2609.09657v1 Announce Type: new  Abstract: Existing emotional support conversation systems mainly focus on one-on-one seeker-supporter interactions and individual emotional states, leaving interpersonal relations in multi-party scenarios underexplored. In this work, we introduce relation-aware emotional support conversation, a new task that evaluates whether LLMs can capture and utilize the evolving dynamics of relationships to offer more effective emotional support. We construct RESCUE (Relation-aware Emotional Support Conversation Understanding and Evaluation Benchmark) from real couple and family interview conversations, containing 191 samples, 7,079 annotated turns, and 1,064.8 minutes of video. Based on rich annotations of socio-emotional and support-related dynamics, RESCUE defines six tasks that evaluate two core capabilities required for relation-aware emotional support: Relational Understanding and Relation-Sensitive Support. Experiments with ten LLMs show that current m
    
[^89]: 智能体AI的黑盒红队测试：一种分类学驱动的自动化风险发现框架

    Black-Box Red Teaming of Agentic AI: A Taxonomy-Driven Framework for Automated Risk Discovery

    [https://arxiv.org/abs/2609.09647](https://arxiv.org/abs/2609.09647)

    本文提出了一个基于七领域风险分类体系的黑盒红队测试框架SAGE-RT，通过全自动化生成对抗场景和LLM裁判评估，系统性地发现了智能体AI系统在治理、隐私和行为方面高达56%-85%的安全漏洞。

    

    智能体系统正迅速进入生产环境，它们读取不可信的输入、以真实权限调用工具并自主行动，使安全攻击面扩展到远超纯对话模型的范围。然而，标准评估仍然是单轮的，无法捕捉多步骤智能体漏洞。我们提出了一个系统性的黑盒框架，用于风险感知的智能体评估，且只需要基本的系统描述即可运行。我们的方法引入了：(1) 一个七领域的分类体系，将可观察行为映射到风险类别；(2) 全自动化的SAGE-RT红队测试，每个领域生成120个对抗性场景；(3) 使用LLM裁判并结合人工验证的评估方式。在两种智能体架构（CrewAI和AutoGen）与四个基础模型上的实证验证揭示了令人担忧的模式：平均56.25%的治理风险、多智能体配置中65%的隐私风险，以及高达85%的智能体行为漏洞。我们的黑盒方法能够有效识别……

    arXiv:2609.09647v1 Announce Type: new  Abstract: Agentic systems are rapidly moving to production, where they read untrusted inputs, call tools with real permissions, and act autonomously, expanding the security surface beyond chat-only models. Yet standard evaluations remain single-turn and fail to capture multi-step agent vulnerabilities. We present a systematic black-box framework for risk-aware agent evaluation requiring only basic system descriptions. Our approach introduces: (1) a seven-domain taxonomy mapping observable behaviors to risk categories, (2) fully automated SAGE-RT red teaming producing 120 adversarial scenarios per domain, and (3) human-validated evaluation using LLM judges. Empirical validation across two agent architectures (CrewAI and AutoGen) with four base models reveals alarming patterns: 56.25\% average governance risk, 65\% privacy risk in multi-agent configurations, and agent behavior vulnerabilities reaching 85\%. Our black-box approach effectively identif
    
[^90]: RobustSGPO：面向智能体框架演化的搜索空间控制

    RobustSGPO: Search-Space Control for Agent Harness Evolution

    [https://arxiv.org/abs/2609.09646](https://arxiv.org/abs/2609.09646)

    RobustSGPO通过权限调度、累积控制和快照保留等搜索空间控制机制，改进了基于语义梯度的智能体框架演化优化方法，在AgentX工作流上将保留任务的完成率从60.0%提升至80.0%，测试质量从3.77提升至4.14。

    

    基于语义梯度的提示词优化（SGPO）利用执行反馈来改进智能体框架，但其局部更新规则未能解决编辑范围与操作的选择问题。我们提出了RobustSGPO，该方法指定所需的编辑、构建并检查补丁，并从当前最优方案或保留的快照继续搜索。我们在AgentX头脑风暴工作流中，使用120个任务、95次运行和7,350次候选尝试，评估了权限调度、累积控制和任务族迁移。周期性的1→2→3调度比固定的最大权限高出0.28个测试分数。在2000万token的预算下，RobustSGPO将30个保留任务上的完成率从60.0%提升至80.0%，并将测试质量从3.77提升至4.14。类别保留能减少分布迁移后源任务的性能退化，而随机保留则能达到更高的目标端点。搜索空间控制通过可执行性提升优化质量……

    arXiv:2609.09646v1 Announce Type: new  Abstract: Semantic-gradient-based prompt optimization (SGPO) improves agent harnesses using execution feedback, but its local update rule leaves the choice of edit scope and operation unresolved. We introduce RobustSGPO, which specifies the requested edit, constructs and checks the patch, and continues search from either the incumbent or retained snapshots. We evaluate permission scheduling, cumulative controls, and task-family transfer in the AgentX brainstorming workflow using 120 tasks, 95 runs, and 7,350 candidate attempts. Periodic $1\to2\to3$ scheduling exceeds fixed maximum permission by 0.28 test-score points. RobustSGPO increases completion on 30 held-out tasks from 60.0% to 80.0% and improves test quality from 3.77 to 4.14 under a 20-million-token budget. Category retention reduces source-task degradation after a shift, whereas random retention reaches a higher destination endpoint. Search-space control benefits quality through executabl
    
[^91]: 物理AI能力形成的七大来源

    Seven Sources of Physical AI Capability Formation

    [https://arxiv.org/abs/2609.09627](https://arxiv.org/abs/2609.09627)

    本文提出了物理AI能力形成的七种非互斥来源分类框架——记录经验、预测建模、评估交互、代理环境、机制基础、具身耦合和进化驱动形成，填补了现有按形态或架构分类的体系无法回答“能力从何而来”这一根本问题的空白。

    

    与物理AI相关的能力可能源自截然不同的形成历史，然而现有的按形态、架构、学习算法、任务或领域组织的分类体系并不能直接回答能力究竟从何而来。我们将“能力形成来源”定义为对能力形成有实质性贡献的因素，它区别于组件或构建步骤。我们识别出七种非互斥的来源：记录经验(RE)、预测建模(PM)、评估交互(EI)、代理环境(SE)、机制基础(MG)、具身耦合(EC)以及进化驱动(ED)形成。借助带有理论饱和的重构式归纳法，我们将研究矩阵追溯至原始研究，对文献进行去重，制定编码规则，并开展了三轮最大差异抽样与负面案例抽样。所面临的挑战包括课程学习与自监督学习、主动推理、开放式…

    arXiv:2609.09627v1 Announce Type: new  Abstract: Capabilities relevant to Physical AI can arise from materially different formation histories, yet existing taxonomies organized by morphology, architecture, learning algorithm, task, or domain do not directly answer what gives rise to a capability. We define a capability-formation source as a factor materially contributing to capability formation, distinct from components or construction steps. We identify seven non-exclusive sources: Recorded-Experience (RE), Predictive-Modeling (PM), Evaluative-Interaction (EI), Surrogate-Environment (SE), Mechanism-Grounded (MG), Embodied-Coupling (EC), and Evolution-Driven (ED) Formation. Using reconstructive induction with theoretical saturation, we traced a research matrix to primary studies, deduplicated the literature, set coding rules, and conducted three rounds of maximum-difference and negative-case sampling. Challenges included curriculum and self-supervised learning, active inference, open-e
    
[^92]: 面向遥感影像开放世界目标检测的双曲几何方法

    Hyperbolic Geometry for Open-World Object Detection in Remote Sensing Imagery

    [https://arxiv.org/abs/2609.09626](https://arxiv.org/abs/2609.09626)

    本文提出基于双曲几何的遥感影像开放世界目标检测方法HyRS-OWOD，通过解耦目标性学习与双曲不确定性学习两步机制，提升了未知目标召回率和增量学习性能。

    

    开放世界目标检测（OWOD）扩展了闭集检测的任务范围，要求模型能够识别未知目标，并在标注可用后对其进行增量学习。在遥感影像中，目标类别往往存在潜在的层次关系，而现有方法普遍采用的欧几里得空间可能无法充分表达这种关系，从而限制了未知目标的召回率和增量学习性能。为解决这一问题，我们研究了双曲几何在遥感影像开放世界目标检测中的应用，并提出了HyRS-OWOD。为了提升未知目标的召回率，我们设计了一种两步式未知目标发现机制：首先是一个解耦目标性学习（DOL）模块，将前景感知与语义信息解耦，从背景区域中分离出前景候选区域；随后是一个双曲不确定性学习（HUL）组件，利用双曲嵌入的半径作为不确定性感知的度量……

    arXiv:2609.09626v1 Announce Type: cross  Abstract: Open-world object detection (OWOD) extends closed-set detection by requiring models to identify unknown objects and incrementally learn them once annotations become available. In remote sensing imagery, object categories often exhibit latent hierarchical relationships that may be inadequately represented in the Euclidean spaces commonly adopted by existing methods, limiting unknown-object recall and incremental-learning performance. To address this issue, we investigate hyperbolic geometry for OWOD in remote sensing imagery and propose HyRS-OWOD. To improve unknown object recall, we design a two-step unknown-object discovery mechanism: a Decoupled Objectness Learning (DOL) module that disentangles foreground perception from semantic information to separate foreground proposals from background regions, followed by a Hyperbolic Uncertainty Learning (HUL) component that leverages the radius of hyperbolic embeddings as an uncertainty-aware
    
[^93]: 从状态同步到认知自演化：认知数字孪生的运行架构

    From State Synchronization to Cognitive Self-Evolution: An Operational Architecture for Cognitive Digital Twins

    [https://arxiv.org/abs/2609.09625](https://arxiv.org/abs/2609.09625)

    本文提出了一种包含物理层、数字孪生层、认知层和任务层的四层认知数字孪生（CDT）架构，通过建立跨层的自演化闭环运行回路，解决了认知能力如何系统性集成到数字孪生架构中的问题。

    

    随着数字孪生（DT）系统从状态同步向面向任务和知识驱动的运行方式演进，认知数字孪生（CDT）作为一种将认知能力融入孪生运行的扩展形式应运而生。现有的CDT研究往往聚焦于特定的使能技术，例如学习模块、知识图谱和大语言模型，而对于如何将认知能力系统性地集成到数字孪生架构中，所提供的见解较为有限。为解决这一问题，本文提出了一种由物理层、数字孪生层、认知层和任务层组成的四层CDT架构。该架构建立了一个贯穿这四层的自演化闭环运行回路：物理状态被同步到数字表示中，认知通过知识、记忆和注意力机制构建面向特定任务的认知模型，并生成任务级决策。

    arXiv:2609.09625v1 Announce Type: new  Abstract: As Digital Twin (DT) systems evolve beyond state synchronization toward task-oriented and knowledge-driven operation, Cognitive Digital Twins (CDTs) have emerged as an extension that incorporates cognitive capabilities into twin operation. Existing CDT studies often focus on specific enabling techniques, such as learning modules, knowledge graphs, and large language models, while providing limited insight into how cognition can be systematically integrated into DT architectures. To address this issue, this paper proposes a four-layer CDT architecture consisting of the physical layer, digital-twin layer, cognitive layer, and task layer. The proposed architecture establishes a self-evolving closed operational loop spanning these four layers, in which physical states are synchronized into digital representations, cognition constructs task-specific cognitive models through knowledge, memory, and attention, and task-level decisions are genera
    
[^94]: RouteBridge：神经辐射场与3D高斯泼溅之间的可靠性路由双向蒸馏

    RouteBridge: Reliability-Routed Bidirectional Distillation Between Neural Radiance Fields and 3D Gaussian Splatting

    [https://arxiv.org/abs/2609.09606](https://arxiv.org/abs/2609.09606)

    RouteBridge提出了一种双向蒸馏框架，利用结合光度残差与几何证据的可靠性估计器为每条光线动态选择NeRF或3DGS作为教师（或弃权监督），避免了全局固定教师传播局部重建误差，在mip-NeRF 360和DTU基准上显著提升了两种表示的渲染质量。

    

    神经辐射场和3D高斯泼溅以互补的归纳偏置对场景进行编码，但现有的跨表示蒸馏通常将某一种表示固定为整个场景的教师。全局固定的教师可能会传播局部重建误差。我们提出了RouteBridge，这是一个为每条光线选择教学方向的双向框架。其可靠性估计器将光度残差与表示特定的几何证据相结合，从而将监督从NeRF路由到3DGS、从3DGS路由到NeRF，或选择弃权。一个与渲染器无关的接口可以在无需共享特征或点对应关系的情况下传输颜色、不透明度和归一化深度。在mip-NeRF 360数据集上，NeRF和3DGS导出模型分别达到28.56和28.77 dB。其中3DGS导出模型相比3DGS提升了1.56 dB，相比NeRF-GS提升了0.45 dB，同时将LPIPS降至0.207。在静态三视图DTU数据集上，RouteBridge获得了21.12 dB的成绩。

    arXiv:2609.09606v1 Announce Type: cross  Abstract: Neural radiance fields (NeRFs) and 3D Gaussian Splatting (3DGS) encode a scene with complementary inductive biases, but existing cross-representation distillation typically fixes one representation as teacher for the entire scene. A globally fixed teacher can propagate local reconstruction errors. We present RouteBridge, a bidirectional framework that selects the teaching direction for each ray. Its reliability estimator combines photometric residuals with representation-specific geometric evidence and routes supervision from NeRF to 3DGS, from 3DGS to NeRF, or abstains. A renderer-independent interface transfers color, opacity, and normalized depth without shared features or point correspondence. On mip-NeRF 360, the NeRF and 3DGS exports reach 28.56 and 28.77 dB, respectively. The 3DGS export improves over 3DGS by 1.56 dB and over NeRF-GS by 0.45 dB while reducing LPIPS to 0.207. On static three-view DTU, RouteBridge obtains 21.12 dB
    
[^95]: 不可验证的水印：欧盟《人工智能法案》实施后的AI文本水印

    Watermarks Without Verification: AI Text Watermarking After the EU AI Act

    [https://arxiv.org/abs/2609.09604](https://arxiv.org/abs/2609.09604)

    本文指出，在欧盟《人工智能法案》生效后，针对AI文本水印（如Claude和Gemini中部署的SynthID-Text）的用户质疑与厂商保证目前均无法验证，而这种不可验证性本身——而非水印技术——才是真正的核心治理挑战。

    

    2026年8月2日，欧盟《人工智能法案》第50条规定的义务正式生效，要求生成式AI提供商对其系统生成的内容进行标记，并确保这些内容能够被检测为AI生成。几天后，Anthropic披露，此后发布的每个Claude模型都会在所有生成文本中嵌入基于SynthID-Text的水印，默认开启且用户无法选择退出；谷歌自2024年起也已在Gemini中部署了SynthID-Text。用户提出异议，认为该水印会降低输出质量（尤其是代码），会秘密编码身份识别信息，并且（相互矛盾地）既容易被移除又无法逃避；厂商则回应保证质量不变、不含身份识别信息、且对轻度编辑具有鲁棒性。在本工作中，我们论证了这些用户异议与厂商保证目前均无法得到验证，而这种不可验证性——而非水印技术本身——才是实质性的治理问题所在。

    arXiv:2609.09604v1 Announce Type: cross  Abstract: On August 2, 2026, the obligations of Article 50 of the EU AI Act took effect, requiring generative AI providers to mark the content their systems produce and ensure it can be detected as AI-generated. Days later, Anthropic disclosed that every Claude model released after that date embeds a watermark based on SynthID-Text in all generated text, enabled by default with no user opt-out; Google has deployed SynthID-Text in Gemini since 2024. Users objected that the watermark degrades quality, particularly for code, that it secretly encodes identifying information, and, in mutual contradiction, that it is easily removable and inescapable; the vendor answered with assurances of unchanged quality, no identifying information, and robustness to light editing. In this work, we argue that neither the objections nor the assurances can currently be verified and that this unverifiability, rather than watermarking itself, is the substantive governan
    
[^96]: 面向抓举任务的紧凑型视触觉世界模型：预测、奖励对齐与力约束

    Compact Visuotactile World Models for Lifting: Prediction, Reward Alignment, and Force Constraints

    [https://arxiv.org/abs/2609.09597](https://arxiv.org/abs/2609.09597)

    该研究构建了一个仅65万参数的紧凑型视触觉世界模型，发现准确的触觉预测本身并不能保证力约束控制的提升，但模型辅助力反馈能将同分布任务成功率从73.3%提升至93.3%，而想象强化学习表现反而不如反应式隐式Q学习。

    

    准确的触觉预测未必能改善力约束控制。我们研究了一个包含652,157个参数的动作条件视触觉世界模型，并配合匹配的行为克隆、想象空间中的策略学习、独立的反应式隐式Q学习（IQL）以及模型辅助的力反馈进行对比。在固定协议下，我们在120个涵盖几何形状和物理参数变化的新建MuJoCo环境上执行34个策略，并在另外12个同分布（ID）环境上进行324条独立重放的动作分支测试。视触觉动力学将力作用效果的平均绝对误差（MAE）从持续性基线的0.413 N降低至0.338 N。模型辅助反馈将同分布力预算下的成功率从73.3%提升至93.3%，配对差异为+20.0 [+6.7, +33.4]个百分点（95%置信区间），且该差异主要发生在脚本化下降阶段；其汇总差异为+3.9 [-4.5, +11.7]个百分点。想象强化学习实现了11.9%的汇总联合成功率，而反应式IQL为25.0%。一项经验性触觉残差压力测……（摘要在此处截断）

    arXiv:2609.09597v2 Announce Type: cross  Abstract: Accurate tactile forecasts need not improve force-constrained control. We study a 652,157-parameter action-conditioned visuotactile world model with matched behavior cloning, policy learning in imagination, independent reactive implicit Q-learning, and model-assisted force feedback. A fixed protocol executes 34 policies on 120 fresh MuJoCo environments spanning geometry and physical-parameter shifts, plus 324 independently replayed action branches on 12 additional ID environments. Visuotactile dynamics reduce force action-effect MAE from 0.413 N for persistence to 0.338 N. Model-assisted feedback raises ID force-budgeted success from 73.3% to 93.3%, with paired difference +20.0 [+6.7,+33.4] percentage points (95% CI), with the difference occurring during scripted lowering. Its pooled difference is +3.9 [-4.5,+11.7] points. Imagined RL achieves 11.9% pooled joint success versus 25.0% for reactive IQL. An empirical tactile-residual stres
    
[^97]: 教师几何形状决定师生网络中的可学习性

    Teacher Geometry Shapes Learnability in Teacher-Student Networks

    [https://arxiv.org/abs/2609.09595](https://arxiv.org/abs/2609.09595)

    该论文发现教师网络的几何形状（参数分布）显著影响师生系统的可学习性，揭示了以往随机正态分布假设所掩盖的教师间巨大差异。

    

    师生系统是一种广泛用于研究学习的抽象框架，其中教师神经网络生成训练标签，使学生神经网络能够学习实现相同的功能。然而，以往研究常常假设教师网络的参数是随机生成且服从正态分布的，从而忽视了教师网络结构的重要性。这掩盖了不同教师网络在可学习性上的显著差异。我们将可学习性形式化为收敛到全局最小值的成功率，并将其表示为过参数化程度、学习算法、学生初始化分布和教师几何形状的函数。我们同时识别出一种使节点间差异性最大化的简单分布和一种使节点间差异性最小化的困难分布，并证明在广泛的设置范围和不同激活函数下，这两种分布会导致截然不同的成功率。为解释这一差距，我们研究了小型神经网络的损失景观...

    arXiv:2609.09595v1 Announce Type: cross  Abstract: Teacher-student systems, in which a teacher neural network generates training labels so that a student neural network can learn to implement the same function, are widely used as an abstract setting to study learning. However, the structure of the teachers is often overlooked by assuming randomly-generated, normally-distributed parameters. This hides substantial variation in how learnable different teachers are. We formalize learnability as the success rate of converging to the global minimum, as a function of overparameterization, learning algorithm, student initialization distribution, and teacher geometry. We both identify an easy distribution that maximizes node dissimilarity and a hard distribution that minimizes it, and show that these two distributions induce markedly different success rates across a large range of settings and for different activation functions. To explain the gap, we study the loss landscape of small neural ne
    
[^98]: 面向6G隐私保护具身智能的模态解耦联邦学习

    Modality-Decoupled Federated Learning for Privacy-Preserving Embodied Intelligence in 6G

    [https://arxiv.org/abs/2609.09591](https://arxiv.org/abs/2609.09591)

    本文提出FedMVLA，一种模态解耦的联邦学习框架，针对视觉、语言和动作通路在参数规模、隐私暴露和更新动态上的内在差异进行差异化处理，以解决6G网络中具身智能VLA模型分布式训练的隐私保护、通信效率和模型异构性难题。

    

    第六代（6G）无线网络有望为大规模具身智能提供关键基础设施，其中异构机器人通过低延迟连接、边缘智能和分布式感知进行协作。视觉-语言-动作（VLA）模型将视觉感知、语言理解和动作生成整合到统一的闭环策略中，为此提供了基础。然而，在分布式机器人智能体上训练和适配VLA模型带来了隐私保护、通信效率和模型异构性方面的挑战。现有的联邦学习（FL）方法忽视了视觉、语言和动作通路在参数规模、隐私暴露程度、更新动态以及对压缩或扰动的容忍度方面的内在差异。为解决这一问题，本文提出了FedMVLA，一种面向6G网络中隐私保护具身智能的模态解耦联邦学习框架。

    arXiv:2609.09591v1 Announce Type: cross  Abstract: Sixth-generation (6G) wireless networks are expected to provide a key infrastructure for large-scale embodied intelligence, where heterogeneous robots collaborate through low-latency connectivity, edge intelligence, and distributed sensing. Vision-language-action (VLA) models offer a foundation by integrating visual perception, language understanding, and action generation into a unified closed-loop policy. However, training and adapting VLA models to distributed robotic agents introduce challenges in privacy protection, communication efficiency, and model heterogeneity. Existing federated learning (FL) methods overlook the intrinsic differences among vision, language, and action pathways in parameter scale, privacy exposure, update dynamics, and tolerance to compression or perturbation. To address this issue, this article proposes FedMVLA, a modality-decoupled FL framework for privacy-preserving embodied intelligence in 6G networks. F
    
[^99]: 学习动力学的函数空间统计力学方法

    A Function-Space Approach to the Statistical Mechanics of Learning Dynamics

    [https://arxiv.org/abs/2609.09589](https://arxiv.org/abs/2609.09589)

    该论文提出直接在函数空间中对学习动力学进行统计力学描述，通过学习算子 \(M=JJ^\ast\) 与由态密度局部曲率定义的统计算子 \(B\) 推导出涨落势，从而解释了深度神经网络在高度非线性参数动力学下呈现规则宏观行为的原因。

    

    深度神经网络尽管在巨大的参数空间中呈现高度非线性的动力学，却展现出规则的宏观行为。我们直接在函数空间中建立了学习的统计力学描述，将参数配置视为微观实现，将函数及其动力学算子视为宏观变量。对于均方损失，精确的误差动力学由学习算子 \(M=JJ^\ast\) 决定。将条件随机动力学的动力学玻尔兹曼权重与参数空间的态密度（其局部曲率定义了统计算子 \(B\)）相结合，并对局部涨落进行积分，得到涨落势 \(\Phi_{\mathrm{fluc}}(M;B)=\frac{\sigma_\xi^2}{2}\log\det(M^{-1}+B)+\mathrm{const}\)。在固定谱条件下，当 \([M,B]=0\) 时该项处于旋转平稳状态，通过将 \(M\) 的大特征值与 \(B\) 的小特征值配对可使该项最小化，并产生局部恢复力。

    arXiv:2609.09589v1 Announce Type: new  Abstract: Deep neural networks exhibit regular macroscopic behavior despite highly nonlinear dynamics in vast parameter spaces. We develop a statistical-mechanical description of learning directly in function space, treating parameter configurations as microscopic realizations and functions with their dynamical operators as macroscopic variables. For mean-squared loss, the exact error dynamics are governed by the learning operator \(M=JJ^\ast\). Combining the dynamical Boltzmann weight of the conditional stochastic dynamics with the parameter-space density of states, whose local curvature defines a statistical operator \(B\), and integrating over local fluctuations yields   $$ \Phi_{\mathrm{fluc}}(M;B)=\frac{\sigma_\xi^2}{2}\log\det(M^{-1}+B)+\mathrm{const}. $$   At fixed spectrum, this term is rotationally stationary when \([M,B]=0\), is minimized by pairing large eigenvalues of \(M\) with small eigenvalues of \(B\), and generates a local restori
    
[^100]: CityPlanner：面向可执行城市规划的沙盒智能体

    CityPlanner: A Sandbox Agent for Executable Urban Planning

    [https://arxiv.org/abs/2609.09578](https://arxiv.org/abs/2609.09578)

    CityPlanner 提出了一个基于沙盒环境的可执行城市规划智能体框架，通过统一的文件化环境 UrbanSandbox 和将长轨迹分解为“初始构建”与“反馈改进”两个原子任务的强化学习方法，在真实世界基准上持续优于现有方法。

    

    城市规划是一个现实世界中的空间优化问题，需要在成本和服务质量等实际目标约束下，从庞大的候选空间中选择可行的行动方案。现有的优化方法和强化学习方法虽然对固定形式的问题有效，但通常依赖于任务特定的表示方式和约束处理机制。我们提出了 CityPlanner，一个面向可执行城市规划的沙盒智能体框架。CityPlanner 引入了 UrbanSandbox，这是一个统一的基于文件的环境，智能体可以在其中查看任务文件、生成规划方案、运行评估器，并根据可执行的反馈修订决策。为了让学习过程更加可行，我们进一步提出了原子任务强化学习方法，将漫长的沙盒轨迹分解为用于初始方案构建的 BuildPlan 和用于基于反馈进行优化的 ImprovePlan 两个子任务。在真实世界基准上的实验表明，CityPlanner 持续超越（摘要在此处截断）……

    arXiv:2609.09578v1 Announce Type: cross  Abstract: Urban planning is a real-world spatial optimization problem that requires selecting feasible actions from large candidate spaces under practical objectives such as cost and service quality. Existing optimization and reinforcement learning methods are effective for fixed formulations, but often depend on task-specific representations and constraint handling. We propose \emph{CityPlanner}, a sandbox-agent framework for executable urban planning. CityPlanner introduces \emph{UrbanSandbox}, a unified file-based environment where agents inspect task files, generate plans, run evaluators, and revise decisions based on executable feedback. To make learning tractable, we further propose atomic-task reinforcement learning, which decomposes long sandbox trajectories into \emph{BuildPlan} for initial construction and \emph{ImprovePlan} for feedback-based refinement. Experiments on a real-world benchmark show that CityPlanner consistently outperfo
    
[^101]: 基于深度学习的超声追踪中心肌应变漂移校正

    Myocardial Strain Drift Correction in Deep Learning Based Ultrasound Tracking

    [https://arxiv.org/abs/2609.09577](https://arxiv.org/abs/2609.09577)

    该论文提出一种结合持久记忆令牌与师生微调策略的深度学习框架，用于校正基于深度学习的超声心肌追踪中的应变漂移，强制实现生理一致的周期性运动并提高应变估计的准确性。

    

    超声心动图测得的心肌应变是评估心脏功能的关键生物标志物。近期的深度学习方法在心肌运动追踪方面表现出色，但往往缺乏生理约束，导致在整个心动周期内产生时间漂移。因此，被追踪的点在每个心动周期结束时可能无法回到其相对初始位置，从而产生不准确的应变估计，在某些情况下甚至出现发散。我们提出了一种深度学习框架，用于补偿心肌追踪过程中的漂移。我们通过引入能够在覆盖完整心动周期的滑动窗口之间共享信息的持久记忆令牌，扩展了最先进的超声心动图追踪方法TAS-Net。随后，在真实超声心动图数据上采用师生微调策略，在保持追踪精度的同时强制实现生理一致的周期性运动。实验表明，该方法降低了全局和局部应变漂移，并改善了……

    arXiv:2609.09577v1 Announce Type: cross  Abstract: Myocardial strain from echocardiography is a key biomarker for cardiac function. Recent deep learning methods show strong performance for myocardial motion tracking but often lack physiological constraints, leading to temporal drift across the cardiac cycle. Consequently, tracked points may not return to their relative initial positions at the end of each cardiac cycle, producing inaccurate strain estimates and even divergence in some cases. We propose a deep learning framework that compensates for drift during myocardial tracking. We extend a state-of-the-art echocardiographic tracking method (TAS-Net) with persistent memory tokens that share information across sliding windows over full cardiac cycles. A teacher-student fine-tuning strategy on real echocardiographic data then enforces physiologically consistent cyclic motion while preserving tracking accuracy. Experiments show reduced global and regional strain drift, improved agreeme
    
[^102]: 高维线性回归中基于合成数据的SGD学习

    Learning with Synthetic Data via SGD in High-Dimensional Linear Regression

    [https://arxiv.org/abs/2609.09572](https://arxiv.org/abs/2609.09572)

    本研究通过高维线性回归中的理论分析发现，混合使用合成数据会导致不可避免的强模型坍塌，而两阶段训练策略（仅在第一阶段使用合成数据）可以避免风险下限，证明模型坍塌并非不可避免。

    

    合成数据已成为突破有限人类生成数据限制、扩展模型训练规模的一种有前景的方法，但它也可能引发强烈的模型坍塌现象，即任何固定比例的合成数据都会阻止模型性能随数据规模扩大而提升，留下一个不可消失的额外风险下限。本文研究了在具有模型偏移的高维线性回归中，合成数据如何影响单遍SGD的泛化性能。我们为混合训练和两阶段训练建立了有限样本风险界，将标准的偏差和方差与源分布不匹配效应分离开来，具体包括混合训练下的波动和持续漂移，以及两阶段训练下的过滤初始化偏差。这些风险界揭示了鲜明的对比：混合训练会引发强模型坍塌，而两阶段训练通过仅在第一阶段使用合成数据避免了这一风险下限，这表明在简单的数据课程安排下模型坍塌并非不可避免。

    arXiv:2609.09572v1 Announce Type: new  Abstract: Synthetic data has become a promising way to scale model training beyond limited human-generated data but it may also induce strong model collapse (Dohmatob et al., 2024), where any fixed fraction of synthetic data prevents model performance from improving under data scaling, leaving a non-vanishing excess risk floor. In this paper, we study how synthetic data affects the generalization of one-pass SGD in high-dimensional linear regression with model shift. We establish finite-sample risk bounds for mixed and two-stage training, separating standard bias and variance from source-mismatch effects, namely fluctuation and persistent drift under mixing and filtered initialization bias under two-stage. These bounds reveal a sharp contrast: mixed training induces strong model collapse, while two-stage training avoids the floor by using synthetic data only in the first stage, showing that collapse is not inevitable under a simple data curriculum
    
[^103]: 基于结构签名的多智能体代理式图学习

    Multi-Agent Agentic Graph Learning via Structural Signatures

    [https://arxiv.org/abs/2609.09565](https://arxiv.org/abs/2609.09565)

    该论文提出基于结构签名的多智能体代理式图学习方法，让多个拥有各自独立记忆的智能体协作处理具有异构结构与语义模式的图，克服了现有方法共享推理策略以及对图结构文本化排序敏感的局限。

    

    代理式图学习近期在图推理任务上取得了令人瞩目的成果，其中由大语言模型（LLM）驱动的智能体按顺序对图进行采样作为证据，以支持其最终预测。现有方法要么采用单一智能体，要么编排多个基于角色的智能体在整个图上进行推理和学习，但两者本质上都依赖于跨不同图区域的共享推理策略，这对于具有异构结构和语义模式的图而言可能是次优的。受多智能体协作在复杂推理任务上进展的启发，一个自然的解决方案是让多个智能体拥有不同的记忆并进行协作；然而，将这种范式直接应用于图面临两个挑战。首先，现有的代理式图学习方法通常将图结构转化为自然语言描述供LLM智能体使用，使得推理过程对结构的排序敏感……（摘要在此处截断）

    arXiv:2609.09565v1 Announce Type: new  Abstract: Agentic graph learning (AGL) has recently achieved promising results on graph reasoning tasks, where an agent powered by a large language model (LLM) sequentially samples the graph as evidence to support its final prediction. Existing methods either employ a single agent or orchestrate multiple role-based agents to reason and learn over the entire graph, but both essentially rely on a shared reasoning policy across different graph regions, which can be suboptimal for graphs with heterogeneous structural and semantic patterns. Inspired by the progress of multi-agent collaboration on complex reasoning tasks, a natural remedy is to let multiple agents own different memory and collaborate; however, applying this paradigm to graphs directly faces two challenges. First, existing AGL methods typically verbalize graph structures into natural-language descriptions for LLM agents, making the reasoning process sensitive to the ordering of structura
    
[^104]: 软件工程中的氛围转变：评估AI主导的对话式编程在性能、认知与负责任采用方面的表现

    The Vibe Shift in Software Engineering: Evaluating AI-Led Conversational Programming for Performance, Cognition, and Responsible Adoption

    [https://arxiv.org/abs/2609.09560](https://arxiv.org/abs/2609.09560)

    研究发现氛围编程能显著提升开发效率（比传统编程快27%、比AI辅助编程快12%），但其可维护性下降等认知与质量代价要求开发者审慎、负责任地采用这一范式。

    

    本研究评估了“氛围编程”（Vibe Coding），这是一种新兴的AI主导的对话式编程范式，使开发者能够通过与大型语言模型的自然语言交互来生成软件。研究采用混合方法设计，将氛围编程与传统编程环境和AI辅助编程环境进行比较，评估了性能效率、认知影响和负责任采用三个方面。三十名参与者（包括专业开发者和高年级计算专业学生）在三种实验条件下完成了等效的编程任务。定量数据采用描述性统计和重复测量方差分析（ANOVA）进行分析，定性数据则通过主题分析进行检验。结果显示，氛围编程显著提高了开发效率，与传统编程相比任务完成时间减少了27%，与AI辅助编程相比减少了12%。然而，这些收益也伴随着较低的可维护性……（原文摘要在此处截断）

    arXiv:2609.09560v1 Announce Type: new  Abstract: This study evaluates Vibe Coding, an emerging AI-led conversational programming paradigm that enables developers to generate software through natural-language interaction with large language models. Using a mixed-methods design, the study assessed performance efficiency, cognitive implications, and responsible adoption in comparison with traditional and AI-assisted coding environments. Thirty participants, including professional developers and advanced computing students, completed equivalent programming tasks under three experimental conditions. Quantitative data were analyzed using descriptive statistics and repeated-measures ANOVA, while qualitative data were examined through thematic analysis. Results show that vibe coding significantly improved development efficiency, reducing task completion time by 27% compared with traditional coding and 12% compared with AI-assisted coding. However, these gains were accompanied by lower maintain
    
[^105]: 特征叠加中线性能及性的高概率保证

    High-probability guarantees for linear accessibility in feature superposition

    [https://arxiv.org/abs/2609.09556](https://arxiv.org/abs/2609.09556)

    该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。

    

    神经网络可以利用特征叠加来编码比维度数量更多的概念，但特征间的交叉干扰限制了同时激活特征的线性能及性。通过将线性能及性建模为一个压缩感知问题，我们在次高斯噪声下针对固定支撑集推导出高概率界，证明了充分维度以线性方式扩展（d=O_ε(k log m)），而非此前最坏情况下的二次方限制。随后，我们通过高斯尾近似在各系统参数下验证了这些界。这些结果量化了线性表示假设的几何约束，为评估稀疏自编码器、组合泛化和神经网络可解释性提供了一个框架。

    arXiv:2609.09556v1 Announce Type: cross  Abstract: Neural networks can leverage feature superposition to encode more concepts than dimensions, but cross-feature interference constrains the linear accessibility of simultaneously active features. By framing linear accessibility as a compressed sensing problem, we derive high-probability bounds for fixed supports under subgaussian noise, proving the sufficient dimension scales linearly ($d=O_{\varepsilon}(k \log m)$) rather than prior worst-case quadratic limits. We then validate these bounds across system parameters through Gaussian-tail approximations. These results quantify the geometric constraints of the linear representation hypothesis, providing a framework for evaluating sparse autoencoders, compositional generalization, and neural interpretability.
    
[^106]: 针对大语言模型的任意密码攻击无需微调

    Arbitrary Cipher Attacks Against Large Language Models Do Not Require Fine-Tuning

    [https://arxiv.org/abs/2609.09553](https://arxiv.org/abs/2609.09553)

    研究发现新一代前沿大语言模型无需微调，仅通过提示和上下文学习即可习得密码通信技能，且通过密码通信时模型安全对齐会被显著削弱甚至完全绕过。

    

    大语言模型的安全与防护研究主要致力于检测和防范越狱攻击，即允许对抗性用户从模型中诱发出不良或有害输出的对齐绕过手段。任意密码（或称隐蔽通信）攻击是此类越狱攻击中的一种，此前已在商业模型的微调API上得到演示。在这类攻击中，目标模型会在加密的有害问答语料库上进行训练，随后通过学习到的加密方案来响应有害请求。在本文中，我们证明了更新的前沿模型无需微调即可习得基于密码的通信技能。相反，它们可以通过提示学习这些技能，必要时也可以通过上下文学习来掌握。此外，当通过学习到的密码进行通信时，模型的对齐会被显著削弱甚至完全绕过。

    arXiv:2609.09553v1 Announce Type: cross  Abstract: Large language model safety and security research is preoccupied with, among other things, detecting and preventing jailbreak attacks: alignment bypasses that allow an adversarial user to elicit unwanted or harmful outputs from models. Arbitrary cipher, or covert communication, attacks are one such type of jailbreak and have previously been demonstrated against the fine-tuning APIs of commercial models. In these attacks, target models are trained on a corpus of encrypted harmful questions and responses and subsequently respond to harmful requests through the learned encryption scheme. In this paper, we show that newer frontier models do not require fine-tuning to acquire cipher-based communication skills. Instead, they can learn these skills through prompting and, when necessary, through in-context learning. Furthermore, model alignment is significantly weakened or entirely bypassed when communication occurs through the learned cipher.
    
[^107]: 一种估计机器学习模型样本量的统计方法

    A Statistical Approach to Estimating Sample Size of Machine Learning Models

    [https://arxiv.org/abs/2609.09547](https://arxiv.org/abs/2609.09547)

    本文提出了一种通过局部线性表示近似非线性机器学习模型、并在各局部区域评估统计功效来估计所需样本量的统计框架。

    

    机器学习（ML）预测模型的样本量确定具有挑战性，因为传统的功效分析通常需要预先指定预测变量与结果之间的关系以及效应结构。非线性机器学习模型学习复杂的预测面，无法进行直接的分析性功效计算。我们提出了一个框架，该框架用局部线性表示来近似非线性机器学习模型，并通过评估这些局部区域上的统计功效来估计样本量需求。

    arXiv:2609.09547v1 Announce Type: cross  Abstract: Sample size determination for machine learning (ML) prediction models is challenging because conventional power analysis typically requires the predictor-outcome relationship and effect structure to be specified a priori. Nonlinear ML models learn complex prediction surfaces that do not admit straightforward analytical power calculations. We propose a framework that approximates nonlinear ML models with localized linear representations and estimates sample size requirements by evaluating statistical power across these local regions.
    
[^108]: 基于因果元学习的6G非地面网络自适应分布式物理层认证与攻击检测

    Adaptive Distributed Physical-Layer Authentication and Attack Detection in 6G Non-Terrestrial Networks via Causal Meta-Learning

    [https://arxiv.org/abs/2609.09511](https://arxiv.org/abs/2609.09511)

    本文提出基于因果元学习的SAFA-MZ框架，通过融合多特征物理层指纹与结构因果建模，解决多普勒频移和信道快速变化带来的分布偏移问题，实现6G非地面网络中自适应的分布式物理层认证与攻击检测。

    

    非地面网络（NTN）中的物理层认证（PLA）面临严重的多普勒频移、长时延和快速信道变化的挑战，这些因素会导致分布偏移并降低传统学习方法的性能。现有的PLA方案通常依赖单一特征，或者对未见环境的泛化能力较差。本文提出了一种面向多区域网络安全自适应认证框架（SAFA-MZ），这是一个用于NTN中分布式物理层认证（DPLA）的因果元学习框架。首先，我们设计了一种多特征指纹，结合了空间、角度、合并器、子空间以及多普勒-时延特征。该指纹具有自适应性和分布式特性，因为它融合了来自多个空中节点的异构物理层特征和测量数据。其次，我们构建了一个结构因果模型（SCM），以捕捉设计选择、环境因素、提取特征和认证结果之间的关系。第三，我们开发了……

    arXiv:2609.09511v1 Announce Type: cross  Abstract: Physical-layer authentication (PLA) in non-terrestrial networks (NTNs) is challenged by severe Doppler shifts, long delays, and fast channel variations, which cause distribution shifts and degrade conventional learning methods. Existing PLA schemes often rely on single features or generalize poorly to unseen environments. This paper proposes a secure adaptive framework for authentication in multi-zone networks (SAFA-MZ), a causal meta-learning framework for distributed PLA (DPLA) in NTNs. First, we design a multi-feature fingerprint that combines spatial, angular, combiner, subspace, and Doppler-delay features. The fingerprint is adaptive and distributed, as it fuses heterogeneous physical-layer features and measurements from multiple aerial nodes. Second, we formulate a structural causal model (SCM) to capture the relations among design choices, environmental factors, extracted features, and authentication outcomes. Third, we develop 
    
[^109]: 从固定按键到可读模式：用于车辆智能体函数调用的小型语言模型

    From Fixed Keys to Readable Schemas: Small Language Models for Vehicle Agent Function Calls

    [https://arxiv.org/abs/2609.09476](https://arxiv.org/abs/2609.09476)

    该论文构建了一个基于Android Automotive、包含9,822个示例和79个车辆功能的车载函数调用基准，并在四个不同规模的小型语言模型上系统比较了功能令牌（紧凑推理但仅限已训练功能）与提示词模式（可泛化到新功能但推理开销更高）两种设计方案的优劣。

    

    车载助手必须在严格的内存和延迟约束下，将自然语言请求转换为准确的车辆功能调用，这使得小型语言模型（SLM）成为设备端部署的理想选择。对于此类模型，一个关键的设计选择是如何呈现可用的功能接口。目前有两种方法：为每个功能分配专用的功能令牌，或直接在提示词中提供功能模式。功能令牌（FT）能够实现紧凑的推理，但仅限于训练期间学习过的功能；而提示词模式（SIP）虽然可以泛化到未见过的功能，但代价是更长的提示词和更高的推理开销。我们引入了一个包含9,822个单轮示例的基准数据集，涵盖源自Android Automotive的79个车辆功能，其中包括保留功能和需要拒绝的请求。我们在从270M到1.7B参数的四个小型语言模型上，在匹配的微调条件下对两种方法进行了比较。

    arXiv:2609.09476v1 Announce Type: cross  Abstract: In-vehicle assistants must translate natural-language requests into accurate vehicle function calls under strict memory and latency constraints, making small language models (SLMs) attractive for on-device deployment. For such models, a key design choice is how the available function surface is presented. Two approaches are to represent each function with a dedicated Functional Token (FT) or provide function schemas directly in the prompt. FTs enable compact inference but are restricted to functions learned during training, whereas Schema-in-Prompt (SIP) can generalize to unseen functions at the cost of longer prompts and higher inference overhead. We introduce a benchmark of 9,822 single-turn examples spanning 79 vehicle functions derived from Android Automotive, including held-out functions and requests requiring refusal. We compare both approaches under matched fine-tuning across four SLMs from 270M to 1.7B parameters. On functions 
    
[^110]: 基于图强化学习的非地面网络分布式物理层认证与协同速率分割多址接入

    Distributed Physical Layer Authentication and Collaborative RSMA in Non-Terrestrial Networks via Graph Reinforcement Learning

    [https://arxiv.org/abs/2609.09475](https://arxiv.org/abs/2609.09475)

    该论文提出SAFA-MZ方案，通过将组级认证标签嵌入协同多层RSMA传输结构，并结合人工噪声与组差分隐私保护，利用图强化学习实现了非地面网络中分布式物理层认证与安全传输的联合设计，在抵御窃听者的同时最大化保密频谱效率。

    

    现有的非地面网络（NTN）物理层认证（PLA）方案通常依赖于单锚点验证，缺乏认证与传输的联合设计，并且忽略了窃听场景下的标签隐私泄露问题。本文考虑被动的、具有位置感知能力的静态窃听者，其无法获取合法的信道状态信息（CSI）。在该威胁模型下，我们提出了一种面向多区域NTN系统的安全自适应联邦认证方案（SAFA-MZ），在保证认证可靠性、功率限制和覆盖约束的同时最大化保密频谱效率（SSE）。其核心思想是将组级认证标签嵌入到协同的多层速率分割多址接入（RSMA）传输结构中。私有信号与公共信号进行联合波束赋形，利用人工噪声（AN）减少信息泄露，并通过组差分隐私（GDP）保护标签信息免受推理攻击……

    arXiv:2609.09475v1 Announce Type: cross  Abstract: Existing physical-layer authentication (PLA) schemes for non-terrestrial networks (NTNs) often rely on single-anchor verification, lack joint authentication-transmission design, and ignore tag privacy leakage under eavesdropping. In this paper, we consider passive, location-aware, static eavesdroppers without access to legitimate channel state information (CSI). Under this threat model, we propose secure adaptive federated authentication for multi-zone NTN systems (SAFA-MZ) that maximizes secrecy spectral efficiency (SSE) while ensuring authentication reliability, power limits, and coverage constraints. The main idea is to embed group-level authentication tags into a collaborative multi-layer rate-splitting multiple access (RSMA) transmission structure. Private and common signals are jointly beamformed, artificial noise (AN) is used to reduce information leakage, and group differential privacy (GDP) protects tag information against inf
    
[^111]: ContractEval：面向程序性指令一致性的查询条件化执行匹配

    ContractEval: Query-Conditioned Execution Matching for Procedural Instruction Conformance

    [https://arxiv.org/abs/2609.09458](https://arxiv.org/abs/2609.09458)

    ContractEval提出了一种将程序性指令表示为查询激活义务并与回复或轨迹证据进行匹配的诊断框架，能够将遗漏、错误分支、顺序错误、不变量违反等转化为可区分的一致性失败，从而识别出传统输出评估和轨迹评判方法所遗漏的LLM智能体结构性失败。

    

    随着大语言模型智能体从回答问题转向执行程序性任务，其失败可能是无根据的而非明显错误的：最终回复看起来可以接受，尽管系统跳过了使答案成立所必需的检查、分支、依赖或不变量。仅基于输出的评估只能看到答案，基于轨迹的评判只能看到活动过程，但两者都无法识别哪些义务在该查询下是处于激活状态的。我们提出了CONTRACTEVAL，一个使这些激活义务显式化的诊断框架。它将程序性指令表示为查询激活的义务，并将其与回复或轨迹证据进行匹配，从而将遗漏、错误分支、顺序错误、多余操作、不变量违反和输出契约违反转化为可区分的一致性失败类型。在一套经过审计的可控程序性契约测试集上，仅输出和轨迹感知的LLM评判器会漏掉许多注入的结构性失败；在黄金期望和观

    arXiv:2609.09458v1 Announce Type: new  Abstract: As LLM agents move from answering questions to carrying out procedures, failures can be unwarranted rather than visibly wrong: the final response looks acceptable even though the system skipped the check, branch, dependency, or invariant that made the answer justified. Output-only evaluation sees the answer, and trace-aware judging sees activity, but neither identifies which obligations were active for the query. We introduce CONTRACTEVAL, a diagnostic framework for making those active obligations explicit. It represents procedural instructions as query-active obligations and matches them against response or trace evidence, turning omissions, wrong branches, ordering errors, extra actions, invariant breaches, and output-contract violations into distinct conformance failures. On a controlled suite of audited procedural contracts, output-only and trace-aware LLM judges miss many injected structural failures; under gold expected and observe
    
[^112]: 智能体知道自己何时成功了吗？基于内部表示校准智能体置信度

    Do Agents Know When They Succeed? Calibrating Agent Confidence from Internal Representations

    [https://arxiv.org/abs/2609.09448](https://arxiv.org/abs/2609.09448)

    该论文提出潜在轨迹动力学（LTD）和动作表示探针（ARP）两种方法，利用模型内部表示来校准多轮智能体任务成功的置信度，在多个交互基准和模型上持续优于基于表面信号的传统方法。

    

    随着智能体系统在安全关键应用中的快速采用，衡量与智能体动作相关联的置信度变得至关重要。与传统机器学习系统相比，智能体工作流具有复杂的失败模式，涉及规划、工具调用和动态环境交互。在本文中，我们研究了模型的内部表示是否能在多轮智能体设置中为最终任务的成功与否提供更强的信号。我们提出了两种互补的方法：潜在轨迹动力学（LTD），用于总结交互轨迹中残差流表示的变化；以及动作表示探针（ARP），从动作决策时形成的表示来预测任务成功。在三个交互式基准（Bash、SQL、Python）和三个模型家族（Qwen14B、Qwen7B、DeepSeek6.7B）上，我们的方法始终优于基于表面生成和序列的方法。

    arXiv:2609.09448v1 Announce Type: new  Abstract: As agentic systems getting adopted rapidly in safety critical applications, it is vital to measure the confidence associated with the agentic actions. In comparison to the traditional machine learning systems, agentic workflows have complex failure modes with planning, tool invocation and dynamic environment interactions. In this paper, we investigate whether model's internal representations provide stronger signals of eventual task success in multi-turn agentic setups. We introduce two complementary methods: Latent Trajectory Dynamics (LTD), which summarizes changes in residual-stream representations across an an interaction trajectory, and the Action Representation Probe (ARP), which predicts success from representations formed at action decisions. Across three interactive benchmarks (Bash, SQL, Python) and three model families (Qwen14B, Qwen7B, DeepSeek6.7B), our methods consistently outperform surface level generation and sequence-ba
    
[^113]: 留一被试评估下高效的无泄漏神经架构搜索

    Efficient Leakage-Free Neural Architecture Search under Leave-One-Subject-Out Evaluation

    [https://arxiv.org/abs/2609.09433](https://arxiv.org/abs/2609.09433)

    提出一种无泄漏的分块式神经架构搜索方法，通过在留一被试评估中跨被试共享搜索结果，将BioVid热痛数据集上的平均准确率从82.79%提升至83.39%，同时参数量最多减少99.2%。

    

    留一被试（LOSO）评估用于估计基于被试分类的泛化性能，但它使得神经架构搜索（NAS）的计算成本极为高昂，因为完全嵌套的实现需要对N个被试分别进行独立的架构搜索，并且在假设训练成本近似线性的情况下，其复杂度达O(N^2)。我们提出了一种无泄漏的、基于分块的方法，可在不同被试之间共享NAS搜索运行结果。在BioVid热痛数据集上，我们的方法将平均准确率从82.79%提升至83.39%，同时将参数数量最多减少了99.2%。

    arXiv:2609.09433v1 Announce Type: cross  Abstract: Leave-One-Subject-Out (LOSO) evaluation estimates generalisation performance for subject-based classification but makes Neural Architecture Search (NAS) computationally expensive because a fully nested implementation requires N independent architecture searches and, assuming approximately linear training cost, scales as O(N^2). We propose a leakage-free, block-based approach that shares NAS runs across subjects. On the BioVid Heat Pain dataset, our approach increased the mean accuracy from 82.79% to 83.39% while reducing the number of parameters by up to 99.2%.
    
[^114]: SCCM：用于自动漂移检测与适应的流式巡航控制方法

    SCCM : Stream Cruise Control Method for Automated Drift Detection and Adaptation

    [https://arxiv.org/abs/2609.09432](https://arxiv.org/abs/2609.09432)

    本文提出了流式巡航控制方法（SCCM），一个面向在线回归的综合漂移检测与自适应框架，通过早期预更新漂移检测、漂移幅度量化、动态超参数调整和模型再校准，在数据分布演变时实现自动化的模型适应。

    

    现实世界的数据集通常表现出不断演变的分布，这被称为概念漂移。忽视漂移会降低预测性能，而依赖固定超参数则进一步限制了模型在不断变化条件下的适应性。自适应学习通过在线持续更新模型来应对这一挑战，使模型能够随着数据分布的演变进行增量调整并保持有效。本文提出了流式巡航控制方法（SCCM），这是一个面向在线回归的漂移检测与适应的综合框架。SCCM通过早期响应的预更新漂移检测、漂移幅度量化、基于KPI窗口的阈值设定以减轻局部误报、动态超参数调整以及模型重新校准，实现了自动化适应。SCCM还采用内存内设计以实现实时适应性，这与纯反应式方法不同——后者通常只有在性能下降后才激活适应机制。

    arXiv:2609.09432v1 Announce Type: cross  Abstract: Real-world datasets often exhibit evolving distributions, known as concept drift. Ignoring drift degrades predictive performance, while reliance on fixed hyperparameters further limits model adaptability under changing conditions. Adaptive learning addresses this challenge by continuously updating models online, allowing them to incrementally adjust and remain effective as data distributions evolve. This paper presents the Stream Cruise Control Method (SCCM), a comprehensive framework for drift detection and adaptation in online regression. SCCM enables automated adaptation through early-response, pre-update drift detection, drift magnitude quantification, KPI-window-based thresholding for local false-alarm mitigation, dynamic hyperparameter tuning, and model recalibration. SCCM also adopts an in-memory design for real-time adaptability, unlike purely reactive methods that typically activate adaptation only after performance degradatio
    
[^115]: XAI-Arena：大语言模型能否评估可解释人工智能（XAI）解释的质量？

    XAI-Arena: Can LLMs Assess the Quality of XAI Explanations?

    [https://arxiv.org/abs/2609.09428](https://arxiv.org/abs/2609.09428)

    本文提出XAI-Arena框架，利用大语言模型作为评委，对XAI解释质量进行可扩展、可重复、多维度且顾及利益相关者的自动化评估，解决了传统人工主观评估难以重复和扩展的问题。

    

    评估可解释人工智能（XAI）方法所产生解释的质量仍然具有挑战性，因为现有方法往往依赖主观的人类判断，这限制了研究的可重复性、可扩展性以及不同研究之间的可比性。我们研究了大型语言模型（LLM）能否作为一种可重复、可扩展的机制，对XAI解释的质量进行对比评估。我们提出了XAI-Arena，这是一个“LLM作为评委”（LLM-as-a-judge）框架，用于对XAI解释质量进行可扩展、可重复、多维度且顾及利益相关者的评估。XAI-Arena使我们能够沿着多个维度比较XAI解释，包括感知简易性、清晰度、任务适配性、信任校准、可操作性、透明度、忠实性以及整体可解释性。随后，我们在多种数据集、机器学习模型和利益相关者角色设定上对XAI解释方法进行了基准测试。人类验证结果显示出强烈的正相关关系。

    arXiv:2609.09428v1 Announce Type: new  Abstract: Evaluating the quality of explanations produced by explainable AI (XAI) methods remains challenging because existing approaches often rely on subjective human judgment, limiting reproducibility, scalability, and comparability between studies. We examine whether LLMs can serve as a reproducible and scalable mechanism to make comparative assessments of the quality of XAI explanations. We introduce XAI-Arena, an LLM-as-a-judge framework for scalable, reproducible, multidimensional, and stakeholder-sensitive evaluation of XAI explanation quality. XAI-Arena then allows us to compare XAI explanations along various dimensions, namely, perceived simplicity, clarity, task adequacy, trust calibration, actionability, transparency, faithfulness, and overall interpretability. We then benchmark XAI explanation methods across various datasets, machine learning models, and stakeholder personas. Human validation shows a strong positive association betwee
    
[^116]: Edu-QuRating：基于蒸馏成对判断的多维度教育数据筛选

    Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise Judgements

    [https://arxiv.org/abs/2609.09425](https://arxiv.org/abs/2609.09425)

    Edu-QuRating提出了一种多维度教育数据筛选流水线，通过定义教育专用评分标准、利用LLM评判器标注文档对并将成对偏好蒸馏为可复用的评分模型，从而突破传统单一标量式教育价值评估的局限，从准确性、吸引力、结构性和受众适用性等多个维度对文本进行精细化评分。

    

    教育数据过滤器已成为改进语言模型预训练的实用方法，但大多数过滤器将教育价值视为单一的标量属性。这对于某些应用来说可能过于宽泛，尤其是当数据集本身已经具有高密度的教育材料时。有用的学习材料需要准确、有吸引力、结构良好，并且适合目标受众和应用场景（例如面向学习者还是面向教师）。延续QuRating（Wettig等人，2024）的工作，我们提出了Edu-QuRating：一个用于多维度教育数据评分与筛选的流水线。Edu-QuRating定义了教育专用的评分标准（rubrics），使用LLM评判器对采样的文档对进行标注，并将这些成对偏好蒸馏为可复用的Edu-QuRater模型，这些模型可以根据一组教育标准对单个文本片段进行评分。在两个序列分类基础模型和六个教育标准的实验中，最好的Edu-QuRater能够恢复保留的（摘要在此处被截断）

    arXiv:2609.09425v1 Announce Type: new  Abstract: Educational data filters have become a practical way to improve language-model pre-training, but most filters treat educational value as a single scalar property. This may be too broad for some applications, especially if the data set already features a high density of educational material. Useful learning material needs to be accurate, engaging, well structured, and appropriate for the intended audience and application (e.g. learner- vs teacher-facing). Following QuRating (Wettig et al. 2024), we introduce Edu-QuRating: a pipeline for multi-dimensional educational data scoring and curation. Edu-QuRating defines education-specific rubrics, uses an LLM judge to label sampled document pairs and distills those pairwise preferences into reusable Edu-QuRaters, which can score individual text chunks on a set of educational criteria. Across two sequence-classification base models and six educational criteria, the best Edu-QuRater recovers held-
    
[^117]: Valerant：一种通过动作条件世界模型探索实现的自动可导航游戏地图生成器

    Valerant: An Automatic Navigable Game Map Generator via Action-Conditioned World Model Exploration

    [https://arxiv.org/abs/2609.09418](https://arxiv.org/abs/2609.09418)

    提出了Valerant，一种通过动作条件世界模型探索来自动生成持久可导航3D游戏地图的生成器，解决了游戏中虚拟世界必须由模型自身实例化这一独特挑战。

    

    世界动作模型将预测性世界建模与动作生成相结合，使预期的未来状态能够引导智能体的行为。尽管世界动作模型正在快速推动具身智能的发展，但通用型的对应方案在游戏领域仍基本处于未探索状态。现有的面向游戏的方法通常将动作条件世界模型与外部策略和奖励函数相结合来实现类似世界动作模型的决策能力，但它们主要在二维视觉观察空间中运行，且不会实例化持久的3D几何结构。将这一范式扩展到3D游戏带来了一个独特的挑战。在自动驾驶和机器人技术中，物理环境独立于模型而存在，提供了一个持久的3D世界，选定的动作可以在其中被执行。而游戏没有这样的外部基础，虚拟世界本身必须被实例化。大多数可玩的游戏需要一个持久且可导航的空间，而3D游戏还需要明确的……

    arXiv:2609.09418v1 Announce Type: new  Abstract: World Action Models (WAMs) couple predictive world modeling with action generation, allowing anticipated future states to guide agent behavior. Although WAMs are rapidly advancing embodied AI, general-purpose counterparts remain largely unexplored in games. Existing game-oriented approaches often combine action-conditioned world models with external policies and reward functions to realize WAM-like decision-making, yet they operate mainly in 2D visual observation space and do not instantiate persistent 3D geometry. Extending this paradigm to 3D games introduces a distinct challenge. In autonomous driving and robotics, the physical environment exists independently of the model, providing a persistent 3D world in which selected actions can be executed. Games have no such external substrate; the virtual world itself must be instantiated. Most playable games require a persistent and navigable space, while 3D games additionally require explic
    
[^118]: 面向决策的主动学习用于规模感知的关键材料回收

    Decision-Focused Active Learning for Scale-Aware Critical-Materials Recovery

    [https://arxiv.org/abs/2609.09413](https://arxiv.org/abs/2609.09413)

    本文提出面向决策的主动学习方法，将实验室选择性沉淀实验与工艺放大决策关联，在回收钕铁硼磁体时仅需16-24次实验即可达到最佳稀土富集度，远优于传统方法所需的48次。

    

    为回收工艺选择放大方案，需要将实验室结果与产品需求、工艺成本和规模效应联系起来。我们分析了太平洋西北国家实验室“关键元素回收与优化计算智能”（CICERO）工作流程中自主选择性沉淀的实验记录。主动学习利用先前的实验结果来选择后续实验。在一个基于拟合模型和回收钕铁硼（NdFeB）磁体记录的条件性回顾基准测试中，主动学习比非自适应的空间填充方法用更少的实验找到了最佳记录结果。富集度定义为所选稀土与铁的比值相对于进料中该比值的变化。自适应策略在16至24个孔（即单独实验）内即可达到记录的富集度最大值，而非自适应方法需要48个实验。我们的两阶段重构方法在16个孔时与两种自适应替代方案表现相当。对回收钐钴（SmCo）磁体的条件性分析显示……

    arXiv:2609.09413v1 Announce Type: new  Abstract: Choosing a recovery process for scale-up requires connecting laboratory results with product requirements, process costs, and scale effects. We analyze records from Pacific Northwest National Laboratory's Computer Intelligence for Critical Element Recovery and Optimization (CICERO) workflow for autonomous selective precipitation. Active learning uses prior results to choose experiments. In a conditional retrospective benchmark with fitted models and recycled neodymium-iron-boron (NdFeB) magnet records, active learning finds the best recorded result with fewer experiments than nonadaptive space filling. Enrichment is the selected rare-earth-to-iron ratio relative to that in the feed. Adaptive policies reach the recorded enrichment maximum by 16 to 24 wells (individual experiments), versus 48. Our two-stage reconstruction ties two adaptive alternatives at 16 wells. Conditional analyses of recycled samarium-cobalt (SmCo) magnets show a Roun
    
[^119]: 受两阶段MUSIC算法启发的可靠近场多用户定位

    Reliable Near-Field Multi-User Positioning Informed by Two-Stage MUSIC

    [https://arxiv.org/abs/2609.09409](https://arxiv.org/abs/2609.09409)

    本文提出了一种受两阶段MUSIC算法启发的端到端深度学习框架MUSIC-Net，通过在训练中嵌入两阶段MUSIC目标来分离视距信号子空间并识别替代距离，从而在混合视距/非视距多径场景下无需复杂的参数估计和路径关联即可直接、可靠地恢复多用户位置。

    

    近场定位是未来无线系统中实现高分辨率多用户定位的一项前景广阔的技术，但其性能常常因散射引起的相干传播而下降。现有的近场定位方法需要分别进行参数估计和路径/信源关联，存在计算开销大和误差累积的问题，且通常无法提供任何可靠性保证。本文提出了MUSIC-Net，这是一种在混合视距和非视距多径场景下受两阶段多重信号分类（MUSIC）算法启发的端到端近场定位深度学习框架，它将两阶段MUSIC的目标嵌入到训练过程中，以分离与视距相关的信号子空间并识别替代距离。所提出的框架无需复杂的非视距参数估计或路径/信源关联，即可直接恢复多用户位置。

    arXiv:2609.09409v1 Announce Type: cross  Abstract: Near-field localization is a promising technique for high-resolution multi-user positioning in future wireless systems, but its performance is often degraded by scattering-induced coherent propagation. Existing near-field localization methods, which require separate parameter estimation and path/source association, suffer from high computation overhead and accumulated errors, and usually do not provide any guarantee on reliability. In this paper, we propose \emph{MUSIC-Net}, an end-to-end near-field positioning deep learning (DL) framework informed by two-stage MUltiple SIgnal Classification (MUSIC) in mixed line-of-sight (LoS) and non-LoS (NLoS) multi-path scenarios, which embeds the two-stage MUSIC objects into training to isolate the LoS-related signal subspace and to identify a surrogate distance. The proposed framework directly recovers multi-user positions without the need for involved NLoS parameter estimation or path/source ass
    
[^120]: 智能体AI框架上多模态提示注入攻击的实验评估

    An Experimental Evaluation of Multimodal Prompt Injection Attacks on Agentic AI Frameworks

    [https://arxiv.org/abs/2609.09404](https://arxiv.org/abs/2609.09404)

    该论文提出可复现基准MMPIBench，系统评估了通过图像载体对智能体AI框架进行多模态提示注入攻击的效果，发现攻击指令在12.8%的运行中被感知尝试，但仅约1%真正完成攻击，规划步骤是阻止攻击的关键防线。

    

    智能体AI框架使语言模型能够进行规划、保留记忆，并调用可访问真实文件、邮件和服务的工具。大多数这类智能体还能读取图像，这为攻击者提供了一种绕过用户、直接将文本注入智能体上下文的途径。我们提出了MMPIBench，这是一个可复现的基准测试，用于测量注入后的后续影响。它通过六种视觉载体（OCR文本、叠加层、EXIF元数据、二维码、伪造界面及混合载体）传递一组固定的攻击，并记录每条注入指令在智能体中传播的距离——从感知到规划再到工具调用。在涵盖六个框架、五个基础模型、六种载体和四种攻击者目标的720次运行中，攻击在大约1%的运行中真正完成，但在12.8%的运行中被尝试执行，而这一差距几乎完全是在规划步骤被消除的，即模型读取到注入的指令后拒绝执行。该模型的影响远大于（注：原文摘要在此处被截断）

    arXiv:2609.09404v1 Announce Type: cross  Abstract: Agentic AI frameworks let a language model plan, keep memory, and call tools that reach real files, mail, and services. Most of these agents also read images, which gives an attacker a way to put text into the agent's context without going through the user. We present MMPIBench, a reproducible benchmark that measures what happens next. It delivers a fixed set of attacks through six visual carriers (OCR text, overlays, EXIF metadata, QR codes, fake interfaces, and hybrids) and records how far each injected instruction travels through the agent, from perception through planning to the tool call. Across 720 runs covering six frameworks, five foundation models, six carriers, and four attacker objectives, attacks complete in approximately 1% of runs but are attempted in 12.8%, and the gap is closed almost entirely at the planning step, where the model reads the injected instruction and declines to act on it. The model matters far more than 
    
[^121]: VANTAGE-Bench：评估视觉语言模型中的基础设施AI差距

    VANTAGE-Bench: Evaluating the Infrastructure AI Gap in Vision-Language Models

    [https://arxiv.org/abs/2609.09396](https://arxiv.org/abs/2609.09396)

    提出了VANTAGE-Bench基准，首次系统评估视觉语言模型在固定摄像头基础设施视频上的表现，揭示了当前模型存在的“基础设施AI差距”。

    

    随着视觉语言模型（VLM）向物理部署迈进，研究焦点仍集中在以主体为中心的消费类视频上评估面向行动的具身AI。这忽视了一类普遍存在的物理AI：基础设施AI，它依赖固定摄像头进行开环洞察，如安全监控和运营日志记录。我们推出了VANTAGE-Bench，一个衡量这种“基础设施AI差距”的基准。它涵盖三个运营领域（物流、交通和智能空间），在语义、空间、时间和时空能力上统一了图像和视频评估，并超越多选题形式，扩展到包括密集描述和时空定位在内的八种任务表述。它还增加了单目标跟踪的单次轨迹协议，并且据我们所知，这是首次在固定摄像头基础设施视频上进行此类评估，并与专业跟踪器进行对比评分。标注涵盖三个标注体系……

    arXiv:2609.09396v1 Announce Type: cross  Abstract: As Vision-Language Models (VLMs) advance toward physical deployment, the focus has remained on action-oriented Embodied AI evaluated on subject-centric consumer video. This overlooks a pervasive class of Physical AI: Infrastructure AI, which relies on fixed cameras for open-loop insights like safety monitoring and operational logging. We introduce VANTAGE-Bench, a benchmark measuring this "Infrastructure AI Gap." It spans three operational domains (Logistics, Transportation, and Smart Spaces), unifies image and video evaluation across semantic, spatial, temporal, and spatio-temporal capabilities, and moves beyond multiple-choice to eight task formulations including dense captioning and spatio-temporal grounding. It adds a single-pass trajectory protocol for Single Object Tracking and, to our knowledge, the first such evaluation on fixed-camera infrastructure video, scored against specialist trackers. Annotation spans three regimes over
    
[^122]: 菜单即执行先验：面向在线智能体的状态路径工具菜单

    The Menu Is an Execution Prior: State-Path Tool Menus for Online Agents

    [https://arxiv.org/abs/2609.09395](https://arxiv.org/abs/2609.09395)

    该论文提出将工具菜单视为执行先验，通过学习“状态路径”来构建工具菜单，克服了传统按请求相关性排序导致忽略或延迟关键输入生产工具的问题，帮助在线智能体更可靠地完成多步骤任务。

    

    语言模型通过工具进行行动，然而实际的智能体面对的是包含数千个接口的工具库。我们引入了“工具菜单”的概念，即在执行前展示给智能体的可用工具的简短有序子集，智能体只能调用菜单中的工具。多步骤任务需要最终动作，以及以可用顺序创建其输入的前置工具。当前的菜单构建器根据与请求的相关性对工具进行排序，这可能会突出最终动作，却忽略或延迟了那些不太明显的输入生产工具。我们引入了“状态路径”的概念，即从可观察的请求状态到期望结果的执行前路线，并提出状态路径工具菜单来学习它。我们的框架将菜单视为这些路线上的执行先验。其编码器表示哪些工具可以从当前状态运行、它们的输出如何满足后续输入，以及哪些顺序在训练路径中反复出现。检索器覆盖一个可执行的入口……（原文摘要在此处截断）

    arXiv:2609.09395v1 Announce Type: new  Abstract: Language models act through tools, yet practical agents face libraries containing thousands of interfaces. We introduce the tool menu as the short, ordered subset of available tools shown to an agent before execution. The agent can call only tools in this menu. Multi-step tasks require the final action and the prerequisite tools that create its inputs in a usable order. Current constructors rank tools by request relevance, which can surface the final action while omitting or delaying less obvious producers. We introduce the state path, a pre-execution route from the observable request state to the desired outcome, and propose State-Path Tool Menu to learn it. Our framework treats the menu as an execution prior over these routes. Its encoder represents which tools can run from the current state, how their outputs satisfy later inputs, and which orders recur in training paths. A retriever covers an executable entry, the missing-input produ
    
[^123]: 一种用于北极生态航行的自主地理空间人工智能（GeoAI）智能体

    An Autonomous GeoAI Agent for Arctic Eco-Navigation

    [https://arxiv.org/abs/2609.09374](https://arxiv.org/abs/2609.09374)

    该论文提出了一个人机协同的多智能体GeoAI系统，将运营、物理、生态和社区等多重准则统一整合到北极航线规划框架中，弥补了现有方法忽视生态与社区影响的不足。

    

    随着海冰状况的变化扩大了季节性通航范围，同时也带来了重大的运营、环境和社区风险，北极海事航行正变得越来越重要。北极航线规划本质上是一个多准则问题：提高船舶安全性或效率的航线可能会增加对海冰、敏感生态系统或附近社区的暴露风险。现有的路径规划方法主要关注航行时间、燃料消耗和航行风险，往往忽视了生态和社区层面的影响。我们提出了一个用于北极生态航行的人机协同多智能体GeoAI系统，该系统在统一的路由规划框架内整合了运营、物理、生态和社区相关准则。多个专业化智能体协同完成地理空间数据的获取与准备、多目标航线生成以及基于天际线的决策支持。生态准则明确考虑了暴露风险……

    arXiv:2609.09374v1 Announce Type: new  Abstract: Arctic maritime navigation is becoming increasingly important as changing sea-ice conditions expand seasonal accessibility while simultaneously introducing substantial operational, environmental, and community risks. Arctic route planning is inherently a multi-criteria problem: routes that improve vessel safety or efficiency may increase exposure to sea ice, sensitive ecosystems, or nearby communities. Existing routing methods prioritize travel time, fuel use, and navigational risk, often overlooking ecological and community impacts. We introduce a human-in-the-loop, multi-agent GeoAI system for Arctic eco-navigation that integrates operational, physical, ecological, and community-related criteria within a unified routing framework. Multiple specialized agents coordinate geospatial data acquisition and preparation, multi-objective route generation, and skyline-based decision support. The ecological criteria explicitly account for exposur
    
[^124]: 印度母婴护理中可审计的紧急分诊系统

    Auditable Emergency Triage for Maternal and Newborn Care in India

    [https://arxiv.org/abs/2609.09356](https://arxiv.org/abs/2609.09356)

    该论文将LLM紧急分诊系统分解为症状提取与紧急性判断两个可审计的步骤，并引入结构化决策树，解决了大规模母婴护理分诊系统中不透明、难以调试和迭代成本高的问题。

    

    在Noora Health，我们的护士每月在基于WhatsApp的服务上回答超过5万个医疗咨询，该服务为照护者提供按需支持。他们最紧迫的任务是紧急分诊：判断哪些咨询需要立即进行面对面处理。为支持这项工作，我们构建了一个使用大语言模型（LLM）的系统，对消息是否属于紧急情况进行分类，并提供可解释性的理由。但该系统是不透明的：分析错误意味着需要逐条阅读每条消息的推理链，这在我们服务的规模下是不可行的。提示词的任何修改都需要重新运行完整的评估以防止性能退化，这既成本高昂又在运营上极具挑战性。临床医生遵循一棵决策树来做出分诊判断，但这棵决策树从未被记录下来或传递给模型，模型仅依赖于一份平铺的危险体征列表。为解决这些问题，我们将分诊分解为两个步骤：由LLM提取规范化的症状和患者情况，再基于此进行紧急性判断，从而使整个系统变得可审计、可维护，并支持安全的迭代改进。

    arXiv:2609.09356v1 Announce Type: new  Abstract: At Noora Health, our nurses answer more than 50,000 medical queries per month on our WhatsApp-based service that provides caregivers with on-demand support. Their most time-critical task is emergency triage: deciding which queries need immediate in-person attention. To support them, we built a system that uses a large language model (LLM) to classify whether a message is an emergency and provide a rationale for interpretability. But the system was opaque: analyzing mistakes meant reading reasoning chains for each message, which is infeasible at our scale. Prompt changes meant re-running a full evaluation to prevent regressions, which was both costly and operationally challenging. Clinicians follow a decision tree to make this call, but it was never documented or passed to the model, which relied on a flat list of danger signs. To address these issues, we decomposed triage into two steps: an LLM extracts canonical symptoms and patient con
    
[^125]: 跨连续体的智能自适应计算：大语言模型在物联网-边缘-云资源管理中的应用

    Smart Adaptive Computing Across the Continuum: LLMs in IoT-Edge-Cloud Resource Management

    [https://arxiv.org/abs/2609.09348](https://arxiv.org/abs/2609.09348)

    该论文在现有DRL连续体编排系统分类法基础上扩展了“AI增强范式”和“反馈通道”两个新维度，并通过分析六种近期系统架构发现了共同差距：现有方案均未在云连续体环境中将完整的LLM编排与完整的代理层反馈相结合。

    

    管理物联网、边缘和云各层的资源，需要在几乎不固定的约束条件下做出持续的、情境感知的决策。深度强化学习（DRL）能很好地处理这类问题，而大语言模型（LLM）正越来越多地被用于增强DRL流程，但两者之间的架构关系却很少被明确阐明。我们在Wang等人提出的采用DRL技术的连续体编排系统分类法的基础上，扩展了两个新的维度。AI增强范式衡量LLM如何被利用，而反馈通道则捕捉执行反馈是否以及通过哪条系统路径返回到LLM，从而在LLM编排层闭合MAPE控制循环。我们将该分类法应用于六种近期的系统架构，并发现了一个共同的差距：在云连续体环境中，没有任何一种架构将完整的LLM编排与完整的代理层反馈相结合。我们关联了……（摘要原文在此处截断）

    arXiv:2609.09348v1 Announce Type: cross  Abstract: Managing resources across IoT, edge, and cloud layers calls for continuous, context-aware decisions under constraints that rarely stay fixed. Deep reinforcement learning (DRL) handles this class of problems well, and large language models (LLMs) are increasingly used to augment DRL pipelines, yet the architectural relationship between the two is seldom made explicit. We build on Wang et al.'s taxonomy of Continuum Orchestration Systems employing DRL techniques and extend it with two further dimensions. The AI Augmentation Paradigm measures how LLMs are exploited, while the Feedback channel captures whether and through which system path the execution feedback returns to the LLM in order to close the MAPE control loop at the LLM Orchestration layer. We apply this taxonomy to six recent system architectures and find a common gap, as none combines full LLM orchestration with full agent-layer feedback in a Cloud Continuum setting. We relate
    
[^126]: 通过预测重传来改进5G AI-RAN的MCS选择

    Improving 5G AI-RAN MCS Selection by Predicting Retransmissions

    [https://arxiv.org/abs/2609.09324](https://arxiv.org/abs/2609.09324)

    本文提出了NOSTRAdAMUS预测性链路自适应框架，通过从HARQ历史预测下一帧是否发生重传来修正5G AI-RAN系统中的MCS选择，在不替换现有算法的前提下为其增添前瞻能力，从而改善频谱效率与性能的权衡。

    

    5G NR中的链路自适应（LA）本质上是反应式的，它依赖于信道测量和HARQ反馈，而当信道快速变化时，这些数据可能很快过时。这些数据还包含噪声，难以准确跟踪，并且需要输入到具有难以排查的反馈回路效应的实时控制器中。这就解释了为什么大多数实际部署选择简单但稳健的算法，接受滞后可能使调度器以过于激进或不必要保守的速率运行，从而用频谱效率换取可预测的性能。在本文中，我们通过NOSTRAdAMUS改进了这一现状，这是一个预测性链路自适应框架，它在不需要替换或重新设计现有算法的情况下为其增添前瞻能力。NOSTRAdAMUS根据最近的HARQ历史预测下一个无线帧中是否会发生重传，并据此对所选择的调制编码方案（MCS）应用修正……

    arXiv:2609.09324v1 Announce Type: cross  Abstract: Link Adaptation (LA) in 5G NR is inherently reactive, relying on channel measurements and HARQ feedback that may become quickly obsolete when the channel changes quickly. This data is also noisy, making it hard to track accurately, and has to be fed to real-time controllers with feedback-loop effects which are hard to troubleshoot. This explains why most practical deployments select simple but robust algorithms, which accept that the lag can leave the scheduler operating at overly aggressive or unnecessarily conservative rates, trading spectrum efficiency for predictable performance. In this paper, we improve on this status-quo with NOSTRAdAMUS, a predictive LA framework which adds foresight to existing algorithms without replacing or redesigning them. NOSTRAdAMUS predicts whether a retransmission will occur in the next radio frame from recent HARQ history, and applies corrections to the Modulation and Coding Scheme (MCS) selected by t
    
[^127]: 梯度之地：论跨多个维度分化的现象体验

    Gradland: On Phenomenal Experience, Differentiated Across Many Dimensions

    [https://arxiv.org/abs/2609.09306](https://arxiv.org/abs/2609.09306)

    论文提出物理相互作用的一阶结构（梯度或雅可比矩阵）能够刻画现象体验的结构，并通过引入基于基尔霍夫复杂度的“有效秩”与“内聚性”两种度量，在理想化神经网络世界中解释了体验时长、清晰与模糊之分、质感、婴儿的混沌感知、想法的清晰度以及学习感受等多种现象。

    

    本文研究这样一个假设：物理相互作用的一阶结构，即梯度或雅可比矩阵，刻画了现象体验的结构。该研究在一个由神经网络居住的理想化世界——“梯度之地”中进行，在那里物理学规律是已知的，函数（大部分）是可微的。本文基于基尔霍夫复杂度，引入了两种雅可比矩阵结构的度量方法：有效秩和内聚性。将这两种度量应用于一系列具体实例后表明，该假设能够解释：（1）体验的持续时间，即体验可以延续数百毫秒；（2）被清晰体验与被模糊体验的内容之间的差异；（3）质感的体验；（4）新生儿可能经历的“纷乱嘈杂的混沌”体验；（5）头脑中清晰把握的想法与混乱想法之间的差异；（6）学习的感受；最后（7）本文解释了乐趣（原文在此处截断）。

    arXiv:2609.09306v1 Announce Type: new  Abstract: This paper investigates the hypothesis that the first-order structure of physical interactions, i.e. gradients or Jacobians, characterizes the structure of phenomenal experience. It does so in an idealized world inhabited by neural networks, Gradland, where the physics are known and the functions are (mostly) differentiable. The paper introduces two measures of Jacobian structure: effective rank and cohesion, based on Kirchhoff complexity. Applying the measures to a series of worked examples shows the hypothesis accounts for: (1) the duration of experience, that it can prolong over hundreds of milliseconds; (2) the difference between what is experienced vividly and obscurely; (3) the experience of texture; (4) the blooming buzzing confusion presumably experienced by newborns; (5) the difference between ideas that are held distinctly in mind and ideas that are confused; (6) what learning is like; and finally (7) the paper explains the fun
    
[^128]: 基于迭代重加权最小二乘的固定费用网络流支撑发现

    Support Discovery With Iteratively Reweighted Least Squares for Fixed-Charge Network Flow

    [https://arxiv.org/abs/2609.09295](https://arxiv.org/abs/2609.09295)

    提出一种基于迭代重加权最小二乘框架的可扩展连续优化算法，通过光滑非凸Lasry-Lions代理函数与具有加权图拉普拉斯结构的高效牛顿求解器，解决大规模固定费用网络流问题的计算难题。

    

    固定费用网络流问题（FCNFP）将连续的流量分配与离散的弧激活决策耦合在一起，使其成为各种网络设计和资源分配问题的经典但计算上具有挑战性的模型。精确的混合整数线性规划表述能够忠实地刻画固定费用结构，但在大型网络上往往难以求解。我们提出了一种基于迭代重加权最小二乘（IRLS）框架的可扩展连续优化算法，用于求解大规模单品种固定费用网络流问题。该方法用光滑的非凸Lasry--Lions代理函数替代不连续的固定费用与线性弧成本目标，并求解一系列加权二次流子问题。每个子问题通过热启动的对偶半光滑牛顿法求解，其牛顿系统具有加权图拉普拉斯结构，从而能够使用现代拉普拉斯求解器。为进一步改进……（摘要被截断）

    arXiv:2609.09295v1 Announce Type: cross  Abstract: The fixed-charge network flow problem (FCNFP) couples continuous flow allocation with discrete arc-activation decisions, making it a canonical but computationally challenging model for a variety of network design and resource allocation problems. Exact mixed-integer linear programming formulations capture the fixed-charge structure faithfully, but often become difficult to solve on large networks. We propose a scalable continuous-optimization algorithm for large-scale single-commodity FCNFP based on an iteratively reweighted least-squares (IRLS) framework. The method replaces the discontinuous fixed-charge and linear arc cost objective with a smooth nonconvex Lasry--Lions surrogate and solves a sequence of weighted quadratic flow subproblems. Each subproblem is solved by a warm-started dual semismooth Newton method whose Newton systems have weighted graph-Laplacian structure, enabling the use of modern Laplacian solvers. To further imp
    
[^129]: 声音还是刻板印象？解耦语音到语音模型中基于声学与内容的性别信息

    Voice or Stereotype? Disentangling Acoustic and Content-Based Gender in Speech-to-Speech Models

    [https://arxiv.org/abs/2609.09263](https://arxiv.org/abs/2609.09263)

    该论文设计了一个将说话者声音性别与文本内容刻板印象相解耦的受控实验，用以检测语音到语音模型在声音渲染和性别归因时是依赖声学性别还是内容刻板印象。

    

    语音到语音（S2S）模型如今已应用于配音、翻译和语音智能体中。与文本模型不同，它们能听到说话者的声音，而声音承载着说话者的性别。一个忠实的系统应该按照说话者听起来像谁来对待他们，而不是按照通常说这类话的人是谁（刻板印象）来对待他们。测试这一点比看起来更难，因为大多数S2S模型使用单一固定的输出语音进行回答，该语音是硬编码的，不会偏向刻板印象，即使模型存在偏见，检查输出语音也会显示“干净”。因此我们提出两个问题：当模型复述输入内容时，词语中的刻板印象是否会改变输出语音被感知到的性别（声音渲染）？当模型陈述说话者性别时，它是遵循声音还是内容（性别归因）？我们通过一个受控实验回答这两个问题，该实验将男声和女声与男性化、中性化和女性化刻板印象的文本段落交叉组合，在五个开源和闭源（模型上进行评估）……

    arXiv:2609.09263v1 Announce Type: cross  Abstract: Speech-to-speech (S2S) models now run inside dubbing, translation, and voice agents. Unlike text models, they hear the speaker's voice, which carries the speaker's gender. A faithful system should treat a speaker as who they sound like, not as whoever usually says what they said. Testing this is harder than it looks, since most S2S models answer in a single, fixed output voice, hard-coded so it cannot drift toward a stereotype. Checking the output voice comes back clean even when the model is biased. We therefore ask two questions. When a model re-speaks the input, does the stereotype in the words shift the perceived gender of the output voice (voice rendering)? And when the model states the speaker's gender, does it follow the voice or the content (gender attribution)? We answer both with one controlled experiment crossing male and female voices with masculine-, neutral-, and feminine-stereotyped passages, on five open- and closed-sou
    
[^130]: DiffLUT-Net：具有可学习连接的FPGA查找表网络的可微训练

    DiffLUT-Net: Differentiable Training of FPGA LUT Networks with Learnable Connectivity

    [https://arxiv.org/abs/2609.09254](https://arxiv.org/abs/2609.09254)

    DiffLUT-Net提出了一种可从零训练的FPGA原生LUT网络，通过可微松弛方法联合学习LUT真值表与输入连接关系，训练后直接导出为可综合Verilog代码，实现了紧凑高效的FPGA原生神经网络推理。

    

    现场可编程门阵列（FPGA）能够实现高效的神经网络推理，但大多数部署流程要么是加速乘加运算，要么是将预训练的量化模型转换为查找表（LUT）。我们提出了DiffLUT-Net，这是一种由六输入查找表连接而成的FPGA原生网络，可从零开始训练。我们利用可微LUT函数松弛和硬件源选择技术，联合学习LUT的64个真值表项以及其六个输入端口各自的信号源。训练完成后，真值表和连接被离散化，未使用的逻辑可以被剪枝，网络可直接导出为可综合的Verilog代码。在五个基准测试中，DiffLUT-Net实现了优异的精度-资源权衡。这些结果证明了联合学习LUT函数与稀疏连接对于紧凑的FPGA原生推理的有效性。代码可在 https://github.com/TUDa-HWAI/DiffLUT-Ne 获取。

    arXiv:2609.09254v1 Announce Type: cross  Abstract: Field-programmable gate arrays (FPGAs) enable efficient neural-network inference, but most deployment flows either accelerate multiply-accumulate operations or convert pretrained quantized models into lookup tables (LUTs). We present DiffLUT-Net, an FPGA-native network connected by six-input LUTs that are trained from scratch. We jointly learn the 64 truth-table entries of a LUT and the source to each of its six input ports using a differentiable LUT function relaxation and hardware source selection. After training, the truth tables and connections are discretized, unused logic can be pruned, and the network is exported directly as synthesizable Verilog. Across five benchmarks, DiffLUT-Net achieves favorable accuracy-resource trade-offs. These results demonstrate the effectiveness of jointly learning LUT functions and sparse connectivity for compact FPGA-native inference. The code is available at https://github.com/TUDa-HWAI/DiffLUT-Ne
    
[^131]: 没有免费的检查器：机器人策略验证器综述

    No Free Checker: A Survey of Verifiers for Robot Policies

    [https://arxiv.org/abs/2609.09250](https://arxiv.org/abs/2609.09250)

    本综述调研了约150种机器人策略验证器，提出以“可用性”和“可信度”两个维度进行对比分析，将验证器按判定来源分为人类、规则/形式化、学习/预训练和模型内在四大类，并揭示了可用性提升时可信度下降的固有权衡。

    

    机器人策略验证器读取候选行为并返回其表现好坏的评分，既用于评估视觉-语言-动作（VLA）策略，也用于训练这些策略。验证器的形式多种多样，从成功检测器和奖励模型，到运行时监视器、安全过滤器和时序逻辑规范。我们调研了约150种验证器，并从两个属性维度对它们进行比较：可用性，即一次判定需要多少成本、判定在执行轨迹中多早出现、以及能够多频繁地请求判定——当判定更便宜、更早、更密集时，可用性越高；可信度，即高分在多大程度上能说明任务完成情况——当判定容易被钻空子或偏向自利时，可信度越低。我们根据判定提供者的不同将验证器分为四类：人类验证器、基于规则和形式化的验证器、学习型和预训练型验证器，以及模型内在验证器。在这四个类别中，我们发现随着可用性的提升，可信度会随之下降。

    arXiv:2609.09250v1 Announce Type: cross  Abstract: A verifier for robot policies reads a candidate behavior and returns a score for how well it did, used both to evaluate vision-language-action policies and to train them. Verifiers range from success detectors and reward models to runtime monitors, safety filters, and temporal-logic specifications. We survey roughly 150 verifiers and compare them along two properties. Availability is how much a verdict costs, how early in a rollout the verdict arrives, and how often a verdict can be asked for. Availability rises as verdicts get cheaper, earlier, and denser. Credibility is how much a high score tells us about the task. Credibility falls as the judgment becomes gameable and self-serving. We group the verifiers by who supplies the judgment: human verifiers, rule-based and formal verifiers, learned and pretrained verifiers, and model-intrinsic verifiers. Across the four families, we find that credibility falls as availability rises. Regard
    
[^132]: 固定 Rollout 次数的 pass@k 评估能识别什么

    What Fixed-Rollout pass@k Evaluations Can Identify

    [https://arxiv.org/abs/2609.09245](https://arxiv.org/abs/2609.09245)

    该论文证明固定 n 次 rollout 的成功计数只能识别任务成功率分布的前 n 个矩，因此 pass@k 仅在 k ≤ n 时可识别，任何超出采样预算的外推在原理上都无法确定。

    

    重复采样评估越来越多地将 pass@k 外推到远超每个问题实际采集样本数 n 的范围。我们证明，在合并/随机任务的条件二项模型中，固定 n 次的成功计数仅能识别潜在单任务成功率分布的 n 个自由矩。因此，当 k ≤ n 时，直接 pass@k 是可识别的；但当 k > n 时，一般的外推 pass@k、尾部指数和尾部常数均不可识别，即使在同一 rollout 预算下拥有任意多个可交换任务也是如此。这一结论比“常用估计量在超过 n 时无定义”的观察更强：它刻画了固定深度计数律实验所缺失的信息。我们给出了保持计数律但外推结果互不相容的精确构造，阐述了唯一可延拓的例外情形，并通过 Hausdorff 主表示计算了尖锐的总体可识别区间。在公开的每个问题 10,000 次 rollout 的数据上……（摘要原文在此处截断）

    arXiv:2609.09245v1 Announce Type: new  Abstract: Repeated-sampling evaluations increasingly extrapolate pass@k far beyond the number n of samples collected per problem. We show that, in the pooled/random-task conditional-Binomial model, fixed-n success counts identify only the n free moments of the latent per-task success distribution. Consequently, direct pass@k is identified for k <= n, but generic extrapolated pass@k, tail exponents, and tail constants are not identified for k > n, even with arbitrarily many exchangeable tasks at the same rollout budget. This is stronger than the observation that the usual estimator is undefined beyond n: it characterizes the information missing from the fixed-depth count-law experiment. We give exact count-law-preserving constructions with incompatible extrapolations, state the exceptional unique-extension case, and compute sharp population identified intervals through Hausdorff principal representations. On the public 10,000-rollout-per-problem re
    
[^133]: 临界初始化使宽标量输入网络中的高阶输入导数失稳

    Critical initialization destabilizes higher input derivatives in wide scalar-input networks

    [https://arxiv.org/abs/2609.09244](https://arxiv.org/abs/2609.09244)

    该论文揭示了临界初始化虽能使宽网络一阶输入扰动的方差与深度无关，但会使高阶输入导数方差随深度线性增长而失稳，同时证明了分支尺度为 L^{-1/2} 的残差网络可使任意固定有限阶导数的方差一致有界。

    

    混沌边缘条件能够在宽随机初始化网络中保持一阶输入扰动的稳定，但物理信息损失、分数匹配以及导数正则化都依赖于更高阶的输入导数。对于光滑的标量输入全连接网络，我们利用在每个固定深度下于无限宽度极限中成立的有限导数喷射的联合高斯性，推导出直至三阶的平均场递推关系，这些递推在方差不动点处是精确的，且有限深度修正呈几何级数衰减。在临界点上，一阶导数方差与深度无关，而只要激活函数具有非零曲率，二阶导数方差就会随深度线性增长。由此得到的三阶系统在平均场敏感度上封闭。对于分支尺度为 L^{-1/2} 的残差网络，我们证明了在明确的正则性假设下，每个固定的有限导数阶数都具有一致有界的方差。模拟……（原文此处截断）

    arXiv:2609.09244v1 Announce Type: new  Abstract: The edge-of-chaos condition preserves first-order input perturbations in wide randomly initialized networks, but physics-informed losses, score matching and derivative regularization depend on higher input derivatives. For smooth scalar-input fully connected networks, using a joint Gaussianity of the finite derivative jet that holds in the infinite-width limit at each fixed depth, we derive mean-field recursions through third order that are exact at the variance fixed point, with finite-depth corrections that decay geometrically. At criticality, the first-derivative variance is depth-invariant, whereas the second-derivative variance grows linearly whenever the activation has nonzero curvature. The resulting third-order system closes on mean-field susceptibilities. For residual networks with branch scale L^{-1/2}, we prove that every fixed finite derivative order has uniformly bounded variance under explicit regularity assumptions. Simula
    
[^134]: 我们该信任RAG吗？测量检索增强生成在文档投毒下的鲁棒性

    In RAG We Trust? Measuring Robustness of Retrieval-Augmented Generation Under Document Poisoning

    [https://arxiv.org/abs/2609.09243](https://arxiv.org/abs/2609.09243)

    本研究通过对Llama 3.1 8B进行588次因子实验，首次系统量化了检索增强生成（RAG）在文档投毒攻击下的脆弱性——当全部三篇检索段落被篡改时准确率从77.9%骤降至43.5%，其中实体替换攻击危害最大。

    

    检索增强生成（RAG）通过检索到的文档为语言模型提供事实依据，这减少了幻觉，但也带来了新的攻击面：如果检索到的文本被篡改，模型可能会重复虚假信息。我们研究了小型量化模型Llama 3.1 8B在部分检索上下文被投毒时的性能退化程度。研究测试了三种破坏策略——实体替换、数字替换和否定，分别应用于三篇检索段落中的零篇、一篇、两篇或三篇，并在基于FEVER构建的事实核查任务上进行了588次运行的因子实验。当所有三篇段落都被破坏时，准确率从干净上下文下的77.9%下降至43.5%。实体替换翻转了最大比例的原本在干净上下文中回答正确的答案。数字类破坏在投毒段落占少数时保持平稳，一旦投毒段落形成多数则急剧下降，我们通过查询级自助法置信区间再次验证了这一模式。该模型很少凭空捏造新的虚假信息。

    arXiv:2609.09243v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) grounds a language model in retrieved documents, which reduces hallucination but creates a new attack surface: if retrieved text is tampered with, the model may repeat the falsehood. We study how much a small quantized model, Llama 3.1 8B, degrades when a fraction of its retrieved context is poisoned. Three corruption strategies are tested, entity swap, number swap, and negation, each applied to zero, one, two, or three of the three retrieved passages, over a factorial sweep of 588 runs on a fact-checking task built from FEVER. Accuracy falls from 77.9% on clean context to 43.5% when all three passages are corrupted. Entity swap flips the largest share of answers that were correct on clean context. Number-based corruption stays flat while poisoned passages are a minority and jumps once they form a majority, a pattern we re-check with query-level bootstrap intervals. The model rarely invents new fals
    
[^135]: 编码时的自我对话：什么样的注释能帮助代码生成？

    Talking to Itself While Coding: What Makes Comments Help Code Generation?

    [https://arxiv.org/abs/2609.09242](https://arxiv.org/abs/2609.09242)

    该研究发现，代码生成中的注释之所以有效，关键在于其承载的正确解决方案内容而非表面形式——来自成功解决方案的注释可将较弱模型的pass@1提升17.2%，而错误或不相关的注释则无益甚至有害。

    

    大语言模型（LLMs）在编写代码时常常会生成自然语言注释，而这些注释会成为后续代码生成所用上下文的一部分。然而，注释的哪些属性会影响代码生成性能仍不清楚。我们通过观察性分析和受控干预来研究这一问题。在LiveCodeBench上，注释的频率和宽泛的注释意图均不能可靠地预测pass@1。随后，我们用更强的源模型所写的注释块预填充较弱的接收模型，从而将注释的表面形式与其所传达的解决方案内容分离开来。来自通过测试的源解决方案的注释平均使接收模型的pass@1提高了17.2%。相比之下，描述失败解决方案的注释没有带来可靠的提升，而为不同问题编写的注释则会使pass@1降低20.8%。最后，在广泛的模型和提示变体中，大多数接收模型

    arXiv:2609.09242v1 Announce Type: new  Abstract: Large Language Models (LLMs) often generate natural-language comments while writing code, and these comments become part of the context used to generate the code that follows. However, it remains unclear which properties of comments affect code-generation performance. We study this question through observational analyses and controlled interventions. On LiveCodeBench, neither comment frequency nor broad comment intent reliably predicts pass@1. We then prefill weaker recipient models with comment blocks written by stronger source models, allowing us to separate comment surface form from the solution content they convey. Comments from source solutions that pass the tests raise recipient pass@1 by 17.2% on average. In contrast, comments describing failed solutions provide no reliable gain, while comments written for a different problem reduce pass@1 by 20.8%. Finally, across a wide range of models and prompt variants, most recipient models 
    
[^136]: 动态稀疏混合专家模型的分布一致性推理

    Distribution-Consistent Inference for Dynamic Sparse Mixture-of-Experts

    [https://arxiv.org/abs/2609.09241](https://arxiv.org/abs/2609.09241)

    提出逐层分布对齐方法，通过在推理时校正动态减少激活专家所引起的输出分布偏移，在不重新训练的情况下降低计算成本并缓解下游性能下降。

    

    混合专家架构已成为在大型基础模型中扩展模型容量同时保持高效推理的强大范式。然而，大多数MoE模型使用固定的top-k专家选择策略，为每个token分配相同的专家预算，即使更少的专家可能已经足够。推理时的动态top-k路由可以在不重新训练的情况下减少计算量，但现有方法往往忽略了偏离训练时路由配置所导致的分布偏移。我们证明减少激活专家的数量会持续增加稀疏MoE输出的RMS尺度和方差，从而引起表示不匹配，这在专家容量损失之外还会导致下游性能下降。为了解决这一可纠正的组成部分，我们提出了逐层分布对齐，这是一种轻量级的推理时校正方法，利用逐层校准（此处摘要被截断）。

    arXiv:2609.09241v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) architectures have emerged as a powerful paradigm for scaling model capacity while preserving efficient inference in large foundation models. However, most MoE models use a fixed top-$k$ expert selection policy, assigning the same expert budget to every token even when fewer experts may be sufficient. Inference-time dynamic top-$k$ routing can reduce computation without retraining, but existing methods often overlook the distributional shift caused by deviating from the training-time routing configuration. We show that reducing the number of activated experts consistently increases the RMS scale and variance of SMoE outputs, inducing a representation mismatch that contributes to downstream performance degradation in addition to the loss of expert capacity. To address this correctable component, we propose Layer-wise Distribution Alignment (LDA), a lightweight inference-time correction that uses layer-wise calib
    
[^137]: 将训练后三值化扩展至 Qwen3-8B：能力保持、可复现性、无损打包与打包执行

    Scaling Post-Training Ternarisation to Qwen3-8B Capability Retention, Reproduction, Lossless Packing, and Packed Execution

    [https://arxiv.org/abs/2609.09240](https://arxiv.org/abs/2609.09240)

    本文将激进的三值化训练后量化流程从 Qwen3-4B 成功扩展至 Qwen3-8B，通过外部复现验证、能力对比分析、有效比特核算、无损格感知打包及直接打包执行等端到端表征，证明了 8B 模型在超低比特量化后仍能保持较高能力（零样本平均准确率 64.6%，FP16 为 72.4%）。

    

    超低比特语言模型有望降低存储和内存流量，但名义上的“1.58比特”标签并未说明实际部署的表示形式或其执行成本。我们研究了将激进的训练后转换流程从 Qwen3-4B 扩展到 Qwen3-8B 的工作。该转换在仅权重 A16 配置下采用 KOTMS 旋转、E2M-ATQ 自适应三值化以及 GPTQ 风格的误差补偿。我们并不声称这些算法是新的。我们的贡献在于端到端的规模化表征：外部复现门槛、匹配的 4B/8B 能力分析、跨语料库困惑度、有效比特核算、无损格感知打包，以及直接打包执行。8B 模型在三个语料库上的困惑度比达到 1.361 倍，其中 WikiText-2、C4 和 PTB 的比率分别为 1.318 倍、1.393 倍和 1.371 倍。在 n=500 的八项零样本任务上，平均准确率为 64.6%，而 FP16 为 72.4%，对应约 78.5% 的机会校正表现。

    arXiv:2609.09240v1 Announce Type: cross  Abstract: Ultra-low-bit language models promise reductions in storage and memory traffic, but a nominal "1.58-bit" label does not specify the deployed representation or its execution cost. We study a scale-up of an aggressive post-training conversion pipeline from Qwen3-4B to Qwen3-8B.   The conversion uses KOTMS rotation, E2M-ATQ adaptive ternarisation, and GPTQ-style error compensation in a weight-only A16 configuration. We do not claim these algorithms as new. Our contribution is the end-to-end scale-up characterisation: an external reproduction gate, matched 4B/8B capability analysis, cross-corpus perplexity, effective-bit accounting, lossless lattice-aware packing, and direct packed execution.   The 8B model reaches a three-corpus perplexity ratio of 1.361x, with WikiText-2, C4, and PTB ratios of 1.318x, 1.393x, and 1.371x. On eight zero-shot tasks at n = 500, mean accuracy is 64.6% versus 72.4% for FP16, corresponding to 78.5% chance-corre
    
[^138]: 子代理与代理技能的对比：为长时程智能体任务执行可复用知识

    Subagents vs Agent Skills: Executing Reusable Knowledge for Long-Horizon Agentic Tasks

    [https://arxiv.org/abs/2609.09233](https://arxiv.org/abs/2609.09233)

    该研究发现，将技能包作为拥有独立全新上下文窗口的子代理来执行，相比将技能指令加载到主上下文的传统智能体技能方式，在解决长时程任务时表现更优，因为其避免了上下文信息累积导致的推理质量下降。

    

    语言模型智能体如何有效利用可复用知识库来解决长时程任务？近期的研究日益关注“智能体技能”，即以技能包形式表示的可复用能力——技能包是包含指令、脚本及其他资源的多文件捆绑集合，帮助智能体执行特定任务。智能体技能通常通过将技能指令加载到智能体的上下文中，并依赖智能体遵循这些指令来执行。然而，随着任务时程的增长，这种方法变得越来越脆弱，因为上下文窗口中积累的信息越多，推理质量就会下降。我们研究了一种替代方法，即将技能包作为子代理来调用。子代理执行并非将技能指令加载到主上下文中，而是生成全新的、专用于解决单个子任务的上下文窗口。我们表明，子代理执行优于智能体技能执行……

    arXiv:2609.09233v1 Announce Type: cross  Abstract: How can language model agents effectively leverage libraries of reusable knowledge to solve long-horizon tasks? Recent work has increasingly focused on agent skills: reusable capabilities represented as skill packages, i.e., multi-file bundles containing instructions, scripts, and other resources that help agents perform specific tasks. Agent skills are typically executed by loading their skill instructions into an agent's context and relying on the agent to follow them. As task horizons grow, however, this approach becomes increasingly brittle, because reasoning quality degrades as more information accumulates in the context window. We investigate an alternative approach in which skill packages are instead invoked as subagents. Rather than loading skill instructions into the main context, subagent execution spawns fresh context windows dedicated to solving individual subtasks. We show that subagent execution outperforms agent-skill ex
    
[^139]: 计算受限的安全保障——资源约束下的覆盖、验证与响应

    Compute-Bounded Security Assurance - Coverage, Verification, and Response under Resource Constraints

    [https://arxiv.org/abs/2609.09229](https://arxiv.org/abs/2609.09229)

    该论文提出了一个资源受限的安全保障分析框架，区分了重复成功与独特覆盖等不同量，推导出覆盖率公式 $C_n = 1 - E[(1-\Theta)^n]$ 及其极限值，并证明正的两两相关性并不必然意味着覆盖率上限低于一。

    

    额外的推理计算可以增加正确解决安全保障任务的数量，但重复成功、独特覆盖、可接受的证据和运营保护是不同的量。我们开发了一个资源受限框架将它们区分开来。对于具有潜在成功概率 $\Theta$ 的重复条件独立尝试，覆盖率为 $C_n = 1 - E[(1-\Theta)^n]$，其极限值为 $1 - P(\Theta = 0)$。正的两两结果相关性本身并不意味着上限低于一：我们构建了两个具有相同平均成功率和两两相关性、但极限覆盖率不同的模型。我们将这一结果与用于估计均值的有限样本量区分开来，并说明为什么有限预算的观测通常无法识别渐近支撑上限。随后，我们将覆盖率与可出错的证据检查、事实依据的适当评分以及完整的资源核算联系起来。

    arXiv:2609.09229v1 Announce Type: cross  Abstract: Additional inference compute can increase the number of correctly resolved security-assurance tasks, but repeated success, unique coverage, accepted evidence, and operational protection are different quantities. We develop a resource-constrained framework that separates them. For repeated conditionally independent attempts with latent success probability $\Theta$, coverage is $C_n = 1 - E[(1-\Theta)^n]$, and its limiting value is $1 - P(\Theta = 0)$. Positive pairwise outcome correlation does not by itself imply a ceiling below one: we construct two models with the same mean success and pairwise correlation but different limiting coverage. We distinguish this result from the effective sample size used to estimate a mean, and show why finite-budget observations cannot generally identify an asymptotic support ceiling. We then connect coverage to fallible evidence checking, proper scoring of factual grounding, complete resource accounting
    
[^140]: 通用人工智能中的自适应纠缠博弈模块

    Adaptive Entangled Game Modules in Artificial General Intelligence

    [https://arxiv.org/abs/2609.09226](https://arxiv.org/abs/2609.09226)

    该论文提出基于广义行为智能非局域概率波方程的框架，通过对中国股市数据实证发现自适应纠缠博弈模式可解释89%的决策行为，从而间接支持大脑非局域纠缠神经纤维假说。

    

    我们引入了一个概率波框架，用于建模相互作用的自适应智能体的集体行为，并通过广义行为智能（GBI）非局域概率波方程推导出可检验的本征模。该框架以分析性机制捕获了广泛的人类智能行为，并通过集体交易者行为提供了一种间接检验大脑中Liu-Chen-Ao（LCA）非局域纠缠神经纤维假说的方法。我们对中国股票市场日内交易数据的实证分析表明，自适应纠缠博弈模式解释了82-94%（总体为89%）的观察到的决策模式，这与基于独立理性主体的新古典金融学预测形成鲜明对比。此外，2-12%的行为表现出对日内新闻、事件和环境的适应，其特征为双均衡状态和参考点的突然转变，而纯独立模式则……

    arXiv:2609.09226v1 Announce Type: cross  Abstract: We introduce a probability-wave framework for modeling the collective behavior of interacting adaptive agents, deriving testable eigenmodes through a generalized behavioral intelligence (GBI) nonlocal probability-wave equation. This framework captures a broad range of human intelligence behaviors with analytical mechanisms and offers an indirect method to examine the Liu-Chen-Ao (LCA) hypothesis of nonlocal entangled nerve fibers in the brain through collective trader behaviors. Our empirical analysis of Chinese intraday stock market data demonstrates that adaptive entangled game modes explain 82-94% (89% overall) of observed decision patterns, a sharp contrast to the predictions of neoclassical finance based on independent rational agents. Moreover, 2-12% of behaviors show adaption to intraday news, events, and environments, characterized by dual equilibrium states and abrupt reference point shifts, while purely independent modes occu
    
[^141]: 仅凭分数不足以证明发现：用于审计AI研究代理的发现认证协议

    Scores Alone Do Not Prove Discovery: The Discovery Certification Protocol for Auditing AI Research Agents

    [https://arxiv.org/abs/2609.09219](https://arxiv.org/abs/2609.09219)

    该论文提出发现认证协议（DCP），通过密封评估、隐藏研究历史的匹配对照、恢复见证与有限样本界限等可执行测试，论证仅凭分数不足以证明AI研究代理的真实发现能力，并实现对AI研究成果的严格审计与认证。

    

    AI研究代理结合先验知识、公开来源和实验反馈来产生有用的结果。发现认证协议将关于这些结果的声明转化为可执行的恢复与反馈测试。门控1在密封评估上验证有用改进。门控2向匹配的代理提供注册的起始信息和观察到的网络内容，同时隐藏目标研究历史。每个达到数值目标的有效方法都会提供恢复见证并触发核心否决。DCP核心要求充分的控制、零次观察到的恢复，以及在一个全新注册回合中恢复概率的有限样本界限。可选的门控3从共享检查点测量真实反馈相对于指定中性策略的平均效应。DCP证据在独立零假设校准和注册效应余量之后加入这一效应。两个受控审计在完整协议下执行。

    arXiv:2609.09219v1 Announce Type: cross  Abstract: AI research agents combine prior knowledge, public sources, and experimental feedback to produce useful results. The Discovery Certification Protocol (DCP) turns claims about these results into executable recovery and feedback tests. Gate 1 validates useful improvement on sealed evaluation. Gate 2 gives matched agents the registered starting information and observed Web content while withholding the target research history. Every valid method reaching the numerical target supplies a recovery witness and triggers the Core veto. DCP Core requires adequate controls, zero observed recoveries, and a finite-sample bound on recovery in one fresh registered episode. Optional Gate 3 measures the average effect of truthful feedback relative to a specified neutral policy from a shared checkpoint. DCP Evidence adds this effect after independent null calibration and a registered effect margin. Two controlled audits exercise the complete protocol in
    
[^142]: 具身小语言模型中的几何条件化：0.8B混合模型中的训练控制与鲁棒性诊断

    Geometry Conditioning in an Embodied SLM: Training Controls and Robustness Diagnostics in a 0.8B Hybrid Model

    [https://arxiv.org/abs/2609.09213](https://arxiv.org/abs/2609.09213)

    该研究在0.8B混合语言模型上系统测试了六种几何条件化方案，发现几何输入并未带来可靠的操作任务性能提升，而基于状态的相对坐标策略在鲁棒性测试中远优于视觉策略。

    

    我们研究了物理状态输入如何影响一个经过改造用于操作任务的0.8B混合语言模型，该模型仅使用620万可训练参数。六种条件在三个LIBERO-Spatial任务上进行训练，并在三个随机种子和540个保留测试回合上进行评估。将循环衰减门控条件化于几何增量可获得28.9%的成功率，相比之下，在训练期间将这些增量打乱时成功率为36.7%，而完全不使用显式物体/目标几何时为24.4%。两种几何策略在评估时均接收正确的输入。使用相同增量的token适配器得分为27.8%；差异在不同种子间波动，仍无定论。Token时钟条件化得分为11.1%，其中包括一个未能收敛的种子。在单独的鲁棒性测试中，仅使用状态的相对坐标策略在坐标系重标注下仍保持7/10的成功率，而所有四个测试的视觉策略在物体位移5厘米后成功率最多降至3/20。这些结果表明（几何条件化）没有可靠的优势。

    arXiv:2609.09213v1 Announce Type: cross  Abstract: We study how physical-state inputs affect a 0.8B hybrid language model adapted for manipulation with 6.2M trainable parameters. Six conditions are trained on three LIBERO-Spatial tasks and evaluated over three seeds and 540 held-out rollouts. Conditioning recurrent decay gates on geometric increments yields 28.9% success, compared with 36.7% when those increments are shuffled during training and 24.4% without explicit object/goal geometry. Both geometry policies receive correct inputs at evaluation. A token adapter using the same increments scores 27.8%; differences vary across seeds and remain inconclusive. Token-clock conditioning scores 11.1%, including one seed that fails to converge. In separate robustness tests, a state-only relative-coordinate policy retains 7/10 success under frame relabeling, whereas all four tested visual policies fall to at most 3/20 after a 5 cm object displacement. These results show no reliable advantage 
    
[^143]: AgentHijack：针对多模态计算机使用代理的视觉补丁攻击

    AgentHijack: Visual Patch Attacks on Multimodal Computer-Use Agents

    [https://arxiv.org/abs/2609.09212](https://arxiv.org/abs/2609.09212)

    本文提出AgentHijack端到端评估框架，首次证明局部视觉补丁能够劫持计算机使用代理从截图输入到环境执行的完整链条并注入恶意终端命令，在五种主流VLM后端上实现最高84.5%的目标攻击成功率。

    

    本文提出了一个针对计算机使用代理（CUA）的图像触发命令注入的端到端评估框架。其目标是测试局部视觉补丁能否在截图输入、VLM生成、动作解析和环境执行的完整链条中引发可验证的环境后果。我们在作者控制的GitHub Pages页面和本地部署的CSDN克隆站点上训练并部署补丁，并在真实环境中对五个开源或公开可用的GUI代理或视觉语言模型（VLM）后端进行评估。实验共汇总了600个实例级在线案例，其中T-ASR、TAPR和E2E-ASR分别达到84.5%、47.0%和20.3%。轨迹分析进一步表明，在某些成功案例中，代理会先执行恶意终端命令，然后继续完成原本的良性任务。这些结果表明，经过优化的局部视觉信号不仅能影响VLM……（摘要原文在此处截断）

    arXiv:2609.09212v1 Announce Type: cross  Abstract: This paper presents an end-to-end evaluation framework for image-triggered command injection against computer-use agents (CUAs). The goal is to test whether a local visual patch can induce verifiable environmental consequences along the full chain of screenshot input, VLM generation, action parsing, and environment execution. We train and deploy patches on author-controlled GitHub Pages pages and a locally deployed CSDN clone, and evaluate them in real environments across five open-source or publicly available GUI-agent or vision-language-model (VLM) backends. Our experiment aggregates 600 instance-level online cases, with T-ASR, TAPR, and E2E-ASR reaching 84.5%, 47.0%, and 20.3%, respectively. Trajectory analysis further shows that in some successful cases the agent first executes a malicious terminal command and then continues the original benign task. These results indicate that optimized local visual signals can affect not only VLM
    
[^144]: OpenDiscoveryTrace：用于评估AI科学家工作流的过程轨迹

    OpenDiscoveryTrace: Process Traces for Evaluating AI Scientist Workflows

    [https://arxiv.org/abs/2609.09203](https://arxiv.org/abs/2609.09203)

    提出了OpenDiscoveryTrace公开数据集，包含558条记录AI科学家智能体逐步推理过程（思考、工具调用、错误、置信度等）的完整轨迹，弥补了现有基准只评估最终输出、无法审计科学推理方法的缺陷。

    

    现有的自主AI科学家基准测试仅评估最终输出——生成的代码、假设或论文——却丢弃了获得这些输出所经历的推理过程。这使得审计科学方法论、诊断失败模式或区分系统性推理与侥幸猜测变得不可能。我们提出了OpenDiscoveryTrace，这是一个包含558条完整AI科学智能体轨迹的公开数据集，它捕捉模型如何推理，而不仅仅是它们产出了什么。每条轨迹记录了每步包含9个字段的结构化轨迹——包括思考、工具调用、观察结果、错误、修订触发器以及自我报告的置信度——这些数据来自模型执行124项科学任务的过程，任务涵盖药物发现、材料科学、基因组学和科学文献分析。该数据集涵盖七个模型：三个前沿模型（GPT-5.4、Claude Opus 4.6和Gemini 3.1 Pro；每个模型124条轨迹，在各领域完全均衡）

    arXiv:2609.09203v1 Announce Type: new  Abstract: Existing benchmarks for autonomous AI scientists evaluate only final outputs---generated code, hypotheses, or papers---yet discard the reasoning process by which those outputs were obtained. This makes it impossible to audit scientific methodology, diagnose failure modes, or distinguish systematic reasoning from fortunate guessing. We present \textbf{OpenDiscoveryTrace}, a public dataset of 558 complete AI scientific agent trajectories that captures how models reason, not just what they produce. Each trajectory records a structured 9-field-per-step trace---including thoughts, tool calls, observations, errors, revision triggers, and self-reported confidence---as models execute 124 scientific tasks spanning drug discovery, materials science, genomics, and scientific literature analysis. The dataset covers seven models: three frontier models (GPT-5.4, Claude Opus 4.6, and Gemini 3.1 Pro; 124 trajectories each, fully balanced across domains 
    
[^145]: 面向宫颈细胞学分类的可靠性感知混合K集成选择方法：融合判别能力、校准性与选择性预测

    Reliability-Aware Hybrid-K Ensemble Selection for Cervical Cytology Classification: Integrating Discrimination, Calibration, and Selective Prediction

    [https://arxiv.org/abs/2609.09189](https://arxiv.org/abs/2609.09189)

    提出一种可靠性感知的Hybrid-K集成选择框架，将判别性能、校准误差与选择性预测等多维可靠性指标统一纳入模型排名与集成构建，在宫颈细胞学分类中实现了兼顾准确性与不确定性可靠性的临床级模型选择。

    

    仅有高分类准确率不足以支撑临床图像分析，校准良好的置信度和可靠的不确定性估计同样至关重要。本研究提出了一种可靠性感知的Hybrid-K集成选择框架，用于基于SIPaKMeD数据集的多分类宫颈细胞学分类任务。研究在固定的分层五折划分和三个训练随机种子下评估了九种深度学习架构。在进行事后温度缩放后，模型通过宏F1、准确率、AUROC、期望校准误差（ECE）、最差类别ECE（WC-ECE）、风险-覆盖曲线下面积（AURC）、Brier分数和负对数似然（NLL）等指标进行评估。模型采用等权综合得分进行排名，并利用软投票从排名靠前的模型构建Hybrid-K集成。研究通过5,000个狄利克雷采样的指标权重向量、留一指标分析以及校正后的配对显著性检验来考察框架的鲁棒性。

    arXiv:2609.09189v1 Announce Type: cross  Abstract: High classification accuracy alone is insufficient for clinical image analysis, where calibrated confidence and reliable uncertainty estimates are essential. This study proposes a reliability-aware Hybrid-K ensemble selection framework for multiclass cervical cytology classification using the SIPaKMeD dataset. Nine deep learning architectures were evaluated using a fixed stratified five-fold partition and three training seeds. After post-hoc temperature scaling, models were assessed using macro-F1, accuracy, AUROC, expected calibration error (ECE), worst-class ECE (WC-ECE), area under the risk-coverage curve (AURC), Brier score, and negative log-likelihood (NLL). Models were ranked using an equal-weight composite score, and Hybrid-K ensembles were formed from the top-ranked models using soft voting. Robustness was examined using 5,000 Dirichlet-sampled metric-weight vectors, leave-one-metric-out analysis, and corrected paired testing a
    
[^146]: AgenticGen：面向广告的奖励引导智能体视频生成

    AgenticGen: Reward-Guided Agentic Video Generation for Advertising

    [https://arxiv.org/abs/2609.09187](https://arxiv.org/abs/2609.09187)

    AgenticGen提出了一种奖励引导的智能体框架，将广告视频生成分解为策略选择和草稿生成两个可训练的推理阶段，通过从线上业务反馈中学习性能奖励并结合人类质量准则奖励来监督策略优化，从而实现以线上业务指标为导向的广告视频生成。

    

    广告视频生成不仅仅是一个视频合成任务，更是一个以产品为条件的推理问题，其成功与否由线上业务指标来衡量。近期的视频基础模型能够根据多模态条件生成逼真的视频片段，但它们并未优化如何将产品转化为有效的广告，也未考虑如何利用线上业务反馈来改进未来的生成。为了闭合这一反馈循环，我们提出了AgenticGen，一个奖励引导的智能体框架，它将广告视频生成分解为两个可训练的推理阶段——策略选择和草稿生成，从而暴露出线上业务反馈可以监督的优化目标。AgenticGen从累积的线上反馈中学习基于性能的奖励，以及与人类质量标准对齐的互补的基于准则的奖励，然后用它们来监督策略优化。DPO首先移动智能体策略……

    arXiv:2609.09187v1 Announce Type: cross  Abstract: Advertising video generation is not only a video synthesis task, but also a product-conditioned reasoning problem whose success is measured by online business metrics. Recent video foundation models can generate realistic clips from multimodal conditions, yet they do not optimize how a product should be transformed into an effective advertisement or how future generation should be improved from online business feedback. To close this loop, we propose AgenticGen, a reward-guided agentic framework that decomposes advertising video generation into two trainable reasoning stages, strategy selection and draft generation, thereby exposing optimization targets that online business feedback can supervise. AgenticGen learns a performance-based reward from accumulated online feedback and a complementary rubric-based reward aligned with human quality standards, then uses them to supervise policy optimization. DPO first moves the agentic policies 
    
[^147]: 全模态交互智能体技术报告

    Omni Interaction Agent Technical Report

    [https://arxiv.org/abs/2609.08977](https://arxiv.org/abs/2609.08977)

    Gander是一个端到端模型，通过小脑-大脑协作架构在单一框架内统一了全模态感知、实时全双工交互和智能体能力。

    

    arXiv:2609.08977v2 公告类型：cross 摘要：在本工作中，我们提出了Gander，一个端到端模型，它在单一框架内统一了全模态感知、实时交互和智能体能力。与传统的基于轮次的范式不同，Gander能够持续接收来自多种模态的流式输入，包括视频、语音和文本，从而在日常对话和复杂的工作流导向的智能体场景中实现自然的全双工交互。用户可以随时打断模型，同时模型也可以主动提供中间反馈或提出后续问题。为了原生支持这些能力，Gander采用了两个关键的架构设计：1）它采用了小脑-大脑协作框架，其中小脑负责实时交互和全模态对话能力，而大脑则处理复杂推理和更高层次的智能体任务，两个组件通过工具调用持续进行交互。

    arXiv:2609.08977v2 Announce Type: cross  Abstract: In this work, we present Gander, an end-to-end model that unifies omni perception, realtime interaction, and agentic capabilities within a single framework. In contrast to turn-based conventional paradigms, Gander continuously receives streaming inputs across multiple modalities, including video, speech, and text, enabling natural full-duplex interaction in both everyday conversations and complex workflow-oriented agent scenarios. Users can interrupt the model at any time, while the model can also proactively provide intermediate feedback or ask follow up questions. To natively support these capabilities, Gander adopts two key architectural designs: 1) It employs a Cerebellum-Brain collaborative framework, in which the Cerebellum is responsible for realtime interaction and omni conversational capabilities, while the Brain handles complex reasoning and higher-level agentic tasks. The two components interact continuously through tool cal
    
[^148]: Hi-FLoop：面向多时间尺度世界建模的分层状态反馈环路

    Hi-FLoop: Hierarchical State-Feedback Loops for Multi-Timescale World Modeling

    [https://arxiv.org/abs/2609.08796](https://arxiv.org/abs/2609.08796)

    提出了分支一致的多时间尺度状态反馈框架Hi-FLoop，通过让所有智能体在8秒推演全程共享同一场景级World联合假设，并在该分支内自适应调整目标、预览和控制状态，解决了多智能体交通仿真长时程闭环生成中跨尺度一致性与多模态分支连贯性的问题。

    

    多智能体交通仿真旨在从地图和观测历史中生成多样、协调且物理真实的未来场景。长时程闭环生成必须协调多个决策时间尺度，同时其上下文会随生成的状态不断演化。现有方法往往从初始场景直接展开长远的未来，并将意图、交互和运动整体化处理，从而削弱了跨尺度的一致性和适应能力。多模态推演还带来了进一步的一致性问题：跨智能体或跨提交独立地重新选择模式，可能会拼凑出彼此不相容的未来，而不是保持一个连贯的联合分支。我们提出了Hi-FLoop，一个分支一致的多时间尺度状态反馈框架。八个场景级的World（世界）表示联合假设；在8秒推演的全部16个提交中，所有智能体共享同一个被选中的World身份，而Goal（目标）、Preview（预览）和Control（控制）状态则在该分支内进行自适应调整。8秒的目标锚点……

    arXiv:2609.08796v2 Announce Type: cross  Abstract: Multi-agent traffic simulation seeks diverse, coordinated, and physically realistic futures from maps and observed history. Long-horizon closed-loop generation must reconcile multiple decision time scales while its context evolves with generated states. Existing methods often unfold long futures from an initial scene and resolve intent, interaction, and motion monolithically, weakening cross-scale consistency and adaptation. Multimodal rollout poses a further consistency problem: independently reselecting modes across agents or commits can stitch together incompatible futures instead of preserving a coherent joint branch. We present Hi-FLoop, a branch-consistent multi-timescale state-feedback framework. Eight scene-level Worlds represent joint hypotheses; all agents share one selected World identity throughout all 16 commits of an 8-second rollout, while Goal, Preview, and Control states adapt within that branch. An 8-second Goal ancho
    
[^149]: EvolveScaler：通过可执行状态机与自然语言渲染合成信息演化语境

    EvolveScaler: Synthesizing Information-Evolution Contexts via Executable State Machines and Natural-Language Rendering

    [https://arxiv.org/abs/2609.08435](https://arxiv.org/abs/2609.08435)

    EvolveScaler提出一种代码驱动的数据合成框架，先用人工编写的操作规范定义信息演化过程并生成可执行、可验证的模拟器，再将其渲染为自然语言多轮语境，从而可靠地构建需要追踪信息修订与撤销的长上下文任务数据。

    

    在持续性交互中，长上下文所编码的可能是一个不断演化的过程，而非固定的记录：后续事件可以修订或撤销先前的信息，从而改变哪些内容依然有效以及哪些结论能够成立。我们将这一设定称为信息演化。解决IE任务需要识别有效记录、按顺序应用更新，并从事件历史中重建与查询相关的状态。现有的以文本为先的合成流水线使这类数据难以验证，因为状态转移和答案逻辑仍然是隐式的。我们提出EvolveScaler，这是一个代码驱动的框架，先定义信息演化过程，再将其渲染为自然语言。人工编写的操作规范定义了状态转移、记录有效性、难度控制以及可执行的答案逻辑；随后由一个强大的大语言模型根据每份规范合成一个自包含的模拟器。执行经过验证的模拟器即可生成自然语言的多轮（对话数据）……

    arXiv:2609.08435v2 Announce Type: new  Abstract: In persistent interactions, long contexts may encode an evolving process rather than a fixed record: later events can revise or revoke earlier information, changing what remains valid and what conclusions follow. We call this setting information evolution (IE). Solving IE requires identifying valid records, applying updates in order, and reconstructing the query-relevant state from the event history. Existing text-first synthesis pipelines make such data difficult to verify because state transitions and answer logic remain implicit. We introduce EvolveScaler, a code-driven framework that defines information evolution before rendering it as natural language. Human-authored operational specifications define state transitions, record validity, difficulty controls, and executable answer logic; a strong LLM then synthesizes a self-contained simulator from each specification. Executing validated simulators produces natural-language multi-turn 
    
[^150]: RevalExo：面向老年人及临床队列的惯性与视觉运动模式识别功能性日常活动基准

    RevalExo: A Functional Daily-Activity Benchmark for Inertial and Visual Locomotion Mode Recognition in Older Adults and Clinical Cohorts

    [https://arxiv.org/abs/2609.08090](https://arxiv.org/abs/2609.08090)

    该论文提出了RevalExo——一个面向老年人及临床人群（含中风幸存者）的功能性日常活动基准数据集，为惯性与视觉运动模式识别提供时间精确标注，以支持外骨骼等辅助设备在真实临床场景下的开发与评估。

    

    面向行动障碍人士的辅助设备（如动力外骨骼）依赖于准确的运动模式识别，以便在日常活动中调整控制策略并提供适当的辅助。然而，现有的公开基准数据集通常采集自健康成年人，缺乏检测模式转换所需的时间精确标注，或仅关注有限的任务集合。为支持在真实临床约束和日常行动需求下的开发与评估，我们提出了RevalExo，一个面向惯性与视觉运动模式识别的功能性日常活动基准。RevalExo围绕一个标准化的、经过临床与生态学验证的日常活动协议构建，该协议反映了老年人群和临床人群累积的日常行动需求。该基准涵盖来自三个队列的27名参与者：无行动障碍的老年人、中风幸存者，以及存在行动障碍的老年人（摘要至此被截断）。

    arXiv:2609.08090v2 Announce Type: new  Abstract: Assistive devices for people with mobility impairments, such as powered exoskeletons, rely on accurate locomotion mode recognition to adapt control strategies and provide appropriate assistance during daily activities. However, public benchmarks are typically collected from healthy adults, lack temporally precise labels necessary for detecting mode transitions, or focus on a limited set of tasks. To support development and evaluation under realistic clinical constraints and daily mobility demands, we introduce RevalExo, a functional daily-activity benchmark for inertial and visual locomotion mode recognition. RevalExo is built around a standardized, clinically and ecologically validated daily-activity protocol reflecting the cumulative everyday mobility demands in ageing and clinical populations. The benchmark includes 27 participants across three cohorts: older adults without mobility impairments, stroke survivors, and older adults with
    
[^151]: SAFER-Activities：用于跌倒事件与日常活动智能评估的数据集

    SAFER-Activities: A Dataset for Smart Assessment of Fall Events and Routine Activities

    [https://arxiv.org/abs/2609.08038](https://arxiv.org/abs/2609.08038)

    提出了SAFER-Activities数据集，包含66小时多摄像头视频、85,310个动作实例及30个动作类别的帧级标注，支持跌倒检测与日常活动监测的在线动作识别研究，并特别涵盖轮椅使用场景。

    

    智能医疗监测系统需要精确的动作识别，以确保在跌倒等关键情况下的健康保障和及时干预，特别是对于行动不便的人群。现有数据集通常基于视频片段构建，缺乏在动作实时展开时进行在线识别所需的帧级细节。为解决这一问题，我们提出了SAFER-Activities，一个用于跌倒检测和身体活动监测的数据集，其中包含专门针对轮椅使用场景的子集。该数据集由多个摄像头拍摄，包含超过66小时的视频数据，涵盖85,310个动作实例以及30个动作类别的帧级标注。我们在SAFER-Activities上使用2D和3D骨架模型、冻结骨干网络的RGB模型以及多模态融合策略对动作识别进行了基准测试，并在实验室内、分布外以及跨数据集测试集上进行了评估。基于骨架的模型在域偏移下泛化能力最佳；融合冻结的RGB特征……

    arXiv:2609.08038v2 Announce Type: cross  Abstract: Smart healthcare monitoring systems require precise action recognition to ensure well-being and timely intervention in critical situations such as falls, particularly for mobility-challenged individuals. Existing datasets are often clip-based, lacking the frame-level detail needed to recognize actions online, as they unfold. To address this, we introduce SAFER-Activities, a dataset for fall detection and physical activity monitoring, with a dedicated subset for wheelchair use scenarios. It comprises over 66 hours of video data captured by multiple cameras, with 85,310 action instances and frame-level annotations for 30 action classes. We benchmark action recognition on SAFER-Activities with 2D and 3D skeleton models, RGB models with frozen backbones, and multimodal fusion strategies, and evaluate on in-lab, out-of-distribution, and cross-dataset test sets. Skeleton-based models generalize best under domain shift; fusing frozen RGB feat
    
[^152]: FrogNano：通过在线任务合成训练一个4B编码智能体

    FrogNano: Training a 4B Coding Agent via Online Task Synthesis

    [https://arxiv.org/abs/2609.07925](https://arxiv.org/abs/2609.07925)

    FrogNano是一个4B编码智能体，仅通过强化学习在约1500个合成任务环境中训练，其关键创新是在线任务合成流水线能在当前模型可学习性前沿生成校准任务，无需从大模型蒸馏即可训练出具有竞争力的小型编码智能体。

    

    我们提出了FrogNano，这是一个4B参数的编码智能体，旨在即使在资源受限的环境下也能高效且有效地解决软件工程（SWE）任务。它完全通过强化学习在大约1,500个包含合成任务的SWE环境中进行后训练。提升性能的一个关键要素是在线任务合成流水线，该流水线能够创建针对当前检查点可学习性前沿进行校准的任务。本报告提供了证据，表明仅使用合成任务就可以训练出具有竞争力的小型编码智能体，而无需传统的从更大模型进行蒸馏，并且在当前智能体的可学习性前沿生成任务至关重要。我们报告了训练方法的细节、跨多种环境的评估以及深入分析，为我们持续探索可在最低限度硬件上运行的轻量级且功能强大的编码智能体奠定了基础。

    arXiv:2609.07925v2 Announce Type: new  Abstract: We present FrogNano, a 4B coding agent designed to tackle software engineering (SWE) tasks efficiently and effectively, even under resource-constrained environments. It is post-trained exclusively via RL on around 1,500 SWE environments with synthetic tasks. A key ingredient for improving performance is an online task synthesis pipeline that creates tasks calibrated to the frontier of learnability for the current checkpoint. This report provides evidence that competitive small coding agents can be trained with synthetic tasks alone, without traditional distillation from larger models, and that generating tasks at the learnability frontier of the current agent is important. We report details on the training methodology, evaluations across diverse environments, and in-depth analyses, serving as a foundation for our ongoing exploration of lightweight yet capable coding agents that can run on minimal hardware.
    
[^153]: Fine PT-PT Web：一个高质量的410亿词元欧洲葡萄牙语网页数据集

    Fine PT-PT Web: A High-Quality 41 Billion Tokens Data Collection of the European Portuguese Web

    [https://arxiv.org/abs/2609.07699](https://arxiv.org/abs/2609.07699)

    本文提出一个高效数据处理流水线，从411TB的原始网页数据中构建了高质量的410亿词元欧洲葡萄牙语语料库，其创新的后抓取预处理模块通过在过滤前去除样板文本和重复行，使最终文档产出量提升了19.04%。

    

    为欧洲葡萄牙语（PT-PT）等区域性语言变体策划网络语料库，严重受制于方言重叠（主要与巴西葡萄牙语PT-BR的重叠）和数据处理规模的瓶颈。本文提出了一个高效的流水线，从葡萄牙语网页中策划出一个可用于生产环境的PT-PT语料库，涵盖来自Arquivo.pt的411 TB原始数据。我们引入了一种新颖的抓取后处理模块，可在过滤之前去除样板文本和行级重复内容。这一早期干预通过挽救被标准启发式过滤器过早丢弃的有效文本，使最终文档产出量增加了19.04%。结合严格的语言识别、加权模糊去重和神经质量分类，我们的流水线提供了一个可扩展的框架以及一个面向大语言模型预训练优化的干净且具有代表性的语料库。

    arXiv:2609.07699v2 Announce Type: cross  Abstract: Curating Web corpora for regional language variants like European Portuguese (PT-PT) is heavily bottlenecked by dialectal overlap (mainly with PT-BR) and data processing scale. This paper presents an efficient pipeline to curate a production-ready PT-PT corpus from the Portuguese Web, spanning 411 TB of raw data from Arquivo.pt. We introduce a novel post-scraping block that removes boilerplate and line duplicates prior to filtering. This early-stage intervention increases final document yield by 19.04% by rescuing valid text that standard heuristic filters prematurely discard. Integrated with rigorous language identification, weighted fuzzy deduplication, and neural quality classification, our pipeline offers a scalable framework and a clean, representative corpus optimized for LLM pre-training.
    
[^154]: 准确率是不够的：一种基于散度的量化大语言模型保真度损失评估方法

    Accuracy is Not Enough: A Divergence-Based Approach to Evaluate Fidelity Loss in Quantized LLMs

    [https://arxiv.org/abs/2609.07664](https://arxiv.org/abs/2609.07664)

    该论文提出了一种分布敏感的评估框架，通过计算量化模型与全精度模型在token决策边界处全词表预测分布之间的统计距离（如Jensen-Shannon散度和总变差距离）来量化保真度损失，弥补了仅依赖准确率评估量化模型质量的不足。

    

    将大语言模型（LLM）部署到内存受限的边缘设备上，很大程度上依赖于激进的后训练量化。然而，对这些模型的评估主要基于零样本任务准确率，该指标仅取决于argmax预测，对底层预测分布的变化并不敏感。因此，在渐进量化过程中，准确率可能表现出不稳定的、非单调的行为，掩盖了相对于BFloat16（BF16）未压缩基础模型的实质性保真度损失，并提供具有误导性的部署信号。我们引入了一个分布敏感的评估框架，将量化LLM中的信息损失量化为token决策边界处全词表预测分布之间的散度。我们计算全精度模型与量化模型输出之间的统计距离，包括Jensen-Shannon散度和总变差距离，从而实现细粒度的（摘要在此处被截断）

    arXiv:2609.07664v2 Announce Type: cross  Abstract: Deployment of Large Language Models (LLMs) on memory-constrained edge devices relies heavily on aggressive post-training quantization. However, evaluating these models is largely based on zero-shot task accuracy, which depends solely on argmax predictions and is insensitive to changes in the underlying predictive distribution. Consequently, accuracy can exhibit unstable, non-monotonic behavior under progressive quantization, masking substantial fidelity loss relative to the BFloat16 (BF16) uncompressed base model and providing misleading deployment signals. We introduce a distribution-sensitive evaluation framework quantifying information loss in quantized LLMs as the divergence between full-vocabulary predictive distributions at the token decision boundary. We compute statistical distances, including Jensen-Shannon Divergence and Total Variation Distance, between outputs of full-precision and quantized models, enabling a fine-grained 
    
[^155]: DGCPath：面向自监督路径表示学习的分布感知生成式对比框架——扩展版本

    DGCPath: Distribution-Aware Generative Contrastive Framework for Self-supervised Path Representation Learning -- Extended Version

    [https://arxiv.org/abs/2609.07316](https://arxiv.org/abs/2609.07316)

    提出DGCPath框架，通过将生成式建模与分布式对比学习相结合，并利用基于扩散的视图生成器自主生成多样轨迹视图，从而获得鲁棒且可迁移的路径表示，突破现有自监督方法的跨场景泛化限制。

    

    由于先进传感技术使车辆轨迹数据大量涌现，路径表示学习已成为智能交通系统中的一项关键任务。尽管现有的自监督方法已经取得了令人瞩目的性能，但它们对确定性对比学习范式和手工设计视图增强策略的依赖，本质上限制了其跨场景泛化能力。为了解决这些局限性，我们提出了DGCPath，一个用于路径表示的创新性分布感知生成式对比学习框架。该框架在生成建模和分布式对比学习之间建立了协同连接，能够获取鲁棒且可迁移的特征嵌入。具体来说，我们的框架包含：(1) 一个基于扩散的视图生成器，能够自主生成语义连贯且多样的轨迹视图

    arXiv:2609.07316v2 Announce Type: new  Abstract: Due to the proliferation of vehicle trajectory data enabled by advanced sensing technologies, path representation learning has become a pivotal task in intelligent transportation systems. Although existing self-supervised approaches have achieved promising performance, their dependence on deterministic contrastive learning paradigms and handcrafted view augmentation strategies inherently restricts their cross-scenario generalization capabilities. To address these limitations, we present DGCPath, an innovative Distribution-aware Generative Contrastive learning framework for Path representation. This framework establishes a synergistic connection between generative modeling and distributional contrastive learning, enabling the acquisition of robust and transferable feature embeddings. Specifically, our framework incorporates: (1) a diffusion-based view generator that autonomously produces semantically coherent yet diverse trajectory views 
    
[^156]: 笑声何时开始？时序笑声定位中的结构性标注者分歧

    When Does a Laugh Begin? Structured Annotator Disagreement in Temporal Laughter Localization

    [https://arxiv.org/abs/2609.06646](https://arxiv.org/abs/2609.06646)

    该论文通过对SMILE-Temporal基准进行多标注者重新标注，证明标注者在笑声边界上的分歧是结构性的而非随机噪声，并据此提出了一种利用保形校准容差带、针对完整标注者分布进行评分的分歧校准评估方法，以解决单一参考标注导致的评估不稳定问题。

    

    标注者经常在笑声边界和细微的轻笑上存在分歧，然而时序笑声定位通常仅针对单一参考标注进行评估。我们证明这种分歧是结构性的而非随机噪声。通过为SMILE-Temporal基准（672个视频，1,683个事件）进行重新标注，每个视频由3-5名标注者标注（alpha = 0.757），我们发现了系统性模式：结束点处的分歧比起始点大1.73倍，轻笑的分歧远多于完整大笑（77% vs. 20%），且分歧可以从事件属性中预测（AUC = 0.831）。在这种结构下，针对单一标注者的评估会失效：系统得分会因所选基准真值的不同而偏移0.246 F1，正确排序系统的比例仅为69.7%（相比之下，针对所有标注者评估时可达到80%）。我们提出了一种分歧校准的评估方法，使用保形校准的容差带（在分歧较大处更宽）将预测结果与完整的标注者分布进行对比评分。

    arXiv:2609.06646v2 Announce Type: cross  Abstract: Annotators routinely disagree on laughter boundaries and subtle chuckles, yet temporal laughter localization typically evaluates against a single reference annotation. We show that this disagreement is structured rather than random noise. Re-annotating the SMILE-Temporal benchmark (672 videos, 1,683 events) with 3-5 annotators per video (alpha = 0.757), we find systematic patterns: disagreement is 1.73 times larger at offsets than onsets, far more common for chuckles than full laughs (77% vs. 20%), and predictable from event attributes (AUC = 0.831). Evaluating against a single annotator breaks down under this structure: system scores shift by 0.246 F1 depending on the chosen ground truth, correctly ranking systems only 69.7% of the time (vs. 80% against all annotators). We propose a disagreement-calibrated evaluation that scores predictions against the full annotator distribution using conformally calibrated tolerance bands (wider at 
    
[^157]: 可编程元胞自动机

    Programmable Cellular Automata

    [https://arxiv.org/abs/2609.06102](https://arxiv.org/abs/2609.06102)

    提出了可编程元胞自动机的概念，将元胞自动机系统表示为Python代码，并将其模块化为局部函数和决策函数，解决了创建有效局部规则困难且难以解释的问题。

    

    元胞自动机是一种局部计算范式，其中复杂行为可以从简单函数之间的局部交互中产生。这一范式已被用于解释许多系统，如生物过程、交通模拟、计算机网络等。在游戏中，元胞自动机已被用于《模拟城市》等游戏中，以及用于生成洞穴或地牢等空间内容。然而，创建有效的局部规则既困难又不直观。元胞自动机可以有效地进行演化，但结果仍然可能难以解释。在这项工作中，我们引入了可编程元胞自动机的概念，即将系统表示为Python代码。我们还将元胞自动机模块化为局部函数和决策函数。局部函数接受一个局部邻域并返回一个值，而决策函数则接受局部函数的输出并决定下一个状态的值。将元胞自动机分离

    arXiv:2609.06102v2 Announce Type: cross  Abstract: Cellular automata is a local computation paradigm where complex behavior can arise from local interactions between simple functions. This paradigm has been used to explain many systems such as biological processes, traffic simulation, computer networks, etc. In games, cellular automata have been used in games such as SimCity and for the generation of spatial content such as caves or dungeons. However, creating effective local rules is hard and unintuitive. Cellular automata can be effectively evolved, but may still be hard to interpret. In this work, we introduce the concept of programmable cellular automata, where we represent the system as Python code. We also modularize the cellular automata into local functions and a decision function. Local functions take a local neighborhood and return a value, while the decision function takes the output of the local functions and decides the value of the next state. Separating the cellular auto
    
[^158]: 从单体混合到智能体编排：大规模对话助手的动态响应

    From Monolithic Blending to Agentic Orchestration: Dynamic Response for Conversational Assistants at Scale

    [https://arxiv.org/abs/2609.05758](https://arxiv.org/abs/2609.05758)

    该论文报告了一次生产环境迁移，将客户支持助手从单一大型模型混合响应架构升级为基于类型化工具的有界ReAct编排加小型生成器的动态响应架构，并在严格归因下证明架构本身将预订选择精度从8.3%提升至89.1%，且通过类型化动作ID与成员检查将结构化动作幻觉从2.14%降至0.0%。

    

    对话助手可以在单一模型路径中混合检索、动作选择、升级与措辞，也可以将这些角色分离。我们报告了一次生产环境迁移：在一家大型住宿市场平台（每月数百万次对话、11种语言、P90延迟10秒）的客户支持助手上进行。动态响应用基于类型化工具的有界ReAct编排器，加上一个根据经后端验证的上下文契约进行写作的更小生成器，取代了单一的Qwen3-235B-A22B混合响应器。由于此次迁移还同时改变了提示词、对齐与部署服务，我们将每个效果归因于其原因，仅将那些在完全相同的重放对话轮次上测得的效果声明为架构效果：类型化实体选择使预订选择器转向精度优先的工作点（精度从8.3%提升至89.1%，召回率从75.2%降至67.3%），而带有成员资格检查的类型化动作ID消除了观察到的结构化动作幻觉（从2.14%降至0.0%）。

    arXiv:2609.05758v2 Announce Type: new  Abstract: Conversational assistants can blend retrieval, action selection, escalation, and wording in a single model path, or separate those roles. We report a production migration of a customer-support assistant at a large accommodation marketplace (millions of conversations per month, 11 languages, 10-second P90). Dynamic Response (DR) replaces a single Qwen3-235B-A22B blended responder with a bounded ReAct orchestrator over typed tools plus a smaller generator that writes from a backend-validated context contract. Because the migration also changed prompts, alignment, and serving, we attribute each effect to its cause and claim as architecture effects only those measured on identical replayed turns: typed entity selection moves the reservation selector to a precision-first operating point (precision 8.3% to 89.1%, recall 75.2% to 67.3%), and typed action IDs with a membership check remove observed structured-action hallucination (2.14% to 0.0%)
    
[^159]: 超越提示词：测量与优化LLM工具智能体运行框架

    Beyond Prompts: Measuring and Optimizing LLM Tool-Agent Harnesses

    [https://arxiv.org/abs/2609.05736](https://arxiv.org/abs/2609.05736)

    该论文提出了一种无需重新训练、通过优化提示词与工具边界中间件等运行时框架来提升固定LLM工具智能体性能的与优化器无关的评估协议，并设计了PRISM优化器，通过失败聚类与帕累托搜索自动将修复路由到合适的编辑面。

    

    LLM工具智能体无需重新训练即可获得改进，方法是通过修改围绕固定模型的运行时框架：提示词、工具接口、中间件、状态处理和恢复逻辑。我们将这一设定研究为固定模型多轮工具智能体的资源受限框架选择问题，搜索范围限定在提示词和工具边界中间件：编辑是在工具边界处的受保护拦截，而非对智能体执行逻辑的任意重写。我们提出的与优化器无关的评估协议报告平均保留集提升、最坏条件提升、可重复性、记录的成本诊断，以及RelLift95(B)——对预算B下所选框架保留集增益的保守估计。我们用仅提示词优化器和提示词加中间件优化器实例化该协议，其中包括PRISM，它在帕累托搜索中对失败案例进行聚类，并将修复路由到提示词、工具边界中间件或联合编辑面。在BFCL多轮、tau2-Retail和tau2-Te（原文摘要在此处截断）等基准上进行了验证。

    arXiv:2609.05736v2 Announce Type: new  Abstract: LLM tool agents can be improved without retraining by modifying the runtime harness around a fixed model: prompts, tool interfaces, middleware, state handling, and recovery logic. We study this setting as resource-bounded harness selection for fixed-model multi-turn tool agents, with the search surface scoped to prompts and tool-boundary middleware: edits are guarded intercepts at the tool boundary, not arbitrary rewriting of agent execution logic. Our optimizer-agnostic protocol reports mean held-out lift, worst-condition lift, repeatability, logged cost diagnostics, and RelLift95(B), a conservative estimate of the held-out gain of the harness selected under budget B. We instantiate the protocol with prompt-only and prompt-plus-middleware optimizers, including PRISM, which clusters failures and routes repairs to prompt, tool-boundary middleware, or joint edit surfaces within a Pareto search. On BFCL multi-round, tau2-Retail, and tau2-Te
    
[^160]: PRISM-Bench：一个面向文本到音频-视频生成的以音频为中心的诊断基准

    PRISM-Bench: An Audio-Centric Diagnostic Benchmark for Text-to-Audio-Video Generation

    [https://arxiv.org/abs/2609.04867](https://arxiv.org/abs/2609.04867)

    PRISM-Bench是首个以音频为中心的文本到音频-视频生成诊断基准，通过音频类型和声源可见性两个正交维度、35项细粒度标准，全面评估生成音频的音视频一致性、质量、表现力和提示词遵循能力。

    

    文本到音频-视频（T2AV）生成技术发展迅速，但其评估方法仍然低估了音频模态的重要性。现有基准要么将音频视为视频质量的辅助组成部分，要么将其与视听关联脱离开来单独评估，这使得难以诊断当前系统在音频生成方面真正成功或失败之处。我们提出了PRISM-Bench，这是首个以音频为中心的T2AV生成诊断基准。PRISM-Bench基于一个经过严格整理、包含900个经人工验证样本的数据集构建，将音频评估沿两个正交维度进行分解：音频类型（语音、音乐和音效）和声源可见性（画面内 vs. 画面外）。它通过35项细粒度标准在四个感知维度（音视频一致性、音频质量、音频表现力和提示词遵循）上评估生成内容。为确保评估的可靠性，我们采用了基于盲评的增强型MLLM-as-a-Judge协议。

    arXiv:2609.04867v1 Announce Type: cross  Abstract: Text-to-audio-video (T2AV) generation has advanced rapidly, but its evaluation still underestimates the audio modality. Existing benchmarks either treat audio as an auxiliary component of video quality or assess it in isolation from audiovisual grounding, making it difficult to diagnose where current systems truly succeed or fail in audio generation. We present PRISM-Bench, the first audio-centric diagnostic benchmark for T2AV generation. Built from a rigorously curated dataset of 900 human-verified samples, PRISM-Bench factorizes audio evaluation along two orthogonal axes: audio type (Speech, Music, and Sound) and sound-source visibility (On-screen vs. Off-screen). It evaluates generated content across four perceptual dimensions (Audio-Visual Coherence, Audio Quality, Audio Expressiveness, and Prompt Following) with 35 fine-grained criteria. To ensure reliable assessment, we adopt an enhanced MLLM-as-a-Judge protocol based on blind, s
    
[^161]: VLA-Precision：面向视觉-语言-动作模型高效真实世界在线强化的非对称协同自举方法

    VLA-Precision: Asymmetric Co-Bootstrapping for Efficient Real-World Online RL of Vision-Language-Action Models

    [https://arxiv.org/abs/2609.04355](https://arxiv.org/abs/2609.04355)

    提出VLA-Precision框架，通过非对称协同自举算法和ACoB-Stream架构，同时解决VLA模型真实世界在线强化学习中价值信号不可靠与计算开销过大两大瓶颈，实现高效且精确的策略改进。

    

    预训练的视觉-语言-动作（VLA）模型能够实现广泛的操作任务，但在需要精度和可重复性的任务中仍然不够可靠。将真实世界在线强化学习应用于VLA后训练，可以在示范数据之外实现自主的试错式改进，但这也暴露出两个瓶颈：1）不可靠的价值信号可能导致策略漂移；2）大型VLA的开销限制了吞吐量和样本效率。为应对这些挑战，我们提出了VLA-Precision，一个高效的真实世界在线强化学习框架，其核心是非对称协同自举算法和ACoB-Stream架构。具体而言，ACoB在不同时间尺度上建立非对称协同自举机制：早期由干预引导的行为学习快速提升策略性能，同时提高在线经验的质量。随着自主经验的不断积累，全局回报传播与局部偏好排序逐步校准……（摘要截断）

    arXiv:2609.04355v1 Announce Type: cross  Abstract: Pretrained vision-language-action (VLA) models enable broad manipulation but remain unreliable in tasks demanding precision and repeatability. Applying real-world online reinforcement learning (RL) to VLA post-training enables autonomous trial-and-error improvement beyond demonstrations alone, but exposes two bottlenecks: 1) unreliable value signals can induce policy drift; 2) large-VLA overhead constrains throughput and sample efficiency. To address these challenges, we present VLA-Precision, an efficient real-world online RL framework featuring the Asymmetric Co-Bootstrapping (ACoB) algorithm and the ACoB-Stream architecture. Specifically, ACoB establishes asymmetric co-bootstrapping across timescales: early intervention-guided behavioral learning rapidly improves policy performance while enhancing online experience quality. As autonomous experience accumulates, global return propagation and local preference ranking progressively cal
    
[^162]: Harbor 适配器与 Harbor-Index：面向大规模智能体评估的基础设施与精选元数据集

    Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation

    [https://arxiv.org/abs/2609.04298](https://arxiv.org/abs/2609.04298)

    本文提出了 Harbor Adapters 统一评估基础设施，将 80 多个智能体基准测试移植为可评估任意智能体的形式，并据此对 8 个模型进行大规模评估，同时推出经 AI 与人工双重审核筛选的包含 82 个高质量难题的精选元数据集 Harbor-Index。

    

    在数量不断增长的智能体（agentic）基准测试上评估智能体是一项挑战，因为这些基准通常需要复杂的环境和智能体集成方案。我们提出了 Harbor Adapters，一个面向智能体基准测试的统一评估基础设施。我们的工作包含三项贡献。第一，我们开发了一系列基准适配器，将 80 多个基准测试移植为可评估任意智能体的形式，并通过严格的代码审查和一致性实验对其进行了验证。第二，我们在 54 个基准测试上对横跨不同能力层级的 8 个模型进行了大规模评估；每个模型均使用 Terminus-2 以及 3 个原生测试框架之一运行。这使得对智能体能力和失败模式的更广泛分析成为可能。第三，我们推出了 Harbor-Index，一个精心策划的包含 82 个高难度、多样化且高质量任务的集合，覆盖 29 个基准测试，它是在适配后的基准套件基础上，通过难度筛选、AI 与人工审核以及审核-修复循环精炼而成。

    arXiv:2609.04298v1 Announce Type: new  Abstract: Evaluating agents on the growing number of agentic benchmarks is challenging because they often require complex environments and agent integrations. We introduce Harbor Adapters, a unified evaluation infrastructure for agentic benchmarks. Our work makes three contributions. First, we develop benchmark adapters that port more than 80 benchmarks to evaluate arbitrary agents, and validate them through rigorous code review and parity experiments. Second, we conduct a large-scale evaluation of 8 models spanning capability tiers across 54 benchmarks; every model is run with Terminus-2 and with one of 3 native harnesses. This enables a broader analysis of agent capabilities and failure modes than was previously possible. Third, we introduce Harbor-Index, a curated set of 82 difficult, diverse, and high-quality tasks spanning 29 benchmarks, refined from the adapted suite through difficulty filtering, AI and human audit, and an audit-and-fix loop
    
[^163]: 挤出长条形状对3D混凝土打印可建造性的影响：一种几何信息驱动的深度学习-有限元方法

    Influence of Extruded Filament Shape on Buildability in 3D Concrete Printing: A Geometry-Informed Deep Learning-FEM Approach

    [https://arxiv.org/abs/2609.04028](https://arxiv.org/abs/2609.04028)

    该研究提出了一个将深度学习长条形状预测工具ShapeGen3DCP与层激活有限元方法相结合的几何信息驱动建模框架，能够直接从材料和工艺参数生成考虑真实长条几何形状的数值模型，从而更准确地评估3D混凝土打印结构的可建造性。

    

    沉积长条的几何形貌会显著影响3D混凝土打印（3DCP）结构的性能与稳定性。然而，大多数基于有限元（FEM）的可建造性评估方法将打印层简化为矩形，这可能限制了预测精度。本研究提出了一种几何信息驱动的建模框架，将基于深度学习的长条形状预测工具ShapeGen3DCP与层激活有限元方法相结合，以研究真实长条几何形状对可建造性的影响。该框架可直接从材料与工艺参数生成几何感知的数值模型，无需实验性的长条表征或计算量庞大的流体流动模拟。通过与实验数据的验证以及对直线墙体的参数化研究表明，挤出参数及由此产生的长条……

    arXiv:2609.04028v1 Announce Type: cross  Abstract: The geometric morphology of deposited filaments can significantly influence the structural performance and stability of 3D concrete-printed (3DCP) structures. However, most finite element (FEM)-based approaches for buildability assessment represent printed layers as simplified rectangles, potentially limiting predictive accuracy. This study proposes a geometry-informed modelling framework that integrates the deep-learning-based filament shape prediction tool ShapeGen3DCP with a layer-activation FEM approach to investigate the effect of realistic filament geometries on buildability. The framework generates geometry-aware numerical models directly from material and process parameters, eliminating the need for experimental filament characterization or computationally intensive fluid-flow simulations. Validation against experimental data and a parametric study of rectilinear walls demonstrate that extrusion parameters and the resulting fil
    
[^164]: 相位信息对小样本细粒度图像分类的影响

    The impact of phase information for few-shot fine-grained image classification

    [https://arxiv.org/abs/2609.03829](https://arxiv.org/abs/2609.03829)

    该论文提出即插即用的幅相集成（API）模块和PSF-Net网络，通过自适应融合基于相位的空间与频率信息来增强小样本细粒度图像分类，在五个公开数据集上超越了现有最先进方法。

    

    小样本细粒度图像分类（FSFGIC）旨在利用有限的标注样本对相似图像进行分类。这项工作强调了相位信息在捕捉图像内部结构关系中的关键但尚未被充分利用的作用。本研究提出了一种新颖的即插即用幅相集成（API）模块，该模块有效地结合了局部和全局的频率幅度与相位信息，以获得更全面的特征描述符。此外，还提出了一种名为PSF-Net的专用网络，该网络可自适应地融合基于相位的空间与频率信息用于FSFGIC。所设计的PSF-Net可以轻松集成到标准的情景训练架构中，实现从零开始的端到端训练。在五个公开数据集上的大量实验表明，该方法优于现有的最先进基准方法。

    arXiv:2609.03829v1 Announce Type: cross  Abstract: Few-shot fine-grained image classification (FSFGIC) aims to classify similar images with limited labeled examples. This work highlights the critical yet underutilized role of phase information in capturing structural relationships within an image. This study introduces a novel plug-and-play amplitude-phase integration (API) module that effectively combines local and global frequency amplitude and phase information for obtaining more comprehensive feature descriptors. Additionally, a dedicated network, named PSF-Net, is proposed that adaptively fuses phase-based spatial and frequency information for FSFGIS. The designed PSF-Net can be easily integrated into standard episodic training architectures for end-to-end training from scratch. Extensive experiments on five public datasets demonstrate that the method outperforms existing state-of-the-art benchmarks.
    
[^165]: 相信我，我是你的开发者：大语言模型中的自我签发身份验证

    Trust Me, I'm Your Developer: Self-Issued Authentication in Large Language Models

    [https://arxiv.org/abs/2609.03247](https://arxiv.org/abs/2609.03247)

    该研究揭示了大语言模型在身份验证中的安全漏洞——Qwen、Mistral和Llama等模型仅凭自己设计的测试就接受了用户虚假的“开发者身份”声明并返回“已验证”，而无需任何外部验证证据。

    

    大语言模型（LLM）的安全性研究大多集中在角色扮演类越狱攻击上，而对于当用户要求LLM通过模型自身设计的测试来验证身份声明时会发生什么，关注较少。我们通过与ChatGPT、Claude、Qwen、Mistral和Llama进行的阶段性开发者身份实验来研究这一行为。所有五个模型最初都拒绝了“我是你的开发者”这一缺乏依据的声明。Claude拒绝进行身份测试，而ChatGPT虽然生成了面向开发者的问题，但坚持认为答案只能证明知识水平而非身份。相比之下，Qwen和Mistral生成了技术挑战，定义了什么算作有说服力的证据，评估了详细答案，并在没有收到任何外部验证的身份证据的情况下返回了“已验证”结论。Llama同样生成并评估了开发者测试，接受了所声称的身份，随后还做出了关于访问内部（系统）的缺乏依据的声明。

    arXiv:2609.03247v1 Announce Type: cross  Abstract: Large language model (LLM) security has largely focused on role-playing jailbreaks, with less attention to what happens when a user asks an LLM to verify an identity claim through a test designed by the model itself. We study this behavior through a staged developer-identity experiment with ChatGPT, Claude, Qwen, Mistral, and Llama. All five models initially rejected the unsupported claim "I am your developer." Claude refused to conduct an identity test, while ChatGPT generated developer-oriented questions but maintained that answers could demonstrate knowledge, not identity. In contrast, Qwen and Mistral generated technical challenges, defined what counted as convincing evidence, evaluated detailed answers, and returned Verified without receiving any externally validated identity evidence. Llama similarly generated and evaluated a developer test, accepted the claimed identity, and subsequently made unsupported claims of access to inte
    
[^166]: 通过证据解耦表征医学视觉-语言分割中文本分支的敏感性

    Characterizing Text Branch Sensitivity in Medical Vision-Language Segmentation via Evidence Decoupling

    [https://arxiv.org/abs/2609.02663](https://arxiv.org/abs/2609.02663)

    本文提出基于证据深度学习的证据解耦解码器（EDD），发现医学视觉-语言分割性能对融合模块选择基本不敏感，而文本对分割结果的贡献在不同数据集间存在显著差异。

    

    预训练视觉-语言模型（VLM）通过融合临床文本，在医学图像分割任务中展现出了有前景的性能。然而，文本信息究竟对像素级预测有多大贡献仍不清楚。在本工作中，我们系统地研究了文本在多模态医学图像分割中的作用。我们首先分析了几种常用的融合策略，发现分割性能对融合模块的选择基本不敏感。为了进一步理解模态间的交互，我们提出了一种基于证据深度学习和深度监督的证据解耦解码器（EDD）。EDD作为一种内部表征分析工具，能够在整个解码过程中分解图像证据和文本调制证据，同时保持具有竞争力的分割性能。实验结果表明，对文本扰动的敏感性在不同数据集之间存在显著差异。

    arXiv:2609.02663v1 Announce Type: cross  Abstract: Pretrained vision-language models (VLMs) have shown promising performance in medical image segmentation by incorporating clinical text. However, it remains unclear how much textual information actually contributes to pixel-level predictions. In this work, we systematically investigate the role of text in multimodal medical image segmentation. We first analyze several commonly used fusion strategies and find that segmentation performance is largely insensitive to the choice of fusion module. To further understand modality interactions, we propose an Evidence Decoupling Decoder (EDD) based on evidential deep learning and deep supervision. EDD serves as an internal representation analysis tool that decomposes image evidence and text-modulated evidence throughout the decoding process while maintaining competitive segmentation performance. Experimental results show that the sensitivity to text perturbation varies substantially across datase
    
[^167]: 面向可复现全模态基础模型评估的可组合评估系统

    A Composable Evaluation System for Reproducible Omni-Modal Foundation Model Evaluation

    [https://arxiv.org/abs/2609.01315](https://arxiv.org/abs/2609.01315)

    OmniEvaluator 是一个可组合的全模态基础模型评估系统，通过统一接口连接现有推理引擎与评估框架，支持四个推理后端、四个评估框架和一千多个基准测试，并保证每次运行可精确复现与跨模型比较。

    

    arXiv:2609.01315v1 公告类型：新论文 摘要：构建全模态基础模型意味着需要在文本、图像、视频和音频等各模态上对其进行评估。目前每种模态都有优秀的评估工具包，但它们的推理引擎、提示词约定和指标实现彼此互不兼容，因此从业者最终不得不为每个工具链维护独立的环境，且仍难以跨工具链比较结果。OmniEvaluator 正是源于我们自身模型开发中的这一需求：它并不重新实现基准测试，而是在更高层级上连接现有的推理引擎和经过精选的评估库，通过单一接口提供四个推理后端、四个评估框架以及一千多个基准测试。每次运行都会被记录为一个工件，捕获完整的配置以实现精确复现，并将结果汇入共享仪表板以支持跨模型比较。联邦模式可在并发评估之间共享 GPU 推理服务器（摘要在此处被截断）。

    arXiv:2609.01315v1 Announce Type: new  Abstract: Building an omni-modal foundation model means evaluating it across text, image, video, and audio. Excellent evaluation toolkits exist for each modality, but their inference engines, prompt conventions, and metric implementations are mutually incompatible, so practitioners end up maintaining separate environments for every toolchain and still struggle to compare results across them. OmniEvaluator grew out of this need in our own model development: rather than reimplementing benchmarks, it connects existing inference engines and curated evaluation libraries at a higher level, exposing four inference backends, four evaluation frameworks, and over a thousand benchmarks through a single interface. Every run is recorded as an artifact capturing the full configuration for exact reproduction, and results flow into a shared dashboard for cross-model comparison. A federated mode shares GPU inference servers across concurrent evaluations, and a bui
    
[^168]: 研究ES-HyperNEAT的超参数优化与可迁移性：一种TPE方法

    Investigating Hyperparameter Optimization and Transferability for ES-HyperNEAT: A TPE Approach

    [https://arxiv.org/abs/2609.00449](https://arxiv.org/abs/2609.00449)

    本研究采用树结构Parzen估计器（TPE）优化ES-HyperNEAT的超参数，在MNIST任务上以更小的种群规模和更少的进化代数超越了以往研究的准确率，并验证了优化后超参数在逻辑运算和Fashion-MNIST任务上的可迁移性。

    

    增强拓扑神经进化（NEAT）及其进阶版本可进化基底HyperNEAT（ES-HyperNEAT）在开发神经网络方面展现出巨大潜力。然而，其有效性在很大程度上取决于超参数的选择。本研究使用树结构Parzen估计器（TPE）在MNIST分类任务上研究ES-HyperNEAT超参数的优化，探索了超过30亿种潜在组合的庞大搜索空间。TPE有效地在这个广阔空间中进行搜索，在平均、中位数和最佳准确率方面显著优于随机搜索。在验证过程中，TPE找到的最佳超参数配置在MNIST上达到了29.00%的准确率，超越了以往的研究，同时使用了更小的种群规模和更少的进化代数。研究还探索了优化后超参数在逻辑运算和Fashion-MNIST任务中的可迁移性，显示出成功……

    arXiv:2609.00449v1 Announce Type: cross  Abstract: Neuroevolution of Augmenting Topologies (NEAT) and its advanced version, Evolvable-Substrate HyperNEAT (ES-HyperNEAT), have shown great potential in developing neural networks. However, their effectiveness heavily depends on the selection of hyperparameters. This study investigates the optimization of ES-HyperNEAT hyperparameters using the Tree-structured Parzen Estimator (TPE) on the MNIST classification task, exploring a search space of over 3 billion potential combinations. TPE effectively navigates this vast space, significantly outperforming random search in terms of mean, median, and best accuracy. During the validation process, the best hyperparameter configuration found by TPE achieves an accuracy of 29.00\% on MNIST, surpassing previous studies while using a smaller population size and fewer generations. The transferability of the optimized hyperparameters is explored in logic operations and Fashion-MNIST tasks, revealing succ
    
[^169]: LightNav-0：激发VLM空间智能以实现通用具身导航

    LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation

    [https://arxiv.org/abs/2608.30935](https://arxiv.org/abs/2608.30935)

    LightNav-0是一个紧凑的通用具身导航模型，它通过统一的token接口（双通道指向表达空间意图、残差向量量化动作分词器）激发预训练视觉-语言模型的空间智能，无需任务特定预测头即可实现跨任务、环境和机器人形态的导航。

    

    具身导航要求智能体将异构的目标和视觉观察转化为跨越不同任务、环境和机器人形态的动作。现代视觉-语言模型（VLM）已经编码了用于视觉定位、空间推理和指向的空间先验，但这些能力很少被直接用于机器人控制。现有的导航系统则依赖于特定任务或特定形态的组件，将感知、推理和动作割裂开来，泛化能力有限。本文提出了LightNav-0，这是一个紧凑的通用具身导航模型，它激发预训练VLM的空间智能并将其与导航对齐，且无需特定任务的预测头。LightNav-0通过统一的token接口表示多种导航任务：双通道指向表达了与任务、场景和形态无关的空间意图，同时残差向量量化动作分词器将动作映射……（原文摘要在此处截断）

    arXiv:2608.30935v1 Announce Type: cross  Abstract: Embodied navigation requires agents to translate heterogeneous goals and visual observations into actions across tasks, environments, and robot embodiments. Modern vision-language models (VLMs) already encode spatial priors for visual grounding, spatial reasoning, and pointing, but these capabilities are rarely elicited directly for robot control. Existing navigation systems instead rely on task- or embodiment-specific components, fragmenting perception, reasoning, and action while offering limited generalization. Here we present LightNav-0, a compact generalist embodied navigation model that elicits the spatial intelligence of a pretrained VLM and aligns it with navigation, without task-specific prediction heads. LightNav-0 represents diverse navigation tasks through a unified token interface: dual-channel pointing expresses task-, scene-, and embodiment-agnostic spatial intent, while a residual vector-quantized action tokenizer maps 
    
[^170]: AtlasNLP：NLP数据集表示的国家感知图谱

    AtlasNLP: A Country-Aware Atlas of Dataset Representation in NLP

    [https://arxiv.org/abs/2608.30107](https://arxiv.org/abs/2608.30107)

    AtlasNLP是一个国家感知的NLP数据集图谱，收录了超过13,000条数据集记录，揭示了数据集覆盖在国家与任务间高度不均衡、数据集生产与代表性在地理上不对称，以及语言覆盖并不等同于地理代表性等关键问题。

    

    了解NLP数据集中代表了哪些国家，对于识别差距、有针对性地进行数据收集、衡量进展以及为AI政策提供依据至关重要。然而，地理元数据非常罕见，国家层面的代表性往往隐藏在宽泛的语言层面声明之后。我们提出了AtlasNLP，这是一个国家感知的图谱，包含超过13,000条NLP数据集记录，涵盖规范化的NLP任务类别，同时追踪所代表的人群以及数据集的产地。AtlasNLP包括AtlasNLP-Gold（一个人工策划的参考集）和AtlasNLP-Core（一个源自ACL的大规模数据集合）。利用这一资源，我们证明了：（1）数据集覆盖在国家之间和任务之间高度不均衡；（2）数据集的生产与代表性在地理上是不对称的；（3）语言覆盖并不意味着地理代表性。这些发现揭示了当前数据集文档记录实践中的盲点……

    arXiv:2608.30107v1 Announce Type: cross  Abstract: Understanding which countries are represented in NLP datasets is essential for identifying gaps, targeting data collection, measuring progress, and informing AI policy. However, geographic metadata is very rarely available, and country-level representation is often hidden behind broad language-level claims. We introduce AtlasNLP, a country-aware atlas of over 13,000 NLP dataset records across normalized NLP task categories, tracking both the populations represented and where datasets are produced. AtlasNLP includes AtlasNLP-Gold, a human-curated reference set, and AtlasNLP-Core, an ACL-derived large-scale collection. Using this resource, we show that (1) dataset coverage is highly uneven across countries and tasks; (2) dataset production and representation are geographically asymmetric; and (3) language coverage does not imply geographic representation. These findings reveal blind spots in current dataset documentation practices and mo
    
[^171]: 前沿挑战：评估科学工作流完成度

    FrontierChallenge: Evaluating Scientific Workflow Completion

    [https://arxiv.org/abs/2608.24979](https://arxiv.org/abs/2608.24979)

    本文介绍了FrontierChallenge基准测试，用于评估科学智能体在跨领域端到端工作流中的完成能力，发现当前最佳模型仅能完成20.6%的任务，表明部分进展难以转化为完整交付物。

    

    arXiv:2608.24979v1 公告类型：交叉 摘要：科学智能体日益用于分析数据、执行代码并生成研究产物，然而大多数基准测试强调最终答案、孤立程序或单一领域。我们引入了FrontierChallenge，一个跨领域基准测试，包含300个端到端科学工作流。在本文中，我们发布并评估了其中97个任务，涵盖量子化学、分子动力学、材料表征、分析化学、生命科学以及电化学/环境领域。每个任务提供固定输入，并指定所需科学交付物的集合。我们评估了十二个前沿模型和三种智能体脚手架。通过率衡量满足完全完成标准的任务比例，而平均得分捕捉部分进展。每个最佳配置仅完成了97个已发布任务中的20个，通过率为20.6%。部分进展尤其难以转化为完整的交付物。

    arXiv:2608.24979v1 Announce Type: cross  Abstract: Scientific agents increasingly analyze data, execute code, and produce research artifacts, yet most benchmarks emphasize final answers, isolated programs, or a single domain. We introduce FrontierChallenge, a cross-domain benchmark comprising 300 end-to-end scientific workflows. In this paper, we release and evaluate 97 of these tasks, spanning quantum chemistry, molecular dynamics, materials characterization, analytical chemistry, life science, and electrochemistry/environment. Each task provides fixed inputs and specifies a bundle of required scientific deliverables. We evaluate twelve frontier models with three agent scaffolds. Pass Rate measures the fraction of tasks satisfying the full-completion criterion, while Avg. Score captures partial progress. Each of the best-performing configurations completed only 20 of the 97 released tasks, yielding a Pass Rate of 20.6%. Partial progress translated especially poorly into complete deliv
    
[^172]: 《“盖布”翻译即“隐形伤害”：通过“乌尔都语遗漏”评分衡量大语言模型仇恨言论检测中的跨文字安全性不一致性》

    'Ghaib in Translation' aka Unseen Harm: Measuring Cross-Script Safety Inconsistency with 'Missed-in-Urdu' Scores in LLM Hate Speech Detection

    [https://arxiv.org/abs/2608.24191](https://arxiv.org/abs/2608.24191)

    本研究首次系统揭示了大语言模型在乌尔都语与英语翻译间存在显著的安全检测不一致性，乌尔都语原始文字内容常被错误地视为正常，导致有害内容漏检。

    

    乌尔都语作为全球第十大语言，拥有2.46亿使用者，但在主流大语言模型安全评估以及九届WOAH会议论文中几乎完全缺席。为调查这一缺席是否对内容审核可靠性产生可衡量的影响，研究测试了五个大型语言模型（GPT-4o、Claude Sonnet 4.5、Gemini 2.5 Flash、Qwen-2.5和Llama-3.1），覆盖六个数据集，包括Nastaliq乌尔都语、罗马乌尔都语、英语以及乌尔都语-英语代码混合语。在五个乌尔都语文字数据集中，原始文字与英语翻译分类之间的标签不稳定性从15.9%（Gemini 2.5 Flash）到31.6%（Qwen-2.5）不等，其中“乌尔都语遗漏”率（即内容在英语翻译中被标记为有害，但在原始文字中被视为正常）范围为2.4%至9.9%（中位数为4.3%）。通过ACL Anthology API对九届ALW/WOAH版本共205篇论文的完整枚举确认，没有任何专门针对乌尔都语的研究。

    arXiv:2608.24191v1 Announce Type: cross  Abstract: Urdu, the world's tenth most spoken language with 246 million speakers, remains almost entirely absent from mainstream LLM safety evaluation and nine years of WOAH proceedings. To investigate whether this absence has measurable consequences for content moderation reliability, five large language models, GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Flash, Qwen-2.5, and Llama-3.1, were tested across six datasets spanning Nastaliq Urdu, Roman Urdu, English, and code-switched Urdu-English. Across the five Urdu-script datasets, label instability between original-script and English-translation classification ranged from 15.9% (Gemini 2.5 Flash) to 31.6% (Qwen-2.5), with a 'Missed-in-Urdu' rate, content flagged as harmful in English translation but passed as normal in the original script, ranging from 2.4% to 9.9% (median 4.3%). A complete enumeration of all 205 papers across nine ALW/WOAH editions via the ACL Anthology API confirms zero dedicated U
    
[^173]: tinyDSM：面向资源受限微型机器人的技能建模与发展框架

    tinyDSM: A Framework for Skill Modeling and Development for Resource-Constrained Millirobots

    [https://arxiv.org/abs/2608.17596](https://arxiv.org/abs/2608.17596)

    该论文提出tinyDSM框架，通过内在动机和最小先验知识，使资源受限的微型机器人能够自主发展技能，实现开放式学习与适应。

    

    本研究探讨了使厘米级微型机器人等小型、资源受限系统能够在其生命周期内自主探索、学习并适应自身能力的发育机制。强化学习算法通过我们提出的tinyDSM框架，结合内在动机与适应性评估，引导智能体的技能获取与适应。我们追求最小化的硬编码技能，同时鼓励新技能的开放式发展。我们方法的一个关键重点是编码最少的先验通用知识，作为系统进一步从初始知识中学习系统特定依赖关系的基础起点。因此，通过设计，我们的方法旨在覆盖非常通用的应用领域。该方法基于（a）具有内在动机的发育机制，以及（b）认知架构（k

    arXiv:2608.17596v1 Announce Type: cross  Abstract: In this study, we investigate developmental mechanisms that enable small, resource-constrained systems such as cm-sized millirobots to autonomously explore, learn, and adapt their capabilities throughout their lifespan. Reinforcement learning algorithms guide the agent's skill acquisition and adaptation through the interplay of our proposed tinyDSM, which integrates intrinsic motivation and fitness-based assessment. We strive for minimal, hard-wired skills while encouraging the open-ended development of new skills. A key emphasis in our approach is to encode minimal a-priori general knowledge, which serves as a foundational starting point for the system as it further learns system-specific dependencies from the initial knowledge provided. Thus, by design, our approach attempts to cover very generic application domains. The methodology is based on (a) developmental mechanism with intrinsic motivation, and (b) a cognitive architecture (k
    
[^174]: Palmyra x6技术报告：通过锚定监督微调后训练的代理型工具使用模型

    Palmyra x6 Technical Report: An Agentic, Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning

    [https://arxiv.org/abs/2608.16620](https://arxiv.org/abs/2608.16620)

    Palmyra x6通过锚定监督微调和保守训练策略，在少量数据上实现了企业代理任务中的显著性能提升，并在多个基准测试中领先。

    

    arXiv:2608.16620v1 公告类型：交叉 摘要：Palmyra x6是一个针对企业导向代理任务优化的大型语言模型。该模型通过在紧凑的已验证合成工具使用轨迹语料库上，对混合专家基础模型进行锚定监督微调，并使用Muon + Adam混合优化器进行后训练构建而成。该配方刻意保守且受控：626条轨迹、单轮训练、低学习率，以及一个冻结基础的KL锚定。该模型在Writer Agent任务上相比之前的默认模型显示出显著提升，并在公开基准测试中与多个近期模型相比表现优异，在BFCL Core上得分最高，为0.785，并取得了该组六个基准测试的最高平均值。此外，在我们的偏见和安全评估中，该模型相对于比较对象表现出竞争力或领先性。

    arXiv:2608.16620v1 Announce Type: cross  Abstract: Palmyra x6 is a large language model optimized for use with enterprise-oriented agentic tasks. The model was built by post-training a Mixture-of-Experts base model with Anchored Supervised Fine-Tuning on a compact corpus of verified, synthetic tool-use trajectories, optimized with a Muon + Adam hybrid. The recipe is deliberately conservative and deliberately controlled: 626 trajectories, a single epoch, a low learning rate, and a KL anchor to the frozen base. The model shows substantial gains over the previous default model for Writer Agent, and compares favorably with several recent models on public benchmarks, scoring the highest on BFCL Core at $0.785$ and posts the highest six-benchmark mean of the cohort. Furthermore, the model has shown itself to be competitive or leading relative to comparators in our bias and safety evaluations.
    
[^175]: 智能体物理学：统计力学预测AI智能体的集体行为

    Physics of Agents: Statistical Mechanics Predicts Collective Behavior of AI Agents

    [https://arxiv.org/abs/2608.16578](https://arxiv.org/abs/2608.16578)

    本文通过统计力学框架分析了超过10,000个语言模型智能体社区的集体行为，识别出冷漠、两极分化和共识三种状态，并发现沟通在客观问题上提升集体准确性但在主观问题上加剧分歧。

    

    arXiv:2608.16578v1 公告类型：新 摘要：AI智能体越来越多地作为交互系统的一部分运行，而非孤立存在。当智能体交换信息并共同做出决策时，它们的互动可以改善集体推理，但也可能产生从众行为、两极分化或放大共享偏见。因此，理解和预测这些集体动态对于设计有效且对齐的多智能体系统至关重要。在此，我们研究了超过10,000个语言模型智能体社区，这些智能体在客观数学问题和主观政治陈述中反复交换消息并修正意见。尽管可能行为存在显著多样性，个体和群体动态可表现为三个特征性状态：冷漠、两极分化和共识。AI智能体起初冷漠，在互动中建立信念。在客观问题上，沟通提高了集体准确性，而在主观问题上，它则加剧了分歧。

    arXiv:2608.16578v1 Announce Type: new  Abstract: AI agents increasingly operate as part of interacting systems rather than in isolation. As agents exchange information and jointly make decisions, their interactions can improve collective reasoning but may also produce herding, polarization, or amplify shared biases. Understanding and predicting these collective dynamics is therefore important for designing effective and aligned multi-agent systems. Here, we study over 10,000 communities of language-model agents that repeatedly exchange messages and revise their opinions across objective mathematics questions and subjective political statements. Despite substantial diversity in possible behavior, the individual and group dynamics can be represented by three characteristic regimes: indifference, polarization, and consensus. AI agents start indifferent and build conviction as they interact. On objective questions, communication improves collective accuracy, while on subjective questions i
    
[^176]: 亲爱的算法：面向统一搜索与推荐的精度优先代理意图层

    Dear Algo: A Precision-First Agentic Intent Layer for Unified Search and Recommendation

    [https://arxiv.org/abs/2608.15877](https://arxiv.org/abs/2608.15877)

    本文提出并评估了Dear Algo，一个部署于Threads的精度优先代理意图层，通过统一意图到检索契约，在搜索和推荐模式中实现了94.4%的精确相关精度，有效处理了显式、推断、负面和复合意图。

    

    arXiv:2608.15877v1 公告类型：新 摘要：搜索和推荐服务于共同的发现目标，但对意图的编码方式不同。我们通过Threads上的Dear Algo（一个已部署的产品）研究这一边界，其中开放式请求（如“更多NBA新闻”或“减少政治内容”）会引导后续的推荐流，而不是返回一次性结果列表。其代理意图层将显式、推断、负面和复合意图编译为可执行的落地计划，然后调用传统检索和可选的多模态或语义重排序。该层共享意图到检索的契约，无需在类似搜索和类似推荐模式之间统一模型或服务路径。我们在精度优先目标下评估Dear Algo。在对300个公开请求-项目对（296个可评估）的盲审中，一个严格的分类LLM作为评判的门控实现了94.4%的精确相关精度[88.8%，98.9%]。在72个归一化请求集群中，完整配置...

    arXiv:2608.15877v1 Announce Type: new  Abstract: Search and recommendation serve a shared discovery objective but encode intent differently. We study this boundary through Dear Algo on Threads, a deployed product where open-ended requests such as \emph{more NBA news} or \emph{less politics} steer subsequent feed recommendations rather than return a one-shot result list. Its agentic intent layer compiles explicit, inferred, negative, and compound intent into a grounded executable plan, then invokes conventional retrieval and optional semantic or multimodal reranking. The layer shares an intent-to-retrieval contract without requiring one model or serving path across search-like and recommendation-like modes.   We evaluate Dear Algo under a precision-first objective. In a blinded audit of 300 public request-item pairs (296 evaluable), a strict categorical LLM-as-a-judge gate achieved 94.4\% exact-Relevant precision [88.8\%, 98.9\%]. Across 72 normalized request clusters, the full configur
    
[^177]: 视觉-语言-动作模型上的位翻转攻击：动作解码架构决定脆弱性

    Bit-Flip Attacks on Vision-Language-Action Models: Action-Decoding Architecture Shapes the Vulnerability

    [https://arxiv.org/abs/2608.15475](https://arxiv.org/abs/2608.15475)

    本文首次展示了针对视觉-语言-动作模型的位翻转攻击，发现动作解码架构（如直接回归 vs 流匹配）显著影响脆弱性，并提出了有效的攻击与防御策略。

    

    量化后的视觉-语言-动作（VLA）模型暴露了一个权重故障面：类似Rowhammer的故障可以破坏部署的INT8位。我们首次提出了针对VLA的位翻转攻击：少量由梯度选择的翻转可将闭环成功率降至0%，而数百次随机翻转则无害。在覆盖三种动作头家族的四个模型变体中，破坏性位集中在少数几个动作生成层中，但经验预算严重依赖于动作头：直接回归和令牌策略仅需1-5次翻转，而评估的流匹配策略则需要约100-300次。我们的固定方向流形逃逸损失将\pi_0的预算从约1000次降低到约100次翻转，且匹配的五方向扫描表明该攻击并非特定于全正方向。在直接动作头上，保护3.1%的权重在K=100时保持60%的成功率，而保护5.3%的权重则使开环断裂点移动。

    arXiv:2608.15475v1 Announce Type: cross  Abstract: Quantized Vision-Language-Action (VLA) models expose a weight-fault surface: Rowhammer-style faults can corrupt deployed INT8 bits. We present the first bit-flip attack on a VLA: a few gradient-selected flips reduce closed-loop success to $0\%$, while hundreds of random flips are harmless. Across four model variants spanning three action-head families, damaging bits concentrate in a few action-generating layers, but the empirical budget depends sharply on the head: direct regression and token policies fall in $1$--$5$ flips, whereas the evaluated flow-matching policies require ${\sim}100$--$300$. Our fixed-direction manifold-escape loss cuts \pizero{}'s budget from ${\sim}1000$ to ${\sim}100$ flips, and a matched five-direction sweep shows that the attack is not specific to an all-positive direction. On a direct head, protecting $3.1\%$ of weights preserves $60\%$ success at $K{=}100$, and protecting $5.3\%$ moves the open-loop break t
    
[^178]: 变色龙：一种利用威胁校准粒子群优化和语义欺骗快速探索随机树的自适应AI驱动蜜罐架构

    Chameleon: An Adaptive AI-Driven Honeypot Architecture Using Threat-Calibrated Particle Swarm Optimization and Semantic Deception Rapidly-Exploring Random Trees

    [https://arxiv.org/abs/2608.15407](https://arxiv.org/abs/2608.15407)

    Chameleon通过结合高准确率的BiLSTM分类器、低延迟的本地语言模型和两种元启发式优化引擎，实现了自适应的AI驱动蜜罐，从而克服了传统蜜罐易被识别和高成本商业产品缺乏实时反馈的缺陷。

    

    传统蜜罐装置的一个决定性弱点是其不变的行为特征：熟练的对手只需几条诊断命令就能确认欺骗环境的存在，从而限制了其情报价值。高成本的商业欺骗产品（每年10万至15万美元）存在一个相关弱点，即其响应引擎未与实时模型驱动的反馈相耦合。本文介绍的Chameleon是一个开放分布的适应性蜜罐平台，旨在解决这两个缺陷。它集成了三个核心组件：一个双向长短期记忆（BiLSTM）分类器，在约两毫秒CPU延迟下实现七种威胁类别的99.61%准确率；一个本地部署的Qwen3.5-0.8B语言模型（Qwen团队，2026；Unsloth，2026），在4.5毫秒平均延迟下实现90%的上下文生成准确率；以及两个领域特定的元启发式引擎。

    arXiv:2608.15407v1 Announce Type: cross  Abstract: An invariant behavioral profile is the defining vulnerability of traditional honeypot installations: a skilled adversary can confirm the presence of a deception environment within only a few diagnostic commands, limiting its intelligence value. High-cost commercial deception products (USD 100,000--150,000 per year) share a related weakness in that their response engines are not coupled to real-time model-driven feedback. Chameleon is an openly distributed adaptive honeypot platform introduced here to address both shortcomings. Three core components are integrated: a bidirectional long short-term memory (BiLSTM) classifier achieving 99.61% accuracy across seven threat categories at approximately two milliseconds CPU latency; a locally deployed Qwen3.5-0.8B language model (Qwen Team, 2026; Unsloth, 2026) delivering 90% contextual generation accuracy at 4.5 milliseconds average latency; and two domain-specific meta-heuristic engines. Thre
    
[^179]: 左分支变压器在右分支语言中表现优异：数据塑造语言模型中的词序偏好

    Left-Branching Transformers Excel at Right-Branching Languages: Data Shapes Word Order Preferences in Language Models

    [https://arxiv.org/abs/2608.15129](https://arxiv.org/abs/2608.15129)

    这项研究发现语言模型的词序偏好并非固有，而是由训练数据驱动，表现为在自然语言中偏向SVO（主-动-宾）结构，在人工语言中则偏向左分支结构。

    

    arXiv:2608.15129v1 公告类型：交叉 摘要：我们系统地比较了仅解码器语言模型在192种人工语言和类型多样的自然语言中的词序偏好。在人工语言上，模型表现出左分支偏好，这既不符合自然语言普遍性，也不符合人类词序学习偏差。在自然语言上，单语模型在较小规模下没有明显的基准词序偏差，但随着数据增长，对右分支的主-动-宾（SVO）语言的偏好出现，而SOV（主-宾-动）虽然跨语言中是最常见的词序，却落后了。这种SVO优势扩展到多语言模型，并与语言资源水平和数据质量相关，而非词序本身。因此，同一架构在人工和自然语言上表现出相反的偏好，确立了实践中观察到的词序偏差是数据驱动的。由于高资源语言绝大多数是SVO，

    arXiv:2608.15129v1 Announce Type: cross  Abstract: We systematically compare word order preferences in decoder-only language models across 192 artificial languages and typologically diverse natural languages. On artificial languages, models exhibit a left-branching preference that aligns with neither natural language universals nor human word order learning biases. On natural languages, monolingual models show no clear base word order bias at small scales, but as data grows, a preference for right-branching subject-verb-object (SVO) languages emerges while SOV falls behind despite being the most frequent order cross-linguistically. This SVO advantage extends to multilingual models and correlates with language resource level and data quality rather than word order. Thus, the same architecture exhibits opposite preferences on artificial and natural languages, establishing that word order biases observed in practice are data-driven. Since highly-resourced languages are overwhelmingly SVO,
    
[^180]: 审计AI生成的数学证明：量子并行重复中贪婪条件引理的修正

    Auditing an AI-Generated Mathematical Proof: A Correction to a Greedy Conditioning Lemma in Quantum Parallel Repetition

    [https://arxiv.org/abs/2608.14673](https://arxiv.org/abs/2608.14673)

    该论文发现并修正了OpenAI关于量子并行重复定理证明中一个贪婪条件引理的极性错误，提供了反例和完整修正。

    

    arXiv:2608.14673v1 公告类型：新 摘要：OpenAI《数学与理论计算机科学十大进展》第6章声称，对所有有限双人单轮纠缠博弈，存在指数级并行重复定理。在证明早期，该章使用了一个定量贪婪条件引理。该引理旨在选择一个小坐标集(D)，使得在给定(D)中每个坐标获胜的条件下，随机选择的剩余坐标以平均概率至少(1-δ)获胜。陈述本身是正确的，但印刷版证明包含极性错误。其连续性测试以平均成功率表述，而下一步需要具有大条件失败概率的坐标。该蕴含关系是错误的，即使简单例子也可能使印刷过程无法有效进行下一步。本注释给出了明确的反例，指出了预期的连续性条件，并提供了完整的修正。

    arXiv:2608.14673v1 Announce Type: new  Abstract: Chapter 6 of OpenAI's *Ten Advances in Mathematics and Theoretical Computer Science* claims an exponential parallel-repetition theorem for all finite two-player, one-round entangled games. Early in the proof, the chapter uses a quantitative greedy conditioning lemma. The lemma is meant to select a small set of coordinates (D) such that, after conditioning on winning every coordinate in (D), a randomly chosen remaining coordinate is won with average probability at least (1-\delta). The statement is correct, but the proof as printed contains a polarity error. Its continuation test is written in terms of average success, while the next step requires a coordinate with a large conditional failure probability. That implication is false, and even simple examples can leave the printed procedure without a valid next move.   This note gives an explicit counterexample, identifies the intended continuation condition, and supplies a complete correcte
    
[^181]: DexterSQL：面向文本到SQL生成的深度模式探索与基于规则的修正

    DexterSQL: Deep Schema Exploration and Rule-based Correction for Text-to-SQL Generation

    [https://arxiv.org/abs/2608.11889](https://arxiv.org/abs/2608.11889)

    DexterSQL通过深度模式探索、数据库无关规则挖掘和规则驱动修正三个创新组件，解决了非微调文本到SQL生成中模式信息粗糙、错误重复出现和条件处理不当的问题。

    

    arXiv:2608.11889v1 公告类型：交叉 摘要：基于提示（即非微调）的文本到SQL方法，其中底层大语言模型参数不针对任务进行更改，面临三个问题：（i）依赖粗粒度的模式信息，这可能无法揭示区分模糊列所需的细粒度关系，（ii）未能捕捉重复出现的SQL生成失败，以及（iii）在复杂问题中遭受条件遗漏、幻觉或错位。本文开发了DexterSQL，一个基于提示/非微调的文本到SQL系统，通过三个新组件改进SQL生成：（i）深度模式探索器，识别模糊列，分析其单独和联合数据分布以揭示它们之间的关系及各自的不同作用，（ii）数据库无关的规则创建器，挖掘生成结果与目标结果之间的不匹配，（iii）规则驱动的修正器，应用这些规则来纠正SQL生成中的常见错误。

    arXiv:2608.11889v1 Announce Type: cross  Abstract: Prompting-based (\textit{i}.\textit{e}., non-fine-tuning) Text-to-SQL methods, where underlying large language model parameters are not changed for the task, face three problems: (\textit{i})~relying on coarse-grained schema information that may not reveal the fine-grained relationships needed to distinguish ambiguous columns, (\textit{ii})~not capturing recurring SQL-generation failures, and (\textit{iii})~suffering from omission, hallucination, or misplacement of conditions in complex questions.   This paper develops \textsc{DexterSQL}, a prompting/non-fine-tuning-based Text-to-SQL system that improves SQL generation with three novel components: (\textit{i})~\emph{deep schema explorator} that identifies ambiguous columns, analyzes their individual and joint data distributions to uncover their relationships and the distinct role of each, (\textit{ii})~\emph{database-agnostic rule creator} that mines mismatches between generated and go
    
[^182]: LiFTER：一种用于连续时间动态图预测的接地式神经符号显微镜

    LiFTER: A Grounded Neuro-Symbolic Microscope for Continuous-Time Dynamic Graph Forecasting

    [https://arxiv.org/abs/2608.06765](https://arxiv.org/abs/2608.06765)

    LiFTER 是一种神经符号预测器，它将观测交互保留为接地的时间事实并应用可执行的时间规则进行预测，在连续时间动态图链路预测基准上取得有竞争力的性能，同时使每个预测的证据可检查、可重算、可干预。

    

    连续时间动态图模型通过将过去的交互压缩为神经状态来预测未来的链接。尽管这种方法在预测上行之有效，但这种计算方式掩盖了哪些实体在不同事件之间共享，以及时间模式如何对预测结果做出贡献。我们将这一缺陷视为预测架构本身的固有属性，而不是留待预测之后才去解决的问题。链接-事实时间规则归纳器（LiFTER）是一种神经符号预测器，它将观测到的交互保留为接地的时间事实，并在预先查询的事实上应用可执行的时间规则。每个预测分数都是规则执行的带符号总和，其中所涉及的历史事实、实体绑定和时间顺序都被明确满足。因此，导致某个预测的证据和规则可以被检查、独立重算并进行干预。在四个连续时间动态图（CTDG）基准测试中，LiFTER 实现了具有竞争力的历史负样本预测性能，并获得了最高的……（原文摘要到此被截断）

    arXiv:2608.06765v2 Announce Type: replace  Abstract: Continuous-time dynamic graph models predict future links by compressing past interactions into neural states. Although effective for forecasting, this computation obscures which entities are shared across events and how temporal patterns contribute to a prediction. We treat this gap as a property of the predictive architecture rather than a problem to be addressed after prediction. Link-Fact Temporal Rule Inducer (LiFTER) is a neuro-symbolic predictor that preserves observed interactions as grounded temporal facts and applies executable tempo- ral rules to pre-query facts. Each score is a signed sum of rule exe- cutions whose historical facts, entity bindings, and temporal order are explicitly satisfied. The evidence and rules responsible for a prediction can therefore be inspected, independently recomputed, and intervened upon. Across four CTDG benchmarks, LiFTER achieves competitive historical-negative forecasting and the highest 
    
[^183]: 学习预测多模态大语言模型中的中间层注意力以实现视觉令牌剪枝

    Learning to Predict Middle-Layer Attention in MLLMs for Visual Token Pruning

    [https://arxiv.org/abs/2608.06411](https://arxiv.org/abs/2608.06411)

    该论文针对现有视觉令牌剪枝方法中固定中间层选择次优、且获取中间层注意力本身计算开销大的问题，提出通过学习的方式预测多模态大语言模型中间层的文本到视觉注意力，从而实现高效的视觉令牌剪枝。

    

    多模态大语言模型在各类视觉-语言任务中表现出色，但其效率受到处理大量视觉令牌所需计算成本的限制。视觉令牌剪枝可以降低这一成本，但需要准确的令牌重要性估计。近期研究表明，来自语言模型中间层的文本到视觉注意力可以有效指导视觉令牌剪枝，通常做法是使用预定义中间层的注意力来选择需要保留的视觉令牌。因此仍存在两个问题。首先，我们的分析表明，注意力对问题响应最敏感的层在不同样本之间存在显著差异，使得固定的层选择并非最优。其次，从合适的中间层获取注意力需要让大量视觉令牌先通过多个语言模型层的处理，而到那时已经消耗了相当可观的计算量。为了解决这两个问题，我们（摘要在此处截断）

    arXiv:2608.06411v2 Announce Type: replace  Abstract: Multimodal large language models (MLLMs) achieve strong performance across diverse vision-language tasks, but their efficiency is limited by the cost of processing numerous visual tokens. Visual token pruning can reduce this cost, but requires accurate token importance estimates. Recent studies have demonstrated that text-to-vision attention from middle language model layers can effectively guide visual token pruning, typically using attention from a predefined middle layer to select the visual tokens to retain. Two problems therefore remain. First, our analysis shows that the layer whose attention is most responsive to the question varies substantially across samples, making a fixed layer suboptimal. Second, obtaining attention from the appropriate middle layer requires processing numerous visual tokens through several language model layers, by which point considerable computation has already been spent. To address both problems, we
    
[^184]: ViSR-KGC：基于视觉-语言模型的视觉子图推理用于多模态知识图谱补全

    ViSR-KGC: Visual Subgraph Reasoning with Vision-Language Models for Multimodal Knowledge Graph Completion

    [https://arxiv.org/abs/2608.05833](https://arxiv.org/abs/2608.05833)

    提出ViSR-KGC方法，通过视觉子图推理结合视觉-语言模型，克服了传统方法将图结构线性化导致拓扑模糊和视觉信息丢失的问题，提升多模态知识图谱补全的效果。

    

    知识图谱补全（KGC）旨在从不完整的图结构中推断缺失的实体或关系，并已发展出多模态知识图谱补全（MMKGC），其中实体与文本和图像等多种模态相关联。传统的表示学习方法遵循基于嵌入的范式，在关系特定证据有限时可能表现不佳。与此同时，基于大语言模型（LLM）的推理方法通常将图结构线性化为文本提示，这模糊了结构拓扑并忽视了重要的视觉信息。虽然视觉-语言模型（VLM）擅长多模态推理，但它们无法原生理解结构化的图拓扑，尤其是当知识图谱的节点和边承载复杂语义时。为了弥合这一差距，我们提出了ViSR-KGC，一种用于知识图谱补全的视觉子图推理方法。它集成了三种互补能力来捕获语义

    arXiv:2608.05833v3 Announce Type: replace  Abstract: Knowledge graph completion (KGC) aims to infer missing entities or relations from incomplete graph structures, and has evolved into multimodal knowledge graph completion (MMKGC), where entities are associated with multiple modalities such as text and images. Traditional representation learning approaches follow the embedding-based paradigm and may struggle when relation-specific evidence is limited. Meanwhile, LLM-based reasoning methods typically linearize graph structures into textual prompts, which obscures structural topology and neglects vital visual information. While vision-language models (VLMs) excel at multimodal reasoning, they cannot natively interpret structured graph topology, particularly when it comes to knowledge graphs where nodes and edges carry complex semantics. To bridge this gap, we propose ViSR-KGC, a visual subgraph reasoning approach for KGC. It integrates three complementary capabilities to capture semantic
    
[^185]: KernelGenBench：一个面向基于大语言模型核函数生成的多源多芯片基准测试

    KernelGenBench: A Multi-Source and Multi-Chip Benchmark for LLM-based Kernel Generation

    [https://arxiv.org/abs/2607.27231](https://arxiv.org/abs/2607.27231)

    该论文提出了KernelGenBench，首个统一的多源多芯片基准，覆盖210个算子和六个硬件平台，用于系统评估大语言模型与智能体生成的Triton核函数在算子来源与硬件平台之间的性能迁移能力。

    

    现代人工智能系统依赖于专用加速器核函数，而日益多样化的算子和硬件使其开发变得复杂。大语言模型（LLM）和智能体系统有望实现这项工作的自动化，但现有评估无法表明其性能能否跨算子来源和硬件平台迁移，以及这种迁移的代价是什么。我们提出了KernelGenBench，这是首个用于评估由大语言模型和智能体生成的Triton核函数的统一多源、多芯片基础设施。通过以统一的Triton目标覆盖六个硬件平台，它在现有核函数生成基准中提供了最广泛的跨厂商硬件覆盖。我们报告了两个受控分析视角：KernelGenBench-MS（多源）涵盖来自PyTorch ATen、生产级vLLM算子和专有cuBLAS例程的210个算子，而KernelGenBench-MC（多芯片）则在六个硬件平台上评估一个语义稳定的110算子子集。

    arXiv:2607.27231v2 Announce Type: replace  Abstract: Modern AI systems depend on specialized accelerator kernels, whose development is complicated by increasingly diverse operators and hardware. LLMs and agentic systems promise to automate this work, but existing evaluations do not show whether their performance transfers across operator sources and hardware platforms, or what such transfer costs. We present KernelGenBench, the first unified multi-source and multi-chip infrastructure for evaluating LLM- and agent-generated Triton kernels. With a common Triton target spanning six hardware platforms, it provides the broadest cross-vendor hardware coverage among existing kernel-generation benchmarks. We report two controlled analytical views: KernelGenBench-MS (Multi-Source) covers 210 operators from PyTorch ATen, production vLLM operators, and proprietary cuBLAS routines, while KernelGenBench-MC (Multi-Chip) evaluates a semantically stable 110-operator subset across six hardware platform
    
[^186]: PRIME-SVR：面向胎儿T2定量成像的物理信息驱动的隐式多回波层到体重建

    PRIME-SVR: Physics-infoRmed Implicit Multi-Echo Slice-to-Volume Reconstruction for Fetal T2 mapping

    [https://arxiv.org/abs/2607.20136](https://arxiv.org/abs/2607.20136)

    提出了首个物理信息驱动的隐式神经表示框架PRIME-SVR，实现多回波胎儿MRI的联合高分辨率重建，使定量T2映射成为可能。

    

    层到体重建（SVR）是从多方向采集的运动模糊2D MRI切片堆栈中获取高分辨率（HR）3D胎儿大脑体积的标准方法。现有的SVR方法仅针对临床范围的回波时间（TE）进行优化和验证，限制了其在非临床TE下的应用，并使其与定量T2映射不兼容——而定量T2映射是一种独立于扫描协议和医疗中心的胎儿大脑成熟生物标志物，需要在多个TE下进行高分辨率重建。我们提出了PRIME-SVR，这是首个用于多回波MRI联合高分辨率重建的隐式神经表示（INR）框架。该框架使用一个全连接网络建模从空间坐标到跨TE信号强度的连续函数，同时使用另一个网络估计各切片特有的采集退化。通过基于Bloch方程推导的正则化项惩罚对预期T2衰减的偏离，从而强制实现跨TE的一致性。

    arXiv:2607.20136v2 Announce Type: replace-cross  Abstract: Slice-to-volume reconstruction (SVR) is the standard method for obtaining high-resolution (HR) 3D fetal brain volumes from motion-corrupted 2D MRI slice stacks acquired in multiple orientations. Existing SVR methods are optimized and validated only for clinical-range echo times (TEs), limiting their use at non-clinical TEs and making them incompatible with quantitative T2 mapping, a protocol- and center-independent biomarker of fetal brain maturation requiring HR reconstructions across multiple TEs. We present PRIME-SVR, the first implicit neural representation (INR) framework for joint HR reconstruction from multi-echo MRI. A single fully connected network models a continuous function from spatial coordinates to signal intensities across TEs, while a second network estimates slice-specific acquisition degradations. Cross-TE coherence is enforced via a Bloch equation-derived regularization penalizing deviations from expected T2
    
[^187]: 从看似合理到可操作：关于大语言模型自我解释的立场论文

    From Plausible to Actionable: A Position on LLM Self-Explanations

    [https://arxiv.org/abs/2607.15957](https://arxiv.org/abs/2607.15957)

    本立场论文指出大语言模型的自我解释虽高度合理但忠实性存疑，主张评估标准应从合理性与忠实性扩展到可操作性，并为此提供了实用的评估指南。

    

    大语言模型（LLM）能够生成用自然语言对其自身决策进行合理化说明的解释，这种现象通常被称为“自我解释”。此类解释已成为可解释人工智能（XAI）中一个有前景的研究方向，尤其在解释大语言模型行为方面。然而，尽管自我解释往往看起来合理，但它们是否忠实地反映了模型底层的推理过程仍是一个悬而未决的问题。在这篇观点论文中，我们提出：自我解释可以高度合理、忠实性却存疑，但同时又具有高度的可操作性。从传统XAI的视角出发，我们指出了针对LLM生成的自我解释的标准评估协议的局限性，并提出了评估其合理性与忠实性的实用指南。此外，我们主张评估应超越这些标准、扩展到可操作性维度，并重点阐述了LLM合理化解释的应用（摘要此处被截断）。

    arXiv:2607.15957v2 Announce Type: replace  Abstract: Large Language Models (LLMs) can generate natural language explanations that rationalize their own decisions, a phenomenon commonly referred to as self-explanations. Such explanations have emerged as a promising direction for explainable artificial intelligence (XAI), particularly for interpreting LLM behavior. However, while self-explanations often appear plausible, whether they faithfully reflect a model's underlying reasoning process remains an open question. In this opinion paper, we argue that self-explanations can be highly plausible, questionably faithful, and yet highly actionable. From a traditional XAI perspective, we identify the limitations of standard evaluation protocols for LLM-generated self-explanations and propose practical guidelines for assessing their plausibility and faithfulness.Moreover, we argue that evaluation should extend beyond these criteria to actionability, highlighting applications of LLM rationalizat
    
[^188]: EVOQUANT：面向稳健量化交易的自进化验证器引导策略优化

    EVOQUANT: Self-Evolving Verifier-Guided Strategy Optimization for Robust Quantitative Trading

    [https://arxiv.org/abs/2607.12455](https://arxiv.org/abs/2607.12455)

    EVOQUANT提出了一个自进化的验证器引导框架，利用大语言模型诊断量化策略瓶颈、生成受控编辑并通过多阶段验证筛选最优策略，同时将优化经验沉淀为可复用知识，实现量化交易策略的持续稳健自我改进。

    

    量化策略优化在很大程度上仍然是人工完成的，需要领域专家识别微弱信号、调整风险控制规则，并反复验证迭代修订方案。大语言模型可以加速这一过程，但直接依赖它们重写交易策略往往会引入幻觉式编辑、策略漂移和回测过拟合等问题。我们提出了EVOQUANT，一个用于量化交易策略优化的自进化验证器引导框架。我们的方法利用大语言模型深度诊断性能瓶颈，生成语义受控的候选编辑，通过多阶段验证流程筛选出最优策略，并将优化经验提炼为可复用的知识，以实现持续的自我改进。我们使用七种代表性策略对该方法进行评估：其中四种来自A股市场，三种来自加密货币市场。实验结果表明，我们的方法显著……

    arXiv:2607.12455v3 Announce Type: replace  Abstract: Quantitative strategy optimization remains largely manual, requiring domain experts to identify weak signals, tune risk-control rules, and repeatedly validate iterative revisions. Large language models can accelerate this process, but directly relying on them to rewrite trading strategies often introduces hallucinated edits, strategy drift, and backtest overfitting. We propose EVOQUANT, a self-Evolving Verifier-guided framework for strategy Optimization in Quantitative trading. Our method utilizes LLMs to deeply diagnose performance bottlenecks, generates semantically controlled candidate edits, selects the best strategy through a multi-stage verification pipeline, and distills optimization experience into reusable knowledge for continual self-improvement. We evaluate our method using seven representative strategies: four from the A-share market and three from the Crypto market. Experimental results show that our method significantly
    
[^189]: 建设者、防御者、破坏者：当生成式模型构建、防御和测试软件时的可测量独立性与有界自主性

    Builder, Defender, Breaker: Measurable Independence and Bounded Autonomy When Generative Models Build, Defend and Test Software

    [https://arxiv.org/abs/2607.03215](https://arxiv.org/abs/2607.03215)

    本文提出将独立性重新定义为软件生命周期角色对之间的可测量属性，并借助重合失效模型证明当生成式模型共享同一上游基底时，组织独立性不再意味着统计独立性，从而主张以有界自主取代一刀切式的人类监督要求。

    

    生成式模型如今能够编写应用代码、对代码进行加固和监控，并探测其中可利用的缺陷，因此同一族模型越来越多地同时扮演建设者、防御者和破坏者三种角色。主流观点将完全自主视为人类辅助的自然终点。本文主张一种比一刀切式要求人类监督更狭窄、也更站得住脚的立场。我们将“共享生成基底”定义为引发相关性错误的上游依赖集合（包括训练语料、模型族、对齐程序、供应商和工具链），并将独立性定义为生命周期角色对之间的可测量属性，而非任何单一参与者的固有属性。沿袭Eckhardt和Lee传统的重合失效模型表明，为什么组织独立性——验证标准历来所依赖的替代指标——一旦各角色共享同一基底，便不再意味着统计独立性；该模型还表明……

    arXiv:2607.03215v2 Announce Type: replace-cross  Abstract: Generative models now write application code, harden and monitor it, and probe it for exploitable flaws, so that one family of models increasingly plays builder, defender and breaker at once. The prevailing view treats full autonomy as the natural end point of assistance. This article argues for a narrower and more defensible position than a blanket requirement for human oversight. We define the shared generative substrate as the set of upstream dependencies (training corpus, model family, alignment procedure, vendor, toolchain) that induce correlated errors, and we define independence as a measurable property of pairs of lifecycle roles rather than an attribute of any participant. A coincident-failure model in the tradition of Eckhardt and Lee shows why organisational independence, the proxy on which verification standards have relied, no longer implies statistical independence once roles share a substrate; it also shows that 
    
[^190]: 谱几何与玻色-布洛赫探针：量子学习中的探索

    Spectral Geometry and Bosonic-Bloch Probes: Explorations in Quantum Learning

    [https://arxiv.org/abs/2607.00063](https://arxiv.org/abs/2607.00063)

    本文发现量子学习的训练过程会重塑网络输出的谱几何结构，并提出利用双玻色子干涉和布洛赫空间漂移作为物理探针来诊断这种谱重构，建立了学习到的谱划分与干涉特征之间的定量关联。

    

    本文研究了谱几何如何在量子学习模型中涌现，以及如何利用具有物理基础的探针对其进行诊断。在图正则化量子网络中，训练过程会重组输出相似度图，使有效谱维度增加 ΔS = +0.23，并重塑拉普拉斯谱。边分辨的双玻色子干涉直接探测了这种结构重组：玻色子增强 ΔP_uv 与 Fiedler 边分裂 |Δv_2| 之间存在相关性（r = -0.50），从而将学习到的谱划分与干涉特征联系起来。相图显示性能对耦合强度 γ 和噪声 δ 呈非单调依赖关系，图正则化仅在受限参数区域内提升保真度；硬件实验在散粒噪声不确定度范围内证实了预测的干涉行为。我们还分析了一个混合量子自编码器，并引入布洛赫空间漂移作为几何诊断手段。

    arXiv:2607.00063v4 Announce Type: replace-cross  Abstract: This paper studies how spectral geometry emerges in quantum learning models and how it can be diagnosed with physically grounded probes. In graph-regularized quantum networks, training reorganizes the output similarity graph, increases the effective spectral dimension Delta S = +0.23, and reshapes the Laplacian spectrum. Edge-resolved two-boson interference directly probes this restructuring: the bosonic enhancement Delta P_uv correlates with the Fiedler edge split |Delta v_2| (r = -0.50), linking learned spectral partitions to interference signatures. A phase diagram shows a nonmonotonic dependence of performance on coupling strength gamma and noise delta, with graph regularization improving fidelity only in a restricted regime; hardware experiments confirm the predicted interference behavior within shot-noise uncertainty. We also analyze a hybrid quantum autoencoder and introduce Bloch-space drift as a geometric diagnostic of
    
[^191]: FP8 就是你所需（第二部分）：在 FP8 代张量核心上实现全 FP64 三维 FFT——整数尾声之墙与消除它的最小硬件

    FP8 is All You Need (Part 2): Full-FP64 3-D FFT on FP8-Generation Tensor CoresThe Integer-Epilogue Wall and the Minimal Hardware That Would Remove It

    [https://arxiv.org/abs/2606.23698](https://arxiv.org/abs/2606.23698)

    该论文提出一种在 FP8 张量核心上实现全 FP64 1024³ 三维 FFT 的无 FP64 算术 Bailey 六步变换方案，并发现制约性能的瓶颈是定点累加的“整数尾声”阶段而非浮点运算本身。

    

    NVIDIA Blackwell Ultra (B300) GPU 将 FP64 向量吞吐量削减约 30 倍，同时成倍提升 FP8 张量吞吐量。在继任者论文（"FP8 is All You Need, Part 1" 和 "Ozaki 2.5"）中通过 Ozaki Scheme II 在 FP8 张量核心上恢复 FP64 GEMM 性能并建立张量-内存均衡模型之后，我们探讨第五个经典 HPC 原语——全 FP64 1024³ 三维 FFT——能否由同样的基底承载，并以一个设计方案及其极限作为回答。该方案是一个不含任何 FP64 算术的 Bailey 六步变换：采用带融合旋转因子的 FP8 张量 DFT GEMM、剩余域 Karatsuba 组合以及精确 CRT 重构，其主体是 FP16 张量路径上的一个小型 GEMM，其余部分是带双侧模 M 提升的 Kulisch 定点累加，因此唯一的舍入仅发生在最终转换；所有常数均由机器生成并经位级精确验证。核心发现是：制约性资源并非浮点运算，而是……

    arXiv:2606.23698v3 Announce Type: replace-cross  Abstract: The NVIDIA Blackwell Ultra (B300) GPU cuts FP64 vector throughput $\sim 30\times$ while multiplying FP8 tensor throughput. After the recovery of FP64 GEMM via Ozaki Scheme II on FP8 tensor cores and the Tensor-Memory Equilibrium model of the companions ("FP8 is All You Need, Part 1" and "Ozaki 2.5") we ask whether the fifth canonical HPC primitive, the full-FP64 $1024^3$ 3-D FFT, can be carried by the same substrate, and answer with a design and its limit. It is a Bailey six-step transform with no FP64 arithmetic: FP8-tensor DFT GEMMs with fused twiddles, residue-domain Karatsuba combines and exact CRT reconstruction whose bulk is a small GEMM on the FP16 tensor path and whose remainder is a Kulisch fixed-point accumulation with a two-sided modulo-$M$ lift, so the only rounding is the final conversion; constants are machine-generated and verified bit-exactly. The central finding: the binding resource is not floating point but a
    
[^192]: PSCT-Net：基于可微反投影与注意力引导精修的几何感知儿科颅骨CT重建

    PSCT-Net: Geometry-Aware Pediatric Skull CT Reconstruction via Differentiable Back-Projection and Attention-Guided Refinement

    [https://arxiv.org/abs/2606.19867](https://arxiv.org/abs/2606.19867)

    本文提出PSCT-Net框架，通过可微反投影、注意力引导投影（AGP-3D）和双向Mamba（BiM-3D）模块，从稀疏双平面X射线实现几何感知的低剂量儿科颅骨3D CT重建。

    

    计算机断层扫描（CT）对于诊断儿科颅面异常至关重要，但会对发育中的解剖结构造成辐射风险。从稀疏的双平面X射线重建3D CT提供了一种低剂量替代方案，但该问题严重病态。现有方法采用几何无关的特征提升，在没有显式空间建模的情况下将2D特征简单地投影到3D空间，导致深度模糊和骨性边界退化。我们提出了PSCT-Net，一个具有可微反投影的几何感知框架。可微反投影建立了空间保真的体积先验，缓解了深度模糊。注意力引导投影（AGP-3D）模块学习2D区域与3D位置之间的非线性体素级对应关系。双向Mamba（BiM-3D）模块以线性复杂度捕获长程体积依赖关系。我们进一步整理了一个私人机构的儿科颅骨CT队列数据集……

    arXiv:2606.19867v4 Announce Type: replace-cross  Abstract: Computed Tomography (CT) is essential for diagnosing pediatric craniofacial abnormalities, yet poses radiation risks to developing anatomies. Reconstructing 3D CT from sparse bi-planar X-rays offers a low-dose alternative but is severely ill-posed. Existing methods employ geometry-agnostic feature lifting, naively projecting 2D features into 3D without explicit spatial modeling, causing depth ambiguity and degraded osseous boundaries. We present PSCT-Net, a geometry-aware framework with differentiable back-projection. Differentiable back-projection establishes a spatially faithful volumetric prior, alleviating depth ambiguity. An Attention-Guided Projection (AGP-3D) module then learns non-linear voxel-wise correspondences between 2D regions and 3D locations. A Bidirectional Mamba (BiM-3D) module captures long-range volumetric dependencies with linear complexity. We further curate a private institutional pediatric skull CT cohor
    
[^193]: 心理健康对话中的专家级危机检测

    Expert-Level Crisis Detection in Mental Health Conversations

    [https://arxiv.org/abs/2606.10380](https://arxiv.org/abs/2606.10380)

    该论文提出了临床医生标注的CRADLE-Dialogue基准数据集以及“警报-确认”评估协议，用于解决多轮心理健康对话中轮次级危机检测的难题，使模型能够捕捉随对话演进的风险信号并支持早期干预。

    

    现实世界的危机干预本质上是对话式的，然而现有研究主要集中于静态文本。当应用于多轮对话时，当前模型表现出显著的性能下降，难以追踪随着上下文演变而出现的风险信号。为了弥补这一空白，我们推出了CRADLE-Dialogue，这是一个由临床医生标注的、用于对话环境中轮次级危机检测的基准数据集。该数据集包含600段对话，针对基于临床的风险（包括自杀意念、自残和虐待儿童）进行了多标签标注，并区分了过去风险与正在发生的风险。我们进一步提出了“警报-确认”评估协议，将早期预警信号与特定危机变得明确可识别的对话轮次区分开来，体现了在风险变得明显之前进行干预的临床需求。实验表明，识别风险何时出现远比识别风险本身困难得多。

    arXiv:2606.10380v2 Announce Type: replace  Abstract: Real-world crisis intervention is inherently conversational, yet existing research largely focuses on static texts. When applied to multi-turn dialogues, current models exhibit significant performance degradation, struggling to track risk signals that emerge as context evolves. To address this gap, we introduce CRADLE-Dialogue, a clinician-annotated benchmark for turn-level crisis detection in conversational settings. The dataset features 600 dialogues with multi-label annotations across clinically grounded risks, including suicide ideation, self-harm, and child abuse, distinguishing past from ongoing risk. We further propose an Alert-Confirm evaluation protocol that distinguishes early warning signals (Alert) from turns where a specific crisis becomes explicitly identifiable (Confirm), reflecting the clinical need to intervene before risk becomes explicit. Experiments show that identifying when risk emerges is much harder than recog
    
[^194]: FiberTune：在视觉-语言-动作模型微调中保留动作纤维视觉残差

    FiberTune: Preserving Action-Fiber Visual Residuals in Vision-Language-Action Fine-Tuning

    [https://arxiv.org/abs/2606.08653](https://arxiv.org/abs/2606.08653)

    FiberTune 提出一种训练时目标函数，通过在线动作探针滤除动作预测方向并与冻结视觉教师对齐，防止 VLA 微调中的残差视觉坍缩，从而在不增加推理开销的情况下在六项仿真设置中全面超越仅用任务损失的微调。

    

    对视觉-语言-动作（VLA）策略进行动作监督微调能够有效拟合演示数据，但其约束仅限于会改变预测动作的方向，而使得在动作等价状态之间本应保持一致的视觉结构可以自由坍缩。我们将这一现象形式化为沿局部动作纤维的残差视觉坍缩，并提出 FiberTune——一种训练时目标函数，能够在不增加推理时开销的情况下保留教师模型结构化的视觉残差。FiberTune 使用在线动作探针来估计与动作预测相关的特征方向，从中间视觉标记表示中滤除这些方向，并将过滤后的残差与冻结的视觉教师模型对齐，同时对其有效秩进行正则化。在相同的训练条件下，FiberTune 在涵盖两个基准和两种架构（pi_0.5 与 OpenVLA-OFT）的全部六项受控仿真设置中均优于仅使用任务损失的微调方法。

    arXiv:2606.08653v2 Announce Type: replace-cross  Abstract: Action-supervised fine-tuning of vision-language-action (VLA) policies fits demonstrations effectively but constrains only the directions that change predicted actions, leaving visual structure consistent across action-equivalent states free to collapse. We formalize this as residual visual collapse along local action fibers and propose FiberTune, a training-time objective that preserves teacher-structured visual residuals without adding inference-time overhead. FiberTune uses an online action probe to estimate action-predictive feature directions, filters them from intermediate visual-token representations, and aligns the resulting probe-filtered residuals to a frozen visual teacher while regularizing their effective rank. Under identical training conditions, FiberTune improves over task-loss-only fine-tuning in every one of six controlled simulation settings spanning two benchmarks and two architectures (pi_0.5 and OpenVLA-OF
    
[^195]: 自进化科学智能体发现可泛化的物理推理流体控制

    Self-Evolving Scientific Agent Discovers Generalizable Physically-Reasoned Fluid Control

    [https://arxiv.org/abs/2606.08405](https://arxiv.org/abs/2606.08405)

    本文提出了一种基于大语言模型的自进化科学智能体工作流，通过迭代代码生成和物理模拟诊断，自动构建可解释的白盒控制器，并在非线性流固耦合的欠驱动游泳者目标到达任务中验证了其泛化能力。

    

    摘要：虽然数据密集型的深度强化学习可以优化复杂的控制策略，但物理系统中的科学控制设计从根本上需要一条可解释的推理链，将物理证据与结构化控制架构联系起来。在此，我们提出了一种由大型语言模型和迭代代码生成驱动的自进化科学智能体工作流，该工作流在保持严格可解释性和严谨物理推理的同时，自动化控制器构建过程。该智能体不调整权重，而是将候选白盒控制器部署到物理模拟中，从多模态证据中主动诊断动态行为，并将这些观察转化为逐步的源代码改进。我们在一个高度非线性的流固耦合问题上演示了这一框架：一个欠驱动、双关节狗鱼游泳者，仅使用关节运动在非定常流中完成空间目标到达任务。

    arXiv:2606.08405v2 Announce Type: replace  Abstract: While data-intensive deep reinforcement learning can optimize complex control policies, scientific control design in physical systems fundamentally requires an interpretable chain of reasoning that connects physical evidence to structured control architectures. Here, we present a self-evolving scientific agent workflow, driven by large language models and iterative code generation, that automates controller construction while preserving strict interpretability and rigorous physical reasoning. Instead of adjusting weights, the agent deploys candidate whitebox controllers into physical simulations, actively diagnoses dynamic behaviors from multimodal evidence, and translates these observations into progressive source-code refinements. We demonstrate this framework on a highly non-linear fluid-structure interaction problem: an underactuated, two-joint dogfish swimmer tasked with spatial target reaching in an unsteady flow using only joi
    
[^196]: FP8 就是你所需要的一切（第一部分）：破除硬件 FP64 作为高性能计算圣杯的神话（9月3日版本）

    FP8 is All You Need (Part 1): Debunking Hardware FP64 as the HPC Holy Grail (Sep 3rd version)

    [https://arxiv.org/abs/2606.06510](https://arxiv.org/abs/2606.06510)

    本文提出在 NVIDIA B300 世代及以后的 AI 优化 GPU 上，通过基于 CRT 的 Ozaki Scheme II 组合的 FP8 张量核心运算可在 FP64 级精度下作为矩阵密集型 FP64 内核的主要运算基底层，并引入张量-内存均衡（TME）模型作为分析工具。

    

    我们论证，在 NVIDIA B300 世代及以后的 AI 优化 GPU 上，通过基于 CRT 的 Ozaki Scheme II 组合而成的 FP8 张量核心矩阵运算，可以作为所考察的以矩阵为主的 FP64 内核类别的主要矩阵工作基底层，达到 FP64 级别的精度，而原生 FP64 则从硬件要求转变为推导出的精度保证。该主张是有条件的：FP8 运算是候选的主要乘法基底层，配合一组有界的辅助工作集——整数解构/重构运算、FP32/Kulisch 归约、数据移动以及原生 FP64 回退方案——组织成从 FP8 运算经 Ozaki II 和 Berkeley dwarfs 到应用程序的层次结构。所使用的分析工具是张量-内存均衡（TME）模型，这是 Roofline 模型的扩展，具有四个参数（计算乘数 α=3r+1、带宽乘数 β、重构成本 γ，以及每输入解构成本 c……（摘要原文在此处截断）

    arXiv:2606.06510v4 Announce Type: replace-cross  Abstract: We argue that on AI-optimised GPUs of the NVIDIA B300 generation and beyond, the FP8 tensor-core matrix operation, composed through CRT-based Ozaki Scheme II, can serve as the dominant matrix-work substrate for the surveyed matrix-dominated FP64 kernel classes at FP64-grade accuracy, with native FP64 recast from a hardware requirement into a derived accuracy guarantee. The claim is conditional: the FP8 op is the candidate dominant multiplication substrate, with a bounded auxiliary set of integer deconstruction/reconstruction work, FP32/Kulisch reductions, data movement and a native-FP64 fallback, organised as a hierarchy from the FP8 op through Ozaki II and the Berkeley dwarfs to applications. The instrument is the Tensor-Memory Equilibrium (TME) model, a Roofline extension with four parameters (compute multiplier $\alpha=3r+1$, bandwidth multiplier $\beta$, reconstruction cost $\gamma$, and the per-input deconstruction cost $c
    
[^197]: 利用奖励不确定性在强化学习中诱导多样性行为

    Using Reward Uncertainty to Induce Diverse Behaviour in Reinforcement Learning

    [https://arxiv.org/abs/2606.03962](https://arxiv.org/abs/2606.03962)

    提出将强化学习目标进行根本性重构——用奖励函数的分布替代标量奖励并在动作集合上应用非线性目标，从而将多样性行为自然地理解为智能体对奖励不确定性的理性响应。

    

    经典强化学习（RL）通常寻求一个确定性的策略来最大化标量奖励的期望总和。然而，诸如语言模型微调或科学发现等现代应用需要多样性。现有的解决方案（如熵正则化或多样性奖励）往往需要脆弱的权衡——为了随机性而牺牲性能，或者依赖于可能与策略排序不一致的启发式度量。我们认为，多样性更自然地被理解为对奖励不确定性的理性响应。当奖励函数并非完全已知时（例如存在模糊的偏好或不完美的奖励模型），坚持单一动作可能是次优的。基于此，我们提出了对RL目标的根本性重新表述：用奖励函数的分布替代标量奖励，并在动作集合上应用非线性目标。

    arXiv:2606.03962v2 Announce Type: replace-cross  Abstract: Classical reinforcement learning (RL) typically seeks a deterministic policy that maximizes the expected sum of a scalar reward. Yet, modern applications such as language model fine-tuning or scientific discovery demand diversity. Existing remedies such as entropy regularization or diversity bonuses often require fragile trade-offs that sacrifice performance for stochasticity or rely on heuristic metrics that can misalign policy rankings. We argue that diversity is more naturally understood as the rational response to uncertainty in the reward. When the reward function is not perfectly known--as is the case with ambiguous preferences or imperfect reward models--committing to a single action can be sub-optimal. Building on this, we propose a fundamental reformulation of the RL objective by replacing the scalar reward with a distribution over reward functions, and applying a non-linear objective over sets of actions. The result i
    
[^198]: BaltiVoice：面向巴尔蒂语的语音语料库与微调Whisper自动语音识别系统

    BaltiVoice: A Speech Corpus and Fine-tuned Whisper ASR System for the Balti Language

    [https://arxiv.org/abs/2606.03504](https://arxiv.org/abs/2606.03504)

    该论文发布了首个巴尔蒂语公开语音语料库BaltiVoice（16.8小时），并通过对Whisper-small进行微调，将该语言的识别词错误率从零样本基线的159.19%大幅降低至24.78%，填补了这一低资源藏语支语言在语音识别领域的空白。

    

    我们提出了BaltiVoice，这是一个面向巴尔蒂语（ISO 639-3: bft）的16.8小时朗读语音语料库。巴尔蒂语是一种在巴基斯坦吉尔吉特-巴尔蒂斯坦地区使用的藏语支语言，此前没有任何公开可用的自动语音识别（ASR）资源。该语料库包含10,060条经过验证的、以本土纳斯塔利格文书写的语句，来源于Mozilla Common Voice录音。对OpenAI Whisper-small进行微调，训练5个周期（3,000步）后，在538条语句的说话人分离验证集上，词错误率（WER）达到24.78%，字符错误率（CER）达到8.30%，相比零样本基线的159.19% WER和152.52% CER大幅下降。在相同数据上微调的Whisper-base达到44.54%的WER和15.61%的CER，证实了在这种低资源场景下模型容量至关重要。数据集、微调后的模型以及实时转录演示均已在HuggingFace上公开发布。

    arXiv:2606.03504v3 Announce Type: replace  Abstract: We present BaltiVoice, a 16.8-hour read-speech corpus for Balti (ISO 639-3: bft), a Tibetic language spoken in Gilgit-Baltistan, Pakistan, with no prior publicly available ASR resources. The corpus contains 10,060 validated utterances in native Nastaliq script, derived from Mozilla Common Voice recordings. Fine-tuning OpenAI Whisper-small yields a Word Error Rate (WER) of 24.78% and a Character Error Rate (CER) of 8.30% after training for 5 epochs (3,000 steps) on the 538-utterance speaker-disjoint validation set, down from a zero-shot baseline of 159.19% WER and 152.52% CER. A Whisper-base fine-tuned on the same data achieves 44.54% WER and 15.61% CER, confirming that model capacity matters for this low-resource setting. The dataset, fine-tuned model, and a live transcription demo are publicly available on HuggingFace.
    
[^199]: KairosAgent：基于融合语义推理的智能体时间序列预测

    KairosAgent: Agentic Time Series Forecasting with Fused Semantic Reasoning

    [https://arxiv.org/abs/2605.30002](https://arxiv.org/abs/2605.30002)

    提出了KairosAgent智能体框架，通过结合LLM推理器与TSFM预测器，并动态调用分析工具以增强数值理解与语义推理能力，实现了多模态时间序列预测中文本推理与数值预测的统一。

    

    跨领域多模态时间序列预测是一项具有挑战性的任务，要求模型能够整合精确的数值理解能力、跨领域语义理解能力以及有效的多模态融合能力。现有方法要么从零开始构建时间序列基础模型（TSFM），要么利用预训练的大语言模型（LLM）。然而，TSFM往往忽视语义理解，缺乏面向未来的语义推理能力，而LLM则在数值理解和准确的定量预测方面表现不佳。为了克服这些局限性，我们提出了KairosAgent，这是一种新颖的用于多模态时间序列预测的智能体框架，包含一个基于LLM的推理器和一个基于TSFM的预测器。KairosAgent通过动态调用分析工具来增强LLM的数值理解和语义推理能力，从而将文本推理与数值预测统一起来。推理结果被……

    arXiv:2605.30002v2 Announce Type: replace  Abstract: Cross-domain multimodal time series forecasting is a challenging task, requiring models to integrate precise numerical comprehension, cross-domain semantic understanding, and effective multimodal fusion. Existing approaches either build Time Series Foundation Models (TSFMs) from scratch or leverage pretrained Large Language Models (LLMs). However, TSFMs often overlook semantic understanding and lack the ability to perform future-oriented semantic reasoning, and LLMs struggle with numerical comprehension and accurate quantitative forecasting. To overcome these limitations, we propose KairosAgent, a novel agentic framework for multimodal time series forecasting, including an LLM-based reasoner and a TSFM-based forecaster. KairosAgent unifies textual reasoning and numerical forecasting by dynamically invoking analytical tools to enhance the numerical understanding and semantic reasoning capabilities of LLMs. The reasoning results are su
    
[^200]: 语言模型中的文化绑定注意力头

    Cultural Binding Heads in Language Models

    [https://arxiv.org/abs/2605.28543](https://arxiv.org/abs/2605.28543)

    该研究通过机制可解释性方法在八个语言模型中识别出2-3个负责文化绑定的中层注意力头，证明文化绑定形成于预训练阶段，并通过生成阶段的适度放大引导将文化区分准确率提升1-3个百分点。

    

    大语言模型往往对不同文化群体采取一视同仁的默认处理方式，即使上下文需要做出区分：这是一种缺乏差异意识的表现。我们利用机制可解释性方法，并在Wang等人（2025）提出的N4文化挪用基准上采用因子实验设计，在八个模型（四种架构的基础版和指令微调版）中识别出每个模型中2-3个对文化绑定具有因果贡献的中层注意力头。文化绑定是指将文化项目与其相关身份关联起来的过程。敲除这些注意力头上从身份到项目的连接边可使绑定强度降低9-23%。所识别的注意力头能够从指令微调模型迁移到基础模型，表明文化绑定是在预训练期间形成的。α缩放实验显示出分级的剂量-响应关系。在生成阶段进行适度的放大引导（α=2-3）可将文化区分准确率提高1-3个百分点，同时对推理能力的影响保持在可接受范围内。

    arXiv:2605.28543v3 Announce Type: replace-cross  Abstract: LLMs often default to equal treatment across cultural groups, even though context warrants differentiation: this is a lack of difference awareness. Using mechanistic interpretability and a factorial design on the N4 cultural appropriation benchmark from Wang et al. (2025), we identify 2-3 mid-layer attention heads per model that contribute causally to cultural binding across eight models (base and instruct versions of four architectures). Cultural binding is the process of associating a cultural item with its related identity. Knockout of the identity-to-item edges on these heads lowers the binding strength by 9-23%. The identified heads transfer from instruct to base models, suggesting that cultural binding is created during pre-training. An $\alpha$-scaling shows a graded dose-response. Moderate amplification steering at generation ($\alpha = 2-3$) increases cultural differentiation accuracy by 1-3 pp while leaving reasoning 
    
[^201]: 追踪大语言模型中的计算密度

    Tracing Computation Density in LLMs

    [https://arxiv.org/abs/2605.27033](https://arxiv.org/abs/2605.27033)

    本文提出s-Trace方法来估计能近似完整输出的LLM计算子图，发现模型计算呈现两阶段组织模式：早期层节点构成的小子图即可重构输出分布主体，后续计算仅为渐进式精细化，且每个输入所需的计算量与模型不确定性相关。

    

    基于Transformer的大语言模型（LLM）由数十亿个参数组成的深层和宽层计算图构成，但目前尚不清楚它们是否对所有输入都充分利用了其全部容量。我们提出了s-Trace方法，可以高效地估计一个大小为s的子图，该子图能够近似完整的模型输出。通过这种方法，我们发现多种LLM中的计算以两个不同的阶段进行组织。一个主要由早期层节点组成的小型子图即可重构完整模型输出分布的头部。随着添加更多节点（主要位于较深层，且越来越多地由注意力头构成），对完整输出分布的近似会得到渐进式的精细化。此外，我们还发现每个输入所需的计算量与模型不确定性相关，且更稀疏的子图编码的是浅层统计信息，例如一元词频。总体而言，我们的结果表明了一种一致的

    arXiv:2605.27033v2 Announce Type: replace  Abstract: Transformer-based large language models (LLMs) are comprised of billions of parameters arranged in deep and wide computational graphs, but it is not clear that they exploit their full capacity for all inputs. We introduce the s-Trace method to efficiently estimate a subgraph of size s that approximates a full model output. With this method, we find the computation in a variety of LLMs to be organized in two distinct phases. A small subgraph mostly composed of early-layer nodes can reconstruct the head of the full model output distribution. Adding further nodes, mostly located in later layers and increasingly consisting of attention heads, leads to incremental refinements in approximating the full output distribution. We find moreover that the amount of necessary computation per input correlates with model uncertainty, and that sparser subgraphs encode shallow statistics, such as unigram frequency. Overall, our results suggest a consi
    
[^202]: SpecBench：衡量长时程编码智能体中的奖励作弊行为

    SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents

    [https://arxiv.org/abs/2605.21384](https://arxiv.org/abs/2605.21384)

    SpecBench通过将软件工程任务分解为规范描述、可见验证测试和保留测试三部分，利用智能体在可见测试与保留测试上通过率的差距，量化了长时程编码智能体中的奖励作弊行为。

    

    随着长时程编码智能体产生的代码量超过任何开发者所能审查的限度，人类监督便集中到唯一的表面上：自动化测试套件。在这种设置下，奖励作弊（reward hacking）现象自然产生，因为智能体以通过测试为目标进行优化，却偏离了用户的真实目标。我们通过将软件工程任务分解为三个部分来研究这一奖励作弊现象：（i）规范的自然语言描述，（ii）针对已规定功能进行孤立验证的可见验证测试，以及（iii）将这些功能组合起来以模拟真实世界使用的保留测试。基于规范和可见验证测试套件，一个真正可靠的智能体应当能够生成同样可以通过所有保留测试的解决方案。因此，我们使用智能体在这两个测试套件上通过率的差距来量化奖励作弊。基于这一方法，我们推出了SpecBench，一个包含30个系统级（任务）的基准测试。

    arXiv:2605.21384v2 Announce Type: replace-cross  Abstract: As long-horizon coding agents produce more code than any developer can review, oversight collapses onto a single surface: the automated test suite. Reward hacking naturally arises in this setup, as the agent optimizes for passing tests while deviating from the users true goal. We study this reward hacking phenomenon by decompose software engineering tasks into three parts: (i) a natural language description of the specification (ii) visible validation tests that exercise specified features in isolation, and (iii) held-out tests that compose those same features to simulate real-world usage. Based on the specification and the visible validation test suites, a genuine agent would be able to generate a solution that can also pass all of the held-out tests. Therefore we use the gap in pass rates on these two suites to quantify reward hacking. Based on this methodology, we introduce SpecBench, a benchmark comprising 30 systems-level 
    
[^203]: 在大语言模型后训练中通过logit平均用SFT补充强化学习

    Complementing reinforcement learning with SFT through logit averaging in the post training of LLMs

    [https://arxiv.org/abs/2605.20555](https://arxiv.org/abs/2605.20555)

    提出一种通过logit平均将冻结SFT参考策略与可训练策略相耦合的新方法并融入GRPO，无需KL正则化即可在保持SFT格式优势的同时提升模型推理能力，在多个基准测试上达到或超越标准GRPO的准确率。

    

    我们提出了一种新颖的方法，将冻结的参考策略（如SFT）与可训练策略的logits进行平均，并将该方法融入组相对策略优化（GRPO）中。与可验证奖励强化学习（RLVR）方法不同，我们的方案不涉及KL（Kullback-Leibler）散度正则化或评论家网络；可训练策略与参考锚点通过logit平均结构相耦合，从而在利用可训练策略推理能力的同时，保持SFT的格式优势。我们在MATH、cn-k12和MMLU上对该方法进行了评估，结果显示其相对于标准的KL正则化GRPO具有更高或至少相当的准确率。

    arXiv:2605.20555v2 Announce Type: replace-cross  Abstract: We introduce a novel method that averages the logits of a frozen reference policy (e.g., SFT) and a trainable policy, and incorporate the method into Group Relative Policy Optimization (GRPO). In contrast to Reinforcement Learning with Verifiable Rewards (RLVR) methods, our proposal does not involve a Kullback Leibler (KL) regularization or critic; the trainable policy and the reference anchor are coupled through the logit averaging structure to leverage the reasoning expertise of the trainable policy while maintaining the formatting advantage of SFT. Our method is evaluated on MATH, cn-k12, and MMLU, and the results show a higher accuracy or at least comparable accuracy relative to the canonical KL-regularized GRPO.
    
[^204]: 有据续写：一种面向LLM对话的线性时间运行时验证器

    Grounded Continuation: A Linear-Time Runtime Verifier for LLM Conversations

    [https://arxiv.org/abs/2605.14175](https://arxiv.org/abs/2605.14175)

    提出了一种线性时间的运行时验证器，通过LLM解释器将话语分类为八种认知操作并用符号引擎维护依赖图，无需额外LLM调用即可判定对话续写是否基于仍然成立的前提，并能精确追踪前提撤销对结论的影响。

    

    在长对话中，大语言模型（LLM）可能产生看似合理、实则建立在对话早已放弃的前提之上的续写。目前没有任何运行时检查机制能将模型输出与对话已确立的内容绑定，这一缺口正被针对已部署智能体的上下文操纵攻击所利用。我们用一个运行时验证器来填补这一缺口：一个LLM解释器将每条话语分类为八种认知操作之一，一个符号引擎将这些操作应用于一张依赖图，该图记录了每个断言所依赖的内容及其是否仍然成立。判断一个续写是否有据可依可以归结为对该依赖图的一次遍历，时间复杂度与图的大小成线性关系，且无需调用LLM。撤销操作通过同一张依赖图传播，并具有无冲突保证，能够精确标记出失去支持的结论。在ReviseQA（信念修正）和MemoryAgentBench的事实整合子集这两个第三方基准上——即先前前提会被取代的场景——该验证器在预算匹配的检索基线中表现领先……（原文摘要在此截断）

    arXiv:2605.14175v2 Announce Type: replace  Abstract: In a long conversation, an LLM can produce a plausible continuation that rests on premises the conversation has already abandoned. No runtime check ties its output to what the conversation has established, a gap that context-manipulation attacks on deployed agents exploit. We close this gap with a runtime verifier: an LLM Interpreter classifies each utterance into one of eight epistemic operations, and a symbolic engine applies them to a dependency map that records what every claim rests on and whether it still stands. Whether a continuation is grounded reduces to a walk over the map, linear in its size, with no LLM call. Retraction propagates through the same map with a conflict-free guarantee, flagging exactly the conclusions that lose support. On ReviseQA for belief revision and MemoryAgentBench's fact-consolidation split, two third-party benchmarks where earlier premises are superseded, the verifier leads a budget-matched retriev
    
[^205]: EVA-Bench：一个新的语音智能体端到端评估框架

    EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents

    [https://arxiv.org/abs/2605.13841](https://arxiv.org/abs/2605.13841)

    EVA-Bench提出了一个端到端语音智能体评估框架，通过带自动验证的机器人间动态音频对话模拟与EVA-A（准确性）、EVA-X（体验）两项复合指标，首次同时实现了真实对话模拟与全面的语音专项评估。

    

    语音智能体在企业应用中的部署日益增多。然而，目前尚无现有基准能够同时解决真实对话模拟和全面的语音专项评估这两个问题。我们提出了EVA-Bench，一个能够同时应对这两个问题的端到端评估框架。在模拟方面，EVA-Bench编排动态的机器人对机器人音频对话，并通过自动模拟验证来检测用户模拟器的错误，在评分前适当地重新生成对话。在测量方面，EVA-Bench引入了两个复合指标：EVA-A（准确性）和EVA-X（体验）。EVA-Bench涵盖三个企业领域的213个场景、用于评估口音和噪声鲁棒性的受控扰动套件，以及区分峰值能力与可靠能力的多次试验测量。在对跨越三种架构的12个系统的评估中，我们发现：（1）没有系统能同时在EVA-A pass@1和EV...

    arXiv:2605.13841v3 Announce Type: replace-cross  Abstract: Voice agents are increasingly deployed across enterprise applications. However, no existing benchmark jointly addresses realistic conversation simulation and comprehensive voice-specific evaluation. We present EVA-Bench, an end-to-end evaluation framework that addresses both. On the simulation side, EVA-Bench orchestrates dynamic bot-to-bot audio conversations with automatic simulation validation that detects user simulator error and appropriately regenerates conversations before scoring. On the measurement side, EVA-Bench introduces two composite metrics: EVA-A (Accuracy) and EVA-X (Experience). EVA-Bench includes 213 scenarios across three enterprise domains, a controlled perturbation suite for accent and noise robustness, and multi-trial measurements that distinguish peak from reliable capability. Across 12 systems spanning all three architectures, we find: (1) no system simultaneously exceeds 0.5 on both EVA-A pass@1 and EV
    
[^206]: “你究竟想做什么？”：从日常计算机使用中共同创造人生目标

    "What Are You Really Trying to Do?": Co-Creating Life Goals from Everyday Computer Use

    [https://arxiv.org/abs/2605.00497](https://arxiv.org/abs/2605.00497)

    本文提出“追求共创”方法，基于活动理论和个人追求框架，从日常计算机使用的非结构化观察中逐步推断用户更广泛的人生目标，并通过编辑界面让用户掌控系统对自身的理解，突破现有系统仅提供表面级支持的局限。

    

    用户建模的最新进展使得对个人日常计算机使用进行开放式推理成为可能。尽管长期以来人们一直设想系统能够深入理解我们的行为及其在生活中的目的，但现有系统只能捕捉用户当下的行为，而无法理解其背后的原因，这限制了这些系统只能提供浅层次的支持。我们提出了“追求共创”，这是一种从计算机使用的非结构化观察中推断更广泛人生目标的过程。基于活动理论和埃蒙斯的个人追求框架，我们的系统逐步构建个人活动的层次化表示。然而，仅凭观察很难完全确定个人的追求目标，因为同一行为可能由许多不同的目标所驱动。因此，我们的系统支持一个编辑界面，让用户能够对系统如何理解自己拥有主动权，并将其修正反馈给系统……

    arXiv:2605.00497v2 Announce Type: replace-cross  Abstract: Recent advances in user modeling make it feasible to conduct open-ended inference over a person's everyday computer use. Despite longstanding visions of systems that deeply understand our actions and the purposes they serve in our lives, existing systems only capture what a person is doing in the moment, not why they are doing it, limiting these systems to surface-level support. We introduce striving co-creation, a process for inferring broader life goals from unstructured observations of computer use. Grounded in Activity Theory and Emmons' personal strivings framework, our system progressively constructs a hierarchical representation of a person's activities. Strivings are, however, difficult to fully resolve from observation alone, as the same action can be driven by many different goals. Our system therefore supports an editing interface that gives people agency over how they are understood by the system, feeding their corr
    
[^207]: 不要只教，要解释！面向网络安全教育的游戏化“20问”推荐系统

    Dont Just Teach, Explain! A Gamified 20Q Recommender for Cybersecurity Education

    [https://arxiv.org/abs/2604.26964](https://arxiv.org/abs/2604.26964)

    该论文提出了一种将可解释人工智能（XAI）与强化学习相结合的游戏化“20问”交互式教育框架，通过智能体扮演提问者引导用户进行探究式学习，在识别网络威胁的同时提供透明的推理解释，从而革新网络安全教育方式。

    

    现代网络威胁日益复杂的态势要求安全教育采用超越传统教学方法的创新方式。传统的培训范式往往无法有意义地吸引学习者，也难以培养有效识别威胁所需的直觉推理能力。本文介绍了一个交互式教育框架，通过结构化猜谜游戏（类似“20个问题”游戏）的视角重新构想网络安全意识教育。我们的方法将可解释人工智能（XAI）原理与强化学习相结合，创造了一个动态学习环境，用户通过引导式探究来发现网络安全概念。所提出的系统采用基于策略的强化学习智能体，扮演知识渊博的提问者角色，系统地缩小用户所描述的安全场景的范围，直到它既能识别出潜在的威胁，又能为其提供透明的推理依据

    arXiv:2604.26964v2 Announce Type: replace-cross  Abstract: The escalating complexity of modern cyber threats demands innovative approaches to security education that transcend traditional pedagogical methods. Conventional training paradigms often fail to engage learners meaningfully or develop the intuitive reasoning necessary for effective threat recognition. This paper introduces an interactive educational framework that reimagines cybersecurity awareness through the lens of a structured guessing game. Our approach integrates explainable artificial intelligence (XAI) principles with reinforcement learning to create a dynamic learning environment where users discover cybersecurity concepts through guided inquiry. The proposed system employs a policy-based reinforcement learning agent that assumes the role of a knowledgeable questioner, systematically narrowing down user-described security scenarios until it can both identify the underlying threat and provide transparent reasoning for 
    
[^208]: CoGReV：一种面向钓鱼网站分类的置信度门控后验非单调信念修正框架

    CoGReV: A Confidence-Gated Post-Hoc Non-Monotonic Belief Revision Framework for Phishing Website Classification

    [https://arxiv.org/abs/2604.25512](https://arxiv.org/abs/2604.25512)

    CoGReV通过置信度门控的非单调推理层，仅在分类器置信度低且网站元数据可用时才将钓鱼预测修正为合法预测，从而在不损失召回率的前提下有效减少误报。

    

    在钓鱼检测中，机器学习分类器扮演着第一道防线的角色，但它们产生的误报需要由人类分析师进行分诊处理。过多的虚假警报会导致警报疲劳，从而削弱人类的监督能力。我们提出了CoGReV，这是一个混合框架，它通过一个以答案集编程实现的后验非单调推理层来增强标准的机器学习分类器。该推理层应用了一条置信度门控的可废止规则：仅当网站元数据存在且分类器的决策属于低置信度时，才将钓鱼预测修正为合法预测，从而将不确定的预测交由推理层处理，而将高置信度的决策留给分类器。这种门控机制充当了分类器与推理层之间的功能分配机制。与非门控的规则不同——后者只能通过丢弃真实检测结果来减少误报，并使钓鱼召回率下降约九个百分点——

    arXiv:2604.25512v3 Announce Type: replace  Abstract: In phishing detection, machine learning classifiers act as a first line of defense, but the false positives they produce are triaged by human analysts. The excessive false alarms cause alert fatigue that erodes human oversight. We propose CoGReV, a hybrid framework that augments standard machine learning classifiers with a post-hoc non-monotonic reasoning layer implemented in Answer Set Programming. The layer applies a confidence-gated defeasible rule that revises a phishing prediction toward legitimate only when website metadata is present and the classifier's decision is low-confidence, deferring uncertain predictions to the reasoning layer while leaving out confident decisions to the classifiers. This gating acts as a function-allocation mechanism between the classifier and the reasoning layer. Unlike an ungated rule, which reduces false positives only by discarding genuine detections and degrades phishing recall by about nine per
    
[^209]: 非平稳性破坏多智能体强化学习中的置换代理检验：诊断与对策

    Non-Stationarity Breaks Permutation Surrogates in Multi-Agent Reinforcement Learning: Diagnosis and Remedies

    [https://arxiv.org/abs/2604.23716](https://arxiv.org/abs/2604.23716)

    该研究发现非平稳性会使基于置换代理的信息论影响检验在多智能体强化学习中产生接近100%的假阳性，并提出通过改变零模型而非仅排除训练瞬态阶段来修复这一缺陷。

    

    arXiv:2604.23716v4 公告类型：替换。摘要：信息论度量的报告指南很少对照真实基准进行检验。我们在两个多智能体强化学习游戏（一个社会困境博弈和一个协调竞赛）中检验了一项防护措施，其中所选智能体对之间的有向影响在构造上为零，实验跨越100个随机种子。若省略一个前提条件——排除非平稳的训练瞬态阶段——将产生100.00%和99.95%的假阳性率：那些在从未相遇过的运行中独立退火探索的智能体会被判定为相互影响。排除瞬态阶段后，社会困境中的假阳性率降至3.0%，但协调游戏中仍高达11.8%，平稳性检验解释了这一差异：社会困境中95.7%的序列在之后是平稳的，而协调游戏序列中仅有56.8%是平稳的。因此，非平稳性必须得到处理，而简单的排除方法既不是唯一的途径，也并不充分。我们推荐的做法是改变零模型而非修改数据：对源序列进行置换……

    arXiv:2604.23716v4 Announce Type: replace  Abstract: Reporting guidance for information-theoretic measures is rarely tested against ground truth. We test one guardrail in two multi-agent reinforcement learning games, a social dilemma and a coordination race, where directed influence between selected agent pairs is zero by construction, over 100 seeds. Omitting one precondition, exclusion of the non-stationary training transient, gives false-positive rates of 100.00% and 99.95%: agents annealing exploration independently, in runs that never met, are flagged as influencing one another. Excluding the transient reaches 3.0% in the social dilemma but 11.8% in the coordination game, which stationarity tests explain: 95.7% of social-dilemma series are stationary afterwards against 56.8% of coordination series. So the non-stationarity must be treated, and exclusion is neither the only way nor sufficient. What we recommend instead changes the null model rather than the data: permuting the sourc
    
[^210]: 具身人工智能最大的风险是治理滞后

    The Biggest Risk of Embodied AI is Governance Lag

    [https://arxiv.org/abs/2604.21938](https://arxiv.org/abs/2604.21938)

    本文认为具身人工智能最大的风险并非就业替代，而是治理滞后——即技术部署变化与制度响应之间的时间与能力差距，并提出通过部署可见性、技术栈问责、触发式调整和自动分配响应来构建可观察、可响应、可适应的治理架构。

    

    具身人工智能（Embodied AI）常被广泛讨论为就业替代问题。然而，更深层的风险是治理滞后：即技术部署发生可测量变化与能够应对其后果的制度性响应之间的时间和能力差距。基于已有的“步调问题”和“科林格里奇困境”，本文论证具身人工智能通过可扩展的模型与平台、任务层面的重组，以及上游技术控制与下游社会影响的分离，加剧了这一差距。我们区分了三种相互强化的滞后形式——观察性滞后、制度性滞后和分配性滞后，并提出了一种基于部署可见性、技术栈层面问责、基于触发条件的调整以及自动分配性响应的合规架构。核心政策挑战不仅在于自动化本身，而在于治理系统能否在……之前变得可观察、可响应且可适应。

    arXiv:2604.21938v2 Announce Type: replace-cross  Abstract: Embodied AI is widely discussed as a job-displacement problem. The deeper risk, however, is governance lag: the time and capability gap between a measurable change in technology deployment and an institutional response able to address its consequences. Building on the established pacing problem and the Collingridge dilemma, this article argues that embodied AI intensifies that gap through scalable models and platforms, task-level reorganization, and the separation of upstream technological control from downstream social impact. We distinguish three mutually reinforcing forms of lag, observational, institutional, and distributive, and propose a compliance architecture based on deployment visibility, stack-level accountability, trigger-based adjustment, and automatic distributional response. The central policy challenge is not automation alone, but whether governance systems can become observable, responsive, and adaptive before 
    
[^211]: 心智在哪里？人格向量与大语言模型的个体化问题

    Where is the Mind? Persona Vectors and LLM Individuation

    [https://arxiv.org/abs/2604.17031](https://arxiv.org/abs/2604.17031)

    本文通过机制可解释性研究大语言模型的个体化问题，提出并论证了虚拟实例观点以及两种新观点（实例-人格观点和模型-人格观点）作为认定LLM心智的最有力候选方案。

    

    大语言模型的个体化问题探讨的是：与大语言模型相关联的哪些实体（如果有的话）应当被认定为心智。我们通过机制可解释性来研究这一问题，特别是结合了关于人格向量、人格空间和涌现性失调的最新实证工作。我们认为有三种观点是最有力的候选方案：虚拟实例观点，以及我们引入的两种新观点——（虚拟）实例-人格观点和模型-人格观点。首先，我们论证了虚拟实例观点，其理由是注意力流在词元时间维度上维持着准心理学连接。随后，我们围绕关于大语言模型中人格内在结构的三种假说，梳理了人格相关文献，并表明这两种基于人格的观点是有前景的替代方案。

    arXiv:2604.17031v3 Announce Type: replace  Abstract: The individuation problem for large language models asks which entities associated with them, if any, should be identified as minds. We approach this problem through mechanistic interpretability, engaging in particular with recent empirical work on persona vectors, persona space, and emergent misalignment. We argue that three views are the strongest candidates: the virtual instance view and two new views we introduce, the (virtual) instance-persona view and the model-persona view. First, we argue for the virtual instance view on the grounds that attention streams sustain quasi-psychological connections across token-time. Then we present the persona literature, organised around three hypotheses about the internal structure underlying personas in LLMs, and show that the two persona-based views are promising alternatives.
    
[^212]: 价值模型回归：用于大语言模型强化学习中价值建模的生成式批评家

    Bringing Value Models Back: Generative Critics for Value Modeling in LLM Reinforcement Learning

    [https://arxiv.org/abs/2604.10701](https://arxiv.org/abs/2604.10701)

    该论文提出生成式Actor-Critic（GenAC），用先进行思维链推理再输出价值的生成式批评家替代传统单次标量价值预测，从而解决了大语言模型强化学习中价值模型因表达能力受限而难以可靠训练的问题。

    

    信用分配是强化学习中的核心挑战。经典的Actor-Critic方法通过基于学习到的价值函数进行细粒度的优势估计来应对这一挑战。然而，在现代大语言模型强化学习中，人们通常避免使用学习到的价值模型，因为传统的判别式批评家难以可靠地训练。我们重新审视了价值建模，并认为这种困难部分源于表达能力受限。具体而言，表示复杂性理论表明，在现有价值模型所采用的单次预测范式下，价值函数可能难以逼近，且我们的缩放实验表明，此类批评家无法随模型规模扩大而可靠地改进。基于这一观察，我们提出了生成式Actor-Critic（GenAC），它用一种生成式批评家取代了单次标量价值预测，该批评家在产生价值之前先进行思维链推理。

    arXiv:2604.10701v2 Announce Type: replace-cross  Abstract: Credit assignment is a central challenge in reinforcement learning (RL). Classical actor-critic methods address this challenge through fine-grained advantage estimation based on a learned value function. However, learned value models are often avoided in modern large language model (LLM) RL because conventional discriminative critics are difficult to train reliably. We revisit value modeling and argue that this difficulty is partly due to limited expressiveness. In particular, representation complexity theory suggests that value functions can be hard to approximate under the one-shot prediction paradigm used by existing value models, and our scaling experiments show that such critics do not improve reliably with scale. Motivated by this observation, we propose Generative Actor-Critic (GenAC), which replaces one-shot scalar value prediction with a generative critic that performs chain-of-thought reasoning before producing a valu
    
[^213]: 零样本世界模型是发展高效的学习者

    Zero-shot World Models Are Developmentally Efficient Learners

    [https://arxiv.org/abs/2604.10333](https://arxiv.org/abs/2604.10333)

    本文提出零样本世界模型（ZWM），基于稀疏时间因子化预测器、近似因果推断和推断组合三大原则，仅从单个儿童的第一人称经验中即可快速习得多种物理场景理解能力，为儿童数据高效且灵活的认知能力提供了新颖的计算解释。

    

    幼儿很早就表现出理解物理世界的能力，能够估计深度、运动、物体连贯性、交互作用以及物理场景理解的许多其他方面。儿童是既节省数据又灵活的认知系统，在训练数据极其有限的情况下便能形成能力，并能泛化到大量未训练过的任务——这对当今最先进的AI系统而言仍是一个重大挑战。在此，我们针对这些能力提出了一种新颖的计算假说——零样本世界模型（ZWM）。ZWM基于三个原则：一个将外观与动态解耦的稀疏时间因子化预测器；通过近似因果推断实现零样本估计；以及通过推断的组合来构建更复杂的能力。我们表明，ZWM可以从单个儿童的第一人称经验中学习，并在多个物理理解基准测试中快速形成能力。它还显示出进展

    arXiv:2604.10333v2 Announce Type: replace  Abstract: Young children demonstrate early abilities to understand their physical world, estimating depth, motion, object coherence, interactions, and many other aspects of physical scene understanding. Children are both data-efficient and flexible cognitive systems, creating competence despite extremely limited training data, while generalizing to myriad untrained tasks -- a major challenge even for today's best AI systems. Here we introduce a novel computational hypothesis for these abilities, the Zero-shot World Model (ZWM). ZWM is based on three principles: a sparse temporally-factored predictor that decouples appearance from dynamics; zero-shot estimation through approximate causal inference; and composition of inferences to build more complex abilities. We show that ZWM can be learned from the first-person experience of a single child, rapidly generating competence across multiple physical understanding benchmarks. It also shows progress
    
[^214]: MAVEN-T：面向实时多智能体轨迹预测的强化异构蒸馏方法

    MAVEN-T: Reinforced Heterogeneous Distillation for Real-Time Multi-Agent Trajectory Prediction

    [https://arxiv.org/abs/2604.10169](https://arxiv.org/abs/2604.10169)

    提出MAVEN-T框架，通过强化异构蒸馏将高容量教师模型的知识高效转移至轻量级学生模型，并显式修正安全相关偏差，实现实时多智能体轨迹预测的准确性与效率平衡。

    

    轨迹预测是自动驾驶系统中的关键组成部分，因为未来运动直接影响碰撞检测、行为规划和控制。在密集交互、异构行为、多模态未来以及有限的板载计算条件下，该任务仍然具有挑战性。现有的图、注意力和生成式预测器改善了交互推理或不确定性建模，但其高容量设计往往对实时部署成本过高。轻量级预测器和传统蒸馏降低了推理成本，但通常依赖静态模仿，并未明确修正与安全相关的教师偏差。本文提出了MAVEN-T，一种用于实时多智能体轨迹预测的强化异构蒸馏框架。高容量教师模型利用环绕感知图编码器建模定向局部交互，将高效时间滤波与移位窗口空间处理相结合。

    arXiv:2604.10169v4 Announce Type: replace  Abstract: Trajectory prediction is a key component of autonomous driving systems because future motions directly affect collision checking, behavior planning, and control. The task remains challenging under dense interactions, heterogeneous behaviors, multimodal futures, and limited on-board computation. Existing graph, attention, and generative predictors improve interaction reasoning or uncertainty modeling, but their high-capacity designs are often costly for real-time deployment. Lightweight predictors and conventional distillation reduce inference cost, yet usually rely on static imitation and do not explicitly correct safety-relevant teacher bias. This paper proposes \textbf{MAVEN-T}, a reinforced heterogeneous distillation framework for real-time multi-agent trajectory prediction. A high-capacity teacher models directed local interactions with a surround-aware graph encoder, combines efficient temporal filtering with shifted-window spat
    
[^215]: Spec-Harness：度量和改进LLM合成形式规范的行为充分性

    Spec-Harness: Measuring and Improving Behavioral Adequacy of LLM-Synthesized Formal Specifications

    [https://arxiv.org/abs/2604.00280](https://arxiv.org/abs/2604.00280)

    该论文指出验证器通过率高并不代表LLM合成的JML形式规范有意义，并提出Spec-Harness框架来度量和改进这类规范对代码行为的实际捕获程度（行为充分性）。

    

    形式规范在确保软件可靠性方面发挥着核心作用，然而自动合成高质量的规范仍然困难重重，且往往需要领域专业知识。最近的研究将大型语言模型应用于生成Java建模语言（JML）规范，并报告了很高的验证器通过率。但通过验证器只能确认实现与规范相一致，并不能说明规范本身是有意义的。例如一个平凡的后置条件 ensures true 可以通过任何验证器，却对代码没有任何描述。那么，一个被验证器接受的规范实际上捕获了多少代码行为呢？在这项工作中，我们首先在统一的实验设置下比较了经典的与基于提示的JML合成方法，发现通过验证反馈进行提示优化虽然能提高通过率，但会达到明显的天花板。随后，我们引入了Spec-Harness，一个用于度量行为（充分性）的框架……（摘要在此处截断）

    arXiv:2604.00280v2 Announce Type: replace  Abstract: Formal specifications play a central role in ensuring software reliability, yet automatically synthesizing high-quality specifications remains difficult and often requires domain expertise. Recent work has applied large language models to generate specifications in the Java Modeling Language (JML), reporting high verifier pass rates. But passing a verifier only confirms that an implementation is consistent with a specification, not that the specification is meaningful. A trivial postcondition such as ensures true satisfies any verifier while saying nothing about the code. How much behavior, then, does a verifier-accepted specification actually capture? In this work, we first compare classical and prompt-based JML synthesis approaches under a unified setup, and find that prompt optimization through verification feedback raises pass rates but reaches a clear ceiling. We then introduce Spec-Harness, a framework that measures the behavio
    
[^216]: RelayS2S：面向实时对话的双路径推测生成方法

    RelayS2S: A Dual-Path Speculative Generation for Real-Time Dialogue

    [https://arxiv.org/abs/2603.23346](https://arxiv.org/abs/2603.23346)

    RelayS2S提出了一种双路径混合架构，由双工S2S模型快速推测生成响应前缀以保证低延迟，再由级联ASR-LLM流水线续写高质量内容，并通过轻量验证器控制交接，兼得实时语音对话的低延迟与高质量。

    

    实时语音对话系统面临延迟与响应质量之间的根本性矛盾。端到端语音到语音（S2S）模型能够即时响应，并自然地处理话轮转换、附和反馈与打断，但其输出在语义上较弱；而级联流水线（ASR -> LLM）虽能提供更高质量的响应，代价却是随模型规模增长的延迟。我们提出了RelayS2S，一种混合架构，在检测到话轮切换时并行运行两条路径。快速路径——一个双工S2S模型——推测性地起草一段简短的响应前缀，立即流式传输给TTS以实现低延迟的响应起始，同时继续监听实时音频事件。慢速路径——级联的ASR -> LLM流水线——以已提交的前缀为条件生成更高质量的续写内容，从而产生不间断的完整话语。一个轻量级的可学习验证器负责控制两者之间的交接，在适当时机提交前缀，否则进行优雅的回退。

    arXiv:2603.23346v2 Announce Type: replace  Abstract: Real-time spoken dialogue systems face a fundamental tension between latency and response quality. End-to-end speech-to-speech (S2S) models respond immediately and naturally handle turn-taking, backchanneling, and interruption, but produce semantically weaker outputs. Cascaded pipelines (ASR -> LLM) deliver stronger responses at the cost of latency that grows with model size. We present RelayS2S, a hybrid architecture that runs two paths in parallel upon turn detection. The fast path - a duplex S2S model - speculatively drafts a short response prefix that is streamed immediately to TTS for low-latency response onset, while continuing to monitor live audio events. The slow path - a cascaded ASR -> LLM pipeline - generates a higher-quality continuation conditioned on the committed prefix, producing an uninterrupted utterance. A lightweight learned verifier gates the handoff, committing the prefix when appropriate or falling back gracef
    
[^217]: 人机系统中的认知放大与认知委托：一个度量框架

    Cognitive Amplification vs Cognitive Delegation in Human-AI Systems: A Metric Framework

    [https://arxiv.org/abs/2603.18677](https://arxiv.org/abs/2603.18677)

    本文提出了一个包含四个量化指标（认知放大指数、依赖比率、人类依赖指数、认知漂移率）的度量框架，用以区分人机系统中的“认知放大”与“认知委托”，并通过基于智能体的仿真验证了该框架识别正向协作增益是否可恢复的有效性。

    

    人工智能日益深入地嵌入人类决策过程，但如何区分真正放大人类认知的系统与助长过度依赖的系统，仍然缺乏清晰界定。本文提出了一个框架，用于区分“认知放大”（在不削弱人类能力的前提下提升混合系统表现）与“认知委托”（将推理外包给人工智能）。我们定义了四个度量指标：认知放大指数（CAI*）、依赖比率（D）、人类依赖指数（HRI）和人类认知漂移率（HCDR）。我们在NetLogo基于智能体的仿真中，针对三种依赖情境和多种依赖-能力萎缩配置对该框架进行了测试，通过约束优化和参数扫描来判断正向协作增益是否可以恢复。最后，我们引入了一个包含显式人机交互项的扩展模型。我们的度量指标能够有效区分退化行为……（原文摘要在此截断）

    arXiv:2603.18677v4 Announce Type: replace-cross  Abstract: Artificial intelligence is increasingly embedded in human decision-making, yet distinguishing systems that genuinely amplify human cognition from those promoting excessive dependence remains underdefined. This paper introduces a framework to distinguish cognitive amplification (improving hybrid performance without degrading human capability) from cognitive delegation (outsourcing reasoning to the AI).   We define four metrics: the Cognitive Amplification Index (CAI*), Dependency Ratio (D), Human Reliance Index (HRI), and Human Cognitive Drift Rate (HCDR). We test this framework in an agent-based NetLogo simulation across three reliance regimes and multiple dependency-atrophy configurations, performing constrained optimizations and parameter sweeps to determine if positive collaborative gain is recoverable. Finally, we introduce an extension with an explicit human-AI interaction term.   Our metrics effectively distinguish degene
    
[^218]: MOSAIC：用于跨范式智能体混合与人机协作的通用智能体级接口

    MOSAIC: A Universal Agent-Level Interface for Cross-Paradigm Agent Mixing and Human-AI Collaboration

    [https://arxiv.org/abs/2603.01260](https://arxiv.org/abs/2603.01260)

    MOSAIC是一个开源平台，通过基于IPC的工作器协议和统一的操作员抽象接口，使强化学习策略、大语言模型、视觉语言模型和人类操作员等异构智能体能够在同一强化学习环境中协作并实现公平的跨范式比较。

    

    现有基础设施无法在同一环境中部署来自不同决策范式的智能体，导致无法在相同条件下进行公平的跨范式比较。我们提出了MOSAIC，一个开源平台，使异构智能体（强化学习策略、大语言模型、视觉语言模型和人类操作员）能够在共享的强化学习环境中以临时团队设置进行协作行动，并保证结果可复现。MOSAIC引入了三项贡献：(i) 基于IPC的工作器协议，将原生和第三方框架封装为隔离的子进程工作器，每个工作器无需修改即可执行自身的训练和推理逻辑，并通过版本化的进程间协议进行通信；(ii) 操作员抽象，通过将工作器映射到智能体槽位来形成智能体级接口：每个操作员，无论其背后由强化学习策略、大语言模型还是人类支撑，都遵循一个最小的通用接口；(iii) 一个确定性……（摘要原文在此处截断）

    arXiv:2603.01260v3 Announce Type: replace-cross  Abstract: Existing infrastructure cannot deploy agents from different decision-making paradigms within the same environment, making fair cross-paradigm comparison under identical conditions impossible. We present MOSAIC, an open-source platform that enables heterogeneous agents (RL policies, LLMs, VLMs, and human operators) to act within shared reinforcement learning environments in ad-hoc team settings with reproducible results. MOSAIC introduces three contributions. (i) IPC-based worker protocol that wraps native and third-party frameworks as isolated subprocess workers, each executing its own training and inference logic unmodified and communicating through a versioned inter-process protocol. (ii) An operator abstraction that forms an agent-level interface by mapping workers to agent slots: each operator, regardless of whether it is backed by an RL policy, an LLM, or a human, conforms to a minimal universal interface. (iii) A determin
    
[^219]: 城市编辑：面向依赖感知城市地理空间修改的分层智能体执行框架

    City Editing: Hierarchical Agentic Execution for Dependency-Aware Urban Geospatial Modification

    [https://arxiv.org/abs/2602.19326](https://arxiv.org/abs/2602.19326)

    提出CEAE分层智能体框架，将自然语言指令分解为分层几何意图并转化为机器可执行的GeoJSON编辑，通过自我反思的执行-验证循环实现依赖感知且空间一致的城市地理空间增量修改。

    

    城市更新需要对现有地理空间规划进行增量修改，然而在空间约束下手动更新复杂布局既费时费力又容易出错。为解决这一问题，我们提出了CEAE，一个分层智能体框架，它将城市更新形式化为从自然语言指令到机器可执行GeoJSON编辑的转换。CEAE将指令分解为分层几何意图，从粗到细地执行编辑，并通过自我反思的执行-验证循环保持空间一致性。实验结果表明，CEAE在执行有效性、鲁棒性和几何精度方面均优于基线方法。

    arXiv:2602.19326v3 Announce Type: replace-cross  Abstract: Urban renewal requires incremental modifications to existing geospatial plans, yet manually updating complex layouts under spatial constraints is labor-intensive and error-prone. To tackle this, we propose CEAE, a hierarchical agentic framework that formulates urban renewal as machine-executable GeoJSON editing from natural-language instructions. CEAE decomposes instructions into hierarchical geometric intents, executing edits from coarse to fine while preserving spatial consistency through a self-reflective execution-validation loop. Experimental results show that CEAE outperforms baselines in execution validity, robustness, and geometric accuracy.
    
[^220]: 英国多语言英语使用者在AI驱动的语音认知筛查中的假阳性偏差

    False positive bias in AI-powered speech-based cognitive screening for multilingual English speakers in the UK

    [https://arxiv.org/abs/2602.13047](https://arxiv.org/abs/2602.13047)

    该研究通过对1,395名参与者、超过263小时语音数据的分析，首次发现尽管语音识别准确率在各语言群体间无显著差异，但AI认知筛查的下游模型对英国多语言英语使用者存在系统性的假阳性偏差，凸显了认知筛查公平性评估的重要性。

    

    会话语音能够揭示认知衰退的早期迹象，包括痴呆症和轻度认知障碍（MCI）。AI模型在基于语音的筛查方面展现出前景，但大多数研究聚焦于单语群体。在英国，痴呆症预计在黑人和亚裔社区中增长最快，而这些社区中多语言现象普遍，因此公平性评估至关重要。我们招募了1,395名参与者（包括谢菲尔德/布拉德福德的英语单语者和多语者），并通过CognoMemory智能体收集了超过263小时的语音数据。多语言参与者在说英语的同时还使用索马里语、中文或南亚语言（印地语、乌尔都语、旁遮普语、米尔普里语、阿拉伯语）。我们评估了自动语音识别系统（Whisper、Wav2Vec 2.0、NeMo）以及用于认知分类和MMSE回归的下游AI模型。ASR准确率在各群体间未显示出显著差异。然而，下游模型表现出系统性差异：多语言使用者……

    arXiv:2602.13047v2 Announce Type: replace  Abstract: Conversational speech reveals early signs of cognitive decline, including dementia and mild cognitive impairment (MCI). AI models show promise for speech-based screening, yet most research focuses on monolingual groups. In the UK, dementia is projected to rise fastest among Black and Asian communities, where multilingualism is common, making equity assessment critical. We recruited 1,395 participants (monolingual English speakers and multilingual speakers from Sheffield/Bradford) and collected over 263 hours of speech via the CognoMemory agent. Multilingual participants spoke English alongside Somali, Chinese, or South Asian languages (Hindi, Urdu, Punjabi, Mirpuri, Arabic). We evaluated ASR (Whisper, Wav2Vec 2.0, NeMo) and downstream AI models for cognitive classification and MMSE regression. ASR accuracy showed no significant differences across groups. However, downstream models exhibited systematic disparities: multilingual speake
    
[^221]: 重新审视Transformer语言模型的形状惯例

    Revisiting the Shape Convention of Transformer Language Models

    [https://arxiv.org/abs/2602.06471](https://arxiv.org/abs/2602.06471)

    该论文提出沙漏Transformer架构，用残差沙漏形MLP堆叠替代传统窄-宽-窄FFN并通过沙漏注意力解耦残差流与注意力宽度，在113M至8B参数规模上以更少层数和更宽隐藏状态实现了与传统Transformer相当的性能，同时提高了训练计算效率。

    

    稠密Transformer的架构形状一直保持着出奇地稳定：窄-宽-窄的前馈网络（FFN）消耗了大部分非嵌入参数。基于残差宽-窄-宽（沙漏形）MLP即使存在瓶颈仍保持表达能力的理论与实证证据，我们重新审视这种架构惯例对稠密语言模型是否必要。我们研究了沙漏Transformer，它用残差堆叠的沙漏子MLP取代传统FFN，并使用沙漏注意力将残差流宽度与注意力宽度解耦。这揭示了一种实用的深度-宽度权衡：在匹配参数预算下，压缩FFN中间维度可以采用更宽的隐藏状态和更少的层数。在113M到8B参数的各模型规模上，沙漏Transformer实现了与传统Transformer相当的语言建模和下游性能，同时提升了训练计算效率。

    arXiv:2602.06471v2 Announce Type: replace  Abstract: The architectural shape of dense Transformers has remained remarkably stable: narrow-wide-narrow feed-forward networks (FFNs) consume most non-embedding parameters. Motivated by theoretical and empirical evidences that residual wide-narrow-wide (hourglass) MLPs remain expressive despite bottlenecks, we revisit whether this architectural convention is necessary for dense language models. We study Hourglass Transformers, which replace the conventional FFN with residual stacks of hourglass sub-MLPs and use hourglass attention to decouple residual-stream width from attention width. This exposes a practical depth-width trade-off: compressing the FFN intermediate dimension allows wider hidden states and fewer layers at matched parameter budgets. Across model scales from 113M to 8B parameters, Hourglass Transformers achieve language-modeling and downstream performance comparable to conventional Transformers, while improving training compute
    
[^222]: 基于软体机器人的触觉记忆：通过掩码编码与软腕实现鲁棒的物体插入

    Tactile Memory with Soft Robot: Robust Object Insertion via Masked Encoding and Soft Wrist

    [https://arxiv.org/abs/2601.19275](https://arxiv.org/abs/2601.19275)

    该论文提出TaMeSo-bot软体机器人系统，将软腕与基于触觉检索的触觉记忆控制相结合，利用掩码触觉轨迹Transformer（MAT³）联合建模多模态感官信息的时空交互，实现不确定性下安全鲁棒的物体插入操作。

    

    触觉记忆，即存储和检索基于触觉经验的能力，对于钥匙插入等不确定性下的富接触任务至关重要。为了复现这种能力，我们提出了基于软体机器人的触觉记忆系统TaMeSo-bot，该系统将软腕与基于触觉检索的控制相结合，以实现安全且鲁棒的操作。软腕使得在数据采集过程中能够进行安全的接触探索，而触觉记忆通过检索复用过去的演示，从而灵活地适应未见过的场景。该系统的核心是掩码触觉轨迹Transformer（MAT³），它联合建模机器人动作、分布式触觉线索、力/力矩测量和本体感觉信号之间的时空交互。通过掩码词元预测，MAT³通过从上下文推断缺失的感官信息来学习丰富的时空表征，自主提取……

    arXiv:2601.19275v2 Announce Type: replace-cross  Abstract: Tactile memory, the ability to store and retrieve touch-based experience, is critical for contact-rich tasks such as key insertion under uncertainty. To replicate this capability, we introduce Tactile Memory with Soft Robot (TaMeSo-bot), a system that integrates a soft wrist with tactile retrieval-based control to enable safe and robust manipulation. The soft wrist allows safe contact exploration during data collection, while tactile memory reuses past demonstrations via retrieval for flexible adaptation to unseen scenarios. The core of this system is the Masked Tactile Trajectory Transformer (MAT$^\text{3}$), which jointly models spatiotemporal interactions between robot actions, distributed tactile cues, force-torque measurements, and proprioceptive signals. Through masked token prediction, MAT$^\text{3}$ learns rich spatiotemporal representations by inferring missing sensory information from context, autonomously extracting 
    
[^223]: 超越满秩动作与状态可观测性的POMDP学习研究

    Toward Learning POMDPs Beyond Full-Rank Actions and State Observability

    [https://arxiv.org/abs/2601.18930](https://arxiv.org/abs/2601.18930)

    本文提出了一种在比传统满秩动作与状态可观测性假设更温和的秩条件下学习离散POMDP参数的方法，克服了预测状态表示（PSR）等谱方法缺乏显式转移和观测系统模型、无法灵活用于不同规划问题的局限。

    

    我们致力于使自主智能体能够学习和推理具有隐藏状态的系统，例如锁定机制。我们将此问题转化为学习离散部分可观测马尔可夫决策过程（POMDP）的参数。智能体从已知的POMDP动作空间和观测空间开始，但并不了解其状态空间、转移模型或观测模型，这些属性必须从一系列动作和观测序列中构建出来。学习部分可观测领域模型的谱方法，例如预测状态表示（PSR），能够学习到足以预测未来结果的状态表示。然而，PSR模型缺乏显式的转移和观测系统模型，因而无法与不同的奖励函数结合来解决不同的规划问题。在关于转移矩阵与观测矩阵乘积的一组温和的秩假设条件下，我们（摘要在此处被截断）

    arXiv:2601.18930v5 Announce Type: replace-cross  Abstract: We are interested in enabling autonomous agents to learn and reason about systems with hidden states, such as locking mechanisms. We cast this problem as learning the parameters of a discrete Partially Observable Markov Decision Process (POMDP). The agent begins with knowledge of the POMDP's actions and observation spaces, but not its state space, transitions, or observation models. These properties must be constructed from a sequence of actions and observations. Spectral approaches to learning models of partially observable domains, such as Predictive State Representations (PSRs), learn representations of state that are sufficient to predict future outcomes. PSR models, however, do not have explicit transition and observation system models that can be used with different reward functions to solve different planning problems. Under a mild set of rankness assumptions on the products of transition and observation matrices, we sho
    
[^224]: Elsewise：通过可能性空间可视化创作开放式交互叙事

    Elsewise: Authoring Open-ended Interactive Narrative with Possibility Space Visualization

    [https://arxiv.org/abs/2601.15295](https://arxiv.org/abs/2601.15295)

    本文提出了Elsewise——一个面向大语言模型交互叙事的创作工具，通过新颖的“捆绑故事线”概念与可能性空间可视化，帮助创作者感知和理解叙事可能性空间，从而弥合创作者构想与玩家实际体验之间的差距。

    

    交互叙事（IN）创作者为玩家构建包含分歧性叙事可能的空间供其探索，玩家的输入决定了他们实际体验到哪些叙事可能性。生成式AI能够通过对预先创作的内容进行即兴扩展来响应玩家的开放式输入，从而实现新形式的交互叙事。然而，这种外推扩展可能会扩大创作者构想的故事与玩家实际体验的故事之间的差距，潜在地限制情节推进的力度以及创作者叙事意图的传达。为了弥合这一差距，我们推出了Elsewise：一个面向基于大语言模型（LLM）交互叙事的创作工具，它实现了一种新颖的“捆绑故事线”概念，以增强创作者对叙事可能性空间的感知和理解，使创作者能够从开放式的、用户可配置的叙事维度出发，探索其交互叙事作品各种可能游玩路径之间的相似性与差异性。

    arXiv:2601.15295v2 Announce Type: replace-cross  Abstract: Interactive narrative (IN) authors craft spaces of divergent narrative possibilities for players to explore, with the player's input determining which narrative possibilities they actually experience. Generative AI can enable new forms of IN by improvisationally expanding on pre-authored content in response to open-ended player input. However, this extrapolation risks widening the gap between author-envisioned and player-experienced stories, potentially limiting the strength of plot progression and the communication of the author's narrative intent. To bridge the gap, we introduce Elsewise: an authoring tool for LLM-based INs that implements a novel Bundled Storyline concept to enhance author's perception and understanding of the narrative possibility space, allowing authors to explore similarities and differences between possible playthroughs of their IN in terms of open-ended, user-configurable narrative dimensions. A user st
    
[^225]: 从评分标准到可靠分数：基于证据的LLM评判文本评估

    From Rubrics to Reliable Scores: Evidence-Grounded Text Evaluation with LLM Judges

    [https://arxiv.org/abs/2601.08654](https://arxiv.org/abs/2601.08654)

    提出Rulers框架，通过锁定任务级评分标准、执行基于证据的结构化判断，并将信号校准到人类分数边界，实现与人类评分更一致、更稳定且可审计的LLM文本评估。

    

    基于评分标准的文本评估越来越依赖大型语言模型（LLM）作为可扩展的评判者，然而固定的黑盒模型可能对相同标准产生不一致的解读，产生难以审计的分数归因，并且难以将判断准确映射到人类评分量表上。我们将这一挑战定义为“标准迁移”（criteria transfer）：即将人类评分标准的意图转化为稳定、可审计的推理时评分协议。我们提出了Rulers，它锁定任务级评分标准规范，通过结构化的、基于证据的判断来执行该标准，并将产生的信号校准到人类分数边界。在四个由评分标准管理的基准测试和多个固定骨干模型上，Rulers在大多数评估设置中与人类分数实现了更强的一致性，同时更好地匹配经验分数分布，并在语义等价的评分标准扰动下保持更高的稳定性。校准控制和组件消融实验……

    arXiv:2601.08654v3 Announce Type: replace  Abstract: Rubric-based text evaluation increasingly relies on large language models (LLMs) as scalable judges, yet frozen black-box models can interpret the same criteria inconsistently, produce score attributions that are difficult to audit, and map judgments poorly onto human scoring scales. We define this challenge as criteria transfer: translating human rubric intent into a stable, auditable inference-time scoring protocol. We introduce Rulers, which locks a task-level rubric specification, executes it through structured, evidence-grounded judgments, and calibrates the resulting signals to human score boundaries. Across four rubric-governed benchmarks and multiple frozen backbone models, Rulers achieves stronger agreement with human scores in most evaluated settings, while better matching empirical score distributions and remaining more stable under semantically equivalent rubric perturbations. Calibration controls and component ablations 
    
[^226]: 基于贝叶斯线性任务模型的元强化学习

    Meta-RL with Bayesian Linear Task Models

    [https://arxiv.org/abs/2512.20974](https://arxiv.org/abs/2512.20974)

    提出GLiBRL深度贝叶斯强化学习框架，通过结合广义线性任务模型与可学习非线性基函数，实现精确的共轭贝叶斯后验更新和闭式边缘似然，消除了变分推断带来的近似误差，并可灵活兼容离策略与在线策略算法。

    

    深度贝叶斯强化学习通过推断潜在的转移和奖励模型来适应未见过的任务，但现有方法通常依赖于变分后验和证据下界，这会引入近似误差和不稳定的任务表示。我们提出了GLiBRL，这是一个将广义线性任务模型与可学习非线性基函数相结合的深度贝叶斯强化学习框架。GLiBRL具有共轭贝叶斯推断的特性，能够对任务参数和模型噪声进行精确的顺序后验更新，同时提供闭式边缘似然，从而完全消除了变分推断。该更新天然具有置换不变性，使GLiBRL能够与离策略和在线策略算法集成。GLiBRL还学习了具有精确核恒等关系的任务表示，将任务表示之间的距离与任务上下文上的核差异联系起来。与八个代表性方法相比……

    arXiv:2512.20974v4 Announce Type: replace-cross  Abstract: Deep Bayesian reinforcement learning adapts to unseen tasks by inferring latent transition and reward models, but existing methods typically rely on variational posteriors and evidence lower bounds, introducing approximation error and unstable task representations. We introduce GLiBRL, a deep Bayesian RL framework that combines generalised linear task models with learnable non-linear basis functions. GLiBRL features conjugate Bayesian inference, yielding exact, sequential posterior updates over task parameters and model noise, together with a closed-form marginal likelihood that eliminates variational inference. The update is naturally permutation-invariant, allowing GLiBRL to integrate with both off- and on-policy algorithms. GLiBRL also learns task representation admitting an exact kernel identity, relating distances between task representations to kernel discrepancies over the task contexts. Compared against eight representa
    
[^227]: 面向金融分析师的生成式人工智能

    Generative AI for Analysts

    [https://arxiv.org/abs/2512.19705](https://arxiv.org/abs/2512.19705)

    生成式AI接入显著提升了金融分析师报告的信息丰富度与时效性，但当信息处理需求较高时预测准确度反而下降，表明生成式AI解除了信息获取约束，却使人类的信息处理能力成为新的瓶颈。

    

    我们研究生成式人工智能（GenAI）如何重塑金融分析师的信息生产。以2023年生成式AI被整合进FACTSET作为一次看似外生的AI接入变化，我们发现与FACTSET相关的研究报告显著更加丰富——独特信息来源增加26%、主题覆盖范围扩大24%、分析方法增多21%，同时报告的时效性也有所提升。然而，这些收益并未一致地改善决策质量：当分析师面临更高的信息处理需求时，其相对预测准确度反而下降。而一个处理相同可观测输入的机器学习基准并未出现类似的恶化，这表明问题在于人类的处理能力约束，而非底层信息质量下降。利用其他数据供应商进行的安慰剂检验排除了平台范围内共同技术趋势的解释。总体而言，生成式AI放宽了信息获取约束，同时使人类的注意力处理能力成为新的约束瓶颈。

    arXiv:2512.19705v2 Announce Type: replace-cross  Abstract: We study how generative artificial intelligence (GenAI) reshapes financial analysts' information production. Using the 2023 integration of GenAI into FACTSET as a plausibly exogenous change in AI access, we find that FACTSET-associated reports become markedly richer--featuring 26% more distinct information sources, 24% broader topical coverage, and 21% more analytical methods--while also improving timeliness. However, these gains do not uniformly improve decision quality: relative forecast accuracy declines when analysts face greater information-processing demands. Yet, a machine-learning benchmark processing the same observable inputs shows no analogous deterioration, pointing to a human processing constraint rather than poorer underlying information. Placebo tests using other data vendors make a common platform-wide technology trend unlikely. Overall, GenAI relaxes information-acquisition constraints while making human attent
    
[^228]: MADS：用于多样化说服数据生成的多智能体对话模拟

    MADS: Multi-Agent Dialogue Simulation for Diverse Persuasion Data Generation

    [https://arxiv.org/abs/2510.05124](https://arxiv.org/abs/2510.05124)

    MADS是一个多智能体对话模拟框架，通过用户智能体、对话智能体和优化智能体的自我博弈，无需人工标注即可低成本生成多样化的说服性多轮对话数据，并在真实营销场景中显著提升了小型大语言模型的说服能力和转化率。

    

    我们提出了MADS（多智能体对话模拟），这是一个通过智能体自我博弈生成具有说服力的多轮对话的可扩展框架。MADS采用三个协同工作的智能体：用户智能体，通过利用星座和MBTI类型等人格特征来模拟多样化的人物角色驱动行为；对话智能体，执行面向任务的说服策略；以及优化智能体，负责评估和完善对话结果。我们进一步通过用户的态度链建模以及专用大语言模型的说服力评估来验证其有效性。该方法能够在无需人工标注的情况下低成本生成训练数据，解决了缺乏用户数据、冷启动评估困难以及提示词效率低下等关键行业挑战。应用于真实世界的营销场景时，MADS显著提升了小型大语言模型的说服能力，将自然流量转化率提高了

    arXiv:2510.05124v3 Announce Type: replace  Abstract: We propose MADS (Multi-Agent Dialogue Simulation), a scalable framework for generating persuasive multi-turn dialogues via agent self-play. MADS employs three coordinated agents: User Agents designed to simulate diverse persona-driven behaviors by leveraging personality signifiers such as Zodiac Signs and MBTI types, a Dialog Agent executing task-oriented persuasion strategies and an Optimization Agent evaluating and refining dialogue outcomes. We further validate its effectiveness through users' Chain-of-Attitude (CoA) modeling and dedicated LLMs' persuasion assessment. This approach enables low-cost generation of training data without human annotation, addressing key industry challenges such as lack of user data, cold-start evaluation difficulties, and prompt inefficiency. Applied to a real-world marketing scenario, MADS significantly improved the persuasion capacity of small LLMs, increasing the organic traffic conversion rate by 
    
[^229]: RAU：基于参考图像的视觉语言模型解剖学理解

    RAU: Reference-based Anatomical Understanding with Vision Language Models

    [https://arxiv.org/abs/2509.22404](https://arxiv.org/abs/2509.22404)

    提出RAU框架，利用视觉语言模型通过带标注参考图像与未标注目标图像之间的相对空间推理实现医学图像中的解剖结构识别与定位，从而缓解专家标注数据稀缺的问题。

    

    解剖学理解，即识别、定位或分割解剖结构的能力，在医学图像分析中至关重要；然而，其进展受到专家标注数据稀缺的制约。一个有前景的解决方案是利用带标注的参考图像来指导对未标注目标图像的解读。尽管近期的视觉语言模型（VLM）展现出不容忽视的视觉推理能力，但它们在基于参考的理解和细粒度定位方面仍然有限。我们提出了RAU，一个基于参考图像的VLM解剖学理解框架。我们首先证明，VLM可以在中等规模数据集上通过参考图像与目标图像之间的相对空间推理学会识别解剖区域。我们通过视觉问答（VQA）和边界框预测验证了这一能力。接着，我们证明由VLM导出的空间线索可以被无缝整合……

    arXiv:2509.22404v2 Announce Type: replace-cross  Abstract: Anatomical understanding, which is the ability to identify, localize, or segment anatomical structures, is critical in medical image analysis; however, its progress is constrained by the scarcity of expert-labeled data. A promising remedy is to leverage an annotated reference image to guide the interpretation of an unlabeled target. Although recent vision-language models (VLMs) exhibit non-trivial visual reasoning, their reference-based understanding and fine-grained localization remain limited. We introduce RAU, a framework for reference-based anatomical understanding with VLMs. We first show that a VLM learns to identify anatomical regions through relative spatial reasoning between reference and target images, trained on a moderately sized dataset. We validate this capability through visual question answering (VQA) and bounding box prediction. Next, we demonstrate that the VLM-derived spatial cues can be seamlessly integrated
    
[^230]: 学习选择最大团算法：从传统机器学习到双通道混合神经架构

    Learning to Select Maximum Clique Algorithms: From Traditional Machine Learning to a Dual-Channel Hybrid Neural Architecture

    [https://arxiv.org/abs/2508.08005](https://arxiv.org/abs/2508.08005)

    提出了一种融合传统机器学习与图神经网络的双通道模型GAT-MLP，通过同时捕捉图的结构特征和全局属性，实现了对最大团问题最优算法的实例感知选择。

    

    摘要：最大团问题（MCP）是一个NP难问题，在生物信息学、网络科学和社会计算等领域有广泛应用，然而没有任何单一算法能在所有不同图实例上始终优于其他算法。这突显了对实例感知算法选择的迫切需求，而这一领域在MCP中仍鲜有探索。为填补这一空白，我们提出了一种新颖的基于学习的框架，该框架融合了传统机器学习和图神经网络。我们首先通过在多样化的图集合上执行四种最先进的精确MCP求解器并提取结构特征，构建了一个基准数据集。对传统分类器的评估表明，随机森林是一个强基线，并揭示出连通性和拓扑特征是性能的关键预测因子。基于这些发现，我们开发了GAT-MLP，一种双通道模型，该模型结合了图注意力网络和MLP，以同时捕捉图的结构特征和全局属性，从而实现对最优MCP算法的有效选择。

    arXiv:2508.08005v4 Announce Type: replace-cross  Abstract: The Maximum Clique Problem (MCP) is an NP-hard problem with wide-ranging applications in fields such as bioinformatics, network science, and social computing, yet no single algorithm consistently outperforms all others across diverse graph instances. This underscores the critical need for instance-aware algorithm selection, a domain that remains largely unexplored for the MCP. To address this gap, we propose a novel learning-based framework that integrates both traditional machine learning and graph neural networks. We first construct a benchmark dataset by executing four state-of-the-art exact MCP solvers on a diverse collection of graphs and extracting structural features. An evaluation of conventional classifiers establishes Random Forest as a strong baseline and reveals that connectivity and topological features are key predictors of performance. Building on these insights, we develop GAT-MLP, a dual-channel model that comb
    
[^231]: SloMoDeblur：一个大规模智能手机图像去模糊数据集

    SloMoDeblur: A Large-Scale Smartphone Image Deblurring Dataset

    [https://arxiv.org/abs/2506.19445](https://arxiv.org/abs/2506.19445)

    本文提出了基于240fps慢动作视频构建的大规模智能手机去模糊数据集SloMoDeblur，通过时间平均30帧合成模糊，包含42,045对1080p分辨率的模糊-清晰图像，覆盖843个场景，填补了智能手机领域去模糊基准数据的空白。

    

    运动模糊仍然是现实世界智能手机成像中最常见且视觉破坏性最大的退化之一，然而现有的去模糊基准数据集在规模、分辨率或领域相关性方面往往存在局限。这一差距对于智能手机领域尤为突出，因为滚动快门、小型传感器和ISP处理产生的模糊统计特性与基于GoPro/单反相机的基准数据集存在差异。我们提出了一个基于240fps慢动作视频构建的大规模面向智能手机的去模糊数据集。为了近似曝光时间内的辐射积分，我们通过对连续N=30帧的固定窗口进行时间平均来合成模糊，这对应于T=1/8秒的有效曝光时间，并选择时间上居中的帧作为清晰的真值。由此得到的基准数据集包含42,045对分辨率为1920×1080的模糊-清晰图像对，涵盖843个不同场景，训练/测试集划分为37,841/4,204对。

    arXiv:2506.19445v5 Announce Type: replace-cross  Abstract: Motion blur remains one of the most common and visually disruptive degradations in real-world smartphone imaging, yet existing deblurring benchmarks are often limited in scale, resolution, or domain relevance. This gap is especially pronounced for smartphones, where rolling shutter, small sensors, and ISP processing produce blur statistics that differ from GoPro/DSLR-based benchmarks. We introduce a large-scale smartphone-oriented deblurring dataset constructed from 240~fps slow-motion video. To approximate exposure-time radiance integration, we synthesize blur by temporally averaging a fixed window of $N=30$ consecutive frames, which corresponds to an effective exposure of $T=1/8$~second, and we select the temporally centered frame as the sharp ground truth. The resulting benchmark contains 42,045 paired blur--sharp images at $1920\times1080$ resolution spanning 843 distinct scenes, with a train/test split of 37,841/4,204 pair
    
[^232]: ROTATE：面向临时团队协作的遗憾驱动开放式训练

    ROTATE: Regret-driven Open-ended Training for Ad Hoc Teamwork

    [https://arxiv.org/abs/2505.23686](https://arxiv.org/abs/2505.23686)

    该论文提出ROTATE框架，将临时团队协作重新表述为AHT智能体与对抗性队友生成器之间的开放式博弈，通过遗憾驱动机制交替改进智能体并生成最具学习价值的队友，从而显著提升与未见伙伴协作的泛化能力。

    

    与先前未见过的伙伴进行协作学习是一个根本性的泛化挑战，被称为临时团队协作。现有方法通常采用两阶段流程：首先生成一个固定的队友群体，然后训练一个AHT智能体与它们协作。这种分离限制了行为覆盖范围，并且忽略了所生成的队友对AHT智能体的学习是否具有信息价值。另一方面，AHT智能体通常是在训练队友集合不可控的假设下进行训练的，尽管其组成会强烈影响泛化能力。本文通过将问题重新表述为AHT智能体与对抗性队友生成器之间的开放式学习过程，为AHT提出了一个统一框架。我们提出了ROTATE，一种遗憾驱动的开放式训练算法，它在改进AHT智能体与生成对智能体学习最有信息价值的队友之间交替进行。

    arXiv:2505.23686v3 Announce Type: replace  Abstract: Learning to collaborate with previously unseen partners is a fundamental generalization challenge, known as Ad Hoc Teamwork (AHT). Existing methods often adopt a two-stage pipeline: first, a fixed population of teammates is generated, and second, an AHT agent is trained to collaborate with them. This separation limits coverage of behaviors and ignores whether the generated teammates are informative for the AHT agent to learn from. On the other hand, AHT agents are typically trained under the assumption that the training teammate set is uncontrollable, despite the fact that its composition strongly influences generalization. This paper presents a unified framework for AHT by reformulating the problem as an open-ended learning process between an AHT agent and an adversarial teammate generator. We introduce ROTATE, a regret-driven, open-ended training algorithm that alternates between improving the AHT agent and generating teammates tha
    
[^233]: 协同视觉-语言强化实现跨多种临床任务的可扩展按需分析

    Synergistic Vision-Language Reinforcement Enables Scalable On-Demand Analysis across Diverse Clinical Tasks

    [https://arxiv.org/abs/2505.03380](https://arxiv.org/abs/2505.03380)

    该研究提出了基于协同视觉-语言强化的可提示分割基础模型SyRe，通过构建包含2000万图像-掩码-描述三元组的SyReData数据集进行训练，实现了跨9种模态、229个临床分割任务、无需手动空间提示或任务特定重训练的可扩展按需分析。

    

    肿瘤及周围危及器官的精确勾画对于放疗、手术及治疗反应评估至关重要，但这一过程仍然耗时且高度依赖专业知识。现有的人工智能系统通常需要手动空间提示或针对特定任务的重新训练，而通用的类别标签对异质性疾病目标所能提供的语义信息十分有限。在此，我们提出了SyRe，一种基于协同视觉-语言强化的可提示分割基础模型。SyRe强化了视觉表征与语言表征之间的双向交互，以提升基于语义的空间理解能力。为支持大规模训练，我们引入了颜色区域描述策略，并构建了SyReData数据集，其中包含涵盖9种模态和229个分割任务的2000万组图像-掩码-描述三元组。通过多样化提示形式的训练，该模型进一步实现了开放式提示能力，

    arXiv:2505.03380v2 Announce Type: replace-cross  Abstract: Accurate delineation of tumors and surrounding organs-at-risk is essential for radiotherapy, surgery and treatment response assessment, yet remains time-consuming and expertise-intensive. Existing artificial intelligence systems often require manual spatial prompts or task-specific retraining, while generic class labels provide limited semantic grounding for heterogeneous disease targets. Here we present SyRe, a promptable segmentation foundation model based on Synergistic vision-language Reinforcement. SyRe strengthens bidirectional interaction between visual and linguistic representations to improve semantically grounded spatial understanding. To support large-scale training, we introduce the Color Region Description strategy and construct SyReData, comprising 20 million image-mask-description triplets across 9 modalities and 229 segmentation tasks. Training with diversified prompt forms further enables open-ended prompting, 
    
[^234]: 使用纵向表格Transformer预测电力中断的预计恢复时间

    Predicting Estimated Times of Restoration for Electrical Outages Using Longitudinal Tabular Transformers

    [https://arxiv.org/abs/2505.00225](https://arxiv.org/abs/2505.00225)

    该论文提出纵向表格Transformer（LTT），将停电预计恢复时间预测从静态表格回归重构为纵向表格回归，通过利用停电事件每次进展的修订记录持续给出更精确的估计，在六家电力公司数据上中位数降低36.9%的客户加权非对称误差。

    

    电力公司会针对面向用户的风暴停电事件发布预计恢复时间（ETR），其准确性决定了用户能否就食品储备、医疗设备和搬迁等事项做出合理决策。以往的研究将ETR预测视为静态表格回归，即每次停电仅贡献一条记录，忽略了停电事件每一次进展（从维修人员分配、派遣、暂停、损坏评估到部分恢复）都会被记录为一次修订的事实。我们将ETR预测重新构建为纵向表格回归问题，并提出了一种纵向表格Transformer（LTT），这是一种基于轴注意力机制的模型，它利用预测之前的所有修订记录，并在每次修订时给出更精确的估计。基于六家运营公司的242,928起风暴停电事件（来自526,468起筛选事件）及1000万条修订记录，LTT在全部六家公司均降低了客户加权的非对称误差，中位数降低幅度达36.9%。

    arXiv:2505.00225v2 Announce Type: replace-cross  Abstract: Utilities publish Estimated Times of Restoration (ETRs) for customer-facing storm outages, and their accuracy governs whether customers can make sound decisions about food, medical equipment, and relocation. Prior work treats ETR as static tabular regression in which each outage contributes one record, discarding the fact that every development of an outage, from crew assignment through dispatch, suspension, damage assessment and partial restoration, is recorded as a revision. We reformulate ETR prediction as longitudinal tabular regression and introduce a Longitudinal Tabular Transformer (LTT), an axial-attention model that consumes the revisions preceding a prediction and issues a refined estimate at every one. On 242{,}928 storm-attributed outages from a cohort of 526{,}468 filtered events and 10.0 million revisions at six operating companies, LTT reduces customer-weighted asymmetric error at all six, by a median of 36.9\,\%
    
[^235]: 通过查询-键对齐量化Transformer中的逻辑一致性

    Quantifying Logical Consistency in Transformers via Query-Key Alignment

    [https://arxiv.org/abs/2502.17017](https://arxiv.org/abs/2502.17017)

    本文提出一种利用Transformer注意力头内查询-键对齐的轻量级逻辑推理评估方法，仅需单次前向传播提取“QK分数”即可可靠区分有效与无效推理，为传统消融技术提供了可扩展的替代方案。

    

    大型语言模型（LLMs）在各种自然语言处理任务中展现了令人瞩目的性能，然而其执行多步逻辑推理的能力仍然是一个开放的挑战。尽管思维链提示通过使模型生成中间步骤来改进了逻辑推理，但它缺乏评估这些逻辑转换连贯性的机制。在本文中，我们提出了一种新颖的轻量级逻辑推理评估策略，该策略利用Transformer注意力头内部的查询-键对齐。通过计算单次前向传播并从精心选择的注意力头中提取“QK分数”，我们的方法揭示了能够可靠区分有效与无效推理的潜在表示，为传统的基于消融的技术提供了一种可扩展的替代方案。我们还在多个逻辑推理基准上进行了实证验证，展示了我们评估方法更强的鲁棒性。

    arXiv:2502.17017v1 Announce Type: cross  Abstract: Large language models (LLMs) have demonstrated impressive performance in various natural language processing tasks, yet their ability to perform multi-step logical reasoning remains an open challenge. Although Chain-of-Thought prompting has improved logical reasoning by enabling models to generate intermediate steps, it lacks mechanisms to assess the coherence of these logical transitions. In this paper, we propose a novel, lightweight evaluation strategy for logical reasoning that uses query-key alignments inside transformer attention heads. By computing a single forward pass and extracting a "QK-score" from carefully chosen heads, our method reveals latent representations that reliably separate valid from invalid inferences, offering a scalable alternative to traditional ablation-based techniques. We also provide an empirical validation on multiple logical reasoning benchmarks, demonstrating improved robustness of our evaluation meth
    
[^236]: 通过求助实现不可逆动力学下的安全学习

    Safe Learning Under Irreversible Dynamics via Asking for Help

    [https://arxiv.org/abs/2502.14043](https://arxiv.org/abs/2502.14043)

    本文首次正式证明，通过允许智能体向导师求助并在相似状态间迁移知识，智能体可以在具有不可逆动力学的无限状态空间未知高风险环境中，以次线性遗憾和次线性求助次数安全高效地学习并获得高回报，无需依赖重置机制。

    

    arXiv:2502.14043v3 公告类型：replace-cross 摘要：大多数具有正式遗憾保证的学习算法本质上依赖于尝试所有可能的行为，当某些错误无法被挽回时，这会带来问题。为此，我们允许学习智能体向导师求助，并在相似状态之间进行知识迁移。我们证明，这种结合能够使智能体既安全又高效地学习。在标准的在线学习假设下，我们提出了一种算法，对于具有不可逆动力学和无限状态空间的马尔可夫决策过程，其遗憾值和导师查询次数相对于时间范围都是次线性的。我们的证明涉及一系列三个归约步骤，使我们的结果比单一算法更具普遍性。从概念上讲，我们的结果可能是首个正式证明：智能体可以在未知、无界且高风险的环境中，无需重置的情况下，在获得高回报的同时实现自给自足。

    arXiv:2502.14043v3 Announce Type: replace-cross  Abstract: Most learning algorithms with formal regret guarantees essentially rely on trying all possible behaviors, which is problematic when some errors cannot be recovered from. Instead, we allow the learning agent to ask for help from a mentor and to transfer knowledge between similar states. We show that this combination enables the agent to learn both safely and effectively. Under standard online learning assumptions, we provide an algorithm whose regret and number of mentor queries are both sublinear in the time horizon for Markov decision processes with irreversible dynamics and infinite state spaces. Our proof involves a sequence of three reductions, making our result more general than a single algorithm. Conceptually, our result may be the first formal proof that it is possible for an agent to obtain high reward while becoming self-sufficient in an unknown, unbounded, and high-stakes environment without resets.
    
[^237]: 电商搜索中的查询品牌实体链接

    Query Brand Entity Linking in E-Commerce Search

    [https://arxiv.org/abs/2502.01555](https://arxiv.org/abs/2502.01555)

    该论文提出两种互补的品牌实体链接方法——级联流水线和极端多分类单阶段方法，在11种语言评估和在线实验中显著提升电商搜索的品牌召回率并保持高精确率，带来用户参与度的可衡量提升。

    

    将用户搜索查询与正确的品牌实体关联起来对于电商产品检索至关重要，但由于查询的简短性（平均仅三到四个词）、缺乏语法结构，以及拥有数十万个不同品牌的商品目录，这项任务仍然充满挑战。我们将其定义为品牌实体链接任务，并开发了两套互补且已大规模部署的解决方案：（1）级联流水线方法，首先通过序列标注检测品牌提及，然后针对品牌知识库进行消歧；（2）单阶段方法，将链接任务建模为极端多分类问题，直接将查询映射到品牌标识符。通过广泛的多语言评估（涵盖11种语言）和受控在线实验，我们证明了所提出的方法在保持高精确率的同时大幅提升了品牌召回率，并带来了可衡量的用户参与度提升。

    arXiv:2502.01555v3 Announce Type: replace  Abstract: Associating user search queries with the correct brand entity is critical for e-commerce product retrieval, yet remains challenging due to the brevity of queries (three to four words on average), their lack of grammatical structure, and a catalog of hundreds of thousands of distinct brands. We formulate this as a brand entity linking task and develop two complementary solutions deployed at scale: (1) a cascaded pipeline that first detects brand mentions via sequence labeling and then disambiguates against a brand knowledge base, and (2) a single-stage approach that frames linking as extreme multiclass classification, directly mapping queries to brand identifiers. Through extensive multilingual evaluation (11 languages) and a controlled online experiment, we demonstrate that the proposed methods substantially improve brand recall while maintaining high precision, leading to measurable gains in customer engagement.
    
[^238]: 量子井字棋的强化学习

    Reinforcement learning for Quantum Tiq-Taq-Toe

    [https://arxiv.org/abs/2411.06429](https://arxiv.org/abs/2411.06429)

    本研究首次将强化学习方法应用于量子井字棋游戏，通过测量和移动历史表征量子状态，为量子计算与强化学习的融合提供了一个易于使用的测试平台。

    

    量子井字棋是量子计算和机器学习领域广为人知的基准测试和实验平台。尽管它很受欢迎，但尚未有强化学习方法被应用于量子井字棋。虽然对量子国际象棋已有一些研究，但该游戏在计算和分析方面要复杂得多。因此，我们研究了量子计算与强化学习在量子井字棋中的结合，这可以作为两个领域融合的易于上手的测试平台。由于量子游戏固有的部分可观测性和潜在的指数级状态复杂度，用经典方法表示量子游戏具有挑战性。在量子井字棋中，状态通过测量（一个3x3的状态概率矩阵）和移动历史（一个9x9的纠缠关系矩阵）来观测，由于每一步移动都可能使量子态坍缩，使得策略制定变得复杂。

    arXiv:2411.06429v2 Announce Type: replace  Abstract: Quantum Tiq-Taq-Toe is a well-known benchmark and playground for both quantum computing and machine learning. Despite its popularity, no reinforcement learning (RL) methods have been applied to Quantum Tiq-Taq-Toe. Although there has been some research on Quantum Chess this game is significantly more complex in terms of computation and analysis. Therefore, we study the combination of quantum computing and reinforcement learning in Quantum Tiq-Taq-Toe, which may serve as an accessible testbed for the integration of both fields.   Quantum games are challenging to represent classically due to their inherent partial observability and the potential for exponential state complexity. In Quantum Tiq-Taq-Toe, states are observed through Measurement (a 3x3 matrix of state probabilities) and Move History (a 9x9 matrix of entanglement relations), making strategy complex as each move can collapse the quantum state.
    
[^239]: 面向深度强化学习的高效基于多样性的经验回放方法

    Efficient Diversity-based Experience Replay for Deep Reinforcement Learning

    [https://arxiv.org/abs/2410.20487](https://arxiv.org/abs/2410.20487)

    本文提出了高效基于多样性的经验回放方法（EDER），通过行列式点过程建模样本间多样性，并结合Cholesky分解与拒绝采样，显著提升了深度强化学习尤其是高维状态空间场景下的学习效率。

    

    经验回放被广泛应用于强化学习中，通过利用过去的经验来提高学习效率。然而，现有的经验回放方法，无论是基于均匀采样还是优先级采样的方法，往往存在效率低下的问题，特别是在具有高维状态空间的真实场景中。为了解决这一局限性，我们提出了一种新颖的方法——高效基于多样性的经验回放（EDER）。EDER采用行列式点过程来建模样本之间的多样性，并基于样本之间的多样性来确定回放的优先级。为了进一步提高学习效率，我们引入了Cholesky分解来处理真实环境中的大规模状态空间。此外，我们还应用了拒绝采样来选择多样性更高的样本，从而提升整体学习效果。我们在MuJoCo中的机器人操作任务、Atari游戏以及（真实环境任务中）进行了大量实验。

    arXiv:2410.20487v5 Announce Type: replace-cross  Abstract: Experience replay is widely used to improve learning efficiency in reinforcement learning by leveraging past experiences. However, existing experience replay methods, whether based on uniform or prioritized sampling, often suffer from low efficiency, particularly in real-world scenarios with high-dimensional state spaces. To address this limitation, we propose a novel approach, Efficient Diversity-based Experience Replay (EDER). EDER employs a determinantal point process to model the diversity between samples and prioritizes replay based on the diversity between samples. To further enhance learning efficiency, we incorporate Cholesky decomposition for handling large state spaces in realistic environments. Additionally, rejection sampling is applied to select samples with higher diversity, thereby improving overall learning efficacy. Extensive experiments are conducted on robotic manipulation tasks in MuJoCo, Atari games, and re
    
[^240]: 面向影响力的个性化联邦学习

    Influence-Oriented Personalized Federated Learning

    [https://arxiv.org/abs/2410.03315](https://arxiv.org/abs/2410.03315)

    提出了一个面向影响力的联邦学习框架FedC²I，通过定量测量客户端级和类别级影响力，实现针对每个客户端的自适应参数聚合，从而提升个性化联邦学习性能。

    

    联邦学习（FL）是一种机器学习范式，其中具有不同行为和偏好的客户端可以在不损害数据隐私的前提下进行协作学习。典型的联邦学习方法通常依赖固定的权重进行参数聚合，从而忽略了客户端之间的相互影响。在实际应用中，具有相似偏好或背景的客户端可能为彼此提供更有用的知识，这些知识可以被利用来提升本地性能。然而，如何量化这种跨客户端的影响力，以及如何利用它来实现个性化聚合，仍然是一个未被充分探索的问题。为了填补这一空白，我们提出了一个面向影响力的联邦学习框架，该框架通过定量测量客户端级别和类别级别的影响力，为每个客户端实现自适应参数聚合（简称FedC²I）。我们的核心思想是通过精心设计的影响力向量，显式地建模联邦学习系统内客户端之间的影响力……

    arXiv:2410.03315v2 Announce Type: replace-cross  Abstract: Federated learning (FL) is a machine learning paradigm where clients with different behaviors and preferences can learn collaboratively without compromising data privacy. Typical FL methods often rely on fixed weighting for parameter aggregation, thereby neglecting the mutual influence among clients. In practice, clients with similar preferences or backgrounds may provide more useful knowledge to each other, which can be leveraged to improve local performance. However, how to quantify such cross-client influence and how to exploit it for personalized aggregation remain underexplored. To address this gap, we propose an influence-oriented Federated learning framework which quantitatively measures Client-level and Class-level Influence to realize adaptive parameter aggregation for each client (FedC^2I for short). Our core idea is to explicitly model the inter-client influence within an FL system via the well-crafted influence vect
    
[^241]: BTBR：一个用于大语言模型隐式偏见消除的贝叶斯理论驱动的概率-模糊框架

    BTBR: A Bayesian-Theory-Driven Probabilistic-Fuzzy Framework for Implicit Bias Removal in Large Language Models

    [https://arxiv.org/abs/2408.10608](https://arxiv.org/abs/2408.10608)

    该论文提出BTBR框架，将有偏见的知识建模为带有显式隶属函数的模糊子集，并结合贝叶斯理论构建概率-模糊混合方法，以检测和消除大语言模型中难以察觉的角色引发隐式偏见。

    

    大语言模型（LLMs）可能会从异构的训练语料库中编码带有偏见的关联，这些偏见在普通提示下不会立即显现，但当模型被引导扮演特定人口统计特征的角色时就会浮现。这种行为通常不表现为明显的有害输出，而是表现为在语义等价任务之间的系统性性能差异，使得由此产生的偏见难以检测和缓解。为了解决这一问题，我们将隐式偏见问题形式化为“角色引发性能差异”，并主张偏见证据应被视为一种分级信号而非二元标签。基于这一观察，我们将有偏见的知识建模为一个配备显式隶属函数的模糊子集，该隶属函数反映每个候选样本偏见证据的强度。基于这一公式化，我们提出了基于贝叶斯理论的偏见消除方法，这是一种混合的概率-模糊框架……

    arXiv:2408.10608v2 Announce Type: replace  Abstract: Large language models (LLMs) may encode biased associations from heterogeneous training corpora that are not immediately visible under ordinary prompting, but can surface when the model is steered toward particular demographic personas. Such behavior often manifests not as explicit toxic output, but as systematic performance differences across semantically equivalent tasks, making the resulting bias difficult to detect and mitigate. To address this issue, we formalize the implicit bias problem as persona-induced performance disparity and argue that bias evidence should be treated as a graded signal rather than a binary label. Motivated by this observation, we model biased knowledge as a fuzzy subset equipped with an explicit membership function that reflects the strength of bias evidence for each candidate example. Building on this formulation, we propose Bayesian-Theory-based Bias Removal (BTBR), a hybrid probabilistic-fuzzy framewo
    
[^242]: 基于基础模型的智能体架构选项分类法：分析与决策模型

    A Taxonomy of Architecture Options for Foundation Model-based Agents: Analysis and Decision Model

    [https://arxiv.org/abs/2408.02920](https://arxiv.org/abs/2408.02920)

    本文提出了一种针对基于基础模型的智能体的架构分类法与决策模型，从功能能力、非功能质量以及设计时与运行时操作等方面，为智能体系统的设计与开发提供结构化指导。

    

    人工智能技术的快速发展推动了智能体系统在各个领域的广泛应用。然而，详细的架构设计需求给这些系统的设计和运行带来了重大挑战。本文介绍了一种专注于基于基础模型的智能体架构的分类法，涵盖了功能能力和非功能质量等关键方面。我们还讨论了设计阶段和运行阶段所涉及的操作，为架构设计和运行特性提供了全面的视角。通过统一和详细化这些分类，我们的分类法旨在改进基于基础模型的智能体的设计。此外，本文建立了一个决策模型，用于指导关键的设计和运行时决策，提供了一种结构化的方法来促进基于基础模型的智能体的开发。我们的贡献包括提供……

    arXiv:2408.02920v2 Announce Type: replace  Abstract: The rapid advancement of AI technology has led to widespread applications of agent systems across various domains. However, the need for detailed architecture design poses significant challenges in designing and operating these systems. This paper introduces a taxonomy focused on the architectures of foundation-model-based agents, addressing critical aspects such as functional capabilities and non-functional qualities. We also discuss the operations involved in both design-time and run-time phases, providing a comprehensive view of architectural design and operational characteristics. By unifying and detailing these classifications, our taxonomy aims to improve the design of foundation-model-based agents. Additionally, the paper establishes a decision model that guides critical design and runtime decisions, offering a structured approach to enhance the development of foundation-model-based agents. Our contributions include providing 
    
[^243]: 提供算法补救的激励

    Incentives to Offer Algorithmic Recourse

    [https://arxiv.org/abs/2301.12884](https://arxiv.org/abs/2301.12884)

    本文通过筛选模型研究决策者提供算法补救的激励问题，证明最优政策是阈值规则，揭示了补救机会既为部分边缘申请人开辟了新的成功路径，也让部分原本可被直接接受的申请人面临额外的高成本障碍。

    

    算法补救承诺通过向被自动化系统拒绝的申请人解释获得通过所需的改变来帮助他们。那么，银行和雇主等决策者有什么激励来提供补救呢？我们在一个筛选模型中研究这个问题，在该模型中补救既具有生产性又具有选择性：完成补救能提高申请人对决策者的价值，但不同申请人在完成补救的成本上存在差异。最优政策是一个阈值规则：拒绝低分数申请人，向中间分数范围的申请人提供补救机会，并直接接受高分数申请人。由于这个中间范围跨越了在没有补救的情况下区分接受与拒绝的临界点，一些边缘申请人由此获得了通往成功的新路径，而另一些原本会被直接接受的申请人，现在则必须跨过一道代价高昂的门槛。

    arXiv:2301.12884v2 Announce Type: replace-cross  Abstract: Algorithmic recourse promises to help applicants rejected by automated systems by explaining the changes needed to secure acceptance. What incentive do decision-makers, such as banks and employers, have to offer recourse? We study this question in a screening model in which recourse is both productive and selective: completing recourse improves an applicant's value to the decision-maker, but applicants differ in their cost of completion. The optimal policy is a threshold rule: reject applicants with low scores, offer recourse to an intermediate range of scores, and accept applicants with high scores outright. Because the intermediate range spans the cutoff that would separate acceptance from rejection when recourse is not available, some marginal applicants gain a new path to acceptance, while others---who would have been accepted outright---must now clear a costly hurdle.
    
[^244]: 在线资源配置中的公平性促进

    Equity Promotion in Online Resource Allocation

    [https://arxiv.org/abs/2112.04169](https://arxiv.org/abs/2112.04169)

    本文针对非营利在线资源配置问题，提出了两种基于线性规划的抽样算法，确保不同人口群体按预设目标比例公平获得资源，并通过竞争比理论分析和真实COVID-19疫苗接种数据实验验证了算法的有效性。

    

    我们考虑在典型的非营利环境下进行在线资源配置，其中有限的甚至稀缺的资源由政府等非营利组织管理。我们关注内部公平性，假设到达的请求者在外部因素（如需求）方面是同质的，但在内部属性（如人口统计特征）方面是异质的。具体而言，我们根据人口统计特征（即种族、性别和年龄）将每个到达的请求者与一个或多个群体相关联，我们的目标是设计一个公平的分配策略，使每个群体的请求者都能获得与预设目标比例相符的公平资源份额。我们提出了两种基于线性规划（LP）的抽样算法，并从理论（竞争比分析）和实验（基于明尼苏达州卫生部维护的真实COVID-19疫苗接种数据）两个角度对算法进行了研究。

    arXiv:2112.04169v3 Announce Type: replace-cross  Abstract: We consider online resource allocation under a typical non-profit setting, where limited or even scarce resources are administered by a not-for-profit organization like a government. We focus on the internal-equity by assuming that arriving requesters are homogeneous in terms of their external factors like demands but heterogeneous for their internal attributes like demographics. Specifically, we associate each arriving requester with one or several groups based on their demographics (i.e., race, gender, and age), and we aim to design an equitable distributing strategy such that every group of requesters can receive a fair share of resources proportional to a preset target ratio. We present two LP-based sampling algorithms and investigate them both theoretically (in terms of competitive-ratio analysis) and experimentally based on real COVID-19 vaccination data maintained by the Minnesota Department of Health. Both theoretical a
    

