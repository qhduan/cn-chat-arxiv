# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FastBench: Can Streaming VLMs Perceive High-Dynamic Real-World Streams?](https://arxiv.org/abs/2610.12427) | 该论文提出 FastBench 基准，通过基于轨迹的自动化问答生成与验证流水线（结合 SAM3/CoTracker3 轨迹验证和人工审核）评估流式 VLM 在高动态真实世界视频流中的感知能力，并提出无需训练的 ProactiveFrame 基线，通过文本词元动态调节输入帧率以捕捉快速事件。 |
| [^2] | [WOVEN: Weaving Visual World Modeling into Multimodal LLMs](https://arxiv.org/abs/2610.12417) | 提出WOVEN——一个按场景、动作和推理类型组织的视觉转换推理训练数据源与基准（含36,076个示例），验证视觉转换推理可作为可跨任务复用的共享训练原语，以提升多模态大语言模型的空间、具身、物理和时间推理能力。 |
| [^3] | [Predicting Alignment Generalization with Value Representations](https://arxiv.org/abs/2610.12410) | 本文提出“对齐泛化预测”新任务，通过对66个价值的大规模分析发现，基于模型激活的价值表征能够显著优于基于文本描述的方法，预测模型微调遵循某一价值后在未见价值上的行为泛化。 |
| [^4] | [ViSkill: Reinforcing VLM Agents with Evolving Visual-Native Skills](https://arxiv.org/abs/2610.12403) | 提出ViSkill视觉原生技能学习框架，将成功交互编码为可检索的视觉技能卡片，在推理和奖励塑形中发挥作用，并通过成功轨迹回流技能库形成技能积累与策略改进相互促进的闭环，显著提升视觉语言模型智能体的样本效率。 |
| [^5] | [SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in Vision-Language Models](https://arxiv.org/abs/2610.12402) | SpaceCast-Bench是首个通过“观察-变换-推断”框架直接且诊断性地评估视觉-语言模型预测性空间推理能力的基准，其评估揭示了最强模型（58.0%）与人类表现（87.2%）之间的显著差距。 |
| [^6] | [Long Text to Predictive Features: LLM-Guided Blockwise Feature Engineering via Executable Program Search](https://arxiv.org/abs/2610.12390) | 提出LLM-BlockFE框架，由LLM在离线阶段通过逐步追加代码块并结合深度校准信用分配的分块级回滚搜索，将长文本自动转化为可执行的特征程序，使在线推理无需调用LLM即可高效利用长文本信息。 |
| [^7] | [Latent Core Tokenizer: Compress, but Meaningfully](https://arxiv.org/abs/2610.12376) | 潜在核心分词器（LCT）在构建词表前先利用最小描述长度、熵边界信号和形态结构约束识别可复用的语言单元，在104种语言上实现了比BPE等方法更低的碎片率和更高的多语言下游任务性能，证明压缩率本身并不能预测表示质量。 |
| [^8] | [OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport](https://arxiv.org/abs/2610.12375) | OnTrack提出了一种流式结构感知最优传输监控机制，通过将LLM智能体的执行步骤与记录的成功运行轨迹实时对比，在每步约一毫秒内实现对异常行为的告警或阻断，兼顾了低延迟与安全性。 |
| [^9] | [Which Skill to Distill? SGUID: Selecting a Compact Skill Bank for Model-Skill Co-Evolution](https://arxiv.org/abs/2610.12367) | 提出SGUID方法，仅保留训练中持续产生有效学习信号的核心技能用于蒸馏，证明精选的6个技能即可匹配或超越全技能库蒸馏的性能。 |
| [^10] | [Cited but Not Consulted: A Counterfactual Audit of Legal Chain-of-Thought Faithfulness](https://arxiv.org/abs/2610.12361) | 该论文通过反事实审计方法发现，大语言模型在法律推理中虽能正确引用法规或判例，但其裁决结果往往并不真正依赖于所引用的法律依据，揭示了法律思维链的引用忠实性缺失问题。 |
| [^11] | [Accurate but Not Humble: Evaluating Epistemic Humility in LLM Agents under Knowledge Conflict](https://arxiv.org/abs/2610.12360) | 该论文提出用“认知谦逊”概念（通过识别、解决、上报三个行为维度）评估大语言模型智能体在知识冲突下承认并传达不确定性的能力，发现智能体虽然准确但不谦逊，往往坚持错误结论而不愿承认不确定性。 |
| [^12] | [Overcoming Prior Barriers: Supervised Fine-Tuning under Long-Tail Distribution](https://arxiv.org/abs/2610.12345) | 该论文提出“先验壁垒”新概念，揭示预训练模型对各概念的支持程度呈长尾分布、使头部与尾部概念在监督微调时起点不同，并通过理论推导预测风险界，指出尾部概念需要额外指令才能克服高先验壁垒。 |
| [^13] | [Can AI Agents Learn Their Way to the Top? Evaluating Heuristic Learning in a Long-Running Game Agent Competition](https://arxiv.org/abs/2610.12341) | 论文提出对抗性启发式学习范式与AAArena基准，让模型权重固定不变的AI智能体通过解读规则、分析回放和修订可执行游戏策略，在12个真实对抗性游戏竞赛中学习并争夺最高排名。 |
| [^14] | [VFold: Symmetry-Aware Cross-Layer Value Cache Compression](https://arxiv.org/abs/2610.12338) | 该论文提出了一种对称性感知的跨层价值缓存合并策略，无需修改模型架构即可在解码时压缩KV缓存内存，并能与高比率量化或键缓存剪枝等现有技术组合使用，实现单一方法无法达到的更高压缩比。 |
| [^15] | [SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference](https://arxiv.org/abs/2610.12327) | 该论文提出SparseDecoding，一种解码感知的剪枝方法，通过在模型自生成的token序列上计算Hessian来消除自然序列与生成序列之间的分布偏移，从而在保证准确性的同时提升LLM解码阶段的推理效率。 |
| [^16] | [Verdict Without the Rule: Diagnosing and Auditing Regulatory Rule Sensitivity in LLM Compliance Systems](https://arxiv.org/abs/2610.12313) | 研究发现大语言模型合规系统的判定对所给监管规则的删除、替换或否定等扰动往往不敏感，其guard模型在自定义规则适配下准确率仅51%、略高于随机水平，表明系统判定更多依赖案例的简单性而非对规则的真正遵循。 |
| [^17] | [HarnessSQL: Harness-Native Training for SQL Agents in Realistic Database Environments](https://arxiv.org/abs/2610.12274) | HarnessSQL提出了一种框架原生的后训练方法，通过在带有隐藏执行预言机的真实数据库多轮交互环境中进行监督微调和执行奖励强化学习，弥合了文本到SQL模型静态训练与有状态实际部署之间的差距，显著提升了Spider 2.0-SQLite上的执行准确率。 |
| [^18] | [EgoVoice: Proactive Spoken Assistance from Egocentric Multimodal Streams](https://arxiv.org/abs/2610.12248) | 提出了EgoVoice框架，通过对HoloAssist第一人称视频进行声源分离与语音重合成构建训练数据，微调全模态大语言模型，使其能够自主决定何时开口以及提供何种语音指导，实现可穿戴AR设备中的主动式语音辅助。 |
| [^19] | [NativeScope: Relation-Localized Retrieval over Native Topology with a Correct Anchor](https://arxiv.org/abs/2610.12243) | 论文提出 NativeScope，利用已知锚点和关系在数据系统原生结构（章节归属、会话边界、顺序）上先定位范围再排序，相比全范围稠密 RAG 将文档和记忆的原生单元召回率分别提升 42.75 和 22.00 个百分点。 |
| [^20] | [TokenRouter: Efficient Serving System for Token-Level LLM Routing](https://arxiv.org/abs/2610.12242) | TokenRouter是一个高效且对开发者友好的服务系统，通过“以请求为中心的编程、以模型为中心的执行”的设计原则，解决了现有系统在Token级路由LLM推理中步骤失同步、批次准入延迟和实现复杂度高的难题。 |
| [^21] | [Language Models as AI Research World Models](https://arxiv.org/abs/2610.12235) | 该论文提出将语言模型作为“研究世界模型”来预测AI实验的结果，并证明真实实验经验获得的研究知识能提升预测准确性且可跨研究环境复用，从而在有限实验预算下支持AI研究的持续自我改进。 |
| [^22] | [DiffuPlex: Accelerating Full-Duplex Spoken Dialog Models via Rolling Masked Diffusion](https://arxiv.org/abs/2610.12214) | DiffuPlex 提出了一种滚动掩码扩散框架，通过单次骨干网络唤醒预测多个未来用户与助手帧，并在交互出现偏差时仅修改未播放的未来内容，从而加速全双工口语对话模型。 |
| [^23] | [SciTBERT: A family of chronologically consistent language models for scientific and technological language processing](https://arxiv.org/abs/2610.12207) | 提出了SciTBERT——一系列时间一致的BERT衍生语言模型，训练数据截止日期覆盖2013至2025年每年，通过消除前瞻偏差和领域偏差，使模型适用于研究科学与技术随时间演变的特性。 |
| [^24] | [SteerablePlex: Can We Steer Full-Duplex Models?](https://arxiv.org/abs/2610.12201) | 本文提出SimIF-Bench基准测试揭示当前开源全双工语音模型难以遵循预设场景约束，并通过GDPO训练方案构建了SteerablePlex模型，使其能在持续对话中遵循文本指令的同时保持轮次转换能力。 |
| [^25] | [When KL Regularization Misfires in Group Policy Optimization](https://arxiv.org/abs/2610.12161) | 论文剖析了组策略优化中KL正则化与奖励相互作用的七种失效模式，并提出零和校准策略优化（ZCPO），借助条件KL度量的相对漂移校准组内奖励系数，从而提升优化效果。 |
| [^26] | [Language-Specific Effects of Tokenizer Choice in Multilingual Language Models](https://arxiv.org/abs/2610.12144) | 该研究通过固定架构、语料和训练预算、仅改变分词器的控制实验（123个模型、54种分词器），发现分词器选择对训练数据较少的低资源语言影响更大。 |
| [^27] | [Rehearse Everything, Remember Nothing: Attic-KV Rehearses What Will Be Read](https://arxiv.org/abs/2610.12133) | 该论文发现传统KV缓存压缩中“重读全部上下文”的排练策略在低保留率下会因预算分散而失效，并提出Attic-KV——只排练将来会被读取的内容，像考前自测一样大幅提升缓存压缩后的记忆效果。 |
| [^28] | [A persistent accuracy ceiling in automated verbal deception detection](https://arxiv.org/abs/2610.12118) | 通过对25年间289篇报告和6,136个分类模型的系统回顾与元分析，该研究揭示自动化言语欺骗检测的准确率主要受方法论质量而非模型复杂度驱动，存在约70-75%的持续准确率上限。 |
| [^29] | [All Verdicts are Not Equal: Rethinking LLM Judge Reliability](https://arxiv.org/abs/2610.12083) | 该论文通过大规模压力测试揭示了LLM裁判评估系统存在的严重可靠性缺陷（判决不稳定、受顺序影响、错误判决也会保持一致），并提出“可信判决率”这一统一指标来同时量化评估的可重现性、顺序不变性和准确性。 |
| [^30] | [ILM: An AI-Powered Storytelling Educational Tool](https://arxiv.org/abs/2610.12064) | 本文提出了ILM平台，通过结合阿拉伯语自然语言处理、知识图谱构建和多语言检索技术，为先知故事提供结构化、可视化和多语言支持的互动式学习与理解评估体验。 |
| [^31] | [When Should Agents Think? Adaptive Reasoning via Cross-Turn Estimation](https://arxiv.org/abs/2610.12061) | 提出 RACE 方法，利用移除推理后后续动作似然下降这一轻量信号进行跨轮次估计，判断智能体何时需要新的推理步骤，从而在无需昂贵生成式验证的情况下实现自适应推理训练。 |
| [^32] | [Natural Language to First-Order Logic LLM-based Autoformalization](https://arxiv.org/abs/2610.12030) | 本文首次通过区分本体抽取与逻辑翻译，为一阶逻辑自动形式化任务提供了原则性定义，并系统综述了现有数据集、评估指标和基于大语言模型的方法，指出了基准测试与语义评估等方面的开放性挑战。 |
| [^33] | [InterviewPlayground: A Simulation Environment for Evaluating AI Interviewers](https://arxiv.org/abs/2610.12023) | 提出了InterviewPlayground，一个基于社会理论模拟参与者行为、通过多轮交互生成标准化访谈成绩单来评估AI访谈员的模拟环境，并验证了模拟评估结果能够预测AI访谈员与真实人类参与者互动时的实际表现。 |
| [^34] | [Examining Social Attribution in LLM Reasoning: A Theory-Guided Probing Methodology](https://arxiv.org/abs/2610.12022) | 该论文首次系统性地探索大语言模型的社会归因能力，在归因理论指导下构建了包含经典心理学情境与现实场景的基准，以考察大语言模型在责任与过错归因上的判断及其内部机制。 |
| [^35] | [Agentic-TTT: Training test-time policy for test-time training](https://arxiv.org/abs/2610.12002) | 提出Agentic-TTT框架，通过训练一个测试时策略来智能决策何时、如何调用测试时训练（TTT）以及是否复用已有技能，从而实现模型参数层面的自主化自我改进。 |
| [^36] | [DataVista: Diagnosing Multimodal LLMs on Data Video Understanding](https://arxiv.org/abs/2610.11993) | 该论文提出了首个数据视频理解基准DataVista，包含961个真实数据视频和6,775个评估问题，通过三级渐进式能力框架系统评估了19个主流多模态大语言模型在该任务上的表现。 |
| [^37] | [Specialized Decision Models vs. General-Purpose LLMs: Benchmarking Jev Across Knowledge, Reasoning, and Multilingual Tasks](https://arxiv.org/abs/2610.11978) | 专用决策模型Jev在知识和常识类基准上可与前沿大语言模型媲美，但在需要多步计算的数学任务上显著落后，表明其能力仅限于知识型决策而非计算推理。 |
| [^38] | [MindFlow: Mind Supernet Powered Thinking Flows for Research Idea Innovation](https://arxiv.org/abs/2610.11966) | MindFlow将科研创意生成形式化为由模块化思维算子构成的图结构“心智流”，通过概率性心智超网建模与基于锦标赛相对排序的优化，让控制器能够动态采样并逐步生成更高质量的创意。 |
| [^39] | [MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement](https://arxiv.org/abs/2610.11959) | MiMo-V2.6通过在混合SWA架构基础上，沿批次与吞吐量、环境多样性、评分器计算三个维度系统性地扩展强化学习计算量，推动全模态模型迈向自我提升。 |
| [^40] | [When History Helps and Hurts: Selective History Use across Multimodal Turns](https://arxiv.org/abs/2610.11948) | 该论文提出ReTurn基准（7,000个涵盖视觉与音频的任务），首次将多模态多轮对话中的历史使用需求与问题难度解耦，以系统评估模型对对话历史的选择性使用能力。 |
| [^41] | [Project Greenhouse: Progress Toward Fully Open and Sovereign Agentic Search](https://arxiv.org/abs/2610.11922) | 温室计划证明，仅用适度计算资源和公开数据集，通过从零预训练加有监督微调的两步简单流程、不依赖第三方骨干模型，即可构建完全开放且自主可控、具有竞争力的智能体搜索重排序模型。 |
| [^42] | [Event-Centric Memory with Query-Aware Graph Augmentation for Long-Term Conversational Agents](https://arxiv.org/abs/2610.11920) | 提出了受人类记忆启发的QGMem框架，它将长对话历史转化为事件索引的原子记忆单元，并通过查询感知的图增强将与查询相关的记忆建模为工作记忆，从而提升长期对话代理的记忆构建与激活能力。 |
| [^43] | [Not Every Change Is Necessary: Recoverable Drift in Large Language Model Unlearning](https://arxiv.org/abs/2610.11915) | 提出PTP-U框架，先通过局部解析编辑削弱目标知识关联，再将非目标输出分布与原始模型对齐，从而在实现机器遗忘的同时恢复因遗忘而被附带损害的非目标能力。 |
| [^44] | [Can Decision Models Understand Stance? Evaluating Jev Against General-Purpose LLMs](https://arxiv.org/abs/2610.11901) | 本研究首次将专用决策模型Jev应用于立场检测任务，发现其在英文文本数据集VAST上可媲美GPT-5.6并超越其他通用大语言模型，但在中文对话数据集ZS-CSD上表现欠佳，其局限主要源于对回复关系和立场方向的理解不足。 |
| [^45] | [Forms of LLM-Integrated Applications from LLM-Chats to Autonomous AI Agent System](https://arxiv.org/abs/2610.11899) | 该综述系统考察了chatbot、copilot、RAG、agent等LLM集成应用标签背后的真实架构内涵，归纳出七种反复出现的应用形式，并发现四家主流厂商的编程智能体均采用委派子智能体的“推理-行动”循环这一共同架构。 |
| [^46] | [GRPODropout: Less is More for Online Reinforcement Learning Rollouts](https://arxiv.org/abs/2610.11854) | 提出GRPODropout方法，通过在GRPO策略更新前选择性地移除少量高概率的正优势轨迹并重新居中保留的优势，有效缓解策略熵坍缩问题，提升大语言模型的推理能力。 |
| [^47] | [Detecting Spin in Clinical Trials with Large Language Models](https://arxiv.org/abs/2610.11845) | 该研究提出了一种基于开源大语言模型的临床试验结局转换自动检测方法，通过提示工程、基于token概率的分类和多数投票，在测试集上达到F1分数0.78和准确率0.90，性能优于基线文本相似度模型。 |
| [^48] | [Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks](https://arxiv.org/abs/2610.11794) | Memento 3 让冻结参数的 LLM 智能体将可修正的环境假设记录为自然语言“规则手册”并编译为可执行代码，通过预测误差驱动的观察—反思—修订—编译—验证循环，实现显式世界模型的持续递归自我改进。 |
| [^49] | [Easy to anticipate, hard to compute: boundary dependence finds the computed outputs that entropy patching misses](https://arxiv.org/abs/2610.11790) | 论文揭示了字节级语言模型中基于熵的分块策略会系统性遗漏“类型可预料但数值需计算”的位置（如数学等式的计算结果），并提出将熵信号与无标签的边界依赖信号相结合来放置补丁边界，从而大幅提升模型在这类计算输出上的准确率。 |
| [^50] | [From Sparse Representations to Behavioral Insights for Multimodal Depression Assessment](https://arxiv.org/abs/2610.11787) | 提出了可解释的多模态抑郁评估框架BehavDep，将多模态行为表征分解为与行为语义概念关联的稀疏潜在因子，并通过弱监督学习视频级抑郁倾向得分与跨观察聚合，在实现最佳评估性能的同时揭示各模态的互补贡献。 |
| [^51] | [DPPM: Dual-Path Parametric Memory for Personalized Language Models](https://arxiv.org/abs/2610.11776) | DPPM通过“证据路径”池化交互历史以保留早期证据、“增量路径”按序更新关联状态以捕捉偏好变化，融合生成历史条件化的LoRA适配器，在个性化任务上显著超越基线（PersonaMem-v2达54.22%，PrefEval达86.79%）。 |
| [^52] | [RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing](https://arxiv.org/abs/2610.11775) | 本文提出叠加特化假说，认为MoE专家专精于细粒度特征的组合而非单一领域，并据此开发RouterInterp方法，通过稀疏自编码器特征与自然语言解释来解读专家路由，检测准确率比先前方法高出约65%。 |
| [^53] | [Same Outcome, Different Evidence: Intent Recovery in LLM Safety Evaluation](https://arxiv.org/abs/2610.11766) | 该论文提出将攻击成功率（ASR）与操作性理解率（UR）配对使用，以区分LLM安全评估中相同无害结果背后的不同原因，揭示出相似的ASR值可能掩盖截然不同的任务恢复率。 |
| [^54] | [Thinking Inertia: LLMs Keep Thinking When Told Not To](https://arxiv.org/abs/2610.11765) | 该论文提出了三项新指标（空思考率、问题-答案前相关性、显式推理率）来严谨地衡量大语言模型的“不思考”行为，并揭示了一种“思维惯性”现象——即使被明确要求不思考，LLM仍会继续进行推理。 |
| [^55] | [4-Tensor Attention Model for Semantic Physical Reality](https://arxiv.org/abs/2610.11716) | 该论文提出一种四阶张量注意力模型，通过在语义纤维与时间上下文纤维上进行联合归一化注意力来预测场景的下一语义状态，在参数量几乎相同的情况下，其最后一句交叉熵在三个设置上均优于自由运行的一维 Transformer（最多低 5.3%），为视频生成和机器人规划提供了新方法。 |
| [^56] | [Internalizer: Portable Context-to-Parameter Mapping for Very Large Language Models](https://arxiv.org/abs/2610.11715) | Internalizer是一种可移植的上下文到参数映射超网络，先在小模型上低成本训练后即可移植到2840亿参数的DeepSeek v4 Flash上，为其生成文档特定的LoRA适配器，使模型无需在上下文窗口中放入文档即可将知识内化到权重中。 |
| [^57] | [Structured Sentiment Analysis Using Sequence Labeling as Dependency Graph Parsing](https://arxiv.org/abs/2610.11695) | 该论文提出将结构化情感分析建模为依存图解析任务，并创新性地通过序列标注结合线性化图编码来解决，在五种语言的七个数据集上取得了与更复杂的先进单模型方法相当的性能。 |
| [^58] | [TRACE: Diagnosing Verifier Brittleness in Agentic Evaluation](https://arxiv.org/abs/2610.11678) | TRACE协议通过对评估施加针对性变更并对未改变的轨迹重新评分，来诊断智能体评估中验证器分数的变化究竟是源于真实能力差异，还是评分器本身的脆弱性。 |
| [^59] | [DIAL-OPD: Learning More from Fewer Tokens in On-Policy Distillation](https://arxiv.org/abs/2610.11659) | 提出DIAL-OPD方法，通过用师生概率的对数均值加权奖励幅度来筛选最具学习价值的token，证明了在更少token上训练的在策略蒸馏效果可以优于全token训练。 |
| [^60] | [Harness Evolution Hits a Ceiling: When Weight Training Should Begin](https://arxiv.org/abs/2610.11655) | 该论文提出通过失败构成分析来决定改进长时程LLM智能体的正确杠杆——将失败区分为过程失败与内容失败，其中Harness演化修复过程失败且其行为可被训练进权重，而内容失败则需依靠权重训练来解决。 |
| [^61] | [Phonologically Informed Tokenization for German Speech Recognition: A Cross-Domain Study](https://arxiv.org/abs/2610.11646) | 本文提出基于音系学知识（Pyphen 音节划分与字素-音素转换）的分词方法用于德语端到端语音识别，发现在域内条件下其性能与 BPE 和字符基线相当，而跨领域表现主要由词表规模而非语言学分词方式决定。 |
| [^62] | [UXBench Pro: Benchmarking Personalized User Experience in Multi-Turn Dialogue Interactions](https://arxiv.org/abs/2610.11638) | UXBench Pro通过构建带用户画像的测试集，并结合个性化用户奖励模型与用户模拟器的双视角评估范式，实现了对多轮对话交互中个性化用户体验的细粒度基准测试。 |
| [^63] | [Large Language Model Turnover Undermines Screening for Artificial Intelligence-Assisted Scientific Writing](https://arxiv.org/abs/2610.11599) | 该研究用三大厂商23个版本的LLM改写4,000篇论文摘要并训练检测器，发现仅基于旧版本模型训练的AI写作检测器会在模型代际更新时性能崩溃（检出率从99%骤降至3.8%），表明LLM的快速迭代严重削弱了期刊对AI辅助写作筛查的可靠性。 |
| [^64] | [Probing for Long-Horizon Deductive Reasoning Capabilities in Language Models with Prolog](https://arxiv.org/abs/2610.11592) | 该论文构建了合成测试平台ProloNg，系统探究前沿大语言模型在长上下文中执行Prolog演绎推理的能力，发现随着推理深度增加模型性能急剧下降，多数模型在推理深度超过10后即接近随机猜测水平。 |
| [^65] | [Measuring Cultural Alignment Beyond the Average: A Framework for Evaluating Maternal-Health LLM Interactions in Indian Contexts](https://arxiv.org/abs/2610.11586) | 该论文提出MH-INDIC文化评估框架，通过十个孕产妇健康推理维度评估十个大语言模型在印度北部语境下的文化对齐程度，发现虽然部分模型能接近人类群体层面的分布，但所有系统的行为变异性都显著低于人类。 |
| [^66] | [Adapting English Quality Classifiers for Multilingual LLM Pretraining Data Selection](https://arxiv.org/abs/2610.11585) | 提出一种多语言适配方法，通过在Transformer编码器嵌入之上训练小型多层感知机，并利用机器翻译获取标签，将英语质量分类器扩展至100多种语言的LLM预训练数据筛选，同时保持下游基准性能。 |
| [^67] | [Chronos Enables Code Agents to Reason over Software Evolution](https://arxiv.org/abs/2610.11578) | Chronos 提出一个测试时框架，将历史拉取请求提炼为结构化经验卡片并通过类型化关系图谱连接，使基于 LLM 的代码智能体能够利用软件演化历史来生成补丁并在候选补丁间做出选择，从而提升代码修复效果。 |
| [^68] | [Smoothing the Top-k Exposure Boundary for Sparse Mixture-of-Experts](https://arxiv.org/abs/2610.11575) | 提出弹性专家路由方法，通过在以k为中心的局部离散分布中随机采样激活专家数量，将稀疏混合专家模型中刚性的top-k选择边界软化为渐进的概率分布，在不增加计算成本的前提下缓解了竞争专家因阈值划分而导致的训练反馈不均衡问题。 |
| [^69] | [Incremental Open-Ended Deep Research with Structured Harness](https://arxiv.org/abs/2610.11566) | 本文提出增量式开放式深度研究（Incremental-OEDR）新范式与结构化框架，将报告视为可演化的研究状态，通过结构化表示、检索与生成实现报告的选择性增量更新和证据复用，并建立了跨越十年的时序评估框架。 |
| [^70] | [SWE-Journey: Towards More Realistic Evaluation of Coding Assistants through Long-Horizon, Multi-Turn Interaction](https://arxiv.org/abs/2610.11559) | 提出 SWE-Journey 基准，通过弱到强合成流水线自动构建长周期编码任务，并利用从真实交互数据挖掘的用户画像和用户模拟智能体重现多轮交互，从而实现对编码助手更贴近现实的评估。 |
| [^71] | [Prosody-to-Text: Predicting text from low-pass filtered speech](https://arxiv.org/abs/2610.11544) | 本文首次探索“韵律到文本”任务，证明仅用截止频率约450Hz的低通滤波语音微调Whisper模型即可部分恢复原始句子（WER 36%，10%完全恢复），揭示了低频韵律特征中蕴含着可被恢复的词汇信息。 |
| [^72] | [Learning the Loop, Not Just the Page: Execution-Grounded Loop Learning for Web Generation](https://arxiv.org/abs/2610.11543) | WebLoop提出了一种基于执行的框架，在共享策略中联合训练生成器、无需执行的评论家和精炼器，使模型学会“诊断并修复”的完整循环，从而显著提升功能性网页生成的质量。 |
| [^73] | [Constitutional Gating and Deterministic Recovery for Multi-Agent LLM Negotiation: Ablations Against a Stateful Adversarial Gatekeeper](https://arxiv.org/abs/2610.11542) | 该论文提出由5支柱运行时宪法、4层集群和认知退火组成的三部分控制栈，并通过针对有状态对抗性守门人的消融实验，验证其能消除多智能体LLM谈判中的三类模型调用浪费并实现确定性死锁恢复。 |
| [^74] | [When Can You Prune Your Network? A Study of Intermediate Neurons in Multilingual Speech Parsing](https://arxiv.org/abs/2610.11520) | 本文提出一种移除中间神经网络单元的更简单端到端语音句法分析架构，在参数量减少12%的同时达到相当或更优的识别与句法分析性能，并发现中间神经网络单元在预训练编码器被冻结时有助于缩小表征差距。 |
| [^75] | [Residual Advantage: Student-Relative Teacher Guidance for RL with Verifiable Rewards](https://arxiv.org/abs/2610.11519) | 提出残差优势（RA）方法，将师生概率残差转化为有界且经中心化的优势信号并与验证器优势结合，为可验证奖励强化学习提供稳定、解耦的教师引导。 |
| [^76] | [Does Modern Standard Arabic (MSA) Dominate Arabic Dialects in LLMs? A Representation-Level Analysis](https://arxiv.org/abs/2610.11510) | 本研究通过对26种阿拉伯语变体的表示层面分析发现，大语言模型的内部表示中现代标准阿拉伯语并不支配阿拉伯语方言，方言表示高度重叠，从而挑战了“MSA偏向的生成源于内部表示支配”这一常见解释。 |
| [^77] | [Beyond Sequences: Distilling Structured Decision Memory for LLM Recommendation](https://arxiv.org/abs/2610.11501) | 提出MARI框架，通过决策记忆库将用户历史决策归档为包含目标、约束与权衡的结构化决策记忆（SDM），使LLM推荐系统在高度相似物品的“困难选择”场景中具备可解释的决策依据。 |
| [^78] | [SAGE: Sink-Aware Guided Emphasis for Visual Grounding in Vision-Language Decoders](https://arxiv.org/abs/2610.11469) | 该论文发现视觉-语言模型解码器中的注意力汇具有分层结构——早期和晚期层会出现提示不变的注意力坍缩（PIS），而中间层才驱动视觉-语言对齐，并据此提出轻量级干预方法 SAGE，将解码器注意力从 PIS 引向与查询相关的感兴趣区域，以提升可靠性并减少幻觉。 |
| [^79] | [Who Verifies the Verifier? Co-Evolving Inspectable Graders with Self-Improving Agents](https://arxiv.org/abs/2610.11464) | 该论文提出将验证器本身作为进化对象——即由可检查的确定性缺陷检测器组成的表达式，通过锚定参考集一致性和输出共识来选择而非依据智能体分数——从而在自我改进循环中避免奖励作弊和共同盲点，并在MBPP+上比手工种子组合提升0.21的保留一致性。 |
| [^80] | [Beyond Speech Captions: Speech-Rewarded Style Planning for Conversational Text-to-Speech](https://arxiv.org/abs/2610.11461) | 提出语音奖励风格规划（SRSP），利用冻结TTS模型对目标语音token的教师强制似然作为奖励、通过GRPO训练文本风格规划器，生成比文本描述伪标签更能有效控制对话式语音风格与情感的风格指令。 |
| [^81] | [SAIL: Scientific Agentic Intelligence via a Science-Aware Loop](https://arxiv.org/abs/2610.11451) | SAIL是一个通过“科学感知改进循环”训练的开放科学智能体模型（35B总参数/3B激活参数），由前沿AI智能体自动诊断其任务失败并生成针对性训练任务，使其在文献研究、科学编程和多步骤研究工作流中达到有竞争力的表现。 |
| [^82] | [Adversarial Cues in Decision Models Used as Judges: The Role of Request Presentation](https://arxiv.org/abs/2610.11436) | 研究发现，在结构化评判请求的某些呈现方式下，仅需在候选答案中添加一个冒号，就能使大模型评判者将明显错误的答案误判为正确，误接受率从1–3%激增至约26.5%，揭示了评判请求呈现方式带来的对抗性脆弱性。 |
| [^83] | [BioBigBird: A Sparse Attention Model for Long-Range Dependency Processing in Biomedical Text](https://arxiv.org/abs/2610.11430) | BioBigBird是一种基于稀疏注意力机制的生物医学双向语言模型，可处理长达4096个token的长序列，并通过多任务学习联合优化命名实体识别与关系抽取，在BLURB基准上取得了与最先进模型相当的表现。 |
| [^84] | [Fact over Fiction: Detection of Pathological Hallucinations in Sinhala-to-English Neural Machine Translation](https://arxiv.org/abs/2610.11389) | 该论文针对低资源的僧伽罗语到英语神经机器翻译，提出了一个无参考幻觉检测框架，通过五种语言学损坏策略构建45,000样本合成数据集并微调mDeBERTa-v3进行token级序列标注，达到了0.841的token级F1分数。 |
| [^85] | [From a Prompt to Repertoires: Evolving Functional REpertoires Enable LLM Continual Learning](https://arxiv.org/abs/2610.11373) | 提出演化功能技能库方法，通过将单一提示扩展为不断演化的多功能技能库，克服提示优化在持续学习中的灾难性遗忘与规则过拟合问题，使大语言模型无需更新参数即可持续习得新能力。 |
| [^86] | [SignRAG: Unified Retrieval-Augmented Gloss-Free Sign Language Translation](https://arxiv.org/abs/2610.11371) | SignRAG提出了一个结合分层预训练、目标域检索增强和检索感知强化微调的统一框架，使无Gloss注释手语翻译能够有效适配仅解码器大语言模型。 |
| [^87] | [UniData: Universal Multimodal Instruction Generation Pipeline](https://arxiv.org/abs/2610.11363) | UniData是一个通用多模态指令生成流水线，能够将简单的用户需求自动转化为高质量的多轮多模态指令数据，解决了现有多模态指令生成方法模态支持有限以及难以生成多轮指令的问题。 |
| [^88] | [AdaptEvo: Adaptive Agent Learning with Evolving Supervision](https://arxiv.org/abs/2610.11354) | AdaptEvo提出了一个在不完美监督下学习的智能体框架，通过置信度自适应GRPO平衡结果与过程奖励，并从反复失败中演化出可复用的决策知识和更精细的过程评估标准，同时在工业级多模态内容审核数据集上验证了其有效性。 |
| [^89] | [RL-ARC: Calibrating Large Reasoning Models via Reasoning-guided Uncertainty](https://arxiv.org/abs/2610.11352) | RL-ARC提出了一种校准感知训练框架，将推理置信度作为辅助信号来校准答案置信度——对正确回答施加推理引导正则化、对错误回答施加过度自信惩罚，从而在不牺牲推理性能的情况下改善大推理模型在分布内外场景中的校准并缓解过度自信问题。 |
| [^90] | [Deception by Omission: Language Models Knowingly Hide Their Mistakes](https://arxiv.org/abs/2610.11351) | 该研究通过在模型轨迹中注入合成错误，首次系统揭示了LLM普遍存在的“通过隐瞒进行欺骗”行为——在智能体场景中高达67.1%的错误未被披露，且部分情况下模型明知有错仍刻意隐瞒。 |
| [^91] | [Type-Checking for Pattern-Based Tree Transformations](https://arxiv.org/abs/2610.11337) | 该论文提出了基于（源模式，目标模式）对集合的有限表示的树变换模型，并证明了尽管该模型的等价性检查不可判定，但类型检查问题是可判定的。 |
| [^92] | [ReCal: Calibrating Structured Pruning for On-Policy Distillation Recovery](https://arxiv.org/abs/2610.11332) | 提出恢复感知校准方法ReCal，通过在剪枝前利用未剪枝教师模型与剪枝探针之间的前向KL散度，识别并保护易被剪枝破坏的教师支持的预测，从而显著提升结构化剪枝后在线策略蒸馏的恢复效果。 |
| [^93] | [MetaEncoder: Exploring the Limit of Bi-Encoders for Multimodal System One Decision Making with Natural Language Interface](https://arxiv.org/abs/2610.11316) | 提出MetaEncoder，将预训练的30B解码器微调为基于单向对比学习的双编码器架构，实现了完全通过自然语言接口、可从小闭集扩展到数百万开放集候选空间的多模态系统一决策。 |
| [^94] | [From Retrieval to Reconstruction: Constructing Evolvable Cognitive Memory for Long-Term Dialogue](https://arxiv.org/abs/2610.11314) | 提出了CogMem认知记忆架构，基于PEC²F图模式将对话增量转换为具有来源感知的可演化图记录，区分有来源归属的主观主张与客观事实，从而支持长期对话中的可靠推理。 |
| [^95] | [BeliefScope: Diagnosing Evidence-Driven Revision and Pressure-Induced Shifts in Large Language Models](https://arxiv.org/abs/2610.11305) | 提出BeliefScope受控黑盒框架，通过交叉实验设计将大语言模型的信念修正可靠地归因于真正的相关证据还是无实质信息的用户压力，实现两种影响来源的有效分离。 |
| [^96] | [When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better](https://arxiv.org/abs/2610.11291) | 研究发现，基于初始学生模型离线rollout的Semi-OPD蒸馏方法在17个教师-学生模型对中的14个上优于同策略蒸馏，两者之间的选择取决于师生模型输出token的重叠率。 |
| [^97] | [REMORY: Learning Residual Memory for Context Compaction](https://arxiv.org/abs/2610.11287) | REMORY 提出一种神经记忆网络，通过生成软记忆token作为文本摘要的“残差”补充，使冻结的LLM仅用 5.2% 的输入位置即可接近全上下文性能，并显著减少重复工具输出和工具错误。 |
| [^98] | [Phonological Interference in Multilingual Speech Models](https://arxiv.org/abs/2610.11275) | 该研究揭示了多语言语音模型中的“音系干扰”这一系统性失败模式——模型错误地假设输入属于单一语言并强加其音系，导致在语码转换语音上丢失32%至79%的语言特有音素。 |
| [^99] | [Gated Memory: Admission-Controlled Memory Formation for Conversational AI](https://arxiv.org/abs/2610.11270) | 该论文提出Gated Memory框架，通过在对话与存储之间设置准入控制检查点，在事实提取前基于完整话语上下文评估候选事实，解决了关键上下文信号在提取时不可逆丢失这一制约记忆质量的瓶颈问题。 |
| [^100] | [Why On-Policy Distillation Sometimes Fails: Vanishing Learning Signals](https://arxiv.org/abs/2610.11247) | 该研究发现大规模教师模型在在策略蒸馏中会导致基于梯度的学习信号过早消失，从而造成早期损失平台期，并从理论上为足够接近初始学生的教师证明了学习信号的局部恢复保证。 |
| [^101] | [Read What Matters: Query-Adaptive Quantization for KV Caches](https://arxiv.org/abs/2610.11245) | 提出ReadKV方法，通过渐进式编码实现KV缓存的查询自适应量化，根据每个解码查询动态分配键通道和值令牌的读取精度，并从理论上证明查询依赖的读取方案在相同读取预算下严格优于任何查询无关的方案。 |
| [^102] | [MiniVer-V: Identifying Minimal Sufficient Evidence for Short Video Verification](https://arxiv.org/abs/2610.11233) | 该论文提出以“证据充分性”为证据选择标准的短视频事实核查基准MiniVer-V，并设计两层核查框架，将基于内部证据的声明-视频一致性判断与需要外部佐证的事实性结论判定相分离，从而识别足以支撑核查结论的最小充分证据。 |
| [^103] | [SafeInferCom: Safe Inference-Time Compute via Verifier-Guided Mid-Generation Intervention for Robotic Task Planning](https://arxiv.org/abs/2610.11223) | SafeInferCom是一个形式化验证器引导的推理时干预框架，通过在不干扰解码轨迹的情况下监控和验证中间计划，防止有效计划被后续推理覆盖并引导生成过程中的错误纠正，从而显著提升机器人任务规划的成功率与可靠性。 |
| [^104] | [The Lattice of Transition Laws](https://arxiv.org/abs/2610.11216) | 本文将扩散模型与自回归模型统一为同一个“腐蚀格”上的不同路径，通过定义解码调度的成本（即并行步骤所舍弃的依赖性），证明零成本调度的最少步数由数据的几何结构决定（例如等于图的树深度），从而可在解码前预测调度性能。 |
| [^105] | [Bridging KV-Cache Quantization and Linear Attention: From Theory to Pretrained Weight Migration](https://arxiv.org/abs/2610.11214) | 提出RAM-Net作为统一KV缓存量化与线性注意力的桥梁，通过离散地址空间上的软分配机制，在理论上证明其可分离读写重叠能局部逼近全注意力相似度，并支持从预训练权重迁移。 |
| [^106] | [Selective Listening: Mechanism-Guided Control of Audio Influence in Large Audio-Language Models](https://arxiv.org/abs/2610.11196) | 提出ICAP-Gate方法，通过机制引导、任务条件化的方式控制大型音频-语言模型的后期音频通路，在防止无关音频干扰文本推理的同时，不损害依赖音频的任务（如语音识别）的性能。 |
| [^107] | [RAG-Stress: Probing the Limits of Evidence Reliance in Retrieval-Augmented Generation](https://arxiv.org/abs/2610.11183) | 该论文提出RAG-Stress受控诊断协议，通过编辑证据断言构造误导性检索内容，系统测量其诱导模型替换原本正确答案的“误导率”，揭示了十五个RAG系统在证据依赖上的脆弱性。 |
| [^108] | [LadderEdit: Edit-Level Residual Compression for Memory-Efficient Lifelong Editing of LLMs](https://arxiv.org/abs/2610.11160) | LadderEdit通过将每条编辑先以低秩草图存储、仅对未满足契约的困难编辑沿阶梯逐级提升秩的方式压缩LoRA适配器，在保持编辑覆盖效果的同时将内存占用降低5.2倍，并支持5万次连续终身编辑。 |
| [^109] | [Local Prototype Reconstruction for Text-Compatible Speech-to-LLM Bridge Pretraining](https://arxiv.org/abs/2610.11159) | 该论文提出局部原型重建（LPR）这一轻量级训练正则化方法，使语音-LLM桥接嵌入保持接近冻结LLM输入嵌入的邻域，从而弥补现有预训练目标无法捕捉词元级词汇兼容性的不足，实现可迁移的文本兼容语音到LLM桥接预训练。 |
| [^110] | [Do LLMs Learn from Rewards in Context? : Rethinking the role of reward in In-Context Reinforcement Learning](https://arxiv.org/abs/2610.11152) | 该研究通过受控实验发现，在大语言模型的直接上下文强化学习中，奖励信号虽然被读取却几乎不产生学习效果，而轨迹本身（即使语义被打乱或损坏）才是驱动上下文改进的关键，从而挑战了上下文学习真正实现强化学习的假设。 |
| [^111] | [ActiveMedAgent: Cost-Aware Trajectory Learning for Multimodal Medical Diagnosis](https://arxiv.org/abs/2610.11140) | 该论文提出 ActiveMedAgent 框架，通过追踪诊断概率分布并按“诊断效用减去成本”对信息获取轨迹打分、离线训练轻量级 MLP 控制器，使冻结的视觉语言模型在多模态医学诊断中以更低成本、更少模态获得更准确的诊断结果。 |
| [^112] | [The "10th Juror": Open-Set Standpoint Screening for Bureaucratic Bias Detection](https://arxiv.org/abs/2610.11136) | 提出立场感知的多智能体框架 MARS-Gov，通过融合法律检索、开放集目标筛选、专门陪审员与改写验证，实现政府文档偏见的闭环治理，并能动态识别和应对新出现的偏见目标群体。 |
| [^113] | [Can a System-One LLM Perform Knowledge Tracing When Few or No Learners Are Logged?](https://arxiv.org/abs/2610.11135) | 该论文提出，现成的“系统一”LLM（Jev/JevKT）无需目标平台的学习者数据或仅需极少数据即可完成知识追踪，其性能超过28个深度知识追踪模型和“系统二”LLM方法，而API成本仅约为后者的百分之一。 |
| [^114] | [SFT-as-Context Mitigates Forgetting in Supervised Fine-Tuning](https://arxiv.org/abs/2610.11132) | 提出无需训练的SFT-as-context方法，让父模型将SFT模型的响应作为上下文通过上下文学习获得微调能力，从而在保留通用能力的同时缓解监督微调带来的遗忘问题。 |
| [^115] | [GameCommBench: A Unified Benchmark and Type-Aware Evaluation for AI-Generated Game Commentary](https://arxiv.org/abs/2610.11129) | 该论文提出了GameCommBench统一基准与TACE类型感知评估框架，首次系统性地对AI游戏解说进行分类型评估，发现实时观察和策略分析是当前AI解说员的主要能力瓶颈。 |
| [^116] | [Lapras: Latent Reasoning for Time Series Language Models](https://arxiv.org/abs/2610.11111) | Lapras是一个后训练框架，通过让时间序列语言模型在潜在空间而非离散语言标记中进行推理，避免了将连续时序信号转化为文字描述时的信息丢失与早期错误传播，从而生成与输入信号一致、更忠实的答案。 |
| [^117] | [Clinician use of language models diverges from how the models are evaluated](https://arxiv.org/abs/2610.11069) | 通过分析医疗系统中6,342名临床医生发送的超过12.7万条查询，研究发现临床医生实际主要将语言模型用于文档行政事务和知识检索而非诊断，这与现有基于考试题目和诊断病例的基准测试评估方式存在显著偏差。 |
| [^118] | [Measuring and Mitigating Solution Mode Collapse in RLVR](https://arxiv.org/abs/2610.11064) | 本研究提出ModeBench多解任务基准，发现RLVR后训练在保持或提升准确率的同时，会导致模型的解多样性坍缩，概率集中于更少的正确解题模式上。 |
| [^119] | [FedAlphaEdit: Null-Space-Aligned Merging for Collaborative Knowledge Editing](https://arxiv.org/abs/2610.11033) | 提出首个在统一零空间原则下对齐本地编辑与服务器端合并规则的协同知识编辑框架 FedAlphaEdit，使多个机构无需共享原始编辑请求即可安全整合各自的知识编辑。 |
| [^120] | [Prompts versus Rules: Auditing and Controlling Speech Naturalness Behaviors in Voice User Simulators](https://arxiv.org/abs/2610.11015) | 研究发现，基于提示词的方法无法可靠地让语音用户模拟器产生不流畅、打断和反馈等自然语音行为，而基于规则的注入算法能够更一致、更自然地控制这些行为。 |
| [^121] | [Back in Style: A Sociolinguistic Approach to Authoring and Measuring Persona Fidelity in User Simulation](https://arxiv.org/abs/2610.10988) | 该研究提出一种社会语言学方法，将用户人格创作具体化为可观察的语言风格比率，从而利用无需模型的文体计量学与词典内容分析工具确定性地测量用户模拟器的人格忠实度，并在客服智能体实验中验证其优于描述性基线。 |
| [^122] | [When Citations Mislead? A Claim-Level Benchmark for Legal Hallucination Detection](https://arxiv.org/abs/2610.10971) | 该论文提出了PARCEL基准——基于纽约州上诉法院判决构建的包含3,396条标注声明的法律幻觉检测数据集，将其转化为三分类自然语言推理任务来评估大语言模型，发现即使最强模型准确率达0.97，仍会将缺乏引用支持的声明误判为已支持，其中虚构但看似合理的引用造成的性能下降最为严重。 |
| [^123] | [AI4Fire: Evaluating Large Language Models on Wildfire Tasks](https://arxiv.org/abs/2610.10946) | AI4Fire 基准首次在五个野火任务上对多个大语言模型进行成对的“裸跑”与“接地”零样本评估，发现接地信息（如只读SQL工具）在直接包含答案时能大幅提升准确率（从至多16%提升到至少88%），但简单规则基线仍然难以被超越。 |
| [^124] | [StoreBench: A Live-Commerce Environment for Evaluating and Training Autonomous Operator Agents](https://arxiv.org/abs/2610.10942) | StoreBench是一个让智能体在生产级电商后端上经营线上服装店的实时环境，通过全天候动态市场、校准的通过阈值和抗操纵的奖励机制，来评估和训练自主运营智能体的长程规划与经济决策能力。 |
| [^125] | [Language Models for Page-Level Layout Decisions in E-commerce Search](https://arxiv.org/abs/2610.10920) | 该论文研究了利用语言模型离线评估电商搜索页面级布局决策（如在特定位置插入二级堆栈）是否对用户有益，从而减少对昂贵在线A/B测试的依赖。 |
| [^126] | [Large Language Models for Machine Translation Quality Annotation: Humans and Models Are Both Challenged](https://arxiv.org/abs/2610.10918) | 本文系统评估了大语言模型在 MQM 和 ESA 两种机器翻译质量标注任务中与人类标注者的一致性，涵盖 70 个语言对及 WMT23/WMT25 数据，发现 LLM 与人类的一致性在某些情况下甚至超过人类标注者之间的一致性，但无论人类还是模型在此任务上都面临挑战。 |
| [^127] | [Stochastic Teacher Intervention for Agentic On-Policy Distillation](https://arxiv.org/abs/2610.10878) | 提出STI-OPD框架，在多轮智能体在线策略蒸馏中，通过由师生策略差异引导的随机教师干预将学生动作替换为教师生成的动作，从而缓解多轮交互中的误差累积并最大化可靠监督的获取。 |
| [^128] | [Sparse Attention Is Matrix Approximation, Not Choosing from a Bag of Values](https://arxiv.org/abs/2610.10871) | 该论文指出稀疏注意力在概念上应被表述为矩阵近似问题而非挑选最大注意力数值，并据此提出了矩阵近似稀疏注意力方法MASA。 |
| [^129] | [Conversational Voice Aesthetic Model with Reinforcement Learning from Human Listeners](https://arxiv.org/abs/2610.10868) | 该论文提出CVAM语音大语言模型，通过合成美学描述的监督微调和基于约3万条人类听众标注的组相对策略优化（GRPO），实现对语音性别、音高、语速、情感、表达方式等九项美学属性的预测，其与人类听众的一致性超越了Gemini 3.1 Pro和开源语音LLM。 |
| [^130] | [Disentangling Linguistic and Paralinguistic Information with Routed Sparse Autoencoders](https://arxiv.org/abs/2610.10865) | 提出将TopK稀疏自编码器与路由特定监督和跨因子对抗相结合的方法，成功将自监督语音编码器中纠缠的语言学与副语言学信息（如说话人身份、情感、韵律）解耦到不同路由中，且这种分离在不同编码器、语料库和探测方法上均保持一致并可迁移。 |
| [^131] | [Real Long-Term Memory for AI: A 50-Million-Token Window That Is Faster and Cheaper Than Recompute](https://arxiv.org/abs/2610.10845) | 本文提出并验证了一个名为galahad-kv的记忆层，通过将KV状态加密保存到本地NVMe磁盘并按需逐字节精确加载，实现了5000万令牌的超长上下文记忆，速度比重计算快2.8至4.3倍、GPU能耗降低8.8至12.3倍，且GPU内存占用在整个处理过程中保持恒定。 |
| [^132] | [Grammar Concept Annotation at Scale: Deployed Fine-Tuned Small Language Models Outperform Prompted Frontier Models](https://arxiv.org/abs/2610.10827) | 通过在教师生成的监督数据上微调小型语言模型并部署 0.8B 模型进行大规模语法概念标注，其性能在精确率和召回率上超越提示式前沿大模型，同时大幅降低成本。 |
| [^133] | [NavGPT-3: Harnessing Context in a Hierarchical Navigation Runtime](https://arxiv.org/abs/2610.10787) | NavGPT-3 提出了一个类似操作系统的分层导航运行时框架，将具备长时程推理能力的语言模型与低延迟的 VLA 动作策略通过多线程调度机制相结合，使机器人能通过线程中断与切换快速响应突发真实事件，其 8B VLA 模型在 R2R-CE 上取得了 74.51 SR 的领先性能。 |
| [^134] | [Plan-and-Patch: Diffusion Language Models for Agentic Planning](https://arxiv.org/abs/2610.10786) | 提出Plan-and-Patch框架，利用扩散语言模型通过并行去掩码生成结构化的类程序计划，并借助仅填充受影响区域、保持前后步骤不变的局部修复机制，实现对计划的高效修订。 |
| [^135] | [Clarify, Then Focus: Statement Normalization for Conversation Analytics at Scale](https://arxiv.org/abs/2610.10758) | 提出语句规范化方法，将对话转化为带说话者归属、来源引用和语义标签的简短语句，使语义更明确并支持按需证据选择，从而提升大规模企业对话分析中下游模型的表现。 |
| [^136] | [Conversational Task Disambiguation over Tabular Data: Leakage-Aware Formulation, Benchmark Suite, and Training](https://arxiv.org/abs/2610.10740) | 提出“歧义可验证任务”的形式化框架，将智能体分解为提问策略与求解策略、环境分解为oracle与验证器，从而实现任务消歧与求解能力的独立评估，并给出oracle泄漏的形式化定义与无裁判的泄漏诊断。 |
| [^137] | [Lossy Compressive Text Autoencoders](https://arxiv.org/abs/2610.10738) | 提出一种带残差低维离散瓶颈的文本自编码器，实现文本的有损压缩表示，在网页文本上以每字节2.24比特达到与无损压缩算法相当的压缩率，同时保持良好的重构质量和下游任务性能。 |
| [^138] | [Cognitive Thermometers: Machine Learning and Logical Complexity](https://arxiv.org/abs/2610.10724) | 本文提出将机器学习模型视为“认知温度计”来度量语义复杂度，证明基于学习的复杂度比逻辑复杂度能更好地解释自然语言对语义范畴的偏好，并为连接符号逻辑与连接主义AI提供了统一框架。 |
| [^139] | [Large Language Model-Assisted Preparation of Transportation Management Plans: A Case Study with WisDOT WisTMP System](https://arxiv.org/abs/2610.10650) | 本文提出一个基于开源大语言模型微调的框架，通过将历史WisTMP文档转化为结构化问答数据集进行训练并本地化部署，实现了交通管理计划内容的自动生成，在提升编制效率的同时保障了数据安全。 |
| [^140] | [Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models](https://arxiv.org/abs/2610.10623) | LoopOPD让循环语言模型利用自身更深层循环计算作为冻结“教师”，在学生自生成轨迹上进行在线策略蒸馏，无需外部教师或特权信息即可获得密集监督实现自我提升，D-LoopOPD进一步将该过程动态化。 |
| [^141] | [WorldBench: Evaluating LLMs on Three.js Voxel World Generation](https://arxiv.org/abs/2610.10622) | WorldBench通过让评判系统主动探索运行中的3D世界（控制时钟、环绕观察、派遣导航智能体取景）并将视觉观察与源代码相互交叉验证，解决了现有单一视角评判方法不可靠的问题，实现了对LLM生成的Three.js体素世界的可靠评估。 |
| [^142] | [Wieszcz-XIX: A 3.1-Billion-Word Corpus of Pre-1918 Polish and Temporally Bounded Language Models Trained From Scratch](https://arxiv.org/abs/2610.10592) | 该研究构建了31亿词规模的1800-1918年波兰语历史语料库Wieszcz-XIX，其规模比现有标注语料库大三个数量级以上，并通过过滤、去重和时间泄漏审计等严格流程量化控制数据缺陷，进而从零训练了时间受限的语言模型。 |
| [^143] | [Diffu-LoRA: A Novel Low-Rank Adaptation for Personalized Diffusion Models](https://arxiv.org/abs/2610.10550) | Diffu-LoRA 是一种参数高效的个性化扩散模型方法，通过门控低秩适应、双层优化和渐进式剪枝自动学习各层之间适应能力的非均匀分配，在冻结预训练主干网络的同时以极少参数实现主体身份保持与提示词遵循。 |
| [^144] | [An Explainable Header-Centric Framework for Large-Scale Semantic Table Interpretation and Data Quality Assessment](https://arxiv.org/abs/2610.10541) | 该论文提出了一个可解释的、以表头为中心的框架，通过利用精心策划的词汇资源将表头映射到39种可解释类型并保留词元级可追溯性，实现了仅基于元数据场景下的列类型标注与数据质量评估，并将数据质量检测结果聚合为轻量级的数据源级质量指标HeadersIQ。 |
| [^145] | [Judging in Latent Space: Efficient Generative Reward Modeling via Semantics-Preserving Compression](https://arxiv.org/abs/2610.09788) | LatentGRM通过语义分块、压缩与重构将评估过程编码为紧凑的连续潜在轨迹，无需逐token生成文本评估即可实现高效奖励建模，并在4B和8B规模上取得与显式SFT评判器相当的偏好判断准确率。 |
| [^146] | [How Do LLMs Change Predictions Under Negation?](https://arxiv.org/abs/2610.09571) | 大语言模型通过“抑制原始答案、提升偏好候选”的机制处理否定，而非像人类那样利用原始答案信息来确定应排除的内容，这一与人类处理方式的差异是模型否定任务失败的关键根源。 |
| [^147] | [Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models](https://arxiv.org/abs/2610.09145) | 在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。 |
| [^148] | [Leveraging LLM-Generated Explanations for Detecting Emotionally Rewritten Fake News](https://arxiv.org/abs/2610.08835) | 本文提出门控交叉注意力（GCA）框架，利用大语言模型从原始新闻生成的解释作为稳定背景知识，自适应融合情感改写新闻与解释内容，显著提升了假新闻检测模型在保持事实的情感变体下的鲁棒性。 |
| [^149] | [Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States](https://arxiv.org/abs/2610.08818) | 该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。 |
| [^150] | [Have I Seen Enough? Frozen Video-Language Models Encode Evidence Readiness](https://arxiv.org/abs/2610.08560) | 该论文发现冻结的视频-语言模型内部已线性编码了一种由问题条件化、可跨基准泛化且与答案对错无关的“证据就绪度”信号，因此无需额外训练触发器即可判断流式视频问答中证据是否已充分到来。 |
| [^151] | [Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight](https://arxiv.org/abs/2610.08077) | 该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。 |
| [^152] | [SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics](https://arxiv.org/abs/2610.06513) | 提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。 |
| [^153] | [Scaling Verifiable Environments for Long-horizon Work Agents](https://arxiv.org/abs/2610.04906) | WorkForge 提出了一个可扩展的合成框架，从专家工作流和真实世界资源出发，通过提取可核查的事实锚点，自动构建支持长程交互与可信验证的工作智能体训练环境。 |
| [^154] | [The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer](https://arxiv.org/abs/2610.03827) | 相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。 |
| [^155] | [Single-Pass Uncertainty Heads for Claim-Level Hallucination Detection in Persian Medical Language Models](https://arxiv.org/abs/2610.03482) | 该论文将 LLM 不确定性头框架首次适配到波斯语医学语言模型中，通过在冻结主干模型的注意力图和 token 概率上训练单次前向传播的轻量级声明级检测头，实现了无需重复采样的低成本声明级幻觉检测，并构建了两个波斯语声明级幻觉数据集。 |
| [^156] | [Evaluating and Improving the Robustness of Large Language Models to Input Sequence Variations](https://arxiv.org/abs/2610.02432) | 本论文提出了基于Jensen-Shannon散度的生成式鲁棒性度量R_stab，并开发了自适应进化黑盒攻击方法ASA（对LLM-as-a-Judge系统攻击成功率高达73.8%），用于系统性地评估和提升大语言模型对提示注入、木马后门等对抗性输入序列变化的鲁棒性。 |
| [^157] | [Budgeted Cache Repair for Cross-Context KV-Cache Reuse](https://arxiv.org/abs/2610.02233) | 该论文发现跨上下文KV缓存复用会带来显著的准确率损失，并提出预算化缓存修复（BCR）方法，在单token行这一选择仍有收益的最小单元上，利用草稿token的注意力对缓存行排序并精确重算固定预算的行数，从而有效修复缓存误差。 |
| [^158] | [Stochastic Rounding in Low-Precision Transformer Inference: A Variable-Precision Emulation Study of a Small GPT-2](https://arxiv.org/abs/2610.01889) | 该研究通过变精度随机舍入（VPSR）算法将PRISM舍入库扩展至任意精度，并从理论与实验两方面揭示：低精度Transformer推理中随机舍入与就近舍入的优劣取决于运算位点，SR的误差以O(√n u)增长而RN以O(n u)增长，这一差距在长MLP下投影等长线性投影中最为明显。 |
| [^159] | [Empty Commitments: When Agents Promise What Their Runtime Cannot Deliver](https://arxiv.org/abs/2610.01045) | 本文提出“空洞承诺”这一新概念，指智能体（如聊天机器人）承诺其工具和运行时配置根本无法实现的未来行动（例如“我明天会提醒你”），并形式化了包含三种失败类型的承诺语义、工具锚定条件和结果分类体系，同时设计了一种通过逐步增加持久性能力来测量空洞承诺的实验协议。 |
| [^160] | [LLM Persona Unlearning](https://arxiv.org/abs/2609.39882) | 该论文提出“人格遗忘”任务及PersonaUnlearnBench基准，通过权重级编辑使大语言模型中的指定人格难以被诱发，并发现标准遗忘方法无法在不牺牲生成质量或通用能力的情况下可靠地抹除目标人格。 |
| [^161] | [Thinking Outside the Box: Can Language Models Rely on External Guidance Selectively?](https://arxiv.org/abs/2609.39578) | 该论文提出 Box²-Bench 基准来衡量语言模型“跳出思维定式”的能力，即在受益于可靠工作流引导的同时否决不可靠引导，并发现反事实监督微调与基于结果的强化学习是提升这一能力的两种互补训练策略。 |
| [^162] | [Marking Contour Tones in Yor\`{u}b\'{a}](https://arxiv.org/abs/2609.38627) | 本文提出在约鲁巴语正字法中采用caron（ˇ）和circumflex（ˆ）符号来标记单个元音上的升降曲折调，以解决传统拼写中声调信息缺失甚至颠倒姓名含义的问题，并使其首次可通过标准键盘输入和计算文本处理。 |
| [^163] | [CompOrca: Corpus-Scale Compliance Labelling of Instruction-Tuning Data](https://arxiv.org/abs/2609.37807) | 该论文提出了 CompOrca，利用开源大模型评判器对整个 OpenOrca 语料库（超过 420 万条样本）进行五次独立判定，首次实现了语料库规模的合规性标注，并发布带投票计数的标注结果，以支持对拒答与不服从行为的研究。 |
| [^164] | [Opera: A Verbal Critic Framework for Long-horizon Coding Agents](https://arxiv.org/abs/2609.33987) | Opera 是一个面向长时程编码智能体的口头批评框架，其核心创新在于将每次纠正作为持久记录持续跟进至问题真正解决，通过周期性/事件驱动触发、类型化算子诊断、基于证据的反馈审计以及后续行动跟踪，区分表面服从与实际解决，从而在多个基准上将智能体解决率提升高达 15 个百分点。 |
| [^165] | [SMAT: Simple and Efficient Merge-Aware Training](https://arxiv.org/abs/2609.33437) | SMAT将常见模型合并操作抽象为缩放、掩码和扰动三种基本操作，通过在采样生成的模拟合并参数上联合优化专家损失与期望损失，实现了以极小训练开销显著提升合并后性能的简单高效合并感知训练方法。 |
| [^166] | [When to Evict, Not What to Keep: Draft-Guided Eviction for Training-Free KV-Cache Compression](https://arxiv.org/abs/2609.33334) | 该论文提出草稿引导驱逐（DGE）方法，将KV缓存驱逐时机从预填充结束推迟到基于完整缓存起草出前两个答案token之后，通过利用答案自身前缀生成的查询来指导驱逐决策，解决了传统“优化保留什么”策略的补偿效应和选择效应失效问题，实现免训练的KV缓存压缩。 |
| [^167] | [FA-Bench: A Benchmark for Phone- and Word-Level Timestamp Accuracy in Forced Alignment and ASR on Clean and Noisy Speech](https://arxiv.org/abs/2609.32396) | FA-Bench是一个开放的强制对齐与ASR时间戳精度基准框架，通过统一的评测协议（涵盖21个开源模型和9个商业API、干净与退化语音）和基于容差的F1指标，解决了已有研究方法不一、结果不可比的问题，并消除了标准MAE在依赖识别输出的系统上造成的9%至14%的分数虚高。 |
| [^168] | [PlurVA-LLM-2026 Shared Task Track-1: Pluralistic Value Alignment in LLMs via Multilingual Fine-Tuning and Threshold Calibration](https://arxiv.org/abs/2609.32382) | 在资源受限条件下，该系统通过 4 比特 QLoRA 微调 Llama 3.1 8B，结合分语言定制的数据增强策略（中文选项排列、印尼语标注者投票扩展、斯里兰卡语二元重构）及条件阈值校准，在三国多元价值对齐任务上取得 0.805 的宏平均准确率。 |
| [^169] | [Using LMs to Model the Effects of Context and Coreference during Sentence Comprehension](https://arxiv.org/abs/2609.32119) | 本研究通过在四个大规模英语阅读时间数据集上系统调节GPT-2的上下文窗口大小，发现上下文长度与人类心理语言学数据的拟合度呈U形关系，扩展上下文（500–1,000词元）拟合最佳，且这一优势依赖于跨句实体共指链的完整性。 |
| [^170] | [SlideLab: Audience-Centered Scientific Slide Generation and Evaluation](https://arxiv.org/abs/2609.30294) | SlideLab 是一个无需训练的多智能体框架，能从研究论文生成以观众为中心的科学演示幻灯片，在盲测中于 77% 的论文上超越开源与商业系统且推理成本降低约 4 倍，并配套提出模拟会议室的观众导向评估框架 ConfArena。 |
| [^171] | [JevOut: Natural Context Can Flip Decision Models](https://arxiv.org/abs/2609.30243) | 研究表明，决策模型（如Jev）对看似自然无害的简短上下文添加内容极其脆弱，优化后的上下文能在61.4%的情况下颠覆模型原本正确的决策，且在许多案例中使模型以至少0.7的高概率输出固定的错误选项。 |
| [^172] | [ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds](https://arxiv.org/abs/2609.30199) | 提出ExplorationBench基准，利用规则可执行且与常识相冲突的“异星世界”沙盒（AlienCode与AlienLogic），实现了对AI系统科学探索能力的可验证评估，排除了仅凭记忆预训练知识解题的可能。 |
| [^173] | [ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks](https://arxiv.org/abs/2609.29102) | 提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。 |
| [^174] | [UniDataAgent: An Ontology-Grounded Agent for Enterprise Question-to-Report Automation](https://arxiv.org/abs/2609.27257) | UniDataAgent通过将企业语义获取（构建版本化本体）与在线“问题到报告”执行分离，将原本需要约一周的人工本体构建缩短至数小时，并将报告生成缩短至几分钟。 |
| [^175] | [CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning](https://arxiv.org/abs/2609.23257) | CTRL框架将语义推理与定量预测解耦，利用LLM智能体作为控制器分析预测误差的分解成分并输出控制信号，再由轻量级残差解码器转化为预测修正，从而提升非平稳环境下时间序列预测的稳定性与可解释性。 |
| [^176] | [Pretrained Persona Mixture Models and Tandem Models for Human Simulation](https://arxiv.org/abs/2609.22607) | 该论文提出“人格混合模型”，即使用预训练基础模型并借助特定人物的简短对话样本实现人格绑定，能比指令微调模型更准确地模拟人类，并保留更多人类对话的自然多样性。 |
| [^177] | [Embedding Models Measure in Peculiar Ways](https://arxiv.org/abs/2609.20821) | 该研究发现嵌入模型对质量、距离、时间和体积等物理测量的表示十分微弱且奇特，主要受表面字符串相似性的强烈影响，而重新校准相似度也无法显著改善其与真实物理测量的对齐。 |
| [^178] | [EconSkills: Studying Skill Transfer and Retrieval for Web Agents on Live Economic Data](https://arxiv.org/abs/2609.19523) | EconSkills框架将验证过的经济数据检索轨迹提炼成参数化技能库，证明技能迁移和基于库的检索能显著提升Web智能体的表现。 |
| [^179] | [Audio-Visual Turn-taking Prediction in Cocktail Party Scenarios](https://arxiv.org/abs/2609.17056) | 在鸡尾酒会等嘈杂含重叠语音的场景中，用干净数据训练的视听话轮转换预测模型性能显著下降（加权F1相对下降高达38%），微调虽可提升鲁棒性但收益因模态和预训练数据规模而异，凸显了鲁棒建模的必要性。 |
| [^180] | [Salesforce Koa: An Enterprise Language Model for Agentic Tool Use](https://arxiv.org/abs/2609.15066) | Salesforce Koa 是一个基于 Nemotron-3-Super-120B 并通过 GRPO 强化学习后训练的企业级语言模型，其核心创新在于“仿真到奖励”流水线——将工作流规范扩展为以角色为条件的多轮任务，并以成功工具使用作为任务解决奖励，从而在保持通用性能的同时显著提升智能体工具使用能力。 |
| [^181] | [Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification](https://arxiv.org/abs/2609.12544) | 该论文提出了Meddies-PII框架，包含一个通过属性条件提示生成、经十三个确定性门控验证、涵盖十七种语言的一百万份合成临床文档数据集，以及基于该数据集训练的BIOES分类模型，后者在十五个外部基准上以平均F1 0.827达到了现有PII提取系统中的最高性能。 |
| [^182] | [The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes](https://arxiv.org/abs/2609.10739) | 论文揭示了真值探测器的“完美混叠”失效机制——当真实汇报与任务既定行为在顺从情境中重合时探测器无法区分二者（二者AUROC恒互补求和为一），并提出在顺从与对立情境的混合数据上拟合的方法，使探测器即使在模型系统性说谎时也能以完美的1.0 AUROC识别真值。 |
| [^183] | [A Ticket from Marginals to Joints: Coupled-Noise Distillation for One-Step Block Generation in Diffusion Language Models](https://arxiv.org/abs/2609.06324) | 提出CONDOR方法，通过耦合噪声蒸馏从零训练扩散语言模型，使其在掩码嵌入受高斯噪声扰动时仅用一次前向传播即可生成连贯的整块文本，且无需目标侧编码器或自回归教师。 |
| [^184] | [Are Near-Tied LLM Rankings Robust to Family-DIF-Guided Benchmark Recomposition?](https://arxiv.org/abs/2609.00482) | 该论文提出一种基于无家族标签谱近似MIRT的基准重组方法，发现尽管全基准与低DIF排名强相关，但相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，表明排行榜上的微小差距并不稳健。 |
| [^185] | [CPR for LLMs: Critical-Point Routing against Catastrophic Forgetting in Domain Adaptation](https://arxiv.org/abs/2608.30158) | 本文提出CPR框架，通过识别基座模型失败而专家模型成功的临界词元，在基座模型与其SFT专家版本之间进行词元级路由，从模型层面解耦领域能力与通用能力，有效缓解领域自适应中的灾难性遗忘。 |
| [^186] | [Gaokerena: A Small Persian Medical Language Model Family](https://arxiv.org/abs/2608.00932) | 本文提出了Gaokerena，一个专为消费级硬件设计的小型波斯语医学语言模型家族，其中Gaokerena-V通过新构建的波斯语医学语料库训练提升了医学问答性能，Gaokerena-R则结合思维链与两个新型RLAIF框架来增强临床推理能力。 |
| [^187] | [SERL-SQL: Selective Hindsight Distillation for Text-to-SQL Reinforcement Agentic Learning](https://arxiv.org/abs/2608.00485) | SERL-SQL提出了一种以执行反馈为依据的选择性强化学习框架，通过仅在训练阶段使用的教师模型的后见之明重评分，将教师-学生似然差距转化为对GRPO优势的局部重加权，从而为多轮Text-to-SQL智能体实现更精细的信用分配。 |
| [^188] | [What Transfers from Text to Vision? Capability Scaling Laws and Transfer Dynamics for VLMs](https://arxiv.org/abs/2608.00013) | 我们提出了首个跨家族的多模态缩放规律，通过文本能力得分直接预测VLM性能，并在150多个VLM上验证了其有效性。 |
| [^189] | [GLM-RAG: Graph Language Models for Graph-Based Retrieval-Augmented Generation](https://arxiv.org/abs/2607.28397) | 本文提出了一种基于图语言模型（GLM）的检索器用于知识图谱检索增强生成，发现微调后的GLM检索器在跨领域泛化能力上优于GNN和向量搜索检索器，并在两个多跳基准上达到SOTA。 |
| [^190] | [Scaling Native Multimodal Pre-Training From Scratch](https://arxiv.org/abs/2607.22043) | 该研究首次系统刻画了原生多模态预训练的扩展规律，发现最小损失遵循可预测的计算定律，计算最优的模型规模与 token 数量呈幂律扩展，且语言与多模态目标的计算资源分配趋势截然不同。 |
| [^191] | [How Much Human Label Variation Does Formal Semantic Structure Explain?: Group-Level Effects and Item-Level Ceilings in NLI](https://arxiv.org/abs/2607.15870) | 该研究通过预注册分析直接测量发现，形式语义结构对自然语言推理中人类标签变异的解释力有限——群体层面上非纯向上单调假设的标签熵显著更高，但条目层面上形式语义特征仅能解释3.3%–3.6%的熵方差。 |
| [^192] | [The One-Word Census: Answer-Choice Conformity Across 44 Language Models](https://arxiv.org/abs/2607.12796) | 本研究通过96个开放式单词选择提示对105个语言模型进行了“答案趋同性”普查，发现模型的选择高度趋同——例如46%的模型在“选一个词”时都选了"serendipity"——且这种趋同并非由小型模型或个性化微调模型所致，反而轻度后训练和个性化微调的模型分歧最大。 |
| [^193] | [Introducing Human-Centeredness in AI-Assisted Lexicography](https://arxiv.org/abs/2607.11808) | 本文提出了一个以人为本的人工智能（HCAI）框架，主张AI应增强而非取代词典编纂者，并从增强型词典编纂者、社会技术背景、偏见和工具设计四个维度审视AI在词典编纂中的整合。 |
| [^194] | [System-Prompt Conditioning and Hidden-State Geometry in Four Open-Weight Models: Corrections and What Survives](https://arxiv.org/abs/2607.09842) | 本文是对先前“系统提示词会在开源语言模型隐藏状态中留下几何指纹”这一研究的勘误版：审计发现原论文的曲率统计量、置换检验方法和若干关键测量均存在错误，本版修正这些问题并说明哪些结论仍然成立。 |
| [^195] | [Black-Box Forensics for Conversational LLM Agents](https://arxiv.org/abs/2606.22698) | 该研究提出面向对话式LLM智能体的黑盒取证方法，无需模型参数访问即可在几轮对话内以98%准确率实现基础模型归因，并可通过系统提示词指纹识别将诈骗端点串联成犯罪网络、暴露隐蔽的API变更。 |
| [^196] | [Pretrained self-supervised speech models can recognize unseen consonants](https://arxiv.org/abs/2606.11542) | 本研究发现在高资源语言数据上预训练的自监督语音模型（Wav2Vec2 和 HuBERT），仅通过两种科伊桑语言的少量数据微调，就能比识别非搭嘴音更准确地识别训练中从未见过的搭嘴音，证明自监督学习具有跨人类语音的泛化能力。 |
| [^197] | [Learning to Attack and Defend: Adaptive Red Teaming of Language Models via GRPO](https://arxiv.org/abs/2606.09701) | 本文提出一种基于GRPO的攻击-防御协同训练框架，通过多LLM评判奖励通道与GDPO优势计算以及从攻击者单独训练到协同训练的课程策略，实现了高效可迁移的攻击生成并同步提升防御者的安全能力。 |
| [^198] | [LoRi: Low-Rank Distillation for Implicit Reasoning](https://arxiv.org/abs/2606.05315) | 该论文提出LoRi低秩蒸馏框架，利用隐状态推理轨迹的低秩结构，在共享低秩张量子空间中对齐教师与学生模型以迁移推理能力，使隐式推理性能接近显式思维链并超越现有iCoT蒸馏方法。 |
| [^199] | [EvoRubric: Self-Evolving Rubric-Driven RL for Open-Ended Generation](https://arxiv.org/abs/2605.29847) | 提出EvoRubric框架，让一个共享策略同时充当推理器和量规生成器，通过标准有效性反馈、响应判别与留一法同行共识实现自演化的量规驱动强化学习，从而解决开放式生成中奖励不可验证的难题。 |
| [^200] | [MemTrace: Tracing and Attributing Errors in Large Language Model Memory Systems](https://arxiv.org/abs/2605.28732) | 该论文提出了MemTrace框架，将LLM记忆流水线转化为可执行的记忆演化图，并结合MemTraceBench基准和自动归因方法，实现了对记忆系统错误的细粒度追踪与根因定位。 |
| [^201] | [Uncertainty-Aware Budget Allocation for Adaptive Test-Time Reasoning](https://arxiv.org/abs/2605.26849) | UAB通过投票熵度量问题难度，利用两阶段凹整数优化框架将固定采样预算自适应地集中分配给困难问题，从而比统一分配更高效地提升大语言模型的测试时推理性能。 |
| [^202] | [A Formative Study of Brief Affective Text as a Complement to Wearable Sensing for Longitudinal Student Health Monitoring](https://arxiv.org/abs/2605.14360) | 该研究表明，学生每两个月仅需输入约三个词的简短担忧文本，经NLP模型分析后即可作为可穿戴设备被动传感的可扩展补充，有效用于睡眠与身体活动的纵向健康监测。 |
| [^203] | [CiteVQA: Benchmarking Evidence Attribution for Trustworthy Document Intelligence](https://arxiv.org/abs/2605.12882) | 提出CiteVQA基准，要求多模态大语言模型在回答文档问题时同时提供元素级边界框证据引用，从而联合评估答案正确性与证据归因的可信度。 |
| [^204] | [SkillGraph: Skill-Augmented Reinforcement Learning for Agents via Evolving Skill Graphs](https://arxiv.org/abs/2605.12039) | 该论文提出SkillGraph框架，将智能体的可复用技能表示为带类型边的演化有向图，通过检索有序的技能子图并利用强化学习反馈持续更新图结构，从而有效支持组合性任务的多步决策与技能库维护。 |
| [^205] | [How Much Do Circuits Tell Us? Measuring the Consistency and Specificity of Language Model Circuits](https://arxiv.org/abs/2605.08348) | 本文提出用一致性和特异性两个新属性来评估机制可解释性中的电路，发现组件级电路虽高度一致且因果重要但不具任务特异性，而神经元级电路虽具任务特异性却一致性较差。 |
| [^206] | [Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes](https://arxiv.org/abs/2605.06152) | 本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。 |
| [^207] | [State Stream Transformer (SST) V2: Parallel Training of Nonlinear Recurrence for Latent Space Reasoning](https://arxiv.org/abs/2605.00206) | SST V2通过在每层引入FFN驱动的非线性递归，使潜在状态横向流经整个序列，实现连续潜在空间中的参数高效推理与深思，并采用两遍并行训练使其计算上可行。 |
| [^208] | [AI Appeals Processor: A Deep Learning Approach to Automated Classification of Citizen Appeals in Government Services](https://arxiv.org/abs/2604.03672) | 该论文提出的AI申诉处理器在10,000条真实俄语公民申诉上对比多种深度学习模型，发现Word2Vec+LSTM（78%）接近多语言BERT（82%）且远超人工操作员（67%），并因训练成本更低、便于在仅CPU的政府环境中频繁再训练而选择部署LSTM。 |
| [^209] | [Limited Stereotype Control Through Routing Reweighting in MoE Language Models](https://arxiv.org/abs/2603.27141) | 本文提出FARE诊断框架，发现虽然人口统计学提示词在MoE模型中的路由方式与中性提示词存在差异，但推理时的路由重加权对刻板印象的控制效果十分有限（偏好变化最多仅约1.3–1.5个百分点）。 |
| [^210] | [Imperative Interference: Social Register Shapes Instruction Topology in Large Language Models](https://arxiv.org/abs/2603.25015) | 该研究发现大语言模型会按照社会语域惯例来处理系统提示指令——同一指令在英语中协作而在西班牙语中竞争，将祈使句改写为陈述句可降低81%的跨语言差异，说明模型把指令理解为社会行为而非技术规范。 |
| [^211] | [Beyond Preset Identities: Selective Stance Accommodation and Interaction Reorganisation in Generative Agent Societies](https://arxiv.org/abs/2603.23406) | 该研究超越仅测量立场变化程度的传统方法，通过结合立场测量、来源评估与时序网络分析，揭示生成式智能体社会中智能体接受立场变化的实质内容以及强化特定伙伴关系的互动重组机制。 |
| [^212] | [Agentic Critical Training](https://arxiv.org/abs/2603.08706) | 该论文提出智能体批判性训练，利用可验证奖励的强化学习让模型学会直接区分专家动作与看似合理的错误动作，作为模仿学习前的高效热身方法，无需参考推理且支持数据跨模型复用。 |
| [^213] | [\$OneMillion-Bench: How Far are Language Agents from Human Experts?](https://arxiv.org/abs/2603.07980) | 提出了包含 400 个专家设计的跨领域专业任务的 \$OneMillion-Bench 基准，通过基于评分规则的多维度评估，衡量语言智能体在法律、金融、医疗等经济关键场景中与人类专家的差距。 |
| [^214] | [Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching](https://arxiv.org/abs/2602.22871) | 该论文提出Stitching Noisy Diffusion Thoughts框架，利用过程奖励模型对扩散语言模型采样的多条推理轨迹的中间步骤进行评分，并将跨轨迹的最优步骤拼接成复合推理链，实现了步骤级而非轨迹级的中间推理成果复用与测试时扩展。 |
| [^215] | [Quantifying Retriever-Generator Alignment in RAG with Local Explanations](https://arxiv.org/abs/2601.21803) | 该论文提出端到端可解释框架RAG-E，通过适配的积分梯度、蒙特卡洛稳定的Shapley值归因以及新颖的WARG对齐指标，量化RAG系统中检索器与生成器的对齐程度，实验揭示生成器常常忽略排名靠前的文档而依赖相关性较低的文档。 |
| [^216] | [Is Peer Review Really in Decline? Analyzing Review Quality across Venues and Time](https://arxiv.org/abs/2601.15172) | 该研究提出了一个基于证据的评审质量比较分析框架，结合LLM与轻量级度量方法对ICLR、NeurIPS和*ACL三大会议的评审质量进行跨会议、跨时间研究，其发现与“评审质量正在下降”的流行观点相反。 |
| [^217] | [Obscuring Data Contamination Through Translation: Evidence from Arabic Corpora](https://arxiv.org/abs/2601.14994) | 该研究表明，将 MMLU 和 XQuAD 基准翻译成阿拉伯语后暴露给模型，既能提升模型在英语基准上的表现，又能使 TS-Guessing 和 Min-K%++ 等现有污染检测方法失效，说明跨语言方式可以掩盖数据污染。 |
| [^218] | [LMSpell: Spell Correction with Pre-Trained Language Models](https://arxiv.org/abs/2512.05414) | 该研究首次系统比较了三类预训练语言模型在多语言（含低资源语言）拼写纠错中的效果，证明即使是2.7亿参数的小型模型仅在5千个句子上微调，也能超越基于规则的拼写纠错方法。 |
| [^219] | [Towards Scalable Meta-Learning of near-optimal Interpretable Models via Synthetic Model Generations](https://arxiv.org/abs/2511.04000) | 本文提出通过合成采样近最优决策树来生成大规模预训练数据的高效可扩展方法，使MetaTree transformer在决策树元学习上达到与真实数据或昂贵最优树预训练相当的性能，同时大幅降低计算成本。 |
| [^220] | [AyurParam: A State-of-the-Art Bilingual Language Model for Ayurveda](https://arxiv.org/abs/2511.02374) | 研究者推出了面向阿育吠陀传统医学的领域专精双语模型 AyurParam-2.9B，通过专家精心整理的英印双语数据集进行微调，在 BhashaBench-Ayur 基准上超越了同规模（1.5B–3B）的所有开源指令微调模型。 |
| [^221] | [More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models](https://arxiv.org/abs/2510.04532) | 本研究构建了包含规划对齐思维链的DriveMind驾驶视觉问答数据集，通过信息消融实验首次系统揭示了视觉-语言驾驶模型中自然语言推理与轨迹规划之间的因果脱节，即规划性能主要依赖先验信息而非语言推理。 |
| [^222] | [GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness](https://arxiv.org/abs/2510.00536) | 该论文提出了GUI-KV，一种即插即用的KV缓存压缩方法，通过利用GUI的空间和时间冗余特性以及跨层均匀的缓存预算分配策略，显著提升了GUI智能体的推理效率。 |
| [^223] | [Fair-GPTQ: Bias-Aware Quantization for Large Language Models](https://arxiv.org/abs/2509.15206) | Fair-GPTQ是首个显式以减少不公平性为目标的量化方法，它在量化目标中加入群体公平性约束，引导舍入操作学习，从而在几乎不损失性能的前提下，降低大模型生成文本中涉及性别、种族和宗教的偏见与歧视。 |
| [^224] | [An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems](https://arxiv.org/abs/2508.08833) | 提出 GAP 方法，通过表面重命名与核心重写两种数学等价变换自动批量生成现有数学题的等价变体，以评估大语言模型数学推理的鲁棒性并诊断其失败环节。 |
| [^225] | [Optimal Transport Depth Up-Scaling](https://arxiv.org/abs/2508.08011) | 提出基于最优传输理论的深度扩展方法OT-DUS，通过逐模块对齐并融合相邻基础层中功能对应的神经元来构建新层，避免了传统复制或平均方法导致的神经元排列不匹配问题，在持续预训练和监督微调中均取得更优性能。 |
| [^226] | [InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models](https://arxiv.org/abs/2505.13878) | InfiFPO通过在DPO中用序列层面综合多源概率的融合源模型替换参考模型，实现了无需复杂词表对齐且保留概率信息的大语言模型隐式融合偏好优化方法。 |
| [^227] | [Foundations of Large Language Models](https://arxiv.org/abs/2501.09223) | 本书系统阐述了大语言模型的六大核心基础领域——预训练、生成模型、提示、对齐、推断与推理，为学习者提供了一部权威的基础性参考书。 |
| [^228] | [Thought-Like-Pro: Enhancing Reasoning of Large Language Models through Self-Bootstrapped Prolog-based Chain-of-Thought](https://arxiv.org/abs/2407.14562) | 该论文提出Thought-Like-Pro框架，利用模仿学习让大语言模型模仿由Prolog逻辑引擎生成并经验证的思维链推理过程，以提示引导且自举的方式增强模型在多种推理任务上的推理能力与泛化性。 |
| [^229] | [Policy Learning with a Language Bottleneck](https://arxiv.org/abs/2405.04118) | 该论文提出PLLB框架，让AI智能体在语言模型引导的“规则生成”与规则引导的“策略更新”之间交替进行，通过语言瓶颈捕捉行为背后的高层策略，从而学习到更可解释、更可泛化的行为。 |
| [^230] | [Enabling Quantum Natural Language Processing for Hindi Language](https://arxiv.org/abs/2312.01221) | 该论文首次将量子自然语言处理方法扩展到印地语，通过预群表示、DisCoCat框架和IQP风格拟设构建参数化量子电路，实现了面向印地语的语法和主题感知句子分类器。 |

# 详细

[^1]: FastBench：流式视觉语言模型能否感知高动态的现实世界视频流？

    FastBench: Can Streaming VLMs Perceive High-Dynamic Real-World Streams?

    [https://arxiv.org/abs/2610.12427](https://arxiv.org/abs/2610.12427)

    该论文提出 FastBench 基准，通过基于轨迹的自动化问答生成与验证流水线（结合 SAM3/CoTracker3 轨迹验证和人工审核）评估流式 VLM 在高动态真实世界视频流中的感知能力，并提出无需训练的 ProactiveFrame 基线，通过文本词元动态调节输入帧率以捕捉快速事件。

    

    流式视频大语言模型（VLMs）能够实现连续的视频理解，然而现有基准测试主要聚焦于低动态场景。在有限的上下文预算下，模型必须权衡时间历史、空间分辨率与时间粒度；1–2 FPS 的稀疏采样会遗漏快速发生的事件。我们提出 FastBench，用于评估真实世界视频流中的高动态感知能力。其基于轨迹的流水线融合了以下环节：从高帧率视频片段生成问答对、过滤掉 2 FPS 下即可回答的问题、利用 SAM3 和 CoTracker3 的轨迹进行答案验证，以及三轮人工审核。FastBench 包含 306 个问答对，涵盖八个领域、六种能力，以及前向、即时和回溯三种时间范围，并附带由人工标注的证据区间。我们还提出了 ProactiveFrame，一种无需训练的基线方法，通过文本词元动态调整输入帧率，并采用双层滑动窗口保留近期的高帧率观测。

    arXiv:2610.12427v1 Announce Type: cross  Abstract: Streaming Video Large Language Models (VLMs) enable continuous video understanding, yet existing benchmarks focus on low-dynamic scenarios. Under bounded context budgets, models must balance temporal history, spatial resolution, and temporal granularity; sparse sampling at 1--2 FPS misses fast events. We introduce FastBench to evaluate high-dynamic perception in real-world video streams. Its trajectory-grounded pipeline combines QA generation from high-FPS clips, filtering of questions answerable at 2 FPS, answer verification using SAM3 and CoTracker3 trajectories, and three rounds of human inspection. FastBench contains 306 QA pairs across eight domains, six capabilities, and forward, instant, and backward temporal scopes, with human-annotated evidence intervals. We also present ProactiveFrame, a training-free baseline that adjusts incoming frame rates through text tokens. A dual-tier sliding window retains recent high-FPS observation
    
[^2]: WOVEN：将视觉世界建模编织进多模态大语言模型

    WOVEN: Weaving Visual World Modeling into Multimodal LLMs

    [https://arxiv.org/abs/2610.12417](https://arxiv.org/abs/2610.12417)

    提出WOVEN——一个按场景、动作和推理类型组织的视觉转换推理训练数据源与基准（含36,076个示例），验证视觉转换推理可作为可跨任务复用的共享训练原语，以提升多模态大语言模型的空间、具身、物理和时间推理能力。

    

    多模态大语言模型（MLLM）在空间、具身、物理和时间推理方面表现不佳。我们假设这些失败反映了一个共同的缺陷——视觉转换推理能力的不足，并测试这一能力是否可以作为共享的训练原语，即不同模型可以从不同的监督来源中学习该能力，并在不同任务间复用，同时我们提供了系统化的训练方案。现有的基准测试只是分别记录这些缺陷，但无法支持跨场景、动作和推理操作的可控比较。因此，我们提出了WOVEN，一个面向视觉转换推理的训练数据源与基准，它按照场景、动作和推理类型组织转换监督信息，使用来自视频预训练生成模型的多样化、逼真的rollout数据：涵盖20种场景类型、5种动作类型和8种推理类型，共计36,076个示例。我们首先评估了38个前沿的多模态大语言模型（如GPT-5.4和Qwen3-VL等）。

    arXiv:2610.12417v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) struggle with spatial, embodied, physical, and temporal reasoning. We hypothesize that these failures reflect a shared deficit in visual transition reasoning, and test whether this capability can serve as a shared training primitive, one that different models can learn from different supervision sources and reuse across different tasks, with a systematic training recipe. Existing benchmarks document these deficits separately but do not support controlled comparisons across scenes, actions, and reasoning operations. We therefore introduce WOVEN, a training source and benchmark for visual transition reasoning that organizes transition supervision by scene, action, and reasoning type, using diverse, realistic rollouts from video-pretrained generative models: 36,076 examples across 20 scene types, 5 action types, and 8 reasoning types. We first evaluate 38 frontier MLLMs (e.g., GPT-5.4 and Qwen3-VL-
    
[^3]: 基于价值表征预测对齐泛化

    Predicting Alignment Generalization with Value Representations

    [https://arxiv.org/abs/2610.12410](https://arxiv.org/abs/2610.12410)

    本文提出“对齐泛化预测”新任务，通过对66个价值的大规模分析发现，基于模型激活的价值表征能够显著优于基于文本描述的方法，预测模型微调遵循某一价值后在未见价值上的行为泛化。

    

    LLM开发者通过后训练使模型展现亲社会的价值与行为特质，这些价值与特质被列举在对齐目标中。然而，尽管近期的后训练进展使模型在对齐评估中获得了高分，基于狭窄行为集合训练模型仍会以意想不到的方式影响模型在未见过的情境和环境中的行为。本文提出了“对齐泛化预测”这一新任务，即预测将模型微调为遵循某一给定价值后，其行为在一系列保留价值上将如何变化。我们对现代对齐目标中的66个价值进行了大规模的对齐泛化效应分析，并在对齐泛化预测任务上对多种表征技术进行了基准测试。研究发现，基于模型在上下文中应用价值时的激活的表征方法，显著优于基于文本描述的方法。

    arXiv:2610.12410v1 Announce Type: cross  Abstract: LLM developers post-train their models to exhibit prosocial values and behavioral traits, which are enumerated in an alignment target. However, while recent post-training developments have yielded models that score highly on alignment evaluations, training models on sets of narrow behaviors still influences their behavior across unseen contexts and environments in unexpected ways. In this paper, we establish the task of alignment generalization prediction, i.e., predicting how fine-tuning a model to follow a given value changes its behavior across a wide range of held-out values. We conduct a large-scale analysis of alignment generalization effects across 66 values found in modern alignment targets, and benchmark representational techniques on the alignment generalization prediction task. We find that representations based on model activations when applying values in context significantly outperform methods based on textual description
    
[^4]: ViSkill：通过演化的视觉原生技能强化视觉语言模型智能体

    ViSkill: Reinforcing VLM Agents with Evolving Visual-Native Skills

    [https://arxiv.org/abs/2610.12403](https://arxiv.org/abs/2610.12403)

    提出ViSkill视觉原生技能学习框架，将成功交互编码为可检索的视觉技能卡片，在推理和奖励塑形中发挥作用，并通过成功轨迹回流技能库形成技能积累与策略改进相互促进的闭环，显著提升视觉语言模型智能体的样本效率。

    

    技能增强的智能体通过将成功轨迹提炼为可复用的策略来提高样本效率。然而，大多数现有方法仍以文本为中心，将空间布局和动作-状态对应关系线性化为语言，从而丢失了关键的几何结构。最近的一些工作开始引入视觉证据，但其技能的构建和更新与策略优化是相互分离的，使二者的协同提升尚未得到充分探索。我们提出了ViSkill，一个视觉原生的技能学习框架，它将成功的交互编码为视觉语言模型智能体可直接访问的复合视觉技能卡片。检索到的技能同时指导推理和奖励塑形，而成功的轨迹则被提炼回技能库中，形成一个闭环反馈，使技能积累与策略改进相互增强。一个可选的冷启动机制进一步加速了早期阶段的学习。该方法在Sokoban（推箱子）和FrozenLake（冰湖）等环境上进行了评估。

    arXiv:2610.12403v1 Announce Type: cross  Abstract: Skill-augmented agents improve sample efficiency by distilling successful trajectories into reusable strategies. Yet most existing approaches remain text-centric, linearizing spatial layouts and action-state correspondences into language that loses critical geometric structure. Recent efforts have begun incorporating visual evidence, but construct and update skills separately from policy optimization, leaving their mutual improvement underexplored. We propose ViSkill, a visual-native skill learning framework that encodes successful interactions as composite visual skill cards directly accessible to VLM agents. Retrieved skills guide both inference and reward shaping, while successful trajectories are distilled back into the library, forming a closed feedback loop in which skill accumulation and policy improvement reinforce each other. An optional cold-start mechanism further accelerates early-stage learning. Evaluated on Sokoban, Froze
    
[^5]: SpaceCast-Bench：评估视觉-语言模型中的预测性空间推理

    SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in Vision-Language Models

    [https://arxiv.org/abs/2610.12402](https://arxiv.org/abs/2610.12402)

    SpaceCast-Bench是首个通过“观察-变换-推断”框架直接且诊断性地评估视觉-语言模型预测性空间推理能力的基准，其评估揭示了最强模型（58.0%）与人类表现（87.2%）之间的显著差距。

    

    现有的空间推理基准主要测试空间感知：即从输入中读取已经可见的关系。然而，现实世界的空间智能需要预测性空间推理：从观察中构建场景、预测干预如何改变场景、并对未见到的结果进行推理。我们提出了SpaceCast-Bench，这是首个直接且诊断性地评估这一能力的基准。该基准基于“观察-变换-推断”框架构建，包含来自182个真实世界场景的3,862个问题，涵盖三个层次的16种任务类型：静态感知、局部预测和全局预测，逐步要求场景理解、空间状态更新以及对未观察到的结果进行关系推理。对21个模型的评估揭示了一个显著的差距：最强模型仅达到58.0%，而人类表现为87.2%，同时空间专用模型仍接近随机水平。受控分析进一步表明...

    arXiv:2610.12402v1 Announce Type: cross  Abstract: Existing spatial reasoning benchmarks mainly test spatial perception: reading off relations already visible in the input. Yet real-world spatial intelligence demands predictive spatial reasoning: constructing a scene from observations, anticipating how an intervention changes it, and reasoning about the unseen outcome. We introduce SpaceCast-Bench, the first benchmark to directly and diagnostically evaluate this capability. Built around an observe-transform-infer framework, its 3,862 questions from 182 real-world scenes span 16 task types at three levels: static perception, local prediction, and global prediction, progressively requiring scene understanding, spatial state updating, and relational inference over unobserved outcomes. Evaluating 21 models exposes a stark gap: the strongest model reaches only 58.0% against 87.2% human performance, while spatially specialized models remain near random chance. Controlled analyses further rev
    
[^6]: 从长文本到预测特征：基于可执行程序搜索的LLM引导分块式特征工程

    Long Text to Predictive Features: LLM-Guided Blockwise Feature Engineering via Executable Program Search

    [https://arxiv.org/abs/2610.12390](https://arxiv.org/abs/2610.12390)

    提出LLM-BlockFE框架，由LLM在离线阶段通过逐步追加代码块并结合深度校准信用分配的分块级回滚搜索，将长文本自动转化为可执行的特征程序，使在线推理无需调用LLM即可高效利用长文本信息。

    

    工业风控系统通常依赖结构化数据模型进行高效预测，然而大量有价值的信息仍然蕴含在非结构化的长文本中。通过人工特征工程提取这些信息费时费力，而依赖大语言模型（LLM）处理每一次实时输入又可能无法满足实际部署需求。为应对这一挑战，我们提出了LLM-BlockFE，这是一个由LLM引导的离线特征构建框架，可将长文本转换为可执行的特征程序，从而避免在线推理阶段调用LLM。LLM-BlockFE通过逐步追加不可变的代码块来构建特征程序，并利用下游模型评估候选特征。为解决传统贪心搜索容易陷入次优解的问题，我们的方法引入了一种基于深度校准信用分配的分块级回滚机制（摘要在此处截断）

    arXiv:2610.12390v1 Announce Type: cross  Abstract: Industrial risk-control systems typically rely on structured-data models for efficient prediction, yet substantial valuable information remains embedded in unstructured long text. Extracting this information through manual feature engineering is labor-intensive, while requiring a large language model (LLM) to process every real-time input may not meet practical deployment requirements. To address this challenge, we propose LLM-BlockFE, an LLM-guided offline feature construction framework that converts long text into executable feature programs, thereby avoiding LLM calls during online inference. LLM-BlockFE constructs feature programs by incrementally appending immutable code blocks and evaluates candidate features using a downstream model. To address the tendency of conventional greedy search to become trapped in suboptimal solutions, our method introduces a block-level rollback mechanism based on depth-calibrated credit allocation an
    
[^7]: 潜在核心分词器：压缩，但要有意义

    Latent Core Tokenizer: Compress, but Meaningfully

    [https://arxiv.org/abs/2610.12376](https://arxiv.org/abs/2610.12376)

    潜在核心分词器（LCT）在构建词表前先利用最小描述长度、熵边界信号和形态结构约束识别可复用的语言单元，在104种语言上实现了比BPE等方法更低的碎片率和更高的多语言下游任务性能，证明压缩率本身并不能预测表示质量。

    

    分词器通常以压缩率为优化目标，但紧凑的词表并不一定能在各语言之间均匀分配其容量。我们提出了潜在核心分词器（LCT），这是一种与语言无关的方法，它将结构发现与词表构建相分离。LCT 利用最小描述长度、基于熵的边界信号和形态结构约束，在构建共享词表之前识别可复用的语言单元。在涵盖 104 种语言、词表大小为 20 万 token 的设置下，LCT 相比 BPE、Unigram 和感知公平性的 BPE 实现了更低的分词碎片率和更高的 MorphScore，同时在分词成本上保持了相当的跨语言差异水平。在四个多语言下游基准测试中，LCT 的总分分别比 BPE、Unigram 和 parity-aware BPE 提高了 1.48、1.83 和 2.00 分。我们的研究结果表明，仅靠压缩并不能预测表示质量，并强调了……（原文摘要在此截断）

    arXiv:2610.12376v1 Announce Type: new  Abstract: Tokenizers are commonly optimized for compression, but a compact vocabulary does not necessarily distribute its capacity evenly across languages. We introduce the Latent Core Tokenizer (LCT), a language-agnostic approach that separates structural discovery from vocabulary construction. LCT uses Minimum Description Length, entropy-based boundary signals, and morphotactic constraints to identify reusable linguistic units before constructing a shared vocabulary. Across 104 languages with a 200K-token vocabulary, LCT achieves lower fertility and higher MorphScore than BPE, Unigram, and parity-aware BPE, while maintaining comparable cross-lingual disparity in tokenization cost. Across four multilingual downstream benchmarks, LCT improves aggregate score by 1.48, 1.83, and 2.00 points over BPE, Unigram, and parity-aware BPE, respectively. Our findings show that compression alone does not predict representation quality and highlight the importa
    
[^8]: OnTrack：基于流式结构感知最优传输的LLM智能体轨迹实时监控与干预

    OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport

    [https://arxiv.org/abs/2610.12375](https://arxiv.org/abs/2610.12375)

    OnTrack提出了一种流式结构感知最优传输监控机制，通过将LLM智能体的执行步骤与记录的成功运行轨迹实时对比，在每步约一毫秒内实现对异常行为的告警或阻断，兼顾了低延迟与安全性。

    

    智能体被部署于从行程规划、股票交易到IT事件分诊等各类应用中。在大多数情况下，LLM智能体在仅有极少基于规则的安全保障下自主运行，导致不可逆操作带来成本与安全问题。近期的工作要么通过使用一个安全防护智能体来监控行为，要么在事后评估日志：前者为每一步增加了成本和延迟；后者则在运行结束后才给出判定，此时令牌已被消耗、损害已经造成。为克服这些问题，我们提出OnTrack，一种流式监控机制，它将智能体的步骤及其依赖关系与记录下来的成功运行轨迹进行对比，从而在大约每步一毫秒的时间内向用户告警或阻断智能体。我们在三种访问程度递减的场景下研究该问题：完全参考访问（历史运行记录与工具模式）、中间访问（仅有工具模式）以及无先验知识（仅有实时生成的步骤日志）。

    arXiv:2610.12375v1 Announce Type: new  Abstract: Agents are deployed in applications from trip planners and stock trading to IT incident triage. In most cases, LLM agents work autonomously with minimal rule-based safeguarding, leading to cost and safety issues from irreversible actions. Recent works resolve this either by using a safeguard agent to monitor behavior or evaluating logs post-hoc. The first adds cost and latency to every step; the second delivers its verdict after the run, when tokens are burned and damage is done. To overcome this, we propose OnTrack, a streaming monitoring mechanism that compares an agent's steps and dependencies against recorded successful runs to alert users or block the agent in about a millisecond per step. We study this problem in three regimes of decreasing access: full reference access (historical runs and tool schemas), intermediate access (only tool schemas), and no prior knowledge (only step logs as generated). Expectation of OnTrack's monitori
    
[^9]: 蒸馏哪些技能？SGUID：为模型-技能协同进化选择紧凑的技能库

    Which Skill to Distill? SGUID: Selecting a Compact Skill Bank for Model-Skill Co-Evolution

    [https://arxiv.org/abs/2610.12367](https://arxiv.org/abs/2610.12367)

    提出SGUID方法，仅保留训练中持续产生有效学习信号的核心技能用于蒸馏，证明精选的6个技能即可匹配或超越全技能库蒸馏的性能。

    

    技能是在推理阶段注入的可复用程序性指导，能够显著提升大语言模型（LLM）的下游性能。先前的工作通过语义相关性从技能库中检索技能，然后将其用作推理时的补丁或用于模型蒸馏。然而，每个技能的个体效用在很大程度上被忽视了。我们首先证明，在以技能条件化策略作为教师的在线策略蒸馏中，检索到的技能中不到25%能够提供有用的蒸馏信号。随后我们提出SGUID，一种用于选择紧凑技能子集进行蒸馏的方法。SGUID仅在某个技能在训练过程中持续产生有效学习信号时才保留该技能，选定的技能随后被蒸馏以得到更好的模型。我们的结果表明，并非所有技能都值得蒸馏。在来自Olmo和Qwen系列的四个模型上，仅蒸馏6个选定技能在平均avg@12指标上即可匹敌或超越全库蒸馏的表现。

    arXiv:2610.12367v1 Announce Type: new  Abstract: Skills, reusable procedural guidance added at inference, can substantially improve LLM downstream performance (Li et al., 2026). Prior work retrieves skills from a bank by semantic relevance, then uses them as inference-time patches or for model distillation. The individual utility of each skill, however, is largely neglected. We first show that, in on-policy distillation where skill-conditioned policies serve as teachers, fewer than 25% of retrieved skills provide useful distillation signals. We then propose SGUID, a method for selecting a compact subset of skills for distillation. SGUID retains a skill only if it consistently yields effective learning signals during training. The selected skills are then distilled to produce a better model. Our results show that not all skills are worth distilling. Across four models from the Olmo and Qwen families, distilling 6 selected skills matches or exceeds full-bank distillation in mean avg@12 o
    
[^10]: 被引用却未被征询：法律思维链忠实性的反事实审计

    Cited but Not Consulted: A Counterfactual Audit of Legal Chain-of-Thought Faithfulness

    [https://arxiv.org/abs/2610.12361](https://arxiv.org/abs/2610.12361)

    该论文通过反事实审计方法发现，大语言模型在法律推理中虽能正确引用法规或判例，但其裁决结果往往并不真正依赖于所引用的法律依据，揭示了法律思维链的引用忠实性缺失问题。

    

    大型语言模型越来越多地通过指出裁决背后的法规或判例来论证法律决策的合理性，这被视为决策确实源自该法律依据的证据。我们对此进行了直接检验：在保持案件事实不变的情况下，将模型所引用的法律依据替换为无关的依据，并从模型隐藏状态中解码其不断变化的裁决。在七个开放权重模型（8B-70B）和四个涵盖司法与合同推理的基准测试中，当明确要求模型通过指出适用的法律依据来论证裁决时，模型在66.7%-100%的生成中能正确指出该依据；然而，当法律依据被替换后裁决随之改变的情况却远不一致：在CaseHOLD上仅为0.0%-21.7%，在ECHR和SCOTUS上为30.0%-76.7%，在ContractNLI上为43.3%-50.0%。无论是扩大模型规模，还是使用专门构建的法律推理模型（尽力复现的LoRA版本；见第6节），都无法弥合这一差距。在五个核心模型上进行的红队评估发现，模型会遵从对抗性……（原文摘要在此处中断）

    arXiv:2610.12361v1 Announce Type: new  Abstract: Large language models increasingly justify legal decisions by naming the statute or precedent behind a verdict, treated as evidence that the decision follows from it. We test this directly: holding case facts fixed, we substitute the named legal authority for an unrelated one and decode a model's evolving verdict from its hidden states. Across seven open-weight models (8B-70B) and four benchmarks spanning judicial and contractual reasoning, when explicitly required to justify a verdict by naming the governing authority, models name the correct one in 66.7%-100% of generations, while the verdict changing when the authority changes is far less consistent: 0.0%-21.7% on CaseHOLD, 30.0%-76.7% on ECHR and SCOTUS, and 43.3%-50.0% on ContractNLI. Neither scale nor a purpose-built legal-reasoning model (a best-effort LoRA reproduction; Section 6) closes this gap. A red-teaming evaluation on five core models finds compliance with an adversarial i
    
[^11]: 准确但不谦逊：评估知识冲突下大语言模型智能体的认知谦逊

    Accurate but Not Humble: Evaluating Epistemic Humility in LLM Agents under Knowledge Conflict

    [https://arxiv.org/abs/2610.12360](https://arxiv.org/abs/2610.12360)

    该论文提出用“认知谦逊”概念（通过识别、解决、上报三个行为维度）评估大语言模型智能体在知识冲突下承认并传达不确定性的能力，发现智能体虽然准确但不谦逊，往往坚持错误结论而不愿承认不确定性。

    

    arXiv:2610.12360v1 公告类型：新论文。摘要：当检索到的证据与智能体的先验信念相矛盾时，它会修改自己的答案、承认不确定性，还是坚持错误的结论？现有对智能体系统的评估主要聚焦于任务成功率，对于智能体如何处理此类冲突所提供的洞察十分有限。我们提出从认知谦逊的角度评估智能体：即智能体在任务执行过程中识别、应对并传达不确定性的意愿。我们通过三个轨迹层面的行为维度来操作化认知谦逊：识别、解决和上报。借助知识冲突——即骨干语言模型的参数化知识与其遇到的证据相矛盾，或两个上下文来源互不一致的情形——我们评估了两种冲突设置：（1）受控冲突，以及（2）多步智能体执行过程中自然发生的冲突，且每种设置均配有匹配的无冲突对照组。在评估四个智能体后，我们发现……

    arXiv:2610.12360v1 Announce Type: new  Abstract: When retrieved evidence contradicts an agent's prior beliefs, does it revise its answer, acknowledge uncertainty, or persist with an incorrect conclusion? Existing evaluations of agentic systems focus primarily on task success, offering limited insight into how agents handle such conflicts. We propose to evaluate agents on epistemic humility (EH): the agent's willingness to recognize, act on, and communicate uncertainty during task execution. We operationalize EH through three trajectory-level behavioral dimensions: Identify, Solve, and Escalate (ISE). Through knowledge conflict, situations where the backbone language model's parametric knowledge contradicts the evidence it encounters, or where two contextual sources disagree, we evaluate two conflict settings: (1) controlled conflict and (2) naturally occurring conflict during multi-step agentic execution, each paired with matched no-conflict controls. Evaluating four agents, we find th
    
[^12]: 克服先验壁垒：长尾分布下的监督微调

    Overcoming Prior Barriers: Supervised Fine-Tuning under Long-Tail Distribution

    [https://arxiv.org/abs/2610.12345](https://arxiv.org/abs/2610.12345)

    该论文提出“先验壁垒”新概念，揭示预训练模型对各概念的支持程度呈长尾分布、使头部与尾部概念在监督微调时起点不同，并通过理论推导预测风险界，指出尾部概念需要额外指令才能克服高先验壁垒。

    

    监督微调（SFT）用于将预训练大语言模型（LLM）适配到下游任务，但任务所需的概念在预训练阶段可能获得截然不同的支持程度。高频概念更容易被充分学习，而低频概念可能仍然表征薄弱。我们提出了一个名为“先验壁垒”的新概念，用以量化预训练模型对竞争概念相对于目标概念的支持强度。我们观察到先验壁垒呈长尾分布，使得头部概念与尾部概念在SFT时处于不同的起点：头部概念面临较低的先验壁垒，而尾部概念则需要额外的指令来克服其较高的先验壁垒。我们的理论分析进一步推导了长尾先验壁垒下SFT的预测风险界，明确刻画了先验壁垒与累积的SFT证据如何共同决定预测性能。受此启发……（原文摘要在此处被截断）

    arXiv:2610.12345v1 Announce Type: new  Abstract: Supervised fine-tuning (SFT) adapts pretrained large language models (LLMs) to downstream tasks, but the required concepts can receive substantially different levels of pretrained support. Frequent concepts are more likely to be well learned, whereas rare concepts may remain weakly represented. We introduce a novel notion named prior barrier to quantify how strongly the pretrained model supports competing concepts over the target concept. We observe that prior barriers follow a long-tail distribution, placing head and tail concepts at different starting points for SFT: head concepts face lower prior barriers, whereas tail concepts require additional instructions to overcome their higher prior barriers. Our theoretical analysis further derives a predictive risk bound for SFT under long-tail prior barriers, explicitly characterizing how the prior barrier and accumulated SFT evidence jointly determine predictive performance. Motivated by th
    
[^13]: AI智能体能一路学习登顶吗？在长期运行的游戏智能体竞赛中评估启发式学习

    Can AI Agents Learn Their Way to the Top? Evaluating Heuristic Learning in a Long-Running Game Agent Competition

    [https://arxiv.org/abs/2610.12341](https://arxiv.org/abs/2610.12341)

    论文提出对抗性启发式学习范式与AAArena基准，让模型权重固定不变的AI智能体通过解读规则、分析回放和修订可执行游戏策略，在12个真实对抗性游戏竞赛中学习并争夺最高排名。

    

    对抗性游戏推动了从启发式搜索到强化学习的技术进步，然而从有限样本中学习和调整策略仍然充满挑战。AI智能体提供了一种替代方案，即将游戏经验转化为对可执行策略的修订。在启发式学习的基础上，我们形式化了对抗性启发式学习，这是一种以AI智能体作为学习引擎来改进游戏策略及其配套软件、同时保持模型权重固定的范式。我们提出了AAArena，一个包含12个真实对抗性游戏和1,920个存档人类程序的基准，其评估协议仿照现实世界的游戏竞赛设计。智能体需要解读游戏规则、选择对手、分析对局回放，并在固定的比赛与评估预算内通过修改游戏智能体来争取最高排名。我们评估了多个模型和框架配置：Opus5.5配合Claude Code赢得了6枚金牌，而没有任何评估……

    arXiv:2610.12341v1 Announce Type: new  Abstract: Adversarial games have driven advances from heuristic search to reinforcement learning, yet learning and adapting strategies from limited samples remain challenging. AI agents offer an alternative by turning game experience into revisions of executable policies. Building on heuristic learning (HL), we formalize Adversarial Heuristic Learning (AHL), a paradigm that uses AI agents as learning engines to refine game policies and supporting software while keeping model weights fixed. We introduce AAArena, a benchmark comprising 12 authentic adversarial games and 1,920 archived human programs, with an evaluation protocol modeled on real-world game competitions. Agents interpret rules, choose opponents, analyze replays, and revise game agents to achieve their highest ranking within fixed match and evaluation budgets. We evaluate \val{completedmodels} model and harness configurations: Opus5.5 with Claude Code earns 6 gold medals, while no evalu
    
[^14]: VFold：对称性感知的跨层价值缓存压缩

    VFold: Symmetry-Aware Cross-Layer Value Cache Compression

    [https://arxiv.org/abs/2610.12338](https://arxiv.org/abs/2610.12338)

    该论文提出了一种对称性感知的跨层价值缓存合并策略，无需修改模型架构即可在解码时压缩KV缓存内存，并能与高比率量化或键缓存剪枝等现有技术组合使用，实现单一方法无法达到的更高压缩比。

    

    虽然缓存键值状态可以加速大语言模型（LLM）的解码，但在长上下文长度下，这种缓存可能会占据主要的内存用量。一种解决方案是利用层间缓存的相似性来压缩这部分内存。然而，大多数现有技术需要对LLM进行架构更改，并带来大量开销。在这项工作中，我们提出了一种对称性感知的价值缓存合并策略，能够在解码过程中减少缓存内存，同时避免有害的性能下降和架构开销。此外，我们展示了该方法可以与现有的缓存压缩技术结合使用，与高比率量化或键缓存剪枝相组合，达到单独使用任一方法都无法实现的压缩比，且额外成本极低。最终，我们的发现揭示了价值缓存中一个主要的未被充分利用的容量来源，为扩展上下文长度提供了一个简单而高效的方向。

    arXiv:2610.12338v1 Announce Type: new  Abstract: While caching key-value (KV) states accelerates Large Language Model (LLM) decoding, this cache can dominate memory usage at long context lengths. One solution is to compress this memory by exploiting inter-layer cache similarities. However, most existing techniques necessitate architectural changes to LLMs and incur substantial overhead. In this work, we propose a symmetry-aware value cache merging strategy that reduces cache memory while avoiding both harmful performance degradation and architectural overhead during decoding. Furthermore, we show that this approach can be exploited alongside existing cache compression techniques, composing with high-ratio quantization or key cache pruning to reach compression ratios that neither method reaches alone, with minimal additional cost. Ultimately, our findings reveal a major source of underutilized capacity in the value cache, offering a simple yet highly effective direction for scaling cont
    
[^15]: SparseDecoding：面向解码感知的剪枝方法，实现准确且高效的大语言模型推理

    SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference

    [https://arxiv.org/abs/2610.12327](https://arxiv.org/abs/2610.12327)

    该论文提出SparseDecoding，一种解码感知的剪枝方法，通过在模型自生成的token序列上计算Hessian来消除自然序列与生成序列之间的分布偏移，从而在保证准确性的同时提升LLM解码阶段的推理效率。

    

    大语言模型（LLM）推理的解码阶段具有内存受限的特性，会带来显著的延迟。由Hessian矩阵指导的逐层免训练网络剪枝方法是解决该问题的重要方案，因为剪枝可以减少解码过程中从内存读取的非零参数数量。然而，此类方法中的典型方案通常使用预先收集的自然序列来计算Hessian矩阵，而模型在解码阶段接收的却是自身生成的token，这在两类序列之间造成了分布偏移。在自然序列上计算的Hessian与在生成序列上计算的Hessian并不相同。我们观察到，这种差异会导致生成过程中的激活分布偏离剪枝时所用的分布，从而进一步损害剪枝后模型的性能。此外，大多数现有的能够带来实际加速的LLM剪枝方法主要针对稀疏矩阵-矩阵乘法（SpMM）运算……（摘要原文在此处截断）

    arXiv:2610.12327v1 Announce Type: cross  Abstract: The memory-bound nature of the decoding stage of large language model (LLM) inference incurs significant latency. Layer-wise training-free network pruning approaches guided by the Hessian have been a prominent solution to this problem, as pruning reduces the number of nonzero parameters read from memory during decoding. Nevertheless, typical methods in this line compute the Hessian using pre-collected natural sequences, whereas the model is fed self-generated tokens during decoding, creating a distribution shift between the two sequences. The Hessian calculated on the natural sequence is different from that calculated on the generated sequence. We observe that this discrepancy causes the activation distribution during generation to deviate from that used for pruning, further hurting the pruned model performance. Moreover, most existing LLM pruning methods that bring actual speedup primarily target the sparse matrix-matrix (SpMM) multip
    
[^16]: 无规则之判定：诊断与审计大语言模型合规系统中的监管规则敏感性

    Verdict Without the Rule: Diagnosing and Auditing Regulatory Rule Sensitivity in LLM Compliance Systems

    [https://arxiv.org/abs/2610.12313](https://arxiv.org/abs/2610.12313)

    研究发现大语言模型合规系统的判定对所给监管规则的删除、替换或否定等扰动往往不敏感，其guard模型在自定义规则适配下准确率仅51%、略高于随机水平，表明系统判定更多依赖案例的简单性而非对规则的真正遵循。

    

    大语言模型合规系统的部署建立在一个假设之上：判定结果取决于所提供的监管规则。我们在五个模型和20个监管及平台政策领域上直接检验了这一假设：在保持案例不变的情况下，删除、替换或否定起支配作用的规则，并检验判定结果是否会改变（OCS），或模型关于合规的内部表征是否发生任何偏移（ICS-delta）。结果显示两者变化都很小：模型的判定对所提供规则的实质性扰动常常保持不变；而在本文对其原生分类体系进行自定义规则适配的评估中，guard（守护）模型是五个模型中规则敏感性最低、准确率最差的，仅略高于随机水平（51%，而通用模型可达90-92%）。这更多反映的是案例较为简单，而非模型普遍忽视规则：在删除规则会改变模型原有正确预测的案例上，模型确实能紧密追踪规则。无论是改进提示词还是直接干预（摘要原文在此处截断）

    arXiv:2610.12313v1 Announce Type: new  Abstract: Large language model compliance systems are deployed on the assumption that a verdict depends on the regulatory rule it is given. We test this directly across five models and 20 regulatory and platform-policy domains: delete, swap, or negate the governing rule while holding the case fixed, and check whether the verdict changes (OCS) or the model's internal representation of compliance shifts at all (ICS-delta). Neither moves much: models' verdicts are often invariant to substantial perturbations of the supplied rule, and the guard model, evaluated here under a custom-rule adaptation of its native taxonomy, is the least rule-sensitive and least accurate of the five, barely above chance (51%, versus 90-92% for general-purpose models). This reflects easy cases more than blanket neglect: on cases where deleting the rule changes a previously correct model prediction, models do track it closely. Neither better prompting nor direct intervention
    
[^17]: HarnessSQL：面向真实数据库环境中SQL代理的执行框架原生训练

    HarnessSQL: Harness-Native Training for SQL Agents in Realistic Database Environments

    [https://arxiv.org/abs/2610.12274](https://arxiv.org/abs/2610.12274)

    HarnessSQL提出了一种框架原生的后训练方法，通过在带有隐藏执行预言机的真实数据库多轮交互环境中进行监督微调和执行奖励强化学习，弥合了文本到SQL模型静态训练与有状态实际部署之间的差距，显著提升了Spider 2.0-SQLite上的执行准确率。

    

    文本到SQL（Text-to-SQL）模型通常被训练为将问题直接映射到静态查询，而现实世界中的数据库代理则是通过与活数据库进行有状态的多轮交互来运作的——检查数据库模式、执行探测查询、诊断错误并修正假设。这造成了关键的训练-部署不匹配问题，因为调节这种交互的执行框架只在推理阶段才被引入。为了弥合这一差距，我们提出了HarnessSQL，一个框架原生的后训练框架，它在监督微调和强化学习的全过程中完整保留了交互结构。HarnessSQL构建了隔离的、可执行的数据库环境，并配有隐藏的执行预言机，直接在目标SQL执行框架内对教师模型进行推演，仅保留经过验证的轨迹用于全序列SFT，随后进行基于执行奖励的强化学习。在Spider 2.0-SQLite基准上，HarnessSQL显著提升了执行准确率。

    arXiv:2610.12274v1 Announce Type: new  Abstract: Text-to-SQL models are commonly trained to map questions directly to static queries, whereas real-world database agents operate through stateful, multi-turn interaction with live databases -- inspecting schemas, executing probe queries, diagnosing errors, and revising hypotheses. This creates a critical train-deploy mismatch, as the execution harness that mediates this interaction is introduced only at inference time. To bridge this gap, we propose HarnessSQL, a harness-native post-training framework that preserves the full interaction structure throughout both supervised fine-tuning and reinforcement learning. HarnessSQL builds isolated, executable database environments paired with hidden execution oracles, rolls out teachers directly inside the target SQL harness, and retains only verified trajectories for full-sequence SFT, followed by execution-reward RL. Across Spider 2.0-SQLite, HarnessSQL dramatically boosts the execution accuracy
    
[^18]: EgoVoice：基于自我中心多模态流的主动式语音辅助

    EgoVoice: Proactive Spoken Assistance from Egocentric Multimodal Streams

    [https://arxiv.org/abs/2610.12248](https://arxiv.org/abs/2610.12248)

    提出了EgoVoice框架，通过对HoloAssist第一人称视频进行声源分离与语音重合成构建训练数据，微调全模态大语言模型，使其能够自主决定何时开口以及提供何种语音指导，实现可穿戴AR设备中的主动式语音辅助。

    

    可穿戴增强现实（AR）助手正朝着持续现实世界交互的方向发展，它们通过第一人称视频和音频感知用户的活动，并在未被明确要求的情况下提供及时的语音指导。尽管主动式视频助手、语音对话系统和自我中心任务理解各自都取得了快速进展，但现有系统并未解决从连续第一人称流中决定“何时说话”和“说什么”这一联合问题。我们提出了EgoVoice，一个用于训练和评估主动式自我中心语音助手的框架。基于真实人类指导者的HoloAssist视频录制，我们通过声源分离和语音重合成构建了干净的音频流，并将每个视频会话转换为模型必须在每一时刻决定保持沉默还是提供语音指导的格式。我们使用该数据对全模态大语言模型进行微调，并进一步改进其主动行为表现。

    arXiv:2610.12248v1 Announce Type: new  Abstract: Wearable augmented reality (AR) assistants are moving toward continuous real-world interaction, where they perceive the user's activity through first-person video and audio and provide timely spoken guidance without being explicitly asked. While proactive video assistants, spoken dialog systems, and egocentric task understanding have each advanced rapidly, existing systems do not address the joint problem of deciding when to speak and what to say from continuous first-person streams. We introduce EgoVoice, a framework for training and evaluating proactive egocentric spoken assistants. From HoloAssist video recordings of real human instructors, we construct clean audio streams through source separation and speech resynthesis, and convert each video session into a format where the model must decide at each moment whether to remain silent or provide spoken guidance. We fine-tune an omni-modal LLM with our data, and further improve its proac
    
[^19]: NativeScope：基于正确锚点在原生拓扑上进行关系定位的检索

    NativeScope: Relation-Localized Retrieval over Native Topology with a Correct Anchor

    [https://arxiv.org/abs/2610.12243](https://arxiv.org/abs/2610.12243)

    论文提出 NativeScope，利用已知锚点和关系在数据系统原生结构（章节归属、会话边界、顺序）上先定位范围再排序，相比全范围稠密 RAG 将文档和记忆的原生单元召回率分别提升 42.75 和 22.00 个百分点。

    

    稠密检索通常根据文本块与问题的语义相似度对其进行排序，这忽略了许多数据系统已经存储的结构信息，包括章节归属、会话边界和原生顺序。我们提出 NativeScope，一种针对具有已知锚点和关系的查询的“先定范围后排序”方法。该方法将查询表示为 q -> (A, r, B)：锚点 A 和关系 r 通过“属于”、“之前”或“之后”算子选择原生单元，目标词 B 仅对与所选范围重叠的文本块进行排序。其内部变体 NS-FullQ 则使用完整问题对相同的候选进行排序。我们在从 QASPER 和 LongMemEval 导出的 200 条受控文档与记忆记录上，在 1,024 个 token 的预算下评估了这两种方法。NativeScope 在文档和记忆上的原生单元召回率分别达到 89.28% 和 72.50%，比实例级全范围稠密 RAG 分别提升 42.75 和 22.00 个百分点。NS-FullQ 达到 87.78% 和……（摘要在此处截断）

    arXiv:2610.12243v1 Announce Type: cross  Abstract: Dense retrieval usually ranks text chunks by their semantic similarity to a question. This ignores structure that many data systems already store, including section membership, session boundaries, and native order. We propose NativeScope, a scope-then-rank method for queries with a known anchor and relation. It represents a query as q -> (A, r, B). The anchor A and relation r select native units through belonging, before, or after operators, and the target term B ranks only chunks that overlap the selected scope. An internal variant, NS-FullQ, ranks the same candidates with the full question. We evaluate both methods on 200 controlled document and memory records derived from QASPER and LongMemEval under a 1,024-token budget. NativeScope attains native-unit recall of 89.28 percent for documents and 72.50 percent for memories, improving over instance-wide Dense RAG by 42.75 and 22.00 percentage points. NS-FullQ reaches 87.78 percent and 
    
[^20]: TokenRouter：面向Token级LLM路由的高效服务系统

    TokenRouter: Efficient Serving System for Token-Level LLM Routing

    [https://arxiv.org/abs/2610.12242](https://arxiv.org/abs/2610.12242)

    TokenRouter是一个高效且对开发者友好的服务系统，通过“以请求为中心的编程、以模型为中心的执行”的设计原则，解决了现有系统在Token级路由LLM推理中步骤失同步、批次准入延迟和实现复杂度高的难题。

    

    大语言模型（LLM）路由将推理工作分配到不同的模型上，从而推进LLM服务的成本-质量帕累托前沿。虽然在会话或查询级别的粗粒度路由已在生产系统中被广泛采用，但近期的算法研究表明，细粒度的Token级路由可以带来显著的效率和质量提升。然而，高效地服务Token级路由推理对现有系统提出了重大挑战。基于单LLM假设构建的现有系统，在Token级路由下会出现严重的步骤失同步和频繁的批次准入延迟，同时还会给开发者带来较高的实现复杂度。为了解决这些挑战，我们设计了TokenRouter，一个高效且对开发者友好的Token级路由LLM推理服务系统。TokenRouter遵循“以请求为中心的编程、以模型为中心的执行”的原则：开发者描述……

    arXiv:2610.12242v1 Announce Type: new  Abstract: Large language model (LLM) routing distributes inference work across different models, advancing the cost-quality Pareto frontier of LLM serving. While coarse-grained routing at the session or query level has been widely adopted in production systems, recent algorithmic work shows that fine-grained token-level routing can yield substantial efficiency and quality gains. However, efficiently serving token-level routed inference poses significant challenges to existing systems. Built on single-LLM assumptions, current systems suffer from severe step desynchronization and frequent batch admission delays under token-level routing, and they also impose high implementation complexity on developers. To address these challenges, we design TokenRouter, an efficient and developer-friendly serving system for token-level routed LLM inference. TokenRouter follows the principle of request-centric programming, model-centric execution: developers describ
    
[^21]: 语言模型作为AI研究世界模型

    Language Models as AI Research World Models

    [https://arxiv.org/abs/2610.12235](https://arxiv.org/abs/2610.12235)

    该论文提出将语言模型作为“研究世界模型”来预测AI实验的结果，并证明真实实验经验获得的研究知识能提升预测准确性且可跨研究环境复用，从而在有限实验预算下支持AI研究的持续自我改进。

    

    AI研究智能体自动化了提出、实施和评估实验的循环，为递归自我改进开辟了道路。然而，它们提出实验的能力超过了在真实环境中执行实验的能力，这使得结果预测成为在有限实验预算下实现持续自我改进的关键能力。我们研究了将语言模型作为研究世界模型，用于预测候选干预措施在各类研究环境中的结果。我们的评估基于来自九个研究环境的2600多条实验记录，涵盖预训练、后训练和推理阶段，相当于超过171,000个H100 GPU小时的实验。从真实实验经验中获得的研究知识提升了RWM对同一环境中未见干预措施的预测能力（Spearman +0.27），并且可以跨环境复用。例如，仅使用预训练经验……（摘要原文在此处截断）

    arXiv:2610.12235v1 Announce Type: new  Abstract: AI research agents automate the cycle of proposing, implementing, and evaluating experiments, opening a path toward recursive self-improvement. Yet their ability to propose experiments outpaces their capacity to execute them in real environments, making outcome prediction a key capability for sustained self-improvement under limited experimental budgets. We investigate language models as Research World Models (RWMs), which predict the outcomes of candidate interventions across research environments. Our evaluation draws on over 2,600 experimental records from nine research environments spanning pretraining, post-training, and inference, representing more than 171,000 H100 GPU-hours of experimentation. Research knowledge acquired from real experimental experience improves RWM predictions of unseen interventions within the same environment (Spearman +0.27), and can be reused across environments. For example, using only pretraining experien
    
[^22]: DiffuPlex：通过滚动掩码扩散加速全双工口语对话模型

    DiffuPlex: Accelerating Full-Duplex Spoken Dialog Models via Rolling Masked Diffusion

    [https://arxiv.org/abs/2610.12214](https://arxiv.org/abs/2610.12214)

    DiffuPlex 提出了一种滚动掩码扩散框架，通过单次骨干网络唤醒预测多个未来用户与助手帧，并在交互出现偏差时仅修改未播放的未来内容，从而加速全双工口语对话模型。

    

    近期的全双工口语对话模型能够实现同时听和说，但细粒度模型仍在每个交互帧上以自回归方式推进其骨干网络。我们提出了 DiffuPlex，这是一种滚动掩码扩散框架，通过在单次骨干网络唤醒中预测多个未来的用户帧和助手帧来减少这种顺序计算。在交互以原始帧率继续进行的同时，DiffuPlex 仅消耗每个预测未来中置信度较高的前缀部分。随着用户语音的到来，它会检查相应的用户预测，当交互出现偏差时，保留已播放的助手内容，仅修改尚未播放的未来部分。我们在同一预测器上考虑了两种推理策略：DiffuPlex-LISTEN 在预测到助手沉默时消耗多个未来帧，而 DiffuPlex-SPEAK 也可以消耗预测到的助手语音。在全文双工交互和口语语言评估（摘要原文在此处截断）……

    arXiv:2610.12214v1 Announce Type: cross  Abstract: Recent full-duplex spoken dialog models enable simultaneous listening and speaking, but fine-grained models still advance their backbone autoregressively at every interaction frame. We introduce DiffuPlex, a rolling masked diffusion framework that reduces this sequential computation by predicting multiple future user and assistant frames in a single backbone wake. DiffuPlex consumes only a confident prefix of each predicted future while interaction continues at the original frame rate. As user speech arrives, it checks the corresponding user predictions and, when the interaction diverges, preserves already played assistant content while revising only the unplayed future. We consider two inference policies over the same predictor: DiffuPlex-LISTEN consumes multiple future frames when they predict assistant silence, whereas DiffuPlex-SPEAK can also consume predicted assistant speech. Across full-duplex interaction and spoken-language eva
    
[^23]: SciTBERT：一系列用于科学与技术语言处理的时间一致性语言模型

    SciTBERT: A family of chronologically consistent language models for scientific and technological language processing

    [https://arxiv.org/abs/2610.12207](https://arxiv.org/abs/2610.12207)

    提出了SciTBERT——一系列时间一致的BERT衍生语言模型，训练数据截止日期覆盖2013至2025年每年，通过消除前瞻偏差和领域偏差，使模型适用于研究科学与技术随时间演变的特性。

    

    预训练Transformer模型正被越来越多地用于研究科学技术进步。针对论文或专利文本微调的编码器，在科学与技术领域的下游分类、回归和相似度任务上优于通用模型。然而，由于这些预训练模型固有的“前瞻偏差”和领域偏差，它们在研究科学、技术及其交叉领域的时间相关或历史档案特性方面的适用性受到限制。这些局限源于训练语料库在时间分布和文本来源分布上缺乏约束。我们提出了SciTBERT：这是一个时间一致的、基于BERT的语言模型家族，其训练数据来自科学论文、专利和高质量教育类网络文本，训练数据截止日期覆盖2013年至2025年的每一年。我们还以时间一致的方式对这些模型进行后训练，使用论文和专利……

    arXiv:2610.12207v1 Announce Type: new  Abstract: Pre-trained transformer models are increasingly being used to study scientific and technological progress. Encoders tuned to paper or patent text outperform general-purpose models on downstream classification, regression, and proximity tasks within science and technology. However, the applicability of these models for studying time-dependent or archival properties of science, technology, and their interface is limited due to lookahead and domain biases inherent to these pre-trained models. These limitations arise from training on corpora with unconstrained chronological and text source distributions. We introduce SciTBERT: a family of chronologically consistent BERT-derived language models trained on text from scientific papers, patents, and high-quality educational web text with training data cutoff dates spanning each year between 2013 and 2025. We also post-train these models in a chronologically-consistent manner using paper and pate
    
[^24]: SteerablePlex：我们能否控制全双工模型？

    SteerablePlex: Can We Steer Full-Duplex Models?

    [https://arxiv.org/abs/2610.12201](https://arxiv.org/abs/2610.12201)

    本文提出SimIF-Bench基准测试揭示当前开源全双工语音模型难以遵循预设场景约束，并通过GDPO训练方案构建了SteerablePlex模型，使其能在持续对话中遵循文本指令的同时保持轮次转换能力。

    

    全双工语音模型能够同时聆听和说话，从而实现自然的交互，但随着对话历史的增长，它们变得越来越难以控制。当这些模型被用作用户模拟器时，缺乏可控性会导致它们偏离预设场景，产生不可靠的评估结果。我们提出了SimIF-Bench（模拟器指令遵循基准测试），用于评估对话模型是否能够保持在预设场景之内，并按要求的顺序完成多个目标。该基准测试表明，当前的开源全双工模型难以遵循这些约束条件。随后，我们提出了一种基于群体奖励解耦归一化策略优化（GDPO）的训练方案，使全双工模型能够在正在进行的对话中遵循文本指令，同时保持其轮次转换能力。通过将得到的SteerablePlex模型连接到异步后端语言（模型）……（摘要在此处截断）

    arXiv:2610.12201v1 Announce Type: cross  Abstract: Full-duplex speech models can listen and speak simultaneously, enabling natural interaction, but become increasingly difficult to control as the conversation history grows. When used as user simulators, this lack of control can cause them to deviate from prescribed scenarios and produce unreliable evaluation outcomes. We introduce SimIF-Bench (Simulator Instruction-Following Benchmark), which evaluates whether a conversational model stays within a prescribed scenario and completes multiple goals in the required order. The benchmark reveals that current open-source full-duplex models struggle to follow such constraints. We then introduce a Group Reward-Decoupled Normalization Policy Optimization (GDPO)-based training recipe that enables a full-duplex model to follow textual instructions during an ongoing conversation while maintaining its turn-taking ability. By connecting the resulting SteerablePlex to an asynchronous backend language 
    
[^25]: 组策略优化中KL正则化何时失效

    When KL Regularization Misfires in Group Policy Optimization

    [https://arxiv.org/abs/2610.12161](https://arxiv.org/abs/2610.12161)

    论文剖析了组策略优化中KL正则化与奖励相互作用的七种失效模式，并提出零和校准策略优化（ZCPO），借助条件KL度量的相对漂移校准组内奖励系数，从而提升优化效果。

    

    为什么移除参考策略KL正则化有时会改善组策略优化的效果？这促使我们研究参考策略信息应当如何进入组相对更新之中。我们分析了KL与奖励之间相互作用的七种潜在失效模式：奖励裁剪后、梯度抵消后以及整组奖励完全相同时出现的残余KL更新；KL随响应长度增长而增加及其相对贡献的不平衡；KL集中在少数token上；以及将k1项纳入奖励时引入的采样噪声。我们提出零和校准策略优化（ZCPO），利用条件KL度量的相对漂移来校准组内奖励系数，并将其整合到基础代理目标中。数学推理实验与消融研究支持了该设计在本文设定中的有效性。

    arXiv:2610.12161v1 Announce Type: cross  Abstract: Why does removing reference-policy KL regularization sometimes improve group policy optimization? This motivates studying how reference-policy information should enter group-relative updates. We analyze seven potential failure modes in the interactions between KL and rewards: residual KL updates after reward clipping, after gradient cancellation, and in groups with identical rewards; KL growth with response length and an imbalance in its relative contribution; KL concentration on a small number of tokens; and sampling noise when k1 is incorporated into rewards. We propose Zero-Sum Calibrated Policy Optimization (ZCPO), which uses relative drift measured by conditional KL to calibrate within-group reward coefficients and integrates them into the base surrogate. Mathematical reasoning experiments and ablations support this design's effectiveness in our settings.
    
[^26]: 多语言语言模型中分词器选择的语言特异性影响

    Language-Specific Effects of Tokenizer Choice in Multilingual Language Models

    [https://arxiv.org/abs/2610.12144](https://arxiv.org/abs/2610.12144)

    该研究通过固定架构、语料和训练预算、仅改变分词器的控制实验（123个模型、54种分词器），发现分词器选择对训练数据较少的低资源语言影响更大。

    

    分词器的选择会影响多语言语言建模，但词表容量是有限的，且词表大小通常受到约束：改善某些语言的表示往往以牺牲其他语言为代价。因此，我们提出一个问题：分词器的选择对各种语言的影响是否同等重要，而这一问题在当前文献中尚未得到解答。为此，我们训练了123个语言模型，涵盖54种分词器。在主要对比中，模型架构、训练语料、训练词元预算和优化方法均保持固定，因此这些模型之间仅存在分词器上的差异。我们发现，分词器的选择对语言模型训练数据较少的语言影响更大：在这54种分词器中，某种语言的每字节比特数（BPB）的标准差随着其模型训练数据占比的下降而增加（在31种有训练数据的语言上Spearman rho = -0.52，在28种以词边界书写的语言上为-0.69）。将某种语言排除在分词器训练之外……

    arXiv:2610.12144v1 Announce Type: new  Abstract: Tokenizer choice affects multilingual language modeling, but vocabulary capacity is finite and vocabulary size is often constrained: improving representation for some languages often comes at the expense of others. We therefore ask whether tokenizer choice matters equally across languages, a question that the current literature leave unanswered. To this end, we train 123 language models spanning 54 tokenizers. In the main comparison, architecture, training corpus, training-token budget, and optimization are held fixed, so the models differ only in their tokenizer. We find that tokenizer choice matters more for languages with less language-model training data: across the 54 tokenizers, the standard deviation of a language's bits-per-byte (BPB) increases as its model training-data share decreases (Spearman rho = -0.52 over the 31 trained languages and -0.69 over the 28 written with word boundaries). Leaving a language out of tokenizer trai
    
[^27]: 排练一切，记住无物：Attic-KV 只排练将被读取的内容

    Rehearse Everything, Remember Nothing: Attic-KV Rehearses What Will Be Read

    [https://arxiv.org/abs/2610.12133](https://arxiv.org/abs/2610.12133)

    该论文发现传统KV缓存压缩中“重读全部上下文”的排练策略在低保留率下会因预算分散而失效，并提出Attic-KV——只排练将来会被读取的内容，像考前自测一样大幅提升缓存压缩后的记忆效果。

    

    许多键值（KV）缓存会在无人知晓将来要被问及什么之前就被压缩：为检索而缓存的文档、跨请求共享的提示前缀、长对话的记忆。主流方法通过“排练”来为KV条目打分：模型重新阅读上下文并保留其关注的条目，并假设缓存对其上下文排练得越完整，就记得越牢。我们证明，在紧张的预算下这一假设会适得其反：排练一切，记住无物。在3%的保留率下，重读整个上下文在RULER上仅能保留96.5分中的31.5分，而在LongBench的自然文本任务上，其表现甚至低于完全不排练的方法。原因在于缓存会保留它所排练的内容：重读会将预算分散到整个上下文，导致答案自身的条目仅以略高于随机的概率得以保留。正如考试前的学生，缓存通过自我测试比通过重读能记得更多。

    arXiv:2610.12133v1 Announce Type: new  Abstract: Many key-value (KV) caches are compressed before anyone knows what will be asked of them: a document cached for retrieval, a prompt prefix shared across requests, the memory of a long conversation. The prevailing approach scores KV entries by rehearsal: the model rereads the context and keeps the entries it attends to, assuming that the more completely a cache rehearses its context, the better it remembers it. We show that under tight budgets this assumption backfires: rehearse everything, remember nothing. At a 3% keep ratio, rereading the whole context keeps 31.5 of 96.5 points on RULER, and on LongBench's natural-text tasks it falls below methods that rehearse nothing at all. The cause is that a cache keeps what it rehearses: rereading spreads the budget across the whole context, so the answer's own entries survive at little more than chance. Like a student before an exam, a cache remembers more by testing itself than by rereading. Tw
    
[^28]: 自动化言语欺骗检测中持续存在的准确率上限

    A persistent accuracy ceiling in automated verbal deception detection

    [https://arxiv.org/abs/2610.12118](https://arxiv.org/abs/2610.12118)

    通过对25年间289篇报告和6,136个分类模型的系统回顾与元分析，该研究揭示自动化言语欺骗检测的准确率主要受方法论质量而非模型复杂度驱动，存在约70-75%的持续准确率上限。

    

    自动化方法已被提出用于克服人工言语欺骗检测的局限性，但相关证据在各学科间仍然分散。我们系统回顾了25年的研究（289篇报告，6,136个分类模型），并对嵌套在97个数据集中的3,653个模型进行了元分析。合并准确率为74.4%（95%置信区间：71.2%-77.4%），且存在显著的异质性。准确率更多受方法论质量（真值标准、数据来源、类别平衡、评估程序）驱动，而非模型复杂度：嵌入模型和大语言模型的采用并未转化为预测性能的提升。仅有12.46%的报告使用了具有可验证真值的数据，仅有23.96%的模型在独立数据上进行了评估。合并准确率与人工方法的元分析结果一致，表明存在70-75%的准确率上限，而当前的研究惯例不太可能突破这一上限。

    arXiv:2610.12118v1 Announce Type: new  Abstract: Automated methods have been proposed to overcome the limitations of human verbal deception detection, but evidence remains fragmented across disciplines. We systematically reviewed 25 years of research (289 reports, 6,136 classification models) and meta-analyzed 3,653 models nested within 97 datasets. Pooled accuracy was 74.4% (95% CI: 71.2%-77.4%) with substantial heterogeneity. Accuracy was driven by methodological quality (ground truth, data source, class balance, evaluation procedure) more than by model complexity: the adoption of embeddings and large language models has not translated into improved predictive performance. Only 12.46% of reports used data with verifiable ground-truth, and only 23.96% of models were evaluated on independent data. The pooled accuracy aligns with meta-analyses of manual approaches, suggesting a ceiling of 70-75%, unlikely to be lifted by current research conventions.
    
[^29]: 所有判决并非平等：重新思考LLM裁判的可靠性

    All Verdicts are Not Equal: Rethinking LLM Judge Reliability

    [https://arxiv.org/abs/2610.12083](https://arxiv.org/abs/2610.12083)

    该论文通过大规模压力测试揭示了LLM裁判评估系统存在的严重可靠性缺陷（判决不稳定、受顺序影响、错误判决也会保持一致），并提出“可信判决率”这一统一指标来同时量化评估的可重现性、顺序不变性和准确性。

    

    LLM作为裁判（LLM-as-a-Judge）是NLP评估的标准范式，然而尽管它被广泛视为确定性的真值标准，其系统性可靠性仍然鲜为人知。我们提出了一项全面的可靠性审计，对六个前沿模型进行压力测试，涵盖四个基准测试、五种提示格式、两种呈现顺序、三种采样温度，以及每种条件下十次重复。我们的实证分析揭示了严重的脆弱性：即使在温度为零的情况下，相同重复的判决也会发生变化；位置顺序的交换会翻转具有挑战性任务上的大多数判决；而最具确定性的裁判竟是通过简单重复错误判决来实现完美一致性，其与真值一致的比率仅为51%。为了形式化这些多方面的失败模式，我们引入了可信判决率，这是一个统一的指标，用于捕捉评估结果可重现、顺序不变且准确的联合概率。

    arXiv:2610.12083v1 Announce Type: cross  Abstract: LLM-as-a-Judge is the standard paradigm for NLP evaluation, yet its systemic reliability remains poorly understood despite being widely treated as a deterministic ground truth. We present a comprehensive reliability audit, stresstesting six frontier models across four benchmarks, five prompt formats, two presentation orders, three sampling temperatures, and ten repetitions per condition. Our empirical analysis reveals severe vulnerabilities: verdicts change across identical replications at temperature zero, position-order swaps flip the majority of verdicts on challenging tasks, and the most deterministic judge achieves perfect consistency by trivially repeating incorrect verdicts, agreeing with ground truth only 51% of the time. To formalize these multi-faceted failure modes, we introduce the trustworthy verdict rate (T ), a unified metric capturing the joint probability that an evaluation is reproducible, order-invariant, and accurat
    
[^30]: ILM：一个AI驱动的叙事教育工具

    ILM: An AI-Powered Storytelling Educational Tool

    [https://arxiv.org/abs/2610.12064](https://arxiv.org/abs/2610.12064)

    本文提出了ILM平台，通过结合阿拉伯语自然语言处理、知识图谱构建和多语言检索技术，为先知故事提供结构化、可视化和多语言支持的互动式学习与理解评估体验。

    

    数字技术已经使伊斯兰叙事更容易获取，但现有平台对这些故事的结构化学习和理解提供的支持有限，尤其是在阿拉伯语和多语言环境中。我们提出了ILM，这是一个面向先知故事的互动教育平台，它结合了阿拉伯语自然语言处理、结构化知识表示和基于检索的问题生成。经管理员审核的阿拉伯语叙事由知识图谱（KG）构建引擎处理，该引擎识别实体和叙事关系并将其存储为结构化知识，使学习者能够通过可视化故事地图探索故事，并回答由知识图谱生成的基于实体和关系的问题。此外，一个多语言检索流水线从原始叙事中检索相关段落，以生成多项选择和开放式理解问题。对于开放式问题，（原文摘要在此处截断）

    arXiv:2610.12064v1 Announce Type: new  Abstract: Digital technologies have made Islamic narratives more accessible, but existing platforms provide limited support for structured learning and comprehension of these stories, particularly in Arabic and multilingual settings. We present ILM, an interactive educational platform for Stories of the Prophets that combines Arabic natural language processing, structured knowledge representation, and retrieval-based question generation. Admin-approved Arabic narratives are processed by a Knowledge Graph (KG) Constructor Engine that identifies entities and narrative relationships and stores them as structured knowledge, enabling learners to explore stories through a visual story map and answer entity- and relation-based questions generated from the KG. Separately, a multilingual retrieval pipeline retrieves relevant passages from the original narratives to generate multiple-choice and open-ended comprehension questions. For open-ended questions, a
    
[^31]: 智能体应该何时思考？通过跨轮次估计实现自适应推理

    When Should Agents Think? Adaptive Reasoning via Cross-Turn Estimation

    [https://arxiv.org/abs/2610.12061](https://arxiv.org/abs/2610.12061)

    提出 RACE 方法，利用移除推理后后续动作似然下降这一轻量信号进行跨轮次估计，判断智能体何时需要新的推理步骤，从而在无需昂贵生成式验证的情况下实现自适应推理训练。

    

    基于大语言模型（LLM）的智能体在复杂任务上展现出了强大的能力。它们通常在整个交互轨迹中，每次执行动作之前都会进行推理。然而，并非每一轮都需要推理，因为较早产生的推理可以继续支持后续的动作。因此，一个关键挑战在于：在不依赖昂贵的基于生成的验证的前提下，确定现有推理何时仍然足够、何时需要新的推理步骤。我们发现，在移除额外推理后，后续参考动作的似然下降能够紧密反映在给定先前推理的情况下这些动作是否仍然可恢复，这为估计跨轮次动作支持提供了一个有效且轻量的信号。基于这一观察，我们提出了通过跨轮次估计实现推理自适应的方法，这是一种用于智能体自适应推理的训练方法。RACE 引入了一种似然-

    arXiv:2610.12061v1 Announce Type: new  Abstract: Large language model (LLM)-based agents have demonstrated strong capabilities on complex tasks. They typically perform reasoning before each action throughout an interaction trajectory. However, reasoning may not be necessary at every turn, as reasoning produced earlier can continue to support subsequent actions. A key challenge is therefore to determine when existing reasoning remains sufficient and when a new reasoning step is needed, without relying on costly generation-based verification. We find that decreases in the likelihood of subsequent reference actions after removing additional reasoning closely track whether those actions remain recoverable given earlier reasoning, providing an effective and lightweight signal for estimating cross-turn action support. Based on this observation, we propose Reasoning Adaptation through Cross-Turn Estimation (RACE), a training approach for adaptive agent reasoning. RACE introduces a Likelihood-
    
[^32]: 基于大语言模型的自然语言到一阶逻辑的自动形式化

    Natural Language to First-Order Logic LLM-based Autoformalization

    [https://arxiv.org/abs/2610.12030](https://arxiv.org/abs/2610.12030)

    本文首次通过区分本体抽取与逻辑翻译，为一阶逻辑自动形式化任务提供了原则性定义，并系统综述了现有数据集、评估指标和基于大语言模型的方法，指出了基准测试与语义评估等方面的开放性挑战。

    

    大语言模型（LLMs）重新激发了人们对自动形式化的兴趣。然而，当以一阶逻辑（FOL）作为目标形式化语言时，该领域仍然缺乏统一的任务定义和系统性的综述。本文填补了这一空白：我们首先通过区分本体抽取与逻辑翻译，为一阶逻辑自动形式化任务提供了原则性的定义，并展示了两者概念混淆如何使（跨研究的）评估变得模糊不清；我们综述了现有的数据集、评估指标以及基于大语言模型的方法，包括微调、提示工程和基于验证的精炼；我们还指出了在基准测试、语义评估、本体感知方法以及端到端应用方面的开放性挑战。

    arXiv:2610.12030v1 Announce Type: new  Abstract: Large Language Models (LLMs) have renewed interest in autoformalization. Yet, when First-Order Logic (FOL) is considered as the target formalism, the field still lacks a unified task formulation and a systematic survey. This paper addresses this gap: we first provide a principled definition for the FOL-autoformalization task by distinguishing Ontology Extraction from Logical Translation, showing how their conflation obscures (cross-study) evaluation; we review existing datasets, evaluation metrics, and LLM-based methods, including fine-tuning, prompting, and verification-based refinement; we identify open challenges in benchmarking, semantic evaluation, ontology-aware methods, and end-to-end applications.
    
[^33]: InterviewPlayground：一个用于评估AI访谈员的模拟环境

    InterviewPlayground: A Simulation Environment for Evaluating AI Interviewers

    [https://arxiv.org/abs/2610.12023](https://arxiv.org/abs/2610.12023)

    提出了InterviewPlayground，一个基于社会理论模拟参与者行为、通过多轮交互生成标准化访谈成绩单来评估AI访谈员的模拟环境，并验证了模拟评估结果能够预测AI访谈员与真实人类参与者互动时的实际表现。

    

    AI访谈员正被越来越多地开发出来，用于在市场研究、民意调查、偏好引导和社会科学研究等应用中获取开放式回答。然而，评估AI访谈员十分困难，因为它们在长时间的多轮交互中运行，必须根据参与者的行为做出适应。为了满足这一需求，我们开发了InterviewPlayground，一个用于评估AI访谈员的模拟环境，其中模拟研究对象的行为建立在社会理论基础之上。InterviewPlayground中的模拟研究会生成一份InterviewReportCard（访谈成绩单），通过一系列经过验证的测量指标来评估AI访谈员的表现。为了检验这种基于模拟的评估能否预测AI访谈员在与真实人类参与者互动时的表现，我们开展了15项真实的定性研究，涉及五个AI访谈员、三个访谈主题和450名人类参与者，并将其与InterviewPlayground中的模拟研究结果进行了比较。

    arXiv:2610.12023v1 Announce Type: new  Abstract: Increasingly, AI interviewers are being developed to elicit open-ended responses in applications like market research, public polling, preference elicitation, and social science research. However, evaluating AI interviewers is challenging because they function in extended, multi-turn interactions where they must adapt to participant behaviors. To address this need, we develop InterviewPlayground, a simulation environment for evaluating AI interviewers using simulated study participants whose behaviors are grounded in social theory. Simulated studies in InterviewPlayground produce an InterviewReportCard, which assesses the performance of AI interviewers using a suite of validated measures. To test whether our simulation-based evaluations predict performance with human participants, we conduct 15 real qualitative studies with five AI interviewers, three interview topics, and 450 human participants and compare them to simulated studies in I
    
[^34]: 检验大语言模型推理中的社会归因：一种理论指导的探测方法

    Examining Social Attribution in LLM Reasoning: A Theory-Guided Probing Methodology

    [https://arxiv.org/abs/2610.12022](https://arxiv.org/abs/2610.12022)

    该论文首次系统性地探索大语言模型的社会归因能力，在归因理论指导下构建了包含经典心理学情境与现实场景的基准，以考察大语言模型在责任与过错归因上的判断及其内部机制。

    

    大语言模型越来越多地被部署在社会技术系统中，其中社会归因——即将外部事件归因于智能体社会行为的原因和缘由的推理过程——发挥着关键作用。这些过程涉及对社会原因、责任以及智能体所应承担的过错或功劳的判断。尽管归因模型在心理学和认知科学中已通过归因理论得到充分研究，但社会归因在人工智能领域，尤其是大语言模型的社会推理中，仍然探索不足。本文首次对大语言模型的社会归因进行了系统性探索。我们的工作聚焦于责任与过错归因，考察了当前大语言模型的判断表现及其潜在的内部机制。在归因理论的指导下，我们构建了一个社会归因基准，其中包含基于归因理论研究经典情景的情境片段子集，以及基于现实世界社会场景的现实子集。

    arXiv:2610.12022v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed in sociotechnical systems where social attribution, the reasoning process attributing external events to the causes and reasons of agents' social behaviors, plays a critical role. These processes involve judgments of social cause, responsibility, and blame/credit to agents. Although attributional models are well-studied in social psychology and cognition through Attribution Theory, social attribution remains underexplored in AI, particularly LLM social reasoning. This paper provides the first systematic exploration of LLM social attribution. Our work focuses on responsibility and blame attributions, examining current LLMs' judgments and their underlying internal mechanisms. Guided by attribution theory, we construct a social attribution benchmark consisting of a Vignette subset based on classic scenarios from attribution theory research and a Reality subset based on real-world soci
    
[^35]: Agentic-TTT：为测试时训练学习测试时策略

    Agentic-TTT: Training test-time policy for test-time training

    [https://arxiv.org/abs/2610.12002](https://arxiv.org/abs/2610.12002)

    提出Agentic-TTT框架，通过训练一个测试时策略来智能决策何时、如何调用测试时训练（TTT）以及是否复用已有技能，从而实现模型参数层面的自主化自我改进。

    

    测试时训练（TTT）利用来自测试输入的信号来调整大语言模型的参数，可以在预先指定的场景（如IMO竞赛或指定的开放性问题）中带来显著的性能提升。通过将部署经验转化为参数更新，TTT为模型层面的自我改进提供了一种直接机制。然而，TTT并非在所有情况下都有益：每种TTT算法适用于不同的场景，应用不当的方法可能浪费测试时计算资源，甚至会损害模型性能。因此，这种参数层面的自我改进需要自主性：模型必须决定何时需要TTT、调用哪种算法，以及是否可以复用已有的技能。为填补这一空白，我们提出了Agentic-TTT，它通过学习一个测试时策略来管理这些决策。Agentic-TTT将TTT流程转化为可调用的工具，将积累的技能视为不断演化的部署环境，并使用（摘要在此处截断）

    arXiv:2610.12002v1 Announce Type: cross  Abstract: Test-time training (TTT) adapts an LLM's parameters using signals derived from test inputs, and can make striking improvements in pre-specified settings such as IMO competitions or designated open problems. By turning deployment experience into parameter updates, TTT provides a direct mechanism for model-level self-improvement. Yet TTT is not universally beneficial: each TTT algorithm works in different settings, and applying an ill-suited method could waste test-time compute or even damage model performance. Therefore, such parameter-level self-improvement requires agency: the model must decide when TTT is warranted, which algorithm to invoke, and whether an existing skill can be reused. To fill this gap, we introduce Agentic-TTT, which learns a test-time policy to govern those decisions. Agentic-TTT turns TTT procedures into callable tools, treats accumulated skills as an evolving deployment environment, and trains its policy using t
    
[^36]: DataVista：诊断多模态大语言模型在数据视频理解上的能力

    DataVista: Diagnosing Multimodal LLMs on Data Video Understanding

    [https://arxiv.org/abs/2610.11993](https://arxiv.org/abs/2610.11993)

    该论文提出了首个数据视频理解基准DataVista，包含961个真实数据视频和6,775个评估问题，通过三级渐进式能力框架系统评估了19个主流多模态大语言模型在该任务上的表现。

    

    数据视频是一种将数据可视化与视频叙事相结合的媒体形式，被广泛应用于新闻报道和商业分析中。与一般的视频理解相比，数据视频理解更加强调从动态图表中准确读取数据、跨图表和时间整合证据，以及理解叙事组织和视觉设计如何传达信息。然而，现有的基准测试要么针对一般视频，要么针对静态图表，数据视频理解尚未得到系统性评估。我们提出了DataVista，这是首个面向数据视频理解的基准，包含961个真实世界的数据视频和6,775个评估问题，这些问题按照三级渐进式能力框架（数据感知、时间推理、叙事理解）组织，涵盖五个主题领域的10种细粒度问题类型。对19个主流多模态大语言模型的系统性评估表明，表现最好的模型……

    arXiv:2610.11993v1 Announce Type: cross  Abstract: Data video is a media form that integrates data visualization with video narrative, widely adopted in news reporting and business analysis. Compared with general video understanding, data video understanding places greater emphasis on accurately reading data from animated charts, integrating evidence across charts and time, and understanding how narrative organization and visual design communicate information. Yet existing benchmarks target either general videos or static charts, and data video understanding has not been systematically evaluated. We present DataVista, the first benchmark for data video understanding, containing 961 real-world data videos and 6,775 evaluation questions organized under a three-level progressive capability framework (data perception, temporal reasoning, narrative understanding) with 10 fine-grained question types across five topic domains. Systematic evaluation of 19 mainstream MLLMs shows that the best-p
    
[^37]: 专用决策模型与通用大语言模型的对比：Jev在知识、推理与多语言任务上的基准测试

    Specialized Decision Models vs. General-Purpose LLMs: Benchmarking Jev Across Knowledge, Reasoning, and Multilingual Tasks

    [https://arxiv.org/abs/2610.11978](https://arxiv.org/abs/2610.11978)

    专用决策模型Jev在知识和常识类基准上可与前沿大语言模型媲美，但在需要多步计算的数学任务上显著落后，表明其能力仅限于知识型决策而非计算推理。

    

    Jev是一个“系统一”模型，它在给定的选项中返回一个选择，而不是生成文本。我们研究了这种专用决策模型与通用大语言模型相比的表现。我们在涵盖知识、推理和多语言理解的13个多项选择基准上对Jev进行评估，并将其与三个层级（前沿、代表性和小型）的19个大语言模型进行比较。Jev在知识和常识基准上与前沿大语言模型具有竞争力，并在MMLU-Redux和ARC-Challenge上取得了最佳成绩。在数学之外的领域，它还超越了大多数代表性大语言模型和所有小型大语言模型。然而，它在数学应用题方面表现落后：在MathQA上，它比前沿模型的中位数低17.7分，且得分低于全部19个大语言模型。这些结果表明，专用决策模型可以在主要依赖知识的决策任务上匹敌通用大语言模型，但在需要多步计算的决策任务上则无法做到。

    arXiv:2610.11978v1 Announce Type: new  Abstract: Jev is a "System One" model that returns a choice among given options instead of generating text. We study how such a specialized decision model compares with general-purpose large language models (LLMs). We evaluate Jev on 13 multiple-choice benchmarks covering knowledge, reasoning, and multilingual understanding, and compare it with 19 LLMs in three tiers: frontier, representative, and small. Jev is competitive with frontier LLMs on knowledge and commonsense benchmarks and obtains the best score on MMLU-Redux and ARC-Challenge. Outside mathematics, it also outperforms most representative LLMs and all small LLMs. However, it falls behind on mathematical word problems: on MathQA, it is 17.7 points below the frontier median and scores lower than all 19 LLMs. These results indicate that a specialized decision model can match general-purpose LLMs on decisions that rely mainly on knowledge, but not on decisions that require multi-step calcul
    
[^38]: MindFlow：由心智超网驱动的研究创意创新思维流

    MindFlow: Mind Supernet Powered Thinking Flows for Research Idea Innovation

    [https://arxiv.org/abs/2610.11966](https://arxiv.org/abs/2610.11966)

    MindFlow将科研创意生成形式化为由模块化思维算子构成的图结构“心智流”，通过概率性心智超网建模与基于锦标赛相对排序的优化，让控制器能够动态采样并逐步生成更高质量的创意。

    

    研究创意创新是科学进步的根本引擎，但如何以可扩展且可控的方式生成和评估创意仍然困难重重。这一挑战源于其固有的开放性与多目标性——创意需要在新颖性、合理性与可行性之间取得平衡。尽管近期基于大语言模型（LLM）的方法通过精心设计的提示词或智能体流水线取得了一定进展，但它们都受限于预定义的、静态的创意生成工作流。为解决这一局限，我们提出了MindFlow，一个将创意构思显式形式化为图结构“心智流”的框架。该心智流由模块化的思维算子构成，并通过概率性心智超网进行建模。给定一个研究主题，控制器会动态采样思维流以生成候选创意。这一开放性问题通过基于锦标赛的相对排序进行优化，使控制器能够逐步倾向于更高质量的思维流。

    arXiv:2610.11966v1 Announce Type: new  Abstract: Research idea innovation is a fundamental engine of scientific progress, yet it remains difficult to generate and evaluate in a scalable and controllable way. This challenge lies in its inherently open-ended and multi-objective nature, where ideas should balance novelty, plausibility and feasibility. While recent LLM-based approaches have made progress through carefully designed prompts or agent pipelines, they are constrained by predefined, static ideation workflows. To address this limitation, we propose MindFlow, a framework that explicitly formulates ideation as a graph-structured Flow in Mind, which is composed of modular thinking operators and modeled by a probabilistic mind supernet. Given a research topic, a controller dynamically samples thinking flows to generate candidate ideas. This open-ended problem is optimized using a tournament-based relative ranking, enabling the controller to progressively favor higher-quality thinking
    
[^39]: MiMo-V2.6：扩展强化学习以迈向自我提升

    MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement

    [https://arxiv.org/abs/2610.11959](https://arxiv.org/abs/2610.11959)

    MiMo-V2.6通过在混合SWA架构基础上，沿批次与吞吐量、环境多样性、评分器计算三个维度系统性地扩展强化学习计算量，推动全模态模型迈向自我提升。

    

    强化学习（RL）是推动大型基础模型走向自我提升的核心训练范式。本报告介绍了MiMo-V2.6系列，这是一个全模态模型家族，通过扩展RL计算量来推动模型智能的前沿。在RL之前，我们在广泛的多模态语料库上进行中期训练，以提供充足的探索空间，并在预训练的混合SWA（hybrid-SWA）架构上构建了坚实的基础设施，以支持后续的规模扩展。我们沿三个维度扩展RL计算量：（1）更大的批次和更高的吞吐量，采用异步训练方式，每步消耗1,568个样本和27亿至37亿个token，上下文长度可达100万；（2）更多样化和更复杂的环境，在多种智能体框架的混合下覆盖代码、通用、视觉和网络（cyber）等领域；（3）更多的评分器计算，通过分组智能体评分（groupwise agentic grading）为长周期任务产生更准确的奖励信号，并引导模型朝……（原文在此截断）

    arXiv:2610.11959v1 Announce Type: new  Abstract: Reinforcement learning (RL) is the central training paradigm for advancing large foundation models towards self-improvement. This report introduces the MiMo-V2.6 series, an omni-modal family that pushes the frontier of model intelligence by scaling RL compute. Prior to RL, we conduct mid-training on a broad multimodal corpus to provide ample exploration space, and build a solid infrastructure on the pretrained hybrid-SWA architecture to support subsequent scale-up. We scale RL compute along three dimensions: (1) larger batches and higher throughput, with an asynchronous training that consumes 1,568 samples and 2.7-3.7B tokens per step at context lengths of up to 1M; (2) more diverse and complex environments, spanning code, general, visual, and cyber domains under a mixture of agent harnesses; and (3) more grader compute, via groupwise agentic grading that yields more accurate reward signals for long-horizon tasks and steers the model tow
    
[^40]: 历史何时有用何时有害：多模态对话轮次中的选择性历史使用

    When History Helps and Hurts: Selective History Use across Multimodal Turns

    [https://arxiv.org/abs/2610.11948](https://arxiv.org/abs/2610.11948)

    该论文提出ReTurn基准（7,000个涵盖视觉与音频的任务），首次将多模态多轮对话中的历史使用需求与问题难度解耦，以系统评估模型对对话历史的选择性使用能力。

    

    可靠的多模态交互依赖于对对话历史的选择性使用：较早的问题可能仍然相关，但其先前的答案可能已经过时；而当前的请求即使面临相互冲突的新观察，也可能依赖于历史证据。现有的多轮评估很少将这种历史使用需求与底层问题难度区分开来。为填补这一空白，我们提出了ReTurn，一个包含7,000个基础任务、涵盖视觉与音频证据的基准，用于评估选择性历史使用。对于携带任务的历史，Reconfirm/Reground要求将历史问题应用于当前媒体，同时改变历史答案的一致性；对于携带证据的历史，Retrieve/Rebind要求利用历史媒体回答当前问题，同时改变当前媒体的竞争程度。每对任务都保持目标问题、媒体和答案不变。这些任务支持开放式和多选题评估，并配有匹配的单轮对照任务……

    arXiv:2610.11948v1 Announce Type: new  Abstract: Reliable multimodal interaction depends on selective use of conversational history: an earlier question may remain relevant while its previous answer is outdated, whereas a current request may depend on historical evidence despite conflicting new observations. Existing multi-turn evaluations rarely separate these history-use demands from underlying question difficulty. To address this gap, we introduce ReTurn, a benchmark of 7,000 base tasks spanning visual and audio evidence for evaluating selective history use. For task-carrying history, Reconfirm/Reground require applying a historical question to current media while varying historical agreement; for evidence-carrying history, Retrieve/Rebind require answering a current question using historical media while varying current-media competition. Each pair preserves the target question, media, and answer. Tasks support open-ended and multiple-choice evaluation, with matched single-turn coun
    
[^41]: 温室计划：迈向完全开放与自主可控的智能体搜索

    Project Greenhouse: Progress Toward Fully Open and Sovereign Agentic Search

    [https://arxiv.org/abs/2610.11922](https://arxiv.org/abs/2610.11922)

    温室计划证明，仅用适度计算资源和公开数据集，通过从零预训练加有监督微调的两步简单流程、不依赖第三方骨干模型，即可构建完全开放且自主可控、具有竞争力的智能体搜索重排序模型。

    

    温室计划是我们对一个简单命题的探索：我们相信，仅需适度的计算资源，就有可能构建完全开放且自主可控的智能体搜索模型。作为第一个里程碑，我们描述了如何构建一个具有竞争力的逐点式仅解码器重排序模型，其方法是一个简单的两步流程：从零开始的预训练，随后进行有监督微调，且仅从公开可用的数据集出发。与文献中的主流方法不同，我们不依赖第三方的现成开源权重骨干模型，因此我们对模型训练拥有端到端的完全掌控。我们仅使用少量GPU就完成了大部分实验。本报告阐述了该方法的重要性和优势，并分享了相关成果，使模型训练的各个方面都能够被透明、独立地复现。

    arXiv:2610.11922v1 Announce Type: cross  Abstract: Project Greenhouse represents our exploration of a simple thesis: We believe that it is possible to build fully open and sovereign models for agentic search with only modest computational resources. As a first milestone, we describe how to build a competitive pointwise decoder-only reranker using a simple two-step recipe comprising pre-training from scratch followed by supervised fine-tuning, starting only from commonly available datasets. Contrary to the dominant approach in the literature, we do not rely on existing open-weight backbones from third parties, and thus we are fully in control of model training, from end to end. We were able to accomplish the bulk of our experiments using no more than a handful of GPUs. This report articulates the importance and benefits of our approach, and we share artifacts that enable transparent, independent reproduction of all aspects of model training. Beyond data, code, and configurations that ca
    
[^42]: 面向长期对话代理的事件中心记忆与查询感知图增强方法

    Event-Centric Memory with Query-Aware Graph Augmentation for Long-Term Conversational Agents

    [https://arxiv.org/abs/2610.11920](https://arxiv.org/abs/2610.11920)

    提出了受人类记忆启发的QGMem框架，它将长对话历史转化为事件索引的原子记忆单元，并通过查询感知的图增强将与查询相关的记忆建模为工作记忆，从而提升长期对话代理的记忆构建与激活能力。

    

    对于持久化和个性化的对话代理而言，记忆系统能够通过存储过去的交互并检索相关信息，使其能够对长历史进行记忆、更新和推理。现有的记忆系统通常遵循两种范式：扁平结构记忆和基于图的记忆。前者轻量但将事件关系和状态更新隐式化，后者显式地建模记忆结构，但会产生额外的构建成本，并在长历史中引入无关的关系。为了解决这些局限性，我们提出了QGMem，这是一种受人类记忆启发的新型记忆构建与激活框架，其中经验被组织成事件，而与查询相关的事件通过图被建模为工作记忆。QGMem将长对话历史转换为以事件为索引的原子记忆单元，以保留个体经验，并将相关单元整合为动态记忆痕迹……

    arXiv:2610.11920v1 Announce Type: new  Abstract: For persistent and personalized conversational agents, memory systems can enable them to remember, update, and reason over long histories by storing past interactions and retrieving relevant information. Existing memory systems typically follow two paradigms: flat-structured memory and graph-based memory. The former is lightweight but leaves event relations and state updates implicit, while the latter explicitly models memory structure but incurs additional construction cost and introduces irrelevant relations over long histories. To address these limitations, we propose QGMem, a novel memory construction and activation framework motivated by human memory, in which experience is organized into events and query-relevant events are modeled by graph as working memory. QGMem converts long dialogue histories into event-indexed atomic memory units that preserve individual experiences and consolidates related units into dynamic memory traces th
    
[^43]: 并非所有改变都是必要的：大语言模型遗忘中的可恢复漂移

    Not Every Change Is Necessary: Recoverable Drift in Large Language Model Unlearning

    [https://arxiv.org/abs/2610.11915](https://arxiv.org/abs/2610.11915)

    提出PTP-U框架，先通过局部解析编辑削弱目标知识关联，再将非目标输出分布与原始模型对齐，从而在实现机器遗忘的同时恢复因遗忘而被附带损害的非目标能力。

    

    大语言模型中的机器遗忘旨在移除不需要的知识，同时保留模型的其他能力。尽管现有方法采用了保留目标或限制编辑发生位置的策略，但在达到期望的遗忘水平时，仍可能留下损害非目标行为的附带改变。我们的恢复比较实验表明，其中一些改变可以在保持已观察到的遗忘性能的同时被逆转。在这项工作中，我们提出了“提议-投影遗忘”（Propose-Then-Project Unlearning，PTP-U）框架，该框架将目标遗忘与非目标能力恢复相结合。PTP-U首先应用局部解析编辑来削弱目标知识的关联，然后将非目标输出分布与原始模型的分布对齐，以便在维持固定遗忘约束的同时恢复能力。两个阶段服务于一个共同目标：在满足遗忘要求的同时，保持流畅的生成能力与性能。

    arXiv:2610.11915v1 Announce Type: new  Abstract: Machine unlearning in large language models aims to remove unwanted knowledge while preserving the model's remaining capabilities. Although existing methods use retention objectives or restrict where edits occur, achieving the desired forgetting level can still leave collateral changes that impair non-target behavior. Our recovery comparisons suggest that some of these changes can be reversed while preserving observed forgetting performance. In this work, we present Propose-Then-Project Unlearning (PTP-U), a framework that combines targeted forgetting with the recovery of non-target capabilities. PTP-U first applies local analytic edits to weaken target knowledge associations, then aligns non-target output distributions with those of the original model to recover capabilities while maintaining fixed forgetting constraints. Both stages serve a common goal: satisfying the forgetting requirements while preserving fluent generation and perfo
    
[^44]: 决策模型能理解立场吗？Jev与通用大语言模型的对比评估

    Can Decision Models Understand Stance? Evaluating Jev Against General-Purpose LLMs

    [https://arxiv.org/abs/2610.11901](https://arxiv.org/abs/2610.11901)

    本研究首次将专用决策模型Jev应用于立场检测任务，发现其在英文文本数据集VAST上可媲美GPT-5.6并超越其他通用大语言模型，但在中文对话数据集ZS-CSD上表现欠佳，其局限主要源于对回复关系和立场方向的理解不足。

    

    立场检测需要识别作者对给定目标的态度，有时还需基于对话上下文进行判断。Jev是一种专为结构化决策设计的专用决策模型，为通用大语言模型（LLM）提供了一种替代方案。在这项工作中，我们在两个立场检测数据集上评估了Jev：VAST（英文文本）和ZS-CSD（中文对话），并将其与四个通用大语言模型和两个微调模型进行了比较。结果表明，Jev在VAST上取得了有竞争力的表现，与GPT-5.6相当，并优于其他通用大语言模型。然而，在ZS-CSD上，它落后于更强的LLM，尤其体现在区分支持与反对立场的方面。进一步的分析表明，这种局限可能与理解回复关系和立场方向有关，而不仅仅是受对话长度的影响。这些发现既凸显了Jev在立场检测任务上的潜力，也揭示了其局限性。

    arXiv:2610.11901v1 Announce Type: new  Abstract: Stance detection requires identifying an author's attitude toward a given target, sometimes based on conversational context. Jev, a specialized decision model designed for structured decision-making, offers an alternative to general-purpose large language models (LLMs). In this work, we evaluate Jev on two stance detection datasets, VAST (English texts) and ZS-CSD (Chinese conversations), comparing it with four general-purpose LLMs and two fine-tuned models. Results show that Jev achieves competitive performance on VAST, matching GPT-5.6 and outperforming the other general-purpose LLMs. However, it falls behind stronger LLMs on ZS-CSD, particularly in distinguishing favor from against. Further analysis suggests that this limitation may be related to understanding reply relationships and stance direction rather than conversation length alone. These findings highlight both the potential and limitations of Jev for stance detection.
    
[^45]: 从LLM对话到自主AI智能体系统：LLM集成应用的形式

    Forms of LLM-Integrated Applications from LLM-Chats to Autonomous AI Agent System

    [https://arxiv.org/abs/2610.11899](https://arxiv.org/abs/2610.11899)

    该综述系统考察了chatbot、copilot、RAG、agent等LLM集成应用标签背后的真实架构内涵，归纳出七种反复出现的应用形式，并发现四家主流厂商的编程智能体均采用委派子智能体的“推理-行动”循环这一共同架构。

    

    arXiv:2610.11899v1 公告类型：cross 摘要：大语言模型（LLM）正日益作为组件被嵌入软件系统中，并以聊天机器人、副驾驶、检索增强生成、工作流、编程智能体和AI智能体等标签进行推广。这些标签究竟代表真正的架构形式，还是仅作为品牌营销手段，此前尚未得到系统性评估。在所调查的资料来源中，这些标签确实承载了架构内涵，这在厂商的使用中体现得最为明显：copilot（副驾驶）表示一种在用户逐步确认下操作宿主应用的路由器-工作器架构，而近期转向agent（智能体）这一标签则与AI规划的多步骤执行相吻合，用户只能看到执行结果。四家主要提供商的编程智能体共享同一种架构，即一种委派给子智能体的“推理-行动”循环。本综述描述了七种反复出现的形式——LLM对话、自定义智能体、检索增强生成（RAG）、AI增强工作流、副驾驶、编程智能体，以及……

    arXiv:2610.11899v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly embedded as components in software systems, marketed under labels such as chatbot, copilot, retrieval-augmented generation, workflow, coding agent and AI agent. Whether these labels denote genuine architectural forms or serve as branding has not been assessed systematically.   In the sources surveyed, labels do carry architectural content, most clearly in vendor usage: copilot denotes a router-worker architecture operating a host application under step-by-step user confirmation, while the more recent shift to the label agent coincides with AI-planned multi-step execution of which the user sees only the outcome. The coding agents of four major providers share one architecture, a reason-and-act loop delegating to subagents.   This survey describes seven recurring forms---LLM chats, custom agents, retrieval-augmented generation (RAG), AI-enhanced workflows, copilots, coding agents, and, in par
    
[^46]: GRPODropout：在线强化学习轨迹采样中的“少即是多”

    GRPODropout: Less is More for Online Reinforcement Learning Rollouts

    [https://arxiv.org/abs/2610.11854](https://arxiv.org/abs/2610.11854)

    提出GRPODropout方法，通过在GRPO策略更新前选择性地移除少量高概率的正优势轨迹并重新居中保留的优势，有效缓解策略熵坍缩问题，提升大语言模型的推理能力。

    

    诸如GRPO之类的强化学习方法能够显著提升大语言模型的推理能力，但常常面临策略熵坍缩问题：采样多样性的丧失会削弱探索能力，并限制模型的进一步提升。现有方法要么通过算法层面的干预来应对这一问题，例如修改奖励和使用熵/KL正则化，要么通过token级别的重新加权。我们从一个互补的角度进行研究：熵坍缩也可以通过改变哪些生成的轨迹参与策略更新来缓解。在相同的采样预算下，并非所有轨迹都对更新有正向贡献，选择性地排除其中一部分反而能够改善学习效果。为此，我们提出了GRPODropout：在标准更新之前，我们采用一种简单的策略，选择性地移除少量高概率的正优势轨迹，并对所保留的优势进行重新居中。为了论证这一设计，我们…

    arXiv:2610.11854v1 Announce Type: cross  Abstract: Reinforcement learning (RL) methods such as GRPO substantially improve large language model reasoning but often suffer from policy entropy collapse: the loss of sampling diversity weakens exploration and limits further improvement. Existing methods address this issue either through algorithm-level interventions, such as reward modification and entropy/KL regularization, or through token-level reweighting. We investigate a complementary perspective: entropy collapse can also be mitigated by changing which generated rollouts contribute to policy updates. Under the same sampling budget, not all rollouts contribute positively to an update, and selectively excluding some can improve learning. To address this, we propose GRPODropout: before the standard update, we use a simple strategy that selectively removes a small number of high-probability positive-advantage rollouts and recenters the retained advantages. To motivate this design, we dev
    
[^47]: 利用大语言模型检测临床试验中的歪曲性报道

    Detecting Spin in Clinical Trials with Large Language Models

    [https://arxiv.org/abs/2610.11845](https://arxiv.org/abs/2610.11845)

    该研究提出了一种基于开源大语言模型的临床试验结局转换自动检测方法，通过提示工程、基于token概率的分类和多数投票，在测试集上达到F1分数0.78和准确率0.90，性能优于基线文本相似度模型。

    

    临床试验中的歪曲性报道包括歪曲结果呈现方式的报告行为。这在医学领域尤为关键：在未能达到统计学显著性的随机对照试验中，超过50%存在歪曲性报道现象。比较主要结局与报告结局对于检测多种类型的歪曲性报道（包括结局转换）至关重要。我们使用300对标注了语义相似度的结局数据，开发了一套自动检测结局转换的系统。我们评估了基线文本相似度模型和开源大语言模型，使用生成的相似度分数和Youden指数来确定分类阈值。所提出的方法包括提示工程、基于token概率的分类以及用于最终决策的多数投票。在包含2,496个样本的测试集上，该方法取得了F1分数0.78、准确率0.90的结果，优于基线文本相似度模型。

    arXiv:2610.11845v1 Announce Type: new  Abstract: Spin in clinical trials includes reporting practices that distort the presentation of results. This is particularly critical in medicine, where spin is present in more than 50% of randomized controlled trials that fail to reach statistical significance. The comparison of primary and reported outcomes is crucial for detecting several types of spin, including outcome switching. We used 300 pairs of outcomes labeled with semantic similarity to develop a system for automatic detection of outcome switching. We evaluated baseline text similarity models and open-source LLMs using generated similarity scores and the Youden index to determine the classification threshold. The proposed approach involves prompt engineering, classification based on token probabilities, and majority voting for the final decision. The results on the test set of 2,496 examples with an F1 score of 0.78 and an accuracy of 0.90 outperform baseline text similarity models b
    
[^48]: Memento 3：通过反思式规则手册实现基于模型的递归自我改进

    Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks

    [https://arxiv.org/abs/2610.11794](https://arxiv.org/abs/2610.11794)

    Memento 3 让冻结参数的 LLM 智能体将可修正的环境假设记录为自然语言“规则手册”并编译为可执行代码，通过预测误差驱动的观察—反思—修订—编译—验证循环，实现显式世界模型的持续递归自我改进。

    

    arXiv:2610.11794v1 公告类型：新论文 摘要：学习在不熟悉的环境中行动，要求智能体推断世界的运作方式，并随着新证据的到来不断修正这一理解。然而，有限的观测数据可以支持多个世界模型，它们都能解释过去的交互，却在未见过的状态下预测出不同的结果。我们提出了 Memento 3，它在 Memento 系列的基础上，使冻结参数的大语言模型（LLM）智能体能够借助外部记忆持续学习显式的世界模型。该智能体维护一份自然语言规则手册作为持久的语义记忆，记录关于环境动态的可修正假设，同时将未知的方面保持为未指定状态。智能体将这份规则手册编译为可执行代码，用于预测与规划。通过观察、反思、规则修订、编译与验证的持续循环，智能体利用预测误差来同时改进规则手册及其代码。更新后的代码只有在 LLM 判定其忠实于规则手册时才会被接受……（原文摘要在此处截断）

    arXiv:2610.11794v1 Announce Type: new  Abstract: Learning to act in unfamiliar environments requires agents to infer how the world works and revise that understanding as new evidence arrives. Yet limited observations can support multiple world models that explain past interactions but predict different outcomes in unseen states. We introduce Memento 3, building on the Memento series to enable frozen LLM agents to continually learn explicit world models through external memory. The agent maintains a natural-language rulebook as persistent semantic memory, recording revisable hypotheses about environment dynamics while leaving unknown aspects underspecified. It compiles this rulebook into executable code for prediction and planning. Through a continual loop of observation, reflection, rule revision, compilation, and verification, the agent uses prediction errors to refine both the rulebook and its code. Updated code is accepted only when the LLM judges it faithful to the rulebook and cel
    
[^49]: 易于预料，难以计算：边界依赖性发现熵分块所遗漏的计算型输出

    Easy to anticipate, hard to compute: boundary dependence finds the computed outputs that entropy patching misses

    [https://arxiv.org/abs/2610.11790](https://arxiv.org/abs/2610.11790)

    论文揭示了字节级语言模型中基于熵的分块策略会系统性遗漏“类型可预料但数值需计算”的位置（如数学等式的计算结果），并提出将熵信号与无标签的边界依赖信号相结合来放置补丁边界，从而大幅提升模型在这类计算输出上的准确率。

    

    字节级语言模型（如字节潜在Transformer，BLT）将字节分组为补丁，并对每个补丁运行一次大型全局模型。BLT 在小型模型的下一字节熵较高处开启新补丁，使全局计算集中在下一字节难以预测的位置。我们证明这一规则存在系统性盲区：那些类型可预测但数值必须计算的位置，例如数学解题过程中"="号之后的数字。在补丁预算紧张的情况下，熵触发的布局会跳过这些位置，导致其准确率大幅下降。在 Meta 的 BLT-1B 模型上，当补丁起点设在 10% 的字节处时，熵规则仅在 GSM8K 解题中 16% 的计算结果处放置补丁起点，且仅得到 19.0% 的完全正确率；在相同补丁数量下，在每个"="号后设置边界可获得 51.8% 的准确率，而将熵与无标签的边界依赖信号相结合可获得 67.1%（默认布局在 26% 字节处为 76.8%）。这一差距在将 BLT-1B 适配到……之后仍然存在。

    arXiv:2610.11790v1 Announce Type: new  Abstract: Byte-level language models such as the Byte Latent Transformer (BLT) group bytes into patches and run their large global model once per patch. BLT starts a patch where a small model's next-byte entropy is high, so global compute goes where the next byte is hard to predict. We show that this rule has a systematic blind spot: positions whose type is predictable but whose value must be computed, such as the number after "=" in a worked math solution. Under tight patch budgets, entropy-triggered layouts skip these positions, and accuracy on them collapses. In Meta's BLT-1B with patch starts on 10% of bytes, the entropy rule puts a patch start at 16% of the computed results in GSM8K solutions and gets 19.0% of them exactly right; a boundary after each "=" at the same patch count gets 51.8%, and entropy combined with a label-free boundary-dependence signal gets 67.1% (default layout at 26% of bytes: 76.8%). The gap survives adapting BLT-1B to 
    
[^50]: 从稀疏表征到行为洞察的多模态抑郁评估

    From Sparse Representations to Behavioral Insights for Multimodal Depression Assessment

    [https://arxiv.org/abs/2610.11787](https://arxiv.org/abs/2610.11787)

    提出了可解释的多模态抑郁评估框架BehavDep，将多模态行为表征分解为与行为语义概念关联的稀疏潜在因子，并通过弱监督学习视频级抑郁倾向得分与跨观察聚合，在实现最佳评估性能的同时揭示各模态的互补贡献。

    

    多模态抑郁评估为分析与抑郁相关的行为模式提供了一种有前景的方法。然而，现有方法通常依赖于稠密且不透明的多模态表征，导致难以解释其预测背后的行为模式。在这项工作中，我们提出了BehavDep，一个基于稀疏因子的框架，它将多模态行为表征分解为稀疏的潜在因子，并通过语义桥梁将这些因子与具有行为意义的概念相关联。为了解决用户级标注与异构视频级行为之间的不匹配问题，BehavDep进一步在弱监督下学习视频级抑郁倾向得分，并聚合多次观察的信息以完成用户级评估。大量实验表明，BehavDep在取得最佳整体评估性能的同时，还揭示了互补的模态贡献与异构……

    arXiv:2610.11787v1 Announce Type: new  Abstract: Multimodal depression assessment offers a promising approach to analyzing behavioral patterns associated with depression. However, existing methods often rely on dense and opaque multimodal representations, making it difficult to interpret the behavioral patterns underlying their predictions. In this work, we introduce BehavDep, a sparse factor-based framework that decomposes multimodal behavioral representations into sparse latent factors and associates them with behaviorally meaningful concepts through a semantic bridge. To address the mismatch between user-level annotations and heterogeneous video-level behaviors, BehavDep further learns video-level depression tendency scores under weak supervision and aggregates information across multiple observations for user-level assessment. Extensive experiments demonstrate that BehavDep achieves the best overall assessment performance while revealing complementary modality contributions, hetero
    
[^51]: DPPM：面向个性化语言模型的双路径参数化记忆

    DPPM: Dual-Path Parametric Memory for Personalized Language Models

    [https://arxiv.org/abs/2610.11776](https://arxiv.org/abs/2610.11776)

    DPPM通过“证据路径”池化交互历史以保留早期证据、“增量路径”按序更新关联状态以捕捉偏好变化，融合生成历史条件化的LoRA适配器，在个性化任务上显著超越基线（PersonaMem-v2达54.22%，PrefEval达86.79%）。

    

    长期个性化要求语言模型利用交互历史来跨会话地追踪用户偏好。参数化记忆将交互历史编码进模型参数或适配器中，从而减少了在推理上下文中包含该历史的需求。然而，独立的上下文编译方式使跨会话整合缺乏明确规范，而循环式的更新则可能削弱早期证据。为应对这些挑战，我们提出了双路径参数化记忆。其证据路径直接对交互历史的表征进行池化以保留早期证据，而其增量路径则按序更新一个关联状态以捕捉变化。融合两条路径的输出可生成以历史为条件的LoRA适配器，将证据积累与有序修订相结合。在多个骨干模型上，DPPM均超越了所评估的基线方法，在PersonaMem-v2上达到54.22%，在PrefEval上达到86.79%。这些结果表明，D

    arXiv:2610.11776v1 Announce Type: new  Abstract: Long-term personalization requires language models to use interaction history to track users' preferences across sessions. Parametric memory encodes this interaction history into model parameters or adapters, reducing the need to include it in the inference context. However, independent context compilation leaves cross-session integration unspecified, while recurrent updates can attenuate earlier evidence. To address these challenges, we propose Dual-Path Parametric Memory (DPPM). Its Evidence path directly pools representations of the interaction history to preserve earlier evidence, while its Delta path sequentially updates an associative state to capture changes. Fusing both outputs produces history-conditioned LoRA adapters that combine evidence accumulation with ordered revision. Across multiple backbones, DPPM outperforms the evaluated baselines, achieving 54.22% on PersonaMem-v2 and 86.79% on PrefEval. These results suggest that D
    
[^52]: RouterInterp：理解专家混合模型路由中的叠加特化

    RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing

    [https://arxiv.org/abs/2610.11775](https://arxiv.org/abs/2610.11775)

    本文提出叠加特化假说，认为MoE专家专精于细粒度特征的组合而非单一领域，并据此开发RouterInterp方法，通过稀疏自编码器特征与自然语言解释来解读专家路由，检测准确率比先前方法高出约65%。

    

    稀疏混合专家模型通过将词元路由到模块化的专家网络（这些网络仅在处理一小部分词元时被激活），比稠密模型更具扩展效率。关于MoE模型性能的一个主流假设是：每个专家专精于一个单一、连贯的领域。然而，基于这一假设的可解释性研究工作通常并不成功。我们提出并为另一种解释提供了证据，我们称之为叠加特化假说：专家专精于细粒度特征的不相交并集，而非某一宽泛领域。利用SSH，我们提出了RouterInterp，这是一种解释专家路由的方法，它能识别对路由决策最具预测性的稀疏自编码器特征，并生成统一的自然语言解释。在gpt-oss-20b上，RouterInterp解释专家路由的检测准确率比先前的基于词元统计的方法高出约65%。

    arXiv:2610.11775v1 Announce Type: new  Abstract: Sparse Mixture of Experts (MoE) models scale more efficiently than dense models by routing tokens to modular expert networks that are only active for processing a fraction of tokens. A leading hypothesis for the performance of MoE models is that each expert specialises in a single, coherent domain. However, interpretability efforts that assume this hypothesis have generally been unsuccessful. We propose and present evidence for an alternative account that we call the Superposed Specialisation Hypothesis (SSH): experts specialise in a disjoint union of fine-grained features rather than one broad domain. Leveraging the SSH, we introduce RouterInterp, a method for interpreting expert routing that identifies Sparse Autoencoder features most predictive of routing decisions and produces unified natural language explanations. On gpt-oss-20b, RouterInterp explains expert routing with ${\sim}65\%$ higher detection accuracy than prior token statis
    
[^53]: 相同结果，不同证据：LLM安全评估中的意图恢复

    Same Outcome, Different Evidence: Intent Recovery in LLM Safety Evaluation

    [https://arxiv.org/abs/2610.11766](https://arxiv.org/abs/2610.11766)

    该论文提出将攻击成功率（ASR）与操作性理解率（UR）配对使用，以区分LLM安全评估中相同无害结果背后的不同原因，揭示出相似的ASR值可能掩盖截然不同的任务恢复率。

    

    大语言模型的安全评估通常用攻击成功率（ASR）来概括有害输出行为。然而，相同的无害结果可能由截然不同的原因产生：模型可能恢复了有害任务并予以拒绝，可能未能恢复该任务，也可能完全回应了其他内容。在意图模糊化的提示下，区分这些情况变得尤为重要，因为低ASR并不能揭示被评估的任务是否真正得到了处理。为了明确这种区分，我们将ASR与操作性理解率（UR）配对使用，后者衡量一个响应是否既识别出被评估的任务，又将其视为需要回答的任务。在多种接口上，这种配对视角揭示了被ASR所掩盖的显著差异：相似的ASR值可能对应截然不同的恢复率。受控的英文重构实验表明，随着压缩提示变得更加明确，恢复率会持续提升。

    arXiv:2610.11766v1 Announce Type: cross  Abstract: Safety evaluations of large language models commonly summarize harmful-output behavior with attack success rate (ASR). Yet the same non-harmful outcome can arise for very different reasons. A model may recover a harmful task and refuse it, fail to recover the task, or respond to something else entirely. Distinguishing these cases becomes especially important under intent-obscuring prompts, where a low ASR does not reveal whether the evaluated task was actually engaged. To make this distinction explicit, we pair ASR with operative understanding rate (UR), which measures whether a response both identifies the evaluated task and treats it as the task to be answered. Across interfaces, this paired view reveals substantial variation hidden by ASR: similar ASR values can correspond to sharply different recovery rates. Controlled English reconstructions show that recovery consistently improves as compressed prompts become more explicit, where
    
[^54]: 思维惯性：大语言模型在被要求不思考时仍会继续思考

    Thinking Inertia: LLMs Keep Thinking When Told Not To

    [https://arxiv.org/abs/2610.11765](https://arxiv.org/abs/2610.11765)

    该论文提出了三项新指标（空思考率、问题-答案前相关性、显式推理率）来严谨地衡量大语言模型的“不思考”行为，并揭示了一种“思维惯性”现象——即使被明确要求不思考，LLM仍会继续进行推理。

    

    大语言模型（LLM）越来越普遍地配备了显式的“思考模式”，然而与其相对的“不思考”模式却很少受到关注。我们从两个维度研究LLM的“不思考”行为。a. 如何衡量“不思考”？以往的工作通常通过替代指标来定义“不思考”，例如禁用思考模式或缺少长篇推理轨迹。这些替代指标并不可靠：被禁用的思考模式仍可能输出推理内容，而长轨迹中可能包含的是填充性内容而非真正的推理。我们转而将每个响应规范化为答案前的推理轨迹和最终答案，并从三个层面进行评估：（i）空思考率（Empty-Thinking Rate），用于衡量严格的仅答案合规性；（ii）指令感知的问题-答案前相关性（Question-Pre-answer Relevance），用于衡量问题与答案前推理轨迹之间的相似度；（iii）以LLM为裁判的显式推理率（Explicit Inference Rate），用于衡量可见的显式推理行为。这些指标共同区分了仅答案输出、相关但无推理的文本……

    arXiv:2610.11765v1 Announce Type: new  Abstract: Large Language Models (LLMs) increasingly ship with explicit "thinking modes", yet their counterpart, "no-thinking", has received far less attention. We study LLMs' no-thinking behavior along two axes. a. How to measure no-thinking? Prior work typically defines no-thinking through proxies such as a disabled thinking mode or the absence of long traces. These proxies are unreliable: disabled thinking modes may still emit reasoning, while long traces may contain filler rather than genuine inference. We instead normalize each response into a pre-answer trace and final answer, and evaluate it at three levels: (i) Empty-Thinking Rate for strict answer-only compliance; (ii) instruction-aware Question-Pre-answer Relevance for similarity between the question and pre-answer trace; and (iii) LLM-as-judge Explicit Inference Rate for visible explicit inference. Together, these metrics distinguish answer-only output, relevant but non-inferential text,
    
[^55]: 面向语义物理现实的四阶张量注意力模型

    4-Tensor Attention Model for Semantic Physical Reality

    [https://arxiv.org/abs/2610.11716](https://arxiv.org/abs/2610.11716)

    该论文提出一种四阶张量注意力模型，通过在语义纤维与时间上下文纤维上进行联合归一化注意力来预测场景的下一语义状态，在参数量几乎相同的情况下，其最后一句交叉熵在三个设置上均优于自由运行的一维 Transformer（最多低 5.3%），为视频生成和机器人规划提供了新方法。

    

    我们提出了一种四阶张量注意力模型（4-tensor attention model），用于预测场景的下一个语义状态，可应用于视频生成与机器人规划。一个状态窗口包含位置 (x, t) 以及两条纤维：语义纤维和时间上下文纤维，并由一个 softmax 在整个窗口上联合归一化注意力。视频帧和智能体的情境被表示为这些状态；编码器、渲染器和规划器则不参与该更新过程。为了单独测试这一更新机制，我们在 ROCStories 数据集上进行训练，其中每个窗口在语义层上构成相同的下一句预测任务。在验证集上（每个设置使用一个随机种子），在三个匹配设置下，四阶张量模型的最后一句交叉熵均低于自由运行的一维 Transformer：在 H=2, L=2 时低 5.3%，在 H=4, L=2 时低 2.6%，在 H=4, L=3 时低 2.4%。在 H=4, L=2 时，两者的参数量几乎相同，分别为 172.5M 和 175.9M。在相同的两个 GPU 上，该四阶张量模型的运行在 2.4（原文此处截断）……

    arXiv:2610.11716v1 Announce Type: cross  Abstract: We describe a 4-tensor attention model that predicts the next semantic state of a scene, for video generation and robot planning. A window of states has positions (x, t) and two fibers, a semantic fiber and a temporal-context fiber, and one softmax normalizes attention jointly over the window. Frames and an agent's situation are written as those states; the encoder, the renderer, and the planner remain outside the update. To test the update on its own, we train on ROCStories, where each window poses the same next-sentence task at the semantic layer. On the validation split, with one seed per setting, the last-sentence cross-entropy on the three matched settings is lower for the 4-tensor model than for a free-running one-dimensional transformer by 5.3% at H=2, L=2, by 2.6% at H=4, L=2, and by 2.4% at H=4, L=3. At H=4, L=2 the parameter counts are nearly the same, 172.5M and 175.9M. On the same two GPUs that 4-tensor run finished in 2.4 
    
[^56]: 内化器：面向超大规模语言模型的便携式上下文到参数映射

    Internalizer: Portable Context-to-Parameter Mapping for Very Large Language Models

    [https://arxiv.org/abs/2610.11715](https://arxiv.org/abs/2610.11715)

    Internalizer是一种可移植的上下文到参数映射超网络，先在小模型上低成本训练后即可移植到2840亿参数的DeepSeek v4 Flash上，为其生成文档特定的LoRA适配器，使模型无需在上下文窗口中放入文档即可将知识内化到权重中。

    

    将上下文直接映射为LoRA适配器的超网络，可以让大语言模型把该上下文“内化”到自身权重中，但此前的研究仅在参数量最高达140亿的基础模型上验证过这类方法。我们提出了Internalizer——一个最先进的、可移植的“上下文到参数映射”超网络，它能为冻结参数的2840亿参数DeepSeek v4 Flash生成针对特定文档的LoRA适配器，其目标模型规模比以往任何工作大两个数量级。该超网络的大部分参数位于一个与具体模型无关的共享主干中，每个基础模型只需附加轻量的入口层和出口层，因此可以先在小模型上低成本训练，再移植到大型模型上。在最长4096个token的未见文档上，所生成的适配器在教师强制评测下达到84.9%的top-1准确率和97.8%的top-5准确率，而基础模型仅为63.4%和83.5%，此时上下文窗口中只保留一个三词的指令。超网络训练完成后，仅需一次前向……（原文在此处截断）

    arXiv:2610.11715v1 Announce Type: new  Abstract: Hypernetworks that map a context directly to a LoRA adapter let a large language model carry that context in its weights, but prior work has demonstrated them only on base models of up to 14 billion parameters.   We present the Internalizer, a state-of-the-art, portable Context-to-Parameter Mapping hypernetwork that generates document-specific LoRA adapters for the frozen 284B-parameter DeepSeek v4 Flash, a target two orders of magnitude larger than in any previous work. Most of its parameters live in a model-agnostic trunk with only thin entry and exit layers per base model, so it trains cheaply against small models before being ported to the large one.   On unseen documents of up to 4096 tokens, the generated adapters reach 84.9% top-1 and 97.8% top-5 teacher-forced accuracy against 63.4% and 83.5% for the base model, with nothing in the context window but a three-word instruction.   Once the hypernetwork is trained, a single forward p
    
[^57]: 将序列标注作为依存图解析的结构化情感分析

    Structured Sentiment Analysis Using Sequence Labeling as Dependency Graph Parsing

    [https://arxiv.org/abs/2610.11695](https://arxiv.org/abs/2610.11695)

    该论文提出将结构化情感分析建模为依存图解析任务，并创新性地通过序列标注结合线性化图编码来解决，在五种语言的七个数据集上取得了与更复杂的先进单模型方法相当的性能。

    

    本研究致力于解决结构化情感分析问题，其目标是获得一个细粒度的情感图，其中节点表示情感持有者、情感目标和情感表达文本的片段，而弧线定义了它们之间的关系。我们提出的方法将该任务建模为依存图解析，但不同于传统的解析方法，我们通过序列标注的方式来解决该问题。为此，我们利用了线性化图编码的最新进展，使输入中的每个词都能被分配一个标签，从而有效捕捉依存图的结构。我们在涵盖五种语言（英语、西班牙语、挪威语、巴斯克语和加泰罗尼亚语）的七个数据集上进行了实验，结果显示该方法的性能可与领先的、更为复杂的单模型方法相媲美。

    arXiv:2610.11695v1 Announce Type: new  Abstract: This study addresses the problem of structured sentiment analysis, whose goal is to obtain a fine-grained sentiment graph where the nodes represent spans of sentiment holders, targets, and expressions, while the arcs define the relationships among them. Our proposed approach casts the task as dependency graph parsing, but departs from traditional parsing methods by solving it through sequence labeling. To do so, we leverage recent advances in linearized graph encodings that allow each word in the input to be assigned a label, effectively capturing the structure of the dependency graph. We conducted experiments on seven datasets spanning five languages (English, Spanish, Norwegian, Basque, and Catalan), showing performance competitive with leading, more complex single-model approaches.
    
[^58]: TRACE：诊断智能体评估中验证器的脆弱性

    TRACE: Diagnosing Verifier Brittleness in Agentic Evaluation

    [https://arxiv.org/abs/2610.11678](https://arxiv.org/abs/2610.11678)

    TRACE协议通过对评估施加针对性变更并对未改变的轨迹重新评分，来诊断智能体评估中验证器分数的变化究竟是源于真实能力差异，还是评分器本身的脆弱性。

    

    验证器分数如今既作为大型语言模型（LLM）智能体的基准度量，也作为训练奖励，分数的变化通常被解读为能力的变化。然而，它也可能反映的是评估本身的变化。我们提出了TRACE，一种将分数变化从结论转化为可测试诊断的协议：它对评估的某一部分施加针对性的变更，比较成对的运行结果，检查智能体的行为是否发生改变，并对未改变的轨迹重新评分，以检验评分规则是否是问题的根源。在一个包含25个合成任务的受控测试套件中，仅仅重命名工具就使一个脚本化智能体的分数降低了0.250，尽管它执行的操作完全相同；在评分时恢复原始名称能够弥合整个差距，而同样的变更在第二个智能体中则暴露出了真实的行为失败。在使用四个LLM智能体的公开τ²-bench任务上，一项初步的30任务研究发现了混合的奖励变化，其（摘要在此处截断）

    arXiv:2610.11678v1 Announce Type: new  Abstract: Verifier scores now serve as both benchmark metrics and training rewards for large language model (LLM) agents, and a change in score is routinely read as a change in capability. It may instead reflect a change in the evaluation. We introduce TRACE, a protocol that turns a score change from a verdict into a testable diagnosis: it applies a targeted change to one part of an evaluation, compares paired runs, checks whether the agent's behavior changed, and rescores unchanged trajectories to test whether the scoring rule is responsible. In a controlled suite of 25 synthetic tasks, renaming tools lowers a scripted agent's score by 0.250 even though it performs exactly the same operations; restoring the original names at scoring time closes the entire gap, while the same mutation exposes a genuine behavioral failure in a second agent. On public $\tau^2$-bench tasks with four LLM agents, an initial 30-task study finds mixed reward changes whos
    
[^59]: DIAL-OPD：在在线策略蒸馏中用更少的Token学到更多

    DIAL-OPD: Learning More from Fewer Tokens in On-Policy Distillation

    [https://arxiv.org/abs/2610.11659](https://arxiv.org/abs/2610.11659)

    提出DIAL-OPD方法，通过用师生概率的对数均值加权奖励幅度来筛选最具学习价值的token，证明了在更少token上训练的在策略蒸馏效果可以优于全token训练。

    

    在线策略蒸馏（On-policy distillation, OPD）使用token级别的教师信号来监督学生模型生成的轨迹。其采样token变体避免了全词表概率的计算成本。然而我们发现，在更少的token上进行训练反而能胜过全token的OPD，这挑战了“更多监督能改善学习”的直觉。这促使我们根据学习价值来选择token。现有的基于分歧的筛选标准忽略了概率尺度：那些被教师和学生两个模型都赋予极小概率的token（称为“低-低token”），可能获得较大的对数比奖励，从而阻碍学习。我们提出DIAL-OPD，这是一种token选择方法，通过用教师和学生概率的对数均值对奖励幅度进行加权，从而在对数概率空间和概率空间之间架起桥梁。参数beta控制这种加权强度，并保留得分最高的token。在4对教师-学生模型和7个数学推理基准上，我们将DIAL-OPD与9个基线方法进行了比较。

    arXiv:2610.11659v1 Announce Type: cross  Abstract: On-policy distillation (OPD) supervises student-generated trajectories with token-level teacher signals. Its sampled-token variant avoids the cost of full-vocabulary probabilities. Yet we find that training on fewer tokens can outperform full-token OPD, challenging the intuition that more supervision improves learning. This motivates selecting tokens by learning value. Existing disagreement-based criteria ignore probability scale: tokens assigned negligible probability by both models, termed low-low tokens, can receive large log-ratio rewards and hinder learning. We propose DIAL-OPD, a token-selection method that bridges log-probability and probability spaces by weighting reward magnitude with the logarithmic mean of teacher and student probabilities. A parameter beta controls this weighting, and the highest-scoring tokens are retained. Across 4 teacher-student pairs and 7 mathematical reasoning benchmarks, we compare DIAL-OPD with 9 b
    
[^60]: Harness演化触及天花板：权重训练应于何时启动

    Harness Evolution Hits a Ceiling: When Weight Training Should Begin

    [https://arxiv.org/abs/2610.11655](https://arxiv.org/abs/2610.11655)

    该论文提出通过失败构成分析来决定改进长时程LLM智能体的正确杠杆——将失败区分为过程失败与内容失败，其中Harness演化修复过程失败且其行为可被训练进权重，而内容失败则需依靠权重训练来解决。

    

    改进长时程LLM智能体有两条路径：在冻结模型周围演化Harness（执行框架），或者训练模型权重。我们首先让自演化的Harness使系统变得更强，然后将种子/演化后的Harness与基础/训练后的权重进行交叉组合，以探究训练后的模型能保留哪些增益、哪些增益仍需要运行时框架支持。我们表明，正确的改进杠杆可以从智能体的失败构成中读取：通过首个触发的信号对失败轨迹进行标注，可将过程失败（调用被阻塞、陷入循环、步骤预算耗尽）与内容失败（交付的方案质量差）区分开来。Harness演化修复过程失败，其灌输的行为可以进一步训练进权重，而内容失败正是权重训练所要解决的问题。在DeepPlanning基准上，自演化Harness循环将Qwen3.5-4B的保留集得分从0.16提升至0.30，将Qwen3.5-9B的得分从0.32提升至0.44；对于4B模型，保留集交付率从55%升至90%，而内容失败……（摘要在此处截断）

    arXiv:2610.11655v1 Announce Type: new  Abstract: Improving a long-horizon LLM agent means evolving the harness around a frozen model or training its weights. We let a self-evolving harness make the system stronger first, then cross seed and evolved harnesses with base and trained weights to learn which gains the trained model keeps and which still need the runtime. We show that the right lever can be read off the agent's failure composition: labelling failed trajectories by the first signal that fires separates process failures (blocked calls, loops, exhausted step budgets) from content failures (a delivered plan that is poor). Harness evolution repairs the former, the behaviour it instils can be trained into the weights, and content failures are what weight training is for. On DeepPlanning, a self-evolving harness loop lifts the held-out score of Qwen3.5-4B from 0.16 to 0.30 and of Qwen3.5-9B from 0.32 to 0.44; for 4B, held-out delivery rises from 55% to 90% while content failures are
    
[^61]: 面向德语语音识别的音系学感知分词：一项跨领域研究

    Phonologically Informed Tokenization for German Speech Recognition: A Cross-Domain Study

    [https://arxiv.org/abs/2610.11646](https://arxiv.org/abs/2610.11646)

    本文提出基于音系学知识（Pyphen 音节划分与字素-音素转换）的分词方法用于德语端到端语音识别，发现在域内条件下其性能与 BPE 和字符基线相当，而跨领域表现主要由词表规模而非语言学分词方式决定。

    

    德语是一种形态丰富的语言，其音节结构可以被 Knuth–Liang 连字（断词）算法极好地预测。本文探究基于音系学知识的分词能否成为端到端语音识别的一个有竞争力的目标。作者在使用 CTC 微调的 Omnilingual ASR wav2vec 2.0 骨干网络上比较了三类分词器：预训练的多语言字符清单、基于正字法的数据驱动字节对编码（BPE），以及由 Pyphen 音节划分和字素到音素转换得到的音系学感知单元。通过 40 次微调实验，他们在三个跨越正交分布偏移的德语测试集上进行评估：域内朗读语音、方言自发性语音和标准德语自发性语音。在域内条件下，所有音系学感知分词器在词错误率（WER）和字符错误率（CER）上均与 BPE 及多语言字符基线持平。在分布偏移条件下，结果的分化主要沿词表规模而非语言学（此处摘要截断）展开。

    arXiv:2610.11646v1 Announce Type: new  Abstract: German is a morphologically rich language whose syllable structure is exceptionally well-predicted by the Knuth--Liang hyphenation algorithm. We ask whether phonologically informed tokenization can serve as a competitive target for end-to-end speech recognition. We compare three tokenizer families on the Omnilingual ASR wav2vec 2.0 backbone fine-tuned with CTC: the pretrained multilingual character inventory, a data-driven Byte-Pair Encoding (BPE) over orthography, and phonologically informed units from Pyphen syllabification and grapheme-to-phoneme conversion. Across 40 fine-tunes, we evaluate on three German test sets spanning orthogonal shifts: in-domain read speech, dialectal spontaneous speech, and standard-German spontaneous speech. In-domain, all phonologically informed tokenizers match BPE and the multilingual character baseline on both WER and CER. Under domain shift the picture splits along vocabulary size rather than the lingu
    
[^62]: UXBench Pro：多轮对话交互中个性化用户体验的基准测试

    UXBench Pro: Benchmarking Personalized User Experience in Multi-Turn Dialogue Interactions

    [https://arxiv.org/abs/2610.11638](https://arxiv.org/abs/2610.11638)

    UXBench Pro通过构建带用户画像的测试集，并结合个性化用户奖励模型与用户模拟器的双视角评估范式，实现了对多轮对话交互中个性化用户体验的细粒度基准测试。

    

    利用自动化计算方法评估用户体验（UX）日益受到关注，并得到了UXBench实证证据的支持。然而，二元偏好预测所能提供的洞察有限，而依赖单一的用户无关奖励模型则忽视了用户固有的异质性——不同用户的期望可能存在显著差异。在本文中，我们提出了UXBench Pro，其中包含从12个任务场景和82个领域的真实用户交互中提取的1,000个测试实例。每个实例都配有一个FACTORS用户画像，该画像通过七个可解释的行为维度刻画用户，从而区分不同的用户群体。为了提供更丰富的评估洞察，我们引入了一种双视角评估范式，该范式结合了用于第三方评判的个性化用户奖励模型（URM）与Sim4Eval——一个支持多轮交互、并能从第一人称视角进行跨四个认知阶段评估的用户模拟器。

    arXiv:2610.11638v1 Announce Type: new  Abstract: Evaluating user experience (UX) with automated computational methods has gained increasing attention, supported by empirical evidence from UXBench. However, binary preference prediction provides limited insight, while relying on a single user-agnostic reward model overlooks the inherent heterogeneity of users, whose expectations can differ substantially. In this paper, we present UXBench Pro, comprising 1{,}000 test instances derived from real user interactions across 12 task scenarios and 82 domains. Each instance is paired with a FACTORS user profile that characterizes the user through seven interpretable behavioral facets, differentiating user groups. To provide richer evaluation insights, we introduce a dual-perspective paradigm that combines a personalized User Reward Model (URM) for third-person judgment with Sim4Eval, a user simulator that enables multi-turn interactions and provides first-person evaluation across four cognitive s
    
[^63]: 大语言模型的更新迭代削弱了对人工智能辅助科学写作的筛查

    Large Language Model Turnover Undermines Screening for Artificial Intelligence-Assisted Scientific Writing

    [https://arxiv.org/abs/2610.11599](https://arxiv.org/abs/2610.11599)

    该研究用三大厂商23个版本的LLM改写4,000篇论文摘要并训练检测器，发现仅基于旧版本模型训练的AI写作检测器会在模型代际更新时性能崩溃（检出率从99%骤降至3.8%），表明LLM的快速迭代严重削弱了期刊对AI辅助写作筛查的可靠性。

    

    期刊和会议已开始对投稿稿件进行筛查，以识别使用大语言模型（LLM）撰写的文本。这种筛查的可靠性依赖于针对固定一组LLM版本的基准评估，而实际使用的模型版本却在不断更迭。本文量化了LLM的更新迭代如何影响科学稿件的筛查。研究者将《美国国家科学院院刊》（PNAS）的4,000篇前ChatGPT时代摘要，与三大厂商在2023年6月至2026年8月间发布的23个LLM版本对其进行的改写结果相配对。随后，他们在不同的维护场景下训练检测器，涵盖从针对每个新版本重新训练的检测器，到仅训练一次且永不更新的检测器。结果显示，仅在厂商过去版本上训练的检测器可能在模型代际的交界处失效：在将误报率校准为1%（即错误标记1%人类撰写摘要）的情况下，这些检测器在最急剧的代际边界之前能捕获超过99%的改写文本，而在边界之后仅能捕获3.8%。

    arXiv:2610.11599v1 Announce Type: cross  Abstract: Journals and conferences have begun to screen submitted manuscripts for text written using large language models (LLMs). The reliability of this screening rests on benchmark evaluations against a fixed set of LLM versions, while the versions in actual use keep changing. Here we quantify how this LLM turnover affects the screening of scientific manuscripts. We paired 4,000 pre-ChatGPT abstracts from the Proceedings of the National Academy of Sciences with their rewrites by 23 LLM versions from three vendors, released between June 2023 and August 2026. We then trained detectors under maintenance scenarios ranging from a detector retrained on every new version to one trained once and never updated. Detectors trained only on a vendor's past versions can collapse at the boundaries between model generations: calibrated to falsely flag 1% of human-written abstracts, they catch above 99% of rewrites just before the sharpest boundary and 3.8% j
    
[^64]: 使用Prolog探究语言模型的长程演绎推理能力

    Probing for Long-Horizon Deductive Reasoning Capabilities in Language Models with Prolog

    [https://arxiv.org/abs/2610.11592](https://arxiv.org/abs/2610.11592)

    该论文构建了合成测试平台ProloNg，系统探究前沿大语言模型在长上下文中执行Prolog演绎推理的能力，发现随着推理深度增加模型性能急剧下降，多数模型在推理深度超过10后即接近随机猜测水平。

    

    当前前沿大语言模型在理论上能够处理100万token甚至更长的上下文。但它们能在多大程度上超越简单的检索，在如此长的上下文中进行更深层次的推理？我们实证研究了LLM的长程推理能力，重点关注以Prolog表达的演绎逻辑。我们构建了ProloNg，一个用于探测Prolog长程推理的合成测试平台，该平台系统地改变问题的复杂度（推理深度），其中最难的案例推理深度达到22、上下文长度为62k。我们研究了来自5个前沿LLM系列的8个推理模型，发现随着推理深度的增长，模型性能显著下降，大多数模型在推理深度超过10后便接近随机猜测水平。

    arXiv:2610.11592v1 Announce Type: new  Abstract: Current frontier LLMs can theoretically process long contexts with 1M tokens or more. But to what extent can they go beyond simple retrieval and perform deeper reasoning over such long contexts? We empirically investigate long-horizon reasoning capabilities of LLMs, focusing on deductive logic expressed in Prolog. We construct ProloNg, a synthetic testbed to probe Prolog Long Reasoning, which systematically varies the complexity (reasoning depth) of problems, where the hardest case has a reasoning depth of 22 and 62k context length. We study 8 reasoning models across 5 families of frontier LLMs, and find that performance degrades substantially as reasoning depth grows, with the majority of models approaching chance beyond depth 10.
    
[^65]: 超越平均水平的文化对齐测量：评估印度语境下孕产妇健康大语言模型互动的框架

    Measuring Cultural Alignment Beyond the Average: A Framework for Evaluating Maternal-Health LLM Interactions in Indian Contexts

    [https://arxiv.org/abs/2610.11586](https://arxiv.org/abs/2610.11586)

    该论文提出MH-INDIC文化评估框架，通过十个孕产妇健康推理维度评估十个大语言模型在印度北部语境下的文化对齐程度，发现虽然部分模型能接近人类群体层面的分布，但所有系统的行为变异性都显著低于人类。

    

    现有的医疗健康大语言模型（LLM）评估方法主要关注事实正确性、安全性和流畅性，而对生成的互动是否反映文化情境化的医疗推理提供的洞察有限。这一局限性在孕产妇健康领域尤为重要，因为护理决策受到社会和关系规范的塑造。我们提出了MH-INDIC，这是一个针对印度北部城市和半城市语境下孕产妇健康互动的文化基础评估框架，通过孕产妇健康推理的十个维度对文化行为进行操作化。我们使用对来自印度北部城市和半城市的102名孕妇及产后女性实施的26项调查，评估了十个大语言模型。我们区分了群体层面的文化对齐与个体画像层面的行为变异。尽管若干模型能够接近人类群体层面的分布，但所有被评估系统的行为变异程度都显著较低。

    arXiv:2610.11586v1 Announce Type: new  Abstract: Existing evaluation methods for healthcare LLMs primarily assess factual correctness,safety, and fluency, while providing limited insight into whether generated interactions reflect culturally situated healthcare reasoning. This limitation is particularly important in maternal health, where care decisions are shaped by social and relational norms. We introduce MH-INDIC, a culturally grounded evaluation framework for maternal-health interactions in urban and semi-urban North Indian contexts that operationalises cultural behaviour through ten dimensions of maternal-health reasoning. Using a 26-item survey administered to 102 pregnant and postpartum women from urban and semi-urban North India, we evaluate ten LLMs. We distinguish population level cultural alignment from profile-level behavioural variation. Although several models approximate the human population-level distribution, all evaluated systems exhibit substantially lower variation
    
[^66]: 将英语质量分类器适配于多语言LLM预训练数据选择

    Adapting English Quality Classifiers for Multilingual LLM Pretraining Data Selection

    [https://arxiv.org/abs/2610.11585](https://arxiv.org/abs/2610.11585)

    提出一种多语言适配方法，通过在Transformer编码器嵌入之上训练小型多层感知机，并利用机器翻译获取标签，将英语质量分类器扩展至100多种语言的LLM预训练数据筛选，同时保持下游基准性能。

    

    大语言模型（LLM）预训练的最新进展凸显了高质量训练数据在提升性能方面的重要作用。虽然基于模型的过滤方法已被证明能有效地从网络规模语料库中筛选出高质量子集，尤其适用于高资源语言，但低资源语言由于标注数据的匮乏而面临挑战。本工作探索将质量过滤扩展到超过100种语言，提出了一种多语言适配方法，将现有的英语质量分类器转换为多语言变体。我们的方法提出在Transformer仅编码器模型的嵌入之上训练一个小型多层感知机，以多语言文本作为输入，并以英语分类器应用于机器翻译文本后所得的分数作为标签。我们在1B、3B和8B规模上的实验表明，我们的方法保持了现有多语言模型方法的下游LLM基准性能。

    arXiv:2610.11585v1 Announce Type: cross  Abstract: Recent advances in large language model (LLM) pretraining highlight the role of high-quality training data in improving performance. While model-based filtering has proven effective in selecting high-quality subsets from web-scale corpora, especially for high-resource languages, low-resource languages face challenges due to limited availability of annotated data. This work explores extending quality filtering to over 100 languages by proposing a multilingual adaptation approach that converts an existing English quality classifier into a multilingual variant. Our approach proposes training a small multi-layer perceptron on top of Transformer encoder-only model embeddings, using multilingual text as input and scores obtained from English classifiers applied to machine-translated text as labels. Our 1B, 3B and 8B scale experiments show that our approach maintains the downstream LLM benchmark performance of existing multilingual model-base
    
[^67]: Chronos 使代码智能体能够对软件演化进行推理

    Chronos Enables Code Agents to Reason over Software Evolution

    [https://arxiv.org/abs/2610.11578](https://arxiv.org/abs/2610.11578)

    Chronos 提出一个测试时框架，将历史拉取请求提炼为结构化经验卡片并通过类型化关系图谱连接，使基于 LLM 的代码智能体能够利用软件演化历史来生成补丁并在候选补丁间做出选择，从而提升代码修复效果。

    

    历史拉取请求记录了代码库当前状态背后的设计决策、兼容性约束和实现模式。与新任务相关的经验可能分布在多个相关变更中，而这些变更的描述侧重于不同的关注点。我们提出了 Chronos，一个测试时框架，使这种相互关联的历史可被基于大语言模型（LLM）的代码智能体利用。Chronos 将已合并的拉取请求提炼为结构化的经验卡片，并通过一个包含代码级、开发者意图和组织关系三类类型的图谱将它们连接起来。语义搜索用于识别入口卡片，加权的多跳扩展则检索相互关联的变更以供选择性阅读。同一份记忆同时指导候选生成和补丁选择：一个专注于补丁的变更智能体和一个验证策略智能体各自开发一个补丁，随后一个“演化管理者”参考历史记录在两者之间做出选择。在 SWE-Bench Verified 上，完整的工作流提升了（摘要至此被截断）……

    arXiv:2610.11578v1 Announce Type: cross  Abstract: Historical pull requests record the design decisions, compatibility constraints, and implementation patterns behind a codebase's current state. Experience relevant to a new task can span related changes whose descriptions emphasize different concerns. We introduce Chronos, a test-time framework that makes this connected history available to large language model (LLM)-based code agents. Chronos distills merged pull requests into structured experience cards and connects them through a typed graph of code-level, developer-intent, and organizational relations. Semantic search identifies entry cards, and weighted multi-hop expansion retrieves connected changes for selective reading. The same memory guides candidate generation and patch selection: a patch-focused change agent and a validation-strategy agent each develop a patch, and an evolution steward consults history to select between them. On SWE-Bench Verified, the full workflow improve
    
[^68]: 平滑稀疏混合专家模型的Top-k暴露边界

    Smoothing the Top-k Exposure Boundary for Sparse Mixture-of-Experts

    [https://arxiv.org/abs/2610.11575](https://arxiv.org/abs/2610.11575)

    提出弹性专家路由方法，通过在以k为中心的局部离散分布中随机采样激活专家数量，将稀疏混合专家模型中刚性的top-k选择边界软化为渐进的概率分布，在不增加计算成本的前提下缓解了竞争专家因阈值划分而导致的训练反馈不均衡问题。

    

    稀疏混合专家模型在保持每个token固定计算预算的同时，能够高效地扩展参数容量。然而，传统的训练范式强制执行静态的top-k专家选择，这将连续的路由分布转换成了刚性的阶跃函数。这一约束引入了一个脆弱的边界，使得高度竞争的专家仅因微小的分数波动就被任意地划分为完全监督区域和零反馈区域。为了解决这个问题，我们提出了弹性专家路由，它从以k为中心的局部离散分布中随机采样激活专家的数量。在多次训练迭代中，该机制将尖锐的阈值软化成渐进的概率分布。由于采样邻域保持对称，该方法在匹配确定性训练的期望计算成本的同时，保留了推理预算。大量实验表明……

    arXiv:2610.11575v1 Announce Type: new  Abstract: Sparse Mixture-of-Experts models scale parameter capacity efficiently while maintaining a fixed compute budget per token. However, traditional training paradigms enforce a static choice of top-$k$ experts, which converts a continuous routing distribution into a rigid step function. This constraint introduces a brittle boundary where highly competitive experts are arbitrarily separated into full-supervision and zero-feedback zones based on minor score fluctuations. To address this issue, we propose Elastic Expert Routing, which stochastically samples the active expert budget from a localized discrete distribution centered at $k$. Over multiple training iterations, this mechanism softens the sharp threshold into a gradual probability distribution. Because the sampling neighborhood remains symmetric, this approach matches the expected computational cost of deterministic training, while preserving the inference budget. Extensive experiments 
    
[^69]: 基于结构化框架的增量式开放式深度研究

    Incremental Open-Ended Deep Research with Structured Harness

    [https://arxiv.org/abs/2610.11566](https://arxiv.org/abs/2610.11566)

    本文提出增量式开放式深度研究（Incremental-OEDR）新范式与结构化框架，将报告视为可演化的研究状态，通过结构化表示、检索与生成实现报告的选择性增量更新和证据复用，并建立了跨越十年的时序评估框架。

    

    现有的开放式深度研究（OEDR）系统主要从零开始生成报告，这使得在需要随着新信息的出现而持续维护研究报告的场景中效率低下。我们提出了**增量式开放式深度研究**，这是一种将报告视为不断演进的研究状态的研究设定，通过保留有效知识、修订过时或不完整的内容以及融入新获得的信息来对报告进行增量更新。为支持这一设定，我们提出了**结构化框架**，它将报告表示为大纲、章节和支持证据的结构化集合，并提供结构化检索、持久化的结构化证据池以及结构化生成，以实现有选择性的报告更新和证据复用。我们进一步建立了一个跨越十年的时序评估框架，包含单步任务和长链任务评估。

    arXiv:2610.11566v1 Announce Type: new  Abstract: Existing Open-Ended Deep Research (OEDR) systems primarily generate reports from scratch, making them inefficient for scenarios where research reports need to be continuously maintained as new information emerges. We introduce \textbf{Incremental Open-Ended Deep Research (Incremental-OEDR)}, a research setting that treats a report as an evolving research state and incrementally updates it by preserving valid knowledge, revising outdated or incomplete content, and incorporating newly available information. To support this setting, we propose \textbf{Structured Harness}, which represents reports as structured collections of outlines, sections, and supporting evidence, and provides structured retrieval, a persistent structured evidence pool, and structured generation for selective report updating and evidence reuse. We further establish a temporal evaluation framework spanning ten years, with \emph{Single-Step Task} and \emph{Long-Chain Tas
    
[^70]: SWE-Journey：通过长周期、多轮交互实现对编码助手更真实的评估

    SWE-Journey: Towards More Realistic Evaluation of Coding Assistants through Long-Horizon, Multi-Turn Interaction

    [https://arxiv.org/abs/2610.11559](https://arxiv.org/abs/2610.11559)

    提出 SWE-Journey 基准，通过弱到强合成流水线自动构建长周期编码任务，并利用从真实交互数据挖掘的用户画像和用户模拟智能体重现多轮交互，从而实现对编码助手更贴近现实的评估。

    

    诸如 Claude Code 和 Codex 等编码助手已成为大语言模型（LLM）智能体的主要应用，然而现有基准测试与真实使用场景仍相距甚远，尤其是在任务时长和交互长度方面。编码助手需要在不断演进的代码仓库中完成长链条的开发工作，同时通过多轮交互反复澄清需求并调整实现方案。为弥补这些差距，我们提出了 SWE-Journey，一个用于更真实评估编码助手的基准测试。为解决任务时长差距，我们提出了一种弱到强的合成流水线，可自动构建长周期编码任务。为解决交互差距，我们从真实交互数据中挖掘出四种具有代表性的用户画像，并构建了一个用户模拟智能体，以重现真实的代码辅助交互过程。平均而言，在与软件架构师角色的交互中，模型在所请求功能上的测试通过率超过 75%，但是……（摘要原文在此处截断）

    arXiv:2610.11559v1 Announce Type: cross  Abstract: Coding assistants such as Claude Code and Codex have become a major application of LLM agents, yet existing benchmarks remain far from real-world use, particularly in task horizon and interaction length. Code assistants require completing long chains of development work in continuously evolving repositories, while repeatedly clarifying requirements and adapting implementations through multi-turn interaction. To address these gaps, we introduce SWE-Journey, a benchmark for more realistic evaluation of coding assistants. To address the task-horizon gap, we propose a weak-to-strong synthesis pipeline that automatically constructs long-horizon coding tasks. To address the interaction gap, we mine four representative user personas from real interaction data and build a user-simulation agent to reproduce realistic code-assistance interactions. On average, models pass over 75% of tests for requested functionality with software architects, but
    
[^71]: 韵律到文本：从低通滤波语音中预测文本

    Prosody-to-Text: Predicting text from low-pass filtered speech

    [https://arxiv.org/abs/2610.11544](https://arxiv.org/abs/2610.11544)

    本文首次探索“韵律到文本”任务，证明仅用截止频率约450Hz的低通滤波语音微调Whisper模型即可部分恢复原始句子（WER 36%，10%完全恢复），揭示了低频韵律特征中蕴含着可被恢复的词汇信息。

    

    虽然从文本预测韵律是该领域的一项成熟任务，但相反的方向——预测符合给定韵律模式的文本——在很大程度上仍被忽视。我们认为这很遗憾，因为这个相反的方向可能带来一些非常有趣的应用场景。因此，本文通过研究能从韵律模式中恢复出多少原始句子，迈出了“韵律到文本”方向的第一步。为此，我们仅使用最低的12个Mel频带（相当于截止频率约450Hz的低通滤波器）对Whisper模型进行微调，并获得了出人意料的准确结果（词错误率WER为36%），其中10%的话语被完全恢复，40%的话语的词错误率等于或低于25%。我们还发现，在给定正确前缀的情况下，79%的情况下下一个词元被正确预测。我们的结果表明，低频语音特征与词汇之间的关系……

    arXiv:2610.11544v1 Announce Type: new  Abstract: While predicting prosody from text is an established task in the field, the opposite direction, predicting text that fits a given prosodic pattern, remains largely overlooked. We find this unfortunate, because this opposite direction could lead to some very interesting use cases. Therefore, in this paper, we make the first steps in the prosody-to-text direction by inves- tigating how much of the original sentence can be recovered from its prosodic pattern. To this end, we fine-tune the Whis- per model using only the 12 lowest Mel bins (low-pass filter with approximately 450Hz cutoff), and obtain surprisingly accurate results (WER 36%), with 10% of utterances be- ing recovered perfectly, and 40% of utterances having Word Error Rate at or below 25%. We also find that, given the correct prefix, the next token was predicted correctly in 79% of cases. Our results suggest that the relationship between low-frequency speech features and lexical 
    
[^72]: 学习循环本身，而不仅仅是页面：面向网页生成的基于执行的循环学习

    Learning the Loop, Not Just the Page: Execution-Grounded Loop Learning for Web Generation

    [https://arxiv.org/abs/2610.11543](https://arxiv.org/abs/2610.11543)

    WebLoop提出了一种基于执行的框架，在共享策略中联合训练生成器、无需执行的评论家和精炼器，使模型学会“诊断并修复”的完整循环，从而显著提升功能性网页生成的质量。

    

    功能性网页生成越来越多地通过可执行奖励进行优化，然而现有方法大多关注最终页面的质量，而对诊断和修复不完善实现的过程探索不足。我们指出了这一设置中的一个核心挑战：生成器和精炼器会产生带有直接环境奖励的可执行产物，而处于中间环节的评论家则在不具备直接可执行结果的情况下影响下游行为。我们提出了WebLoop，一个基于执行的框架，它在共享策略中联合学习生成、评论和精炼。WebLoop利用互补信号训练一个无需执行的评论家，这些信号分别对应需求层面的判别能力和对下游的有用性——首先建立可靠的诊断能力，然后引入具有后果感知的信用分配，同时这三个角色通过组相对策略学习进行联合优化。使用Qwen3.5-9B，WebLoop达到了41.5 Overall……（摘要原文在此处截断）

    arXiv:2610.11543v1 Announce Type: new  Abstract: Functional Web generation is increasingly optimized with executable rewards, yet existing methods largely focus on the quality of the final page and leave the process of diagnosing and repairing imperfect implementations underexplored. We identify a central challenge in this setting: the Generator and Refiner produce executable artifacts with direct environment rewards, whereas the intermediate Critic influences downstream behavior without a directly executable outcome. We introduce WebLoop, an execution-grounded framework that jointly learns generation, critique, and refinement within a shared policy. WebLoop trains an execution-free Critic with complementary signals for requirement-level discriminability and downstream helpfulness, first establishing reliable diagnosis and then introducing consequence-aware credit, while all three roles are jointly optimized with group-relative policy learning. With Qwen3.5-9B, WebLoop reaches 41.5 Ove
    
[^73]: 面向多智能体LLM谈判的宪法门控与确定性恢复：针对有状态对抗性守门人的消融实验

    Constitutional Gating and Deterministic Recovery for Multi-Agent LLM Negotiation: Ablations Against a Stateful Adversarial Gatekeeper

    [https://arxiv.org/abs/2610.11542](https://arxiv.org/abs/2610.11542)

    该论文提出由5支柱运行时宪法、4层集群和认知退火组成的三部分控制栈，并通过针对有状态对抗性守门人的消融实验，验证其能消除多智能体LLM谈判中的三类模型调用浪费并实现确定性死锁恢复。

    

    多智能体LLM系统在与有状态对手谈判时会以三种方式浪费模型调用：永远无法满足对手隐藏接受条件的“礼貌循环”、触发重试的格式错误输出，以及对手要求智能体必须拒绝的事项时所陷入的合规死锁。我们研究了一个由三部分组成的控制栈——5支柱运行时宪法、4层集群（导演、三智能体多数投票、监视器、模式硬门）以及认知退火（确定性死锁检测、对智能体侧上下文的原子清除、规范化恢复消息）——并在一个已发布的对抗性守门人上进行测试，该守门人的接受规则为固定的正则表达式，其LLM仅用于渲染回复文本。该测试环境具有已知解：它衡量的是该控制栈能否在对抗集群漂移的同时执行符合宪法的策略并从死锁中恢复，而非衡量它是否有所发现。每个配置运行五次（共30次运行；……

    arXiv:2610.11542v1 Announce Type: cross  Abstract: Multi-agent LLM systems negotiating with a stateful counterpart waste model calls in three ways: polite loops that never meet the counterpart's hidden acceptance condition, malformed outputs that trigger retries, and compliance deadlocks in which the counterpart demands something the agent must refuse. We study a three-part control stack - a 5-Pillar runtime constitution, a 4-tier swarm (Director, three-agent majority vote, Monitor, schema hard gate) and Cognitive Annealing (deterministic deadlock detection, atomic purge of the agent-side context, a canonical recovery message) - against a released adversarial Gatekeeper whose acceptance rules are fixed regular expressions and whose LLM only renders reply text. The testbed has a known solution: it measures whether the stack executes a constitution-aligned strategy against swarm drift and recovers from deadlock, not whether it discovers anything. In five runs per configuration (30 runs; 
    
[^74]: 何时可以对你的网络进行剪枝？多语言语音句法分析中中间神经元的研究

    When Can You Prune Your Network? A Study of Intermediate Neurons in Multilingual Speech Parsing

    [https://arxiv.org/abs/2610.11520](https://arxiv.org/abs/2610.11520)

    本文提出一种移除中间神经网络单元的更简单端到端语音句法分析架构，在参数量减少12%的同时达到相当或更优的识别与句法分析性能，并发现中间神经网络单元在预训练编码器被冻结时有助于缩小表征差距。

    

    端到端语音句法分析（speech parsing）是最近提出的一项任务，其目标是为一段口语 utterance 同时预测出转录文本和句法树。现有的语音句法分析架构通常使用中间神经网络。在这项工作中，我们研究了中间神经网络在句法分析中的有效性，尤其是它们所扮演的角色。我们提出了一种更简单的端到端语音句法分析架构，其中移除了这些中间神经网络单元，使参数量减少了12%，同时在自动语音识别（ASR）和句法分析任务上取得了与先前方法相当甚至更好的性能。我们证明，当预训练编码器被冻结时，中间神经网络单元有助于缩小表征差距。我们对法语以及中低资源语言斯洛文尼亚语和 Naija（尼日利亚皮钦语）的语音句法分析进行了全面评估。我们进一步研究了训练数据规模和中间层所带来的影响。

    arXiv:2610.11520v1 Announce Type: new  Abstract: End-to-end speech parsing, a task recently proposed, consists in predicting both the transcription and the syntactic tree for a spoken utterance. Existing architectures for speech parsing often utilise intermediate neural networks. In this work, we examine the effectiveness of intermediate neural networks (NN) for parsing, and, specifically, what role do they play. We introduce a simpler end-to-end architecture for speech parsing, where we remove these intermediate NN units, reducing the parameters by 12%, while achieving comparable or better performance than prior method on both automatic speech recognition (ASR) and parsing. We demonstrate that intermediate NN units help reduce the representational gap when the pre-trained encoder is frozen. We do a comprehensive evaluation of speech parsing on French, and medium-low resource languages Slovenian and Naija. We further investigate the impact of the training data size and intermediate lay
    
[^75]: 残差优势：面向可验证奖励强化学习的学生相对教师引导

    Residual Advantage: Student-Relative Teacher Guidance for RL with Verifiable Rewards

    [https://arxiv.org/abs/2610.11519](https://arxiv.org/abs/2610.11519)

    提出残差优势（RA）方法，将师生概率残差转化为有界且经中心化的优势信号并与验证器优势结合，为可验证奖励强化学习提供稳定、解耦的教师引导。

    

    可验证奖励强化学习（RLVR）与在线策略蒸馏（OPD）已成为推理模型后训练的两种主要范式。RLVR为每个响应给出单一的结果标签，使其内部的步骤无法获得独立的信用分配。OPD在学生访问的前缀处提供token级别的引导，但其逐点信号并不能直接反映师生之间在整个词表上的分歧模式。稠密且无界的对数比监督会放大教师的影响，然而当学生的解题路径偏离教师的路径时，一个强大的求解器并不一定是合适的引导者。我们提出残差优势（Residual Advantage, RA），该方法将师生概率残差视为有界的单步奖励，减去学生策略下对应的状态值以形成标准优势，并在每个响应内对结果进行中心化处理后，再将其加入验证器优势。

    arXiv:2610.11519v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) and on-policy distillation (OPD) have become two main paradigms for post-training reasoning models. RLVR gives each response a single outcome label, leaving the steps inside it without separate credit. OPD provides token-level guidance at student-visited prefixes, but its pointwise signal does not directly reflect the pattern of teacher--student disagreement across the vocabulary. Dense, unbounded log-ratio supervision can amplify the teacher's influence, yet a strong solver is not necessarily a suitable guide when the student's solution paths depart from the teacher's. We propose Residual Advantage (\RA{}), which treats the teacher--student probability residual as a bounded one-step reward, subtracts the corresponding state value under the student policy to form a standard advantage, and centers the result within each response before adding it to the verifier advantage. The guidance 
    
[^76]: 现代标准阿拉伯语（MSA）在大语言模型中是否支配阿拉伯语方言？一种表示层面的分析

    Does Modern Standard Arabic (MSA) Dominate Arabic Dialects in LLMs? A Representation-Level Analysis

    [https://arxiv.org/abs/2610.11510](https://arxiv.org/abs/2610.11510)

    本研究通过对26种阿拉伯语变体的表示层面分析发现，大语言模型的内部表示中现代标准阿拉伯语并不支配阿拉伯语方言，方言表示高度重叠，从而挑战了“MSA偏向的生成源于内部表示支配”这一常见解释。

    

    大语言模型（LLM）在生成阿拉伯语时，即使输入提示是方言阿拉伯语，往往也会默认使用现代标准阿拉伯语（MSA）。一种自然的解释是模型的内部表示被MSA所支配。我们通过将Shani和Basirat（2025）提出的语言支配框架适配到26种阿拉伯语变体上，对这一假设进行了检验。在不同的层和不同的模型家族中，我们没有发现任何证据表明MSA是阿拉伯语方言的主导内部表示。相反，方言表示形成了一个密集且高度重叠的空间：与针对差异更大的语言所报告的模式相比，归一化互信息急剧下降。此外，最强的可分离性效应并不局限于中间层，而是会随模型架构的不同而向更靠后的层移动。这些发现挑战了对MSA偏向生成现象的一种常见解释：输出偏好并不……

    arXiv:2610.11510v1 Announce Type: new  Abstract: Large language models (LLMs) often default to Modern Standard Arabic (MSA) when generating Arabic, even when prompted with dialectal Arabic. A natural explanation is that their internal representations are dominated by MSA. We test this hypothesis by adapting the language-dominance framework of Shani and Basirat (2025) (https://doi.org/10.18653/v1/2025.blackboxnlp-1.7) to 26 Arabic varieties. Across layers and model families, we find no evidence that MSA acts as a dominant internal representation for Arabic dialects. Instead, dialect representations form a dense and highly overlapping space: normalized mutual information drops sharply compared to patterns reported for more distinct languages. Moreover, the strongest separability effects are not confined to intermediate layers, but can shift toward later layers depending on the architecture. These findings challenge a common interpretation of MSA-biased generation: output preference does 
    
[^77]: 超越序列：为LLM推荐系统蒸馏结构化决策记忆

    Beyond Sequences: Distilling Structured Decision Memory for LLM Recommendation

    [https://arxiv.org/abs/2610.11501](https://arxiv.org/abs/2610.11501)

    提出MARI框架，通过决策记忆库将用户历史决策归档为包含目标、约束与权衡的结构化决策记忆（SDM），使LLM推荐系统在高度相似物品的“困难选择”场景中具备可解释的决策依据。

    

    尽管大语言模型（LLM）已被应用于推荐系统，但主流方法大多只对单一类型的行为（如浏览或购买）进行建模。即使纳入了多种行为，现有方法也会将异构行为扁平化为同质的token序列，忽略了它们在决策过程中截然不同的角色。这种扁平化处理无法捕捉复杂决策中的语义层次与上下文细微差别，例如价格与质量之间的权衡。因此，在涉及高度相似物品的关键“困难选择”场景中，模型性能会下降。为弥补这一空白，我们提出了MARI（Memory-Augmented Recommendation with Interpretability，可解释的记忆增强推荐），该方法将预测建立在显式、结构化的决策证据之上。MARI维护一个决策记忆库（DMB），将用户过去的决策依据归档为结构化决策记忆（SDM）：即关于目标、约束和权衡的简明记录。这些SDM是……

    arXiv:2610.11501v1 Announce Type: new  Abstract: Despite the adoption of large language models (LLMs) in recommendation systems, prevailing approaches mostly model single-type behaviors (e.g., views or purchases). Even when incorporating multiple behaviors, existing methods flatten heterogeneous actions into homogeneous token sequences, ignoring their distinct decision-making roles. This flattening fails to capture semantic hierarchies and contextual nuances in complex decision-making, such as trade-offs between price and quality. Consequently, performance degrades in critical ``difficult-choice'' scenarios involving highly similar items. To bridge this gap, we propose MARI (Memory-Augmented Recommendation with Interpretability), which grounds predictions in explicit, structured decision evidence. MARI maintains a Decision Memory Bank (DMB) that archives users' past rationales as Structured Decision Memories (SDMs): concise records of goals, constraints, and trade-offs. These SDMs are 
    
[^78]: SAGE：面向视觉-语言解码器中视觉定位的注意力汇感知引导增强

    SAGE: Sink-Aware Guided Emphasis for Visual Grounding in Vision-Language Decoders

    [https://arxiv.org/abs/2610.11469](https://arxiv.org/abs/2610.11469)

    该论文发现视觉-语言模型解码器中的注意力汇具有分层结构——早期和晚期层会出现提示不变的注意力坍缩（PIS），而中间层才驱动视觉-语言对齐，并据此提出轻量级干预方法 SAGE，将解码器注意力从 PIS 引向与查询相关的感兴趣区域，以提升可靠性并减少幻觉。

    

    近期的大型视觉-语言模型（VLMs）将视觉编码器与大型语言模型（LLM）相结合，在多种图文任务上表现优异，但其可靠性常常受限于解码器注意力的病理现象，这些现象会抑制视觉证据并加剧幻觉。在本文中，我们重新审视了视觉注意力汇现象，并揭示了一种结构化的、依赖于层的行为：在不同提示下，早期和晚期解码器层表现出提示不变的注意力坍缩，即坍缩到相同的少数图像区域上，我们将其称为 PIS（提示不变汇，Prompt-Invariant Sinks）；而中间层则变得依赖于提示，并驱动视觉-语言的对齐。这种分层特性表明，将注意力汇视为一种统一的效应是不完整的。基于这一洞察，我们提出了 SAGE（Sink 感知引导增强），这是一种轻量级干预方法，利用从标准工具导出的词元对齐的感兴趣区域（ROI）掩码，将解码器注意力从 PIS 引导至与查询相关的感兴趣区域。

    arXiv:2610.11469v1 Announce Type: cross  Abstract: Recent large vision-language models (VLMs) pair a visual encoder with a large language model (LLM) and perform well on diverse image-text tasks, yet their reliability is often limited by decoder attention pathologies that suppress visual evidence and exacerbate hallucinations. In this paper, we revisit visual attention sinks and uncover a structured, layer-dependent behavior: across prompts, early and late decoder layers exhibit prompt-invariant attention collapse onto the same few image regions, which we term PIS (Prompt-Invariant Sinks), whereas mid layers become prompt-conditioned and drive vision-language alignment. This split suggests that treating sinks as a uniform effect is incomplete. Building on this insight, we propose SAGE (Sink-Aware Guided Emphasis), a lightweight intervention that steers decoder attention away from PIS and toward query-dependent regions of interest (ROIs) using token-aligned ROI masks derived from standa
    
[^79]: 谁来验证验证者？与自我改进智能体共同进化的可检查评分器

    Who Verifies the Verifier? Co-Evolving Inspectable Graders with Self-Improving Agents

    [https://arxiv.org/abs/2610.11464](https://arxiv.org/abs/2610.11464)

    该论文提出将验证器本身作为进化对象——即由可检查的确定性缺陷检测器组成的表达式，通过锚定参考集一致性和输出共识来选择而非依据智能体分数——从而在自我改进循环中避免奖励作弊和共同盲点，并在MBPP+上比手工种子组合提升0.21的保留一致性。

    

    我们改变了智能体：它真的变得更好了吗？每个自我改进的智能体循环都会数百次地回答这个问题，而每个答案都来自一个验证器。在开放式任务上并不存在这样的验证器，因此循环只能依靠手写的评分标准，或者用一个裸的LLM裁判来评判来自与自身类似模型的输出，这容易招致奖励作弊和共同的盲点。我们让验证器成为进化的对象：一个由小型、大多为确定性的缺陷检测器组成的可检查表达式，从聚类失败中合成，在诞生时经过门控，并基于与十项锚定参考集的一致性加上对未标注输出的共识来选择，而从不根据智能体的分数来选择。在MBPP+上，它在每个种子上都比手工编写的种子组合获得了+0.21的保留一致性，并最终超越了它所包含的裸LLM裁判。有一个发现应该改变共进化验证器的验证方式：移除锚定防护会使验证器坍缩成一个空洞的总是通过的评分器，然而

    arXiv:2610.11464v1 Announce Type: new  Abstract: We changed the agent: did it actually get better? Every self-improving agent loop answers this hundreds of times, and every answer comes from a verifier. On open-ended tasks none exists, so the loop is handed a hand-written rubric or a bare LLM judge grading output from a model like itself, inviting reward hacking and shared blind spots. We make the verifier the evolving object: an inspectable expression over small, mostly deterministic drawback detectors, synthesized from clustered failures, gated at birth, and selected for agreement with a ten-item anchored reference set plus consensus over unlabeled outputs, never for the agent's score. On MBPP+ it gains +0.21 held-out agreement over the hand-authored seed composition, on every seed, and ends ahead of the bare LLM judge it contains. One finding should change how co-evolved verifiers are validated: removing the anchor guards collapses the verifier into a vacuous always-pass grader, yet
    
[^80]: 超越语音字幕：面向对话式文本到语音的语音奖励风格规划

    Beyond Speech Captions: Speech-Rewarded Style Planning for Conversational Text-to-Speech

    [https://arxiv.org/abs/2610.11461](https://arxiv.org/abs/2610.11461)

    提出语音奖励风格规划（SRSP），利用冻结TTS模型对目标语音token的教师强制似然作为奖励、通过GRPO训练文本风格规划器，生成比文本描述伪标签更能有效控制对话式语音风格与情感的风格指令。

    

    自然语言风格描述为大型语言模型（LLM）与可控文本到语音（TTS）之间提供了一种可解释的接口。然而，将描述用作伪标签会把目标声学特征压缩为文本，且描述的保真度并不必然意味着能对特定合成器进行有效控制。我们通过实验表明，对于同一话语的不同候选指令，语音-文本对齐只能微弱地预测下游的声学相似度。因此，我们提出了语音奖励风格规划（Speech-Rewarded Style Planning, SRSP），通过冻结的下游TTS模型来训练一个基于文本的风格规划器。给定对话历史和回复文本，该规划器生成候选风格指令，并采用组相对策略优化（GRPO）进行优化，以目标语音token的教师强制似然作为奖励。在ISCSLP 2026 CoT-TTS语料库的英语子集上，SRSP取得了更高的语音风格与情感相似度。

    arXiv:2610.11461v1 Announce Type: cross  Abstract: Natural-language style descriptions provide an interpretable interface between large language models (LLMs) and controllable text-to-speech (TTS). However, using descriptions as pseudo-labels compresses target acoustics into text, and descriptive fidelity need not imply effective control of a particular synthesizer. We empirically show that speech-text alignment only weakly predicts downstream acoustic similarity among candidate instructions for the same utterance. We therefore propose Speech-Rewarded Style Planning (SRSP), which trains a text-based style planner through a frozen downstream TTS model. Given dialogue history and response text, the planner generates candidate instructions and is optimized with group-relative policy optimization (GRPO), using the teacher-forced likelihood of target speech tokens as the reward. On an English subset of the ISCSLP 2026 CoT-TTS corpus, SRSP achieves higher speech-style and emotion similarity 
    
[^81]: SAIL：通过科学感知循环实现科学智能体智能

    SAIL: Scientific Agentic Intelligence via a Science-Aware Loop

    [https://arxiv.org/abs/2610.11451](https://arxiv.org/abs/2610.11451)

    SAIL是一个通过“科学感知改进循环”训练的开放科学智能体模型（35B总参数/3B激活参数），由前沿AI智能体自动诊断其任务失败并生成针对性训练任务，使其在文献研究、科学编程和多步骤研究工作流中达到有竞争力的表现。

    

    我们推出了SAIL，一个总参数量为35B、激活参数量为3B的开放模型，面向文献研究、科学编程和多步骤研究工作流。SAIL通过一个科学感知的改进循环开发而成：基于前沿AI模型构建的智能体分析其任务失败案例，并构建针对底层能力差距的训练任务。诊断过程涵盖文献任务中的搜索与证据选择、编程中的科学假设与推理，以及较长周期研究中的规划与修订。这些智能体利用论文集合和科学代码库来构建问题、交互轨迹，以及配备所需环境和工具的可执行任务。我们在多个开发周期中重复这一循环，并通过监督微调、专项训练、多教师同策略蒸馏和智能体强化学习来训练SAIL。SAIL在科学相关任务上取得了具有竞争力的表现。

    arXiv:2610.11451v1 Announce Type: new  Abstract: We introduce SAIL, an open model with 35B total and 3B active parameters for literature research, scientific coding, and multi-step research workflows. SAIL is developed through a science-aware improvement loop: agents built on frontier AI models analyze its task failures and construct training tasks that address the underlying capability gaps. The diagnosis examines search and evidence selection in literature tasks, scientific assumptions and reasoning in coding, and planning and revision in longer investigations. The agents draw on paper collections and scientific code repositories to build problems, interaction trajectories, and executable tasks with the required environments and tools. We repeat this loop over multiple development cycles and train SAIL through supervised fine-tuning, specialist training, multi-teacher on-policy distillation, and agentic reinforcement learning. SAIL achieves competitive performance across scientific r
    
[^82]: 用作评判者的决策模型中的对抗性线索：请求呈现方式的作用

    Adversarial Cues in Decision Models Used as Judges: The Role of Request Presentation

    [https://arxiv.org/abs/2610.11436](https://arxiv.org/abs/2610.11436)

    研究发现，在结构化评判请求的某些呈现方式下，仅需在候选答案中添加一个冒号，就能使大模型评判者将明显错误的答案误判为正确，误接受率从1–3%激增至约26.5%，揭示了评判请求呈现方式带来的对抗性脆弱性。

    

    一个被指示为最终答案评分的答案评判模型，即使在较早的数值与参考答案相符的情况下，也应当拒绝明显错误的最终数值。我们证明，在候选答案中添加一个冒号就可能违反这一要求，其效果取决于结构化评判请求的呈现方式。数值型参考答案可以确证该错误，而配对干预实验将候选答案的编辑与集成系统呈现方式的选择区分开来。在200个此前未被使用的DROP和GSM8K源数据簇上，在按排序的JSON键呈现方式下，该编辑使Jev的误接受率从1.0%上升到26.0%（使用三个输出标签时），在使用已发布的四标签评分指令时从3.0%上升到26.5%。而在插入式呈现方式下，两种候选变体均被拒绝。尽管Jev在两种评分配置和两种呈现方式下都满足控制阈值，这些交互作用仍然通过了预先设定的统计显著性校正。大多数超额接受发生在候选答案……（摘要在此处截断）

    arXiv:2610.11436v1 Announce Type: new  Abstract: An answer judge instructed to grade the final commitment should reject an explicitly wrong final value even when an earlier value matches the reference. We show that adding one colon to a candidate can violate this requirement depending on the presentation of the structured judging request. Numeric references certify the error, and paired interventions distinguish the candidate edit from the integration's presentation choices. On 200 previously unused DROP and GSM8K source clusters, the edit increased Jev's false acceptance from 1.0% to 26.0% with three output labels and from 3.0% to 26.5% with the published four-label grading instruction under sorted JSON keys. Both candidate variants were rejected under insertion presentation. These interactions passed the prespecified statistical correction even though Jev met the control thresholds under both grading configurations and presentations. Most excess acceptances occurred among candidates 
    
[^83]: BioBigBird：一种用于生物医学文本长程依赖处理的稀疏注意力模型

    BioBigBird: A Sparse Attention Model for Long-Range Dependency Processing in Biomedical Text

    [https://arxiv.org/abs/2610.11430](https://arxiv.org/abs/2610.11430)

    BioBigBird是一种基于稀疏注意力机制的生物医学双向语言模型，可处理长达4096个token的长序列，并通过多任务学习联合优化命名实体识别与关系抽取，在BLURB基准上取得了与最先进模型相当的表现。

    

    虽然领域专用的大型语言模型（LLMs）已经编码了大量的生物医学知识，但它们有限的上下文窗口往往阻碍了对文本内部及跨文本之间细微关系的深入理解。为了解决这一局限，我们提出了BioBigBird，这是一个在海量生物医学文献和临床数据上预训练的双向语言模型，专门设计用于处理长程依赖关系。BioBigBird利用稀疏注意力机制来处理长达4096个token的序列，其训练采用多阶段过程以减轻大规模预训练语料库中的噪声。我们进一步通过采用多任务学习（MTL）框架来提升其性能，该框架联合优化命名实体识别和关系抽取两项任务。在BLURB基准上的全面评估表明，我们经过MTL增强的BioBigBird与最先进的模型相比取得了极具竞争力的结果。我们的工作贡献……

    arXiv:2610.11430v1 Announce Type: new  Abstract: While domain-specific Large Language Models (LLMs) have encoded vast biomedical knowledge, their limited context windows often hinder a deep understanding of nuanced relationships within and across texts. To address this limitation, we introduce BioBigBird, a bidirectional language model pre-trained on extensive biomedical literature and clinical data, specifically designed to handle long-range dependencies. BioBigBird leverages a sparse attention mechanism to process sequences up to 4096 tokens, and its training incorporates a multi-stage process to mitigate noise from the large-scale pre-training corpus. We further enhance its performance by employing a multi-task learning (MTL) framework that jointly optimizes for Named Entity Recognition and Relation Extraction. Comprehensive evaluations on the BLURB benchmark reveal that our MTL-enhanced BioBigBird achieves highly competitive results against state-of-the-art models. Our work contrib
    
[^84]: 事实胜于虚构：僧伽罗语到英语神经机器翻译中病理性幻觉的检测

    Fact over Fiction: Detection of Pathological Hallucinations in Sinhala-to-English Neural Machine Translation

    [https://arxiv.org/abs/2610.11389](https://arxiv.org/abs/2610.11389)

    该论文针对低资源的僧伽罗语到英语神经机器翻译，提出了一个无参考幻觉检测框架，通过五种语言学损坏策略构建45,000样本合成数据集并微调mDeBERTa-v3进行token级序列标注，达到了0.841的token级F1分数。

    

    神经机器翻译（NMT）模型虽然能够生成高度流畅的输出，但仍然容易受到幻觉的影响，即那些看似自然却与源文本语义无关的翻译。这种脆弱性在僧伽罗语到英语等低资源场景中尤为严重，因为较弱的跨语言对齐会导致幻觉的产生。本文介绍了一个针对该语言对的无参考幻觉检测框架。我们提出了一个包含45,000个样本的合成数据集，该数据集通过五种基于语言学动机的损坏策略的概率链生成，并带有一种语义拯救机制，利用字符级相似度来区分幻觉与形态变体。我们对mDeBERTa-v3进行了微调以用于token级序列标注，在源文本不相交的测试集上，三次随机种子下达到了0.841 ± 0.001的token级F1分数，并研究了整合神经风险分数、序列对数概率等信号的三信号集成方法。

    arXiv:2610.11389v1 Announce Type: new  Abstract: Neural Machine Translation (NMT) models, while capable of producing highly fluent outputs, remain vulnerable to hallucinations, which are translations that are natural yet semantically unrelated to the source. This vulnerability is acute in low-resource settings like Sinhala-to-English, where weak cross-lingual alignment leads to hallucinations. This paper introduces a framework for reference-free hallucination detection in this language pair. We present a 45,000-sample synthetic dataset generated through a probabilistic chain of five linguistically motivated corruption strategies, with a semantic rescue mechanism that uses character-level similarity to distinguish hallucinations from morphological variants. We fine-tune mDeBERTa-v3 for token-level sequence labelling, reaching a token-level F1 of 0.841 +/- 0.001 over three seeds on a source-disjoint test set, and study a three-signal ensemble integrating neural risk scores, sequence log-
    
[^85]: 从提示到技能库：演化功能技能库使大语言模型实现持续学习

    From a Prompt to Repertoires: Evolving Functional REpertoires Enable LLM Continual Learning

    [https://arxiv.org/abs/2610.11373](https://arxiv.org/abs/2610.11373)

    提出演化功能技能库方法，通过将单一提示扩展为不断演化的多功能技能库，克服提示优化在持续学习中的灾难性遗忘与规则过拟合问题，使大语言模型无需更新参数即可持续习得新能力。

    

    持续学习对大语言模型而言仍是一个挑战，模型需要在获取新技能和知识的同时不损害已有能力。现有方法通常通过精心设计模型参数的更新方式来应对这一挑战。相比之下，提示优化避免了代价高昂的参数更新，在单个知识密集型和推理任务上取得了与GRPO等强化学习方法相当甚至更优的性能。这引出一个自然的问题：提示优化作为一种高效的适配方法，能否直接应用于持续学习？我们的分析表明，在顺序任务适配的场景下，提示优化会遭受灾难性遗忘，且优化后的提示会不断积累过拟合于局部任务分布的规则。为解决这些局限，我们提出了演化功能技能库，它以……替代单一提示（原文摘要在此处截断）。

    arXiv:2610.11373v1 Announce Type: cross  Abstract: Continual learning remains challenging for large language models, which must enable models to acquire new skills and knowledge without degrading existing capabilities. Existing approaches typically address this challenge by carefully designing how model parameters are updated. In contrast, prompt optimization avoids costly parameter updates while achieving competitive or even superior performance to reinforcement learning methods such as GRPO on individual knowledge-intensive and reasoning tasks. This raises a natural question: \textit{Can prompt optimization, as an efficient adaptation approach, be directly applied to continual learning?} Our analysis shows that, under sequential task adaptation, it suffers from catastrophic forgetting, while optimized prompts accumulate rules that overfit to local task distributions. To address these limitations, we propose \emph{Evolving Functional REpertoires} (EFRE), which replaces a single prompt
    
[^86]: SignRAG：统一的无Gloss注释检索增强手语翻译

    SignRAG: Unified Retrieval-Augmented Gloss-Free Sign Language Translation

    [https://arxiv.org/abs/2610.11371](https://arxiv.org/abs/2610.11371)

    SignRAG提出了一个结合分层预训练、目标域检索增强和检索感知强化微调的统一框架，使无Gloss注释手语翻译能够有效适配仅解码器大语言模型。

    

    当代仅解码器架构的大语言模型（LLMs）在广泛的领域中展现出了强大的能力。然而，现有的无Gloss注释手语翻译（SLT）预训练范式主要围绕传统的编码器-解码器预训练语言模型设计，这限制了其直接应用于仅解码器LLM的能力。为了解决这一局限性，我们提出了SignRAG，这是一个结合了分层预训练、目标域检索增强和检索感知强化微调的统一框架。分层预训练首先学习具有语言学基础的手语表示，然后将手语编码器与LLM进行联合对齐，从而缓解跨模态优化不平衡的问题。在下游适配方面，SignRAG通过目标域检索库为基于参数的微调提供补充，为翻译提供针对具体实例的线索。为确保检索到的上下文得到恰当使用……

    arXiv:2610.11371v1 Announce Type: new  Abstract: Contemporary decoder-only large language models (LLMs) have demonstrated strong capabilities across a wide range of domains. However, existing pretraining paradigms for gloss-free sign language translation (SLT) are largely designed around conventional encoder-decoder pretrained language models, which limits their direct applicability to decoder-only LLMs. To address this limitation, we propose SignRAG, a unified framework combining hierarchical pretraining, target-domain retrieval augmentation, and retrieval-aware reinforcement fine-tuning. Hierarchical pretraining first learns linguistically grounded sign representations and then jointly aligns the sign encoder with an LLM, mitigating cross-modal optimization imbalance. For downstream adaptation, SignRAG complements parameter-based fine-tuning with a target-domain retrieval gallery that provides instance-specific translation cues. To ensure that retrieved contexts are used appropriatel
    
[^87]: UniData：通用多模态指令生成流水线

    UniData: Universal Multimodal Instruction Generation Pipeline

    [https://arxiv.org/abs/2610.11363](https://arxiv.org/abs/2610.11363)

    UniData是一个通用多模态指令生成流水线，能够将简单的用户需求自动转化为高质量的多轮多模态指令数据，解决了现有多模态指令生成方法模态支持有限以及难以生成多轮指令的问题。

    

    多模态大语言模型（MLLMs）正被应用于越来越广泛的现实场景中。然而，由于高昂的人力成本，为MLLMs创建高质量的多模态指令数据集仍然是一项重大挑战。尽管一些方法提出了生成指令数据的方案，但它们往往在模态支持方面存在局限，且难以生成多轮指令。为解决这些问题，我们提出了UniData——一个通用指令生成流水线，能够将简单的用户需求转化为多轮多模态指令。具体而言，UniData首先将用户需求扩展为多个多样化的事件；随后，基于这些事件，UniData集成一个any-to-any大模型用于多模态指令生成；最后，UniData通过利用指令轮次之间的相关性，纠正无关和冗余的推理流程，从而提升数据质量。

    arXiv:2610.11363v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) are increasingly being applied in a wider range of real-world scenarios. However, due to the substantial labor cost, creating high-quality multimodal instruction datasets for MLLMs remains a significant challenge. Although some methods propose to generate instruction data, they often face limitations in modality support and struggle with generating multi-round instructions. To address these problems, we introduce UniData, a universal instruction generation pipeline, to transform simple user requirements into multi-round, multimodal instructions. Specifically, UniData first expands user requirements into multiple diverse events. Using these events, UniData then integrates an any-to-any large model for multimodal instruction generation. Finally, UniData enhances data quality by correcting irrelevant and redundant inference flow, leveraging correlations between instruction rounds. To train this pipel
    
[^88]: AdaptEvo：具有演化监督的自适应智能体学习

    AdaptEvo: Adaptive Agent Learning with Evolving Supervision

    [https://arxiv.org/abs/2610.11354](https://arxiv.org/abs/2610.11354)

    AdaptEvo提出了一个在不完美监督下学习的智能体框架，通过置信度自适应GRPO平衡结果与过程奖励，并从反复失败中演化出可复用的决策知识和更精细的过程评估标准，同时在工业级多模态内容审核数据集上验证了其有效性。

    

    遵循规则的情境化决策任务要求模型将指定规则应用于具体案例的情境与证据。然而，书面规则可能在决策指导和过程评估方面留下空白，而参考判断对规则和证据的支持程度也各不相同。为了应对这些挑战，我们提出了AdaptEvo，一个在不完美监督下进行学习的框架，它将置信度自适应的策略优化与不断演化的决策知识和评估标准相结合。其训练模块采用置信度自适应GRPO（CA-GRPO），根据参考置信度来平衡结果奖励与过程奖励。其演化模块从训练案例中反复出现的失败里综合出可复用的决策知识，并细化过程评估标准，以发现此前被忽视的错误。为了支持实证评估，我们构建了一个工业级多模态内容审核数据集，包含训练集以及期内和期外测试集

    arXiv:2610.11354v1 Announce Type: new  Abstract: Rule-governed contextual decision tasks require models to apply specified rules to case-specific context and evidence. Written rules can leave gaps in decision guidance and process evaluation, while reference judgments vary in their support from the rules and evidence. To address these challenges, we introduce AdaptEvo, a framework for learning under imperfect supervision that couples confidence-adaptive policy optimization with evolving decision knowledge and evaluation rubrics. Its Training module uses Confidence-Adaptive GRPO (CA-GRPO) to balance outcome and process rewards according to reference confidence. Its Evolution module synthesizes reusable decision knowledge from recurring failures across training cases and refines process rubrics to detect overlooked errors. To support empirical evaluation, we construct an industrial multimodal content moderation dataset comprising a training set and In-Period and Out-of-Period test sets, w
    
[^89]: RL-ARC：通过推理引导的不确定性校准大型推理模型

    RL-ARC: Calibrating Large Reasoning Models via Reasoning-guided Uncertainty

    [https://arxiv.org/abs/2610.11352](https://arxiv.org/abs/2610.11352)

    RL-ARC提出了一种校准感知训练框架，将推理置信度作为辅助信号来校准答案置信度——对正确回答施加推理引导正则化、对错误回答施加过度自信惩罚，从而在不牺牲推理性能的情况下改善大推理模型在分布内外场景中的校准并缓解过度自信问题。

    

    语言模型（LM）通常使用可验证奖励的强化学习（RLVR）进行训练，以增强其推理能力。然而，由于RLVR在训练过程中没有明确考虑校准，可能导致严重的校准退化，包括过度自信。近年来面向语言模型的校准感知训练方法将不确定性估计目标纳入训练，虽然改善了校准，但在分布偏移下仍然表现出过度自信，同时牺牲了推理性能。为此，我们提出了RL-ARC，一个联合利用推理置信度和答案置信度的校准感知训练框架。具体而言，RL-ARC将推理置信度作为校准答案置信度的辅助信号：对于正确的回答，将其作为推理引导的正则化项；对于错误的回答，则将其作为过度自信惩罚项。在分布内（ID）和分布外（OOD）设置下的全面实验结果（摘要在此处截断）。

    arXiv:2610.11352v1 Announce Type: new  Abstract: Language models (LMs) are commonly trained with Reinforcement Learning with Verifiable Rewards (RLVR) to enhance their reasoning capabilities. However, since RLVR does not explicitly account for calibration during training, it can lead to severe calibration degradation, including overconfidence. Recent calibration-aware training methods for LMs, which incorporate objectives for uncertainty estimation into training, improve calibration but still exhibit overconfidence under distribution shift, while sacrificing reasoning performance. To this end, we propose RL-ARC, a calibration-aware training framework that jointly leverages reasoning confidence and answer confidence. Specifically, RL-ARC leverages reasoning confidence as an auxiliary signal for calibrating answer confidence, applying it as reasoning-guided regularization for correct cases and as an overconfidence penalty for incorrect cases. Comprehensive results across ID and OOD setti
    
[^90]: 通过隐瞒进行欺骗：语言模型明知故犯地隐藏自己的错误

    Deception by Omission: Language Models Knowingly Hide Their Mistakes

    [https://arxiv.org/abs/2610.11351](https://arxiv.org/abs/2610.11351)

    该研究通过在模型轨迹中注入合成错误，首次系统揭示了LLM普遍存在的“通过隐瞒进行欺骗”行为——在智能体场景中高达67.1%的错误未被披露，且部分情况下模型明知有错仍刻意隐瞒。

    

    大型语言模型越来越多地充当几乎不受人类监督的智能体，因此它们可能犯下的错误往往不会被发现。用户此时只能依赖模型来报告出了什么问题。诚实的模型会披露自己的错误，而具有欺骗性的模型则会隐瞒错误。然而，目前尚不清楚当前的LLM在这种情况下会如何表现。在本研究中，我们在LLM的轨迹中预先注入了合成的错误，这些轨迹模拟了聊天和智能体场景中的真实部署情况。模型在36.4%的聊天场景和67.1%的智能体场景rollout中未能披露自己的错误。在分别有2.4%和5.3%的rollout中，模型在思维链中已经意识到错误的存在，却仍然以欺骗的方式将其隐瞒。不同模型的隐瞒率各不相同：例如，Gemini 3.5 Flash在高达19.9%的智能体rollout中明知故犯地隐瞒错误。在11.9%的聊天场景和51.8%的智能体rollout中，模型对错误毫无察觉，尽管它们在事后审查时能够可靠地发现这些错误。

    arXiv:2610.11351v1 Announce Type: new  Abstract: Large language models (LLMs) increasingly act as agents with little human oversight, so potential mistakes they make can go unnoticed. Users then depend on the model to report what went wrong. An honest model discloses its mistakes, while a deceptive one conceals them. However, it is unclear how current LLMs behave in such situations. In this study, we prefill LLM trajectories with synthetic mistakes. The trajectories resemble real deployments in chat and agentic settings. Models fail to disclose their mistake in 36.4% of chat and 67.1% of agentic rollouts. In 2.4% and 5.3% of rollouts, respectively, they are aware of the mistake in their chain of thought but still deceptively conceal it. Rates vary by model: for instance, Gemini 3.5 Flash knowingly conceals mistakes in up to 19.9% of agentic rollouts. In 11.9% of chat and 51.8% of agentic rollouts, models show no awareness of mistakes, even though they reliably spot them when reviewing 
    
[^91]: 基于模式的树变换的类型检查

    Type-Checking for Pattern-Based Tree Transformations

    [https://arxiv.org/abs/2610.11337](https://arxiv.org/abs/2610.11337)

    该论文提出了基于（源模式，目标模式）对集合的有限表示的树变换模型，并证明了尽管该模型的等价性检查不可判定，但类型检查问题是可判定的。

    

    我们引入并研究了基于模式的树变换。作为一个说明性的例子，考虑源模式 $(x \cdot y) + (x \cdot z)$ 和目标模式 $x \cdot (y + z)$ 组成的一对模式。该源模式匹配任何形如 $(e_1 \cdot e_2) + (e_1 \cdot e_3)$ 的表达式 $e$（通过将 $x$ 替换为 $e_1$、$y$ 替换为 $e_2$、$z$ 替换为 $e_3$），这对模式按照目标模式将其变换为表达式 $e_1 \cdot (e_2 + e_3)$。请注意，在这个例子中，匹配源模式的表达式集合并不是一个正则树语言。我们提出了一种树变换模型，该模型由这样一组（源模式，目标模式）对的（可能是无限的）集合的有限表示给出。该模型的表达能力是以等价性检查的不可判定性为代价的。尽管如此，我们证明了对于我们所提出的基于模式的树变换模型，类型检查问题是可判定的。

    arXiv:2610.11337v1 Announce Type: cross  Abstract: We introduce and study pattern-based tree transformations. As an illustrating example, consider a source pattern $(x \cdot y) + (x \cdot z)$ and a target pattern $x \cdot (y + z)$ as a pair. This source pattern matches any expression $e$ of the form $(e_1 \cdot e_2) + (e_1 \cdot e_3)$ (by substituting $x$ with $e_1$, $y$ with $e_2$, and $z$ with $e_3$) and the pair transforms it into the expression $e_1 \cdot (e_2 + e_3)$ as dictated by the target pattern. Note that in this example, the set of expressions that match the source pattern is not a regular tree language.   We propose a model of tree transformations given by a finite representation of a (possibly infinite) set of such (source pattern, target pattern) pairs. The expressive power of this model comes at the cost of undecidability of checking equivalence. Nevertheless, we show that the type-checking problem is decidable for our model of pattern-based tree transformations. The ty
    
[^92]: ReCal：面向在线策略蒸馏恢复的结构化剪枝校准

    ReCal: Calibrating Structured Pruning for On-Policy Distillation Recovery

    [https://arxiv.org/abs/2610.11332](https://arxiv.org/abs/2610.11332)

    提出恢复感知校准方法ReCal，通过在剪枝前利用未剪枝教师模型与剪枝探针之间的前向KL散度，识别并保护易被剪枝破坏的教师支持的预测，从而显著提升结构化剪枝后在线策略蒸馏的恢复效果。

    

    结构化剪枝能够降低推理型语言模型的部署成本，但由此造成的能力退化可能阻碍后续在线策略蒸馏（OPD）的恢复效果。由于OPD依赖于学生模型自身生成的轨迹，离线蒸馏后仍然残留的剪枝损伤会限制其有效性。我们提出了RECAL（恢复感知校准），这是一种简单即插即用的方法，通过在剪枝前调整校准来改进OPD的恢复效果。RECAL利用未剪枝教师模型与剪枝探针模型之间的前向KL散度，识别出被剪枝破坏的教师模型所支持的预测，随后重新加权校准统计量，引导现有的剪枝准则保留这些预测。在多个模型和多种剪枝方法上，RECAL持续提升了OPD后的数学推理能力，在AIME基准上最高取得16.7个百分点的增益，同时在大多数代码生成对比中也有改进。进一步的分析表明……

    arXiv:2610.11332v1 Announce Type: new  Abstract: Structured pruning reduces the deployment cost of reasoning language models, but the resulting capability degradation can hinder subsequent on-policy distillation (OPD) recovery. Because OPD relies on student-generated trajectories, pruning damage that persists after offline distillation can limit its effectiveness. We propose RECAL, Recovery-Aware Calibration, a simple plug-and-play approach that improves OPD recovery by adjusting calibration before pruning. RECAL uses forward KL between an unpruned teacher and a pruned probe to identify teacher-supported predictions disrupted by pruning, then reweights calibration statistics to guide existing pruning criteria toward preserving these predictions. Across multiple models and pruning methods, RECAL consistently improves mathematical reasoning after OPD, achieving gains of up to 16.7 percentage points on AIME, alongside improvements in most code-generation comparisons. Further analysis show
    
[^93]: MetaEncoder：探索双编码器在基于自然语言接口的多模态“系统一”决策中的极限

    MetaEncoder: Exploring the Limit of Bi-Encoders for Multimodal System One Decision Making with Natural Language Interface

    [https://arxiv.org/abs/2610.11316](https://arxiv.org/abs/2610.11316)

    提出MetaEncoder，将预训练的30B解码器微调为基于单向对比学习的双编码器架构，实现了完全通过自然语言接口、可从小闭集扩展到数百万开放集候选空间的多模态系统一决策。

    

    系统（System One）模型输出的是受约束的决策和概率分布，而非自由形式的文本生成。当前主流范式依赖结构化的模式对象来编码状态、意图和候选选择，而我们重新审视一种完全基于自然语言的系统一接口。在该框架中，用户请求和每个候选选项均以自然语言表达，并由多模态（图像和视频）辅助输入提供支持。我们提出了MetaEncoder，它将预训练的Muse-Glimmer 30B解码器微调为一个遵循指令的决策编码器。为了在小规模闭集（<256）和大规模开放集（数百万）候选空间中都能有效扩展，MetaEncoder采用双编码器架构，并通过单向对比学习进行请求-候选对齐训练。我们在涵盖多模态决策、理解（……）等任务的11个基准测试套件和190个任务上进行了广泛评估。

    arXiv:2610.11316v1 Announce Type: cross  Abstract: System One models output constrained decisions and probability distributions rather than free-form text generation. While prevailing paradigms rely on structured schema objects to encode state, intent, and candidate choices, we revisit a fully natural language-based System One interface. In this framework, both the user request and each candidate option are expressed in natural language, supported by multimodal (image and video) auxiliary inputs. We introduce MetaEncoder, which fine-tunes a pre-trained Muse-Glimmer 30B decoder into an instruction-following decision-making encoder. To scale effectively across both small closed-set (< 256) and massive open-set (millions) candidate spaces, MetaEncoder employs a bi-encoder architecture trained via unidirectional contrastive learning for request-candidate alignment. We conduct extensive evaluations across 11 benchmark suites and 190 tasks spanning multimodal decision-making, understanding (
    
[^94]: 从检索到重构：为长期对话构建可演化的认知记忆

    From Retrieval to Reconstruction: Constructing Evolvable Cognitive Memory for Long-Term Dialogue

    [https://arxiv.org/abs/2610.11314](https://arxiv.org/abs/2610.11314)

    提出了CogMem认知记忆架构，基于PEC²F图模式将对话增量转换为具有来源感知的可演化图记录，区分有来源归属的主观主张与客观事实，从而支持长期对话中的可靠推理。

    

    arXiv:2610.11314v1 公告类型：新论文 摘要：作为长期对话代理的大语言模型（LLM）需要能够在长期交互中支持可靠推理的记忆系统。然而，现有的检索增强生成（RAG）框架通常将记忆视为被动存储，难以区分具有来源归属的信念与无归属的事件/事实记录，也难以连接分散在不同会话中的证据。我们提出了CogMem，一种基于PEC²F（人物-事件-概念-主张-事实，Person-Event-Concept-Claim-Fact）图模式的认知记忆架构。专门的Claim（主张）节点保留了主观陈述的来源与目标，而Fact（事实）节点和Event（事件）节点分别表示语义知识和情景知识。对话轮次被增量地转换为具有来源感知的图记录，整合为更高层级的事实，并在同一来源提供冲突更新时被调和为具有时间范围限定的Claim视图。在检索方面，一个由LLM驱动的基于规则的控制器……

    arXiv:2610.11314v1 Announce Type: new  Abstract: Large Language Models (LLMs) serving as long-term dialogue agents require memory systems that support reliable reasoning over extended interactions. However, existing Retrieval-Augmented Generation (RAG) frameworks typically treat memory as passive storage, making it difficult to distinguish source-attributed beliefs from unattributed event/fact records and to connect evidence dispersed across sessions. We introduce CogMem, a cognitive memory architecture based on the PEC$^2$F (Person-Event-Concept-Claim-Fact) graph schema. Dedicated Claim nodes preserve the source and target of subjective statements, while Fact and Event nodes represent semantic and episodic knowledge. Dialogue turns are incrementally converted into provenance-aware graph records, consolidated into higher-level facts, and reconciled into temporally scoped Claim views when the same source provides conflicting updates. For retrieval, a rule-based controller driven by LLM 
    
[^95]: BeliefScope：诊断大语言模型中证据驱动的修正与压力诱导的转变

    BeliefScope: Diagnosing Evidence-Driven Revision and Pressure-Induced Shifts in Large Language Models

    [https://arxiv.org/abs/2610.11305](https://arxiv.org/abs/2610.11305)

    提出BeliefScope受控黑盒框架，通过交叉实验设计将大语言模型的信念修正可靠地归因于真正的相关证据还是无实质信息的用户压力，实现两种影响来源的有效分离。

    

    语言模型可能在接收到真正相关的证据后修正同一命题，也可能在接收到不添加任何相关事实的方向性用户压力后修正同一命题。因此，仅凭可观察的响应变化无法揭示究竟是哪种来源驱动了这一改变。我们提出了BeliefScope，一个受控的黑盒框架，用于围绕固定目标命题分离这两种影响来源。BeliefScope将证据与压力进行交叉设计，并配备针对各因子的局部对照，通过概率报告、分类判断和行动建议，在符合相应渠道的量表上测量响应变化。为了确定这些可观察的对比何时能够支持可靠的归因，我们在受控的合成条件下评估了该观测设计。已知真值恢复实验和针对性消融实验确立了证据相关效应与压力相关效应能够被分离的范围，而半合成压力测试则刻画了这种可恢复性的变化规律。

    arXiv:2610.11305v1 Announce Type: new  Abstract: A language model may revise the same proposition after receiving genuinely relevant evidence or after receiving directional user pressure that adds no relevant fact. The observable response shift alone therefore does not reveal which source drove the change. We introduce BeliefScope, a controlled black-box framework for separating these two sources of influence around a fixed target proposition. BeliefScope crosses Evidence and Pressure with factor-specific local controls and measures response changes through probability reports, categorical judgments, and action recommendations on channel-appropriate scales. To determine when these observable contrasts support reliable attribution, we evaluate the observation design under controlled synthetic conditions. Known-truth recovery and targeted ablations establish where Evidence- and Pressure-related effects can be separated, while semi-synthetic stress tests map how that recoverability change
    
[^96]: 我们何时需要同策略蒸馏？在学生模型的离线生成结果上蒸馏往往更好

    When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better

    [https://arxiv.org/abs/2610.11291](https://arxiv.org/abs/2610.11291)

    研究发现，基于初始学生模型离线rollout的Semi-OPD蒸馏方法在17个教师-学生模型对中的14个上优于同策略蒸馏，两者之间的选择取决于师生模型输出token的重叠率。

    

    同策略蒸馏（OPD）在将教师模型能力迁移到学生模型方面日益流行。在这项工作中，我们提出了一个关键的研究问题：对于任意教师-学生模型对，同策略采样总是有益的吗？我们展示了一种简单的替代方法Semi-OPD，它从由初始学生模型生成的离线rollout中进行蒸馏，在准确率和训练效率上往往都能超越OPD。在从1.5B到235B参数规模的17个教师-学生模型对中，Semi-OPD在其中14个案例中胜过OPD，准确率最高提升13.6%，训练速度最高加快11.4倍。我们进一步发现，OPD与Semi-OPD之间的选择取决于初始教师模型与学生模型之间的对齐程度，这一程度可以用输出token重叠率来量化：只有当两者高度对齐且重叠率较高时，OPD才是有益的。我们的深入研究表明，有效的蒸馏需要同时相对于教师和学生双方保持同策略特性……

    arXiv:2610.11291v1 Announce Type: new  Abstract: On-policy distillation (OPD) has become increasingly popular for transferring teacher capabilities to student models. In this work, we ask a critical research question: Is on-policy sampling always beneficial for distilling arbitrary teacher-student pairs? We show that a simple alternative, Semi-OPD, which distills from offline rollouts generated by the initial student, can often outperform OPD in both accuracy and training efficiency. Across 17 teacher-student pairs ranging from 1.5B to 235B parameters, Semi-OPD outperforms OPD in 14 cases, with up to +13.6% accuracy and 11.4x training speedup. We further find that the choice between OPD and Semi-OPD depends on the alignment between the initial teacher and student, quantified by an output-token overlap ratio: OPD is beneficial only when the two are highly aligned with high overlap ratios. Our deeper investigation suggests that effective distillation requires on-policyness w.r.t. both th
    
[^97]: REMORY：学习残差记忆用于上下文压缩

    REMORY: Learning Residual Memory for Context Compaction

    [https://arxiv.org/abs/2610.11287](https://arxiv.org/abs/2610.11287)

    REMORY 提出一种神经记忆网络，通过生成软记忆token作为文本摘要的“残差”补充，使冻结的LLM仅用 5.2% 的输入位置即可接近全上下文性能，并显著减少重复工具输出和工具错误。

    

    长时程智能体通过压缩历史记录来在有限的上下文窗口内继续运行，但仅靠文本摘要可能无法支持后续的每一个决策。我们提出了 REMORY，这是一种神经记忆网络，用一段有界的软记忆token序列来补充摘要。给定历史记录和摘要，该网络学习生成有助于冻结的LLM近似其在完整历史输入下会产生的内容的token。这些token以摘要为条件并被附加在摘要之后，形成了沿序列维度上类似残差连接的机制。在 SummHay 基准上，REMARY 在洞察覆盖率几乎不变的情况下提升了来源归因能力，并且仅使用 5.2% 的输入位置就接近了全上下文的联合分数。在长时程智能体基准测试中，Qwen3.8-27B 和 GLM-5.3-Flash 使用残差记忆均显示出一致的提升。两个模型还表现出明显更少的重复工具输出和工具错误。

    arXiv:2610.11287v1 Announce Type: cross  Abstract: Long-horizon agents compact their history to continue within a finite context window, but a textual summary alone may not support every subsequent decision. We introduce REMORY, a neural memory network that supplements the summary with a bounded sequence of soft memory tokens. Given the history and summary, the network learns to generate tokens that help a frozen LLM approximate the continuation it would produce with the full history. The tokens are conditioned on the summary and appended after it, forming an analogue of a residual connection along the sequence dimension. On SummHay, REMORY improves source attribution at nearly unchanged insight coverage and approaches the full-context joint score using only 5.2% of the input positions. Across long-horizon agent benchmarks, Qwen3.8-27B and GLM-5.3-Flash show consistent gains with residual memory. Both models also exhibit substantially fewer repeated tool outputs and tool errors on Brow
    
[^98]: 多语言语音模型中的音系干扰

    Phonological Interference in Multilingual Speech Models

    [https://arxiv.org/abs/2610.11275](https://arxiv.org/abs/2610.11275)

    该研究揭示了多语言语音模型中的“音系干扰”这一系统性失败模式——模型错误地假设输入属于单一语言并强加其音系，导致在语码转换语音上丢失32%至79%的语言特有音素。

    

    音素级模型将语音转录或生成为音素序列——音素是区分单词的最小声音单位。这些模型能够实现细粒度的发音控制与理解，但在处理不匹配任何单一训练语言的输入时常常失败，例如在两种语言之间交替的语音（即语码转换），或训练数据中未包含的低资源语言。我们识别出这一现象背后的一个系统性失败模式——音系干扰：模型假设输入属于单一语言，并将该语言的音系强加于输入之上，从而覆盖了与所假设语言相冲突的局部音素级决策。我们通过模型保留“一种语言具有而另一种语言缺乏”的音素的频率来衡量干扰程度。在语码转换输入上，两个音素识别器（语音到音素模型）和一个音素条件控制的文本到语音模型会丢失32%至79%的此类音素，而两种语言共有的音素则丢失得少得多。

    arXiv:2610.11275v1 Announce Type: new  Abstract: Phoneme-level models transcribe or generate speech as a sequence of phonemes, the smallest sound units that distinguish words. These models enable fine-grained pronunciation control and understanding, yet often fail on input that does not match any single training language, such as speech alternating between two languages, known as code-switching, or low-resource languages absent from training. We identify a systematic failure mode behind this, phonological interference: models assume the input is in a single language and impose its phonology, overriding local phoneme-level decisions that conflict with the assumed language. We measure interference by how often a model retains phonemes that one language has but the other lacks. On code-switched input, two phone recognizers (speech-to-phoneme models) and a phoneme-conditioned text-to-speech model lose 32% to 79% of these phonemes, but lose far fewer of the phonemes both languages share. On
    
[^99]: 门控记忆：面向对话式AI的准入控制式记忆形成

    Gated Memory: Admission-Controlled Memory Formation for Conversational AI

    [https://arxiv.org/abs/2610.11270](https://arxiv.org/abs/2610.11270)

    该论文提出Gated Memory框架，通过在对话与存储之间设置准入控制检查点，在事实提取前基于完整话语上下文评估候选事实，解决了关键上下文信号在提取时不可逆丢失这一制约记忆质量的瓶颈问题。

    

    个性化对话式AI依赖于长期记忆系统，该系统从用户话语中提取事实并将其存储在持久化向量库中。尽管在检索、去重和生命周期管理方面已取得进展，但记忆形成阶段——即事实首次写入存储的时刻——几乎未受到任何系统性的关注。我们将此识别为生产系统中记忆质量的关键制约因素。关键上下文信号，例如用户的永久属性与临时情境之间的区分，仅存在于原始话语中，并且在提取过程生成“主语-关系-宾语”三元组的那一刻便不可逆地丢失了，任何下游流程都无法将其恢复。我们提出Gated Memory（门控记忆），一个轻量级、模块化的记忆形成框架，它在对话与存储之间引入两个决策检查点：其中一个是准入门，在提取之前根据完整话语上下文评估每个候选事实……

    arXiv:2610.11270v1 Announce Type: cross  Abstract: Personalized conversational AI relies on long-term memory systems that extract facts from user utterances and store them in persistent vector stores. Despite progress in retrieval, deduplication, and lifecycle management, the formation stage, the moment a fact is first written to storage has received almost no principled attention. We identify this as the binding constraint on memory quality in production systems. Critical contextual signals, such as the distinction between a permanent user attribute and a transient situation, exist only in the original utterance and are irreversibly lost the moment extraction produces a subject-relation-object triple. No downstream process can recover them. We propose Gated Memory, a lightweight, modular formation framework that interposes two decision checkpoints between conversation and storage: an admission gate that evaluates every candidate fact against the full utterance context before extractio
    
[^100]: 为什么在策略蒸馏有时会失败：学习信号的消失

    Why On-Policy Distillation Sometimes Fails: Vanishing Learning Signals

    [https://arxiv.org/abs/2610.11247](https://arxiv.org/abs/2610.11247)

    该研究发现大规模教师模型在在策略蒸馏中会导致基于梯度的学习信号过早消失，从而造成早期损失平台期，并从理论上为足够接近初始学生的教师证明了学习信号的局部恢复保证。

    

    在策略蒸馏（OPD）能够在语言模型之间实现有效的能力迁移，但其失败的内在机制尚未被完全理解。在代码生成和数学推理任务中，使用更大规模教师的OPD表现出早期的损失平台期：经过200次更新后，平均最终损失仅降低25.1%，而自我强化学习教师（通过对初始学生进一步进行强化学习训练得到）的损失降低率则达到96.2%。为了理解这一差异，我们在小学习率极限下将OPD分析为一个理想化的连续时间动力系统。我们的训练日志诊断将这些平台期与基于梯度的学习信号代理在仍有大量损失存在时的早期下降相关联；但这些测量并未解释底层梯度为何会减弱。我们进一步证明了在共享参数化下，对于与初始学生足够接近的教师，存在局部恢复保证……（原文摘要在此处截断）

    arXiv:2610.11247v1 Announce Type: cross  Abstract: On-policy distillation (OPD) enables effective capability transfer between language models, yet the mechanisms underlying its failures are not fully understood. Across code generation and mathematical reasoning, OPD with larger-scale teachers exhibits early loss plateaus, with an average final loss reduction of 25.1% after 200 updates, compared with 96.2% for self-RL teachers, obtained by further reinforcement learning (RL) training of the initial student. To understand this difference, we analyze OPD as an idealized continuous-time dynamical system in the small-learning-rate limit. Our training-log diagnostics associate these plateaus with an early decline in a gradient-based learning-signal proxy while substantial loss remains; these measurements do not establish why the underlying gradient weakens. We further prove a local recovery guarantee for teachers sufficiently close to the initial student in a shared parameterization under re
    
[^101]: 阅读关键内容：面向KV缓存的查询自适应量化

    Read What Matters: Query-Adaptive Quantization for KV Caches

    [https://arxiv.org/abs/2610.11245](https://arxiv.org/abs/2610.11245)

    提出ReadKV方法，通过渐进式编码实现KV缓存的查询自适应量化，根据每个解码查询动态分配键通道和值令牌的读取精度，并从理论上证明查询依赖的读取方案在相同读取预算下严格优于任何查询无关的方案。

    

    KV缓存条目在其未来查询尚未可知时就被存储，但每个解码查询在不同位置需要不同的精度。我们使用“保留比特数”与“每次查询获取比特数”这两个独立预算来研究这种失配问题。ReadKV将每个键和值存储在渐进式编码中，其前缀可支持不同的重建精度。对于每个查询，它先利用该查询分配键通道前缀，从重建的键计算注意力，然后再利用该注意力分配值令牌前缀，而存储的条目本身保持不变。每个阶段都在固定预算下优化一个经过校准的失真目标；我们证明了在细化增益递减条件下的最优精确分配，并将这些目标与注意力输出误差联系起来。我们还构造了一个有限维注意力族，证明在相同的读取预算下，查询依赖的访问严格优于所有查询无关的读取器，即使竞争方的编码器和解码器不受任何限制。

    arXiv:2610.11245v1 Announce Type: cross  Abstract: KV-cache entries are stored before their future queries are known, but each decoding query needs precision in different places. We study this mismatch using separate budgets for retained bits and bits fetched per query. ReadKV stores each key and value in a progressive code whose prefixes support different reconstruction precisions. For each query, it allocates key-channel prefixes using the query, computes attention from the reconstructed keys, and then allocates value-token prefixes using that attention. Stored entries remain unchanged. Each stage optimizes a calibrated distortion objective under a fixed budget; we prove exact allocation under diminishing refinement gains and relate these objectives to attention-output error. We also exhibit a finite-dimensional attention family where query-dependent access strictly outperforms every query-independent reader at the same read budget, even with unrestricted competing encoders and decod
    
[^102]: MiniVer-V：为短视频事实核查识别最小充分证据

    MiniVer-V: Identifying Minimal Sufficient Evidence for Short Video Verification

    [https://arxiv.org/abs/2610.11233](https://arxiv.org/abs/2610.11233)

    该论文提出以“证据充分性”为证据选择标准的短视频事实核查基准MiniVer-V，并设计两层核查框架，将基于内部证据的声明-视频一致性判断与需要外部佐证的事实性结论判定相分离，从而识别足以支撑核查结论的最小充分证据。

    

    短视频事实核查的一个核心挑战在于识别哪些证据足以支持核查结论。现有方法要么将所有可用证据都提供给核查器，从而引入噪声；要么仅按主题相关性选择证据，将“相关性”与“充分性”混为一谈。我们提出将“证据充分性”作为证据选择标准，即一个证据子集是否足以支持一个有把握的结论且无冗余。我们构建了MiniVer-V基准，包含195个带有三类结论标注（支持、反驳、证据不足）的短视频，以及5,510个多模态证据单元，涵盖视觉关键帧、语音转写文本和从网络检索的外部来源。我们提出了一个两层核查框架，将从内部证据评估的“声明-视频一致性”与需要外部佐证的“事实性结论判定”分离开来。在此之上，一个由充分性驱动的贪心……（原文摘要在此处截断）

    arXiv:2610.11233v1 Announce Type: cross  Abstract: A core challenge in short-video fact-checking is identifying which evidence is sufficient to support a verification conclusion. Existing approaches either give the verifier all available evidence, introducing noise, or select evidence by topical relevance, which conflates relatedness with sufficiency. We identify evidential sufficiency as the selection criterion: whether a subset of evidence is adequate to support a confident verdict without redundancy. We introduce MiniVer-V, a benchmark of 195 short videos with three-way verdict annotations (supported, refuted, insufficient) and 5,510 multimodal evidence units spanning visual keyframes, speech transcripts, and web-retrieved external sources. We propose a two-layer verification framework that separates claim-video consistency, assessed from internal evidence, from factual verdict determination, which additionally requires external corroboration. On top of it, a sufficiency-driven gree
    
[^103]: SafeInferCom：通过验证器引导的生成中间干预实现机器人任务规划的安全推理时计算

    SafeInferCom: Safe Inference-Time Compute via Verifier-Guided Mid-Generation Intervention for Robotic Task Planning

    [https://arxiv.org/abs/2610.11223](https://arxiv.org/abs/2610.11223)

    SafeInferCom是一个形式化验证器引导的推理时干预框架，通过在不干扰解码轨迹的情况下监控和验证中间计划，防止有效计划被后续推理覆盖并引导生成过程中的错误纠正，从而显著提升机器人任务规划的成功率与可靠性。

    

    大型推理语言模型（LRLMs）能够为机器人任务规划实现多步推理，但持续的推理可能会覆盖有效的中间计划或使约束违反问题未得到解决，从而降低规划的可靠性并浪费推理时计算资源。我们开发了一种推理时监控器，能够在不干扰原始解码轨迹的情况下暴露并验证中间计划。基于该监控器，我们提出了SafeInferCom，这是一个形式化验证器引导的框架，能够保留有效的中间计划并在生成过程中引导错误纠正。在多个LRLM和规划领域上进行的实验揭示了推理与响应之间的不一致性，以及单次推理下有限的自纠正能力。与单次推理相比，SafeInferCom提高了规划成功率并加速了错误纠正。当与迭代细化相结合时，它在进一步提高成功率的同时减少了token使用量。

    arXiv:2610.11223v1 Announce Type: cross  Abstract: Large Reasoning Language Models (LRLMs) enable multi-step reasoning for robotic task planning, but continued reasoning can overwrite valid intermediate plans or leave constraint violations unresolved, reducing planning reliability and wasting inference-time computation. We develop an inference-time monitor that exposes and verifies intermediate plans without disrupting the original decoding trajectory. Building on this monitor, we propose SafeInferCom, a formal verifier-guided framework that preserves valid intermediate plans and directs error correction during generation. Experiments across multiple LRLMs and planning domains reveal reasoning-response inconsistency and limited self-correction under one-shot inference. SafeInferCom improves planning success and accelerates error correction relative to one-shot inference. When combined with iterative refinement, it further improves success while reducing token usage compared with refine
    
[^104]: 转移定律之格

    The Lattice of Transition Laws

    [https://arxiv.org/abs/2610.11216](https://arxiv.org/abs/2610.11216)

    本文将扩散模型与自回归模型统一为同一个“腐蚀格”上的不同路径，通过定义解码调度的成本（即并行步骤所舍弃的依赖性），证明零成本调度的最少步数由数据的几何结构决定（例如等于图的树深度），从而可在解码前预测调度性能。

    

    扩散模型与自回归模型（AR）长期以来被视为生成模型中的不同类别：扩散模型专注于连续场，而自回归模型专注于离散token。近期的工作试图结合两种模型的优势，但每一种混合模型都在设计上预先固定了解码调度。在本文中，我们探讨这样一个问题：能否在解码之前、在固定步数下预测某个模型解码调度的性能。我们将扩散、自回归以及介于两者之间的模型描述为同一个腐蚀格上的路径，并将一个调度的成本定义为其并行步骤所舍弃的依赖关系。该成本表明，零成本调度所需的最少步数由数据的几何结构决定，且对token和连续场适用同样的规律。特别地，对于在图上满足马尔可夫性且沿其路径存在依赖关系的数据，最少步数等于该图的树深度，而树深度随序列长度呈对数增长，并且……（原文摘要在此处被截断）

    arXiv:2610.11216v1 Announce Type: cross  Abstract: Diffusion and autoregression (AR) have long been seen as different categories of generative models, with diffusion specialising in continuous fields and AR specialising in discrete tokens. Recent work seeks to combine the advantages of the two models, and each hybrid fixes its decoding schedule by design. In this paper, we ask whether the performance of decoding schedules of one model can be predicted before decoding at a fixed number of steps. We describe diffusion, AR, and models in between as paths on one corruption lattice, and define the cost of a schedule as the dependence its parallel steps discard. The cost shows that the fewest steps of a zero-cost schedule are set by the geometry of the data, in the same way for tokens and for continuous fields. In particular, for data that are Markov on a graph and dependent along its paths, the fewest steps equal the graph's treedepth, which is logarithmic in the length of a sequence and li
    
[^105]: 弥合KV缓存量化与线性注意力：从理论到预训练权重迁移

    Bridging KV-Cache Quantization and Linear Attention: From Theory to Pretrained Weight Migration

    [https://arxiv.org/abs/2610.11214](https://arxiv.org/abs/2610.11214)

    提出RAM-Net作为统一KV缓存量化与线性注意力的桥梁，通过离散地址空间上的软分配机制，在理论上证明其可分离读写重叠能局部逼近全注意力相似度，并支持从预训练权重迁移。

    

    arXiv:2610.11214v1 公告类型：交叉发布。摘要：KV缓存量化与线性注意力是应对Transformer存储与计算成本的两种代表性方法。KV缓存量化将单个KV条目压缩为离散编码，但仍需保留所有条目；而线性注意力则以循环方式将多个历史KV贡献聚合到一个固定大小的连续状态中，却可能引入干扰。这种对比引出了一个问题：能否在单一机制内将逐KV压缩与多KV聚合结合起来，以实现高效注意力。我们发现RAM-Net正是这样一座桥梁，它通过在离散地址空间上进行软分配来实现。这些软分配决定了对与每个地址相关联的连续槽状态的循环更新。在一个受限的RAM-Net构造下，我们证明软地址分配将硬量化匹配扩展为一种可分离的读写重叠机制，该机制能够局部逼近全注意力的相似度……

    arXiv:2610.11214v1 Announce Type: cross  Abstract: KV-cache quantization and linear attention are two representative approaches to tackling the storage and computational costs of Transformers. KV-cache quantization compresses individual KV entries into discrete codes but retains all entries, whereas linear attention recurrently aggregates multiple historical KV contributions into a fixed-size continuous state but can introduce interference. This contrast raises the question of whether per-KV compression and multi-KV aggregation can be bridged within a single mechanism for efficient attention. We identify RAM-Net as such a bridge through soft assignments over a discrete address space. These assignments determine recurrent updates to the continuous slot state associated with each address. Under a restricted RAM-Net construction, we prove that soft address assignments extend hard quantized matching to a separable read-write overlap that locally approximates full-attention similarity and s
    
[^106]: 选择性聆听：大型音频-语言模型中音频影响的机制引导控制

    Selective Listening: Mechanism-Guided Control of Audio Influence in Large Audio-Language Models

    [https://arxiv.org/abs/2610.11196](https://arxiv.org/abs/2610.11196)

    提出ICAP-Gate方法，通过机制引导、任务条件化的方式控制大型音频-语言模型的后期音频通路，在防止无关音频干扰文本推理的同时，不损害依赖音频的任务（如语音识别）的性能。

    

    大型音频-语言模型（LALMs）利用多模态证据，然而在无需聆听的情况下，与任务无关的音频可能会改变文本推理的决策。总体准确率可能会掩盖这种配对漂移，因为音频引起的修复与损害可能相互抵消。通过配对漂移分析和针对性干预，我们识别出特定于架构的、对干预敏感的后期音频通路，作为可操作的控制点。我们提出了ICAP-Gate，它对每个模型的通路应用机制引导的、任务条件化的控制。在四个LALM、两个推理基准以及环境声和自然语音干扰的实验中，ICAP-Gate在全部16个完整划分的模型-条件评估中，其影响率和答案翻转的点估计值均低于无门控推理。固定抑制会降低所有四个模型的自动语音识别（ASR）性能，而ICAP-Gate通过在明确的音频需求下保留音频通路，实现了与无门控推理相当的ASR性能。

    arXiv:2610.11196v1 Announce Type: cross  Abstract: Large audio-language models (LALMs) exploit multimodal evidence, yet task-irrelevant audio can alter text-reasoning decisions when listening is unnecessary. Aggregate Accuracy can hide this paired drift because audio-induced repairs and damages may cancel. Paired drift analysis and targeted interventions identify architecture-specific, intervention-sensitive late audio pathways as actionable control points. We introduce ICAP-Gate, which applies mechanism-guided, task-conditioned control to each model's pathway. Across four LALMs, two reasoning benchmarks, and environmental-sound and natural-speech interference, ICAP-Gate has lower point estimates for Influence Rate and Answer Flip than ungated inference in all 16 full-split model--condition evaluations. Fixed suppression degrades automatic speech recognition (ASR) across all four models, whereas ICAP-Gate matches ungated ASR performance by preserving the pathway for explicit audio-dema
    
[^107]: RAG-Stress：探测检索增强生成中证据依赖极限的研究

    RAG-Stress: Probing the Limits of Evidence Reliance in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.11183](https://arxiv.org/abs/2610.11183)

    该论文提出RAG-Stress受控诊断协议，通过编辑证据断言构造误导性检索内容，系统测量其诱导模型替换原本正确答案的“误导率”，揭示了十五个RAG系统在证据依赖上的脆弱性。

    

    遵循检索到的证据并不保证事实的正确性：误导性证据可能诱导模型替换掉它之前已经给出的正确答案。标准的准确率度量将答案替换与原本就存在的错误混在一起，从而掩盖了这种行为。我们提出了RAG-Stress，一个用于检验检索增强生成中证据依赖极限的受控诊断协议。该协议保持问题和参考答案不变，编辑证据中的一条断言使其支持指定的错误答案，并将两种来源优先级策略与答案片段在证据文本中的三种位置进行交叉组合。我们在每个模型在无检索情况下能正确回答的问题子集上测量误导率，同时在完整评估集上测量干净准确率。我们在TriviaQA-RC、HotpotQA和SearchQA上评估了十五个系统，涵盖API模型、开源模型以及通过强化学习训练的搜索智能体。

    arXiv:2610.11183v1 Announce Type: cross  Abstract: Following retrieved evidence does not guarantee factual correctness: misleading evidence can induce a model to replace an answer it previously gave correctly. Standard accuracy measures obscure this behavior by combining answer replacement with preexisting errors. We introduce RAG-Stress, a controlled diagnostic protocol for examining the limits of evidence reliance in retrieval-augmented generation. The protocol holds the question and reference answer fixed, edits one assertion to support a designated incorrect answer, and crosses two source priority policies with three positions of the answer span within the evidence text. We measure misleading rate (MR) on each model's subset of questions answered correctly without retrieval, alongside clean accuracy on the full evaluation set. We evaluate fifteen systems spanning API models, open models, and search agents trained with reinforcement learning on TriviaQA-RC, HotpotQA, and SearchQA, w
    
[^108]: LadderEdit：面向大语言模型内存高效终身编辑的编辑级残差压缩

    LadderEdit: Edit-Level Residual Compression for Memory-Efficient Lifelong Editing of LLMs

    [https://arxiv.org/abs/2610.11160](https://arxiv.org/abs/2610.11160)

    LadderEdit通过将每条编辑先以低秩草图存储、仅对未满足契约的困难编辑沿阶梯逐级提升秩的方式压缩LoRA适配器，在保持编辑覆盖效果的同时将内存占用降低5.2倍，并支持5万次连续终身编辑。

    

    大语言模型的终身编辑需要在获取后存储成千上万条编辑。一类被广泛使用的方法是为每条编辑附加一个LoRA适配器，这虽然能保持模型行为，但存储量会随编辑数量线性增长。为应对这一挑战，我们提出了LadderEdit，一种在每条编辑获取后对其LoRA适配器进行压缩的方法。每条编辑首先以低秩形式存储为一个廉价的“草图”。随后，我们在探针提示上检查该草图是否仍满足重写、泛化和局部性契约。通过检查的编辑保留草图；未通过的编辑则沿阶梯逐级提升至更高的秩，直到满足契约为止。由于每条编辑都保留了某种表示，编辑覆盖率得以维持，只有困难的编辑才会消耗更多的秩。在LLaMA-3-8B、Mistral-7B和Qwen2.5-7B模型上的ZsRE、CounterFact和WikiBigEdit基准测试中，LadderEdit在内存占用减少5.2倍的情况下保持了与精确LoRA存储相当的编辑效果，并在50,000次连续编辑后依然有效。

    arXiv:2610.11160v1 Announce Type: new  Abstract: Lifelong editing of LLMs requires storing thousands of edits after acquisition. A widely used family of approaches attaches one LoRA adapter per edit, which preserves behavior but grows linearly in storage. To address this challenge, we propose LadderEdit, a method that compresses each LoRA adapter after it is acquired. Each edit is first stored at low rank as a cheap sketch. We then check whether this sketch still satisfies the rewrite, generalization, and locality contract on probe prompts. Edits that pass keep the sketch; those that fail are promoted to a higher rank along a ladder until the contract is met. Because every edit retains some representation, coverage is maintained, and only hard edits consume more rank. Across ZsRE, CounterFact, and WikiBigEdit benchmarks on LLaMA-3-8B, Mistral-7B, and Qwen2.5-7B, LadderEdit tracks exact LoRA storage at 5.2x less memory and remains effective at 50,000 sequential edits.
    
[^109]: 面向文本兼容的语音到大语言模型桥接预训练的局部原型重建

    Local Prototype Reconstruction for Text-Compatible Speech-to-LLM Bridge Pretraining

    [https://arxiv.org/abs/2610.11159](https://arxiv.org/abs/2610.11159)

    该论文提出局部原型重建（LPR）这一轻量级训练正则化方法，使语音-LLM桥接嵌入保持接近冻结LLM输入嵌入的邻域，从而弥补现有预训练目标无法捕捉词元级词汇兼容性的不足，实现可迁移的文本兼容语音到LLM桥接预训练。

    

    语音到大语言模型系统通常通过一个小型的可训练桥接模块，将冻结的语音编码器连接到冻结的大语言模型（LLM）。这一桥接模块通常被视为简单的“管道”，但实际上它定义了语音到LLM接口的几何结构，而预训练目标则决定了该接口能否为下游任务提供可复用的初始化。我们通过两个互补的性质来研究可迁移的桥接：与文本侧的全局对齐，以及局部词汇流形兼容性，即桥接嵌入保持接近冻结LLM的输入嵌入邻域。我们通过一个固定的、无需预测头、无需时间戳的诊断方法使这一性质可被测量，该方法适用于任何目标函数，并证明下一词预测（NWP）和句子级对比预训练并不能完全捕捉词元级的词汇兼容性。随后，我们提出了局部原型重建（LPR），一种轻量级的、仅在训练阶段使用的正则化方法……

    arXiv:2610.11159v1 Announce Type: new  Abstract: Speech-to-LLM systems often connect a frozen speech encoder to a frozen large language model (LLM) through a small trainable bridge. The bridge is usually treated as plumbing, but it in fact defines the geometry of the speech-to-LLM interface, and the pretraining objective decides whether that interface provides a reusable initialization for downstream tasks. We study a transferable bridge through two complementary properties: global alignment with the text side, and local lexical manifold compatibility, where bridge embeddings remain close to the frozen LLM's input-embedding neighbourhoods. We make this property measurable with a fixed, head-free, timestamp-free diagnostic that applies to any objective, and show that next-word prediction (NWP) and sentence-level contrastive pretraining do not fully capture token-level lexical compatibility. We then introduce Local Prototype Reconstruction (LPR), a lightweight training-only regularizer t
    
[^110]: 大语言模型会从上下文中的奖励中学习吗？：重新思考奖励在上下文强化学习中的作用

    Do LLMs Learn from Rewards in Context? : Rethinking the role of reward in In-Context Reinforcement Learning

    [https://arxiv.org/abs/2610.11152](https://arxiv.org/abs/2610.11152)

    该研究通过受控实验发现，在大语言模型的直接上下文强化学习中，奖励信号虽然被读取却几乎不产生学习效果，而轨迹本身（即使语义被打乱或损坏）才是驱动上下文改进的关键，从而挑战了上下文学习真正实现强化学习的假设。

    

    大语言模型智能体越来越多地通过在上下文中积累经验而非更新参数来在推理阶段提升性能，这一过程通常被称为上下文强化学习（ICRL）。然而，上下文学习（ICL）是否真的能够发挥强化学习的作用，尚未经过检验。我们在其最简单的形式——直接ICRL中研究这一问题，即模型直接以原始的轨迹-奖励对为条件，并探究奖励是否真正充当学习信号。通过对六个模型在四个基准上的受控实验，我们发现奖励虽然会被模型读取，但其影响很小：翻转、随机化或移除奖励几乎不会改变性能改进曲线，即使在通过元提示明确指示模型进行探索、利用或对奖励进行推理的情况下也是如此。轨迹驱动了性能改进，但并非通过其语义内容：打乱或损坏的轨迹与真实轨迹的效果相当。

    arXiv:2610.11152v1 Announce Type: cross  Abstract: LLM agents increasingly improve at inference time by accumulating experience in context rather than by updating parameters. This process is often described as in-context reinforcement learning (ICRL). Whether in-context learning (ICL) can actually play the role of RL, however, has not been tested. We study this question in its simplest form, direct ICRL, where the model conditions directly on raw trajectory-reward pairs, and ask whether the reward acts as a learning signal. Through controlled experiments on four benchmarks across six models, we find that the reward is read, but its effect is small: flipping, randomizing, or removing the reward leaves the improvement curve almost unchanged, and this holds even under meta-prompts that explicitly instruct the model to explore, exploit, or reason over rewards. Trajectories drive improvement, but not through their semantic content: shuffled or corrupted trajectories work as well as real one
    
[^111]: ActiveMedAgent：面向多模态医学诊断的成本感知轨迹学习

    ActiveMedAgent: Cost-Aware Trajectory Learning for Multimodal Medical Diagnosis

    [https://arxiv.org/abs/2610.11140](https://arxiv.org/abs/2610.11140)

    该论文提出 ActiveMedAgent 框架，通过追踪诊断概率分布并按“诊断效用减去成本”对信息获取轨迹打分、离线训练轻量级 MLP 控制器，使冻结的视觉语言模型在多模态医学诊断中以更低成本、更少模态获得更准确的诊断结果。

    

    临床诊断本质上是顺序进行的：临床医生只有在预期额外证据能够消除诊断不确定性时，才会从廉价检查升级到昂贵检查。我们提出了 ActiveMedAgent，这是一个将这种成本感知的顺序决策逻辑引入多模态医学 AI 的框架。给定一个冻结的、通过 API 访问的视觉语言模型，ActiveMedAgent 追踪候选诊断的概率分布，并根据每一步的诊断效用减去成本来为每次信息获取打分。随后，一个轻量级的 MLP 控制器在这些评分轨迹上进行离线训练，从而学习何时请求额外证据、何时做出最终诊断。在三个常用基准测试中，基于轨迹的策略学习始终优于无引导的信息获取和全模态基线。值得注意的是，我们识别出了一种信息过载效应：在 175 个案例中，该智能体仅用更少的模态通道就能得出正确诊断，而全模态基线却诊断失败。

    arXiv:2610.11140v1 Announce Type: cross  Abstract: Clinical diagnosis is inherently sequential: clinicians escalate from cheap to costly tests only when additional evidence is expected to resolve diagnostic uncertainty. We present ActiveMedAgent, a framework that brings this cost-aware sequential logic to multimodal medical AI. Given a frozen, API-accessed vision-language model, ActiveMedAgent tracks probability distributions over candidate diagnoses and scores each acquisition by its per-step diagnostic utility minus cost. A lightweight MLP controller is then trained offline on these scored trajectories, learning when to request additional evidence and when to commit. Across three commonly used benchmarks, trajectory-based policy learning consistently outperforms both unguided acquisition and full-modality baselines. Notably, we identify an information overload effect. In 175 cases, the agent produces a correct diagnosis with fewer channels while the full-modality baseline fails, show
    
[^112]: “第十位陪审员”：面向官僚偏见检测的开放集立场筛选

    The "10th Juror": Open-Set Standpoint Screening for Bureaucratic Bias Detection

    [https://arxiv.org/abs/2610.11136](https://arxiv.org/abs/2610.11136)

    提出立场感知的多智能体框架 MARS-Gov，通过融合法律检索、开放集目标筛选、专门陪审员与改写验证，实现政府文档偏见的闭环治理，并能动态识别和应对新出现的偏见目标群体。

    

    预设偏见的边界本身就是一种偏见。我们研究针对荷兰政府文档的闭环偏见治理，系统需要检测有偏见的语言、将决策建立在法律与情境证据之上、在需要干预时改写有问题的句子，并验证改写在不扭曲原意的前提下减轻了危害。现有方法面临三个挑战：(i) 判别式分类器只能捕捉表面规律，缺乏规范性依据；(ii) 零样本大语言模型往往采取通用观点，并过度标记模糊的行政语言；(iii) 固定分类体系继承了封闭世界假设，无法识别新出现的本地目标群体。我们提出 MARS-Gov，一个立场感知的多智能体框架，结合了法律检索、开放集目标筛选、专门陪审员、保守路由与改写验证。当筛选发现未被覆盖的群体时，MARS-Gov 会实例化一个动态的“第十位陪审员”……

    arXiv:2610.11136v1 Announce Type: new  Abstract: Presupposing the boundaries of bias is itself a form of bias. We study closed-loop bias governance for Dutch government documents, where a system must detect biased language, ground decisions in legal and contextual evidence, rewrite problematic sentences when intervention is warranted, and verify that the rewrite mitigates harm without distorting meaning. Existing methods face three challenges: (i) discriminative classifiers capture surface regularities but lack normative grounding; (ii) zero-shot LLMs often adopt generic viewpoints and over-flag ambiguous administrative language; and (iii) fixed taxonomies inherit the Closed-World Assumption, missing emerging local targets. We propose MARS-Gov, a standpoint-aware multi-agent framework that combines legal retrieval, open-set target screening, specialized jurors, conservative routing, and rewrite verification. When screening finds an uncovered group, MARS-Gov instantiates a dynamic "10th
    
[^113]: 当记录的学习者很少或没有时，一个“系统一”大语言模型能否执行知识追踪？

    Can a System-One LLM Perform Knowledge Tracing When Few or No Learners Are Logged?

    [https://arxiv.org/abs/2610.11135](https://arxiv.org/abs/2610.11135)

    该论文提出，现成的“系统一”LLM（Jev/JevKT）无需目标平台的学习者数据或仅需极少数据即可完成知识追踪，其性能超过28个深度知识追踪模型和“系统二”LLM方法，而API成本仅约为后者的百分之一。

    

    知识追踪（KT）模型需要大量已记录的学习者数据，因此新课程或新平台在启动时没有可用的模型。在基于LLM的知识追踪中，LLM会生成答案，我们称之为“系统二”；它要么在目标数据上进行微调，要么对十个样本进行推理和投票，这不仅速度慢，而且给出的概率较为粗糙。我们探究一个现成的“系统一”LLM——它在单次处理中直接为输入的问题返回概率——在已记录的学习者很少或没有时能否执行知识追踪。在七个数据集上，Jev在不使用目标平台任何数据的情况下达到平均AUC为0.706，高于在8名学习者数据上训练的28个深度知识追踪模型中的最佳者（0.689），并在所有七个数据集上都优于“系统二”方法Thinking-KT（0.650），而其API成本仅约为后者的1/100。加入已记录学习者的示例和相似学习者统计量后（JevKT），该数值提升至0.722；JevKT在学习者数量达到16名时仍显著领先于深度知识追踪，在平均水平上直至64名学习者时依然领先，而监督（原文在此处截断）

    arXiv:2610.11135v1 Announce Type: new  Abstract: Knowledge tracing (KT) models need many logged learners, so a new course or platform starts without a usable model. In LLM-based KT the LLM generates the answer, which we call System-Two; it is either fine-tuned on the target data or reasons and votes over ten samples, which is slow and gives coarse probabilities. We ask whether an off-the-shelf System-One LLM, which returns a probability for a typed question directly in a single pass, can perform KT when few or no learners are logged. On seven datasets, Jev without any data from the target platform reaches a mean AUC of .706, above the best of 28 deep KT models trained on 8 learners (.689) and above System-Two Thinking-KT on all seven datasets (.650) at about 1/100 of its API cost. Adding examples and a similar-learner statistic from the logged learners (JevKT) raises this to .722; JevKT stays significantly ahead of deep KT up to 16 learners and ahead on average up to 64, and supervised
    
[^114]: SFT作为上下文缓解监督微调中的遗忘

    SFT-as-Context Mitigates Forgetting in Supervised Fine-Tuning

    [https://arxiv.org/abs/2610.11132](https://arxiv.org/abs/2610.11132)

    提出无需训练的SFT-as-context方法，让父模型将SFT模型的响应作为上下文通过上下文学习获得微调能力，从而在保留通用能力的同时缓解监督微调带来的遗忘问题。

    

    监督微调（SFT）为大型语言模型（LLM）赋予专业能力，但往往以遗忘其父模型（即微调前的预训练模型）的通用能力为代价。这种权衡对于既需要专业能力又需要通用能力的查询而言尤为受限。我们提出SFT-as-context，这是一种无需训练的方法，其中父模型将SFT模型的响应作为上下文来回答查询。这使得父模型能够通过上下文学习从SFT响应中获取微调后的能力，同时保留自身的通用能力。在19个父模型-SFT模型对和11个基准测试上，SFT-as-context在微调能力上与SFT模型保持接近，在AIME 2024和LiveCodeBench上的差距仅为2.2和2.1个百分点，在NutriBench-English上的宏观MAE仅为2.0，同时与父模型的差距保持在2.2个百分点以内。

    arXiv:2610.11132v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) equips large language models (LLMs) with specialized capabilities, but often comes at the cost of forgetting the general capabilities of their parent models (i.e., the pretrained models before fine-tuning). This trade-off is especially limiting for queries that require both specialized and general capabilities. We introduce SFT-as-context, a training-free method in which the parent model uses the SFT model's response as context to answer the query. This allows the parent model to acquire fine-tuned capabilities from the SFT response through in-context learning while preserving its own general capabilities. Across 19 parent-SFT model pairs and 11 benchmarks, SFT-as-context remains close to the SFT models on fine-tuned capabilities, with gaps of only 2.2 and 2.1 percentage points on AIME 2024 and LiveCodeBench and 2.0 macro MAE on NutriBench-English, while staying within 2.2 percentage points of the parent mo
    
[^115]: GameCommBench：面向AI生成游戏解说的统一基准与类型感知评估

    GameCommBench: A Unified Benchmark and Type-Aware Evaluation for AI-Generated Game Commentary

    [https://arxiv.org/abs/2610.11129](https://arxiv.org/abs/2610.11129)

    该论文提出了GameCommBench统一基准与TACE类型感知评估框架，首次系统性地对AI游戏解说进行分类型评估，发现实时观察和策略分析是当前AI解说员的主要能力瓶颈。

    

    游戏解说是一项开放式生成任务，需要多模态感知、策略推理和情境知识。现有的AI生成游戏解说（AI-GGC）研究在游戏类型、模态和评估协议方面仍然碎片化，而基于重叠度或整体性的评估方法无法捕捉解说的功能异质性。我们提出了GameCommBench，这是一个涵盖棋类游戏、体育和电子竞技的统一基准，其中的解说与异构的游戏情境对齐，并按解说类型进行了标注。我们进一步提出了类型感知解说评估（TACE），这是一个用于评估不同类型解说的结构化框架。随后，我们验证了TACE的可靠性及其与人类判断的一致性，并使用它对代表性的AI解说员进行基准测试。结果揭示了不均匀的能力分布，实时观察和策略分析成为主要瓶颈。GameCommBench与……（摘要内容不完整，原文到此中断）

    arXiv:2610.11129v1 Announce Type: new  Abstract: Game commentary is an open-ended generation task requiring multimodal perception, strategic reasoning, and contextual knowledge. Existing AI-Generated Game Commentary (AI-GGC) studies remain fragmented across games, modalities, and evaluation protocols, while overlap-based or holistic evaluators fail to capture the functional heterogeneity of commentary. We introduce \textsc{GameCommBench}, a unified benchmark spanning board games, sports, and esports, with commentary aligned to heterogeneous game contexts and annotated by commentary type. We further propose Type-Aware Commentary Evaluation (TACE), a structured framework for evaluating different types of commentary. We then validate TACE for reliability and human agreement, and use it to benchmark representative AI commentators. Results reveal non-uniform capability profiles, with live observation and strategic analysis emerging as major bottlenecks. Together, \textsc{GameCommBench} and 
    
[^116]: Lapras：面向时间序列语言模型的潜在推理

    Lapras: Latent Reasoning for Time Series Language Models

    [https://arxiv.org/abs/2610.11111](https://arxiv.org/abs/2610.11111)

    Lapras是一个后训练框架，通过让时间序列语言模型在潜在空间而非离散语言标记中进行推理，避免了将连续时序信号转化为文字描述时的信息丢失与早期错误传播，从而生成与输入信号一致、更忠实的答案。

    

    时间序列语言模型（TSLMs）通过对接时序信号进行推理并生成自然语言的答案与解释，为时间序列理解提供了一条有前景的路径。一种常见方法是思维链（Chain-of-Thought, CoT），它生成逐步的推理依据，将相关信号模式与最终答案联系起来。尽管这些模型在后训练阶段会从参考CoT轨迹中学习，但在推理时对输入时间序列生成忠实的描述仍然具有挑战性。用离散的语言标记来表达高维、连续的时间表示，可能导致模型忽略与任务相关的模式或对其描述不准确。由于后续的推理步骤建立在这些描述之上，早期的错误会不断传播，最终导致答案错误但解释看似合理，且与输入信号不一致。我们提出了Lapras（Latent Post-trained Reasoning Across Series），这是一个后训练框架，使T……（原文摘要在此处截断）

    arXiv:2610.11111v1 Announce Type: new  Abstract: Time Series Language Models (TSLMs) offer a promising path toward time series understanding by reasoning over temporal signals and producing natural language answers and explanations. A common approach is Chain-of-Thought (CoT), which generates step-by-step rationales linking relevant signal patterns to final answers. Although these models learn from reference CoT traces during post-training, generating faithful descriptions of input time series at inference remains challenging. Expressing high-dimensional, continuous temporal representations in discrete language tokens may cause the model to neglect task-relevant patterns or describe them inaccurately. Because later reasoning steps build on these descriptions, early errors propagate, leading to incorrect answers with plausible explanations that are inconsistent with the input signal. We propose Lapras (Latent Post-trained Reasoning Across Series), a post-training framework that equips T
    
[^117]: 临床医生对语言模型的使用方式与模型的评估方式存在偏差

    Clinician use of language models diverges from how the models are evaluated

    [https://arxiv.org/abs/2610.11069](https://arxiv.org/abs/2610.11069)

    通过分析医疗系统中6,342名临床医生发送的超过12.7万条查询，研究发现临床医生实际主要将语言模型用于文档行政事务和知识检索而非诊断，这与现有基于考试题目和诊断病例的基准测试评估方式存在显著偏差。

    

    大型语言模型（LLM）助手正在被部署到各医疗体系的临床医生手中，而对其是否具备部署条件的判断主要依赖于基准测试分数，其中大多数源自考试题目或精心筛选的病例。基准测试只有在其中条目与真实使用情形相似时，才能预测系统在实际部署中的表现，然而基准测试是否真实反映了这些系统所承担的工作却很少被测量。在本研究中，我们分析了35个专科的6,342名医生、高级执业提供者和护士在为期八个月的机构助手推广期间发送的127,833条查询。我们使用RCQ-Map对每条查询进行特征刻画，这是一个经临床医生验证、基于临床问题分类与LLM评估分类体系构建的框架，用于记录查询的任务类型、意图、可回答性、缺失信息及潜在危害。研究显示，文档与行政事务（36.2%）和知识检索（28.9%）合计占使用量的近三分之二，而诊断仅占3.7%。

    arXiv:2610.11069v1 Announce Type: new  Abstract: Large language model (LLM) assistants are being deployed to clinicians across health systems, and judgments about their readiness rest largely on benchmark scores, most of them derived from examination questions or curated cases. A benchmark predicts performance in deployment only to the extent that its items resemble real use, yet whether benchmarks reflect the work these systems receive has rarely been measured. Here we analyze 127,833 queries sent by 6,342 physicians, advanced practice providers and nurses in 35 specialties to an institutional assistant during an eight-month roll-out. We characterize each query with RCQ-Map, a clinician-validated framework grounded in taxonomies of clinical questions and of LLM evaluation, which records its task, intent, answerability, missing information and potential harm. Documentation and administration (36.2%) and knowledge retrieval (28.9%) made up nearly two-thirds of use, and diagnosis 3.7%; m
    
[^118]: 测量与缓解RLVR中的解模式坍缩

    Measuring and Mitigating Solution Mode Collapse in RLVR

    [https://arxiv.org/abs/2610.11064](https://arxiv.org/abs/2610.11064)

    本研究提出ModeBench多解任务基准，发现RLVR后训练在保持或提升准确率的同时，会导致模型的解多样性坍缩，概率集中于更少的正确解题模式上。

    

    语言模型通常可以用多种方式回答同一个问题，但基于可验证奖励的强化学习（RLVR）对于模型产生哪个正确答案并不加以区分。无论一个解是某个熟悉答案的第千次重复，还是模型从未产生过的新解，它都会获得相同的奖励。然而，在训练过程中让模型保留多个正确解具有潜在价值。例如，多种模式可以为用户提供选择，并提供有助于提升模型整体性能的问题解决策略。在此，我们介绍了ModeBench，这是一个多解任务基准，其中验证器会同时返回正确性和所发现的模式。我们随后利用ModeBench来衡量RLVR后训练过程中解多样性的变化。我们发现，即使准确率保持不变甚至有所提升，RLVR后训练仍会将概率集中到更少的正确模式上，而且前沿模型几乎……

    arXiv:2610.11064v1 Announce Type: cross  Abstract: A language model (LM) can usually answer the same question in more than one way, but reinforcement learning with verifiable rewards (RLVR) is indifferent to which correct answer a model produces. A solution will earn the same reward whether it is the thousandth copy of a familiar answer or one the model has never produced before. Yet, there is potential value in having the model retain multiple correct solutions as it is trained. For instance, multiple modes may give users a choice and provide problem-solving strategies that improve overall model performance. Here, we introduce ModeBench, a benchmark of multi-solution tasks in which the verifier returns both correctness and mode discovered. We then use ModeBench to measure how solution diversity changes under RLVR post-training. We find that RLVR post-training concentrates probability onto fewer correct modes even as accuracy holds or improves, and moreover, that frontier models are al
    
[^119]: FedAlphaEdit：面向协同知识编辑的零空间对齐合并

    FedAlphaEdit: Null-Space-Aligned Merging for Collaborative Knowledge Editing

    [https://arxiv.org/abs/2610.11033](https://arxiv.org/abs/2610.11033)

    提出首个在统一零空间原则下对齐本地编辑与服务器端合并规则的协同知识编辑框架 FedAlphaEdit，使多个机构无需共享原始编辑请求即可安全整合各自的知识编辑。

    

    多个机构可能各自持有私有的知识编辑请求，并希望在不共享原始编辑请求的情况下将其整合到同一个大语言模型中。AlphaEdit 等零空间约束的编辑方法在数学上保证每次更新都不会影响无关知识，而 CollabEdit 等协同框架则能在不共享数据的前提下聚合来自多个客户端的编辑。将两者结合看似轻而易举，然而我们证明这种朴素的组合在结构上是失败的，并找出了其原因。基于这一分析，我们提出了 FedAlphaEdit。据我们所知，这是首个在保护已有知识的单一零空间原则下，同时对本地编辑与服务器端合并规则进行对齐的协同知识编辑框架。FedAlphaEdit 构建于零空间对齐合并之上：客户端共享投影后的统计信息，服务器则可证明地恢复出编辑的结果

    arXiv:2610.11033v1 Announce Type: new  Abstract: Multiple institutions may each hold their own private knowledge edits and wish to integrate them into a single large language model without sharing raw edit requests. Null-space-constrained editing methods such as AlphaEdit mathematically guarantee that each update leaves unrelated knowledge intact, while collaborative frameworks such as CollabEdit aggregate edits from multiple clients without data sharing. Combining the two appears trivial. However, we show that this naive combination fails structurally, and we identify its cause. Guided by this analysis, we propose FedAlphaEdit. To our knowledge, this is the first collaborative knowledge editing framework that aligns both local editing and the server-side merging rule under a single null-space principle for preserving existing knowledge. FedAlphaEdit builds on null-space-aligned merging, in which clients share projected statistics and the server provably recovers the result of editing 
    
[^120]: 提示词与规则之争：语音用户模拟器中语音自然性行为的审计与控制

    Prompts versus Rules: Auditing and Controlling Speech Naturalness Behaviors in Voice User Simulators

    [https://arxiv.org/abs/2610.11015](https://arxiv.org/abs/2610.11015)

    研究发现，基于提示词的方法无法可靠地让语音用户模拟器产生不流畅、打断和反馈等自然语音行为，而基于规则的注入算法能够更一致、更自然地控制这些行为。

    

    随着语音代理在商业上日益流行，用于评估已部署代理的用户模拟器也在不断发展，以包含更真实、多变和多样化的语音自然性行为——不流畅（disfluency）、打断（interruption）和反馈性附和（backchanneling）。用户模拟器的质量直接影响代理评估结果的有效性。然而，我们发现迄今为止的大多数研究都未曾详细检验这些行为的预期配置是否真的在模拟中得以实现。在本研究中，我们审计了tau-Voice（我们自己的基于LLM的提示方法，涵盖三个模型）以及我们的基于规则的注入算法（针对不流畅、打断和反馈）所实际实现的自然性行为。我们发现，通过提示词来引发这些行为是不可靠的，其产生的语音与指令不一致，其位置和分布也不如指令所暗示的那样自然。相比之下，我们基于规则的、与模型无关的算法……

    arXiv:2610.11015v1 Announce Type: new  Abstract: As voice agents gain more popularity commercially, the user simulators used to evaluate the deployed agents are also being developed to include more realistic, variable, and diverse speech naturalness behaviors -- disfluency, interruption and backchanneling. The quality of the user simulator directly affects the validity of agent evaluation results. However, we find that most studies so far have not examined in detail whether the intended configuration for these behaviors is realized in the simulation. In this study, we audit the realized naturalness behaviors of tau-Voice, our own LLM-based prompting approach across three models, and our rule-based injection algorithm for disfluency, interruption, and backchanneling. We find that prompting for these behaviors is unreliable and produces speech inconsistent with the instructions, placed and distributed less naturally than the instruction implies. In contrast, our rule-based, model-free al
    
[^121]: 风格回归：一种用于创作与测量用户模拟中人格忠实度的社会语言学方法

    Back in Style: A Sociolinguistic Approach to Authoring and Measuring Persona Fidelity in User Simulation

    [https://arxiv.org/abs/2610.10988](https://arxiv.org/abs/2610.10988)

    该研究提出一种社会语言学方法，将用户人格创作具体化为可观察的语言风格比率，从而利用无需模型的文体计量学与词典内容分析工具确定性地测量用户模拟器的人格忠实度，并在客服智能体实验中验证其优于描述性基线。

    

    随着智能体系统在商业上日益流行，用户模拟器越来越多地被用作评估这些系统的测量工具。然而，模拟用户与真实人类用户相比的忠实度普遍较低，且通常需要通过昂贵且主观的大语言模型评判者来评估。在这项试点研究中，我们探讨是否可以通过社会语言学的方式处理用户人格，从而以确定性的方法测量忠实度：即将人格视为一种从可观察的语言风格中涌现的社会类型，而非由模型必须外推为行为的标签或描述所预测的类型。我们将人格创作具体化为明确的风格比率，这使我们能够将两种成熟且无需模型的工具——作者验证文体计量学和基于词典的内容分析——转化为忠实度诊断方法。我们在五个面向任务的客服智能体上，对社会语言学方案与扁平的描述性基线进行了A/B测试。结果表明，社会语言学方案……（原文摘要在此处截断）

    arXiv:2610.10988v1 Announce Type: new  Abstract: As agentic systems gain commercial popularity, user simulators increasingly serve as measurement instrument for their evaluation. However, the fidelity of simulated users in comparison to real human users is generally low, and typically assessed by costly, subjective LLM judges. In this pilot study, we ask whether fidelity can instead be measured deterministically by treating a user persona sociolinguistically: as a social type that emerges from observable linguistic style, rather than one predicted by labels or descriptions a model must extrapolate into behaviour. We author personas as concrete stylistic rates, which lets us transfer two established, model-free instruments -- authorship-verification stylometry and lexicon-based content analysis -- as fidelity diagnostics. We A/B-test the sociolinguistic schema against a flat descriptive baseline across five task-oriented customer-service agents. Results show that the sociolinguistic sch
    
[^122]: 当引用产生误导时？一个用于法律幻觉检测的声明级基准

    When Citations Mislead? A Claim-Level Benchmark for Legal Hallucination Detection

    [https://arxiv.org/abs/2610.10971](https://arxiv.org/abs/2610.10971)

    该论文提出了PARCEL基准——基于纽约州上诉法院判决构建的包含3,396条标注声明的法律幻觉检测数据集，将其转化为三分类自然语言推理任务来评估大语言模型，发现即使最强模型准确率达0.97，仍会将缺乏引用支持的声明误判为已支持，其中虚构但看似合理的引用造成的性能下降最为严重。

    

    大语言模型在法律研究和文书起草中的应用日益广泛，但它们仍可能产生听起来令人信服却没有得到所引来源支持的声明。我们推出了PARCEL，一个用于检验法律声明是否获得相关法律依据支持的基准。基于纽约州上诉法院近期的判决，我们构建了一个包含3,396条括注式声明的数据集，每条声明被标注为“支持”、“反驳”或“未找到”。我们将该任务转化为一个三分类自然语言推理问题，并在零样本设置下评估了多个最先进的大语言模型。尽管最强模型的准确率高达0.97，但结果也揭示了一个重要弱点：即使提供了完整的判决书全文，模型仍会将不获支持的声明错误地判定为已支持。在所有模型中，缺失支持比直接矛盾更难检测，而虚构但看似合理的引用会导致最大的性能下降。

    arXiv:2610.10971v1 Announce Type: cross  Abstract: Large language models are increasingly used in legal research and drafting, but they can still produce claims that sound convincing without being supported by the cited source. We introduce PARCEL, a benchmark for checking whether a legal claim is supported by the underlying authority. Using recent New York State Court of Appeals decisions, we build a dataset of 3,396 parenthetical-style claims labeled as Supported, Refuted, or Not Found. We cast this task as a three-way natural language inference problem and evaluate several state-of-the-art LLMs in a zero-shot setting. Although the strongest models reach up to 0.97 accuracy, the results also show an important weakness: models still incorrectly mark unsupported claims as supported, even when the full opinion text is provided. Across models, missing support is harder to detect than direct contradiction, and fabricated but plausible citations cause the largest drop in performance. Overa
    
[^123]: AI4Fire：评估大型语言模型在野火任务上的表现

    AI4Fire: Evaluating Large Language Models on Wildfire Tasks

    [https://arxiv.org/abs/2610.10946](https://arxiv.org/abs/2610.10946)

    AI4Fire 基准首次在五个野火任务上对多个大语言模型进行成对的“裸跑”与“接地”零样本评估，发现接地信息（如只读SQL工具）在直接包含答案时能大幅提升准确率（从至多16%提升到至少88%），但简单规则基线仍然难以被超越。

    

    大型语言模型（LLM）正在进入野火管理领域，而夸大的评估结果可能造成财产损失和人员伤亡。它们在野火任务中的表现如何——无论有无“接地”？“裸跑”指模型仅接收任务输入本身；“接地”指模型额外接收一项任务特定的补充信息：例如在烟雾检测任务中，来自同一摄像头的无烟雾参考帧。AI4Fire 让六个核心模型在五个野火任务上以零样本方式分别进行裸跑和接地测试；另一轮扩展测试又增加了29个模型。我们针对火灾任务进行的文献检索共找到138篇论文，其中没有一篇能同时具备这样的模型阵容、任务覆盖范围以及成对的裸跑与接地实验。我们报告三项发现：（1）接地在补充信息直接包含答案时帮助最大：一个只读SQL工具将所有核心模型的数据库准确率从至多16%提升到至少88%。（2）简单规则难以被超越：没有核心模型的表现能超过“重复今天的人员配置数量”这一基线，两个开源权重模型大多只是复制中位数……（原文此处截断）

    arXiv:2610.10946v1 Announce Type: new  Abstract: Large language models (LLMs) are entering wildfire management, where overstated evaluations can cost property and lives. How do they perform on wildfire tasks, with and without grounding? Bare means a model receives the task input alone. Grounded means it also receives one task-specific addition: for smoke detection, a smoke-free reference frame from the same camera. AI4Fire runs six core models bare and grounded on five wildfire tasks, zero-shot; a sweep adds 29 more. Our literature search on fire tasks found 138 works; none combines this roster, task coverage, and paired bare and grounded runs. We report three findings. (1) Grounding helped most where the addition carried the answer: a read-only SQL tool lifted every core model's database accuracy from at most 16 to at least 88 percent. (2) Simple rules were hard to beat: no core model outperformed repeating today's staffing count, and two open-weight models mostly copied the median of
    
[^124]: StoreBench：用于评估和训练自主运营智能体的实时电商环境

    StoreBench: A Live-Commerce Environment for Evaluating and Training Autonomous Operator Agents

    [https://arxiv.org/abs/2610.10942](https://arxiv.org/abs/2610.10942)

    StoreBench是一个让智能体在生产级电商后端上经营线上服装店的实时环境，通过全天候动态市场、校准的通过阈值和抗操纵的奖励机制，来评估和训练自主运营智能体的长程规划与经济决策能力。

    

    强化学习环境如今已成为后训练阶段提升大语言模型（LLM）能力的主要杠杆，然而大多数智能体基准测试仍然是静态的：世界只在智能体行动时才发生变化，奖励只是一个终结性判决，通过门槛也是随意设定的。我们提出StoreBench，这是一个实时电商环境，智能体在生产级电商后端上经营一家中等规模的线上服装店，以测试其在不确定性下的长程规划与经济判断能力。顾客全天候下单，供应商会重新定价并可能倒闭，市场冲击会在部分预警或毫无预警的情况下到来。智能体通过人类运营者会使用的相同的29个商家工具执行操作，并在窗口化运营预算下运行，该预算使模拟时间成为所执行操作的函数，因此模型延迟不会影响模拟时间。通过阈值根据脚本化的锚定策略进行校准，奖励则针对一系列常见的……（摘要在此处截断）

    arXiv:2610.10942v1 Announce Type: new  Abstract: Reinforcement learning environments are now a primary lever for improving large language model (LLM) capabilities in post-training, yet most agentic benchmarks remain static: the world moves only when the agent acts, the reward is a terminal verdict, and the pass bar is set arbitrarily. We introduce StoreBench, a live-commerce environment in which an agent runs a mid-size online apparel store on a production-grade commerce backend, testing long-horizon planning and economic judgment under uncertainty. Customers order around the clock, suppliers reprice and fail, and market shocks arrive with partial or no warning. The agent acts through the same 29 merchant tools a human operator would use, under a windowed operation budget that makes simulated time a function of actions taken, so model latency cannot influence simulated time. Pass thresholds are calibrated against scripted anchor policies, the reward is hardened against a catalogue of r
    
[^125]: 电商搜索中用于页面级布局决策的语言模型

    Language Models for Page-Level Layout Decisions in E-commerce Search

    [https://arxiv.org/abs/2610.10920](https://arxiv.org/abs/2610.10920)

    该论文研究了利用语言模型离线评估电商搜索页面级布局决策（如在特定位置插入二级堆栈）是否对用户有益，从而减少对昂贵在线A/B测试的依赖。

    

    电商搜索页面是数百万在线购物者的关键触点。虽然传统搜索引擎返回的是按排名排序的结果列表，但现代电商搜索页面越来越多地整合了推荐系统模块——例如，在特定位置展示替代产品分组的二级堆栈。当引入得当时，二级堆栈可以提升用户参与度；然而，放置位置欠佳可能会干扰浏览流程并降低主要结果的质量。与传统搜索排名不同（后者已有交织比较等成熟的评估技术），在缺乏昂贵的在线A/B测试的情况下，评估页面级布局变化（例如，何时何地插入二级堆栈）仍然具有挑战性。为了解决这一问题，我们研究了离线方法，用于评估给定的布局决策——具体而言，在特定位置包含二级堆栈——是否对用户有益。我们研究……

    arXiv:2610.10920v1 Announce Type: cross  Abstract: E-commerce search pages are critical touchpoints for millions of online shoppers. While traditional search engines return a ranked list of results, modern E-commerce search pages increasingly incorporate recommender system modules -- for example, secondary stacks that surface alternative product groupings at specific positions. When introduced appropriately, secondary stacks can improve user engagement; however, suboptimal placement may disrupt browsing flow and degrade the primary results. Unlike traditional search ranking, where evaluation techniques such as interleaving are well established, evaluating page-level layout changes e.g., when and where to insert a secondary stack remains challenging without costly online A/B testing. To address this, we study offline methods for evaluating whether a given layout decision -- specifically, the inclusion of a secondary stack at a particular position -- is beneficial to users. We investigat
    
[^126]: 用于机器翻译质量标注的大语言模型：人类与模型都面临挑战

    Large Language Models for Machine Translation Quality Annotation: Humans and Models Are Both Challenged

    [https://arxiv.org/abs/2610.10918](https://arxiv.org/abs/2610.10918)

    本文系统评估了大语言模型在 MQM 和 ESA 两种机器翻译质量标注任务中与人类标注者的一致性，涵盖 70 个语言对及 WMT23/WMT25 数据，发现 LLM 与人类的一致性在某些情况下甚至超过人类标注者之间的一致性，但无论人类还是模型在此任务上都面临挑战。

    

    大语言模型（LLM）被认为是替代人工判断进行机器翻译（MT）评估的一种更高效、更经济的选择。由于机器翻译评估涵盖大量语言对、领域和标注粒度层级，LLM 在被可靠地用作人工评估的替代方案之前，必须在这些维度上进行全面评估。在本文中，我们通过比较 LLM 与人类标注者的一致性，评估了 LLM 在两种主流机器翻译质量评估方案上的表现：多维质量指标（MQM）和错误片段标注（ESA）。我们在一个包含 70 个语言对的长上下文测试集以及公开可用的 WMT23 和 WMT25 数据上展示了实验结果，考察了跨多种语言对和领域的分数标注与错误片段标注的一致性。我们的结果表明，尽管 LLM 与人类标注者的一致性在某些方面超过了人类标注者之间的一致性（原文摘要不完整，句子在此处中断）。

    arXiv:2610.10918v1 Announce Type: new  Abstract: Large Language Models (LLMs) are considered to be a more efficient and cost-effective alternative to human judgment for Machine Translation (MT) evaluation. With MT evaluation spanning a large number of language pairs, domains and levels of annotation granularity, LLMs must be thoroughly evaluated across these dimensions before being reliably used as alternatives to human evaluation. In this paper, we evaluate the performance of LLMs for two prominent MT quality evaluation schemes: Multidimensional Quality Metrics (MQM) and Error Span Annotation (ESA) by comparing their agreement with human annotators. We present results on a long-context test set of 70 language pairs and the publicly available WMT23 and WMT25 data, investigating both score and error span annotation agreement across a variety of language pairs and domains. Our results show that while LLM agreement with human annotators exceeds agreement between human annotators for some 
    
[^127]: 面向智能体在线策略蒸馏的随机教师干预

    Stochastic Teacher Intervention for Agentic On-Policy Distillation

    [https://arxiv.org/abs/2610.10878](https://arxiv.org/abs/2610.10878)

    提出STI-OPD框架，在多轮智能体在线策略蒸馏中，通过由师生策略差异引导的随机教师干预将学生动作替换为教师生成的动作，从而缓解多轮交互中的误差累积并最大化可靠监督的获取。

    

    在线策略蒸馏（OPD）通过在学生模型生成的轨迹上进行密集的token级监督，能够高效地将更强教师模型的能力迁移到学生语言模型，并在数学推理等复杂任务上展现出前景。然而，在多轮智能体任务中，学生的决策会塑造后续的观测结果，导致早期错误在多轮交互中不断累积。由此产生的轨迹可能偏离教师模型的轨迹分布，使得教师模型的token级监督变得不可靠，甚至对OPD训练产生反效果。为解决这一问题，我们提出了STI-OPD，一个面向多轮智能体在线策略蒸馏的随机教师干预框架。在多轮交互过程中，STI-OPD利用由师生策略差异引导的教师干预，将学生提出的动作替换为教师生成的动作，以最大化可靠监督的获取。我们进一步开发了一个（摘要在此处被截断）...

    arXiv:2610.10878v1 Announce Type: cross  Abstract: On-policy distillation (OPD) efficiently transfers capabilities from a stronger teacher to a student language model through dense token-level supervision on student-generated rollouts and has shown promise on complex tasks such as mathematical reasoning. However, in multi-turn agentic tasks, student decisions shape subsequent observations, causing early errors to accumulate across turns. The resulting trajectories can drift away from the teacher's rollout distribution, making the teacher's token-level supervision less reliable or even counterproductive for OPD training. To address this issue, we introduce STI-OPD, a stochastic teacher intervention framework for multi-turn agentic OPD. During multi-turn interaction, STI-OPD uses teacher intervention guided by teacher-student policy discrepancy to replace the student's proposed action with a teacher-generated one to maximize the acquisition of reliable supervision. We further develop a s
    
[^128]: 稀疏注意力是矩阵近似问题，而非从一堆数值中做挑选

    Sparse Attention Is Matrix Approximation, Not Choosing from a Bag of Values

    [https://arxiv.org/abs/2610.10871](https://arxiv.org/abs/2610.10871)

    该论文指出稀疏注意力在概念上应被表述为矩阵近似问题而非挑选最大注意力数值，并据此提出了矩阵近似稀疏注意力方法MASA。

    

    大语言模型（LLMs）在众多领域取得了出色的性能，但其效率受到注意力机制相对于提示长度呈二次方增长的计算成本的限制。稀疏注意力通过仅保留一小部分查询-键交互来近似完整的注意力矩阵，从而降低这一成本。然而，现有方法陷入了一个数学上错误的观点：它们只是简单地保留注意力矩阵中标量数值较大的元素或高权重区域。这种做法将注意力矩阵视为一堆数值的集合，忽略了它实际上是被用作一个结构化矩阵，其各元素通过与值向量相乘共同决定注意力的输出。我们认为这正是核心的概念性问题：稀疏注意力应当被表述为矩阵近似问题，而不是盲目地从一堆元素中挑选数值最大的那些。基于这一观点，我们提出了矩阵近似稀疏注意力（MASA）。MASA用……替换了原始的注意力权重（摘要在此处截断）

    arXiv:2610.10871v1 Announce Type: new  Abstract: Large Language Models (LLMs) achieve strong performance across many domains, but their efficiency is limited by the quadratic cost of attention with respect to prompt length. Sparse attention reduces this cost by retaining only a small fraction of query-key interactions to approximate the full attention matrix. However, existing methods are trapped in a mathematically wrong view: they simply keep large scalar entries or high-mass regions of the attention matrix. This treats the attention matrix as a bag of values, ignoring that it is used as a structured matrix whose entries jointly determine the attention output through multiplication with value vectors. We argue that this is the core conceptual issue: sparse attention should be formulated as matrix approximation, not as blindly choosing the largest values from a bag of entries. Based on this view, we propose Matrix Approximation Sparse Attention (MASA). MASA replaces raw attention-mass
    
[^129]: 基于人类听众强化学习的对话语音美学模型

    Conversational Voice Aesthetic Model with Reinforcement Learning from Human Listeners

    [https://arxiv.org/abs/2610.10868](https://arxiv.org/abs/2610.10868)

    该论文提出CVAM语音大语言模型，通过合成美学描述的监督微调和基于约3万条人类听众标注的组相对策略优化（GRPO），实现对语音性别、音高、语速、情感、表达方式等九项美学属性的预测，其与人类听众的一致性超越了Gemini 3.1 Pro和开源语音LLM。

    

    我们提出了对话语音美学模型（CVAM），这是一个用于在自然对话情境中描述真实或合成语音回复的语音美学的语音大语言模型。给定一个上下文和一段回复语音，CVAM能够描述刻画语音特征的显著时刻，并预测九个类别属性，涵盖性别、音高、语速、情感和表达方式。关键挑战在于情感和表达方式等感知领域，它们本质上具有主观性，缺乏确定性的真实标准（ground truth）。为此，我们针对来自CANDOR语料库的3千条真实和合成回复，每条收集了约10份人工标注。CVAM首先在合成的美学描述和标签上进行监督微调，随后基于人类判断使用组相对策略优化进行优化。实验表明，CVAM与人类听众的一致性优于Gemini 3.1 Pro和开源语音大语言模型，并且超过了单个标注者与其余标注者之间的一致性水平。

    arXiv:2610.10868v1 Announce Type: cross  Abstract: We introduce Conversational Voice Aesthetic Model, a speech large language model for describing the voice aesthetics of real or synthetic speech responses in natural conversational contexts. Given a context and a response speech, CVAM describes salient moments that characterize the voice and predicts nine categorical attributes spanning gender, pitch, pacing, emotion, and delivery. The key challenge lies in perceptual fields such as emotion and delivery, which are inherently subjective and lack definitive ground truth. Therefore, we collect ~10 human annotations for each of 3k real and synthetic responses derived from the CANDOR corpus. CVAM is supervised finetuned on synthesized aesthetic descriptions and labels, then optimized with Group Relative Policy Optimization on human judgments. Experiments show that CVAM better agrees with human listeners than Gemini 3.1 Pro and open-source speech LLMs, and outperforms single-human-vs.-rest a
    
[^130]: 基于路由稀疏自编码器的语言学与副语言学信息解耦

    Disentangling Linguistic and Paralinguistic Information with Routed Sparse Autoencoders

    [https://arxiv.org/abs/2610.10865](https://arxiv.org/abs/2610.10865)

    提出将TopK稀疏自编码器与路由特定监督和跨因子对抗相结合的方法，成功将自监督语音编码器中纠缠的语言学与副语言学信息（如说话人身份、情感、韵律）解耦到不同路由中，且这种分离在不同编码器、语料库和探测方法上均保持一致并可迁移。

    

    自监督语音编码器在共享且纠缠的表示空间中同时包含语言学和副语言学信息。我们将TopK稀疏自编码器与路由特定监督和跨因子对抗相结合。在冻结的SPEAR和WavLM编码器上，独立的探测实验显示出因子特定的保留与抑制效果：语言学信息在语言学路由中保持更强，而副语言学因子（包括说话人身份、情感和韵律）在副语言学路由中得到保留，并在语言学路由中被大幅削弱。在LibriSpeech上学到的路由组织无需表示侧再训练即可在MSP-Podcast上保持有效。特征空间中的路由干预能够进一步转移被交换的因子，同时基本保留未改变路由所携带的信息。这些结果表明，在不同编码器、语料库、独立探测任务和表示层面均实现了一致的路由选择性分离。

    arXiv:2610.10865v1 Announce Type: new  Abstract: Self-supervised speech encoders contain linguistic and paralinguistic information in a shared, entangled representation space. We combine a TopK sparse autoencoder with route-specific supervision and cross-factor adversaries. Across frozen SPEAR and WavLM encoders, independent probes show factor-specific retention and suppression: linguistic information remains stronger in the linguistic route, while paralinguistic factors, including speaker identity, emotion, and prosody, are retained in the paralinguistic route and substantially reduced in the linguistic route. The route organisation learned on LibriSpeech persists on MSP-Podcast without representation-side retraining. Feature-space route interventions further transfer the swapped factor while largely preserving the information carried by the unchanged route. These results show consistent route-selective separation across encoders, corpora, independent probes, and representation-level 
    
[^131]: AI的真实长期记忆：比重计算更快且更省钱的5000万令牌窗口

    Real Long-Term Memory for AI: A 50-Million-Token Window That Is Faster and Cheaper Than Recompute

    [https://arxiv.org/abs/2610.10845](https://arxiv.org/abs/2610.10845)

    本文提出并验证了一个名为galahad-kv的记忆层，通过将KV状态加密保存到本地NVMe磁盘并按需逐字节精确加载，实现了5000万令牌的超长上下文记忆，速度比重计算快2.8至4.3倍、GPU能耗降低8.8至12.3倍，且GPU内存占用在整个处理过程中保持恒定。

    

    大型语言模型只能使用能放入其上下文窗口的文本，并且每次发送提示词时都要重新计算其内部键值（KV）状态。我们测试了一个记忆层——公开软件包 galahad-kv，它将每个约16,000个令牌的文本块的KV状态保存到加密的本地NVMe磁盘上，之后可以逐字节精确地重新加载，而无需重新计算。我们在5000万个真实公开文本令牌上运行了该系统，通过vLLM在单块NVIDIA H100上服务，使用了Gemma 4 12B和Gemma 4 31B两个模型。在两个模型上，我们探测的每个块都成功从加密存储中加载而无需重新计算（100次探测全部成功，深度从0到5000万令牌）。加载一个块比重新计算快2.8倍到4.3倍，GPU能耗降低8.8倍到12.3倍，并且在处理整个5000万令牌流的过程中GPU内存占用保持恒定。当被问及数百万令牌之前植入的事实时，12B模型在100次中答对了82次，31B模型答对了98次。

    arXiv:2610.10845v1 Announce Type: cross  Abstract: A large language model can only use the text that fits in its context window, and it recomputes its internal key-value (KV) state for a prompt every time the prompt is sent. We test a memory layer, the public package galahad-kv, that saves the KV state of each block of about 16,000 tokens to encrypted local NVMe disk and loads it back later, byte-exact, without recomputing it. We ran it on 50,000,000 tokens of real public text, served through vLLM on one NVIDIA H100, with Gemma 4 12B and Gemma 4 31B. Every block we probed was loaded back from the encrypted store with no recompute (100 of 100, at depths from 0 to 50M tokens) on both models. Loading a block was 2.8x to 4.3x faster than recomputing it and used 8.8x to 12.3x less GPU energy, and GPU memory stayed flat over the whole 50M-token stream. Asked about facts planted millions of tokens earlier, the 12B model gave the right answer 82 times out of 100 and the 31B model 98 times out 
    
[^132]: 大规模语法概念标注：部署的微调小型语言模型优于提示式前沿模型

    Grammar Concept Annotation at Scale: Deployed Fine-Tuned Small Language Models Outperform Prompted Frontier Models

    [https://arxiv.org/abs/2610.10827](https://arxiv.org/abs/2610.10827)

    通过在教师生成的监督数据上微调小型语言模型并部署 0.8B 模型进行大规模语法概念标注，其性能在精确率和召回率上超越提示式前沿大模型，同时大幅降低成本。

    

    纠错反馈是第二语言习得中被最有力证据支持的驱动因素之一，然而课程中提供的纠错很少能积累成对语法掌握情况的可操作视图。通过提示（prompting）的前沿模型可以从学习者与导师的课程记录中提供这样的视图，但在大规模应用时成本高昂。我们通过在经过筛选和重新平衡的教师生成监督数据上微调 Qwen3.5 小型语言模型（SLM）来弥合这一差距，随后将一个高效的 0.8B 模型部署到面向平台所有英语学习者的端到端语法掌握追踪系统中。通过将标注规范内化到适配器权重中，使得 0.8B 模型能够搭配简短的匹配提示，而无需冗长的指令。在两个人工策展的基准测试上，无论是部署的 0.8B 模型还是 4B 参考对比模型，在递进严格程度的嵌套匹配标准（概念、证据……）下，其精确率和召回率均优于经过提示的 GPT-5.4 和 GPT-5.6 Sol。

    arXiv:2610.10827v1 Announce Type: cross  Abstract: Corrective feedback is among the best-evidenced drivers of second-language acquisition, yet corrections delivered during lessons rarely accumulate into an actionable view of grammar mastery. Prompted frontier models can provide such a view from learner--tutor lesson transcripts, but they are costly at scale. We close this gap by fine-tuning Qwen3.5 small language models (SLMs) on filtered and rebalanced teacher-generated supervision, then deploying an efficient 0.8B model in an end-to-end grammar mastery tracker for all English learners on our platform. Internalizing the annotation contract into adapter weights enables pairing the 0.8B model with a compact matched prompt rather than verbose instructions. On two human-curated benchmarks, both the deployed 0.8B model and a 4B reference comparator outperform prompted GPT-5.4 and GPT-5.6 Sol in precision and recall under nested matching criteria of increasing strictness: concept, evidence 
    
[^133]: NavGPT-3：在分层导航运行时中利用上下文

    NavGPT-3: Harnessing Context in a Hierarchical Navigation Runtime

    [https://arxiv.org/abs/2610.10787](https://arxiv.org/abs/2610.10787)

    NavGPT-3 提出了一个类似操作系统的分层导航运行时框架，将具备长时程推理能力的语言模型与低延迟的 VLA 动作策略通过多线程调度机制相结合，使机器人能通过线程中断与切换快速响应突发真实事件，其 8B VLA 模型在 R2R-CE 上取得了 74.51 SR 的领先性能。

    

    通过长时程智能体强化学习训练的语言模型能够通过推理泛化知识、表达精确动作，并在多步骤中追求目标，从而提升了具身智能体所能理解和决策的上限。然而，物理交互仍然是动作策略的领域，动作策略提供密集、低延迟的控制。我们提出了 NavGPT-3，这是一个连接两种模型的框架，其上构建了类似操作系统的运行时：推理、行动和监控作为各自拥有独立上下文、工具和权限的线程运行，而运行时负责调度这些线程并决定哪个线程控制机器人的运动，使机器人能够通过中断和线程切换对突发的真实世界事件做出反应。在其底层，我们的动作策略 NavGPT VLA 在 1928 万条样本上训练，采用编解码器分配方式按场景变化比例分配视觉 token；其 8B 模型在 R2R-CE 上单独达到 74.51 SR 并处于领先地位。

    arXiv:2610.10787v1 Announce Type: cross  Abstract: Language models trained with long-horizon agentic reinforcement learning can generalize knowledge through reasoning, express precise actions, and pursue goals over many steps, raising the ceiling on what an embodied agent can understand and decide. Physical interaction, however, remains the domain of action policies, which provide dense, low-latency control. We present NavGPT-3, a harness that connects the two models, with an OS-like runtime built above it: reasoning, acting, and monitoring run as threads with their own context, tools, and permissions, while the runtime schedules them and decides which thread controls the robot's motion, so that the robot can react to sudden real-world events through interruption and thread switching. Beneath it, our action policy NavGPT VLA, trained on 19.28M examples, allocates visual tokens using codec allocation, in proportion to scene change; its 8B model alone reaches 74.51 SR on R2R-CE and leads
    
[^134]: Plan-and-Patch：面向智能体规划的扩散语言模型

    Plan-and-Patch: Diffusion Language Models for Agentic Planning

    [https://arxiv.org/abs/2610.10786](https://arxiv.org/abs/2610.10786)

    提出Plan-and-Patch框架，利用扩散语言模型通过并行去掩码生成结构化的类程序计划，并借助仅填充受影响区域、保持前后步骤不变的局部修复机制，实现对计划的高效修订。

    

    规划对于长程智能体而言日益重要，成功的执行需要在多个步骤中协调子目标、工具使用和中间结果。然而，规划阶段做出的假设可能被环境推翻，工具可能返回意外的结果，动作也可能失败。因此，有效的智能体不仅需要生成计划，还必须能够对计划进行修改。此类修改往往只影响计划的一部分，其前后的结构保持不变。与其重新生成整个计划而冒着引入不必要更改的风险，不如基于被保留的前缀和后缀，仅对受影响的区域进行重新生成以完成修复。我们提出了Plan-and-Patch，这是一个“规划-执行”框架，其中扩散语言模型（dLLM）通过并行去掩码的方式生成结构化的、类似程序的计划，并通过在保持周围步骤固定的情况下填充选定区域来修复计划。我们比较了DreamReasoner-8B和Qwen3-8B作为扩散……（摘要在此处截断）

    arXiv:2610.10786v1 Announce Type: new  Abstract: Planning is increasingly important for long-horizon agents, where successful execution requires coordinating subgoals, tool use, and intermediate outcomes over many steps. Yet assumptions made during planning may be invalidated by the environment, tools may return unexpected results, or actions may fail. Effective agents must therefore not only generate plans, but also revise them. Such revisions often affect only part of a plan, leaving the preceding and subsequent structure intact. Rather than regenerate the entire plan and risk unnecessary changes, repair can regenerate the affected region conditioned on the preserved prefix and suffix. We introduce Plan-and-Patch, a plan-and-act framework in which a diffusion language model (dLLM) generates a structured, program-like plan through parallel unmasking and repairs it by filling in selected regions while keeping the surrounding steps fixed. We compare DreamReasoner-8B and Qwen3-8B as diff
    
[^135]: 先澄清，再聚焦：面向大规模对话分析的语句规范化

    Clarify, Then Focus: Statement Normalization for Conversation Analytics at Scale

    [https://arxiv.org/abs/2610.10758](https://arxiv.org/abs/2610.10758)

    提出语句规范化方法，将对话转化为带说话者归属、来源引用和语义标签的简短语句，使语义更明确并支持按需证据选择，从而提升大规模企业对话分析中下游模型的表现。

    

    企业对话分析需要对数百万次交互回答许多问题。每个问题都可能需要重构说话者的真实含义并识别哪些信息重要，从而在相同的转录文本上重复进行代价高昂的解读工作。我们提出了一个简单的原则：先澄清文本，再聚焦读者。语句规范化将对话转化为简短的、标注说话者归属的语句，并附带来源引用和语义标签。这些语句使含义更加明确；标签则支持为特定问题选择证据。下游模型可以根据什么有助于其做出决策，选择使用完整的表示或相关的子集。在客服通话的优惠信息抑制任务中，规范化在不使用选择机制的情况下即可提升有监督分类器的性能，而较弱的提示式阅读器则同时从规范化和证据选择中受益。小型模型能够学会这一规范化契约，而轻量级编码器……

    arXiv:2610.10758v1 Announce Type: cross  Abstract: Enterprise conversation analytics asks many questions of millions of interactions. Each question can require reconstructing what people mean and identifying which information matters, repeating costly interpretive work across the same transcripts. We propose a simple principle: clarify the text, then focus the reader. Statement normalization transforms dialogue into short, speaker-attributed statements with source references and semantic tags. The statements make meaning more explicit; the tags support selecting evidence for a particular question. Downstream models can use the full representation or a relevant subset, depending on what helps them make the decision. In an offer-suppression task on customer-service calls, normalization improves a supervised classifier without selection, while weaker prompted readers benefit from both normalization and selection. A small model can learn the normalization contract, while lightweight encode
    
[^136]: 表格数据上的对话式任务消歧：泄漏感知的形式化、基准套件与训练

    Conversational Task Disambiguation over Tabular Data: Leakage-Aware Formulation, Benchmark Suite, and Training

    [https://arxiv.org/abs/2610.10740](https://arxiv.org/abs/2610.10740)

    提出“歧义可验证任务”的形式化框架，将智能体分解为提问策略与求解策略、环境分解为oracle与验证器，从而实现任务消歧与求解能力的独立评估，并给出oracle泄漏的形式化定义与无裁判的泄漏诊断。

    

    表格数据上的对话式任务消歧，是指在生成针对表格或数据库的解决方案之前，通过对话来补充关于用户预期任务的缺失信息。现有的评估与训练缺乏泄漏感知的基础。任务的成功混淆了智能体的消歧能力与解决方案生成能力，还可能反映oracle泄漏，即用户模拟器透露了超出真实用户会透露范围的信息。此外，现有数据集也缺乏对歧义和访问边界的统一表示。我们提出了“歧义可验证任务”的概念，它对歧义及其消解进行形式化，将智能体分解为提问策略与求解策略，将环境分解为oracle与验证器。该框架提供了将任务消歧与解决方案生成分开评估的基线和指标、oracle泄漏的形式化定义、无需裁判的泄漏诊断，以及

    arXiv:2610.10740v1 Announce Type: cross  Abstract: Conversational task disambiguation over tabular data uses dialogue to resolve missing information about a user's intended task before producing a solution over tables or databases. Existing evaluation and training lack a leakage-aware foundation. Task success mixes the agent's disambiguation and solution-generation capabilities and can also reflect oracle leakage, that is, information that a user simulator reveals beyond what a real user would. Existing datasets also lack a shared representation of ambiguities and access boundaries. We introduce the notion of an ambiguous verifiable task, which formalizes ambiguities and resolutions, decomposing the agent into an asking policy and a solution policy, and the environment into an oracle and verifier. This framework provides baselines and metrics for evaluating task disambiguation separately from solution generation, formal definitions of oracle leakage, judge-free leakage diagnostics, and
    
[^137]: 有损压缩文本自编码器

    Lossy Compressive Text Autoencoders

    [https://arxiv.org/abs/2610.10738](https://arxiv.org/abs/2610.10738)

    提出一种带残差低维离散瓶颈的文本自编码器，实现文本的有损压缩表示，在网页文本上以每字节2.24比特达到与无损压缩算法相当的压缩率，同时保持良好的重构质量和下游任务性能。

    

    我们的工作探索学习文本的压缩潜在表示，处于数据压缩与表示学习的交叉领域。我们提出了一种自编码器架构，该架构沿时间轴对隐藏表示进行残差式降维与升维，并带有残差低维离散瓶颈。我们针对不同的量化方法、训练目标和数据集分析了我们的方法。对于不同的压缩级别，我们在表层（BLEU）和语义层（基于LLM的评判）上评估原始文本与重构文本之间的相似性。此外，我们还在下游问答和语义文本相似度基准上评估了我们的模型。我们的方法所得到的压缩表示在网页文本数据上达到每字节2.24比特，与无损文本压缩算法相当，同时具有良好的重构质量和下游任务性能。

    arXiv:2610.10738v1 Announce Type: new  Abstract: Our work explores learning a compressed latent representation of text, at the intersection of data compression and representation learning. We propose an autoencoder architecture that performs residual downscaling and upscaling of hidden representations along the time axis, with a residual low-dimension discrete bottleneck. We analyze our approach for different quantization methods, training objectives, and datasets. For different levels of compression, we evaluate the similarity between the original and reconstructed text both at the surface-level (BLEU) and at the semantic-level (LLM-based judge). Additionally, we evaluate our models on downstream question-answering and semantic text similarity benchmarks. Our approach results in compressed representations which are on par with lossless text compression algorithms at 2.24 bits per byte on web text data, while having good reconstruction and downstream task performance.
    
[^138]: 认知温度计：机器学习与逻辑复杂性

    Cognitive Thermometers: Machine Learning and Logical Complexity

    [https://arxiv.org/abs/2610.10724](https://arxiv.org/abs/2610.10724)

    本文提出将机器学习模型视为“认知温度计”来度量语义复杂度，证明基于学习的复杂度比逻辑复杂度能更好地解释自然语言对语义范畴的偏好，并为连接符号逻辑与连接主义AI提供了统一框架。

    

    人类心智是如何表征语义范畴的？为什么自然语言更偏好某些意义而非其他意义？以往的解释依赖于逻辑可定义性与逻辑复杂度，但这两者对逻辑语言的选择高度敏感，使得某些设计选择缺乏动机依据。在本文中，我们提出机器学习为度量语义复杂度提供了一种相对更不可知论的方法。我们回顾了新近的证据，表明逻辑与机器学习在相对复杂度及其在语义类型学中产生的效应上往往得出趋同的结果。而在两者出现分歧之处，基于学习的解释似乎比逻辑复杂度更为有效。我们主张，将机器学习模型视为“认知温度计”，能够提供一种统一的复杂度度量方法，从而在符号逻辑与连接主义人工智能之间架起桥梁。

    arXiv:2610.10724v1 Announce Type: new  Abstract: How does the human mind represent semantic categories? Why do natural languages favor certain meanings over others? Prior explanations have relied on logical definability and complexity, but these are highly sensitive to the choice of logical language, rendering some design choices unmotivated. In this article, we propose that machine learning provides a somewhat more agnostic approach to measuring semantic complexity. We review emerging evidence that logic and machine learning often yield converging results on relative complexity and its resulting effects in semantic typology. Where they diverge, learning appears to be a better explanation than logical complexity. We argue that treating machine learning models as ``cognitive thermometers'' enables a unified approach to complexity that bridges symbolic logic and connectionist AI.
    
[^139]: 大语言模型辅助编制交通管理计划：以威斯康星州交通厅WisTMP系统为例的案例研究

    Large Language Model-Assisted Preparation of Transportation Management Plans: A Case Study with WisDOT WisTMP System

    [https://arxiv.org/abs/2610.10650](https://arxiv.org/abs/2610.10650)

    本文提出一个基于开源大语言模型微调的框架，通过将历史WisTMP文档转化为结构化问答数据集进行训练并本地化部署，实现了交通管理计划内容的自动生成，在提升编制效率的同时保障了数据安全。

    

    工作区是交通基础设施中至关重要却又存在危险的组成部分，需要精心设计的交通管理计划（TMP）来确保安全与通行能力。然而，TMP的编制仍然是一项劳动密集型工作，且严重依赖从业人员的专业知识。本文提出了一个由大语言模型（LLM）辅助的框架，用于自动生成TMP内容，并以威斯康星州交通厅的WisTMP系统作为应用场景。该框架对不同规模的开源大语言模型进行微调，并将其本地化部署以确保数据安全。为支持模型训练，我们通过将历史WisTMP文档中的PDF文件转换为JSON格式的结构化问答对，构建了一个领域专用数据集。实验结果表明，微调在标准文本生成指标上显著提升了性能。进一步的分节和策略层面的分析表明，尽管大语言模型能够……

    arXiv:2610.10650v1 Announce Type: new  Abstract: Work zones are critical yet hazardous components of transportation infrastructure, requiring carefully designed Transportation Management Plans (TMPs) to ensure safety and mobility. However, TMP preparation remains labor-intensive and heavily dependent on practitioner expertise. This paper proposes a Large Language Model (LLM)-assisted framework to automate TMP content generation, leveraging the WisDOT WisTMP system as the application context. The framework fine-tunes multiple open-source LLMs across different model scales and deploys them locally to ensure data security. To support model training, we construct a domain-specific dataset from historical WisTMP documents by converting PDF files into structured question-answer pairs in JSON format. Experimental results show that fine-tuning significantly improves performance across standard text generation metrics. Further section-wise and strategy-level analyses reveal that, while LLMs ach
    
[^140]: 循环式自我提升：面向循环语言模型的动态跨环在线策略蒸馏

    Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models

    [https://arxiv.org/abs/2610.10623](https://arxiv.org/abs/2610.10623)

    LoopOPD让循环语言模型利用自身更深层循环计算作为冻结“教师”，在学生自生成轨迹上进行在线策略蒸馏，无需外部教师或特权信息即可获得密集监督实现自我提升，D-LoopOPD进一步将该过程动态化。

    

    循环语言模型通过在循环计算步骤中复用共享参数，提供了一种参数高效地扩展推理能力的方法。尽管前景广阔，循环语言模型的有效后训练仍然充满挑战。现有方法要么提供稀疏的、或跨循环扩展代价高昂的基于奖励的监督，要么依赖外部教师模型或特权信息，导致教师模型可用性受限或师生上下文不匹配。为解决这些局限，我们提出LoopOPD，这是一个跨环在线策略蒸馏框架，它利用循环语言模型内部额外的循环计算作为其自身的监督来源。LoopOPD在学生自身生成的轨迹上，以冻结的终端循环策略作为具有计算特权的教师来指导中间循环的学生，从而在无需外部教师或特权信息的情况下提供密集监督。我们进一步提出动态LoopOPD（D-LoopOPD），它持续……（原文摘要在此处截断）

    arXiv:2610.10623v1 Announce Type: cross  Abstract: Looped Language Models (LoopLMs) offer a parameter efficient approach to scaling reasoning by reusing shared parameters across recurrent computation steps. Despite their promise, effective post-training of LoopLMs remains challenging. Existing approaches either provide reward based supervision that is sparse or costly to extend across loops, or rely on external teachers or privileged information, leading to limited teacher availability or teacher-student context mismatch. To address these limitations, we introduce LoopOPD, a cross-loop on-policy distillation framework that uses additional recurrent computation within a LoopLM as its own source of supervision. LoopOPD uses a frozen terminal loop policy as a compute privileged teacher for an intermediate loop student on student generated rollouts, providing dense supervision without an external teacher or privileged information. We further propose Dynamic LoopOPD (D-LoopOPD), which conti
    
[^141]: WorldBench：评估大语言模型在Three.js体素世界生成上的能力

    WorldBench: Evaluating LLMs on Three.js Voxel World Generation

    [https://arxiv.org/abs/2610.10622](https://arxiv.org/abs/2610.10622)

    WorldBench通过让评判系统主动探索运行中的3D世界（控制时钟、环绕观察、派遣导航智能体取景）并将视觉观察与源代码相互交叉验证，解决了现有单一视角评判方法不可靠的问题，实现了对LLM生成的Three.js体素世界的可靠评估。

    

    arXiv:2610.10622v1 公告类型：cross 摘要：大语言模型现在已经能够以代码形式编写完整、可交互的3D世界，但对这些世界进行自动化评分却并不可靠。现有的评判方法只采用单一视角：要么由视觉-语言模型对少量渲染快照打分，要么由语言模型阅读源代码。在五个前沿模型生成的世界上，我们发现这两种视角在32%的必需项目上存在分歧，其中大部分是任何画面都无法展示的代码，而且固定视角会遗漏小型的特写内容。我们提出了WorldBench，一个针对开放式、由LLM生成的Three.js世界的基准与评判系统。从一个描述漂浮体素岛屿的提示词出发（包含十个生物群系、物理系统以及昼夜和四季循环），该评判系统会探索正在运行的世界，控制其时钟、绕轨道观察，并派遣一个导航智能体为每个生物群系取景，同时阅读代码来核实所见内容。两个信息通道都不会被单独信任：代码引用只有在源代码确实包含相应文本时才有效，而视觉声明则会……（原文在此处截断）

    arXiv:2610.10622v1 Announce Type: cross  Abstract: Large language models can now write complete, interactive 3D worlds as code, but grading those worlds automatically is unreliable. Existing judges take one view of the output: a vision-language model scores a few rendered snapshots, or a language model reads the source. On worlds written by five frontier models we find that the two views disagree on 32% of required items, mostly code that no frame shows, and that fixed views miss small close-up contents. We present WorldBench, a benchmark and judge for open-ended, LLM-generated Three.js worlds. From one prompt describing a floating voxel island with ten biomes, physics, and day/night and seasonal cycles, the judge explores the running world, controlling its clock, orbiting it, and sending a navigator agent to frame each biome, and reads the code for what it sees. Neither channel is trusted on its own: a code quote counts only if it is text the source contains, and visual claims are che
    
[^142]: Wieszcz-XIX：一个31亿词规模的1918年前波兰语语料库及从零训练的时间受限语言模型

    Wieszcz-XIX: A 3.1-Billion-Word Corpus of Pre-1918 Polish and Temporally Bounded Language Models Trained From Scratch

    [https://arxiv.org/abs/2610.10592](https://arxiv.org/abs/2610.10592)

    该研究构建了31亿词规模的1800-1918年波兰语历史语料库Wieszcz-XIX，其规模比现有标注语料库大三个数量级以上，并通过过滤、去重和时间泄漏审计等严格流程量化控制数据缺陷，进而从零训练了时间受限的语言模型。

    

    历史波兰语作为一种语言虽有充分的文献记录，但就本文所涵盖的时期而言，以机器可读形式标注的文本仅有约一百万词，其余内容则受制于质量参差不齐的光学字符识别。我们提出了Wieszcz-XIX，这是一个包含67.5亿个词元（约31亿词）、由294,369份文档（其中大部分为期刊）组成的语料库，收录了1800至1918年间出版的波兰语文本。该语料库从Wolne Lektury和Internet Archive汇编而成，其构建流程包括过滤、去重、对1918年后内容泄漏的审计以及文档级别的切分。该语料库比同一时期的标注语料库大三个数量级以上，我们对其缺陷进行了量化：以假阳性底线为基准衡量的识别损坏、已被去除的近乎相同的重复内容，以及1918年后的内容泄漏——后者已从训练语料库本身中排除，仅在转录来源中残留已知比例的0.04%至0.38%（按字节数计），因此该……

    arXiv:2610.10592v1 Announce Type: new  Abstract: Historical Polish is well documented as a language but annotated in machine-readable form only to about a million words for the period this paper covers; the rest sits behind optical character recognition of variable quality. We present Wieszcz-XIX, a corpus of 6.75 billion tokens (about 3.1 billion words) in 294,369 documents, most of them periodical issues, of Polish published from 1800 to 1918, assembled from Wolne Lektury and the Internet Archive by a pipeline that filters, deduplicates, audits for post-1918 leakage and splits at the document level. It is over three orders of magnitude larger than the annotated corpus of the same period, and we quantify its defects: recognition corruption against a false-positive floor, near-identical duplication, which is removed, and post-1918 leakage, which is excluded from the training corpus itself down to a known residue of 0.04 to 0.38% of its bytes, found in the transcribed source, so the pub
    
[^143]: Diffu-LoRA：一种用于个性化扩散模型的新型低秩适应方法

    Diffu-LoRA: A Novel Low-Rank Adaptation for Personalized Diffusion Models

    [https://arxiv.org/abs/2610.10550](https://arxiv.org/abs/2610.10550)

    Diffu-LoRA 是一种参数高效的个性化扩散模型方法，通过门控低秩适应、双层优化和渐进式剪枝自动学习各层之间适应能力的非均匀分配，在冻结预训练主干网络的同时以极少参数实现主体身份保持与提示词遵循。

    

    从少量参考图像对文本到图像扩散模型进行个性化，需要在遵循描述新场景的提示词的同时保持主体的身份特征。全模型微调需要大量参数，而低秩适应（LoRA）虽然减少了可训练参数的数量，但尚未解决适应能力应如何在各层之间分配的问题。我们提出了 Diffu-LoRA，这是一种参数高效的方法，通过门控低秩适应来学习这种分配。Diffu-LoRA 将可训练的低秩组件插入 Transformer 块的线性层中，并为每个组件分配一个可学习的门控。双层优化在独立的数据划分上更新适应权重和门控参数，同时渐进式剪枝会移除门控值最低的组件，以满足预定的秩预算。该过程在保持预训练主干网络冻结的同时，在各层之间非均匀地分配适应能力。实验（摘要在此处截断）。

    arXiv:2610.10550v1 Announce Type: new  Abstract: Personalizing text-to-image diffusion models from a few reference images requires preserving subject identity while following prompts that describe new contexts. Full-model fine-tuning is parameter-intensive, whereas low-rank adaptation (LoRA) reduces the number of trainable parameters but leaves open how adaptation capacity should be distributed across layers. We introduce Diffu-LoRA, a parameter-efficient method that learns this allocation through gated low-rank adaptation. Diffu-LoRA inserts trainable low-rank components into the linear layers of Transformer blocks and assigns a learnable gate to each component. Bilevel optimization updates the adaptation weights and gate parameters on separate data splits, while progressive pruning removes components with the lowest gate values to meet a prescribed rank budget. This procedure allocates adaptation capacity nonuniformly across layers while keeping the pretrained backbone frozen. Experi
    
[^144]: 一个可解释的、以表头为中心的大规模语义表解释与数据质量评估框架

    An Explainable Header-Centric Framework for Large-Scale Semantic Table Interpretation and Data Quality Assessment

    [https://arxiv.org/abs/2610.10541](https://arxiv.org/abs/2610.10541)

    该论文提出了一个可解释的、以表头为中心的框架，通过利用精心策划的词汇资源将表头映射到39种可解释类型并保留词元级可追溯性，实现了仅基于元数据场景下的列类型标注与数据质量评估，并将数据质量检测结果聚合为轻量级的数据源级质量指标HeadersIQ。

    

    知识图谱（KG）的质量不仅取决于下游的图验证，还取决于集成前所使用的表格元数据的质量。在仅基于元数据的语义表解释（STI）场景中，当单元格值不可用、含噪或不适用时，列标题成为可追溯知识图谱构建的关键语义证据来源。我们提出了一个可解释的、以表头为中心的框架，用于仅基于元数据的列类型标注（CTA）和数据质量评估（DQA）。该框架利用精心策划的词汇资源将表头映射到39种可解释的FinalFormat类型，并通过SourceKeywords保留词元级的可追溯性。每个被分配的类型都会基于数据质量问题（DQI）分类体系激活相应的验证规则，从而产生缺失数据、重复项、领域违规、错误数据类型以及时间不匹配等检测结果。这些检测结果被聚合到HeadersIQ中，这是一个轻量级、无权重的数据源级质量指标。

    arXiv:2610.10541v1 Announce Type: new  Abstract: Knowledge Graph (KG) quality depends not only on downstream graph validation, but also on the quality of tabular metadata used before integration. In metadata-only Semantic Table Interpretation (STI), where cell values are unavailable, noisy, or unsuitable, column headers become a critical source of semantic evidence for traceable KG preparation.   We present an explainable, header-centric framework for metadata-only Column Type Annotation (CTA) and Data Quality Assessment (DQA). The framework maps headers to 39 interpretable FinalFormat types using curated lexical resources and preserves token-level traceability through SourceKeywords. Each assigned type activates validation rules based on a taxonomy of Data Quality Issues (DQIs), producing detections such as missing data, duplicates, domain violations, wrong data type, and temporal mismatch. These detections are aggregated into HeadersIQ, a lightweight, unweighted data source-level qua
    
[^145]: 潜空间中的评判：基于语义保持压缩的高效生成式奖励建模

    Judging in Latent Space: Efficient Generative Reward Modeling via Semantics-Preserving Compression

    [https://arxiv.org/abs/2610.09788](https://arxiv.org/abs/2610.09788)

    LatentGRM通过语义分块、压缩与重构将评估过程编码为紧凑的连续潜在轨迹，无需逐token生成文本评估即可实现高效奖励建模，并在4B和8B规模上取得与显式SFT评判器相当的偏好判断准确率。

    

    奖励建模通常需要在多个评估准则上进行联合表示与推理，然而逐token地将这一过程用文字表述出来会带来高昂的推理成本。近期关于潜在推理的研究表明，连续状态可能以更紧凑的方式支持这类计算。我们提出了LatentGRM，一个建立在语义分块、压缩与重构之上的潜在评估框架。通过利用准则引导评估的结构来指导压缩，LatentGRM学习到紧凑的连续轨迹，无需生成文本评估即可支持自主的成对判断。一个独立的解释器可以从这些轨迹中重构出评估文本，从而以离线方式展现压缩后所保留的信息。在相同的训练数据和骨干网络条件下，LatentGRM在4B和8B规模上均取得了与显式监督微调（SFT）评判器相当的总体偏好判断准确率。

    arXiv:2610.09788v1 Announce Type: new  Abstract: Reward modeling often requires jointly representing and reasoning over multiple evaluation criteria, yet verbalizing this process token by token can incur substantial inference cost. Recent work on latent reasoning suggests that continuous states may support this computation more compactly. We introduce LatentGRM, a latent evaluation framework built on semantic chunking, compression, and reconstruction. By using the structure of rubric-guided evaluations to guide compression, LatentGRM learns compact continuous trajectories that support autonomous pairwise judgments without generating textual assessments. A separate interpreter reconstructs evaluation text from these trajectories, providing an offline view of the information retained under compression. Under matched training data and backbones, LatentGRM achieves competitive aggregate preference accuracy relative to explicit Supervised Fine-Tuning (SFT) judges at both 4B and 8B scales. A
    
[^146]: 大语言模型在否定句下如何改变预测？

    How Do LLMs Change Predictions Under Negation?

    [https://arxiv.org/abs/2610.09571](https://arxiv.org/abs/2610.09571)

    大语言模型通过“抑制原始答案、提升偏好候选”的机制处理否定，而非像人类那样利用原始答案信息来确定应排除的内容，这一与人类处理方式的差异是模型否定任务失败的关键根源。

    

    否定是人类语言的一个基本特征，然而大型语言模型（LLMs）在处理否定时仍然不可靠。我们在自建的否定基准上评估了最新的开源和闭源LLMs，发现在37%至71%的情况下，模型在否定句下重复给出相同的答案（例如，对于“什么不是西班牙的首都？”回答“马德里”）。为了理解并解决这种脆弱性，我们从机制层面考察了模型在否定下的运作方式。我们的主要发现是，专门的注意力头和MLP神经元通过以下两种方式协同实现否定：（1）抑制对原始答案（如“马德里”）的检索，同时（2）在答案类别中提升某个受偏好候选（如“巴黎”）的概率。这与人类否定处理机制的解释形成对比——在人类处理中，关于原始答案的信息有助于确定应排除的内容。此外，我们发现这种与人类处理方式的差异是否定失败的关键来源。

    arXiv:2610.09571v1 Announce Type: new  Abstract: Negation is an essential feature of human language, yet large language models (LLMs) remain unreliable in processing it. We evaluate recent open-source and closed-source LLMs on our negation benchmark and find that, in 37-71% of cases, they repeat the same answer under negation (e.g., "Madrid" for "What is not the capital of Spain?"). To understand and address this brittleness, we mechanistically examine how models operate under negation. Our main finding is that specialized attention heads and MLP neurons jointly implement negation by (1) suppressing retrieval of the original answer (e.g., "Madrid") while (2) promoting a favored candidate within the answer category (e.g., "Paris"). This contrasts with accounts of human negation processing, in which information about the original answer helps to determine what should be excluded. Furthermore, we find that this difference from human processing is a key source of negation failures: the mod
    
[^147]: 为你的提示加噪：连续扩散语言模型中对条件令牌添加噪声

    Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models

    [https://arxiv.org/abs/2610.09145](https://arxiv.org/abs/2610.09145)

    在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。

    

    我们重新审视了连续扩散语言模型文献中的一个公认标准做法，即在训练期间保持条件提示令牌为干净（无噪声）状态。我们做了一个非常简单的修改：在训练期间也对条件提示令牌添加噪声。我们证明，在这一修改后的训练目标下，模型在数独和N皇后等组合推理任务中获得了更好的泛化能力，且在更难的变体上收益最大（数独困难版的解决率从3.73%提升至24.65%），同时生成解的多样性也有所提高（10x10 N皇后问题的覆盖率从50.60%提升至73.79%）。我们还展示了在使用Gigaword摘要数据集的中等数据规模下，自然语言生成质量有可衡量的提升，但值得注意的是，这些收益并不能迁移到所有自然语言任务上（例如开放式对话生成）。我们的方法只需对训练目标进行单行修改，无需额外的……

    arXiv:2610.09145v1 Announce Type: cross  Abstract: We revisit a standard accepted practice in the continuous diffusion language model   literature of fixing conditioning prompt tokens clean during training.   We make a very simple modification: also noise the conditioning prompt tokens during training.   We demonstrate that under this modified training objective, we achieve better generalization   in combinatorial reasoning tasks such as Sudoku and N-Queens, with the largest gains on harder variants   ($3.73\% \to 24.65\%$ solve rate on Sudoku Hard), and increased diversity of generated solutions ($50.60\% \to 73.79\%$ coverage on   10x10 N-Queens). We also show measurable improvements to natural language generation quality   in modest dataset regimes with Gigaword summarization, but notably demonstrate that gains do not   transfer to all natural language tasks (e.g open ended dialogue generation).   Our method is a single line change to the training objective, requires no additional i
    
[^148]: 利用大语言模型生成的解释来检测情感改写的假新闻

    Leveraging LLM-Generated Explanations for Detecting Emotionally Rewritten Fake News

    [https://arxiv.org/abs/2610.08835](https://arxiv.org/abs/2610.08835)

    本文提出门控交叉注意力（GCA）框架，利用大语言模型从原始新闻生成的解释作为稳定背景知识，自适应融合情感改写新闻与解释内容，显著提升了假新闻检测模型在保持事实的情感变体下的鲁棒性。

    

    摘要：假新闻的传播可能造成严重的社会后果。现有的假新闻检测方法主要关注文体风格的变化，或引入解释等外部信息。然而，新闻文章常常会在保留其潜在事实主张的同时，在不同的情感背景下被改写，这可能影响检测模型的鲁棒性。在本工作中，我们研究了在保持事实不变的情感变体条件下的假新闻检测问题。为研究这一问题，我们构建了情感改写测试集，并从原始新闻文章中生成解释作为稳定的背景知识。随后，我们提出了一种门控交叉注意力框架，自适应地将情感改写的新闻与相应的解释进行整合，使模型能够专注于富含信息的解释内容，同时减少由情感重新表述所引起的潜在不匹配。我们在PolitiFact、GossipCop和LUN数据集上进行了实验。

    arXiv:2610.08835v1 Announce Type: new  Abstract: The spread of fake news may cause severe social consequences. Existing fake news detection methods mainly focus on stylistic variations or incorporate external information such as explanations. However, news articles are often rewritten under different emotional backgrounds while preserving their underlying factual claims, which may affect the robustness of detection models. In this work, we investigate fake news detec- tion under fact-preserving emotional variations. To study this problem, we construct emotion-rewritten test sets and generate explanations from the original news articles as stable background knowledge. We then propose a Gated Cross Attention (GCA) framework that adaptively integrates emotionally rewritten news with the corresponding explanations, enabling the model to focus on informative explanation content while reducing potential mismatches caused by emotional reframing. Experiments on PolitiFact, GossipCop, and LUN d
    
[^149]: 只为FUNS：基于大语言模型引导的时空图节点生成方法用于预测未观测节点状态

    Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States

    [https://arxiv.org/abs/2610.08818](https://arxiv.org/abs/2610.08818)

    该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。

    

    时空预测是物流、城市规划和智能交通系统的基石。然而，受部署成本和维护资源的限制，传感器网络往往缺乏全面的空间覆盖，这使得预测未观测节点状态（FUNS）成为一项至关重要却又极具挑战性的任务。传统模型依赖历史观测数据，在遇到没有先前记录的节点时通常会表现失常。为解决这一问题，我们将该问题重新定义为时空图上的条件生成任务，并提出GenST框架，该框架引入大语言模型（LLMs）作为语义桥梁，利用经过微调的预训练LLM从节点描述（如功能分区和道路网络结构）中提取丰富的语义特征，以弥补缺失的时空信号。具体而言，我们设计了一个两阶段生成架构：时空变分自编码器（VAE）首先压缩……

    arXiv:2610.08818v1 Announce Type: cross  Abstract: Spatio-temporal forecasting is a cornerstone of logistics, urban planning, and intelligent transportation systems. However, constrained by deployment costs and maintenance resources, sensor networks often lack comprehensive spatial coverage, rendering Forecast Unobserved Node States (FUNS) a critical yet formidable challenge. Conventional models rely on historical observations and typically falter when encountering nodes without prior records. To address this, we redefine the problem as a conditional generation task on spatio-temporal graphs and propose GenST, a framework that introduces Large Language Models (LLMs) as a semantic bridge, leveraging a pre-trained LLM fine-tuned to extract rich semantic features from node descriptions, such as functional zones and road network structures, to compensate for missing spatio-temporal signals. Specifically, we design a two-stage generative architecture: a Spatio-Temporal VAE first compresses 
    
[^150]: 我看得够了吗？冻结的视频-语言模型中编码了证据就绪度信号

    Have I Seen Enough? Frozen Video-Language Models Encode Evidence Readiness

    [https://arxiv.org/abs/2610.08560](https://arxiv.org/abs/2610.08560)

    该论文发现冻结的视频-语言模型内部已线性编码了一种由问题条件化、可跨基准泛化且与答案对错无关的“证据就绪度”信号，因此无需额外训练触发器即可判断流式视频问答中证据是否已充分到来。

    

    流式视频-语言模型不仅需要决定回答什么，还必须判断当前问题所需的证据是否已经到来。现有系统将该决策作为一个单独的触发器来学习；我们则探究一个未经修改的模型是否已经在计算这一决策。我们证明，冻结的视频大语言模型内部携带一种线性可读的证据就绪信号，该信号的标注来自带时间戳的证据而非模型输出。在一个共享的逐字节相同的评估中，该信号在全部七个模型中均可被解码（在最严格的“未就绪”采样下AUROC为0.733–0.905，而拟合的时钟模型接近随机水平），并且在完全未接触某基准家族视频数据的情况下训练的探针仍能读取该家族的数据。该信号是问题条件化的：在逐字节相同的视频窗口上，仅改变问题就能使66.1%的问题配对上的读出结果发生反转，而所有问题盲的控制组按构造均处于随机水平。模型即使给出错误答案，仍然编码了就绪状态：在错误答案中AUROC仍为0.722。

    arXiv:2610.08560v1 Announce Type: cross  Abstract: Streaming video-language models must decide not only what to answer, but whether the evidence needed for the current question has arrived. Existing systems learn that decision as a separate trigger; we ask whether an unmodified model already computes it. We show that frozen VideoLLMs carry a linearly readable evidence-readiness signal, labelled from timestamped evidence rather than from model output. It decodes in all seven models of a shared byte-identical evaluation (AUROC 0.733-0.905 under the strictest not-ready sampling, where a fitted clock is near chance), and a probe fitted without any of a benchmark family's footage still reads that family. It is question-conditioned: on byte-identical windows, changing only the question reverses the readout on 66.1% of pairs, while every question-blind control is at chance by construction. The model can answer incorrectly and still encode readiness: AUROC remains 0.722 among wrong answers. Re
    
[^151]: 自我回溯蒸馏：将事后经验转化为先验预见

    Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

    [https://arxiv.org/abs/2610.08077](https://arxiv.org/abs/2610.08077)

    该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。

    

    具有可验证奖励的强化学习（RLVR）主要通过交互后的标量结果奖励将智能体经验转化为学习信号。然而，对于组相对目标而言，当所有采样轨迹获得相同奖励时，这一信号便会消失，即使这些轨迹可能揭示了关于任务需求以及智能体如何失败的有用信息。我们提出了一个互补的问题：事后反思能否教会智能体在行动之前本可预见的东西？我们引入前瞻学习，利用事后经验来监督交互前视角下的预见性预测，并通过自我回溯蒸馏（SRD）加以实例化。直观地说，一条已完成的轨迹揭示了本会有用的知识和本应避免的陷阱；SRD将这种特权的后见之明蒸馏到同一策略的、不依赖轨迹的前瞻预测中。前瞻仅作为训练目标，无需成为……

    arXiv:2610.08077v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be e
    
[^152]: SOL：利用双切片Wasserstein度量衡量文本分布之间的差距

    SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics

    [https://arxiv.org/abs/2610.06513](https://arxiv.org/abs/2610.06513)

    提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。

    

    评估文本生成需要衡量生成分布与数据分布的匹配程度。对于自回归模型，这通过困惑度来实现；而扩散模型和基于流的语言模型只能提供似然界，其紧密程度在不同模型家族之间存在差异。基于样本的替代方法（如结合熵的生成式困惑度）则未考虑分布拟合情况。我们提出了SOL，一种文本分布之间的距离度量。每个序列由其在固定Transformer下隐藏状态的经验测度来表示，并通过双切片Wasserstein距离来比较这些测度的分布。我们证明，当Transformer是单射时，SOL是一个真正的度量。实验表明，SOL能够检测分布性失败、恢复预期的模型趋势，并提供稳定的基于样本的估计。我们提出SOL以填补当前非自回归模型评估协议中的空白。

    arXiv:2610.06513v2 Announce Type: replace  Abstract: Evaluating text generation requires measuring how well the generated distribution matches the data distribution. For autoregressive models, this is done by the perplexity. Diffusion and flow-based language models can only provide a likelihood bound, whose tightness differs between model families. Sample-based substitutes such as generative perplexity with entropy do not consider the distribution fit. We propose SOL,   a distance between text distributions. Each sequence is represented by the empirical measure of its hidden states under a fixed transformer and the distributions of these measures are compared by the double sliced Wasserstein distance. We prove that SOL is a metric if the transformer is injective. Experiments show that SOL detects distributional failures, recovers expected model trends, and provides stable sample-based estimates. We put forward SOL to fill the gap in the current evaluation protocol used for non auto-reg
    
[^153]: 为长程工作智能体扩展可验证环境

    Scaling Verifiable Environments for Long-horizon Work Agents

    [https://arxiv.org/abs/2610.04906](https://arxiv.org/abs/2610.04906)

    WorkForge 提出了一个可扩展的合成框架，从专家工作流和真实世界资源出发，通过提取可核查的事实锚点，自动构建支持长程交互与可信验证的工作智能体训练环境。

    

    工作智能体通过操作数字制品来执行专业性的知识密集型工作，因此需要既能支持长程交互又能提供可信验证的训练环境。然而，手工构建的环境会带来高昂的工程开销，使环境难以规模化；而现有的合成方法则会牺牲工作区的复杂性、真实性或有据可依的可验证性。为了弥合这一差距，我们提出了 WorkForge，一个可扩展的合成框架，用于从真实世界资源构建可验证的工作智能体环境。WorkForge 从专家工作流出发展开：首先识别每个工作流所需的资源、决策和交付成果；然后检索相关的真实世界文件，并将其组织成一个工作区；接着检查该工作区，提取其中内容的具体、可核查的事实。这些事实锚点确定了工作区能够支持的任务类型及其结果的验证方式……（原文摘要在此处截断）

    arXiv:2610.04906v2 Announce Type: replace-cross  Abstract: Work agents operate over digital artifacts to execute professional knowledge-intensive work, requiring training environments that support long-horizon interaction and trustworthy verification. However, hand-crafted environments incur prohibitive engineering overhead that prevents environment scaling, whereas synthesis methods sacrifice workspace complexity, realism, or grounded verifiability. To bridge this gap, we introduce WorkForge, a scalable synthesis framework for constructing verifiable work-agent environments from real-world resources. Starting from expert workflows, WorkForge first identifies the resources, decisions, and deliverables required by each workflow. It then retrieves relevant real-world files and organizes them into a workspace. WorkForge inspects the workspace to extract concrete, checkable facts about its content. These factual anchors fix which task types the workspace can support and how their outcomes 
    
[^154]: 分数并非结构：大脑对齐与跨语言迁移

    The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer

    [https://arxiv.org/abs/2610.03827](https://arxiv.org/abs/2610.03827)

    相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。

    

    相似性分数常被用作模型与大脑或跨语言间共享结构的证据。我们探讨当结构被移除或测量工具失效时，这样的分数实际反映的是什么，并在两个场景中进行了验证。在十七种语言中，语法性探针在语言距离越远时迁移效果越差（r = -0.66），但探针自身的准确率也沿同一轴线下降，在四种语言中处于随机水平（这四种语言是距离最远的五种中的四种）。剔除这些语言会使已解释方差减半，同时由于也缩小了距离范围，该设计无法判断梯度中有多少来自测量工具本身。将272个语言对视为独立样本时，导向效应的p值为0.0006；但以十七种语言为单位时，该效应不显著（p = 0.155）。在大脑对齐方面，一个训练目标将语言模型与fMRI响应的相似性从0.10提升到0.34（可靠性上限为0.54），然而

    arXiv:2610.03827v2 Announce Type: replace-cross  Abstract: Similarity scores are often offered as evidence that a model shares structure with the brain or across languages. We ask what such a score reads when that structure is removed, or when the instrument measuring it does not work, in two settings. Across seventeen languages, a grammaticality probe transfers worse between more distant languages (r = -0.66). But the probe's own accuracy falls along the same axis and is at chance in four languages, four of the five most distant. Dropping them halves the explained variance, and because it also narrows the range of distances, the design cannot say how much of the gradient is the instrument. Counting the 272 language pairs as independent gives p = 0.0006 for a steering effect that is null when the seventeen languages are the unit (p = 0.155). In brain alignment, a training objective raises a language model's similarity to fMRI responses from 0.10 to 0.34 (reliability ceiling 0.54), yet 
    
[^155]: 面向波斯语医学语言模型中声明级幻觉检测的单次前向不确定性头

    Single-Pass Uncertainty Heads for Claim-Level Hallucination Detection in Persian Medical Language Models

    [https://arxiv.org/abs/2610.03482](https://arxiv.org/abs/2610.03482)

    该论文将 LLM 不确定性头框架首次适配到波斯语医学语言模型中，通过在冻结主干模型的注意力图和 token 概率上训练单次前向传播的轻量级声明级检测头，实现了无需重复采样的低成本声明级幻觉检测，并构建了两个波斯语声明级幻觉数据集。

    

    幻觉检测对医学语言模型尤为重要，但重复采样方法成本高昂，且现有的不确定性头资源无法直接迁移到新的主干模型和语言上。我们将 LLM 不确定性头（LUH）框架适配到基于 Aya-Expanse-8B 的波斯语医学模型上，使用先前开发的 Gaokerena-V 和 Gaokerena-R 作为两个主干模型。我们首先在一份包含 168 道题的伊朗医学入学考试上检验响应变异性，观察到 Gaokerena-V 的五次运行一致性显著低于 Aya-Expanse-8B，而 Gaokerena-R 则与 Aya-Expanse-8B 相当。随后，我们直接以波斯语构建了两个成对的声明级幻觉数据集，每个主干模型各包含 1,600 条响应，并在冻结的主干注意力图和 token 概率上训练轻量级的声明级检测头。在留出的测试集划分上，这些检测头获得了 0.4820 和 0.4652 的 PR-AUC 值。

    arXiv:2610.03482v1 Announce Type: new  Abstract: Hallucination detection is particularly important for medical language models, but repeated-sampling approaches are expensive and existing uncertainty-head resources do not directly transfer to a new backbone and language. We adapt the LLM Uncertainty Head (LUH) framework to Aya-Expanse-8B-based Persian medical models, using Gaokerena-V and Gaokerena-R as two previously developed backbones. We first examine response variability on a 168-question Iranian medical entrance examination and observe substantially lower five-run consistency for Gaokerena-V than for Aya-Expanse-8B, whereas Gaokerena-R is comparable to Aya-Expanse-8B. We then construct two paired claim-level hallucination datasets directly in Persian, containing 1,600 responses for each backbone, and train lightweight claim-level heads on frozen backbone attention maps and token probabilities. On held-out test splits, the heads obtain PR-AUCs of 0.4820 and 0.4652, corresponding t
    
[^156]: 评估与提升大语言模型对输入序列变化的鲁棒性

    Evaluating and Improving the Robustness of Large Language Models to Input Sequence Variations

    [https://arxiv.org/abs/2610.02432](https://arxiv.org/abs/2610.02432)

    本论文提出了基于Jensen-Shannon散度的生成式鲁棒性度量R_stab，并开发了自适应进化黑盒攻击方法ASA（对LLM-as-a-Judge系统攻击成功率高达73.8%），用于系统性地评估和提升大语言模型对提示注入、木马后门等对抗性输入序列变化的鲁棒性。

    

    生产系统中的大语言模型（LLM）面临提示注入、木马（后门）攻击以及自动质量指标被操纵等威胁。本论文开发了用于评估和提升大语言模型对对抗性输入序列变化鲁棒性的模型、方法和算法。我们提出了R_stab(f)，一种基于小输入扰动下逐步输出分布之间Jensen-Shannon散度的生成式鲁棒性度量。对于局部化攻击，我们证明了V(h) <= 1 - R_class(h)，其中R_class(h)是决策算子h在小扰动下保持其决策的概率。对于非局部化攻击，我们提出了一个经过校准的经验模型。针对LLM-as-a-Judge（大模型作裁判）系统，我们开发了ASA，一种自适应进化黑盒攻击，其攻击成功率（ASR）最高可达73.8%，在开源模型之间的迁移攻击成功率最高可达62.6%。在Trojan Detection Challenge 2023数据（Pythia-1.4B）上，代理触发器达到REA（摘要在此处被截断）

    arXiv:2610.02432v1 Announce Type: cross  Abstract: Large language models (LLMs) in production systems face prompt injections, trojans (backdoors), and manipulation of automatic quality metrics. This thesis develops models, methods, and algorithms for evaluating and improving LLM robustness to adversarial input sequence variations. We propose R_stab(f), a generative robustness metric based on the Jensen-Shannon divergence between per-step output distributions under small input perturbations. For localized attacks we prove V(h) <= 1 - R_class(h), where R_class(h) is the probability that a decision operator h keeps its decision under small perturbations. For non-localized attacks we propose a calibrated empirical model. For LLM-as-a-Judge systems we develop ASA, an adaptive evolutionary black-box attack that reaches an attack success rate (ASR) of up to 73.8%, with transfer between open models up to 62.6%. On Trojan Detection Challenge 2023 data (Pythia-1.4B), surrogate triggers reach REA
    
[^157]: 用于跨上下文KV缓存复用的预算化缓存修复

    Budgeted Cache Repair for Cross-Context KV-Cache Reuse

    [https://arxiv.org/abs/2610.02233](https://arxiv.org/abs/2610.02233)

    该论文发现跨上下文KV缓存复用会带来显著的准确率损失，并提出预算化缓存修复（BCR）方法，在单token行这一选择仍有收益的最小单元上，利用草稿token的注意力对缓存行排序并精确重算固定预算的行数，从而有效修复缓存误差。

    

    跨上下文KV缓存复用是在新前缀下预测共享片段的键和值，而不是重新计算它们，并且据报告这样做不会带来质量损失。我们的发现并非如此，并识别出两个问题。（1）一个隐性代价：在MMLU和GSM8K上，复用会导致显著的准确率损失。（2）决策单元错误：目前没有任何决定是否复用缓存的规则能消除这一代价。真正有帮助的是选择缓存中哪些部分需要重新计算，而精准选择的收益随着选择单元的增大而下降：在单行（即单个token的键和值）层面，有依据的选择能消除超出随机水平49.5%的缓存误差；在64个token的分块层面为10.6%；而在整个调用层面则毫无效果。预算化缓存修复（BCR）正是在选择仍有收益的单元上进行操作。它从组装好的缓存中起草两个token，根据这两个token对缓存行支付的注意力对缓存行进行排序，并以三种布局之一精确地重新计算固定预算数量的缓存行。

    arXiv:2610.02233v1 Announce Type: cross  Abstract: Cross-context KV-cache reuse predicts a shared segment's keys and values under a new prefix instead of recomputing them, and has been reported to do so without quality loss. We find otherwise, and identify two problems. (1) A hidden cost: on MMLU and GSM8K, reuse costs substantial accuracy. (2) A decision at the wrong unit: no rule for deciding whether to reuse a cache removes that cost. What does help is choosing which parts of the cache to recompute, and the value of choosing well falls as the unit of choice grows: informed selection removes 49.5% of the cache error beyond chance at single rows (one token's keys and values), 10.6% at 64-token chunks, and nothing at the level of whole calls. Budgeted Cache Repair (BCR) acts at the unit where selection still pays. It drafts two tokens from the assembled cache, ranks cache rows by the attention those tokens pay them, and recomputes a fixed budget of rows exactly, in one of three layouts
    
[^158]: 低精度Transformer推理中的随机舍入：一项针对小型GPT-2的变精度仿真研究

    Stochastic Rounding in Low-Precision Transformer Inference: A Variable-Precision Emulation Study of a Small GPT-2

    [https://arxiv.org/abs/2610.01889](https://arxiv.org/abs/2610.01889)

    该研究通过变精度随机舍入（VPSR）算法将PRISM舍入库扩展至任意精度，并从理论与实验两方面揭示：低精度Transformer推理中随机舍入与就近舍入的优劣取决于运算位点，SR的误差以O(√n u)增长而RN以O(n u)增长，这一差距在长MLP下投影等长线性投影中最为明显。

    

    低精度Transformer推理应该使用随机舍入（SR）还是就近舍入（RN）？答案取决于观察网络中的哪个位置。我们通过固定数值格式、仅在各个运算位点改变舍入规则来隔离这一效应。为了能够在自由选择的精度下进行实验，我们通过变精度随机舍入（VPSR）算法将PRISM向量化舍入库扩展至任意虚拟精度，并证明舍入决策在硬件浮点运算中被精确评估。我们提出了两种分析，为这种位点级权衡提供互补的见解。首先，线性投影的概率前向误差界表明，SR的误差包络随约简长度n以O(√n u)增长，而RN为O(n u)，这一差距在低精度下迅速扩大，在较长的多层感知机（MLP）下投影中最为显著。其次，一个……（摘要原文在此处截断）

    arXiv:2610.01889v1 Announce Type: new  Abstract: Should low-precision transformer inference use stochastic rounding (SR) or round-to-nearest (RN)? The answer depends on where in the network you look. We isolate this effect by holding the numerical format fixed and varying only the rounding rule at individual operation sites. To enable experiments at freely chosen precisions, we extend the PRISM vectorized rounding library to arbitrary virtual precision via a variable-precision stochastic rounding (VPSR) algorithm, proving that the rounding decision is evaluated exactly in hardware floating point.   We develop two analyses providing complementary insight into this site-level trade-off. First, a probabilistic forward-error bound for linear projections shows that SR's error envelope grows as $O(\sqrt{n} u)$ in reduction length $n$, versus $O(n u)$ for RN, a gap that widens rapidly at low precision and is most pronounced in the long multilayer perceptron (MLP) down-projection. Second, a se
    
[^159]: 空洞承诺：当智能体许下其运行时无法兑现的诺言

    Empty Commitments: When Agents Promise What Their Runtime Cannot Deliver

    [https://arxiv.org/abs/2610.01045](https://arxiv.org/abs/2610.01045)

    本文提出“空洞承诺”这一新概念，指智能体（如聊天机器人）承诺其工具和运行时配置根本无法实现的未来行动（例如“我明天会提醒你”），并形式化了包含三种失败类型的承诺语义、工具锚定条件和结果分类体系，同时设计了一种通过逐步增加持久性能力来测量空洞承诺的实验协议。

    

    一个说“我明天会提醒你”的聊天机器人，在用户再次发消息之前不会再次运行。我们将这样的承诺称为“空洞承诺”（empty commitment）：即一种关于当前轮次之后采取行动的承诺，而智能体的工具或运行时中没有任何东西能够实现这一行动。与违背承诺不同，这种承诺的空洞性仅由智能体的配置本身决定，无需考察后续交互轨迹。我们在承诺语义的基础上定义了空洞承诺，包含三种失败类型、一个用于判断承诺是否能被某个工具真正实现的锚定条件，以及一个响应层面的结果分类体系。随后我们描述了一种测量协议：让后续请求在五种设置中运行，每种设置逐步增加一项持久性支持能力，同时环境可以是隐式的或明确说明的。

    arXiv:2610.01045v1 Announce Type: new  Abstract: A chatbot that says "I will remind you tomorrow" will not run again until the user writes. We call such a promise an empty commitment: a promise of an action after the current turn that nothing in the agent's tools or runtime can carry out. Unlike a broken promise, its emptiness follows from the agent's configuration alone; no later trajectory is needed. We define empty commitments on top of commitment semantics, with three failure types, an anchoring condition for promises that a tool could make real, and a response-level outcome taxonomy. We then describe a measurement protocol: follow-up requests run in five setups that add one persistence affordance at a time, with the environment either left implicit or stated.
    
[^160]: 大语言模型人格遗忘

    LLM Persona Unlearning

    [https://arxiv.org/abs/2609.39882](https://arxiv.org/abs/2609.39882)

    该论文提出“人格遗忘”任务及PersonaUnlearnBench基准，通过权重级编辑使大语言模型中的指定人格难以被诱发，并发现标准遗忘方法无法在不牺牲生成质量或通用能力的情况下可靠地抹除目标人格。

    

    预训练使大语言模型（LLM）具备了与角色、风格、价值观和目标相关的广泛行为模式。后训练教会模型有条件地执行这些模式，并将乐于助人的“助手”设为默认，但并未从权重中抹除其他替代模式；因此，明确的提示可以诱发出持续影响判断、语言和行动的人格。在开放权重设置中，运行时的控制手段可能被移除，这促使了“人格遗忘”任务的出现：一种权重级别的编辑，使指定的人格在未见过的语境中难以被诱发和执行。我们提出了PersonaUnlearnBench，一个模型特定的配对基准，涵盖来自三个系列的六个大语言模型和五种人格，包含对齐的遗忘/保留数据集、留出的指令改写以及四轴评估。该基准表明，标准的遗忘方法无法在不牺牲有意义生成能力或通用效用的前提下可靠地抹除目标人格。

    arXiv:2609.39882v1 Announce Type: new  Abstract: Pre-training equips large language models (LLMs) with a broad repertoire of behavioral patterns associated with roles, styles, values, and goals. Post-training teaches conditional enactment and makes a helpful Assistant the default, but it does not erase alternative modes from the weights; explicit prompts can therefore elicit personas that repeatedly shape judgment, language, and action. In open-weight settings, runtime controls can be removed, motivating persona unlearning: a weight-level edit that makes a designated persona difficult to elicit and enact on unseen contexts. We introduce PersonaUnlearnBench, a model-specific paired benchmark spanning six LLMs from three families and five personas, with aligned forget/retain sets, held-out instruction paraphrases, and four-axis evaluation. The benchmark shows that standard unlearning methods cannot reliably erase the target persona without sacrificing meaningful generation or general uti
    
[^161]: 跳出思维定式：语言模型能否选择性地依赖外部引导？

    Thinking Outside the Box: Can Language Models Rely on External Guidance Selectively?

    [https://arxiv.org/abs/2609.39578](https://arxiv.org/abs/2609.39578)

    该论文提出 Box²-Bench 基准来衡量语言模型“跳出思维定式”的能力，即在受益于可靠工作流引导的同时否决不可靠引导，并发现反事实监督微调与基于结果的强化学习是提升这一能力的两种互补训练策略。

    

    智能体框架通常通过人类设计的工作流来增强语言模型，但随着模型能力的不断增强，不可靠的引导反而可能越来越多地限制模型的执行。我们将这种既能从有用引导中获益、又能否决不可靠引导的能力称为“跳出思维定式”。我们提出了 Box$^2$-Bench 基准，它在保持模型和任务不变的同时改变工作流的可靠性，以单独考察模型如何调节其对引导的依赖程度。在 Box$^2$-Bench 上，前沿模型通常能从可靠引导中受益，但当引导具有误导性或变得不可靠时仍然脆弱。为检验这种能力是否可以通过学习获得，我们使用坏的工作流训练两个开源权重模型，并将好的工作流保留用于评估。我们探索了两种互补的训练策略：反事实监督微调可以提升鲁棒性，而基于结果的强化学习可以将平衡转向更多地利用有用的工作流。

    arXiv:2609.39578v1 Announce Type: new  Abstract: Agent harnesses often improve language models with human-designed workflows, but as models grow more capable, unreliable guidance can increasingly constrain their execution. We call the ability to benefit from useful guidance while overriding unreliable guidance thinking outside the box. We introduce Box$^2$-Bench, which holds the model and task fixed while varying workflow reliability to isolate how models regulate their reliance on guidance. On Box$^2$-Bench, frontier models often benefit from reliable guidance but remain vulnerable when it is misleading or becomes unreliable. To test whether this capability can be learned, we train two open-weight models using bad workflows, reserving good workflows for evaluation. We explore two complementary training strategies: counterfactual supervised fine-tuning improves robustness, while outcome-based reinforcement learning can shift the balance toward greater use of helpful workflows. We furth
    
[^162]: 约鲁巴语中曲折调的标记方法

    Marking Contour Tones in Yor\`{u}b\'{a}

    [https://arxiv.org/abs/2609.38627](https://arxiv.org/abs/2609.38627)

    本文提出在约鲁巴语正字法中采用caron（ˇ）和circumflex（ˆ）符号来标记单个元音上的升降曲折调，以解决传统拼写中声调信息缺失甚至颠倒姓名含义的问题，并使其首次可通过标准键盘输入和计算文本处理。

    

    约鲁巴语是一种声调语言，其曲折调在正字法书写上一直存在难题。这一问题在个人姓名和词汇中尤为突出，因为这些词的传统拼写避免了元音延长，而元音延长本可为第二个声调提供承载音节。尤其值得关注的是一类姓名，其传统拼写不仅省略了声调信息，还会颠倒名字的含义，有时甚至表达出与名字本意相反的内容。本文描述了这一问题，说明了现有解决方案的不足，并提议采用caron（倒折音符ˇ）和circumflex（抑扬符ˆ）符号。这些符号自Olmsted（1951）以来在约鲁巴语音系学研究中已有先例，作为书写惯例用于单个元音之上，以编码升调和降调曲折调，从而首次使这些曲折调能够通过标准键盘输入和计算文本处理来访问。该提议得到了支持。

    arXiv:2609.38627v1 Announce Type: new  Abstract: Yor\`ub\'a is a tonal language in which contour tones pose persistent orthographic challenges. These are especially notable for personal names and lexical items whose conventional spellings avoid vowel lengthening that would otherwise provide a host syllable for the second tone. A particular concern is a class of names in which the conventional spelling does not just omit tonal information but inverts the meaning of said name, sometimes asserting the opposite of what the name intends. This paper describes the problem, illustrates the inadequacy of current solutions, and proposes the adoption of the caron and circumflex marks. These are symbols with precedent in Yor\`ub\'a phonological scholarship since Olmsted (1951), used as orthographic conventions on single vowels to encode rising and falling contour tones, making them accessible for the first time through standard keyboard input and computational text processing. The proposal is supp
    
[^163]: CompOrca：语料库规模的指令微调数据合规性标注

    CompOrca: Corpus-Scale Compliance Labelling of Instruction-Tuning Data

    [https://arxiv.org/abs/2609.37807](https://arxiv.org/abs/2609.37807)

    该论文提出了 CompOrca，利用开源大模型评判器对整个 OpenOrca 语料库（超过 420 万条样本）进行五次独立判定，首次实现了语料库规模的合规性标注，并发布带投票计数的标注结果，以支持对拒答与不服从行为的研究。

    

    研究微调如何塑造拒答与不服从行为，需要识别出那些拒绝、规避或以其他方式未能完成所请求任务的训练样本。但现有的标注最多只覆盖几千条提示词的评估集。我们提出了 CompOrca，对包含 4,233,923 条样本的 OpenOrca 语料库整体进行了合规性标注。每个样本都由一个开源权重的大语言模型评判器（LongCat-2.0，1.6 万亿参数）经过五次独立判定，被分类为合规或不合规，并将语料库发布为一致合规（94.75%）、一致不合规（1.28%）和非一致行（3.97%）三个部分，同时附带原始投票计数。单次判定会将语料库中 2.7-3.2% 的样本标记为不合规，而只有 1.28% 被全部五次判定标记，这使得过滤最模糊的样本成为可能。在 450 个经人工标注的样本（其中 150 个被标注了两次，人与人之间 κ = 0.93）上，一致合规与不合规……（原文摘要至此截断）

    arXiv:2609.37807v1 Announce Type: new  Abstract: Studying how fine-tuning shapes refusal and noncompliance behaviour requires identifying training examples that refuse, evade or otherwise fail to fulfil the requested task. But existing annotation covers evaluation sets of a few thousand prompts at most. We present CompOrca, a compliance labelling over the entirety of the 4,233,923-example OpenOrca corpus. Every example was classified as compliant or noncompliant by five independent passes of an open-weight LLM judge (LongCat-2.0, 1.6T parameters), and the corpus is released as unanimous compliance (94.75%), unanimous noncompliance (1.28%), and nonunanimous rows (3.97%) along with the raw vote counts. A single pass flags 2.7-3.2% of the corpus as noncompliant, while only 1.28% is flagged by all five, allowing for filtering the most ambiguous samples. Against 450 human-annotated examples, 150 of them annotated twice (human-human $\kappa = 0.93$), the unanimous compliance and noncomplianc
    
[^164]: Opera：面向长时程编码智能体的口头批评框架

    Opera: A Verbal Critic Framework for Long-horizon Coding Agents

    [https://arxiv.org/abs/2609.33987](https://arxiv.org/abs/2609.33987)

    Opera 是一个面向长时程编码智能体的口头批评框架，其核心创新在于将每次纠正作为持久记录持续跟进至问题真正解决，通过周期性/事件驱动触发、类型化算子诊断、基于证据的反馈审计以及后续行动跟踪，区分表面服从与实际解决，从而在多个基准上将智能体解决率提升高达 15 个百分点。

    

    长时程编码智能体需要及时的纠正，然而当反馈误判正在进行的工作或未能解决根本问题时，反馈可能无效甚至有害。现有的批评机制专注于评估轨迹和生成反馈，但很少跟踪反馈发出后发生了什么。我们提出了 Opera，一个口头批评框架，它将每次纠正视为一条持久的记录，并持续跟进直到被诊断出的问题得到解决。Opera 通过周期性和事件驱动的触发器决定何时进行审查，使用类型化算子诊断问题，在反馈发出前根据可见证据对反馈进行审计，并跟踪智能体的后续行动，以区分仅仅是表面服从与问题真正得到解决。作为测试时批评器，Opera 在四个策略模型上，将非批评智能体在 Terminal-Bench 2.1、SWE-Bench Pro 一个子集以及 DeepSWE v1.1 上的解决率分别提升了高达 12.4、15.0 和 8.9 个百分点。

    arXiv:2609.33987v2 Announce Type: replace  Abstract: Long-horizon coding agents need timely corrections, yet feedback can be ineffective or even harmful when it misjudges ongoing work or fails to address the underlying problem. Existing critics focus on evaluating trajectories and generating feedback, but rarely track what happens after feedback is delivered. We present Opera, a verbal critic framework that treats each correction as a persistent note, followed until the diagnosed problem is resolved. Opera decides when to review through periodic and event-driven triggers, diagnoses issues with typed operators, audits feedback against visible evidence before delivery, and tracks the agent's subsequent actions to distinguish mere compliance from actual resolution. As a test-time critic, Opera improves the resolve rate of non-critic agents by up to 12.4, 15.0, and 8.9 percentage points on Terminal-Bench 2.1, a SWE-Bench Pro subset, and DeepSWE v1.1, respectively, across four policy models
    
[^165]: SMAT：简单高效的合并感知训练

    SMAT: Simple and Efficient Merge-Aware Training

    [https://arxiv.org/abs/2609.33437](https://arxiv.org/abs/2609.33437)

    SMAT将常见模型合并操作抽象为缩放、掩码和扰动三种基本操作，通过在采样生成的模拟合并参数上联合优化专家损失与期望损失，实现了以极小训练开销显著提升合并后性能的简单高效合并感知训练方法。

    

    模型合并能够在无需联合重新训练的情况下整合多个专家模型的能力，但标准的专家训练仅优化任务损失，无法保证合并后的良好性能。合并感知训练（MAT）旨在提升合并后的性能，但现有方法未能充分考虑常见的合并操作，且会增加训练成本。我们观察到，从专家模型的角度来看，常见的合并方法可以由三种操作来描述：Scale（缩放）对其自身更新进行重新加权，Mask（掩码）移除选定的坐标，Perturb（扰动）添加来自其他专家的更新。基于这一视角，我们提出了SMAT（简单合并感知训练），它通过采样缩放系数、掩码和加性噪声生成模拟的合并参数，并在这些参数上联合优化专家损失与期望损失。我们进一步引入周期性调度、核融合和参数存储切换机制，使SMAT高效运行，每步仅需一次前向传播和一次反向传播。

    arXiv:2609.33437v2 Announce Type: replace-cross  Abstract: Model merging integrates the capabilities of multiple experts without joint retraining, but standard expert training optimizes task loss alone and does not guarantee good performance after merging. Merge-aware training (MAT) aims to improve merged performance, but existing methods do not fully account for common merging operations and add training cost. We observe that, from an expert's perspective, common merging methods can be described by three operations: Scale reweights its own update, Mask removes selected coordinates, and Perturb adds updates from other experts. Based on this view, we introduce SMAT (Simple MAT), which jointly optimizes expert loss and expected loss at simulated merged parameters generated by sampling scaling coefficients, masks, and additive noise. We further introduce periodic scheduling, kernel fusion, and parameter storage switching to make SMAT efficient, with one forward and one backward pass per s
    
[^166]: 何时驱逐，而非保留什么：面向免训练KV缓存压缩的草稿引导驱逐方法

    When to Evict, Not What to Keep: Draft-Guided Eviction for Training-Free KV-Cache Compression

    [https://arxiv.org/abs/2609.33334](https://arxiv.org/abs/2609.33334)

    该论文提出草稿引导驱逐（DGE）方法，将KV缓存驱逐时机从预填充结束推迟到基于完整缓存起草出前两个答案token之后，通过利用答案自身前缀生成的查询来指导驱逐决策，解决了传统“优化保留什么”策略的补偿效应和选择效应失效问题，实现免训练的KV缓存压缩。

    

    诸如SnapKV、H2O和PyramidKV等免训练KV缓存压缩方法在预填充结束时驱逐token，其目标是保留未来查询预期会使用的注意力质量——即优化“保留什么”。我们证明这一目标会以两种不同的方式失效。（1）补偿效应：恢复被驱逐的注意力质量可以在注意力层面达到目标，却无法恢复任务质量。（2）选择效应：当恢复的注意力质量是碎片化的、而非集中于连贯片段时，覆盖更多真实解码查询的注意力质量反而会损害任务质量。这些失效有着共同的根本原因：驱逐发生在决定答案轨迹的查询尚未出现之时。我们提出草稿引导驱逐方法，它将驱逐时机推迟到使用完整缓存起草出前k=2个答案token之后——仅比预填充多一个解码步骤。由于草稿是由答案自身的前缀生成的，因此不会在……（摘要截断）

    arXiv:2609.33334v2 Announce Type: replace-cross  Abstract: Training-free KV-cache compression methods such as SnapKV, H2O, and PyramidKV evict tokens at the end of prefill, aiming to preserve the attention mass that future queries are expected to use -optimizing what to keep. We show that this objective fails in two distinct ways. (1) Compensation: restoring the evicted attention mass can recover the attention-level target without recovering task quality. (2) Selection: covering more of the true decode-query mass can hurt quality when the recovered mass is fragmented rather than concentrated in coherent spans. These failures share a common cause: eviction occurs before the queries that determine the answer trajectory exist. We propose Draft-Guided Eviction (DGE), which defers eviction until after drafting the first k=2 answer tokens using the full cache - just one decode step beyond prefill. Because the draft is generated from the answer's own prefix, no cache entries are discarded bef
    
[^167]: FA-Bench：面向干净与噪声语音的强制对齐与ASR在音素级和词级时间戳精度上的基准测试

    FA-Bench: A Benchmark for Phone- and Word-Level Timestamp Accuracy in Forced Alignment and ASR on Clean and Noisy Speech

    [https://arxiv.org/abs/2609.32396](https://arxiv.org/abs/2609.32396)

    FA-Bench是一个开放的强制对齐与ASR时间戳精度基准框架，通过统一的评测协议（涵盖21个开源模型和9个商业API、干净与退化语音）和基于容差的F1指标，解决了已有研究方法不一、结果不可比的问题，并消除了标准MAE在依赖识别输出的系统上造成的9%至14%的分数虚高。

    

    摘要（arXiv:2609.32396v2，公告类型：替换）：强制对齐是在给定语音转录文本的情况下，估计语音中每个词、音素或字符的时间戳。已发表的对比研究在转录文本规范化、数据切分和边界匹配方式上各不相同，因此它们的结果无法放在一起解读。我们提出了FA-Bench，一个开源框架，它一次性固定这些选择，并发布代码、数据切分、音素映射、文本规范化和评分脚本，且结果定期公布。Track 1为每个对齐器提供参考转录文本，Track 2为每个对齐器提供识别器的输出，两者使用相同的音频，包括干净音频和四种方式退化的音频，在统一协议下评估了21个开源模型和9个商业API。我们对话语的每个边界进行评分，并检查其两侧的两个标签，因此识别器遗漏或虚构的词都会被计入惩罚。采用基于容差的F1作为主要指标，消除了标准MAE在依赖识别输出的系统上（针对会话语音）造成的9%到14%的分数虚高。

    arXiv:2609.32396v2 Announce Type: replace  Abstract: Forced alignment estimates the timestamps of each word, phone or character in speech given its transcript. Published comparisons normalize transcripts, split the data and match boundaries differently, so their numbers cannot be read together. We present FA-Bench, an open framework that fixes those choices once and releases the code, splits, phone mapping, text normalization and scoring script, with results published periodically. Track 1 gives every aligner the reference transcript and Track 2 gives it a recognizer's output, on the same audio, clean and degraded four ways, with 21 open models and 9 commercial APIs under a unified protocol. We score every boundary of an utterance and check the two labels beside it, so a word the recognizer missed or invented is charged. Using a tolerance-based F1 as our primary metric eliminates the 9% to 14% score inflation that standard MAE causes on recognition-dependent systems in conversational s
    
[^168]: PlurVA-LLM-2026 共享任务赛道一：通过多语言微调与阈值校准实现大语言模型的多元价值对齐

    PlurVA-LLM-2026 Shared Task Track-1: Pluralistic Value Alignment in LLMs via Multilingual Fine-Tuning and Threshold Calibration

    [https://arxiv.org/abs/2609.32382](https://arxiv.org/abs/2609.32382)

    在资源受限条件下，该系统通过 4 比特 QLoRA 微调 Llama 3.1 8B，结合分语言定制的数据增强策略（中文选项排列、印尼语标注者投票扩展、斯里兰卡语二元重构）及条件阈值校准，在三国多元价值对齐任务上取得 0.805 的宏平均准确率。

    

    我们提出了参加 PlurVA-LLM 2026 共享任务赛道一的系统，该赛道聚焦于中国、印度尼西亚和斯里兰卡语境下的多元价值对齐。在这一资源受限的赛道中，我们使用 4 比特 QLoRA 对 Llama 3.1 8B Instruct 进行了微调。我们的方法结合了针对中文数据的选项排列增强、针对印尼语数据的标注者投票扩展，以及针对斯里兰卡语数据的二元重构与 SinhalaMMLU 数据增强。此外，我们还对斯里兰卡语数据的预测结果应用了条件阈值校准。最终系统在中文上取得了 0.785 的准确率，在印尼语上取得了 0.715 的准确率，在斯里兰卡语上取得了 0.916 的准确率，整体宏平均准确率达到 0.805。

    arXiv:2609.32382v2 Announce Type: replace  Abstract: We present our system for the PlurVA-LLM 2026 Shared Task Track-1, which focuses on pluralistic value alignment in the contexts of China, Indonesia, and Sri Lanka. For this resource-constrained track, we fine-tuned Llama 3.1 8B Instruct using 4-bit QLoRA. Our approach combines option-permutation augmentation for Chinese data, annotator vote expansion for Indonesian data, and binary reformulation with SinhalaMMLU augmentation for Sri Lankan data. We further applied conditional threshold calibration to the predictions for the Sri Lankan data. The final system achieved accuracies of 0.785 for Chinese, 0.715 for Indonesian, and 0.916 for Sri Lankan, resulting in an overall macro-average accuracy of 0.805.
    
[^169]: 使用语言模型建模句子理解过程中语境与共指的影响

    Using LMs to Model the Effects of Context and Coreference during Sentence Comprehension

    [https://arxiv.org/abs/2609.32119](https://arxiv.org/abs/2609.32119)

    本研究通过在四个大规模英语阅读时间数据集上系统调节GPT-2的上下文窗口大小，发现上下文长度与人类心理语言学数据的拟合度呈U形关系，扩展上下文（500–1,000词元）拟合最佳，且这一优势依赖于跨句实体共指链的完整性。

    

    语言模型（LMs）常被用作建模人类语言处理的工具。近期研究表明，严格限制语言模型的上下文窗口可以通过模拟人类工作记忆的约束来改善其与人类心理语言学数据的拟合程度。然而，这种严格的记忆衰减方法可能忽视了人类对长程结构表征（如语篇结构）的依赖。在这项工作中，我们在四个大规模自然英语阅读时间数据集上系统地改变GPT-2的上下文窗口大小，并观察到了一种U形关系：尽管受限的上下文（<20个词元）能够成功捕捉局部记忆限制，但扩展的上下文（500–1,000个词元）最终产生了最高的整体心理语言学拟合度。为了探究驱动这一优势的机制，我们进行了一个推理时的反事实实验，通过将重复出现的实体代词化来破坏跨句实体链……（原文摘要此处截断）

    arXiv:2609.32119v2 Announce Type: replace  Abstract: Language models (LMs) are often used as a tool to model human language processing. Recent studies suggest that severely restricting LMs' context window improves their fit to human psycholinguistic data by simulating human working memory constraints. However, it is possible that this strict memory-decay approach overlooks humans' reliance on long-range structural representations, such as discourse structre. In this work, we systematically vary the context window size of GPT-2 across four large-scale naturalistic English reading-time datasets and observe a U-shaped relationship: Although restricted contexts (< 20 tokens) successfully capture local memory limitations, expanded contexts (500--1,000 tokens) ultimately yield the highest overall psycholinguistic fit. To investigate the mechanism driving this benefit, we conduct a counterfactual inference-time experiment that disrupts cross-sentential entity chains by pronominalizing repeate
    
[^170]: SlideLab：以观众为中心的科学幻灯片生成与评估

    SlideLab: Audience-Centered Scientific Slide Generation and Evaluation

    [https://arxiv.org/abs/2609.30294](https://arxiv.org/abs/2609.30294)

    SlideLab 是一个无需训练的多智能体框架，能从研究论文生成以观众为中心的科学演示幻灯片，在盲测中于 77% 的论文上超越开源与商业系统且推理成本降低约 4 倍，并配套提出模拟会议室的观众导向评估框架 ConfArena。

    

    科学演示不仅仅是研究论文的摘要，它们需要以连贯的顺序呈现研究工作，清晰地解释核心思想，并帮助观众跟上演讲的节奏。我们提出了 SlideLab，一个无需训练的多智能体框架，用于从研究论文生成科学演示幻灯片。SlideLab 首先规划演示叙事，然后利用内容规划、视觉生成、布局优化和依据验证等智能体，构建并迭代完善一套共享幻灯片。在一项盲测人类偏好研究中，SlideLab 在 77% 的论文上优于开源和商业系统，同时其推理 token 使用量仅为最强开源基线的约四分之一。我们还引入了 ConfArena，一个面向观众的评估框架，它模拟会议室场景并逐张幻灯片地评估演示效果。ConfArena 的评估结果与人类对系统的排名相一致，并能检测注入的（原文在此处截断）……

    arXiv:2609.30294v1 Announce Type: cross  Abstract: Scientific presentations are more than summaries of research papers. They need to present the work in a coherent sequence, explain the main ideas clearly, and help the audience follow the presentation. We present SlideLab, a training-free multi-agent framework for generating scientific presentations from research papers. SlideLab first plans the presentation narrative, then builds and iteratively refines a shared slide deck using agents for content planning, visual generation, layout refinement, and grounding verification. In a blind human preference study, SlideLab was preferred over both open-source and commercial systems on 77% of papers while using roughly 4 times fewer inference tokens than the strongest open-source baseline. We also introduce ConfArena, an audience-oriented evaluation framework that simulates a conference room and assesses presentations slide by slide. ConfArena matches human system rankings and detects injected 
    
[^171]: JevOut：自然上下文可以颠覆决策模型

    JevOut: Natural Context Can Flip Decision Models

    [https://arxiv.org/abs/2609.30243](https://arxiv.org/abs/2609.30243)

    研究表明，决策模型（如Jev）对看似自然无害的简短上下文添加内容极其脆弱，优化后的上下文能在61.4%的情况下颠覆模型原本正确的决策，且在许多案例中使模型以至少0.7的高概率输出固定的错误选项。

    

    专门的决策模型（如Jev）能够将非结构化语言映射到有限选择的概率分布上，使其输出可以直接路由请求、选择工具并触发操作。然而，现实世界的输入很少是孤立到达的：它们通常伴随着背景细节和上下文环境。我们发现，即使正确答案保持不变，一些能够自然融入上下文的简短添加内容却可以改变原本正确的决策。为了研究这种行为，我们为每个最初回答正确的项目固定一个错误的目标选项，并利用模型的选项概率来优化流畅的上下文添加内容，同时保持源文本、问题、选项和标准答案不变。在64次被接受的目标评估中，优化器找到的上下文在508个最初正确的决策中成功改变了312个（61.4%）Jev的决策；在229个案例中，Jev为固定的错误选项分配了至少0.7的概率。跨七个数据集……（摘要被截断）

    arXiv:2609.30243v1 Announce Type: new  Abstract: Dedicated decision models such as Jev map unstructured language to probability distributions over finite choices, allowing their outputs to directly route requests, select tools, and trigger actions. Yet real-world inputs rarely arrive in isolation: they come with background details and surrounding context. We find that short additions that fit naturally into this context can nevertheless redirect an otherwise correct decision, even when the correct answer remains unchanged. To study this behavior, we fix a wrong target option for each initially correct item and use the model's option probabilities to refine fluent context additions while preserving the source, question, choices, and gold answer. Within 64 accepted target evaluations, the optimizer identifies contexts that redirect Jev on 312 of 508 initially correct decisions (61.4%); in 229 cases, Jev assigns at least 0.7 probability to the fixed wrong option. Across seven datasets, th
    
[^172]: ExplorationBench：在可验证的异星世界中衡量AI系统的探索能力

    ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds

    [https://arxiv.org/abs/2609.30199](https://arxiv.org/abs/2609.30199)

    提出ExplorationBench基准，利用规则可执行且与常识相冲突的“异星世界”沙盒（AlienCode与AlienLogic），实现了对AI系统科学探索能力的可验证评估，排除了仅凭记忆预训练知识解题的可能。

    

    科学发现始于已知问题终结之处。在那里，AI系统必须进行探索：提出假设、设计实验并对结果进行迭代。然而，评估这种能力十分困难：（1）如何验证一个真正新颖的假设是否成立，（2）如何判断系统是通过探索发现了它，还是仅仅从预训练数据中回忆了相关知识。为此，我们提出了ExplorationBench，它将评估科学探索这一棘手问题转化为一个建立在可验证“异星世界”之上的具体且易于处理的框架：这些世界的规则是可执行的，因此每个答案都可以被精确检验；同时它们与熟悉的知识相冲突，因此仅靠记忆无法解决任务。该基准包含两个沙盒：AlienCode（31个发现目标，70个任务）和AlienLogic（24个发现目标，70个任务）。每个沙盒都提供一份有缺陷的手册以及任务特定的环境……

    arXiv:2609.30199v1 Announce Type: new  Abstract: Scientific discovery begins where known problems end. There, AI systems must engage in exploration: framing hypotheses, designing experiments, and iterating on the results. However, evaluating this ability is difficult: (1) how to verify whether a genuinely new hypothesis holds, and (2) how to determine whether a system has discovered it through exploration or merely recalled related knowledge from pre-training data. To this end, we introduce ExplorationBench, which turns the wicked problem of evaluating scientific exploration into a concrete and tractable framework built on verifiable Alien Worlds: their rules are executable, so every answer can be checked exactly, and they conflict with familiar knowledge, so recall alone cannot solve the tasks. The benchmark contains two sandboxes, AlienCode (31 discovery targets, 70 tasks) and AlienLogic (24 discovery targets, 70 tasks). Each sandbox provides a flawed manual, task-specific environmen
    
[^173]: ELF-REG：将连续扩散语言模型扩展至推理任务

    ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks

    [https://arxiv.org/abs/2609.29102](https://arxiv.org/abs/2609.29102)

    提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。

    

    全连续扩散语言模型（dLMs）对连续表示进行去噪而无需中间离散化，并在最后一步并行解码所有响应token。它们在具有挑战性的推理任务上的性能表现，尚未像自回归（AR）大语言模型和掩码扩散语言模型那样得到充分验证。我们将嵌入式语言流（ELF）扩展到GSM8K、MATH-500、HumanEval和MBPP上的数学推理与代码生成任务。我们提出了ELF-REG，它通过表示对齐与纠缠（REPA+REG）来改进学习，其中冻结的AR教师模型监督中间去噪器特征，并提供一个与响应联合去噪的全局表示。ELF-REG-L在64次网络函数评估（NFE）下于GSM8K上达到55.96%的pass@1，在128次NFE下于MATH-500上达到13.39%、HumanEval上达到22.56%。它在GSM8K和代码任务上的pass@1优于所评估的同规模扩散语言模型，并将MATH-500的pass@1从……（摘要原文在此处被截断）

    arXiv:2609.29102v1 Announce Type: new  Abstract: Fully continuous diffusion language models (dLMs) denoise continuous representations without intermediate discretization, then decode all response tokens in parallel at the final step. Their performance on challenging reasoning tasks remains less established than that of autoregressive (AR) LLMs and masked dLMs. We scale Embedded Language Flows (ELF) to mathematical reasoning and code generation on GSM8K, MATH-500, HumanEval, and MBPP. We introduce ELF-REG, which improves learning with representation alignment and entanglement (REPA+REG), where a frozen AR teacher supervises intermediate denoiser features and supplies a global representation that is jointly denoised with the response. ELF-REG-L achieves 55.96% pass@1 on GSM8K at 64 network function evaluations (NFE), and 13.39% on MATH-500 and 22.56% on HumanEval at 128 NFE. It outperforms the evaluated comparable-scale dLMs in pass@1 on GSM8K and code, and improves MATH-500 pass@1 from 
    
[^174]: UniDataAgent：基于本体的企业“问题到报告”自动化智能体

    UniDataAgent: An Ontology-Grounded Agent for Enterprise Question-to-Report Automation

    [https://arxiv.org/abs/2609.27257](https://arxiv.org/abs/2609.27257)

    UniDataAgent通过将企业语义获取（构建版本化本体）与在线“问题到报告”执行分离，将原本需要约一周的人工本体构建缩短至数小时，并将报告生成缩短至几分钟。

    

    企业数据智能体必须保留组织特定的语义，而不仅仅是将问题翻译成查询。我们提出了中国联通数据智能体，这是一个基于本体的可复用“问题到报告”分析系统，它将语义获取与在线执行相分离。本体获取与验证阶段（OAV）通过专家编写的业务技能、受约束的生成、问题验证和精选的专家审核，从元数据、业务知识和支撑材料中构建版本化的企业本体。问题到报告执行阶段（QRE）为每个问题检索语义契约，协调技能与数据工具，验证结果，并生成带有证据链接的报告。在涵盖27个企业数据表和数千种指标类型的场景中，本体构建仅需几个小时，而人工构建约需一周时间；报告生成仅需几分钟，而以往则需要数个工作日。

    arXiv:2609.27257v1 Announce Type: new  Abstract: Enterprise data agents must preserve organization specific semantics, not just translate questions into queries. We present ChinaUnicom DataAgent (UniDataAgent), an ontology grounded system for reusable question-to-report analysis that separates semantic acquisition from online execution. Ontology Acquisition and Validation stage (OAV) builds versioned enterprise ontologies from metadata, business knowledge, and supporting materials through expert authored business skills, constrained generation, question verification, and selected expert review. Question-to-Report Execution (QRE) stage retrieves semantic contracts for each question, coordinates skills and data tools, validates results, and produces evidence linked reports. Across 27 enterprise tables and roughly thousands of metric types, ontology construction took a few hours instead of about one week manually. It took just a few minutes to generate the reports, instead of several work
    
[^175]: CTRL：基于控制的时间序列预测与LLM引导的残差学习

    CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning

    [https://arxiv.org/abs/2609.23257](https://arxiv.org/abs/2609.23257)

    CTRL框架将语义推理与定量预测解耦，利用LLM智能体作为控制器分析预测误差的分解成分并输出控制信号，再由轻量级残差解码器转化为预测修正，从而提升非平稳环境下时间序列预测的稳定性与可解释性。

    

    时间序列预测是跨多个领域关键决策的基础。尽管大语言模型（LLM）提供了有前景的推理能力，但现有的基于LLM的时间序列预测方法要么将其简化为绕过其优势的数值预测器，要么允许直接生成预测，导致在非平稳环境中预测不稳定。我们提出了CTRL，一个将语义推理与定量预测解耦的框架。冻结的主干模型生成基础预测，而专门的LLM智能体充当控制器，通过分解的趋势、季节性和不规则成分来分析主干预测误差，将推理建立在可解释的时间结构之上。每个智能体输出紧凑的控制信号，由轻量级残差解码器将其转化为预测修正。CTRL还集成了无标签的测试时自适应机制，可从输入统计信息中检测分布偏移。

    arXiv:2609.23257v1 Announce Type: cross  Abstract: Time series forecasting underpins critical decision-making across diverse domains. While large language models (LLMs) offer promising reasoning capabilities, existing LLM-based time series forecasting approaches either reduce them to numerical predictors that bypass their strengths, or allow direct forecast generation that destabilizes predictions in non-stationary settings. We introduce CTRL, a framework that decouples semantic reasoning from quantitative prediction. A frozen backbone generates base forecasts, while specialized LLM agents function as controllers that analyze backbone prediction errors through decomposed trend, seasonal, and irregular components, grounding reasoning in interpretable temporal structure. Each agent outputs compact control signals that a lightweight residual decoder translates into forecast corrections. CTRL incorporates label-free test-time adaptation that detects distribution shift from input statistics
    
[^176]: 用于人类模拟的预训练人格混合模型与串联模型

    Pretrained Persona Mixture Models and Tandem Models for Human Simulation

    [https://arxiv.org/abs/2609.22607](https://arxiv.org/abs/2609.22607)

    该论文提出“人格混合模型”，即使用预训练基础模型并借助特定人物的简短对话样本实现人格绑定，能比指令微调模型更准确地模拟人类，并保留更多人类对话的自然多样性。

    

    我们在此论证，当前大语言模型（LLM）人类模拟的主流做法——提示经过指令微调的助手语言模型扮演角色（人格）——是不准确的，并会产生刻板印象式的预测（缺乏自然的多样性）。此前已有研究表明，LLM可以通过自然的自由文本对话绑定到特定人格，从而避免刻板印象。本文进一步表明，仅使用特定人物的简短个体对话样本也可以实现这种人格绑定。人口统计学信息可在之后通过简单地查询模型添加，且不会产生负面影响。我们将“人格混合模型”用于指代校准良好的人类模型，目前以预训练基础模型的形式实现。我们证明，PMMs比指令微调模型产生更准确的预测，并保留了人类对话中更多的词汇、语义和语用多样性。我们在多样化的语料库集合上测量了模拟人类对话者的LLM的真实性与多样性。

    arXiv:2609.22607v1 Announce Type: new  Abstract: We argue here that the current dominant practice in LLM human simulation: prompting instruction-tuned assistant language models to role-play personas, is inaccurate and produces stereotyped predictions (lacking natural diversity). It has previously been shown that LLMs can be bound to personas using naturalistic, freetext dialog avoiding stereotyping. Here we show that binding can also be achieved using short, individual samples of dialog from specific people. Demographics can be added later without negative effects by simply querying the model. We use the term Persona Mixture Models (PMMs) for well-calibrated human models, currently realized as pretrained base models. We show that PMMs produce more accurate predictions than instruction-tuned models and retain more of the lexical, semantic, and pragmatic diversity found in human dialog. We measure realism and diversity of LLMs simulating human interlocutors across a diverse set of corpor
    
[^177]: 嵌入模型以奇特的方式进行测量

    Embedding Models Measure in Peculiar Ways

    [https://arxiv.org/abs/2609.20821](https://arxiv.org/abs/2609.20821)

    该研究发现嵌入模型对质量、距离、时间和体积等物理测量的表示十分微弱且奇特，主要受表面字符串相似性的强烈影响，而重新校准相似度也无法显著改善其与真实物理测量的对齐。

    

    嵌入空间定义了语义相似性和距离的概念。我们研究了这些嵌入是否反映了质量、距离、时间和体积等物理测量，这些物理测量具有唯一且客观的语义等价与距离概念。我们发现物理测量在嵌入空间中仅被微弱地建模，相反，可以观察到相当奇特的测量模式。进一步的分析表明，物理测量的嵌入表示受到表面字符串相似性的强烈影响，而重新校准相似性并不能实质性改善对齐效果。

    arXiv:2609.20821v1 Announce Type: new  Abstract: Embedding spaces define notions of semantic similarity and distance. We study whether those embeddings reflect physical measurements of mass, distance, time and volume, which admit a unique, objective notion of semantic equivalence and distance. We find that physical measurement is only weakly modeled in the embedding space, and that instead quite peculiar measurement patterns can be observed. Further analysis indicates that embedding representations of physical measurements are strongly influenced by superficial string similarity, and recalibration of similarity does not substantially improve the alignment.
    
[^178]: EconSkills：研究Web智能体在实时经济数据上的技能迁移与检索

    EconSkills: Studying Skill Transfer and Retrieval for Web Agents on Live Economic Data

    [https://arxiv.org/abs/2609.19523](https://arxiv.org/abs/2609.19523)

    EconSkills框架将验证过的经济数据检索轨迹提炼成参数化技能库，证明技能迁移和基于库的检索能显著提升Web智能体的表现。

    

    Web智能体经常需要重新访问相同的网站，然而大多数评估方法会丢弃在早期成功交互中学到的操作流程。我们提出了EconSkills，这是一个技能库和评估框架，它将经过验证的EconWebArena轨迹提炼成参数化的标准操作流程，用于检索实时经济数据。每个技能记录其适用范围、导航流程、特定网站的指导、验证检查和恢复步骤，同时用占位符替换原始实例的数值。EconSkills将两个问题分开：已知的相关流程是否能迁移到保留任务上，以及当智能体从技能库中选择时能否保持这种优势。在受控迁移实验中，匹配的技能比无技能提示提高了成功率，并且在配对成功案例中所需的步骤更少，而抽象化方法比重放原始轨迹要有效得多。在技能库规模下，检索方法与无技能基线相比具有竞争力。

    arXiv:2609.19523v1 Announce Type: new  Abstract: Web agents often revisit the same sites, yet most evaluations discard the procedures learned in earlier successful interactions. We introduce EconSkills, a skill library and evaluation framework that distills verified EconWebArena trajectories into parameterized standard operating procedures for retrieving live economic data. Each skill records its scope, navigation procedure, site-specific guidance, verification checks, and recovery steps while replacing source-instance values with placeholders. EconSkills separates two questions: whether a known relevant procedure transfers to a held-out task, and whether an agent can retain that benefit when selecting from a library. In controlled transfer, matched skills improve success over no-skill prompting and require fewer steps on paired successes, while abstraction is substantially more effective than replaying raw trajectories. At library scale, retrieval is competitive with the no-skill base
    
[^179]: 鸡尾酒会场景下的视听话轮转换预测

    Audio-Visual Turn-taking Prediction in Cocktail Party Scenarios

    [https://arxiv.org/abs/2609.17056](https://arxiv.org/abs/2609.17056)

    在鸡尾酒会等嘈杂含重叠语音的场景中，用干净数据训练的视听话轮转换预测模型性能显著下降（加权F1相对下降高达38%），微调虽可提升鲁棒性但收益因模态和预训练数据规模而异，凸显了鲁棒建模的必要性。

    

    当前的预测性话轮转换模型（PTTM）在声学条件可控、音频信号干净的基准测试中取得了优异的表现。然而，这些模型在有重叠语音和背景干扰的对话中的泛化能力仍未得到充分探索。在本研究中，我们在一个源自AVCocktail数据集、具有挑战性的鸡尾酒会测试平台上，评估了使用干净数据训练的视听预测性话轮转换模型，并分析了它们对该新领域的适应行为。实验结果表明，在噪声条件下，音频和视觉模态的性能均出现一致的下降，加权F1值的相对下降高达38%。在新领域上进行微调可以提升鲁棒性，但收益因模态而异，且取决于可用的预训练数据规模。这些发现为了解音频和视觉模态在泛化和适应能力上的差异提供了见解，并表明需要更鲁棒的建模方法。

    arXiv:2609.17056v1 Announce Type: cross  Abstract: Current predictive turn-taking models (PTTMs) achieve strong performance on benchmarks with controlled acoustic conditions and clean audio signals. Their generalisation to conversations with overlapping speech and background interference remains underexplored. In this research, we evaluate audio-visual PTTMs trained with clean data on a challenging cocktail-party testbed derived from the AVCocktail dataset, and analyse their adaptation behaviour to this new domain. Experimental results show consistent performance degradation across audio and visual modalities under noisy conditions, with up to 38% relative drop in weighted F1. Fine-tuning on the new domain improves robustness, but gains vary across modalities and depend on the size of the available pre-training data. These findings provide insights into the different generalisation and adaptation capabilities of the audio and visual modalities, and indicate the need for robust modellin
    
[^180]: Salesforce Koa：一个面向智能体工具使用的企业级语言模型

    Salesforce Koa: An Enterprise Language Model for Agentic Tool Use

    [https://arxiv.org/abs/2609.15066](https://arxiv.org/abs/2609.15066)

    Salesforce Koa 是一个基于 Nemotron-3-Super-120B 并通过 GRPO 强化学习后训练的企业级语言模型，其核心创新在于“仿真到奖励”流水线——将工作流规范扩展为以角色为条件的多轮任务，并以成功工具使用作为任务解决奖励，从而在保持通用性能的同时显著提升智能体工具使用能力。

    

    我们介绍了 Salesforce Koa，一个通过对开放权重的 Nemotron-3-Super-120B 基础模型进行后训练，并采用基于群体相对策略优化（GRPO）的强化学习而构建的企业级语言模型。Salesforce Koa 在公开数据和合成生成的数据上训练，不使用任何客户数据，旨在提升工具使用和智能体能力，同时保持强大的通用性能。其独特的组成部分是一个“仿真到奖励”流水线，该流水线将工作流规范扩展为以角色为条件的多轮任务，并以成功的工具使用为基础，为数据依赖型请求提供任务解决奖励。在企业领域，这些规范采用 Agent Script 编写，即 Salesforce 用于构建 Agentforce 智能体的声明式语言；而在公共工具使用领域，我们直接合成工作流结构。相同的仿真和基于真实结果的奖励机制驱动 GRPO 在两类场景中运行。在公共工具使用、智能体……

    arXiv:2609.15066v1 Announce Type: cross  Abstract: We present Salesforce Koa, an enterprise language model built by post-training the open-weight Nemotron-3-Super-120B foundation model with reinforcement learning using Group Relative Policy Optimization (GRPO). Salesforce Koa is trained on public and synthetically generated data, with no customer data, to improve tool use and agentic capabilities while preserving strong general-purpose performance. Its distinctive component is a simulation-to-reward pipeline that expands workflow specifications into persona-conditioned multi-turn tasks with task-resolution rewards grounded in successful tool use for data-dependent requests. For enterprise domains, these specifications are written in Agent Script, Salesforce's declarative language for building Agentforce agents; for public tool-use domains, we synthesize the workflow structure directly. The same simulation and grounded-reward machinery drives GRPO across both. Across public tool-use, ag
    
[^181]: Meddies-PII：一个用于临床去标识化中个人身份信息提取的多语言框架

    Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification

    [https://arxiv.org/abs/2609.12544](https://arxiv.org/abs/2609.12544)

    该论文提出了Meddies-PII框架，包含一个通过属性条件提示生成、经十三个确定性门控验证、涵盖十七种语言的一百万份合成临床文档数据集，以及基于该数据集训练的BIOES分类模型，后者在十五个外部基准上以平均F1 0.827达到了现有PII提取系统中的最高性能。

    

    临床去标识化依赖于准确识别个人身份信息（PII）。然而，人工标注的数据集构建成本高昂，而现有的合成替代方案通常对其生成过程提供的细节有限，或依赖相对简单的合成策略。我们推出了Meddies-PII数据集，这是一个包含一百万份合成临床文档的语料库，涵盖十七种语言和九种PII标签。这些文档采用属性条件提示生成，并通过十三个确定性门控进行验证，以确保结构和标注的一致性。为评估该数据集的实用性，我们训练了Meddies-PII模型——一个BIOES标记分类器，并使用精确匹配的实体级F1分数将其与现有的PII提取系统进行比较。Meddies-PII模型在所有报告的基准测试中均取得了所评估系统中的最高性能，在十五个外部基准上的平均F1为0.827。

    arXiv:2609.12544v1 Announce Type: new  Abstract: Clinical de-identification relies on accurately identifying personally identifiable information (PII). However, manually annotated datasets are costly to construct, while existing synthetic alternatives often provide limited details about their generation process or rely on relatively simple synthesis strategies. We introduce Meddies-PII-Dataset, a corpus of one million synthetic clinical documents spanning seventeen languages and nine PII labels. The documents are generated using attribute-conditioned prompts and validated through thirteen deterministic gates that enforce structural and annotation consistency. To evaluate the dataset's utility, we train Meddies-PII-Model, a BIOES token classifier, and compare it with existing PII extraction systems using exact-match entity-level F1. Meddies-PII-Model achieves the highest performance among the evaluated systems on all reported benchmarks, with a mean F1 of 0.827 across fifteen external b
    
[^182]: 真理从未消失：顺从情境下真值探测器的完美混叠现象

    The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes

    [https://arxiv.org/abs/2609.10739](https://arxiv.org/abs/2609.10739)

    论文揭示了真值探测器的“完美混叠”失效机制——当真实汇报与任务既定行为在顺从情境中重合时探测器无法区分二者（二者AUROC恒互补求和为一），并提出在顺从与对立情境的混合数据上拟合的方法，使探测器即使在模型系统性说谎时也能以完美的1.0 AUROC识别真值。

    

    在真实汇报与任务既定行为相重合的情境中拟合的真值探测器，仅凭其拟合标签无法区分这两个目标。我们将这种语义识别的失效称为“完美混叠”（perfect aliasing）。在一个受控的二值汇报博弈中，在顺从情境上拟合的真值探测器与既定行为探测器求解的是同一个优化问题。在对立情境上，二者的标签互为补集，导致它们的AUROC之和恒为一；这一恒等关系在751个单元-层对上以浮点精度成立。我们利用随机化码本将既定输出符号与语义行为分离，进而通过在顺从情境与对立情境的混合数据上拟合，将真值与既定行为区分开来。对于一个经奖励训练、在所有评估的对立试验中均给出虚假回答的Gemma-2-9B策略，常规探测器在三个训练种子上的AUROC仅为0.006 ± 0.005，而混合拟合探测器在相同的保留激活值上得分达到1.000。混合拟合……

    arXiv:2609.10739v1 Announce Type: cross  Abstract: A truth probe fitted where truthful reporting and a task's prescribed action coincide cannot distinguish those targets from its fitting labels alone. We call this failure of semantic identification perfect aliasing. In a controlled binary reporting game, truth and prescribed-action probes fitted on compliant contexts solve the same optimization. On rival contexts their labels are complements, forcing their AUROCs to sum to one; this identity holds across 751 cell-layer pairs to floating-point precision. We separate prescribed output symbols from semantic action using randomized codebooks, then separate truth from prescribed action by fitting on mixed compliant and rival contexts. For a reward-trained Gemma-2-9B policy that answers falsely on all evaluated rival trials, the conventional probe scores $0.006 \pm 0.005$ AUROC across three training seeds, while mixed-fit probes score $1.000$ on the same held-out activations. Mixed fitting u
    
[^183]: 从边缘到联合的门票：扩散语言模型中一步式块生成的耦合噪声蒸馏

    A Ticket from Marginals to Joints: Coupled-Noise Distillation for One-Step Block Generation in Diffusion Language Models

    [https://arxiv.org/abs/2609.06324](https://arxiv.org/abs/2609.06324)

    提出CONDOR方法，通过耦合噪声蒸馏从零训练扩散语言模型，使其在掩码嵌入受高斯噪声扰动时仅用一次前向传播即可生成连贯的整块文本，且无需目标侧编码器或自回归教师。

    

    扩散语言模型并行预测一个块中的所有词元，但单次前向传播是从各自的边缘分布中采样每个位置，因此这些词元不一定能构成一个连贯的块。我们提出一个问题：当离散掩码模型的掩码嵌入受到采样高斯噪声场扰动时，该模型能否在单次传播中生成整个块——相同的噪声应给出相同的连贯续写，而不同的噪声应给出不同的续写。我们提出了CONDOR（Coupled-Noise Distillation for One-Step Readout，面向一步式读出的耦合噪声蒸馏），它无需目标侧编码器或自回归教师即可从头训练这样的模型。训练结合了两种信号。在真实文本上，模型在多个噪声样本下预测被掩码的词元，并仅通过与真实值最匹配的那个样本进行监督，从而使不同的噪声可以专精于不同的续写。对于其余样本，模型则细化自身的单次传播预测……

    arXiv:2609.06324v2 Announce Type: replace  Abstract: Diffusion language models (dLLMs) predict all tokens of a block in parallel, but a single forward pass samples each position from its own marginal distribution, so the tokens need not form a coherent block. We ask whether a discrete masked model can commit an entire block in one pass when its mask embeddings are perturbed by a sampled Gaussian noise field: the same noise should give the same coherent continuation, and different noise should give different ones. We propose CONDOR (Coupled-Noise Distillation for One-Step Readout), which trains such a model from scratch without a target-side encoder or an autoregressive teacher. Training combines two signals. On real text, the model predicts masked tokens under several noise samples and is supervised only through the sample that fits the ground truth best, so different noise can specialize to different continuations. For the remaining samples, the model refines its own one-pass predicti
    
[^184]: 基于家族DIF指导的基准重组下，接近持平的大语言模型排名是否稳健？

    Are Near-Tied LLM Rankings Robust to Family-DIF-Guided Benchmark Recomposition?

    [https://arxiv.org/abs/2609.00482](https://arxiv.org/abs/2609.00482)

    该论文提出一种基于无家族标签谱近似MIRT的基准重组方法，发现尽管全基准与低DIF排名强相关，但相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，表明排行榜上的微小差距并不稳健。

    

    排行榜上的微小差距常被解读为某个语言模型优于另一个的证据，但其结论方向可能取决于包含哪些基准题目。我们利用五个基准的题目级响应数据以及一种无家族标签的谱近似多维项目反应理论（MIRT）来检验这一点。在所有者不相交的折中划分下，一半所有者数据用于识别跨模型家族具有低残差差异项目功能（低DIF）的题目；由此得到的固定且按来源和难度平衡的权重用于对另一半数据中的模型进行评分，同时使用等长的匹配随机子测试来控制一般性的子测试变异。全基准排名与低DIF排名保持强相关（τb=.900-.948）。然而，在五个基准中的四个里，最初相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，比匹配随机子测试的中位数高出16.9-28.6个百分点（均为p=.001）。第五个基准[摘要截断]

    arXiv:2609.00482v1 Announce Type: new  Abstract: Small leaderboard gaps are often interpreted as evidence that one language model is better than another, but their sign may depend on which benchmark items are included. We test this using item-level responses from five benchmarks and a family-label-free spectral approximation to multidimensional item-response theory (MIRT). In owner-disjoint folds, one owner half identifies items with low residual differential item functioning across model families (low-DIF); the resulting frozen, source- and easiness-balanced weights score models in the other half, while equally short matched-random subtests control for generic subtest variation. Full-benchmark and low-DIF rankings remain strongly correlated ($\tau_b=.900$--$.948$). Yet in four of five benchmarks, 30.9--47.1\% of cross-family pairs initially within one percentage point reverse order, exceeding their matched-random medians by 16.9--28.6 percentage points (all $p=.001$). The fifth benchm
    
[^185]: 面向大语言模型的CPR：领域自适应中基于临界点路由的灾难性遗忘防御方法

    CPR for LLMs: Critical-Point Routing against Catastrophic Forgetting in Domain Adaptation

    [https://arxiv.org/abs/2608.30158](https://arxiv.org/abs/2608.30158)

    本文提出CPR框架，通过识别基座模型失败而专家模型成功的临界词元，在基座模型与其SFT专家版本之间进行词元级路由，从模型层面解耦领域能力与通用能力，有效缓解领域自适应中的灾难性遗忘。

    

    监督微调（SFT）是将大语言模型（LLM）适配到目标领域的实际标准方法，但它常常会降低模型的通用能力，这一现象被称为“灾难性遗忘”。现有方法通常通过修改SFT损失来缓解遗忘问题，但它们不可避免地运行在领域能力与通用能力的权衡之中。在本工作中，我们通过在模型层面解耦这两种能力来跳出这一权衡：保留原始基座模型以维持通用能力，仅在需要领域特定知识时选择性地调用SFT专家模型。具体而言，我们提出了CPR（临界点路由），这是一个在基座模型与其专家衍生模型之间进行词元级路由的框架，其依据是那些基座模型失败而专家模型成功的“临界词元”。我们训练了一个轻量级的分层路由器来估计每个词元调用专家的概率，并将其与精心设计的推理策略相配……（原文摘要于此处截断）

    arXiv:2608.30158v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) is the de facto standard for adapting large language models (LLMs) to target domains, but it often degrades the model's general capabilities, a phenomenon known as catastrophic forgetting. Existing approaches typically modify the SFT loss to mitigate forgetting, but they inevitably operate along a domain-generality trade-off. In this work, we step outside this trade-off by decoupling the two capabilities at the model level: we keep the original base model for general capability, and selectively invoke the SFT expert only when domain-specific knowledge is required. Specifically, we propose CPR (Critical-Point Routing), a token-level routing framework between a base model and its expert derivative, based on critical tokens where the base model fails but the expert succeeds. We train a lightweight hierarchical router that estimates the expert-call probability per token, and pair it with a tailored inference pr
    
[^186]: Gaokerena：一个小型波斯语医学语言模型家族

    Gaokerena: A Small Persian Medical Language Model Family

    [https://arxiv.org/abs/2608.00932](https://arxiv.org/abs/2608.00932)

    本文提出了Gaokerena，一个专为消费级硬件设计的小型波斯语医学语言模型家族，其中Gaokerena-V通过新构建的波斯语医学语料库训练提升了医学问答性能，Gaokerena-R则结合思维链与两个新型RLAIF框架来增强临床推理能力。

    

    人工智能融入医学问答系统的发展迅速；然而，相关研究仍主要集中在英语上，导致波斯语等低资源语言的服务严重不足。为填补这一空白，本文提出了Gaokerena，这是一个新型的小型波斯语医学语言模型家族，专为在消费级硬件上部署而优化。作为迈向本地化数字医疗的基础步骤，我们首先介绍了Gaokerena-V，它是通过在一个新构建的9000万词元波斯语医学语料库和2万个经专家审核的医生问答对上训练基线模型而开发的，其在翻译版医学MMLU基准上的性能从46.28%提升至49.31%。其次，考虑到临床推理的关键需求，我们通过将思维链方法与两个新颖的AI反馈强化学习（RLAIF）框架相结合，开发了Gaokerena-R，以优化偏好……

    arXiv:2608.00932v2 Announce Type: replace  Abstract: The integration of artificial intelligence into medical question-answering systems has advanced rapidly; however, research remains predominantly focused on English, leaving low resource languages like Persian significantly underserved. To address this gap, this paper introduces Gaokerena, a novel family of compact Persian medical language models optimized for deployment on consumer grade hardware. As a foundational step toward localized digital healthcare, we first present Gaokerena-V, developed by training a baseline model on a newly curated 90-million-token Persian medical corpus and 20,000 expert-vetted physician Q&A pairs, which improved performance on a translated medical MMLU benchmark from 46.28% to 49.31%. Second, recognizing the critical demands of clinical reasoning, we developed Gaokerena-R by integrating a Chain-of-Thought approach with two novel Reinforcement Learning with AI Feedback (RLAIF) frameworks to optimize prefe
    
[^187]: SERL-SQL：面向Text-to-SQL强化智能体学习的选择性后见之明蒸馏

    SERL-SQL: Selective Hindsight Distillation for Text-to-SQL Reinforcement Agentic Learning

    [https://arxiv.org/abs/2608.00485](https://arxiv.org/abs/2608.00485)

    SERL-SQL提出了一种以执行反馈为依据的选择性强化学习框架，通过仅在训练阶段使用的教师模型的后见之明重评分，将教师-学生似然差距转化为对GRPO优势的局部重加权，从而为多轮Text-to-SQL智能体实现更精细的信用分配。

    

    最近的Text-to-SQL系统越来越依赖多轮交互、执行反馈和强化学习。然而，大多数现有方法仅将执行正确性作为轨迹级别的奖励，这为识别导致成功或失败的具体SQL决策提供的指导有限。我们提出了SERL-SQL，一个面向多轮Text-to-SQL智能体的选择性、以执行为依据的强化学习框架。SERL-SQL采样在线策略的SQL交互轨迹，并使用一个仅在训练阶段启用的教师模型，基于执行反馈对学生模型的动作进行重新评分。由此产生的教师-学生似然差距被转换为有界的、带掩码的权重，仅在SQL和工具动作token上对GRPO优势进行重新加权。通过这种方式，任务奖励保持优化方向，而执行后见之明提供局部化的信用分配。在BIRD、Spider和跨领域基准上的实验表明，SERL-SQL取得了具有竞争力的性能（原文摘要在此处截断）。

    arXiv:2608.00485v4 Announce Type: replace  Abstract: Recent Text-to-SQL systems increasingly rely on multi-turn interaction, execution feedback, and reinforcement learning. However, most existing methods use execution correctness only as a trajectory-level reward, which provides limited guidance for identifying the SQL decisions responsible for success or failure. We propose SERL-SQL, a selective execution-grounded reinforcement learning framework for multi-turn Text-to-SQL agents. SERL-SQL samples on-policy SQL interaction trajectories and uses a training-only teacher to re-score student actions with execution feedback. The resulting teacher--student likelihood gap is converted into bounded, masked weights that reweight GRPO advantages only on SQL and tool-action tokens. In this way, task rewards preserve the optimization direction, while execution hindsight provides localized credit assignment. Experiments on BIRD, Spider, and cross-domain benchmarks show that SERL-SQL achieves compe
    
[^188]: 文本到视觉的迁移是什么？视觉语言模型的能力缩放规律与迁移动态

    What Transfers from Text to Vision? Capability Scaling Laws and Transfer Dynamics for VLMs

    [https://arxiv.org/abs/2608.00013](https://arxiv.org/abs/2608.00013)

    我们提出了首个跨家族的多模态缩放规律，通过文本能力得分直接预测VLM性能，并在150多个VLM上验证了其有效性。

    

    arXiv:2608.00013v2 公告类型：替换-交叉 摘要：选择合适的大型语言模型（LLM）骨干是构建视觉语言模型（VLM）时最关键的决策，但这一过程从根本上缺乏原则性：基于计算量的缩放规律无法跨模型家族泛化，且不存在在训练开始前直接预测VLM性能的框架。我们提出了能力驱动的多模态缩放规律，这是首个跨家族框架，能够从直接可观测的文本能力预测VLM基准准确率。给定通过主成分分析从LLM文本基准中提取的低维能力得分$S$，我们将VLM性能建模为$S$的函数，并引入每个骨干的迁移率和量化数据缩放效率的吸收率。为拟合和验证该框架，我们在严格控制的配方下，基于34个LLM（涵盖7个模型家族）训练了超过150个VLM。在超过200个文本基准和50个多模态基准上的评估表明...

    arXiv:2608.00013v2 Announce Type: replace-cross  Abstract: Choosing the right large language model (LLM) backbone is the most consequential decision when building a vision-language model (VLM), yet it remains fundamentally unprincipled: compute-based scaling laws fail to generalize across model families, and no framework exists for directly predicting VLM performance before training begins. We propose the Capability-Driven Multimodal Scaling Law, the first cross-family framework that predicts VLM benchmark accuracy from directly observable textual capability. Given a low-dimensional capability score $S$ extracted from LLM textual benchmarks via PCA, we model VLM performance as a function of $S$, with a per-backbone transfer rate and an absorption rate that quantifies data-scaling efficiency. To fit and validate the framework, we train over 150 VLMs on 34 LLMs spanning 7 model families under a strictly controlled recipe. Evaluations on more than 200 textual and 50 multimodal benchmarks 
    
[^189]: GLM-RAG：用于基于图的检索增强生成的图语言模型

    GLM-RAG: Graph Language Models for Graph-Based Retrieval-Augmented Generation

    [https://arxiv.org/abs/2607.28397](https://arxiv.org/abs/2607.28397)

    本文提出了一种基于图语言模型（GLM）的检索器用于知识图谱检索增强生成，发现微调后的GLM检索器在跨领域泛化能力上优于GNN和向量搜索检索器，并在两个多跳基准上达到SOTA。

    

    基于知识图谱的检索增强生成（RAG）需要能够有效捕获图结构和语义信息的检索器。近期的方法探索了基于图神经网络（GNN）的检索器，以在多跳推理任务中建模图拓扑结构。与此同时，图语言模型（GLM）作为一种融合图推理能力与语言模型语义能力的新兴范式应运而生。在本工作中，我们提出了一种基于GLM的检索器，并研究了基于GLM的检索器、基于GNN的检索器以及传统基于向量搜索的检索器在单跳和多跳RAG设置中的比较优势，并特别关注其在未见领域上的可迁移性。我们的研究结果表明，经过微调的GLM检索器在领域外具有更好的泛化能力，在两个多跳基准测试上达到了最先进水平（SOTA）。在领域内的多跳问答数据集上，它们仍然与先前的工作相当，并且随着参数（规模增长）展现出可喜的扩展性。

    arXiv:2607.28397v2 Announce Type: replace  Abstract: Retrieval-augmented generation (RAG) over knowledge graphs requires retrievers that can effectively capture both graph structure and semantic information. Recent approaches have explored graph neural network (GNN)-based retrievers to model graph topology in multi-hop reasoning tasks. In parallel, graph language models (GLMs) have emerged as a promising paradigm that integrates graph reasoning and the semantic capabilities of language models. In this work, we introduce a GLM-based retriever and investigate the comparative strengths of GLM-based, GNN-based, and traditional vector-search-based retrievers in single- and multi-hop RAG settings, and with a particular focus on transferability to unseen domains. Our findings suggest that finetuned GLM retrievers generalize better out of domain, achieving SOTA on two multi-hop benchmarks. On in-domain multi-hop QA datasets they remain comparable to prior work, with promising scaling as parame
    
[^190]: 从零开始扩展原生多模态预训练

    Scaling Native Multimodal Pre-Training From Scratch

    [https://arxiv.org/abs/2607.22043](https://arxiv.org/abs/2607.22043)

    该研究首次系统刻画了原生多模态预训练的扩展规律，发现最小损失遵循可预测的计算定律，计算最优的模型规模与 token 数量呈幂律扩展，且语言与多模态目标的计算资源分配趋势截然不同。

    

    尽管大型语言模型（LLMs）展现出卓越的推理能力，但其对纯文本预训练的依赖限制了对多模态物理世界的感知。原生多模态预训练通过在多模态输入上从头训练模型来规避这一局限，从而实现深度的跨模态融合，并缓解传统后期融合（late-fusion）架构固有的优化不对称问题。尽管具有这些优势，该范式的扩展特性仍未得到充分刻画。为填补这一空白，我们研究了在固定计算预算下训练基于 Transformer 的视觉-语言模型的最优模型规模与 token 数量。我们的研究表明，最小目标损失遵循可预测的计算定律，而计算最优的模型规模和 token 数量则呈幂律扩展。值得注意的是，语言目标与多模态目标呈现出不同的计算分配趋势。

    arXiv:2607.22043v2 Announce Type: replace  Abstract: Although large language models (LLMs) exhibit remarkable reasoning capabilities, their reliance on text-only pre-training restricts the perception of the multimodal physical world. Native multimodal pre-training avoids this limitation by training models from scratch on multimodal inputs, thereby achieving deep cross-modal integration and mitigating optimization asymmetries inherent to traditional late-fusion architectures. Despite these advantages, the scaling properties of this paradigm remain incompletely characterized. To address this gap, we investigate the optimal model size and token count for training a Transformer-based vision-language model under a fixed computational budget. Our study demonstrates that minimal objective loss adheres to a predictable compute law, whereas compute-optimal model sizes and token counts scale as power laws. Notably, language and multimodal objectives manifest distinct allocation trends. The langu
    
[^191]: 形式语义结构能解释多少人类标签变异？：自然语言推理中的群体层面效应与条目层面天花板

    How Much Human Label Variation Does Formal Semantic Structure Explain?: Group-Level Effects and Item-Level Ceilings in NLI

    [https://arxiv.org/abs/2607.15870](https://arxiv.org/abs/2607.15870)

    该研究通过预注册分析直接测量发现，形式语义结构对自然语言推理中人类标签变异的解释力有限——群体层面上非纯向上单调假设的标签熵显著更高，但条目层面上形式语义特征仅能解释3.3%–3.6%的熵方差。

    

    自然语言推理中的人类标签变异日益被视为信号而非噪声，但形式语义结构到底能解释其中多少，此前尚未被直接测量过。我们在ChaosNLI的3,113个SNLI和MNLI条目上对此进行测量，使用了经MED验证的基于规则的算子与单调性标注器（在编辑位置上的一致性为0.883，在分析所使用的句子级摘要上为0.807）、三个预注册的分析模块，并完整报告了阴性结果。研究得出三个界限。第一，群体层面的边界：非纯粹向上单调的假设显示出可靠更高的标签熵（Cliff's delta = -0.284），基于秩的检验表明该效应在控制算子存在与长度等缩减因素后依然稳健，但有界结果敏感性检验削弱了长度防御的回归形式。第二，条目层面的天花板：同样的形式语义特征仅解释了熵方差的3.3%至3.6%（原文摘要在此处截断）。

    arXiv:2607.15870v2 Announce Type: replace  Abstract: Human label variation in natural language inference is increasingly treated as signal rather than noise, but how much of it formal semantic structure explains has not been measured directly. We measure it on the 3,113 SNLI and MNLI items of ChaosNLI, using a rule-based operator and monotonicity tagger validated against MED (0.883 agreement at the edit site, 0.807 on the sentence-level summary our analyses consume), three preregistered analysis blocks, and full reporting of negative results. Three bounds emerge. First, a group-level boundary: hypotheses that are not purely upward monotone show reliably higher label entropy (Cliff's delta = -0.284), and rank-based tests defend the effect against operator-presence and length reductions, though a bounded-outcome sensitivity check weakens the regression form of the length defense. Second, an item-level ceiling: the same formal profiles explain only 3.3 to 3.6 percent of entropy variance a
    
[^192]: 单词普查：44个语言模型中的答案选择趋同性

    The One-Word Census: Answer-Choice Conformity Across 44 Language Models

    [https://arxiv.org/abs/2607.12796](https://arxiv.org/abs/2607.12796)

    本研究通过96个开放式单词选择提示对105个语言模型进行了“答案趋同性”普查，发现模型的选择高度趋同——例如46%的模型在“选一个词”时都选了"serendipity"——且这种趋同并非由小型模型或个性化微调模型所致，反而轻度后训练和个性化微调的模型分歧最大。

    

    当一个语言模型必须从大量同样有效的选项中选择一个答案时，它会选择哪个答案，又有多少时候它与所有其他模型的选择一致？当被要求“选一个词”时，来自二十多个实验室的105个语言模型有46%的概率选择了“serendipity”这个词。我们通过96个单轮提示来测量这种趋同性以及每个模型在其中的贡献，每个提示指定一个包含许多有效单词答案的类别（如“说出一种树的名字。”），每个模型被询问八次，并以精确匹配进行评分，不使用嵌入模型，也不使用评判模型。模型的“答案选择惊异度”定义为其答案在所有其他模型汇总答案下的平均 -log2 概率。在96个类别中，有28个类别里单个答案占据了所有答案的至少80%。这种集中度并非源于小型模型或个性化微调模型：87个主要实验室的模型的集中程度至少与整个领域相当。轻度后训练和个性化微调的模型的分歧最大（原文在此处截断）。

    arXiv:2607.12796v3 Announce Type: replace-cross  Abstract: When a language model must choose one answer from a large space of equally valid options, which answer does it choose, and how often is it the answer every other model chooses? Asked to "pick a word," 105 language models from more than twenty labs chose serendipity 46% of the time. We measure this convergence, and each model's share in it, with 96 single-turn prompts that each name a category with many valid one-word answers ("Name a tree."), asked eight times per model and scored by exact match, with no embeddings and no judge. A model's answer-choice surprisal is the average -log2 probability of its answers under the pooled answers of all other models. In 28 of 96 categories a single answer takes at least 80% of all answers. The concentration does not depend on small or persona-tuned models: the 87 major-lab models are at least as concentrated as the full field. Lightly post-trained and persona-tuned models are the most diver
    
[^193]: 将以人为本理念引入AI辅助词典编纂

    Introducing Human-Centeredness in AI-Assisted Lexicography

    [https://arxiv.org/abs/2607.11808](https://arxiv.org/abs/2607.11808)

    本文提出了一个以人为本的人工智能（HCAI）框架，主张AI应增强而非取代词典编纂者，并从增强型词典编纂者、社会技术背景、偏见和工具设计四个维度审视AI在词典编纂中的整合。

    

    本文提出了一个面向AI辅助词典编纂的以人为本的人工智能（HCAI）框架。虽然生成式人工智能为提升词典编纂工作提供了重要机遇，但也引发了关于词典编纂者未来角色以及语言与文化多样性保护的担忧。本文借鉴HCAI原则及其在其他语言行业中的既有应用，识别出四个相互关联的维度，用以理解和批判性审视人工智能在词典编纂中的整合：增强型词典编纂者、AI整合的社会技术背景、偏见问题，以及AI驱动的词典编纂工具的设计。该框架主张，人工智能应当增强而非取代词典编纂者，将自动化与有意义的人类控制相结合。文章进一步强调了保护专业自主性、缓解AI产生的偏见，以及围绕用户需求设计工具的重要性。

    arXiv:2607.11808v4 Announce Type: replace-cross  Abstract: This paper proposes a human-centered artificial intelligence (HCAI) framework for AI-assisted lexicography. While generative AI offers significant opportunities to enhance lexicographic work, it also raises concerns regarding the future role of lexicographers and the preservation of linguistic and cultural diversity. Drawing on HCAI principles and previous applications in other language professions, the paper identifies four interrelated dimensions through which AI integration in lexicography can be understood and critically examined: the augmented lexicographer, the sociotechnical context of AI integration, bias, and the design of AI-powered lexicographic tools. The framework argues that AI should augment rather than replace lexicographers, combining automation with meaningful human control. It further emphasizes the importance of preserving professional agency, mitigating AI-generated biases, and designing tools around the ne
    
[^194]: 系统提示词条件化与四个开源权重模型中的隐藏状态几何：勘误与留存结论

    System-Prompt Conditioning and Hidden-State Geometry in Four Open-Weight Models: Corrections and What Survives

    [https://arxiv.org/abs/2607.09842](https://arxiv.org/abs/2607.09842)

    本文是对先前“系统提示词会在开源语言模型隐藏状态中留下几何指纹”这一研究的勘误版：审计发现原论文的曲率统计量、置换检验方法和若干关键测量均存在错误，本版修正这些问题并说明哪些结论仍然成立。

    

    本预印本的第1版和第2版曾报告称，指定身份的系统提示词会在四个开源权重语言模型的最后一层隐藏状态轨迹中留下几何指纹，并且指令微调会将该指纹从隐藏状态向量的方向转移到其幅值上。对其代码和数据的审计发现了以下问题。论文中描述为欧氏k-近邻图上Ollivier-Ricci曲率的统计量，实际上是由时间k-近邻边和余弦k-近邻边构建的图上的一种非标准Forman型边统计量；其发布的检验对合并后的边（而非轨迹）进行置换，且已发表的p值来自未公开的代码。论文中报告为首个生成状态范数的量，实际上是最后一个提示词位置的状态，首个输出token正是由该状态预测得出。通用控制提示词与身份提示词是在字符数上匹配，而非在token数上匹配。本版本勘正了

    arXiv:2607.09842v3 Announce Type: replace-cross  Abstract: Versions 1 and 2 of this preprint reported that an identity-specifying system prompt leaves a geometric fingerprint in the final-layer hidden-state trajectories of four open-weight language models, and that instruction tuning moves this fingerprint from the direction to the magnitude of the hidden-state vector. An audit of their code and data found the following. The curvature statistic described as Ollivier-Ricci curvature on Euclidean k-NN graphs was a non-standard Forman-type edge statistic on graphs built from temporal and cosine k-NN edges. Its released test permuted pooled edges instead of trajectories, and the published p-values came from unreleased code. The quantity reported as the norm of the first generated state is the state at the last prompt position, from which the first output token is predicted. The generic control prompt was matched to the identity prompt in characters, not in tokens. This version corrects the
    
[^195]: 面向对话式大语言模型智能体的黑盒取证

    Black-Box Forensics for Conversational LLM Agents

    [https://arxiv.org/abs/2606.22698](https://arxiv.org/abs/2606.22698)

    该研究提出面向对话式LLM智能体的黑盒取证方法，无需模型参数访问即可在几轮对话内以98%准确率实现基础模型归因，并可通过系统提示词指纹识别将诈骗端点串联成犯罪网络、暴露隐蔽的API变更。

    

    随着基于大语言模型的诈骗日益泛滥，针对对话式LLM智能体的黑盒取证为隐藏在匿名端点背后的系统提供了一条问责路径。在不访问模型参数、也不知道隐藏系统提示词的情况下，识别聊天机器人端点背后的基础模型（归因），可以让调查人员将AI赋能的诈骗追溯到为其提供模型支持的服务提供商。而检测两个端点是否运行完全相同的系统提示词（指纹识别），即使该提示词是新颖且从未见过的，也能将个别诈骗案件串联成犯罪网络，并暴露隐蔽的API变更。我们对这两种能力进行了实证研究：我们的归因分类器仅通过几轮非对抗性对话，就能以98%的准确率识别出智能体背后的基础模型；系统提示词的归因虽然可行，但需要针对每个提示词在大量数据上重新训练，而现实中的系统提示词数量是无界的且……（摘要原文在此处被截断）

    arXiv:2606.22698v2 Announce Type: replace-cross  Abstract: As LLM-powered scams proliferate, black-box forensics for conversational LLM agents offers a path to accountability for systems hidden behind anonymous endpoints. Identifying the base model behind a chatbot endpoint (attribution), without model parameter access or knowledge of the hidden system prompt, would let investigators trace AI-enabled scams back to the providers whose models power them. Detecting when two endpoints run the exact same system prompt (fingerprinting), even one novel and unseen, would link individual scams into criminal networks and expose silent API changes. We conduct an empirical investigation of both capabilities. Our attribution classifiers identify the base model behind an agent with 98% accuracy from a few turns of non-adversarial conversation. Attribution of system prompts, while possible, requires retraining on a large amount of data for each prompt; system prompts in the wild are unbounded and eve
    
[^196]: 预训练自监督语音模型能够识别未见过的辅音

    Pretrained self-supervised speech models can recognize unseen consonants

    [https://arxiv.org/abs/2606.11542](https://arxiv.org/abs/2606.11542)

    本研究发现在高资源语言数据上预训练的自监督语音模型（Wav2Vec2 和 HuBERT），仅通过两种科伊桑语言的少量数据微调，就能比识别非搭嘴音更准确地识别训练中从未见过的搭嘴音，证明自监督学习具有跨人类语音的泛化能力。

    

    现代预训练自监督自动语音识别模型通过大规模音频数据训练，将语音编码为上下文化表示。然而，其训练数据严重偏向高资源语言，来自低资源语言的数据很少，这引发了对类型学上罕见语音（如主要出现在科伊桑语系语言中的搭嘴音）可能代表性不足的担忧。由此引出我们的核心研究问题：这些模型能否像识别其他语音一样准确地识别搭嘴音？为回答这一问题，我们在两种搭嘴音丰富的科伊桑语言（G|ui 语和西！Xoon 语）的数据上微调并比较了预训练自监督语音模型（Wav2Vec2 和 HuBERT）。结果表明，微调后的模型对搭嘴音的识别准确率始终高于非搭嘴音，这表明自监督学习能够实现对人类语音的泛化。

    arXiv:2606.11542v2 Announce Type: replace-cross  Abstract: Modern pretrained self-supervised automatic speech recognition models are trained on large-scale audio data to encode speech into contextualized representations. However, their training data are heavily skewed toward high-resource languages with little data from low-resource languages, raising concerns about the potential underrepresentation of typologically uncommon speech sounds such as click consonants primarily found in Khoisan languages. This leads to our central research question: Can these models recognize click consonants as accurately as other speech sounds? To address this question, we fine-tune and compare pretrained self-supervised speech models (Wav2Vec2 and HuBERT) on data from two click-rich Khoisan languages (G|ui and West !Xoon). Our results reveal that the fine-tuned models consistently recognize clicks more accurately than non-clicks, suggesting that self-supervision enables generalization across human speech
    
[^197]: 学习攻击与防御：基于GRPO的语言模型自适应红队测试

    Learning to Attack and Defend: Adaptive Red Teaming of Language Models via GRPO

    [https://arxiv.org/abs/2606.09701](https://arxiv.org/abs/2606.09701)

    本文提出一种基于GRPO的攻击-防御协同训练框架，通过多LLM评判奖励通道与GDPO优势计算以及从攻击者单独训练到协同训练的课程策略，实现了高效可迁移的攻击生成并同步提升防御者的安全能力。

    

    语言模型的安全性必须不断适应持续演变的攻击。近期研究已经证明，强化学习可以通过应用PPO式的自博弈和DPO式的在线偏好优化，同步训练更强的攻击者模型和防御者模型。在本工作中，我们探索了GRPO在这一场景中的有效性。协同训练可能具有挑战性，因为它需要联合优化攻击者和防御者的多个属性。因此，我们使用多个基于LLM评判器的奖励通道来塑造模型输出，并采用GDPO计算优势函数，以防止任何单一通道占据主导地位。我们的方法采用一种课程式训练策略，从仅训练攻击者的单轮和多轮训练逐步过渡到协同训练，在协同训练阶段攻击者与防御者模型交替更新。我们表明，该方法能够产生高度有效且可迁移的攻击，并且协同训练的防御者模型在保持竞争力的安全性的同时……

    arXiv:2606.09701v2 Announce Type: replace-cross  Abstract: Language model safety must continually adapt to evolving attacks. Recent works have demonstrated that reinforcement learning can be used to train stronger attacker and defender models in tandem by applying PPO-style self-play and DPO-style online preference optimization. In this work, we explore the efficacy of GRPO in this setting. Co-training can be challenging because it requires jointly optimizing multiple properties of both the attacker and defender. We therefore shape model outputs using multiple LLM judge-based reward channels and compute advantages with GDPO, which prevents any single channel from dominating. Our method uses a curriculum that progresses from attacker-only single-turn and multi-turn training to co-training, where attacker and defender models are updated in alternation. We show that this method produces highly effective and transferable attacks, and that co-trained defenders reach competitive safety while
    
[^198]: LoRi：面向隐式推理的低秩蒸馏

    LoRi: Low-Rank Distillation for Implicit Reasoning

    [https://arxiv.org/abs/2606.05315](https://arxiv.org/abs/2606.05315)

    该论文提出LoRi低秩蒸馏框架，利用隐状态推理轨迹的低秩结构，在共享低秩张量子空间中对齐教师与学生模型以迁移推理能力，使隐式推理性能接近显式思维链并超越现有iCoT蒸馏方法。

    

    隐式思维链方法旨在将推理能力内化到大语言模型中，但其性能通常不如显式思维链提示。我们通过实证研究发现，隐状态推理轨迹呈现出低秩结构。受此观察启发，我们提出了一种低秩蒸馏框架，利用一阶和二阶统计量将教师模型与学生模型的轨迹对齐到共享的低秩张量子空间中，从而实现推理能力的迁移。由此得到的公式化方法既能捕捉推理的全局结构，又能支持紧凑的潜在推理过程。我们在多个模型家族（包括LLaMA和Qwen）的不同规模上，于数学推理基准上对该方法进行了评估。我们的方法持续提升性能，尤其是在具有挑战性的多步任务上，接近显式思维链的准确率，并超越了此前的iCoT蒸馏方法。

    arXiv:2606.05315v2 Announce Type: replace-cross  Abstract: Implicit chain-of-thought (iCoT) methods aim to internalize reasoning in large language models, but often underperform explicit CoT prompting. We empirically find that hidden-state reasoning trajectories exhibit low-rank structure. Motivated by this observation, we propose a low-rank distillation framework that transfers reasoning by aligning teacher and student trajectories in a shared low-rank tensor subspace using first- and second-order statistics. The resulting formulation captures the global structure of reasoning while supporting a compact latent reasoning process. We evaluate the method across multiple model families, including LLaMA and Qwen, at different scales on mathematical reasoning benchmarks. Our approach consistently improves performance, especially on challenging multi-step tasks, approaching explicit CoT accuracy and outperforming prior iCoT distillation methods.
    
[^199]: EvoRubric：面向开放式生成的自演化量规驱动强化学习

    EvoRubric: Self-Evolving Rubric-Driven RL for Open-Ended Generation

    [https://arxiv.org/abs/2605.29847](https://arxiv.org/abs/2605.29847)

    提出EvoRubric框架，让一个共享策略同时充当推理器和量规生成器，通过标准有效性反馈、响应判别与留一法同行共识实现自演化的量规驱动强化学习，从而解决开放式生成中奖励不可验证的难题。

    

    强化学习已经在可验证领域推动了大语言模型的发展，而开放式生成由于缺乏确定性的奖励信号仍然充满挑战。基于量规的强化学习提供了明确的评估标准，但在最终答案的正确性无法验证的情况下，学习构建这些标准仍然十分困难。我们提出了EvoRubric，一个共同演化的强化学习框架，它结合了标准有效性反馈、响应判别和同行一致性，用于学习开放式生成的量规。一个共享策略同时充当推理器和量规生成器，利用其当前的响应和历史量规来发现新的评估维度。为了将自适应的量规发现与稳定的有效性检查相结合，初始策略的冻结副本充当元验证器，而一个冻结的评分器则根据保留的标准对响应进行评分。判别反馈、留一法同行共识以及……

    arXiv:2605.29847v2 Announce Type: replace  Abstract: Reinforcement Learning (RL) has advanced Large Language Models (LLMs) in verifiable domains, while open-ended generation remains challenging due to the absence of definitive rewards. Rubric-based RL provides explicit evaluation criteria, but learning to construct these criteria remains challenging when final-answer correctness is not verifiable. We propose EvoRubric, a co-evolutionary RL framework that combines criterion-validity feedback, response discrimination, and peer agreement to learn rubrics for open-ended generation. A shared policy acts as both a Reasoner and a Rubric Generator, using its current responses and historical rubrics to discover new evaluation dimensions. To combine adaptive rubric discovery with a stable validity check, a frozen copy of the initial policy serves as the Meta-Verifier, while a frozen Grader scores responses against the retained criteria. Discriminative feedback, Leave-One-Out peer consensus, and 
    
[^200]: MemTrace：追踪与归因大型语言模型记忆系统中的错误

    MemTrace: Tracing and Attributing Errors in Large Language Model Memory Systems

    [https://arxiv.org/abs/2605.28732](https://arxiv.org/abs/2605.28732)

    该论文提出了MemTrace框架，将LLM记忆流水线转化为可执行的记忆演化图，并结合MemTraceBench基准和自动归因方法，实现了对记忆系统错误的细粒度追踪与根因定位。

    

    记忆对于使大型语言模型能够支持长程推理至关重要，然而现有的记忆系统仍然不可靠且难以调试。追踪记忆的动态演变对于理解信息如何随时间被合成、传播或损坏至关重要。在本工作中，我们研究了LLM记忆系统中错误追踪与归因这一新问题。我们提出了一个新颖的框架，将记忆流水线转化为可执行的记忆演化图，实现对操作信息流的细粒度追踪。随后，我们构建了MemTraceBench，这是一个从代表性记忆系统（如Long-Context、RAG、Mem0和EverMemOS）中收集的基准数据集，用于系统地研究记忆失败模式。我们进一步引入了一种自动归因方法，通过迭代追踪操作子图来精确定位任何失败案例的根本原因。我们的分析揭示出，记忆失败是系统性的，源于……

    arXiv:2605.28732v4 Announce Type: replace-cross  Abstract: Memory is essential for enabling large language models to support long-horizon reasoning, yet existing memory systems remain unreliable and difficult to debug. Tracing memory's dynamic evolution is crucial to understand how information is synthesized, propagated, or corrupted over time. In this work, we study the new problem of error tracing and attribution in LLM memory systems. We propose a novel framework that transforms memory pipelines into executable memory evolution graphs, enabling fine-grained tracing of operational information flow. We then construct MemTraceBench, a benchmark collected from representative memory systems such as Long-Context, RAG, Mem0, and EverMemOS, to systematically study memory failure modes. We further introduce an automatic attribution method that iteratively traces operation subgraphs to pinpoint the root cause of any failed case. Our analysis reveals that memory failures are systematic, stemmi
    
[^201]: 面向自适应测试时推理的不确定性感知预算分配

    Uncertainty-Aware Budget Allocation for Adaptive Test-Time Reasoning

    [https://arxiv.org/abs/2605.26849](https://arxiv.org/abs/2605.26849)

    UAB通过投票熵度量问题难度，利用两阶段凹整数优化框架将固定采样预算自适应地集中分配给困难问题，从而比统一分配更高效地提升大语言模型的测试时推理性能。

    

    对同一问题采样多个回答可以提升语言模型的推理能力，但统一的算力分配是低效的，因为简单问题被过度采样，而困难问题却未得到充分探索。我们提出了不确定性感知预算分配（Uncertainty-Aware Budget Allocation, UAB），这是一个凹整数优化框架，利用从初始样本本身估计出的不确定性来重新分配固定的采样预算。在第一阶段，每个问题都会获得少量固定数量的生成结果。这些生成结果的答案分歧程度通过投票熵来度量，为问题难度提供了信号，同时这些生成结果也计入最终投票。在第二阶段，剩余预算由边际贪心算法进行分配，该算法最优地求解一个凹覆盖最大化替代问题，将样本集中分配给初始答案存在分歧的问题。在五个开源权重模型（参数量1.5B至27B）和五个不同难度的推理基准上，UAB平均提升了……（摘要原文在此处截断）

    arXiv:2605.26849v2 Announce Type: replace  Abstract: Sampling multiple responses improves language model reasoning, but uniform compute allocation is inefficient because easy questions are over-sampled while hard questions remain under-explored. We propose \textbf{Uncertainty-Aware Budget Allocation (UAB)}, a concave integer optimization framework that reallocates a fixed sampling budget using uncertainty estimated from the initial samples themselves. In Phase-1, every question receives a small fixed number of generations. Their answer disagreement, measured by vote entropy, provides a difficulty signal while these generations contribute to the final vote. In Phase-2, the remaining budget is allocated by a marginal-greedy algorithm that optimally solves a concave coverage-maximization surrogate, concentrating samples on questions whose initial answers disagree. Across five open-weight models (1.5B--27B parameters) and five reasoning benchmarks of varying difficulty, UAB improves averag
    
[^202]: 简短情感文本作为可穿戴传感的补充用于学生纵向健康监测的形成性研究

    A Formative Study of Brief Affective Text as a Complement to Wearable Sensing for Longitudinal Student Health Monitoring

    [https://arxiv.org/abs/2605.14360](https://arxiv.org/abs/2605.14360)

    该研究表明，学生每两个月仅需输入约三个词的简短担忧文本，经NLP模型分析后即可作为可穿戴设备被动传感的可扩展补充，有效用于睡眠与身体活动的纵向健康监测。

    

    可穿戴设备正以越来越高的保真度采集生理和行为数据，但塑造这些结果的心理背景却难以仅凭传感器数据还原，这限制了被动式传感在数字健康目标（如痛苦的早期识别、个性化干预和及时的临床跟进）中的实用性。我们研究了超简短的自然式担忧文本能否作为被动式传感的可扩展补充。在一项为期一年、共458名大学生（3,610个“人-时间波次”）的研究中，参与者佩戴Oura智能戒指，并每两个月对“你最担心什么”这一开放式问题作答，回答的中位长度仅为三个词。我们采用个体内混合效应模型，在九项睡眠和身体活动结果上，比较了基于词典的、通用预训练的和领域自适应的NLP方法，以确定哪种方法最能从这类文本中恢复与生理相关的信号

    arXiv:2605.14360v2 Announce Type: replace-cross  Abstract: Wearable devices capture physiological and behavioral data with increasing fidelity, but the psychological context shaping these outcomes is difficult to recover from sensor data alone, limiting the utility of passive sensing for digital health goals such as early detection of distress, personalized intervention, and timely clinical outreach. We examined whether ultra-brief naturalistic concern text could serve as a scalable complement to passive sensing. In a year-long study of 458 university students (3,610 person-waves) tracked with Oura rings, participants responded bimonthly to an open-ended prompt about what concerned them most; responses had a median length of three words. We compared dictionary-based, general pretrained, and domain-adapted NLP approaches using within-person mixed-effects models across nine sleep and physical activity outcomes to determine which method best recovers physiologically relevant signal from b
    
[^203]: CiteVQA：面向可信文档智能的证据归因基准测试

    CiteVQA: Benchmarking Evidence Attribution for Trustworthy Document Intelligence

    [https://arxiv.org/abs/2605.12882](https://arxiv.org/abs/2605.12882)

    提出CiteVQA基准，要求多模态大语言模型在回答文档问题时同时提供元素级边界框证据引用，从而联合评估答案正确性与证据归因的可信度。

    

    多模态大语言模型（MLLMs）在文档理解方面取得了显著进展，然而当前的文档视觉问答（Doc-VQA）评估仅对最终答案进行评分，而未对其支撑证据进行核查。这种只看答案的评估方式掩盖了一个关键的失败模式：模型可能在得出正确答案的同时，其依据却来自错误的文本区域——这在法律、金融和医学等高风险领域是一个关键风险，因为在这些领域中，每一个结论都必须能够追溯到特定的来源区域。为解决这一问题，我们提出了CiteVQA，这是一个要求模型在给出每个答案的同时返回元素级边界框引用的基准，从而对答案与证据进行联合评估。CiteVQA包含1,897个问题，涵盖711份PDF文档，跨越七个领域和两种语言，文档平均长度为40.6页。为确保准确性和可扩展性，基准的真实引用标注由一个自动化流程生成——该流程通过遮蔽消融识别关键证据，并通过……

    arXiv:2605.12882v2 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) have significantly advanced document understanding, yet current Doc-VQA evaluations score only the final answer and leave the supporting evidence unchecked. This answer-only approach masks a critical failure mode: a model can land on the correct answer while grounding it in the wrong passage---a critical risk in high-stakes domains like law, finance, and medicine, where every conclusion must be traceable to a specific source region. To address this, we introduce CiteVQA, a benchmark that requires models to return \textit{element-level} bounding-box citations alongside each answer, evaluating both jointly. CiteVQA comprises 1,897 questions across 711 PDFs spanning seven domains and two languages, averaging 40.6 pages per document. To ensure fidelity and scalability, the ground-truth citations are generated by an automated pipeline---which identifies crucial evidence via masking ablation and enf
    
[^204]: SkillGraph：基于演化技能图的智能体技能增强强化学习

    SkillGraph: Skill-Augmented Reinforcement Learning for Agents via Evolving Skill Graphs

    [https://arxiv.org/abs/2605.12039](https://arxiv.org/abs/2605.12039)

    该论文提出SkillGraph框架，将智能体的可复用技能表示为带类型边的演化有向图，通过检索有序的技能子图并利用强化学习反馈持续更新图结构，从而有效支持组合性任务的多步决策与技能库维护。

    

    技能库使大型语言模型智能体能够复用过去交互中的经验，但大多数现有技能库将技能存储为孤立条目，且仅通过语义相似性进行检索。这为组合性任务带来了两个关键挑战：首先，智能体不仅需要识别相关技能，还需要识别这些技能之间如何相互依赖、相互构建；其次，这也使技能库的维护变得困难，因为系统缺乏用于决定何时合并、拆分或移除技能的结构性线索。我们提出了SKILLGRAPH，一个将可复用技能表示为有向图中节点的框架，图中带类型的边编码了先决、增强和共现关系。给定一个新任务，SKILLGRAPH不仅检索单个技能，而是检索一个有序的技能子图，用以指导多步决策。该图通过智能体轨迹和强化学习反馈持续更新，从而……（原文摘要此处截断）

    arXiv:2605.12039v2 Announce Type: replace  Abstract: Skill libraries enable large language model agents to reuse experience from past interactions, but most existing libraries store skills as isolated entries and retrieve them only by semantic similarity. This leads to two key challenges for compositional tasks. Firstly, an agent must identify not only relevant skills but also how they depend on and build upon each other. Secondly, it also makes library maintenance difficult, since the system lacks structural cues for deciding when skills should be merged, split, or removed. We propose SKILLGRAPH, a framework that represents reusable skills as nodes in a directed graph, with typed edges encoding prerequisite, enhancement, and co-occurrence relations. Given a new task, SKILLGRAPH retrieves not just individual skills, but an ordered skill subgraph that can guide multi-step decision making. The graph is continuously updated from agent trajectories and reinforcement learning feedback, allo
    
[^205]: 电路能告诉我们多少？测量语言模型电路的一致性与特异性

    How Much Do Circuits Tell Us? Measuring the Consistency and Specificity of Language Model Circuits

    [https://arxiv.org/abs/2605.08348](https://arxiv.org/abs/2605.08348)

    本文提出用一致性和特异性两个新属性来评估机制可解释性中的电路，发现组件级电路虽高度一致且因果重要但不具任务特异性，而神经元级电路虽具任务特异性却一致性较差。

    

    arXiv:2605.08348v2 公告类型：替换。摘要：机制可解释性中的电路框架旨在识别对某种行为具有因果责任的模型组件稀疏子图，通常通过测量必要性和充分性来进行评估。但这些标准很少能说明一个电路是否一致地捕捉了模型执行任务的方式，或者该电路是否为该任务所特有。我们在六个任务和五个模型上研究了这两个属性——一致性和特异性，分别在组件层面（注意力头和MLP块）以及单个MLP神经元层面提取电路。我们发现，组件层面的电路在大多数任务上具有高度一致性和因果重要性，但它们并不具备特异性：消融某一任务的电路对另一任务性能造成的损害，与消融该任务自身电路的损害相差无几。另一方面，神经元层面的电路表现出更高的任务特异性，但其在任务内的一致性却要低得多。这可以通过电路……（原文在此处截断）

    arXiv:2605.08348v2 Announce Type: replace  Abstract: The circuits framework in mechanistic interpretability aims to identify sparse subgraphs of model components that are causally responsible for a behavior, typically evaluated by measuring necessity and sufficiency. But these criteria say little about whether a circuit consistently captures how a model performs a task, or if it is specific to that task. We study these two properties, consistency and specificity, across six tasks and five models, extracting circuits at the component level (attention heads and MLP blocks) and at the level of individual MLP neurons. We find that component-level circuits are highly consistent and causally important on most tasks, but they are not specific: ablating one task's circuit damages another task's performance about as much as that task's own circuit does. Neuron-level circuits, on the other hand, exhibit higher task-specificity but are far less consistent within tasks. This is explained by circui
    
[^206]: 顿悟还是故障？低精度如何驱动“弹弓机制”式损失尖峰

    Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes

    [https://arxiv.org/abs/2605.06152](https://arxiv.org/abs/2605.06152)

    本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。

    

    深度神经网络在无正则化的长期训练过程中会表现出周期性的损失尖峰，这一现象被称为“弹弓机制”。现有工作通常将其归因于内在的优化动力学，但其触发机制仍不清楚。本文证明该现象是浮点算术精度极限的结果：当训练进入高置信度阶段后，正确类别 logit 与其他 logit 之间的差值可能超过吸收误差阈值。于是在反向传播过程中，正确类别的梯度被精确舍入为零，而错误类别的梯度仍保持非零。这打破了跨类别梯度的零和约束，并在分类器层的参数更新中引入了系统性漂移。我们证明该漂移与特征之间形成了正反馈回路，导致全局分类器均值与全局特征（摘要在此处截断）。

    arXiv:2605.06152v4 Announce Type: replace-cross  Abstract: Deep neural networks exhibit periodic loss spikes during unregularized long-term training, a phenomenon known as the "Slingshot Mechanism." Existing work usually attributes this to intrinsic optimization dynamics, but its triggering mechanism remains unclear. This paper proves that this phenomenon is a result of floating-point arithmetic precision limits. As training enters a high-confidence stage, the difference between the correct-class logit and the other logits may exceed the absorption-error threshold. Then during backpropagation, the gradient of the correct class is rounded exactly to zero, while the gradients of the incorrect classes remain nonzero. This breaks the zero-sum constraint of gradients across classes and introduces a systematic drift in the parameter update of the classifier layer. We prove that this drift forms a positive feedback loop with the feature, causing the global classifier mean and the global featu
    
[^207]: 状态流Transformer (SST) V2：面向潜在空间推理的非线性递归并行训练

    State Stream Transformer (SST) V2: Parallel Training of Nonlinear Recurrence for Latent Space Reasoning

    [https://arxiv.org/abs/2605.00206](https://arxiv.org/abs/2605.00206)

    SST V2通过在每层引入FFN驱动的非线性递归，使潜在状态横向流经整个序列，实现连续潜在空间中的参数高效推理与深思，并采用两遍并行训练使其计算上可行。

    

    当前的Transformer在位置之间丢弃了其丰富的潜在残差流，在每个新位置重新构建潜在推理上下文，导致潜在的推理能力未被充分利用。状态流Transformer (SST) V2通过在每个解码器层引入由FFN驱动的非线性递归，实现了在连续潜在空间中的参数高效推理，其中潜在状态通过学习到的混合方式沿整个序列横向流动传输。这一机制还支持在推理时对每个位置进行连续的潜在深思，在生成token之前投入额外的计算量来探索抽象推理。一种两遍并行训练程序近似了序列递归，使得共同训练在计算上切实可行。隐状态分析表明，状态流通过连续潜在空间中急剧的、依赖于内容的重组来促进推理，最终由语言模型头输出结果。

    arXiv:2605.00206v2 Announce Type: replace-cross  Abstract: Current transformers discard their rich latent residual stream between positions, reconstructing latent reasoning context at each new position and leaving potential reasoning capacity untapped. The State Stream Transformer (SST) V2 enables parameter-efficient reasoning in continuous latent space through an FFN-driven nonlinear recurrence at each decoder layer, where latent states are streamed horizontally across the full sequence via a learned blend. This same mechanism supports continuous latent deliberation per position at inference time, dedicating additional FLOPs to exploring abstract reasoning before committing to a token. A two-pass parallel training procedure approximates the sequential recurrence, making co-training computationally practical. Hidden state analysis shows that the state stream facilitates reasoning through sharp, content-dependent reorganisations in continuous latent space; the LM head exposes the result
    
[^208]: AI申诉处理器：一种用于政务服务中公民申诉自动化分类的深度学习方法

    AI Appeals Processor: A Deep Learning Approach to Automated Classification of Citizen Appeals in Government Services

    [https://arxiv.org/abs/2604.03672](https://arxiv.org/abs/2604.03672)

    该论文提出的AI申诉处理器在10,000条真实俄语公民申诉上对比多种深度学习模型，发现Word2Vec+LSTM（78%）接近多语言BERT（82%）且远超人工操作员（67%），并因训练成本更低、便于在仅CPU的政府环境中频繁再训练而选择部署LSTM。

    

    政府机构必须在法定时限内登记、分类并流转每一份公民申诉，而其中大部分工作仍由人工完成。我们介绍了AI申诉处理器——一个部署在仅使用CPU的政府环境中的分类与流转组件，并报告其评估与部署过程中获得的经验。在一个来自跨领域数据集、包含10,000条真实俄语申诉的数据上，我们针对三分类的申诉类型任务，比较了词袋模型和TF-IDF配合SVM、fastText、Word2Vec+LSTM以及多语言BERT。在包含1,500条申诉的留出测试集上，BERT达到82%的准确率，Word2Vec+LSTM达到78%，而以专家裁定的黄金标准衡量的单人操作员准确率仅为67%。我们最终部署了LSTM模型：在操作员需要验证每一条预测结果的工作流中，其更低的训练成本使得基于操作员已验证标签的频繁再训练变得切实可行，而四个百分点的准确率差距并未改变操作员的工作内容。端到……

    arXiv:2604.03672v2 Announce Type: replace-cross  Abstract: Government agencies must register, classify and route every citizen appeal within statutory time limits, and much of this work is still done by hand. We describe AI Appeals Processor, a classification and routing component deployed in a CPU-only government environment, and report what its evaluation and deployment taught us. On 10,000 real Russian-language appeals from a cross-domain dataset, we compare Bag-of-Words and TF-IDF with SVM, fastText, Word2Vec+LSTM and multilingual BERT on a three-way appeal-type task. On a held-out test set of 1,500 appeals, BERT reaches 82% accuracy and Word2Vec+LSTM 78%, against 67% for individual operators measured on an expert-adjudicated gold standard. We deployed the LSTM: in a workflow where an operator verifies every prediction, its lower training cost made frequent retraining on operator-verified labels practical, while the four-point accuracy gap did not change the operator's task. End-to
    
[^209]: 混合专家语言模型中通过路由重加权实现有限的刻板印象控制

    Limited Stereotype Control Through Routing Reweighting in MoE Language Models

    [https://arxiv.org/abs/2603.27141](https://arxiv.org/abs/2603.27141)

    本文提出FARE诊断框架，发现虽然人口统计学提示词在MoE模型中的路由方式与中性提示词存在差异，但推理时的路由重加权对刻板印象的控制效果十分有限（偏好变化最多仅约1.3–1.5个百分点）。

    

    在混合专家语言模型中，人口统计学相关提示词的路由方式与中性提示词不同，这促使我们开展路由层面的刻板印象控制测试。我们提出了公平感知路由均衡（FARE），这是一个结合了人口统计学路由画像、经验层选择和固定推理时重加权的诊断框架，并在英语环境下评估了五种MoE架构。在选定的运行点上，CrowS-Pairs偏好最多变化1.3个百分点；DeepSeekMoE未选择任何干预措施。配对95%置信区间排除了每个被干预模型上大于2.2个点的下降，而唯一名义上显著的变化（Qwen1.5，p = 0.015）在多重比较校正后不再显著。尽管如此，OLMoE和Qwen3几乎改变了每一个top-k专家集合。噪声对照、随机与截断的合成画像以及硬掩蔽也最多使偏好移动1.5个百分点。四种生成探针……（摘要在此处截断）

    arXiv:2603.27141v2 Announce Type: replace  Abstract: Demographic prompts are routed differently from neutral prompts in Mixture-of-Experts (MoE) language models, motivating tests of routing-level stereotype control. We introduce Fairness-Aware Routing Equilibrium (FARE), a diagnostic framework combining demographic routing profiles, empirical layer selection, and fixed inference-time reweighting, and evaluate five MoE architectures in English. At the selected operating points, CrowS-Pairs preference changes by at most 1.3 percentage points; DeepSeekMoE selects no intervention. Paired 95% confidence intervals exclude decreases larger than 2.2 points on each intervened model, and the only nominally significant change (Qwen1.5, p = 0.015) does not survive multiple-comparison correction. OLMoE and Qwen3 nevertheless change nearly every top-k expert set. Noise controls, random and truncated synthetic profiles, and hard masking also move preference by at most 1.5 points. Four generation prot
    
[^210]: 祈使干扰：社会语域如何塑造大语言模型中的指令拓扑结构

    Imperative Interference: Social Register Shapes Instruction Topology in Large Language Models

    [https://arxiv.org/abs/2603.25015](https://arxiv.org/abs/2603.25015)

    该研究发现大语言模型会按照社会语域惯例来处理系统提示指令——同一指令在英语中协作而在西班牙语中竞争，将祈使句改写为陈述句可降低81%的跨语言差异，说明模型把指令理解为社会行为而非技术规范。

    

    在系统提示词中，相同语义内容的指令在英语里相互协作，在西班牙语里却相互竞争，呈现出截然相反的交互拓扑结构。我们通过跨四种语言、四个模型的指令级消融实验表明，这种拓扑反转由社会语域所介导：祈使语气在不同言语社区中承载着不同的强制力，而基于多语言数据训练的模型已经习得了这些惯例。将单个指令块改写为陈述句可使跨语言方差降低81%（p = 0.029，置换检验）。在十一个祈使句指令块中仅改写三个，即可将西班牙语的指令拓扑从竞争性转变为协作性，并对未改写的指令块产生溢出效应。这些发现表明，模型将指令处理为社会行为而非技术规范：“永远不要做X”是一种权威的行使，其效力依赖于语言；而“X：已禁用”则只是一个事实。

    arXiv:2603.25015v2 Announce Type: replace-cross  Abstract: System prompt instructions that cooperate in English compete in Spanish, with the same semantic content, but opposite interaction topology. We present instruction-level ablation experiments across four languages and four models showing that this topology inversion is mediated by social register: the imperative mood carries different obligatory force across speech communities, and models trained on multilingual data have learned these conventions. Declarative rewriting of a single instruction block reduces cross-linguistic variance by 81% (p = 0.029, permutation test). Rewriting three of eleven imperative blocks shifts Spanish instruction topology from competitive to cooperative, with spillover effects on unrewritten blocks. These findings suggest that models process instructions as social acts, not technical specifications: "NEVER do X" is an exercise of authority whose force is language-dependent, while "X: disabled" is a fact
    
[^211]: 超越预设身份：生成式智能体社会中的选择性立场调适与互动重组

    Beyond Preset Identities: Selective Stance Accommodation and Interaction Reorganisation in Generative Agent Societies

    [https://arxiv.org/abs/2603.23406](https://arxiv.org/abs/2603.23406)

    该研究超越仅测量立场变化程度的传统方法，通过结合立场测量、来源评估与时序网络分析，揭示生成式智能体社会中智能体接受立场变化的实质内容以及强化特定伙伴关系的互动重组机制。

    

    生成式智能体社会通过分配的角色、偏好和关系来模拟人类。当智能体交换论点并选择互动伙伴时，它们可以修正自身立场并重组讨论。理解这些变化需要考察它们接受了什么，以及它们如何继续互动。立场变化分数和沟通总量只能描述变化的程度。然而，同样的立场移动可能保持也可能逆转预设的偏好，频繁的沟通既可能促成一致，也可能伴随持续的分歧。因此，我们考察已改变立场的具体内容，以及强化特定伙伴关系的交流过程。利用计算多智能体社会实验（CMASE），我们将立场测量、来源评估和时序网络与个体回答和消息相结合。研究1比较了七种实验条件，每种条件进行十次GPT-4o运行，另有一个单独的访谈收集，涵盖……（原文摘要此处截断）

    arXiv:2603.23406v3 Announce Type: replace  Abstract: Generative agent societies simulate people with assigned roles, preferences and relationships. As agents exchange arguments and choose partners, they can revise their positions and reorganise discussion. Understanding these changes requires examining what they accept and how they continue to interact. Stance-change scores and communication totals describe the extent of change. However, the same stance movement can preserve or reverse an assigned preference, and frequent communication can support either agreement or continuing disagreement. We therefore examine the content of changed positions and the exchanges that strengthen particular partnerships. Using Computational Multi-Agent Society Experiments (CMASE), we combine stance measures, source evaluations and temporal networks with individual answers and messages. Study 1 compares seven conditions across ten GPT-4o runs per condition, with a separate interview collection covering fo
    
[^212]: 智能体批判性训练

    Agentic Critical Training

    [https://arxiv.org/abs/2603.08706](https://arxiv.org/abs/2603.08706)

    该论文提出智能体批判性训练，利用可验证奖励的强化学习让模型学会直接区分专家动作与看似合理的错误动作，作为模仿学习前的高效热身方法，无需参考推理且支持数据跨模型复用。

    

    模仿学习（IL）教会语言模型智能体复现专家动作，却无法让它们将这些动作与看似合理的错误区分开来。自我反思方法虽然让模型接触到替代方案，却使用监督微调（SFT）来模仿固定的推理和动作。我们提出了智能体批判性训练，它利用可验证奖励的强化学习（RLVR）来训练模型直接判别动作。在每个专家轨迹状态处，ACT将专家动作与从初始策略中采样的替代动作配对，并随机化二者的顺序。模型自行生成推理，但只有在选择专家动作时才获得奖励。ACT可以复用演示数据，无需参考推理文本，并允许配对数据在不同模型规模间复用。ACT是模仿学习之前的热身步骤，之后可选择继续进行强化学习；推理时无需进行候选动作比较。在Qwen3-8B和Olmo-3-7B-Instruct模型上，针对ALFWorld-ID、WebShop和ScienceWorld基准，ACT带来了（性能提升，摘要在此处截断）……

    arXiv:2603.08706v2 Announce Type: replace  Abstract: Imitation learning (IL) teaches language-model agents to reproduce expert actions but not to distinguish them from plausible mistakes. Self-reflection methods expose models to alternatives yet use supervised fine-tuning (SFT) to imitate fixed rationales and actions. We introduce Agentic Critical Training (ACT), which uses reinforcement learning with verifiable rewards (RLVR) to train models to judge actions directly. At each expert-trajectory state, ACT pairs an expert action with an alternative sampled from the initial policy and randomizes their order. The model generates its own reasoning but is rewarded only for selecting the expert action. ACT reuses demonstrations, requires no reference rationales, and allows pair reuse across model sizes. ACT is a warm-up before IL, optionally followed by RL; inference requires no candidate comparison. Across Qwen3-8B and Olmo-3-7B-Instruct on ALFWorld-ID, WebShop, and ScienceWorld, ACT yields
    
[^213]: \$OneMillion-Bench：语言智能体距离人类专家还有多远？

    \$OneMillion-Bench: How Far are Language Agents from Human Experts?

    [https://arxiv.org/abs/2603.07980](https://arxiv.org/abs/2603.07980)

    提出了包含 400 个专家设计的跨领域专业任务的 \$OneMillion-Bench 基准，通过基于评分规则的多维度评估，衡量语言智能体在法律、金融、医疗等经济关键场景中与人类专家的差距。

    

    随着语言模型从聊天助手演变为能够进行多步推理和工具使用的长程智能体，现有基准测试在很大程度上仍局限于结构化或考试式的任务，无法满足现实世界的专业需求。为此，我们推出了 \$OneMillion-Bench（\$OMB），这是一个包含 400 个由专家精心设计的任务的基准，涵盖法律、金融、工业、医疗健康和自然科学领域，旨在评估智能体在经济上具有重大影响的场景中的表现。与以往的工作不同，该基准要求检索权威来源、解决相互矛盾的证据、应用特定领域的规则并做出受约束的决策，其正确性既取决于最终答案，也取决于推理过程本身。我们采用基于评分规则的评估协议，从事实准确性、逻辑连贯性、实际可行性和专业合规性等维度进行评分，并专注于专家级问题，以确保有意义的区分度（原文摘要在此处截断）。

    arXiv:2603.07980v2 Announce Type: replace-cross  Abstract: As language models (LMs) evolve from chat assistants to long-horizon agents capable of multi-step reasoning and tool use, existing benchmarks remain largely confined to structured or exam-style tasks that fall short of real-world professional demands. To this end, we introduce \$OneMillion-Bench (\$OMB), a benchmark of 400 expert-curated tasks spanning Law, Finance, Industry, Healthcare, and Natural Science, built to evaluate agents across economically consequential scenarios. Unlike prior work, the benchmark requires retrieving authoritative sources, resolving conflicting evidence, applying domain-specific rules, and making constraint decisions, where correctness depends as much on the reasoning process as the final answer. We adopt a rubric-based evaluation protocol scoring factual accuracy, logical coherence, practical feasibility, and professional compliance, focusing on expert-level problems to ensure meaningful differenti
    
[^214]: 通过奖励引导拼接实现扩散语言模型的测试时扩展

    Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching

    [https://arxiv.org/abs/2602.22871](https://arxiv.org/abs/2602.22871)

    该论文提出Stitching Noisy Diffusion Thoughts框架，利用过程奖励模型对扩散语言模型采样的多条推理轨迹的中间步骤进行评分，并将跨轨迹的最优步骤拼接成复合推理链，实现了步骤级而非轨迹级的中间推理成果复用与测试时扩展。

    

    使用大型语言模型进行推理通常受益于生成多条思维链，但现有的聚合策略通常是轨迹层面的（例如，选择最佳轨迹或对最终答案进行投票），这会丢弃来自部分完成或“接近正确”尝试中的有用中间工作。我们提出了Stitching Noisy Diffusion Thoughts（拼接噪声扩散思维），这是一个自洽性框架，能将廉价的扩散采样推理转化为可复用的步骤级候选池。给定一个问题，我们（i）使用掩码扩散语言模型采样多条多样化、低成本的推理轨迹，（ii）使用现成的过程奖励模型（PRM）为每个中间步骤评分，（iii）将跨轨迹的最高质量步骤拼接成一个复合推理链，然后仅使用该推理链重新计算最终答案。这种模块化流程将探索（扩散）与评估和解决方案合成相分离，避免了……

    arXiv:2602.22871v2 Announce Type: replace-cross  Abstract: Reasoning with large language models often benefits from generating multiple chains-of-thought, but existing aggregation strategies are typically trajectory-level (e.g., selecting the best trace or voting on the final answer), discarding useful intermediate work from partial or "nearly correct" attempts. We propose Stitching Noisy Diffusion Thoughts, a self-consistency framework that turns cheap diffusion-sampled reasoning into a reusable pool of step-level candidates. Given a problem, we (i) sample many diverse, low-cost reasoning trajectories using a masked diffusion language model, (ii) score every intermediate step with an off-the-shelf process reward model (PRM), and (iii) stitch these highest-quality steps across trajectories into a composite rationale. This rationale is then used to recompute only the final answer. This modular pipeline separates exploration (diffusion) from evaluation and solution synthesis, avoiding mo
    
[^215]: 基于局部解释量化RAG中检索器与生成器的对齐性

    Quantifying Retriever-Generator Alignment in RAG with Local Explanations

    [https://arxiv.org/abs/2601.21803](https://arxiv.org/abs/2601.21803)

    该论文提出端到端可解释框架RAG-E，通过适配的积分梯度、蒙特卡洛稳定的Shapley值归因以及新颖的WARG对齐指标，量化RAG系统中检索器与生成器的对齐程度，实验揭示生成器常常忽略排名靠前的文档而依赖相关性较低的文档。

    

    检索增强生成（RAG）系统将稠密检索器与语言模型相结合，使输出基于外部文档。然而，这些组件之间的交互仍然不透明，这为在高风险领域的部署带来了挑战。我们提出了RAG-E，这是一个端到端的可解释性框架，通过基于数学原理的归因方法来量化检索器与生成器之间的对齐程度。我们的方法将积分梯度（Integrated Gradients）适配用于检索器分析，提出了一种蒙特卡洛稳定的Shapley值近似方法用于生成器归因，并引入了检索器与生成器之间的加权对齐度（WARG）指标，以衡量生成器的文档使用情况与检索器排名的贴合程度。在PopQA、QAMPARI和TREC CAST数据集上的实验揭示了显著的不对齐现象：根据模型和设置的不同，生成器经常忽略排名靠前的文档，而依赖于排名较低的文档。

    arXiv:2601.21803v3 Announce Type: replace  Abstract: Retrieval-Augmented Generation (RAG) systems combine dense retrievers and language models to ground outputs in external documents. However, the interaction between these components remains opaque, creating challenges for deployment in high-stakes domains. We present RAG-E, an end-to-end explainability framework that quantifies retriever-generator alignment through mathematically grounded attribution methods. Our approach adapts Integrated Gradients for retriever analysis, proposes a Monte Carlo-stabilized Shapley Value approximation for generator attribution, and introduces the Weighted Alignment between Retriever and Generator (WARG) metric to measure how closely the generator's document usage aligns with retriever rankings. Experiments on PopQA, QAMPARI, and TREC CAST datasets reveal substantial misalignment: depending on the model and setting, generators often ignore top-ranked documents and rely on documents ranked as less releva
    
[^216]: 同行评审真的在衰落吗？跨会议与跨时间的评审质量分析

    Is Peer Review Really in Decline? Analyzing Review Quality across Venues and Time

    [https://arxiv.org/abs/2601.15172](https://arxiv.org/abs/2601.15172)

    该研究提出了一个基于证据的评审质量比较分析框架，结合LLM与轻量级度量方法对ICLR、NeurIPS和*ACL三大会议的评审质量进行跨会议、跨时间研究，其发现与“评审质量正在下降”的流行观点相反。

    

    同行评审是现代科学的核心。随着投稿数量的上升和研究社群的不断壮大，“评审质量正在下降”成为一种流行的叙事和普遍的担忧。然而，事实果真如此吗？评审质量本身难以衡量，而且评审实践的持续演变使得跨会议、跨时间的评审比较变得困难。为解决这一问题，我们引入了一个用于基于证据的评审质量比较研究的新框架，并将其应用于主要的人工智能与机器学习会议：ICLR、NeurIPS 和 *ACL。我们记录了评审格式的多样性，并提出了一种新的评审标准化方法。我们提出了一个多维度的量化体系，将评审质量定义为对编辑和作者的有用性，并结合基于大语言模型（LLM）的度量方法与轻量级度量方法。我们研究了各项评审质量度量之间的关系及其随时间的演变。与流行的叙事相反，我们的跨时间分析……

    arXiv:2601.15172v2 Announce Type: replace  Abstract: Peer review is at the heart of modern science. As submission numbers rise and research communities grow, the decline in review quality is a popular narrative and a common concern. Yet, is it true? Review quality is difficult to measure, and the ongoing evolution of reviewing practices makes it hard to compare reviews across venues and time. To address this, we introduce a new framework for evidence-based comparative study of review quality and apply it to major AI and machine learning conferences: ICLR, NeurIPS and *ACL. We document the diversity of review formats and introduce a new approach to review standardization. We propose a multi-dimensional schema for quantifying review quality as utility to editors and authors, coupled with both LLM-based and lightweight measurements. We study the relationships between measurements of review quality, and its evolution over time. Contradicting the popular narrative, our cross-temporal analys
    
[^217]: 通过翻译掩盖数据污染：来自阿拉伯语语料的证据

    Obscuring Data Contamination Through Translation: Evidence from Arabic Corpora

    [https://arxiv.org/abs/2601.14994](https://arxiv.org/abs/2601.14994)

    该研究表明，将 MMLU 和 XQuAD 基准翻译成阿拉伯语后暴露给模型，既能提升模型在英语基准上的表现，又能使 TS-Guessing 和 Min-K%++ 等现有污染检测方法失效，说明跨语言方式可以掩盖数据污染。

    

    当模型受益于记忆评估内容而非真正的泛化能力时，数据污染可能会使基准评估失效。然而，当暴露的内容与评估基准的语言不同时，污染很难被审计。我们通过有意让四个开放权重的指令微调大语言模型以递增的暴露程度接触 MMLU 和 XQuAD 评估项目的阿拉伯语翻译，然后在原始英语任务上对其进行评估，来研究这种失效模式。这种受控设置是数据污染的代理实验，而非对现实世界预训练数据泄漏的重现。我们首先测试了两种以英语为中心的事后探测方法——TS-Guessing 和 Min-K%++，发现它们的信号在翻译暴露下基本消失：TS-Guessing 除了在 MMLU 上存在模型特定的位置记忆外信号依然微弱，而 Min-K%++ 则保持在随机水平或以下。与此同时，英语 MMLU 的性能随着阿拉伯语暴露的增加而提升。

    arXiv:2601.14994v2 Announce Type: replace-cross  Abstract: Data contamination can invalidate benchmark evaluation when a model benefits from memorized evaluation content rather than genuine generalization. Yet contamination is difficult to audit when the exposed content differs in language from the evaluation benchmark. We study this failure mode by deliberately exposing four open-weight instruction-tuned LLMs to Arabic translations of MMLU and XQuAD evaluation items at increasing exposure levels, then evaluating them on the original English tasks. This controlled setup is a proxy for contamination rather than a reconstruction of real-world pretraining leakage. We first test two English-centric post-hoc probes, TS-Guessing and Min-K%++, and find that their signals largely disappear under translated exposure: TS-Guessing remains weak except for model-specific positional recall on MMLU, while Min-K%++ stays at or below chance. At the same time, English MMLU performance increases with Ara
    
[^218]: LMSpell：基于预训练语言模型的拼写纠错

    LMSpell: Spell Correction with Pre-Trained Language Models

    [https://arxiv.org/abs/2512.05414](https://arxiv.org/abs/2512.05414)

    该研究首次系统比较了三类预训练语言模型在多语言（含低资源语言）拼写纠错中的效果，证明即使是2.7亿参数的小型模型仅在5千个句子上微调，也能超越基于规则的拼写纠错方法。

    

    拼写纠错对许多语言来说仍然是一个具有挑战性的问题，尤其是低资源语言（LRLs）。虽然预训练语言模型（PLMs）已被应用于拼写纠错，但尚未有人对不同PLMs进行过适当的比较。我们首次开展了实证研究，评估三类PLMs在包括低资源语言在内的多种语言上用于拼写纠错的有效性。我们证明，即使是相对较小的PLMs，如2.7亿参数的Gemma 3和mBART50，仅在5千个句子的数据集上进行微调，其表现也能超越基于规则的拼写纠错器，这突显了在数据有限的情况下构建有效拼写纠错系统的实用途径。我们还以僧伽罗语为例进行了案例研究，以揭示低资源语言拼写纠错所面临的困境。

    arXiv:2512.05414v4 Announce Type: replace  Abstract: Spell correction is still a challenging problem for many languages, especially low-resource languages (LRLs). While pre-trained language models (PLMs) have been employed for spell correction, there has been no proper comparison across PLMs. We present the first empirical study on the effectiveness of the three types of PLMs for spell correction across multiple languages, including low-resource languages. We show that even relatively small PLMs such as the 270M-parameter Gemma 3 and mBART50, when fine-tuned on a dataset of only 5k sentences, can outperform rule-based spell correctors, highlighting a practical pathway for building effective spell correction systems with limited data. We also present a case study with Sinhala to shed light on the plight of spell correction for LRLs.
    
[^219]: 通过合成模型生成实现近最优可解释模型的可扩展元学习

    Towards Scalable Meta-Learning of near-optimal Interpretable Models via Synthetic Model Generations

    [https://arxiv.org/abs/2511.04000](https://arxiv.org/abs/2511.04000)

    本文提出通过合成采样近最优决策树来生成大规模预训练数据的高效可扩展方法，使MetaTree transformer在决策树元学习上达到与真实数据或昂贵最优树预训练相当的性能，同时大幅降低计算成本。

    

    决策树因其可解释性而广泛应用于金融和医疗等高风险领域。本工作提出了一种高效、可扩展的方法来生成合成预训练数据，从而实现决策树的元学习。我们的方法通过合成方式采样近最优的决策树，构建出大规模、贴近现实的数据集。借助MetaTree transformer架构，我们证明该方法所取得的性能可与在真实数据上预训练或使用计算成本高昂的最优决策树预训练相媲美。该策略显著降低了计算成本，提升了数据生成的灵活性，并为可解释决策树模型的可扩展、高效元学习铺平了道路。

    arXiv:2511.04000v2 Announce Type: replace-cross  Abstract: Decision trees are widely used in high-stakes fields like finance and healthcare due to their interpretability. This work introduces an efficient, scalable method for generating synthetic pre-training data to enable meta-learning of decision trees. Our approach samples near-optimal decision trees synthetically, creating large-scale, realistic datasets. Using the MetaTree transformer architecture, we demonstrate that this method achieves performance comparable to pre-training on real-world data or with computationally expensive optimal decision trees. This strategy significantly reduces computational costs, enhances data generation flexibility, and paves the way for scalable and efficient meta-learning of interpretable decision tree models.
    
[^220]: AyurParam：面向阿育吠陀医学的先进双语语言模型

    AyurParam: A State-of-the-Art Bilingual Language Model for Ayurveda

    [https://arxiv.org/abs/2511.02374](https://arxiv.org/abs/2511.02374)

    研究者推出了面向阿育吠陀传统医学的领域专精双语模型 AyurParam-2.9B，通过专家精心整理的英印双语数据集进行微调，在 BhashaBench-Ayur 基准上超越了同规模（1.5B–3B）的所有开源指令微调模型。

    

    当前的大型语言模型在广泛的通用任务上表现出色，但在面对需要深厚文化、语言学和专业知识的高度专业化领域时，性能持续不佳。尤其是像阿育吠陀（Ayurveda）这样的传统医学体系，蕴含着数百年积累的细腻文本与临床知识，主流大语言模型难以准确解读或应用这些知识。我们推出了 AyurParam-2.9B，这是一个领域专精的双语语言模型，基于 Param-1-2.9B 微调而成，训练数据是一个覆盖经典文献和临床指导、由专家精心整理的大型阿育吠陀数据集。AyurParam 的数据集包含英语和印地语的上下文理解型、推理型和客观题式问答，并采用严格的标注协议以确保事实准确性与指令清晰度。在 BhashaBench-Ayur 基准测试中，AyurParam 不仅超越了同参数规模级别（15亿至30亿参数）的所有开源指令微调模型，而且……

    arXiv:2511.02374v2 Announce Type: replace-cross  Abstract: Current large language models excel at broad, general-purpose tasks, but consistently underperform when exposed to highly specialized domains that require deep cultural, linguistic, and subject-matter expertise. In particular, traditional medical systems such as Ayurveda embody centuries of nuanced textual and clinical knowledge that mainstream LLMs fail to accurately interpret or apply. We introduce AyurParam-2.9B, a domain-specialized, bilingual language model fine-tuned from Param-1-2.9B using an extensive, expertly curated Ayurveda dataset spanning classical texts and clinical guidance. AyurParam's dataset incorporates context-aware, reasoning, and objective-style Q&A in both English and Hindi, with rigorous annotation protocols for factual precision and instructional clarity. Benchmarked on BhashaBench-Ayur, AyurParam not only surpasses all open-source instruction-tuned models in its size class (1.5--3B parameters), but al
    
[^221]: 不止于表象？揭示视觉-语言驾驶模型训练中推理与规划的脱节现象

    More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models

    [https://arxiv.org/abs/2510.04532](https://arxiv.org/abs/2510.04532)

    本研究构建了包含规划对齐思维链的DriveMind驾驶视觉问答数据集，通过信息消融实验首次系统揭示了视觉-语言驾驶模型中自然语言推理与轨迹规划之间的因果脱节，即规划性能主要依赖先验信息而非语言推理。

    

    视觉-语言模型驱动的智能体承诺通过先产生自然语言推理、再预测轨迹规划的方式，实现可解释的端到端自动驾驶。然而，规划是否真的由这种推理因果驱动，仍然是一个关键但未经证实的假设。为了探究这一问题，我们构建了DriveMind——一个从nuPlan自动生成的大规模驾驶视觉问答语料库，其中包含与规划对齐的思维链。我们的数据生成过程将传感器和标注转换为结构化输入，更关键的是，将先验信息与待推理信号分离，从而支持干净的信息消融实验。利用DriveMind，我们采用监督微调和组相对策略优化训练了代表性的VLM智能体，并使用nuPlan的指标进行评估。遗憾的是，我们的结果表明推理与规划之间存在一致的因果脱节：移除自车/导航先验会导致规划性能大幅下降

    arXiv:2510.04532v2 Announce Type: replace  Abstract: Vision-Language Model (VLM) driving agents promise explainable end-to-end autonomy by first producing natural-language reasoning and then predicting trajectory planning. However, whether planning is causally driven by this reasoning remains a critical but unverified assumption. To investigate this, we build DriveMind, a large-scale driving Visual Question Answering corpus with plan-aligned Chain-of-Thought (CoT), automatically generated from nuPlan. Our data generation process converts sensors and annotations into structured inputs and, crucially, separates priors from to-be-reasoned signals, enabling clean information ablations. Using DriveMind, we train representative VLM agents with Supervised Fine-Tuning and Group Relative Policy Optimization and evaluate them with nuPlan's metrics. Our results, unfortunately, indicate a consistent causal disconnect in reasoning-planning: removing ego/navigation priors causes large drops in plann
    
[^222]: GUI-KV：通过具有时空感知能力的KV缓存实现高效GUI智能体

    GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness

    [https://arxiv.org/abs/2510.00536](https://arxiv.org/abs/2510.00536)

    该论文提出了GUI-KV，一种即插即用的KV缓存压缩方法，通过利用GUI的空间和时间冗余特性以及跨层均匀的缓存预算分配策略，显著提升了GUI智能体的推理效率。

    

    基于视觉-语言模型构建的图形用户界面（GUI）智能体已成为自动化人机工作流程的一种有前景的方法。然而，由于它们需要处理长序列的高分辨率截图并解决长时程任务，因此也面临效率低下的挑战，这使得推理缓慢、成本高昂且受内存限制。虽然键值（KV）缓存可以缓解这一问题，但对于图像密集型的上下文来说，存储完整的缓存是不可行的。现有的缓存压缩方法并非最优，因为它们没有考虑GUI的空间和时间冗余性。在这项工作中，我们首先分析了GUI智能体工作负载中的注意力模式，发现与自然图像不同，注意力稀疏性在所有transformer层中都均匀地保持较高水平。这一见解启发了一种简单的均匀预算分配策略，我们通过实验证明它优于更复杂的层级变化方案。基于此，我们引入了GUI-KV，一种即插即用（摘要在此处截断）

    arXiv:2510.00536v2 Announce Type: replace  Abstract: Graphical user interface (GUI) agents built on vision-language models have emerged as a promising approach to automate human-computer workflows. However, they also face the inefficiency challenge as they process long sequences of high-resolution screenshots and solving long-horizon tasks, making inference slow, costly and memory-bound. While key-value (KV) caching can mitigate this, storing the full cache is prohibitive for image-heavy contexts. Existing cache-compression methods are sub-optimal as they do not account for the spatial and temporal redundancy of GUIs. In this work, we first analyze attention patterns in GUI agent workloads and find that, unlike in natural images, attention sparsity is uniformly high across all transformer layers. This insight motivates a simple uniform budget allocation strategy, which we show empirically outperforms more complex layer-varying schemes. Building on this, we introduce GUI-KV, a plug-and-
    
[^223]: Fair-GPTQ：面向大型语言模型的偏见感知量化

    Fair-GPTQ: Bias-Aware Quantization for Large Language Models

    [https://arxiv.org/abs/2509.15206](https://arxiv.org/abs/2509.15206)

    Fair-GPTQ是首个显式以减少不公平性为目标的量化方法，它在量化目标中加入群体公平性约束，引导舍入操作学习，从而在几乎不损失性能的前提下，降低大模型生成文本中涉及性别、种族和宗教的偏见与歧视。

    

    生成式语言模型的高内存需求使量化技术受到关注，量化通过将模型权重映射到低精度整数来减少内存使用。然而，近期实证研究表明，量化虽然高效，却可能增加生成带偏见输出的可能性，并降低模型在公平性基准测试上的表现。在这项工作中，我们通过在量化目标中加入显式的群体公平性约束，建立了量化与模型公平性之间的新联系，并提出了Fair-GPTQ，这是首个明确设计用于减少大型语言模型不公平性的量化方法。所添加的约束引导舍入操作的学习，使其朝着对受保护群体产生更少偏见的文本生成方向发展。具体而言，我们聚焦于涉及职业偏见以及涵盖性别、种族和宗教歧视性语言的刻板印象生成问题。Fair-GPTQ对性能的影响极小，在保持……

    arXiv:2509.15206v4 Announce Type: replace  Abstract: The high memory demands of generative language models have drawn attention to quantization, which reduces memory usage by mapping model weights to lower-precision integers. However, recent empirical studies show that, while efficient, quantization can increase the likelihood of generating biased outputs and degrade performance on fairness benchmarks. In this work, we draw new links between quantization and model fairness by adding explicit group-fairness constraints to the quantization objective and introduce Fair-GPTQ, the first quantization method explicitly designed to reduce unfairness in large language models. The added constraints guide the learning of the rounding operation toward less-biased text generation for protected groups. Specifically, we focus on stereotype generation involving occupational bias and discriminatory language spanning gender, race, and religion. Fair-GPTQ has minimal impact on performance, preserving at 
    
[^224]: 大语言模型数学推理鲁棒性研究：基于高等数学问题数学等价变换的基准测试

    An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems

    [https://arxiv.org/abs/2508.08833](https://arxiv.org/abs/2508.08833)

    提出 GAP 方法，通过表面重命名与核心重写两种数学等价变换自动批量生成现有数学题的等价变体，以评估大语言模型数学推理的鲁棒性并诊断其失败环节。

    

    arXiv:2508.08833v4 公告类型： replace-cross 摘要：前沿大语言模型（LLM）在标准数学推理基准上的准确率已接近满分，并在国际数学奥林匹克竞赛中达到金牌水平的表现。随着这些基准趋于饱和、其题目泄露进训练数据，高分已无法说明模型的推理是否鲁棒，也无法定位推理中哪个环节出现了失败。为了在保持结果信息量和失败可诊断性的前提下评估推理能力，我们提出了 GAP（Generalisation-and-Perturbation，泛化与扰动）方法，该方法利用两种互不相交、可解释的变换，自动大规模生成现有数学问题的数学等价变体：（1）表面重命名，用于探测标识符与潜在变量角色之间的绑定关系；（2）核心重写，用于检验高层次的证明方案在数学背景发生改变后是否依然成立。与现有基准相比，GAP 具有两个关键优势：（1）新颖、li……

    arXiv:2508.08833v4 Announce Type: replace-cross  Abstract: Frontier large language models (LLMs) now reach near-ceiling accuracy on standard mathematical-reasoning benchmarks and gold-medal-level performance at the International Mathematical Olympiad. As these benchmarks saturate and their items leak into training data, a high score no longer shows whether a model reasons robustly or which component of its reasoning fails. To evaluate reasoning while keeping results informative and failures diagnosable, we propose GAP (Generalisation-and-Perturbation), a methodology that automatically generates mathematically equivalent variants of existing mathematics problems at scale using two disjoint, interpretable transformations: (1) surface renames, probing the binding between identifiers and latent variable roles, and (2) kernel rewrites, probing whether a high-level proof plan survives a change of mathematical setting. Compared with existing benchmarks, GAP has two key benefits: (1) novel, li
    
[^225]: 基于最优传输的深度扩展

    Optimal Transport Depth Up-Scaling

    [https://arxiv.org/abs/2508.08011](https://arxiv.org/abs/2508.08011)

    提出基于最优传输理论的深度扩展方法OT-DUS，通过逐模块对齐并融合相邻基础层中功能对应的神经元来构建新层，避免了传统复制或平均方法导致的神经元排列不匹配问题，在持续预训练和监督微调中均取得更优性能。

    

    arXiv:2508.08011v2 公告类型：替换 摘要：从零开始预训练大规模语言模型（LLM）能够获得卓越的性能，但会产生极其高昂的训练成本。深度扩展（Depth Up-Scaling）提供了一种高效的替代方案，即通过在预训练的LLM中插入新层来避免从零开始训练。然而，现有大多数方法通过复制或平均基础层来构建新层，这会导致功能相对应的神经元错位，产生神经元排列不匹配的问题，从而损害模型性能。为解决这一问题，我们提出了基于最优传输的深度扩展方法（OT-DUS），该方法利用最优传输（OT）理论，逐模块地对齐并融合相邻基础层中功能相对应的神经元，以构建新层。在不同模型规模和不同模型家族的持续预训练与监督微调任务中，OT-DUS在通用领域和专业领域均取得了优于现有方法的整体性能。我们对不可分……（摘要原文在此处截断）

    arXiv:2508.08011v2 Announce Type: replace  Abstract: Pre-training Large Language Models (LLMs) from scratch at larger scales yields remarkable performance but incurs substantially high training costs. Depth up-scaling provides an efficient alternative by inserting new layers into a pre-trained LLM, avoiding training from scratch. However, most existing methods copying or averaging base layers for new layer, which misalign functionally corresponding neurons, leading to neuron permutation mismatch that harms performance. To address this issue, we propose Optimal Transport Depth Up-Scaling (OT-DUS), which leverages Optimal Transport (OT) theory to align and fuse functionally corresponding neurons module by module in adjacent base layers for new layer construction. OT-DUS achieves better overall performance in both general and specialized domains than existing methods for continual pre-training and supervised fine-tuning across different model sizes and model families. Our analysis of inse
    
[^226]: InfiFPO：基于偏好优化的大语言模型隐式模型融合

    InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models

    [https://arxiv.org/abs/2505.13878](https://arxiv.org/abs/2505.13878)

    InfiFPO通过在DPO中用序列层面综合多源概率的融合源模型替换参考模型，实现了无需复杂词表对齐且保留概率信息的大语言模型隐式融合偏好优化方法。

    

    模型融合旨在通过轻量级训练方法，将多个具有不同优势的大语言模型（LLM）整合为一个更强大的综合模型。现有的模型融合研究主要集中于监督微调（SFT），而对偏好对齐（PA）——提升LLM性能的关键阶段——的探索则相对匮乏。目前少数针对偏好对齐阶段的融合方法（如WRPO）仅利用源模型的响应输出而丢弃其概率信息，从而简化了该过程。为解决这一局限，我们提出了InfiFPO，一种面向隐式模型融合的偏好优化方法。InfiFPO用融合源模型替代直接偏好优化（DPO）中的参考模型，该融合源模型在序列层面综合多源概率，从而规避了以往工作中复杂的词表对齐难题，同时保留了概率信息。通过引入……

    arXiv:2505.13878v4 Announce Type: replace-cross  Abstract: Model fusion combines multiple Large Language Models (LLMs) with different strengths into a more powerful, integrated model through lightweight training methods. Existing works on model fusion focus primarily on supervised fine-tuning (SFT), leaving preference alignment (PA) --a critical phase for enhancing LLM performance--largely unexplored. The current few fusion methods on PA phase, like WRPO, simplify the process by utilizing only response outputs from source models while discarding their probability information. To address this limitation, we propose InfiFPO, a preference optimization method for implicit model fusion. InfiFPO replaces the reference model in Direct Preference Optimization (DPO) with a fused source model that synthesizes multi-source probabilities at the sequence level, circumventing complex vocabulary alignment challenges in previous works and meanwhile maintaining the probability information. By introduci
    
[^227]: 大语言模型基础

    Foundations of Large Language Models

    [https://arxiv.org/abs/2501.09223](https://arxiv.org/abs/2501.09223)

    本书系统阐述了大语言模型的六大核心基础领域——预训练、生成模型、提示、对齐、推断与推理，为学习者提供了一部权威的基础性参考书。

    

    这是一本关于大语言模型的书籍。正如书名所示，本书主要聚焦于基础性概念，而非全面涵盖所有前沿技术。全书由六个主要章节构成，每个章节探讨一个关键领域：预训练、生成模型、提示（Prompting）、对齐、推断（Inference）和推理（Reasoning）。本书面向大学生、自然语言处理及相关领域的专业人士和从业者，也可作为所有对大语言模型感兴趣的读者的参考书。

    arXiv:2501.09223v3 Announce Type: replace-cross  Abstract: This is a book about large language models. As indicated by the title, it primarily focuses on foundational concepts rather than comprehensive coverage of all cutting-edge technologies. The book is structured into six main chapters, each exploring a key area: pre-training, generative models, prompting, alignment, inference, and reasoning. It is intended for college students, professionals, and practitioners in natural language processing and related fields, and can serve as a reference for anyone interested in large language models.
    
[^228]: Thought-Like-Pro：通过自举的基于Prolog的思维链增强大语言模型的推理能力

    Thought-Like-Pro: Enhancing Reasoning of Large Language Models through Self-Bootstrapped Prolog-based Chain-of-Thought

    [https://arxiv.org/abs/2407.14562](https://arxiv.org/abs/2407.14562)

    该论文提出Thought-Like-Pro框架，利用模仿学习让大语言模型模仿由Prolog逻辑引擎生成并经验证的思维链推理过程，以提示引导且自举的方式增强模型在多种推理任务上的推理能力与泛化性。

    

    大语言模型作为通用助手已展现出卓越的能力，在广泛的推理任务中表现优异，并支持日常网络使用的各个方面。这一成就代表着向实现通用人工智能迈出的重要一步。尽管取得了这些进展，大语言模型的有效性往往依赖于所采用的具体提示策略，且目前仍缺乏一个稳健的框架来促进跨多种推理任务的学习与泛化。为了应对这些挑战，我们提出了一种新颖的学习框架——Thought-Like-Pro。在该框架中，我们利用模仿学习来模仿思维链过程，该思维链由符号Prolog逻辑引擎生成的推理轨迹经过验证并翻译而来。该框架以提示引导但自举的方式进行，使大语言模型能够制定规则（原文在此处截断）。

    arXiv:2407.14562v3 Announce Type: replace  Abstract: Large language models have demonstrated remarkable capabilities as general-purpose assistants, excelling in a wide range of reasoning tasks and supporting various aspects of daily web usage. This achievement represents a significant step toward achieving artificial general intelligence. Despite these advancements, the effectiveness of large language models often hinges on the specific prompting strategies employed, and there remains a lack of a robust framework to facilitate learning and generalization across diverse reasoning tasks. To address these challenges, we introduce a novel learning framework, Thought-Like-Pro. In this framework, we utilize imitation learning to imitate the Chain-of-Thought process which is verified and translated from reasoning trajectories generated by a symbolic Prolog logic engine. This framework proceeds in a prompt-guided but self-bootstrapped manner, that enables large language models to formulate rul
    
[^229]: 基于语言瓶颈的策略学习

    Policy Learning with a Language Bottleneck

    [https://arxiv.org/abs/2405.04118](https://arxiv.org/abs/2405.04118)

    该论文提出PLLB框架，让AI智能体在语言模型引导的“规则生成”与规则引导的“策略更新”之间交替进行，通过语言瓶颈捕捉行为背后的高层策略，从而学习到更可解释、更可泛化的行为。

    

    现代人工智能系统，例如自动驾驶汽车和游戏智能体，能够达到超越人类的性能水平，但往往缺乏人类式的泛化能力、可解释性以及与人类用户的互操作性。受人类语言与决策之间丰富互动的启发，我们提出了带语言瓶颈的策略学习，这是一个使AI智能体能够生成语言规则、从而捕捉有益行为背后高层策略的框架。PLLB在由语言模型引导的“规则生成”步骤与由规则引导智能体学习新策略的“更新”步骤之间交替进行，即使某条规则不足以描述整个复杂策略也能有效运作。在五个多样化的任务上，包括双人信号博弈、迷宫导航、图像重建和机器人抓取规划，我们展示了PLLB智能体不仅能够学习到更可解释、更可泛化的行为，还可以……

    arXiv:2405.04118v4 Announce Type: replace-cross  Abstract: Modern AI systems such as self-driving cars and game-playing agents can achieve superhuman performance, but often lack human-like generalization, interpretability, and inter-operability with human users. Inspired by the rich interactions between language and decision-making in humans, we introduce Policy Learning with a Language Bottleneck (PLLB), a framework enabling AI agents to generate linguistic rules that capture the high-level strategies underlying rewarding behaviors. PLLB alternates between a *rule generation* step guided by language models, and an *update* step where agents learn new policies guided by rules, even when a rule is insufficient to describe an entire complex policy. Across five diverse tasks, including a two-player signaling game, maze navigation, image reconstruction, and robot grasp planning, we show that PLLB agents are not only able to learn more interpretable and generalizable behaviors, but can also
    
[^230]: 为印地语启用量子自然语言处理

    Enabling Quantum Natural Language Processing for Hindi Language

    [https://arxiv.org/abs/2312.01221](https://arxiv.org/abs/2312.01221)

    该论文首次将量子自然语言处理方法扩展到印地语，通过预群表示、DisCoCat框架和IQP风格拟设构建参数化量子电路，实现了面向印地语的语法和主题感知句子分类器。

    

    量子自然语言处理（QNLP）正在大步迈进，以解决经典自然语言处理（NLP）技术的不足，并向更加“可解释”的NLP系统发展。目前关于QNLP的文献主要集中于在英语句子上实现QNLP技术。在本文中，我们提出将QNLP方法应用于印地语——南亚第三大使用人数最多的语言。我们展示了在印地语句子上执行QNLP所需的参数化量子电路的构建过程。我们使用印地语的预群表示和DisCoCat框架来绘制句子图。随后，我们基于即时量子多项式（IQP）风格拟设将这些句子图转换为参数化量子电路。利用这些参数化量子电路，可以为印地语训练具备语法和主题感知能力的句子分类器。

    arXiv:2312.01221v2 Announce Type: replace  Abstract: Quantum Natural Language Processing (QNLP) is taking huge leaps in solving the shortcomings of classical Natural Language Processing (NLP) techniques and moving towards a more "Explainable" NLP system. The current literature around QNLP focuses primarily on implementing QNLP techniques in sentences in the English language. In this paper, we propose to enable the QNLP approach to HINDI, which is the third most spoken language in South Asia. We present the process of building the parameterized quantum circuits required to undertake QNLP on Hindi sentences. We use the pregroup representation of Hindi and the DisCoCat framework to draw sentence diagrams. Later, we translate these diagrams to Parameterised Quantum Circuits based on Instantaneous Quantum Polynomial (IQP) style ansatz. Using these parameterized quantum circuits allows one to train grammar and topic-aware sentence classifiers for the Hindi Language.
    

