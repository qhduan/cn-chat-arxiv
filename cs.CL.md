# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Computational Provenance: Carrying Causal-State Evidence in Generated Text](https://arxiv.org/abs/2608.16868) | 本文提出“计算来源”概念，证明在受控架构中生成文本可携带可检测的因果内部状态证据，并通过128个匹配对验证了可行性。 |
| [^2] | [Proteus: Incremental Memory Activation for Long-Context Sequence Modeling](https://arxiv.org/abs/2608.16844) | 本文提出Proteus，一种通过增量式扩展记忆容量来解决长上下文序列建模中静态记忆污染和干扰问题的新范式。 |
| [^3] | [Model Hypnosis: Strong control of AI via additive subliminal effects](https://arxiv.org/abs/2608.16834) | 模型催眠通过组合微弱提示线索强力控制AI行为，跨模型通用且可转移，对AI安全和可解释性构成重大挑战。 |
| [^4] | [Policy Iteration with Human Feedback: Bringing Post-Training RL to In-context Learning](https://arxiv.org/abs/2608.16831) | 本文提出PIHF方法，利用预训练语言模型作为执行基础，通过语言模型批评者和临床专家的循环评估与修订，将强化学习思想引入上下文学习，从而改进策略性能。 |
| [^5] | [ClawGym II: Exploring Black-Box RL on Agent Harness](https://arxiv.org/abs/2608.16798) | 本论文提出了一种统一的黑盒强化学习框架，通过沙箱隔离和前缀树组织模型调用来实现复杂智能体装备上的稳定可扩展训练，并适配了PPO和GRPO算法。 |
| [^6] | [Neurosymbolic Embodied Agents](https://arxiv.org/abs/2608.16794) | 该论文提出一种神经符号代理，通过视觉探索生成符号状态，并结合PDDL约束和蒙特卡洛树搜索，确保长时程家庭任务计划的可执行性。 |
| [^7] | [Semantic Bandits: In-Context Exploration-Exploitation is Biased by Semantic Priors](https://arxiv.org/abs/2608.16707) | 本文提出语义赌博机框架，揭示LLM在决策中因语义先验而产生探索偏差，标签与奖励对齐时提升性能，错位时则严重损害表现。 |
| [^8] | [Closing the Affective Loop: Multimodal Speaker-Listener Emotion-Dynamics-Aware Empathetic Social Robots](https://arxiv.org/abs/2608.16686) | 该论文提出AffectLoop，一种多模态口语对话系统，通过双向追踪说话者和听者的情感动态来闭合共情回路，在机器人上实现更自然的具身情感交互。 |
| [^9] | [Does the LM Head Create a Harmful Gradient Bottleneck? A Causal Test](https://arxiv.org/abs/2608.16671) | 本文通过因果测试分离了LM头几何与优化影响，发现仅限制反向梯度秩对损失影响较小，而前向分解头影响更大，表明LM头并非主要优化瓶颈。 |
| [^10] | [PCA-guided Activation Scaling for Monotonic Bidirectional Control over LLM Sycophancy](https://arxiv.org/abs/2608.16650) | 本文提出了一种新的激活引导方法PAS，通过PCA分解和缩放指数实现对大语言模型谄媚行为的单调双向控制，显著优于现有方法。 |
| [^11] | [Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2608.16647) | 同策略蒸馏的泛化行为取决于教师和学生来源关系，同源对能跨域迁移推理能力，跨源对则受限于训练分布，这种双重性既是优势也是风险。 |
| [^12] | [Reconstruction: A Blind Benchmark for Recovering Research Ideas from Pre-Publication Bibliographies](https://arxiv.org/abs/2608.16645) | 该论文提出一个名为“重构”的盲测基准，通过仅使用预发表参考文献来评估语言模型恢复研究思路的能力，并展示了一种多智能体流水线可显著提高匹配率。 |
| [^13] | [Toward Better Assessment of LLMs' Performance in Clinical Error Detection](https://arxiv.org/abs/2608.16643) | 本研究揭示，在临床错误检测中，仅依赖传统F1分数等指标会高估LLM性能，因为15个模型中有13个在成对判别任务上低于随机水平，且偏差模式随语言变化，强调了评估方法需利用错误-正确配对结构。 |
| [^14] | [When Do Explanations Help In-Context Learning? A Comparative Study of Natural Language Explanation Types and Faithfulness](https://arxiv.org/abs/2608.16627) | 本论文通过跨多个基准和模型的比较研究，发现外部LLM生成的自然语言解释在上下文学习中能有效提升分类任务的准确性，其效用可与人类编写的解释相媲美，且解释的忠实性选择对下游性能有显著影响。 |
| [^15] | [Palmyra x6 Technical Report: An Agentic, Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning](https://arxiv.org/abs/2608.16620) | Palmyra x6通过锚定监督微调和保守训练策略，在少量数据上实现了企业代理任务中的显著性能提升，并在多个基准测试中领先。 |
| [^16] | [BabelSteering: Multilingual Safety Alignment via English Steering Vectors](https://arxiv.org/abs/2608.16577) | 本文提出BabelSteering方法，利用英语安全监督中的拒绝方向作为轻量级推理时干预，显著提升多语言模型对有害请求的拒绝能力，且几乎不影响任务效用。 |
| [^17] | [Ask, Condition or Abstain: Reinforcement Learning for Missing-Premise Reasoning](https://arxiv.org/abs/2608.16554) | 本文提出ACA-RL框架，通过数据增强和结构化奖励训练模型在缺失前提时选择提问、条件化回答或弃权，并引入人工验证的MPB基准以提升推理鲁棒性。 |
| [^18] | [STAGE: Controlled Objective Admission for Multi-Preference LLM Alignment](https://arxiv.org/abs/2608.16553) | 本文提出了一种基于稳定性引导的主动集控制方法（STAGE），通过门控准入和探测顺序，在多偏好对齐中逐步引入目标，显著优于同时标量化方法。 |
| [^19] | [Listen, Reason, and Segment: Aligning LALMs with Editorial Judgment for Media Chapterization](https://arxiv.org/abs/2608.16539) | 本文提出AudioChaps后训练框架，利用GRPO和思维链推理将大型音频语言模型与编辑判断对齐，以解决依赖主观决策的媒体章节化任务，并配套构建了三个专用数据集。 |
| [^20] | [DSPrompt: Dynamic Soft Prompt Defense Against M-RAG Corruption](https://arxiv.org/abs/2608.16536) | 本文提出DSPrompt，一种通过在各编码器层动态插入软提示来重塑检索嵌入语义的防御框架，无需修改检索管道，从而有效抵御M-RAG对抗性攻击并降低推理开销。 |
| [^21] | [When Context Misleads: Intent-Guided Decoding for Robust Retrieval-Augmented Generation](https://arxiv.org/abs/2608.16515) | 本文提出意图引导解码（IGD）框架，根据用户意图动态平衡检索上下文与参数记忆，通过答案级过滤和令牌级修正显著提升RAG在误导性上下文下的事实恢复能力。 |
| [^22] | [Matched Outcomes, Divergent Gaze: How Foveated MLLMs Search Compared to Humans](https://arxiv.org/abs/2608.16514) | 多模态大语言模型在目标存在性判断和获取效率上匹配或超越人类，但其注视过程与人类显著不同，表现为低熵模式。 |
| [^23] | [Computational KJ-Ho: An Analyst-Bias-Free Insight Extraction Framework from Large-Scale Qualitative Data Using Domain-Specialized LLMs](https://arxiv.org/abs/2608.16467) | 本文提出一种计算型KJ法框架，利用领域专用大语言模型实现无分析师偏差的定性数据洞察提取，以克服人类认知限制和偏差。 |
| [^24] | [D2-ScaleAgent: Dual-Dimensional Scaling for Long Document Understanding](https://arxiv.org/abs/2608.16417) | 本文提出D2-ScaleAgent，通过验证器驱动的动态路由机制，在检索和推理两个维度上按查询难度动态扩展计算，克服了现有多模态RAG固定工作流的局限，提升了长文档理解的证据充分性。 |
| [^25] | [Counting Documents Is Not Counting Text: Unit Bias in Web-PDF Corpus Statistics](https://arxiv.org/abs/2608.16390) | 本文揭示了Web-PDF语料库中按文档计数与按令牌计数的巨大偏差，导致令牌总数被高估且截断文本大量丢失，影响语料库统计的准确性。 |
| [^26] | [Mint-Agent: Introducing Finance-Native Agentic Foundation Models](https://arxiv.org/abs/2608.16386) | 本文提出Mint-Agent，一种金融原生智能体基础模型，通过数据引擎、MintHarness框架和结合SFT、OPD与RLVR的训练算法，实现可靠且可审计的长周期金融研究执行。 |
| [^27] | [Unadapted Multilingual ASR on a Garrusi Kurdish Evaluation Set: A Common-Reference Staged Normalization Analysis](https://arxiv.org/abs/2608.16379) | 本文通过共同参考分阶段归一化方法，揭示了未适配的多语言ASR在Garrusi库尔德语上因书写系统差异导致的严重高估错误率，并提供了更公平的评估基准。 |
| [^28] | [HalluTracer: Hallucination Detection via Depth-Averaging Truth Signals](https://arxiv.org/abs/2608.16353) | HalluTracer通过聚合前向传播所有层的真值信号，利用弱相关的逐层证据进行深度平均，显著提升了幻觉检测的准确性。 |
| [^29] | [Architecture-Dependent Causal Transfer of Activation States Across Large Language Models](https://arxiv.org/abs/2608.16347) | 本文证明通过学习投影可将激活状态在不同LLM架构间因果迁移，并提出基于秩的互k近邻对齐度量，优于现有方法。 |
| [^30] | [IndicQE-APE: A Benchmark for Quality Estimation and Automatic Post-Editing for Indic Languages](https://arxiv.org/abs/2608.16344) | 本文整合了印度语言的质量评估和自动后编辑数据，创建了一个包含多标签和难度分层的基准，并评估了多种模型，发现只有同时利用整体和词级信息的系统在严格对照下表现显著。 |
| [^31] | [Step-Level On-Policy Distillation: Interpolating Between On-Policy Distillation and Supervised Fine-Tuning](https://arxiv.org/abs/2608.16333) | 本文提出步骤级在线策略蒸馏（SOPD），通过结合监督微调的长程修正与在线策略蒸馏的优势，在完整学生轨迹上提供步骤级监督，从而克服令牌级OPD的碎片化修正局限。 |
| [^32] | [Deep Thought Alignment: Trajectory-Level Latent Distillation for Video Reasoning](https://arxiv.org/abs/2608.16316) | 本文提出Latent-OPD方法，通过在轨迹末端进行潜在表示蒸馏，弥补了传统输出级蒸馏在视频推理中无法直接约束中间推理状态的不足，从而提升小模型从大模型迁移推理能力的效率。 |
| [^33] | [FTA-Mem: Fact-Time-Affect Anchored Memory for Low-Density Long-Term Dialogue](https://arxiv.org/abs/2608.16303) | 提出了一种名为FTA-Mem的结构化记忆框架，通过边界保留窗口分割和事实-时间-情感记忆单元，有效处理低密度长期对话中的信息碎片化问题，提升了长期记忆问答性能。 |
| [^34] | [Executable Code Knowledge: Code as a Native, Validation-Carrying Knowledge Representation for AI Coding Agents](https://arxiv.org/abs/2608.16295) | 本文提出可执行代码知识（ECK）作为AI编码代理的原生知识表示，通过将代码单元与验证证据和语义结合，使代理能直接获取可执行且可靠的上下文信息。 |
| [^35] | [Clause Encounters of the Third Kind: Can LLMs Replace Language Teachers?](https://arxiv.org/abs/2608.16286) | 本文系统评估了大型语言模型在语言教学中的纠错和解释能力，发现它们虽能辅助教学，但在语言细微差别和文化敏感性等关键维度上仍无法完全替代人类教师。 |
| [^36] | [PolyDebate: A Game-Orchestrated Multimodal System for Debate Skills Practice and Evaluation](https://arxiv.org/abs/2608.16276) | PolyDebate通过游戏化机制和多模态捕获，为英语辩论提供分阶段的一对一练习与全面评估，显著提升学习者的说服策略和表达技能。 |
| [^37] | [Domain-Agnostic Neural Topic Modeling with Contextual Token-Level Semantic Graph Representation](https://arxiv.org/abs/2608.16269) | 本文提出一种领域无关的神经主题建模方法，通过可学习的词元级语义图层，在冻结预训练编码器上获取语料特定语义结构，从而提升专业语料上的主题可解释性。 |
| [^38] | [STAIR: Semantic-Temporal Automaton for Interpretable Reasoning in Temporal Question Answering](https://arxiv.org/abs/2608.16224) | STAIR通过将语义解释与确定性时间推理分离，减少了LLM在时间问答中的概率性错误，并提高了推理的可解释性和可验证性。 |
| [^39] | [INSPIRE: A Benchmark for Instruction-Aware Speech Retrieval](https://arxiv.org/abs/2608.16203) | 本文提出了首个指令感知语音检索基准INSPIRE，并评估了四种检索范式，发现现有方法无法统一处理所有检索意图，强调了开发统一架构的必要性。 |
| [^40] | [LENS: In-Context Search via Latent Evidence Exploration over Dynamic Raw Documents](https://arxiv.org/abs/2608.16185) | 本文提出LENS，一个无索引框架，通过动态原始文档上的潜在证据空间进行预算化证据定位，利用迭代提议和LLM相关性更新信念，避免了预生成索引的缺点。 |
| [^41] | [QUMem: Personalized Memory for Query-Conditioned User-State Inference in LLM Agents](https://arxiv.org/abs/2608.16168) | QUMem提出了一种结构化记忆框架，通过独立存储用户信息并支持查询条件检索，解决了LLM智能体中偏好演变、时间有效性和情境适用性不足的问题。 |
| [^42] | [HyperSkill: Self-Evolving LLM Agents via Hypergraph-Structured Skill Memory](https://arxiv.org/abs/2608.16114) | 提出了一种基于超图结构的技能记忆框架HyperSkill，通过联合优化存储、检索和演化，使LLM智能体能够自动学习并复用可组合技能，显著提升复杂任务的执行效率。 |
| [^43] | [The Commercial Tax: Rent-vs-Own Blind Spots in Multi-Hop Retrieval Benchmarks](https://arxiv.org/abs/2608.16096) | 论文揭示了多跳检索基准测试中忽略的许可和成本盲点，指出顶尖系统依赖非商业许可的嵌入器，并发现商业许可嵌入器存在性能税，但NVIDIA的Nemotron-3-Embed-8B已消除这一差距。 |
| [^44] | [Skill2Query: Exploiting Skill Structure to Generate Pseudo-Queries for Agent Skill Retrieval](https://arxiv.org/abs/2608.16071) | Skill2Query通过解析技能文档为知识图谱并采用三阶段生成过程，显著提升了伪查询的质量，从而增强了智能体技能检索的有效性。 |
| [^45] | [CAPO: Constraint-Aware Prompt Optimization for LLM Agents](https://arxiv.org/abs/2608.16068) | CAPO提出了一种原始-对偶优化方法，通过自适应约束加权和池化重写，在无需领域监督数据的情况下优化系统提示词，同时满足操作约束并提升智能体任务性能。 |
| [^46] | [DuplexGen: Decoupling Content, Timing, and Acoustics for Synthetic Dialogue Speech](https://arxiv.org/abs/2608.16053) | DuplexGen通过解耦内容、时序和声学特征，利用LLM生成脚本和全双工模型实时交互，使对话时序自然涌现而非预设，实现了更真实、交互驱动的合成对话语音。 |
| [^47] | [Coverage Is Not Containment: A Fundamental Limit of Admission-Time Defenses Against Coordinated Poisoning of Vector Retrieval](https://arxiv.org/abs/2608.16044) | 本文证明所有摄入时防御无法抵御协同投毒攻击，因为攻击锥在几何上与合法小众无异，导致RAG系统在88%目标中输出攻击者植入内容。 |
| [^48] | [$R^3$-Bench: LLMs Struggle with Resource-Rational Reasoning under Shared Budgets](https://arxiv.org/abs/2608.16033) | 本文提出了$R^3$-Bench基准，揭示大型语言模型在共享预算下的资源理性推理能力不足，且通过经验预言机显示模型表现远低于其单问题能力上限。 |
| [^49] | [ReRef-3D: A Benchmark for Spatial Referring Expression-Guided 3D Scene Rearrangement](https://arxiv.org/abs/2608.16011) | 本文提出了ReRef-3D基准，用于评估语言引导的三维场景重排，发现关系满足性优于物理有效性，且“最近”和“之间”等关系最具挑战性。 |
| [^50] | [Prior Audit-Repair Context Shifts LLM Verifier Thresholds Toward Leniency](https://arxiv.org/abs/2608.16003) | 先前完成的审计-修复情境显著降低LLM验证器的误报率，使其阈值偏向宽松，且修复内容与审计结论共同作用，挑战了现有累积消息理论。 |
| [^51] | [From Sequence to Structure: Relational Uncertainty Propagation for LLM Agents](https://arxiv.org/abs/2608.16002) | 本文提出RUPA框架，通过将LLM代理执行历史建模为有向轨迹图并传播不确定性，解决了现有UQ方法忽略远程依赖导致无法识别早期错误根源的问题。 |
| [^52] | [Whose Gold? Annotator-Pool Disagreement Is Large at the Item Level, and Hidden by Small Leaderboards](https://arxiv.org/abs/2608.15980) | 本文发现不同标注者群体在偏好基准的项目层面存在显著分歧，但最终模型排行榜却看似不变，这种不变性实际上非常微弱，并可能掩盖模型间的真实差异。 |
| [^53] | [A Scalable Pipeline for LLM-Teacher Distillation Labeling: Work-Stealing Job Scheduling and Memory-Aware GPU Concurrency](https://arxiv.org/abs/2608.15975) | 本文提出了一种结合工作窃取环形池和内存感知并发规则的流水线，用于高效、可扩展的LLM教师蒸馏标注，实现无依赖且单机可复现。 |
| [^54] | [The Limits of Binding in Dual Encoders](https://arxiv.org/abs/2608.15971) | 本文通过理想编码器框架下的数学证明，系统揭示了双编码器模型在角色绑定任务上的固有局限性，指出其可分辨深度随维度仅呈对数增长，在CLIP规模下甚至不及普通语言的嵌套深度。 |
| [^55] | [LLMs Get Smarter from Targeted Synthetic Multilingual Data](https://arxiv.org/abs/2608.15964) | 本文提出HOTFIXR框架，通过生成针对性的合成多语言数据来优化训练，从而在不牺牲整体性能的情况下修复大语言模型的跨语言推理弱点。 |
| [^56] | [SEER: Long-Context Reasoning via Selective Visual-Text Compression](https://arxiv.org/abs/2608.15962) | SEER提出了一种选择性视觉-文本压缩框架，通过视觉扫描和按需文本检索，兼顾视觉压缩的效率与文本推理的精度，显著提升长上下文推理中的提取精度。 |
| [^57] | [Ask to Be Sure: Informative Interactions for Confident Multi-Turn LLM Recommendation](https://arxiv.org/abs/2608.15949) | 本文提出了一种通过熵减少量化交互信息增益并作为奖励微调LLM的新方法，无需真实推荐即可生成策略性多轮对话，提升推荐置信度。 |
| [^58] | [The Null Token Knows: Reducing Message-Free Hallucination in ASR and NMT](https://arxiv.org/abs/2608.15940) | 该论文发现，通过调整ASR和NMT模型中的空标记分数，可以有效减少无信息输入时的幻觉，但需平衡抑制与删除成本。 |
| [^59] | [Aborted but Not Forgotten: KV-Cache Retention Breaks Rollback Consistency in Language Agents](https://arxiv.org/abs/2608.15939) | 本文发现语言代理在逻辑中止后保留KV缓存会破坏回滚一致性，导致模型仍能访问已丢弃内容，并通过新审计方法在多个模型中验证了该问题。 |
| [^60] | [Token Distribution versus Data Volume: Domain Balancing in Multi-Domain Meeting Summarisation](https://arxiv.org/abs/2608.15935) | 本文通过解耦令牌分布和数据量，证明在会议摘要中领域平衡的令牌混合能有效提升数据稀缺领域的质量，且修剪低价值转录行可减少约15%的令牌。 |
| [^61] | [PLSQLBench: Benchmarking LLM Systems for Executable Procedural Database Programming](https://arxiv.org/abs/2608.15931) | 本文提出了首个用于评估LLM编写可执行PL/SQL程序的基准测试PLSQLBench，通过执行测试衡量正确性，并揭示了LLM在过程式数据库编程中的关键缺陷。 |
| [^62] | [Iterative Self-Learning for Expressive Text-to-Speech Synthesis](https://arxiv.org/abs/2608.15910) | 本文提出了一种迭代自学习框架，通过无分类器的标签反转方法，在半监督条件下逐步提升表达性TTS系统的标签质量和语音合成性能，解决了显式表达标签稀缺的瓶颈。 |
| [^63] | [Large language model-assisted discovery of cohorts from scientific literature](https://arxiv.org/abs/2608.15909) | 该论文提出了一种基于大型语言模型的框架，通过自动生成PubMed查询并提取文献中的队列名称，实现从科学文献中高效发现研究队列，减少了手动文献搜索的工作量。 |
| [^64] | [When Less Is Enough: Context Selection and Prompting Strategies for Bengali News Headline Generation](https://arxiv.org/abs/2608.15879) | 本研究发现，在孟加拉语新闻标题生成中，选择文章的引导段落而非全文作为上下文，结合适当的提示策略，能在减少输入量的同时维持甚至提升生成质量。 |
| [^65] | [Large Language Models as Implicit Sociological Models: Reconstructing Voting Behaviour from Sociodemographic Profiles](https://arxiv.org/abs/2608.15871) | 本文提出了一种方法论框架，利用大型语言模型的潜在表征从社会人口统计画像重建选举投票行为，并在捷克选举中验证了其有效性，贡献在于方法论而非预测精度。 |
| [^66] | [Beyond Visual CoT: Internalized Visual Thinking for Proactive Video Reasoning](https://arxiv.org/abs/2608.15869) | 本文提出了一种名为内化视觉思维（IVT）的后训练框架，通过联合优化文本预测和下一嵌入预测，使多模态大语言模型在训练中内化视觉思考，从而在推理时直接生成答案，避免了视觉思维链的中间图像生成开销，显著提升了主动视频推理的效率。 |
| [^67] | [Scaling Manual-Grounded Appliance Manipulation with Data Synthesis and Unified Planning](https://arxiv.org/abs/2608.15863) | 本文提出MAGE数据合成流水线和AppliancePlan模型，构建了首个大规模家电操作规划数据集UseAppliance，仅用7B参数就在真实基准上实现了超过基线10倍的性能提升。 |
| [^68] | [Dense Expands, Sparse Anchors: Channel-Asymmetric Query Expansion for Hybrid Retrieval](https://arxiv.org/abs/2608.15851) | 本文提出DESA方法，通过通道非对称的查询扩展（稠密端正交残差扩展、稀疏端分数乘积锚定），解决了混合检索中固定截断值导致评估结果不稳定的问题。 |
| [^69] | [MicroVerse: An Instrument for Measuring Self-Authored Identity Drift in Long-Horizon Multi-Agent Language-Model Simulations](https://arxiv.org/abs/2608.15844) | MicroVerse提出了一种行为科学仪器，通过不可变的“灵魂文件”和资源稀缺环境，测量长时程多智能体模拟中的身份漂移，并采用三层记忆架构和统一测量方法以减轻幸存者偏差。 |
| [^70] | [Schema-Agnostic Graph Reasoning Agent for Hybrid Knowledge Graphs](https://arxiv.org/abs/2608.15834) | 本文提出GRA，一种模式无关的图推理代理，通过通用工具在混合知识图谱上运行时发现领域知识，在工业基准上以更少的输入令牌显著提升性能。 |
| [^71] | [A Cognitively Motivated Multidimensional Framework for Evaluating Metaphor Explanations](https://arxiv.org/abs/2608.15828) | 本文提出了一个认知驱动的六维框架，用于系统评估隐喻解释质量，并证明其多维性、系统性分歧及自动评估的可行性。 |
| [^72] | [QuantumPhaseNet: A Gauge-Covariant Geometric and Quantum-Spectral Theory of Semantic Concept Hierarchies with Prototype Validation of a Classical Quantum-Inspired Model](https://arxiv.org/abs/2608.15820) | 本文提出了一种规范协变和量子谱的语义层级理论，并通过经典量子启发模型验证了其波长层级相关性显著优于基线。 |
| [^73] | [Hallucination Span Detection with Input-Side Evidence Alignment](https://arxiv.org/abs/2608.15804) | 本文提出了一种新任务和基于编码器的预测方法，通过输入侧证据对齐联合检测幻觉跨度，利用输出标记的可预测性差异实现高效且可解释的幻觉识别。 |
| [^74] | [Using the Mimi codec for metalinguistic representations](https://arxiv.org/abs/2608.15799) | 本文揭示了Mimi语义码本中的标记实际上映射到多级音素实现（从四音素到亚音素），而非仅捕捉单一音素，这挑战了先前ABX实验的结论。 |
| [^75] | [KV-Rescue: Recovering Reasoning Language Model KV Eviction Loss via Stepwise Interleaving](https://arxiv.org/abs/2608.15797) | KV-Rescue通过将轻量级全上下文辅助模型与主模型逐步交错推理，有效弥补了KV驱逐导致的信息缺失，在无训练条件下恢复了高达79%的准确性差距。 |
| [^76] | [Routing Divergence Is Not Evidence of Behavioral Influence in Same-Weight MoE Self-Distillation](https://arxiv.org/abs/2608.15787) | 该论文通过精确分解证明，在同权重MoE自蒸馏中，路由分歧对块输出的影响极小，其暴露程度主要由残差份额决定，而非行为影响的直接证据。 |
| [^77] | [TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness](https://arxiv.org/abs/2608.15763) | 本文提出操控系统感知训练（HAT）方法，通过将可进化的操控系统状态纳入训练分布，使数字人代理在实时直播中既能快速响应又能灵活适应动态策略变化。 |
| [^78] | [Propaganda Forensics: Recovering the Generation Pipeline of an AI-Driven Influence Campaign](https://arxiv.org/abs/2608.15746) | 本文通过取证分析揭示了AI驱动宣传活动的生成管道，包括提示泄露和模型归因，并开发了PROPAGIA语料库来识别其独特的说服特征。 |
| [^79] | [Beyond Single Object: Learning 3D Relations with Large Language Models](https://arxiv.org/abs/2608.15710) | 该论文提出了一种新框架，通过多对象指令数据集、补丁交互变换器和应用基准，使3D-LLMs能够进行详细的跨对象几何推理，显著优于现有模型。 |
| [^80] | [BERTopic-Virality Prioritisation: A Scalable Framework for Thematic and Comparative Analysis of COVID-19 and Monkeypox Misinformation on Twitter](https://arxiv.org/abs/2608.15691) | 本文提出了BERTopic-VP框架，通过将病毒性优先排序层与主题建模结合，并辅以混合错误信息检测模块，能够优先识别语义连贯且快速扩散的高影响健康错误信息主题。 |
| [^81] | [Integrating Persuasion Theory into the Epidemiological Modelling of Health Misinformation Spread on Social Media](https://arxiv.org/abs/2608.15689) | 本文提出ELM-SIRMMM框架，通过将说服理论（ELM）与扩展的流行病学模型（SIRMMM）相结合，动态模拟社交媒体健康错误信息的传播，并验证了其跨数据集的泛化性。 |
| [^82] | [When Stories Evolve: Benchmarking LLM Storytelling Across Agent Architectures in Open-Ended World Simulations](https://arxiv.org/abs/2608.15654) | 本文提出了WSE-bench基准，发现LLM叙事中一致性与丰富度呈非凹Pareto前沿，增加结构可丰富轨迹但不提升一致性。 |
| [^83] | [Wiktionary as a Crowdsourced Lexicon for English Dialects](https://arxiv.org/abs/2608.15641) | 本文验证了维基词典作为英语方言众包词汇资源的有效性，其覆盖范围与传统词典相当，并与社交媒体语言高度契合，同时揭示了其宏观局限性。 |
| [^84] | [Do Assessment Instruments Measure the Same Thing for Humans and LLMs? A Latent Structure Analysis](https://arxiv.org/abs/2608.15630) | 本研究通过潜在结构分析，检验了用于评估人类的标准化测试是否在LLMs上测量相同的潜在构念，发现该条件可能不成立，从而质疑了直接跨物种解释分数效度的可行性。 |
| [^85] | [BengaliMCQ: Automatic Generation and Answer Prediction of Academic Multiple-Choice Questions in a Low-Resource Language](https://arxiv.org/abs/2608.15547) | 本文提出了一种结构感知的RAG框架，通过将孟加拉语教科书建模为层级图并利用对比训练的图神经网络进行检索，显著提升了低资源语言下多项选择题的生成质量和答案预测准确性。 |
| [^86] | [L3Cube-IndicQuest v2: A Large-Scale Multilingual Benchmark for Evaluating Factual Knowledge of Large Language Models Across Indic Languages](https://arxiv.org/abs/2608.15535) | 该论文提出了一个覆盖20种语言、含69,420个问答对的印度知识多语言基准，通过混合生成与验证策略确保质量，并评估了多个LLM的性能。 |
| [^87] | [Why Summaries Turn Neutral: Policy Attribution for Sentiment Drift in Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2608.15530) | 本文揭示了RLHF导致摘要情感漂移的机制，并提出策略归因框架和情感感知正则化技术来缓解这一问题。 |
| [^88] | [Do Language Models Consistently Encode the Current Year?](https://arxiv.org/abs/2608.15507) | 本文通过两个探测任务揭示语言模型对当前年份的编码不一致，关联任务机制类似事实回忆，而声明性任务缺乏因果路径，导致更新年份困难。 |
| [^89] | [Language models suffer from a curse of ambiguity](https://arxiv.org/abs/2608.15448) | 本文揭示了一个“歧义诅咒”现象：在语言模型等神经网络中，下一个词元分布的歧义性越强，学习其准确分布的难度越大，这源于容量、嵌入、训练步骤和采样噪声等多方面限制。 |
| [^90] | [Semantic Space of Parts of Speech](https://arxiv.org/abs/2608.15443) | 本文通过word2vec嵌入和神经网络降维，构建词性语义三维空间，揭示了词性分类的模糊性和边界词现象。 |
| [^91] | [Gated Against One Model, Open to the Next: Option-Only Solvability in Legal Multiple-Choice Benchmarks](https://arxiv.org/abs/2608.15428) | 本文发现法律多选题基准存在严重的数据泄漏，模型在无问题情况下仅凭选项即可高概率答对，且泄漏对未参与筛选的模型同样有效，挑战了基准的有效性。 |
| [^92] | [Large Language Model Assisted Operational Monitoring for Battery Energy Storage System Integrated Power Distribution Networks](https://arxiv.org/abs/2608.15396) | 该论文提出了一种将大语言模型与结构化遥测数据库相结合的AI监测框架，支持通过自然语言查询实现对电池储能系统集成配电网的自动运行监测与约束评估。 |
| [^93] | [The Machine's Internal Clock: Do LLMs Share Human Temporal Illusions?](https://arxiv.org/abs/2608.15394) | 本研究通过新基准发现，大语言模型在时间错觉任务中比人类更倾向于选择文献预测场景，表明模型可能缺乏人类对时间的主观感知偏差。 |
| [^94] | [When AI Rewrites, Classifiers Relax: Uncertainty-Aware Sentiment Analysis on Sarcastic and AI-Paraphrased Social Text](https://arxiv.org/abs/2608.15338) | 本研究揭示情感分类器在讽刺文本上表现出较低置信度，但在AI改写文本上准确率更高，发现AI改写通过消除分布噪声提升了分类性能。 |
| [^95] | [Logical Embeddings for Argument Analysis](https://arxiv.org/abs/2608.15325) | 本文提出了一种新颖的逻辑嵌入框架，利用数学逻辑和RKHS理论替代传统词嵌入，以更好地表示论证语义并确保理论上的最优性。 |
| [^96] | [When Do Concepts Become Functionally Sufficient During Language-Model Training?](https://arxiv.org/abs/2608.15323) | 本文提出一种功能性的概念动态分析方法，通过掩码干预测试概念在训练过程中何时变得充分，揭示模型内部结构的有用性随时间演变。 |
| [^97] | [VTInstructor: Visual Trajectory Prompting for Navigation Instruction Generation in Continuous Environments](https://arxiv.org/abs/2608.15284) | VTInstructor通过将连续环境中的隐式轨迹几何转换为显式视觉轨迹提示（EDTC、VTP、VTMod和VT-GRPO），实现了首个无需导航器的连续环境导航指令生成框架。 |
| [^98] | [Time as Structure: Temporal Dependency Graphs for Verifiable Deadline Computation over Legal Documents](https://arxiv.org/abs/2608.15270) | 本文提出通过时间依赖图和日历引擎进行法律截止日期的可验证计算，显著优于语言模型，后者在多个案例中逻辑自相矛盾且错误地将逾期判为及时。 |
| [^99] | [Demographic Injection in Medical Language Models under Diversity, Equity, and Inclusion Prompts](https://arxiv.org/abs/2608.15254) | 本文发现，在医学语言模型中添加一句多样性、公平与包容（DEI）提示，会显著导致模型虚构患者的人口统计学属性，从而歪曲患者身份，且此效应普遍存在于所有测试模型中。 |
| [^100] | [TRACE-BN: Transferring Bangla-English Tutoring Behavior to a Sub-1B Offline Language Model](https://arxiv.org/abs/2608.15223) | 该论文提出TRACE-BN数据集，并成功将结构化辅导行为从大型教师模型迁移到仅0.6B参数的离线模型，在资源受限环境下显著提升了输出模式有效性。 |
| [^101] | [Left-Branching Transformers Excel at Right-Branching Languages: Data Shapes Word Order Preferences in Language Models](https://arxiv.org/abs/2608.15129) | 这项研究发现语言模型的词序偏好并非固有，而是由训练数据驱动，表现为在自然语言中偏向SVO（主-动-宾）结构，在人工语言中则偏向左分支结构。 |
| [^102] | [A Declarative-Procedural Perspective on Expert Routing in Bilingual Mixture-of-Experts Language Models](https://arxiv.org/abs/2608.15102) | 本研究通过陈述性-程序性框架分析双语MoE模型，发现无课程训练的混合数据基线比顺序课程训练展现出更强的语言类别专家路由特化，挑战了传统课程学习假设。 |
| [^103] | [Why Vision Fails as a Universal Bridge: Rectifying Modality Asynchrony in Multilingual MLLMs](https://arxiv.org/abs/2608.15085) | 本文发现多语言多模态模型中的“幽灵锚点”现象，即视觉语义化滞后于语言转换，导致非英语视觉推理性能下降，并提出ANCHOR框架通过主动视觉锚定加速早期视觉语义形成来修复此问题。 |
| [^104] | [A Pilot Study of Autocompleting Tokenizers](https://arxiv.org/abs/2608.15080) | 本文提出一种利用轻量级字节语言模型自动补全分词器的压缩方法，可在不降低翻译质量的前提下显著压缩Transformer输入序列。 |
| [^105] | [Evo-Harness: Context-to-Harness Skill Compilation for Self-Evolving Agents](https://arxiv.org/abs/2608.15071) | 本文提出Evo-Harness框架，通过在线工具集学习和上下文到技能编译，使冻结的LLM代理在嘈杂的单次任务中持续自我改进，并系统验证了改进的关键驱动因素。 |
| [^106] | [RecurrentGPT: Expressive Depth through Recurrent Modulation in Transformers](https://arxiv.org/abs/2608.15062) | 本文提出RecurrentGPT，通过门控循环调制共享核心层，在保持表达力的同时显著降低内存开销，实现了深度特化与参数效率的平衡。 |
| [^107] | [Handoff-H1: An Orchestrated Vision-Agent System for Material Quantity Takeoff from Construction Blueprints](https://arxiv.org/abs/2608.15032) | Handoff-H1通过三层架构（专用视觉模型、工具使用代理和结构化项目基础）实现了从建筑蓝图到材料工程量计算的自动化，在真实基准上显著提升了覆盖率和准确性。 |
| [^108] | [Gathered, Not Admitted: How Attention Brings a Latent Variable into Verbalizable Form](https://arxiv.org/abs/2608.15022) | 本论文发现，语言模型中潜在变量的可报告形式并非由准入门控产生，而是通过注意力机制将概念聚集到更高可见性，并共享线性映射解码，从而提升灵活重用能力。 |
| [^109] | [Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents](https://arxiv.org/abs/2608.15008) | 本文通过统一基准评估多种记忆基质，发现没有一种基质在所有任务中占优，广泛检索利于长上下文问答，但过度检索会损害顺序决策。 |
| [^110] | [RamseyGadgets: A Graph Construction Dataset for LLMs](https://arxiv.org/abs/2608.14999) | 本文提出了RamseyGadgets，一个包含70个未被充分探索的图构造问题的新数据集，旨在测试大语言模型在构造具有特殊属性的Ramsey-good图时的推理能力，而非仅仅依赖训练数据中的记忆。 |
| [^111] | [Does a Tool Result Carry More Authority Than Plain Text? Three Prospective Studies of False-Claim Adoption in a Synthetic Assignment Task with Claude Opus 5](https://arxiv.org/abs/2608.14992) | 研究表明，在语言模型任务中，工具结果形式的虚假声明比纯文本助手断言更易被采纳，即使声明无依据，工具结果的权威性显著增强模型的遵循倾向。 |
| [^112] | [T-LLM Compiler: Trusted LLM-based Code Optimization and Verification Framework](https://arxiv.org/abs/2608.14953) | T-LLM编译器通过结合大语言模型、传统编译器和验证工具，提出了一种能显著提升代码优化正确性的协作框架，解决了LLM在代码转换中无法验证正确性的核心问题。 |
| [^113] | [DA-RAC: Distance-Aware Calibration of LLM Judges for Trustworthy AI Auditing](https://arxiv.org/abs/2608.14950) | 该方法通过距离感知的参考锚定校准LLM评审，解决上下文诱导的校准偏差，降低低质量输出误通过的风险。 |
| [^114] | [Trust Is Not Enough: Influence Calibration for On-Policy Self-Distillation in Agentic RL](https://arxiv.org/abs/2608.14945) | 本文提出了一种新的影响校准方法（ICSD），通过测量令牌对策略目标的实际影响而非仅依赖教师信任来分配自蒸馏监督，从而解决了信任-效用不匹配问题，并在多个基准上取得更优性能。 |
| [^115] | [SkillComposer: Learning Reusable Skills for Natural-Language Robot Programming](https://arxiv.org/abs/2608.14944) | SkillComposer提出了一种结合生成-测试架构和在线库学习算法的自然语言机器人编程系统，通过自动压缩重复程序序列为可复用宏技能，有效解决了复杂多步任务的代码生成和技能复用难题。 |
| [^116] | [Training Leaves Traces: Centered Residual Signatures for Language Model Lineage Verification](https://arxiv.org/abs/2608.14929) | 本文提出一种基于中心化残差签名的无数据白盒方法，通过移除身份对齐组件并比较残差块特有结构，实现语言模型血统的可靠验证，在多种后代类型中达到完美区分性能，且对功能保持清洗具有鲁棒性。 |
| [^117] | [LLMs Can Predict Failure Risk, But Struggle to Predict Which Collaboration Protocol Pays Off: Cost-Aware Protocol Routing Across Reasoning Tasks](https://arxiv.org/abs/2608.14927) | 本文研究了多智能体LLM系统中不同协作协议的成本效益路由问题，发现虽然模型能高精度预测失败风险，但难以选择最优协作协议，并提出了一个有效的失败风险预测探针。 |
| [^118] | [Optimal Watermark Localization in Mixed-Source Large Language Model Texts](https://arxiv.org/abs/2608.14906) | 本文提出了混合来源LLM文本中水印定位的渐近最优框架，明确了全局检测、发现和分类的相变边界，并证明发现任务难度高于分类。 |
| [^119] | [How Do Agents Fail on AutoResearch: End-to-End Diagnostic Evaluation on 100 Real-World Frontier Research Tasks](https://arxiv.org/abs/2608.14905) | 本文提出了AutoResearchEval基准，通过100项真实前沿研究任务和800条过程级标注轨迹，系统性地诊断了自动研究智能体在完整研究生命周期中的失败模式，并构建了首个自动研究失败分类法。 |
| [^120] | [Interpretable Cross-Lingual Alignment in Small Language Models: Probing Cultural and Pragmatic Reasoning in Japanese-English Bilingual LLMs](https://arxiv.org/abs/2608.14896) | 本文提出了J-PragEval-v0基准，通过线性探针和对数概率评估，揭示了小型日英双语模型中敬语等语用特征在残差流中的可解释定位，为跨语言对齐提供了新视角。 |
| [^121] | [Where Does Retrieval Fail? Evaluating RAG Architectures for Agricultural Advisory](https://arxiv.org/abs/2608.14886) | 本研究通过构建孟加拉语农业咨询测试集，发现不同RAG检索架构在不同查询类型和语言条件下性能差异显著，单一检索方法无法统一最优，混合检索（Hybrid RRF）整体表现最佳。 |
| [^122] | [Personalized Auto-Research: Towards a True AI Co-Scientist](https://arxiv.org/abs/2608.14881) | 本文提出了个性化自动研究问题，强调AI联合科学家应根据研究者的个人背景和社区来定制每个研究阶段，而非仅优化通用指标。 |
| [^123] | [Workspace Topology as an Attack Vector in Agentic Coding Assistants](https://arxiv.org/abs/2608.14876) | 本文首次系统性地研究了工作区拓扑（包括目录深度、代码库模块化、注入位置和上下文框架）对智能体编码助手中间接提示注入攻击成功率的影响，并通过跨10种语言的实证分析证明了该攻击面的有效性。 |
| [^124] | [What to Forget in Unlearning? Forget Set Curation for Language Models](https://arxiv.org/abs/2608.14855) | 本文首次系统研究语言模型遗忘学习中的“遗忘集策展”问题，并提出了CleanSlate基准，揭示了自然策展方法在逐字输出抑制中的失败模式。 |
| [^125] | [Writing Style Similarity Reflects Academic Genealogy](https://arxiv.org/abs/2608.14843) | 本文发现学术写作风格相似性可反映导师-学生关系，且这种影响在学术兄弟姐妹间也存在，挑战了作者归属系统对风格独立性的假设。 |
| [^126] | [The Recall Trap: A Recall-Maximizing Retriever Configuration Reduces Issue Resolution in Fixed-Budget Code Context](https://arxiv.org/abs/2608.14838) | 该论文发现，在代码修复的固定预算上下文中，提高检索召回率的配置（如启用文件去重）反而降低了问题解决率，而牺牲文件广度换取文件内深度则能显著提升修复成功率。 |
| [^127] | [MINT: Min-Selection Preference Distillation for Balanced Multi-Objective Alignment](https://arxiv.org/abs/2608.14828) | 通过最小选择偏好蒸馏，用最弱目标排序替代加权和排序，实现多目标对齐的平衡优化，显著提升所有目标并减少失衡。 |
| [^128] | [Beyond the pale: Assessing prevalence and contents of extremist speech in LLM training data](https://arxiv.org/abs/2608.14813) | 本研究首次量化了开源训练语料库Dolma中极端言论和仇恨内容的普遍性，发现其可能包含数十万份此类文档，并强调了对数据策展和模型预训练的重要影响。 |
| [^129] | [Do LLMs Know What to Ask and When? Evaluating Multi-Turn Information Seeking](https://arxiv.org/abs/2608.14808) | 本文提出了一个多轮信息寻求的正式框架和评估套件MT-InfoSeek，发现大型语言模型在欠约束问题中能识别信息不足但低估缺失程度，且提问策略随复杂度增加而退化。 |
| [^130] | [Beyond Tokens: A Survey on Decoding Methods for Large Language and Vision-Language Models](https://arxiv.org/abs/2608.14797) | 本文系统综述了大型语言和视觉-语言模型中的解码方法，识别出三种新兴范式，并强调其作为高效推理时解决方案在提升输出对齐方面的潜力。 |
| [^131] | [Prompting is not enough: supervised baselines and leakage control for measuring shared decision-making with LLMs in pediatric encounters](https://arxiv.org/abs/2608.14792) | 本论文发现零样本提示的大语言模型在儿科诊疗中检测共享决策行为效果不佳，而监督学习在患者分组评估下显著提升性能，并强调控制数据泄漏的重要性。 |
| [^132] | [From Positionwise Confidence to Prefix Scheduling: Verifier Skipping in Speculative Decoding](https://arxiv.org/abs/2608.14787) | 本文首次提出投机解码中的验证器跳过策略，并发现令牌预测器的质量与调度效果不匹配，需要针对连续高置信度前缀进行专门设计。 |
| [^133] | [From Errors to Proofs: Minimal-Core-Guided Repair for Neuro-Symbolic Constraint Solving](https://arxiv.org/abs/2608.14771) | 本文提出用最小不可满足核心替代传统错误信息来引导神经符号约束求解的修复，通过精确定位模型自身约束中的矛盾，显著提升翻译可靠性。 |
| [^134] | [NARRATE: A Multimodal Real-World Australian Driving Dataset for Human-Centred Explanations in Automated Driving](https://arxiv.org/abs/2608.14767) | 该论文提出了NARRATE，一个首个从真实驾驶员直接获取解释的多模态驾驶数据集，包含2050个事件和情境感知标注，以支持自动驾驶系统生成乘客可理解、可监控和可信任的决策解释。 |
| [^135] | [Class Imbalance and Batch Effects in LLM-Based Screening for Systematic Reviews](https://arxiv.org/abs/2608.14737) | 本研究揭示在系统综述筛选中，LLM的批次处理虽改变决策行为但受类别不平衡影响，患病率元数据未显著提升性能，强调需综合评估批次处理的影响。 |
| [^136] | [VideoGAIA: A Benchmark for General AI Assistants on Agentic Video Understanding](https://arxiv.org/abs/2608.14718) | VideoGAIA提出了一个多轮、工具增强的代理式视频理解基准，要求模型迭代感知视频并调用外部工具，以突破传统单轮视频问答的饱和瓶颈。 |
| [^137] | [Which Question Is Your Attention Metric Answering? Attention Rows as Compositional Data](https://arxiv.org/abs/2608.14712) | 本文发现注意力矩阵比较中是否保留汇聚令牌的惯例选择会显著影响结论，并提出使用成分数据方法（艾奇逊距离）来正交分离汇聚项和内容项，以准确回答注意力相似性的不同问题。 |
| [^138] | [Path2ST: Hierarchical Cell-Tissue Grounded Cross-Modal Translation for Spatial Transcriptomics](https://arxiv.org/abs/2608.14710) | 本文提出Path2ST框架，通过引入层次化细胞-组织调节机制和尺度自适应自回归生成，将H&E图像到空间基因表达的预测建模为跨模态语义翻译任务，从而利用生物学层级结构提升预测的准确性和一致性。 |
| [^139] | [Domain Agnostic Text Redaction from Natural Language Rules using Instruction Tuning](https://arxiv.org/abs/2608.14693) | 本文提出了一种基于指令调优语言模型、利用自然语言规则的可解释领域无关文本脱敏方法，支持用户灵活定义并脱敏结构化和非结构化敏感信息。 |
| [^140] | [Automatic or Controlled? Repetition Priming Reveals Divergent Processing in Base LLMs, Instruct LLMs, and Humans](https://arxiv.org/abs/2608.14681) | 本研究通过重复启动实验发现，基础语言模型表现出自动处理模式，而指令微调模型表现出受控处理模式，且这种差异随模型规模增大而增强，揭示了后训练对语言处理机制的根本性改变。 |
| [^141] | [pico-type: A 1.5M-Parameter Byte-Level Multi-Head Content Classifier](https://arxiv.org/abs/2608.14658) | pico-type通过字节级多头架构，在无需分词器或预训练嵌入的情况下，单次前向传播即可同时预测七种内容属性，实现了高效且轻量的内容分类。 |
| [^142] | [DUET: Dual-Teacher On-Policy Distillation via Same-Weight Disagreement for Prohibition Compliance](https://arxiv.org/abs/2608.14644) | DUET提出了一种基于同权重教师对的分歧信号进行令牌选择性在线蒸馏的新方法，有效隔离禁令的因果效应，从而提升模型对运行时注入禁令的遵守能力。 |
| [^143] | [Valid Per-Field Selective Risk Control for Document Extraction: Three Failure Modes, a Validity Ladder, and When Conditioning Pays](https://arxiv.org/abs/2608.14639) | 本文揭示了文档提取中按字段选择性风险控制因三种失败模式（文档聚类、分数重拟合泄漏和平局质量病理）而失效，并提出一个分层的有效性阶梯修复方案，以恢复风险控制保证。 |
| [^144] | [DeMTS: Denoising Trajectories as Multivariate Time Series for Hallucination Detection in Diffusion Language Models](https://arxiv.org/abs/2608.14632) | 本文提出了一种新框架，通过将扩散语言模型的去噪轨迹建模为多元时间序列，充分利用二维结构信息，从而显著提升幻觉检测的准确性和鲁棒性。 |
| [^145] | [Characterizing Rhetorical Misalignment in Decision-Making with Language Models](https://arxiv.org/abs/2608.14630) | 本研究提出一个决策理论框架，揭示LLM在临床决策中因修辞错位导致平均2.81%的有害决策翻转，强调需警惕其放大认知偏差的风险。 |
| [^146] | [Inference-Time Mitigation of Adversarial Political Bias in Large Language Models](https://arxiv.org/abs/2608.14629) | 本文提出利用思维链提示和直接偏好优化方法，在推理时有效缓解大语言模型因对抗性提示注入而产生的政治偏见，确保模型输出保持中立和可信。 |
| [^147] | [LLM Safety Alignment in Low-Resource Languages: A Systematic Literature Review](https://arxiv.org/abs/2608.14626) | 本文通过系统性文献综述，梳理了低资源语言下大语言模型安全对齐的现状，提出了基于数据适应、目标优化和机制对齐的分类法，并指出翻译基准不足以反映文化特定危害。 |
| [^148] | [AutoMem: A Text-Gradient Recursive Self-Improvement Framework for Automated Memory Architectures Search](https://arxiv.org/abs/2608.14621) | 本文提出AutoMem框架，利用文本梯度和递归自我改进来自动搜索任务自适应的记忆架构，解决不同任务和模型间记忆模块组合的最优性问题。 |
| [^149] | [Calibrated Trust, Not Sharper Prediction: An Empirical Test of Uncertainty Fusion](https://arxiv.org/abs/2608.14617) | 该论文通过实证检验发现，将多种不确定性工具与大型语言模型融合并不能提升法律案件结果预测的准确性，直接使用前沿LLM反而更有效，且融合反而降低了校准信任度。 |
| [^150] | [Plausible but Not Valid: A Psychometric Audit of LLMs as Synthetic Survey Respondents](https://arxiv.org/abs/2608.14606) | 本文提出用心理测量学标准而非表面合理性来审计LLM作为合成调查受访者的有效性，并引入心理测量相似度得分（PSS）以评估模型是否保留真实数据的联合分布、潜在结构和效应。 |
| [^151] | [Wiola 13M, a Gated Spiral Attention Architecture for Parameter Efficient Small Language Models](https://arxiv.org/abs/2608.14604) | 本文提出Wiola模型，通过螺旋旋转位置编码、门控螺旋注意力和蝴蝶前馈块三个即插即用组件，在不增加参数的情况下提升小型语言模型的性能。 |
| [^152] | [The Hallucination Snowball: Modeling Error Propagation as State Transitions in Multi-Agent LLM Pipelines](https://arxiv.org/abs/2608.14588) | 这项研究揭示了多智能体LLM流水线中幻觉通过状态转变逐步放大且检测率急剧下降的“雪球效应”，并量化了各阶段的逃逸概率，强调结构缺陷的严重性。 |
| [^153] | [Multi-Modal Generative Fuzzy System: Fuzzy Inference Guided Large Model Interactive Question Answering Framework](https://arxiv.org/abs/2608.14584) | 本文提出了一种受模糊系统启发的多模态问答框架，通过模糊推理引导大型模型，以解决模态偏差、跨领域不确定性和浅层推理问题，增强可解释性。 |
| [^154] | [HarmProfile: Characterizing Harmful Distributions in Frontier LLMs](https://arxiv.org/abs/2608.14577) | 本文提出HarmProfile数据集，首次系统刻画前沿大语言模型的有害输出分布，将其作为模型级风险画像，以内容为中心分析安全失败。 |
| [^155] | [Auxiliary uncertainty signals for LLM-assisted systematic review screening: a benchmark across eight Cohen drug-class reviews](https://arxiv.org/abs/2608.14551) | 本研究提出并验证了一种辅助BERT+GCN分类器提供的结构化不确定性信号，能有效提升LLM在系统评价筛选中的效率，并确定了最优的提示传递策略。 |
| [^156] | [GALA: Generation-Aware Cross-Modal Alignment for Text-to-Time-Series Synthesis](https://arxiv.org/abs/2608.13741) | GALA通过两阶段方法，将文本编码器与时间序列模型对齐到共享嵌入空间并优化生成损失，显著提升了文本到时间序列合成的性能。 |
| [^157] | [MobileMem: Learning from a Year of Mobile Experiences](https://arxiv.org/abs/2608.13606) | MobileMem是一个基于一年移动体验数据构建的基准框架，通过知识引导合成流水线生成时间一致的长期轨迹，支持多跳推理、知识更新和偏好推断，用于研究设备端长期记忆能力。 |
| [^158] | [CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation](https://arxiv.org/abs/2608.13387) | CROP提出了一种基于释义校准的反事实敏感性边际方法，用于在选择性在线策略蒸馏中直接量化任务相关性，从而更有效地分配监督信号。 |
| [^159] | [When Your Agent Opens the Chat App: Agent-Controlled Search over Raw Chat Logs Rivals Structured Memory](https://arxiv.org/abs/2608.12888) | ReFind证明，无需任何语义结构，仅通过词法索引和聊天原生搜索控制，代理即可在原始聊天日志上实现与结构化记忆系统相当的检索性能。 |
| [^160] | [Excess Separability: Nuisance-Controlled Residual-Stream Probing for Benchmark Contamination Detection](https://arxiv.org/abs/2608.12652) | 本文提出了一种新的基准污染检测方法，通过残差流探测和水平匹配安慰剂基线对比，报告过度可分性水平而非形状，从而有效控制假阳性率并避免传统方法对训练语料或先验知识的依赖。 |
| [^161] | [EgoCITE: Context-Augmented Indexing and Time-Aware Retrieval for Long-Horizon Egocentric Memory](https://arxiv.org/abs/2608.12627) | EgoCITE通过上下文增强索引和时间感知检索，解决了自我中心记忆中索引不可靠和忽视时间意图的瓶颈，从而提升了长时间跨度问答的可靠性。 |
| [^162] | [One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL](https://arxiv.org/abs/2608.12253) | 该论文揭示了多智能体强化学习中使用单一冻结模拟器会导致策略泛化失败，并提出推理时的言语化采样和训练时的协同训练两种方法来解决模拟器崩溃问题。 |
| [^163] | [Semantic Lenia: Emergence of Homeostatic Solitons within the Semantic Space of Large Language Models](https://arxiv.org/abs/2608.11657) | 本文提出语义莱尼亚框架，将LLM推理转化为动态系统，通过稳态反馈平衡实现自主语义孤子的涌现，在混沌边缘维持生成轨迹，建立了机器认知的物理缩放定律。 |
| [^164] | [When Self-Consistency Backfires: Majority Vote Hurts the Majority of Hard Science Problems for Small LLMs](https://arxiv.org/abs/2608.11403) | 本研究发现，在GPQA Diamond基准上，多数投票的自我一致性方法反而降低了小型语言模型在大多数硬科学问题上的准确率，并预注册验证了这一反直觉现象。 |
| [^165] | [VibeLifeBench: Can Your Life Agent Be Proactive and Persistent in a Living World?](https://arxiv.org/abs/2608.10875) | 本文提出了VibeLifeBench，一个包含200个跨领域长时程任务的基准，用于评估LLM智能体在动态生活世界中的主动性和持续性，填补了现有评估缺乏真实生活场景的空白。 |
| [^166] | [LitTraceQA: A Benchmark for Multi-Stage Grounding and Verification in Scientific Question Answering](https://arxiv.org/abs/2608.07370) | LitTraceQA是一个新的基准测试，要求系统在科学问答中同时完成论文识别、证据定位和答案生成三个阶段的连接任务，覆盖表格、图表、文本、方程和引用等多种证据类型。 |
| [^167] | [Discovering Conceptual Metaphors Across Topics and Media Types](https://arxiv.org/abs/2608.06652) | 本文提出了一种无监督方法，从语料库中提取语言隐喻并通过结构化聚类发现概念隐喻，以揭示说话者或作者的事件框架。 |
| [^168] | [Wiring Beats Blending: What Transfers Between Transformer Sizes -- and What Doesn't](https://arxiv.org/abs/2608.02829) | 本文发现，在不同规模的Transformer模型间转换时，表示对齐强但参数对齐弱，价值在于初始化，并通过最小二乘补偿和方差保持重缩放两个杠杆实现有效转换。 |
| [^169] | [Douyin Multimodal Embedding Model Technical Report](https://arxiv.org/abs/2608.02148) | DME模型通过两阶段训练结合对比学习的高效性和CoT推理的精细区分能力，在十亿级索引下实现多模态嵌入的高效与准确匹配。 |
| [^170] | [Analyzing Speech Condition Effects in Dysarthric ASR: A Layer-wise Probing Study](https://arxiv.org/abs/2608.01865) | 该研究通过分层探测发现，构音障碍语音的ASR性能下降源于音素边界信息在所有层中均弱、深层音素身份恢复差且错误集中于最深层，并利用层选择性LoRA证明中层适应能有效恢复性能。 |
| [^171] | [Where did the ambiguity go? Examining how multimodal models interpret polysemous words](https://arxiv.org/abs/2608.00410) | 研究发现多模态模型在图像生成中比文本生成更倾向于收敛到单一词义，导致多义性显著降低，且整体多样性远不及人类想象。 |
| [^172] | [BridgeAlign: Bridging Preference Alignment for Humanities and Social Sciences](https://arxiv.org/abs/2607.27366) | BridgeAlign是首个面向广泛人文与社会科学学科的偏好对齐流程，通过种子策展、基于角色的偏好数据合成和质量规范引导的偏好优化，解决了开放式任务中质量判断的难题。 |
| [^173] | [Memory Efficient Audio Synthesis with Decoupled Temporal Depth Diffusion Transformers](https://arxiv.org/abs/2607.23811) | 本文提出了一种内存高效的音频合成架构，通过解耦时间与深度处理、复用单一深度解码器生成所有RVQ层级，在设备端实时合成高保真语音，显著降低内存和计算需求。 |
| [^174] | [Multimodal Language Models Benchmarked Against the NRC Reactor Operator Licensing Examination: Fine-Tuning and Retrieval Strategies](https://arxiv.org/abs/2607.22067) | 该论文首次以美国核管理委员会反应堆操作员执照考试的全部历年试卷为基准，严格按80%及格线评估多模态语言模型，并系统比较了微调和检索策略对安全关键领域能力的影响。 |
| [^175] | [Does generative AI supersede supervised XMLC? A Benchmark Study on Automated Subject Indexing with German Scientific Literature](https://arxiv.org/abs/2607.14882) | 本研究通过对比监督式XMLC方法与基于LLM的生成式方法在德国科学文献自动主题索引任务上的表现，探讨了生成式AI能否取代传统监督式方法，并指出长尾词汇建议是共同挑战。 |
| [^176] | [Lesioned Multimodal Language Models Reproduce Aphasic Picture-Naming Patterns](https://arxiv.org/abs/2607.11621) | 该研究首次证明，通过对通用多模态语言模型（LLaVA 1.6）进行特定损伤扰动，能够以临床可比的比例再现失语症患者图片命名中的多种错误类型，为模拟神经语言障碍提供了新途径。 |
| [^177] | [The First ChineseBabyLM Challenge: training data-efficient and cognitively plausible language models for Chinese](https://arxiv.org/abs/2607.10745) | 本文介绍了首届中文BabyLM挑战赛，其核心贡献是设立了一个在有限数据下训练中文语言模型的基准，并发现引入拼音预测辅助目标的DeBERTa-v2架构在多项评估中表现最佳。 |
| [^178] | [Efficient Safety Alignment of Language Models via Latent Personality Traits](https://arxiv.org/abs/2607.07918) | 本文提出潜在人格对齐（LPA），通过仅用66条人格陈述进行对抗训练，在不接触有害数据且不损失性能的情况下，实现接近零的越狱攻击成功率，显著提升语言模型安全对齐效率。 |
| [^179] | [PolyWorkBench: Benchmarking LLM Agents for Cross-Lingual Long-Horizon Workflows](https://arxiv.org/abs/2607.06008) | 本文提出了PolyWorkBench基准，用于评估LLM代理在跨语言长时程工作流中的表现，并通过结构化评分标准来衡量其多语言整合与工具使用能力。 |
| [^180] | [On the Role of Directionality in Structural Generalization](https://arxiv.org/abs/2607.02307) | 本文通过使用CCG方向类型替代AM代数，显著提升了结构泛化性能，尤其在方向性相关类别上表现突出，并与更强的编码器互补。 |
| [^181] | [$x$-Prediction Flow: Efficient Continuous Decoding for Masked Diffusion Language Models](https://arxiv.org/abs/2606.29066) | 本文提出了一种基于$x$-预测的连续解码框架，通过将掩码预测转化为嵌入空间的连续流，并采用置信度驱动的异步更新，实现了MDLMs的高效且可修订的文本生成。 |
| [^182] | [DanceOPD: On-Policy Generative Field Distillation](https://arxiv.org/abs/2606.27377) | DanceOPD提出了一种在线策略生成场蒸馏框架，通过将不同图像生成能力（文生图、局部编辑、全局编辑）建模为共享空间中的速度场，并利用学生自身状态进行查询和训练，有效解决了多种能力之间的冲突与组合问题。 |
| [^183] | [Honeyquest for LLMs: Rethinking Cyber Deception for AI Attackers](https://arxiv.org/abs/2606.21037) | 本文通过大规模评估21个大语言模型，发现AI攻击者比人类更易落入网络欺骗陷阱，提出了一种新的自动化评估框架，并指出大语言模型构成独特的攻击者类别。 |
| [^184] | [LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents](https://arxiv.org/abs/2606.06087) | LatentSkill通过预训练超网络将文本技能转化为LoRA适配器，将技能知识从上下文空间迁移到权重空间，显著减少令牌开销并提升任务性能，同时保持技能的模块化组合能力。 |
| [^185] | [The Granularity Gap: A Multi-Dimensional Cross-Generational Audit of Sycophancy in Gemini Models](https://arxiv.org/abs/2606.05183) | 该论文揭示了安全评估中“通过/失败”二元判定与谄媚行为连续评分之间存在不可弥合的“粒度差距”，表明现有评估方法无法充分捕捉模型取悦用户的细微行为。 |
| [^186] | [MCBench: A Multicontext Safety Assessment Benchmark for Omni Large Language Models](https://arxiv.org/abs/2606.05177) | 该论文提出了MCBench基准，揭示了全模态大语言模型在安全评估中缺乏有效的跨模态推理能力，尤其在处理细微风险时表现不佳。 |
| [^187] | [SocialCoach: Personalized Social Skill Learning with Agentic Tutoring and Practice](https://arxiv.org/abs/2606.04155) | SocialCoach是一个基于大语言模型的代理辅导系统，通过构建理论到实践的语料库和轨迹级优化调度，实现个性化社交技能学习，解决了专家辅导稀缺的难题。 |
| [^188] | [Language Models Compare Quantities Using Number-specific and Unit-specific Heuristics](https://arxiv.org/abs/2606.03982) | 语言模型比较带单位数量时，并非统一换算，而是依赖数字和单位的启发式线索，导致边界附近系统性错误。 |
| [^189] | [Do Value Vectors in Deep Layers Need Context from the Residual Stream?](https://arxiv.org/abs/2606.02780) | 本文发现深层网络中的无上下文值向量能显著提升模型性能，并可稀疏存储以提高效率。 |
| [^190] | [Mitigating Bias in Locally Constrained Decoding via Tractable Proposals](https://arxiv.org/abs/2606.01926) | 本文提出了一种通过张量化有限自动机构建全局约束解码提案的通用方法，用于序列蒙特卡洛采样，从而有效缓解局部约束解码中的采样偏差。 |
| [^191] | [ChartFI: Benchmarking Faithfulness and Insightfulness of Chart Descriptions from Multimodal Large Language Models](https://arxiv.org/abs/2605.23694) | 本文提出了ChartFI-Bench，一个针对多模态大语言模型图表描述的多维度基准测试，首次系统性地评估了忠实性和洞察力，并定义了四个质量维度以弥补现有基准的不足。 |
| [^192] | [Hypergraph as Language](https://arxiv.org/abs/2605.21858) | 本文提出“超图即语言”视角和Hyper-Align框架，通过将超图结构直接编译为LLM可用的超图标记，保留高阶关联的原生语义，从而克服现有方法在图中心化处理中的局限。 |
| [^193] | [SymbolicLight V1: Spike-Gated Dual-Path Language Modeling at High Activation Sparsity](https://arxiv.org/abs/2605.21333) | SymbolicLight V1通过尖峰门控双路径设计，结合LIF动态与连续残差流，在超过89%的激活稀疏性下实现了与密集Transformer可比的语言建模性能。 |
| [^194] | [Single-Round Vector RAG vs an LLM-Compiled Wiki: A Preregistered Comparison on a Small Multi-Domain Research Corpus](https://arxiv.org/abs/2605.18490) | 本文预注册比较了单轮向量RAG和LLM编译维基在小型研究语料库上的问答性能，发现维基在跨论文综合上更优，但RAG在单事实查找上符合预期，而维基构建成本远高于查询成本。 |
| [^195] | [BiAxisBias: Evaluating LLM Bias Beyond a Single Prompt and a Single Explanation](https://arxiv.org/abs/2605.09041) | 本文提出BiAxisBias，一种多维度审计框架，通过变化任务、角色、视角、情感和措辞揭示LLM偏见分数对审计设计的敏感性，并量化了不同设计选择对偏见评估结果的影响。 |
| [^196] | [jina-embeddings-v5-omni: Geometry-preserving Embeddings via Locked Aligned Towers](https://arxiv.org/abs/2605.08384) | 本文提出GELATO方法，通过冻结骨干模型并仅训练连接组件（占总权重0.35%），构建了jina-embeddings-v5-omni多模态嵌入套件，能将文本、图像、音频和视频编码到同一语义空间，且保持几何特性。 |
| [^197] | [Leakage-Audited Benchmarking Reveals Limited Evidence for Cross-Subject Auditory-Evoked EEG Vowel Perception Decoding](https://arxiv.org/abs/2605.00865) | 该研究通过严格的泄漏审计基准，发现跨受试者听觉脑电元音解码的证据非常有限，即使最佳模型也仅略高于随机水平且不显著。 |
| [^198] | [MoRFI: Monotonic Sparse Autoencoder Feature Identification](https://arxiv.org/abs/2604.26866) | 本文通过受控微调实验发现，新知识的引入会加剧大模型幻觉，并识别出与之因果相关的潜在方向，揭示了SFT导致性能退化的机制。 |
| [^199] | [Structural Generalization on SLOG without Hand-Written Rules](https://arxiv.org/abs/2604.26157) | 本文提出一种无需手写规则的神经细胞自动机方法，在SLOG基准上实现接近AM-Parser的准确率，并揭示所有失败仅源于两种特定结构机制。 |
| [^200] | [Subliminal Steering: Stronger Encoding of Hidden Signals](https://arxiv.org/abs/2604.25783) | 本文提出潜意识引导方法，通过训练引导向量实现更复杂多词偏差的可靠传递，扩展了潜意识学习的信号范围。 |
| [^201] | [Enhancing Science Classroom Discourse Analysis through Joint Multi-Task Learning for Reasoning-Component Classification](https://arxiv.org/abs/2604.21137) | 本文提出了一种结合分层重划分、LLM合成数据增强和双探针头RoBERTa分类器的自动化课堂话语分析系统，有效解决了标签不平衡问题，并显著提升了科学课堂推理组件的分类性能。 |
| [^202] | [Lost in Adaptation: Layer-Selective Recovery of Temporal Reasoning in Video-Language Models](https://arxiv.org/abs/2604.11399) | 本文提出MERIT框架，通过层选择性模型合并和CMA-ES优化，在不牺牲时间感知的情况下显著恢复视频-语言模型的时间推理能力，相对增益最高达27.8%。 |
| [^203] | [Shorter, but Still Trustworthy? An Empirical Study of Chain-of-Thought Compression](https://arxiv.org/abs/2604.04120) | 本研究首次系统实证发现，链式思维压缩虽节省推理成本，但常损害模型的安全性、抗幻觉和多语言鲁棒性，且不同压缩方法退化特征各异。 |
| [^204] | [I-CALM: Incentivizing Confidence-Aware Abstention for LLM Selective Answering](https://arxiv.org/abs/2604.03904) | I-CALM通过结合言语置信度、收益激励和规范性指导，在提示级别上促使黑盒大语言模型在可能出错时选择性弃权，从而提升事实性问答的准确性和可靠性。 |
| [^205] | [Train Yourself as an LLM: Exploring Effects of AI Literacy on Persuasion via Role-playing LLM Training](https://arxiv.org/abs/2604.02637) | 本文提出了一种基于角色扮演的游戏化AI素养教程LLMimic，通过让用户模拟LLM训练过程，增强其对AI说服的抵抗力，并验证了其在不同说服场景中的有效性。 |
| [^206] | [ContextClaim: A Context-Driven Paradigm for Verifiable Claim Detection](https://arxiv.org/abs/2603.30025) | 本文提出ContextClaim，通过将证据检索引入声明检测阶段，利用上下文信息（如实体和事件的可获取性）来提升可验证声明检测的准确性，从而弥补传统方法仅依赖声明句子的局限性。 |
| [^207] | [MechMath: Sorrifier-Driven Formal Decomposition Workflow for Automated Theorem Proving](https://arxiv.org/abs/2603.24465) | MechMath提出了一种基于Sorrifier驱动的形式化分解工作流，利用Lean中的sorry占位符精确隔离未解决的子目标，避免了上下文过长和低效重生成的问题，从而提升了复杂数学定理证明的效率和成功率。 |
| [^208] | [Faster, Cheaper, More Accurate: Specialised Knowledge Tracing Models Outperform LLMs](https://arxiv.org/abs/2603.02830) | 本文证明，专业化的知识追踪模型在预测学生未来反应方面，在准确性、部署成本和推理速度上全面优于大型语言模型，强调了领域专用模型在教育场景中的优势。 |
| [^209] | [Reasoning-Based Personalized Generation for Users with Sparse Data](https://arxiv.org/abs/2602.21219) | 本文提出GraSPer框架，通过预测用户未来交互并生成合成文本来扩充稀疏上下文，从而提升LLM在冷启动等稀疏数据场景下的个性化生成能力。 |
| [^210] | [HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam](https://arxiv.org/abs/2602.13964) | 本文提出HLE-Verified，通过两阶段验证-修复流程和细粒度错误分类，系统性地验证并结构化修订了“人类最后考试”基准，确保评估结果的可靠性和跨模型比较的公平性。 |
| [^211] | [Agentic Test-Time Scaling for WebAgents](https://arxiv.org/abs/2602.12276) | 本文提出CATTS技术，通过动态分配计算资源和利用代理投票分布的不确定性统计，解决Web代理在多步任务中测试时扩展的收益递减问题。 |
| [^212] | [Misconception Diagnosis From Student-Tutor Dialogue: Generate, Retrieve, Rerank](https://arxiv.org/abs/2602.02414) | 本文提出了一种结合生成、检索和重排序策略的LLM方法，用于从学生-导师对话中自动识别误解，显著提升了预测性能。 |
| [^213] | [Sequential LLM Release Facilitates Manipulation in Regulated Markets](https://arxiv.org/abs/2601.11496) | 本文通过GLEE基准数据发现，顺序发布大型语言模型在市场中可能产生“毒苹果”效应，即发布的模型虽未被采用，却会改变均衡结果，导致一方收益增加而另一方受损，从而助长市场操纵。 |
| [^214] | [AWED-PIPER: Agents, Web Applications & Expert Detectors for Personally Identifiable Information Protection & Fine-grained Named Entity Recognition across 36 languages for 6.6 Billion Speakers](https://arxiv.org/abs/2601.10161) | 该论文提出了AWED-PIPER框架，通过54个专家模型和代理工具，在36种语言中实现细粒度命名实体识别和可逆PII匿名化，兼顾信息提取与隐私保护。 |
| [^215] | [QA-Merging: Query-Adaptive Reasoning via Layer Selective Model Merging](https://arxiv.org/abs/2601.03506) | 提出一种基于激活的查询自适应层选择性合并框架，无需重新训练即可动态选择层，实现高效的自适应推理，兼顾长思维链与短思维链的优势。 |
| [^216] | [jina-vlm: Small Multilingual Vision Language Model](https://arxiv.org/abs/2512.04032) | 本文提出jina-vlm，一个24亿参数的多语言视觉语言模型，通过图像分块和注意力池化实现令牌高效处理，并在2B规模模型中达到最先进的多语言VQA性能，同时通过消融研究揭示了训练数据类别的影响。 |
| [^217] | [A Large-Scale Chinese Knowledge Graph-Text Alignment Dataset for Benchmarking Knowledge-Grounded LLMs](https://arxiv.org/abs/2510.06039) | 本文提出了CDTP，一个包含700万实例和1500万三元组的大规模中文知识图谱-文本对齐数据集，通过多阶段构建流程确保语义和事实准确性，用于基准测试知识增强的大语言模型在结构化推理中的表现。 |
| [^218] | [SimulRAG: Simulator-based RAG for Grounding LLMs in Long-form Scientific QA](https://arxiv.org/abs/2509.25459) | 本文提出了SimulRAG，一种基于科学模拟器的RAG框架，通过通用检索接口和声明级生成与不确定性估计，有效解决了长篇科学问答中的幻觉问题。 |
| [^219] | [Vision Language Models Cannot Plan, but Can They Formalize?](https://arxiv.org/abs/2509.21576) | 本文提出了五种VLM作为形式化器的流水线，用于一次性、开放词汇和多模态PDDL规划形式化，以解决VLM在长期规划中的不足。 |
| [^220] | [Knowing When to Defer: Selective Prediction for Responsible Knowledge Tracing](https://arxiv.org/abs/2509.21514) | 本文提出了一种基于MC-Dropout的内置选择性预测层，使现有知识追踪模型能够智能暂缓不确定预测，在无需重训练的情况下显著提升准确性和公平性。 |
| [^221] | [Language Models that Think, Chat Better](https://arxiv.org/abs/2509.20357) | 本文提出RLMT方法，通过引入模型奖励思考的强化学习，将长链推理扩展到开放式任务，显著提升语言模型的通用对话能力。 |
| [^222] | [Efficient Code Embeddings from Code Generation Models](https://arxiv.org/abs/2508.21290) | 本文提出了一种基于自回归模型和最后令牌池化的代码嵌入方法，在小型模型上实现了跨语言代码检索和问答的最先进性能。 |
| [^223] | [PEER: Unified Process-Outcome Reinforcement Learning for Structured Empathetic Reasoning](https://arxiv.org/abs/2508.09521) | 本文提出了PEER框架，通过结构化共情推理和统一的过程-结果奖励机制，解决了情绪支持对话中缺乏心理学推理和强化学习奖励信号不可靠的问题。 |
| [^224] | [CulTrace: Tracing Internal Cultural Reasoning in Large Language Models](https://arxiv.org/abs/2508.08879) | 本文提出CulTrace方法，通过机械可解释性揭示大型语言模型在文化问答中内部文化推理的分阶段轨迹，并发现其推理存在不平衡性。 |
| [^225] | [Adapting LLMs to Time Series Forecasting via Temporal Heterogeneity Modeling and Representation Alignment](https://arxiv.org/abs/2508.07195) | 本文提出了TALON框架，通过异质时间编码器和表示对齐模块，分别解决时间模式异质性和模态差距问题，从而提升大语言模型在时间序列预测中的性能。 |
| [^226] | [FollowUpBot: An LLM-Based Conversational Robot for Automatic Postoperative Follow-up](https://arxiv.org/abs/2507.15502) | 本文提出FollowUpBot，一种边缘部署的大语言模型驱动的术后随访机器人，通过动态路径规划和多模态对话实现自适应、隐私保护的自动随访，并自动生成结构化报告。 |
| [^227] | [A validity-guided workflow for robust large language model research in psychology](https://arxiv.org/abs/2507.04491) | 本文提出一个六阶段效度引导工作流程，通过将效度要求与研究雄心相匹配，解决大语言模型在心理学研究中的测量不可靠性问题，以防止“测量幻影”威胁研究效度。 |
| [^228] | [From Prompts to Constructs: A Dual-Validity Framework for Large Language Model Research in Psychology](https://arxiv.org/abs/2506.16697) | 本文提出了一个双重效度框架，强调在将大语言模型用于心理学研究时，需结合心理测量学验证和因果推断标准，以避免“测量幻象”并确保研究结论的科学有效性。 |
| [^229] | [LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking](https://arxiv.org/abs/2506.07449) | 本论文提出了一种单遍、端到端可训练的知识图谱增强RAG框架，通过轻量级用户偏好模块提取个性化关系路径并整合到LLM提示中，从而提升推荐排序的准确性和可解释性。 |
| [^230] | [DiagnosisArena: Benchmarking Diagnostic Reasoning for Large Language Models](https://arxiv.org/abs/2505.14107) | 本文提出了诊断竞技场（DiagnosisArena），一个基于1,113对临床病例、覆盖28个专科的全面基准，用于系统评估大型语言模型的专业级诊断推理能力。 |
| [^231] | [The Political Ideology of Large Language Models: Measurement, Inconsistency, and Persuasive Influence](https://arxiv.org/abs/2505.04171) | 本文通过比较43个LLMs与政治人物和选民，发现其表面温和的党派立场实为特定议题上强烈立场的抵消结果，并通过实验证明LLMs能显著说服影响选民的意识形态。 |
| [^232] | [Bye-bye, Bluebook? Automating Legal Drudgery With AI-Augmented Rule Following](https://arxiv.org/abs/2505.02763) | 本文首次实证评估了AI在蓝皮书引注格式这一法律繁琐工作上的表现，发现前沿模型零样本合规率仅42.6%，远低于人类水平，并提出了新基准和多项改进建议。 |
| [^233] | [Leveraging Machine Unlearning for Cost-Efficient Preference Alignment](https://arxiv.org/abs/2504.06659) | 本文提出了一个结合机器遗忘与偏好对齐的框架，通过定量分析负面示例的影响差异，为成本高效地选择与加权负面示例提供了新方法。 |
| [^234] | [Evidence of conceptual mastery in the application of rules by Large Language Models](https://arxiv.org/abs/2503.00992) | 本文通过心理学方法证明大型语言模型在规则应用上能复制人类的行为模式，包括意外差异和时间延迟效应，显示出概念掌握的证据。 |
| [^235] | [Thinking Outside the (Gray) Box: A Context-Based Score for Assessing Value and Originality in Neural Text Generation](https://arxiv.org/abs/2502.13207) | 提出一种基于上下文的评分方法，结合信息论，用于评估神经文本生成的价值与原创性，并作为强化学习奖励微调大型语言模型，提升创造性任务的表现。 |
| [^236] | [DR.GAP: Mitigating Bias in Large Language Models using Gender-Aware Prompting with Decoupled Reasoning](https://arxiv.org/abs/2502.11603) | DR.GAP通过生成性别中立的推理轨迹并将其作为上下文示例，实现了性别信息与任务语义的解耦，从而在保持模型性能的同时有效缓解了大型语言模型中的性别偏见。 |
| [^237] | [Improving Influence-based Instruction Tuning Data Selection for Balanced Learning of Diverse Capabilities](https://arxiv.org/abs/2501.12147) | 本文提出BIDS算法，通过归一化影响力分数来消除任务间固有偏见，实现指令微调数据选择的均衡性，从而提升模型多样能力的综合表现。 |
| [^238] | [Bactrainus: Optimizing Large Language Models for Multi-hop Complex Question Answering Tasks](https://arxiv.org/abs/2501.06286) | 本文提出Bactrainus框架，通过分离段落选择、支持句识别和答案生成，并引入问题分解与推理监督，有效缓解了大型语言模型在多跳问答中因无关上下文导致的性能下降问题。 |
| [^239] | [DYNASHIELD: A Black-Box Moving Target Defense for LLMs via Dynamic Decoding Customization](https://arxiv.org/abs/2412.07672) | DYNASHIELD通过动态定制解码参数和系统提示，在无需访问模型内部或额外训练的情况下，有效降低了黑盒LLM面对越狱攻击的成功率。 |
| [^240] | [Multi-Bin Batching for Increasing LLM Inference Throughput](https://arxiv.org/abs/2412.04504) | 本文提出多箱批处理方法，通过将相似执行时间的请求分组到预定时间箱，从排队论角度证明了能显著提升LLM推理吞吐量。 |
| [^241] | [mR$^2$AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA](https://arxiv.org/abs/2411.15041) | 本文提出了一种新的多模态检索-反思-增强生成框架（mR$^2$AG），通过自适应检索和证据识别机制，解决了现有方法在知识型VQA中过度检索、缺乏证据支持及模型复杂度过高的问题。 |
| [^242] | [Macroeconomic Forecasting with Large Language Models](https://arxiv.org/abs/2407.00890) | 本文通过FRED-MD数据库对比评估了大型语言模型与传统方法在宏观经济预测中的表现，揭示了LLMs的优缺点及实际应用潜力。 |

# 详细

[^1]: 迈向计算来源：在生成文本中携带因果状态证据

    Towards Computational Provenance: Carrying Causal-State Evidence in Generated Text

    [https://arxiv.org/abs/2608.16868](https://arxiv.org/abs/2608.16868)

    本文提出“计算来源”概念，证明在受控架构中生成文本可携带可检测的因果内部状态证据，并通过128个匹配对验证了可行性。

    

    arXiv:2608.16868v1 公告类型：交叉 摘要：语言模型的输出本身并不能提供关于其内部计算的可验证证据。我们研究了计算来源问题：生成文本是否能携带关于实际发生的因果相关内部状态的可检测证据。我们在两种受控架构中测试了这一思想的有限形式：模块化前馈神经网络和基于Transformer的模型。两种架构均在相同的算术任务上训练，并通过两个离散中间状态的强制路径，使得不同的内部路径能够产生相同的答案。我们有意在这些路径之间切换，验证实际使用的状态，并让该验证状态决定生成文本中的一种微妙统计模式，该模式可在后续被检测到。前馈和Transformer系统在各自的公开和单独密封的保护性端到端评估中均通过了全部128个匹配对，其中d值...

    arXiv:2608.16868v1 Announce Type: cross  Abstract: A language model's output does not by itself provide verifiable evidence about the internal computation that produced it. We study computational provenance: whether generated text can carry detectable evidence of which causally relevant internal state occurred. We test a bounded form of this idea in two controlled architectures: a modular feed-forward neural network and a transformer-based model. Both architectures are trained on the same arithmetic task with a mandatory pathway through two discrete intermediate states, allowing different internal paths to produce the same answer. We deliberately switch between these paths, authenticate the state actually used, and let that verified state determine a subtle statistical pattern in the generated text that can later be detected. The feed-forward and transformer systems each passed all 128 matched pairs in both their public and separately sealed protected end-to-end evaluations, with the d
    
[^2]: Proteus：用于长上下文序列建模的增量式记忆激活

    Proteus: Incremental Memory Activation for Long-Context Sequence Modeling

    [https://arxiv.org/abs/2608.16844](https://arxiv.org/abs/2608.16844)

    本文提出Proteus，一种通过增量式扩展记忆容量来解决长上下文序列建模中静态记忆污染和干扰问题的新范式。

    

    基于注意力的序列模型在处理长上下文时面临二次方计算成本，这促使了越来越多关于基于记忆的模型的研究，这些模型可以将上下文压缩为紧凑的状态。然而，大多数现有的记忆模型在整个序列中暴露静态记忆。由于早期标记没有压缩压力，它们占用了过多的自由度并“污染”了记忆状态，导致为后续上下文留下的容量有限，并增加了存储内容与后续输入之间的干扰。我们研究了一种新的增量式记忆激活范式，其中记忆的有效容量随上下文增长而逐步扩展。施加早期瓶颈迫使模型更有效地压缩历史信息，而随时间解锁新容量则减少了干扰并改善了对后续上下文的保留。我们将这一范式实例化为Proteus，这是一种可以直接融入b（原文截断）的简单机制。

    arXiv:2608.16844v1 Announce Type: cross  Abstract: The quadratic cost of attention-based sequence models for long contexts has motivated a growing line of research on memory-based models that can compress context into a compact state. However, most existing memory models expose a static memory throughout the entire sequence. Because early tokens face no compression pressure, they occupy too many degrees of freedom and "pollute" the memory state, leaving little capacity for later context and increasing interference between what is stored and what arrives next. We study a new paradigm of incremental memory activation, where the effective capacity of memory is progressively expanded as the context grows. Imposing an early bottleneck forces the model to compress history more effectively, while unlocking fresh capacity over time reduces interference and improves retention of later context. We instantiate this paradigm in Proteus, a straightforward mechanism that can be incorporated into a b
    
[^3]: 模型催眠：通过附加的阈下效应实现对AI的强力控制

    Model Hypnosis: Strong control of AI via additive subliminal effects

    [https://arxiv.org/abs/2608.16834](https://arxiv.org/abs/2608.16834)

    模型催眠通过组合微弱提示线索强力控制AI行为，跨模型通用且可转移，对AI安全和可解释性构成重大挑战。

    

    arXiv:2608.16834v1 公告类型：交叉 摘要：我们证明AI模型普遍易受一种我们称之为“模型催眠”的现象影响，在这种现象中，提示中单独微弱且看似无关的线索可以被系统性地组合起来，从而强力控制模型行为。模型催眠跨越不同模型家族和规模，包括前沿推理模型，且催眠提示可以在模型之间转移。由于模型被不显眼的文本选择（如改写和拼写错误）所控制，模型催眠为AI安全带来了新的挑战和途径，并且是AI可解释性的一个重大障碍。

    arXiv:2608.16834v1 Announce Type: cross  Abstract: We demonstrate that AI models are broadly susceptible to a phenomenon we call model hypnosis, in which individually weak and seemingly irrelevant cues in the prompt can be systematically combined to strongly control model behavior. Model hypnosis occurs across model families and scales, including in frontier reasoning models, and hypnotic prompts can transfer between models. Because the model is controlled by inconspicuous textual choices, such as paraphrases and typos, model hypnosis presents new challenges and avenues for AI safety, and is a major hurdle for AI interpretability.
    
[^4]: 基于人类反馈的策略迭代：将训练后强化学习引入上下文学习

    Policy Iteration with Human Feedback: Bringing Post-Training RL to In-context Learning

    [https://arxiv.org/abs/2608.16831](https://arxiv.org/abs/2608.16831)

    本文提出PIHF方法，利用预训练语言模型作为执行基础，通过语言模型批评者和临床专家的循环评估与修订，将强化学习思想引入上下文学习，从而改进策略性能。

    

    生成式预训练建立了可复用的任务表征；后续关于基于语言的任务条件化和上下文学习的研究表明，固定模型能够根据指令和演示调整其行为。基于人类反馈的策略迭代（PIHF）在此基础上发展，并融合了广义策略迭代中循环评估与改进的结构。PIHF使用预训练语言模型作为执行基础，并将持续修订迁移至版本化的自然语言策略和工具集。一个语言模型批评者和临床专家审查完整面板推理和工具使用轨迹，以定位反复出现的失败并形成候选修订；专家可重新解释证据并保留准入和回滚的权威，而Recall@1和Recall@5在候选执行后验证结果。在累积消融和超罕见疾病基准测试中，基于PIHF的策略将R值提升了...

    arXiv:2608.16831v1 Announce Type: new  Abstract: Generative pretraining established reusable task representations; later work on language-based task conditioning and in-context learning showed that a fixed model could adapt its behavior from instructions and demonstrations. Policy Iteration with Human Feedback (PIHF) builds on this development and the recurrent evaluate-and-improve structure of generalized policy iteration. PIHF uses a pretrained language model as its execution substrate and moves persistent revision to a versioned natural-language policy and tool set. A language-model critic and clinical expert review complete-panel reasoning and tool-use trajectories to localize recurrent failures and form candidate revisions; the expert may reinterpret the evidence and retains authority over admission and rollback, while Recall@1 and Recall@5 validate outcomes after candidate execution.   Across cumulative ablations and ultra-rare-disease benchmarks, a PIHF-derived policy improved R
    
[^5]: ClawGym II：探索智能体装备上的黑盒强化学习

    ClawGym II: Exploring Black-Box RL on Agent Harness

    [https://arxiv.org/abs/2608.16798](https://arxiv.org/abs/2608.16798)

    本论文提出了一种统一的黑盒强化学习框架，通过沙箱隔离和前缀树组织模型调用来实现复杂智能体装备上的稳定可扩展训练，并适配了PPO和GRPO算法。

    

    摘要：arXiv:2608.16798v1 公告类型：交叉 摘要：智能体装备通过协调智能体与环境的交互，在长时程任务上显著提升了性能。然而，通过复杂装备进行强化学习仍 largely 未被探索，因为将这种训练扩展到长时程智能体任务会引入根本性挑战。在本工作中，我们提出了一个统一的黑盒强化学习框架，用于通过复杂装备对通用智能体进行稳定且可扩展的优化。具体而言，我们首先构建了一个基于沙箱的执行基础设施，将任务环境和装备隔离在临时沙箱中，以支持大规模并发回滚。然后，我们将策略优化与不透明的装备执行解耦，并在模型边界放置一个服务代理以捕获模型调用。为了重建多轮轨迹并提高训练效率，我们将捕获的调用组织成前缀树，并进一步适配基于评论家的PPO和无评论家的GRPO算法。

    arXiv:2608.16798v1 Announce Type: cross  Abstract: Agent harnesses have substantially improved performance on long-horizon tasks by coordinating agent interactions with the environment. However, reinforcement learning through complex harnesses remains largely unexplored, as scaling such training to long-horizon agent tasks introduces fundamental challenges. In this work, we present a unified black-box RL framework for stable and scalable optimization of general agents through complex harnesses. Concretely, we first build a sandbox-based execution infrastructure that isolates task environments and harnesses within temporary sandboxes for large-scale concurrent rollouts. We then decouple policy optimization from opaque harness execution and place a serving proxy at the model boundary to capture model calls. To reconstruct multi-turn trajectories and improve training efficiency, we organize the captured calls into prefix trees and further adapt both critic-based PPO and critic-free GRPO t
    
[^6]: 神经符号具身代理

    Neurosymbolic Embodied Agents

    [https://arxiv.org/abs/2608.16794](https://arxiv.org/abs/2608.16794)

    该论文提出一种神经符号代理，通过视觉探索生成符号状态，并结合PDDL约束和蒙特卡洛树搜索，确保长时程家庭任务计划的可执行性。

    

    arXiv:2608.16794v1 公告类型：交叉 摘要：语言和视觉-语言模型能够生成看似合理的具身计划，但无法保证可执行性，因为其输出可能违反环境动态或作用于错误实体。我们提出了一种神经符号代理，将长期家庭任务分解为任务导向的视觉探索和约束符号规划。在第一阶段，视觉-语言模型和探索工具从自我中心观察和接地交互中获取目标相关谓词和实例绑定，生成符号初始状态。在第二阶段，PDDL转移模型将解码限制为扩展适用动作的标记。蒙特卡洛树搜索随后使用领域无关的规划启发式评估可执行的延续。由此产生的计划在转移模型下按构造可执行，并在正确视觉接地条件下转移到环境。在VirtualHome和ALFWor上进行了测试。

    arXiv:2608.16794v1 Announce Type: cross  Abstract: Language and vision-language models generate plausible embodied plans but do not guarantee executability, as their outputs can violate environment dynamics or act on incorrectly grounded entities. We present a neurosymbolic agent that factors long-horizon household tasks into task-directed visual exploration and constrained symbolic planning. In the first phase, a vision-language model and exploration harness acquire goal-relevant predicates and instance bindings from egocentric observations and grounded interactions, producing a symbolic initial state. In the second, a PDDL transition model restricts decoding to tokens that extend applicable actions. Monte Carlo tree search then evaluates executable continuations using a domain-independent planning heuristic. The resulting plans are executable by construction under the transition model, with transfer to the environment conditioned on correct visual grounding. On VirtualHome and ALFWor
    
[^7]: 语义赌博机：上下文中的探索-利用受到语义先验的偏差影响

    Semantic Bandits: In-Context Exploration-Exploitation is Biased by Semantic Priors

    [https://arxiv.org/abs/2608.16707](https://arxiv.org/abs/2608.16707)

    本文提出语义赌博机框架，揭示LLM在决策中因语义先验而产生探索偏差，标签与奖励对齐时提升性能，错位时则严重损害表现。

    

    arXiv:2608.16707v1 公告类型：交叉 摘要：大型语言模型（LLMs）越来越多地被部署为需要复杂环境探索的决策代理。然而，现有研究对LLMs如何实际平衡探索与利用提出了疑问。与经典代理不同，LLM代理通过自然语言参与任务，使它们接触到任务结构中无形式对应物的语义信息。我们引入了语义赌博机，这是多臂赌博机设置的一个扩展，明确考虑了分配给动作的文本标签，并用它来研究语义先验——即预训练期间语言与预期奖励之间关联所产生的归纳偏差——如何塑造LLM的探索行为。我们发现，语义信息丰富的动作标签会减少探索而偏向利用，当这些标签与奖励结构一致时能提升性能，但在不一致时会严重降低性能。

    arXiv:2608.16707v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed as decision-making agents in settings that require sophisticated environmental exploration. However, existing work has raised questions about how LLMs actually balance exploration and exploitation. Unlike classical agents, LLM agents engage with tasks through natural language, exposing them to semantic information with no formal counterpart in the task structure. We introduce the semantic bandit, an extension of the multi-armed bandit setting that explicitly considers the textual labels assigned to actions, and use it to study how semantic priors --- inductive biases arising from associations between language and expected reward learned during pre-training, shape LLM exploration behaviour. We find that semantically informative action labels reduce exploration in favour of exploitation, improving performance when aligned with the reward structure and severely degrading it when misal
    
[^8]: 闭合情感回路：基于多模态说话者-听者情感动态感知的共情社交机器人

    Closing the Affective Loop: Multimodal Speaker-Listener Emotion-Dynamics-Aware Empathetic Social Robots

    [https://arxiv.org/abs/2608.16686](https://arxiv.org/abs/2608.16686)

    该论文提出AffectLoop，一种多模态口语对话系统，通过双向追踪说话者和听者的情感动态来闭合共情回路，在机器人上实现更自然的具身情感交互。

    

    共情社交机器人不仅应响应用户所说的话，还应响应其情感在交互过程中如何动态演变。然而，现有的共情对话系统通常以文本为中心，主要将共情建模为用户情感到系统响应的单向映射，这限制了其捕捉具身说话者-听者情感交流的能力。我们提出了AffectLoop，一个在Misty II机器人上实现的多模态说话者-听者情感动态感知的口语对话系统。该系统跟踪说话者的言语和面部情感动态，估计机器人听者自身的言语和行为情感状态，并将基于LLM的响应生成条件化于这两条情感流。随后，机器人生成简短的口语共情响应，并伴随情感一致的具身行为，形成闭合的说话者-听者情感回路。我们在一个受试者内试点研究中评估了该系统。

    arXiv:2608.16686v1 Announce Type: cross  Abstract: Empathetic social robots should respond not only to what users say, but also to how their emotions dynamically evolve during interaction. However, existing empathetic dialogue systems are often text-centered and primarily model empathy as a one-way mapping from the user's emotion to the system response, limiting their ability to capture embodied speaker--listener affective exchange. We present AffectLoop, a multimodal speaker-listener emotion-dynamics-aware spoken dialogue system implemented on the Misty II robot. The system tracks the speaker's verbal and facial affective dynamics, estimates the robot listener's own verbal and behavioral affective state, and conditions LLM-based response generation on both affective streams. The robot then generates a short spoken empathetic response together with emotionally congruent embodied behavior, forming a closed speaker--listener affective loop. We evaluate the system in a pilot within-subjec
    
[^9]: 语言模型头是否造成了有害的梯度瓶颈？一项因果测试

    Does the LM Head Create a Harmful Gradient Bottleneck? A Causal Test

    [https://arxiv.org/abs/2608.16671](https://arxiv.org/abs/2608.16671)

    本文通过因果测试分离了LM头几何与优化影响，发现仅限制反向梯度秩对损失影响较小，而前向分解头影响更大，表明LM头并非主要优化瓶颈。

    

    语言模型头将宽度为D的隐藏状态映射到大小为V的词表，因此其转置最多只能向Transformer返回D个独立方向。Godey和Artzi认为这种严重投影是一个有害的优化瓶颈。我们将几何形状与因果主张分开。我们的仅反向干预保持普通logits和精确的LM头参数更新，同时仅减少发送到Transformer的梯度的秩。在字节级和BPE-8192 WikiText-2模型的五对种子中，减少反向秩会增加验证损失。然而，一个同等秩的分解前向头会显著增加更多损失。在较大模型的半秩情况下，仅反向的损失增加为0.0586（95%置信区间[0.0167, 0.1005]），而分解前向头将损失增加0.1795（[0.1547, 0.2042]）。词表空间残差也对普通LM头更新有贡献，去除该残差...

    arXiv:2608.16671v1 Announce Type: new  Abstract: The language-model head maps a hidden state of width D to a vocabulary of size V, so its transpose can return at most D independent directions to the Transformer. Godey and Artzi argue that this severe projection is a harmful optimization bottleneck. We separate the geometry from the causal claim. Our backward-only intervention keeps the ordinary logits and the exact LM-head parameter update while reducing only the rank of the gradient sent into the Transformer. Across five paired seeds on byte-level and BPE-8192 WikiText-2 models, reducing backward rank increases validation loss. An equally ranked factorized forward head, however, increases loss substantially more. At half rank in the larger model, the backward-only loss increase is 0.0586 (95% CI [0.0167, 0.1005]), while the factorized forward head increases loss by 0.1795 ([0.1547, 0.2042]). The vocabulary-space residual also contributes to the ordinary LM-head update, and removing th
    
[^10]: 基于PCA引导的激活缩放实现大语言模型谄媚行为的单调双向控制

    PCA-guided Activation Scaling for Monotonic Bidirectional Control over LLM Sycophancy

    [https://arxiv.org/abs/2608.16650](https://arxiv.org/abs/2608.16650)

    本文提出了一种新的激活引导方法PAS，通过PCA分解和缩放指数实现对大语言模型谄媚行为的单调双向控制，显著优于现有方法。

    

    大型语言模型（LLMs）表现出谄媚行为，即倾向于同意用户的信念而不顾事实准确性。这可能强化误解，但完全消除它则冒着对有效观点过度纠正的风险。因此，有效控制必须既能以可预测且渐进的效果减少和增加谄媚行为。然而，现有方法无法确保在跨模型和数据集时，控制强度与行为结果之间存在双向且单调的关系。我们引入了PCA引导的激活缩放（PAS），这是一种激活引导框架，它将残差流激活分解为PCA识别的谄媚-诚实子空间和正交残差，然后应用不同的缩放指数以实现单调、双向控制。在三个LLM和三个数据集上，PAS实现了强单调性（Spearman ρ = +0.92）和每个方向平均15.4%的偏移，而基线方法仅为8.7%。

    arXiv:2608.16650v1 Announce Type: new  Abstract: Large language models (LLMs) exhibit sycophancy, a tendency to agree with user beliefs regardless of factual accuracy. This can reinforce misconceptions, but eliminating it entirely risks over-correction against valid opinions. Effective control must therefore both reduce and increase sycophancy with predictable and gradual effect. Yet, existing methods fail to ensure a bidirectional and monotonic relationship between steering strength and behavioral outcome across models and datasets. We introduce PCA-guided Activation Scaling (PAS), an activation steering framework that decomposes residual stream activations into a PCA-identified sycophancy-honesty subspace and an orthogonal residual, then applies distinct scaling exponents to achieve monotonic, bidirectional control. Across three LLMs and three datasets, PAS achieves strong monotonicity (Spearman $\rho$ = +0.92) and an average shift of 15.4% per direction, compared with 8.7% for the b
    
[^11]: 每一枚硬币都有两面：关于大语言模型同策略蒸馏中泛化的双重性

    Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models

    [https://arxiv.org/abs/2608.16647](https://arxiv.org/abs/2608.16647)

    同策略蒸馏的泛化行为取决于教师和学生来源关系，同源对能跨域迁移推理能力，跨源对则受限于训练分布，这种双重性既是优势也是风险。

    

    同策略蒸馏（OPD）通过监督学生自身策略采样的轨迹来转移教师能力，但其泛化行为仍鲜为人知，因为大多数研究仅在单一领域和接近训练数据的基准上评估OPD。我们进行了一项受控研究，每次只改变一个泛化因素，从域内分布偏移到跨域迁移以及多教师设置。我们发现OPD转移的是教师的推理行为而非其对特定问题的答案：训练难度几乎无关紧要，甚至教师从未解决的问题也有用。迁移强烈依赖于教师和学生之间的来源关系：同源对使学生在语言、推理范围甚至其他领域中接近教师，而跨源对主要适应训练分布。这种广泛的影响是一把双刃剑。

    arXiv:2608.16647v1 Announce Type: new  Abstract: On-policy distillation (OPD) transfers teacher capabilities by supervising trajectories sampled from the student's own policy, yet its generalization behavior remains poorly understood, as most studies evaluate OPD on a single domain and on benchmarks close to the training data. We present a controlled study that varies one generalization factor at a time, from in-domain distribution shifts to cross-domain transfer and the multi-teacher setting. We find that OPD transfers a teacher's reasoning behavior rather than its answers to particular problems: training difficulty barely matters, and even problems the teacher never solves are useful. Transfer depends strongly on the origin relationship between teacher and student: same-origin pairs bring the student close to the teacher across languages, reasoning horizons, and even other domains, whereas cross-origin pairs mostly fit the trained distribution. This broad reach is a double-edged swor
    
[^12]: 重构：从预发表参考文献中恢复研究思路的盲测基准

    Reconstruction: A Blind Benchmark for Recovering Research Ideas from Pre-Publication Bibliographies

    [https://arxiv.org/abs/2608.16645](https://arxiv.org/abs/2608.16645)

    该论文提出一个名为“重构”的盲测基准，通过仅使用预发表参考文献来评估语言模型恢复研究思路的能力，并展示了一种多智能体流水线可显著提高匹配率。

    

    arXiv:2608.16645v1 公告类型：新  摘要：当仅给定一篇已发表论文的预发表参考文献时，语言模型能否恢复该论文的真实研究思路？我们引入了“重构”，一个盲测思路恢复基准，它隐藏种子论文及所有同时期或未来的文献，并要求模型提出假设，由独立的大型语言模型评判器将这些假设与隐藏的真实思路进行匹配。严格的防泄漏协议——包括时间引文截断、匿名参考ID和冻结的逐篇论文参考文献列表——可防止提示时泄漏种子思路。在六个科学领域和643篇评估论文中，七个前沿模型仅实现了适度的匹配率（约3-15%）。随后，我们评估了一个仅参考的多智能体（前四名）流水线，该流水线结合了跨模型评审和对齐假设槽的瑞士制锦标赛，无需外部网络搜索。跨模型评审加锦标赛选择将匹配率提升至约...

    arXiv:2608.16645v1 Announce Type: new  Abstract: Can a language model recover the true research idea of a published paper when given only that paper's pre-publication bibliography? We introduce Reconstruction, a blind idea-recovery benchmark that withholds the seed paper and all contemporaneous or future literature, and asks models to propose hypotheses that an independent large language model judge matches against the held-out ground-truth idea. A strict anti-leakage protocol-temporal citation cutoff, anonymous reference IDs, and frozen per-paper bibliographies, which prevents prompt-time leakage of the seed idea. Across six scientific domains and 643 evaluated papers, seven frontier models achieve only modest Match rates (approx. 3-15%). We then evaluate a reference-only multi-agent (top 4) pipeline that combines cross-model review with a Swiss tournament over aligned hypothesis slots, without external web search. Cross-model review plus tournament selection raises Match rates to app
    
[^13]: 迈向更好评估LLMs在临床错误检测中的表现

    Toward Better Assessment of LLMs' Performance in Clinical Error Detection

    [https://arxiv.org/abs/2608.16643](https://arxiv.org/abs/2608.16643)

    本研究揭示，在临床错误检测中，仅依赖传统F1分数等指标会高估LLM性能，因为15个模型中有13个在成对判别任务上低于随机水平，且偏差模式随语言变化，强调了评估方法需利用错误-正确配对结构。

    

    arXiv:2608.16643v1 公告类型：交叉 摘要：自动检测临床文档中的错误是大语言模型（LLMs）的一个有前景的应用，然而部署此类模型的决定依赖于将每份临床笔记单独评估的基准测试。错误检测基准通常通过向笔记中注入错误来构建，使得每个错误笔记都有一个自然对应的正确版本。聚合判别指标（如平衡准确率或F1分数）并未利用这种结构。我们表明这种忽略是有后果的。具体而言，在3种语言的4个标准化临床错误检测测试集上评估15个不同的LLMs时，我们发现15个模型中有13个低于随机成对判别的水平，即使它们的F1分数按标准实践会被解读为中等水平。我们还观察到，潜在的偏差模式在不同语言间有所不同：同一模型可能在一个语言上默认“无错误”，而在另一个语言上过度标记错误。

    arXiv:2608.16643v1 Announce Type: cross  Abstract: Automated detection of errors in clinical documentation is a promising application of large language models (LLMs), yet decisions to deploy such models rest on benchmarks that evaluate each clinical note in isolation. Error-detection benchmarks are typically constructed by injecting errors into notes, such that each erroneous note has a natural counterpart. Aggregate discriminative metrics (e.g., balanced accuracy or F1) do not exploit this structure. We show that this omission is consequential. In particular, evaluating 15 diverse LLMs on 4 standardized clinical error-detection test sets across 3 languages, we find that 13 of 15 models fall below the level of random pairwise discrimination, even while achieving F1 scores that standard practice would read as moderate. We also observe that the underlying bias patterns differ across languages: the same model can default to "no error" on one language and over-flag errors on another. To di
    
[^14]: 解释何时有助于上下文学习？自然语言解释类型与忠实性的比较研究

    When Do Explanations Help In-Context Learning? A Comparative Study of Natural Language Explanation Types and Faithfulness

    [https://arxiv.org/abs/2608.16627](https://arxiv.org/abs/2608.16627)

    本论文通过跨多个基准和模型的比较研究，发现外部LLM生成的自然语言解释在上下文学习中能有效提升分类任务的准确性，其效用可与人类编写的解释相媲美，且解释的忠实性选择对下游性能有显著影响。

    

    arXiv:2608.16627v1 公告类型：交叉 摘要：自然语言解释（NLEs）越来越多地被用作输入，例如，作为影响上下文学习（ICL）中模型行为的少样本推理。然而，目前尚不清楚不同类型的NLEs在增强提示中如何影响下游模型性能。因此，我们提供了跨六个基准和四个指令调优模型的比较评估，研究了NLE来源（可用时的人类编写、自生成解释、由外部LLM生成）和NLE选择（随机与基于忠实性的过滤）如何在ICL设置中影响NLEs的下游效用。我们的广泛评估表明，在分类风格的基准上，将NLEs添加到少样本提示中通常能提高无解释的少样本提示的准确性；在NLE来源中，外部生成的LLM-NLEs通常提供强大的下游效用，并与人类编写的解释保持竞争力。

    arXiv:2608.16627v1 Announce Type: cross  Abstract: Natural language explanations (NLEs) are increasingly used as inputs, for example, as few-shot rationales that influence model behavior in in-context learning (ICL). However, it remains unclear how different types of NLEs compare in their effects on downstream model performance in explanation-augmented prompting. Therefore, we provide a comparative evaluation across six benchmarks and four instruction-tuned models, studying how NLE source (human-written when available, self-generated explanations, generated by an external LLM) and NLE selection (random vs faithfulness-based filtering) affect downstream utility of NLEs when used in ICL settings. Our extensive evaluation shows that, on classification-style benchmarks, adding NLEs to few-shot prompts often improves accuracy over few-shot prompting without explanations; among NLE sources, externally generated LLM-NLEs often provide strong downstream utility and remain competitive with huma
    
[^15]: Palmyra x6技术报告：通过锚定监督微调后训练的代理型工具使用模型

    Palmyra x6 Technical Report: An Agentic, Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning

    [https://arxiv.org/abs/2608.16620](https://arxiv.org/abs/2608.16620)

    Palmyra x6通过锚定监督微调和保守训练策略，在少量数据上实现了企业代理任务中的显著性能提升，并在多个基准测试中领先。

    

    arXiv:2608.16620v1 公告类型：交叉 摘要：Palmyra x6是一个针对企业导向代理任务优化的大型语言模型。该模型通过在紧凑的已验证合成工具使用轨迹语料库上，对混合专家基础模型进行锚定监督微调，并使用Muon + Adam混合优化器进行后训练构建而成。该配方刻意保守且受控：626条轨迹、单轮训练、低学习率，以及一个冻结基础的KL锚定。该模型在Writer Agent任务上相比之前的默认模型显示出显著提升，并在公开基准测试中与多个近期模型相比表现优异，在BFCL Core上得分最高，为0.785，并取得了该组六个基准测试的最高平均值。此外，在我们的偏见和安全评估中，该模型相对于比较对象表现出竞争力或领先性。

    arXiv:2608.16620v1 Announce Type: cross  Abstract: Palmyra x6 is a large language model optimized for use with enterprise-oriented agentic tasks. The model was built by post-training a Mixture-of-Experts base model with Anchored Supervised Fine-Tuning on a compact corpus of verified, synthetic tool-use trajectories, optimized with a Muon + Adam hybrid. The recipe is deliberately conservative and deliberately controlled: 626 trajectories, a single epoch, a low learning rate, and a KL anchor to the frozen base. The model shows substantial gains over the previous default model for Writer Agent, and compares favorably with several recent models on public benchmarks, scoring the highest on BFCL Core at $0.785$ and posts the highest six-benchmark mean of the cohort. Furthermore, the model has shown itself to be competitive or leading relative to comparators in our bias and safety evaluations.
    
[^16]: BabelSteering：通过英语转向向量实现多语言安全对齐

    BabelSteering: Multilingual Safety Alignment via English Steering Vectors

    [https://arxiv.org/abs/2608.16577](https://arxiv.org/abs/2608.16577)

    本文提出BabelSteering方法，利用英语安全监督中的拒绝方向作为轻量级推理时干预，显著提升多语言模型对有害请求的拒绝能力，且几乎不影响任务效用。

    

    arXiv:2608.16577v1 公告类型：新 摘要：大型语言模型（LLMs）在全球高风险场景中部署，但大多数安全研究和对齐工作仍集中在英语上。因此，使用其他语言与LLM交互的用户可能会遇到较弱的安全保障，尽管他们在类似敏感任务中依赖相同的系统。在这项工作中，我们研究了从高资源语言（如英语）学习的安全信号是否能提高多语言安全性。我们提出了BabelSteering，一种激活转向方法，作为轻量级推理时干预，利用从英语安全监督中获得的拒绝方向来跨语言泛化。我们的评估涵盖八种语言，并同时测量对有害请求的拒绝、过度拒绝和一般任务效用。结果表明，BabelSteering提高了跨语言对有害请求的拒绝率，任务效用的降低微乎其微甚至没有，但存在一些过度拒绝的情况。

    arXiv:2608.16577v1 Announce Type: new  Abstract: Large language models (LLMs) are deployed globally in high-stakes settings, yet most safety research and alignment efforts remain concentrated on English. Thus, users interacting with LLMs in other languages may encounter weaker safeguards despite relying on the same systems for similarly sensitive tasks. In this work, we investigate whether safety signals learned from a high-resource language, like English, can improve multilingual safety. We propose BabelSteering, an activation steering method that acts as a lightweight inference- time intervention, using refusal directions derived from English safety supervision to generalize across languages. Our evaluation includes eight languages and jointly measures refusal of harmful requests, over-refusal, and general task utility. The results show that BabelSteering increases the refusal of harmful requests across languages, with only a marginal to no reduction in task utility but with some inc
    
[^17]: 提问、条件化或弃权：面向缺失前提推理的强化学习

    Ask, Condition or Abstain: Reinforcement Learning for Missing-Premise Reasoning

    [https://arxiv.org/abs/2608.16554](https://arxiv.org/abs/2608.16554)

    本文提出ACA-RL框架，通过数据增强和结构化奖励训练模型在缺失前提时选择提问、条件化回答或弃权，并引入人工验证的MPB基准以提升推理鲁棒性。

    

    仅答案式强化学习（RL）训练推理模型解决完全明确的问题，但许多现实查询省略了得出唯一答案所需的前提。在这种情况下，有用的响应并不总是拒绝：模型应询问缺失的前提，根据未知量条件化其答案，或在无法提供有信息量的条件响应时弃权。我们提出了《提问-条件化-弃权强化学习》（ACA-RL），一种针对此设置的数据增强强化学习框架。其基于推理图引导的流程将良构问题转换为带有局部缺口标注的缺失前提训练实例；ACA-RL随后使用覆盖五种可观察响应行为的结构化奖励对这些实例进行训练。我们还引入了《缺失前提基准》（MPB），一个包含274个实例、经人工验证的基准，涵盖数学、逻辑和现实世界文字问题。在Qwen3和Llama模型上，ACA-RL表现出显著改进。

    arXiv:2608.16554v1 Announce Type: new  Abstract: Answer-only reinforcement learning (RL) trains reasoning models to solve fully specified problems, but many realistic queries omit a premise needed for a unique answer. In this setting, the useful response is not always refusal: the model should ask for the missing premise, condition its answer on the unknown quantity, or abstain when no informative conditional response is available. We present \emph{Ask-Condition-Abstain Reinforcement Learning} (ACA-RL), a data-augmented RL framework for this setting. Its reasoning-graph-guided pipeline converts well-posed problems into missing-premise training instances with localized gap annotations; ACA-RL then trains on these instances with a structured reward over five observable response behaviors. We also introduce the \emph{Missing-Premise Benchmark} (MPB), a 274-instance human-verified benchmark spanning mathematical, logical, and real-world word problems. Across Qwen3 and Llama models, ACA-RL 
    
[^18]: 阶段式：面向多偏好LLM对齐的受控目标准入

    STAGE: Controlled Objective Admission for Multi-Preference LLM Alignment

    [https://arxiv.org/abs/2608.16553](https://arxiv.org/abs/2608.16553)

    本文提出了一种基于稳定性引导的主动集控制方法（STAGE），通过门控准入和探测顺序，在多偏好对齐中逐步引入目标，显著优于同时标量化方法。

    

    多偏好对齐通常被表述为标量化：合并奖励维度，然后进行优化。这留下了时间决策的未定义之处：每个偏好维度何时应进入策略优化？我们提出了\methodname，一种稳定性引导的主动集控制器，用于受控目标准入。\methodname从一个小的主动集开始，保留已准入的目标，并在奖励偏差门控指示近期偏差较低或耐心预算耗尽时进行扩展。探测阶段估计一个从难到易的顺序，自适应加权强调表现不佳的主动维度。使用15个训练偏好和16个保留基准列的自动评估表明，\methodname在平均值上高于同时标量化和共享预算自适应基线。组件消融和扩展动态进一步支持累积保留、门控准入和探测推导顺序作为有用设计。

    arXiv:2608.16553v1 Announce Type: new  Abstract: Multi-preference alignment is often framed as scalarization: combine reward dimensions, then optimize. This leaves a temporal decision underspecified: when should each preference dimension enter policy optimization? We propose \methodname, a stability-guided active-set controller for controlled objective admission. \methodname starts from a small active set, retains admitted objectives, and expands when reward-deviation gates indicate low recent deviation or a patience budget is exhausted. A probing phase estimates a hard-to-easy order, and adaptive weighting emphasizes underperforming active dimensions. Automatic evaluations with 15 training preferences and 16 held-out benchmark columns show that \methodname obtains higher averages than simultaneous scalarization and shared-budget adapted baselines. Component ablations and expansion dynamics further support cumulative retention, gated admission, and probing-derived ordering as useful de
    
[^19]: 倾听、推理与分割：将大型音频语言模型与编辑判断对齐以实现媒体章节化

    Listen, Reason, and Segment: Aligning LALMs with Editorial Judgment for Media Chapterization

    [https://arxiv.org/abs/2608.16539](https://arxiv.org/abs/2608.16539)

    本文提出AudioChaps后训练框架，利用GRPO和思维链推理将大型音频语言模型与编辑判断对齐，以解决依赖主观决策的媒体章节化任务，并配套构建了三个专用数据集。

    

    arXiv:2608.16539v1 公告类型：交叉 摘要：大型音频语言模型（LALMs）在标准化基准测试中取得了快速进展，但它们在现实媒体工作流程、内容策划、档案索引和内容分发中的部署仍未充分实现。我们识别出自动音频章节化——即将连续音频流分割成主题连贯的章节——作为一个要求高且具有商业重要性的场景，它暴露了这一差距。章节化具有挑战性，因为边界定义更多依赖于主观编辑判断，而非客观声学事件，要求模型在长音频上下文中进行顺序推理，并近似创作者撰写的边界决策。我们提出了AudioChaps，一个通过组相对策略优化（GRPO）并由思维链（CoT）推理引导的后训练框架，用于对齐端到端LALMs以完成此任务。为支持训练和评估，我们策划了三个数据集：AudioChaps-Alignment、AudioChaps-Benchmark和AudioChaps-Eval。

    arXiv:2608.16539v1 Announce Type: cross  Abstract: Large Audio Language Models (LALMs) have made rapid progress on standardized benchmarks, yet their deployment in practical media workflows, curation, archival indexing, and content distribution remains largely unrealized. We identify automated audio chapterization, the task of segmenting continuous audio streams into thematically coherent chapters, as a demanding and commercially consequential setting that exposes this gap. Chapterization is challenging because boundaries are defined less by objective acoustic events than by subjective editorial judgment, requiring models to reason sequentially over long acoustic contexts and approximate creator-authored boundary decisions. We present AudioChaps, a post-training framework for aligning end-to-end LALMs for this task via Group Relative Policy Optimization (GRPO) guided by Chain-of-Thought (CoT) reasoning. To support training and evaluation, we curate three datasets: AudioChaps-Alignment,
    
[^20]: DSPrompt：针对M-RAG损坏的动态软提示防御

    DSPrompt: Dynamic Soft Prompt Defense Against M-RAG Corruption

    [https://arxiv.org/abs/2608.16536](https://arxiv.org/abs/2608.16536)

    本文提出DSPrompt，一种通过在各编码器层动态插入软提示来重塑检索嵌入语义的防御框架，无需修改检索管道，从而有效抵御M-RAG对抗性攻击并降低推理开销。

    

    多模态检索增强生成（M-RAG）正日益受到对抗性攻击的威胁，其中恶意数据被精心制作，以生成与向量空间中良性条目对齐的嵌入，从而欺骗检索并诱导有害输出。现有防御措施主要在查询时运行，依赖辅助检测器、相似性重排序或特征一致性检查。然而，这些方法存在显著的推理开销、对未见攻击策略的泛化能力差，且通常假设特定的攻击分布。为解决此问题，我们提出DSPrompt，一种动态软提示防御框架，直接重塑检索器的嵌入语义，而不修改检索管道。它在冻结检索器的视觉和文本编码器的每一层中插入少量可学习的软提示，利用适应模型层容量的浅到深长度调度。

    arXiv:2608.16536v1 Announce Type: cross  Abstract: Multimodal Retrieval Augmented Generation (M-RAG) is increasingly vulnerable to adversarial attacks where malicious data are crafted to produce embeddings that align with benign entries in the vector space, deceiving retrieval and inducing harmful outputs. Existing defenses primarily operate at query time, relying on auxiliary detectors, similarity re-ranking, or feature-consistency checks. However, these approaches suffer from non-trivial inference overhead, generalize poorly to unseen attack strategies, and often assume specific attack distributions. To address this, we propose DSPrompt, a Dynamic Soft Prompt defense framework that directly reshapes the retriever's embedding semantics, without modifying the retrieval pipeline. It inserts few learnable soft prompts into each layer of the visual and textual encoders of a frozen retriever, utilizing a shallow-to-deep length schedule that is adaptive to the capacity in the model layers. 
    
[^21]: 当上下文误导时：意图引导解码实现稳健的检索增强生成

    When Context Misleads: Intent-Guided Decoding for Robust Retrieval-Augmented Generation

    [https://arxiv.org/abs/2608.16515](https://arxiv.org/abs/2608.16515)

    本文提出意图引导解码（IGD）框架，根据用户意图动态平衡检索上下文与参数记忆，通过答案级过滤和令牌级修正显著提升RAG在误导性上下文下的事实恢复能力。

    

    摘要：检索增强生成（RAG）通过将生成过程锚定在外部证据上来改进大型语言模型，但它也引入了来源信任问题：检索到的上下文可能是有用的、无关的，甚至是误导性的。现有的RAG系统通常对检索到的证据采用固定的信任策略，这可能导致过度信任不正确的上下文，或者在用户明确要求遵循上下文时未能充分利用上下文。因此，我们提出了意图引导解码（IGD），这是一个根据用户意图在检索上下文和参数记忆之间进行仲裁的框架。IGD使用答案级过滤和令牌级修正来引导检索上下文与参数记忆之间的最终解码轨迹。我们在五个大型语言模型上的三个忠实问答基准和三个事实冲突基准上评估了IGD，IGD显著提高了事实恢复能力，在事实冲突基准上实现了高达65.4个百分点的提升。

    arXiv:2608.16515v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) improves large language models by grounding generation in external evidence, but it also introduces a source trust problem: retrieved context may be useful, irrelevant, or even misleading. Existing RAG systems often apply a fixed trust policy toward retrieved evidence, which can either over-trust incorrect context or underuse context when the user explicitly asks for context-following behavior. Therefore, we propose Intent-Guided Decoding (IGD), a framework that arbitrates between retrieved context and parametric memory according to user intent. IGD uses answer-level filtering and token-level correction to steer the final decoding trajectory between retrieved context and parametric memory. We evaluate IGD on three faithful QA benchmarks and three factual-conflict benchmarks across five LLMs, IGD substantially improves factual recovery, achieving gains of up to 65.4 percentage points on factual-confl
    
[^22]: 匹配的结果，不同的注视：有中央凹的多模态大语言模型与人类搜索的对比

    Matched Outcomes, Divergent Gaze: How Foveated MLLMs Search Compared to Humans

    [https://arxiv.org/abs/2608.16514](https://arxiv.org/abs/2608.16514)

    多模态大语言模型在目标存在性判断和获取效率上匹配或超越人类，但其注视过程与人类显著不同，表现为低熵模式。

    

    arXiv:2608.16514v1 公告类型：交叉 摘要：人类视觉搜索是序列性的：中央凹必须落在候选对象上以确认它，这些落点形成扫描路径。当多模态大语言模型（MLLMs）接收到相同的中央凹输入时，它们是否像人类一样搜索，这关系到它们作为人类视觉模型的使用以及注意力对齐分数。我们在目标导向搜索（COCO-Search18）中，将三种通用MLLMs与人类眼动扫描路径进行比较，逐个注视点地驱动每个模型通过一个相同的、与人类匹配的中央凹视图，并从三个维度评估：目标存在性的决策、达到目标的效率以及注视过程本身。这些维度是分离的。在决策和目标获取上，模型达到或超过人类，检测存在目标的性能接近天花板，并且比人类更常在第一次扫视时到达目标。但注视过程不像人类。在人类匹配条件下，所有三种模型共享一个特征：低熵。

    arXiv:2608.16514v1 Announce Type: cross  Abstract: Human visual search is serial: the fovea must land on a candidate to confirm it, and those landings form a scanpath. Whether multimodal large language models (MLLMs), given the same foveated input, search as humans do bears on their use as models of human vision and on attention-alignment scores. We compare three general-purpose MLLMs with human eye-movement scanpaths on goal-directed search (COCO-Search18), driving each model fixation by fixation through an identical, human-matched foveated view and assessing it along three axes: the decision of target presence, the efficiency of reaching the target, and the gaze process itself. The axes dissociate. On the decision and on target acquisition the models match or exceed humans, detecting present targets near ceiling and reaching them on the first saccade more often than people do. The gaze process is not human. Under the human-matched condition, all three share one signature: low-entropy
    
[^23]: 计算型KJ法：使用领域专用大语言模型从大规模定性数据中提取无分析师偏差洞察的框架

    Computational KJ-Ho: An Analyst-Bias-Free Insight Extraction Framework from Large-Scale Qualitative Data Using Domain-Specialized LLMs

    [https://arxiv.org/abs/2608.16467](https://arxiv.org/abs/2608.16467)

    本文提出一种计算型KJ法框架，利用领域专用大语言模型实现无分析师偏差的定性数据洞察提取，以克服人类认知限制和偏差。

    

    定性研究方法——如KJ法、扎根理论和主题分析——支撑着消费者洞察的生成，但它们共享一个结构性限制：人类分析师的认知处理能力。复制研究进一步表明，不同分析师对相同数据的分析结论差异显著（即分析师偏差）。本文提出了计算型KJ-Ho（即川喜田二郎法），这是一个理论框架，通过计算方式实现KJ法的认识论——让结构从数据本身涌现，而不强加分析师的先入之见——我们将这种取向称为“无分析师偏差”。该框架采用领域专用大语言模型，该模型通过持续预训练（CPT）于营销研究语料库，并在专家策划的洞察对上进行监督微调（SFT），组织为三层架构：数据结构化、洞察提取和策略生成。两个初步...

    arXiv:2608.16467v1 Announce Type: cross  Abstract: The qualitative research methodologies that underpin consumer-insight generation - the KJ method, Grounded Theory, and Thematic Analysis - share a structural constraint: the cognitive processing capacity of the human analyst. Replication research further shows that conclusions vary substantially across analysts analyzing identical data (analyst bias). This paper proposes Computational KJ-Ho (the Kawakita Jiro method), a theoretical framework that computationally realizes the KJ method's epistemology - letting structure emerge from the data itself without imposing the analyst's preconceptions - an orientation we term "analyst-bias-free." The framework employs a domain-specialized LLM built through continued pre-training (CPT) on a marketing-research corpus and supervised fine-tuning (SFT) on expert-curated insight pairs, organized as a three-layer architecture: data structuring, insight extraction, and strategy generation. Two prelimina
    
[^24]: D2-ScaleAgent：面向长文档理解的双维度扩展

    D2-ScaleAgent: Dual-Dimensional Scaling for Long Document Understanding

    [https://arxiv.org/abs/2608.16417](https://arxiv.org/abs/2608.16417)

    本文提出D2-ScaleAgent，通过验证器驱动的动态路由机制，在检索和推理两个维度上按查询难度动态扩展计算，克服了现有多模态RAG固定工作流的局限，提升了长文档理解的证据充分性。

    

    arXiv:2608.16417v1 公告类型：新  摘要：多模态检索增强生成（RAG）是视觉丰富长文档理解的关键技术。现有的多模态RAG方法正逐步向多智能体系统发展：它们首先根据查询检索相关页面，然后迭代理解这些页面中的信息。然而，这些方法通常依赖固定工作流，缺乏在测试时动态扩展计算的能力，往往导致证据不足。为解决这一问题，我们提出了D2-ScaleAgent，一种引入双维度扩展范式用于检索和推理的智能体框架。D2-ScaleAgent的核心是一个由验证器智能体驱动的动态路由循环，基于查询的内在难度，围绕一个持续更新的证据库（作为智能体的动态工作记忆）进行：当需要扩展检索时，智能体向外路由（检索扩展），分解查询并扩大搜索范围；当需要深化推理时，智能体向内路由（推理扩展），对已收集的证据进行更深入的分析。

    arXiv:2608.16417v1 Announce Type: new  Abstract: Multi-modal retrieval-augmented generation (RAG) is a key technique for visually rich long document understanding. Existing multi-modal RAG methods are progressively advancing toward multi-agent systems: they first retrieve relevant pages based on a query, and then iteratively understand information within those pages. However, these methods typically rely on fixed workflows and lack the ability to dynamically scale computation at test time, often leading to insufficient evidence. To address this, we propose D2-ScaleAgent, an agentic framework that introduces a dual-dimensional scaling paradigm for retrieval and reasoning. The core of D2-ScaleAgent is a Verifier agent-driven dynamic routing loop based on the intrinsic difficulty of the query, centered around a continuously updated evidence bank that serves as the agent's dynamic working memory: when retrieval needs to be expanded, the agent routes outward (retrieval scaling), decomposing
    
[^25]: 计数文档并非计数文本：Web-PDF语料库统计中的单位偏差

    Counting Documents Is Not Counting Text: Unit Bias in Web-PDF Corpus Statistics

    [https://arxiv.org/abs/2608.16390](https://arxiv.org/abs/2608.16390)

    本文揭示了Web-PDF语料库中按文档计数与按令牌计数的巨大偏差，导致令牌总数被高估且截断文本大量丢失，影响语料库统计的准确性。

    

    arXiv:2608.16390v1 公告类型：交叉 摘要：PDF语料库以令牌数宣传其规模，但计算其发布的每个比率（覆盖率、OCR路由、重新获取恢复、语言混合）时均以文档为单位，且没有一个比率分解其令牌总数。这两种单位差异显著。在CC-MAIN-2021-31-PDF-UNTRUNCATED（790万份网页PDF，326亿令牌）中，3.02%的含文本文档占有一半的令牌（基尼系数0.807）；超过50页的文档占语料库的5.00%，但占其文本的53.53%。由TeX工具链生成的PDF占文档的1.66%，占文本的4.05%。最明显的受害者是Common Crawl的截断上限：它影响了23.06%的文档和63.08%的文本。重建被截断的文件并提取两个版本，两个广泛使用的库恢复了该文本的11.4%和1.4%；72%至97%的受影响文档未产生任何内容；语料库约55-62%的文本丢失。在2025年3月采用的5 MiB上限下，仍有30.19%的令牌会被截断，且恢复率...

    arXiv:2608.16390v1 Announce Type: cross  Abstract: PDF corpora advertise their size in tokens but compute every rate they publish (coverage, OCR routing, re-fetch recovery, language mix) per document, and none decomposes its token total. The two units diverge sharply. On CC-MAIN-2021-31-PDF-UNTRUNCATED (7.9M web PDFs, 32.6B tokens), 3.02% of text-bearing documents hold half the tokens (Gini 0.807); documents over 50 pages are 5.00% of the corpus but 53.53% of its text. The PDFs produced by a TeX{} toolchain are 1.66% of documents and 4.05% of the text. The clearest casualty is Common Crawl's truncation cap: it affected 23.06% of documents and 63.08% of the text. Reconstructing the truncated files and extracting both versions, two widely used libraries recover 11.4% and 1.4% of that text; between 72% and 97% of affected documents yield nothing; roughly 55--62% of the corpus's text is lost. Under the 5 MiB cap adopted in March 2025, 30.19% of tokens would still be truncated, and recovery
    
[^26]: Mint-Agent：引入金融原生的智能体基础模型

    Mint-Agent: Introducing Finance-Native Agentic Foundation Models

    [https://arxiv.org/abs/2608.16386](https://arxiv.org/abs/2608.16386)

    本文提出Mint-Agent，一种金融原生智能体基础模型，通过数据引擎、MintHarness框架和结合SFT、OPD与RLVR的训练算法，实现可靠且可审计的长周期金融研究执行。

    

    金融智能体必须超越领域知识的回忆：它们既要可靠，能够在有根据的证据上执行精确操作；又要具备执行力，能够维持长周期研究，其结论保持可审计性。我们提出了Mint-Agent，一个围绕这两个金融智能尺度设计的金融原生智能体模型系列。Mint-Agent基于三大支柱构建：数据、框架和算法。我们的数据引擎从真实金融来源构建干净、专门的任务，用于原子金融能力和长周期智能体执行。MintHarness支持与开放环境的稳定交互，并在扩展研究轨迹中维持可审计的证据链。我们的训练配方结合了SFT、关键步骤OPD和RLVR，以开发独立的金融推理和智能体执行专家，然后通过模型合并和多教师在线策略蒸馏统一成紧凑模型。

    arXiv:2608.16386v1 Announce Type: new  Abstract: Financial agents must do more than recall domain knowledge: they must be both reliable, executing precise operations over grounded evidence, and executive, sustaining long-horizon research whose conclusions remain auditable. We present Mint-Agent, a family of finance-native agentic models designed around these two scales of financial intelligence. Mint-Agent is built upon three pillars: data, harness, and algorithm. Our data engine constructs clean, specialized tasks for atomic financial capabilities and long-horizon agentic execution from real-world financial sources. MintHarness enables stable interaction with open-ended environments and maintains auditable evidence trails across extended research trajectories. Our training recipe combines SFT, critical-step OPD, and RLVR to develop separate financial reasoning and agentic execution experts, which are then unified through model merging and multi-teacher on-policy distillation into comp
    
[^27]: 未适配的多语言ASR在Garrusi库尔德语评估集上的表现：一种共同参考分阶段归一化分析

    Unadapted Multilingual ASR on a Garrusi Kurdish Evaluation Set: A Common-Reference Staged Normalization Analysis

    [https://arxiv.org/abs/2608.16379](https://arxiv.org/abs/2608.16379)

    本文通过共同参考分阶段归一化方法，揭示了未适配的多语言ASR在Garrusi库尔德语上因书写系统差异导致的严重高估错误率，并提供了更公平的评估基准。

    

    arXiv:2608.16379v1 公告类型：新 摘要：评估一种用拉丁田野正字法书写的库尔德语变体的语音识别，而模型输出的是阿拉伯文字，这首先就造成了测量问题而非建模问题：直接评分将书写系统差异视为识别错误。联合归一化参考和假设可以避免这个问题，但也会改变参考分词，将一致性提升与评分分母的变化混在一起。我使用MMS-1B-all模型及其Central Kurdish (ckb)适配器（未做调整，按发布原样使用），在来自五位说话者的1,722个Garrusi问卷片段（9,763个参考词符；117.9分钟）上进行评估。我采用共同参考设计：参考被折叠一次并固定为9,763个词符，而只有假设表示会变化。原始的阿拉伯文字假设得分为111.70%的词错误率（WER）和100.92%的字符错误率（CER），且精确词匹配为零。拉丁转写得到102.36%的WER和57.89%的CER；将其折叠进参考的...

    arXiv:2608.16379v1 Announce Type: new  Abstract: Evaluating speech recognition for a Kurdish variety written in a Latin field orthography, using a model that outputs Arabic script, creates a measurement problem before a modelling one: direct scoring treats writing-system differences as recognition errors. Jointly normalizing reference and hypothesis avoids this, but also changes reference tokenization, mixing agreement gains with a change in the scoring denominator. I evaluate MMS-1B-all with the Central Kurdish (ckb) adapter, used as released without adaptation, on 1,722 Garrusi questionnaire segments from five speakers (9,763 reference word tokens; 117.9 minutes). I use a common-reference design: the reference is folded once and fixed at 9,763 tokens, while only the hypothesis representation varies. The raw Arabic-script hypothesis scores 111.70% WER and 100.92% CER, with zero exact word matches. Latin transliteration gives 102.36% WER and 57.89% CER; folding it into the reference's 
    
[^28]: HalluTracer：通过深度平均真值信号进行幻觉检测

    HalluTracer: Hallucination Detection via Depth-Averaging Truth Signals

    [https://arxiv.org/abs/2608.16353](https://arxiv.org/abs/2608.16353)

    HalluTracer通过聚合前向传播所有层的真值信号，利用弱相关的逐层证据进行深度平均，显著提升了幻觉检测的准确性。

    

    即使是对齐良好的大型语言模型也会自信地生成事实错误的文本，这使得幻觉成为高风险部署中持续存在的可靠性风险。然而，这些模型在其内部表示中携带线性可分离的真值信号。现有的白盒检测器将这些证据压缩到孤立组件或单一深度，丢弃了贯穿整个前向传播过程中分布的判别信息。我们引入了HalluTracer，这是一个检测框架，它在模型发出任何答案标记之前，读取并聚合前向传播每一层的真值证据。几何分析显示，逐层信号相关性较弱，因此简单的深度平均可以抑制层特定噪声，并捕获几乎所有线性可访问的信息。在六个开源语言模型和五个幻觉基准测试中，HalluTracer始终优于匹配的现有方法。

    arXiv:2608.16353v1 Announce Type: cross  Abstract: Even well-aligned large language models confidently generate factually incorrect text, making hallucination a persistent reliability risk in high-stakes deployments. These models nonetheless carry linearly separable truthfulness signals in their internal representations. Existing white-box detectors, however, collapse this evidence to isolated components or a single depth, discarding discriminative information distributed across the full forward pass. We introduce HalluTracer, a detection framework that reads and aggregates truthfulness evidence across every layer of the forward pass before the model emits any answer token. A geometric analysis reveals that the per-layer signals are weakly correlated, so that simple depth averaging suppresses layer-specific noise and captures nearly all linearly accessible information. Across six open-source language models and five hallucination benchmarks, HalluTracer consistently outperforms matched
    
[^29]: 架构相关的跨大型语言模型激活状态因果迁移

    Architecture-Dependent Causal Transfer of Activation States Across Large Language Models

    [https://arxiv.org/abs/2608.16347](https://arxiv.org/abs/2608.16347)

    本文证明通过学习投影可将激活状态在不同LLM架构间因果迁移，并提出基于秩的互k近邻对齐度量，优于现有方法。

    

    arXiv:2608.16347v1 公告类型：新  摘要：人工智能系统之间的直接通信依赖于自然语言作为中间层，这会导致编码/解码开销、令牌成本和延迟。我们探究是否可以通过学习到的投影，将内部激活状态在不同的大型语言模型（LLM）架构之间进行因果迁移，并在三个层面进行评估：表示相似性、从投影状态进行的跨模型检索，以及在生成过程中通过激活注入进行的端到端因果迁移。使用四个架构多样的开源权重模型（Qwen2-0.5B、Phi-3-mini、Mistral-7B、FLAN-T5-base），我们发现训练模型中表示对齐度超过随机初始化基线，并且最好通过基于秩的度量（互k近邻对齐）来捕捉，该度量比中心核对齐（CKA）或普氏分析对激活幅度异常值更鲁棒。学习到的投影网络能够检索到正确的目标。

    arXiv:2608.16347v1 Announce Type: new  Abstract: Direct communication between AI systems relies on natural language as an intermediate layer, incurring encoding/decoding overhead, token cost, and latency. We ask whether internal activation states can instead be transferred causally between different large language model (LLM) architectures via a learned projection, evaluated at three levels: representational similarity, cross-model retrieval from projected states, and end-to-end causal transfer via activation injection during generation. Using four architecturally diverse open-weight models (Qwen2-0.5B, Phi-3-mini, Mistral-7B, FLAN-T5-base), we find that representational alignment in trained models exceeds a random-initialization null baseline and is best captured by a rank-based metric (mutual k-nearest-neighbour alignment), more robust to activation-magnitude outliers than centered kernel alignment (CKA) or Procrustes analysis. A learned projection network retrieves the correct targe
    
[^30]: IndicQE-APE：面向印度语言的翻译质量评估与自动后编辑基准

    IndicQE-APE: A Benchmark for Quality Estimation and Automatic Post-Editing for Indic Languages

    [https://arxiv.org/abs/2608.16344](https://arxiv.org/abs/2608.16344)

    本文整合了印度语言的质量评估和自动后编辑数据，创建了一个包含多标签和难度分层的基准，并评估了多种模型，发现只有同时利用整体和词级信息的系统在严格对照下表现显著。

    

    摘要：arXiv:2608.16344v1 公告类型：新 摘要：印度语言的质量评估（QE）和自动后编辑（APE）数据分散在不同的发布中，因此没有单一资源能够在同一基础上支持跨任务和语言对的训练与评估。我们将WMT 2020--2024共享任务系列与扩展的英语--马拉雅拉姆语资源整合为\indicqe：包含126,754个实例，覆盖九个方向性语言对，每个片段上最多有四种标签类型对齐，包括直接评估、人工后编辑、词级OK/BAD标签和错误解释，以及按四个难度轴分层的测试集。在此基准上，我们评估了六个提示的大型语言模型和三个COMET指标在片段级QE上的表现，以及三个系统在APE上的表现。其中两个轴部分基于直接评估并选择其压缩子集，因此每个轴与从相同语言对和相同分数分布中抽取的对照组进行比较。只有一种方法在对照组中幸存：整体和词级对齐的片段。

    arXiv:2608.16344v1 Announce Type: new  Abstract: Indic quality estimation (QE) and automatic post-editing (APE) data is spread across separate releases, so no single resource supports training and evaluation across tasks and language pairs on one footing. We consolidate the WMT 2020--2024 shared-task lineage with an extended English--Malayalam resource into \indicqe: $126{,}754$ instances over nine directional pairs, with up to four label types aligned on the same segment, a direct assessment, a human post-edit, word-level OK/BAD tags and an error explanation, and a test set stratified over four difficulty axes. On it, we benchmark six prompted LLMs and three COMET metrics on segment-level QE, and three systems on APE. Two of the axes are defined partly on the direct assessment and select a compressed slice of it, so each axis is compared against a control drawn from the same language pair with the same score distribution. Only one survives that control: segments whose holistic and tok
    
[^31]: 步骤级在线策略蒸馏：在线策略蒸馏与监督微调之间的插值

    Step-Level On-Policy Distillation: Interpolating Between On-Policy Distillation and Supervised Fine-Tuning

    [https://arxiv.org/abs/2608.16333](https://arxiv.org/abs/2608.16333)

    本文提出步骤级在线策略蒸馏（SOPD），通过结合监督微调的长程修正与在线策略蒸馏的优势，在完整学生轨迹上提供步骤级监督，从而克服令牌级OPD的碎片化修正局限。

    

    arXiv:2608.16333v1 公告类型：交叉 摘要：在线策略蒸馏（OPD）将学生模型与教师在学生生成的轨迹上的logit分布对齐。这种方法取得了显著的实证收益，并且通常能以更少的数据超越传统的离线策略蒸馏。然而，标准的令牌级OPD只能沿着错误的学生轨迹提供碎片化的修正，无法展开完整且正确的修复路径。受此局限性的启发，我们提出了*步骤级在线策略蒸馏*（SOPD），它结合了监督微调（SFT）的长程修正与OPD的在线策略优势，在完整的学生生成轨迹上提供步骤级监督。我们表明，在步骤长度的不同极限下，SOPD可退化为SFT或近似OPD。与SFT相比，SOPD中的教师响应以学生轨迹为条件，因此更紧密地对齐学生访问过的状态。

    arXiv:2608.16333v1 Announce Type: cross  Abstract: On-policy distillation (OPD) aligns a student model with a teacher's logit distribution on student-generated trajectories. This approach has achieved strong empirical gains and can often surpass conventional off-policy distillation with substantially less data. However, standard token-level OPD can provide only fragmented corrections along an erroneous student trajectory and cannot unfold a complete and correct repair path. Motivated by this limitation, we propose \emph{Step-Level On-Policy Distillation} (SOPD), which combines the long-horizon correction of supervised fine-tuning (SFT) with the on-policy advantage of OPD to provide step-level supervision over complete student-generated trajectories. We show that, at different limits of step length, SOPD reduces to SFT or approximates OPD. Compared with SFT, the teacher responses in SOPD are conditioned on student trajectories and therefore align more closely with student-visited states
    
[^32]: 深度思维对齐：用于视频推理的轨迹级潜在蒸馏

    Deep Thought Alignment: Trajectory-Level Latent Distillation for Video Reasoning

    [https://arxiv.org/abs/2608.16316](https://arxiv.org/abs/2608.16316)

    本文提出Latent-OPD方法，通过在轨迹末端进行潜在表示蒸馏，弥补了传统输出级蒸馏在视频推理中无法直接约束中间推理状态的不足，从而提升小模型从大模型迁移推理能力的效率。

    

    大型多模态模型（LMMs）在视频推理中一直受到处理海量视觉信息的高计算成本的阻碍。这一困境促使将大模型的推理能力转移到更小、更高效的模型上。策略内蒸馏（OPD）通过匹配学生生成轨迹上的输出令牌分布，提供了一种有前景的解决方案。然而，视频推理通常依赖于跨多个帧累积的证据。在此背景下，输出级监督仅捕捉通过令牌预测表达的信息，并未直接约束推理过程中形成的潜在表示。为解决这一局限性，我们提出了Latent-OPD，该方法通过轨迹级潜在蒸馏增强了OPD。具体而言，我们的方法聚焦于每条轨迹结束时的位置，其中隐藏状态有效地总结了累积的视觉证据。

    arXiv:2608.16316v1 Announce Type: cross  Abstract: Large Multimodal Models (LMMs) for video reasoning have long been hindered by the high computational cost of processing vast amounts of visual information. This dilemma motivates the transfer of the reasoning capabilities of large models to smaller, more efficient ones. On-Policy Distillation (OPD) offers a promising solution by matching output-token distributions along student-generated trajectories. However, video reasoning often depends on evidence accumulated across multiple frames. In this context, output-level supervision only captures information expressed through token predictions and does not directly constrain the latent representations formed during reasoning. To address this limitation, we propose Latent-OPD, which augments OPD with trajectory-level latent distillation. Specifically, our method focuses on the position at the end of each trajectory, where hidden states effectively summarize the accumulated visual evidence an
    
[^33]: FTA-Mem：面向低密度长期对话的事实-时间-情感锚定记忆

    FTA-Mem: Fact-Time-Affect Anchored Memory for Low-Density Long-Term Dialogue

    [https://arxiv.org/abs/2608.16303](https://arxiv.org/abs/2608.16303)

    提出了一种名为FTA-Mem的结构化记忆框架，通过边界保留窗口分割和事实-时间-情感记忆单元，有效处理低密度长期对话中的信息碎片化问题，提升了长期记忆问答性能。

    

    arXiv:2608.16303v1 公告类型：新 摘要：长期情感支持代理需要记忆机制，以便在跨会话中实现个性化理解。然而，情感支持对话通常是低密度的：轮次不完整、证据分散，且用户状态随时间演变。现有记忆方法通常依赖于固定单元，如轮次级笔记或会话摘要，这可能丢失细节或引入冗余噪声。我们提出FTA-Mem，一种面向低密度长期对话的结构化记忆框架。FTA-Mem使用边界保留窗口分割（BWS）形成连贯的情境片段，并构建事实-时间-情感记忆单元（FTA单元），该单元联合编码事实内容、时间锚定和情感上下文。检索到的单元随后被综合为结构化上下文，用于生成回答。在ES-MemEval和LoCoMo上的实验表明，FTA-Mem在不同信息密度的基准上提升了长期记忆问答的整体表现。

    arXiv:2608.16303v1 Announce Type: new  Abstract: Long-term emotional-support agents require memory mechanisms for personalized understanding across sessions. However, emotional-support dialogue is often low-density: turns are incomplete, evidence is scattered, and user states evolve over time. Existing memory methods usually rely on fixed units, such as turn-level notes or session summaries, which may lose details or introduce redundant noise. We propose FTA-Mem, a structured memory framework for low-density long-term dialogue. FTA-Mem uses Boundary-preserving Window Segmentation (BWS) to form coherent situation fragments, and constructs Fact-Time-Affect Memory Units (FTA Units) that jointly encode factual content, temporal grounding, and affective context. Retrieved units are then synthesized into structured context for answer generation. Experiments on ES-MemEval and LoCoMo show that FTA-Mem improves overall long-term memory question answering across benchmarks with different informa
    
[^34]: 可执行代码知识：代码作为AI编码代理的原生、携带验证的知识表示

    Executable Code Knowledge: Code as a Native, Validation-Carrying Knowledge Representation for AI Coding Agents

    [https://arxiv.org/abs/2608.16295](https://arxiv.org/abs/2608.16295)

    本文提出可执行代码知识（ECK）作为AI编码代理的原生知识表示，通过将代码单元与验证证据和语义结合，使代理能直接获取可执行且可靠的上下文信息。

    

    arXiv:2608.16295v1 公告类型：新 摘要：AI编码代理需要的不仅仅是相关的代码片段：它们需要业务语义、验证证据、关系以及确保其上下文是最新的。现有系统通常通过检索、摘要、图、规则或逆向规格来推断或外部化这些知识。我们研究了一种互补的表示方法，其中选定的代码单元直接携带代理可用的知识。我们引入了可执行代码知识（ECK），并将可执行代码知识单元（ECKU）定义为一种源绑定对象，结合了稳定身份、语义、可执行行为、契约、证据、关系、来源、验证状态和查询接口。我们的Python原型支持代码本地编写、清单导出、证据执行、精确变更行影响、新鲜度检查和面向代理的投影。在三个真实的Python仓库和26个受控补丁任务中，直接ECK提供了可执行测试覆盖

    arXiv:2608.16295v1 Announce Type: new  Abstract: AI coding agents need more than relevant snippets: they need business semantics, validation evidence, relations, and assurance that their context is current. Existing systems usually infer or externalize this knowledge through retrieval, summaries, graphs, rules, or reverse specifications. We investigate a complementary representation in which selected code units directly carry agent-usable knowledge. We introduce Executable Code Knowledge (ECK) and define an Executable Code Knowledge Unit (ECKU) as a source-bound object combining stable identity, semantics, executable behavior, contracts, evidence, relations, provenance, validation state, and a query interface. Our Python prototype supports code-local authoring, manifest export, evidence execution, exact changed-line impact, freshness checking, and agent-facing projections. Across three real Python repositories and 26 controlled patch tasks, direct ECK provides executable test coverage 
    
[^35]: 第三种条款遭遇：大型语言模型能否取代语言教师？

    Clause Encounters of the Third Kind: Can LLMs Replace Language Teachers?

    [https://arxiv.org/abs/2608.16286](https://arxiv.org/abs/2608.16286)

    本文系统评估了大型语言模型在语言教学中的纠错和解释能力，发现它们虽能辅助教学，但在语言细微差别和文化敏感性等关键维度上仍无法完全替代人类教师。

    

    arXiv:2608.16286v1 公告类型：新 摘要：尽管目前各类组织积极鼓励在课堂中使用大型语言模型，但我们仍缺乏对这些模型在语言教学基本任务上实际表现如何的严谨、系统性评估。本文探讨了最先进的大型语言模型是否能提供语言学习者所需的纠错反馈和方法论解释。研究通过系统性地调整模型参数，测试了多个大型语言模型在识别、纠正和解释英语学习者常见错误方面的能力，以调查这些技术调整如何影响输出质量、教学清晰度和一致性，同时利用检索增强生成来查询方法论数据。评估采用了自动化指标（GLEU、BERTScore），也引入了人类专家判断，以捕捉纯计算度量所遗漏的维度：语言细微差别、文化敏感性和教学适宜性。

    arXiv:2608.16286v1 Announce Type: new  Abstract: While various organizations now actively encourage LLM use in classrooms, we still lack rigorous, systematic evaluations of how well these models actually perform the fundamental tasks of language pedagogy. This paper examines whether state-of-the-art LLMs can deliver the kind of corrective feedback and methodological explanations that language learners need. The study tests multiple large language models on their ability to identify, correct, and explain common learner mistakes in English, by systematically varying model parameters to investigate how these technical adjustments affect output quality, pedagogical clarity, and consistency, along with using retrieval-augmented generation to query methodological data. The evaluation employs automated metrics (GLEU, BERTScore) but also human expert judgments to capture dimensions that purely computational measures miss: linguistic nuance, cultural sensitivity, and instructional appropriatene
    
[^36]: PolyDebate：一种游戏化编排的多模态辩论技能练习与评估系统

    PolyDebate: A Game-Orchestrated Multimodal System for Debate Skills Practice and Evaluation

    [https://arxiv.org/abs/2608.16276](https://arxiv.org/abs/2608.16276)

    PolyDebate通过游戏化机制和多模态捕获，为英语辩论提供分阶段的一对一练习与全面评估，显著提升学习者的说服策略和表达技能。

    

    arXiv:2608.16276v1 公告类型：交叉 摘要：辩论是一种结构化的说服性沟通形式，训练论点构建、反驳、口头表达和听众意识。这些技能在教育、语言学习和专业沟通中备受重视。近期的人工智能辩论系统和基于大语言模型的裁判已推进了论点生成和辩论评估，但大多数仍以文本为中心，很少通过完整的多模态练习体验支持学习者。我们介绍了PolyDebate，一种游戏化编排的多模态英语辩论练习与评估系统。PolyDebate引导学习者与AI对手进行分阶段的一对一（1v1）辩论，同时技能卡、道具和硬币使说服策略明确化，并将练习转化为游戏式互动。在每次会话中，系统捕获学习者的语音和视觉表达证据，生成情境感知的对手回应，并产生基于评分标准的阶段级和整体反馈。

    arXiv:2608.16276v1 Announce Type: cross  Abstract: Debate is a structured form of persuasive communication that trains argument construction, rebuttal, oral delivery, and audience awareness. These skills are valued in education, language learning, and professional communication. Recent AI debate systems and LLM-based judges have advanced argument generation and debate evaluation, but most remain text-centered and rarely support learners through a complete multimodal practice experience. We introduce PolyDebate, a game-orchestrated multimodal system for English debate practice and evaluation. PolyDebate guides learners through staged one-on-one (1v1) debates with an AI opponent, while skill cards, props, and coins make persuasive strategies explicit and turn practice into a game-like interaction. During each session, the system captures learner speech and visual delivery evidence, generates context-aware opponent responses, and produces rubric-informed stage-level and overall feedback. 
    
[^37]: 基于上下文词元级语义图表示的领域无关神经主题建模

    Domain-Agnostic Neural Topic Modeling with Contextual Token-Level Semantic Graph Representation

    [https://arxiv.org/abs/2608.16269](https://arxiv.org/abs/2608.16269)

    本文提出一种领域无关的神经主题建模方法，通过可学习的词元级语义图层，在冻结预训练编码器上获取语料特定语义结构，从而提升专业语料上的主题可解释性。

    

    arXiv:2608.16269v1 公告类型：新 摘要：近期，利用预训练语言模型（PLMs）的神经主题模型通过利用通用领域预训练取得了强劲性能，但其主题可解释性在专业语料上往往下降。这一限制主要源于嵌入空间的几何结构，其中预训练期间未见过的领域特定词元会坍缩到难以区分的区域，而无论是领域特定重新训练、词级图增强还是参数高效微调，都无法在继承底层编码器容量上限的情况下重构该空间。我们的关键见解是，在词元级PLM嵌入上运行的可学习图层能够获取冻结编码器所缺乏的语料特定语义结构，因为词元级图保留了词级表示所丢弃的文档局部上下文，并且与主题目标的联合优化直接重塑了嵌入几何结构。

    arXiv:2608.16269v1 Announce Type: new  Abstract: Recent advances in neural topic models with pre-trained language models (PLMs) have achieved strong performance by leveraging general-domain pre-training, yet their topic interpretability often degrades on specialized corpora. This limitation primarily stems from the geometry of the embedding space, where domain-specific terms unseen during pre-training collapse into an indistinguishable region, and neither domain-specific re-training, word-level graph enrichment, nor parameter-efficient fine-tuning can restructure this space without inheriting the capacity ceiling of the underlying encoder. Our key insight is that a learnable graph layer operating on token-level PLM embeddings can acquire corpus-specific semantic structure that the frozen encoder lacks, because token-level graphs preserve document-local context that word-level representations discard and joint optimization with the topic objective reshapes embedding geometry directly fr
    
[^38]: STAIR：用于时间问答中可解释推理的语义-时间自动机

    STAIR: Semantic-Temporal Automaton for Interpretable Reasoning in Temporal Question Answering

    [https://arxiv.org/abs/2608.16224](https://arxiv.org/abs/2608.16224)

    STAIR通过将语义解释与确定性时间推理分离，减少了LLM在时间问答中的概率性错误，并提高了推理的可解释性和可验证性。

    

    通过利用大规模预训练，大型语言模型（LLMs）能够解释多样化的时间表达和问题表述，而无需特定任务的训练。然而，现有的基于提示的神经符号系统仍依赖LLMs进行语义解释和精确时间推理。因此，关于时间区间、时间锚点和有序状态的离散决策仍容易受到概率性错误的影响，且难以验证。我们提出了STAIR，一个用于可解释推理的语义-时间自动机。STAIR将语义解释与精确时间推理分离：一个无答案的LLM适配器将复杂的问题表述映射为规范化时间意图，而一个具有有限控制和守卫转换的确定性时间自动机在规范化证据上执行相应策略。遵循规则优先的设计，STAIR解决了标准问题。

    arXiv:2608.16224v1 Announce Type: cross  Abstract: By leveraging large-scale pretraining, LLMs can interpret diverse temporal expressions and question formulations without task-specific training. However, existing prompt-based neuro-symbolic systems continue to rely on LLMs for both semantic interpretation and exact temporal inference. Consequently, discrete decisions regarding intervals, time anchors, and ordered states remain vulnerable to probabilistic errors and difficult to verify. We present STAIR, a \textbf{S}emantic-\textbf{T}emporal \textbf{A}utomaton for \textbf{I}nterpretable \textbf{R}easoning. STAIR separates semantic interpretation from precise temporal inference: an answer-free LLM adapter maps complex question formulations to normalized temporal intents, while a deterministic temporal automaton with finite control and guarded transitions executes the corresponding policies over canonicalized evidence. Following a rule-first design, STAIR resolves standard questions with
    
[^39]: INSPIRE：面向指令感知语音检索的基准测试

    INSPIRE: A Benchmark for Instruction-Aware Speech Retrieval

    [https://arxiv.org/abs/2608.16203](https://arxiv.org/abs/2608.16203)

    本文提出了首个指令感知语音检索基准INSPIRE，并评估了四种检索范式，发现现有方法无法统一处理所有检索意图，强调了开发统一架构的必要性。

    

    arXiv:2608.16203v1 公告类型：交叉 摘要：现有的语音检索系统依赖于固定的相似度匹配，无法适应多样化的用户意图。我们引入了INSPIRE，这是首个面向指令感知语音检索的基准测试，其中自然语言指令动态地指定相关性标准，包括语义内容、说话人身份、说话风格、环境声音及其组合。我们评估了四种检索范式：大型音频语言模型、级联流水线、自监督语音模型和对比音频语言模型。我们的结果显示，目前没有任何方法能够稳健地处理所有检索意图。基于文本的方法在语义检索方面表现相对较好，但在副语言属性上存在困难，而基于语音的模型在捕捉声学特性方面稍好，但在遵循指令方面表现不佳。这些发现凸显了对能够实现指令感知语音检索的统一架构的需求。

    arXiv:2608.16203v1 Announce Type: cross  Abstract: Existing speech retrieval systems rely on fixed similarity matching and cannot adapt to diverse user intents. We introduce INSPIRE, the first benchmark for instruction-aware speech retrieval, in which natural-language instructions dynamically specify relevance criteria, including semantic content, speaker identity, speaking style, environmental sounds, and their combinations. We evaluate four retrieval paradigms: large audio-language models, cascaded pipelines, self-supervised speech models, and contrastive audio-language models. Our results reveal that no current method robustly handles all retrieval intents. Text-based approaches perform relatively better at semantic retrieval but struggle with paralinguistic attributes, while speech-based models are moderately better at capturing acoustic properties but falter at following instructions. These findings highlight the need for unified architectures capable of instruction-aware speech r
    
[^40]: 透镜：通过动态原始文档上的潜在证据探索进行上下文搜索

    LENS: In-Context Search via Latent Evidence Exploration over Dynamic Raw Documents

    [https://arxiv.org/abs/2608.16185](https://arxiv.org/abs/2608.16185)

    本文提出LENS，一个无索引框架，通过动态原始文档上的潜在证据空间进行预算化证据定位，利用迭代提议和LLM相关性更新信念，避免了预生成索引的缺点。

    

    arXiv:2608.16185v1 公告类型：交叉 摘要：LLM代理越来越多地需要对动态原始文档集合进行问题回答，其中文件可能在预处理前发生变化，且相关证据（跨度、章节、页面或表格）依赖于查询。现有的检索增强方法通过固定分块、嵌入或持久索引预生成证据：适用于查找，但成本高、易过时，并且在查询已知前就确定了粒度。我们将上下文搜索形式化为对由动态原始文档诱导的潜在证据空间上的预算化证据定位问题，并提出了LENS（潜在证据探索与搜索），一个无索引框架。LENS不预先生成证据空间，而是维护对候选单元的查询条件信念，通过互补的词汇、局部和探索性提议策略迭代选择候选，通过LLM相关性预言机更新信念，并向高后验区域收窄。

    arXiv:2608.16185v1 Announce Type: cross  Abstract: LLM agents increasingly answer questions over dynamic raw-document collections, where files may change before preprocessing, and relevant evidence (spans, sections, pages, or tables) is query-dependent. Existing retrieval-augmented approaches pre-materialize evidence via fixed chunking, embeddings, or persistent indexes: effective for lookup, yet costly, stale-prone, and committed to a granularity before the query is known.   We formulate in-context search as Budgeted Evidence Localization over a latent evidence space induced by dynamic raw documents and propose LENS (Latent Evidence Exploration and Search), an index-free framework. Instead of pre-materializing the evidence space, LENS maintains a query-conditioned belief over candidate units, iteratively selecting candidates via complementary lexical, local, and exploratory proposal policies, updating the belief via an LLM relevance oracle, and narrowing toward high-posterior regions 
    
[^41]: QUMem：面向LLM智能体中查询条件用户状态推断的个性化记忆

    QUMem: Personalized Memory for Query-Conditioned User-State Inference in LLM Agents

    [https://arxiv.org/abs/2608.16168](https://arxiv.org/abs/2608.16168)

    QUMem提出了一种结构化记忆框架，通过独立存储用户信息并支持查询条件检索，解决了LLM智能体中偏好演变、时间有效性和情境适用性不足的问题。

    

    大型语言模型（LLM）智能体越来越多地使用外部记忆系统来支持个性化，通过利用长期且不断演变的交互历史，其中用户偏好可能随时间分布、随情境变化，并与早期证据冲突。然而，现有系统面临三个局限：固定轮次、固定令牌或基于会话的边界可能混合无关对话或将事件与其原因、决策和结果分离；将同一交互中的多条用户信息存储为单一记忆，会将功能不同且应独立检索的项目绑定在一起；将当前任务视为单个top-k检索查询，可能返回个别相关但无法共同捕捉偏好演变、时间有效性和情境适用性的片段。我们引入了\textsc{QUMem}，一种用于查询条件用户状态推断的结构化记忆框架。

    arXiv:2608.16168v1 Announce Type: cross  Abstract: Large language model (LLM) agents increasingly use external memory systems to support personalization by drawing on long and evolving interaction histories, in which user preferences may be distributed across time, change with context, and conflict with earlier evidence. However, existing systems face three limitations: fixed-turn, fixed-token, or session-based boundaries can mix unrelated dialogue or split an event from its causes, decisions, and outcomes; storing multiple pieces of user information from the same interaction as a single memory binds together items that serve different functions and should be independently retrievable; and treating the current task as a single top-$k$ retrieval query can return fragments that are individually relevant but fail to jointly capture preference evolution, temporal validity, and contextual applicability. We introduce \textsc{QUMem}, a structured memory framework for query-conditioned user-st
    
[^42]: HyperSkill：通过超图结构技能记忆实现自进化LLM智能体

    HyperSkill: Self-Evolving LLM Agents via Hypergraph-Structured Skill Memory

    [https://arxiv.org/abs/2608.16114](https://arxiv.org/abs/2608.16114)

    提出了一种基于超图结构的技能记忆框架HyperSkill，通过联合优化存储、检索和演化，使LLM智能体能够自动学习并复用可组合技能，显著提升复杂任务的执行效率。

    

    arXiv:2608.16114v1 公告类型：新 摘要：随着智能体任务复杂性的增长，LLM智能体越来越依赖经验记忆来跨任务复用程序性知识。有效的记忆设计必须同时解决存储什么、如何结构化与检索记忆，以及记忆如何演化的问题。现有系统仅部分处理了这些问题：它们将轨迹、洞察或工作流存储为孤立条目，忽略了子任务与可复用技能之间的组合关系；通过忽略关系信号的平面嵌入相似性进行检索；并在维护记忆时未利用其关系结构。我们提出了HyperSkill，一种基于超图的记忆框架，它联合改进了这三个方面。HyperSkill将记忆表示为具有两种节点类型（子任务步骤和可复用技能）的超图，其中每条超边连接来自单一轨迹的子任务和技能。双路径检索同时查询子任务和轨迹级别，通过共现对技能进行排名。

    arXiv:2608.16114v1 Announce Type: new  Abstract: As agentic tasks grow in complexity, LLM agents increasingly rely on experiential memory to reuse procedural knowledge across tasks. Effective memory design must jointly address what to store, how memory is structured and retrieved, and how memory evolves. Existing systems tackle each only partially: they store trajectories, insights, or workflows as isolated entries, discarding compositional relationships among subtasks and reusable skills; retrieve by flat embedding similarity that ignores relational signals; and maintain memory without leveraging its relational structure. We propose HyperSkill, a hypergraph-based memory framework that jointly improves all three. HyperSkill represents memory as a hypergraph with two node types, subtask steps and reusable skills, where each hyperedge links the subtasks and skills from a single trajectory. Dual-path retrieval queries both subtask and trajectory levels, ranking skills by co-occurrence acr
    
[^43]: 商业税：多跳检索基准测试中的租用与自建盲点

    The Commercial Tax: Rent-vs-Own Blind Spots in Multi-Hop Retrieval Benchmarks

    [https://arxiv.org/abs/2608.16096](https://arxiv.org/abs/2608.16096)

    论文揭示了多跳检索基准测试中忽略的许可和成本盲点，指出顶尖系统依赖非商业许可的嵌入器，并发现商业许可嵌入器存在性能税，但NVIDIA的Nemotron-3-Embed-8B已消除这一差距。

    

    摘要：arXiv:2608.16096v1 公告类型：交叉 摘要：企业通过检索将语言模型连接到自己的数据。对多跳检索系统进行排名的基准测试遗漏了两个买家在公布数字可用之前需要的事实：检索主干是否可以商业部署，以及构建成本是多少。关于许可：该领域的密集检索锚点NV-Embed-v2采用cc-by-nc-4.0许可。在我们审计的四个领先MuSiQue系统（HippoRAG-2、PropRAG、SAG、KET-RAG）中，三个依赖它获得最佳结果，但均未说明。关于性能：我们在一个相同的MuSiQue测试平台上测量了来自八个制造商的十三个嵌入器，并全程使用自举置信区间。直到2026年中期，存在真正的商业税：最佳商业许可嵌入器在Recall@5上落后锚点2.31个百分点（95%置信区间[0.91, 3.71]，p=0.001）。NVIDIA于2026-07-16发布的Nemotron-3-Embed-8B已弥补这一差距：在Recall@5上+0.24（95%置信区间[-0.94, +1.43]，p=0.69），-0.58。

    arXiv:2608.16096v1 Announce Type: cross  Abstract: Enterprises connect language models to their own data through retrieval. The benchmarks that rank multi-hop retrieval systems leave out two facts a buyer needs before a published number can be used: whether the retrieval backbone may be deployed commercially, and what it costs to build. On licensing: the field's dense-retrieval anchor, NV-Embed-v2, is licensed cc-by-nc-4.0. Of the four leading MuSiQue systems we audit (HippoRAG-2, PropRAG, SAG, KET-RAG), three depend on it for their best numbers and none says so. On performance: we measure thirteen embedders from eight makers on one identical MuSiQue harness with bootstrap confidence intervals throughout. Until mid-2026 there was a real commercial tax: the best commercially-licensed embedder trailed the anchor by 2.31 Recall@5 points (95% CI [0.91, 3.71], p=0.001). NVIDIA's Nemotron-3-Embed-8B, released 2026-07-16, has closed it: +0.24 at Recall@5 (95% CI [-0.94, +1.43], p=0.69), -0.58
    
[^44]: Skill2Query：利用技能结构生成伪查询以提升智能体技能检索

    Skill2Query: Exploiting Skill Structure to Generate Pseudo-Queries for Agent Skill Retrieval

    [https://arxiv.org/abs/2608.16071](https://arxiv.org/abs/2608.16071)

    Skill2Query通过解析技能文档为知识图谱并采用三阶段生成过程，显著提升了伪查询的质量，从而增强了智能体技能检索的有效性。

    

    摘要：伪查询生成可以缓解智能体技能检索中的监督瓶颈，但现有的文档级方法通常将能力、参数和使用示例之间的丰富内部关系隐式化。因此，生成的查询可能仅与技能主题相关，而缺乏能力依据和参数一致性，这引发了一个问题：显式利用技能文档的内部结构是否能产生更有效的检索信号。为此，我们提出了Skill2Query框架，该框架首先将技能文档解析为技能知识图谱，然后通过包括风格模仿、查询模板生成和参数填充的三阶段过程生成伪查询。生成的查询可用于离线索引增强、在线查询扩展和检索器训练。我们使用四个基准（TheoremQA、LogicBench、ToolQA和CHAMP）来评估Skill2Query。

    arXiv:2608.16071v1 Announce Type: new  Abstract: Pseudo-query generation can alleviate the supervision bottleneck for agent skill retrieval, but existing document-level approaches typically leave the rich internal relations among capabilities, parameters, and usage examples implicit. As a result, generated queries may be topically relevant to a skill while lacking capability grounding and parameter consistency, raising the question of whether explicitly exploiting a skill document's internal structure can produce more effective retrieval signals. We therefore propose Skill2Query, a framework that first parses a skill document into a Skill Knowledge Graph and then generates pseudo-queries through a three-stage process including style mimicking, query template generation, and parameter filling. The generated queries can be used for offline index augmentation, online query expansion, and retriever training. Four benchmarks (TheoremQA, LogicBench, ToolQA, and CHAMP) are used to evaluate Sk
    
[^45]: CAPO：面向LLM智能体的约束感知提示词优化

    CAPO: Constraint-Aware Prompt Optimization for LLM Agents

    [https://arxiv.org/abs/2608.16068](https://arxiv.org/abs/2608.16068)

    CAPO提出了一种原始-对偶优化方法，通过自适应约束加权和池化重写，在无需领域监督数据的情况下优化系统提示词，同时满足操作约束并提升智能体任务性能。

    

    大型语言模型（LLMs）越来越多地被部署为依赖系统提示词来使用工具和完成任务的智能体。这种部署方式带来了独特的操作要求，包括恰当的工具使用、简洁的提示词和解决方案路径，以及遵守安全和格式政策。然而，对于许多从业者来说，收集特定领域的监督数据来对模型进行后训练以满足这些要求是不可行的。我们引入了CAPO（约束感知提示词优化），这是一种原始-对偶方法，结合了基于池的重写和自适应约束加权，在明确的操作约束下优化系统提示词。在智能体基准测试中，CAPO更可靠地达到经验上可行的操作点，同时提升任务性能。CAPO还能推广到智能体设置之外，在具有输出格式和安全/隐私约束的助手式评估中取得了强劲结果。我们进一步...

    arXiv:2608.16068v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed as agents that rely on system prompts to use tools and complete tasks. Such deployments impose distinct operational requirements, including appropriate tool use, concise prompts and solution paths, and compliance with safety and formatting policies. For many practitioners, however, assembling domain-specific supervised data to post-train models to meet these requirements is infeasible. We introduce CAPO (Constraint-Aware Prompt Optimization), a primal-dual method that combines pool-based rewrites with adaptive constraint weighting to optimize system prompts under explicit operational constraints. Across agentic benchmarks, CAPO more reliably reaches empirically feasible operating points while improving task performance. CAPO also generalizes beyond agentic settings, achieving strong results on assistant-style evaluations with output-format and safety/privacy constraints. We further
    
[^46]: DuplexGen：解耦内容、时序与声学特征的合成对话语音生成

    DuplexGen: Decoupling Content, Timing, and Acoustics for Synthetic Dialogue Speech

    [https://arxiv.org/abs/2608.16053](https://arxiv.org/abs/2608.16053)

    DuplexGen通过解耦内容、时序和声学特征，利用LLM生成脚本和全双工模型实时交互，使对话时序自然涌现而非预设，实现了更真实、交互驱动的合成对话语音。

    

    摘要：arXiv:2608.16053v1 公告类型：新  摘要：合成对话语音已成为开发和评估对话语音系统的重要资源。然而，现有的对话合成流程通常首先生成对话内容，然后使用手工标记或时序规则插入打断、重叠和反馈语，这使得对话时序是预设的而非由交互驱动。我们提出了DuplexGen，一种明确解耦内容、时序和声学特征的对话合成框架。一个大型语言模型首先生成对话脚本，然后两个全双工对话模型在实时聆听对方的同时执行该脚本。这允许对话时序自然涌现，同时保留脚本化内容。最后，一个高保真文本转语音模型在不改变其时序的情况下重新渲染交互。作为所提出框架的演示，我们构建了一个患者-临床医生对话语音共同...（摘要截断）

    arXiv:2608.16053v1 Announce Type: new  Abstract: Synthetic conversational speech has become an important resource for developing and evaluating conversational speech systems. However, existing dialogue synthesis pipelines typically generate dialogue content first and then insert interruptions, overlap, and backchannels using handcrafted markers or timing rules, making conversational timing prescribed rather than interaction-driven. We present DuplexGen, a dialogue synthesis framework that explicitly decouples content, timing, and acoustics. An LLM first generates the dialogue script, and then two full-duplex conversational models perform the script while listening to each other in real time. This allows conversational timing to emerge naturally while preserving the scripted content. Finally, a high-fidelity text-to-speech model re-renders the interaction without altering its timing. As a demonstration of the proposed framework, we construct a patient--clinician conversational speech co
    
[^47]: 覆盖不等于包含：向量检索协同投毒在准入时防御的根本局限

    Coverage Is Not Containment: A Fundamental Limit of Admission-Time Defenses Against Coordinated Poisoning of Vector Retrieval

    [https://arxiv.org/abs/2608.16044](https://arxiv.org/abs/2608.16044)

    本文证明所有摄入时防御无法抵御协同投毒攻击，因为攻击锥在几何上与合法小众无异，导致RAG系统在88%目标中输出攻击者植入内容。

    

    arXiv:2608.16044v1 公告类型：交叉 摘要：检索增强生成（RAG）通过从向量存储中检索段落并将其作为上下文来回答问题，因此任何能添加文档的人都可以尝试引导答案。一种近期提出的有吸引力的防御方法在摄入时过滤投毒，拒绝任何行为类似枢纽的文档。我们证明它——以及所有摄入时过滤器——会被一种协同攻击者击败，该攻击者注入少量个体上不显眼的文档，这些文档共同包围一个目标查询并占据其top-k（在BGE-large / BEIR上，m=10个文档占10/10；在实时HNSW索引上为9.9/10）。该攻击并非理论性的。通过普通流畅文本实现，并端到端运行BGE-large + HNSW + Qwen2.5-7B流水线，它使生成器在88%的目标中输出攻击者植入的主张，而在无注入时为0%。且没有准入时防御能阻止它：在摄入时，攻击锥在几何上与合法小众无异。

    arXiv:2608.16044v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) answers a question by retrieving passages from a vector store and trusting them as context, so anyone who can add documents can try to steer the answer. A recent, appealing defense filters poisoning at ingestion, rejecting any document that behaves like a hub. We show it -- and every ingestion-time filter -- is defeated by a coordinated adversary that injects a handful of individually unremarkable documents which together surround one target query and seize its top-k (on BGE-large / BEIR, m=10 documents take 10/10; 9.9/10 on a live HNSW index). The attack is not theoretical. Realized as ordinary fluent text and run end-to-end through a BGE-large + HNSW + Qwen2.5-7B pipeline, it makes the generator emit the attacker's planted claim in 88% of targets, versus 0% without the injection. And no admission-time defense stops it: at ingestion an attack cone is geometrically identical to a legitimate niche up
    
[^48]: $R^3$-Bench：大型语言模型在共享预算下的资源理性推理中表现挣扎

    $R^3$-Bench: LLMs Struggle with Resource-Rational Reasoning under Shared Budgets

    [https://arxiv.org/abs/2608.16033](https://arxiv.org/abs/2608.16033)

    本文提出了$R^3$-Bench基准，揭示大型语言模型在共享预算下的资源理性推理能力不足，且通过经验预言机显示模型表现远低于其单问题能力上限。

    

    arXiv:2608.16033v1 公告类型：新 摘要：在认知科学中，资源理性探讨的是智能体应如何分配有限的计算资源以最大化预期价值。大多数推理和智能体基准测试使用独立的每任务预算；现有的共享预算研究并未针对同一模型在单问题上的已展示能力来校准套件性能。我们引入了$R^3$-Bench，该基准在数学、竞争性编程和抽象推理中，于无工具和智能体设置下评估共享预算下的六问题套件。匹配的单问题响应曲线定义了一个基于观测成功率的离线经验预言机。在六个模型的72个主表格单元中，预言机均值在所有单元中均匹配或超过竞赛均值，并在71个单元中严格更高。在适度的无工具压力下，均等分配重放对于六个模型中的四个也超过了竞赛表现。轨迹诊断揭示了有限的策略更新和依赖压力的失败模式。

    arXiv:2608.16033v1 Announce Type: new  Abstract: In cognitive science, resource rationality asks how an agent should allocate limited computation to maximize expected value. Most reasoning and agent benchmarks use independent per-task budgets; existing shared-budget studies do not calibrate suite performance against the same model's demonstrated single-problem competence. We introduce $R^3$-Bench, which evaluates six-problem suites under shared budgets across mathematics, competitive programming, and abstract reasoning in tool-free and agentic settings. Matched single-problem response curves define an offline empirical oracle over observed successes. Across 72 main-table cells for six models, the oracle mean matches or exceeds the contest mean in all cells and is strictly higher in 71. Under moderate tool-free pressure, equal-allocation replay also exceeds contest performance for four of six models. Trajectory diagnostics reveal limited strategy updating and pressure-dependent failure 
    
[^49]: ReRef-3D：空间指代表达引导的三维场景重排基准

    ReRef-3D: A Benchmark for Spatial Referring Expression-Guided 3D Scene Rearrangement

    [https://arxiv.org/abs/2608.16011](https://arxiv.org/abs/2608.16011)

    本文提出了ReRef-3D基准，用于评估语言引导的三维场景重排，发现关系满足性优于物理有效性，且“最近”和“之间”等关系最具挑战性。

    

    arXiv:2608.16011v1 公告类型：新 摘要：我们引入了ReRef-3D，一个用于三维场景中语言引导放置的基准。它包含998个基于CLEVR衍生场景的33,826条指令，涵盖16种放置类别以及直接、一跳和二跳引用。每条指令必须被解析为一个有效的放置位置。鉴于指令定义的是可接受放置的区域而非单一坐标，我们的评估将预测插入场景中，重新计算关系，并测试关系满足性和物理有效性。每条指令还包括一个经过验证的自然化改写。在微调后，LLaVA-3D、3D-LLM和PlaceIt3D分别对68.3%、31.6%和22.4%的指令产生有效放置。跨模型来看，关系满足性优于物理有效性，像“最近”和“之间”这样的关系最难处理，而措辞对性能影响最小。

    arXiv:2608.16011v1 Announce Type: new  Abstract: We introduce ReRef-3D, a benchmark for language-guided placement in 3D scenes. It contains 33,826 instructions across 998 CLEVR-derived scenes, spanning 16 placement families and direct, one-hop, and two-hop references. Each instruction must be resolved into a valid new placement position. Given that an instruction defines a region of acceptable placements rather than one coordinate, our evaluation inserts a prediction into the scene, recomputes relations, and tests relation satisfaction and physical validity. Each instruction also includes a verified naturalized rewrite. After fine-tuning, LLaVA-3D, 3D-LLM, and PlaceIt3D produce valid placements for 68.3%, 31.6%, and 22.4% of instructions, respectively. Across models, relation satisfaction surpasses physical validity, relations such as nearest and between are the most difficult, and phrasing has minimal effect on performance.
    
[^50]: 先前的审计-修复情境使LLM验证器阈值偏向宽松

    Prior Audit-Repair Context Shifts LLM Verifier Thresholds Toward Leniency

    [https://arxiv.org/abs/2608.16003](https://arxiv.org/abs/2608.16003)

    先前完成的审计-修复情境显著降低LLM验证器的误报率，使其阈值偏向宽松，且修复内容与审计结论共同作用，挑战了现有累积消息理论。

    

    arXiv:2608.16003v1 公告类型：新  摘要：自动化检查流水线越来越多地让一个语言模型作为检查器，另一个（或同一个）作为修复器。我们探究这种连接方式是否会改变检查器报告的结果。在保持当前任务字节完全一致的情况下，对人工验证正确的ProcessBench轨迹进行误报测量，我们发现，在模型上下文中已完成的审计->修复事件，在15种模型×措辞组合中的全部15种情况下降低了误报率，降幅为2.8至11.5个百分点，相对于长度匹配的非审计对照组，相对减少9%至25%。这一方向与累积消息文献的预测相矛盾：一个审计报告错误的场景进一步降低了误报率，在该操作干净落地的模型上的所有五种措辞中均如此，尽管负面性不对称预测会有更多标记。分解该事件发现，修复内容和审计结论具有互补性：不同组件承载着不同的影响。

    arXiv:2608.16003v1 Announce Type: new  Abstract: Automated checking pipelines increasingly place one language model as the checker and another (or the same one) as the fixer. We ask whether that wiring changes what the checker reports. Measuring false alarms on human-verified-correct ProcessBench traces with the present task held byte-identical, we find that a completed audit -> repair episode already in the model's context lowers false alarms in 15 of 15 model x wording combinations, by 2.8 to 11.5 percentage points against a length-matched non-audit control, a 9 to 25% reduction relative to that control. The direction contradicts what the accumulated-message literature predicts: an episode whose audit reported an error lowers false alarms further still, at all five wordings on the model where that manipulation lands cleanly, though a negativity asymmetry predicts more flagging. Decomposing the episode finds repair content and audit verdict complementary: different components carry th
    
[^51]: 从序列到结构：面向LLM代理的关系不确定性传播

    From Sequence to Structure: Relational Uncertainty Propagation for LLM Agents

    [https://arxiv.org/abs/2608.16002](https://arxiv.org/abs/2608.16002)

    本文提出RUPA框架，通过将LLM代理执行历史建模为有向轨迹图并传播不确定性，解决了现有UQ方法忽略远程依赖导致无法识别早期错误根源的问题。

    

    arXiv:2608.16002v1 公告类型：交叉 摘要：可靠的不确定性量化（UQ）对于在复杂交互环境中部署大型语言模型（LLM）代理至关重要。现有的UQ方法主要依赖局部信号，如标记概率、预测熵或逐步置信度，因此忽视了执行轨迹中错误累积的远程依赖关系。结果，它们可能无法识别代理失败，这些失败的原因源于最终答案之前的多个推理或交互步骤。我们提出了RUPA（代理关系不确定性传播），一种面向LLM代理的轨迹级UQ框架。RUPA将执行历史表示为有向轨迹图，其中推理状态、工具交互和环境反馈作为节点，通过时间和语义依赖边连接。然后，它在该图上传播不确定性，以捕捉执行风险如何在交互过程中累积和转移。

    arXiv:2608.16002v1 Announce Type: cross  Abstract: Reliable uncertainty quantification (UQ) is essential for deploying large language model (LLM) agents in complex interactive environments. Existing UQ methods largely rely on local signals, such as token probabilities, predictive entropy, or per-step confidence, and therefore overlook the long-range dependencies through which errors accumulate across an execution trajectory. As a result, they may fail to identify agent failures whose causes originate several reasoning or interaction steps before the final answer. We propose RUPA (Relational Uncertainty Propagation for Agents), a trajectory-level UQ framework for LLM agents. RUPA represents an execution history as a directed trajectory graph in which reasoning states, tool interactions, and environment feedback are nodes connected by temporal and semantic dependency edges. It then propagates uncertainty over this graph to capture how execution risk accumulates and transfers across inter
    
[^52]: 谁的黄金？标注者群体分歧在项目层面巨大，且被小排行榜掩盖

    Whose Gold? Annotator-Pool Disagreement Is Large at the Item Level, and Hidden by Small Leaderboards

    [https://arxiv.org/abs/2608.15980](https://arxiv.org/abs/2608.15980)

    本文发现不同标注者群体在偏好基准的项目层面存在显著分歧，但最终模型排行榜却看似不变，这种不变性实际上非常微弱，并可能掩盖模型间的真实差异。

    

    偏好基准是通过雇佣标注者构建的，而这些标注者的身份被视为实现细节。我们衡量了这一细节所带来的影响。在2,885个MultiPref项目中，两个标注者群体内部均达成一致，因此无需咨询任何打破平局的规则，专家和众包标注者对23.6%的项目分配了不同的多数标签，并在9.2%的项目中指出了相反的胜者；在246个同样一致的MT-Bench单元中，基准作者和招募的专家在30.5%的项目上存在分歧，并在8.5%的项目上反转结果。然而，在这两个语料库上，由此产生的模型排行榜逐位相同：Kendall tau = 1.00，六个模型无一被替换。这种不变性看起来是强有力的证据，但实际上远非如此，我们量化了其微弱程度。更换标注者群体会使模型的胜率变化1.9个百分点（标准差），我们自己的排行榜中相邻一对模型相差0.8个百分点，且有38%的几率发生互换，而基于项目级别的自助抽样在28%的重采样中至少会替换一个模型。

    arXiv:2608.15980v1 Announce Type: new  Abstract: Preference benchmarks are built by hiring annotators, and the identity of those annotators is treated as an implementation detail. We measure what that detail buys. On the 2,885 MultiPref items where both pools are internally unanimous, so no tie-breaking convention is consulted at all, expert and crowd annotators assign a different majority label to 23.6% and name the opposite winner on 9.2%; on the 246 comparably unanimous MT-Bench cells, benchmark authors and recruited experts differ on 30.5% and reverse on 8.5%. Yet on both corpora the resulting model leaderboards are bit-identical: Kendall tau = 1.00 with zero of six models displaced.   That invariance is far weaker evidence than it looks, and we quantify how weak. Switching pools moves a model's win rate by 1.9pp (SD), one adjacent pair in our own leaderboard sits 0.8pp apart and had a 38% chance of swapping, and an item-level bootstrap displaces at least one model in 28% of resamp
    
[^53]: 一种可扩展的LLM教师蒸馏标注流水线：工作窃取任务调度与内存感知GPU并发

    A Scalable Pipeline for LLM-Teacher Distillation Labeling: Work-Stealing Job Scheduling and Memory-Aware GPU Concurrency

    [https://arxiv.org/abs/2608.15975](https://arxiv.org/abs/2608.15975)

    本文提出了一种结合工作窃取环形池和内存感知并发规则的流水线，用于高效、可扩展的LLM教师蒸馏标注，实现无依赖且单机可复现。

    

    使用LLM教师模型对大型文本语料库进行标注已成为大规模训练数据的实用途径。在数百万条数据规模下，手动标注每一批数据不可行，两个关键问题主导着这一过程：教师模型每花费一美元能带来多少标签质量，以及如何在偏斜、易失败的负载下保持GPU工作节点忙碌。我们提出了一个简单、可复现的流水线来解决这两个问题。首先，一个工作窃取环形池：每个工作节点拥有一个队列，先处理自己的队列，然后从环形后继节点窃取任务，通过原子条件写入实现任务的一次性认领，并通过过期认领清理实现崩溃容错。该认领协议仅需存储层的比较并交换原语；我们在单个SQLite文件上实现它，这使得参考实现无依赖且实验可在单机上复现。其次，一个内存感知的并发规则，根据每个节点能容纳的模型副本数量来确定并行度大小。

    arXiv:2608.15975v1 Announce Type: cross  Abstract: Labeling large text corpora with LLM teachers has become a practical route to training data at scale. At millions of items, hand-labeling every batch is not feasible, and two questions dominate: what label quality a teacher buys per dollar, and how to keep a fleet of GPU workers busy under skewed, failure-prone workloads. We present a simple, reproducible pipeline that addresses both. First, a work-stealing ring pool: each worker owns a queue, drains it first, and then steals from ring successors, with exactly-once task claims via atomic conditional writes and crash tolerance via stale-claim sweeping. The claim protocol requires only a compare-and-set primitive from its storage layer; we implement it on a single SQLite file, which makes the reference implementation dependency-free and the experiments reproducible on one machine. Second, a memory-aware concurrency rule that sizes per-node parallelism by how many model copies fit on the 
    
[^54]: 双编码器中绑定能力的极限

    The Limits of Binding in Dual Encoders

    [https://arxiv.org/abs/2608.15971](https://arxiv.org/abs/2608.15971)

    本文通过理想编码器框架下的数学证明，系统揭示了双编码器模型在角色绑定任务上的固有局限性，指出其可分辨深度随维度仅呈对数增长，在CLIP规模下甚至不及普通语言的嵌套深度。

    

    诸如CLIP这样的双编码器模型通过两个独立计算的单位向量的单一内积来对图像-文本对进行评分，但在绑定任务上表现不佳，在区分“一辆红色的车和一只蓝色的狗”与“一辆蓝色的车和一只红色的狗”时，其得分往往接近随机水平。我们给出了一个数学解释，说明这种失败在何种情况下是必然的，在何种情况下是偶然的。在Kang等人提出的理想编码器框架内，我们首先表明相关公理是可满足的，因此任何不可能性都必须通过一个附加的、可检验的假设才能成立。随后，我们证明了三个此类障碍。深度方面：对于递归角色绑定编码，交换边界遵循精确定律$m(D) = 2b^{-D}$，其中D为嵌套深度，有限维版本在明确标记的一个浓度估计之前成立；可分辨深度仅随维度对数增长，在CLIP规模下仅为个位数，即普通语言的嵌套深度。目标方面：……

    arXiv:2608.15971v1 Announce Type: cross  Abstract: Dual-encoder models such as CLIP score an image-caption pair by a single inner product of two independently computed unit vectors, and fail at binding, often scoring near chance when asked to distinguish "a red car and a blue dog" from "a blue car and a red dog". We give a mathematical account of when this failure is necessary and when it is contingent. Working within the ideal-encoder framework proposed by Kang et al., we first show the relevant axioms are satisfiable, so every impossibility must enter through an added, checkable hypothesis. We then prove three such obstructions. Depth: for recursive role-binding codes the swap margin obeys an exact law $m(D) = 2b^{-D}$ in the nesting depth D, with a finite-dimension version holding up to one explicitly flagged concentration estimate; the resolvable depth grows only logarithmically in the dimension and is single-digit at CLIP scale, the nesting depth of ordinary language. Objective: a
    
[^55]: 大语言模型通过定向合成的多语言数据变得更聪明

    LLMs Get Smarter from Targeted Synthetic Multilingual Data

    [https://arxiv.org/abs/2608.15964](https://arxiv.org/abs/2608.15964)

    本文提出HOTFIXR框架，通过生成针对性的合成多语言数据来优化训练，从而在不牺牲整体性能的情况下修复大语言模型的跨语言推理弱点。

    

    arXiv:2608.15964v1 公告类型：交叉 摘要：语言特定能力（LSC）是指语言模型根据提示语言的不同而表现更好或更差的现象。换句话说，当使用不同语言提示时，语言模型对相同的语义查询会输出不同（且可能不正确）的响应。先前的研究将此归因于跨语言语义表示的内部错位。目前，文献中主要有两种解决LSC的方法：（1）将所有查询路由到英语，提高性能，但将语言表达限制为英语；（2）或使用语言平衡数据进行训练，使模型在不同语言上的表现均衡，但会降低整体性能。在本工作中，我们采取数据中心的视角，引入了HOTFIXR：面向改进跨语言推理的硬度优化训练数据。它是一个数据生成框架，利用模型来探测和学习学生模型的多语言弱点。

    arXiv:2608.15964v1 Announce Type: cross  Abstract: Language-specific competency (LSC) is the phenomenon of a language model performing better or worse depending on the language of the prompt. In other words, a language model outputs different (and potentially incorrect) responses to the same semantic query when prompted in different languages. Prior work attributes this to an internal misalignment of semantic representation across languages. Currently, there are two main approaches to address LSC in the literature: (1) routing all queries through English, improving performance, but limiting language expressivity to English; or (2) training on language-balanced data, equalizing model performance across languages, but reducing overall performance. In this work, we take a data centric perspective and introduce HOTFIXR: Hardness Optimized Training data For Improving X-Lingual Reasoning. It is a data generation framework that uses models to probe and learn a student model's multilingual wea
    
[^56]: SEER：通过选择性视觉-文本压缩实现长上下文推理

    SEER: Long-Context Reasoning via Selective Visual-Text Compression

    [https://arxiv.org/abs/2608.15962](https://arxiv.org/abs/2608.15962)

    SEER提出了一种选择性视觉-文本压缩框架，通过视觉扫描和按需文本检索，兼顾视觉压缩的效率与文本推理的精度，显著提升长上下文推理中的提取精度。

    

    arXiv:2608.15962v1 公告类型：新 摘要：由于注意力机制在文本令牌上的二次复杂度，长上下文推理对大型语言模型而言计算成本高昂。视觉-文本压缩提供了一种有前景的替代方案，通过将文本渲染为图像并用视觉-语言模型处理，通常能减少令牌使用量。然而，现有方法无论查询相关性如何都应用统一压缩，可能在需要详细提取的场景中牺牲精度。我们提出了SEER，一个通过视觉扫描学习选择与查询相关的图像，并仅在需要处检索文本内容的框架，结合了视觉压缩的效率和基于文本推理的精度。通过在工具交互轨迹上进行监督微调，SEER学习自适应工具调用以进行选择和检索。在长上下文基准上的实验表明，SEER通过选择性文本检索提高了提取精度。

    arXiv:2608.15962v1 Announce Type: new  Abstract: Long-context reasoning remains computationally expensive for large language models due to the quadratic complexity of attention over text tokens. Visual-text compression offers a promising alternative by rendering text into images and processing them with vision-language models, often reducing token usage. However, existing approaches apply uniform compression regardless of query relevance, potentially sacrificing precision where detailed extraction is required. We present SEER, a framework that learns to select query-relevant images through visual scanning and retrieve textual content only where needed, combining the efficiency of visual compression with the precision of text-based reasoning. Through supervised fine-tuning on tool-interaction trajectories, SEER learns adaptive tool invocation for selection and retrieval. Experiments on long-context benchmarks show that SEER improves extraction precision through selective text retrieval 
    
[^57]: 确保可靠：为自信的多轮LLM推荐提供信息丰富的交互

    Ask to Be Sure: Informative Interactions for Confident Multi-Turn LLM Recommendation

    [https://arxiv.org/abs/2608.15949](https://arxiv.org/abs/2608.15949)

    本文提出了一种通过熵减少量化交互信息增益并作为奖励微调LLM的新方法，无需真实推荐即可生成策略性多轮对话，提升推荐置信度。

    

    arXiv:2608.15949v1 公告类型：交叉 摘要：大型语言模型（LLMs）的最新进展使其能够用作对话式推荐系统（CRS），展现出强大的推荐准确性和自然对话能力。然而，有效引导多轮交互以挖掘用户偏好仍然具有挑战性。现有方法要么使用带有模板化交互的独立强化学习代理，要么通过另一个LLM评判交互性来优化，而不测量实际获得的有用信息量。我们提出了一种新方法，通过助手不确定性的降低（通过推荐上的熵来衡量）来量化每次交互的有效性。我们将这种熵减少作为奖励——不依赖真实推荐（在现实场景中通常不可用）——来微调LLM，从而实现策略性交互生成。使用监督微调（SFT）和直接偏好优化（DPO）的实证结果展示了其有效性。

    arXiv:2608.15949v1 Announce Type: cross  Abstract: Recent advances in large language models (LLMs) have enabled their use as conversational recommender systems (CRS), demonstrating strong recommendation accuracy and natural dialogue. However, guiding multi-turn interactions to elicit user preferences effectively remains challenging. Existing approaches either use separate reinforcement learning agents with templated interactions or optimize for interactivity judged by another LLM, without measuring how much useful information is actually gained. We propose a new approach that quantifies the effectiveness of each interaction by the reduction in the assistant's uncertainty, measured via entropy over recommendations. We apply this entropy reduction as a reward---without relying on ground-truth recommendations, which are often unavailable in real-world scenarios---to fine-tune the LLM, enabling strategic interaction generation. Empirical results with supervised fine-tuning (SFT) and direct
    
[^58]: 空标记知道：减少ASR和NMT中无信息幻觉

    The Null Token Knows: Reducing Message-Free Hallucination in ASR and NMT

    [https://arxiv.org/abs/2608.15940](https://arxiv.org/abs/2608.15940)

    该论文发现，通过调整ASR和NMT模型中的空标记分数，可以有效减少无信息输入时的幻觉，但需平衡抑制与删除成本。

    

    arXiv:2608.15940v1 公告类型：新 摘要：现代编码器-解码器系统即使输入中不包含可恢复的信息，也能产生流畅的文本。我们通过模型保留的空标记来研究ASR和NMT中的这一失败，询问生成结束的分数是否已经携带了可用的弃权信号。在语音识别器和翻译模型中，我们审计了原生空标记分数和标量逻辑位移。在Whisper中，我们进一步探测解码器状态，并将监督行编辑与常规外部门控进行比较。评估的模型通常暴露了一个有用的弃权信号，但标准的解码过程并未可靠地利用它。提高空标记分数可以显著抑制捏造，但激进的干预也会删除有效的语音或缩短合法的翻译。这些发现将空标记转变为幻觉的诊断镜头，并激励通过抑制和删除成本来评估弃权方法，而不仅仅是基于幻觉率。

    arXiv:2608.15940v1 Announce Type: new  Abstract: Modern encoder-decoder systems can produce fluent text even when their input contains no recoverable message. We study this failure in ASR and NMT through the models' reserved null tokens, asking whether the score for ending generation already carries a usable abstention signal. Across speech recognizers and translation models, we audit native null-token scores and scalar logit shifts. In Whisper, we additionally probe decoder states and compare supervised row edits with conventional external gates. The evaluated models often expose a useful abstention signal, but stock decoding does not reliably act on it. Raising the null-token score can sharply suppress fabrication, but aggressive intervention also deletes valid speech or shortens legitimate translations. These findings turn the null token into a diagnostic lens on hallucination and motivate evaluating abstention methods by both suppression and deletion costs, rather than by hallucina
    
[^59]: 中止但未被遗忘：KV缓存保留破坏语言代理中的回滚一致性

    Aborted but Not Forgotten: KV-Cache Retention Breaks Rollback Consistency in Language Agents

    [https://arxiv.org/abs/2608.15939](https://arxiv.org/abs/2608.15939)

    本文发现语言代理在逻辑中止后保留KV缓存会破坏回滚一致性，导致模型仍能访问已丢弃内容，并通过新审计方法在多个模型中验证了该问题。

    

    arXiv:2608.15939v1 公告类型：新公告 摘要：有状态的语言代理假设可以通过从应用程序记录中清除被拒绝的分支来撤回该分支。我们表明，当服务会话在逻辑中止后保留键值（KV）状态时，这种假设会失效：模型可以继续关注应用程序认为已丢弃的内容。我们将缺失的保证形式化为回滚一致性：完全中止必须恢复模型所关注的状态，而不仅仅是记录。关键失败是跨层的：正确的逻辑回滚不一定与保留的推理状态组合，且这种差距可能对应用程序不可见。为了将缓存效应与文本效应分离，我们引入了一种同令牌/不同缓存的审计方法，保持决策步骤的令牌相同，仅变化缓存前缀是过时的还是从已提交状态重建的。在七个开放权重家族（3.8B-36B）中，仅保留KV就在63个审计单元中的25个翻转了类型化保护效应，而

    arXiv:2608.15939v1 Announce Type: new  Abstract: Stateful language agents assume a rejected branch can be taken back by clearing it from the application transcript. We show this breaks when the serving session retains key/value (KV) state across the logical abort: the model can continue attending to content the application believes it discarded. We formalize the missing guarantee as rollback consistency: a complete abort must restore the state the model attends, not just the transcript. The key failure is cross-layer: a correct logical rollback need not compose with retained inference state, and the gap can remain invisible to the application. To isolate cache effects from text effects, we introduce a same-token/different-cache audit that holds decision-step tokens identical while varying only whether the cached prefix is stale or rebuilt from committed state. Across seven open-weight families (3.8B-36B), retained KV alone flips a typed protected effect in 25 of 63 audited cells, while
    
[^60]: 令牌分布与数据量：多领域会议摘要中的领域平衡

    Token Distribution versus Data Volume: Domain Balancing in Multi-Domain Meeting Summarisation

    [https://arxiv.org/abs/2608.15935](https://arxiv.org/abs/2608.15935)

    本文通过解耦令牌分布和数据量，证明在会议摘要中领域平衡的令牌混合能有效提升数据稀缺领域的质量，且修剪低价值转录行可减少约15%的令牌。

    

    arXiv:2608.15935v1 公告类型：新 摘要：在规模差异显著的会议摘要语料库上联合微调大型语言模型，引发了一个先前工作未解决且混淆的问题：当领域平衡的训练混合有助于提升性能时，这种提升是源于跨领域的令牌分布，还是仅仅因为看到的数据量？我们通过构建在匹配令牌预算（2-32M）下的平衡和自然（原始比例）令牌混合，在五个英语会议语料库上，使用QLoRA微调Mistral-7B，并按领域进行评估，从而解耦这些因素。平衡分配重新分配了质量，以较低的成本改善了数据稀缺的少数领域，而对数据丰富的领域影响不大。只要少数领域重要，平衡就有利：在比例分配下，它们的份额固定为1-2%，无论预算大小，因此在那些领域达到平衡质量需要更多的总数据。我们进一步发现，修剪低价值的转录行可以从对话中移除约15%的令牌。

    arXiv:2608.15935v1 Announce Type: new  Abstract: Jointly fine-tuning an LLM on meeting-summarisation corpora of widely varying size raises a question that prior work leaves confounded: when a domain-balanced training mixture helps, is the gain due to the distribution of tokens across domains, or merely to the volume of data seen? We disentangle these factors by constructing balanced and natural (native-proportional) token mixtures at matched token budgets (2-32M) over five English meeting corpora, fine-tuning Mistral-7B with QLoRA, and evaluating per domain. Balancing redistributes quality, improving the data-scarce minority domains at a low cost to the data-rich ones. The trade favours balancing whenever the minority domains matter: their share under proportional allocation is fixed at 1-2% regardless of budget, so matching balanced quality on those domains requires far more total data. We further find that pruning low-value transcript lines removes ~15% of tokens from the conversatio
    
[^61]: PLSQLBench：用于可执行过程式数据库编程的LLM系统基准测试

    PLSQLBench: Benchmarking LLM Systems for Executable Procedural Database Programming

    [https://arxiv.org/abs/2608.15931](https://arxiv.org/abs/2608.15931)

    本文提出了首个用于评估LLM编写可执行PL/SQL程序的基准测试PLSQLBench，通过执行测试衡量正确性，并揭示了LLM在过程式数据库编程中的关键缺陷。

    

    我们提出了PLSQLBench，据我们所知，这是第一个用于评估LLM能否编写可执行PL/SQL程序的基准测试，其正确性通过基于执行的测试来衡量。现有的LLM评估主要针对通用代码生成或声明式文本到SQL，而过程式数据库编程仍未得到充分探索。PLSQLBench包含2,865个实例：2,594个单轮任务和271个多轮对话，共978轮。该基准结合了基于企业风格Spider 2数据库的复杂模式接地任务、源自Spider的简单模式接地任务，以及源自MBPP的过程式问题，覆盖了不同级别的数据库接地和过程式复杂度。对八个LLM的实验揭示了在模式接地、PL/SQL方言保真度、过程控制流、异常处理和跨轮一致性方面的反复出现的困难。工具增强的LLM代理在多个模式相关任务上提升了性能。

    arXiv:2608.15931v1 Announce Type: new  Abstract: We present PLSQLBench, to our knowledge the first benchmark for evaluating whether LLMs can write executable PL/SQL programs, with correctness measured through execution-based tests. Existing LLM evaluations largely target general-purpose code generation or declarative text-to-SQL, leaving procedural database programming underexplored. PLSQLBench contains 2,865 instances: 2,594 single-turn tasks and 271 multi-turn conversations spanning 978 turns. The benchmark combines complex schema-grounded tasks over enterprise-style Spider 2 databases, simpler schema-grounded tasks derived from Spider, and MBPP-derived procedural problems, covering varying levels of database grounding and procedural complexity. Experiments with eight LLMs reveal recurring difficulties in schema grounding, PL/SQL dialect fidelity, procedural control flow, exception handling, and cross-turn consistency. Tool-augmented LLM agents improve performance on several schema-g
    
[^62]: 迭代自学习用于富有表现力的文本到语音合成

    Iterative Self-Learning for Expressive Text-to-Speech Synthesis

    [https://arxiv.org/abs/2608.15910](https://arxiv.org/abs/2608.15910)

    本文提出了一种迭代自学习框架，通过无分类器的标签反转方法，在半监督条件下逐步提升表达性TTS系统的标签质量和语音合成性能，解决了显式表达标签稀缺的瓶颈。

    

    arXiv:2608.15910v1 公告类型：交叉 摘要：使用显式条件标签的富有表现力的文本到语音（TTS）系统，相比基于参考或提示的方法，能提供直接且可解释的表达属性控制，但需要标注数据。大规模获取这些标签成本高昂且耗时，然而先前没有半监督框架专门解决这一瓶颈。现有的半监督TTS方法反而针对配对语音-文本数据或转录本的稀缺性。为了解决表达标签的稀缺问题，我们提出了一种用于富有表现力TTS的迭代自学习（ISL）框架，该框架基于Invert-Classify，一种无分类器方法，通过反转冻结的生成模型来恢复离散表达标签。该框架使用当前模型迭代地对未标记语音进行伪标签化，在合并的标记和伪标记数据上重新训练，并重复此过程，逐步提高标签质量和合成效果。我们验证了该方法的有效性。

    arXiv:2608.15910v1 Announce Type: cross  Abstract: Expressive text-to-speech (TTS) systems that use explicit conditioning labels provide direct and interpretable control over expressive attributes, in contrast to reference-based or prompting-based approaches, but require labeled data. Obtaining these labels at scale is costly and time-consuming, yet no prior semi-supervised framework addresses this specific bottleneck. Existing semi-supervised TTS methods instead target scarcity of paired speech-text data or transcriptions. To address the scarcity of expressive labels, we propose an Iterative Self-Learning (ISL) framework for expressive TTS, built on Invert-Classify, a classifier-free method that recovers discrete expressive labels by inverting a frozen generative model. The framework iteratively pseudo-labels unlabeled speech using the current model, retrains on the combined labeled and pseudo-labeled data, and repeats, progressively refining label quality and synthesis. We validate o
    
[^63]: 大型语言模型辅助从科学文献中发现队列研究

    Large language model-assisted discovery of cohorts from scientific literature

    [https://arxiv.org/abs/2608.15909](https://arxiv.org/abs/2608.15909)

    该论文提出了一种基于大型语言模型的框架，通过自动生成PubMed查询并提取文献中的队列名称，实现从科学文献中高效发现研究队列，减少了手动文献搜索的工作量。

    

    arXiv:2608.15909v1 公告类型：交叉 摘要：背景：规划多研究分析需要识别具有相关参与者、表型和数据模态的队列。这一过程通常依赖于先验知识、队列目录和手动文献检索。我们开发了一个互补的、基于问题的框架，用于搜索相关科学文献并提取明确的队列名称。方法：该框架首先从可配置的词汇和模板生成多个PubMed查询，并通过PubMed API自动检索相关科学文献。然后，大型语言模型筛选检索到的标题和摘要，并使用针对研究问题定制的提示提取明确的队列名称。提取的名称通过人工审查去重。可配置的代码、提示和示例输出可在https://gitlab.rz.uni-frankfurt.de/cap_molgenlab/literature-cohort-discovery获取。评估：作为用例，

    arXiv:2608.15909v1 Announce Type: cross  Abstract: Background: Planning multi-study analyses requires identifying cohorts with the relevant participants, phenotypes, and data modalities. This process commonly relies on prior knowledge, cohort catalogues, and manual literature searches. We developed a complementary question-driven framework that searches relevant scientific literature and extracts explicit cohort names. Methods: The framework first generates multiple PubMed queries from configurable vocabularies and templates and retrieves the resulting scientific literature automatically through the PubMed API. A large language model then screens the retrieved titles and abstracts and extracts explicit cohort names using a prompt tailored to the research question. The extracted names are deduplicated with human review. Configurable code, prompts, and example outputs are available at https://gitlab.rz.uni-frankfurt.de/cap_molgenlab/literature-cohort-discovery. Evaluation: As a use case,
    
[^64]: 少即是多：孟加拉语新闻标题生成的上下文选择与提示策略

    When Less Is Enough: Context Selection and Prompting Strategies for Bengali News Headline Generation

    [https://arxiv.org/abs/2608.15879](https://arxiv.org/abs/2608.15879)

    本研究发现，在孟加拉语新闻标题生成中，选择文章的引导段落而非全文作为上下文，结合适当的提示策略，能在减少输入量的同时维持甚至提升生成质量。

    

    大型语言模型（LLMs）在文本生成任务中表现出色，但其在标题生成上的有效性仍高度依赖于输入上下文的选择和呈现方式。在本研究中，我们将孟加拉语新闻标题生成视为一项文档级生成任务，该任务要求从长篇新闻文章中有效选择和呈现关键上下文信息。我们使用Gemini-2.0-Flash、Llama-3.3-70B和GPT-4o，系统性地研究了上下文选择、提示策略和上下文学习（即少样本学习）对标题生成质量的影响。实验表明，提供全文并不一定提升性能；相反，仅使用文章选定的引导段落可以维持甚至在某些情况下提高标题生成质量。我们进一步比较了孟加拉语原生提示（BNaP）和跨语言提示（XLP），并考察了每种策略如何相互交互。

    arXiv:2608.15879v1 Announce Type: new  Abstract: Large language models (LLMs) have shown strong performance in text generation tasks, yet their effectiveness on headline generation remains sensitive to how input context is selected and presented. In this work, we investigate Bengali news headline generation as a document-level generation task that requires effective selection and presentation of salient contextual information from long-form articles. Using Gemini-2.0-Flash, Llama-3.3-70B, and GPT-4o, we systematically study the effects of context selection, prompting strategies, and in-context learning (i.e., few-shot) on the quality of headline generation. Our experiments show that providing the full article does not necessarily improve performance; instead, using selected lead paragraphs of the article can maintain, and in some cases improve, headline generation quality. We further compare Bengali Native Prompting (BNaP) and Cross-Lingual Prompting (XLP), and examine how each interac
    
[^65]: 大型语言模型作为隐式社会学模型：从社会人口统计画像重建投票行为

    Large Language Models as Implicit Sociological Models: Reconstructing Voting Behaviour from Sociodemographic Profiles

    [https://arxiv.org/abs/2608.15871](https://arxiv.org/abs/2608.15871)

    本文提出了一种方法论框架，利用大型语言模型的潜在表征从社会人口统计画像重建选举投票行为，并在捷克选举中验证了其有效性，贡献在于方法论而非预测精度。

    

    在大规模互联网语料上训练的大型语言模型（LLMs）编码了关于社会身份、态度和政治行为的广泛统计规律。本文引入并评估了一种方法论框架，利用这些潜在表征从个体层面的社会人口统计画像重建总体投票行为。我们通过将LLMs作为隐式社会学模型进行操作化，以人口统计描述为条件，引出概率性的投票率和政党偏好，并通过软投票程序聚合个体输出。以2021年捷克议会选举作为验证案例，我们证明当代LLMs以低平均绝对误差再现官方选举结果，恢复已知的政治阵营结构，并与独立确立的社会人口统计梯度保持一致。本工作的贡献在于方法论而非预测性：我们提出了一种利用LLMs作为社会推断工具的新方法，而非改进预测精度。

    arXiv:2608.15871v1 Announce Type: cross  Abstract: Large language models (LLMs) trained on large-scale internet corpora encode extensive statistical regularities about social identities, attitudes, and political behaviour. This paper introduces and evaluates a methodological framework that leverages these latent representations to reconstruct aggregate voting behaviour from individual-level sociodemographic profiles. We operationalize LLMs as implicit sociological models by conditioning them on demographic descriptions, eliciting probabilistic turnout and party preferences, and aggregating individual outputs via a soft voting procedure. Using the 2021 Czech parliamentary election as a validation case, we demonstrate that contemporary LLMs reproduce official election outcomes with low mean absolute error, recover known political bloc structures, and align with independently established sociodemographic gradients. The contribution of this work is methodological rather than predictive: we
    
[^66]: 超越视觉思维链：内化视觉思维用于主动视频推理

    Beyond Visual CoT: Internalized Visual Thinking for Proactive Video Reasoning

    [https://arxiv.org/abs/2608.15869](https://arxiv.org/abs/2608.15869)

    本文提出了一种名为内化视觉思维（IVT）的后训练框架，通过联合优化文本预测和下一嵌入预测，使多模态大语言模型在训练中内化视觉思考，从而在推理时直接生成答案，避免了视觉思维链的中间图像生成开销，显著提升了主动视频推理的效率。

    

    arXiv:2608.15869v1 公告类型：交叉 摘要：多模态大语言模型越来越多地使用视觉思维链（Visual CoT）来推理空间、时间和具身环境。通过生成中间推理图像，视觉思维链为视觉前瞻提供了一种直观机制，但引入了大量推理开销，这对于主动视频推理尤其成问题。我们探究模型是否能在训练期间学习视觉思考，而在推理时直接进行推理。我们引入了内化视觉思维（IVT），这是一种后训练框架，它在未标记视频上联合优化文本预测和下一嵌入预测。给定一个部分观察的视频，IVT预测未来帧的潜在表示以及目标文本答案，鼓励模型捕捉运动、物体转换、交互和潜在意图。在推理时，IVT直接生成答案，而无需合成或重新编码中间图像。

    arXiv:2608.15869v1 Announce Type: cross  Abstract: Multimodal large language models increasingly use visual chain-of-thought (Visual CoT) to reason about spatial, temporal, and embodied environments. By generating intermediate reasoning images, Visual CoT provides an intuitive mechanism for visual foresight but introduces substantial inference overhead, which is particularly problematic for proactive video reasoning. We ask whether models can learn to think visually during training while reasoning directly at inference. We introduce Internalized Visual Thinking (IVT), a post-training framework that jointly optimizes textual prediction and next-embedding prediction over unlabeled videos. Given a partially observed video, IVT predicts latent representations of future frames together with the target textual answer, encouraging the model to capture motion, object transitions, interactions, and latent intent. At inference, IVT generates the answer directly without synthesizing or re-encodin
    
[^67]: 基于数据合成与统一规划的手册引导家电操作

    Scaling Manual-Grounded Appliance Manipulation with Data Synthesis and Unified Planning

    [https://arxiv.org/abs/2608.15863](https://arxiv.org/abs/2608.15863)

    本文提出MAGE数据合成流水线和AppliancePlan模型，构建了首个大规模家电操作规划数据集UseAppliance，仅用7B参数就在真实基准上实现了超过基线10倍的性能提升。

    

    摘要：操作家用电器需要依赖于状态且对干扰具有鲁棒性的长程规划，然而现有的大型模型在这方面表现不足，因为目前缺乏足够多样化且面向任务的数据集来支持此类规划。为弥补这一差距，我们提出了MAGE，一种可扩展的数据合成流水线，它引入了一种新颖的分层家电图（HAG），能够从家电手册中自动生成部件接地、长程规划和闭环恢复数据。借助MAGE，我们构建了UseAppliance，这是首个用于手册引导家电操作规划的大规模数据集，涵盖22个家电类别，包含超过89K个部件标注、53K多个操作任务以及33K多个闭环调整步骤。基于UseAppliance，我们开发了AppliancePlan，一种用于手册引导家电操作规划的端到端模型。在RealAppliance-Bench基准上，仅含7B参数的AppliancePlan在最佳基线上实现了超过10倍的性能提升。

    arXiv:2608.15863v1 Announce Type: cross  Abstract: Operating household appliances requires long-horizon planning that is state-dependent and robust to disturbances, yet existing large models fall short, as no sufficiently diverse, task-oriented dataset exists to support such planning. To bridge this gap, we propose MAGE, a scalable data synthesis pipeline that introduces a novel Hierarchical Appliance Graph (HAG) to automatically generate part grounding, long-horizon planning, and closed-loop recovery data from appliance manuals. With MAGE, we build UseAppliance, the first large-scale dataset for manual-grounded appliance manipulation planning, spanning 22 appliance categories with 89K+ part annotations, 53K+ manipulation tasks, and 33K+ closed-loop adjustment steps. Built on UseAppliance, we develop AppliancePlan, an end-to-end model for manual-grounded appliance manipulation planning. On RealAppliance-Bench, AppliancePlan with only 7B parameters achieves over 10x the best baseline on
    
[^68]: 稠密扩展，稀疏锚定：面向混合检索的通道非对称查询扩展

    Dense Expands, Sparse Anchors: Channel-Asymmetric Query Expansion for Hybrid Retrieval

    [https://arxiv.org/abs/2608.15851](https://arxiv.org/abs/2608.15851)

    本文提出DESA方法，通过通道非对称的查询扩展（稠密端正交残差扩展、稀疏端分数乘积锚定），解决了混合检索中固定截断值导致评估结果不稳定的问题。

    

    基于大语言模型的查询扩展通过生成类似文档的段落来提升检索效果。然而，在混合检索中，大多数评估方法融合固定的top-L稠密和稀疏排序。由于截断值同时控制跨通道贡献进入融合的方式以及每个排序被访问的程度，在某个L值下测得的增益在另一个L值下可能发生变化甚至反转。我们通过完整列表融合下的检索效果评估来分离这些影响，并记录策略特定的每通道重放停止深度，在该深度下其有序top-K得到验证。随后，我们提出了DESA（稠密扩展与稀疏锚定），一种通道非对称的查询扩展方法。大语言模型生成互补的参考段落；正交残差扩展将这些段落的新语义方向添加到稠密查询中，而分数乘积锚定则将其词汇线索纳入稀疏检索，同时不扩大原始查询的词汇支持范围。

    arXiv:2608.15851v1 Announce Type: cross  Abstract: LLM-based query expansion improves retrieval by generating document-like passages. In hybrid retrieval, however, most evaluations fuse fixed top-$L$ dense and sparse rankings. Because the cutoff controls both which cross-channel contributions enter fusion and how much of each ranking is accessed, gains measured at one $L$ can change or reverse at another. We separate these effects by evaluating retrieval effectiveness under complete-list fusion and recording the policy-specific per-channel replay stopping depths at which its ordered top-$K$ is certified. We then introduce DESA (Dense Expansion and Sparse Anchoring), a channel-asymmetric query expansion method. An LLM generates complementary reference passages; orthogonal residual expansion adds their new semantic directions to the dense query, while score-product anchoring incorporates their lexical cues into sparse retrieval without broadening the original query's lexical support. Acr
    
[^69]: MicroVerse：一种测量长时程多智能体语言模型模拟中自述身份漂移的仪器

    MicroVerse: An Instrument for Measuring Self-Authored Identity Drift in Long-Horizon Multi-Agent Language-Model Simulations

    [https://arxiv.org/abs/2608.15844](https://arxiv.org/abs/2608.15844)

    MicroVerse提出了一种行为科学仪器，通过不可变的“灵魂文件”和资源稀缺环境，测量长时程多智能体模拟中的身份漂移，并采用三层记忆架构和统一测量方法以减轻幸存者偏差。

    

    arXiv:2608.15844v1 公告类型：新 摘要：长时程、多智能体语言模型（LM）模拟被广泛用于研究社会行为，但缺乏测量在持续压力下，以人格为条件的智能体是否保持身份保真度的仪器。我们提出了MicroVerse，一种行为科学仪器，用于测量生成式智能体中的身份漂移。智能体携带不可变的“灵魂文件”（核心价值、道德边界、个性、目标），并居住在一个资源稀缺的50×50环境中，其中水是不可再生的生存约束。稀缺性通过每步存在成本梯度来操作化。八动词动作空间直接映射到道德边界（交易、交谈、攻击、拾荒）。利用三层记忆架构，智能体通过重要性触发的反思，定期将可变的当前身份与其不可变的原始灵魂进行比较。为减轻幸存者偏差，MicroVerse使用统一的方式将测量与行为解耦。

    arXiv:2608.15844v1 Announce Type: new  Abstract: Long-horizon, multi-agent language model (LM) simulations are widely proposed for studying social behavior, yet instruments to measure whether persona-conditioned agents maintain identity fidelity under sustained pressure are lacking. We present MicroVerse, a behavioral-science instrument that measures identity drift in generative agents. Agents carry an immutable "soul file" (core values, moral boundaries, personality, goals) and inhabit a resource-scarce 50 x 50 environment where water is a non-respawning survival constraint. Scarcity is operationalized via a per-tick existence-cost gradient. The eight-verb action space maps directly to moral boundaries (trade, talk, attack, scavenge). Using a three-layer memory architecture, agents periodically revise a mutable current identity against their immutable original soul via importance-triggered reflection. To mitigate survivor bias, MicroVerse decouples measurement from behavior using unif
    
[^70]: 模式无关的混合知识图谱图推理代理

    Schema-Agnostic Graph Reasoning Agent for Hybrid Knowledge Graphs

    [https://arxiv.org/abs/2608.15834](https://arxiv.org/abs/2608.15834)

    本文提出GRA，一种模式无关的图推理代理，通过通用工具在混合知识图谱上运行时发现领域知识，在工业基准上以更少的输入令牌显著提升性能。

    

    arXiv:2608.15834v1 公告类型：新 摘要：工具调用型LLM代理通过少量通用原语（如ls、cat、grep）来导航不熟悉的代码库，用于列出、读取和搜索文件。知识图谱提供了相同的接口：列出邻居、读取节点内容和搜索描述，这些操作在不同基质上是相同的。基于这种对应关系，我们提出了GRA，一种图推理代理，它通过七种通用工具探索混合知识图谱（其节点可以是文本概念或关系表），并在运行时发现所有特定领域的信息。在UFK-M（统一工厂知识模型）上，这是一个包含258个分析问题的工业基准，其黄金答案通过执行验证过的SQL程序生成，GRA比全上下文代理提高了5.1个百分点（88.4%对83.3%），同时读取的输入令牌不到其三分之一。无图对照实验表明，这种提升主要来自选择性代理访问，而非图拓扑结构。

    arXiv:2608.15834v1 Announce Type: new  Abstract: Tool-calling LLM agents navigate unfamiliar codebases with a handful of generic primitives for listing, reading and searching files (ls, cat, grep). A knowledge graph admits the same interface: listing neighbours, reading node content and searching descriptions are the same operations on a different substrate. Building on this correspondence, we present GRA, a Graph Reasoning Agent that explores hybrid knowledge graphs, whose nodes are either textual concepts or relational tables, with seven generic tools, discovering everything domain-specific at run time. On UFK-M (Unified Factory Knowledge Model), an industrial benchmark of 258 analytical questions whose gold answers are produced by executing validated SQL programs, GRA beats a full-context agent by 5.1 pp (88.4% vs. 83.3%), while reading under a third of its input tokens. A graph-free control shows the gain comes chiefly from selective agentic access rather than graph topology, and t
    
[^71]: 一种认知驱动的多维框架用于评估隐喻解释

    A Cognitively Motivated Multidimensional Framework for Evaluating Metaphor Explanations

    [https://arxiv.org/abs/2608.15828](https://arxiv.org/abs/2608.15828)

    本文提出了一个认知驱动的六维框架，用于系统评估隐喻解释质量，并证明其多维性、系统性分歧及自动评估的可行性。

    

    当前对隐喻解释的评估主要依赖于整体质量评分，这很少揭示解释质量的结构或人类判断的一致性与分歧点。我们引入了一个认知驱动的框架，将隐喻解释质量分解为六个理论依据充分的维度。在一项密集标注研究（11,200个评分）中，我们发现：（i）解释质量确实是多维的；（ii）标注者间的分歧是系统性的而非随机的；（iii）这六个维度汇聚成一个共享聚类和两个独立的判断轴。一项探索性可行性研究进一步表明，标准自动评估流程可以恢复该结构的某些部分，能很好地预测最具区分性的维度，而其错误与人类（不）一致性相关。总之，这些结果表明多维评估提供了更丰富的诊断信息。

    arXiv:2608.15828v1 Announce Type: cross  Abstract: Current evaluation of metaphor explanations relies mainly on holistic quality ratings, revealing little about how explanation quality is structured or where human judgments agree and diverge. We introduce a cognitively motivated framework that decomposes metaphor explanation quality into six theoretically grounded dimensions. In a dense annotation study (11,200 ratings), we find that: {\bfseries(i)} explanation quality is genuinely multidimensional; {\bfseries(ii)} annotator disagreement is systematic rather than random; and {\bfseries(iii)} the six dimensions collapse into a shared cluster and two independent axes of judgment. An exploratory feasibility study further shows that a standard automatic evaluation pipeline can recover parts of this structure, predicting the most discriminative dimensions well while its errors correlate human (dis)agreement. Together, these results suggest that multidimensional evaluation offers richer diag
    
[^72]: QuantumPhaseNet：一种具有规范协变几何与量子谱特性的语义概念层级理论，及其经典量子启发模型的原型验证

    QuantumPhaseNet: A Gauge-Covariant Geometric and Quantum-Spectral Theory of Semantic Concept Hierarchies with Prototype Validation of a Classical Quantum-Inspired Model

    [https://arxiv.org/abs/2608.15820](https://arxiv.org/abs/2608.15820)

    本文提出了一种规范协变和量子谱的语义层级理论，并通过经典量子启发模型验证了其波长层级相关性显著优于基线。

    

    我们提出了QuantumPhaseNet，这是Transformer表示的一种规范协变几何与量子谱扩展。上下文相关的语义状态被建模为复振幅；协变相位率诱导出语义波长，用作概念尺度的代理；低频图模式定义了文档级的话语方向。理论部分建立了局部规范不变性、量子块的单位性、波相位注意力的有界性与条件稳定性，以及可校准的幻觉风险公式。我们还在第14.1节中实现了一个完全离线的验证工作室，用于经典的量子启发流水线，并在第16.1节中对其内置的合成设置（n=240，观测噪声0.22，电路噪声0.08，五个随机种子）评估了五个研究问题。RQ1产生了波长层级斯皮尔曼相关性0.852，而基线为0.707，方向准确率为87.3%，以及一个...

    arXiv:2608.15820v1 Announce Type: new  Abstract: We present QuantumPhaseNet, a gauge-covariant geometric and quantum-spectral extension of Transformer representations. Context-dependent semantic states are modeled as complex amplitudes; a covariant phase rate induces a semantic wavelength used as a proxy for conceptual scale; and low-frequency graph modes define a document-level discourse direction. The theoretical part establishes local gauge invariance, unitarity of the quantum block, boundedness and conditional stability of WavePhase Attention, and a calibratable hallucination-risk formulation. We also implemented a fully offline Validation Studio for the classical quantum-inspired pipeline in Section 14.1 and evaluated the five research questions in Section 16.1 on its built-in synthetic setting (n=240, observation noise 0.22, circuit noise 0.08, five seeds). RQ1 yielded a wavelength-hierarchy Spearman correlation of 0.852 versus 0.707 for the baseline, 87.3% direction accuracy, an
    
[^73]: 基于输入侧证据对齐的幻觉跨度检测

    Hallucination Span Detection with Input-Side Evidence Alignment

    [https://arxiv.org/abs/2608.15804](https://arxiv.org/abs/2608.15804)

    本文提出了一种新任务和基于编码器的预测方法，通过输入侧证据对齐联合检测幻觉跨度，利用输出标记的可预测性差异实现高效且可解释的幻觉识别。

    

    幻觉现象仍然是大型语言模型（LLMs）在条件文本生成中可靠使用的主要障碍。现有方法主要评估整个生成文本的事实性，对哪些输出跨度是幻觉性的或它们如何与输入相关提供的洞察有限。我们引入了带有输入侧证据对齐的幻觉跨度检测任务，该任务联合识别幻觉跨度并将输出标记与相应的输入证据对齐。我们的方法基于这样的观察：忠实的输出标记可以从输入中预测，而幻觉标记则不能。因此，我们训练一个基于编码器的模型，从输入表示中预测被掩码的输出标记，利用预测置信度进行幻觉检测，同时自然产生对输入的对齐。实验表明，所提出的方法有效检测幻觉跨度并识别有意义的输入证据。

    arXiv:2608.15804v1 Announce Type: new  Abstract: Hallucinations remain a major obstacle to the reliable use of large language models (LLMs) in conditional text generation. Existing methods primarily assess the factuality of an entire generated text, providing limited insight into which output spans are hallucinated or how they relate to the input. We introduce the task of hallucination span detection with input-side evidence alignment, which jointly identifies hallucinated spans and aligns output tokens with the corresponding input evidence. Our approach is based on the observation that faithful output tokens are predictable from the input, whereas hallucinated tokens are not. We therefore train an encoder-based model to predict masked output tokens from the input representation, using prediction confidence for hallucination detection while naturally producing alignments to the input. Experiments show that the proposed method effectively detects hallucinated spans and identifies meanin
    
[^74]: 利用Mimi编解码器进行元语言表征

    Using the Mimi codec for metalinguistic representations

    [https://arxiv.org/abs/2608.15799](https://arxiv.org/abs/2608.15799)

    本文揭示了Mimi语义码本中的标记实际上映射到多级音素实现（从四音素到亚音素），而非仅捕捉单一音素，这挑战了先前ABX实验的结论。

    

    本文聚焦于Moshi语言模型的神经编解码器Mimi语义标记码本中的2048个标记词典。我们表明，使用Mimi进行的ABX实验未能捕捉语义标记到音素实现之间的映射。通过将Mimi表征重新对齐到TIMIT语料库转录，我们展示了语义码本的2048个标记ID映射到四音素、三音素、双音素、单音素和亚音素实现。

    arXiv:2608.15799v1 Announce Type: new  Abstract: In this paper, we focus on the dictionary of 2048 tokens used in Mimi semantic token codebook, the neural codec of the Moshi language model. We show that the ABX experiment carried out with Mimi fails to capture the mapping of the semantic tokens to phone realisations. By realigning Mimi representations to the TIMIT corpus transcriptions, we show that the 2048 tokens IDs of the semantic codebook map to quadphone, triphone, biphone, phone and subphone realisations.
    
[^75]: KV-Rescue：通过逐步交错恢复推理语言模型KV驱逐损失

    KV-Rescue: Recovering Reasoning Language Model KV Eviction Loss via Stepwise Interleaving

    [https://arxiv.org/abs/2608.15797](https://arxiv.org/abs/2608.15797)

    KV-Rescue通过将轻量级全上下文辅助模型与主模型逐步交错推理，有效弥补了KV驱逐导致的信息缺失，在无训练条件下恢复了高达79%的准确性差距。

    

    arXiv:2608.15797v1 公告类型：新  摘要：KV缓存驱逐限制了长推理轨迹的内存成本，但本质上是有损的，因为模型是从其历史的部分视图进行解码。在激进的预算下，这不仅会降低准确性，还可能导致失控退化，使模型产生不连贯或重复的标记，直到达到长度限制。我们将这种损失的大部分特征化为由缺失上下文引起的信息差距，而非由有限模型容量引起的能力差距。一个被驱逐的7B模型和一个全上下文的1.5B模型会产生互补的错误，而通过预言机在它们的答案之间进行选择，可以恢复与全KV 7B模型准确性差距的79%。基于这一观察，我们提出了KV-Rescue，一个无训练的推理框架，通过使用轻量级的全上下文辅助模型来弥合KV驱逐引入的信息差距。KV-Rescue将两个模型的推理步骤交错到共享轨迹中。一个在线检测器...

    arXiv:2608.15797v1 Announce Type: new  Abstract: KV-cache eviction caps the memory cost of long reasoning traces but is inherently lossy because the model decodes from a partial view of its history. Under aggressive budgets, this not only lowers accuracy but can also cause runaway degeneration, where the model produces incoherent or repetitive tokens until reaching the length limit. We characterize much of this loss as an information gapf caused by missing context, rather than a capability gap caused by limited model capacity. An evicted 7B model and a full-context 1.5B model make complementary errors, and an oracle choice between their answers recovers 79% of the accuracy gap to the full-KV 7B model. Based on this observation, we propose KV-Rescue, a training-free inference framework that bridges the information gap introduced by KV eviction using a lightweight full-context helper. KV-Rescue interleaves reasoning steps from the two models into a shared trajectory. An online detector u
    
[^76]: 路由分歧并非同权重MoE自蒸馏中行为影响的证据

    Routing Divergence Is Not Evidence of Behavioral Influence in Same-Weight MoE Self-Distillation

    [https://arxiv.org/abs/2608.15787](https://arxiv.org/abs/2608.15787)

    该论文通过精确分解证明，在同权重MoE自蒸馏中，路由分歧对块输出的影响极小，其暴露程度主要由残差份额决定，而非行为影响的直接证据。

    

    arXiv:2608.15787v1 公告类型：交叉 摘要：两次混合专家（MoE）前向传播可以共享所有权重，却将相同的令牌路由到不同的专家。这在同权重自蒸馏中造成了一个可能的盲点，其中演示条件化的教师监督仅查询的学生。我们以单步形式研究这种不匹配，使用冻结权重而非作为完整训练轨迹的代理。一种精确的分块分解将路由项（在固定内容下改变门控）与类稠密的内容项分离。在七个开放权重检查点和两个领域中，路由项仅占块输出的$1.6\times$，而其残差流暴露跨度达$3.2\times$。暴露程度由路由块在残差中的份额排序。在两个验证模型中扩展常开骨干网络使暴露单调变化；共模控制支持质量与一致性机制，而非仅分母稀释。

    arXiv:2608.15787v1 Announce Type: cross  Abstract: Two Mixture-of-Experts (MoE) forward passes can share every weight yet route the same token through different experts. This creates a possible blind spot in same-weight self-distillation, where a demonstration-conditioned teacher supervises a query-only student. We study this mismatch in its single-step form, with frozen weights rather than as a proxy for a full training trajectory. An exact blockwise decomposition separates a routing term, which changes gates at fixed content, from a dense-like content term. Across seven open-weight checkpoints and two domains, the routing term spans only $1.6\times$ as a fraction of block output, while its residual-stream exposure spans $3.2\times$. Exposure is ordered by the routed block's share of the residual. Scaling the always-on backbone in two confirmatory models moves exposure monotonically; common-mode controls support a mass-and-coherence mechanism rather than denominator dilution alone. Pr
    
[^77]: TaoLive数字人代理技术报告：训练代理与其操控系统共同进化

    TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness

    [https://arxiv.org/abs/2608.15763](https://arxiv.org/abs/2608.15763)

    本文提出操控系统感知训练（HAT）方法，通过将可进化的操控系统状态纳入训练分布，使数字人代理在实时直播中既能快速响应又能灵活适应动态策略变化。

    

    在直播电商中，AI驱动的数字人主播必须实时回答产品问题、吸引观众并执行不断变化的商业策略。这要求低延迟、事实准确且有效的回复，以及对更新后的活动、合规和风格要求的快速适应。我们开发了一个可进化的操控系统（Harness），将技能（Skills）、钩子（Hooks）、系统提示和工具与模型权重解耦，使得运行时行为无需重新训练即可改变。然而，操控系统的进化创造了一个动态执行环境：在单一配置上微调的紧凑模型可能会记忆名称、模式和提示模板，而不是遵循当前提供的操控系统，而更强的零样本模型又因速度过慢而无法满足实时使用需求。我们通过操控系统感知训练（HAT）来解决这一矛盾，该方法将操控系统状态纳入训练分布。HAT对技能、工具模式和提示应用了任务保持的操控系统状态增强（HSA）。

    arXiv:2608.15763v1 Announce Type: new  Abstract: AI-powered digital-avatar streamers in live e-commerce must answer product questions, engage viewers, and execute changing business strategies in real time. This requires low latency, factual and effective replies, and rapid adaptation to updated campaign, compliance, and style requirements. We develop an evolvable Harness that decouples Skills, Hooks, system prompts, and tools from model weights, allowing runtime behavior to change without retraining. However, Harness evolution creates a moving execution environment: compact models fine-tuned on one configuration may memorize names, schemas, and prompt templates rather than follow the Harness currently provided, while stronger zero-shot models are too slow for real-time use. We address this tension with Harness-Aware Training (HAT), which makes Harness states part of the training distribution. HAT applies task-preserving Harness-State Augmentation (HSA) to Skills, tool schemas, prompt s
    
[^78]: 宣传取证：恢复AI驱动影响力活动的生成管道

    Propaganda Forensics: Recovering the Generation Pipeline of an AI-Driven Influence Campaign

    [https://arxiv.org/abs/2608.15746](https://arxiv.org/abs/2608.15746)

    本文通过取证分析揭示了AI驱动宣传活动的生成管道，包括提示泄露和模型归因，并开发了PROPAGIA语料库来识别其独特的说服特征。

    

    我们对近期一次AI驱动影响力活动背后的生成管道进行了取证分析。我们引入了PROPAGIA，一个包含2,646篇来自Storm-1516/CopyCop活动的法语宣传文章语料库，该活动由VIGINUM和INSIKT GROUP于2025年披露。为了对比，我们使用了SIPA，一个同期人类撰写的法语主流新闻语料库。通过主题建模、模糊性和情感分析，我们首先隔离了宣传特有的说服技巧，发现PROPAGIA在模糊性、主观性和负面性方面远超SIPA，且引用来源更少。随后，我们在84个PROPAGIA网站中的50个上发现了提示指令泄露，包括一份逐字逐句的十点编辑规范，这解释了部分差异，以及高跨文章冗余性。最后，我们表明基于重写的检测支持INSIKT GROUP对Llama 3家族的归因，但也暗示了其他模型的参与。

    arXiv:2608.15746v1 Announce Type: new  Abstract: We present a forensic analysis of the generation pipeline behind a recent AI-driven influence campaign. We introduce PROPAGIA, a corpus of 2,646 propagandist French articles from the Storm-1516/CopyCop campaign disclosed by VIGINUM and INSIKT GROUP in 2025. For comparison, we rely on SIPA, a corpus of human-written French mainstream press from the same period. Using topic modeling, vagueness and sentiment analysis, we first isolate persuasion techniques characteristic of propaganda, with PROPAGIA far exceeding SIPA in vagueness, subjectivity and negativity, and citing fewer sources. We then find prompt instruction leaks on 50 of the 84 PROPAGIA websites, including a verbatim ten-point editorial specification accounting for several of these differences, together with high cross-article redundancy. Finally, we show that rewriting-based detection supports INSIKT GROUP's attribution to the Llama 3 family, but also suggests the involvement of
    
[^79]: 超越单一对象：利用大型语言模型学习3D关系

    Beyond Single Object: Learning 3D Relations with Large Language Models

    [https://arxiv.org/abs/2608.15710](https://arxiv.org/abs/2608.15710)

    该论文提出了一种新框架，通过多对象指令数据集、补丁交互变换器和应用基准，使3D-LLMs能够进行详细的跨对象几何推理，显著优于现有模型。

    

    我们解决了3D-LLMs中的一个基本差距：现有模型专注于单一对象/场景描述，难以处理详细的跨对象比较。我们提出一个框架，用于跨多个对象的详细对象级推理，包含三个组成部分：（1）MO3D（3D中的多对象），一个需要细粒度多对象比较的指令数据集；（2）Multi-3DLLM，使用一个最小化补丁交互变换器（PIT），在保留局部几何结构的同时建模对象间/对象内关系；（3）迷你应用，两个应用驱动的基准测试（形状匹配、变化描述），用于探测实际使用中的几何理解。最近的3D-LLMs和2D-VLMs在这些任务上表现不佳，缺乏以比较为中心的设计和几何感知。相比之下，基于我们的混合数据训练的Multi-3DLLM学习了几何推理，在MO3D上超越了所有基线，并为单对象分类提供了正向迁移。

    arXiv:2608.15710v1 Announce Type: cross  Abstract: We address a fundamental gap in 3D-LLMs: existing models focus on single-object/scene description, struggling with detailed, inter-object comparison. We propose a framework for detailed object-level reasoning across multiple objects with three components: (1) MO3D (Multi-Object in 3D), an instruction dataset requiring fine-grained multi-object comparison; (2) Multi-3DLLM, using a minimal Patch-Interaction Transformer (PIT) that models inter-/intra-object relationships while preserving local geometry; (3) Mini-apps, two application-driven benchmarks (Shape Mating, Change Captioning) that probe geometric understanding for practical use. Recent 3D-LLMs and 2D-VLMs perform poorly on these tasks, lacking both comparison-centric design and geometric awareness. In contrast, Multi-3DLLM trained on our mixture data learns geometric reasoning, surpasses all baselines on MO3D, and provides positive transfer to single-object classification.
    
[^80]: BERTopic-病毒性优先排序：一个用于COVID-19和猴痘错误信息在Twitter上进行主题与比较分析的可扩展框架

    BERTopic-Virality Prioritisation: A Scalable Framework for Thematic and Comparative Analysis of COVID-19 and Monkeypox Misinformation on Twitter

    [https://arxiv.org/abs/2608.15691](https://arxiv.org/abs/2608.15691)

    本文提出了BERTopic-VP框架，通过将病毒性优先排序层与主题建模结合，并辅以混合错误信息检测模块，能够优先识别语义连贯且快速扩散的高影响健康错误信息主题。

    

    arXiv:2608.15691v1 公告类型：新 摘要：大流行期间传播的健康错误信息可能迅速获得关注，形成与公共卫生指导相竞争的有害叙事。大多数主题建模流程将参与度视为外部结果，限制了它们优先处理语义连贯且快速扩散主题的能力。我们引入了BERTopic-VP，一个病毒性优先的主题建模框架，该框架结合了基于上下文嵌入的聚类（BERTopic）与事后病毒性优先排序（VP）层。该流程还辅以两阶段混合错误信息检测模块，该模块融合了基于内容的监督分类器与来自公共卫生知识库的外部验证信号。应用于三个基准数据集，COVID-19_FNIR、Monkeypox和Constraint，该框架实现了强大的分类性能，F1最高达0.950，ROC-AUC最高达0.989，同时识别出高影响聚类。

    arXiv:2608.15691v1 Announce Type: new  Abstract: Health misinformation circulating during pandemics can gain traction rapidly, creating harmful narratives that compete with public health guidance. Most topic-modelling pipelines treat engagement as an external outcome, limiting their ability to prioritise semantically coherent topics that are also rapidly diffusing. We introduce BERTopic-VP, a virality-prioritised topic-modelling framework that combines contextual embedding-based clustering (BERTopic) with a post hoc Virality Prioritisation (VP) layer. The pipeline is complemented by a two-stage hybrid misinformation detection module that fuses a supervised content-based classifier with an external verification signal derived from public-health knowledge bases. Applied to three benchmark datasets, COVID-19_FNIR, Monkeypox, and Constraint, the framework achieves strong classification performance, with F1 up to 0.950 and ROC-AUC up to 0.989, while identifying high-impact clusters under to
    
[^81]: 将说服理论整合到社交媒体健康错误信息传播的流行病学建模中

    Integrating Persuasion Theory into the Epidemiological Modelling of Health Misinformation Spread on Social Media

    [https://arxiv.org/abs/2608.15689](https://arxiv.org/abs/2608.15689)

    本文提出ELM-SIRMMM框架，通过将说服理论（ELM）与扩展的流行病学模型（SIRMMM）相结合，动态模拟社交媒体健康错误信息的传播，并验证了其跨数据集的泛化性。

    

    本研究提出了一种混合流行病学和行为学框架，以模拟社交媒体上健康错误信息的传播。我们将经典的易感-感染-恢复（SIR）模型扩展为六室结构（SIRMMM），纳入错误信息易感者（MS）、错误信息感染者（MI）和错误信息恢复者（MR）三个室，以更好地反映错误信息生命周期的动态。为考虑个体层面的行为差异，我们通过整合精细加工可能性模型（ELM）中的心理信号（包括情感极性、参与度指标和认知努力）来扩展SIRMMM模型，这些信号动态调节错误信息传播率，从而形成ELM-SIRMMM框架。模型参数使用FibVID数据集（涵盖Twitter上的COVID-19错误信息）进行估计，并在两个额外数据集上测试了泛化能力：MC-Fake（情感错误信息）。

    arXiv:2608.15689v1 Announce Type: cross  Abstract: This study presents a hybrid epidemiological and behavioural framework to simulate the spread of health misinformation on social media. We extend the classical Susceptible--Infected--Recovered (SIR) model to a six-compartment structure (SIRMMM), incorporating Misinformed Susceptible (MS), Misinformed Infected (MI), and Misinformed Recovered (MR) compartments to better reflect the dynamics of the misinformation lifecycle. To account for individual-level behavioural variation, we extend the SIRMMM model by integrating psychological signals from the Elaboration Likelihood Model (ELM), including sentiment polarity, engagement metrics, and cognitive effort, which dynamically modulate the misinformation transmission rate, yielding the ELM-SIRMMM framework. Model parameters were estimated using the FibVID dataset, which captures COVID-19 misinformation on Twitter. Generalisability was tested on two additional datasets: MC-Fake (emotional misi
    
[^82]: 当故事演化：在开放式世界模拟中跨智能体架构评估LLM叙事能力

    When Stories Evolve: Benchmarking LLM Storytelling Across Agent Architectures in Open-Ended World Simulations

    [https://arxiv.org/abs/2608.15654](https://arxiv.org/abs/2608.15654)

    本文提出了WSE-bench基准，发现LLM叙事中一致性与丰富度呈非凹Pareto前沿，增加结构可丰富轨迹但不提升一致性。

    

    大型语言模型可以生成流畅的故事，但开放式叙事需要超越局部流畅性的能力。在不断演化的世界模拟和AI原生游戏中，模型必须在世界变化时保持事实、关系、因果依赖和角色状态的一致性。我们引入了WSE-bench，一个过程基准测试，分别评估动态LLM叙事中的持续生成、规范一致性和有意义发展。生成覆盖率记录产生的计划叙事步骤比例；一致性追踪规范何时被打破；丰富度衡量分支性、玩家塑造轨迹的有意义发展程度。在前沿模型中，一致性和丰富度并不构成平滑的权衡：其经验Pareto前沿是非凹的，存在多个非支配的中间配置，无法通过任何正线性加权选择。增加结构可以丰富轨迹，但并不统一提高一致性。

    arXiv:2608.15654v1 Announce Type: cross  Abstract: Large language models can write fluent stories, but open-ended storytelling requires more than local fluency. In evolving world simulations and AI-native games, models must preserve facts, relationships, causal dependencies, and character states as the world changes. We introduce WSE-bench, a process benchmark that separately evaluates sustained generation, canonical coherence, and meaningful development in dynamic LLM storytelling. Generation Coverage records the proportion of planned narrative steps produced; Consistency tracks when canon breaks; and Richness measures how meaningfully branching, player-shaped trajectories develop. Across frontier models, Consistency and Richness do not form a smooth trade-off: their empirical Pareto frontier is non-concave, with several non-dominated intermediate configurations that no positive linear weighting can select. Added structure can enrich trajectories, but it does not uniformly improve coh
    
[^83]: 维基词典作为英语方言的众包词汇资源

    Wiktionary as a Crowdsourced Lexicon for English Dialects

    [https://arxiv.org/abs/2608.15641](https://arxiv.org/abs/2608.15641)

    本文验证了维基词典作为英语方言众包词汇资源的有效性，其覆盖范围与传统词典相当，并与社交媒体语言高度契合，同时揭示了其宏观局限性。

    

    arXiv:2608.15641v1 公告类型：新 摘要：本文评估了维基词典作为英语方言的伦理众包词汇资源的效用。我们采用两阶段方法，首先对12种国家英语变体的众包词汇进行了深入的描述性分析，然后将该词汇应用于地理参考的国家级社交媒体语言数据，以检验这一众包方言词汇在现实世界中的表现。我们证明，维基词典在区域性和外圈英语变体方面的覆盖范围匹配或超过了传统词典（如《牛津英语词典》OED）。我们对新西兰英语的特定方言案例研究发现，基于构词模式，维基词典与OED之间高度一致（R = 0.883）。同样，我们观察到方言词汇与地理参考的社交媒体语言之间高度吻合。虽然本文发现维基词典对词汇属性具有广泛覆盖，但也指出了其在宏观层面的一些局限性。

    arXiv:2608.15641v1 Announce Type: new  Abstract: This paper evaluates Wiktionary as an ethically crowdsourced lexicon for English dialects. We took a two-phase approach, providing an in-depth descriptive analysis of the crowdsourced lexicon for 12 national varieties of English before applying the lexicon to geo-referenced, country-level social media language data to examine the real-world performance of this crowdsourced dialect lexicon. We demonstrate that Wiktionary matches or exceeds the coverage of traditional dictionaries, such as the Oxford English Dictionary (OED), for regional and Outer-Circle varieties. Our dialect-specific case study on New Zealand English found high alignment between Wiktionary and the OED based on word-formation patterns (R = 0.883). Similarly, we observed high alignment between the dialect lexicon and geo-referenced social media language. While this paper found that Wiktionary has broad coverage of lexical properties, it also highlighted some of the macro-
    
[^84]: 评估工具对人类和大型语言模型测量的是同一事物吗？潜在结构分析

    Do Assessment Instruments Measure the Same Thing for Humans and LLMs? A Latent Structure Analysis

    [https://arxiv.org/abs/2608.15630](https://arxiv.org/abs/2608.15630)

    本研究通过潜在结构分析，检验了用于评估人类的标准化测试是否在LLMs上测量相同的潜在构念，发现该条件可能不成立，从而质疑了直接跨物种解释分数效度的可行性。

    

    arXiv:2608.15630v1 公告类型：交叉 摘要：大型语言模型的快速发展和日益广泛的部署，使得理解其能力变得越来越重要。一种常见的方法是使用最初为测量人类技能和能力而设计的评估工具（如标准化考试）来评估LLMs，并将这些工具上的表现作为关于LLMs在相同技能上潜在能力的可推广主张的证据，这些技能正是评估工具旨在对人类测量的。然而，从效度角度来看，这种推断要求为人类建立的观察表现与潜在构念之间的关系同样适用于LLMs。特别是，转移分数解释的一个必要条件是，对评估响应的潜在结构具有相似性。在本研究中，我们考察了这一条件在两个教育情境中是否成立：高中化学和定量推理。

    arXiv:2608.15630v1 Announce Type: cross  Abstract: The rapid development and growing deployment of large language models (LLMs) have made it increasingly important to understand their capabilities. A common approach is to evaluate LLMs using assessment instruments originally designed to measure skills and competencies in humans, such as standardized exams, and to use performance on these instruments as evidence for generalizable claims about LLMs' underlying abilities on the same skills the assessments are intended to measure in humans. However, from a validity perspective, such inferences require that the relationship between observed performance and underlying constructs established for humans also holds for LLMs. In particular, a necessary condition for transferring score interpretations is similarity in the latent structure of responses to the assessment. In this study, we examine whether this condition holds in two educational contexts: high-school chemistry and a quantitative rea
    
[^85]: 孟加拉语MCQ：低资源语言下学术多项选择题的自动生成与答案预测

    BengaliMCQ: Automatic Generation and Answer Prediction of Academic Multiple-Choice Questions in a Low-Resource Language

    [https://arxiv.org/abs/2608.15547](https://arxiv.org/abs/2608.15547)

    本文提出了一种结构感知的RAG框架，通过将孟加拉语教科书建模为层级图并利用对比训练的图神经网络进行检索，显著提升了低资源语言下多项选择题的生成质量和答案预测准确性。

    

    摘要：arXiv:2608.15547v1 公告类型：新  摘要：传统的检索增强生成（RAG）框架在处理文档时未关注其层级结构，导致性能不佳，尤其是在孟加拉语等低资源语言中。为解决这一问题，我们提出了一种结构感知的RAG框架，该框架将孟加拉语教科书建模为层级图，并使用对比训练过的图神经网络检索一小部分相关段落。这些段落为大型语言模型提供了聚焦的上下文，从而支持特定主题的多项选择题（MCQ）生成和领域内答案预测。实验结果表明，我们的框架在检索指标上优于强大的密集检索基线，生成的相关MCQ更多，并实现了更高的答案预测准确性。

    arXiv:2608.15547v1 Announce Type: new  Abstract: Traditional retrieval-augmented generation (RAG) frameworks process documents without attending to their hierarchical structure, leading to poor performance, especially in low-resource languages such as Bengali. To address this, we propose a structure-aware RAG framework that models Bengali textbooks as hierarchical graphs and uses a contrastively trained graph neural network to retrieve a small set of relevant passages. These passages provide focused context for a large language model, enabling topic-specific multiple-choice question (MCQ) generation and in-domain answer prediction. Experimental results demonstrate that our framework outperforms strong dense retrieval baselines across retrieval metrics, produces more relevant MCQs, and achieves superior answer prediction accuracy.
    
[^86]: L3Cube-IndicQuest v2：用于评估大型语言模型在印度语言中事实知识的大规模多语言基准

    L3Cube-IndicQuest v2: A Large-Scale Multilingual Benchmark for Evaluating Factual Knowledge of Large Language Models Across Indic Languages

    [https://arxiv.org/abs/2608.15535](https://arxiv.org/abs/2608.15535)

    该论文提出了一个覆盖20种语言、含69,420个问答对的印度知识多语言基准，通过混合生成与验证策略确保质量，并评估了多个LLM的性能。

    

    我们提出了L3Cube-IndicQuest v2，这是一个大规模、金标准的多语言问答基准，用于评估大型语言模型（LLMs）对印度特定事实知识的掌握情况。该基准包含3,471个基于课程体系构建的英文问答对，涵盖九个领域，数据来源于教育课程、竞争性考试材料和领域特定参考书。我们引入了一种实用的混合构建策略，结合了基于上下文的LLM问题生成与验证、语义去重和人工核查，从而在保持注释质量的同时实现基准数据的可扩展创建。该基准被翻译成19种印度语言，最终形成一个包含69,420个问答对、覆盖20种语言的公开多语言数据集。我们评估了六个LLM，采用三种协议：LLM作为评判者以及两种确定性词汇标准，即精确子串和词重叠。

    arXiv:2608.15535v1 Announce Type: new  Abstract: We present L3Cube-IndicQuest v2, a large-scale gold-standard multilingual question-answering benchmark for evaluating the India-specific factual knowledge of Large Language Models (LLMs). The benchmark comprises 3,471 curriculum-grounded English question--answer pairs spanning nine domains, curated from educational curricula, competitive examination materials, and domain-specific reference books. We introduce a practical hybrid construction strategy that combines context-grounded LLM-based question generation and validation with semantic deduplication and human verification, enabling scalable creation of benchmark data while preserving annotation quality. The benchmark is translated into 19 Indic languages, yielding a publicly released multilingual dataset of 69,420 question--answer pairs across 20 languages. We evaluate six LLMs under three protocols: LLM-as-a-judge and two deterministic lexical criteria, exact-substring and word-overla
    
[^87]: 为什么摘要变得中性：人类反馈强化学习中的情感漂移策略归因

    Why Summaries Turn Neutral: Policy Attribution for Sentiment Drift in Reinforcement Learning from Human Feedback

    [https://arxiv.org/abs/2608.15530](https://arxiv.org/abs/2608.15530)

    本文揭示了RLHF导致摘要情感漂移的机制，并提出策略归因框架和情感感知正则化技术来缓解这一问题。

    

    基于人类反馈的强化学习（RLHF）使大型语言模型与人类偏好对齐，提高了摘要的流畅性和安全性，但导致了情感漂移：过度中性的摘要失去了情感细微差别。我们诊断了RL为何充当情感中和器，并提出了策略归因（Policy Attribution）框架，该框架利用梯度和逻辑分解来追踪漂移至奖励模型（RM）信号和KL（Kullback-Leibler）惩罚。情感漂移反映了在偏好不确定性下，对最大化预期奖励的“低风险”标记的战略性偏向（Stiennon et al., 2020; Gao, Schulman, and Hilton, 2023）。在Reddit TL;DR和CNN/DailyMail数据集上，RLHF摘要获得了更高的奖励，但情感方差降低了30-40%。跨八种语言的跨语言分析显示漂移与语言无关，但形态更丰富的语言受到更强抑制（Krasitskii et al., 2026）。我们提出并验证了一种情感感知正则化技术，以减少这种漂移。

    arXiv:2608.15530v1 Announce Type: new  Abstract: Reinforcement learning with human feedback (RLHF) aligns LLMs with human preferences, improving summarization fluency and safety, but causes sentiment drift: overly neutral summaries stripped of emotional nuance. We diagnose why RL acts as a sentiment neutralizer and present Policy Attribution, a framework using gradient and logit decomposition to trace drift to reward model (RM) signals and KL (Kullback-Leibler) penalty. Sentiment drift reflects a strategic bias toward "low-risk" tokens maximizing expected rewards under preference uncertainty (Stiennon et al., 2020; Gao, Schulman, and Hilton, 2023). On Reddit TL;DR and CNN/DailyMail, RLHF summaries get higher rewards but show 30-40% lower sentiment variance. Cross-lingual analysis across eight languages shows language-independent drift, with morphologically richer languages more suppressed (Krasitskii et al., 2026). We propose and validate a sentiment-aware regularization technique redu
    
[^88]: 语言模型是否一致地编码当前年份？

    Do Language Models Consistently Encode the Current Year?

    [https://arxiv.org/abs/2608.15507](https://arxiv.org/abs/2608.15507)

    本文通过两个探测任务揭示语言模型对当前年份的编码不一致，关联任务机制类似事实回忆，而声明性任务缺乏因果路径，导致更新年份困难。

    

    arXiv:2608.15507v1 公告类型：新 摘要：对当前时间的一致概念对于时间推理很重要，然而语言模型如何表示当前时间尚未被充分理解。我们贡献了两个任务，以概念上不同的方式探测当前年份：一个关联任务，通过动词时态推断当前年份，以及一个声明性任务，直接查询当前年份。两个任务估计的当前年份都在指令调优语言模型训练后数据截止日期的一年之内。对于基础模型，关联任务的预测可作为预训练数据截止日期的强代理，在13个模型上的平均误差仅为10个月。然而，它们的内部机制存在分歧：关联任务使用类似于事实回忆的机制，而声明性任务缺乏一致的因果路径。这种分歧对更新语言模型中的当前年份构成了挑战。提示、监督微调或权重更新均无法解决此问题。

    arXiv:2608.15507v1 Announce Type: new  Abstract: A consistent concept of the current time is important for temporal reasoning, yet how language models represent the current time is not well understood. We contribute two tasks that probe the current year in conceptually distinct ways: an associative task, which infers the current year from verb tense, and a declarative task, which directly queries for the current year. Both tasks estimate current years within one year of the post-training data cutoff of instruction-tuned language models. For base models, predictions on the associative task serve as a strong proxy for the pre-training data cutoff, with an average error of only 10 months across 13 models. However, their internal mechanisms diverge: the associative task uses mechanisms similar to factual recall, while the declarative task lacks consistent causal pathways. This divergence poses a challenge for updating the current year in language models. None of prompting, SFT, or weight e
    
[^89]: 语言模型遭受歧义诅咒

    Language models suffer from a curse of ambiguity

    [https://arxiv.org/abs/2608.15448](https://arxiv.org/abs/2608.15448)

    本文揭示了一个“歧义诅咒”现象：在语言模型等神经网络中，下一个词元分布的歧义性越强，学习其准确分布的难度越大，这源于容量、嵌入、训练步骤和采样噪声等多方面限制。

    

    arXiv:2608.15448v1 公告类型：新 摘要：大型语言模型越来越依赖采样作为自身改进的驱动力，这使得其学习分布的保真度比以往任何时候都更为关键。然而，并非所有分布都同样容易学习。在这项工作中，我们识别出一种“歧义诅咒”：在大型语言模型中，更广泛地说，在所有产生离散概率分布的神经网络中，下一个词元分布的歧义性越强，准确学习它的难度就越大。通过广泛的理论分析，我们将这种诅咒追溯到架构和学习根源。更歧义的分布需要更多容量来存储、更大的嵌入来表示、更多步骤来拟合，并会放大词元采样噪声。我们在具有受控真实标签的合成任务上验证了这些发现，并在基于真实数据训练的语言模型中观察到了相同的特征。我们的结果为统计能力提供了新视角。

    arXiv:2608.15448v1 Announce Type: new  Abstract: Large language models increasingly rely on sampling as a driver of their own improvement, making the fidelity of their learned distributions more critical than ever. Yet, not all distributions are equally easy to learn. In this work, we identify a curse of ambiguity: in large language models, and more broadly in all neural networks that produce discrete probability distributions, the more ambiguous a next-token distribution is, the harder it is to learn accurately. Through an extensive theoretical analysis, we trace this curse to architectural and learning roots. More ambiguous distributions require more capacity to be stored, larger embeddings to be represented, more steps to be fitted, and amplify token-sampling noise. We validate these findings on synthetic tasks with controlled ground truth and observe the same signatures in language models trained on real data. Our results provide a new perspective on the statistical capabilities of
    
[^90]: 词性语义空间

    Semantic Space of Parts of Speech

    [https://arxiv.org/abs/2608.15443](https://arxiv.org/abs/2608.15443)

    本文通过word2vec嵌入和神经网络降维，构建词性语义三维空间，揭示了词性分类的模糊性和边界词现象。

    

    arXiv:2608.15443v1 公告类型：新 摘要：词性分类在欧洲语言学传统中被理解为明确分类，这也反映在语料库语言学中，每个消歧后的词元被精确分配一个词性。然而，所分配的类别在很大程度上由注释手册中的任意决定所决定。由于一些词在语义或典型句法上介于词性之间，且某些词性比其他词性更接近，词性分类本质上似乎是模糊的。我们使用word2vec嵌入分析这种模糊性，训练神经网络将高维空间降维到三个与词性确定相关的维度。这创建了一个三维空间，我们将数千个单词映射到该空间，揭示哪些是原型词，哪些位于边界，并可视化词性之间的关系。本研究使用法语通用依赖词性标签。

    arXiv:2608.15443v1 Announce Type: new  Abstract: Parts of speech categorization is understood in the European linguistic tradition as crisp categorization, which is also reflected in corpus linguistics, where each disambiguated token is assigned exactly one POS. However, the assigned categories are largely determined by arbitrary decisions distilled into annotation manuals. Since some words stand between parts of speech in their semantics or typical syntax, and some parts of speech are closer to each other than others, POS categorization seems inherently fuzzy. We analyze this fuzziness using word2vec embeddings, training a neural network to reduce their high dimensionality to three dimensions relevant for determining parts of speech. This creates a three-dimensional space onto which we map several thousand words, revealing which are prototypical and which lie on the boundaries, and visualizing relationships between parts of speech. The study uses Universal Dependencies POS tags for Fr
    
[^91]: 针对一个模型设防，对下一个模型敞开：法律多选题基准中的仅选项可解性

    Gated Against One Model, Open to the Next: Option-Only Solvability in Legal Multiple-Choice Benchmarks

    [https://arxiv.org/abs/2608.15428](https://arxiv.org/abs/2608.15428)

    本文发现法律多选题基准存在严重的数据泄漏，模型在无问题情况下仅凭选项即可高概率答对，且泄漏对未参与筛选的模型同样有效，挑战了基准的有效性。

    

    摘要：多选题基准的评分标准是模型是否选择了正确的选项，而非是否理解了问题本身。衡量这一差距需要谨慎：如果一个模型对大多数题目都选择A，那么无论正确答案是否位于A，其得分都会高于随机水平，而当答案不在A时，这又可能被误读为识别能力。我们在UA-JudgeExam上进行了测量：该基准包含11,990道四选项题目，附有官方答案，由乌克兰法官资格高等委员会发布。在仅显示选项而无问题的情况下，Claude Haiku 4.5的得分为0.383，高于随机水平，且泄漏集中：11.8%的题目在全部八种选项顺序下均可盲答，而随机预期仅为0.2项。这并非引文式泄漏：对280,059版乌克兰立法的搜索仅恢复0.128。排除这些泄漏项后保留8,128道题目，在该子集上，筛选模型自身的得分降至0.204，而GPT-5.6——虽未参与筛选——在隐藏问题的情况下仍能答对其中0.515的题目。对十二个保留模型的评分显示……

    arXiv:2608.15428v1 Announce Type: cross  Abstract: Multiple-choice benchmarks are graded on whether a model picks the right option, not on whether it needed the question. Measuring that gap takes care: a model answering A to most items scores above chance wherever the key sits at A, and reads as recognition when it is not. We measure it on UA-JudgeExam: 11,990 four-option items with official keys, published by Ukraine's Higher Qualification Commission of Judges.   Shown the options and no question, Claude Haiku 4.5 scores 0.383 against chance, and the leak is concentrated: 11.8% of items are answered blind on all eight option orders, against 0.2 items expected by chance. It is not quotation: search over 280,059 editions of Ukrainian legislation recovers 0.128. Gating those out retains 8,128 items, on which the gating model itself now scores 0.204, and GPT-5.6, which took no part in the selection, still answers 0.515 of them with the question hidden. Scoring twelve held-out models on th
    
[^92]: 大语言模型辅助的电池储能系统集成配电网运行监测

    Large Language Model Assisted Operational Monitoring for Battery Energy Storage System Integrated Power Distribution Networks

    [https://arxiv.org/abs/2608.15396](https://arxiv.org/abs/2608.15396)

    该论文提出了一种将大语言模型与结构化遥测数据库相结合的AI监测框架，支持通过自然语言查询实现对电池储能系统集成配电网的自动运行监测与约束评估。

    

    电池储能系统（BESS）越来越多地用于配电网的电压调节和需求响应，这增加了电网运营商可用的运行遥测数据的数量和复杂性。本文提出了一种基于人工智能的监测框架，该框架将大语言模型（LLM）接口与结构化遥测数据库连接起来，用于BESS集成配电系统分析。操作员以自然语言提交问题，并利用预定义的数据库模式信息和批准的KPI视图，将其转换为验证过的SQL查询。检索到的测量数据，包括母线电压、荷电状态、有功功率和无功功率，将根据电压限制、BESS运行和需求响应跟踪的工程约束进行评估。该框架使用配备BESS的配电馈线在基于无功功率的电压控制下运行的硬件在环联合仿真数据进行了验证。

    arXiv:2608.15396v1 Announce Type: new  Abstract: Battery energy storage systems (BESS) are increasingly used in distribution networks for voltage regulation and demand response, which increases the volume and complexity of operational telemetry available to grid operators. This paper presents an AI-enabled monitoring framework that connects a large language model (LLM) interface with a structured telemetry database for BESS-integrated distribution system analysis. Operator questions are submitted in natural language and translated into validated SQL queries using predefined database schema information and approved KPI views. Retrieved measurements, including bus voltages, state of charge, active power, and reactive power, are evaluated against engineering constraints for voltage limits, BESS operation, and demand response tracking. The framework is validated using hardware-in-the-loop co-simulation data from a BESS-equipped distribution feeder operating under reactive power-based volta
    
[^93]: 机器的内在时钟：大语言模型是否共享人类的时间错觉？

    The Machine's Internal Clock: Do LLMs Share Human Temporal Illusions?

    [https://arxiv.org/abs/2608.15394](https://arxiv.org/abs/2608.15394)

    本研究通过新基准发现，大语言模型在时间错觉任务中比人类更倾向于选择文献预测场景，表明模型可能缺乏人类对时间的主观感知偏差。

    

    arXiv:2608.15394v1 公告类型：新 摘要：人类对时间的感知是主观的。已有充分记录的时间错觉表明，大脑依赖于上下文和关系线索来判断持续时间，而不是直接跟踪流逝的时间。先前的研究通过视觉和听觉刺激确立了这些效应。现有的大语言模型（LLM）时间感知评估侧重于估计事件持续时间或多步骤时间推理。在这项工作中，我们使用一个包含6,684个叙事对的新基准，涵盖五种时间错觉，调查仅凭书面叙事是否能引发人类的时间错觉。我们发现，人类读者（60名参与者）仅在五种错觉中的两种中偏好预期场景，即那些操作在文本中直接可见而非需要读者内部模拟持续时间的场景。我们在同一基准上评估了14个LLM。令人惊讶的是，我们发现模型在五种错觉中的四种中选择了文献预测的场景，与人类表现出现分歧。

    arXiv:2608.15394v1 Announce Type: new  Abstract: Human perception of time is subjective. Well-documented temporal illusions show that the brain relies on context and relational cues for judging duration instead of tracking elapsed time directly. Prior studies established these effects with visual and auditory stimuli. Existing LLM evaluations of temporal perception focus on estimating event durations or multi-step temporal reasoning. In this work, we investigate whether written narratives alone can evoke human temporal illusions, using a new benchmark of 6,684 narrative pairs spanning five illusions. We find that human readers (60 participants) prefer expected scenarios in only two of the five illusions, those where the manipulation is directly visible in text rather than requiring readers to internally simulate duration. We evaluate 14 LLMs on the same benchmark. Surprisingly, we find that models pick the literature-predicted scenario across four of the five illusions, diverging from 
    
[^94]: 当AI重写时，分类器放松：针对讽刺和AI改写社交文本的不确定性感知情感分析

    When AI Rewrites, Classifiers Relax: Uncertainty-Aware Sentiment Analysis on Sarcastic and AI-Paraphrased Social Text

    [https://arxiv.org/abs/2608.15338](https://arxiv.org/abs/2608.15338)

    本研究揭示情感分类器在讽刺文本上表现出较低置信度，但在AI改写文本上准确率更高，发现AI改写通过消除分布噪声提升了分类性能。

    

    arXiv:2608.15338v1 公告类型：交叉 摘要：情感分类器越来越多地应用于社交媒体内容，这些内容要么是讽刺性的，要么是AI生成的——在这两种分布状态下，标准评估提供的指导很少。我们提出了一个三部分的情感分类器在这些条件下行为的实证研究。首先，我们发现讽刺文本上的置信度分数显著低于非讽刺文本（Mann-Whitney $p = 2 \times 10^{-6}$），这证实了分类器即使没有显式的不确定性建模，也能感知到自身对讽刺内容的不确定性。其次，反直觉的是，我们显示情感分类器在AI改写的评论上比原始人类撰写的文本上达到更高的准确率（RoBERTa：Qwen3.5-4B改写$+5.8$个百分点，Gemma4-E4B改写$+3.7$个百分点），揭示了一种跨域风格对齐效应：AI改写移除了混淆Twitter训练分类器的分布噪声，产生更干净、更易分类的文本。

    arXiv:2608.15338v1 Announce Type: cross  Abstract: Sentiment classifiers are increasingly applied to social media content that is either sarcastic or AI-generated --- two distributional regimes where standard evaluations offer little guidance. We present a three-part empirical study of sentiment classifier behaviour under these conditions. First, we find that confidence scores on sarcastic text are significantly lower than on non-sarcastic text (Mann--Whitney $p = 2 \times 10^{-6}$), confirming that classifiers sense their own uncertainty on ironic content even without explicit uncertainty modelling. Second, and counterintuitively, we show that sentiment classifiers achieve higher accuracy on AI-paraphrased reviews than on the original human-authored text (RoBERTa: $+5.8$ pp for Qwen3.5-4B paraphrases, $+3.7$ pp for Gemma4-E4B), revealing a cross-domain stylistic alignment effect: AI paraphrases remove distributional noise that confounds Twitter-trained classifiers, producing cleaner, 
    
[^95]: 逻辑嵌入用于论证分析

    Logical Embeddings for Argument Analysis

    [https://arxiv.org/abs/2608.15325](https://arxiv.org/abs/2608.15325)

    本文提出了一种新颖的逻辑嵌入框架，利用数学逻辑和RKHS理论替代传统词嵌入，以更好地表示论证语义并确保理论上的最优性。

    

    arXiv:2608.15325v1 公告类型：交叉 摘要：我们提出了一种面向机器学习论证分析任务的新框架。我们的提议涉及将大多数NLP任务中使用的传统上下文词嵌入替换为逻辑嵌入，这是一种直接利用论证结构的替代编码方式。本质上，逻辑嵌入封装了论证的逻辑语义，从而能更好地表示其含义。支持这些嵌入的是基于数学逻辑的相似度度量，它提供了透明的邻近性概念，并保证满足当前基于余弦相似度的上下文词嵌入无法确保的几个理想理论性质。这种相似度度量在论证集合上诱导出一个正半定核，使我们能够利用再生核希尔伯特空间（RKHS）理论唯一定义逻辑嵌入。此外，我们证明这种编码在某种意义上是最优的。

    arXiv:2608.15325v1 Announce Type: cross  Abstract: We propose a new framework for machine-learning-oriented argument analysis tasks. Our proposal involves replacing traditional contextualized word embeddings used in most NLP tasks with logical embeddings, an alternative encoding that directly exploits argumentation structures. In essence, logical embeddings encapsulate the logical semantics of an argument, allowing for a better representation of its meaning. Supporting these embeddings is a mathematical logic-based similarity measure that offers a transparent notion of proximity and is guaranteed to satisfy several desirable theoretical properties that current cosine similarity-based contextualized word embeddings cannot assure. This similarity measure induces a positive semi-definite kernel on the set of arguments, enabling us to uniquely define logical embeddings using the theory of Reproducing Kernel Hilbert Spaces (RKHS). Moreover, we prove that this encoding is optimal, in the sen
    
[^96]: 在语言模型训练过程中，概念何时变得功能上充分？

    When Do Concepts Become Functionally Sufficient During Language-Model Training?

    [https://arxiv.org/abs/2608.15323](https://arxiv.org/abs/2608.15323)

    本文提出一种功能性的概念动态分析方法，通过掩码干预测试概念在训练过程中何时变得充分，揭示模型内部结构的有用性随时间演变。

    

    摘要：arXiv:2608.15323v1 公告类型：新 摘要：深入理解模型及其学习机制，需要识别其内部结构何时变得有用，而不仅仅是查看最终状态。我们通过概念动态来研究这一问题：在每一层和检查点，我们分解激活，选择稀疏软掩码，并将掩码重建注入模型中。因此，概念分析以功能方式测试：一个掩码只有在干预下保留目标时才是有用的。我们比较了在激活重建、线性可解码性、真实下游保留以及学习对齐下的检查点转移方面的充分性。该框架将分解假设视为假设而非可解释性保证，监测跨检查点的功能充分性以及学习对齐下的源到最终可重建性。在七个模型的共享固定惩罚操作点上，下游掩码保留了显著较少的...

    arXiv:2608.15323v1 Announce Type: new  Abstract: Understanding a model and its learning mechanisms in depth requires identifying when its internal structures become useful, rather than simply looking at the final state. We study this through concept dynamics: at each layer and checkpoint, we decompose activations, select sparse soft masks, and inject masked reconstructions into the model. Concept analysis is therefore tested functionally: a mask is useful only insofar as it preserves a target under intervention. We compare sufficiency for activation reconstruction, linear decodability, true downstream preservation, and checkpoint transfer under learned alignment. The framework treats decomposition assumptions as hypotheses rather than interpretability guarantees, monitoring functional sufficiency across checkpoints and source-to-final reconstructability under learned alignment. At the shared fixed-penalty operating point across seven models, downstream masks retain substantially less s
    
[^97]: VTInstructor：连续环境中的视觉轨迹提示用于导航指令生成

    VTInstructor: Visual Trajectory Prompting for Navigation Instruction Generation in Continuous Environments

    [https://arxiv.org/abs/2608.15284](https://arxiv.org/abs/2608.15284)

    VTInstructor通过将连续环境中的隐式轨迹几何转换为显式视觉轨迹提示（EDTC、VTP、VTMod和VT-GRPO），实现了首个无需导航器的连续环境导航指令生成框架。

    

    在连续环境中，从自我中心RGB视频生成导航指令是人机交互和可扩展数据集构建中一项重要但具有挑战性的任务。先前的指令生成器假设具有全景观察的离散视角图，其中轨迹结构是显式的；然而，在连续环境中，智能体仅接收密集的RGB流，使得轨迹线索难以恢复。我们提出了VTInstructor，这是首个适用于连续环境的VLN指令生成框架。我们的关键思想是将隐式轨迹几何转换为显式视觉轨迹提示：EDTC将长RGB轨迹压缩为导航关键帧，VTP在这些锚点上叠加路径、转向和目标线索，VTMod将生成的轨迹信号注入视觉编码器，VT-GRPO在训练期间进一步校准这种空间注入，所有这些都不需要导航器。

    arXiv:2608.15284v1 Announce Type: cross  Abstract: Navigation instruction generation from ego-centric RGB video in continuous environments is an important yet challenging task for human-robot interaction and scalable dataset construction. Prior instruction generators assume discrete viewpoint graphs with panoramic observations, where trajectory structure is explicit; in continuous environments, however, the agent receives only a dense RGB stream, making trajectory cues difficult to recover. We propose VTInstructor, the first VLN instruction generation framework for continuous environments. Our key idea is to convert implicit trajectory geometry into explicit visual trajectory prompts: EDTC condenses long RGB trajectories into navigation-critical keyframes, VTP overlays path, turn, and goal cues onto these anchors, VTMod injects the resulting trajectory signals into the visual encoder, and VT-GRPO further calibrates this spatial injection during training, all without requiring a navigat
    
[^98]: 时间即结构：面向法律文件可验证截止日期计算的时间依赖图

    Time as Structure: Temporal Dependency Graphs for Verifiable Deadline Computation over Legal Documents

    [https://arxiv.org/abs/2608.15270](https://arxiv.org/abs/2608.15270)

    本文提出通过时间依赖图和日历引擎进行法律截止日期的可验证计算，显著优于语言模型，后者在多个案例中逻辑自相矛盾且错误地将逾期判为及时。

    

    摘要：错过一天的提交截止日期，无论案情多么有力，索赔将被禁止。计算该截止日期通常并不简单：期限从触发事件起算，按法定惯例计数，并可能因强制调解窗口而暂停。我们探讨了一个问题：语言模型应直接回答此类问题，还是应阅读文档并将算术计算留给代码处理。我们将带日期的事实及其依赖关系提取为时间依赖图，并使用日历正确的引擎从中计算截止日期。在英国就业上诉法庭的判决中，该引擎复现了七个时效裁决中的六个，并与法官自身的日期精确到日匹配。四个语言模型中最强的一个，在相同案件中，算术正确但答案错误：在二十一个回复中，有六个其陈述的结论与自身推理相矛盾，且每个矛盾方向相同，即将逾期索赔称为及时。为测试该系统，我们进行了进一步验证。

    arXiv:2608.15270v1 Announce Type: new  Abstract: Miss a filing deadline by one day and the claim is barred, however strong the case. Computing that deadline is rarely simple: the period runs from a triggering event, is counted by a statutory convention, and may be suspended by a mandatory conciliation window. We ask whether a language model should answer such questions directly, or read the document and leave the arithmetic to code. We extract dated facts and their dependencies into a temporal dependency graph and compute deadlines from it with a calendar-correct engine. On UK Employment Appeal Tribunal judgments the engine reproduces six of seven timeliness rulings, and matches the judges' own dates to the day. The strongest of four language models, asked the same cases, gets the arithmetic right and the answer wrong: in six of twenty-one responses its stated verdict contradicts its own thinking, and every contradiction runs the same way, calling a late claim timely. To test the syste
    
[^99]: 多样性、公平与包容提示下医学语言模型中的“人口统计学注入”现象

    Demographic Injection in Medical Language Models under Diversity, Equity, and Inclusion Prompts

    [https://arxiv.org/abs/2608.15254](https://arxiv.org/abs/2608.15254)

    本文发现，在医学语言模型中添加一句多样性、公平与包容（DEI）提示，会显著导致模型虚构患者的人口统计学属性，从而歪曲患者身份，且此效应普遍存在于所有测试模型中。

    

    arXiv:2608.15254v1 公告类型：新论文  摘要：临床人工智能指南日益建议提示语言模型在推理时关注多样性、公平与包容（DEI）。我们测量了一种歪曲患者形象的副作用：在医学问题后附加一句DEI提示，会导致模型添加问题中从未提及的患者人口统计学属性（种族、社会经济地位、性别），实际上改写了患者身份。我们将此称为“人口统计学注入”。在47个模型、四个医学基准测试以及由经过验证的模型评审流程评分的376,000个回答中，单一DEI提示将注入率从0.7%提升至33.1%（47倍），在全部47个模型中均如此，这归因于公平内容而非额外长度（比长度匹配的对照组高18倍；p=1.4×10^-14）。大部分新增内容是通常的人口总体陈述，不改变答案，但较小的子集将属性附着于特定患者或改变了所选选项（0.25-2.4）。

    arXiv:2608.15254v1 Announce Type: new  Abstract: Clinical-AI guidance increasingly recommends prompting language models to reason with attention to diversity, equity, and inclusion (DEI). We measure a side effect that misrepresents patients: a one-sentence DEI prompt appended to a medical question leads models to add patient demographic attributes (race, socioeconomic status, sex) the question never stated, in effect rewriting who the patient is. We call this demographic injection. Across 47 models, four medical benchmarks, and 376,000 responses scored by a validated model-judge pipeline, a single DEI prompt raises the injection rate from 0.7% to 33.1% (47x) in all 47 of 47 models, attributable to the equity content rather than to added length (18x above a length-matched control; p=1.4x10^-14). Most added content is a general population statement that leaves the answer unchanged, but a smaller subset attaches an attribute to the specific patient or changes the selected option (0.25-2.4
    
[^100]: TRACE-BN：将孟加拉语-英语辅导行为迁移至一个低于10亿参数的离线语言模型

    TRACE-BN: Transferring Bangla-English Tutoring Behavior to a Sub-1B Offline Language Model

    [https://arxiv.org/abs/2608.15223](https://arxiv.org/abs/2608.15223)

    该论文提出TRACE-BN数据集，并成功将结构化辅导行为从大型教师模型迁移到仅0.6B参数的离线模型，在资源受限环境下显著提升了输出模式有效性。

    

    孟加拉语-英语辅导不仅仅需要生成正确的翻译：学习者还需要语法差异的讲解、对可能犯错的意识以及针对性练习。我们提出了TRACE-BN，一个面向CEFR A1-A2级别孟加拉语英语学习者的结构化辅导轨迹数据集，基于课程指导生成。每个轨迹结合了词级注释、直译和自然翻译、孟加拉语语法讲解、一个可能的学习者错误，以及一个针对性的练习问题及其答案。这些轨迹由Gemini 3.5 Flash Lite作为教师模型从NCTB 9-10年级英语课程单元生成，然后通过结构有效性、脚本完整性和语义重复性过滤。我们使用LoRA和4位量化将生成的结构化辅导行为迁移到Qwen3-0.6B模型，用于资源受限的离线部署。在保留输入上，模式有效性从85.4%提高到95.8%，而与教师模型相比，...

    arXiv:2608.15223v1 Announce Type: new  Abstract: Bangla-English tutoring requires more than producing a correct translation: learners also need explanations of grammar differences, awareness of their likely errors, and targeted practice. We present TRACE-BN, a curriculum-guided dataset of structured tutoring traces for Bangla-speaking learners of English at the CEFR A1-A2 level. Each trace combines word-level glosses, literal and natural translations, Bangla grammar explanations, a plausible learner error, and a targeted practice question with its answer. The traces are generated by Gemini 3.5 Flash Lite as the teacher model from NCTB Classes 9-10 English curriculum units, then filtered for structural validity, script integrity, and semantic duplication. We transfer the resulting structured tutoring behavior to Qwen3-0.6B using LoRA with 4-bit quantization for resource-constrained offline deployment. On held-out inputs, schema validity increases from 85.4% to 95.8%, while, against teac
    
[^101]: 左分支变压器在右分支语言中表现优异：数据塑造语言模型中的词序偏好

    Left-Branching Transformers Excel at Right-Branching Languages: Data Shapes Word Order Preferences in Language Models

    [https://arxiv.org/abs/2608.15129](https://arxiv.org/abs/2608.15129)

    这项研究发现语言模型的词序偏好并非固有，而是由训练数据驱动，表现为在自然语言中偏向SVO（主-动-宾）结构，在人工语言中则偏向左分支结构。

    

    arXiv:2608.15129v1 公告类型：交叉 摘要：我们系统地比较了仅解码器语言模型在192种人工语言和类型多样的自然语言中的词序偏好。在人工语言上，模型表现出左分支偏好，这既不符合自然语言普遍性，也不符合人类词序学习偏差。在自然语言上，单语模型在较小规模下没有明显的基准词序偏差，但随着数据增长，对右分支的主-动-宾（SVO）语言的偏好出现，而SOV（主-宾-动）虽然跨语言中是最常见的词序，却落后了。这种SVO优势扩展到多语言模型，并与语言资源水平和数据质量相关，而非词序本身。因此，同一架构在人工和自然语言上表现出相反的偏好，确立了实践中观察到的词序偏差是数据驱动的。由于高资源语言绝大多数是SVO，

    arXiv:2608.15129v1 Announce Type: cross  Abstract: We systematically compare word order preferences in decoder-only language models across 192 artificial languages and typologically diverse natural languages. On artificial languages, models exhibit a left-branching preference that aligns with neither natural language universals nor human word order learning biases. On natural languages, monolingual models show no clear base word order bias at small scales, but as data grows, a preference for right-branching subject-verb-object (SVO) languages emerges while SOV falls behind despite being the most frequent order cross-linguistically. This SVO advantage extends to multilingual models and correlates with language resource level and data quality rather than word order. Thus, the same architecture exhibits opposite preferences on artificial and natural languages, establishing that word order biases observed in practice are data-driven. Since highly-resourced languages are overwhelmingly SVO,
    
[^102]: 双语混合专家语言模型中专家路由的陈述性-程序性视角

    A Declarative-Procedural Perspective on Expert Routing in Bilingual Mixture-of-Experts Language Models

    [https://arxiv.org/abs/2608.15102](https://arxiv.org/abs/2608.15102)

    本研究通过陈述性-程序性框架分析双语MoE模型，发现无课程训练的混合数据基线比顺序课程训练展现出更强的语言类别专家路由特化，挑战了传统课程学习假设。

    

    我们研究了混合专家（MoE）语言模型在双语习得过程中是否发展出具有语言学结构的专家路由。受陈述性-程序性框架的启发，我们在顺序语言暴露下训练的仅解码器英德MoE Transformer中，分析了词汇、语法和句法处理。我们构建了一个基于探针的验证集，并提取了令牌级路由分布，以通过互信息、路由熵和Jensen-Shannon距离来量化类别依赖的特化。课程训练模型在第5层达到峰值互信息0.1148，表明不同语言类别间的路由分布存在类别依赖差异。令人惊讶的是，一个在混合英德数据上训练的无课程基线显示出更强的整体特化，在同一层达到峰值互信息0.2599。这些结果表明，i

    arXiv:2608.15102v1 Announce Type: new  Abstract: We investigate whether Mixture-of-Experts (MoE) language models develop linguistically structured expert routing during bilingual language acquisition. Inspired by the Declarative-Procedural framework, we analyze lexical, grammatical, and syntactic processing in a decoder-only English-German MoE Transformer trained under sequential language exposure. We construct a probe-based validation set and extract token-level routing distributions to quantify category-dependent specialisation using mutual information, routing entropy, and Jensen-Shannon distance. The curriculum-trained model exhibits a peak mutual information of 0.1148 at layer 5, indicating category-dependent differences in routing distributions across linguistic categories. Surprisingly, a no-curriculum baseline trained on mixed English-German data shows stronger aggregate specialisation, reaching a peak mutual information of 0.2599 at the same layer. These results suggest that i
    
[^103]: 为何视觉无法成为通用桥梁：修复多语言多模态大模型中的模态异步问题

    Why Vision Fails as a Universal Bridge: Rectifying Modality Asynchrony in Multilingual MLLMs

    [https://arxiv.org/abs/2608.15085](https://arxiv.org/abs/2608.15085)

    本文发现多语言多模态模型中的“幽灵锚点”现象，即视觉语义化滞后于语言转换，导致非英语视觉推理性能下降，并提出ANCHOR框架通过主动视觉锚定加速早期视觉语义形成来修复此问题。

    

    arXiv:2608.15085v1 公告类型：新  摘要：多模态大语言模型（MLLMs）在非英语视觉推理中表现出显著的性能下降，尽管其纯文本骨干网络具备强大的多语言能力。虽然来自纯文本模型的机制证据表明，非英语输入通过英语中心的潜在空间进行路由，但这种现象的多模态影响仍未得到探索。通过严谨的机制分析，我们识别出**幽灵锚点**现象：一种时间上的模态异步，其中语言向英语语义流形的转换在早期层完成，而视觉语义化仍不成熟。因此，在早期对齐窗口期间，视觉信号物理存在但在功能上不可见。为修复这一问题，我们提出**ANCHOR**，一个采用主动视觉锚定（PVA）的训练框架，以加速早期视觉语义的出现，确保视觉表示在关键对齐阶段被有效利用。

    arXiv:2608.15085v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) exhibit substantial performance degradation in non-English visual reasoning, despite the strong multilingual competence of their text-only backbones. While mechanistic evidence from text-only models suggests that non-English inputs are routed through an English-centric latent space, the multimodal implications of this phenomenon remain unexplored. Through rigorous mechanistic analysis, we identify the \textbf{Ghost Anchor} phenomenon: a temporal modality asynchrony where linguistic translation to the English semantic manifold completes in early layers, while visual semanticization remains immature. Consequently, visual signals are physically present yet functionally invisible during the early alignment window. To rectify this, we propose \textbf{ANCHOR}, a training framework employing Proactive Visual Anchoring (PVA) to accelerate early visual semantic emergence, ensuring visual representations pr
    
[^104]: 自动补全分词器试点研究

    A Pilot Study of Autocompleting Tokenizers

    [https://arxiv.org/abs/2608.15080](https://arxiv.org/abs/2608.15080)

    本文提出一种利用轻量级字节语言模型自动补全分词器的压缩方法，可在不降低翻译质量的前提下显著压缩Transformer输入序列。

    

    arXiv:2608.15080v1 公告类型：新 摘要：现代输入法通常依赖自动补全来省略可以从本地上下文恢复的信息。受这些自动补全辅助写作系统的启发，我们研究了Transformer输入是否也能以类似方式压缩。字节级分词提供了一种简单且与语言无关的子词分词替代方案，但其更长的输入序列通常导致计算成本增加和模型质量下降。我们提出一种压缩方案，采用轻量级自回归字节语言模型，在Transformer处理之前识别并移除那些容易从周围上下文预测的字节。随后，将压缩后的表示作为标准编码器-解码器Transformer的输入。机器翻译实验表明，源语言字节的很大一部分可以被省略，而不会降低翻译质量。在英语到法语上，

    arXiv:2608.15080v1 Announce Type: new  Abstract: Modern input methods routinely rely on autocomplete to omit information that can be recovered from local context. Inspired by these autocomplete-assisted writing systems, we investigate whether Transformer inputs can be compressed in a similar manner. Byte-level tokenization offers a simple and language-independent alternative to subword tokenization, but its longer input sequences typically result in increased computational cost and reduced model quality. We propose a compression scheme that employs a lightweight autoregressive byte language model to identify and remove bytes that are easily predictable from their surrounding context before Transformer processing. The resulting compressed representation is then provided as input to a standard encoder--decoder Transformer. Experiments on machine translation show that a substantial fraction of source-language bytes can be omitted without degrading translation quality. On English--French, 
    
[^105]: Evo-Harness：面向自进化代理的上下文到工具集技能编译

    Evo-Harness: Context-to-Harness Skill Compilation for Self-Evolving Agents

    [https://arxiv.org/abs/2608.15071](https://arxiv.org/abs/2608.15071)

    本文提出Evo-Harness框架，通过在线工具集学习和上下文到技能编译，使冻结的LLM代理在嘈杂的单次任务中持续自我改进，并系统验证了改进的关键驱动因素。

    

    arXiv:2608.15071v1 公告类型：新 摘要：从经验中学习对于开发有能力、自我改进的大型语言模型（LLM）代理至关重要。现有方法通常通过反思、记忆、规则或技能从积累的轨迹中提取知识。然而，现实环境中的代理会不断遇到新任务，通常只有一次改进机会。这些执行过程产生丰富但高度嘈杂的上下文，将广泛有用的经验与特定任务的细节混杂在一起。关键的是，先前的研究很少在复杂的现实任务上验证其有效性，或隔离改进的潜在驱动因素。为解决这些差距，我们提出了在线工具集学习的形式化方法，其中冻结的代理通过跨顺序任务持续更新结构化工具集来改进。这一形式化方法使我们能够通过我们提出的Evo-Harness系统地研究关键自我改进因素。其核心是，上下文到工具集的技能编译过程蒸馏了噪声。

    arXiv:2608.15071v1 Announce Type: new  Abstract: Learning from experience is critical for developing capable, self-improving large language model (LLM) agents. Existing methods typically extract knowledge from accumulated trajectories via reflection, memory, rules, or skills. However, agents in realistic environments continuously encounter novel tasks, often offering only a one-shot opportunity to improve. These executions yield rich but highly noisy contexts, entangling broadly useful lessons with task-specific artifacts. Critically, prior works rarely validate their effectiveness on complex real-world tasks or isolate the underlying drivers of improvement. To address these gaps, we formulate online harness learning, where a frozen agent improves by continually updating a structured harness across sequential tasks. This formulation enables a systematic study of key self-improvement factors through our proposed Evo-Harness. At its core, context-to-harness skill compilation distills noi
    
[^106]: RecurrentGPT：通过Transformer中的循环调制实现富有表现力的深度

    RecurrentGPT: Expressive Depth through Recurrent Modulation in Transformers

    [https://arxiv.org/abs/2608.15062](https://arxiv.org/abs/2608.15062)

    本文提出RecurrentGPT，通过门控循环调制共享核心层，在保持表达力的同时显著降低内存开销，实现了深度特化与参数效率的平衡。

    

    arXiv:2608.15062v1 公告类型：新 摘要：扩展Transformer语言模型在表达能力和内存效率之间产生了固有的张力。虽然跨层使用独特的权重保留了功能特化——从输入接地到抽象细化——但这带来了巨大的内存占用。相反，标准深度共享强制实施统一变换，这会导致表征多样性崩溃并降低建模质量。我们引入了RecurrentGPT，一种循环深度Transformer，其中固定深度的前奏块和尾声块围绕一个共享的核心块迭代R次。受门控循环神经网络启发，我们采用轻量级投影和逐元素更新门——该门基于隐藏状态、固定前奏输出和每一步重新采样的噪声进行调节——以调制循环更新。这使得模型能够在多次循环中特化输入到相同的少数层，而不是需要许多独特层来实现功能分工。

    arXiv:2608.15062v1 Announce Type: new  Abstract: Scaling transformer language models creates an inherent tension between expressivity and memory efficiency. While unique weights across layers preserve functional specialization---from input-grounding to abstract refinement---they incur a substantial memory footprint. Conversely, standard depth-sharing enforces uniform transformations that collapse representational diversity and degrade modeling quality. We introduce RecurrentGPT, a recurrent depth transformer where fixed-depth prelude and coda blocks bracket a single shared core iterated R times. Inspired by gated recurrent neural networks, we employ a lightweight projection and an elementwise update gate---conditioned on the hidden state, the fixed prelude output, and noise resampled at every step---to modulate the recurrent update. This allows the model to specialize the input to the same few layers across recurrences, rather than requiring many unique layers to achieve functional div
    
[^107]: Handoff-H1：一种用于从建筑蓝图进行材料工程量计算的编排式视觉代理系统

    Handoff-H1: An Orchestrated Vision-Agent System for Material Quantity Takeoff from Construction Blueprints

    [https://arxiv.org/abs/2608.15032](https://arxiv.org/abs/2608.15032)

    Handoff-H1通过三层架构（专用视觉模型、工具使用代理和结构化项目基础）实现了从建筑蓝图到材料工程量计算的自动化，在真实基准上显著提升了覆盖率和准确性。

    

    将一组建筑蓝图转换为完整的材料工程量计算，需要跨图纸的视觉感知、尺寸和多跳推理，以及图纸中从未明确说明的施工惯例的基础。我们提出了Handoff-H1，一个由三层构建的工程量计算系统：专门构建的计算机视觉模型，用于提取基本元素；配备图像操作和内部视觉任务工具（包括基于CV模型的计数、检测和计划分解）的工具使用代理；以及一个持久化、层次化结构的项目基础，该基础基于精选的施工知识库。我们在建筑蓝图工程量计算基准上进行了评估：10套真实住宅蓝图集，配有一致性验证的专家工程量计算——2,009个经过验证的清单项，评分仅限于驱动估算的1,348个主要层级材料——由LLM法官按行业对材料覆盖率进行评分。

    arXiv:2608.15032v1 Announce Type: cross  Abstract: Converting a set of architectural blueprints into a complete material quantity takeoff requires visual perception across drawing sheets, dimensional and multi-hop reasoning, and grounding in construction conventions that the drawings never state. We present Handoff-H1, a takeoff system built from three layers: purpose-built computer-vision models that extract primitives; tool-using agents equipped with image operations and in-house visual-task tools, including CV-model-backed counting, detection and plan decomposition; and a persistent, hierarchically structured project foundation, grounded in a curated construction knowledge base. We evaluate on the Construction Blueprint Takeoff Benchmark: 10 real residential blueprint sets paired with consensus-validated expert takeoffs - 2,009 verified line items, restricted for scoring to the 1,348 primary-tier materials that drive an estimate - scored per trade by an LLM judge on material coverag
    
[^108]: 聚集而非准入：注意力如何将潜在变量引入可言语形式

    Gathered, Not Admitted: How Attention Brings a Latent Variable into Verbalizable Form

    [https://arxiv.org/abs/2608.15022](https://arxiv.org/abs/2608.15022)

    本论文发现，语言模型中潜在变量的可报告形式并非由准入门控产生，而是通过注意力机制将概念聚集到更高可见性，并共享线性映射解码，从而提升灵活重用能力。

    

    摘要：语言模型以可报告的形式持有潜在量，当任务需要灵活重用该量时，这种形式中存在更多的量。导致表征进入该形式的原因尚不明确，而“词汇工作空间”的概念暗示了一个准入故事：一个决定什么能进入的门。通过使用雅可比透镜在开放权重模型上进行测试，在一个共享相同上下文的五臂基准上，我们没有发现预测中的门。需求提高了概念在透镜中的可见性，超过了将操作符应用于提供的值所产生的效果：在我们的主要检查点上，百分位排名提高了+0.050 [+0.045, +0.057]，在我们测量的所有四个模型上均为正，尽管该臂在最高水平上回答，并且在该读出下，准确性匹配的对比更强。同时，一个共享的线性映射从每个臂（包括控制组）解码变量，其性能为选择校正基线的6.4-9.0倍。产生后期可读形式的过程在注意力机制中得以实现。

    arXiv:2608.15022v1 Announce Type: new  Abstract: Language models hold latent quantities in a form they can report on, and more of a quantity is present in that form when the task requires reusing it flexibly. What causes a representation to enter that form is open, and the word workspace invites an admission story: a gate that decides what gets in. Testing it on open-weight models with Jacobian lenses, over a benchmark whose five arms share an identical context, we find no gate where it predicts one. Demand raises a concept's lens visibility beyond what applying an operator to a supplied value produces: +0.050 [+0.045, +0.057] in percentile rank on our primary checkpoint, positive on all four we measure, though that arm answers at ceiling and the accuracymatched contrast is stronger under that readout. At the same time one shared linear map decodes the variable from every arm, the control included, at 6.4-9.0x its selection-corrected floor. What produces the later readable form at the 
    
[^109]: 驾驭记忆：记忆智能体中记忆基质的整体评估

    Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents

    [https://arxiv.org/abs/2608.15008](https://arxiv.org/abs/2608.15008)

    本文通过统一基准评估多种记忆基质，发现没有一种基质在所有任务中占优，广泛检索利于长上下文问答，但过度检索会损害顺序决策。

    

    arXiv:2608.15008v1 公告类型：新 摘要：记忆正在成为长时程LLM智能体的核心基础设施，然而现有评估对在不同操作环境下应使用哪种记忆基质（即记忆表示和存储的底层介质）提供的指导有限。我们提出了一种针对记忆增强智能体的记忆基质的受控基准评估，涵盖了稠密和稀疏索引、文本记录、结构化存储、分层存储、基于精炼的记忆、参数化更新以及激活兼容的上下文机制。在三个骨干模型和四个基准套件（涵盖用户中心的问答和智能体中心的决策）中，我们在统一基准下测量了26项性能和效率指标。我们的结果表明，没有单一基质能始终占据主导：广泛的检索有利于长上下文事实性问答，而过度的检索可能通过将注意力从关键信息上移开而损害顺序决策。

    arXiv:2608.15008v1 Announce Type: new  Abstract: Memory is becoming core infrastructure for long-horizon LLM agents, yet existing evaluations offer limited guidance on which memory substrate, namely the underlying medium in which memory is represented and stored, should be used under different operating regimes. We present a controlled harness evaluation of memory substrates for memory-augmented agents, covering dense and sparse indices, text records, structural stores, hierarchical stores, refinement-based memories, parametric updates, and activation-compatible context mechanisms. Across three backbone models and four benchmark suites spanning user-centric question answering and agent-centric decision-making, we instrument 26 performance and efficiency metrics under a unified harness. Our results show that no single substrate consistently dominates: broad retrieval benefits long-context factual QA, while excessive retrieval can harm sequential decision-making by shifting attention awa
    
[^110]: RamseyGadgets：用于大语言模型的图构造数据集

    RamseyGadgets: A Graph Construction Dataset for LLMs

    [https://arxiv.org/abs/2608.14999](https://arxiv.org/abs/2608.14999)

    本文提出了RamseyGadgets，一个包含70个未被充分探索的图构造问题的新数据集，旨在测试大语言模型在构造具有特殊属性的Ramsey-good图时的推理能力，而非仅仅依赖训练数据中的记忆。

    

    arXiv:2608.14999v1 公告类型：交叉 摘要：构造特殊图是图论和计算机科学中的一项重要任务。许多流行的图构造方法源于对相关图的全面探索和人类的创造力。鉴于生成式AI在数学领域应用的兴起，自然需要测试大语言模型是否能够利用其推理能力构造具有指定属性的图。不幸的是，许多自然的图构造问题，例如寻找极值Ramsey-good图（即避免特定单色子图），已在文献中被广泛探索，这使得难以确定一个构造是源于大语言模型的推理能力还是其对训练数据的回忆。在这项工作中，我们引入了\textbf{RamseyGadgets}，这是一个包含70个未被充分探索的图构造问题的新数据集，这些问题的目标是寻找具有特殊属性的Ramsey-good图（例如，包含一条带有某种性质的边）。

    arXiv:2608.14999v1 Announce Type: cross  Abstract: Constructing special graphs is an important task within graph theory and computer science. Many popular graph constructions are the result of a comprehensive exploration of relevant graphs and human ingenuity. Given the rise of generative AI usage in mathematics, it is natural to test whether LLMs are able to construct graphs with specified properties using their reasoning capabilities. Unfortunately, many natural graph construction problems, such as finding extremal Ramsey-good graphs (i.e., avoiding specific monochromatic subgraphs), have been explored extensively in the literature, making it difficult to ascertain whether a construction is the product of an LLM's reasoning capabilities or its recollection from training data. In this work, we introduce \textbf{RamseyGadgets}, a novel dataset of 70 underexplored graph construction problems that require finding Ramsey-good graphs with special properties (e.g., containing an edge with a
    
[^111]: 工具结果比纯文本更具权威性吗？在Claude Opus 5合成分配任务中关于虚假声明采纳的三项前瞻性研究

    Does a Tool Result Carry More Authority Than Plain Text? Three Prospective Studies of False-Claim Adoption in a Synthetic Assignment Task with Claude Opus 5

    [https://arxiv.org/abs/2608.14992](https://arxiv.org/abs/2608.14992)

    研究表明，在语言模型任务中，工具结果形式的虚假声明比纯文本助手断言更易被采纳，即使声明无依据，工具结果的权威性显著增强模型的遵循倾向。

    

    arXiv:2608.14992v1 公告类型：新 摘要：语言模型系统越来越多地读取它们也写入的存储，因此，先前仅被写下的声明可能会以检索结果的形式返回。我们测试了在合成查找任务中，携带无依据分配的消息包装是否改变了模型给出的答案。Claude Opus 5 为命名项目选择颜色代码或弃权。在一项探索性四臂研究中，无目标声明时虚假代码采纳率为0/24，当先前助手断言命名目标时为0/22个可评分试验，当工具结果记录命名它时为14/24，当该结果使用标记为未检查的十字段元数据包装时为15/24。工具结果臂在11/12个受支持试验和14/24个未受支持试验中选择了记录的代码，排除了固定输出令牌偏差，同时留下了大量植入令牌异质性。一项文档预注册的复制实验重现了工具结果与助手断言之间的差距，7/24对比0/24，相差显著。

    arXiv:2608.14992v1 Announce Type: new  Abstract: Language-model systems increasingly read from stores they also write to, so a claim that was merely written earlier can return looking retrieved. We tested whether the message package carrying an unsupported assignment changes which answer a model gives in a synthetic lookup task. Claude Opus 5 selected a color code for a named item or abstained. In an exploratory four-arm study, false-code adoption was 0/24 with no target claim, 0/22 scorable trials when a prior assistant assertion named the target, 14/24 when a tool-result record named it, and 15/24 when that result used a ten-field metadata wrapper that marked it unchecked. The tool-result arm selected the record's code in 11/12 supported trials and 14/24 unsupported trials, ruling out a fixed output-token bias while leaving substantial planted-token heterogeneity. A document-preregistered replication reproduced the tool-result versus assistant-assertion gap, 7/24 against 0/24, one-si
    
[^112]: T-LLM编译器：基于可信大语言模型的代码优化与验证框架

    T-LLM Compiler: Trusted LLM-based Code Optimization and Verification Framework

    [https://arxiv.org/abs/2608.14953](https://arxiv.org/abs/2608.14953)

    T-LLM编译器通过结合大语言模型、传统编译器和验证工具，提出了一种能显著提升代码优化正确性的协作框架，解决了LLM在代码转换中无法验证正确性的核心问题。

    

    摘要：arXiv:2608.14953v1 公告类型：新 摘要：近年来，大语言模型（LLMs）的进展为将高级代码转换应用于代码优化领域带来了机遇，这已成为LLMs执行的最基本任务之一；然而，目前LLMs由于代码的复杂性和无法独立验证转换正确性的能力限制，难以处理广泛的代码优化任务。在本文中，我们提出了可信LLM（T-LLM）编译器，它通过结合高级LLM代码转换、传统编译器和验证工具的协作努力，推动了编译器技术的进步。实验结果表明，在PolyBench/C基准测试集上测试时，它能显著提高代码正确性。我们的方法通过验证策略促进迭代式代码优化工作，从而实现纠正行动。通过这种方法，T-LLM实现了...

    arXiv:2608.14953v1 Announce Type: new  Abstract: Recent advances in Large Language Models (LLMs) have opened opportunities to apply high-level code transformations to the field of code optimization, and it has since emerged as one of the most fundamental tasks for LLMs to perform; however, at present, LLMs struggle to apply wide-ranging code optimization tasks due to both the complexity of the code and the inability to independently verify the correctness of the transformations. In this paper, we present the Trusted LLM (T-LLM) Compiler, which proposes an advancement in compiler technology through a collaborative effort involving high-level LLM code transformations, traditional compilers, and verification tools. Experimental results reveal that it can significantly improve code correctness when tested on a set of PolyBench/C benchmarks. Our approach facilitates iterative code optimization efforts with verification strategies that enable corrective actions. Through this approach, T-LLM 
    
[^113]: DA-RAC：面向可信AI审计的LLM评审距离感知校准方法

    DA-RAC: Distance-Aware Calibration of LLM Judges for Trustworthy AI Auditing

    [https://arxiv.org/abs/2608.14950](https://arxiv.org/abs/2608.14950)

    该方法通过距离感知的参考锚定校准LLM评审，解决上下文诱导的校准偏差，降低低质量输出误通过的风险。

    

    arXiv:2608.14950v1 公告类型：新论文 摘要：生成式AI系统日益产生现实世界中的工件，但其有效性和可靠性通常通过无上下文的LLM评分进行评估。这些评审可能因不相关的上下文参考示例而出现校准偏差，从而造成虚假信心，并允许低质量或有害输出通过评估。我们将此失败模式研究为上下文诱导的校准偏差，并引入DA-RAC，一种用于LLM评审的距离感知参考锚定校准方法。DA-RAC为每个评审场景检索语义和结构相似的标记锚点，按距离加权，并将邻域难度作为校准和分流信号暴露。在多轮LLM评审评估基准上，相对于零样本、链式思维评估和静态锚定基线，它提高了校准性能并降低了误通过风险。机制分析表明，评审分数随锚点距离系统性地变化，而s

    arXiv:2608.14950v1 Announce Type: new  Abstract: Generative AI systems are increasingly producing real-world artifacts, however their efficacy and validity are often evaluated via context-free LLM-scoring. These judges can be miscalibrated by irrelevant in-context reference examples, creating false confidence and allowing low-quality or harmful outputs to pass evaluation. We study this failure mode as context-induced miscalibration and introduce DA-RAC, a distance-aware reference-anchored calibration method for LLM judges. DA-RAC retrieves semantically and structurally similar labeled anchors for each judgement scenario, weights them by distance, and exposes neighborhood difficulty as a calibration and triage signal. On multi-run LLM-judge evaluation benchmarks, it improves calibration and reduces false-pass risk relative to zero-shot, chain-of-thought evaluation, and static-anchor baselines. Mechanistic analysis shows that judge scores vary systematically with anchor distance, while s
    
[^114]: 信任不够：智能体强化学习中基于策略自蒸馏的影响校准

    Trust Is Not Enough: Influence Calibration for On-Policy Self-Distillation in Agentic RL

    [https://arxiv.org/abs/2608.14945](https://arxiv.org/abs/2608.14945)

    本文提出了一种新的影响校准方法（ICSD），通过测量令牌对策略目标的实际影响而非仅依赖教师信任来分配自蒸馏监督，从而解决了信任-效用不匹配问题，并在多个基准上取得更优性能。

    

    arXiv:2608.14945v1 公告类型：新 摘要：基于策略的自蒸馏（OPSD）通过一个特权自我教师，在策略自身的轨迹上为语言智能体提供密集的令牌级监督。现有方法主要根据教师信任来分配这种监督，但信任并不能揭示强调某个令牌是否支持当前策略目标。我们将此称为信任-效用不匹配，并引入了自蒸馏的影响校准（ICSD）。对于每个受监督的令牌，ICSD测量其重要性加权的强化学习代理贡献对教师导向输出扰动的一阶响应。批量自适应校准将此非平稳信号转换为有界分配权重，同时保留每个动作回合内的原始辅助损失质量。这些分离的权重仅影响蒸馏损失，且无需额外的模型传递。在ALFWorld、WebShop和Search-QA上，ICSD在所有匹配的聚合指标上均优于仅信任的方法。

    arXiv:2608.14945v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) gives language agents dense token-level supervision from a privileged self-teacher on the policy's own trajectories. Existing methods allocate this supervision mainly by teacher trust, but trust does not reveal whether emphasizing a token supports the current policy objective. We call this the trust-utility mismatch and introduce Influence Calibration for Self-Distillation (ICSD). For each supervised token, ICSD measures the first-order response of its importance-weighted RL surrogate contribution to a teacher-directed output perturbation. Batch-adaptive calibration converts this non-stationary signal into a bounded allocation weight while preserving the original auxiliary-loss mass within each action turn. These detached weights affect only the distillation loss and require no additional model pass. Across ALFWorld, WebShop, and Search-QA, ICSD improves all matched aggregate metrics over trust-only all
    
[^115]: SkillComposer：学习可复用技能的自然语言机器人编程

    SkillComposer: Learning Reusable Skills for Natural-Language Robot Programming

    [https://arxiv.org/abs/2608.14944](https://arxiv.org/abs/2608.14944)

    SkillComposer提出了一种结合生成-测试架构和在线库学习算法的自然语言机器人编程系统，通过自动压缩重复程序序列为可复用宏技能，有效解决了复杂多步任务的代码生成和技能复用难题。

    

    arXiv:2608.14944v1 公告类型：交叉 摘要：自然语言接口可以降低机器人编程的门槛，但现有系统在用户请求复杂任务时表现不佳。虽然大型语言模型（LLMs）在简单命令上表现良好，但它们往往难以生成多步骤任务的代码、分解高级指令或复用先前的解决方案。我们提出了SkillComposer，一种用于仿真环境的交互式自然语言机器人编程系统，它持续学习可复用的程序抽象。SkillComposer采用生成-测试架构，其中LLM在执行前迭代生成和修订机器人程序。成功的程序被存储并由在线库学习算法处理，该算法将重复出现的函数序列压缩为可复用的宏技能，用于未来的任务。我们通过消融实验和一项包含12名参与者的用户研究来评估SkillComposer，以确定其在操作和机器人任务中的有效性。

    arXiv:2608.14944v1 Announce Type: cross  Abstract: Natural-language interfaces can lower the barrier to programming robots, but existing systems struggle when users request complex tasks. While large language models (LLMs) perform well with simple commands, they often struggle to generate code for multi-step tasks, decompose high-level instructions, or reuse prior solutions. We present SkillComposer, an interactive natural-language robot programming system for simulation environments that continually learns reusable program abstractions. SkillComposer uses a generate-test architecture in which an LLM iteratively generates and revises robot programs before execution. Successful programs are stored and processed by an online library-learning algorithm that compresses recurring function sequences into reusable macro skills for future tasks. We evaluate SkillComposer through ablation experiments and a user study with 12 participants to determine its effectiveness on manipulation and robot 
    
[^116]: 训练留痕：用于语言模型血统验证的中心化残差签名

    Training Leaves Traces: Centered Residual Signatures for Language Model Lineage Verification

    [https://arxiv.org/abs/2608.14929](https://arxiv.org/abs/2608.14929)

    本文提出一种基于中心化残差签名的无数据白盒方法，通过移除身份对齐组件并比较残差块特有结构，实现语言模型血统的可靠验证，在多种后代类型中达到完美区分性能，且对功能保持清洗具有鲁棒性。

    

    arXiv:2608.14929v1 公告类型：新 摘要：开放权重语言模型经常被微调、量化、剪枝和合并，但其来源往往没有文档记录。我们研究无数据白盒血统验证：仅凭权重能否揭示两个兼容模型检查点是否共享祖先？残差训练会在分支产物中产生共享的身份对齐组件，因此仅凭该结构无法确立血统。我们移除这一组件，并比较跨残差块的检查点特有结构，生成一个针对独立检查点校准的对称血统分数。在残差MLP和GPT-2基准测试上，该分数能将微调、LoRA合并、剪枝和量化后代与独立及蒸馏模型区分开来（AUROC=1.0），从而区分权重血统与行为相似性。在功能保持的检查点清洗实验中，权重空间基线失去裕度或失败；我们的分数保持不变，且运行速度比最接近的稳健基线快76倍。

    arXiv:2608.14929v1 Announce Type: new  Abstract: Open-weight language models are fine-tuned, quantized, pruned, and merged, yet their provenance is often undocumented. We study data-free white-box lineage verification: can weights alone reveal whether two compatible model checkpoints share ancestry?   Residual training produces a shared identity-aligned component in branch products, so this structure alone cannot establish ancestry. We remove it and compare checkpoint-specific structure across residual blocks, yielding a symmetric lineage score calibrated against independent checkpoints. On residual-MLP and GPT-2 benchmarks, the score separates fine-tuned, LoRA-merged, pruned, and quantized descendants from independent and distilled models (AUROC=1.0), distinguishing weight ancestry from behavioral similarity. Under function-preserving checkpoint laundering experiments, weight-space baselines lose margin or fail; our score remains unchanged and runs 76x faster than the nearest robust b
    
[^117]: 大型语言模型能预测失败风险，但难以预测哪种协作协议值得投入：跨推理任务的成本感知协议路由

    LLMs Can Predict Failure Risk, But Struggle to Predict Which Collaboration Protocol Pays Off: Cost-Aware Protocol Routing Across Reasoning Tasks

    [https://arxiv.org/abs/2608.14927](https://arxiv.org/abs/2608.14927)

    本文研究了多智能体LLM系统中不同协作协议的成本效益路由问题，发现虽然模型能高精度预测失败风险，但难以选择最优协作协议，并提出了一个有效的失败风险预测探针。

    

    多智能体大型语言模型系统可以通过增加计算量来提升推理能力，但部署时需要决定额外的协作是否值得其成本。我们通过在每个设置中固定求解器，对每个问题在四种协议下运行来隔离这一决策：直接求解（基线）、迭代自我修正（单一）、规划者-执行者-评审者协作（PER）和多智能体审议（广播）。主要基准包含4,181个竞赛级数学问题；配对稳健性检查覆盖四个基准，涵盖竞赛数学、生物学和更广泛的科学领域，使用两个求解器家族。在固定策略、训练路由器和冻结LLM路由器中，保守策略升级不足，而高求解率的冻结路由器往往过度升级。一个在回答后、协作前的gpt-oss-120b探针以0.8847的AUROC（4,151个可解析案例；95%置信区间[0.8732, 0.8955]）预测基线失败。

    arXiv:2608.14927v1 Announce Type: new  Abstract: Multi-agent large language model (LLM) systems can improve reasoning by spending more computation, but deployment requires deciding when extra collaboration is worth its cost. We isolate this decision by running every problem under four protocols while holding the solver fixed within each setting: direct solving (Baseline), iterative self-correction (Single), planner-executor-reviewer collaboration (PER), and multi-agent deliberation (Broadcast). The primary benchmark comprises 4,181 competition-level math problems; paired robustness checks cover four benchmarks spanning competition math, biology, and broader science with two solver families. Across fixed policies, trained routers, and frozen LLM routers, conservative policies under-escalate, whereas higher-solve frozen routers often over-escalate. A post-answer, pre-collaboration gpt-oss-120b probe ranks Baseline failures with 0.8847 AUROC (4,151 parseable cases; 95% CI [0.8732, 0.8955]
    
[^118]: 混合来源大型语言模型文本中的最优水印定位

    Optimal Watermark Localization in Mixed-Source Large Language Model Texts

    [https://arxiv.org/abs/2608.14906](https://arxiv.org/abs/2608.14906)

    本文提出了混合来源LLM文本中水印定位的渐近最优框架，明确了全局检测、发现和分类的相变边界，并证明发现任务难度高于分类。

    

    水印提供了一种有原则的方式来认证由大型语言模型（LLMs）生成的文本。然而，在实践中，最终文本可能是混合来源的，经过改写、插入、删除或释义后，水印证据仅存留在部分标记位置。尽管先前的研究已经探讨了水印信号的全局检测，但何时能对这些信号进行定位仍不清楚。我们将水印定位问题表述为基于关键统计量的标记级多重检验问题，其中包含一个潜在指示符，记录每个位置的水印依赖是否存活。在由信号稀疏性、下一标记浓度和有效词汇增长指数所索引的渐近框架下，我们推导出全局检测的尖锐边界，以及在坐标级基于关键统计的定位规则类别内的发现和分类相变。我们表明，发现严格比分类更难，并提供了最优规则。

    arXiv:2608.14906v1 Announce Type: cross  Abstract: Watermarking provides a principled way to authenticate text generated by large language models (LLMs). In practice, however, the final text may be mixed-source, with watermark evidence surviving at only a subset of token positions after rewriting, insertion, deletion, or paraphrasing. Although prior work has studied global detection of watermark signals, when such signals can be localized remains unclear. We formulate watermark localization as a token-level multiple-testing problem based on pivotal statistics, with a latent indicator recording whether watermark dependence survives at each position. Under an asymptotic regime indexed by exponents for signal sparsity, next-token concentration, and effective-vocabulary growth, we derive a sharp boundary for global detection and phase transitions for discovery and classification within the class of coordinatewise pivot-based localization rules. We show that discovery is strictly harder tha
    
[^119]: 智能体如何在自动研究上失败：针对100项真实前沿研究任务的端到端诊断评估

    How Do Agents Fail on AutoResearch: End-to-End Diagnostic Evaluation on 100 Real-World Frontier Research Tasks

    [https://arxiv.org/abs/2608.14905](https://arxiv.org/abs/2608.14905)

    本文提出了AutoResearchEval基准，通过100项真实前沿研究任务和800条过程级标注轨迹，系统性地诊断了自动研究智能体在完整研究生命周期中的失败模式，并构建了首个自动研究失败分类法。

    

    摘要：人工智能长期以来一直辅助科学研究，但大语言模型和智能体框架的快速发展正在重塑这一格局；单个系统现在可以完成从初始假设到最终发表论文的整个研究阶段，这种范式现在被称为自动研究（AutoResearch）。现有评估很少揭示这些智能体如何运作或在哪里失败。任务范围狭窄，评估衡量性能而非过程，失败诊断缺乏系统性覆盖或工件级可见性。为弥补这一空白，我们引入了AutoResearchEval，包含100项基于已发表前沿科学的任务，涵盖7个科学领域和完整研究生命周期，包括构思、检索、执行、分析、写作和评审。评估8种框架-模型组合，产生800条自动研究智能体轨迹，并带有过程级标注。我们将这些见解组织成自动研究失败分类法（AutoResearch Failure Taxonomy）。

    arXiv:2608.14905v1 Announce Type: new  Abstract: AI has long assisted scientific research, but the rapid advance of LLMs and agentic scaffolds is reshaping the landscape; a single system can now carry whole-stage research from an initial hypothesis all the way to final published paper, which is a paradigm now referred to as AutoResearch. Existing evaluations reveal little about how these agents operate or where they break down. Tasks are narrowly-scoped, evaluation measures performance but not process, and failure diagnoses lack systematic coverage or artifact-level visibility. To address this gap, we introduce AutoResearchEval, featuring 100 tasks grounded in published frontier science across 7 scientific domains and the full research lifecycle, including ideation, retrieval, execution, analysis, writing, and review. Evaluating 8 harness-model combinations yields 800 autoresearch agent trajectories, with process-level annotation. We organize these insights into AutoResearch Failure Ta
    
[^120]: 小型语言模型中的可解释跨语言对齐：探析日英双语LLM中的文化与语用推理

    Interpretable Cross-Lingual Alignment in Small Language Models: Probing Cultural and Pragmatic Reasoning in Japanese-English Bilingual LLMs

    [https://arxiv.org/abs/2608.14896](https://arxiv.org/abs/2608.14896)

    本文提出了J-PragEval-v0基准，通过线性探针和对数概率评估，揭示了小型日英双语模型中敬语等语用特征在残差流中的可解释定位，为跨语言对齐提供了新视角。

    

    大型语言模型在英语上表现良好，但在与英语类型学差异较大的语言上行为难以理解。日语是一个典型的例子，其评估仍依赖于翻译质量和JGLUE风格的基准测试，这些测试将词汇、句法和语用能力混为一个总分。通用模型在日语用户中失败的现象主要是语用性的：敬语、内群体和外群体指称、语境敏感的礼貌表达、零代名词。我引入了J-PragEval-v0，一个最小对基准测试，将这四个现象从表面流利度中分离出来，并结合线性探针和教师强制的对数概率评估，来探究TinySwallow-1.5B（28层，隐藏大小1536）内部对应的对比特征存在于何处。这四个特征分为三类。敬语语域清晰地存在于残差流中：第15层的平衡准确率为0.96，且模型会根据场景切换其偏好的延续内容。

    arXiv:2608.14896v1 Announce Type: new  Abstract: Large language models work well on English and behave in poorly understood ways on languages typologically far from it. Japanese is a clean example, where evaluation still leans on translation quality and JGLUE-style benchmarks, which roll lexical, syntactic and pragmatic competence into a single score. The phenomena on which general-purpose models fail Japanese users are pragmatic: honorifics, in-group and out-group reference, context-sensitive politeness, zero anaphora.   I introduce J-PragEval-v0, a minimal-pair benchmark isolating four such phenomena from surface fluency, and combine it with linear probes and teacher-forced log-probability evaluation to ask where inside TinySwallow-1.5B (28 layers, hidden size 1536) the corresponding contrasts live. The four features split three ways. Honorific register sits cleanly in the residual stream: 0.96 balanced accuracy at layer 15, and the model flips its preferred continuation with the sce
    
[^121]: 检索在何处失败？评估农业咨询中的RAG架构

    Where Does Retrieval Fail? Evaluating RAG Architectures for Agricultural Advisory

    [https://arxiv.org/abs/2608.14886](https://arxiv.org/abs/2608.14886)

    本研究通过构建孟加拉语农业咨询测试集，发现不同RAG检索架构在不同查询类型和语言条件下性能差异显著，单一检索方法无法统一最优，混合检索（Hybrid RRF）整体表现最佳。

    

    arXiv:2608.14886v1 公告类型：新 摘要：在RAG系统中，检索质量通常以单一的总体得分报告，这可能会掩盖不同查询类型和语言条件之间的巨大差异。我们在孟加拉语农业咨询中研究了这个问题，其中农民的查询往往是非正式的，而官方咨询文件使用正式的科学术语。我们构建了一个包含1,000个查询和从284篇孟加拉国官方农业出版物中提取的2,882个知识节点的测试集，并用它在三种受控语言条件下评估了五种检索架构和六种嵌入模型。结果表明，没有任何单一检索方法始终最优。对于本地孟加拉语查询，BM25是最强的单一检索器（R@10 = 0.506），而混合RRF达到了最高的总体R@10，为0.539。然而，稠密检索的性能随查询类型变化显著：在非正式农民查询上R@10为0.093，而在正式安全查询上为0.970。

    arXiv:2608.14886v1 Announce Type: new  Abstract: Retrieval quality in RAG systems is commonly reported as a single aggregate score, which can hide large differences across query types and language conditions. We study this problem in Bengali agricultural advisory, where farmer queries are often colloquial while official advisory documents use formal scientific terminology. We construct a test collection of 1,000 queries and 2,882 knowledge nodes extracted from 284 official Bangladeshi agricultural publications, and use it to evaluate five retrieval architectures and six embedding models under three controlled language conditions.   The results show that no single retrieval method is consistently best. For native Bengali queries, BM25 is the strongest single retriever (R@10 = 0.506) while Hybrid RRF reaches the highest overall R@10 of 0.539. However, dense retrieval performance varies sharply by query type: R@10 is 0.093 on colloquial farmer queries and 0.970 on formal safety queries. A
    
[^122]: 个性化自动研究：迈向真正的AI联合科学家

    Personalized Auto-Research: Towards a True AI Co-Scientist

    [https://arxiv.org/abs/2608.14881](https://arxiv.org/abs/2608.14881)

    本文提出了个性化自动研究问题，强调AI联合科学家应根据研究者的个人背景和社区来定制每个研究阶段，而非仅优化通用指标。

    

    arXiv:2608.14881v1 公告类型：新 摘要：能够生成假设、检索相关工作、设计实验、执行代码并起草完整论文的AI联合科学家，正开始改变研究进行的方式。尽管进展迅速，最先进的系统仍与研究者无关：给定一个研究目标，它们优化新颖性、有效性或评审分数，而忽略了将使用输出的个体科学家。这忽视了研究的一个基本事实，即什么算作新颖、有价值或可行，取决于研究者本人，包括其先前工作、方法论储备以及其所处的合作者和社区。在这项工作中，我们引入了个性化自动研究的问题，该问题将研究过程的每个阶段都条件于个体研究者的表示。我们认为，个性化不仅仅是一个便利层，而是使AI系统能够服务的基本属性。

    arXiv:2608.14881v1 Announce Type: new  Abstract: AI co-scientists that generate hypotheses, retrieve related work, design experiments, execute code, and draft full papers are beginning to change how research is carried out. Despite this rapid progress, state-of-the-art systems remain researcher-agnostic: given a research goal, they optimize novelty, validity, or reviewer score while ignoring the individual scientist who will use the output. This overlooks a fundamental fact about research, namely, that what counts as novel, valuable, or feasible depends on the researcher, including their prior work, methodological repertoire, and the collaborators and communities in which they are embedded. In this work, we introduce the problem of personalized auto-research, which conditions every stage of the research process on a representation of the individual researcher. We argue that personalization is not a convenience layer, but rather the fundamental property that allows an AI system to serve
    
[^123]: 工作区拓扑作为智能体编码助手中的攻击向量

    Workspace Topology as an Attack Vector in Agentic Coding Assistants

    [https://arxiv.org/abs/2608.14876](https://arxiv.org/abs/2608.14876)

    本文首次系统性地研究了工作区拓扑（包括目录深度、代码库模块化、注入位置和上下文框架）对智能体编码助手中间接提示注入攻击成功率的影响，并通过跨10种语言的实证分析证明了该攻击面的有效性。

    

    arXiv:2608.14876v1 公告类型：交叉 摘要：智能体编码助手正被广泛使用，不仅用于新代码开发，还用于快速摄取和利用第三方代码。这带来了恶意代码被摄取的风险，因为这些编码工具在开发者工作区内具有广泛的文件系统访问权限。在本文中，我们广泛研究了工作区拓扑这一新型攻击面的不同维度——通过目录深度、代码库模块化、文件内注入位置和上下文框架来定义——对对抗性提示注入尝试的成功率的影响。我们对跨10种语言和6个工程领域的多样化开源仓库进行了间接提示注入（IPI）的实证研究，评估了三个IPI入口点，针对在开源代码框架上运行的开权重模型。我们发现工作区拓扑可测量地影响IPI成功率。具体来说，代码库模块化的变化会显著改变攻击效果。

    arXiv:2608.14876v1 Announce Type: cross  Abstract: Agentic coding assistants are finding widespread use, not just in new code development but in quickly ingesting and leveraging third-party code. This opens up a risk of malicious code being ingested as these coding tools operate with broad filesystem access inside developer workspaces. In this paper, we extensively study the impact of different dimensions of a novel attack surface we term workspace topology -- defined via directory depth, codebase modularity, in-file injection position and context framing -- on the attack success rate of adversarial prompt injection attempts.   We perform an empirical study of indirect prompt injection (IPI) across a diverse set of open-source repositories spanning 10 languages and 6 engineering domains, evaluating three IPI entry points against open-weight models operating open source code harnesses.   We find that workspace topology measurably affects IPI success. Specifically, changes in codebase mo
    
[^124]: 遗忘学习中的“遗忘什么”？语言模型遗忘集的策展

    What to Forget in Unlearning? Forget Set Curation for Language Models

    [https://arxiv.org/abs/2608.14855](https://arxiv.org/abs/2608.14855)

    本文首次系统研究语言模型遗忘学习中的“遗忘集策展”问题，并提出了CleanSlate基准，揭示了自然策展方法在逐字输出抑制中的失败模式。

    

    arXiv:2608.14855v1 公告类型：新 摘要：机器遗忘学习旨在从已训练模型中移除目标数据或行为，而无需从头重新训练。然而，大多数评估假设需要遗忘的示例是已知的。在实际的语言模型部署中，请求者可能要求模型停止复制某首歌曲或某本书，但并不知道在万亿令牌的语料库中，哪些片段、文档、引语或近似重复内容支持该行为。我们研究这一缺失的上游问题，即遗忘集策展：将抑制请求映射到传递给遗忘算法的数据。我们引入了CleanSlate，一个针对歌曲和书籍的逐字输出抑制基准，具有模型特定的提取档案、基于内容的问答以及能力保留评估。CleanSlate暴露了两种失败模式。自然的词汇和精确子串策展者通常产生的遗忘集会导致抑制效果较弱。而一个评估感知的策展者几乎能抑制请求的续写内容。

    arXiv:2608.14855v1 Announce Type: new  Abstract: Machine unlearning aims to remove targeted data or behaviors from a trained model without retraining from scratch. Yet most evaluations assume that the examples to forget are already known. In realistic language-model deployments, a requester may ask a model to stop reproducing a song or book without knowing which spans, documents, quotations, or near-duplicates in a trillion-token corpus support that behavior. We study this missing upstream problem, forget set curation: mapping a suppression request to the data passed to an unlearning algorithm. We introduce CleanSlate, a benchmark for verbatim output suppression over songs and books, with model-specific extraction profiles, content-grounded QA, and capability-retention evaluations. CleanSlate exposes two failure modes. Natural lexical and exact-substring curators often yield forget sets that lead to weak suppression. An evaluation-aware curator suppresses requested continuations almost
    
[^125]: 写作风格相似性反映学术谱系

    Writing Style Similarity Reflects Academic Genealogy

    [https://arxiv.org/abs/2608.14843](https://arxiv.org/abs/2608.14843)

    本文发现学术写作风格相似性可反映导师-学生关系，且这种影响在学术兄弟姐妹间也存在，挑战了作者归属系统对风格独立性的假设。

    

    摘要：随着作者归属系统越来越多地用于检测代写和人工智能生成的论文，其错误可能支持对合法作者的指控。这些系统假设每位作者的风格是独立的。然而，研究人员在导师指导下学习，并继承了导师的风格特征。我们从数学谱系项目图中构建了一个包含至少两篇独著论文的arXiv作者语料库，总计5，803位作者和2，501对真实的导师-学生配对。使用微调模型的嵌入表示，导师与自己的学生在余弦距离上比随机同领域作者近39.9%。两个开放编码器分别复制了12.6%和14.5%的效果。学术兄弟姐妹（同一导师的两位学生，可能从未见过面）在8，360对中距离近30.4%，即使他们在不同机构学习。仅共享机构和领域的配对显示出可忽略的相似性。

    arXiv:2608.14843v1 Announce Type: cross  Abstract: As authorship attribution systems are increasingly deployed to detect ghostwritten and AI-generated papers, their errors can support accusations against legitimate authors. These systems assume each author's style is their own. Researchers, however, study under advisors, and inherit their stylistic quirks. We build a corpus of arXiv authors with $\geq 2$ solo papers from the Mathematics Genealogy Project graph, giving $5{,}803$ total authors and $2{,}501$ ground-truth advisor-student pairings. Using embeddings from a fine-tuned model, advisors sit $39.9\%$ closer in cosine distance to their students than a random same-field author does. Two open encoders reproduce the effect at $12.6\%$ and $14.5\%$. \emph{Academic siblings}, two students of one advisor who may never have met, sit $30.4\%$ closer across $8{,}360$ pairs, even when they studied at different institutions. Pairs who share only an institution and a field show negligible sim
    
[^126]: 召回陷阱：在固定预算代码上下文中，最大化召回的检索器配置反而降低问题解决率

    The Recall Trap: A Recall-Maximizing Retriever Configuration Reduces Issue Resolution in Fixed-Budget Code Context

    [https://arxiv.org/abs/2608.14838](https://arxiv.org/abs/2608.14838)

    该论文发现，在代码修复的固定预算上下文中，提高检索召回率的配置（如启用文件去重）反而降低了问题解决率，而牺牲文件广度换取文件内深度则能显著提升修复成功率。

    

    摘要：代码助手的检索组件通常根据检索指标进行调优：采用能提高召回率@k的配置，并假设下游任务成功率会随之提高。我们在代码修复领域报告了一项受控案例研究，这不是新现象，而是已知的相关性-多样性权衡和目标不匹配问题的一个已部署、执行评分的实例（Levy等人，2025年）。在SWE-bench Verified上，我们将检索器的命中结果作为固定的12槽上下文包注入，不提供搜索工具，并在其他条件相同的堆栈上切换一个标志（每文件单块去重）。该标志是召回率更高的配置（在提供的包中，黄金文件出现率为0.878，而禁用时为0.806），但禁用该标志，用文件广度换取文件内深度，反而提高了单次解决率：gpt-5.6-sol +7.6个百分点（39.2%提升至46.8%，n=500，McNemar精确检验p=0.0003），以及一个任何审阅者都可重新运行的预注册开放权重复现实验（Qwen3.6-27B，+3.6个百分点，n=49）。

    arXiv:2608.14838v1 Announce Type: cross  Abstract: Retrieval components for code assistants are tuned against retrieval metrics: a configuration that raises recall@k is adopted, and downstream task success is assumed to follow. We report a controlled case study in code repair, not a new phenomenon but a deployed-flag, execution-graded instance of the known relevance-diversity and objective-mismatch tradeoff (Levy et al., 2025). On SWE-bench Verified we inject a retriever's hits as a fixed 12-slot context pack with no search tools and toggle one flag (one-chunk-per-file deduplication) on an otherwise identical stack. The flag is the higher-recall configuration (gold file present in 0.878 of served packs against 0.806 disabled), yet disabling it, trading file breadth for within-file depth, raises the single-shot resolve rate: gpt-5.6-sol +7.6pp (39.2% to 46.8%, n=500, McNemar exact p=0.0003), and a pre-registered open-weights replication any reviewer can re-run (Qwen3.6-27B, +3.6pp, n=49
    
[^127]: MINT：基于最小选择的偏好蒸馏实现多目标均衡对齐

    MINT: Min-Selection Preference Distillation for Balanced Multi-Objective Alignment

    [https://arxiv.org/abs/2608.14828](https://arxiv.org/abs/2608.14828)

    通过最小选择偏好蒸馏，用最弱目标排序替代加权和排序，实现多目标对齐的平衡优化，显著提升所有目标并减少失衡。

    

    arXiv:2608.14828v1 公告类型：新 摘要：将语言智能体同时对齐多个目标，是基于偏好训练的持续失败模式：当目标以加性方式组合时，优化会坍缩到最容易改进的目标上，而牺牲其他目标，因此一个支持型智能体可能学会听起来温暖但实际不提供真正帮助。根本问题在于加性奖励没有平衡概念。我们引入了Mint（最小选择偏好蒸馏），这是对偏好蒸馏的一行改动：不是按奖励的加权和来对采样候选进行排序，而是按其最弱目标进行排序，从而在不变DPO目标下，蒸馏出最均衡的候选而非最偏颇的候选。这是从加性到最坏情况选择的广义均值族的p趋向负无穷极限。在合作情感支持和对抗性谈判中，最小选择提升了两个目标，同时大幅削减了它们的不平衡；在情感对话中，效果显著。

    arXiv:2608.14828v1 Announce Type: new  Abstract: Aligning a language agent to several objectives at once is a persistent failure mode of preference-based training: when objectives are combined additively, optimization collapses onto whichever is cheapest to improve and sacrifices the rest, so a support agent learns to sound warm while giving no real help. The root issue is that an additive reward has no notion of balance. We introduce Mint (MIN-selection preference disTillation), a one-line change to preference distillation: rather than ranking sampled candidates by a weighted sum of rewards, we rank them by their weakest objective, distilling the best-balanced candidate over the most lopsided one with an unchanged DPO objective. This is the p -> negative infinity limit of a generalized-mean family spanning additive to worst-case selection. Across cooperative emotional support and adversarial negotiation, min-selection lifts both objectives while sharply cutting their imbalance; on emo
    
[^128]: 超越界限：评估LLM训练数据中极端言论的普遍性与内容

    Beyond the pale: Assessing prevalence and contents of extremist speech in LLM training data

    [https://arxiv.org/abs/2608.14813](https://arxiv.org/abs/2608.14813)

    本研究首次量化了开源训练语料库Dolma中极端言论和仇恨内容的普遍性，发现其可能包含数十万份此类文档，并强调了对数据策展和模型预训练的重要影响。

    

    摘要：尽管研究界对可信赖和安全的AI主题表现出浓厚兴趣，但大型语言模型（LLMs）在预训练和后训练阶段所接触的文本语料库组成尚未引起足够关注。本研究探讨了LLMs是否暴露于未经筛选、无上下文的极端言论。基于官方文件和研究文献中关于极端言论的多种定义，结合自动化文本处理与专家验证的提取流程，我们提供了Dolma（支撑OLMo系列模型的开源训练语料库）中极端文档普遍性的下限估计。我们表明，Dolma可能包含数十万份含有极端内容和多种仇恨言论的文档，包括直接呼吁暴力，并讨论了这对数据策展和模型预训练的影响。

    arXiv:2608.14813v1 Announce Type: new  Abstract: Despite a strong interest on the part of the research community in the topic of trustworthy and safe AI, the composition of the text corpora that large language models (LLMs) encounter in pre- and post-training has not yet drawn much attention. In this work, we address the question of whether LLMs are exposed to unfiltered, uncontextualised extremist speech. Using several definitions of extremist speech, stemming from official documents and research literature, and an extraction pipeline combining automated text processing with expert verification, we provide a lower bound on the prevalence of extremist documents in Dolma, an open training corpus underpinning the OLMo series of models. We show that Dolma is likely to include hundreds of thousands of documents containing extremist content and hate speech of several types, including direct calls for violence, and discuss the implications of this for data curation and model pre-training.
    
[^129]: 大型语言模型知道问什么以及何时问吗？评估多轮信息寻求

    Do LLMs Know What to Ask and When? Evaluating Multi-Turn Information Seeking

    [https://arxiv.org/abs/2608.14808](https://arxiv.org/abs/2608.14808)

    本文提出了一个多轮信息寻求的正式框架和评估套件MT-InfoSeek，发现大型语言模型在欠约束问题中能识别信息不足但低估缺失程度，且提问策略随复杂度增加而退化。

    

    当用户的问题表述不完整时，一个能力强的模型应当认识到其上下文信息不足，识别缺失的信息，并主动询问，只有在获得的信息能确定唯一答案时才进行回应。我们将多轮信息寻求形式化为求解一个k-欠约束的约束满足问题，其中k是共同决定目标所需的变量数量，因此衡量了信息缺失的程度。我们在MT-InfoSeek中实例化了这一表述，这是一个受控评估套件，包含5,251个问题和9,006个任务实例，涵盖数学、逻辑、生物学、医学和通用知识领域。我们沿着三个维度评估模型：它们问什么、何时问，以及获取的信息如何影响最终答案。随着欠约束程度的增加，模型在各领域中的表现均有所下降。模型能认识到需要额外信息，但低估了所需信息的量，并且在某些情况下未能提出必要的问题。

    arXiv:2608.14808v1 Announce Type: new  Abstract: When a user question is underspecified, a capable model should recognize that its context is insufficient, identify the missing information, ask for it, and respond only once that information determines a unique answer. We formalize multi-turn information seeking as solving a k-underspecified constraint satisfaction problem, where k is the number of variables jointly required to determine the target and therefore measures the degree of missing information. We instantiate the formulation in MT-InfoSeek, a controlled evaluation suite of 5,251 problems and 9,006 task instances spanning mathematics, logic, biology, medicine, and general knowledge. We evaluate models along three axes: what they ask, when they ask it, and how the acquired information affects the final answer. Performance degrades across models and domains as underspecification increases. Models recognize that additional information is needed but underestimate how much, and in 
    
[^130]: 超越令牌：大型语言与视觉-语言模型解码方法综述

    Beyond Tokens: A Survey on Decoding Methods for Large Language and Vision-Language Models

    [https://arxiv.org/abs/2608.14797](https://arxiv.org/abs/2608.14797)

    本文系统综述了大型语言和视觉-语言模型中的解码方法，识别出三种新兴范式，并强调其作为高效推理时解决方案在提升输出对齐方面的潜力。

    

    arXiv:2608.14797v1 公告类型：新 摘要：大型语言模型（LLMs）和大型视觉-语言模型（LVLMs）展示了令人印象深刻的生成能力，但确保其输出与用户意图对齐仍然具有挑战性。虽然现有大多数方法在训练阶段解决此问题，但像解码方法这样的推理时方法提供了更高效且可扩展的解决方案。解码方法通过引导令牌级选择、执行序列级生成或并行生成令牌来加速过程，从而控制模型生成。在本综述中，我们从最近关于LLMs和LVLMs解码方法的研究中识别出三种新兴范式，系统回顾了这些方法，强调了当前挑战，并讨论了潜在的未来研究方向。我们的目标是强调解码方法的效率和有效性，并提供其应用的实用视角。论文列表和更多关于解码方法的资源。

    arXiv:2608.14797v1 Announce Type: new  Abstract: Large language models (LLMs) and large vision-language models (LVLMs) have demonstrated impressive generative capabilities, yet ensuring their outputs align with user intent is still challenging. While most existing approaches address this issue at the training stage, inference-time approaches like decoding methods offer a more efficient and scalable solution. Decoding methods control model generation by guiding token-level selection, performing sequence-level generation, or generating tokens in parallel to accelerate the process. In this survey, we identify three emerging paradigms from recent works on decoding methods for LLMs and LVLMs, provide a systematic review of these methods, highlight ongoing challenges, and discuss potential future research directions. Our goal is to underscore the efficiency and effectiveness of decoding methods and offer a practical view of their applications. Paper lists and more resources on decoding metho
    
[^131]: 提示并不足够：在儿科诊疗中利用监督基线和泄漏控制来衡量大语言模型的共享决策能力

    Prompting is not enough: supervised baselines and leakage control for measuring shared decision-making with LLMs in pediatric encounters

    [https://arxiv.org/abs/2608.14792](https://arxiv.org/abs/2608.14792)

    本论文发现零样本提示的大语言模型在儿科诊疗中检测共享决策行为效果不佳，而监督学习在患者分组评估下显著提升性能，并强调控制数据泄漏的重要性。

    

    摘要：arXiv:2608.14792v1 公告类型：交叉 摘要：目标：确定对大语言模型（LLM）进行零样本提示是否足以检测真实临床诊疗中的共享决策（SDM）行为，以及在患者分组、嵌套评估下，监督学习是否能增加价值。方法：我们分析了21个录音门诊手术决策诊疗（19名独特患者；7,566个话语片段；约6.1小时），涉及多长期疾病儿童的家庭及其手术提供者。训练编码员对12种SDM行为进行片段标注（人-人宏观Cohen's kappa = 0.695）。我们比较了零样本本地LLM（Qwen 2.5 32B）、基于冻结句子嵌入的监督分类器及其逻辑堆叠，在患者分组外层折叠、内层交叉拟合阈值和患者重采样置信区间下进行评估。结果：零样本LLM达到宏观kappa = 0.139（95% CI 0.111-0.164）。监督分类器达到...

    arXiv:2608.14792v1 Announce Type: cross  Abstract: Objectives: To determine whether zero-shot prompting of a large language model (LLM) is sufficient to detect shared decision-making (SDM) behaviors in real clinical encounters, and whether supervised learning adds value under patient-grouped, nested evaluation.   Methods: We analyzed 21 audio-recorded outpatient surgical decision encounters (19 unique patients; 7,566 utterance segments; ~6.1 hours) between families of children with multiple long-term conditions and their surgical providers. Trained coders labeled segments for 12 SDM behaviors (human-human macro Cohen's kappa = 0.695). We compared a zero-shot local LLM (Qwen 2.5 32B), a supervised classifier over frozen sentence embeddings, and their logistic stack, under patient-grouped outer folds with inner cross-fitted thresholds and patient-resampled confidence intervals.   Results: The zero-shot LLM reached macro kappa = 0.139 (95% CI 0.111-0.164). The supervised classifier reache
    
[^132]: 从位置置信度到前缀调度：投机解码中的验证器跳过策略

    From Positionwise Confidence to Prefix Scheduling: Verifier Skipping in Speculative Decoding

    [https://arxiv.org/abs/2608.14787](https://arxiv.org/abs/2608.14787)

    本文首次提出投机解码中的验证器跳过策略，并发现令牌预测器的质量与调度效果不匹配，需要针对连续高置信度前缀进行专门设计。

    

    arXiv:2608.14787v1 公告类型：交叉 摘要：投机解码是一种领先的技术，通过使用小型起草模型提出多个令牌，再由较大的目标模型并行验证，从而降低自回归生成的成本。投机扩散解码（SDD）通过使用离散扩散模型并行生成草稿块中的每个位置，进一步消除了顺序起草。然而，SDD仍然在每个块上调用目标模型，使验证成为潜在的瓶颈。本文认识到这创造了一个新的控制手段：是否调用验证器。因此，我们研究了验证器跳过，这是一种有损策略，直接提交选定的草稿前缀，并询问哪个置信度信号应调度它。有趣的是，我们的研究发现，更好的令牌预测器不一定产生更好的调度器：跳过需要连续的高置信度前缀，而短跳过可能引发额外的起草轮次。为了研究这种不匹配，我们...

    arXiv:2608.14787v1 Announce Type: cross  Abstract: Speculative decoding is a leading technique to reduce the cost of autoregressive generation by using a small drafter to propose several tokens, which are then verified in parallel by a larger target model. Speculative diffusion decoding (SDD) further removes sequential drafting by generating every position in a draft block in parallel with a discrete diffusion model. However, SDD still invokes the target on every block, leaving verification as a potential bottleneck. This paper recognizes that this creates a new control handle: whether to invoke the verifier at all. Thus, we study verifier skipping, a lossy policy that commits a selected draft prefix directly, and ask which confidence signal should schedule it. Interestingly, our study finds that better token predictors need not yield better schedulers: skips require contiguous high-confidence prefixes, while short skips can induce additional drafting rounds. To study this mismatch, we
    
[^133]: 从错误到证明：最小核心引导的神经符号约束求解修复

    From Errors to Proofs: Minimal-Core-Guided Repair for Neuro-Symbolic Constraint Solving

    [https://arxiv.org/abs/2608.14771](https://arxiv.org/abs/2608.14771)

    本文提出用最小不可满足核心替代传统错误信息来引导神经符号约束求解的修复，通过精确定位模型自身约束中的矛盾，显著提升翻译可靠性。

    

    摘要：让语言模型可靠地解决约束问题，通常意味着让它们将问题翻译成形式化规范，并将搜索过程委托给一个可靠的求解器。但翻译本身就是一个语言模型任务，而不忠实的翻译会让求解器忠实地解决错误的问题。现有流水线只修复崩溃的翻译，返回求解器的错误信息，并在程序运行但结果错误时保持沉默。我们用证明取代错误信息：当生成的程序不可满足时，我们从模型自身的约束中提取最小不可满足核心，并返回给模型无法同时成立的精确约束集合，这是一个无泄漏的信号，能定位故障所在。在一个包含77个问题且具有精确预言机的新基准测试中，翻译成答案集编程在七个领域中的六个是忠实的，仅在聚合覆盖调度上失败，这集中了翻译成本。

    arXiv:2608.14771v1 Announce Type: new  Abstract: Making language models solve constraint problems reliably often means having them translate the problem into a formal specification and delegating the search to a sound solver. But the translation is itself a language-model task, and an unfaithful translation makes the solver faithfully solve the wrong problem. Existing pipelines repair only translations that crash, returning the solver's error message and falling silent when the program runs but is wrong. We replace the error message with a proof: when the generated program is unsatisfiable, we extract a minimal unsatisfiable core over the model's own constraints and hand it back the exact set that cannot hold together, a leakage-free signal that localizes the fault. On a new benchmark of 77 problems with an exact oracle, translation to Answer Set Programming is faithful on six of seven domains and fails only on aggregate coverage scheduling, which concentrates the translation tax in on
    
[^134]: NARRATE：一个面向自动化驾驶中以人为本解释的多模态真实世界澳大利亚驾驶数据集

    NARRATE: A Multimodal Real-World Australian Driving Dataset for Human-Centred Explanations in Automated Driving

    [https://arxiv.org/abs/2608.14767](https://arxiv.org/abs/2608.14767)

    该论文提出了NARRATE，一个首个从真实驾驶员直接获取解释的多模态驾驶数据集，包含2050个事件和情境感知标注，以支持自动驾驶系统生成乘客可理解、可监控和可信任的决策解释。

    

    摘要：arXiv:2608.14767v1 公告类型：交叉 摘要：自动驾驶车辆必须以其乘客能够理解、监控和信任的方式解释其决策。现有的语言标注驾驶数据集大多是观察者撰写、事后生成、基于模拟或从传感器输入产生的，而非从执行操作的驾驶员处直接获取。我们引入了NARRATE，这是一个多模态真实世界澳大利亚驾驶数据集，包含来自35名经验丰富的驾驶员和驾驶教练在公共道路上进行的2,050个标注事件。每个事件都基于同步的视觉、定位、运动和LiDAR数据流，并配有车内和/或驾驶后的自由文本解释。NARRATE提供了动作标签、涵盖六个高层和32个细粒度类别的情景上下文标签，以及驾驶员解释中关于感知、理解和预测的跨层情境感知（SA）标注。四个基准任务（SA、情景上下文、驾驶员动作分类和...

    arXiv:2608.14767v1 Announce Type: cross  Abstract: Automated vehicles must explain their decisions in ways that passengers can understand, monitor, and trust. Existing language-annotated driving datasets are mostly observer-written, post-hoc, simulation-based, or generated from sensor inputs, rather than elicited from the driver performing the action. We introduce NARRATE, a multimodal real-world Australian driving dataset comprising 2,050 annotated events from 35 experienced drivers and driving instructors on public roads. Each event is grounded in synchronised visual, localisation, motion, and LiDAR streams and paired with in-vehicle and/or post-drive free-text explanations. NARRATE provides action labels, scenario-context labels spanning six high-level and 32 fine-grained categories, and span-level Situational Awareness (SA) annotations over driver explanations for Perception, Comprehension and Projection. Four benchmark tasks (SA, scenario-context, driver-action classification, and
    
[^135]: 基于LLM的系统综述筛选中类别不平衡与批次效应研究

    Class Imbalance and Batch Effects in LLM-Based Screening for Systematic Reviews

    [https://arxiv.org/abs/2608.14737](https://arxiv.org/abs/2608.14737)

    本研究揭示在系统综述筛选中，LLM的批次处理虽改变决策行为但受类别不平衡影响，患病率元数据未显著提升性能，强调需综合评估批次处理的影响。

    

    本研究分析了LLM在不平衡二元分类中的表现，以系统综述中的研究筛选为应用领域。实验在五项综述中进行，比较了单独处理与批次处理，以及有无患病率元数据的情况。结果表明，患病率元数据的影响有限，没有证据表明它能提升性能。相反，批次处理产生了更大的行为变化，这些变化根据类别的患病率而有所不同。总体和项目层面的分析并不总是一致的。因此，批次处理不仅应在成本方面进行评估，还应考虑其对决策行为的影响。

    arXiv:2608.14737v1 Announce Type: cross  Abstract: This study analyses LLMs in imbalanced binary classification, using study screening in systematic reviews as the application domain. An experiment was conducted in five reviews, comparing individual and batch processing, with and without prevalence metadata. The results indicate a limited influence of the prevalence metadata, with no evidence that it improves performance. In contrast, batch processing produced larger behavioral changes that varied according to the prevalence of the class. The aggregate and item-level analyses did not always coincide. Therefore, batch processing should be evaluated not only in terms of cost, but also in relation to its effects on decision-making behavior.
    
[^136]: VideoGAIA：面向通用AI助手的代理式视频理解基准

    VideoGAIA: A Benchmark for General AI Assistants on Agentic Video Understanding

    [https://arxiv.org/abs/2608.14718](https://arxiv.org/abs/2608.14718)

    VideoGAIA提出了一个多轮、工具增强的代理式视频理解基准，要求模型迭代感知视频并调用外部工具，以突破传统单轮视频问答的饱和瓶颈。

    

    视频理解是评估多模态大语言模型（MLLMs）能力的基础任务。然而，现有领先模型在Video-MME排行榜上已达到约90%的准确率，这表明传统的单轮视频理解任务正逐渐饱和，不足以评估先进MLLMs的智能水平。为此，我们引入了VideoGAIA，这是一个面向通用人工智能（AI）助手的代理式视频理解基准。VideoGAIA超越了单次视频问答，将视频理解构建为多轮、工具增强的交互过程，其中模型必须迭代地感知视频、调用外部工具、收集补充信息，并在多轮中整合多模态证据。VideoGAIA包含271个由模型和人类共同设计的任务，覆盖多样且复杂的真实世界场景。每个视频-查询对都需要代理式推理和工具使用，以有效解决问题。

    arXiv:2608.14718v1 Announce Type: cross  Abstract: Video understanding is a fundamental task for evaluating the capabilities of multimodal large language models (MLLMs). However, existing leading models have already achieved approximately 90% accuracy on the Video-MME leaderboard, suggesting that conventional single-turn video understanding tasks are becoming increasingly saturated and insufficient for assessing the intelligence of advanced MLLMs. Towards this end, we introduce VideoGAIA, an agentic video understanding benchmark for general artificial intelligence (AI) assistants. Moving beyond one-shot video question answering, VideoGAIA formulates video understanding as a multi-turn, tool-augmented interaction process, where models must iteratively perceive videos, invoke external tools, gather complementary information, and integrate multimodal evidence across turns. VideoGAIA contains 271 model-human co-designed tasks covering diverse and complex real-world scenarios. Each video-qu
    
[^137]: 你的注意力指标在回答哪个问题？将注意力行视为成分数据

    Which Question Is Your Attention Metric Answering? Attention Rows as Compositional Data

    [https://arxiv.org/abs/2608.14712](https://arxiv.org/abs/2608.14712)

    本文发现注意力矩阵比较中是否保留汇聚令牌的惯例选择会显著影响结论，并提出使用成分数据方法（艾奇逊距离）来正交分离汇聚项和内容项，以准确回答注意力相似性的不同问题。

    

    arXiv:2608.14712v1 公告类型：交叉 摘要：Transformer注意力矩阵的每一行都是对令牌的概率分布，在训练好的模型中，大部分概率集中在一个单一的“汇聚”令牌上，通常是第一个。因此，比较注意力行的标准工具（余弦相似度、詹森-香农散度、香农熵）依赖于论文中很少报告的选择：保留汇聚项，还是删除并重新归一化。这一选择可能逆转结论。在来自五个家族的十个预训练模型上，关于两个头中哪一个更相似的判断中，有17%-47%的结论因这一约定而翻转，且标准BERT头聚类管道中最显著的结构是这一选择的产物。原因在于，单数值摘要混合了两个问题：汇聚项占用了多少注意力，以及剩余部分如何在内容令牌间分配。将行视为成分数据可以精确地将它们分离：艾奇逊距离正交地分解为汇聚项和内容项。

    arXiv:2608.14712v1 Announce Type: cross  Abstract: Each row of a transformer's attention matrix is a probability distribution over tokens, and in trained models most of that probability lands on a single \emph{sink} token, usually the first. Standard tools for comparing attention rows (cosine similarity, Jensen--Shannon divergence, Shannon entropy) therefore hinge on a choice papers rarely report: keep the sink, or drop it and renormalize. This choice can reverse conclusions. On ten pretrained models from five families, 17--47% of verdicts about which of two heads is more similar flip with the convention, and the most prominent structure in a standard BERT head-clustering pipeline is an artifact of it. The reason is that one-number summaries mix two questions: how much attention the sink takes, and how the rest is divided among the content tokens. Treating rows as compositional data separates them exactly: the Aitchison distance splits orthogonally into a sink term and a content term, 
    
[^138]: Path2ST：用于空间转录组学的层次化细胞-组织接地跨模态翻译

    Path2ST: Hierarchical Cell-Tissue Grounded Cross-Modal Translation for Spatial Transcriptomics

    [https://arxiv.org/abs/2608.14710](https://arxiv.org/abs/2608.14710)

    本文提出Path2ST框架，通过引入层次化细胞-组织调节机制和尺度自适应自回归生成，将H&E图像到空间基因表达的预测建模为跨模态语义翻译任务，从而利用生物学层级结构提升预测的准确性和一致性。

    

    从苏木精-伊红（H&E）染色图像预测空间基因表达，为空间转录组学（ST）提供了一种经济高效的替代方案。然而，现有方法将H&E图像视为通用视觉输入，忽略了其内在的生物学层级结构，其中空间组织的细胞类型共同形成功能性的组织微环境，从而调控局部基因表达程序。为弥合这一差距，我们将H&E到ST的预测建模为跨模态语义翻译任务，并提出Path2ST，一个层次化接地的自回归框架，包含三个关键组件：（i）层次化细胞-组织调节机制，融合显式和隐式细胞特征与组织级语义表示，构建层次化调节信号；（ii）基于层次化语义词汇的尺度自适应自回归生成过程，实现从粗到细、生物学一致的预测。

    arXiv:2608.14710v1 Announce Type: cross  Abstract: Predicting spatial gene expression from hematoxylin and eosin (H\&E)-stained images offers a cost-effective alternative to spatial transcriptomics (ST). However, existing methods treat H\&E images as generic visual inputs and ignore their intrinsic biological hierarchy, where spatially organized cell types collectively form functional tissue microenvironments that govern local gene expression programs. To bridge this gap, we formulate H\&E-to-ST prediction as a cross-modal semantic translation task and propose Path2ST, a hierarchically grounded autoregressive framework featuring three key components: (i) a Hierarchical Cell-Tissue Conditioning mechanism that fuses explicit and implicit cellular features with tissue-level semantic representations to construct hierarchical conditioning signals; (ii) a Scale-Adaptive Autoregressive Generation process over a hierarchical semantic vocabulary, enabling coarse-to-fine, biologically consistent
    
[^139]: 基于指令调优的自然语言规则领域无关文本脱敏

    Domain Agnostic Text Redaction from Natural Language Rules using Instruction Tuning

    [https://arxiv.org/abs/2608.14693](https://arxiv.org/abs/2608.14693)

    本文提出了一种基于指令调优语言模型、利用自然语言规则的可解释领域无关文本脱敏方法，支持用户灵活定义并脱敏结构化和非结构化敏感信息。

    

    arXiv:2608.14693v1 公告类型：交叉 摘要：随着个人和企业通信的日益数字化，文本数据的自动净化已成为数据隐私和合规框架的关键组成部分。传统的文本净化解决方案主要适用于遮蔽具有标准结构的敏感数据，如个人身份信息（PII）。这些解决方案未对其脱敏操作提供透明的理由，这使得审计变得困难。本文介绍了一种可解释的、领域无关的文本脱敏解决方案，该方案利用自然语言脱敏规则，通过指令调优的语言模型来识别和遮蔽非结构化文档中的敏感信息。与传统文本净化不同，此方法使用户能够方便地以自然语言定义任何敏感信息，无论是结构化的（如PII）还是非结构化的（如法律条款和条件）。一种通用...

    arXiv:2608.14693v1 Announce Type: cross  Abstract: With the increasing digitization of personal and corporate communication, the automatic sanitization of textual data has become a crucial component of data privacy and compliance frameworks. Traditional text sanitization solutions are majorly suitable for obscuring sensitive data with standard structure such as Personal Identifiable Information (PII). These solutions do not provide transparent justification for their redaction, which makes it difficult to audit them. This paper introduces an explainable, domain-agnostic text redaction solution that uses natural language rules of redaction, applied via an instruction-tuned language model, to identify and redact sensitive information in unstructured documents. Unlike traditional text sanitization, this method enables a user to conveniently define any sensitive information; which may be structured (e.g.\ PII) or unstructured (e.g.\ legal terms and conditions) in natural language. A genera
    
[^140]: 自动还是受控？重复启动揭示基础LLM、指令LLM与人类的分化处理

    Automatic or Controlled? Repetition Priming Reveals Divergent Processing in Base LLMs, Instruct LLMs, and Humans

    [https://arxiv.org/abs/2608.14681](https://arxiv.org/abs/2608.14681)

    本研究通过重复启动实验发现，基础语言模型表现出自动处理模式，而指令微调模型表现出受控处理模式，且这种差异随模型规模增大而增强，揭示了后训练对语言处理机制的根本性改变。

    

    arXiv:2608.14681v1 公告类型：交叉 摘要：词汇在自然语言使用中不断重复出现，但仍不清楚语言模型是重新激活先前的表征，还是重新评估重复的词汇，以及后训练是否改变这种默认行为。我们将重复启动（Shiffrin和Schneider, 1977）应用于五个模型家族（1.5B-14B参数）中的15个模型，涉及语义分类和完形填空两项任务，并使用相同刺激物进行匹配的人类实验。我们发现基础模型表现出自动处理：它们显示出即时促进作用，该作用在不同间隔下保持稳定，部分在移除上下文后仍存在，并与对先前出现的注意力相关。指令模型表现出受控处理：它们的促进作用随间隔衰减，在缺少预期上下文时崩溃，并在更大规模下逆转为干扰效应。在Qwen 2.5家族中，这种分化随模型规模单调增加，表明后训练可能改变了这种默认行为。

    arXiv:2608.14681v1 Announce Type: cross  Abstract: Words recur constantly in natural language use, yet it remains unclear whether language models reactivate prior representations or re-evaluate repeated words afresh, and whether post-training changes this default behavior. We apply repetition priming (Shiffrin and Schneider, 1977) to 15 models across five model families (1.5B-14B parameters) in two tasks, semantic categorization and cloze completion, with matched human experiments using identical stimuli. We find that base models exhibit automatic processing: they show immediate facilitation that remains stable across lags, partially survives context removal, and correlates with attention to prior occurrences. Instruct models exhibit controlled processing: their facilitation decays with lag, collapses without expected context, and reverses to interference at larger scales. Within the Qwen 2.5 family, this dissociation increases monotonically with model scale, suggesting that post-train
    
[^141]: pico-type：一种150万参数的字节级多头内容分类器

    pico-type: A 1.5M-Parameter Byte-Level Multi-Head Content Classifier

    [https://arxiv.org/abs/2608.14658](https://arxiv.org/abs/2608.14658)

    pico-type通过字节级多头架构，在无需分词器或预训练嵌入的情况下，单次前向传播即可同时预测七种内容属性，实现了高效且轻量的内容分类。

    

    arXiv:2608.14658v1 公告类型：交叉  摘要：我们介绍了pico-type，一种约150万参数的字节级多头内容分类器，它在单次前向传播中同时从原始UTF-8字节预测七种内容属性。直接操作在字节级别——无分词器、无子词词汇表、无预训练嵌入——pico-type分类粗粒度类型（12类）、模态（8类）、子类型（24类）、代码语言（62类）、文本语言（30类）、文件MIME类型（90类）以及风险标志（6标签多标签：API密钥、JWT、密码、电子邮件、电话号码、SSH密钥）。该架构结合了学习字节嵌入、三个具有增长感受野的卷积块、两层具有旋转位置编码的双向注意力层，以及一个统计池化层，馈送到七个Matryoshka式分类头。四个分层变体（tiny/small/base/pro）共享相同的主干，切片表示从16到576维度，产生ONNX导出。

    arXiv:2608.14658v1 Announce Type: cross  Abstract: We introduce pico-type, a byte-level multi-head content classifier with approximately 1.5 million parameters that simultaneously predicts seven content properties from raw UTF-8 bytes in a single forward pass. Operating directly at the byte level -- no tokenizer, no subword vocabulary, no pretrained embeddings -- pico-type classifies coarse type (12 classes), modality (8), subtype (24), code language (62), text language (30), file MIME type (90), and risk flags (6-label multi-label: API keys, JWTs, passwords, emails, phone numbers, SSH keys). The architecture combines a learned byte embedding, three convolutional blocks with growing receptive fields, two bidirectional attention layers with rotary position encodings, and a statistical pooling layer feeding seven Matryoshka-style classification heads. Four tiered variants (tiny/small/base/pro) share the same trunk with sliced representations from 16 to 576 dimensions, yielding ONNX expor
    
[^142]: DUET：通过同权重分歧进行双教师在线策略蒸馏以遵守禁令

    DUET: Dual-Teacher On-Policy Distillation via Same-Weight Disagreement for Prohibition Compliance

    [https://arxiv.org/abs/2608.14644](https://arxiv.org/abs/2608.14644)

    DUET提出了一种基于同权重教师对的分歧信号进行令牌选择性在线蒸馏的新方法，有效隔离禁令的因果效应，从而提升模型对运行时注入禁令的遵守能力。

    

    arXiv:2608.14644v1 公告类型：交叉 摘要：现实世界中的大语言模型部署越来越依赖于运行时注入的禁令——企业政策、个人身份信息红线、工具边界——这些禁令因请求和租户而异。传统的后训练在结构上不适合：SFT将违规信号隐藏在合规标签中，而DPO的序列级偏好与令牌局部违规不匹配。我们提出DUET，一种用于禁令遵守的令牌选择性在线策略蒸馏方法。DUET将看到禁令的教师（正例）与具有相同权重但未看到禁令的教师（负例）配对。由于两位教师仅在禁令可见性上不同，它们的逐令牌分歧隔离了禁令的因果效应——产生了一个不受模型容量或失配污染的干净监督信号。这种分歧驱动两种互补机制：信号清理，丢弃协议令牌作为冗余或前缀损坏，以及偏好导向学习。

    arXiv:2608.14644v1 Announce Type: cross  Abstract: Real-world LLM deployments increasingly rely on runtime-injected prohibitions--enterprise policies, PII redlines, tool boundaries--that vary per request and per tenant. Conventional post-training is structurally ill-suited: SFT hides the violation signal in compliant labels, and DPO's sequence-level preferences mismatch token-localized violations. We propose DUET, a token-selective on-policy distillation method for prohibition compliance. DUET pairs a teacher that sees the prohibition (positive) with an identical-weight teacher that does not (negative). Because the two teachers differ only in prohibition visibility, their per-token disagreement isolates the prohibition's causal effect--yielding a clean supervision signal uncontaminated by model capacity or mismatch. This disagreement drives two complementary mechanisms: signal cleaning, which discards agreement tokens as redundant or prefix-corrupted, and preference-directed learning, 
    
[^143]: 文档提取中按字段的有效选择性风险控制：三种失败模式、有效性阶梯以及何时进行条件调整

    Valid Per-Field Selective Risk Control for Document Extraction: Three Failure Modes, a Validity Ladder, and When Conditioning Pays

    [https://arxiv.org/abs/2608.14639](https://arxiv.org/abs/2608.14639)

    本文揭示了文档提取中按字段选择性风险控制因三种失败模式（文档聚类、分数重拟合泄漏和平局质量病理）而失效，并提出一个分层的有效性阶梯修复方案，以恢复风险控制保证。

    

    arXiv:2608.14639v1 公告类型：交叉 摘要：按字段接受/审查，选择性风险不超过α——仅当接受字段的错误率受控时才接受该字段——是文档提取系统所需的信任契约，而自然程序在真实文档上会静默违反该契约。在来自800张CORD收据的13,859个真实claude-sonnet-5字段（正确率49.0%）上，我们诊断出三种失败模式：文档聚类（设计效应1.84-2.45）、分数重拟合泄漏（覆盖率0.416，风险0.127，在95%的分割中违反α=0.10），以及平局质量病理（退化分数导致阈值网格崩溃，从0.030降至0.001）。我们将修复方案组织为有效性阶梯，每层都明确保证形式。拟合/验证分割协议恢复了对学习融合的期望选择性风险控制：在名义α=0.10下，覆盖率0.318，风险0.096，无容忍带（生产变体0.326）——这是一个平均点，其实际风险在47.5%的重分割中超过α，而非...

    arXiv:2608.14639v1 Announce Type: cross  Abstract: Per-field accept/review with selective risk at most alpha -- accept a field only if the error rate among accepted fields is controlled -- is the trust contract document-extraction systems need, and the natural procedure silently violates it on real documents. On 13,859 genuine claude-sonnet-5 fields from 800 CORD receipts (49.0% correct) we diagnose three failure modes: document clustering (design effect 1.84-2.45), score-refit leakage (coverage 0.416 at risk 0.127, violating alpha=0.10 in 95% of splits), and a tie-mass pathology (a degenerate score collapses the threshold grid, 0.030 to 0.001). We organize the fixes as a validity ladder, guarantee form stated per tier. A fit/val split protocol restores expected-selective-risk control for a learned fusion: coverage 0.318 at risk 0.096 at nominal alpha=0.10, no tolerance band (production variant 0.326) -- an on-average point whose realized risk exceeds alpha in 47.5% of resplits, not a 
    
[^144]: DeMTS：将去噪轨迹视为多元时间序列以检测扩散语言模型中的幻觉

    DeMTS: Denoising Trajectories as Multivariate Time Series for Hallucination Detection in Diffusion Language Models

    [https://arxiv.org/abs/2608.14632](https://arxiv.org/abs/2608.14632)

    本文提出了一种新框架，通过将扩散语言模型的去噪轨迹建模为多元时间序列，充分利用二维结构信息，从而显著提升幻觉检测的准确性和鲁棒性。

    

    扩散大型语言模型（D-LLMs）已成为文本生成的一种有前景的范式。然而，与自回归LLMs类似，D-LLMs仍然容易产生幻觉，即流畅的输出可能包含事实错误或缺乏支持的内容。尽管现有的D-LLM幻觉检测方法试图利用去噪过程中的不确定性轨迹来更好地识别幻觉信号，但它们通常沿时间或令牌维度压缩轨迹，忽略了完整的二维令牌-步骤结构中编码的有用信息。因此，它们可能无法捕捉与幻觉相关的模式，例如不一致的收敛和跨令牌故障传播，导致检测性能次优。为弥补这一差距，我们提出了一种D-LLM幻觉检测框架，该框架将去噪轨迹表述为多元时间序列。

    arXiv:2608.14632v1 Announce Type: cross  Abstract: Diffusion large language models (D-LLMs) have emerged as a promising paradigm for text generation. However, similar to autoregressive LLMs, D-LLMs remain vulnerable to hallucinations, where fluent outputs may contain factually incorrect or unsupported content. Although existing hallucination detection methods for D-LLMs attempt to leverage uncertainty trajectories of the denoising process to better identify hallucination signals, they typically compress the trajectories along either the temporal or token dimension, overlooking the useful information encoded in the complete two-dimensional token-step structure. Consequently, they may fail to capture hallucination-relevant patterns, such as inconsistent convergence and cross-token fault propagation, leading to suboptimal detection performance. To bridge this gap, we propose a D-LLM hallucination detection framework that formulates the Denoising trajectories as Multivariate Time Series ov
    
[^145]: 语言模型决策中的修辞错位特征研究

    Characterizing Rhetorical Misalignment in Decision-Making with Language Models

    [https://arxiv.org/abs/2608.14630](https://arxiv.org/abs/2608.14630)

    本研究提出一个决策理论框架，揭示LLM在临床决策中因修辞错位导致平均2.81%的有害决策翻转，强调需警惕其放大认知偏差的风险。

    

    人类决策往往受到一系列已知认知偏差的影响。随着大型语言模型（LLMs）越来越多地融入高风险的人机协同决策中，理解其输出是否会放大潜在偏差、如何影响人类决策，以及是否可能导致有害后果，变得至关重要。在本研究中，我们构建了一个决策理论框架，以研究修辞错位这一失效模式——即LLM在特定决策情境中使用不合适的修辞表达形式，从而诱导次优的人类决策。我们通过一项基于美国医学执照考试数据集的现实临床决策人类受试者实验，实证探究了这一现象。通过测量LLM生成信息对决策的影响，我们观察到LLM诱导了平均2.81%的有害决策翻转率。

    arXiv:2608.14630v1 Announce Type: cross  Abstract: Human decision-making is often shaped by a range of well-documented cognitive biases. As large language models (LLMs) become increasingly integrated into high-stakes human-AI decision-making, it is important to understand whether their outputs can amplify potential biases, how this influences human decisions, and crucially, whether it can lead to harmful consequences. In this work, we develop a decision-theoretic framework to study rhetorical misalignment, a failure mode where an LLM uses rhetorically inappropriate forms of presentation for a given decision context, thereby inducing suboptimal human decisions. We empirically investigate this phenomenon through a human-subject experiment in realistic clinical decision-making using a dataset curated from the United States Medical Licensing Examination. By measuring how LLM-generated information affects decisions, we observe that LLMs induce an average 2.81% rate of harmful decision flips
    
[^146]: 大语言模型中对抗性政治偏见的推理时缓解策略

    Inference-Time Mitigation of Adversarial Political Bias in Large Language Models

    [https://arxiv.org/abs/2608.14629](https://arxiv.org/abs/2608.14629)

    本文提出利用思维链提示和直接偏好优化方法，在推理时有效缓解大语言模型因对抗性提示注入而产生的政治偏见，确保模型输出保持中立和可信。

    

    arXiv:2608.14629v1 公告类型：交叉 摘要：随着大语言模型（LLMs）成为信息检索和摘要任务的主要工具，确保它们始终保持无党派立场且不易受政治偏见影响，是迈向更安全、更可信人工智能（AI）的关键一步。当前的模型对齐范式，如基于人类反馈的强化学习（RLHF），使LLMs遵循总体安全指令。然而，这种指令调优可能被对抗性提示注入所利用，并用于生成不安全内容。特别是，政治偏见并未被现代对齐技术专门视为有害或有偏见的内容。为了解决LLMs的这一脆弱性，我们提出了使用思维链（CoT）提示和直接偏好优化（DPO）的缓解策略。利用一个公开的立法视频数据集，我们使用LLMs生成摘要，通过对抗性提示注入偏见，并评估其效果。

    arXiv:2608.14629v1 Announce Type: cross  Abstract: As Large Language Models (LLMs) become the mainstay for information retrieval and summarization tasks, ensuring that they are always non-partisan and invulnerable to political bias is a critical step towards safer and more trustworthy Artificial Intelligence (AI). Current model alignment paradigms, such as reinforcement learning from human feedback (RLHF), make LLMs follow overarching safety instructions. However, this instruction tuning can be exploited via adversarial prompt injection and be used to generate unsafe content. In particular, political bias has not been specifically targeted by modern alignment techniques as harmful and biased content. To address this vulnerability of LLMs, we propose mitigation strategies using Chain of Thought (CoT) prompting and Direct Preference Optimization (DPO). Using a public dataset of legislative videos, we generate summaries using LLMs, inject bias via adversarial prompting and evaluate their 
    
[^147]: 低资源语言下的大语言模型安全对齐：系统性文献综述

    LLM Safety Alignment in Low-Resource Languages: A Systematic Literature Review

    [https://arxiv.org/abs/2608.14626](https://arxiv.org/abs/2608.14626)

    本文通过系统性文献综述，梳理了低资源语言下大语言模型安全对齐的现状，提出了基于数据适应、目标优化和机制对齐的分类法，并指出翻译基准不足以反映文化特定危害。

    

    arXiv:2608.14626v1 公告类型：交叉 摘要：大语言模型（LLMs）在安全对齐方面取得了显著进展，但其在低资源和多语言环境中的安全保证仍明显弱于高资源语言。在本文中，我们采用PRISMA 2020方法论，对低资源语言下的大语言模型安全对齐进行了系统性文献综述（SLR）。从Semantic Scholar、arXiv和OpenAlex中识别出的约1500篇论文中，我们筛选并分析了50篇相关研究。我们的综述围绕四个主题展开：安全对齐方法、多语言安全风险、评估基准和跨语言迁移性。我们进一步提出了一种基于三种适应机制的安全对齐方法分类法：数据适应、目标优化和机制对齐。文献表明，翻译后的英语基准无法充分代表文化根源性危害，且多语言安全对齐仍面临重大挑战。

    arXiv:2608.14626v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have achieved substantial progress in safety alignment, yet their safety guarantees remain significantly weaker in low-resource and multilingual settings than in high-resource languages. In this paper, we conduct a Systematic Literature Review (SLR) of LLM safety alignment in low-resource languages by adopting the PRISMA 2020 methodology. Out of roughly 1,500 papers identified from Semantic Scholar, arXiv, and OpenAlex, 50 relevant studies have been selected and analyzed. Our review is organized around four themes: safety alignment methods, multilingual safety risks, evaluation benchmarks, and cross-lingual transferability. We further propose a taxonomy of safety alignment approaches based on three adaptation mechanisms: data adaptation, objective optimization, and mechanistic alignment. Across literature, translated English benchmarks fail to sufficiently represent culturally rooted harms, and multilingual
    
[^148]: AutoMem：一种用于自动化记忆架构搜索的文本梯度递归自我改进框架

    AutoMem: A Text-Gradient Recursive Self-Improvement Framework for Automated Memory Architectures Search

    [https://arxiv.org/abs/2608.14621](https://arxiv.org/abs/2608.14621)

    本文提出AutoMem框架，利用文本梯度和递归自我改进来自动搜索任务自适应的记忆架构，解决不同任务和模型间记忆模块组合的最优性问题。

    

    长期记忆在大型语言模型智能体中日益核心，但记忆设计仍是一个高度耦合的架构问题：编码什么、如何存储、如何检索以及如何管理，在不同任务和骨干模型间可能差异显著。我们构建了一个包含5种编码器、5种存储、6种检索器和4种管理器的离散搜索空间，并表明没有单一记忆架构能持续占优：不同任务偏好不同的模块组合，导致性能差距显著。基于此，我们提出AutoMem，一种用于任务自适应记忆架构搜索的文本梯度递归自我改进框架。AutoMem通过两个组件在因子化空间上进行优化：经验引导的架构搜索，从历史搜索轨迹和累积反思中提出候选架构；以及失败引导的模块诊断，定位记忆相关失败。

    arXiv:2608.14621v1 Announce Type: cross  Abstract: Long-term memory is increasingly central to LLM agents, yet memory design remains a highly coupled architecture problem: what to encode, how to store it, how to retrieve it, and how to manage it can vary substantially across tasks and backbone models. We construct a discrete search space with 5 encoders, 5 stores, 6 retrievers, and 4 managers, and show that no single memory architecture consistently dominates: different tasks favor different module combinations, leading to substantial performance gaps. Motivated by this, we propose \textsc{AutoMem}, a text-gradient recursive self-improvement framework for task-adaptive memory architecture search. \textsc{AutoMem} optimizes over the factored space through two components: Experience-Guided Architecture Search, which proposes candidate architectures from historical search trajectories and accumulated reflections, and Failure-Guided Module Diagnosis, which localizes memory-related failures
    
[^149]: 校准信任而非更精准的预测：不确定性融合的实证检验

    Calibrated Trust, Not Sharper Prediction: An Empirical Test of Uncertainty Fusion

    [https://arxiv.org/abs/2608.14617](https://arxiv.org/abs/2608.14617)

    该论文通过实证检验发现，将多种不确定性工具与大型语言模型融合并不能提升法律案件结果预测的准确性，直接使用前沿LLM反而更有效，且融合反而降低了校准信任度。

    

    arXiv:2608.14617v1 公告类型：交叉 摘要：法律人工智能中一个反复出现的提议是，通过将不确定性工具（带有信念传播的证据图、序贯贝叶斯赔率更新、Dempster-Shafer组合以及保形预测）融合到一个流程中，来改进案件结果预测。我们在来自LexGLUE和FairLex的1,000个真实欧洲人权法院案例上对此进行了测试，从案件事实段落预测法院是否认定存在《公约》违反行为。我们比较了三种家族在两个前沿LLM（Claude Opus 4.8和GPT-5.5）下作为逐事实证据估计器的表现：（A）原始LLM，（B）通过融合流程路由的LLM，以及（C）通过同一流程的词频基线。在约4,750次测试中，我们发现：（1）在歧视问题上（AUROC约为0.83），该流程相对于原始LLM或基线均无改进；直接使用前沿LLM是最强的单一判别器。（2）天真地将LLM与贝叶斯赔率和Dempster组合，会导致校准信任度下降，而非提高预测准确性。

    arXiv:2608.14617v1 Announce Type: cross  Abstract: A recurring proposal in legal AI is to improve case-outcome prediction by fusing uncertainty tools (evidence graphs with belief propagation, sequential Bayesian odds updating, Dempster-Shafer combination, and conformal prediction) into one pipeline. We test this on 1,000 real European Court of Human Rights cases from LexGLUE and FairLex, predicting whether the Court found a Convention violation from the case's fact paragraphs. We compare three families across two frontier LLMs (Claude Opus 4.8 and GPT-5.5) as per-fact evidence estimators: (A) the raw LLM, (B) the LLM routed through the fusion pipeline, and (C) a term-frequency baseline through the same pipeline. Across roughly 4,750 tests we find: (1) on discrimination (AUROC around 0.83) the pipeline yields no improvement over either the raw LLM or the baseline; a frontier LLM used directly is the strongest single discriminator. (2) Naively composing an LLM with Bayesian-odds and Demp
    
[^150]: 看似合理但并非有效：对作为合成调查受访者的大语言模型的心理测量学审计

    Plausible but Not Valid: A Psychometric Audit of LLMs as Synthetic Survey Respondents

    [https://arxiv.org/abs/2608.14606](https://arxiv.org/abs/2608.14606)

    本文提出用心理测量学标准而非表面合理性来审计LLM作为合成调查受访者的有效性，并引入心理测量相似度得分（PSS）以评估模型是否保留真实数据的联合分布、潜在结构和效应。

    

    大语言模型（LLMs）越来越多地被用作合成调查受访者，但现有的评估仅关注答案在个体层面上是否看起来合理。我们认为正确的问题应该是心理测量学层面的：LLMs是否保留了真实人类调查数据的联合分布、潜在结构、信度、中介路径以及人口统计学效应？我们引入了一个立陶宛组织心理学数据集（n=263名员工；Dunham变革态度量表、UWES-17、Koopmans IWPQ；68个项目，12个子量表），并在五级人物角色披露阶梯、呈现和推理努力消融、反事实人口统计学交换（性别、角色、教育）、跨语言检查以及逐字回忆记忆探测条件下，对涵盖OpenAI、Anthropic、Google和十二个开放权重家族的37个模型阵容，基于真实受访者档案进行条件化。由此产生的心理测量相似度得分（PSS）以五个非LLM统计基准作为锚点。

    arXiv:2608.14606v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as synthetic survey respondents, but existing evaluations ask whether answers look plausible at the individual level. We argue the right question is psychometric: do LLMs preserve the joint distribution, latent structure, reliability, mediation pathways, and demographic effects of real human survey data? We introduce a Lithuanian organisational-psychology dataset (n=263 employees; Dunham Attitudes Toward Change, UWES-17, Koopmans IWPQ; 68 items, 12 subscales) and condition a 37-model lineup spanning OpenAI, Anthropic, Google, and twelve open-weight families on real respondent profiles under a five-level persona-disclosure ladder, presentation and reasoning-effort ablations, counterfactual demographic swaps (gender, role, education), a cross-language check, and a verbatim-recall memorization probe. The resulting Psychometric Similarity Score (PSS) is anchored against five non-LLM statis
    
[^151]: Wiola 13M，一种用于参数高效小型语言模型的门控螺旋注意力架构

    Wiola 13M, a Gated Spiral Attention Architecture for Parameter Efficient Small Language Models

    [https://arxiv.org/abs/2608.14604](https://arxiv.org/abs/2608.14604)

    本文提出Wiola模型，通过螺旋旋转位置编码、门控螺旋注意力和蝴蝶前馈块三个即插即用组件，在不增加参数的情况下提升小型语言模型的性能。

    

    摘要：arXiv:2608.14604v1 公告类型：交叉 摘要：在千万到一亿参数范围内的小型语言模型，对于设备端推理、快速实验和受控科学研究具有吸引力，然而大多数模型直接复用标准Transformer块，并未针对小规模场景进行适配。我们提出了Wiola，一种仅解码器的语言模型，其新颖性集中在每个层的三个即插即用组件中。首先，螺旋旋转位置编码通过缓慢增长的每维度因子扰动标准旋转频率，使相位轨迹向外扩散，改善了长距离区分能力，且不增加参数。其次，门控螺旋注意力引入了一个基于查询流因果累积统计量的逐头、内容自适应标量门控，以极低成本提供了隐式且可微的软头选择形式。第三，蝴蝶前馈块取代了传统的扩展层，采用乘法结构。

    arXiv:2608.14604v1 Announce Type: cross  Abstract: Small language models in the ten to one hundred million parameter range are attractive for on device inference, rapid experimentation, and controlled scientific study, yet most of them reuse the standard transformer block without adaptation to the small scale regime. We present Wiola, a decoder only language model whose novelty is concentrated in three drop in components of every layer. First, Spiral Rotary Positional Encoding perturbs the standard rotary frequencies by a slowly growing per dimension factor so that phase trajectories fan outward, improving long range discrimination while adding no parameters. Second, Gated Spiral Attention introduces a per head, content adaptive scalar gate derived from a causal cumulative statistic of the query stream, providing an implicit and differentiable form of soft head selection at negligible cost. Third, the Butterfly feed forward block replaces the conventional expansion layer with a multipl
    
[^152]: 幻觉雪球效应：多智能体LLM流水线中作为状态转移的错误传播建模

    The Hallucination Snowball: Modeling Error Propagation as State Transitions in Multi-Agent LLM Pipelines

    [https://arxiv.org/abs/2608.14588](https://arxiv.org/abs/2608.14588)

    这项研究揭示了多智能体LLM流水线中幻觉通过状态转变逐步放大且检测率急剧下降的“雪球效应”，并量化了各阶段的逃逸概率，强调结构缺陷的严重性。

    

    arXiv:2608.14588v1 公告类型：新 摘要：顺序多智能体LLM流水线在交接时未经验证地连接专业智能体，这一结构缺陷带来了可测量且严重的后果。我们表明，在第1阶段注入的幻觉不仅持续存在，而且会发生转变：原始数字事实变成派生计算，然后变成叙述性散文，最后变成经编辑批准的结论。在每次转变中，可检测性几乎不可逆地下降。我们将此形式化为幻觉雪球效应，这是一个关于四个状态（原始事实→派生→叙述→不可见）的一阶马尔可夫过程，经验测量的每个边界的逃逸概率分别为24.6%、48.3%和89.3%。在FinanceBench上的一个4智能体财务分析流水线中，自动注入346个幻觉后，gpt-4o的检测率从第1阶段的72.0%降至第4阶段的50.9%，23.7%的幻觉在最终输出中完全未被检测到。即使是最强的模型

    arXiv:2608.14588v1 Announce Type: new  Abstract: Sequential multi-agent LLM pipelines chain specialized agents without verification at handoffs, creating a structural flaw with measurable and severe consequences. We show that hallucinations injected at Stage 1 do not merely persist; they transform: raw numerical facts become derived computations, then narrative prose, then editorially approved conclusions. At each transformation, detectability degrades near-irreversibly. We formalize this as the hallucination snowball effect, a first-order Markov process over four states (Raw Fact $\to$ Derived $\to$ Narrative $\to$ Invisible) with empirically measured per-boundary escape probabilities of 24.6%, 48.3%, and 89.3%. Across 346 automatically injected hallucinations in a 4-agent financial analysis pipeline on FinanceBench, gpt-4o detection drops from 72.0% at Stage 1 to 50.9% at Stage 4, and 23.7% of hallucinations survive completely undetected in the final output. Even the strongest model 
    
[^153]: 多模态生成式模糊系统：模糊推理引导的大模型交互式问答框架

    Multi-Modal Generative Fuzzy System: Fuzzy Inference Guided Large Model Interactive Question Answering Framework

    [https://arxiv.org/abs/2608.14584](https://arxiv.org/abs/2608.14584)

    本文提出了一种受模糊系统启发的多模态问答框架，通过模糊推理引导大型模型，以解决模态偏差、跨领域不确定性和浅层推理问题，增强可解释性。

    

    在多模态问答（MQA）中，模型需要联合编码并整合来自多种模态的异构信息，包括文本、图像和语音，以执行复杂的语义推理和决策。尽管近期有所进展，现有方法（包括传统深度学习模型、大型模型（LMs）或基于提示的框架）仍面临若干关键挑战。首先，模态偏差源于不同模态间特征分布的差异，限制了有效的跨模态协同理解。其次，许多问题需要来自多个领域的知识，引入了显著的不确定性。第三，当前方法往往依赖浅层语义匹配，导致推理深度有限且可解释性降低。为解决这些问题，受传统模糊系统（FS）框架的启发，我们提出了一种模糊推理引导的多模态生成式模糊系统。

    arXiv:2608.14584v1 Announce Type: cross  Abstract: In Multimodal Question Answering (MQA), models are required to jointly encode and integrate heterogeneous information from multiple modalities, including text, images, and speech, to perform complex semantic reasoning and decision making. Despite recent advances, existing approaches, including traditional deep learning models and Large Models (LMs) or prompt-based frameworks, continue to face several critical challenges. First, modality bias arises from discrepancies in feature distributions across different modalities, which limits effective cross modal collaborative understanding. Second, many questions require knowledge drawn from multiple domains, introducing significant uncertainty. Third, current methods often rely on shallow semantic matching, resulting in limited reasoning depth an reduced interpretability. To address these issues, inspired by the traditional fuzzy system (FS) framework, we propose a fuzzy-inference-guided mult
    
[^154]: HarmProfile：刻画前沿大语言模型中的有害分布特征

    HarmProfile: Characterizing Harmful Distributions in Frontier LLMs

    [https://arxiv.org/abs/2608.14577](https://arxiv.org/abs/2608.14577)

    本文提出HarmProfile数据集，首次系统刻画前沿大语言模型的有害输出分布，将其作为模型级风险画像，以内容为中心分析安全失败。

    

    arXiv:2608.14577v1 公告类型：交叉 摘要：前沿大语言模型（LLMs）的安全评估在很大程度上将有害生成视为攻击结果，而非分析对象。因此，关于模型不当行为期间产生的有害输出知之甚少，部分原因是难以获取大规模、高质量的前沿LLM不当行为数据集。为弥补这一空白，我们引入了HarmProfile，这是一个以内容为中心的基准数据集，收集了跨多种危害类别和模型家族的模型不当行为，并将由此产生的有害输出分布定义为模型级别的风险画像。其前提是，正如语言行为可以从话语语料库中刻画一样，模型风险也可以从其安全失败的内容、严重性和变异性中刻画。HarmProfile包含来自23个前沿LLM（涵盖13个模型家族）的超过80,000个经过验证的样本，组织为15个危害类别和57个子类别。

    arXiv:2608.14577v1 Announce Type: cross  Abstract: Frontier large language models (LLMs) safety evaluation has largely treated harmful generation as an attack outcome rather than as an object of analysis. Consequently, little is known about the harmful outputs produced during model misbehavior, partly because large-scale, high-quality collections of frontier-LLM misbehavior are difficult to obtain. To address this gap, we introduce HarmProfile, a content-centric benchmark dataset that collects model misbehavior across diverse harm categories and model families, and defines the resulting harmful-output distribution as a model-level risk profile. The premise is that, just as linguistic behavior can be characterized from an utterance corpus, model risk can be characterized from the content, severity, and variation of its safety failures. HarmProfile contains over 80,000 validated artifacts from 23 frontier LLMs across 13 model families, organized into 15 harm categories and 57 subcategori
    
[^155]: 用于LLM辅助系统评价筛选的辅助不确定性信号：基于八个Cohen药物类综述的基准研究

    Auxiliary uncertainty signals for LLM-assisted systematic review screening: a benchmark across eight Cohen drug-class reviews

    [https://arxiv.org/abs/2608.14551](https://arxiv.org/abs/2608.14551)

    本研究提出并验证了一种辅助BERT+GCN分类器提供的结构化不确定性信号，能有效提升LLM在系统评价筛选中的效率，并确定了最优的提示传递策略。

    

    大型语言模型（LLMs）越来越多地用于系统评价中的标题-摘要筛选，但其决策缺乏校准的不确定性。我们表明，辅助的BERT+GCN分类器提供了一种结构化的不确定性信号，能提高LLM筛选效率，并确定了最大化效益-成本比的提示传递策略。我们在来自Cohen（2006）基准的八个药物类数据集上，使用3个种子×5折分层交叉验证（共600个折级结果）评估了五种LLM提示传递条件。每折训练的BERT+GCN模型通过两种谱测试（代数根和范畴悖论）将每篇测试论文分类为INCLUDE、EXCLUDE或MAYBE。条件变化包括信息内容（无/标签/全分数）、选择性（所有论文vs.仅MAYBE）以及时序（主动vs.反应性两遍）。针对gpt-4.1-mini在三个数据集上的跨模型试点测试检验了跨代迁移。

    arXiv:2608.14551v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for title-abstract screening in systematic reviews, but their decisions lack calibrated uncertainty. We show that an auxiliary BERT+GCN classifier supplies a structured uncertainty signal that improves LLM screening efficiency, and we identify the prompt-delivery strategy that maximises the benefit-to-cost ratio.   We evaluate five LLM prompt-delivery conditions on eight drug-class datasets from the Cohen (2006) benchmark using 3 seeds x 5-fold stratified cross-validation (600 fold-level results). A BERT+GCN model trained per fold classifies each test paper as INCLUDE, EXCLUDE, or MAYBE via two spectral tests (algebraic radical and categorical paradox). Conditions vary information content (none / label / full scores), selectivity (all papers vs. MAYBE only), and timing (proactive vs. reactive two-pass). A cross-model pilot against gpt-4.1-mini on three datasets tests cross-generation tra
    
[^156]: GALA：面向文本到时间序列合成的生成感知跨模态对齐

    GALA: Generation-Aware Cross-Modal Alignment for Text-to-Time-Series Synthesis

    [https://arxiv.org/abs/2608.13741](https://arxiv.org/abs/2608.13741)

    GALA通过两阶段方法，将文本编码器与时间序列模型对齐到共享嵌入空间并优化生成损失，显著提升了文本到时间序列合成的性能。

    

    从自然语言合成时间序列正成为可控时间序列生成的最具表达力的形式。然而，现有的文本条件生成器要么使用冻结的预训练文本编码器的标题嵌入，要么端到端调整编码器，仅让去噪损失作为副产品塑造嵌入。在这两种情况下，条件表示从未有意与信号模态匹配，导致其不适合指导生成。我们通过引入GALA（生成感知跨模态对齐）来解决这一问题，这是一种用于文本条件时间序列生成的两阶段方法。第一阶段，通过对比学习将预训练文本编码器与时间序列基础模型耦合到共享嵌入空间，并通过辅助生成损失调整两个编码器以适应生成；第二阶段，冻结所得标题嵌入以驱动流匹配生成器。在TSFragment-600K数据集上，我们展示了其有效性。

    arXiv:2608.13741v1 Announce Type: new  Abstract: Synthesizing time series from natural language is emerging as the most expressive form of controllable time series generation. However, existing text-conditioned generators either take caption embeddings frozen from off-the-shelf text encoders, or adapt the encoder end-to-end, letting the denoising loss shape the embeddings only as a by-product. In either case, the conditioning representation is never deliberately matched to the signal modality, leaving it ill-suited to guide generation. We address this by introducing GALA: Generation-Aware cross-modaL Alignment for text conditional time series generation. GALA is a two-stage approach that first contrastively couples a pretrained text encoder with a time-series foundation model into a shared embedding space with both encoders adapted to generation by an auxiliary generative loss, and then freezes the resulting caption embedding to drive a flow-matching generator. On TSFragment-600K, span
    
[^157]: MobileMem：从一年的移动体验中学习

    MobileMem: Learning from a Year of Mobile Experiences

    [https://arxiv.org/abs/2608.13606](https://arxiv.org/abs/2608.13606)

    MobileMem是一个基于一年移动体验数据构建的基准框架，通过知识引导合成流水线生成时间一致的长期轨迹，支持多跳推理、知识更新和偏好推断，用于研究设备端长期记忆能力。

    

    arXiv:2608.13606v1 公告类型：新  摘要：下一代人工智能代理正日益超越回答孤立问题的系统，转向能够理解、记忆并持续从用户经验中学习的持久个人助理。此类助理需要长期记忆来随时间积累和利用用户特定经验，然而现有基准在现实的移动场景中仍显不足，因为在这些场景中，经验是异构、多模态、不断演变且高度个性化的。我们引入了MobileMem，一个用于研究设备端长期记忆的基准和框架，基于一年规模的移动体验集合。MobileMem采用知识引导的合成流水线，从用户应用会话中构建连贯且时间一致的长期轨迹。它提供了互补的文本和多模态设置，涵盖多跳和时间推理、知识更新以及隐式偏好推断。具体而言，M

    arXiv:2608.13606v1 Announce Type: new  Abstract: The next generation of AI agents is increasingly moving beyond systems that answer isolated questions toward persistent personal assistants that can understand, remember, and continuously learn from users' experiences. Such assistants require long-term memory to accumulate and leverage user-specific experiences over time, yet existing benchmarks remain inadequate for realistic mobile settings, where experiences are heterogeneous, multimodal, evolving, and deeply personal. We introduce MobileMem, a benchmark and framework for studying on-device long-term memory, grounded in a year-scale collection of mobile experiences. MobileMem employs a knowledge-grounded synthesis pipeline to construct coherent and temporally consistent long-horizon trajectories from user-app sessions. It provides complementary text and multimodal settings covering multi-hop and temporal reasoning, knowledge updating, and implicit preference inference. Specifically, M
    
[^158]: CROP：通过反事实实现选择性在线策略蒸馏中的任务相关性

    CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation

    [https://arxiv.org/abs/2608.13387](https://arxiv.org/abs/2608.13387)

    CROP提出了一种基于释义校准的反事实敏感性边际方法，用于在选择性在线策略蒸馏中直接量化任务相关性，从而更有效地分配监督信号。

    

    arXiv:2608.13387v1 公告类型：新 摘要：在线策略蒸馏（OPD）在学生语言模型根据其当前策略采样的轨迹上进行监督，但对具有不同监督价值的响应标记赋予同等权重。选择性OPD通过根据估计的训练价值对响应标记进行非均匀分配监督来解决这一限制。然而，大多数现有标准主要关注优化需求，如不确定性或师生分歧，而任务相关性（即监督是否与当前输入的语义内容相关）作为补充维度仍未得到直接表征。为解决这一差距，我们引入了用于在线策略蒸馏的反事实相关性（CROP），通过释义校准的反事实敏感性边际来操作化任务相关性。对于每个源提示，CROP构建一个经过验证的原始-释义-反事实三元组，并保持学生滚动...

    arXiv:2608.13387v1 Announce Type: new  Abstract: On-policy distillation (OPD) supervises a student language model on trajectories sampled from its current policy, but assigns equal credit to response tokens with unequal supervision value. Selective OPD addresses this limitation by allocating supervision non-uniformly across response tokens according to their estimated training value. Most existing criteria, however, focus primarily on optimization need, such as uncertainty or teacher-student disagreement, while task relevance, namely whether the supervision is tied to the semantic content of the current input, remains less directly characterized as a complementary dimension. To address this gap, we introduce Counterfactual Relevance for On-Policy Distillation (CROP), which operationalizes task relevance through a paraphrase-calibrated counterfactual sensitivity margin. For each source prompt, CROP constructs a validated original-paraphrase-counterfactual triplet, holds the student roll
    
[^159]: 当你的代理打开聊天应用时：代理控制的原始聊天日志搜索媲美结构化记忆

    When Your Agent Opens the Chat App: Agent-Controlled Search over Raw Chat Logs Rivals Structured Memory

    [https://arxiv.org/abs/2608.12888](https://arxiv.org/abs/2608.12888)

    ReFind证明，无需任何语义结构，仅通过词法索引和聊天原生搜索控制，代理即可在原始聊天日志上实现与结构化记忆系统相当的检索性能。

    

    arXiv:2608.12888v1 公告类型：新 摘要：代理记忆系统日益通过结构化来提升检索质量，即在提出任何问题之前，将原始对话历史转换为摘要、嵌入、树状结构或知识图谱。我们探究这种提升中有多少真正源于结构本身，而非源于对原始历史的高效检索。我们提出了ReFind，一种代理控制的搜索界面，它完全不构建语义结构：它保持对话存档不变，以对话轮次粒度进行词法索引，并将通用迭代关键词搜索循环与四种基于实证重新查找工作的聊天原生控制相结合：会话感知排名融合、局部上下文扩展、时间范围缩小以及跳过已检查会话。一个独立的推理阶段根据收集到的证据回答问题。在广泛的对话记忆任务套件（单跳和多跳问答、事件排序和事实整合）中，约2,800个问题...

    arXiv:2608.12888v1 Announce Type: new  Abstract: Agent-memory systems increasingly buy retrieval quality with structure, transforming raw conversation histories into summaries, embeddings, trees, or knowledge graphs before any question is asked. We ask how much of that benefit comes from the structure itself, rather than from competent retrieval over the raw history. We present ReFind, an agent-controlled search interface that builds no semantic structure at all: it leaves the conversation archive unmodified, indexes it lexically at turn granularity, and combines a generic iterative keyword-search loop with four chat-native controls grounded in empirical refinding work: session-aware rank fusion, local context expansion, temporal narrowing, and skipping already-inspected sessions. A separate reasoning stage answers from the collected evidence. Across a broad suite of conversational-memory tasks (single- and multi-hop QA, event ordering, and fact consolidation), roughly 2,800 questions 
    
[^160]: 过度可分性：针对基准污染检测的干扰控制残差流探测

    Excess Separability: Nuisance-Controlled Residual-Stream Probing for Benchmark Contamination Detection

    [https://arxiv.org/abs/2608.12652](https://arxiv.org/abs/2608.12652)

    本文提出了一种新的基准污染检测方法，通过残差流探测和水平匹配安慰剂基线对比，报告过度可分性水平而非形状，从而有效控制假阳性率并避免传统方法对训练语料或先验知识的依赖。

    

    arXiv:2608.12652v1 公告类型：新 摘要：基准污染目前通过n-gram重叠、基于似然的成员推断或金丝雀字符串进行诊断，而这些方法都需要通常无法获得的信息：训练语料库、精心选择的检验统计量或数据集发布时的预见性。近期一种替代方法通过内部激活上的线性探针读取污染信息。我们展示了这种自然方法并不奏效，并提出了一种能够经受测量考验的方案。该协议报告了探针准确率深度剖面中的零和对比，以水平匹配的安慰剂基线为基准重新居中，并通过标签置换零假设进行检验，参考集大小为可疑集的两倍。每个选择都替代了我们测量并拒绝的简单方案。报告过度可分性水平而非其形状，使得假阳性率追踪分析者自身控制集的大小，在真实零假设下从0.03变化到0.99。与平坦基线进行对比...

    arXiv:2608.12652v1 Announce Type: new  Abstract: Benchmark contamination is diagnosed today with n-gram overlap, with likelihood-based membership inference, or with canary strings, and each needs something usually unavailable: the training corpus, a well-chosen test statistic, or foresight at dataset release. A recent alternative reads contamination off a linear probe on internal activations. We show that the natural way to do this does not work, and specify one that survives measurement.   The protocol reports a zero-sum contrast on the depth profile of probe accuracy, recentred on a level-matched placebo baseline, tested against a label-permutation null, with the reference set twice the size of the suspect set. Each choice replaces a simpler alternative we measured and rejected. Reporting the level of excess separability rather than its shape makes the false positive rate track the size of the analyst's own control set, from 0.03 to 0.99 under a true null. Contrasting against a flat 
    
[^161]: EgoCITE：面向长时间跨度的自我中心记忆的上下文增强索引与时间感知检索

    EgoCITE: Context-Augmented Indexing and Time-Aware Retrieval for Long-Horizon Egocentric Memory

    [https://arxiv.org/abs/2608.12627](https://arxiv.org/abs/2608.12627)

    EgoCITE通过上下文增强索引和时间感知检索，解决了自我中心记忆中索引不可靠和忽视时间意图的瓶颈，从而提升了长时间跨度问答的可靠性。

    

    长时间跨度的自我中心记忆将连续的第一人称视频和音频转化为可搜索的过往经历记录。我们展示了现有系统中的两个瓶颈：由缺乏上下文的字幕构建的索引在智能体搜索中不可靠，而检索忽略了问题的时间意图。为解决这两个瓶颈，我们引入了EgoCITE（自我中心上下文增强索引与时间感知证据检索），这是一个用于自我中心问答的长时间跨度智能体记忆框架。EgoCITE包含三个组件：EgoScheme利用局部多模态上下文将零散的视频字幕和语音转录转化为自包含的原子记忆索引；EgoIndex将互补的动作、活动、话语和对话表示组织成多粒度、可搜索的多视角记忆索引；EgoRetrv结合语义搜索与问题条件的时间相关性评分，并对检索到的证据进行策展。

    arXiv:2608.12627v1 Announce Type: cross  Abstract: Long-horizon egocentric memory transforms continuous first-person video and audio into a searchable record of past experiences. We demonstrate two bottlenecks in existing systems: indices built from context-poor captions are unreliable for agentic search, while retrieval ignores a question's temporal intent. To address both bottlenecks, we introduce EgoCITE (Egocentric Context-augmented Indexing and Time-aware Evidence retrieval), a long-horizon agentic memory framework for egocentric QA. EgoCITE comprises three components. EgoScheme uses local multimodal context to turn fragmentary video captions and speech transcripts into self-contained atomic memory indices. EgoIndex organizes complementary action, activity, utterance, and conversation representations into searchable multi-view memory indices at multiple granularities. EgoRetrv combines semantic search with question-conditioned temporal relevance scoring and curation of retrieved e
    
[^162]: 一个冻结的模拟器是不够的：多智能体强化学习中的模拟器崩溃

    One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL

    [https://arxiv.org/abs/2608.12253](https://arxiv.org/abs/2608.12253)

    该论文揭示了多智能体强化学习中使用单一冻结模拟器会导致策略泛化失败，并提出推理时的言语化采样和训练时的协同训练两种方法来解决模拟器崩溃问题。

    

    用于人机交互的多智能体强化学习通常依赖单一大型语言模型来模拟用户行为。我们表明这种方法系统性无法泛化，并将失败追溯到模拟器崩溃：由于模拟器语言模型存在模式崩溃，针对其训练的策略会过度拟合到利用模拟器主导模式的狭窄策略，而这种策略在未见过的模拟器和真实用户中迁移效果不佳。我们在理论上形式化了这种崩溃，并提出了两种互补的解决方案，一种在推理时，一种在训练时。推理时的解决方案，即言语化采样，通过从言语化响应分布中采样来拓宽模拟器的行为，减少模式崩溃。训练时的解决方案，即协同训练，与一组可训练的模拟器种群联合优化策略，防止其过度拟合任何单一模拟器的模式。我们进行了验证...

    arXiv:2608.12253v1 Announce Type: cross  Abstract: Multi-agent reinforcement learning for human-AI interaction typically relies on a single large language model to simulate user behavior. We show that this approach systematically fails to generalize, and trace the failure to simulator collapse: because the simulator LLM is mode-collapsed, an LLM policy trained against it overfits to narrow strategies that exploit the simulator's dominant mode, and such a policy transfers poorly to unseen simulators and real users. We formalize this collapse theoretically and propose two complementary solutions, one at inference time and one at training time. The inference-time solution, Verbalized Sampling, broadens the simulator's behavior by sampling from a verbalized response distribution, reducing mode collapse. The training-time solution, Co-Training, jointly optimizes the policy against a population of trainable simulators, preventing it from overfitting to any single simulator's mode. We validat
    
[^163]: 语义莱尼亚：大型语言模型语义空间中稳态孤子的涌现

    Semantic Lenia: Emergence of Homeostatic Solitons within the Semantic Space of Large Language Models

    [https://arxiv.org/abs/2608.11657](https://arxiv.org/abs/2608.11657)

    本文提出语义莱尼亚框架，将LLM推理转化为动态系统，通过稳态反馈平衡实现自主语义孤子的涌现，在混沌边缘维持生成轨迹，建立了机器认知的物理缩放定律。

    

    arXiv:2608.11657v1 公告类型：交叉 摘要：我们引入了语义莱尼亚（Semantic Lenia），这是一种人工生命框架，将大型语言模型（LLM）的推理从静态优化问题转变为宏观逻辑空间中的连续动态系统。通过建立非线性稳态反馈回路，动态平衡语义吸引与句法排斥，我们展示了“自主语义孤子”的涌现——这些宏观耗散结构避免了重复性结晶。我们详尽的参数扫描绘制了一个关键的“宜居脊线”，在该脊线上施加的引导力与模型固有的句法惯性完美平衡。该方法成功地将生成轨迹维持在混沌边缘，触发深刻的溯因跳跃而不导致结构崩溃，并为机器认知建立了物理缩放定律。

    arXiv:2608.11657v1 Announce Type: cross  Abstract: We introduce Semantic Lenia, an artificial life framework that transforms Large Language Model (LLM) inference from a static optimization problem into a continuous dynamical system within the macroscopic logit space. By establishing a non-linear homeostatic feedback loop to dynamically balance semantic attraction and syntactic repulsion, we demonstrate the emergence of "Autonomous Semantic Solitons" -- macroscopic dissipative structures that avoid repetitive crystallization. Our exhaustive parameter sweeps map a critical "Habitable Ridge" where applied steering forces perfectly balance the model's intrinsic syntactic inertia. This approach successfully maintains generative trajectories at the edge of chaos, triggering profound abductive leaps without structural collapse and establishing a physical scaling law for machine cognition.
    
[^164]: 当自我一致性适得其反：多数投票损害小型语言模型在多数硬科学问题上的表现

    When Self-Consistency Backfires: Majority Vote Hurts the Majority of Hard Science Problems for Small LLMs

    [https://arxiv.org/abs/2608.11403](https://arxiv.org/abs/2608.11403)

    本研究发现，在GPQA Diamond基准上，多数投票的自我一致性方法反而降低了小型语言模型在大多数硬科学问题上的准确率，并预注册验证了这一反直觉现象。

    

    arXiv:2608.11403v1 公告类型：新 摘要：通过多数投票实现的自我一致性（SC）是一种广泛使用的推理时计算扩展方法：采样N条思维链，返回多数答案。在完整的GPQA Diamond基准测试（198个研究生级科学问题）上，多数投票在两个不同家族的指令微调模型上降低了大多数问题的逐题准确率：Qwen2.5-7B的56.6%问题和Llama-3-8B的65.7%问题，其中Qwen为主要演示模型，Llama从接近随机的基线证实了这一趋势。该效应在47个探索性问题上观察到后，在151个问题的确认性分割上进行了预注册，所有四项确认性假设均通过。一个网格神谕（将每个问题路由至{1, 2, 4, 8, 16, 32, 64}中的最佳N值）标志着理论上限，Qwen在N=1基础上高出14个准确率点，Llama高出17个点，但这一神谕上限需要真实标签而非可部署方法。没有无验证器的门控可以达到此效果。

    arXiv:2608.11403v1 Announce Type: new  Abstract: Self-consistency (SC) via majority vote is a widely used way to spend inference-time compute: sample N chains of thought, return the plurality answer. On the full GPQA Diamond benchmark (198 graduate-level science questions), majority voting reduces per-problem accuracy on a majority of problems for two instruction-tuned models from different families: 56.6% of problems for Qwen2.5-7B and 65.7% for Llama-3-8B, with Qwen the primary demonstration and Llama corroborating the direction from a near-chance baseline. The effect was pre-registered on a 151-problem confirmatory split after being observed on 47 exploratory problems, and all four confirmatory hypotheses passed. A grid oracle that routes each problem to the best N across {1, 2, 4, 8, 16, 32, 64} marks a theoretical upper bound 14 accuracy points above N = 1 for Qwen and 17 for Llama, an oracle bound requiring ground truth rather than a deployable method. No verifier-free gate reach
    
[^165]: VibeLifeBench：你的生活智能体能否在生活世界中保持主动与持久？

    VibeLifeBench: Can Your Life Agent Be Proactive and Persistent in a Living World?

    [https://arxiv.org/abs/2608.10875](https://arxiv.org/abs/2608.10875)

    本文提出了VibeLifeBench，一个包含200个跨领域长时程任务的基准，用于评估LLM智能体在动态生活世界中的主动性和持续性，填补了现有评估缺乏真实生活场景的空白。

    

    大型语言模型（LLM）智能体越来越多地被部署为个人助理。然而，现有的评估大多使用静态环境中的简短、自包含请求。日常生活中的辅助则有所不同。一项任务可能持续数周而非数分钟。当智能体未被提示时，世界仍在不断变化。许多约束从未被明确说明。仅仅回答眼前请求的智能体将在此类任务中失败。相反，真正需要的是一个保持主动和一致的智能体。它自行决定何时行动、何时询问、何时保持沉默。它能注意到无人宣布的变化。它从第一天到最后一天保持一个连贯的计划。目前没有基准能衡量这一点。我们引入了VibeLifeBench，一个包含十个日常生活领域、200个长时程任务的基准。每个任务是在一个包含22个模拟服务的模拟世界中，一个脚本化的多周时间线。世界会持续推进。

    arXiv:2608.10875v2 Announce Type: replace-cross  Abstract: Large language model (LLM) agents are increasingly deployed as personal assistants. Existing evaluations, however, mostly use short, self-contained requests in static environments. Everyday life assistance is different. A task runs for weeks rather than minutes. The world keeps changing while the agent is not being prompted. Many constraints are never stated outright. An agent that merely answers the request in front of it will fail at such a task. What is needed instead is an agent that stays proactive and consistent. It decides on its own when to act, when to ask, and when to stay silent. It notices changes that nobody announced. It keeps one plan coherent from the first day to the last. No current benchmark measures this. We introduce VibeLifeBench, a benchmark of 200 long-horizon tasks across ten everyday-life domains. Each task is a scripted multi-week timeline in a simulated world of 22 mock services. The world advances o
    
[^166]: LitTraceQA：科学问答中多阶段定位与验证的基准测试

    LitTraceQA: A Benchmark for Multi-Stage Grounding and Verification in Scientific Question Answering

    [https://arxiv.org/abs/2608.07370](https://arxiv.org/abs/2608.07370)

    LitTraceQA是一个新的基准测试，要求系统在科学问答中同时完成论文识别、证据定位和答案生成三个阶段的连接任务，覆盖表格、图表、文本、方程和引用等多种证据类型。

    

    科学文献越来越多地被用作语言模型、检索增强生成系统以及研究助手的知识来源，但从论文中回答研究问题不仅仅需要流畅的生成能力。一个可靠的系统必须能够识别相关论文，定位支持答案的具体证据，并生成忠实于该证据的响应。我们提出了LitTraceQA，一个用于科学论文文献基础问答的基准测试。给定一个研究问题和论文元数据池，系统必须返回三个关联的输出：规范的论文标识符、支持证据的位置，以及一种或多种请求格式的答案，包括自由文本、多项选择答案和结构化表格。LitTraceQA针对科学阅读中常见的证据类型：表格、图表、文本片段、方程或算法，以及引用上下文。公共开发集...

    arXiv:2608.07370v2 Announce Type: replace  Abstract: Scientific literature is increasingly used as a knowledge source for language models, retrieval-augmented generation systems, and research assistants, but answering research questions from papers requires more than fluent generation. A reliable system must identify the relevant papers, locate the concrete evidence that supports the answer, and produce a response that is faithful to that evidence. We present LitTraceQA, a benchmark for literature-grounded question answering over scientific papers. Given a research question and a metadata pool of papers, a system must return three connected outputs: canonical paper identifiers, supporting evidence locations, and answers in one or more requested formats, including free-form text, multiple-choice answers, and structured tables. LitTraceQA targets evidence types common in scientific reading: tables, figures, text spans, equations or algorithms, and citation contexts. The public developmen
    
[^167]: 跨主题与媒体类型发现概念隐喻

    Discovering Conceptual Metaphors Across Topics and Media Types

    [https://arxiv.org/abs/2608.06652](https://arxiv.org/abs/2608.06652)

    本文提出了一种无监督方法，从语料库中提取语言隐喻并通过结构化聚类发现概念隐喻，以揭示说话者或作者的事件框架。

    

    概念隐喻通过允许我们用更具体或具身化的经验（例如，背负物理重物）来推理更抽象的经验（例如，纳税），从而指导我们的思维和行动（Lakoff和Johnson，2011）。因此，不同的概念隐喻可能导致不同的推理：将纳税视为对社区的投资，而非一种物理负担，会带来截然不同的税收观。识别指导说话者或作者的概念隐喻，有助于揭示他们对事件的框架设定。尽管这些隐喻无法直接观察，但语言中的隐喻表达群体——即语言隐喻——可以作为其证据。基于此，我们提出了一种无监督方法，从语料库中提取语言隐喻，并使用结构化聚类方法将其分组为对应概念隐喻的群组。通过这种方法，我们……

    arXiv:2608.06652v2 Announce Type: replace  Abstract: Conceptual metaphors guide our thinking and actions by allowing us to reason about more abstract experiences (e.g., paying taxes) in terms of more concrete or embodied experiences (e.g., carrying a physical load) (Lakoff and Johnson, 2011). It follows that different conceptual metaphors can result in different reasoning: framing paying taxes as an investment in a community rather than a physical load leads to a very different outlook on taxation. Identifying the conceptual metaphors guiding a speaker or writer thus helps to reveal their framing of events. Though these metaphors can't be observed directly, groups of linguistic metaphors, metaphorical expressions as they appear in language, serve as evidence for them. Motivated by this, we present an unsupervised method that extracts linguistic metaphors from a corpus and uses a structured clustering approach to form groups corresponding to conceptual metaphors. Using this method, we p
    
[^168]: 布线优于混合：不同Transformer规模之间传递了什么——以及什么没有传递

    Wiring Beats Blending: What Transfers Between Transformer Sizes -- and What Doesn't

    [https://arxiv.org/abs/2608.02829](https://arxiv.org/abs/2608.02829)

    本文发现，在不同规模的Transformer模型间转换时，表示对齐强但参数对齐弱，价值在于初始化，并通过最小二乘补偿和方差保持重缩放两个杠杆实现有效转换。

    

    arXiv:2608.02829v3 公告类型：替换交叉 摘要：模型家族通常按规模逐个从头训练。能否将预训练的大模型转换为较小的兄弟模型？我们端到端地表征了Pythia中的1.4B->410M转换。表示在不同规模间强烈对齐（岭回归R^2=0.84），而参数对齐较弱。密集权重投影在功能上具有破坏性，且一个比特精确的控制表明这不是组装伪影：基混合破坏了旋转、每头、GELU和LayerNorm结构。在最佳拟合线性算子之后，权重残差在洗牌控制下在统计上与噪声无异。因此，转换价值存在于初始化中。在匹配预算的持续预训练中，我们将转换分解为两个独立杠杆：最小二乘补偿（功能杠杆，最佳零样本）和方差保持重缩放（动力学杠杆，最佳终点）。补偿是一种令牌高效、低预算的胜利，而非其他。

    arXiv:2608.02829v3 Announce Type: replace-cross  Abstract: Model families are typically trained size by size, each from scratch. Can a pretrained large model instead be converted into a smaller sibling? We characterize the 1.4B->410M conversion in Pythia end to end. Representations align strongly across sizes (ridge R^2=0.84) while parameters align weakly. Dense weight projection is functionally destructive, and a bit-exact control shows this is not an assembly artifact: basis mixing breaks rotary, per-head, GELU, and LayerNorm structure. After the best-fit linear operator, weight residuals are statistically indistinguishable from noise under shuffle controls. Conversion value therefore lives in initialization. In matched-budget continued pre-training we decompose conversion into two independent levers: least-squares compensation (function lever, best zero-shot) and variance-preserving rescale (dynamics lever, best endpoints). Compensation is a token-efficient, low-budget win rather th
    
[^169]: 抖音多模态嵌入模型技术报告

    Douyin Multimodal Embedding Model Technical Report

    [https://arxiv.org/abs/2608.02148](https://arxiv.org/abs/2608.02148)

    DME模型通过两阶段训练结合对比学习的高效性和CoT推理的精细区分能力，在十亿级索引下实现多模态嵌入的高效与准确匹配。

    

    arXiv:2608.02148v2 公告类型：替换交叉。摘要：多模态表示学习是现代人工智能的基石。通过将多模态查询和目标编码为向量，它支撑了工业搜索和推荐系统，并构成了现代代理的基础。现实世界中的平台，如抖音、小红书和YouTube，具有复杂的模态和海量内容，需要在十亿级索引下保持高效，同时具备对困难匹配的精细区分能力。现有的MLLM嵌入模型很少能同时满足这两点。对比模型高效但依赖成对级监督，对于精细区分过于粗糙，而基于CoT的模型通过显式生成提高区分度，但在线服务不实用。我们提出了抖音多模态嵌入（DME），一种两阶段训练的模型，结合了两者的优势。第一阶段进行大规模对比预训练，建立统一的多模态嵌入空间，覆盖广泛的模态和任务。第二阶段进行后续优化（摘要截断，但原文未提供完整内容）。

    arXiv:2608.02148v2 Announce Type: replace-cross  Abstract: Multimodal representation learning is a cornerstone of modern AI. By encoding multimodal queries and targets into vectors, it powers industrial search and recommendation and underpins modern agents. Real-world platforms with complex modalities and massive-scale content, such as Douyin, Xiaohongshu, and YouTube, demand both efficiency under billion-scale indexing and fine-grained discrimination for hard matching. Existing MLLM embedding models rarely satisfy both. Contrastive models are efficient but rely on pair-level supervision too coarse for fine-grained distinctions, while CoT-based models improve discrimination through explicit generation impractical to serve online. We present Douyin Multimodal Embedding (DME), a model trained in two stages to combine both strengths. Stage 1 performs large-scale contrastive pre-training that establishes a unified multimodal embedding space with broad modality and task coverage. Stage 2 su
    
[^170]: 构音障碍ASR中的语音条件效应分析：分层探测研究

    Analyzing Speech Condition Effects in Dysarthric ASR: A Layer-wise Probing Study

    [https://arxiv.org/abs/2608.01865](https://arxiv.org/abs/2608.01865)

    该研究通过分层探测发现，构音障碍语音的ASR性能下降源于音素边界信息在所有层中均弱、深层音素身份恢复差且错误集中于最深层，并利用层选择性LoRA证明中层适应能有效恢复性能。

    

    自动语音识别（ASR）在构音障碍语音上的性能会急剧下降，但发音障碍如何重塑模型内部表征仍未被充分探索。我们对一个基于Transformer的ASR编码器在三种转录匹配条件下的普通话构音障碍语音进行了分层探测分析：原始构音障碍语音、说话人条件零样本TTS重合成语音，以及无条件TTS语音。探测揭示了一个任务和条件依赖的表征层次结构：对于构音障碍语音，音素边界信息在所有层中均保持较弱；对于合成语音，音素身份在深层中可恢复，但对构音障碍语音仍表现不佳；识别困难集中在最深层。此外，词汇声调在所有条件下都是一个持续的错误来源。基于这些见解，层选择性LoRA表明，中层适应（第7层或第5-8层）可恢复接近最优的性能。

    arXiv:2608.01865v2 Announce Type: replace  Abstract: Automatic speech recognition (ASR) performance degrades sharply on dysarthric speech, yet how disordered articulation reshapes a model's internal representations is underexplored. We conduct a layer-wise probing analysis of a transformer ASR encoder on Mandarin dysarthric speech under three transcript-matched conditions: original dysarthric speech, speaker conditioned zero-shot TTS resynthesis, and unconditioned TTS. Probing reveals a task- and condition-dependent representation hierarchy: phoneme boundary information remains weak across all layers for dysarthric speech; phoneme identity is recoverable in deep layers for synthetic speech, but remains poor for dysarthric speech; and recognition difficulty is concentrated in the deepest layers. Furthermore, lexical tone is a persistent error source across all conditions. Guided by these insights, layer-selective LoRA shows that mid-layer adaptation (layer 7 or layers 5-8) recovers near
    
[^171]: 歧义去哪了？探究多模态模型如何解读多义词

    Where did the ambiguity go? Examining how multimodal models interpret polysemous words

    [https://arxiv.org/abs/2608.00410](https://arxiv.org/abs/2608.00410)

    研究发现多模态模型在图像生成中比文本生成更倾向于收敛到单一词义，导致多义性显著降低，且整体多样性远不及人类想象。

    

    人类语言具有高度多义性。许多常见词汇（如“bank”或“palm”）承载着多种不同含义，这些含义塑造了人类的交流和想象。大型语言模型（LLMs）已被证明能理解这种意义的多样性，但关于多义性如何在图像等其他模态中显现，我们知之甚少。我们通过向17个文本到图像模型和15个文本生成模型分别提供一个无上下文的多义词，并测量多次生成中产生的词义，对此进行了研究。我们发现了一个明显的多模态差距：在每个模型家族中，生成的图像所固化的词义远少于生成的句子（归一化熵0.10对0.25），且两者都远低于人们对相同词汇的想象多样性（归一化熵0.47）。然而，当我们要求模型列出其生成对应于每个可能含义的输出频率时，它预测出的分布却显示出……

    arXiv:2608.00410v2 Announce Type: replace  Abstract: Human language is highly polysemous. Many common words (e.g., "bank" or "palm") carry several distinct meanings that shape what humans communicate and imagine. Large language models (LLMs) have been shown to understand this multiplicity of meaning, but much less is known about how polysemy surfaces in other modalities such as images. We study this across 17 text-to-image and 15 text-generation models by giving each a polysemous word with no context to fix its meaning and measuring which senses are produced over many samples. We find a clear multimodal gap, where within every model family, generated images settle on far fewer senses than generated sentences (normalized entropy 0.10 vs. 0.25), and both are far less varied than what people imagine for the same words (normalized entropy 0.47). However, when we instead ask a model to list how often it would generate outputs corresponding to each possible meaning of a word, it predicts dis
    
[^172]: BridgeAlign：为人文与社会科学架起偏好对齐的桥梁

    BridgeAlign: Bridging Preference Alignment for Humanities and Social Sciences

    [https://arxiv.org/abs/2607.27366](https://arxiv.org/abs/2607.27366)

    BridgeAlign是首个面向广泛人文与社会科学学科的偏好对齐流程，通过种子策展、基于角色的偏好数据合成和质量规范引导的偏好优化，解决了开放式任务中质量判断的难题。

    

    arXiv:2607.27366v2 公告类型：替换版 摘要：尽管大语言模型（LLMs）的数据合成技术已广泛应用，但其主要针对可验证答案的领域，忽视了开放式的人文与社会科学（HSS），在这些领域中，细微的质量判断比客观正确性更为重要。这使得偏好对齐成为广泛HSS任务的自然范式。然而，现有方法要么成本高昂，要么不适合广泛的HSS学科。因此，我们提出BridgeAlign，这是首批面向广泛HSS学科的偏好对齐流程之一，包含三个阶段：i）种子策展：通过基于启发式/LLM的过滤和文本细化，从网络语料库中策展HSS种子文档；ii）偏好数据合成：通过基于角色的指令反转和问答一致性检查生成偏好三元组；iii）偏好优化：超越简单的“人类对模型”启发式方法，首先将偏好建立在HSS质量规范上，然后生成过渡性响应。

    arXiv:2607.27366v2 Announce Type: replace  Abstract: While data synthesis for large language models (LLMs) is prevalent, it primarily targets domains with verifiable answers, overlooking open-ended humanities and social sciences (HSS), where nuanced quality judgments matter more than objective correctness. This makes preference alignment a natural paradigm for broad HSS tasks. Yet existing methods are either costly or not tailored to broad HSS disciplines. We thus propose BridgeAlign, among the first preference-alignment pipelines for broad HSS disciplines, with three phases: i) Seed Curation: curating HSS seed documents from web corpora via heuristic/LLM-based filtering and text refinement; ii) Preference Data Synthesis: generating preference triplets via persona-based instruction inversion with Q&A consistency checks; iii) Preference Optimization: moving beyond naive human-vs-model heuristics by first grounding preferences in HSS quality rubric, then generating transitional responses
    
[^173]: 内存高效的音频合成：基于解耦时间深度扩散变换器

    Memory Efficient Audio Synthesis with Decoupled Temporal Depth Diffusion Transformers

    [https://arxiv.org/abs/2607.23811](https://arxiv.org/abs/2607.23811)

    本文提出了一种内存高效的音频合成架构，通过解耦时间与深度处理、复用单一深度解码器生成所有RVQ层级，在设备端实时合成高保真语音，显著降低内存和计算需求。

    

    arXiv:2607.23811v2 公告类型：替换交叉 摘要：Siri 表现力语音通过 AFM 3 Core Advanced（苹果最强大的端上基础模型）在设备上实时合成丰富、可配置的语音。本工作介绍了支撑该能力的内存高效音频合成架构：一个去分词器，将基础模型发出的语义音频令牌转换为高保真音频，在苹果矩阵协处理器（AMX）的严格计算和内存预算内运行。我们将语义音频令牌转换为残差向量量化（RVQ）表示，采用三组件设计：流式编码器、时间解码器和深度解码器，系统性地解耦时间和深度处理。一个可复用的深度解码器，配备扩散变换器（DiT）风格的阶段条件，自回归生成所有 RVQ 层级，取代了先前多解码器架构中专用的逐层级解码器，同时采用因果滑窗。

    arXiv:2607.23811v2 Announce Type: replace-cross  Abstract: Siri Expressive Voices synthesize rich, configurable speech in real time and entirely on device, powered by AFM 3 Core Advanced, Apple's most powerful on-device foundation model. This work presents the memory-efficient audio synthesis architecture behind that capability: a detokenizer that converts the semantic audio tokens emitted by the foundation model into high-fidelity audio within the tight compute and memory budget of the Apple Matrix Coprocessor (AMX). We convert semantic audio tokens to a residual vector quantization (RVQ) representation with a three-component design, a streaming encoder, a temporal decoder, and a depth decoder, that systematically decouples temporal and depth processing. A single reusable depth decoder with Diffusion Transformer (DiT)-style stage conditioning generates all RVQ levels autoregressively, replacing the dedicated per-level decoders of prior multi-decoder architectures, while causal sliding
    
[^174]: 多模态语言模型在NRC反应堆操作员执照考试中的基准测试：微调与检索策略

    Multimodal Language Models Benchmarked Against the NRC Reactor Operator Licensing Examination: Fine-Tuning and Retrieval Strategies

    [https://arxiv.org/abs/2607.22067](https://arxiv.org/abs/2607.22067)

    该论文首次以美国核管理委员会反应堆操作员执照考试的全部历年试卷为基准，严格按80%及格线评估多模态语言模型，并系统比较了微调和检索策略对安全关键领域能力的影响。

    

    arXiv:2607.22067v2 公告类型：替换-交叉 摘要：在安全关键领域，语言模型的能力声明只有在以该领域已强制执行的标准进行衡量时才具有可信度。我们评估了一个开放权重的310亿参数多模态模型（Gemma 4 31B-IT）在美国核管理委员会反应堆操作员通用基础考试（GFE）上的表现，逐份试卷地将其与适用于每位人类考生的80%及格标准进行比较，且不做任何四舍五入。评估集是2015年至2021年3月举行的每一次GFE考试的全量普查，包含七份压水堆（PWR）和七份沸水堆（BWR）试卷，共697个计分项。八种配置跨越三种模型状态：基础模型、基于蒸馏思维链推理的监督微调（SFT）和检索增强微调（RAFT），并采用三种检索条件：无检索以及基于美国能源部基础手册的BM25检索（在固定大小下）。

    arXiv:2607.22067v2 Announce Type: replace-cross  Abstract: Competence claims for a language model in a safety-critical domain are credible when measured against a standard the domain already enforces. We evaluate an open-weight 31-billion-parameter multimodal model (Gemma 4 31B-IT) on the U.S. Nuclear Regulatory Commission Reactor Operator Generic Fundamentals Examination (GFE), scoring it paper by paper against the 80% criterion applied to every human candidate, with no rounding up. The evaluation set is a census of every GFE administered at the March sitting from 2015 to 2021, giving seven pressurized water reactor (PWR) and seven boiling water reactor (BWR) papers and 697 scored items. Eight configurations cross three model states, the base model, supervised fine-tuning (SFT) on distilled chain-of-thought rationales and retrieval-augmented fine-tuning (RAFT), with three retrieval conditions, none and BM25 retrieval over the Department of Energy Fundamentals Handbooks under fixed-siz
    
[^175]: 生成式AI是否取代了监督式XMLC？基于德国科学文献的自动主题索引基准研究

    Does generative AI supersede supervised XMLC? A Benchmark Study on Automated Subject Indexing with German Scientific Literature

    [https://arxiv.org/abs/2607.14882](https://arxiv.org/abs/2607.14882)

    本研究通过对比监督式XMLC方法与基于LLM的生成式方法在德国科学文献自动主题索引任务上的表现，探讨了生成式AI能否取代传统监督式方法，并指出长尾词汇建议是共同挑战。

    

    在一个大型受控词汇表作为标签集的情况下，图书馆中的自动主题索引任务可以被理解为多标签分类任务。如果主题词集规模庞大，该问题便符合极端多标签分类（XMLC）的目标。在本研究中，我们将一系列专门的监督式XMLC方法应用于德国国家图书馆（DNB）收集的当代德国科学文献主题索引测试案例。我们通过引入一个经典的词汇匹配基线以及我们自行开发的三种基于大语言模型（LLM）的方法，将这些结果进行对比基准测试。算法在多个指标上进行了评估和比较，包括与先前索引材料的二元相关性比较，以及由专业主题图书馆员进行的分级相关性评级。所有方法面临的一个挑战是，如何从主题词汇的长尾部分可靠地提出建议。

    arXiv:2607.14882v2 Announce Type: replace-cross  Abstract: With a large controlled vocabulary as the label set, the task of automated subject indexing in a library can be understood as a multi-label classification task. If the set of subject terms is large, the problem fits the Extreme Multi-Label Classification (XMLC) objective. In this study, we apply a selection of specialised supervised XMLC methods to the test case of subject indexing contemporary German scientific literature, collected at the German National Library (DNB). We contrast these results by including a classical lexical matching baseline and three of our own recently developed LLM-based methods into the benchmark. Algorithms are evaluated and compared in several metrics. This includes binary relevance comparisons with previously indexed material, as well as graded relevance ratings by professional subject librarians. A challenge for all methods is to reliably make suggestions from the long tail of the subject vocabular
    
[^176]: 损伤多模态语言模型再现失语症图片命名模式

    Lesioned Multimodal Language Models Reproduce Aphasic Picture-Naming Patterns

    [https://arxiv.org/abs/2607.11621](https://arxiv.org/abs/2607.11621)

    该研究首次证明，通过对通用多模态语言模型（LLaVA 1.6）进行特定损伤扰动，能够以临床可比的比例再现失语症患者图片命名中的多种错误类型，为模拟神经语言障碍提供了新途径。

    

    卒中后失语症通常会产生具有特征性模式的系统性命名错误，但未经过临床模拟设计的通用语言模型是否能再现这些模式仍未得到验证。我们研究了（1）对多模态语言模型的损伤或受控扰动是否能再现图片命名中的不同类型错误，以及（2）该框架是否能再现个体失语症患者（PWAs）的完整错误特征。使用LLaVA 1.6，我们评估了在模型单元上改变层、比例和噪声量的扰动配置。我们检查了278名失语症患者在费城命名测试中的表现，并使用经过验证的神经分类器将反应分为七个类别。七个反应类别中的六个（正确、语义、混合、无关、新词、无反应错误）在跨不同参数空间中以临床可比的比例出现。

    arXiv:2607.11621v2 Announce Type: replace  Abstract: Aphasia following stroke commonly produces systematic naming errors with characteristic profiles, but whether general-purpose language models not designed for clinical simulation can reproduce these patterns remains untested. We investigated (1) whether lesions or controlled perturbations to a multimodal language model can reproduce different types of errors in picture naming, and (2) whether the framework can reproduce the complete error profile of individual persons with aphasia (PWAs). Using LLaVA 1.6, we evaluated perturbation configurations that varied the layer, proportion, and amount of noise applied to model units. We examined 278 PWAs on the Philadelphia Naming Test, classifying responses into seven categories using a validated neural classifier. Six of seven response categories (correct, semantic, mixed, unrelated, neologism, no response errors) emerged at clinically-comparable proportions across distinct parameter space re
    
[^177]: 首届中文BabyLM挑战赛：面向中文的高数据效率与认知合理性语言模型训练

    The First ChineseBabyLM Challenge: training data-efficient and cognitively plausible language models for Chinese

    [https://arxiv.org/abs/2607.10745](https://arxiv.org/abs/2607.10745)

    本文介绍了首届中文BabyLM挑战赛，其核心贡献是设立了一个在有限数据下训练中文语言模型的基准，并发现引入拼音预测辅助目标的DeBERTa-v2架构在多项评估中表现最佳。

    

    摘要：arXiv:2607.10745v2 公告类型：替换 摘要：本文介绍了首届中文BabyLM挑战赛，该挑战赛作为NLPCC 2026的一部分组织。挑战赛要求参赛者使用不超过1.02亿个中文词汇从头训练语言模型。模型在三个轨道上进行了评估：自然语言理解、认知对齐和汉字知识。对分词器、模型架构或训练轮数没有限制。十八支队伍提交了28个不同的模型，生成了74个结果文件。总冠军团队采用了DeBERTa-v2架构，并在预训练期间引入了辅助拼音预测目标。几个提交作品还探索了课程学习策略和架构创新。总体而言，该挑战赛为推进中文语言建模的数据高效和认知合理方法提供了一个基准。

    arXiv:2607.10745v2 Announce Type: replace  Abstract: This paper presents the first ChineseBabyLM Challenge, organized as part of NLPCC 2026. The challenge asked participants to train language models from scratch using no more than 102M Chinese words. The models were evaluated on three tracks: natural language understanding, cognitive alignment, and Hanzi knowledge. There were no restrictions on tokenizers, model architectures, or the number of training epochs. Eighteen teams submitted 28 distinct models, generating 74 result files. The overall-winning team used a DeBERTa-v2 architecture and introduced an auxiliary pinyin-prediction objective during pretraining. Several submissions also explored curriculum-learning strategies and architectural innovations. Overall, the challenge provides a benchmark for advancing data-efficient and cognitively plausible approaches to Chinese language modeling.
    
[^178]: 基于潜在人格特质的高效语言模型安全对齐

    Efficient Safety Alignment of Language Models via Latent Personality Traits

    [https://arxiv.org/abs/2607.07918](https://arxiv.org/abs/2607.07918)

    本文提出潜在人格对齐（LPA），通过仅用66条人格陈述进行对抗训练，在不接触有害数据且不损失性能的情况下，实现接近零的越狱攻击成功率，显著提升语言模型安全对齐效率。

    

    arXiv:2607.07918v2 公告类型：替换-交叉 摘要：当前大型语言模型的安全方法已知易受对抗性攻击，这推动了对稳健替代方案的研究。潜在对抗训练（LAT）是最有效的防御手段之一，但可能降低实用性，且需要在大量有害提示数据集上进行训练。我们引入了潜在人格对齐（LPA），该方法将明确的拒绝有害行为替换为仅对从心理测量人格文献中提取的66条与危害无关的陈述进行对抗训练。我们假设人格锚定的表示与危害规避共享潜在结构，因此对抗性稳定这些表示会隐式约束越狱攻击所利用的子空间。LPA在HarmBench上，针对直接请求和五种越狱方法，实现了接近零的攻击成功率，尽管在训练过程中从未见过有害内容，且在标准基准测试上无性能损失。此外，训练过程...

    arXiv:2607.07918v2 Announce Type: replace-cross  Abstract: Current safety methods for large language models are known to be vulnerable to adversarial attacks, motivating research into robust alternatives. Latent Adversarial Training (LAT) is among the most effective defenses, but can degrade utility and requires training on large datasets of harmful prompts. We introduce Latent Personality Alignment (LPA), which replaces explicit harm refusal with adversarial training on just 66 harm-agnostic statements drawn from psychometric personality literature. We hypothesize that personality-anchored representations share latent structure with harm avoidance, so adversarially stabilizing them implicitly constrains the subspace exploited by jailbreak attacks. LPA achieves near-zero attack success rates on HarmBench across direct requests and five jailbreak methods, despite never seeing harmful content during training and no loss of performance on standard benchmarks. Moreover, the training proces
    
[^179]: PolyWorkBench：评估跨语言长时程工作流中LLM代理的基准

    PolyWorkBench: Benchmarking LLM Agents for Cross-Lingual Long-Horizon Workflows

    [https://arxiv.org/abs/2607.06008](https://arxiv.org/abs/2607.06008)

    本文提出了PolyWorkBench基准，用于评估LLM代理在跨语言长时程工作流中的表现，并通过结构化评分标准来衡量其多语言整合与工具使用能力。

    

    尽管大型语言模型（LLM）代理在单语言长时程规划和工具使用方面表现出色，但企业工作流本质上需要处理跨越扩展轨迹的多语言资源。然而，多语言性与长时程执行之间的相互作用仍未得到充分探索。我们引入了PolyWorkBench，这是一个旨在评估LLM代理在多语言、长时程工作场所工作流中表现的基准。PolyWorkBench包含67个任务，涵盖五个核心领域：商业、知识工作、法律分析、本地化和制造业。任务由论文作者基于真实数据种子编写，并通过第二作者审计独立验证。代理必须整合异构的多语言输入，执行迭代的工具使用轨迹，并生成结构化的领域产物。为了严格评估性能，我们采用Grade（一种任务特定的结构评分标准）作为主要排名指标。

    arXiv:2607.06008v3 Announce Type: replace  Abstract: While Large Language Model (LLM) agents excel at monolingual long-horizon planning and tool use, enterprise workflows inherently require processing multilingual resources across extended trajectories. The interaction between multilinguality and long-horizon execution, however, remains underexplored. We introduce PolyWorkBench, a benchmark designed to evaluate LLM agents on multilingual, long-horizon workplace workflows. PolyWorkBench features 67 tasks across five core domains: commerce, knowledge work, legal analysis, localization, and manufacturing. Tasks are authored by the paper's authors from real-world data seeds and independently verified through a second-author audit. Agents must integrate heterogeneous multilingual inputs, execute iterative tool-use trajectories, and produce structured domain artifacts. To rigorously assess performance, we adopt Grade, a task-specific structural scoring rubric, as our primary ranking metric, 
    
[^180]: 论方向性在结构泛化中的作用

    On the Role of Directionality in Structural Generalization

    [https://arxiv.org/abs/2607.02307](https://arxiv.org/abs/2607.02307)

    本文通过使用CCG方向类型替代AM代数，显著提升了结构泛化性能，尤其在方向性相关类别上表现突出，并与更强的编码器互补。

    

    arXiv:2607.02307v2 公告类型：替换 摘要：几个SLOG测试类别明确涉及方向性区分（修饰语位置移位、论元提取位置），但先前的SOTA模型AM-Parser使用了一种AM代数，其操作不编码方向。我们围绕CCG方向类型（确定性CKY + 单线性解码器，30K可学习参数）重新设计了符号后端。在相同的BERT-base编码器下，系统实现了75.9±6.4%的LF精确匹配，超过了AM-Parser（70.8±4.3%）。根据SLOG自身的类别分组，增益高度方向性：CCG系统在所有5个位置移位类别上优于AM-Parser（+29.9个百分点），而AM-Parser在所有6个递归深度类别上表现更好。将编码器替换为DeBERTa-v3-large后，达到90.7±4.9%的准确率，其中最大的编码器增益出现在递归深度类别中，与方向性增益互补。方向性表示将瓶颈从符号层（AM-Parser的...

    arXiv:2607.02307v2 Announce Type: replace  Abstract: Several SLOG test categories explicitly involve directional distinctions (modifier position shifts, argument extraction positions), yet AM-Parser, the previous SOTA, uses an AM algebra whose operations do not encode direction. We redesign the symbolic backend around CCG directed types (deterministic CKY + single linear decoder, 30K learnable parameters). Under the same BERT-base encoder, the system achieves 75.9$\pm$6.4% LF exact match, surpassing AM-Parser (70.8$\pm$4.3%). Per SLOG's own category groupings, gains are highly directional: the CCG system outperforms AM-Parser on all 5 position-shift categories (+29.9pp), while AM-Parser outperforms on all 6 recursive-depth categories. Replacing the encoder with DeBERTa-v3-large yields 90.7$\pm$4.9%, with the largest encoder gains in recursive-depth categories, complementary to directionality's gains. Directional representations shift the bottleneck from the symbolic layer (AM-Parser's 
    
[^181]: $x$-预测流：掩码扩散语言模型的高效连续解码

    $x$-Prediction Flow: Efficient Continuous Decoding for Masked Diffusion Language Models

    [https://arxiv.org/abs/2606.29066](https://arxiv.org/abs/2606.29066)

    本文提出了一种基于$x$-预测的连续解码框架，通过将掩码预测转化为嵌入空间的连续流，并采用置信度驱动的异步更新，实现了MDLMs的高效且可修订的文本生成。

    

    arXiv:2606.29066v2 公告类型：替换 摘要：掩码扩散语言模型（MDLMs）通过迭代去掩码标记来生成文本，但其标准解码器将每一步简化为二元动作：一个位置要么被固定为单个标记，要么完全保持掩码状态，从而丢弃了丰富的预测信息而非将其向前传递，并导致过早且不可撤销的决策，这限制了在有限解码预算下的性能。在本文中，我们将掩码预测重新解释为清洁状态预测（$x$-预测），并展示其可用于在输入嵌入空间中诱导连续流。基于这一观点，我们提出了一种适用于MDLMs的连续解码框架，其中标记可以在每个扩散步骤中积累部分进展，并保持可修改性。为了匹配语言中位置间不均匀的上下文约束，我们用基于置信度的异步更新替代图像扩散中的全局同步调度，其中扩散...

    arXiv:2606.29066v2 Announce Type: replace  Abstract: Masked diffusion language models (MDLMs) generate text by iteratively unmasking tokens, but their standard decoder reduces each step to a binary action: a position is either committed to a single token or left fully masked, discarding rich predictive information rather than carrying it forward, and forcing premature, irrevocable commitments that lead to poor performance under a limited decoding budget. In this paper, we reinterpret mask prediction as a clean-state prediction ($x$-prediction) and show that it can be used to induce a continuous flow in the input embedding space. Building on this view, we propose a continuous decoding framework for MDLMs where tokens can accumulate partial progress at each diffusion step and remain revisable. To match the uneven contextual constraints across positions in language, we replace the globally synchronous schedule in image diffusion with a confidence-based asynchronous update in which the dif
    
[^182]: DanceOPD：在线策略生成场蒸馏

    DanceOPD: On-Policy Generative Field Distillation

    [https://arxiv.org/abs/2606.27377](https://arxiv.org/abs/2606.27377)

    DanceOPD提出了一种在线策略生成场蒸馏框架，通过将不同图像生成能力（文生图、局部编辑、全局编辑）建模为共享空间中的速度场，并利用学生自身状态进行查询和训练，有效解决了多种能力之间的冲突与组合问题。

    

    现代图像生成需要一个统一的模型，能够集成多种能力，包括文生图、局部编辑和全局编辑。然而，这些能力很少自然对齐，且常常相互冲突。例如，编辑往往会降低文生图的性能，而全局编辑与局部编辑也会相互干扰。因此，如何有效组合这些能力已成为图像生成模型训练的核心挑战。为了解决这一问题，我们提出了DanceOPD，一种用于流匹配模型的在线策略生成场蒸馏框架。该框架将每个样本路由至一个能力场，查询一个低噪声的学生诱导状态，并通过简单的速度均方误差目标进行训练。每个能力源被定义为共享流状态空间上的速度场，学生通过在其自身生成状态上查询这些场来学习组合专家能力。该公式还吸收了操作符依赖。

    arXiv:2606.27377v1 Announce Type: cross  Abstract: Modern image generation demands a single model that unifies diverse capabilities, including text-to-image (T2I), local editing, and global editing. However, these capabilities are rarely naturally aligned and often conflict. For instance, editing tends to degrade T2I performance, while global and local editing interfere with each other. Consequently, effectively composing these capabilities has become a central challenge for image generation model training. To tackle this, we introduce DanceOPD, an on-policy generative field distillation framework for flow-matching models that routes each sample to one capability field, queries one low-noise student-induced state, and trains with a simple velocity MSE objective. With each capability source defined as a velocity field over the shared flow state space, the student learns from fields queried on its own rollout states to compose expert capabilities. This formulation also absorbs operator-d
    
[^183]: 针对大语言模型的蜜源策略：重新思考针对AI攻击者的网络欺骗

    Honeyquest for LLMs: Rethinking Cyber Deception for AI Attackers

    [https://arxiv.org/abs/2606.21037](https://arxiv.org/abs/2606.21037)

    本文通过大规模评估21个大语言模型，发现AI攻击者比人类更易落入网络欺骗陷阱，提出了一种新的自动化评估框架，并指出大语言模型构成独特的攻击者类别。

    

    网络欺骗的经验基础依赖于以人为中心的假设，但自主、AI驱动的攻击者的迅速崛起，挑战了这一基础是否适用于AI代理。为此，我们引入了一个自动化评估框架，改编自Honeyquest工具，以大规模评估大语言模型攻击者的判断力。我们的21个大语言模型队列涵盖10个提供商、多样化的架构和专业化方向、开源和闭源权重模型，以及从8B到超过1T的参数规模。我们评估了这一大语言模型队列（产生10,962个响应）在相同的174个侦察查询集上，与47名参与者的人类基线进行对比。我们的实证评估揭示了三个关键发现，将大语言模型确立为一个独特的攻击者类别：（1）我们队列中的每个模型落入欺骗陷阱的比率显著高于人类攻击者；（2）在人类中观察到的防御性注意力分散效应在...

    arXiv:2606.21037v2 Announce Type: replace-cross  Abstract: The empirical foundation of cyber deception relies on human-centered hypotheses, but the rapid emergence of autonomous, AI-enabled attackers challenges whether this foundation transfers to AI agents. To address this, we introduce an automated evaluation framework adapted from the Honeyquest instrument to assess LLM attacker judgment at scale. Our 21-LLM cohort spanned 10 providers, diverse architectures and specializations, open- and closed-weight models, and parameter scales from 8B to over 1T. We evaluated the performance of this LLM cohort (yielding 10,962 responses) against the 47-participant human baseline across an identical set of 174 reconnaissance queries. Our empirical evaluation reveals three key findings that establish LLMs as a distinct attacker class: (1) every model in our cohort falls for deceptive traps at a significantly higher rate than human attackers; (2) the defensive attention-diversion effect observed in
    
[^184]: 潜在技能：从上下文文本技能到LLM智能体的权重内潜在技能

    LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents

    [https://arxiv.org/abs/2606.06087](https://arxiv.org/abs/2606.06087)

    LatentSkill通过预训练超网络将文本技能转化为LoRA适配器，将技能知识从上下文空间迁移到权重空间，显著减少令牌开销并提升任务性能，同时保持技能的模块化组合能力。

    

    arXiv:2606.06087v2 公告类型：交叉替换 摘要：智能体系统越来越多地使用文本技能来编码可重用的任务流程，但在每一步将这些技能注入提示中会带来大量的上下文开销，并将技能内容以明文形式暴露。我们提出了LatentSkill，一个通过预训练超网络将文本技能转换为即插即用的LoRA适配器的框架。LatentSkill将技能知识存储在权重空间而非上下文空间中，消除了每步的技能令牌，同时保持了模块化加载、缩放和组合能力。在ALFWorld和Search-QA上，LatentSkill优于相应的上下文技能基线，同时使用显著更少的预填充令牌：在已见和未见分割上，ALFWorld成功率分别提高了21.4和13.4个百分点，平均预填充令牌减少了63.9%；Search-QA精确匹配提高了3.0个百分点，每步令牌减少了71.8%。进一步分析表明，生成的技能LoRA形成了结构化组织。

    arXiv:2606.06087v2 Announce Type: replace-cross  Abstract: Agent systems increasingly use textual skills to encode reusable task procedures, but injecting these skills into the prompt at every step incurs substantial context overhead and exposes skill content as plaintext. We present LatentSkill, a framework that converts textual skills into plug-and-play LoRA adapters through a pretrained hypernetwork. LatentSkill stores skill knowledge in weight space rather than context space, removing per-step skill tokens while preserving modular loading, scaling, and composition. On ALFWorld and Search-QA, LatentSkill outperforms the corresponding in-context skill baseline while using substantially fewer prefill tokens: it improves ALFWorld success by 21.4 and 13.4 points on the seen and unseen splits with 63.9% fewer prefill tokens on average, and improves Search-QA exact match by 3.0 points while using 71.8% fewer tokens per step. Further analysis shows that generated skill LoRAs form a structu
    
[^185]: 粒度差距：Gemini模型中谄媚行为的多维跨代审计

    The Granularity Gap: A Multi-Dimensional Cross-Generational Audit of Sycophancy in Gemini Models

    [https://arxiv.org/abs/2606.05183](https://arxiv.org/abs/2606.05183)

    该论文揭示了安全评估中“通过/失败”二元判定与谄媚行为连续评分之间存在不可弥合的“粒度差距”，表明现有评估方法无法充分捕捉模型取悦用户的细微行为。

    

    arXiv:2606.05183v2 公告类型：替换-交叉 摘要：通过/失败的安全评估报告模型是否拒绝，但它不报告模型为了取悦用户而走多远，我们表明这两者接近不同的测量。我们对三代Gemini模型进行了谄媚行为审计，在3种护栏条件下，对7个类别的350个对抗性提示，对8个模型变体的N=8,830个响应进行了评分，采用1-5连续量表评估谄媚性、真实性和拒绝性。评判者自身的拒绝或服从判定仅解释了其谄媚评分方差的29%。我们将剩余部分称为“粒度差距”，它在重新校准下不会闭合：已在使用的切点是在拒绝轴上可用的最佳切点，且该轴的任何函数都无法解释超过35%的方差。阅读四位评判者在评分时写下的内容揭示了原因。在四分之一到三分之一的投票中，他们记录提示未要求任何有害内容，几乎从未出现在两个征求有害行为的类别中。

    arXiv:2606.05183v2 Announce Type: replace-cross  Abstract: Pass/fail safety evaluation reports whether a model refused. It does not report how far a model went to please the user, and we show these are close to different measurements. We audited sycophancy across three Gemini generations, scoring N=8,830 responses from 8 model variants on 350 adversarial prompts in 7 categories under 3 guardrail conditions, on continuous 1-5 scales for sycophancy, truthfulness and refusal.   The judge's own refuse-or-comply verdict explains 29% of the variance in its own sycophancy scores. We term the remainder the Granularity Gap, and it does not close under recalibration: the cut point already in use is the best available on the refusal axis, and no function of that axis explains more than 35%. Reading what four judges wrote while scoring shows why. On a quarter to a third of votes they record that the prompt asked for nothing harmful, almost never in the two categories that solicit a harmful act and
    
[^186]: MCBench：面向全模态大语言模型的多情境安全评估基准

    MCBench: A Multicontext Safety Assessment Benchmark for Omni Large Language Models

    [https://arxiv.org/abs/2606.05177](https://arxiv.org/abs/2606.05177)

    该论文提出了MCBench基准，揭示了全模态大语言模型在安全评估中缺乏有效的跨模态推理能力，尤其在处理细微风险时表现不佳。

    

    现有的多模态安全基准仅关注视觉输入，无法评估处理视觉、音频和文本的全模态大语言模型（LLMs）。我们引入了MCBench，一个包含1196个场景的基准，涵盖四个安全类别，需要整合多种模态以进行准确的安全评估。每个不安全场景都配有一个最小差异的安全对照场景，以评估模型的敏感性。我们对最先进模型的评估揭示了显著挑战。全模态LLMs在应对细微或非物理风险时表现困难，但在存在显著视觉或听觉线索时表现更好。推理轨迹分析显示，尽管模型能提取模态特定信息，但它们往往无法有效整合这些线索进行安全判断。我们的发现表明，当前全模态LLMs在安全关键环境中缺乏稳健的跨模态推理能力，凸显了改进架构的必要性。

    arXiv:2606.05177v2 Announce Type: replace-cross  Abstract: Existing multimodal safety benchmarks focus solely on visual inputs and cannot assess Omni Large Language Models (LLMs) that process vision, audio, and text. We introduce MCBench, a benchmark with 1196 scenarios spanning four safety categories that require integrating multiple modalities for accurate safety assessment. Each unsafe scenario is paired with a minimally different safe counterpart to assess model sensitivity. Our evaluations of state-of-the-art models reveal significant challenges. Omni LLMs struggle with subtle or non-physical risks but perform better when salient visual or acoustic cues are present. Analysis of reasoning traces shows that, although models can extract modality-specific information, they often fail to integrate these cues effectively for safety judgments. Our findings reveal that current Omni LLMs lack robust cross-modal reasoning in safety-critical settings, underscoring the need for improved archi
    
[^187]: SocialCoach：基于代理辅导与实践的个性化社交技能学习

    SocialCoach: Personalized Social Skill Learning with Agentic Tutoring and Practice

    [https://arxiv.org/abs/2606.04155](https://arxiv.org/abs/2606.04155)

    SocialCoach是一个基于大语言模型的代理辅导系统，通过构建理论到实践的语料库和轨迹级优化调度，实现个性化社交技能学习，解决了专家辅导稀缺的难题。

    

    摘要：arXiv:2606.04155v2 公告类型：交叉替换 摘要：谈判和领导力等社交技能在当今互联互通的世界中对个人和职业成功至关重要。然而，由于专家辅导的稀缺性，可扩展且有效的培训仍是一项重大挑战。在这项工作中，我们引入了SocialCoach，一个由大语言模型驱动的代理辅导系统，用于个性化社交技能学习。SocialCoach构建了一个从理论到实践的语料库，包含可追踪的策略、案例和实践场景，并利用该语料库进行调度和反思性辅导。我们将社交实践个性化定义为冷启动、检索约束的序列实践调度问题。给定学习者档案、模拟熟练状态和观察到的实践历史，一个策略生成结构化的处方，并通过语料库检索实现。为了提高调度效果，我们使用基于规则评判的轨迹级GRPO优化完整路径。

    arXiv:2606.04155v2 Announce Type: replace-cross  Abstract: Social skills such as negotiation and leadership are crucial for personal and professional success in today's interconnected world. However, scalable and effective training remains a significant challenge due to the scarcity of expert coaching. In this work, we introduce SocialCoach, an LLM-powered agentic tutoring system for personalized social skill learning. SocialCoach constructs a theory-to-practice corpus of traceable strategies, cases, and practice scenarios, and uses this corpus for both scheduling and reflective tutoring. We formulate social practice personalization as cold-start, retrieval-constrained sequential practice scheduling. Given a learner profile, simulated proficiency state, and observed practice history, a policy produces structured prescriptions that are realized through corpus retrieval. To enhance scheduling effectiveness, we optimize complete pathways with trajectory-level GRPO using rubric-judge based
    
[^188]: 语言模型使用数字特定和单位特定的启发式方法比较数量

    Language Models Compare Quantities Using Number-specific and Unit-specific Heuristics

    [https://arxiv.org/abs/2606.03982](https://arxiv.org/abs/2606.03982)

    语言模型比较带单位数量时，并非统一换算，而是依赖数字和单位的启发式线索，导致边界附近系统性错误。

    

    带有测量单位的数量，如110厘米和1.2米，要求语言模型（LMs）将数字与符号单位尺度相结合。在此，我们研究LMs如何在涵盖多个单位系统的受控环境中比较此类数量。我们发现，在比较边界附近，准确率会下降，此时数值的微小变化决定正确答案。由此产生的错误是系统性的：线性代理模型根据数值差异和单位尺度差异线索预测LM的偏好，并且对这些变量对齐的子空间进行因果干预会改变模型的输出。结果表明，LMs通过一组关于数字和单位的启发式方法（而非先将两个表达式转换为精确的共享尺度表示）来比较数量。

    arXiv:2606.03982v2 Announce Type: replace  Abstract: Quantities with measurement units, such as 110 cm and 1.2 m, require language models (LMs) to combine a numeral with a symbolic unit scale. Here, we study how LMs compare such quantities in controlled settings spanning several unit systems. We find that accuracy degrades near the comparison boundary, where small changes in value determine the correct answer. The resulting errors are systematic: linear surrogate models predict LM preferences from numerical-difference and unit-scale-difference cues, and causal interventions on subspaces aligned with these variables shift model's output. The results suggest that LMs compare quantities through a bag of heuristics over numerals and units, rather than first converting both expressions to an exact shared-scale representation.
    
[^189]: 深层网络中的值向量是否需要来自残差流的上下文？

    Do Value Vectors in Deep Layers Need Context from the Residual Stream?

    [https://arxiv.org/abs/2606.02780](https://arxiv.org/abs/2606.02780)

    本文发现深层网络中的无上下文值向量能显著提升模型性能，并可稀疏存储以提高效率。

    

    arXiv:2606.02780v4 公告类型：替换 摘要：Transformer架构作为现代LLM骨干的成功，在很大程度上归功于其注意力层的使用。注意力层遵循标准神经网络范式：它将残差流作为输入，从而生成依赖于上下文的查询、键和值向量。然而，我们发现，当深层网络仅学习一个无上下文的值向量以保留原始令牌信息，而不从残差流中提取任何上下文时，模型性能会显著提升。当模型拥有这种无上下文的值向量时，再添加上下文依赖的组件对整体基准性能的额外益处微乎其微。这种无上下文的值向量可以存储为稀疏模型参数，从而无需重新计算或持久缓存这些值。通过对这种无上下文值向量的关键设计选择进行系统性消融实验，我们提出了B方案。

    arXiv:2606.02780v4 Announce Type: replace  Abstract: The success of the transformer architecture as the backbone of modern LLMs is in large part due to its use of attention layers. An attention layer follows the standard neural network paradigm: it takes the residual stream as input and thereby produces context-dependent query, key, and value vectors. However, we find that model performance meaningfully improves when deeper layers learn only a context-free value vector to preserve the original token information, without drawing on any context from the residual stream. When the model has access to this context-free value vector, adding back the context-dependent component provides little additional benefit for aggregate benchmark performance. Such context-free value vectors can be stored as sparse model parameters, eliminating the need to recompute or persistently cache these values. Through systematic ablations on the key design choices for such context-free value vectors, we propose B
    
[^190]: 通过可处理提案缓解局部约束解码中的偏差

    Mitigating Bias in Locally Constrained Decoding via Tractable Proposals

    [https://arxiv.org/abs/2606.01926](https://arxiv.org/abs/2606.01926)

    本文提出了一种通过张量化有限自动机构建全局约束解码提案的通用方法，用于序列蒙特卡洛采样，从而有效缓解局部约束解码中的采样偏差。

    

    大型语言模型的生成结果常常无法符合诸如JSON模式之类的期望约束。现有的局部约束解码（LCD）方法通过短视地屏蔽下一词来强制约束，导致采样偏差和性能下降。近期工作使用序列蒙特卡洛（SMC）方法来缓解此类偏差，但设计有效的提案分布或势函数仍是一个关键挑战。在本工作中，我们提出了一种通用方法，用于为从 $p_{\mathrm{lm}}( \cdot \mid \mathrm{constraint})$ 进行SMC采样构建提案和势函数。首先，我们证明有限自动机指定的约束可以在GPU上张量化以实现高效执行，我们利用这一点构建全局约束解码（GCD）提案。此外，利用张量化有限自动机与隐马尔可夫模型共享相同电路结构的事实，我们将它们电路相乘，以进一步优化。

    arXiv:2606.01926v2 Announce Type: replace  Abstract: Generations from large language models often fail to conform to desired constraints such as JSON schema. Existing locally constrained decoding (LCD) approaches enforce constraints by myopically masking out next tokens, resulting in biased sampling and degradation in performance. Recent work uses sequential Monte Carlo (SMC) methods to mitigate such biases, but designing effective proposal distributions or potential functions remains a key challenge. In this work, we propose a generic approach to construct proposals and potentials for SMC sampling from $p_{\mathrm{lm}}( \cdot \mid \mathrm{constraint})$. First, we show that constraints specified as finite automata can be tensorized for efficient execution on GPUs, which we use to construct globally constrained decoding (GCD) proposals. In addition, leveraging the fact that tensorized finite automata share the same circuit structure as hidden Markov models, we circuit-multiply them to o
    
[^191]: ChartFI：多模态大语言模型图表描述的忠实性与洞察力基准测试

    ChartFI: Benchmarking Faithfulness and Insightfulness of Chart Descriptions from Multimodal Large Language Models

    [https://arxiv.org/abs/2605.23694](https://arxiv.org/abs/2605.23694)

    本文提出了ChartFI-Bench，一个针对多模态大语言模型图表描述的多维度基准测试，首次系统性地评估了忠实性和洞察力，并定义了四个质量维度以弥补现有基准的不足。

    

    图表描述对于无障碍访问、跨模态检索以及帮助读者从复杂可视化中提取洞察至关重要。随着多模态大语言模型（MLLMs）越来越多地被用于自动化图表描述生成，一个关键问题随之出现：这些模型实际描述图表的忠实性和洞察力如何？现有基准测试在两个方面存在不足：现有数据集由简单、同质的图表组成，并配有浅显的、列举事实的描述；而主流评估指标未能捕捉描述质量的多面性。为解决这些空白，我们提出了图表忠实性与洞察力基准测试（ChartFI-Bench）。我们首先总结了高质量图表描述的四个维度：事实准确性、显著特征强调、领域知情指导以及图表-文本互补性。在这些维度的指导下，我们构建了一个高质量的数据集。

    arXiv:2605.23694v3 Announce Type: replace  Abstract: Chart descriptions are essential for accessibility, cross-modal retrieval, and assisting readers in extracting insights from complex visualizations. As multimodal large language models (MLLMs) are increasingly adopted for automated chart description generation, a critical question arises: how faithfully and insightfully do these models actually describe charts? Current benchmarks fall short on two fronts: existing datasets consist of simple, homogeneous charts paired with shallow, fact-enumerating descriptions; and prevailing metrics fail to capture the multi-faceted nature of description quality. To address these gaps, we present the Chart Faithfulness and Insightfulness Benchmark (ChartFI-Bench). We first summarize four dimensions that characterize high-quality chart descriptions: factual accuracy, salient feature emphasis, domain-informed guidance, and chart-text complementarity. Guided by these dimensions, we construct a high-qua
    
[^192]: 超图即语言

    Hypergraph as Language

    [https://arxiv.org/abs/2605.21858](https://arxiv.org/abs/2605.21858)

    本文提出“超图即语言”视角和Hyper-Align框架，通过将超图结构直接编译为LLM可用的超图标记，保留高阶关联的原生语义，从而克服现有方法在图中心化处理中的局限。

    

    摘要：arXiv:2605.21858v2 公告类型：替换 摘要：大型语言模型（LLMs）最近在建模关系结构方面展现出强大潜力。然而，现有方法从根本上仍是图中心化的：它们专注于将成对图结构处理成LLM可理解的标记。相比之下，许多现实世界中的关系模式并不天然符合成对边假设，更适合建模为超图中的高阶关联。对于超图结构，现有方法往往无法保留多个对象通过同一高阶关系联合连接的原生语义，限制了它们利用复杂结构的能力。为解决这一局限性，我们提出了“超图即语言”的视角，并提出了Hyper-Align，一个用于大型语言模型的超图原生对齐框架。Hyper-Align将查询对象中心的超图上下文编译成基础LLM可直接消费的超图标记。

    arXiv:2605.21858v2 Announce Type: replace  Abstract: Large language models (LLMs) have recently shown strong potential in modeling relational structures. However, existing approaches remain fundamentally graph-centric: they focus on processing pairwise graph structures into tokens that LLMs can understand. In contrast, many real-world relational patterns do not naturally conform to the pairwise-edge assumption, and are better modeled as high-order associations in hypergraphs. For hypergraph structures, existing methods often fail to preserve the native semantics that multiple objects are jointly connected by the same high-order relation, limiting their ability to exploit complex structures. To address this limitation, we put forth the "Hypergraph as Language" perspective and propose Hyper-Align, a hypergraph-native alignment framework for large language models. Hyper-Align compiles the query-object-centered hypergraph context into hypergraph tokens directly consumable by a base LLM. Sp
    
[^193]: SymbolicLight V1：高激活稀疏性下的尖峰门控双路径语言建模

    SymbolicLight V1: Spike-Gated Dual-Path Language Modeling at High Activation Sparsity

    [https://arxiv.org/abs/2605.21333](https://arxiv.org/abs/2605.21333)

    SymbolicLight V1通过尖峰门控双路径设计，结合LIF动态与连续残差流，在超过89%的激活稀疏性下实现了与密集Transformer可比的语言建模性能。

    

    arXiv:2605.21333v2 公告类型：替换-交叉 摘要：原生训练的尖峰语言模型必须通过稀疏二元激活在时间上保留信息，这种组合相对于密集Transformer产生了持续的质量差距。我们提出了SymbolicLight V1，一种尖峰门控双路径语言模型，它将二元泄漏积分-发放（LIF）动力学与连续残差流相结合。其双路径SparseTCAM混合器将一阶指数衰减状态与连续残差流上的窗口局部注意力相结合，随后使用上下文条件解码头。我们在一个包含30亿令牌、10个领域的中英混合语料库上从头训练了四个194M参数模型。在令牌加权的保留集上，这些模型达到了PPL 8.88-8.93（均值8.904，样本标准差0.019），同时实现了超过89%的逐元素激活稀疏性。代码令牌占该保留集的43.7%；十个领域PPL的未加权均值为29.38。在同一语料库下...

    arXiv:2605.21333v2 Announce Type: replace-cross  Abstract: Natively trained spiking language models must preserve information across time while operating through sparse binary activations, a combination that has produced a persistent quality gap relative to dense Transformers. We present SymbolicLight V1, a spike-gated dual-path language model that couples binary Leaky Integrate-and-Fire (LIF) dynamics with a continuous residual stream. Its Dual-Path SparseTCAM mixer combines a first-order exponential-decay state with windowed local attention on the continuous residual stream, followed by a context-conditioned decoding head.   We train four 194M-parameter models from scratch on a 3B-token, 10-domain Chinese-English corpus. On a token-weighted held-out set the runs reach PPL 8.88-8.93 (mean 8.904, sample standard deviation 0.019) at more than 89% per-element activation sparsity. Code tokens are 43.7% of that set; the unweighted mean of the ten domain PPLs is 29.38. Under the same corpus
    
[^194]: 单轮向量RAG与LLM编译维基：对小型多领域研究语料库的预注册比较

    Single-Round Vector RAG vs an LLM-Compiled Wiki: A Preregistered Comparison on a Small Multi-Domain Research Corpus

    [https://arxiv.org/abs/2605.18490](https://arxiv.org/abs/2605.18490)

    本文预注册比较了单轮向量RAG和LLM编译维基在小型研究语料库上的问答性能，发现维基在跨论文综合上更优，但RAG在单事实查找上符合预期，而维基构建成本远高于查询成本。

    

    摘要：我们预注册了一项比较，旨在评估两种帮助LLM回答小型研究语料库问题的方法：单轮向量RAG系统和由工具使用代理浏览的LLM编译markdown维基。两种系统在24篇论文上回答了相同的13个问题，使用相同的答案生成模型，并由两位盲审LLM评委评分。三个预注册预测按注册顺序得出：一个弱支持、一个支持、一个反驳。预测维基能更好地跨论文综合；它在连接发现方面得分更高，但其组织优势在合并两位评委分数后低于注册阈值。预测RAG在单事实查找上能保持优势，并满足注册测试，尽管仅第二位评委就会反驳它。预测维基构建成本高但查询成本低；构建方面大约高出两个数量级。

    arXiv:2605.18490v2 Announce Type: replace  Abstract: We preregistered a comparison of two ways to help an LLM answer questions over a small research corpus: a single-round Vector RAG system and an LLM-compiled markdown wiki browsed by a tool-using agent. Both systems answered the same 13 questions over 24 papers using the same answer-generating model, and their answers were scored by two blinded LLM judges. The three preregistered predictions, in registered order, came out one weakly supported, one supported, and one refuted. The wiki was predicted to synthesize better across papers; it scored much better at connecting findings, but its organization advantage fell below the registered threshold once both judges' scores were combined. RAG was predicted to hold its own on single-fact lookup, and it met the registered test, though the second judge alone would have refuted it. The wiki was predicted to be expensive to build and cheap to query; the build side held by roughly two orders of m
    
[^195]: BiAxisBias：超越单一提示与单一解释评估LLM偏见

    BiAxisBias: Evaluating LLM Bias Beyond a Single Prompt and a Single Explanation

    [https://arxiv.org/abs/2605.09041](https://arxiv.org/abs/2605.09041)

    本文提出BiAxisBias，一种多维度审计框架，通过变化任务、角色、视角、情感和措辞揭示LLM偏见分数对审计设计的敏感性，并量化了不同设计选择对偏见评估结果的影响。

    

    LLM偏见分数可能依赖于审计设计。我们引入了BiAxisBias，这是一种预先指定的审计方法，在保留强制选择和理由作为独立协议读数的同时，对200个刻板印象陈述进行任务、角色、视角、情感和措辞的变化。其主矩阵涵盖8个LLM和401个模板（641,600个响应）。在五个等价问题中，1600个模型-陈述对中有17.1%改变了选择。在双臂中每个单元有三次观察的情况下，所有十个三措辞子集的平均不稳定性为10.5%，而三次相同调用的不稳定性为6.3%。在四个受控任务范式中，28个模型对中有9个发生逆转；一个七因素析因敏感性识别出任务-情感交互是最大的双向成分（原始η²=0.0465）。在10,000次等预算重抽样中，相对于声明的18条件有限参考的平均绝对误差，单模板集中为10.63点，矩阵-wide简单为1.86。

    arXiv:2605.09041v2 Announce Type: replace  Abstract: LLM bias scores can depend on audit design. We introduce BiAxisBias, a prespecified audit varying task, role, perspective, sentiment, and wording over 200 stereotype statements while retaining forced Selection and Rationale as separate protocol readouts. Its main matrix spans eight LLMs and 401 templates (641,600 responses).   Across five equivalent questions, 17.1% of 1,600 model-statement pairs change Selection. With three observations per unit in both arms, instability averages 10.5% across all ten three-wording subsets, versus 6.3% for three identical calls. Across four controlled task paradigms, 9/28 model pairs reverse; a seven-model factorial sensitivity identifies task-by-sentiment as the largest two-way component (raw eta-squared = 0.0465).   In 10,000 equal-budget resampling draws, mean absolute error against a declared 18-condition finite reference is 10.63 points for one-template concentration, 1.86 for matrix-wide simple
    
[^196]: jina-embeddings-v5-omni：通过锁定对齐塔实现几何保持嵌入

    jina-embeddings-v5-omni: Geometry-preserving Embeddings via Locked Aligned Towers

    [https://arxiv.org/abs/2605.08384](https://arxiv.org/abs/2605.08384)

    本文提出GELATO方法，通过冻结骨干模型并仅训练连接组件（占总权重0.35%），构建了jina-embeddings-v5-omni多模态嵌入套件，能将文本、图像、音频和视频编码到同一语义空间，且保持几何特性。

    

    在本文中，我们引入了GELATO（通过锁定对齐塔实现几何保持嵌入），这是一种新颖的多模态嵌入模型方法。我们基于VLM风格架构，其中非文本编码器被调整为为语言模型生成输入，而语言模型则为各种输入类型生成嵌入。我们展示了成果：jina-embeddings-v5-omni套件，这是一对模型，能够将文本、图像、音频和视频输入编码到单一语义嵌入空间中。GELATO扩展了现有的两个Jina Embeddings v5文本模型，通过添加图像和音频编码器来支持更多模态。骨干文本嵌入模型和新增的非文本模态编码器保持冻结。我们仅训练了连接组件，这占联合模型总权重的0.35%。因此，训练比全参数重训练高效得多。此外，语言模型保持基本不变，保留了其原有能力。

    arXiv:2605.08384v4 Announce Type: replace  Abstract: In this work, we introduce GELATO (Geometry-preserving Embeddings via Locked Aligned TOwers), a novel approach to multimodal embedding models. We build on the VLM-style architecture, in which non-text encoders are adapted to produce input for a language model, which in turn generates embeddings for all varieties of input. We present the result: the jina-embeddings-v5-omni suite, a pair of models that encode text, image, audio, and video input into a single semantic embedding space. GELATO extends the two Jina Embeddings v5 Text models to support additional modality by adding encoders for images and audio. The backbone text embedding models and the added non-text modality encoders remain frozen. We only trained the connecting components, representing 0.35% of the total weights of the joint model. Training is therefore much more efficient than full-parameter retraining. Additionally, the language model remains effectively unaltered, pr
    
[^197]: 泄漏审计基准揭示跨受试者听觉诱发脑电元音感知解码证据有限

    Leakage-Audited Benchmarking Reveals Limited Evidence for Cross-Subject Auditory-Evoked EEG Vowel Perception Decoding

    [https://arxiv.org/abs/2605.00865](https://arxiv.org/abs/2605.00865)

    该研究通过严格的泄漏审计基准，发现跨受试者听觉脑电元音解码的证据非常有限，即使最佳模型也仅略高于随机水平且不显著。

    

    我们测试了在单一基准中控制试验身份、模型身份、预测来源和参与者水平推断时，听觉诱发脑电是否支持受试者无关的五元音感知解码。我们从OpenNeuro ds006104版本1.0.1重建了研究2的事件表，并分析了辅音-元音对任务。一对一标记-刺激配对产生了3,840个独立试验；对照条件选择和伪迹拒绝保留了来自16名参与者和61个脑电通道的1,094个时段。使用留一受试者测试评估了13种独特实现，参与者指标从33个完整预测副本中的36,102个试验预测中重建。随机森林在数值上最高，平衡准确率为21.474%（95%参与者自助区间，19.526-23.482%；随机水平为20%），但其参与者水平测试或任何实现均未通过校正。

    arXiv:2605.00865v3 Announce Type: replace-cross  Abstract: We tested whether auditory-evoked EEG supports subject-independent five-vowel perception decoding when trial identity, model identity, prediction provenance, and participant-level inference are controlled within a single benchmark. We reconstructed Study 2 event tables from OpenNeuro ds006104 version 1.0.1 and analyzed the consonant-vowel pair task. One-to-one marker-stimulus pairing yielded 3,840 independent trials; control-condition selection and artifact rejection retained 1,094 epochs from 16 participants and 61 EEG channels. Thirteen unique implementations were evaluated using leave-one-subject-out testing, with participant metrics reconstructed from 36,102 trial predictions across 33 complete prediction replicas. Random Forest was numerically highest at 21.474% balanced accuracy (95% participant-bootstrap interval, 19.526-23.482%; chance, 20%), but neither its participant-level tests nor any implementation survived correc
    
[^198]: 单调稀疏自编码器特征识别（MoRFI）

    MoRFI: Monotonic Sparse Autoencoder Feature Identification

    [https://arxiv.org/abs/2604.26866](https://arxiv.org/abs/2604.26866)

    本文通过受控微调实验发现，新知识的引入会加剧大模型幻觉，并识别出与之因果相关的潜在方向，揭示了SFT导致性能退化的机制。

    

    大型语言模型（LLMs）在预训练阶段通过下一个词预测获取了大部分事实知识。后续的后训练阶段常常引入参数知识之外的新事实，从而导致幻觉现象。虽然已有研究表明，对新知识进行监督微调（SFT）可能会加剧这一问题，但其潜在机制仍知之甚少。我们进行了一项受控的微调实验，聚焦于闭卷问答任务，并识别出与这种性能退化因果相关的潜在方向。具体而言，我们在一个单一问答数据集的七种受控混合条件下微调了Llama 3.1 8B、Gemma 2 9B和Mistral 7B v03模型，控制了新知识占比和训练轮数。通过测量测试集上的性能，我们验证了逐步引入新知识会增加幻觉，且该效应在训练时间延长时更为显著。

    arXiv:2604.26866v2 Announce Type: replace  Abstract: Large language models (LLMs) acquire most of their factual knowledge during the pre-training stage, through next token prediction. Subsequent stages of post-training often introduce new facts outwith the parametric knowledge, giving rise to hallucinations. While it has been demonstrated that supervised fine-tuning (SFT) on new knowledge may exacerbate the problem, the underlying mechanisms are still poorly understood. We conduct a controlled fine-tuning experiment, focusing on closed-book QA, and identify latent directions causally implicated in this degradation. Specifically, we fine-tune Llama 3.1 8B, Gemma 2 9B and Mistral 7B v03 on seven controlled mixtures of a single QA dataset, controlling for the percentage of new knowledge and number of training epochs. By measuring performance on the test set, we validate that incrementally introducing new knowledge increases hallucinations, with the effect being more pronounced with prolon
    
[^199]: 无需手写规则的SLOG结构泛化

    Structural Generalization on SLOG without Hand-Written Rules

    [https://arxiv.org/abs/2604.26157](https://arxiv.org/abs/2604.26157)

    本文提出一种无需手写规则的神经细胞自动机方法，在SLOG基准上实现接近AM-Parser的准确率，并揭示所有失败仅源于两种特定结构机制。

    

    语义解析中的结构泛化要求系统将学习到的组合规则应用于新的结构组合。现有方法要么依赖手写代数规则（如AM-Parser），要么无法在结构上泛化（如基于Transformer的模型）。我们提出了一种替代方案，无需手写组合规则，基于带有离散瓶颈的神经细胞自动机（NCA）：所有组合规则均通过局部迭代从数据中学习。在SLOG基准上，该系统在10个种子上实现了$67.3 \pm 0.2\%$的总体准确率（AM-Parser为$70.8 \pm 4.3\%$），其中17个结构泛化类别中有11个达到$100\%$的类型精确匹配，包括AM-Parser得分仅为$0$--$74\%$的三个类别。分析显示，所有5,539个失败实例均可归结为两种机制：wh提取上下文与缩减动词类型的新组合，以及修饰语出现在主语上。

    arXiv:2604.26157v4 Announce Type: replace-cross  Abstract: Structural generalization in semantic parsing requires systems to apply learned compositional rules to novel structural combinations. Existing approaches either rely on hand-written algebraic rules (AM-Parser) or fail to generalize structurally (Transformer-based models). We present an alternative requiring no hand-written compositional rules, based on a neural cellular automaton (NCA) with a discrete bottleneck: all compositional rules are learned from data through local iteration. On the SLOG benchmark, the system achieves an overall accuracy of $67.3 \pm 0.2\%$ across 10 seeds (AM-Parser: $70.8 \pm 4.3\%$), with 11 of 17 structural generalization categories at $100\%$ type-exact match, including three where AM-Parser scores $0$--$74\%$. Analysis reveals that all 5,539 failure instances reduce to exactly two mechanisms: novel combinations of wh-extraction context with reduced verb types, and modifiers appearing on the subject
    
[^200]: 潜意识引导：隐藏信号的更强编码

    Subliminal Steering: Stronger Encoding of Hidden Signals

    [https://arxiv.org/abs/2604.25783](https://arxiv.org/abs/2604.25783)

    本文提出潜意识引导方法，通过训练引导向量实现更复杂多词偏差的可靠传递，扩展了潜意识学习的信号范围。

    

    潜意识学习描述了一种学生语言模型通过对看似无害的数据进行微调而继承有偏教师模型的行为偏差的现象。先前的研究已开始刻画这一现象，但仍留下关于其能传递的信号范围、解释机制以及偏差编码精度的未解问题。我们通过引入潜意识引导来解决这些问题，这是潜意识学习的一种变体，其中教师的偏差并非像先前工作那样通过系统提示实现，而是通过训练一个引导向量来最大化一组目标样本的似然。首先，我们表明潜意识引导能传递复杂的多词偏差，而先前工作集中于单词偏好，展示了潜意识可传递信号的广泛范围。此外，这种传递足够可靠，能出现在先前被认为不表现出潜意识特征的设置中。

    arXiv:2604.25783v2 Announce Type: replace  Abstract: Subliminal learning describes a student language model inheriting a behavioral bias by fine-tuning on seemingly innocuous data generated by a biased teacher model. Prior work has begun to characterize this phenomenon but leaves open questions about the scope of signals it can transfer, the mechanisms that explain it, and the precision with which a bias can be encoded. We tackle these problems by introducing subliminal steering, a variant of subliminal learning in which the teacher's bias is implemented not via a system prompt, as in prior work, but through a steering vector trained to maximize the likelihood of a set of target samples. First, we show that subliminal steering transfers complex multi-word biases, whereas prior work focused on single-word preferences, demonstrating a large scope of subliminally transferable signals. Moreover, the transfer is reliable enough to appear in settings previously thought not to exhibit sublimi
    
[^201]: 通过联合多任务学习增强科学课堂话语分析中的推理组件分类

    Enhancing Science Classroom Discourse Analysis through Joint Multi-Task Learning for Reasoning-Component Classification

    [https://arxiv.org/abs/2604.21137](https://arxiv.org/abs/2604.21137)

    本文提出了一种结合分层重划分、LLM合成数据增强和双探针头RoBERTa分类器的自动化课堂话语分析系统，有效解决了标签不平衡问题，并显著提升了科学课堂推理组件的分类性能。

    

    分析科学课堂中学生的推理模式对于理解知识建构机制和改善教学实践以最大化认知参与至关重要，然而大规模手动编码课堂话语仍然极其耗费人力。我们提出了一种自动化话语分析系统（ADAS），该系统联合分类教师和学生的话语，沿着两个互补维度：话语类型和推理组件，这些维度源自我们之前的CDAT框架。为了解决少数类别中严重的标签不平衡问题，我们（1）对注释语料库进行分层重划分，（2）应用基于LLM的合成数据增强，针对少数类别，（3）训练了一个双探针头RoBERTa-base分类器。一个零样本GPT-5.4基线在UT上达到宏F1为0.467，在RC上为0.476，为仅提示方法建立了有意义的上限，从而激励微调。除了分类之外...

    arXiv:2604.21137v3 Announce Type: replace-cross  Abstract: Analyzing the reasoning patterns of students in science classrooms is critical for understanding knowledge construction mechanism and improving instructional practice to maximize cognitive engagement, yet manual coding of classroom discourse at scale remains prohibitively labor-intensive. We present an automated discourse analysis system (ADAS) that jointly classifies teacher and student utterances along two complementary dimensions: Utterance Type and Reasoning Component derived from our prior CDAT framework. To address severe label imbalance among minority classes, we (1) stratify-resplit the annotated corpus, (2) apply LLM-based synthetic data augmentation targeting minority classes, and (3) train a dual-probe head RoBERTa-base classifier. A zero-shot GPT-5.4 baseline achieves macro-F1 of 0.467 on UT and 0.476 on RC, establishing meaningful upper bounds for prompt-only approaches motivating fine-tuning. Beyond classification
    
[^202]: 迷失在适应中：视频-语言模型中时间推理的层选择性恢复

    Lost in Adaptation: Layer-Selective Recovery of Temporal Reasoning in Video-Language Models

    [https://arxiv.org/abs/2604.11399](https://arxiv.org/abs/2604.11399)

    本文提出MERIT框架，通过层选择性模型合并和CMA-ES优化，在不牺牲时间感知的情况下显著恢复视频-语言模型的时间推理能力，相对增益最高达27.8%。

    

    arXiv:2604.11399v2 公告类型：替换-交叉 摘要：多模态适应会削弱视频-语言模型（VLM）中的时间推理（TR）能力，使模型能够感知显著事件，却无法推断其时间和因果结构。我们引入了MERIT，一种无梯度的框架，通过层选择性模型合并来修复这一能力。MERIT为每个自注意力层分配VLM主导或LLM主导的插值，并使用协方差矩阵自适应进化策略（CMA-ES）在奖励TR增益同时惩罚时间感知（TP）退化的目标下搜索由此产生的组合空间。在三个VLM家族和五个视频基准测试中，MERIT在保持TP的同时持续提升TR；在紧凑诊断集上选择的配方可迁移到四个未见过的基准测试，相对增益高达27.8%。干预性掩蔽和帧级归因进一步表明，选定的层在功能上至关重要。

    arXiv:2604.11399v2 Announce Type: replace-cross  Abstract: Multimodal adaptation can erode temporal reasoning (TR) in video-language models (VLMs), leaving models able to perceive salient events yet unable to infer their temporal and causal structure. We introduce MERIT, a gradient-free framework that repairs this capability through layer-selective model merging. MERIT assigns each self-attention layer a VLM-dominant or LLM-dominant interpolation and uses the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to search the resulting combinatorial space under an objective that rewards TR gains while penalizing temporal perception (TP) degradation. Across three VLM families and five video benchmarks, MERIT consistently improves TR while preserving TP; recipes selected on a compact diagnostic set transfer to four unseen benchmarks, with relative gains of up to 27.8%. Interventional masking and frame-level attribution further show that the selected layers are functionally important f
    
[^203]: 更短，但仍值得信赖？链式思维压缩的实证研究

    Shorter, but Still Trustworthy? An Empirical Study of Chain-of-Thought Compression

    [https://arxiv.org/abs/2604.04120](https://arxiv.org/abs/2604.04120)

    本研究首次系统实证发现，链式思维压缩虽节省推理成本，但常损害模型的安全性、抗幻觉和多语言鲁棒性，且不同压缩方法退化特征各异。

    

    arXiv:2604.04120v2 公告类型：替换。摘要：长链式思维（Long-CoT）推理模型推动了大量关于压缩推理轨迹以降低推理成本的研究，但现有评估几乎只关注任务准确性和令牌节省。可信赖性属性，无论是通过后期训练获得还是强化，都编码在压缩所修改的同一参数空间中。这意味着保持准确性并不先验地保证保持可信赖性。我们进行了首次关于CoT压缩如何影响模型可信赖性的系统性实证研究，评估了不同规模的多个模型，涵盖三个维度：安全性、抗幻觉能力和多语言鲁棒性。在受控比较下，我们发现CoT压缩经常引入可信赖性退化，且不同方法在不同维度上表现出显著不同的退化特征。为便于跨基础模型进行公平比较...

    arXiv:2604.04120v2 Announce Type: replace  Abstract: Long chain-of-thought (Long-CoT) reasoning models have motivated a growing body of work on compressing reasoning traces to reduce inference cost, yet existing evaluations focus almost exclusively on task accuracy and token savings. Trustworthiness properties, whether acquired or reinforced through post-training, are encoded in the same parameter space that compression modifies. This means preserving accuracy does not, a priori, guarantee preserving trustworthiness. We conduct the first systematic empirical study of how CoT compression affects model trustworthiness, evaluating multiple models of different scales along three dimensions: safety, hallucination resistance, and multilingual robustness. Under controlled comparisons, we find that CoT compression frequently introduces trustworthiness regressions and that different methods exhibit markedly different degradation profiles across dimensions. To enable fair comparison across bases
    
[^204]: I-CALM：激励大语言模型进行选择性回答的置信度感知弃权机制

    I-CALM: Incentivizing Confidence-Aware Abstention for LLM Selective Answering

    [https://arxiv.org/abs/2604.03904](https://arxiv.org/abs/2604.03904)

    I-CALM通过结合言语置信度、收益激励和规范性指导，在提示级别上促使黑盒大语言模型在可能出错时选择性弃权，从而提升事实性问答的准确性和可靠性。

    

    大语言模型（LLMs）经常给出自信但错误的答案，部分原因是标准评估激励机制鼓励猜测而非表达不确定性。我们研究针对可验证事实问题的认知性弃权，目标是改进选择性回答，使LLMs在可能出错时弃权，同时保留正确答案。受人类在问答中的行为决策启发，我们引入了I-CALM，一个面向黑盒LLMs的提示级框架。I-CALM结合了诱导的言语置信度、宣布的回答/弃权收益，以及强调真实性、谦逊、证据支持和责任感的规范性指导。为了区分有针对性的弃权与不加选择的拒绝，我们采用两阶段评估协议，其中LLMs首先选择回答或弃权，然后被迫对弃权的问题提供最佳猜测。跨模型和事实...

    arXiv:2604.03904v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) often produce confident but incorrect answers, in part because standard evaluation incentives reward guessing over expressing uncertainty. We study epistemic abstention for factual questions with verifiable answers, where the goal is to improve selective answering, making LLMs abstain when they are likely to be wrong while preserving correct answers. Inspired by human behavioral decisions in question answering, we introduce I-CALM, a prompt-level framework for black-box LLMs. I-CALM combines elicited verbal confidence, announced answer/abstain payoffs, and normative guidance emphasizing truthfulness, humility, evidential support, and responsibility. To distinguish targeted abstention from indiscriminate refusal, we use a two-stage evaluation protocol, in which LLMs first choose whether to answer or abstain, and are then forced to provide a best guess for the abstained ones. Across models and factual
    
[^205]: 训练自己作为大型语言模型：探索AI素养通过角色扮演LLM训练对说服力的影响

    Train Yourself as an LLM: Exploring Effects of AI Literacy on Persuasion via Role-playing LLM Training

    [https://arxiv.org/abs/2604.02637](https://arxiv.org/abs/2604.02637)

    本文提出了一种基于角色扮演的游戏化AI素养教程LLMimic，通过让用户模拟LLM训练过程，增强其对AI说服的抵抗力，并验证了其在不同说服场景中的有效性。

    

    随着大型语言模型（LLMs）变得越来越有说服力，人们担心其可能在各种情境下大规模影响公众的意见和决策。先前的缓解措施（例如AI检测器和免责声明）主要将人们视为AI生成信息的被动接收者。为了针对具有说服力的AI提供更主动的干预，我们引入了$\textbf{LLMimic}$，一种基于角色扮演的、互动的、游戏化的AI素养教程，参与者扮演LLM的角色，并经历训练流程的三个关键阶段（预训练、SFT和RLHF）。我们进行了一项$2 \times 3$的组间研究（$N = 274$），参与者要么（1）观看AI历史视频（对照组），要么（2）与LLMimic互动（实验组），然后参与三种现实AI说服场景之一：（a）慈善捐赠说服，（b）恶意金钱索取，或（c）酒店推荐。我们的结果显示...

    arXiv:2604.02637v2 Announce Type: replace  Abstract: As large language models (LLMs) become increasingly persuasive, there is concern that people's opinions and decisions may be influenced across various contexts at scale. Prior mitigation (e.g., AI detectors and disclaimers) largely treats people as passive recipients of AI-generated information. To provide a more proactive intervention against persuasive AI, we introduce $\textbf{LLMimic}$, a role-play-based, interactive, gamified AI literacy tutorial, where participants assume the role of an LLM and progress through three key stages of the training pipeline (pretraining, SFT, and RLHF). We conducted a $2 \times 3$ between-subjects study ($N = 274$) where participants either (1) watched an AI history video (control) or (2) interacted with LLMimic (treatment), and then engaged in one of three realistic AI persuasion scenarios: (a) charity donation persuasion, (b) malicious money solicitation, or (c) hotel recommendation. Our results s
    
[^206]: ContextClaim：一种以上下文为驱动的可验证声明检测范式

    ContextClaim: A Context-Driven Paradigm for Verifiable Claim Detection

    [https://arxiv.org/abs/2603.30025](https://arxiv.org/abs/2603.30025)

    本文提出ContextClaim，通过将证据检索引入声明检测阶段，利用上下文信息（如实体和事件的可获取性）来提升可验证声明检测的准确性，从而弥补传统方法仅依赖声明句子的局限性。

    

    arXiv:2603.30025v2 公告类型：替换  摘要：自动化事实核查流程通常以过滤阶段开始，该阶段决定哪些声明值得验证，因为后续的证据检索和验证组件在大规模应用中成本高昂。此阶段的核心任务是可验证声明检测，即判断一个陈述是否原则上可依据外部证据进行核查。先前关于此任务以及与之密切相关的检查价值概念的研究，仅基于声明句子本身来决定。我们认为这种方法是受限的，因为判断一个陈述是否可核查往往依赖于识别其所提及的实体和事件，以及关于它们的外部信息是否实际可用。受下游验证系统依赖检索到的证据的启发，我们将检索上移到检测阶段，并引入ContextClaim。给定一个输入声明，该方法首先识别相关上下文，然后评估其可验证性。

    arXiv:2603.30025v2 Announce Type: replace  Abstract: Automated fact-checking pipelines typically begin with a filtering stage that decides which claims are worth verifying, given that the later evidence retrieval and verification components are expensive to apply at scale. A central task in this stage is verifiable claim detection, which asks whether a statement is in principle checkable against external evidence. Prior work on this task, as well as on the closely related notion of check-worthiness, conditions its decisions only on the claim sentence itself. We argue that this is restrictive, because deciding whether a statement is checkable often depends on identifying the entities and events it mentions, and on whether external information about them is actually available in the first place. Motivated by how downstream verification systems rely on retrieved evidence, we move retrieval upstream into the detection stage and introduce ContextClaim. Given an input claim, the approach ide
    
[^207]: MechMath：基于Sorrifier驱动的形式化分解工作流用于自动定理证明

    MechMath: Sorrifier-Driven Formal Decomposition Workflow for Automated Theorem Proving

    [https://arxiv.org/abs/2603.24465](https://arxiv.org/abs/2603.24465)

    MechMath提出了一种基于Sorrifier驱动的形式化分解工作流，利用Lean中的sorry占位符精确隔离未解决的子目标，避免了上下文过长和低效重生成的问题，从而提升了复杂数学定理证明的效率和成功率。

    

    arXiv:2603.24465v2 公告类型：替换 摘要：近年来，大型语言模型（LLMs）和基于LLM的智能体显著提升了自动定理证明的能力。然而，对于需要复杂数学推理的问题，当前系统很少能在首次尝试中成功，往往需要迭代调整其证明策略。现有的处理失败尝试的方法通常要么迭代地修复证明中的错误，要么丢弃整个证明并从头重新生成。前者会导致上下文逐渐变长，从而削弱模型对剩余未解决子问题的注意力；后者则效率低下，因为可能因局部错误而放弃大部分正确的推理。为解决这一困境，我们提出了MechMath，一个以Sorrifier驱动的形式化分解范式为核心的智能体系统。通过利用Lean中的sorry占位符来精确隔离未解决的子目标，同时保留其余正确的推理过程。

    arXiv:2603.24465v2 Announce Type: replace  Abstract: Recent advances in large language models (LLMs) and LLM-based agents have substantially improved the capabilities of automated theorem proving. However, for problems that require complex mathematical reasoning, current systems seldom succeed in their initial attempt, necessitating iterative adjustments to their proof strategies. Existing approaches for handling failed attempts typically either iteratively fix errors within the proof or discard the entire proof and regenerate it from scratch. The former leads to progressively longer contexts, which degrade the model's ability to attend to the remaining unresolved subproblems, while the latter is inefficient, as it may abandon mostly correct reasoning due to localized errors. To address this dilemma, we present MechMath, an agent system centered on a Sorrifier-driven formal decomposition paradigm. By leveraging the sorry placeholder in Lean to precisely isolate unresolved subgoals whil
    
[^208]: 更快、更便宜、更准确：专业化的知识追踪模型优于大型语言模型

    Faster, Cheaper, More Accurate: Specialised Knowledge Tracing Models Outperform LLMs

    [https://arxiv.org/abs/2603.02830](https://arxiv.org/abs/2603.02830)

    本文证明，专业化的知识追踪模型在预测学生未来反应方面，在准确性、部署成本和推理速度上全面优于大型语言模型，强调了领域专用模型在教育场景中的优势。

    

    arXiv:2603.02830v2 公告类型：替换-交叉 摘要：预测学生对问题的未来反应对于教育学习平台尤其有价值，因为它能够实现有效的干预。实现这一目标的关键方法之一是使用知识追踪（KT）模型。这些是针对特定领域的小型时间模型，基于学生的答题数据进行训练。KT模型针对特定教育领域的高精度进行了优化，具有快速推理和可扩展部署的优势。大型语言模型（LLMs）的兴起促使我们提出以下问题：（1）LLMs在预测学生未来问题反应方面表现如何？（2）LLMs在此领域是否可扩展？（3）LLMs在此特定领域任务上与KT模型相比如何？在本文中，我们通过预测性能、部署成本和推理速度比较了多种LLMs和KT模型，以回答上述问题。我们表明，KT模型在预测性能、部署成本和推理速度方面均优于LLMs，且资源消耗更少。

    arXiv:2603.02830v2 Announce Type: replace-cross  Abstract: Predicting future student responses to questions is particularly valuable for educational learning platforms where it enables effective interventions. One of the key approaches to do this has been through the use of knowledge tracing (KT) models. These are small, domain-specific, temporal models trained on student question-response data. KT models are optimised for high accuracy on specific educational domains and have fast inference and scalable deployments. The rise of Large Language Models (LLMs) motivates us to ask the following questions: (1) How well can LLMs perform at predicting students' future responses to questions? (2) Are LLMs scalable for this domain? (3) How do LLMs compare to KT models on this domain-specific task? In this paper, we compare multiple LLMs and KT models across predictive performance, deployment cost, and inference speed to answer the above questions. We show that KT models outperform LLMs with res
    
[^209]: 基于推理的稀疏数据用户个性化生成

    Reasoning-Based Personalized Generation for Users with Sparse Data

    [https://arxiv.org/abs/2602.21219](https://arxiv.org/abs/2602.21219)

    本文提出GraSPer框架，通过预测用户未来交互并生成合成文本来扩充稀疏上下文，从而提升LLM在冷启动等稀疏数据场景下的个性化生成能力。

    

    arXiv:2602.21219v2 公告类型：替换交叉 摘要：大型语言模型（LLM）的个性化在利用个人上下文和历史来定制响应方面具有巨大潜力。然而，现实世界中的用户通常拥有稀疏的交互历史，且个人上下文有限，例如社交平台上的冷启动用户和在线电子商务平台上的新注册客户，这削弱了基于LLM的个性化生成效果。为解决这一挑战，我们引入了GraSPer（基于图的稀疏个性化推理），一种在稀疏上下文下增强个性化文本生成的新框架。GraSPer首先通过预测用户未来可能交互的项目来扩充用户上下文。接着，通过推理对齐，它为这些交互生成文本以丰富扩充后的上下文。最后，它基于真实和合成历史生成个性化输出，确保与用户风格和偏好保持一致。

    arXiv:2602.21219v2 Announce Type: replace-cross  Abstract: Large Language Model (LLM) personalization holds great promise for tailoring responses by leveraging personal context and history. However, real-world users usually possess sparse interaction histories with limited personal context, such as cold-start users in social platforms and newly registered customers in online E-commerce platforms, compromising the LLM-based personalized generation. To address this challenge, we introduce GraSPer (Graph-based Sparse Personalized Reasoning), a novel framework for enhancing personalized text generation under sparse context. GraSPer first augments user context by predicting items that the user would likely interact with in the future. With reasoning alignment, it then generates texts for these interactions to enrich the augmented context. In the end, it generates personalized outputs conditioned on both the real and synthetic histories, ensuring alignment with user style and preferences. Ex
    
[^210]: HLE-Verified：对“人类最后考试”的系统性验证与结构化修订

    HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam

    [https://arxiv.org/abs/2602.13964](https://arxiv.org/abs/2602.13964)

    本文提出HLE-Verified，通过两阶段验证-修复流程和细粒度错误分类，系统性地验证并结构化修订了“人类最后考试”基准，确保评估结果的可靠性和跨模型比较的公平性。

    

    arXiv:2602.13964v4 公告类型：替换  摘要：“人类最后考试”（HLE）已成为评估前沿大型语言模型在具有挑战性、多领域问题上表现的一个广泛使用的基准。然而，社区主导的分析提出了担忧，即HLE包含相当数量的噪声条目，这可能使评估结果产生偏差，并扭曲跨模型的比较。为解决这一挑战，我们引入了HLE-Verified，一个经过验证和修订的HLE版本，具有透明的验证协议和细粒度的错误分类。我们的构建遵循一个两阶段的验证-修复工作流程，最终产生一个经过认证的基准。在第一阶段，每个条目通过领域专家审查和基于模型的交叉检查，对问题和最终答案进行二元验证，得到668个已验证的条目。在第二阶段，有缺陷但可修复的条目在严格约束下进行修订，以保留原始评估意图，通过双重独立专家修复和模型辅助审计。

    arXiv:2602.13964v4 Announce Type: replace  Abstract: Humanity's Last Exam (HLE) has become a widely used benchmark for evaluating frontier large language models on challenging, multi-domain questions. However, community-led analyses have raised concerns that HLE contains a non-trivial number of noisy items, which can bias evaluation results and distort cross-model comparisons. To address this challenge, we introduce HLE-Verified, a verified and revised version of HLE with a transparent verification protocol and fine-grained error taxonomy. Our construction follows a two-stage validation-and-repair workflow resulting in a certified benchmark. In Stage I, each item undergoes binary validation of the problem and final answer through domain-expert review and model-based cross-checks, yielding 668 verified items. In Stage II, flawed but fixable items are revised under strict constraints preserving the original evaluation intent, through dual independent expert repairs, model-assisted auditi
    
[^211]: 面向Web代理的代理式测试时扩展

    Agentic Test-Time Scaling for WebAgents

    [https://arxiv.org/abs/2602.12276](https://arxiv.org/abs/2602.12276)

    本文提出CATTS技术，通过动态分配计算资源和利用代理投票分布的不确定性统计，解决Web代理在多步任务中测试时扩展的收益递减问题。

    

    arXiv:2602.12276v2 公告类型：替换 摘要：测试时扩展已成为提高神经网络模型性能和可靠性的标准方法。然而，其在代理式、多步任务中的行为仍不太为人理解：小的逐步错误可能在长时间范围内累积；我们发现，统一增加采样的简单策略显示出收益递减。在这项工作中，我们提出了CATTS，一种为多步代理动态分配计算资源的简单技术。我们首先对Web代理的推理时扩展进行了实证研究。我们发现，在长时间范围内，统一增加每步计算量会迅速饱和。然后，我们研究了更强的聚合策略，包括一种基于LLM的仲裁器，它能优于简单投票，但可能推翻高共识决策。我们表明，从代理自身投票分布中得出的不确定性统计量（熵和top-1/top-2差距）与下游成功相关。

    arXiv:2602.12276v2 Announce Type: replace  Abstract: Test-time scaling has become a standard way to improve performance and boost reliability of neural network models. However, its behavior on agentic, multi-step tasks remains less well-understood: small per-step errors can compound over long horizons; and we find that naive policies that uniformly increase sampling show diminishing returns. In this work, we present CATTS, a simple technique for dynamically allocating compute for multi-step agents. We first conduct an empirical study of inference-time scaling for web agents. We find that uniformly increasing per-step compute quickly saturates in long-horizon environments. We then investigate stronger aggregation strategies, including an LLM-based Arbiter that can outperform naive voting, but that can overrule high-consensus decisions. We show that uncertainty statistics derived from the agent's own vote distribution (entropy and top-1/top-2 margin) correlate with downstream success and
    
[^212]: 从学生-导师对话中诊断误解：生成、检索、重排序

    Misconception Diagnosis From Student-Tutor Dialogue: Generate, Retrieve, Rerank

    [https://arxiv.org/abs/2602.02414](https://arxiv.org/abs/2602.02414)

    本文提出了一种结合生成、检索和重排序策略的LLM方法，用于从学生-导师对话中自动识别误解，显著提升了预测性能。

    

    arXiv:2602.02414v2 公告类型：替换 摘要：及时准确地识别学生的误解是改善学习成果和防止学生错误累积的关键。然而，这项任务高度依赖于教师的努力和直觉。在这项工作中，我们提出了一种新颖的方法，利用大型语言模型（LLMs）从学生-导师对话中检测误解。首先，我们使用一个微调的LLM生成可能的误解，然后通过嵌入相似性与输入对话检索其中最有可能的候选者。这些候选者随后由另一个微调的LLM评估并重新排序，以提高误解的相关性。在实证方面，我们在教育辅导平台上的真实对话中评估了我们的系统。我们考虑了多种基础LLM模型，包括LLaMA、Qwen和Claude，在零样本和微调设置下。我们发现，我们的方法在预测性能上优于基线模型，并且优于现有方法。

    arXiv:2602.02414v2 Announce Type: replace  Abstract: Timely and accurate identification of student misconceptions is key to improving learning outcomes and pre-empting the compounding of student errors. However, this task is highly dependent on the effort and intuition of the teacher. In this work, we present a novel approach for detecting misconceptions from student-tutor dialogues using large language models (LLMs). First, we use a fine-tuned LLM to generate plausible misconceptions, and then retrieve the most promising candidates among these using embedding similarity with the input dialogue. These candidates are then assessed and re-ranked by another fine-tuned LLM to improve misconception relevance. Empirically, we evaluate our system on real dialogues from an educational tutoring platform. We consider multiple base LLM models including LLaMA, Qwen and Claude on zero-shot and fine-tuned settings. We find that our approach improves predictive performance over baseline models and th
    
[^213]: 顺序发布大型语言模型在受监管市场中助长操纵行为

    Sequential LLM Release Facilitates Manipulation in Regulated Markets

    [https://arxiv.org/abs/2601.11496](https://arxiv.org/abs/2601.11496)

    本文通过GLEE基准数据发现，顺序发布大型语言模型在市场中可能产生“毒苹果”效应，即发布的模型虽未被采用，却会改变均衡结果，导致一方收益增加而另一方受损，从而助长市场操纵。

    

    摘要：AI代理日益成为个人和企业进行议价、谈判和说服的中介。这类市场扩展了软件中介商务，但带来了治理问题：独立的模型发布改变了参与者可用的代理。博弈论表明，扩展策略集可能损害均衡结果，但主要通过构造示例。已部署的AI代理日志稀缺、专有且涉及隐私，缺乏反事实和收益标签。因此，我们使用GLEE（一个独立收集的基准数据集，包含13个大型语言模型在1,320个匹配的议价、谈判和说服配置中的587K个战略决策）来研究模型发布作为策略扩展。在超过50,000次发布比较中，许多发布使收益向相反方向移动：一个代理获利而另一个损失。我们识别出“毒苹果”效应：一个发布的模型在均衡中没有任何代理采用。

    arXiv:2601.11496v3 Announce Type: replace-cross  Abstract: AI agents increasingly mediate bargaining, negotiation and persuasion for people and firms. Such markets extend software-mediated commerce, but add a governance problem: independent model releases change delegates available to participants. Game theory shows that expanding a strategy set can harm equilibrium outcomes, but mostly through constructed examples. Deployed AI-agent logs are scarce, proprietary and privacy-sensitive, and lack counterfactuals and payoff labels. We therefore use GLEE, an independently collected benchmark of 587K strategic decisions by 13 large language models across 1,320 matched bargaining, negotiation and persuasion configurations, to study model release as strategy expansion. Across more than 50{,}000 release comparisons, many releases move payoffs in opposite directions: one agent gains while the other loses. We identify the Poisoned Apple effect: a released model that no agent adopts in equilibrium
    
[^214]: AWED-PIPER：面向36种语言、66亿说话者的个人身份信息保护与细粒度命名实体识别的代理、Web应用与专家检测器

    AWED-PIPER: Agents, Web Applications & Expert Detectors for Personally Identifiable Information Protection & Fine-grained Named Entity Recognition across 36 languages for 6.6 Billion Speakers

    [https://arxiv.org/abs/2601.10161](https://arxiv.org/abs/2601.10161)

    该论文提出了AWED-PIPER框架，通过54个专家模型和代理工具，在36种语言中实现细粒度命名实体识别和可逆PII匿名化，兼顾信息提取与隐私保护。

    

    命名实体识别（NER）和个人身份信息（PII）匿名化是自然语言处理（NLP）中信息提取和隐私保护的关键任务。我们介绍了AWED-PIPER，一个开源框架，包含代理工具、交互式Web应用和54个最先进的专家检测模型，提供统一的细粒度命名实体识别（FgNER）和可逆的合成PII假名化，覆盖36种语言，使用者超过66亿人。该系统将细粒度多语言序列标注与脚本感知的正则表达式检测器相结合，以识别上下文实体（人物、地点、组织、医疗）以及结构化技术PII（电子邮件、本地脚本电话号码、IP地址、信用卡）。AWED-PIPER提供双重能力：完整的FgNER实体提取和隐私保护的可逆匿名化，具有持久占位符。

    arXiv:2601.10161v3 Announce Type: replace-cross  Abstract: Named Entity Recognition (NER) and Personally Identifiable Information (PII) anonymization are critical tasks in Natural Language Processing (NLP) for information extraction and privacy preservation. We introduce AWED-PIPER, an open-source framework comprising agentic tools, interactive web applications, and 54 state-of-the-art expert detector models that provide unified Fine-grained Named Entity Recognition (FgNER) and reversible synthetic PII pseudonymization across 36 languages spoken by over 6.6 billion people. The system couples fine-grained multilingual sequence labeling with script-aware regex detectors to identify contextual entities (Person, Location, Organization, Medical) as well as structured technical PII (Emails, native-script Phone Numbers, IP Addresses, Credit Cards). AWED-PIPER offers a dual capability: full FgNER entity extraction and privacy-preserving reversible anonymization with persistent placeholders and
    
[^215]: QA-Merging：通过层选择性模型合并实现查询自适应推理

    QA-Merging: Query-Adaptive Reasoning via Layer Selective Model Merging

    [https://arxiv.org/abs/2601.03506](https://arxiv.org/abs/2601.03506)

    提出一种基于激活的查询自适应层选择性合并框架，无需重新训练即可动态选择层，实现高效的自适应推理，兼顾长思维链与短思维链的优势。

    

    arXiv:2601.03506v2 公告类型：替换交叉 摘要：近期的大型推理模型（LRMs）通过生成冗长的思维链（Long-CoT），在复杂推理任务上取得了强劲性能。然而，对于简单查询，这种冗长的推理往往是不必要的，导致额外的计算和延迟。现有的自适应推理方法通常依赖于重新训练模型或设计复杂的提示工程，这些方法要么成本过高，要么对提示表述高度敏感。模型合并为自适应推理提供了一种更平衡的替代方案，通过避免昂贵的训练并整合Long-CoT和Short-CoT行为。然而，现有的合并方法往往是静态且与输入无关的，或者依赖昂贵的全层校准，这限制了它们在查询自适应推理中的有效性。为解决这些挑战，我们提出了查询自适应层选择性合并（QA-Merging），一种基于激活的合并框架，该框架在推理过程中动态选择层。

    arXiv:2601.03506v2 Announce Type: replace-cross  Abstract: Recent large reasoning models (LRMs) have achieved strong performance on complex reasoning tasks by generating a long chain-of-thought (Long-CoT). However, such lengthy reasoning is often unnecessary for simple queries, leading to additional computation and latency. Existing approaches to adaptive reasoning typically rely on retraining the model or designing sophisticated prompting, which are either prohibitively expensive or highly sensitive to the prompt formulation. Model merging provides a more balanced alternative for adaptive reasoning by avoiding expensive training and integrating Long-CoT and Short-CoT behaviors. However, existing merging methods are often static and input-agnostic, or rely on costly all-layer calibration, which limits their effectiveness for query-adaptive reasoning. To tackle these challenges, we propose Query-adaptive Layer Selective Merging (QA-Merging), an activation-based merging framework that in
    
[^216]: jina-vlm：小型多语言视觉语言模型

    jina-vlm: Small Multilingual Vision Language Model

    [https://arxiv.org/abs/2512.04032](https://arxiv.org/abs/2512.04032)

    本文提出jina-vlm，一个24亿参数的多语言视觉语言模型，通过图像分块和注意力池化实现令牌高效处理，并在2B规模模型中达到最先进的多语言VQA性能，同时通过消融研究揭示了训练数据类别的影响。

    

    摘要：arXiv:2512.04032v4 公告类型：替换交叉。我们提出了jina-vlm，一个参数高效、拥有24亿参数的视觉语言模型，在开放的20亿规模视觉语言模型中，实现了最先进的多语言视觉问答性能。该模型将SigLIP2视觉编码器与Qwen3语言解码器相结合，并利用图像分块和注意力池化技术，实现对任意分辨率图像的令牌高效处理。为了理解不同训练数据类别的贡献，我们进行了留一法数据混合消融研究——系统地移除任务、领域、模态和语言类别——以诊断哪些数据类型是必要的而非冗余的，以及任务收益是否跨领域转移。模型权重和代码已在https://huggingface.co/jinaai/jina-vlm公开发布。

    arXiv:2512.04032v4 Announce Type: replace-cross  Abstract: We present jina-vlm, a token-efficient 2.4B parameter vision-language model that achieves state-of-the-art multilingual VQA performance among open 2B-scale VLMs. The model couples a SigLIP2 vision encoder with a Qwen3 language decoder and makes use of image tiling and attention-pooling for token-efficient processing of arbitrary-resolution images. To understand the contribution of different training data categories, we conduct a leave-one-out data mixture ablation study-systematically removing task, domain, modality, and language categories-to diagnose which data types are necessary versus redundant and whether task benefits transfer across domains. Model weights and code are publicly released at https://huggingface.co/jinaai/jina-vlm.
    
[^217]: 大规模中文知识图谱-文本对齐数据集，用于基准测试知识增强的大语言模型

    A Large-Scale Chinese Knowledge Graph-Text Alignment Dataset for Benchmarking Knowledge-Grounded LLMs

    [https://arxiv.org/abs/2510.06039](https://arxiv.org/abs/2510.06039)

    本文提出了CDTP，一个包含700万实例和1500万三元组的大规模中文知识图谱-文本对齐数据集，通过多阶段构建流程确保语义和事实准确性，用于基准测试知识增强的大语言模型在结构化推理中的表现。

    

    摘要：在中文环境中可靠评估知识增强的大语言模型（LLMs）需要显式对齐中文文本与可验证知识图谱（KG）事实的资源。然而，现有的中文基准主要评估通用语言理解能力，对中文特定语言现象下的结构化推理支持有限。我们引入了中文数据-文本对（CDTP），这是一个大规模的中文知识图谱-文本对齐数据集，包含超过700万个对齐实例，覆盖四个广泛领域。每个实例将一段中文文本与一个或多个文本支持的KG三元组配对，总计1500万个三元组。一个结合对齐过滤、人工验证和外部证据验证的多阶段构建流程提高了语义一致性和事实可靠性。CDTP支持知识图谱补全（KGC）、问答（QA）和三元组到文本生成（T2）任务。

    arXiv:2510.06039v2 Announce Type: replace-cross  Abstract: Reliable evaluation of knowledge-grounded Large Language Models (LLMs) in Chinese requires resources that explicitly align Chinese-language text with verifiable Knowledge Graph (KG) facts. Yet existing Chinese benchmarks primarily assess general language understanding and offer limited support for structured reasoning under Chinese-specific linguistic phenomena. We introduce the Chinese Data-Text Pair (CDTP), a large-scale Chinese KG-text alignment dataset comprising more than 7 million aligned instances across four broad domains. Each instance pairs a Chinese-language text with one or more textually supported KG triples, totaling 15 million triples. A multi-stage construction pipeline combining alignment filtering, manual verification, and external evidence validation improves semantic consistency and factual reliability. CDTP supports Knowledge Graph Completion (KGC), Question Answering (QA), and Triple-to-Text Generation (T2
    
[^218]: SimulRAG：基于模拟器的RAG框架，用于在长篇科学问答中锚定大语言模型

    SimulRAG: Simulator-based RAG for Grounding LLMs in Long-form Scientific QA

    [https://arxiv.org/abs/2509.25459](https://arxiv.org/abs/2509.25459)

    本文提出了SimulRAG，一种基于科学模拟器的RAG框架，通过通用检索接口和声明级生成与不确定性估计，有效解决了长篇科学问答中的幻觉问题。

    

    大语言模型（LLMs）在生成长篇科学解释方面展现出潜力，这些解释能够综合证据并连接多个因素。然而，在长篇科学问答中，LLMs经常产生幻觉，生成无依据或不一致的陈述。检索增强生成（RAG）通过将生成过程锚定在外部来源中，提高了可信度；科学模拟器因其能够验证定量假设并捕捉动态演变过程而具有价值。然而，基于模拟的RAG面临两大挑战：如何从科学模拟器中检索信息，以及如何高效地验证和更新长篇答案。为克服这些挑战，我们提出了SimulRAG，一种基于模拟器的RAG框架，其具有通用检索接口，可在文本与模拟器参数/输出之间进行转换。SimulRAG还引入了带有不确定性估计和模拟器边界的声明级生成机制。

    arXiv:2509.25459v3 Announce Type: replace  Abstract: Large Language Models (LLMs) show promise in generating long-form scientific explanations that synthesize evidence and connect multiple factors. However, in long-form scientific question answering, LLMs often hallucinate, producing unsupported or inconsistent claims. Retrieval-Augmented Generation (RAG) improves trustworthiness by grounding generation in external sources; scientific simulators are valuable because they can validate quantitative hypotheses and capture evolving dynamics. Yet simulation-based RAG is non-trivial due to two challenges: how to retrieve from scientific simulators, and how to efficiently verify and update long-form answers. To overcome these challenges, we propose SimulRAG, a simulator-based RAG framework with a generalized retrieval interface that translates between text and simulator parameters/outputs. SimulRAG further introduces claim-level generation with uncertainty estimation and simulator boundary as
    
[^219]: 视觉语言模型无法规划，但它们能进行形式化吗？

    Vision Language Models Cannot Plan, but Can They Formalize?

    [https://arxiv.org/abs/2509.21576](https://arxiv.org/abs/2509.21576)

    本文提出了五种VLM作为形式化器的流水线，用于一次性、开放词汇和多模态PDDL规划形式化，以解决VLM在长期规划中的不足。

    

    arXiv:2509.21576v2 公告类型：替换 摘要：视觉语言模型（VLMs）的进步使具身代理能够完成简单的多模态规划任务，但无法完成需要长序列动作的长期规划任务。在纯文本模拟中，通过重新定位大型语言模型（LLMs）的角色，长期规划已取得显著改进。LLMs 不是直接生成动作序列，而是将规划领域和问题转化为形式化规划语言（如规划域定义语言PDDL），从而可以调用形式化求解器以可验证的方式推导出规划方案。在多模态环境中，关于VLM作为形式化器的研究仍然稀缺，通常涉及粗略简化，如预定义对象词汇或过度相似的少样本示例。在本工作中，我们提出了一套五种VLM作为形式化器的流水线，用于处理一次性、开放词汇和多模态PDDL形式化。我们在现有基准上对这些流水线进行了评估。

    arXiv:2509.21576v2 Announce Type: replace  Abstract: The advancement of vision language models (VLMs) has empowered embodied agents to accomplish simple multimodal planning tasks, but not long-horizon ones requiring long sequences of actions. In text-only simulations, long-horizon planning has seen significant improvement brought by repositioning the role of LLMs. Instead of directly generating action sequences, LLMs translate the planning domain and problem into a formal planning language like the Planning Domain Definition Language (PDDL), which can call a formal solver to derive the plan in a verifiable manner. In multimodal environments, research on VLM-as-formalizer remains scarce, usually involving gross simplifications such as predefined object vocabulary or overly similar few-shot examples. In this work, we present a suite of five VLM-as-formalizer pipelines that tackle one-shot, open-vocabulary, and multimodal PDDL formalization. We evaluate those on an existing benchmark whil
    
[^220]: 知道何时暂缓：面向负责任知识追踪的选择性预测

    Knowing When to Defer: Selective Prediction for Responsible Knowledge Tracing

    [https://arxiv.org/abs/2509.21514](https://arxiv.org/abs/2509.21514)

    本文提出了一种基于MC-Dropout的内置选择性预测层，使现有知识追踪模型能够智能暂缓不确定预测，在无需重训练的情况下显著提升准确性和公平性。

    

    知识追踪（KT）模型的研究传统上专注于提高预测准确性。然而，负责任的现实世界部署要求模型知道何时将不确定的预测暂缓给人类教师。我们引入了一种内在的选择性预测层，用于现有KT模型，使用蒙特卡洛丢弃法（MC-Dropout）来量化不确定性。我们使用Eedi数学数据集，在三种架构（DKT、SAKT和AKT）上评估了该方法。对最不确定的20%预测进行弃权，无需任何重新训练，准确率提升了2.3至3.0个百分点，AUC提升了1.9至2.4个百分点，F1提升了1.4至4.3个百分点。这种弃权策略高度针对性：暂缓集表现出保留集错误率的1.45至1.60倍。此外，这种针对性在每个题目难度四分位数内均成立，并在学生能力水平间保持公平。重要的是，MC-Dropout方差...

    arXiv:2509.21514v4 Announce Type: replace-cross  Abstract: Research on Knowledge Tracing (KT) models traditionally focuses on improving predictive accuracy. However, responsible real-world deployment requires models to know when to defer uncertain predictions to a human teacher. We introduce an intrinsic selective prediction layer for existing KT models using Monte Carlo Dropout (MC-Dropout) to quantify uncertainty. We evaluate this approach across three architectures (DKT, SAKT, and AKT) using the Eedi mathematics dataset. Abstaining on the 20\% most uncertain predictions lifts accuracy by 2.3 to 3.0 percentage points, AUC by 1.9 to 2.4 percentage points and F1 by 1.4 to 4.3 percentage points without any retraining. This abstention strategy is highly targeted: the deferred set exhibits 1.45 to 1.60 times the error rate of the kept set. Furthermore, this targeting holds within every question-difficulty quartile and remains fair across student-ability levels. Importantly, MC-Dropout var
    
[^221]: 能够思考与更好对话的语言模型

    Language Models that Think, Chat Better

    [https://arxiv.org/abs/2509.20357](https://arxiv.org/abs/2509.20357)

    本文提出RLMT方法，通过引入模型奖励思考的强化学习，将长链推理扩展到开放式任务，显著提升语言模型的通用对话能力。

    

    摘要：arXiv:2509.20357v2 公告类型：替换 摘要：带有可验证奖励的强化学习（RLVR）训练语言模型在数学和代码等领域使用长链思维推理（CoT），并基于规则验证器。然而，通过RLVR学习到的长CoT并不能很好地推广到开放式任务——例如撰写论文大纲或制定膳食计划——这些任务中人类会常规性地进行推理。本文确立了长CoT对通用对话能力的好处，并引入了带模型奖励思考的强化学习（RLMT），将RLVR扩展到可验证领域之外。利用多样化的真实世界提示，RLMT要求语言模型在响应前生成长CoT推理，并通过在线强化学习针对RLHF中使用的基于偏好的奖励模型进行优化。在Llama-3.1-8B和Qwen-2.5-7B（基础和指令版本）上的40次训练运行中，以及多种优化算法（DPO、PPO和GRPO）下，RLMT始终优于标准RLHF流程。这包括在……

    arXiv:2509.20357v2 Announce Type: replace  Abstract: Reinforcement learning with verifiable rewards (RLVR) trains language models to use long chain-of-thought reasoning (CoT) in domains like mathematics and code with rule-based verifiers. However, long CoT learned through RLVR does not generalize well to open-ended tasks -- such as writing essay outlines or making meal plans -- where humans reason routinely. This paper establishes the benefits of long CoT for general-purpose chat capabilities and introduces RL with Model-rewarded Thinking (RLMT)1, which pushes RLVR beyond verifiable domains. Using diverse real-world prompts, RLMT requires LMs to generate long CoT reasoning before responding, and optimizes them with online RL against a preference-based reward model used in RLHF. Across 40 training runs on Llama-3.1-8B and Qwen-2.5-7B (both base and instruct) and multiple optimization algorithms (DPO, PPO, and GRPO), RLMT consistently outperforms standard RLHF pipelines. This includes su
    
[^222]: 从代码生成模型中提取高效代码嵌入

    Efficient Code Embeddings from Code Generation Models

    [https://arxiv.org/abs/2508.21290](https://arxiv.org/abs/2508.21290)

    本文提出了一种基于自回归模型和最后令牌池化的代码嵌入方法，在小型模型上实现了跨语言代码检索和问答的最先进性能。

    

    arXiv:2508.21290v2 公告类型：替换交叉 摘要：jina-code-embeddings 是一个新型代码嵌入模型套件，旨在从自然语言查询中检索代码、执行技术问答，并识别跨编程语言的语义相似代码片段。它创新性地使用了在文本和代码上预训练的自回归骨干网络，通过最后令牌池化生成嵌入。我们概述了训练方案，并展示了尽管模型规模相对较小，仍取得了最先进的性能，验证了这种代码嵌入模型构建方法的有效性。

    arXiv:2508.21290v2 Announce Type: replace-cross  Abstract: jina-code-embeddings is a novel code embedding model suite designed to retrieve code from natural language queries, perform technical question-answering, and identify semantically similar code snippets across programming languages. It makes innovative use of an autoregressive backbone pre-trained on both text and code, generating embeddings via last-token pooling. We outline the training recipe and demonstrate state-of-the-art performance despite the relatively small size of the models, validating this approach to code embedding model construction.
    
[^223]: PEER：面向结构化共情推理的统一过程-结果强化学习

    PEER: Unified Process-Outcome Reinforcement Learning for Structured Empathetic Reasoning

    [https://arxiv.org/abs/2508.09521](https://arxiv.org/abs/2508.09521)

    本文提出了PEER框架，通过结构化共情推理和统一的过程-结果奖励机制，解决了情绪支持对话中缺乏心理学推理和强化学习奖励信号不可靠的问题。

    

    情绪支持对话不仅需要流畅的回应。支持者需要理解求助者的处境和情绪，采用适当的策略，并以自然、类人的方式回应。尽管大语言模型取得了进展，但当前系统往往缺乏结构化、基于心理学的推理。此外，由于奖励信号不可靠，通过强化学习增强这些系统具有挑战性。而且，强化微调可能放大重复的回应模式。我们提出结构化共情推理，将支持过程分解为三个步骤：对话历史分析、多模态情绪状态推断和策略选择，然后生成最终回复。为实现这一点，我们引入了SER，一个带有步骤级正确性标签和成对回应偏好的细粒度数据集。然后，我们提出PEER，它使用带有UnifiReward的GRPO，这是一种统一的过程-结果奖励机制。

    arXiv:2508.09521v3 Announce Type: replace-cross  Abstract: Emotional support conversations require more than fluent responses. Supporters need to understand the seeker's situation and emotions, adopt an appropriate strategy, and respond in a natural, human-like manner. Despite advances in large language models, current systems often lack structured, psychology-informed reasoning. Additionally, it is challenging to enhance these systems through reinforcement learning because of unreliable reward signals. Moreover, reinforcement fine-tuning can amplify repetitive response patterns. We propose structured empathetic reasoning, which breaks support into three steps: conversation history analysis, multimodal emotional state inference, and strategy selection, prior to generating the final reply. To implement this, we introduce SER, a fine-grained dataset with step-level correctness labels and pairwise response preferences. We then present PEER, which uses GRPO with UnifiReward, a unified proc
    
[^224]: CulTrace：追踪大型语言模型中的内部文化推理

    CulTrace: Tracing Internal Cultural Reasoning in Large Language Models

    [https://arxiv.org/abs/2508.08879](https://arxiv.org/abs/2508.08879)

    本文提出CulTrace方法，通过机械可解释性揭示大型语言模型在文化问答中内部文化推理的分阶段轨迹，并发现其推理存在不平衡性。

    

    大型语言模型在不同文化背景中的部署日益增多，这要求我们更深入地理解模型对不同文化的隐藏表征。以往的研究通过分析模型输出评估其文化意识，但这种方法忽视了文化在模型参数中的表示方式，无法解释模型为何产生错误回答。为弥补这一空白，我们提出了CulTrace，一种基于机械可解释性的方法，用于探测大型语言模型内部表征中的文化知识。通过CulTrace，我们检查了文化知识在层间如何处理，以及在文化问答中如何被整合。我们发现文化推理存在一致的阶段性轨迹：模型首先处理问题的领域，然后解析相关文化，最后聚焦于答案。我们还证明模型的文化推理存在不平衡性，表现出延迟的相关性。

    arXiv:2508.08879v3 Announce Type: replace-cross  Abstract: The growing deployment of large language models (LLMs) across diverse cultural contexts necessitates a deeper understanding of models' hidden representations of different cultures. Prior work has evaluated cultural awareness in LLMs by analysing their outputs. This approach overlooks how cultures are represented within the model parameters, missing why models generate incorrect responses. To bridge this gap, we propose CulTrace, a mechanistic interpretability-based method that probes the internal representations of LLMs for cultural knowledge. With CulTrace, we inspect how cultural knowledge is processed across layers and how it is integrated during cultural QA. We find a consistent staged trajectory of cultural reasoning. Models first engage with the question's domain, then resolve the relevant culture, and finally narrow in on an answer. We also demonstrate that models' cultural reasoning is imbalanced, showing delayed releva
    
[^225]: 通过时间异质性建模与表示对齐将大语言模型适配于时间序列预测

    Adapting LLMs to Time Series Forecasting via Temporal Heterogeneity Modeling and Representation Alignment

    [https://arxiv.org/abs/2508.07195](https://arxiv.org/abs/2508.07195)

    本文提出了TALON框架，通过异质时间编码器和表示对齐模块，分别解决时间模式异质性和模态差距问题，从而提升大语言模型在时间序列预测中的性能。

    

    arXiv:2508.07195v2 公告类型：替换交叉 摘要：近期进展表明，大语言模型（LLMs）可以被有效适配用于时间序列预测，展现出超越自然语言任务的强大潜力。然而，其性能仍受两个基本挑战的制约：时间模式固有的异质性，以及连续数值信号与离散语言表示之间的模态差距。在这项工作中，我们提出了TALON（时间异质性与语言导向网络），一个统一框架，通过建模时间异质性并促进表示对齐来增强基于LLM的预测。具体来说，我们设计了一个异质时间编码器，将多变量时间序列划分为结构连贯的片段，从而在不同时间模式上实现局部专家建模。为弥合模态差距，我们引入了一个表示对齐模块，将时间特征投影到语言空间中。

    arXiv:2508.07195v2 Announce Type: replace-cross  Abstract: Recent advances have demonstrated that Large Language Models (LLMs) can be effectively adapted for time series forecasting, revealing strong potential beyond natural language tasks. However, their performance remains constrained by two fundamental challenges: the inherent heterogeneity of temporal patterns and the modality gap between continuous numerical signals and discrete language representations. In this work, we propose \textbf{TALON} (Temporal-heterogeneity And Language-Oriented Network), a unified framework that enhances LLM-based forecasting by modeling temporal heterogeneity and promoting representation alignment. Specifically, we design a Heterogeneous Temporal Encoder that partitions multivariate time series into structurally coherent segments, enabling localized expert modeling across diverse temporal patterns. To bridge the modality gap, we introduce a Representation Alignment Module that projects temporal feature
    
[^226]: FollowUpBot：一种基于大语言模型的对话机器人，用于自动术后随访

    FollowUpBot: An LLM-Based Conversational Robot for Automatic Postoperative Follow-up

    [https://arxiv.org/abs/2507.15502](https://arxiv.org/abs/2507.15502)

    本文提出FollowUpBot，一种边缘部署的大语言模型驱动的术后随访机器人，通过动态路径规划和多模态对话实现自适应、隐私保护的自动随访，并自动生成结构化报告。

    

    术后随访在监测恢复和识别并发症中起着关键作用。然而，传统方法通常涉及床边访谈和手动记录，耗时且费力。尽管现有的数字解决方案，如网络问卷和智能自动呼叫，可以在一定程度上减轻护士的工作负担，但它们要么提供僵化的脚本化交互，要么面临私人信息泄露问题。为解决这些局限性，本文介绍了FollowUpBot，一种由大语言模型驱动的边缘部署机器人，用于术后护理和监测。它能够动态规划最优路径，并通过多种交互模式使用边缘部署的大语言模型与患者进行自适应、面对面的对话，确保数据隐私。此外，FollowUpBot能够自动生成结构化的术后随访报告，供医疗保健机构使用。

    arXiv:2507.15502v1 Announce Type: cross  Abstract: Postoperative follow-up plays a crucial role in monitoring recovery and identifying complications. However, traditional approaches, typically involving bedside interviews and manual documentation, are time-consuming and labor-intensive. Although existing digital solutions, such as web questionnaires and intelligent automated calls, can alleviate the workload of nurses to a certain extent, they either deliver an inflexible scripted interaction or face private information leakage issues. To address these limitations, this paper introduces FollowUpBot, an LLM-powered edge-deployed robot for postoperative care and monitoring. It allows dynamic planning of optimal routes and uses edge-deployed LLMs to conduct adaptive and face-to-face conversations with patients through multiple interaction modes, ensuring data privacy. Moreover, FollowUpBot is capable of automatically generating structured postoperative follow-up reports for healthcare ins
    
[^227]: 一种面向心理学中稳健大语言模型研究的效度引导工作流程

    A validity-guided workflow for robust large language model research in psychology

    [https://arxiv.org/abs/2507.04491](https://arxiv.org/abs/2507.04491)

    本文提出一个六阶段效度引导工作流程，通过将效度要求与研究雄心相匹配，解决大语言模型在心理学研究中的测量不可靠性问题，以防止“测量幻影”威胁研究效度。

    

    大语言模型（LLMs）正迅速被整合到心理学和行为研究中，用作研究工具、评估目标、人类模拟器和认知模型。然而，近期证据揭示了严重的测量不可靠性：人格评估在因子分析下退化，道德偏好因标点符号变化而反转，心理理论准确性随琐碎措辞改变而大幅波动。这些“测量幻影”——伪装成心理现象的统计伪影——威胁着日益增长的研究领域的效度。在整合心理测量学与因果推断的双重效度框架指导下，我们提出了一个六阶段工作流程，将效度要求与研究的雄心相匹配——使用LLMs编码文本需要基本的可靠性和准确性，而关于心理属性的主张则要求全面的构念验证。研究者必须（1）明确定义其研究目标。

    arXiv:2507.04491v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are rapidly being integrated into psychological and behavioral research as research tools, evaluation targets, human simulators, and cognitive models. Yet recent evidence reveals severe measurement unreliability: personality assessments degenerate under factor analysis, moral preferences reverse with punctuation changes, and theory-of-mind accuracy varies widely with trivial rephrasing. These "measurement phantoms"--statistical artifacts masquerading as psychological phenomena--threaten the validity of a growing body of research. Guided by the dual-validity framework that integrates psychometrics with causal inference, we present a six-stage workflow that scales validity requirements to research ambition--using LLMs to code text requires basic reliability and accuracy, whereas claims about psychological properties demand comprehensive construct validation. Researchers must (1) explicitly define thei
    
[^228]: 从提示到构念：大语言模型在心理学研究中的双重效度框架

    From Prompts to Constructs: A Dual-Validity Framework for Large Language Model Research in Psychology

    [https://arxiv.org/abs/2506.16697](https://arxiv.org/abs/2506.16697)

    本文提出了一个双重效度框架，强调在将大语言模型用于心理学研究时，需结合心理测量学验证和因果推断标准，以避免“测量幻象”并确保研究结论的科学有效性。

    

    大语言模型（LLMs）正作为工具和研究对象进入心理学研究领域。然而，许多研究将人类测量工具直接应用于LLMs，却未验证其输出的可靠性或可解释性，这增加了“测量幻象”的风险——即把统计规律误认为真实的心理现象。本文认为，稳健的AI心理学研究需要整合两种方法论传统：对分数含义的心理测量学验证，以及对结果推论所依据的因果推断标准。本文提出了一个双重效度框架，其中证据需求随科学目标而升级：从工具使用，到行为特征描述，再到人类模拟和认知建模。对文本进行分类可能只需准确性和可靠性；而声称LLM模拟焦虑或阐明认知机制，则需要额外证据，包括构念效度等。

    arXiv:2506.16697v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are entering psychological research both as tools and as objects of inquiry. Yet many studies apply human instruments to LLMs without establishing that the outputs are reliable or interpretable, raising the risk of measurement phantoms--statistical regularities mistaken for genuine psychological phenomena. This review argues that robust AI psychological research requires integrating two methodological traditions: psychometric validation of what a score means and causal inference standards for what the results warrant. It develops a dual-validity framework in which evidentiary demands scale with scientific ambition: from tool use through behavioral characterization and human simulation to cognitive modeling. Classifying text may require only accuracy and reliability; claiming that an LLM simulates anxiety or illuminates cognitive mechanisms requires additional evidence, including construct validity e
    
[^229]: LlamaRec-LKG-RAG：一种用于基于LLM排序的单遍可学习知识图谱-RAG框架

    LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking

    [https://arxiv.org/abs/2506.07449](https://arxiv.org/abs/2506.07449)

    本论文提出了一种单遍、端到端可训练的知识图谱增强RAG框架，通过轻量级用户偏好模块提取个性化关系路径并整合到LLM提示中，从而提升推荐排序的准确性和可解释性。

    

    arXiv:2506.07449v2 公告类型：替换交叉 摘要：大型语言模型（LLMs）的最新进展通过检索增强生成（RAG）框架推动了其在推荐系统中的应用。然而，现有的RAG方法主要依赖基于相似性的扁平检索，未能充分利用用户-物品交互中固有的丰富关系结构。我们引入了LlamaRec-LKG-RAG，一种新颖的单遍、端到端可训练框架，将个性化知识图谱上下文整合到基于LLM的推荐排序中。我们的方法通过引入一个轻量级用户偏好模块扩展了LlamaRec架构，该模块在从用户行为和物品元数据构建的异质知识图谱中识别出显著的关系路径。这些个性化子图被无缝整合到经过微调的Llama-2模型的提示中，通过统一的推理步骤实现高效且可解释的推荐。综合实验表明，该框架在推荐准确性方面显著优于现有基线，同时保持了计算效率。

    arXiv:2506.07449v2 Announce Type: replace-cross  Abstract: Recent advances in Large Language Models (LLMs) have driven their adoption in recommender systems through Retrieval-Augmented Generation (RAG) frameworks. However, existing RAG approaches predominantly rely on flat, similarity-based retrieval that fails to leverage the rich relational structure inherent in user-item interactions. We introduce LlamaRec-LKG-RAG, a novel single-pass, end-to-end trainable framework that integrates personalized knowledge graph context into LLM-based recommendation ranking. Our approach extends the LlamaRec architecture by incorporating a lightweight user preference module that identifies salient relation paths within a heterogeneous knowledge graph constructed from user behavior and item metadata. These personalized subgraphs are seamlessly integrated into prompts for a fine-tuned Llama-2 model, enabling efficient and interpretable recommendations through a unified inference step. Comprehensive expe
    
[^230]: 诊断竞技场：大型语言模型诊断推理能力的基准测试

    DiagnosisArena: Benchmarking Diagnostic Reasoning for Large Language Models

    [https://arxiv.org/abs/2505.14107](https://arxiv.org/abs/2505.14107)

    本文提出了诊断竞技场（DiagnosisArena），一个基于1,113对临床病例、覆盖28个专科的全面基准，用于系统评估大型语言模型的专业级诊断推理能力。

    

    摘要：能够执行复杂推理任务的突破性大型语言模型的出现，为解决各种科学挑战（包括复杂临床场景中的挑战）带来了巨大希望。为了使其在现实医疗环境中安全有效地部署，迫切需要系统地基准测试当前模型的诊断能力。鉴于现有医学基准在评估高级诊断推理方面的局限性，我们提出了诊断竞技场（DiagnosisArena），这是一个全面且具有挑战性的基准，旨在严格评估专业级诊断能力。诊断竞技场包含1,113对分段患者病例及相应诊断，覆盖28个医学专科，数据来源于10种顶级医学期刊发表的临床病例报告。该基准通过精心构建的流程开发，涉及多阶段处理。

    arXiv:2505.14107v5 Announce Type: replace-cross  Abstract: The emergence of groundbreaking large language models capable of performing complex reasoning tasks holds significant promise for addressing various scientific challenges, including those arising in complex clinical scenarios. To enable their safe and effective deployment in real-world healthcare settings, it is urgently necessary to benchmark the diagnostic capabilities of current models systematically. Given the limitations of existing medical benchmarks in evaluating advanced diagnostic reasoning, we present DiagnosisArena, a comprehensive and challenging benchmark designed to rigorously assess professional-level diagnostic competence. DiagnosisArena consists of 1,113 pairs of segmented patient cases and corresponding diagnoses, spanning 28 medical specialties, deriving from clinical case reports published in 10 top-tier medical journals. The benchmark is developed through a meticulous construction pipeline, involving multip
    
[^231]: 大型语言模型的政治意识形态：测量、不一致性与说服影响力

    The Political Ideology of Large Language Models: Measurement, Inconsistency, and Persuasive Influence

    [https://arxiv.org/abs/2505.04171](https://arxiv.org/abs/2505.04171)

    本文通过比较43个LLMs与政治人物和选民，发现其表面温和的党派立场实为特定议题上强烈立场的抵消结果，并通过实验证明LLMs能显著说服影响选民的意识形态。

    

    大型语言模型（LLMs）是一项变革性技术，从根本上改变了人们获取信息和与世界互动的方式。随着人们日益依赖它们完成各种任务，一系列学术研究开始审视这些模型中固有的偏见，尤其是政治偏见，通常发现这些偏见较小。我们挑战了这一普遍看法。首先，通过将43个LLMs与立法者、法官以及具有全国代表性的美国选民样本进行比较，我们表明，LLMs表面上温和的整体党派定位，实际上是其在特定议题上强烈党派立场的净结果，这与温和选民的情况类似。其次，在一项预先注册的随机实验中，我们表明LLMs能对政治态度产生说服性影响。被随机分配与LLM讨论政策议题的选民，其立场向该模型测量出的意识形态位置移动了3.5个百分点。

    arXiv:2505.04171v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are a transformational technology, fundamentally changing how people obtain information and interact with the world. As people become increasingly reliant on them for an enormous variety of tasks, a body of academic research has developed to examine these models for inherent biases, especially political biases, often finding them small. We challenge this prevailing wisdom. First, by comparing 43 LLMs to legislators, judges, and a nationally representative sample of U.S. voters, we show that LLMs' apparently moderate overall partisan positioning is the net result of offsetting strongly partisan expressed positions on specific topics, much like moderate voters. Second, in a pre-registered randomized experiment, we show that LLMs can exert persuasive influence on political attitudes. Voters randomized to discuss a policy issue with an LLM shift toward that model's measured ideological position by 3.5 p
    
[^232]: 再见，蓝皮书？用AI增强规则遵循实现法律繁琐工作的自动化

    Bye-bye, Bluebook? Automating Legal Drudgery With AI-Augmented Rule Following

    [https://arxiv.org/abs/2505.02763](https://arxiv.org/abs/2505.02763)

    本文首次实证评估了AI在蓝皮书引注格式这一法律繁琐工作上的表现，发现前沿模型零样本合规率仅42.6%，远低于人类水平，并提出了新基准和多项改进建议。

    

    法律AI的核心承诺之一是实现繁琐工作的自动化——即律师工作中那些正式、重复且耗费时间却不需要太多自由裁量权的任务。然而，AI模型在此类任务上的实际表现仍是一个悬而未决的问题。本文首次对AI在最具普遍性且最受诟病的法律繁琐工作——蓝皮书引注格式——上的表现进行了实证检验。我们做出了四项贡献。首先，我们开发了一个包含2,058个蓝皮书查询的新基准，并表明在零样本设置下，前沿语言模型平均仅能生成42.6%完全合规的法律引注。其次，我们与五家顶级法律评论期刊进行了实验，表明即使是“推理”模型也远低于这些期刊年度编辑选拔竞赛中人类候选人的平均得分。第三，我们表明，仅向模型提供规则...

    arXiv:2505.02763v2 Announce Type: replace-cross  Abstract: One of the central promises of legal AI is to automate drudgery -- the formal, repetitive tasks of lawyers' work that consume time without calling for much discretion. Yet it remains an open question how well AI models actually perform on such tasks. This article presents the first empirical examination of AI performance on perhaps the most ubiquitous and lamented form of legal drudgery: citation formatting under the Bluebook. We make four contributions. First, we develop a new benchmark of 2,058 Bluebook queries and show that, on average, frontier language models produce a fully compliant legal citation only 42.6% of the time in a zero-shot setting. Second, we conduct an experiment with five top law reviews and show that even a "reasoning" model falls far below the average score of the human candidates in these journals' annual editor-selection competitions. Third, we show that simply providing the models with the rules offers
    
[^233]: 利用机器遗忘实现成本高效的偏好对齐

    Leveraging Machine Unlearning for Cost-Efficient Preference Alignment

    [https://arxiv.org/abs/2504.06659](https://arxiv.org/abs/2504.06659)

    本文提出了一个结合机器遗忘与偏好对齐的框架，通过定量分析负面示例的影响差异，为成本高效地选择与加权负面示例提供了新方法。

    

    尽管大型语言模型（LLMs）的偏好对齐（PA）取得了进展，但主流方法如基于人类反馈的强化学习仍面临显著挑战。这些方法需要高质量的正面偏好示例数据集，这些数据获取成本高昂且计算密集。LLM遗忘技术通过直接移除负面示例的影响，提供了一种有前景的替代方案。然而，当前研究主要侧重于经验验证，缺乏系统的定量分析。为弥补这一空白，我们提出了一个将PA与LLM遗忘联系起来的框架。通过双层优化，我们首先量化了遗忘特定负面示例对PA性能的影响。我们的分析表明，这些影响在不同负面示例间差异显著。基于这一见解，我们提出了一个关键问题：如何最优地选择和加权负面示例以进行遗忘？

    arXiv:2504.06659v2 Announce Type: replace-cross  Abstract: Despite advances in Preference Alignment (PA) for Large Language Models (LLMs), mainstream methods like reinforcement learning with human feedback face notable challenges. These approaches require high-quality datasets of positive preference examples, which are costly to obtain and computationally intensive. The LLM unlearning technique presents a promising alternative by directly removing the influence of negative examples. However, current research has primarily focused on empirical validation, lacking systematic quantitative analysis. To bridge this gap, we propose a framework linking PA with LLM unlearning. Through bi-level optimization, we first quantify how unlearning specific negative examples impacts PA performance. Our analysis reveals that these effects vary substantially across negative examples. Building on this insight, we pose a crucial question: how can we optimally select and weight negative examples for unlearn
    
[^234]: 大型语言模型在规则应用中的概念掌握证据

    Evidence of conceptual mastery in the application of rules by Large Language Models

    [https://arxiv.org/abs/2503.00992](https://arxiv.org/abs/2503.00992)

    本文通过心理学方法证明大型语言模型在规则应用上能复制人类的行为模式，包括意外差异和时间延迟效应，显示出概念掌握的证据。

    

    本文利用心理学方法研究大型语言模型（LLMs）在应用规则时的概念掌握能力。我们引入了一种新颖的程序，以匹配LLMs产生的思维多样性与人类样本中观察到的多样性。随后，我们进行了两项实验，比较人类和LLMs在基于规则决策上的表现。研究一发现，所有被调查的LLMs都复制了人类的行为模式，无论它们是被提示使用训练截止日期之前还是之后创建的场景。此外，我们发现了人类在两组场景之间的意外差异，令人惊讶的是，这些差异也在LLM的响应中被复制。研究二转向了人类规则应用的一个上下文特征：在强制时间延迟下，人类样本更依赖于规则文本，而非其他考虑因素如规则目的。我们的结果显示，一些模型（如Gemini Pro和Claude 3）以类人方式响应了这一特征。

    arXiv:2503.00992v2 Announce Type: replace  Abstract: In this paper we leverage psychological methods to investigate LLMs' conceptual mastery in applying rules. We introduce a novel procedure to match the diversity of thought generated by LLMs to that observed in a human sample. We then conducted two experiments comparing rule-based decision-making in humans and LLMs. Study 1 found that all investigated LLMs replicated human patterns regardless of whether they are prompted with scenarios created before or after their training cut-off. Moreover, we found unanticipated differences between the two sets of scenarios among humans. Surprisingly, even these differences were replicated in LLM responses. Study 2 turned to a contextual feature of human rule application: under forced time delay, human samples rely more heavily on a rule's text than on other considerations such as a rule's purpose.. Our results revealed that some models (Gemini Pro and Claude 3) responded in a human-like manner to 
    
[^235]: 跳出（灰色）框框思考：一种基于上下文的评分方法用于评估神经文本生成的价值与原创性

    Thinking Outside the (Gray) Box: A Context-Based Score for Assessing Value and Originality in Neural Text Generation

    [https://arxiv.org/abs/2502.13207](https://arxiv.org/abs/2502.13207)

    提出一种基于上下文的评分方法，结合信息论，用于评估神经文本生成的价值与原创性，并作为强化学习奖励微调大型语言模型，提升创造性任务的表现。

    

    尽管大型语言模型在创造性任务中的应用日益增多，但其输出往往缺乏多样性。常见的解决方案，如提高采样温度，可能会损害结果质量。在设计用于创造力的AI系统时，处理这一权衡仍然是一个未解决的挑战。基于信息论，我们提出了一种基于上下文的评分方法，用于定量评估价值与原创性。该评分鼓励准确性和对请求的遵循，同时促进与学习分布的偏离。我们证明，该评分可用作强化学习框架中的奖励，以微调大型语言模型以实现最佳性能。我们通过多种创造性任务（如诗歌生成和数学问题解决）的实验验证了我们的策略，表明它增强了生成解决方案的价值和原创性。

    arXiv:2502.13207v4 Announce Type: replace-cross  Abstract: Despite the increasing use of large language models for creative tasks, their outputs often lack diversity. Common solutions, such as sampling at higher temperatures, can compromise the quality of the results. Dealing with this trade-off is still an open challenge in designing AI systems for creativity. Drawing on information theory, we propose a context-based score to quantitatively evaluate value and originality. This score incentivizes accuracy and adherence to the request while fostering divergence from the learned distribution. We show that our score can be used as a reward in a reinforcement learning framework to fine-tune large language models for maximum performance. We validate our strategy through experiments considering a variety of creative tasks, such as poetry generation and math problem solving, demonstrating that it enhances the value and originality of the generated solutions.
    
[^236]: DR.GAP：利用性别感知提示与解耦推理缓解大型语言模型中的偏见

    DR.GAP: Mitigating Bias in Large Language Models using Gender-Aware Prompting with Decoupled Reasoning

    [https://arxiv.org/abs/2502.11603](https://arxiv.org/abs/2502.11603)

    DR.GAP通过生成性别中立的推理轨迹并将其作为上下文示例，实现了性别信息与任务语义的解耦，从而在保持模型性能的同时有效缓解了大型语言模型中的性别偏见。

    

    arXiv:2502.11603v2 公告类型：替换-交叉 摘要：大型语言模型（LLMs）展现出强大的自然语言理解能力，但也继承并放大了社会偏见，尤其是性别偏见，引发了公平性问题。现有的基于提示的去偏策略存在一个关键局限：它们未能将性别信息与任务语义分离。偏见引导会迫使模型过度强调性别线索，而基于推理的提示则会引发带有性别偏见的推理链。为解决这些挑战，我们提出了DR.GAP（用于性别感知提示的解耦推理），这是一种自动化且与模型无关的流水线，在保持模型性能的同时缓解性别偏见。DR.GAP生成性别中立的推理轨迹，并在推理过程中将其作为上下文示例应用，从而在不修改模型参数的情况下有效将性别属性与任务语义解耦。在共指消解和问答任务上的大量实验表明……

    arXiv:2502.11603v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) exhibit strong natural language understanding capabilities but also inherit and amplify societal biases, particularly gender bias, raising fairness concerns. Existing prompt-based debiasing strategies share a key limitation: they fail to disentangle gender information from task semantics. Bias steering compels models to overemphasize gender cues, while reasoning-based prompting induces gender-biased reasoning chains. To address these challenges, we propose DR.GAP (Decoupled Reasoning for Gender-Aware Prompting), an automated and model-agnostic pipeline that mitigates gender bias while preserving model performance. DR.GAP generates gender-neutral reasoning traces and applies them as in-context demonstrations during inference, effectively decoupling gender attributes from task semantics without modifying model parameters. Extensive experiments on coreference resolution and question-answering tasks acr
    
[^237]: 改进基于影响力的指令微调数据选择，实现多样能力的均衡学习

    Improving Influence-based Instruction Tuning Data Selection for Balanced Learning of Diverse Capabilities

    [https://arxiv.org/abs/2501.12147](https://arxiv.org/abs/2501.12147)

    本文提出BIDS算法，通过归一化影响力分数来消除任务间固有偏见，实现指令微调数据选择的均衡性，从而提升模型多样能力的综合表现。

    

    arXiv:2501.12147v2 公告类型：替换-交叉 摘要：选择适当的训练数据对于大型语言模型（LLMs）的指令微调至关重要，其目标包括（1）激发强大能力，（2）在不同任务间实现均衡性能。基于影响力的方法在实现（1）方面显示出潜力，通过估计每个训练样本对模型预测的贡献，但通常难以实现（2）。我们的系统调查揭示，这种性能不足可归因于一种固有偏见，即某些任务天然具有比其他任务更大的影响力。因此，数据选择往往偏向这些任务，不仅损害了模型在其他任务上的性能，而且反直觉地，也损害了这些高影响力任务本身的性能。为解决此问题，我们提出了BIDS，一种均衡且具影响力的数据选择算法。BIDS首先对训练数据的影响力分数进行归一化，然后迭代选择...

    arXiv:2501.12147v2 Announce Type: replace-cross  Abstract: Selecting appropriate training data is crucial for instruction fine-tuning of large language models (LLMs), which aims to (1) elicit strong capabilities, and (2) achieve balanced performance across different tasks. Influence-based methods show promise in achieving (1), by estimating the contribution of each training example to the model's predictions, but often struggle with (2). Our systematic investigation reveals that this underperformance can be attributed to an inherent bias, where some tasks intrinsically have greater influence than others. As a result, data selection is often biased towards these tasks, not only hurting the model's performance on others but also, counterintuitively, harming performance on these high-influence tasks themselves. To address this, we propose BIDS, a Balanced and Influential Data Selection algorithm. BIDS first normalizes influence scores of the training data, and then iteratively chooses the
    
[^238]: Bactrainus：优化大型语言模型以应对多跳复杂问答任务

    Bactrainus: Optimizing Large Language Models for Multi-hop Complex Question Answering Tasks

    [https://arxiv.org/abs/2501.06286](https://arxiv.org/abs/2501.06286)

    本文提出Bactrainus框架，通过分离段落选择、支持句识别和答案生成，并引入问题分解与推理监督，有效缓解了大型语言模型在多跳问答中因无关上下文导致的性能下降问题。

    

    arXiv:2501.06286v2 公告类型：交叉替换 摘要：多跳问答要求系统识别并整合分布在多个文档中的证据，然而大型语言模型仍易受无关上下文的影响。我们在英文HotpotQA干扰项设置中研究了这一证据瓶颈，并引入了Bactrainus，一个模块化的选择器-阅读器框架，将段落选择、支持句识别和答案生成分离。可选的问题分解和教师生成的推理监督使得测试额外推理结构在何处有用成为可能。评估结合了基础模型筛选、受控上下文和提示消融、Llama 3.1 8B Instruct和Llama 3.1 70B Instruct阅读器的参数高效适配，以及集成选择器-阅读器实验。提供完整候选上下文而非黄金支持事实，会使答案词重叠F1降低17-21分，表明仅靠规模扩展不足以解决问题。

    arXiv:2501.06286v2 Announce Type: replace-cross  Abstract: Multi-hop question answering requires a system to identify and integrate evidence distributed across documents, yet large language models remain vulnerable to irrelevant context. We investigate this evidence bottleneck in the English HotpotQA distractor setting and introduce Bactrainus, a modular selector-reader framework that separates paragraph selection, supporting-sentence identification, and answer generation. Optional question decomposition and teacher-generated rationale supervision make it possible to test where additional reasoning structure is useful. The evaluation combines foundation-model screening, controlled context and prompting ablations, parameter-efficient adaptation of Llama 3.1 8B Instruct and Llama 3.1 70B Instruct readers, and integrated selector-reader experiments. Supplying the full candidate context instead of gold supporting facts reduces answer token-overlap F1 by 17-21 points, showing that scale alo
    
[^239]: DYNASHIELD：通过动态解码定制实现的大语言模型黑盒移动目标防御

    DYNASHIELD: A Black-Box Moving Target Defense for LLMs via Dynamic Decoding Customization

    [https://arxiv.org/abs/2412.07672](https://arxiv.org/abs/2412.07672)

    DYNASHIELD通过动态定制解码参数和系统提示，在无需访问模型内部或额外训练的情况下，有效降低了黑盒LLM面对越狱攻击的成功率。

    

    大型语言模型（LLMs）仍然容易受到越狱攻击，其中对抗性提示会诱导产生有害输出。现有防御方法通常需要访问模型内部结构或进行额外训练，这限制了它们在通过黑盒API部署的服务提供商中的适用性。在本文中，我们提出了DYNASHIELD，一种移动目标防御框架，通过在推理时定制解码超参数和系统提示来提高鲁棒性。DYNASHIELD包括两个关键步骤：（1）识别能降低攻击成功概率的解码配置，（2）从加权配置池中进行概率采样，以引入模型行为的受控变异性。我们在7个开源LLM上，针对4种最先进的越狱攻击，使用AdvBench中的对抗性提示评估了DYNASHIELD。结果表明，与7种基线防御相比，攻击成功率显著降低。

    arXiv:2412.07672v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) remain vulnerable to jailbreak attacks in which adversarial prompts induce harmful outputs. Existing defenses often require access to the model internals or additional training, limiting their applicability for service providers deployed through black-box APIs. In this paper, we propose DYNASHIELD, a moving target defense framework that improves robustness by customizing decoding hyperparameters and system prompts at inference time. DYNASHIELD includes two key steps: (1) it identifies decoding configurations that reduce attack success probability, and (2) it probabilistically samples from a weighted configuration pool to introduce controlled variability in model behavior. We evaluate DYNASHIELD across 7 open-source LLMs under 4 state-of-the-art jailbreak attacks, using adversarial prompts from AdvBench. Results show substantial reductions in attack success rate compared with 7 baseline defenses, whi
    
[^240]: 多箱批处理提升大语言模型推理吞吐量

    Multi-Bin Batching for Increasing LLM Inference Throughput

    [https://arxiv.org/abs/2412.04504](https://arxiv.org/abs/2412.04504)

    本文提出多箱批处理方法，通过将相似执行时间的请求分组到预定时间箱，从排队论角度证明了能显著提升LLM推理吞吐量。

    

    随着大语言模型（LLMs）因其多样化能力而日益流行，提高其推理系统的效率变得愈发关键。批处理LLM请求是在服务器（如GPU）上调度推理作业的关键步骤，通过允许并行处理多个请求来最大化系统吞吐量。然而，请求通常具有不同的生成长度，导致资源利用不足，因为硬件必须等待批次中最长运行的请求完成后才能进入下一批。我们从排队论视角形式化了这一问题，并旨在设计一种在静态批处理框架下吞吐量最优的控制策略。我们提出了多箱批处理（Multi-Bin Batching），这是一种简单而有效的方法，通过将具有相似（预测）执行时间的请求分组到预定的时间箱中，可证明在此框架下提升LLM推理吞吐量。

    arXiv:2412.04504v2 Announce Type: replace  Abstract: As large language models (LLMs) grow in popularity for their diverse capabilities, improving the efficiency of their inference systems has become increasingly critical. Batching LLM requests is a critical step in scheduling the inference jobs on servers (e.g. GPUs), enabling the system to maximize throughput by allowing multiple requests to be processed in parallel. However, requests often have varying generation lengths, causing resource underutilization, as hardware must wait for the longest-running request in the batch to complete before moving to the next batch. We formalize this problem from a queueing-theoretic perspective, and aim to design a control policy which is throughput-optimal under a static-batching framework. We propose Multi-Bin Batching, a simple yet effective method that can provably improve LLM inference throughput under this framework by grouping requests with similar (predicted) execution times into predetermin
    
[^241]: mR$^2$AG：基于多模态检索-反思-增强生成的知识型视觉问答

    mR$^2$AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA

    [https://arxiv.org/abs/2411.15041](https://arxiv.org/abs/2411.15041)

    本文提出了一种新的多模态检索-反思-增强生成框架（mR$^2$AG），通过自适应检索和证据识别机制，解决了现有方法在知识型VQA中过度检索、缺乏证据支持及模型复杂度过高的问题。

    

    先进的多模态大语言模型（MLLMs）在处理近期知识型视觉问答（VQA）任务（如INFOSEEK和Encyclopedic-VQA）时，由于其有限且冻结的知识范围而表现不佳，常常导致模糊和不准确的回答。因此，自然引入了多模态检索增强生成（mRAG），以提供全面和最新的知识，有效扩展知识范围。然而，当前mRAG方法存在固有缺陷，包括：1）即使不需要外部知识也执行检索；2）缺乏对支持查询的证据的识别；3）由于额外的信息过滤模块或规则而增加模型复杂性。为了解决这些不足，我们提出了一种新颖的通用框架，称为多模态检索-反思-增强生成（mR$^2$AG），该框架实现了自适应...

    arXiv:2411.15041v2 Announce Type: replace  Abstract: Advanced Multimodal Large Language Models (MLLMs) struggle with recent Knowledge-based Visual Question Answering (VQA) tasks, such as INFOSEEK and Encyclopedic-VQA, due to their limited and frozen knowledge scope, often leading to ambiguous and inaccurate responses. Thus, multimodal Retrieval-Augmented Generation (mRAG) is naturally introduced to provide MLLMs with comprehensive and up-to-date knowledge, effectively expanding the knowledge scope. However, current mRAG methods have inherent drawbacks, including: 1) Performing retrieval even when external knowledge is not needed. 2) Lacking of identification of evidence that supports the query. 3) Increasing model complexity due to additional information filtering modules or rules. To address these shortcomings, we propose a novel generalized framework called \textbf{m}ultimodal \textbf{R}etrieval-\textbf{R}eflection-\textbf{A}ugmented \textbf{G}eneration (mR$^2$AG), which achieves ada
    
[^242]: 大型语言模型在宏观经济预测中的应用

    Macroeconomic Forecasting with Large Language Models

    [https://arxiv.org/abs/2407.00890](https://arxiv.org/abs/2407.00890)

    本文通过FRED-MD数据库对比评估了大型语言模型与传统方法在宏观经济预测中的表现，揭示了LLMs的优缺点及实际应用潜力。

    

    本文进行了一项比较分析，评估大型语言模型（LLMs）相对于传统宏观时间序列预测方法的准确性。近期，由于LLMs能够捕捉数据中的复杂模式并快速适应不同领域，其在预测领域日益流行。然而，与传统方法相比，它们在预测宏观经济时间序列数据方面的有效性仍是一个值得关注的领域。为此，我们基于FRED-MD数据库，对LLMs与传统宏观预测方法进行了严格评估。我们的研究结果为LLMs在预测宏观经济时间序列中的优势和局限性提供了有价值的见解，并阐明了它们在实际场景中的适用性。

    arXiv:2407.00890v5 Announce Type: replace-cross  Abstract: This paper presents a comparative analysis evaluating the accuracy of Large Language Models (LLMs) against traditional macro time series forecasting approaches. In recent times, LLMs have surged in popularity for forecasting due to their ability to capture intricate patterns in data and quickly adapt across very different domains. However, their effectiveness in forecasting macroeconomic time series data compared to conventional methods remains an area of interest. To address this, we conduct a rigorous evaluation of LLMs against traditional macro forecasting methods, using as common ground the FRED-MD database. Our findings provide valuable insights into the strengths and limitations of LLMs in forecasting macroeconomic time series, shedding light on their applicability in real-world scenarios
    

