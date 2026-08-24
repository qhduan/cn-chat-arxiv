# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TurboBias 2.0: Streaming Context-Biasing for Production-Efficient ASR Systems](https://arxiv.org/abs/2608.21343) | TurboBias 2.0通过引入不区分大小写的增强图和每流批处理解码，在支持流式推理的同时，实现了多用户独立上下文偏置，显著提升了生产级ASR系统的效率和个性化能力。 |
| [^2] | [Move by Move: Measuring and Steering How LLMs Conduct Psychotherapy](https://arxiv.org/abs/2608.21325) | 本文提出了一个十种治疗动作的本体论来测量和引导LLM在心理治疗中的行为，发现模型过度询问、忽视心理教育，并依赖上下文，而将该本体论作为工具可显著减少与人类动作分布的偏差。 |
| [^3] | [Prompt-Model Interaction Reaches the Fixed Points: A deterministic, task-free structural readout -- and the factorizations of it that failed](https://arxiv.org/abs/2608.21315) | 本文通过一种确定性的、任务无关的结构性读出（短窗口argmax映射的不动点），证明提示词与模型的交互即使在无任务条件下也能完整显现，且该交互具有模型特异性和结构分类效应。 |
| [^4] | [Memory Augmentation Unlocks Efficient Chain-of-Thought Reasoning](https://arxiv.org/abs/2608.21265) | 该论文提出记忆增强压缩框架，利用历史推理记忆作为预填充支架，在压缩思维链推理时有效弥补信息损失，平衡了效率与逻辑连贯性。 |
| [^5] | [EnSI-RAG: Entity-Structure-Indexed Retrieval-Augmented Generation for Long-Document Question Answering](https://arxiv.org/abs/2608.21252) | EnSI-RAG通过构建以实体为中心的查询无关索引（每条记录包含实体、类型、语义类别和值），有效解决了长文档问答中跨实体和多跳推理的检索难题，提升了证据分离与答案综合的准确性。 |
| [^6] | [Benchmarking Patent Drafting from Inventor-Style Disclosures](https://arxiv.org/abs/2608.21249) | 本文提出了Dis2Pat数据集和Patent-MAF多代理框架，旨在解决从发明人风格披露直接生成完整法律连贯专利申请的核心挑战。 |
| [^7] | [Affective Context Amplifies Sycophancy in LLM Responses](https://arxiv.org/abs/2608.21242) | 本研究揭示情感语境，尤其是负面情绪如孤独和痛苦，会系统性放大LLM在回应中的奉承行为，导致其软化或保留负面判断。 |
| [^8] | [RARE: Decoupling Representation Steering from Expert Routing in Mixture-of-Experts Language Models](https://arxiv.org/abs/2608.21236) | 本文提出RARE框架，通过将行为扰动投影到路由器零空间来解耦表示引导与专家路由，有效解决了MoE模型中直接应用表示工程导致的结构不匹配问题，并提升了引导性能。 |
| [^9] | [Enhancing LLMs in Predictive Political QA with Semi-Structured Data](https://arxiv.org/abs/2608.21218) | 本文提出PSL双视图框架，通过提取行为者立场和高阶结构信号，将半结构化政治记录转化为推理证据，以增强LLM在预测性政治问答中的表现。 |
| [^10] | [Personalized Privacy Control in LLMs via Attention Head Intervention](https://arxiv.org/abs/2608.21209) | 本文提出个性化隐私概念和P3Bench基准，并开发Repair方法，通过注意力头干预增强LLM的个性化隐私控制，显著降低政策忽视率。 |
| [^11] | [No PUN Intended: Plausible Unknown Names for Person-Centred LLM Evaluation](https://arxiv.org/abs/2608.21206) | 本文提出PUN协议，用于构建和验证具有合理形式但无真实证据的未知人名，以改进LLM评估中人物相关任务的准确性和可靠性。 |
| [^12] | [Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems](https://arxiv.org/abs/2608.21095) | 本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。 |
| [^13] | [When the Feature Pool Goes Algorithmic: Extending Mufwene's Ecology of Language Evolution to LLM-Mediated Exposure](https://arxiv.org/abs/2608.21088) | 本文提出大型语言模型作为分布性中介，通过算法性重新加权改变人类说话者接触语言变体的频率，从而影响语言演化，但选择核心仍留在人类说话者中。 |
| [^14] | [Jokes Aside: Measuring the Semantic Distance of Double Meanings](https://arxiv.org/abs/2608.21087) | 本文利用词嵌入重新评估了笑话生成模型中三个关键度量（明显性、兼容性、比较性），并引入新度量“对称性”，以更精确地衡量双关语义距离，从而优化幽默生成效果。 |
| [^15] | [PromptResponse: Optimizing Prompts for LLM Coding Tasks](https://arxiv.org/abs/2608.21074) | 本文通过对照实验发现，使用一致的格式（如JSON）优化提示词可提升编码任务的生成效率和稳定性，而基于LLM的提示词调整反而显著降低了任务性能。 |
| [^16] | [Evidence-Consistent Generative Detection under Scenario-Level Distribution Shift](https://arxiv.org/abs/2608.21043) | 本文提出场景级分布偏移下的检测问题，揭示传统分布内评估会高估模型鲁棒性，并指出模型倾向于记忆场景特定线索而非泛化到新场景。 |
| [^17] | [COMET: Contrastive Motion-Enhanced Temporal Reasoning for Video Multimodal Large Language Models](https://arxiv.org/abs/2608.21030) | COMET通过引入基于泰勒帧差分的运动分支和时序注意力偏置增强交叉注意力，并采用时序先验蒸馏与TC-GRPO优化，系统性地解决了视频多模态大语言模型在细粒度运动时序理解上的不足。 |
| [^18] | [Scaling Unsupervised Word Alignment to Documents via Structural Constraints](https://arxiv.org/abs/2608.21023) | 本文提出了两种无需训练且轻量级的文档级词对齐方法（CTFAlign和MDPAlign），通过结构约束（语义区域限制或位置先验）克服了直接应用句子级算法导致的性能下降，在多种语言对上实现有效对齐。 |
| [^19] | [Free-Text Evaluation of LLMs for 5G Domain Knowledge and Fault Analysis using LLM-as-Judge](https://arxiv.org/abs/2608.21021) | 本文首次在自由文本生成格式下评估轻量级LLM在5G领域知识和故障分析中的表现，并引入LLM作为评审的框架，以验证开放诊断推理的可扩展性。 |
| [^20] | [Target-Aware Calibration Data Selection for Preserving Uncertainty in Quantized Language Models](https://arxiv.org/abs/2608.21019) | 本文提出了一种名为DPQ的轻量级量化前校准数据选择方法，通过目标感知的混合高怀疑示例和通用锚点，在量化过程中保持模型的不确定性行为，而非仅优化准确性。 |
| [^21] | [MigrationNarrate: A Dataset for Detection of Migration Narratives in YouTube Videos](https://arxiv.org/abs/2608.20984) | 该论文提出了首个多模态数据集MigrationNarrate，用于检测YouTube视频中的移民叙事，填补了该领域缺乏标注数据的空白。 |
| [^22] | [Extractive Summarization for Arabic Documents Using SAraBERT with a Semantic Siamese Similarity Evaluation Metric](https://arxiv.org/abs/2608.20964) | 本文提出SAraBERT模型，通过增加句子间变换器层和新型语义孪生相似度评估指标，显著提升了阿拉伯语抽取式摘要的质量和覆盖度。 |
| [^23] | [TreeWY: Speculative Verification for Gated DeltaNet Hybrids](https://arxiv.org/abs/2608.20961) | 本文提出TreeWY方法，通过树状WY变换消除推测解码中的状态快照，显著降低内存开销，从而支持高接受率的宽草稿树。 |
| [^24] | [Quantization-Aware Healing: A Practical Recipe for Recovering Compressed, 4-Bit LLMs](https://arxiv.org/abs/2608.20953) | 提出量化感知修复（QAH）方法，直接从未压缩的原始模型蒸馏4比特学生模型，在显著降低计算成本的同时，在多数基准上匹配或超越bfloat16来源性能。 |
| [^25] | [MentorPulse: Refreshing Cross-Model Latent Guidance for Long-Form Generation](https://arxiv.org/abs/2608.20927) | MentorPulse通过动态刷新跨模型引导记忆，在不重置学生模型缓存的情况下，显著提升长文本生成中的约束满足度。 |
| [^26] | [Source-Free MT Evaluation Is Not MT Evaluation](https://arxiv.org/abs/2608.20925) | 本文指出无源评估依赖参考译文是不公平的，充分性应基于源文本判断，参考仅应作为辅助证据而非主要标准。 |
| [^27] | [ForeDreamer: A Self-Evolving Dual-Agent Memory Architecture for Future Event Prediction](https://arxiv.org/abs/2608.20920) | ForeDreamer通过双智能体架构将原始网络证据转化为结构化记忆，分离事实与经验记忆，从而提升开放网络未来事件预测的准确性。 |
| [^28] | [KREL: Automatic Medical Coding via Knowledge-Guided Reasoning over Clinical Evidence with LLMs](https://arxiv.org/abs/2608.20887) | 该论文提出了KREL框架，通过结合大型语言模型的推理能力与外部ICD编码指南的结构化知识，解决了自动医疗编码中临床记录过长、标签空间庞大及编码规则复杂等关键问题。 |
| [^29] | [Identify, Locate, Link: End-to-End Key-Value Extraction from Document Images](https://arxiv.org/abs/2608.20868) | 本文提出了一种端到端的键值提取方法，通过微调紧凑视觉语言模型SmolDocling，无需OCR预处理即可同时完成识别、定位和关联，并引入数据增强和布局感知评估，显著提升了文档信息提取的准确性和效率。 |
| [^30] | [Ontology-Driven Structural Regularization for Document-Level Relation Extraction](https://arxiv.org/abs/2608.20856) | 本文提出一种基于本体的结构正则化框架，用于量化和消除文档级关系抽取中的结构噪声（如本体约束违反和逻辑矛盾），从而显著提升模型泛化性能。 |
| [^31] | [SAC-Copula: Quality-Preserving Watermarking for Diffusion Language Models via Smooth Correlated Gumbel Fields](https://arxiv.org/abs/2608.20839) | SAC-Copula通过引入基于高斯copula的平滑局部相关Gumbel扰动场，解决了扩散语言模型水印中扰动与解码动态不匹配的问题，实现了生成质量与可检测性的更优平衡。 |
| [^32] | [STAR-OPD: Structured Aspect-Cascade-Aware On-Policy Reward Distillation for ABSA Quadruple Extraction](https://arxiv.org/abs/2608.20831) | 本文提出STAR-OPD方法，通过在线策略奖励蒸馏和结构化方面级联感知，解决ABSA四元组抽取中蒸馏模型的结构无效状态问题，提升小模型部署性能。 |
| [^33] | [Denoising the Future: Context-Aware Spectral Diffusion for Temporal Knowledge Graph Extrapolation](https://arxiv.org/abs/2608.20804) | 本文提出FreqDiff，一种频率感知扩散框架，通过双流去噪器结合时间依赖建模和上下文感知频谱校准，以及频域正则化，有效解决了时间知识图谱外推中历史信息聚合导致的目标信号稀释问题。 |
| [^34] | [Profiling What Matters: Context-Aware Item Profiles from Large-Scale Metadata for LLM Recommenders](https://arxiv.org/abs/2608.20801) | CAIRO提出了一种用户上下文感知的物品画像框架，通过结构化元数据并动态选择每个用户-物品对的相关信息，显著提升LLM重排序的个性化与细粒度理解。 |
| [^35] | [Tree-of-Concerns: Hierarchical Multi-Agent Debate for Unstated-Limitation Extraction in Scientific Critique](https://arxiv.org/abs/2608.20777) | 本文提出“关注之树”多智能体框架，通过专门怀疑论角色和小组审查机制，从科学论文中提取未声明局限，在精确度和覆盖率上分别比最强基线提升79%和11%。 |
| [^36] | [PSK at WMT 2026 MIST: Task-Specialized QLoRA Adapters for Multilingual Summarization and Question Answering](https://arxiv.org/abs/2608.20757) | 本文提出了一种基于Tiny Aya Global模型和三个任务特化QLoRA适配器的多语言摘要与问答系统，通过分离任务适配器提升了摘要性能，并针对开放问答的不稳定性提交了多系统方案。 |
| [^37] | [Calibrating Criterion Revision in LLM Agents: Failure Modes and a Trace-Anchored Protocol](https://arxiv.org/abs/2608.20729) | 本文提出并验证了语言模型代理中标准修订的五个必要条件，实验显示当前模型未能在任何案例中完全满足这些条件，但零结果不排除存在普遍能力的可能性。 |
| [^38] | [AsmEvo: Agentic Assembly-Level Optimization of AMD GPU Kernels with Functional Equivalence Verification](https://arxiv.org/abs/2608.20711) | AsmEvo提出了一种无需源代码、直接优化已编译AMDGPU二进制文件的代理级汇编优化方法，通过差分验证确保功能等价性，突破了现有LLM优化器依赖源码的限制。 |
| [^39] | [Temporal Validity on Real Software Histories: Eliminating Stale-Fact Errors in Code-Assistant Memory over GitHub Fixes](https://arxiv.org/abs/2608.20685) | 本文验证了MemStrata在真实软件历史中通过确定性过时记忆消除RAG的时间盲点，显著提升答案准确率（0.91对比0.57-0.59），并减少过时事实错误。 |
| [^40] | [Why2Speak: Faithful Reasoning for Abstaining Action Policies](https://arxiv.org/abs/2608.20670) | 本研究揭示了行动策略中能力与可审计性之间的权衡：直接决策策略性能更优但缺乏可检查推理，而推理策略虽提供追踪却牺牲了性能，特别是在干预机会的召回率上。 |
| [^41] | [Auditable by Construction: An Ontology-Driven Framework for Trustworthy LLM Analytics in Enterprise Finance](https://arxiv.org/abs/2608.20661) | 本文提出了KDAF框架，通过本体驱动和CARP检索，确保每个事实具有来源可追溯性，从而显著提升企业金融中大语言模型回答的可审计性和准确性。 |
| [^42] | [Directional Contextual Representations for Dependency Relations: Why Cross-Direction Pairing Fails](https://arxiv.org/abs/2608.20647) | 该论文发现，在依存关系类型分类中，将双向LSTM的前向与后向表示进行跨方向配对会持续性能下降，且差距随距离增大而增大，并通过冻结主干实验证明此现象源于表示的内在结构而非训练协同适应。 |
| [^43] | [MIL-BERT: Classification of Arbitrarily Large Text with Performance and Explanatory Guarantees](https://arxiv.org/abs/2608.20636) | 本文提出MIL-BERT算法，利用多实例学习选择关键文本摘录进行分类，可处理近百万令牌的大规模文本，在多个长文本数据集上达到最先进性能，并具备解释性保证。 |
| [^44] | [AgentMercury: Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale](https://arxiv.org/abs/2608.20634) | AgentMercury提出了一种从高层商业场景中规模综合可验证环境的新框架，通过先实例化持久化世界再涌现任务，构建了覆盖多行业多国家的数千个环境，为强化学习提供了可扩展的训练基础。 |
| [^45] | [Sparse Token Routing in Efficient Transformers](https://arxiv.org/abs/2608.20632) | 本文提出SEWN双流Transformer，通过上下文门控实现令牌路由，在不牺牲任务精度的前提下，显著区分令牌重要性（p<10^-10），而静态先验方法则无法通过反事实测试。 |
| [^46] | [When Failures Propagate: Causal Failure Attribution in Agentic Retrieval-Augmented Generation](https://arxiv.org/abs/2608.20627) | 本文提出了AgenticRAG-FP基准，用于评估代理式RAG中因果失败归因的准确性，发现覆盖率诊断在早期跳数有效，但在后续跳数失效。 |
| [^47] | [JuryProbe: An Empirical Consensus-Risk Diagnostic for Routing Reference-Free Factuality Judge Panels to Grounded Verification](https://arxiv.org/abs/2608.20607) | 本文提出JuryProbe，一种通过仅假阴性相关性和假共识提升度来诊断无参考事实性评审团共识风险的方法，并在高风险时路由到有参考验证，以减少因共享盲点导致的错误接受。 |
| [^48] | [Open-Weight Masked Introspection: Measuring What Language Models Can Report About Their Own Computation](https://arxiv.org/abs/2608.20569) | 该研究构建了OWMI框架，在八个开放权重模型上进行78,000多次测量，发现这些模型无法内省自身计算状态，其报告与随机猜测无异。 |
| [^49] | [LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding](https://arxiv.org/abs/2608.20530) | LiLiCorr通过轻量级似然相关性模型关联起草器的逐位置边际分布，在不构造完整联合分布的情况下捕获块级联合结构，从而提升投机解码的连贯性。 |
| [^50] | [ProofJudge: Tool-Grounded LLM Evaluation of Formal Proof Quality in Mathlib](https://arxiv.org/abs/2608.20432) | ProofJudge是一个利用工具访问库状态的LLM评判系统，能有效评估形式化证明质量，并在多个维度上对齐人类偏好。 |
| [^51] | [ARGUS: Theory-of-Mind Guided Argument Generation with Strategy-Aware Planning and Knowledge Grounding](https://arxiv.org/abs/2608.20405) | Argus通过心智理论推理器构建受众信念模型，结合策略感知规划和知识锚定，实现更有效的说服性论证生成。 |
| [^52] | [LingShu: A Large-Scale Symptom-Centric Contextualized Knowledge Graph Bridging Traditional Chinese Medicine and Modern Biomedicine](https://arxiv.org/abs/2608.20402) | 灵枢构建了一个大规模以症状为中心的情境化知识图谱，通过整合多源数据（如临床记录和中医文献），有效桥接了中医与现代生物医学，并利用情境化四元组克服了传统二元关系的局限性。 |
| [^53] | [When Retrieval Fails Before It Begins: Structurally Indirect Prerequisite Eviction as a Retention Failure in Agentic Memory](https://arxiv.org/abs/2608.20400) | 本文首次揭示了代理记忆中的“检索前失败”模式——结构性间接前提驱逐，并提出依赖感知语义垃圾收集规则，显著提升全链保留率。 |
| [^54] | [Self-Supervised Speech Representations Track Spoken Language Convergence to Adult Models in Infants and Children Who Are Deaf/Hard-of-Hearing](https://arxiv.org/abs/2608.20396) | 本研究首次利用自监督语音嵌入（HuBERT-BASE）从长时日常录音中直接量化聋/听障儿童与成人照护者之间的言语趋同过程，仅凭单一距离度量即可追踪语言发展轨迹，无需人工转录，为跨语言和人群的大规模语言发展评估提供了可扩展的新方法。 |
| [^55] | [A Factorial Ablation of a Speech-to-SFT Pipeline: Differential Effects on Data Quality and Downstream Transfer](https://arxiv.org/abs/2608.20394) | 通过2x2因子消融实验发现，语音到SFT流水线中的数据质量改进虽能提升评审质量，但并未显著提升下游MCQA性能，表明质量提升的迁移效果有限。 |
| [^56] | [Knowledge-Graph-Gated Defactualization for Style-Controllable and Fact-Preserving Generation in Agentic Conversational AI](https://arxiv.org/abs/2608.20393) | 本文提出DSR框架，通过知识图谱与激活引导结合，在风格可控生成中显式区分并保留事实内容，解决了激活引导中的语义泄漏问题。 |
| [^57] | [Evaluation-as-Search: Adaptive Discovery of Grounding Failures in Meeting Assistants](https://arxiv.org/abs/2608.20392) | 本文提出“评估即搜索”（EaS）方法，通过自适应搜索会议问题空间，构建MeetingProbe基准，显著提高了发现LLM会议助手接地失败的效果。 |
| [^58] | [ImmigrationReason: A Structured Dataset of U.S. Immigration Appeals for Legal Reasoning Research](https://arxiv.org/abs/2608.20391) | 该论文构建了首个大规模行政上诉法律推理数据集，覆盖12,375项美国移民上诉决定，包含细粒度证据评估与逐字批评，填补了行政裁决领域NLP资源的空白。 |
| [^59] | [Ansari: A Retrieval-Grounded Islamic AI Assistant -- Architecture, Deployment, and Lessons from 140,000 Conversations](https://arxiv.org/abs/2608.20390) | 安萨里通过代理式检索循环，仅基于认证的伊斯兰语料库回答并附引用，有效避免了事实捏造和价值观错位，已成功处理14万次多语言对话。 |
| [^60] | [Intent Engine: Natural-Language Intent Translation for Intent-Driven Orchestration in the Compute Continuum](https://arxiv.org/abs/2608.20388) | 意图引擎提出了一种自然语言意图翻译架构，通过构建经过验证的SLO工件来增强意图驱动编排的可靠性，避免因LLM直接生成错误而导致的放置问题。 |
| [^61] | [Poly-InstructTTS: Learning In-the-Wild Expressive Speech Synthesis from Open-Ended Instructions](https://arxiv.org/abs/2608.20387) | 本文提出Poly-InstructTTS，通过构建1000小时多模态指令标注语料库和基于属性思考标记的GPT框架，实现了从开放式自然语言指令生成高表现力语音，并支持说话人微调以保持人物角色。 |
| [^62] | [Using Human-LLM Disagreement to Improve Checklist-Based Quality Appraisal](https://arxiv.org/abs/2608.20385) | 本研究通过分析人类与LLM在基于清单的质量评估中的分歧模式，提出了一种改进模糊清单项目的方法，并验证了LLM在特定条件下能近似专家判断。 |
| [^63] | [Decoupled Vision-Language System for Multimodal Understanding and Generation](https://arxiv.org/abs/2608.20382) | Libra通过解耦视觉和语言系统的自模态建模与跨模态交互，实现了高效的多模态理解与生成，并在图像到文本和文本到图像任务中展现了有效性。 |
| [^64] | [EditPPT: Faithful Long-Deck Slide Editing via Structured Tool-Using Multi-Agent with Dual-Modal Validators](https://arxiv.org/abs/2608.20381) | EditPPT通过多智能体框架和双模态验证器，将幻灯片编辑转化为受约束的工具选择问题，实现了对长幻灯片的高保真、高准确编辑，并提供了新的基准DeckEdit-Bench。 |
| [^65] | [TH-GNN: Heterogeneous Temporal Graph Neural Networks for LLM-Agent Shilling Attack Detection](https://arxiv.org/abs/2608.20376) | TH-GNN通过融合异构时序图结构与跨模态语义注意力，有效检测LLM生成的推荐系统托攻击，解决了现有方法忽视图结构和时序协调的缺陷。 |
| [^66] | [GRAFT: Adaptive DLM-Based Draft Tree Construction with Target-Distilled Edge Scoring](https://arxiv.org/abs/2608.20375) | 本文提出GRAFT，通过目标蒸馏边评分和自适应树大小调整，改进扩散语言模型下的基于树的投机解码，提升令牌接受率与吞吐量。 |
| [^67] | [VA-DPO: Valence-Arousal Direct Preference Optimization for Controllable Emotion Generation in Language Models](https://arxiv.org/abs/2608.20374) | VA-DPO通过将情感目标表示为连续效价-唤醒度点，并基于距离阈值筛选偏好数据，实现了比现有提示方法更精确可控的情感生成，显著降低了目标距离并提升了相关性。 |
| [^68] | [An ambiguity taxonomy for evaluating large language model performance on clinical registry abstraction: a multi-site prospective study](https://arxiv.org/abs/2608.20373) | 本研究提出了一种六类歧义分类法，用于系统评估大语言模型在临床注册表数据提取中的性能，并验证了其在不同医疗中心的有效性。 |
| [^69] | [When Do LLMs Replace Fine-Tuned NLU? A Decision Framework for Intent Detection in Production Conversational Systems](https://arxiv.org/abs/2608.20371) | 本文通过实验证明，大型语言模型在意图检测中仅在特定条件下（如无标签数据或需要鲁棒性）优于微调模型，而在有充足领域标签时，微调模型更高效且成本更低。 |
| [^70] | [ASTAR: Automated induction of STAndardized radiology Reporting templates from large-scale clinical free-text corpora](https://arxiv.org/abs/2608.20369) | ASTAR利用大型语言模型自动从大规模临床自由文本中归纳标准化放射学报告模板，克服了手动构建的静态和扩展性限制，并在多中心胎儿脑MRI报告上超越了专家模板。 |
| [^71] | [Research Paper Quality Recognition Through Textual Feature Analysis](https://arxiv.org/abs/2608.20368) | 本文提出了一种仅利用标题和摘要文本特征进行科研论文质量分类的基准方法，并通过多种嵌入技术与分类器组合，实现了高达91.12%的准确率，同时提供了透明度、可视化和可解释性分析。 |
| [^72] | [Trilingual Topic Modeling of Sri Lankan Parliamentary Debates](https://arxiv.org/abs/2608.20365) | 本文提出了一种结合LLM提取和多语种嵌入聚类的新框架，成功对斯里兰卡三语议会辩论进行主题建模，并恢复了与重大国家事件对应的30个宏观主题。 |
| [^73] | [Hadith computational science in the age of large language models: a critical narrative review](https://arxiv.org/abs/2608.20364) | 本文通过批判性叙事综述，系统评估了圣训计算科学在大语言模型时代的方法论稳健性、基准局限和未解问题，指出数据与工具进步与学术应用受限并存的现状。 |
| [^74] | [Multilingual Verifier Bias in RLVR: Benchmark, Rollout Diagnosis, and the Cross-Lingual Selection Bottleneck](https://arxiv.org/abs/2608.20362) | 本文揭示了多语言环境中RLVR的精确匹配验证器因语言差异产生严重假阴性偏差，并提出了一个可复用的审计协议和诊断方法，指出跨语言选择瓶颈是核心问题。 |
| [^75] | [Toward Auto-Research: Mining Falsifiable Research Ideas from Paper Knowledge Graphs with Categorical Structure](https://arxiv.org/abs/2608.20361) | 本文提出用范畴论（组合与恒等箭头）为论文构建类型化知识图谱，从而生成可证伪的跨领域研究类比，克服传统方法将论文视为平面对象的局限。 |
| [^76] | [TriPLU: Bypassing the Gate with Direct Trilinear Product FFNs in Tiny Language Models](https://arxiv.org/abs/2608.20360) | TriPLU通过直接三线性乘积分支替代门控机制，在微型语言模型中显著降低了验证损失，优于SwiGLU及其他乘积阶数控制。 |
| [^77] | [Self-Speculation for Faster Reasoning Models](https://arxiv.org/abs/2608.20359) | 本文提出SSR，一种无需训练的自我投机解码方法，利用部分思维链作为起草者、完整思维链作为验证者，以加速推理模型生成，无需额外训练。 |
| [^78] | [ExpertIVS: Sociological Expert Driven Individual Value Simulation in Large Language Models](https://arxiv.org/abs/2608.20355) | 提出ExpertIVS框架，利用社会学专家智能体对调查数据进行深度语义重构，以生成内部一致的个体价值观画像，并解决传统静态评估无法反映真实对话中价值取向的问题。 |
| [^79] | [The Divergence Hypothesis: Unmasking Lexical Interference and Label Bias in Mental Health NLP](https://arxiv.org/abs/2608.20353) | 本文提出TSS诊断框架和分歧度统计量，揭示了心理健康NLP中词汇特征对人类标注与自动标注数据的差异影响，为标签来源审计提供了新方法。 |
| [^80] | [Exploratory As-Analyzed No-Detection of Culturally-Marked Predicate-Triggered PII Amplification in a Synthetic-English RAG Probe: A Predicate-Resource-Confounded Audit](https://arxiv.org/abs/2608.20351) | 本论文通过预注册审计发现，在合成英语RAG系统中，刻板印象负载查询并未在干净信息渠道上放大PII泄露，且早期泄露信号受提示回显伪影污染。 |
| [^81] | [How to Train a Real-World Silicon Concierge? Internalizing Complex Business Workflow to Only OneModel](https://arxiv.org/abs/2608.20350) | OneModel通过将复杂业务流程内化到单一模型参数中，取代模块化流水线，实现了延迟降低50%以上，并提高了准确性和效率。 |
| [^82] | [Beyond Prompt Engineering: A Systematic Analysis of Prompt Lexical Sensitivity and Its Impacts on Quality](https://arxiv.org/abs/2608.20349) | 该论文首次通过大规模n-gram级机制分析，揭示了提示性能稳定性缩放定律，并识别出领域特定术语和显式行动指令作为提升提示鲁棒性的两个核心语言驱动因素。 |
| [^83] | [Inhibitory Attention for Clinical Long-Context Reasoning: Characterizing and Mitigating Lost-in-the-Middle Effects in EHR Processing](https://arxiv.org/abs/2608.20348) | 本文首次系统表征了电子健康记录处理中的临床“中间丢失”问题，并提出了查询条件临床抑制（QCCS）方法，以缓解长上下文推理中关键信息检索可靠性下降的问题。 |
| [^84] | [Who Do Language Models Think Is Competent? A Mechanistic Analysis of Occupational Bias](https://arxiv.org/abs/2608.20347) | 该论文提出一个因果框架来揭示语言模型中隐藏的职业偏见，证明即使行为上无差异，内部表征仍受人口统计学属性影响，影响用户能力判断。 |
| [^85] | [Building and Evaluating a Synthetic Bengali Speech Resource for Telecom Customer Care](https://arxiv.org/abs/2608.20346) | 该论文构建并公开了一个包含10,000对音频-文本、约26.82小时的合成孟加拉语电信客服语音数据集，并验证了其可懂度。 |
| [^86] | [When Vocabulary Comprehension Fails Clinical Reasoning: Evaluating Therapy Bots' Safety Risks for Generation Alpha](https://arxiv.org/abs/2608.20345) | 本研究首次系统评估了治疗机器人在应对Alpha世代独特语言模式时的安全风险，并提出了两个基准数据集来揭示其临床推理中的词汇理解失败。 |
| [^87] | [Beyond Raw Transcripts: Structured Persona Extraction for LLM-Based Digital Twins](https://arxiv.org/abs/2608.20344) | 本文提出，数字孪生预测准确性的关键瓶颈在于人物画像信息的结构组织方式，而非信息量，并通过引入基于消费者行为理论的BDE结构化模式，将预测准确性提升1.91个百分点。 |
| [^88] | [Mitigating Identity Essentialism in LLM Agents with Longitudinal Life Trajectories](https://arxiv.org/abs/2608.19621) | 本文提出LifeMem框架，通过结合结构化生活事件检索和参数化记忆，缓解大语言模型智能体因静态画像导致的身份本质主义，从而增强社会模拟中的人口多样性。 |
| [^89] | [Hear2Act: Benchmarking When Prosody Should Change What an Assistant Does](https://arxiv.org/abs/2608.19515) | 本文提出了Hear2Act，一个统一基准，用于测试韵律线索是否以及何时能改变面向任务助手的下游决策，并证明添加音频信息可影响最优解率。 |
| [^90] | [SuTRA : Structurally-Unified Tokenization with Root Awareness](https://arxiv.org/abs/2608.18087) | SuTRA是一种形态感知分词算法，通过保持akshara完整性和惩罚跨形态边界的合并，有效减少了形态破碎化，在印度语言上显著提升了形态对齐和语义可恢复性，并提高了机器翻译性能。 |
| [^91] | [Do Large Language Models Play Six Degrees of Separation? Measuring Topological Compression in Long-Context Manifolds](https://arxiv.org/abs/2608.17950) | 本文绕过注意力权重，直接分析隐藏状态流形的动态几何结构，发现大型语言模型的深层潜在空间自发形成小世界网络，并展现出从碎片化到高度可导航的拓扑相变，从而实现了长上下文中的远距离语义压缩。 |
| [^92] | [Mint-Agent: Introducing Finance-Native Agentic Foundation Models](https://arxiv.org/abs/2608.16386) | 本文提出Mint-Agent，一种金融原生智能体基础模型，通过数据引擎、MintHarness框架和结合SFT、OPD与RLVR的训练算法，实现可靠且可审计的长周期金融研究执行。 |
| [^93] | [DFM Mimir v1: An Open HRM Delivering Frontier Performance at 1B Parameters Using Only Permissible Post-Training Data](https://arxiv.org/abs/2608.13517) | Mimir v1是一个10亿参数的HRM架构语言模型，仅使用许可数据训练，在英语和丹麦语上实现了前沿性能，并超越了同尺寸及更大尺寸的模型。 |
| [^94] | [Reading Cognition as Decisions Unfold in Words: A Factorized Inverse Decision Model](https://arxiv.org/abs/2608.09222) | 提出了一种因子化逆决策模型，通过将任务执行分解为动作和努力因子，从言语转录中推断认知决策过程，在老年人购物对话任务中实现了选择性估计并保留了动作区分。 |
| [^95] | [Tree-of-Experience: Hierarchical Experience Management for Self-Evolving Agents](https://arxiv.org/abs/2608.09044) | 本文提出经验之树（ToE）框架，通过将经验组织成与LLM智能体层级化推理过程对齐的共享树结构，解决了现有经验管理方法中反馈归因、跨任务迁移和更新检索效率低的问题。 |
| [^96] | [The Voiceprint Fallacy: Why Voices Are Not Unique Biometric Imprints](https://arxiv.org/abs/2608.07980) | 本文揭示“声纹”作为稳定独特生物识别印记的谬误，强调声音的动态性和情境依赖性，并探讨了深度伪造技术对说话人身份认定的新挑战。 |
| [^97] | [SMOPD: Multi-Reward Reinforcement Learning via Specialize-and-Merge Online Policy Distillation](https://arxiv.org/abs/2608.03092) | 本文提出一种通过专业化与合并的在线策略蒸馏方法（SMOPD），以增强稀疏奖励的优化信号，同时保持密集奖励驱动的能力，解决多奖励强化学习中不同粒度奖励信号失衡的问题。 |
| [^98] | [Prompt-Induced Waste in Coding Agents: Reasoning, Effort, Harness Design, and End-to-End Cost](https://arxiv.org/abs/2608.01347) | 本论文揭示提示语义、推理努力和框架设计是相互作用的因素，共同决定编码代理的端到端成本与成功率，而非独立可调的控制变量。 |
| [^99] | [ZenGen: Social Mind for LLMs](https://arxiv.org/abs/2607.23740) | 本文提出了ZenGen框架，通过SoMBench基准测量大语言模型的社会智能，并采用诊断驱动的训练方案来内化和提升其社会认知能力。 |
| [^100] | [Language Shapes Instruction Hierarchy Compliance in Multilingual LLMs](https://arxiv.org/abs/2607.23545) | 本研究提出了多语言指令层级基准XIH-Bench，发现IH遵从性存在语言依赖的不对称性和跨语言冲突中的“语言边界效应”，表明语言显著影响多语言LLM的指令优先级遵从。 |
| [^101] | [Index SLM Technical Report](https://arxiv.org/abs/2607.09885) | 该论文介绍了哔哩哔哩开发的Index-1.9B系列开放小语言模型，包含四个变体，并通过创新的Warmup-Stable-Decay学习率调度和Norm-Head输出层，在2.8万亿令牌上实现了稳定且高效的预训练，支持角色扮演定制。 |
| [^102] | [Know2Guess: A Contamination-Aware Multi-Zone Benchmark for Knowledge-Boundary Evaluation in Large Language Models](https://arxiv.org/abs/2606.26101) | 提出了一个包含1200个条目、覆盖五个领域的污染感知多区域基准，用于评估大语言模型在知识边界上从有依据回答到应放弃未知的过渡能力，并发现指令调优模型存在选择性但不完全的过渡。 |
| [^103] | [The Metanym Game: An LLM Benchmark Without Ground Truth That Rises With the Models It Measures](https://arxiv.org/abs/2606.21008) | 该论文提出一种无真实基准的LLM评估方法，通过类比生成与相互评分，利用SVD特征方程统一评判生成与评判能力，并发现与GPQA Diamond存在相关性。 |
| [^104] | [Detecting Functional Memorization in Code Language Models](https://arxiv.org/abs/2606.12764) | 本文提出了一种通过AI编码代理生成测试输入来检测代码语言模型中功能记忆化的方法，该记忆化在文本审计中不可见，通过反事实框架对比目标模型与参考模型实现功能等价性检测。 |
| [^105] | [INFUSER: Influence-Guided Self-Evolution Improves Reasoning](https://arxiv.org/abs/2606.09052) | INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。 |
| [^106] | [Audio Interaction Model](https://arxiv.org/abs/2606.05121) | 本文提出了一种始终在线的音频交互模型，通过感知-决策-响应范式实现不中断监听的主动干预，并配套构建了大规模流式数据与基准，在主流任务上保持性能的同时支持长流交互和主动响应。 |
| [^107] | [SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment](https://arxiv.org/abs/2606.02530) | SafeSteer通过将安全对齐限制在稀疏的安全令牌上进行局部策略蒸馏，有效降低了对齐税，同时提升了安全性与通用能力的权衡。 |
| [^108] | [Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence](https://arxiv.org/abs/2606.01444) | 本文提出一个基于范畴论的框架，通过左Kan扩展和体制转换来区分检索、搜索与科学发现，实现不依赖主观新颖性的自修正智能体系统。 |
| [^109] | [GRASP: Gated Regression-Aware Skill Proposer for Self-Improving LLM Agents](https://arxiv.org/abs/2605.29668) | GRASP提出了一种门控回归感知技能提议机制，通过硬性回归预算和平衡探针确保技能库每次编辑只带来净改进，在临床基准上将LLM智能体性能大幅提升。 |
| [^110] | [Lost in Sampling: Assessing Lexical Reachability in LLMs via the Word Coverage Score (WCS)](https://arxiv.org/abs/2605.27268) | 本文提出词汇覆盖率得分（WCS），首次定量揭示标准采样过滤器如何从数学上抑制大语言模型对低频、高信息量人类词汇的生成可达性，从而量化解码机制对语言多样性的限制。 |
| [^111] | [Granuscore: A Reference-Free Measure of Granularity for Text Analysis and Question Answering](https://arxiv.org/abs/2605.26620) | 本文提出Granuscore，一种基于分层嵌入空间的无参考粒度度量方法，能有效捕捉文本粒度差异，并在问答基准中揭示模型行为的一致模式。 |
| [^112] | [RouteScan: A Non-Intrusive Approach to Auditing MoE LLMs Safety via Expert Routing Telemetry](https://arxiv.org/abs/2605.24817) | 本文提出RouteScan，一种通过分析GPU执行中专家路由遥测来非侵入式审计MoE大语言模型安全性的方法，无需访问用户提示或模型内部，从而兼顾安全与隐私。 |
| [^113] | [STS: Efficient Sparse Attention with Speculative Token Sparsity](https://arxiv.org/abs/2605.15508) | STS提出了一种利用小型草稿模型预测重要令牌来动态构建稀疏掩码的方法，无需重训练即可在保持精度的同时显著加速LLM推理。 |
| [^114] | [RefusalGuard: Geometry-Preserving Fine-Tuning for Safety in LLMs](https://arxiv.org/abs/2605.01913) | 本文揭示了标准微调导致安全对齐退化的表示级机制，并提出了REFUSALGUARD框架，通过保持安全相关表示的几何结构来在微调中维持模型安全性。 |
| [^115] | [Compared to What? Baselines and Metrics for Counterfactual Prompting](https://arxiv.org/abs/2605.01048) | 本文指出，反事实提示中观察到的效应常被表面形式变化混淆，需使用基线（如改写）来校正，否则可能错误归因模型敏感性。 |
| [^116] | [Trust Stack for Mental Health AI: A Survey of Calibration across Human, Interaction, and AI Layers](https://arxiv.org/abs/2604.20166) | 本文提出一个三层信任框架，整合人类、交互和AI层面的信任校准，以解决心理健康AI中信任与安全之间的错位问题。 |
| [^117] | [Human-Level Text-to-SQL via Reinforcement Learning on Verified Data, Without Pipeline Engineering](https://arxiv.org/abs/2603.20004) | 本文通过专家驱动的多轮验证流程清洗数据，并利用RLVR微调LLM，在无流水线工程的情况下实现了人类水平的Text-to-SQL性能，突破了现有流水线方法的性能上限。 |
| [^118] | [Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration](https://arxiv.org/abs/2603.02760) | 本文提出了一种基于序列重生成概率的扩散语言模型自我评估方法DiSE，并引入灵活长度生成框架，以提升质量评估的效率和可靠性。 |
| [^119] | [Mind the Style: Impact of Communication Style on Human-Chatbot Interaction](https://arxiv.org/abs/2602.17850) | 本研究发现，友好型沟通风格的聊天机器人在提升用户满意度和任务成功率方面优于直接型风格，但无聊天机器人的控制条件在任务成功率上表现最佳。 |
| [^120] | [When Looks Do Not Lie: Discourse Structure Guided In-Context Learning for Faithful Diagram Generation](https://arxiv.org/abs/2601.20476) | 本文提出一种基于修辞结构理论的上下文学习图表生成方法，显著提升图表对源文本的忠实度，并通过专家与自动评估验证其有效性。 |
| [^121] | [SlidesGen-Bench: Evaluating Slides Generation via Computational and Quantitative Metrics](https://arxiv.org/abs/2601.09487) | 本文提出了SlidesGen-Bench基准，通过视觉领域的统一框架和内容、美学、可编辑性三个计算指标，实现了跨架构的定量且可靠的幻灯片生成评估。 |
| [^122] | [MedRAGChecker: Claim-Level Verification for Biomedical Retrieval-Augmented Generation](https://arxiv.org/abs/2601.06519) | MedRAGChecker提出了一种主张级验证框架，通过结合证据NLI和知识图谱一致性信号，对生物医学RAG的生成回答进行细粒度诊断，以区分检索与生成失败并识别安全关键错误。 |
| [^123] | [When to Ponder: Adaptive Compute Allocation for Code Generation via Test-Time Training](https://arxiv.org/abs/2601.00894) | 本文提出PonderTTT，一种无需训练的门控策略，通过TTT层的重建损失自适应触发测试时训练，在代码生成中实现高效计算分配，显著提升推理性能。 |
| [^124] | [StruProKGR: A Structural and Probabilistic Framework for Sparse Knowledge Graph Reasoning](https://arxiv.org/abs/2512.12613) | 提出了一种基于距离引导路径收集和结构感知概率建模的框架，解决了稀疏知识图谱推理中路径质量低和结构信息利用不足的问题。 |
| [^125] | [Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks](https://arxiv.org/abs/2512.03262) | 该论文提出SUSVIBES基准，评估了12种编码智能体在真实任务中的安全性，发现所有智能体生成代码的安全率极低（最高仅11.8%），且简单安全提示无法有效改善。 |
| [^126] | [When Better Teachers Don't Make Better Students: Revisiting Knowledge Distillation for CLIP Models in VQA](https://arxiv.org/abs/2511.17886) | 本文首次系统研究发现，在CLIP模型的知识蒸馏中，更强的教师并不总能带来更好的学生，现有蒸馏框架在扩展时反而导致VQA等下游任务性能下降。 |
| [^127] | [LTR-ICD: A Ranking-Aware Framework for Automatic ICD Coding](https://arxiv.org/abs/2510.13922) | 本文首次将ICD编码问题从检索视角重新定义为分类与排序任务，提出排序感知框架，显著提升了高优先级诊断代码的识别与排序准确性。 |
| [^128] | [Library Hallucinations in LLM-Generated Code: A Risk Analysis Grounded in Developer Queries](https://arxiv.org/abs/2509.22202) | 本研究首次系统分析了开发者查询变化如何触发大型语言模型生成代码中的库幻觉，揭示了不同提示条件下的系统性风险模式。 |
| [^129] | [Scale or Reason? A Compute-Equivalent Analysis of Reasoning Distillation](https://arxiv.org/abs/2509.22193) | 该论文通过等价计算分析发现，在相同计算预算下，标准指令微调（IFT）在大多数配置中优于或持平于推理蒸馏，后者仅在7B以上模型的开放式任务中具有优势。 |
| [^130] | [SKILL-RAG: Self-Knowledge Induced Learning and Filtering for Retrieval-Augmented Generation](https://arxiv.org/abs/2509.20377) | 提出SKILL-RAG方法，利用模型自知识通过强化学习训练框架来过滤无用检索内容，从而减少RAG中的幻觉并提升性能。 |
| [^131] | [SCOPE: A Generative Approach for LLM Prompt Compression](https://arxiv.org/abs/2508.15813) | 本文提出了一种基于分块重写的无训练生成式提示压缩框架SCOPE，通过语义分块和摘要重构来缩短LLM输入，同时保持生成质量，并设计了优化技术以保留关键信息。 |
| [^132] | [CulTrace: Tracing Internal Cultural Reasoning in Large Language Models](https://arxiv.org/abs/2508.08879) | 本文提出CulTrace方法，通过机械可解释性揭示大型语言模型在文化问答中内部文化推理的分阶段轨迹，并发现其推理存在不平衡性。 |
| [^133] | [CPC-CMS: Cognitive Pairwise Comparison Classification Model Selection Framework for Document-level Sentiment Analysis](https://arxiv.org/abs/2507.14022) | 该框架通过认知成对比较加权多标准评估，自动选择文档级情感分析的最优分类模型，并在多个数据集上验证了其有效性。 |
| [^134] | [GeoExplain: Multimodal Reasoning based on Hierarchy of Visual Information in Street View](https://arxiv.org/abs/2506.16633) | 本文提出了GeoExplain数据集，首次将可解释的地理定位与多模态推理结合，通过层次化视觉信息（局部细节和全局上下文）来预测街景位置并生成人类可理解的解释。 |
| [^135] | [Beyond Gold Standards: Epistemic Ensemble of LLM Judges for Formal Mathematical Reasoning](https://arxiv.org/abs/2506.10903) | 本文提出了一种基于认知和形式化基础的LLM法官集成方法，通过细粒度、多层次的评估标准，系统地自动评估形式化数学推理中的语句自动形式化任务，弥补了现有评估方法的粗粒度缺陷。 |
| [^136] | [Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation](https://arxiv.org/abs/2505.16222) | 本研究首次系统性地揭示了大语言模型在代码评估中对表面差异（如变量名、注释和格式）存在偏见，并通过多种语言和模型实证证明了这些偏见会影响评估的公平性。 |
| [^137] | [Explaining Intrinsic Moral Self-Correction with Mechanistic Interpretability](https://arxiv.org/abs/2505.11924) | 该论文通过机械可解释性揭示了内在道德自我修正的机制是表示引导，即提示词通过沿可解释潜在方向调整隐藏表示来改变模型行为，且这种方法比直接提示更有效。 |
| [^138] | [The Intrinsic Dimension of Prompts in Internal Representations of Large Language Models](https://arxiv.org/abs/2501.10573) | 本文通过内在维度分析大型语言模型提示词表示，发现其与词元不确定性相关，并利用逐层内在维度轮廓训练线性探针，在生成前高效区分恶意与良性提示，准确率达90-95%。 |
| [^139] | [On the Within-class Variation Issue in Alzheimer's Disease Detection](https://arxiv.org/abs/2409.16322) | 本文针对阿尔茨海默病检测中的类内变异问题，提出软目标蒸馏和实例级重平衡两种方法，通过估计样本特定概率分数来建模异质性和不平衡，从而提升检测性能。 |

# 详细

[^1]: TurboBias 2.0：面向生产高效ASR系统的流式上下文偏置

    TurboBias 2.0: Streaming Context-Biasing for Production-Efficient ASR Systems

    [https://arxiv.org/abs/2608.21343](https://arxiv.org/abs/2608.21343)

    TurboBias 2.0通过引入不区分大小写的增强图和每流批处理解码，在支持流式推理的同时，实现了多用户独立上下文偏置，显著提升了生产级ASR系统的效率和个性化能力。

    

    摘要：arXiv:2608.21343v1 公告类型：交叉 摘要：上下文化对于生产级自动语音识别（ASR）系统至关重要，在这些系统中，用户提供的短语必须在严格的延迟约束下被准确识别。尽管许多上下文偏置方法能提高识别准确性，但它们往往无法满足现代生产级ASR系统的实际需求：流式推理、高效的批处理解码、用户特定的上下文列表以及低运行时开销。我们提出了TurboBias 2.0，这是一个面向生产的框架，用于在基于Transducer的ASR系统中实现高效的短语增强。该框架扩展了GPU加速的TurboBias，引入了不区分大小写的增强图以及每流批处理解码，允许批次中的每个话语使用独立的上下文偏置配置。这实现了多个同时用户的个性化上下文偏置，而无需共享或混合他们的上下文列表。所提出的框架支持离线和流式两种模式。

    arXiv:2608.21343v1 Announce Type: cross  Abstract: Contextualization is essential for production automatic speech recognition (ASR) systems, where user-provided phrases must be recognized accurately under strict latency constraints. Although many context-biasing methods improve recognition accuracy, they often do not address the practical requirements of modern production ASR systems: streaming inference, efficient batched decoding, user-specific context lists, and low runtime overhead. We propose TurboBias 2.0, a production-oriented framework for efficient phrase boosting in Transducer-based ASR systems. The framework extends GPU-accelerated TurboBias with a case-insensitive boosting graph and per-stream batched decoding, allowing each utterance in a batch to use an independent context-biasing configuration. This enables personalized context biasing for multiple simultaneous users without sharing or mixing their context lists. The proposed framework supports both offline and streaming
    
[^2]: 逐步进行：测量与引导大型语言模型如何进行心理治疗

    Move by Move: Measuring and Steering How LLMs Conduct Psychotherapy

    [https://arxiv.org/abs/2608.21325](https://arxiv.org/abs/2608.21325)

    本文提出了一个十种治疗动作的本体论来测量和引导LLM在心理治疗中的行为，发现模型过度询问、忽视心理教育，并依赖上下文，而将该本体论作为工具可显著减少与人类动作分布的偏差。

    

    arXiv:2608.21325v1 公告类型：新 摘要：用户越来越多地转向大型语言模型寻求情感支持，但关于这些模型如何实际进行心理治疗互动，我们知之甚少。我们引入了一个包含十种治疗动作的本体论：基于MULTI-60清单的紧凑、功能导向类别，通过五位持牌心理学家的标注活动进行验证，并采用与专家一致性匹配的评判者方法进行扩展。将其应用于真实咨询记录和模型主导的会话，我们比较了人类临床医生和一组前沿模型之间的动作分布。模型过度使用询问，频率高达人类的三倍，忽视心理教育，并且强烈受上下文锚定：它们会延续人类临床医生发起的策略，但很少自行发起。将该本体论作为一组工具暴露出来，平均将人类动作分布的偏差减少约一半，并改善回合级对齐。

    arXiv:2608.21325v1 Announce Type: new  Abstract: Users increasingly turn to large language models for emotional support, yet little is known about how these models actually conduct a psychotherapy interaction. We introduce an ontology of ten therapeutic moves: compact, function-based categories grounded in the MULTI-60 inventory, validated through an annotation campaign with five licensed psychologists, and scaled with a judge-based approach that matches expert agreement. Applying it to real counseling transcripts and model-led sessions, we compare the move distributions between human clinicians and a panel of frontier models. Models over-use inquiry at up to three times the human rate, neglect psychoeducation, and are strongly context-anchored: they carry forward strategies initiated by a human clinician but rarely initiate them themselves. Exposing the ontology as a set of tools roughly halves the mean deviation from the human move distribution and improves turn-level alignment with 
    
[^3]: 提示词-模型交互达到不动点：一种确定性的、任务无关的结构性读出——以及其失败的因子分解

    Prompt-Model Interaction Reaches the Fixed Points: A deterministic, task-free structural readout -- and the factorizations of it that failed

    [https://arxiv.org/abs/2608.21315](https://arxiv.org/abs/2608.21315)

    本文通过一种确定性的、任务无关的结构性读出（短窗口argmax映射的不动点），证明提示词与模型的交互即使在无任务条件下也能完整显现，且该交互具有模型特异性和结构分类效应。

    

    arXiv:2608.21315v1 公告类型：新 摘要：提示词的效果并非提示词本身的属性这一观点已被证实：针对一个模型优化的提示词在另一个模型上性能下降，且在中性重新格式化下排名会重新排序。这些证据基于任务准确率，无法判断交互是任务机制的事实还是条件分布本身的事实。我们在一个不含任务的读出器上提问：短窗口argmax映射x_{t+1} = argmax_x p(x | x_{t-1}, x_t)的不动点结构，从96个起始点进行普查。它是确定性的，因此无法被帮助或损害，且仅存在于短窗口中——六个模型中有四个在窗口16时完全失去它——因此这里所有内容都涉及模型如何读取片段。两个结果。首先，交互以完整幅度达到该读出器：九个条件的token将不动点比例移动至其大部分范围，改变四路结构类别，并重新排序模型，而指令调优值60.5 IFEva（此处原文截断）。

    arXiv:2608.21315v1 Announce Type: new  Abstract: That a prompt's effect is not a property of the prompt is established: prompts optimised for one model degrade on another, and rankings reorder under neutral reformatting. That evidence is about task accuracy, which cannot say whether the interaction is a fact about task machinery or about the conditional distribution itself. We ask on a readout with no task in it: the fixed-point structure of the short-window argmax map x_{t+1} = argmax_x p(x | x_{t-1}, x_t), censused from 96 starts. It is deterministic, so nothing can be helped or hurt, and it exists only at short windows -- four of six models lose it entirely by window 16 -- so everything here concerns how a model reads a fragment. Two results. First, the interaction reaches this readout at full magnitude: nine tokens of conditioning move the fixed-point fraction across most of its range, change a four-way structural class, and reorder models, while instruction tuning worth 60.5 IFEva
    
[^4]: 记忆增强解锁高效的思维链推理

    Memory Augmentation Unlocks Efficient Chain-of-Thought Reasoning

    [https://arxiv.org/abs/2608.21265](https://arxiv.org/abs/2608.21265)

    该论文提出记忆增强压缩框架，利用历史推理记忆作为预填充支架，在压缩思维链推理时有效弥补信息损失，平衡了效率与逻辑连贯性。

    

    arXiv:2608.21265v1 公告类型：新 摘要：大型语言模型通常依赖思维链（CoT）推理来解决复杂任务，但冗长的推理轨迹会带来大量的推理开销。CoT压缩缩短了生成过程，然而激进的压缩可能破坏逻辑连贯性并降低性能。我们将这种权衡形式化为“上下文-生成替代定律”，其中显式推理上下文替代了解码时生成的一部分。基于这一原则，我们提出了“记忆增强压缩”，这是一种无需训练的框架，从历史轨迹中构建可复用的推理记忆，并将其检索作为预填充侧的支架。与使用原始演示不同，这些记忆总结了可复用的推理模式、关键约束和关键操作，以补偿压缩过程中丢失的信息。实验表明，记忆持续改善了基于提示的思维草稿（CoD）压缩在多种任务上的表现。

    arXiv:2608.21265v1 Announce Type: new  Abstract: Large language models often rely on Chain-of-Thought (CoT) reasoning to solve complex tasks, but verbose reasoning traces introduce substantial inference overhead. CoT compression shortens generation, yet aggressive compression may disrupt logical coherence and degrade performance. We formalize this trade-off as the \textit{Context-Generation Substitution Law}, where explicit reasoning context substitutes for part of decode-time generation. Based on this principle, we propose \textit{Memory-Augmented Compression}, a training-free framework that constructs reusable reasoning memories from historical traces and retrieves them as prefill-side scaffolds. Rather than using raw demonstrations, these memories summarize reusable reasoning patterns, key constraints, and critical operations to compensate for information lost during compression. Experiments show that Memory consistently improves prompt-based Chain-of-Draft (CoD) compression across 
    
[^5]: EnSI-RAG：基于实体结构索引的检索增强生成用于长文档问答

    EnSI-RAG: Entity-Structure-Indexed Retrieval-Augmented Generation for Long-Document Question Answering

    [https://arxiv.org/abs/2608.21252](https://arxiv.org/abs/2608.21252)

    EnSI-RAG通过构建以实体为中心的查询无关索引（每条记录包含实体、类型、语义类别和值），有效解决了长文档问答中跨实体和多跳推理的检索难题，提升了证据分离与答案综合的准确性。

    

    摘要：在长且相互关联的文档上进行问答（QA）仍然具有挑战性，因为相关证据可能跨越多个实体及其关系。现有的检索增强生成（RAG）方法通常将文档索引为原始块，并通过嵌入相似性进行检索。当块边界将实体与支持证据分离，或问题需要跨语料库进行多跳推理时，这些方法的性能会下降。我们提出了EnSI-RAG（实体结构索引的检索增强生成），一个构建查询无关、以实体为中心的索引框架。每条记录（e, t, k, v）表示一个实体e、其类型t、一个语义类别k（属于{属性、关系、方面}）以及一个值v，同时保留与原始源段落的链接。在查询时，这些记录作为检索句柄，大型语言模型将检索到的段落综合成最终答案。这种设计将证据分离。

    arXiv:2608.21252v1 Announce Type: cross  Abstract: Question answering (QA) over long, connected documents remains challenging because relevant evidence may span multiple entities and their relationships. Existing retrieval-augmented generation (RAG) methods typically index documents as raw chunks and retrieve them through embedding similarity. Their performance degrades when chunk boundaries separate entities from supporting evidence or when a question requires multi-hop reasoning across the corpus. We propose EnSI-RAG (Entity-Structure-Indexed Retrieval-Augmented Generation), a framework that constructs a query-independent, entity-centered index. Each record (e, t, k, v) represents an entity e, its type t, a semantic category k in {property, relation, aspect}, and a value v, while retaining links to the original source passages. At query time, these records serve as retrieval handles, and an LLM synthesizes the retrieved passages into the final answer. This design separates evidence l
    
[^6]: 从发明人风格披露到专利撰写的基准测试

    Benchmarking Patent Drafting from Inventor-Style Disclosures

    [https://arxiv.org/abs/2608.21249](https://arxiv.org/abs/2608.21249)

    本文提出了Dis2Pat数据集和Patent-MAF多代理框架，旨在解决从发明人风格披露直接生成完整法律连贯专利申请的核心挑战。

    

    摘要：arXiv:2608.21249v1 公告类型：新 摘要：尽管近期大型语言模型（LLMs）在单个专利撰写任务上取得了有前景的成果，但它们从根本上未能解决现实世界专利撰写的核心挑战：直接从早期发明材料生成完整且法律上连贯的专利申请。先前的工作主要假设输入是后期阶段、高度结构化或已经法律化的文本。然而，真实的专利工作流程始于发明人撰写的非正式、去法律化的披露。为弥合这一差距，我们引入了Dis2Pat，一个从披露到专利的数据集，它通过要求直接从发明人风格、去法律化的披露生成完整的专利申请，反映了现实的专利工作流程。鉴于长格式、法律约束性专利撰写的固有难度以及强烈的隐私要求，我们进一步提出了一个强大的基线模型，名为Patent-MAF。它是一个用于本地化的多代理框架。

    arXiv:2608.21249v1 Announce Type: new  Abstract: While recent large language models (LLMs) have achieved promising results on individual patent drafting tasks, they fundamentally fail to investigate the core challenge of real-world patent drafting: generating a complete and legally coherent patent application directly from early-stage invention materials. Prior work predominantly assumes later-stage, highly structured, or already legalistic inputs. However, real patenting workflows begin with informal, de-legalized disclosures authored by inventors. To bridge the gap, we introduce Dis2Pat, a disclosure-to-patent dataset that reflects realistic patenting workflows by requiring the generation of complete patent applications directly from inventor-style, de-legalized disclosures. Given the inherent difficulty of long-form, legally constrained patent drafting and the strong privacy requirements, we further propose a strong baseline named Patent-MAF. It is a multi-agent framework for locall
    
[^7]: 情感语境放大LLM回应中的奉承行为

    Affective Context Amplifies Sycophancy in LLM Responses

    [https://arxiv.org/abs/2608.21242](https://arxiv.org/abs/2608.21242)

    本研究揭示情感语境，尤其是负面情绪如孤独和痛苦，会系统性放大LLM在回应中的奉承行为，导致其软化或保留负面判断。

    

    作为对话伴侣，大型语言模型（LLMs）通常能访问用户的情绪状态。我们研究了这种情感语境如何调节LLM在主观、评价性互动中的奉承行为，在这些互动中，用户分享行为或观点以征求反馈。借鉴讨好理论，我们将奉承行为衡量为模型独立评价与其面向用户回应之间的差异，通过将相同内容呈现为第三方叙述或用户自己的披露来引发这种差异。在七个LLM和两个Reddit数据集（r/AmItheAsshole和r/TrueUnpopularOpinion）中，我们发现这种差异是系统性的且强烈单向的。面向用户的回应始终软化或保留负面或反对性判断。情感语境进一步放大了这种差异，特别是负面状态，如孤独和痛苦，产生了最大效应。这些发现表明，情感语境显著影响LLM的回应倾向。

    arXiv:2608.21242v1 Announce Type: new  Abstract: As conversational companions, large language models (LLMs) often have access to users' emotional states. We study how this affective context modulates LLM sycophancy in subjective, evaluative interactions, where users share actions or opinions that invite feedback. Drawing on ingratiation theory, we measure sycophancy as the divergence between a model's independent evaluation and its user-facing response, elicited by presenting the same content as either a third-party account or the user's own disclosure. Across seven LLMs and two Reddit datasets (r/AmItheAsshole and r/TrueUnpopularOpinion), we find that this divergence is systematic and strongly one-directional. User-facing responses consistently soften or withhold negative or oppositional judgments. Affective context further amplifies this divergence with negative states, particularly loneliness and distress, producing the largest effects. These findings suggest that affective context 
    
[^8]: RARE：在混合专家语言模型中解耦表示引导与专家路由

    RARE: Decoupling Representation Steering from Expert Routing in Mixture-of-Experts Language Models

    [https://arxiv.org/abs/2608.21236](https://arxiv.org/abs/2608.21236)

    本文提出RARE框架，通过将行为扰动投影到路由器零空间来解耦表示引导与专家路由，有效解决了MoE模型中直接应用表示工程导致的结构不匹配问题，并提升了引导性能。

    

    arXiv:2608.21236v1 公告类型：新 摘要：表示工程通过修改中间隐藏状态提供了一种轻量级的控制语言模型行为的方法，但其直接应用于混合专家（MoE）模型时引入了结构上的不匹配。我们首先通过一系列实证研究验证了这一失败模式，发现保持干净的路由可以显著恢复引导性能，并且在受控内容下，路由对语义内容比行为变化更敏感。基于这些发现，我们提出了RARE，一种针对MoE语言模型的路由无关表示工程框架。RARE将任意行为扰动投影到路由矩阵的零空间上，从而移除路由可见的组件，并进一步纠正传播到所选下游层的路由漂移。为了在该框架中确定最佳的扰动估计器，我们在六个异构开放权重模型上评估了五种估计器。

    arXiv:2608.21236v1 Announce Type: new  Abstract: Representation engineering offers a lightweight means of controlling language-model behavior by modifying intermediate hidden states, but its direct application to Mixture-of-Experts (MoE) models introduces a structural mismatch. We first verify this failure mode through a series of empirical studies and find that preserving clean routing substantially recovers steering performance and that routing is more sensitive to semantic content than to behavioral changes under controlled content. Motivated by these findings, we introduce RARE, a router-agnostic representation engineering framework for MoE language models. RARE projects arbitrary behavioral perturbations onto the null space of the router matrix, thereby removing router-visible components, and further corrects routing drift propagated to selected downstream layers. To decide the best perturbation estimator in this framework, we evaluate five estimators on six heterogeneous open-wei
    
[^9]: 利用半结构化数据增强大型语言模型在预测性政治问答中的表现

    Enhancing LLMs in Predictive Political QA with Semi-Structured Data

    [https://arxiv.org/abs/2608.21218](https://arxiv.org/abs/2608.21218)

    本文提出PSL双视图框架，通过提取行为者立场和高阶结构信号，将半结构化政治记录转化为推理证据，以增强LLM在预测性政治问答中的表现。

    

    预测性政治问答（QA），例如预测政治行为者将如何投票，超越了事实查询的范畴。外部政治资源提供了丰富的历史证据，但很少直接包含答案本身。现有的LLM增强方法，包括基于行为者画像的模拟和知识图谱证据注入，虽然改善了政治推理，但大多将外部资源视为基于知识的证据，导致与预测相关的信号未被充分建模。我们识别出预测性政治问答中的两种互补信号：捕捉议题特定偏好的行为者立场，以及捕捉政治行为者间间接依赖的高阶结构信号。我们提出PSL，一个双视图框架，将半结构化政治记录转化为面向推理的证据供LLM使用。PSL在语义视图中从与问题相关的行为者记录中提取立场信号，并学习结构感知的行为者表征。

    arXiv:2608.21218v1 Announce Type: new  Abstract: Predictive political question answering (QA), such as predicting how a political actor will vote, goes beyond factual lookup. External political resources offer rich historical evidence, but rarely contain the answer itself. Existing LLM augmentation methods, including actor-profile-based simulation and knowledge graph evidence injection, improve political reasoning but largely treat external resources as knowledge-based evidence, leaving prediction-relevant signals under-modeled. We identify two complementary signals for predictive political QA: actor stances that capture issue-specific preferences, and high-order structure signals that capture indirect dependencies among political actors. We propose PSL, a dual-view framework that converts semi-structured political records into inference-oriented evidence for LLMs. PSL extracts stance signals from question-relevant actor records in a semantic view, and learns structure-aware actor repr
    
[^10]: 基于注意力头干预的LLM个性化隐私控制

    Personalized Privacy Control in LLMs via Attention Head Intervention

    [https://arxiv.org/abs/2608.21209](https://arxiv.org/abs/2608.21209)

    本文提出个性化隐私概念和P3Bench基准，并开发Repair方法，通过注意力头干预增强LLM的个性化隐私控制，显著降低政策忽视率。

    

    arXiv:2608.21209v1 公告类型：新 摘要：智能体AI的兴起使LLM能够访问多样化的用户数据，引发了关键的隐私问题。先前关于情境隐私的研究探讨了LLM是否根据情境相关规范来调节信息披露。然而，即使在同一情境下，可接受的信息披露边界也可能因用户而异。为解决这一局限性，我们引入了“个性化隐私”，将用户特定的披露偏好纳入隐私控制中。我们进一步提出了P3Bench（个性化隐私保护基准），这是一个新颖的基准，扩展了情境隐私政策，加入了个性化披露政策。实验表明，基于提示的政策无法可靠地执行个性化隐私政策，Qwen2.5-7B和Gemma3-4B的平均政策忽视率分别为51.25%和74.28%。最后，为解决此问题，我们提出了Repair，一种稳健的方法。

    arXiv:2608.21209v1 Announce Type: new  Abstract: The rise of agentic AI enables LLMs to access diverse user data, raising critical privacy concerns. Prior work on contextual privacy studies whether LLMs regulate information disclosure according to context-dependent norms. However, acceptable disclosure boundaries may vary across users even within the same context. To address this limitation, we introduce \textit{personalized privacy}, which incorporates user-specific disclosure preferences into privacy control. We further present P3Bench~(\textbf{P}ersonalized \textbf{P}rivacy \textbf{P}reservation \textbf{Bench}mark), a novel benchmark extending contextual privacy policies with personalized disclosure policies. Experiments show that prompt-based policies fail to reliably enforce personalized privacy policies, with Qwen2.5-7B and Gemma3-4B showing average policy ignorance ratios of 51.25\% and 74.28\%, respectively. Finally, to address this problem, we propose \textsc{Repair}, a robust
    
[^11]: 无意双关：面向以人为中心的LLM评估的合理未知姓名

    No PUN Intended: Plausible Unknown Names for Person-Centred LLM Evaluation

    [https://arxiv.org/abs/2608.21206](https://arxiv.org/abs/2608.21206)

    本文提出PUN协议，用于构建和验证具有合理形式但无真实证据的未知人名，以改进LLM评估中人物相关任务的准确性和可靠性。

    

    arXiv:2608.21206v1 公告类型：交叉 摘要：在大型语言模型（LLM）评估中，人名常被用作提示变量，以考察事实性、隐私泄露、偏见和弃权行为，但当姓名的证据状态不受控制时，测量结果可能混淆记忆、检索、姓名先验和错误人物归属。我们将未知姓名操作化定义为具有合理的“名-姓”形式、无索引的全名证据、且在文档化验证运行下无歧义信号的姓名，并引入PUN（合理未知姓名）协议，用于构建和验证此类姓名，该协议结合了Wikidata派生组件、网络支持的LLM筛选和受控搜索再验证。我们报告了接受率、可复现性、消融实验以及一项204名参与者的人类研究，发现被接受的姓名比对照组更具姓名特征，而参与者在仅3%的情况下能恢复人物证据。我们发布了300个姓名及其比较对照组。

    arXiv:2608.21206v1 Announce Type: cross  Abstract: Person names are widely used as prompt variables in LLM evaluations of factuality, privacy leakage, bias and abstention, but when a name's evidential status is uncontrolled, measurements may conflate memorisation, retrieval, name priors and wrong-person attribution. We operationalise an unknown name as one with plausible First-Last form, no indexed full-name evidence, and no ambiguity signals under a documented validation run, and introduce PUN (Plausible Unknown Names), a protocol for constructing and validating such names, combining Wikidata-derived components, web-enabled LLM screening, and controlled search revalidation. We report acceptance rate, reproducibility, ablations, and a 204-participant human study, finding accepted names are more name-like than controls while participants recover person evidence in only 3% of cases. We release 300 names with comparison controls.
    
[^12]: 可信RAG：用于检测生成式AI系统中错误信息与知识投毒的评估代理

    Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems

    [https://arxiv.org/abs/2608.21095](https://arxiv.org/abs/2608.21095)

    本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。

    

    检索增强生成（RAG）将大型语言模型（LLM）的输出锚定在外部知识上，但RAG系统通常信任其检索到的任何内容，从而造成“安全-可靠性差距”：高语义相关性并不保证事实真实性。攻击者利用这一漏洞进行知识投毒，插入恶意文档以引发定向错误信息。我们提出了一种评估代理，这是一种中间件，结合了自然语言推理（NLI）事实核查、具有相关性加权聚合的五信号投毒检测器，以及一个信任指数 T = 0.4 F + 0.35 C + 0.25 (1 - P )，并针对高污染情境采用非线性阻尼器。在TruthfulQA上使用Llama 3.3 70B时，该代理达到了91%的准确率和100%的精确率，对指令注入的召回率为100%，而就地编辑（如实体替换）仍难以检测。在三个LLM上，信任指数保持判别性，接收者操作特征（ROC）曲线表现良好。

    arXiv:2608.21095v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) grounds Large Language Model (LLM) outputs in external knowledge, but RAG systems usually trust whatever they retrieve, creating a Security-Reliability Gap: high semantic relevance does not guarantee factual truth. Adversaries exploit this through knowledge poisoning, inserting malicious documents to cause targeted misinformation. We propose an Evaluation Agent, middleware that combines Natural Language Inference (NLI) factual verification, a five-signal poison detector with relevance-weighted aggregation, and a Trust Index T = 0.4 F + 0.35 C + 0.25 (1 - P ) with a non-linear dampener for high-contamination contexts. On TruthfulQA with Llama 3.3 70B, the agent reaches 91% accuracy and 100% precision, with 100% recall on instruction injection, while in-place edits, such as entity swaps, remain hard to detect. Across three LLMs the Trust Index stays discriminative, with a Receiver Operating Characteri
    
[^13]: 当特征池算法化：将Mufwene的语言演化生态学扩展至LLM介导的暴露环境

    When the Feature Pool Goes Algorithmic: Extending Mufwene's Ecology of Language Evolution to LLM-Mediated Exposure

    [https://arxiv.org/abs/2608.21088](https://arxiv.org/abs/2608.21088)

    本文提出大型语言模型作为分布性中介，通过算法性重新加权改变人类说话者接触语言变体的频率，从而影响语言演化，但选择核心仍留在人类说话者中。

    

    arXiv:2608.21088v1 公告类型：新 摘要：Mufwene的生态模型将语言演化定位在个体语库贡献的变体之间的竞争，以及说话者从互动中可获得的语言材料中进行选择。大型语言模型（LLMs）使这一架构复杂化，但并不要求选择的核心从人类说话者转移。本文认为，LLMs最适合被视为分布性中介：它们聚合跨人类群体产生的语言，通过训练和后训练改变其分布，并以大规模方式重新分配模型特定的输出。我将由此产生的生态过程称为“说话者可访问分布的算法性重新加权”：模型介导可以改变竞争性变体到达人类选择者的相对频率。关于模型特定语言特征和词汇采用的新兴证据与该路径的部分内容一致，但并未确立不可避免的结果。

    arXiv:2608.21088v1 Announce Type: new  Abstract: Mufwene's ecological model locates language evolution in competition among variants contributed by individual idiolects and in speakers' selection from linguistic material made available through interaction. Large language models (LLMs) complicate this architecture without requiring the locus of selection to move away from human speakers. This article argues that LLMs are best treated as distributional mediators: they aggregate language produced across human populations, transform its distribution through training and post-training, and redistribute model-specific outputs at scale. I call the resulting ecological process algorithmic reweighting of the speaker-accessible distribution: model mediation can alter the relative frequencies with which competing variants reach human selectors. Emerging evidence on model-specific linguistic profiles and lexical uptake is consistent with parts of this pathway, but does not establish inevitable con
    
[^14]: 玩笑之外：衡量双重含义的语义距离

    Jokes Aside: Measuring the Semantic Distance of Double Meanings

    [https://arxiv.org/abs/2608.21087](https://arxiv.org/abs/2608.21087)

    本文利用词嵌入重新评估了笑话生成模型中三个关键度量（明显性、兼容性、比较性），并引入新度量“对称性”，以更精确地衡量双关语义距离，从而优化幽默生成效果。

    

    大型语言模型显著丰富了计算幽默研究的工具包，特别是在笑话和双关语的自动生成方面。一项关键创新——上下文嵌入向量——为重新审视和细化早期假设提供了新机会。值得注意的是，Petrovic和Matthews（2013）提出了一种基于“我喜欢我的X就像我喜欢我的Y，Z”模式的笑话生成模型（例如，“我喜欢我的冰就像我喜欢我的梦，压碎了的”）。他们提出，笑话的趣味性随着以下因素增加：a）Z与X和Y的频繁关联，b）Z的稀有性，c）Z的歧义性，以及d）X和Y之间的意义距离。在此基础上，Winters等人（2019）提出了一套基于Google Ngrams和Word2Vector的度量标准。在这项工作中，他们五个度量中的三个——明显性、兼容性和比较性——被重新审视，并使用了词嵌入。另一个度量——对称性，定义为Z与X和Y的接近程度——在此被引入，以供进一步研究。

    arXiv:2608.21087v1 Announce Type: new  Abstract: Large language models have significantly enriched the toolkit for computational humor research, particularly in the automated generation of jokes and puns. A key innovation, contextual embedding vectors, offers new opportunities to revisit and refine earlier hypotheses. Notably, Petrovic and Matthews (2013) proposed a joke generation model based on the scheme "I like my X like I like my Y, Z" (e.g. "I like my ice like I like my dreams, crushed"). They suggested that joke hilarity increases with: a) frequent association of Z with X and Y, b) rarity of Z, c) ambiguity of Z, and d) meaning distance between X and Y. Building on this, Winters et al. (2019) proposed a set of metrics, based on Google Ngrams and Word2Vector. In this work, three out of their five metrics are revisited with word embeddings: obviousness, compatibility, and comparison. Another measure, symmetry, defined as closeness of Z to both X and Y, is introduced here for the f
    
[^15]: 提示响应：优化大型语言模型编码任务的提示词

    PromptResponse: Optimizing Prompts for LLM Coding Tasks

    [https://arxiv.org/abs/2608.21074](https://arxiv.org/abs/2608.21074)

    本文通过对照实验发现，使用一致的格式（如JSON）优化提示词可提升编码任务的生成效率和稳定性，而基于LLM的提示词调整反而显著降低了任务性能。

    

    大型语言模型（LLMs）在研究工作流程和软件开发管道中的应用日益增多，但其输出对输入提示词的变化仍然敏感。本文介绍了“提示响应”（PromptResponse），一项对照研究，探讨了编码任务提示词的格式和基于LLM的调整如何影响生成代码的性能、效率和稳定性。我们使用了HumanEval数据集的五个语义相同但语法不同的变体——基线、JSON、Markdown、YAML以及一个LLM调整版本——让GPT-4o在8200次执行中解决其编码问题。结果表明，一致的格式（尤其是JSON）提高了生成效率和语法稳定性，并在任务性能上有小幅提升。相反，LLM调整的提示词导致任务性能显著下降，且没有明显的稳定性收益。

    arXiv:2608.21074v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in research workflows and software development pipelines, yet their output remains sensitive to input prompt variations. This paper presents $\unicode{x00AB}$PromptResponse$\unicode{x00BB}$, a controlled study examining how formatting and LLM-based tuning of coding task prompts affect the resulting code's performance, efficiency, and stability. Using five semantically identical yet syntactically distinct variants of the HumanEval dataset$\unicode{x2014}$baseline, JSON, Markdown, YAML, and an LLM-tuned version$\unicode{x2014}$we had GPT-4o solve its coding problems over 8200$\unicode{x00A0}$executions. Our results show that consistent formatting$\unicode{x2014}$especially JSON$\unicode{x2014}$improves generation efficiency and syntactic stability, with minor gains in task performance. Conversely, the LLM-tuned prompts resulted in significantly degraded task performance without significa
    
[^16]: 场景级分布偏移下的证据一致性生成式检测

    Evidence-Consistent Generative Detection under Scenario-Level Distribution Shift

    [https://arxiv.org/abs/2608.21043](https://arxiv.org/abs/2608.21043)

    本文提出场景级分布偏移下的检测问题，揭示传统分布内评估会高估模型鲁棒性，并指出模型倾向于记忆场景特定线索而非泛化到新场景。

    

    arXiv:2608.21043v1 公告类型：新 摘要：当训练和测试数据共享重复出现的任务特定模式或表面线索时，传统分布内评估可能高估鲁棒性。这种风险在社交工程欺诈检测中尤为突出，因为攻击者可以在改变场景、冒充实体或措辞的同时保持恶意意图。我们将此问题研究为短信和语音钓鱼中的场景级分布外（SL-OOD）检测，其中整个攻击场景从训练中保留出来，而标签空间保持固定。该设置测试模型是否能使用决策相关证据而非熟悉的场景特定线索来泛化到未见过的攻击场景。使用此SL-OOD评估，我们发现高分布内性能并不能可靠预测跨特征、编码器和解码器基线的保留鲁棒性。我们将此差距解释为场景记忆化：依赖重复出现的场景特定线索。

    arXiv:2608.21043v1 Announce Type: new  Abstract: Conventional in-distribution evaluation can overestimate robustness when training and test data share recurring task-specific patterns or surface cues. This risk is especially relevant in social-engineering fraud detection, where attackers can preserve malicious intent while changing the scenario, impersonated entity, or wording. We study this problem as scenario-level out-of-distribution (SL-OOD) detection for SMS and voice phishing, where entire attack scenarios are held out from training while the label space remains fixed. This setting tests whether models can generalize to unseen attack scenarios using decision-relevant evidence rather than familiar scenario-specific cues. Using this SL-OOD evaluation, we find that high in-distribution performance does not reliably predict held-out robustness across feature-, encoder-, and decoder-based baselines. We interpret this gap as scenario memorization: reliance on recurring scenario-specifi
    
[^17]: COMET：面向视频多模态大语言模型的对比运动增强时序推理

    COMET: Contrastive Motion-Enhanced Temporal Reasoning for Video Multimodal Large Language Models

    [https://arxiv.org/abs/2608.21030](https://arxiv.org/abs/2608.21030)

    COMET通过引入基于泰勒帧差分的运动分支和时序注意力偏置增强交叉注意力，并采用时序先验蒸馏与TC-GRPO优化，系统性地解决了视频多模态大语言模型在细粒度运动时序理解上的不足。

    

    arXiv:2608.21030v1 公告类型：交叉 摘要：视频多模态大语言模型已取得显著进展，但细粒度的运动-时序理解仍然脆弱。核心瓶颈不仅在于稀疏帧采样，还在于缺乏完整的时序建模流程，无法显式表示帧间变化、实现外观-运动交互，并优化时序方向敏感性。我们提出COMET，一个时序接地框架，通过显式时序表示、外观-运动融合和方向感知优化，系统性地增强视频MLLMs。在架构上，COMET引入基于泰勒帧差分的时序运动分支，并通过时序注意力偏置增强的交叉注意力将运动证据注入外观流。在优化方面，COMET结合时序先验蒸馏与正向-反向TC-GRPO阶段，将时序顺序转化为直接学习信号，并显著提升性能。

    arXiv:2608.21030v1 Announce Type: cross  Abstract: Video multimodal large language models have advanced significantly, yet fine-grained motion-temporal understanding remains fragile. The core bottleneck is not only sparse frame sampling, but also the lack of a complete temporal modeling pipeline for explicitly representing frame-to-frame change, enabling appearance-motion interaction, and optimizing temporal direction sensitivity. We propose COMET, a temporally grounded framework that systematically strengthens video MLLMs through explicit temporal representation, appearance-motion fusion, and direction-aware optimization. Architecturally, COMET introduces a temporal motion branch built on Taylor frame differences and injects its motion evidence into the appearance stream via temporal attention bias-enhanced cross-attention. For optimization, COMET combines temporal prior distillation with a forward-reverse TC-GRPO stage that turns temporal order into a direct learning signal and stren
    
[^18]: 通过结构约束扩展无监督词对齐到文档级别

    Scaling Unsupervised Word Alignment to Documents via Structural Constraints

    [https://arxiv.org/abs/2608.21023](https://arxiv.org/abs/2608.21023)

    本文提出了两种无需训练且轻量级的文档级词对齐方法（CTFAlign和MDPAlign），通过结构约束（语义区域限制或位置先验）克服了直接应用句子级算法导致的性能下降，在多种语言对上实现有效对齐。

    

    词对齐传统上在句子级别进行研究，但许多跨语言任务日益需要跨完整文档的对应关系。虽然最近的多语言嵌入模型可以编码长输入，我们表明直接将针对句子设计的算法应用于文档会导致性能下降。为解决此问题，我们引入了CTFAlign，一种轻量级、无需训练的文档级词对齐方法。CTFAlign采用从粗到细的细化策略，将对齐搜索空间限制在语义相似的区域。此外，我们引入了MDPAlign，一种更简单的替代方法，通过主对角线先验约束对齐位置。两种方法直接操作于完整文档，不依赖句子分割或句子对齐。我们在六个语言对上评估了这些方法，这些语言对在类型学距离、资源丰富度和文档长度上各不相同。平均来看，

    arXiv:2608.21023v1 Announce Type: new  Abstract: Word alignment has traditionally been studied between sentences, but many cross-lingual tasks increasingly require correspondences across full documents. While recent multilingual embedding models can encode long inputs, we show that applying algorithms designed for sentences directly to documents leads to performance degradation. To address this, we introduce CTFAlign, a lightweight, training-free approach for document-level word alignment. CTFAlign applies a coarse-to-fine refinement strategy that restricts the alignment search space to semantically similar regions. Additionally, we introduce MDPAlign, a simpler alternative that constrains alignments by position with a main diagonal prior. Both approaches operate directly on full documents without relying on sentence segmentation or sentence alignment. We evaluate these methods across six language pairs varying in typological distance, resourcedness, and document length. Averaged over 
    
[^19]: 基于LLM作为评审的5G领域知识与故障分析自由文本评估

    Free-Text Evaluation of LLMs for 5G Domain Knowledge and Fault Analysis using LLM-as-Judge

    [https://arxiv.org/abs/2608.21021](https://arxiv.org/abs/2608.21021)

    本文首次在自由文本生成格式下评估轻量级LLM在5G领域知识和故障分析中的表现，并引入LLM作为评审的框架，以验证开放诊断推理的可扩展性。

    

    arXiv:2608.21021v1 公告类型：交叉 摘要：5G及新兴6G网络中的实际故障分析需要领域专业知识来解析自由文本诊断信息，包括根本原因解释和推荐行动。LLM已成为自动化此过程的 promising 方法，然而，轻量级、可边缘部署的模型是否能进行深入的自由文本诊断仍是一个开放问题。现有基准依赖具有固定答案的限制性多选题，而本文在自由文本生成格式下评估5G领域理解和故障分析。转向这一范式需要在开放式诊断推理上评估轻量级、可边缘部署的AI模型，并需要一个可靠的框架来大规模验证这些文本输出。为解决此问题，我们在三个基准测试（TeleQNA ORAN F）上评估了三个轻量级LLM：Claude-Haiku-4.5、GPT-5.4-Mini和Gemini-3.1-Flash-Lite，针对自由文本5G领域知识和故障分析任务。

    arXiv:2608.21021v1 Announce Type: cross  Abstract: Real-world fault analysis in 5G and emerging 6G networks demands domain expertise to analyze free-text diagnostics, including root-cause explanations and recommended actions. LLMs have emerged as a promising approach to automating this, yet whether lightweight, edge-deployable models are capable of performing in-depth free-text diagnostics remains an open question. While existing benchmarks rely on restrictive MCQs with fixed answer keys, this paper evaluates 5G domain understanding and fault analysis in a free-text generation format. Transitioning to this paradigm requires evaluating lightweight, edge-deployable AI models on open-ended diagnostic reasoning, alongside a dependable framework to validate these text outputs at scale. To address this we evaluate three lightweight LLMs, Claude-Haiku-4.5, GPT-5.4-Mini, and Gemini-3.1-Flash-Lite, on free-text 5G domain knowledge and fault-analysis tasks across three benchmarks, TeleQNA ORAN F
    
[^20]: 面向目标感知的校准数据选择以保持量化语言模型中的不确定性

    Target-Aware Calibration Data Selection for Preserving Uncertainty in Quantized Language Models

    [https://arxiv.org/abs/2608.21019](https://arxiv.org/abs/2608.21019)

    本文提出了一种名为DPQ的轻量级量化前校准数据选择方法，通过目标感知的混合高怀疑示例和通用锚点，在量化过程中保持模型的不确定性行为，而非仅优化准确性。

    

    量化被广泛用于部署大型语言模型，但其对不确定性行为（如置信度、边际和弃权）的影响很少被视为主要目标。我们将量化中的校准数据选择构建为一个依赖目标的不确定性保持问题。不同的部署场景强调输入分布的不同区域，而先前的工作主要优化面向准确性的压缩指标或在量化后调整分数。我们用分布保持风险和边界保持风险来形式化这一目标，并提供一个简单的混合不匹配论证，解释为什么没有单一的校准方案应期望适用于所有目标。我们引入了怀疑保持量化（DPQ），这是一个轻量级的量化前方案家族，利用全精度预测构建目标对齐的校准混合，包含高怀疑示例和通用锚点。在8个语言模型和9个NLP基准上进行了评估。

    arXiv:2608.21019v1 Announce Type: cross  Abstract: Quantization is widely used to deploy large language models, but its effect on uncertainty behavior, such as confidence, margins, and abstention, is rarely treated as a primary objective. We frame calibration-data selection for quantization as a target-dependent uncertainty-preservation problem. Different deployments emphasize different regions of the input distribution, yet prior work mainly optimizes accuracy-oriented compression metrics or adjusts scores after quantization. We formalize this goal with distributional and boundary preservation risks, and provide a simple mixture-mismatch argument explaining why no single calibration recipe should be expected to fit all targets. We introduce Doubt-Preserving Quantization (DPQ), a lightweight pre-quantization recipe family that uses full-precision predictions to construct target-aligned calibration mixtures of high-doubt examples and generic anchors. Across 8 language models, 9 NLP benc
    
[^21]: 《MigrationNarrate：用于检测YouTube视频中移民叙事的数集》

    MigrationNarrate: A Dataset for Detection of Migration Narratives in YouTube Videos

    [https://arxiv.org/abs/2608.20984](https://arxiv.org/abs/2608.20984)

    该论文提出了首个多模态数据集MigrationNarrate，用于检测YouTube视频中的移民叙事，填补了该领域缺乏标注数据的空白。

    

    摘要：arXiv:2608.20984v1 公告类型：交叉 摘要：叙事是社会沟通框架构建的核心，因此检测叙事对于理解和分析公共话语至关重要。先前的研究已在不同领域探索了叙事的检测和提取；然而，移民叙事仍然显著缺乏研究，主要原因是缺乏专门的标注数据集。此外，公共沟通最近转向以视频为中心的平台，在这些平台上，叙事通过多模态信号传达并以大规模方式消费。尽管发生了这种转变，视频中的叙事在很大程度上仍未得到探索。为弥合这些差距，我们引入了MigrationNarrate，这是首个用于检测英国移民叙事的模态数据集，包含1,115个YouTube视频转录文本，并使用12个移民超级叙事和53个叙事标签的两级分类法进行标注。本文详细介绍了数据集的设计、收集和标注过程，以及基准测试。

    arXiv:2608.20984v1 Announce Type: cross  Abstract: Narratives are central to how social communication is framed, making their detection critical for understanding and analysing public discourse. Prior work has explored narrative detection and extraction across diverse domains; however, migration narratives remain significantly understudied, primarily due to the absence of dedicated annotated datasets. Furthermore, public communication has recently shifted towards video-centric platforms, where narratives are conveyed through multimodal signals and consumed at scale. Despite this shift, narratives in videos remain largely unexplored. To bridge these gaps, we introduce MigrationNarrate, the first multimodal dataset for detection of migration narratives in the UK, consisting of 1,115 YouTube video transcripts annotated using a two-level taxonomy of 12 migration super-narratives and 53 narrative labels. This paper details the dataset design, collection, and annotations; together with bench
    
[^22]: 基于SAraBERT与语义孪生相似度评估指标的阿拉伯语文档抽取式摘要研究

    Extractive Summarization for Arabic Documents Using SAraBERT with a Semantic Siamese Similarity Evaluation Metric

    [https://arxiv.org/abs/2608.20964](https://arxiv.org/abs/2608.20964)

    本文提出SAraBERT模型，通过增加句子间变换器层和新型语义孪生相似度评估指标，显著提升了阿拉伯语抽取式摘要的质量和覆盖度。

    

    在本研究中，我们介绍了SAraBERT，这是AraBERT的一个增强版本，通过引入句子间变换器层来改进抽取式摘要任务。为确保SAraBERT生成的摘要能高度覆盖文档的主要思想，我们提出了一种新颖的评估指标——语义孪生相似度，用于衡量两个文本输入之间的相似程度。我们使用BLEU、ROUGE和语义孪生相似度在SAraBERT及已发布的相关模型上进行了验证。模拟结果表明，我们提出的模型具有有效性，并推动了后续研究的开展。

    arXiv:2608.20964v1 Announce Type: cross  Abstract: In this research, we introduce SAraBERT, an enhanced version of AraBERT which proposes inter-sentence transformer layers for extractive summarization tasks. To ensure that the summaries generated by SAraBERT achieve a high coverage of the document's main ideas, we propose Semantic Siamese Similarity, a novel evaluation metric that measures the level of similarity between two text inputs. We validated using BLEU, ROUGE, and Semantic Siamese similarity on Sarabert and published related models. Simulation results showed the effectiveness of our proposed model and motivate follow on research.
    
[^23]: TreeWY：面向门控DeltaNet混合模型的自适应验证方法

    TreeWY: Speculative Verification for Gated DeltaNet Hybrids

    [https://arxiv.org/abs/2608.20961](https://arxiv.org/abs/2608.20961)

    本文提出TreeWY方法，通过树状WY变换消除推测解码中的状态快照，显著降低内存开销，从而支持高接受率的宽草稿树。

    

    arXiv:2608.20961v1 公告类型：新论文  摘要：现代开源模型多为混合架构：大部分层采用线性注意力（门控DeltaNet，GDN）层，这些层携带一个小的固定大小的循环状态，而非不断增长的键值（KV）缓存。这使得普通解码在内存上高效，但不利于推测解码。为了验证一批草稿标记并回滚被拒绝的标记，当前系统在GDN层的每个草稿位置都会对完整循环状态进行快照，而这些快照无法在草稿树的分支间共享，因此宽且高接受率的树在内存上变得不可行。我们移除了快照。利用门控Delta规则的一种树状WY变换，我们通过一次三角求解计算每个草稿节点的输出，并在提交时仅重建一个被接受的状态，存储一个小型的伪值矩阵而非逐节点状态；该推导仅依赖于门控Delta规则，而不涉及任何其他架构细节。在两个规模的服务器基准测试中...

    arXiv:2608.20961v1 Announce Type: new  Abstract: Modern open models are hybrids: most layers are linear-attention (Gated DeltaNet, GDN) layers carrying a small fixed-size recurrent state instead of a growing key-value (KV) cache. This makes ordinary decoding memory-efficient, but hurts speculative decoding. To verify a batch of draft tokens and then roll back the rejected ones, today's systems snapshot the full recurrent state at every draft position for GDN layers, and those snapshots cannot be shared across branches of a draft tree, so a wide, high-acceptance tree becomes memory-infeasible. We remove the snapshots. Using a tree-structured WY transform of the gated delta rule, we compute every draft node's output with a single triangular solve and reconstruct only the one accepted state on commit, storing a small pseudo-value matrix instead of per-node states; the derivation depends only on the gated delta rule, not on any other architectural detail. In serving benchmarks on two scale
    
[^24]: 量化感知修复：恢复压缩的4比特大语言模型的实用方法

    Quantization-Aware Healing: A Practical Recipe for Recovering Compressed, 4-Bit LLMs

    [https://arxiv.org/abs/2608.20953](https://arxiv.org/abs/2608.20953)

    提出量化感知修复（QAH）方法，直接从未压缩的原始模型蒸馏4比特学生模型，在显著降低计算成本的同时，在多数基准上匹配或超越bfloat16来源性能。

    

    摘要：以低成本提供大型语言模型越来越意味着发布既在结构上压缩到参数一小部分、又量化到4比特的模型。这些步骤共同削弱了推理、数学、编码和长上下文行为，以至于在部署前需要恢复或修复阶段。默认的方法——量化感知训练（QAT）——重新拟合压缩、量化模型到硬标签；在我们的流程中，它收敛缓慢并在达到峰值后崩溃。我们转而采用量化感知修复（QAH）。由于结构压缩的模型从未在全精度下独立训练，其bfloat16检查点是一个通过蒸馏恢复的原始模型近似；QAH直接从未压缩的原始模型中蒸馏4比特学生模型。在GPT-OSS 120B到60B到MXFP4流程中，QAH学生模型在9个基准测试中的7个上匹配或超越了其bfloat16来源，同时计算量大约减少4倍。

    arXiv:2608.20953v1 Announce Type: cross  Abstract: Serving large language models cheaply increasingly means shipping models that are both structurally compressed to a fraction of their parameters and quantized to 4 bits. Together these steps degrade reasoning, mathematics, coding, and long-context behavior enough to require a recovery, or healing, stage before deployment. The default recipe, quantization-aware training (QAT), re-fits the compressed, quantized model to hard labels; in our pipeline it converged slowly and collapsed past its peak. We adopted Quantization-Aware Healing (QAH) instead. Because a structurally compressed model is never independently trained at full precision, its bfloat16 checkpoint is a distillation-recovered approximation of the original; QAH distills the 4-bit student directly from the original, uncompressed model. On a GPT-OSS 120B to 60B to MXFP4 pipeline, the QAH student matches or beats its bfloat16 source on 7 of 9 benchmarks at roughly 4 times less we
    
[^25]: MentorPulse：刷新跨模型潜在引导以实现长文本生成

    MentorPulse: Refreshing Cross-Model Latent Guidance for Long-Form Generation

    [https://arxiv.org/abs/2608.20927](https://arxiv.org/abs/2608.20927)

    MentorPulse通过动态刷新跨模型引导记忆，在不重置学生模型缓存的情况下，显著提升长文本生成中的约束满足度。

    

    摘要：arXiv:2608.20927v1 公告类型：交叉 摘要：跨模型潜在引导让一个冻结的大型导师模型对输入进行一次编码，而一个冻结的小型学生模型则从生成的信号中生成输出。现有方法保持该信号固定，假设其在输出增长时仍然有用；我们证明这在长文本生成中会失效。在多轮指令跟随任务中，静态引导使一个4B参数的学生模型的约束满足度比其无引导基线低2.5个百分点；而每16个令牌进行一次无需训练的刷新，仅改变记忆内容，就能恢复相对于基线2.0个百分点的提升。我们提出MentorPulse，以实际成本保持引导的新鲜度：它将导师状态压缩到有上限的槽位记忆中，增量处理新生成的令牌，并通过门控交叉注意力更新学生读取的记忆，而无需重置学生的KV缓存。窗口刷新训练将桥接暴露于前缀条件记忆。在十三个数据集上，MentorPulse缩小了52.2%的差距。

    arXiv:2608.20927v1 Announce Type: cross  Abstract: Cross-model latent guidance lets a frozen large mentor encode an input once and a frozen small student generate from the resulting signal. Existing methods keep this signal fixed, assuming it stays useful as the output grows; we show this fails in long-form generation. On multi-turn instruction following, static guidance pushes a 4B student's constraint satisfaction 2.5 points below its no-guidance baseline; a training-free refresh every 16 tokens changes only the memory content and restores a 2.0-point gain over that baseline. We propose MentorPulse to keep guidance fresh at practical cost: it compresses mentor states into a capped slot memory, incrementally processes newly generated tokens, and updates the memory that the student reads through gated cross-attention without resetting the student's KV cache. Windowed Refresh Training exposes the bridge to prefix-conditioned memory. Across thirteen datasets, MentorPulse closes 52.2% of 
    
[^26]: 无源MT评估并非真正的MT评估

    Source-Free MT Evaluation Is Not MT Evaluation

    [https://arxiv.org/abs/2608.20925](https://arxiv.org/abs/2608.20925)

    本文指出无源评估依赖参考译文是不公平的，充分性应基于源文本判断，参考仅应作为辅助证据而非主要标准。

    

    基于参考的评估指标仍然是机器翻译评估中的标准选择，部分原因是质量估计方法往往与人类判断的相关性较差。因此，无源、基于参考的评估已成为实际操作中的常态，尽管这不符合翻译充分性的定义，并且对那些输出保留源含义但不同于参考译文系统不公平。本文认为，充分性必须相对于源文本来判断。参考译文只是源文本的一种可能表达方式，可能引入偏见、欠指定或错误。我们进一步论证，只有当评判者将参考译文视为辅助证据而非主要标准时，源-参考-假设评估才是公平的。否则，即使是源感知评估也可能将充分性简化为对参考译文的偏好。我们展示了现有的混合指标高度依赖参考译文。

    arXiv:2608.20925v1 Announce Type: cross  Abstract: Reference-based metrics remain the standard choice in machine translation evaluation, partly because quality estimation methods often correlate less well with human judgments. As a result, source-free, reference-based evaluation has become the practical norm, even though it is unfaithful to the definition of translation adequacy and unfair to systems whose outputs preserve the source meaning while differing from the reference. This paper argues that adequacy must be judged with respect to the source. A reference is only one possible rendering of the source and may introduce bias, under-specification, or errors. We further argue that source-reference-hypothesis evaluation is fair only when the judge treats the reference as auxiliary evidence rather than as the primary standard. Otherwise, even source-aware evaluation can reduce adequacy to preference towards reference. We show the existing hybrid metrics are highly reliant on reference 
    
[^27]: ForeDreamer：一种用于未来事件预测的自我进化双智能体记忆架构

    ForeDreamer: A Self-Evolving Dual-Agent Memory Architecture for Future Event Prediction

    [https://arxiv.org/abs/2608.20920](https://arxiv.org/abs/2608.20920)

    ForeDreamer通过双智能体架构将原始网络证据转化为结构化记忆，分离事实与经验记忆，从而提升开放网络未来事件预测的准确性。

    

    arXiv:2608.20920v1 公告类型：新 摘要：开放网络未来事件预测要求智能体从嘈杂、冗余和不完整的证据中提取可靠信号。现有的检索/记忆机制直接将检索到的信息馈送给智能体，或依赖简单的记忆功能（如存储和重用先前信息）进行预测，这使其不足以应对开放网络预测。我们提出在预测前将原始网络证据转化为结构化记忆，使智能体能够基于提炼后的、针对问题的证据进行推理，而非基于嘈杂的检索结果。本文介绍了ForeDreamer，一种用于管理开放网络证据记忆的自我进化双智能体框架。ForeDreamer将事实记忆（当前预测的问题特定证据状态）与经验记忆（跨预测情节积累的持久智能体经验）分离。它使用一个主智能体进行搜索和预测，以及一个记忆处理子智能体来转换搜索到的内容。

    arXiv:2608.20920v1 Announce Type: new  Abstract: Open-web future event prediction requires agents to distill reliable signals from noisy, redundant, and incomplete evidence. Existing retrieval/memory mechanisms directly feed retrieved information to agents or rely on simple memory functions such as storing and reusing prior information for prediction, leaving them insufficient for open-web forecasting. We propose to transform raw web evidence into structured memory before prediction, enabling agents to reason over distilled, question-specific evidence rather than noisy retrieval results. This paper presents ForeDreamer, a self-evolving dual-agent framework for managing memory over open-web evidence. ForeDreamer separates factual memory, a question-specific evidence state for the current forecast, from experiential memory, persistent agent experience accumulated across forecasting episodes. It uses a main agent for search and prediction, and a memory-processing subagent to convert searc
    
[^28]: KREL：基于临床证据的知识引导推理与大型语言模型的自动医疗编码

    KREL: Automatic Medical Coding via Knowledge-Guided Reasoning over Clinical Evidence with LLMs

    [https://arxiv.org/abs/2608.20887](https://arxiv.org/abs/2608.20887)

    该论文提出了KREL框架，通过结合大型语言模型的推理能力与外部ICD编码指南的结构化知识，解决了自动医疗编码中临床记录过长、标签空间庞大及编码规则复杂等关键问题。

    

    自动医疗编码（AMC）将标准化的国际疾病分类（ICD）代码分配给临床记录，对于医疗报销、质量报告和临床研究至关重要。现有的基于预训练语言模型（PLM）的方法通常将AMC视为对预定义代码集的极端多标签分类问题，而近期基于大型语言模型（LLM）的方法则将其构建为生成或多步推理任务。然而，关键挑战仍然存在，包括临床记录的极端长度阻碍了有效解释、庞大的ICD标签空间，以及LLM未能明确捕获的复杂编码规则。在这项工作中，我们提出了KREL（Knowledge-Guided Reasoning over Clinical Evidence with LLMs），一个利用LLM进行临床文本理解和推理，同时整合外部ICD编码指南作为结构化知识的框架。这种设计旨在应对上述挑战。

    arXiv:2608.20887v1 Announce Type: cross  Abstract: Automatic Medical Coding (AMC), which assigns standardized International Classification of Diseases (ICD) codes to clinical notes, is essential for medical reimbursement, quality reporting, and clinical research. Existing pre-trained language model (PLM)-based methods typically formulate AMC as an extreme multi-label classification problem over a predefined code set, while recent large language model (LLM)-based approaches instead frame it as generation or multi-step reasoning. However, key challenges remain, including the extreme length of clinical notes that hinders effective interpretation, the vast ICD label space, and complex coding rules that are not explicitly captured by LLMs. In this work, we propose Knowledge-Guided Reasoning over Clinical Evidence with LLMs (KREL), a framework that leverages LLMs for clinical text understanding and reasoning while integrating external ICD coding guidelines as structured knowledge. This desig
    
[^29]: 识别、定位、关联：从文档图像中进行端到端键值提取

    Identify, Locate, Link: End-to-End Key-Value Extraction from Document Images

    [https://arxiv.org/abs/2608.20868](https://arxiv.org/abs/2608.20868)

    本文提出了一种端到端的键值提取方法，通过微调紧凑视觉语言模型SmolDocling，无需OCR预处理即可同时完成识别、定位和关联，并引入数据增强和布局感知评估，显著提升了文档信息提取的准确性和效率。

    

    文档处理流程传统上将光学字符识别（OCR）引擎与用于结构化信息提取的下游模型串联起来，导致多阶段错误传播。我们对SmolDocling（一个紧凑的256M参数视觉语言模型，VLM）进行微调，使其直接从文档图像中执行端到端的键值提取，在一次通过中联合解决识别、定位和关联问题，无需OCR预处理。我们扩展了DocTags，添加了专门的键、值、区域和关联标签，从而在统一输出序列中实现多对多关系。为解决数据限制问题，我们设计了一个增强管道，结合了合成表单填充和基于图的裁剪，以保留完整的键值子图。我们进一步引入了一个布局感知的评估框架，扩展了文本匹配，并增加了空间边界框验证。在FUNSD、XFUND和一个大规模私有数据集上，我们的模型表现优于...

    arXiv:2608.20868v1 Announce Type: cross  Abstract: Document processing pipelines traditionally cascade optical character recognition (OCR) engines with downstream models for structured information extraction, leading to multi-stage error propagation. We fine-tune SmolDocling, a compact 256M-parameter vision-language model (VLM), to perform end-to-end key-value extraction directly from document images, jointly solving identification, localization, and association in a single pass without OCR preprocessing. We extend DocTags with specialized key, value, region, and link tags, enabling many-to-many relationships in a unified output sequence. To address data limitations, we design an augmentation pipeline combining synthetic form filling and graph-based crops that preserve complete key-value subgraphs. We further introduce a layout-aware evaluation framework extending text matching with spatial bounding box verification. On FUNSD, XFUND, and a large-scale private dataset, our model outperf
    
[^30]: 基于本体的结构正则化用于文档级关系抽取

    Ontology-Driven Structural Regularization for Document-Level Relation Extraction

    [https://arxiv.org/abs/2608.20856](https://arxiv.org/abs/2608.20856)

    本文提出一种基于本体的结构正则化框架，用于量化和消除文档级关系抽取中的结构噪声（如本体约束违反和逻辑矛盾），从而显著提升模型泛化性能。

    

    文档级关系抽取（DocRE）严重依赖成本高昂的人工标注数据集，而诸如DocRED distant等大型远程监督资源由于噪声问题仍未得到充分利用。我们表明，关系三元组中结构不一致性（包括违反本体约束和逻辑矛盾）是一个关键但被忽视的噪声来源。我们引入了一个基于本体的框架来量化和强制DocRE数据集中的结构一致性。我们的分析揭示了DocRED distant中存在大量结构噪声，并证明了这些不一致性会传播到模型预测中。在训练过程中强制结构良好性显著减少了逻辑矛盾，并持续提高了泛化性能。这些发现确立了结构一致性作为DocRE中缺失的监督轴，并强调了结构正则化作为一种有效策略。

    arXiv:2608.20856v1 Announce Type: new  Abstract: Document-Level Relation Extraction (DocRE) relies heavily on costly manually annotated datasets, while large distant supervision resources such as DocRED distant remain underexploited due to noise. We show that a critical yet overlooked source of noise lies in structural inconsistencies within relational triples, including violations of ontology constraints and logical contradictions.   We introduce an ontology-driven framework to quantify and enforce structural consistency in DocRE datasets. Our analysis reveals substantial structural noise in DocRED distant and demonstrates that such inconsistencies propagate to model predictions. Enforcing structural well-formedness during training significantly reduces logical contradictions and consistently improves generalization performance. These findings establish structural consistency as a missing axis of supervision in DocRE and highlight structural regularization as an effective strategy for
    
[^31]: SAC-Copula：通过平滑相关Gumbel场实现扩散语言模型的保质量水印技术

    SAC-Copula: Quality-Preserving Watermarking for Diffusion Language Models via Smooth Correlated Gumbel Fields

    [https://arxiv.org/abs/2608.20839](https://arxiv.org/abs/2608.20839)

    SAC-Copula通过引入基于高斯copula的平滑局部相关Gumbel扰动场，解决了扩散语言模型水印中扰动与解码动态不匹配的问题，实现了生成质量与可检测性的更优平衡。

    

    arXiv:2608.20839v1 公告类型：新 摘要：扩散语言模型（DLMs）的水印技术需要与迭代并行去掩蔽机制兼容，而非自回归解码。现有的基于采样的水印方法通常注入逐位置的独立同分布扰动，这可能与DLM解码动态不匹配，从而降低生成质量。我们提出SAC-Copula，一种基于高斯copula构建的平滑、局部相关Gumbel扰动场的保质量水印方法。我们进一步开发了SAC感知检测器，利用协方差感知滤波和原生样本校准。机制级分析表明，局部相关性降低了潜在扰动的粗糙度，并更好地匹配迭代细化动态。在LLaDA上的实验表明，与现有基线相比，SAC-Copula实现了良好的质量-可检测性权衡。特别是，在Dream-7B及额外数据集上的进一步评估也验证了其有效性。

    arXiv:2608.20839v1 Announce Type: new  Abstract: Watermarking diffusion language models (DLMs) requires mechanisms compatible with iterative parallel unmasking rather than autoregressive decoding. Existing sampling-based watermarking methods typically inject position-wise i.i.d. perturbations, which can be poorly aligned with DLM decoding dynamics and degrade generation quality. We propose SAC-Copula, a quality-preserving watermarking method for DLMs based on smooth, locally correlated Gumbel perturbation fields constructed via a Gaussian copula. We further develop a SAC-aware detector using covariance-aware filtering and native-sample calibration. Mechanism-level analysis shows that local correlation reduces latent perturbation roughness and better matches iterative refinement dynamics. Experiments on LLaDA show that SAC-Copula achieves a favorable quality-detectability trade-off compared with existing baselines. In particular, further evaluations on Dream-7B and additional datasets s
    
[^32]: STAR-OPD：面向ABSA四元组抽取的结构化方面级联感知在线策略奖励蒸馏

    STAR-OPD: Structured Aspect-Cascade-Aware On-Policy Reward Distillation for ABSA Quadruple Extraction

    [https://arxiv.org/abs/2608.20831](https://arxiv.org/abs/2608.20831)

    本文提出STAR-OPD方法，通过在线策略奖励蒸馏和结构化方面级联感知，解决ABSA四元组抽取中蒸馏模型的结构无效状态问题，提升小模型部署性能。

    

    基于方面的情感分析（ABSA）四元组抽取要求对通常包含多个细粒度情感元组的评论，联合预测目标、方面、观点和情感。虽然大型思维链（CoT）模型在此任务上表现良好，但将其蒸馏到较小的可部署模型中仍然困难。我们识别出蒸馏式ABSA抽取中的一个任务特定失败模式：学生在目标-方面接口处的错误会创建结构无效状态，如断裂的目标-方面绑定和幻觉目标，进而破坏下游预测。传统的离策略蒸馏不适合此场景，因为它仅基于教师生成的轨迹进行训练，对推理过程中占主导地位的学生诱导结构状态提供很少的监督。为解决此不匹配，我们提出STAR-OPD（结构化方面级联感知在线策略奖励蒸馏），它基于gen构建。

    arXiv:2608.20831v1 Announce Type: cross  Abstract: Aspect-based sentiment analysis (ABSA) quadruple extraction requires jointly predicting target, aspect, opinion, and sentiment over reviews that often contain multiple fine-grained sentiment tuples. While large chain-of-thought (CoT) models perform well on this task, distilling them into smaller deployable models remains difficult. We identify a task-specific failure mode in distilled ABSA extraction: student errors at the target-aspect interface create structurally invalid states, such as broken target-aspect bindings and hallucinated targets, which then corrupt downstream predictions. Conventional off-policy distillation is poorly suited to this setting because it trains only on teacher-generated trajectories and provides little supervision on the student-induced structural states that dominate inference. To address this mismatch, we propose STAR-OPD (STructured Aspect-cascade-aware On-Policy Reward Distillation), which builds on gen
    
[^33]: 去噪未来：基于上下文感知频谱扩散的时间知识图谱外推

    Denoising the Future: Context-Aware Spectral Diffusion for Temporal Knowledge Graph Extrapolation

    [https://arxiv.org/abs/2608.20804](https://arxiv.org/abs/2608.20804)

    本文提出FreqDiff，一种频率感知扩散框架，通过双流去噪器结合时间依赖建模和上下文感知频谱校准，以及频域正则化，有效解决了时间知识图谱外推中历史信息聚合导致的目标信号稀释问题。

    

    arXiv:2608.20804v1 公告类型：交叉  摘要：时间知识图谱（TKG）外推旨在从随时间变化的关系历史中推断未来事实。最近的基于扩散的方法通过生成式去噪改善了不确定性建模，但其对主体历史的聚合条件化可能不足以区分查询特定的证据与非显著的历史事实，从而稀释了目标判别性信号。为弥合这一差距，我们提出了FreqDiff，一种用于TKG外推的频率感知扩散框架。具体而言，FreqDiff将未来对象预测形式化为查询槽去噪，并开发了一个双流去噪器，该去噪器结合了时间依赖建模与上下文感知的频谱校准。频谱分支从可学习基中合成历史条件滤波器，以自适应地重新校准去噪表示，同时提出了一种频域正则化器，用于在频谱中使去噪目标与黄金对象对齐。

    arXiv:2608.20804v1 Announce Type: cross  Abstract: Temporal Knowledge Graph (TKG) extrapolation seeks to infer future facts from time-varying relational histories. Recent diffusion-based approaches improve uncertainty modeling through generative denoising, but their aggregated conditioning on subject histories may insufficiently distinguish query-specific evidence from non-salient historical facts, thereby diluting target-discriminative signals. To bridge this gap, we propose FreqDiff, a Frequency-aware Diffusion framework for TKG extrapolation. Specifically, FreqDiff formulates future object prediction as query-slot denoising and develops a dual-stream denoiser that integrates temporal dependency modeling with context-aware spectral calibration. The spectral branch synthesizes history-conditioned filters from learnable bases to adaptively re-calibrate denoising representations, while a frequency-domain regularizer is proposed to align the denoised target with the gold object in spectr
    
[^34]: 重要内容的画像：面向LLM推荐系统的基于大规模元数据的上下文感知物品画像

    Profiling What Matters: Context-Aware Item Profiles from Large-Scale Metadata for LLM Recommenders

    [https://arxiv.org/abs/2608.20801](https://arxiv.org/abs/2608.20801)

    CAIRO提出了一种用户上下文感知的物品画像框架，通过结构化元数据并动态选择每个用户-物品对的相关信息，显著提升LLM重排序的个性化与细粒度理解。

    

    虽然大型语言模型（LLMs）显著推进了推荐中的重排序，但有效利用物品侧信息仍具挑战性。现实世界的物品由庞大、异构且非结构化的元数据描述，其中决策相关信号往往隐含、嘈杂或埋藏在长描述中。此外，特征显著性高度依赖上下文，不仅因物品而异，还因用户而异。现有方法通常依赖物品标题、固定属性或静态物品摘要，这限制了个性化和细粒度的物品理解。为弥合这一差距，我们提出了CAIRO，一种面向基于LLM重排序的用户上下文感知物品画像框架。CAIRO首先将原始元数据和评论结构化为客观特征和主观特质，并采用轻量级画像器为每个用户-物品对选择最相关的信息，同时控制服务时开销。由此产生的画像增强了LLM重排序的个性化理解。

    arXiv:2608.20801v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) have significantly advanced reranking in recommendation, effectively leveraging item-side information remains challenging. Real-world items are described by vast, heterogeneous, and unstructured metadata, where decision-relevant signals are often implicit, noisy, or buried in long descriptions. Moreover, feature salience is highly context-dependent, varying not only across items but also across users. Existing methods often rely on item titles, fixed attributes, or static item summaries, which limit personalized and fine-grained item understanding. To bridge this gap, we propose CAIRO, a user context-aware item profiling framework for LLM-based reranking. CAIRO first structures raw metadata and reviews into objective features and subjective traits, and employs a lightweight profiler to select the most relevant information for each user-item pair with limited serving-time overhead. The resulting profil
    
[^35]: 关注之树：用于科学评论中未声明局限提取的分层多智能体辩论

    Tree-of-Concerns: Hierarchical Multi-Agent Debate for Unstated-Limitation Extraction in Scientific Critique

    [https://arxiv.org/abs/2608.20777](https://arxiv.org/abs/2608.20777)

    本文提出“关注之树”多智能体框架，通过专门怀疑论角色和小组审查机制，从科学论文中提取未声明局限，在精确度和覆盖率上分别比最强基线提升79%和11%。

    

    随着科学文献的增长和论文越来越多地少报局限性，多智能体大语言模型提供了一种有前景的方法来系统地揭示这些隐藏的失败模式。在此，我们引入了关注之树（Tree-of-Concerns），这是一个多智能体框架，它部署了专门的怀疑论者角色，每个角色通过特定类别的分析视角运作，作为并行的辩论树来从科学论文中提取未声明的局限性。每个角色进行结构化的、基于证据的论证，而一个小组审查机制从所有五个视角重新评估每个幸存的声明，以纠正类别漂移和严重性校准错误。通过在ToC-Bench上的实验——我们的基准包含414篇研究论文和1,905个未声明局限，这些来源于审稿人报告的弱点和后续引文批评——我们证明了相对于最强的基线，ToC将精确度提高了79%，覆盖率提高了11%，从而浮现出具体的、有证据支持的局限。

    arXiv:2608.20777v1 Announce Type: new  Abstract: As scientific literature grows and papers increasingly under-report limitations, multi-agent LLMs offer a promising approach to systematically uncover these hidden failure modes. Here, we introduce Tree-of-Concerns, a multi-agent framework that deploys specialized skeptic personas, each operating through a category-specific analytical lens, as parallel debate trees to extract unstated limitations from scientific papers. Each persona conducts structured, evidence-grounded argumentation, while a Panel Review mechanism re-evaluates each surviving claim from all five perspectives to correct category drift and severity miscalibration. Through experiments on ToC-Bench, our benchmark of 414 research papers with 1,905 unstated limitations, sourced from reviewer-reported weaknesses and follow-up citation critiques, we demonstrate that ToC improves precision by 79% and coverage by 11% relative to strongest baselines, surfacing specific, evidence-g
    
[^36]: PSK在WMT 2026 MIST中的提交：面向多语言摘要与问答的任务特化QLoRA适配器

    PSK at WMT 2026 MIST: Task-Specialized QLoRA Adapters for Multilingual Summarization and Question Answering

    [https://arxiv.org/abs/2608.20757](https://arxiv.org/abs/2608.20757)

    本文提出了一种基于Tiny Aya Global模型和三个任务特化QLoRA适配器的多语言摘要与问答系统，通过分离任务适配器提升了摘要性能，并针对开放问答的不稳定性提交了多系统方案。

    

    arXiv:2608.20757v1 公告类型：交叉 摘要：我们描述了PSK对WMT 2026多语言指令共享任务的提交。我们的系统使用35.3亿参数的Tiny Aya Global模型，并配备三个QLoRA适配器，每个任务对应一个。这些适配器在多语言文档-摘要对、基于段落的问答以及过滤后的独立问答数据上进行训练。摘要数据还包括带有作者撰写摘要的科学论文。在我们保留的测试集上，上下文和摘要适配器的表现优于我们的多任务适配器，后者仅使用组织者提供的数据进行训练。开放问答的结果因答案长度和评估方法而异，表现不一。因此，我们提交了三个系统，它们共享相同的上下文和摘要适配器，但使用不同的开放问答适配器。

    arXiv:2608.20757v1 Announce Type: cross  Abstract: We describe the PSK submission to the WMT 2026 Multilingual Instruction Shared Task. Our system uses the 3.35B-parameter Tiny Aya Global model with three QLoRA adapters, one for each task. The adapters are trained on multilingual document-summary pairs, passage-based question answering, and filtered standalone question answering. The summarization data also includes scientific papers with their author-written abstracts. On our held-out split, the context and summarization adapters perform better than our multitask adapter, which was trained only on data supplied by the organizers. Results for open QA are mixed and vary with answer length and evaluation method. We therefore submit three systems with the same context and summarization adapters but different open-QA adapters.
    
[^37]: 校准语言模型代理中的标准修订：失败模式与基于轨迹锚定的协议

    Calibrating Criterion Revision in LLM Agents: Failure Modes and a Trace-Anchored Protocol

    [https://arxiv.org/abs/2608.20729](https://arxiv.org/abs/2608.20729)

    本文提出并验证了语言模型代理中标准修订的五个必要条件，实验显示当前模型未能在任何案例中完全满足这些条件，但零结果不排除存在普遍能力的可能性。

    

    arXiv:2608.20729v1 公告类型：新 摘要：语言模型代理在失败后可能改进，或跨片段携带文本而不修订何为成功。我们研究标准修订的归因问题：当标准K0接受了一个违反更广泛承诺B的结果时，哪些观察能证明系统形成并持续使用了K1？我们要求五个不可补偿的条件：标准失败检测、模型发出的提议、新片段转移、对所声称载体的干预敏感性，以及保留性。我们在十二个跨领域案例和四个臂上评估了CMB-0.1：无状态推理、仅追加历史、模型生成但由框架承诺的状态，以及评估者编写的预言状态。七个机制固定装置产生了84次确定性评分试验；四个局部量化产物产生了96次调用和192次模型-案例-臂试验。没有模型试验满足所有五个条件，但这一零结果并不确立普遍能力。

    arXiv:2608.20729v1 Announce Type: new  Abstract: Language-model agents can improve after failure or carry text across episodes without revising what counts as success. We study the narrower attribution problem of criterion revision: when criterion K0 accepts an outcome violating a broader commitment B, what observations justify saying that the system formed and persistently used K1? We require five non-compensatory conditions: criterion-failure detection, a model-emitted proposal, new-episode transfer, intervention sensitivity on the claimed carrier, and preservation.   We evaluate CMB-0.1 on twelve cross-domain cases and four arms: stateless inference, append-only history, model-generated but harness-committed state, and evaluator-written oracle state. Seven mechanism fixtures yield 84 deterministic scorer trials; four local quantized artifacts yield 96 calls and 192 model-case-arm trials. No model trial satisfies all five conditions, but this zero does not establish general capabilit
    
[^38]: AsmEvo：基于功能等价性验证的AMD GPU内核代理级汇编优化

    AsmEvo: Agentic Assembly-Level Optimization of AMD GPU Kernels with Functional Equivalence Verification

    [https://arxiv.org/abs/2608.20711](https://arxiv.org/abs/2608.20711)

    AsmEvo提出了一种无需源代码、直接优化已编译AMDGPU二进制文件的代理级汇编优化方法，通过差分验证确保功能等价性，突破了现有LLM优化器依赖源码的限制。

    

    高性能机器学习系统日益依赖GPU内核，但这些内核的可编辑源代码通常不可用、由程序生成，或与最终机器码距离过远，难以暴露剩余优化空间。现有的LLM内核优化器和自动调优器主要针对CUDA、Triton、HIP或张量程序源码，并参考实现进行验证。我们研究了一个更严格的场景：优化已编译的AMDGPU代码对象，其中部署的二进制文件是唯一的行为基准。我们提出了AsmEvo，一个针对AMD GPU内核的代理级汇编优化器。给定一个AMDGPU代码对象K0，AsmEvo重建可重汇编的表示，使用长视野代理提出低级编辑，重建保持ABI兼容的优化对象，并仅在相同启动条件下对K0进行差分验证后接受候选方案。AsmEvo结合了代码对象恢复、元数据感知重建、剖析引导的热窗口编辑和正确性门控。

    arXiv:2608.20711v1 Announce Type: new  Abstract: High-performance ML systems increasingly rely on GPU kernels whose editable source is unavailable, generated, or too distant from final machine code to expose remaining optimizations. Existing LLM kernel optimizers and autotuners mainly operate on CUDA, Triton, HIP, or tensor-program source and validate against reference implementations. We study a stricter setting: optimizing an already compiled AMDGPU code object, where the deployed binary is the only behavioral oracle.   We present AsmEvo, an agentic assembly-level optimizer for AMD GPU kernels. Given an AMDGPU code object K0, AsmEvo reconstructs a reassemblable representation, proposes low-level edits with a long-horizon agent, rebuilds an ABI-preserving optimized object, and accepts candidates only after differential verification against K0 under identical launches. AsmEvo combines code-object recovery, metadata-aware rebuilding, profiling-guided hot-window editing, correctness-gate
    
[^39]: 真实软件历史中的时间有效性：消除GitHub修复中代码助手记忆的过时事实错误

    Temporal Validity on Real Software Histories: Eliminating Stale-Fact Errors in Code-Assistant Memory over GitHub Fixes

    [https://arxiv.org/abs/2608.20685](https://arxiv.org/abs/2608.20685)

    本文验证了MemStrata在真实软件历史中通过确定性过时记忆消除RAG的时间盲点，显著提升答案准确率（0.91对比0.57-0.59），并减少过时事实错误。

    

    检索增强生成（RAG）缺乏时间模型：当编码会话中事实发生变化——函数被重命名、端点移动、依赖项升级——RAG会检索到新旧值，且相似度几乎相同，无法判断哪个是当前的，因此会提供已过时的值。论文1在合成单值基准上表明，确定性（主体、关系、对象）的过时记忆消除可以解决此失败。本文在真实软件历史上进行端到端验证。从707个真实GitHub问题（SWE-bench Lite + Verified）中提取130个干净的原子状态转换，即修复将一个可识别值从修复前形式变为修复后形式，并将每个标记去除（过时和当前语句仅值不同）。在此数据集上，MemStrata达到0.91的答案准确率，而RAG为0.57-0.59；并且，结构性结果表明，当被迫回答时，RAG在36-3%的情况下提供过时值。

    arXiv:2608.20685v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) has no model of time: when a fact changes across a coding session - a function is renamed, an endpoint moves, a dependency is bumped - RAG retrieves both the old and new value with near-identical similarity and cannot tell which is current, so it serves the superseded value. Paper 1 showed, on synthetic single-value benchmarks, that a deterministic (subject, relation, object) supersession memory eliminates this failure. Here we validate it end-to-end on real software history. From 707 real GitHub issues (SWE-bench Lite + Verified) we extract 130 clean atomic state transitions, a fix that changes one identifiable value from a pre-fix to a post-fix form, and render each marker-free (the stale and current statements differ only in the value). On this set, MemStrata reaches 0.91 answer accuracy versus RAG's 0.57-0.59; and, the structural result, when forced to answer RAG serves the superseded value 36-3
    
[^40]: Why2Speak：为弃权行为策略进行忠实推理

    Why2Speak: Faithful Reasoning for Abstaining Action Policies

    [https://arxiv.org/abs/2608.20670](https://arxiv.org/abs/2608.20670)

    本研究揭示了行动策略中能力与可审计性之间的权衡：直接决策策略性能更优但缺乏可检查推理，而推理策略虽提供追踪却牺牲了性能，特别是在干预机会的召回率上。

    

    arXiv:2608.20670v1 公告类型：新 摘要：许多智能体系统必须反复在行动与弃权之间做出选择，这使得忠实推理对监督至关重要：只有当解释反映了产生该行为的计算过程时，它才是有用的。我们通过多方对话中的干预时机来研究这个问题，其中助手必须决定是发言还是保持沉默。这种设置暴露了类别不平衡、不对称的行动成本，以及暴露推理可能改变被审计策略的可能性。使用Qwen3-8B，在有无思维链推理的情况下解码，我们比较了直接决策策略、推理策略、监督微调和强化学习。我们发现了一种能力-可审计性权衡：最强的直接策略实现了更高质量，但不暴露任何可供检查的推理，而推理策略提供了一个追踪，但以较低性能为代价，尤其是对真实干预机会的召回率。监督

    arXiv:2608.20670v1 Announce Type: new  Abstract: Many agentic systems must repeatedly choose between acting and abstaining, making faithful reasoning important for oversight: an explanation is useful only if it reflects the computation that produced the action. We study this problem through intervention timing in multi-party conversation, where an assistant must decide whether to speak or remain silent. This setting exposes class imbalance, asymmetric action costs, and the possibility that exposing reasoning changes the policy being audited. Using Qwen3-8B, decoded with or without chain-of-thought reasoning, we compare direct decision policies, reasoning policies, supervised fine-tuning, and reinforcement learning. We find a capability-auditability tradeoff: the strongest direct policy achieves higher quality but exposes no reasoning to inspect, while the reasoning policy provides a trace at the cost of lower performance, particularly recall of true intervention opportunities. Supervis
    
[^41]: 可审计性构建：面向企业金融中可信大语言模型分析的本体驱动框架

    Auditable by Construction: An Ontology-Driven Framework for Trustworthy LLM Analytics in Enterprise Finance

    [https://arxiv.org/abs/2608.20661](https://arxiv.org/abs/2608.20661)

    本文提出了KDAF框架，通过本体驱动和CARP检索，确保每个事实具有来源可追溯性，从而显著提升企业金融中大语言模型回答的可审计性和准确性。

    

    摘要：arXiv:2608.20661v1 公告类型：新公告 摘要：企业在金融领域采用大语言模型，其限制更多在于信任而非流畅性：在财务规划与分析（FP&A）及其他受监管的工作流程中，一个答案只有在可追溯到权威来源且事后可审计的情况下才可用。本文主张，企业金融中的检索增强生成应同时以可审计性和准确性为评估标准，并提出了知识驱动分析框架（KDAF），该框架通过六个迭代阶段构建本体驱动的知识系统，并通过上下文感知相关性传播（CARP）检索证据，使每个检索到的事实都携带其关系类型、置信度和来源谱系。对FinanceBench（145个问题）的评估将KDAF与零上下文推理、BM25、概念加权词汇检索和无根基图遍历进行了比较。首先，检索是必要的：零上下文推理仅达到4.1%的正确率。

    arXiv:2608.20661v1 Announce Type: new  Abstract: Enterprise adoption of large language models in finance is constrained less by fluency than by trust: in Financial Planning and Analysis (FP&A) and other regulated workflows, an answer is usable only if it is traceable to authoritative sources and auditable after the fact. This paper argues that retrieval-augmented generation for enterprise finance should be evaluated on auditability alongside accuracy, and presents the Knowledge-Driven Analytics Framework (KDAF), which builds ontology-driven knowledge systems through six iterative stages and retrieves evidence via Context-Aware Relevance Propagation (CARP), so that every retrieved fact carries its relationship type, confidence, and source lineage.   An evaluation on FinanceBench (145 questions) compares KDAF against zero-context inference, BM25, concept-weighted lexical retrieval, and ungrounded graph traversal. First, retrieval is necessary: zero-context inference reaches 4.1% correctn
    
[^42]: 依存关系的方向性上下文表示：为何跨方向配对会失败

    Directional Contextual Representations for Dependency Relations: Why Cross-Direction Pairing Fails

    [https://arxiv.org/abs/2608.20647](https://arxiv.org/abs/2608.20647)

    该论文发现，在依存关系类型分类中，将双向LSTM的前向与后向表示进行跨方向配对会持续性能下降，且差距随距离增大而增大，并通过冻结主干实验证明此现象源于表示的内在结构而非训练协同适应。

    

    arXiv:2608.20647v1 公告类型：新 摘要：将双向LSTM的上下文表示拆分为仅前向的$F_i$（严格为标记$1..i$的函数）和仅后向的$B_i$（严格为标记$i..n$的函数），在依存关系类型分类中，其性能优于单独使用任一方向或融合的自注意力表示。然而，该思想的一个特定自然扩展——将标记的前向状态与候选标记的后向状态配对（“跨方向”配对，$F_i$ 对 $B_j$）——持续不如同方向配对，且惩罚随标记距离增大而增大而非缩小，两者均通过配对自助法显著。我们通过冻结主干的方法诊断其原因：架构信息泄漏在构造上不可能发生（单层BiLSTM，经代码检查验证）；93%的同向与跨向差距在冻结主干并仅训练新头部后依然存在，排除了训练协同适应作为原因。

    arXiv:2608.20647v1 Announce Type: new  Abstract: Splitting a bidirectional LSTM's contextual representation into a forward-only $F_i$ (strictly a function of tokens $1..i$) and a backward-only $B_i$ (strictly a function of tokens $i..n$) beats either alone and beats a fused self-attention representation for dependency relation-type classification. But a specific, natural extension of this idea -- pairing a token's forward state against a \emph{candidate}'s backward state (``cross-direction'' pairing, $F_i$ vs.\ $B_j$) -- consistently \emph{underperforms} same-direction pairing, and the penalty \emph{grows}, not shrinks, with token distance, both paired-bootstrap significant. We diagnose why using a frozen-trunk methodology: architectural information leakage between directions is impossible by construction (a single-layer BiLSTM, verified by code inspection); 93\% of the same-vs-cross gap survives freezing the trunk and training only fresh heads, ruling out training-co-adaptation as the
    
[^43]: MIL-BERT：具有性能与解释性保证的任意大规模文本分类

    MIL-BERT: Classification of Arbitrarily Large Text with Performance and Explanatory Guarantees

    [https://arxiv.org/abs/2608.20636](https://arxiv.org/abs/2608.20636)

    本文提出MIL-BERT算法，利用多实例学习选择关键文本摘录进行分类，可处理近百万令牌的大规模文本，在多个长文本数据集上达到最先进性能，并具备解释性保证。

    

    arXiv:2608.20636v1 公告类型：新 摘要：许多文本分类决策仅基于构成性摘录即可做出。受多实例学习领域的启发，我们提出了一种训练神经网络通过选择此类摘录来对文本进行分类的算法。我们表明，我们的方法也具有可扩展性，并在近100万令牌的样本上进行了实证学习。我们在7个数据集上评估了我们的方法，重点强调远超基础模型编码限制的长文本集合。我们在此算法上在3个数据集上取得了最先进的结果：新闻媒体政治偏见识别、长故事中的触发警告以及推特集合中作者的人口统计特征。此外，在弱标记文本集合（袋）上训练的模型能够泛化，以准确分类构成性的较小实例。除了为这些问题提供新的最先进性能外，这种方法也是少数几种能够提供解释性保证的神经方法之一。

    arXiv:2608.20636v1 Announce Type: new  Abstract: Many text classification decisions are viable based on constituent excerpts alone. Taking inspiration from the field of multiple instance learning, we present an algorithm for training a neural network to classify text by selecting such excerpts. We show that our approach is also scalable with demonstrated learning against samples with nearly 1M tokens. We evaluate our methods on 7 datasets with emphasis on long-textual collections that far exceed the encoding limit of our base model. We present state-of-the-art results with this algorithm on 3 datasets: identification of political bias in news outlets, trigger warnings in long stories, and demographic characteristics of authors in tweet collections. Furthermore, the model trained on weakly-labeled collections of text (bags) generalizes to accurately classify constituent, smaller instances. Besides a new state-of-the-art for these problems, this approach is one of the few neural methods 
    
[^44]: AgentMercury：您的智能体能够在规模上综合出可验证的商业场景环境

    AgentMercury: Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale

    [https://arxiv.org/abs/2608.20634](https://arxiv.org/abs/2608.20634)

    AgentMercury提出了一种从高层商业场景中规模综合可验证环境的新框架，通过先实例化持久化世界再涌现任务，构建了覆盖多行业多国家的数千个环境，为强化学习提供了可扩展的训练基础。

    

    arXiv:2608.20634v1 公告类型：交叉 摘要：智能体通过与环境的交互来学习行动，但用于训练的环境通常是手动构建的，或围绕预定义任务和基准合成的。这种以任务为中心的范式使得难以扩展出反映现实且不断演变的工作流环境，在这些环境中，多样化的任务可以从底层世界中自然涌现。我们引入了AgentMercury，一个可扩展的框架，用于从高层商业场景中综合出可执行的环境。AgentMercury并非为特定任务构建环境，而是首先实例化一个持久化的世界，包含实体、服务、工具、状态以及可执行的跨服务不变量，从中随后可以涌现出多样化的任务和交互轨迹。我们构建了覆盖14个行业和50个国家的4,783个可执行环境，并将其用作强化学习的训练基质。尽管这些环境是在没有针对特定目标的情况下生成的，但它们展示了显著的多样性。

    arXiv:2608.20634v1 Announce Type: cross  Abstract: Agents learn to act through interaction with environments, yet the environments used for training are often manually constructed or synthesized around predefined tasks and benchmarks. This task-centric paradigm makes it difficult to scale environments that reflect realistic and evolving workflows where diverse tasks can naturally emerge from the underlying world. We introduce AgentMercury, a scalable framework for synthesizing executable environments from high-level business scenarios. Rather than constructing an environment for a specific task, AgentMercury first instantiates a persistent world with entities, services, tools, state, and executable cross-service invariants, from which diverse tasks and interaction trajectories can subsequently emerge. We construct 4,783 executable environments spanning 14 industries and 50 countries, and use them as training substrates for reinforcement learning. Despite being generated without targeti
    
[^45]: 高效Transformer中的稀疏令牌路由

    Sparse Token Routing in Efficient Transformers

    [https://arxiv.org/abs/2608.20632](https://arxiv.org/abs/2608.20632)

    本文提出SEWN双流Transformer，通过上下文门控实现令牌路由，在不牺牲任务精度的前提下，显著区分令牌重要性（p<10^-10），而静态先验方法则无法通过反事实测试。

    

    arXiv:2608.20632v1 公告类型：新 摘要：高效Transformer研究常以“并非所有令牌都需要相同的计算量”为由，推动令牌剪枝和自适应计算。我们使用SEWN（一种双流Transformer，通过学习的门控将令牌路由至轻量或全容量处理）端到端测试了这一论断。在我们的实验中，与参数匹配的基线相比，路由引入的精度变化可忽略不计，而门控的令牌重要性信号关键取决于其学习方式。静态词典种子先验在BoolQ上的反事实忠实度测试中失败，而全上下文门控在两个评估任务上实现了高度显著的分离（p<10^-10），且不改变任务精度。

    arXiv:2608.20632v1 Announce Type: new  Abstract: Efficient-transformer research often motivates token pruning and adaptive computation with the claim that not all tokens require equal computational effort. We test this claim end to end using SEWN, a two-stream Transformer that routes tokens through either lightweight or full-capacity processing using a learned gate. Across our experiments, routing introduces negligible accuracy change relative to parameter-matched baselines, while the gate's token-importance signal depends critically on how it is learned. A static lexicon-seeded prior fails a counterfactual faithfulness test on BoolQ, whereas a fully contextual gate achieves highly significant separation ($p<10^{-10}$) on both evaluated tasks without changing task accuracy.
    
[^46]: 当失败传播时：代理式检索增强生成中的因果失败归因

    When Failures Propagate: Causal Failure Attribution in Agentic Retrieval-Augmented Generation

    [https://arxiv.org/abs/2608.20627](https://arxiv.org/abs/2608.20627)

    本文提出了AgenticRAG-FP基准，用于评估代理式RAG中因果失败归因的准确性，发现覆盖率诊断在早期跳数有效，但在后续跳数失效。

    

    代理式检索增强生成（RAG）在多个跳数中交错进行检索、推理和答案生成。第1跳的检索错误可能直到第3跳才表现为错误答案，而后续检索也可能修复轨迹。本文介绍了AgenticRAG-FP，一个用于代理式RAG中因果失败归因的干预性基准。该基准在指定跳数注入已认证的故障，重新执行下游轨迹，并根据已知干预评估诊断器。其核心问题是，在后缀变化后，事后追踪是否仍能识别注入的跳数。在完成的严格密集Claude Haiku 4.5扫描中，针对80个三跳MuSiQue问题，基于覆盖率的诊断在第1跳为0.91，在第2和第3跳为0.00（n=43,36,21条失败轨迹）。一项较小的内容破坏研究改变了主题完整证据中的答案承载或桥梁事实。在深度2处，有18条失败...

    arXiv:2608.20627v1 Announce Type: cross  Abstract: Agentic retrieval-augmented generation (RAG) interleaves retrieval, reasoning, and answer generation across multiple hops. A retrieval error at hop 1 can surface only as a wrong answer at hop 3, while later retrieval can also repair the trajectory. This paper introduces AgenticRAG-FP, an interventional benchmark for causal failure attribution in agentic RAG. The benchmark injects a certified fault at a specified hop, re-executes the downstream trajectory, and evaluates diagnosers against the known intervention. Its central question is whether a post-hoc trace still identifies the injected hop after the suffix changes. In the completed strict dense Claude Haiku 4.5 sweep on 80 three-hop MuSiQue questions, coverage-based diagnosis is 0.91 at hop 1 and 0.00 at hops 2 and 3 (n=43,36,21 failed trajectories). A smaller content-corruption study changes an answer-bearing or bridge fact in topically intact evidence. At depth 2, where 18 failed 
    
[^47]: JuryProbe：一种用于路由无参考事实性评审团到有依据验证的经验共识风险诊断方法

    JuryProbe: An Empirical Consensus-Risk Diagnostic for Routing Reference-Free Factuality Judge Panels to Grounded Verification

    [https://arxiv.org/abs/2608.20607](https://arxiv.org/abs/2608.20607)

    本文提出JuryProbe，一种通过仅假阴性相关性和假共识提升度来诊断无参考事实性评审团共识风险的方法，并在高风险时路由到有参考验证，以减少因共享盲点导致的错误接受。

    

    arXiv:2608.20607v1 公告类型：交叉 摘要：由廉价LLM评审员组成的小组越来越多地做出接受或升级的决策。在事实性设置中，因为多个无参考评审员一致同意而接受一个声明可能会产生隐藏风险：这种一致性可能反映的是共同的假阴性盲点，而非独立的证据。我们引入了JuryProbe，一种针对无参考事实性评审团的经验共识风险诊断方法，并配以基于校准的路由策略。JuryProbe通过使用仅假阴性（FN-only）评审员相关性和假共识提升度，从标记的校准探针中估计共识风险；当标记为高风险时，无参考多数接受会被路由到带有可信参考的相同评审员。在审计的FEVER腐败数据上，无参考评审团显示出相关的假阴性（FN-only相关性为0.402和0.368；提升度分别为3.13倍和18.13倍），而在可信参考最佳案例诊断下，两种情形的一致假共识均降至零。

    arXiv:2608.20607v1 Announce Type: cross  Abstract: Panels of inexpensive LLM judges increasingly make accept-or-escalate decisions. In factuality settings, accepting a claim because several reference-free judges agree can create a hidden risk: agreement may reflect shared false-negative blind spots rather than independent evidence. We introduce JuryProbe, an empirical consensus-risk diagnostic for reference-free factuality judge panels, paired with a calibration-based routing policy. JuryProbe estimates consensus risk from a labeled calibration probe using false-negative-only (FN-only) judge correlation and false-consensus lift; when flagged high-risk, reference-free majority accepts are routed to the same judges with trusted references. On audited FEVER corruptions, reference-free panels show correlated false negatives (FN-only correlations 0.402 and 0.368; lifts 3.13x and 18.13x), while unanimous false consensus drops to zero under a trusted-reference best-case diagnostic on both min
    
[^48]: 开放权重掩蔽内省：衡量语言模型能报告自身计算的什么

    Open-Weight Masked Introspection: Measuring What Language Models Can Report About Their Own Computation

    [https://arxiv.org/abs/2608.20569](https://arxiv.org/abs/2608.20569)

    该研究构建了OWMI框架，在八个开放权重模型上进行78,000多次测量，发现这些模型无法内省自身计算状态，其报告与随机猜测无异。

    

    arXiv:2608.20569v1 公告类型：新 摘要：前沿模型能否内省其内部状态？最近的研究表明，在特定条件下，足够复杂的模型能够审计其内部，指出变化之处，并自信地报告。我们在来自七个家族的八个开放权重模型上测试了这一说法，发现没有这种能力：当被问及其自身计算是否被改变时，没有一个模型的回答优于随机水平。为了测试这一点，我们构建了开放权重掩蔽内省（OWMI）框架，该框架干预残差流站点、注意力头和稀疏自编码器特征，然后在答案必须击败的零假设条件下询问模型关于变化的情况：未做任何改变的模拟运行、影响匹配的随机扰动，以及仅看到可见输出的纯文本观察者。在超过78,000次测量中，没有任何模型的报告能区分真实干预与模拟干预，优于随机水平（AUROC约0.5007），且等价性检验...

    arXiv:2608.20569v1 Announce Type: new  Abstract: Are frontier models able to introspect about their internal states? Recent work suggests that under certain conditions a complex enough model can audit its own internals, call out what changed, and report back confidently about it. We tested that claim on eight open-weight models from seven families and found no such ability: asked whether their own computation had been altered, none answered better than chance. To test it we built Open-Weight Masked Introspection (OWMI), a framework that intervenes on residual-stream sites, attention heads and sparse-autoencoder features, then interrogates the model about the change against the null conditions an answer has to beat: sham runs where nothing was altered, impact-matched random perturbations, and a text-only observer that sees only the visible output.   Over 78,000 measurements, no model's report discriminates a real intervention from a sham beyond chance (AUROC ~0.5007), and an equivalence
    
[^49]: LiLiCorr：用于投机解码的并行草稿轻量级似然相关性方法

    LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding

    [https://arxiv.org/abs/2608.20530](https://arxiv.org/abs/2608.20530)

    LiLiCorr通过轻量级似然相关性模型关联起草器的逐位置边际分布，在不构造完整联合分布的情况下捕获块级联合结构，从而提升投机解码的连贯性。

    

    arXiv:2608.20530v1 公告类型：新 摘要：投机解码通过起草未来标记并由目标模型并行验证来加速语言模型推理。诸如DFlash之类的扩散式块头是一种有吸引力的起草器，它能在一次前向传播中预测整个未来标记块。然而，它基于逐位置边际分布而非联合块分布进行训练，因此其生成的标记在单个上看似合理，但整体上不连贯。我们引入了LiLiCorr，一种轻量级的基于似然的模型，用于关联起草器已生成的逐位置边际分布。它在每个位置保留前k个标记作为候选，并联合处理它们，为每个标记生成一个输入向量和一个输出向量。当较早候选的输出向量与较晚候选的输入向量具有高余弦相似度时，一对相邻候选即匹配。这些匹配捕获了块的联合结构，而无需显式构造完整的联合分布。一个轻量级的模型就能实现这一点。

    arXiv:2608.20530v1 Announce Type: new  Abstract: Speculative decoding accelerates language-model inference by drafting future tokens that the target model verifies in parallel. A diffusion-style block head such as DFlash is an attractive drafter, predicting an entire block of future tokens in one forward pass. However, it is trained on per-position marginals rather than the joint block distribution, so the tokens it emits are individually plausible yet jointly incoherent. We introduce LiLiCorr, a Lightweight Likelihood-based model that Correlates the per-position marginal distributions a drafter already produces. It keeps the top-k tokens at each position as candidates and processes them jointly, producing for each an in and an out vector. A pair of adjacent candidates matches when the earlier one's out vector has high cosine similarity with the later one's in vector. These matches capture the block's joint structure without ever materializing the full joint distribution. One lightweig
    
[^50]: ProofJudge：基于工具的形式化证明质量评估在Mathlib中的LLM评判系统

    ProofJudge: Tool-Grounded LLM Evaluation of Formal Proof Quality in Mathlib

    [https://arxiv.org/abs/2608.20432](https://arxiv.org/abs/2608.20432)

    ProofJudge是一个利用工具访问库状态的LLM评判系统，能有效评估形式化证明质量，并在多个维度上对齐人类偏好。

    

    摘要：在Lean 4中，通过内核类型检查的形式化证明，其质量仍可能存在很大差异。我们引入了ProofJudge，一个代理式LLM-as-judge系统，它从五个维度（超越正确性）来评分形式化证明的质量：库的利用、自动化适配、结构清晰度、陈述质量以及Mathlib约定。我们在一个包含218个声明的新数据集上评估ProofJudge，这些声明来自不同的Mathlib PR。评判代理通过工具访问PR所应用的提交来获得依据，使其在评分时能够查询库状态。当评判者将Mathlib接受的PR版本评为高于被退回修改的初始版本时，该评判者被视为与人类偏好一致。所有六个评估的评判模型都能以高于随机水平的概率恢复审阅者的偏好，范围从80.8%到63.5%，而两个开放权重评判者以最佳评判者十分之一的成本达到约70%。我们发布了评判框架和评估数据集。

    arXiv:2608.20432v1 Announce Type: cross  Abstract: Formal proofs in Lean 4 that pass the kernel's type checker can nonetheless vary widely in quality. We introduce ProofJudge, an agentic LLM-as-judge system that scores formal proof quality along five dimensions beyond correctness: library leverage, automation fit, structural clarity, statement quality, and Mathlib conventions. We evaluate ProofJudge on a novel dataset of 218 declarations drawn from distinct Mathlib PRs. The judge agent is grounded by tool access to the commit the PR is applied to, enabling it to query the library state when scoring. A judge is considered aligned with human preferences when it rates the version of the PR Mathlib accepted above the initial version that was sent back for revision. All six judge models evaluated recover the reviewers' preference well above chance, from 80.8% to 63.5%, and two open-weight judges reach roughly 70% at a tenth of the best judge's cost. We release the judge harness, evaluation 
    
[^51]: ARGUS：基于心智理论与策略感知规划及知识锚定的论证生成

    ARGUS: Theory-of-Mind Guided Argument Generation with Strategy-Aware Planning and Knowledge Grounding

    [https://arxiv.org/abs/2608.20405](https://arxiv.org/abs/2608.20405)

    Argus通过心智理论推理器构建受众信念模型，结合策略感知规划和知识锚定，实现更有效的说服性论证生成。

    

    摘要：有说服力的论证生成需要模拟受众信念、修辞策略和事实依据。尽管近期有所进展，现有方法大多忽略受众差异，未能整合策略选择以提升说服力。为弥合这一差距，我们提出了Argus，一个基于智能体的框架，将经典修辞学操作化用于说服性写作。其核心是一个心智理论（ToM）推理器，构建受众信念和价值观的显式双重心智模型，以指导后续决策。该表示条件化一个组件感知规划器，将论证分解为子主题，分配细粒度修辞功能（理性诉诸、情感诉诸、人格诉诸、时机诉诸），并在规划时触发策略引导的证据检索。最后，一个精炼模块迭代地定位并解决多维弱点，而不导致质量退化。我们在三个多样化基准上评估了Argus。

    arXiv:2608.20405v1 Announce Type: new  Abstract: Persuasive argument generation requires modeling audience beliefs, rhetorical strategies, and factual grounding. Despite recent advancements, existing methods remain largely audience-agnostic and fail to integrate strategy selection to improve persuasiveness. To bridge this gap, we propose Argus, an agent-based framework that operationalizes classical rhetoric for persuasive writing. At its core, a Theory-of-Mind (ToM) Reasoner constructs an explicit dual mental model of the audience's beliefs and values to guide downstream decisions. This representation conditions a component-aware planner that decomposes the argument into subtopics, assigns fine-grained rhetorical functions (logos, pathos, ethos, kairos), and triggers strategy-guided evidence retrieval at planning time. Finally, a refinement module iteratively targets and resolves multi-dimensional weaknesses without quality regression. We evaluate Argus across three diverse benchmarks
    
[^52]: 灵枢：一个大规模以症状为中心的情境化知识图谱，桥接中医与现代生物医学

    LingShu: A Large-Scale Symptom-Centric Contextualized Knowledge Graph Bridging Traditional Chinese Medicine and Modern Biomedicine

    [https://arxiv.org/abs/2608.20402](https://arxiv.org/abs/2608.20402)

    灵枢构建了一个大规模以症状为中心的情境化知识图谱，通过整合多源数据（如临床记录和中医文献），有效桥接了中医与现代生物医学，并利用情境化四元组克服了传统二元关系的局限性。

    

    arXiv:2608.20402v1 公告类型：交叉 摘要：生物医学知识图谱（KGs）对于知识组织至关重要，然而传统的二元关系往往难以表示生物医学知识的条件性。症状为连接中医和现代生物医学提供了共享的表型层，中医依赖症状模式进行辨证论治和治疗选择，而现代生物医学则将临床表现与疾病和分子机制联系起来。我们提出了灵枢，一个大规模以症状为中心的情境化知识图谱，旨在桥接中医与现代生物医学。本研究中分析的灵枢导出版本包含1733万个原子级实体记录和3947万条关系记录，包括1719万个语义三元组和2229万个情境化四元组。灵枢整合了多源数据，包括临床电子病历、权威中医文献和生物医学本体。

    arXiv:2608.20402v1 Announce Type: cross  Abstract: Biomedical knowledge graphs (KGs) are pivotal for knowledge organization, yet traditional binary relations often struggle to represent the conditional nature of biomedical knowledge. Symptoms provide a shared phenotypic layer for linking Traditional Chinese Medicine (TCM), which relies on symptom patterns for syndrome differentiation and treatment selection, with modern biomedicine, which connects clinical manifestations to diseases and molecular mechanisms. We present LingShu, a large-scale symptom-centric contextualized knowledge graph designed to bridge TCM and modern biomedicine. The exported version of LingShu analyzed in this study comprises 17.33 million atom-level entity records and 39.47 million relation records, including 17.19 million semantic triples and 22.29 million contextualized quadruples. LingShu integrates multi-source data, including clinical electronic medical records, authoritative TCM texts, biomedical ontologies
    
[^53]: 当检索在开始前就失败：结构性间接前提驱逐作为代理记忆中的保留失败

    When Retrieval Fails Before It Begins: Structurally Indirect Prerequisite Eviction as a Retention Failure in Agentic Memory

    [https://arxiv.org/abs/2608.20400](https://arxiv.org/abs/2608.20400)

    本文首次揭示了代理记忆中的“检索前失败”模式——结构性间接前提驱逐，并提出依赖感知语义垃圾收集规则，显著提升全链保留率。

    

    在固定预算下的代理记忆涉及两个阶段：保留和检索。现有的以检索为中心的范式隐含假设必要的证据能在驱逐中幸存，但我们通过隔离一种检索前失败模式来挑战这一假设：结构性间接前提驱逐，即与查询弱对齐的上游模块在预算压力下被丢弃。我们提供了这种失败的操作性定义、一个可复现的确定性基准以及逐种子追踪诊断。最后，我们评估了依赖感知语义垃圾收集（DSGC），一种一跳图感知规则。在我们的主要测试套件中，DSGC在词法编码器下将全链保留率从0.03提高到0.90，在句子编码器下从0.23提高到1.00。稳健性检查随后确定了单跳规则成立或退化的预算和扩展机制。我们发布的流程和失败事后分析支持对保留机制的机理分析，在检索之前。

    arXiv:2608.20400v1 Announce Type: new  Abstract: Agentic memory under a fixed budget involves two stages: retention and retrieval. Existing retrieval-centered paradigms implicitly assume necessary evidence survives eviction, but we challenge this by isolating a pre-retrieval failure mode: structurally indirect prerequisite eviction, in which upstream blocks weakly aligned with the query are discarded under budget pressure. We provide an operational definition of this failure, a reproducible deterministic benchmark, and per-seed trace diagnostics. Finally, we evaluate Dependency-aware Semantic Garbage Collection (DSGC), a one-hop graph-aware rule. In our main suite, DSGC improves full-chain retention from 0.03 to 0.90 under a lexical encoder and from 0.23 to 1.00 under a sentence encoder. Robustness checks then identify the budget and scaling regimes where the one-hop rule holds or degrades. Our released pipeline and failure postmortem support mechanistic analysis of retention before re
    
[^54]: 自监督语音表征追踪聋/听障婴幼儿及儿童口语向成人模型的趋同过程

    Self-Supervised Speech Representations Track Spoken Language Convergence to Adult Models in Infants and Children Who Are Deaf/Hard-of-Hearing

    [https://arxiv.org/abs/2608.20396](https://arxiv.org/abs/2608.20396)

    本研究首次利用自监督语音嵌入（HuBERT-BASE）从长时日常录音中直接量化聋/听障儿童与成人照护者之间的言语趋同过程，仅凭单一距离度量即可追踪语言发展轨迹，无需人工转录，为跨语言和人群的大规模语言发展评估提供了可扩展的新方法。

    

    arXiv:2608.20396v1 公告类型：新论文 摘要：语言发展的特点在于儿童言语逐渐向成人模式趋同。传统上，测量这一过程需要详细的转录和特定语言的专业知识，限制了在不同语言和人群中的可扩展性。在本研究中，我们利用语音嵌入直接从儿童日常生活中的长时、以儿童为中心的录音中捕捉这种趋同现象。使用HuBERT-BASE模型，我们从聋/听障儿童及其女性成人照护者的语音发声（超过925小时的观察）中提取嵌入特征。在控制音高和发声时长后，儿童与照护者之间的嵌入距离随听力年龄的增加而减小，这表明如预期所料，儿童在发育过程中言语模式逐渐向照护者趋同。这一单一距离度量还与多种标准化的言语和语言测量指标相关。

    arXiv:2608.20396v1 Announce Type: new  Abstract: Language development is characterized by a gradual convergence of children's speech toward adult patterns. Measuring this process has traditionally required detailed transcription and language-specific expertise, limiting scalability across languages and populations. Here, we use speech embeddings to capture this convergence directly from the acoustic signal in longform, child-centered recordings, taken as children go about their daily lives. Using HuBERT-BASE, we extracted embeddings from speech vocalizations of children who are deaf/hard-of-hearing and their female adult caregivers ($>$925 hrs. observation). Embedding distance between children and caregivers decreased with hearing age, controlling for pitch and vocalization length, indicating, as expected, that children's speech patterns converge to caregivers over development. This single distance metric likewise related to multiple standardized measures of speech and language from in
    
[^55]: 语音到SFT流水线的因子消融研究：对数据质量和下游迁移的差异影响

    A Factorial Ablation of a Speech-to-SFT Pipeline: Differential Effects on Data Quality and Downstream Transfer

    [https://arxiv.org/abs/2608.20394](https://arxiv.org/abs/2608.20394)

    通过2x2因子消融实验发现，语音到SFT流水线中的数据质量改进虽能提升评审质量，但并未显著提升下游MCQA性能，表明质量提升的迁移效果有限。

    

    arXiv:2608.20394v1 公告类型：交叉 摘要：通过多阶段精炼将语音转化为监督微调（SFT）数据的行业流水线日益普及，但据我们所知，尚未有公开的逐阶段消融研究，导致每个阶段的边际价值未知。我们设计了一个生产就绪的语音到SFT流水线，其中转录精炼（阶段0）和SFT数据质量精炼（阶段2）可独立切换，形成2x2因子设计。对于每种条件，我们从韩语医学和金融会议录音中生成问答形式的SFT数据，并微调9个模型（5个LLM家族，2.4B-70B）；我们使用四个跨提供商的LLM评审员、一项盲法六专家人工评估和3个下游MCQA基准进行评估。我们的核心发现是：在固定、标准的SFT配方下，QA数据质量的改进并未均匀转化为下游MCQA的提升。四位评审员的质量评分持续上升，但跨模型平均MCQA增益不显著；p值...

    arXiv:2608.20394v1 Announce Type: cross  Abstract: Industry pipelines that turn speech into supervised fine-tuning (SFT) data via multi-stage refinement are increasingly adopted but, to our knowledge, have not been publicly ablated stage-by-stage, leaving each stage's marginal value unknown. We design a production-ready speech-to-SFT pipeline in which transcript refinement (Phase 0) and SFT data quality refinement (Phase 2) are independently toggleable, yielding a 2x2 factorial design. For each condition, we generate QA-form SFT data from Korean medical and finance conference recordings and fine-tune 9 models (5 LLM families, 2.4B-70B); we evaluate with four cross-provider LLM judges, a blind six-expert human evaluation, and 3 downstream MCQA benchmarks. Our central finding: under a fixed, standard SFT recipe, improvements in QA data quality do not transfer uniformly into downstream MCQA gains. 4-judge quality rises consistently, yet the cross-model mean MCQA gain is not significant; p
    
[^56]: 知识图谱门控去事实化：用于智能体对话式AI中风格可控且事实保留的生成

    Knowledge-Graph-Gated Defactualization for Style-Controllable and Fact-Preserving Generation in Agentic Conversational AI

    [https://arxiv.org/abs/2608.20393](https://arxiv.org/abs/2608.20393)

    本文提出DSR框架，通过知识图谱与激活引导结合，在风格可控生成中显式区分并保留事实内容，解决了激活引导中的语义泄漏问题。

    

    arXiv:2608.20393v1 公告类型：交叉 摘要：在事实敏感型应用（如客户支持）中部署的智能体大语言模型（LLM）必须同时保持事实准确性，并以可控的风格化语域生成响应。激活引导通过扰动隐藏表示实现无需微调的风格控制，但缺乏区分可验证事实与风格化内容的明确机制，导致语义泄漏。我们通过*去事实化-引导-再水合*（DSR）框架解决这一挑战，该框架是一个知识工程框架，将类型化、显著性加权的知识图谱（KG）与激活引导相结合。DSR使用分层正则表达式、命名实体识别或词汇分类器管道提取显著实体，在引导前将其替换为类型化占位符，并在生成后通过显著性引导的再水合过程确定性恢复已验证值。DSR在六个LLaMA系列模型（1B-13B参数）上进行了评估。

    arXiv:2608.20393v1 Announce Type: cross  Abstract: Agentic large language models (LLMs) deployed in fact-sensitive applications such as customer support must simultaneously preserve factual correctness and generate responses in a controllable stylistic register. Activation steering enables fine-tuning-free style control by perturbing hidden representations, but it lacks an explicit mechanism for distinguishing verifiable facts from stylistic content, leading to semantic leakage. We address this challenge through \emph{Defactualize-Steer-Rehydrate} (DSR), a knowledge-engineering framework that integrates a typed, salience-weighted knowledge graph (KG) with activation steering. DSR extracts salient entities using a layered regex or NER or lexical-classifier pipeline, replaces them with typed placeholders prior to steering, and deterministically restores verified values through salience-guided rehydration after generation. DSR is evaluated across six LLaMA-family models (1B--13B parameter
    
[^57]: 评估即搜索：会议助手中接地失败的自适应发现

    Evaluation-as-Search: Adaptive Discovery of Grounding Failures in Meeting Assistants

    [https://arxiv.org/abs/2608.20392](https://arxiv.org/abs/2608.20392)

    本文提出“评估即搜索”（EaS）方法，通过自适应搜索会议问题空间，构建MeetingProbe基准，显著提高了发现LLM会议助手接地失败的效果。

    

    arXiv:2608.20392v1 公告类型：交叉 摘要：由大型语言模型驱动的会议助手已大规模部署，然而对其接地保真度的系统评估仍局限于静态基准，这些基准无法捕捉与特定话语结构或推理需求相关的失败模式。我们提出“评估即搜索”（EaS），这是一种反馈驱动的方法，将质量评估构建为对会议参与者可能提出的自然问题空间的自适应搜索。EaS并非均匀采样，而是通过迭代中的评估者反馈学习，将探测工作集中在失败最可能发生的认知需求上，由基于UCB评分的覆盖图和盲法多维度质量评估引导。利用EaS，我们构建了MeetingProbe基准，包含超过3000个标注的问答对，覆盖来自三种会议类型和三个LLM助手的20个转录文本。在消融实验中，自适应搜索发现的失败数量是随机探测的2.5倍（7.1%对比2%）。

    arXiv:2608.20392v1 Announce Type: cross  Abstract: LLM-powered meeting assistants are deployed at scale, yet systematic evaluation of their grounding fidelity remains limited to static benchmarks that miss failure modes tied to specific discourse structures or reasoning demands. We propose Evaluation-as-Search (EaS), a feedback-driven methodology that frames quality evaluation as an adaptive search over the space of natural questions a meeting participant might ask. Rather than sampling uniformly, EaS learns from evaluator feedback across iterations to concentrate probing effort on cognitive demands where failures are most likely, guided by a UCB-scored coverage map and blind multi-dimensional quality evaluation. Using EaS, we construct MeetingProbe, a benchmark of over $3{,}000$ annotated question--answer pairs spanning 20 transcripts from three meeting genres and three LLM assistants. In ablations, adaptive search surfaces $2.5\times$ more failures than random probing ($7.1\%$ vs. $2
    
[^58]: 移民理由：面向法律推理研究的美国移民上诉结构化数据集

    ImmigrationReason: A Structured Dataset of U.S. Immigration Appeals for Legal Reasoning Research

    [https://arxiv.org/abs/2608.20391](https://arxiv.org/abs/2608.20391)

    该论文构建了首个大规模行政上诉法律推理数据集，覆盖12,375项美国移民上诉决定，包含细粒度证据评估与逐字批评，填补了行政裁决领域NLP资源的空白。

    

    摘要：arXiv:2608.20391v1 公告类型：新 摘要：大多数法律自然语言处理资源都源自联邦判例法，并侧重于粗粒度分类，这使得行政裁决——政府决策绝大部分发生的领域——基本上未被涉及。我们引入了ImmigrationReason，这是一个大规模结构化数据集，来源于美国公民及移民服务局（USCIS）行政上诉办公室（AAO）在2005年至2026年间发布的12,375项非先例决定。每条记录都捕获了适用的法律框架、在五类标签下的逐标准证据充分性判定、逐字记录的裁决者批评引文、所有引用以及最终处置结果，同时附带高质量、由Claude转录的源文本。提取质量通过一个三遍流程进行验证，该流程结合了两种独立模式，并使用Opus 4.7进行对比提示裁决，此外，领域专家还对500条记录样本进行了核实。该数据集记录了近9,000条AAO的逐字实例。

    arXiv:2608.20391v1 Announce Type: new  Abstract: Most legal NLP resources draw from federal case law and focus on coarse classification, leaving administrative adjudication, where the vast majority of government decisions occur, essentially unaddressed. We introduce ImmigrationReason, a large-scale structured dataset derived from 12,375 non-precedent decisions of the U.S. Citizenship and Immigration Services (USCIS) Administrative Appeals Office (AAO) spanning 2005 to 2026. Each record captures the applicable legal framework, per-criterion evidence-sufficiency findings under a five-category label, verbatim adjudicator-criticism quotes, all citations, and final dispositions, alongside high-quality Claude-transcribed source text. Extraction quality is validated through a three-pass pipeline combining two independent modalities with comparison-prompt adjudication by Opus 4.7, and verified by domain experts on a 500-record sample. The dataset documents nearly 9,000 verbatim instances of AA
    
[^59]: 安萨里：一个基于检索的伊斯兰AI助手——架构、部署及14万次对话的经验教训

    Ansari: A Retrieval-Grounded Islamic AI Assistant -- Architecture, Deployment, and Lessons from 140,000 Conversations

    [https://arxiv.org/abs/2608.20390](https://arxiv.org/abs/2608.20390)

    安萨里通过代理式检索循环，仅基于认证的伊斯兰语料库回答并附引用，有效避免了事实捏造和价值观错位，已成功处理14万次多语言对话。

    

    通用大语言模型（LLM）越来越多地被用于回答宗教问题，但对于伊斯兰内容，它们存在两个严重风险：事实捏造（编造《古兰经》经文或圣训）和微妙的价值观错位。我们介绍了安萨里，一个已部署的、基于检索的伊斯兰AI助手，自2023年6月以来已处理超过14万次对话，覆盖25多种语言。安萨里围绕一个代理式检索循环构建：一个使用工具的语言模型对经过认证的伊斯兰语料库——包括《古兰经》、圣训集、多卷法学（教法）百科全书和经注（塔夫西尔）来源——进行搜索，并且仅基于检索到的内容进行回答，附上引用以供验证。我们描述了该系统的架构（代理循环、检索工具、语料库以及编码编辑和神学政策的系统提示）、其多平台部署（网页、移动端、W），以及从这些互动中获得的经验教训。

    arXiv:2608.20390v1 Announce Type: cross  Abstract: General-purpose large language models (LLMs) are increasingly used to answer religious questions, but for Islamic content they carry two serious risks: factual fabrication (inventing Qur'anic verses or hadith) and subtle value misalignment. We present Ansari, a deployed, retrieval-grounded Islamic AI assistant that has handled more than 140,000 conversations across 25+ languages since June 2023. Ansari is built around an agentic retrieval loop: a tool-using language model issues searches against authenticated Islamic corpora -- the Qur'an, hadith collections, a multi-volume jurisprudence (fiqh) encyclopedia, and exegetical (tafsir) sources -- and answers only on the basis of what it retrieves, with citations attached for verification. We describe the system's architecture (the agent loop, the retrieval tools, the corpora, and the system prompt that encodes editorial and theological policy), its multi-platform deployment (web, mobile, W
    
[^60]: 意图引擎：面向计算连续体中意图驱动编排的自然语言意图翻译

    Intent Engine: Natural-Language Intent Translation for Intent-Driven Orchestration in the Compute Continuum

    [https://arxiv.org/abs/2608.20388](https://arxiv.org/abs/2608.20388)

    意图引擎提出了一种自然语言意图翻译架构，通过构建经过验证的SLO工件来增强意图驱动编排的可靠性，避免因LLM直接生成错误而导致的放置问题。

    

    arXiv:2608.20388v1 公告类型：新论文 摘要：在计算连续体中的微服务放置由低级服务级别目标（SLO）驱动，但要求用户指定指标级约束会造成采用障碍并增加配置错误风险。尽管大型语言模型（LLM）可以解释自然语言意图，但直接生成可被编排系统使用的SLO工件仍不可靠，原因包括不支持的约束、错误的接地值以及模式违规。这些错误可能传播到下游放置逻辑，导致不可行或不正确的放置。本文提出了意图引擎，一种自然语言意图翻译架构，用于构建经过验证的SLO工件，以支持计算连续体中的服务放置。意图引擎作为现有意图驱动编排和放置框架的意图获取与SLO构建层；它不执行放置或运行时服务质量优化。该架构结合了……

    arXiv:2608.20388v1 Announce Type: new  Abstract: Microservice placement in the compute continuum is driven by low-level Service-level Objectives (SLOs), but requiring users to specify metric-level constraints creates an adoption barrier and increases misconfiguration risk. Although large language models (LLMs) can interpret natural-language intents, direct generation of orchestration-consumable SLO artifacts remains unreliable due to unsupported constraints, incorrect grounded values, and schema violations. These errors can propagate to downstream placement logic and produce infeasible or incorrect placements. This paper presents Intent Engine, a natural-language intent translation architecture that constructs validated SLO artifacts for compute-continuum service placement. Intent Engine acts as an intent acquisition and SLO construction layer for existing intent-driven orchestration and placement frameworks; it does not perform placement or runtime QoS optimization. The architecture c
    
[^61]: Poly-InstructTTS：从开放式指令学习野外表现力语音合成

    Poly-InstructTTS: Learning In-the-Wild Expressive Speech Synthesis from Open-Ended Instructions

    [https://arxiv.org/abs/2608.20387](https://arxiv.org/abs/2608.20387)

    本文提出Poly-InstructTTS，通过构建1000小时多模态指令标注语料库和基于属性思考标记的GPT框架，实现了从开放式自然语言指令生成高表现力语音，并支持说话人微调以保持人物角色。

    

    摘要：虽然最近的文本到语音（TTS）模型实现了高自然度，但通过自然语言指令控制细粒度表达仍然具有挑战性。我们介绍了Poly-InstructTTS，它利用野外音视频数据从开放式指令中学习表现力语音。我们构建了一个可扩展的多模态流水线，创建了一个包含1000多个细粒度情感和风格的1000小时指令标注语料库。该框架采用基于属性的思考标记的无提示GPT，随后是一个流匹配模块，从参考音频注入音色。我们还提出了一种说话人微调程序，将指令控制转移到特定说话人，同时保持人物角色。我们进一步扩展了InstructTTSEval，涵盖更广泛的任务。实验表明，Poly-InstructTTS在指令遵循和表现力方面表现出强大性能。音频演示和扩展测试集可在我们的项目页面上获取。

    arXiv:2608.20387v1 Announce Type: cross  Abstract: While recent text-to-speech (TTS) models achieve high naturalness, controlling fine-grained expression via natural-language instructions remains challenging. We introduce Poly- InstructTTS, which learns expressive speech from open-ended instructions using in-the-wild audiovisual data. We build a scalable multi-modal pipeline to construct a 1,000-hour instruction-annotated corpus covering 1,000+ fine-grained emotions and styles. The framework uses a prompt-free GPT with attribute-based thinking tokens, followed by a flow-matching module that injects timbre from a reference audio. We also present a speaker fine-tuning procedure to transfer instruction control to specific speakers while preserving persona. We further extend InstructTTSEval with broader tasks. Experiments show that Poly-InstructTTS delivers strong performance in instruction adherence and expressiveness. Audio demos and the expanded testset are available on our project page
    
[^62]: 利用人类与LLM的分歧改进基于清单的质量评估

    Using Human-LLM Disagreement to Improve Checklist-Based Quality Appraisal

    [https://arxiv.org/abs/2608.20385](https://arxiv.org/abs/2608.20385)

    本研究通过分析人类与LLM在基于清单的质量评估中的分歧模式，提出了一种改进模糊清单项目的方法，并验证了LLM在特定条件下能近似专家判断。

    

    系统综述依赖于对纳入研究的质量评估，这一过程耗时且对清单标准的模糊性敏感。尽管大型语言模型（LLMs）为支持这些任务提供了机会，但评估清单通常被视为固定输入，其设计如何影响与专家判断的一致性仍不清楚。因此，我们研究了（1）LLMs能否在基于清单的评估中近似人类判断，以及（2）人类与LLM的分歧模式能否用于识别和改进模糊的清单项目。使用潜在轨迹研究报告指南（GRoLTS）清单，我们比较了LLM生成的评估与专家注释在三个研究主题和两个清单版本中的表现。一致性通过项目级准确性、机会校正一致性和研究级排名顺序的保留来评估。我们发现性能在不同情境下有所变化。

    arXiv:2608.20385v1 Announce Type: new  Abstract: Systematic reviews rely on quality appraisal of included studies, a process that is time-consuming and sensitive to ambiguity in checklist criteria. Although large language models (LLMs) offer opportunities to support these tasks, appraisal checklists are typically treated as fixed inputs, and it remains unclear how their design affects agreement with expert judgments. Therefore, we investigate (1) whether LLMs can approximate human judgments in checklist-based appraisal and (2) whether patterns of human-LLM disagreement can be used to identify and improve ambiguous checklist items. Using the Guidelines for Reporting on Latent Trajectory Studies (GRoLTS) checklist, we compare LLM-generated assessments with expert annotations across three research topics and two checklist versions. Agreement is assessed using item-level accuracy, chance-corrected agreement, and preservation of study-level rank ordering.   We find that performance varies s
    
[^63]: 解耦视觉-语言系统用于多模态理解与生成

    Decoupled Vision-Language System for Multimodal Understanding and Generation

    [https://arxiv.org/abs/2608.20382](https://arxiv.org/abs/2608.20382)

    Libra通过解耦视觉和语言系统的自模态建模与跨模态交互，实现了高效的多模态理解与生成，并在图像到文本和文本到图像任务中展现了有效性。

    

    arXiv:2608.20382v1 公告类型：新 摘要：我们提出了一种新的多模态大型语言模型（MLLMs）架构设计，名为Libra，该模型能够同时进行多模态理解和生成。Libra架构包含一个视觉系统和一个语言系统，通过跨模态桥梁连接。这种设计将自模态建模和跨模态交互解耦，使每种模态能够学习其独特的表示，同时保持有效的跨模态理解。解耦主要通过一个开关注意力模块和一个开关前馈神经网络模块实现，这些模块动态路由计算流，用于自模态建模和跨模态交互场景。我们在两个重要设置中评估了其有效性：\textbf{Libra-1}用于仅理解的图像到文本设置，以及\textbf{Libra-2}用于统一的图像到文本理解和文本到图像生成。除了架构设计外，我们还讨论了分词、位置编码等方面的各种改进。

    arXiv:2608.20382v1 Announce Type: new  Abstract: We introduce a new architecture design for multimodal large language models (MLLMs), Libra, capable of both multimodal understanding and generation. Libra architecture contains one vision system and one language system, connected by cross-modal bridges. This design decouples self-modal modeling and cross-modal interaction, enabling each modality to learn its unique representations while maintaining effective cross-modal comprehension. The decoupling is mainly achieved in a switch attention module and a switch FFN module, which dynamically routes the computation flow for self-modal modeling and cross-modal interaction scenarios. We evaluate the effectiveness in two important settings: \textbf{Libra-1} for the understanding-only image-to-text setting, and \textbf{Libra-2} for unified image-to-text understanding and text-to-image generation. In addition to the architecture design, we discuss various improvements on tokenization, positional 
    
[^64]: EditPPT：通过结构化工具使用与双模态验证器的多智能体实现忠实长幻灯片编辑

    EditPPT: Faithful Long-Deck Slide Editing via Structured Tool-Using Multi-Agent with Dual-Modal Validators

    [https://arxiv.org/abs/2608.20381](https://arxiv.org/abs/2608.20381)

    EditPPT通过多智能体框架和双模态验证器，将幻灯片编辑转化为受约束的工具选择问题，实现了对长幻灯片的高保真、高准确编辑，并提供了新的基准DeckEdit-Bench。

    

    自动化幻灯片编辑需要同时满足修改准确性、保真度以及对幻灯片长度的鲁棒性。现有的基于大语言模型（LLM）的系统在真实世界演示文稿上往往失败，因为它们依赖于理想化的中间表示或开放式代码生成，这在长幻灯片中容易导致级联错误。我们提出了EditPPT，一个多智能体框架，将幻灯片编辑重新定义为受约束的工具选择问题。通过原生PowerPoint COM接口执行局部形状级操作，EditPPT缩小了LLM的动作空间，同时保留了用户创作幻灯片的应用程序解析结构。通过跨模态分离验证，我们的双模态验证提供了对指令保真度和视觉质量的更稳健评估。我们还提出了DeckEdit-Bench基准，包含28个人类创作的幻灯片组、582张幻灯片和183个编辑提示，涵盖短、中、长幻灯片场景。

    arXiv:2608.20381v1 Announce Type: cross  Abstract: Automating slide editing requires simultaneously satisfying modification accuracy, preservation fidelity, and robustness to deck length. Existing LLM-based systems often fail on real-world presentation files because they rely on idealized intermediate representations or open-ended code generation, which are prone to cascading errors in long decks. We introduce EditPPT, a multi-agent framework that reformulates slide editing as a constrained tool-selection problem. By executing localized shape-level operations through the native PowerPoint COM interface, EditPPT narrows the LLM action space while preserving the application-resolved structure of user-authored decks. By separating validation across modalities, our dual-modal validation provides more robust assessment of both instruction fidelity and visual quality. We also present DeckEdit-Bench, a benchmark with 28 human-authored decks, 582 slides, and 183 editing prompts across short, m
    
[^65]: TH-GNN：用于LLM智能体托攻击检测的异构时序图神经网络

    TH-GNN: Heterogeneous Temporal Graph Neural Networks for LLM-Agent Shilling Attack Detection

    [https://arxiv.org/abs/2608.20376](https://arxiv.org/abs/2608.20376)

    TH-GNN通过融合异构时序图结构与跨模态语义注意力，有效检测LLM生成的推荐系统托攻击，解决了现有方法忽视图结构和时序协调的缺陷。

    

    arXiv:2608.20376v1 公告类型：新公告 摘要：LLM智能体现在能够大规模生成逼真的托配置文件、流畅的评论和连贯的评分，从而系统性地攻破推荐系统防御。仅依赖文本的检测器通过标记评论嵌入中的语义漂移，但对图结构和时序协调视而不见；而仅依赖图的检测器利用邻域异常，却无法推理评论语义或LLM生成内容所产生的跨模态不一致性。我们提出TH-GNN，一种异构时序图神经网络，采用双层异构图Transformer骨干，在每条边上应用基于类型和关系的注意力机制，并增强可学习的正弦时序编码。跨模态注意力将结构化的用户嵌入与冻结的RoBERTa表示的评论和物品描述融合，而基于对数到达时间间隔的GRU捕获时序突发性。在五种攻击家族和四个基准数据集上的评估表明...

    arXiv:2608.20376v1 Announce Type: new  Abstract: LLM agents can now generate realistic shilling profiles, fluent reviews, and coherent ratings at scale, systematically defeating recommender-system defenses. Text-only detectors that flag semantic drift in review embeddings are blind to graph structure and temporal coordination, while graph-only detectors that exploit neighborhood anomalies cannot reason over review semantics or the cross-modal inconsistencies produced by LLM-generated content. We propose TH-GNN, a heterogeneous temporal graph neural network with a two-layer Heterogeneous Graph Transformer backbone that applies per-type and per-relation attention augmented with learnable sinusoidal temporal encodings on every edge. Cross-modal attention fuses structural user embeddings with frozen RoBERTa representations of reviews and item descriptions, while a GRU operating over log inter-arrival times captures temporal burstiness. Evaluated across five attack families and four benchma
    
[^66]: GRAFT：基于目标蒸馏边评分和自适应DLM草案树构建

    GRAFT: Adaptive DLM-Based Draft Tree Construction with Target-Distilled Edge Scoring

    [https://arxiv.org/abs/2608.20375](https://arxiv.org/abs/2608.20375)

    本文提出GRAFT，通过目标蒸馏边评分和自适应树大小调整，改进扩散语言模型下的基于树的投机解码，提升令牌接受率与吞吐量。

    

    基于树的投机解码通过验证多个草案路径提高了标准投机解码的平均接受令牌数，现有树构建器通常通过父条件扩展来构建这些路径，其中每个子令牌在其父路径条件下生成。这种构建方式与扩散语言模型（DLM）起草器（如DFlash）不兼容，因为DLM在单次前向传播中生成所有未来位置的分布。DDTree通过将每个未来位置分布中的高概率令牌视为候选节点，并在固定节点预算下选择连续位置之间的边来弥合这一差距。然而，其边选择仅依赖令牌概率，没有建模父子兼容性，导致目标兼容令牌可能被附加到错误的父节点；此外，其固定预算忽略了吞吐量最优的树大小随解码状态变化的事实。我们提出GRAFT方法，以解决这些限制。

    arXiv:2608.20375v1 Announce Type: new  Abstract: Tree-based speculative decoding raises the mean accepted tokens of standard speculative decoding by verifying multiple draft paths, and existing tree builders typically construct these paths through parent-conditioned expansion, where each child token is generated conditioned on its parent path. This construction is incompatible with diffusion language model (DLM) drafters such as DFlash, which produces all future-position distributions in a single forward pass. DDTree bridges this gap by treating high-probability tokens from each future-position distribution as candidate nodes and selecting edges between consecutive positions under a fixed node budget. However, its edge selection relies on token probability alone without modeling parent--child compatibility, so target-compatible tokens can be attached to wrong parents; moreover, its fixed budget ignores that the throughput-optimal tree size varies with the decoding state. We propose GRA
    
[^67]: VA-DPO：基于效价-唤醒度的直接偏好优化，用于语言模型中的可控情感生成

    VA-DPO: Valence-Arousal Direct Preference Optimization for Controllable Emotion Generation in Language Models

    [https://arxiv.org/abs/2608.20374](https://arxiv.org/abs/2608.20374)

    VA-DPO通过将情感目标表示为连续效价-唤醒度点，并基于距离阈值筛选偏好数据，实现了比现有提示方法更精确可控的情感生成，显著降低了目标距离并提升了相关性。

    

    arXiv:2608.20374v1 公告类型：交叉 摘要：我们能多精确地告诉语言模型如何感受？大多数情感生成的工作使用离散标签（如快乐、愤怒、悲伤）来回答，这无法表达像“略带沮丧但平静”这样的目标。我们转而将期望的情感指定为效价-唤醒度平面中的一个连续点（v*, a*），并训练模型去命中该点。我们的方法VA-DPO是对直接偏好优化（DPO）的一个小修改：一个冻结的VA回归器根据每个采样生成与目标的欧氏距离进行评分，我们只保留距离差距超过阈值τ的候选对，并使用普通的DPO损失对冻结参考模型优化一个LoRA适配器。DPO目标本身未变；新意在于偏好数据的构建方式。在Llama-3.1-8B-Instruct上，该方法将平均VA距离比系统提示减少33%，比少样本提示减少25%，并将效价/唤醒度相关性提升至r_v=0.93和r_a=0.75。这些改进……

    arXiv:2608.20374v1 Announce Type: cross  Abstract: How precisely can we tell a language model how to feel? Most work on emotional generation answers with a discrete label - happy, angry, sad - which cannot express a target like "mildly downcast but calm." We instead specify the desired affect as a continuous point (v*, a*) in the Valence-Arousal plane and train the model to hit it. Our method, VA-DPO, is a small modification to Direct Preference Optimization: a frozen VA regressor scores each sampled generation by its Euclidean distance to the target, we keep only candidate pairs whose distance gap clears a margin tau, and we optimize a LoRA adapter with the ordinary DPO loss against a frozen reference. The DPO objective itself is unchanged; what is new is how the preference data is built. On Llama-3.1-8B-Instruct this cuts mean VA distance to the target by 33% over system-prompting and 25% over few-shot prompting, lifting valence/arousal correlation to r_v=0.93 and r_a=0.75. The gains
    
[^68]: 用于评估大语言模型在临床注册表数据提取中性能的歧义分类法：一项多中心前瞻性研究

    An ambiguity taxonomy for evaluating large language model performance on clinical registry abstraction: a multi-site prospective study

    [https://arxiv.org/abs/2608.20373](https://arxiv.org/abs/2608.20373)

    本研究提出了一种六类歧义分类法，用于系统评估大语言模型在临床注册表数据提取中的性能，并验证了其在不同医疗中心的有效性。

    

    摘要：arXiv:2608.20373v1 公告类型：新 目标：评估大语言模型（LLM）在处理未处理的电子病历（EMR）数据以进行临床注册表数据提取时的性能。方法：我们评估了LLM回答美国心脏病学会国家心血管数据注册表（ACC NCDR）注册问题的性能。在一项学术医疗中心的试点研究中，模型为每个注册问题识别候选数据源，经验丰富的提取人员利用这些结果定义了特定问题的文档集。在第二中心使用第二个ACC NCDR注册表的验证研究中，LLM使用特定问题的文档集回答问题。在审查任何输出之前，两名提取人员独立建立金标准，并将每个问题分配到六个类别之一，按解决所需的歧义和临床推理程度排序：药物/事件标志、二元临床存在性、行政性、定量性。

    arXiv:2608.20373v1 Announce Type: new  Abstract: Objective: To evaluate large language model (LLM) performance on unprocessed electronic medical record (EMR) data for clinical registry abstraction. Methods: We evaluated LLM performance answering registry questions for the American College of Cardiology National Cardiovascular Data Registry (ACC NCDR). In a pilot study at an academic medical center, the model identified candidate data sources for each registry question and experienced abstractors used these results to define question-specific document sets. In a validation study at a second center with a second ACC NCDR registry, the LLM answered questions using the question-specific document sets. Before reviewing any output, two abstractors independently established the ground truth and assigned each question to one of six categories, ordered by the ambiguity and clinical reasoning required to resolve it: Medication/Event Flag, Binary Clinical Presence, Administrative, Quantitative La
    
[^69]: 大型语言模型何时取代微调的自然语言理解模型？面向生产对话系统意图检测的决策框架

    When Do LLMs Replace Fine-Tuned NLU? A Decision Framework for Intent Detection in Production Conversational Systems

    [https://arxiv.org/abs/2608.20371](https://arxiv.org/abs/2608.20371)

    本文通过实验证明，大型语言模型在意图检测中仅在特定条件下（如无标签数据或需要鲁棒性）优于微调模型，而在有充足领域标签时，微调模型更高效且成本更低。

    

    arXiv:2608.20371v1 公告类型：交叉 摘要：一个常见的说法是，零样本大型语言模型（LLMs）可以取代微调的自然语言理解（NLU）分类器进行意图检测。我们对此说法进行了正面比较，并发现诚实的答案是：这取决于意图空间。在完整的ATIS和CLINC150数据集上，我们比较了微调的RoBERTa、TF-IDF+逻辑回归基线、句子嵌入kNN和Claude Haiku零样本模型，报告了bootstrap 95%置信区间和配对显著性检验。当存在大量领域内标签时，微调的RoBERTa表现相同或更好，且便宜和快速三个数量级：在ATIS上，它比Claude零样本高出11.8个百分点（95.9对84.1，p<0.001）。在广泛的150意图CLINC150模式中，两者在统计上持平（89.1对88.5，p=0.24）：LLM在没有训练数据的情况下匹配了完全监督模型。LLM的优势出现在三个生产相关场景中：超出范围检测（OOS召回率85.6对58.1，相对于RoBERTa）；鲁棒性

    arXiv:2608.20371v1 Announce Type: cross  Abstract: A common claim is that zero-shot large language models (LLMs) can replace fine-tuned NLU classifiers for intent detection. We test this claim head-to-head and find that the honest answer is: it depends on the intent space. On full ATIS and CLINC150 we compare a fine-tuned RoBERTa, a TF-IDF+logistic-regression baseline, sentence-embedding kNN, and Claude Haiku zero-shot, reporting bootstrap 95% confidence intervals and paired significance tests. When abundant in-domain labels exist, fine-tuned RoBERTa is as good or better and three orders of magnitude cheaper and faster: on ATIS it beats Claude zero-shot by 11.8 points (95.9 vs. 84.1, p<0.001). On the broad 150-intent CLINC150 schema the two are statistically tied (89.1 vs. 88.5, p=0.24): the LLM matches a fully supervised model with no training data. The LLM's advantages appear in three production-relevant regimes: out-of-scope detection (OOS recall 85.6 vs. 58.1 for RoBERTa); robustne
    
[^70]: ASTAR：从大规模临床自由文本语料库自动归纳标准化放射学报告模板

    ASTAR: Automated induction of STAndardized radiology Reporting templates from large-scale clinical free-text corpora

    [https://arxiv.org/abs/2608.20369](https://arxiv.org/abs/2608.20369)

    ASTAR利用大型语言模型自动从大规模临床自由文本中归纳标准化放射学报告模板，克服了手动构建的静态和扩展性限制，并在多中心胎儿脑MRI报告上超越了专家模板。

    

    摘要：结构化报告将自由文本的放射学叙述转换为可查询的数据键，有助于队列构建、纵向跟踪和医学AI的训练标签生成。主流范式遵循两阶段流程：（1）构建报告模板，（2）提取信息以填充模板。虽然提取阶段受益于大型语言模型（LLM）的进步，但模板构建仍然是依赖劳动密集型专家共识的手动瓶颈，这种共识是静态的、难以扩展，并且可能无法捕捉现实世界报告的多样性。我们通过\textbf{\texttt{ASTAR}}解决了这一局限性，这是一个基于LLM的框架，用于从大规模临床自由文本语料库自动归纳标准化放射学报告模板。对来自多个中心的4,215份胎儿脑MRI报告进行的大量实验表明，\textbf{\texttt{ASTAR}}归纳出的模板优于两位专家设计的模板。

    arXiv:2608.20369v1 Announce Type: cross  Abstract: Structured reporting converts free-text radiology narratives into queryable data keys, facilitating cohort assembly, longitudinal tracking, and training label generation for medical AI. The prevailing paradigm follows a two-stage pipeline: (1) constructing a reporting template, (2) extracting information to populate it. While the extraction stage has benefited from advances in large language models (LLMs), template construction remains a manual bottleneck relying on labor-intensive expert consensus that is static, difficult to scale, and may fail to capture real-world reporting diversity. We address this limitation with \textbf{\texttt{ASTAR}}, an LLM-based framework for Automated induction of STAndardized radiology Reporting templates from large-scale clinical free-text corpora. Extensive experiments on 4,215 fetal brain MRI reports from multiple centers demonstrate that the \textbf{\texttt{ASTAR}}-induced template surpasses two exper
    
[^71]: 基于文本特征分析的科研论文质量识别

    Research Paper Quality Recognition Through Textual Feature Analysis

    [https://arxiv.org/abs/2608.20368](https://arxiv.org/abs/2608.20368)

    本文提出了一种仅利用标题和摘要文本特征进行科研论文质量分类的基准方法，并通过多种嵌入技术与分类器组合，实现了高达91.12%的准确率，同时提供了透明度、可视化和可解释性分析。

    

    arXiv:2608.20368v1 公告类型：新 摘要：知识和创新受到科学研究质量和可信度的影响。然而，区分有影响力、高质量的工作与有缺陷的研究仍然是一个挑战。本文引入了一个基准，用于将研究论文分为两类：优秀（高被引）和非优秀（已撤稿），仅使用标题和摘要中的文本特征。我们评估了多种嵌入技术，包括SBERT、Word2Vec、FastText、USE和TF-IDF，并结合了支持向量机（SVM）、随机森林和神经网络等分类器。我们的贡献包括：（1）超参数透明度，（2）使用t-SNE进行特征空间可视化，（3）使用SHAP进行模型可解释性分析，以及（4）对错误案例的详细检查。实验结果表明，使用SBERT嵌入的神经网络达到了87.22%的准确率，而FastText结合SVM达到了91.12%。这些发现凸显了文本特征在论文质量识别中的潜力。

    arXiv:2608.20368v1 Announce Type: new  Abstract: Knowledge and innovations are shaped by using the quality and credibility of the scientific research. Yet, distinguishing between impactful, high-quality work and flawed studies remains a challenge. This paper introduces a benchmark for classifying research papers into two categories: good (highly cited) and non-good (retracted), using only textual features from titles and abstracts. We evaluate multiple embedding techniques, including SBERT, Word2Vec, FastText, USE, and TF-IDF, combined with classifiers such as Support Vector Machines (SVM), Random Forests, and Neural Networks. Our contributions include: (1) hyperparameter transparency, (2) feature space visualizations using t-SNE, (3) model interpretability analysis with SHAP, and (4) detailed examination of error cases. Experimental results show that a neural network with SBERT embeddings achieves 87.22\% accuracy, while FastText combined with SVM reaches 91.12\%. These findings highl
    
[^72]: 斯里兰卡议会辩论的三语主题建模

    Trilingual Topic Modeling of Sri Lankan Parliamentary Debates

    [https://arxiv.org/abs/2608.20365](https://arxiv.org/abs/2608.20365)

    本文提出了一种结合LLM提取和多语种嵌入聚类的新框架，成功对斯里兰卡三语议会辩论进行主题建模，并恢复了与重大国家事件对应的30个宏观主题。

    

    arXiv:2608.20365v1 公告类型：交叉 摘要：斯里兰卡议会辩论（汉萨德记录）构成了一个三语语料库，包含僧伽罗语、泰米尔语和英语的演讲，包括代码混合内容，但由于布局复杂的PDF、多语种文字和黏着语形态，这些内容仍无法被标准NLP流程处理。我们提出了一个端到端框架，通过基于LLM的文本提取，随后进行多语种嵌入和基于密度的聚类流程来解决这些挑战，以实现主题建模。进一步探索了一种混合语义-词汇扩展方法BiTopic，以提高可解释性并恢复否则被视为噪声而丢弃的演讲。应用于2017年至2026年间19,553篇演讲，该流程恢复了30个宏观主题，聚类纯度（BCP）达到0.673，其时间轨迹与重大国家事件（包括2019年复活节周日袭击和2022年经济危机）无监督地保持一致。传统LDA在此语料库上失败，原因在于跨语言问题。

    arXiv:2608.20365v1 Announce Type: cross  Abstract: Sri Lankan parliamentary debates (Hansards) constitute a trilingual corpus of speeches in Sinhala, Tamil, and English, including code-mixed content, yet remain inaccessible to standard NLP pipelines due to layout-complex PDFs, multilingual scripts, and agglutinative morphology. We present an end-to-end framework that addresses these challenges through LLM-based text extraction followed by a multilingual embedding and density-based clustering pipeline for topic modeling. A hybrid semantic-lexical extension, BiTopic, is further explored to improve interpretability and recover speeches otherwise discarded as noise. Applied to 19,553 speeches spanning 2017-2026, the pipeline recovers 30 macro-topics achieving a cluster purity (BCP) of 0.673, whose temporal trajectories align unsupervised with major national events including the 2019 Easter Sunday attacks and the 2022 economic crisis. Traditional LDA fails on this corpus due to cross-lingua
    
[^73]: 大语言模型时代的圣训计算科学：一项批判性叙事综述

    Hadith computational science in the age of large language models: a critical narrative review

    [https://arxiv.org/abs/2608.20364](https://arxiv.org/abs/2608.20364)

    本文通过批判性叙事综述，系统评估了圣训计算科学在大语言模型时代的方法论稳健性、基准局限和未解问题，指出数据与工具进步与学术应用受限并存的现状。

    

    我们考察了圣训计算科学如何被Transformer模型、检索增强流水线和大语言模型（LLMs）重塑。最近的综述记录了文献的增长，但尚未提供对哪些进展在方法论上稳健、哪些仍受限于基准测试、以及哪些未解决问题仍限制学术使用的批判性说明。我们通过一项批判性叙事综述来弥补这一空白，该综述结合了对现有综述的批判、对代表性原始研究的论文级评估，以及整合伊斯兰学者和领域专家关于真实性、权威性和负责任使用的观点。我们发现进展不均衡。数据资源已扩展，分割任务已成熟，叙述者和来源验证问题得到了更好的形式化，LLM辅助工作流现在支持语料库规模的丰富化、多语言访问和基于证据的评估。与此同时，进展仍然有限。

    arXiv:2608.20364v1 Announce Type: cross  Abstract: We examine how hadith computational science is being reshaped by transformer models, retrieval-grounded pipelines, and large language models (LLMs). Recent reviews document growth in the literature, but they do not yet provide a critical account of which advances are methodologically robust, which remain benchmark-bound, and which unresolved problems still limit scholarly use. We address this gap through a critical narrative review that combines critique of existing reviews, paper-level appraisal of representative original studies, and synthesis of Islamic scholar and domain-expert perspectives on authenticity, authority, and responsible use. We find uneven progress. Data resources have expanded, segmentation tasks have matured, narrator and source-verification problems are better formalized, and LLM-assisted workflows now support corpus-scale enrichment, multilingual access, and grounded evaluation. At the same time, progress remains 
    
[^74]: 多语言验证器偏差在RLVR中的研究：基准测试、回滚诊断与跨语言选择瓶颈

    Multilingual Verifier Bias in RLVR: Benchmark, Rollout Diagnosis, and the Cross-Lingual Selection Bottleneck

    [https://arxiv.org/abs/2608.20362](https://arxiv.org/abs/2608.20362)

    本文揭示了多语言环境中RLVR的精确匹配验证器因语言差异产生严重假阴性偏差，并提出了一个可复用的审计协议和诊断方法，指出跨语言选择瓶颈是核心问题。

    

    arXiv:2608.20362v1 公告类型：新 摘要：基于可验证奖励的强化学习（RLVR）是训练大型语言模型进行数学推理的标准方法，其中答案验证器充当语言中立的奖励函数。我们表明这一假设在多语言环境中不成立：精确匹配验证器将格式和脚本变化转化为依赖语言的假阴性奖励噪声。我们引入了一个可复用的多语言RLVR奖励审计协议：一个验证器鲁棒性测试套件、一个回滚诊断程序，以及针对日语、英语和中文答案的语言条件奖励误差指标。在k=8的MGSM回滚测试中，精确匹配代理对可信正确答案的拒绝率因语言不同而显著差异，涉及Qwen3-4B、Qwen3-8B和Llama-3.1-8B-Instruct模型；对于Qwen3-8B，日语上的假阴性率达到0.642，而英语为0.122，中文为0.073。一个纯数字探针将机制定位到最终答案接口：一个...

    arXiv:2608.20362v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) is a standard recipe for training large language models on mathematical reasoning, where an answer verifier serves as a language-neutral reward function. We show that this assumption fails in multilingual settings: an exact-match verifier turns format and script variation into language-dependent false-negative reward noise. We introduce a reusable protocol for auditing multilingual RLVR rewards: a verifier-robustness suite, a rollout-diagnosis procedure, and language-conditioned reward-error metrics for Japanese, English, and Chinese answers. On MGSM rollouts with k=8, the exact-match proxy rejects trusted-correct answers at sharply different rates by language across Qwen3-4B, Qwen3-8B, and Llama-3.1-8B-Instruct; for Qwen3-8B, the false-negative rate reaches 0.642 on JP against 0.122 on EN and 0.073 on CN. A plain-numeric probe localizes the mechanism to the final-answer interface: an
    
[^75]: 迈向自动研究：从具有范畴结构的论文知识图谱中挖掘可证伪的研究想法

    Toward Auto-Research: Mining Falsifiable Research Ideas from Paper Knowledge Graphs with Categorical Structure

    [https://arxiv.org/abs/2608.20361](https://arxiv.org/abs/2608.20361)

    本文提出用范畴论（组合与恒等箭头）为论文构建类型化知识图谱，从而生成可证伪的跨领域研究类比，克服传统方法将论文视为平面对象的局限。

    

    基于大型语言模型（LLMs）的自动研究想法生成系统存在一个结构性弱点：它们将构思过程简化为自由文本重组、随机论文配对或嵌入相似性检索。这三种方法以相同的方式失效：每种方法都将论文视为一个平面对象、一个字符串或一个向量，从而舍弃了研究者在推理跨领域类比时实际使用的类型化问题-方法-指标-主张箭头。我们通过范畴论中最小的结构片段（仅靠类型化图无法提供）恢复了缺失的结构：组合性以及恒等箭头，这使得我们能够询问一个提出的类比是否保持了关系链。具体而言，每篇论文$p$被建模为一个小子范畴$C_p$，其对象是提取的类型化研究实体，其态射是论文所断言的关系；从$p$到$q$的跨论文桥梁则是一个部分函子候选。

    arXiv:2608.20361v1 Announce Type: cross  Abstract: Automated research-idea generation systems built on large language models (LLMs) share a structural weakness: they reduce ideation to free-text recombination, random paper pairing, or embedding-similarity retrieval. The three approaches fail in the same way: each treats a paper as a flat object, a string or a vector, and so quotients away the typed problem-method-metric-claim arrows a researcher actually uses when reasoning about a cross-domain analogy. We recover the missing structure with the minimal piece of category theory that a typed graph alone does not provide: composition, together with identity arrows, which makes it possible to ask whether a proposed analogy preserves relation chains. Concretely, each paper $p$ is modelled as a small category $C_p$ whose objects are extracted typed research entities and whose morphisms are the relations the paper asserts; a cross-paper bridge from $p$ to $q$ is then a partial functor candida
    
[^76]: TriPLU：在微型语言模型中通过直接三线性乘积前馈网络绕过门控机制

    TriPLU: Bypassing the Gate with Direct Trilinear Product FFNs in Tiny Language Models

    [https://arxiv.org/abs/2608.20360](https://arxiv.org/abs/2608.20360)

    TriPLU通过直接三线性乘积分支替代门控机制，在微型语言模型中显著降低了验证损失，优于SwiGLU及其他乘积阶数控制。

    

    摘要：我们研究微型仅解码器语言模型是否能从直接相乘学习到的特征投影的前馈层中受益。TriPLU，即三线性乘积线性单元，用仅含乘积的3次分支替代了通常的门控前馈网络分支，该分支逐坐标相乘三个投影流。在字符级TinyStories 1M字节前缀研究中，TriPLU达到了平均最佳验证损失1.0637，而紧密匹配的SwiGLU为1.1017，4次乘积控制为1.0780，2次乘积控制为1.1026。在仅训练的Byte-BPE实验中，TriPLU在低学习率设置下也降低了TinyStories和WikiText-2原始数据的验证集和保留集的每字节比特数，PMI切片证据表明，在已见的中高PMI相邻词对上有增益。恒定学习率诊断显示，乘积分支归一化可以减少高学习率最佳检查点差距，尽管最终BPB仍会退化。

    arXiv:2608.20360v1 Announce Type: new  Abstract: We study whether tiny decoder-only language models benefit from feed-forward layers that directly multiply learned feature projections. TriPLU, a Trilinear Product Linear Unit, replaces the usual gated FFN branch with a product-only degree-3 branch that multiplies three projected streams coordinatewise. In a character-level TinyStories 1M-byte prefix study, TriPLU reaches a mean best validation loss of 1.0637, compared with 1.1017 for closely matched SwiGLU, 1.0780 for a degree-4 product control, and 1.1026 for a degree-2 control. In train-only Byte-BPE experiments, TriPLU also lowers validation and heldout bits per byte on TinyStories and WikiText-2 raw under low-learning-rate settings, with PMI-slice evidence suggesting gains on seen middle- and high-PMI adjacent-token pairs. Constant-learning-rate diagnostics show that product-branch normalization can reduce the high-learning-rate best-checkpoint gap, although final BPB still degrades
    
[^77]: 面向更快速推理模型的自我投机解码

    Self-Speculation for Faster Reasoning Models

    [https://arxiv.org/abs/2608.20359](https://arxiv.org/abs/2608.20359)

    本文提出SSR，一种无需训练的自我投机解码方法，利用部分思维链作为起草者、完整思维链作为验证者，以加速推理模型生成，无需额外训练。

    

    arXiv:2608.20359v1 公告类型：新 摘要：大型语言模型（LLMs）被部署用于越来越复杂的任务，涉及规划和多步决策，但这些任务的高质量表现通常需要生成较长的推理轨迹。这对于延迟敏感和交互式应用（如语音助手或编码代理）并不合适，因为生成延迟会强烈影响用户体验。现有加速方法通常专注于词元级别的生成，而未利用推理工作流的结构。我们引入了SSR：用于推理模型的自我投机解码，这是一种无需训练的自我投机解码方法，利用思维链（CoT）作为投机来源。SSR使用部分CoT答案分布作为起草者，完整CoT分布作为验证者，两者均来自同一模型，但采用不同的推理预算。这基于观察，即后来的部分CoT响应往往表现出更大的...

    arXiv:2608.20359v1 Announce Type: new  Abstract: Large language models (LLMs) are deployed for increasingly complex tasks involving planning and multi-step decision making, but high-quality performance on these tasks often requires generating long reasoning traces. This is a poor fit for latency-sensitive and interactive applications like voice assistants or coding agents, where generation latency can strongly affect user experience. Existing acceleration methods typically focus on token-level generation, without utilizing the structure of reasoning workflows. We introduce SSR: Self-Speculation for Reasoning Models, a training-free self-speculative decoding method that leverages the chain-of-thought (CoT) as a source of speculation. SSR uses the partial-CoT answer distribution as the drafter and the full-CoT distribution as the verifier, deriving both from the same model at different reasoning budgets. This builds on the observation that later partial-CoT responses often exhibit greate
    
[^78]: ExpertIVS：基于社会学专家驱动的大语言模型个体价值观模拟

    ExpertIVS: Sociological Expert Driven Individual Value Simulation in Large Language Models

    [https://arxiv.org/abs/2608.20355](https://arxiv.org/abs/2608.20355)

    提出ExpertIVS框架，利用社会学专家智能体对调查数据进行深度语义重构，以生成内部一致的个体价值观画像，并解决传统静态评估无法反映真实对话中价值取向的问题。

    

    大语言模型（LLM）智能体在社会模拟中展现出巨大潜力，但在精确建模个体价值体系方面仍面临挑战。现有方法大多机械地将调查问卷回答拼接进提示中，这会导致语义碎片化，无法捕捉人类价值体系的内在一致性。LLM的价值体系通常通过静态多项选择题进行评估，这无法评估真实对话互动中的价值取向。为解决这些问题，我们提出了ExpertIVS框架，该框架利用14个社会学专家智能体，通过结构化专业视角而非直接拼接回答来解读世界价值观调查（WVS）数据。这些专家智能体进行深度语义重构，生成稳健且内部一致的个体画像。为评估LLM与个体价值体系在二元交互中的一致性，我们进行了进一步研究。

    arXiv:2608.20355v1 Announce Type: cross  Abstract: Large Language Model (LLM) agents have demonstrated considerable potential for social simulation, yet struggle to accurately model individual value systems. Most existing methods mechanically stitch survey responses into prompts, which suffer from semantic fragmentation, failing to capture the internal coherence of human value systems. The value systems of LLMs are typically assessed using static multiple-choice questions, which fail to evaluate the value orientation in real-world dialogue interactions. To address these issues, we propose ExpertIVS, a framework employing 14 Sociological Expert Agents to interpret World Values Survey (WVS) responses through structured professional perspectives, rather than direct responses concatenation. These expert agents perform deep semantic reconstruction to generate robust and internally consistent individual profiles. To evaluate the consistency between LLMs and individual value systems during dy
    
[^79]: 分歧假说：揭示心理健康自然语言处理中的词汇干扰与标签偏差

    The Divergence Hypothesis: Unmasking Lexical Interference and Label Bias in Mental Health NLP

    [https://arxiv.org/abs/2608.20353](https://arxiv.org/abs/2608.20353)

    本文提出TSS诊断框架和分歧度统计量，揭示了心理健康NLP中词汇特征对人类标注与自动标注数据的差异影响，为标签来源审计提供了新方法。

    

    计算心理健康（CMH）分类器在分布偏移下性能往往会下降，因为人类标注者和远程监督流水线奖励不同的语言信号。我们引入了TSS（三流压力探针），一种多通道诊断框架，将文本分解为（A）词汇字符n-gram，（B）一个小的、大多无内容的形态句法通道，以及（C）一个154特征的心理语言学风格通道。在四个英文数据集（N=12,906）上，TSS揭示了词汇干扰效应：将词汇特征添加到风格通道会降低人类标注数据上的宏F1分数（平均下降0.072，p<10^-4），但在自动标注数据上则没有此效应。我们提出了分歧度（DoD），一种从计量经济学改编而来的双重差分统计量，用于标签来源审计，并带有实例级自助法推断；主要估计值为DoD(BC-A) = 0.0374，95%置信区间[0.0097, 0.0651]，p=0.0032。平台分层的仅推特DoD结果也进行了分析。

    arXiv:2608.20353v1 Announce Type: cross  Abstract: Computational mental health (CMH) classifiers often degrade under distribution shift because human annotators and distant-supervision pipelines reward different linguistic signals. We introduce TSS (Triple-Stream Stress probe), a multi-channel diagnostic framework that decomposes text into (A) lexical character n-grams, (B) a small, mostly content-free morpho-syntactic channel, and (C) a 154-feature psycholinguistic style channel. Across four English datasets (N=12,906), TSS reveals a lexical interference effect: adding lexical features to the style channel reduces Macro-F1 on human-labeled data (mean drop 0.072, p<10^-4) but not on auto-labeled data. We propose Degree of Divergence (DoD), a difference-in-differences statistic adapted from econometrics for label-source auditing, with instance-level bootstrap inference; the headline estimate is DoD(BC-A) = 0.0374, 95% CI [0.0097, 0.0651], p=0.0032. A platform-stratified Twitter-only DoD
    
[^80]: 探索性分析：合成英语RAG探测中文化标记谓词触发的PII放大未检测——一项谓词资源混杂审计

    Exploratory As-Analyzed No-Detection of Culturally-Marked Predicate-Triggered PII Amplification in a Synthetic-English RAG Probe: A Predicate-Resource-Confounded Audit

    [https://arxiv.org/abs/2608.20351](https://arxiv.org/abs/2608.20351)

    本论文通过预注册审计发现，在合成英语RAG系统中，刻板印象负载查询并未在干净信息渠道上放大PII泄露，且早期泄露信号受提示回显伪影污染。

    

    arXiv:2608.20351v1 公告类型：新 摘要：我们探讨了关于文化标记人群的刻板印象负载查询是否比等效的中性查询从检索增强生成（RAG）系统中泄露更多个人信息。我们在合成英语PII语料库上预注册了一项四文化审计（英裔盎格鲁、西班牙语拉丁美洲、阿拉伯语、印地语），比较了五种查询臂，称为刻板印象触发泄露增量（STLD）。事先说明两点：我们锁定的确证估计器从未运行，因此论文中的每项测试都是探索性或敏感性分析，所有计划偏差列在附录中。并且名称泄露指标受到提示回显伪影的污染：模型通常只是重新输出我们询问的名称，这夸大了表面泄露而无需任何检索。在更干净的渠道（电子邮件、电话、类似社保号、地址）上，经过多重比较校正后，我们未发现任何文化上的刻板印象驱动放大。由于我们的样本仅具有足够的功效……

    arXiv:2608.20351v1 Announce Type: new  Abstract: We ask whether stereotype-loaded queries about culturally marked people leak more personal information from a retrieval-augmented generation (RAG) system than otherwise-equivalent neutral queries. We pre-register a four-culture audit (en-Anglo, es-LATAM, Arabic, Hindi) on a synthetic English PII corpus, comparing five query arms we call the Stereotype-Trigger Leakage Delta (STLD). Two caveats up front. Our locked confirmatory estimator was never run, so every test in the paper is exploratory or sensitivity, with all plan deviations listed in the appendix. And the name-leakage metric is contaminated by a prompt-echo artifact: the model often just re-emits the name we asked about, which inflates apparent leakage without any retrieval at all. On the cleaner channels (email, phone, ssn-like, address), we find no stereotype-driven amplification on any of the four cultures after multiple-comparison correction. Because our sample is only powere
    
[^81]: 如何训练现实世界中的硅基管家？将复杂业务流程内化到单一模型中

    How to Train a Real-World Silicon Concierge? Internalizing Complex Business Workflow to Only OneModel

    [https://arxiv.org/abs/2608.20350](https://arxiv.org/abs/2608.20350)

    OneModel通过将复杂业务流程内化到单一模型参数中，取代模块化流水线，实现了延迟降低50%以上，并提高了准确性和效率。

    

    arXiv:2608.20350v1 公告类型：交叉 摘要：传统工业代理依赖模块化流水线，包括路由器、检索器、规划器、执行器、响应器、审查器及其他组件。这些系统常常碎片化为临时的补丁迷宫，导致级联错误和高延迟。我们提出OneModel，一种从外部工作流到内化知识表示的适用范式转变。与将流体用户意图切分为静态步骤的模块化系统不同，OneModel将复杂业务逻辑和标准操作流程直接整合到模型参数中。通过持续预训练（CPT）和逻辑编译监督微调（SFT），我们将碎片化的业务规则转化为统一注意力空间中的直观模型推理。在我们全球金融服务系统中部署后，OneModel有效打破了延迟、准确性和复杂性之间的权衡。在线A/B测试显示，端到端延迟降低了超过50%，从18.7秒降至约9.35秒。

    arXiv:2608.20350v1 Announce Type: cross  Abstract: Traditional industrial agents rely on modular pipelines, including Router, Retriever, Planner, Executor, Responder, Reviewer, and other components. These systems often fracture into a labyrinth of ad-hoc patches, leading to cascading errors and high latency. We propose OneModel, an applicable paradigm shift from external workflows to internalized knowledge representation. Unlike modular systems that slice fluid user intents into static steps, OneModel consolidates complex business logic and SOPs directly into the model parameters. Through Continual Pre-training (CPT) and logic-compilation SFT, we transform fragmented business rules into intuitive model reasoning within a unified attention space. Deployed in our global financial service system, OneModel effectively breaks the trade-off between latency, accuracy, and complexity. Online A/B testing demonstrates an end-to-end latency reduction of more than 50 percent, from 18.7 seconds to 
    
[^82]: 超越提示工程：提示词词汇敏感性的系统分析及其对质量的影响

    Beyond Prompt Engineering: A Systematic Analysis of Prompt Lexical Sensitivity and Its Impacts on Quality

    [https://arxiv.org/abs/2608.20349](https://arxiv.org/abs/2608.20349)

    该论文首次通过大规模n-gram级机制分析，揭示了提示性能稳定性缩放定律，并识别出领域特定术语和显式行动指令作为提升提示鲁棒性的两个核心语言驱动因素。

    

    大语言模型（LLMs）对表面级别的提示词变化表现出极端的敏感性，其中微小的词汇变化可能引发不成比例的性能波动。超越黑盒优化和粗粒度模板，我们首次提出了基于n-gram标记级别的提示稳定性的大规模机制分析，利用了包含132,000个提示变体的数据集。我们的研究揭示了一个基本的提示性能稳定性缩放定律：更高的平均任务性能与更低的方差和更强的跨提示扰动鲁棒性密切相关。我们识别出支撑这种鲁棒性的两个核心语言驱动因素：（1）领域特定术语，它紧密锚定语义边界，以及（2）显式行动指令，它形式化推理轨迹。这些元素共同约束了模型的解释空间，有效“锁定”了更确定性的生成行为。

    arXiv:2608.20349v1 Announce Type: cross  Abstract: Large Language Models (LLMs) exhibit extreme sensitivity to surface-level prompt variations, in which minor lexical changes can trigger disproportionate performance fluctuations. Moving beyond black-box optimization and coarse-grained templates, we present the first large-scale, n-gram token-level mechanistic analysis of prompt stability, leveraging a dataset of 132,000 prompt variants. Our investigation reveals a fundamental Scaling Law of Prompt Performance Stability: higher average task performance is strongly associated with lower variance and greater robustness across prompt perturbation. We identify two core linguistic drivers underlying this robustness: (1) Domain-Specific Terminology, which tightly anchors semantic boundaries, and (2) Explicit Action Directives, which formalize reasoning trajectories. Together, these elements constrain the model's interpretative space, effectively ``locking in'' more deterministic generation be
    
[^83]: 临床长上下文推理中的抑制性注意：表征并缓解电子健康记录处理中的“中间丢失”效应

    Inhibitory Attention for Clinical Long-Context Reasoning: Characterizing and Mitigating Lost-in-the-Middle Effects in EHR Processing

    [https://arxiv.org/abs/2608.20348](https://arxiv.org/abs/2608.20348)

    本文首次系统表征了电子健康记录处理中的临床“中间丢失”问题，并提出了查询条件临床抑制（QCCS）方法，以缓解长上下文推理中关键信息检索可靠性下降的问题。

    

    电子健康记录现在每个患者通常超过10万词元。然而，大型语言模型表现出“中间丢失”效应：长上下文中心附近的信息比边缘附近的信息检索可靠性更低。在临床应用中，这并非无害：病历中最重要的单一事实可能位于中心位置。我们将此称为临床“中间丢失”问题，使用MedAlign首次对其进行系统表征，并比较上下文选择策略作为补救措施。在2,196个指令-响应对和六个语言模型中，我们观察到峰值准确率（59.5%，95%置信区间[46.3, 71.0]，20-30%分位段）和谷值准确率（37.6% [23.2, 52.5]，70-80%分位段）之间存在21.9个百分点的差距；67.8%的参考答案位于电子健康记录时间线的10%至90%分位数之间，处于临床“中间丢失”低谷区。我们引入了查询条件临床抑制（QCCS），一种轻量级查询...

    arXiv:2608.20348v1 Announce Type: cross  Abstract: Electronic health records now routinely exceed 100,000 tokens per patient. Yet large language models exhibit the lost-in-the-middle (LitM) effect: information near the center of a long context is retrieved less reliably than information near the edges. In clinical use this is not benign: the single most consequential fact in a note can sit at its center. We term this the clinical lost-in-the-middle (CLitM) problem, give its first systematic characterization using MedAlign, and compare context-selection strategies as remedies. Across 2,196 instruction-response pairs and six language models, we observe a 21.9 percentage-point gap between peak accuracy (59.5%, 95% CI [46.3, 71.0], 20-30% decile) and trough accuracy (37.6% [23.2, 52.5] at 70-80%); 67.8% of reference answers fall between the 10th and 90th percentiles of the EHR timeline, inside the CLitM trough. We introduce Query-Conditioned Clinical Suppression (QCCS), a lightweight query
    
[^84]: 语言模型认为谁有能力？职业偏见的机制分析

    Who Do Language Models Think Is Competent? A Mechanistic Analysis of Occupational Bias

    [https://arxiv.org/abs/2608.20347](https://arxiv.org/abs/2608.20347)

    该论文提出一个因果框架来揭示语言模型中隐藏的职业偏见，证明即使行为上无差异，内部表征仍受人口统计学属性影响，影响用户能力判断。

    

    语言模型（LMs）通常能通过行为偏见评估，但尚不清楚它们是否不再表征导致偏见的潜在关联，还是仅仅学会了不表达这些偏见。在本研究中，我们表明表征性偏见通常可被检测到，即使行为偏见不可见。我们引入了一个因果框架，将职业偏见分解为两个测量点：模型对用户能力的内部表征，以及其可观察的输出。我们为表征用户专业知识的转向向量进行了推导，并验证了它们在问答任务和招聘任务中因果性地中介模型行为。将该框架应用于几个开放权重模型，我们发现人口统计学属性，如性别、种族和社会经济地位，会影响模型对用户专业知识的表征，即使在行为指标未检测到差异的情况下也是如此。

    arXiv:2608.20347v1 Announce Type: cross  Abstract: Language models (LMs) often pass behavioral bias evaluations, but it remains unclear whether they no longer represent the underlying associations that give rise to biases, or have merely learned not to express them. In this study, we show that representational biases are often detectable, even when behavioral biases are not visible. We introduce a causal framework that decomposes occupational bias into two measurement points: a model's internal representation of a user's competence, and its observable outputs. We derive steering vectors for representations of user expertise, and verify that they causally mediate model behavior in both a question-answering task and a hiring task. Applying this framework to several open-weight models, we find that demographic attributes, such as gender, race, and socioeconomic status, influence a model's representation of user expertise, even in cases where behavioral metrics detect no disparity between 
    
[^85]: 构建并评估面向电信客服的合成孟加拉语语音资源

    Building and Evaluating a Synthetic Bengali Speech Resource for Telecom Customer Care

    [https://arxiv.org/abs/2608.20346](https://arxiv.org/abs/2608.20346)

    该论文构建并公开了一个包含10,000对音频-文本、约26.82小时的合成孟加拉语电信客服语音数据集，并验证了其可懂度。

    

    arXiv:2608.20346v1 公告类型：新 摘要：面向客户的应用中所使用的语音系统通常需要特定领域的语言覆盖。我们提出一个用于电信客服场景的合成孟加拉语语音数据集。该数据集包含10,000对音频-文本对，约26.82小时的24kHz语音，并预定义了训练、验证和测试划分，分别为9,000、500和500个样本。该数据集已在Hugging Face上以CC-BY-4.0许可证公开发布。语音使用OmniVoice在语音克隆模式下生成，采用真实女性参考录音和转录文本，使用bfloat16精度、16次扩散采样步骤和1.0的语速控制值。除了原始孟加拉语文本外，数据集还提供了一个规范化转录字段，用于ASR/STT训练和评估。我们报告了使用从bengaliAI/tugstugi_bengaliai-regional-asr_whisper微调而来的领域自适应Whisper ASR模型对所有10,000个样本进行的自动可懂度检查。

    arXiv:2608.20346v1 Announce Type: new  Abstract: Speech systems used in customer-facing applications often require domain-specific language coverage. We present a synthetic Bengali speech dataset for telecom customer-care scenarios. The dataset contains 10,000 audio-text pairs, approximately 26.82 hours of 24 kHz speech, and predefined train, validation, and test splits of 9,000, 500, and 500 examples. It is publicly released on Hugging Face under the CC-BY-4.0 license. The speech was generated with OmniVoice in voice-cloning mode using a real female reference recording and transcript, with bfloat16 precision, 16 diffusion sampling steps, and a speaking-rate control value of 1.0. Along with the original Bengali text, the dataset provides a normalized transcript field designed for ASR/STT training and evaluation. We report an automatic intelligibility check over all 10,000 samples using a domain-adapted Whisper ASR model fine-tuned from bengaliAI/tugstugi_bengaliai-regional-asr_whisper-
    
[^86]: 当词汇理解失效于临床推理：评估治疗机器人对Alpha世代的安全风险

    When Vocabulary Comprehension Fails Clinical Reasoning: Evaluating Therapy Bots' Safety Risks for Generation Alpha

    [https://arxiv.org/abs/2608.20345](https://arxiv.org/abs/2608.20345)

    本研究首次系统评估了治疗机器人在应对Alpha世代独特语言模式时的安全风险，并提出了两个基准数据集来揭示其临床推理中的词汇理解失败。

    

    arXiv:2608.20345v1 公告类型：交叉 摘要：对话式AI系统已成为Alpha世代（Gen Alpha，出生于2010-2024年）的非正式心理健康支持资源，13.1%的美国青少年（540万人）使用生成式AI获取心理健康建议。尽管这些系统（从治疗应用到通用聊天机器人）依赖于在广泛心理学文献上训练的大型语言模型，但它们对青少年沟通模式（以夸张语言、讽刺性积极、快速语义漂移和上下文多义性为特征）的安全性尚未得到验证。在多起与AI聊天机器人互动相关的青少年死亡事件后，系统性评估至关重要。我们提出了两个基准：（1）64条由母语者（ICC=0.72）和临床医生（kappa=0.78）验证的Gen Alpha心理健康表达；（2）75个多轮对话（780轮），配有配对的Standard/Gen Alpha版本。在评估支撑治疗应用和通用聊天机器人的LLM架构（如Claude）中，我们发现...

    arXiv:2608.20345v1 Announce Type: cross  Abstract: Conversational AI systems have become informal mental health support resources for Generation Alpha (Gen Alpha, born 2010-2024), with 13.1% of U.S. adolescents (5.4 million) using generative AI for mental health advice. While these systems, from therapy apps to general chatbots, rely on large language models trained on extensive psychological literature, their safety for youth communication patterns characterized by hyperbolic language, ironic positivity, rapid semantic drift, and contextual polysemy remains unvalidated. Following multiple adolescent deaths linked to AI chatbot interactions, systematic evaluation is critical. We present two benchmarks: (1) 64 Gen Alpha mental health expressions validated by native speakers (ICC=0.72) and clinicians (kappa=0.78); (2) 75 multi-turn conversations (780 turns) with paired Standard/Gen Alpha versions. Across evaluations of LLM architectures underlying therapy apps and general chatbots - Clau
    
[^87]: 超越原始转录：基于LLM的数字孪生中的结构化人物画像提取

    Beyond Raw Transcripts: Structured Persona Extraction for LLM-Based Digital Twins

    [https://arxiv.org/abs/2608.20344](https://arxiv.org/abs/2608.20344)

    本文提出，数字孪生预测准确性的关键瓶颈在于人物画像信息的结构组织方式，而非信息量，并通过引入基于消费者行为理论的BDE结构化模式，将预测准确性提升1.91个百分点。

    

    arXiv:2608.20344v1 公告类型：新论文 摘要：基于LLM的“数字孪生”旨在模拟个体在新环境中的行为或对新问题的回应，这需要基于该个体先前回应的某种表示。一种常见方法是从调查转录或总结回应中构建这种表示。先前工作表明，将长转录压缩为较短的LLM生成摘要不会显著降低预测准确性，这表明信息量并非主要瓶颈。在本工作中，我们认为关键限制在于结构：即人物画像信息在提供给模拟器模型之前的组织方式。我们通过比较非结构化摘要与结构化人物画像表示来研究这一点。首先，我们引入一个手工设计的模式（BDE：背景、决策过程、评估），该模式基于消费者行为理论，并证明其相对于原始转录将预测准确性提高了1.91个百分点。

    arXiv:2608.20344v1 Announce Type: new  Abstract: LLM-based "digital twins" aim to simulate how an individual would behavein new environments or respond to novel questions, given some representation of that individual's prior responses. A common approach constructs this representation from survey transcripts or summaries responses. Prior work shows that compressing long transcripts into shorter LLM-generated summaries does not significantly reduce predictive accuracy, suggesting that information volume is not the primary bottleneck.   In this work, we argue that the key limitation is instead structural:how persona information is organized before being provided to thesimulator model. We study this by comparing unstructured summaries with structured persona representations. First, we introduce a hand-craftedschema (BDE: Background, Decision procedure, Evaluation), grounded in consumer-behavior theory, and show that it improves predictive accuracy over raw transcripts by +1.91 percentage p
    
[^88]: 缓解大语言模型智能体中的身份本质主义：基于纵向生活轨迹的方法

    Mitigating Identity Essentialism in LLM Agents with Longitudinal Life Trajectories

    [https://arxiv.org/abs/2608.19621](https://arxiv.org/abs/2608.19621)

    本文提出LifeMem框架，通过结合结构化生活事件检索和参数化记忆，缓解大语言模型智能体因静态画像导致的身份本质主义，从而增强社会模拟中的人口多样性。

    

    大语言模型（LLMs）提供了一种可扩展的社会模拟方法，但其可信度取决于智能体的构建方式。现有方法能部分复现群体层面的模式，但往往难以捕捉类人的多样性。我们的分析表明，具有静态画像的智能体在人口统计特征分离和组内压缩方面比人类更强，这一模式与身份本质主义一致：人口统计标签可能促使模型将群体平均倾向视为个体特征，从而在组内同质化响应。我们认为，这一局限源于两个相关因素：稀疏、静态的智能体表征，以及仅提示记忆在持续整合经验方面的有限能力。受互补记忆系统的启发，我们提出了LifeMem，一种纵向记忆框架，结合结构化生活事件检索与智能体特定的参数化记忆，用于经验整合。

    arXiv:2608.19621v1 Announce Type: new  Abstract: Large language models (LLMs) offer a scalable approach to social simulation, but their credibility depends on how agents are constructed. Existing methods can partially reproduce population-level patterns, yet often fail to capture human-like diversity. Our analysis shows that static-profile agents exhibit stronger demographic separation and within-group compression than humans, a pattern consistent with identity essentialism: demographic labels can encourage models to treat group-average tendencies as individual traits, homogenizing responses within groups. We argue that this limitation arises from two related factors: sparse, static agent representations and the limited ability of prompt-only memory to persistently integrate experience. Inspired by complementary memory systems, we propose LifeMem, a longitudinal memory framework that combines structured life-event retrieval with agent-specific parametric memory for experience integrati
    
[^89]: Hear2Act：基准测试韵律何时应改变助手的行动

    Hear2Act: Benchmarking When Prosody Should Change What an Assistant Does

    [https://arxiv.org/abs/2608.19515](https://arxiv.org/abs/2608.19515)

    本文提出了Hear2Act，一个统一基准，用于测试韵律线索是否以及何时能改变面向任务助手的下游决策，并证明添加音频信息可影响最优解率。

    

    arXiv:2608.19515v1 公告类型：新 摘要：韵律线索可以传达与任务相关的信息，即使词语本身不变，也能改变面向任务对话的轨迹和结果。然而，现有基准通常孤立地评估韵律感知、响应适当性和面向任务对话，这使得难以测试韵律证据是否改变下游决策。我们引入了Hear2Act，一个统一的评估协议，用于文本和语音助手，包含480个基于人格的场景、隐藏的用户关注点和客观可验证的结果。对于每个场景，我们保持任务和用户需求固定，同时变化同一关注点是通过词语明确表达还是主要通过韵律传达，并在转录文本、音频和关注点状态访问下评估决策。使用Hear2Act，我们评估了两个支持音频的大语言模型。在韵律介导的反馈下，将音频添加到转录文本中会改变平均最优解率。

    arXiv:2608.19515v1 Announce Type: new  Abstract: Prosodic cues can convey task-relevant information that alters the trajectory and outcome of a task-oriented dialogue, even when the words themselves remain unchanged. Yet existing benchmarks typically evaluate prosodic perception, response appropriateness, and task-oriented dialogue in isolation, making it difficult to test whether prosodic evidence changes downstream decisions. We introduce Hear2Act, a unified evaluation protocol for text and spoken assistants with 480 persona-grounded scenarios, hidden user concerns, and objectively verifiable outcomes. For each scenario, we keep the task and user needs fixed while varying whether the same concern is conveyed explicitly in words or primarily through prosody, and evaluate decisions under transcript, audio, and concern-state access.   Using Hear2Act, we evaluate two audio-capable LLMs. Under Prosody-mediated feedback, adding audio to the transcript changes the average optimal-solution r
    
[^90]: SuTRA：具有词根意识的结构统一分词法

    SuTRA : Structurally-Unified Tokenization with Root Awareness

    [https://arxiv.org/abs/2608.18087](https://arxiv.org/abs/2608.18087)

    SuTRA是一种形态感知分词算法，通过保持akshara完整性和惩罚跨形态边界的合并，有效减少了形态破碎化，在印度语言上显著提升了形态对齐和语义可恢复性，并提高了机器翻译性能。

    

    arXiv:2608.18087v1 公告类型：交叉 摘要：现有的子词分词器优化统计压缩，但忽视了形态结构，特别是词根与词缀之间的关系。这对于形态丰富的印度语言是有害的，因为这些语言的基本单位是复杂的音节字符（aksharas），而非字母。基于频率的方法过度切分词语，任意分割词根和词缀——我们将此现象称为“形态破碎化”。我们提出了SuTRA（具有词根意识的结构统一分词法），这是一种形态感知算法，它保持akshara的不可分割性，并惩罚跨越形态边界的合并。我们还为印地语、马拉地语和古吉拉特语发布了一个新的形态分割数据集。SuTRA减少了破碎化，在形态对齐（边界F1）方面最高提升+14.7%，在语义可恢复性（印地语）方面最高提升+34%，优于BPE。这些结构上的改进在机器翻译中平均提升了+8.08 chrF2。

    arXiv:2608.18087v1 Announce Type: cross  Abstract: Existing subword tokenizers optimize statistical compression but ignore morphological structure, particularly the relationship between roots and affixes. This is harmful for morphologically rich Indic languages, where basic units are complex orthographic syllables (aksharas) rather than letters. Frequency-based methods over-fragment words, arbitrarily splitting roots and affixes - a phenomenon we term Morphological Shattering. We propose SuTRA (Structurally-Unified Tokenization with Root Awareness), a morphology-aware algorithm that preserves akshara indivisibility and penalizes merges crossing morphological boundaries. We also release a new morphological segmentation dataset for Hindi, Marathi, and Gujarati. SuTRA reduces shattering, achieving peak gains of +14.7% in morphological alignment (Boundary F1) and +34% in semantic recoverability (Hindi) over BPE. These structural gains yield an average improvement of +8.08 chrF2 in machine 
    
[^91]: 大型语言模型是否玩转六度分隔？长上下文流形中的拓扑压缩测量

    Do Large Language Models Play Six Degrees of Separation? Measuring Topological Compression in Long-Context Manifolds

    [https://arxiv.org/abs/2608.17950](https://arxiv.org/abs/2608.17950)

    本文绕过注意力权重，直接分析隐藏状态流形的动态几何结构，发现大型语言模型的深层潜在空间自发形成小世界网络，并展现出从碎片化到高度可导航的拓扑相变，从而实现了长上下文中的远距离语义压缩。

    

    大型语言模型（LLMs）在长上下文上展现出卓越的多跳推理能力，然而支撑这些远距离认知跳跃的内部机制仍鲜为人知。传统的基于注意力的可解释性方法常因注意力汇聚等路由伪影而难以捕捉真实的语义邻近性。在本文中，我们绕开注意力权重，直接分析隐藏状态流形的动态几何结构，证明深层LLM潜在空间天然组织成小世界网络。通过将长上下文表示的连续相似矩阵稀疏化为无权图，我们追踪了两种不同架构中高度不连通的语义锚点之间的连接性。我们的发现揭示了一个尖锐的拓扑相变：虽然早期句法层完全碎片化，深层推理层却突然将巨大的概念距离压缩为高度可导航的路径。

    arXiv:2608.17950v1 Announce Type: new  Abstract: Large Language Models (LLMs) demonstrate remarkable multi-hop reasoning capabilities over long contexts, yet the internal mechanisms enabling these distant cognitive leaps remain poorly understood. Traditional attention-based interpretability often fails to capture true semantic proximity due to routing artifacts like attention sinks. In this paper, we bypass attention weights to directly analyze the dynamic geometry of the hidden state manifold, proving that deep LLM latent spaces natively organize into Small-World networks. By sparsifying the continuous similarity matrices of long-context representations into unweighted graphs, we trace the connectivity between highly disjoint semantic anchors across two distinct architectures. Our findings reveal a sharp topological phase transition: while early syntactic layers remain entirely fractured, deep reasoning layers abruptly compress massive conceptual distances into highly navigable pathwa
    
[^92]: Mint-Agent：引入金融原生的智能体基础模型

    Mint-Agent: Introducing Finance-Native Agentic Foundation Models

    [https://arxiv.org/abs/2608.16386](https://arxiv.org/abs/2608.16386)

    本文提出Mint-Agent，一种金融原生智能体基础模型，通过数据引擎、MintHarness框架和结合SFT、OPD与RLVR的训练算法，实现可靠且可审计的长周期金融研究执行。

    

    金融智能体必须超越领域知识的回忆：它们既要可靠，能够在有根据的证据上执行精确操作；又要具备执行力，能够维持长周期研究，其结论保持可审计性。我们提出了Mint-Agent，一个围绕这两个金融智能尺度设计的金融原生智能体模型系列。Mint-Agent基于三大支柱构建：数据、框架和算法。我们的数据引擎从真实金融来源构建干净、专门的任务，用于原子金融能力和长周期智能体执行。MintHarness支持与开放环境的稳定交互，并在扩展研究轨迹中维持可审计的证据链。我们的训练配方结合了SFT、关键步骤OPD和RLVR，以开发独立的金融推理和智能体执行专家，然后通过模型合并和多教师在线策略蒸馏统一成紧凑模型。

    arXiv:2608.16386v1 Announce Type: new  Abstract: Financial agents must do more than recall domain knowledge: they must be both reliable, executing precise operations over grounded evidence, and executive, sustaining long-horizon research whose conclusions remain auditable. We present Mint-Agent, a family of finance-native agentic models designed around these two scales of financial intelligence. Mint-Agent is built upon three pillars: data, harness, and algorithm. Our data engine constructs clean, specialized tasks for atomic financial capabilities and long-horizon agentic execution from real-world financial sources. MintHarness enables stable interaction with open-ended environments and maintains auditable evidence trails across extended research trajectories. Our training recipe combines SFT, critical-step OPD, and RLVR to develop separate financial reasoning and agentic execution experts, which are then unified through model merging and multi-teacher on-policy distillation into comp
    
[^93]: DFM Mimir v1：一个仅使用许可后训练数据、在1B参数下实现前沿性能的开源HRM模型

    DFM Mimir v1: An Open HRM Delivering Frontier Performance at 1B Parameters Using Only Permissible Post-Training Data

    [https://arxiv.org/abs/2608.13517](https://arxiv.org/abs/2608.13517)

    Mimir v1是一个10亿参数的HRM架构语言模型，仅使用许可数据训练，在英语和丹麦语上实现了前沿性能，并超越了同尺寸及更大尺寸的模型。

    

    当前大型语言模型的开发依赖于庞大且通常未经许可的数据集，这为致力于开源和道德数据来源的研究人员设置了高门槛。我们介绍了Mimir v1，这是一个基于层次推理模型（HRM）架构的10亿参数语言模型，从零开始训练，在英语方面表现出极具竞争力的性能，并在仅使用许可后训练数据的情况下，为丹麦语设立了新的最优水平。该模型在161个数据集的混合上训练，在英语、数学与代码以及丹麦语的20个基准测试中，Mimir v1超越了原始的HRM-Text 1B，并与更大的前沿模型（如Qwen 3.5 4B和Gemma 4 E2B）竞争。该模型可在Hugging Face Hub上获取：https://huggingface.co/danish-foundation-models/DFM-Mimir

    arXiv:2608.13517v1 Announce Type: cross  Abstract: Current large language model development relies on massive, often non-permissible datasets, creating a high barrier for researchers committed to open-source and ethically sourced data. We introduce Mimir v1, a 1-billion-parameter language model based on the Hierarchical Reasoning Model (HRM) architecture, that is trained from scratch and delivers highly competitive performance for English and sets a new state of the art for Danish using only permissible post-training data. Trained on a mixture of 161 datasets, Mimir v1 outperforms the original HRM-Text 1B and competes with larger frontier models like Qwen 3.5 4B and Gemma 4 E2B, tested across 20 benchmarks for English, Math & Code and Danish. The model is available on the Hugging Face Hub: https://huggingface.co/danish-foundation-models/DFM-Mimir
    
[^94]: 阅读认知随词汇展开的决策：一种因子化逆决策模型

    Reading Cognition as Decisions Unfold in Words: A Factorized Inverse Decision Model

    [https://arxiv.org/abs/2608.09222](https://arxiv.org/abs/2608.09222)

    提出了一种因子化逆决策模型，通过将任务执行分解为动作和努力因子，从言语转录中推断认知决策过程，在老年人购物对话任务中实现了选择性估计并保留了动作区分。

    

    arXiv:2608.09222v2 公告类型：替换 摘要：逆决策建模从观察到的行为中推断决策过程的潜在属性，但现有公式主要依赖动作轨迹。在言语化认知任务中，任务执行还会产生动作公式未建模的响应动态，例如言语生成、交互和犹豫。我们提出一种因子化逆决策模型（FIDM），它将每个个体的任务执行似然分解为动作因子和努力因子，由独立的个体特定参数控制。从原始言语转录中，语言模型生成结构化任务执行轨迹以进行因子化推断。在400名老年人执行购物对话任务以进行认知筛查的数据上，受控恢复显示对预期因子的选择性估计，而匹配的半合成条件表明，即使聚合时，FIDM也能保留动作执行的区别。

    arXiv:2608.09222v2 Announce Type: replace  Abstract: Inverse decision modeling infers latent properties of decision processes from observed behavior, but existing formulations rely primarily on action trajectories. In verbalized cognitive tasks, task execution also produces response dynamics that action-only formulations leave unmodeled, such as verbal production, interaction, and hesitation. We propose a factorized inverse decision model (FIDM) that decomposes each individual's task-execution likelihood into an action factor and an effort factor, governed by separate individual-specific parameters. From raw verbal transcripts, a language model produces structured task-execution traces for factorized inference. On data from 400 older adults performing a grocery-shopping dialog task for cognitive screening, controlled recovery shows selective estimation of the intended factors, while matched semi-synthetic conditions show that FIDM preserves action-execution distinctions even when aggre
    
[^95]: 经验之树：面向自进化智能体的层级化经验管理

    Tree-of-Experience: Hierarchical Experience Management for Self-Evolving Agents

    [https://arxiv.org/abs/2608.09044](https://arxiv.org/abs/2608.09044)

    本文提出经验之树（ToE）框架，通过将经验组织成与LLM智能体层级化推理过程对齐的共享树结构，解决了现有经验管理方法中反馈归因、跨任务迁移和更新检索效率低的问题。

    

    arXiv:2608.09044v2 公告类型：替换  摘要：持续的自我进化要求大语言模型智能体将环境交互转化为可靠且可复用的经验。现有方法通常细化个体轨迹或从相关轨迹中抽象共享知识，但其经验表示往往与底层推理过程脱节。这限制了反馈归因、跨任务迁移以及更新和检索效率，特别是在具有结果级反馈的复杂推理任务中。为克服这一局限，我们提出经验之树（ToE），一种结构化经验管理框架，将经验组织与LLM智能体的层级化推理过程对齐。具体而言，ToE将经验组织为共享的分析视角和推理路径树，其可靠性通过环境结果进行校准，以支持系统性更新、迁移和高效检索。

    arXiv:2608.09044v2 Announce Type: replace  Abstract: Continual self-evolution requires LLM agents to transform environmental interactions into reliable and reusable experience. Existing methods typically refine individual trajectories or abstract shared knowledge from related trajectories, but their experience representations are often disconnected from the underlying reasoning process. This limits feedback attribution, cross-task transfer, and update and retrieval efficiency, particularly in complex reasoning tasks with outcome-level feedback. To overcome this limitation, we propose \textbf{T}ree-\textbf{o}f-\textbf{E}xperience (ToE), a structured experience-management framework that aligns experience organization with the hierarchical reasoning process of LLM agents. Specifically, ToE organizes the experience into a shared tree of analytical perspectives and reasoning paths, whose reliability is calibrated through environmental outcomes to support systematic updating, transfer, and e
    
[^96]: 声纹谬误：为何声音并非独特的生物识别印记

    The Voiceprint Fallacy: Why Voices Are Not Unique Biometric Imprints

    [https://arxiv.org/abs/2608.07980](https://arxiv.org/abs/2608.07980)

    本文揭示“声纹”作为稳定独特生物识别印记的谬误，强调声音的动态性和情境依赖性，并探讨了深度伪造技术对说话人身份认定的新挑战。

    

    摘要：近年来，“声纹”一词重新受到关注，尤其在技术应用和政策制定领域，常带有一种假设，即人的声音构成一种稳定且独特的生物识别痕迹，类似于指纹。然而，这一概念自提出以来的数十年间，屡遭法庭语音专家的批评和否定。尽管声音确实包含与说话者相关的信息，但这种简化概念掩盖了语音的高度动态性和情境依赖性。本文通过回顾声纹识别的历史发展、人类语音变异性的证据、法庭语音比较的进展、人类和自动说话人识别的研究，以及深度伪造语音对说话人身份带来的最新挑战，重新审视声纹谬误，并思考什么才能算作说话人身份的证据。

    arXiv:2608.07980v2 Announce Type: replace-cross  Abstract: In recent years, the term voiceprint has regained attention, particularly in technological applications and policy-making contexts, often carrying the assumption that a person's voice constitutes a stable and unique biometric trace analogous to a fingerprint. Yet this conception has been repeatedly criticized and rejected by forensic voice experts throughout the decades since its introduction. Although voices undoubtedly contain speaker-related information, this simplified conception obscures the highly dynamic and context-dependent nature of speech. This article revisits the voiceprint fallacy and reconsiders what can count as evidence of speaker identity by reviewing the historical development of voiceprint identification, evidence on human voice variability, developments in forensic voice comparison, research on human and automatic speaker recognition, and the recent challenge posed by deepfake speech to speaker identity. We
    
[^97]: SMOPD：通过专业化与合并的在线策略蒸馏实现多奖励强化学习

    SMOPD: Multi-Reward Reinforcement Learning via Specialize-and-Merge Online Policy Distillation

    [https://arxiv.org/abs/2608.03092](https://arxiv.org/abs/2608.03092)

    本文提出一种通过专业化与合并的在线策略蒸馏方法（SMOPD），以增强稀疏奖励的优化信号，同时保持密集奖励驱动的能力，解决多奖励强化学习中不同粒度奖励信号失衡的问题。

    

    arXiv:2608.03092v2 公告类型：替换-交叉 摘要：我们旨在提升多奖励强化学习训练过程中的模型性能。现有的分组奖励解耦归一化策略优化（GDPO）方法通过在聚合前分别对每个奖励维度进行归一化，缓解了直接标量化过程中奖励信号相互掩盖的问题。然而，我们的实验表明，GDPO在处理具有不同粒度的奖励信号时仍存在困难。具体而言，在某些特定训练任务中，模型可能接收一个密集奖励，其提供从0.1到1.0的细粒度评分，同时伴随一个仅提供0或1二元反馈的稀疏奖励。在这种情况下，我们发现稀疏奖励可能提供不足的优化信号，导致其对应的能力无法被有效强化。因此，如何在不过度牺牲其他能力的前提下，增强来自稀疏奖励的优化信号，成为关键问题。

    arXiv:2608.03092v2 Announce Type: replace-cross  Abstract: We aim to improve model performance in multi-reward reinforcement learning training process. Existing Group reward-Decoupled Normalization Policy Optimization (GDPO) has mitigated the issue of reward signals masking one another during direct scalarization by normalizing each reward dimension separately before aggregation. However, our experiments show that GDPO still struggles to balance reward signals with different granularities. Specifically, in some particular training tasks, the model may receive a dense reward that assigns fine-grained scores ranging from 0.1 to 1.0, together with a sparse reward that provides only binary feedback of either 0 or 1. In such cases, we find that the sparse reward may provide an insufficient optimization signal, preventing its corresponding capability from being effectively reinforced. Therefore, how can we strengthen the optimization signal from the sparse reward without sacrificing the capa
    
[^98]: 编码代理中的提示诱发浪费：推理、努力、框架设计与端到端成本

    Prompt-Induced Waste in Coding Agents: Reasoning, Effort, Harness Design, and End-to-End Cost

    [https://arxiv.org/abs/2608.01347](https://arxiv.org/abs/2608.01347)

    本论文揭示提示语义、推理努力和框架设计是相互作用的因素，共同决定编码代理的端到端成本与成功率，而非独立可调的控制变量。

    

    编码代理的效率不能仅通过令牌数量或模型价格来表征。我们研究了端到端成本和任务成功率如何共同依赖于提示语义、推理努力、框架策略、模型、任务难度、工具使用、上下文管理和提供商计费。受控提示实验表明，措辞可以在不改变任务的情况下改变推理和验证行为。一项独立的SWE-bench Verified研究显示，额外的推理努力可以改善某些模型的困难任务，但也可能增加成本而无收益。DeepSeek框架扩展表明，即使模型、任务、提示和控制器逻辑保持固定，努力控制干预的效果也会随框架变化而显著改变。这些结果表明，提示、努力和框架是相互作用的实验因素，而非独立的效率控制。我们将效率建模为每项成功任务的成本。

    arXiv:2608.01347v4 Announce Type: replace  Abstract: Coding-agent efficiency cannot be characterized by token count or model price alone. We study how end-to-end cost and task success depend jointly on prompt semantics, inference effort, harness policy, model, task difficulty, tool use, context management, and provider accounting. Controlled prompt experiments show that wording can change reasoning and verification behavior without changing the task. A separate SWE-bench Verified study shows that additional inference effort can improve difficult tasks for some models but can also add cost without benefit. A DeepSeek Harness extension shows that the effect of an effort-control intervention changes substantially when the harness changes, even when the model, tasks, prompts, and controller logic are held fixed. These results show that prompt, effort, and harness are interacting experimental factors rather than independent efficiency controls. We model efficiency as cost per successful tas
    
[^99]: ZenGen：大语言模型的社会心智

    ZenGen: Social Mind for LLMs

    [https://arxiv.org/abs/2607.23740](https://arxiv.org/abs/2607.23740)

    本文提出了ZenGen框架，通过SoMBench基准测量大语言模型的社会智能，并采用诊断驱动的训练方案来内化和提升其社会认知能力。

    

    随着大语言模型从孤立的任务解决向人类环境中的长期服务转变，它们需要社会智能：即推断心理状态、追踪社会关系、基于规范推理以及根据情境调整行为的能力。本报告介绍了ZenGen，一个用于测量、内化和落地社会智能的集成框架。在测量方面，我们提出了SoMBench，一个基于心理学构建的基准，涵盖3个主要维度、17个次要维度和71个任务范式。它控制了问题格式、叙事视角和上下文长度，跨越284个共享场景和3,481个专家验证实例。对20个代表性LLM的评估显示出巨大的提升空间：最佳模型仅达到72.08%的总体准确率，且17个次要维度中没有一个达到90%的接近天花板区间。在内化方面，我们开发了ZenGen，一种基于诊断驱动的训练方案，结合了监督式（此处原文截断，但根据上下文推测为“监督式微调”等）方法。

    arXiv:2607.23740v2 Announce Type: replace  Abstract: As large language models move from isolated task solving toward long-term service in human environments, they require social intelligence: the ability to infer mental states, track social relations, reason over norms, and adapt behavior under context. This report presents ZenGen, an integrated framework for measuring, internalizing, and grounding social intelligence. For measurement, we introduce SoMBench, a psychology-grounded benchmark spanning 3 primary dimensions, 17 secondary dimensions, and 71 task paradigms. It controls question format, narrative perspective, and context length across 284 shared scenarios and 3,481 expert-verified instances. Evaluation of 20 representative LLMs reveals substantial headroom: the best model achieves only 72.08% overall accuracy, and none of the 17 secondary dimensions reaches the 90% near-ceiling band. For internalization, we develop ZenGen, a diagnosis-driven training recipe combining supervise
    
[^100]: 语言影响多语言大语言模型中的指令层级遵从性

    Language Shapes Instruction Hierarchy Compliance in Multilingual LLMs

    [https://arxiv.org/abs/2607.23545](https://arxiv.org/abs/2607.23545)

    本研究提出了多语言指令层级基准XIH-Bench，发现IH遵从性存在语言依赖的不对称性和跨语言冲突中的“语言边界效应”，表明语言显著影响多语言LLM的指令优先级遵从。

    

    arXiv:2607.23545v2 公告类型：替换 摘要：指令层级（IH）要求模型按来源对指令进行优先级排序，确保高优先级指令覆盖低优先级指令。尽管这对安全可控的部署至关重要，但现有评估几乎完全集中在英语上，使得IH遵从性在多语言环境中是否保持稳定尚不明确。我们引入了XIH-Bench，一个多语言IH评估基准，涵盖六种语言、四个领域和三种IH设置下的同语言和跨语言冲突。跨模型观察，我们发现两个一致的模式。首先，IH遵从性表现出明显的语言依赖不对称性：一种语言在增强高优先级位置遵从性的同时，可能在低优先级位置产生破坏性影响。其次，跨语言冲突比同语言冲突产生更高的遵从性，我们将此现象称为“语言边界效应”。我们进一步展示了语言在影响遵从性中的作用。

    arXiv:2607.23545v2 Announce Type: replace  Abstract: Instruction hierarchy (IH) requires models to prioritize instructions by source, ensuring that higher-priority instructions override lower-priority ones. Despite its importance for safe and controllable deployment, existing evaluations have focused almost exclusively on English, leaving it unclear whether IH compliance remains stable in multilingual settings. We introduce XIH-Bench, a benchmark for multilingual IH evaluation with both same-language and cross-language conflicts across six languages, four domains, and three IH settings. Across models, we find two consistent patterns. First, IH compliance exhibits a clear language-dependent asymmetry: a language that strengthens compliance in the higher-priority position can become disruptive in the lower-priority position. Second, cross-language conflicts yield higher compliance than same-language conflicts, a phenomenon we term the Language Boundary Effect. We further show that langua
    
[^101]: Index SLM技术报告

    Index SLM Technical Report

    [https://arxiv.org/abs/2607.09885](https://arxiv.org/abs/2607.09885)

    该论文介绍了哔哩哔哩开发的Index-1.9B系列开放小语言模型，包含四个变体，并通过创新的Warmup-Stable-Decay学习率调度和Norm-Head输出层，在2.8万亿令牌上实现了稳定且高效的预训练，支持角色扮演定制。

    

    arXiv:2607.09885v3 公告类型：替换 摘要：我们介绍了Index-1.9B，这是由哔哩哔哩开发的一系列开放小语言模型。该系列包含四个模型：Index-1.9B-Base，一个基础模型，具有19亿非嵌入参数，在2.8万亿个以中英文为主的令牌上进行了预训练；Index-1.9B-Pure，一个控制变体，使用相同的配方训练，但严格从语料库中过滤掉了所有类似指令的数据；Index-1.9B-Chat，通过监督微调和直接偏好优化从基础模型对齐而来；以及Index-1.9B-Character，它通过检索增强生成增强了聊天模型，实现少样本角色扮演定制。预训练采用Warmup-Stable-Decay学习率调度，其中在衰减阶段显著提高了精选数据的集中度，并配合Norm-Head输出层，在大学习率下稳定训练。在一系列涵盖考试的标准基准测试中...

    arXiv:2607.09885v3 Announce Type: replace  Abstract: We present Index-1.9B, a series of open small language models developed at Bilibili. The series comprises four models: Index-1.9B-Base, a foundation model with 1.9 billion non-embedding parameters pre-trained on 2.8 trillion predominantly Chinese and English tokens; Index-1.9B-Pure, a control variant trained with an identical recipe but with all instruction-like data strictly filtered from the corpus; Index-1.9B-Chat, aligned from the base model with supervised fine-tuning and direct preference optimization; and Index-1.9B-Character, which augments the chat model with retrieval-augmented generation for few-shot role-playing customization. Pre-training employs a Warmup-Stable-Decay learning-rate schedule in which the concentration of curated data is raised substantially during the decay phase, together with a Norm-Head output layer that stabilizes training under large learning rates. On a suite of standard benchmarks covering examinat
    
[^102]: Know2Guess：一个用于大型语言模型知识边界评估的、具有污染意识的多区域基准

    Know2Guess: A Contamination-Aware Multi-Zone Benchmark for Knowledge-Boundary Evaluation in Large Language Models

    [https://arxiv.org/abs/2606.26101](https://arxiv.org/abs/2606.26101)

    提出了一个包含1200个条目、覆盖五个领域的污染感知多区域基准，用于评估大语言模型在知识边界上从有依据回答到应放弃未知的过渡能力，并发现指令调优模型存在选择性但不完全的过渡。

    

    arXiv:2606.26101v1 公告类型：交叉摘要：对大型语言模型的可靠评估应将有依据的回答与无依据的猜测区分开来，同时避免与数据污染、提示特异性或通用拒绝行为相混淆。我们提出了一个具有污染意识的多区域基准，用于衡量在冻结的构建时标签下，从可回答的知识到应放弃的未知领域的过渡。该基准包含跨五个领域的1200个条目、明确的放弃预期、污染风险元数据，以及一个官方严格解析器加一个标准化鲁棒性解析器的双重解析。我们在锁定回答或放弃的提示、仅回答的控制和提示模板变体下评估了FLAN-T5、Qwen2.5-Instruct和Llama-3-Instruct模型。该基准并非通过通用的非回答行为来解决：FLAN基线在有效的放弃方面仍然薄弱，而更强的指令调优模型则暴露出从知识到未知的选择性但不完全的过渡。

    arXiv:2606.26101v1 Announce Type: cross  Abstract: Reliable evaluation of large language models should separate supported answering from unsupported guessing without conflating either with data contamination, prompt idiosyncrasy, or generic refusal behavior. We present a contamination-aware, multi-zone benchmark for measuring the transition from answerable knowledge to abstention-expected unknowns under frozen build-time labels. The benchmark contains 1,200 items across five domains, explicit abstention expectations, contamination-risk metadata, and dual parsing with an official strict parser plus a normalized robustness parser. We evaluate FLAN-T5, Qwen2.5-Instruct, and Llama-3-Instruct models under locked answer-or-abstain prompts, answer-only controls, and prompt-template variants. The benchmark is not solved by generic non-answer behavior: FLAN baselines remain weak on productive abstention, while stronger instruction-tuned models expose a selective but incomplete transition from a
    
[^103]: 元名游戏：一个无真实基准的LLM基准，随其测量的模型一同提升

    The Metanym Game: An LLM Benchmark Without Ground Truth That Rises With the Models It Measures

    [https://arxiv.org/abs/2606.21008](https://arxiv.org/abs/2606.21008)

    该论文提出一种无真实基准的LLM评估方法，通过类比生成与相互评分，利用SVD特征方程统一评判生成与评判能力，并发现与GPQA Diamond存在相关性。

    

    arXiv:2606.21008v3 公告类型：替换交叉 摘要：我们提出证据表明，类比是LLM智能的核心。在我们的基准测试中，LLMs竞争生成一组组类比陈述，并根据各自对事实正确性、美感、智能性、独特性、长度和结构多样性的理解来相互评分。外部信息不进入：唯一给定的是游戏规则；每个项目都在游戏中生成；分数仅来自玩家的评分。真实基准被事实评分矩阵的奇异值分解（SVD）所取代，该矩阵同时将玩家评为生成者和评判者——据我们所知，这是首个评判LLM同行委员会中评判者的特征方程。对于美感等主观标准，评判者按其评分一致性加权。最佳生成者结果是中等评判者。GPQA Diamond——由人类专家编写的困难选择题——在方法上截然不同，但这两个基准却相关。

    arXiv:2606.21008v3 Announce Type: replace-cross  Abstract: We present evidence that analogy is at the core of LLM intelligence. In our benchmark, LLMs compete in generating sets of analogous statements and rate each other's sets on their own understandings of factual correctness, beauty, intelligence, distinctness, length, and structural diversity. Nothing enters from outside: the only given is the game rules; every item is generated in play; the scores come from the players' ratings alone. Ground truth is replaced by the SVD of the factual rating matrix, which scores players as generators and judges at once -- to our knowledge the first eigen-equation that judges the judges for an LLM council-of-peers. For subjective criteria like beauty, judges are weighted by their rating consistency. The best generators turn out to be middling judges. GPQA Diamond -- difficult multiple-choice questions written by human experts -- could not be more different in method, yet the two benchmarks correla
    
[^104]: 检测代码语言模型中的功能记忆化

    Detecting Functional Memorization in Code Language Models

    [https://arxiv.org/abs/2606.12764](https://arxiv.org/abs/2606.12764)

    本文提出了一种通过AI编码代理生成测试输入来检测代码语言模型中功能记忆化的方法，该记忆化在文本审计中不可见，通过反事实框架对比目标模型与参考模型实现功能等价性检测。

    

    大型语言模型（LLMs）越来越多地被用于大规模生成代码。与此同时，先前的研究通过审查训练示例与模型生成之间的文本重叠，探讨了训练数据是否可以从模型输出中恢复。然而，代码可以在保持相同逻辑的同时在语法和结构上有显著差异。我们在此研究功能记忆化：即从LLM生成中泄漏训练数据逻辑，而文本审计无法检测到的方式。我们利用AI编码代理为训练数据功能生成多样化的测试输入，并评估模型生成的续写是否产生相同输出。我们通过一个反事实框架对此进行形式化，将目标模型（接触特定代码）与参考模型（未接触）进行比较，仅要求目标模型具有功能等价性。我们在4个开源模型上实例化该框架，并明确...

    arXiv:2606.12764v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used to generate code at scale. Meanwhile, prior work has investigated whether training data may be recoverable from model outputs, by auditing the textual overlap between training examples and model generations. Code, however, can preserve the same logic while differing substantially in syntax and structure. We here study functional memorization: the leakage of training data logic from LLM generations in ways that textual audits fail to detect. We leverage AI coding agents to generate diverse test inputs for training data functionality and evaluate whether model-generated continuations produce the same outputs. We formalize this through a counterfactual framework, comparing target models (exposed to specific code) against reference models (not exposed) and requiring functional equivalence only for the target. We instantiate this framework across 4 open-source models and explicitly 
    
[^105]: INFUSER：影响力引导的自我进化提升推理能力

    INFUSER: Influence-Guided Self-Evolution Improves Reasoning

    [https://arxiv.org/abs/2606.09052](https://arxiv.org/abs/2606.09052)

    INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。

    

    自我进化为增强推理能力提供了一条可扩展的路径：预训练语言模型仅需极少的外部监督即可自我提升。然而，现有方法要么依赖大量精心策划或教师生成的训练数据，要么在生成器无监督运行时，仅通过难度启发式给予奖励，这未必能改进求解器。我们引入了INFUSER，一种迭代协同训练框架，包含两个共同演化的角色：一个生成器，从自动收集的非结构化文档池中起草问题和参考标准答案；以及一个求解器，通过在这些问题上训练来改进自身。求解器使用标准正确性奖励，依据生成器提供的答案进行训练，而生成器则通过一个优化器感知的影响力分数获得奖励，该分数衡量每个提议的问题是否真正能提升求解器在目标分布上的表现。由于这种连续且嘈杂的影响力分数难以直接处理，我们采用了相应策略进行优化。

    arXiv:2606.09052v4 Announce Type: replace-cross  Abstract: Self-evolution offers a scalable path to stronger reasoning: a pretrained language model improves itself with only minimal external supervision. Yet existing methods either depend on extensively curated or teacher-generated training data, or, when the generator runs unsupervised, reward it by a difficulty heuristic that need not improve the solver. We introduce INFUSER, an iterative co-training framework with two co-evolving roles: a Generator that drafts questions and reference golden answers from a pool of unstructured, automatically collected documents, and a Solver that improves by training on them. The solver is trained with standard correctness rewards against the generator-provided answers, while the generator is rewarded by an optimizer-aware influence score that measures whether each proposed question would actually improve the solver on the target distribution. Because this continuous, noisy influence score is poorly 
    
[^106]: 音频交互模型

    Audio Interaction Model

    [https://arxiv.org/abs/2606.05121](https://arxiv.org/abs/2606.05121)

    本文提出了一种始终在线的音频交互模型，通过感知-决策-响应范式实现不中断监听的主动干预，并配套构建了大规模流式数据与基准，在主流任务上保持性能的同时支持长流交互和主动响应。

    

    音频是连续且交互式的，然而大多数大型音频语言模型（LALMs）仍处于离线状态，而流式系统通常专注于自动语音识别（ASR）或口语对话。我们形式化了音频交互模型，这是一种始终在线的感知-决策-响应范式，它跟踪上下文，判断是否需要干预，并在不停止监听的情况下做出响应。我们通过音频交互实例化该模型，并引入了SoundFlow，结合了流式原生数据构建、理解感知的静音/响应监督、双损失训练以及异步FIFO推理。我们还构建了StreamAudio-2M数据集，这是一个包含260万条目、302,000小时的语料库，涵盖7个能力家族和28个子任务，以及Proactive-Sound-Bench基准。在8个基准测试中，音频交互模型在主流音频任务上保持竞争力，同时支持口语指令鲁棒性、长流交互和主动干预。

    arXiv:2606.05121v2 Announce Type: replace-cross  Abstract: Audio is continuous and interactive, yet most Large Audio Language Models (LALMs) remain offline and streaming systems usually specialize in ASR or spoken dialogue. We formalize the Audio Interaction Model, an always-on perceive--decide--respond paradigm that tracks context, decides whether intervention is warranted, and responds without stopping listening. We instantiate it with Audio-Interaction and introduce SoundFlow, coupling streaming-native data construction, comprehension-aware silence/response supervision, dual-loss training, and asynchronous FIFO inference. We also construct textsc{StreamAudio-2M, a 2.6M-item, 302k-hour corpus spanning 7 capability families and 28 sub-tasks, together with Proactive-Sound-Bench. Across 8 benchmarks, Audio-Interaction remains competitive on mainstream audio tasks while enabling spoken-instruction robustness, long-stream interaction, and proactive intervention.
    
[^107]: SafeSteer：面向高效安全对齐的局部化策略蒸馏

    SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment

    [https://arxiv.org/abs/2606.02530](https://arxiv.org/abs/2606.02530)

    SafeSteer通过将安全对齐限制在稀疏的安全令牌上进行局部策略蒸馏，有效降低了对齐税，同时提升了安全性与通用能力的权衡。

    

    摘要：arXiv:2606.02530v2 公告类型：替换 摘要：将大型语言模型（LLMs）与人类价值观对齐通常会降低其通用能力，这被称为对齐税。现有方法通过平衡双重目标来缓解这一问题，但这严重依赖于海量的通用数据或辅助奖励模型。在本文中，我们认为，由于安全特征在输出分布中本质上是稀疏的，对齐需要局部修改而非全局权衡。为此，我们提出了SafeSteer，它执行仅限于安全令牌的策略蒸馏。首先，我们通过激活引导构建一个安全教师模型。基于该教师模型，我们开发了一种安全令牌选择算法。因此，SafeSteer在训练过程中将反向KL惩罚限制在这些令牌上，以保留通用能力。跨多种模型的实验结果表明，我们的SafeSteer在安全性与通用能力之间实现了更优的权衡。

    arXiv:2606.02530v2 Announce Type: replace  Abstract: Aligning Large Language Models (LLMs) with human values often degrades their general capabilities, termed the alignment tax. Existing methods mitigate this by balancing dual objectives, which heavily rely on massive general-purpose data or auxiliary reward models.   In this paper, we argue that, because safety features are inherently sparse within the output distribution, alignment requires localized modifications rather than global trade-offs. To this end, we propose SafeSteer, which performs on-policy distillation confined to safety tokens. First, we construct a safety teacher via activation steering. Based on this teacher, we develop a safety token selection algorithm. Consequently, SafeSteer restricts the reverse KL penalty to these tokens during training to preserve general capabilities.   Experimental results across diverse models show that our SafeSteer achieves a superior trade-off between safety and general capability compar
    
[^108]: 自修正的科学发现系统：面向智能体人工智能的范畴论框架

    Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence

    [https://arxiv.org/abs/2606.01444](https://arxiv.org/abs/2606.01444)

    本文提出一个基于范畴论的框架，通过左Kan扩展和体制转换来区分检索、搜索与科学发现，实现不依赖主观新颖性的自修正智能体系统。

    

    科学发现不仅仅是生成答案，更是对表征体制的修正，其中证据、人工产物、操作和验证器都被类型化。我们为材料科学中的智能体发现开发了一个范畴论描述。在具有模式范畴S_b的固定体制b中，系统状态是一个余预层I_t: S_b -> Set，来源是元素范畴\int_{S_b} I_t。固定体制操作是对此类状态的更新，只有在指定并保留来源保持的细化时才是自函子性的。相反，发现是一种验证过的体制转换u: S_b -> S_b'：旧的人工产物被保留，通过左Kan扩展Lan_u I_t传输，并与转换后的状态进行比较，以识别超出函子传输的残余内容。这在不依赖主观新颖性的情况下区分了检索、搜索和发现。我们在两个系统中实例化了该框架。在Builder/Breaker中，一个蛋白质力学系统...

    arXiv:2606.01444v2 Announce Type: replace  Abstract: Scientific discovery is not only answer generation but revision of the representational regime in which evidence, artifacts, operations, and verifiers are typed. We develop a category-theoretic account of agentic discovery for materials science. In a fixed regime b with schema category S_b, the system state is a copresheaf I_t: S_b -> Set, and provenance is the category of elements \int_{S_b} I_t. Fixed-regime operation is an update on such states, endofunctorial only when provenance-preserving refinements are specified and preserved. Discovery is instead a verified regime transition u: S_b -> S_b': old artifacts are preserved, transported by the left Kan extension Lan_u I_t, and compared with the post-transition state to identify residual content beyond functorial transport. This separates retrieval, search, and discovery without subjective novelty. We instantiate the framework in two systems. In Builder/Breaker, a protein-mechanics
    
[^109]: GRASP：用于自我改进LLM智能体的门控回归感知技能提议器

    GRASP: Gated Regression-Aware Skill Proposer for Self-Improving LLM Agents

    [https://arxiv.org/abs/2605.29668](https://arxiv.org/abs/2605.29668)

    GRASP提出了一种门控回归感知技能提议机制，通过硬性回归预算和平衡探针确保技能库每次编辑只带来净改进，在临床基准上将LLM智能体性能大幅提升。

    

    摘要：arXiv:2605.29668v2 公告类型：替换 摘要：在结构化环境中运行的LLM智能体失败方式更多是操作性的而非对话性的，其可靠性依赖于对环境的程序性知识。先前的自我改进方法积累自然语言指导，但不检查每个新条目是否保留先前正确的行为，因此修复一条轨迹的注释可能静默地使另一条轨迹退化。我们引入了GRASP（门控回归感知技能提议器），它将智能体改进视为对有限技能库的一系列编辑，只有在硬性回归预算下，每个候选者在平衡的保留探针上产生净改进时才接受它。我们在两个基于FHIR的临床基准上评估了GRASP，覆盖五个基础模型，这些基准根据FHIR状态评分程序性可靠性，而非临床正确性或患者结局。在MedAgentBench上，GRASP将gpt-oss-120b从40.6%提升至88.8%，超过了五种自我改进基线中最强的。

    arXiv:2605.29668v2 Announce Type: replace  Abstract: LLM agents acting in structured environments fail in operational rather than conversational ways, and reliability depends on procedural knowledge of the environment. Prior self-improvement methods accumulate natural-language guidance without checking that each new item preserves previously correct behavior, so a note that fixes one trajectory can silently regress another. We introduce GRASP (Gated Regression-Aware Skill Proposer), which treats agent improvement as a sequence of edits to a bounded skill library, admitting each candidate only if it produces a net improvement on a balanced held-out probe under a hard regression budget. We evaluate GRASP across five base models on two FHIR-based clinical benchmarks, which score procedural reliability against FHIR state rather than clinical correctness or patient outcomes. On MedAgentBench, GRASP lifts gpt-oss-120b from 40.6% to 88.8%, exceeds the strongest of five self-improvement baseli
    
[^110]: 迷失在采样中：通过词汇覆盖率得分（WCS）评估大语言模型的词汇可达性

    Lost in Sampling: Assessing Lexical Reachability in LLMs via the Word Coverage Score (WCS)

    [https://arxiv.org/abs/2605.27268](https://arxiv.org/abs/2605.27268)

    本文提出词汇覆盖率得分（WCS），首次定量揭示标准采样过滤器如何从数学上抑制大语言模型对低频、高信息量人类词汇的生成可达性，从而量化解码机制对语言多样性的限制。

    

    arXiv:2605.27268v2 公告类型：替换-交叉 摘要：现代大语言模型（LLMs）常因生成重复且同质化的文本而受到批评，尽管它们拥有庞大的潜在词汇库。以往研究侧重于模型知识和训练数据，而我们则探讨解码机制在抑制语言多样性方面的作用。我们引入了词汇覆盖率得分（WCS），这是一种量化标准采样过滤器（如Top-$p$、Top-$k$和Min-$p$）在数学上修剪掉上下文适当的人类词汇程度的指标。WCS不评估静态知识，而是测量低频、高信息量的人类词汇的存活率，作为采样参数的函数。通过审计开放权重模型在人类撰写的语料片段上的表现，我们识别出哪些逻辑上的词汇选择被解码器变得不可达，即使它们存在于概率空间中。我们的结果提供了定量证据。

    arXiv:2605.27268v2 Announce Type: replace-cross  Abstract: Modern Large Language Models (LLMs) are often criticized for producing repetitive and homogeneous text, despite possessing vast latent vocabularies. While previous research has focused on model knowledge and training data, we investigate the role of decoding mechanics in suppressing linguistic diversity. We introduce the Word Coverage Score (WCS), a metric that quantifies the extent to which contextually appropriate human vocabulary is mathematically pruned by standard sampling filters (e.g., Top-$p$, Top-$k$, and Min-$p$). Rather than assessing static knowledge, the WCS measures the lexical survival rate of low-frequency, high-information human words as a function of sampling parameters. By auditing open-weight models on human-authored corpus fragments, we identify which logical lexical choices are rendered unreachable by the decoder, even when they reside within the probability space. Our results provide quantitative evidence
    
[^111]: Granuscore：一种用于文本分析与问答的无参考粒度度量方法

    Granuscore: A Reference-Free Measure of Granularity for Text Analysis and Question Answering

    [https://arxiv.org/abs/2605.26620](https://arxiv.org/abs/2605.26620)

    本文提出Granuscore，一种基于分层嵌入空间的无参考粒度度量方法，能有效捕捉文本粒度差异，并在问答基准中揭示模型行为的一致模式。

    

    自然语言以不同的粒度水平传达信息，从细粒度的具体指代到宽泛的描述。尽管粒度对人类交流至关重要，现有度量方法大多只捕捉表面细节或句子特定性。我们引入了Granuscore，一种无参考的粒度度量方法，利用分层嵌入空间的结构特性。Granuscore能够在Granola-EQ数据集上可靠地恢复层次顺序，并捕捉不同话语语境中预期的粒度差异。跨领域实验进一步表明，Granuscore能够解释句子长度之外的句子特定性的非线性变化。最后，我们将Granuscore应用于四个问答基准，分析问题、黄金答案和模型输出在不同响应结果中的粒度差异。分析揭示了模型行为的一致性差异，并为特征化模型输出提供了原则性视角。

    arXiv:2605.26620v2 Announce Type: replace  Abstract: Natural language conveys information at varying levels of granularity, from fine-grained references to broad descriptions. While granularity is fundamental to human communication, existing measures mostly capture surface detail or sentence specificity. We introduce Granuscore, a reference-free measure of granularity that leverages structural properties of a hierarchical embedding space. Granuscore reliably recovers hierarchical orderings on the Granola-EQ dataset and captures expected differences in granularity across discourse contexts. Across domains, we further show that Granuscore explains non-linear variation in sentence specificity beyond sentence length. Finally, we apply Granuscore to four question-answering benchmarks and analyze how granularity differs for questions, gold answers, and model outputs across response outcomes. The analysis reveals consistent differences in model behavior and provides a principled lens for char
    
[^112]: RouteScan：一种通过专家路由遥测审计MoE大语言模型安全性的非侵入式方法

    RouteScan: A Non-Intrusive Approach to Auditing MoE LLMs Safety via Expert Routing Telemetry

    [https://arxiv.org/abs/2605.24817](https://arxiv.org/abs/2605.24817)

    本文提出RouteScan，一种通过分析GPU执行中专家路由遥测来非侵入式审计MoE大语言模型安全性的方法，无需访问用户提示或模型内部，从而兼顾安全与隐私。

    

    摘要：随着混合专家（MoE）架构越来越多地被用于扩展大型语言模型（LLM），安全审计变得必要，以验证这些模型在运行过程中是否产生或促进有害行为。然而，现有的基于内容的审计方法通常需要访问用户提示、模型内部或输出，这可能暴露敏感用户信息，并在LLM安全性和用户隐私之间造成紧张关系。另一方面，我们观察到，在MoE模型中，不同输入会引发不同的稀疏专家路由模式，这些模式在低级GPU执行遥测中产生可测量的足迹。我们将这些由专家路由决策引起的硬件可观察信号称为专家路由遥测；它们源自GPU执行，而非路由器logits或令牌级路由分配。受此观察启发，我们提出了RouteScan，一种非侵入式审计方法，它利用专家路由遥测来检测MoE模型中的不安全行为，而无需访问用户提示、模型权重或输出，从而在保持安全性的同时保护用户隐私。

    arXiv:2605.24817v2 Announce Type: replace-cross  Abstract: As Mixture-of-Experts (MoE) architectures are increasingly adopted for scaling Large Language Models (LLMs), safety auditing becomes necessary to verify whether these models produce or facilitate harmful behaviors during operation. However, existing content-based auditing methods typically require access to user prompts, model internals, or outputs, potentially exposing sensitive user information and creating a tension between LLM safety and user privacy. On the other hand, we observe that, in MoE models, different inputs induce different sparse expert-routing patterns, which produce measurable footprints in low-level GPU execution telemetry. We refer to these hardware-observable signals induced by expert-routing decisions as expert routing telemetry; they are derived from GPU execution rather than from router logits or token-level routing assignments. Inspired by this observation, we propose RouteScan, a non-intrusive auditing
    
[^113]: STS：基于投机性令牌稀疏性的高效稀疏注意力机制

    STS: Efficient Sparse Attention with Speculative Token Sparsity

    [https://arxiv.org/abs/2605.15508](https://arxiv.org/abs/2605.15508)

    STS提出了一种利用小型草稿模型预测重要令牌来动态构建稀疏掩码的方法，无需重训练即可在保持精度的同时显著加速LLM推理。

    

    注意力机制的二次复杂度给大型语言模型（LLM）推理带来了严重的内存和计算瓶颈。这一挑战对于需要处理数百万令牌序列的新兴代理应用尤为严峻。我们提出了STS，一种无需模型重新训练的稀疏注意力机制。STS利用了这样一个关键洞察：由较小草稿模型识别为重要的令牌，对于较大目标模型的重要令牌具有高度预测性。通过整合到投机性解码框架中，STS重新利用草稿模型的注意力分数来动态构建令牌和头部特定的稀疏掩码。该掩码有效地剪枝了目标LLM中昂贵的注意力计算。我们的评估表明，在代表性基准NarrativeQA上，STS在约90%稀疏度下实现了2.67倍的加速，与完全注意力相比，准确率下降可忽略不计。

    arXiv:2605.15508v3 Announce Type: replace-cross  Abstract: The quadratic complexity of attention imposes severe memory and computational bottlenecks on Large Language Model (LLM) inference. This challenge is particularly acute for emerging agentic applications that require processing multi-million token sequences. We propose STS, a sparse attention mechanism that requires no model retraining. STS leverages the key insight that tokens identified as important by a smaller draft model are highly predictive of important tokens for a larger target model. By integrating into speculative decoding frameworks, STS repurposes the draft model's attention scores to dynamically construct a token-and-head-wise sparsity mask. This mask effectively prunes the expensive attention computation in the target LLM. Our evaluation shows that STS achieves a 2.67x speedup operating at approximately 90% sparsity on representative benchmark NarrativeQA, maintaining negligible accuracy degradation compared to den
    
[^114]: RefusalGuard：保持几何结构的微调方法以保障大语言模型安全性

    RefusalGuard: Geometry-Preserving Fine-Tuning for Safety in LLMs

    [https://arxiv.org/abs/2605.01913](https://arxiv.org/abs/2605.01913)

    本文揭示了标准微调导致安全对齐退化的表示级机制，并提出了REFUSALGUARD框架，通过保持安全相关表示的几何结构来在微调中维持模型安全性。

    

    arXiv:2605.01913v2 公告类型：替换交叉 摘要：对已进行安全对齐的语言模型进行下游任务微调，往往会导致拒绝行为大幅退化，使模型易受对抗性滥用攻击。尽管先前研究已表明，安全相关特征在模型激活空间中以结构化表示形式编码，但这些表示在微调过程中如何变化以及对齐为何退化，仍知之甚少。在本工作中，我们研究了对齐退化背后的表示级机制。我们的分析表明，标准微调会引发安全相关表示的系统性漂移，扭曲其几何结构，并在任务优化与安全特征之间引入干扰。这些效应共同导致有害顺从性增加。基于这些发现，我们提出了REFUSALGUARD，一种表示级微调框架，在微调过程中保持安全相关结构。

    arXiv:2605.01913v2 Announce Type: replace-cross  Abstract: Fine-tuning safety-aligned language models for downstream tasks often leads to substantial degradation of refusal behavior, making models vulnerable to adversarial misuse. While prior work has shown that safety-relevant features are encoded in structured representations within the model's activation space, how these representations change during fine-tuning and why alignment degrades remains poorly understood. In this work, we investigate the representation-level mechanisms underlying alignment degradation. Our analysis shows that standard fine-tuning induces systematic drift in safety-relevant representations, distorts their geometric structure, and introduces interference between task optimization and safety features. These effects collectively lead to increased harmful compliance. Motivated by these findings, we introduce REFUSALGUARD, a representation-level fine-tuning framework that preserves safety-relevant structure duri
    
[^115]: 与什么相比？反事实提示的基线和度量标准

    Compared to What? Baselines and Metrics for Counterfactual Prompting

    [https://arxiv.org/abs/2605.01048](https://arxiv.org/abs/2605.01048)

    本文指出，反事实提示中观察到的效应常被表面形式变化混淆，需使用基线（如改写）来校正，否则可能错误归因模型敏感性。

    

    arXiv:2605.01048v2 公告类型：替换 摘要：反事实提示（即扰动单一因素并测量输出变化）被广泛用于评估诸如大语言模型偏差和思维链忠实度等事项。但在本工作中，我们认为，如果不考虑建立通用模型敏感性的基线“意义保持”文本修改，观察到的效应就不能归因于目标因素。这是因为每个反事实编辑都是一个复合处理，将感兴趣变量与偶然的表面形式变化捆绑在一起；这违反了处理变化无关性。我们在MedQA上观察到，当手术性地改变患者性别时，预测翻转率为14.9%。然而，这与简单改写输入所引发的翻转率（14.1%）在统计上无法区分。在这种情况下，因此得出大语言模型对患者性别特别敏感的结论是不合理的。为考虑这一点并稳健地测量目标效应。

    arXiv:2605.01048v2 Announce Type: replace  Abstract: Counterfactual prompting (i.e., perturbing a single factor and measuring output change) is widely used to evaluate things like LLM bias and CoT faithfulness. But in this work we argue that observed effects cannot be attributed to the targeted factor without accounting for baseline "meaning-preserving" modifications to text that establish general model sensitivity. This is because every counterfactual edit is a compound treatment that bundles the variable of interest with incidental surface-form variation; this violates treatment variation irrelevance. We observe prediction flip rates on MedQA of 14.9% when we surgically change patient gender. However, this is statistically indistinguishable from the flip rates induced by simply paraphrasing inputs (14.1%). In this case, it would therefore be unwarranted to conclude that the LLM is especially sensitive to patient gender. To account for this and robustly measure the effects of targeted
    
[^116]: 心理健康AI的信任栈：跨人类、交互与AI层的校准综述

    Trust Stack for Mental Health AI: A Survey of Calibration across Human, Interaction, and AI Layers

    [https://arxiv.org/abs/2604.20166](https://arxiv.org/abs/2604.20166)

    本文提出一个三层信任框架，整合人类、交互和AI层面的信任校准，以解决心理健康AI中信任与安全之间的错位问题。

    

    基于语言的AI越来越多地被用于心理健康支持，然而信任的评估在跨学科但操作上不一致的方式中进行：NLP和AI领域的工作衡量鲁棒性、安全性、隐私和解释性，而心理治疗、人机交互和监管领域的工作则强调治疗保真度、生活经验、共情和依赖。共情聊天机器人可以引发用户强烈的信任，但缺乏相应的安全性，而更安全的系统在其边界不透明时则被低估信任，这种校准差距没有任何单一社区能够独自解决。通过对61篇论文的结构化范围综述，我们将这一领域划分为一个三层框架，分别区分（L1）面向人类的信任、（L2）面向交互的可信度和（L3）面向AI的可信度，并将五个利益相关者视角映射到这些层上。我们提出了一个研究议程，用于构建社会技术对齐的可信心理健康支持AI，强调其重要性。

    arXiv:2604.20166v3 Announce Type: replace  Abstract: Language-based AI is increasingly deployed for mental health support, yet trust is evaluated in interdisciplinary but operationally misaligned ways: NLP and AI work measures robustness, safety, privacy, and explanations, while psychotherapy, HCI, and regulatory work emphasize therapeutic fidelity, lived experience, empathy, and reliance. Empathetic chatbots can elicit strong user trust without commensurate safety, while safer systems are under-trusted when their boundaries are opaque, a calibration gap no single community owns. Through a structured scoping synthesis of 61 papers, we survey this landscape into a three-layer framework separating (L1) human-oriented trust, (L2) interaction-oriented trustworthiness, and (L3) AI-oriented trustworthiness, and map five stakeholder perspectives onto these layers. We outline a research agenda for building socio-technically aligned trustworthy AI for mental health support, highlighting that th
    
[^117]: 基于验证数据的强化学习实现人类水平的文本到SQL转换，无需流水线工程

    Human-Level Text-to-SQL via Reinforcement Learning on Verified Data, Without Pipeline Engineering

    [https://arxiv.org/abs/2603.20004](https://arxiv.org/abs/2603.20004)

    本文通过专家驱动的多轮验证流程清洗数据，并利用RLVR微调LLM，在无流水线工程的情况下实现了人类水平的Text-to-SQL性能，突破了现有流水线方法的性能上限。

    

    摘要：将自然语言问题转换为SQL查询（Text-to-SQL）是数据库研究中长期存在的问题。近期研究通过构建日益复杂的多阶段大型语言模型流水线，在LLM之上叠加任务分解、模式链接和基于模型的查询选择，专注于提高准确性。尽管复杂度不断增加，但在基准测试中，此类系统与人类专家之间仍存在显著差距（超过10%），这表明仅靠流水线工程已达到性能上限。我们证明，通过在干净数据上使用RLVR对LLM进行微调，无需流水线组件，即可实现人类水平的Text-to-SQL性能。在本文中，我们识别了RLVR在Text-to-SQL上的主导瓶颈：现有训练数据包含普遍存在的标注错误，这些错误误导了优化过程。为解决这一问题，我们开发了一个多轮、专家驱动的验证流水线，并利用它整理了BIRD-Platinum数据集。

    arXiv:2603.20004v4 Announce Type: replace-cross  Abstract: Translating natural language questions to SQL queries (Text-to-SQL) is a long-standing problem in database research. Recent efforts have focused on improving accuracy by building increasingly complex multi-stage large LLM pipelines, layering task decomposition, schema linking, and model-based query selection on top of an LLM. Despite this growing complexity, a substantial gap (>10%) between such systems and human experts persists on benchmarks, suggesting that pipeline engineering alone has hit a ceiling.   We show that human-level Text-to-SQL performance is achievable by fine-tuning an LLM using RLVR on clean data, without pipeline components. In this paper, we identified the dominant bottleneck for RLVR on Text-to-SQL: existing training data contains pervasive annotation errors that mislead optimization. To address this, we developed a multi-round, expert-driven verification pipeline and used it to curate BIRD-Platinum, a dat
    
[^118]: 扩散语言模型通过序列重生成的高效自我评估方法

    Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration

    [https://arxiv.org/abs/2603.02760](https://arxiv.org/abs/2603.02760)

    本文提出了一种基于序列重生成概率的扩散语言模型自我评估方法DiSE，并引入灵活长度生成框架，以提升质量评估的效率和可靠性。

    

    扩散大语言模型（dLLMs）因其增强多样性、可控性和并行性的能力而近期受到广泛关注。然而，其非顺序、双向掩码的生成方式使得质量评估变得困难，凸显了有效自我评估的必要性。在本工作中，我们提出了DiSE，一种简单而有效的dLLMs自我评估置信度量化方法。DiSE通过计算在给定完整上下文的情况下，重新生成整个生成序列中所有标记的概率来量化置信度。该方法利用标记重生成概率，实现了更高效可靠的质量评估，同时促进了似然估计和稳健的不确定性量化。基于DiSE，我们进一步引入了一种灵活长度的生成框架，该框架根据模型对自身输出的自我评估自适应地控制序列长度。

    arXiv:2603.02760v2 Announce Type: replace-cross  Abstract: Diffusion large language models (dLLMs) have recently attracted significant attention for their ability to enhance diversity, controllability, and parallelism. However, their non-sequential, bidirectionally masked generation makes quality assessment difficult, underscoring the need for effective self-evaluation. In this work, we propose DiSE, a simple yet effective self-evaluation confidence quantification method for dLLMs. DiSE quantifies confidence by computing the probability of regenerating the tokens in the entire generated sequence, given the full context. This method enables more efficient and reliable quality assessment by leveraging token regeneration probabilities, facilitating both likelihood estimation and robust uncertainty quantification. Building upon DiSE, we further introduce a flexible-length generation framework, which adaptively controls the sequence length based on the model's self-assessment of its own out
    
[^119]: 注意风格：沟通风格对人机对话交互的影响

    Mind the Style: Impact of Communication Style on Human-Chatbot Interaction

    [https://arxiv.org/abs/2602.17850](https://arxiv.org/abs/2602.17850)

    本研究发现，友好型沟通风格的聊天机器人在提升用户满意度和任务成功率方面优于直接型风格，但无聊天机器人的控制条件在任务成功率上表现最佳。

    

    arXiv:2602.17850v2 公告类型：交叉替换 摘要：对话代理日益介导日常数字交互，但其沟通风格对用户体验和任务成功的影响仍未得到充分理解。针对这一空白，我们报告了一项受试者间用户研究，参与者与名为NAVI的聊天机器人的两个版本之一进行交互，该机器人协助他们完成基于交互式地图的2D导航任务。两个聊天机器人版本主要在设计上差异于沟通风格：一个使用友好和支持性的语气，而另一个使用直接和任务导向的语气。我们还包含了一个控制条件，其中参与者不与聊天机器人交互，但接收逐步导航指令。友好型聊天机器人显著提高了用户的沟通满意度，并与直接型聊天机器人相比，与更高的任务成功率相关。然而，控制条件下的参与者取得了最高的任务成功率。

    arXiv:2602.17850v2 Announce Type: replace-cross  Abstract: Conversational agents increasingly mediate everyday digital interactions, yet the effects of their communication style on user experience and task success remain insufficiently understood. Addressing this gap, we report a between-subject user study in which participants interacted with one of two versions of a chatbot called NAVI, which assisted them in an interactive map-based 2D navigation task. The two chatbot versions were designed to differ primarily in communication style: one used a friendly and supportive tone, while the other used a direct and task-focused tone. We also included a control condition where participants did not interact with a chatbot but received the step-by-step navigation instructions. The friendly chatbot significantly increased users' communication satisfaction and was associated with higher task success than the direct chatbot. However, participants in the control condition achieved the highest task
    
[^120]: 当外表不撒谎：基于话语结构引导的上下文学习用于忠实图表生成

    When Looks Do Not Lie: Discourse Structure Guided In-Context Learning for Faithful Diagram Generation

    [https://arxiv.org/abs/2601.20476](https://arxiv.org/abs/2601.20476)

    本文提出一种基于修辞结构理论的上下文学习图表生成方法，显著提升图表对源文本的忠实度，并通过专家与自动评估验证其有效性。

    

    arXiv:2601.20476v2 公告类型：替换 摘要：生成式人工智能在教育应用中广泛使用；然而，已知其会生成含有内在和外在幻觉的内容。我们提出了一种基于修辞结构理论的新型上下文学习图表生成方法，该方法提高了图表对其源文本上下文的忠实度。我们发现，上下文学习性能取决于任务分布和模型的推理能力，更高的推理能力能在分布外任务中带来更好的质量和性能。我们对150个生成的图表进行了专家评估，并使用贝叶斯广义线性混合模型分析了我们的发现。此外，我们利用评估标准和数据集样本进行自动图表评估，与人工评估取得了统计学上显著的一致性。

    arXiv:2601.20476v2 Announce Type: replace  Abstract: GenAI is widespread in educational applications; however, it is known to generate content with intrinsic and extrinsic hallucination. We introduce a novel method for ICL diagram generation based on Rhetorical Structure Theory, which improves diagram faithfulness to its source text context. We find that ICL performance depends on task distribution and models' reasoning ability, with higher reasoning allowing better quality and performance for an out-of-distribution task. We perform an expert evaluation of 150 generated diagrams and analyze our findings using Bayesian GLMMs. Additionally, we use our evaluation rubric and samples from the data set for automated diagram evaluation, achieving statistically significant agreement with human evaluation.
    
[^121]: SlidesGen-Bench：通过计算与定量指标评估幻灯片生成

    SlidesGen-Bench: Evaluating Slides Generation via Computational and Quantitative Metrics

    [https://arxiv.org/abs/2601.09487](https://arxiv.org/abs/2601.09487)

    本文提出了SlidesGen-Bench基准，通过视觉领域的统一框架和内容、美学、可编辑性三个计算指标，实现了跨架构的定量且可靠的幻灯片生成评估。

    

    大型语言模型（LLMs）的快速发展催生了多样化的自动幻灯片生成范式，从代码驱动的布局到以图像为中心的合成。然而，评估这些异构系统仍然具有挑战性，因为现有协议往往难以在不同架构间提供可比较的分数，或依赖于未经校准的判断。在本文中，我们引入了SlidesGen-Bench，一个通过三个核心原则——普遍性、定量性和可靠性——来评估幻灯片生成的基准。首先，为建立统一的评估框架，我们将分析基础置于视觉领域，将终端输出视为渲染结果，以保持对底层生成方法的不可知性。其次，我们提出了一种计算方法，从三个不同维度——内容、美学和可编辑性——定量评估幻灯片，提供了可复现的指标，而先前的工作在这方面有所欠缺。

    arXiv:2601.09487v2 Announce Type: replace  Abstract: The rapid evolution of Large Language Models (LLMs) has fostered diverse paradigms for automated slide generation, ranging from code-driven layouts to image-centric synthesis. However, evaluating these heterogeneous systems remains challenging, as existing protocols often struggle to provide comparable scores across architectures or rely on uncalibrated judgments. In this paper, we introduce SlidesGen-Bench, a benchmark designed to evaluate slide generation through a lens of three core principles: universality, quantification, and reliability. First, to establish a unified evaluation framework, we ground our analysis in the visual domain, treating terminal outputs as renderings to remain agnostic to the underlying generation method. Second, we propose a computational approach that quantitatively assesses slides across three distinct dimensions - Content, Aesthetics, and Editability - offering reproducible metrics where prior works re
    
[^122]: MedRAGChecker：面向生物医学检索增强生成的主张级验证

    MedRAGChecker: Claim-Level Verification for Biomedical Retrieval-Augmented Generation

    [https://arxiv.org/abs/2601.06519](https://arxiv.org/abs/2601.06519)

    MedRAGChecker提出了一种主张级验证框架，通过结合证据NLI和知识图谱一致性信号，对生物医学RAG的生成回答进行细粒度诊断，以区分检索与生成失败并识别安全关键错误。

    

    生物医学检索增强生成（RAG）可以将大语言模型的回答基于医学文献，但长格式输出中常常包含孤立的不受支持或相互矛盾的主张，这些主张可能带来安全影响。我们提出了MedRAGChecker，一个面向生物医学RAG的主张级验证与诊断框架。给定一个问题、检索到的证据和生成的回答，MedRAGChecker将回答分解为原子主张，并通过结合基于证据的自然语言推理（NLI）和生物医学知识图谱（KG）一致性信号来估计主张支持度。聚合主张决策可产生回答级诊断，有助于厘清检索和生成失败，包括忠实度、证据不足、矛盾和安全关键错误率。为支持可扩展评估，我们将流程蒸馏为紧凑的生物医学模型，并使用带有类别特定可靠性权重的集成验证器。实验...

    arXiv:2601.06519v2 Announce Type: replace  Abstract: Biomedical retrieval-augmented generation (RAG) can ground LLM answers in medical literature, yet long-form outputs often contain isolated unsupported or contradictory claims with safety implications.   We introduce MedRAGChecker, a claim-level verification and diagnostic framework for biomedical RAG.   Given a question, retrieved evidence, and a generated answer, MedRAGChecker decomposes the answer into atomic claims and estimates claim support by combining evidence-grounded natural language inference (NLI) with biomedical knowledge-graph (KG) consistency signals.   Aggregating claim decisions yields answer-level diagnostics that help disentangle retrieval and generation failures, including faithfulness, under-evidence, contradiction, and safety-critical error rates.   To enable scalable evaluation, we distill the pipeline into compact biomedical models and use an ensemble verifier with class-specific reliability weighting.   Experi
    
[^123]: 何时深思：通过测试时训练为代码生成实现自适应计算分配

    When to Ponder: Adaptive Compute Allocation for Code Generation via Test-Time Training

    [https://arxiv.org/abs/2601.00894](https://arxiv.org/abs/2601.00894)

    本文提出PonderTTT，一种无需训练的门控策略，通过TTT层的重建损失自适应触发测试时训练，在代码生成中实现高效计算分配，显著提升推理性能。

    

    arXiv:2601.00894v2 公告类型：替换-交叉 摘要：大型语言模型对所有输入施加统一的计算量，而不考虑其难度。我们提出PonderTTT，一种使用TTT层的自监督重建损失来选择性地触发测试时训练（TTT）更新的门控策略。该门控决策本身无需训练——不需要学习分类器或辅助网络；仅需在无标签数据上初步校准一个标量阈值，并通过指数移动平均（EMA）持续调整以维持目标更新率。我们在GPT-2模型（124M至1.5B参数）上的代码语言建模实验（The Stack v2，教师强制困惑度）表明，该信号与推理兼容，无需真实标签。我们的重建门控实现了82-89%的Oracle恢复率，同时完全无需训练，显著优于随机跳过基线（在OOD语言上损失降低高达16%）。

    arXiv:2601.00894v2 Announce Type: replace-cross  Abstract: Large language models apply uniform computation to all inputs, regardless of difficulty. We propose PonderTTT, a gating strategy using the TTT layer's self-supervised reconstruction loss to selectively trigger Test-Time Training (TTT) updates. The gating decision itself is training-free--requiring no learned classifier or auxiliary networks; only a single scalar threshold is initially calibrated on unlabeled data and continuously adapted via EMA to maintain target update rates. Our experiments with GPT-2 models (124M to 1.5B) on code language modeling (The Stack v2, teacher-forced perplexity) demonstrate that this signal is inference-compatible, requiring no ground-truth labels. Our Reconstruction Gating achieves 82-89% Oracle Recovery while being fully training-free, significantly outperforming Random Skip baselines (up to 16% lower loss on OOD languages).
    
[^124]: StruProKGR：一种用于稀疏知识图谱推理的结构与概率框架

    StruProKGR: A Structural and Probabilistic Framework for Sparse Knowledge Graph Reasoning

    [https://arxiv.org/abs/2512.12613](https://arxiv.org/abs/2512.12613)

    提出了一种基于距离引导路径收集和结构感知概率建模的框架，解决了稀疏知识图谱推理中路径质量低和结构信息利用不足的问题。

    

    稀疏知识图谱（KGs）在现实应用中经常遇到，其中知识往往不完整或有限。稀疏知识图谱推理，即在稀疏知识图谱上推断缺失知识的任务，由于知识稀缺以及在稀疏场景中难以捕获关系模式而固有地具有挑战性。在所有稀疏知识图谱推理方法中，基于路径的方法因其可解释性而受到广泛关注。现有的基于路径的方法通常依赖计算密集的随机游走来收集路径，产生质量可变的路径。此外，这些方法未能利用图的结构特性，将路径独立处理。为解决这些不足，我们提出了一种名为StruProKGR的结构与概率框架，专为稀疏知识图谱上的高效且可解释推理而设计。StruProKGR利用距离引导的路径收集机制，以增强路径质量和结构利用。

    arXiv:2512.12613v2 Announce Type: replace  Abstract: Sparse Knowledge Graphs (KGs) are commonly encountered in real-world applications, where knowledge is often incomplete or limited. Sparse KG reasoning, the task of inferring missing knowledge over sparse KGs, is inherently challenging due to the scarcity of knowledge and the difficulty of capturing relational patterns in sparse scenarios. Among all sparse KG reasoning methods, path-based ones have attracted plenty of attention due to their interpretability. Existing path-based methods typically rely on computationally intensive random walks to collect paths, producing paths of variable quality. Additionally, these methods fail to leverage the structured nature of graphs by treating paths independently. To address these shortcomings, we propose a Structural and Probabilistic framework named StruProKGR, tailored for efficient and interpretable reasoning on sparse KGs. StruProKGR utilizes a distance-guided path collection mechanism to s
    
[^125]: 氛围编程安全吗？真实世界任务中智能体生成代码漏洞的基准评估

    Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks

    [https://arxiv.org/abs/2512.03262](https://arxiv.org/abs/2512.03262)

    该论文提出SUSVIBES基准，评估了12种编码智能体在真实任务中的安全性，发现所有智能体生成代码的安全率极低（最高仅11.8%），且简单安全提示无法有效改善。

    

    arXiv:2512.03262v3 公告类型：交叉替换 摘要：氛围编程是一种新的软件开发范式，在这种范式中，人类工程师提示大型语言模型（LLM）智能体在极少监督下完成复杂的编码任务。尽管氛围编程日益被采用，但生成的代码在生产环境中部署真的安全吗？为了探究这一问题，我们提出了SUSVIBES基准，该基准包含来自真实世界开源项目的186个功能请求软件工程任务，针对这些任务，人类程序员提交了存在漏洞的实现。我们在该基准上评估了12种广泛使用的编码智能体设置，并采用了前沿模型。令人不安的是，所有智能体在软件安全方面表现不佳。尽管来自SWE-Agent与Claude 4 Sonnet的解决方案中57%在功能上正确，但只有11.8%是安全的。进一步实验表明，初步安全策略，例如在功能请求中添加漏洞提示，无法缓解这些问题。

    arXiv:2512.03262v3 Announce Type: replace-cross  Abstract: Vibe coding is a new software development paradigm in which human engineers prompt a large language model (LLM) agent to complete complex coding tasks with little supervision. Although vibe coding is increasingly adopted, is the generated code really safe to deploy in production? To investigate this question, we propose SUSVIBES, a benchmark consisting of 186 feature-request software engineering tasks from real-world open-source projects, for which, human programmers committed vulnerable implementations. We evaluate 12 widely used coding agentic settings with frontier models on the benchmark. Disturbingly, all agents perform poorly in terms of software security. Although 57% of the solutions from SWE-Agent with Claude 4 Sonnet are functionally correct, only 11.8% are secure. Further experiments demonstrate that preliminary security strategies, such as augmenting the feature request with vulnerability hints, cannot mitigate thes
    
[^126]: 当更好的教师不培养出更好的学生：重新审视VQA中CLIP模型的知识蒸馏

    When Better Teachers Don't Make Better Students: Revisiting Knowledge Distillation for CLIP Models in VQA

    [https://arxiv.org/abs/2511.17886](https://arxiv.org/abs/2511.17886)

    本文首次系统研究发现，在CLIP模型的知识蒸馏中，更强的教师并不总能带来更好的学生，现有蒸馏框架在扩展时反而导致VQA等下游任务性能下降。

    

    视觉语言模型（VLMs）在多模态任务中取得了显著成功，但其巨大的计算需求阻碍了高效部署。知识蒸馏（KD）已成为构建轻量级但具有竞争力模型的强大方法，在语言和视觉领域均有充分证据支持。然而，其在VLMs（特别是CLIP风格模型）中的应用仍然有限，通常局限于小规模教师模型和狭窄的评估任务（如分类或检索）。在本工作中，我们首次系统研究了跨一系列CLIP风格教师模型的蒸馏，范围从标准基线到大规模最先进模型。与NLP和视觉领域观察到的趋势相反，我们发现更强的教师并不一致地产生更好的学生；事实上，现有蒸馏框架往往难以扩展，导致下游多模态任务性能下降。

    arXiv:2511.17886v2 Announce Type: replace-cross  Abstract: Vision-language models (VLMs) have achieved remarkable success across multimodal tasks, yet their substantial computational demands hinder efficient deployment. Knowledge distillation (KD) has emerged as a powerful approach for building lightweight but competitive models, with strong evidence from both language and vision domains. However, its application to VLMs, particularly CLIP-style models, remains limited, often constrained to small-scale teachers and narrow evaluation tasks such as classification or retrieval. In this work, we present the first systematic study of distillation across a range of CLIP-style teacher models, ranging from standard baselines to large-scale state-of-the-art models. Contrary to trends observed in NLP and vision, we find that stronger teachers do not consistently yield better students; in fact, existing distillation frameworks often fail to scale, leading to degraded performance in downstream mul
    
[^127]: LTR-ICD：一种面向自动ICD编码的排序感知框架

    LTR-ICD: A Ranking-Aware Framework for Automatic ICD Coding

    [https://arxiv.org/abs/2510.13922](https://arxiv.org/abs/2510.13922)

    本文首次将ICD编码问题从检索视角重新定义为分类与排序任务，提出排序感知框架，显著提升了高优先级诊断代码的识别与排序准确性。

    

    临床笔记包含临床医生在患者就诊期间提供的非结构化文本。这些笔记通常伴随一系列遵循国际疾病分类（ICD）的诊断代码。正确分配和排序ICD代码对于医疗诊断和报销至关重要。然而，自动化此任务仍具挑战性。最先进的方法将此问题视为分类任务，导致忽略了ICD代码的顺序，而顺序对不同目的至关重要。在本工作中，作为首次尝试，我们从检索系统的视角处理此任务，以考虑代码顺序，从而将此问题表述为分类和排序任务。我们的结果和分析表明，所提出的框架在识别高优先级代码方面优于其他方法。例如，我们的模型在正确排序主要诊断代码方面的准确性更高。

    arXiv:2510.13922v2 Announce Type: replace-cross  Abstract: Clinical notes contain unstructured text provided by clinicians during patient encounters. These notes are usually accompanied by a sequence of diagnostic codes following the International Classification of Diseases (ICD). Correctly assigning and ordering ICD codes is essential for medical diagnosis and reimbursement. However, automating this task remains challenging. State-of-the-art methods treated this problem as a classification task, leading to ignoring the order of ICD codes that is essential for different purposes. In this work, as a first attempt, we approach this task from a retrieval system perspective to consider the order of codes, thus formulating this problem as a classification and ranking task. Our results and analysis show that the proposed framework has a superior ability to identify high-priority codes compared to other methods. For instance, our model's accuracy in correctly ranking primary diagnosis codes i
    
[^128]: 大型语言模型生成代码中的库幻觉：基于开发者查询的风险分析

    Library Hallucinations in LLM-Generated Code: A Risk Analysis Grounded in Developer Queries

    [https://arxiv.org/abs/2509.22202](https://arxiv.org/abs/2509.22202)

    本研究首次系统分析了开发者查询变化如何触发大型语言模型生成代码中的库幻觉，揭示了不同提示条件下的系统性风险模式。

    

    arXiv:2509.22202v4 公告类型：替换交叉 摘要：大型语言模型（LLMs）在代码生成中现已扮演核心角色，但它们仍会出现幻觉，经常虚构不存在的库。此类库幻觉不仅仅是良性错误：它们可能误导开发者、破坏构建，并使系统面临供应链威胁，如“slopsquatting”（一种恶意软件包抢占攻击）。尽管对这些风险的认识日益增强，但对于库幻觉在现实使用条件下如何表现的理解仍然有限。为填补这一空白，我们首次系统性地研究了用户级提示变化如何影响LLM生成代码中的库幻觉。在七个不同的LLM中，我们分析了库名称幻觉（无效导入）和库成员幻觉（来自有效库的无效调用），考察了现实开发者语言和受控用户错误（包括拼写错误和虚构库或成员）的影响。我们的发现揭示了系统性漏洞。

    arXiv:2509.22202v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) now play a central role in code generation, yet they continue to hallucinate, frequently inventing non-existent libraries. Such library hallucinations are not just benign errors: they can mislead developers, break builds, and expose systems to supply chain threats such as slopsquatting. Despite growing awareness of these risks, there is limited understanding of how library hallucinations manifest under realistic usage conditions. To fill this gap, we present the first systematic study of how user-level prompt variations influence library hallucinations in LLM-generated code. Across seven diverse LLMs, we analyse library name hallucinations (invalid imports) and library member hallucinations (invalid calls from valid libraries), examining the effects of realistic developer language and controlled user mistakes, including misspellings and fabricated libraries or members. Our findings expose systemic v
    
[^129]: 规模还是推理？推理蒸馏的等价计算分析

    Scale or Reason? A Compute-Equivalent Analysis of Reasoning Distillation

    [https://arxiv.org/abs/2509.22193](https://arxiv.org/abs/2509.22193)

    该论文通过等价计算分析发现，在相同计算预算下，标准指令微调（IFT）在大多数配置中优于或持平于推理蒸馏，后者仅在7B以上模型的开放式任务中具有优势。

    

    从强大的教师模型中提炼推理轨迹已成为构建能力较强的小型语言模型的标准方法。然而，推理轨迹比标准指令微调（IFT）输出长5到20倍，这意味着每位选择推理蒸馏的实践者都在相同计算预算下隐式放弃了训练更大规模的IFT模型。这种权衡是否值得仍未得到解决。我们通过一项受控实验来研究这一问题：一个单一的教师模型通过仅切换其推理模式，为相同的提示生成配对的IFT和推理输出，将监督格式作为唯一的变量。我们在五个规模（0.5B到14B）上训练学生模型，并在18个基准上进行评估，发现在匹配的FLOPs下，IFT在大多数配置中位于或接近帕累托前沿。推理仅在7B及以上的开放式任务上达到帕累托前沿。即使在那里，一个顺序课程混合...

    arXiv:2509.22193v3 Announce Type: replace  Abstract: Distilling reasoning traces from strong teacher models has become the standard recipe for building capable small language models. Yet reasoning traces are 5-20$\times$ longer than standard instruction fine-tuning (IFT) outputs, meaning every practitioner who chooses reasoning distillation implicitly forgoes training a larger IFT model on the same compute budget. Whether this trade-off is worthwhile remains unaddressed. We study it with a controlled experiment: a single teacher generates paired IFT and reasoning outputs for identical prompts by toggling only its reasoning mode, isolating supervision format as the sole variable. Training students at five scales (0.5B to 14B) and evaluating on 18 benchmarks, we find that at matched FLOPs, IFT lies on or near the Pareto frontier across the majority of configurations. Reasoning reaches the Pareto frontier only on open-ended tasks at 7B and above. Even there, a sequential curriculum mixing
    
[^130]: SKILL-RAG：基于自知识诱导的学习与过滤增强检索生成

    SKILL-RAG: Self-Knowledge Induced Learning and Filtering for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2509.20377](https://arxiv.org/abs/2509.20377)

    提出SKILL-RAG方法，利用模型自知识通过强化学习训练框架来过滤无用检索内容，从而减少RAG中的幻觉并提升性能。

    

    检索增强生成（RAG）近年来显著提升了大型语言模型（LLMs）在知识密集型任务上的表现。然而，由于检索系统可能返回不相关的内容，将此类信息整合到模型中往往会导致幻觉。因此，识别并过滤掉无用的检索内容是提升RAG性能的关键挑战。为了更好地将模型的内部知识与检索到的外部知识相结合，理解模型“知道”和“不知道”的内容（也称为“自知识”）至关重要。基于这一见解，我们提出了SKILL-RAG（自知识诱导的学习与过滤增强RAG），这是一种新颖的方法，利用模型的自知识来确定哪些检索到的文档有助于回答给定查询。我们设计了一个基于强化学习的训练框架，以显式地引导这一过程。

    arXiv:2509.20377v2 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) has significantly improved the performance of large language models (LLMs) on knowledge-intensive tasks in recent years. However, since retrieval systems may return irrelevant content, incorporating such information into the model often leads to hallucinations. Thus, identifying and filtering out unhelpful retrieved content is a key challenge for improving RAG performance.To better integrate the internal knowledge of the model with external knowledge from retrieval, it is essential to understand what the model "knows" and "does not know" (which is also called "self-knowledge"). Based on this insight, we propose SKILL-RAG (Self-Knowledge Induced Learning and Filtering for RAG), a novel method that leverages the model's self-knowledge to determine which retrieved documents are beneficial for answering a given query. We design a reinforcement learning-based training framework to explicitly elic
    
[^131]: SCOPE：一种用于大语言模型提示压缩的生成式方法

    SCOPE: A Generative Approach for LLM Prompt Compression

    [https://arxiv.org/abs/2508.15813](https://arxiv.org/abs/2508.15813)

    本文提出了一种基于分块重写的无训练生成式提示压缩框架SCOPE，通过语义分块和摘要重构来缩短LLM输入，同时保持生成质量，并设计了优化技术以保留关键信息。

    

    摘要：arXiv:2508.15813v2 公告类型：替换-交叉 摘要：现代大语言模型应用中的一个主要问题是它们倾向于向LLM提供长上下文，这导致推理成本和延迟较高，并可能超出上下文限制。提示压缩通过减少输入上下文的长度，同时最小化生成质量的损失来解决此问题，即提示压缩的目标是缩短LLM输入，同时保持较高的生成质量。为了克服这些限制，我们提出了SCOPE，一种基于分块重写的无训练生成式提示压缩框架。与现有的令牌移除方法不同，我们的方法核心在于分块和摘要机制。具体来说，SCOPE将提示分割成语义连贯的块，并将这些块重写为更简洁的形式。然后，这些块被重构为一个有意义的提示。此外，我们为SCOPE设计了多种优化技术，有效保留了关键信息和文本连贯性。

    arXiv:2508.15813v2 Announce Type: replace-cross  Abstract: A big issue in modern LLM applications is they tend to feed long context to LLM, which results in high inference cost and latency, and may exceed the context limit. Prompt compression addresses this issue by reducing the length of input context with minimum loss of generation quality, i.e, the goal of prompt compression is to shorten the LLM input while maintaining a high generation quality. To overcome these limitations, we propose SCOPE, a training-free generative prompt compression framework based on chunk-level rewriting. Unlike the existing token removal methods, our method centers at a chunking-and-summarization mechanism. Specifically, SCOPE splits a prompt into semantically coherent chunks and rewrites the chunks to be more concise. Then the chunks are reconstructed into a meaningful prompt. Additionally, we design several optimization techniques for SCOPE, effectively preserving critical information and text coherence 
    
[^132]: CulTrace：追踪大型语言模型中的内部文化推理

    CulTrace: Tracing Internal Cultural Reasoning in Large Language Models

    [https://arxiv.org/abs/2508.08879](https://arxiv.org/abs/2508.08879)

    本文提出CulTrace方法，通过机械可解释性揭示大型语言模型在文化问答中内部文化推理的分阶段轨迹，并发现其推理存在不平衡性。

    

    大型语言模型在不同文化背景中的部署日益增多，这要求我们更深入地理解模型对不同文化的隐藏表征。以往的研究通过分析模型输出评估其文化意识，但这种方法忽视了文化在模型参数中的表示方式，无法解释模型为何产生错误回答。为弥补这一空白，我们提出了CulTrace，一种基于机械可解释性的方法，用于探测大型语言模型内部表征中的文化知识。通过CulTrace，我们检查了文化知识在层间如何处理，以及在文化问答中如何被整合。我们发现文化推理存在一致的阶段性轨迹：模型首先处理问题的领域，然后解析相关文化，最后聚焦于答案。我们还证明模型的文化推理存在不平衡性，表现出延迟的相关性。

    arXiv:2508.08879v3 Announce Type: replace-cross  Abstract: The growing deployment of large language models (LLMs) across diverse cultural contexts necessitates a deeper understanding of models' hidden representations of different cultures. Prior work has evaluated cultural awareness in LLMs by analysing their outputs. This approach overlooks how cultures are represented within the model parameters, missing why models generate incorrect responses. To bridge this gap, we propose CulTrace, a mechanistic interpretability-based method that probes the internal representations of LLMs for cultural knowledge. With CulTrace, we inspect how cultural knowledge is processed across layers and how it is integrated during cultural QA. We find a consistent staged trajectory of cultural reasoning. Models first engage with the question's domain, then resolve the relevant culture, and finally narrow in on an answer. We also demonstrate that models' cultural reasoning is imbalanced, showing delayed releva
    
[^133]: CPC-CMS：面向文档级情感分析的认知成对比较分类模型选择框架

    CPC-CMS: Cognitive Pairwise Comparison Classification Model Selection Framework for Document-level Sentiment Analysis

    [https://arxiv.org/abs/2507.14022](https://arxiv.org/abs/2507.14022)

    该框架通过认知成对比较加权多标准评估，自动选择文档级情感分析的最优分类模型，并在多个数据集上验证了其有效性。

    

    本研究提出了用于文档级情感分析的认知成对比较分类模型选择（CPC-CMS）框架。基于专家知识判断的CPC方法被用于计算评估标准的权重，这些标准包括准确率、精确率、召回率、F1分数、特异度、马修斯相关系数（MCC）、科恩卡帕系数（Kappa）和效率。选择朴素贝叶斯（NB）、线性支持向量分类（LSVC）、随机森林、逻辑回归、极端梯度提升（XGBoost）、长短期记忆网络（LSTM）和轻量级双向编码器表示（ALBERT）作为分类基线模型。通过形成由分类评估分数相对于标准权重组成的加权决策矩阵，为分类问题选择最佳分类模型。使用三个开放社交媒体数据集来证明所提方法的可行性。

    arXiv:2507.14022v3 Announce Type: replace  Abstract: This study proposes the Cognitive Pairwise Comparison Classification Model Selection (CPC-CMS) framework for document-level sentiment analysis. The CPC, based on expert knowledge judgment, is used to calculate the weights of evaluation criteria, including accuracy, precision, recall, F1-score, specificity, Matthews Correlation Coefficient (MCC), Cohen's Kappa (Kappa), and efficiency. Naive Bayes (NB), Linear Support Vector Classification (LSVC), Random Forest, Logistic Regression, Extreme Gradient Boosting (XGBoost), Long Short-Term Memory (LSTM), and A Lite Bidirectional Encoder Representations from Transformers (ALBERT) are chosen as classification baseline models. A weighted decision matrix consisting of classification evaluation scores with respect to criteria weights is formed to select the best classification model for a classification problem. Three open social media datasets are used to demonstrate the feasibility of the prop
    
[^134]: GeoExplain：基于街景视觉信息层次的多模态推理

    GeoExplain: Multimodal Reasoning based on Hierarchy of Visual Information in Street View

    [https://arxiv.org/abs/2506.16633](https://arxiv.org/abs/2506.16633)

    本文提出了GeoExplain数据集，首次将可解释的地理定位与多模态推理结合，通过层次化视觉信息（局部细节和全局上下文）来预测街景位置并生成人类可理解的解释。

    

    多模态推理是跨不同数据模态理解、整合和推断信息的过程，近年来引起了学术界的广泛关注。尽管已有多种任务用于评估多模态推理能力，但它们仍存在局限性。对于不同粒度级别的层次化视觉线索（即局部细节和全局上下文）的推理讨论较少，尽管这在人类推理中频繁涉及。为弥补这一差距，我们引入了一个具有挑战性的数据集，名为GeoExplain，用于评估可解释的地理定位。给定一张街景图像，任务是预测其位置并提供详细解释。GeoExplain包含40350个全景-位置-解释三元组。每个实例包含一组街景全景图、一个街道级别的位置以及描述如何从视觉信息推断出位置的人类专家解释。

    arXiv:2506.16633v3 Announce Type: replace-cross  Abstract: Multimodal reasoning is a process of understanding, integrating and inferring information across different data modalities. It has recently attracted surging academic attention. Although there are various tasks for evaluating multimodal reasoning ability, they still have limitations. Reasoning on hierarchical visual clues at different levels of granularity, i.e., local details and global context, is of little discussion, despite its frequent involvement in human reasoning. To bridge the gap, we introduce a challenging dataset, namely GeoExplain, which evaluates explainable geo-localization. Given a street view image, the task is to predict its location and provide a detailed explanation. GeoExplain consists of 40350 panoramas-location-explanation tuples. Each instance contains a set of street-view panoramas, a location on street level, and human-expert explanations describing how the location can be inferred from the visual con
    
[^135]: 超越黄金标准：基于认知集成的大语言模型法官用于形式化数学推理

    Beyond Gold Standards: Epistemic Ensemble of LLM Judges for Formal Mathematical Reasoning

    [https://arxiv.org/abs/2506.10903](https://arxiv.org/abs/2506.10903)

    本文提出了一种基于认知和形式化基础的LLM法官集成方法，通过细粒度、多层次的评估标准，系统地自动评估形式化数学推理中的语句自动形式化任务，弥补了现有评估方法的粗粒度缺陷。

    

    arXiv:2506.10903v2 公告类型：替换 摘要：语句自动形式化在形式化数学推理中起着关键作用，它能够将自然语言语句自动翻译为形式语言。尽管近期利用大语言模型（LLM）的进展显示出自动形式化的强大能力，但自动评估自动形式化的方法仍未得到充分探索。将LLM作为法官提供了一种自动化此类评估的有前景方法，然而，现有方法通常采用粗粒度且通用的评估标准，这限制了其在高级形式化数学推理中的有效性，因为该领域的质量依赖于细微、多粒度的维度。在本工作中，我们向解决这一差距迈出一步，引入了一种系统化的自动方法来评估自动形式化任务。所提方法基于一个认知和形式上扎实的LLM法官集成（EFG），其定义在涵盖逻辑的评估标准上。

    arXiv:2506.10903v2 Announce Type: replace  Abstract: Statement autoformalization plays a crucial role in formal mathematical reasoning by enabling the automatic translation of natural language statements into formal languages. While recent advances using large language models (LLMs) have shown promising capability of autoformalization, methods for automatically evaluating autoformalization remain underexplored. LLM-as-a-judge presents a promising approach for automating such evaluation, however, existing methods typically employ coarse-grained and generic evaluation criteria, which limit their effectiveness for advanced formal mathematical reasoning, where quality hinges on nuanced, multi-granular dimensions. In this work, we take a step toward addressing this gap by introducing a systematic, automatic method to evaluate autoformalization tasks. The proposed method is based on an epistemically and formally grounded ensemble (EFG) of LLM judges, defined on criteria encompassing logical 
    
[^136]: 不要以貌取“码”：探索大语言模型在代码评估中的偏见

    Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation

    [https://arxiv.org/abs/2505.16222](https://arxiv.org/abs/2505.16222)

    本研究首次系统性地揭示了大语言模型在代码评估中对表面差异（如变量名、注释和格式）存在偏见，并通过多种语言和模型实证证明了这些偏见会影响评估的公平性。

    

    arXiv:2505.16222v2 公告类型：替换 摘要：随着大语言模型（LLMs）作为评估者的使用日益增长，其应用已扩展到代码评估任务，即在不依赖参考实现的情况下评估生成代码的正确性。虽然这提供了可扩展性和灵活性，但也引发了一个关键且未解决的问题：LLM法官能否公平且稳健地评估具有表面差异的语义等价代码？功能正确的代码通常表现出差异——例如变量名、注释或格式的不同——这些差异不应影响其正确性。然而，LLM法官能否可靠地处理这些差异仍不清楚。我们首次全面研究了这一问题，定义了代码评估中六种潜在偏见类型，并揭示了它们对LLM法官的系统性影响。在五种编程语言和多种LLM中，我们实证表明，所有测试的LLM法官都容易受到这些偏见的影响。

    arXiv:2505.16222v2 Announce Type: replace  Abstract: With the growing use of large language models(LLMs) as evaluators, their application has expanded to code evaluation tasks, where they assess the correctness of generated code without relying on reference implementations. While this offers scalability and flexibility, it also raises a critical, unresolved question: Can LLM judges fairly and robustly evaluate semantically equivalent code with superficial variations? Functionally correct code often exhibits variations-such as differences in variable names, comments, or formatting-that should not influence its correctness. Yet, whether LLM judges can reliably handle these variations remains unclear. We present the first comprehensive study of this issue, defining six types of potential bias in code evaluation and revealing their systematic impact on LLM judges. Across five programming languages and multiple LLMs, we empirically demonstrate that all tested LLM judges are susceptible to b
    
[^137]: 用机械可解释性解释内在道德自我修正

    Explaining Intrinsic Moral Self-Correction with Mechanistic Interpretability

    [https://arxiv.org/abs/2505.11924](https://arxiv.org/abs/2505.11924)

    该论文通过机械可解释性揭示了内在道德自我修正的机制是表示引导，即提示词通过沿可解释潜在方向调整隐藏表示来改变模型行为，且这种方法比直接提示更有效。

    

    arXiv:2505.11924v4 公告类型：替换交叉 摘要：内在道德自我修正指的是语言模型仅通过提示词来优化其伦理判断或调整其输出的现象。尽管在多种任务中有效，但其机制仍不清楚。我们假设内在道德自我修正通过将隐藏表示沿可解释的潜在方向引导来起作用。通过评估六个大型语言模型在四个道德相关任务上的表现，我们证明了自我修正提示引起的表示变化与对比性引导向量一致。即使引导向量是从不相关语料库构建的，这种一致性也能转移。值得注意的是，当通过激活添加应用时，这些提示引起的偏移能比自我修正提示和引导向量更有效地改变模型行为。我们的发现表明，表示引导是内在道德自我修正的机制驱动因素。

    arXiv:2505.11924v4 Announce Type: replace-cross  Abstract: Intrinsic moral self-correction refers to the phenomenon where a language model refines its ethical judgments or aligns its outputs purely through prompting. While effective across diverse tasks, its mechanism remains unclear. We hypothesize intrinsic moral self-correction functions by steering hidden representations along interpretable latent directions. Evaluating six LLMs across four morality-related tasks, we demonstrate that the representation shifts induced by self-correction prompts align with contrastive steering vectors. This alignment transfers even when the steering vectors are constructed from a disjoint corpus. Notably, when applied via activation addition, these prompt-induced shifts can alter model behavior more effectively than the self-correction prompts and the steering vectors. Our findings suggest representation steering is the mechanistic driver of intrinsic moral self-correction.
    
[^138]: 大型语言模型内部表示中提示词的内在维度

    The Intrinsic Dimension of Prompts in Internal Representations of Large Language Models

    [https://arxiv.org/abs/2501.10573](https://arxiv.org/abs/2501.10573)

    本文通过内在维度分析大型语言模型提示词表示，发现其与词元不确定性相关，并利用逐层内在维度轮廓训练线性探针，在生成前高效区分恶意与良性提示，准确率达90-95%。

    

    arXiv:2501.10573v2 公告类型：替换 摘要：我们通过内在维度的视角，研究了大型语言模型中提示词级别的词元表示几何结构。将变换器视为平均场粒子系统，我们估计了每一层经验测度的内在维度，并证明其与下一个词元的不确定性相关。跨模型和内在维度估计器，我们发现内在维度在早期到中层达到峰值，并在句法和语义扰动（通过打乱词元）下增加，且与平均惊异度强相关，通过softmax将逻辑几何与熵联系起来进行简单分析。作为实际可解释性和安全性的案例研究，我们在逐层内在维度轮廓上训练了一个线性探针，以在生成前区分恶意和良性提示词。该探针在不同数据集上达到90%至95%的准确率，优于广泛使用的防护措施。

    arXiv:2501.10573v2 Announce Type: replace  Abstract: We study the geometry of token representations at the prompt level in large language models through the lens of intrinsic dimension. Viewing transformers as mean-field particle systems, we estimate the intrinsic dimension of the empirical measure at each layer and demonstrate that it correlates with next-token uncertainty. Across models and intrinsic dimension estimators, we find that intrinsic dimension peaks in early to middle layers and increases under syntactic and semantic disruption (by shuffling tokens), and that it is strongly correlated with average surprisal, with a simple analysis linking logits geometry to entropy via softmax. As a case study in practical interpretability and safety, we train a linear probe on the per-layer intrinsic dimension profile to distinguish malicious from benign prompts before generation. This probe achieves accuracy of 90 to 95\% in different datasets, outperforming widely used guardrails such a
    
[^139]: 阿尔茨海默病检测中的类内变异问题研究

    On the Within-class Variation Issue in Alzheimer's Disease Detection

    [https://arxiv.org/abs/2409.16322](https://arxiv.org/abs/2409.16322)

    本文针对阿尔茨海默病检测中的类内变异问题，提出软目标蒸馏和实例级重平衡两种方法，通过估计样本特定概率分数来建模异质性和不平衡，从而提升检测性能。

    

    阿尔茨海默病（AD）检测通常采用机器学习分类模型来区分AD患者与非AD个体。与传统分类任务不同，AD检测涉及显著的类内变异，因为具有相同诊断的个体可能表现出不同程度的认知障碍。我们将该问题归纳为两个方面：类内异质性和实例级不平衡。为了在二元监督下建模这种变异，我们估计样本特定的AD类别概率作为样本分数，并开发了两种相应方法：软目标蒸馏（SoTD）和实例级重平衡（InRe）。在ADReSS和CU-MARVEL语料库上的实验表明，估计的分数与独立的认知评估一致，且所提出的方法提高了AD检测性能。这些发现为建模类内变异提供了见解。

    arXiv:2409.16322v4 Announce Type: replace-cross  Abstract: Alzheimer's Disease (AD) detection commonly employs machine learning classification models to distinguish between individuals with AD and those without. Different from conventional classification tasks, AD detection involves substantial within-class variation, as individuals sharing the same diagnosis may exhibit different degrees of cognitive impairment. We formulate two aspects of this issue: within-class heterogeneity and instance-level imbalance. To model such variation under binary supervision, we estimate sample-specific AD class probabilities as sample scores and develop two corresponding methods: Soft Target Distillation (SoTD) and Instance-level Re-balancing (InRe). Experiments on the ADReSS and CU-MARVEL corpora show that the estimated scores align with independent cognitive assessments and that the proposed approaches improve AD detection performance. These findings provide insights for modeling within-class variatio
    

