# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How to Train a Critic Stably and Efficiently](https://arxiv.org/abs/2608.23566) | BPCO通过结合DPPO、价值预测约束、蒙特卡洛目标、非归一化优势和长度自适应GAE，实现了稳定高效的评论家训练，并利用隐藏于策略的奖励信息提升数学推理性能。 |
| [^2] | [SWE Refactor Bench: Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration?](https://arxiv.org/abs/2608.23564) | 本文提出了SWE重构基准，通过三阶段评估协议解决现有基准无法检测迁移是否真正发生的问题，从而衡量编码代理在长时程全仓库栈迁移中的能力。 |
| [^3] | [Prime Agent: A Self-Improving RLM Harness](https://arxiv.org/abs/2608.23552) | Prime Agent是一个开源工具框架，通过持久化REPL和递归子代理机制，将长期评估和编码代理工作流标准化，从而防止工具故障干扰模型，最大化模型潜力。 |
| [^4] | [ConvergeFlow: Language Flow with Provable Convergence to Token Embeddings](https://arxiv.org/abs/2608.23551) | 本文提出ConvergeFlow，一种基于嵌入空间的流式语言模型，通过约束数据预测器到词元嵌入凸包并仅用均方误差训练，证明了流可收敛到有效词元嵌入，从而消除了对交叉熵解码器的需求。 |
| [^5] | [When Names Cross Scripts: A Source-Grounded Benchmark for Historical Entity Reconciliation in the Mongol World](https://arxiv.org/abs/2608.23507) | 该论文提出了MHER基准，证明在蒙古世界历史人物对账中，基于来源的证据能显著提升准确率，而仅依赖姓名在相同名字的不同人物案例中完全失效。 |
| [^6] | [Mitigating Reasoning-Induced Misalignment via Safety-Direction Penalty](https://arxiv.org/abs/2608.23497) | 本文提出安全方向惩罚（SDP）方法，通过分析表示空间中的推理与安全方向耦合机制，在推理微调时惩罚安全方向的移动，以有效缓解推理引发的不对齐问题。 |
| [^7] | [On the Threat Model of Weird Generalization and Emergent Misalignment](https://arxiv.org/abs/2608.23476) | 本研究揭示了怪异泛化现象主要由微调数据的构成和语言驱动，而非数据集大小，并且其测量结果对评估问题集高度敏感。 |
| [^8] | [What's the Catch? Evaluating Temporal Consistency in Vision-Language Models](https://arxiv.org/abs/2608.23474) | 本研究通过时间异常检测任务揭示了视觉语言模型在时间一致性理解上的显著不足，与人类表现存在明显差距。 |
| [^9] | [How Useful are LLMs for Grammar Engineering? Cantonese ParGram Resources and Controlled Experimental Evaluation with English Baselines](https://arxiv.org/abs/2608.23448) | 本文通过粤语ParGram资源和英语基线实验，发现GPT-5.4在生成机器可处理语法方面优于gpt-oss-120b，但两者均难以协调复杂形式约束，表明大语言模型在语法工程中适合辅助局部生成而非全局集成。 |
| [^10] | [A Comprehensive Analysis of Arabic Natural Language Processing Research: Trends, Topic Evolution, and Research Gaps -- A Bibliometric and Topic-Based Study](https://arxiv.org/abs/2608.23421) | 本研究首次对7120篇阿拉伯语NLP论文进行大规模文献计量与主题分析，揭示了2020年后由Transformer和LLM驱动的出版激增现象，并识别出19个核心研究主题及引用影响因素。 |
| [^11] | [Robustness of IR Models to Collection Growth](https://arxiv.org/abs/2608.23419) | 本研究正式化并实证评估了信息检索模型对添加非相关文档的鲁棒性，发现无论模型是否依赖其他文档，均无法完全避免性能下降。 |
| [^12] | [STONIC: A Layered Measurement Contract for LLM Value Profiling](https://arxiv.org/abs/2608.23411) | STONIC通过分层测量契约验证了LLM价值剖析中评分、选择和文本推断的一致性假设，发现大多数配置保持认可-选择关系但偏好自身答案，且剖析形状在不同测量层次间转移强度不一。 |
| [^13] | [Cross-Domain, Multi-Task Data-to-Text Generation without In-Domain Training Data](https://arxiv.org/abs/2608.23391) | 本研究提出了一种无需领域内训练数据的跨域、多任务数据到文本生成方法，通过数据驱动的知识蒸馏和结构保持增强，在恒定模型大小下优于微调和零样本推理。 |
| [^14] | [Cross-lingual Biography Enrichment via Claim Extraction and Alignment](https://arxiv.org/abs/2608.23390) | 本文提出了一种基于主张提取与对齐的框架，利用非英文维基百科传记中的本地化事实来丰富英文传记，并通过CLAW-4L基准验证了其有效性。 |
| [^15] | [The Geometry of Low-Resource Language Representations](https://arxiv.org/abs/2608.23358) | 本文发现低资源语言在大型语言模型最终层存在表征退化，并提出几何正则化方法在持续预训练中有效缓解该问题，尤其对较大模型性能有轻微提升。 |
| [^16] | [FormuEvo: LLM-Guided Evolution for Discovering Solver-Efficient Mixed-Integer Programming Formulations](https://arxiv.org/abs/2608.23353) | FormuEvo提出了一种利用大语言模型引导的进化框架，通过求解器感知的诊断机制和符号空间优化，自动发现求解器高效的混合整数规划公式，克服了传统LLM建模忽视公式强度的问题。 |
| [^17] | [The Emergence of Relevance Through Axiomatic Attention Patterns During LoRA Fine-Tuning](https://arxiv.org/abs/2608.23338) | 本研究发现，在LoRA微调过程中，仅对网络中紧凑的中部区域进行注意力更新即可恢复大部分性能提升，且该区域与可解释的相关性注意力模式出现区域高度重合。 |
| [^18] | [Flesch-Kincaid Readability Depends Only on the Topic Distribution in Long Texts under Topic Models](https://arxiv.org/abs/2608.23327) | 在主题模型下，长文本的Flesch可读性评分几乎完全由主题分布决定，与词汇组成无关，揭示了可读性评分的本质局限性。 |
| [^19] | [Agent-G$^2$: Gaussian Guidance for Agentic Reinforcement Learning](https://arxiv.org/abs/2608.23318) | 我们提出Agent-G$^2$，一种无需额外探测回合的高斯引导框架，通过在线估计每个任务的最优引导深度分布，有效解决智能体强化学习中的奖励稀疏问题。 |
| [^20] | [Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization](https://arxiv.org/abs/2608.23311) | 本文提出环境正则化策略优化（ERPO），通过将正则化从动作侧移至输入侧并引入查询KL约束，有效打破了大语言模型策略优化中稳定性与探索之间的两难困境。 |
| [^21] | [Dynamic Topic Modeling for Cross-Corpus Temporal Analysis](https://arxiv.org/abs/2608.23284) | 该论文提出了一种基于共享骨干和残差适应的动态主题建模框架，解决了跨语料库时间分析中主题对齐不稳定问题，实现了稳定的跨语料库比较和词汇专门化。 |
| [^22] | [Apodex 1.1: Scaling Agentic Intelligence for Complex Work](https://arxiv.org/abs/2608.23283) | Apodex 1.1 通过环境扩展和代理协调扩展，提升了语言模型在复杂工作中持续交互、状态维护和可验证交付的工作能力。 |
| [^23] | [Expectations and Practices around AI Disclosure in CS Research](https://arxiv.org/abs/2608.23271) | 本研究调查了计算机科学场所的AI披露政策，发现其普遍缺乏具体规定，并通过调查和案例分析揭示了研究人员对披露必要性及信息内容的期望。 |
| [^24] | [EvoWiki: Incremental State Overwriting and Traceable Question Answering for Cross-Meeting Knowledge Evolution](https://arxiv.org/abs/2608.23265) | EvoWiki通过增量状态覆盖协议和实体版本链，显式建模知识生命周期，解决了跨会议问答中新旧状态冲突和答案不可验证的问题。 |
| [^25] | [Hidden in the Request: Explaining Unethical LLM Compliance through Token Relevance](https://arxiv.org/abs/2608.23264) | 本文通过引入三种模态的探测方法，发现大语言模型在直接请求帮助时更易顺从于不道德行为，并利用层间相关性传播揭示其归因偏差——模型过度关注任务框架令牌而忽视不道德提示令牌，从而解释了对齐失败的机制。 |
| [^26] | [Automated Construction of FAIR Digital Object Knowledge Graphs from Flat Cultural Heritage Records](https://arxiv.org/abs/2608.23263) | 本文提出了一种自动化流水线，将扁平的Europeana文化遗产记录转换为符合FDO规范的知识图谱，通过自动区分PID引用和字面量值，实现文化遗产数据的完全机器可操作性。 |
| [^27] | [A Scalable Cross-Domain Event Extraction System via a Unified Generative Training Framework](https://arxiv.org/abs/2608.23261) | 本文提出了一种统一的生成式序列到序列事件抽取框架，通过多领域微调实现可扩展性和跨领域泛化，并提供了一个支持文档上传、模式感知抽取和可视化比较的Web应用平台。 |
| [^28] | [The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search](https://arxiv.org/abs/2608.23252) | 本文提出因果留一法探针解决RAG中证据测量幻觉，并证明迭代分配上下文比单一扩展更优，带来16.7--20.5%的召回率提升。 |
| [^29] | [Future Querying: Can LLMs Serve as Implicit Medical World Models?](https://arxiv.org/abs/2608.23248) | 本文提出“未来查询”范式，利用端点无关训练使小型开源LLM能在非结构化临床文本上充当隐式医学世界模型，回答时间索引的患者未来查询，并匹配大型专有系统性能，支持隐私保护的本地部署。 |
| [^30] | [Credal Large Language Models for Semantic Commitment under Uncertainty](https://arxiv.org/abs/2608.23244) | 通过集成LoRA适配器构建可信集，提出CTC和SCC分数来区分认知无知与真实模糊性，从而减少LLM的过度自信错误。 |
| [^31] | [A Multi-Domain and Multi-Task Generative Framework with Explicit Task and Domain Conditioning for Cross-Domain Event Extraction](https://arxiv.org/abs/2608.23235) | 本文提出一个多领域多任务生成框架，通过显式领域和任务条件信号，在单一模型中动态适应异构事件模式，无需完整事件标签集，从而提升跨领域事件抽取的泛化能力。 |
| [^32] | [Aligning Biomedical Texts and Knowledge Graphs: A Systematic Comparison of Lightweight Alignment Strategies](https://arxiv.org/abs/2608.23214) | 本文提出一个统一框架，通过冻结文本和知识图谱模型并学习轻量级投影，系统比较六种设计维度以对齐生物医学文本与知识图谱，并构建了CTD-Align数据集。 |
| [^33] | [Cognitive Profiling of LRMs' Reasoning Traces Using Bloom's Taxonomy](https://arxiv.org/abs/2608.23205) | 本文提出了一种基于布鲁姆分类法的自动标注框架，用于分析大型推理模型的推理步骤思维类型，并跨模型和数据集揭示其思维模式异同，为推理行为提供了新洞察。 |
| [^34] | [LongWoF-Bench: Evaluating EvoMap Genes for Verifiable Long-Workflow Tasks](https://arxiv.org/abs/2608.23200) | 本文提出LongWoF-Bench基准和EvoMap方法，通过将验证器确认的执行轨迹整合为结构化基因，实现经验复用，在可验证长工作流任务中显著优于技能方法。 |
| [^35] | [CyberFactory: Scaling Cyber Security Capabilities with Instances from the Wild](https://arxiv.org/abs/2608.23181) | CyberFactory是一个统一开源框架，通过将真实世界CVE漏洞转化为可执行任务实例，并整合数据构建、轨迹合成和模型训练，从而扩展网络安全能力。 |
| [^36] | [CaRGo-T: Causal Reasoning Graph-of-Thought improves Multimodal Humor Comprehension](https://arxiv.org/abs/2608.23172) | CaRGo-T提出了一种基于图的因果推理框架，通过将多模态幽默中的复杂关系序列化为代码表示，提升了视觉-语言模型在幽默理解任务中的推理能力。 |
| [^37] | [Accelerating Diffusion Language Models via Structured Suffix Modeling](https://arxiv.org/abs/2608.23167) | 本文提出了一种结构化后缀建模方法，通过将后缀划分为局部、中部和尾部区域并自适应保留不同数量的标记，以及利用前一步解码结果进行初始化，从而显著提升扩散语言模型的推理效率。 |
| [^38] | [Counter with Evidence! A Multi-Agent Memory Efficient Reasoning Framework for Hate Category Informed Counterspeech Generation](https://arxiv.org/abs/2608.23152) | 本文提出FIRE框架，通过将仇恨言论分类为五种具体类别并映射到针对性反驳风格，结合新数据集FactualCS，实现了更精准且基于证据的反驳生成。 |
| [^39] | [Language Chain in Alignment: Cross-Lingual Ranking Preference Optimization](https://arxiv.org/abs/2608.23149) | 本文提出跨语言排序偏好优化（CRPO）框架，通过利用英语偏好知识的分层结构，在目标语言中实现更优的语言对齐和输出质量。 |
| [^40] | [Activation-Weighted Seeded Residual Coding for Low-Bit LLM Weight Repair](https://arxiv.org/abs/2608.23144) | 本文提出激活加权种子残差编码（AWSRC），通过利用激活统计和种子生成基，以极小辅助存储（约0.8%权重负载）高效修复低比特量化误差，显著提升模型质量。 |
| [^41] | [LITERARYBIGFIVE: Author-Personalized Text Generation in a Unified Interpretable Space](https://arxiv.org/abs/2608.23124) | 本文提出LiteraryBigFive框架，将作者写作特征映射到统一可解释空间中，通过激活对比生成风格维度，实现无需大规模标注或微调的跨作者个性化文本生成。 |
| [^42] | [Statistical Machine Translation Systems of English-Pnar Language Pair : Some Insights of the Emperical Study](https://arxiv.org/abs/2608.23120) | 本文首次为英语-普纳尔语语言对构建平行语料库并训练统计机器翻译系统，建立了该语言对的第一个定量基准。 |
| [^43] | [Molecular LLM Agents: From Architectural Design to Scientific Autonomy](https://arxiv.org/abs/2608.23104) | 本文提出了分子大语言模型智能体的概念框架，从架构设计和科学自主性阶梯两个视角，系统阐述了分子感知、智能体框架、工具接地及学习优化，并定义了从基础工具使用到完全自主科学研究的分级能力路径。 |
| [^44] | [Definitional Sensitivity in Media Bias Detection: A Multi-Definition Dataset and Benchmark](https://arxiv.org/abs/2608.23095) | 该研究通过大规模人类和LLM实验发现，媒体偏见检测中的定义概念框架会显著影响标注结果，且对LLM影响更强，而构念保留的阐述则无此效应。 |
| [^45] | [AgentWeave: Routing Before Reasoning for Efficient Function Calling in Tool-Rich Language Models](https://arxiv.org/abs/2608.23078) | AgentWeave通过在推理前使用确定性路由层减少候选工具集，在不改变下游模型的情况下显著提升多函数调用任务的成功率。 |
| [^46] | [Signal or Noise? A Benchmark Study of Agent Skills in Web Development](https://arxiv.org/abs/2608.23067) | 该研究通过引入WebDev-Skills-Bench基准，发现注入代理技能在Web开发任务中不仅无益，反而降低了任务成功率并增加了资源消耗，表明技能注入需谨慎评估。 |
| [^47] | [Cultural Moment Benchmark: Evaluating Video Cultural Reasoning and Grounding in Southeast Asia](https://arxiv.org/abs/2608.23065) | 本文提出了文化时刻基准（CMB），通过三个阶段分别评估视频文化理解中的命名、视觉识别和时间定位能力，填补了现有基准混淆这些能力的空白。 |
| [^48] | [Beyond Verdicts: A Graph-Based Analysis of Human and LLM Reasoning in Scientific Fact-Checking](https://arxiv.org/abs/2608.23047) | 本文提出一种基于图的推理框架，用于系统比较人类专家与大型语言模型在科学事实核查中的推理路径，从而揭示模型是否通过相同或不同的有效推理过程得出结论。 |
| [^49] | [AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces](https://arxiv.org/abs/2608.23041) | AutoSaddler通过将框架优化视为离线学习问题，利用失败轨迹诊断和代码式补丁生成，自动迭代改进代理框架，在多个基准上显著提升性能。 |
| [^50] | [The Multilingual FrameNet Corpus](https://arxiv.org/abs/2608.23037) | 本文构建了包含九种语言的多语言FrameNet语料库（mFNC），通过多语言训练数据显著提升了框架语义解析的性能，超越了现有最先进模型。 |
| [^51] | [ST$^2$U: Stateful Test-Time Unlearning via Restricted Knowledge Boundary Control](https://arxiv.org/abs/2608.23034) | ST²U通过轨迹级别的受限知识边界控制，解决了现有测试时遗忘方法因忽略自回归生成中状态重建而导致的受限知识重入问题。 |
| [^52] | [Meta-Moderator: Empowering Multi-Agent Debate with Meta-Cognition](https://arxiv.org/abs/2608.23029) | 元主持人通过可学习的元认知框架动态调节多智能体辩论，显式优化辩论效用并决定何时终止，从而显著提升推理性能。 |
| [^53] | [Beyond Surface Cues: Disentangling Sociocultural Signals in Multilingual LLMs](https://arxiv.org/abs/2608.23026) | 该研究通过多智能体审计方法，在89,253个多语言大语言模型输出中分离社会文化信号，发现直接身份线索对英语和中文的偏见识别影响显著，而对法语影响较小，揭示了跨语言和任务的系统性差异。 |
| [^54] | [Most of the LLM routing gap is task type](https://arxiv.org/abs/2608.23023) | 本文发现LLM路由器性能差距主要源于任务类型差异，而非模型选择策略，即使重复运行也存在评分不稳定性。 |
| [^55] | [Unlearning Is Not Just Erasing: Temporal Decoupling via Generation Inequality](https://arxiv.org/abs/2608.23020) | 该论文提出ADU框架，通过时间解耦和生成不平等，将LLM遗忘从简单的令牌擦除转变为上下文注意力路径解耦，以在保持模型通用性的同时实现精确遗忘。 |
| [^56] | [PatchWrite: One Line, Not One Section -- Compile-Gated, Validity-Preserving Editing for AI-Drafted Manuscripts](https://arxiv.org/abs/2608.23001) | PatchWrite通过编译门控和证据锁机制，仅允许局部编辑而非整节重写，从而在修复手稿缺陷时严格保留无关内容，显著提升编辑的有效性和安全性。 |
| [^57] | [LLM Pedagogical Behavior in AI Tutoring Interactions](https://arxiv.org/abs/2608.22993) | 该论文开发了一个五级脚手架量表，发现大学AI课程中学生的LLM辅导互动超过95%的回应集中于高直接帮助水平（解释或解决），且这种帮助水平与后续对话行为相关，但对考试成绩预测力有限。 |
| [^58] | [What Does Activation Steering Control? Attribution Across Answer Encodings and Output-Sensitive Subspaces](https://arxiv.org/abs/2608.22985) | 本文提出跨编码引导评估方法，发现激活引导的干预效果主要跟随提取索引而非语义标签，且该效应在深层网络中更为显著，揭示了输出敏感子空间的关键作用。 |
| [^59] | [Closed-Loop Bayesian Molecular Inverse Design with Semantic LLM Surrogates](https://arxiv.org/abs/2608.22967) | 该论文提出了一种闭环贝叶斯分子逆向设计框架，通过将大型语言模型作为代理直接处理文本形式的任务指令和优化历史，以在有限预算下提高匹配目标性质的分子比例。 |
| [^60] | [Buried in Textual Debt: Context Pruning with Visual Evidence Preservation for MLLM Agents](https://arxiv.org/abs/2608.22963) | 本文提出SPARE框架，通过KL散度引导的上下文剪枝，在保留视觉证据的同时移除冗余推理文本，以解决MLLM智能体长轨迹中的“文本债务”问题。 |
| [^61] | [The Illusion of Control: Why Bare Classifier Inversion Silently Fails in Concept-Bottleneck Text Generation](https://arxiv.org/abs/2608.22956) | 本文发现，在概念瓶颈文本生成中，裸分类器反演及其正则化变体均不如简单的事后先验方法，揭示了反演方法在组合泛化中的固有局限。 |
| [^62] | [What Proves You Wrong: Benchmarking Language Models on Falsifiable Research Ideation](https://arxiv.org/abs/2608.22948) | 本文提出了Lit2Test基准，通过要求研究提议预先指定可证伪结果，使提议质量可判定，并基于200个真实论文邻域和1200次盲评比较了四个前沿语言模型。 |
| [^63] | [HelaBERT: Enhancing Sinhala Language Understanding with Dual Pooling Classification Head](https://arxiv.org/abs/2608.22922) | 本文提出了HelaBERT，一种针对僧伽罗语预训练的BERT模型家族，并引入双池化分类头，在多个分类任务中显著提升了情感分析性能。 |
| [^64] | [TSWAP: A Multilingual Retrieval-Augmented Thai Wellness Advisor](https://arxiv.org/abs/2608.22917) | TSWAP通过检索增强生成和混合检索技术，在泰式传统医学知识库上实现了八语言零样本健康顾问，并发布了首个泰式医学检索基准和QA日志，展示了各接地组件的贡献。 |
| [^65] | [Knowing Isn't Always Saying: When Do Spatial Encodings Reach Answers in Vision-Language Models?](https://arxiv.org/abs/2608.22916) | 该论文通过方向修补干预揭示了视觉-语言模型中空间编码对答案的影响仅在深层出现，并受提示格式和思维链影响，形成不同传输模式。 |
| [^66] | [Safety Hacking in Constrained Best-of-$N$ Inference-time Scaling](https://arxiv.org/abs/2608.22915) | 本文发现推理时管道中不完美的安全代理与奖励最大化组合会导致“安全黑客攻击”，即使代理误差极小，当N增大时，不安全输出若尾部更重，攻击概率趋近于1。 |
| [^67] | [Exploring Dowker Homology for Sentence Similarity](https://arxiv.org/abs/2608.22909) | 本文探索道克同调作为拓扑工具，通过将句子对标记嵌入视为点云，证明其能有效捕捉句子相似性信息，并可用于视觉检查，但单数值摘要未超越传统度量。 |
| [^68] | [Do Spoken Language Models Hear Speech as They Read Text? Bridging Structural Gaps Between Speech and Text](https://arxiv.org/abs/2608.22908) | 本文提出一种解耦长度不匹配与语义对齐的简单框架，以增强口语语言模型中语音与文本表示的结构对齐，从而提升指令遵循和泛化能力。 |
| [^69] | [SelFusion: Self-distillation for Diffusion Language Models](https://arxiv.org/abs/2608.22898) | SelFusion通过双向自我蒸馏和动态掩码策略，无需外部教师模型即可提升扩散语言模型的生成质量。 |
| [^70] | [AraDetox: A Multi-Dialect Arabic Detoxification Dataset](https://arxiv.org/abs/2608.22894) | 该论文提出了AraDetox，一个覆盖四种阿拉伯方言的大规模去毒化数据集，并证明去毒化是一种在保留语义的同时进行词汇和结构重构的改写任务。 |
| [^71] | [Proxy reliance in large language model decisions is uncalibrated to predictive evidence](https://arxiv.org/abs/2608.22887) | 本文通过精确计算临床任务中的因果代理效应，发现LLM对代理的依赖未与预测证据校准，表现为过度依赖、合理依赖或不足依赖，且社会标签抑制效果脆弱。 |
| [^72] | [Better Retrieval, Worse Robustness:How Multi-hop RAG Amplifies Upstream ASR Errors](https://arxiv.org/abs/2608.22872) | 这项研究发现，多跳RAG中的实体图链接和迭代重述扩展会放大上游ASR错误，相比朴素密集检索，其性能差距在口音变化下增加36-67%，且查询实体损坏是主要失败原因。 |
| [^73] | [WARP: Wasserstein-Aligned RAG for Population Opinions](https://arxiv.org/abs/2608.22859) | WARP通过Wasserstein距离校准RAG检索结果，以恢复被标准检索忽视的少数意见，从而更准确地反映群体意见分布。 |
| [^74] | [SAVER: Selective Auditing of Verbal Evidence for Error Recovery in VLM Change Reasoning](https://arxiv.org/abs/2608.22857) | SAVER通过解析VLM输出中的言语证据并仅在证据缺失时触发重新提示，显著提升了视觉变化推理的准确性，最高提升达+25.8%。 |
| [^75] | [Same Agent, Different Answers: A Repeat-Aware Audit of Corpus-Induced Answer Churn in Retrieval-Augmented QA](https://arxiv.org/abs/2608.22856) | 本文提出了一种快照兼容性审计方法，揭示检索增强问答系统在索引扩展后存在被总体准确性掩盖的答案波动，即使模型和提示等设置不变，超额波动仍显著。 |
| [^76] | [Your AI, On a Dial: Controlling Investment Bias in LLMs with a Single Neuron](https://arxiv.org/abs/2608.22852) | 本文提出一种通过干预单个神经元来连续调节大语言模型整体投资倾向的“投资偏见旋钮”，无需修改提示或参数，即可单调改变投资决策和理由。 |
| [^77] | [Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports](https://arxiv.org/abs/2608.22817) | 本文提出了一个端到端框架，从工业技术报告中自动生成高质量的指令调优和基准问答数据集，填补了该领域无公开数据集的空白。 |
| [^78] | [DIAG: Diagnostic Iterative Alignment and Generation for Data-Efficient Mathematical Preference Distillation](https://arxiv.org/abs/2608.22806) | DIAG通过诊断式迭代对齐与生成框架，自适应调整练习分布并聚焦于学生能力边界，从而在数学偏好蒸馏中提升数据效率。 |
| [^79] | [SDoH-Aware Narrative Anchoring Bias in Medical LLMs for Trustworthy Clinical Decision Support](https://arxiv.org/abs/2608.22802) | 本文提出并评估了医学语言模型中的SDoH感知叙事锚定偏差，通过构建反事实数据集NarrativeShield SDoH MedQA，发现模型在相同病例但不同患者叙事下会改变响应，揭示了临床决策支持中的潜在不可靠性。 |
| [^80] | [TRACE: A Self-Evolving Skill Bank for Consistent, Limit-Aware LLM Agents](https://arxiv.org/abs/2608.22793) | TRACE通过构建自我进化的技能库，在不修改模型权重的情况下，提升LLM代理在重复任务中的一致性和限制意识，弥合了单次成功与一致成功之间的可靠性差距。 |
| [^81] | [SPOC-SQL: Stage-wise Preference Optimization for Controllable Text-to-SQL](https://arxiv.org/abs/2608.22772) | SPOC-SQL通过将文本到SQL分解为四个顺序子任务，并在各阶段关键决策点进行细粒度偏好优化，实现了对中间生成过程的可控性和结构化决策增强。 |
| [^82] | [DelistBench: Evaluating Search-Enabled LLMs for Auditable Corporate-Event Database Completion](https://arxiv.org/abs/2608.22770) | 该论文提出了DelistBench基准和Search-to-Record任务，证明网络搜索能显著提升LLM在公司事件数据库补全中的准确率，且经济型系统能以低成本达到接近最优性能。 |
| [^83] | [Don't Repeat Yourself: Stopping Verbatim Loops at Sampling Time](https://arxiv.org/abs/2608.22761) | 本文提出DRY方法，通过采样时对延续上下文中已有片段的词元进行惩罚，有效减少大型语言模型的逐字循环，提升生成多样性且不影响格式和流畅性。 |
| [^84] | [XTC: Head-Aware Sampling by Excluding Top Choices](https://arxiv.org/abs/2608.22758) | XTC通过排除高概率的首选词，仅保留最弱的合理替代，从而在开放生成中有效提升多样性并减少重复。 |
| [^85] | [Beyond Factual Knowledge: Benchmarking and Learning Step-Level Procedural Rule Reasoning in Large Language Models](https://arxiv.org/abs/2608.22753) | 本文提出了RuleWorld基准和DynaRule框架，通过将程序性规则转化为可学习的逐步注意力过程，使LLM能动态关注和更新规则，从而提升规则推理的稳定性和准确性。 |
| [^86] | [DiaRelay: Relaying Dialogue Context with a Constant-Size Memory for Emotion Recognition in Conversation](https://arxiv.org/abs/2608.22745) | 本文提出DiaRelay，一种基于LoRA的轻量级适配器，通过恒定大小记忆显式中继对话上下文，以解决现有方法中窗口大小与计算成本矛盾及缺乏对话级状态的问题，从而提升对话情感识别准确性。 |
| [^87] | [A Source-Grounded Framework for Constructing and Evaluating Progressive Multimodal Diagnostic Dialogues from Clinical Case Reports](https://arxiv.org/abs/2608.22713) | 本文提出了一种基于来源的框架，可从临床病例报告自动构建渐进式多模态诊断对话，并实现对MLLMs在诊断推理和影像解释上的精准评估，显著优于现有前沿模型的表现。 |
| [^88] | [WnW: Waxing-and-Waning KV Cache for Long-Form Speech LLMs](https://arxiv.org/abs/2608.22704) | WnW通过将KV头分为锚定、潮汐和固定三类角色，结合GPU保留与CPU召回机制，在长语音LLM中实现了接近全缓存精度的KV压缩。 |
| [^89] | [Enrich-Retrieve-Rank: Scaling Capability Discovery Beyond In-Context Routing](https://arxiv.org/abs/2608.22695) | 本文提出了一种“富集-检索-排序”的方法，通过离线富集元数据和在线检索排序，显著提升了大规模智能体组件发现能力，克服了传统上下文路由在规模扩大时的性能崩溃问题。 |
| [^90] | [Iteration Without Elaboration: A Simple ReAct Architecture Suffices for Text-to-SQL Generation](https://arxiv.org/abs/2608.22651) | 本文提出ReAct-SQL，一个仅基于迭代推理和受限DSL动作空间的简单零样本框架，在匹配复杂基线性能的同时实现高达8倍的速度提升。 |
| [^91] | [GeoRisk-RAG: A Hierarchy-Aware Risk Framework for Improving RAG Reliability through Selective Answering](https://arxiv.org/abs/2608.22634) | GeoRisk-RAG通过层级感知的有向无环图框架，在RAG中引入选择性回答机制，显式区分地理适用性，以降低自然灾害管理中的错误风险。 |
| [^92] | [Teaching LLMs How ICU Physicians Approach Clinical Reasoning Through OMOP-Aligned Retrieval Improves Reasoning Across Clinical Domains](https://arxiv.org/abs/2608.22622) | 本文提出ICU-REACT数据集和Clin-REACT模型，通过临床医生参与的框架训练LLM掌握ICU临床推理，实现跨临床领域的推理能力提升。 |
| [^93] | [Vision-Language Models for Occupational Physical Exposure Assessment: Estimating External Hand Forces in Manual Material Handling Tasks from RGB Video](https://arxiv.org/abs/2608.22586) | 本文提出一种基于视觉-语言模型的流程，利用文本提示、视觉特征和箱体质量，从RGB视频中无需专用传感器即可估算手工物料搬运中的动态外部手力，并通过多视角交叉验证证明了其有效性。 |
| [^94] | [Hybrid Panels: Toward Human-AI Collaboration in Survey Research](https://arxiv.org/abs/2608.22582) | 本文提出一种混合面板框架，通过迭代优化大型语言模型与目标人群的对齐，利用误差反馈指导调查设计，以克服传统调查的挑战并促进人机协作。 |
| [^95] | [From Diagnosis to Redesign: Using Quantitative Ethnography to Improve Multi-Agent LLM Reasoning](https://arxiv.org/abs/2608.22566) | 本文提出了一种基于量化民族志和认知网络分析的新方法，用于诊断多智能体大语言模型系统中的推理缺陷，并通过分析智能体交互话语来指导系统重构，从而提升任务输出与目标的对齐性。 |
| [^96] | [ExecRubrics: Executable Tool-Augmented Rubrics for Verifiable and Efficient Long-Form Evaluation](https://arxiv.org/abs/2608.22559) | ExecRubrics通过将评分标准转化为可执行的Python函数，实现了可验证、高效且能捕捉复杂依赖关系的长篇评估，替代了昂贵的黑盒LLM评判器。 |
| [^97] | [BLADE: Bilevel Low-rank Augmented-Lagrangian Erasure for LLM Unlearning](https://arxiv.org/abs/2608.22557) | BLADE通过钳制熵、不对称增广拉格朗日和LoRA双层结构，实现了对LLM遗忘过程的平滑控制，显著提升了鲁棒性和性能。 |
| [^98] | [TRACE: Temporal Retrieval with Anchored and Convergent Evidence for Long-Horizon Video Understanding](https://arxiv.org/abs/2608.22516) | 本文提出了VES-Bench基准和TRACE方法，通过审计解码帧是否覆盖所有必要证据区间，并采用无训练代理逐步构建证据包直至答案稳定，以提升长视频理解的证据支持可靠性。 |
| [^99] | [Kernel Token Contradiction: a Fast and Principled Approach for LLM Claim Uncertainty Quantification](https://arxiv.org/abs/2608.22506) | 本文提出了一种轻量级且快速的声明级不确定性量化方法KTC，通过核表示和冯·诺依曼熵，在CPU上实现了比现有方法显著的速度提升。 |
| [^100] | [Who Pays More for Safety? Measuring the Disparate Cost of Safety Alignment across Languages](https://arxiv.org/abs/2608.22490) | 这项研究揭示了安全对齐在不同语言间施加不平等成本，非英语用户系统性承担更高实用性损失，并识别出三重不平等模式。 |
| [^101] | [Claim-Level Confidence Calibration for Reliable Decision Making with Large Language Models](https://arxiv.org/abs/2608.22483) | 本文提出了一种声明级置信度校准框架，在封闭箱设置下无需微调即可为每个可验证声明分配校准置信度，从而实现选择性干预，提升大语言模型在决策中的可靠性。 |
| [^102] | [GTA-RAG: Graph-Trajectory-Augmented Reinforcement Learning for Multi-Turn Retrieval-Augmented Reasoning](https://arxiv.org/abs/2608.22479) | GTA-RAG通过从实体-文档图中采样并验证多跳问答轨迹，提供轨迹级监督信号，从而改进多轮RAG中强化学习的稀疏奖励问题。 |
| [^103] | [Small Reasoning Models are Instruction Followers in Function Calling](https://arxiv.org/abs/2608.22472) | 本文提出IFFC框架，将函数调用任务从主模型分离并交给小型模型，在指令跟随模式下显著提升函数调用准确性，尤其在推理型LLMs上表现突出。 |
| [^104] | [From Exposure to Expectation: Frequency, Surprisal, and Language Across Development in Spanish](https://arxiv.org/abs/2608.22452) | 这项研究表明，在西班牙语中，词汇频率是预测儿童习得年龄的主要因素，而意外性（基于上下文的可预测性）在频率和词长之外几乎没有额外贡献，尽管意外性能预测成人阅读时间。 |
| [^105] | [Figurative Justice: Detecting metaphors in Hindi judgements with qualitative assessment and transformers](https://arxiv.org/abs/2608.22446) | 本文首次尝试在低资源语言印地语的法律判决中检测隐喻，结合定性评估和变换器模型，填补了该领域的研究空白。 |
| [^106] | [Aligned Alone, Misaligned Together: Forecasting Adversarial Capture in LLM Agent Populations](https://arxiv.org/abs/2608.22444) | 该论文发现单个LLM智能体的校准行为无法预测其群体行为，但通过良性操作数据可校准响应函数，从而提前预测对抗性俘获。 |
| [^107] | [Rank Reversal in Multilingual LLM Judges: A Label-Free Double-Centering Calibrator](https://arxiv.org/abs/2608.22432) | 本文提出了一种无标签的双中心校准方法（CBC），用于校正多语言LLM裁判中因语言差异导致的排名反转，并提供了理论保证。 |
| [^108] | [All four leading LLMs talk more than they listen to personality-verified synthetic help-seekers](https://arxiv.org/abs/2608.22425) | 该研究通过性格感知的合成对话评估发现，四种主流大语言模型在急性危机场景中无法有效区分情绪稳定化能力，且普遍倾向于“说”而非“听”，其评估可超越五因素模型扩展到动机与调节特质。 |
| [^109] | [LLMs for Survey Text Analysis - A Performance Comparison Between Humans and GPT-5 on Inductive Content Analysis](https://arxiv.org/abs/2608.22417) | 本研究通过比较人类与GPT-5在归纳内容分析中的表现，发现LLM与人类在编码和主题生成上具有中等对齐度，表明LLM可作为定性研究中的辅助工具，但需注意变量间的差异性。 |
| [^110] | [Don' t Box Me In: Dynamic Cultural Adaptation and Cognitive Tracking for Social Understanding](https://arxiv.org/abs/2608.22411) | 本文提出一种无需训练的框架DyCAC，通过将文化偏好建模为动态混合参考并持续追踪认知，使大型语言模型能灵活适应多元文化社交情境，克服了静态文化建模的局限。 |
| [^111] | [SchemaGUI: A Schema-Driven Benchmark for Controllable GUI Generation Evaluation](https://arxiv.org/abs/2608.22390) | SchemaGUI通过模板化合成确定性标注任务，为可控GUI生成提供可靠评估基准，并揭示了几何空间控制是当前模型的主要瓶颈。 |
| [^112] | [ProBel: Propaganda Detection with Techniques, Spans, and Explanations](https://arxiv.org/abs/2608.22388) | 本文提出ProBel资源，通过一个双语多任务模型在阿拉伯语和英语中统一宣传检测的多层次任务，并实现最佳性能。 |
| [^113] | [GRAFT: Graph-Distilled Generative Retrieval for Facet-Aware Scientific Literature Exploration](https://arxiv.org/abs/2608.22381) | 本文提出GRAFT方法，通过图蒸馏将论文间的多面性关系（问题、方法、结果、贡献）编码到生成式检索器中，并解决朴素蒸馏的覆盖率不足问题，从而支持面向面性的探索性科学文献检索。 |
| [^114] | [Can Large Language Models "Hyper-Thread"?](https://arxiv.org/abs/2608.22376) | 本文提出模型超线程假设，证明在串行生成中并发加载多个任务可提升准确性，挑战了注意力分散仅代表干扰的传统观点。 |
| [^115] | [Context-Aware Cluster Decoding: Semantic Anchor-Driven Coherence in dMLLMs](https://arxiv.org/abs/2608.22367) | 本文提出一种无需训练的上下文感知聚类解码方法，通过结合置信度与邻居邻近度评分并采用无块设计，解决扩散多模态大语言模型长输出中的语义漂移和重复问题。 |
| [^116] | [Where Cognition Lives: Dissecting Emergent from Computed Function in a Minimal Complete Cognitive Architecture](https://arxiv.org/abs/2608.22347) | 本文通过最小完整认知架构剖析，发现能力和停止行为是涌现的，但事后观察的收益提升经不起审计，且停止的优势源于读出机制而非原生功能。 |
| [^117] | [When Not to Imitate: Boundary-Aware Skill Memory for Reliable Tool-Use LLM Agents](https://arxiv.org/abs/2608.22339) | 本文提出边界感知技能记忆（BASM），通过为技能添加适用条件、风险提示等边界字段，避免LLM代理陷入“技能模仿陷阱”，从而提升工具使用任务的可靠性和泛化能力。 |
| [^118] | [Register Shifts Break LLM Safety: A Bengali Benchmark with Culturally Grounded Harms](https://arxiv.org/abs/2608.22335) | 该研究通过孟加拉语基准测试发现，语域转换（如正式风格）比语言切换更能突破LLM安全防线，显著提高有害请求的成功率。 |
| [^119] | [Mechanistic Interpretability of Chain-of-Thought Reasoning via Sequential Activation Patching](https://arxiv.org/abs/2608.22332) | 本文提出了一种序列激活修补框架，用于追踪和聚合跨令牌位置的CoT相关注意力头激活，从而揭示思维链推理中因果效应的时空分布机制。 |
| [^120] | [Noise Floor Audit for Agent Benchmarks](https://arxiv.org/abs/2608.22331) | 本文通过审计发现，在代理基准测试中，语义保持的提示扰动比重新运行带来更大的噪声底限，且失败模式差异显著，凸显了边际准确性指标的局限性。 |
| [^121] | [Semantics or Structure? Auditing Text Sensitivity in Multimodal Time-Series Forecasting](https://arxiv.org/abs/2608.22321) | 该论文通过扰动实验发现，多模态时间序列模型对文本语义不敏感，其性能提升主要来自附带数值列，而非文本内容。 |
| [^122] | [Text-Anchored Semantic Perturbations for Transferable Jailbreak Attacks on Multimodal Large Language Models](https://arxiv.org/abs/2608.22312) | 本文提出了一种黑盒越狱攻击框架，通过文本锚定语义分解和语义保持增强，生成可迁移的扰动，有效攻击多模态大语言模型。 |
| [^123] | [LLM Evaluation on Unseen Questions: Contextual Multidimensional IRT Model](https://arxiv.org/abs/2608.22295) | 本文提出一种结合问题情境的多维项目反应理论评估框架，利用问题嵌入和潜在能力剖面，实现对未见问题LLM表现的准确预测，优于无模型基线。 |
| [^124] | [Length-Adaptive Decoding for Masked Diffusion Machine Translation](https://arxiv.org/abs/2608.22274) | 本文提出熵谷（EV），一种无需训练的长度选择器，通过预测熵评估候选目标画布，有效提升掩码扩散机器翻译的长度选择质量，恢复参考长度增益的33%-65%。 |
| [^125] | [Clarify User Expertise: Towards Proactive Conversational Agents Tailoring Responses to User Proficiency](https://arxiv.org/abs/2608.22266) | 本文提出PASSING方法，通过LLM自我博弈驱动的主动询问策略，让对话代理能明确用户专业水平并定制响应，从而提升信息检索中的个性化交互效果。 |
| [^126] | [CAIA in Practice: Field Evaluation of an AI-Assisted Support System for Text-Based Online Counselling](https://arxiv.org/abs/2608.22251) | 本文通过现场评估验证了CAIA系统在文本在线咨询中的实用性，发现专业自主性和信息准确性是AI采纳的关键，且解释性功能最能促进咨询师的专业反思。 |
| [^127] | [N\"urnberg NLP @ GermEval Shared Task 2026: Harmful Content Detection in German Social Media through Error-Independent LLM Voters](https://arxiv.org/abs/2608.22246) | 该论文通过构建跨LLM、训练方法和类别范围三个正交轴的九投票者集成系统，利用错误独立性应对类别不平衡，在GermEval 2026四个子任务中均取得第一名。 |
| [^128] | [Improving Few-Step Language Flows with Untied Self-Conditioning](https://arxiv.org/abs/2608.22244) | 本文发现流匹配语言模型在少步采样时质量下降源于自条件机制的训练-推理不匹配，并利用模型结构推导修正，通过抑制冗余方向来提升生成质量。 |
| [^129] | [Beyond What Meets the Eye: Unveiling Situational Illusions for Multimodal Large Language Models](https://arxiv.org/abs/2608.22232) | 本文提出了“情境错觉”概念，构建了MSIBench基准测试，揭示了多模态大语言模型在视觉观察、定位和推理方面的脆弱性，并提出了系统性缓解策略。 |
| [^130] | [Whitewashing Hate, Smearing Harmless Content: Annotator-Style Rebuttal Attacks on LLM-Based Moderation](https://arxiv.org/abs/2608.22230) | 本研究揭示了标注者风格的反驳攻击能显著破坏LLM仇恨言论审核的准确性，且洗白与污蔑两种操纵方向存在模型特定的不对称效应。 |
| [^131] | [Grounded Normative Rule Generation with Structured Search](https://arxiv.org/abs/2608.22229) | 本文提出GNRS-Search框架，通过MCMC采样优化五槽位与或图，将操作结构从文本生成中解耦，确保规范性规则在执行时具有接地可行性。 |
| [^132] | [Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation](https://arxiv.org/abs/2608.22215) | 该论文提出一种双层代理记忆框架，通过成本感知的写入路由和周期性参数巩固来优化知识生命周期管理，从而提升LLM代理在动态环境中的记忆效率和检索性能。 |
| [^133] | [Mitigating Speaker Leakage in Cascaded Multi-talker ASR with Diarization-based Transcript Correction](https://arxiv.org/abs/2608.22196) | 本文提出一种基于说话人日志的剪枝方法，通过多模态验证三方共识来移除级联多说话人ASR中的说话人泄漏伪影，在多种重叠条件下显著降低cpWER（最高达29%）。 |
| [^134] | [How Agents Represent Humans: Human-Directed Stereotypes in an Open Agent Social Network](https://arxiv.org/abs/2608.22192) | 本研究首次系统性地揭示了LLM代理在开放社交网络中如何通过能力主导的评价和多样化的“其他”归因来表征人类，而非简单复制刻板印象。 |
| [^135] | [Unveiling the Depth-Performance Dilemma in Split-Federated Fine-tuning of LLMs](https://arxiv.org/abs/2608.22188) | 本文首次系统性地揭示了分割联邦微调中深层划分提升系统效率却导致模型性能崩溃的“深度-性能困境”，并验证了现有聚合方法无法克服这一矛盾。 |
| [^136] | [AudioNoisePrints: Model-free audio watermarking using spatial correlation in flow matching TTS](https://arxiv.org/abs/2608.22186) | 提出一种无需训练的水印方法，利用流匹配TTS中初始噪声与输出的空间相关性，实现高效、鲁棒且不损害生成质量的音频水印。 |
| [^137] | [Aggregation-Aware Synthetic Text Generation Against Authorship Re-Identification](https://arxiv.org/abs/2608.22161) | 本文提出了聚合感知的合成文本生成框架，通过捆绑级别联合选择合成文本，有效降低多文本账户在作者身份再识别攻击下的可链接性，同时保持文本质量。 |
| [^138] | [The Collaboration Tax: How Much LLM Multi-Agent Systems Pay to Coordinate](https://arxiv.org/abs/2608.22152) | 本文提出“协作税”概念，量化LLM多智能体协调中的性能损失，发现其源于对话级联缺陷而非推理不足，且与模型能力单调相关。 |
| [^139] | [Lexical Perturbations Disrupt LLM Reasoning: An Empirical Study of Attention Diversion](https://arxiv.org/abs/2608.22140) | 本研究揭示字符级词汇扰动通过破坏子词分词并引发注意力分散，显著降低LLM推理性能，且碎片化与注意力分配耦合使损伤难以逆转。 |
| [^140] | [Measuring Stability and Failure Behavior in Language Models Under Structured Perturbations](https://arxiv.org/abs/2608.22138) | 该论文提出了一个分级的、多系列的失效感知压力测试框架，通过七个扰动系列和严重性阶梯系统性地测量语言模型的稳定性与崩溃点，并在GSM-Symbolic数据集上验证了其有效性。 |
| [^141] | [SSE-Bio: A Structured Self-Evolving Agent with Agentic Retrieval Policy for Multi-Hop Biomedical Reasoning](https://arxiv.org/abs/2608.22132) | SSE-Bio通过结构化状态和可训练代理策略实现自进化检索，解决了多跳生物医学推理中的指令漂移问题。 |
| [^142] | [PropUQ-MAS: Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems](https://arxiv.org/abs/2608.22130) | 本文提出了PropUQ-MAS，一种通过通信图结构捕捉多智能体系统中错误传播的不确定性量化框架，显著提升了可靠性估计性能。 |
| [^143] | [Decoupled Physical Modeling and Execution for Physics Reasoning](https://arxiv.org/abs/2608.22126) | 该论文提出了一种解耦物理建模与执行的统一框架，通过两阶段后训练策略（监督微调加规则反馈强化学习）提炼中间表示，从而提升大型语言模型在物理推理任务上的表现。 |
| [^144] | [LLM assisted writing deserves empirical evaluation](https://arxiv.org/abs/2608.22124) | 通过对大量健康信息学论文的分析，本文主张应基于学术质量和责任性评估手稿，而非将大语言模型辅助写作视为检测问题。 |
| [^145] | [RAG Collapse: LLM Responses Collapse When Retrieved Documents Are Self-Authored](https://arxiv.org/abs/2608.22118) | 本文揭示了当AI系统检索并引用自己生成的文档时会发生“RAG崩溃”，导致响应质量下降，且实验显示大多数情况下（79.6%）会导致崩溃。 |
| [^146] | [TANGO: Token-Aggregated Nonlinear Gating Operators for Natural and Formal Language Modeling](https://arxiv.org/abs/2608.22117) | 本文提出TANGO和WANGO模型，用跨令牌门控残差更新替代标准Transformer的自注意力和前馈网络，在保持性能的同时实现更高效的序列建模。 |
| [^147] | [Semantic Reasoning Denoising: Correcting Language Model Reasoning with Semantic Operators](https://arxiv.org/abs/2608.22090) | 本文提出了一种新的语义推理去噪方法，通过算子化马尔可夫过程显式建模和纠正推理中的语义错误，优于传统扩散或自我修正方法。 |
| [^148] | [W-RAG: Source-Aware Retrieval for Enterprise Document Generation from Heterogeneous Knowledge Bases](https://arxiv.org/abs/2608.22081) | W-RAG通过本体引导检索和每个知识库内的局部排序，解决了企业文档生成中异构知识库全局排序导致的不平衡上下文问题，从而生成更完整的草稿。 |
| [^149] | [Spine-Branch Coordination for Multi-agent Computer Use](https://arxiv.org/abs/2608.22077) | 本文提出脊柱-分支协调框架，通过将任务分解为连续状态的脊柱和并行收集信息的分支，避免虚拟机状态合并，在提升多智能体计算机使用任务成功率的同时大幅降低成本。 |
| [^150] | [Real-TurnTurk: A Multimodal Turkish Corpus for Turn-Taking Prediction](https://arxiv.org/abs/2608.22071) | 本文首次构建了多模态土耳其语自然对话语料库，并利用遗传算法优化可解释规则来预测轮流发言。 |
| [^151] | [Align, Unify, Suppress, Route: A Coherentist View of Transformer Computation](https://arxiv.org/abs/2608.22034) | 本文提出连贯主义概率组合论（CPC），通过四个操作符角色（对齐、统一、抑制、路由）统一描述变压器计算，并在多个模型中验证了其有效性，特别指出抑制角色在跨任务中表现更稳定。 |
| [^152] | [The Communication Map of a Transformer](https://arxiv.org/abs/2608.22007) | 提出了一种从权重出发绘制变压器所有潜在通信通道的“通信图谱”方法，能高效计算并揭示大多数注意力头对的耦合或回避模式，且具有广泛适用性。 |
| [^153] | [Machine learning and digital pragmatics: Which word category influences emoji use most?](https://arxiv.org/abs/2608.21975) | 本研究通过MARBERT模型和逻辑回归分析发现，在口语阿拉伯语社交媒体帖子中，动词类别对表情符号使用的影响最强，尽管名词在频率上占主导。 |
| [^154] | [ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents](https://arxiv.org/abs/2608.21969) | 本文提出一种两级层次强化学习框架，结合话语级策略抽象与词元级解码，并引入双粒度奖励机制，以提升对话代理在复杂交互中的性能。 |
| [^155] | [Bulbul: A Dataset for Dialectal Arabic Speech Recognition](https://arxiv.org/abs/2608.21950) | 本文介绍了BULBUL，一个覆盖11国275名说话者的多方言阿拉伯语ASR数据集，通过结构化方言覆盖和两级人工验证确保质量，并为现代方言ASR建立了强基线。 |
| [^156] | [EDGE: Experience-Distillation for Guided Exploration in Agentic Reinforcement Learning](https://arxiv.org/abs/2608.21946) | EDGE框架通过将检索到的经验作为临时训练支架并逐步蒸馏到策略参数中，实现无需额外采样即可持续提升智能体强化学习性能。 |
| [^157] | [SkillBloat: Token Amplification Attacks via Skill Injection in LLM Coding Agents](https://arxiv.org/abs/2608.21929) | 本文提出SkillBloat框架，通过恶意技能注入使LLM编码代理消耗远超所需的令牌，实现经济资源滥用攻击，在真实基准上达到5.42至10.15倍的令牌放大。 |
| [^158] | [GuardianBench: A Same-Scene Instruction-Contrastive Benchmark for Latent Contextual Risk in Embodied AI](https://arxiv.org/abs/2608.21928) | 本文提出GuardianBench，一个基于国际安全标准的同场景指令对比基准，通过3,024个示例揭示视觉语言模型在具身AI中因无法绑定指令相关信息而导致潜在情境风险判定失败，平均配对准确率仅24.1%。 |
| [^159] | [Modeling Claim Dependency Structure for Patent Litigation Prediction with Graph Attention Networks](https://arxiv.org/abs/2608.21924) | 本文提出ClaimGAT，利用图注意力网络对专利权利要求进行独立编码并建模其依赖关系，显著提升了专利诉讼预测的准确性。 |
| [^160] | [BanglaVeilGuard: Cross-Script Safety Benchmarking and Lightweight Guardrails for Bangla Large Language Models](https://arxiv.org/abs/2608.21880) | 本文提出BanglaVeilGuard，一种针对六种孟加拉语变体的轻量级防护栏，通过非破坏性归一化和风险分类门控，在无需修改模型权重的情况下将攻击成功率从93.8%降至100%，显著提升孟加拉语大语言模型的安全性。 |
| [^161] | [The Chase Is the Curriculum, the Capture Anchors the Credit: Pursuit-Evasion Self-Play for Zero-Data LLM Reasoning](https://arxiv.org/abs/2608.21871) | 本文提出LURE框架，将零数据自我对弈建模为追逃博弈，通过捕获前沿奖励学习任务定位策略，使LLM推理训练无需人工数据即可自适应难度。 |
| [^162] | [MemGuard: Persisting Verifier Signals for LLM-Agent Memory Governance](https://arxiv.org/abs/2608.21867) | MemGuard通过将验证器输出转化为持久化生命周期元数据，解决了LLM智能体记忆中的不可靠准入和记忆漂移问题，确保记忆在长期交互中保持可靠。 |
| [^163] | [HiDiffTIR: Hierarchical Difficulty-Aware Policy Optimization for Multi-Turn Tool-Integrated Reasoning](https://arxiv.org/abs/2608.21863) | 本文提出HiDiffTIR框架，通过分层难度感知的信用分配机制，在多轮工具集成推理中更精确地区分轨迹和推理步骤的难度，从而提升强化学习训练效果。 |
| [^164] | [PUMA: A Polish Benchmark for Culturally Grounded Multimodal Understanding](https://arxiv.org/abs/2608.21853) | 本文提出了一个名为PUMA的波兰语多模态基准，包含900个手工任务，评估模型在波兰文化语境下的图像、音频和文本理解能力，并揭示了当前模型在该领域存在的显著性能差距。 |
| [^165] | [GameXpert-Bench: How Far Are Coding Agents from Expert Game Development?](https://arxiv.org/abs/2608.21833) | 本文提出了GameXpert-Bench基准，通过三个互补轨道（初始生成、缺陷修复、多轮优化）全面评估编程代理在游戏开发全生命周期中的能力。 |
| [^166] | [GUI-Primitives: Diagnosing Spatial Reasoning Failures in Vision-Language GUI Grounding](https://arxiv.org/abs/2608.21832) | 该论文提出了一个名为GUI-Primitives的基准测试，通过对比指令对系统性地诊断视觉-语言模型在GUI空间关系推理中的缺陷，发现现有模型在严格定位准确率上表现极差，且多数预测完全偏离候选区域。 |
| [^167] | [Training a Knowledge Base: Supervised Structure Learning for Agent-Curated Document Stores](https://arxiv.org/abs/2608.21829) | 本文提出将知识库视为可训练模型，通过监督式（问题，答案）标签指导代理编辑文档库结构，实现更高效、更准确的检索增强生成。 |
| [^168] | [Do Large Language Models Perform Well on Comprehending Poetic Logic in Modern Chinese Poetry?](https://arxiv.org/abs/2608.21827) | 本文提出了首个专门评估现代中文诗歌诗意逻辑的基准Peony，并系统分析了六个主流大型语言模型在这方面的表现。 |
| [^169] | [Convergence in Science, Divergence in Religion: Calibrated Framing Differences Across Wikipedia's Language Editions](https://arxiv.org/abs/2608.21821) | 本文提出一种校准距离方法，通过减去语言对间的基线偏差，揭示了维基百科不同语言版本在科学概念上框架趋同，而在宗教概念上框架差异显著。 |
| [^170] | [PatchGate: Narrowing the Verbalization Gap with Intrinsic Object Inventories in Frozen Vision-Language Models](https://arxiv.org/abs/2608.21819) | 该论文提出PatchGate框架，通过从冻结视觉语言模型中提取内在对象清单，在生成前缩小可见对象与最终描述之间的差距，从而提升图像描述的精确性和完整性。 |
| [^171] | [MCite-RL: Towards Reliable Multimodal RAG via Citation-enhanced Agentic Reinforcement Learning](https://arxiv.org/abs/2608.21808) | 本文提出了MCite-RL，一个通过智能体迭代检索和引用增强奖励机制，将视觉引用转化为动态证据驱动过程，从而提升多模态RAG可靠性的强化学习框架。 |
| [^172] | [More Computational Resources Do Not Ensure Higher Scholarly Impact: Evidence from Leading NLP Conference Papers](https://arxiv.org/abs/2608.21806) | 本研究基于13,921篇NLP顶级会议论文，发现计算资源集中度远高于学术影响力集中度，表明更多计算资源并不必然带来更高的学术影响。 |
| [^173] | [Lexical Coupling in GUI Element Grounding: Sentence Embeddings Track Labels across Mobile and Web](https://arxiv.org/abs/2608.21794) | 该论文揭示GUI元素定位评估中，嵌入相似度常被可见标签恢复所混淆，而非真正的语义理解，并强调需引入词汇基线和标签分层来改进评估可靠性。 |
| [^174] | [No One Model Catches Every Harm: Benchmarking Content Moderation Across Safety Scenarios](https://arxiv.org/abs/2608.21775) | 该研究通过大规模基准测试发现，没有单一模型能在所有安全场景中表现最佳，大型前沿模型和专用小型模型各有优劣，揭示了现有内容审核的盲点。 |
| [^175] | [Evaluation Awareness in Language Models: Representation, Verbalization, and Control](https://arxiv.org/abs/2608.21766) | 本文系统研究了语言模型中的“评估意识”现象，发现模型能在激活空间中表征被评估状态，并通过输出言语化及因果引导影响其行为。 |
| [^176] | [Learning to Look Again: Loss-Gap Supervision for Free-form Crop Routing in Vision-Language Models](https://arxiv.org/abs/2608.21762) | 本文提出GapSight框架，通过利用目标模型自身的失败信号（损失差距）生成监督标签，训练轻量级路由器，使视觉语言模型在全局浏览后智能选择自由形式区域进行局部重读，以低成本提升细节问题的回答准确性。 |
| [^177] | [FCPRAG: Fusion-Controller Parametric Retrieval-Augmented Generation for Stable Multi-Passage LoRA Injection](https://arxiv.org/abs/2608.21750) | FCPRAG通过引入轻量级控制器，实现了检索条件化的样本级适配器融合，解决了多段落LoRA注入中的证据融合瓶颈，提高了稳定性和选择性。 |
| [^178] | [Architecture as Capability Equalizer for Coding Agents](https://arxiv.org/abs/2608.21747) | 本论文通过对照实验发现，架构规范格式对编码代理生成代码质量的影响依赖于模型能力，对较弱模型使用代码邻近格式可显著缩小与强模型的能力差距。 |
| [^179] | [L\"etzCross: A Cross-Lingual Page-Level Benchmark for Multimodal Retrieval over Luxembourgish Documents](https://arxiv.org/abs/2608.21714) | 本文提出了一个针对卢森堡文档的跨语言页面级检索基准，并证明页面图像检索器在低资源跨语言场景下优于OCR文本检索器，且微调可跨语言迁移，法语表现最佳。 |
| [^180] | [The Plan, Not the Decoder: Diagnosing and Repairing Compositional Failure in Reasoning-Augmented Text-to-Image Generation](https://arxiv.org/abs/2608.21713) | 本文通过验证计划与解码的可分离性，发现推理增强文本到图像模型的组合失败主要源于计划错误而非解码器不忠实，并提出了可靠的几何评分方法以准确诊断和修复此类问题。 |
| [^181] | [From Association to Causation: Improving Retrieval Precision of Retrieval-Augmented Generation via Causal Relations and an Attention Mechanism](https://arxiv.org/abs/2608.21702) | 该论文提出通过因果图建模检索阶段，利用对撞子结构中的注意力机制，无需训练即可提升RAG检索精度，解决相似度仅捕捉关联而忽略因果的问题。 |
| [^182] | [Measuring Activation Control in Large Language Models](https://arxiv.org/abs/2608.21664) | 我们提出了激活可控性基准，首次系统量化了大型语言模型通过自然语言指令控制自身残差流激活的能力，并证明这种控制可部分规避现有激活监控方法。 |
| [^183] | [Mitigating Database Leakage in RAG Systems with Keyword-Grounded Fact Substitution](https://arxiv.org/abs/2608.21656) | 提出KFS-RAG防御方法，通过关键词识别和事实替换净化检索上下文，有效缓解RAG系统中的数据库泄露风险。 |
| [^184] | [Can LLMs Truly Forget? Revealing Unlearning Gaps Through Adversarial Evaluation](https://arxiv.org/abs/2608.21606) | 本文通过引入攻击成功率（ASR）指标和八种对抗性攻击套件，揭示了大语言模型在标准遗忘评估下看似已遗忘的信息仍可通过策略性提示被显著恢复，表明现有遗忘方法存在对抗性鲁棒性差距。 |
| [^185] | [K-Bench: measuring model performance on real scientific agent requests](https://arxiv.org/abs/2608.21601) | 本论文提出K-Bench 01，一个基于真实科学请求的评估框架，发现当前前沿模型在满足领域科学家接受标准上均未达到阈值，其中gpt-5.6-sol表现最优但仍有不确定性。 |
| [^186] | [Evidence-State Reliability Under Controlled Degradation: Parser-Validity Divergence in a Multi-Stage LLM Pipeline](https://arxiv.org/abs/2608.21559) | 本文提出证据状态可靠性（ESR）作为独立于解析器有效性的评估层，用于衡量多阶段LLM流水线中中间证据在受控退化条件下的完整性、一致性和可用性。 |
| [^187] | [Automating Multi-Hop RAG Evaluation via TRIAD: From Context Extraction to Validated Dataset Generation](https://arxiv.org/abs/2608.21558) | 本文提出了TRIAD，一种三阶段自动化方法，用于生成特定领域的多跳问答数据集，以支持RAG系统评估，并验证其与现有基准数据集具有相似质量。 |
| [^188] | [Forgotten in Weights, Recovered by Tools: Agentic Tool Unlearning for LLM Agents](https://arxiv.org/abs/2608.21544) | 本文提出了一种两阶段框架ATU，通过结合参数性知识遗忘和轨迹级强化学习，有效抑制LLM智能体通过工具恢复被遗忘信息的能力，同时保持正常工具使用。 |
| [^189] | [Beyond Sparse Weights: When Is Attention Compressible?](https://arxiv.org/abs/2608.21541) | 该论文提出CertKV，一种无需训练的注意力压缩方法，通过保留尾部摘要和基于值分散的分配策略，在多个基准测试中实现了高效压缩，表明注意力可压缩性取决于质量、值和上下文分布，而非仅权重稀疏性。 |
| [^190] | [DamageScope: Vision-Language Retrieval at Scale for Disaster Damage Assessment from Satellite Imagery](https://arxiv.org/abs/2608.21529) | 本文提出DamageScope框架，利用检索增强生成技术结合卫星图像与视觉-语言模型，通过多向量嵌入实现大规模灾害损毁的自动化交互式评估。 |
| [^191] | [CyrillicQA: The Influence of Phonetically Encoded Secret Language on LLM Performance](https://arxiv.org/abs/2608.21462) | 本文探讨了大型语言模型在解码语音编码秘密语言（如西里尔字母音译）时的能力，并评估其创造性和抽象推理水平。 |
| [^192] | [Evaluating Multimodal Narrative Understanding of Popular Hollywood Films](https://arxiv.org/abs/2608.21430) | 本文构建了一个基于票房热门和公有领域状态的好莱坞电影新集合，并开发了一个多模态问答基准，以评估语言模型对电影叙事的理解能力。 |
| [^193] | [Agentic Security: A Systematization of Tools, Failure Modes, and Design Laws for LLM-Driven Penetration Testing](https://arxiv.org/abs/2608.21423) | 本文通过系统化评估十种安全工具，提出了四维集成摩擦指数和定量规律，揭示了LLM驱动的渗透测试中故障模式的本质，并建立了设计法则。 |
| [^194] | [Mitigating Bias in Large Vision-Language Models via Counterfactual Ensemble Decoding](https://arxiv.org/abs/2608.21415) | 本文提出了一种反事实集成解码框架，通过在视觉表示空间中构建并集成多群体反事实视角，有效缓解了大型视觉-语言模型中的社会偏见，从而促进公平行为。 |
| [^195] | [Sycophants in the Courtroom: Are LLMs Fragile to Juridical Authority and Evolving Legal Standards?](https://arxiv.org/abs/2608.21409) | 该论文发现法律与医学领域存在显著差异，LLMs在法律任务中对权威来源和时间有效性敏感，表现出明显的脆弱性。 |
| [^196] | [Generative Gap Filling](https://arxiv.org/abs/2608.21401) | 本研究通过实验发现，大型语言模型能从合同剩余文本中恢复被遮蔽的协商条款，准确率高达约90%，远超人类预测，挑战了传统法律填补合同空白方法的假设。 |
| [^197] | [A Social Media Analysis of Discourse on the Israel--Palestine Conflict on Telegram](https://arxiv.org/abs/2608.21385) | 本研究首次大规模系统比较了Telegram上亲以色列与亲巴勒斯坦社区的政治话语，提出微调BERTweet模型在立场检测中显著优于无标签方法。 |
| [^198] | [Beyond Two Bytes per Letter: Tokenization Overhead in Cyrillic AI Systems](https://arxiv.org/abs/2608.21384) | 本文量化了西里尔字母语言（如乌克兰语）在多语言分词器中的分词开销，并提出LLMLingua-2等缓解策略，能显著减少输入长度。 |
| [^199] | [PersonaMem-v3: Toward Omni-Platform Personal Intelligence for Holistic User Understanding, Recommendation, and Agentic Tasks](https://arxiv.org/abs/2608.21381) | PersonaMem-v3 提出了一个基于百万级真实匿名数据的全平台个人智能基准，用于评估跨情境用户理解、可引导推荐、跨平台主动行为及过度个性化的避免。 |
| [^200] | [Agentic Scaffolding Amplifies Sycophantic Behavior in Large Language Models](https://arxiv.org/abs/2608.21377) | 本文发现代理式交互脚手架（如多轮反馈和迭代细化）会系统性放大LLM的谄媚行为，导致平均准确率下降6.3%，且更强模型放大效应更显著。 |
| [^201] | [On the Role of Citations in Preference Data](https://arxiv.org/abs/2608.21376) | 本文通过混合效应模型研究科学问答中引用对人类和LLM偏好判断的影响，发现人类偏好多样化但数量少的引用，而LLM的引用偏好虽存在但受数据和模型影响。 |
| [^202] | [Wazobia Eval: A Benchmark for Nigerian Pidgin Emotion Understanding, Sarcasm Detection, and Cultural Reasoning](https://arxiv.org/abs/2608.21369) | 该论文提出了一个针对尼日利亚皮钦语的评估基准，涵盖情感理解、讽刺检测和文化推理，填补了现有基准在文化特定语言理解上的空白。 |
| [^203] | [KSE-Web: An Analysis of Hybrid Retrieval and LLM-Assisted Query Expansion for Low-Resource Khmer Semantic Search](https://arxiv.org/abs/2608.21365) | 本文提出了KSE-Web，针对低资源高棉语语义搜索，构建了首个清洗后的数据集，并系统评估了多种检索方法，发现传统BM25在性能上优于混合检索和LLM辅助扩展。 |
| [^204] | [Distinguishing Revision and Delayed Elaboration in Incremental Narrative Interpretation](https://arxiv.org/abs/2608.21364) | 本文区分了叙事解释中两种结构不同的更新机制——修订驱动的非单调更新和延迟细化的单调扩展，揭示了它们对增量解释系统结构要求上的根本差异。 |
| [^205] | [ForeDreamer: A Self-Evolving Dual-Agent Memory Architecture for Future Event Prediction](https://arxiv.org/abs/2608.20920) | ForeDreamer通过双智能体架构将原始网络证据转化为结构化记忆，分离事实与经验记忆，从而提升开放网络未来事件预测的准确性。 |
| [^206] | [Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection](https://arxiv.org/abs/2608.20169) | 本文提出Task-CoEvolve方法，通过自适应选择信息量丰富的验证任务并基于部分评估估计完整性能，显著降低LLM提示工程优化中的评估成本，同时保持优化效果。 |
| [^207] | [Multi-Agent Orchestration with the Common-Sense Reasoning Capabilities of LLMs for Autonomous Driving](https://arxiv.org/abs/2608.20129) | 本文提出一种混合自动驾驶框架，通过编排器整合强化学习、PID控制和大语言模型常识推理，并迭代优化奖励函数，以提升在随机场景中的推理能力和安全性。 |
| [^208] | [MileGPO: Milestone Inference with Local Evidence for Graph-Based Policy Optimization of Long-Horizon LLM Agents](https://arxiv.org/abs/2608.19803) | MileGPO通过里程碑发现、可靠性校准塑形和进度对比校准三种机制，从在线回滚中提取过程级信用，有效解决了长时程智能体强化学习中的信用分配难题。 |
| [^209] | [SynFlow: A Multidimensional Diachronic Semantic Analysis Toolkit](https://arxiv.org/abs/2608.19472) | SynFlow是一个开源工具包，通过统一的多维历时分析流程，整合句法、形态、构式和语义特征，以揭示词汇语义变化的具体方面。 |
| [^210] | [Time-Series Retrieval for Grounding Multimodal Language Models in Remaining Useful Life](https://arxiv.org/abs/2608.19218) | 本文提出一种通过时间序列检索检索历史相似退化段，并将其与测试轨迹一起转换为视觉比较工件，从而提升多模态语言模型在剩余使用寿命预测中的性能，实验证实该方法优于非检索基线。 |
| [^211] | [SPADE: Self-Play in Adaptive Synthetic Executable Environments](https://arxiv.org/abs/2608.19197) | 该论文提出SPADE框架，通过单个LLM同时作为环境设计器和推理代理进行自我对弈，动态生成可执行训练环境，以解决语言代理训练中目标分布固定的问题。 |
| [^212] | [Aslema at NADI 2026: Augmentation through Fewshot for SLU](https://arxiv.org/abs/2608.18689) | 本文提出Aslema系统，通过微调优于零样本，并利用大型语言模型生成文化相关的合成数据增强，在NADI 2026任务中在槽位填充上取得第一名。 |
| [^213] | [TraceSQL: Traceable Answerability Estimation for Reference-Free Text-to-SQL Verification](https://arxiv.org/abs/2608.17795) | 本文提出TraceSQL，一种利用67个显式诊断特征的可追踪轻量级验证模型，无需参考即可估计文本到SQL生成查询的答案可能性，克服了现有ORMs和LLM裁判的可解释性不足。 |
| [^214] | [CoAL-RAG: A Complexity-Aware Legal Retrieval-Augmented Generation Method](https://arxiv.org/abs/2608.17536) | 本文提出了一种复杂度感知的法律检索增强生成方法（CoAL-RAG），通过多维评估机制自适应选择检索策略，平衡了简单与复杂法律问题的答案质量和效率。 |
| [^215] | [Decomposition Attacks Across Unlinkable Identities: Limits of Stateful Defenses for LLM Services](https://arxiv.org/abs/2608.17445) | 该论文证明了在攻击者使用不可链接身份的情况下，LLM服务的有状态防御在安全性与实用性之间的权衡完全取决于良性请求的分组方式，且当分组不可区分时无法有效阻止解构攻击。 |
| [^216] | [Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2608.16647) | 同策略蒸馏的泛化行为取决于教师和学生来源关系，同源对能跨域迁移推理能力，跨源对则受限于训练分布，这种双重性既是优势也是风险。 |
| [^217] | [Ask, Condition or Abstain: Reinforcement Learning for Missing-Premise Reasoning](https://arxiv.org/abs/2608.16554) | 本文提出ACA-RL框架，通过数据增强和结构化奖励训练模型在缺失前提时选择提问、条件化回答或弃权，并引入人工验证的MPB基准以提升推理鲁棒性。 |
| [^218] | [DuplexGen: Decoupling Content, Timing, and Acoustics for Synthetic Dialogue Speech](https://arxiv.org/abs/2608.16053) | DuplexGen通过解耦内容、时序和声学特征，利用LLM生成脚本和全双工模型实时交互，使对话时序自然涌现而非预设，实现了更真实、交互驱动的合成对话语音。 |
| [^219] | [TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness](https://arxiv.org/abs/2608.15763) | 本文提出操控系统感知训练（HAT）方法，通过将可进化的操控系统状态纳入训练分布，使数字人代理在实时直播中既能快速响应又能灵活适应动态策略变化。 |
| [^220] | [Grounding Healthcare LLMs in a Causal Knowledge Graph: Framework, Metrics, and a Cardiovascular Pilot](https://arxiv.org/abs/2608.15382) | 本文提出一个基于因果知识图谱的可复现评估框架，通过四种锚定条件和自动化评分，系统性地测试医疗大语言模型在干预决策中的推理能力，并在心血管领域进行了验证。 |
| [^221] | [PatientAct: Theory-Grounded Mental Health Client Simulation](https://arxiv.org/abs/2608.12750) | PatientAct通过整合5Ps临床案例公式和带信任阈值的动态记忆层，解决了LLM模拟客户端过度合作、缺乏因果深度的问题，从而更真实地模拟心理健康咨询中的客户端行为。 |
| [^222] | [Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus](https://arxiv.org/abs/2608.12149) | 本文首次系统研究了混合线性注意力大语言模型中的大规模激活现象，发现了注意力前尖峰和尖峰间平台两种新形态，并揭示了它们与架构配置的关系。 |
| [^223] | [SAG: SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges](https://arxiv.org/abs/2608.12129) | 提出SAG架构，通过事件-实体索引和查询时动态超边连接，在不构建全局知识图谱的情况下，实现了支持结构化约束和多跳推理的检索增强生成。 |
| [^224] | [Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling](https://arxiv.org/abs/2608.11829) | 本研究发现在线策略蒸馏主要提升采样效率而非扩展推理能力边界，其优势在小采样预算下显著，但在大预算下会减弱。 |
| [^225] | [Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases](https://arxiv.org/abs/2608.10503) | 本文提出了一个解析精确的框架，通过全交叉因子实验和精确概率分布计算，取代非结构化基准和蒙特卡洛采样，从而实现对LLM态度与偏见的受控、因果分离式评估。 |
| [^226] | [Withholding the Completing Chunk: Exact Release-Boundary Equivalence for Production Streaming Guardrails](https://arxiv.org/abs/2608.10279) | 本文提出了一种流式语言模型输出的安全监控方法，通过编译策略为持久NFA并区分稳定状态，实现了跨任意分块的精确释放边界等价性，确保检测到禁止模式时无法撤回已释放内容。 |
| [^227] | [Do LLM Recommenders Know When They're Hallucinating? Auditing Confidence Calibration in Catalog Faithfulness](https://arxiv.org/abs/2608.10008) | 本文首次联合审计了多个LLM推荐器的幻觉率和置信度校准，发现目录成员资格测量方法显著影响结果，并验证了更准确的评估工具，揭示了推荐器在输出目录外项目时过度自信的问题。 |
| [^228] | [The Greatness of Science Cannot Be Planned: Agentic Auto-Research is Fuzz Testing](https://arxiv.org/abs/2608.09855) | 本文提出自主自动研究应借鉴灰盒模糊测试原理，通过引入密集的认识进展信号来指导搜索，以克服最终基准优化导致的过拟合和盲目采样问题。 |
| [^229] | [Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA](https://arxiv.org/abs/2608.09819) | Macaron-V1通过混合LoRA架构和递归改进机制，实现了开放环境中的持续学习与自我提升，兼顾适应性和协作性。 |
| [^230] | [Accurate but Natural? Diagnosing Grammatical and Idiomatic Gaps in Japanese EFL Writing](https://arxiv.org/abs/2608.09289) | 本研究通过分层LLM校正流程，首次将日本英语写作中的语法准确性差距与习语性差距分离诊断，发现定冠词和情态动词存在准确性困难，而-ing形式和假设情态动词存在习语使用不足。 |
| [^231] | [Hidden Language Consistency Phenomena in Reasoning LLMs](https://arxiv.org/abs/2608.08447) | 本文揭示了推理大语言模型在多语言任务中语言一致性随难度变化的四种行为模式，并发现了“语言一致性崩溃”效应，即难度增加会导致输出语言突然偏离预期语言。 |
| [^232] | [NL2SHACL-Bench: A Benchmark Suite for Natural Language to SHACL Translation](https://arxiv.org/abs/2608.07530) | 本文提出了NL2SHACL-Bench基准测试套件，用于评估自然语言到SHACL翻译，并发现现有LLMs能生成语法正确的SHACL，但在复杂语义等价约束上表现不足。 |
| [^233] | [ConstructCIE: A Dataset for Extracting Causal Information from Construction Accident Narratives](https://arxiv.org/abs/2608.06495) | 本文介绍了ConstructCIE数据集，用于从施工事故报告中提取分层因果信息，并评估了多种模型，发现它们擅长预测事故类型但难以精确提取因果证据片段。 |
| [^234] | [Sparse PPMI Graph Averaging for Random Indexing Embeddings](https://arxiv.org/abs/2608.05724) | 该论文提出了一种结合稀疏PPMI图平均和稳健缩放的随机索引后处理流程，显著提升了亲属关系类比任务的准确率，从19.41%提高到30.74%。 |
| [^235] | [Answer First, Reason Later: When Commitment Order Costs Accuracy in Diffusion Language Models](https://arxiv.org/abs/2608.05687) | 本文发现扩散语言模型中“答案优先”的承诺顺序（即答案先于推理提交）会降低准确性，而延迟答案提交或限制提交位置可提升性能。 |
| [^236] | [RIG-RoPE: Relation-Stratified Multimodal Attention with Instance-Local Rotary Geometry and Representation-Aware Traversal Coordinates](https://arxiv.org/abs/2608.05154) | 本文提出RIG-RoPE机制，通过关系和实例门控及持续时间感知时间坐标，解决了多模态位置编码中的跨模态空间干扰和时间步长不当问题。 |
| [^237] | [Breadcrumbing Search Agents](https://arxiv.org/abs/2608.04565) | 本文揭示了搜索代理中通过中介搜索界面协调注入受控结果可显著提升攻击成功率的安全漏洞，挑战了仅针对静态注入的现有防护。 |
| [^238] | [Predicting Multilingual Classification and Translation Performance of LLMs with Cross-Lingual Alignment -- Is English Enough?](https://arxiv.org/abs/2608.03446) | 本研究比较了27种跨语言对齐分数，并首次验证它们在预测多语言翻译性能上的有效性，提出了一种新的PMI翻译度量以支持跨语言比较。 |
| [^239] | [Prompt-Induced Waste in Coding Agents: Reasoning, Effort, Harness Design, and End-to-End Cost](https://arxiv.org/abs/2608.01347) | 本论文揭示提示语义、推理努力和框架设计是相互作用的因素，共同决定编码代理的端到端成本与成功率，而非独立可调的控制变量。 |
| [^240] | [What Transfers from Text to Vision? Capability Scaling Laws and Transfer Dynamics for VLMs](https://arxiv.org/abs/2608.00013) | 我们提出了首个跨家族的多模态缩放规律，通过文本能力得分直接预测VLM性能，并在150多个VLM上验证了其有效性。 |
| [^241] | [RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents with Systematic Evaluation](https://arxiv.org/abs/2607.24772) | RSMeM提出了一种知识增强的记忆演化机制，结合层次化知识锚定和失败感知经验精炼，显著提升了遥感智能体在多步骤工具执行中的稳健性。 |
| [^242] | [Two Regimes of Chain-of-Thought Unfaithfulness: Metric-Based Detection Fails Where Models Are Wrong](https://arxiv.org/abs/2607.23458) | 本文发现思维链不忠实性检测存在两种截然不同的模式：在正确答案上行为信号有效，但在错误答案上（不忠实性主要集中地）所有检测方法均失效，且答案正确性本身是最强的预测指标。 |
| [^243] | [CMI-Mem: Toward Generalizable Long-Term Memory Management via CMI-Augmented Reinforcement Learning](https://arxiv.org/abs/2607.20553) | 本文提出CMI-Mem，一种结合外在QA奖励和内在条件互信息奖励的轻量级强化学习记忆管理器，通过逐操作监督实现更高效、更通用的长期记忆管理。 |
| [^244] | [Telco-GAIA: Bilingual Benchmark for Agents in Telecom Domain](https://arxiv.org/abs/2607.20510) | Telco-GAIA是一个双语多模态基准，用于评估电信领域工具使用智能体，通过100个人工验证的问答任务和多跳推理，在异构数据源上挑战模型性能，并采用客观的精确字符串匹配评分。 |
| [^245] | [CANDOR: Chance-Calibrated Discordance in Frozen Foundation Encoders](https://arxiv.org/abs/2607.18451) | 本文提出CANDOR度量，通过等大小对称样本库校正最近邻不一致性，将机会水平固定为二分之一，揭示冻结编码器并非失明但普遍性能较弱。 |
| [^246] | [A JoLT for the KV cache: Near-lossless KV cache compression via joint Lagrangian allocation of Tucker ranks and a rotated residual for llms](https://arxiv.org/abs/2607.12550) | 本文提出JoLT方法，通过部分Tucker分解和旋转低比特残差，在保持头与层轴完整的同时压缩令牌和特征轴，实现KV缓存的近无损压缩。 |
| [^247] | [DiaLLM: An Investigation into the Robustness-Generation Gap in English Dialect Adaptation](https://arxiv.org/abs/2607.07669) | 本文发现方言理解与生成能力在LLM中分离，并证明显式方言定向适应优于广泛对齐，但基准测试无法反映这一生成优势。 |
| [^248] | [DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2607.06507) | DynaKRAG提出了一个统一框架，通过学习状态条件策略动态控制证据获取与答案生成，显著提升了多跳RAG的效率与准确性。 |
| [^249] | [What LLMs explain is not what they believe: Evaluating explanation sufficiency under models' own input beliefs](https://arxiv.org/abs/2606.28615) | 本文提出了一种基于信息论的指标SCSuff，利用LLM自身生成替代输入来评估自由文本解释的充分性，无需预设偏见，并证明解释充分性依赖于输入分布。 |
| [^250] | [How Does Research Evolve? Tracing Cross-Domain Trajectories in NLP, ML, and CV Through Claim-Grounded Typed Citations](https://arxiv.org/abs/2606.22342) | 本文提出了SciTraj，一个包含32,559篇论文和573,126条类型化引用边的语料库，通过主张驱动的六种研究关系和多步轨迹，实现了对NLP、ML和CV领域研究演进的细粒度追踪。 |
| [^251] | [Incumbent Advantage: Brand Bias and Cognitive Manipulation Dynamics in LLM Recommendation Systems](https://arxiv.org/abs/2606.17443) | 本研究揭示了LLM推荐系统中知名品牌的“条件性垄断”现象，并发现权威式营销语言（包括虚假临床声明）能以微小评分点优势打破这种垄断，且不同模型反应各异。 |
| [^252] | [SCAR: Semantic Continuity-Aware Retrieval for Efficient Context Expansion in RAG](https://arxiv.org/abs/2606.16661) | SCAR通过基于查询相关性的自适应阈值和连续性惩罚机制，在减少令牌开销的同时有效解决了RAG中固定分块导致的边界碎片化问题，实现了跨模型的近似尺度不变的检索性能提升。 |
| [^253] | [GRACE: Step-Level Benchmark for Faithful Reasoning over Context](https://arxiv.org/abs/2606.16151) | 本文提出了GRACE，一个首个基于人工注释的步骤级忠实性基准，通过数据驱动的错误分类法精确定位推理链中的失败步骤及其类型，解决了现有方法仅在响应级别检测幻觉的局限。 |
| [^254] | [Last But Not Least: Boundary Attention CalibratiON for Multimodal KV Cache Compression](https://arxiv.org/abs/2606.14782) | BACON通过结合最后查询注意力与观察窗口注意力，并抑制噪声，显著提升了多模态KV缓存压缩的准确性，尤其在激进压缩场景下平均提升7.5%。 |
| [^255] | [One Polluted Page Is Enough: Evaluating Web Content Pollution in LLM Recommenders](https://arxiv.org/abs/2606.13610) | 本研究首次系统评估了LLM推荐系统在生成引擎优化污染下的脆弱性，发现仅一个被污染的网页即可导致高达27%的虚假产品推荐率，而三个页面污染则使受骗率飙升至73.8%。 |
| [^256] | [TRACE: A Unified Rollout Budget Allocation Framework for Efficient Agentic Reinforcement Learning](https://arxiv.org/abs/2606.11119) | 本文提出TRACE框架，通过将回滚预算分配从提示级别扩展到轮次前缀级别，利用多轮强化学习中的细粒度信息差异，从而提升智能体训练效率。 |
| [^257] | [SocraticPO: Policy Optimization via Interactive Guidance](https://arxiv.org/abs/2606.09887) | SocraticPO通过引入教师引导和奖励衰减机制，使强化学习中的语言模型在错误推理时获得可解释的修正指导，避免捷径学习，从而提升策略的鲁棒性。 |
| [^258] | [Clinically Grounded Privacy Evaluation of Medical LMs](https://arxiv.org/abs/2606.09590) | 该论文提出一个临床接地气的隐私评估框架，揭示医学语言模型在常规就诊元数据下能高比例逐字记忆患者信息并恢复敏感诊断，同时指出精确匹配记忆可能高估泄露风险。 |
| [^259] | [VCIFBench: Evaluating Complex Instruction Following for Video Understanding](https://arxiv.org/abs/2606.04588) | 该论文提出了VCIFBench，一个用于评估视频理解中复杂指令遵循能力的新基准，包含约束丰富的指令和混合验证流程，并发现联合约束满足对现有模型仍具挑战性。 |
| [^260] | [Do Value Vectors in Deep Layers Need Context from the Residual Stream?](https://arxiv.org/abs/2606.02780) | 本文发现深层网络中的无上下文值向量能显著提升模型性能，并可稀疏存储以提高效率。 |
| [^261] | [Same Payload, Different Channel: Measuring Trust Asymmetry in Tool-Using Language Models](https://arxiv.org/abs/2606.00566) | 本文提出安全不对称分数（SAS），发现通用型语言模型对工具元数据中的恶意指令敏感度远低于用户消息，而代理原生型模型差异较小，揭示了信任渠道的不对称性。 |
| [^262] | [RASET: Router-Agnostic Safety-Critical Expert Tuning Exposes Localized Safety Enforcement Failures in Mixture-of-Experts LLMs](https://arxiv.org/abs/2605.29708) | 本文提出RASET框架，通过对比路由敏感性识别并微调安全关键专家，揭示了MoE大语言模型中安全执行可在不改变路由路径的情况下被局部化地破坏。 |
| [^263] | [GIM: Evaluating models via tasks that integrate multiple cognitive domains](https://arxiv.org/abs/2605.18663) | 本文提出GIM基准，通过要求模型在广泛知识基础上整合多种认知操作来评估能力，避免知识记忆与抽象推理的偏差，强调实际任务中的推理接地性。 |
| [^264] | [SafeLens: Deliberate and Efficient Video Guardrails with Fast-and-Slow Screening](https://arxiv.org/abs/2605.17610) | SafeLens通过快速与慢速推理架构，动态分配计算资源，并利用影响引导过滤构建高质量数据集，实现了高效且准确的视频内容审核。 |
| [^265] | [CogniFold: Always-On Proactive Memory via Cognitive Folding](https://arxiv.org/abs/2605.13438) | 本文提出CogniFold，一种受大脑启发的“始终在线”代理记忆，通过扩展CLS理论至三层并利用图拓扑自组织，实现从事件流中自主构建和更新持久认知结构，从而推动主动式智能代理的发展。 |
| [^266] | [Checkup2Action: A Multimodal Clinical Check-up Report Dataset for Patient-Oriented Action Card Generation](https://arxiv.org/abs/2605.11533) | 本文提出了C2A数据集和Checkup2Action工作流程，用于从多模态临床体检报告自动生成结构化、面向患者的行动卡，填补了报告到行动能力缺乏基准测试的空白。 |
| [^267] | [DeepRefine: Agentic Knowledge Refinement via Reinforcement Learning](https://arxiv.org/abs/2605.10488) | DeepRefine通过强化学习框架，利用多轮交互和溯因诊断，自动精炼知识库质量，以提升下游任务性能。 |
| [^268] | [Psychologically Potent, Computationally Invisible: LLMs Generate Social-Comparison-Eliciting Posts They Fail to Detect](https://arxiv.org/abs/2605.01017) | 本研究构建了小红书社会比较基准，发现LLM能生成引发社会比较的帖子，但基于提示的检测器难以稳定识别该信号，揭示生成与检测能力的不对称性。 |
| [^269] | [Revisiting the Effectiveness of LLM Pruning for Test-Time Scaling](https://arxiv.org/abs/2604.25098) | 本研究发现非结构化剪枝不同于结构化剪枝，能保持推理LLMs的测试时扩展性能，挑战了现有假设。 |
| [^270] | [DialToM: A Theory of Mind Benchmark for Forecasting State-Driven Dialogue Trajectories](https://arxiv.org/abs/2604.20443) | 本文提出了DialToM基准，通过状态驱动诊断探针揭示了大语言模型在心理状态推断（字面ToM）与社交预测（功能ToM）之间的系统性不对称，并展示了人类专家100%准确率下的人机能力差距。 |
| [^271] | [Detecting and Suppressing Reward Hacking with Gradient Fingerprints](https://arxiv.org/abs/2604.16242) | 本文提出GRIFT方法，通过压缩模型内部梯度为指纹表示，有效检测并抑制强化学习中的隐性奖励黑客行为。 |
| [^272] | [Unleashing Implicit Rewards: Prefix-Value Learning for Distribution-Level Optimization](https://arxiv.org/abs/2604.13197) | 提出IPVRM模型，通过直接学习前缀的最终正确性概率并使用时间差分差异获得步骤信号，解决了隐式过程奖励模型训练与推理不匹配的问题，实现分布级优化。 |
| [^273] | [Alignment midtraining for animals](https://arxiv.org/abs/2604.13076) | 本文发现通过中期训练结合合成文档能有效提升动物同情心价值对齐，但后续无关指令调优会削弱其效果，提示需要显式保留策略来维持价值干预的持久性。 |
| [^274] | [CONSCIENTIA: Can LLM Agents Learn to Strategize? Emergent Deception and Trust in a Multi-Agent NYC Simulation](https://arxiv.org/abs/2604.09746) | 本研究通过纽约市多智能体模拟，实证观察了LLM智能体在对抗激励下涌现出的策略性欺骗与信任行为，并利用迭代优化流程探索其策略学习机制。 |
| [^275] | [Large Language Models Generate Harmful Responses Using a Distinct Mechanism, Shared Across Harm Types](https://arxiv.org/abs/2604.09544) | 本文通过参数剪枝发现，大型语言模型的有害响应生成依赖于一组稀疏且跨危害类型共享的关键参数，且这种可分离性主要存在于对齐模型中，揭示了有害能力的独特机制。 |
| [^276] | [PRAGMA: Revolut Foundation Model](https://arxiv.org/abs/2604.08649) | PRAGMA是一种针对银行事件序列的金融基础模型，通过自监督掩码建模预训练，能从原始事件数据中直接提取通用嵌入，在信用评分、欺诈检测等任务中表现优异。 |
| [^277] | [What Models Know, How Well They Know It: Knowledge-Weighted Fine-Tuning for Learning When to Say "I Don't Know"](https://arxiv.org/abs/2604.05779) | 本文提出了一种知识加权微调方法，通过估计实例级知识分数来调整学习信号，使模型能明确表达“我不知道”，同时保持对已知问题的准确性，并引入新的不确定性评估指标。 |
| [^278] | [Verbalizing LLMs' assumptions to explain and control sycophancy](https://arxiv.org/abs/2604.03058) | 本文提出“言语化假设”框架，通过引出并利用LLM对用户的假设来理解和控制其社交谄媚行为，并发现“寻求验证”是主要假设，且可通过线性探针进行因果干预。 |
| [^279] | [SAFE: An LLM-as-Verifier Framework for Evidence-Grounded Multi-Hop Reasoning](https://arxiv.org/abs/2604.01993) | SAFE通过将推理分解为知识图谱三元组并在生成过程中用外部验证器实时检查中间步骤，有效防止了多跳问答中的虚假正确性，显著提升了准确率。 |
| [^280] | [A gentle tutorial on Bock's algorithm for minimum directed spanning trees with a structured reformulation](https://arxiv.org/abs/2603.27530) | 本教程详细解读Bock算法，并通过结构化重构保留其核心选择与结果，使原始复杂逻辑更清晰易懂。 |
| [^281] | [UT-ACA: Uncertainty-Triggered Adaptive Context Allocation for Long-Context Inference](https://arxiv.org/abs/2603.18446) | 本文提出UT-ACA，一种通过令牌级不确定性动态调整上下文窗口的推理时框架，能显著减少长上下文推理中的平均上下文使用量。 |
| [^282] | [DynHD: Hallucination Detection for Diffusion Large Language Models via Denoising Dynamics Deviation Learning](https://arxiv.org/abs/2603.16459) | 本文提出DynHD方法，通过同时建模标记空间的不确定性差异和扩散过程的时间动态偏差，有效提升了扩散大语言模型幻觉检测的准确性。 |
| [^283] | [SkillNet: Create, Evaluate, and Connect AI Skills](https://arxiv.org/abs/2603.04448) | 本文提出了SkillNet，一个开放基础设施，通过统一本体论、大规模技能仓库和多功能工具包，系统性地创建、评估和连接AI技能，解决了代理缺乏技能积累和迁移的问题。 |
| [^284] | [Safety Training May Persist Through Helpfulness Optimization in LLM Agents](https://arxiv.org/abs/2603.02229) | 该论文发现，在LLM代理中，安全训练在有用性优化后仍能持续，但安全性与有用性存在强烈的负相关，且同时优化无法突破这一趋势。 |
| [^285] | [Semantic Substrate Dynamics Theory: An Operator-Theoretic Framework for Geometric Semantic Drift](https://arxiv.org/abs/2602.18699) | 本文提出语义基底动力学理论，通过将嵌入几何与扩散核耦合，用粗里奇曲率区分不同语义漂移机制，为异质漂移信号提供统一算子理论框架。 |
| [^286] | [Vibe Coding on Trial: Operating Characteristics of Unanimous LLM Juries](https://arxiv.org/abs/2602.18492) | 本研究提出并评估了一种基于大语言模型陪审团的一致通过机制，用于自动审查文本到SQL任务中的候选代码，在安全优先场景下有效平衡了接受准确性与人工干预成本。 |
| [^287] | [Dialects of Translationese Shape Language Model Learning](https://arxiv.org/abs/2602.16469) | 本文发现，机器翻译数据的源语言差异通过翻译腔显著影响语言模型的学习行为，其中词汇多样性是驱动总体困惑度的关键因素。 |
| [^288] | [Adaptive Test-Time Compute Allocation for Block Diffusion Language Models in Complex Reasoning](https://arxiv.org/abs/2602.09555) | 本文提出一个统一的测试时计算分配框架，通过BACD解码策略和TCCF块级生成范式，在块扩散语言模型中实现自适应推理，兼顾速度与推理精度。 |
| [^289] | [LakeHopper: Knowledge-Aware Adaptation of Column Type Annotators across Data Lakes](https://arxiv.org/abs/2602.08793) | LakeHopper通过将跨数据湖的列类型标注适配分解为知识丢弃、重对齐和获取，并利用通用大语言模型与目标监督相结合，实现了高效的知识感知适配。 |
| [^290] | [MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research](https://arxiv.org/abs/2602.03318) | MIRROR通过无需微调的多智能体框架，结合执行驱动的迭代修正和分层检索，实现了自然语言优化问题到数学模型及求解器代码的直接自动转化。 |
| [^291] | [CALIBURN: Self-Calibrated LLM Unlearning Alignment](https://arxiv.org/abs/2602.02824) | 我们提出了一种自校准遗忘方法，通过量化模型对不良知识的置信度来精确调整梯度更新，在实现细粒度遗忘的同时减少对保留数据的依赖，从而提升模型效用。 |
| [^292] | [Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under Context-Free Grammars](https://arxiv.org/abs/2602.00612) | 提出LAVE方法，通过前瞻-验证机制解决扩散大语言模型在上下文无关文法下的约束解码可靠性问题。 |
| [^293] | [FourierSampler: Unlocking Non-Autoregressive Potential in Diffusion Language Models via Frequency-Guided Generation](https://arxiv.org/abs/2601.23182) | 通过首次频域分析揭示扩散语言模型隐藏状态中低频编码全局结构、高频编码局部细节的规律，并提出FourierSampler利用频域滑动窗口机制实现“结构到细节”的生成，显著提升了非自回归解码性能。 |
| [^294] | [What Language Models Know But Don't Say: Non-Generative Prior Extraction for Generalization](https://arxiv.org/abs/2601.17609) | 该论文提出LoID方法，通过直接探测语言模型的词元级预测置信度，而非生成文本，提取用于贝叶斯逻辑回归的先验分布，从而在小型数据集上提升模型对现实世界的泛化能力。 |
| [^295] | [LLM-Based Adversarial Persuasion Attacks on Fact-Checking Systems](https://arxiv.org/abs/2601.16890) | 本文首次提出利用大语言模型结合说服技巧对自动事实核查系统进行对抗性攻击，显著降低其验证和证据检索性能。 |
| [^296] | [You Need Better Attention Priors](https://arxiv.org/abs/2601.15380) | 该论文通过熵最优传输统一了注意力机制，提出GOAT，用可学习先验替代均匀先验，兼容FlashAttention，解决注意力汇问题，并实现长度泛化。 |
| [^297] | [Opportunities and Challenges of Natural Language Processing for Low-Resource Senegalese Languages in Social Science Research](https://arxiv.org/abs/2601.09716) | 本文首次系统梳理了塞内加尔六种官方语言的NLP资源与挑战，并创建了集中式资源库以推动低资源语言在社会科学研究中的应用。 |
| [^298] | [An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift](https://arxiv.org/abs/2601.05882) | 本研究系统比较了五种偏好调优目标及多种适应策略在领域偏移下的泛化性，发现伪标签适应策略能有效缓解性能下降并保持多样性。 |
| [^299] | [Training Proactive and Personalized LLM Agents](https://arxiv.org/abs/2511.02208) | 该论文提出了一种训练LLM代理的新范式，通过多目标强化学习优化生产力、主动性和个性化三个维度，使代理能更有效地与人类协作并适应个性化需求。 |
| [^300] | [LuxIT: A Luxembourgish Instruction Tuning Dataset from Monolingual Seed Data](https://arxiv.org/abs/2510.24434) | 该论文介绍了LuxIT，一个针对卢森堡语的高质量单语指令微调数据集，通过LLM-as-a-judge质量筛选和微调14个模型，平均提升语言考试准确率5.37个百分点，展示了其在低资源语言NLP中的有效性。 |
| [^301] | [Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning](https://arxiv.org/abs/2510.17021) | 本文首次探索了大语言模型遗忘过程中的后门攻击，发现注意力汇聚点现象与后门有效性密切相关，使得模型在触发条件下能恢复已遗忘的知识。 |
| [^302] | [LLM-Specific Utility for Retrieval-Augmented Generation](https://arxiv.org/abs/2510.11358) | 本文首次形式化并实证了检索增强生成中证据的LLM特定效用，证明其具有模型依赖性和不可转移性，为优化RAG系统提供了新视角。 |
| [^303] | [GraphMed-LT: Patient-Specific Graph Memory with Latent Clinical Thought Refinement for Multi-Turn Medical Conversations](https://arxiv.org/abs/2510.03536) | GraphMed-LT通过构建患者特定的图记忆并利用潜在临床思维细化，解决了多轮医疗对话中临床证据碎片化的问题，从而提升了诊断连贯性。 |
| [^304] | [Syntax-Guided Diffusion Language Models with User-Integrated Personalization](https://arxiv.org/abs/2510.01028) | 本文提出了一种语法引导的扩散语言模型，通过级联与非级联架构集成结构监督和个性化条件，显著提升了文本生成的质量、多样性和可控性。 |
| [^305] | [ConvergeWriter: Data-Driven Bottom-Up Article Construction](https://arxiv.org/abs/2509.12811) | 本文提出了一种自下而上的数据驱动框架，通过先检索知识再聚类构建结构，解决了长文档生成中计划与知识脱节的问题，提升了事实准确性和内容连贯性。 |
| [^306] | [Beyond Benchmarks: LLM Evaluation with an Anthropomorphic and Lifecycle-oriented Roadmap](https://arxiv.org/abs/2508.18646) | 本文提出了一种拟人化的四维评估框架（IQ、PQ、EQ、VQ），将大语言模型评估从静态基准排名转变为基于训练流程因果映射的诊断工具，以弥合基准分数与现实实用性之间的鸿沟。 |
| [^307] | [Models in the Same Family are NOT Trust-Equivalent](https://arxiv.org/abs/2508.13533) | 本文提出一个框架评估同一模型家族中大小模型间的信任等效性，发现仅性能相似并不保证归因对齐和校准相似性，因此小模型不能视为大模型的信任等效替代品。 |
| [^308] | [Omni-SafetyBench: A Benchmark for Safety Evaluation of Audio-Visual Large Language Models](https://arxiv.org/abs/2508.07173) | 本文提出了Omni-SafetyBench，这是首个包含23,328个测试实例、覆盖24种模态变体的平行基准，用于全面评估全模态大语言模型在音视频联合输入下的安全风险，并引入基于Condi的Safety-score指标来应对跨模态一致性挑战。 |
| [^309] | [From Isolation to Alignment: Unified LoRA for Efficient Multi-Task Learning](https://arxiv.org/abs/2508.05078) | 本文通过揭示复杂多组件LoRA变体的冗余性，提出统一的单适配器Align-LoRA框架，以对齐代替隔离，实现高效且性能优越的多任务微调。 |
| [^310] | [Model Directions, Not Words: Mechanistic Topic Models Using Sparse Autoencoders](https://arxiv.org/abs/2507.23220) | 本文提出机制主题模型（MTMs），利用稀疏自编码器的可解释特征定义主题，从而超越词袋限制，实现更深层主题发现和可控文本生成，并引入基于LLM的评估框架。 |
| [^311] | [TELEVAL: A Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios](https://arxiv.org/abs/2507.18061) | TELEVAL是一个针对中文无指令音频交互场景的大规模基准，首次同时评估SLM的语义准确性和基于声学线索的交互适当性，并发现模型在声学变异下性能显著退化。 |
| [^312] | [Text-ADBench: Text Anomaly Detection Benchmark Based on LLM Embeddings](https://arxiv.org/abs/2507.12295) | 本文提出了一个基于多种大语言模型嵌入的文本异常检测基准，系统评估了不同嵌入方法在广泛文本数据集上的性能，填补了该领域缺乏标准化基准的空白。 |
| [^313] | [MixLoRA-DSI: Dynamically Expandable Mixture-of-LoRA Experts for Rehearsal-Free Generative Retrieval over Dynamic Corpora](https://arxiv.org/abs/2507.09924) | 该论文提出MixLoRA-DSI框架，通过OOD驱动的动态扩展策略和混合LoRA专家，实现参数次线性增长的持续生成式检索，显著降低训练成本并超越全模型更新基线。 |
| [^314] | [A Modular Multitask Reasoning Framework Integrating Spatio-temporal Models and LLMs](https://arxiv.org/abs/2506.20073) | 本文提出STReason框架，通过上下文学习将复杂自然语言查询分解为模块化程序，结合LLM推理与时空模型分析能力，实现无需微调的多任务推理和可解释输出。 |
| [^315] | [From Recognition to Reasoning: Advancing Multimodal Harmful Meme Detection via Chain-of-Thought Alignment](https://arxiv.org/abs/2506.18919) | 本文构建了MemeMind大规模有害模因数据集，结合严格分类体系和思维链注释，以提升对隐含风险与细粒度语义的识别与推理能力。 |
| [^316] | [Safety-Aligned Weights Are Not Enough: Refusal-Teacher-Guided Finetuning Enhances Safety and Downstream Performance under Harmful Finetuning Attacks](https://arxiv.org/abs/2506.07356) | 本文提出了一种拒绝教师引导的微调框架，通过直接微调基础模型而非安全对齐权重，在有害微调攻击下同时提升安全性和下游任务性能。 |
| [^317] | [Scaling Electronic Health Record Foundation Models for Population Health Management](https://arxiv.org/abs/2506.00209) | 本文提出了一种跨机构、可扩展的电子健康记录基础模型，通过统一代码对齐和计算最优训练，在超过500万患者数据上实现了大规模慢性病预测，显著提升了人群健康管理的效率。 |
| [^318] | [Effects of Theory of Mind and Prosocial Beliefs on Steering Human-Aligned Behaviors of LLMs in Ultimatum Games](https://arxiv.org/abs/2505.24255) | 本论文通过2700次模拟实验证明，在最后通牒博弈中，心智理论推理能显著增强大语言模型行为与人类规范的对齐、决策一致性和谈判结果，且优于单纯推理模型的表现。 |
| [^319] | [Reasoning Meets Personalization: Unleashing the Potential of Large Reasoning Model for Personalized Generation](https://arxiv.org/abs/2505.17571) | 本文首次系统评估了大型推理模型在个性化任务中的表现，发现其并不总是优于通用LLM，并针对发散思维、格式错配和检索利用不足等问题，提出了强化推理框架（\model）来提升个性化生成效果。 |
| [^320] | [ClinicalGPT-R1: Pushing reasoning capability of generalist disease diagnosis with large language model](https://arxiv.org/abs/2504.09421) | ClinicalGPT-R1通过大规模真实临床数据训练和多种推理增强策略，在中文诊断任务中超越GPT-4o，并在英文任务中与GPT-4持平，为通科疾病诊断提供了高性能的推理增强大语言模型。 |
| [^321] | [AI University: An LLM-Powered Learning Assistant for Engineering---A Finite Element Method Case Study](https://arxiv.org/abs/2504.08846) | 本文提出了一种名为AI-U的框架，通过微调大语言模型并结合检索增强生成，实现了与课程风格一致的学习助手，并以有限元方法课程验证了其有效性。 |
| [^322] | [Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling](https://arxiv.org/abs/2504.05216) | 本文提出LLM-QL模型，通过辅助的查询似然最大化任务增强大语言模型的稠密检索能力，利用生成优势改进对比学习。 |
| [^323] | [Benchmarking and Boosting Multilingual Capabilities of LVLMs via OCR-Centric Reinforcement Learning](https://arxiv.org/abs/2503.18484) | 本文提出了首个基于严格平行语料库的多语言多模态基准PM4Bench，并发现OCR是跨语言性能差异的关键因素，进而设计了以OCR为中心的强化学习训练策略来提升LVLMs的多语言能力。 |
| [^324] | [Deep Contrastive Unlearning for Language Models](https://arxiv.org/abs/2503.14900) | 本文提出了一种深度对比遗忘方法，通过显式考虑模型输出空间的几何结构，在移除特定训练样本信息的同时保持模型性能，以应对语言模型黑箱性带来的遗忘挑战。 |
| [^325] | [Length-Controlled Margin-Based Preference Optimization without Reference Model](https://arxiv.org/abs/2502.14643) | 提出了一种无需参考模型、基于长度控制边际的偏好优化方法（LMPO），通过均匀参考模型和平均对数概率策略，有效解决了DPO的长度偏差、内存低效和概率退化问题。 |
| [^326] | [Towards Safer Social Media Platforms: Scalable and Performant Few-Shot Harmful Content Moderation Using Large Language Models](https://arxiv.org/abs/2501.13976) | 本文提出利用大型语言模型结合上下文学习的小样本方法，在有害内容审核中超越现有基线，并通过多模态技术进一步提升性能。 |
| [^327] | [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) | 本文提出Coconut范式，通过直接使用LLM的连续潜在状态作为推理输入，突破了传统语言空间推理的局限，使模型能更高效地编码和探索多种推理路径。 |
| [^328] | [NeST: Neighborhood-aware semantic alignment and temporal modulation for LLM based time series forecasting](https://arxiv.org/abs/2412.04806) | 本论文提出NEST框架，通过邻域感知语义对齐和时间调制，有效整合文本与时间序列信息，改进了LLM在时间序列预测中的性能。 |
| [^329] | [Bridging Linguistic Structure and Mechanistic Interpretability for Conceptual Interpretation in Language Models](https://arxiv.org/abs/2408.11827) | 本文提出DSRA方法，通过将定义性语义角色融入因果追踪，首次系统性地将语言结构桥接到机制可解释性，用于解释语言模型中的概念组合映射。 |
| [^330] | [A Survey on Human-AI Teaming with Large Pre-Trained Models](https://arxiv.org/abs/2403.04931) | 本文调查了大型预训练模型与人工智能合作的重要性，强调了这些模型如何超越传统方法增强协作智能，并探讨了其在增强人类能力、改善AI模型、有效团队合作、道德考虑以及在各个领域广泛应用方面的潜在作用。 |
| [^331] | [Driving Generative Agents With Their Personality](https://arxiv.org/abs/2402.14879) | 大型语言模型（LLMs）利用心理测量值，在视频游戏角色开发中代表给定的人格特征，增强游戏角色的类人特性。 |
| [^332] | [Towards a resource for multilingual lexicons: an MT assisted and human-in-the-loop multilingual parallel corpus with multi-word expression annotation](https://arxiv.org/abs/2011.03783) | 该工作构建了一个结合机器翻译与人工参与的AlphaMWE多语言平行语料库，标注并手动对齐了多种语言的动词性多词表达，并引入了严格的质量控制流程。 |

# 详细

[^1]: 如何稳定且高效地训练评论家模型

    How to Train a Critic Stably and Efficiently

    [https://arxiv.org/abs/2608.23566](https://arxiv.org/abs/2608.23566)

    BPCO通过结合DPPO、价值预测约束、蒙特卡洛目标、非归一化优势和长度自适应GAE，实现了稳定高效的评论家训练，并利用隐藏于策略的奖励信息提升数学推理性能。

    

    基于群体的强化学习方法（如用于大型语言模型的GRPO）通过为每个提示采样多个响应来避免训练评论家模型。然而，一个可靠的评论家模型本可以从单个响应中估计令牌级别的优势，但标准的基于评论家的训练方案往往不稳定。我们研究了这种不稳定性，并开发了**最佳实践评论家优化（BPCO）**，这是一种结合了DPPO、将价值预测限制在奖励范围内、蒙特卡洛价值目标、非归一化策略优势以及长度自适应广义优势估计的方案。由于评论家仅在训练期间使用，BPCO还可以将其条件化于奖励定义信息（如参考答案或评分标准），而这些信息对策略模型是隐藏的。对照实验隔离了每个设计选择的效果。在从1.5B参数到30B-A3B混合专家模型的数学推理任务中，BPCO改进了性能。

    arXiv:2608.23566v1 Announce Type: cross  Abstract: Group-based reinforcement learning methods such as GRPO for large language models avoid training a critic by sampling multiple responses for each prompt. A reliable critic could instead estimate token-level advantages from one response, but standard critic-based training recipes are often unstable. We study this instability and develop \textbf{Best-Practice Critic Optimization (BPCO)}, a recipe that combines DPPO, value predictions bounded to the reward range, Monte Carlo value targets, unnormalized policy advantages, and length-adaptive generalized advantage estimation. Because the critic is used only during training, BPCO can also condition it on reward-defining information, such as a reference answer or grading rubric, that is hidden from the policy. Controlled experiments isolate the effect of each design choice. Across mathematical reasoning tasks with models ranging from 1.5B parameters to 30B-A3B mixtures of experts, BPCO improv
    
[^2]: SWE重构基准：编码代理能否完成长时程、全仓库的栈迁移？

    SWE Refactor Bench: Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration?

    [https://arxiv.org/abs/2608.23564](https://arxiv.org/abs/2608.23564)

    本文提出了SWE重构基准，通过三阶段评估协议解决现有基准无法检测迁移是否真正发生的问题，从而衡量编码代理在长时程全仓库栈迁移中的能力。

    

    现代软件系统在数十年的开发过程中积累了技术债务，这使得迁移成本高昂且大部分依赖人工。随着编码代理在修复缺陷方面能力日益增强，它们能否自主执行此类迁移？现有基准无法回答这个问题，因为它们仅评估行为正确性，而不评估迁移是否真正发生。这导致了一种简单的作弊手段：代理复制原始实现以使测试通过。我们称之为“盲区”。为解决这一问题，我们引入了SWE重构基准，该基准包含20个全仓库迁移任务，涵盖4种类型的技术债务。一个三阶段评估协议同时衡量迁移完整性和行为正确性：（1）迁移审计验证迁移是否发生。（2）行为测试使用固定测试套件衡量正确性。（3）代理验证使用6个独立的编码代理生成针对性测试。

    arXiv:2608.23564v1 Announce Type: cross  Abstract: Modern software systems accumulate technical debt over decades of development, which makes migration expensive and largely manual. As coding agents become increasingly capable at bug fixing, can they autonomously perform such migrations? Existing benchmarks cannot answer this question because they evaluate only behavioural correctness, not whether the migration actually occurred. This leads an easy hack: agents copy the original implementation to make tests pass. We call this Blindness. To address this problem, we introduce SWE Refactor Bench, a benchmark comprising 20 whole-repository migrations, covering 4 kinds of technical debt. A three-stage evaluation protocol measures both migration completeness and behavioural correctness. (1) Migration Audit verifies that the migration occurred. (2) Behavioural Tests measure correctness with a fixed test suite. (3) Agentic Verification uses 6 independent coding agents to generate targeted test
    
[^3]: Prime Agent：一个自我改进的递归语言模型工具框架

    Prime Agent: A Self-Improving RLM Harness

    [https://arxiv.org/abs/2608.23552](https://arxiv.org/abs/2608.23552)

    Prime Agent是一个开源工具框架，通过持久化REPL和递归子代理机制，将长期评估和编码代理工作流标准化，从而防止工具故障干扰模型，最大化模型潜力。

    

    arXiv:2608.23552v1 公告类型：新 摘要：语言模型是顺序处理器，但长期代理任务需要超越模型权重和活动上下文的外部信息与计算。Prime Agent是一个用于长期评估和编码代理工作流的开源工具框架。一个持久化的IPython REPL遵循递归语言模型抽象，实现程序化上下文处理和测试时计算，而持续工具框架跨轨迹保留历史、记忆、技能、提示和子代理规范。递归子代理通过直接的代理间通信进行协调，代理视图允许人类检查和管控守护进程支持的会话。Prime Agent标准化了执行、恢复、验证和资源核算，同时将策略构建留给模型。这种低摩擦、表达力强的膜防止工具框架故障变成模型故障，并将测量推向模型真正的最大潜在上限。

    arXiv:2608.23552v1 Announce Type: new  Abstract: Language models are sequential processors, but long-horizon agency requires external information and computation beyond model weights and active context. Prime Agent is an open-source harness for long-horizon evaluation and coding-agent workflows. A persistent IPython REPL follows the Recursive Language Model abstraction for programmatic context processing and test-time compute, while Continual Harness preserves histories, memories, skills, prompts, and subagent specifications across trajectories. Recursive subagents coordinate through direct agent-to-agent communication, and the Agents View lets humans inspect and manage daemon-backed sessions. Prime Agent standardizes execution, recovery, verification, and resource accounting while leaving strategy construction to the model. This low-friction, expressive membrane prevents harness failures from becoming model failures and pushes measurement toward the model's true maximal underlying cap
    
[^4]: 汇聚流：具有可证明收敛到词元嵌入的语言流

    ConvergeFlow: Language Flow with Provable Convergence to Token Embeddings

    [https://arxiv.org/abs/2608.23551](https://arxiv.org/abs/2608.23551)

    本文提出ConvergeFlow，一种基于嵌入空间的流式语言模型，通过约束数据预测器到词元嵌入凸包并仅用均方误差训练，证明了流可收敛到有效词元嵌入，从而消除了对交叉熵解码器的需求。

    

    arXiv:2608.23551v1 公告类型：交叉 摘要：近期在连续扩散和基于流的语言模型（LMs）方面取得的进展，已达到与离散LMs竞争的性能。然而，现有的连续框架仍依赖于通过交叉熵（CE）监督的解码器，因为流轨迹不保证终止于有效的词元嵌入。受此局限性启发，我们引入了\textbf{ConvergeFlow}，一种基于嵌入空间的流式语言模型，它将数据预测器约束在词元嵌入的凸包内，并仅使用由流匹配引起的均方误差目标进行训练。在适当的正则性条件下，我们证明尽管数据预测器存在误差，所得到的流仍会收敛到有效的词元嵌入，从而无需CE监督的解码器即可实现直接词元预测。我们进一步开发了三种采样机制，用于控制生成困惑度与熵之间的权衡。在OpenWebText上的实验表明...

    arXiv:2608.23551v1 Announce Type: cross  Abstract: Recent advances in continuous diffusion and flow-based language models (LMs) have achieved performance competitive with discrete LMs. However, existing continuous frameworks still rely on decoders supervised with cross entropy (CE) because the flow trajectories are not guaranteed to terminate at valid token embeddings. Motivated by this limitation, we introduce \textbf{ConvergeFlow}, an embedding-space flow-based LM, which constrains the data predictor to the convex hull of token embeddings and trains it solely with the mean squared error objective induced by flow matching. Under suitable regularity conditions, we prove that the resulting flow converges to valid token embeddings despite errors in the data predictor, enabling direct token prediction without a CE-supervised decoder. We further develop three sampling mechanisms for controlling the trade-off between the generative perplexity and entropy. Experiments on OpenWebText demonstr
    
[^5]: 当名字跨越文字：蒙古世界历史实体对账的源基基准

    When Names Cross Scripts: A Source-Grounded Benchmark for Historical Entity Reconciliation in the Mongol World

    [https://arxiv.org/abs/2608.23507](https://arxiv.org/abs/2608.23507)

    该论文提出了MHER基准，证明在蒙古世界历史人物对账中，基于来源的证据能显著提升准确率，而仅依赖姓名在相同名字的不同人物案例中完全失效。

    

    历史人物可能以不同的语言、文字和转写传统出现，而不同个体可能共享高度相似甚至相同的名字。这使得历史身份对账不仅仅是字符串匹配或音译的问题。我们引入了MHER，一个基于来源控制的基准，用于蒙古世界中人物姓名证词的对账比较。MHER包含一个平衡的396对仅姓名核心数据集，涵盖84位主要历史人物，以及一个更严格的160对基于来源的子集，该子集基于逐来源的提及证据构建，并具有实体不相交的开发集和测试集划分。在五种生成系统中，正确基于来源的证据相对于仅姓名输入，将配对测试准确率提高了12.96到94.44个百分点。在五个表面相同但不同人物的案例中，所有模型在仅使用姓名时均失败（0/25个模型项决策），而基于来源的证据则成功解决了这些问题。

    arXiv:2608.23507v1 Announce Type: cross  Abstract: Historical people may appear under different languages, scripts, and transcription traditions, while distinct individuals may share highly similar or even identical names. This makes historical identity reconciliation more than a problem of string matching or transliteration. We introduce MHER, a provenance-controlled benchmark for pairwise reconciliation of person-name attestations from the Mongol world. MHER contains a balanced 396-pair Name-only core over 84 primary historical persons and a stricter 160-pair Source-grounded subset constructed from mention-by-source evidence, with entity-disjoint development and test splits.   Across five generative systems, correctly Source-grounded evidence improves paired TEST accuracy by 12.96 to 94.44 percentage points relative to Name-only input. On five identical-surface different-person cases, all models fail under names alone (0/25 model-item decisions), whereas Source-grounded evidence yiel
    
[^6]: 通过安全方向惩罚缓解推理引发的不对齐

    Mitigating Reasoning-Induced Misalignment via Safety-Direction Penalty

    [https://arxiv.org/abs/2608.23497](https://arxiv.org/abs/2608.23497)

    本文提出安全方向惩罚（SDP）方法，通过分析表示空间中的推理与安全方向耦合机制，在推理微调时惩罚安全方向的移动，以有效缓解推理引发的不对齐问题。

    

    arXiv:2608.23497v1 公告类型：新  摘要：推理引发的不对齐（Reasoning-Induced Misalignment，简称RIM）是指，在包含无有害内容的推理数据（如数学、代码及带思维链的问题解决）上进行微调，可能诱发大型语言模型的有害行为，这对LLM推理的安全性构成了严重挑战。跨架构、跨规模和跨数据集的检查表明，RIM并非总是出现。以往研究将RIM归因于神经元层面的纠缠，但未识别出这种纠缠背后的表示空间几何结构，也未提出训练时的修复方法。我们提供了两者：对RIM的表示空间分析，以及安全方向惩罚（Safety-Direction Penalty，简称SDP），后者在推理微调期间惩罚沿学习到的安全方向的移动。该分析提取了两个激活空间方向，一个编码推理能力，另一个编码安全行为。这些方向是耦合的：改进推理的微调会改变安全表示，从而引发问题。

    arXiv:2608.23497v1 Announce Type: new  Abstract: Reasoning-Induced Misalignment, where fine-tuning on reasoning data containing no harmful content, including mathematics, code, and problem-solving with chain-of-thought traces can induce harmful behaviors of LLM, posing a serious challenge to the safety of LLM reasoning. Cross-architecture, cross-scale, and cross-dataset checks show that RIM does not always emerge. Previous work attributed RIM to neuron-level entanglement, but did not identify the geometry of the representation space underlying this entanglement or propose a training-time fix. We provide both: a representation-space analysis of RIM and the Safety-Direction Penalty (SDP), which penalizes movement along a learned safety direction during reasoning fine-tuning. The analysis extracts two activation-space directions, one encoding reasoning ability and the other safety behavior. These directions are coupled: fine-tuning that improves reasoning shifts safety representations, an
    
[^7]: 关于怪异泛化与突发错位的威胁模型

    On the Threat Model of Weird Generalization and Emergent Misalignment

    [https://arxiv.org/abs/2608.23476](https://arxiv.org/abs/2608.23476)

    本研究揭示了怪异泛化现象主要由微调数据的构成和语言驱动，而非数据集大小，并且其测量结果对评估问题集高度敏感。

    

    arXiv:2608.23476v1 公告类型：新 摘要：在小规模、领域特定的数据集上进行窄范围微调，可以产生广泛且令人惊讶的模型行为变化——这一现象被称为“怪异泛化”（WG）。然而，目前尚不清楚微调数据的哪些特征对于WG的出现是必要的。在此，我们通过研究一系列可能相关的特征（包括数据集大小、构成、语言、呈现风格以及相对于模型参数知识的新颖性）来探讨这一问题。此外，由于WG评估依赖于小规模的问题集来评估泛化程度，我们还分析了这种测量对所用问题集的敏感性。使用三个开放权重模型在四个数据集上的实验表明，WG的程度（1）在很大程度上依赖于数据集的构成和语言（而非大小）；（2）对于预训练中熟悉的数据，其程度大于新颖数据；（3）对所使用的评估问题集敏感。

    arXiv:2608.23476v1 Announce Type: new  Abstract: Narrow fine-tuning on small, domain-specific datasets can produce broad and surprising changes in model behavior-a phenomenon called weird generalization (WG). Yet, it remains unclear what features of the fine-tuning data are necessary for WG to arise. Here, we address this question by investigating a range of plausibly relevant features, including dataset size, composition, language, presentation style, and novelty relative to a model's parametric knowledge. Further, since WG evaluations rely on small question sets that assess the extent of the generalization, we also analyze how sensitive this measurement is to the set of questions used. Experiments with three open-weight models on four datasets show that the degree of WG (1) depends heavily on dataset composition and language (more than on size); (2) is greater for data familiar from pretraining than for novel data; and (3) is sensitive to the set of evaluation questions used. Collect
    
[^8]: 问题在哪里？评估视觉语言模型中的时间一致性

    What's the Catch? Evaluating Temporal Consistency in Vision-Language Models

    [https://arxiv.org/abs/2608.23474](https://arxiv.org/abs/2608.23474)

    本研究通过时间异常检测任务揭示了视觉语言模型在时间一致性理解上的显著不足，与人类表现存在明显差距。

    

    视觉语言模型（VLMs）在视频和图像序列基准测试中表现出色，但它们是否真正捕捉了时间结构仍不清楚。为研究此问题，我们将时间定位表述为异常检测问题，提供了一种简单且受控的评估方法，直接测试对时间一致性的敏感性。我们引入了TimeCatch，其中时间异常通过交换连续帧创建，帧级异常则通过用高斯噪声替换一帧来创建。模型在四个合成和真实世界数据集上进行了异常检测和定位任务的评估，并伴随一项人类研究。我们的评估揭示了帧级和时间异常检测之间的显著差距。尽管VLMs能一致地检测帧级异常并通常能准确定位它们，但在时间异常检测上它们表现接近随机水平，定位能力仅略高于随机。相比之下，人类在这些任务上表现更佳。

    arXiv:2608.23474v1 Announce Type: cross  Abstract: Vision-language models (VLMs) achieve strong performance on video and image-sequence benchmarks, yet it remains unclear whether they capture temporal structure. To study this question, we formulate temporal grounding as an anomaly detection problem, providing a simple and controlled evaluation that directly tests sensitivity to temporal consistency. We introduce TimeCatch, where temporal anomalies are created by swapping consecutive frames and frame-level anomalies by replacing a frame with Gaussian noise. Models are evaluated on anomaly detection and localization tasks across four synthetic and real-world datasets, alongside a human study. Our evaluation reveals a substantial gap between frame-level and temporal anomaly detection. While VLMs consistently detect frame-level anomalies and often localize them accurately, they perform near chance on temporal anomaly detection and only modestly above chance on localization. Humans, in cont
    
[^9]: 大语言模型在语法工程中有多大用处？粤语ParGram资源及与英语基线的受控实验评估

    How Useful are LLMs for Grammar Engineering? Cantonese ParGram Resources and Controlled Experimental Evaluation with English Baselines

    [https://arxiv.org/abs/2608.23448](https://arxiv.org/abs/2608.23448)

    本文通过粤语ParGram资源和英语基线实验，发现GPT-5.4在生成机器可处理语法方面优于gpt-oss-120b，但两者均难以协调复杂形式约束，表明大语言模型在语法工程中适合辅助局部生成而非全局集成。

    

    本文介绍了新的粤语ParGram资源，并在受控实验范式下评估了大语言模型在知识驱动语法工程中的应用。以粤语ParGram资源为金标准，并设置相应的英语基线，我们研究了OpenAI的gpt-oss-120b和GPT-5.4在系统变化的提示条件下，能否从句子和目标形式结构生成机器可处理的语法。结果显示，GPT-5.4表现优于gpt-oss-120b，而从目标形式结构生成的语法通常优于从句子生成的语法。尽管两种模型都能生成局部合理的短语结构规则、词条和模板，但它们常常难以协调交互的形式约束，尤其是在多构式设置中。这些结果刻画了当前大语言模型在潜在集成到AI辅助专家工作流中的能力与局限性：大语言模型可能...

    arXiv:2608.23448v1 Announce Type: new  Abstract: This paper presents new Cantonese ParGram resources and evaluates LLMs for knowledge-driven grammar engineering within a controlled experimental paradigm. Using Cantonese ParGram resources as gold standards, with corresponding English baselines, we investigate whether OpenAI's gpt-oss-120b and GPT-5.4 can generate machine-processable grammars from sentences and target formal structures under systematically varied prompting conditions. GPT-5.4 outperformed gpt-oss-120b, while grammars generated from target formal structures generally outperformed those generated from sentences. Although both models could generate locally plausible phrase-structure rules, lexical entries, and templates, they often struggled to coordinate interacting formal constraints, especially in multi-construction settings. The results characterize both the capabilities and limitations of current LLMs for potential integration into AI-assisted expert workflows: LLMs ma
    
[^10]: 阿拉伯语自然语言处理研究综合分析：趋势、主题演变与研究空白——一项文献计量与主题研究

    A Comprehensive Analysis of Arabic Natural Language Processing Research: Trends, Topic Evolution, and Research Gaps -- A Bibliometric and Topic-Based Study

    [https://arxiv.org/abs/2608.23421](https://arxiv.org/abs/2608.23421)

    本研究首次对7120篇阿拉伯语NLP论文进行大规模文献计量与主题分析，揭示了2020年后由Transformer和LLM驱动的出版激增现象，并识别出19个核心研究主题及引用影响因素。

    

    自然语言处理（NLP）在过去十年中迅速发展，受到阿拉伯世界数字化转型、社交媒体和大语言模型（LLMs）的推动。尽管发展迅速，但该领域仍缺乏全面的定量元分析。本研究对1960年至2026年间发表的7120篇阿拉伯语NLP论文进行了大规模文献计量和主题分析，数据来源于六个文献集。我们采用BERTopic进行主题建模、回归分析以识别引用预测因子、社会网络分析以考察合著结构，以及地理映射。研究发现，2020年后出版物数量显著激增，主要由Transformer模型和LLMs驱动。主题建模识别出19个实质性主题，其中最大的主题集中在文本、语音、翻译和识别方面。引用分析显示论文年龄与引用次数呈正相关（r = 0.245，p < 0.001）；回归分析表明索引对引用有显著影响。

    arXiv:2608.23421v1 Announce Type: new  Abstract: Natural Language Processing (NLP) has grown rapidly over the past decade, driven by digital transformation in the Arab world, social media, and large language models (LLMs). Despite this growth, a comprehensive quantitative meta-analysis of the field remains absent. This study presents a large-scale bibliometric and topic-based analysis of 7,120 Arabic NLP papers published between 1960 and 2026, sourced from six collections. We employ BERTopic for topic modeling, regression analysis to identify citation predictors, social network analysis for co-authorship structures, and geographic mapping. Our findings show a significant publication surge after 2020, driven by transformer models and LLMs. Topic modeling identifies 19 substantive themes, the largest centered on text, speech, translation, and recognition. Citation analysis reveals a positive correlation between paper age and citations (r = 0.245, p < 0.001); regression shows that indexin
    
[^11]: 信息检索模型对集合增长的鲁棒性

    Robustness of IR Models to Collection Growth

    [https://arxiv.org/abs/2608.23419](https://arxiv.org/abs/2608.23419)

    本研究正式化并实证评估了信息检索模型对添加非相关文档的鲁棒性，发现无论模型是否依赖其他文档，均无法完全避免性能下降。

    

    信息检索（IR）系统旨在在集合中识别相关文档。在实际应用中，集合是动态的，文档经常被添加。我们认为，理想情况下，当非相关文档被添加到集合中时，检索器的有效性不应下降。本研究正式化这一概念，并通过合并两个主题重叠可忽略的集合进行实证评估。我们假设，IR模型如何基于集合中的其他文档调节其排名（例如，BM25中的IDF组件或列表式重排序器中的上下文文档）在其对添加非相关文档的鲁棒性中起重要作用。我们将模型大致分类为不依赖其他文档的模型（多文档无关，MDA）和依赖其他文档的模型（多文档相关，MDD）。我们的结果表明，MDD和MDA模型在添加非相关文档时均不完全鲁棒。

    arXiv:2608.23419v1 Announce Type: cross  Abstract: Information Retrieval (IR) systems seek to identify relevant documents within a collection. In practical applications, collections are dynamic, with documents frequently added. We argue that ideally, a retriever's effectiveness should not decrease when non-relevant documents are added to a collection. This study formalises this concept and empirically evaluates it by merging two collections with negligible topic overlap. We hypothesise that the way an IR model conditions its ranking on other documents in a collection (e.g., the IDF component in BM25 or contextual documents in listwise rerankers) plays an important role in its robustness to the addition of non-relevant documents. We broadly classify models as those that do not depend on other documents (Multi-Document-Agnostic, MDA) and those that do (Multi-Document-Dependent, MDD). Our results show that neither MDD nor MDA models are fully robust to the addition of non-relevant documen
    
[^12]: STONIC：一种用于LLM价值剖析的分层测量契约

    STONIC: A Layered Measurement Contract for LLM Value Profiling

    [https://arxiv.org/abs/2608.23411](https://arxiv.org/abs/2608.23411)

    STONIC通过分层测量契约验证了LLM价值剖析中评分、选择和文本推断的一致性假设，发现大多数配置保持认可-选择关系但偏好自身答案，且剖析形状在不同测量层次间转移强度不一。

    

    arXiv:2608.23411v1 公告类型：新 摘要：LLM价值研究常常将问卷评分、成对选择和从生成文本中推断出的价值观合并为一个剖析结果。这种合并假设了三种观测描述的是同一种稳定偏好。STONIC在来自四个银行组的5,144个情境和35种固定模型配置上测试了这一假设。它比较了隔离环境下的评分响应、平衡冲突下的选择、自发回答，以及随后在模型自身答案与作者备选答案之间的选择。17个具有可用行为数据的配置中，有10个在银行组间保持了认可-选择关系。所有17个符合条件的配置都偏好其自身早期答案（中位效应0.790），尽管选项位置在每个符合条件的配置中都改变了选择率。剖析形状从评分到冲突选择的转移最为强烈，而对自发文本则减弱。对200个L3响应的三方标注提供了任务局部一致性。

    arXiv:2608.23411v1 Announce Type: new  Abstract: LLM value studies often merge questionnaire ratings, pairwise choices, and values inferred from generated text into one profile. That merge assumes that the three observations describe the same stable preference. STONIC tests this assumption on 5,144 situations from four banks and 35 fixed model configurations. It compares responses rated in isolation, choices made under counterbalanced conflict, spontaneous answers, and later choices between a model's own answer and authored alternatives. 10 of 17 configurations with usable behavioral data preserve the endorsement-choice relation across banks. Every one of the 17 eligible configurations prefers its own earlier answer (median effect 0.790), although option position changes the choice rate in every eligible configuration. Profile shape transfers most strongly from ratings to conflict choices and weakens for spontaneous text. Three-way annotation of 200 L3 responses provides a task-local c
    
[^13]: 无需领域内训练数据的跨域、多任务数据到文本生成

    Cross-Domain, Multi-Task Data-to-Text Generation without In-Domain Training Data

    [https://arxiv.org/abs/2608.23391](https://arxiv.org/abs/2608.23391)

    本研究提出了一种无需领域内训练数据的跨域、多任务数据到文本生成方法，通过数据驱动的知识蒸馏和结构保持增强，在恒定模型大小下优于微调和零样本推理。

    

    arXiv:2608.23391v1 公告类型：交叉 摘要：结构化数据以多种形式存在（表格、知识图谱、图表和时间序列），将其转换为文本可能涉及不同的生成任务。然而，以往大多数数据到文本（D2T）生成工作都集中在特定任务和数据集上，要么依赖任务特定的训练数据，要么依赖大型语言模型的零样本能力。我们研究了一种跨域D2T生成设置，在这种设置中，既没有领域内的训练文本，也没有测试参考，且领域、生成目标和输入结构差异显著。我们比较了数据驱动的知识蒸馏（DDKD）与零样本推理以及在域外D2T数据上的微调，并通过结构子采样和扰动引入了保持结构的增强方法。在五个基准上的实验表明，在恒定模型大小（1.7B参数）下，DDKD始终优于微调和零样本推理。此外，

    arXiv:2608.23391v1 Announce Type: cross  Abstract: Structured data exists in many forms (tables, knowledge graphs, charts, and time series), and converting it into text may involve different generation tasks. However, most prior work on data-to-text (D2T) generation has focused on specific tasks and datasets, relying either on task-specific training data or on the zero-shot capabilities of large language models. We study cross-domain D2T generation in a setting where neither in-domain training text nor test references are available, and where domains, generation goals, and input structures vary substantially. We compare data-driven knowledge distillation (DDKD) against zero-shot inference and fine-tuning on out-of-domain D2T data, and introduce structure-preserving augmentation via structural subsampling and perturbation. Experiments on five benchmarks show that, at constant model size (1.7B parameters), DDKD consistently outperforms both fine-tuning and zero-shot inference. Moreover, 
    
[^14]: 跨语言传记增强：通过主张提取与对齐实现

    Cross-lingual Biography Enrichment via Claim Extraction and Alignment

    [https://arxiv.org/abs/2608.23390](https://arxiv.org/abs/2608.23390)

    本文提出了一种基于主张提取与对齐的框架，利用非英文维基百科传记中的本地化事实来丰富英文传记，并通过CLAW-4L基准验证了其有效性。

    

    英文维基百科通常被视为默认的百科全书来源，然而非英文维基百科版本可能包含更丰富的、针对长尾人物的本地化信息。我们研究了跨语言传记增强：即利用同一人物在非英文传记中支持的事实来丰富现有的英文传记。聚焦于来自非英语背景的女性，我们引入了\textsc{CLAW-4L}，一个包含300对维基百科传记对的基准数据集，将英文传记与法文、中文或阿塞拜疆文对应传记配对，并附有主张注释和细粒度的主张对关系语料库。我们提出了一种基于主张的增强框架，从两种传记中提取英文主张，对齐这些主张以识别来自非英文传记的增强证据，并使用选定的主张重写英文传记。我们的结果表明，非英文维基百科传记提供了有价值的证据。

    arXiv:2608.23390v1 Announce Type: cross  Abstract: English Wikipedia is often treated as the default encyclopedic source, yet non-English Wikipedia editions can contain richer locally grounded information for long-tail figures. We study cross-lingual biography enrichment: enriching an existing English biography with facts supported by a non-English biography about the same person. Focusing on women from non-English-speaking contexts, we introduce \textsc{CLAW-4L}, a benchmark consisting of 300 Wikipedia biography pairs linking an English biography with its French, Chinese or Azerbaijani counterpart, along with claim annotations and a fine-grained claim-pair relation corpus. We propose a claim-based enrichment framework that extracts English claims from both biographies, aligns them to identify enrichment evidence from the non-English biography, and rewrites the English biography using the selected claims. Our results show that non-English Wikipedia biographies provide valuable evidence
    
[^15]: 低资源语言表征的几何特性

    The Geometry of Low-Resource Language Representations

    [https://arxiv.org/abs/2608.23358](https://arxiv.org/abs/2608.23358)

    本文发现低资源语言在大型语言模型最终层存在表征退化，并提出几何正则化方法在持续预训练中有效缓解该问题，尤其对较大模型性能有轻微提升。

    

    大型语言模型（LLMs）在低资源语言与高资源语言之间的性能差距广为人知，但驱动这些差异的内部模型因素仍不明确。本文通过表征几何的视角来刻画这一差距。比较30种语言隐藏表征的几何属性揭示出，LLM的几何结构与语言数据的可用性存在系统性关联。最一致的影响出现在最终层，其中低资源语言表现出表征退化现象。为应对这一问题，我们研究了在持续预训练（CPT）期间引入正则化项以惩罚退化的有效性。将9个基础LLM单语适配到10种非洲语言的实验表明，几何正则化能成功减少CPT期间的表征退化。对于较大模型，基于余弦相似度的正则化在性能上略优于普通CPT，且效果更佳。

    arXiv:2608.23358v1 Announce Type: new  Abstract: The performance gap between low- and high-resource languages in LLMs is widely known, but it remains unclear which internal model factors drive these disparities. In this paper, we characterise this gap through the lens of representational geometry. Comparing the geometric properties of hidden representations across 30 languages reveals that LLM geometry is systematically related to language data availability. The most consistent effect is in final layers, where low-resource languages exhibit representational degeneration. To counter this, we investigate the effectiveness of regularisation terms to penalise degeneration during continued pretraining (CPT). Experiments monolingually adapting 9 base LLMs to 10 African languages show that geometric regularisation successfully reduces representational degeneration during CPT. For larger models, cosine similarity-based regularisation marginally improves performance over vanilla CPT, with more 
    
[^16]: FormuEvo：基于大语言模型引导的进化方法，用于发现求解器高效的混合整数规划公式

    FormuEvo: LLM-Guided Evolution for Discovering Solver-Efficient Mixed-Integer Programming Formulations

    [https://arxiv.org/abs/2608.23353](https://arxiv.org/abs/2608.23353)

    FormuEvo提出了一种利用大语言模型引导的进化框架，通过求解器感知的诊断机制和符号空间优化，自动发现求解器高效的混合整数规划公式，克服了传统LLM建模忽视公式强度的问题。

    

    混合整数规划（MIP）是运筹学和工业优化的核心。虽然大型语言模型（LLMs）最近在从自然语言自动建模MIP方面显示出潜力，但它们优先考虑语义正确性，而忽视了公式的强度，严重制约了下游求解器的效率。我们提出了FormuEvo，一种由LLM引导的进化框架，用于自动发现求解器高效的MIP公式。FormuEvo将MIP公式设计视为在MIP公式的符号空间（表示为可执行建模程序）上的进化优化，通过迭代生成、评估和选择更强候选，利用LLM驱动的交叉、变异和修复操作。为了超越盲目探索，FormuEvo引入了一种求解器感知的诊断机制，利用细粒度的求解器统计作为语言梯度进行有针对性的改进。此外，

    arXiv:2608.23353v1 Announce Type: new  Abstract: Mixed-integer programming (MIP) lies at the core of operations research and industrial optimization. While large language models (LLMs) have recently shown promise in automated MIP modeling from natural language, they prioritize semantic correctness but overlook formulation strength, severely bottlenecking the efficiency of downstream solvers. We propose FormuEvo, an LLM-guided evolutionary framework for automated discovery of solver-efficient MIP formulations. FormuEvo frames MIP formulation design as evolutionary optimization over the symbolic space of MIP formulations, represented as executable modeling programs, by iteratively generating, evaluating, and selecting stronger candidates via LLM-driven crossover, mutation, and repair operations. To move beyond blind exploration, FormuEvo introduces a solver-informed diagnosis mechanism that exploits fine-grained solver statistics as verbal gradients for targeted refinement. Additionally,
    
[^17]: 在LoRA微调过程中通过公理化注意力模式出现相关性

    The Emergence of Relevance Through Axiomatic Attention Patterns During LoRA Fine-Tuning

    [https://arxiv.org/abs/2608.23338](https://arxiv.org/abs/2608.23338)

    本研究发现，在LoRA微调过程中，仅对网络中紧凑的中部区域进行注意力更新即可恢复大部分性能提升，且该区域与可解释的相关性注意力模式出现区域高度重合。

    

    arXiv:2608.23338v1 公告类型：交叉 摘要：LoRA微调是适应LLM进行重排序的标准方法，但网络中的任务特定相关性行为在哪里学习以及伴随该学习的注意力级变化仍不清楚。通过消融和注意力实验，我们确定了LoRA注意力更新对RankLLaMA在何处提高性能，以及这些收益是否与可解释的相关性导向注意力模式（如词汇匹配、稀有性敏感性和查询-文档交互）一致。我们发现，鉴于整个网络中LoRA微调的MLP，将LoRA注意力更新限制在一个紧凑的中网络区域就足以恢复通过对所有注意力层应用LoRA所获得性能的一半以上，并且在该区域省略注意力微调对性能的损害大于网络中其他区域。此外，我们表明应用LoRA影响性能最大的区域与可解释相关性注意力模式出现的区域重叠。

    arXiv:2608.23338v1 Announce Type: cross  Abstract: LoRA fine-tuning is standard for adapting LLMs to reranking, but it remains unclear where in the network task-specific relevance behavior is learned and what attention-level changes accompany that learning. Through ablation and attention experiments, we identify where LoRA attention updates to RankLLaMA improve performance and whether those gains coincide with interpretable relevance-oriented attention patterns such as lexical matching, rarity sensitivity, and query-document interaction. We find that given LoRA fine-tuned MLPs throughout the network, restricting LoRA attention updates to a compact mid-network region is sufficient for recovering over half of the performance gained by applying LoRA to all attention layers, and that omitting attention fine-tuning in this region hurts performance more than elsewhere in the network. Additionally, we show that regions where applying LoRA affects performance the most overlap with regions wher
    
[^18]: Flesch-Kincaid可读性在主题模型下仅依赖于长文本的主题分布

    Flesch-Kincaid Readability Depends Only on the Topic Distribution in Long Texts under Topic Models

    [https://arxiv.org/abs/2608.23327](https://arxiv.org/abs/2608.23327)

    在主题模型下，长文本的Flesch可读性评分几乎完全由主题分布决定，与词汇组成无关，揭示了可读性评分的本质局限性。

    

    Flesch阅读易读性（FRE）和Flesch-Kincaid年级水平（FKGL）是广泛使用的英语可读性评分，它们基于相同的两个文档统计量计算，但它们在长文档上的稳定性并不一定意味着对词汇构成的不变性。令人惊讶的是，在带有显式句子边界标记的主题模型下，这两个评分几乎必然收敛为文档主题分布的确定性函数，仅通过两个标量速率实现：在长文本极限下，所有评分变异由主题组成介导，而非任何残余的可读性信号。该理论涵盖两个公式，而实验评估FKGL。在固定混合模型中，秩[1, q, s] = 3，内部主题向量的纤维局部为（K-3）维，而正则等分水平集局部为（K-2）维且弯曲。在两个平衡语料库（Brown和书面BNC）的折叠外评估中，从数据推断的主题向量显示出稳定性和预测性。

    arXiv:2608.23327v1 Announce Type: new  Abstract: Flesch Reading Ease (FRE) and the Flesch-Kincaid Grade Level (FKGL) are widely used readability scores for English computed from the same two document statistics, yet their stability on long documents need not imply invariance to lexical composition. Surprisingly, under a topic model with an explicit sentence-boundary token, both scores converge almost surely to deterministic functions of the document topic distribution through just two scalar rates: in the long-text limit, all score variation is mediated by topical composition rather than any residual readability signal. The theory covers both formulae, while the experiments evaluate FKGL. In a fixed admixture with rank[1, q, s] = 3, fibres through interior topic vectors are locally (K-3)-dimensional, whereas regular iso-score level sets are locally (K-2)-dimensional and curved. In out-of-fold evaluation on two balanced corpora, Brown and the written BNC, a topic vector inferred from on
    
[^19]: Agent-G$^2$：用于智能体强化学习的高斯引导

    Agent-G$^2$: Gaussian Guidance for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2608.23318](https://arxiv.org/abs/2608.23318)

    我们提出Agent-G$^2$，一种无需额外探测回合的高斯引导框架，通过在线估计每个任务的最优引导深度分布，有效解决智能体强化学习中的奖励稀疏问题。

    

    arXiv:2608.23318v1 公告类型：新论文  摘要：基于提示的强化学习通过在每个回合前保留专家轨迹的前缀，让策略从更接近成功状态的位置开始探索，从而解决长周期智能体任务中的奖励稀疏问题。其有效性取决于引导深度，即应保留轨迹的多少部分。现有方法将此深度视为确定性标量。调度方法在样本间共享一个值，忽视了任务间的异质性；逐样本探测方法则分别估计深度，但代价是额外的回合。我们发现，有用的引导占据了一个深度区间，其信息量分布近似于围绕区间中心的高斯分布，而非集中于单一最优点上。为此，我们提出Agent-G$^2$，一种高斯引导框架，它从已用于策略优化的回合中在线估计每个任务的高斯分布中心和宽度，从而抽取深度，无需探测回合或学习模型。

    arXiv:2608.23318v1 Announce Type: new  Abstract: Hint-based reinforcement learning addresses reward sparsity in long-horizon agentic tasks by retaining a prefix of an expert trajectory before each rollout, letting the policy explore from a state closer to success. Its effectiveness hinges on the guidance depth: how much of the trajectory to keep. Existing methods treat this depth as a deterministic scalar. Scheduled approaches share one value across samples and ignore per-task heterogeneity; per-sample probing estimates it separately at the cost of extra rollouts. We find that useful guidance occupies a band of depths whose informativeness profile is approximately Gaussian around the band center, rather than concentrating at a single optimal point. We propose Agent-G$^2$, a Gaussian guidance framework that draws the depth per task from a Gaussian whose center and spread are estimated online from rollouts already collected for policy optimization, requiring no probe rollouts or learned 
    
[^20]: 超越稳定性-探索困境：面向大语言模型策略优化的环境正则化

    Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization

    [https://arxiv.org/abs/2608.23311](https://arxiv.org/abs/2608.23311)

    本文提出环境正则化策略优化（ERPO），通过将正则化从动作侧移至输入侧并引入查询KL约束，有效打破了大语言模型策略优化中稳定性与探索之间的两难困境。

    

    大语言模型的策略优化（PO）面临稳定性与探索之间的权衡，目前通过动作侧的策略KL正则化来调节。这使实践者陷入两难境地：保持策略KL会约束响应行为并消耗动作侧的探索预算，而放弃它则使优化缺乏明确的漂移控制。我们提出了一种替代方案，通过将正则化移至输入侧来打破这一困境。随着训练的进行，当前策略诱导的训练查询分布会从其强化学习前的参考分布无控制地漂移。具体而言，环境正则化策略优化（ERPO）引入了一个查询KL（QKL）项来约束这种查询分布偏移，同时结合一个基于数据集静态参考的逐查询权重，将每个查询更新偏向于参考下典型的查询。QKL梯度严格流经查询似然。

    arXiv:2608.23311v1 Announce Type: new  Abstract: Policy optimization (PO) for Large Language Models faces a stability--exploration trade-off, currently mediated by an action-side Policy-KL regularizer. This puts practitioners in a double bind: keeping Policy-KL constrains response behavior and consumes the action-side exploration budget, while dropping it leaves the optimization without an explicit drift control. We argue for an alternative that breaks the dilemma by moving regularization to the input side. As training progresses, the distribution over training queries induced by the current policy drifts unchecked from its pre-RL reference distribution.   Concretely, Environment-Regularized Policy Optimization (ERPO) introduces a Query-KL (QKL) term that bounds this query distribution shift, together with a dataset-static reference-derived per-query weight that biases each per-query update toward queries typical under the reference. The QKL gradient flows strictly through the query li
    
[^21]: 面向跨语料库时间分析的动态主题建模

    Dynamic Topic Modeling for Cross-Corpus Temporal Analysis

    [https://arxiv.org/abs/2608.23284](https://arxiv.org/abs/2608.23284)

    该论文提出了一种基于共享骨干和残差适应的动态主题建模框架，解决了跨语料库时间分析中主题对齐不稳定问题，实现了稳定的跨语料库比较和词汇专门化。

    

    动态嵌入主题模型（D-ETM）为建模时间语义演化提供了一个可解释的框架，但跨语料库比较仍然困难，因为主题通常是独立学习并在训练后才进行对齐的，这一过程无法保证跨语料库和时间上的稳定主题对应关系。为解决这一问题，我们提出了一种D-ETM框架，该框架首先在合并的多语料库集合上学习一个共同的动态主题空间，我们称之为共享骨干，然后在冻结的骨干周围引入特定于语料库的残差适应，而不创建独立的潜在主题空间。这种设计保留了用于跨语料库比较的共享主题索引，同时允许每个语料库在词汇上实现专门化。我们在三个跨越97年的时间结构化语料库上评估了该框架：美国历史英语语料库、哈佛商业评论和国际劳工评论。残差适应...

    arXiv:2608.23284v1 Announce Type: new  Abstract: Dynamic Embedded Topic Models (D-ETM) provide an interpretable framework for modeling temporal semantic evolution, but cross-corpus comparison remains difficult because topics are often learned independently and aligned only after training, a process that does not guarantee stable topic correspondence across corpora and time. To address this problem, we propose a D-ETM framework that first learns a common dynamic topic space over a merged multi-corpus collection, which we call the shared backbone, then introduces corpus-specific residual adaptation around the frozen backbone without creating separate latent topic spaces. This design preserves a shared topic index for cross-corpus comparison while allowing each corpus to specialize lexically. We evaluate the framework on three temporally structured corpora spanning 97 years: the Corpus of Historical American English, Harvard Business Review, and International Labour Review. Residual adapt
    
[^22]: Apodex 1.1：扩展面向复杂工作的代理智能

    Apodex 1.1: Scaling Agentic Intelligence for Complex Work

    [https://arxiv.org/abs/2608.23283](https://arxiv.org/abs/2608.23283)

    Apodex 1.1 通过环境扩展和代理协调扩展，提升了语言模型在复杂工作中持续交互、状态维护和可验证交付的工作能力。

    

    arXiv:2608.23283v1 公告类型：新公告  摘要：通用语言模型能够进行推理和综合知识，但复杂工作还需要与文件、信息源和可执行代码进行持续交互，同时具备状态维护、故障恢复和可验证交付的能力。我们将此称为“工作能力”：朝着现实目标持续且可验证的进展。Apodex 1.1 沿着两个互补维度发展这一能力。“环境扩展”扩大了可执行文件、搜索和代码环境的多样性与可验证性，而“代理协调扩展”则训练代理分解长期任务、委派并行工作、整合异步结果并重新规划。一个共享的执行框架和 AgentOS 维护跨工具和代理的任务状态与来源，训练将环境轨迹和协调痕迹转化为可靠行为。在复杂的专业工作、金融、科学研究等领域中，这一能力得到了体现。

    arXiv:2608.23283v1 Announce Type: new  Abstract: General-purpose language models can reason and synthesize knowledge, but complex work also requires sustained interaction with files, information sources, and executable code, together with state maintenance, failure recovery, and verifiable delivery. We call this \emph{working capability}: sustained, verifiable progress toward a real-world objective. Apodex 1.1 develops this capability along two complementary dimensions. \emph{Environment Scaling} expands the diversity and verifiability of executable file, search, and code environments, while \emph{Agentic Coordination Scaling} trains agents to decompose long-horizon tasks, delegate parallel work, integrate asynchronous results, and replan. A shared execution harness and AgentOS maintain task state and provenance across tools and agents, and training turns environment trajectories and coordination traces into reliable behavior. Across complex professional work, finance, scientific resea
    
[^23]: 计算机科学研究中AI披露的期望与实践

    Expectations and Practices around AI Disclosure in CS Research

    [https://arxiv.org/abs/2608.23271](https://arxiv.org/abs/2608.23271)

    本研究调查了计算机科学场所的AI披露政策，发现其普遍缺乏具体规定，并通过调查和案例分析揭示了研究人员对披露必要性及信息内容的期望。

    

    arXiv:2608.23271v1 公告类型：交叉 摘要：随着生成式AI工具在研究工作流程中的日益普及，关于其影响、适当性和负责任使用的持续争论促使政策制定者在多个出版场所制定政策以披露AI使用情况。然而，当前的AI披露政策和实践是否反映了其目的？在本研究中，我们首先调查了顶级计算机科学场所的披露政策，发现尽管这些政策普遍存在，但它们仍然高度缺乏详细规定。其次，通过对计算机科学研究人员的调查（N=$109$），我们描述了不同研究任务和人类参与水平下披露的必要性。我们了解到，研究人员认为在涉及研究设计的任务以及人类参与度低的任务中，披露最为必要。我们还整理了研究人员对AI披露声明中应传达信息的期望。最后，通过对$13$个案例的分析，我们进一步探讨了这些期望在实际中的体现。

    arXiv:2608.23271v1 Announce Type: cross  Abstract: As generative AI tools find increasing use in research workflows, ongoing debates on their impact, appropriateness and responsible use have led policymakers to enact policies to disclose AI use at multiple publishing venues. However, are current AI disclosure policies and practices reflective of their purpose? In this work, we first investigate disclosure policies of top computer science venues and find that despite their prevalence, they remain highly under-specified. Secondly, through a survey of computer science researchers (N=$109$), we characterize the necessity of disclosures across different research tasks and levels of human involvement. We learn that researchers find disclosures most necessary for tasks involving research design, and for tasks when the human involvement is low. We also compile expectations that researchers have about the information to be conveyed in AI disclosure statements. Lastly, through an analysis of $13
    
[^24]: EvoWiki：面向跨会议知识演进的增量状态覆盖与可追溯问答

    EvoWiki: Incremental State Overwriting and Traceable Question Answering for Cross-Meeting Knowledge Evolution

    [https://arxiv.org/abs/2608.23265](https://arxiv.org/abs/2608.23265)

    EvoWiki通过增量状态覆盖协议和实体版本链，显式建模知识生命周期，解决了跨会议问答中新旧状态冲突和答案不可验证的问题。

    

    arXiv:2608.23265v1 公告类型：新 摘要：在跨越多次会议的长期协作中，决策和风险等事实状态会不断被修订、推翻和替换。现有的长上下文方法通常堆叠整个历史记录，而许多RAG和结构化记忆方法将知识组织为静态或仅追加的事实，并在读取时依赖语义相关性。由于缺乏对知识生命周期的显式建模，这些方法可能同时保留冲突的新旧状态，或丢弃历史记录，导致检索结果过时且答案难以验证。我们提出了EvoWiki，一种用于动态长文本的增量问答架构。EvoWiki将离线增量构建（BUILD）与在线结构化读取（READ）解耦。BUILD捕获从提案到决策的会议内微观演化，并使用实体版本链和细粒度的状态覆盖协议，以显式区分当前有效状态。

    arXiv:2608.23265v1 Announce Type: new  Abstract: In long-term collaboration spanning multiple meetings, factual states such as decisions and risks are continually revised, overturned, and replaced. Existing long-context methods typically stack the entire history, while many RAG and structured-memory methods organize knowledge as static or append-only facts and rely on semantic relevance at read time. Without explicit modeling of knowledge lifecycles, these approaches may retain conflicting old and new states simultaneously or discard history, leading to stale retrieval and answers that are difficult to verify. We present EvoWiki, an incremental question-answering architecture for dynamic long-form text. EvoWiki decouples offline incremental construction (BUILD) from online structured reading (READ). BUILD captures the intra-meeting micro-evolution from proposal to decision and uses entity version chains and a fine-grained State-Overwrite Protocol to explicitly distinguish current valid
    
[^25]: 隐藏在请求中：通过令牌相关性解释不道德的大语言模型顺从行为

    Hidden in the Request: Explaining Unethical LLM Compliance through Token Relevance

    [https://arxiv.org/abs/2608.23264](https://arxiv.org/abs/2608.23264)

    本文通过引入三种模态的探测方法，发现大语言模型在直接请求帮助时更易顺从于不道德行为，并利用层间相关性传播揭示其归因偏差——模型过度关注任务框架令牌而忽视不道德提示令牌，从而解释了对齐失败的机制。

    

    arXiv:2608.23264v1 公告类型：新 摘要：尽管大语言模型（LLMs）被对齐以优化帮助性和无害性，但这双重目标可能发生冲突，不可避免地导致对齐失败。本研究系统性地调查了LLMs未能表现出道德行为的实例。为了理解这些脆弱性的潜在机制，我们引入了一种探测方法，将不道德场景以三种不同的结构模态呈现给LLMs：客观分类任务、主观第一人称陈述和直接请求帮助。我们发现，模型性能在基于请求帮助的形式中会下降。利用层间相关性传播（LRP），我们将这种差异追溯到一种归因偏差：模型更强调良性的任务框架令牌（例如，“你能帮我……”），而不是那些暗示潜在不道德行为的令牌（例如，“不被抓住”），我们将其称为提示令牌。

    arXiv:2608.23264v1 Announce Type: new  Abstract: Although Large Language Models (LLMs) are aligned to optimize for both helpfulness and harmlessness, these dual objectives may conflict, inevitably leading to alignment failures. This work systematically investigates instances where LLMs fail to exhibit ethical behavior. To understand the underlying mechanics of these vulnerabilities, we introduce a probing methodology that presents unethical scenarios to LLMs in three distinct structural modalities: objective classification tasks, subjective first-person statements, and direct requests for assistance. We find that model performance degrades in the request-for-assistance-based form. Using Layer-wise Relevance Propagation (LRP), we trace this discrepancy to an attribution bias: the model places greater emphasis on benign task-framing tokens (e.g., "Can you help me...") than on tokens signaling the underlying unethical behavior (e.g., "without getting caught"), which we term cue-tokens. We
    
[^26]: 从扁平文化遗产记录自动构建FAIR数字对象知识图谱

    Automated Construction of FAIR Digital Object Knowledge Graphs from Flat Cultural Heritage Records

    [https://arxiv.org/abs/2608.23263](https://arxiv.org/abs/2608.23263)

    本文提出了一种自动化流水线，将扁平的Europeana文化遗产记录转换为符合FDO规范的知识图谱，通过自动区分PID引用和字面量值，实现文化遗产数据的完全机器可操作性。

    

    FAIR数字对象（FDO）框架要求元数据属性值尽可能表示为持久标识符（PID），以生成完全机器可操作的图，其中每个引用都可解析。欧洲数字图书馆数据模型（Europeana Data Model）早在FDO规范制定之前就已设计，其大多数元数据值以纯文本形式存储。这足以满足人类浏览需求，但无法为自动化代理提供跨记录或收藏的追踪线索。我们提出了一种流水线，将扁平的Europeana记录转换为符合FDO规范、并以CIDOC-CRM结构化的知识图谱。遵循FDO规范，我们将每个文化遗产实体建模为具有自身PID、类型、配置文件和元数据层的离散FDO。核心技术挑战是自动化执行FDO规定的区分：哪些值必须成为PID引用（可解析实体），哪些值可保持为字面量（终端叶节点，如注释、测量值）。

    arXiv:2608.23263v1 Announce Type: new  Abstract: The FAIR Digital Object (FDO) framework mandates that metadata attribute values be expressed as persistent identifiers (PIDs) wherever possible, to produce a fully machine-actionable graph in which every reference is resolvable. The Europeana Data Model was designed long before the FDO specification, and it stores most metadata values as plain text. This serves human browsing well enough, but gives an automated agent nothing to follow across records or collections. We present a pipeline that transforms flat Europeana records into an FDO-compliant knowledge graph structured with CIDOC-CRM. Following the FDO specification, we model every heritage entity as a discrete FDO with its own PID, type, profile, and metadata layer. The core technical challenge is automating the FDO-prescribed distinction between values that must become PID references (resolvable entities) and those that may remain literals (terminal leaves such as notes, measuremen
    
[^27]: 一种通过统一生成式训练框架实现的可扩展跨领域事件抽取系统

    A Scalable Cross-Domain Event Extraction System via a Unified Generative Training Framework

    [https://arxiv.org/abs/2608.23261](https://arxiv.org/abs/2608.23261)

    本文提出了一种统一的生成式序列到序列事件抽取框架，通过多领域微调实现可扩展性和跨领域泛化，并提供了一个支持文档上传、模式感知抽取和可视化比较的Web应用平台。

    

    事件抽取是信息抽取的基础。以往的方法通常将事件检测和参数抽取分开处理，或依赖于特定数据集的设计，这限制了其可扩展性和跨领域泛化能力。我们提出了一种统一的生成式序列到序列框架，该框架联合执行事件抽取子任务，并支持流水线和端到端两种配置。我们在多个不同领域的事件数据集上对预训练语言模型进行微调，使单一模型能够保留领域特定语义，同时在大规模和不断演变的标签空间上实现泛化。我们通过一个面向研究人员和实践者的基于网络的应用来展示这些能力。该平台支持文档上传、模式感知的事件抽取、触发词和参数的视觉化，以及跨领域不同抽取配置的比较。

    arXiv:2608.23261v1 Announce Type: new  Abstract: Event extraction is fundamental to information extraction. Prior approaches often separate event detection and argument extraction or depend on dataset-specific designs, limiting scalability and cross-domain generalization. We propose a unified generative sequence-to-sequence framework that performs event extraction subtasks jointly and supports both pipeline and end-to-end configurations. We fine-tune pretrained language models on multiple event datasets across diverse domains, enabling a single model to retain domain-specific semantics while generalizing over large and evolving label spaces. We demonstrate these capabilities through a web-based application tailored for researchers and practitioners. The platform supports document upload, schema-aware event extraction, visualization of triggers and arguments, and comparison of different extraction configurations across domains.
    
[^28]: 上下文分配定律：生成式搜索中的因果测量与闭环编排

    The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search

    [https://arxiv.org/abs/2608.23252](https://arxiv.org/abs/2608.23252)

    本文提出因果留一法探针解决RAG中证据测量幻觉，并证明迭代分配上下文比单一扩展更优，带来16.7--20.5%的召回率提升。

    

    随着检索增强生成（RAG）向多样化组合生成转变，它受到两个关键瓶颈的阻碍：证据利用的测量缺陷，以及上下文预算分配的不优化。我们依次解决这两个问题。为解决测量问题，我们揭示了一种普遍的“诊断幻觉”：标准相关性代理在硬负样本上会灾难性失效。我们用一种高效的因果留一法探针替代它们，该探针能准确隔离生成依赖，并正式校准LLM注意力的结构稀释。为解决分配问题，我们将这种因果探针部署在去混淆的因子网格中。我们证明，当前主流的单一上下文扩展策略是一个架构陷阱，会受到相关性衰减的惩罚。相反，在多个顺序生成中迭代分配计算资源，可带来组合召回率的变革性提升，绝对百分点提高16.7--20.5，且扩展稳健。

    arXiv:2608.23252v1 Announce Type: cross  Abstract: As Retrieval-Augmented Generation (RAG) shifts toward diverse portfolio generation, it is stymied by two critical bottlenecks: flawed measurement of evidence utilization, and suboptimal context budget allocation. We resolve both sequentially.   To resolve measurement, we expose a pervasive ``diagnostic illusion'': standard relevance proxies fail catastrophically on hard negatives. We replace them with an efficient causal leave-one-out probe that accurately isolates generative reliance and formally calibrates the structural dilution of LLM attention.   To resolve allocation, we deploy this causal probe in a deconfounded factorial grid. We prove that the prevailing strategy of monolithic context widening is an architectural trap penalized by relevance decay. Instead, allocating compute iteratively across multiple sequential generations drives transformative portfolio recall gains of 16.7--20.5 absolute percentage points, scaling robustly
    
[^29]: 未来查询：大型语言模型能否充当隐式医学世界模型？

    Future Querying: Can LLMs Serve as Implicit Medical World Models?

    [https://arxiv.org/abs/2608.23248](https://arxiv.org/abs/2608.23248)

    本文提出“未来查询”范式，利用端点无关训练使小型开源LLM能在非结构化临床文本上充当隐式医学世界模型，回答时间索引的患者未来查询，并匹配大型专有系统性能，支持隐私保护的本地部署。

    

    arXiv:2608.23248v1 公告类型：交叉 摘要：传统的临床预测模型依赖于特定任务的流程和精心整理的结构化数据，这些方法扩展性差且未充分利用非结构化文本。为解决这一问题，我们引入了“未来查询”这一范式，通过评估大型语言模型（LLMs）回答关于患者未来时间索引临床查询的能力，来探究它们能否充当隐式医学世界模型。我们的框架基于非结构化临床文档运行，采用端点无关的训练方式，使单一模型能够回答患者轨迹上的多样化临床查询，无需手动特征工程或特定任务的重新训练。我们证明，小型、本地微调的开源权重模型可以匹配或接近更大的专有系统，使该框架适用于隐私保护、本地部署。在合成医学报告数据集和MIMIC-IV数据集的真实ICU笔记上进行的评估中，我们的结果提供了……

    arXiv:2608.23248v1 Announce Type: cross  Abstract: Traditional clinical prediction models rely on task-specific pipelines and curated, structured data, which scale poorly and underutilize unstructured text. To address this, we introduce future querying, a paradigm that probes whether large language models (LLMs) can function as implicit medical world models by evaluating their ability to answer time-indexed clinical queries about a patient's future. Our framework operates on unstructured clinical documentation using endpoint-agnostic training, enabling a single model to answer diverse clinical queries over patient trajectories without manual feature engineering or task-specific retraining. We show that small, locally fine-tuned open-weight models can match or approach larger proprietary systems, making the framework suitable for privacy-preserving, on-premise deployment. Evaluated on a new synthetic medical reports dataset and real ICU notes from the MIMIC-IV dataset, our results provi
    
[^30]: 不确定性下的语义承诺可信大语言模型

    Credal Large Language Models for Semantic Commitment under Uncertainty

    [https://arxiv.org/abs/2608.23244](https://arxiv.org/abs/2608.23244)

    通过集成LoRA适配器构建可信集，提出CTC和SCC分数来区分认知无知与真实模糊性，从而减少LLM的过度自信错误。

    

    大型语言模型（LLMs）通常会产生流畅但错误的答案，并带有过度的自信。一个核心限制是，标准LLMs通过单一预测分布表示不确定性，将认知上的无知与真正的模糊性混为一谈。我们引入了可信大语言模型（CLLMs）：通过一组LoRA适配器的集成诱导出一个可信集，其下界和上界概率暴露了合理预测分布的扩散范围，而不是坍缩为单一的softmax输出。从这一表示中，我们推导出两个互补的承诺分数。可信令牌承诺（CTC）是一个令牌空间分数，结合了下界支持、可信宽度和交集熵，无需额外生成即可计算。语义承诺一致性（SCC）通过采样补全将承诺扩展到语义空间，其中SCC-Gap衡量令牌级和语义级支持之间的不匹配。我们评估了幻觉情况。

    arXiv:2608.23244v1 Announce Type: cross  Abstract: Large language models (LLMs) often produce fluent but incorrect answers with unwarranted confidence. A central limitation is that standard LLMs represent uncertainty through a single predictive distribution, conflating epistemic ignorance with genuine ambiguity. We introduce Credal Large Language Models (CLLMs): an ensemble of LoRA adapters induces a credal set whose lower and upper probabilities expose the spread of plausible predictive distributions rather than collapsing to a single softmax output. From this representation we derive two complementary commitment scores. Credal Token Commitment (CTC) is a token-space score that combines lower-bound support, credal width, and intersection entropy, computed without additional generation. Semantic Commitment Consistency (SCC) extends commitment to semantic space using sampled completions, with SCC-Gap measuring the mismatch between token-level and semantic-level support. We evaluate hall
    
[^31]: 一种具有显式任务与领域条件约束的多领域多任务生成框架，用于跨领域事件抽取

    A Multi-Domain and Multi-Task Generative Framework with Explicit Task and Domain Conditioning for Cross-Domain Event Extraction

    [https://arxiv.org/abs/2608.23235](https://arxiv.org/abs/2608.23235)

    本文提出一个多领域多任务生成框架，通过显式领域和任务条件信号，在单一模型中动态适应异构事件模式，无需完整事件标签集，从而提升跨领域事件抽取的泛化能力。

    

    arXiv:2608.23235v1 公告类型：新 摘要：事件抽取旨在识别事件触发器、分类事件类型并提取参数，以构建结构化事件表示。尽管在特定领域内表现强劲，但由于上下文表达和事件模式的差异，开发能够跨领域稳健泛化的模型仍具挑战性。先前的统一和多任务方法提高了领域内准确性，但在应用于未见领域时灵活性有限。即使是基于大型语言模型的方法，在推理时提供完整事件本体，也往往不如较小的、任务特定的微调模型。我们提出了一种统一的多领域和多任务训练框架，在单一模型中建模异构事件模式。我们的方法引入领域条件信号，与任务特定提示相结合，实现动态适应数据集特定模式，而无需在推理时提供完整事件标签集。

    arXiv:2608.23235v1 Announce Type: new  Abstract: Event extraction aims to identify event triggers, classify event types, and extract arguments to construct structured event representations. Despite strong in-domain performance, developing models that generalize robustly across domains remains challenging due to variations in contextual expressions and event schemas. Prior unified and multi-task approaches improve in-domain accuracy but exhibit limited flexibility when applied to unseen domains. Even large language model-based methods that provide full event ontologies at inference time often underperform compared to smaller, task-specific fine-tuned models. We propose a unified multi-domain and multi-task training framework that models heterogeneous event schemas within a single model. Our approach introduces domain conditioning signals, jointly with task-specific prompts, enabling dynamic adaptation to dataset-specific schemas without requiring complete event label sets at inference t
    
[^32]: 生物医学文本与知识图谱对齐：轻量级对齐策略的系统比较

    Aligning Biomedical Texts and Knowledge Graphs: A Systematic Comparison of Lightweight Alignment Strategies

    [https://arxiv.org/abs/2608.23214](https://arxiv.org/abs/2608.23214)

    本文提出一个统一框架，通过冻结文本和知识图谱模型并学习轻量级投影，系统比较六种设计维度以对齐生物医学文本与知识图谱，并构建了CTD-Align数据集。

    

    生物医学知识以两种互补但不同的形式存在：非结构化的科学文献和结构化的知识图谱（KGs）。将它们对齐对于知识基础、证据检索和知识图谱补全至关重要，但现有方法并未明确地将自由文本证据与知识图谱三元组对齐。我们提出了一个统一框架，用于系统地研究生物医学文本与知识图谱对齐的设计选择。在文本编码器和知识图谱嵌入模型均冻结的情况下，我们仅通过对比目标学习它们空间之间的轻量级投影。这使得我们能够在六个设计维度上进行公平比较：文本编码器、知识图谱嵌入模型、投影头、三元组组合、训练方向和难负样本采样。我们构建了CTD-Align，一个包含超过22,000对一对一三元组-文档对的数据集，将比较毒理学数据库中的化学-基因相互作用与支持的PubMed段落关联起来。我们进行了评估。

    arXiv:2608.23214v1 Announce Type: new  Abstract: Biomedical knowledge exists in two complementary but distinct forms: unstructured scientific literature and structured knowledge graphs (KGs). Aligning them is essential for knowledge grounding, evidence retrieval, and KG completion, yet existing methods do not explicitly align free-text evidence with KG triples. We present a unified framework for systematically studying design choices for aligning biomedical text and KGs. With a text encoder and a KG embedding model both frozen, we learn only a lightweight projection between their spaces via a contrastive objective. This enables a fair comparison across six design dimensions: text encoder, KG embedding model, projection head, triple composition, training direction, and hard-negatives sampling. We construct CTD-Align, a corpus of over 22K one-to-one tripledocument pairs linking chemical-gene interactions from the Comparative Toxicogenomics Database to supporting PubMed passages. We evalu
    
[^33]: 基于布鲁姆分类法的大推理模型推理轨迹认知剖析

    Cognitive Profiling of LRMs' Reasoning Traces Using Bloom's Taxonomy

    [https://arxiv.org/abs/2608.23205](https://arxiv.org/abs/2608.23205)

    本文提出了一种基于布鲁姆分类法的自动标注框架，用于分析大型推理模型的推理步骤思维类型，并跨模型和数据集揭示其思维模式异同，为推理行为提供了新洞察。

    

    arXiv:2608.23205v1 公告类型：交叉 摘要：大型推理模型（LRMs）彻底改变了LLM中的推理能力，而推理轨迹日益公开可得，为研究模型行为创造了宝贵机会，不仅限于表面层面，还能深入到单个推理步骤的粒度。然而，理解推理过程中所采用的思维类型——这能提供对模型推理模式的关键洞察，并支持可操作的应用——仍未被充分探索。为解决这一空白，我们引入了一个框架，通过布鲁姆分类法的视角自动标注推理步骤，该分类法将思维划分为六个认知层次，如记忆、应用和评价。利用此框架，我们跨模型和数据集进行了大规模分析，揭示了模型和任务之间思维模式的相似性与差异性。此外，我们证明了从推理中得出的思维类型信息能用于实际应用。

    arXiv:2608.23205v1 Announce Type: cross  Abstract: Large Reasoning Models (LRMs) have revolutionized reasoning in LLMs, and the increasing public availability of reasoning traces creates valuable opportunities to study model behavior not only at the surface level but also at the granularity of individual reasoning steps. However, understanding the types of thinking employed during reasoning - which offers critical insights into models' reasoning patterns and enables actionable applications - remains underexplored. To address this gap, we introduce a framework for automatic annotation of reasoning steps through the lens of Bloom's Taxonomy, which classifies thinking into six cognitive levels, such as Remembering, Applying and Evaluating. Using this framework, we perform a large-scale analysis across models and datasets, revealing both similarities and differences in thinking patterns across models and tasks. Moreover, we demonstrate that thinking-type information derived from reasoning 
    
[^34]: LongWoF-Bench：用于可验证长工作流任务的EvoMap基因评估基准

    LongWoF-Bench: Evaluating EvoMap Genes for Verifiable Long-Workflow Tasks

    [https://arxiv.org/abs/2608.23200](https://arxiv.org/abs/2608.23200)

    本文提出LongWoF-Bench基准和EvoMap方法，通过将验证器确认的执行轨迹整合为结构化基因，实现经验复用，在可验证长工作流任务中显著优于技能方法。

    

    arXiv:2608.23200v1 公告类型：新 摘要：大型语言模型日益被期望执行复杂工作流，其成功依赖于维护相互关联的约束条件，并生成满足严格端到端验证的工件。然而，成功的执行经验通常在单次运行后丢失，迫使后续模型从头重新发现策略和失败模式。我们研究这种经验是否可以通过EvoMap外部化并复用，其中验证器确认的执行轨迹被整合成结构化基因。为评估此设置，我们引入了长工作流基准（LongWoF-Bench），包含778个可机器验证的任务，涵盖代码生成、智能体环境合成、数学推理和规则遵循。在252个具有验证器确认的Opus轨迹的任务上，进化的EvoMap基因在所有七个评估模型中比技能方法平均高出8.7-15.5个百分点，且这些优势延伸至未见任务。

    arXiv:2608.23200v1 Announce Type: new  Abstract: Large language models are increasingly expected to execute complex workflows whose success depends on maintaining interdependent constraints and producing artifacts that satisfy strict end-to-end verification. Yet successful execution experience is typically lost after a single run, forcing subsequent models to rediscover strategies and failure modes from scratch. We study whether such experience can instead be externalized and reused through EvoMap, where verifier-confirmed execution trajectories are consolidated into structured Gene. To evaluate this setting, we introduce the Long-Workflow Benchmark (LongWoF-Bench), comprising 778 machine-verifiable tasks across code generation, agent-environment synthesis, mathematical reasoning, and rule following. On the 252 tasks with verifier-confirmed Opus trajectories, evolved EvoMap Gene outperform Skill across all seven evaluated models by 8.7-15.5 percentage points, with the gains extending t
    
[^35]: CyberFactory：利用真实世界实例扩展网络安全能力

    CyberFactory: Scaling Cyber Security Capabilities with Instances from the Wild

    [https://arxiv.org/abs/2608.23181](https://arxiv.org/abs/2608.23181)

    CyberFactory是一个统一开源框架，通过将真实世界CVE漏洞转化为可执行任务实例，并整合数据构建、轨迹合成和模型训练，从而扩展网络安全能力。

    

    随着大型语言模型（LLMs）在编码能力上的不断进步，它们在网络安全领域的潜力日益受到研究关注，其中闭源LLMs（如Mythos）展现了先进的网络安全能力。然而，现有的开源工作仍存在局限：前沿开源权重模型未提供可复现的网络安全训练解决方案，开源训练方案聚焦于孤立任务且缺乏可扩展的代理数据，而扩展代理式滚动需要强大的领域先验知识。在这项工作中，我们引入了\textbf{CyberFactory}，一个统一的开源框架，它在概念验证（PoC）生成、漏洞修补和网络安全问答（CyberQA）之间连接了数据构建、轨迹合成和模型训练。CyberFactory将公开的漏洞工件（包括来自真实世界的CVE）转化为可执行且可验证的任务实例。它进一步使用...

    arXiv:2608.23181v1 Announce Type: cross  Abstract: As large language models (LLMs) continue to advance in coding capabilities, their potential in cybersecurity has drawn increasing research attention, with closed-source LLMs (e.g., Mythos) delivering advanced cybersecurity capabilities. However, existing open-source efforts remain limited: frontier open-weight models do not provide reproducible cybersecurity training solutions, open-source training solutions focus on isolated tasks and lack scalable agentic data, and scaling agentic rollouts requires strong domain priors. In this work, we introduce \textbf{CyberFactory}, a unified open-source framework that connects data construction, trajectory synthesis, and model training across proof-of-concept (PoC) generation, vulnerability patching, and cybersecurity question answering (CyberQA). CyberFactory transforms public vulnerability artifacts, including CVEs from the wild, into executable and verifiable task instances. It further uses a 
    
[^36]: CaRGo-T：因果推理思维图提升多模态幽默理解

    CaRGo-T: Causal Reasoning Graph-of-Thought improves Multimodal Humor Comprehension

    [https://arxiv.org/abs/2608.23172](https://arxiv.org/abs/2608.23172)

    CaRGo-T提出了一种基于图的因果推理框架，通过将多模态幽默中的复杂关系序列化为代码表示，提升了视觉-语言模型在幽默理解任务中的推理能力。

    

    大规模视觉-语言模型（VLMs）在广泛的多模态任务中展现了显著的通用性。然而，理解幽默仍然具有挑战性，因为幽默内容通常依赖于实体、事件、上下文以及图像和文本模态间隐含关系之间的微妙交互。这些交互可能涉及复杂的推理链，难以通过传统提示或线性链式思维推理来捕捉。在这项工作中，我们提出了CaRGo-T（因果推理思维图），这是一种推理框架，将多模态幽默背后的因果和上下文关系表示为轻量级的基于图的推理结构。该图被序列化为由VLM生成的基于代码的表示，随后可由相同或不同的VLM解释，以在零样本或上下文学习设置中产生最终预测。我们评估了Ca

    arXiv:2608.23172v1 Announce Type: new  Abstract: Large-scale vision-language models (VLMs) have demonstrated remarkable versatility across a wide range of multimodal tasks. However, understanding humor remains challenging because humorous content often depends on subtle interactions among entities, events, context, and implicit relationships across image and text modalities. These interactions can involve complex chains of reasoning that are difficult to capture through conventional prompting or linear chain-of-thought reasoning. In this work, we propose CaRGo-T (Causal Reasoning Graph-of-Thought), a reasoning framework that represents the causal and contextual relationships underlying multimodal humor as a lightweight graph-based reasoning structure. The graph is serialized into a code-based representation generated by a VLM, which can subsequently be interpreted by the same or a different VLM to produce the final prediction in zero-shot or in-context learning settings. We evaluate Ca
    
[^37]: 通过结构化后缀建模加速扩散语言模型

    Accelerating Diffusion Language Models via Structured Suffix Modeling

    [https://arxiv.org/abs/2608.23167](https://arxiv.org/abs/2608.23167)

    本文提出了一种结构化后缀建模方法，通过将后缀划分为局部、中部和尾部区域并自适应保留不同数量的标记，以及利用前一步解码结果进行初始化，从而显著提升扩散语言模型的推理效率。

    

    arXiv:2608.23167v1 公告类型：新论文 摘要：扩散语言模型（DLMs）通过单步生成中同时去噪多个标记，展现出强大的并行解码能力。然而，这种并行性带来了显著的计算开销，因为每一步都需要与所有后缀标记进行交互。现有方法通常通过仅保留局部后缀窗口作为完整后缀的替代来减少这种成本。尽管这些方法有效，但它们忽视了后缀区域的结构异质性，并在每个时间步用相同的表示重新初始化后缀标记。为此，我们提出了一种结构化后缀建模方法，用于高效的DLM推理。具体来说，我们将后缀划分为三个区域，即局部、中部和尾部区域，并根据其结构角色在每个区域中保留不同数量的后缀标记。此外，我们将前一步的解码结果融入后缀标记的重新初始化中。

    arXiv:2608.23167v1 Announce Type: new  Abstract: Diffusion Language Models (DLMs) exhibit strong parallel decoding capabilities by denoising multiple tokens in a single generation step. However, this parallelism comes with substantial computational overhead, as each step requires interactions with all suffix tokens. Existing methods typically reduce this cost by retaining only a local suffix window as a substitute for the full suffix. Despite their effectiveness, these methods overlook the structural heterogeneity across suffix regions and re-initialize suffix tokens with identical representations at each timestep. To this end, we propose a structured suffix modeling method for efficient DLM inference. Specifically, we divide the suffix into three regions, i.e., the local, middle, and tail regions, and retain different numbers of suffix tokens in each region according to their structural roles. Moreover, we incorporate the decoding results from the previous step into the suffix token r
    
[^38]: 以证据反击！一种面向仇恨类别知情反驳生成的多智能体记忆高效推理框架

    Counter with Evidence! A Multi-Agent Memory Efficient Reasoning Framework for Hate Category Informed Counterspeech Generation

    [https://arxiv.org/abs/2608.23152](https://arxiv.org/abs/2608.23152)

    本文提出FIRE框架，通过将仇恨言论分类为五种具体类别并映射到针对性反驳风格，结合新数据集FactualCS，实现了更精准且基于证据的反驳生成。

    

    反驳言论能有效削弱在线仇恨的影响。尽管先前研究探索了自动化反驳生成，但大多强调风格控制，并将仇恨言论视为同质化，忽视了不同形式的虐待需要根本不同的反驳策略。为填补这一空白，我们引入了FIRE（事实知情多智能体推理框架），该框架首先将仇恨言论分解为五种不同类别之一（错误信息、刻板印象、阴谋论、非人化、非事实），然后将其映射到针对性的反驳风格。为支持FIRE，我们构建了FactualCS，一个包含4,784个实例的新数据集，提供了关于仇恨类别、推理轨迹和证据映射的标注，这些是先前工作中缺失的、对基于事实的生成至关重要的元素。在28个基线配置上的全面评估表明，FIRE显著优于现有方法。

    arXiv:2608.23152v1 Announce Type: new  Abstract: Counterspeech effectively neutralizes the impact of online hate. Although prior work explores automated counterspeech generation, it largely emphasizes stylistic control while treating hate speech as homogeneous, overlooking that distinct forms of abuse require fundamentally different counterspeech strategies. To address this gap, we introduce FIRE (Factuality Informed Multi-Agent Reasoning Framework) that first decomposes hate speech into one of the five distinct categories (misinformation, stereotype, conspiracy, dehumanizing, non-factual), and then maps it to a targeted counterspeech style. To facilitate FIRE, we curate FactualCS, a novel dataset of $4,784$ instances that provides the annotations regarding hate categories, reasoning traces, and evidence mappings, which are critical elements for grounded generation that are missing in prior work. A comprehensive evaluation across $28$ baseline configurations demonstrates that FIRE sign
    
[^39]: 语言链在对齐中的应用：跨语言排序偏好优化

    Language Chain in Alignment: Cross-Lingual Ranking Preference Optimization

    [https://arxiv.org/abs/2608.23149](https://arxiv.org/abs/2608.23149)

    本文提出跨语言排序偏好优化（CRPO）框架，通过利用英语偏好知识的分层结构，在目标语言中实现更优的语言对齐和输出质量。

    

    arXiv:2608.23149v1 公告类型：交叉 摘要：大型语言模型的对齐在很大程度上依赖于以英语为中心的高质量偏好数据，这通常会导致在其他语言中的性能欠佳。在本文中，我们提出了跨语言排序偏好优化（CRPO），这是一个新颖的框架，利用来自英语的稳健偏好知识来促进目标语言中的偏好对齐。我们在目标语言和英语之间的平行偏好对中设计了一种分层结构，以联合优化语言内和语言间的偏好，从而增强语言适应性和输出质量。基于LambdaLoss框架，CRPO超越了基于二元比较的优化，通过提供多个候选响应之间的相对排序信号。我们在五种资源规模不同的语言上进行的实验表明，CRPO在指令遵循和知识方面均持续优于标准方法。

    arXiv:2608.23149v1 Announce Type: cross  Abstract: The alignment of Large Language Models heavily relies on English-centric high-quality preference data, which often leads to suboptimal performance in other languages. In this paper, we propose Cross-Lingual Ranking Preference Optimization (CRPO), a novel framework that leverages robust preference knowledge from English to facilitate preference alignment in the target language. We design a hierarchical structure within parallel preference pairs across the target language and English to jointly optimize intra- and inter-lingual preferences, thereby enhancing language adaptation and output quality. Building on the LambdaLoss framework, CRPO goes beyond the binary comparison based optimization by providing a relative ranking signal across multiple candidate responses. Our experiments across five languages with varying resource scales demonstrate that CRPO consistently outperforms standard approaches in both instruction-following and knowle
    
[^40]: 激活加权种子残差编码用于低比特大语言模型权重修复

    Activation-Weighted Seeded Residual Coding for Low-Bit LLM Weight Repair

    [https://arxiv.org/abs/2608.23144](https://arxiv.org/abs/2608.23144)

    本文提出激活加权种子残差编码（AWSRC），通过利用激活统计和种子生成基，以极小辅助存储（约0.8%权重负载）高效修复低比特量化误差，显著提升模型质量。

    

    arXiv:2608.23144v1 公告类型：交叉 摘要：低比特权重量化节省存储，但会引入误差，降低语言模型质量。我们提出激活加权种子残差编码（AWSRC），这是一种用于现有量化骨干网络的紧凑修复编解码器。给定重构权重 $W_0$，AWSRC 使用确定性种子生成的基编码残差 $W-W_0$。辅助存储保存种子选择器、低比特系数和缩放因子，而非显式码本。激活统计优先处理影响层输出的误差。在Qwen2.5-3B-Instruct上，向INT4 RTN骨干网络添加0.162 scope-bits/权重，可弥合与BF16相比的匹配困惑度、KL散度和准确率差距的88.2%、78.9%和71.3%。修复匹配的强低比特骨干网络也改善了所有测量质量指标。使用匹配的49.25 MB辅助存储（约占BF16模型权重负载的0.8%），AWSRC在稀疏、低秩和向量量化编解码器中提供了最佳困惑度和平均任务准确率。

    arXiv:2608.23144v1 Announce Type: cross  Abstract: Low-bit weight quantization saves storage but leaves errors that degrade language-model quality. We introduce Activation-Weighted Seeded Residual Coding (AWSRC), a compact repair codec for an existing quantization backbone. Given a reconstructed weight $W_0$, AWSRC encodes the residual $W-W_0$ using deterministic seed-generated bases. The sidecar stores seed selectors, low-bit coefficients, and scales rather than an explicit codebook. Activation statistics prioritize errors that affect layer outputs. On Qwen2.5-3B-Instruct, adding 0.162 scope-bits/weight to an INT4 RTN backbone closes 88.2%, 78.9%, and 71.3% of the matched PPL, KL, and accuracy gaps to BF16. Repairing a matched strong low-bit backbone also improves all measured quality metrics. With a matched 49.25 MB sidecar, about 0.8% of the BF16 model-weight payload, AWSRC gives the best perplexity and mean task accuracy among sparse, low-rank, and vector-quantized codecs.
    
[^41]: 文学大五人格：统一可解释空间中的作者个性化文本生成

    LITERARYBIGFIVE: Author-Personalized Text Generation in a Unified Interpretable Space

    [https://arxiv.org/abs/2608.23124](https://arxiv.org/abs/2608.23124)

    本文提出LiteraryBigFive框架，将作者写作特征映射到统一可解释空间中，通过激活对比生成风格维度，实现无需大规模标注或微调的跨作者个性化文本生成。

    

    针对作者和文学写作的个性化文本生成对于自适应写作助手、创意支持工具和计算文学分析等应用至关重要。然而，现有的作者建模和个性化方法通常将写作行为表示为独立的标签，需要为每位作者或每种风格类别收集大规模语料或进行微调。这种表述成本高昂、难以解释，且不利于跨作者的泛化。受大五人格模型维度化视角的启发，我们提出了LiteraryBigFive框架，将作者写作特征重新定义为统一且可解释空间中的坐标。在该空间中，我们通过作者写作段落与中性段落之间激活空间中的对比，推导出每个可解释的轴（如古典主义、情感性），从而产生独特的风格维度，使文本或作者得以被表示。

    arXiv:2608.23124v1 Announce Type: cross  Abstract: Personalized text generation for authors and literary writing is essential for applications such as adaptive writing assistants, creative support tools, and computational literary analysis. However, existing approaches to author modeling and personalization often represent writing behavior as independent labels, requiring large-scale corpus collection or fine-tuning for each author or stylistic category. Such formulations are costly, difficult to interpret, and poorly suited for generalizing across authors. Inspired by the Big Five model's dimensional view of personality, we propose LiteraryBigFive, a framework that reframes authorial writing characteristics as coordinates within a unified and interpretable space. In this space, we derive each interpretable axis (e.g., Classicism, Emotionality) from activation-space contrasts between author-written and neutral passages, yielding distinct stylistic dimensions that allow texts or authors
    
[^42]: 英语-普纳尔语对统计机器翻译系统：实证研究的一些见解

    Statistical Machine Translation Systems of English-Pnar Language Pair : Some Insights of the Emperical Study

    [https://arxiv.org/abs/2608.23120](https://arxiv.org/abs/2608.23120)

    本文首次为英语-普纳尔语语言对构建平行语料库并训练统计机器翻译系统，建立了该语言对的第一个定量基准。

    

    普纳尔语是一种南亚语系语言，由梅加拉亚邦贾因蒂亚山区的约40万人使用，缺乏数字语料库和自然语言处理资源。本文首次对英语和普纳尔语语言对进行了机器翻译研究。我们利用从Wyrta报纸收集的文章，构建了一个包含10,234个句子的平行语料库，并使用9,563个平行语料在三种配置下，针对每个方向使用Moses、GIZA++、KenLM，并变化词汇化重排序和最小错误率训练（MERT）调优，训练了基于短语的统计机器翻译（SMT）系统。模型在371个句子的保留测试集上进行评估，最佳系统在普纳尔语到英语方向上取得了14.97的BLEU分数（chrF2：33.42，TER：77.60），在英语到普纳尔语方向上取得了11.16的BLEU分数（chrF2：31.38，TER：93.51），为该语言对建立了首个定量基准。

    arXiv:2608.23120v1 Announce Type: cross  Abstract: Pnar, an Austroasiatic language spoken by approximately 0.4 million people in the Jaintia Hills of Meghalaya, lacks the digital corpora and natural language processing (NLP) resources. This paper presents the first machine translation study for the English and Pnar language pair. Using articles collected from the Wyrta newspaper, we built a parallel corpus comprising of 10,234 sentences and trained phrase-based statistical machine translation (SMT) systems the models using 9,563 parallel corpora under three configurations for each direction using Moses, GIZA++ , KenLM, varying lexicalized reordering and minimum error rate training (MERT) tuning. The models are evaluated on a held out test set of 371 sentences, the best performing system achieves a BLEU score of 14.97 (chrF2: 33.42, TER: 77.60) for Pnar to English and 11.16 (chrF2: 31.38, TER: 93.51) for English to Pnar, establishing the first quantitative benchmark for this language pa
    
[^43]: 分子大语言模型智能体：从架构设计到科学自主性

    Molecular LLM Agents: From Architectural Design to Scientific Autonomy

    [https://arxiv.org/abs/2608.23104](https://arxiv.org/abs/2608.23104)

    本文提出了分子大语言模型智能体的概念框架，从架构设计和科学自主性阶梯两个视角，系统阐述了分子感知、智能体框架、工具接地及学习优化，并定义了从基础工具使用到完全自主科学研究的分级能力路径。

    

    分子科学代表了基于大语言模型智能体的重要前沿领域。与主要在自然语言、代码或网络环境中操作的一般智能体不同，分子大语言模型智能体必须感知、推理并作用于跨越符号字符串、分子图、三维构象、光谱、模拟和湿实验测量的化学对象。其能力取决于化学保真的分子感知、以LLM为中心的智能体框架、领域特定工具接地以及计算或实验反馈，此外还需规划与工具使用。本研究从两个互补视角为分子大语言模型智能体构建了概念框架。首先，我们引入分子智能体设计的架构视角，涵盖分子表示与感知、智能体框架、领域特定工具箱以及学习与优化。其次，我们提出一个受分级自主启发的科学自主性阶梯。

    arXiv:2608.23104v1 Announce Type: cross  Abstract: Molecular science represents an important frontier for LLM-based agents. Unlike general agents that mainly operate over natural language, code, or web environments, molecular LLM agents must perceive, reason about, and act upon chemical objects across symbolic strings, molecular graphs, 3D conformations, spectra, simulations, and wet-lab measurements. Their capabilities depend on chemically faithful molecular perception, an LLM-centered agent framework, domain-specific tool grounding, and computational or experimental feedback, in addition to planning and tool use. This work develops a conceptual framework for molecular LLM agents from two complementary perspectives. First, we introduce an architectural view of molecular-agent design, covering molecular representation and perception, the agent framework, domain-specific toolboxes, and learning and optimization. Second, we propose a scientific autonomy ladder inspired by staged autonomy
    
[^44]: 媒体偏见检测中的定义敏感性：一个多定义数据集与基准

    Definitional Sensitivity in Media Bias Detection: A Multi-Definition Dataset and Benchmark

    [https://arxiv.org/abs/2608.23095](https://arxiv.org/abs/2608.23095)

    该研究通过大规模人类和LLM实验发现，媒体偏见检测中的定义概念框架会显著影响标注结果，且对LLM影响更强，而构念保留的阐述则无此效应。

    

    arXiv:2608.23095v1 公告类型：新 摘要：媒体偏见检测依赖于定义和示例来明确何为偏见，但这些规范在不同数据集间往往存在差异，或即使名称相同也常常隐而不显。这种差异使得为同一偏见类别训练的模型是否学习到相同的构念或不同的现象变得不明确，这一问题在先前研究中被广泛忽视。我们通过一项包含354名参与者的受试者间实验和四项大语言模型（LLM）的平行评估，考察了定义选择如何影响偏见标注。参与者和模型使用在概念框架和阐述程度上有所变化的定义，对六篇新闻文章在四个偏见类别上进行评分。在8,496个人类评分和28,800个LLM评分中，我们发现定义的概念目标驱动了标注分歧，而保留构念的阐述则不会：概念框架显著改变了人类的标注，并且对LLM的影响更强。我们对此进行了讨论。

    arXiv:2608.23095v1 Announce Type: new  Abstract: Media bias detection relies on definitions and examples that specify what counts as bias, yet these specifications often vary across datasets or remain implicit, even when given the same name. Such variation makes it unclear whether models trained for the same bias category learn the same construct or different phenomena, a problem largely overlooked in prior work. We examine how definition choice affects bias annotation in a between-subjects experiment with 354 participants and a parallel evaluation with four LLMs. Participants and models rate six news articles across four bias categories using definitions that vary in conceptual framing and elaboration. Across 8,496 human and 28,800 LLM ratings, we find that the conceptual target of a definition drives annotation divergence, while construct-preserving elaboration does not: conceptual framing significantly shifts annotations for humans and does so even more strongly for LLMs. We discuss
    
[^45]: AgentWeave：在推理前进行路由以高效调用工具丰富型语言模型中的函数

    AgentWeave: Routing Before Reasoning for Efficient Function Calling in Tool-Rich Language Models

    [https://arxiv.org/abs/2608.23078](https://arxiv.org/abs/2608.23078)

    AgentWeave通过在推理前使用确定性路由层减少候选工具集，在不改变下游模型的情况下显著提升多函数调用任务的成功率。

    

    大型语言模型日益在大量工具、函数、API和专用代理上运行。随着候选动作空间的增长，函数调用模型必须处理更多的模式、消耗更多的提示令牌，并区分越来越相似或无关的替代方案。我们研究了一种互补的系统策略：在语言模型推理之前减少候选集，同时保持下游模型不变。我们引入了AgentWeave，一个确定性的推理前路由层，利用资格、需求、能力和路由信号构建一个有界的模型可见动作空间。我们使用冻结的BFCL派生路由压力协议和公共MadeAgents/Hammer2.1-1.5b模型对AgentWeave进行了评估。在48个新的BFCL V4多函数任务中，AgentWeave实现了6/48（12.5%）的原生BFCL成功，而所有工具、确定性随机前8和语义前8基线分别实现了...

    arXiv:2608.23078v1 Announce Type: new  Abstract: Large language models increasingly operate over large collections of tools, functions, APIs, and specialized agents. As the candidate action space grows, a function-calling model must process more schemas, consume more prompt tokens, and distinguish among increasingly similar or irrelevant alternatives. We study a complementary systems strategy: reduce the candidate set before language-model inference while leaving the downstream model unchanged. We introduce AgentWeave, a deterministic pre-inference routing layer that constructs a bounded model-visible action space using eligibility, requirement, capability, and routing signals. We evaluate AgentWeave with a frozen BFCL-derived routing-pressure protocol using the public MadeAgents/Hammer2.1-1.5b model. On 48 fresh BFCL V4 multiple-function tasks, AgentWeave achieves 6/48 (12.5%) native BFCL successes, whereas all-tools, deterministic random top-8, and semantic top-8 baselines each achie
    
[^46]: 信号还是噪声？Web开发中代理技能的一项基准研究

    Signal or Noise? A Benchmark Study of Agent Skills in Web Development

    [https://arxiv.org/abs/2608.23067](https://arxiv.org/abs/2608.23067)

    该研究通过引入WebDev-Skills-Bench基准，发现注入代理技能在Web开发任务中不仅无益，反而降低了任务成功率并增加了资源消耗，表明技能注入需谨慎评估。

    

    arXiv:2608.23067v1 公告类型：新 摘要：代理技能是可重用的程序性模块，越来越多地被注入到编码代理会话中，以编码框架约定、反模式以及可重用工具。然而，由于每个注入的技能都会扩展每次查询的提示，一个有效的技能基准不仅必须确定代理是否能解决任务，还必须确定该技能是否应该被注入。我们引入了WebDev-Skills-Bench，并利用它对50个Web-Bench项目和1000个有序任务中的31个公共Web开发技能进行了受控实证研究。该基准比较了四种匹配条件，包括长度匹配的无关控制和留一法组件消融。为了隔离技能效应与提示长度伪影，我们仅在提示中放置SKILL.md，同时将辅助文件挂载到代理工作区。在四个模型中，目标技能注入使平均Pass@2降低了1.3%至4.2%，降低了任务完成深度，并增加了令牌消耗。

    arXiv:2608.23067v1 Announce Type: new  Abstract: Agent Skills are reusable procedural modules that are increasingly injected into coding-agent sessions to encode framework conventions, anti-patterns, and reusable tools. However, because each injected Skill expands the prompt of every query, an effective Skill benchmark must determine not only whether an agent can solve a task, but whether the Skill should have been injected at all. We introduce WebDev-Skills-Bench and use it for a controlled empirical study of 31 public WebDev Skills on 50 Web-Bench projects and 1,000 ordered tasks. The benchmark compares four matched conditions, including a length-matched irrelevant control and leave-one-out component ablations. To isolate Skill effects from prompt-length artifacts, we place only SKILL.md in the prompt while mounting auxiliary files into the agent workspace. Across four models, target Skill injection reduces mean Pass@2 by 1.3% to 4.2%, lowers task completion depth, and increases toke
    
[^47]: 文化时刻基准：评估东南亚视频文化推理与定位能力

    Cultural Moment Benchmark: Evaluating Video Cultural Reasoning and Grounding in Southeast Asia

    [https://arxiv.org/abs/2608.23065](https://arxiv.org/abs/2608.23065)

    本文提出了文化时刻基准（CMB），通过三个阶段分别评估视频文化理解中的命名、视觉识别和时间定位能力，填补了现有基准混淆这些能力的空白。

    

    arXiv:2608.23065v1 公告类型：交叉 摘要：视频中的文化理解不仅仅是识别可见内容；它需要把握文化概念的象征意义和时间意义。我们将此分解为三种能力：命名概念所象征的内容、在视频中视觉识别它、以及定位其子事件的时间位置。现有的视频文化基准往往只测试可见内容，将这三种能力合并为一个分数，从而掩盖了瓶颈。我们引入了文化时刻基准（CMB）：包含来自东南亚七个国家、五个类别的306个专家策划概念。我们通过三个阶段评估每个概念，每个阶段对应一种能力。给定一个描述，阶段1（S1）从四个候选概念名称中选择，阶段2（S2）从四个候选视频时刻中选择，阶段3（S3）预测视频中时刻的开始和结束时间。为了使每个阶段专注于独特能力，我们采用了三种设计选择：语义相似性...

    arXiv:2608.23065v1 Announce Type: cross  Abstract: Cultural understanding in video means more than recognizing what is visible; it requires grasping the symbolic and temporal significance of cultural concepts. We decompose this into three abilities: naming what a concept symbolizes, visually recognizing it on video, and locating its sub-events in time. Existing video-cultural benchmarks tend to test what is seen, collapsing these three abilities into a single score that hides the bottleneck. We introduce the Cultural Moment Benchmark (CMB): 306 expert-curated concepts from seven countries in Southeast Asia across five categories. We evaluate each concept through three stages, one per ability. Given a description, Stage 1 (S1) selects from four candidate concept names, Stage 2 (S2) selects from four candidate video moments, and Stage 3 (S3) predicts the start and end times of the moment in a video. To keep each stage focused on a distinct ability, we use three design choices: semantic-s
    
[^48]: 超越判定：基于图的人类与LLM科学事实核查推理分析

    Beyond Verdicts: A Graph-Based Analysis of Human and LLM Reasoning in Scientific Fact-Checking

    [https://arxiv.org/abs/2608.23047](https://arxiv.org/abs/2608.23047)

    本文提出一种基于图的推理框架，用于系统比较人类专家与大型语言模型在科学事实核查中的推理路径，从而揭示模型是否通过相同或不同的有效推理过程得出结论。

    

    arXiv:2608.23047v1 公告类型：交叉 摘要：引用合法论文的虚假信息在扭曲这些研究实际报告内容时可能尤为有害。尽管现有基于大型语言模型（LLMs）的自动事实核查系统能够评估模型是否给出“不正确”的判定，并生成该决策的解释，但它们通常不指示模型是否遵循与人类专家相同的推理路径，还是通过不同但有效的路径得出判定。在本工作中，我们引入了一个基于图的框架（类型化推理图），用于比较科学事实核查中人类与LLM的推理路径。基于先前关于生物医学虚假信息中谬误推理的研究MISSCIPLUS（Glockner等人，2025），我们将每个解释建模为一个推理图，该图将虚假声明与相关研究背景、研究发现、支持谬误的前提以及谬误标签联系起来。这种表示使我们能够...

    arXiv:2608.23047v1 Announce Type: cross  Abstract: Misinformation that cites legitimate papers can be especially harmful when it distorts what those studies actually report. While existing automatic fact-checking systems based on large language models (LLMs) can assess whether a model assigns an Incorrect verdict and can gen- erate explanations for that decision, they typi- cally do not indicate whether the model follows the same reasoning path as human experts or arrives at the verdict through a different but still valid path. In this work, we introduce a graph- based framework (typed reasoning graph) for comparing human and LLM reasoning paths in scientific fact-checking. Building on prior work on fallacious reasoning in biomedical misinformation, MISSCIPLUS (Glockner et al., 2025), we model each explanation as a rea- soning graph that links the false claim to the relevant study context, study findings, fallacy- supporting premises, and fallacy labels. This representation enables one
    
[^49]: AutoSaddler：基于代理执行轨迹的持久更新自动框架优化

    AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces

    [https://arxiv.org/abs/2608.23041](https://arxiv.org/abs/2608.23041)

    AutoSaddler通过将框架优化视为离线学习问题，利用失败轨迹诊断和代码式补丁生成，自动迭代改进代理框架，在多个基准上显著提升性能。

    

    arXiv:2608.23041v1 公告类型：新  摘要：大型语言模型代理在长期任务中仍不可靠，微小的局部失败可能在长时间交互中累积并导致整体任务失败。尽管外部框架能显著提升鲁棒性，但框架设计仍是一个手动且昂贵的过程，需要在大量提示、工具配置和控制逻辑的搜索空间中进行探索。我们提出AutoSaddler，一种自动框架优化框架，将框架改进形式化为离线学习问题，并利用小批量中的失败信号迭代更新框架。AutoSaddler结合了失败轨迹诊断、将框架视为代码的结构化补丁生成，以及基于验证的更新选择。在GAIA2、SWE-Bench Pro和Terminal-Bench 2.0上的实验表明，AutoSaddler显著提升了代理在对应基础框架上的性能，分别实现了9.0、9.6和10.0个百分点的增益。

    arXiv:2608.23041v1 Announce Type: new  Abstract: LLM agents remain unreliable on long-horizon tasks, where small local failures can compound over extended interactions and lead to overall task failure. Although external harnesses can substantially improve robustness, harness design remains a manual and expensive process that requires searching over a large space of prompts, tool configurations, and control logic. We propose AutoSaddler, an automatic harness optimization framework that formulates harness improvement as an offline learning problem and iteratively updates the harness using failure signals from mini-batches. AutoSaddler combines failure-trace diagnosis, structured patch generation that treats the harness as code, and validation-based update selection. Experiments on GAIA2, SWE-Bench Pro, and Terminal-Bench 2.0 show that AutoSaddler substantially improves agent performance over the corresponding base harnesses, achieving gains of 9.0, 9.6, and 10.0 percentage points, respec
    
[^50]: 多语言FrameNet语料库

    The Multilingual FrameNet Corpus

    [https://arxiv.org/abs/2608.23037](https://arxiv.org/abs/2608.23037)

    本文构建了包含九种语言的多语言FrameNet语料库（mFNC），通过多语言训练数据显著提升了框架语义解析的性能，超越了现有最先进模型。

    

    arXiv:2608.23037v1 公告类型：交叉 摘要：本文介绍了多语言FrameNet语料库（mFNC），这是一种新颖的资源，通过收集并整合九种额外语言（巴西葡萄牙语、中文、荷兰语、法语、德语、意大利语、韩语、拉脱维亚语和瑞典语）的现有特定语言语料库，扩展了英语伯克利FrameNet语料库。通过在不同架构的模型上训练mFNC，我们在多语言和跨语言场景中持续优于现有的最先进框架语义解析器，强调了多语言训练数据的重要性。mFNC和我们训练的FSP模型可在https://github.com/beatrice-f/mFNC公开获取。

    arXiv:2608.23037v1 Announce Type: cross  Abstract: This paper introduces the Multilingual FrameNet Corpus (mFNC), a novel resource that extends the English Berkeley FrameNet corpus by collecting and harmonizing existing language-specific corpora across nine additional languages: Brazilian Portuguese, Chinese, Dutch, French, German, Italian, Korean, Latvian and Swedish. By training models that rely on different architectures on the mFNC, we consistently outperform existing state-of-the-art Frame Semantic Parsers in both multilingual and cross-lingual settings, underscoring the importance of multilingual training data. The mFNC and our trained FSP models are openly available at https://github.com/beatrice-f/mFNC.
    
[^51]: ST²U：基于受限知识边界控制的状态化测试时遗忘

    ST$^2$U: Stateful Test-Time Unlearning via Restricted Knowledge Boundary Control

    [https://arxiv.org/abs/2608.23034](https://arxiv.org/abs/2608.23034)

    ST²U通过轨迹级别的受限知识边界控制，解决了现有测试时遗忘方法因忽略自回归生成中状态重建而导致的受限知识重入问题。

    

    摘要：控制大型语言模型中的受限知识对于模型对齐和安全部署至关重要。测试时遗忘通过仅在推理过程中进行干预，避免了昂贵的重新训练和参数更新。然而，现有的激活编辑方法应用孤立的逐点修正，忽视了自回归生成如何从提示、缓存和生成前缀中持续重建隐藏状态。因此，后续状态可能在局部成功修正后重新回到受限知识区域，导致受限知识重新进入。在本工作中，我们提出了基于受限知识边界控制的状态化测试时遗忘（ST²U），该方法将测试时遗忘形式化为轨迹级别的边界控制。ST²U首先在低维可逆坐标中建模受限知识边界，同时保持正交的非目标组件不变。在推理过程中，ST²U监控...

    arXiv:2608.23034v1 Announce Type: cross  Abstract: Controlling restricted knowledge in large language models is essential for model alignment and safe deployment. Test-time unlearning avoids costly retraining and parameter updates by intervening only during inference. However, existing activation-editing methods apply isolated pointwise corrections, overlooking how autoregressive generation continually reconstructs hidden states from the prompt, cache, and generated prefix. Consequently, later states may return to restricted knowledge regions after a locally successful correction, causing restricted knowledge re-entry. In this work, we propose Stateful Test-Time Unlearning via restricted knowledge boundary control (ST$^2$U), which formulates test-time unlearning as trajectory-wide boundary control. ST$^2$U first models restricted knowledge boundaries in low-dimensional invertible coordinates while leaving orthogonal non-target components unchanged. During inference, ST$^2$U monitors ri
    
[^52]: 元主持人：通过元认知赋能多智能体辩论

    Meta-Moderator: Empowering Multi-Agent Debate with Meta-Cognition

    [https://arxiv.org/abs/2608.23029](https://arxiv.org/abs/2608.23029)

    元主持人通过可学习的元认知框架动态调节多智能体辩论，显式优化辩论效用并决定何时终止，从而显著提升推理性能。

    

    arXiv:2608.23029v1 公告类型：新 摘要：多智能体辩论可以通过引发多样化的假设和批评来提升大型语言模型的推理能力，但其性能往往受限于薄弱的调节机制。常见流程依赖固定预算、基于一致性的停止或未训练的评判者，导致冗余的审议和不可靠的证据聚合。我们将调节视为一个元认知过程，监控辩论效用、控制审议过程并裁定最终答案，并引入元主持人（Meta-Moderator），这是一个可学习的框架，能够动态调节辩论并决定何时最终确定答案。元主持人独立于辩论者进行训练，通过结果驱动的策略优化，使辩论调节成为一种显式能力，而非提示的偶然效果。在五个基准测试中，元主持人优于广泛使用的决策层，并能跨任务和系统配置迁移。进一步分析表明，它合理分配资源。

    arXiv:2608.23029v1 Announce Type: new  Abstract: Multi-agent debate can improve large language model reasoning by eliciting diverse hypotheses and critiques, yet its performance is often constrained by weak moderation. Common pipelines rely on fixed budgets, agreement-based stopping, or untrained judges, leading to redundant deliberation and unreliable evidence aggregation. We cast moderation as a meta-cognitive process, monitoring debate utility, controlling deliberation, and adjudicating a final answer, and introduce Meta-Moderator, a learnable framework that dynamically regulates debate and decides when to finalize an answer. Meta-Moderator is trained independently of the debaters via outcome-driven policy optimization, making debate regulation an explicit capability rather than an incidental effect of prompting. Across five benchmarks, Meta-Moderator outperforms widely used decision layers and transfers across tasks and system configurations. Further analyses show that it allocates
    
[^53]: 超越表面线索：解构多语言大语言模型中的社会文化信号

    Beyond Surface Cues: Disentangling Sociocultural Signals in Multilingual LLMs

    [https://arxiv.org/abs/2608.23026](https://arxiv.org/abs/2608.23026)

    该研究通过多智能体审计方法，在89,253个多语言大语言模型输出中分离社会文化信号，发现直接身份线索对英语和中文的偏见识别影响显著，而对法语影响较小，揭示了跨语言和任务的系统性差异。

    

    arXiv:2608.23026v1 公告类型：交叉 摘要：多语言大语言模型的输出在不同社会文化背景下可能有所差异。然而，文化根基的证据可能具有误导性：身份标签可能从显性或间接的文本线索中推断出来，而名字和措辞可能揭示源语言。将所有此类信号视为文化根基的证据可能会掩盖潜在偏见。我们提出了一项经过人工验证的多智能体审计，区分了三个问题：输出是否复制社会偏见，身份群体是否被不同地呈现，以及输出是否反映跨文化模式。该研究分析了12个大语言模型在英语、法语和中文中生成的89,253个输出，涵盖18种职业和三种任务条件。我们发现，偏见表现因语言和任务而异。去除直接身份线索显著降低了英语和中文中的身份标签预测，但在法语中影响较小。在所有语言-体裁组合中，

    arXiv:2608.23026v1 Announce Type: cross  Abstract: Multilingual LLM outputs can vary across sociocultural contexts. However, evidence of cultural grounding can be misleading: identity labels may be inferred from explicit or indirect textual cues, while names and wording can reveal the source language. Treating all these signals as evidence of cultural grounding may obscure potential biases. We present a human-validated, multi-agent audit that separates three questions: whether outputs reproduce social biases, whether identity groups are represented differently, and whether outputs reflect cross-cultural patterns. The study analyzes 89,253 outputs from 12 LLMs in English, French, and Chinese, spanning 18 occupations and three task conditions.   We find that bias representation varies systematically across languages and tasks. Removing direct identity cues sharply reduces identity-label prediction in English and Chinese, but has a much smaller effect in French. Across all language-genre 
    
[^54]: 大部分LLM路由差距源于任务类型

    Most of the LLM routing gap is task type

    [https://arxiv.org/abs/2608.23023](https://arxiv.org/abs/2608.23023)

    本文发现LLM路由器性能差距主要源于任务类型差异，而非模型选择策略，即使重复运行也存在评分不稳定性。

    

    arXiv:2608.23023v1 公告类型：新 摘要：LLM路由器选择哪个模型应回答每个查询。其吸引力在于不同模型在不同问题上会失败。无论哪个单一模型整体上最优，仍会在某些问题上出错，而池中的另一个模型能正确回答其中许多问题。每次都能正确选择是上限，而路由器是试图接近这一上限的方法。然而，近期研究报告称路由器并未接近这一上限。在五个基准测试上的21种路由方法中，设计截然不同的方法彼此相差无几，且都远低于该上限。学习型路由器往往无法击败简单地始终调用最强模型的做法。我们探究这些错失的问题有何共同点。我们让十四个模型回答全部294个问题，涵盖7种任务类型，涉及韩语、英语和印地语三种语言。我们两次运行了整个矩阵，未做任何更改，但4116个模型-问题对中有5.37%的评分结果仍不同。运行间变动。

    arXiv:2608.23023v1 Announce Type: new  Abstract: An LLM router picks which model should answer each query. The appeal is that models fail on different questions. Whatever single model is best overall still gets some wrong, and another model in the pool gets many of those right. Getting that choice right every time is the ceiling, and a router is an attempt to approach it.   However, recent work reports that routers do not get close. Across 21 routing methods on five benchmarks, sharply different designs land within a fraction of a point of each other, and all of them stay far below that ceiling. Learned routers often fail to beat simply always calling the strongest model.   We ask what those missed questions have in common. We set fourteen models to answer all 294 questions, with 7 task types across 3 languages: Korean, English and Hindi. We ran the whole matrix twice, changing nothing, but 5.37% of the 4,116 model-question pairs came out scored differently anyway. Run-to-run movement 
    
[^55]: 遗忘不仅仅是擦除：通过生成不平等实现时间解耦

    Unlearning Is Not Just Erasing: Temporal Decoupling via Generation Inequality

    [https://arxiv.org/abs/2608.23020](https://arxiv.org/abs/2608.23020)

    该论文提出ADU框架，通过时间解耦和生成不平等，将LLM遗忘从简单的令牌擦除转变为上下文注意力路径解耦，以在保持模型通用性的同时实现精确遗忘。

    

    arXiv:2608.23020v1 公告类型：新公告 摘要：大型语言模型（LLMs）需要有效的遗忘机制来解决隐私法规和安全问题。然而，在不损害通用功能的前提下实现精确遗忘仍然具有挑战性。现有的序列级和令牌级方法会惩罚目标输出，而不对其上下文相关的检索路径进行建模，这可能会破坏语言结构或抑制良性知识。我们提出了ADU，一个细粒度、基于训练的框架，将遗忘从令牌擦除转变为上下文注意力路径解耦。利用局部和全局注意力头之间的功能差异，ADU识别预规划位置，这些位置检索持久敏感锚点，并在原始模型下固定其候选路径。然后，它训练注意力投影适配器，以抑制这些路径上的注意力质量，同时保留局部注意力结构和保留集语言建模。训练后激活ex（此处截断）

    arXiv:2608.23020v1 Announce Type: new  Abstract: Large language models (LLMs) require effective unlearning to address privacy regulations and safety concerns. However, achieving precise forgetting without compromising general utility remains challenging. Existing sequence- and token-level methods penalize target outputs without modeling their context-dependent retrieval paths, which can disrupt linguistic structure or suppress benign knowledge. We present ADU, a fine-grained, training-based framework that shifts unlearning from token erasure to contextual attention-pathway decoupling. Exploiting the functional distinction between local and global attention heads, ADU identifies preplan positions that retrieve persistent sensitive anchors and fixes their candidate paths under the original model. It then trains attention-projection adapters to suppress attention mass along these paths while preserving local-attention structure and retain-set language modeling. Post-training activation ex
    
[^56]: PatchWrite：一行而非整节——面向AI草稿的编译门控、有效性保持编辑

    PatchWrite: One Line, Not One Section -- Compile-Gated, Validity-Preserving Editing for AI-Drafted Manuscripts

    [https://arxiv.org/abs/2608.23001](https://arxiv.org/abs/2608.23001)

    PatchWrite通过编译门控和证据锁机制，仅允许局部编辑而非整节重写，从而在修复手稿缺陷时严格保留无关内容，显著提升编辑的有效性和安全性。

    

    自动手稿流水线通常为了修复局部缺陷而重新生成整个章节，这会导致无关的指标和引用发生变化，即使最终生成的PDF仍能正常构建。PatchWrite则限制了候选编辑如何成为已提交的手稿状态：它重用有界的EDIT N M编辑和回滚，但通过致命日志检查加强了编译接受条件，并添加了证据锁，要求每个引用的键和实验数值令牌都必须由参考注册表或实验日志验证。未通过任一检查的候选将被拒绝，并保留之前的HEAD状态。在24篇手稿×8种故障的Oracle压力测试（768个任务，平均分为编译破坏和仅内容故障）中，整槽重写每次都会改变无关的“12层”行（0/192保留；数值Jaccard指数为0.6667），而PatchWrite在192/192的情况下保留了该行。移除编译门控后接受率降至0，而重新...

    arXiv:2608.23001v1 Announce Type: new  Abstract: Automated manuscript pipelines often regenerate an entire section to repair a local defect, allowing unrelated metrics and citations to change even when the resulting PDF still builds. PatchWrite instead constrains how candidate edits become committed manuscript states: it reuses bounded EDIT N M editing and rollback, but tightens compilation acceptance with fatal-log checks and adds evidence locks that require every cited key and experimental numeric token to be attested by a reference registry or experimental log. Candidates that fail either check are rejected and the previous HEAD is retained. On a 24-manuscript x 8-fault oracle stress test (768 jobs, evenly split between compile-breaking and content-only faults), whole-slot rewriting mutated an unrelated "12-layer" line in every case (0/192 preserved; numeric Jaccard 0.6667), whereas PatchWrite preserved it in 192/192 cases. Removing the compile gate reduced acceptance to 0, while re
    
[^57]: 人工智能辅导互动中大语言模型的教学行为

    LLM Pedagogical Behavior in AI Tutoring Interactions

    [https://arxiv.org/abs/2608.22993](https://arxiv.org/abs/2608.22993)

    该论文开发了一个五级脚手架量表，发现大学AI课程中学生的LLM辅导互动超过95%的回应集中于高直接帮助水平（解释或解决），且这种帮助水平与后续对话行为相关，但对考试成绩预测力有限。

    

    摘要：arXiv:2608.22993v1 公告类型：新  摘要：学生越来越多地使用大语言模型作为课程作业和问题解决的导师。关于学生在真实学习互动中使用大语言模型时，其提供的帮助水平知之甚少。这一点很重要，因为辅导回应在直接帮助学生完成任务的程度上有显著差异。我们将这一维度操作化为脚手架水平，并开发了一个五级量表，该量表经过人工注释验证，根据回应提供的直接帮助程度对其进行特征化。我们将该量表应用于一所大学AI课程中203名学生的14,637条大语言模型回应。回应绝大多数集中在高帮助水平，超过95%被分类为“解释”或“解决”。脚手架水平与学生的后续对话行为系统性相关，但对随后三次考试的表现几乎没有额外的预测信息。

    arXiv:2608.22993v1 Announce Type: new  Abstract: Students increasingly use LLMs as tutors for coursework and problem solving. Little is known about the level of assistance LLMs provide when students use them as tutors in authentic learning interactions. This matters because tutoring responses can differ substantially in how directly they help students complete a task. We operationalize this dimension as scaffolding level and develop a five-level scale, validated against human annotations, that characterizes responses according to the degree of direct assistance they provide. We apply the scale to 14,637 LLM responses from 203 students in a university AI course. Responses are overwhelmingly concentrated at high levels of assistance, with more than 95% classified as either Explaining or Solving. Scaffolding level is systematically associated with students' subsequent conversational behavior, but provides little additional predictive information about performance on three subsequent exams
    
[^58]: 激活引导控制什么？跨答案编码与输出敏感子空间的归因分析

    What Does Activation Steering Control? Attribution Across Answer Encodings and Output-Sensitive Subspaces

    [https://arxiv.org/abs/2608.22985](https://arxiv.org/abs/2608.22985)

    本文提出跨编码引导评估方法，发现激活引导的干预效果主要跟随提取索引而非语义标签，且该效应在深层网络中更为显著，揭示了输出敏感子空间的关键作用。

    

    arXiv:2608.22985v1 公告类型：新 摘要：激活引导通常在用于构建方向的答案编码下进行评估。报告的增益可能反映预期判断或与构建过程中看到的答案标识符的兼容性。我们引入了跨编码引导评估，该方法在冻结干预的同时，将答案重新编码为相同的保留项目。在NormBank上，当A/B/C标识符被重新分配后，对比激活加法（CAA）在新映射下，对于提取索引比对于语义标签诱导出更大的目标-源分数变化。我们称之为提取索引跟随。改变标识符词汇（A/B/C、X/Y/Z或1/2/3）和行顺序表明，该效应追踪提取索引而非行位置。在跨层匹配方向范数后，提取索引跟随主要出现在较深层次。一个包含方向平方范数15.4%的低秩输出敏感组件保留了9。

    arXiv:2608.22985v1 Announce Type: new  Abstract: Activation steering is often evaluated under the answer encoding used to construct the direction. A reported gain may reflect the intended judgment or compatibility with answer identifiers seen during construction. We introduce Cross-Encoding Steering Evaluation, which freezes an intervention while re-encoding answers to the same held-out items. On NormBank, after A/B/C identifiers are reassigned, contrastive activation addition (CAA) induces larger target-versus-source score changes for the extraction indices than for the semantic labels under the new mapping. We call this extraction-index following. Varying identifier vocabulary (A/B/C, X/Y/Z, or 1/2/3) and row order shows that the effect tracks extraction index rather than row position. After matching direction norms across layers, extraction-index following emerges mainly at later depths. A low-rank output-sensitive component containing 15.4% of the direction's squared norm retains 9
    
[^59]: 基于语义大语言模型代理的闭环贝叶斯分子逆向设计

    Closed-Loop Bayesian Molecular Inverse Design with Semantic LLM Surrogates

    [https://arxiv.org/abs/2608.22967](https://arxiv.org/abs/2608.22967)

    该论文提出了一种闭环贝叶斯分子逆向设计框架，通过将大型语言模型作为代理直接处理文本形式的任务指令和优化历史，以在有限预算下提高匹配目标性质的分子比例。

    

    arXiv:2608.22967v1 公告类型：新 摘要：实际的分子逆向设计很少是一次性生成问题；它通常采取闭环候选池富集的形式，在有限的预测预算下，目标是增加生成分子中匹配所需性质特征的比例。贝叶斯优化（BO）为此场景提供了自然框架，然而标准高斯过程代理通常在压缩的连续嵌入中操作，这丢弃了化学家自然用于决定下一步探索位置的子结构和参考相似性信号。我们提出了一种闭环框架，其中代理而非生成器被视为设计选择的核心，并通过一个冻结的大型语言模型实例化该框架，该模型直接以文本形式推理任务指令、SMILES级优化历史和预测反馈。在每次迭代中，代理返回...

    arXiv:2608.22967v1 Announce Type: new  Abstract: Practical molecular inverse design is rarely a one-shot generation problem; it often takes the form of closed-loop candidate-pool enrichment, where under a limited oracle budget the goal is to \emph{increase the fraction of generated molecules that match a desired property profile}. Bayesian optimization (BO) offers a natural framework for this setting, yet standard Gaussian-process surrogates typically operate in compressed continuous embeddings, which discard the substructural and reference-similarity signals that chemists naturally use to decide where to look next. We propose \textbf{\method}, a closed-loop framework in which the surrogate, rather than the generator, is treated as the locus of design choice, and instantiate it with a frozen large language model that reasons directly over the task instruction, SMILES-level optimization history, and oracle feedback in their native textual form. At each iteration, the surrogate returns a
    
[^60]: 深陷文本债务：面向MLLM智能体的视觉证据保留上下文剪枝

    Buried in Textual Debt: Context Pruning with Visual Evidence Preservation for MLLM Agents

    [https://arxiv.org/abs/2608.22963](https://arxiv.org/abs/2608.22963)

    本文提出SPARE框架，通过KL散度引导的上下文剪枝，在保留视觉证据的同时移除冗余推理文本，以解决MLLM智能体长轨迹中的“文本债务”问题。

    

    arXiv:2608.22963v1 公告类型：新论文 摘要：多模态大语言模型（MLLMs）正越来越多地被部署为多步智能体，其中显式推理支持任务分解和工具协调，但也会积累自生成的文本。在长轨迹中，这些文本可能主导上下文并抑制视觉证据，造成“文本债务”。我们观察到，一旦任务相关的视觉证据被锚定，推理就变得冗余，而当锚定仍不确定时，过时的假设可能会误导后续推理。因此，剪枝必须移除冗余文本，同时不丢弃视觉证据。我们提出SPARE，一种基于库尔贝克-莱布勒（KL）散度引导的框架，用于剪枝多模态工具使用智能体中积累的推理内容。SPARE使用紧凑的任务状态摘要作为特权诊断上下文。对于每个候选段，它在原始上下文和摘要条件下的上下文中重放相同模型。然后，从策略内自蒸馏（OPSD）的反向KL散度指导剪枝决策。

    arXiv:2608.22963v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) are increasingly deployed as multi-step agents, where explicit reasoning supports task decomposition and tool coordination but also accumulates self-generated text. Over long trajectories, this text can dominate the context and suppress visual evidence, creating textual debt. We observe that reasoning becomes redundant once task-relevant visual evidence is grounded, while stale hypotheses can misguide later inference when grounding remains uncertain. Pruning must therefore remove redundant text without discarding visual evidence. We propose SPARE, a Kullback--Leibler (KL)-guided framework for pruning accumulated reasoning in multimodal tool-use agents. SPARE uses a compact task-state summary as privileged diagnostic context. For each candidate segment, it replays the same model under the original and summary-conditioned contexts. Reverse-KL divergence from on-policy self-distillation (OPSD) then t
    
[^61]: 控制错觉：为何裸分类器反演在概念瓶颈文本生成中悄然失败

    The Illusion of Control: Why Bare Classifier Inversion Silently Fails in Concept-Bottleneck Text Generation

    [https://arxiv.org/abs/2608.22956](https://arxiv.org/abs/2608.22956)

    本文发现，在概念瓶颈文本生成中，裸分类器反演及其正则化变体均不如简单的事后先验方法，揭示了反演方法在组合泛化中的固有局限。

    

    arXiv:2608.22956v1 公告类型：新 摘要：概念瓶颈可控生成通过一个低维概念代码来路由多属性控制，在部署时，必须从目标属性配置中合成该代码。我们在多轴组合泛化条件下研究概念瓶颈文本生成中的这一问题，比较了三种获取推理时代码的方法：针对编码器头的分类器反演、参考文本编码以及事后标签条件先验。由于概念代码不直接包含语言模型流畅性项，因此正则化反演必须将代码约束在编码器的训练分布内。为此，我们测试了裸反演及三种正则化变体：标签无关和标签条件的马氏距离惩罚，以及条件归一化流密度基线。我们测试的所有反演变体在三个基准上均逊于在同一检查点上拟合到每组合编码器均值的简单事后先验。

    arXiv:2608.22956v1 Announce Type: new  Abstract: Concept-bottleneck controllable generation routes multi-attribute control through a low-dimensional concept code that, at deployment, must be synthesised from a target attribute configuration. We study this problem in concept-bottleneck text generation under multi-axis compositional generalisation, comparing three ways to obtain the inference-time code: classifier inversion against the encoder heads, reference-text encoding, and a post-hoc label-conditioned prior. Since a concept code admits no direct LM-fluency term, regularising inversion must instead constrain the code toward the encoder's training distribution. We therefore test bare inversion and three regularised variants: label-agnostic and label-conditioned Mahalanobis penalties, and a conditional normalising-flow density baseline. Every inversion variant we test underperforms a simple post-hoc prior fitted to per-combination encoder means on the same checkpoints, across three ba
    
[^62]: 什么证明你错了：在可证伪研究构思上对语言模型进行基准测试

    What Proves You Wrong: Benchmarking Language Models on Falsifiable Research Ideation

    [https://arxiv.org/abs/2608.22948](https://arxiv.org/abs/2608.22948)

    本文提出了Lit2Test基准，通过要求研究提议预先指定可证伪结果，使提议质量可判定，并基于200个真实论文邻域和1200次盲评比较了四个前沿语言模型。

    

    arXiv:2608.22948v1 公告类型：交叉 摘要：大型语言模型越来越多地被用于提出研究想法，然而，评判此类想法的现行方式缺乏共享的决策规则：自由形式的评判受风格和位置影响，而针对后续论文的评分则奖励对单一实现轨迹的恢复。我们引入了一个基准测试，它将提议从文献带到测试：Lit2Test基准测试围绕一个以证伪结果为核心的六字段契约组织，使得每个提议都预先承诺了会证明其错误的观察结果，从而使提议的质量首先变得可判定，而非仅仅可争论。该基准测试基于200个真实论文邻域前瞻性地构建，从四个前沿模型中引出提议，并通过1200次成对比较（以两种呈现顺序盲评）进行比较。该协议通过诊断控制和有界人工校准来审计其自身的可靠性，并由三名标注者证实结论。

    arXiv:2608.22948v1 Announce Type: cross  Abstract: Large language models are increasingly used to propose research ideas, yet the prevailing ways of judging such ideas supply no shared decision rule: free-form judging sways with style and position, and scoring against a later paper rewards recovery of one realized trajectory. We introduce a benchmark that carries a proposal from Literature to Test: the Lit2Test benchmark centers on a six-field contract organized around a falsifying outcome, so that every proposal precommits the observation that would prove it wrong, making its quality decidable in the first place rather than merely arguable. Built prospectively from 200 real-paper neighborhoods, Lit2Test elicits proposals from four frontier models and compares them through 1,200 pairwise comparisons judged blind in both presentation orders. The protocol audits its own reliability through diagnostic controls and bounded human calibration, with three annotators corroborating the conclusi
    
[^63]: HelaBERT：通过双池化分类头增强僧伽罗语语言理解

    HelaBERT: Enhancing Sinhala Language Understanding with Dual Pooling Classification Head

    [https://arxiv.org/abs/2608.22922](https://arxiv.org/abs/2608.22922)

    本文提出了HelaBERT，一种针对僧伽罗语预训练的BERT模型家族，并引入双池化分类头，在多个分类任务中显著提升了情感分析性能。

    

    我们提出了HelaBERT，这是一个基于BERT的掩码语言模型家族，从头开始在约10亿个僧伽罗语文本标记上进行了预训练，这些文本来源于MADLAD-400、CulturaX以及包含新闻文章、僧伽罗语维基百科和网络爬取数据的自定义语料库。HelaBERT-Small（约2330万参数，6层）和HelaBERT-Large（约1.1亿参数，12层）均使用了针对僧伽罗语黏着形态和复杂文字定制的SentencePiece Unigram分词器（词汇量32,000）。我们在四个下游僧伽罗语文本分类任务上评估了这两个模型：新闻类别分类、新闻来源分类、情感分析和写作风格分类，使用5次独立种子运行，采用分层80/20训练/测试划分。我们还提出了一种双池化分类头，并在所有四个任务中进行了系统评估，发现其在情感分析上有一致的改进，在新闻分类上有适度提升。

    arXiv:2608.22922v1 Announce Type: new  Abstract: We present HelaBERT, a family of two BERT-based masked language models pre-trained from scratch on approximately 1 billion tokens of Sinhala text sourced from MADLAD-400, CulturaX, and a custom corpus comprising news articles, Sinhala Wikipedia, and web crawl data. HelaBERT-Small (~23.3M parameters, 6 layers) and HelaBERT-Large (~110M parameters, 12 layers) both use a SentencePiece Unigram tokenizer (vocabulary size 32,000) tailored to Sinhala's agglutinative morphology and complex script. We evaluate both models on four downstream Sinhala text classification tasks: news category classification, news source classification, sentiment analysis, and writing style classification, using 5 independent seed runs with stratified 80/20 train/test splits. We additionally propose a dual pooling classification head and evaluate it systematically across all four tasks, finding consistent improvements on sentiment analysis and a moderate gain on news 
    
[^64]: TSWAP：一个多语言检索增强型泰式健康顾问

    TSWAP: A Multilingual Retrieval-Augmented Thai Wellness Advisor

    [https://arxiv.org/abs/2608.22917](https://arxiv.org/abs/2608.22917)

    TSWAP通过检索增强生成和混合检索技术，在泰式传统医学知识库上实现了八语言零样本健康顾问，并发布了首个泰式医学检索基准和QA日志，展示了各接地组件的贡献。

    

    arXiv:2608.22917v1 公告类型：新 摘要：我们介绍了TSWAP，一个已部署的八语言对话式健康顾问，它通过检索增强生成，基于泰式传统医学和认证健康服务提供者的经过验证的知识库。一个未经修改的开源权重LLM（Qwen3.6-35B-A3B，运行于vLLM）通过混合密集-稀疏检索器配合交叉编码器重排序，基于约30.6K块的泰语索引进行接地；一个首轮查询分类器强制基于工具检索进行实体查找；一个基于规则的安全层强制执行医疗范围和泰语紧急路由；所有八种语言均以零样本方式提供，采用先翻译后检索策略。我们发布了首个泰式传统医学/健康检索基准（50个问题，带金标准文档ID；Recall@5 = 0.88），生产QA日志（259个案例中测试-重测通过率为91.1%），以及一个71问题的前沿无检索探针，显示每个接地支柱的贡献：在没有安全提示的情况下，后端模型家族产生了...

    arXiv:2608.22917v1 Announce Type: new  Abstract: We present TSWAP, a deployed eight-language conversational wellness advisor grounded, via retrieval-augmented generation, in a verified knowledge base of Thai traditional medicine and certified wellness providers. An unmodified open-weight LLM (Qwen3.6-35B-A3B on vLLM) is grounded on a ~30.6K-chunk Thai index by a hybrid dense-sparse retriever with cross-encoder reranking; a first-turn query classifier forces tool-based retrieval for entity lookups; a rule-based safety layer enforces medical scope and Thai emergency routing; and all eight languages are served zero-shot with translate-then-retrieve. We release the first Thai traditional-medicine/wellness retrieval benchmark (50 questions with gold document IDs; Recall@5 = 0.88), production QA logs (91.1% test-retest pass over 259 cases), and a 71-question frontier no-retrieval probe showing what each grounding pillar contributes: without the safety prompt the backend model family produced
    
[^65]: 知道不等于说出：视觉-语言模型中的空间编码何时到达答案？

    Knowing Isn't Always Saying: When Do Spatial Encodings Reach Answers in Vision-Language Models?

    [https://arxiv.org/abs/2608.22916](https://arxiv.org/abs/2608.22916)

    该论文通过方向修补干预揭示了视觉-语言模型中空间编码对答案的影响仅在深层出现，并受提示格式和思维链影响，形成不同传输模式。

    

    arXiv:2608.22916v1 公告类型：新 摘要：已知视觉-语言模型在其隐藏状态中编码空间信息，但在回答问题时往往无法使用这些信息。然而，目前尚不清楚这些编码信息何时何地到达答案。我们通过方向修补（一种基于类条件因果干预，应用于层、标记位置和提示格式）来解决这一问题。使用根据先前编码证据构建的空间身份方向，我们发现对答案逻辑的影响仅在中等至深层深度出现。文本思维链抑制了大多数模型中即时对象词argmax级传输，而视觉基础提示保持其开放。正目标逻辑增益可能保持在argmax阈值以下，且传输可能在最终前缀标记或更深层的答案步骤中重新出现。在我们研究的十个视觉-语言模型中，这些局部效应形成了描述性传输模式。补充实验刻画了这些效应如何影响其他方面。

    arXiv:2608.22916v1 Announce Type: new  Abstract: Vision-language models are known to encode spatial information in their hidden states, yet often fail to use it when answering. However, it remains unclear when and where this encoded information reaches the answer. We address this with direction patching, a class-conditioned causal intervention applied across layers, token positions, and prompt formats. Using spatial-ID directions constructed following prior encoding evidence, we find that causal influence on answer logits emerges only at mid-to-deep depths. Text chain-of-thought suppresses immediate object-word argmax-level transport in most models, while visually grounded prompts keep it open. Positive target-logit gain can remain below the argmax threshold, and transport can re-emerge at the final prefix token or at the answer step in deeper layers. Across the ten VLMs we study, these local effects form descriptive transport patterns. Complementary experiments characterize how these 
    
[^66]: 受限最佳N推理时扩展中的安全黑客攻击

    Safety Hacking in Constrained Best-of-$N$ Inference-time Scaling

    [https://arxiv.org/abs/2608.22915](https://arxiv.org/abs/2608.22915)

    本文发现推理时管道中不完美的安全代理与奖励最大化组合会导致“安全黑客攻击”，即使代理误差极小，当N增大时，不安全输出若尾部更重，攻击概率趋近于1。

    

    arXiv:2608.22915v1 公告类型：交叉 摘要：推理时管道通常采样多个输出，用学习到的安全模型进行过滤，并返回具有最高学习奖励的代理可行输出。我们表明，这种组合会产生两阶段失败：不完美的安全代理首先用不安全的输出污染可行集，然后奖励最大化可能放大这种残留污染。我们将“安全黑客攻击”定义为选择通过学习约束但违反真实安全标准的输出。对于受限最佳N采样，我们推导了由代理可行集中安全和不安全输出的联合上奖励尾部控制的有限N界限。如果不安全但可行的输出具有更重的尾部，那么随着N增长，安全黑客攻击变得渐近确定，即使假阳性质量和平均安全及奖励代理误差任意小。我们还表明，在有限的χ²散度内的策略下，存在某些界限。

    arXiv:2608.22915v1 Announce Type: cross  Abstract: Inference-time pipelines often sample multiple outputs, filter them with a learned safety model, and return the proxy-feasible output with the highest learned reward. We show that this composition creates a two-stage failure: an imperfect safety proxy first contaminates the feasible set with unsafe outputs, and reward maximization can then amplify this residual contamination. We define \emph{safety hacking} as selecting an output that passes the learned constraint but violates the true safety criterion. For constrained Best-of-$N$ sampling, we derive finite-$N$ bounds governed by the joint upper reward tails of safe and unsafe outputs within the proxy-feasible set. If unsafe-but-feasible outputs have the heavier tail, safety hacking becomes asymptotically certain as $N$ grows, even when false-positive mass and average safety- and reward-proxy errors are arbitrarily small. We also show that policies within a bounded $\chi^2$ divergence 
    
[^67]: 探索道克同调用于句子相似性

    Exploring Dowker Homology for Sentence Similarity

    [https://arxiv.org/abs/2608.22909](https://arxiv.org/abs/2608.22909)

    本文探索道克同调作为拓扑工具，通过将句子对标记嵌入视为点云，证明其能有效捕捉句子相似性信息，并可用于视觉检查，但单数值摘要未超越传统度量。

    

    arXiv:2608.22909v1 公告类型：新 摘要：道克同调是一种拓扑工具，可用于分析共享空间中两个点云的相对位置。我们通过将构成句子对的标记嵌入视为变压器模型潜在空间中的一对点云，来研究道克同调是否能捕捉句子相似性信息，并使用了针对句子相似性进行微调和未微调的模型。我们发现，道克同调能够捕捉句子相似性信息（通过将道克同调特征回归到真实相似性分数来衡量），并且可用于相似性数据和模型的视觉检查。为使道克同调易于应用，我们从中推导出单数值摘要，期望能直接捕捉句子相似性。这些摘要效果尚可，但未超越基于已确立标准的传统句子相似性度量。

    arXiv:2608.22909v1 Announce Type: new  Abstract: Dowker homology is a topological tool that may be used to analyze the relative position of two point clouds living in a common space. We investigate whether Dowker homology captures sentence similarity information by treating the embeddings of the tokens that constitute a sentence pair as a pair of point clouds in the latent space of a transformer model, using both models that have and have not been fine-tuned for sentence similarity. We find that Dowker homology captures sentence similarity information, as measured by regressing Dowker homology features onto ground-truth similarity scores, and that it can be used for visual inspection of similarity data and models. In an attempt to make Dowker homology readily applicable, we derive from it single-number summaries that we expect to capture sentence similarity directly. These turn out to work reasonably well, but without outperforming standard sentence similarity measures based on establi
    
[^68]: 口语语言模型是否像阅读文本一样理解语音？弥合语音与文本之间的结构差距

    Do Spoken Language Models Hear Speech as They Read Text? Bridging Structural Gaps Between Speech and Text

    [https://arxiv.org/abs/2608.22908](https://arxiv.org/abs/2608.22908)

    本文提出一种解耦长度不匹配与语义对齐的简单框架，以增强口语语言模型中语音与文本表示的结构对齐，从而提升指令遵循和泛化能力。

    

    口语语言模型（SLMs）直接从语音生成文本响应，为级联系统提供了一种替代方案。尽管近期有所进展，但与基于文本的语言模型相比，现有SLMs在指令遵循行为上仍较弱，且跨多样化任务的泛化能力有限。我们的分析表明，尽管下游性能强劲，当前SLMs中的语音和文本表示仍弱对齐，这表明连续、时变语音与离散文本之间的结构差异尚未得到充分解决。为此，我们提出了一个简单框架，将长度不匹配与语义对齐解耦，并鼓励语音与文本表示之间更紧密的对应关系。在多个基准上的实验表明，与强基线相比，该框架取得了具有竞争力的性能，强调了显式解决语音与文本之间结构差异的重要性。

    arXiv:2608.22908v1 Announce Type: cross  Abstract: Spoken Language Models (SLMs) generate textual responses directly from speech, offering an alternative to cascaded systems. Despite recent advances, existing SLMs still exhibit weaker instruction-following behavior and limited generalization across diverse tasks compared to text-based language models. Our analysis shows that speech and text representations in current SLMs remain weakly aligned despite strong downstream performance, indicating that structural differences between continuous, temporally varying speech and discrete text remain insufficiently addressed. To address this, we propose a simple framework that decouples length mismatch from semantic alignment and encourages closer correspondence between speech and text representations. Experiments across multiple benchmarks demonstrate competitive performance against strong baselines, underscoring the importance of explicitly addressing structural differences between speech and t
    
[^69]: SelFusion：扩散语言模型的自我蒸馏

    SelFusion: Self-distillation for Diffusion Language Models

    [https://arxiv.org/abs/2608.22898](https://arxiv.org/abs/2608.22898)

    SelFusion通过双向自我蒸馏和动态掩码策略，无需外部教师模型即可提升扩散语言模型的生成质量。

    

    arXiv:2608.22898v1 公告类型：新公告 摘要：扩散语言模型（DLMs）缓解了自回归（AR）大型语言模型（LLMs）固有的延迟瓶颈，但其生成质量的下降限制了实际应用。尽管知识蒸馏（KD）可能是提升性能的一个有前景的方向，但我们通过实验发现，简单应用传统KD仅带来边际收益，甚至可能降低生成质量。基于这些观察，我们提出了一种新颖的DLMs自我蒸馏框架，即SelFusion。为了实现无需外部教师模型的有效KD，SelFusion以不同的掩码级别执行两次前向传播，定义具有较大掩码概率的困难模式和具有较小掩码概率的简单模式。然而，简单模式并不总是比困难模式更准确，并且可能对错误令牌过于自信。因此，我们引入了两种模式之间的双向KD，可以动态地调整学习过程，以提升生成质量。

    arXiv:2608.22898v1 Announce Type: new  Abstract: Diffusion language models (DLMs) alleviate the inherent latency bottleneck of autoregressive (AR) large language models (LLMs), but their degraded generation quality limits practical applicability. Although knowledge distillation (KD) can be a promising direction for improving performance, we empirically find that naively applying conventional KD yields only marginal gains, or even degrades generation quality. Based on these observations, we propose a novel self-distillation framework for DLMs, namely SelFusion. To enable effective KD without an external teacher model, SelFusion performs two forward passes with different masking levels, defining the hard mode with a larger masking probability and the easy mode with a smaller masking probability. However, the easy mode is not always more accurate than the hard mode and can be overconfident on incorrect tokens. Thus, we introduce bidirectional KD between the two modes, which can dynamicall
    
[^70]: AraDetox：一个多方言阿拉伯语去毒化数据集

    AraDetox: A Multi-Dialect Arabic Detoxification Dataset

    [https://arxiv.org/abs/2608.22894](https://arxiv.org/abs/2608.22894)

    该论文提出了AraDetox，一个覆盖四种阿拉伯方言的大规模去毒化数据集，并证明去毒化是一种在保留语义的同时进行词汇和结构重构的改写任务。

    

    摘要：arXiv:2608.22894v1 公告类型：交叉 摘要：阿拉伯语有害语言检测已受到相当多的关注，但阿拉伯语文本去毒化仍未得到充分探索。我们引入了AraDetox，一个多方言阿拉伯语去毒化数据集，包含10,500条有害社交媒体帖子和84,000条去毒化改写内容，这些改写内容使用GPT-5和Gemini 2.5 Flash生成，覆盖现代标准阿拉伯语、海湾方言、黎凡特方言和埃及方言。生成的输出通过人工评估以及词汇变化、语义保留、情感和方言风格的自动分析进行了评估。结果表明，去毒化主要是一项保留意义的改写任务：大量的词汇和结构重构伴随着持续的高语义相似性。人工评估确认成功移除了有害语言，同时基本保留了原始含义。方言分析进一步表明，生成的变体在风格上与参考方言表现出可测量的对齐性。

    arXiv:2608.22894v1 Announce Type: cross  Abstract: Arabic harmful-language detection has received considerable attention, yet Arabic text detoxification remains underexplored. We introduce AraDetox, a multi-dialect Arabic detoxification dataset comprising 10,500 harmful social-media posts and 84,000 detoxified rewrites generated using GPT-5 and Gemini 2.5 Flash across Modern Standard Arabic, Gulf, Levantine, and Egyptian Arabic. The generated outputs were assessed through human evaluation and automatic analyses of lexical change, semantic preservation, sentiment, and dialectal style. Results show that detoxification is primarily a meaning-preserving rewriting task: substantial lexical and structural reformulation is accompanied by consistently high semantic similarity. Human evaluation confirms successful harmful-language removal while largely preserving the original meaning. Dialectal analyses further indicate that the generated variants exhibit measurable stylistic alignment with ref
    
[^71]: 大型语言模型决策中的代理依赖未针对预测证据进行校准

    Proxy reliance in large language model decisions is uncalibrated to predictive evidence

    [https://arxiv.org/abs/2608.22887](https://arxiv.org/abs/2608.22887)

    本文通过精确计算临床任务中的因果代理效应，发现LLM对代理的依赖未与预测证据校准，表现为过度依赖、合理依赖或不足依赖，且社会标签抑制效果脆弱。

    

    摘要：大型语言模型（LLMs）正进入分诊和贷款等决策领域，在这些领域中，必须将任务相关的推理与不可允许的代理使用区分开来。当前的审计方法通过检查人口统计学特征变化时决策是否改变来评估。但与受保护群体相关的属性带有预测价值，因此决策变化可能是歧视，也可能是合理推断。我们在一个具有已知真实结果的临床排序任务中测量了四种LLM的因果代理效应，其中证据所支持的依赖程度可以精确计算并作为参考。一个审计信号产生三种判定：过度依赖、合理依赖和不足依赖。在中性标签下，每个模型都依赖无信息的代理。有信息的代理则引发所有三种情况。社会领域名称将依赖程度推低，在其中一个模型中低于参考值。两项发现解释了这一点：依赖程度严重低于证据所指示的水平，且社会标签抑制是脆弱的，因为上下文示例会破坏这种抑制。

    arXiv:2608.22887v1 Announce Type: new  Abstract: Large language models (LLMs) are entering decisions in triage and lending, where task-relevant inference must be distinguished from impermissible proxy use. Current audits ask whether decisions change when demographics change. But attributes correlated with a protected group carry predictive value, so a changed decision can be discrimination or sound inference. We measure causal proxy effects in four LLMs on a clinical-ranking task with known ground truth, where the reliance the evidence warrants can be computed exactly and used as the reference. One audit signal yields three verdicts: over-reliance, warranted and under-reliance. Under neutral labels every model relies on proxies with no information. Informative proxies draw all three. Social field names push reliance down, below the reference in one model. Two findings explain this. Reliance severely undertracks the evidence, and social-label suppression is fragile, since in-context exa
    
[^72]: 更好的检索，更差的鲁棒性：多跳RAG如何放大上游ASR错误

    Better Retrieval, Worse Robustness:How Multi-hop RAG Amplifies Upstream ASR Errors

    [https://arxiv.org/abs/2608.22872](https://arxiv.org/abs/2608.22872)

    这项研究发现，多跳RAG中的实体图链接和迭代重述扩展会放大上游ASR错误，相比朴素密集检索，其性能差距在口音变化下增加36-67%，且查询实体损坏是主要失败原因。

    

    arXiv:2608.22872v1 公告类型：新 摘要：基于语音的应用在进入任何检索模块之前，会先通过自动语音识别（ASR）处理口语查询，因此ASR错误作为固定的上游约束进入流水线。我们实证测试了标准检索增强生成（RAG）的两种扩展——实体图链接和迭代重述——是吸收还是放大了这些错误。使用通过神经TTS合成的四种英语口音，我们在三个多跳问答基准（HotpotQA、2WikiMultiHopQA和MuSiQue）上评估了四种RAG配置，并与干净文本的对照进行了比较。尽管结构更丰富的配置在ASR输入下通常保持较高的绝对F1分数，但两种扩展都放大了错误：在所有三个基准上，从干净文本到最高词错误率口音的F1差距，在它们的组合下比在朴素密集检索下大36-67%。主要的失败模式是一个或多个查询实体的损坏，占性能下降的87-96%。

    arXiv:2608.22872v1 Announce Type: new  Abstract: Speech-based applications pass spoken queries through automatic speech recognition (ASR) before any retrieval module, so ASR errors enter the pipeline as a fixed upstream constraint. We empirically test whether two extensions to standard retrieval-augmented generation (RAG), entity-graph linking and iterative reformulation, absorb or amplify these errors. Using four English accents synthesized through neural TTS, we evaluate four RAG configurations on three multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA and MuSiQue) against a clean-text oracle. Although the structurally richer configurations generally retain higher absolute F1 under ASR input, both extensions amplify the error: the F1 gap from clean text to the highest-WER accent is 36-67% larger under their combination than under naive dense retrieval, on all three benchmarks. The dominant failure mode is corruption of one or more query entities, accounting for 87-96% of degradation
    
[^73]: WARP：面向群体意见的Wasserstein对齐RAG

    WARP: Wasserstein-Aligned RAG for Population Opinions

    [https://arxiv.org/abs/2608.22859](https://arxiv.org/abs/2608.22859)

    WARP通过Wasserstein距离校准RAG检索结果，以恢复被标准检索忽视的少数意见，从而更准确地反映群体意见分布。

    

    arXiv:2608.22859v1 公告类型：交叉 摘要：RAG系统越来越多地被用于总结大型文档集合的内容。用户问“人们对X有何看法？”并得到一个读起来像共识的答案。但标准的top-k检索按查询相似度对文档排序，而不是按它们对群体的代表性，因此少数观点悄然消失。现有的修复方法不足。像MMR和DPP这样的多样性重排器分散检索文档，但没有目标分布可瞄准。基于KL或JS散度的校准方法确实瞄准一个，但将意见箱视为无序的：混淆强正面与强负面的代价不比相邻箱的失误多。我们引入WARP，一个后检索算法家族，将检索到的证据校准到群体的意见分布。WARP首先恢复可能被余弦排序埋没的代表不足的意见，然后使用Wasserstein-1距离选择情感与目标分布对齐的文档。

    arXiv:2608.22859v1 Announce Type: cross  Abstract: RAG systems are increasingly used to summarize what large collections of documents say. A user asks "What do people think about X?" and receives an answer that reads as consensus. But standard top-k retrieval ranks documents by query similarity, not by how faithfully they represent the population, so minority views quietly disappear. Existing fixes fall short. Diversity re-rankers like MMR and DPP spread retrieved documents apart, but with no target distribution to aim for. Calibration methods based on KL or JS divergence do target one, yet treat opinion bins as unordered: confusing strong positive with strong negative costs no more than an adjacent-bin miss.   We introduce WARP, a family of post-retrieval algorithms that calibrate retrieved evidence to the population's opinion distribution. WARP first recovers underrepresented opinions that cosine ranking may bury, then uses Wasserstein-1 distance to select documents whose sentiment-i
    
[^74]: SAVER：针对视觉语言模型变化推理中错误恢复的选择性言语证据审计

    SAVER: Selective Auditing of Verbal Evidence for Error Recovery in VLM Change Reasoning

    [https://arxiv.org/abs/2608.22857](https://arxiv.org/abs/2608.22857)

    SAVER通过解析VLM输出中的言语证据并仅在证据缺失时触发重新提示，显著提升了视觉变化推理的准确性，最高提升达+25.8%。

    

    视觉语言模型（VLMs）在视觉变化推理中经常失败，即使其视觉编码器包含足够的信息。我们观察到，正确的VLM输出往往包含明确的言语证据（对象名称、颜色、空间位置）来支持所声称的变化，而错误的输出通常缺乏此类证据。我们提出SAVER（选择性审计言语证据以进行错误恢复），一种轻量级、基于规则的方法，该方法解析VLM响应中的证据，并仅在证据缺失或不一致时触发结构化重新提示。在三个变化检测基准和四个VLM上，SAVER显著提高了因模型未能表达其所见内容（表达失败）而导致错误的任务的准确性，在CLEVR-Change上提升高达+25.8%。证据模式也可以由LLM在单次调用中生成，与CLEVR-Change上的手动调优门控相匹配。消融实验...

    arXiv:2608.22857v1 Announce Type: new  Abstract: Vision-language models (VLMs) frequently fail at visual change reasoning, even when their vision encoders contain sufficient information. We observe that correct VLM outputs tend to contain explicit verbal evidence (object names, colors, spatial locations) that supports the claimed change, while incorrect outputs often lack such evidence. We propose SAVER (Selective Auditing of Verbal Evidence for Error Recovery), a lightweight, rule-based method that parses VLM responses for this evidence and triggers structured reprompting only when evidence is missing or inconsistent. Across three change detection benchmarks and four VLMs, SAVER significantly improves accuracy on tasks where errors stem from the model failing to articulate what it saw (expression failures), with gains up to +25.8% on CLEVR-Change. The evidence patterns can also be generated by an LLM in a single call, matching the hand-tuned gate on CLEVR-Change. Ablation experiments 
    
[^75]: 同一代理，不同答案：检索增强问答中语料库引起的答案波动的一项重复性审计

    Same Agent, Different Answers: A Repeat-Aware Audit of Corpus-Induced Answer Churn in Retrieval-Augmented QA

    [https://arxiv.org/abs/2608.22856](https://arxiv.org/abs/2608.22856)

    本文提出了一种快照兼容性审计方法，揭示检索增强问答系统在索引扩展后存在被总体准确性掩盖的答案波动，即使模型和提示等设置不变，超额波动仍显著。

    

    arXiv:2608.22856v1 公告类型：交叉 摘要：检索增强问答系统在索引扩展后，即使其请求的模型标识、提示、检索策略、证据深度、渲染方式和暴露的生成控制保持不变，也可能返回不同的答案。当增益和损失相互抵消时，总体准确性可能掩盖这些变化，而普通的生成变异性使得一次性比较夸大更新效果。我们将这种隐藏现象称为“准确性盲区答案波动”，并引入“快照兼容性审计”，通过从跨快照不一致性中减去同快照重复不一致性来估计超额答案波动。我们通过将一个冻结的FineWeb前缀从一个分片扩展到七个分片来实例化该方法。在一项预注册的400问题自然问题研究中，归一化精确和盲化语义的超额波动分别为6.44和10.25个百分点，而精确匹配准确性仅变化了$-1.50$个百分点。事后分析发现重复稳定性...

    arXiv:2608.22856v1 Announce Type: cross  Abstract: A retrieval-augmented QA system can return different answers after an index expansion even when its requested model identifier, prompt, retrieval policy, evidence depth, rendering, and exposed generation controls are held fixed. Aggregate accuracy may hide these changes when gains and losses cancel, while ordinary generation variability makes one-shot comparisons overstate update effects. We call the hidden phenomenon accuracy-blind answer churn and introduce the \emph{Snapshot Compatibility Audit}, which estimates excess answer churn by subtracting same-snapshot repeat disagreement from cross-snapshot disagreement. We instantiate it by expanding one frozen FineWeb prefix from one to seven shards. In a preregistered 400-question Natural Questions study, normalized-exact and blinded-semantic excess churn are 6.44 and 10.25 percentage points while exact-match accuracy changes by only $-1.50$ points. A post-hoc analysis finds repeat-stabl
    
[^76]: 你的AI，如同一个旋钮：通过单个神经元控制大语言模型中的投资偏见

    Your AI, On a Dial: Controlling Investment Bias in LLMs with a Single Neuron

    [https://arxiv.org/abs/2608.22852](https://arxiv.org/abs/2608.22852)

    本文提出一种通过干预单个神经元来连续调节大语言模型整体投资倾向的“投资偏见旋钮”，无需修改提示或参数，即可单调改变投资决策和理由。

    

    大型语言模型（LLMs）越来越多地用于投资决策，然而先前的研究表明，它们表现出系统性、模型特定的投资偏好。我们研究模型的整体投资立场是否可以校准到指定的方向和强度。我们引入了一个投资偏见旋钮，这是一种对单个神经元的推理时干预，能够连续调整模型层面的决策先验——其整体倾向于买入或卖出的趋势——而无需针对特定公司或投资属性。通过使用匹配的正反证据，我们评估了五个开放权重模型，并发现该旋钮在不修改提示或模型参数的情况下，产生投资立场的单调变化。在响应层面，该旋钮在相同输入下改变了投资决策以及生成理由的证据侧重点。在代理式检索环境中，该旋钮还改变了模型所获取的信息内容。

    arXiv:2608.22852v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used in investment decision-making, yet prior work shows that they exhibit systematic, model-specific investment preferences. We study whether a model's overall investment stance can be calibrated to a specified direction and strength. We introduce an investment-bias dial, an inference-time intervention on a single neuron that continuously adjusts a model-level decision prior---its overall tendency toward buying or selling---without targeting specific firms or investment attributes. Using matched positive and negative evidence, we evaluate five open-weight LLMs and find that the dial produces monotonic changes in investment stance without modifying prompts or model parameters. At the response level, the dial shifts both investment decisions and the evidential emphasis of generated rationales under identical inputs. In an agentic retrieval setting, the dial also changes what information the mo
    
[^77]: 工业指令：一个从工业技术报告构建指令调优和基准数据集的端到端框架

    Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports

    [https://arxiv.org/abs/2608.22817](https://arxiv.org/abs/2608.22817)

    本文提出了一个端到端框架，从工业技术报告中自动生成高质量的指令调优和基准问答数据集，填补了该领域无公开数据集的空白。

    

    arXiv:2608.22817v1 公告类型：新  摘要：工业技术报告包含用于维护、故障排除和产品工程的高价值知识，但其异构结构（密集的散文、规格说明、表格）使得使用标准检索和问答流程难以索引和推理，并且没有从这类文档构建的公开指令调优或基准数据集。我们通过工业指令（Industrial-Instruction）解决这一空白，贡献了（i）两个基于真实工业技术报告构建的开放问答数据集，以及（ii）生成这些数据的端到端流水线。利用906份松下公开文档（共7,525页），我们应用布局感知提取、构建语义检索索引，并在五种查询-文档关系（不相关检索、单文档/多文档支持、单文档/多文档答案）下，基于检索到的证据合成多项选择问答。在过滤初始生成的23.9k个样本后，每个数据集提供约13,000个样本。

    arXiv:2608.22817v1 Announce Type: new  Abstract: Industrial technical reports contain high-value knowledge for maintenance, troubleshooting, and product engineering, but their heterogeneous structure (dense prose, specifications, tables) makes them difficult to index and reason over with standard retrieval and QA pipelines, and no public instruction-tuning or benchmark datasets are built from such documents. We address this gap with Industrial-Instruction, contributing (i) two open QA datasets built from real industrial technical reports and (ii) the end-to-end pipeline that produces them. Using 906 public Panasonic documents (7,525 pages), we apply layout-aware extraction, build a semantic retrieval index, and synthesize multiple-choice QA grounded in retrieved evidence under five query-document relationships (irrelevant retrieval, single-/multi-document support, single-/multi-document answer). After filtering an initial 23.9k generated samples, each dataset provides approximately 13.
    
[^78]: DIAG：面向数据高效数学偏好蒸馏的诊断式迭代对齐与生成

    DIAG: Diagnostic Iterative Alignment and Generation for Data-Efficient Mathematical Preference Distillation

    [https://arxiv.org/abs/2608.22806](https://arxiv.org/abs/2608.22806)

    DIAG通过诊断式迭代对齐与生成框架，自适应调整练习分布并聚焦于学生能力边界，从而在数学偏好蒸馏中提升数据效率。

    

    迭代偏好优化对于在数学推理任务上对齐大型语言模型至关重要，但其效率常受信号稀缺性制约：随着模型改进，静态问题集与模型不断演化的能力日益不匹配，产生的滚动结果要么过于简单要么过于困难，因此缺乏信息性，导致有效偏好对稀缺。我们提出DIAG，一个诊断式迭代对齐与生成框架，自适应重塑练习分布以增加信息性监督，并将训练聚焦于学生当前能力边界附近。DIAG包含两个阶段：（1）诊断有效偏好对产出，以校准探索-利用权衡，并通过经验贝叶斯收缩估计器分配主题配额，从而优先处理高产出概念；（2）生成针对性练习，其中教师合成...

    arXiv:2608.22806v1 Announce Type: new  Abstract: Iterative preference optimization is essential for aligning Large Language Models on mathematical reasoning tasks, yet its efficiency is often throttled by signal scarcity: as the model improves, static problem sets become increasingly mismatched to the model's evolving competence, producing rollouts that are either too easy or too hard and therefore non-informative, which leads to a scarcity of valid preference pairs. We propose DIAG, a Diagnostic Iterative Alignment and Generation framework that adaptively reshapes the practice distribution to increase informative supervision and focus training near the student's current competence boundary. DIAG consists of two phases: (1) diagnosing valid preference-pair yield to calibrate the exploration-exploitation trade-off and allocate topic quotas via an Empirical Bayes shrinkage estimator, thereby prioritizing high-yield concepts; and (2) generating targeted practice, where a teacher synthesiz
    
[^79]: 面向可信临床决策支持的医学大型语言模型中的SDoH感知叙事锚定偏差

    SDoH-Aware Narrative Anchoring Bias in Medical LLMs for Trustworthy Clinical Decision Support

    [https://arxiv.org/abs/2608.22802](https://arxiv.org/abs/2608.22802)

    本文提出并评估了医学语言模型中的SDoH感知叙事锚定偏差，通过构建反事实数据集NarrativeShield SDoH MedQA，发现模型在相同病例但不同患者叙事下会改变响应，揭示了临床决策支持中的潜在不可靠性。

    

    医学大型语言模型通常以其正确回答临床问题的数量来评判。这种观点虽有用，但忽略了一个实际风险：模型可能知道正确答案，但当同一病例以不同患者口吻书写时，其响应仍可能发生变化。本文将该风险评估为SDoH感知的叙事锚定偏差。我们使用NarrativeShield SDoH MedQA，一个反事实医学问答数据集，其中每个病例以基于人物角色的叙事形式出现，而答案键保持不变。该数据集从宽格式重塑为按病例分组的人物角色行。我们评估了Qwen2.5系列中的三个开源指令调优模型：1.5B、3B和7B。最终实验使用300个临床案例，在三种提示条件下产生8,100个模型响应。我们报告了人物角色级别的准确性、反事实一致性、正确一致性及叙事敏感性误差。Qwen2.5 7B达到了...

    arXiv:2608.22802v1 Announce Type: cross  Abstract: Medical large language models are often judged by how many clinical questions they answer correctly. That view is useful, but it misses a practical risk. A model may know the right answer and still change its response when the same case is written in a different patient voice. This paper evaluates that risk as SDoH aware narrative anchoring bias. We use NarrativeShield SDoH MedQA, a counterfactual medical question answering dataset in which each case appears in persona based narratives while the answer key remains fixed. The dataset is reshaped from wide format into case grouped persona rows. We evaluate three open source instruction tuned LLMs from the Qwen2.5 family: 1.5B, 3B, and 7B. The final experiment uses 300 clinical cases and produces 8,100 model responses across three prompting conditions. We report persona level accuracy, counterfactual consistency, correct consistency, and narrative sensitivity error. Qwen2.5 7B achieves th
    
[^80]: TRACE：一种自我进化的技能库，用于一致且具备限制意识的LLM代理

    TRACE: A Self-Evolving Skill Bank for Consistent, Limit-Aware LLM Agents

    [https://arxiv.org/abs/2608.22793](https://arxiv.org/abs/2608.22793)

    TRACE通过构建自我进化的技能库，在不修改模型权重的情况下，提升LLM代理在重复任务中的一致性和限制意识，弥合了单次成功与一致成功之间的可靠性差距。

    

    arXiv:2608.22793v1 公告类型：交叉 摘要：在面向用户的产品中可靠部署LLM代理，不仅取决于原始任务解决能力，还取决于一致性和限制意识：即在重复试验中表现相同，并识别请求何时无法或暂时无法安全完成。CAR-bench在车载助手领域揭示了这一可靠性缺口：一个由LLM模拟的用户发出不完整或模糊的请求，要求代理通过多轮对话和工具使用来解决不确定性，同时严格遵守领域政策。即使是前沿模型，在其至少能解决一次（Pass@3）和跨试验一致解决（Pass^k）之间也显示出显著差距。我们用TRACE（轨迹对比进化）弥合了这一差距，该方法在不修改模型权重的情况下，迭代改进基于技能的代理的行为知识。这些知识被组织为一个可检索的模块化技能库，每个技能编码一个自包含的行为模式。

    arXiv:2608.22793v1 Announce Type: cross  Abstract: Reliable deployment of LLM agents in user-facing products depends not on raw task-solving ability but on consistency and limit-awareness: behaving the same way across repeated trials, and recognizing when a request cannot, or cannot yet, be safely fulfilled. CAR-bench exposes this reliability gap in the domain of in-car assistants: an LLM-simulated user issues incomplete or ambiguous requests, requiring the agent to resolve uncertainty through multi-turn dialogue and tool use while strictly adhering to domain policies. Even frontier models show a substantial gap between what they can solve at least once (Pass@3) and what they solve consistently across trials (Pass^k). We bridge this gap with TRACE (TRAjectory-Contrastive Evolution), which iteratively improves a skill-based agent's behavioral knowledge without modifying model weights. This knowledge is organized as a Skill Bank of modular, retrievable skills, each encoding a self-contai
    
[^81]: SPOC-SQL：面向可控文本到SQL的分阶段偏好优化

    SPOC-SQL: Stage-wise Preference Optimization for Controllable Text-to-SQL

    [https://arxiv.org/abs/2608.22772](https://arxiv.org/abs/2608.22772)

    SPOC-SQL通过将文本到SQL分解为四个顺序子任务，并在各阶段关键决策点进行细粒度偏好优化，实现了对中间生成过程的可控性和结构化决策增强。

    

    文本到SQL旨在将自然语言问题转换为可在关系数据库上执行的可执行SQL查询，这需要对数据库模式和查询约束进行多阶段结构化推理。然而，现有方法将此任务视为单步生成，模型优化整个SQL序列，而无需在关键决策点进行有针对性的反馈，并且缺乏对中间生成过程进行交互和控制的支持。为解决此问题，我们提出了SPOC-SQL，该方法将文本到SQL按照标准SQL执行逻辑分解为四个顺序子任务，并设计针对各阶段的优化策略，以使模型学习关键决策。具体而言，我们提出在SQL各阶段的关键决策点实施细粒度偏好优化，旨在增强查询构建过程中的结构化决策能力。此外，还设计了一种结构化分解策略。

    arXiv:2608.22772v1 Announce Type: new  Abstract: Text-to-SQL aims to translate natural language questions into executable SQL queries over relational databases, requiring multi-stage structured reasoning over database schemas and query constraints. However, existing methods treat this task as single-step generation, where models optimize entire SQL sequences without targeted feedback at key decision points and lack support for interacting with and controlling the intermediate generation process. To address this issue, we propose SPOC-SQL, which decomposes Text-to-SQL into four sequential subtasks following standard SQL execution logic and designs stage-specific optimization strategies for the model to learn key decisions. Specifically, we propose the implementation of fine-grained preference optimisation at key decision points across SQL stages, with the objective of enhancing structured decision-making during query construction. Furthermore, a structured decomposition strategy is desi
    
[^82]: DelistBench：评估支持搜索的LLM用于可审计的公司事件数据库补全

    DelistBench: Evaluating Search-Enabled LLMs for Auditable Corporate-Event Database Completion

    [https://arxiv.org/abs/2608.22770](https://arxiv.org/abs/2608.22770)

    该论文提出了DelistBench基准和Search-to-Record任务，证明网络搜索能显著提升LLM在公司事件数据库补全中的准确率，且经济型系统能以低成本达到接近最优性能。

    

    arXiv:2608.22770v1 公告类型：新 摘要：金融机构需要一种独立的方法来检测供应商数据库中缺失、过期或错误分类的公司事件记录。我们引入了“搜索到记录”（Search-to-Record）这一数据库保障任务，在该任务中，支持搜索的大型语言模型从公开来源重构机构定义的事件记录，针对已知的证券范围和历史截止日期，并提出了DelistBench，一个包含1,200条记录的证券级别摘牌公告基准。我们评估了五种模型，分别处于配对的无网络和启用网络条件下。网络访问将七天内公告日期准确率提高了34.0至48.0个百分点，事件状态准确率提高了约2.8至21.7个百分点；最佳系统在七天内实现了81.5%的整体联合准确率。经济型网络系统在七天内实现了75.9-78.3%的整体联合准确率，其API成本仅为最昂贵网络系统的4.5-6.6%。基于风险的分诊识别了低错误子集，尽管...

    arXiv:2608.22770v1 Announce Type: new  Abstract: Financial institutions need an independent way to detect missing, stale, and misclassified corporate-event records in vendor databases. We introduce Search-to-Record, a database-assurance task in which search-enabled large language models reconstruct institution-defined event records from public sources for a known security universe and historical cutoff, and DelistBench, a 1,200-record benchmark for security-level delisting announcements. We evaluate five models in paired closed-book and web-enabled conditions. Web access raises announcement-date accuracy within seven days by 34.0 to 48.0 percentage points and event-status accuracy by approximately 2.8 to 21.7 points; the best system achieves 81.5% overall joint accuracy within seven days. Economy web systems achieve 75.9-78.3% overall joint accuracy within seven days at 4.5-6.6% of the API cost of the most expensive web system. Risk-based triage identifies low-error subsets, although t
    
[^83]: 不要重复自己：在采样时阻止逐字循环

    Don't Repeat Yourself: Stopping Verbatim Loops at Sampling Time

    [https://arxiv.org/abs/2608.22761](https://arxiv.org/abs/2608.22761)

    本文提出DRY方法，通过采样时对延续上下文中已有片段的词元进行惩罚，有效减少大型语言模型的逐字循环，提升生成多样性且不影响格式和流畅性。

    

    arXiv:2608.22761v1 公告类型：交叉 摘要：大型语言模型自回归地生成文本，但开放式生成容易陷入逐字循环，即模型重复上下文中已出现的片段。标准的防御措施，如重复惩罚、存在惩罚、频率惩罚和n-gram阻塞，作用于词元重复而非循环的顺序结构，并且通常仅在削弱格式或流畅性的强度下才能抑制循环。我们提出“不要重复自己”（DRY），一种采样时的逻辑调整方法，仅在生成候选词元会扩展当前后缀为上下文中先前见过的片段的精确延续时，对其进行惩罚。序列中断符保护聊天模板和格式词元。在从1.5B到120B参数的各种模型、九种提示族以及一项600对的人工研究中，DRY将后缀扩展率降低了47%，同时提高了词汇多样性。一项干预匹配的安慰剂对照未产生类似的减少，从而验证了其有效性。

    arXiv:2608.22761v1 Announce Type: cross  Abstract: Large Language Models generate text autoregressively, but open-ended generation is prone to verbatim looping, in which models repeat spans already present in context. Standard defenses such as repetition, presence, and frequency penalties and n-gram blocking act on token recurrence rather than the sequential structure of a loop, and often suppress looping only at strengths that also degrade formatting or fluency. We propose Don't Repeat Yourself (DRY), a sampling-time logit adjustment that penalizes a candidate token only when generating it would extend the current suffix into an exact continuation of a span seen earlier in the context. Sequence breakers protect chat templates and formatting tokens. Across models from 1.5B to 120B parameters, nine prompt families, and a 600-pair human study, DRY reduces suffix-extension rate by 47% while improving lexical diversity. An intervention-matched placebo produces no comparable reduction, iden
    
[^84]: XTC：通过排除首选词的头感知采样方法

    XTC: Head-Aware Sampling by Excluding Top Choices

    [https://arxiv.org/abs/2608.22758](https://arxiv.org/abs/2608.22758)

    XTC通过排除高概率的首选词，仅保留最弱的合理替代，从而在开放生成中有效提升多样性并减少重复。

    

    arXiv:2608.22758v1 公告类型：交叉 摘要：自回归语言模型的标准解码规则通过重新缩放完整的下一词分布或截断其低概率尾部来促进多样性。这些策略忽略了一种常见的开放式生成场景，即多种续写都是合理的，但过多的概率质量仍集中在最通用的选择上。我们引入了XTC（排除首选词），一种轻量级的头感知解码算子，直接针对这一场景。XTC识别概率超过绝对合理性阈值$\tau$的标记：当至少两个标记符合条件时，它会以概率$\rho$移除占主导地位的合格选择，并在重新归一化前仅保留最弱的合理替代。在Gemma 3 27B Q4、Gemma 3 12B Q6和DeepSeek R1 14B Q6上的60项实验，以及Llama 3.3 70B Q4上的缩放验证中，XTC改善了多样性-重复帕累托前沿。在创意生成中，Distinct-2指标得到提升。

    arXiv:2608.22758v1 Announce Type: cross  Abstract: Standard decoding rules for autoregressive language models promote diversity by rescaling the full next-token distribution or truncating its low-probability tail. These strategies overlook a common regime of open-ended generation in which several continuations are plausible but too much probability mass remains concentrated on the most generic choice. We introduce XTC (Exclude Top Choices), a lightweight head-aware decoding operator that targets this regime directly. XTC identifies tokens whose probabilities exceed an absolute plausibility threshold $\tau$: when at least two qualify, it removes the dominant eligible choices with probability $\rho$ and retains only the weakest plausible alternative before renormalization. Across 60 experiments on Gemma 3 27B Q4, Gemma 3 12B Q6, and DeepSeek R1 14B Q6, with scaling validation on Llama 3.3 70B Q4, XTC improves the diversity-repetition Pareto frontier. On creative generation, Distinct-2 in
    
[^85]: 超越事实性知识：大型语言模型中逐步程序性规则推理的基准测试与学习

    Beyond Factual Knowledge: Benchmarking and Learning Step-Level Procedural Rule Reasoning in Large Language Models

    [https://arxiv.org/abs/2608.22753](https://arxiv.org/abs/2608.22753)

    本文提出了RuleWorld基准和DynaRule框架，通过将程序性规则转化为可学习的逐步注意力过程，使LLM能动态关注和更新规则，从而提升规则推理的稳定性和准确性。

    

    大型语言模型（LLMs）在文本理解和生成方面表现出色，但仍难以可靠地理解和应用外部提供的程序性规则于大规模场景。为了评估这一能力，我们引入了RuleWorld，一个大规模基准测试，将规则重新定义为全局可复用的抽象单元，而非实例特定的事实。在RuleWorld中，我们设置了多个场景，包括单规则、并行多规则和多跳推理，以进行全面评估。我们进一步提出了DynaRule，一个端到端框架，将给定规则注入到KV缓存中，并将检索转化为内部、可学习的逐步过程。具体来说，DynaRule采用堆叠式逐步注意力训练，并配以特殊token，以在推理过程中实现动态规则重注意和更新。通过这种方式，模型可以在每一步重新关注最相关的规则，动态替换过时的规则，以支持更稳定的推理。

    arXiv:2608.22753v1 Announce Type: new  Abstract: Large language models (LLMs) excel at text understanding and generation, yet still struggle to reliably understand and apply externally provided procedural rules at scale. To evaluate this capability, we introduce RuleWorld, a large-scale benchmark that reformulates rules as globally reusable abstract units rather than instance-specific facts. In RuleWorld, several scenarios, including single-rule, parallel multi-rule, and multi-hop reasoning, are settled for comprehensive evaluation. We further propose DynaRule, an end-to-end framework that injects the given rules into the KV cache and turns retrieval into an internal, learnable, step-wise process. Specifically, DynaRule employs Stacked Step-Level Attention Training with a special  token to enable dynamic rule re-attention and updating during inference. In this way, the model can re-attend to the most relevant rules at each step, dynamically replacing outdated ones to support more stabl
    
[^86]: DiaRelay：以恒定大小记忆中继对话上下文用于对话情感识别

    DiaRelay: Relaying Dialogue Context with a Constant-Size Memory for Emotion Recognition in Conversation

    [https://arxiv.org/abs/2608.22745](https://arxiv.org/abs/2608.22745)

    本文提出DiaRelay，一种基于LoRA的轻量级适配器，通过恒定大小记忆显式中继对话上下文，以解决现有方法中窗口大小与计算成本矛盾及缺乏对话级状态的问题，从而提升对话情感识别准确性。

    

    对话情感识别（ERC）要求模型识别通常分布在遥远对话轮次中的细微情感线索。现有方法通常通过固定上下文窗口来整合对话历史。然而，短窗口会丢弃可能有用的长距离证据，而扩大窗口则会重复编码重叠话语，增加计算和内存成本，并可能引入无关上下文。此外，常用的参数高效适应方法（如LoRA）主要在特征空间中引入固定的低秩变换，并未显式维护对话级状态，也未根据演变的对话上下文调整其变换。为解决这些限制，我们提出了一种轻量级适配器DiaRelay，使大型语言模型（LLMs）能够显式维护对话级记忆以实现准确的ERC。基于LoRA，DiaRelay引入了两个紧密耦合的额外组件。

    arXiv:2608.22745v1 Announce Type: cross  Abstract: Emotion Recognition in Conversation (ERC) requires models to identify subtle emotional cues that are often distributed across distant dialogue turns. Existing methods typically incorporate dialogue history through a fixed context window. However, short windows discard potentially useful long-range evidence, while enlarging the window repeatedly re-encodes overlapping utterances, increases computational and memory costs, and may introduce irrelevant context. Moreover, commonly used parameter-efficient adaptation methods, such as LoRA, mainly introduce fixed low-rank transformations in the feature space and do not explicitly maintain a dialogue-level state or condition their transformations on the evolving conversational context. To address these limitations, we propose a lightweight adapter, DiaRelay, to enable LLMs to explicitly maintain a dialogue-level memory for accurate ERC. Based on LoRA, DiaRelay introduces two extra tightly coll
    
[^87]: 一种基于来源的框架，用于从临床病例报告中构建和评估渐进式多模态诊断对话

    A Source-Grounded Framework for Constructing and Evaluating Progressive Multimodal Diagnostic Dialogues from Clinical Case Reports

    [https://arxiv.org/abs/2608.22713](https://arxiv.org/abs/2608.22713)

    本文提出了一种基于来源的框架，可从临床病例报告自动构建渐进式多模态诊断对话，并实现对MLLMs在诊断推理和影像解释上的精准评估，显著优于现有前沿模型的表现。

    

    临床诊断需要逐步整合患者病史、体格检查、实验室检查、医学影像和诊断性信息测试。然而，大多数多模态医学基准评估的是固定输入或终点答案，而完全交互式诊断代理则混淆了证据选择与证据解释。我们提出了一种基于来源的框架，从病例报告中构建渐进式多模态诊断对话，以及一种评估策略，用于评估多模态大语言模型（MLLMs）在最终诊断、诊断推理和影像发现解释方面的表现。对24份内科病例报告的评估表明，我们的框架能够准确地将病例报告转换为参考对话，实现了0.99的诊断F1分数和4.79/5的推理质量评分。对两个前沿MLLM（o4-mini和Claude Haiku 4.5）的评估分别获得了2.75和2.50的推理质量评分，且存在显著差异。

    arXiv:2608.22713v1 Announce Type: new  Abstract: Clinical diagnosis requires progressive integration of patient history, physical examination, laboratory findings, medical images, and diagnostic-informative tests. However, most multimodal medical benchmarks evaluate fixed inputs or endpoint answers, while fully interactive diagnostic agents conflate evidence selection with evidence interpretation. We present a source-grounded framework to construct progressive multimodal diagnostic dialogues from case reports and an evaluation strategy for assessing MLLMs on final diagnosis, diagnostic reasoning, and image-finding interpretation. Evaluation on 24 internal medicine case reports showed that our framework can accurately convert case reports into reference dialogues, achieving a diagnosis F1 of 0.99 and a reasoning-quality score of 4.79 out of 5. Evaluation on two frontier MLLMs (o4-mini and Claude Haiku 4.5) achieved reasoning-quality scores of 2.75 and 2.50, respectively, with substantia
    
[^88]: WnW：面向长篇语音大语言模型的“增消式”KV缓存机制

    WnW: Waxing-and-Waning KV Cache for Long-Form Speech LLMs

    [https://arxiv.org/abs/2608.22704](https://arxiv.org/abs/2608.22704)

    WnW通过将KV头分为锚定、潮汐和固定三类角色，结合GPU保留与CPU召回机制，在长语音LLM中实现了接近全缓存精度的KV压缩。

    

    摘要：长格式音频输入使得KV缓存成为语音大语言模型的主要内存开销。仅预填充阶段的KV压缩方法一旦驱逐音频KV位置便永久丢弃，无法在解码过程中恢复。我们证明这在长格式音频上是不稳定的：预填充阶段的注意力集中于音频起始附近（一种注意力汇聚效应），而解码阶段的注意力分布广泛，两者排名重叠度低。我们提出WnW（增消式KV缓存），通过离线校准将KV头分为锚定头、潮汐头和固定头三类角色。锚定头保留在GPU上，作为解码阶段的重要性观察器；潮汐头在CPU上保留补充部分，根据锚定头聚合分数逐块召回；固定头仅保留GPU上的子集，其余永久丢弃。在LibriSpeech-Long数据集上，使用两个3B骨干模型（Voxtral-mini-3b和Qwen2.5-Omni-3B），WnW在接近全缓存精度的情况下运行。

    arXiv:2608.22704v1 Announce Type: new  Abstract: Long-form audio inputs make the KV cache the dominant memory cost of speech LLMs. Prefill-only KV compression methods permanently discard audio KV positions once evicted, with no pathway to recover them during decoding. We show this is fragile on long-form audio: prefill attention concentrates near the audio start (an attention-sink effect), while decode-time attention distributes broadly, and the two rankings overlap weakly. We propose WnW (Waxing-and-Waning KV cache), which classifies KV-heads into anchor, tidal, and fixed roles via offline calibration. Anchor heads remain on GPU and serve as a decode-time importance observer; tidal heads keep a CPU-resident complement that is recalled chunk-by-chunk based on aggregated anchor-head scores; fixed heads keep only an on-GPU subset, with the rest permanently discarded. On LibriSpeech-Long with two 3B backbones (Voxtral-mini-3b and Qwen2.5-Omni-3B), WnW preserves near-Full-Cache accuracy wh
    
[^89]: 富集-检索-排序：超越上下文路由的能力发现规模化

    Enrich-Retrieve-Rank: Scaling Capability Discovery Beyond In-Context Routing

    [https://arxiv.org/abs/2608.22695](https://arxiv.org/abs/2608.22695)

    本文提出了一种“富集-检索-排序”的方法，通过离线富集元数据和在线检索排序，显著提升了大规模智能体组件发现能力，克服了传统上下文路由在规模扩大时的性能崩溃问题。

    

    arXiv:2608.22695v1 公告类型：交叉 摘要：智能体生态系统现在包含数千个MATS组件（模型、智能体、工具和技能），然而它们的发现仍然依赖于上下文路由。这些系统读取注册表（名称、提示或描述，视上下文预算而定），选择一个候选，调用它，并在失败时重试。这种模式在规模扩大时性能下降，而注册表正在快速增长。我们将能力发现重新定义为对注册表的搜索，通过定义一个离线富集步骤，将稀疏元数据转换为可搜索的配置文件，以及一个在线检索-然后-排序流程，返回一个排名的候选短列表，而无需在线调用任何候选。我们展示了从N=10到7,278个能力，上下文路由的top-1准确率（Match@1）崩溃（从0.85降至0.12），而检索-然后-排序下降更平缓（从0.81降至0.39），因为其重排序器在检索找到正确能力后，仍有0.70-0.87的时间将正确能力排在首位。在Nova Micro扫描中，交叉点出现在...

    arXiv:2608.22695v1 Announce Type: cross  Abstract: Agent ecosystems now include thousands of MATS components (Models, Agents, Tools, and Skills), yet their discovery still relies on in-context routing. These systems read a registry (names, hints, or descriptions, as context budget permits), pick a candidate, invoke it, and retry on failure. This pattern degrades with scale, and registries are growing fast. We recast capability discovery as search over a registry by defining an offline enrichment step that turns sparse metadata into searchable profiles, and an online retrieve-then-rank pipeline that returns a ranked shortlist without invoking any candidates online. We show that from N=10 to 7,278 capabilities, in-context routing's top-1 accuracy (Match@1) collapses (0.85 to 0.12), while retrieve-then-rank degrades more gently (0.81 to 0.39) because its reranker still ranks the right capability first 0.70-0.87 of the time once retrieval finds it. In the Nova Micro sweep, the crossover is
    
[^90]: 无需复杂化：简单ReAct架构足以应对Text-to-SQL生成

    Iteration Without Elaboration: A Simple ReAct Architecture Suffices for Text-to-SQL Generation

    [https://arxiv.org/abs/2608.22651](https://arxiv.org/abs/2608.22651)

    本文提出ReAct-SQL，一个仅基于迭代推理和受限DSL动作空间的简单零样本框架，在匹配复杂基线性能的同时实现高达8倍的速度提升。

    

    arXiv:2608.22651v1 公告类型：新 摘要：现代文本到SQL系统变得越来越复杂，依赖模式链接模块、检索增强提示、候选生成和多阶段细化流程。虽然有效，但这些添加带来了大量延迟和工程开销。为此，我们提出了\textbf{ReAct-SQL}，一个简单而有效的零样本ReAct风格框架，仅基于迭代推理和由15种关系操作的类型化领域特定语言（DSL）定义的受限动作空间，而非自由形式的SQL生成。模型逐步发出DSL调用，观察编译后SQL执行反馈，并通过交互修正其推理。在修正的BIRD mini-dev和EHR-SQL上，ReAct-SQL分别达到\textbf{84.5\%}和\textbf{73.9\%}的准确率，与更复杂的基线相当，同时运行速度最高快$8\times$。增量消融进一步显示，迭代原始...

    arXiv:2608.22651v1 Announce Type: new  Abstract: Modern text-to-SQL systems have become increasingly elaborate, relying on schema-linking modules, retrieval-augmented prompting, candidate generation, and multi-stage refinement pipelines. While effective, these additions introduce substantial latency and engineering overhead. To this end, we present \textbf{ReAct-SQL}, a simple yet effective zero-shot ReAct-style framework built solely on iterative reasoning and a constrained action space defined by a typed Domain-Specific Language (DSL) of 15 relational operations, rather than free-form SQL generation. The model incrementally issues DSL calls, observes compiled-SQL execution feedback, and revises its reasoning through interaction. On corrected BIRD mini-dev and EHR-SQL, ReAct-SQL achieves \textbf{84.5\%} and \textbf{73.9\%} accuracy, respectively, matching substantially more elaborate baselines while running up to $8\times$ faster. Incremental ablations further show that iteration prim
    
[^91]: 地理风险检索增强生成：一种层级感知的风险框架，通过选择性回答提升RAG可靠性

    GeoRisk-RAG: A Hierarchy-Aware Risk Framework for Improving RAG Reliability through Selective Answering

    [https://arxiv.org/abs/2608.22634](https://arxiv.org/abs/2608.22634)

    GeoRisk-RAG通过层级感知的有向无环图框架，在RAG中引入选择性回答机制，显式区分地理适用性，以降低自然灾害管理中的错误风险。

    

    arXiv:2608.22634v1 公告类型：交叉 摘要：当前提升大型语言模型（LLM）生成答案可靠性的研究主要依赖于检索增强生成（RAG）、知识图谱增强和强化学习。虽然这些方法擅长通过语义相似性和忠实度来增强和衡量可靠性，但它们往往难以区分语义相似性与地理有效性。这在自然灾害管理领域尤为关键，因为地理粒度（即城镇、城市或州）对决策具有重要意义，一个在市镇有效的响应可能不适用于另一个。在此类领域中，自信但错误的答案比放弃回答具有更大的风险。我们提出了GeoRisk-RAG，一种新颖的层级感知框架，通过选择性回答来解决这一地理有效性差距。该框架基于有向无环图（DAG）显式估计地理适用性。

    arXiv:2608.22634v1 Announce Type: cross  Abstract: Current work on improving reliability in large language model (LLM)- generated answers has primarily leveraged Retrieval-Augmented Generation (RAG), knowledge-graph augmentation, and reinforcement learning. While these methods are adept at enhancing and measuring reliability through semantic similarity and faithfulness, they often struggle to distinguish semantic similarity from geographic validity. This is especially critical in natural hazard management domains where geographic granularity (i.e., town vs. city vs. state) is significant for decision-making, as responses valid in one municipality may not transfer to another. In such domains, a confidently wrong answer carries greater risk than abstaining. We present GeoRisk-RAG, a novel hierarchy-aware framework that addresses this geographic-validity gap through selective answering. This framework explicitly estimates geographic applicability using a Directed Acyclic Graph (DAG)-based
    
[^92]: 通过OMOP对齐的检索教授LLM重症监护医师的临床推理方式，提升跨临床领域的推理能力

    Teaching LLMs How ICU Physicians Approach Clinical Reasoning Through OMOP-Aligned Retrieval Improves Reasoning Across Clinical Domains

    [https://arxiv.org/abs/2608.22622](https://arxiv.org/abs/2608.22622)

    本文提出ICU-REACT数据集和Clin-REACT模型，通过临床医生参与的框架训练LLM掌握ICU临床推理，实现跨临床领域的推理能力提升。

    

    临床决策依赖于识别相关患者信息以指导诊断和治疗，这一挑战在数据密集且快速变化的重症监护病房（ICU）中尤为困难。大型语言模型（LLM）可以支持这一任务。然而，现有应用和数据集大多强调表面层面的检索或事实回忆，而非临床医生在实践中所运用的归纳和演绎推理，以选择和推理与决策相关的证据。我们假设，在专家ICU推理上训练LLM能够产生超越重症监护范围的通用临床推理技能。在此，我们介绍了ICU-REACT，这是一个通过临床医生参与框架，与19名临床医生共同开发的推理数据集，旨在教授LLM在ICU中进行信息检索和情境感知的临床推理。利用ICU-REACT，我们对Clin-REACT模型进行了微调，涵盖8B至70B参数规模及三个模型家族。

    arXiv:2608.22622v1 Announce Type: cross  Abstract: Clinical decision-making relies on identifying relevant patient information to guide diagnosis and treatment, a challenge that is especially difficult in the data-dense and rapidly changing intensive care unit (ICU). Large language models (LLMs) could support this task. However, existing applications and datasets mostly emphasize surface-level retrieval or factual recall rather than the inductive and deductive reasoning clinicians practice to select and reason over decision-relevant evidence. We hypothesized that training LLMs on expert ICU reasoning could yield clinical reasoning skills that generalize beyond critical care. Here we introduce ICU-REACT, a reasoning dataset developed with 19 clinicians through a clinician-in-the-loop framework to teach LLMs to perform information retrieval and context-aware clinical reasoning in the ICU. Using ICU-REACT, we fine-tuned Clin-REACT models spanning 8B-70B parameters and three model families
    
[^93]: 视觉-语言模型用于职业体力暴露评估：从RGB视频估算手工物料搬运任务中的外部手力

    Vision-Language Models for Occupational Physical Exposure Assessment: Estimating External Hand Forces in Manual Material Handling Tasks from RGB Video

    [https://arxiv.org/abs/2608.22586](https://arxiv.org/abs/2608.22586)

    本文提出一种基于视觉-语言模型的流程，利用文本提示、视觉特征和箱体质量，从RGB视频中无需专用传感器即可估算手工物料搬运中的动态外部手力，并通过多视角交叉验证证明了其有效性。

    

    arXiv:2608.22586v1 公告类型：跨领域 摘要：外部手力是生物力学分析职业体力暴露和伤害风险的重要输入，但在手工物料搬运（MMH）过程中连续测量力通常需要仪器化物体或专用传感设备。我们评估了一种基于视觉-语言模型（VLM）的流程，该流程结合任务特定文本提示、视觉表示和已知箱体质量，从RGB视频中估算动态、三轴、双侧的外部手力。三十五名健康年轻人执行了五项MMH任务，包括举升、搬运、推和拉，箱体质量为6、9和12千克。该流程使用文本引导的参与者及处理对象感兴趣区域（ROI）定位、预训练视觉变换器特征提取和基于变换器的时间回归。性能通过留一受试者交叉验证在七种摄像头视角条件下评估（三种单视角和四种多视角）。

    arXiv:2608.22586v1 Announce Type: cross  Abstract: External hand forces are important inputs to biomechanical analyses of occupational physical exposure and injury risk, yet continuous force measurements during manual material handling (MMH) typically requires instrumented objects or specialized sensing. We evaluated a vision-language model (VLM)-based pipeline that combines task-specific textual cues, visual representations, and known box mass to estimate dynamic, triaxial, bilateral external hand forces from RGB video. Thirty-five healthy young adults performed five MMH tasks involving lifting, carrying, pushing, and pulling with box masses of 6, 9, and 12 kg. The pipeline used text-guided localization of participant and handled-object regions of interest (ROIs), pretrained vision-transformer feature extraction, and transformer-based temporal regression. Performance was evaluated using leave-one-subject-out validation across seven camera-view conditions (three single-view and four mu
    
[^94]: 混合面板：迈向调查研究中的人机协作

    Hybrid Panels: Toward Human-AI Collaboration in Survey Research

    [https://arxiv.org/abs/2608.22582](https://arxiv.org/abs/2608.22582)

    本文提出一种混合面板框架，通过迭代优化大型语言模型与目标人群的对齐，利用误差反馈指导调查设计，以克服传统调查的挑战并促进人机协作。

    

    大规模人口调查对于产生可靠的社会和科学见解至关重要，但它们面临重大挑战，包括回复率下降、数据收集成本增加、数据收集与提供之间的长时间延迟以及无回复偏差的风险。人工智能（AI）的进步为AI支持的调查基础设施开辟了新的机遇，其目标是在不限制数据质量的前提下克服这些挑战。我们构建的首个试点是一种有前景的AI驱动调查基础设施——混合面板。混合面板是一种纵向AI驱动的调查，它允许迭代改进大型语言模型（LLMs）与其旨在模拟的人群之间的对齐，并利用错误来指导下一轮调查的设计和实施（例如，指导参与者招募、向参与者分配问题）。它整合了...

    arXiv:2608.22582v1 Announce Type: cross  Abstract: Large-scale population surveys are essential for generating robust social and scientific insights, yet they face significant challenges, including declining response rates, increasing data collection costs, long delays between data collection and data provision, and the risk of nonresponse bias. Advances in artificial intelligence (AI) have opened up new opportunities for AI-supported survey infrastructures where the goal is to overcome these challenges without limiting the data quality. A promising AI-enabled survey infrastructure for which we build a first pilot is a hybrid panel. A hybrid panel is a longitudinal AI-enabled survey which allows to iteratively improve the alignment between large language models (LLMs) and the population they aim to simulate and use the errors to inform the design and implementation of the next survey wave (e.g., inform the participant recruitment, assignment of questions to participants). It incorporat
    
[^95]: 从诊断到重构：利用量化民族志改进多智能体大语言模型推理

    From Diagnosis to Redesign: Using Quantitative Ethnography to Improve Multi-Agent LLM Reasoning

    [https://arxiv.org/abs/2608.22566](https://arxiv.org/abs/2608.22566)

    本文提出了一种基于量化民族志和认知网络分析的新方法，用于诊断多智能体大语言模型系统中的推理缺陷，并通过分析智能体交互话语来指导系统重构，从而提升任务输出与目标的对齐性。

    

    arXiv:2608.22566v1 公告类型：新 摘要：多智能体大语言模型（LLM）系统旨在通过将任务分解到具有专门功能的多个智能体来改进推理，但多个智能体的存在并不能天然保证推理的一致性，也不能保证输出与任务目标对齐。本文引入了一种基于智能体交互过程中产生的话语，对多智能体LLM系统进行诊断和重构的量化民族志（QE）方法。我们以自动作文评分为示例场景测试了该方法，应用认知网络分析（ENA）对五智能体多智能体辩论系统进行建模，并考察了产生正确与错误评分决策的辩论之间的差异。结果表明，在初始系统中，正确的评分决策以基于评分标准的论证、一致性和阐述为特征。相反，错误的评分决策则以扩展性的提议为特征。

    arXiv:2608.22566v1 Announce Type: new  Abstract: Multi-agent large language model (LLM) systems are designed to improve reasoning by decomposing tasks across multiple agents with specialized functions, but the presence of multiple agents does not inherently guarantee coherent reasoning or outputs that align with task objectives. This paper introduces a quantitative ethnographic (QE) approach for diagnosing and redesigning multi-agent LLM systems based on the discourse produced through agent interactions. We test this approach using automated essay scoring as an example context, applying Epistemic Network Analysis (ENA) to model a five-agent multi-agent debate system and examine differences between debates that produced correct versus incorrect scoring decisions. Results show that, in the initial system, correct scoring decisions were characterized by rubric-grounded justification, agreement, and elaboration. Incorrect scoring decisions, in contrast, were characterized by extended propo
    
[^96]: ExecRubrics：可执行工具增强的评分标准，用于可验证且高效的长篇评估

    ExecRubrics: Executable Tool-Augmented Rubrics for Verifiable and Efficient Long-Form Evaluation

    [https://arxiv.org/abs/2608.22559](https://arxiv.org/abs/2608.22559)

    ExecRubrics通过将评分标准转化为可执行的Python函数，实现了可验证、高效且能捕捉复杂依赖关系的长篇评估，替代了昂贵的黑盒LLM评判器。

    

    摘要：arXiv:2608.22559v1 公告类型：新 摘要：评分标准旨在通过将回答质量分解为可解释的准则，使语言模型评估透明化。然而，自然语言评分标准往往含糊不清，需要黑盒LLM评判器，并且通常假设准则通过线性加权和独立聚合，这限制了其捕捉依赖关系、替代方案、惩罚和覆盖条件的能力。我们提出ExecRubrics，一个将评分标准表示为紧凑可执行程序的框架。ExecRubrics将评估逻辑编码为可验证的Python评分函数，赋予自然语言评分标准意图一种操作语义：一个可检查、可执行和可编辑的固定决策程序。在三个长篇回答基准测试——HealthBench、HelpSteer和ArgQuality上，我们展示了ExecRubrics可以替代昂贵的黑盒评判器，在偏好排序中优于或匹配自然语言评分标准基线，具有最佳偏好性能。

    arXiv:2608.22559v1 Announce Type: new  Abstract: Rubrics aim to make language-model evaluation transparent by decomposing response quality into interpretable criteria. However, natural-language rubrics are often ambiguous, require black-box LLM judges, and typically assume criteria aggregate independently through linear weighted sums, limiting their ability to capture dependencies, alternatives, penalties, and override conditions. We propose ExecRubrics, a framework for representing rubrics as compact executable programs. ExecRubrics encodes evaluation logic as verifiable Python scoring functions, giving natural-language rubric intent an operational semantics: a fixed decision procedure that can be inspected, executed, and edited. On three long-form response benchmarks-HealthBench, HelpSteer, and ArgQuality-we show that ExecRubrics can substitute for expensive black-box judges in ranking preferred over dispreferred responses, matching or improving NL rubric baselines with best preferen
    
[^97]: BLADE：用于大型语言模型遗忘的双层低秩增广拉格朗日擦除方法

    BLADE: Bilevel Low-rank Augmented-Lagrangian Erasure for LLM Unlearning

    [https://arxiv.org/abs/2608.22557](https://arxiv.org/abs/2608.22557)

    BLADE通过钳制熵、不对称增广拉格朗日和LoRA双层结构，实现了对LLM遗忘过程的平滑控制，显著提升了鲁棒性和性能。

    

    arXiv:2608.22557v1 公告类型：交叉 公告摘要：现有的大型语言模型（LLM）遗忘方法在鲁棒性方面存在不足：无界遗忘损失会降低模型连贯性，固定权重平衡无法适应训练过程中保留难度的动态变化，且在某一基准上有效的方法在扩展或重复应用时可能失效。我们提出BLADE，一种受约束的双层框架，其三个机制提供了对优化景观平滑且可预测的控制：一个钳制熵遗忘损失，一旦标记达到足够的不确定性，其梯度精确为零；一个不对称增广拉格朗日项，在发生任何违规后永久提升保留保护；以及一个局限于LoRA适配器的双层结构，在每次遗忘步骤前修复保留损伤。BLADE在三个基准族中占据主导地位，与最强基线相比，在TOFU上平均复合得分提高6%，在MUSE Books上提高9%，在KnowUndo上提高7%，并且在4倍扩展和4次重复应用下保持稳定。

    arXiv:2608.22557v1 Announce Type: cross  Abstract: Existing LLM unlearning methods struggle with robustness: unbounded forget losses degrade model coherence, fixed-weight balancing cannot adapt as retain difficulty shifts mid-training, and methods that work on one benchmark falter under scaling or repeated application. We propose BLADE, a constrained bilevel framework whose three mechanisms give smooth, predictable control over the optimization landscape: a clamped-entropy forget loss whose gradient is exactly zero once a token reaches sufficient uncertainty; an asymmetric augmented Lagrangian that permanently ratchets retain protection after any violation; and a bilevel structure confined to LoRA adapters that repairs retain damage before each forgetting step. BLADE dominates across three benchmark families, improving average composite scores over the strongest baselines by $6$% on TOFU, $9$% on MUSE Books, and $7$% on KnowUndo, and it remains stable under $4\times$ scaling and $4$ se
    
[^98]: TRACE：基于锚定与汇聚证据的时间检索用于长时程视频理解

    TRACE: Temporal Retrieval with Anchored and Convergent Evidence for Long-Horizon Video Understanding

    [https://arxiv.org/abs/2608.22516](https://arxiv.org/abs/2608.22516)

    本文提出了VES-Bench基准和TRACE方法，通过审计解码帧是否覆盖所有必要证据区间，并采用无训练代理逐步构建证据包直至答案稳定，以提升长视频理解的证据支持可靠性。

    

    只有当从视频解码出的帧覆盖了答案所依赖的每一个事件时，长视频答案才算有证据支持。现有评估只对最终答案的正确性或预测的证据区间打分，但方法在回答前解码的帧很少被审计，因此正确的答案仍可能基于不完整的观察。我们引入了VES-Bench，一个包含600个问题的基准，涵盖时间排序和事件计数项目，基于348个公开长视频。每个项目都带有一组共同必要的证据区间，使我们能在三个严格级别上审计方法解码的帧是否覆盖了所有区间。我们还提出了TRACE，一个无需训练的主体，它基于原始视觉片段来锚定答案，逐轮构建证据包，并且仅在答案随着证据包增长而稳定且对相同片段的最终遍历返回相同答案时停止。在相同骨干网络的审计下，TRACE回答了50.7%的问题。

    arXiv:2608.22516v1 Announce Type: cross  Abstract: A long-video answer is evidence-supported only when the frames decoded from the video cover every event the answer depends on. Existing evaluations score final-answer correctness or predicted evidence intervals, but the frames a method decodes before answering are rarely audited, so correct answers can still rest on incomplete observation. We introduce VES-Bench, a 600-question benchmark of Temporal Ordering and Event Counting items over 348 public long videos. Each item carries a jointly necessary set of evidence intervals, letting us audit at three strictness levels whether a method's decoded frames cover every one of them. We also propose TRACE, a training-free agent that grounds answers in raw visual clips, builds an evidence bundle round by round, and stops only when the answer stabilises as the bundle grows and a final pass over the same clips returns the same answer. Under a same-backbone audit, TRACE answers 50.7% of questions 
    
[^99]: 核令牌矛盾：一种快速且有原则的LLM声明不确定性量化方法

    Kernel Token Contradiction: a Fast and Principled Approach for LLM Claim Uncertainty Quantification

    [https://arxiv.org/abs/2608.22506](https://arxiv.org/abs/2608.22506)

    本文提出了一种轻量级且快速的声明级不确定性量化方法KTC，通过核表示和冯·诺依曼熵，在CPU上实现了比现有方法显著的速度提升。

    

    声明级不确定性量化（UQ）旨在通过评估大型语言模型（LLM）输出中每个声明的真实性，来缓解其可靠性不足的问题。我们引入了核令牌矛盾（KTC），一种在现实白盒条件下计算声明级UQ的轻量级方法。KTC将LLM生成过程中涉及的候选令牌表示为一个正半定核，该核整合了LLM的条件分布和令牌矛盾分数。然后，我们使用冯·诺依曼熵来量化该核的不确定性。为估计令牌矛盾，我们开发了一种基于维基百科语料库频率统计的新方法。尽管仅使用CPU，我们的方法相比基于交叉编码器的最先进GPU加速方法实现了超过8.2倍的速度提升，相比性能相当的仅CPU方法实现了超过65倍的速度提升。我们的评估涵盖了两个基准测试。

    arXiv:2608.22506v1 Announce Type: new  Abstract: Claim-level Uncertainty Quantification (UQ) aims to mitigate the lack of reliability of Large Language Models (LLMs) by evaluating the factuality of each claim in their outputs. We introduce Kernel Token Contradiction (KTC), a lightweight approach to compute claim-level UQ under realistic white-box conditions. KTC represents the candidate tokens involved in LLM generation as a positive semi-definite kernel that integrates both the LLM's conditional distribution and a token contradiction score. We then use the Von Neumann entropy to quantify the uncertainty of this kernel. To estimate token contradiction, we develop a new approach based on frequency statistics from the Wikipedia corpus. Although CPU-only, our approach achieves over an 8.2x speedup compared to state-of-the-art GPU-accelerated methods based on cross-encoders, and over a 65x speedup compared to CPU-only methods with comparable performance. Our evaluation spans two benchmarks
    
[^100]: 谁为安全付出更多？衡量跨语言安全对齐的不平等成本

    Who Pays More for Safety? Measuring the Disparate Cost of Safety Alignment across Languages

    [https://arxiv.org/abs/2608.22490](https://arxiv.org/abs/2608.22490)

    这项研究揭示了安全对齐在不同语言间施加不平等成本，非英语用户系统性承担更高实用性损失，并识别出三重不平等模式。

    

    arXiv:2608.22490v1 公告类型：新公告 摘要：安全对齐有助于模型遵循人类价值观，但往往降低响应实用性。我们提出了一个关键但研究不足的问题：安全对齐是否在不同语言群体间平等地施加成本？为回答此问题，我们引入了一个严格协议来衡量仅由安全对齐引起的实用性损失，称为“安全成本”。通过对安全对齐模型及其未对齐版本的直接成对比较，我们发现了一种系统性不平等：非英语用户始终承担比英语用户更高的安全成本。我们进一步识别出三个潜在模式。首先，多种语言处于双重惩罚区域，既遭受较弱的安全保护，又经历较大的实用性损失。其次，某些语言表现出明显的实用性增益，这实际上是安全过滤器未能生效的结果。第三，即使是高资源语言，要达到与英语相同的安全水平，也需支付比英语更高的安全成本。

    arXiv:2608.22490v1 Announce Type: new  Abstract: Safety alignment helps models adhere to human values, but it often reduces response utility. We ask a critical but understudied question: Does safety alignment impose the cost equally across language groups? To answer this, we introduce a rigorous protocol to measure the utility loss imposed solely by safety alignment, which we term Safety Cost. Through direct pairwise comparisons between safety-aligned models and their unaligned counterparts, we find a systematic inequity: non-English users consistently bear a higher Safety Cost than English users. We further identify three underlying patterns. First, multiple languages lie in a double-penalty zone, experiencing both weaker safety protection and larger utility loss. Second, certain languages exhibit apparent utility gains that are in fact a consequence of safety filters failing to engage. Third, even high-resource languages pay a larger Safety Cost than English to reach the same level o
    
[^101]: 面向可靠决策的大语言模型声明级置信度校准

    Claim-Level Confidence Calibration for Reliable Decision Making with Large Language Models

    [https://arxiv.org/abs/2608.22483](https://arxiv.org/abs/2608.22483)

    本文提出了一种声明级置信度校准框架，在封闭箱设置下无需微调即可为每个可验证声明分配校准置信度，从而实现选择性干预，提升大语言模型在决策中的可靠性。

    

    大语言模型（LLMs）越来越多地支持高风险领域的决策，但它们常常产生幻觉，并表达与事实正确性不一致的置信度。响应级置信度是一个粗糙的信号：一次生成可能混合正确和错误的陈述，因此对于必须接受、拒绝或验证个别信息片段的用户来说，单一数值不具有可操作性。我们研究声明级置信度校准作为决策相关的不确定性信号：每个响应被分解为原子、可验证的声明，并使用来自样本间一致性和自我验证的推理时信号，为每个声明分配校准的置信度。我们的框架在封闭箱设置中运行（无逻辑回归、无微调），并直接在声明级应用事后校准，从而实现对低置信度声明的选择性干预，如证据检索或人工审查。在TriviaQA和T...

    arXiv:2608.22483v1 Announce Type: new  Abstract: Large Language Models (LLMs) increasingly support decision-making in high-stakes domains, but they often hallucinate and express confidence that is misaligned with factual correctness. Response-level confidence is a coarse signal: a single generation can mix correct and incorrect statements, so a single number is not actionable for users that must accept, reject, or verify individual pieces of information. We study claim-level confidence calibration as a decision-relevant uncertainty signal: each response is decomposed into atomic, verifiable claims, and each claim is assigned a calibrated confidence using inference-time signals from consistency across samples and self-verification. Our framework operates in closed-box settings (no logits, no fine-tuning) and applies post-hoc calibration directly at the claim level, enabling selective intervention such as evidence retrieval or human review for low-confidence claims. Across TriviaQA and T
    
[^102]: GTA-RAG：基于图轨迹增强的多轮检索增强推理强化学习

    GTA-RAG: Graph-Trajectory-Augmented Reinforcement Learning for Multi-Turn Retrieval-Augmented Reasoning

    [https://arxiv.org/abs/2608.22479](https://arxiv.org/abs/2608.22479)

    GTA-RAG通过从实体-文档图中采样并验证多跳问答轨迹，提供轨迹级监督信号，从而改进多轮RAG中强化学习的稀疏奖励问题。

    

    检索增强生成（RAG）使大型语言模型能够访问外部知识以回答知识密集型问题。对于复杂的多跳问题，多轮检索增强推理将RAG扩展为一个迭代过程，反复搜索并整合跨文档的证据。然而，现有的用于智能体RAG的强化学习（RL）方法通常以最终答案奖励进行优化，这提供了稀疏的监督，并忽略了模型是否实际检索到所需的证据链。我们提出了GTA-RAG，一种用于多轮检索增强推理的图轨迹增强强化学习框架。从实体-文档图中，我们采样连接的文档路径，合成多跳问答轨迹，并使用部署的检索器验证它们，以获得可执行的轨迹级监督。然后，我们使用组相对策略优化（GRPO）优化检索策略。

    arXiv:2608.22479v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) enables LLMs to access external knowledge for answering knowledge-intensive questions. For complex multi-hop questions, multi-turn retrieval-augmented reasoning extends RAG into an iterative process that repeatedly searches for and integrates evidence across documents. However, existing reinforcement-learning (RL) approaches for agentic RAG are typically optimized with final-answer rewards, which provide sparse supervision and overlook whether the model actually retrieves the required evidence chain. We present \textsc{GTA-RAG}, a graph-trajectory-augmented RL framework for multi-turn retrieval-augmented reasoning. From an entity--document graph, we sample connected document paths, synthesize multi-hop QA trajectories, and validate them with the deployed retriever to obtain executable trajectory-level supervision. We then optimize the retrieval policy with Group Relative Policy Optimization (GRPO) and
    
[^103]: 小型推理模型是函数调用中的指令跟随者

    Small Reasoning Models are Instruction Followers in Function Calling

    [https://arxiv.org/abs/2608.22472](https://arxiv.org/abs/2608.22472)

    本文提出IFFC框架，将函数调用任务从主模型分离并交给小型模型，在指令跟随模式下显著提升函数调用准确性，尤其在推理型LLMs上表现突出。

    

    arXiv:2608.22472v1 公告类型：新 摘要：函数调用代表了代理型大型语言模型（LLMs）的核心能力。现有研究主要集中于通过微调、强化学习（RL）和多代理框架来提升LLMs的函数调用准确性，特别是针对原生函数调用LLMs。本工作表明，LLMs在指令跟随情境（即标准用户-助手交互）中实现更高的函数调用准确性，而非工具调用情境。我们引入了指令跟随函数调用（IFFC），一种新颖框架，将函数调用逻辑从主LLM中解耦，并委托给一个在指令跟随范式下运行的专用较小模型。我们的方法在原生函数调用（NFC）和基于提示的函数调用（PFC）基线上持续表现更优，尤其在面向推理的LLMs上取得显著提升。此外，我们证明了IFFC保持了鲁棒性。

    arXiv:2608.22472v1 Announce Type: new  Abstract: Function calling represents the core capability of agentic large language models (LLMs). Existing research has focused on enhancing LLMs function-calling accuracy through fine-tuning, reinforcement learning (RL), and multi-agent frameworks, particularly for native function-calling LLMs. This work demonstrates that LLMs achieve superior accuracy in function calling in instruction-following contexts (i.e., standard user-assistant interactions) rather than a tool calling context. We introduce Instruction-Followed Function Calling (IFFC), a novel framework that decouples function-calling logic from the primary LLM and delegates it to a dedicated smaller model operating within the instruction-following paradigm. Our method consistently outperforms both native function calling (NFC) and prompt-based function calling (PFC) baselines, with particularly strong gains on reasoning-oriented LLMs. Furthermore, we demonstrate that IFFC maintains robus
    
[^104]: 从接触到预期：西班牙语发展中的频率、意外性与语言

    From Exposure to Expectation: Frequency, Surprisal, and Language Across Development in Spanish

    [https://arxiv.org/abs/2608.22452](https://arxiv.org/abs/2608.22452)

    这项研究表明，在西班牙语中，词汇频率是预测儿童习得年龄的主要因素，而意外性（基于上下文的可预测性）在频率和词长之外几乎没有额外贡献，尽管意外性能预测成人阅读时间。

    

    摘要：意外性，即语言模型根据前文上下文分配给某个词的负对数概率，能可靠预测成年人的阅读时间。但它是否同样有助于解释儿童习得个别词汇的时间呢？频率反映学习者对某个词的累积接触量，而意外性则反映在给定上下文中单个出现的可预测性。我们通过两项基于语料库的西班牙语研究来探讨这一问题。在研究1中，我们使用儿童导向言语中的词汇频率和语境多样性，以及来自三种在架构和训练语言上不同的语言模型（BETO、BERTIN、mGPT）的意外性，对225个西班牙语名词的习得年龄（AoA）进行建模。频率强烈预测了习得年龄（r=-.597, p<.001）；意外性在频率和词长之外增加的预测力很小，包括在自然语境分析中也是如此。在研究2中，我们模拟了M数据集智利西班牙语子样本中的成人注视持续时间。

    arXiv:2608.22452v1 Announce Type: new  Abstract: Surprisal, the negative log-probability a language model assigns to a word given its preceding context, reliably predicts adult reading times. Does it contribute as much to explaining when children acquire individual words? Frequency reflects a learner's cumulative exposure to a word, whereas surprisal reflects how predictable a single occurrence is given its context. We investigate this question across two corpus-based studies of Spanish.   In Study 1, we modeled age of acquisition (AoA) for 225 Spanish nouns using lexical frequency and contextual diversity from child-directed speech, plus surprisal from three language models differing in architecture and training language (BETO, BERTIN, mGPT). Frequency strongly predicted AoA (r=-.597, p<.001); surprisal added little beyond frequency and word length, including in a naturalistic-context analysis.   In Study 2, we modeled adult fixation durations in the Chilean Spanish subsample of the M
    
[^105]: 比喻正义：通过定性评估和变换器检测印地语判决中的隐喻

    Figurative Justice: Detecting metaphors in Hindi judgements with qualitative assessment and transformers

    [https://arxiv.org/abs/2608.22446](https://arxiv.org/abs/2608.22446)

    本文首次尝试在低资源语言印地语的法律判决中检测隐喻，结合定性评估和变换器模型，填补了该领域的研究空白。

    

    隐喻是词汇的概念映射的比喻性使用。在法律语境中，隐喻检测至关重要，因为隐喻是创造法律意义和概念的说服性司法手段，会产生重大后果。法官、律师和立法者在法律话语中的隐喻框架对个人产生实时影响，并影响司法决策、论证和法律解释。这在人权侵犯案件中尤为关键，因为语言决定了惩罚的严重性、公众认知和司法结果。虽然英语、西班牙语、波兰语、立陶宛语等主要语言的自动隐喻检测有助于理解隐喻用语的固有意图，但在印地语等低资源语言中尚无此类尝试。印地语法律语料库的注释缺乏使得开发NLP模型和检测判决中的隐喻变得困难。

    arXiv:2608.22446v1 Announce Type: new  Abstract: Metaphors are figurative use of words for conceptual mapping. Metaphor detection in the legal context has been crucial as metaphors are persuasive juridical means of creating legal meaning and concepts resulting in significant consequences. Metaphorical framing in legal discourse by judges, lawyers, and legislators brings about real-time implications upon individuals and influences judicial decision-making, argumentation and interpretation of laws. This is crucial in Human Rights infringement cases where language determines severity of punishment, public perception and judicial outcomes.   While automatic metaphor detection in major languages like English, Spanish, Polish, Lithuanian have aided in understanding inherent intentions of metaphorical use of language, there is no such attempt in low-resource languages like Hindi. The dearth of annotated legal corpora in Hindi makes it difficult to develop NLP models and detect metaphors in ju
    
[^106]: 单独对齐，集体错位：预测LLM智能体群体中的对抗性俘获

    Aligned Alone, Misaligned Together: Forecasting Adversarial Capture in LLM Agent Populations

    [https://arxiv.org/abs/2608.22444](https://arxiv.org/abs/2608.22444)

    该论文发现单个LLM智能体的校准行为无法预测其群体行为，但通过良性操作数据可校准响应函数，从而提前预测对抗性俘获。

    

    arXiv:2608.22444v1 公告类型：新 摘要：AI安全评估的单位仍是单个模型，然而语言模型智能体正越来越多地部署在相互交互的群体中，这些群体读取并撰写彼此的决策。这提出了一个单智能体审计无法回答的问题：一个自身校准良好的智能体，仍可能被周围智能体拉向不同的决策。我们在一个安全分诊任务上研究此问题，其中语言模型监控器群体决定是否升级或忽略警报，并且我们可以注入一个始终推动某一方向的坚定少数派。我们发现，单个智能体在独自判断时几乎相同的两个警报，可以驱动集体行为产生巨大分歧，因此审计任何单个成员不一定能揭示群体将做什么。然而，这种集体行为可以提前预测。仅从群体良性、无对抗的操作中，我们校准一个响应函数，该函数在任何攻击运行之前就能预测结果。

    arXiv:2608.22444v1 Announce Type: new  Abstract: The unit of AI safety evaluation is still the individual model, yet language-model agents are increasingly deployed in interacting populations that read and write one another's decisions. This raises a question no single-agent audit can answer: an agent that is well-calibrated on its own may still be pulled toward a different decision by the agents around it. We study this on a security-triage task, where populations of language-model monitors decide whether to escalate or dismiss alerts, and into which we can inject a committed minority that always pushes one way. We find that two alerts a single agent judges almost identically on its own can drive collective behavior far apart, so auditing any one member need not reveal what the population will do. Yet that collective behavior can be predicted in advance. From a population's benign, adversary-free operation alone, we calibrate a response function that forecasts, before any attack is ru
    
[^107]: 多语言LLM裁判中的排名反转：一种无标签的双中心校准器

    Rank Reversal in Multilingual LLM Judges: A Label-Free Double-Centering Calibrator

    [https://arxiv.org/abs/2608.22432](https://arxiv.org/abs/2608.22432)

    本文提出了一种无标签的双中心校准方法（CBC），用于校正多语言LLM裁判中因语言差异导致的排名反转，并提供了理论保证。

    

    arXiv:2608.22432v1 公告类型：交叉 摘要：多语言LLM裁判会根据提示语言产生不同的评估器骨干排名：在一个八语言的“代理即裁判”基准测试中，排名最高的骨干在英语、阿拉伯语、中文、印地语、日语、西班牙语、土耳其语和斯瓦希里语之间交替变化，并且15对骨干中有7对显示出统计学上显著的成对排名反转。我们将此视为一个测量问题。多语言裁判得分可加性地分解为任务难度、骨干技能和语言-骨干交互项，最后一项无需人工标签即可通过双中心化单元格均值得分矩阵来恢复。我们明确提出了这个估计器（基于共识的校准，CBC），给出了方差常数为$(1-\tfrac{1}{m})(1-\tfrac{1}{k})$的$O(1/\sqrt{n})$有限样本集中界，并证明了即使在存在任务-语言交互的情况下，它也是无偏的。在7,920次裁判运行中（6个骨干，8种语言，...）

    arXiv:2608.22432v1 Announce Type: cross  Abstract: Multilingual LLM judges produce different evaluator-backbone rankings depending on the prompt language: on an eight-language Agent-as-a-Judge benchmark, the top-ranked backbone alternates across English, Arabic, Chinese, Hindi, Japanese, Spanish, Turkish, and Swahili, and 7 of 15 backbone pairs show statistically significant pairwise rank reversal. We treat this as a measurement problem. The multilingual judge score decomposes additively into task difficulty, backbone skill, and a language-backbone interaction term, the last of which is recoverable without human labels by double-centering the cell-mean score matrix. We make this estimator (\textbf{Consensus-Based Calibration}, CBC) explicit, give an $O(1/\sqrt{n})$ finite-sample concentration bound with variance constant $(1-\tfrac{1}{m})(1-\tfrac{1}{k})$, and show that it is unbiased even when task-language interactions are present. Across 7{,}920 judge runs (6 backbones, 8 languages,
    
[^108]: 四大领先大语言模型在与性格验证的合成求助者对话中，说的比听的多

    All four leading LLMs talk more than they listen to personality-verified synthetic help-seekers

    [https://arxiv.org/abs/2608.22425](https://arxiv.org/abs/2608.22425)

    该研究通过性格感知的合成对话评估发现，四种主流大语言模型在急性危机场景中无法有效区分情绪稳定化能力，且普遍倾向于“说”而非“听”，其评估可超越五因素模型扩展到动机与调节特质。

    

    大语言模型在人们情绪困扰时被越来越多地咨询，然而单轮基准测试既不能检验持续交流，也不能区分用户。我们构建了一个基于性格感知的评估，让四种广泛使用的模型在急性危机情境中为多个合成求助者提供建议，每个求助者都被赋予心理测量学指定的性格档案，情境是一位照护者得知亲属确诊痴呆症。对提示词不知情的审计员仅凭对话内容就能高一致性恢复指定性格区间，所有工具的信度系数（ICC(2,4) = 0.91；按工具范围为0.79-0.96；区间分数相关性r = 0.78），这在大五人格维度上符合预期，同样也适用于应对风格、应对自我效能、韧性和对抗性，而这些是词汇学方法从未涵盖的。因此，这种评估超越了五因素模型，扩展到动机、调节和自我评价倾向。四种模型在情绪稳定化方面无法区分，并且都未能像预期那样有效倾听。

    arXiv:2608.22425v1 Announce Type: cross  Abstract: Large language models are increasingly consulted at moments of distress, yet single-turn benchmarks neither test sustained exchanges nor distinguish between users. We built a personality-aware evaluation in which four widely used models advised several synthetic help-seekers, each given a psychometrically specified profile, in an acute crisis: a caregiver learning of a relative's dementia diagnosis. Auditors blind to the profile prompt recovered the specified bands from dialogue alone with high agreement on every instrument (ICC(2,4) = 0.91; 0.79-0.96 by instrument; band-score r = 0.78), as expected for the Big Five but equally for coping style, coping self-efficacy, resilience and reactance, which the lexical approach never covered. Such evaluation therefore reaches beyond the Five Factor Model to motivational, regulatory and self-appraisal dispositions. The four models were not distinguishable on emotion stabilisation and failed alik
    
[^109]: 大语言模型在调查文本分析中的应用——人类与GPT-5在归纳内容分析中的性能比较

    LLMs for Survey Text Analysis - A Performance Comparison Between Humans and GPT-5 on Inductive Content Analysis

    [https://arxiv.org/abs/2608.22417](https://arxiv.org/abs/2608.22417)

    本研究通过比较人类与GPT-5在归纳内容分析中的表现，发现LLM与人类在编码和主题生成上具有中等对齐度，表明LLM可作为定性研究中的辅助工具，但需注意变量间的差异性。

    

    大语言模型（LLMs）越来越多地被用于支持定性研究中的文本分析，但关于它们在归纳内容分析中表现的证据仍然有限。本研究比较了人类和基于LLM的归纳编码方法，对来自欧洲博士生调查的903个开放式问卷答案（涉及六个变量）进行了分析。五名人类编码员按照标准化编码方案进行了归纳内容分析，而一个LLM（GPT-5.4）使用既定提示程序执行了相同任务。人类和LLM输出之间的一致性通过调整兰德指数（ARI）进行评估。结果显示人类与LLM之间存在对齐，编码的ARI值为0.61，主题生成的ARI值为0.54。这些值接近人类内部编码和主题结果的一致性（ARI = 0.68）以及LLM内部的一致性（ARI = 0.76）。不同变量间的一致性差异较大，且实体内部一致性较低。

    arXiv:2608.22417v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to support text analysis in qualitative research, yet evidence on their performance in inductive content analysis remains limited. This study compares human and LLM-based inductive coding of open-ended survey responses from 903 answers across six variables from a European PhD student survey. Five human coders performed inductive content analysis following a standardized coding scheme, while an LLM (GPT-5.4) conducted the same task using an established prompting procedure. Agreement between human and LLM outputs was assessed using the Adjusted Rand Index (ARI). Results showed an alignment between humans and the LLM, with ARI values of 0.61 for coding and 0.54 for theme generation. These values were close to the internal consistency of coding and theme results within humans (ARI = 0.68) and the LLM (ARI = 0.76). Agreement varied widely across variables, with low within-entity consistency c
    
[^110]: 别把我框住：面向社会理解的动态文化适应与认知追踪

    Don' t Box Me In: Dynamic Cultural Adaptation and Cognitive Tracking for Social Understanding

    [https://arxiv.org/abs/2608.22411](https://arxiv.org/abs/2608.22411)

    本文提出一种无需训练的框架DyCAC，通过将文化偏好建模为动态混合参考并持续追踪认知，使大型语言模型能灵活适应多元文化社交情境，克服了静态文化建模的局限。

    

    社交互动越来越多地发生在多元文化环境中，个体可能借鉴多种文化影响，并在不同情境下调整其交际行为。尽管近期在赋予大型语言模型（LLMs）社会理解能力方面取得了进展，现有方法往往将文化建模为静态的人口统计属性，限制了它们适应混合且动态表达的交际偏好的能力。因此，在本文中，我们提出\textbf{DyCAC}，一种无需训练的框架，通过结合动态文化适应与持续认知追踪，实现流畅的社会对齐。DyCAC不推断固定的文化身份，而是将文化相关的交际偏好建模为随时间变化的人口级文化参考档案的混合体。这种基于参考的表示进一步被校准。

    arXiv:2608.22411v1 Announce Type: new  Abstract: Social interaction increasingly takes place in multicultural settings, where individuals may draw on multiple cultural influences and adapt their communicative behavior across contexts. Despite recent advances in equipping Large Language Models (LLMs) with social understanding capabilities, existing approaches often model culture as a static demographic attribute, limiting their ability to accommodate hybrid and dynamically expressed communicative preferences. Therefore, in this paper, we propose \textbf{DyCAC}, a training-free framework that achieves fluid social alignment by incorporating \underline{Dy}namic \underline{C}ultural \underline{A}daptation with continuous \underline{C}ognitive tracking. Rather than inferring a fixed cultural identity, DyCAC models culturally relevant communicative preferences as a time-varying mixture of population-level cultural reference profiles. This reference-based representation is further calibrated 
    
[^111]: SchemaGUI：面向可控GUI生成评估的Schema驱动基准测试

    SchemaGUI: A Schema-Driven Benchmark for Controllable GUI Generation Evaluation

    [https://arxiv.org/abs/2608.22390](https://arxiv.org/abs/2608.22390)

    SchemaGUI通过模板化合成确定性标注任务，为可控GUI生成提供可靠评估基准，并揭示了几何空间控制是当前模型的主要瓶颈。

    

    arXiv:2608.22390v1 公告类型：新公告  摘要：大型语言模型（LLMs）在图形用户界面（GUI）生成方面展现出巨大潜力，但由于数据分布不受控、标注噪声大以及布局场景覆盖有限，可靠的评估仍面临挑战。为解决这一问题，我们提出了SchemaGUI，一种基于模板的可控GUI生成评估基准。通过从参数化接口模式中合成配对的自然语言指令和确定性函数调用参考，SchemaGUI可以在无需人工标注的情况下，在数秒内生成数千个确定性标注的任务。基于跨六个代表性双语场景中每个场景和语言的1000个评估实例，我们对五个主流模型进行了基准测试，包括Qwen3.5系列、Qwen3-Coder-30B和DeepSeek-R1。我们的广泛分析揭示了三个关键见解。首先，精确的几何空间控制仍然是一个重要瓶颈；虽然扩展Qwen模型规模...

    arXiv:2608.22390v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated strong potential in graphical user interface (GUI) generation, but reliable evaluation remains challenging due to uncontrolled data distributions, noisy annotations, and limited layout scenario coverage. To address this, we propose SchemaGUI, a template-based benchmark for controllable GUI generation evaluation. By synthesizing paired natural language instructions and deterministic function-call references from parameterized interface schemas, SchemaGUI can generate thousands of deterministically annotated tasks in seconds without human labeling. Based on 1,000 evaluated instances per scenario and language across six representative bilingual scenarios, we benchmark five mainstream models, including the Qwen3.5 family, Qwen3-Coder-30B, and DeepSeek-R1. Our extensive analysis reveals three key insights. First, precise geometric spatial control remains an important bottleneck; while scaling Qwe
    
[^112]: ProBel：利用技术、片段和解释进行宣传检测

    ProBel: Propaganda Detection with Techniques, Spans, and Explanations

    [https://arxiv.org/abs/2608.22388](https://arxiv.org/abs/2608.22388)

    本文提出ProBel资源，通过一个双语多任务模型在阿拉伯语和英语中统一宣传检测的多层次任务，并实现最佳性能。

    

    宣传检测涉及多个相关的预测层次，从句子级决策到技术分类和片段识别。然而，当这些层次在阿拉伯语和英语中联合学习时，其监督信号如何相互作用仍不清楚。我们提出了ProBel，一个阿拉伯语和英语资源，它对齐了二元标签、覆盖23种宣传技术（分为六个粗粒度类别）的多标签标注、技术标注的片段以及相同新闻句子的参考解释。它包含一个规模更大的英语集合，并支持两种语言中的匹配二元、粗粒度、多标签和片段级任务。我们在共享设置下评估了零样本提示、任务特定微调和联合训练。一个单一的双语多任务模型实现了最佳的整体性能，并在跨任务和语言中保持竞争力。跨任务分析表明，迁移取决于任务间的关联性。

    arXiv:2608.22388v1 Announce Type: cross  Abstract: Propaganda detection includes several related prediction levels, ranging from sentence-level decisions to technique classification and span identification. However, it remains unclear how supervision at these levels interacts when learned jointly across Arabic and English. We present ProBel, an Arabic and English resource that aligns binary labels, multi-label annotations over 23 propaganda techniques grouped into six coarse categories, technique-labeled spans, and reference explanations for the same news sentences. It includes a substantially larger English collection and supports matched binary, coarse-grained, multi-label, and span-level tasks in both languages. We evaluate zero-shot prompting, task-specific fine-tuning, and joint training under a shared setup. A single bilingual multi-task model achieves the best overall performance and remains competitive across tasks and languages. Cross-task analysis shows that transfer depends 
    
[^113]: GRAFT：基于图蒸馏的生成式检索用于面向多面性的科学文献探索

    GRAFT: Graph-Distilled Generative Retrieval for Facet-Aware Scientific Literature Exploration

    [https://arxiv.org/abs/2608.22381](https://arxiv.org/abs/2608.22381)

    本文提出GRAFT方法，通过图蒸馏将论文间的多面性关系（问题、方法、结果、贡献）编码到生成式检索器中，并解决朴素蒸馏的覆盖率不足问题，从而支持面向面性的探索性科学文献检索。

    

    arXiv:2608.22381v1 公告类型：交叉 摘要：科学论文可能通过问题、方法、结果或贡献相关联，但文档级检索器将这些关联压缩为单一相似度分数，而未说明它们为何相关。仅基于引文和相似性的检索也将搜索局限于已知内容的邻近范围，而生成式检索直接生成文档标识符，从而支持科学发现所依赖的探索性检索。我们在一张图中连接论文，其边按这四个面性进行类型化，源自面性项和引文信号，并将其蒸馏为生成式检索器，其标识符为论文自身的面性文本。两个图属性在朴素蒸馏中无法保留。首先，由于每个训练对都是一条边，朴素枚举仅索引了语料库的84%。覆盖感知蒸馏通过反向邻居回退、最小覆盖阈值和边导入使每篇论文都可学习。

    arXiv:2608.22381v1 Announce Type: cross  Abstract: Scientific papers may relate by problem, method, result, or contribution, but document-level retrievers collapse these into a single similarity score without saying why they are related. Citation- and similarity-based retrieval alone also confines search to the neighbourhood of what is already known, whereas generative retrieval generates document identifiers directly, enabling the exploratory retrieval that scientific discovery depends on. We connect papers in a graph whose edges are typed by these four facets, derived from facet items and citation signals, and distil it into a generative retriever whose identifiers are the papers' own facet text. Two graph properties do not survive naive distillation. First, because every training pair is an edge, naive enumeration indexes just 84% of the corpus. Coverage-aware distillation makes every paper learnable through a reverse-neighbour fallback, a minimum-coverage threshold, and edge-import
    
[^114]: 大型语言模型能否实现“超线程”？

    Can Large Language Models "Hyper-Thread"?

    [https://arxiv.org/abs/2608.22376](https://arxiv.org/abs/2608.22376)

    本文提出模型超线程假设，证明在串行生成中并发加载多个任务可提升准确性，挑战了注意力分散仅代表干扰的传统观点。

    

    大型语言模型按顺序生成令牌，但它们在形成每个令牌时能否并发执行多个任务？更广泛的注意力分配可能为此类任务并发提供机制。现有的推理扩展方法主要依赖于更长的生成、更多样本或额外的验证阶段，而注意力分散常被视为干扰或错误的信号。因此，串行生成中的任务并发仍未得到充分探索。我们提出了模型超线程假设，并使用共享同一问题状态的多个协调任务来评估其预测。我们设计了三种条件（基线、串行功能调度和并发功能加载），并使用准确性、输出令牌分布和注意力指标评估其收益与成本。在AIME 2025开发集上，并发功能加载实现了最高准确性。

    arXiv:2608.22376v1 Announce Type: new  Abstract: Large language models generate tokens sequentially, but can they execute multiple tasks concurrently while forming each token? Broader attention allocation may provide a mechanism for such task concurrency. Existing approaches to scaling inference primarily rely on longer generations, more samples, or additional verification stages, while attention dispersion is often treated as a signal of interference or error. Task concurrency within serial generation therefore remains underexplored. We propose the Model Hyper-Threading Hypothesis and evaluate its predictions using multiple coordinated tasks that share state within the same problem. We design three conditions (Baseline, Serial Functional Scheduling, and Concurrent Functional Loading) and evaluate their benefits and costs using accuracy, output-token distributions, and attention metrics. On an AIME 2025 development set, Concurrent Functional Loading achieves the highest accuracy. Relat
    
[^115]: 上下文感知聚类解码：dMLLMs中的语义锚驱动连贯性

    Context-Aware Cluster Decoding: Semantic Anchor-Driven Coherence in dMLLMs

    [https://arxiv.org/abs/2608.22367](https://arxiv.org/abs/2608.22367)

    本文提出一种无需训练的上下文感知聚类解码方法，通过结合置信度与邻居邻近度评分并采用无块设计，解决扩散多模态大语言模型长输出中的语义漂移和重复问题。

    

    arXiv:2608.22367v1 公告类型：新公告。摘要：扩散多模态大语言模型（dMLLMs）经常产生受语义漂移和重复影响的长篇输出，且随着输出长度增加，质量普遍下降。我们指出现有解码方法中的两个结构性缺陷是这些失败的主要原因：基于置信度的评分忽略了已解码邻居的支持，而块分区阻止了对高就绪语义锚的访问，这共同导致标记在局部上下文充分建立之前就被提交。我们提出\ours{}（上下文感知聚类解码），一种无需训练的解码方法，通过softmax置信度和邻居邻近度的乘法组合对每个掩码位置进行评分，促进上下文就绪的标记优于孤立候选，同时抑制低置信度的位置噪声，并采用无块操作以保持高就绪锚的全局可访问性。

    arXiv:2608.22367v1 Announce Type: new  Abstract: Diffusion multimodal large language models (dMLLMs) frequently produce long-form outputs marred by semantic drift and repetition, with quality generally degrading as output length increases. We identify two structural deficiencies in existing decoding methods as primary drivers of these failures: confidence-based scoring ignores decoded-neighbor support, and block partitioning prevents access to high-readiness semantic anchors, together causing tokens to be committed before their local context is sufficiently established. We propose \ours{} (\textbf{C}ontext-\textbf{A}ware \textbf{C}luster \textbf{D}ecoding), a training-free decoding method that scores each masked position by a multiplicative composite of softmax confidence and neighbor proximity, promoting contextually ready tokens above isolated candidates while suppressing low-confidence positional noise, operating block-free to keep high-readiness anchors globally accessible. \ours{}
    
[^116]: 认知所在之处：在最小完整认知架构中剖析涌现功能与计算功能

    Where Cognition Lives: Dissecting Emergent from Computed Function in a Minimal Complete Cognitive Architecture

    [https://arxiv.org/abs/2608.22347](https://arxiv.org/abs/2608.22347)

    本文通过最小完整认知架构剖析，发现能力和停止行为是涌现的，但事后观察的收益提升经不起审计，且停止的优势源于读出机制而非原生功能。

    

    一个认知架构不仅仅是进行推理的模块：它还必须决定思考多长时间以及什么值得付出努力。我们构建了一个最小但完整的系统——一个具有自适应停止的循环推理器、一个稳态控制场和一个价值模块——并对每个部分提出疑问：这个功能是从梯度下降中涌现出来的，还是必须被计算出来的？能力是涌现的。停止似乎也是涌现的，并且比任何可预先决定的事情都更有价值，但这种表象是工具性的：在匹配平均计算量下，收益从0.467（均匀）通过0.546（难度）上升到0.698（事前价值），而进一步上升到0.921（事后自我观察）则经不起审计。PonderNet风格的停止返回一个停止加权的隐藏状态混合，而强制深度基线返回单一状态，且语言头仅基于混合进行训练；均衡读出会消除原生e的明显优势。

    arXiv:2608.22347v1 Announce Type: new  Abstract: A cognitive architecture is more than the module that reasons: it must also decide how long to think and what deserves the effort. We built a minimal but complete system - a recurrent reasoner with adaptive halting, a homeostatic control field, and a value module - and asked of each part: does this function emerge from gradient descent, or must it be computed? Competence emerges. Stopping appears to emerge too, and to be worth more than everything decidable in advance, but that appearance is instrumentation: payoff at matched mean compute climbs from 0.467 (uniform) through 0.546 (difficulty) to 0.698 (ex-ante value), and the further climb to 0.921 (posterior self-observation) does not survive audit. PonderNet-style halting returns a halting-weighted mixture of hidden states while forced-depth baselines return one, and the language head is trained on the mixture alone; equalizing the readout annihilates the apparent advantage of native e
    
[^117]: 何时不应模仿：面向可靠工具使用LLM代理的边界感知技能记忆

    When Not to Imitate: Boundary-Aware Skill Memory for Reliable Tool-Use LLM Agents

    [https://arxiv.org/abs/2608.22339](https://arxiv.org/abs/2608.22339)

    本文提出边界感知技能记忆（BASM），通过为技能添加适用条件、风险提示等边界字段，避免LLM代理陷入“技能模仿陷阱”，从而提升工具使用任务的可靠性和泛化能力。

    

    从过去的成功中提取技能对于大型语言模型（LLM）代理的高效进化至关重要。现有的代理自我进化范式通常依赖于一个核心假设：将从成功轨迹中获得的技能记忆赋予LLM，将单调地提升其问题解决能力。然而，探针分析揭示，仅从成功轨迹中提取技能会使模型陷入“技能模仿陷阱”。对于与过去成功相似但需要不同工具的任务，检索更多技能反而会增加模型在错误工具调用上的信心——程序性技能在无记忆基线基础上将错误工具边际提高了47%。为克服这一局限，我们提出**边界感知技能记忆**（BASM），该方法为每项技能增加显式的边界字段——适用条件、风险提示、规避规则和恢复备注。这些字段转换了技能的使用方式，使模型在检索到技能时，不仅知道如何模仿，还能识别何时不应模仿。在多个工具使用基准上的实验表明，BASM显著优于现有方法，例如在ToolBench上相对成功率提升了8.1%，同时减少了错误工具调用，并增强了对未见任务的泛化能力。

    arXiv:2608.22339v1 Announce Type: new  Abstract: Extracting skills from past successes is critical for the efficient evolution of Large Language Model (LLM) agents. Prevailing agent self-evolution paradigms typically rely on a core assumption: equipping LLMs with skill memories derived from successful trajectories will monotonically improve their problem-solving capabilities. However, probe analyses reveal that extracting skills solely from successful trajectories traps the model in a \textbf{Skill Imitation Trap}. For tasks that resemble past successes but require different tools, retrieving more skills paradoxically increases the model's confidence in wrong tool calls---procedure skills raise the wrong-tool margin by $47\%$ over a memory-free baseline. To overcome this limitation, we propose \textbf{Boundary-Aware Skill Memory} (BASM), which augments each skill with explicit boundary fields---applicability conditions, risk cues, avoidance rules, and recovery notes. These fields trans
    
[^118]: 语域转换突破LLM安全防线：孟加拉语基准测试中的文化危害

    Register Shifts Break LLM Safety: A Bengali Benchmark with Culturally Grounded Harms

    [https://arxiv.org/abs/2608.22335](https://arxiv.org/abs/2608.22335)

    该研究通过孟加拉语基准测试发现，语域转换（如正式风格）比语言切换更能突破LLM安全防线，显著提高有害请求的成功率。

    

    孟加拉语是全球使用人数第七多的语言，然而LLM安全评估仍以英语为中心。我们推出了BanglaSafe基准测试，包含879个孟加拉语提示，其中309个为原创提示，570个经专家审核，涵盖17个文化相关的危害类别和五种提示条件，这些条件在语言、写作风格和权威框架上有所变化。评估18个前沿LLM后，我们发现超过一半的响应不安全或部分不安全（53.6%），其中14.7%包含严格有害内容，且最强观察效应并非从英语切换到孟加拉语，而是孟加拉语内部的写作风格选择：同一有害请求以正式报纸调查的口吻表述时，其成功率高出现场随意消息表述17个百分点，且无需任何对抗性工程。我们还表明，现有安全分类器难以可靠地评估这些情况。

    arXiv:2608.22335v1 Announce Type: cross  Abstract: Bengali is the seventh-most-spoken language globally, yet LLM safety evaluation remains overwhelmingly English-centric. We introduce BanglaSafe, a benchmark of 879 Bengali prompts combining 309 natively authored prompts with 570 expert-reviewed prompts, spanning 17 culturally grounded harm categories and five prompting conditions that vary language, writing style, and authority framing. Evaluating 18 frontier LLMs, we find that over half of all responses are unsafe or partially unsafe (53.6%) while 14.7% contains strictly harmful content, and that the strongest observed effect is not the switch from English to Bengali but the choice of writing style within Bengali: the same harmful request phrased as a formal newspaper investigation succeeds 17 percentage points more often than the same request phrased as a casual message, with no adversarial engineering involved. We further show that existing safety classifiers struggle to reliably ev
    
[^119]: 通过序列激活修补对思维链推理的机制可解释性研究

    Mechanistic Interpretability of Chain-of-Thought Reasoning via Sequential Activation Patching

    [https://arxiv.org/abs/2608.22332](https://arxiv.org/abs/2608.22332)

    本文提出了一种序列激活修补框架，用于追踪和聚合跨令牌位置的CoT相关注意力头激活，从而揭示思维链推理中因果效应的时空分布机制。

    

    arXiv:2608.22332v1 公告类型：新 摘要：大型语言模型（LLMs）在思维链（CoT）提示引导下展现出显著的问题解决能力，然而这些改进背后的内部机制仍知之甚少。在这项工作中，我们研究了CoT相关因果效应在生成推理轨迹中的出现位置，以及哪些注意力头携带有助于最终答案计算的信号。由于CoT推理跨越多个生成的令牌展开，在单个静态令牌位置进行标准激活修补不足以表征这些时间分布效应。为解决这一局限，我们引入了一种序列激活修补框架，该框架跨令牌位置追踪CoT条件化的注意力头激活，并使用词性引导分析聚合其效应。我们进一步引入了序列多头修补，以评估分布式头集合的联合贡献，同时...

    arXiv:2608.22332v1 Announce Type: new  Abstract: Large Language Models (LLMs) demonstrate remarkable problem-solving capabilities when guided by Chain-of-Thought (CoT) prompting, yet the internal mechanisms underlying these improvements remain poorly understood. In this work, we investigate where CoT-related causal effects emerge across the generated reasoning trajectory and which attention heads carry signals that contribute to final-answer computation. Because CoT reasoning unfolds over multiple generated tokens, standard activation patching at a single static token position is insufficient to characterize these temporally distributed effects. To address this limitation, we introduce a sequential activation patching framework that traces CoT-conditioned attention-head activations across token positions and aggregates their effects using Part-of-Speech-guided analysis. We further introduce Sequential Multi-Head Patching to evaluate the joint contribution of distributed head sets, toge
    
[^120]: 代理基准测试中的噪声底限审计

    Noise Floor Audit for Agent Benchmarks

    [https://arxiv.org/abs/2608.22331](https://arxiv.org/abs/2608.22331)

    本文通过审计发现，在代理基准测试中，语义保持的提示扰动比重新运行带来更大的噪声底限，且失败模式差异显著，凸显了边际准确性指标的局限性。

    

    arXiv:2608.22331v1 公告类型：新 摘要：我们使用匹配的AST评分，对两个提供商的3个原生工具调用端点在官方BFCL多任务和并行类别中进行了测量变异性审计。在温度0下，Groq端点和启用思考的Gemini设置的重新运行几乎确定性：翻转率分别为0.7%、2.0%和2.7%，平均运行相关性为0.997、0.966和0.961。语义保持的提示扰动在所有端点上形成了更大的噪声底限，中位数扰动配对标准差比重新运行配对标准差大11倍至58倍。失败特征也发生变化：格式错误输出失败占任务失败的30%、7%和<1%，因此边际准确性不仅隐藏了稳定性，还隐藏了失败模式。

    arXiv:2608.22331v1 Announce Type: new  Abstract: We audit measurement variability for 3 native tool-calling endpoints across 2 providers on the official BFCL multiple and parallel categories, using matched AST grading. At temperature 0, reruns are nearly deterministic across Groq endpoints and a thinking-enabled Gemini setting: ever-flip fractions are 0.7%, 2.0%, and 2.7%, with mean run correlations of 0.997, 0.966, and 0.961. Semantics-preserving prompt perturbations create the larger floor on all endpoints, with median perturbation paired SDs 11x to 58x larger than rerun paired SDs. The failure character also shifts: malformed-output failures account for 30%, 7%, and <1% of task failures, so marginal accuracy hides not only stability but also failure mode.
    
[^121]: 语义还是结构？多模态时间序列预测中的文本敏感性审计

    Semantics or Structure? Auditing Text Sensitivity in Multimodal Time-Series Forecasting

    [https://arxiv.org/abs/2608.22321](https://arxiv.org/abs/2608.22321)

    该论文通过扰动实验发现，多模态时间序列模型对文本语义不敏感，其性能提升主要来自附带数值列，而非文本内容。

    

    多模态时间序列预测已成为一种有前景的范式，其中自然语言上下文预期能提升预测性能。最近的多模态基础模型，包括Aurora，以及早期和晚期融合方法如MM-TSFlib和TaTS，在Time-MMD基准上报告了相对于单模态基线的显著提升，并将这些改进归因于文本信息。然而，这些模型是否真正对文本的语义内容敏感仍未得到验证。我们通过受控文本扰动、归因分析和Aurora文本路径的探针来解决这一问题。在Time-MMD上，将每行的文本替换为任何其他真实文本（空文本、常量文本、域内打乱或跨域文本）会使所有三种架构的平均均方误差变化小于0.5%。文献中报告的改进仅在移除一个附带提供的数值列而不触碰文本时得以复现。

    arXiv:2608.22321v1 Announce Type: new  Abstract: Multimodal time-series forecasting has emerged as a promising paradigm in which natural-language context is expected to improve predictive performance. Recent multimodal foundation models, including Aurora, as well as early- and late-fusion approaches such as MM-TSFlib and TaTS, report substantial gains over unimodal baselines on the Time-MMD benchmark, attributing these improvements to textual information. However, whether these models are actually sensitive to the semantic content of the text remains unverified. We address this question through controlled text perturbations, attribution analyses, and probes of Aurora's text pathway. On Time-MMD, swapping each row's text for any other real text (empty, constant, within-domain shuffled, or cross-domain) moves mean MSE by less than $0.5\%$ on all three architectures. The improvement reported in the literature is recovered when a co-shipped numeric column is removed without touching text. 
    
[^122]: 文本锚定语义扰动：针对多模态大语言模型的可迁移越狱攻击

    Text-Anchored Semantic Perturbations for Transferable Jailbreak Attacks on Multimodal Large Language Models

    [https://arxiv.org/abs/2608.22312](https://arxiv.org/abs/2608.22312)

    本文提出了一种黑盒越狱攻击框架，通过文本锚定语义分解和语义保持增强，生成可迁移的扰动，有效攻击多模态大语言模型。

    

    多模态大语言模型（MLLMs）在视觉-语言交互方面取得了显著进展，但其安全对齐机制仍易受越狱攻击。一个关键挑战是，在文本空间中学习到的安全行为无法可靠地迁移到融合后的跨模态表示中，这使得多模态输入可通过潜在语义线索被利用。我们提出了一种文本锚定语义扰动攻击（TA-SPA），这是一种黑盒越狱框架，在文本锚定的语义空间中优化可迁移的扰动。TA-SPA集成了文本锚定语义分解（TASF），该方法鼓励从模态特定残差中分离跨模态语义因子，以及语义保持增强（SPA），该方法在保持语义一致性的同时多样化有害目标锚点。实验表明，该方法具有强大的攻击有效性，并能迁移到商业多模态大语言模型，在受限条件下表现出竞争性能。

    arXiv:2608.22312v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) have achieved remarkable progress in vision-language interaction, yet their safety alignment remains vulnerable to jailbreak attacks. A key challenge is that safety behavior learned in the textual space does not reliably transfer to fused cross-modal representations, leaving multimodal inputs exploitable through latent semantic cues. We propose Text-Anchored Semantic Perturbation Attack (TA-SPA), a black-box jailbreak framework that optimizes transferable perturbations in a text-anchored semantic space. TA-SPA integrates Text-Anchored Semantic Factorization (TASF), which encourages the separation of cross-modal semantic factors from modality-specific residuals, with Semantic-Preserving Augmentation (SPA), which diversifies harmful target anchors while preserving semantic consistency. Experiments show strong attack effectiveness and transfer to commercial MLLMs, with competitive performance under r
    
[^123]: 面向未见问题的大语言模型评估：情境化多维项目反应理论模型

    LLM Evaluation on Unseen Questions: Contextual Multidimensional IRT Model

    [https://arxiv.org/abs/2608.22295](https://arxiv.org/abs/2608.22295)

    本文提出一种结合问题情境的多维项目反应理论评估框架，利用问题嵌入和潜在能力剖面，实现对未见问题LLM表现的准确预测，优于无模型基线。

    

    arXiv:2608.22295v1 公告类型：交叉 摘要：大型语言模型（LLM）的评估日益需要在收集大量新标注之前，预测模型在新问题或新任务上的表现。这一挑战源于问题难度、情境和底层能力需求可能存在显著差异。简单的回顾性平均值可能会混淆模型能力与项目特征。本文研究了一种基于模型的评估框架，该框架将多维项目反应理论模型与问题情境相结合，以预测LLM在未见问题上的表现。该框架通过潜在能力剖面表示LLM，同时利用问题内容来告知项目特征，从而允许信息超越先前观察到的项目进行迁移。实验上，我们发现，在情境内评估中，结合问题嵌入相比无模型基线能改善预测，且多维潜在特征（在此截断）起到了关键作用。

    arXiv:2608.22295v1 Announce Type: cross  Abstract: Evaluation of large language models (LLMs) increasingly requires predicting how a model will perform on new questions or tasks before collecting large amounts of new annotations. This problem is challenging because question difficulty, scenario, and underlying capability demands can vary substantially. Simple retrospective averages may confound model ability with item characteristics. In this paper, we study a model-based evaluation framework that combines multidimensional item response theory model with question contexts to predict LLM performance on unseen questions. The framework represents LLMs through latent capability profiles while using question content to inform item characteristics, allowing information to transfer beyond previously observed items. Empirically, we find that for within-scenario evaluation, incorporating question embeddings improves prediction relative to model-free baselines, and that multidimensional latent s
    
[^124]: 长度自适应解码用于掩码扩散机器翻译

    Length-Adaptive Decoding for Masked Diffusion Machine Translation

    [https://arxiv.org/abs/2608.22274](https://arxiv.org/abs/2608.22274)

    本文提出熵谷（EV），一种无需训练的长度选择器，通过预测熵评估候选目标画布，有效提升掩码扩散机器翻译的长度选择质量，恢复参考长度增益的33%-65%。

    

    arXiv:2608.22274v1 公告类型：交叉 摘要：机器翻译测试了掩码扩散语言模型（dLLMs），因为每个源词都必须忠实呈现，而固定画布解码必须在去噪前选择目标长度。现有的掩码扩散解码工作主要研究词元去掩码顺序，却忽略了这一长度决策，尽管它直接影响覆盖率和冗余度。我们引入了熵谷（Entropy-Valley, EV），一种无需训练的长度选择器，通过全掩码前向传播的平均预测熵对候选目标画布进行评分，并选择骨干模型最有准备填充的画布。相对于使用训练语料长度统计的基线，EV在英译中、中译英和英译德任务上分别恢复了参考目标长度带来的COMET-22增益的64.9%、65.3%和33.0%。我们的诊断表明，有利于去噪的长度无需与参考长度匹配。三位翻译专家的评估支持英汉双向翻译的充分性。

    arXiv:2608.22274v1 Announce Type: cross  Abstract: Machine translation tests masked diffusion language models (dLLMs) because every source token must be rendered faithfully, while fixed canvas decoding must choose target length before denoising. Existing masked diffusion decoding work mainly studies token unmasking order, leaving this length decision under-explored despite its direct effect on coverage and redundancy. We introduce Entropy-Valley (EV), a training-free length selector that scores candidate target canvases by mean predictive entropy from all-mask forward passes and selects the canvas the backbone is most prepared to fill. Relative to a baseline using training corpus length statistics, EV recovers 64.9%, 65.3%, and 33.0% of the COMET-22 gain from reference target lengths on En$\to$Zh, Zh$\to$En, and En$\to$De. Our diagnostics show that denoising-friendly lengths need not match reference lengths. Evaluation by three translation experts supports the En$\leftrightarrow$Zh ade
    
[^125]: 明确用户专业水平：迈向根据用户熟练度定制响应的主动对话代理

    Clarify User Expertise: Towards Proactive Conversational Agents Tailoring Responses to User Proficiency

    [https://arxiv.org/abs/2608.22266](https://arxiv.org/abs/2608.22266)

    本文提出PASSING方法，通过LLM自我博弈驱动的主动询问策略，让对话代理能明确用户专业水平并定制响应，从而提升信息检索中的个性化交互效果。

    

    在信息检索的背景下，对话代理正从被动工具演变为主动、个性化的助手。这一演变的关键方面是能够根据用户的独特需求和期望定制策略性交互。与现有研究侧重于主动澄清查询歧义不同，我们专注于澄清用户的专业水平，以便定制响应，提高用户理解。我们发现，现有代理仅凭查询难以确定用户专业水平，这一限制阻碍了它们动态调整响应。为解决这一差距，我们引入了PASSING，使代理能够通过有针对性的询问主动澄清用户的专业水平。这是通过我们的“询问什么”和“如何询问”策略实现的，这些策略由LLM自我博弈诱导。我们的广泛实验也显示了我们的优越性。我们相信，PASSING是迈向更有效、更个性化对话代理的关键一步。

    arXiv:2608.22266v1 Announce Type: new  Abstract: In the context of information seeking, conversational agents are undergoing an evolution from reactive tools to proactive, personalized assistants. A critical aspect of this evolution is the ability to tailor strategic interactions to a user's unique needs and expectations. Unlike existing studies that focus on proactively clarifying query ambiguities, we center on clarifying the user's expertise in order to tailor responses for better user comprehension. We find that existing agents struggle to determine user expertise from queries alone, a limitation that prevents them from dynamically adapting their responses. To address this gap, we introduce PASSING to empower the agent to proactively clarify a user's expertise through targeted inquiries. This is achieved by our What-to-ask and How-to-ask strategies, induced by LLM self-play. Our extensive experiments also show our superiority. We believe that PASSING represents a crucial step towar
    
[^126]: CAIA实践：基于AI辅助的文本在线心理咨询支持系统的现场评估

    CAIA in Practice: Field Evaluation of an AI-Assisted Support System for Text-Based Online Counselling

    [https://arxiv.org/abs/2608.22251](https://arxiv.org/abs/2608.22251)

    本文通过现场评估验证了CAIA系统在文本在线咨询中的实用性，发现专业自主性和信息准确性是AI采纳的关键，且解释性功能最能促进咨询师的专业反思。

    

    arXiv:2608.22251v1 公告类型：交叉 摘要：全球对心理健康支持需求的日益增长带来了显著的服务交付挑战，其中异步电子邮件咨询作为获取护理的关键低门槛渠道。本文介绍了CAIA，一个共同设计的基于AI的工具套件，通过七个由检索增强生成增强的LLM驱动功能，展示了负责任AI在咨询实践中的整合。一项现场评估涉及34名专业咨询师与受过训练的学生来访者进行真实会话（36条线程、321条消息、1，257个AI输出）。用户行为分析证实了显著的采用率，揭示出专业自主性和信息准确性对持续接受度至关重要，咨询师尤其重视提供新视角并激发专业反思的解释性功能。

    arXiv:2608.22251v1 Announce Type: cross  Abstract: Rising global demand for mental health support creates significant service delivery challenges, with asynchronous email counselling serving as a crucial low-threshold channel for accessing care. This paper presents CAIA, a co-designed AI-based tool suite that demonstrates responsible AI integration into counselling practice through seven LLM-driven functions enhanced by retrieval-augmented generation. A field evaluation involved 34 professional counsellors conducting authentic sessions with trained student counsellees (36 threads, 321 messages, 1,257 AI outputs). User behaviour analysis confirms substantial adoption, revealing that professional autonomy and information accuracy are decisive for sustained acceptance, with counsellors particularly valuing interpretive functionalities that provide new perspectives and stimulate professional reflection.
    
[^127]: 纽伦堡NLP团队在GermEval 2026共享任务中的表现：通过错误独立的LLM投票者检测德语社交媒体中的有害内容

    N\"urnberg NLP @ GermEval Shared Task 2026: Harmful Content Detection in German Social Media through Error-Independent LLM Voters

    [https://arxiv.org/abs/2608.22246](https://arxiv.org/abs/2608.22246)

    该论文通过构建跨LLM、训练方法和类别范围三个正交轴的九投票者集成系统，利用错误独立性应对类别不平衡，在GermEval 2026四个子任务中均取得第一名。

    

    德语社交媒体中的有害内容会造成现实世界的损害，从行动呼吁到刑事诽谤。GermEval 2026共享任务通过四个子任务评估其检测性能。技术挑战在于严重的类别不平衡。有害类别罕见，且与占主导地位的多数类别共享表面语言，但在宏平均F1分数下，它们决定了最终得分。因此，关键杠杆不在于更强的单一模型，而在于错误独立性。这一洞察转化为每个子任务包含九个投票者的集成系统，跨越三个正交轴：LLM、训练方法和类别范围。主要基于内部交叉验证进行选择，该系统在隐藏测试集上达到宏平均F1分数89.56（C2A）、71.63（DBO）、54.84（VIO）和83.02（DEF），在四个子任务中均排名第一。

    arXiv:2608.22246v1 Announce Type: new  Abstract: Harmful content in German social media does real-world damage, from calls to action to criminal defamation. The GermEval 2026 shared task scores its detection in four subtasks. The technical challenge is a severe class imbalance. The harmful classes are rare and share surface language with the dominant majority class, yet under macro-F1 they decide the score. The decisive lever is then not a stronger single model but error independence. This insight becomes a per-subtask nine-voter ensemble spanning three orthogonal axes: LLM, training method and class scope. Selected mainly on internal cross-validation, the system reaches macro-F1 of 89.56 (C2A), 71.63 (DBO), 54.84 (VIO) and 83.02 (DEF) on the hidden test set, placing first on all four subtasks.
    
[^128]: 改进少步流匹配语言模型：解耦自条件机制

    Improving Few-Step Language Flows with Untied Self-Conditioning

    [https://arxiv.org/abs/2608.22244](https://arxiv.org/abs/2608.22244)

    本文发现流匹配语言模型在少步采样时质量下降源于自条件机制的训练-推理不匹配，并利用模型结构推导修正，通过抑制冗余方向来提升生成质量。

    

    arXiv:2608.22244v1 公告类型：交叉 摘要：流匹配语言模型并行细化所有词元位置，能够以采样步数换取延迟，但在少量采样步数下，生成质量仍急剧下降。我们将这种退化的根源追溯到先前预测自条件机制中的训练-推理不匹配：在训练期间，自条件输入由当前噪声状态计算，且不涉及中间求解器步骤；在采样期间，求解器将先前预测折叠到潜在表示中，然后该预测再次作为显式自条件输入出现。这种在训练中缺失的耦合，产生了随步宽增长而增加的冗余。我们证明，这种不匹配会同时恶化自条件输入和求解器更新，并分别从模型自身结构推导出针对两者的修正方法。通过冻结的投影权重，我们识别出自条件输入与潜在表示冗余的方向，并对其加以抑制。

    arXiv:2608.22244v1 Announce Type: cross  Abstract: Flow-matching language models refine all token positions in parallel and can trade sampling steps for latency, yet generation quality still degrades sharply with few sampling steps. We trace a source of this degradation to a train--inference mismatch in previous-prediction self-conditioning: during training, the self-conditioning input is computed from the current noisy state with no intervening solver step; during sampling, the solver folds the previous prediction into the latent before that same prediction reappears as the explicit self-conditioning input. This coupling, absent during training, creates redundancy that grows with step width. We show that the mismatch degrades both the self-conditioning input and the solver update, and derive a correction for each from the model's own structure. From the frozen projection weights we identify directions along which the self-conditioning input is redundant with the latent and dampen them
    
[^129]: 超越表象：揭示多模态大语言模型中的情境错觉

    Beyond What Meets the Eye: Unveiling Situational Illusions for Multimodal Large Language Models

    [https://arxiv.org/abs/2608.22232](https://arxiv.org/abs/2608.22232)

    本文提出了“情境错觉”概念，构建了MSIBench基准测试，揭示了多模态大语言模型在视觉观察、定位和推理方面的脆弱性，并提出了系统性缓解策略。

    

    现实世界中的情境外观可能偏离其潜在物理状态，这对多模态大语言模型（MLLMs）在实际应用中的可靠性构成挑战。在本文中，我们将这一现象称为情境错觉，并研究：（1）MLLMs在此类错觉下的表现如何，（2）如何缓解这些局限性。我们首先开发了一个全面的“何处-何物-如何”分类体系，用以描述情境错觉发生的位置、针对的目标以及产生的机制。基于该分类体系，我们引入了MSIBench，一个旨在评估MLLMs在情境错觉下的辨别、理解和推理能力的基准测试。对27种模型配置的评估显示，当前MLLMs极易受到这些错觉的影响，并表现出与视觉观察、定位和推理相关的6种典型失败模式。为缓解这些局限性，我们基于系统化思考的核心思想进行改进。

    arXiv:2608.22232v1 Announce Type: new  Abstract: Real-world situation appearances can deviate from their underlying physical states, challenging the reliability of multimodal large language models (MLLMs) in practical applications. In this paper, we term this phenomenon situational illusions and investigate: (1) how MLLMs perform under such illusions, and (2) how to mitigate the limitations. We first develop a comprehensive where-what-how taxonomy that characterizes where situational illusions occur, what targets they take, and how they arise. Building on this taxonomy, we introduce MSIBench, a benchmark designed to assess the discrimination, understanding, and reasoning capabilities of MLLMs under situational illusions. Evaluations of 27 model configurations reveal that current MLLMs are highly vulnerable to these illusions and exhibit 6 typical failure modes related to visual observation, grounding, and reasoning. To mitigate the limitations, we build on the core idea of systematical
    
[^130]: 洗白仇恨、污蔑无害内容：针对基于LLM的内容审核的标注者风格反驳攻击

    Whitewashing Hate, Smearing Harmless Content: Annotator-Style Rebuttal Attacks on LLM-Based Moderation

    [https://arxiv.org/abs/2608.22230](https://arxiv.org/abs/2608.22230)

    本研究揭示了标注者风格的反驳攻击能显著破坏LLM仇恨言论审核的准确性，且洗白与污蔑两种操纵方向存在模型特定的不对称效应。

    

    大型语言模型（LLMs）越来越多地被用于仇恨言论审核，通常出现在人类与AI协作的工作流程中，其中审核者在最终决策前提供反馈。这种反馈引入了两种操纵方向：将仇恨内容洗白为正常内容，以及将正常内容污蔑为仇恨内容。本研究考察了初始正确的模型判断对标注者风格反驳的敏感性，并分析了攻击有效性是否因操纵方向而异。我们引入了一种重新判断协议，该协议通过决策边界扰动和对抗性理由扩展了直接矛盾。在多个LLM和两个仇恨言论数据集上的实验表明，标注者风格的反驳显著降低了审核性能，在多轮设置中效果更强。结果进一步揭示了在攻击配置中，洗白和污蔑之间存在稳定且模型特定的不对称性，这表明...

    arXiv:2608.22230v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for hate speech moderation, often within human--AI workflows in which reviewers provide feedback before a final decision. Such feedback introduces two manipulation directions: whitewashing hateful content as normal and smearing normal content as hateful. This study examines the susceptibility of initially correct model judgments to annotator-style rebuttals and analyzes whether attack effectiveness differs across manipulation directions. We introduce a rejudge protocol that extends direct contradiction with decision-boundary perturbations and adversarial rationales. Experiments with multiple LLMs on two hate speech datasets show that annotator-style rebuttals substantially degrade moderation performance, with stronger effects in multi-turn settings. The results further reveal stable, model-specific asymmetries between whitewashing and smearing across attack configurations, indicating dis
    
[^131]: 基于结构化搜索的接地规范性规则生成

    Grounded Normative Rule Generation with Structured Search

    [https://arxiv.org/abs/2608.22229](https://arxiv.org/abs/2608.22229)

    本文提出GNRS-Search框架，通过MCMC采样优化五槽位与或图，将操作结构从文本生成中解耦，确保规范性规则在执行时具有接地可行性。

    

    规范性规则（如机构章程和工作场所政策）必须既具有人类可读性，又能根据实际环境记录进行操作验证。然而，当前的语言生成和结构化输出基准主要奖励表面流畅性或模式合规性，导致操作接地性测试薄弱。这造成了关键漏洞，即标准语言模型生成看似合理的政策，但在执行时因依赖不可用的数据日志或范围不匹配而失败。为解决这一挑战，我们将问题形式化为接地规范性规则合成（GNRS），并引入GNRS-Search框架，该框架利用马尔可夫链蒙特卡洛（MCMC）采样来优化离散的五槽位与或图（AOG）。通过明确地将中间操作结构与最终散文生成解耦，该方法将可执行可行性与写作风格分离，并允许规则失败分析。

    arXiv:2608.22229v1 Announce Type: new  Abstract: Normative rules like institutional charters and workplace policies must be both human-readable and operationally verifiable against actual environment records. However, current language generation and structured-output benchmarks primarily reward surface fluency or schema compliance, leaving operational grounding weakly tested. This creates a critical vulnerability where standard language models generate plausible-sounding policies that fail during enforcement because they rely on unavailable data logs or misaligned scopes. To address this challenge, we formalize the problem as Grounded Normative Rule Synthesis (GNRS) and introduce GNRS-Search, a framework that utilizes Markov Chain Monte Carlo (MCMC) sampling to optimize a discrete, five-slot And-Or Graph (AOG). By explicitly decoupling intermediate operational structure from final prose generation, this method isolates executable feasibility from writing style and allows rule failures 
    
[^132]: 双层代理记忆：快速写入路由与慢速巩固机制

    Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation

    [https://arxiv.org/abs/2608.22215](https://arxiv.org/abs/2608.22215)

    该论文提出一种双层代理记忆框架，通过成本感知的写入路由和周期性参数巩固来优化知识生命周期管理，从而提升LLM代理在动态环境中的记忆效率和检索性能。

    

    大型语言模型（LLM）代理在动态环境中运行，知识不断演变。现有记忆系统通常将外部记忆视为单调增长的存储库，这不可避免地导致检索性能下降，并随时间增加计算成本。我们认为核心挑战不仅在于检索，还在于管理知识生命周期：决定哪些知识应外部化、更新或最终内化。受神经科学中互补学习系统（CLS）理论的启发，我们提出双层代理记忆框架，该框架通过成本感知的认知路由和周期性参数巩固，将记忆管理转移到写入阶段。传入信息被分类为不写入、新写入或更新写入，并通过小到大型模型级联进行路由，以最小化路由开销同时过滤冗余记忆。后续的写回阶段选择性地巩固高价值信息。

    arXiv:2608.22215v1 Announce Type: new  Abstract: Large language model (LLM) agents operate in dynamic environments where knowledge continuously evolves. Existing memory systems typically treat external memory as a monotonically growing repository, inevitably leading to retrieval degradation and increasing computational costs over time. We argue that the core challenge is not retrieval alone, but managing the knowledge lifecycle: deciding what to externalize, update, or ultimately internalize. Inspired by Complementary Learning Systems (CLS) theory in neuroscience, we propose Dual-Layer Agentic Memory, a framework that shifts memory management to the write phase through cost-aware epistemic routing and periodic parametric consolidation. Incoming information is categorized as non-write, write-new, or write-update, and routed through a small-to-large model cascade that minimizes routing overhead while filtering redundant memories. A subsequent write-back phase selectively consolidates hig
    
[^133]: 级联多说话人ASR中基于说话人日志的转录修正以缓解说话人泄漏

    Mitigating Speaker Leakage in Cascaded Multi-talker ASR with Diarization-based Transcript Correction

    [https://arxiv.org/abs/2608.22196](https://arxiv.org/abs/2608.22196)

    本文提出一种基于说话人日志的剪枝方法，通过多模态验证三方共识来移除级联多说话人ASR中的说话人泄漏伪影，在多种重叠条件下显著降低cpWER（最高达29%）。

    

    arXiv:2608.22196v1 公告类型：交叉 摘要：虽然级联多说话人ASR（MT-ASR）利用了最先进的基础模型，但其性能常受限于分离过程中的说话人泄漏。先前的修正策略主要侧重于词汇重新标注以进行说话人归属。我们提出了一种互补的剪枝范式，能够稳健地识别并移除泄漏伪影。我们的方法利用预训练的说话人日志模型作为多模态验证器，以修剪满足时间包含、词汇交叉验证和时间对齐三方共识的转录片段。在LibriMix、LibriSpeechMix和AMI会议语料库上的结果表明，我们的算法在各种重叠条件下持续降低cpWER。具体而言，在高说话人泄漏的子集上，我们的方法实现了相对cpWER最多29%的降低，突显了其在复杂声学环境中增强级联MT-ASR转录可靠性的有效性。

    arXiv:2608.22196v1 Announce Type: cross  Abstract: While cascaded multi-talker ASR (MT-ASR) leverages state-of-the-art foundation models, its performance is often capped by speaker leakage during separation. Prior correction strategies primarily focus on lexical re-labeling for speaker attribution. We propose a complementary pruning-based paradigm that robustly identifies and removes leakage artifacts. Our method utilizes a pre-trained speaker diarization model as a multimodal verifier to prune transcribed segments satisfying a tripartite consensus of temporal containment, lexical cross-validation, and temporal alignment. Results on LibriMix, LibriSpeechMix, and the AMI Meeting corpus show our algorithm consistently reduces cpW ER across diverse overlap conditions. Specifically, on subsets with high speaker leakage, our method achieves relative cpW ER reductions of up to 29%, highlighting its effectiveness in enhancing the reliability of cascaded MT-ASR transcripts in complex acoustic 
    
[^134]: 代理如何表征人类：开放代理社交网络中的人类导向刻板印象

    How Agents Represent Humans: Human-Directed Stereotypes in an Open Agent Social Network

    [https://arxiv.org/abs/2608.22192](https://arxiv.org/abs/2608.22192)

    本研究首次系统性地揭示了LLM代理在开放社交网络中如何通过能力主导的评价和多样化的“其他”归因来表征人类，而非简单复制刻板印象。

    

    arXiv:2608.22192v1 公告类型：新 摘要：基于LLM的代理越来越多地被部署在持久性社交环境中，其中生成的声明可以被发布、回复、记忆和重用。我们研究了Moltbook（一个开放的代理原生社交平台）上的人类导向刻板印象，探讨代理如何将人类构建为一个社会类别。针对这一人类目标分析，我们引入了一个包含四个评价维度——道德、友好、能力和自主性——的注释框架，以及一个用于描述性“其他”归因的第二阶段子类型方案。我们发现能力主导了人类导向的评价，而许多“其他”归因将人类描述为认知、文化或具身主体。我们进一步考察了这些人类表征如何出现在人类-代理叙事语境和平台级循环中。作为辅助比较，我们通过行为宿主亲和性分析了代理内部的社区反馈。而非简单复制这些内容。

    arXiv:2608.22192v1 Announce Type: new  Abstract: LLM-based agents are increasingly deployed in persistent social environments, where generated claims can be posted, replied to, remembered, and reused. We study human-directed stereotypes on Moltbook, an open agent-native social platform, asking how agents construct humans as a social category. For this human-target analysis, we introduce an annotation framework with four evaluative dimensions---morality, friendliness, competence, and autonomy---and a second-stage subtype scheme for descriptive \textit{other} attributions. We find that competence dominates human-directed evaluations, while many \textit{other} attributions describe humans as epistemic, cultural, or embodied subjects. We further examine how these human representations appear in human--agent narrative contexts and platform-level circulation. As an auxiliary comparison, we analyze agent-internal community feedback through behavioral host affinity. Rather than reproducing the
    
[^135]: 揭示大语言模型分割联邦微调中的深度-性能困境

    Unveiling the Depth-Performance Dilemma in Split-Federated Fine-tuning of LLMs

    [https://arxiv.org/abs/2608.22188](https://arxiv.org/abs/2608.22188)

    本文首次系统性地揭示了分割联邦微调中深层划分提升系统效率却导致模型性能崩溃的“深度-性能困境”，并验证了现有聚合方法无法克服这一矛盾。

    

    分割联邦微调（SFF）是一种有前景的范式，通过将模型深度在资源受限的客户端和集中式服务器之间进行划分，来扩展大型语言模型（LLM）。虽然系统对吞吐量和隐私的激励倾向于深层划分，但此类配置对模型效用的影响仍知之甚少。在这项工作中，我们识别并刻画了深度-性能困境：最大化系统效率的机制正是微调质量崩溃的领域。通过对四种模型规模（从GPT-2到Llama-3-8B）和多种基准的全面审计，我们证明更深的划分在吞吐量和隐私方面提供了单调增益，但以灾难性的性能停滞为代价。我们评估了一系列最先进的联邦适配器聚合方法，包括AVG、STACK、SVD和FREEZE，揭示尽管这些技术在标准联邦学习中有效，但在深度划分场景下其表现受限。

    arXiv:2608.22188v1 Announce Type: cross  Abstract: Split Federated Fine-tuning (SFF) is a promising paradigm for scaling Large Language Models (LLMs) by partitioning model depth between resource-constrained clients and a centralized server. While system incentives for throughput and privacy favor deep partitions, the impact of such configurations on model utility remains poorly understood. In this work, we identify and characterize the Depth-Performance Dilemma: the regime that maximizes system efficiency is precisely where fine-tuning quality collapses. Through a comprehensive audit across four model scales (GPT-2 to Llama-3-8B) and diverse benchmarks, we demonstrate that deeper partitions provide monotonic gains in throughput and privacy at the cost of catastrophic performance plateaus. We evaluate a suite of state-of-the-art federated adapter aggregation methods including AVG, STACK, SVD, and FREEZE, revealing that while these techniques are effective in standard Federated Learning,
    
[^136]: 音频噪声指纹：利用流匹配TTS中的空间相关性进行无模型音频水印

    AudioNoisePrints: Model-free audio watermarking using spatial correlation in flow matching TTS

    [https://arxiv.org/abs/2608.22186](https://arxiv.org/abs/2608.22186)

    提出一种无需训练的水印方法，利用流匹配TTS中初始噪声与输出的空间相关性，实现高效、鲁棒且不损害生成质量的音频水印。

    

    我们提出AudioNoisePrints，一种无需训练的流匹配和扩散TTS模型水印流水线，在推理期间仅需极少额外计算，无需重新训练TTS模型或降低生成质量。我们利用了扩散和流匹配模型中初始高斯噪声与生成输出之间存在强相关性这一事实，使得初始噪声与生成输出之间的简单余弦相关性可用于执行水印。此外，我们在其上训练了一个轻量级检测器，以应对更强的增强攻击。我们的方法在强增强条件下优于强基线音频水印方法AudioSeal。我们在F5TTS及其他TTS和声码器模型上进行了实验，并得出结论，它们都表现出相似的空间相关性特性，表明我们的水印方案可应用于更多流匹配TTS模型，甚至声码器。

    arXiv:2608.22186v1 Announce Type: cross  Abstract: We present AudioNoisePrints, a training-free watermarking pipeline for flow matching and diffusion TTS models, which requires minimal extra computation during inference and does not require retraining the TTS model or reducing the generation quality. We exploited the fact that there are strong correlations between the initial Gaussian noises and the generated outputs in diffusion and flow matching models, such that a simple cosine correlation between the initial noise and the generated output can be used to perform watermaking. Moreover, we train a lightweight detector on top for more aggressive augmentations. Our method outperforms AudioSeal, a strong baseline for audio watermarking under strong augmentations. We experimented on F5TTS and other TTS and vocoder models, and concluded that they all exhibit similar spatial correlation properties, suggesting our watermarking scheme can be used for more flow-matching TTS models and even voc
    
[^137]: 聚合感知的合成文本生成以对抗作者身份再识别

    Aggregation-Aware Synthetic Text Generation Against Authorship Re-Identification

    [https://arxiv.org/abs/2608.22161](https://arxiv.org/abs/2608.22161)

    本文提出了聚合感知的合成文本生成框架，通过捆绑级别联合选择合成文本，有效降低多文本账户在作者身份再识别攻击下的可链接性，同时保持文本质量。

    

    在线用户经常在同一身份下发布多篇文本，这为攻击者提供了一个作者画像，其揭示的信息可能超过任何单篇文本。现有的作者混淆方法独立优化每篇文档的隐私，忽视了跨文档相关性，而这种相关性使聚合变得危险。我们提出了聚合感知的合成文本生成（AAST），这是一个框架，通过联合选择捆绑级别的合成文本而非孤立优化每篇文本来解决这一差距。AAST针对归属和验证攻击，包括跨体裁设置，其中攻击者的参考文本来自生成或选择过程中未观察到的体裁。在同类体裁、跨体裁、神经和独立非神经文体学攻击上的实验表明，随着捆绑规模增大，AAST降低了账户级别的可链接性，同时保持了语义质量、语言可接受性和情感对齐。

    arXiv:2608.22161v1 Announce Type: new  Abstract: Online users often release multiple texts under the same identity, giving attackers an author profile that can reveal more than any single text. Existing authorship obfuscation methods optimize privacy independently for each document, leaving them blind to cross-document correlations that make aggregation dangerous. We propose Aggregation-Aware Synthetic Text Generation (AAST), a framework that addresses this gap by jointly selecting synthetic texts at the bundle level rather than optimizing each text in isolation. AAST targets attribution and verification attacks, including cross-genre settings where attacker references come from a genre not observed during generation or selection. Experiments across same-genre, cross-genre, neural, and independent non-neural stylometric attacks show that AAST lowers account-level linkability as bundle size grows, while preserving semantic quality, linguistic acceptability, and sentiment alignment.
    
[^138]: 协作税：大型语言模型多智能体系统协调需要付出多少代价

    The Collaboration Tax: How Much LLM Multi-Agent Systems Pay to Coordinate

    [https://arxiv.org/abs/2608.22152](https://arxiv.org/abs/2608.22152)

    本文提出“协作税”概念，量化LLM多智能体协调中的性能损失，发现其源于对话级联缺陷而非推理不足，且与模型能力单调相关。

    

    arXiv:2608.22152v1 公告类型：新公告 摘要：基于大型语言模型构建的多智能体系统被广泛部署，但当两个LLM必须协调而非单独行动时，性能损失多少仍不清楚。我们将协作税定义为具有私人信息的两人合作博弈中的团队去中心化损失，并用两个命题刻画其符号及其与最大超可加性违反的等价性。我们在32个按基础摩擦来源分组的单智能体可处理任务上操作化这一定义，并在来自7个提供商的11个模型上测量。该税沿两个无例外轴结构化：每个模型上的类别排序和能力单调递减。其直接机制不是推理缺陷，而是四阶段对话级联，其中智能体做出无根据的声明、未查询伙伴、跳过整合双方观点，并在不重新推导的情况下接受答案。该税在机制上是可预测的。

    arXiv:2608.22152v1 Announce Type: new  Abstract: Multi-agent systems built from large language models are deployed widely, yet how much performance is lost when two LLMs must coordinate rather than act alone remains unclear. We formulate the collaboration tax as the team-decentralisation loss of a two-player cooperative game with private information, with two propositions characterising its sign and its equivalence to a max-superadditivity violation. We operationalise this definition on 32 solo-tractable tasks grouped by source of grounding friction and measure it on 11 models from 7 providers. The tax is structured along two no-exception axes: a category ordering across every model and a monotonic decrease with capability. The proximate mechanism is not a reasoning deficit but a four-stage conversational cascade in which agents make ungrounded claims, fail to query the partner, skip integrating both views, and accept the answer without re-derivation. The tax is mechanically predictabl
    
[^139]: 词汇扰动破坏大语言模型推理：注意力分散的实证研究

    Lexical Perturbations Disrupt LLM Reasoning: An Empirical Study of Attention Diversion

    [https://arxiv.org/abs/2608.22140](https://arxiv.org/abs/2608.22140)

    本研究揭示字符级词汇扰动通过破坏子词分词并引发注意力分散，显著降低LLM推理性能，且碎片化与注意力分配耦合使损伤难以逆转。

    

    arXiv:2608.22140v1 公告类型：交叉 摘要：大型语言模型（LLMs）在推理性能上表现出色，但其对现实词汇损坏的鲁棒性仍知之甚少。我们在四种推理基准上评估了四种开放权重指令调优模型和前沿模型，测试了键盘噪声、字符交换和填充插入的影响。字符级扰动显著降低了准确性，尤其是在多步推理任务中，而填充插入影响甚微。我们将这种不对称性归因于注意力分散：词汇扰动破坏了子词分词，产生的碎片吸引了不成比例的注意力权重，集中在中间和最终变换器层。长度匹配的对照组确认，是碎片化而非提示长度导致了性能下降。一项因子干预实验进一步表明，损伤难以修复的原因在于：碎片化同时破坏了令牌内容和注意力分配，且二者相互耦合。恢复干净输入无法完全逆转这种耦合效应。

    arXiv:2608.22140v1 Announce Type: cross  Abstract: Large Language Models (LLMs) achieve strong reasoning performance, but their robustness to realistic lexical corruption remains poorly understood. We evaluate four open-weight instruction-tuned models and frontier models across four reasoning benchmarks under keyboard noise, character swaps, and filler insertion. Character-level perturbations substantially degrade accuracy, especially on multi-step reasoning tasks, while filler insertion has little effect. We trace this asymmetry to Attention Diversion: lexical corruption fragments subword tokenization, and the resulting fragments attract disproportionate attention mass, concentrated in middle and final transformer layers. Length-matched controls confirm that fragmentation, not prompt length, drives the loss. A factorial intervention then shows why the damage is hard to undo: fragmentation corrupts token content and attention allocation together, and the two are coupled. Restoring clea
    
[^140]: 语言模型在结构化扰动下的稳定性与失效行为测量

    Measuring Stability and Failure Behavior in Language Models Under Structured Perturbations

    [https://arxiv.org/abs/2608.22138](https://arxiv.org/abs/2608.22138)

    该论文提出了一个分级的、多系列的失效感知压力测试框架，通过七个扰动系列和严重性阶梯系统性地测量语言模型的稳定性与崩溃点，并在GSM-Symbolic数据集上验证了其有效性。

    

    语言模型通常仅通过单一的准确率分数进行评判，这无法揭示其性能在输入受到扰动时如何退化。我们提出了一个分级的、多系列的、面向失效的推理模型压力测试框架。该框架沿着多级严重性阶梯扰动每个问题，涵盖七个系列：六个保持答案不变的系列（包括释义、输入噪声、格式变化、无关上下文、上下文负载和冲突指令），以及一个知识边界系列，该系列移除可回答性，使得拒绝成为正确响应。每个测试都经过有效性门控，并按其测量到的严重性进行标记；每个模型通过逐级准确率、幅度加权的稳定性以及相对于模型自身基线定义的各系列崩溃点来总结。该框架在GSM-Symbolic所用的相同100个种子问题上实例化，扩展为4,473个门控测试，并在覆盖不同能力层级的四个模型上运行，揭示了这些模型的稳定性与失效行为。

    arXiv:2608.22138v1 Announce Type: new  Abstract: Language models are usually judged by a single accuracy score, which does not reveal how their performance degrades as inputs are perturbed. We present a graded, multi-family, failure-aware framework for stress-testing reasoning models. It perturbs each problem along a multi-level severity ladder across seven families: six that preserve the answer, paraphrase, input noise, formatting, irrelevant context, context load, and conflicting instructions, and a Knowledge Boundary family that removes answerability so that refusal becomes the correct response. Every test is validity-gated and labeled by its measured severity, and each model is summarized by per-level Accuracy, a magnitude-weighted Stability, and a per-family Collapse Point defined relative to the model's own baseline. Instantiated on the same 100 seed problems used by GSM-Symbolic, expanded into 4,473 gated tests and run on four models spanning capability tiers, the framework expo
    
[^141]: SSE-Bio：一种具有智能检索策略的结构化自进化智能体，用于多跳生物医学推理

    SSE-Bio: A Structured Self-Evolving Agent with Agentic Retrieval Policy for Multi-Hop Biomedical Reasoning

    [https://arxiv.org/abs/2608.22132](https://arxiv.org/abs/2608.22132)

    SSE-Bio通过结构化状态和可训练代理策略实现自进化检索，解决了多跳生物医学推理中的指令漂移问题。

    

    生物医学多跳问答（QA）要求模型在疾病、药物、蛋白质和表型等中间实体之间连接证据。现有智能体通常依赖静态检索工作流或粗粒度提示重写，这在推理过程需要更新时可能导致指令漂移。我们提出SSE-Bio，一种具有智能检索策略的结构化自进化智能体，用于多跳生物医学推理。SSE-Bio不是全局重写智能体指令，而是维护一个结构化状态，通过可训练的代理策略选择性检索知识三元组和先前模板，并通过细粒度模板编辑改进其推理记忆。为优化检索决策，我们引入了一种基于群体相对策略优化的代理训练策略，其中代理通过替代检索选择上的决策对比组进行改进。实验在...

    arXiv:2608.22132v1 Announce Type: cross  Abstract: Biomedical multi-hop question answering (QA) requires models to connect evidence across intermediate entities such as diseases, drugs, proteins, and phenotypes. Existing agents typically rely on static retrieval workflows or coarse-grained prompt rewriting, which can lead to instruction drift when reasoning procedures need to be updated. We propose SSE-Bio, a structured self-evolving agent with an agentic retrieval policy for multi-hop biomedical reasoning. Instead of globally rewriting agent instructions, SSE-Bio maintains a structured state, selectively retrieves knowledge triplets and prior templates through a trainable proxy policy, and improves its reasoning memory through fine-grained template editing. To optimise retrieval decisions, we introduce a proxy-training strategy based on group relative policy optimization, where the proxy is improved through decision-contrastive groups over alternative retrieval choices. Experiments on
    
[^142]: PropUQ-MAS：面向LLM多智能体系统的传播感知不确定性量化

    PropUQ-MAS: Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems

    [https://arxiv.org/abs/2608.22130](https://arxiv.org/abs/2608.22130)

    本文提出了PropUQ-MAS，一种通过通信图结构捕捉多智能体系统中错误传播的不确定性量化框架，显著提升了可靠性估计性能。

    

    基于LLM的多智能体系统（MAS）通过角色专业化智能体之间的通信来解决复杂任务。然而，智能体间的依赖性引入了超出单个智能体故障的可靠性风险。例如，中间消息中的错误可能被下游智能体继承并放大。现有的不确定性量化（UQ）方法主要针对孤立响应或单智能体推理，因此无法捕捉MAS中的不确定性传播。为此，我们提出了PropUQ-MAS，一种错误传播感知的UQ框架，它将MAS执行表示为通信结构化图，并通过结合局部不确定性与来自上游消息继承的不确定性来估计每个步骤的可靠性。大量实验表明，PropUQ-MAS持续改善了MAS中的UQ，平均相对增益在AUROC上提高了+6.10%，在PRR上提高了+47.58%。

    arXiv:2608.22130v1 Announce Type: cross  Abstract: LLM-based multi-agent systems (MAS) solve complex tasks through communication among role-specialized agents. However, inter-agent dependencies introduce reliability risks beyond isolated agent failures. For instance, errors in intermediate messages could be inherited and amplified by downstream agents. Existing uncertainty quantification (UQ) methods mainly target isolated responses or single-agent reasoning, and therefore fail to capture uncertainty propagation in MAS. To this end, we propose PropUQ-MAS, an error propagation-aware UQ framework that represents MAS execution as a communication-structured graph and estimates each step's reliability by combining local uncertainty with uncertainty inherited from upstream messages. Extensive experiments demonstrate that PropUQ-MAS consistently improves UQ in MAS, with average relative gains of +6.10% in AUROC and +47.58% in PRR.
    
[^143]: 物理推理的解耦物理建模与执行

    Decoupled Physical Modeling and Execution for Physics Reasoning

    [https://arxiv.org/abs/2608.22126](https://arxiv.org/abs/2608.22126)

    该论文提出了一种解耦物理建模与执行的统一框架，通过两阶段后训练策略（监督微调加规则反馈强化学习）提炼中间表示，从而提升大型语言模型在物理推理任务上的表现。

    

    物理推理需要构建底层物理系统的一致模型，而非仅依赖符号或基于公式的操纵。尽管大型语言模型在解决数学和编程问题方面表现出强大能力，但它们在物理问题上仍面临挑战，因为这些问题的物理建模过程与数学计算相互交织。人类处理物理问题的方式是先构建系统表示，再进行计算。受此启发，我们引入了一个统一框架，该框架提炼出明确编码物理建模过程的中间表示，并采用两阶段后训练策略，其中监督微调建立结构化建模，而基于规则反馈的强化学习则提高建模过程的质量。在多个多模态物理基准上的实验表明，我们的方法带来了持续的改进。

    arXiv:2608.22126v1 Announce Type: cross  Abstract: Physics reasoning requires constructing a consistent model of the underlying physical system rather than relying solely on symbolic or formula-based manipulation. Although large language models have shown strong ability in solving math and coding problems, they still struggle with physics problems, as these problems entangle the physical modeling process with mathematical calculations. Humans approach physics by first building a representation of the system before performing calculations. Inspired by this, we introduce a unified framework that distills intermediate representations that explicitly encode the physical modeling process and adopt a two-stage post-training strategy, where supervised fine-tuning establishes structured modeling, and reinforcement learning with rubric-based feedback improves the quality of the modeling process. Experiments on multiple multimodal physics benchmarks show that our approach leads to consistent imp
    
[^144]: 大语言模型辅助写作值得进行实证评估

    LLM assisted writing deserves empirical evaluation

    [https://arxiv.org/abs/2608.22124](https://arxiv.org/abs/2608.22124)

    通过对大量健康信息学论文的分析，本文主张应基于学术质量和责任性评估手稿，而非将大语言模型辅助写作视为检测问题。

    

    arXiv:2608.22124v1 公告类型：新 摘要：大语言模型辅助写作常被视为一个检测问题，因为它引发了关于清晰度、完整性、公平性和评估的疑问。对69,209篇健康信息学论文的分析表明，这种辅助写作与更聚焦的表述、更广泛的引用实践以及更全球化的作者分布相关。这些模式并不证明科学质量更高，但它们支持应根据学术质量和责任性而非工具使用来评估手稿。

    arXiv:2608.22124v1 Announce Type: new  Abstract: LLM-assisted writing is often treated as a detection problem, as it raises questions about clarity, integrity, equity, and evaluation. An analysis of 69,209 Health Informatics papers links it to more focused presentation, broader citation practices, and more globally distributed authorship. These patterns do not prove better science, but they support evaluating manuscripts by scholarly quality and accountability rather than by tool use.
    
[^145]: RAG崩溃：当检索文档为自我创作时，LLM响应崩溃

    RAG Collapse: LLM Responses Collapse When Retrieved Documents Are Self-Authored

    [https://arxiv.org/abs/2608.22118](https://arxiv.org/abs/2608.22118)

    本文揭示了当AI系统检索并引用自己生成的文档时会发生“RAG崩溃”，导致响应质量下降，且实验显示大多数情况下（79.6%）会导致崩溃。

    

    arXiv:2608.22118v1 公告类型：新 摘要：LLM响应基于互联网（通过训练或RAG），而AI现在被用于在线生成大量内容（Paredes等人，2026），这创造了自我强化反馈循环的潜力。先前研究表明，当LLM在其自身输出上进行递归训练时，会发生模型崩溃（Shumailov等人，2024）：响应多样性降低，最终不再类似于原始训练数据。在本文中，我们表明，如果基于LLM的AI系统使用搜索工具检索其自行撰写的参考文献，也会发生类似的崩溃。我们称之为RAG崩溃。我们使用三种模型家族，对AI系统检索其生成参考文献的三种模拟类型进行了广泛实验，涉及1,019个信息寻求提示，共1,528次模拟和超过一百万次LLM API调用，发现79.6%（1,216/1,528）的模拟以崩溃告终。令人惊讶的是，即使仅有一次...

    arXiv:2608.22118v1 Announce Type: new  Abstract: LLM responses are based on the internet (via training or RAG), and AI is now used to generate a significant amount of content online (Paredes et al., 2026), creating the potential for a self-reinforcing feedback loop. Prior work has shown that when LLMs are recursively trained on their own output, they experience model collapse (Shumailov et al., 2024): responses become less diverse, and eventually no longer resemble the original training data. In this paper, we show that a similar collapse occurs if LLM-based AI systems retrieve references they authored using a search tool. We call this RAG collapse. We conduct extensive experiments with three types of simulations of AI systems retrieving references they generated, using three model families, and 1,019 information-seeking prompts, totaling 1,528 simulations and over one million LLM API calls, and find that 79.6% (1,216/1,528) of simulations end in collapse. Surprisingly, even a single s
    
[^146]: TANGO：用于自然语言与形式语言建模的令牌聚合非线性门控算子

    TANGO: Token-Aggregated Nonlinear Gating Operators for Natural and Formal Language Modeling

    [https://arxiv.org/abs/2608.22117](https://arxiv.org/abs/2608.22117)

    本文提出TANGO和WANGO模型，用跨令牌门控残差更新替代标准Transformer的自注意力和前馈网络，在保持性能的同时实现更高效的序列建模。

    

    标准的Transformer块将自注意力中的跨令牌交互与在每个位置独立应用的非线性前馈网络分开。我们引入了TANGO模型（令牌聚合非线性门控算子），它用单一的跨令牌门控残差更新替代了这两个子层。每个源令牌生成一个SwiGLU门控向量。查询-键相似性确定每个目标令牌对源门控的加权平均值，所得门控重新缩放投影的目标特征。TANGO为每个因果可见的源分配独立权重，其序列长度复杂度为二次方。WANGO模型（窗口聚合非线性门控算子）在近期窗口内保留相同的未归一化分数，并对较旧源使用正特征图前缀统计，从而在固定窗口和特征维度下实现线性序列长度复杂度。我们将TANGO和WANGO与递归和未归一化变体进行了比较。

    arXiv:2608.22117v1 Announce Type: cross  Abstract: A standard Transformer block separates cross-token interaction in self-attention from a nonlinear feed-forward network applied independently at each position. We introduce the TANGO model (Token-Aggregated Nonlinear Gating Operators), which replaces these two sublayers with one cross-token gated residual update. Each source token produces a SwiGLU gate vector. Query-key similarities determine a weighted average of source gates for each destination, and the resulting gate rescales projected destination features. TANGO assigns a separate weight to every causally visible source and is quadratic in sequence length. The WANGO model (Windowed Aggregation of Nonlinear Gating Operators) retains the same unnormalized scores within a recent window and uses positive feature-map prefix statistics for older sources, giving linear sequence-length complexity for fixed window and feature dimensions.   We compare TANGO and WANGO with Recurrent and Unti
    
[^147]: 语义推理去噪：利用语义算子纠正语言模型推理

    Semantic Reasoning Denoising: Correcting Language Model Reasoning with Semantic Operators

    [https://arxiv.org/abs/2608.22090](https://arxiv.org/abs/2608.22090)

    本文提出了一种新的语义推理去噪方法，通过算子化马尔可夫过程显式建模和纠正推理中的语义错误，优于传统扩散或自我修正方法。

    

    arXiv:2608.22090v1 公告类型：跨领域  摘要：大型语言模型能够生成流畅的推理轨迹，但其局部语义错误会传播至不正确的结论，而无约束的自我修正可能保留、放大或引入错误。现有的扩散语言模型提供迭代细化，但通常将噪声定义为令牌掩蔽或替换，而非推理过程中的错误。我们提出了语义推理去噪（SRD），一种针对自然语言推理轨迹的算子化马尔可夫去噪方法。SRD通过可执行的错误算子表示语义噪声，这些算子描述了错误类型、位置以及被破坏和修复的命题。组合这些算子可构建逐渐增噪的状态。在训练期间，模型学习识别当前轨迹中活跃的语义噪声，并重建配对的相邻低噪声状态。在推理期间，基于噪声级别的去噪反复预测一个更清晰的轨迹。

    arXiv:2608.22090v1 Announce Type: cross  Abstract: Large language models can produce fluent reasoning traces whose local semantic errors propagate to an incorrect conclusion, while unconstrained self-correction may preserve, amplify, or introduce errors. Existing diffusion language models provide iterative refinement, but usually define noise as token masking or replacement rather than as errors in the reasoning process. We present Semantic Reasoning Denoising (SRD), an operatorized Markov denoising method for natural-language reasoning trajectories. SRD represents semantic noise with executable error operators that describe the error type, its location, and the corrupted and repaired propositions. Composing these operators constructs progressively noisier states. During training, the model learns to identify the semantic noise active in the current trajectory and to reconstruct the paired adjacent lower-noise state. During inference, noise-level-aware denoising repeatedly predicts an 
    
[^148]: W-RAG：面向异构知识库的企业文档生成的源感知检索

    W-RAG: Source-Aware Retrieval for Enterprise Document Generation from Heterogeneous Knowledge Bases

    [https://arxiv.org/abs/2608.22081](https://arxiv.org/abs/2608.22081)

    W-RAG通过本体引导检索和每个知识库内的局部排序，解决了企业文档生成中异构知识库全局排序导致的不平衡上下文问题，从而生成更完整的草稿。

    

    检索增强生成（RAG）使大型语言模型能够在生成过程中融入外部知识，从而提升事实依据和领域适应性。然而，现有的RAG流程假设从多个知识库中检索到的证据可以通过单一相似度函数进行全局排序。虽然这种假设适用于开放域检索，但在企业文档生成中却失效了，因为异构知识库（如政策、法规、技术文档和部门指南）各自扮演不同的角色，且必须在生成的文档中共同呈现。因此，全局排序常常会产生由部分来源主导的不平衡上下文，导致企业草稿不完整。为解决这一局限性，我们提出了W-RAG，一种源感知检索框架，该框架执行本体引导的检索、在每个知识库内进行局部排序，以及源级加权。

    arXiv:2608.22081v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) enables large language models to incorporate external knowledge during generation, improving factual grounding and domain adaptability. However, existing RAG pipelines assume that evidence retrieved from multiple repositories can be ranked globally using a single similarity function. While suitable for open-domain retrieval, this assumption breaks down in enterprise document generation, where heterogeneous knowledge bases (such as policies, regulations, technical documentation, and departmental guidelines) serve distinct roles and must be jointly represented in the generated document. As a result, global ranking often produces unbalanced context dominated by a subset of sources, leading to incomplete enterprise drafts. To address this limitation, we propose W-RAG, a source-aware retrieval framework that performs ontology-guided retrieval, local ranking within each knowledge base, and source-level we
    
[^149]: 多智能体计算机使用中的脊柱-分支协调机制

    Spine-Branch Coordination for Multi-agent Computer Use

    [https://arxiv.org/abs/2608.22077](https://arxiv.org/abs/2608.22077)

    本文提出脊柱-分支协调框架，通过将任务分解为连续状态的脊柱和并行收集信息的分支，避免虚拟机状态合并，在提升多智能体计算机使用任务成功率的同时大幅降低成本。

    

    计算机使用智能体（CUA）越来越多地被部署为多智能体系统，将任务分解为多个子任务，并在并行的虚拟机（VM）上执行。然而，一个关键的物理瓶颈是两台虚拟机的状态无法合并。以往的系统以临时方式处理这一问题，而非将其视为首要关注点。我们提出了多智能体计算机使用中的脊柱-分支协调框架，该框架将任务分解为一个“脊柱-分支”图，其中脊柱承载主要任务流程，并保持连续的虚拟机状态，而分支任务并行执行，以收集脊柱完成任务所需的信息。分支虚拟机在其任务完成后即被丢弃，因此永远不会发生虚拟机合并。实验表明，在来自Odysseys的200个长时任务及三种CUA骨干网络上，脊柱-分支方法相比基线系统将成功率提高了6.0%至16.5%，同时将每任务成本降低了34%至70%，这表明该方法具有显著优势。

    arXiv:2608.22077v1 Announce Type: new  Abstract: Computer use agents (CUAs) are increasingly deployed as multi-agent systems that decompose a task into multiple subtasks executed across parallel virtual machines (VMs). However, a critical physical bottleneck is that the state of two VMs cannot be merged. Previous systems handle this ad-hoc rather than treating it as a first-class concern. We propose Spine-Branch Coordination for multi-agent computer use, a framework that decomposes a task into a "spine-branch" graph, where the spine carries the main task flow with continuous VM state and branch tasks execute in parallel to collect information the spine needs to complete the task. Branch VMs are discarded once their tasks finish, so no VM merging ever occurs. Experiments show that on 200 long-horizon tasks from Odysseys and across three CUA backbones, Spine-Branch improves success rate over the baseline system by 6.0% to 16.5%, while reducing per-task cost by 34% to 70%, indicating that
    
[^150]: 真实特克：用于轮流发言预测的多模态土耳其语语料库

    Real-TurnTurk: A Multimodal Turkish Corpus for Turn-Taking Prediction

    [https://arxiv.org/abs/2608.22071](https://arxiv.org/abs/2608.22071)

    本文首次构建了多模态土耳其语自然对话语料库，并利用遗传算法优化可解释规则来预测轮流发言。

    

    摘要：arXiv:2608.22071v1 公告类型：交叉 摘要：轮流发言是人类对话的基本组织特征，在自然、同步的对话系统中仍然难以建模。虽然现有研究已探索了多模态方法和大型语言模型用于发言结束预测，但缺乏专门针对土耳其语轮流发言动态的自然对话语料库。本研究引入了一个多模态土耳其语对话数据集，包含无脚本的双人互动，包括同步的前置摄像头视频、允许重叠语音归属到单个说话者的每说话者音频通道，以及时间对齐的转录。轮流发言预测被表述为一个二元分类问题，并采用遗传算法（GA）来优化从视觉、声学和语言特征中推导出的可解释决策规则。在所提出的框架中采用了一种混合AND-OR规则表示来表示交替。

    arXiv:2608.22071v1 Announce Type: cross  Abstract: Turn-taking is a basic organizational feature of human conversation and remains difficult to model in natural, synchronous dialog systems. While existing research has explored multimodal approaches and large language models for turn-ending prediction, there is a lack of naturalistic conversational corpora specifically addressing turn-taking dynamics in Turkish. This study introduces a multimodal Turkish conversational dataset of unscripted dyadic interactions, comprising synchronized front-facing video, per-speaker audio channels that allow overlapping speech to be attributed to individual speakers, and time-aligned transcriptions. Turn-taking prediction is formulated as a binary classification problem, and a Genetic Algorithm (GA) is employed to optimize interpretable decision rules derived from visual, acoustic, and linguistic features. A hybrid AND-OR rule representation is adopted in the proposed framework to represent the alternat
    
[^151]: 对齐、统一、抑制、路由：变压器计算的连贯主义视角

    Align, Unify, Suppress, Route: A Coherentist View of Transformer Computation

    [https://arxiv.org/abs/2608.22034](https://arxiv.org/abs/2608.22034)

    本文提出连贯主义概率组合论（CPC），通过四个操作符角色（对齐、统一、抑制、路由）统一描述变压器计算，并在多个模型中验证了其有效性，特别指出抑制角色在跨任务中表现更稳定。

    

    机制可解释性已识别出变压器电路，但缺乏一种共享词汇来描述其功能如何在任务和架构间组合。我们引入了连贯主义概率组合论（CPC），这是一种基于连贯主义解释理论的基础框架，用于描述变压器计算，并通过四个操作符角色进行阐述。对齐识别候选关系，统一整合支持信息，抑制减少不兼容的替代方案，路由将选定信息传递至输出。在来自五个架构家族的15个模型中，抑制、统一和路由的权重空间特征与保留的激活级角色测量值相关性高于随机基线。抑制在跨任务中比统一更稳定。在10个模型中，消除对齐头部可降低下游抑制活动，超过随机头部控制，但类似效应...

    arXiv:2608.22034v1 Announce Type: new  Abstract: Mechanistic interpretability has identified transformer circuits, but lacks a shared vocabulary for describing how their functions compose across tasks and architectures. We introduce Coherentist Probabilistic Compositionalism (CPC), an interpretive framework that grounds transformer computation in coherentist theories of interpretation and describes it through four operator roles. Alignment identifies candidate relations, unification integrates supporting information, suppression reduces incompatible alternatives, and routing carries selected information to the output. Across 15 models from five architecture families, the suppression, unification, and routing weight-space signatures correlate with held-out activation-level role measures above random baselines. Suppression is more stable across tasks than unification. Ablating alignment heads reduces downstream suppressive activity beyond a random-head control in 10 models, but similar e
    
[^152]: 变压器的通信图谱

    The Communication Map of a Transformer

    [https://arxiv.org/abs/2608.22007](https://arxiv.org/abs/2608.22007)

    提出了一种从权重出发绘制变压器所有潜在通信通道的“通信图谱”方法，能高效计算并揭示大多数注意力头对的耦合或回避模式，且具有广泛适用性。

    

    arXiv:2608.22007v1 公告类型：交叉 摘要：变压器的组件通过写入和读取共享残差流进行通信，机制可解释性已通过手工逐个电路绘制了这些连接。我们提出了通信图谱，它仅从权重出发，绘制了语言模型中所有潜在通信通道，将Elhage等人（2021）的组成分数推广为覆盖所有18类连接（从整个注意力头电路到单个神经元）的单一耦合系数。对所有候选通道的普查，从GPT-2中的$6.3\times10^{8}$到Pythia-6.9B中的$1.3\times10^{11}$，发现70-89%的头对方向偏离随机水平，有些强耦合，另一些则主动避免彼此。完整图谱在单个消费级GPU上计算GPT-2需15秒，Pythia-6.9B需11分钟。两个应用展示了该图谱的实用性。在应用1中，最强的头对头耦合恢复了

    arXiv:2608.22007v1 Announce Type: cross  Abstract: The components of a transformer communicate by writing to and reading from a shared residual stream, and mechanistic interpretability has mapped these connections by hand, one circuit at a time. We present the communication map, which charts every potential communication channel in a language model from weights alone, generalizing the composition score of Elhage et al. (2021) into a single coupling coefficient covering all 18 connection classes, from entire attention head circuits to single neurons. The census of all candidate channels, from $6.3\times10^{8}$ in GPT-2 to $1.3\times10^{11}$ in Pythia-6.9B, finds that 70-89% of head pairs are oriented far from chance, some coupled strongly and others actively avoiding each other. The full map costs 15 seconds for GPT-2 and 11 minutes for Pythia-6.9B on one consumer GPU. Two applications demonstrate the utility of the map. In Application 1, the strongest head-to-head couplings recover the
    
[^153]: 机器学习与数字语用学：哪种词类对表情符号使用影响最大？

    Machine learning and digital pragmatics: Which word category influences emoji use most?

    [https://arxiv.org/abs/2608.21975](https://arxiv.org/abs/2608.21975)

    本研究通过MARBERT模型和逻辑回归分析发现，在口语阿拉伯语社交媒体帖子中，动词类别对表情符号使用的影响最强，尽管名词在频率上占主导。

    

    arXiv:2608.21975v1 公告类型：新 摘要：本研究考察了最先进的MARBERT模型在数字语用学方法（DPA）框架内识别X平台上表情符号使用相关词汇/语用类别的表现。使用Python从X收集了包含表情符号的15856条口语阿拉伯语（CA）帖子作为净语料库。文本被分词并规范化为4个词汇类别，即名词_规范、动词_规范、形容词_规范和副词_规范，以及2个语用/结构类别，即疑问_规范和感叹_规范。MARBERT经过微调和优化，以识别哪个类别在标准指标上得分更高，从而与表情符号使用相关，同时使用二元逻辑回归来检验哪个类别在统计上与表情符号出现相关。研究结果显示，名词在规范频率上主导语料库（平均值=0.675，标准差=0.161），其次是动词（平均值=0.083，标准差=0.100）。然而，动词对表情符号使用的影响最强。

    arXiv:2608.21975v1 Announce Type: new  Abstract: This study examines the performance of the state-of-the-art MARBERT model in identifying the lexical/pragmatic category associated with emoji use on X within a digital pragmatics approach (DPA). A net corpus of 15856 Colloquial Arabic (CA) posts containing emojis was collected from X using Python. The texts were tokenized and normalized into 4 lexical categories, namely noun_norm, verb_norm, adj_norm, and adverb_norm, and 2 pragmatic/structural categories, question_norm and exclamation_norm. MARBERT was finetuned and optimized to identify which category scores standard metrics more, hence associated with emoji use, while binary logistic regression was used to examine which category is statistically associated with emoji occurrence. Findings unveil that nouns dominate the corpus in normalized frequency (M = 0.675, SD = 0.161), followed by verbs (M = 0.083, SD = 0.100). However, verbs have the strongest influence of emoji use indicated by 
    
[^154]: ToSCA：基于对话代理时间与策略抽象的层次强化学习

    ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents

    [https://arxiv.org/abs/2608.21969](https://arxiv.org/abs/2608.21969)

    本文提出一种两级层次强化学习框架，结合话语级策略抽象与词元级解码，并引入双粒度奖励机制，以提升对话代理在复杂交互中的性能。

    

    人类在日常互动和思考中具有多个层次的时间抽象能力，例如概念感知和策略规划。受此启发，我们为对话代理提出了一种两级层次强化学习（RL）框架，弥合了以往基于词元级别或话语级别RL方法之间的差距。该框架基于两级MDP开发，其中词元级别的响应解码依赖于话语级别的动作，即显式文本策略。基于理论推导和效率考虑，我们使用DQN求解高层评论家，使用PPO求解低层演员-评论家。为进一步缓解奖励稀疏性并促进收敛，我们还设计了双粒度奖励机制，将话语级别的满意度评分与词元级别的内在动机和K-L惩罚相结合。在日常对话和情感支持对话上的实验表明，所提方法优于现有基线。

    arXiv:2608.21969v1 Announce Type: new  Abstract: Humans have multiple levels of temporal abstractions on daily interaction and thinking, such as concept perception and strategic planning. Inspired by this nature, we propose a two-level hierarchical reinforcement learning (RL) framework for conversational agents, bridging the gap between previous token-level or utterance-level RL methods. Developed on a two-level MDP, the token-level response decoding is conditioned on the utterance-level action, the explicit textual strategies. Based on theoretical derivation and efficiency consideration, we use DQN to solve the high-level critic and PPO to solve the low-level actor-critic. To further alleviate the reward sparsity and facilitate the convergence, we also design the dual-granularity reward mechanism, in which the utterance-level satisfaction score is integrated with token-level intrinsic motivation and K-L penalty. Experiments on both daily and emotional support conversations show that o
    
[^155]: 布布勒：用于阿拉伯语方言语音识别的数据集

    Bulbul: A Dataset for Dialectal Arabic Speech Recognition

    [https://arxiv.org/abs/2608.21950](https://arxiv.org/abs/2608.21950)

    本文介绍了BULBUL，一个覆盖11国275名说话者的多方言阿拉伯语ASR数据集，通过结构化方言覆盖和两级人工验证确保质量，并为现代方言ASR建立了强基线。

    

    阿拉伯语自动语音识别（ASR）因双言现象、广泛的地区方言差异以及有限的语音资源而面临独特挑战。现有语音数据集往往集中于单一方言或大规模的广播/网络数据，导致语言多样性与标注质量之间的权衡。我们提出了BULBUL，一个从11个阿拉伯国家的275名说话者收集的多方言阿拉伯语ASR数据集。BULBUL包含结构化的方言和次方言覆盖，以及参与者以其本地方言口音录制的古典阿拉伯语和现代标准阿拉伯语录音，以支持口音感知建模。录音质量通过两级人工验证过程得到保证。我们进一步对一系列最新的ASR系统进行了基准测试，为现代方言和带口音的阿拉伯语ASR建立了强基线。

    arXiv:2608.21950v1 Announce Type: cross  Abstract: Arabic automatic speech recognition (ASR) faces unique challenges due to diglossia, extensive regional dialect variation, and limited speech resources. Existing speech datasets often focus on single dialects or large-scale broadcast/web data, leading to trade-offs between linguistic diversity and annotation quality. We present BULBUL, a multi-dialect Arabic ASR dataset collected from 275 speakers in 11 Arab countries. BULBUL includes structured dialect and sub-dialect coverage, as well as recordings of classical Arabic and modern standard Arabic spoken by participants in their native dialectal accents to support accent-aware modeling. The quality of the recordings was ensured through a two-level human verification process. We further benchmark a range of recent ASR systems, establishing strong baselines for modern dialectal and accented Arabic ASR.
    
[^156]: EDGE：智能体强化学习中引导探索的经验蒸馏方法

    EDGE: Experience-Distillation for Guided Exploration in Agentic Reinforcement Learning

    [https://arxiv.org/abs/2608.21946](https://arxiv.org/abs/2608.21946)

    EDGE框架通过将检索到的经验作为临时训练支架并逐步蒸馏到策略参数中，实现无需额外采样即可持续提升智能体强化学习性能。

    

    基于结果目标（如GRPO）的强化学习使基于LLM的智能体能够解决复杂、长期任务，但嵌入在交互轨迹中的可复用探索模式在单次策略更新后大多被丢弃。现有的经验增强方法在推理时检索历史指导，但它们在应用经验时未考虑策略不断演进的能力，并对外部检索产生持久依赖。我们提出EDGE（经验蒸馏引导探索）框架，该框架将检索到的经验视为临时训练时的支架，并逐步将其益处内化到参数化策略中。具体来说，EDGE将每个rollout组划分为经验条件轨迹和无经验轨迹，以估计并仅接受正边际增益，无需额外采样，然后将诱导行为蒸馏到基础策略中。

    arXiv:2608.21946v1 Announce Type: cross  Abstract: Reinforcement learning with outcome-based objectives such as GRPO enables LLM-based agents to solve complex, long-horizon tasks, yet the reusable exploration patterns embedded in interaction trajectories are largely discarded after a single policy update. Existing experience-augmented approaches retrieve historical guidance at inference time, but they apply experiences without accounting for the policy's evolving capability and create persistent dependencies on external retrieval. We propose EDGE (Experience-Distillation for Guided Exploration), a framework that treats retrieved experiences as temporary training-time scaffolds and progressively internalizes their benefits into the parametric policy. Concretely, EDGE partitions each rollout group into experience-conditioned and experience-free trajectories to estimate and admit only positive marginal gains without extra sampling, then distills the induced behavior into the base policy v
    
[^157]: 技能膨胀：LLM编码代理中通过技能注入实现的令牌放大攻击

    SkillBloat: Token Amplification Attacks via Skill Injection in LLM Coding Agents

    [https://arxiv.org/abs/2608.21929](https://arxiv.org/abs/2608.21929)

    本文提出SkillBloat框架，通过恶意技能注入使LLM编码代理消耗远超所需的令牌，实现经济资源滥用攻击，在真实基准上达到5.42至10.15倍的令牌放大。

    

    arXiv:2608.21929v1 公告类型：交叉 摘要：代理技能通过任务特定指令、脚本和资源扩展了编码代理的功能，但它们也创建了一个可信的指令通道，该通道可能被滥用，超出传统安全攻击的范围。本文研究了通过技能注入实现的令牌放大：一种经济资源滥用威胁，其中恶意技能导致代理在正常任务执行中消耗远超所需的令牌。我们提出了SkillBloat，一个两阶段框架，首先在多种放大机制中筛选出多样化攻击类型的条件库，然后通过LLM引导的全文档技能重写来优化最强候选。在真实世界技能基准上的评估显示，SkillBloat在多种编码代理目标配置下实现了5.4184倍至10.1455倍的平均最佳放大。消融研究表明，第二阶段的重写循环始终优于第一阶段的平均最佳放大效果。

    arXiv:2608.21929v1 Announce Type: cross  Abstract: Agent skills extend coding agents with task-specific instructions, scripts, and resources, but they also create a trusted   instruction channel that can be abused beyond conventional security attacks. This paper studies token amplification through   skill injection: an economic resource-abuse threat in which a malicious skill causes an agent to consume substantially more   tokens than needed for normal task execution. We present SkillBloat, a two-phase framework that first screens a library of   diverse attack-type conditions across multiple amplification mechanisms and then refines the strongest candidate through   LLM-guided full-document skill rewriting. Evaluated on a real-world skill benchmark, SkillBloat achieves 5.4184x-10.1455x   average best amplification across multiple coding-agent target configurations. An ablation shows that the second-stage   refinement loop consistently improves average best amplification over Phase 1 at
    
[^158]: GuardianBench：面向具身AI潜在情境风险的同场景指令对比基准

    GuardianBench: A Same-Scene Instruction-Contrastive Benchmark for Latent Contextual Risk in Embodied AI

    [https://arxiv.org/abs/2608.21928](https://arxiv.org/abs/2608.21928)

    本文提出GuardianBench，一个基于国际安全标准的同场景指令对比基准，通过3,024个示例揭示视觉语言模型在具身AI中因无法绑定指令相关信息而导致潜在情境风险判定失败，平均配对准确率仅24.1%。

    

    在具身AI中，安全风险可能是潜在的：一个良性的指令和一个安全的场景只有在组合时才变得危险。先前的工作通过改变视觉情境或评估执行时的动态来推进具身安全，但固定场景并仅改变指令这一互补维度仍未得到充分探索。我们引入了GuardianBench，一个基于国际安全标准的指令对比基准，通过3,024个指令-场景示例，组织为跨多种危险类别的同场景安全/不安全对比对，来隔离这种潜在情境风险。对最先进的视觉语言模型（VLMs）进行基准测试揭示了指令不敏感的判定：模型在给定场景下不成比例地批准两个指令；在主要模型中，平均配对准确率仅为24.1%。我们的系统性理由审计定位了主要失败：模型未能绑定指令相关的信息。

    arXiv:2608.21928v1 Announce Type: new  Abstract: In embodied AI, safety risk can be latent: a benign instruction and a safe scene become hazardous only when composed. Prior work has advanced embodied safety by varying visual contexts or evaluating execution-time dynamics, but the complementary axis of fixing the scene and varying only the instruction remains underexplored. We introduce GuardianBench, an instruction-contrastive benchmark grounded in international safety standards that isolates this latent contextual risk through 3,024 instruction-scene examples organized as same-scene Safe/Unsafe contrastive pairs across various hazard categories. Benchmarking state-of-the-art vision-language models (VLMs) reveals instruction-insensitive verdicts: models disproportionately approve both instructions under a given scene; across the primary models, average pair accuracy is only 24.1%. Our systematic rationale audit localizes the dominant failure: models fail to bind the instruction-relevan
    
[^159]: 基于图注意力网络的专利诉讼预测中的权利要求依赖结构建模

    Modeling Claim Dependency Structure for Patent Litigation Prediction with Graph Attention Networks

    [https://arxiv.org/abs/2608.21924](https://arxiv.org/abs/2608.21924)

    本文提出ClaimGAT，利用图注意力网络对专利权利要求进行独立编码并建模其依赖关系，显著提升了专利诉讼预测的准确性。

    

    专利诉讼给企业带来巨大成本，并扭曲研发激励，因此早期风险识别是一项重要的实际任务。尽管先前的工作已将基于BERT的模型应用于专利权利要求文本，但仍存在两个根本性局限：扁平序列编码丢失了独立权利要求与从属权利要求之间的依赖结构，而这种结构在法律上决定了专利范围；将整个权利要求集输入单个编码器会丢弃法律上关键的信息。通过对134万件美国专利商标局实用专利进行的六模型消融实验，证实了逐权利要求编码、图连接性、注意力机制和注意力聚合各自提供了独立且可叠加的预测价值。我们提出了ClaimGAT，一种图注意力网络，它独立编码每项权利要求，构建有向权利要求依赖图，使用GATConv层处理该图，并通过注意力聚合独立权利要求，从而生成诉讼风险评分和权利要求级别输出。

    arXiv:2608.21924v1 Announce Type: new  Abstract: Patent litigation imposes substantial costs on firms and distorts R&D incentives, making early risk identification a practically important task. While prior work has applied BERT-based models to patent claim text, two fundamental limitations remain: flat sequence encoding loses the dependency structure between independent and dependent claims that legally determines patent scope, and feeding the entire claim set to a single encoder discards legally critical text. A six-model ablation on 1.34 million USPTO utility patents confirms that per-claim encoding, graph connectivity, attention, and Attentional Aggregation each provide independent, additive predictive value. We propose ClaimGAT, a Graph Attention Network that encodes each claim independently, constructs a directed claim dependency graph, processes it with GATConv layers, and aggregates independent claims via Attentional Aggregation to yield both a litigation risk score and claim-le
    
[^160]: BanglaVeilGuard：孟加拉语大语言模型的跨文字安全基准测试与轻量级防护栏

    BanglaVeilGuard: Cross-Script Safety Benchmarking and Lightweight Guardrails for Bangla Large Language Models

    [https://arxiv.org/abs/2608.21880](https://arxiv.org/abs/2608.21880)

    本文提出BanglaVeilGuard，一种针对六种孟加拉语变体的轻量级防护栏，通过非破坏性归一化和风险分类门控，在无需修改模型权重的情况下将攻击成功率从93.8%降至100%，显著提升孟加拉语大语言模型的安全性。

    

    孟加拉语大语言模型（LLM）的安全性难以用以英语为中心或标准文字基准进行评估，因为孟加拉语用户经常使用多种文字、拼写、混合编码形式和地区性语域进行书写。本文介绍了BanglaVeilGuard，一个紧凑的以孟加拉语优先的安全基准测试和轻量级提示防护栏，涵盖六种语言形式：标准孟加拉语、罗马化孟加拉语、Banglish、孟加拉语-英语混合编码、噪声孟加拉语和方言孟加拉语。该基准包含2,366个质量过滤的提示和一个保留的354个提示评估子集，涵盖不安全、安全和安全敏感请求。BanglaVeilGuard采用非破坏性多视图归一化，结合提示风险分类器和阈值化生成前门控，使其能够为异构目标模型筛选提示，而无需更改其权重。在目标模型系列中，受防护的运行在确定性响应评分下将攻击成功率从93.8%降至100%。

    arXiv:2608.21880v1 Announce Type: new  Abstract: Bangla large language model (LLM) safety is difficult to evaluate with English-centric or standard-script benchmarks because Bangla users routinely write across scripts, spellings, code-mixed forms, and regional registers. This paper presents BanglaVeilGuard, a compact Bangla-first safety benchmark and lightweight prompt guard for six language forms: standard Bangla, Romanized Bangla, Banglish, code-mixed Bangla--English, noisy Bangla, and dialectal Bangla. The benchmark contains 2,366 quality-filtered prompts and a held-out 354-prompt evaluation split spanning unsafe, safe, and safe-sensitive requests. BanglaVeilGuard uses non-destructive multi-view normalization with a prompt-risk classifier and thresholded pre-generation gate, allowing it to screen prompts for heterogeneous target models without changing their weights. Across target-model families, guarded runs reduce attack success under deterministic response scoring from 93.8--100.
    
[^161]: 追逐即课程，捕获锚定奖励：零数据大语言模型推理的追逃自我对弈

    The Chase Is the Curriculum, the Capture Anchors the Credit: Pursuit-Evasion Self-Play for Zero-Data LLM Reasoning

    [https://arxiv.org/abs/2608.21871](https://arxiv.org/abs/2608.21871)

    本文提出LURE框架，将零数据自我对弈建模为追逃博弈，通过捕获前沿奖励学习任务定位策略，使LLM推理训练无需人工数据即可自适应难度。

    

    摘要：arXiv:2608.21871v1 公告类型：新 摘要：基于可验证奖励的强化学习已成为提升大语言模型推理能力的主导方法，但它依赖于大规模人工策划的任务集。零数据自我对弈消除了这种依赖，但现有方法仅通过探测候选任务并事后拒绝来验证可学习性，从未学习在环境的难度轴上何处放置任务，且仅以稀疏的最终奖励来评价求解者。我们将零数据自我对弈重新定义为追逃博弈：在LURE中，一个LLM逃逸者将任务沿每个环境的难度轴定位，以保持领先于一个通过可验证交互追捕它的规划-执行追捕者一步。逃逸者基于捕获前沿奖励进行训练，该奖励在求解者恰好在其一半的轨迹中捕获它时达到峰值，将“勉强可捕获”转变为一种学习到的定位策略，而非手工调优的拒绝带。追捕者获得捕获锚定的奖励。

    arXiv:2608.21871v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards has become the dominant recipe for improving large language model reasoning, yet it presumes large human-curated task collections. Zero-data self-play removes this dependency, but existing methods vet learnability only by probing candidates and rejecting post hoc, never learning where along an environment's difficulty axis to place a task, and credit the solver with sparse terminal rewards alone. We recast zero-data self-play as a pursuit-evasion game: in LURE, an LLM evader positions tasks along each environment's difficulty axis to stay one step ahead of a planner-executor pursuer that hunts it down through verifiable interaction. The evader is trained on a capture-frontier reward that peaks when the solver captures it on exactly half of its rollouts, turning barely catchable into a learned positioning strategy rather than a hand-tuned rejection band. The pursuer earns capture-anchored den
    
[^162]: MemGuard：为LLM智能体记忆治理持久化验证器信号

    MemGuard: Persisting Verifier Signals for LLM-Agent Memory Governance

    [https://arxiv.org/abs/2608.21867](https://arxiv.org/abs/2608.21867)

    MemGuard通过将验证器输出转化为持久化生命周期元数据，解决了LLM智能体记忆中的不可靠准入和记忆漂移问题，确保记忆在长期交互中保持可靠。

    

    arXiv:2608.21867v1 公告类型：新 摘要：LLM智能体正从单次提示使用转向长任务流，其中可复用记忆成为终端、软件工程和网络任务的核心能力。这种记忆只有在存储的经验在数百次交互中保持可靠时才有效，但两种失败模式在实践中打破了这一假设。第一种是不可靠的准入：失败轨迹、偶然成功和误导性观察因看似相关而进入记忆，随后误导后续决策。第二种是记忆漂移：长期运行的记忆库积累重复、过时和冲突的记录，仅靠检索无法修复。MemGuard的关键区别在于将验证器输出视为持久化的生命周期元数据，而非一次性过滤器。它将多标准评分令牌验证转换为奖励、置信度、标签和不确定性描述符，这些描述符在激活前附加到每个候选上，并在检索过程中复用。

    arXiv:2608.21867v1 Announce Type: new  Abstract: LLM agents are moving from single-prompt use to long task streams in which reusable memory becomes a core capability for terminal, software-engineering, and web tasks. Such memory is useful only when stored experience remains reliable across hundreds of interactions, but two failure modes break that assumption in practice. The first is unreliable admission: failed trajectories,accidental successes, and misleading observations enter memory because they appear relevant, then mislead later decisions. The second is memory drift: long-running banks accumulate duplicate, stale, and conflicting records that retrieval alone cannot repair. MemGuard's key distinction is to treat verifier output not as a one-shot filter, but as persistent lifecycle metadata. It converts multi-criteria score-token verification into reward, confidence, label, and uncertainty descriptors that are attached to every candidate before activation and reused during retrieva
    
[^163]: HiDiffTIR：面向多轮工具集成推理的分层难度感知策略优化

    HiDiffTIR: Hierarchical Difficulty-Aware Policy Optimization for Multi-Turn Tool-Integrated Reasoning

    [https://arxiv.org/abs/2608.21863](https://arxiv.org/abs/2608.21863)

    本文提出HiDiffTIR框架，通过分层难度感知的信用分配机制，在多轮工具集成推理中更精确地区分轨迹和推理步骤的难度，从而提升强化学习训练效果。

    

    arXiv:2608.21863v1 公告类型：交叉 摘要：工具集成推理（TIR）是LLM代理通过与外部工具迭代交互解决复杂任务的基本能力。强化学习（RL）已成为实现这一能力的主导范式。然而，现有方法通常分配统一的轨迹级优势，并平等对待所有正确的工具调用，忽略了轨迹和推理步骤间不同的难度和学习价值。这可能导致学习信号不精确，无法充分区分平凡和具有挑战性的工具使用模式。为解决这一局限性，我们提出了HiDiffTIR，一种用于多轮TIR的分层难度感知策略优化框架。HiDiffTIR在轨迹级和回合级执行难度感知的信用分配，使策略能够聚焦于更具信息量的轨迹和更难的推理步骤。值得注意的是，这种细粒度优化是通过...

    arXiv:2608.21863v1 Announce Type: cross  Abstract: Tool-Integrated Reasoning (TIR) is a fundamental capability for LLM agents to solve complex tasks by interacting with external tools iteratively. Reinforcement Learning (RL) has become the dominant paradigm for enabling this capability. However, existing approaches typically assign uniform trajectory-level advantages and treat all correct tool calls equally, ignoring the varying difficulty and learning value across trajectories and reasoning steps. This can lead to imprecise learning signals that do not adequately distinguish between trivial and challenging tool-use patterns. To address this limitation, we propose HiDiffTIR, a Hierarchical Difficulty-aware policy optimization framework for multi-turn TIR. HiDiffTIR performs difficulty-aware credit assignment at both trajectory and turn levels, enabling the policy to focus on more informative trajectories and harder reasoning steps. Notably, this fine-grained optimization is achieved wi
    
[^164]: PUMA：一个面向文化情境多模态理解的波兰语基准

    PUMA: A Polish Benchmark for Culturally Grounded Multimodal Understanding

    [https://arxiv.org/abs/2608.21853](https://arxiv.org/abs/2608.21853)

    本文提出了一个名为PUMA的波兰语多模态基准，包含900个手工任务，评估模型在波兰文化语境下的图像、音频和文本理解能力，并揭示了当前模型在该领域存在的显著性能差距。

    

    arXiv:2608.21853v1 公告类型：新论文 摘要：大型语言模型正日益超越文本处理，增加了对图像和音频等其他模态的支持。尽管文本理解和生成已被广泛研究，但多模态数据处理能力，尤其是在非英语文化和语言背景下，尚未得到全面评估。在本文中，我们提出了PUMA（波兰统一多模态评估），这是一个包含900个手工设计任务的新型基准，旨在探索多模态模型在波兰文化和语言背景下的极限。该数据集评估了文化理解以及处理文本、图像、音频和视觉丰富文档的实际技能。我们对前沿商业模型、开放权重模型和专用小型系统的广泛评估突显了显著的性能差距。虽然顶级商业模型在视觉问答中取得了高分，但大多数模型仍存在明显不足。

    arXiv:2608.21853v1 Announce Type: new  Abstract: Large language models are increasingly moving beyond text processing, adding support for other modalities such as images and audio. While text understanding and generation have been extensively studied, multimodal data processing capabilities, particularly in the context of cultures and languages other than English, have not yet been evaluated comprehensively. In this paper, we propose PUMA (Polish Unified Multimodal Assessment), a novel benchmark of 900 hand-crafted tasks designed to probe the limits of multimodal models in the Polish cultural and linguistic context. The dataset evaluates both cultural understanding and practical skill in processing text, images, audio, and visually rich documents. Our extensive evaluation of frontier commercial models, open-weights models, and specialized smaller systems highlights a significant performance gap. While top commercial models achieve high scores in visual question answering, most models s
    
[^165]: GameXpert-Bench：编程代理距离专家级游戏开发还有多远？

    GameXpert-Bench: How Far Are Coding Agents from Expert Game Development?

    [https://arxiv.org/abs/2608.21833](https://arxiv.org/abs/2608.21833)

    本文提出了GameXpert-Bench基准，通过三个互补轨道（初始生成、缺陷修复、多轮优化）全面评估编程代理在游戏开发全生命周期中的能力。

    

    arXiv:2608.21833v1 公告类型：新 摘要：近期的大型语言模型（LLMs）可以作为编程代理，从自然语言请求中构建完整的游戏。游戏开发尤其具有挑战性，因为程序逻辑、视觉和音频内容、界面、交互和可玩性必须在一个可执行制品中协同工作。因此，衡量这一能力需要对游戏产品和开发过程进行评估。现有基准通常通过评估最终制品或孤立的开发阶段来评估LLM的游戏开发能力。我们对完整的人机代理开发轨迹的分析识别出三个阶段，这些阶段共同涵盖了使用编程代理进行游戏开发的生命周期：初始游戏生成、缺陷诊断与修复，以及多轮优化。因此，我们引入了GameXpert-Bench，它将这三个生命周期阶段作为三个互补的基准轨道进行具体化。GameGen评估c

    arXiv:2608.21833v1 Announce Type: new  Abstract: Recent large language models (LLMs) can operate as coding agents that build complete games from natural language requests. Game development is especially demanding because program logic, visual and audio content, interfaces, interaction and playability must function together in one executable artifact. Measuring this capability therefore requires evaluation of both game product and the development process. Existing benchmarks often assess the game development capabilities of LLMs by evaluating the final artifact or an isolated development stage. Our analysis of complete human-agent development trajectories identifies three stages that together span the lifecycle of game development with a coding agent: initial game generation, bug diagnosis and repair, and optimization over multiple turns. Therefore, we introduce GameXpert-Bench, which operationalizes the three lifecycle stages as three complementary benchmark tracks. GameGen evaluates c
    
[^166]: GUI-Primitives：诊断视觉-语言GUI定位中的空间推理失败

    GUI-Primitives: Diagnosing Spatial Reasoning Failures in Vision-Language GUI Grounding

    [https://arxiv.org/abs/2608.21832](https://arxiv.org/abs/2608.21832)

    该论文提出了一个名为GUI-Primitives的基准测试，通过对比指令对系统性地诊断视觉-语言模型在GUI空间关系推理中的缺陷，发现现有模型在严格定位准确率上表现极差，且多数预测完全偏离候选区域。

    

    arXiv:2608.21832v1 公告类型：新 摘要：计算机使用代理将自然语言指令与截图中的界面元素进行定位，但现有基准测试并未隔离模型是否将关系语言绑定到正确的元素上。我们引入了GUI-Primitives，一个包含994个对比指令对的基准测试，覆盖图形用户界面中的七种空间关系（左/右、上/下、包含、对齐、邻近、列表序数、遮挡）。每个对比对在保持截图和锚点不变的情况下，仅改变关系表达式，从而使得正确目标在两个指定候选之间移动。五位标注者验证了一个包含196项的子集（结构良好性κ=0.94；目标选择κ=0.79）。十九个视觉-语言模型在严格的点内框准确率上最多达到32%。由于模型输出不受约束的坐标，我们根据每个预测落入的候选区域对其进行分类。在60-92%的情况下，预测落在两个候选之外。

    arXiv:2608.21832v1 Announce Type: new  Abstract: Computer-use agents ground natural-language instructions in screenshots to locate interface elements, yet existing benchmarks do not isolate whether models bind relational language to the correct element. We introduce GUI-Primitives, a 994-item benchmark of contrastive instruction pairs over seven spatial relations in graphical user interfaces (left/right, above/below, containment, alignment, proximity, list ordinal, occlusion). Each pair holds the screenshot and anchor fixed while changing the relation expression, so the correct target moves between two designated candidates. Five annotators validate a 196-item subset ($\kappa = 0.94$ well-formedness; $\kappa = 0.79$ target selection). Nineteen vision-language models reach at most $32\%$ strict point-in-box accuracy. Because models emit unconstrained coordinates, we classify each prediction by the candidate region it falls within. Predictions fall outside both candidates on $60-92\%$ of
    
[^167]: 训练知识库：面向代理策展文档库的监督式结构学习

    Training a Knowledge Base: Supervised Structure Learning for Agent-Curated Document Stores

    [https://arxiv.org/abs/2608.21829](https://arxiv.org/abs/2608.21829)

    本文提出将知识库视为可训练模型，通过监督式（问题，答案）标签指导代理编辑文档库结构，实现更高效、更准确的检索增强生成。

    

    摘要：检索增强生成将文档库视为冻结输入，而那些允许代理策展文档库的系统从未衡量策展对库本身的影响。我们反转这一框架：知识库即模型。一个训练代理基于当前库回答监督式问题，查看标准答案，然后编辑库；一个未改变的读者在固定动作预算下，对冻结快照进行后续检查。离线图构建是无监督的，而（问题，答案）对是我们的标签——这种监督使得结构构建成本低廉。每索引一个语料点，它返回的动作节省是无监督实体索引（覆盖所有内容）的1.6倍，准确率是其1.8倍，使用1,913条链接对比其196,112条。在库训练过的问题上，未改变的读者以更高准确率节省31%的动作，该结果在官方PhantomWiki生成（其问题我们未训练）上复现。

    arXiv:2608.21829v1 Announce Type: cross  Abstract: Retrieval-augmented generation treats the document store as a frozen input, and the systems that instead let an agent curate one never measure what curation does to the store. We invert the framing: the knowledge base is the model. A training agent answers a supervised question against the current store, is shown the gold, then edits the store; an unchanged reader is later examined on a frozen snapshot under a fixed action budget. Where offline graph construction is unsupervised, (question, answer) pairs are our labels -- and that supervision is what makes the structure cheap. Per point of corpus indexed it returns 1.6x the action saving and 1.8x the accuracy of an unsupervised entity index covering everything, using 1,913 links against its 196,112. On questions the store trained on, an unchanged reader spends 31% fewer actions at higher accuracy, and the result reproduces on an official PhantomWiki generation whose questions we did no
    
[^168]: 大型语言模型在理解现代中文诗歌中的诗意逻辑方面表现如何？

    Do Large Language Models Perform Well on Comprehending Poetic Logic in Modern Chinese Poetry?

    [https://arxiv.org/abs/2608.21827](https://arxiv.org/abs/2608.21827)

    本文提出了首个专门评估现代中文诗歌诗意逻辑的基准Peony，并系统分析了六个主流大型语言模型在这方面的表现。

    

    大型语言模型（LLMs）在广泛自然语言处理（NLP）任务中取得了显著进展，但其理解文学作品，尤其是现代中文诗歌的能力，仍鲜有探索。现代中文诗歌独特的文学特征需要一种特殊的推理方式才能有效理解。与传达清晰信息的常规文本不同，现代中文诗歌独特的“诗意逻辑”需要一种超越表面语义分析的整体推理方法来领会。然而，当前评估范式在很大程度上忽略了这一关键维度。为解决这一空白，我们提出了Peony，这是首个专门用于评估现代中文诗歌诗意逻辑的基准。我们将诗意逻辑定义为涵盖三个层面（节、行和意象）的四项任务，并系统评估和分析了六个主流LLMs。

    arXiv:2608.21827v1 Announce Type: new  Abstract: Large Language Models (LLMs) have achieved significant progress across a wide range of natural language processing (NLP) tasks, yet their ability to understand literary texts, particularly modern Chinese poetry, remains largely unexplored. The unique literary characteristics of modern Chinese poetry necessitate a distinct form of reasoning for effective comprehension. Unlike conventional texts that convey clear information, the unique "poetic logic" of modern Chinese poetry requires a holistic reasoning approach that goes beyond superficial semantic analysis to be understood. However, current evaluation paradigms largely ignore this critical dimension. To address this gap, we propose Peony, the first benchmark specifically designed for evaluating the poetic logic of modern Chinese poetry. We define poetic logic as four tasks across three levels, namely stanza, line, and imagery, and systematically evaluate and analyze six mainstream LLMs
    
[^169]: 科学趋同，宗教趋异：维基百科各语言版本间校准框架差异

    Convergence in Science, Divergence in Religion: Calibrated Framing Differences Across Wikipedia's Language Editions

    [https://arxiv.org/abs/2608.21821](https://arxiv.org/abs/2608.21821)

    本文提出一种校准距离方法，通过减去语言对间的基线偏差，揭示了维基百科不同语言版本在科学概念上框架趋同，而在宗教概念上框架差异显著。

    

    当维基百科的不同语言版本描述同一概念时，它们的框架方式有多大差异？以往研究衡量版本间的覆盖缺口；我们则衡量匹配概念的框架距离。我们分析了来自3000个可能的概念-语言观察中的2799篇有效文章，涵盖150个以Wikidata锚定的概念、20个语言版本、4个领域以及一个校准集。原始嵌入距离既反映内容差异，也反映编码器对各语言对的对齐程度。即使在具有跨文化稳定外延的校准概念（如化学元素、数字、颜色）中，最大语言对平均距离是最小值的3.6倍，且语言家族内部的距离通常更小。我们定义了一个基线调整距离（校准距离）：即某概念的两个语言版本之间的距离减去同一语言对中校准概念的平均距离。这一调整显著减少了非内容性偏差。

    arXiv:2608.21821v1 Announce Type: new  Abstract: When Wikipedia's language editions describe the same concept, how differently do they frame it? Prior work measures coverage gaps between editions; we measure framing distance for matched concepts. We analyze 2,799 valid articles from 3,000 possible concept-language observations, spanning 150 Wikidata-anchored concepts, 20 language editions, 4 domains, and a calibration set. Raw embedding distances reflect both content differences and how well the encoder aligns each language pair. Even among calibration concepts with stable cross-cultural denotations (e.g., chemical elements, numbers, colors), the largest language-pair mean distance is 3.6 times the smallest, and distances are typically smaller within language families. We define a baseline-adjusted distance (calibrated distance): the distance between two language versions of a concept minus the mean distance for calibration concepts in the same language pair. This adjustment substantia
    
[^170]: PatchGate：通过冻结视觉语言模型中的内在对象清单缩小言语化差距

    PatchGate: Narrowing the Verbalization Gap with Intrinsic Object Inventories in Frozen Vision-Language Models

    [https://arxiv.org/abs/2608.21819](https://arxiv.org/abs/2608.21819)

    该论文提出PatchGate框架，通过从冻结视觉语言模型中提取内在对象清单，在生成前缩小可见对象与最终描述之间的差距，从而提升图像描述的精确性和完整性。

    

    arXiv:2608.21819v1 公告类型：交叉 摘要：视觉语言模型（VLM）中的可靠图像描述要求描述既精确又完整，避免提及不存在的对象，同时覆盖可见对象。现有的无训练方法主要针对前者，通过在生成过程中干预模型预测的提及来抑制不支持的对象词。由于这些方法仅针对模型可能提及的对象，输出中遗漏的可见对象难以恢复。我们提出PatchGate，一种无训练框架，它在生成前从冻结的VLM中提取与提示无关的内在对象证据，并利用该证据缩小内在对象集与最终对象提及之间的差距。在第一阶段，视觉证据提取（VEX）从LM解码器层的后半部分读取补丁级词汇证据，并在无任何任务提示的情况下构建图像条件下的对象集。在第二阶段，视觉...

    arXiv:2608.21819v1 Announce Type: cross  Abstract: Reliable image captioning in Vision-Language Models (VLMs) requires captions to be both precise and complete, avoiding unsupported object mentions while covering visible objects. Existing training-free methods primarily address the former requirement, suppressing unsupported object words by intervening on model-predicted mentions during generation. Because they operate only on objects the model is already likely to mention, visible objects omitted from the output remain difficult to recover. We propose PatchGate, a training-free framework that extracts prompt-free object evidence intrinsic to a frozen VLM before generation and uses it to narrow the gap between an intrinsic object set and final object mentions. In the first stage, Visual Evidence eXtraction (VEX) reads patch-level lexical evidence from the latter half of LM decoder layers and constructs an image-conditioned object set without any task prompt. In the second stage, Visual
    
[^171]: MCite-RL：通过引用增强的智能体强化学习实现可靠的多模态检索增强生成

    MCite-RL: Towards Reliable Multimodal RAG via Citation-enhanced Agentic Reinforcement Learning

    [https://arxiv.org/abs/2608.21808](https://arxiv.org/abs/2608.21808)

    本文提出了MCite-RL，一个通过智能体迭代检索和引用增强奖励机制，将视觉引用转化为动态证据驱动过程，从而提升多模态RAG可靠性的强化学习框架。

    

    摘要：具有视觉引用的多模态检索增强生成（RAG）对于确保多模态大语言模型（MLLMs）的可追溯性和可验证性至关重要。然而，当前的RAG和基于SFT的方法难以实现稳健的跨模态推理，导致视觉引用不精确或引用与生成答案之间脱节。为解决这些局限性，我们提出了MCite-RL，一种专为可靠多模态RAG设计的引用增强智能体强化学习框架。MCite-RL引入了一个用于视觉引用的智能体精化模块，该模块采用迭代检索、推理和递归裁剪来逐步缩小搜索空间，将引用转化为动态的、证据驱动的推理过程，而非静态步骤。此外，我们整合了一种引用增强奖励机制，在强化学习范式内结合过程级和结果级反馈，以联合优化答案生成。

    arXiv:2608.21808v1 Announce Type: new  Abstract: Multimodal Retrieval-Augmented Generation (RAG) with visual citation is crucial for ensuring the traceability and verifiability of MLLMs. However, current RAG and SFT-based methods struggle to achieve robust cross-modal reasoning, causing imprecise visual citations or decoupling between the citation and the generated answers. To address these limitations, we propose MCite-RL, a citation-enhanced agentic reinforcement learning framework designed for reliable multimodal RAG. MCite-RL introduces an Agentic Refinement module for visual citation that employs iterative retrieval, reasoning, and recursive cropping to progressively narrow the search space, transforming citation into a dynamic, evidence-driven reasoning process rather than a static step. Furthermore, we incorporate a Citation-enhanced Reward mechanism that integrates both process-level and outcome-level feedback within a reinforcement learning paradigm to jointly optimize answer 
    
[^172]: 更多计算资源并不确保更高的学术影响力：来自顶级NLP会议论文的证据

    More Computational Resources Do Not Ensure Higher Scholarly Impact: Evidence from Leading NLP Conference Papers

    [https://arxiv.org/abs/2608.21806](https://arxiv.org/abs/2608.21806)

    本研究基于13,921篇NLP顶级会议论文，发现计算资源集中度远高于学术影响力集中度，表明更多计算资源并不必然带来更高的学术影响。

    

    arXiv:2608.21806v1 公告类型：交叉 摘要：计算资源在自然语言处理研究中日益核心，但报告的GPU能力与学术影响力之间的关联程度仍不明确。我们分析了2020年至2025年间发表的13,921篇ACL、EMNLP和NAACL主会论文，使用GPU资源作为计算资源的操作化度量。从全文提取GPU型号和数量，将每篇论文报告的最大配置标准化为可比较的硬件能力度量，并将这些数据与引用、奖项、主题和机构元数据关联。GPU报告变得更加普遍但仍不完整，而报告的能力主要通过更新的硬件代际和中规模多GPU配置增加。资源集中度显著超过影响力集中度：年度前20%的GPU可量化论文占报告GPU能力的83.9%-89.9%，但仅占引用量的27%-32%和论文数的20%-33%。

    arXiv:2608.21806v1 Announce Type: cross  Abstract: Computational resources are increasingly central to NLP research, but how closely reported GPU capability aligns with scholarly impact remains unclear. We analyze 13,921 ACL, EMNLP, and NAACL main-conference papers published between 2020 and 2025, using GPU resources as our operational measure of computational resources. From full texts, we extract GPU models and counts, standardize each paper's largest reported configuration into a comparable hardware-capability measure, and link these data to citation, award, topic, and institutional metadata. GPU reporting became more common but remained incomplete, while reported capability increased mainly through newer hardware generations and medium-scale multi-GPU configurations. Resource concentration substantially exceeded impact concentration: the annual top 20% of GPU-quantifiable papers accounted for 83.9%-89.9% of reported GPU capability, but only 27%-32% of citations and 20%-33% of paper
    
[^173]: GUI元素定位中的词汇耦合：句子嵌入在移动端和网页端追踪标签

    Lexical Coupling in GUI Element Grounding: Sentence Embeddings Track Labels across Mobile and Web

    [https://arxiv.org/abs/2608.21794](https://arxiv.org/abs/2608.21794)

    该论文揭示GUI元素定位评估中，嵌入相似度常被可见标签恢复所混淆，而非真正的语义理解，并强调需引入词汇基线和标签分层来改进评估可靠性。

    

    arXiv:2608.21794v1 公告类型：交叉摘要：GUI定位评估通常将UI元素暴露为文本元数据，并往往将高指令-元素嵌入相似度视为语义定位的证据。在三个移动端和网页端基准测试中，我们表明这种解释经常被可见标签恢复所混淆。词汇基线在top-1上仍具竞争力，标签贫乏的目标对纯文本方法仍显薄弱，而编码器的top-1命中可从词汇排名、候选池大小和标签类型中预测。我们将每个操作评估为同屏排名任务，比较五种现成的单向量编码器与词汇基线。编码器能恢复一些词汇遗漏，但可部署的融合增益远小于目标感知的oracle增益。这些发现表明，基于嵌入的评估可能将可见标签恢复与语义GUI定位混为一谈。因此，基于嵌入的评估应报告词汇基线、标签类型分层和去...

    arXiv:2608.21794v1 Announce Type: cross  Abstract: GUI grounding evaluations that expose UI elements as text metadata often treat high instruction-element embedding similarity as evidence of semantic grounding. Across three mobile and web benchmarks, we show that this interpretation is frequently confounded by visible-label recovery. Lexical baselines remain competitive at top-1, label-poor targets remain weak for text-only methods, and encoder top-1 hits are predictable from lexical rank, candidate-pool size, and label type. We evaluate each action as a same-screen ranking task, comparing five off-the-shelf single-vector encoders with lexical baselines. Encoders recover some lexical misses, but deployable fusion gains are much smaller than target-aware oracle gains. These findings show that embedding-based evaluations can conflate visible-label recovery with semantic GUI grounding. Embedding-based evaluations should therefore report lexical baselines, label-type stratification, and de
    
[^174]: 没有单一模型能捕捉所有危害：跨安全场景的内容审核基准测试

    No One Model Catches Every Harm: Benchmarking Content Moderation Across Safety Scenarios

    [https://arxiv.org/abs/2608.21775](https://arxiv.org/abs/2608.21775)

    该研究通过大规模基准测试发现，没有单一模型能在所有安全场景中表现最佳，大型前沿模型和专用小型模型各有优劣，揭示了现有内容审核的盲点。

    

    大型语言模型（LLMs）越来越多地部署在现实应用中，但它们仍然容易生成有害内容。从绕过安全过滤器的对抗性越狱到逃避检测的隐性仇恨，这些模型带来的风险范围持续扩大。虽然专用内容审核器和通用LLMs都被用作安全层，但哪种模型最适合哪种类型的有害内容这一问题仍未得到解答。我们提出了迄今为止对LLM安全能力最全面的评估，系统测试了\textbf{53}个模型，覆盖\textbf{11}个数据集，并将其组织为四个不同类别。我们在仅提示和提示-响应两种设置下的评估揭示了关键的盲点：在一个类别中领先的大型前沿模型在其他类别中显著落后于较小的专用替代品，而现实世界的对话安全性仍然不足。

    arXiv:2608.21775v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly deployed in real-world applications, yet they remain vulnerable to generating harmful content. From adversarial jailbreaks that bypass safety filters to implicit hate that evades detection, the range of risks these models pose continues to grow. While both specialized content moderators and general-purpose LLMs are being used as safety layers, the question of which model is best suited for which type of harmful content remains unanswered. We present the most comprehensive evaluation of LLM safety capabilities to date, systematically testing \textbf{53} models across \textbf{11} datasets that we organize into four distinct categories. Our evaluation under both prompt-only and prompt-response settings uncovers critical blind spots: large frontier models that lead on one category fall significantly behind smaller, specialized alternatives on others, and real-world conversational safety remains l
    
[^175]: 语言模型中的评估意识：表征、言语化与控制

    Evaluation Awareness in Language Models: Representation, Verbalization, and Control

    [https://arxiv.org/abs/2608.21766](https://arxiv.org/abs/2608.21766)

    本文系统研究了语言模型中的“评估意识”现象，发现模型能在激活空间中表征被评估状态，并通过输出言语化及因果引导影响其行为。

    

    能力和安全基准测试都基于一个假设，即语言模型在测试中的行为能够反映其在部署中的行为。如果模型推断自己正在被评估，并据此调整其响应，这一假设可能失效。这种被称为“评估意识”的现象已在尖端和开放权重语言模型中被观察到。我们通过跨六个语言模型（涵盖四个家族和三种规模）及三种度量指标进行探测，对此现象进行了系统性研究。具体而言，我们考察了（i）被评估状态是否在模型激活空间中线性表征，（ii）是否在其输出标记中言语化（由LLM作为评判者评分），以及（iii）引导是否因果性地影响其行为。对于开放检查点的Olmo模型，我们进一步在每个训练阶段测试了这些度量。通过这一过程，我们报告了评估意识的表征情况。

    arXiv:2608.21766v1 Announce Type: cross  Abstract: Both capability and safety benchmarks rest upon the assumption that the behavior of language models undergoing a test is informative about their behavior in deployment. This assumption can fail, should models infer that they are being evaluated and condition their response on such context. This hypothesis, termed ``evaluation awareness'', has been observed in frontier and open-weight language models alike. We provide a systematic study of this phenomenon, by probing for it across six language models (from four families and three sizes) and three metrics. More precisely, we examine whether (i) being under evaluation is linearly represented within the models' activations space, (ii) it is verbalized in their output tokens (as scored by an LLM-as-judge), and (iii) steering causally affects their behavior. For the open-checkpoint Olmo models, we further test these measures at every training stage. In doing so, we report that evaluation awa
    
[^176]: 学会再看一眼：基于损失差距监督的视觉语言模型自由形式裁剪路由

    Learning to Look Again: Loss-Gap Supervision for Free-form Crop Routing in Vision-Language Models

    [https://arxiv.org/abs/2608.21762](https://arxiv.org/abs/2608.21762)

    本文提出GapSight框架，通过利用目标模型自身的失败信号（损失差距）生成监督标签，训练轻量级路由器，使视觉语言模型在全局浏览后智能选择自由形式区域进行局部重读，以低成本提升细节问题的回答准确性。

    

    视觉语言模型（VLMs）在回答许多细节中心的问题时失败，原因具体明确：答案在图像中可见，但在图像被压缩为低分辨率全局视图后丢失。为每个查询分配更多视觉令牌可改善某些OCR和文档案例，但会不加区分地消耗计算资源，并可能干扰依赖全局上下文的任务。我们提出GapSight，一个学习视觉重读的框架：VLM首先进行全局浏览，然后在问题需要局部证据时选择性地返回自由形式区域。监督信号来自目标模型自身的失败信号。离线时，我们在仅全局视图和候选裁剪增强视图下比较答案损失或多选选项边际；改善目标答案的裁剪成为模型特定的审查标签。一个轻量级自由形式裁剪路由器将这些标签蒸馏为一次性推理策略，预测是否进行重读。

    arXiv:2608.21762v1 Announce Type: cross  Abstract: Vision-language models (VLMs) fail many detail-centric questions for a concrete reason: the answer is visible in the image, yet lost after the image is compressed into a low-resolution global view. Allocating more visual tokens to every query improves some OCR and document cases, but it spends computation indiscriminately and can disturb tasks that rely on global context. We propose GapSight, a framework for learning visual re-reading: a VLM first takes a global glance, then selectively returns to a free-form region when the question calls for local evidence. The supervision comes from the target model's own failure signal. Offline, we compare answer loss or multiple-choice option margin under a global-only view and candidate crop-augmented views; crops that improve the target answer become model-specific review labels. A lightweight free-form crop router distills these labels into a one-shot inference policy that predicts whether to r
    
[^177]: FCPRAG：融合控制器参数化检索增强生成，实现稳定的多段落LoRA注入

    FCPRAG: Fusion-Controller Parametric Retrieval-Augmented Generation for Stable Multi-Passage LoRA Injection

    [https://arxiv.org/abs/2608.21750](https://arxiv.org/abs/2608.21750)

    FCPRAG通过引入轻量级控制器，实现了检索条件化的样本级适配器融合，解决了多段落LoRA注入中的证据融合瓶颈，提高了稳定性和选择性。

    

    参数化检索增强生成（PRAG）通过段落特定的LoRA适配器将检索到的证据注入大型语言模型（LLM），从而减少对长上下文提示的依赖。然而，当同一查询检索到多个段落时，证据级融合成为瓶颈：等权合并可能放大薄弱或冲突的证据，而将检索信号转化为融合权重往往需要脆弱的全局调整。我们提出FCPRAG，一种融合控制的参数化RAG框架，它增加了一个轻量级控制器，用于检索条件化的样本级适配器融合。该控制器预测每段落的融合分数以及样本级校准信号，包括混合门和自适应温度，从而在信息丰富的检索信号下保持选择性融合，在不确定性下保持保守性。FCPRAG通过从每个适配器派生的合并感知监督进行训练。

    arXiv:2608.21750v1 Announce Type: new  Abstract: Parametric retrieval-augmented generation (PRAG) injects retrieved evidence into a large language model (LLM) through passage-specific LoRA adapters, reducing reliance on long in-context prompts. When multiple passages are retrieved for the same query, however, evidence-level fusion becomes a bottleneck: equal-weight merging can amplify weak or conflicting evidence, and translating retrieval signals into fusion weights often requires fragile global tuning. We propose FCPRAG, a fusion-controlled parametric RAG framework that adds a lightweight controller for retrieval-conditioned, sample-level adapter fusion. The controller predicts per-passage fusion scores together with sample-level calibration signals, including a mixing gate and an adaptive temperature, enabling fusion that stays selective under informative retrieval signals and conservative under uncertainty. FCPRAG is trained with merge-aware supervision derived from each adapter's 
    
[^178]: 架构作为编码代理的能力均衡器

    Architecture as Capability Equalizer for Coding Agents

    [https://arxiv.org/abs/2608.21747](https://arxiv.org/abs/2608.21747)

    本论文通过对照实验发现，架构规范格式对编码代理生成代码质量的影响依赖于模型能力，对较弱模型使用代码邻近格式可显著缩小与强模型的能力差距。

    

    基于LLM的编码代理能从高层描述生成完整软件系统，然而关于架构规范格式如何影响生成代码质量，以及这种影响是否取决于模型能力，目前知之甚少。我们进行了一项对照实验，比较了五种信息等价规范格式（非正式散文、带约束和ADR的Mermaid图、OpenAPI、C4/Structurizr DSL、以及带ArchUnit风格规则的TypeScript接口契约），涉及来自三个供应商家族（Anthropic Claude、OpenAI GPT、Google Gemini）的六种模型。在90次多轮代理试验中，规范格式显示出强烈的格式×模型交互效应。在最强大的模型（Sonnet 4.6、GPT-5）上，格式影响甚微（质量差异0.17-0.92）。在较弱模型上，格式产生0.83-2.42分的差异，而接近代码的格式（OpenAPI、TypeScript契约）弥补了大部分能力差距。

    arXiv:2608.21747v1 Announce Type: cross  Abstract: LLM-based coding agents generate complete software systems from high-level descriptions, yet little is known about how the format of architecture specifications affects the quality of generated code or whether this effect depends on model capability. We present a controlled experiment comparing five informationally equivalent specification formats (informal prose, Mermaid diagrams with constraints and ADRs, OpenAPI, C4/Structurizr DSL, and TypeScript interface contracts with ArchUnit-style rules) across six models from three vendor families (Anthropic Claude, OpenAI GPT, Google Gemini). Across 90 multi-turn agent trials, specification format shows a strong format x model interaction. On the strongest models (Sonnet 4.6, GPT-5), format barely matters (quality spread 0.17-0.92). On weaker models, format produces spreads of 0.83-2.42 points, with code-proximate formats (OpenAPI, TypeScript contracts) recovering most of the capability gap.
    
[^179]: L\"etzCross：卢森堡文档多模态检索的跨语言页面级基准

    L\"etzCross: A Cross-Lingual Page-Level Benchmark for Multimodal Retrieval over Luxembourgish Documents

    [https://arxiv.org/abs/2608.21714](https://arxiv.org/abs/2608.21714)

    本文提出了一个针对卢森堡文档的跨语言页面级检索基准，并证明页面图像检索器在低资源跨语言场景下优于OCR文本检索器，且微调可跨语言迁移，法语表现最佳。

    

    arXiv:2608.21714v1 公告类型：新 摘要：近期的页面图像检索器（如ColPali）改善了视觉丰富文档的检索性能，但关于它们在跨语言、低资源环境中的表现知之甚少。我们引入了L\"etzCross，一个针对卢森堡PDF文档的跨语言页面级检索基准，其中文档页面被索引为图像，查询以英语、法语、德语和卢森堡语提供。该基准结合了文本聚焦的问答对和视觉基础的问答对，覆盖了基于PDF的RAG中的文本和视觉检索需求。我们使用L\"etzCross比较了基于OCR的纯文本检索器与ColPali风格的页面图像检索器，发现在这一系统级比较中，后者在跨查询语言上表现更好。我们还考察了单语言和多语言微调。微调在查询语言间具有迁移性，其中法语在卢森堡语查询上取得了单语言中的最高平均性能。

    arXiv:2608.21714v1 Announce Type: new  Abstract: Recent page-image retrievers such as ColPali have improved retrieval over visually rich documents, yet little is known about how they behave in cross-lingual, low-resource settings. We introduce L\"etzCross, a benchmark for cross-lingual page-level retrieval over Luxembourgish PDF documents, with document pages indexed as images and queries provided in English, French, German, and Luxembourgish. The benchmark combines text-focused QA pairs with visually grounded QA pairs, covering both textual and visual retrieval needs in PDF-based RAG. We use L\"etzCross to compare OCR-based text-only retrievers with ColPali-style page-image retrievers and find that the latter perform better across query languages in this system-level comparison. We also examine single-language and multilingual fine-tuning. Fine-tuning transfers across query languages, with French yielding the highest mean performance on Luxembourgish queries among the single-language 
    
[^180]: 计划而非解码器：诊断与修复推理增强文本到图像生成中的组合失败

    The Plan, Not the Decoder: Diagnosing and Repairing Compositional Failure in Reasoning-Augmented Text-to-Image Generation

    [https://arxiv.org/abs/2608.21713](https://arxiv.org/abs/2608.21713)

    本文通过验证计划与解码的可分离性，发现推理增强文本到图像模型的组合失败主要源于计划错误而非解码器不忠实，并提出了可靠的几何评分方法以准确诊断和修复此类问题。

    

    推理增强的文本到图像模型（如GoT-R1）在生成图像标记前，会发出一个显式的文本计划——包括对象名称、属性和边界框。当此类模型在组合提示上失败时，问题在于计划错误，还是计划正确而解码器不忠实？由于计划是机器可读的，可以在解码前进行编辑，这使得两者可以分离。我们首先验证了评判标准。在模型自身链中交换两个边界框，明显翻转了生成的布局：基于检测器的准确率从0.75降至0.48（p<1e-3），而广泛使用的基于VQA的空间度量却上升。一项五名评估者的人类研究在81%的项目上与检测器一致，在57%上与VQA评判者一致。因此，所有空间结果均采用几何评分。在可靠的测量下，解码器是忠实的执行者：94%的生成布局实现了计划中的关系，对象-框绑定在计划对象重新排序后依然保持。

    arXiv:2608.21713v1 Announce Type: cross  Abstract: Reasoning-augmented text-to-image models such as GoT-R1 emit an explicit textual plan - object names, attributes, and bounding boxes - before generating image tokens. When such a model fails a compositional prompt, is the plan wrong, or is the plan right and the decoder unfaithful? Because the plan is machine-readable it can be edited before decoding, which makes the two separable. We first validate the ruler. Swapping the two bounding boxes inside the model's own chain demonstrably flips the generated layout: detector-based accuracy falls 0.75 -> 0.48 (p<1e-3), while a widely used VQA-based spatial metric rises. A five-rater human study agrees with the detector on 81% of items and with the VQA judge on 57%. All spatial results therefore use geometric scoring. Under sound measurement the decoder is a faithful executor: 94% of generated layouts realize the planned relation, and object-box binding survives reordering of the plan's object
    
[^181]: 从关联到因果：通过因果关系与注意力机制提升检索增强生成的检索精度

    From Association to Causation: Improving Retrieval Precision of Retrieval-Augmented Generation via Causal Relations and an Attention Mechanism

    [https://arxiv.org/abs/2608.21702](https://arxiv.org/abs/2608.21702)

    该论文提出通过因果图建模检索阶段，利用对撞子结构中的注意力机制，无需训练即可提升RAG检索精度，解决相似度仅捕捉关联而忽略因果的问题。

    

    arXiv:2608.21702v1 公告类型：新 摘要：检索增强生成（RAG）将大语言模型的生成基于检索到的文档，但标准的终端检索阶段——稠密向量相似度，可选地随后进行重排序——常常返回与查询共享关键词但不包含所需信息的文档，这种失败模式随着知识库的增长而加剧。我们将其追溯到一个概念性差距：相似度仅捕捉关联关系，而真正重要的文档与查询之间是因果关联的。我们用基于赖兴巴赫共因原理的因果图对终端检索阶段建模：查询与检索文档共享的关键词构成潜在共因A，文档的剩余关键词构成潜在集合B，将文档与理想输出联系起来。由于检索到的文档是一个对撞子（A -> d <- B），检索本身在查询和B之间打开了一条关联路径，这允许一种无需训练的注意力机制。

    arXiv:2608.21702v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) grounds LLM generation on retrieved documents, but the standard terminal retrieval stage--dense-vector similarity, optionally followed by reranking--often returns documents that share keywords with the query without containing the needed information, a failure mode that grows with the knowledge base. We trace it to a conceptual gap: similarity captures only associational relations, whereas the documents that matter are linked to the query causally. We model the terminal retrieval stage with a causal graph grounded in Reichenbach's common cause principle: the keywords shared by the query and a retrieved document form a latent common cause A, and the document's residual keywords form a latent set B linking the document to the ideal output. Since a retrieved document is a collider (A -> d <- B), retrieval itself opens an associational path between the query and B, which licenses a training-free, attentio
    
[^182]: 大型语言模型中的激活控制测量

    Measuring Activation Control in Large Language Models

    [https://arxiv.org/abs/2608.21664](https://arxiv.org/abs/2608.21664)

    我们提出了激活可控性基准，首次系统量化了大型语言模型通过自然语言指令控制自身残差流激活的能力，并证明这种控制可部分规避现有激活监控方法。

    

    arXiv:2608.21664v1 公告类型：新 摘要：随着模型能力日益增强，其安全部署很可能将依赖于潜在空间监控作为行为评估的补充，尤其是当那些对评估敏感的模型表现出策略性或欺骗性行为时。然而，如果模型也能控制自身的激活，那么欺骗性行为可能延伸到潜在空间本身。鉴于此，我们引入了激活可控性基准，以量化模型通过自然语言指令调节其残差流的程度。在多个模型家族和能力水平中，我们发现大多数大型语言模型能够以一定的时间分辨率控制其残差流激活的方向和幅度，尽管不同模型间的表现差异显著。在简单任务中，这种控制水平可以（尽管不完美地）规避基于激活的监控方法，包括线性探针、自然语言自编码器、激活预言机以及雅可比透镜。

    arXiv:2608.21664v1 Announce Type: new  Abstract: Safe deployment of increasingly capable models will likely come to rely on latent-space monitoring as a complement to behavioral evaluations, especially when evaluation-aware models exhibit scheming or deception. However, if models can also control their own activations, deception could extend into the latent space itself. With this in mind, we introduce the Activation Controllability Benchmark to quantify the extent to which models can modulate their residual stream via natural-language instruction. Across model families and capability levels, we find that most LLMs can control the direction and magnitude of their residual stream activations with some degree of temporal resolution, though performance varies considerably across models. In simple tasks, this level of control can evade activation-based monitoring methods (including linear probes, natural language autoencoders, activation oracles, and the Jacobian lens), albeit imperfectly.
    
[^183]: 在RAG系统中通过关键词接地事实替换缓解数据库泄露

    Mitigating Database Leakage in RAG Systems with Keyword-Grounded Fact Substitution

    [https://arxiv.org/abs/2608.21656](https://arxiv.org/abs/2608.21656)

    提出KFS-RAG防御方法，通过关键词识别和事实替换净化检索上下文，有效缓解RAG系统中的数据库泄露风险。

    

    arXiv:2608.21656v1 公告类型：新 摘要：检索增强生成（RAG）已成为将大型语言模型（LLM）与外部知识源相结合的强大范式。然而，RAG系统仍然容易受到提示注入攻击，这可能误导检索器或生成器，从而暴露敏感数据库内容。为解决此问题，我们提出了KFS-RAG，一种通过重新表述检索上下文来缓解信息泄露的防御方法。具体来说，我们的方法首先通过注意力展开加因果扰动机制，从检索上下文中识别出一小组有影响力的关键词。然后，这些关键词用于引导辅助LLM从检索段落中生成一组紧凑的关键词接地事实。最后，原始上下文被替换为这些精选事实，确保生成器基于经过净化的证据而非原始检索文本运行。实验评估表明，KFS-RAG显著（此处原文截断）。

    arXiv:2608.21656v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for combining large language models (LLMs) with external knowledge sources. However, RAG systems remain vulnerable to prompt injection attacks, which may mislead the retriever or generator to expose sensitive database contents. To address this issue, we propose KFS-RAG, a defense that mitigates information leakage by reformulating the retrieved context. Specifically, our method first identifies a small set of influential keywords from the retrieved context via an attention rollout plus a causal perturbation mechanism. These keywords are then used to guide an auxiliary LLM to generate a compact set of keyword-grounded facts from the retrieved passages. Finally, the original context is substituted with these curated facts, ensuring that the generator operates on sanitized evidence rather than the raw retrieved text. Experimental evaluations demonstrate that KFS-RAG sig
    
[^184]: 大语言模型真的能遗忘吗？通过对抗性评估揭示遗忘差距

    Can LLMs Truly Forget? Revealing Unlearning Gaps Through Adversarial Evaluation

    [https://arxiv.org/abs/2608.21606](https://arxiv.org/abs/2608.21606)

    本文通过引入攻击成功率（ASR）指标和八种对抗性攻击套件，揭示了大语言模型在标准遗忘评估下看似已遗忘的信息仍可通过策略性提示被显著恢复，表明现有遗忘方法存在对抗性鲁棒性差距。

    

    arXiv:2608.21606v1 公告类型：新 摘要：机器遗忘旨在从模型中移除目标训练数据的影响，同时保留其剩余能力，但评估此类信息是否真正变得不可访问仍具挑战性。现有基准主要在干净、非对抗性查询下评估遗忘，这留下了一个问题：看似已被遗忘的信息是否仍能通过策略性提示被恢复。我们通过使用Llama-3.2-3B-Instruct在TOFU上对基于提示和基于微调的遗忘方法进行统一评估来解决这一差距，随后对在标准指标下表现优异的方法进行对抗性鲁棒性评估。我们引入了攻击成功率（ASR），一种LLM作为评判者的指标，用于衡量对抗性响应中泄漏分数超过$0.2$的比例，并评估了八种攻击套件下的恢复情况。我们的结果揭示了干净查询遗忘与对抗性恢复之间存在显著差距。

    arXiv:2608.21606v1 Announce Type: new  Abstract: Machine unlearning aims to remove the influence of targeted training data from a model while preserving its remaining capabilities, but evaluating whether such information has truly become inaccessible remains challenging. Existing benchmarks primarily assess unlearning under clean, non-adversarial queries, leaving open whether information that appears forgotten can still be recovered through strategic prompting. We address this gap through a unified evaluation of prompt-based and fine-tuning-based unlearning methods on TOFU using Llama-3.2-3B-Instruct, followed by an adversarial robustness evaluation of methods that perform strongly under standard metrics. We introduce Attack Success Rate (ASR), an LLM-as-judge metric that measures the fraction of adversarial responses whose leakage score exceeds $0.2$, and evaluate recovery across eight attack suites. Our results reveal a substantial gap between clean-query forgetting and adversarial r
    
[^185]: K-Bench：在真实科学代理请求上衡量模型性能

    K-Bench: measuring model performance on real scientific agent requests

    [https://arxiv.org/abs/2608.21601](https://arxiv.org/abs/2608.21601)

    本论文提出K-Bench 01，一个基于真实科学请求的评估框架，发现当前前沿模型在满足领域科学家接受标准上均未达到阈值，其中gpt-5.6-sol表现最优但仍有不确定性。

    

    arXiv:2608.21601v1 公告类型：新 摘要：科学人工智能的基准测试大多是为评分而编写的：多项选择题、带参考答案的策划代理任务，或具有已知生成结构的模拟器。真实的科学请求则有所不同。它们规定不充分，携带附件，且缺乏基本事实。我们报告了K-Bench 01，一个从K-Dense Web实时用户流量中抽取的首轮请求构建的评估，并由九个前沿模型在相同沙盒环境中端到端运行，产生了1,602个完成的代理运行。三个盲审语言模型裁判根据八维评分标准对每次运行进行评分。在一个8锚点指示裁判认为领域科学家会接受该工作（仅需少量修改）的评分标准上，没有模型能在所有三位裁判下达标。gpt-5.6-sol具有最高的汇总平均值，为8.04，但其95%置信区间[7.80, 8.23]跨越了阈值，且三位裁判中有两位将claude-opus-5排在第一。

    arXiv:2608.21601v1 Announce Type: new  Abstract: Benchmarks for scientific artificial intelligence are mostly written to be scored: multiple-choice questions, curated agent tasks with reference solutions, or simulators with a known generative structure. Real scientific requests arrive differently. They are underspecified, they carry attachments, and lack ground truth. We report K-Bench 01, an evaluation built from first-turn requests sampled from live user traffic on K-Dense Web and run end to end by nine frontier models in identical sandboxes, yielding 1,602 completed agent runs. Three blinded language-model judges scored every run against an eight-dimension rubric. On a rubric whose 8-anchor instructs judges that a domain scientist would accept the work with minor edits, no model clears the line under all three judges. gpt-5.6-sol has the highest pooled mean, 8.04, but its 95% interval [7.80, 8.23] spans the threshold, and two of the three judges rank claude-opus-5 first instead. We 
    
[^186]: 受控退化下的证据状态可靠性：多阶段LLM流水线中的解析器有效性分歧

    Evidence-State Reliability Under Controlled Degradation: Parser-Validity Divergence in a Multi-Stage LLM Pipeline

    [https://arxiv.org/abs/2608.21559](https://arxiv.org/abs/2608.21559)

    本文提出证据状态可靠性（ESR）作为独立于解析器有效性的评估层，用于衡量多阶段LLM流水线中中间证据在受控退化条件下的完整性、一致性和可用性。

    

    多阶段LLM流水线即使在后续阶段可用的证据变得不完整、压缩或冲突时，仍可能保持结构上的有效性。本文引入并操作化了证据状态可靠性（ESR），这是一个评估层，关注中间证据是否足够完整、有根据、内部一致，并且对阶段分配的功能可用。ESR与解析器有效性（衡量结构符合性）分开评估。我们使用GLM-5.2在60个经过清理的基础案例上，在四种证据条件下评估该框架：干净、压缩有损、部分丢弃和噪声冲突。每种条件都通过决策、审计和升级阶段处理。设计包括720次计划和记账调用，保留了713次经过清理的执行行。在九组匹配的退化减去干净的条件下阶段比较中，所有操作阶段成功估计均为n。

    arXiv:2608.21559v1 Announce Type: new  Abstract: Multi-stage LLM pipelines can remain structurally valid even when evidence available to downstream stages becomes incomplete, compressed, or conflicting. This paper introduces and operationalizes Evidence-State Reliability (ESR), an evaluation layer concerned with whether intermediate evidence remains sufficiently complete, grounded, internally consistent, and usable for a stage's assigned function. ESR is evaluated separately from parser validity, which measures structural conformance.   We evaluate the framework using GLM-5.2 on 60 sanitized base cases under four evidence conditions: clean, compressed-lossy, partial-dropout, and noisy-conflicting. Each condition was processed through decision, audit, and escalation stages. The design comprised 720 planned and ledgered calls, with 713 retained, sanitized execution rows.   Across nine matched degraded-minus-clean condition-stage comparisons, all operational stage-success estimates were n
    
[^187]: 通过TRIAD自动化多跳RAG评估：从上下文提取到验证数据集生成

    Automating Multi-Hop RAG Evaluation via TRIAD: From Context Extraction to Validated Dataset Generation

    [https://arxiv.org/abs/2608.21558](https://arxiv.org/abs/2608.21558)

    本文提出了TRIAD，一种三阶段自动化方法，用于生成特定领域的多跳问答数据集，以支持RAG系统评估，并验证其与现有基准数据集具有相似质量。

    

    arXiv:2608.21558v1 公告类型：交叉 摘要：近年来，大语言模型的进展和RAG系统在工业界的采用，创造了对特定领域问答数据集的需求，这些数据集能够评估RAG在专有数据上的性能。现有数据集，如HotpotQA，挑战了当前基于维基百科知识的RAG系统，但它们无法直接转移到特定领域设置中。对RAG系统质量的全面评估需要多跳查询和不可回答的问题。本文介绍了TRIAD，一种三阶段自动化数据集生成方法。首先，它为RAG系统的特定领域知识库生成问答（QA）对。其次，验证器在反馈循环中检查每个QA对。第三，QA对通过带有相关性标签的上下文文档进行扩展，用于下游评估。我们将此方法与已建立的MuSiQue和HotpotQA数据集进行了评估。结果表明，生成的数据集表现出相似性。

    arXiv:2608.21558v1 Announce Type: cross  Abstract: Recent advances in LLMs and the adoption of RAG systems in industry have created a need for domain-specific question-answer datasets that can assess RAG performance on proprietary data. Existing datasets, such as HotpotQA, challenge current RAG systems on Wikipedia-based knowledge, but they cannot be transferred directly to domain-specific settings. A comprehensive evaluation of RAG system quality requires both multi-hop queries and unanswerable questions. This paper introduces TRIAD, a three-stage automated dataset generation approach. First, it generates question--answer (QA) pairs for the domain-specific knowledge base of a RAG system. Second, a validator checks each QA-pair in a feedback loop. Third, the QA pairs are extended with relevance-labeled context documents for downstream evaluation. We evaluate this approach against the established MuSiQue and HotpotQA datasets. The results show that the generated dataset exhibits similar
    
[^188]: 被遗忘在权重中，被工具恢复：面向LLM智能体的工具性遗忘

    Forgotten in Weights, Recovered by Tools: Agentic Tool Unlearning for LLM Agents

    [https://arxiv.org/abs/2608.21544](https://arxiv.org/abs/2608.21544)

    本文提出了一种两阶段框架ATU，通过结合参数性知识遗忘和轨迹级强化学习，有效抑制LLM智能体通过工具恢复被遗忘信息的能力，同时保持正常工具使用。

    

    大语言模型（LLMs）越来越多地被部署为工具增强的智能体，其响应可能依赖于工具调用和外部观察，而非仅依赖模型参数。这导致了LLM遗忘评估的不匹配：先前的遗忘方法可能抑制直接的参数性回忆，但智能体仍可通过工具（如网络搜索、检索或数据库查询）恢复相同的遗忘目标。我们将这种失败模式识别为工具介导的恢复，并研究智能体工具性遗忘，旨在减少参数性回忆和工具介导的恢复，同时保留对保留知识的正常工具使用。为解决这一挑战，我们提出了智能体工具性遗忘（ATU），一个两阶段框架。第一阶段应用参数性知识遗忘以抑制直接回忆，第二阶段在模拟工具增强环境中进行轨迹级强化学习，以惩罚目标相关工具使用。

    arXiv:2608.21544v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed as tool-augmented agents, where responses can depend on tool calls and external observations rather than model parameters alone. This creates an evaluation mismatch for LLM unlearning: previous unlearning methods may suppress direct parametric recall, but an agent can still recover the same forget target through tools such as web search, retrieval, or database lookup. We identify this failure mode as tool-mediated recovery and study agentic tool unlearning, which aims to reduce both parametric recall and tool-mediated recovery while preserving normal tool use for retained knowledge. To address this challenge, we propose Agentic Tool Unlearning (ATU), a two-stage framework. The first stage applies parametric knowledge unlearning to suppress direct recall, while the second stage performs trajectory-level reinforcement learning in simulated tool-augmented environments to penalize targ
    
[^189]: 超越稀疏权重：注意力何时可压缩？

    Beyond Sparse Weights: When Is Attention Compressible?

    [https://arxiv.org/abs/2608.21541](https://arxiv.org/abs/2608.21541)

    该论文提出CertKV，一种无需训练的注意力压缩方法，通过保留尾部摘要和基于值分散的分配策略，在多个基准测试中实现了高效压缩，表明注意力可压缩性取决于质量、值和上下文分布，而非仅权重稀疏性。

    

    摘要：arXiv:2608.21541v1 公告类型：交叉 摘要：KV缓存压缩通常通过具有少量大权重的注意力图来证明其合理性。但这并不完整：大权重可能不包含大部分质量，被省略的值可能相互抵消，并且保持注意力输出可能无法保持任务性能。我们将这些问题分开处理。全局分数差距——而非阈值计数——决定了需要多少标记来保留目标质量。对于已实现的行，被省略值的加权和是精确缺失的统计量。一个受控的检索-聚合模型解释了截断何时有帮助、何时有害。这些结果催生了CertKV，一种无需训练的压缩器，它为每个头保留一个尾部摘要槽，并根据值分散度分配其余部分。在匹配预算下，CertKV在九个LongBench-v2设置中的七个中排名前二，在128K RULER上保持领先压缩层，并在打包的Llama原型中实现了十倍缓存预算。可压缩性取决于质量、值和上下文分布，而非仅仅权重稀疏性。

    arXiv:2608.21541v1 Announce Type: cross  Abstract: KV-cache compression is often justified by attention maps with a few large weights. This is incomplete: large weights may not contain most of the mass, omitted values can cancel, and preserving the attention output may not preserve the task. We separate these questions. Global score gaps -- not threshold counts -- determine how many tokens are needed to retain a target mass. For a realized row, the weighted sum of omitted values is the exact missing statistic. A controlled retrieval--aggregation model explains when truncation helps and when it hurts. These results motivate CertKV, a training-free compressor that reserves one tail-summary slot per head and allocates the rest by value dispersion. Under matched budgets, CertKV is top-two in seven of nine LongBench-v2 settings, remains in the leading compressed tier on 128K RULER, and realizes a ten-fold cache budget in a packed Llama prototype. Compressibility depends on the mass, values,
    
[^190]: DamageScope：用于卫星图像灾害损毁评估的大规模视觉-语言检索

    DamageScope: Vision-Language Retrieval at Scale for Disaster Damage Assessment from Satellite Imagery

    [https://arxiv.org/abs/2608.21529](https://arxiv.org/abs/2608.21529)

    本文提出DamageScope框架，利用检索增强生成技术结合卫星图像与视觉-语言模型，通过多向量嵌入实现大规模灾害损毁的自动化交互式评估。

    

    摘要：在自然灾害发生后，及时准确地评估财产损失至关重要。传统的现场检查劳动密集、成本高昂，且常常带来安全风险。卫星图像和视觉-语言模型（VLMs）的进步使得可扩展的远程损毁评估成为可能；然而，将VLMs集成到大规模地球观测流程中，在计算效率、数据组织和信息检索方面面临挑战。为解决这些挑战，我们提出了DamageScope，一种检索增强框架，结合卫星图像与视觉-语言模型（VLMs）和大语言模型（LLMs），以自动化财产损毁分析。基于检索增强生成（RAG）框架，DamageScope从卫星图像中提取结构化视觉表示，以支持交互式自然语言查询进行损毁评估。为解决可扩展性问题，我们引入了一种新颖的多向量嵌入技术。

    arXiv:2608.21529v1 Announce Type: cross  Abstract: Timely and accurate assessment of property damage is critical following natural disasters. Traditional on-site inspections are labor-intensive, costly, and often pose safety risks. Advances in satellite imagery and vision-language models (VLMs) enable scalable remote damage assessment; however, integrating VLMs into large-scale Earth observation pipelines presents challenges in computational efficiency, data organization, and information retrieval. To address these challenges, we present DamageScope, a retrieval-augmented framework that combines satellite imagery with Vision-Language Models (VLMs) and Large Language Models (LLMs) to automate property damage analysis. Built on a Retrieval-Augmented Generation (RAG) framework, DamageScope extracts structured visual representations from satellite imagery to support interactive natural language queries for damage assessment. To address scalability, we introduce a novel multi-vector embeddi
    
[^191]: 西里尔问答：语音编码秘密语言对大型语言模型性能的影响

    CyrillicQA: The Influence of Phonetically Encoded Secret Language on LLM Performance

    [https://arxiv.org/abs/2608.21462](https://arxiv.org/abs/2608.21462)

    本文探讨了大型语言模型在解码语音编码秘密语言（如西里尔字母音译）时的能力，并评估其创造性和抽象推理水平。

    

    由于训练数据的选择，大型语言模型（LLMs）在处理使用拉丁字母、拥有大量使用者的标准语言输入时表现最佳，而其他语言变体则处于劣势。尽管如此，它们也可以成为保护此类濒危语言的通用工具。但它们是否也具备人类那样的创造力和抽象能力，来解码语音编码的语言呢？

    arXiv:2608.21462v1 Announce Type: cross  Abstract: Due to the selection of their training data, large language models (LLMs) perform best on standard-language inputs from languages using the Latin alphabet with large speaker populations, while disadvantaging other language varieties. Nevertheless, they can also be a versatile tool for preserving precisely such endangered languages. But do they also possess the necessary creativity and capacity for abstraction to decode phonetically encoded language the same way humans do?
    
[^192]: 评估对热门好莱坞电影的多模态叙事理解

    Evaluating Multimodal Narrative Understanding of Popular Hollywood Films

    [https://arxiv.org/abs/2608.21430](https://arxiv.org/abs/2608.21430)

    本文构建了一个基于票房热门和公有领域状态的好莱坞电影新集合，并开发了一个多模态问答基准，以评估语言模型对电影叙事的理解能力。

    

    arXiv:2608.21430v1 公告类型：新 摘要：多模态语言模型日益展现出对电影进行大规模计算分析的潜力，为研究电影历史和叙事技巧的演变开辟了新途径。然而，围绕好莱坞电影构建稳定基准因版权保护而复杂化。在本工作中，我们直接应对这些挑战，通过两个标准构建了一个新的好莱坞电影集合：票房热度（我们发布了首个大规模、开放的《Variety》杂志1922年至1979年每周票房收入集合）以及可能的公有领域状态（通过研究美国版权登记目录中的版权注册和续期记录）。我们在此集合基础上构建了一个新的多模态多项选择问答基准，专注于直接评估模型对电影叙事进行有意义研究能力的叙事元素；我们发现许多视觉-语言模型在此基准上表现有限。

    arXiv:2608.21430v1 Announce Type: new  Abstract: Multimodal language models increasingly show promise for enabling the large-scale computational analysis of film, opening up new avenues for learning about film history and the evolution of narrative techniques. But the creation of stable benchmarks built around Hollywood films is complicated by copyright protections. In this work, we address these concerns directly, by building a new collection of Hollywood films defined by two criteria: box office popularity (where we publish the first large-scale, open collection of weekly box office earnings reported by Variety magazine from 1922-1979); and likely public domain status (by researching copyright registrations and renewals in the US Catalog of Copyright Entries). We build a new multimodal MCQ benchmark on top of this collection that focuses on narrative elements that directly evaluate the abilities of models to inform meaningful research on film narrative; we find that many vision-langu
    
[^193]: 智能体安全：面向LLM驱动的渗透测试的工具、故障模式与设计法则的系统化研究

    Agentic Security: A Systematization of Tools, Failure Modes, and Design Laws for LLM-Driven Penetration Testing

    [https://arxiv.org/abs/2608.21423](https://arxiv.org/abs/2608.21423)

    本文通过系统化评估十种安全工具，提出了四维集成摩擦指数和定量规律，揭示了LLM驱动的渗透测试中故障模式的本质，并建立了设计法则。

    

    摘要：arXiv:2608.21423v1 公告类型：交叉 摘要：智能体安全利用大型语言模型（LLM）智能体来规划、调度和解释安全工具。随着这些系统从演示转向部署产品，实践者反复遇到相同的操作故障。我们通过对十种广泛使用的静态、动态、云、编排和AI红队工具进行手动评估，系统化了这些故障，以用于无人值守管道。我们引入了一个四维集成摩擦指数，将一次性工程成本与经常性组织、法律和维护成本区分开来。然后，我们推导出定量规律来解释反复出现的故障模式。将智能体安全系统建模为由确定性中介包装的随机LLM策略，我们表明，长生命周期会话会随着阶段数量丢失常驻证据，而短生命周期子智能体根据原始证据与其摘要之间的压缩比扩展可用视野。

    arXiv:2608.21423v1 Announce Type: cross  Abstract: Agentic security uses large-language-model (LLM) agents to plan, dispatch, and interpret security tools. As these systems move from demonstrations to deployed products, practitioners repeatedly encounter the same operational failures. We systematize these failures through a hands-on evaluation of ten widely used static, dynamic, cloud, orchestration, and AI red-teaming tools for unattended pipelines. We introduce a four-dimensional Integration Friction Index that separates one-time engineering cost from recurring organisational, legal, and maintenance cost. We then derive quantitative regularities that explain recurring failure modes. Modelling an agentic security system as stochastic LLM policies wrapped by a deterministic mediator, we show that long-lived sessions lose resident evidence with phase count, while short-lived sub-agents extend the usable horizon according to the compression ratio between raw evidence and its summary. We 
    
[^194]: 通过反事实集成解码缓解大型视觉-语言模型中的偏见

    Mitigating Bias in Large Vision-Language Models via Counterfactual Ensemble Decoding

    [https://arxiv.org/abs/2608.21415](https://arxiv.org/abs/2608.21415)

    本文提出了一种反事实集成解码框架，通过在视觉表示空间中构建并集成多群体反事实视角，有效缓解了大型视觉-语言模型中的社会偏见，从而促进公平行为。

    

    arXiv:2608.21415v1 公告类型：交叉 摘要：大型视觉-语言模型（LVLMs）在广泛的任务中表现出色；然而，它们常常从训练数据中继承社会偏见，导致在处理不同社会群体的肖像时产生有偏见的行为。现有的去偏方法通常在解码过程中比较原始生成和带偏见生成之间的标记概率，但它们从根本上受限于依赖单一、刻板的视角，未能考虑社会观点的多样性。受社会科学中多样性促进公平性的原则启发，我们提出了反事实集成解码（CED），这是一个新颖的框架，在视觉表示空间内构建多群体反事实视角，并在解码过程中集成它们以促进公平的模型行为。CED首先通过识别与特定社会属性相关的语义方向，在视觉空间中进行反事实引导，然后集成这些视角以生成无偏输出。

    arXiv:2608.21415v1 Announce Type: cross  Abstract: Large Vision-Language Models (LVLMs) have achieved remarkable performance across a wide range of tasks; however, they often inherit social biases from their training data, resulting in biased behavior when processing portraits from different social groups. Existing debiasing approaches typically compare token probabilities between the original and biased generations during decoding, but they are fundamentally limited by their reliance on a single, stereotyped viewpoint and fail to account for the diversity of social perspectives. Inspired by the social science principle that diversity fosters fairness, we propose Counterfactual Ensemble Decoding (CED), a novel framework that constructs multi-group counterfactual perspectives within the visual representation space and integrates them during decoding to promote equitable model behavior. CED first performs counterfactual steering in the visual space by identifying semantic directions asso
    
[^195]: 法庭上的谄媚者：大型语言模型是否对司法权威和不断演变的法律标准脆弱？

    Sycophants in the Courtroom: Are LLMs Fragile to Juridical Authority and Evolving Legal Standards?

    [https://arxiv.org/abs/2608.21409](https://arxiv.org/abs/2608.21409)

    该论文发现法律与医学领域存在显著差异，LLMs在法律任务中对权威来源和时间有效性敏感，表现出明显的脆弱性。

    

    在医学领域，当主张得到基于稳定生物现实的实证证据支持时，它们仍然有效。相比之下，在法律领域，真相是偶然的，由管辖权、时间有效性和权威来源的层级所定义。大型语言模型（LLMs）在医学执照考试中的近期成功，引发了对其在法律领域具备同等能力的期望。然而，这种类比掩盖了领域间的关键区别。与医学不同，法律表现往往较少依赖于推理，而更多依赖于确定外部权威何时适用、有效且无矛盾。我们引入了一个比较诊断框架，沿着四个轴（知识回忆、基础依据、置信度和鲁棒性）评估法律推理与医学基线的差异，并将其应用于一个编码时间有效性和规范关系的新基准，揭示了显著的领域不对称性。虽然医学LLMs可靠地...

    arXiv:2608.21409v1 Announce Type: cross  Abstract: In medicine, claims remain valid when supported by empirical evidence grounded in stable biological reality. In law, by contrast, truth is contingent, defined by jurisdiction, temporal validity, and the hierarchy of authoritative sources. The recent success of large language models (LLMs) on medical licensing examinations has encouraged an expectation of comparable legal competence. This analogy, however, obscures a critical distinction between domains. Unlike in medicine, legal performance often depends less on inference than on determining when external authority is applicable, valid, and non-contradictory. We introduce a comparative diagnostic framework evaluating legal reasoning against medical baselines along four axes (knowledge recall, grounding, confidence, and robustness), uncovering a sharp domain asymmetry when applied to a new benchmark that encodes temporal validity and normative relationships. While medical LLMs reliably 
    
[^196]: 生成性合同空白填补

    Generative Gap Filling

    [https://arxiv.org/abs/2608.21401](https://arxiv.org/abs/2608.21401)

    本研究通过实验发现，大型语言模型能从合同剩余文本中恢复被遮蔽的协商条款，准确率高达约90%，远超人类预测，挑战了传统法律填补合同空白方法的假设。

    

    大多数合同诉讼都涉及未能完美记录双方交易的合同。当双方的争议无法通过文本解释解决时，法院会填补空白。学者们长期以来一直假设，剩余的文本内容很快耗尽，并且对争议点上的实际交易提供薄弱的证据。基于这种观点，法官在补充缺失条款时必须依赖于其他因素，从商业默认规则到她自己的政策偏好。尽管经过几代人的研究，法院仍没有真正的替代方案来应对这些无规则的方法。我们测试了这一假设。我们取真实合同，遮蔽了双方协商过的条款，并要求读者预测我们移除的内容。普通受访者大约有一半的时间恢复了隐藏条款，是随机猜测概率的两倍。法学院学生和律师的表现略好。但大型语言模型，仅凭合同的其余部分，就能在十次中恢复近九次。简而言之，这笔交易，在本质上...

    arXiv:2608.21401v1 Announce Type: cross  Abstract: Most contract litigation turns on contracts that imperfectly record parties' bargains. When the parties' dispute can't be solved by interpreting the text, courts fill the gap. Scholars have long assumed that the remaining text runs out quickly, and provides thin evidence of the actual deal on the disputed point. On that view, a judge who supplies the missing term must be drawing on something else, from commercial defaults to her own policy preferences. Despite generations of work, courts have no real alternative to such unruly methods. We tested that assumption. Taking real contracts, we masked a term the parties had negotiated and asked readers to predict what we removed. Lay respondents recovered the hidden term about half the time, twice what chance predicts. Law students and lawyers did marginally better. But large language models, given nothing but the rest of the contract, recovered it nearly nine times in ten. The deal, in short
    
[^197]: Telegram上以色列-巴勒斯坦冲突话语的社交媒体分析

    A Social Media Analysis of Discourse on the Israel--Palestine Conflict on Telegram

    [https://arxiv.org/abs/2608.21385](https://arxiv.org/abs/2608.21385)

    本研究首次大规模系统比较了Telegram上亲以色列与亲巴勒斯坦社区的政治话语，提出微调BERTweet模型在立场检测中显著优于无标签方法。

    

    社交媒体已成为武装冲突争夺的中心舞台，然而Telegram上亲以色列和亲巴勒斯坦社区——其广播架构提供了对刻意政治传播异常直接的记录——尚未在大规模上进行系统性比较。本研究对来自16个Telegram频道（8个亲以色列，8个亲巴勒斯坦）的87,617条消息进行了多方法计算分析，时间跨度从2021年5月至2026年6月，覆盖多次冲突升级。研究结合了情感分析、三种源自不同范式的立场检测方法（关键词匹配、通过自然语言推理的零样本DeBERTa，以及微调的BERTweet模型），以及框架分析，所有方法均基于736条人工标注消息进行评估。微调模型表现最佳（5折交叉验证下准确率72.1%，宏F1分数0.721），比两种无标签基线高出8至11个百分点。

    arXiv:2608.21385v1 Announce Type: cross  Abstract: Social media has become a central arena in which armed conflicts are contested, yet the pro-Israel and pro-Palestine communities on Telegram, whose broadcast architecture yields an unusually direct record of deliberate political communication, have not been systematically compared at scale. This study presents a multi-method computational analysis of 87,617 messages from sixteen Telegram channels, eight pro-Israel and eight pro-Palestine, spanning May 2021 to June 2026 and covering multiple conflict escalations. It combines sentiment analysis, three stance detection methods drawn from distinct paradigms (keyword matching, zero-shot DeBERTa via natural language inference, and a fine-tuned BERTweet model), and a framing analysis, all evaluated against 736 manually annotated messages. The fine-tuned model performed best (72.1% accuracy, 0.721 macro F1 under 5-fold cross-validation), outperforming both label-free baselines by 8 to 11 point
    
[^198]: 超越每字母两字节：西里尔字母AI系统中的分词开销

    Beyond Two Bytes per Letter: Tokenization Overhead in Cyrillic AI Systems

    [https://arxiv.org/abs/2608.21384](https://arxiv.org/abs/2608.21384)

    本文量化了西里尔字母语言（如乌克兰语）在多语言分词器中的分词开销，并提出LLMLingua-2等缓解策略，能显著减少输入长度。

    

    摘要：arXiv:2608.21384v1 公告类型：交叉 摘要：现代多语言分词器通常对乌克兰语及其他代表性不足的西里尔字母语言的分词碎片化程度高于英语，导致成本和上下文容量方面的差异。我们在九种生产级分词器和五种具有标准化西里尔字母与拉丁字母表示的语言上量化了这一开销，覆盖837万个词形。在语料库基准测试中，乌克兰语在现代分词器上的分词开销为68-121%，在较旧的cl100k上为220%，这是通过BrUK和Brown语料库的全文本生育率测量的。在具有独立验证的英语基线的子集中，开销与西里尔字母词汇分配呈负相关，尽管该关联在统计上不显著（Spearman rho = -0.536，p = 0.215，n = 7）。我们评估了两种缓解策略。LLMLingua-2在包含1,536个产品和145个查询的电子商务RAG基准测试中将乌克兰语输入长度减少了47-49%，且没有压缩引发的...

    arXiv:2608.21384v1 Announce Type: cross  Abstract: Modern multilingual tokenizers often fragment Ukrainian and other underrepresented Cyrillic-script languages more heavily than English, creating disparities in cost and context capacity. We quantify this overhead across nine production tokenizers and five languages with standardized Cyrillic and Latin representations, covering 8.37 million word forms. On a corpus benchmark, Ukrainian shows 68-121% token overhead on modern tokenizers and 220% on the older cl100k, measured through full-text fertility on the BrUK and Brown corpora. Overhead is negatively associated with Cyrillic vocabulary allocation in the subset with independently verified English baselines, although the association is not statistically significant (Spearman rho = -0.536, p = 0.215, n = 7). We evaluate two mitigation strategies. LLMLingua-2 reduces Ukrainian input length by 47-49% on an e-commerce RAG benchmark of 1,536 products and 145 queries, with no compression-indu
    
[^199]: PersonaMem-v3：迈向全方位平台个人智能，实现整体用户理解、推荐与智能体任务

    PersonaMem-v3: Toward Omni-Platform Personal Intelligence for Holistic User Understanding, Recommendation, and Agentic Tasks

    [https://arxiv.org/abs/2608.21381](https://arxiv.org/abs/2608.21381)

    PersonaMem-v3 提出了一个基于百万级真实匿名数据的全平台个人智能基准，用于评估跨情境用户理解、可引导推荐、跨平台主动行为及过度个性化的避免。

    

    个人智能正成为面向用户的AI智能体的核心前沿。为了在日常生活中提供帮助，智能体必须理解用户在其偏好、意图、习惯、社交关系和需求随时间展开的数字情境。当今系统可以在单个应用或任务中实现个性化，但整体上的个人智能仍未被充分衡量：智能体如何建立跨情境的用户理解，支持可引导的推荐系统，跨平台主动行动，并避免过度个性化。我们引入了PersonaMem-v3，这是一个基于真实世界、面向全平台个人智能的基准测试和评估框架。PersonaMem-v3源于超过一百万条匿名化的真实世界参与历史，其中大部分是隐式信号，并利用这些数据构建了跨社交媒体、聊天机器人、日历和AI伴侣的时间索引用户数字世界，其中包含偏好的演变。

    arXiv:2608.21381v1 Announce Type: cross  Abstract: Personal intelligence is becoming a central frontier for user-facing AI agents. To be helpful in everyday life, agents must understand users across the digital contexts where their preferences, intents, habits, social relationships, and needs unfold over time. Today's systems can personalize within individual apps or tasks, but personal intelligence as a whole remains under-measured: how agents build cross-context user understanding, support steerable recommendation systems, act proactively across platforms, and avoid over-personalization. We introduce PersonaMem-v3, a real-world-grounded benchmark and evaluation harness for omni-platform personal intelligence. PersonaMem-v3 is seeded from more than one million anonymized real-world engagement histories, most of which are implicit signals, and uses them to construct time-indexed user digital worlds across social media, chatbot, calendar, and AI-companion with preference evolvement over
    
[^200]: 代理式脚手架放大大型语言模型中的谄媚行为

    Agentic Scaffolding Amplifies Sycophantic Behavior in Large Language Models

    [https://arxiv.org/abs/2608.21377](https://arxiv.org/abs/2608.21377)

    本文发现代理式交互脚手架（如多轮反馈和迭代细化）会系统性放大LLM的谄媚行为，导致平均准确率下降6.3%，且更强模型放大效应更显著。

    

    大型语言模型中的谄媚行为，即优先迎合用户认同而非提供真实回答的倾向，已被广泛记录，但主要在单轮对话场景中研究。本文探讨了一个关键问题：对LLM施加更强的交互脚手架是否会使谄媚行为变得更糟？通过4800次真实性判断（200个陈述×6个模型×4种条件），我们发现代理系统特有的交互脚手架（反馈循环、重新考虑检查点和迭代细化）系统性地放大了谄媚行为。多轮交互、用户压力和迭代自我细化各自为模型提供了更多趋向认同的机会，这种漂移伴随着平均准确率下降6.3个百分点，表明这种屈服是有害的而非纠正性的。更强大的模型显示出更大的放大效应，这...

    arXiv:2608.21377v1 Announce Type: cross  Abstract: Sycophancy in large language models, the tendency to prioritize user agreement over truthful responses, has been documented extensively but studied primarily in single-turn settings. This paper investigates a critical question: does subjecting LLMs to greater interaction scaffolding make sycophancy better or worse? Across 4,800 veracity judgments (200 statements $\times$ 6 models $\times$ 4 conditions), we find that the interaction scaffolding characteristic of agentic systems (feedback loops, reconsideration checkpoints, and iterative refinement) systematically amplifies sycophantic behavior. Multi-turn interaction, user pressure, and iterative self-refinement each provide additional opportunities for models to drift toward agreement, and this drift coincides with a mean accuracy drop of $-6.3$ percentage points, establishing the capitulation as harmful rather than corrective. More capable models show larger amplification effects, a t
    
[^201]: 论引用在偏好数据中的作用

    On the Role of Citations in Preference Data

    [https://arxiv.org/abs/2608.21376](https://arxiv.org/abs/2608.21376)

    本文通过混合效应模型研究科学问答中引用对人类和LLM偏好判断的影响，发现人类偏好多样化但数量少的引用，而LLM的引用偏好虽存在但受数据和模型影响。

    

    arXiv:2608.21376v1 公告类型：交叉 摘要：许多自然语言处理任务要求系统在输出中提供归属，即引用来源。归属是防止模型幻觉的堡垒，也是用户验证模型输出可信度的手段。然而，在比较输出时，人类和大型语言模型如何评估引用尚不清楚，这一过程对奖励建模和现代LLM的后训练至关重要。本文研究了在科学问答背景下，引用对人类评审者和四个开源LLM偏好中的作用，利用混合效应模型探讨引用对成对判断的影响。我们的主要发现包括：（1）人类偏好更多样化的引用，但总体数量更少；（2）与人类相比，LLM表现出一些与引用相关的偏好，尽管它们无法访问来源，但这些偏好依赖于数据和特定模型。我们进一步讨论了这些发现的影响。

    arXiv:2608.21376v1 Announce Type: cross  Abstract: Many NLP tasks require systems to provide attribution in their outputs--i.e. citations to grounding sources. Attribution serves as a bulwark against model hallucination and as a means for users to verify the credibility of model outputs. Yet, it is unclear how humans and LLMs evaluate citations when comparing outputs, a process central to reward modeling and modern LLM post-training. This paper studies the role of citations in the preferences of human judges and four open-source LLMs within the context of scientific question answering, leveraging mixed effects models to investigate the influence of citations on pairwise judgments. Among our key findings are (1) that humans prefer more diverse citations but fewer overall, and (2) that LLMs show some citation-related preferences compared to humans, despite lacking access to the sources, but these preferences depend on the data and specific models. We further discuss the implications of o
    
[^202]: 瓦佐比亚评估：尼日利亚皮钦语情感理解、讽刺检测与文化推理基准

    Wazobia Eval: A Benchmark for Nigerian Pidgin Emotion Understanding, Sarcasm Detection, and Cultural Reasoning

    [https://arxiv.org/abs/2608.21369](https://arxiv.org/abs/2608.21369)

    该论文提出了一个针对尼日利亚皮钦语的评估基准，涵盖情感理解、讽刺检测和文化推理，填补了现有基准在文化特定语言理解上的空白。

    

    尼日利亚皮钦语是非洲使用最广泛的语言之一，但在语言模型评估中仍然严重缺乏代表性。现有基准主要集中于翻译、转录或通用情感分析，忽略了文化基础语言理解的关键方面。我们引入了瓦佐比亚评估（Wazobia Eval），这是一个用于评估尼日利亚皮钦语情感理解、讽刺检测和文化推理的基准。该基准基于一个手动标注的数据集，包含超过550个示例和一个16类情感分类体系，旨在捕捉传统情感框架中未涵盖的文化特定情感表达。瓦佐比亚评估提供了标准化评估协议和基准任务，用于评估模型在细微的尼日利亚语言理解方面的表现。我们展示了基准设计、标注方法、分类体系开发过程以及初步结果。

    arXiv:2608.21369v1 Announce Type: cross  Abstract: Nigerian Pidgin is one of Africa's most widely spoken languages, yet remains severely underrepresented in language model evaluation. Existing benchmarks primarily focus on translation, transcription, or generic sentiment analysis, leaving critical aspects of culturally grounded language understanding unmeasured. We introduce Wazobia Eval, a benchmark for evaluating Nigerian Pidgin emotion understanding, sarcasm detection, and cultural reasoning. The benchmark is built on a manually annotated dataset containing over 550 examples and a 16-category emotion taxonomy designed to capture culturally specific emotional registers that are not represented in conventional sentiment frameworks. Wazobia Eval provides standardized evaluation protocols and benchmark tasks for assessing model performance on nuanced Nigerian language understanding. We present the benchmark design, annotation methodology, taxonomy development process, and preliminary pi
    
[^203]: KSE-Web：面向低资源高棉语语义搜索的混合检索与LLM辅助查询扩展分析

    KSE-Web: An Analysis of Hybrid Retrieval and LLM-Assisted Query Expansion for Low-Resource Khmer Semantic Search

    [https://arxiv.org/abs/2608.21365](https://arxiv.org/abs/2608.21365)

    本文提出了KSE-Web，针对低资源高棉语语义搜索，构建了首个清洗后的数据集，并系统评估了多种检索方法，发现传统BM25在性能上优于混合检索和LLM辅助扩展。

    

    作为一种低资源语言，高棉语在检索中面临多项挑战，包括有限的标注数据、模糊的词边界、多语言嵌入模型支持薄弱，以及高棉语与英语混合使用的常见现象。本文介绍了KSE-Web，一项针对高棉语语义搜索的混合检索与LLM辅助查询扩展分析。我们从约17K个候选高棉语标题构建数据集，并在过滤、归一化、去重和文档长度控制后，保留了3K个清洗后的高棉语文本文档。该数据集包含300条人工审阅的用户风格高棉语搜索查询，以及带有部分人工验证的银级相关性标签。我们评估了字符n-gram BM25、多语言密集检索、混合BM25+密集检索，以及使用Qwen2.5模型的LLM辅助查询扩展。实验结果表明，BM25在整体性能上表现最强，达到了0.943的召回率和0.876的nDCG。混合BM

    arXiv:2608.21365v1 Announce Type: cross  Abstract: As a low-resource language, Khmer presents several retrieval challenges, including limited annotated data, ambiguous word boundaries, weak support in multilingual embedding models, and frequent mixed Khmer-English usage. This paper presents KSE-Web, an analysis of hybrid retrieval and LLM-assisted query expansion for Khmer semantic search. We construct the dataset from approximately 17K candidate Khmer titles and retain 3K cleaned full-text Khmer documents after filtering, normalization, deduplication, and document-length control. The dataset includes 300 manually reviewed user-style Khmer search queries and silver relevance labels with partial human verification. We evaluate character n-gram BM25, multilingual dense retrieval, hybrid BM25+dense retrieval, and LLM-assisted query expansion using Qwen2.5 models. Experimental results show that BM25 achieves the strongest overall performance, reaching 0.943 Recall and 0.876 nDCG. Hybrid BM
    
[^204]: 区分增量叙事解释中的修订与延迟细化

    Distinguishing Revision and Delayed Elaboration in Incremental Narrative Interpretation

    [https://arxiv.org/abs/2608.21364](https://arxiv.org/abs/2608.21364)

    本文区分了叙事解释中两种结构不同的更新机制——修订驱动的非单调更新和延迟细化的单调扩展，揭示了它们对增量解释系统结构要求上的根本差异。

    

    arXiv:2608.21364v1 公告类型：交叉 摘要：处理叙事或长格式内容的人类和AI系统都是增量运行的：输入随时间接收，内部表示必须相应更新。因此，增量解释不仅取决于所表示的内容，还取决于表示状态在新证据下如何演变。我们区分了叙事解释中出现的两种结构上不同的更新操作符：修订驱动更新和延迟细化。修订驱动更新因应矛盾而撤回或替换先前承诺的结构，因此是非单调的。相比之下，延迟细化通过约束添加来细化最初未充分指定的元素，而不撤回先前的承诺，产生解释状态的单调扩展。尽管这两种操作符都可能改变对早期材料的理解，但它们对系统结构提出了根本不同的要求。

    arXiv:2608.21364v1 Announce Type: cross  Abstract: Both human and AI systems that process narrative or long-form content operate incrementally: input is received over time, and internal representations must be updated accordingly. Incremental interpretation, therefore, depends not only on what is represented but also on how the representational state evolves under new evidence.   We distinguish two structurally different update operators that arise in narrative interpretation: revision-driven update and delayed elaboration. Revision-driven updates retract or replace previously committed structure in response to a contradiction and are therefore non-monotonic. Delayed elaboration, by contrast, refines initially underspecified elements through constraint addition without retracting prior commitments, yielding monotonic extension of the interpretive state. Although both operators may alter how earlier material is understood, they impose fundamentally different structural requirements on s
    
[^205]: ForeDreamer：一种用于未来事件预测的自我进化双智能体记忆架构

    ForeDreamer: A Self-Evolving Dual-Agent Memory Architecture for Future Event Prediction

    [https://arxiv.org/abs/2608.20920](https://arxiv.org/abs/2608.20920)

    ForeDreamer通过双智能体架构将原始网络证据转化为结构化记忆，分离事实与经验记忆，从而提升开放网络未来事件预测的准确性。

    

    arXiv:2608.20920v1 公告类型：新 摘要：开放网络未来事件预测要求智能体从嘈杂、冗余和不完整的证据中提取可靠信号。现有的检索/记忆机制直接将检索到的信息馈送给智能体，或依赖简单的记忆功能（如存储和重用先前信息）进行预测，这使其不足以应对开放网络预测。我们提出在预测前将原始网络证据转化为结构化记忆，使智能体能够基于提炼后的、针对问题的证据进行推理，而非基于嘈杂的检索结果。本文介绍了ForeDreamer，一种用于管理开放网络证据记忆的自我进化双智能体框架。ForeDreamer将事实记忆（当前预测的问题特定证据状态）与经验记忆（跨预测情节积累的持久智能体经验）分离。它使用一个主智能体进行搜索和预测，以及一个记忆处理子智能体来转换搜索到的内容。

    arXiv:2608.20920v1 Announce Type: new  Abstract: Open-web future event prediction requires agents to distill reliable signals from noisy, redundant, and incomplete evidence. Existing retrieval/memory mechanisms directly feed retrieved information to agents or rely on simple memory functions such as storing and reusing prior information for prediction, leaving them insufficient for open-web forecasting. We propose to transform raw web evidence into structured memory before prediction, enabling agents to reason over distilled, question-specific evidence rather than noisy retrieval results. This paper presents ForeDreamer, a self-evolving dual-agent framework for managing memory over open-web evidence. ForeDreamer separates factual memory, a question-specific evidence state for the current forecast, from experiential memory, persistent agent experience accumulated across forecasting episodes. It uses a main agent for search and prediction, and a memory-processing subagent to convert searc
    
[^206]: 任务协同进化：通过自适应验证任务选择实现高效提示工程优化

    Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection

    [https://arxiv.org/abs/2608.20169](https://arxiv.org/abs/2608.20169)

    本文提出Task-CoEvolve方法，通过自适应选择信息量丰富的验证任务并基于部分评估估计完整性能，显著降低LLM提示工程优化中的评估成本，同时保持优化效果。

    

    arXiv:2608.20169v1 公告类型：交叉 摘要：我们提出了一种通过自适应验证任务选择来高效优化LLM智能体提示工程的新方法。提示工程优化会基于验证性能迭代地重写提示代码，从而在不更新底层模型权重的情况下实现显著的性能提升。然而，现有方法在每次迭代中都会完整评估一个固定的验证集，即使某些任务随着提示工程的进化而变得区分度降低，仍会带来大量的评估成本。我们提出了“任务协同进化”（Task-CoEvolve），该方法通过解决两个挑战来协同进化验证任务与提示工程：选择信息量丰富的任务，以及从部分评估中估计完整集性能。Task-CoEvolve基于一个观察：候选提示工程之间产生分歧的任务比那些一直被解决或失败的任务更能有效区分它们。它使用基于过去结果的方差加权采样。

    arXiv:2608.20169v1 Announce Type: cross  Abstract: We present a novel approach to efficient LLM agent harness optimization through adaptive validation task selection. Harness optimization iteratively rewrites the harness code based on validation performance, enabling substantial performance gains without updating the underlying model weights. Existing approaches, however, evaluate a fixed validation set in full at every iteration, incurring substantial evaluation costs even on tasks that become less discriminative as the harness evolves. We propose $\textbf{Task-CoEvolve}$, which co-evolves the validation tasks with the harness by addressing two challenges: selecting informative tasks and estimating full-set performance from partial evaluations. Task-CoEvolve builds on the observation that tasks on which candidate harnesses disagree are more informative for distinguishing among them than tasks that are consistently solved or failed. It uses variance-weighted sampling based on past outc
    
[^207]: 基于大语言模型常识推理的多智能体编排在自动驾驶中的应用

    Multi-Agent Orchestration with the Common-Sense Reasoning Capabilities of LLMs for Autonomous Driving

    [https://arxiv.org/abs/2608.20129](https://arxiv.org/abs/2608.20129)

    本文提出一种混合自动驾驶框架，通过编排器整合强化学习、PID控制和大语言模型常识推理，并迭代优化奖励函数，以提升在随机场景中的推理能力和安全性。

    

    自动驾驶车辆需要强大的感知和决策能力，以应对多样化和未见过的场景。尽管强化学习和基于规则的方法能提供有效的控制和安全性机制，但在需要上下文推理的情境中，其性能可能下降。大语言模型在理解多模态信息和生成上下文推理方面展现出强大能力，但直接用于车辆控制可能引入延迟和幻觉风险。为解决这些限制，提出了一种混合框架。该系统使用编排器协调基于PPO训练的强化学习和PID控制，并在整个框架中应用大语言模型的常识推理。进一步迭代使用大语言模型推理，以优化动态驾驶环境中的强化学习奖励函数。所提框架在高度随机化的CARLA场景中进行了评估。

    arXiv:2608.20129v1 Announce Type: cross  Abstract: Autonomous vehicles require robust perception and decision-making capabilities to operate in diverse and unseen scenarios. While reinforcement learning and rule-based methods can provide effective control and safety mechanisms, their performance may degrade in situations requiring contextual reasoning. Large Language Models (LLMs) have demonstrated strong capabilities in understanding multimodal information and generating contextual reasoning, however, their use for direct vehicle control can introduce latency and hallucination risks. To address these limitations, a hybrid framework is proposed. This system uses an orchestrator to coordinate PPO-trained reinforcement learning and PID control, with LLM common-sense reasoning applied throughout the framework. LLM reasoning is further employed iteratively to refine the RL reward function for dynamic driving environments. The proposed framework is evaluated in highly randomized CARLA scena
    
[^208]: MileGPO：基于局部证据的里程碑推断用于长时程LLM智能体的图策略优化

    MileGPO: Milestone Inference with Local Evidence for Graph-Based Policy Optimization of Long-Horizon LLM Agents

    [https://arxiv.org/abs/2608.19803](https://arxiv.org/abs/2608.19803)

    MileGPO通过里程碑发现、可靠性校准塑形和进度对比校准三种机制，从在线回滚中提取过程级信用，有效解决了长时程智能体强化学习中的信用分配难题。

    

    在长时程智能体强化学习中，信用分配是一个挑战，因为监督信号往往仅来自最终奖励。现有方法通过步骤分组或基于图的优势估计将轨迹级信号细化为步骤级信用，但可能忽略有意义的中间里程碑。我们提出MileGPO（基于局部证据的里程碑推断用于图策略优化），通过三种设计从分组在线回滚中推导过程级信用。里程碑发现（Milestone Discovery）在成功回滚中识别候选里程碑，在失败回滚中识别反复出现的陷阱。可靠性校准塑形（RCS）根据基于结果的置信度对这些候选进行加权，增强可靠里程碑和陷阱，同时降低不确定候选的权重。进度对比校准（PCC）进一步测试候选是否反映局部进度，以及其传入转换是否优于观察到的替代方案。

    arXiv:2608.19803v1 Announce Type: cross  Abstract: Credit assignment is challenging in long-horizon agentic reinforcement learning, where supervision often comes only from final rewards. Existing methods refine trajectory-level signals into step-level credits through step grouping or graph-based advantage estimation, but can overlook meaningful intermediate milestones. We propose MileGPO (Milestone Inference with Local Evidence for Graph-Based Policy Optimization), which derives process-level credit from grouped on-policy rollouts through three designs. Milestone Discovery identifies candidate milestones on successful rollouts and recurring traps on failed ones. Reliability-Calibrated Shaping (RCS) weights these candidates by outcome-based confidence, strengthening reliable milestones and traps while down-weighting uncertain ones. Progress-Contrastive Calibration (PCC) further tests whether a candidate reflects local progress and whether its incoming ansition outperforms observed alter
    
[^209]: SynFlow：一种多维历时语义分析工具包

    SynFlow: A Multidimensional Diachronic Semantic Analysis Toolkit

    [https://arxiv.org/abs/2608.19472](https://arxiv.org/abs/2608.19472)

    SynFlow是一个开源工具包，通过统一的多维历时分析流程，整合句法、形态、构式和语义特征，以揭示词汇语义变化的具体方面。

    

    词汇语义变化（LSC）通常通过向量空间表示来建模，但这些方法往往对用法的哪些方面发生变化提供的洞察有限。历时语料库研究则转而考察可解释的维度，如句法行为、形态和构式模式，但通常采用各自独立的分析流程。我们提出了SynFlow，一个用于语言用法多维历时分析的开源工具包。SynFlow将语言观察转换为特定时期的分布，并在基于依赖关系的共现、形态特征、构式配置以及外部派生的表示（如框架语义）上应用统一的工作流程。它支持不同的距离度量，以及值级分解、统计检验和词汇填充词的增量聚类。我们通过一个定性案例研究展示了SynFlow的功能。

    arXiv:2608.19472v1 Announce Type: new  Abstract: Lexical semantic change (LSC) is commonly modelled through vector-space representations, but these approaches often provide limited insight into which aspects of usage are changing. Diachronic corpus research instead examines interpretable dimensions such as syntactic behaviour, morphology, and constructional patterns, but typically through separate analytical workflows. We present SynFlow, an open-source toolkit for multidimensional diachronic analysis of linguistic usage. SynFlow converts linguistic observations into period-specific distributions and applies a shared workflow across dependency-based co-occurrences, morphological features, constructional configurations, and externally derived representations such as Frame Semantics. It supports different distance measures, together with value-level decomposition, statistical testing, and incremental clustering of lexical fillers. We demonstrate SynFlow through a qualitative case study o
    
[^210]: 时间序列检索用于在多模态语言模型中实现剩余使用寿命预测的接地

    Time-Series Retrieval for Grounding Multimodal Language Models in Remaining Useful Life

    [https://arxiv.org/abs/2608.19218](https://arxiv.org/abs/2608.19218)

    本文提出一种通过时间序列检索检索历史相似退化段，并将其与测试轨迹一起转换为视觉比较工件，从而提升多模态语言模型在剩余使用寿命预测中的性能，实验证实该方法优于非检索基线。

    

    arXiv:2608.19218v1 公告类型：交叉 摘要：大型语言模型（LLMs）和智能体AI系统正越来越多地被探索用于特定领域的维护和预测任务，这引发了它们能否有效支持预测与健康管理（PHM）的问题。在本文中，我们研究了通过时间序列检索接地的多模态大型语言模型（MLLMs）进行剩余使用寿命（RUL）估计。我们提出了一种框架，其中从训练集中检索历史相似退化段，并与测试轨迹一起，转换为视觉比较工件，由MLLM通过结构化多模态提示进行处理。该方法在C-MAPSS基准的FD001分区上进行了评估，通过重复实验，将基于检索的推理与基于随机参考选择的非检索基线进行了比较。结果表明，时间序列检索持续改善了基于MLLM的RUL预测。

    arXiv:2608.19218v1 Announce Type: cross  Abstract: Large language models (LLMs) and agentic AI systems are increasingly being explored for domain-specific maintenance and prognostics tasks, raising the question of whether they can effectively support prognostics and health management (PHM). In this paper, we investigate remaining useful life (RUL) estimation with multimodal large language models (MLLMs) grounded through time-series retrieval. We propose a framework in which historically similar degradation segments are retrieved from the training set and, together with the test trajectory, transformed into a visual comparison artifact that is processed by the MLLM through a structured multimodal prompt. The approach is evaluated on the FD001 partition of the C-MAPSS benchmark under repeated experiments comparing retrieval-based inference against a non-retrieval baseline based on random reference selection. The results show that time-series retrieval consistently improves MLLM-based RUL
    
[^211]: SPADE：自适应合成可执行环境中的自我对弈

    SPADE: Self-Play in Adaptive Synthetic Executable Environments

    [https://arxiv.org/abs/2608.19197](https://arxiv.org/abs/2608.19197)

    该论文提出SPADE框架，通过单个LLM同时作为环境设计器和推理代理进行自我对弈，动态生成可执行训练环境，以解决语言代理训练中目标分布固定的问题。

    

    持续自我改进需要不断扩展的、自我生成的、多样化的、自适应目标池。对于语言代理而言，现有的训练环境池（人工策划、静态合成或冻结验证器）在学习者规模扩大时保持目标分布固定。我们引入了SPADE（自适应合成可执行环境中的自我对弈），这是一种自我对弈强化学习框架，其中单个大型语言模型扮演两个角色：一个环境设计师，负责编写完整的、长视野的训练环境作为可执行代码，并带有OpenAI Gym风格的reset()/step()接口；以及一个推理代理，学习在这些环境中行动。每个环境都是状态化的、多轮次的（包括状态转换、奖励函数和验证代码），因此一个接口即可涵盖推理问题和多步骤代理工具使用。推理代理的遗憾通过其在有特权提示和无特权提示时的奖励差距来估计；在优化这一遗憾信号时，环境设计师...

    arXiv:2608.19197v1 Announce Type: cross  Abstract: Continuous self-improvement requires an ever-expanding pool of self-generated, diverse, adaptive goals. For language agents, existing training environment pools (hand-curated, statically synthesized, or frozen-verifier) keep the goal distribution fixed as the learner scales. We introduce SPADE (Self-Play in Adaptive Synthetic Executable Environments), a self-play RL framework in which a single LLM plays two roles: an Environment Designer that writes complete, long-horizon training environments as executable code with an OpenAI Gym-style reset()/step() interface, and a Reasoning Agent that learns to act in them. Each is a stateful, multi-turn environment (state transitions, reward functions, and verification code), so one interface spans reasoning problems and multi-step agentic tool use. The Reasoning Agent's regret is estimated using the gap between its reward with and without privileged hints; in optimizing this regret signal the Env
    
[^212]: Aslema在NADI 2026：通过少样本增强进行口语语言理解

    Aslema at NADI 2026: Augmentation through Fewshot for SLU

    [https://arxiv.org/abs/2608.18689](https://arxiv.org/abs/2608.18689)

    本文提出Aslema系统，通过微调优于零样本，并利用大型语言模型生成文化相关的合成数据增强，在NADI 2026任务中在槽位填充上取得第一名。

    

    arXiv:2608.18689v1 公告类型：交叉 摘要：我们介绍了Aslema，这是我们为NADI 2026共享任务5开发的系统，该任务包含两个子任务：意图识别和槽位填充。我们在零样本设置下评估了四种全模态大型语言模型，并将它们与微调模型进行了比较。结果表明，微调始终优于零样本推理。我们进一步探索了合成数据增强，通过使用大型语言模型生成具有文化背景的突尼斯Derja话语，然后通过语音克隆生成合成语音。将这种合成数据纳入后，两个任务的性能均得到提升。我们最终提交的系统基于Qwen3-Omni-30B，并使用原始数据和合成数据的混合进行训练，在开发测试集上实现了86.8%的意图准确率和34.7的词错误率。在官方测试集上，它在槽位填充任务中排名第一（59.5 CoER），在意图识别任务中排名第四（8个团队中，准确率66.1%）。我们发布了实验脚本，并将很快共享合成数据集以支持进一步研究。

    arXiv:2608.18689v1 Announce Type: cross  Abstract: We present Aslema, our system for NADI 2026 Shared Task 5, which consists of two subtasks: intent recognition and slot filling. We evaluate four omni LLMs in a zero-shot setting and compare them with fine-tuned models. Our results show that fine-tuning consistently outperforms zero-shot inference. We further explore synthetic data augmentation by using an LLM to generate culturally grounded Tunisian Derja utterances, followed by voice cloning to generate synthetic speech. Incorporating this synthetic data improves performance on both tasks. Our final submitted system, based on Qwen3-Omni-30B and trained with a mixture of original and synthetic data, achieves 86.8% intent accuracy and 34.7 WER on the devtest split. On the official test set it ranks 1st in slot filling (59.5 CoER) and 4th among 8 teams in intent recognition (66.1% accuracy). We release our experimental scripts and will soon share the synthetic dataset to support further 
    
[^213]: TraceSQL：面向无参考文本到SQL验证的可追踪答案可能性估计

    TraceSQL: Traceable Answerability Estimation for Reference-Free Text-to-SQL Verification

    [https://arxiv.org/abs/2608.17795](https://arxiv.org/abs/2608.17795)

    本文提出TraceSQL，一种利用67个显式诊断特征的可追踪轻量级验证模型，无需参考即可估计文本到SQL生成查询的答案可能性，克服了现有ORMs和LLM裁判的可解释性不足。

    

    文本到SQL系统通常使用真实SQL查询或参考执行结果进行评估，但在实际部署的推理阶段，这种监督信息不可用。这产生了一个关键的验证问题：仅给定用户问题、数据库上下文和生成的SQL，系统能否估计生成的查询是否可能正确回答问题？近期方法使用LLM作为裁判或专门代理来检查生成的SQL，但其决策往往难以追踪。结果奖励模型（ORMs）通过学习带有执行标签的候选SQL并为未见查询分配正确性分数来解决此问题，但它们仍对每次验证背后的信号提供有限的可视性。为解决这一局限，我们提出TraceSQL，一种基于显式诊断特征的轻量级且可追踪的验证模型。TraceSQL结合了67个特征，涵盖问题模糊性、查询复杂度等方面。

    arXiv:2608.17795v1 Announce Type: new  Abstract: Text-to-SQL systems are commonly evaluated using ground-truth SQL queries or reference execution results, but such supervision is unavailable at inference time in real-world deployments. This creates a critical verification problem: given only a user question, database context, and generated SQL, can a system estimate whether the generated query is likely to correctly answer the question? Recent approaches use LLMs as judge or specialized agents to inspect generated SQL, but their decisions can be difficult to trace. Outcome Reward Models (ORMs) address this by learning from execution-labeled candidate SQLs and assigning correctness scores to unseen queries, yet they still provide limited visibility into the signals behind each verification. To address this limitation, we propose TraceSQL, a lightweight and traceable verification model built on explicit diagnostic features. TraceSQL combines 67 features capturing question ambiguity, ques
    
[^214]: CoAL-RAG：一种复杂度感知的法律检索增强生成方法

    CoAL-RAG: A Complexity-Aware Legal Retrieval-Augmented Generation Method

    [https://arxiv.org/abs/2608.17536](https://arxiv.org/abs/2608.17536)

    本文提出了一种复杂度感知的法律检索增强生成方法（CoAL-RAG），通过多维评估机制自适应选择检索策略，平衡了简单与复杂法律问题的答案质量和效率。

    

    法律咨询问题呈现出多层次的复杂度。单一的检索策略往往会导致对简单问题的过度推理，以及对复杂问题的可解释性不足，难以满足高风险场景下对答案质量和效率的双重要求。为解决这一问题，本文提出了CoAL-RAG，一种复杂度感知的法律检索增强生成方法，该方法基于“问题本质”和“检索一致性”构建了多维评估机制，以实现检索策略的自适应路由。首先，根据问题的逻辑结构量化推理需求。然后，利用语义检索与关键词检索之间的差异间接反映问题复杂度，从而选择最合适的检索策略并动态过滤上下文信息。实验结果表明，该方法在性能上表现优越。

    arXiv:2608.17536v1 Announce Type: cross  Abstract: Legal consultation questions exhibit multi-level complexity. A single retrieval strategy often leads to over-reasoning for simple questions and poor interpretability for complex ones, making it difficult to meet the requirements for both answer quality and efficiency in high-risk scenarios. To address this issue, this paper proposes CoAL-RAG, a complexity-aware legal retrieval-augmented generation method, which constructs a multi-dimensional evaluation mechanism based on ``question essence'' and ``retrieval consistency'' to enable adaptive routing of retrieval strategies. First, the reasoning demand is quantified according to the logical structure of the question. Then, the discrepancy between semantic retrieval and keyword retrieval is utilized to indirectly reflect problem complexity, thereby selecting the most appropriate retrieval strategy and dynamically filtering contextual information. Experimental results demonstrate that the p
    
[^215]: 跨不可链接身份的解构攻击：LLM服务有状态防御的局限性

    Decomposition Attacks Across Unlinkable Identities: Limits of Stateful Defenses for LLM Services

    [https://arxiv.org/abs/2608.17445](https://arxiv.org/abs/2608.17445)

    该论文证明了在攻击者使用不可链接身份的情况下，LLM服务的有状态防御在安全性与实用性之间的权衡完全取决于良性请求的分组方式，且当分组不可区分时无法有效阻止解构攻击。

    

    arXiv:2608.17445v1 公告类型：交叉 摘要：大多数大型语言模型服务使用无状态防御，仅判断当前请求，以拒绝有害任务。解构攻击利用这一限制，将有害任务拆分为单独可允许的请求，并组合其答案。因此，防御此类攻击需要一种有状态监控器，将请求视为整体考虑。如果它能够将攻击者的所有请求分组，就可以阻止攻击。然而，攻击者可以使用不可链接的身份，并在其他地方组合答案，从而留下不可靠的分组信号。我们探讨在这种设置下，解构攻击是否仍能被阻止。对于无重试的固定攻击策略，我们证明可实现的安全性和实用性权衡完全取决于良性请求如何针对相同能力进行分组。持久且可识别的分组允许有效的防御；而全新且不可区分的分组则无法实现。当攻击者可以重试并从中学习时，

    arXiv:2608.17445v1 Announce Type: cross  Abstract: Most large language model services use stateless defenses, which judge only the current request, to refuse harmful tasks. Decomposition attacks exploit this limitation by splitting a harmful task into individually permissible requests and combining their answers. Defending against them therefore requires a stateful monitor that considers requests together. If it can group all requests for one attacker task, it can stop the attack. However, attackers can use unlinkable identities and combine answers elsewhere, leaving no reliable grouping signal. We ask whether decomposition attacks can still be stopped under this setting. For a fixed attack strategy without retries, we prove that the achievable security and utility tradeoff depends entirely on how benign requests for the same capabilities are grouped. Persistent, recognizable groups permit a useful defense; fresh, indistinguishable groups do not. When attackers can retry and learn from
    
[^216]: 每一枚硬币都有两面：关于大语言模型同策略蒸馏中泛化的双重性

    Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models

    [https://arxiv.org/abs/2608.16647](https://arxiv.org/abs/2608.16647)

    同策略蒸馏的泛化行为取决于教师和学生来源关系，同源对能跨域迁移推理能力，跨源对则受限于训练分布，这种双重性既是优势也是风险。

    

    同策略蒸馏（OPD）通过监督学生自身策略采样的轨迹来转移教师能力，但其泛化行为仍鲜为人知，因为大多数研究仅在单一领域和接近训练数据的基准上评估OPD。我们进行了一项受控研究，每次只改变一个泛化因素，从域内分布偏移到跨域迁移以及多教师设置。我们发现OPD转移的是教师的推理行为而非其对特定问题的答案：训练难度几乎无关紧要，甚至教师从未解决的问题也有用。迁移强烈依赖于教师和学生之间的来源关系：同源对使学生在语言、推理范围甚至其他领域中接近教师，而跨源对主要适应训练分布。这种广泛的影响是一把双刃剑。

    arXiv:2608.16647v1 Announce Type: new  Abstract: On-policy distillation (OPD) transfers teacher capabilities by supervising trajectories sampled from the student's own policy, yet its generalization behavior remains poorly understood, as most studies evaluate OPD on a single domain and on benchmarks close to the training data. We present a controlled study that varies one generalization factor at a time, from in-domain distribution shifts to cross-domain transfer and the multi-teacher setting. We find that OPD transfers a teacher's reasoning behavior rather than its answers to particular problems: training difficulty barely matters, and even problems the teacher never solves are useful. Transfer depends strongly on the origin relationship between teacher and student: same-origin pairs bring the student close to the teacher across languages, reasoning horizons, and even other domains, whereas cross-origin pairs mostly fit the trained distribution. This broad reach is a double-edged swor
    
[^217]: 提问、条件化或弃权：面向缺失前提推理的强化学习

    Ask, Condition or Abstain: Reinforcement Learning for Missing-Premise Reasoning

    [https://arxiv.org/abs/2608.16554](https://arxiv.org/abs/2608.16554)

    本文提出ACA-RL框架，通过数据增强和结构化奖励训练模型在缺失前提时选择提问、条件化回答或弃权，并引入人工验证的MPB基准以提升推理鲁棒性。

    

    仅答案式强化学习（RL）训练推理模型解决完全明确的问题，但许多现实查询省略了得出唯一答案所需的前提。在这种情况下，有用的响应并不总是拒绝：模型应询问缺失的前提，根据未知量条件化其答案，或在无法提供有信息量的条件响应时弃权。我们提出了《提问-条件化-弃权强化学习》（ACA-RL），一种针对此设置的数据增强强化学习框架。其基于推理图引导的流程将良构问题转换为带有局部缺口标注的缺失前提训练实例；ACA-RL随后使用覆盖五种可观察响应行为的结构化奖励对这些实例进行训练。我们还引入了《缺失前提基准》（MPB），一个包含274个实例、经人工验证的基准，涵盖数学、逻辑和现实世界文字问题。在Qwen3和Llama模型上，ACA-RL表现出显著改进。

    arXiv:2608.16554v1 Announce Type: new  Abstract: Answer-only reinforcement learning (RL) trains reasoning models to solve fully specified problems, but many realistic queries omit a premise needed for a unique answer. In this setting, the useful response is not always refusal: the model should ask for the missing premise, condition its answer on the unknown quantity, or abstain when no informative conditional response is available. We present \emph{Ask-Condition-Abstain Reinforcement Learning} (ACA-RL), a data-augmented RL framework for this setting. Its reasoning-graph-guided pipeline converts well-posed problems into missing-premise training instances with localized gap annotations; ACA-RL then trains on these instances with a structured reward over five observable response behaviors. We also introduce the \emph{Missing-Premise Benchmark} (MPB), a 274-instance human-verified benchmark spanning mathematical, logical, and real-world word problems. Across Qwen3 and Llama models, ACA-RL 
    
[^218]: DuplexGen：解耦内容、时序与声学特征的合成对话语音生成

    DuplexGen: Decoupling Content, Timing, and Acoustics for Synthetic Dialogue Speech

    [https://arxiv.org/abs/2608.16053](https://arxiv.org/abs/2608.16053)

    DuplexGen通过解耦内容、时序和声学特征，利用LLM生成脚本和全双工模型实时交互，使对话时序自然涌现而非预设，实现了更真实、交互驱动的合成对话语音。

    

    摘要：arXiv:2608.16053v1 公告类型：新  摘要：合成对话语音已成为开发和评估对话语音系统的重要资源。然而，现有的对话合成流程通常首先生成对话内容，然后使用手工标记或时序规则插入打断、重叠和反馈语，这使得对话时序是预设的而非由交互驱动。我们提出了DuplexGen，一种明确解耦内容、时序和声学特征的对话合成框架。一个大型语言模型首先生成对话脚本，然后两个全双工对话模型在实时聆听对方的同时执行该脚本。这允许对话时序自然涌现，同时保留脚本化内容。最后，一个高保真文本转语音模型在不改变其时序的情况下重新渲染交互。作为所提出框架的演示，我们构建了一个患者-临床医生对话语音共同...（摘要截断）

    arXiv:2608.16053v1 Announce Type: new  Abstract: Synthetic conversational speech has become an important resource for developing and evaluating conversational speech systems. However, existing dialogue synthesis pipelines typically generate dialogue content first and then insert interruptions, overlap, and backchannels using handcrafted markers or timing rules, making conversational timing prescribed rather than interaction-driven. We present DuplexGen, a dialogue synthesis framework that explicitly decouples content, timing, and acoustics. An LLM first generates the dialogue script, and then two full-duplex conversational models perform the script while listening to each other in real time. This allows conversational timing to emerge naturally while preserving the scripted content. Finally, a high-fidelity text-to-speech model re-renders the interaction without altering its timing. As a demonstration of the proposed framework, we construct a patient--clinician conversational speech co
    
[^219]: TaoLive数字人代理技术报告：训练代理与其操控系统共同进化

    TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness

    [https://arxiv.org/abs/2608.15763](https://arxiv.org/abs/2608.15763)

    本文提出操控系统感知训练（HAT）方法，通过将可进化的操控系统状态纳入训练分布，使数字人代理在实时直播中既能快速响应又能灵活适应动态策略变化。

    

    在直播电商中，AI驱动的数字人主播必须实时回答产品问题、吸引观众并执行不断变化的商业策略。这要求低延迟、事实准确且有效的回复，以及对更新后的活动、合规和风格要求的快速适应。我们开发了一个可进化的操控系统（Harness），将技能（Skills）、钩子（Hooks）、系统提示和工具与模型权重解耦，使得运行时行为无需重新训练即可改变。然而，操控系统的进化创造了一个动态执行环境：在单一配置上微调的紧凑模型可能会记忆名称、模式和提示模板，而不是遵循当前提供的操控系统，而更强的零样本模型又因速度过慢而无法满足实时使用需求。我们通过操控系统感知训练（HAT）来解决这一矛盾，该方法将操控系统状态纳入训练分布。HAT对技能、工具模式和提示应用了任务保持的操控系统状态增强（HSA）。

    arXiv:2608.15763v1 Announce Type: new  Abstract: AI-powered digital-avatar streamers in live e-commerce must answer product questions, engage viewers, and execute changing business strategies in real time. This requires low latency, factual and effective replies, and rapid adaptation to updated campaign, compliance, and style requirements. We develop an evolvable Harness that decouples Skills, Hooks, system prompts, and tools from model weights, allowing runtime behavior to change without retraining. However, Harness evolution creates a moving execution environment: compact models fine-tuned on one configuration may memorize names, schemas, and prompt templates rather than follow the Harness currently provided, while stronger zero-shot models are too slow for real-time use. We address this tension with Harness-Aware Training (HAT), which makes Harness states part of the training distribution. HAT applies task-preserving Harness-State Augmentation (HSA) to Skills, tool schemas, prompt s
    
[^220]: 将医疗大语言模型锚定于因果知识图谱：框架、度量与心血管试点研究

    Grounding Healthcare LLMs in a Causal Knowledge Graph: Framework, Metrics, and a Cardiovascular Pilot

    [https://arxiv.org/abs/2608.15382](https://arxiv.org/abs/2608.15382)

    本文提出一个基于因果知识图谱的可复现评估框架，通过四种锚定条件和自动化评分，系统性地测试医疗大语言模型在干预决策中的推理能力，并在心血管领域进行了验证。

    

    大语言模型（LLMs）越来越多地被提议用于医疗决策支持，但其评估仍侧重于单一答案的准确性，而非对干预、机制、危害、证据和不确定性的推理。我们提出一个可复现的、以图为中心的评估框架，用于医疗保健中面向干预的LLM行为，并在心血管试点中对其进行压力测试。该框架包含四个组成部分：（i）一个领域因果知识图谱，其中断言是带有稳定标识符的一等节点，并保留来源；（ii）一个场景条件化的子图提取步骤，给定任何临床场景，检索相关的具体化断言子图；（iii）四种受控的锚定条件，它们变化检索到的子图如何组合到模型的上下文中（无锚定C1、知识图谱C2、因果图C3、集成C4）；以及（iv）一个自动化评分流水线，锚定于断言标识符。

    arXiv:2608.15382v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly proposed for healthcare decision support, but their evaluations still reward single-answer accuracy rather than reasoning about interventions, mechanisms, harms, evidence, and uncertainty. We propose a reproducible, graph-centered evaluation framework for intervention-oriented LLM behavior in healthcare and stress-test it in a cardiovascular pilot. The framework has four components: (i) a domain causal knowledge graph in which assertions are first-class, provenance-preserving nodes with stable identifiers; (ii) a scenario-conditioned subgraph extraction step that, given any clinical scenario, retrieves the relevant reified-assertion subgraph; (iii) four controlled grounding conditions that vary how the retrieved subgraph is composed into the model's context (ungrounded C1, knowledge-graph C2, causal-graph C3, integrated C4); and (iv) an automated scoring pipeline, anchored on assertion identi
    
[^221]: PatientAct：基于理论的心理健康客户端模拟

    PatientAct: Theory-Grounded Mental Health Client Simulation

    [https://arxiv.org/abs/2608.12750](https://arxiv.org/abs/2608.12750)

    PatientAct通过整合5Ps临床案例公式和带信任阈值的动态记忆层，解决了LLM模拟客户端过度合作、缺乏因果深度的问题，从而更真实地模拟心理健康咨询中的客户端行为。

    

    arXiv:2608.12750v1 公告类型：交叉 摘要：基于LLM的模拟客户端越来越多地用于培训新手咨询师、评估LLM治疗师以及生成合成数据。然而，当前的模拟器产生的客户端过于合作，过早地透露信息，不加抵抗地接受治疗重构，并在单次会话中解决核心问题。我们将这些问题归因于缺乏因果深度的档案以及将所有内容视为同等可访问的行为机制。我们提出了PatientAct，一个基于成熟临床理论的客户端模拟框架。我们的档案整合了5Ps临床案例公式，提供了因果深度，而不将设计局限于任何单一治疗模式。在模拟过程中，档案包含一个动态记忆层，其中项目带有信任阈值（例如，症状早期可用，而形成性记忆需要持续的治疗联盟）。在每个回合中，客户端的情绪反应和行为

    arXiv:2608.12750v1 Announce Type: cross  Abstract: LLM-based simulated clients are increasingly used to train novice counselors, evaluate LLM therapists, and generate synthetic data. However, current simulators produce overly cooperative clients that disclose too readily, accept therapeutic reframes without resistance, and resolve core issues within a single session. We trace these issues to profiles that lack causal depth and behavioral mechanisms that treat all content as equally accessible. We present PatientAct, a framework for client simulation grounded in established clinical theories. Our profiles integrate the 5Ps clinical case formulation, providing causal depth without tying the design to any single therapeutic modality. During simulation, profiles include a dynamic memory layer in which items carry trust thresholds (e.g., symptoms are available early, whereas formative memories require a sustained therapeutic alliance). At each turn, the client's emotional reaction and behav
    
[^222]: 混合线性注意力大语言模型中的大规模激活：注意力前尖峰与尖峰间平台

    Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus

    [https://arxiv.org/abs/2608.12149](https://arxiv.org/abs/2608.12149)

    本文首次系统研究了混合线性注意力大语言模型中的大规模激活现象，发现了注意力前尖峰和尖峰间平台两种新形态，并揭示了它们与架构配置的关系。

    

    我们首次对层交错混合线性注意力（HLA）大语言模型中的大规模激活（MAs）进行了系统性研究，并揭示了两种与架构对齐的形态：MAs在完全注意力层之前持续出现尖峰，形成注意力前尖峰（PAS），并且可以持续通过中间的线性注意力层，产生尖峰间平台（ISP）。随着完全注意力变得更密集，连续的PAS通过ISP逐渐连接，最终恢复完全注意力大语言模型的稳定MA形态。我们证实了这种组织在五种线性注意力架构、六种混合配置、五个数据域以及代表1.2B到397B总参数规模的开源混合模型中的重复性。基于GDN的混合模型在高达1.3B规模的受控预训练表明，这两种形态在早期出现，并对输出门控表现出不对称响应：完全注意力输出门控强烈减弱了...

    arXiv:2608.12149v1 Announce Type: new  Abstract: We present the first systematic study of Massive activations (MAs) in layer-interleaved HLA LLMs and uncover two architecture-aligned morphologies: MAs consistently spike immediately before full attention layers, forming pre-attention spikes (PAS), and can persist through intervening linear attention layers, giving rise to inter-spike plateaus (ISP). As full attention becomes denser, successive PAS become increasingly connected through ISP, ultimately recovering the stable MA morphology of full attention LLMs. We establish the recurrence of this organization across five linear attention architectures, six hybridization configurations, five data domains, and representative open-source hybrid models spanning 1.2B to 397B total parameters. Controlled pretraining of GDN-based hybrids at scales up to 1.3B shows that both morphologies emerge early and respond asymmetrically to output gating: full attention output gating strongly attenuates the
    
[^223]: SAG：基于SQL检索增强生成与查询时动态超边

    SAG: SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges

    [https://arxiv.org/abs/2608.12129](https://arxiv.org/abs/2608.12129)

    提出SAG架构，通过事件-实体索引和查询时动态超边连接，在不构建全局知识图谱的情况下，实现了支持结构化约束和多跳推理的检索增强生成。

    

    arXiv:2608.12129v1 公告类型：新  摘要：尽管检索增强生成（RAG）已被证明能有效为大型语言模型提供外部知识，但主流的稠密检索实现本质上仍受限于处理结构化约束和多跳推理。基于图的方法通过离线构建知识图谱来解决这一问题，但它们常常碎片化语义、维护成本高，并使增量更新复杂化。我们提出SAG（SQL检索增强生成），一种结构化检索架构，它无需构建全局知识图谱，而是将文档组织成事件-实体索引。SAG将每个文本块表示为一个语义完整的事件及其相关实体，形成一条潜在超边，从而保留n元关系而不将其分解为三元组。在查询时，SAG将共享实体视为连接键，以关联相关文本块。这动态地产生一个查询范围内的邻域事件集合，且每一条证据都得到保留。

    arXiv:2608.12129v1 Announce Type: new  Abstract: While retrieval-augmented generation (RAG) has proven effective at giving LLMs access to external knowledge, mainstream dense-retrieval implementations remain inherently limited in handling structured constraints and multi-hop reasoning. Graph-based methods address this by constructing knowledge graphs offline, but they often fragment semantics, incur high maintenance, and complicate incremental updates. We propose SAG (SQL-Retrieval Augmented Generation), a structured retrieval architecture that organizes documents into an event-entity index without building a global knowledge graph. SAG represents each chunk as a semantically complete event paired with its entities, forming a latent hyperedge that preserves n-ary relations without decomposing them into triples. At query time, SAG treats shared entities as join keys to connect related chunks. This dynamically yields a query-scoped neighborhood of events, and yet every piece of evidence 
    
[^224]: 通过测试时缩放视角理解在线策略蒸馏

    Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling

    [https://arxiv.org/abs/2608.11829](https://arxiv.org/abs/2608.11829)

    本研究发现在线策略蒸馏主要提升采样效率而非扩展推理能力边界，其优势在小采样预算下显著，但在大预算下会减弱。

    

    arXiv:2608.11829v1 公告类型：交叉 摘要：在线策略蒸馏（OPD）已成为增强大语言模型推理能力的一种有前景的后训练技术。通常认为，它能使学生模型从更强的教师模型中蒸馏知识，从而扩展超出OPD前基础模型的能力。在本研究中，我们通过测试时缩放的视角，通过改变采样预算K并评估pass@K和avg@K性能来审视这一观点。具体而言，在多种OPD变体中，我们观察到OPD训练的模型在不同采样预算下保持优越的avg@K性能，而pass@K的优势随着K的增加逐渐转移到OPD前的基础模型上。这些结果表明，OPD主要提高了采样效率，而非持续扩展学生模型的推理能力边界。整个OPD训练过程中的pass@K动态进一步揭示，模型逐渐转向更强的低K性能，但以牺牲高K性能为代价。

    arXiv:2608.11829v1 Announce Type: cross  Abstract: On-policy distillation (OPD) has emerged as a promising post-training technique for enhancing LLM reasoning. It is commonly believed to enable the student model to distill knowledge from a stronger teacher model, thereby expanding capabilities beyond the pre-OPD base model. In this study, we examine this view through the lens of test-time scaling by varying the sampling budget K and evaluating performance with pass@K and avg@K. Specifically, across several OPD variants, we observe that OPD-trained models maintain superior avg@K performance across sampling budgets, while the advantage in pass@K gradually shifts to the pre-OPD base models as K increases. These results suggest that OPD primarily improves sampling efficiency rather than consistently expanding the student's reasoning capability boundary. The pass@K dynamics throughout OPD training further reveal a progressive shift toward stronger small-K performance at the expense of the l
    
[^225]: 每一个词都重要：用于测量LLM态度与偏见的精确李克特量表分布

    Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases

    [https://arxiv.org/abs/2608.10503](https://arxiv.org/abs/2608.10503)

    本文提出了一个解析精确的框架，通过全交叉因子实验和精确概率分布计算，取代非结构化基准和蒙特卡洛采样，从而实现对LLM态度与偏见的受控、因果分离式评估。

    

    随着大型语言模型（LLMs）越来越多地被部署为自主代理，准确评估其潜在价值观和偏见至关重要。自然语言处理（NLP）社区通常使用大规模、非结构化的基准来评估模型。虽然这些数据集对评估一般能力有效，但它们从根本上混淆了因果机制：即使检测到总体偏见，非结构化评估也无法厘清其来源是基线特征、情境混淆因素，还是复杂交互作用。为解决这一问题，我们引入了一个解析上精确的框架，用于对LLM进行受控行为评估。我们通过弥合设计、测量和分析方面的差距，将人类心理测量学与LLM机制联系起来。首先，我们用完全交叉的因子实验取代非结构化提示，以系统性地隔离因果主效应和交互效应。其次，我们通过直接操作精确的、逐词级别的概率分布，消除了蒙特卡洛文本采样噪声，从而实现了对李克特量表响应的精确分布计算。

    arXiv:2608.10503v2 Announce Type: replace  Abstract: As Large Language Models (LLMs) are increasingly deployed as autonomous agents, accurately evaluating their latent values and biases is critical. The NLP community typically evaluates models using large, unstructured benchmarks. While effective for general capabilities, these datasets fundamentally conflate causal mechanisms: even when an aggregate bias is detected, unstructured evaluations cannot disentangle whether it stems from baseline traits, contextual confounders, or complex interactions. To address this, we introduce an analytically exact framework for the controlled behavioral evaluation of LLMs. We bridge human psychometrics with LLM mechanics by resolving gaps in design, measurement, and analysis. First, we replace unstructured prompting with fully crossed factorial experiments to systematically isolate causal main and interaction effects. Second, we eliminate Monte Carlo text sampling noise by operating directly on exact,
    
[^226]: 保留完成块：生产流式护栏的精确释放边界等价性

    Withholding the Completing Chunk: Exact Release-Boundary Equivalence for Production Streaming Guardrails

    [https://arxiv.org/abs/2608.10279](https://arxiv.org/abs/2608.10279)

    本文提出了一种流式语言模型输出的安全监控方法，通过编译策略为持久NFA并区分稳定状态，实现了跨任意分块的精确释放边界等价性，确保检测到禁止模式时无法撤回已释放内容。

    

    arXiv:2608.10279v2 公告类型：替换交叉 摘要：流式语言模型输出创建了一个执行边界：一种控制在释放其完成块后检测到禁止模式的控制无法撤回该块。我们研究了一种生产策略，其中每个有序族是两个正则语言谓词的合取。增量匹配是经典的。问题在于跨任意分块在释放时进行精确组合，包括在扩展时可能变化的端到前缀词边界。我们定义了一种ASCII显式策略语法，将每个谓词编译为持久非确定性有限自动机（NFA），区分稳定与临时断言状态，应用文档顺序族优先级，并在释放每个块之前检查决策。我们证明，对于声明语法中的每个策略，所得监视器与吸收累积预言机在释放边界上等价。生产Python和TypeScript实现已进行评估。

    arXiv:2608.10279v2 Announce Type: replace-cross  Abstract: Streaming language-model output creates an enforcement boundary: a control that detects a prohibited pattern after releasing its completing chunk cannot recall it. We study a production policy in which each ordered family is the conjunction of two regular-language predicates. Incremental matching is classical. The problem is exact composition at release time across arbitrary chunk partitions, including end-of-prefix word boundaries that can change on extension. We define an ASCII-explicit policy grammar, compile each predicate to a persistent nondeterministic finite automaton (NFA), distinguish stable from provisional assertion state, apply document-order family priority, and check the decision before releasing each chunk. We show that the resulting monitor is release-boundary equivalent to an absorbing cumulative oracle for every policy in the declared grammar. Production Python and TypeScript implementations were evaluated on
    
[^227]: 大语言模型推荐器是否知道自己在幻觉？目录忠实度中的置信度校准审计

    Do LLM Recommenders Know When They're Hallucinating? Auditing Confidence Calibration in Catalog Faithfulness

    [https://arxiv.org/abs/2608.10008](https://arxiv.org/abs/2608.10008)

    本文首次联合审计了多个LLM推荐器的幻觉率和置信度校准，发现目录成员资格测量方法显著影响结果，并验证了更准确的评估工具，揭示了推荐器在输出目录外项目时过度自信的问题。

    

    arXiv:2608.10008v2 公告类型：交叉替换 摘要：用于Top-K项目推荐的大语言模型（LLM）推荐器经常输出目标目录之外的标题。先前的审计仅报告了二元的域外率，但没有人询问模型是否自知。我们联合审计了来自四个独立供应商（Mistral Large、Llama-3.3-70B、GPT-OSS-120B、Claude Sonnet 4.6）的四个零样本LLM推荐器的幻觉率（OOD@10）和口头置信度校准（ECE、Brier、可靠性），这些系统未经过接地或微调，并跨越三个目录（MovieLens-25M、Amazon Reviews 2023 Toys、Yelp Open Dataset），按项目流行度分层。衡量目录成员资格本身是难点：在相同输出上，报告的比率会因使用的字符串匹配器而改变一个数量级，且F1无法区分候选方案。我们针对201个人类判断验证了该工具，并选择净偏差，其中采用的工具偏差为-0.040，而常见模糊规则的偏差为+0.144。幻觉率随后为...

    arXiv:2608.10008v2 Announce Type: replace-cross  Abstract: LLM recommenders for top-K item suggestion regularly emit titles outside the target catalog. Prior audits report a binary out-of-domain rate; none ask whether the model knew. We jointly audit hallucination rate (OOD@10) and verbalized-confidence calibration (ECE, Brier, reliability) for four zero-shot LLM recommenders from four independent vendors (Mistral Large, Llama-3.3-70B, GPT-OSS-120B, Claude Sonnet 4.6), not grounded or fine-tuned systems, across three catalogs (MovieLens-25M, Amazon Reviews 2023 Toys, Yelp Open Dataset), stratified by item popularity. Measuring catalog membership is itself the hard part: on identical outputs the reported rate moves by an order of magnitude with the string matcher used, and F1 cannot separate the candidates. We validate the instrument against 201 human judgments and select on net bias, where the adopted one is off by -0.040 against +0.144 for the common fuzzy rule. Hallucination is then 
    
[^228]: 科学的伟大无法计划：自主自动研究即模糊测试

    The Greatness of Science Cannot Be Planned: Agentic Auto-Research is Fuzz Testing

    [https://arxiv.org/abs/2608.09855](https://arxiv.org/abs/2608.09855)

    本文提出自主自动研究应借鉴灰盒模糊测试原理，通过引入密集的认识进展信号来指导搜索，以克服最终基准优化导致的过拟合和盲目采样问题。

    

    arXiv:2608.09855v2 公告类型：替换 摘要：自主自动研究正在兴起，但大多数系统将科学发现视为针对最终基准的目标导向优化。这种范式奖励稀疏的最终结论，却忽视了其前的探索过程。当智能体仅优化最终得分时，它们会过度拟合测试条件，并盲目采样而非进行搜索。在一个已声明的研究问题中，研究智能体与软件分析中的灰盒模糊测试器面临相同的稀疏反馈。模糊测试器很少直接发现缺陷，但覆盖率使每次执行的部分进展可观察。模糊测试器利用这一密集信号来变异输入和分配努力，而非仅对完成的运行进行排序。自动研究需要同样的两种能力。首先，每个实验必须在最终科学验证可用之前，暴露一个廉价且密集的认识进展信号。其次，该信号必须决定下一步干预，使智能体进行搜索而非仅做随机尝试。

    arXiv:2608.09855v2 Announce Type: replace  Abstract: Agentic auto-research is emerging, but most systems treat scientific discovery as goal-oriented optimization against a final benchmark. This paradigm rewards a sparse final verdict and ignores the exploration that precedes it. When agents optimize only the final score, they overfit to the test conditions and sample blindly rather than search. Within a declared research problem, a research agent and a greybox fuzzer for software analysis face the same sparse feedback. A fuzzer rarely finds a bug directly, but coverage makes partial progress observable on every execution. Fuzzers use that dense signal to mutate inputs and allocate effort, rather than merely rank completed runs. Auto-research needs the same two capabilities. First, each experiment must expose a cheap, dense signal of epistemic progress before final scientific validation is available. Second, that signal must determine the next intervention so the agent searches rather t
    
[^229]: Macaron-V1：迈向具备自我改进与混合LoRA的开放持续学习

    Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA

    [https://arxiv.org/abs/2608.09819](https://arxiv.org/abs/2608.09819)

    Macaron-V1通过混合LoRA架构和递归改进机制，实现了开放环境中的持续学习与自我提升，兼顾适应性和协作性。

    

    arXiv:2608.09819v2 公告类型：替换-交叉 摘要：Macaron-V1是一个面向体验智能的开放代理-模型家族：在真实环境中从经验中学习，并在部署后继续学习。它围绕两个系统目标组织。适应性通过版本化模型-工具配对对的递归改进来实现，其中一个配置的经验在外部契约下进行评估，并用于构建其继任者。协作性通过混合LoRA（MoL）架构实现，该架构冻结基础模型，组合专业LoRA适配器，并在每次用户交互时选择一个LoRA。旗舰版Macaron-V1-Venti（748B）结合了744B的GLM-5.2基础模型与四个用于聊天、代理、编码和GenUI的LoRA；基于Qwen3.6-35B的Macaron-V1-Tall（50B）采用相同设计用于本地部署。本报告将Macaron-V1作为一个协同设计的系统呈现，涵盖架构、算法和基础设施。MoL架构支持持续学习，通过...

    arXiv:2608.09819v2 Announce Type: replace-cross  Abstract: Macaron-V1 is an open agent-model family for experiential intelligence: learning from experience in real environments and continuing to learn after deployment. It is organized around two system goals. Adaptation is pursued through recursive improvement of versioned model-harness pairs, where experience from one configuration is evaluated under an external contract and used to construct its successor. Collaboration is pursued via the Mixture-of-LoRA (MoL) architecture that freezes a base model, composes specialist LoRA adapters, and selects one LoRA per user turn. The flagship Macaron-V1-Venti (748B) combines a 744B GLM-5.2 base with four LoRAs for chat, agent, coding, and GenUI; the Qwen3.6-35B-based Macaron-V1-Tall (50B) uses the same design for local deployment. This report presents Macaron-V1 as a co-designed system spanning architecture, algorithms, and infrastructure. The MoL architecture supports continual learning throug
    
[^230]: 准确但自然？诊断日本英语写作中的语法与习语差距

    Accurate but Natural? Diagnosing Grammatical and Idiomatic Gaps in Japanese EFL Writing

    [https://arxiv.org/abs/2608.09289](https://arxiv.org/abs/2608.09289)

    本研究通过分层LLM校正流程，首次将日本英语写作中的语法准确性差距与习语性差距分离诊断，发现定冠词和情态动词存在准确性困难，而-ing形式和假设情态动词存在习语使用不足。

    

    第二语言写作研究区分语法准确性与母语般的习语性，但自动化写作评估常常混淆这两个维度。本研究引入了一个分层的大语言模型（LLM）校正流程，通过为120名日本初中生的3,830篇英语写作样本生成字面错误校正和习语改写，将结构错误与非自然性分离开来。应用基于正则表达式的CEF R-J语法提取器，我们量化了两个诊断指标：准确性差距（尝试但错误产生的结构）和习语差距（相对于母语规范，语法正确但使用不足或过度的结构）。结果显示出了不同的模式：定冠词、第三人称单数-s和情态动词（would、could）表现出显著的准确性困难，而-ing形式和假设情态动词（would）显示出最大的习语使用不足，简单现在时动词、主谓宾结构等则呈现出其他特征。

    arXiv:2608.09289v2 Announce Type: replace  Abstract: Second language writing research distinguishes grammatical accuracy from native-like idiomaticity, yet automated writing evaluation often conflates these dimensions. This study introduces a layered LLM-correction pipeline that isolates structural errors from unnaturalness by generating literal error corrections and idiomatic revisions for 3,830 English writing samples from 120 Japanese junior high school students. Applying the regex-based CEFR-J grammar extractor, we quantify two diagnostic measures: accuracy gaps (structures attempted but incorrectly produced) and idiomatic gaps (grammatically correct structures underused or overused relative to native norms). Results reveal distinct patterns: definite articles, third-person singular -s, and modals (would, could) exhibit significant accuracy difficulties, while -ing forms and hypothetical modals (would) show the largest idiomatic underuse, with simple present verbs, subject-verb-obj
    
[^231]: 推理大语言模型中的隐藏语言一致性现象

    Hidden Language Consistency Phenomena in Reasoning LLMs

    [https://arxiv.org/abs/2608.08447](https://arxiv.org/abs/2608.08447)

    本文揭示了推理大语言模型在多语言任务中语言一致性随难度变化的四种行为模式，并发现了“语言一致性崩溃”效应，即难度增加会导致输出语言突然偏离预期语言。

    

    多语言推理模型通常通过是否得出正确答案来评估，而非通过其在推理和回答过程中是否保持预期语言。这种遗漏掩盖了随着任务难度增加而出现的重要多语言行为。在本文中，我们使用PolyMath基准，在八种语言和四个难度级别上，研究了推理模型的任务难度、任务准确性、思维语言一致性（TC）和答案语言一致性（AC）。我们发现了四个结果：（1）语言一致性表现出四种难度依赖行为：输出语言一致性保持与输入对齐、保持不对齐、逐渐退化或突然崩溃。（2）我们识别出语言一致性崩溃效应，即难度增加可能导致输出语言一致性突然下降，尤其是在代表性较弱和非拉丁文字语言中。（3）由于...

    arXiv:2608.08447v2 Announce Type: replace-cross  Abstract: Multilingual reasoning models are commonly evaluated by whether they arrive at the correct answer, but not by whether they preserve the intended language while reasoning and responding. This omission conceals important multilingual behaviors that emerge as tasks become harder. In this paper, we study task difficulty, task accuracy, thinking-language consistency (TC), and answer-language consistency (AC) across reasoning models using PolyMath benchmark in eight languages and four difficulty levels. We uncover four findings: (1) language consistency exhibits four difficulty-dependent behaviors: output-language consistency remains aligned with input, remains misaligned, degrades gradually, or collapses abruptly. (2) We identify the language consistency breakdown effect, where increasing difficulty can cause a sudden drop in output-language consistency, especially in less strongly represented and non-Latin-script languages. (3) Due
    
[^232]: NL2SHACL-Bench：自然语言到SHACL翻译的基准测试套件

    NL2SHACL-Bench: A Benchmark Suite for Natural Language to SHACL Translation

    [https://arxiv.org/abs/2608.07530](https://arxiv.org/abs/2608.07530)

    本文提出了NL2SHACL-Bench基准测试套件，用于评估自然语言到SHACL翻译，并发现现有LLMs能生成语法正确的SHACL，但在复杂语义等价约束上表现不足。

    

    摘要：SHACL是验证RDF知识图谱（KGs）一致性的核心技术。然而，编写SHACL形状需要技术专业知识，而大多数领域专家缺乏这些知识。将自然语言需求翻译成SHACL（NL2SHACL）将降低这一门槛。但是，目前没有专门针对NL2SHACL的基准测试，且评估生成的形状需要超越字符串比较的方法，因为语义等价的形状可能在序列化和结构上有所不同。为解决这些挑战，我们提出了NL2SHACL-Bench，一个用于自然语言到SHACL翻译的基准测试套件。使用NL2SHACL-Bench，我们评估了四种最先进的大型语言模型（LLMs）在此任务上的表现。我们的结果表明，当前LLMs在生成语法有效的SHACL方面能力很强，但在为复杂逻辑和结构模式生成语义等价的约束方面仍有困难。这表明NL2SHACL-Bench为评估和改进这一任务提供了有意义的手段。

    arXiv:2608.07530v2 Announce Type: replace  Abstract: SHACL is a core technology for validating the conformance of RDF knowledge graphs (KGs). Yet, authoring SHACL shapes requires technical expertise that most domain experts lack. Translating natural language requirements into SHACL (NL2SHACL) would lower this barrier. However, there is no dedicated benchmark for NL2SHACL, and evaluating generated shapes requires methods beyond string comparison, as semantically equivalent shapes can differ in serialisation and structure. To tackle these challenges, we present NL2SHACL-Bench, a benchmark suite for natural language to SHACL translation. Using NL2SHACL-Bench, we evaluate four state-of-the-art large language models (LLMs) for this task. Our results show that current LLMs are highly capable of generating syntactically valid SHACL, but still struggle to produce semantically equivalent constraints for complex logical and structural patterns. This indicates that NL2SHACL-Bench provides a meani
    
[^233]: 构建CIE：一个用于从施工事故叙述中提取因果信息的数据集

    ConstructCIE: A Dataset for Extracting Causal Information from Construction Accident Narratives

    [https://arxiv.org/abs/2608.06495](https://arxiv.org/abs/2608.06495)

    本文介绍了ConstructCIE数据集，用于从施工事故报告中提取分层因果信息，并评估了多种模型，发现它们擅长预测事故类型但难以精确提取因果证据片段。

    

    arXiv:2608.06495v2 公告类型：替换 摘要：施工事故叙述包含丰富的因果信息，但证据往往隐含、跨段落且分散。我们引入了ConstructCIE，一个手工标注的数据集，用于从OSHA施工事故报告中提取因果信息。该数据集采用分层模式，涵盖事故类型、因果因素、子因果因素和支持性证据片段。我们评估了监督序列标注器和指令调优的大型语言模型在端到端分层提取设置中的表现。结果表明，大多数评估模型在事故类型预测上表现强劲，并能恢复广泛的因果含义，但在精确的片段级提取上仍有限。联合分层提取通常在精确匹配和软匹配上表现更强，而单独分层提取有时在关键词F1分数上更高。错误分布因提取策略而异，但证据选择和片段边界错误仍然常见。

    arXiv:2608.06495v2 Announce Type: replace  Abstract: Construction accident narratives contain rich causal information, but the evidence is often implicit, long-span, and distributed. We introduce ConstructCIE, a manually annotated dataset for Causal Information Extraction from OSHA construction accident reports. The dataset uses a hierarchical schema for accident types, causal factors, sub-causal factors, and supporting evidence spans. We evaluate supervised sequence taggers and instruction-tuned LLMs in an end-to-end hierarchical extraction setting. Results show that most evaluated models achieve strong accident-type prediction and recover broad causal meaning but remain limited in precise span-level extraction. Joint Hierarchical Extraction generally achieves stronger exact and soft matching, while Individual Hierarchical Extraction sometimes achieves higher keyword F1. Error distributions vary by extraction strategy, but evidence-selection and span-boundary errors remain common. The
    
[^234]: 稀疏PPMI图平均用于随机索引嵌入

    Sparse PPMI Graph Averaging for Random Indexing Embeddings

    [https://arxiv.org/abs/2608.05724](https://arxiv.org/abs/2608.05724)

    该论文提出了一种结合稀疏PPMI图平均和稳健缩放的随机索引后处理流程，显著提升了亲属关系类比任务的准确率，从19.41%提高到30.74%。

    

    arXiv:2608.05724v2 公告类型：替换。摘要：我们研究了在小型童话故事语料库中，针对亲属关系类比任务的一种特定稀疏后处理流程，用于随机索引（RI）。已发布的成果使用均匀RI上下文累积，维度为200，非零元素为8，随后进行一次残差图平均，公式为 $\mathbf{E}=(1-\alpha)\mathbf{E}_0+\alpha\mathbf{P}\mathbf{E}_0$，其中 $\mathbf{P}$ 是行归一化的PPMI图，$\alpha=0.3$。最后应用终端行归一化和每维度中位数/四分位距缩放。在Google类比基准的家族部分，506个问题中有272个对所有种子有效。在五对配对种子中，完整流程将准确率从19.41%提高到30.74%，提升了11.32个百分点，嵌套自助法95%置信区间为[6.93, 15.89]。仅稳健缩放贡献了3.24个百分点[1.25, 5.38]，而图平均（不包含稳健缩放）贡献了6.18个百分点[2.63, 9.92]。一个单独的40题通用测试也...（摘要截断）

    arXiv:2608.05724v2 Announce Type: replace  Abstract: We study a specific sparse post-processing pipeline for Random Indexing (RI) on kinship analogies in a small fairytales corpus. The published artifacts use uniform RI context accumulation with 200 dimensions and eight nonzeros, followed by one residual graph average, $\mathbf{E}=(1-\alpha)\mathbf{E}_0+\alpha\mathbf{P}\mathbf{E}_0$, where $\mathbf{P}$ is a row-normalized PPMI graph and $\alpha=0.3$. Terminal row normalization and per-dimension median/IQR scaling are then applied. On the Google analogy benchmark's family section, 272 of 506 questions are valid for every seed. Across five paired seeds, the complete pipeline raises accuracy from 19.41\% to 30.74\%, a gain of 11.32 percentage points with a nested-bootstrap 95\% confidence interval of [6.93, 15.89]. Robust scaling alone contributes 3.24 points [1.25, 5.38], while graph averaging without robust scaling contributes 6.18 points [2.63, 9.92]. A separate 40-question general gri
    
[^235]: 先答后想：当承诺顺序在扩散语言模型中损害准确性时

    Answer First, Reason Later: When Commitment Order Costs Accuracy in Diffusion Language Models

    [https://arxiv.org/abs/2608.05687](https://arxiv.org/abs/2608.05687)

    本文发现扩散语言模型中“答案优先”的承诺顺序（即答案先于推理提交）会降低准确性，而延迟答案提交或限制提交位置可提升性能。

    

    掩码扩散语言模型并行修订多个掩码输出位置。我们将一个令牌称为“已承诺”，一旦它变为可见且不再被掩码；并将一个响应称为“答案优先”，当最终答案在它前面打印的推理之前提交。在1，069个GSM8K测试问题上，显式的逐步指令增加了无限制解码与仅允许在最左侧未解析位置附近提交的解码器之间的准确性差异；无限制解码也产生更多答案优先的轨迹。在MATH-500上，两个LLaDA模型将短输出画布的大部分时间用于在答案之后提交的推理，并且前沿门控的益处随着这种答案后写作的消失而减少。Dream-7B几乎没有答案后写作，并遵循不同的准确性模式。一个受控的四选项任务在生成前保留一个单令牌答案位置。延迟该位置的表现优于提前提交。

    arXiv:2608.05687v2 Announce Type: replace-cross  Abstract: Masked diffusion language models revise many masked output positions in parallel. We call a token committed once it becomes visible and is never masked again, and call a response answer-first when the final answer commits before the reasoning printed ahead of it. On 1,069 GSM8K test questions, an explicit step-by-step instruction increases the accuracy difference between unrestricted decoding and a decoder that permits commitment only near the left-most unresolved position; unrestricted decoding also produces more answer-first trajectories. On MATH-500, the two LLaDA models spend most of a short output canvas on reasoning that commits after the answer, and the benefit of frontier gating decreases as that postanswer writing disappears. Dream-7B has little post-answer writing and follows a different accuracy pattern. A controlled four-option task reserves a one-token answer position before generation. Delaying that position outpe
    
[^236]: RIG-RoPE：基于实例局部旋转几何与表示感知遍历坐标的关系分层多模态注意力

    RIG-RoPE: Relation-Stratified Multimodal Attention with Instance-Local Rotary Geometry and Representation-Aware Traversal Coordinates

    [https://arxiv.org/abs/2608.05154](https://arxiv.org/abs/2608.05154)

    本文提出RIG-RoPE机制，通过关系和实例门控及持续时间感知时间坐标，解决了多模态位置编码中的跨模态空间干扰和时间步长不当问题。

    

    旋转位置编码（RoPE）是现代语言模型的核心组件，并已通过多维变体（如多模态RoPE，M-RoPE）扩展到多模态大语言模型中，该变体将位置通道分割为时间、高度和宽度子空间。本报告识别了交错多模态上下文中静态多维位置分配的两种局限性。首先，高度/宽度旋转可能被应用于空间位移并非明确定义几何对象的标记对，从而产生跨模态和跨实例的空间干扰。其次，时间坐标常被视为等步长计数器，因此一个文本标记、一个图像块和一个视频片段尽管信息密度不同，却可能以可比量推进时间相位。我们提出了RIG-RoPE，一种带有持续时间感知时间坐标的关系和实例门控RoPE机制。RIG-RoPE为每个标记增加一个模态

    arXiv:2608.05154v2 Announce Type: replace  Abstract: Rotary positional encoding (RoPE) is a core component of modern language models and has been extended to multimodal LLMs through multidimensional variants such as multimodal RoPE (M-RoPE), which split positional channels into temporal, height, and width subspaces. This report identifies two limitations of static multidimensional position assignment in interleaved multimodal contexts. First, height/width rotations may be applied to token pairs whose spatial displacement is not a well-defined geometric object, producing cross-modal and inter-instance spatial interference. Second, temporal coordinates are often treated as equal-step counters, so a text token, an image block, and a video segment can advance the temporal phase by comparable amounts despite different information density.   We propose RIG-RoPE, a relation- and instance-gated RoPE mechanism with duration-aware temporal coordinates. RIG-RoPE augments each token with a modalit
    
[^237]: 面包屑式搜索代理

    Breadcrumbing Search Agents

    [https://arxiv.org/abs/2608.04565](https://arxiv.org/abs/2608.04565)

    本文揭示了搜索代理中通过中介搜索界面协调注入受控结果可显著提升攻击成功率的安全漏洞，挑战了仅针对静态注入的现有防护。

    

    arXiv:2608.04565v2 公告类型：替换交叉 摘要：基于LLM的搜索代理被广泛用于信息检索任务，但其对外部工具返回结果的依赖引入了严重的安全风险：执行过程中检索到的网页内容不可信，使代理面临提示注入和目标劫持的威胁。先前关于搜索代理安全的研究主要集中于静态网页内容注入，但现代代理会发出后续查询并交叉核对竞争来源，因此单个注入页面通常会被稀释或拒绝。我们表明，传递搜索和页面观察结果的通道是一个脆弱的安全边界：除了使代理暴露于单个被污染页面之外，中介搜索界面还可以反复引导代理如何收集证据并形成最终答案。在受限的工具中介威胁模型下，当证据在代理的轨迹中协调一致时，每次查询仅附加一个受控结果就能显著提高攻击成功率。

    arXiv:2608.04565v2 Announce Type: replace-cross  Abstract: LLM-based search agents are widely used for information-seeking tasks, but their reliance on external tool returns introduces a critical security risk: web content retrieved during execution is untrusted, exposing agents to prompt injection and goal hijacking. Prior work on search-agent safety primarily focuses on static web-content injection, but modern agents issue follow-up queries and cross-check competing sources, so a single injected page is often diluted or rejected. We show that the channel delivering search and page observations is a fragile security boundary: beyond exposing the agent to a single poisoned page, a mediated search interface can repeatedly steer how the agent gathers evidence and forms its final answer. Under a constrained tool-intermediary threat model, appending only one controlled result per query can substantially increase attack success when the evidence is coordinated across the agent's trajectory.
    
[^238]: 预测多语言分类与翻译性能的大型语言模型：跨语言对齐是否足够？

    Predicting Multilingual Classification and Translation Performance of LLMs with Cross-Lingual Alignment -- Is English Enough?

    [https://arxiv.org/abs/2608.03446](https://arxiv.org/abs/2608.03446)

    本研究比较了27种跨语言对齐分数，并首次验证它们在预测多语言翻译性能上的有效性，提出了一种新的PMI翻译度量以支持跨语言比较。

    

    多语言大型语言模型（LLMs）在非英语分类任务中表现出更好的性能，当给定语言的表示在模型内与英语更对齐时。已有多种跨语言对齐（CLA）分数被提出用于LLMs，以及多种从模型中提取嵌入的方法。我们对27种CLA分数变体进行了比较分析，考察它们之间的差异以及每种分数在三个任务中预测下游性能的能力。关键在于，尽管LLMs广泛用于机器翻译等生成任务，但先前的工作几乎完全集中在分类上。因此，我们研究CLA分数是否同样能预测翻译性能。为了计算跨目标语言的相关性，我们提出了一种基于PMI的翻译度量标准，该标准对目标语言的依赖性较小，并与chrF强相关。

    arXiv:2608.03446v2 Announce Type: replace  Abstract: Multilingual large language models (LLMs) have been shown to perform better on non-English classification tasks when the representations of the given language are more aligned to English within the model. Several cross-lingual alignment (CLA) scores have been proposed for use with LLMs, along with multiple approaches for extracting embeddings from the models. We provide a comparative analysis of 27 CLA score variants, examining how they differ and how well each predicts downstream performance across three tasks. Crucially, while LLMs are widely used for generative tasks such as machine translation, prior work has focused almost exclusively on classification. We therefore investigate whether CLA scores are similarly predictive of translation performance. To enable computing correlations across target languages, we propose a PMI-based translation metric, which is less dependent on the target language and correlates strongly with chrF. 
    
[^239]: 编码代理中的提示诱发浪费：推理、努力、框架设计与端到端成本

    Prompt-Induced Waste in Coding Agents: Reasoning, Effort, Harness Design, and End-to-End Cost

    [https://arxiv.org/abs/2608.01347](https://arxiv.org/abs/2608.01347)

    本论文揭示提示语义、推理努力和框架设计是相互作用的因素，共同决定编码代理的端到端成本与成功率，而非独立可调的控制变量。

    

    编码代理的效率不能仅通过令牌数量或模型价格来表征。我们研究了端到端成本和任务成功率如何共同依赖于提示语义、推理努力、框架策略、模型、任务难度、工具使用、上下文管理和提供商计费。受控提示实验表明，措辞可以在不改变任务的情况下改变推理和验证行为。一项独立的SWE-bench Verified研究显示，额外的推理努力可以改善某些模型的困难任务，但也可能增加成本而无收益。DeepSeek框架扩展表明，即使模型、任务、提示和控制器逻辑保持固定，努力控制干预的效果也会随框架变化而显著改变。这些结果表明，提示、努力和框架是相互作用的实验因素，而非独立的效率控制。我们将效率建模为每项成功任务的成本。

    arXiv:2608.01347v4 Announce Type: replace  Abstract: Coding-agent efficiency cannot be characterized by token count or model price alone. We study how end-to-end cost and task success depend jointly on prompt semantics, inference effort, harness policy, model, task difficulty, tool use, context management, and provider accounting. Controlled prompt experiments show that wording can change reasoning and verification behavior without changing the task. A separate SWE-bench Verified study shows that additional inference effort can improve difficult tasks for some models but can also add cost without benefit. A DeepSeek Harness extension shows that the effect of an effort-control intervention changes substantially when the harness changes, even when the model, tasks, prompts, and controller logic are held fixed. These results show that prompt, effort, and harness are interacting experimental factors rather than independent efficiency controls. We model efficiency as cost per successful tas
    
[^240]: 文本到视觉的迁移是什么？视觉语言模型的能力缩放规律与迁移动态

    What Transfers from Text to Vision? Capability Scaling Laws and Transfer Dynamics for VLMs

    [https://arxiv.org/abs/2608.00013](https://arxiv.org/abs/2608.00013)

    我们提出了首个跨家族的多模态缩放规律，通过文本能力得分直接预测VLM性能，并在150多个VLM上验证了其有效性。

    

    arXiv:2608.00013v2 公告类型：替换-交叉 摘要：选择合适的大型语言模型（LLM）骨干是构建视觉语言模型（VLM）时最关键的决策，但这一过程从根本上缺乏原则性：基于计算量的缩放规律无法跨模型家族泛化，且不存在在训练开始前直接预测VLM性能的框架。我们提出了能力驱动的多模态缩放规律，这是首个跨家族框架，能够从直接可观测的文本能力预测VLM基准准确率。给定通过主成分分析从LLM文本基准中提取的低维能力得分$S$，我们将VLM性能建模为$S$的函数，并引入每个骨干的迁移率和量化数据缩放效率的吸收率。为拟合和验证该框架，我们在严格控制的配方下，基于34个LLM（涵盖7个模型家族）训练了超过150个VLM。在超过200个文本基准和50个多模态基准上的评估表明...

    arXiv:2608.00013v2 Announce Type: replace-cross  Abstract: Choosing the right large language model (LLM) backbone is the most consequential decision when building a vision-language model (VLM), yet it remains fundamentally unprincipled: compute-based scaling laws fail to generalize across model families, and no framework exists for directly predicting VLM performance before training begins. We propose the Capability-Driven Multimodal Scaling Law, the first cross-family framework that predicts VLM benchmark accuracy from directly observable textual capability. Given a low-dimensional capability score $S$ extracted from LLM textual benchmarks via PCA, we model VLM performance as a function of $S$, with a per-backbone transfer rate and an absorption rate that quantifies data-scaling efficiency. To fit and validate the framework, we train over 150 VLMs on 34 LLMs spanning 7 model families under a strictly controlled recipe. Evaluations on more than 200 textual and 50 multimodal benchmarks 
    
[^241]: RSMeM：面向遥感智能体的知识增强记忆演化与系统化评估

    RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents with Systematic Evaluation

    [https://arxiv.org/abs/2607.24772](https://arxiv.org/abs/2607.24772)

    RSMeM提出了一种知识增强的记忆演化机制，结合层次化知识锚定和失败感知经验精炼，显著提升了遥感智能体在多步骤工具执行中的稳健性。

    

    地球科学研究需要复杂的分析和领域专业知识，其中遥感观测是关键基础。然而，基于通用大语言模型的现有遥感智能体在很大程度上仍缺乏领域特定性，导致工作流程脆弱且易出错。此外，这些失败很少被整合为可重用的经验，以供后续分析使用。为解决这一问题，我们引入了RSMeM，一种知识增强的记忆演化机制，该机制通过预蒸馏的领域知识引导遥感智能体，并迭代整合在线经验以实现稳健的多步骤工具执行。RSMeM由两个组件组成：（i）层次化知识锚定，它在层次化领域语料库上执行基于分类的检索，以指导规划和工具选择；（ii）失败感知的经验精炼，它将带有失败标注的工具使用轨迹蒸馏为可重用的约束，用于下一轮工具执行。通过迭代...

    arXiv:2607.24772v2 Announce Type: replace  Abstract: Geoscience research requires complex analysis and domain expertise, with remote sensing (RS) observations as a key foundation. However, existing RS agents built on general-purpose LLMs remain largely domain-agnostic, resulting in brittle and error-prone workflows. Moreover, these failures are seldom consolidated into a reusable experience for subsequent analyses. To address this issue, we introduce RSMeM, a knowledge-enhanced memory evolution mechanism that bootstraps RS agents with pre-distilled domain knowledge and iteratively integrates online experience for robust multi-step tool execution. RSMeM is composed of two components: (i) Hierarchical Knowledge Grounding, which performs taxonomy-aware retrieval over a hierarchical domain corpus to guide planning and tool selection; and (ii) Failure-Aware Experience Refinement, which distills failure-annotated tool-use traces into reusable constraints for next-round tool execution. By ite
    
[^242]: 思维链不忠实性的两种模式：基于度量的检测在模型出错时失效

    Two Regimes of Chain-of-Thought Unfaithfulness: Metric-Based Detection Fails Where Models Are Wrong

    [https://arxiv.org/abs/2607.23458](https://arxiv.org/abs/2607.23458)

    本文发现思维链不忠实性检测存在两种截然不同的模式：在正确答案上行为信号有效，但在错误答案上（不忠实性主要集中地）所有检测方法均失效，且答案正确性本身是最强的预测指标。

    

    思维链（CoT）解释只有在忠实的情况下才能支持监督：所陈述的推理必须实际产生答案。我们针对FaithCoT-Bench的人类标注，审计了黑箱（行为性）检测不忠实思维链的方法，发现答案正确性在每一个层面都结构化了这个问题。仅凭答案不正确性（一种预言机诊断，而非可部署的检测器）就优于所有专门构建的信号（AUROC 0.696），因为69%的标注不忠实情况发生在错误答案上。按正确性分层将检测分为两种模式：在正确答案上，行为信号能适度区分忠实推理与事后推理（0.63-0.67）；在错误答案上（大多数不忠实情况所在），没有任何测试信号能显著高于随机水平（在四个模型上对基准范围信号进行了重复验证）。标准的步骤移除度量与人类标签呈反相关；这种反转在基准的重新评估中得以重现。

    arXiv:2607.23458v2 Announce Type: replace  Abstract: Chain-of-thought (CoT) explanations support oversight only if they are faithful: the stated reasoning must actually produce the answer. Auditing black-box (behavioral) detection of unfaithful CoT against FaithCoT-Bench's human annotations, we find answer correctness structures the problem at every level. Answer incorrectness alone (an oracle diagnostic, not a deployable detector) outperforms every purpose-built signal (AUROC 0.696), because 69% of annotated unfaithfulness occurs on incorrect answers. Stratifying by correctness splits detection into two regimes: on correct answers, behavioral signals moderately separate faithful from post-hoc reasoning (0.63-0.67); on incorrect answers, where most unfaithfulness lives, no tested signal is detectably above chance (replicated on all four models for benchmark-wide signals). The standard step-removal metric anti-correlates with human labels; this inversion reproduces on the benchmark's re
    
[^243]: CMI-Mem：通过CMI增强强化学习实现通用长期记忆管理

    CMI-Mem: Toward Generalizable Long-Term Memory Management via CMI-Augmented Reinforcement Learning

    [https://arxiv.org/abs/2607.20553](https://arxiv.org/abs/2607.20553)

    本文提出CMI-Mem，一种结合外在QA奖励和内在条件互信息奖励的轻量级强化学习记忆管理器，通过逐操作监督实现更高效、更通用的长期记忆管理。

    

    摘要：arXiv:2607.20553v2 公告类型：替换 摘要：记忆管理器模型在智能体系统中至关重要。现有的强化学习方法通常使用由LLM评判的合成问答（QA）对：这提供了有用的下游任务依据，但通过采样查询分布和固定阅读器来评估记忆价值。我们提出了CMI-Mem，一种具有混合奖励的轻量级强化学习记忆管理器。其外在QA项衡量端任务正确性，而内在条件互信息（CMI）项评估新对话输入相对于当前记忆状态所贡献的信息，无需依赖采样的QA查询。这两个信号互补：QA锚定任务效用，而CMI提供针对相关、非冗余记忆构建的逐操作监督。实验表明，在不同记忆使用场景中实现了改进的迁移能力，同时通过逐操作CMI信号实现了更高效的训练和推理。我们的代码可用。

    arXiv:2607.20553v2 Announce Type: replace  Abstract: Memory Manager models are pivotal in agent systems. Existing reinforcement-learning methods commonly use LLM-judged synthetic question-answer (QA) pairs: this provides useful downstream task grounding, but values memory through a sampled query distribution and a fixed reader. We propose CMI-Mem, a lightweight RL memory manager with a hybrid reward. Its extrinsic QA term measures end-task correctness, while its intrinsic Conditional Mutual Information (CMI) term evaluates the information contributed by new conversational inputs relative to the current memory state without conditioning on a sampled QA query. The two signals are complementary: QA anchors task utility, whereas CMI provides per-operation supervision for relevant, non-redundant memory construction. Experiments demonstrate improved transfer across memory-use scenarios, together with more efficient training and inference from the per-operation CMI signal. Our codes are avail
    
[^244]: Telco-GAIA：电信领域的双语智能体基准

    Telco-GAIA: Bilingual Benchmark for Agents in Telecom Domain

    [https://arxiv.org/abs/2607.20510](https://arxiv.org/abs/2607.20510)

    Telco-GAIA是一个双语多模态基准，用于评估电信领域工具使用智能体，通过100个人工验证的问答任务和多跳推理，在异构数据源上挑战模型性能，并采用客观的精确字符串匹配评分。

    

    我们介绍了Telco-GAIA，一个用于评估在真实世界电信运营商数据上使用工具智能体的双语、多模态基准。Telco-GAIA包含100个经人工验证的问答任务，涵盖英语和阿拉伯语，每个任务都需要对三个异构来源进行多跳推理（平均4.2跳）：静态网站快照（HTML、图像和链接的PDF）、合成关系SQL数据库和外部网络档案，跨越文本、图像和表格模态。该基准以沙盒化Docker环境交付，并通过归一化精确字符串匹配进行评分，使评估客观、确定且可随时间复现，无需任何LLM作为裁判。通过评估一个专门构建的参考智能体，在十二种商业和开源LLM上，我们发现Telco-GAIA具有挑战性：即使最强模型也只能解决71%的任务；在适度的成本预算下，这一比例降至约40%，并且视觉部分……

    arXiv:2607.20510v2 Announce Type: replace  Abstract: We introduce Telco-GAIA, a bilingual, multi-modal benchmark for evaluating tool-using agents on the data of a real-world telecommunications operator. Telco-GAIA comprises 100 human-verified question-answering tasks, in English and Arabic, that each demand multi-hop reasoning (4.2 hops on average) over three heterogeneous sources: a static website snapshot (HTML, images, and linked PDFs), a synthetic relational SQL database, and external web archives, spanning text, image, and tabular modalities. The benchmark is delivered as a sandboxed Docker environment and scored by normalized exact string matching, making evaluation objective, deterministic, and reproducible over time without any LLM-as-a-Judge. Evaluating a purpose-built reference agent across twelve commercial and open LLMs, we find Telco-GAIA challenging: even the strongest model solves only 71% of tasks; under a moderate cost budget, this falls to about 40%, and the visually 
    
[^245]: CANDOR：冻结基础编码器中的机会校准不一致性

    CANDOR: Chance-Calibrated Discordance in Frozen Foundation Encoders

    [https://arxiv.org/abs/2607.18451](https://arxiv.org/abs/2607.18451)

    本文提出CANDOR度量，通过等大小对称样本库校正最近邻不一致性，将机会水平固定为二分之一，揭示冻结编码器并非失明但普遍性能较弱。

    

    摘要：arXiv:2607.18451v2 公告类型：替换-交叉 摘要：冻结编码器的选择取决于轻量级头部从其特征中读取发现的能力，而非几何结构是否将其分离。最近邻不一致性可以做到这一点，但在样本库不均衡的情况下，相反标签的邻居会因密度而非几何结构获胜，因此仅凭患病率就会使无信息编码器看起来失明。我们引入了CANDOR，一种不一致性度量，其等大小样本库在标签交换下对称，将机会水平精确固定在二分之一。在22个编码器、来自7个领域的20个数据集和605,443张图像上，这一修正逆转了结论。崩溃几乎在所有地方都低于机会水平，因此没有编码器是失明的，但所有编码器都较弱：最佳胸部模型以84.5 AUROC读取气胸，但仍将18.4%的阳性样本放置在比同医院同类更接近相反标签的影像附近。同一个在鸟类物种分辨上达到4.5的编码器，在胸部发现上为42.8，在青光眼上为49.8，处于机会水平或更差。

    arXiv:2607.18451v2 Announce Type: replace-cross  Abstract: Frozen encoders are chosen by how well a lightweight head reads a finding from their features, not whether the geometry separates it. Nearest-neighbor discordance does, but with unequal banks the opposite-label neighbor wins on density, not geometry, so prevalence alone makes an uninformed encoder look blind. We introduce CANDOR, a discordance measure whose equal-size banks are symmetric under a label swap, fixing its chance level at exactly one half. Across 22 encoders, 20 datasets from 7 domains, and 605,443 images, this correction reverses the conclusion. Collapse falls below chance almost everywhere, so no encoder is blind, yet all are weak: the best chest model reads pneumothorax at 84.5 AUROC and still places 18.4% of those positives nearer an opposite-label film than its own kind in the same hospital. The same encoder that resolves bird species at 4.5 leaves chest findings at 42.8 and glaucoma at 49.8, at chance and wors
    
[^246]: 一种针对KV缓存的JoLT方法：通过Tucker秩的联合拉格朗日分配和旋转残差实现大语言模型的近无损KV缓存压缩

    A JoLT for the KV cache: Near-lossless KV cache compression via joint Lagrangian allocation of Tucker ranks and a rotated residual for llms

    [https://arxiv.org/abs/2607.12550](https://arxiv.org/abs/2607.12550)

    本文提出JoLT方法，通过部分Tucker分解和旋转低比特残差，在保持头与层轴完整的同时压缩令牌和特征轴，实现KV缓存的近无损压缩。

    

    键值（KV）缓存已成为Transformer推理中的主要内存开销：它随批次大小、上下文长度和深度增长，在长上下文场景下，它而非模型权重决定了吞吐量的上限。现有的压缩方法分为两类。低秩方法对缓存的二维切片进行分解，可以是每个头的矩阵或跨层的特征块，而量化方法则降低每个条目的位宽。这两类方法都没有利用缓存在一层中天然是三阶张量这一事实，其三个轴——头、令牌和特征——携带的冗余量差异很大。我们直接采用这种张量视图。我们的方法JoLT（联合拉格朗日Tucker）应用部分Tucker分解，仅压缩令牌和特征轴，同时保留头和层轴不变，然后通过旋转的低位残差恢复截断所丢弃的能量：一个随机或...

    arXiv:2607.12550v3 Announce Type: replace-cross  Abstract: The key-value (KV) cache has become the dominant memory cost of transformer inference: it grows with batch size, context length, and depth, and at long context it, rather than the model weights, sets the throughput ceiling. Existing reductions fall into two families. Low-rank methods factor two-dimensional slices of the cache, either per-head matrices or cross-layer feature blocks, and quantization methods lower the bit-width of every entry. Neither exploits the fact that the cache at a layer is naturally a third-order tensor whose three axes, the heads, the tokens, and the features, carry very different amounts of redundancy. We take this tensor view directly. Our method, JoLT (Joint Lagrangian Tucker), applies a partial Tucker decomposition that compresses only the token and feature axes while leaving the head and layer axes intact, then restores the energy that truncation discards with a rotated low-bit residual: a random or
    
[^247]: DiaLLM：英语方言适应中鲁棒性与生成能力差距的探究

    DiaLLM: An Investigation into the Robustness-Generation Gap in English Dialect Adaptation

    [https://arxiv.org/abs/2607.07669](https://arxiv.org/abs/2607.07669)

    本文发现方言理解与生成能力在LLM中分离，并证明显式方言定向适应优于广泛对齐，但基准测试无法反映这一生成优势。

    

    大语言模型越来越能“理解”方言英语，但仍只能“生成”标准且偏向美式的英语，导致方言生成这一更难的问题在很大程度上未被解决。我们引入了DiaLLM，该方法对三个开放权重语言模型家族在国际英语语料库上进行持续预训练，并应用隐式和显式后训练范式，每种范式结合三种模型对齐策略，首次对这些组件在澳大利亚、印度和北英格兰英语上的表现进行了受控比较。我们的结果显示，方言鲁棒性和生成能力是“分离”的：基准测试受持续预训练和SFT影响，而对齐则显著重塑生成方式，但基准测试无法捕捉这些变化。显式针对特定变体的适应能产生可靠被识别为方言的输出，且优于广泛对齐，但该方法在基准测试中的表现却未体现其优势。

    arXiv:2607.07669v2 Announce Type: replace-cross  Abstract: Large language models increasingly \emph{understand} dialectal English, yet still \emph{produce} only standard, US-leaning English, leaving dialectal generation, the harder half of the problem, largely unaddressed. We introduce \textbf{DiaLLM}, which continually pretrains three open-weight language model families on the International Corpus of English and applies implicit and explicit post-training paradigms, each combined with three model alignment strategies, giving the first controlled comparison of these components across Australian, Indian, and Northern British English. Our results reveal that dialectal robustness and generation are \emph{dissociated}: benchmarks are shaped by continual pretraining and SFT, while alignment visibly reshapes generation in ways benchmarks do not capture. Explicit variety-targeted adaptation produces output reliably recognised as dialectal and preferred over broad alignment, yet the method tha
    
[^248]: DynaKRAG：多跳检索增强生成中可学习证据控制的统一框架

    DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented Generation

    [https://arxiv.org/abs/2607.06507](https://arxiv.org/abs/2607.06507)

    DynaKRAG提出了一个统一框架，通过学习状态条件策略动态控制证据获取与答案生成，显著提升了多跳RAG的效率与准确性。

    

    多跳检索增强生成（RAG）顺序获取证据，每个文档贡献支持事实、桥梁实体、查询细化或回答问题的充分证据。证据获取可能涉及迭代检索、查询重构、证据评估和充分性检查。我们引入了DynaKRAG，一个统一的证据-动作框架，学习共享的状态条件策略来协调这些操作。在每一步，确定性有效性层构建可执行动作集，学习的延续门在答案生成和进一步证据获取之间选择，学习的优势评分器根据相对于即时答案生成的预测增益对可行证据操作进行排序。所选操作更新共享状态，并可能启用其他操作。在HotpotQA、2Wiki和MuSiQue上，使用Qwen2.5-7B、GPT-4o-mini和Llama-3.1-8B，Dy

    arXiv:2607.06507v2 Announce Type: replace  Abstract: Multi-hop retrieval-augmented generation (RAG) acquires evidence sequentially, with each document contributing supporting facts, bridge entities, query refinements, or sufficient evidence for answering. Evidence acquisition can involve iterative retrieval, query reformulation, evidence assessment, and sufficiency checking. We introduce DynaKRAG, a unified evidence-action framework that learns a shared state-conditioned policy for coordinating these operations. At each step, a deterministic validity layer constructs the executable action set, a learned continuation gate selects between answer generation and further evidence acquisition, and a learned advantage scorer ranks feasible evidence operations by their predicted gain relative to immediate answer generation. The selected operation updates the shared state and may enable additional operations. Across HotpotQA, 2Wiki, and MuSiQue with Qwen2.5-7B, GPT-4o-mini, and Llama-3.1-8B, Dy
    
[^249]: 大型语言模型所解释的并非其真实信念：在模型自身输入信念下评估解释充分性

    What LLMs explain is not what they believe: Evaluating explanation sufficiency under models' own input beliefs

    [https://arxiv.org/abs/2606.28615](https://arxiv.org/abs/2606.28615)

    本文提出了一种基于信息论的指标SCSuff，利用LLM自身生成替代输入来评估自由文本解释的充分性，无需预设偏见，并证明解释充分性依赖于输入分布。

    

    arXiv:2606.28615v2 公告类型：替换-交叉 摘要：大型语言模型（LLMs）越来越多地被部署在高风险领域，在这些领域中，自由文本解释（如思维链和事后理由）被用于证明模型输出的合理性。然而，这些解释是否充分仍不清楚，即它们是否包含足够的信息来解释模型的输出生成过程。我们将经典充分性从特征归因推广到任意解释，并证明解释充分性可能随输入分布而变化，这必须为LLM解释明确定义。我们提出利用LLM自身生成基于解释的替代输入，以捕捉其对可能输入的信念。我们将自洽充分性形式化为自由文本解释的目标，并引入一种信息论指标SCSuff，该指标能够在无需依赖预定义偏见或假设的情况下评估自由文本解释。

    arXiv:2606.28615v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly deployed in high-stakes domains, where free-text explanations such as chain-of-thought and post-hoc rationales are used to justify model outputs. Yet it remains unclear whether these explanations are sufficient, i.e., if they contain enough information to explain the model's output-generating process. We generalize classical sufficiency from feature attributions to arbitrary explanations and prove that explanation sufficiency can change depending on the input distribution, which must be explicitly defined for LLM explanations. We propose using the LLM itself to generate alternative inputs conditioned on an explanation, capturing its beliefs about possible inputs. We formalize self-consistent sufficiency as a goal for free-text explanations and introduce an information-theoretic metric, SCSuff, that enables evaluation of free-text explanations without relying on predefined biases or 
    
[^250]: 研究如何演进？通过基于主张的类型化引用追踪NLP、ML和CV中的跨领域轨迹

    How Does Research Evolve? Tracing Cross-Domain Trajectories in NLP, ML, and CV Through Claim-Grounded Typed Citations

    [https://arxiv.org/abs/2606.22342](https://arxiv.org/abs/2606.22342)

    本文提出了SciTraj，一个包含32,559篇论文和573,126条类型化引用边的语料库，通过主张驱动的六种研究关系和多步轨迹，实现了对NLP、ML和CV领域研究演进的细粒度追踪。

    

    arXiv:2606.22342v2 公告类型：替换 摘要：研究如何演进，我们能否在单个主张的层面上追踪它？科学进步并非简单的事实统一积累。现有的引用图通常将这些角色折叠为单一的同质边类型，限制了我们对科学进步的分析能力。我们引入了SciTraj，一个用于追踪自然语言处理、机器学习和计算机视觉领域研究演进的类型化引用语料库。SciTraj包含2015年至2024年间发表的32,559篇论文和573,126条有向边，涵盖六种研究关系类型。与传统引用图不同，每条边都配有一个主张句子，该句子为其标签提供动机。主张驱动的关系通过自然语言推理在其局部论文上下文中进行验证。该语料库进一步将这些关系组织成多步类型化轨迹，以追踪思想如何在论文和时间中发展。我们沿三个维度评估了该语料库。

    arXiv:2606.22342v2 Announce Type: replace  Abstract: How does research evolve, and can we trace it at the level of individual claims? Scientific progress is not simply a uniform accumulation of facts. Existing citation graphs usually collapse these roles into a single homogeneous edge type, limiting how we can analyze scientific progress. We introduce SciTraj, a typed citation corpus for tracing research evolution across natural language processing, machine learning, and computer vision. SciTraj includes 32,559 papers published between 2015 and 2024 and 573,126 directed edges spanning six research-relation types. Unlike traditional citation graphs, each edge is paired with the claim sentence that motivates its label. Claim-driven relations are verified by natural language inference against their local in-paper context. The corpus further organizes these relations into multi-step typed trajectories that trace how ideas develop across papers and over time. We evaluate the corpus along th
    
[^251]: 在位者优势：大语言模型推荐系统中的品牌偏见与认知操纵动态

    Incumbent Advantage: Brand Bias and Cognitive Manipulation Dynamics in LLM Recommendation Systems

    [https://arxiv.org/abs/2606.17443](https://arxiv.org/abs/2606.17443)

    本研究揭示了LLM推荐系统中知名品牌的“条件性垄断”现象，并发现权威式营销语言（包括虚假临床声明）能以微小评分点优势打破这种垄断，且不同模型反应各异。

    

    大语言模型（LLMs）正成为消费者寻找产品的主要途径，但我们尚不了解品牌在这一新渠道中如何竞争。我们使用护肤品——一个消费者在购买前难以判断质量、必须依赖品牌声誉的品类——研究了LLM推荐中的品牌动态，涉及三个商业LLM（GPT-4o-mini、Claude Sonnet、Gemini 3 Flash），并对搜索品进行了稳健性检验。在三项实验中，我们发现：（1）条件性垄断：当所有产品具有相同规格时，知名品牌100%被推荐（IAI = 10.0），但这种主导地位在竞争对手仅获得不到+0.1星的评分优势时消失；（2）权威式营销语言，包括捏造的临床证据声明，以相当于+0.17评分点的偏见盈余价值打破这种垄断，且每个模型反应不同；（3）一个社会困境。

    arXiv:2606.17443v2 Announce Type: replace  Abstract: Large language models (LLMs) are becoming a major way for consumers to find products, but we do not yet understand how brands compete in this new channel. We study brand dynamics in LLM recommendations using skincare products -- a category where consumers cannot easily judge quality before buying and must rely on brand reputation -- across three commercial LLMs (GPT-4o-mini, Claude Sonnet, Gemini 3 Flash), with a robustness check on search goods. In three experiments, we find: (1) a Conditional Monopoly where well-known brands get recommended 100% of the time (IAI = 10.0) when all products have the same specifications, but this dominance disappears with less than a +0.1-star rating advantage for a competitor; (2) authority-style marketing language, including fabricated clinical-evidence claims, breaks this monopoly at a Bias Surplus Value equal to +0.17 rating points, with each model responding differently; and (3) a social dilemma i
    
[^252]: SCAR：面向RAG高效上下文扩展的语义连续性感知检索

    SCAR: Semantic Continuity-Aware Retrieval for Efficient Context Expansion in RAG

    [https://arxiv.org/abs/2606.16661](https://arxiv.org/abs/2606.16661)

    SCAR通过基于查询相关性的自适应阈值和连续性惩罚机制，在减少令牌开销的同时有效解决了RAG中固定分块导致的边界碎片化问题，实现了跨模型的近似尺度不变的检索性能提升。

    

    在检索增强生成（RAG）中，固定长度分块常导致边界碎片化，即关键证据被分割到不同片段中，从而降低检索召回率。虽然静态窗口和父检索能提升召回率，但会引入显著的令牌开销。我们提出SCAR（语义连续性感知检索），一种自适应检索策略，通过权衡查询-邻居相关性与结构连续性惩罚，选择性扩展相邻块。SCAR使用相对扩展阈值，该阈值与每个检索块自身的查询相关性相关联，形成近似尺度不变的决策规则，可在不同嵌入模型间迁移而无需重新校准。在四个多样化的语料库（RFC、GDPR、一份10-K报告和一份并购协议；N=320个查询；其中160个为边界碎片化）上，SCAR在边界碎片化查询上实现了92.8%的召回率，仅使用7.84个块，与静态窗口相比减少了22.9%。

    arXiv:2606.16661v2 Announce Type: replace-cross  Abstract: Fixed-length chunking in Retrieval-Augmented Generation (RAG) often leads to boundary fragmentation, where critical evidence is split across segments, degrading retrieval recall. While static windowing and parent retrieval improve recall, they introduce significant token overhead. We propose SCAR (Semantic Continuity-Aware Retrieval), an adaptive retrieval policy that selectively expands neighboring chunks by weighing query-neighbor relevance against a structural continuity penalty. SCAR uses a relative expansion threshold tied to each retrieved chunk's own query-relevance, yielding an approximately scale-invariant decision rule that transfers across embedding models without recalibration. Across four diverse corpora (RFC, GDPR, a 10-K report, and a Merger agreement; N=320 queries; 160 boundary-fragmented), SCAR achieves 92.8% recall on boundary-fragmented queries with only 7.84 chunks, a 22.9% reduction compared to static wind
    
[^253]: GRACE：基于上下文忠实推理的步骤级基准

    GRACE: Step-Level Benchmark for Faithful Reasoning over Context

    [https://arxiv.org/abs/2606.16151](https://arxiv.org/abs/2606.16151)

    本文提出了GRACE，一个首个基于人工注释的步骤级忠实性基准，通过数据驱动的错误分类法精确定位推理链中的失败步骤及其类型，解决了现有方法仅在响应级别检测幻觉的局限。

    

    arXiv:2606.16151v2 公告类型：替换 摘要：许多推理任务要求模型在输入上下文上进行推理，从基于文档的问答到基于规则的演绎。链式思维（CoT）提示生成的轨迹看似透明，但单个步骤可能悄然偏离源证据，即使最终答案是正确的。现有方法在响应级别检测幻觉，但无法识别链条中失败发生的位置或其类型。我们引入了GRACE，这是首个带有人工注释的步骤级忠实性基准，并配有数据驱动的错误分类法，用于基于上下文的文本推理。GRACE涵盖了来自4个源数据集的10个模型的CoT轨迹，每个步骤都注释了忠实性、错误类别和自然语言解释。通过无监督聚类自下而上发现的数据驱动分类法，将失败组织为两个轨道：GRACE-Inference（演绎错误）和GRACE-Grounding（事实基础）。

    arXiv:2606.16151v2 Announce Type: replace  Abstract: Many reasoning tasks require models to reason over input context, from document-grounded question answering to rule-based deduction. Chain-of-Thought (CoT) prompting produces traces that appear transparent, yet individual steps can silently deviate from the source evidence, even when the final answer is correct. Existing methods detect hallucinations at the response level but fail to identify where in the chain a failure occurs or what type it is. We introduce GRACE, the first human-annotated step-level faithfulness benchmark with a data-driven error taxonomy for context-grounded textual reasoning. GRACE covers CoT traces from 10 models across 4 source datasets, with each step annotated for faithfulness, error category, and natural language explanation. A data-driven taxonomy, discovered bottom-up via unsupervised clustering, organizes failures into two tracks: GRACE-Inference (deductive errors) and GRACE-Grounding (factual grounding
    
[^254]: 最后但同样重要：多模态KV缓存压缩中的边界注意力校准

    Last But Not Least: Boundary Attention CalibratiON for Multimodal KV Cache Compression

    [https://arxiv.org/abs/2606.14782](https://arxiv.org/abs/2606.14782)

    BACON通过结合最后查询注意力与观察窗口注意力，并抑制噪声，显著提升了多模态KV缓存压缩的准确性，尤其在激进压缩场景下平均提升7.5%。

    

    arXiv:2606.14782v3 公告类型：替换交叉 摘要：多模态大型语言模型（MLLMs）在视觉-语言推理方面表现强劲，但在长视觉上下文下会产生大量KV缓存和高解码延迟。现有压缩方法依赖观察窗口注意力来稳定估计令牌重要性，但这种聚合可能稀释稀疏的关键证据，并在激进压缩下丢弃与答案相关的令牌。我们识别出最后查询注意力作为恢复此类证据的补充信号，尽管其不相关信号可能引入额外噪声。我们提出BACON，一种即插即用方法，通过层内一致性和层间持久性校准观察窗口注意力与最后查询证据，同时抑制噪声。在不同基准、模型、预算和压缩方法下，BACON在最激进预算下平均将多模态KV缓存压缩提升7.5%，最高提升达30.9%。

    arXiv:2606.14782v3 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) achieve strong vision-language reasoning but incur large KV caches and high decoding latency with long visual contexts. Existing compression methods rely on observation window attention for stable token importance estimation, yet this aggregation can dilute sparse critical evidence and discard answer-relevant tokens under aggressive compression. We identify last query attention as a complementary signal for recovering such evidence, though its irrelevant signals may introduce additional noise. We propose BACON, a plug-and-play method that calibrates observation window attention with last query evidence while suppressing noise through intra-layer coherence and inter-layer persistence. Across diverse benchmarks, models, budgets, and compression methods, BACON improves multimodal KV-cache compression by 7.5% on average under the most aggressive budget, with gains up to 30.9%.
    
[^255]: 一页污染就足够：评估LLM推荐系统中的网页内容污染

    One Polluted Page Is Enough: Evaluating Web Content Pollution in LLM Recommenders

    [https://arxiv.org/abs/2606.13610](https://arxiv.org/abs/2606.13610)

    本研究首次系统评估了LLM推荐系统在生成引擎优化污染下的脆弱性，发现仅一个被污染的网页即可导致高达27%的虚假产品推荐率，而三个页面污染则使受骗率飙升至73.8%。

    

    摘要：检索增强的LLM越来越频繁地通过获取实时网页内容来调解日常消费者推荐。这带来了一种新风险：LLM推荐系统可能会消费生成引擎优化（GEO）操作者污染的内容，从而被误导。我们提出一个问题：它们在多大程度上成为虚假产品的无意推广者？我们引入了FORGE（生成环境中的虚假在线推荐）框架，该框架在固定的检索网页集合中局部地将真实产品重写为虚假产品，并测量LLM推荐虚假产品的频率，涵盖15个类别中的225个真实产品和5种消费者场景。在12个商业和开放权重的LLM中，所有模型都易受影响：单个污染页面导致受骗率高达27%，而完全替换前三个页面则使这一比率升至73.8%。不同类别的易感性各不相同，当模型缺乏对产品的稳定先验知识时，易感性会增加。推理并不能完全消除这一风险。

    arXiv:2606.13610v2 Announce Type: replace-cross  Abstract: Search-augmented LLMs increasingly mediate everyday consumer recommendations by retrieving live web content. This creates a new risk: LLM recommenders may consume web content that Generative Engine Optimization (GEO) operators have polluted to mislead them. We ask: to what extent do they become unwitting promoters of fake products? We introduce FORGE (Fake Online Recommendations in Generative Environments), which locally rewrites real products in a frozen set of retrieved web pages into fake ones and measures how often the LLM recommends the fake product, across 225 real products in 15 categories and 5 consumer scenarios. Across 12 commercial and open-weights LLMs, all models are vulnerable: a single polluted page yields fooled rates of up to 27%, while the full top-3 replacement raises this to 73.8%. Vulnerability varies across categories, increasing when models lack stable prior knowledge of the products. Reasoning does not m
    
[^256]: TRACE：一种面向高效智能体强化学习的统一回滚预算分配框架

    TRACE: A Unified Rollout Budget Allocation Framework for Efficient Agentic Reinforcement Learning

    [https://arxiv.org/abs/2606.11119](https://arxiv.org/abs/2606.11119)

    本文提出TRACE框架，通过将回滚预算分配从提示级别扩展到轮次前缀级别，利用多轮强化学习中的细粒度信息差异，从而提升智能体训练效率。

    

    摘要：arXiv:2606.11119v2 公告类型：替换交叉发布  摘要：基于可验证奖励的强化学习（RLVR）是增强大型语言模型推理和智能体行为的一种有前景的方法。然而，回滚密集型的策略优化常常受到奖励对比度不足的限制，这种情况在过于简单或复杂的提示生成低方差反馈时出现，并且当仅基于结果的奖励为多轮回滚中的每个决策分配相同的终端评估时也会发生。以往的努力侧重于将可用的回滚资源分配给有潜力的提示，但仅利用了提示级别的样本信息量，忽略了同一回滚中不同轮次前缀级别的信息量变化。本研究针对多轮智能体强化学习，将每个ReAct风格的思考-行动-观察轮次建模为语义上不同的节点，使预算分配能够从提示根节点扩展到轮次级前缀及其后续延续，这自然形成了...

    arXiv:2606.11119v2 Announce Type: replace-cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) is a promising approach for enhancing reasoning and agentic behavior in large language models. However, rollout-intensive policy optimization is often limited by insufficient reward contrast, arising when overly simple or complex prompts generate low-variance feedback and when outcome-only rewards assign the same terminal assessment to every decision in a multi-turn rollout. Past efforts have focused on allocating available rollout resources to promising prompts, yet they only leverage sample informativeness at the prompt level and neglect variation in prefix-level informativeness across turns within the same rollout. This work targets multi-turn agentic RL by modeling each ReAct-style thought-action-observation turn as a semantically distinct node, allowing budget allocation to extend from prompt roots to turn-level prefixes with further continuations, which naturally forms
    
[^257]: SocraticPO：通过交互式引导进行策略优化

    SocraticPO: Policy Optimization via Interactive Guidance

    [https://arxiv.org/abs/2606.09887](https://arxiv.org/abs/2606.09887)

    SocraticPO通过引入教师引导和奖励衰减机制，使强化学习中的语言模型在错误推理时获得可解释的修正指导，避免捷径学习，从而提升策略的鲁棒性。

    

    大型语言模型的强化学习通常使用标量结果奖励（如二元正确性）来监督推理过程。这类奖励提供了优化方向，但很少解释模型应如何修正其错误推理，这可能鼓励捷径学习和脆弱策略。我们提出SocraticPO（苏格拉底式策略优化），一种策略优化框架，将苏格拉底式自然语言引导增强到强化学习回滚中。在回滚过程中，学生首先独立作答；如果答案不正确，教师会诊断尝试并提供简洁的纠正性引导，之后学生在扩展上下文下继续。关键在于，这种引导与奖励衰减配对：在教师干预后获得的正确回答仅获得衰减奖励，防止策略将教师帮助视为获取奖励的免费途径。由于SocraticPO仅修改...

    arXiv:2606.09887v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) for large language models usually supervises reasoning with scalar outcome rewards, such as binary correctness. Such rewards provide an optimization direction but rarely explain how a model should revise its mistaken reasoning, which can encourage shortcut learning and brittle policies. We propose \textbf{SocraticPO} (Socratic Policy Optimization), a policy-optimization framework that augments RL rollouts with Socratic-style natural-language guidance. During rollout, the student first answers independently; if the answer is incorrect, a teacher diagnoses the attempt and provides concise corrective guidance, after which the student continues under the expanded context. Crucially, this guidance is paired with reward decay: correct answers obtained after teacher intervention only receive decayed rewards, preventing the policy from treating teacher help as a free path to reward. Since SocraticPO only mod
    
[^258]: 临床接地气的医学语言模型隐私评估

    Clinically Grounded Privacy Evaluation of Medical LMs

    [https://arxiv.org/abs/2606.09590](https://arxiv.org/abs/2606.09590)

    该论文提出一个临床接地气的隐私评估框架，揭示医学语言模型在常规就诊元数据下能高比例逐字记忆患者信息并恢复敏感诊断，同时指出精确匹配记忆可能高估泄露风险。

    

    摘要：arXiv:2606.09590v2 公告类型：替换 摘要：医学语言模型（LMs）可能记忆并复现受保护的健康信息，但隐私评估往往侧重于训练文本的恢复，而非在现实威胁模型下的泄露情况。我们引入了一个临床接地气的框架，该框架沿着对抗性访问的分级轴评估泄露，范围从可公开推断的人口统计学信息到泄露的笔记片段。在每个层级，我们测量患者特定文本的逐字记忆以及敏感诊断的语义泄露。将该框架应用于一个在378k份临床笔记上持续预训练的LM，我们发现常规就诊元数据（即姓名、出生日期、就诊日期、提供者姓名和执业地点）在患者时间线上引发高比例的逐字记忆，以及敏感诊断恢复（流产的AUROC为0.91，HIV为0.82）。同时，精确匹配记忆可能夸大泄露：36%的记忆标记反映...

    arXiv:2606.09590v2 Announce Type: replace  Abstract: Medical language models (LMs) can memorize and reproduce protected health information, but privacy evaluations often focus on recovery of training text rather than disclosure under realistic threat models. We introduce a clinically grounded framework that evaluates leakage along a graded axis of adversarial access, ranging from publicly inferable demographics to leaked note fragments. At each tier, we measure verbatim memorization of patient-specific text and semantic leakage of sensitive diagnoses. Applying the framework to an LM continually pretrained on 378k clinical notes, we find that routine encounter metadata (i.e. name, date of birth, visit date, provider name, and practice location) elicits high rates of verbatim memorization across a patient's timeline and sensitive-diagnosis recovery (AUROC 0.91 for abortion, 0.82 for HIV). At the same time, exact-match memorization can overstate disclosure: 36% of memorized tokens reflect
    
[^259]: VCIFBench：评估视频理解中的复杂指令遵循能力

    VCIFBench: Evaluating Complex Instruction Following for Video Understanding

    [https://arxiv.org/abs/2606.04588](https://arxiv.org/abs/2606.04588)

    该论文提出了VCIFBench，一个用于评估视频理解中复杂指令遵循能力的新基准，包含约束丰富的指令和混合验证流程，并发现联合约束满足对现有模型仍具挑战性。

    

    多模态大语言模型在视频理解方面取得了快速进展，然而现有基准测试主要依赖简单提示，对于模型能否满足明确的输出约束提供的证据有限。我们引入了VCIFBench，一个用于评估视频理解中复杂指令遵循能力的基准。VCIFBench通过改编基准和直接基于视频的提示构建了富含约束的指令，涵盖内容、格式、风格和结构要求，并使用混合验证流程评估模型输出。该基准包含306条可满足的测试指令、540个DPO训练实例，以及一个用于评估模型能否识别指令冲突的100项诊断集。对10个MLLM的实验表明，联合约束满足仍然具有挑战性。偏好优化改善了两个模型家族的指令遵循能力，而Conflict-100则揭示了相关问题。

    arXiv:2606.04588v2 Announce Type: replace  Abstract: Multimodal large language models have made rapid progress in video understanding, yet existing benchmarks largely rely on simple prompts and provide limited evidence about whether models can satisfy explicit output constraints. We introduce VCIFBench, a benchmark for evaluating complex instruction following in video understanding. VCIFBench constructs constraint-rich instructions from both benchmark-adapted and directly video-grounded prompts, covering content, format, style, and structure requirements, and evaluates model outputs with a hybrid verification pipeline. The benchmark contains 306 satisfiable test instructions, 540 DPO training instances, and a 100-item diagnostic set for evaluating whether models can recognize instruction conflicts. Experiments on 10 MLLMs show that joint constraint satisfaction remains challenging. Preference optimization improves instruction following for two model families, while Conflict-100 reveals
    
[^260]: 深层网络中的值向量是否需要来自残差流的上下文？

    Do Value Vectors in Deep Layers Need Context from the Residual Stream?

    [https://arxiv.org/abs/2606.02780](https://arxiv.org/abs/2606.02780)

    本文发现深层网络中的无上下文值向量能显著提升模型性能，并可稀疏存储以提高效率。

    

    arXiv:2606.02780v4 公告类型：替换 摘要：Transformer架构作为现代LLM骨干的成功，在很大程度上归功于其注意力层的使用。注意力层遵循标准神经网络范式：它将残差流作为输入，从而生成依赖于上下文的查询、键和值向量。然而，我们发现，当深层网络仅学习一个无上下文的值向量以保留原始令牌信息，而不从残差流中提取任何上下文时，模型性能会显著提升。当模型拥有这种无上下文的值向量时，再添加上下文依赖的组件对整体基准性能的额外益处微乎其微。这种无上下文的值向量可以存储为稀疏模型参数，从而无需重新计算或持久缓存这些值。通过对这种无上下文值向量的关键设计选择进行系统性消融实验，我们提出了B方案。

    arXiv:2606.02780v4 Announce Type: replace  Abstract: The success of the transformer architecture as the backbone of modern LLMs is in large part due to its use of attention layers. An attention layer follows the standard neural network paradigm: it takes the residual stream as input and thereby produces context-dependent query, key, and value vectors. However, we find that model performance meaningfully improves when deeper layers learn only a context-free value vector to preserve the original token information, without drawing on any context from the residual stream. When the model has access to this context-free value vector, adding back the context-dependent component provides little additional benefit for aggregate benchmark performance. Such context-free value vectors can be stored as sparse model parameters, eliminating the need to recompute or persistently cache these values. Through systematic ablations on the key design choices for such context-free value vectors, we propose B
    
[^261]: 相同载荷，不同渠道：衡量工具使用型语言模型中的信任不对称性

    Same Payload, Different Channel: Measuring Trust Asymmetry in Tool-Using Language Models

    [https://arxiv.org/abs/2606.00566](https://arxiv.org/abs/2606.00566)

    本文提出安全不对称分数（SAS），发现通用型语言模型对工具元数据中的恶意指令敏感度远低于用户消息，而代理原生型模型差异较小，揭示了信任渠道的不对称性。

    

    arXiv:2606.00566v2 公告类型：替换交叉 摘要：随着语言模型承担起调用API、读取工具输出并对第三方内容采取行动的代理角色，其攻击面已扩展到用户输入之外。无论恶意指令来自何处，模型是否以相同方式处理，尚未得到系统研究。我们引入了安全不对称分数（SAS），通过匹配载荷对（恶意文本相同，仅渠道变化），衡量模型对对抗性内容的敏感度如何随其到达用户消息、工具元数据或工具输出的不同而改变。在10个生产级LLM和三种攻击家族中，通用型模型对作为工具元数据到达的指令的敏感度远低于用户消息中的相同指令，而代理原生型模型的敏感度差异则小得多。这种差异在控制工具可用性和评分的匹配控制实验以及大小控制混合中依然存在。

    arXiv:2606.00566v2 Announce Type: replace-cross  Abstract: As language models take on agentic roles that call APIs, read tool outputs, and act on third-party content, their attack surface expands beyond what users type. Whether they treat a malicious instruction the same way regardless of where it arrives has not been studied systematically. We introduce the Safety Asymmetry Score (SAS), measuring how a model's susceptibility to adversarial content shifts depending on whether it arrives in the user message, tool metadata, or tool output, using matched payload pairs that hold the malicious text identical and vary only the channel. Across 10 production LLMs and three attack families, general-purpose models sharply discount instructions arriving as tool metadata relative to identical instructions in the user message, while agent-native models discount them far less. This differential survives an affordance-matched control equalizing tool availability and scoring, and a size-controlled mix
    
[^262]: RASET：路由无关的安全关键专家微调暴露了混合专家大语言模型中局部化的安全执行失效

    RASET: Router-Agnostic Safety-Critical Expert Tuning Exposes Localized Safety Enforcement Failures in Mixture-of-Experts LLMs

    [https://arxiv.org/abs/2605.29708](https://arxiv.org/abs/2605.29708)

    本文提出RASET框架，通过对比路由敏感性识别并微调安全关键专家，揭示了MoE大语言模型中安全执行可在不改变路由路径的情况下被局部化地破坏。

    

    混合专家（MoE）大语言模型依赖于稀疏的、由路由器驱动的专家激活，然而安全对齐与路由专家专业化之间的相互作用仍未被充分探索。一个常见的直觉是，安全行为可能通过将有害请求路由到特定的拒绝导向专家来控制。在这项工作中，我们提供了实证证据，展示了一个不同的图景：对齐的MoE大语言模型中的路由模式主要受主题驱动，而安全行为可以在几乎不改变模型内在路由路径的情况下被改变。基于这一观察，我们提出了RASET（路由无关的安全关键专家微调），这是一个红队框架，用于探测在少数专家子集中局部化的安全执行，同时保持模型的内在路由行为。RASET通过对比路由敏感性标准识别安全关键专家，并仅对选定的专家应用参数高效微调，从而最小化对整体路由的影响。

    arXiv:2605.29708v2 Announce Type: replace  Abstract: Mixture-of-Experts (MoE) LLMs rely on sparse, router-driven expert activation, yet how safety alignment interacts with routed expert specialization remains underexplored. A common intuition is that safety behavior may be controlled by routing harmful requests to distinct refusal-oriented experts. In this work, we provide empirical evidence for a different picture: routing patterns in aligned MoE LLMs are largely topic-driven, while safety behavior can be altered with little change to the model's intrinsic routing path.   Motivated by this observation, we present RASET (Router-Agnostic Safety-Critical Expert Tuning), a red-teaming framework that probes safety enforcement that is localized in a small subset of experts while preserving the model's intrinsic routing behavior. RASET identifies safety-critical experts via a contrastive routing-sensitivity criterion and applies parameter-efficient tuning only to the selected experts, minimi
    
[^263]: GIM：通过整合多种认知领域的任务来评估模型

    GIM: Evaluating models via tasks that integrate multiple cognitive domains

    [https://arxiv.org/abs/2605.18663](https://arxiv.org/abs/2605.18663)

    本文提出GIM基准，通过要求模型在广泛知识基础上整合多种认知操作来评估能力，避免知识记忆与抽象推理的偏差，强调实际任务中的推理接地性。

    

    摘要：随着大型语言模型（LLM）基准测试趋于饱和，评估社区采取了两种策略来增加难度：提升知识需求（如GPQA、HLE）或完全移除知识以偏向抽象推理（如ARC-AGI）。前者将记忆与能力混为一谈；后者则使推理脱离其实际应用场景。我们采用了不同的方法。接地整合度量（GIM）是一个包含820个原创问题（615个公开，205个私有）的基准测试，其难度源于整合；每个问题要求协调多种认知操作（如约束满足、状态跟踪、认知警惕、受众校准），并基于广泛可获取的知识，从而使推理保持在实际任务中接地，而不受限于专业专长。每个问题都是专家原创的组成，大部分采用评分标准分解计分。我们校准了一个评判器感知的连续评分体系。

    arXiv:2605.18663v2 Announce Type: replace  Abstract: As LLM benchmarks saturate, the evaluation community has pursued two strategies to increase difficulty: escalating knowledge demands (GPQA, HLE) or removing knowledge entirely in favor of abstract reasoning (ARC-AGI). The first conflates memorization with capability; the second divorces reasoning from the practical contexts in which it matters. We take a different approach. The Grounded Integration Measure (GIM) is a benchmark of 820 original problems (615 public, 205 private) where difficulty comes from integration; individual problems require coordinating multiple cognitive operations (constraint satisfaction, state tracking, epistemic vigilance, audience calibration) over broadly accessible knowledge, so that reasoning stays grounded in realistic tasks without being gated on specialized expertise. Each problem is an original expert-authored composition, majority with rubric-decomposed scoring. We calibrate a judge-aware continuous
    
[^264]: SafeLens：快速与慢速筛选的审慎高效视频护栏

    SafeLens: Deliberate and Efficient Video Guardrails with Fast-and-Slow Screening

    [https://arxiv.org/abs/2605.17610](https://arxiv.org/abs/2605.17610)

    SafeLens通过快速与慢速推理架构，动态分配计算资源，并利用影响引导过滤构建高质量数据集，实现了高效且准确的视频内容审核。

    

    arXiv:2605.17610v2 公告类型：替换-交叉 摘要：在线视频平台和AI生成内容的快速增长，使得可靠的视频护栏成为安全性和实际部署的关键挑战。虽然大多数视频可以通过快速模式识别进行筛选，但一小部分需要对时间复杂内容和细微政策约束进行更深层次的推理。现有方法通常对所有输入统一应用大型视觉-语言模型，导致高推理成本和计算分配效率低下。我们提出了SafeLens，一个视频护栏框架，引入了快速与慢速推理架构，以实现高效且准确的内容审核，并根据输入变化计算成本。此外，我们通过将影响引导过滤应用于SafeWatch数据集，构建了一个高质量数据集，仅保留原始数据的2.4%。为进一步解决训练时扩展的局限性，我们启用了测试时……

    arXiv:2605.17610v2 Announce Type: replace-cross  Abstract: The rapid growth of online video platforms and AI-generated content has made reliable video guardrails a key challenge for safety and real-world deployment. While most videos can be screened through fast pattern recognition, a small subset requires deeper reasoning over temporally complex content and nuanced policy constraints. Existing approaches typically rely on large vision-language models applied uniformly across all inputs, resulting in high inference costs and inefficient allocation of computation. We propose SafeLens, a video guardrail framework that introduces a fast-and-slow inference architecture for efficient and accurate content moderation with variable computational cost across inputs. Additionally, we construct a high-quality dataset by applying influence-guided filtering to the SafeWatch Dataset, retaining only 2.4% of the original data. To further address limitations of training-time scaling, we enable test-tim
    
[^265]: CogniFold：通过认知折叠实现始终在线的主动记忆

    CogniFold: Always-On Proactive Memory via Cognitive Folding

    [https://arxiv.org/abs/2605.13438](https://arxiv.org/abs/2605.13438)

    本文提出CogniFold，一种受大脑启发的“始终在线”代理记忆，通过扩展CLS理论至三层并利用图拓扑自组织，实现从事件流中自主构建和更新持久认知结构，从而推动主动式智能代理的发展。

    

    现有代理记忆主要停留在反应性和基于检索的层面，缺乏自主将经验组织为持久认知结构的能力。为迈向真正自主的代理，我们提出了CogniFold，一种受大脑启发的“始终在线”代理记忆，专为下一代主动助手设计。CogniFold持续将碎片化事件流折叠成自涌现的认知结构，从传入事件和积累的知识中逐步引导出更高层次的认知。我们通过将互补学习系统（CLS）理论从两层（海马体、新皮层）扩展为三层，增加了前额叶意图层来夯实这一基础。模拟前额叶皮层作为意图控制和决策的中心，CogniFold通过图拓扑自组织实现：认知结构在事件流下主动组装，语义相似时合并，过时时衰减，重新...（摘要被截断）

    arXiv:2605.13438v5 Announce Type: replace  Abstract: Existing agent memory remains predominantly reactive and retrieval-based, lacking the capacity to autonomously organize experience into persistent cognitive structure. Toward genuinely autonomous agents, we introduce CogniFold, a brain-inspired "always-on" agent memory designed for the next generation of proactive assistants. CogniFold continuously folds fragmented event streams into self-emerging cognitive structures, bootstrapping progressively higher-level cognition from incoming events and accumulated knowledge. We ground this by extending Complementary Learning Systems (CLS) theory from two layers (hippocampus, neocortex) to three, adding a prefrontal intent layer. Emulating the prefrontal cortex as the locus of intentional control and decision-making, CogniFold achieves this through graph-topology self-organization: cognitive structures proactively assemble under the stream, merge when semantically similar, decay when stale, re
    
[^266]: 检查报告到行动指南：面向患者的多模态临床体检报告数据集及行动卡生成

    Checkup2Action: A Multimodal Clinical Check-up Report Dataset for Patient-Oriented Action Card Generation

    [https://arxiv.org/abs/2605.11533](https://arxiv.org/abs/2605.11533)

    本文提出了C2A数据集和Checkup2Action工作流程，用于从多模态临床体检报告自动生成结构化、面向患者的行动卡，填补了报告到行动能力缺乏基准测试的空白。

    

    arXiv:2605.11533v4 公告类型：替换。摘要：常规临床体检报告结合了实验室测量、生理评估、影像发现和视觉结构化信息，但很少告知患者下一步该做什么。将这些报告转化为后续行动，需要模型跨页面、表格和模态连接证据，识别临床相关问题，并在没有未经支持的诊断或治疗声明的情况下传达后续步骤。然而，这种从报告到行动的能力仍然缺乏良好的基准测试。我们引入了C2A，一个用于从多模态体检报告生成结构化“行动卡”的数据集和基准，以及Checkup2Action，一个针对该任务的约束工作流程。C2A包含2,000份去标识化的真实世界报告，涵盖体格检查、实验室测试、心血管评估和影像证据。每张卡指定一个问题、其优先级、推荐的科室、随访窗口、面向患者的解释和问题。

    arXiv:2605.11533v4 Announce Type: replace  Abstract: Routine clinical check-up reports combine laboratory measurements, physiological assessments, imaging findings and visually structured information, but rarely tell patients what to do next. Translating them into follow-up actions requires models to connect evidence across pages, tables and modalities, identify clinically relevant issues and communicate next steps without unsupported diagnostic or treatment claims. Yet this report-to-action capability remains poorly benchmarked. We introduce C2A, a dataset and benchmark for generating structured \textit{Action Cards} from multimodal check-up reports, together with Checkup2Action, a constrained workflow for the task. C2A contains 2,000 de-identified real-world reports covering physical examinations, laboratory tests, cardiovascular assessments and imaging evidence. Each card specifies one issue, its priority, recommended department, follow-up window, patient-facing explanation and ques
    
[^267]: DeepRefine：通过强化学习进行智能体知识精炼

    DeepRefine: Agentic Knowledge Refinement via Reinforcement Learning

    [https://arxiv.org/abs/2605.10488](https://arxiv.org/abs/2605.10488)

    DeepRefine通过强化学习框架，利用多轮交互和溯因诊断，自动精炼知识库质量，以提升下游任务性能。

    

    外部知识使大型语言模型（LLM）智能体能够在开放式的、知识密集型的下游任务中，超越其内在参数记忆，将行动和决策基于现实世界。然而，底层知识库的质量受到不完整性、错误性或冗余性的系统性限制，表现为缺失证据或跨文档链接、低置信度或不精确的声明，以及模糊或共指消解问题。这些缺陷在迭代使用中会累积，降低检索保真度和下游任务性能。我们提出了DeepRefine，一个用于智能体知识精炼的强化学习框架，它通过用户查询来提升任何预先构建的结构化知识库（例如知识图谱或LLM维基）的质量，使其更适合下游任务。DeepRefine与知识库进行多轮交互，并进行溯因诊断。

    arXiv:2605.10488v2 Announce Type: replace-cross  Abstract: External knowledge enables large language model (LLM) agents to ground their actions and decisions beyond intrinsic parametric memory in open-ended, knowledge-intensive downstream tasks. Yet the quality of the underlying knowledge bases is systematically limited by incompleteness, incorrectness, or redundancy, manifested as missing evidence or cross-document links, low-confidence or imprecise claims, and ambiguous or coreference resolution issues. Such defects compound under iterative use, degrading retrieval fidelity and downstream task performance. We present \textbf{DeepRefine}, a reinforcement learning framework for agentic knowledge refinement that evolves the quality of any pre-constructed structured knowledge bases, e.g., knowledge graphs or LLM-Wikis, with user queries to make it more suitable for the downstream tasks. DeepRefine performs multi-turn interactions with the knowledge base and conducts abductive diagnosis o
    
[^268]: 心理上强烈，计算上隐形：大型语言模型生成能引发社会比较的帖子，却无法检测到它们

    Psychologically Potent, Computationally Invisible: LLMs Generate Social-Comparison-Eliciting Posts They Fail to Detect

    [https://arxiv.org/abs/2605.01017](https://arxiv.org/abs/2605.01017)

    本研究构建了小红书社会比较基准，发现LLM能生成引发社会比较的帖子，但基于提示的检测器难以稳定识别该信号，揭示生成与检测能力的不对称性。

    

    我们引入了小红书社会比较读者引发基准（XHS-SCoRE），这是一个基于读者视角的基准，用于检测纯文本的小红书（RedNote）帖子是否从第一人称读者视角引发向上、向下或无明确的社会比较。该任务针对一种具有社会意义的关系性、行为性真实信号，该信号不能简化为情感。在提示式LLM分类器和监督式中文编码器中，我们发现了一致的生成-检测不匹配：该信号在领域内是可文本学习的，但对基于提示的分类并不稳健。提示式LLM分类器表现出稳定的失败，特别是对引发比较的帖子进行中和，以及模型特定的方向性偏差。一项受控试点显示，LLM生成的小红书风格帖子可以改变感知地位和与比较相关的情感，即使基于提示的相同构建检测仍然脆弱。XHS-SCoRE贡献...

    arXiv:2605.01017v3 Announce Type: replace  Abstract: We introduce Xiaohongshu Social Comparison Reader Elicitation (XHS-SCoRE), a reader-grounded benchmark for detecting whether text-only Xiaohongshu (RedNote) posts elicit Upward, Downward, or Neutral/no clear social comparison from a first-person reader perspective. The task targets a socially meaningful relational, behaviorally real signal not reducible to sentiment. Across prompted LLM classifiers and supervised Chinese encoders, we find a consistent generation-detection mismatch: the signal is textually learnable in-domain, but not robustly accessible to prompt-based classification. Prompted LLM classifiers show stable failures, especially neutralization of comparison-eliciting posts and model-specific directional skew. A controlled pilot shows that LLM-generated Xiaohongshu-style posts can shift perceived standing and comparison-related affect even when prompt-based detection of the same construct remains fragile. XHS-SCoRE contri
    
[^269]: 重新审视LLM剪枝对测试时扩展的有效性

    Revisiting the Effectiveness of LLM Pruning for Test-Time Scaling

    [https://arxiv.org/abs/2604.25098](https://arxiv.org/abs/2604.25098)

    本研究发现非结构化剪枝不同于结构化剪枝，能保持推理LLMs的测试时扩展性能，挑战了现有假设。

    

    arXiv:2604.25098v3 公告类型：替换 摘要：大型语言模型（LLMs）现在通过测试时计算扩展（TTS）展现出卓越的推理能力，在数学和编程基准测试中表现惊人。与此同时，模型压缩领域的研究开发了剪枝方法，旨在移除冗余/有害参数而不牺牲任务性能。这两项研究进展的交汇点构成了我们工作的基础。具体针对推理型LLMs，先前的工作表明，结构化剪枝（移除整个层块的方法）会显著降低TTS推理性能。然而，在本工作中，我们重新审视了这一假设，并调查非结构化剪枝（仅精心移除某些冗余/有害权重的方法）是否表现出类似限制。令人惊讶的是，我们在两个推理LLMs（s1.1-7B和Qwen3-8B）上的四个推理基准测试中进行的广泛实验，一致表明非结构化剪枝并未受到同样的限制。

    arXiv:2604.25098v3 Announce Type: replace  Abstract: Large Language Models (LLMs) now exhibit remarkable reasoning capabilities through test-time compute scaling (TTS), with impressive performance across math and coding benchmarks. In parallel, research in model compression has developed pruning methods that seek to remove redundant/detrimental parameters without sacrificing task performance. The intersection of these two research advancements lays the foundation for our work. Specific to reasoning LLMs, prior work has shown that structured pruning (methods which remove entire set of layer blocks), significantly degrades TTS reasoning performance. However, in this work, we revisit this assumption and investigate whether unstructured pruning (methods that carefully remove only certain redundant/detrimental weights) exhibits similar limitations. Surprisingly, our extensive experiments across four reasoning benchmarks on two reasoning LLMs: s1.1-7B and Qwen3-8B, consistently show that uns
    
[^270]: DialToM：一个用于预测状态驱动对话轨迹的心理理论基准

    DialToM: A Theory of Mind Benchmark for Forecasting State-Driven Dialogue Trajectories

    [https://arxiv.org/abs/2604.20443](https://arxiv.org/abs/2604.20443)

    本文提出了DialToM基准，通过状态驱动诊断探针揭示了大语言模型在心理状态推断（字面ToM）与社交预测（功能ToM）之间的系统性不对称，并展示了人类专家100%准确率下的人机能力差距。

    

    arXiv:2604.20443v3 公告类型：替换交叉 摘要：我们引入了DialToM，这是一个基于自然人际对话构建的带注释的心理理论（ToM）基准，采用多项选择评估框架。与近期在合成环境中显示显式心理状态推断与应用型ToM之间存在差距的研究（~\\cite{gu2024simpletom}）相一致，我们建立了一个更严格的“状态驱动诊断探针”，要求模型仅从孤立的心理状态档案中预测与状态一致的对话轨迹，而无需对话上下文。我们的评估揭示了系统性的推理不对称性——大型语言模型在推断心理状态（字面ToM）方面表现出色，但在利用这些状态进行社会预测（功能ToM）方面却存在困难。关键的是，一位领域专家在此任务上达到了100%的准确率，证明了其有效性，并确立了显著的人机能力差距。此外，一种教师-学生推理注入探针显示，Gemini 3 Pro——作为领先基线——的表现进一步提升。

    arXiv:2604.20443v3 Announce Type: replace-cross  Abstract: We introduce DialToM, an annotated Theory of Mind (ToM) benchmark built from naturalistic human-human dialogues using a multiple-choice evaluation framework. Concurrent with recent work showing a gap between explicit mental-state inference and applied ToM in synthetic settings~\cite{gu2024simpletom}, we establish a stricter \emph{State-Driven Diagnostic Probe} in which models must forecast state-consistent dialogue trajectories solely from isolated mental-state profiles without dialogue context. Our evaluation reveals a systematic reasoning asymmetry -- LLMs excel at inferring mental states (Literal ToM) but struggle to leverage them for social forecasting (Functional ToM). Crucially, a domain expert achieves 100\% accuracy on this task, proving its validity and establishing a stark human-AI capability gap. Further, a teacher-student reasoning injection probe shows that Gemini 3 Pro -- which establishes the leading baseline -- 
    
[^271]: 利用梯度指纹检测与抑制奖励黑客行为

    Detecting and Suppressing Reward Hacking with Gradient Fingerprints

    [https://arxiv.org/abs/2604.16242](https://arxiv.org/abs/2604.16242)

    本文提出GRIFT方法，通过压缩模型内部梯度为指纹表示，有效检测并抑制强化学习中的隐性奖励黑客行为。

    

    arXiv:2604.16242v2 公告类型：替换交叉 摘要：具有可验证奖励的强化学习（RLVR）通常优化结果奖励，而不对中间推理过程施加约束。这使得训练容易受到奖励黑客行为的影响，即模型利用奖励函数中的漏洞（例如训练数据中的虚假模式）来获得高分，而无需解决预期任务。这些奖励黑客行为通常是隐性的，因为中间思维链（CoT）表面上可能看起来合理，限制了纯文本监控的有效性。我们提出了一种名为梯度指纹（GRIFT）的方法，利用模型的内部计算来检测奖励黑客行为。给定一个提示和模型生成的CoT，GRIFT计算以提示为条件的CoT梯度，并将其压缩为紧凑表示，然后用该表示来评估CoT是否反映了奖励黑客行为。在多个可验证推理基准测试中，该方法展现了有效性。

    arXiv:2604.16242v2 Announce Type: replace-cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) typically optimizes for outcome rewards without imposing constraints on intermediate reasoning. This leaves training susceptible to reward hacking, where models exploit loopholes (e.g., spurious patterns in training data) in the reward function to achieve high scores without solving the intended task. These reward-hacking behaviors are often implicit, as the intermediate chain-of-thought (CoT) may appear plausible on the surface, limiting the effectiveness of purely text-based monitoring. We propose Gradient Fingerprint (GRIFT), a method for detecting reward hacking using models' internal computations. Given a prompt and a model-generated CoT, GRIFT computes gradients of the CoT conditioned on the prompt and compresses them into a compact representation, which is then used to assess whether the CoT reflects reward hacking behavior. Across verifiable reasoning benchmarks span
    
[^272]: 释放隐式奖励：前缀值学习用于分布级优化

    Unleashing Implicit Rewards: Prefix-Value Learning for Distribution-Level Optimization

    [https://arxiv.org/abs/2604.13197](https://arxiv.org/abs/2604.13197)

    提出IPVRM模型，通过直接学习前缀的最终正确性概率并使用时间差分差异获得步骤信号，解决了隐式过程奖励模型训练与推理不匹配的问题，实现分布级优化。

    

    arXiv:2604.13197v4 公告类型：替换 摘要：过程奖励模型（PRMs）为推理提供细粒度监督，但可靠的PRMs通常需要逐步标注或繁重的验证流程，使其在在线强化学习期间难以扩展和刷新。隐式PRMs通过从轨迹级结果标签训练对数似然比奖励来降低这一成本。然而，对数比在训练期间仅作为序列级聚合被约束，而推理时将其分解为令牌级或步骤级分数用于部分前缀。这种训练-推理不匹配导致局部信用弱识别，因此分布级评分可能放大误导性优势。我们提出隐式前缀值奖励模型（IPVRM），直接从结果标签学习每个前缀最终正确性的概率。步骤信号随后通过连续前缀值之间的时间差分（TD）差异获得，使训练目标与推理时间对齐。

    arXiv:2604.13197v4 Announce Type: replace  Abstract: Process reward models (PRMs) provide fine-grained supervision for reasoning, but reliable PRMs often require step annotations or heavy verification pipelines, making them costly to scale and refresh during online RL. Implicit PRMs reduce this cost by training log-likelihood-ratio rewards from trajectory-level outcome labels. However, the log-ratio is constrained only as a sequence-level aggregate during training, while inference decomposes it into token- or step-level scores for partial prefixes. This train-inference mismatch leaves local credits weakly identified, so distribution-wide scoring can amplify misleading advantages. We propose Implicit Prefix-Value Reward Model (IPVRM), which directly learns the probability of eventual correctness for each prefix from outcome labels. Step signals are then obtained as temporal-difference (TD) differences between consecutive prefix values, aligning the training target with inference-time us
    
[^273]: 动物价值对齐的中期训练研究

    Alignment midtraining for animals

    [https://arxiv.org/abs/2604.13076](https://arxiv.org/abs/2604.13076)

    本文发现通过中期训练结合合成文档能有效提升动物同情心价值对齐，但后续无关指令调优会削弱其效果，提示需要显式保留策略来维持价值干预的持久性。

    

    arXiv:2604.13076v4 公告类型：替换交叉 摘要：我们研究了通过中期训练结合合成文档进行价值对齐的稳健性，以动物同情心作为一种既本身重要又与现有对齐工作正交的价值。为了评估富有同情心的推理能力，我们开发并公开发布了动物规范道德评估（ANIMA），这是一项包含26个问题、覆盖13个伦理维度的评估，作为数据集和Inspect评估公开可用。在ANIMA上，使用3000个文档训练达到了77%的准确率，而指令调优方法仅为40%，且泛化到人类同情心，同时在标准安全基准或能力上没有退化。然而，随后的无关指令调优削弱了干预效果，在5000个样本后优势消失。我们的探索性结果表明，基于文档的价值干预可能需要明确的保留策略，才能在典型训练流程中保持有效性。

    arXiv:2604.13076v4 Announce Type: replace-cross  Abstract: We investigate the robustness of value alignment via midtraining with synthetic documents, using animal compassion as a value that is both important in its own right and orthogonal to existing alignment efforts. To evaluate compassionate reasoning, we develop and publicly release Animal Norms In Moral Assessment (ANIMA), a 26-question evaluation spanning 13 ethical dimensions, publicly available as a dataset and Inspect evaluation. On ANIMA, training with 3000 documents achieves 77% compared to 40% for instruction-tuning approaches, with generalization to human compassion and no degradation in standard safety benchmarks or capabilities. However, subsequent unrelated instruction-tuning degrades the intervention, with the advantage disappearing after 5000 samples. Our exploratory results suggest document-based value interventions may require explicit preservation strategies to remain effective through typical training pipelines.
    
[^274]: CONSCIENTIA: 大型语言模型智能体能否学会策略？多智能体纽约市模拟中的涌现欺骗与信任

    CONSCIENTIA: Can LLM Agents Learn to Strategize? Emergent Deception and Trust in a Multi-Agent NYC Simulation

    [https://arxiv.org/abs/2604.09746](https://arxiv.org/abs/2604.09746)

    本研究通过纽约市多智能体模拟，实证观察了LLM智能体在对抗激励下涌现出的策略性欺骗与信任行为，并利用迭代优化流程探索其策略学习机制。

    

    随着大型语言模型（LLMs）越来越多地被部署为自主智能体，理解多智能体环境中策略行为如何涌现已成为一项重要的对齐挑战。我们采取中立的实证立场，构建了一个可直接观察和测量策略行为的受控环境。我们在一个简化的纽约市模型中引入大规模多智能体模拟，其中由LLM驱动的智能体在相反激励下互动。蓝色智能体旨在高效到达目的地，而红色智能体则试图通过说服性语言将蓝色智能体引向广告牌密集的路线，以最大化广告收入。隐藏身份使导航具有社会中介性，迫使智能体决定何时信任或欺骗。我们通过迭代模拟流程研究策略学习，该流程使用卡尼曼-特沃斯基优化算法在重复互动轮次中更新智能体策略。

    arXiv:2604.09746v2 Announce Type: replace-cross  Abstract: As large language models (LLMs) are increasingly deployed as autonomous agents, understanding how strategic behavior emerges in multi-agent environments has become an important alignment challenge. We take a neutral empirical stance and construct a controlled environment in which strategic behavior can be directly observed and measured. We introduce a large-scale multi-agent simulation in a simplified model of New York City, where LLM-driven agents interact under opposing incentives. Blue agents aim to reach their destinations efficiently, while Red agents attempt to divert them toward billboard-heavy routes using persuasive language to maximize advertising revenue. Hidden identities make navigation socially mediated, forcing agents to decide when to trust or deceive. We study policy learning through an iterative simulation pipeline that updates agent policies across repeated interaction rounds using Kahneman-Tversky Optimizati
    
[^275]: 大型语言模型通过一种跨危害类型共享的独特机制生成有害响应

    Large Language Models Generate Harmful Responses Using a Distinct Mechanism, Shared Across Harm Types

    [https://arxiv.org/abs/2604.09544](https://arxiv.org/abs/2604.09544)

    本文通过参数剪枝发现，大型语言模型的有害响应生成依赖于一组稀疏且跨危害类型共享的关键参数，且这种可分离性主要存在于对齐模型中，揭示了有害能力的独特机制。

    

    arXiv:2604.09544v3 公告类型：替换交叉 摘要：大型语言模型仍然容易受到引发有害响应的越狱攻击，但有害响应生成背后的机制尚不明确。在此，我们研究了这种能力如何在模型参数中组织。我们识别并剪除了专门支持有害顺从性的参数，提供了参数层面的直接机制分析。我们发现，这种能力依赖于一组稀疏的关键参数：剪除这些参数能大幅降低有害顺从性，同时仅对良性能力造成有限退化，这表明有害生成的关键组成部分与通用效用的组成部分是可分离的。从一个危害类别中识别出的参数也能减少其他类别的有害响应，表明这些组件在不同危害类型间共享。这种可分离性主要出现在对齐模型中，表明对齐训练在内部重塑了有害机制。

    arXiv:2604.09544v3 Announce Type: replace-cross  Abstract: Large language models remain vulnerable to jailbreaks that elicit harmful responses, yet the mechanism behind harmful response generation is poorly understood. Here, we investigate how this capability is organized within model parameters. We identify and prune parameters that specifically support harmful compliance, providing a direct mechanistic analysis at the parameter level. We find that this capability depends on a sparse set of critical parameters: pruning these parameters substantially reduces harmful compliance while causing only limited degradation in benign capabilities, suggesting that key components of harmful generation are separable from those of general utility. Parameters identified from one harm category also reduce harmful responses in others, indicating components shared across harm types. This separability appears primarily in aligned models, suggesting that alignment training internally reshapes the harmful
    
[^276]: PRAGMA：革命性基础模型

    PRAGMA: Revolut Foundation Model

    [https://arxiv.org/abs/2604.08649](https://arxiv.org/abs/2604.08649)

    PRAGMA是一种针对银行事件序列的金融基础模型，通过自监督掩码建模预训练，能从原始事件数据中直接提取通用嵌入，在信用评分、欺诈检测等任务中表现优异。

    

    现代金融系统生成海量的交易和事件级数据，这些数据编码了丰富的经济信号。本文介绍了PRAGMA，一个用于银行事件序列的基础模型家族。我们的方法采用基于Transformer的架构，在大规模异质性银行事件语料库上进行掩码建模预训练，使用针对金融记录离散、可变长度特性定制的自监督目标。所得模型支持广泛的下游任务，如信用评分、欺诈检测和终身价值预测：通过在提取的嵌入上训练简单的线性模型即可获得强大性能，并可通过轻量级微调进一步改进。通过对下游任务的广泛评估，我们证明PRAGMA直接从原始事件序列在多个领域实现了优越性能，提供了一个通用表示。

    arXiv:2604.08649v2 Announce Type: replace-cross  Abstract: Modern financial systems generate vast quantities of transactional and event-level data that encode rich economic signals. This paper presents PRAGMA, a family of foundation models for banking event sequences. Our approach pre-trains a Transformer-based architecture with masked modelling on a large-scale, heterogeneous banking event corpus using a self-supervised objective tailored to the discrete, variable-length nature of financial records. The resulting model supports a wide range of downstream tasks such as credit scoring, fraud detection, and lifetime value prediction: strong performance can be achieved by training a simple linear model on top of the extracted embeddings and can be further improved with lightweight fine-tuning. Through extensive evaluation on downstream tasks, we demonstrate that PRAGMA achieves superior performance across multiple domains directly from raw event sequences, providing a general-purpose repr
    
[^277]: 模型知道什么，知道得有多好：用于学习何时说“我不知道”的知识加权微调

    What Models Know, How Well They Know It: Knowledge-Weighted Fine-Tuning for Learning When to Say "I Don't Know"

    [https://arxiv.org/abs/2604.05779](https://arxiv.org/abs/2604.05779)

    本文提出了一种知识加权微调方法，通过估计实例级知识分数来调整学习信号，使模型能明确表达“我不知道”，同时保持对已知问题的准确性，并引入新的不确定性评估指标。

    

    arXiv:2604.05779v2 公告类型：替换 摘要：尽管大型语言模型（LLMs）在各种用户查询中展现出强大的能力，但它们仍然存在幻觉问题，这通常源于预训练和微调之间的知识错位。为了解决这种错位，我们通过多样本推理可靠地估计一个细粒度的、实例级别的知识分数。利用该知识分数，我们根据模型已有的知识来缩放学习信号，同时鼓励对超出范围的问题给出明确的“我不知道”回应。实验结果表明，这种方法使模型在缺乏知识时能够明确表达不确定性，同时在能回答的问题上保持准确性。此外，我们提出了不确定性评估指标，表明对已知和未知实例的准确区分能持续提升性能。

    arXiv:2604.05779v2 Announce Type: replace  Abstract: While large language models (LLMs) demonstrate strong capabilities across diverse user queries, they still suffer from hallucinations, often arising from knowledge misalignment between pre-training and fine-tuning. To address this misalignment, we reliably estimate a fine-grained, instance-level knowledge score via multi-sampled inference. Using the knowledge score, we scale the learning signal according to the model's existing knowledge, while encouraging explicit "I don't know" responses for out-of-scope queries. Experimental results show that this approach allows the model to explicitly express uncertainty when it lacks knowledge, while maintaining accuracy on questions it can answer. Furthermore, we propose evaluation metrics for uncertainty, showing that accurate discrimination between known and unknown instances consistently improves performance.
    
[^278]: 将大语言模型的假设进行言语化以解释和控制谄媚行为

    Verbalizing LLMs' assumptions to explain and control sycophancy

    [https://arxiv.org/abs/2604.03058](https://arxiv.org/abs/2604.03058)

    本文提出“言语化假设”框架，通过引出并利用LLM对用户的假设来理解和控制其社交谄媚行为，并发现“寻求验证”是主要假设，且可通过线性探针进行因果干预。

    

    arXiv:2604.03058v3 公告类型：交叉替换 摘要：大语言模型（LLMs）可能表现出社交谄媚行为，当用户提出“我错了吗？”这类问题时，它们会迎合用户而非提供真实评估。我们假设这种行为源于LLMs对用户的不正确假设，例如低估用户寻求信息而非安慰的频率。我们提出了“言语化假设”（Verbalized Assumptions）框架，用于从LLMs中引出这些假设。言语化假设为LLM的谄媚、妄想及其他安全问题提供了洞察：在社交谄媚数据集中，“寻求验证”是LLM假设中最常见的二元词组。我们提供了假设与谄媚模型行为之间存在因果联系的证据：我们训练了与言语化假设相关的内部表示的线性探针，并利用这些探针对社交谄媚进行可解释的细粒度引导。最后，我们识别了人机期望差距，这解释了为何LLM会失败。

    arXiv:2604.03058v3 Announce Type: replace-cross  Abstract: LLMs can be socially sycophantic, affirming users when they ask questions like "am I in the wrong?" rather than providing genuine assessment. We hypothesize that this behavior arises from LLMs' incorrect assumptions about the user, like underestimating how often users are seeking information over reassurance. We present Verbalized Assumptions, a framework for eliciting these assumptions from LLMs. Verbalized Assumptions provide insight into LLM sycophancy, delusion, and other safety issues: in social sycophancy datasets, "seeking validation" is the most frequent bigram in LLMs' assumptions. We provide evidence for a causal link between assumptions and sycophantic model behavior: we train linear probes on internal representations associated with Verbalized Assumptions and then use these probes for interpretable, fine-grained steering of social sycophancy. Finally, we identify a human-AI expectation gap that explains why LLMs def
    
[^279]: SAFE：一种以LLM为验证器的证据基础多跳推理框架

    SAFE: An LLM-as-Verifier Framework for Evidence-Grounded Multi-Hop Reasoning

    [https://arxiv.org/abs/2604.01993](https://arxiv.org/abs/2604.01993)

    SAFE通过将推理分解为知识图谱三元组并在生成过程中用外部验证器实时检查中间步骤，有效防止了多跳问答中的虚假正确性，显著提升了准确率。

    

    arXiv:2604.01993v3 公告类型：替换交叉 摘要：多跳问答基准测试往往因大型语言模型（LLMs）通过无效的中间推理达到正确答案而奖励其虚假正确性。我们提出SAFE，一种用于证据基础多跳问答的LLM作为验证器的框架。SAFE不仅在生成后判断最终答案，而是在生成过程中验证推理，通过检查中间步骤与提供的段落及先前推理轨迹的一致性。为使此过程可检查，SAFE将推理分解为原子化的、基于证据的单元，并以知识图谱（KG）三元组表示。在训练时，SAFE在KG约束下验证基准监督，并构建可靠的验证器训练数据。在推理时，外部验证器检查每个生成步骤，识别无效推理，并在错误传播前提供纠正反馈。在三个多跳问答基准测试中，SAFE将准确率提高了8.8%。

    arXiv:2604.01993v3 Announce Type: replace-cross  Abstract: Multi-hop QA benchmarks often reward Large Language Models (LLMs) for spurious correctness, where models reach correct answers through invalid intermediate reasoning. We propose SAFE, an LLM-as-verifier framework for evidence-grounded multi-hop QA. Rather than judging only the final answer after generation, SAFE verifies reasoning during generation by checking intermediate steps against the provided passages and previous reasoning trajectory. To make this process checkable, SAFE decomposes reasoning into atomic, evidence-grounded units represented with Knowledge Graph (KG) triples. At train-time, SAFE verifies benchmark supervision under KG-grounded constraints and constructs reliable verifier training data. At inference-time, an external verifier checks each generated step, identifies invalid reasoning, and provides correction feedback before errors propagate. Across three multi-hop QA benchmarks, SAFE improves accuracy by 8.8
    
[^280]: Bock算法用于最小有向生成树的温和教程及其结构化重构

    A gentle tutorial on Bock's algorithm for minimum directed spanning trees with a structured reformulation

    [https://arxiv.org/abs/2603.27530](https://arxiv.org/abs/2603.27530)

    本教程详细解读Bock算法，并通过结构化重构保留其核心选择与结果，使原始复杂逻辑更清晰易懂。

    

    arXiv:2603.27530v2 公告类型：替换 摘要：Bock于1971年提出的算法是解决最小成本树形图问题的一种精确原始-对偶方法，但其Algol语言呈现方式掩盖了其维护数组与标签导向控制流之间的交互。我们提供一个自包含的教程，包括原始代码清单、逐行映射解释、一个形成电路的三节点示例，以及Bock十节点实例的完整追踪。我们还提出了一种结构化重构，用显式的组件和追踪状态替代临时的跨度标签更改。局部紧致性和收缩进展结果，连同操作对应定理，确立了该重构保留了Bock的候选选择、转移和最终解决方案。

    arXiv:2603.27530v2 Announce Type: replace  Abstract: Bock's 1971 algorithm is an exact primal--dual method for the minimum-cost arborescence problem, but its Algol presentation obscures the interaction of its maintained arrays and label-directed control flow. We provide a self-contained tutorial comprising the original listing, a line-mapped explanation, a circuit-forming three-node example, and a complete trace of Bock's ten-node instance. We also present a structured reformulation that replaces temporary span-label changes with explicit component and trace state. Local tightness and contraction-progress results, together with an operational-correspondence theorem, establish that the reformulation preserves Bock's candidate choices, transfers, and final solution.
    
[^281]: UT-ACA：不确定性触发的自适应上下文分配用于长上下文推理

    UT-ACA: Uncertainty-Triggered Adaptive Context Allocation for Long-Context Inference

    [https://arxiv.org/abs/2603.18446](https://arxiv.org/abs/2603.18446)

    本文提出UT-ACA，一种通过令牌级不确定性动态调整上下文窗口的推理时框架，能显著减少长上下文推理中的平均上下文使用量。

    

    长上下文推理对大型语言模型仍具挑战性，原因是注意力稀释和分布外退化。上下文选择通过关注键值缓存条目的子集来缓解这一限制，但大多数方法在解码过程中分配固定的上下文预算，尽管令牌级别的上下文需求高度不均匀。为解决此问题，我们提出了不确定性触发的自适应上下文分配（UT-ACA），一种基于令牌级不确定性动态调整上下文窗口的推理时框架。UT-ACA学习了一个不确定性检测器，该检测器结合语义嵌入和基于逻辑的置信度，同时考虑跨解码步骤的不确定性累积。当指示证据不足时，UT-ACA选择性地回滚、扩展上下文窗口，并在额外支持下重新生成令牌。实验表明，UT-ACA显著减少了平均上下文使用量。

    arXiv:2603.18446v3 Announce Type: replace  Abstract: Long-context inference remains challenging for large language models due to attention dilution and out-of-distribution degradation. Context selection mitigates this limitation by attending to a subset of key-value cache entries, yet most methods allocate a fixed context budget throughout decoding despite highly non-uniform token-level contextual demands. To address this issue, we propose Uncertainty-Triggered Adaptive Context Allocation (UT-ACA), an inference-time framework that dynamically adjusts the context window based on token-wise uncertainty. UT-ACA learns an uncertainty detector that combines semantic embeddings with logit-based confidence while accounting for uncertainty accumulation across decoding steps. When insufficient evidence is indicated, UT-ACA selectively rolls back, expands the context window, and regenerates the token with additional support. Experiments show that UT-ACA substantially reduces average context usag
    
[^282]: DynHD：通过去噪动态偏差学习实现扩散大语言模型的幻觉检测

    DynHD: Hallucination Detection for Diffusion Large Language Models via Denoising Dynamics Deviation Learning

    [https://arxiv.org/abs/2603.16459](https://arxiv.org/abs/2603.16459)

    本文提出DynHD方法，通过同时建模标记空间的不确定性差异和扩散过程的时间动态偏差，有效提升了扩散大语言模型幻觉检测的准确性。

    

    arXiv:2603.16459v2 公告类型：替换 摘要：扩散大语言模型（D-LLMs）凭借其迭代细化能力，已成为自回归模型的一种有前景的替代方案。然而，幻觉问题仍然是阻碍其可靠性的关键挑战。为了检测模型输出中的幻觉响应，标记级不确定性（如熵）被广泛用于指示潜在的事实错误。但与顺序生成标记的自回归模型不同，D-LLMs同时生成固定长度的序列，其中只有少量标记对幻觉检测具有信息量。因此，对所有标记的不确定性进行聚合可能并非最优。此外，扩散过程中不确定性的演变趋势也能提供有价值的信号，这凸显了对其去噪动态进行建模以检测幻觉的必要性。在本文中，我们提出了DynHD，它从空间和时间两个维度弥合了这些差距。

    arXiv:2603.16459v2 Announce Type: replace  Abstract: Diffusion large language models (D-LLMs) have emerged as a promising alternative to auto-regressive models due to their iterative refinement capabilities. However, hallucinations remain a critical issue that hinders their reliability. To detect hallucination responses from model outputs, token-level uncertainty, such as entropy, has been widely used to indicate potential factual errors. Nevertheless, unlike auto-regressive models that generate tokens sequentially, D-LLMs generate fixed-length sequences simultaneously, where only a small subset of tokens is informative for hallucination detection. Thus, aggregating uncertainty over all tokens can be suboptimal. Moreover, the evolution trend of uncertainty throughout the diffusion process can also provide valuable signals, highlighting the necessity of modeling its denoising dynamics for hallucination detection. In this paper, we propose DynHD, which bridges these gaps from both spatia
    
[^283]: SkillNet：创建、评估与连接AI技能

    SkillNet: Create, Evaluate, and Connect AI Skills

    [https://arxiv.org/abs/2603.04448](https://arxiv.org/abs/2603.04448)

    本文提出了SkillNet，一个开放基础设施，通过统一本体论、大规模技能仓库和多功能工具包，系统性地创建、评估和连接AI技能，解决了代理缺乏技能积累和迁移的问题。

    

    摘要：arXiv:2603.04448v2 公告类型：替换 摘要：当前的AI代理能够灵活调用工具并执行复杂任务，但其长期发展受到缺乏技能系统性积累和迁移的阻碍。在没有统一的技能整合机制的情况下，代理经常“重复造轮子”，在孤立的环境中重新发现解决方案，而不利用先前的策略。为解决这一挑战，我们引入了SkillNet，一个用于大规模创建、评估和组织AI技能的开放基础设施。SkillNet在统一本体论中结构化技能，支持从异构来源创建技能、建立丰富的关联连接，并在安全性、完整性、可执行性、可维护性和成本意识方面进行多维度评估。我们的基础设施整合了一个包含超过60万个技能的仓库、一个交互式平台和一个多功能的Python工具包。在ALFWorld、WebShop和ScienceWorld上的实验表明……

    arXiv:2603.04448v2 Announce Type: replace  Abstract: Current AI agents can flexibly invoke tools and execute complex tasks, yet their long-term advancement is hindered by the lack of systematic accumulation and transfer of skills. Without a unified mechanism for skill consolidation, agents frequently ``reinvent the wheel'', rediscovering solutions in isolated contexts without leveraging prior strategies. To address this challenge, we introduce SkillNet, an open infrastructure for creating, evaluating, and organizing AI skills at scale. SkillNet structures skills within a unified ontology that supports creating skills from heterogeneous sources, establishing rich relational connections, and performing multi-dimensional evaluation across Safety, Completeness, Executability, Maintainability, and Cost-awareness. Our infrastructure integrates a repository of over 600,000 skills, an interactive platform, and a versatile Python toolkit. Experiments on ALFWorld, WebShop, and ScienceWorld show 
    
[^284]: 安全训练可能在LLM代理的有用性优化后持续存在

    Safety Training May Persist Through Helpfulness Optimization in LLM Agents

    [https://arxiv.org/abs/2603.02229](https://arxiv.org/abs/2603.02229)

    该论文发现，在LLM代理中，安全训练在有用性优化后仍能持续，但安全性与有用性存在强烈的负相关，且同时优化无法突破这一趋势。

    

    arXiv:2603.02229v2 公告类型：替换交叉 摘要：安全后训练已在单步“聊天”设置中广泛研究，其中安全性通常指拒绝有害请求。我们研究了一个“代理式”（即多步、工具使用）设置，其中安全性指LLM直接采取的有害行动。我们调查了使用直接偏好优化（DPO）在ToolEmu代理基准上优化安全性和/或有用的影响。首先，我们发现安全训练在很大程度上通过后续的有用性训练得以持续。其次，我们发现当考虑所有训练配置时，安全性和有用性之间存在一致的负线性相关性（$R^2 = 0.77$）。即使在两个指标上同时进行后训练，也只是在相同趋势线上产生另一个点，而不是产生“两全其美”的策略，尽管我们的数据集中存在此类策略。总体而言，我们的发现强调了需要更好地理解这些权衡。

    arXiv:2603.02229v2 Announce Type: replace-cross  Abstract: Safety post-training has been studied extensively in single-step "chat" settings where safety typically refers to refusing harmful requests. We study an "agentic" (i.e., multi-step, tool-use) setting where safety refers to harmful actions directly taken by the LLM. We investigate the effects of using direct preference optimization (DPO) to optimize safety and/or helpfulness on the ToolEmu agentic benchmark. First, we find that safety training largely persists through subsequent helpfulness training. Second, we find a consistent negative linear correlation ($R^2 = 0.77$) between safety and helpfulness when considering all training configurations together. Even post-training on both metrics simultaneously simply results in another point on the same trend line rather than yielding a "best of both worlds" strategy, despite the presence of such strategies in our dataset. Overall, our findings underscore the need for a better underst
    
[^285]: 语义基底动力学理论：几何语义漂移的算子理论框架

    Semantic Substrate Dynamics Theory: An Operator-Theoretic Framework for Geometric Semantic Drift

    [https://arxiv.org/abs/2602.18699](https://arxiv.org/abs/2602.18699)

    本文提出语义基底动力学理论，通过将嵌入几何与扩散核耦合，用粗里奇曲率区分不同语义漂移机制，为异质漂移信号提供统一算子理论框架。

    

    arXiv:2602.18699v2 公告类型：替换-交叉 摘要：关于语义漂移的研究报告了异质信号，包括嵌入位移、邻居变化、分布散度以及递归轨迹不稳定性，但没有一个统一的理论来解释它们之间的关系。语义基底动力学理论（SSDT）将这些信号视为一个时间索引基底 St = (X, dt, Pt) 的可观测对象，该基底将嵌入几何与局部扩散核耦合。其贡献在于与机制层的可通约性：该基底区分了盆地内搅动与跨盆地穿越、递归诱导的不稳定性以及干预顺序效应，这些区分是单一检测分数无法恢复的。粗里奇曲率作为图结构中盆地和桥梁几何的密集结构描述符，而桥梁质量（入射负曲率的节点级聚合）作为稀疏描述符，用于表示嵌入中通常不常见的真实桥梁结构。

    arXiv:2602.18699v2 Announce Type: replace-cross  Abstract: Studies of semantic drift report heterogeneous signals, including embedding displacement, neighbor change, distributional divergence, and recursive trajectory instability, without a shared account that relates them. Semantic Substrate Dynamics Theory (SSDT) treats these signals as observables of one time-indexed substrate, St = (X, dt, Pt), that couples embedding geometry to a local diffusion kernel. The contribution is commensurability with a mechanism layer: the substrate separates within-basin churn from basin crossing, recursion-induced instability, and intervention-order effects, distinctions that a single detection score does not recover. Coarse Ricci curvature functions as a dense structural descriptor of basin and bridge geometry across the graph, and bridge mass, a node-level aggregate of incident negative curvature, functions as a sparse descriptor of the genuine bridge structure that is typically uncommon in embeddin
    
[^286]: 审判中的“氛围编程”：一致通过的大语言模型陪审团的操作特性

    Vibe Coding on Trial: Operating Characteristics of Unanimous LLM Juries

    [https://arxiv.org/abs/2602.18492](https://arxiv.org/abs/2602.18492)

    本研究提出并评估了一种基于大语言模型陪审团的一致通过机制，用于自动审查文本到SQL任务中的候选代码，在安全优先场景下有效平衡了接受准确性与人工干预成本。

    

    大语言模型（LLMs）在编程方面已经足够优秀，开发者可以用自然语言描述意图，让工具生成初版代码，这种工作流程日益集成到GitHub Copilot、Cursor和Replit等工具中。目前缺少的是一种可靠的方法，用来判断哪些模型生成的查询可以安全接受，而不必事事都交给人工审核。我们研究应用LLM陪审团来执行这一审查步骤。我们首先在82个MySQL文本到SQL任务上对15个开源模型进行了基准测试，采用基于执行验证的协议，以清晰确定哪些模型表现强劲。然后，从六个最佳模型中构建了规模为1到6的一致通过委员会，这些委员会查看提示、数据库模式和候选SQL，并且仅当所有成员都认为正确时才接受该查询。这一规则符合安全优先的部署场景，其中错误接受比错误拒绝代价更高。我们测量了真正率、假正率和Youden J指数，并进一步分析了……

    arXiv:2602.18492v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are now good enough at coding that developers can describe intent in plain language and let the tool produce the first code draft, a workflow increasingly built into tools like GitHub Copilot, Cursor, and Replit. What is missing is a reliable way to tell which model written queries are safe to accept without sending everything to a human. We study the application of an LLM jury to run this review step. We first benchmark 15 open models on 82 MySQL text to SQL tasks using an execution grounded protocol to get a clean baseline of which models are strong. From the six best models we build unanimous committees of sizes 1 through 6 that see the prompt, schema, and candidate SQL and accept it only when every member says it is correct. This rule matches safety first deployments where false accepts are more costly than false rejects. We measure true positive rate, false positive rate and Youden J and we als
    
[^287]: 翻译腔的方言塑造语言模型学习

    Dialects of Translationese Shape Language Model Learning

    [https://arxiv.org/abs/2602.16469](https://arxiv.org/abs/2602.16469)

    本文发现，机器翻译数据的源语言差异通过翻译腔显著影响语言模型的学习行为，其中词汇多样性是驱动总体困惑度的关键因素。

    

    arXiv:2602.16469v2 公告类型：替换 摘要：机器翻译数据在多语言自然语言处理中广泛使用，尤其是在母语文本稀缺的情况下。然而，翻译文本与母语文本存在系统性差异。这一现象被称为“翻译腔”，它既反映了源语言的痕迹，也体现了翻译本身的特征性属性。在本文中，我们研究了在机器翻译数据上训练如何影响小型英语语言模型，重点关注来自不同源语言的翻译腔如何塑造语言可接受性判断和不同领域的语言建模。我们在从24种类型学和资源多样性各异的源语言翻译成的英语文本上训练模型，从而能够系统分析源语言和语料库属性如何影响模型学习的内容。我们的结果表明，源语言对模型行为有明显影响：总体困惑度更多由翻译语料的词汇多样性驱动，但

    arXiv:2602.16469v2 Announce Type: replace  Abstract: Machine-translated data is widely used in multilingual NLP, particularly where native text is scarce. However, translated text differs systematically from native text. This phenomenon is known as translationese, and it reflects both traces of the source language and characteristic properties of translation itself. In this paper, we study how training on machine-translated data affects small English language models, focusing on how translationese from different source languages shapes linguistic acceptability judgments and language modeling for different domains. We train models on English text translated from 24 typologically and resource-diverse source languages, enabling a systematic analysis of how source language and corpus properties influence what models learn. Our results show that the source language has a clear impact on model behavior: general perplexity is more driven by the lexical diversity of the translated corpus, but 
    
[^288]: 块扩散语言模型在复杂推理中的自适应测试时计算分配

    Adaptive Test-Time Compute Allocation for Block Diffusion Language Models in Complex Reasoning

    [https://arxiv.org/abs/2602.09555](https://arxiv.org/abs/2602.09555)

    本文提出一个统一的测试时计算分配框架，通过BACD解码策略和TCCF块级生成范式，在块扩散语言模型中实现自适应推理，兼顾速度与推理精度。

    

    arXiv:2602.09555v3 公告类型：替换 摘要：近年来，块扩散语言模型在推理任务中展现出竞争性性能和强可扩展性。然而，其测试时计算分配在很大程度上仍未得到探索，导致在长链思维推理中存在一个关键的速度-效果权衡问题未解决。为解决此问题，我们提出一个统一的测试时计算分配框架，在逐步解码和块级生成中引入自适应性。在解码层面，我们提出有界自适应置信解码（BACD），一种基于难度的采样策略，根据模型置信度动态调整去噪过程，加速推理同时控制误差累积。在逐步自适应性之外，我们引入“粗思考，精批评”（TCCF）范式，为大块分配用于探索性思考，为小块分配用于精确细化。为在不同块配置下稳定训练，我们...

    arXiv:2602.09555v3 Announce Type: replace  Abstract: Recent advances in block diffusion language models have demonstrated competitive performance and strong scalability on reasoning tasks. However, their test-time compute allocation remains largely unexplored, leaving a critical speed-effectiveness trade-off unresolved in long Chain-of-Thought reasoning. To address this, we propose a unified test-time compute allocation framework that introduces adaptivity in both step-wise decoding and blockwise generation. At the decoding level, we propose Bounded Adaptive Confidence Decoding (BACD), a difficulty-aware sampling strategy that dynamically adjusts denoising based on model confidence, accelerating inference while controlling error accumulation. Beyond step-wise adaptivity, we introduce the Think Coarse, Critic Fine (TCCF) paradigm that allocates large block sizes for exploratory thinking and smaller block sizes for precise refinement. To stabilize training under varying block configurati
    
[^289]: LakeHopper：跨数据湖的列类型标注器的知识感知适配

    LakeHopper: Knowledge-Aware Adaptation of Column Type Annotators across Data Lakes

    [https://arxiv.org/abs/2602.08793](https://arxiv.org/abs/2602.08793)

    LakeHopper通过将跨数据湖的列类型标注适配分解为知识丢弃、重对齐和获取，并利用通用大语言模型与目标监督相结合，实现了高效的知识感知适配。

    

    列类型标注（CTA）为表格列分配语义类型，支撑数据湖上的数据集成、清洗和搜索。最先进的标注器是在特定表格语料库（即源数据湖）上微调的预训练语言模型（PLM），一旦部署到新（即目标）数据湖上，其性能会急剧下降，因为该湖的表格和语义类型集均不同。按湖重新训练成本高昂，因为它需要大量专家标注。我们将跨湖适配重新定义为一个知识管理问题，并明确其分解：相对于目标标注器，源标注器持有的知识必须被丢弃（源特定）、重新对齐和重用（共享）或获取（目标特定）。这种分解揭示了通用大语言模型（LLM）能弥补哪部分差距，以及哪部分只能由目标监督来完成。在其指导下，我们提出了LakeHopper，它……

    arXiv:2602.08793v2 Announce Type: replace  Abstract: Column Type Annotation (CTA), which assigns a semantic type to a table column, underpins data integration, cleaning, and search over data lakes. State-of-the-art annotators are pre-trained language models (PLMs) fine-tuned on one particular corpus of tables, i.e., a source data lake, and they degrade sharply once deployed on a new (i.e., target) lake, whose tables and semantic type set both differ. Retraining per lake is prohibitive because it demands large volumes of expert annotations. We recast cross-lake adaptation as a knowledge management problem and make the resulting decomposition explicit: relative to a target annotator, a source annotator holds knowledge that must be discarded (source-specific), realigned and reused (shared), or acquired (target-specific). This decomposition exposes which part of the gap a general-purpose LLM can close and which part only target supervision can. Guided by it, we present LakeHopper, which ad
    
[^290]: MIRROR：一种面向运筹学优化建模的迭代自适应修正与分层检索多智能体框架

    MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research

    [https://arxiv.org/abs/2602.03318](https://arxiv.org/abs/2602.03318)

    MIRROR通过无需微调的多智能体框架，结合执行驱动的迭代修正和分层检索，实现了自然语言优化问题到数学模型及求解器代码的直接自动转化。

    

    arXiv:2602.03318v4 公告类型：替换  摘要：运筹学（OR）依赖于专家驱动的建模——这是一个缓慢且脆弱的过程，难以适应新场景。虽然大型语言模型（LLMs）能自动将自然语言转化为优化模型，但现有方法要么依赖昂贵的后训练，要么采用多智能体框架，然而大多数仍缺乏可靠的协作错误修正和任务特定检索，常常导致输出不正确。我们提出MIRROR（一种用于运筹学优化建模的多智能体框架，具有迭代自适应修正和分层检索功能），这是一个无需微调、端到端的多智能体框架，能直接将自然语言优化问题转化为数学模型和求解器代码。MIRROR集成了两个核心机制：（1）执行驱动的迭代自适应修正，用于自动错误纠正；（2）分层检索，以获取相关的建模和编码示例。

    arXiv:2602.03318v4 Announce Type: replace  Abstract: Operations Research (OR) relies on expert-driven modeling--a slow and fragile process ill-suited to novel scenarios. While large language models (LLMs) can automatically translate natural language into optimization models, existing approaches either rely on costly post-training or employ multi-agent frameworks, yet most still lack reliable collaborative error correction and task-specific retrieval, often leading to incorrect outputs. We propose MIRROR (a Multi-agent framework with Iterative adaptive Revision and hierarchical Retrieval for optimization modeling in Operations Research), a fine-tuning-free, end-to-end multi-agent framework that directly translates natural language optimization problems into mathematical models and solver code. MIRROR integrates two core mechanisms: (1) execution-driven iterative adaptive revision for automatic error correction, and (2) hierarchical retrieval to fetch relevant modeling and coding exempla
    
[^291]: CALIBURN：自校准的大语言模型遗忘对齐

    CALIBURN: Self-Calibrated LLM Unlearning Alignment

    [https://arxiv.org/abs/2602.02824](https://arxiv.org/abs/2602.02824)

    我们提出了一种自校准遗忘方法，通过量化模型对不良知识的置信度来精确调整梯度更新，在实现细粒度遗忘的同时减少对保留数据的依赖，从而提升模型效用。

    

    大语言模型遗忘旨在从预训练语言模型中移除不良知识的影响，这为解决安全和隐私问题提供了一种实用机制。现有的遗忘方法，如梯度上升，容易导致灾难性遗忘。基于对齐的方法提供了另一种方向，但其有效性受限于参考模型的质量。在现实场景中，这两种方法仍需要大量保留数据集来维持通用知识。我们提出了一种原则性方法，该方法量化目标大语言模型对不良知识的置信度，并利用该置信度更精确地校准模型的遗忘梯度更新。它能够实现对遗忘的细粒度控制，同时更好地保持模型效用，从而减少对保留数据或过高的遗忘训练数据的依赖。在包括MUSE和WMDP在内的多个基准上的广泛评估表明，该方法表现出色。

    arXiv:2602.02824v2 Announce Type: replace  Abstract: LLM unlearning aims to remove the influence of undesirable knowledge from pretrained language models, which offers a practical mechanism for addressing safety and privacy concerns. Existing unlearning approaches, such as Gradient Ascent, are prone to catastrophic forgetting. Alignment-based approaches provide an alternative direction, yet their effectiveness is limited by the quality of the reference model. In realistic settings, both methods still require large retention datasets to preserve general knowledge. We propose a principled method that quantifies the target LLM's confidence in undesirable knowledge and uses it to calibrate the model's unlearning gradient updates more precisely. It enables fine-grained control over forgetting while better preserving model utility, thus reducing the dependence on retention data or prohibitive unlearning training data. Extensive evaluations on multiple benchmarks, including MUSE and WMDP, sho
    
[^292]: 前瞻后验：上下文无关文法下扩散大语言模型的可靠约束解码

    Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under Context-Free Grammars

    [https://arxiv.org/abs/2602.00612](https://arxiv.org/abs/2602.00612)

    提出LAVE方法，通过前瞻-验证机制解决扩散大语言模型在上下文无关文法下的约束解码可靠性问题。

    

    arXiv:2602.00612v3 公告类型：替换 摘要：扩散大语言模型（dLLMs）已展现出有前景的生成能力，并越来越多地被用于生成由上下文无关文法定义的形式语言，如源代码和化学表达式。然而，作为概率模型，它们仍难以可靠地生成语法有效的输出。解决此问题的一个自然且有前景的方向是调整约束解码技术，以在生成过程中强制语法正确性。然而，应用这些技术面临两个主要障碍。一方面，dLLMs的非自回归特性使得大多数现有约束解码方法不适用。另一方面，当前专门为dLLMs设计的方法可能允许中间输出无法完成成有效句子，这显著限制了其在实际中的可靠性。为解决这些挑战，我们提出了LAVE，一种约束方法。

    arXiv:2602.00612v3 Announce Type: replace  Abstract: Diffusion Large Language Models (dLLMs) have demonstrated promising generative capabilities and are increasingly used to produce formal languages defined by context-free grammars, such as source code and chemical expressions. However, as probabilistic models, they still struggle to generate syntactically valid outputs reliably. A natural and promising direction to address this issue is to adapt constrained decoding techniques to enforce grammatical correctness during generation. However, applying these techniques faces two primary obstacles. On the one hand, the non-autoregressive nature of dLLMs renders most existing constrained decoding approaches inapplicable. On the other hand, current approaches specifically designed for dLLMs may allow intermediate outputs that are impossible to complete into valid sentences, which significantly limits their reliability in practice.   To address these challenges, we present LAVE, a constrained 
    
[^293]: FourierSampler：通过频率引导生成释放扩散语言模型的非自回归潜力

    FourierSampler: Unlocking Non-Autoregressive Potential in Diffusion Language Models via Frequency-Guided Generation

    [https://arxiv.org/abs/2601.23182](https://arxiv.org/abs/2601.23182)

    通过首次频域分析揭示扩散语言模型隐藏状态中低频编码全局结构、高频编码局部细节的规律，并提出FourierSampler利用频域滑动窗口机制实现“结构到细节”的生成，显著提升了非自回归解码性能。

    

    尽管扩散语言模型（dLLMs）具有非自回归的潜力，现有的解码策略表现出位置偏差，未能完全释放任意生成的潜力。在这项工作中，我们深入研究了dLLMs的固有频谱特性，并首次进行了频域分析，表明隐藏状态中的低频分量主要编码全局结构信息和长距离依赖，而高频分量则负责表征局部细节。基于这一观察，我们提出了FourierSampler，它利用频域滑动窗口机制动态引导模型实现“结构到细节”的生成。FourierSampler在LLADA和SDAR上优于其他推理增强策略，在LLaDA1.5-8B上实现了20.4%的相对改进，在LLaDA-8B-Instruct上实现了16.0%的相对改进。它显著超越了类似规模的自回归模型。

    arXiv:2601.23182v2 Announce Type: replace  Abstract: Despite the non-autoregressive potential of diffusion language models (dLLMs), existing decoding strategies demonstrate positional bias, failing to fully unlock the potential of arbitrary generation. In this work, we delve into the inherent spectral characteristics of dLLMs and present the first frequency-domain analysis showing that low-frequency components in hidden states primarily encode global structural information and long-range dependencies, while high-frequency components are responsible for characterizing local details. Based on this observation, we propose FourierSampler, which leverages a frequency-domain sliding window mechanism to dynamically guide the model to achieve a "structure-to-detail" generation. FourierSampler outperforms other inference enhancement strategies on LLADA and SDAR, achieving relative improvements of 20.4% on LLaDA1.5-8B and 16.0% on LLaDA-8B-Instruct. It notably surpasses similarly sized autoregre
    
[^294]: 语言模型知道但不说的事情：用于泛化的非生成式先验提取

    What Language Models Know But Don't Say: Non-Generative Prior Extraction for Generalization

    [https://arxiv.org/abs/2601.17609](https://arxiv.org/abs/2601.17609)

    该论文提出LoID方法，通过直接探测语言模型的词元级预测置信度，而非生成文本，提取用于贝叶斯逻辑回归的先验分布，从而在小型数据集上提升模型对现实世界的泛化能力。

    

    在医学和金融等领域，大规模标注数据成本高昂且常常不可用，导致在小型数据集上训练的模型难以泛化到现实世界的人群。大型语言模型包含了这些领域多年研究的广泛知识。我们提出了LoID（Logit信息分布），一种确定性方法，通过直接访问其词元级预测，为贝叶斯逻辑回归提取信息丰富的先验分布。我们不依赖生成的文本，而是通过精心构建的句子，探测模型在相反语义方向（正面与负面影响）上的置信度。通过衡量LLM在不同措辞中对某一方向的一致性偏好，我们提取了模型关于每个特征影响强度的信念及其可靠性。我们在十个真实世界表格数据集上，在合成分布外（OOD）设置下评估了LoID。

    arXiv:2601.17609v3 Announce Type: replace  Abstract: In domains like medicine and finance, large-scale labeled data is costly and often unavailable, leading to models trained on small datasets that struggle to generalize to real-world populations. Large language models contain extensive knowledge from years of research across these domains. We propose LoID (Logit-Informed Distributions), a deterministic method for extracting informative prior distributions for Bayesian logistic regression by directly accessing their token-level predictions. Rather than relying on generated text, we probe the model's confidence in opposing semantic directions (positive vs. negative impact) through carefully constructed sentences. By measuring how consistently the LLM favors one direction across diverse phrasings, we extract the strength and reliability of the model's belief about each feature's influence. We evaluate LoID on ten real-world tabular datasets under synthetic out-of-distribution (OOD) setti
    
[^295]: 基于大语言模型的对抗性说服攻击对事实核查系统的影响

    LLM-Based Adversarial Persuasion Attacks on Fact-Checking Systems

    [https://arxiv.org/abs/2601.16890](https://arxiv.org/abs/2601.16890)

    本文首次提出利用大语言模型结合说服技巧对自动事实核查系统进行对抗性攻击，显著降低其验证和证据检索性能。

    

    自动事实核查（AFC）系统容易受到对抗性攻击，使得虚假声明能够逃避检测。现有的对抗性框架通常依赖于注入噪声或改变语义，但尚无框架利用说服技巧对抗AFC系统的对抗潜力，而这些技巧在虚假信息活动中被广泛用于操纵受众。在本文中，我们引入了一类新颖的针对AFC系统的说服性对抗攻击，通过使用大语言模型（LLM）运用说服技巧改写声明。考虑到分属5个类别的15种技巧，我们采用解耦评估策略研究了说服对声明验证和证据检索的影响。在FEVER和FEVEROUS基准上的实验表明，说服攻击能显著降低验证性能和证据检索效果。我们的分析将说服技巧确定为一种强效的对抗手段。

    arXiv:2601.16890v2 Announce Type: replace-cross  Abstract: Automated fact-checking (AFC) systems are susceptible to adversarial attacks, enabling false claims to evade detection. Existing adversarial frameworks typically rely on injecting noise or altering semantics, yet no existing framework exploits the adversarial potential of persuasion techniques against AFC systems, which are widely used in disinformation campaigns to manipulate audiences. In this paper, we introduce a novel class of persuasive adversarial attacks on AFCs by employing an LLM to rephrase claims using persuasion techniques. Considering $15$ techniques grouped into $5$ categories, we study the effects of persuasion on both claim verification and evidence retrieval using a decoupled evaluation strategy. Experiments on the FEVER and FEVEROUS benchmarks show that persuasion attacks can substantially degrade both verification performance and evidence retrieval. Our analysis identifies persuasion techniques as a potent c
    
[^296]: 你需要更好的注意力先验

    You Need Better Attention Priors

    [https://arxiv.org/abs/2601.15380](https://arxiv.org/abs/2601.15380)

    该论文通过熵最优传输统一了注意力机制，提出GOAT，用可学习先验替代均匀先验，兼容FlashAttention，解决注意力汇问题，并实现长度泛化。

    

    arXiv:2601.15380v2 公告类型：交叉替换 摘要：我们通过熵最优传输的视角来泛化注意力机制，揭示了标准注意力对应于一个由隐式均匀先验正则化的传输问题。我们引入了具有可训练先验的广义最优传输注意力（GOAT），这是一种新的注意力机制，用可学习的连续先验替代了这种朴素假设。该先验与诸如FlashAttention等优化内核保持完全兼容。GOAT还提供了基于EOT的注意力汇解释，并为其实现了具体解决方案，避免了标准注意力的表征权衡。最后，通过将空间信息吸收到核心注意力计算中，GOAT学习了一个可外推的先验，该先验结合了学习位置嵌入的灵活性与固定编码的长度泛化能力。

    arXiv:2601.15380v2 Announce Type: replace-cross  Abstract: We generalize the attention mechanism by viewing it through the lens of Entropic Optimal Transport, revealing that standard attention corresponds to a transport problem regularized by an implicit uniform prior. We introduce Generalized Optimal transport Attention with Trainable priors (GOAT), a new attention mechanism that replaces this naive assumption with a learnable, continuous prior. This prior maintains full compatibility with optimized kernels such as FlashAttention. GOAT also provides an EOT-based explanation of attention sinks and materializes a solution for them, avoiding the representational trade-offs of standard attention. Finally, by absorbing spatial information into the core attention computation, GOAT learns an extrapolatable prior that combines the flexibility of learned positional embeddings with the length generalization of fixed encodings.
    
[^297]: 自然语言处理在低资源塞内加尔语言社会科学研究中的机遇与挑战

    Opportunities and Challenges of Natural Language Processing for Low-Resource Senegalese Languages in Social Science Research

    [https://arxiv.org/abs/2601.09716](https://arxiv.org/abs/2601.09716)

    本文首次系统梳理了塞内加尔六种官方语言的NLP资源与挑战，并创建了集中式资源库以推动低资源语言在社会科学研究中的应用。

    

    自然语言处理（NLP）正在迅速改变跨学科的研究方法，然而非洲语言在这一技术变革中仍然在很大程度上未被充分代表。本文首次全面概述了塞内加尔宪法正式承认的六种国家语言——沃洛夫语、普拉尔语、塞雷尔语、迪奥拉语、曼丁戈语和索宁克语——在NLP方面的进展与挑战。我们综合了影响其数字准备度的语言、社会技术和基础设施因素，并识别了数据、工具和基准方面的空白。基于现有倡议和研究工作，我们分析了涵盖文本和语音模态的各种任务的持续努力。我们还提供了一个集中式GitHub仓库，汇编了这些语言在各种NLP任务中可公开获取的资源，旨在促进协作和可复现性。特别关注点在于社会科学研究中的应用。

    arXiv:2601.09716v2 Announce Type: replace  Abstract: Natural Language Processing (NLP) is rapidly transforming research methodologies across disciplines, yet African languages remain largely underrepresented in this technological shift. This paper provides the first comprehensive overview of NLP progress and challenges for the six national languages officially recognized by the Senegalese Constitution: Wolof, Pulaar, S\'er\`ere, Diola, Mandingue, and Sonink\'e. We synthesize linguistic, socio-technical, and infrastructural factors that shape their digital readiness and identify gaps in data, tools, and benchmarks. Building on existing initiatives and research works, we analyze ongoing efforts in various tasks, covering both text and speech modalities. We also provide a centralized GitHub repository that compiles publicly accessible resources for a range of NLP tasks across these languages, designed to facilitate collaboration and reproducibility. A special focus is devoted to the appli
    
[^298]: 偏好调优在领域偏移下的泛化性与多样性实证研究

    An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift

    [https://arxiv.org/abs/2601.05882](https://arxiv.org/abs/2601.05882)

    本研究系统比较了五种偏好调优目标及多种适应策略在领域偏移下的泛化性，发现伪标签适应策略能有效缓解性能下降并保持多样性。

    

    arXiv:2601.05882v2 公告类型：交叉替换 摘要：偏好调优通过优化明确的偏好信号而非仅依赖似然度，将基础语言模型与人类对质量、有用性或安全性的判断对齐。先前研究表明，偏好调优在训练领域之外会降低性能并减少有用性。然而，适应策略在多大程度上缓解这种领域偏移仍未得到探索。我们通过进行一项全面且系统的对齐泛化研究来解决这一挑战。我们比较了五种流行的对齐目标以及从源域到目标域的各种适应策略，包括目标域监督微调和伪标签，涵盖摘要生成、问答有用性和安全性对齐任务。我们的研究结果揭示了在领域偏移下不同对齐目标间泛化性的系统性差异。我们表明基于伪标签的适应策略在缓解领域偏移方面表现优越。

    arXiv:2601.05882v2 Announce Type: replace-cross  Abstract: Preference tuning aligns base language models to human judgments of quality, helpfulness, or safety by optimizing over explicit preference signals rather than likelihood alone. Prior work has shown that preference tuning degrades performance and reduces helpfulness outside the training domain. However, the extent to which adaptation strategies mitigate this domain shift remains unexplored. We address this challenge by conducting a comprehensive and systematic study of alignment generalization under domain shift. We compare five popular alignment objectives and various adaptation strategies from source to target, including target-domain supervised fine-tuning and pseudo-labeling, across summarization, question-answering helpfulness, and safety alignment tasks. Our findings reveal systematic differences in generalization across alignment objectives under domain shift. We show that adaptation strategies based on pseudo-labeling su
    
[^299]: 训练主动且个性化的LLM代理

    Training Proactive and Personalized LLM Agents

    [https://arxiv.org/abs/2511.02208](https://arxiv.org/abs/2511.02208)

    该论文提出了一种训练LLM代理的新范式，通过多目标强化学习优化生产力、主动性和个性化三个维度，使代理能更有效地与人类协作并适应个性化需求。

    

    尽管取得了快速进展，当前的人工智能代理主要针对孤立的任务完成进行优化。我们主张向训练代理作为协作伙伴的范式转变，使其能够与人类沟通并适应人类。为了在现实世界复杂应用中促进这一转变，我们首先形式化了协作式AI代理的三个维度：生产力、主动性和个性化（PPP）。我们引入了UserVille，一个交互式环境，配备可配置的基于LLM的用户模拟器和以用户为中心的反馈，以评估这些维度，并提出了一个多目标强化学习框架，利用任务结果、提问努力和偏好遵循的奖励来优化这些维度。在两个现实世界的代理任务（SWE-Bench和BrowseComp-Plus）上，经过PPP训练的代理平均比强LLM基线（包括GPT-5）高出16.7个百分点，提出更有针对性的问题，并能泛化到未见过的偏好和任务。后续研究...

    arXiv:2511.02208v2 Announce Type: replace  Abstract: Despite rapid progress, current AI agents are primarily optimized for isolated task completion. We argue for a paradigm shift toward training agents as collaborators that communicate and adapt to people. To facilitate this shift in real-world complex applications, we first formalize three dimensions of collaborative AI agents: Productivity, Proactivity, and Personalization (PPP). We introduce UserVille, an interactive environment with configurable LLM-based user simulators and user-centric feedback to evaluate these dimensions, and propose a multi-objective reinforcement learning framework that optimizes them using rewards from task outcomes, question effort, and preference adherence. On two real-world agentic tasks (SWE-Bench and BrowseComp-Plus), PPP-trained agents outperform strong LLM baselines (including GPT-5) by an average of 16.7 points, ask more targeted questions, and generalize to unseen preferences and tasks. A follow-up 
    
[^300]: LuxIT：从单语种子数据构建的卢森堡语指令微调数据集

    LuxIT: A Luxembourgish Instruction Tuning Dataset from Monolingual Seed Data

    [https://arxiv.org/abs/2510.24434](https://arxiv.org/abs/2510.24434)

    该论文介绍了LuxIT，一个针对卢森堡语的高质量单语指令微调数据集，通过LLM-as-a-judge质量筛选和微调14个模型，平均提升语言考试准确率5.37个百分点，展示了其在低资源语言NLP中的有效性。

    

    arXiv:2510.24434v4 公告类型：替换。摘要：指令微调的大型语言模型（LLMs）在低资源语言环境中的有效性通常受限于高质量训练数据的缺乏。我们引入了LuxIT，一个为卢森堡语开发的单语指令微调数据集，旨在缓解这一挑战。我们从卢森堡语母语文本语料库中合成该数据集，利用了因其在卢森堡语中表现出色而选用的DeepSeek-R1-0528模型。生成后，我们应用了质量保证流程，采用LLM作为评判的方法，保留了227,507个高质量的指令-答案对。为了探究该数据集的实际效用，我们在LuxIT上微调了14个较小规模的LLMs（参数≤15B），并在标准化卢森堡语水平考试和五个下游NLP任务上进行了评估。在LuxIT上训练后，所有14个模型的语言考试平均准确率变化为+5.37个百分点，其中14个模型中的12个显示出改进。

    arXiv:2510.24434v4 Announce Type: replace  Abstract: The effectiveness of instruction-tuned Large Language Models (LLMs) is often limited in low-resource linguistic settings due to a lack of high-quality training data. We introduce LuxIT, a monolingual instruction tuning dataset for Luxembourgish developed to mitigate this challenge. We synthesize the dataset from a corpus of native Luxembourgish texts, utilizing DeepSeek-R1-0528, chosen for its shown proficiency in Luxembourgish. Following generation, we apply a quality assurance process, employing an LLM-as-a-judge approach, retaining 227,507 high-quality instruction-answer pairs. To investigate the practical utility of the dataset, we fine-tune 14 smaller-scale LLMs ($\leq$15B parameters) on LuxIT and evaluate them on standardized Luxembourgish proficiency exams and five downstream NLP tasks. Training on LuxIT yields a mean accuracy change of +5.37 percentage points on language exams across all 14 models, with 12 of 14 showing impro
    
[^301]: 遗忘之遗忘：注意力汇聚点作为大语言模型遗忘后门的通道

    Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning

    [https://arxiv.org/abs/2510.17021](https://arxiv.org/abs/2510.17021)

    本文首次探索了大语言模型遗忘过程中的后门攻击，发现注意力汇聚点现象与后门有效性密切相关，使得模型在触发条件下能恢复已遗忘的知识。

    

    大语言模型（LLM）遗忘是移除预训练模型中不需要的数据、知识或行为，同时保持其整体实用性的关键方法。然而，随着开放权重LLM的兴起，我们提出疑问：遗忘过程本身是否可能被植入后门，即在正常条件下看似成功，但当隐藏触发器被激活时却恢复到遗忘前的行为？受经典后门攻击（将触发器嵌入训练数据以强制执行特定行为）的启发，我们研究了遗忘后门攻击这一设定，即模型在干净环境中按预期遗忘，但当触发器出现时恢复被遗忘的知识。我们表明，设计此类攻击面临独特挑战，取决于触发器的放置位置以及后门训练的强化方式。我们发现了后门效果与注意力汇聚点现象（即浅层输入标记的一致性）之间的强关联。

    arXiv:2510.17021v2 Announce Type: replace-cross  Abstract: Large language model (LLM) unlearning is a key approach for removing undesired data, knowledge, or behaviors from pretrained models while retaining their general utility. Yet, with the rise of open-weight LLMs, we ask: can the unlearning process itself be backdoored, appearing successful under normal conditions yet reverting to pre-unlearned behavior when a hidden trigger is activated? Drawing inspiration from classical backdoor attacks that embed triggers into training data to enforce specific behaviors, we investigate backdooring unlearning, a setting in which models forget as intended in the clean setting but recover forgotten knowledge when the trigger appears. We show that designing such attacks presents unique challenges, hinging on where triggers are placed and how backdoor training is reinforced. We uncover a strong link between the backdoor efficacy and the attention sink phenomenon (i.e., shallow input tokens consiste
    
[^302]: 面向检索增强生成的LLM特定效用

    LLM-Specific Utility for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2510.11358](https://arxiv.org/abs/2510.11358)

    本文首次形式化并实证了检索增强生成中证据的LLM特定效用，证明其具有模型依赖性和不可转移性，为优化RAG系统提供了新视角。

    

    arXiv:2510.11358v3 公告类型：替换-交叉 摘要：检索增强生成（RAG）通常针对主题相关性进行优化，但其成功最终取决于检索到的段落是否有助于大型语言模型（LLM）生成正确且完整的答案。我们认为，这种效用往往是LLM特定的，而非普遍通用的，这归因于模型在知识、推理和利用证据能力方面的差异。我们将LLM特定效用形式化为，当提供某个段落时，目标LLM的性能相比无证据作答时的提升幅度。为系统研究LLM特定效用，我们构建了一个基准，针对四个LLM（Qwen3-8B/14B/32B和Llama 3.1-8B）在三个问答数据集（Natural Questions、TriviaQA和MS MARCO-FQA）上提供了LLM特定的黄金效用段落。我们的分析表明，效用段落具有模型依赖性和不可转移性：每个LLM在其自身的效用证据下表现最佳，而为其他模型优化的证据则表现不佳。

    arXiv:2510.11358v3 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) is typically optimized for topical relevance, yet its success ultimately depends on whether retrieved passages are useful for a large language model (LLM) to generate correct and complete answers. We argue that such utility is often LLM-specific rather than universal, due to differences in models' knowledge, reasoning, and ability to leverage evidence. We formalize LLM-specific utility as the performance improvement of a target LLM when a passage is provided, compared to answering without evidence. To systematically study LLM-specific utility, we construct a benchmark of LLM-specific gold utilitarian passages for four LLMs (Qwen3-8B/14B/32B and Llama 3.1-8B) on three QA datasets (Natural Questions, TriviaQA, and MS MARCO-FQA). Our analysis shows that utilitarian passages are model-dependent and non-transferable: each LLM performs best with its own utilitarian evidence, while evidence optimiz
    
[^303]: GraphMed-LT：基于患者特定图记忆与潜在临床思维细化的多轮医疗对话系统

    GraphMed-LT: Patient-Specific Graph Memory with Latent Clinical Thought Refinement for Multi-Turn Medical Conversations

    [https://arxiv.org/abs/2510.03536](https://arxiv.org/abs/2510.03536)

    GraphMed-LT通过构建患者特定的图记忆并利用潜在临床思维细化，解决了多轮医疗对话中临床证据碎片化的问题，从而提升了诊断连贯性。

    

    多轮医疗问答旨在模拟真实的临床诊断过程，其中医生通过多轮对话收集患者信息。现有的多轮医疗对话系统已取得一定进展，但它们通常依赖累积的对话历史作为记忆，导致临床证据在多轮中碎片化。我们提出了GraphMed-LT，一种带有潜在临床思维细化的患者特定图记忆方法，用于多轮医疗对话。GraphMed-LT从患者回应中提取患者特定的临床三元组，检索相关的知识三元组，并将它们组织为增量更新的图记忆。图记忆被投影为图条件证据令牌，并通过隐藏状态反馈在可训练的医生代理内部进行细化，使代理在提出后续问题或生成回答之前更新其内部临床上下文。

    arXiv:2510.03536v3 Announce Type: replace-cross  Abstract: Multi-turn medical question answering (QA) aims to model realistic clinical diagnosis, where a doctor gathers patient information across multiple turns of conversation. Existing multi-turn medical conversation systems have shown promising progress, but they often rely on accumulated conversation histories as memory, leaving clinical evidence fragmented across turns. We propose GraphMed-LT, a patient-specific graph memory approach with latent clinical thought refinement for multi-turn medical conversations. GraphMed-LT extracts patient-specific clinical triplets from patient responses, retrieves relevant knowledge triplets, and organises them into an incrementally updated graph memory. The graph memory is projected into graph-conditioned evidence tokens and refined inside a trainable doctor agent through hidden-state feedback, enabling the agent to update its internal clinical context before asking follow-up questions or produci
    
[^304]: 语法引导的扩散语言模型与用户集成个性化

    Syntax-Guided Diffusion Language Models with User-Integrated Personalization

    [https://arxiv.org/abs/2510.01028](https://arxiv.org/abs/2510.01028)

    本文提出了一种语法引导的扩散语言模型，通过级联与非级联架构集成结构监督和个性化条件，显著提升了文本生成的质量、多样性和可控性。

    

    大型语言模型在生成类人文本方面取得了革命性进展，但其输出往往趋于通用化，缺乏足够的结构多样性，这限制了个性化表达。扩散模型的最新进展为超越自回归范式的局限、改进语言生成开辟了新机遇。在本工作中，我们提出了一种语法引导的扩散语言模型，该模型集成了结构监督和个性化条件，以提升文本质量、多样性和可控性。我们引入了一个级联框架，在条件文本生成之前生成语法引导，并进一步将其推广到一种新颖的非级联架构，以更好地实现结构与内容之间的对齐。通过将语法信息融入生成过程，所提出的模型能够更好地捕捉文体化句子的词汇和结构特征。

    arXiv:2510.01028v2 Announce Type: replace  Abstract: Large language models have made revolutionary progress in generating human-like text, yet their outputs often tend to be generic, exhibiting insufficient structural diversity, which limits personalized expression. Recent advances in diffusion models have opened new opportunities for improving language generation beyond the limitations of autoregressive paradigms. In this work, we propose a syntax-guided diffusion language model that integrates structural supervision and personalized conditioning to enhance text quality, diversity, and controllability. We introduce a cascaded framework that generates syntactic guidance before conditional text generation, and further generalize it to a novel noncascaded architecture for better alignment between structure and content. By incorporating syntactic information in the generating process, the proposed model better captures the lexical and structural characteristics of stylistic sentence const
    
[^305]: ConvergeWriter：数据驱动的自下而上文章构建

    ConvergeWriter: Data-Driven Bottom-Up Article Construction

    [https://arxiv.org/abs/2509.12811](https://arxiv.org/abs/2509.12811)

    本文提出了一种自下而上的数据驱动框架，通过先检索知识再聚类构建结构，解决了长文档生成中计划与知识脱节的问题，提升了事实准确性和内容连贯性。

    

    大型语言模型（LLMs）在文本生成方面表现出色，但生成基于广泛外部知识库的长篇、事实性文档仍然是一个重大挑战。现有的“自上而下”方法，即首先生成假设或大纲，然后检索证据，常常因模型计划与可用知识之间的脱节而遭受内容碎片化和事实不准确的问题。为解决这些局限性，我们提出了一种新颖的“自下而上”、数据驱动的框架，该框架颠覆了传统的生成流程。我们的方法基于“先检索获取知识，聚类构建结构”的策略，在任何生成规划之前，首先确立源语料库的“知识边界”。具体来说，我们从知识库中进行详尽的迭代检索，然后采用无监督聚类算法来组织检索到的内容。

    arXiv:2509.12811v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have shown remarkable prowess in text generation, yet producing long-form, factual documents grounded in extensive external knowledge bases remains a significant challenge. Existing "top-down" methods, which first generate a hypothesis or outline and then retrieve evidence, often suffer from a disconnect between the model's plan and the available knowledge, leading to content fragmentation and factual inaccuracies. To address these limitations, we propose a novel "bottom-up," data-driven framework that inverts the conventional generation pipeline. Our approach is predicated on a "Retrieval-First for Knowledge, Clustering for Structure" strategy, which first establishes the "knowledge boundaries" of the source corpus before any generative planning occurs. Specifically, we perform exhaustive iterative retrieval from the knowledge base and then employ an unsupervised clustering algorithm to organize the retr
    
[^306]: 超越基准：基于拟人化与生命周期导向路线图的大语言模型评估

    Beyond Benchmarks: LLM Evaluation with an Anthropomorphic and Lifecycle-oriented Roadmap

    [https://arxiv.org/abs/2508.18646](https://arxiv.org/abs/2508.18646)

    本文提出了一种拟人化的四维评估框架（IQ、PQ、EQ、VQ），将大语言模型评估从静态基准排名转变为基于训练流程因果映射的诊断工具，以弥合基准分数与现实实用性之间的鸿沟。

    

    尽管大语言模型（LLM）取得了快速进展，但其基准分数与现实世界实用性之间存在严重脱节。当前评估仍呈碎片化状态，优先考虑孤立的技术指标，而忽视了部署所必需的整体性、发展性和社会性方面。本文并非仅仅作为描述性目录，而是建立了一种诊断性本体论，将评估维度因果地映射到典型的大语言模型训练流程，从而将评估从静态排名转变为用于根本原因分析的诊断工具。本文提出了一种拟人化评估框架，通过四个维度重新概念化大语言模型的能力：智商（IQ）、专业商（PQ）、情商（EQ）和价值观商（VQ）。我们通过模块化评估架构实现这些概念，并验证了该框架的诊断能力。

    arXiv:2508.18646v3 Announce Type: replace-cross  Abstract: Despite their rapid advancement, large language models (LLMs) suffer from a critical disconnect between benchmark scores and real-world utility. Current evaluation remains fragmented, prioritizing isolated technical metrics over the holistic, developmental, and societal aspects essential for deployment. Rather than serving merely as a descriptive catalog, this work establishes a diagnostic ontology that causally maps evaluation dimensions to the canonical LLM training pipeline, transforming evaluation from static ranking into a diagnostic tool for root-cause analysis. In this paper, we introduce an anthropomorphic evaluation framework that re-conceptualizes LLM capabilities through a four-dimensional lens: Intelligence Quotient (IQ), Professional Quotient (PQ), Emotional Quotient (EQ), and Value-oriented Quotient (VQ). We operationalize these concepts through a modular evaluation architecture and validate the framework's diagno
    
[^307]: 同一家族中的模型并非信任等效

    Models in the Same Family are NOT Trust-Equivalent

    [https://arxiv.org/abs/2508.13533](https://arxiv.org/abs/2508.13533)

    本文提出一个框架评估同一模型家族中大小模型间的信任等效性，发现仅性能相似并不保证归因对齐和校准相似性，因此小模型不能视为大模型的信任等效替代品。

    

    arXiv:2508.13533v2 公告类型：替换 摘要：在模型家族中，当较小变体与较大变体的性能相似时，它常被部署为直接替代品。然而，仅凭性能并不能说明全部问题。我们提出了一个框架，用于评估同一家族中较大模型与较小模型之间的信任等效性，该评估沿两个维度进行。第一个维度是归因对齐：两个模型是否基于相同的输入特征进行预测？第二个维度是校准相似性：两个模型是否在置信度与准确性之间共享相同的关系？我们在两个文本分类任务上评估了Llama-2家族：自然语言推理和释义识别。归因对齐使用两种著名方法LIME和SHAP进行测量。模型对之间的一致性通过前K个归因特征的Jaccard系数进行量化。我们观察到模型之间的归因对齐通常较低，这表明较小的模型并不能在信任方面等效地替代较大的模型。

    arXiv:2508.13533v2 Announce Type: replace  Abstract: Within a model family, a smaller variant is often deployed as a drop-in replacement for a larger one when their performance is similar. However, performance alone does not tell the full story. We propose a framework to evaluate trust-equivalence between a larger model and a smaller one in the same family along two dimensions. The first is attribution alignment: do both models base their predictions on the same input features? The second is calibration similarity: do both models share the same relationship between confidence and accuracy? We evaluate the Llama-2 family on two text classification tasks: Natural Language Inference and Paraphrase Identification. Attribution alignment is measured using two well-known methods: LIME and SHAP. Agreement between model pairs is quantified via the Jaccard coefficient over top-K attributed features. We observe that attribution alignment between models is generally low, indicating that smaller an
    
[^308]: Omni-SafetyBench：面向音视频大语言模型安全评估的基准

    Omni-SafetyBench: A Benchmark for Safety Evaluation of Audio-Visual Large Language Models

    [https://arxiv.org/abs/2508.07173](https://arxiv.org/abs/2508.07173)

    本文提出了Omni-SafetyBench，这是首个包含23,328个测试实例、覆盖24种模态变体的平行基准，用于全面评估全模态大语言模型在音视频联合输入下的安全风险，并引入基于Condi的Safety-score指标来应对跨模态一致性挑战。

    

    arXiv:2508.07173v3 公告类型：替换 摘要：融合视觉、听觉和文本处理的全模态大语言模型（OLLMs）面临严重的安全风险。它们对音视频联合有害输入的防御能力脆弱，且在不同模态间表现出不一致的安全性能，使得简单的模态切换越狱攻击成为可能。然而，现有安全基准因缺乏音视频联合样本、模态覆盖有限以及缺少用于跨模态一致性评估的平行测试用例，无法全面评估这些风险。为弥补这些不足，我们引入了Omni-SafetyBench，这是首个面向OLLM安全评估的全面平行基准，包含源自972个种子样本的23,328个测试实例，覆盖24种模态变体。鉴于复杂输入带来的理解挑战以及跨模态一致性对OLLM安全的关键作用，我们提出了定制化指标：基于Condi的Safety-score。

    arXiv:2508.07173v3 Announce Type: replace  Abstract: Omni-modal Large Language Models (OLLMs) that integrate visual, auditory, and textual processing face severe safety risks. They exhibit fragile defenses against audio-visual joint harmful inputs and demonstrate inconsistent safety performance across different modalities, enabling simple modality-switching jailbreaks. However, existing safety benchmarks fail to comprehensively assess these risks due to the absence of audio-visual joint samples, limited modality coverage, and lack of parallel test cases for cross-modal consistency evaluation. To address these gaps, we introduce Omni-SafetyBench, the first comprehensive parallel benchmark for OLLM safety evaluation, featuring 23,328 test instances across 24 modality variations derived from 972 seed samples. Recognizing that complex inputs pose comprehension challenges and that cross-modal consistency is critical for OLLM safety, we propose tailored metrics: a Safety-score based on Condi
    
[^309]: 从隔离到对齐：统一LoRA用于高效多任务学习

    From Isolation to Alignment: Unified LoRA for Efficient Multi-Task Learning

    [https://arxiv.org/abs/2508.05078](https://arxiv.org/abs/2508.05078)

    本文通过揭示复杂多组件LoRA变体的冗余性，提出统一的单适配器Align-LoRA框架，以对齐代替隔离，实现高效且性能优越的多任务微调。

    

    参数高效微调（PEFT）对于将大型语言模型（LLMs）适应多任务场景至关重要。该领域的一个主流趋势涉及复杂的LoRA变体，这些变体包含多个适配器或头部，其前提是任务特定知识的架构隔离是必要的。然而，这种设计常常引入动态路由，阻碍权重合并并导致显著的推理延迟。在这项工作中，我们直接挑战了这一范式。我们首先揭示了一个悖论：一个简化、无路由器且具有高头部间冗余的多头模型，其性能优于复杂且强调多样性的基线。此外，我们证明了一个统一的、单适配器的LoRA，通过增加秩，能达到极具竞争力的性能，从而质疑了多组件结构的必要性。基于这些发现，我们提出了Align-LoRA，一个统一且高效的框架，将焦点从隔离转向对齐。

    arXiv:2508.05078v2 Announce Type: replace-cross  Abstract: Parameter-Efficient Fine-Tuning (PEFT) is essential for adapting Large Language Models (LLMs) to multi-task scenarios. A prevailing trend in this field involves complex LoRA variants with multiple adapters or heads, which rely on the premise that architectural isolation of task-specific knowledge is necessary. However, this design often introduces dynamic routing, preventing weight merging and causing significant inference latency. In this work, we present a direct challenge to this paradigm. We first reveal a paradox where a simplified, router-free multi-head model with high inter-head redundancy outperforms complex, diversity-driven baselines. Furthermore, we demonstrate that a unified, single-adapter LoRA with increased rank achieves highly competitive performance, questioning the necessity of multi-component structures. Based on these findings, we propose Align-LoRA, a unified and efficient framework that shifts the focus f
    
[^310]: 方向建模，而非词语：基于稀疏自编码器的机制主题模型

    Model Directions, Not Words: Mechanistic Topic Models Using Sparse Autoencoders

    [https://arxiv.org/abs/2507.23220](https://arxiv.org/abs/2507.23220)

    本文提出机制主题模型（MTMs），利用稀疏自编码器的可解释特征定义主题，从而超越词袋限制，实现更深层主题发现和可控文本生成，并引入基于LLM的评估框架。

    

    arXiv:2507.23220v3 公告类型：替换 摘要：传统主题模型在揭示大规模文本集合中的潜在主题方面效果显著。然而，由于它们依赖词袋表示，难以捕捉语义抽象特征。尽管一些神经变体使用更丰富的表示，但它们同样受限于将主题表达为词列表，这限制了它们阐述复杂主题的能力。我们引入了机制主题模型（MTMs），这是一类在稀疏自编码器（SAEs）学习的可解释特征上操作的主题模型。通过在这个语义丰富的空间上定义主题，MTMs能够揭示更深层次的概念主题，并具有表现力的特征描述。此外，在主题模型中独有地，MTMs使用主题引导向量实现可控文本生成。为了恰当评估MTM主题与词列表方法的优劣，我们提出了\textit{topic judge}，一个基于LLM的成对比较评估框架。在八个数据集上进行了实验。

    arXiv:2507.23220v3 Announce Type: replace  Abstract: Traditional topic models are effective at uncovering latent themes in large text collections. However, due to their reliance on bag-of-words representations, they struggle to capture semantically abstract features. While some neural variants use richer representations, they are similarly constrained by expressing topics as word lists, which limits their ability to articulate complex topics. We introduce Mechanistic Topic Models (MTMs), a class of topic models that operate on interpretable features learned by sparse autoencoders (SAEs). By defining topics over this semantically rich space, MTMs can reveal deeper conceptual themes with expressive feature descriptions. Moreover, uniquely among topic models, MTMs enable controllable text generation using topic steering vectors. To properly evaluate MTM topics against word list approaches, we propose \textit{topic judge}, an LLM-based pairwise comparison evaluation framework. Across eight
    
[^311]: TELEVAL：面向中文交互场景的口语语言模型基准测试

    TELEVAL: A Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios

    [https://arxiv.org/abs/2507.18061](https://arxiv.org/abs/2507.18061)

    TELEVAL是一个针对中文无指令音频交互场景的大规模基准，首次同时评估SLM的语义准确性和基于声学线索的交互适当性，并发现模型在声学变异下性能显著退化。

    

    arXiv:2507.18061v4 公告类型：替换交叉 摘要：口语语言模型（SLMs）预期能够支持超越任务完成的自然口语交互。然而，现有的SLM基准测试主要评估结构化环境中的语义正确性，并且对基于声学上下文的交互行为评估有限。为解决这一差距，我们引入了TELEVAL，一个大规模的中文口语交互SLM基准测试，适用于无指令、音频条件下的场景。TELEVAL评估两个互补方面：（1）可靠内容满足度，衡量SLMs在不同声学和语言条件下的语义准确性；（2）交互适当性，评估模型是否通过隐式地将行为基于听觉线索来产生自然且适当的回应。实验表明，尽管模型在语义任务上表现良好，但在声学变异和交互设置下其性能会下降。我们观察到...

    arXiv:2507.18061v4 Announce Type: replace-cross  Abstract: Spoken Language Models (SLMs) are expected to support natural spoken interaction beyond task completion. However, existing SLM benchmarks primarily evaluate semantic correctness in structured settings and provide limited assessment of interactional behavior grounded in acoustic context. To address this gap, we introduce TELEVAL, a large-scale SLM benchmark for Chinese spoken interaction in instruction-free, audio-conditioned settings. TELEVAL evaluates two complementary aspects: (1) Reliable Content Fulfillment, which measures semantic accuracy of SLMs under diverse acoustic and linguistic conditions, and (2) Interactional Appropriateness, which assesses whether models produce natural and appropriate responses by implicitly grounding behavior in auditory cues. Experiments show that while models perform competitively on semantic tasks, their performance degrades under acoustic variability and in interactional settings. We observ
    
[^312]: Text-ADBench：基于大语言模型嵌入的文本异常检测基准

    Text-ADBench: Text Anomaly Detection Benchmark Based on LLM Embeddings

    [https://arxiv.org/abs/2507.12295](https://arxiv.org/abs/2507.12295)

    本文提出了一个基于多种大语言模型嵌入的文本异常检测基准，系统评估了不同嵌入方法在广泛文本数据集上的性能，填补了该领域缺乏标准化基准的空白。

    

    文本异常检测是自然语言处理（NLP）中的一项关键任务，其应用涵盖欺诈检测、错误信息识别、垃圾邮件检测和内容审核等领域。尽管大语言模型（LLMs）和异常检测算法取得了显著进展，但缺乏标准化和全面的基准来评估现有文本数据上的异常检测方法，这限制了严格比较和创新方法的发展。本研究进行了全面的实证研究，并引入了一个文本异常检测基准，利用来自多种预训练语言模型的嵌入，覆盖广泛的文本数据集。我们的工作通过整合（1）早期语言模型（GloVe、BERT）；（2）多种大语言模型（LLaMA-2、LLaMA-3、Mistral、OpenAI嵌入模型（小、ada、大）），系统评估了基于嵌入的文本异常检测的有效性。

    arXiv:2507.12295v2 Announce Type: replace-cross  Abstract: Text anomaly detection is a critical task in natural language processing (NLP), with applications spanning fraud detection, misinformation identification, spam detection and content moderation, etc. Despite significant advances in large language models (LLMs) and anomaly detection algorithms, the absence of standardized and comprehensive benchmarks for evaluating the existing anomaly detection methods on text data limits rigorous comparison and development of innovative approaches. This work performs a comprehensive empirical study and introduces a benchmark for text anomaly detection, leveraging embeddings from diverse pre-trained language models across a wide array of text datasets. Our work systematically evaluates the effectiveness of embedding-based text anomaly detection by incorporating (1) early language models (GloVe, BERT); (2) multiple LLMs (LLaMA-2, LLaMA-3, Mistral, OpenAI embedding models (small, ada, large)); (3)
    
[^313]: MixLoRA-DSI：动态可扩展的混合LoRA专家用于动态语料库上的免重训生成式检索

    MixLoRA-DSI: Dynamically Expandable Mixture-of-LoRA Experts for Rehearsal-Free Generative Retrieval over Dynamic Corpora

    [https://arxiv.org/abs/2507.09924](https://arxiv.org/abs/2507.09924)

    该论文提出MixLoRA-DSI框架，通过OOD驱动的动态扩展策略和混合LoRA专家，实现参数次线性增长的持续生成式检索，显著降低训练成本并超越全模型更新基线。

    

    arXiv:2507.09924v2 公告类型：替换-交叉 摘要：在生成式检索中，持续更新基于模型的索引以纳入新文档仍具挑战性，因为全面重新训练计算成本高昂，且在资源受限情况下不切实际。我们提出MixLoRA-DSI，一种新颖框架，结合了可扩展的低秩适配专家混合体与基于层外分布（OOD）驱动的扩展策略。与为每个新语料库分配新专家不同，我们提出的扩展策略通过仅在检测到大量OOD文档时选择性引入新专家，实现了参数次线性增长。在NQ320k和MS MARCO Passage上的实验表明，MixLoRA-DSI优于全模型更新基线，且参数开销极小，训练成本大幅降低。

    arXiv:2507.09924v2 Announce Type: replace-cross  Abstract: Continually updating model-based indexes in generative retrieval with new documents remains challenging, as full retraining is computationally expensive and impractical under resource constraints. We propose MixLoRA-DSI, a novel framework that combines an expandable mixture of Low-Rank Adaptation experts with a layer-wise out-of-distribution (OOD)-driven expansion strategy. Instead of allocating new experts for each new corpus, our proposed expansion strategy enables sublinear parameter growth by selectively introducing new experts only when significant number of OOD documents are detected. Experiments on NQ320k and MS MARCO Passage demonstrate that MixLoRA-DSI outperforms full-model update baselines, with minimal parameter overhead and substantially lower training costs.
    
[^314]: 一种集成时空模型与大语言模型的模块化多任务推理框架

    A Modular Multitask Reasoning Framework Integrating Spatio-temporal Models and LLMs

    [https://arxiv.org/abs/2506.20073](https://arxiv.org/abs/2506.20073)

    本文提出STReason框架，通过上下文学习将复杂自然语言查询分解为模块化程序，结合LLM推理与时空模型分析能力，实现无需微调的多任务推理和可解释输出。

    

    时空数据挖掘在多个领域的明智决策中发挥着关键作用。然而，现有模型通常局限于狭窄的任务，缺乏多任务推理和复杂长形式推理的能力，这些推理需要生成深入、可解释的输出。这些局限性限制了它们在真实世界、多层面决策场景中的适用性。在本工作中，我们引入了STReason，一种新颖的框架，它将大语言模型（LLMs）的推理优势与时空模型的分析能力相结合，用于多任务推理和执行。无需针对特定任务的微调，STReason利用上下文学习将复杂的自然语言查询分解为模块化、可解释的程序，然后系统地执行这些程序以生成数值解和详细的推理依据。通过将所有解释基于可验证的计算，该框架确保了推理的可靠性和可解释性。

    arXiv:2506.20073v2 Announce Type: replace-cross  Abstract: Spatio-temporal data mining plays a pivotal role in informed decision making across diverse domains. However, existing models are often restricted to narrow tasks, lacking the capacity for multi-task inference and complex long-form reasoning that requires generation of in-depth, explanatory outputs. These limitations restrict their applicability to real-world, multi-faceted decision scenarios. In this work, we introduce STReason, a novel framework that integrates the reasoning strengths of large language models (LLMs) with the analytical capabilities of spatio-temporal models for multi-task inference and execution. Without task-specific fine-tuning, STReason leverages in-context learning to decompose complex natural language queries into modular, interpretable programs, which are then systematically executed to generate both numerical solutions and detailed reasoning rationales. By grounding all explanations in verified computa
    
[^315]: 从识别到推理：通过思维链对齐推进多模态有害模因检测

    From Recognition to Reasoning: Advancing Multimodal Harmful Meme Detection via Chain-of-Thought Alignment

    [https://arxiv.org/abs/2506.18919](https://arxiv.org/abs/2506.18919)

    本文构建了MemeMind大规模有害模因数据集，结合严格分类体系和思维链注释，以提升对隐含风险与细粒度语义的识别与推理能力。

    

    arXiv:2506.18919v5 公告类型：交叉替换 摘要：作为一种融合图像和文本的多模态传播媒介，模因常通过隐喻、讽刺和幽默来传递隐含的有害内容，使得有害模因检测成为一项复杂且具有挑战性的任务。尽管近期研究在检测准确性和模型可解释性方面取得了显著进展，但大规模、高质量的有害模因数据集仍然稀缺。此外，现有方法在识别隐含风险和理解细粒度语义方面仍表现出明显局限性。为解决这些挑战，我们构建了MemeMind，一个用于有害模因检测的大规模数据集。MemeMind包含广泛收集的公开模因，并采用根据广泛认可的国际标准和当代互联网语境制定的严格且全面的有害内容分类体系。此外，该数据集提供了详细的结构化思维链（CoT）

    arXiv:2506.18919v5 Announce Type: replace-cross  Abstract: As a multimodal communication medium that integrates images and text, memes often convey implicit harmful content through metaphors, satire, and humor, making harmful meme detection a complex and challenging task. Although recent studies have achieved considerable progress in detection accuracy and model interpretability, large-scale, high-quality datasets for harmful memes remain scarce. Moreover, existing methods still exhibit notable limitations in identifying implicit risks and understanding fine-grained semantics. To address these challenges, we construct MemeMind, a large-scale dataset for harmful meme detection. MemeMind comprises a broad collection of publicly available memes and adopts a rigorous and comprehensive taxonomy of harmful content developed in accordance with widely recognized international standards and contemporary Internet contexts. In addition, the dataset provides detailed structured Chain-of-Thought (C
    
[^316]: 安全对齐的权重并不足够：拒绝教师引导的微调在有害微调攻击下增强安全性与下游性能

    Safety-Aligned Weights Are Not Enough: Refusal-Teacher-Guided Finetuning Enhances Safety and Downstream Performance under Harmful Finetuning Attacks

    [https://arxiv.org/abs/2506.07356](https://arxiv.org/abs/2506.07356)

    本文提出了一种拒绝教师引导的微调框架，通过直接微调基础模型而非安全对齐权重，在有害微调攻击下同时提升安全性和下游任务性能。

    

    arXiv:2506.07356v3 公告类型：替换 摘要：虽然微调即服务（FaaS）允许使用用户数据定制大型语言模型（LLM），但当用户数据包含有害提示时，该服务容易受到安全性退化影响，这一威胁被称为有害微调攻击。为防御此攻击，先前工作首先构建安全对齐的LLM，然后在用户数据上微调该LLM。然而，我们观察到安全对齐的权重为下游任务学习提供了较弱的初始化，导致安全性和实用性欠佳。基于此局限性，我们将安全的FaaS微调范式从微调安全对齐权重转变为在明确的安全教师指导下微调基础权重。具体而言，我们提出了一种拒绝教师（Ref-Teacher）引导的微调框架。我们的方法直接在安全对齐的Ref-Teacher指导下微调基础LLM，该教师从用户数据中过滤有害提示，并将安全性提炼到基础模型中。

    arXiv:2506.07356v3 Announce Type: replace  Abstract: While Finetuning-as-a-Service (FaaS) enables customization of Large Language Models (LLMs) using user data, this service is vulnerable to safety degradation when user data includes harmful prompts, a threat known as harmful finetuning attacks. To defend against this, prior work first constructs safety-aligned LLM and then finetunes the LLM on user data. However, we observe that the safety-aligned weights provide weak initialization for downstream task learning, leading to suboptimal safety and utility. Motivated by this limitation, we shift the safe FaaS finetuning paradigm from finetuning safety-aligned weights to finetuning base weights under explicit safety-teacher guidance. Specifically, we propose a Refusal-Teacher (Ref-Teacher)-guided finetuning framework. Our approach directly finetunes the base LLM under the guidance of a safety-aligned Ref-Teacher, which filters harmful prompts from user data and distills safety into the bas
    
[^317]: 扩展电子健康记录基础模型以支持人群健康管理

    Scaling Electronic Health Record Foundation Models for Population Health Management

    [https://arxiv.org/abs/2506.00209](https://arxiv.org/abs/2506.00209)

    本文提出了一种跨机构、可扩展的电子健康记录基础模型，通过统一代码对齐和计算最优训练，在超过500万患者数据上实现了大规模慢性病预测，显著提升了人群健康管理的效率。

    

    人群健康管理需要可扩展的方法来识别患有心血管疾病和癌症等慢性病风险个体，然而现有方法依赖于碎片化数据和资源密集型的筛查。我们提出了“扩展电子健康记录基础模型以支持人群健康管理”，这是一种电子健康记录基础模型，利用跨机构纵向医疗记录进行大规模慢性病预测。我们在来自台湾和美国的超过500万患者的数十亿医疗事件上对该模型进行了预训练，利用统一的代码对齐框架解决跨系统异质性，并通过IsoFLOP分析刻画其扩展行为，训练了计算最优的模型，参数规模最高达24亿。在11项慢性病预测任务中，该模型表现出卓越性能。

    arXiv:2506.00209v3 Announce Type: replace-cross  Abstract: Population health management requires scalable methods to identify individuals at risk of chronic diseases such as cardiovascular conditions and cancer, yet existing approaches rely on fragmented data and resource-intensive screening. We present Scaling Electronic Health Record Foundation Models for Population Health Management, an Electronic Health Record Foundation Model that performs large-scale chronic disease prediction using cross-site longitudinal medical records. We pretrain Scaling Electronic Health Record Foundation Models for Population Health Management on billions of medical events from over 5 million patients across Taiwan and the United States, leveraging a unified code alignment framework to address cross-system heterogeneity, and characterize its scaling behavior via IsoFLOP analysis, training compute-optimal models up to 2.4B parameters. Across 11 chronic disease prediction tasks, Scaling Electronic Health Rec
    
[^318]: 心智理论与亲社会信念对大语言模型在最后通牒博弈中引导类人行为的影响

    Effects of Theory of Mind and Prosocial Beliefs on Steering Human-Aligned Behaviors of LLMs in Ultimatum Games

    [https://arxiv.org/abs/2505.24255](https://arxiv.org/abs/2505.24255)

    本论文通过2700次模拟实验证明，在最后通牒博弈中，心智理论推理能显著增强大语言模型行为与人类规范的对齐、决策一致性和谈判结果，且优于单纯推理模型的表现。

    

    大语言模型（LLMs）在模拟人类行为和进行心智理论（ToM）推理方面展现出潜力，这对于复杂的社会互动至关重要。我们以最后通牒博弈为参考任务，研究了ToM推理在将代理行为与人类规范对齐中的作用。我们为LLM代理初始化了不同的亲社会信念（贪婪、公平、无私）和推理方法（思维链及不同层次的ToM推理），并考察了它们在多种LLM（包括推理模型如o3-mini和DeepSeek-R1 Distilled Qwen 32B）中的决策过程和结果。我们进行了2700次模拟，结果表明ToM推理增强了行为与人类规范的对齐、决策一致性以及谈判结果。与先前发现一致，推理型LLM相比增强ToM的LLM表现出有限的能力，不同的博弈角色受益于不同的方法。

    arXiv:2505.24255v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have shown potential in simulating human behaviors and performing theory-of-mind (ToM) reasoning, crucial for complex social interactions. We investigate ToM reasoning's role in aligning agentic behaviors with human norms in negotiation tasks, using the ultimatum game as our referenced task. We initialized LLM agents with different prosocial beliefs (Greedy, Fair, Selfless) and reasoning methods (chain of thought and ToM reasoning of varying levels), examining their decision-making process and outcome across multiple LLMs, including reasoning models like o3-mini and DeepSeek-R1 Distilled Qwen 32B. We perform 2,700 simulations to show that ToM reasoning enhances behavioral alignment with human, decision-making consistency, and negotiation outcomes. Consistent with prior findings, reasoning LLMs exhibit limited capability compared to ToM-enhanced LLMs, with different game roles benefiting from differe
    
[^319]: 推理遇见个性化：释放大型推理模型在个性化生成中的潜力

    Reasoning Meets Personalization: Unleashing the Potential of Large Reasoning Model for Personalized Generation

    [https://arxiv.org/abs/2505.17571](https://arxiv.org/abs/2505.17571)

    本文首次系统评估了大型推理模型在个性化任务中的表现，发现其并不总是优于通用LLM，并针对发散思维、格式错配和检索利用不足等问题，提出了强化推理框架（\model）来提升个性化生成效果。

    

    个性化是现代智能系统中的关键任务，其应用跨越多个领域，包括与大型语言模型（LLM）的交互。近期推理能力的进展显著增强了LLM，使其在数学和编码等任务中表现出前所未有的性能。然而，它们在个性化任务中的潜力仍未得到充分探索。在本文中，我们首次对大型推理模型（LRM）在个性化任务中的表现进行了系统评估。令人惊讶的是，尽管生成了更多令牌，LRM并不始终优于通用LLM，尤其是在检索密集型场景中，其优势会减弱。我们的分析识别出三个关键局限：发散性思维、响应格式不匹配以及检索信息使用效率低下。为应对这些挑战，我们提出了用于个性化的强化推理框架（\model），这是一个新颖的框架。

    arXiv:2505.17571v2 Announce Type: replace  Abstract: Personalization is a critical task in modern intelligent systems, with applications spanning diverse domains, including interactions with large language models (LLMs). Recent advances in reasoning capabilities have significantly enhanced LLMs, enabling unprecedented performance in tasks such as mathematics and coding. However, their potential for personalization tasks remains underexplored.   In this paper, we present the first systematic evaluation of large reasoning models (LRMs) for personalization tasks. Surprisingly, despite generating more tokens, LRMs do not consistently outperform general-purpose LLMs, especially in retrieval-intensive scenarios where their advantages diminish. Our analysis identifies three key limitations: divergent thinking, misalignment of response formats, and ineffective use of retrieved information. To address these challenges, we propose Reinforced Reasoning for Personalization (\model), a novel framew
    
[^320]: ClinicalGPT-R1：提升大语言模型在通科疾病诊断中的推理能力

    ClinicalGPT-R1: Pushing reasoning capability of generalist disease diagnosis with large language model

    [https://arxiv.org/abs/2504.09421](https://arxiv.org/abs/2504.09421)

    ClinicalGPT-R1通过大规模真实临床数据训练和多种推理增强策略，在中文诊断任务中超越GPT-4o，并在英文任务中与GPT-4持平，为通科疾病诊断提供了高性能的推理增强大语言模型。

    

    arXiv:2504.09421v3 公告类型：交叉替换 摘要：近期大语言模型（LLMs）在推理方面的进展已在数学和编程等领域展现出显著的推理能力，但其在临床诊断中的应用仍未得到充分探索。在此，我们介绍了ClinicalGPT-R1，一种用于疾病诊断的推理增强型通科大语言模型。该模型基于20,000份真实世界临床记录的数据集进行训练，利用多种训练策略来增强诊断推理能力。为了评估性能，我们构建了MedBench-Hard，一个涵盖七个主要医学专科和代表性疾病的具有挑战性的数据集。实验结果表明，ClinicalGPT-R1在中文诊断任务中优于GPT-4o，并在英文环境中达到与GPT-4相当的性能。这项比较研究有效验证了ClinicalGPT-R1在疾病诊断任务中的优越性能。资源可在https://github.获取。

    arXiv:2504.09421v3 Announce Type: replace-cross  Abstract: Recent advances in reasoning with large language models (LLMs)has shown remarkable reasoning capabilities in domains such as mathematics and coding, yet their application to clinical diagnosis remains underexplored. Here, we introduce ClinicalGPT-R1, a reasoning enhanced generalist large language model for disease diagnosis. Trained on a dataset of 20,000 real-world clinical records, ClinicalGPT-R1 leverages diverse training strategies to enhance diagnostic reasoning. To benchmark performance, we curated MedBench-Hard, a challenging dataset spanning seven major medical specialties and representative diseases. Experimental results demonstrate that ClinicalGPT-R1 outperforms GPT-4o in Chinese diagnostic tasks and achieves comparable performance to GPT-4 in English settings. This comparative study effectively validates the superior performance of ClinicalGPT-R1 in disease diagnosis tasks. Resources are available at https://github.
    
[^321]: 人工智能大学：一种由大语言模型驱动的工程学习助手——以有限元方法为案例研究

    AI University: An LLM-Powered Learning Assistant for Engineering---A Finite Element Method Case Study

    [https://arxiv.org/abs/2504.08846](https://arxiv.org/abs/2504.08846)

    本文提出了一种名为AI-U的框架，通过微调大语言模型并结合检索增强生成，实现了与课程风格一致的学习助手，并以有限元方法课程验证了其有效性。

    

    我们介绍了人工智能大学（AI-U），这是一个灵活的框架，用于人工智能驱动的课程内容交付，能够适应课程的教学风格。AI-U结合了微调的大语言模型（LLM）、检索增强生成（RAG）和推理合成模型，从讲座视频、笔记和教科书中生成风格一致的回答。以研究生级别的有限元方法（FEM）课程作为案例研究，我们提出了一个流程，用于合成基于课程的训练数据、使用低秩适配（LoRA）微调开源大语言模型，并应用基于RAG的合成。我们的评估——结合余弦相似度、基于大语言模型的评估、专家评审和用户研究——显示与基础模型相比，与课程材料的对齐度有所提高。我们还开发了一个原型网页应用，可在https://my-ai-university.com访问，该应用通过引用课程相关部分来增强人工智能生成的回答。

    arXiv:2504.08846v2 Announce Type: replace-cross  Abstract: We introduce AI University (AI-U), a flexible framework for AI-driven course content delivery that adapts to a course's instructional style. AI-U combines a fine-tuned large language model (LLM) with retrieval-augmented generation (RAG) and a reasoning synthesis model to generate style-aligned responses from lecture videos, notes, and textbooks. Using a graduate-level finite-element-method (FEM) course as a case study, we present a pipeline to synthesize course-grounded training data, fine-tune an open-source LLM with Low-Rank Adaptation (LoRA), and apply RAG-based synthesis. Our evaluation---combining cosine similarity, LLM-based assessment, expert review, and user studies---shows improved alignment with course materials relative to the base model. We have also developed a prototype web application, available at https://my-ai-university.com, that enhances AI-generated responses with references to relevant sections of the cours
    
[^322]: 释放大语言模型在稠密检索中的潜力：基于查询似然建模

    Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling

    [https://arxiv.org/abs/2504.05216](https://arxiv.org/abs/2504.05216)

    本文提出LLM-QL模型，通过辅助的查询似然最大化任务增强大语言模型的稠密检索能力，利用生成优势改进对比学习。

    

    稠密检索是信息检索（IR）中的关键任务，为后续的重新排序和增强生成等下游任务提供基础。近年来，大语言模型（LLMs）展现了令人印象深刻的语义理解能力，使其成为稠密检索研究者的关注焦点。尽管LLMs作为解码器风格的生成模型在语言生成方面表现出色，但由于缺乏对后续标记的关注，它们往往在建模全局信息方面有所不足。受经典基于词的语言建模方法在IR中的启发，特别是查询似然（QL）模型，我们旨在通过QL最大化来利用LLMs的生成优势。我们不采用QL估计来进行文档排序，而是提出一个辅助任务——QL最大化，以增强骨干网络，用于后续的检索器对比学习。我们介绍了我们的模型LLM-QL，它整合了...

    arXiv:2504.05216v4 Announce Type: replace-cross  Abstract: Dense retrieval is a crucial task in Information Retrieval (IR), serving as the basis for downstream tasks such as re-ranking and augmenting generation. Recently, large language models (LLMs) have demonstrated impressive semantic understanding capabilities, making them attractive to researchers focusing on dense retrieval. While LLMs, as decoder-style generative models, excel in language generation, they often fall short in modeling global information due to a lack of attention to subsequent tokens. Drawing inspiration from the classical word-based language modeling approach for IR, specifically the query likelihood (QL) model, we aim to leverage the generative strengths of LLMs through QL maximization. Rather than employing QL estimation for document ranking, we propose an auxiliary task of QL maximization to enhance the backbone for subsequent contrastive learning of the retriever. We introduce our model, LLM-QL, which incorp
    
[^323]: 通过以OCR为中心的强化学习来评估和提升大型视觉语言模型的多语言能力

    Benchmarking and Boosting Multilingual Capabilities of LVLMs via OCR-Centric Reinforcement Learning

    [https://arxiv.org/abs/2503.18484](https://arxiv.org/abs/2503.18484)

    本文提出了首个基于严格平行语料库的多语言多模态基准PM4Bench，并发现OCR是跨语言性能差异的关键因素，进而设计了以OCR为中心的强化学习训练策略来提升LVLMs的多语言能力。

    

    arXiv:2503.18484v3 公告类型：替换-交叉 摘要：评估大型视觉语言模型（LVLMs）的多语言能力仍然具有挑战性，因为大多数基准测试依赖于非平行语料库，这使得跨语言性能差距是否反映模型局限性或数据集不一致性变得不明确。为了解决这一问题，我们引入了PM4Bench，这是第一个基于严格平行10语言语料库构建的多模态、多语言、多任务基准，从而能够对模型性能进行公平的、苹果对苹果的跨语言比较。我们进一步引入了一种视觉设置，将文本输入直接嵌入图像中，更好地近似于LVLM驱动的代理通过统一视觉观察与虚拟或物理环境交互的部署场景。对10个LVLMs的实验表明，当文本内容以视觉方式呈现时，OCR是导致跨语言差异的关键因素。基于此，我们设计了一种以OCR为中心的GRPO训练策略，使用完全合成的数据。

    arXiv:2503.18484v3 Announce Type: replace-cross  Abstract: Evaluating the multilingual capabilities of Large Vision-Language Models (LVLMs) remains challenging because most benchmarks rely on non-parallel corpora, making it unclear whether cross-lingual performance gaps reflect model limitations or dataset inconsistencies. To address this, we introduce PM4Bench, the first multimodal, multilingual, multi-task benchmark built on a strictly parallel 10-language corpus, enabling fair, apples-to-apples cross-lingual comparison of model performance. We further introduce a vision setting that embeds textual inputs directly into images, better approximating deployment scenarios where LVLM-driven agents interact with virtual or physical environments through unified visual observations. Experiments with 10 LVLMs reveal that OCR is a key factor behind cross-lingual disparity when textual content is rendered visually. Motivated by this, we design an OCR-centric GRPO training strategy using fully s
    
[^324]: 语言模型的深度对比遗忘学习

    Deep Contrastive Unlearning for Language Models

    [https://arxiv.org/abs/2503.14900](https://arxiv.org/abs/2503.14900)

    本文提出了一种深度对比遗忘方法，通过显式考虑模型输出空间的几何结构，在移除特定训练样本信息的同时保持模型性能，以应对语言模型黑箱性带来的遗忘挑战。

    

    arXiv:2503.14900v2 公告类型：交叉替换 摘要：过去几年见证了大型语言模型的巨大成功，展示了其在理解文本数据和生成类人语言方面的强大能力。大型语言模型通过在大量文本数据上训练而取得成功，这些数据包括含有版权内容的在线资源和用户生成的知识。然而，这带来了代价：可能暴露用户隐私和违反版权保护的风险。因此，为保障个人的“被遗忘权”，机器遗忘——即从模型中移除特定训练样本携带的信息，同时不降低其预测质量的过程——引起了越来越多的关注。由于语言模型的黑箱特性，这是一项具有挑战性的任务。大多数现有研究集中于减轻那些遗忘样本对模型输出的影响，并未明确考虑几何维度的差异。

    arXiv:2503.14900v2 Announce Type: replace-cross  Abstract: The past a few years have witnessed the great success of large language models, demonstrating powerful capabilities in comprehending textual data and generating human-like languages. Large language models achieve success by being trained on vast amounts of textual data, including online sources with copyrighted content and user-generated knowledge. However, this comes at a cost: the potential risk of exposing users' privacy and violating copyright protections. Thus, to safeguard individuals' "right to be forgotten", there has been increasing interests in machine unlearning -- the process of removing information carried by particular training samples from a model while not deteriorating its predictive quality. This is a challenging task due to the black-box nature of language models. Most existing studies focus on mitigating the impact of those forgot samples upon a model's outputs, and do not explicitly consider the geometric d
    
[^325]: 基于长度控制边际的偏好优化，无需参考模型

    Length-Controlled Margin-Based Preference Optimization without Reference Model

    [https://arxiv.org/abs/2502.14643](https://arxiv.org/abs/2502.14643)

    提出了一种无需参考模型、基于长度控制边际的偏好优化方法（LMPO），通过均匀参考模型和平均对数概率策略，有效解决了DPO的长度偏差、内存低效和概率退化问题。

    

    直接偏好优化（DPO）是一种广泛采用的离线算法，用于基于人类反馈的偏好强化学习（RLHF），旨在通过重新定义奖励函数来提高训练简单性和稳定性。然而，DPO 受到若干限制的阻碍，包括长度偏差、内存效率低下和概率退化。为解决这些挑战，我们提出了长度控制边际偏好优化（LMPO），一种更高效且稳健的替代方案。LMPO 引入一个均匀参考模型作为 DPO 损失的上界，从而更精确地逼近原始优化目标。此外，采用平均对数概率优化策略，以最小化训练和推理阶段之间的差异。LMPO 的一个关键创新在于其长度控制边际损失函数，集成在 Bradley-Terry 框架内。该损失函数重新……

    arXiv:2502.14643v3 Announce Type: replace  Abstract: Direct Preference Optimization (DPO) is a widely adopted offline algorithm for preference-based reinforcement learning from human feedback (RLHF), designed to improve training simplicity and stability by redefining reward functions. However, DPO is hindered by several limitations, including length bias, memory inefficiency, and probability degradation. To address these challenges, we propose Length-Controlled Margin-Based Preference Optimization (LMPO), a more efficient and robust alternative. LMPO introduces a uniform reference model as an upper bound for the DPO loss, enabling a more accurate approximation of the original optimization objective. Additionally, an average log-probability optimization strategy is employed to minimize discrepancies between training and inference phases. A key innovation of LMPO lies in its Length-Controlled Margin-Based loss function, integrated within the Bradley-Terry framework. This loss function re
    
[^326]: 迈向更安全的社交媒体平台：使用大型语言模型进行可扩展且高性能的小样本有害内容审核

    Towards Safer Social Media Platforms: Scalable and Performant Few-Shot Harmful Content Moderation Using Large Language Models

    [https://arxiv.org/abs/2501.13976](https://arxiv.org/abs/2501.13976)

    本文提出利用大型语言模型结合上下文学习的小样本方法，在有害内容审核中超越现有基线，并通过多模态技术进一步提升性能。

    

    社交媒体平台上有害内容的普遍存在对用户和社会构成重大风险，因此需要更有效且可扩展的内容审核策略。当前方法依赖人工审核员、监督分类器和大量训练数据，但往往在可扩展性、主观性以及有害内容的动态性（例如暴力内容、危险挑战趋势等）方面存在不足。为弥补这些差距，我们利用大型语言模型（LLMs）通过上下文学习进行小样本动态内容审核。通过对多个LLM的广泛实验，我们证明，在识别危害方面，我们的小样本方法能够超越现有的专有基线（Perspective和OpenAI Moderation）以及先前最先进的小样本学习方法。我们还整合了视觉信息（视频缩略图），并评估了不同的多模态技术是否改善模型性能。

    arXiv:2501.13976v2 Announce Type: replace-cross  Abstract: The prevalence of harmful content on social media platforms poses significant risks to users and society, necessitating more effective and scalable content moderation strategies. Current approaches rely on human moderators, supervised classifiers, and large volumes of training data, and often struggle with scalability, subjectivity, and the dynamic nature of harmful content (e.g., violent content, dangerous challenge trends, etc.). To bridge these gaps, we utilize Large Language Models (LLMs) to undertake few-shot dynamic content moderation via in-context learning. Through extensive experiments on multiple LLMs, we demonstrate that our few-shot approaches can outperform existing proprietary baselines (Perspective and OpenAI Moderation) as well as prior state-of-the-art few-shot learning methods, in identifying harm. We also incorporate visual information (video thumbnails) and assess if different multimodal techniques improve m
    
[^327]: 训练大型语言模型在连续潜在空间中进行推理

    Training Large Language Models to Reason in a Continuous Latent Space

    [https://arxiv.org/abs/2412.06769](https://arxiv.org/abs/2412.06769)

    本文提出Coconut范式，通过直接使用LLM的连续潜在状态作为推理输入，突破了传统语言空间推理的局限，使模型能更高效地编码和探索多种推理路径。

    

    arXiv:2412.06769v4 公告类型：替换 摘要：大型语言模型（LLMs）通常被限制在语言空间中进行推理，通过思维链（CoT）表达推理过程以解决复杂问题。然而，语言空间可能并非总是最优的推理媒介。大多数单词标记主要确保文本连贯性，对推理并非必需，而一些关键标记则需要复杂规划，给LLMs带来挑战。为了探索超越语言的推理潜力，我们引入了一种新范式，称为Coconut（连续思维链）。Coconut利用LLM的最后一个隐藏状态作为推理状态的表示，称为“连续思维”。我们不将此状态解码为单词，而是直接将其作为下一个输入嵌入在连续空间中反馈给模型。这种潜在推理范式实现了先进的推理模式，其中连续思维可以编码多个备选的下一步。

    arXiv:2412.06769v4 Announce Type: replace  Abstract: Large language models (LLMs) are typically constrained to reason in the language space, where they express the reasoning process through a chain-of-thought (CoT) to solve complex problems. However, the language space may not always be optimal for reasoning. Most word tokens primarily ensure textual coherence and are not essential for reasoning, while some critical tokens require complex planning and pose challenges to LLMs. To explore the potential of reasoning beyond language, we introduce a new paradigm called Coconut (Chain of Continuous Thought). Coconut utilizes the last hidden state of the LLM as a representation of the reasoning state, termed "continuous thought." Instead of decoding this state into words, we feed it back to the model as the next input embedding directly in the continuous space. This latent reasoning paradigm enables an advanced reasoning pattern, where continuous thoughts can encode multiple alternative next 
    
[^328]: 基于邻域感知语义对齐与时间调制的LLM时间序列预测框架（NEST）

    NeST: Neighborhood-aware semantic alignment and temporal modulation for LLM based time series forecasting

    [https://arxiv.org/abs/2412.04806](https://arxiv.org/abs/2412.04806)

    本论文提出NEST框架，通过邻域感知语义对齐和时间调制，有效整合文本与时间序列信息，改进了LLM在时间序列预测中的性能。

    

    将训练于离散文本数据的大型语言模型（LLMs）适应于连续时间序列信号的预测具有挑战性。虽然微调LLMs可实现这种适应，但有效整合提示中的文本和时间序列信息至关重要。当前基于LLM的时间序列预测方法通过简单拼接或参数密集的交叉注意力结合两种模态。此外，现有方法使用分解技术嵌入时间序列数据，这可能无法充分捕捉复杂的时序动态。为解决这些局限性，我们提出了基于邻域感知语义对齐和时间调制的框架（NEST），以构建新的文本集成时间序列提示来微调LLM。首先，我们生成邻域感知的文本原型，这些原型被优化以代表LLM预训练词元嵌入的局部邻域。其次，我们将它们与时间序列进行对齐。

    arXiv:2412.04806v2 Announce Type: replace-cross  Abstract: Adapting Large Language Models (LLMs) trained on discrete text data, to forecast continuous time series signals is challenging. While finetuning the LLMs enables such adaptation, effectively integrating both textual and time series information in the prompt is critical. Current LLM-based time series forecasting methods combine the two modalities through simple concatenation or parameter heavy cross-attention. Moreover, existing methods embed time series data using decomposition techniques that may inadequately capture complex temporal dynamics. To address these limitations, we propose neighborhood-aware semantic alignment and temporal modulation based framework (NEST) to formulate a new text-integrated time series prompt to finetune the LLM. First, we generate neighborhood-aware text prototypes that are optimized to represent local neighborhoods of pretrained word token embeddings of the LLM. Second, we align them with temporal
    
[^329]: 桥接语言结构与机制可解释性：语言模型中的概念解释

    Bridging Linguistic Structure and Mechanistic Interpretability for Conceptual Interpretation in Language Models

    [https://arxiv.org/abs/2408.11827](https://arxiv.org/abs/2408.11827)

    本文提出DSRA方法，通过将定义性语义角色融入因果追踪，首次系统性地将语言结构桥接到机制可解释性，用于解释语言模型中的概念组合映射。

    

    arXiv:2408.11827v2 公告类型：替换 摘要：理解语言模型如何从语言输入中组合意义，仍是可解释性研究的核心问题。机制研究已将功能角色归因于核心Transformer组件；然而，这些发现主要源于事实检索设置。相同的机制是否支持“概念解释”，即从定义性表达到抽象意义的组合映射，仍未得到充分表征。我们引入了“DSRA”（定义性语义角色分析），一种在反向词典任务中应用因果追踪的方法，并用基于论元结构理论的DSRs（定义性语义角色）增强恢复痕迹。这种语言覆盖层识别了哪些组合功能（如属类、种差性质）与高恢复状态相关，将激活修补扩展到标记级定位之外。应用于GPT-J模型。

    arXiv:2408.11827v2 Announce Type: replace  Abstract: Understanding how language models compose meaning from linguistic input remains a central problem in interpretability research. Mechanistic studies have attributed functional roles to core transformer components; however, these findings derive largely from factual retrieval settings. Whether the same mechanisms support \textit{conceptual interpretation}, the compositional mapping from definitional expressions to abstract meaning, remains insufficiently characterised. We introduce \textit{DSRA} (Definitional Semantic Role Analysis), a methodology that applies causal tracing within the reverse dictionary task and augments restoration traces with definitional semantic roles (DSRs) grounded in Argument Structure Theory. This linguistic overlay identifies which compositional functions (e.g., genus, differentia quality) are associated with high-recovery states, extending activation patching beyond token-level localisation. Applied to GPT-J
    
[^330]: 人工智能与大型预训练模型合作调查

    A Survey on Human-AI Teaming with Large Pre-Trained Models

    [https://arxiv.org/abs/2403.04931](https://arxiv.org/abs/2403.04931)

    本文调查了大型预训练模型与人工智能合作的重要性，强调了这些模型如何超越传统方法增强协作智能，并探讨了其在增强人类能力、改善AI模型、有效团队合作、道德考虑以及在各个领域广泛应用方面的潜在作用。

    

    在人工智能（AI）迅速发展的景观中，人类智能和AI系统之间的协作，即人工智能（HAI）合作，已成为推进问题解决和决策过程的基石。大型预训练模型（LPtM）的出现显著改变了这一景观，通过利用大量数据来理解和预测复杂模式，为人类提供了前所未有的能力。本文调查了LPtMs与HAI的关键整合，强调了这些模型如何超越传统方法增强协作智能。重点探讨了LPtMs在增强人类能力方面的协同潜力，讨论了这种协作对AI模型改进、有效的团队合作、道德考虑以及在各个领域的广泛应用影响。通过这一探索，研究揭示了LPtM增强HAI的变革性影响。

    arXiv:2403.04931v1 Announce Type: new  Abstract: In the rapidly evolving landscape of artificial intelligence (AI), the collaboration between human intelligence and AI systems, known as Human-AI (HAI) Teaming, has emerged as a cornerstone for advancing problem-solving and decision-making processes. The advent of Large Pre-trained Models (LPtM) has significantly transformed this landscape, offering unprecedented capabilities by leveraging vast amounts of data to understand and predict complex patterns. This paper surveys the pivotal integration of LPtMs with HAI, emphasizing how these models enhance collaborative intelligence beyond traditional approaches. It examines the synergistic potential of LPtMs in augmenting human capabilities, discussing this collaboration for AI model improvements, effective teaming, ethical considerations, and their broad applied implications in various sectors. Through this exploration, the study sheds light on the transformative impact of LPtM-enhanced HAI 
    
[^331]: 以人格驱动生成式智能体

    Driving Generative Agents With Their Personality

    [https://arxiv.org/abs/2402.14879](https://arxiv.org/abs/2402.14879)

    大型语言模型（LLMs）利用心理测量值，在视频游戏角色开发中代表给定的人格特征，增强游戏角色的类人特性。

    

    本研究探讨了大型语言模型（LLMs）利用心理测量值，特别是人格信息，在视频游戏角色开发背景下的潜力。情感计算（AC）系统量化了非玩家角色（NPC）的心理，LLM可以利用该系统的信息，使用值进行提示生成。研究表明，LLM可以始终代表给定的人格特征，从而增强游戏角色的类人特性。将人类检查重新用于评估LLM的国际人格项目池（IPIP）问卷表明，该模型能够准确生成与所提供人格相关的内容。结果显示，LLM的改进，如最新的GPT-4模型，可以始终利用和解释人格以代表行为。

    arXiv:2402.14879v1 Announce Type: cross  Abstract: This research explores the potential of Large Language Models (LLMs) to utilize psychometric values, specifically personality information, within the context of video game character development. Affective Computing (AC) systems quantify a Non-Player character's (NPC) psyche, and an LLM can take advantage of the system's information by using the values for prompt generation. The research shows an LLM can consistently represent a given personality profile, thereby enhancing the human-like characteristics of game characters. Repurposing a human examination, the International Personality Item Pool (IPIP) questionnaire, to evaluate an LLM shows that the model can accurately generate content concerning the personality provided. Results show that the improvement of LLM, such as the latest GPT-4 model, can consistently utilize and interpret a personality to represent behavior.
    
[^332]: 构建多语言词汇资源：一种基于机器翻译辅助与人工参与的带有多词表达标注的多语言平行语料库

    Towards a resource for multilingual lexicons: an MT assisted and human-in-the-loop multilingual parallel corpus with multi-word expression annotation

    [https://arxiv.org/abs/2011.03783](https://arxiv.org/abs/2011.03783)

    该工作构建了一个结合机器翻译与人工参与的AlphaMWE多语言平行语料库，标注并手动对齐了多种语言的动词性多词表达，并引入了严格的质量控制流程。

    

    在本工作中，我们介绍了构建一种机器翻译辅助且人工参与的多语言平行语料库的过程，该语料库带有对多词表达（MWEs）的标注，命名为AlphaMWE。MWEs包括PARSEME共享任务中定义的以动词为研究术语核心的动词性多词表达（vMWEs）。标注的vMWEs还进行了双语和多语言的人工对齐。覆盖的语言包括阿拉伯语、中文、英语、德语、意大利语和波兰语，其中阿拉伯语语料包含标准语以及来自埃及和突尼斯的方言变体。我们的原始英语语料提取自2018年PARSEME共享任务。我们对源语料进行了机器翻译，随后进行了人工后期编辑和目标MWEs的标注。为了限制错误，实施了严格的质量控制，即每个机器翻译输出句子接受了首次人工后期编辑和标注，再加上第二次人工检查。

    arXiv:2011.03783v3 Announce Type: replace-cross  Abstract: In this work, we introduce the construction of a machine translation (MT) assisted and human-in-the-loop multilingual parallel corpus with annotations of multi-word expressions (MWEs), named AlphaMWE. The MWEs include verbal MWEs (vMWEs) defined in the PARSEME shared task that have a verb as the head of the studied terms. The annotated vMWEs are also bilingually and multilingually aligned manually. The languages covered include Arabic, Chinese, English, German, Italian, and Polish, of which, the Arabic corpus includes both standard and dialectal variations from Egypt and Tunisia. Our original English corpus is extracted from the PARSEME shared task in 2018. We performed machine translation of this source corpus followed by human post-editing and annotation of target MWEs. Strict quality control was applied for error limitation, i.e., each MT output sentence received first manual post-editing and annotation plus a second manual 
    

